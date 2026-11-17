"""DQN agent supporting Double DQN, Dueling networks and Prioritized Replay."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from cyberdef.networks import make_q_network
from cyberdef.replay import make_replay
from cyberdef.seeding import set_seed


class DQNAgent:
    """DQN agent with optional Double DQN, Dueling and PER."""

    def __init__(
        self,
        config: dict,
        double: bool = False,
        dueling: bool = False,
        per: bool = False,
        seed: int = 0,
    ):
        if not isinstance(config, Mapping):
            raise TypeError("config must be a mapping")

        # Accept either the complete YAML dictionary or its agent section.
        root_config = dict(config)
        env_config = root_config.get("env", {})
        agent_config = root_config.get("agent", root_config)

        if not isinstance(agent_config, Mapping):
            raise TypeError("agent configuration must be a mapping")
        if not isinstance(env_config, Mapping):
            raise TypeError("env configuration must be a mapping")

        cfg = dict(agent_config)
        n_hosts = int(env_config.get("n_hosts", 10))

        self.obs_dim = 4 * n_hosts
        self.n_actions = 3 * n_hosts + 1

        self.gamma = float(cfg.get("gamma", 0.99))
        self.lr = float(cfg.get("lr", 5e-4))
        self.grad_clip_norm = float(cfg.get("grad_clip_norm", 10.0))
        self.batch_size = int(cfg.get("batch_size", 64))
        self.target_update_interval = int(
            cfg.get("target_update_interval", 1000)
        )
        self.hidden_sizes = tuple(cfg.get("hidden_sizes", [128, 128]))

        self.double = bool(double)
        self.dueling = bool(dueling)
        self.per = bool(per)
        self.seed = int(seed)

        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be between 0 and 1")
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.grad_clip_norm <= 0:
            raise ValueError("grad_clip_norm must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.target_update_interval <= 0:
            raise ValueError("target_update_interval must be positive")
        if not self.hidden_sizes or any(int(h) <= 0 for h in self.hidden_sizes):
            raise ValueError("hidden_sizes must contain positive integers")

        # Seed before creating the neural networks.
        set_seed(self.seed)

        self.rng = np.random.default_rng(self.seed)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.online_net = make_q_network(
            self.obs_dim,
            self.n_actions,
            dueling=self.dueling,
            hidden_sizes=self.hidden_sizes,
        ).to(self.device)

        self.target_net = make_q_network(
            self.obs_dim,
            self.n_actions,
            dueling=self.dueling,
            hidden_sizes=self.hidden_sizes,
        ).to(self.device)

        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        for parameter in self.target_net.parameters():
            parameter.requires_grad_(False)

        self.optimizer = torch.optim.Adam(
            self.online_net.parameters(),
            lr=self.lr,
        )

        self.replay = make_replay(
            capacity=int(cfg.get("buffer_size", 50000)),
            obs_dim=self.obs_dim,
            prioritized=self.per,
            alpha=float(cfg.get("per_alpha", 0.6)),
            eps=float(cfg.get("per_eps", 1e-6)),
            seed=self.seed,
        )

        self.update_count = 0

    def act(self, obs: np.ndarray, epsilon: float = 0.0) -> int:
        """Choose an action using epsilon-greedy exploration."""
        epsilon = float(epsilon)
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be between 0 and 1")

        obs = np.asarray(obs, dtype=np.float32)
        if obs.shape != (self.obs_dim,):
            raise ValueError(
                f"Expected observation shape {(self.obs_dim,)}, "
                f"got {obs.shape}"
            )

        # Greedy evaluation must not consume the training exploration RNG.
        if epsilon > 0.0 and self.rng.random() < epsilon:
            return int(self.rng.integers(self.n_actions))

        obs_tensor = torch.as_tensor(
            obs, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        self.online_net.eval()
        with torch.no_grad():
            action = int(self.online_net(obs_tensor).argmax(dim=1).item())
        self.online_net.train()

        return action

    def sync_target(self) -> None:
        """Hard-copy online network parameters into the target network."""
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

    def update(
        self,
        batch: dict,
        weights: np.ndarray,
        env_step: int | None = None,
    ) -> tuple[float, np.ndarray]:
        """Perform one gradient update.

        Returns:
            Scalar Huber loss and signed TD errors as a NumPy array.
            The replay buffer applies absolute values when updating priorities.
        """
        required = {"obs", "actions", "rewards", "next_obs", "terminated"}
        missing = required.difference(batch)
        if missing:
            raise KeyError(f"Missing batch fields: {sorted(missing)}")

        obs = torch.as_tensor(
            batch["obs"], dtype=torch.float32, device=self.device
        )
        actions = torch.as_tensor(
            batch["actions"], dtype=torch.long, device=self.device
        ).reshape(-1)
        rewards = torch.as_tensor(
            batch["rewards"], dtype=torch.float32, device=self.device
        ).reshape(-1)
        next_obs = torch.as_tensor(
            batch["next_obs"], dtype=torch.float32, device=self.device
        )
        terminated = torch.as_tensor(
            batch["terminated"], dtype=torch.float32, device=self.device
        ).reshape(-1)
        importance_weights = torch.as_tensor(
            weights, dtype=torch.float32, device=self.device
        ).reshape(-1)

        batch_size = obs.shape[0]
        if obs.shape != (batch_size, self.obs_dim):
            raise ValueError("Invalid obs batch shape")
        if next_obs.shape != (batch_size, self.obs_dim):
            raise ValueError("Invalid next_obs batch shape")

        for name, tensor in (
            ("actions", actions),
            ("rewards", rewards),
            ("terminated", terminated),
            ("weights", importance_weights),
        ):
            if tensor.shape != (batch_size,):
                raise ValueError(f"Invalid {name} batch shape")

        if torch.any(actions < 0) or torch.any(actions >= self.n_actions):
            raise ValueError("Batch contains an invalid action")
        if torch.any((terminated != 0) & (terminated != 1)):
            raise ValueError("terminated must contain only 0 or 1")
        if torch.any(importance_weights < 0):
            raise ValueError("importance weights must be non-negative")

        self.online_net.train()

        # Q(s, a) predicted by the online network.
        current_q = self.online_net(obs).gather(
            1, actions.unsqueeze(1)
        ).squeeze(1)

        # Target computation does not participate in backpropagation.
        with torch.no_grad():
            if self.double:
                # Double DQN: online network selects, target network evaluates.
                next_actions = self.online_net(next_obs).argmax(
                    dim=1, keepdim=True
                )
                next_q = self.target_net(next_obs).gather(
                    1, next_actions
                ).squeeze(1)
            else:
                next_q = self.target_net(next_obs).max(dim=1).values

            # A truncation is not stored as terminated, so it still bootstraps.
            targets = rewards + self.gamma * (1.0 - terminated) * next_q

        td_errors = targets - current_q

        # Per-sample Huber loss, weighted for prioritized replay.
        elementwise_loss = F.smooth_l1_loss(
            current_q, targets, reduction="none"
        )
        loss = (elementwise_loss * importance_weights).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.online_net.parameters(), self.grad_clip_norm
        )
        self.optimizer.step()

        self.update_count += 1

        # Prefer environment-step synchronization when the training loop
        # provides it. Keep update-count synchronization for standalone use.
        if env_step is None:
            should_sync = self.update_count % self.target_update_interval == 0
        else:
            if isinstance(env_step, bool) or not isinstance(env_step, int):
                raise TypeError("env_step must be an integer")
            if env_step < 0:
                raise ValueError("env_step must be non-negative")
            should_sync = (
                env_step > 0
                and env_step % self.target_update_interval == 0
            )

        if should_sync:
            self.sync_target()

        return (
            float(loss.detach().cpu().item()),
            td_errors.detach().cpu().numpy().astype(np.float32),
        )
