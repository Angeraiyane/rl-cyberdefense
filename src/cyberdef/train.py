"""Training loop for DQN cyber-defense experiments."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import yaml

from cyberdef.agent import DQNAgent
from cyberdef.env import CyberDefenseEnv


CSV_FIELDS = [
    "step",
    "eval_return_mean",
    "eval_return_std",
    "eval_n_compromised",
    "train_return",
    "epsilon",
    "loss",
]


def _linear_schedule(start: float, end: float, step: int, duration: int) -> float:
    """Linear interpolation from start to end over duration steps."""
    fraction = min(1.0, max(0.0, step / duration))
    return float(start + fraction * (end - start))


def _evaluate(
    agent: DQNAgent,
    env_config: dict,
    seeds,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate greedily on fixed seeds; return episode returns and final compromises."""
    returns = []
    compromised_counts = []

    for episode_seed in seeds:
        env = CyberDefenseEnv(dict(env_config))
        obs, _ = env.reset(seed=int(episode_seed))
        total_return = 0.0
        terminated = False
        truncated = False
        info = {}

        while not (terminated or truncated):
            action = agent.act(obs, epsilon=0.0)
            obs, reward, terminated, truncated, info = env.step(action)
            total_return += float(reward)

        returns.append(total_return)
        compromised_counts.append(int(info["n_compromised"]))

    return (
        np.asarray(returns, dtype=np.float64),
        np.asarray(compromised_counts, dtype=np.float64),
    )


def train(
    config: Mapping,
    double: bool = False,
    dueling: bool = False,
    per: bool = False,
    seed: int = 0,
    out_csv: str | Path = "results/raw/training.csv",
) -> dict:
    """Train one DQN variant and write periodic evaluation results to CSV."""
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")

    cfg = dict(config)
    env_config = dict(cfg["env"])
    agent_config = dict(cfg["agent"])
    training_config = dict(cfg["training"])

    total_steps = int(training_config["total_steps"])
    eval_interval = int(training_config["eval_interval"])
    eval_episodes = int(training_config["eval_episodes"])
    final_eval_episodes = int(training_config["final_eval_episodes"])
    eval_seed_start = int(training_config["eval_env_seed_start"])
    train_seed_offset = int(training_config["train_env_seed_offset"])

    learning_starts = int(agent_config["learning_starts"])
    train_freq = int(agent_config["train_freq"])
    batch_size = int(agent_config["batch_size"])
    epsilon_start = float(agent_config["epsilon_start"])
    epsilon_end = float(agent_config["epsilon_end"])
    epsilon_decay_steps = int(agent_config["epsilon_decay_steps"])
    beta_start = float(agent_config["per_beta_start"])
    beta_end = float(agent_config["per_beta_end"])

    if min(total_steps, eval_interval, eval_episodes, final_eval_episodes) <= 0:
        raise ValueError("Training and evaluation lengths must be positive")
    if min(learning_starts, train_freq, batch_size, epsilon_decay_steps) <= 0:
        raise ValueError("Agent training parameters must be positive")

    agent = DQNAgent(
        cfg,
        double=double,
        dueling=dueling,
        per=per,
        seed=seed,
    )

    env = CyberDefenseEnv(env_config)
    obs, _ = env.reset(seed=int(seed) + train_seed_offset)

    output_path = Path(out_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    episode_return = 0.0
    last_train_return = 0.0
    latest_loss = 0.0
    losses_since_eval = []
    evaluation_history = []

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for step in range(1, total_steps + 1):
            epsilon = _linear_schedule(
                epsilon_start, epsilon_end, step, epsilon_decay_steps
            )

            action = agent.act(obs, epsilon=epsilon)
            next_obs, reward, terminated, truncated, _ = env.step(action)

            # Only true termination masks the Bellman bootstrap.
            agent.replay.add(
                obs,
                action,
                float(reward),
                next_obs,
                bool(terminated),
            )

            episode_return += float(reward)
            obs = next_obs

            if terminated or truncated:
                last_train_return = episode_return
                episode_return = 0.0
                obs, _ = env.reset()

            if (
                step >= learning_starts
                and step % train_freq == 0
                and len(agent.replay) >= batch_size
            ):
                if per:
                    beta = _linear_schedule(
                        beta_start, beta_end, step, total_steps
                    )
                else:
                    beta = 1.0

                batch, weights, indices = agent.replay.sample(
                    batch_size, beta=beta
                )
                latest_loss, td_errors = agent.update(
                    batch,
                    weights,
                    env_step=step,
                )
                losses_since_eval.append(latest_loss)

                if per:
                    agent.replay.update_priorities(indices, td_errors)

            is_checkpoint = step % eval_interval == 0
            is_final_step = step == total_steps

            if is_checkpoint or is_final_step:
                n_eval = (
                    final_eval_episodes if is_final_step else eval_episodes
                )
                eval_seeds = range(eval_seed_start, eval_seed_start + n_eval)

                eval_returns, compromised = _evaluate(
                    agent,
                    env_config,
                    eval_seeds,
                )

                eval_std = (
                    float(eval_returns.std(ddof=1))
                    if len(eval_returns) > 1
                    else 0.0
                )
                mean_loss = (
                    float(np.mean(losses_since_eval))
                    if losses_since_eval
                    else float(latest_loss)
                )

                row = {
                    "step": step,
                    "eval_return_mean": float(eval_returns.mean()),
                    "eval_return_std": eval_std,
                    "eval_n_compromised": float(compromised.mean()),
                    "train_return": float(last_train_return),
                    "epsilon": epsilon,
                    "loss": mean_loss,
                }
                writer.writerow(row)
                csv_file.flush()

                evaluation_history.append(row)
                losses_since_eval.clear()

                print(
                    f"step={step:>7} | "
                    f"eval={row['eval_return_mean']:>8.2f} "
                    f"+/- {row['eval_return_std']:.2f} | "
                    f"compromis={row['eval_n_compromised']:.2f} | "
                    f"epsilon={epsilon:.3f} | loss={mean_loss:.5f}"
                )

    if hasattr(env, "close"):
        env.close()

    return {
        "agent": agent,
        "csv_path": output_path,
        "evaluations": evaluation_history,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Entraînement DQN cyberdéfense")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--double", action="store_true")
    parser.add_argument("--dueling", action="store_true")
    parser.add_argument("--per", action="store_true")
    parser.add_argument("--out-csv", default=None)
    args = parser.parse_args()

    with Path(args.config).open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    fpr = float(config["env"]["fpr"])
    suffix = (
        f"fpr{fpr:g}_d{int(args.double)}_du{int(args.dueling)}"
        f"_p{int(args.per)}_seed{args.seed}.csv"
    )
    out_csv = args.out_csv or str(Path("results/raw") / suffix)

    train(
        config,
        double=args.double,
        dueling=args.dueling,
        per=args.per,
        seed=args.seed,
        out_csv=out_csv,
    )


if __name__ == "__main__":
    main()
