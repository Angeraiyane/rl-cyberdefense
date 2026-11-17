import numpy as np
import pytest
import torch

from cyberdef.agent import DQNAgent


@pytest.fixture
def config():
    return {
        "env": {"n_hosts": 10},
        "agent": {
            "hidden_sizes": [128, 128],
            "gamma": 0.99,
            "lr": 0.0005,
            "grad_clip_norm": 10.0,
            "buffer_size": 500,
            "batch_size": 8,
            "target_update_interval": 2,
            "per_alpha": 0.6,
            "per_eps": 1e-6,
        },
    }


def make_batch(batch_size=8):
    rng = np.random.default_rng(123)
    return {
        "obs": rng.random((batch_size, 40), dtype=np.float32),
        "actions": rng.integers(0, 31, batch_size),
        "rewards": rng.normal(size=batch_size).astype(np.float32),
        "next_obs": rng.random((batch_size, 40), dtype=np.float32),
        "terminated": np.zeros(batch_size, dtype=np.float32),
    }


@pytest.mark.parametrize("double", [False, True])
@pytest.mark.parametrize("dueling", [False, True])
@pytest.mark.parametrize("per", [False, True])
def test_agent_variants_construct(config, double, dueling, per):
    agent = DQNAgent(config, double, dueling, per, seed=42)

    assert agent.obs_dim == 40
    assert agent.n_actions == 31
    assert len(agent.replay) == 0
    assert agent.double is double
    assert agent.dueling is dueling
    assert agent.per is per


def test_act_returns_valid_action(config):
    agent = DQNAgent(config, seed=42)
    obs = np.zeros(40, dtype=np.float32)

    for epsilon in (0.0, 0.5, 1.0):
        action = agent.act(obs, epsilon)
        assert isinstance(action, int)
        assert 0 <= action < 31


def test_act_is_reproducible(config):
    a = DQNAgent(config, seed=42)
    b = DQNAgent(config, seed=42)
    obs = np.zeros(40, dtype=np.float32)

    actions_a = [a.act(obs, epsilon=1.0) for _ in range(20)]
    actions_b = [b.act(obs, epsilon=1.0) for _ in range(20)]

    assert actions_a == actions_b


def test_update_returns_loss_and_td_errors(config):
    agent = DQNAgent(config, double=True, dueling=True, per=True, seed=42)
    batch = make_batch()
    weights = np.ones(8, dtype=np.float32)

    loss, td_errors = agent.update(batch, weights)

    assert np.isfinite(loss)
    assert loss >= 0
    assert td_errors.shape == (8,)
    assert np.all(np.isfinite(td_errors))


def test_target_network_is_synchronized_at_interval(config):
    agent = DQNAgent(config, seed=42)

    # First update must not synchronize (interval = 2).
    agent.update(make_batch(), np.ones(8, dtype=np.float32))
    assert agent.update_count == 1

    # Second update triggers synchronization.
    agent.update(make_batch(), np.ones(8, dtype=np.float32))
    assert agent.update_count == 2

    for online, target in zip(
        agent.online_net.parameters(), agent.target_net.parameters()
    ):
        assert torch.equal(online, target)


def test_target_parameters_do_not_require_grad(config):
    agent = DQNAgent(config, seed=42)

    assert all(
        not parameter.requires_grad
        for parameter in agent.target_net.parameters()
    )


def test_truncated_transition_keeps_bootstrap(config):
    agent = DQNAgent(config, seed=42)
    batch = make_batch()
    batch["terminated"][:] = 0.0

    # A truncation is deliberately represented by terminated=0.
    _, td_errors = agent.update(batch, np.ones(8, dtype=np.float32))
    assert td_errors.shape == (8,)


def test_invalid_epsilon_is_rejected(config):
    agent = DQNAgent(config, seed=42)

    with pytest.raises(ValueError):
        agent.act(np.zeros(40, dtype=np.float32), epsilon=1.5)


def test_invalid_observation_shape_is_rejected(config):
    agent = DQNAgent(config, seed=42)

    with pytest.raises(ValueError):
        agent.act(np.zeros(39, dtype=np.float32), epsilon=0.0)


def test_full_yaml_config_and_agent_section_are_supported(config):
    full = DQNAgent(config, seed=7)
    section = DQNAgent(config["agent"], seed=7)

    assert full.obs_dim == section.obs_dim
    assert full.n_actions == section.n_actions

def test_greedy_action_does_not_consume_exploration_rng(config):
    a = DQNAgent(config, seed=42)
    b = DQNAgent(config, seed=42)
    obs = np.zeros(40, dtype=np.float32)

    # Une action gloutonne ne doit pas avancer le RNG d'exploration.
    a.act(obs, epsilon=0.0)

    next_a = a.act(obs, epsilon=1.0)
    next_b = b.act(obs, epsilon=1.0)

    assert next_a == next_b


def test_target_sync_uses_environment_step_when_provided(config):
    agent = DQNAgent(config, seed=42)
    weights = np.ones(8, dtype=np.float32)

    agent.update(make_batch(), weights, env_step=1)

    # Au pas 1, le réseau cible ne doit pas être synchronisé.
    assert any(
        not torch.equal(online, target)
        for online, target in zip(
            agent.online_net.parameters(), agent.target_net.parameters()
        )
    )

    agent.update(make_batch(), weights, env_step=2)

    # Au pas 2, la synchronisation doit avoir lieu.
    assert all(
        torch.equal(online, target)
        for online, target in zip(
            agent.online_net.parameters(), agent.target_net.parameters()
        )
    )
