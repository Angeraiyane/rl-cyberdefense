import numpy as np
import pytest

from cyberdef.env import CyberDefenseEnv, ISOLATE, RESTORE, PATCH, NOOP


def run_episode(env, seed, policy):
    obs, info = env.reset(seed=seed)
    traj = [obs.copy()]
    done = False
    while not done:
        obs, r, te, tr, info = env.step(policy(obs))
        traj.append((obs.copy(), r, te, tr))
        done = te or tr
    return traj


def test_spaces():
    env = CyberDefenseEnv()
    obs, info = env.reset(seed=0)
    assert env.action_space.n == 31
    assert obs.shape == (40,) and obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    assert info["n_compromised"] == 1  # un seul foyer au départ


def test_same_seed_same_episode():
    pol = lambda o: 0
    a = run_episode(CyberDefenseEnv(), 42, pol)
    b = run_episode(CyberDefenseEnv(), 42, pol)
    assert len(a) == len(b)
    for x, y in zip(a[1:], b[1:]):
        assert np.array_equal(x[0], y[0]) and x[1] == y[1]


def test_different_seeds_differ():
    pol = lambda o: 0
    a = run_episode(CyberDefenseEnv(), 1, pol)
    b = run_episode(CyberDefenseEnv(), 2, pol)
    assert not (len(a) == len(b) and all(np.array_equal(x[0], y[0]) for x, y in zip(a[1:], b[1:])))


def test_action_encoding_roundtrip():
    env = CyberDefenseEnv()
    for a in range(env.n_actions):
        kind, host = env.decode_action(a)
        assert env.encode_action(kind, host) == a
    assert env.decode_action(0) == (NOOP, None)
    assert env.decode_action(1) == (ISOLATE, 0)
    assert env.decode_action(11) == (RESTORE, 0)
    assert env.decode_action(30) == (PATCH, 9)
    with pytest.raises(ValueError):
        env.decode_action(31)


def test_isolated_host_never_spreads_nor_gets_infected():
    # Propagation certaine, aucune intrusion externe.
    env = CyberDefenseEnv({"p_spread": 1.0, "p_external": 0.0, "isolation_steps": 50})
    env.reset(seed=0)
    env.compromised[:] = False
    env.compromised[3] = True
    env.step(env.encode_action(ISOLATE, 3))  # le seul hôte infecté est isolé
    for _ in range(10):
        env.step(0)
    assert env.compromised.sum() == 1  # personne d'autre n'a été infecté

    env.reset(seed=0)
    env.compromised[:] = False
    env.compromised[2] = True
    env.step(env.encode_action(ISOLATE, 1))  # voisin isolé : protégé
    assert not env.compromised[1]
    assert env.compromised[3]  # l'autre voisin, lui, est infecté (p = 1)


def test_restore_cleans_host():
    env = CyberDefenseEnv({"p_spread": 0.0, "p_external": 0.0})
    env.reset(seed=0)
    h = int(np.flatnonzero(env.compromised)[0])
    env.step(env.encode_action(RESTORE, h))
    assert env.compromised.sum() == 0


def test_patch_reduces_infection_rate():
    def infection_rate(patched):
        env = CyberDefenseEnv({"p_spread": 0.5, "p_external": 0.0, "patch_factor": 0.2})
        hits = 0
        for s in range(2000):
            env.reset(seed=s)
            env.compromised[:] = False
            env.compromised[0] = True
            env.patched[1] = patched
            env.step(0)
            hits += env.compromised[1]
        return hits / 2000
    assert infection_rate(False) == pytest.approx(0.5, abs=0.04)
    assert infection_rate(True) == pytest.approx(0.1, abs=0.03)


def test_alert_rates_match_tpr_fpr():
    env = CyberDefenseEnv({"tpr": 0.8, "fpr": 0.1})
    env.reset(seed=0)
    env.compromised[:] = np.arange(10) < 5
    alerts = np.array([env._draw_alerts() for _ in range(5000)])
    assert alerts[:, :5].mean() == pytest.approx(0.8, abs=0.02)
    assert alerts[:, 5:].mean() == pytest.approx(0.1, abs=0.02)


def test_truncation_at_horizon_is_not_terminal():
    # Aucune infection possible : l'épisode doit aller jusqu'à l'horizon.
    env = CyberDefenseEnv({"p_spread": 0.0, "p_external": 0.0, "horizon": 100})
    env.reset(seed=0)
    for t in range(1, 101):
        _, _, te, tr, _ = env.step(0)
        assert not te
        assert tr == (t == 100)


def test_real_termination_when_network_lost():
    env = CyberDefenseEnv({"p_spread": 1.0, "p_external": 0.0, "loss_threshold": 8})
    env.reset(seed=0)
    done, steps = False, 0
    while not done:
        _, r, te, tr, info = env.step(0)
        steps += 1
        done = te or tr
    assert te and not tr
    assert info["n_compromised"] >= 8
    assert r < -(env.horizon - steps) + 1e-9  # la pénalité de fin réelle est appliquée


def test_reward_formula():
    env = CyberDefenseEnv({"p_spread": 0.0, "p_external": 0.0})
    env.reset(seed=0)
    env.compromised[:] = False
    env.compromised[[0, 1]] = True
    _, r, *_ = env.step(env.encode_action(ISOLATE, 5))
    # 2 compromis, 1 isolé, coût d'isolation 0,05
    assert r == pytest.approx(-(2 + 0.3 * 1) / 10 - 0.05)


def test_isolation_expires():
    env = CyberDefenseEnv({"isolation_steps": 5, "p_spread": 0.0, "p_external": 0.0})
    env.reset(seed=0)
    env.step(env.encode_action(ISOLATE, 4))
    remaining = []
    for _ in range(6):
        remaining.append(env.isolation[4])
        env.step(0)
    assert remaining == [4, 3, 2, 1, 0, 0]
