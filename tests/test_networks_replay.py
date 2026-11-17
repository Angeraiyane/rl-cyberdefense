import numpy as np
import pytest
import torch

from cyberdef.networks import make_q_network, DuelingQNetwork, QNetwork
from cyberdef.replay import make_replay, SumTree, UniformReplay, PrioritizedReplay


# ----------------------------------------------------------------- réseaux
@pytest.mark.parametrize("dueling", [False, True])
def test_network_output_shape(dueling):
    net = make_q_network(40, 31, dueling=dueling)
    assert isinstance(net, DuelingQNetwork if dueling else QNetwork)
    assert net(torch.zeros(7, 40)).shape == (7, 31)


def test_dueling_combination_by_hand():
    V = torch.tensor([[2.0]])
    A = torch.tensor([[1.0, 3.0, 5.0]])           # moyenne = 3
    Q = DuelingQNetwork.combine(V, A)
    assert torch.allclose(Q, torch.tensor([[0.0, 2.0, 4.0]]))  # 2 + A - 3
    assert torch.allclose(Q.mean(), V.squeeze())  # la moyenne de Q vaut V


def test_same_trunk_size():
    a = sum(p.numel() for p in make_q_network(40, 31, False).trunk.parameters())
    b = sum(p.numel() for p in make_q_network(40, 31, True).trunk.parameters())
    assert a == b


# ------------------------------------------------------------ rejeu uniforme
def fill(buf, n, obs_dim=4):
    for k in range(n):
        o = np.full(obs_dim, k, dtype=np.float32)
        buf.add(o, k % 3, float(k), o + 1, k % 2 == 0)


def test_uniform_overwrites_oldest():
    buf = UniformReplay(capacity=5, obs_dim=4)
    fill(buf, 8)
    assert len(buf) == 5
    assert sorted(buf.rewards.tolist()) == [3, 4, 5, 6, 7]  # 0, 1, 2 ont été écrasés


def test_uniform_sample_shapes_and_weights():
    buf = make_replay(100, 4, prioritized=False)
    fill(buf, 50)
    batch, w, idx = buf.sample(16)
    assert batch["obs"].shape == (16, 4) and batch["terminated"].shape == (16,)
    assert np.all(w == 1.0)
    assert np.all(idx < 50)


# ---------------------------------------------------------------- PER
def test_sumtree_total_and_find():
    t = SumTree(5)
    for i, p in enumerate([1, 2, 3, 4, 0]):
        t.update(i, p)
    assert t.total == pytest.approx(10)
    assert [t.find(u) for u in [0.5, 1.5, 3.5, 6.5, 9.9]] == [0, 1, 2, 3, 3]
    t.update(2, 0.0)
    assert t.total == pytest.approx(7)


def test_per_sampling_follows_priorities():
    buf = PrioritizedReplay(capacity=4, obs_dim=4, alpha=1.0, eps=0.0, seed=0)
    fill(buf, 4)
    buf.update_priorities(np.arange(4), np.array([1.0, 2.0, 3.0, 4.0]))
    counts = np.zeros(4)
    for _ in range(25000):
        _, _, idx = buf.sample(4, beta=1.0)
        np.add.at(counts, idx, 1)
    freq = counts / counts.sum()
    assert np.allclose(freq, [0.1, 0.2, 0.3, 0.4], atol=0.01)


def test_per_importance_weights():
    buf = PrioritizedReplay(capacity=4, obs_dim=4, alpha=1.0, eps=0.0, seed=0)
    fill(buf, 4)
    buf.update_priorities(np.arange(4), np.array([1.0, 2.0, 3.0, 4.0]))
    _, w, idx = buf.sample(64, beta=1.0)
    assert np.all(w <= 1.0 + 1e-6) and np.all(w > 0)
    # Avec beta = 1 : w_i = (N * P(i))^-1, puis division par le plus grand poids du lot.
    raw = 1.0 / (4 * np.array([0.1, 0.2, 0.3, 0.4])[idx])
    assert np.allclose(w, raw / raw.max(), atol=1e-5)


def test_per_new_transitions_get_max_priority():
    buf = PrioritizedReplay(capacity=8, obs_dim=4, alpha=1.0, eps=0.0)
    fill(buf, 2)
    buf.update_priorities(np.array([0, 1]), np.array([5.0, 1.0]))
    i = buf.add(np.zeros(4), 0, 0.0, np.zeros(4), False)
    assert buf.tree.get(i) == pytest.approx(5.0)


def test_per_never_samples_empty_slots():
    buf = make_replay(1000, 4, prioritized=True, seed=1)
    fill(buf, 10)
    for _ in range(200):
        _, _, idx = buf.sample(32, beta=0.4)
        assert np.all(idx < 10)
