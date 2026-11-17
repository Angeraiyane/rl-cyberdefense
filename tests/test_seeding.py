import random

import numpy as np
import pytest
import torch

from cyberdef.seeding import set_seed


def test_same_seed_reproduces_python_numpy_and_torch():
    set_seed(42)

    python_a = [random.random() for _ in range(5)]
    numpy_a = np.random.random(5)
    torch_a = torch.rand(5)

    set_seed(42)

    python_b = [random.random() for _ in range(5)]
    numpy_b = np.random.random(5)
    torch_b = torch.rand(5)

    assert python_a == python_b
    assert np.array_equal(numpy_a, numpy_b)
    assert torch.equal(torch_a, torch_b)


def test_different_seeds_produce_different_sequences():
    set_seed(42)
    values_a = torch.rand(10)

    set_seed(43)
    values_b = torch.rand(10)

    assert not torch.equal(values_a, values_b)


def test_seed_sets_python_hashseed_for_child_processes(monkeypatch):
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)

    set_seed(123)

    import os
    assert os.environ["PYTHONHASHSEED"] == "123"


@pytest.mark.parametrize("seed", [-1, -100])
def test_negative_seed_is_rejected(seed):
    with pytest.raises(ValueError):
        set_seed(seed)


@pytest.mark.parametrize("seed", [1.5, "42", None, True])
def test_non_integer_seed_is_rejected(seed):
    with pytest.raises(TypeError):
        set_seed(seed)


def test_deterministic_argument_must_be_boolean():
    with pytest.raises(TypeError):
        set_seed(42, deterministic=1)


def test_deterministic_configuration():
    set_seed(42, deterministic=True)

    if hasattr(torch.backends, "cudnn"):
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False

    set_seed(42, deterministic=False)

    if hasattr(torch.backends, "cudnn"):
        assert torch.backends.cudnn.deterministic is False
        assert torch.backends.cudnn.benchmark is True
