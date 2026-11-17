import csv

import pytest

from cyberdef.train import CSV_FIELDS, train


@pytest.fixture
def small_config():
    return {
        "env": {
            "n_hosts": 10,
            "edges": [
                [0, 1], [1, 2], [2, 3], [3, 4], [4, 0],
                [5, 6], [6, 7], [7, 8], [8, 9], [9, 5], [0, 5],
            ],
            "p_spread": 0.15,
            "patch_factor": 0.2,
            "p_external": 0.01,
            "tpr": 0.80,
            "fpr": 0.10,
            "isolation_steps": 5,
            "horizon": 5,
            "loss_threshold": 8,
            "availability_weight": 0.3,
            "action_costs": {
                "noop": 0.0,
                "isolate": 0.05,
                "restore": 0.10,
                "patch": 0.05,
            },
        },
        "agent": {
            "hidden_sizes": [32, 32],
            "gamma": 0.99,
            "lr": 0.0005,
            "grad_clip_norm": 10.0,
            "buffer_size": 100,
            "batch_size": 2,
            "learning_starts": 2,
            "train_freq": 1,
            "target_update_interval": 2,
            "epsilon_start": 1.0,
            "epsilon_end": 0.05,
            "epsilon_decay_steps": 5,
            "per_alpha": 0.6,
            "per_beta_start": 0.4,
            "per_beta_end": 1.0,
            "per_eps": 1e-6,
        },
        "training": {
            "total_steps": 6,
            "eval_interval": 3,
            "eval_episodes": 2,
            "final_eval_episodes": 3,
            "train_env_seed_offset": 1000,
            "eval_env_seed_start": 900000,
        },
    }


@pytest.mark.parametrize("per", [False, True])
def test_train_writes_expected_csv(small_config, tmp_path, per):
    output = tmp_path / f"training_per_{int(per)}.csv"

    result = train(
        small_config,
        double=True,
        dueling=True,
        per=per,
        seed=42,
        out_csv=output,
    )

    assert output.exists()
    assert result["csv_path"] == output
    assert len(result["evaluations"]) == 2

    with output.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        assert reader.fieldnames == CSV_FIELDS

    assert len(rows) == 2
    assert int(rows[0]["step"]) == 3
    assert int(rows[1]["step"]) == 6

    for row in rows:
        assert float(row["eval_return_std"]) >= 0
        assert float(row["eval_n_compromised"]) >= 0
        assert 0.0 <= float(row["epsilon"]) <= 1.0
        assert float(row["loss"]) >= 0


def test_final_checkpoint_uses_final_evaluation_count(small_config, tmp_path):
    output = tmp_path / "final.csv"

    result = train(
        small_config,
        seed=7,
        out_csv=output,
    )

    # The final checkpoint is evaluated on the configured 3 episodes.
    assert len(result["evaluations"]) == 2
    assert result["evaluations"][-1]["step"] == 6


def test_train_return_keeps_last_completed_episode(
    small_config, tmp_path
):
    output = tmp_path / "train_return.csv"

    train(
        small_config,
        seed=42,
        out_csv=output,
    )

    with output.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    # At step 3, the episode horizon is 5:
    # no complete training episode exists yet.
    assert float(rows[0]["train_return"]) == 0.0
