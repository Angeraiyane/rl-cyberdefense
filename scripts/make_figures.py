"""
Generate the DQN evaluation curve against the two baselines.

Run from the repository root:
    uv run --with-requirements requirements.txt python scripts/make_figures.py
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

CSV_PATH = ROOT / "results" / "raw" / "fpr0.1_d0_du0_p0_seed0.csv"
OUTPUT_PATH = ROOT / "results" / "figures" / "dqn_vs_baselines.png"

RANDOM_BASELINE = -43.28
HEURISTIC_BASELINE = -7.05

DQN_COLOR = "#1f77b4"
RANDOM_COLOR = "#ff7f0e"
HEURISTIC_COLOR = "#2ca02c"


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV introuvable : {CSV_PATH}")

    with CSV_PATH.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError("Le CSV ne contient aucune donnée.")

    required_columns = {
        "step",
        "eval_return_mean",
        "eval_return_std",
    }

    missing = required_columns - set(rows[0].keys())
    if missing:
        raise ValueError(f"Colonnes manquantes dans le CSV : {missing}")

    steps = np.array(
        [float(row["step"]) for row in rows],
        dtype=float,
    )
    means = np.array(
        [float(row["eval_return_mean"]) for row in rows],
        dtype=float,
    )
    stds = np.array(
        [float(row["eval_return_std"]) for row in rows],
        dtype=float,
    )

    order = np.argsort(steps)
    steps = steps[order]
    means = means[order]
    stds = stds[order]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(
        steps,
        means,
        color=DQN_COLOR,
        linewidth=2,
        label="DQN",
    )

    ax.fill_between(
        steps,
        means - stds,
        means + stds,
        color=DQN_COLOR,
        alpha=0.18,
        label="± 1 écart-type sur 20 épisodes d'évaluation (1 graine)",
    )

    ax.axhline(
        RANDOM_BASELINE,
        color=RANDOM_COLOR,
        linestyle="--",
        linewidth=1.8,
        label="Politique aléatoire (-43,28)",
    )

    ax.axhline(
        HEURISTIC_BASELINE,
        color=HEURISTIC_COLOR,
        linestyle="--",
        linewidth=1.8,
        label="RestoreOnAlert (-7,05)",
    )

    ax.set_title("DQN vs baselines")
    ax.set_xlabel("Pas d'entraînement")
    ax.set_ylabel("Retour moyen d'évaluation")

    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Figure générée : {OUTPUT_PATH}")
    print(f"Nombre de points : {len(steps)}")
    print(f"Dernier retour moyen DQN : {means[-1]:.4f}")
    print(f"Dernier écart-type : {stds[-1]:.4f}")


if __name__ == "__main__":
    main()
