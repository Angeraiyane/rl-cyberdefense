"""Politiques de référence : un agent appris n'a de valeur que comparé à elles.

- RandomPolicy    : une action tirée uniformément parmi toutes les actions.
- RestoreOnAlert  : s'il y a une alerte, restaurer un hôte en alerte (en priorité
                    celui qui était déjà en alerte au pas précédent, puis le plus
                    petit indice) ; sinon ne rien faire.

Les deux politiques ne lisent QUE l'observation, comme l'agent : jamais la vérité cachée.
"""

from __future__ import annotations

import numpy as np

from cyberdef.env import CyberDefenseEnv, RESTORE, NOOP


class RandomPolicy:
    def __init__(self, n_actions: int, seed: int = 0):
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray) -> int:
        return int(self.rng.integers(self.n_actions))


class RestoreOnAlert:
    def __init__(self, env: CyberDefenseEnv):
        self.n = env.n
        self.env = env

    def act(self, obs: np.ndarray) -> int:
        alerts = obs[: self.n] > 0.5
        prev = obs[self.n: 2 * self.n] > 0.5
        if not alerts.any():
            return self.env.encode_action(NOOP)
        repeated = np.flatnonzero(alerts & prev)
        host = repeated[0] if len(repeated) > 0 else np.flatnonzero(alerts)[0]
        return self.env.encode_action(RESTORE, int(host))


def evaluate_policy(policy, env_config: dict | None, seeds) -> np.ndarray:
    """Joue un épisode par graine et renvoie le tableau des retours (sommes des récompenses)."""
    returns = []
    for s in seeds:
        env = CyberDefenseEnv(env_config)
        obs, _ = env.reset(seed=int(s))
        total, done = 0.0, False
        while not done:
            obs, r, terminated, truncated, _ = env.step(policy.act(obs))
            total += r
            done = terminated or truncated
        returns.append(total)
    return np.asarray(returns)


if __name__ == "__main__":
    # Usage : python -m cyberdef.baselines   (depuis la racine, avec PYTHONPATH=src)
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Évalue les politiques de référence.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--fpr", type=float, default=None, help="remplace le taux de fausses alertes")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    env_cfg = cfg["env"]
    if args.fpr is not None:
        env_cfg["fpr"] = args.fpr
    start = cfg["training"]["eval_env_seed_start"]
    seeds = range(start, start + args.episodes)

    env = CyberDefenseEnv(env_cfg)
    for name, pol in [("aleatoire", RandomPolicy(env.n_actions, seed=0)),
                      ("restaurer_si_alerte", RestoreOnAlert(env))]:
        R = evaluate_policy(pol, env_cfg, seeds)
        print(f"{name:22s} retour moyen = {R.mean():8.2f}  (écart-type {R.std(ddof=1):.2f}, n={len(R)})")
