"""Environnement de cyberdéfense simulé (strictement défensif, aucun code d'attaque).

Un réseau de N hôtes est représenté par un graphe. Une infection simulée se propage
le long des liens. À chaque pas, l'agent défenseur choisit UNE action :
ne rien faire, isoler, restaurer ou patcher un hôte. Il ne voit pas l'état réel
des hôtes, seulement des alertes bruitées.

Ordre des événements dans un pas (step) :
    1. l'action de l'agent est appliquée ;
    2. l'infection se propage (à partir de l'état juste après l'action) ;
    3. intrusion externe éventuelle ;
    4. calcul de la récompense ;
    5. les compteurs d'isolation diminuent de 1 ;
    6. tirage des alertes et construction de l'observation.

Fin d'épisode :
    - terminated=True  : fin RÉELLE, au moins `loss_threshold` hôtes compromis (réseau perdu).
                         La cible du DQN ne doit PAS utiliser Q(s').
    - truncated=True   : fin par TRONCATURE, l'horizon est atteint mais le monde continue.
                         La cible du DQN DOIT garder gamma * max Q(s').
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Valeurs par défaut = configs/base.yaml (section env). Dupliquées ici pour que
# l'environnement soit utilisable seul, par exemple dans les tests.
DEFAULT_ENV_CONFIG = {
    "n_hosts": 10,
    "edges": [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0],
              [5, 6], [6, 7], [7, 8], [8, 9], [9, 5],
              [0, 5]],
    "p_spread": 0.15,
    "patch_factor": 0.2,
    "p_external": 0.01,
    "tpr": 0.80,
    "fpr": 0.10,
    "isolation_steps": 5,
    "horizon": 100,
    "loss_threshold": 8,
    "availability_weight": 0.3,
    "action_costs": {"noop": 0.0, "isolate": 0.05, "restore": 0.10, "patch": 0.05},
}

# Types d'actions (le décodage d'un indice d'action est dans decode_action).
NOOP, ISOLATE, RESTORE, PATCH = "noop", "isolate", "restore", "patch"


class CyberDefenseEnv(gym.Env):
    """Défense d'un réseau simulé contre une infection qui se propage.

    Espace d'actions : Discrete(3N + 1)
        0              -> ne rien faire
        1 .. N         -> isoler l'hôte (a - 1)
        N+1 .. 2N      -> restaurer l'hôte (a - N - 1)
        2N+1 .. 3N     -> patcher l'hôte (a - 2N - 1)

    Observation : vecteur float32 de taille 4N
        [alertes du pas t | alertes du pas t-1 | isolation restante / isolation_steps | patché]
    """

    metadata = {"render_modes": []}

    def __init__(self, config: dict | None = None):
        super().__init__()
        cfg = dict(DEFAULT_ENV_CONFIG)
        if config is not None:
            cfg.update(config)
        self.cfg = cfg

        self.n = int(cfg["n_hosts"])
        self.p_spread = float(cfg["p_spread"])
        self.patch_factor = float(cfg["patch_factor"])
        self.p_external = float(cfg["p_external"])
        self.tpr = float(cfg["tpr"])
        self.fpr = float(cfg["fpr"])
        self.isolation_steps = int(cfg["isolation_steps"])
        self.horizon = int(cfg["horizon"])
        self.loss_threshold = int(cfg["loss_threshold"])
        self.availability_weight = float(cfg["availability_weight"])
        self.action_costs = dict(cfg["action_costs"])

        # Liste d'adjacence : neighbors[i] = voisins de l'hôte i.
        self.neighbors = [[] for _ in range(self.n)]
        for u, v in cfg["edges"]:
            if not (0 <= u < self.n and 0 <= v < self.n) or u == v:
                raise ValueError(f"Arête invalide : {(u, v)}")
            self.neighbors[u].append(v)
            self.neighbors[v].append(u)

        self.n_actions = 3 * self.n + 1
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(4 * self.n,), dtype=np.float32)

        # État caché (initialisé dans reset).
        self.compromised = np.zeros(self.n, dtype=bool)
        self.isolation = np.zeros(self.n, dtype=np.int64)   # pas d'isolation restants
        self.patched = np.zeros(self.n, dtype=bool)
        self.alerts = np.zeros(self.n, dtype=bool)
        self.prev_alerts = np.zeros(self.n, dtype=bool)
        self.t = 0

    # ------------------------------------------------------------------ outils
    def decode_action(self, action: int) -> tuple[str, int | None]:
        """Transforme un indice d'action en (type, hôte)."""
        a = int(action)
        if not 0 <= a < self.n_actions:
            raise ValueError(f"Action hors limites : {a}")
        if a == 0:
            return NOOP, None
        if a <= self.n:
            return ISOLATE, a - 1
        if a <= 2 * self.n:
            return RESTORE, a - self.n - 1
        return PATCH, a - 2 * self.n - 1

    def encode_action(self, kind: str, host: int | None = None) -> int:
        """Inverse de decode_action (utile pour l'heuristique et les tests)."""
        if kind == NOOP:
            return 0
        offset = {ISOLATE: 1, RESTORE: self.n + 1, PATCH: 2 * self.n + 1}[kind]
        return offset + int(host)

    def _draw_alerts(self) -> np.ndarray:
        """Alerte avec proba TPR si compromis, FPR sinon."""
        u = self.np_random.random(self.n)
        p_alert = np.where(self.compromised, self.tpr, self.fpr)
        return u < p_alert

    def _observation(self) -> np.ndarray:
        return np.concatenate([
            self.alerts.astype(np.float32),
            self.prev_alerts.astype(np.float32),
            (self.isolation / self.isolation_steps).astype(np.float32),
            self.patched.astype(np.float32),
        ])

    def _info(self) -> dict:
        # ATTENTION : "compromised" est la vérité cachée. Elle sert uniquement aux
        # analyses et aux tests ; l'agent ne doit JAMAIS l'utiliser pour agir.
        return {
            "t": self.t,
            "n_compromised": int(self.compromised.sum()),
            "n_isolated": int((self.isolation > 0).sum()),
            "compromised": self.compromised.copy(),
        }

    # ------------------------------------------------------------- API Gym
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)  # initialise self.np_random avec la graine
        self.t = 0
        self.compromised[:] = False
        self.isolation[:] = 0
        self.patched[:] = False
        self.compromised[self.np_random.integers(self.n)] = True  # foyer initial
        self.prev_alerts[:] = False
        self.alerts = self._draw_alerts()
        return self._observation(), self._info()

    def step(self, action: int):
        kind, host = self.decode_action(action)

        # 1. Appliquer l'action.
        if kind == ISOLATE:
            self.isolation[host] = self.isolation_steps
        elif kind == RESTORE:
            self.compromised[host] = False
        elif kind == PATCH:
            self.patched[host] = True

        # 2. Propagation, calculée à partir d'une copie : toutes les infections
        #    du pas se font "en même temps" (l'ordre des hôtes ne compte pas).
        isolated = self.isolation > 0
        sources = self.compromised & ~isolated
        new_compromised = self.compromised.copy()
        for u in np.flatnonzero(sources):
            for v in self.neighbors[u]:
                if self.compromised[v] or isolated[v]:
                    continue
                p = self.p_spread * (self.patch_factor if self.patched[v] else 1.0)
                if self.np_random.random() < p:
                    new_compromised[v] = True
        self.compromised = new_compromised

        # 3. Intrusion externe : un hôte au hasard, sauf s'il est isolé.
        if self.np_random.random() < self.p_external:
            h = self.np_random.integers(self.n)
            p_ok = self.patch_factor if self.patched[h] else 1.0
            if not isolated[h] and self.np_random.random() < p_ok:
                self.compromised[h] = True

        # 4. Récompense.
        n_comp = int(self.compromised.sum())
        n_iso = int(isolated.sum())
        reward = -(n_comp + self.availability_weight * n_iso) / self.n - self.action_costs[kind]

        # 5. Les isolations avancent d'un pas.
        self.isolation = np.maximum(self.isolation - 1, 0)

        # 6. Nouvelles alertes et observation.
        self.t += 1
        self.prev_alerts = self.alerts
        self.alerts = self._draw_alerts()

        terminated = n_comp >= self.loss_threshold
        if terminated:
            # Pénalité = coût d'un réseau entièrement compromis jusqu'à l'horizon.
            # Sans elle, l'agent aurait intérêt à "perdre vite" pour ne plus payer.
            reward -= float(self.horizon - self.t)
        truncated = (not terminated) and self.t >= self.horizon

        return self._observation(), float(reward), bool(terminated), bool(truncated), self._info()
