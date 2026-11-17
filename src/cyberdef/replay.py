"""Mémoires de rejeu (replay buffers).

L'agent stocke ses transitions (s, a, r, s', terminated) puis apprend sur des lots
tirés au hasard dans cette mémoire. Cela casse la corrélation entre transitions
successives et permet de réutiliser chaque expérience plusieurs fois.

- UniformReplay     : chaque transition a la même probabilité d'être tirée.
- PrioritizedReplay : (Schaul et al., 2016, variante proportionnelle)
      priorité       p_i = (|erreur TD_i| + eps) ** alpha
      probabilité    P(i) = p_i / somme_j p_j
      poids          w_i = (N * P(i)) ** (-beta) / max_j w_j
  Les poids d'importance corrigent le biais introduit par le tirage non uniforme :
  sans eux, l'apprentissage converge vers de mauvaises valeurs. beta monte vers 1
  au cours de l'entraînement (correction complète à la fin).
  Une nouvelle transition reçoit la priorité maximale vue jusqu'ici, pour être
  tirée au moins une fois.

Contrat commun (section 6 du plan) :
    buf.add(obs, action, reward, next_obs, terminated)
    batch, weights, idx = buf.sample(batch_size, beta)
    buf.update_priorities(idx, td_errors)
On stocke `terminated` et PAS `truncated` : une troncature n'est pas une vraie fin,
la cible doit continuer à utiliser Q(s').
"""

from __future__ import annotations

import numpy as np


class UniformReplay:
    def __init__(self, capacity: int, obs_dim: int, seed: int = 0):
        self.capacity = int(capacity)
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.float32)
        self.pos = 0      # prochaine case à écrire (tableau circulaire)
        self.size = 0     # nombre de cases remplies
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    def add(self, obs, action, reward, next_obs, terminated) -> int:
        i = self.pos
        self.obs[i] = obs
        self.next_obs[i] = next_obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.terminated[i] = float(terminated)
        self.pos = (self.pos + 1) % self.capacity   # quand c'est plein, on écrase la plus ancienne
        self.size = min(self.size + 1, self.capacity)
        return i

    def _gather(self, idx: np.ndarray) -> dict:
        return {
            "obs": self.obs[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_obs": self.next_obs[idx],
            "terminated": self.terminated[idx],
        }

    def sample(self, batch_size: int, beta: float = 1.0):
        idx = self.rng.integers(0, self.size, size=batch_size)
        weights = np.ones(batch_size, dtype=np.float32)
        return self._gather(idx), weights, idx

    def update_priorities(self, idx, td_errors) -> None:
        pass  # rien à faire : tout est uniforme


class SumTree:
    """Arbre binaire où chaque nœud vaut la somme de ses deux enfants.

    Les feuilles contiennent les priorités. La racine contient leur somme totale.
    Tirer une transition avec une probabilité proportionnelle à sa priorité revient
    à tirer un nombre u uniforme dans [0, total) puis à descendre dans l'arbre :
    O(log N) au lieu de O(N).
    """

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        size = 1
        while size < capacity:
            size *= 2
        self.leaf_start = size              # indice de la première feuille
        self.tree = np.zeros(2 * size, dtype=np.float64)  # tree[1] = racine

    @property
    def total(self) -> float:
        return float(self.tree[1])

    def update(self, data_idx: int, priority: float) -> None:
        i = self.leaf_start + data_idx
        delta = priority - self.tree[i]
        while i >= 1:          # on remonte jusqu'à la racine en corrigeant les sommes
            self.tree[i] += delta
            i //= 2

    def get(self, data_idx) -> np.ndarray:
        return self.tree[self.leaf_start + np.asarray(data_idx)]

    def find(self, u: float) -> int:
        """Renvoie l'indice de la feuille où tombe la valeur cumulée u."""
        i = 1
        while i < self.leaf_start:
            left = 2 * i
            if u < self.tree[left]:
                i = left
            else:
                u -= self.tree[left]
                i = left + 1
        return i - self.leaf_start


class PrioritizedReplay(UniformReplay):
    def __init__(self, capacity: int, obs_dim: int, alpha: float = 0.6, eps: float = 1e-6, seed: int = 0):
        super().__init__(capacity, obs_dim, seed)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.tree = SumTree(capacity)
        self.max_priority = 1.0   # priorité déjà élevée à la puissance alpha

    def add(self, obs, action, reward, next_obs, terminated) -> int:
        i = super().add(obs, action, reward, next_obs, terminated)
        self.tree.update(i, self.max_priority)
        return i

    def sample(self, batch_size: int, beta: float):
        # Échantillonnage stratifié : on coupe [0, total) en batch_size segments
        # égaux et on tire une valeur dans chacun (variance plus faible).
        total = self.tree.total
        segment = total / batch_size
        u = (np.arange(batch_size) + self.rng.random(batch_size)) * segment
        idx = np.array([self.tree.find(min(x, total - 1e-12)) for x in u], dtype=np.int64)
        idx = np.minimum(idx, self.size - 1)   # sécurité numérique

        probs = self.tree.get(idx) / total
        weights = (self.size * probs) ** (-beta)
        weights = (weights / weights.max()).astype(np.float32)
        return self._gather(idx), weights, idx

    def update_priorities(self, idx, td_errors) -> None:
        p = (np.abs(np.asarray(td_errors, dtype=np.float64)) + self.eps) ** self.alpha
        for i, pi in zip(np.asarray(idx), p):
            self.tree.update(int(i), float(pi))
        self.max_priority = max(self.max_priority, float(p.max()))


def make_replay(capacity: int, obs_dim: int, prioritized: bool, alpha: float = 0.6,
                eps: float = 1e-6, seed: int = 0) -> UniformReplay:
    """Point d'entrée unique utilisé par l'agent (contrat de la section 6 du plan)."""
    if prioritized:
        return PrioritizedReplay(capacity, obs_dim, alpha=alpha, eps=eps, seed=seed)
    return UniformReplay(capacity, obs_dim, seed=seed)
