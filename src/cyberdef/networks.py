"""Réseaux de neurones qui estiment Q(s, a) pour toutes les actions a d'un coup.

- QNetwork        : MLP classique  obs -> couches cachées -> Q(s, ·)
- DuelingQNetwork : même tronc, puis deux têtes
                      V(s)    : "la situation est-elle bonne ?"         (1 sortie)
                      A(s, a) : "cette action est-elle meilleure que la moyenne ?" (n_actions sorties)
                    recombinées par  Q(s, a) = V(s) + A(s, a) - moyenne_a' A(s, a')
                    (Wang et al., 2016). Soustraire la moyenne rend V et A identifiables :
                    sans ça, on pourrait ajouter une constante à V et l'enlever à A.

Les deux réseaux ont le même tronc (mêmes couches cachées) : la seule différence
entre les configurations "avec / sans Dueling" est la tête.
"""

from __future__ import annotations

import torch
from torch import nn


def _trunk(obs_dim: int, hidden_sizes) -> tuple[nn.Sequential, int]:
    layers, last = [], obs_dim
    for h in hidden_sizes:
        layers += [nn.Linear(last, h), nn.ReLU()]
        last = h
    return nn.Sequential(*layers), last


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_sizes=(128, 128)):
        super().__init__()
        self.trunk, last = _trunk(obs_dim, hidden_sizes)
        self.head = nn.Linear(last, n_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(obs))


class DuelingQNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_sizes=(128, 128)):
        super().__init__()
        self.trunk, last = _trunk(obs_dim, hidden_sizes)
        self.value = nn.Linear(last, 1)
        self.advantage = nn.Linear(last, n_actions)

    @staticmethod
    def combine(value: torch.Tensor, advantage: torch.Tensor) -> torch.Tensor:
        """Q = V + A - moyenne(A). Séparé pour pouvoir le tester à la main."""
        return value + advantage - advantage.mean(dim=-1, keepdim=True)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        z = self.trunk(obs)
        return self.combine(self.value(z), self.advantage(z))


def make_q_network(obs_dim: int, n_actions: int, dueling: bool, hidden_sizes=(128, 128)) -> nn.Module:
    """Point d'entrée unique utilisé par l'agent (contrat de la section 6 du plan)."""
    cls = DuelingQNetwork if dueling else QNetwork
    return cls(obs_dim, n_actions, tuple(hidden_sizes))
