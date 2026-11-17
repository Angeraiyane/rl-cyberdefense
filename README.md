# Améliorations de DQN pour la cyberdéfense simulée

Mini-projet du module *Apprentissage par renforcement*, réalisé par Ange Raïyane ESSOAZINA et Babacar FAYE.

**Question de recherche.** Parmi Double DQN, Dueling et le replay priorisé (PER), lesquelles améliorent réellement DQN, seules ou combinées, pour un agent qui défend un réseau simulé contre une infection qui se propage ? Variation du groupe : cet effet résiste-t-il quand le taux de fausses alertes augmente ?

> Cadre strictement défensif : aucun code d'attaque, aucun test sur un système réel. La propagation est une simple dynamique aléatoire sur un graphe.

## Installation

Python 3.11 ou plus récent.

```bash
git clone https://github.com/Angeraiyane/rl-cyberdefense.git
cd rl-cyberdefense
python -m venv .venv
# Windows : .venv\Scripts\activate      Mac/Linux : source .venv/bin/activate
pip install -r requirements.txt
```

## Vérifier que tout fonctionne

```bash
pytest -q
```

## Évaluer les politiques de référence

```bash
# Windows (PowerShell) : $env:PYTHONPATH="src"     Mac/Linux : export PYTHONPATH=src
python -m cyberdef.baselines                # FPR de référence 0,10
python -m cyberdef.baselines --fpr 0.25     # variation
```

Résultats attendus (100 épisodes d'évaluation, graines 900000 à 900099) :

| Politique | FPR 0,02 | FPR 0,10 | FPR 0,25 |
| --- | --- | --- | --- |
| Aléatoire | −43,3 | −43,3 | −43,3 |
| Restaurer si alerte | −2,3 | −7,1 | −11,6 |

## Organisation du code

| Fichier | Rôle | Responsable |
| --- | --- | --- |
| `configs/base.yaml` | Tous les paramètres (environnement, agent, entraînement) | les deux |
| `src/cyberdef/env.py` | Environnement `CyberDefenseEnv` (Gymnasium) | Ange |
| `src/cyberdef/baselines.py` | Politique aléatoire et heuristique | Ange |
| `src/cyberdef/networks.py` | Réseau Q classique et Dueling | Ange |
| `src/cyberdef/replay.py` | Rejeu uniforme et rejeu priorisé (sum-tree) | Ange |
| `src/cyberdef/agent.py` | Agent DQN, option Double | Babacar |
| `src/cyberdef/train.py` | Entraînement et évaluation | Babacar |
| `scripts/` | Campagne, analyse statistique, figures | les deux |
| `results/raw/` | Un CSV par entraînement (résultats bruts) | — |

## Reproductibilité

- Toutes les graines sont fixées. Entraînement : graines 0 à 9. Évaluation : environnement séparé, graines 900000+.
- L'évaluation se fait sans exploration (ε = 0).
- Les 8 configurations de l'ablation utilisent exactement les mêmes hyperparamètres (`configs/base.yaml`).

## Contributions et usage de l'IA

Voir `CONTRIBUTIONS.md`. Un assistant d'IA (Claude) a été utilisé pendant le projet ; c'est déclaré dans le rapport.
