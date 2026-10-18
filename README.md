# Détection d'impayés sur des paiements par chèque : classification très déséquilibrée

Prédire si une transaction par chèque finira en impayé, alors que **seulement 0,65 % des transactions** sont des impayés
(4,6 millions de transactions). Le projet compare plusieurs modèles et plusieurs stratégies de rééquilibrage des classes.

*Projet réalisé en binôme : Marvin Curty-Ménez et Marin Nagy, Master SISE, Université Lumière Lyon 2.*

## Démarche

1. **Nettoyage** (`01_clean.ipynb`) : le fichier source contenait une ligne d'en-tête dupliquée au milieu des données,
   ce qui faussait les types de toutes les colonnes. Ligne retirée, types définis explicitement, export en Parquet.
2. **Exploration** (`02_explore.ipynb`) : proportion des classes, valeurs manquantes (aucune).
3. **Modélisation** (`03_build.ipynb`) :
   - découpage **temporel** : entraînement jusqu'au 31/08/2017, test sur les transactions suivantes
     (3,9 M / 0,76 M), pour reproduire une utilisation réelle du modèle ;
   - métrique : **F1-score de la classe impayé**, avec recherche du seuil de décision qui le maximise
     (le seuil par défaut de 0,5 n'a pas de sens sur des données aussi déséquilibrées) ;
   - modèles : Random Forest, SVM, XGBoost (avec recherche d'hyperparamètres en validation croisée) ;
   - rééquilibrage : pondération des classes (`scale_pos_weight`), sous-échantillonnage aléatoire, NearMiss,
     TomekLinks, ClusterCentroids, puis recherche conjointe du taux de sous-échantillonnage et de la pondération.

## Résultats (jeu de test, F1 de la classe impayé au meilleur seuil)

| Modèle | F1 |
|---|---|
| Random Forest (paramètres par défaut) | 0,053 |
| XGBoost (paramètres par défaut) | 0,090 |
| XGBoost optimisé (ROC-AUC en CV : 0,87) + pondération des classes | 0,115 |
| XGBoost + TomekLinks | 0,096 |
| XGBoost + ClusterCentroids | 0,109 |
| **XGBoost + sous-échantillonnage aléatoire (5 pour 1)** | **0,139** |

Enseignements :
- le **sous-échantillonnage aléatoire** a donné de meilleurs résultats que les méthodes plus élaborées
  (NearMiss, TomekLinks, ClusterCentroids) ;
- la pondération des classes et le sous-échantillonnage **décalent fortement le seuil optimal** (de 0,4 à plus de 0,95) :
  ajuster ce seuil est indispensable ;
- limite : le seuil est choisi sur le jeu de test, ces scores sont donc légèrement optimistes.
  Une version plus rigoureuse le fixerait sur le jeu de validation.

## Structure

```
Bank-Imbalanced/
├── notebooks/
│   ├── 01_clean.ipynb      # Nettoyage et export Parquet
│   ├── 02_explore.ipynb    # Exploration
│   └── 03_build.ipynb      # Modèles et stratégies de rééquilibrage
├── models/                 # Modèles XGBoost entraînés (.joblib)
├── data/                   # Données (non versionnées)
└── requirements.txt
```

## Lancer le projet

```bash
python -m venv imbalanced-env
source imbalanced-env/bin/activate      # Windows : imbalanced-env\Scripts\activate
pip install -r requirements.txt xgboost imbalanced-learn
```

Placer le fichier de données `guillaume.txt` dans `data/`, puis exécuter les notebooks dans l'ordre.

**Stack :** Python · pandas · scikit-learn · XGBoost · imbalanced-learn · matplotlib
