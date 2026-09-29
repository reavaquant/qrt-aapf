# Boucle d’itération du projet QRT

L’objectif est de trouver une configuration qui prédit bien sur des observations
non vues. Une configuration comprend le preprocessing, les features, le modèle
et ses hyperparameters. Ces éléments interagissent : une feature peu utile pour
un modèle peut aider un autre modèle.

## Point de départ

Notre benchmark interne est le run LightGBM décrit dans
`notebooks/01_benchmark1.ipynb` :

- 41 features numériques d’origine, sans `TS`, `ALLOCATION` ni `GROUP` en entrée
  du modèle ; valeurs manquantes conservées et prises en charge par LightGBM.
- Target de classification : `1` si `target > 0`, sinon `0`.
- 8 folds construits sur les dates `TS` uniques mélangées, avec `seed=42`.
  Toutes les observations d’une même date restent dans le même fold.
- `LGBMClassifier` avec `n_estimators=200`, `num_leaves=7`,
  `learning_rate=0.05` et `random_state=42`.
- Classe prédite positive si la probabilité est supérieure ou égale à `0.5`.
- Accuracy OOF : **0.5225556991**, soit **52,26 %**.

Les prédictions OOF sont les prédictions de validation : chaque observation est
prédite par un modèle entraîné sans son fold. L’accuracy globale est calculée sur
toutes ces observations avec `sklearn.metrics.accuracy_score` ; elle n’est pas
la moyenne non pondérée des scores des folds, dont les tailles diffèrent.

Ce score est notre référence de validation interne. Il n’est ni le score du test
officiel du challenge ni une mesure de performance économique. Les identifiants
`TS` ne fournissent pas une chronologie exploitable : notre CV ne constitue pas
un backtest chronologique.

## Comment conduire une expérience

1. **Formuler une hypothèse.** Partir d’une observation de l’EDA, d’un diagnostic
   du modèle ou d’une propriété pertinente d’une méthode. Expliquer ce que la
   modification pourrait améliorer.
2. **Choisir une modification cohérente.** Une expérience doit répondre à une
   question identifiable. Cela ne signifie pas forcément changer un seul
   paramètre : essayer un modèle peut nécessiter un preprocessing adapté.
3. **Évaluer sur les mêmes folds.** Conserver les observations évaluées, la
   définition de la target et la métrique pour permettre une comparaison directe.
4. **Comparer et interpréter.** Examiner l’accuracy OOF et les différences de
   score fold par fold face au benchmark ou à la meilleure configuration retenue.
   Utiliser des diagnostics par groupe lorsqu’ils répondent à une question précise.
5. **Décider.** Garder la modification, l’abandonner ou préciser l’hypothèse pour
   une nouvelle expérience. Un résultat négatif est aussi une information utile.

## Articuler features, modèles et tuning

| Levier | Question |
| --- | --- |
| Features | Une autre représentation des données facilite-t-elle la prédiction ? |
| Modèle | Une autre famille exploite-t-elle mieux les informations disponibles ? |
| Hyperparameters | Quelle complexité et quelle régularisation conviennent à cette configuration ? |

Il n’y a pas d’ordre rigide « toutes les features, puis tous les modèles, puis le
tuning ». Au début, explorer quelques hypothèses motivées et quelques familles
de modèles avec des paramètres raisonnables. Consacrer davantage de tuning aux
configurations prometteuses et réexaminer les interactions si nécessaire.

Une distribution apparemment normale d’une feature ne démontre pas son utilité
prédictive et ne suffit pas à choisir une famille de modèles. Distinguer les faits
observés, les hypothèses proposées et les résultats de validation.

## Conditions de comparaison

- Ajuster l’imputation, le scaling et toute transformation apprise uniquement
  sur la partie train de chaque fold. Une `Pipeline` clonée et entraînée dans la
  CV permet de respecter cette séparation.
- Construire les features avec les seules informations disponibles au moment de
  la prédiction. Ne pas utiliser la target de validation dans leur construction.
- Garder les données brutes identiques entre expériences comparables et tracer
  explicitement un changement de données.
- Conserver les folds exacts, les seeds et les paramètres. Ne pas changer de seed
  pour sélectionner un score plus favorable.
- Un petit gain isolé ne suffit pas à conclure. Examiner son amplitude et sa
  répartition entre folds ; leur dispersion n’est pas un intervalle de confiance
  fondé sur des expériences indépendantes.
- À force de sélectionner des configurations sur la même CV, on peut s’y adapter.
  Pour les finalistes, vérifier la sensibilité à d’autres découpages par date sans
  présenter cette vérification comme un test sur de nouvelles données indépendantes.

## Historique dans MLflow

**Un run correspond à une expérience complète avec ses 8 folds.**

Enregistrer le modèle et ses paramètres, la configuration de CV, la liste des
features, la définition de la target, le seuil de décision, les scores globaux
et par fold, ainsi que les prédictions OOF indexées par `ROW_ID` et leurs folds.
Conserver le commit Git, une copie du code effectivement utilisé et l’environnement
Python. Sauvegarder le notebook avant de l’archiver dans le run.

La métrique `accuracy` est enregistrée entre 0 et 1 ; son affichage en pourcentage
est une présentation. Comparer les accuracies avec la même unité et exprimer leur
différence en points de pourcentage.

Documenter brièvement la raison de l’expérience et la décision prise après lecture
des résultats dans les tags ou la description du run. MLflow conserve l’historique
des expériences ; ce document décrit la méthode de travail.
