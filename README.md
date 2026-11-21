# Réseau de neurones à une couche cachée

Le projet entraîne deux modèles avec NumPy. `SimpleSLPClassifier` prédit une classe et `SimpleSLPRegressor` prédit une valeur continue. Les deux proposent `fit`, `predict` et `score`. Le classifieur propose aussi `predict_proba`, avec une colonne par classe dans l’ordre de `classes_`.

## Installation et exécution

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python demo.py
```

Le script affiche les scores sur le jeu de test et génère `loss_curves.png`.

## Fonctionnement du réseau

Pour un mini-lot de $m$ observations, la couche cachée calcule $Z_1=XW_1+b_1$, puis applique une activation pour obtenir $A_1=f(Z_1)$. Les activations proposées sont la sigmoïde, tanh, ReLU et l’identité. La couche de sortie calcule $Z_2=A_1W_2+b_2$. La sortie et la fonction de perte dépendent de la tâche.

| Tâche | Sortie | Perte |
| --- | --- | --- |
| Classification binaire | Sigmoïde, qui donne la probabilité de la deuxième classe | Entropie croisée binaire |
| Classification multiclasse | Softmax, qui répartit une probabilité totale de 1 entre les classes | Entropie croisée multiclasse |
| Régression | Identité, qui conserve la valeur calculée | Erreur quadratique moyenne (MSE) |

En rétropropagation, on calcule d’abord le gradient de la perte par rapport à $Z_2$. En classification, il vaut $D_2=(P-Y)/m$, où $P$ contient les probabilités et $Y$ les cibles encodées. En régression, il vaut $D_2=2(\hat Y-Y)/(m q)$ pour $q$ sorties. Puis on transmet ce gradient à la couche cachée avec $D_1=(D_2W_2^\top)\odot f'(Z_1)$. Les gradients des deux matrices de poids sont $X^\top D_1$ et $A_1^\top D_2$. Pour chaque biais, on somme le gradient de sa couche sur les observations du mini-lot.

À chaque époque, les exemples sont mélangés puis traités par mini-lots. Les opérations NumPy calculent les gradients de tout un lot à la fois. `batch_size=1` traite un exemple par mise à jour et `batch_size=None` utilise toutes les données d’entraînement. Le momentum conserve une fraction du déplacement précédent. Avec `momentum=0`, la mise à jour revient à soustraire le gradient multiplié par le taux d’apprentissage.

Avec `early_stopping=True`, une partie des données est réservée à la validation. Si la perte de validation ne baisse pas d’au moins `tol` pendant `n_iter_no_change` époques, l’entraînement s’arrête. Le modèle revient aux paramètres de l’époque où la perte de validation était la plus basse. En régression, les cibles sont standardisées à partir des données utilisées pour les mises à jour, puis les prédictions sont remises dans leur unité d’origine.

Pour stabiliser les calculs, softmax soustrait le plus grand logit de chaque observation avant de calculer les exponentielles. L’entropie croisée d’entraînement est calculée directement à partir des logits afin de garder une perte exploitable pour des prédictions très proches de 0 ou 1.

## Démonstration et résultats

Chaque expérience réserve 25 % des données au test et utilise une graine de 42. La répartition conserve les proportions de classes pour Breast Cancer et Iris. Les caractéristiques sont standardisées avec les statistiques du jeu d’entraînement. Pour chacun des deux modèles, le script utilise une couche cachée de 24 neurones, tanh, des mini-lots de 16, 200 époques, aucun momentum et aucun arrêt anticipé. En régression, le taux de scikit-learn est doublé pour compenser sa convention de demi-MSE. Les deux modèles tirent leurs poids initiaux selon leurs propres règles. [Documentation de scikit-learn sur les MLP](https://scikit-learn.org/stable/modules/neural_networks_supervised.html)

| Jeu de données | Mesure sur le test | SimpleSLP | scikit-learn |
| --- | --- | ---: | ---: |
| Diabetes | R² | 0,4991 | 0,5000 |
| Breast Cancer | Exactitude | 97,20 % | 97,20 % |
| Iris | Exactitude | 94,74 % | 94,74 % |
| Régression synthétique | R² | 0,9756 | 0,9721 |

![Courbes de perte d’entraînement pour les quatre jeux de données](loss_curves.png)

L’exactitude est la part des classes correctement prédites. Un R² de 1 correspond à des prédictions parfaites. Sur cette séparation, les scores sont proches pour les quatre jeux. Le R² d’environ 0,50 sur Diabetes indique qu’il reste une part importante de variation à expliquer. Les scores proviennent d’une seule séparation entraînement/test. Les courbes montrent la perte sur les données d’entraînement. SimpleSLP la recalcule avec les poids de fin d’époque, tandis que scikit-learn agrège les pertes des mini-lots au fil de l’époque. La MSE est affichée sur les cibles standardisées.

## Questions pour l’oral

- **Pourquoi une activation cachée non linéaire ?** Sans elle, la composition des deux transformations affines reste affine. Une activation non linéaire permet par exemple d’apprendre XOR.
- **Comment lire une sortie binaire de 0,8 ?** C’est la probabilité estimée de la deuxième classe de `classes_`. La première reçoit 0,2.
- **Pourquoi garder le jeu de test à part ?** Les poids et les statistiques de standardisation sont estimés sans ses observations. Le score de test mesure alors la performance sur des observations que le modèle n’a pas utilisées pour apprendre.
- **Que vérifient les différences finies ?** On modifie légèrement chaque poids et chaque biais, puis on mesure la variation de la perte. Le test compare cette estimation numérique au gradient calculé par rétropropagation.
