# TD-ML


Dans ce TP on compare les performances de deux modèles de réseaux de neurones différents, RNN et CNN. Pour ce faire on compare leurs performances dans la preédiction d'une timeseries de la météo.

Pour comparer de façon équitable les modèles on va comparer le nombre de paramètres et la loss donnée par le modèle

params rnn: 2701
params cnn: 361

## RNN
le nombre de paramètres dans le RNN est de 2701.

En ce qui concerne la loss, on peut voir dns le graphique suivant:
![3d function value](https://github.com/HatefulRock/TD-ML/blob/main/images/3d_function_value.png?raw=true  "Comparaison des veleurs calculées et réelles de la fonction")

## Résultats:

Les résultats sont montrés dans les graphiques ci-dessous :

![3d function value](https://github.com/HatefulRock/TD-ML/blob/main/images/3d_function_value.png?raw=true  "Comparaison des veleurs calculées et réelles de la fonction")

Les axes x et y sont les paramètres en entrée et l'axe z est la sortie de la fonction. Les points bleus sont les valeurs calculées par le réseau de neurone et les points rouges sont les valeurs réelles de la fonction.

Seuls les 100 premiers des 1000 points ont été affichés par soucis de lisibilité.

On observe que les points bleus et rouges ne coincident pas mais les points bleus semblent suivre la même tendance que les points rouges à x et y fixés. Le réseau créé, bien qu'imparfait calcule une approximation de la fonction.


![mean squared error](https://github.com/HatefulRock/TD-ML/blob/main/images/mean_squared_error.png?raw=true  "Métrique MSE calculée sur les données d'entraînement et de test")

Ce graphe présente la métrique mean squared error calculée sur les données d'entraînement (courbe bleue) et de test (courbe rouge) en fonction de l'époque en abscisse.
La courbe bleue est décroissante, le modèle s'améliore durant toutes les époques.
Cependant la courbe rouge est croissante après 120 époques. Cela signifie qu'après 200 époques, nous sommes en situation de surapprentissage. Avec une valeur de métrique supérieure à 0.7, les données sur le jeu de test ne sont pas satisfaisantes.




## Changements a tester
Afin d'améliorer le modèle présent, on pourrait changer les hyperparamètres du modèle tels que le learning rate, couches cachées ou fonction d'activation. En effet, on pourrait ajouter plus de couches cachées au modèle et voir si le modèle arive a mieux prédire la focntion f. De meme, on pourrait choisir une fonction d'activation qui est plus adaptée aux fonctions periodiques qui ont des valeurs de faible amplitude.
