# TD-ML


La fonction que nous avons choisie est f(x,y)=cos(2x+y)

Le reseau qu'on a cherché à implémenter pour résoudre ce problème de régression est un réseau feed-forward a deux entrées , 10 couches cachées et une sortie. La fonction d'activation utilisée est tanh.
L'optimiseur utilisé est Adam, une fonction qui donne des bons résultats et qui converge rapidement. Le nombre d'époques est 1000.
Pour calculer la loss, on utilise comme métrique la Mean Squared Error qui est une métrique adaptée pour les problèmes de régression. Elles est definie comme etant sqrt(mean((y-ypred)**2)) avec y la valeur qu'on cherche à prédire et ypred la valeur prédite.

## Installation

En plus des bibliothèques habituelles en Machine Learning, nous avons choisi d'utiliser la bibliothèque plotly pour afficher les graphes, ce qui permet de les manipuler (par exemple tourner pour des graphes 3D) puis de les enregistrer en format png très facilement.

Pour installer la bibliothèque il faut lancer la commande **pip install plotly**

## Données

Pour créer les données d'entrée du réseau on crée un array et on l'initialise avec des valeurs aléatoirement distribuées dans l'intervalle [-5,5]. Le nombre de valeurs est un paramètre modifiable. Ensuite, on crée les valeurs target comme étant les images par la fonction f des valeurs d'entrée. On fait de même pour les données de test, en ayant une répartition 80% training, 20% test. Ces valeurs la sont donc les valeurs que le réseau va chercher a apprendre et calculer. on choisit d'entraîner nos données sur 1000 points. (ie 800 points d'entrainement et 200 de test)



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
