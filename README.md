# TD-ML


La fonction que nous avons choisie est f(u,v)=cos(x+y)

Le reseau qu'on a cherché à impémenter pour résoudre ce problème de régression est un réseau feed-forward a deux entrées , 10 couches chachées et une sortie. La fonction d'activation utilisée est tanh.
L'optimiseur utilisé est Adam, une fonction qui donne des bons résultats et qui converge rapidement. Le nombre d'epochs est 500.
Pour calculer la loss, on utilise comme métrique la Mean Squared Error qui est une métrique adaptée pour les problèmes de régression. Elles est definie comme etant sqrt(mean((y-ypred)**2)) avec y la valeur qu'on cherche à prédire et ypred la valeur prédite.

## Données

Pour créer les données d'entrée du réseau on crée un array et on l'initialise avec des valeurs aléatoirement distribuées dans l'intervalle [-5,5]. Le nombre de valeurs est un paramètre modifiable. Ensuite, on crée les valeurs target comme étant les images par la fonction f des valeurs d'entrée. Ces valeurs la sont donc les valeurs que le réseau va chercher a apprendre et calculer.



## Résultats:

Les résultats ne semblent pas etre très concluants comme on peut voir avec les graphiques ci dessous:

![Training and validation loss](/home/hatefulrock/Pictures/loss.png'  "Training and validation loss")
![Training and validation accuracy](/home/hatefulrock/Pictures/acc.png'  "Training and validation accuracy")

