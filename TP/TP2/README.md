# TD-ML


Dans ce TP on compare les performances de deux modèles de réseaux de neurones différents, RNN et CNN. Pour ce faire on compare leurs performances dans la preédiction d'une timeseries de la météo.

Pour comparer de façon équitable les modèles on va comparer le nombre de paramètres et la loss donnée par le modèle.


## RNN
le nombre de paramètres dans le RNN est de 2701.

En ce qui concerne la loss, on peut voir que dans le graphique suivant:
![Loss](https://github.com/HatefulRock/TD-ML/blob/main/TP/TP2/images/RNNloss.png?raw=true  "Training loss and testing loss")

La loss est très basse ce qui est surprenant car on prédit une valeur réelle et donc on devrait pas avoir autant de précision.

On peut voir la prédiction du réseau dans le graphique ci contre:
![Prediction of the network](https://github.com/HatefulRock/TD-ML/blob/main/TP/TP2/images/RNNpredict.png?raw=true  "Prediction")

On peut voir que la prediction à la fin ne semble pas suivre la meme trajectoire que celle de la timeseries.


## CNN
Le nombre de paramètres dans le CNN est de 361 ce qui est inférieur au RNN.

Pour la loss, qu'on peut voir dans le graphique suivant:
![Loss](https://github.com/HatefulRock/TD-ML/blob/main/TP/TP2/images/CNNloss.png?raw=true  "Training loss and testing loss")

De meme, on peut voir ici que la loss est très basse ce qui est surprenant aussi car on devrait pas avoir une aussi bonne précision sur une prédiction.

On peut voir la prédiction du réseau dans le graphique ci contre:
![Prediction of the network](https://github.com/HatefulRock/TD-ML/blob/main/TP/TP2/images/CNNpredict.png?raw=true  "Prediction")

De meme, à la fin la prediction ne semble pas suivre la timeseries précisement.

## Conclusions
Les résultats des deux réseaux semblent etre équivalents sauf que le CNN a moins de paramètres. Pour completer les tests, on pourrait tester la vitesse que met le réseau a calculer sa prédiction. 
