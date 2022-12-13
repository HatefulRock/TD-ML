
## Projet Machine Learning

### Prédiction du cours d'une action avec un LSTM et un GRU

## Les données
J'utilise une base de données trouvée sur Kaggle: https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs
Ces données contiennent le cours d'actions de plusieurs entreprises de 1970 à 2015. Les colonnes présentes sont: la date du jour, le prix de l'action a l'ouverture du marché ce jour la, le prix maximal atteint ce jour la, le prix minimal de l'action atteint ce jour la, le prix de l'action a la fermeture du marché et le volume de transcations. J'ai décidé de restreindre mon étude sur le prix de l'ouverture de l'action dénoté "Open".
Je vais donc entrainer sepearmment un LSTM et un GRU sur les mêmes données et ensuite évaluer les performances de chaque modèle sur d'autres cours.


## Prétraitement des données
A partir des données, j'ai utilisé le module StandardScaler de scikit learn afin de normaliser les données d'entrée. Ensuite, je separe les données dans 4 batchs différents avec la fonction split_data: x_train, x_test, y_train et y_test. Cette fonction incorpore une valuer lookback qui me permet de regler la fenetre temporelle des données. 

## Architecture du réseau
Pour fiare des etudes sur des time series, il est intéressant d'uiliser une architecture de type RNN car ils sont très puissants pour traiter des données séquentielles. En effet, on peut utiliser l'information dans la mémoire pour éffectuer des meilleures prédictions. Cependant, les RNN ont des soucis pour faire face au probleme du vanishing gradient. Ces pour ça que d'autres architectures de type RNN ont etées créees telles que les LSTM( Long Short Term Memory) ou les GRU( Gated Recurrent Unit). Les lSTM et les GRU font usage de gates pour trier les informations qui sont importantes et ainsi ne garder en mémoire uniquement des informations nécessaires aux prédictions futures.

# GRU
![GRU](https://github.com/HatefulRock/TD-ML/blob/main/projet/images/gru.jpg?raw=true  "GRU")

Le GRU contient deux gates: la reset gate et l'update gate. La reset gate decide si l'inforamtion conetenue dans la cellule precedente est importante ou non. L'update gate quant a elle decide de mettre a jour la cellule actuelle avec l'information actuelle.

# LSTM
![LSTM](https://github.com/HatefulRock/TD-ML/blob/main/projet/images/lstm.png?raw=true  "LSTM")

En plus de la reset gate et de l'update gate, le LSTM contient aussi une forget gate et une output gate. La forget gate décide la quantité d'information qui est passée d'etat en etat. L'output gate détermine le prochain hidden state en controlant l'information qui arrive vers celle ci.


## Implementation
L'implementation de ces deux modèles est tres simple sur Pytorch car il suffit d'utiliser l'architecture deja existante. Ainsi, l'architecture est composée d'une couche GRU ou LSTM suivie d'une couche linéaire qui nous sort la prédiction.
Les hyperparmètres que j'ai choisi d'utiliser sont une hidden state size de 32 et 5 couches cachées car sinon les calculs prennet beaucoup de temps sur mon ordinateur.
L'optimiseur utilisé est ADAM avec un learning rate de 1e-3. La loss utilisée est la MSE.


## Analyse des résultats
J'ai éffectué des tests pour analyser les performances des deux modèles et de les comparer. 

Loss avec 4 couches cachées:
![Loss curve with 4 hidden layers](https://github.com/HatefulRock/TD-ML/blob/main/projet/images/loss%204hd.png?raw=true  "Loss curve")

Loss avec 2 couches cachées et hidden state size égal à 50:
![Loss curve with 2 hidden layers, hidden sate size =50](https://github.com/HatefulRock/TD-ML/blob/main/projet/images/Loss%20hidden%20state%2050.png?raw=true  "Loss curve")




Prédiction:

Prédiction avec 5 couches cachées:
![Prediction of the network, 5 hidden layers](https://github.com/HatefulRock/TD-ML/blob/main/projet/images/Predicted.png?raw=true  "Prediction")

Prédiction avec 2 couches cachées et hidden state size a 50:
![Prediction of the network, hidden state size=50](https://github.com/HatefulRock/TD-ML/blob/main/projet/images/Loss%20hidden%20state%2050.png?raw=true  "Prediction")



Temps de calcul:

Temps pris pour 5 couches cachées:
![Time taken to train and test networks, 5 hidden layers](https://github.com/HatefulRock/TD-ML/blob/main/projet/images/time_5hid.png?raw=true  "Time taken")

Temps pris pour 4 couches cachées:
![Time taken to train and test networks, 4 hidden layers](https://github.com/HatefulRock/TD-ML/blob/main/projet/images/time4hd.png?raw=true  "Time taken")

Temps pris pour 3 couches cachées:
![Time taken to train and test networks, 3 hidden layers](https://github.com/HatefulRock/TD-ML/blob/main/projet/images/time_3hd.png?raw=true  "Time taken")

Temps pris 2 couches cachées et hidden state size égal à 50:
![Time taken to train and test networks, 2 hidden layers, hidden sate size=50](https://github.com/HatefulRock/TD-ML/blob/main/projet/images/time%2050%20hidden%20state.png?raw=true  "Time taken")



J'ai aussi crée des fonctions qui permettent de tester les modèles sur d'autres cours pris aleatoirement dans le dossier Stocks. Cette fonction calcule la MSE pour chaque cours et fait une moyenne sur 10 cours différents. Ces fonction me permettent donc d'evaluer les performances des modèles sur d'autres types de données et donc de comparer la robustesse des modèles. Pour cela je sauvegarde les modèles grace a la fonctionnalité torch.save. Ceci me permet donc d'enregistrer les poids du réseau et de pouvoir les réutilser sans avoir à refaire l'entrainement a chaque fois. Cependant, j'ai eu des soucis avec les résultats de ces fonctions car les valeurs prédites par les modèles semblent être constantes et ne suivent pas le cours de l'action.

Mauvaise prédiction:
![Failed prediction](https://github.com/HatefulRock/TD-ML/blob/main/projet/images/testing_fail.png?raw=true  "Prediction")

## Conclusion

Pour conclure, on peut voir que le GRU semble mieux performer en moyenne que le LSTM
