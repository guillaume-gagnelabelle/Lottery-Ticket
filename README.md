# Hypothèse du billet de lotterie gagnant en Pytorch
(Adaptation de https://github.com/rahulvigneswaran/Lottery-Ticket-Hypothesis-in-Pytorch pour la quantification énergétique des réseaux de neurones éparses)

Ce répositoire contient un implémentation **Pytorch** de l'article [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/abs/1803.03635) par [Jonathan Frankle](https://github.com/jfrankle) et [Michael Carbin](https://people.csail.mit.edu/mcarbin/).
		
## Dépendances
```
pip3 install -r requirements.txt
```
## Exécution
```
python3 lottery_ticket.py --arch_type=fc1 --dataset=mnist --prune_percent=90 --prune_iterations=2
```

- `--arch_type`	 : Type d'architecture
	- Options : `fc1` - Réseau de neurones dense, `lenet5` - LeNet5,  `resnet18` - Resnet18,
	- Par défaut : `fc1`
- `--dataset`	: Choix du jeu de données 
	- Options : `mnist`, `cifar10` 
	- Par défaut : `mnist`
- `--prune_percent`	: Pourcentage des poids qui seront élaguer à chaque itération 
	- Par défaut : `90`
- `--prune_iterations`	: Nombre n de cycles d'entraînement qui sera effectué (1 pour aucun élaguag, n-1 élagage en général) 
	- Par défaut : `2`
- `--lr`	: Taux d'apprentissage
	- Par défaut : `1e-04`
- `--decay`	: Régularisation L2 ("weight decay")
	- Par défaut : `1e-05`
- `--batch_size`	: Taille du lot 
	- Par défaut : `512`
- `--end_iter`	: Nombre d'époques d'entraînement d'un cycle
	- Par défaut : `32`
- `--print_freq`	: Fréquence d'impression des métriques de performance (précision, fonction de perte) dans le terminal
	- Par défaut : `1`
- `--valid_freq`	: Fréquence d'évaluation des performances du modèle
	- Par défaut : `1`
- `--train_type`	: type d'entraînement effectué
	- Options : `lt` (Lottery Ticket), `regular` (rétropropagation classique: équivalent à --prune_percent 0 --prune_iterations 1)
	- Par défaut : `lt`
- `--co2_tracking_mode` : Active la sonde de traquage d'émission de CO2
	- Note : désactive toutes les évaluation du modèle lors de l'entraînement
	- Par défaut : False (`store_true`)



## Structure du répositoire
```
Lottery-Ticket
├── archs
|   ├── archs_utils.py
│   ├── cifar10
│   │   ├── LeNet5.py
│   │   └── resnet.py
│   ├── mnist
│       ├── fc1.py
│       ├── LeNet5.py
│       └── resnet.py
|   └── archs_utils.py
├── data
|   └──  data_utils.py
├── inference_emission.py
├── lottery_ticket.py (Entraînement - main)
├── lottery_ticket_emissions.py (Graphiques entraînement - émissions CO2)
├── lottery_ticket_performance.py (Graphiques entraînement - performances)
├── plots
├── post_training_inference.py (Inférence - main)
├── post_training_inference_emissions.py (Graphiques inférence)
├── README.md
├── requirements.txt
├── saves
└── utils.py

```

## Reconnaissance 
Une grande partie du code a été emprunté de [rahulvigneswaran](https://github.com/rahulvigneswaran/Lottery-Ticket-Hypothesis-in-Pytorch).
