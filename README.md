# Lottery Ticket Hypothesis in Pytorch 

This repository contains a **Pytorch** implementation of the paper [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/abs/1803.03635) by [Jonathan Frankle](https://github.com/jfrankle) and [Michael Carbin](https://people.csail.mit.edu/mcarbin/) that can be **easily adapted to any model/dataset**.
		
## Requirements
```
pip3 install -r requirements.txt
```
## How to run the code ? 
### Using datasets/architectures included with this repository :
```
python3 lottery_ticket.py --arch_type=fc1 --dataset=mnist --prune_percent=90 --prune_iterations=2
```

- `--arch_type`	 : Type of architecture
	- Options : `fc1` - Simple fully connected network, `lenet5` - LeNet5,  `resnet18` - Resnet18,
	- Default : `fc1`
- `--dataset`	: Choice of dataset 
	- Options : `mnist`, `cifar10` 
	- Default : `mnist`
- `--prune_percent`	: Percentage of weight to be pruned after each cycle. 
	- Default : `90`
- `--prune_iterations`	: Number of cycle of pruning that should be done (1 for no pruning). 
	- Default : `2`
- `--lr`	: Learning rate 
	- Default : `1.2e-3`
- `--batch_size`	: Batch size 
	- Default : `64`
- `--end_iter`	: Number of Epochs 
	- Default : `32`
- `--print_freq`	: Frequency for printing accuracy and loss 
	- Default : `1`
- `--valid_freq`	: Frequency for Validation 
	- Default : `1`

### Using datasets/architectures that are not included with this repository :
- Adding a new architecture :
	- For example, if you want to add an architecture named `new_model` with `mnist` dataset compatibility. 
		- Go to `/archs/mnist/` directory and create a file `new_model.py`.
		- Now paste your **Pytorch compatible** model inside `new_model.py`.
		- **IMPORTANT** : Make sure the *input size*, *number of classes*, *number of channels*, *batch size* in your `new_model.py` matches with the corresponding dataset that you are adding (in this case, it is `mnist`).
		- Now open `main.py` and go to `line 36` and look for the comment `# Data Loader`. Now find your corresponding dataset (in this case, `mnist`) and add `new_model` at the end of the line `from archs.mnist import AlexNet, LeNet5, fc1, vgg, resnet`.
		- Now go to `line 82` and add the following to it :
			```
			elif args.arch_type == "new_model":
        		model = new_model.new_model_name().to(device)
			``` 
			Here, `new_model_name()` is the name of the model that you have given inside `new_model.py`.
- Adding a new dataset :
	- For example, if you want to add a dataset named `new_dataset` with `fc1` architecture compatibility.
		- Go to `/archs` and create a directory named `new_dataset`.
		- Now go to /archs/new_dataset/` and add a file named `fc1.py` or copy paste it from existing dataset folder.
		- **IMPORTANT** : Make sure the *input size*, *number of classes*, *number of channels*, *batch size* in your `new_model.py` matches with the corresponding dataset that you are adding (in this case, it is `new_dataset`).
		- Now open `main.py` and goto `line 58` and add the following to it :
			```
			elif args.dataset == "cifar100":
        		traindataset = datasets.new_dataset('../data', train=True, download=True, transform=transform)
        		testdataset = datasets.new_dataset('../data', train=False, transform=transform)from archs.new_dataset import fc1
			``` 
			**Note** that as of now, you can only add dataset that are [natively available in Pytorch](https://pytorch.org/docs/stable/torchvision/datasets.html). 



## Repository Structure
```
Lottery-Ticket-Hypothesis-in-Pytorch
├── archs
|   ├── archs_utils.py
│   ├── cifar10
│   │   ├── fc1.py
│   │   ├── LeNet5.py
│   │   └── resnet.py
│   └── mnist
│       ├── fc1.py
│       ├── LeNet5.py
│       └── resnet.py
├── data
├── inference_emission.py
├── lottery_ticket.py
├── lottery_ticket_emissions.py
├── lottery_ticket_performance.py
├── plots
├── post_training_inference.py
├── README.md
├── requirements.txt
├── saves
└── utils.py

```

## Acknowledgement 
Parts of code were borrowed from [ktkth5](https://github.com/ktkth5/lottery-ticket-hyopothesis).
