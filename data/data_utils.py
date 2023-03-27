import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np


def getData(args, train_percent=0.7, val_percent=0.15):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    if args.dataset == "mnist":
        traindataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        testdataset = datasets.MNIST('./data', train=False, transform=transform)

    elif args.dataset == "cifar10":
        traindataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
        testdataset = datasets.CIFAR10('./data', train=False, transform=transform)

    elif args.dataset == "fashionmnist":
        traindataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
        testdataset = datasets.FashionMNIST('./data', train=False, transform=transform)

    elif args.dataset == "cifar100":
        traindataset = datasets.CIFAR100('./data', train=True, download=True, transform=transform)
        testdataset = datasets.CIFAR100('./data', train=False, transform=transform)

    else:
        print("\nWrong Dataset choice \n")
        exit()

    
    # merge training and test datasets
    dataset_total = torch.utils.data.ConcatDataset([traindataset, testdataset])
        
    # Determine the number of samples for each dataset split
    total_size = len(dataset_total)
    train_size = int(train_percent * total_size)
    val_size = int(val_percent * total_size)
    test_size = total_size - train_size - val_size

    # split the data into train, validation and test
    traindataset, valdataset, testdataset = torch.utils.data.random_split(dataset_total, (train_size, val_size, test_size))

    # Create data loaders for the training, validation, and test sets
    train_loader = torch.utils.data.DataLoader(dataset = traindataset, batch_size=args.batch_size, shuffle = True, num_workers=0, drop_last=False)
    val_loader = torch.utils.data.DataLoader(dataset = valdataset, batch_size=args.batch_size, shuffle = False, num_workers=0, drop_last=False)
    test_loader = torch.utils.data.DataLoader(dataset = testdataset, batch_size=args.batch_size, shuffle = False, num_workers=0, drop_last=True)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--dataset', type=str, default='mnist', help='name of dataset (default: mnist)')
    args = parser.parse_args()

    train_loader, val_loader, test_loader = getData(args)

    # Check the length of each dataset
    print("Number of training samples: ", len(train_loader.dataset))
    print("Number of validation samples: ", len(val_loader.dataset))
    print("Number of test samples: ", len(test_loader.dataset))

    # Calculate the ratios of each dataset
    total_size = len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset)
    train_ratio = len(train_loader.dataset) / total_size
    val_ratio = len(val_loader.dataset) / total_size
    test_ratio = len(test_loader.dataset) / total_size

    # Check that the ratios match the expected split
    assert train_ratio == 0.7, "Incorrect training set ratio"
    assert val_ratio == 0.15, "Incorrect validation set ratio"
    assert test_ratio == 0.15, "Incorrect test set ratio"

    # Print confirmation message if all ratios are correct
    print("Data split correctly into training, validation, and test sets")