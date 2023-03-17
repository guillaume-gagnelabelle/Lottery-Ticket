import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from AnimalDataset import AnimalDataset


def getData(args):
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

    elif args.dataset == "animals":
        traindataset = AnimalDataset('trainclasses.txt', transform)
        testdataset = AnimalDataset('testclasses.txt', transform)

    else:
        print("\nWrong Dataset choice \n")
        exit()

    train_loader = torch.utils.data.DataLoader(traindataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)
    test_loader = torch.utils.data.DataLoader(testdataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True)

    return train_loader, test_loader
