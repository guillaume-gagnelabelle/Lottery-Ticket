import argparse
import copy
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from tensorboardX import SummaryWriter
import seaborn as sns
import pickle
from codecarbon import EmissionsTracker
from collections import defaultdict, OrderedDict
import wandb
import logging
import time

# Custom Libraries
import utils
from data import data_utils
from archs import archs_utils
from plots import plots_utils

writer = SummaryWriter()
wandb.login(key="6650aaf8018bf14396b47b6869c885d2156d86c7")


def main(args, ITE=0):
    args.seed = ITE
    args.nb_images_seen = 0  # in unit of the number of training images seen
    utils.set_seed(args)

    project =f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/logs_{args.train_type}_pp{args.prune_percent}x{args.prune_iterations}_{args.seed}.pt"
    print(f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/logs_{args.train_type}_pp{args.prune_percent}x{args.prune_iterations}_{args.seed}.pt")


    # Wandb initialization
    wandb.init(project=project, entity="ift3710-h23", config=args)

    # Carbon tracker initialization
    tracker = EmissionsTracker(project_name=project,
                               measure_power_secs=1,
                               tracking_mode="process",
                               log_level="critical",
                               save_to_logger=True
                               )
    tracker.start()
    start = time.time()
    reinit = True if args.prune_type == "reinit" else False

    train_loader, val_loader, test_loader = data_utils.getData(args)

    model = archs_utils.getModel(args)
    model.apply(archs_utils.weight_init)

    # Copying and Saving Initial State
    initial_state_dict = copy.deepcopy(model.state_dict())

    mask = archs_utils.make_mask(model)
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for name, param in model.named_parameters():
        print(name, param.size())

    # Pruning
    # NOTE First Pruning Iteration is of No Compression
    best_accuracy = 0
    ITERATION = args.prune_iterations
    comp = np.zeros(ITERATION, float)

    for _ite in range(args.start_epoch, ITERATION):
        if not _ite == 0:
            archs_utils.prune_by_percentile(model, mask, args.prune_percent, resample=resample, reinit=reinit)
            if reinit:
                model.apply(archs_utils.weight_init)
                step = 0
                for name, param in model.named_parameters():
                    if 'weight' in name:
                        weight_dev = param.device
                        param.data = torch.from_numpy(param.data.cpu().numpy() * mask[step]).to(weight_dev)
                        step = step + 1
            else:
                archs_utils.original_initialization(model, mask, initial_state_dict)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        print(f"\n--- Pruning Level [{ITE}:{_ite}/{ITERATION}]: ---")

        # Print the table of Nonzeros in each layer
        comp1 = utils.print_nonzeros(model)
        comp[_ite] = comp1
        args.logs["non_zeros_weights"][args.nb_images_seen] = comp1
        pbar = tqdm(range(args.end_epoch))

        for iter_ in pbar:

            # Frequency for Testing
            if iter_ % args.valid_freq == 0:
                if not args.co2_tracking_mode:
                    test_loss, test_accuracy = test(model, test_loader, criterion)
                    args.logs["test_loss"][args.nb_images_seen] = test_loss
                    args.logs["test_accuracy"][args.nb_images_seen] = test_accuracy
                    args.logs["co2"][args.nb_images_seen] = 0

                elif args.co2_tracking_mode:
                    args.logs["test_loss"][args.nb_images_seen] = 0
                    args.logs["test_accuracy"][args.nb_images_seen] = 0
                    args.logs["co2"][args.nb_images_seen] = tracker.flush()
                args.logs["time"][args.nb_images_seen] = time.time() - start

                # Save Weights
                if test_accuracy > best_accuracy:
                    best_accuracy = test_accuracy
                    best_state_dict = copy.deepcopy(model.state_dict())

            # ----------------------------------- CORE --------- TRAINING ---------------------------------------------
            train_loss, train_accuracy = train(model, train_loader, optimizer, criterion, args)
            args.logs["train_loss"][args.nb_images_seen] = train_loss
            args.logs["train_accuracy"][args.nb_images_seen] = train_accuracy

            # Frequency for Printing Accuracy and Loss
            if iter_ % args.print_freq == 0:
                pbar.set_description(
                    f'Train Epoch: {iter_}/{args.end_epoch} Loss: {test_loss:.6f} Accuracy: {test_accuracy:.2f}% Best Accuracy: {best_accuracy:.2f}%')

        writer.add_scalar('Accuracy/test', best_accuracy, comp1)
        best_accuracy = 0

    # Copying and Saving Final State
    final_state_dict = copy.deepcopy(model.state_dict())
    utils.checkdir(f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/")
    torch.save({
        "time": args.logs["time"],
        "non_zeros_weights": args.logs["non_zeros_weights"],
        "co2": args.logs["co2"],
        "test_loss": args.logs["test_loss"],
        "train_loss": args.logs["train_loss"],
        "test_accuracy": args.logs["test_accuracy"],
        "train_accuracy": args.logs["train_accuracy"],
        "initial_state_dict": initial_state_dict,
        "best_state_dict": best_state_dict,
        "final_state_dict": final_state_dict,
    },
        f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/logs_{args.train_type}_pp{args.prune_percent}x{args.prune_iterations}_{args.seed}.pt")

    # Carbon Emissions
    # tracker.add_metric("Energy Consumption (Joules)", tracker.emissions)
    # tracker.add_metric("CO2 Emissions (kg)", tracker.estimate_carbon_emissions())
    tracker.stop()


def train(model, train_loader, optimizer, criterion, args):
    EPS = 1e-6
    model.train()
    train_loss = 0
    correct = 0
    for batch_idx, (imgs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        imgs, targets = imgs.to(args.device), targets.to(args.device)
        output = model(imgs)
        loss = criterion(output, targets)
        train_loss += loss.item()
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(targets.data.view_as(pred)).sum().item()
        args.nb_images_seen += len(targets)
        loss.backward()

        if args.train_type == "lt":
            # Freezing Pruned weights by making their gradients Zero
            for name, p in model.named_parameters():
                if 'weight' in name:
                    tensor = p.data.cpu().numpy()
                    grad_tensor = p.grad.data.cpu().numpy()
                    grad_tensor = np.where(tensor < EPS, 0, grad_tensor)
                    p.grad.data = torch.from_numpy(grad_tensor).to(args.device)
        optimizer.step()

    train_loss /= len(train_loader.dataset)
    train_accuracy = 100. * correct / len(train_loader.dataset)

    return train_loss, train_accuracy


def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(args.device), target.to(args.device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)

    model.train()
    return test_loss, accuracy


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=1.2e-3, type=float, help="Learning rate")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--end_epoch", default=10, type=int)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--valid_freq", default=1, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prune_type", default="lt", type=str, help="lt | reinit")
    parser.add_argument("--dataset", default="mnist", type=str, help="mnist | cifar10 | fashionmnist | cifar100")
    parser.add_argument("--arch_type", default="fc1", type=str, help="fc1 | lenet5 | alexnet | vgg16 | resnet18 | densenet121")
    parser.add_argument("--prune_percent", default=90, type=int, help="Pruning percent")
    parser.add_argument("--prune_iterations", default=2, type=int, help="Pruning iterations count")
    parser.add_argument("--train_type", default="lt", type=str, help="lt | regular")
    parser.add_argument("--co2_tracking_mode", action="store_true")

    args = parser.parse_args()
    args.logs = defaultdict(OrderedDict)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.train_type == "regular":
        args.prune_percent = 0     # single iteration (no pruning)
        args.prune_iterations = 1  # No pruning with regular training
    print(args.device)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1" if args.device == "cuda" else "0"  # args.gpu

    # FIXME resample
    resample = False

    # Looping Entire process
    for i in range(0, 3):
        main(args, ITE=i)
