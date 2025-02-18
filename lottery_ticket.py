import argparse
import copy
from tqdm import tqdm
import torch
import torch.nn as nn
import os
from codecarbon import EmissionsTracker
from collections import defaultdict, OrderedDict
import time
import utils
from data import data_utils
from archs import archs_utils

'''
 @Author: Gagné-Labelle, Guillaume & Finoude, Meriadec 
 - Inspired heavily from https://github.com/rahulvigneswaran/Lottery-Ticket-Hypothesis-in-Pytorch
 @Student number: 20174375 & B9592
 @Date: April, 2023
 @Project: Rentabilisation énergétique des réseaux de neurones - IFT3710 - UdeM
 
 This program is an extension of rahulvigneswaran/Lottery-Ticket-Hypothesis-in-Pytorch. The adaptation adds a carbon
 emission tracker to keep track of the emissions of sparse neural networks. The repository contains a Pytorch 
 implementation of the paper "The Lottery Ticket Hypothesis: Finding Sparse, Trainable, Neural Networks" by Jonathan 
 Frankle and Michael Carbin.
'''


def main(args, ITE=0):


    args.seed = ITE
    args.nb_images_seen = 0
    utils.set_seed(args)
    project = f"pp{args.prune_percent}x{args.prune_iterations}_seed{args.seed}_co2{args.co2_tracking_mode}_{args.dataset}"
    tracker = EmissionsTracker(project_name=project,
                               measure_power_secs=1,
                               tracking_mode="process",
                               log_level="critical",
                               save_to_logger=True,
                               output_dir=f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}",
                               output_file=project+".csv"
                               )
    print(project)
    tracker.start()
    start = time.time()

    train_loader, val_loader, test_loader = data_utils.getData(args)

    model = archs_utils.getModel(args)
    model.apply(archs_utils.weight_init)

    # Copying and Saving Initial State
    initial_state_dict = copy.deepcopy(model.state_dict())

    mask = archs_utils.make_mask(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    criterion = nn.CrossEntropyLoss()

    for name, param in model.named_parameters():
        print(name, param.size())

    # Pruning
    # NOTE First Pruning Iteration is of No Compression
    best_accuracy = 0

    for ite in range(args.prune_iterations):
        if not ite == 0:
            archs_utils.prune_by_percentile(model, mask, args.prune_percent)
            archs_utils.original_initialization(model, mask, initial_state_dict)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
        print(f"\n--- Pruning Level [{ITE}:{ite}/{args.prune_iterations}]: ---")

        # Print the table of Nonzeros in each layer
        compression = utils.print_nonzeros(model)
        args.logs["non_zeros_weights"][args.nb_images_seen] = compression
        pbar = tqdm(range(args.end_epoch))

        for iter_ in pbar:

            # Frequency for Testing
            if iter_ % args.valid_freq == 0:
                if not args.co2_tracking_mode:
                    test_loss, test_accuracy = test(model, test_loader, criterion)
                    val_loss, val_accuracy = test(model, val_loader, criterion)
                    args.logs["val_loss"][args.nb_images_seen] = val_loss
                    args.logs["val_accuracy"][args.nb_images_seen] = val_accuracy
                    args.logs["test_loss"][args.nb_images_seen] = test_loss
                    args.logs["test_accuracy"][args.nb_images_seen] = test_accuracy
                    args.logs["co2"][args.nb_images_seen] = 0

                elif args.co2_tracking_mode:
                    # In co2_tracking_mode, we do not evaluate the performance to not include an emission bias
                    val_loss, val_accuracy, test_loss, test_accuracy = 0, 0, 0, 0
                    args.logs["test_loss"][args.nb_images_seen] = 0
                    args.logs["test_accuracy"][args.nb_images_seen] = 0
                    args.logs["val_loss"][args.nb_images_seen] = 0
                    args.logs["val_accuracy"][args.nb_images_seen] = 0
                    args.logs["co2"][args.nb_images_seen] = tracker.flush()
                args.logs["time"][args.nb_images_seen] = time.time() - start

                # Save Weights
                if test_accuracy >= best_accuracy:
                    best_accuracy = test_accuracy
                    best_state_dict = copy.deepcopy(model.state_dict())

            # -------------------------------- TRAINING CORE | LOTTERY-TICKET ------------------------------------------
            train_loss, train_accuracy = train(model, train_loader, optimizer, criterion, args)
            args.logs["train_loss"][args.nb_images_seen] = train_loss
            args.logs["train_accuracy"][args.nb_images_seen] = train_accuracy
            # ----------------------------------------------------------------------------------------------------------

            # Frequency for Printing Accuracy and Loss
            if iter_ % args.print_freq == 0:
                pbar.set_description(
                    f'Train Epoch: {iter_}/{args.end_epoch} Loss: {test_loss:.6f} Accuracy: {test_accuracy:.2f}% Best Accuracy: {best_accuracy:.2f}%')

        best_accuracy = 0

    # Copying and Saving Final State
    final_state_dict = copy.deepcopy(model.state_dict())
    utils.checkdir(f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/")
    torch.save({
        "time": args.logs["time"],
        "non_zeros_weights": args.logs["non_zeros_weights"],
        "co2": args.logs["co2"],
        "test_loss": args.logs["test_loss"],
        "val_loss": args.logs["val_loss"],
        "train_loss": args.logs["train_loss"],
        "test_accuracy": args.logs["test_accuracy"],
        "train_accuracy": args.logs["train_accuracy"],
        "val_accuracy": args.logs["val_accuracy"],
        "initial_state_dict": initial_state_dict,
        "best_state_dict": best_state_dict,
        "final_state_dict": final_state_dict,
    }, f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/" + project + ".pt"
    )

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
        # output = model.forward_sparse(imgs)
        loss = criterion(output, targets)
        train_loss += loss.item()
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(targets.data.view_as(pred)).sum().item()
        args.nb_images_seen += len(targets)
        loss.backward()

        # --------------------------- PRUNING ------------------------------------
        if args.train_type == "lt":
            # Freezing Pruned weights by making their gradients Zero
            for name, p in model.named_parameters():
                if 'weight' in name:
                    tensor = p.data
                    grad_tensor = p.grad
                    grad_tensor = torch.where(tensor.abs() < EPS, torch.zeros_like(grad_tensor), grad_tensor)
                    p.grad.data = grad_tensor

                    # Convert the pruned weights to a sparse tensor
                    # sparse_weights = sparse.FloatTensor(p.data._indices(), p.data._values(), p.data.size())
                    # p.data = sparse_weights
        # ----------------------- END OF PRUNING ---------------------------------
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
    parser.add_argument("--lr", default=0.001, type=float, help="Learning rate")
    parser.add_argument("--decay", default=0.001, type=float, help="Weight decay")
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--end_epoch", default=32, type=int)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--valid_freq", default=1, type=int)
    parser.add_argument("--dataset", default="mnist", type=str, help="mnist | cifar10")
    parser.add_argument("--arch_type", default="fc1", type=str, help="fc1 | lenet5 | resnet18")
    parser.add_argument("--prune_percent", default=90, type=int, help="Pruning percent")
    parser.add_argument("--prune_iterations", default=2, type=int, help="Pruning iterations count")
    parser.add_argument("--train_type", default="lt", type=str, help="lt | regular")
    parser.add_argument("--co2_tracking_mode", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])

    args = parser.parse_args()
    args.logs = defaultdict(OrderedDict)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.train_type == "regular":
        args.prune_percent = 0     # single iteration (no pruning)
        args.prune_iterations = 1  # No pruning with regular training
    print(args.device)

    # Looping Entire process
    for i in args.seeds:
        main(args, ITE=i)
