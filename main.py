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

# Custom Libraries
import utils
from data import data_utils
from archs import archs_utils
from plots import plots_utils

writer = SummaryWriter()

sns.set_style('whitegrid')

wandb.login(key="6650aaf8018bf14396b47b6869c885d2156d86c7")


def main(args, ITE=0):
    args.seed = ITE
    args.time = 0  # in unit of the number of training images seen
    utils.set_seed(args)

    # Wandb initialization
    wandb.init(
        project="Lottery-Ticket", 
        entity="ift3710-h23",
        config=args)

    # Carbon tracker initialization
    tracker = EmissionsTracker(project_name="Lottery-Ticket",
                               measure_power_secs=1,
                               tracking_mode="process",
                               save_to_logger=True
                               )
    tracker.start()
    reinit = True if args.prune_type=="reinit" else False

    train_loader, test_loader = data_utils.getData(args)

    model = archs_utils.getModel(args)
    model.apply(archs_utils.weight_init)

    # Copying and Saving Initial State
    initial_state_dict = copy.deepcopy(model.state_dict())
    utils.checkdir(f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/")
    torch.save(initial_state_dict, f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/initial_state_dict_{args.seed}.pt")

    mask = archs_utils.make_mask(model)
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for name, param in model.named_parameters():
        print(name, param.size())

    # Pruning
    # NOTE First Pruning Iteration is of No Compression
    best_accuracy = 0
    ITERATION = args.prune_iterations
    comp = np.zeros(ITERATION,float)
    bestacc = np.zeros(ITERATION,float)
    all_loss = np.zeros(args.end_epoch,float)
    all_accuracy = np.zeros(args.end_epoch,float)


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
        args.logs["non_zeros_weights"][args.time] = comp1
        pbar = tqdm(range(args.end_epoch))

        for iter_ in pbar:

            # Frequency for Testing
            if iter_ % args.valid_freq == 0:
                test_loss, test_accuracy = test(model, test_loader, criterion)
                args.logs["test_loss"][args.time] = test_loss
                args.logs["test_accuracy"][args.time] = test_accuracy

                # Save Weights
                # if test_accuracy > best_accuracy:
                #     best_accuracy = test_accuracy
                #     utils.checkdir(f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/")
                #     torch.save(model,f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/{_ite}_model_{args.prune_type}.pth.tar")

            # ----------------------------------- CORE --------- TRAINING ---------------------------------------------
            if args.train_type == "lt": train_loss, train_accuracy = train_lt(model, train_loader, optimizer, criterion)
            elif args.train_type == "regular": train_loss, train_accuracy = train_reg(model, train_loader, optimizer, criterion)
            args.logs["train_loss"][args.time] = train_loss
            args.logs["train_accuracy"][args.time] = train_accuracy

            # Frequency for Printing Accuracy and Loss
            if iter_ % args.print_freq == 0:
                pbar.set_description(
                    f'Train Epoch: {iter_}/{args.end_epoch} Loss: {test_loss:.6f} Accuracy: {test_accuracy:.2f}% Best Accuracy: {best_accuracy:.2f}%')

        writer.add_scalar('Accuracy/test', best_accuracy, comp1)
        # bestacc[_ite]=best_accuracy

        # Plotting Loss (Training), Accuracy (Testing), Iteration Curve
        #NOTE Loss is computed for every iteration while Accuracy is computed only for every {args.valid_freq} iterations. Therefore Accuracy saved is constant during the uncomputed iterations.
        # plots_utils.plot(args, all_loss, comp1, "Test Loss")
        # plots_utils.plot(args, all_accuracy, comp1, "Test Accuracy")

        # Dump Plot values
        # utils.checkdir(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/")
        # all_loss.dump(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_all_loss_{comp1}.dat")
        # all_accuracy.dump(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_all_accuracy_{comp1}.dat")
        
        # Log metrics from your script to W&B
        # wandb.log({"acc": all_accuracy, "loss": all_loss})

        # Dumping mask
        # utils.checkdir(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/")
        # with open(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_mask_{comp1}.pkl", 'wb') as fp:
        #     pickle.dump(mask, fp)
        
        # Reseting performance variables
        best_accuracy = 0
        all_loss = np.zeros(args.end_epoch, float)
        all_accuracy = np.zeros(args.end_epoch, float)

    # Copying and Saving Final State
    final_state_dict = copy.deepcopy(model.state_dict())
    utils.checkdir(f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/")
    torch.save(final_state_dict, f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/final_state_dict_{args.seed}.pt")


    # Dumping Values for Plotting
    # utils.checkdir(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/")
    # comp.dump(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_compression.dat")
    # bestacc.dump(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_bestaccuracy.dat")

    # plots_utils.final_plot(args, bestacc, comp)

    torch.save({
        "test_loss": args.logs["test_loss"],
        "train_loss": args.logs["train_loss"],
        "test_accuracy": args.logs["test_accuracy"],
        "train_accuracy": args.logs["train_accuracy"],
        "initial_state_dict": initial_state_dict,
        "final_state_dict": final_state_dict,
    })

    # Carbon Emissions
    # tracker.add_metric("Energy Consumption (Joules)", tracker.emissions)
    # tracker.add_metric("CO2 Emissions (kg)", tracker.estimate_carbon_emissions())
    tracker.stop()

    print("Energy Consumption: {} Joules".format(tracker.emissions))
    print("CO2 Emissions: {} kg".format(tracker.estimate_carbon_emissions()))


def train_lt(model, train_loader, optimizer, criterion, args):
    EPS = 1e-6
    model.train()
    train_loss = 0
    correct = 0
    for batch_idx, (imgs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        imgs, targets = imgs.to(args.device), targets.to(args.device)
        output = model(imgs)
        loss = criterion(output, targets)
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(targets.data.view_as(pred)).sum().item()
        args.time += len(targets)
        loss.backward()

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

def train_reg(model, train_loader, optimizer, criterion):
    model.train()
    train_loss = 0
    correct = 0
    for batch_idx, (imgs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        imgs, targets = imgs.to(args.device), targets.to(args.device)
        output = model(imgs)
        loss = criterion(output, targets)
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(targets.data.view_as(pred)).sum().item()
        args.time += len(targets)
        loss.backward()

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


if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr",default= 1.2e-3, type=float, help="Learning rate")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--end_epoch", default=100, type=int)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--valid_freq", default=1, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prune_type", default="lt", type=str, help="lt | reinit")
    # parser.add_argument("--gpu", default="0", type=str)
    parser.add_argument("--dataset", default="mnist", type=str, help="mnist | cifar10 | fashionmnist | cifar100")
    parser.add_argument("--arch_type", default="fc1", type=str, help="fc1 | lenet5 | alexnet | vgg16 | resnet18 | densenet121")
    parser.add_argument("--prune_percent", default=90, type=int, help="Pruning percent")
    parser.add_argument("--prune_iterations", default=2, type=int, help="Pruning iterations count")
    parser.add_argument("--train_type", default="lt", type=str, help="lt | regular")
    
    args = parser.parse_args()
    args.logs = defaultdict(OrderedDict)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if(args.train_type == "regular"): args.prune_iterations = 1  # No pruning with regular training
    print(args.device)

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]= "1" if args.device == "cuda" else "0"  # args.gpu
    
    
    #FIXME resample
    resample = False

    # Looping Entire process
    #for i in range(0, 5):
    main(args, ITE=1)
