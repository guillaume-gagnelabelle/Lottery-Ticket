import argparse
import copy
import os
import sys
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import os
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import seaborn as sns
import torch.nn.init as init
import pickle

#ANCHOR Custom Libraries
import utils
from data import data_utils
from archs import archs_utils

writer = SummaryWriter()

sns.set_style('darkgrid')


def main(args, ITE=0):
    reinit = True if args.prune_type=="reinit" else False

    train_loader, test_loader = data_utils.getData(args)

    model = archs_utils.getModel(args)

    #ANCHOR Weight Initialization
    model.apply(archs_utils.weight_init)

    #ANCHOR Copying and Saving Initial State
    initial_state_dict = copy.deepcopy(model.state_dict())
    utils.checkdir(f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/")
    torch.save(model, f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/initial_state_dict_{args.prune_type}.pth.tar")

    #ANCHOR Making Initial Mask
    mask = archs_utils.make_mask(model)

    #ANCHOR Optimizer and Loss
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss() # Default was F.nll_loss

    #ANCHOR Layer Looper
    for name, param in model.named_parameters():
        print(name, param.size())

    #ANCHOR Pruning
    # NOTE First Pruning Iteration is of No Compression
    bestacc = 0.0
    best_accuracy = 0
    ITERATION = args.prune_iterations
    comp = np.zeros(ITERATION,float)
    bestacc = np.zeros(ITERATION,float)
    all_loss = np.zeros(args.end_iter,float)
    all_accuracy = np.zeros(args.end_iter,float)


    for _ite in range(args.start_iter, ITERATION):
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

        #ANCHOR Print the table of Nonzeros in each layer
        comp1 = utils.print_nonzeros(model)
        comp[_ite] = comp1
        pbar = tqdm(range(args.end_iter))

        for iter_ in pbar:

            #ANCHOR Frequency for Testing
            if iter_ % args.valid_freq == 0:
                accuracy = test(model, test_loader, criterion)

                #ANCHOR Save Weights
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    utils.checkdir(f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/")
                    torch.save(model,f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/{_ite}_model_{args.prune_type}.pth.tar")

            #ANCHOR Training
            loss = train(model, train_loader, optimizer, criterion)
            all_loss[iter_] = loss
            all_accuracy[iter_] = accuracy
            
            #ANCHOR Frequency for Printing Accuracy and Loss
            if iter_ % args.print_freq == 0:
                pbar.set_description(
                    f'Train Epoch: {iter_}/{args.end_iter} Loss: {loss:.6f} Accuracy: {accuracy:.2f}% Best Accuracy: {best_accuracy:.2f}%')       

        writer.add_scalar('Accuracy/test', best_accuracy, comp1)
        bestacc[_ite]=best_accuracy

        #ANCHOR Plotting Loss (Training), Accuracy (Testing), Iteration Curve
        #NOTE Loss is computed for every iteration while Accuracy is computed only for every {args.valid_freq} iterations. Therefore Accuracy saved is constant during the uncomputed iterations.
        #NOTE Normalized the accuracy to [0,100] for ease of plotting.
        plt.plot(np.arange(1,(args.end_iter)+1), 100*(all_loss - np.min(all_loss))/np.ptp(all_loss).astype(float), c="blue", label="Loss") 
        plt.plot(np.arange(1,(args.end_iter)+1), all_accuracy, c="red", label="Accuracy") 
        plt.title(f"Loss Vs Accuracy Vs Iterations ({args.dataset},{args.arch_type})") 
        plt.xlabel("Iterations") 
        plt.ylabel("Loss and Accuracy") 
        plt.legend() 
        plt.grid(color="gray") 
        utils.checkdir(f"{os.getcwd()}/plots/lt/{args.arch_type}/{args.dataset}/")
        plt.savefig(f"{os.getcwd()}/plots/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_LossVsAccuracy_{comp1}.png", dpi=1200) 
        plt.close()

        #ANCHOR Dump Plot values
        utils.checkdir(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/")
        all_loss.dump(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_all_loss_{comp1}.dat")
        all_accuracy.dump(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_all_accuracy_{comp1}.dat")
        
        #ANCHOR Dumping mask
        utils.checkdir(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/")
        with open(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_mask_{comp1}.pkl", 'wb') as fp:
            pickle.dump(mask, fp)
        
        #ANCHOR Making variables into 0
        best_accuracy = 0
        all_loss = np.zeros(args.end_iter,float)
        all_accuracy = np.zeros(args.end_iter,float)

    #ANCHOR Dumping Values for Plotting
    utils.checkdir(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/")
    comp.dump(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_compression.dat")
    bestacc.dump(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_bestaccuracy.dat")

    #ANCHOR Plotting
    a = np.arange(args.prune_iterations)
    plt.plot(a, bestacc, c="blue", label="Winning tickets") 
    plt.title(f"Test Accuracy vs Unpruned Weights Percentage ({args.dataset},{args.arch_type})") 
    plt.xlabel("Unpruned Weights Percentage") 
    plt.ylabel("test accuracy") 
    plt.xticks(a, comp, rotation ="vertical") 
    plt.ylim(0,100)
    plt.legend() 
    plt.grid(color="gray") 
    utils.checkdir(f"{os.getcwd()}/plots/lt/{args.arch_type}/{args.dataset}/")
    plt.savefig(f"{os.getcwd()}/plots/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_AccuracyVsWeights.png", dpi=1200) 
    plt.close()                    
   
#ANCHOR Function for Training
def train(model, train_loader, optimizer, criterion):
    EPS = 1e-6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    for batch_idx, (imgs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        #imgs, targets = next(train_loader)
        imgs, targets = imgs.to(device), targets.to(device)
        output = model(imgs)
        train_loss = criterion(output, targets)
        train_loss.backward()

        #ANCHOR Freezing Pruned weights by making their gradients Zero
        for name, p in model.named_parameters():
            if 'weight' in name:
                tensor = p.data.cpu().numpy()
                grad_tensor = p.grad.data.cpu().numpy()
                grad_tensor = np.where(tensor < EPS, 0, grad_tensor)
                p.grad.data = torch.from_numpy(grad_tensor).to(device)
        optimizer.step()
    return train_loss.item()

#ANCHOR Function for Testing
def test(model, test_loader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
    return accuracy


if __name__=="__main__":
    
    #from gooey import Gooey
    #@Gooey      
    
    #ANCHOR Arguement Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr",default= 1.2e-3, type=float, help="Learning rate")
    parser.add_argument("--batch_size", default=60, type=int)
    parser.add_argument("--start_iter", default=0, type=int)
    parser.add_argument("--end_iter", default=100, type=int)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--valid_freq", default=1, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prune_type", default="lt", type=str, help="lt | reinit")
    # parser.add_argument("--gpu", default="0", type=str)
    parser.add_argument("--dataset", default="mnist", type=str, help="mnist | cifar10 | fashionmnist | cifar100")
    parser.add_argument("--arch_type", default="fc1", type=str, help="fc1 | lenet5 | alexnet | vgg16 | resnet18 | densenet121")
    parser.add_argument("--prune_percent", default=10, type=int, help="Pruning percent")
    parser.add_argument("--prune_iterations", default=35, type=int, help="Pruning iterations count")
    
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]= "1" if args.device == "cuda" else "0"  # args.gpu
    
    
    #FIXME resample
    resample = False

    #ANCHOR Looping Entire process
    #for i in range(0, 5):
    main(args, ITE=1)
