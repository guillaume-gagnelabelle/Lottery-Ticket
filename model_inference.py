import argparse
import copy
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
from codecarbon import EmissionsTracker
from data.data_utils import getData
from archs.archs_utils import getModel
import utils


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
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--dataset", default="mnist", type=str, help="mnist | cifar10 | fashionmnist | cifar100")
    parser.add_argument("--arch_type", default="fc1", type=str, help="fc1 | lenet5 | alexnet | vgg16 | resnet18 | densenet121")

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    project = "energy_per_image_"
    tracker = EmissionsTracker(project_name=project,
                               measure_power_secs=1,
                               tracking_mode="process",
                               log_level="critical",
                               save_to_logger=True
                               )
    tracker.start()
    start = time.time()

    model_pruned_90x2 = torch.load(f"{os.getcwd()}/saves/fc1/mnist/logs_lt_pp90x2_0.pt")["final_state_dict"]
    model_pruned_68x3 = torch.load(f"{os.getcwd()}/saves/fc1/mnist/logs_lt_pp68x3_0.pt")["final_state_dict"]
    model_pruned_90x1 = torch.load(f"{os.getcwd()}/saves/fc1/mnist/logs_regular_pp90x1_0.pt")["final_state_dict"]  # not pruned

    data_loader, _, _ = getData(args, train_percent=1.0, val_percent=0.0)
    criterion = nn.CrossEntropyLoss()

    model = getModel(args)
    for model_state in [model_pruned_68x3, model_pruned_90x2, model_pruned_90x1]:
        model.load_state_dict(model_state)
        model.eval()

        test_loss = np.zeros(5)
        test_acc = np.zeros(5)
        for i in range(5):
            data_loader, _, _ = getData(args, train_percent=0.2, val_percent=0.0)
            test_loss[i], test_acc[i] = test(model, data_loader, criterion)
        print(test_loss.mean(), "±", test_loss.std())
        print(test_acc.mean(), "±", test_acc.std())
        print()
        tracker.flush()

    tracker.stop()
