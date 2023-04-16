import argparse
import numpy as np
import torch
import torch.nn as nn
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

    return test_loss, accuracy

def test_sparse(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
    
        for data, target in test_loader:
            data, target = data.to(args.device), target.to(args.device)
            output = model.forward_sparse(data)
            test_loss += criterion(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)

    return test_loss, accuracy



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--dataset", default="mnist", type=str, help="mnist | cifar10")
    parser.add_argument("--arch_type", default="fc1", type=str, help="fc1 | lenet5 | resnet18")
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for seed in [0, 1, 2, 3, 4]:

        proj_names = [f"inference_sparse_lt_pp68x3_seed{seed}", f"inference_sparse_lt_pp90x2_seed{seed}", f"inference_sparse_pp0x1_seed{seed}"]
        projects = [f"logs_NEW_lt_pp68x3_seed{seed}", f"logs_NEW_lt_pp90x2_seed{seed}", f"logs_NEW_regular_pp0x1_seed{seed}"]
        model_pruned_68x3 = torch.load(f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/new_run_v2/{projects[0]}_co2False_{args.dataset}.pt", map_location=torch.device(args.device))["final_state_dict"]
        model_pruned_90x2 = torch.load(f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/new_run_v2/{projects[1]}_co2False_{args.dataset}.pt", map_location=torch.device(args.device))["final_state_dict"]
        model_regular_0x1 = torch.load(f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/new_run_v2/{projects[2]}_co2False_{args.dataset}.pt", map_location=torch.device(args.device))["final_state_dict"]  # not pruned

        _, _, data_loader = getData(args)
        criterion = nn.CrossEntropyLoss()

        model = getModel(args).to(args.device)
        for idx, model_state in enumerate([model_pruned_68x3, model_pruned_90x2, model_regular_0x1]):
            tracker = EmissionsTracker(project_name=projects[idx],
                                       measure_power_secs=1,
                                       tracking_mode="process",
                                       log_level="critical",
                                       output_dir=f"saves/{args.arch_type}/{args.dataset}/inference_sparse/",
                                       output_file=proj_names[idx]+".csv",
                                       save_to_logger=True
                                       )
            tracker.start()
            tracker.flush()
            start = time.time()

            model.load_state_dict(model_state)
            model.eval()
            utils.print_nonzeros(model)

            test_loss = np.zeros(10)
            test_acc = np.zeros(10)
            for i in range(10):
                if idx == 2:
                    args.device = "cuda"
                    model.to(args.device)
                    test_loss[i], test_acc[i] = test(model, data_loader, criterion)
                else:
                    args.device = "cpu"
                    model.to(args.device)
                    test_loss[i], test_acc[i] = test_sparse(model, data_loader, criterion)

                tracker.flush()
            print(test_loss.mean(), "±", test_loss.std())
            print(test_acc.mean(), "±", test_acc.std())
            print()

            tracker.stop()