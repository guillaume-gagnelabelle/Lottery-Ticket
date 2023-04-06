import time
import torch
import os
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="mnist", type=str, help="mnist | cifar10")
parser.add_argument("--arch_type", default="fc1", type=str, help="fc1 | lenet5")
parser.add_argument("--perf_type", default="lt", type=str, help="lt | hyperparameter")

args = parser.parse_args()
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logs = ["logs_lt_pp68x3", "logs_lt_pp90x2", "logs_regular_pp0x1"]
metrics = ["non_zeros_weights", "test_loss", "test_accuracy", "train_loss", "train_accuracy", "val_loss",
           "val_accuracy"]
units = [" %", "", " %", "", " %", "", " %"]
legends = ["Élagage 2x68%", "Élagage 90%", "Sans élagage"]
seeds = [0]
y_label = ["Paramètres non-nuls [%]", "Erreur de test", "Précision de test  [%]", "Erreur d'entraînement",
           "Précision d'entraînement [%]", "Erreur de validation", "Précision de validation [%]"]

if args.perf_type == "lt":
    logs = ["logs_lt_pp68x3", "logs_lt_pp90x2", "logs_regular_pp0x1"]
    logs = ["logs_NEW_lt_pp68x3", "logs_NEW_lt_pp90x2", "logs_NEW_regular_pp0x1"]
    seeds = [0, 1, 2, 3, 4]
    legends = ["Élagage 2x68%", "Élagage 90%", "Sans élagage"]
else:
    logs = ["lr0.01_wd1e-05", "lr0.001_wd1e-05", "lr0.0001_wd1e-05", "lr1e-05_wd1e-05",
            "lr0.01_wd0.0001", "lr0.001_wd0.0001", "lr0.0001_wd0.0001", "lr1e-05_wd0.0001",
            "lr0.01_wd0.001", "lr0.001_wd0.001", "lr0.0001_wd0.001", "lr1e-05_wd0.001",
            "lr0.01_wd1e-06", "lr0.001_wd1e-06", "lr0.0001_wd1e-06", "lr1e-05_wd1e-06",
            ]
    # logs = ["lr0.001_wd0.001", "lr0.01_wd0.001", "lr0.001_wd1e-06", "lr1e-05_wd1e-06"]
    seeds = [0]
    legends = logs
    # logs = ["logs_regular_pp0x1"]
    # logs = [logs[2]]  # best

nb_seen_images = []
times = []
ys_mean = []
ys_std = []
for log in logs:
    dicts = []
    for seed in seeds:
        if args.perf_type == "lt":
            dicts.append(torch.load(f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/new_run_v2/{log}_seed{seed}_co2False_{args.dataset}.pt", map_location=torch.device('cpu')))
        else:
            dicts.append(torch.load(f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/hyperSearch/logs_regular_pp0x1_seed{seed}_co2False_{log}.pt", map_location=torch.device('cpu')))
    for metric in metrics:

        images = []
        t = []
        y = []

        for seed in seeds:
            images.append(list(dicts[seed][metric].keys()))
            y.append(list(dicts[seed][metric].values()))

        ys_mean.append(np.array(y).mean(0))
        ys_std.append(np.array(y).std(0))
        nb_seen_images.append(images[0])

        for img in images[0]:
            if img in dicts[0]["time"]:
                t.append(dicts[0]["time"][img])
            else:
                t.append(t[-1] + (t[-1] - t[-2]))  # extrapolation
        times.append(t)

# Plots of images seen
for idx, metric in enumerate(metrics):
    plt.figure()
    plt.ylabel(y_label[idx])
    plt.xlabel("Nombre d'images d'entraînement")
    for i in range(0, len(logs)):
        plt.plot(nb_seen_images[idx + i * len(metrics)], ys_mean[idx + i * len(metrics)], label=legends[i])
        plt.fill_between(nb_seen_images[idx + i * len(metrics)],
                         ys_mean[idx + i * len(metrics)] - ys_std[idx + i * len(metrics)],
                         ys_mean[idx + i * len(metrics)] + ys_std[idx + i * len(metrics)],
                         alpha=0.3)
        print(metric, ": ", ys_mean[idx + i * len(metrics)][-1], "±", ys_std[idx + i * len(metrics)][-1])
    plt.grid()
    plt.legend()
    plt.show()

# Plots of time
for idx, metric in enumerate(metrics):
    plt.figure()
    plt.ylabel(y_label[idx])
    plt.xlabel("Temps [s]")
    for i in range(0, len(logs)):
        plt.plot(times[idx + i * len(metrics)], ys_mean[idx + i * len(metrics)], label=legends[i])
        plt.fill_between(times[idx + i * len(metrics)],
                         ys_mean[idx + i * len(metrics)] - ys_std[idx + i * len(metrics)],
                         ys_mean[idx + i * len(metrics)] + ys_std[idx + i * len(metrics)],
                         alpha=0.3)
    plt.grid()
    plt.legend()
    plt.show()
