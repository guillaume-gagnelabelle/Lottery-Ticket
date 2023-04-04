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
import matplotlib.pyplot as plt
import pandas as pd
import numpy.ma as ma
from itertools import zip_longest

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="mnist", type=str, help="mnist | cifar10")
parser.add_argument("--arch_type", default="fc1", type=str, help="fc1 | lenet5")

args = parser.parse_args()
# args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


logs = ["logs_NEW_lt_pp68x3", "logs_NEW_lt_pp90x2", "logs_NEW_regular_pp0x1"]
logs = ["logs_NEW_lt_pp68x3", "logs_NEW_regular_pp0x1"]
legends = ["Élagage 2x68%", "Élagage 90%", "Sans élagage"]
legends = ["logs_NEW_lt_pp68x3", "Sans élagage"]

metrics = ["duration", "emissions", "emissions_rate", "cpu_power", "gpu_power", "ram_power", "cpu_energy", "gpu_energy", "ram_energy", "energy_consumed"]
y_titles = ["Durée [s]", "Émissions de CO2 [kg]", "Taux d'émissions de CO2 [kg/s]", "Puissance CPU [W]", "Puissance GPU [W]", "Puissance RAM [W]", "Énergie CPU [kWh]", "Énergie GPU [kWh]", "Énergie RAM [kWh]", "Énergie consommée [kWh]"]
seeds = [0, 1, 2, 3, 4]

# Plots vs nb_seen_images
for idx, metric in enumerate(metrics):
    plt.figure()
    for i, log in enumerate(logs):
        x = []
        y = []
        for seed in seeds:
            emissions = pd.read_csv(f"saves/{args.arch_type}/{args.dataset}/new_run_v2/{log}_seed{seed}_co2True_{args.dataset}.csv").to_dict()
            x.append(list(emissions[metric].keys()))
            y.append(list(emissions[metric].values()))

        x = np.nanmean(np.array(list(zip_longest(*x)), dtype=float), axis=1) * 50000
        y_mean = np.nanmean(np.array(list(zip_longest(*y)), dtype=float), axis=1)
        y_std = np.nanstd(np.array(list(zip_longest(*y)), dtype=float), axis=1, ddof=1)

        plt.plot(x, y_mean, label=legends[i])
        plt.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.3)
        print(metric, ": ", np.array(y_mean).mean(), "±", np.array(y_std).mean())
    plt.ylabel(y_titles[idx])
    plt.xlabel("Nombre d'images d'entraînement")
    plt.grid()
    plt.legend()
    plt.show()

# Plots vs time
for idx, metric in enumerate(metrics):
    plt.figure()
    for i, log in enumerate(logs):
        x = []
        y = []
        for seed in seeds:
            emissions = pd.read_csv(f"saves/{args.arch_type}/{args.dataset}/new_run_v2/{log}_seed{seed}_co2True_{args.dataset}.csv").to_dict()
            x.append(list(emissions["duration"].values()))
            y.append(list(emissions[metric].values()))
        x = np.nanmean(np.array(list(zip_longest(*x)), dtype=float), axis=1)
        y_mean = np.nanmean(np.array(list(zip_longest(*y)), dtype=float), axis=1)
        y_std = np.nanstd(np.array(list(zip_longest(*y)), dtype=float), axis=1, ddof=1)
        plt.plot(x, y_mean, label=legends[i])
        plt.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.3)
        print(metric, ": ", np.array(y_mean).mean(), "±", np.array(y_std).mean())

    plt.ylabel(y_titles[idx])
    plt.xlabel("Durée [s]")
    plt.grid()
    plt.legend()
    plt.show()
