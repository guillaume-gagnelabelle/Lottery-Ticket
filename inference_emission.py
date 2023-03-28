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

logs = ["inference_lt_pp68x3", "inference_lt_pp90x2", "inference_regular_pp0x1"]
legends = ["lt_pp68x3", "lt_pp90x2", "regular_pp0x1"]
seeds = [0, 1, 2]
metrics = ["duration", "emissions", "emissions_rate", "cpu_power", "gpu_power", "ram_power", "cpu_energy", "gpu_energy",
           "ram_energy", "energy_consumed"]
units = [" [s]", " [kg]", " [kg/s]", " [W]", " [W]", " [W]", " [kW]", " [kW]", " [kW]", " [kW]"]

# Plots vs nb_seen_images
for idx, metric in enumerate(metrics):
    plt.figure()
    for log in logs:
        y = []
        for seed in seeds:
            emissions = pd.read_csv(f"saves/fc1/mnist/{log}_seed{seed}.csv").to_dict()
            if len(y) == 0: x = list(emissions[metric].keys())[:-1]
            y.append(list(emissions[metric].values())[:-1])
        y = np.array(y)
        plt.plot(x, y.mean(0), label=log)
        plt.fill_between(x, y.mean(0) - y.std(0), y.mean(0) + y.std(0), alpha=0.3)
    plt.ylabel(metric + units[idx])
    plt.xlabel("nb_inference [x50 000]")
    plt.grid()
    plt.title(metric)
    plt.legend()
    plt.show()

# Plots vs time
for idx, metric in enumerate(metrics):
    plt.figure()
    for log in logs:
        y = []
        for seed in seeds:
            emissions = pd.read_csv(f"saves/fc1/mnist/{log}_seed{seed}.csv").to_dict()
            if len(y) == 0: x = list(emissions["duration"].values())[:-1]
            y.append(list(emissions[metric].values())[:-1])
        y = np.array(y)
        plt.plot(x, y.mean(0), label=log)
        plt.fill_between(x, y.mean(0) - y.std(0), y.mean(0) + y.std(0), alpha=0.3)
    plt.ylabel(metric + units[idx])
    plt.xlabel("duration [s]")
    plt.grid()
    plt.title(metric)
    plt.legend()
    plt.show()
