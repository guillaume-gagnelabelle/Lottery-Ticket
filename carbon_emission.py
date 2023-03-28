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

seeds = [0, 1, 2]
metrics = ["duration","emissions","emissions_rate","cpu_power","gpu_power","ram_power","cpu_energy","gpu_energy","ram_energy","energy_consumed"]

for metric in metrics:
    plt.figure()
    x = []
    y = []
    for seed in seeds:
        emissions = pd.read_csv(f"saves/fc1/mnist/logs_lt_pp90x2_{seed}_True.csv").to_dict()
        x.append(list(emissions[metric].keys()))
        y.append(list(emissions[metric].values()))
    y = np.array(y)
    plt.plot(x[0], y.mean(0))
    plt.fill_between(x[0], y.mean(0) - y.std(0), y.mean(0) + y.std(0), alpha=0.3)
    plt.grid()
    plt.title(metric)
    plt.show()
