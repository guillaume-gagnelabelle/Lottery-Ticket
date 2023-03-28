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

emissions = pd.read_csv('emissions.csv').to_dict()
print(emissions.keys())
print(emissions["energy_consumed"])
plt.plot(emissions["energy_consumed"].keys(), emissions["energy_consumed"].values())
plt.grid()
plt.title("energy_consumed")
plt.figure()
plt.grid()
plt.plot(emissions["emissions"].keys(), emissions["emissions"].values())
plt.title("emissions")
plt.figure()
plt.plot(emissions["emissions_rate"].keys(), emissions["emissions_rate"].values())
plt.title("emissions_rate")
plt.grid()
plt.show()
