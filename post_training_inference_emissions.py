import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="mnist", type=str, help="mnist | cifar10")
parser.add_argument("--arch_type", default="fc1", type=str, help="fc1 | lenet5")
args = parser.parse_args()


# logs = ["inference_lt_pp68x3", "inference_lt_pp90x2", "inference_regular_pp0x1"]
logs = ["inference_SPARSE_lt_pp90x2", "inference_SPARSE_regular_pp0x1"]
seeds = [0, 1, 2, 3, 4]
metrics = ["duration", "emissions", "emissions_rate", "cpu_power", "gpu_power", "ram_power", "cpu_energy", "gpu_energy",
           "ram_energy", "energy_consumed"]
y_titles = ["Durée [s]", "Émissions de CO2 [kg]", "Taux d'émissions de CO2 [kg/s]", "Puissance CPU [W]", "Puissance GPU [W]", "Puissance RAM [W]", "Énergie CPU [kWh]", "Énergie GPU [kWh]", "Énergie RAM [kWh]", "Énergie consommée [kWh]"]
# legends = ["Élagage 2x68%", "Élagage 90%", "Sans élagage"]
legends = ["Élagage 90%", "Sans élagage"]


# Plots vs nb_seen_images
for idx, metric in enumerate(metrics):
    plt.figure()
    for i, log in enumerate(logs):
        y = []
        for seed in seeds:
            emissions = pd.read_csv(f"saves/{args.arch_type}/{args.dataset}/inference_sparse/{log}_seed{seed}.csv").to_dict()
            if len(y) == 0: x = list(emissions[metric].keys())[:-1]
            y.append(list(emissions[metric].values())[:-1])
        x = np.array(x)*50000
        y = np.array(y)
        print(y.shape)
        plt.plot(x, y.mean(0), label=legends[i])
        plt.fill_between(x, y.mean(0) - y.std(0), y.mean(0) + y.std(0), alpha=0.3)
    plt.ylabel(y_titles[idx])
    plt.xlabel("nombre d'inférences")
    plt.grid()
    plt.legend()
    plt.show()

# Plots vs time
for idx, metric in enumerate(metrics):
    plt.figure()
    for i, log in enumerate(logs):
        y = []
        for seed in seeds:
            emissions = pd.read_csv(f"saves/{args.arch_type}/{args.dataset}/inference_sparse/{log}_seed{seed}.csv").to_dict()
            if len(y) == 0: x = list(emissions["duration"].values())[:-1]
            y.append(list(emissions[metric].values())[:-1])
        y = np.array(y)
        plt.plot(x, y.mean(0), label=legends[i])
        plt.fill_between(x, y.mean(0) - y.std(0), y.mean(0) + y.std(0), alpha=0.3)
    plt.ylabel(y_titles[idx])
    plt.xlabel("Durée [s]")
    plt.grid()
    plt.legend()
    plt.show()
