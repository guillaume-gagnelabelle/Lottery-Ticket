import time
import torch
import os
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np


logs = ["logs_lt_pp68x3","logs_lt_pp90x2","logs_regular_pp0x1"]
metrics = ["non_zeros_weights", "test_loss", "test_accuracy", "train_loss", "train_accuracy", "val_loss", "val_accuracy"]
units = [" %", "", " %", "", " %", "", " %"]
legends = ["lt_pp68x3", "lt_pp90x2", "regular_pp0x1"]
seeds = [0]

logs = ["lr0.01_wd1e-05", "lr0.001_wd1e-05", "lr0.0001_wd1e-05", "lr1e-05_wd1e-05"]
legends = logs

nb_seen_images = []
times = []
ys_mean = []
ys_std = []
for log in logs:
    dicts = []
    for seed in seeds:
        # dicts.append(torch.load(f"{os.getcwd()}/saves/fc1/mnist/{log}_seed{seed}_co2False_lr0.0012_wd0.0001.pt"))
        dicts.append(torch.load(f"{os.getcwd()}/saves/fc1/mnist/logs_regular_pp0x1_seed{seed}_co2False_{log}.pt", map_location=torch.device('cpu')))

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
    plt.ylabel(metric + units[idx])
    plt.xlabel("nb_seen_images")
    for i in range(0, len(logs)):
        plt.plot(nb_seen_images[idx + i * len(metrics)], ys_mean[idx + i * len(metrics)], label=legends[i])
        plt.fill_between(nb_seen_images[idx + i * len(metrics)], ys_mean[idx + i * len(metrics)] - ys_std[idx + i * len(metrics)],
                         ys_mean[idx + i * len(metrics)] + ys_std[idx + i * len(metrics)],
                         alpha=0.7)
    plt.grid()
    plt.legend()
    plt.show()

# Plots of time
for idx, metric in enumerate(metrics):
    plt.figure()
    plt.ylabel(metric + units[idx])
    plt.xlabel("time [s]")
    for i in range(0, len(logs)):
        plt.plot(times[idx + i * len(metrics)], ys_mean[idx + i * len(metrics)], label=legends[i])
        plt.fill_between(times[idx + i * len(metrics)], ys_mean[idx + i * len(metrics)] - ys_std[idx + i * len(metrics)],
                         ys_mean[idx + i * len(metrics)] + ys_std[idx + i * len(metrics)],
                         alpha=0.7)
    plt.grid()
    plt.legend()
    plt.show()




# x[0] contains multiple OrderDict, i.e. multiples dictionaries, of the "logs_lt_pp68x3_0.pt" file. These are the different available dictionnaries:


# print(x[0]["time"])
# print(x[0]["co2"])
# print(x[0]["test_loss"])
# print(x[0]["test_accuracy"])
# print(x[0]["train_loss"])
# print(x[0]["train_accuracy"])
# print(x[0]["initial_state_dict"])
# print(x[0]["best_state_dict"])
# print(x[0]["final_state_dict"])

# plt.plot(x[0]["test_loss"].keys(), x[0]["test_loss"].values())
# plt.show()

# experiments = ["lt_pp68x3", "lt_pp90x2", "regular_pp90x1"]
# metrics = ["non_zeros_weights", "co2", "test_loss", "test_accuracy", "train_loss", "train_accuracy"]
#
# # print(x)
# for metric in metrics:
#     my_dict_list = []
#
#     for i in range(len(experiments)):
#         my_dict_list.append(dicts[i][metric])
#
#     # print(my_dict_list)
#
#     # Loop through each OrderedDict in my_dict_list
#     for k, my_dict in enumerate(my_dict_list):
#         # Define x & y axis values (values of the OrderedDict)
#         y_values = my_dict.values().numpy()
#         x_values = my_dict.keys().numpy()
#
#     plt.title(f"{metric} vs. Number of Images Seen")
#     plt.xlabel("Number of Images Seen")
#     plt.ylabel(metric)
#     plt.legend()
#     plt.gcf().set_size_inches(15, 10)
#     # Save the plot to a file
#     plt.savefig(f"plots/{metric}_vs_nb_images.png")
#     # Clear the plot for the next iteration
#     plt.clf()

# Code for plotting the metrics vs time - Need to figure out
# for metric in metrics:
#     plt.figure(figsize=(10,6))
#     plt.title(f"{metric} vs Time")
#     plt.xlabel("Time (s)")
#     plt.ylabel(metric)

#     my_dict_list_metric = []
#     for i in range(3):
#         my_dict_list_metric.append(x[i][metric])

#     x_values = list(x[0]["time"].values())
    
#     for k, my_dict in enumerate(my_dict_list_metric):
#         y_values = list(my_dict.values())
#         grouped_y_values = [list(y_values[i:i+10]) for i in range(0, len(y_values), 10)]
#         for i in range(len(grouped_y_values)):
#             plt.plot(x_values[i*10:(i+1)*10], grouped_y_values[i], label=f"{experiments[k]}, Seed {i}")

#     plt.legend()
#     plt.savefig(f"plots/{metric}_vs_time.png")
#     plt.show()
