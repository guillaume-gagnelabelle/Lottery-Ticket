import time
import torch
import os
from collections import OrderedDict
import matplotlib.pyplot as plt

os.chdir("C:/Users/traor/Lottery-Ticket" )

logs = ["logs_lt_pp68x3", "logs_lt_pp90x2", "logs_regular_pp90x1"]
x = []
for log in logs:
    for seed in range(3):
        x.append(torch.load(f"{os.getcwd()}/saves/fc1/mnist/{log}_{seed}.pt"))


# x[0] contains multiple OrderDict, i.e. multiples dictionaries, of the "logs_lt_pp68x3_0.pt" file. These are the different available dictionnaries:

# print(x[0]["time"])

# print(x[0]["non_zeros_weights"])
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

experiments = ["lt_pp68x3", "lt_pp90x2", "regular_pp90x1"]
metrics = ["non_zeros_weights", "co2", "test_loss", "test_accuracy", "train_loss", "train_accuracy"] 

for metric in metrics:
    my_dict_list = []

    for i in range(len(experiments)):
        my_dict_list.append(x[i][metric])

    # Define x axis values (keys of the OrderedDict)
    x_values = list(my_dict_list[0].keys())

    # # Loop through each OrderedDict in my_dict_list
    for k, my_dict in enumerate(my_dict_list):
        # Define y axis values (values of the OrderedDict)
        y_values = list(my_dict.values())
        # Divide y_values into groups of 10
        grouped_y_values = [y_values[i:i+10] for i in range(0, len(y_values), 10)]
        # Plot each group of 10 values as a line
        for i in range(len(grouped_y_values)):
            plt.plot(x_values[i*10:(i+1)*10], grouped_y_values[i], label=f"{experiments[k]}, Seed {i}")

    plt.title(f"{metric} vs. Number of Images Seen")
    plt.xlabel("Number of Images Seen")
    plt.ylabel(metric)
    plt.legend()
    plt.gcf().set_size_inches(15, 10)
    # Save the plot to a file
    plt.savefig(f"plots/{metric}_vs_nb_images.png")
    # Clear the plot for the next iteration
    plt.clf()

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
