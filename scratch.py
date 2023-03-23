import time
import torch
import os


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

print(x[0]["train_loss"])

# Every dictionnary contains multiples ordered (key, value) pairs such that the key is an integer: the number of training images seen
# and the value is the defined variables: time, test_loss, acc, etc.
# Thus, the plots can be easily implemented: plt.plot(x[0]["metric"].keys(), x[0]["metric"].values()). This will
# plot the relevant metric with respect to the number of images seen.
# The metric vs time plot is trickier, but feasible. Be aware that the "time", "non_zeros_weight

