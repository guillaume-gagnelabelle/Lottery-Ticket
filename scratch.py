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


# print(x[0]["train_loss"].keys())
# print(x[1]["train_loss"].keys())
# print(x[2]["train_loss"].keys())

# length = len(x[2]["train_loss"])
# print(length)
# # print(x[0])

# Every dictionnary contains multiples ordered (key, value) pairs such that the key is an integer: the number of training images seen
# and the value is the defined variables: time, test_loss, acc, etc.
# Thus, the plots can be easily implemented: plt.plot(x[0]["metric"].keys(), x[0]["metric"].values()). This will
# plot the relevant metric with respect to the number of images seen.
# The metric vs time plot is trickier, but feasible. Be aware that the "time", "non_zeros_weight


# if __name__ == "__main__":

#     import matplotlib.pyplot as plt
#     from collections import OrderedDict

#     # Create an example OrderedDict
#     my_dict = OrderedDict([(60000, 0.0), (120000, 0.0), (180000, 0.0), (240000, 0.0), (300000, 0.0), (360000, 0.0), (420000, 0.0), (480000, 0.0), (540000, 0.0), (600000, 0.0), (660000, 0.0), (720000, 0.0), (780000, 0.0), (840000, 0.0), (900000, 0.0), (960000, 0.0), (1020000, 0.0), (1080000, 0.0), (1140000, 0.0), (1200000, 0.0), (1260000, 0.0), (1320000, 0.0), (1380000, 0.0), (1440000, 0.0), (1500000, 0.0), (1560000, 0.0), (1620000, 0.0), (1680000, 0.0), (1740000, 0.0), (1800000, 0.0)])

#     # Split the OrderedDict into 3 parts
#     part1 = list(my_dict.items())[:10]
#     part2 = list(my_dict.items())[10:20]
#     part3 = list(my_dict.items())[20:]

#     # Extract the keys and values for each part
#     part1_keys, part1_values = zip(*part1)
#     part2_keys, part2_values = zip(*part2)
#     part3_keys, part3_values = zip(*part3)

#     # Create a plot with 3 lines
#     plt.plot(part1_keys, part1_values, label='Part 1')
#     plt.plot(part2_keys, part2_values, label='Part 2')
#     plt.plot(part3_keys, part3_values, label='Part 3')

#     # Add a title, axis labels, and legend
#     plt.title('Training Loss vs. Number of Images Seen')
#     plt.xlabel('Number of Images Seen')
#     plt.ylabel('Training Loss')
#     plt.legend()

#     # Display the plot
#     plt.show()

my_dict_list_weights = []

for i in range(3):
    my_dict_list_weights.append(x[i]["non_zeros_weights"])

# Define x axis values (keys of the OrderedDict)
x_values = list(my_dict_list_weights[0].keys())

# Loop through each OrderedDict in my_dict_list
for my_dict in my_dict_list_weights:
    # Define y axis values (values of the OrderedDict)
    y_values = list(my_dict.values())
    # Divide y_values into groups of 10
    grouped_y_values = [y_values[i:i+10] for i in range(0, len(y_values), 10)]
    # Plot each group of 10 values as a line
    for i in range(len(grouped_y_values)):
        plt.plot(x_values[i*10:(i+1)*10], grouped_y_values[i], label=f"Experiment {my_dict_list_weights.index(my_dict)+1}, Set {i+1}")

# Set plot title, x label, y label, and legend
plt.title("Number of non-zero weights vs. Number of Images Seen")
plt.xlabel("Number of Images Seen")
plt.ylabel("Number of non-zero weights")
plt.legend()

# Show the plot
plt.show()
