# import numpy as np
# import matplotlib.pyplot as plt

# # Function to find the intersection
# def find_intersection(x, y1, y2):
#     for i in range(len(x) - 1):
#         if (y1[i] - y2[i]) * (y1[i + 1] - y2[i + 1]) <= 0:
#             return x[i]
#     return None

# # Define the x range
# x = np.linspace(0, 700000, 1000)

# # Define the linear equations and their uncertainties
# pp68x3 = 0.0016651689157558323 + 6.49e-05/50000 * x
# pp68x3_error = 1.8595905139006065e-05 + 1.45e-06/50000 * x

# pp90x2 = 0.001100812432392153 + 8.06e-05/50000 * x
# pp90x2_error = 1.4643315207477953e-05 + 7.16e-06/50000 * x

# pp0x1 = 0.00053387075448972 + 0.000171/50000 * x
# pp0x1_error = 5.0089333010023565e-06 + 1.77e-05/50000 * x

# # Plot the linear equations
# plt.plot(x, pp68x3, label="Élagage 2x68%")
# plt.plot(x, pp90x2, label="Élagage 90%")
# plt.plot(x, pp0x1, label="Sans élagage")

# # Fill the area between y - error and y + error
# plt.fill_between(x, pp68x3 - pp68x3_error, pp68x3 + pp68x3_error, alpha=0.3)
# plt.fill_between(x, pp90x2 - pp90x2_error, pp90x2 + pp90x2_error, alpha=0.3)
# plt.fill_between(x, pp0x1 - pp0x1_error, pp0x1 + pp0x1_error, alpha=0.3)

# # Add labels and title
# plt.xlabel("Nombre d'images")
# plt.ylabel("Émissions en CO2 (Kg)")

# # Show the legend
# plt.legend()

# # Display the plot
# plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Function to find the intersection
def find_intersection(x, y1, y2):
    for i in range(len(x) - 1):
        if (y1[i] - y2[i]) * (y1[i + 1] - y2[i + 1]) <= 0:
            return x[i]
    return None

# Define the x range
x = np.linspace(0, 700000, 1000)

# Define the linear equations and their uncertainties
pp68x3 = 0.0016651689157558323 + 6.49e-05/50000 * x
pp68x3_error = 1.8595905139006065e-05 + 1.45e-06/50000 * x

pp90x2 = 0.001100812432392153 + 8.06e-05/50000 * x
pp90x2_error = 1.4643315207477953e-05 + 7.16e-06/50000 * x

pp0x1 = 0.00053387075448972 + 0.000171/50000 * x
pp0x1_error = 5.0089333010023565e-06 + 1.77e-05/50000 * x

# Find intersections
intersection_68x3_max = find_intersection(x, pp0x1 - pp0x1_error, pp68x3 + pp68x3_error)
intersection_68x3_no_incert = find_intersection(x, pp0x1, pp68x3)
intersection_68x3_min = find_intersection(x, pp0x1 - pp0x1_error, pp68x3 - pp68x3_error)

intersection_90x2_min = find_intersection(x, pp0x1 + pp0x1_error, pp90x2 - pp90x2_error)
intersection_90x2_no_incert = find_intersection(x, pp0x1, pp90x2)
intersection_90x2_max = find_intersection(x, pp0x1 - pp0x1_error, pp90x2 + pp90x2_error)

# Print intersection points
print("intersection_68x3_min:", intersection_68x3_min)
print("intersection_68x3_no_incert:", intersection_68x3_no_incert)
print("intersection_68x3_max:", intersection_68x3_max)

print('================================')

print("intersection_90x2_min:", intersection_90x2_min)
print("intersection_90x2_no_incert:", intersection_90x2_no_incert)
print("intersection_90x2_max:", intersection_90x2_max)

# Plot the linear equations
plt.plot(x, pp68x3, label="Élagage 2x68%")
plt.plot(x, pp90x2, label="Élagage 90%")
plt.plot(x, pp0x1, label="Sans élagage")

# Fill the area between y - error and y + error
plt.fill_between(x, pp68x3 - pp68x3_error, pp68x3 + pp68x3_error, alpha=0.3)
plt.fill_between(x, pp90x2 - pp90x2_error, pp90x2 + pp90x2_error, alpha=0.3)
plt.fill_between(x, pp0x1 - pp0x1_error, pp0x1 + pp0x1_error, alpha=0.3)

# Add labels and title
plt.xlabel("Nombre d'images")
plt.ylabel("Émissions en CO2 (Kg)")

# Show the legend
plt.legend()

# Display the plot
plt.show()



