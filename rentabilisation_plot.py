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

# Define the linear equations
pp68x3 = 0.0016651689157558323 + 6.49e-05/50000 * x
pp90x2 = 0.001100812432392153 + 8.06e-05/50000 * x
pp0x1 = 0.00053387075448972 + 0.000171/50000 * x

# Calculate intersection points
intersection_pp68x3_pp90x2 = find_intersection(x, pp68x3, pp90x2)
intersection_pp68x3_pp0x1 = find_intersection(x, pp68x3, pp0x1)
intersection_pp90x2_pp0x1 = find_intersection(x, pp90x2, pp0x1)

# Plot the linear equations
plt.plot(x, pp68x3, label="Élagage 2x68%")
plt.plot(x, pp90x2, label="Élagage 90%")
plt.plot(x, pp0x1, label="Sans élagage")

# Plot the intersection points with dotted lines
intersections = [(intersection_pp68x3_pp90x2, pp68x3, pp90x2),
                 (intersection_pp68x3_pp0x1, pp68x3, pp0x1),
                 (intersection_pp90x2_pp0x1, pp90x2, pp0x1)]

for ix, y1, y2 in intersections:
    if ix is not None:
        iy = y1[np.where(x == ix)][0]
    #     plt.scatter(ix, iy, c='red', marker='o')
    #     plt.plot([0, ix], [iy, iy], color='gray', linestyle='dotted')
    #     plt.plot([ix, ix], [0, iy], color='gray', linestyle='dotted')

        # Print the intersection point coordinates
        print(f'Intersection point: ({ix:.2f}, {iy:.2f})')

# Add labels and title
plt.xlabel("Nombre d'images")
plt.ylabel("Émissions en CO2 (Kg)")

# Show the legend
plt.legend()

# Display the plot
plt.show()


