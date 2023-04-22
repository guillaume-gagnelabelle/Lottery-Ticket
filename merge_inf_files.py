import os
import csv


directory = 'saves/fc1/mnist/final_inference_v2'
combined_file = 'combined.csv'

# List to store the last row of each CSV file, along with the filename
last_rows = []

# Loop through each CSV file in the directory
for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        with open(os.path.join(directory, filename), 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            # Iterate through each row of the file
            for row in csv_reader:
                # Extract columns E to L and store them in last_row
                last_row = row[4:12]
                # Add the filename as the first element of last_row
                last_row.insert(0, filename)
            # Append the last row to the list
            last_rows.append(last_row)

# Write the combined file
with open(combined_file, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    # Write the header row
    csv_writer.writerow(['Filename', "emissions", "emissions_rate", "cpu_power", "gpu_power", "ram_power", "cpu_energy", "gpu_energy", "ram_energy"])
    # Write the last row of each file, along with the filename
    for row in last_rows:
        csv_writer.writerow(row)

