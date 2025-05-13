import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# File paths
distance_file = '../save/distance_results.txt'

# Initialize lists for table data
epochs = []
mean_distances = []
variance_distances = []

# Read the distance results file
with open(distance_file, 'r') as f:
    for line in f:
        data = json.loads(line)
        epoch = data['epoch']
        distances = data['distances']
        mean_distance = data['mean_distance']
        variance_distance = data['variance_distance']

        # Append to table data
        epochs.append(epoch)
        mean_distances.append(mean_distance)
        variance_distances.append(variance_distance)

        # Plot KDE for distances of the current epoch
        plt.figure()
        sns.kdeplot(distances, shade=True, color='blue', label=f'Epoch {epoch}')
        plt.title(f"KDE of Distances for Epoch {epoch}")
        plt.xlabel("Distance")
        plt.ylabel("Density")
        plt.legend()
        plt.savefig(f'../save/kde_distances_epoch_{epoch}.png')
        plt.close()

# Create a table for mean and variance
table_data = pd.DataFrame({
    "Epoch": epochs,
    "Mean Distance": mean_distances,
    "Variance Distance": variance_distances
})

# Save the table as a CSV file
table_data.to_csv('../save/distance_summary.csv', index=False)

# Print the table
print(table_data)