import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine

def calculate_weight_distances(distance_metric, weights_epoch_1, weights_epoch_2):
    """
    Calculate the mean and variance of distances between weights of two epochs.

    Args:
        distance_metric (str): The distance metric ('l1', 'l2', 'cosine').
        weights_epoch_1 (list): Local weights from epoch 1.
        weights_epoch_2 (list): Local weights from epoch 2.

    Returns:
        tuple: Mean and variance of the distances.
    """
    distances = []

    for w1, w2 in zip(weights_epoch_1, weights_epoch_2):
        # Flatten the weights to 1D arrays for comparison
        w1_flat = np.concatenate([v.cpu().numpy().flatten() for v in w1.values()])
        w2_flat = np.concatenate([v.cpu().numpy().flatten() for v in w2.values()])

        # Calculate distance based on the metric
        if distance_metric == 'l1':
            dist = np.sum(np.abs(w1_flat - w2_flat))
        elif distance_metric == 'l2':
            dist = np.sqrt(np.sum((w1_flat - w2_flat) ** 2))
        elif distance_metric == 'cosine':
            dist = cosine(w1_flat, w2_flat)
        else:
            raise ValueError("Unsupported distance metric. Use 'l1', 'l2', or 'cosine'.")

        distances.append(dist)
    print(f"Distances for epoch {iter}: {distances}")
    # Calculate mean and variance
    mean_distance = np.mean(distances)
    variance_distance = np.var(distances)

    return mean_distance, variance_distance