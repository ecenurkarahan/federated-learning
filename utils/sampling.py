#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms
#eğer farklı datasetler eklemek istersem tam buraya ekleme yapacağım
#pytorchta train ve test yapılıp sonra olay buraya geliyor iid ve non iid için
# does identical distribution
#change the non iid distribution
def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users

# why cifar non iid does not exist?
def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

#ben ekledim
def cifar_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 250
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    
    # Get labels for the CIFAR10 dataset
    labels = np.array([target for _, target in dataset])
    
    # Sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]
    
    # Divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    
    return dict_users
 #ben ekledim   
def fashion_mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from Fashion MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users
#ben ekledim
def fashion_mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from Fashion MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    
    # Get labels for the Fashion MNIST dataset
    labels = dataset.train_labels.numpy()
    
    # Sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]
    
    # Divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    
    return dict_users

# small alpha scews the distribution more : more non iid
def noniid_dirichlet(dataset, num_users, num_classes, alpha=0.5):
    """
    Sample non-I.I.D. client data using Dirichlet distribution
    :param dataset: PyTorch dataset (CIFAR10, FashionMNIST, etc.)
    :param num_users: Number of clients
    :param alpha: Dirichlet concentration parameter (smaller means more skewed)
    :return: Dictionary mapping user_id to a list of data indices
    """
    print("non-iid dirichlet started")
    labels = np.array([target for _, target in dataset])  # Extract labels
    idxs = np.arange(len(labels))  # Indices of the dataset
    
    # Initialize user indices dictionary
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    
    # Group indices by class
    class_idxs = [idxs[labels == c] for c in range(num_classes)]
    
    # Dirichlet distribution to generate proportions for each user
    label_distribution = np.random.dirichlet([alpha] * num_users, num_classes)
    
    # Assign samples to users based on the Dirichlet distribution
    for c, c_idxs in enumerate(class_idxs):
        np.random.shuffle(c_idxs)
        
        # Calculate proportions and ensure they sum to the class size
        proportions = label_distribution[c]
        # Convert to sample counts and handle rounding
        split_idxs = np.zeros(num_users, dtype=int)
        
        # Allocate samples to users, adjusting for rounding issues
        remaining = len(c_idxs)
        for i in range(num_users-1):
            split_idxs[i] = min(int(proportions[i] * len(c_idxs)), remaining)
            remaining -= split_idxs[i]
        split_idxs[-1] = remaining  # Last user gets all remaining samples
        
        # Distribute samples
        start = 0
        for i in range(num_users):
            if split_idxs[i] > 0:  # Only add samples if there are any to add
                end = start + split_idxs[i]
                dict_users[i] = np.concatenate((dict_users[i], c_idxs[start:end]))
                start = end
    
    return dict_users

# datayı parçalara bölüp bir clienta gönder her clientta 2 veya 3 classtan sample olsun n class sample (parametreli)
def noniid_class_partition(dataset, num_users, num_classes, n):
        """
        Partition dataset in a non-IID way, ensuring each client receives samples from exactly `n` distinct classes.
        Each class's samples are split across clients such that the same sample is not used more than once.
        """
        print("Sampling started for partition")
        assert n <= num_classes, "n must be less than or equal to the number of classes"
        assert (num_users * n) % num_classes == 0, "Ensure clients can evenly divide class samples"

        labels = np.array([target for _, target in dataset])
        idxs = np.arange(len(labels))
        
        # Group indices by class
        class_idxs = {c: idxs[labels == c] for c in range(num_classes)}

        # Shuffle within each class to ensure random distribution
        for c in class_idxs:
            np.random.shuffle(class_idxs[c])

        # Track how many times each class will be assigned
        class_quota = {c: (num_users * n) // num_classes for c in range(num_classes)}  # how many clients should get class c

        dict_users = {i: np.array([], dtype="int64") for i in range(num_users)}
        client_classes = {}

        # Step 1: Assign n classes per client, reusing classes but tracking quota
        available_classes = list(range(num_classes))

        for i in range(num_users):
            client_classes[i] = set()
            tries = 0
            while len(client_classes[i]) < n:
                # Filter out classes whose quota is 0
                valid_classes = [c for c in available_classes if class_quota[c] > 0]
                if not valid_classes:
                    raise ValueError("Ran out of valid classes to assign.")
                chosen_class = np.random.choice(valid_classes)
                client_classes[i].add(chosen_class)
                class_quota[chosen_class] -= 1
                tries += 1
                if tries > 1000:
                    raise RuntimeError("Infinite loop in class assignment")

        # Step 2: Distribute data samples (non-overlapping) from those classes to each client
        class_allocation_index = {c: 0 for c in range(num_classes)}  # pointer to where we are in class sample list

        for i in range(num_users):
            for c in client_classes[i]:
                # Compute how many samples to assign to each client per class
                total_clients_with_c = (num_users * n) // num_classes
                samples_per_client = len(class_idxs[c]) // total_clients_with_c
                start_idx = class_allocation_index[c]
                end_idx = start_idx + samples_per_client
                dict_users[i] = np.concatenate((dict_users[i], class_idxs[c][start_idx:end_idx]))
                class_allocation_index[c] += samples_per_client
        print("Sampling completed")
        return dict_users

def noniid_class_partition2(dataset, num_users, num_classes, n):
    """
    Partition dataset in a non-IID way, ensuring each client receives samples from exactly `n` distinct classes.
    When multiple clients are assigned the same class, they receive different samples.
    Optimized for scenarios where number of clients significantly exceeds number of classes.
    
    :param dataset: PyTorch dataset (MNIST, CIFAR10, FashionMNIST, etc.)
    :param num_users: Number of clients
    :param num_classes: Total number of classes in the dataset
    :param n: Number of different classes assigned to each client
    :return: Dictionary mapping user_id to a list of data indices
    """
    print("partition 2 started")
    assert n <= num_classes, "n must be less than or equal to the number of classes"
    
    # Extract labels
    labels = np.array([target for _, target in dataset])
    idxs = np.arange(len(labels))
    
    # Group indices by class
    class_idxs = {c: idxs[labels == c].tolist() for c in range(num_classes)}
    original_class_sizes = {c: len(indices) for c, indices in class_idxs.items()}
    
    # Shuffle indices within each class for randomness
    for c in class_idxs:
        np.random.shuffle(class_idxs[c])
    
    # Initialize user dictionaries with lists (not numpy arrays)
    dict_users = {i: [] for i in range(num_users)}
    
    # First, determine how many samples we can allocate per client for each class
    # Assuming approximately equal distribution
    samples_per_client_per_class = {}
    for c in range(num_classes):
        # We need to estimate how many clients will be assigned this class
        # For simplicity, estimate that each class will be assigned to (num_users * n / num_classes) clients
        estimated_clients_for_class = (num_users * n) / num_classes
        # Allow some buffer (80% of equal distribution) to ensure we don't run out
        samples_per_client_per_class[c] = int(original_class_sizes[c] / (estimated_clients_for_class * 1.2))
        
        if samples_per_client_per_class[c] < 1:
            samples_per_client_per_class[c] = 1
            print(f"Warning: Very limited samples for class {c}. Each client will get only 1 sample.")
    
    # Assign n classes to each client
    for i in range(num_users):
        # Prioritize classes with more remaining samples
        available_classes = sorted(range(num_classes), key=lambda c: len(class_idxs[c]), reverse=True)
        
        # Select n classes for this client
        selected_classes = []
        for c in available_classes:
            if len(selected_classes) >= n:
                break
                
            # Only select if there are enough samples left
            if len(class_idxs[c]) >= samples_per_client_per_class[c]:
                selected_classes.append(c)
        
        # If we couldn't get enough classes with sufficient samples, take what's available
        if len(selected_classes) < n:
            print(f"Warning: Client {i} could only be assigned {len(selected_classes)} classes instead of {n}")
            # Take remaining classes with any samples left
            for c in available_classes:
                if c not in selected_classes and len(class_idxs[c]) > 0:
                    selected_classes.append(c)
                    if len(selected_classes) >= n:
                        break
        
        # Assign samples for selected classes
        for c in selected_classes:
            # Determine how many samples to give this client
            num_samples = min(samples_per_client_per_class[c], len(class_idxs[c]))
            
            # Take samples from the front of the list (they're already shuffled)
            samples = class_idxs[c][:num_samples]
            dict_users[i].extend(samples)
            
            # Remove these samples from the available pool
            class_idxs[c] = class_idxs[c][num_samples:]
    
    # Convert to numpy arrays - fixed to resolve the type error
    result = {}
    for i in range(num_users):
        result[i] = np.array(dict_users[i], dtype='int64')
    print("partition 2 ended")
    return result
    
if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
    


