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
"""def noniid_class_partition(dataset, num_users, num_classes, n, imbalance_factor=5.0):
    shouldcome
    Partition dataset in a non-IID way, ensuring each client receives samples from exactly `n` distinct classes.
    Each class's samples are split across clients such that the same sample is not used more than once.
    
    Parameters:
    - dataset: PyTorch dataset
    - num_users: Number of clients
    - num_classes: Total number of classes in the dataset
    - n: Number of classes per client
    - imbalance_factor: Controls the class imbalance within each client (higher = more imbalanced)
    shouldcome
    print("Completely fixed non-IID class partition sampling started")
    
    # Validate input parameters
    assert n <= num_classes, "n must be less than or equal to the number of classes"
    
    # Extract labels and create indices
    labels = np.array([target for _, target in dataset])
    idxs = np.arange(len(labels))
    
    # Group indices by class
    class_idxs = {c: idxs[labels == c] for c in range(num_classes)}
    class_sizes = {c: len(indices) for c, indices in class_idxs.items()}
    
    # Shuffle within each class to ensure random distribution
    for c in class_idxs:
        np.random.shuffle(class_idxs[c])
    
    # Step 1: First, evenly assign n classes to each client
    client_classes = {i: set() for i in range(num_users)}
    
    # Calculate how many users should get each class (approximately)
    users_per_class = (num_users * n) // num_classes
    
    # First, assign classes systematically to ensure good coverage
    for c in range(num_classes):
        # For each class, assign to users_per_class clients
        assigned_users = 0
        for i in range(num_users):
            if len(client_classes[i]) < n and c not in client_classes[i]:
                client_classes[i].add(c)
                assigned_users += 1
                if assigned_users >= users_per_class:
                    break
    
    # Fill in any remaining slots (clients that don't have n classes yet)
    for i in range(num_users):
        while len(client_classes[i]) < n:
            # Find a class that this client doesn't have yet
            available_classes = [c for c in range(num_classes) if c not in client_classes[i]]
            if not available_classes:
                # This should not happen with proper initialization
                print(f"Warning: No available classes for client {i}")
                break
            
            chosen_class = np.random.choice(available_classes)
            client_classes[i].add(chosen_class)
    
    # Step 2: Distribute data samples with intentional imbalance
    dict_users = {i: np.array([], dtype="int64") for i in range(num_users)}
    
    # Calculate how many samples of each class should go to each client
    # to ensure all samples are used and distributed reasonably
    class_allocation = {c: {} for c in range(num_classes)}
    
    for c in range(num_classes):
        # Get clients that have this class
        clients_with_class = [i for i, classes in client_classes.items() if c in classes]
        num_clients_with_class = len(clients_with_class)
        
        if num_clients_with_class == 0:
            print(f"Warning: No clients have class {c}. This should not happen.")
            continue
        
        # Calculate base allocation (equal division)
        samples_per_client = class_sizes[c] // num_clients_with_class
        remaining_samples = class_sizes[c] % num_clients_with_class
        
        # First, distribute base allocation
        for client_idx in clients_with_class:
            class_allocation[c][client_idx] = samples_per_client
        
        # Distribute remaining samples
        for i in range(remaining_samples):
            client_idx = clients_with_class[i % num_clients_with_class]
            class_allocation[c][client_idx] += 1
        
        # Now apply imbalance to make distribution non-IID
        # Only do this if we have enough samples to work with
        if num_clients_with_class >= 3:
            # Calculate average samples per client for this class
            avg_samples = class_sizes[c] / num_clients_with_class
            
            # Sort clients to decide which get more and which get less
            np.random.shuffle(clients_with_class)
            
            # Give more samples to the first 30% of clients
            favor_count = max(1, int(num_clients_with_class * 0.3))
            # Give fewer samples to the last 40% of clients
            reduce_count = max(1, int(num_clients_with_class * 0.4))
            
            # Calculate total adjustment (at most 30% of average)
            max_adjustment = int(avg_samples * 0.3)
            
            # Redistribute samples to create imbalance
            samples_to_redistribute = 0
            
            # Take samples from clients at the end
            for i in range(num_clients_with_class - reduce_count, num_clients_with_class):
                client_idx = clients_with_class[i]
                # Take up to 40% of this client's samples
                reduction = min(int(class_allocation[c][client_idx] * 0.4), max_adjustment)
                # Ensure client keeps at least 1 sample
                reduction = min(reduction, class_allocation[c][client_idx] - 1)
                class_allocation[c][client_idx] -= reduction
                samples_to_redistribute += reduction
            
            # Distribute extra samples to clients at the beginning
            for i in range(favor_count):
                if samples_to_redistribute <= 0:
                    break
                client_idx = clients_with_class[i]
                # Add up to imbalance_factor times their original allocation
                increase = min(samples_to_redistribute, max_adjustment)
                class_allocation[c][client_idx] += increase
                samples_to_redistribute -= increase
            
            # If any samples remain, distribute them to the middle clients
            if samples_to_redistribute > 0:
                middle_clients = clients_with_class[favor_count:num_clients_with_class-reduce_count]
                for client_idx in middle_clients:
                    if samples_to_redistribute <= 0:
                        break
                    increase = min(samples_to_redistribute, max_adjustment // len(middle_clients) + 1)
                    class_allocation[c][client_idx] += increase
                    samples_to_redistribute -= increase
    
    # Finally, actually distribute the data samples
    class_pointers = {c: 0 for c in range(num_classes)}
    
    for i in range(num_users):
        for c in client_classes[i]:
            if c in class_allocation and i in class_allocation[c]:
                num_samples = class_allocation[c][i]
                if num_samples > 0:
                    start_idx = class_pointers[c]
                    end_idx = start_idx + num_samples
                    
                    # Ensure we don't go out of bounds
                    end_idx = min(end_idx, len(class_idxs[c]))
                    
                    if start_idx < end_idx:  # Only add if there are samples to add
                        dict_users[i] = np.concatenate((dict_users[i], class_idxs[c][start_idx:end_idx]))
                        class_pointers[c] = end_idx
    
    # Verify that all clients have data and report statistics
    client_sample_counts = [len(dict_users[i]) for i in range(num_users)]
    empty_clients = [i for i, count in enumerate(client_sample_counts) if count == 0]
    
    if empty_clients:
        print(f"Warning: {len(empty_clients)} clients have no data.")
        
        # Emergency fix: give these clients some data from anywhere
        for client_idx in empty_clients:
            # Get classes for this client
            assigned_classes = client_classes[client_idx]
            for c in assigned_classes:
                # Find a client with a lot of samples of this class
                class_users = [(i, len([idx for idx in dict_users[i] if labels[idx] == c])) 
                              for i in range(num_users) if i != client_idx and len(dict_users[i]) > 0]
                if class_users:
                    # Sort by number of samples of this class
                    class_users.sort(key=lambda x: x[1], reverse=True)
                    donor_idx = class_users[0][0]
                    
                    # Get indices of samples of this class from the donor
                    donor_samples = [idx for idx in dict_users[donor_idx] if labels[idx] == c]
                    if len(donor_samples) > 10:  # Make sure donor has enough samples
                        # Take half of these samples
                        num_to_take = len(donor_samples) // 2
                        samples_to_move = donor_samples[:num_to_take]
                        
                        # Move samples from donor to empty client
                        dict_users[client_idx] = np.array(samples_to_move, dtype='int64')
                        dict_users[donor_idx] = np.array([idx for idx in dict_users[donor_idx] if idx not in samples_to_move], 
                                                       dtype='int64')
                        print(f"Emergency: Moved {num_to_take} samples of class {c} from client {donor_idx} to {client_idx}")
                        break
    
    # Calculate and print statistics about the distribution
    client_sample_counts = [len(dict_users[i]) for i in range(num_users)]
    print(f"Average samples per client: {np.mean(client_sample_counts):.1f}")
    print(f"Min samples: {np.min(client_sample_counts)}, Max samples: {np.max(client_sample_counts)}")
    
    # Calculate class distribution statistics
    client_class_counts = {i: {} for i in range(num_users)}
    for i in range(num_users):
        for idx in dict_users[i]:
            label = labels[idx]
            if label not in client_class_counts[i]:
                client_class_counts[i][label] = 0
            client_class_counts[i][label] += 1
    
    # Print class statistics
    print("Class distribution summary:")
    class_per_client = [len(client_class_counts[i]) for i in range(num_users)]
    print(f"Average classes per client: {np.mean(class_per_client):.1f}")
    print(f"Min classes: {np.min(class_per_client)}, Max classes: {np.max(class_per_client)}")
    
    print("Non-IID sampling completed")
    return dict_users"""
def extreme_noniid_partition(dataset, num_users, num_classes, n=2, labels_per_shard=1):
    """
    Creates an extremely challenging non-IID partition that closely mimics the original
    MNIST non-IID function but with stricter class isolation.
    
    Parameters:
    - dataset: PyTorch dataset
    - num_users: Number of clients
    - num_classes: Total number of classes in the dataset
    - n: Number of classes per client
    - labels_per_shard: How many classes are in each shard (typically 1 for high heterogeneity)
    """
    print("Extreme non-IID partition sampling started")
    
    # Extract labels and create indices
    labels = np.array([target for _, target in dataset])
    idxs = np.arange(len(labels))
    
    # Step 1: Create shards where each shard contains samples from only 1 class
    shards = []
    for c in range(num_classes):
        class_idxs = idxs[labels == c]
        np.random.shuffle(class_idxs)
        
        # Split this class into multiple shards
        samples_per_class = len(class_idxs)
        shards_per_class = max(3, samples_per_class // 600)  # Create enough shards
        
        # Create shards for this class
        for i in range(shards_per_class):
            start_idx = (i * samples_per_class) // shards_per_class
            end_idx = ((i + 1) * samples_per_class) // shards_per_class
            shards.append(class_idxs[start_idx:end_idx])
    
    # Shuffle shards to mix up different classes
    np.random.shuffle(shards)
    
    # Step 2: Assign exactly n shards to each client
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    
    # Track which classes each client has received
    client_classes = {i: set() for i in range(num_users)}
    
    # First pass: try to give each client exactly n different classes
    shards_per_client = n
    
    # Sort clients randomly
    client_order = np.arange(num_users)
    np.random.shuffle(client_order)
    
    # This will track which shards are assigned
    assigned_shards = set()
    
    # First pass: Try to assign n different classes to each client
    for i in client_order:
        assigned_count = 0
        
        # Keep trying until we find n distinct classes or run out of options
        for shard_idx in range(len(shards)):
            if shard_idx in assigned_shards:
                continue
                
            # Determine the class(es) in this shard
            shard_labels = set(labels[shards[shard_idx]])
            
            # Only add this shard if it gives a new class to this client
            if not any(label in client_classes[i] for label in shard_labels):
                dict_users[i] = np.concatenate((dict_users[i], shards[shard_idx]))
                client_classes[i].update(shard_labels)
                assigned_shards.add(shard_idx)
                assigned_count += 1
                
                # Break once we've assigned enough different classes
                if assigned_count >= n:
                    break
    
    # Second pass: For any client that didn't get enough classes, 
    # assign random remaining shards
    for i in range(num_users):
        # If client doesn't have enough data/classes, give them more shards
        while len(client_classes[i]) < n:
            # Find available shards
            remaining_shards = [idx for idx in range(len(shards)) if idx not in assigned_shards]
            
            if not remaining_shards:
                # If no unassigned shards left, take from clients with more than n shards
                # Sort clients by number of shards they have (descending)
                donor_candidates = [(j, len(dict_users[j])) for j in range(num_users) if j != i]
                donor_candidates.sort(key=lambda x: x[1], reverse=True)
                
                # Take from the client with the most data
                if donor_candidates and donor_candidates[0][1] > 0:
                    donor = donor_candidates[0][0]
                    donor_labels = [labels[idx] for idx in dict_users[donor]]
                    
                    # Group donor's samples by class
                    donor_samples_by_class = {}
                    for idx in dict_users[donor]:
                        label = labels[idx]
                        if label not in donor_samples_by_class:
                            donor_samples_by_class[label] = []
                        donor_samples_by_class[label].append(idx)
                    
                    # Find classes that recipient doesn't have but donor does
                    missing_classes = set(range(num_classes)) - client_classes[i]
                    common_classes = [c for c in missing_classes if c in donor_samples_by_class]
                    
                    if common_classes:
                        # Take samples of one missing class from donor
                        chosen_class = np.random.choice(common_classes)
                        samples_to_take = donor_samples_by_class[chosen_class]
                        
                        # Take half of these samples
                        num_take = max(1, len(samples_to_take) // 2)
                        take_indices = samples_to_take[:num_take]
                        
                        # Add to recipient
                        dict_users[i] = np.concatenate((dict_users[i], take_indices))
                        client_classes[i].add(chosen_class)
                        
                        # Remove from donor
                        dict_users[donor] = np.array([idx for idx in dict_users[donor] 
                                                   if idx not in take_indices], dtype='int64')
                        
                        print(f"Transferred {num_take} samples of class {chosen_class} from client {donor} to {i}")
                    else:
                        # No suitable class to transfer
                        print(f"Warning: Cannot find suitable class to transfer to client {i}")
                        break
                else:
                    print(f"Warning: No donors available for client {i}")
                    break
            else:
                # Assign a random remaining shard
                shard_idx = np.random.choice(remaining_shards)
                shard_labels = set(labels[shards[shard_idx]])
                
                dict_users[i] = np.concatenate((dict_users[i], shards[shard_idx]))
                client_classes[i].update(shard_labels)
                assigned_shards.add(shard_idx)
    
    # Create extreme class imbalance within clients
    for i in range(num_users):
        if len(dict_users[i]) > 0:
            # Calculate class distribution for this client
            client_labels = [labels[idx] for idx in dict_users[i]]
            client_label_counts = {}
            for label in client_labels:
                if label not in client_label_counts:
                    client_label_counts[label] = 0
                client_label_counts[label] += 1
            
            # Sort classes by frequency (ascending)
            sorted_classes = sorted(client_label_counts.items(), key=lambda x: x[1])
            
            # If client has more than one class, make distribution very skewed
            if len(sorted_classes) > 1:
                minority_class = sorted_classes[0][0]
                
                # Keep only 10-20% of minority class samples
                minority_indices = [idx for idx in dict_users[i] if labels[idx] == minority_class]
                keep_count = max(1, len(minority_indices) // np.random.randint(5, 10))
                
                # Remove some minority samples to create extreme imbalance
                indices_to_remove = minority_indices[keep_count:]
                dict_users[i] = np.array([idx for idx in dict_users[i] if idx not in indices_to_remove], 
                                       dtype='int64')
    
    # Check if any clients ended up with no data
    empty_clients = [i for i in range(num_users) if len(dict_users[i]) == 0]
    if empty_clients:
        print(f"Warning: {len(empty_clients)} clients have no data")
        
        # Emergency fix: give these clients some data
        donors = sorted([(i, len(dict_users[i])) for i in range(num_users) if i not in empty_clients], 
                      key=lambda x: x[1], reverse=True)
        
        for empty_client in empty_clients:
            if donors:
                donor_idx = donors[0][0]
                # Take 30% of donor's data
                num_samples = max(10, len(dict_users[donor_idx]) // 3)
                samples_to_take = dict_users[donor_idx][:num_samples]
                
                dict_users[empty_client] = samples_to_take
                dict_users[donor_idx] = np.array([idx for idx in dict_users[donor_idx] 
                                             if idx not in samples_to_take], dtype='int64')
                
                # Update donor list
                donors[0] = (donor_idx, len(dict_users[donor_idx]))
                donors.sort(key=lambda x: x[1], reverse=True)
                
                print(f"Emergency: Transferred {num_samples} samples to empty client {empty_client}")
    
    # Calculate and print statistics
    client_sizes = [len(dict_users[i]) for i in range(num_users)]
    print(f"Average samples per client: {np.mean(client_sizes):.1f}")
    print(f"Min samples: {np.min(client_sizes)}, Max samples: {np.max(client_sizes)}")
    
    class_distribution = []
    for i in range(num_users):
        client_counts = {}
        for idx in dict_users[i]:
            label = labels[idx]
            if label not in client_counts:
                client_counts[label] = 0
            client_counts[label] += 1
        
        if client_counts:
            # Calculate ratio between largest and smallest class
            values = list(client_counts.values())
            class_ratio = max(values) / max(1, min(values))
            class_distribution.append(class_ratio)
    
    if class_distribution:
        print(f"Average class imbalance ratio within clients: {np.mean(class_distribution):.1f}")
        print(f"Max class imbalance ratio: {np.max(class_distribution):.1f}")
    
    classes_per_client = [len(client_classes[i]) for i in range(num_users)]
    print(f"Average classes per client: {np.mean(classes_per_client):.1f}")
    
    print("Extreme non-IID partition sampling completed")
    return dict_users
"""def improved_noniid_partition(dataset, num_users, num_classes, n):
    shouldcome
    Creates a non-IID partition where each client receives exactly n different classes.
    Each class is divided into shards, and clients receive different shards of their assigned classes.
    
    Parameters:
    - dataset: PyTorch dataset
    - num_users: Number of clients
    - num_classes: Total number of classes in the dataset
    - n: Exact number of classes each client will receive
    
    Returns:
    - dict_users: Dictionary mapping client indices to their data indices
   shouldcome
    print(f"Improved non-IID partition started (classes per client: {n})")
    
    # Validate parameters
    assert n <= num_classes, "n must be less than or equal to the number of classes"
    
    # Calculate required number of shards per class
    required_class_assignments = num_users * n
    shards_per_class = required_class_assignments // num_classes
    
    # Ensure we can create an even distribution
    assert required_class_assignments % num_classes == 0, \
        f"num_users*n ({num_users}*{n}={required_class_assignments}) must be divisible by num_classes ({num_classes})"
    
    print(f"Each class will be divided into {shards_per_class} shards")
    
    # Extract labels and create indices
    labels = np.array([target for _, target in dataset])
    idxs = np.arange(len(labels))
    
    # Group indices by class
    class_idxs = {}
    for c in range(num_classes):
        class_idxs[c] = idxs[labels == c]
        np.random.shuffle(class_idxs[c])  # Shuffle within each class
    
    # Create shards for each class
    class_shards = {}
    for c in range(num_classes):
        # Calculate samples per shard for this class
        samples_in_class = len(class_idxs[c])
        samples_per_shard = samples_in_class // shards_per_class
        
        # Create the shards
        class_shards[c] = []
        for i in range(shards_per_class):
            start_idx = i * samples_per_shard
            # Last shard gets any remaining samples
            end_idx = samples_in_class if i == shards_per_class - 1 else (i + 1) * samples_per_shard
            class_shards[c].append(class_idxs[c][start_idx:end_idx])
            
    # Prepare tracking variables
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    client_classes = {i: set() for i in range(num_users)}
    available_shards = {c: list(range(shards_per_class)) for c in range(num_classes)}
    
    # First, decide which n classes each client will get
    for i in range(num_users):
        # Find available classes (those with remaining shards)
        available_classes = [c for c in range(num_classes) if available_shards[c]]
        
        # Randomly select n classes for this client
        selected_classes = np.random.choice(available_classes, size=n, replace=False)
        
        # Assign classes to client
        client_classes[i] = set(selected_classes)
        
        # For each assigned class, allocate one shard and update availability
        for c in selected_classes:
            # Get an available shard index
            shard_idx = available_shards[c].pop(0)
            
            # Add the shard's indices to the client
            dict_users[i] = np.concatenate((dict_users[i], class_shards[c][shard_idx]))
    
    # Verify the distribution
    # 1. Check that each client got exactly n classes
    for i in range(num_users):
        actual_classes = set(labels[idx] for idx in dict_users[i])
        if len(actual_classes) != n:
            print(f"Warning: Client {i} has {len(actual_classes)} classes instead of {n}")
    
    # 2. Compute statistics on the distribution
    client_sizes = [len(dict_users[i]) for i in range(num_users)]
    print(f"Average samples per client: {np.mean(client_sizes):.1f}")
    print(f"Min samples: {np.min(client_sizes)}, Max samples: {np.max(client_sizes)}")
    
    # 3. Calculate class distribution statistics
    client_class_counts = {}
    for i in range(num_users):
        client_class_counts[i] = {}
        for idx in dict_users[i]:
            label = labels[idx]
            if label not in client_class_counts[i]:
                client_class_counts[i][label] = 0
            client_class_counts[i][label] += 1
    
    # Calculate ratio between largest and smallest class for each client
    class_ratios = []
    for i in range(num_users):
        if len(client_class_counts[i]) > 1:  # Need at least 2 classes to compute ratio
            values = list(client_class_counts[i].values())
            class_ratio = max(values) / max(1, min(values))
            class_ratios.append(class_ratio)
    
    if class_ratios:
        print(f"Average class imbalance ratio within clients: {np.mean(class_ratios):.1f}")
        print(f"Max class imbalance ratio: {np.max(class_ratios):.1f}")
    
    # Create additional imbalance within clients to make the task more challenging
    if n > 1:  # Only applicable if clients have more than one class
        print("Creating additional within-client imbalance...")
        
        for i in range(num_users):
            if len(client_class_counts[i]) > 1:
                # Sort classes by number of samples (ascending)
                sorted_classes = sorted(client_class_counts[i].items(), key=lambda x: x[1])
                
                # Select the smallest class
                minority_class = sorted_classes[0][0]
                
                # Keep only 10-30% of minority class samples
                minority_indices = [idx for idx in dict_users[i] if labels[idx] == minority_class]
                keep_percentage = np.random.uniform(0.1, 0.3)  # Random percentage between 10% and 30%
                keep_count = max(1, int(len(minority_indices) * keep_percentage))
                
                # Remove some minority samples to create extreme imbalance
                indices_to_keep = minority_indices[:keep_count]
                indices_to_remove = minority_indices[keep_count:]
                
                # Update client's data
                dict_users[i] = np.array([idx for idx in dict_users[i] if idx not in indices_to_remove], 
                                      dtype='int64')
                
                print(f"Client {i}: Reduced class {minority_class} from {len(minority_indices)} to {keep_count} samples")
    
    # Recalculate class distribution statistics after creating additional imbalance
    if n > 1:
        client_class_counts = {}
        for i in range(num_users):
            client_class_counts[i] = {}
            for idx in dict_users[i]:
                label = labels[idx]
                if label not in client_class_counts[i]:
                    client_class_counts[i][label] = 0
                client_class_counts[i][label] += 1
        
        # Calculate new class ratios
        class_ratios = []
        for i in range(num_users):
            if len(client_class_counts[i]) > 1:
                values = list(client_class_counts[i].values())
                class_ratio = max(values) / max(1, min(values))
                class_ratios.append(class_ratio)
        
        if class_ratios:
            print(f"New average class imbalance ratio: {np.mean(class_ratios):.1f}")
            print(f"New max class imbalance ratio: {np.max(class_ratios):.1f}")
    
    print("Improved non-IID partition completed")
    return dict_users"""
if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
    


