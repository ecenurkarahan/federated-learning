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
def noniid_dirichlet(dataset, num_users,num_classes, alpha=0.5):
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
    
    # Dirichlet distribution to generate proportions for each user
    label_distribution = np.random.dirichlet([alpha] * num_users, num_classes)
    
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    
    # Assign samples to users based on the Dirichlet distribution
    for c in range(num_classes):
        class_idxs = idxs[labels == c]
        np.random.shuffle(class_idxs)
        
        # Split indices according to Dirichlet proportions
        split_idxs = (label_distribution[c] * len(class_idxs)).astype(int)
        
        start = 0
        for i in range(num_users):
            end = start + split_idxs[i]
            dict_users[i] = np.concatenate((dict_users[i], class_idxs[start:end]))
            start = end
    
    return dict_users
if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
    


