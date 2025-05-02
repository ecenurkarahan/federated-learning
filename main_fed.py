#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
from utils.sampling import extreme_noniid_partition, mnist_iid, mnist_noniid, cifar_iid, cifar_noniid, fashion_mnist_iid, fashion_mnist_noniid, noniid_dirichlet
from utils.options import args_parser
from utils.distances import calculate_weight_distances
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar, ShuffleNetV2, CNNFashionMnist, ResNet18
from models.Fed import FedAvg
from models.test import test_img
import time

start_time = time.time()
isDirichlet = False
isPartition = False
alpha = None
partition_n= None
distance_means = []
distance_variances = []
weights_over_epochs = []

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    print("experiment is started")

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        elif args.partition_noniid:
            print(f"Using extreme non-IID partitioning on mnist with {args.partition_noniid} partitions")
            isPartition= True
            partition_n= args.partition_noniid
            dict_users = extreme_noniid_partition(dataset = dataset_train, num_users = args.num_users,num_classes=args.num_classes, n = args.partition_noniid)
        else:
            print("partititioning mnist with dirichlet")
            isDirichlet=True
            alpha= args.dirichlet_alpha
            dict_users = noniid_dirichlet(dataset_train, args.num_users,num_classes= args.num_classes,alpha=args.dirichlet_alpha)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        elif args.partition_noniid:
            print(f"Using new non-IID partitioning on cifar10 with {args.partition_noniid} partitions")
            isPartition= True
            partition_n= args.partition_noniid
            dict_users = extreme_noniid_partition(dataset = dataset_train, num_users = args.num_users,num_classes=args.num_classes, n = args.partition_noniid)
        else:
            print("partititioning cifar10 with dirichlet")
            isDirichlet=True
            alpha= args.dirichlet_alpha
            dict_users = noniid_dirichlet(dataset_train, args.num_users,num_classes= args.num_classes,alpha=args.dirichlet_alpha)
    elif args.dataset== 'fashion_mnist':
        trans_fashion_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.FashionMNIST('../data/fashion_mnist/', train=True, download=True, transform=trans_fashion_mnist)
        dataset_test = datasets.FashionMNIST('../data/fashion_mnist/', train=False, download=True, transform=trans_fashion_mnist)
        # sample users
        if args.iid:
            dict_users = fashion_mnist_iid(dataset_train, args.num_users)
        elif args.partition_noniid:
            print(f"Using new non-IID partitioning on fashion-mnist with {args.partition_noniid} partitions")
            isPartition= True            
            partition_n= args.partition_noniid
            dict_users = extreme_noniid_partition(dataset = dataset_train, num_users = args.num_users,num_classes=args.num_classes, n = args.partition_noniid)
        else:
            #tried to do dirichlet, dataset independent
            print("partititioning fashion_mnist with dirichlet")
            isDirichlet=True
            alpha= args.dirichlet_alpha
            dict_users = noniid_dirichlet(dataset_train, args.num_users,num_classes= args.num_classes,alpha=args.dirichlet_alpha)
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape
#-----------------------------------------------
#model eklenicekse buraya eklenip değiştirilebilir
    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'fashion_mnist':
        net_glob = CNNFashionMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    elif args.model == 'shufflenet':
        net_glob = ShuffleNetV2(args=args).to(args.device)
    elif args.model == 'resnet18':
        net_glob = ResNet18(args=args).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train : list = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    if args.all_clients: 
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]
    
    # Initialize lists to store test loss and accuracy
    test_loss_over_rounds = []
    test_acc_over_rounds = []
    round_numbers = []
    # bu roundda dönen weight modle burada
    # önceki versiyonları alsak
    # cosine similarity farklı çıkmalı for non iid
    # l1 distance
    # l2 distance
    # çıkan valueların mean ve variance ı
    for iter in range(args.epochs):
        loss_locals = []
        if not args.all_clients:
            w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        # Store local weights for the current epoch
        if iter == 0:
            weights_over_epochs = [w_locals]
        else:
            weights_over_epochs.append(w_locals)

        # Compare weights with the previous epoch
        if iter > 0:
            mean_dist, var_dist = calculate_weight_distances(
                distance_metric=args.distance_metric,  # Change to 'l1' or 'cosine' as needed
                weights_epoch_1=weights_over_epochs[iter - 1],
                weights_epoch_2=weights_over_epochs[iter]
            )
            distance_means.append(mean_dist)
            distance_variances.append(var_dist)
            print(f"Epoch {iter}: Mean Distance = {mean_dist:.4f}, Variance = {var_dist:.4f}")
        """
        # Print local weights for the current epoch
        print("************************************************************************")
        print(f"Epoch {iter + 1}: Local weights for selected clients:")
        for i, w_local in enumerate(w_locals):
            print(f"Client {i}: {w_local}")"""
        # update global weights
        # bburada client 
        w_glob = FedAvg(w_locals)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)
        # Testing
        net_glob.eval()
        acc_train, loss_train_test = test_img(net_glob, dataset_train, args)
        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        
        # Store test loss and accuracy
        test_loss_over_rounds.append(loss_test)
        test_acc_over_rounds.append(acc_test)
        round_numbers.append(iter)

        print("Training accuracy: {:.2f}".format(acc_train))
        print("Testing accuracy: {:.2f}".format(acc_test))
    
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))

    # Plot train loss
    plt.plot(range(len(loss_train)), loss_train, label='Train Loss', color='blue')

    # Plot test loss
    plt.plot(round_numbers, test_loss_over_rounds, label='Test Loss', color='red')

    # Labels and title
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.title('Train and Test Loss Over Rounds')
    plt.legend()  # Show legend to differentiate lines

# Save the figure
    plt.savefig('./save/loss_comparison_{}_{}_{}_C{}_iid{}_isDirichlet{}_isPartition{}_dirichletA{}_partitionN{}.png'.format(
        args.dataset, args.model, args.epochs, args.frac, args.iid, isDirichlet,isPartition, alpha,partition_n
    ))
# Record elapsed time to a file
end_time = time.time()
elapsed_time = end_time - start_time
#model, dataset, epoch,fraction,num_channels,num_users,local_ep,iddness, accuracy train, accuracy test, elapsed time, time as minutes
with open('noniid_experiment_results.txt', 'a') as f:
    f.write(f"Model: {args.model}, Dataset: {args.dataset}, Epochs: {args.epochs}, Frac: {args.frac}, Num_channels: {args.num_channels}, Local_ep: {args.local_ep}, Iid: {args.iid},Partitioning-based n: {args.partition_noniid}, Dirichlet alpha: {args.dirichlet_alpha},Train Accuracy: {acc_train},Test Accuracy: {acc_test}, Elapsed Time: {elapsed_time}, Time in minutes: {elapsed_time/60}\n")
    # Plot and save the histogram of mean distances
plt.figure()
plt.hist(distance_means, bins=30, alpha=0.7, color='blue', label='Mean Distances')
plt.title("Histogram of Mean Distances Across Epochs")
plt.xlabel("Mean Distance")
plt.ylabel("Frequency")
plt.legend()
plt.savefig('./save/distances/mean_distance_histogram_mwteic{}.png'.format(args.distance_metric))

# Plot and save the histogram of variance distances
plt.figure()
plt.hist(distance_variances, bins=30, alpha=0.7, color='green', label='Variance of Distances')
plt.title("Histogram of Variance of Distances Across Epochs")
plt.xlabel("Variance")
plt.ylabel("Frequency")
plt.legend()
plt.savefig('./save/distances/variance_distance_histogram_metric_{}.png'.format(args.distance_metric))