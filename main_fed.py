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

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid, fashion_mnist_iid, fashion_mnist_noniid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar, ShuffleNetV2Mnist, ShuffleNetV2Cifar, CNNFashionMnist, ShuffleNetV2FashionMnist
from models.Fed import FedAvg
from models.test import test_img

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            dict_users = cifar_noniid(dataset_train, args.num_users)
    elif args.dataset== 'fashion_mnist':
        trans_fashion_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.FashionMNIST('../data/fashion_mnist/', train=True, download=True, transform=trans_fashion_mnist)
        dataset_test = datasets.FashionMNIST('../data/fashion_mnist/', train=False, download=True, transform=trans_fashion_mnist)
        # sample users
        if args.iid:
            dict_users = fashion_mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = fashion_mnist_noniid(dataset_train, args.num_users)
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
    elif args.model == 'shufflenet' and args.dataset == 'cifar':
        net_glob = ShuffleNetV2Cifar(args=args).to(args.device)
    elif args.model == 'shufflenet' and args.dataset == 'mnist':
        net_glob = ShuffleNetV2Mnist(args=args).to(args.device)
    elif args.model == 'shufflenet' and args.dataset == 'fashion_mnist':
        net_glob = ShuffleNetV2FashionMnist(args=args).to(args.device)
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
        # update global weights
        w_glob = FedAvg(w_locals)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)
        plt.figure()
        plt.plot(range(len(loss_train)), loss_train)
        plt.ylabel('train_loss')
        plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))
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

    # Plot test loss over rounds
    plt.figure()
    plt.plot(round_numbers, test_loss_over_rounds)
    plt.xlabel('Round')
    plt.ylabel('Test Loss')
    plt.title('Test Loss Over Rounds')
    plt.savefig('./save/test_loss_over_rounds-{args}.png')

    # Optionally, plot test accuracy over rounds
    plt.figure()
    plt.plot(round_numbers, test_acc_over_rounds)
    plt.xlabel('Round')
    plt.ylabel('Test Accuracy')
    plt.title('Test Accuracy Over Rounds')
    plt.savefig('./save/test_accuracy_over_rounds-{args}.png')
