#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
# this part gives information about how to run the application
import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    # round of training yüksek tutmak gerekir (saldırı için) 100,200
    parser.add_argument('--epochs', type=int, default=10, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users included in the federated learning process: K")
    # fraction of clients chosen to be included in the training process in each round
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    # each client trains this many epochs before sending it to the main model
    # düşük tutuluyor 1,2
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    # batch size used in every client
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    # learning rate from each client, but ask about this!!!
    # parametre değiştirirken değişme hızı
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    #how data is split among clients
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")

    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name (cnn, mlp, shufflenet)')
    #can be useful when multiple types of kernels is used in a model but currently is not used
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    # can be useful when changing the architecture of cnn models, num of kernels used in each convolutional layer
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    # this again can be used to change the cnn model architechture
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset (mnist, cifar, fashion_mnist)")
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    # datasette ne kadar class olduğuyla ilgili bizim bütün datasetlerde 10 tane type var
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    # numbe rof color channels, for mnist it is grayscale so it is 1, but for cifar-10 it is 3 (rgb)
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    # if there are no improvements in certain number of rounds, it stops training
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    # training will happen in all of the clients, if we use this frac will be ignored
    parser.add_argument('--all_clients', action='store_true', help='aggregation over all clients')
    args = parser.parse_args()
    return args
