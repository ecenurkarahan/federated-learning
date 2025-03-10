#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models

# this file contains the neural network models used in the project
# mlp is dataset independent, cnn is dataset dependent
class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return x

# model arch öner yeni model arch oluştur
class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, args.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ShuffleNetV2Mnist(nn.Module):
    def __init__(self, args):
        super(ShuffleNetV2Mnist, self).__init__()
        
        # Initialize ShuffleNetV2 with pre-trained weights
        self.shufflenet = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
        
        # Replace the first convolutional layer to handle grayscale images
        self.shufflenet.conv1[0] = nn.Conv2d(args.num_channels, 24, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Replace the last fully connected layer
        self.shufflenet.fc = nn.Linear(1024, args.num_classes)
    
    def forward(self, x):
        return self.shufflenet(x)

class ShuffleNetV2Cifar(nn.Module):
    def __init__(self, args):
        super(ShuffleNetV2Cifar, self).__init__()
        
        # Initialize ShuffleNetV2 with pre-trained weights
        self.shufflenet = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
        
        # Replace the first convolutional layer to handle the smaller images of CIFAR-10
        self.shufflenet.conv1[0] = nn.Conv2d(args.num_channels, 24, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Replace the last fully connected layer
        self.shufflenet.fc = nn.Linear(1024, args.num_classes)
    
    def forward(self, x):
        return self.shufflenet(x)
