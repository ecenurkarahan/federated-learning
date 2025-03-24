#!/bin/bash

experiments=(
    "python main_fed.py --model resnet18 --dataset cifar --epochs 100 --frac 0.4 --num_channels 3 --local_ep 1 --iid --gpu 0"
    "python main_fed.py --model shufflenet --dataset fashion_mnist --epochs 100 --frac 0.4 --num_channels 1 --local_ep 1 --iid --gpu 0"
    "python main_fed.py --model shufflenet --dataset mnist --epochs 100 --frac 0.4 --num_channels 1 --local_ep 1 --iid --gpu 0"
    "python main_fed.py --model shufflenet --dataset cifar --epochs 100 --frac 0.4 --num_channels 3 --local_ep 2 --iid --gpu 0"
    "python main_fed.py --model cnn --dataset mnist --epochs 100 --frac 0.4 --num_channels 1 --local_ep 5 --iid --gpu 0"
    "python main_fed.py --model cnn --dataset fashion_mnist --epochs 100 --frac 0.4 --num_channels 1 --local_ep 5 --iid --gpu 0"
    "python main_fed.py --model cnn --dataset fashion_mnist --epochs 100 --frac 0.4 --num_channels 1 --local_ep 1 --iid --gpu 0"
    "python main_fed.py --model cnn --dataset cifar --epochs 100 --frac 0.4 --num_channels 3 --local_ep 5 --iid --gpu 0"
    "python main_fed.py --model cnn --dataset cifar --epochs 100 --frac 0.4 --num_channels 1 --local_ep 5 --iid --gpu 0"
    "python main_fed.py --model mlp --dataset fashion_mnist --epochs 100 --frac 0.4 --num_channels 1 --local_ep 5 --iid --gpu 0"
    "python main_fed.py --model mlp --dataset cifar --epochs 100 --frac 0.4 --num_channels 3 --local_ep 5 --iid --gpu 0"
    "python main_fed.py --model mlp --dataset mnist --epochs 100 --frac 0.4 --num_channels 1 --local_ep 5 --iid --gpu 0"
)

for p in "${experiments[@]}"; do
    echo "Running: $p"
    eval "$p"
done
