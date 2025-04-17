#!/bin/bash

experiments=(
    "python main_fed.py --model shufflenet --dataset fashion_mnist --epochs 100 --frac 0.4 --num_channels 1 --local_ep 1 --dirichlet_alpha 1.0 --gpu 0"
    "python main_fed.py --model shufflenet --dataset fashion_mnist --epochs 100 --frac 0.4 --num_channels 1 --local_ep 1 --partition_noniid 2 --gpu 0"
)

for p in "${experiments[@]}"; do
    echo "Running: $p"
    eval "$p"
done
