import subprocess
import sys

# List of parameter combinations to test
experiments = [ 
"python main_fed.py --model mlp --dataset mnist --epochs 100 --frac 0.4 --num_channels 1 --local_ep 5 --iid --gpu 0"
]
exp_cont= [
    "python main_fed.py --model resnet18 --dataset cifar --epochs 100 --frac 0.4 --num_channels 3 --local_ep 5 --iid --gpu 0",
                        "python main_fed.py --model resnet18 --dataset cifar --epochs 100 --frac 0.4 --num_channels 3 --local_ep 2 --iid --gpu 0",
                        "python main_fed.py --model resnet18 --dataset cifar --epochs 100 --frac 0.4 --num_channels 3 --local_ep 1 --iid --gpu 0",
                        "python main_fed.py --model cnn --dataset mnist --epochs 100 --frac 0.4 --num_channels 1 --local_ep 5 --iid --gpu 0",
                        "python main_fed.py --model cnn --dataset fashion_mnist --epochs 100 --frac 0.4 --num_channels 1 --local_ep 5 --iid --gpu 0",
                        "python main_fed.py --model cnn --dataset fashion_mnist --epochs 100 --frac 0.4 --num_channels 1 --local_ep 1 --iid --gpu 0",
                        "python main_fed.py --model cnn --dataset cifar --epochs 100 --frac 0.4 --num_channels 3 --local_ep 5 --iid --gpu 0",
                        "python main_fed.py --model cnn --dataset cifar --epochs 100 --frac 0.4 --num_channels 1 --local_ep 5 --iid --gpu 0",
                        "python main_fed.py --model mlp --dataset fashion_mnist --epochs 100 --frac 0.4 --num_channels 1 --local_ep 5 --iid --gpu 0",
                        "python main_fed.py --model mlp --dataset cifar --epochs 100 --frac 0.4 --num_channels 3 --local_ep 5 --iid --gpu 0",
                        "python main_fed.py --model mlp --dataset mnist --epochs 100 --frac 0.4 --num_channels 1 --local_ep 5 --iid --gpu 0",
                        "python main_fed.py --model shufflenet --dataset fashion_mnist --epochs 100 --frac 0.4 --num_channels 1 --local_ep 5 --iid --gpu 0",
                        "python main_fed.py --model shufflenet --dataset mnist --epochs 100 --frac 0.4 --num_channels 1 --local_ep 5 --iid --gpu 0",
                        "python main_fed.py --model shufflenet --dataset cifar --epochs 100 --frac 0.4 --num_channels 3 --local_ep 5 --iid --gpu 0"]
experiments_noniid = [
    "python main_fed.py --model resnet18 --dataset cifar --epochs 100 --frac 0.4 --num_channels 3 --local_ep 5 --gpu 0",
                        "python main_fed.py --model resnet18 --dataset cifar --epochs 100 --frac 0.4 --num_channels 3 --local_ep 2 --gpu 0",
                        "python main_fed.py --model resnet18 --dataset cifar --epochs 100 --frac 0.4 --num_channels 3 --local_ep 1 --gpu 0",
                        "python main_fed.py --model cnn --dataset mnist --epochs 100 --frac 0.4 --num_channels 1 --local_ep 5 --gpu 0",
                        "python main_fed.py --model cnn --dataset fashion_mnist --epochs 100 --frac 0.4 --num_channels 1 --local_ep 5 --gpu 0",
                        "python main_fed.py --model cnn --dataset fashion_mnist --epochs 100 --frac 0.4 --num_channels 1 --local_ep 1 --gpu 0",
                        "python main_fed.py --model cnn --dataset cifar --epochs 100 --frac 0.4 --num_channels 3 --local_ep 5 --gpu 0",
                        "python main_fed.py --model cnn --dataset cifar --epochs 100 --frac 0.4 --num_channels 1 --local_ep 5 --gpu 0",
                        "python main_fed.py --model mlp --dataset fashion_mnist --epochs 100 --frac 0.4 --num_channels 1 --local_ep 5 --gpu 0",
                        "python main_fed.py --model mlp --dataset cifar --epochs 100 --frac 0.4 --num_channels 3 --local_ep 5 --gpu 0",
                        "python main_fed.py --model mlp --dataset mnist --epochs 100 --frac 0.4 --num_channels 1 --local_ep 5 --gpu 0",
                        "python main_fed.py --model shufflenet --dataset fashion_mnist --epochs 100 --frac 0.4 --num_channels 1 --local_ep 5 --gpu 0",
                        "python main_fed.py --model shufflenet --dataset mnist --epochs 100 --frac 0.4 --num_channels 1 --local_ep 5 --gpu 0",
                        "python main_fed.py --model shufflenet --dataset cifar --epochs 100 --frac 0.4 --num_channels 3 --local_ep 5 --gpu 0"]

for exp in experiments:
    print(f"Running experiment: {exp}")  # Debugging output
    
    # Run with explicit Python interpreter
    cmd = [sys.executable] + exp.split()[1:]  # Use current Python interpreter
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    stdout, stderr = process.communicate()
    
    if process.returncode == 0:
        print(f"Experiment succeeded: {exp}")
        print(stdout)
    else:
        print(f"Experiment failed: {exp}")
        print(stderr)
"""geriye kalanlar: //non iidler yapılmadı, local epoch 1 kalsın
    "python main_fed.py --model cnn --dataset fashion_mnist --epochs 100 --frac 0.4 --num_channels 1 --local_ep 5 --iid --gpu 0"
    "python main_fed.py --model cnn --dataset fashion_mnist --epochs 100 --frac 0.4 --num_channels 1 --local_ep 1 --iid --gpu 0"
    "python main_fed.py --model cnn --dataset cifar --epochs 100 --frac 0.4 --num_channels 3 --local_ep 5 --iid --gpu 0"
    "python main_fed.py --model cnn --dataset cifar --epochs 100 --frac 0.4 --num_channels 1 --local_ep 5 --iid --gpu 0"
    "python main_fed.py --model mlp --dataset fashion_mnist --epochs 100 --frac 0.4 --num_channels 1 --local_ep 5 --iid --gpu 0"
    "python main_fed.py --model mlp --dataset cifar --epochs 100 --frac 0.4 --num_channels 3 --local_ep 5 --iid --gpu 0"
    "python main_fed.py --model mlp --dataset mnist --epochs 100 --frac 0.4 --num_channels 1 --local_ep 5 --iid --gpu 0"
    """
""" Yapılan deneyler:
    "python main_fed.py --model mlp --dataset mnist --epochs 100 --frac 0.4 --num_channels 1 --local_ep 5 --iid --gpu 0"
    python main_fed.py --model resnet18 --dataset cifar --epochs 100 --frac 0.4 --num_channels 3 --local_ep 5 --iid --gpu 0"
    "python main_fed.py --model resnet18 --dataset cifar --epochs 100 --frac 0.4 --num_channels 3 --local_ep 2 --iid --gpu 0"
    "python main_fed.py --model resnet18 --dataset cifar --epochs 100 --frac 0.4 --num_channels 3 --local_ep 1 --iid --gpu 0"
    "python main_fed.py --model shufflenet --dataset fashion_mnist --epochs 100 --frac 0.4 --num_channels 1 --local_ep 1 --iid --gpu 0"
    "python main_fed.py --model shufflenet --dataset mnist --epochs 100 --frac 0.4 --num_channels 1 --local_ep 1 --iid --gpu 0"
    "python main_fed.py --model shufflenet --dataset cifar --epochs 100 --frac 0.4 --num_channels 3 --local_ep 2 --iid --gpu 0"
    "python main_fed.py --model cnn --dataset mnist --epochs 100 --frac 0.4 --num_channels 1 --local_ep 1 --iid --gpu 0"
    """
# cnn-> mnist
# shufflenet -> fashion_mnist
# resnet-> cifar
""" Non iid için deneyler:
    "python main_fed.py --model shufflenet --dataset fashion_mnist --epochs 100 --frac 0.4 --num_channels 1 --local_ep 1 --dirichlet_alpha 0.3 --gpu 0"
    "python main_fed.py --model shufflenet --dataset fashion_mnist --epochs 100 --frac 0.4 --num_channels 1 --local_ep 1 --dirichlet_alpha 0.6 --gpu 0"
    "python main_fed.py --model shufflenet --dataset fashion_mnist --epochs 100 --frac 0.4 --num_channels 1 --local_ep 1 --dirichlet_alpha 0.9 --gpu 0"
    "python main_fed.py --model shufflenet --dataset fashion_mnist --epochs 100 --frac 0.4 --num_channels 1 --local_ep 1 --partition_noniid 3 --gpu 0"
    "python main_fed.py --model shufflenet --dataset fashion_mnist --epochs 100 --frac 0.4 --num_channels 1 --local_ep 1 --partition_noniid 6 --gpu 0"
    "python main_fed.py --model shufflenet --dataset fashion_mnist --epochs 100 --frac 0.4 --num_channels 1 --local_ep 1 --partition_noniid 9 --gpu 0"
    "python main_fed.py --model resnet --dataset cifar --epochs 100 --frac 0.4 --num_channels 3 --local_ep 1 --dirichlet_alpha 0.3 --gpu 0"
    "python main_fed.py --model resnet --dataset cifar --epochs 100 --frac 0.4 --num_channels 3 --local_ep 1 --dirichlet_alpha 0.6 --gpu 0"
    "python main_fed.py --model resnet --dataset cifar --epochs 100 --frac 0.4 --num_channels 3 --local_ep 1 --dirichlet_alpha 0.9 --gpu 0"
    "python main_fed.py --model resnet --dataset cifar --epochs 100 --frac 0.4 --num_channels 3 --local_ep 1 --partition_noniid 3 --gpu 0"
    "python main_fed.py --model resnet --dataset cifar --epochs 100 --frac 0.4 --num_channels 3 --local_ep 1 --partition_noniid 6 --gpu 0"
    "python main_fed.py --model resnet --dataset cifar --epochs 100 --frac 0.4 --num_channels 3 --local_ep 1 --partition_noniid 9 --gpu 0"
    "python main_fed.py --model cnn --dataset mnist --epochs 100 --frac 0.4 --num_channels 1 --local_ep 1 --dirichlet_alpha 0.3 --gpu 0"
    "python main_fed.py --model cnn --dataset mnist --epochs 100 --frac 0.4 --num_channels 1 --local_ep 1 --dirichlet_alpha 0.6 --gpu 0"
    "python main_fed.py --model cnn --dataset mnist --epochs 100 --frac 0.4 --num_channels 1 --local_ep 1 --dirichlet_alpha 0.9 --gpu 0"
    "python main_fed.py --model cnn --dataset mnist --epochs 100 --frac 0.4 --num_channels 1 --local_ep 1 --partition_noniid 3 --gpu 0" //test için bunu çalıştırdım
    "python main_fed.py --model cnn --dataset mnist --epochs 100 --frac 0.4 --num_channels 1 --local_ep 1 --partition_noniid 6 --gpu 0"
    "python main_fed.py --model cnn --dataset mnist --epochs 100 --frac 0.4 --num_channels 1 --local_ep 1 --partition_noniid 9 --gpu 0"
    """