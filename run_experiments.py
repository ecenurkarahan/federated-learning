import os
import time
import subprocess

# List of parameter combinations to test
experiments = [
    {"dataset": "mnist", "model": "cnn", "epochs": 100, "iid": True,"fracs":0.4,"local_ep":5,"num_channels":1},
    {"dataset": "mnist", "model": "mlp", "epochs": 100, "iid": True, "fracs":0.4, "local_ep":5, "num_channels":1},
    {"dataset": "cifar", "model": "cnn", "epochs": 100, "iid": True, "fracs":0.4, "local_ep":5, "num_channels":3},
    {"dataset": "fashion_mnist", "model": "shufflenet", "epochs": 30, "iid": True, "fracs":0.4,"local_ep":1,"num_channels":1},
]

log_file = "experiment_results.txt"

for exp in experiments:
    cmd = f"python main_fed.py --dataset {exp['dataset']} --model {exp['model']} --epochs {exp['epochs']} --fracs {exp['fracs']} --local_ep {exp['local_ep']} --num_channels {exp['num_channels']}"
    if exp["iid"]:
        cmd += " --iid"
    
    print(f"\nRunning: {cmd}")
    start_time = time.time()
    
    # Run the main script and capture the output
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Extract accuracy from stdout (assuming the script prints accuracy like "Testing accuracy: 95.2")
    acc_line = [line for line in stdout.split("\n") if "Testing accuracy" in line]
    accuracy = acc_line[0].split(":")[1].strip() if acc_line else "N/A"

    # Append results to log file
    with open(log_file, "a") as f:
        f.write(f"Dataset: {exp['dataset']}, Model: {exp['model']}, Epochs: {exp['epochs']}, IID: {exp['iid']}, Accuracy: {accuracy}, Time: {elapsed_time:.2f} sec\n")

    print(f"Experiment completed! Accuracy: {accuracy}, Time: {elapsed_time:.2f} sec")
