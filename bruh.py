import csv
import sys
import numpy as np
import torch
import torch.nn as nn
import snntorch as snn

# Increase the CSV field size limit
csv.field_size_limit(10**9)

# Define your model parameters (adjust as needed)
beta = 0.9  # Example value, change to your actual beta
spike_grad = None  # Example value, change to your actual spike_grad
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the model
net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(2312, 784),
    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
    nn.Linear(784, 784),
    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
    nn.Linear(784, 392),
    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
    nn.Linear(392, 196),
    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
    nn.Linear(196, 10),
    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True),
).to(device)

# Load parameters from CSV
net_params = {}

with open("./tmp/params.csv", "r") as f:
    reader = csv.reader(f)
    next(reader)  # Skip the header row
    for row in reader:
        name = row[0]
        shape = eval(row[1])  # Convert string representation of tuple back to tuple
        values_str = row[2]

        # Convert space-separated string back to numpy array
        values = np.array(list(map(float, values_str.split())))

        # Reshape to original shape and convert to tensor
        net_params[name] = torch.tensor(values.reshape(shape), dtype=torch.float32)

# Load parameters into the model
net.load_state_dict(net_params, strict=False)

# Save in PyTorch native format
torch.save(net.state_dict(), "./tmp/params.pth")

print("Successfully converted CSV parameters to PyTorch .pth format!")
print(f"Saved to: ./tmp/params.pth")
