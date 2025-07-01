import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from snntorch import utils


def test_against_snntorch():
    # Set fixed seed for reproducibility
    torch.manual_seed(42)  # Constant seed value # type:ignore

    num_steps = 25
    batch_size = 1
    beta = 0.5
    spike_grad = surrogate.fast_sigmoid()  # type:ignore

    net = nn.Sequential(
        nn.Conv2d(1, 8, 5),
        nn.MaxPool2d(2),
        snn.Leaky(beta=beta, init_hidden=True, spike_grad=spike_grad),
        nn.Conv2d(8, 16, 5),
        nn.MaxPool2d(2),
        snn.Leaky(beta=beta, init_hidden=True, spike_grad=spike_grad),
        nn.Flatten(),
        nn.Linear(16 * 4 * 4, 10),
        snn.Leaky(beta=beta, init_hidden=True, spike_grad=spike_grad, output=True),
    )

    # Extract and store constant weights/biases
    weights_biases = {}
    for name, param in net.named_parameters():
        weights_biases[name] = param.data.clone()

    # Simulation remains unchanged
    data_in = torch.rand(num_steps, batch_size, 1, 28, 28)
    spike_recording = []
    utils.reset(net)  # type: ignore

    for step in range(num_steps):
        spike, _state = net(data_in[step])
        spike_recording.append(spike)  # type: ignore

    print("spike_recording:", spike_recording)  # type:ignore
    print("weights_biases:", weights_biases)  # type:ignore
