import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from snntorch import utils
import spiky.network as sn
import numpy as np


# type: ignore
def test_against_snntorch():
    # Set fixed seed for reproducibility
    torch.manual_seed(42)  # Constant seed value # type:ignore

    input_size = 8
    size = [input_size, 6, 4, 2]
    num_steps = 25
    batch_size = 3
    beta = 0.5  # The "decay" factor.
    spike_grad = surrogate.atan()  # type:ignore

    snn_net = nn.Sequential(  # type: ignore
        nn.Linear(size[0], size[1]),  # type: ignore
        snn.Leaky(beta=beta, init_hidden=True, spike_grad=spike_grad),  # type: ignore
        nn.Linear(size[1], size[2]),  # type: ignore
        snn.Leaky(beta=beta, init_hidden=True, spike_grad=spike_grad),  # type: ignore
        nn.Linear(size[2], size[3]),  # type: ignore
        snn.Leaky(beta=beta, init_hidden=True, spike_grad=spike_grad, output=True),  # type: ignore
    )

    # Extract and store constant weights/biases
    model_parameters = {}
    for name, param in snn_net.named_parameters():  # type: ignore
        model_parameters[name] = param.data.numpy()  # type: ignore
    print("model_parameters:", model_parameters)  # type:ignore

    # Simulation remains unchanged
    data_in = [
        np.random.rand(batch_size, input_size).astype(np.float32)
        for _ in range(num_steps)
    ]
    print("data_in:", data_in)

    snn_spike_recording = []
    utils.reset(snn_net)  # type: ignore
    for data in data_in:
        spike, _state = snn_net(torch.from_numpy(data))  # type: ignore
        snn_spike_recording.append(spike.detach().numpy())  # type: ignore
    print("snn_spike_recording:", snn_spike_recording)  # type:ignore

    spiky_neuron = sn.LIF(np.float32(beta))
    spiky_net = sn.FeedForwardNetwork(
        batch_size,
        input_size,
        [
            (sn.Linear(size[1]), spiky_neuron),
            (sn.Linear(size[2]), spiky_neuron),
            (sn.Linear(size[3]), spiky_neuron),
        ],
    )

    # todo set the biases on spiky net
    for i in range(len(size) - 1):
        spiky_net.layers[i].synapse_values = model_parameters[f"{2 * i}.weight"]

    spiky_spike_recording = []
    for data in data_in:
        spike = spiky_net.forward(data)
        spiky_spike_recording.append(spike)  # type:ignore
    print("spiky_spike_recording:", spiky_spike_recording)  # type:ignore
