import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from snntorch import utils
import spiky.network as sn
import numpy as np
from numpy.typing import NDArray


# type: ignore
def test_against_snntorch():
    # Set fixed seed for reproducibility
    torch.manual_seed(42)  # Constant seed value # type:ignore

    input_size = 8
    size = [input_size, 6, 4, 2]
    depth = len(size) - 1
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
    snn_parameters = {}
    for name, param in snn_net.state_dict().items(): # type: ignore
        print("params: ", name)
        snn_parameters[name] = param.numpy() # type: ignore
    print("snn_parameters:", snn_parameters)  # type:ignore


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
    
    i = 0
    weights: list[NDArray[np.float32]] = []
    biases: list[NDArray[np.float32]] = []
    while True:
        if not (f"{i}.weight" in snn_parameters):
            assert not (f"{i}.bias" in snn_parameters)
            break
        weights.append(snn_parameters[f"{i}.weight"]) # type: ignore
        biases.append(snn_parameters[f"{i}.bias"]) # type: ignore
        i += 1
    assert len(weights) == depth
    assert len(biases) == depth

    for i in range(depth):
        assert spiky_net.layers[i].synapse_weights.shape == weights[i].T.shape  # type:ignore
        assert spiky_net.layers[i].synapse_biases.shape == biases[i].T.shape  # type:ignore
        spiky_net.layers[i].synapse_weights = weights[i].T
        spiky_net.layers[i].synapse_biases = biases[i].T

    spiky_spike_recording = []
    for data in data_in:
        spike = spiky_net.forward(data)
        spiky_spike_recording.append(spike)  # type:ignore
    print("spiky_spike_recording:", spiky_spike_recording)  # type:ignore
    assert snn_spike_recording == spiky_spike_recording
