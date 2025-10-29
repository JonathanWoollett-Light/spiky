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
    # Set fixed seeds for reproducibility
    torch.manual_seed(42)  # type:ignore
    np.random.seed(0)

    input_size = 8
    size = [input_size, 6, 4, 2]
    num_steps = 25
    batch_size = 3
    beta = 0.5  # The "decay" factor.
    spike_grad = surrogate.atan()  # type:ignore

    class SNNTorchNet(nn.Module):
        def __init__(self):
            super().__init__()  # type:ignore
            self.fc1 = nn.Linear(size[0], size[1])
            self.lif1 = snn.Leaky(beta=beta, init_hidden=True, spike_grad=spike_grad)
            self.fc2 = nn.Linear(size[1], size[2])
            self.lif2 = snn.Leaky(beta=beta, init_hidden=True, spike_grad=spike_grad)
            self.fc3 = nn.Linear(size[2], size[3])
            self.lif3 = snn.Leaky(
                beta=beta, init_hidden=True, spike_grad=spike_grad, output=True
            )

    snn_net = SNNTorchNet()

    # Extract and store constant weights/biases
    snn_parameters = {}
    for name, param in snn_net.state_dict().items():  # type: ignore
        snn_parameters[name] = param.numpy()  # type: ignore

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

    # Simulation remains unchanged
    data_in = [
        np.random.rand(batch_size, input_size).astype(np.float32)
        for _ in range(num_steps)
    ]

    snn_parameters = {}
    for name, param in snn_net.state_dict().items():
        snn_parameters[name] = param.numpy()

    layer_names = ["fc1", "fc2", "fc3"]
    for i, layer_name in enumerate(layer_names):
        weight = snn_parameters[f"{layer_name}.weight"]  # type: ignore
        bias = snn_parameters[f"{layer_name}.bias"]  # type: ignore
        spiky_net.layers[i].synapse_weights = weight.T  # type: ignore
        spiky_net.layers[i].synapse_biases = bias.T  # type: ignore

    utils.reset(snn_net)  # type: ignore
    # Run simulation and validate both spikes and membrane potentials
    for step, data in enumerate(data_in):
        print(f"Timestep {step}")
        print(f"Layer 1")

        # Layer 1 - SNNTorch
        snn_1_in = snn_net.fc1(torch.from_numpy(data))  # type:ignore
        snn_1_spikes = snn_net.lif1(snn_1_in)
        snn_1_mem = snn_net.lif1.mem  # type:ignore

        # Layer 1 - Spiky
        spiky_net.layers[0].forward(data)
        spiky_1_spikes = spiky_net.layers[0].spike_values.copy()
        spiky_1_mem = spiky_net.layers[0].neuron_values.copy()

        # SNNTorch applies the threshold reset on the next forward pass, while
        # spiky applies it immediately after the forward pass which resulted in
        # the membrane potential breaching the threshold.
        # As such to compare the values properly we apply the threshold reset
        # ourselves by simply subtracting the spikes in the comparison.

        # Compare layer 1
        snn_1_spikes_np = snn_1_spikes.detach().numpy()
        snn_1_mem_np = snn_1_mem.detach().numpy() - snn_1_spikes_np  # type:ignore
        assert np.allclose(
            snn_1_spikes_np, spiky_1_spikes, rtol=1e-5, atol=1e-6
        ), f"{snn_1_spikes_np}\n{spiky_1_spikes}"
        assert np.allclose(
            snn_1_mem_np, spiky_1_mem, rtol=1e-5, atol=1e-6  # type:ignore
        ), f"{snn_1_mem_np}\n{spiky_1_mem}"  # type:ignore

        print(f"Layer 2")

        # Layer 2 - SNNTorch
        snn_2_in = snn_net.fc2(snn_1_spikes)
        snn_2_spikes = snn_net.lif2(snn_2_in)
        snn_2_mem = snn_net.lif2.mem  # type:ignore

        # Layer 2 - Spiky
        spiky_net.layers[1].forward(spiky_1_spikes)
        spiky_2_spikes = spiky_net.layers[1].spike_values.copy()
        spiky_2_mem = spiky_net.layers[1].neuron_values.copy()

        # Compare layer 2
        snn_2_spikes_np = snn_2_spikes.detach().numpy()
        snn_2_mem_np = snn_2_mem.detach().numpy() - snn_2_spikes_np  # type:ignore
        assert np.allclose(
            snn_2_spikes_np, spiky_2_spikes, rtol=1e-5, atol=1e-6
        ), f"{snn_2_spikes_np}\n{spiky_2_spikes}"
        assert np.allclose(
            snn_2_mem_np, spiky_2_mem, rtol=1e-5, atol=1e-6  # type:ignore
        ), f"{snn_2_mem_np}\n{spiky_2_mem}"  # type:ignore

        print(f"Layer 3")

        # Layer 3 - SNNTorch
        snn_3_in = snn_net.fc3(snn_2_spikes)
        snn_3_spikes, snn_3_mem = snn_net.lif3(snn_3_in)
        snn_3_mem_inner = snn_net.lif3.mem  # type:ignore
        assert np.allclose(
            snn_3_mem.detach().numpy(),
            snn_3_mem_inner.detach().numpy(),  # type:ignore
            rtol=1e-5,
            atol=1e-6,
        )  # type:ignore

        # Layer 3 - Spiky
        spiky_net.layers[2].forward(spiky_2_spikes)
        spiky_3_spikes = spiky_net.layers[2].spike_values.copy()
        spiky_3_mem = spiky_net.layers[2].neuron_values.copy()

        # Compare layer 3
        snn_3_spikes_np = snn_3_spikes.detach().numpy()
        snn_3_mem_np = snn_3_mem.detach().numpy() - snn_3_spikes_np
        assert np.allclose(
            snn_3_spikes_np, spiky_3_spikes, rtol=1e-5, atol=1e-6
        ), f"{snn_3_spikes_np}\n{spiky_3_spikes}"
        assert np.allclose(
            snn_3_mem_np, spiky_3_mem, rtol=1e-5, atol=1e-6
        ), f"{snn_3_mem_np}\n{spiky_3_mem}"  # type:ignore
