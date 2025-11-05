import torch  # type:ignore
import torch.nn as nn  # type:ignore
import snntorch as snn  # type:ignore
from snntorch import surrogate  # type:ignore
from snntorch import utils  # type:ignore
from numpy.typing import NDArray
import spiky.network as sn
import numpy as np
from numpy import float32

R = 1e-5
A = 1e-6


def tnp(tensor):  # type:ignore
    return tensor.detach().numpy()  # type:ignore


# type: ignore
def test_against_snntorch():
    # Set fixed seeds for reproducibility
    torch.manual_seed(42)  # type:ignore
    np.random.seed(0)

    input_size = 8
    output_size = 2
    size = [input_size, 6, 4, output_size]
    num_steps = 4
    batch_size = 3
    beta = 0.5  # The "decay" factor.
    spike_grad = surrogate.atan()  # type:ignore

    class SNNTorchNet(nn.Module):  # type:ignore
        def __init__(self):
            super().__init__()  # type:ignore
            self.fc1 = nn.Linear(size[0], size[1])  # type:ignore
            self.lif1 = snn.Leaky(  # type:ignore
                beta=beta, init_hidden=True, spike_grad=spike_grad, reset_delay=False
            )
            self.fc2 = nn.Linear(size[1], size[2])  # type:ignore
            self.lif2 = snn.Leaky(  # type:ignore
                beta=beta, init_hidden=True, spike_grad=spike_grad, reset_delay=False
            )
            self.fc3 = nn.Linear(size[2], size[3])  # type:ignore
            self.lif3 = snn.Leaky(  # type:ignore
                beta=beta,
                init_hidden=True,
                spike_grad=spike_grad,
                reset_delay=False,
                output=True,
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

    # Generate data
    data_in = [
        np.random.rand(batch_size, input_size).astype(np.float32)
        for _ in range(num_steps)
    ]

    snn_parameters = {}
    for name, param in snn_net.state_dict().items():  # type:ignore
        snn_parameters[name] = param.numpy()  # type:ignore

    layer_names = ["fc1", "fc2", "fc3"]
    for i, layer_name in enumerate(layer_names):
        weight = snn_parameters[f"{layer_name}.weight"]  # type: ignore
        bias = snn_parameters[f"{layer_name}.bias"]  # type: ignore
        spiky_net.layers[i].synapse_weights = weight.T  # type: ignore
        spiky_net.layers[i].synapse_biases = bias.T  # type: ignore

    utils.reset(snn_net)  # type: ignore
    snn_spikes = []
    spiky_spikes: list[NDArray[float32]] = []

    # Run simulation and validate both spikes and membrane potentials
    print()
    for step, data in enumerate(data_in):
        print(f"Timestep {step}")
        print(f"Layer 1")

        # Layer 1 - snnTorch
        snn_1_in = snn_net.fc1(torch.from_numpy(data))  # type:ignore
        snn_1_spikes = snn_net.lif1(snn_1_in)  # type:ignore
        snn_1_mem = snn_net.lif1.mem  # type:ignore
        snn_1_in_np = tnp(snn_1_in)  # type:ignore
        snn_1_spikes_np = tnp(snn_1_spikes)  # type:ignore
        snn_1_mem_np = tnp(snn_1_mem)  # type:ignore

        # Layer 1 - Spiky
        spiky_net.layers[0].forward(data)
        spiky_1_in = spiky_net.layers[0].weighted_input_values.copy()
        spiky_1_spikes = spiky_net.layers[0].spike_values.copy()
        spiky_1_mem = spiky_net.layers[0].neuron_values.copy()

        # snnTorch applies the threshold reset on the next forward pass, while
        # spiky applies it immediately after the forward pass which resulted in
        # the membrane potential breaching the threshold.
        # As such to compare the values properly we apply the threshold reset
        # ourselves by simply subtracting the spikes in the comparison.

        # Compare layer 1
        assert np.allclose(
            snn_1_in_np, spiky_1_in, R, A  # type:ignore
        ), f"{snn_1_in_np}\n{spiky_1_in}"
        assert np.allclose(
            snn_1_spikes_np, spiky_1_spikes, R, A  # type:ignore
        ), f"{snn_1_spikes_np}\n{spiky_1_spikes}"
        assert np.allclose(
            snn_1_mem_np, spiky_1_mem, R, A  # type:ignore
        ), f"{snn_1_mem_np}\n{spiky_1_mem}"  # type:ignore

        print(f"Layer 2")

        # Layer 2 - snnTorch
        snn_2_in = snn_net.fc2(snn_1_spikes)  # type:ignore
        snn_2_spikes = snn_net.lif2(snn_2_in)  # type:ignore
        snn_2_mem = snn_net.lif2.mem  # type:ignore
        snn_2_in_np = tnp(snn_2_in)  # type:ignore
        snn_2_spikes_np = tnp(snn_2_spikes)  # type:ignore
        snn_2_mem_np = tnp(snn_2_mem)  # type:ignore

        # Layer 2 - Spiky
        spiky_net.layers[1].forward(spiky_1_spikes)
        spiky_2_in = spiky_net.layers[1].weighted_input_values.copy()
        spiky_2_spikes = spiky_net.layers[1].spike_values.copy()
        spiky_2_mem = spiky_net.layers[1].neuron_values.copy()

        # Compare layer 2
        assert np.allclose(
            snn_2_in_np, spiky_2_in, R, A  # type:ignore
        ), f"{snn_2_in_np}\n{spiky_2_in}"
        assert np.allclose(
            snn_2_spikes_np, spiky_2_spikes, R, A  # type:ignore
        ), f"{snn_2_spikes_np}\n{spiky_2_spikes}"
        assert np.allclose(
            snn_2_mem_np, spiky_2_mem, R, A  # type:ignore
        ), f"{snn_2_mem_np}\n{spiky_2_mem}"  # type:ignore

        print(f"Layer 3")

        # Layer 3 - snnTorch
        snn_3_in = snn_net.fc3(snn_2_spikes)  # type:ignore
        snn_3_spikes, snn_3_mem = snn_net.lif3(snn_3_in)  # type:ignore
        snn_3_mem_inner = snn_net.lif3.mem  # type:ignore
        snn_3_in_np = tnp(snn_3_in)  # type:ignore
        snn_3_spikes_np = tnp(snn_3_spikes)  # type:ignore
        snn_3_mem_np = tnp(snn_3_mem)  # type:ignore
        snn_spikes.append(snn_3_spikes)  # type:ignore

        # Check that the output membrane potential is the same as the stored membrane potential.
        assert np.allclose(
            tnp(snn_3_mem),  # type:ignore
            tnp(snn_3_mem_inner),  # type:ignore
            rtol=1e-5,
            atol=1e-6,
        )  # type:ignore

        # Layer 3 - Spiky
        spiky_net.layers[2].forward(spiky_2_spikes)
        spiky_3_in = spiky_net.layers[2].weighted_input_values.copy()
        spiky_3_spikes = spiky_net.layers[2].spike_values.copy()
        spiky_3_mem = spiky_net.layers[2].neuron_values.copy()
        spiky_spikes.append(spiky_3_spikes)

        # Compare layer 3
        assert np.allclose(
            snn_3_in_np, spiky_3_in, R, A  # type:ignore
        ), f"{snn_3_in_np}\n{spiky_3_in}"
        assert np.allclose(
            snn_3_spikes_np, spiky_3_spikes, R, A  # type:ignore
        ), f"{snn_3_spikes_np}\n{spiky_3_spikes}"
        assert np.allclose(
            snn_3_mem_np, spiky_3_mem, R, A  # type:ignore
        ), f"{snn_3_mem_np}\n{spiky_3_mem}"  # type:ignore
