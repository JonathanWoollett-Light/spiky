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

        def forward(self, x):  # type:ignore
            cur1 = self.fc1(x)
            spk1 = self.lif1(cur1)
            mem1 = self.lif1.mem  # type:ignore

            cur2 = self.fc2(spk1)
            spk2 = self.lif2(cur2)
            mem2 = self.lif2.mem  # type:ignore

            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3)  # output layer returns both

            return [(spk1, mem1), (spk2, mem2), (spk3, mem3)]  # type:ignore

    snn_net = SNNTorchNet()

    # Extract and store constant weights/biases
    snn_parameters = {}
    for name, param in snn_net.state_dict().items():  # type: ignore
        print("params: ", name)
        snn_parameters[name] = param.numpy()  # type: ignore
    print("snn_parameters:", snn_parameters)  # type:ignore

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
        # SNNTorch forward pass
        # When output=True, SNNTorch returns (spikes, membrane_potential)
        snn_outputs = snn_net(torch.from_numpy(data))  # type:ignore

        # Spiky forward pass - returns spikes
        spiky_outputs = []
        spikes = data
        for layer in spiky_net.layers:
            layer.forward(spikes)
            spiky_outputs.append(
                [layer.spike_values.copy(), layer.neuron_values.copy()]
            )  # type:ignore
            spikes = layer.spike_values.copy()

        print(f"\n=== Timestep {step} ===")

        # Get membrane potentials from each layer in spiky_net
        for layer_index in range(depth):
            snn_spike_tensor, snn_mem_tensor = snn_outputs[layer_index]
            snn_spike = snn_spike_tensor.detach().numpy()
            snn_mem = snn_mem_tensor.detach().numpy()

            spiky_spike, spiky_mem = spiky_outputs[layer_index]  # type:ignore

            print(f"\nLayer {layer_index}:")
            print(f"  SNNTorch spikes: {snn_spike}")
            print(f"  Spiky spikes: {spiky_spike}")
            print(f"  SNNTorch membrane: {snn_mem-snn_spike}")
            print(f"  Spiky membrane: {spiky_mem}")

            # Validate spikes
            assert np.allclose(
                snn_spike, spiky_spike, rtol=1e-5, atol=1e-6
            ), f"Layer {layer_index}, Timestep {step}: Spikes don't match!"

            # Validate membrane potentials
            # For some reason, SNNTorch doesn't update membrane potential after
            # a spike, it waits until the next forward pass to apply the
            # threshold reset. Thus when a spike occurs, spiky will apply the
            # threshold reset immediately while SNNTorch will not so we need to
            # consider that here.
            assert np.allclose(
                snn_mem - snn_spike, spiky_mem, rtol=1e-5, atol=1e-6
            ), f"Layer {layer_index}, Timestep {step}: Membrane potentials don't match!"

    print("\n=== All timesteps passed! ===")
    print("Both spikes and membrane potentials match between SNNTorch and Spiky.")
