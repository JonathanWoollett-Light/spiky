import torch  # type:ignore
import torch.nn as nn  # type:ignore
import snntorch as snn  # type:ignore
from snntorch import surrogate  # type:ignore
from snntorch import utils  # type:ignore
import spiky.network as sn
import numpy as np

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
    learning_rate = 0.01

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

        def forward(self, x):  # type:ignore
            x = self.fc1(x)
            x = self.lif1(x)
            x = self.fc2(x)
            x = self.lif2(x)
            x = self.fc3(x)
            x, mem = self.lif3(x)
            return x, mem

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
    spiky_trainer = sn.BackpropagationThroughTime(spiky_net)

    # Generate data
    data_in = [
        np.random.rand(batch_size, input_size).astype(np.float32)
        for _ in range(num_steps)
    ]
    targets = [
        np.random.rand(batch_size, output_size).astype(np.float32)
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

    # Forward pass - snnTorch
    snn_spikes = []  # type:ignore
    for data in data_in:
        spikes, _membrane = snn_net.forward(torch.from_numpy(data))  # type: ignore
        snn_spikes.append(spikes)  # type:ignore

    # Forward pass - Spiky
    spiky_trainer.forward(data_in)

    # Check spike shapes - snnTorch
    snn_spikes = torch.stack(snn_spikes)  # type:ignore
    snn_targets = torch.stack([torch.from_numpy(t) for t in targets])  # type:ignore
    assert snn_spikes.shape == snn_targets.shape

    # Backward pass - snnTorch
    snn_optimizer = torch.optim.SGD(snn_net.parameters(), lr=learning_rate)
    snn_optimizer.zero_grad()  # type:ignore
    snn_loss = nn.functional.mse_loss(snn_spikes, snn_targets)  # type:ignore
    snn_loss.backward()  # type:ignore

    snn_gradients = {}
    for name, param in snn_net.named_parameters():  # type:ignore
        if param.grad is not None:  # type:ignore
            snn_gradients[name] = np.transpose(
                param.grad.clone().detach().numpy() # type:ignore
            )
            print(
                f"\n{name}: {snn_gradients[name]}"  # type:ignore
            )

    # Backward pass - Spiky
    spiky_trainer.backward(targets)

    print(f"\ndelta_weights[0]: {spiky_trainer.delta_weights[0]}")
    print(f"\ndelta_biases[0]: {spiky_trainer.delta_biases[0]}")
    print(f"\ndelta_weights[1]: {spiky_trainer.delta_weights[1]}")
    print(f"\ndelta_biases[1]: {spiky_trainer.delta_biases[1]}")
    print(f"\ndelta_weights[2]: {spiky_trainer.delta_weights[2]}")
    print(f"\ndelta_biases[2]: {spiky_trainer.delta_biases[2]}")

    assert np.allclose(
        spiky_trainer.delta_biases[0], snn_gradients["fc1.bias"]  # type:ignore
    ), f"{spiky_trainer.delta_biases[0]}\n{snn_gradients["fc1.bias"]}"  # type:ignore
    assert np.allclose(
        spiky_trainer.delta_weights[0], snn_gradients["fc1.weight"]  # type:ignore
    ), f"{spiky_trainer.delta_weights[0]}\n{snn_gradients["fc1.weight"]}"  # type:ignore
    assert np.allclose(
        spiky_trainer.delta_biases[1], snn_gradients["fc2.bias"]  # type:ignore
    ), f"{spiky_trainer.delta_biases[1]}\n{snn_gradients["fc2.bias"]}"  # type:ignore
    assert np.allclose(
        spiky_trainer.delta_weights[1], snn_gradients["fc2.weight"]  # type:ignore
    ), f"{spiky_trainer.delta_weights[1]}\n{snn_gradients["fc2.weight"]}"  # type:ignore
    assert np.allclose(
        spiky_trainer.delta_biases[2], snn_gradients["fc3.bias"]  # type:ignore
    ), f"{spiky_trainer.delta_biases[2]}\n{snn_gradients["fc3.bias"]}"  # type:ignore
    assert np.allclose(
        spiky_trainer.delta_weights[2], snn_gradients["fc3.weight"]  # type:ignore
    ), f"{spiky_trainer.delta_weights[2]}\n{snn_gradients["fc3.weight"]}"  # type:ignore
