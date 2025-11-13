import torch  # type:ignore
import torch.nn as nn  # type:ignore
import snntorch as snn  # type:ignore
from snntorch import surrogate  # type:ignore
from snntorch import utils  # type:ignore
import spiky.network as sn
import numpy as np


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
    num_steps = 2
    batch_size = 2
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

    forward_values = {}
    backward_gradients = {}

    # Hook counters for tracking execution order
    hook_counter = {"forward": 0, "backward": 0}

    def make_forward_hook(name):
        """Create a hook that captures forward pass values"""

        def hook(module, input, output):
            hook_counter["forward"] += 1
            if isinstance(output, tuple):
                # For modules that return (spike, membrane)
                forward_values[f"{name}_spike"] = tnp(output[0])
                forward_values[f"{name}_membrane"] = tnp(output[1])
                print(f"[FORWARD {hook_counter['forward']}] {name}")
                print(
                    f"  Spike shape: {output[0].shape}, mean: {output[0].mean():.6f}, std: {output[0].std():.6f}"
                )
                print(
                    f"  Membrane shape: {output[1].shape}, mean: {output[1].mean():.6f}, std: {output[1].std():.6f}"
                )
            else:
                forward_values[name] = tnp(output)
                print(f"[FORWARD {hook_counter['forward']}] {name}")
                print(
                    f"  Shape: {output.shape}, mean: {output.mean():.6f}, std: {output.std():.6f}"
                )

        return hook

    def make_backward_hook(name):
        """Create a hook that captures backward pass gradients"""

        def hook(module, grad_input, grad_output):
            hook_counter["backward"] += 1
            print(f"\n[BACKWARD {hook_counter['backward']}] {name}")

            # grad_output is gradient w.r.t. output of this module
            if grad_output[0] is not None:
                backward_gradients[f"{name}_grad_output"] = tnp(grad_output[0])
                print(
                    f"  Grad output shape: {grad_output[0].shape}, mean: {grad_output[0].mean():.6f}, std: {grad_output[0].std():.6f}"
                )

            # grad_input is gradient w.r.t. input of this module
            if isinstance(grad_input, tuple):
                for idx, gi in enumerate(grad_input):
                    if gi is not None:
                        backward_gradients[f"{name}_grad_input_{idx}"] = tnp(gi)
                        print(
                            f"  Grad input[{idx}] shape: {gi.shape}, mean: {gi.mean():.6f}, std: {gi.std():.6f}"
                        )
            elif grad_input is not None:
                backward_gradients[f"{name}_grad_input"] = tnp(grad_input)
                print(f"  Grad input shape: {grad_input.shape}")

        return hook

    # Register hooks on all layers
    snn_net.fc1.register_forward_hook(make_forward_hook("fc1"))
    snn_net.fc1.register_full_backward_hook(make_backward_hook("fc1"))

    snn_net.lif1.register_forward_hook(make_forward_hook("lif1"))
    snn_net.lif1.register_full_backward_hook(make_backward_hook("lif1"))

    snn_net.fc2.register_forward_hook(make_forward_hook("fc2"))
    snn_net.fc2.register_full_backward_hook(make_backward_hook("fc2"))

    snn_net.lif2.register_forward_hook(make_forward_hook("lif2"))
    snn_net.lif2.register_full_backward_hook(make_backward_hook("lif2"))

    snn_net.fc3.register_forward_hook(make_forward_hook("fc3"))
    snn_net.fc3.register_full_backward_hook(make_backward_hook("fc3"))

    snn_net.lif3.register_forward_hook(make_forward_hook("lif3"))
    snn_net.lif3.register_full_backward_hook(make_backward_hook("lif3"))

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

    # Loss - snnTorch
    snn_optimizer = torch.optim.SGD(snn_net.parameters(), lr=learning_rate)
    snn_optimizer.zero_grad()  # type:ignore
    snn_loss = nn.functional.mse_loss(snn_spikes, snn_targets)  # type:ignore

    # Loss - Spiky
    spiky_loss = ((spiky_trainer.network.layers[-1].spike_values - targets) ** 2).mean()

    # Check losses match
    assert np.allclose(tnp(snn_loss), spiky_loss), f"{snn_loss}\n{spiky_loss}"

    # Backward pass - snnTorch
    snn_loss.backward()  # type:ignore

    # Backward pass - Spiky
    spiky_trainer.backward(targets)

    # Gather gradients - snnTorch
    snn_gradients = {}
    for name, param in snn_net.named_parameters():  # type:ignore
        if param.grad is not None:  # type:ignore
            snn_gradients[name] = np.transpose(
                param.grad.clone().detach().numpy()  # type:ignore
            )

            # Print snNTorch gradients for debugging
            print(
                f"\n{name}: {snn_gradients[name]}"  # type:ignore
            )

    # Print Spiky gradients for debugging
    print(f"\ndelta_weights[0]: {spiky_trainer.delta_weights[0]}")
    print(f"\ndelta_biases[0]: {spiky_trainer.delta_biases[0]}")
    print(f"\ndelta_weights[1]: {spiky_trainer.delta_weights[1]}")
    print(f"\ndelta_biases[1]: {spiky_trainer.delta_biases[1]}")
    print(f"\ndelta_weights[2]: {spiky_trainer.delta_weights[2]}")
    print(f"\ndelta_biases[2]: {spiky_trainer.delta_biases[2]}")

    # Check gradients match
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
