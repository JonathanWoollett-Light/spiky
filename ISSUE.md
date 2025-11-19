I'm continuing to work on my snnTorch copy and I've been stuck on the gradients mismatching for quite a while now. Below is an example of the gradient values in the output layer of snnTorch and my library (spiky). The forward values match but they immedately stop matching on the first back step. You can run the test with `poetry run pytest .\tests\test_against_snntorch_backprop.py -s`.

For `setup run:
```
poetry install
poetry run pip install torch torchvision
````

Partial test output:
```
snnTorch forward lif3
        (tensor([[ 0.1057, -0.1275],
        [ 0.1057, -0.1275]], grad_fn=<BackwardHookFunctionBackward>),)
        (tensor([[0., 0.],
        [0., 0.]], grad_fn=<MulBackward0>), tensor([[ 0.1586, -0.1912],
        [ 0.1586, -0.1912]], grad_fn=<SubBackward0>))
snnTorch backward lif3
        (tensor([[-0.0192, -0.0103],
        [-0.0295, -0.0114]]),)
        (tensor([[-0.1530, -0.1542],
        [-0.2359, -0.1705]]), None)
snnTorch backward fc3
        (tensor([[-0.0080,  0.0079, -0.0032,  0.0007],
        [-0.0111,  0.0104, -0.0061,  0.0018]]),)
        (tensor([[-0.0192, -0.0103],
        [-0.0295, -0.0114]]),)
...
spiky backward:
        spike_values: [0. 0. 0. 0.]
        weighted_input_values: [ 0.10570377 -0.12747937  0.10570377 -0.12747937]
        membrane_potentials: [ 0.15855566 -0.19121906  0.15855566 -0.19121906]
        pre_spike_membrane_potentials: [ 0.15855566 -0.19121906  0.15855566 -0.19121906]
        gradient: [0.25503132 0.23390011 0.25503132 0.23390011]
        self.errors: [-0.07805179 -0.07215047 -0.12034266 -0.07973892]
spiky backward:
        spike_values: [0. 0. 0. 0.]
        weighted_input_values: [ 0.10570377 -0.12747937  0.10570377 -0.12747937]
        membrane_potentials: [ 0.10570377 -0.12747937  0.10570377 -0.12747937]
        pre_spike_membrane_potentials: [ 0.10570377 -0.12747937  0.10570377 -0.12747937]
        gradient: [0.28669438 0.27431265 0.28669438 0.27431265]
        self.errors: [-0.06538787 -0.07796431 -0.00269346 -0.08471261]
```

The output error is calculated with:
```python
# output_layer_index
ol = self.network.layers[-1]

# MSE gradient
self.errors[-1] = (
    float32(2) * (spike_values[-1] - target) / float32(target.size)
)

# Calculate gradients using surrogate gradient
gradient = self.__surrogate(ol.neurons, pre_spike_membrane_potentials[-1])

# Set errors for output layer
self.errors[-1] *= gradient

# ...

# Weights errors/gradients is the outer product of the membrane
# potential errors with the previous layer spikes.
# `np.outer` cannot be used as this does not support a batch
# dimension.
gemm(
    spike_values[-2],
    self.errors[-1],
    self.delta_weights[-1],
    trans_a=True,
    beta=float32(0) if self.new_ts_batch else float32(1),
)
# Bias errors/gradient is sum of the membrane potential errors/gradients
# across the batch dimension.
bias_gradient = np.sum(self.errors[-1], axis=0)
if self.new_ts_batch:
    np.copyto(self.delta_biases[-1], bias_gradient)
else:
    self.delta_biases[-1] += bias_gradient
return self.errors[-1]
```
where `self.__surrogate` calls `arctan_surrogate_gradient` which is:
```python
def arctan_surrogate_gradient(
    self,
    membrane_potential_in: NDArray[float32],
    alpha=float32(2.0),
) -> NDArray[float32]:
    """Numpy implementation of arctan surrogate gradient kernel"""
    grad = (1.0 / np.pi) * (
        1.0 / (1.0 + np.power(np.pi * membrane_potential_in * (alpha / 2.0), 2.0))
    )
    return grad.astype(float32)
```

The full test output:
```
PS C:\Users\Jonathan\Documents\spiky> poetry run pytest .\tests\test_against_snntorch_backprop.py -s
================================================================================================================== test session starts ===================================================================================================================
platform win32 -- Python 3.13.7, pytest-8.4.1, pluggy-1.6.0
rootdir: C:\Users\Jonathan\Documents\spiky
configfile: pyproject.toml
plugins: hypothesis-6.135.4
collected 1 item                                                                                                                                                                                                                                          

tests\test_against_snntorch_backprop.py snnTorch forward fc3
        (tensor([[ 0.1057, -0.1275],
        [ 0.1057, -0.1275]], grad_fn=<BackwardHookFunctionBackward>),)
        (tensor([[0., 0.],
        [0., 0.]], grad_fn=<MulBackward0>), tensor([[ 0.1057, -0.1275],
        [ 0.1057, -0.1275]], grad_fn=<SubBackward0>))
snnTorch forward lif3
        (tensor([[ 0.1057, -0.1275],
        [ 0.1057, -0.1275]], grad_fn=<BackwardHookFunctionBackward>),)
        (tensor([[0., 0.],
        [0., 0.]], grad_fn=<MulBackward0>), tensor([[ 0.1057, -0.1275],
        [ 0.1057, -0.1275]], grad_fn=<SubBackward0>))
snnTorch forward fc3
        (tensor([[ 0.1057, -0.1275],
        [ 0.1057, -0.1275]], grad_fn=<BackwardHookFunctionBackward>),)
        (tensor([[0., 0.],
        [0., 0.]], grad_fn=<MulBackward0>), tensor([[ 0.1586, -0.1912],
        [ 0.1586, -0.1912]], grad_fn=<SubBackward0>))
snnTorch forward lif3
        (tensor([[ 0.1057, -0.1275],
        [ 0.1057, -0.1275]], grad_fn=<BackwardHookFunctionBackward>),)
        (tensor([[0., 0.],
        [0., 0.]], grad_fn=<MulBackward0>), tensor([[ 0.1586, -0.1912],
        [ 0.1586, -0.1912]], grad_fn=<SubBackward0>))
snn_loss: 0.37797811627388
snnTorch backward lif3
        (tensor([[-0.0192, -0.0103],
        [-0.0295, -0.0114]]),)
        (tensor([[-0.1530, -0.1542],
        [-0.2359, -0.1705]]), None)
snnTorch backward fc3
        (tensor([[-0.0080,  0.0079, -0.0032,  0.0007],
        [-0.0111,  0.0104, -0.0061,  0.0018]]),)
        (tensor([[-0.0192, -0.0103],
        [-0.0295, -0.0114]]),)
snnTorch backward lif3
        (tensor([[-0.0213, -0.0153],
        [-0.0136, -0.0167]]),)
        (tensor([[-0.1140, -0.1421],
        [-0.0047, -0.1544]]), None)
snnTorch backward fc3
        (tensor([[-1.0019e-02,  1.0234e-02, -2.6230e-03,  2.0535e-04],
        [-8.3824e-03,  9.2277e-03,  6.5562e-05, -9.2160e-04]]),)
        (tensor([[-0.0213, -0.0153],
        [-0.0136, -0.0167]]),)
spiky backward:
        spike_values: [0. 0. 0. 0.]
        weighted_input_values: [ 0.10570377 -0.12747937  0.10570377 -0.12747937]
        membrane_potentials: [ 0.15855566 -0.19121906  0.15855566 -0.19121906]
        pre_spike_membrane_potentials: [ 0.15855566 -0.19121906  0.15855566 -0.19121906]
        gradient: [0.25503132 0.23390011 0.25503132 0.23390011]
        self.errors: [-0.07805179 -0.07215047 -0.12034266 -0.07973892]
spiky backward:
        spike_values: [0. 0. 0. 0.]
        weighted_input_values: [ 0.10570377 -0.12747937  0.10570377 -0.12747937]
        membrane_potentials: [ 0.10570377 -0.12747937  0.10570377 -0.12747937]
        pre_spike_membrane_potentials: [ 0.10570377 -0.12747937  0.10570377 -0.12747937]
        gradient: [0.28669438 0.27431265 0.28669438 0.27431265]
        self.errors: [-0.06538787 -0.07796431 -0.00269346 -0.08471261]
F

======================================================================================================================== FAILURES ======================================================================================================================== 
_________________________________________________________________________________________________________________ test_against_snntorch __________________________________________________________________________________________________________________ 

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

        def make_forward_hook(name):
            """Create a hook that captures forward pass gradients"""

            def hook(module, inner, outer):
                print(f"snnTorch forward {name}\n\t{inner}\n\t{outer}")

            return hook

        def make_backward_hook(name):
            """Create a hook that captures backward pass gradients"""

            def hook(module, grad_input, grad_output):
                print(f"snnTorch backward {name}\n\t{grad_input}\n\t{grad_output}")

            return hook

        # Register hooks
        snn_net.lif3.register_forward_hook(make_forward_hook("fc3"))
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

        print(f"snn_loss: {snn_loss}")

        # Loss - Spiky
        spiky_loss = ((spiky_trainer.network.layers[-1].spike_values - targets) ** 2).mean()

        # Check losses match
        assert np.allclose(tnp(snn_loss), spiky_loss), f"{snn_loss}\n{siky_loss}"

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

        # print(f"snn_gradients: {snn_gradients}")

        # Check gradients match
>       assert np.allclose(
            spiky_trainer.delta_biases[0], snn_gradients["fc1.bias"]  # type:ignore
        ), f"{spiky_trainer.delta_biases[0]}\n{snn_gradients["fc1.bias"]}"  # type:ignore
E       AssertionError: [-1.8060303e-04 -2.5951376e-03 -2.8685003e-04 -5.0962652e-05
E          -2.9501407e-03  4.3336369e-04]
E         [-6.5503595e-04 -6.5628707e-04  6.3494634e-05 -2.8457202e-05
E          -1.9124434e-04  2.2318613e-04]
E       assert False
E        +  where False = <function allclose at 0x00000261C3DC13F0>(array([-1.8060303e-04, -2.5951376e-03, -2.8685003e-04, -5.0962652e-05,\n       -2.9501407e-03,  4.3336369e-04], dtype=float32), array([-6.5503595e-04, -6.5628707e-04,  6.3494634e-05, -2.8457202e-05,\n       -1.9124434e-04,  2.2318613e-04], dtype=float32))
E        +    where <function allclose at 0x00000261C3DC13F0> = np.allclose

tests\test_against_snntorch_backprop.py:167: AssertionError
================================================================================================================ short test summary info =================================================================================================================
FAILED tests/test_against_snntorch_backprop.py::test_against_snntorch - AssertionError: [-1.8060303e-04 -2.5951376e-03 -2.8685003e-04 -5.0962652e-05
=================================================================================================================== 1 failed in 2.07s ==================================================================================================================== 
PS C:\Users\Jonathan\Documents\spiky>
```