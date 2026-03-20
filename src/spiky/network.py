# CPU-only NumPy version of network.py

import numpy as np
from numpy import float32
from dataclasses import dataclass
from numpy.typing import NDArray

"""
A small `pdoc` example.
"""


@dataclass
class Linear:
    """A description of linear connections between layers in a feedforward neural network"""

    outputs: int
    """The number of output values"""


@dataclass
class Pooling:
    kernel_size: tuple[int, int]
    """Kernel size `[height, width]`"""
    stride: tuple[int, int]
    """Stride of the kernel `[height, width]`"""

    def __init__(
        self,
        kernel_size: tuple[int, int] | int,
        stride: tuple[int, int] | int = 1,
    ):
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)


@dataclass
class Convolutional:
    """A description of convolutional connections between layers in a feedforward neural network"""

    out_channels: int
    """Number of output channels"""
    kernel_size: tuple[int, int]
    """Kernel size `[height, width]`"""
    stride: tuple[int, int]
    """Stride of the kernel `[height, width]`"""
    pooling: Pooling | None
    """Pooling applied after convolution"""

    def __init__(
        self,
        out_channels: int,
        kernel_size: tuple[int, int] | int,
        stride: tuple[int, int] | int = 1,
        pooling: Pooling | None = None,
    ):
        """Number of input channels is dependant on the previous layer"""
        self.out_channels = out_channels
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.pooling = pooling


@dataclass
class LIF:
    """A description of a set of leaky-integrate-and-fire neurons"""

    decay: float32 = float32(0.8) # Commonly used decay value.
    """The rate of decay of the value/voltage/membrane-potential of the neurons"""
    threshold: float32 = float32(1) # Commonly used threshold value.
    """The threshold at which the neuron spikes"""

    def spike_update(
        self, weighted_input: NDArray[float32], membrane_potential_in: NDArray[float32]
    ) -> tuple[NDArray[float32], NDArray[float32], NDArray[float32]]:
        """Updates the neurons, passing a timestep, accepting a new input, and returning a new output."""
        # Step 1: Compute membrane potential after decay and input
        membrane_potential_pre = weighted_input + self.decay * membrane_potential_in

        # Step 2: Check if membrane potential exceeds threshold
        spiked = (membrane_potential_pre > self.threshold).astype(float32)

        # Step 3: Apply reset by subtraction
        membrane_potential_out = membrane_potential_pre - spiked * self.threshold

        # Its a little burdensome, but we need to return the pre and post spike
        # membrane potentials for some of the learning algorithms (e.g. BPTT).
        return (
            spiked,
            membrane_potential_out.astype(float32),
            membrane_potential_pre.astype(float32),
        )


    def arctan_surrogate_gradient(
        self,
        membrane_potential_in: NDArray[float32],
        alpha: float32 = float32(2.0),
    ) -> NDArray[float32]:
        """
        Arctan surrogate gradient function commonly used for LIF neurons

        This function aims to provides a smooth approximation to the gradient of
        the spiking function (which would otherwise be non-differentiable (not
        usable for learning)). The `alpha` parameter controls the steepness of
        the surrogate gradient, with higher values it is a more accurate
        approximation but can create difficulty learning (e.g. vanishing
        gradients etc.).
        """
        grad = (1.0 / np.pi) * (
            1.0 / (1.0 + np.power(np.pi * membrane_potential_in * (alpha / 2.0), 2.0))
        )
        return grad.astype(float32)


@dataclass
class Placeholder:
    pass


# Presumably this could be re-written to far fewer lines using `np.einsum` but I
# find this notation to be difficult to read and the performance gain is lightly
# very marginal.
def conv2d_numpy(
    input_data: NDArray[float32],
    kernel: NDArray[float32],
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int] = (0, 0),
):
    """
    Numpy implementation of 2D convolution
    input_data: (batch, height, width, in_channels) - NHWC format
    kernel: (out_channels, in_channels, kernel_h, kernel_w) - OIHW format
    """
    batch_size, in_h, in_w, in_channels = input_data.shape
    out_channels, kernel_in_channels, kernel_h, kernel_w = kernel.shape

    assert (
        in_channels == kernel_in_channels
    ), f"Input channels mismatch: {in_channels} vs {kernel_in_channels}"

    stride_h, stride_w = stride
    pad_h, pad_w = padding

    # Calculate output dimensions
    out_h = (in_h + 2 * pad_h - kernel_h) // stride_h + 1
    out_w = (in_w + 2 * pad_w - kernel_w) // stride_w + 1

    # Initialize output
    output = np.zeros((batch_size, out_h, out_w, out_channels), dtype=input_data.dtype)

    # Apply padding if needed
    if pad_h > 0 or pad_w > 0:
        padded_input = np.pad(
            input_data,
            ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
            mode="constant",
        )
    else:
        padded_input = input_data

    # Perform convolution
    for b in range(batch_size):
        for oc in range(out_channels):
            for oh in range(out_h):
                for ow in range(out_w):
                    h_start = oh * stride_h
                    w_start = ow * stride_w
                    h_end = h_start + kernel_h
                    w_end = w_start + kernel_w

                    # Extract input patch
                    input_patch = padded_input[b, h_start:h_end, w_start:w_end, :]

                    # Compute convolution for this output position
                    output[b, oh, ow, oc] = np.sum(input_patch * kernel[oc, :, :, :])

    return output


@dataclass
class FeedForwardLayer:
    """A layer within a feedforward neuron network"""

    neurons: LIF | Placeholder
    """The type of the neurons in this layer"""
    neuron_values: NDArray[float32]
    """The values of the neurons in this layer"""
    synapses: Linear | Convolutional
    """The type of the synapses connecting into this layer"""
    synapse_weights: NDArray[float32]  # The weights
    """The synapse weights of neurons in this layer."""
    synapse_biases: NDArray[float32]  # The biases
    """The synapse biases of neurons in this layer."""
    spike_values: NDArray[float32]
    """The spike values of neurons in this layer"""
    weighted_input_values: NDArray[float32]
    """The weighted input values into neurons"""
    pre_spike_membrane_potentials: NDArray[float32]
    """The pre-spike membrane potentials of neurons"""

    def __init__(
        self,
        incoming_neurons: tuple[int, ...],
        synapses: Linear | Convolutional,
        neurons: LIF | Placeholder,
    ):
        """
        Creates a feed forward network layer.

        Args:
            incoming_neurons: `[samples x feature dimensions..]`
        """
        self.neurons = neurons
        self.synapses = synapses

        # `synapses.outputs` here is the number of neurons in this layer.
        # `samples` is the batch size.
        # `features` is the number of incoming neurons.
        if isinstance(synapses, Linear):
            match incoming_neurons:
                case (samples, features):
                    self.neuron_values = np.zeros((samples, synapses.outputs), float32)
                    self.synapse_weights = np.zeros(
                        (features, synapses.outputs), float32
                    )
                    self.synapse_biases = np.zeros((synapses.outputs,), float32)
                    self.spike_values = np.zeros((samples, synapses.outputs), float32)
                    self.weighted_input_values = np.zeros(
                        (samples, synapses.outputs), float32
                    )
                    self.pre_spike_membrane_potentials = np.zeros(
                        (samples, synapses.outputs), float32
                    )
                    pass
                case _:
                    raise Exception("todo")
        else:
            assert isinstance(synapses, Convolutional)
            # Unpack kernel and stride dimensions
            kernel_h, kernel_w = synapses.kernel_size
            stride_h, stride_w = synapses.stride

            match incoming_neurons:
                case (samples, height, width, channels):
                    # Pooling and the rest of convolution is not yet implemented
                    # pool_kernel = (1,1)
                    # pool_stride = (1,1)
                    # if synapses.pooling is not None:
                    #     pool_kernel = synapses.pooling.kernel_size
                    #     pool_stride = synapses.pooling.stride

                    # Calculate output dimensions
                    out_h = (height - kernel_h) // stride_h + 1
                    out_w = (width - kernel_w) // stride_w + 1

                    # Initialize arrays
                    self.neuron_values = np.zeros(
                        (samples, out_h, out_w, synapses.out_channels), float32
                    )
                    self.synapse_weights = np.zeros(
                        (synapses.out_channels, channels, kernel_h, kernel_w), float32
                    )
                    self.spike_values = np.zeros(
                        (samples, out_h, out_w, synapses.out_channels), float32
                    )
                    self.weighted_input_values = np.zeros(
                        (samples, out_h, out_w, synapses.out_channels), float32
                    )
                    self.pre_spike_membrane_potentials = np.zeros(
                        (samples, out_h, out_w, synapses.out_channels), float32
                    )

                    raise Exception("todo")
                case _:
                    raise Exception("Unsupported input shape for convolutional layer")

    def forward(self, inputs: NDArray[float32]):
        """Passes an input into this layer"""
        # Propagate through synapses.
        if isinstance(self.synapses, Linear):
            match len(inputs.shape):
                case 2:  # [samples, features]
                    np.matmul(
                        inputs, self.synapse_weights, out=self.weighted_input_values
                    )
                    # Add biases (broadcasting across batch dimension)
                    self.weighted_input_values += self.synapse_biases
                case _:
                    raise Exception(f"todo {inputs.shape}")
        else:
            assert isinstance(self.synapses, Convolutional)
            self.weighted_input_values = conv2d_numpy(
                inputs, self.synapse_weights, stride=self.synapses.stride
            )
            # Add biases (broadcasting across batch, height, width dimensions)
            self.weighted_input_values += self.synapse_biases

        # Propagate into neurons.
        if isinstance(self.neurons, LIF):
            spikes, membrane, pre_spike = self.neurons.spike_update(
                self.weighted_input_values, self.neuron_values
            )
            self.spike_values = spikes
            self.neuron_values = membrane
            self.pre_spike_membrane_potentials = pre_spike
        else:
            raise Exception("todo")


@dataclass
class FeedForwardNetwork:
    """A feed forward neural network"""

    inputs: int
    """Number of neurons in the first layer"""
    batch_size: int
    """The size of batches"""
    layers: list[FeedForwardLayer]
    """Layers in the network"""
    spikes: NDArray[float32]
    """The spikes from the final layer at the final timestep"""
    trainer: BackpropagationThroughTime | None
    """The trainer for this network if currently training"""

    def __init__(
        self,
        batch_size: int,
        inputs: int,
        layers: list[tuple[Linear | Convolutional, LIF]],
    ):
        self.inputs = inputs
        self.batch_size = batch_size
        incoming_neurons = (batch_size, inputs)
        self.layers = []
        self.trainer = None

        for synapses, neurons in layers:
            new_layer = FeedForwardLayer(incoming_neurons, synapses, neurons)
            incoming_neurons = new_layer.neuron_values.shape
            self.layers.append(new_layer)

    def forward(self, inputs: list[NDArray[float32]], training: str | None = None):
        match training:
            case "bptt":
                self.trainer = BackpropagationThroughTime(self, inputs)
            case None:
                self.__forward(inputs)
            case _:
                raise Exception(f"Unsupported training mode: {training}")

    def __forward(self, inputs: list[NDArray[float32]]):
        assert all([x.shape == (self.batch_size, self.inputs) for x in inputs])
        # Iterate over timesteps.
        spikes = None
        for ts_input in inputs:
            spikes = ts_input
            # Iterate through net layers.
            for layer in self.layers:
                layer.forward(spikes)
                spikes = layer.spike_values
        # Return the spikes from the final layer at the final timestep.
        if spikes is None:
            raise Exception("No inputs provided")
        self.spikes = spikes

    def backward(self, targets: list[NDArray[float32]]):
        if self.trainer is None:
            raise Exception("Cannot call backward without forward pass with training mode")
        self.trainer.backward(targets)

    def update(self, learning_rate: float32):
        """Updates the weights and biases in the network"""

        match self.trainer:
            case None:
                raise Exception("Cannot call update without forward pass with training mode")
            case BackpropagationThroughTime():
                self.__update_bptt(learning_rate)
            case _:
                raise Exception(f"Unsupported trainer type: {type(self.trainer)}")

    def __update_bptt(self, learning_rate: float32):
        """Updates the weights and biases in the network"""
        assert isinstance(self.trainer, BackpropagationThroughTime)
        
        # Update network.
        for layer, delta_weights, delta_biases in zip(
            self.layers, self.trainer.delta_weights, self.trainer.delta_biases
        ):
            layer.synapse_weights -= (
                learning_rate * delta_weights / float32(len(self.trainer.inputs))
            )
            layer.synapse_biases -= (
                learning_rate * delta_biases / float32(len(self.trainer.inputs))
            )


# We currently don't support "truncated BPPT" which would use batches through time to train
# for deep timesteps (in the same way using batches through samples saves memory usage but
# reduces accuracy).

# With some changes backprop could be parallelized since each call only needs 1
# timestep of foreprop, potentially when each foreprop timestep completes it could dispatch a
# backprop call for that step and then it could resolve after all backprop steps
# have resolved. This approach would increase the flat amount of compute and
# memory required but given the high parallelism could offer massive gains for
# large models by using multiple GPUs potentially across multiple systems.


@dataclass
class BackpropagationThroughTime:
    """Underlying feed forward network"""
    weighted_input_values: list[list[NDArray[float32]]]
    """Weighted inputs for each timestep for each layer"""
    membrane_potentials: list[list[NDArray[float32]]]
    """Membrane potentials for each timestep for each layer"""
    spike_values: list[list[NDArray[float32]]]
    """The spikes for each timestep for each layer"""
    delta_weights: list[NDArray[float32]]
    """Change in weights for each layer"""
    delta_biases: list[NDArray[float32]]
    """Change in biases for each layer"""
    errors: list[NDArray[float32]]
    """Errors in spikes for each layer"""
    inputs: list[NDArray[float32]]
    """Input spikes for each timestep"""

    def __init__(self, network: FeedForwardNetwork, inputs: list[NDArray[float32]]):
        # Some pre-work.
        timesteps = len(inputs)

        # Sets new values.
        self.network = network
        self.errors = [
            np.zeros(layer.neuron_values.shape, dtype=float32)
            for layer in network.layers
        ]
        self.delta_weights = [
            np.zeros(layer.synapse_weights.shape, dtype=float32)
            for layer in network.layers
        ]
        self.delta_biases = [
            np.zeros(layer.synapse_biases.shape, dtype=float32)
            for layer in network.layers
        ]
        self.inputs = inputs
        # Sets up zeroed arrays to store the forward pass values for each layer
        # for each new timestep
        self.membrane_potentials = [
            [
                np.zeros(layer.neuron_values.shape, dtype=float32)
                for layer in self.network.layers
            ]
            for _ in range(timesteps)
        ]
        self.weighted_input_values = [
            [
                np.zeros(layer.weighted_input_values.shape, dtype=float32)
                for layer in self.network.layers
            ]
            for _ in range(timesteps)
        ]
        self.spike_values = [
            [
                np.zeros(layer.spike_values.shape, dtype=float32)
                for layer in self.network.layers
            ]
            for _ in range(timesteps)
        ]
        self.pre_spike_membrane_potentials = [
            [
                np.zeros(layer.pre_spike_membrane_potentials.shape, dtype=float32)
                for layer in self.network.layers
            ]
            for _ in range(timesteps)
        ]

        # Execute forward propagation
        self.__forward()

    def __forward(self):
        """Performs and records the forward propagation of `inputs` through the network"""

        # Iterate through each timestep
        for (
            input_data,
            timestep_weighted_input_values,
            timestep_spike_values,
            timestep_membrane_potentials,
            timestep_pre_spike_membrane_potentials,
        ) in zip(
            self.inputs,
            self.weighted_input_values,
            self.spike_values,
            self.membrane_potentials,
            self.pre_spike_membrane_potentials,
        ):
            assert input_data.shape == (self.network.batch_size, self.network.inputs)
            spikes = input_data
            # Iterate through each layer
            for (
                layer,
                layer_timestep_weighted_input_values,
                layer_timestep_spike_values,
                layer_timestep_membrane_potentials,
                layer_timestep_pre_spike_membrane_potentials,
            ) in zip(
                self.network.layers,
                timestep_weighted_input_values,
                timestep_spike_values,
                timestep_membrane_potentials,
                timestep_pre_spike_membrane_potentials,
            ):
                layer.forward(spikes)

                # Store weighted inputs and spikes for this layer at this timestep for backpropagation
                np.copyto(
                    layer_timestep_weighted_input_values, layer.weighted_input_values
                )
                np.copyto(layer_timestep_spike_values, layer.spike_values)
                np.copyto(layer_timestep_membrane_potentials, layer.neuron_values)
                np.copyto(
                    layer_timestep_pre_spike_membrane_potentials,
                    layer.pre_spike_membrane_potentials,
                )
                spikes = layer.spike_values

    def __surrogate(
        self,
        neurons: LIF | Placeholder,
        membrane_potential: NDArray[float32],
    ) -> NDArray[float32]:
        """Calculate the (surrogate) gradient for a given type of neurons with the given neuron values"""
        if isinstance(neurons, LIF):
            return neurons.arctan_surrogate_gradient(membrane_potential)
        else:
            raise Exception("todo")

    def __backward_output(
        self,
        target: NDArray[float32],
        spike_values: list[NDArray[float32]],
        weighted_input_values: list[NDArray[float32]],
        membrane_potentials: list[NDArray[float32]],
        pre_spike_membrane_potentials: list[NDArray[float32]],
    ) -> NDArray[float32]:
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

        print("spiky backward:")
        print(f"\tspike_values: {spike_values[-1].flatten()}")
        print(f"\tweighted_input_values: {weighted_input_values[-1].flatten()}")
        print(f"\tmembrane_potentials: {membrane_potentials[-1].flatten()}")
        print(
            f"\tpre_spike_membrane_potentials: {pre_spike_membrane_potentials[-1].flatten()}"
        )
        print(f"\tgradient: {gradient.flatten()}")
        print(f"\tself.errors: {self.errors[-1].flatten()}")

        if isinstance(ol.synapses, Linear):
            # Weights errors/gradients is the outer product of the membrane
            # potential errors with the previous layer spikes.
            # `np.outer` cannot be used as this does not support a batch
            # dimension.
            gemm(
                spike_values[-2],
                self.errors[-1],
                self.delta_weights[-1],
                trans_a=True,
                beta=float32(1),
            )
            # Bias errors/gradient is sum of the membrane potential errors/gradients
            # across the batch dimension.
            bias_gradient = np.sum(self.errors[-1], axis=0)
            self.delta_biases[-1] += bias_gradient
            return self.errors[-1]
        else:
            raise Exception("todo")

    def __backward_hidden(
        self,
        spike_values: list[NDArray[float32]],
        weighted_input_values: list[NDArray[float32]],
        # The errors or gradient w.r.t to the succeeding layers membrane potentials.
        delta_next: NDArray[float32],
        li: int,
        pre_spike_membrane_potentials: list[NDArray[float32]],
    ) -> NDArray[float32]:
        layer = self.network.layers[li]
        after_layer = self.network.layers[li + 1]

        # Calculate the error for this layer
        if isinstance(after_layer.synapses, Linear):
            self.errors[li] = np.matmul(delta_next, after_layer.synapse_weights.T)
        else:
            raise Exception("todo")

        # Calculate surrogate gradient
        gradient = self.__surrogate(
            layer.neurons,
            pre_spike_membrane_potentials[li],
        )

        self.errors[li] *= gradient

        # Update weights for current layer
        if isinstance(layer.synapses, Linear):
            gemm(
                spike_values[li - 1],
                self.errors[li],
                self.delta_weights[li],
                trans_a=True,
                beta=float32(0) if self.new_ts_batch else float32(1),
            )
            bias_gradient = np.sum(self.errors[li], axis=0)
            self.delta_biases[li] += bias_gradient
        else:
            raise Exception("todo")

        return self.errors[li]

    def __backward_input(
        self,
        weighted_input_values: list[NDArray[float32]],
        delta_next: NDArray[float32],
        input_values: NDArray[float32],
        pre_spike_membrane_potentials: list[NDArray[float32]],
    ) -> None:
        input_layer = self.network.layers[0]
        first_hidden_layer = self.network.layers[1]

        if isinstance(first_hidden_layer.synapses, Linear):
            self.errors[0] = np.matmul(delta_next, first_hidden_layer.synapse_weights.T)
        else:
            raise Exception("todo")

        gradient = self.__surrogate(
            input_layer.neurons, pre_spike_membrane_potentials[0]
        )

        self.errors[0] *= gradient

        if isinstance(input_layer.synapses, Linear):
            gemm(
                input_values,
                self.errors[0],
                self.delta_weights[0],
                trans_a=True,
                beta=float32(1),
            )
            # Bias gradient is sum of errors across batch dimension
            bias_gradient = np.sum(self.errors[0], axis=0)
            self.delta_biases[0] += bias_gradient
        else:
            raise Exception("todo")

    def backward(self, targets: list[NDArray[float32]]):
        """
        Backpropagates `targets`.
        """

        # Check the number of targets given matchesa the number of inputs.
        assert len(self.inputs) == len(targets)

        for target in reversed(targets):
            # If its the first `backward` call since `update` set to `0.0` so
            # we re-zero `delta_weights` else set to `1.0` so we add to
            # `delta_weights`. This saves having to make a separate call to
            # re-zero `delta_weights`.
            number_of_layers = len(self.network.layers)

            # Inputs values, weighted input values and spikes for this time step.
            input_values = self.inputs.pop()
            timestep = len(self.inputs)
            weighted_input_values = self.weighted_input_values[timestep]
            spike_values = self.spike_values[timestep]
            membrane_potentials = self.membrane_potentials[timestep]
            pre_spike_membrane_potentials = self.pre_spike_membrane_potentials[timestep]

            # Output layer
            delta_next = self.__backward_output(
                target,
                spike_values,
                weighted_input_values,
                membrane_potentials,
                pre_spike_membrane_potentials,
            )

            # Hidden layers
            for li in range(number_of_layers - 2, 0, -1):
                delta_next = self.__backward_hidden(
                    spike_values,
                    weighted_input_values,
                    delta_next,
                    li,
                    pre_spike_membrane_potentials,
                )

            # Input layer
            self.__backward_input(
                weighted_input_values,
                delta_next,
                input_values,
                pre_spike_membrane_potentials,
            )


def gemm(
    a: NDArray[float32],
    b: NDArray[float32],
    c: NDArray[float32],
    trans_a: bool = False,
    trans_b: bool = False,
    beta: float32 = float32(0.0),
) -> None:
    """
    Computes C = A @ B + beta * C using numpy matrix multiplication.

    Args:
        a: Input matrix A
        b: Input matrix B
        c: Pre-allocated output matrix (overwritten in-place)
        trans_a: Whether to transpose A
        trans_b: Whether to transpose B
        beta: Scaling factor for existing values in C
    """
    # Apply transpositions
    a_op = a.T if trans_a else a
    b_op = b.T if trans_b else b

    # Compute matrix multiplication
    result = np.matmul(a_op, b_op)

    # Apply beta scaling and store result
    if beta == 0.0:
        np.copyto(c, result)
    else:
        c[:] = result + beta * c
