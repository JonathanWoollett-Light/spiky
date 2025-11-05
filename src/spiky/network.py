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
    """A description of a leaky-integrate-and-fire neuron"""

    decay: float32
    """The rate of decay of the value/voltage of the neurons"""
    threshold: float32 = float32(1)
    """The threshold at which the neuron spikes"""

    def spike_update(
        self, weighted_input: NDArray[float32], membrane_potential_in: NDArray[float32]
    ) -> tuple[NDArray[float32], NDArray[float32]]:
        """Numpy implementation of spike update kernel"""
        # Step 1: Compute membrane potential after decay and input
        membrane_potential_pre = weighted_input + self.decay * membrane_potential_in

        # Step 2: Check if membrane potential exceeds threshold
        spiked = (membrane_potential_pre > self.threshold).astype(float32)

        # Step 3: Apply reset by subtraction
        membrane_potential_out = membrane_potential_pre - spiked * self.threshold

        return spiked, membrane_potential_out.astype(float32)

    def arctan_surrogate_gradient(
        self, membrane_potential_in: NDArray[float32]
    ) -> NDArray[float32]:
        """Numpy implementation of arctan surrogate gradient kernel"""
        x = np.pi * membrane_potential_in
        return (1.0 / (1.0 + (x * x))).astype(float32)


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
                    raise Exception("todo")
        else:
            assert isinstance(self.synapses, Convolutional)
            self.weighted_input_values = conv2d_numpy(
                inputs, self.synapse_weights, stride=self.synapses.stride
            )
            # Add biases (broadcasting across batch, height, width dimensions)
            self.weighted_input_values += self.synapse_biases

        # Propagate into neurons.
        if isinstance(self.neurons, LIF):
            spikes, membrane = self.neurons.spike_update(
                self.weighted_input_values, self.neuron_values
            )
            self.spike_values = spikes
            self.neuron_values = membrane
        else:
            raise Exception("todo")


@dataclass
class FeedForwardNetwork:
    """A feed forward neural network"""

    inputs: int
    """Number of neurons in the first layer"""
    layers: list[FeedForwardLayer]
    """Layers in the network"""

    def __init__(
        self,
        batch_size: int,
        inputs: int,
        layers: list[tuple[Linear | Convolutional, LIF]],
    ):
        self.inputs = inputs
        incoming_neurons = (batch_size, inputs)
        self.layers = []

        for synapses, neurons in layers:
            new_layer = FeedForwardLayer(incoming_neurons, synapses, neurons)
            incoming_neurons = new_layer.neuron_values.shape
            self.layers.append(new_layer)

    def forward(self, inputs: NDArray[float32]):
        spikes = inputs
        for layer in self.layers:
            layer.forward(spikes)
            spikes = layer.spike_values
        return spikes


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
    network: FeedForwardNetwork
    """Underlying feed forward network"""
    weighted_input_values: list[list[NDArray[float32]]]
    """Weighted inputs for each timestep for each layer"""
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
    backed: bool
    """If `backward` has been called since last `forward`"""
    number_backed: int
    """Number of timesteps backpropagated through"""

    def __init__(self, network: FeedForwardNetwork):
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
        self.backed = False
        self.weighted_input_values = []
        self.spike_values = []
        self.inputs = []
        self.number_backed = 0

    def forward(self, inputs: list[NDArray[float32]]):
        """Performs and records the forward propagation of `inputs` through the network"""
        self.inputs += inputs
        self.__update_cache()

        for input_data, timestep_weighted_input_values, timestep_spike_values in zip(
            inputs, self.weighted_input_values, self.spike_values
        ):
            spikes = input_data
            for (
                layer,
                layer_timestep_weighted_input_values,
                layer_timestep_spike_values,
            ) in zip(
                self.network.layers,
                timestep_weighted_input_values,
                timestep_spike_values,
            ):
                layer.forward(spikes)
                np.copyto(
                    layer_timestep_weighted_input_values, layer.weighted_input_values
                )
                np.copyto(layer_timestep_spike_values, layer.spike_values)
                spikes = layer.spike_values

    def __update_cache(self):
        """Since a user can change how many forward timesteps they want, we may need to extend the number of arrays we keep cached"""
        timesteps = len(self.inputs)

        while len(self.weighted_input_values) < timesteps:
            self.weighted_input_values.append(
                [
                    np.zeros(layer.weighted_input_values.shape, dtype=float32)
                    for layer in self.network.layers
                ]
            )

        while len(self.spike_values) < timesteps:
            self.spike_values.append(
                [
                    np.zeros(layer.spike_values.shape, dtype=float32)
                    for layer in self.network.layers
                ]
            )

    def __surrogate(
        self, neurons: LIF | Placeholder, membrane_potential: NDArray[float32]
    ) -> NDArray[float32]:
        """Calculate the (surrogate) gradient for a given type of neurons with the given neuron values"""
        if isinstance(neurons, LIF):
            return neurons.arctan_surrogate_gradient(membrane_potential)
        else:
            raise Exception("todo")

    def __conv2d_backward_weight(
        self,
        prev_spikes: NDArray[float32],
        errors: NDArray[float32],
        delta_weights: NDArray[float32],
        stride: tuple[int, int],
        beta: float32,
    ) -> None:
        """Numpy-based weight gradient calculation for convolution"""
        batch_size, out_h, out_w, out_channels = errors.shape
        batch_size_prev, in_h, in_w, in_channels = prev_spikes.shape
        _out_channels_dw, in_channels_dw, kernel_h, kernel_w = delta_weights.shape

        assert batch_size == batch_size_prev
        assert in_channels == in_channels_dw

        stride_h, stride_w = stride

        # Initialize or accumulate based on beta
        if beta == 0:
            delta_weights.fill(0)

        # Compute weight gradients
        for oc in range(out_channels):
            for ic in range(in_channels):
                for kh in range(kernel_h):
                    for kw in range(kernel_w):
                        grad_sum = 0.0
                        for b in range(batch_size):
                            for oh in range(out_h):
                                for ow in range(out_w):
                                    ih = oh * stride_h + kh
                                    iw = ow * stride_w + kw
                                    if ih < in_h and iw < in_w:
                                        grad_sum += (
                                            prev_spikes[b, ih, iw, ic]
                                            * errors[b, oh, ow, oc]
                                        )
                        delta_weights[oc, ic, kh, kw] += grad_sum

    def __conv2d_backward_input(
        self,
        weight: NDArray[float32],
        errors: NDArray[float32],
        input_shape: tuple[int, ...],
        stride: tuple[int, int],
    ) -> NDArray[float32]:
        """Numpy-based input gradient calculation for convolution"""

        # The tuple length cannot be specified in the argument type since
        # `NDArray[float32]` can be an arbitrary number of dimensions.
        assert len(input_shape) == 4

        batch_size, in_h, in_w, in_channels = input_shape
        batch_size_err, out_h, out_w, out_channels = errors.shape
        out_channels_w, _in_channels_w, kernel_h, kernel_w = weight.shape

        assert batch_size == batch_size_err
        assert out_channels == out_channels_w

        stride_h, stride_w = stride

        grad_input = np.zeros(input_shape, dtype=float32)

        # Compute input gradients
        for b in range(batch_size):
            for ic in range(in_channels):
                for ih in range(in_h):
                    for iw in range(in_w):
                        grad_sum = 0.0
                        for oc in range(out_channels):
                            for kh in range(kernel_h):
                                for kw in range(kernel_w):
                                    oh = (ih - kh) // stride_h
                                    ow = (iw - kw) // stride_w
                                    if (
                                        oh >= 0
                                        and oh < out_h
                                        and ow >= 0
                                        and ow < out_w
                                        and (ih - kh) % stride_h == 0
                                        and (iw - kw) % stride_w == 0
                                    ):
                                        grad_sum += (
                                            weight[oc, ic, kh, kw]
                                            * errors[b, oh, ow, oc]
                                        )
                        grad_input[b, ih, iw, ic] = grad_sum

        return grad_input

    def __backward_output(
        self,
        target: NDArray[float32],
        spike_values: list[NDArray[float32]],
        weighted_input_values: list[NDArray[float32]],
        beta: float32,
    ) -> NDArray[float32]:
        number_of_layers = len(self.network.layers)
        # output_layer_index
        oli = number_of_layers - 1
        ol = self.network.layers[oli]

        # Calculate output gradients using MSE loss
        output_gradients = self.__surrogate(ol.neurons, weighted_input_values[oli])
        self.errors[oli] = float32(2) * (spike_values[oli] - target) * output_gradients

        if isinstance(ol.synapses, Linear):
            # Linear layer weight update
            gemm(
                spike_values[oli - 1],
                self.errors[oli],
                self.delta_weights[oli],
                trans_a=True,
                beta=beta,
            )
            # Bias gradient is sum of errors across batch dimension
            bias_gradient = np.sum(self.errors[oli], axis=0)
            if beta == 0.0:
                np.copyto(self.delta_biases[oli], bias_gradient)
            else:
                self.delta_biases[oli] += bias_gradient
            return self.errors[oli]
        else:
            assert isinstance(ol.synapses, Convolutional)
            # Convolutional layer weight update
            self.__conv2d_backward_weight(
                prev_spikes=spike_values[oli - 1],
                errors=self.errors[oli],
                delta_weights=self.delta_weights[oli],
                stride=ol.synapses.stride,
                beta=beta,
            )
            # Bias gradient is sum of errors across batch, height, and width dimensions
            bias_gradient = np.sum(self.errors[oli], axis=(0, 1, 2))
            if beta == 0.0:
                np.copyto(self.delta_biases[oli], bias_gradient)
            else:
                self.delta_biases[oli] += bias_gradient

            # Calculate input gradients for previous layer
            return self.__conv2d_backward_input(
                weight=ol.synapse_weights,
                errors=self.errors[oli],
                input_shape=self.network.layers[oli - 1].neuron_values.shape,
                stride=ol.synapses.stride,
            )

    def __backward_hidden(
        self,
        spike_values: list[NDArray[float32]],
        weighted_input_values: list[NDArray[float32]],
        beta: float32,
        delta_next: NDArray[float32],
        li: int,
    ) -> NDArray[float32]:
        layer = self.network.layers[li]
        after_layer = self.network.layers[li + 1]

        gradient = self.__surrogate(layer.neurons, weighted_input_values[li])

        # Calculate the error for this layer
        if isinstance(after_layer.synapses, Linear):
            self.errors[li] = np.matmul(delta_next, after_layer.synapse_weights.T)
        else:
            assert isinstance(after_layer.synapses, Convolutional)
            self.errors[li] = self.__conv2d_backward_input(
                weight=after_layer.synapse_weights,
                errors=delta_next,
                input_shape=layer.neuron_values.shape,
                stride=after_layer.synapses.stride,
            )

        self.errors[li] *= gradient

        # Update weights for current layer
        if isinstance(layer.synapses, Linear):
            gemm(
                spike_values[li - 1],
                self.errors[li],
                self.delta_weights[li],
                trans_a=True,
                beta=beta,
            )
            # Bias gradient is sum of errors across batch dimension
            bias_gradient = np.sum(self.errors[li], axis=0)
            if beta == 0.0:
                np.copyto(self.delta_biases[li], bias_gradient)
            else:
                self.delta_biases[li] += bias_gradient
        else:
            assert isinstance(layer.synapses, Convolutional)
            self.__conv2d_backward_weight(
                prev_spikes=spike_values[li - 1],
                errors=self.errors[li],
                delta_weights=self.delta_weights[li],
                stride=layer.synapses.stride,
                beta=beta,
            )
            # Bias gradient is sum of errors across batch, height, and width dimensions
            bias_gradient = np.sum(self.errors[li], axis=(0, 1, 2))
            if beta == 0.0:
                np.copyto(self.delta_biases[li], bias_gradient)
            else:
                self.delta_biases[li] += bias_gradient

        return self.errors[li]

    def __backward_input(
        self,
        weighted_input_values: list[NDArray[float32]],
        beta: float32,
        delta_next: NDArray[float32],
        input_values: NDArray[float32],
    ) -> None:
        input_layer = self.network.layers[0]
        first_hidden_layer = self.network.layers[1]

        gradient = self.__surrogate(input_layer.neurons, weighted_input_values[0])

        if isinstance(first_hidden_layer.synapses, Linear):
            self.errors[0] = np.matmul(delta_next, first_hidden_layer.synapse_weights.T)
        else:
            assert isinstance(first_hidden_layer.synapses, Convolutional)
            self.errors[0] = self.__conv2d_backward_input(
                weight=first_hidden_layer.synapse_weights,
                errors=delta_next,
                input_shape=input_layer.neuron_values.shape,
                stride=first_hidden_layer.synapses.stride,
            )

        self.errors[0] *= gradient

        if isinstance(input_layer.synapses, Linear):
            gemm(
                input_values,
                self.errors[0],
                self.delta_weights[0],
                trans_a=True,
                beta=beta,
            )
            # Bias gradient is sum of errors across batch dimension
            bias_gradient = np.sum(self.errors[0], axis=0)
            if beta == 0.0:
                np.copyto(self.delta_biases[0], bias_gradient)
            else:
                self.delta_biases[0] += bias_gradient
        else:
            assert isinstance(input_layer.synapses, Convolutional)
            self.__conv2d_backward_weight(
                prev_spikes=input_values,
                errors=self.errors[0],
                delta_weights=self.delta_weights[0],
                stride=input_layer.synapses.stride,
                beta=beta,
            )
            # Bias gradient is sum of errors across batch, height, and width dimensions
            bias_gradient = np.sum(self.errors[0], axis=(0, 1, 2))
            if beta == 0.0:
                np.copyto(self.delta_biases[0], bias_gradient)
            else:
                self.delta_biases[0] += bias_gradient

    def backward(self, targets: list[NDArray[float32]]):
        """
        Backpropagates `targets`.

        - `len(target) == timesteps`.
        - `targets[:].shape == [samples x feature dimensions...]`.

        A timestep is considered *unprocessed* when it has been propagated forward.
        A timestep is considered *processed* when it has been propagated backwards.

        When calling `backward()` it must be called with a number of `targets` less than or equal to the number of unprocessed timestep.

        When calling `update()` unprocessed timestep are discarded.

        Valid code might look like:

        ```python
        # len(inputs) == len(targets) == timesteps
        # inputs[:].shape == targets[:].shape == [samples x feature dimensions...]
        train = BackpropagationThroughTime(/* .. */)
        train.forward(inputs[0:4])
        train.backward(targets[3:4])
        train.backward(targets[0:3])
        train.forward(inputs[4:8])
        train.backward(targets[4:8])
        train.update()
        train.forward(inputs[0:7])
        train.forward(inputs[7])
        train.backward(targets[0:8])
        train.update()
        train.forward(inputs[0:8])
        train.backward(targets[4:8])
        train.update()
        ```
        """
        self.number_backed += len(targets)
        assert self.number_backed <= len(self.inputs)
        assert self.number_backed <= len(self.weighted_input_values)
        assert self.number_backed <= len(self.spike_values)

        for target in reversed(targets):
            # If its the first `backward` call since `update` set to `0.0` so we re-zero `delta_weights` else set to `1.0` so we add to `delta_weights`. This saves having to make a seperate call to re-zero `delta_weights`.
            beta = float32(1.0) if self.backed else float32(0.0)
            number_of_layers = len(self.network.layers)

            # Inputs values, weighted input values and spikes for this time step.
            input_values = self.inputs.pop()
            timestep = len(self.inputs)
            weighted_input_values = self.weighted_input_values[timestep]
            spike_values = self.spike_values[timestep]

            # Output layer
            delta_next = self.__backward_output(
                target, spike_values, weighted_input_values, beta
            )

            # Hidden layers
            for li in range(number_of_layers - 2, 0, -1):
                delta_next = self.__backward_hidden(
                    spike_values,
                    weighted_input_values,
                    beta,
                    delta_next,
                    li,
                )

            # Input layer
            self.__backward_input(
                weighted_input_values,
                beta,
                delta_next,
                input_values,
            )

            self.backed = True

    def update(self, learning_rate: float32):
        """Updates the weights and biases in the network"""
        for layer, delta_weights, delta_biases in zip(
            self.network.layers, self.delta_weights, self.delta_biases
        ):
            layer.synapse_weights -= (
                learning_rate * delta_weights / float32(self.number_backed)
            )
            layer.synapse_biases -= (
                learning_rate * delta_biases / float32(self.number_backed)
            )

        self.backed = False
        self.inputs = []
        self.number_backed = 0


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
