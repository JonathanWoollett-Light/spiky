import cupy as cp
from numpy import float32

"""
A small `pdoc` example.
"""


class Linear:
    """A description of linear connections between layers in a feedforward neural network"""

    outputs: int
    """The number of output values"""

    def __init__(self, outputs: int):
        """The number of input values is dependant on the previous layer"""
        self.outputs = outputs


class Convolutional:
    """A description of convolutional connections between layers in a feedforward neural network"""

    out_channels: int
    """The number of output channels"""
    kernel_size: tuple[int, int]
    """The kernel size `[height, width]`"""
    stride: tuple[int, int]
    """The stride of the kernel `[height, width]`"""

    def __init__(
        self,
        out_channels: int,
        kernel_size: tuple[int, int] | int,
        stride: tuple[int, int] | int = 1,
    ):
        """The number of input channels is dependant on the previous layer"""
        self.out_channels = out_channels
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)


class LIF:
    """A description of a leaky-integrate-and-fire neuron"""

    decay: float32
    """The rate of decay of the value/voltage of the neurons"""
    threshold: float32
    """The threshold at which the neuron spikes"""

    spike_update_kernel = cp.ElementwiseKernel(  # type: ignore
        in_params="float32 weighted_input, float32 decay, float32 threshold, float32 membrane_potential_in",
        out_params="float32 spiked_output, float32 membrane_potential_out",
        operation="""
            float spiked = (membrane_potential_in > threshold) ? 1.0f : 0.0f;
            spiked_output = spiked;
            membrane_potential_out = weighted_input + decay * membrane_potential_in - decay * spiked * threshold;
        """,
        name="spike_update_kernel",
    )

    def __init__(self, decay: float32, threshold: float32):
        self.decay = decay
        self.threshold = threshold


class Placeholder:
    pass


class FeedForwardLayer:
    """A layer within a feedforward neuron network"""

    neurons: LIF | Placeholder
    """The type of the neurons in this layer"""
    neuron_values: cp.ndarray  # type: ignore
    """The values of the neurons in this layer"""
    synapses: Linear | Convolutional
    """The type of the synapses connecting into this layer"""
    synapse_values: cp.ndarray  # type: ignore
    """The spike values of neurons in this layer"""
    spike_values: cp.ndarray  # type: ignore

    def __init__(
        self,
        incoming_neurons: tuple[int, ...],
        synapses: Linear | Convolutional,
        neurons: LIF | Placeholder,
    ):
        """
        Creates a feed forward network layer.

        `incoming_neurons` is `[samples x feature dimensions..]`
        """
        self.neurons = neurons
        self.synapses = synapses

        if isinstance(synapses, Linear):
            match incoming_neurons:
                case (samples, features):
                    self.neuron_values = cp.zeros((samples, synapses.outputs), float32)  # type: ignore
                    self.synapse_values = cp.zeros((features, synapses.outputs), float32)  # type: ignore
                    self.spike_values = cp.zeros((samples, synapses.outputs), float32)  # type: ignore
                    pass
                case _:
                    raise Exception("todo")
        else:
            raise Exception("todo")

    def forward(self, inputs: cp.ndarray):  # type: ignore
        """Passes an input into this layer"""
        if isinstance(self.synapses, Linear):
            match len(inputs.shape):  # type: ignore
                case 2:
                    cp.matmul(inputs, self.synapse_values, self.neuron_values)  # type: ignore
                    if isinstance(self.neurons, LIF):
                        self.neurons.spike_update_kernel(  # type: ignore
                            self.neuron_values,  # type: ignore
                            self.neurons.decay,
                            self.neurons.threshold,
                            self.neuron_values,  # type: ignore
                            self.spike_values,  # type: ignore
                            self.neuron_values,  # type: ignore
                        )
                    else:
                        raise Exception("todo")

                case _:
                    raise Exception("todo")
        else:
            raise Exception("todo")


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
            new_layer = FeedForwardLayer(incoming_neurons, synapses, neurons)  # type: ignore
            incoming_neurons = new_layer.neuron_values.shape  # type: ignore
            self.layers.append(new_layer)

    def forward(self, inputs: cp.ndarray):  # type: ignore
        spikes = inputs  # type: ignore
        for layer in self.layers:
            layer.forward(spikes)  # type: ignore
            spikes = layer.neuron_values  # type: ignore
        return spikes  # type: ignore
