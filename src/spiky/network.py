import cupy as cp  # type: ignore
from numpy import float32
from cupy.cuda import cublas  # type: ignore

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

    # Is it worth this being a custom kernel?
    arctan_surrogate_gradient_kernel = cp.ElementwiseKernel(  # type: ignore
        in_params="float32 membrane_potential_in",
        out_params="float32 membrane_potential_out",
        operation="""
            float x = PI * membrane_potential_in
            membrane_potential_out = 1.0f / (1.0f + (x * x));
        """,
        name="arctan_surrogate_gradient_kernel",
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
    """The synapse values of neurons in this layer"""
    spike_values: cp.ndarray  # type: ignore
    """The spike values of neurons in this layer"""
    weighted_input_values: cp.ndarray  # type: ignore
    """The weighted input values into neurons"""

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
                    self.weighted_input_values = cp.zeros((samples, synapses.outputs), float32)  # type: ignore
                    pass
                case _:
                    raise Exception("todo")
        else:
            raise Exception("todo")

    def forward(self, inputs: cp.ndarray):  # type: ignore
        """Passes an input into this layer"""
        self.rforward(inputs, None, None)  # type: ignore

    def rforward(self, inputs: cp.ndarray, weighted_input_values: cp.ndarray | None, spike_values: cp.ndarray | None):  # type: ignore
        """Passes an input into this layer"""
        if isinstance(self.synapses, Linear):
            match len(inputs.shape):  # type: ignore
                case 2:
                    cp.matmul(inputs, self.synapse_values, self.weighted_input_values)  # type: ignore
                    if weighted_input_values:
                        cp.copyto(weighted_input_values, self.weighted_input_values)  # type: ignore
                    if isinstance(self.neurons, LIF):
                        self.neurons.spike_update_kernel(  # type: ignore
                            self.weighted_input_values,  # type: ignore
                            self.neurons.decay,
                            self.neurons.threshold,
                            self.neuron_values,  # type: ignore
                            self.spike_values,  # type: ignore
                            self.neuron_values,  # type: ignore
                        )
                        if spike_values:
                            cp.copyto(spike_values, self.spike_values)  # type: ignore
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


# We currently don't support "truncated BPPT" which would use batches through time to train
# for deep timesteps (in the same way using batches through samples saves memory usage but
# reduces accuracy).
class BackpropagationThroughTime:
    network: FeedForwardNetwork
    """Underlying feed forward network"""
    weighted_input_values: list[list[cp.ndarray]]  # type: ignore
    """Weighted inputs for each timestep for each layer"""
    spike_values: list[list[cp.ndarray]]  # type: ignore
    """The spikes for each timestep for each layer"""
    delta_weights: list[cp.ndarray]  # type: ignore
    """Change in weights for each layer"""
    errors: list[cp.ndarray]  # type: ignore
    """Errors in spikes for each layer"""
    inputs: list[cp.ndarray]  # type: ignore
    """Input spikes for each timestep"""
    backed: bool
    """If `backward` has been called since last `forward`"""
    number_backed: int
    """Number of timesteps backpropagated through"""

    def __init__(self, network: FeedForwardNetwork):
        self.network = network
        self.errors = [cp.zeros(layer.neuron_values.shape) for layer in network.layers]  # type: ignore
        self.delta_weights = [cp.zeros(layer.synapse_values.shape) for layer in network.layers]  # type: ignore
        self.backed = False

    def forward(self, inputs: list[cp.ndarray]):  # type: ignore
        """Performs and records the forward propagation of `inputs` through the network"""
        self.inputs = inputs
        self.__update_cache()
        for input, timestep_weighted_input_values, timestep_spike_values in zip(inputs, self.weighted_input_values, self.spike_values):  # type: ignore
            for layer, layer_timestep_weighted_input_values, layer_timestep_spike_values in zip(self.network.layers, timestep_weighted_input_values, timestep_spike_values):  # type: ignore
                layer.rforward(input, layer_timestep_weighted_input_values, layer_timestep_spike_values)  # type: ignore

    def __update_cache(self):
        """Since a user can module how many forward timesteps they want, we may need to extend the number of arrays we keep cached"""
        timesteps = len(self.inputs)  # type: ignore
        while len(self.weighted_input_values) < timesteps:  # type: ignore
            self.weighted_input_values.append([cp.zeros(layer.weighted_input_values.shape) for layer in self.network.layers])  # type: ignore

        while len(self.spike_values) < timesteps:  # type: ignore
            self.spike_values.append([cp.zeros(layer.spike_values.shape) for layer in self.network.layers])  # type: ignore

    def __surrogate(self, neurons: LIF | Placeholder, membrane_potential: cp.ndarray):  # type: ignore
        """Calculate the (surrogate) gradient for a given type of neurons with the given neuron values"""
        if isinstance(neurons, LIF):
            neurons.arctan_surrogate_gradient_kernel(membrane_potential, membrane_potential)  # type: ignore
        else:
            raise Exception("todo")

    def backward(self, targets: list[cp.ndarray]):  # type: ignore
        """
        Backpropagates `len(target)` timesteps.

        Valid code might look like:
        ```python
        bptt.forward(inputs[0][0:4])
        bptt.backward(targets[0][3:4])
        bptt.backward(targets[0][0:3])
        bptt.forward(inputs[0][4:8])
        bptt.backward(targets[0][4:8])
        bptt.update()
        bptt.forward(inputs[1][0:8])
        bptt.backward(targets[1][0:8])
        bptt.update()
        bptt.forward(inputs[2][0:8])
        bptt.backward(targets[2][4:8])
        bptt.update()
        ```
        """

        self.number_backed += len(targets)  # type: ignore
        assert self.number_backed <= len(self.inputs)  # type: ignore
        assert self.number_backed <= len(self.weighted_input_values)  # type: ignore
        assert self.number_backed <= len(self.spike_values)  # type: ignore

        for target in reversed(targets):  # type: ignore
            # If its the first `backward` call since `update` set to `0.0` so we re-zero `delta_weights` else set to `1.0` so we add to `delta_weights`. This saves having to make a seperate call to re-zero `delta_weights`.
            beta = 1.0 if self.backed else 0.0

            number_of_layers = len(self.network.layers)
            # output_layer_index
            oli = number_of_layers - 1

            # Inputs values, weighted input values and spikes for this time step.
            input_values = self.inputs.pop()  # type: ignore
            timestep = len(self.inputs)  # type: ignore
            weighted_input_values = self.weighted_input_values[timestep]  # type: ignore
            spike_values = self.spike_values[timestep]  # type: ignore

            # Output layer
            self.__surrogate(self.network.layers[oli].neurons, weighted_input_values[oli])  # type: ignore
            output_gradients = weighted_input_values  # type: ignore
            self.errors[oli] = (spike_values - target) * output_gradients  # type: ignore
            gemm(self.spike_values[oli - 1], self.errors[oli], self.delta_weights[oli], trans_a=True, beta=beta)  # type: ignore
            delta_next = self.errors[oli]  # type: ignore

            # Hidden layers
            for li in range(oli - 1, 0, -1):
                self.__surrogate(self.network.layers[li].neurons, weighted_input_values[li])  # type: ignore
                gradient = weighted_input_values[li]  # type: ignore
                gemm(delta_next, weighted_input_values[li + 1], self.errors[oli], trans_b=True)  # type: ignore
                self.errors[oli] *= gradient  # type: ignore
                gemm(spike_values[oli - 1], self.errors[oli], self.delta_weights[li], trans_a=True, beta=beta)  # type: ignore
                delta_next = self.errors[oli]  # type: ignore

            # Input layer
            self.__surrogate(weighted_input_values[0], weighted_input_values[0])  # type: ignore
            gradient = weighted_input_values[0]  # type: ignore
            gemm(delta_next, weighted_input_values[1], self.errors[0], trans_b=True)  # type: ignore
            self.errors[0] *= gradient  # type: ignore
            gemm(input_values, self.errors[0], self.delta_weights[0], trans_a=True, beta=beta)  # type: ignore

            self.backed = True

    def update(self, learning_rate: float32):
        """Updates the weights in the network"""
        for layer, delta_weights in zip(self.network.layers, self.delta_weights):  # type: ignore
            layer.synapse_values -= learning_rate * delta_weights / float32(len(self.number_backed))  # type: ignore
        self.backed = False
        self.inputs = []


def gemm(
    a: cp.ndarray,  # type: ignore
    b: cp.ndarray,  # type: ignore
    c: cp.ndarray,  # type: ignore
    trans_a: bool = False,
    trans_b: bool = False,
    beta: float32 = float32(0.0),
) -> None:
    """
    Computes C = A @ B + beta * C using cuBLAS GEMM with pre-allocated arrays.

    Args:
        a: Input matrix A (Fortran-contiguous, shape depends on trans_a)
        b: Input matrix B (Fortran-contiguous, shape depends on trans_b)
        c: Pre-allocated output matrix (Fortran-contiguous, overwritten in-place)
        trans_a: Whether to transpose A
        trans_b: Whether to transpose B
        beta: Scaling factor for existing values in C
    """
    # Validate array contiguity (critical for cuBLAS performance)
    if not (a.flags.f_contiguous and b.flags.f_contiguous and c.flags.f_contiguous):  # type: ignore
        raise ValueError("Arrays must be Fortran-contiguous (column-major order)")

    # Determine matrix dimensions
    m = c.shape[0]  # Rows of output C # type: ignore
    n = c.shape[1]  # Columns of output C # type: ignore
    k = a.shape[0] if trans_a else a.shape[1]  # Inner dimension # type: ignore

    # Set transpose operations
    op_a = cublas.CUBLAS_OP_T if trans_a else cublas.CUBLAS_OP_N  # type: ignore
    op_b = cublas.CUBLAS_OP_T if trans_b else cublas.CUBLAS_OP_N  # type: ignore

    # Leading dimensions (original matrix rows)
    lda = a.shape[0]  # type: ignore
    ldb = b.shape[0]  # type: ignore
    ldc = c.shape[0]  # type: ignore

    # Alpha is fixed to 1.0 per your Rust implementation
    alpha = 1.0

    # Execute GEMM
    cp.cublas.gemm(
        op_a,  # Transpose flag for A
        op_b,  # Transpose flag for B
        m,  # Rows of op(A) and C
        n,  # Columns of op(B) and C
        k,  # Inner dimension
        alpha,  # Always 1.0
        a,
        lda,  # A and leading dimension
        b,
        ldb,  # B and leading dimension
        beta,  # User-specified beta
        c,
        ldc,  # C and leading dimension
        out=c,  # In-place output
    )
