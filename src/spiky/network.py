from typing import Any
import cupy as cp  # type: ignore
from numpy import float32
from cupy.cuda import cublas  # type: ignore
from dataclasses import dataclass
import cupy.cudnn

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


@dataclass
class Placeholder:
    pass


@dataclass
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

        Args:
            incoming_neurons: `[samples x feature dimensions..]`
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
            assert isinstance(synapses, Convolutional)
            # Unpack kernel and stride dimensions
            kernel_h, kernel_w = synapses.kernel_size
            stride_h, stride_w = synapses.stride

            match incoming_neurons:
                case (samples, height, width, channels):
                    is_pooling = synapses.pooling is not None
                    pool_kernel = synapses.pooling.kernel_size if is_pooling else (1,1)
                    pool_stride = synapses.pooling.stride if is_pooling else (1,1)

                    # Calculate output dimensions
                    out_h = (height - kernel_h) // stride_h + 1
                    out_w = (width - kernel_w) // stride_w + 1
                        

                    # Initialize arrays
                    self.neuron_values = cp.zeros((samples, out_h, out_w, synapses.out_channels), float32)  # type: ignore
                    self.synapse_values = cp.zeros((synapses.out_channels, channels, kernel_h, kernel_w), float32)  # type: ignore
                    self.spike_values = cp.zeros((samples, out_h, out_w, synapses.out_channels), float32)  # type: ignore
                    self.weighted_input_values = cp.zeros((samples, out_h, out_w, synapses.out_channels), float32)  # type: ignore
                case _:
                    raise Exception("Unsupported input shape for convolutional layer")

    def forward(self, inputs: cp.ndarray):  # type: ignore
        """Passes an input into this layer"""
        self.rforward(inputs, None, None)  # type: ignore

    def rforward(self, inputs: cp.ndarray, weighted_input_values: cp.ndarray | None, spike_values: cp.ndarray | None):  # type: ignore
        """Passes an input into this layer"""

        # Propagate through synapses.
        if isinstance(self.synapses, Linear):
            match len(inputs.shape):  # type: ignore
                case 2:  # [samples, features]
                    cp.matmul(inputs, self.synapse_values, self.weighted_input_values)  # type: ignore
                    if weighted_input_values:
                        cp.copyto(weighted_input_values, self.weighted_input_values)  # type: ignore
                case _:
                    raise Exception("todo")
        else:
            assert isinstance(self.synapses, Convolutional)

            # Get convolution parameters
            stride_h, stride_w = self.synapses.stride

            # Create cuDNN descriptors
            x_desc = cupy.cudnn.create_tensor_descriptor(  # type: ignore
                inputs, format=cupy.cudnn.CUDNN_TENSOR_NHWC  # type: ignore
            )
            w_desc = cupy.cudnn.create_filter_descriptor(  # type: ignore
                self.synapse_values, format=cupy.cudnn.CUDNN_TENSOR_NCHW  # type: ignore
            )
            y_desc = cupy.cudnn.create_tensor_descriptor(  # type: ignore
                self.weighted_input_values, format=cupy.cudnn.CUDNN_TENSOR_NHWC  # type: ignore
            )
            conv_desc = cupy.cudnn.create_convolution_descriptor(  # type: ignore
                pad=(0, 0),
                stride=(stride_h, stride_w),
                dtype=inputs.dtype,  # type: ignore
                mode=cupy.cudnn.CUDNN_CROSS_CORRELATION,  # type: ignore
            )

            # Find optimal convolution algorithm
            algo = cupy.cudnn.get_convolution_forward_algorithm(  # type: ignore
                x_desc,
                w_desc,
                conv_desc,
                y_desc,
                preference=cupy.cudnn.CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,  # type: ignore
            )

            # Perform convolution
            cupy.cudnn.convolution_forward(  # type: ignore
                alpha=1.0,
                src_desc=x_desc,
                src_data=inputs,
                filter_desc=w_desc,
                filter_data=self.synapse_values,  # type: ignore
                conv_desc=conv_desc,
                algo=algo,
                work_space=None,  # Auto-managed by cuDNN
                beta=0.0,
                dest_desc=y_desc,
                dest_data=self.weighted_input_values,  # type: ignore
            )

        # Propagate into neurons.
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
#
# With some changes backprop could be parallelized since each call only needs 1
# timestep of foreprop, potentially when each foreprop timestep completes it could dispatch a
# backprop call for that step and then it could resolve after all backprop steps
# have resolved. This approach would increase the flat amount of compute and
# memory required but given the high parallelism could offer massive gains for
# large models by using multiple GPUs potentially across multiple systems.@dataclass
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
    cudnn_handles: list[Any]  # type: ignore

    def __init__(self, network: FeedForwardNetwork):
        self.network = network
        self.errors = [cp.zeros(layer.neuron_values.shape) for layer in network.layers]  # type: ignore
        self.delta_weights = [cp.zeros(layer.synapse_values.shape) for layer in network.layers]  # type: ignore
        self.backed = False
        self.cudnn_handles = {}  # Store cuDNN handles per device # type: ignore

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

    def __get_cudnn_handle(self):  # type: ignore
        """Get or create cuDNN handle for current device"""
        device_id = cp.cuda.Device().id  # type: ignore
        if device_id not in self.cudnn_handles:  # type: ignore
            handle = cp.cuda.cudnn.get_handle()  # type: ignore
            self.cudnn_handles[device_id] = handle  # type: ignore
        return self.cudnn_handles[device_id]  # type: ignore

    def __conv2d_backward_weight(
        self,
        prev_spikes: cp.ndarray,  # type: ignore
        errors: cp.ndarray,  # type: ignore
        delta_weights: cp.ndarray,  # type: ignore
        stride: tuple[int, int],
        beta: float,
    ) -> None:
        """cuDNN-accelerated weight gradient calculation"""
        handle = self.__get_cudnn_handle()  # type: ignore

        # Create tensor descriptors (NHWC format)
        x_desc = cp.cuda.cudnn.create_tensor_descriptor(  # type: ignore
            prev_spikes, format=cp.cuda.cudnn.CUDNN_TENSOR_NHWC  # type: ignore
        )
        dy_desc = cp.cuda.cudnn.create_tensor_descriptor(  # type: ignore
            errors, format=cp.cuda.cudnn.CUDNN_TENSOR_NHWC  # type: ignore
        )
        dw_desc = cp.cuda.cudnn.create_filter_descriptor(delta_weights)  # type: ignore

        # Create convolution descriptor
        conv_desc = cp.cuda.cudnn.create_convolution_descriptor(  # type: ignore
            pad=(0, 0),
            stride=stride,
            dtype=prev_spikes.dtype,  # type: ignore
            mode=cp.cuda.cudnn.CUDNN_CROSS_CORRELATION,  # type: ignore
        )

        # Find optimal algorithm
        algo = cp.cuda.cudnn.get_convolution_backward_filter_algorithm(  # type: ignore
            handle,
            x_desc,
            dy_desc,
            conv_desc,
            dw_desc,
            preference=cp.cuda.cudnn.CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,  # type: ignore
        )

        # Get workspace size
        workspace_size = cp.cuda.cudnn.get_convolution_backward_filter_workspace_size(  # type: ignore
            handle, x_desc, dy_desc, conv_desc, dw_desc, algo
        )
        workspace = cp.empty(workspace_size, dtype=cp.uint8) if workspace_size > 0 else None  # type: ignore

        # Perform convolution backward
        alpha = 1.0
        beta_cudnn = 1.0 if beta != 0 else 0.0  # Accumulate if beta != 0
        cp.cuda.cudnn.convolution_backward_filter(  # type: ignore
            handle,
            alpha,
            x_desc,
            prev_spikes,
            dy_desc,
            errors,
            conv_desc,
            algo,
            workspace,
            workspace_size,
            beta_cudnn,
            dw_desc,
            delta_weights,
        )

    def __conv2d_backward_input(  # type: ignore
        self,
        weight: cp.ndarray,  # type: ignore
        errors: cp.ndarray,  # type: ignore
        input_shape: tuple[int, int, int, int],
        stride: tuple[int, int],
    ) -> cp.ndarray:  # type: ignore
        """cuDNN-accelerated input gradient calculation"""
        handle = self.__get_cudnn_handle()  # type: ignore
        grad_input = cp.zeros(input_shape, dtype=cp.float32)  # type: ignore

        # Create tensor descriptors (NHWC format)
        dx_desc = cp.cuda.cudnn.create_tensor_descriptor(  # type: ignore
            grad_input, format=cp.cuda.cudnn.CUDNN_TENSOR_NHWC  # type: ignore
        )
        dy_desc = cp.cuda.cudnn.create_tensor_descriptor(  # type: ignore
            errors, format=cp.cuda.cudnn.CUDNN_TENSOR_NHWC  # type: ignore
        )
        w_desc = cp.cuda.cudnn.create_filter_descriptor(weight)  # type: ignore

        # Create convolution descriptor
        conv_desc = cp.cuda.cudnn.create_convolution_descriptor(  # type: ignore
            pad=(0, 0),
            stride=stride,
            dtype=errors.dtype,  # type: ignore
            mode=cp.cuda.cudnn.CUDNN_CROSS_CORRELATION,  # type: ignore
        )

        # Find optimal algorithm
        algo = cp.cuda.cudnn.get_convolution_backward_data_algorithm(  # type: ignore
            handle,
            w_desc,
            dy_desc,
            conv_desc,
            dx_desc,
            preference=cp.cuda.cudnn.CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,  # type: ignore
        )

        # Get workspace size
        workspace_size = cp.cuda.cudnn.get_convolution_backward_data_workspace_size(  # type: ignore
            handle, w_desc, dy_desc, conv_desc, dx_desc, algo
        )
        workspace = cp.empty(workspace_size, dtype=cp.uint8) if workspace_size > 0 else None  # type: ignore

        # Perform convolution backward
        alpha = 1.0
        beta = 0.0  # Overwrite grad_input
        cp.cuda.cudnn.convolution_backward_data(  # type: ignore
            handle,
            alpha,
            w_desc,
            weight,
            dy_desc,
            errors,
            conv_desc,
            algo,
            workspace,
            workspace_size,
            beta,
            dx_desc,
            grad_input,
        )
        return grad_input  # type: ignore

    def __backward_output(  # type: ignore
        self,
        target: cp.ndarray,  # type: ignore
        spike_values: list[cp.ndarray],  # type: ignore
        weighted_input_values: list[cp.ndarray],  # type: ignore
        beta: float,
    ) -> cp.ndarray:  # type: ignore
        number_of_layers = len(self.network.layers)
        # output_layer_index
        oli = number_of_layers - 1

        ol = self.network.layers[oli]
        self.__surrogate(ol.neurons, weighted_input_values[oli])  # type: ignore
        output_gradients = weighted_input_values  # type: ignore
        self.errors[oli] = (spike_values - target) * output_gradients  # type: ignore

        if isinstance(ol.synapses, Linear):
            # Linear layer weight update
            gemm(self.spike_values[oli - 1], self.errors[oli], self.delta_weights[oli], trans_a=True, beta=beta)  # type: ignore
            return self.errors[oli]  # type: ignore
        else:
            assert isinstance(ol.synapses, Convolutional)
            # Convolutional layer weight update
            self.__conv2d_backward_weight(  # type: ignore
                prev_spikes=spike_values[oli - 1],  # type: ignore
                errors=self.errors[oli],  # type: ignore
                delta_weights=self.delta_weights[oli],  # type: ignore
                stride=ol.synapses.stride,
                beta=beta,
            )
            # Calculate input gradients for previous layer
            return self.__conv2d_backward_input(  # type: ignore
                weight=ol.synapse_values,  # type: ignore
                errors=self.errors[oli],  # type: ignore
                input_shape=self.network.layers[oli - 1].neuron_values.shape,  # type: ignore
                stride=ol.synapses.stride,
            )

    def __backward_hidden(  # type: ignore
        self,
        spike_values: list[cp.ndarray],  # type: ignore
        weighted_input_values: list[cp.ndarray],  # type: ignore
        beta: float,
        delta_next: cp.ndarray,  # type: ignore
        li: int,
    ) -> cp.ndarray:  # type: ignore
        layer = self.network.layers[li]
        after_layer = self.network.layers[li + 1]
        self.__surrogate(layer.neurons, weighted_input_values[li])  # type: ignore
        gradient = weighted_input_values[li]  # type: ignore

        # Calculate the error for this layer
        if isinstance(after_layer.synapses, Linear):
            gemm(delta_next, weighted_input_values[li + 1], self.errors[li], trans_b=True)  # type: ignore
        else:
            assert isinstance(after_layer.synapses, Convolutional)
            self.errors[li] = self.__conv2d_backward_input(  # type: ignore
                weight=after_layer.synapse_values,  # type: ignore
                errors=delta_next,  # type: ignore
                input_shape=layer.shape,  # type: ignore
                stride=after_layer.synapses.stride,  # type: ignore
            )

        self.errors[li] *= gradient  # type: ignore

        # Update weights for current layer
        if isinstance(layer.synapses, Linear):
            gemm(spike_values[li - 1], self.errors[li], self.delta_weights[li], trans_a=True, beta=beta)  # type: ignore
        else:
            assert isinstance(layer.synapses, Convolutional)
            self.__conv2d_backward_weight(  # type: ignore
                prev_spikes=spike_values[li - 1],  # type: ignore
                errors=self.errors[li],  # type: ignore
                delta_weights=self.delta_weights[li],  # type: ignore
                stride=layer.synapses.stride,
                beta=beta,
            )

        return self.errors[li]  # type: ignore

    def __backward_input(
        self,
        weighted_input_values: list[cp.ndarray],  # type: ignore
        beta: float,
        delta_next: cp.ndarray,  # type: ignore
        input_values: cp.ndarray,  # type: ignore
    ) -> None:
        input_layer = self.network.layers[0]
        first_hidden_layer = self.network.layers[1]
        self.__surrogate(weighted_input_values[0], weighted_input_values[0])  # type: ignore
        gradient = weighted_input_values[0]  # type: ignore

        if isinstance(first_hidden_layer.synapses, Linear):
            gemm(delta_next, weighted_input_values[1], self.errors[0], trans_b=True)  # type: ignore
        else:
            assert isinstance(first_hidden_layer.synapses, Convolutional)
            self.errors[0] = self.__conv2d_backward_input(  # type:ignore
                weight=first_hidden_layer.synapse_values,  # type:ignore
                errors=delta_next,  # type:ignore
                input_shape=input_layer.neuron_values.shape,  # type:ignore
                stride=first_hidden_layer.synapses.stride,
            )

        self.errors[0] *= gradient  # type: ignore

        if isinstance(input_layer.synapses, Linear):
            gemm(input_values, self.errors[0], self.delta_weights[0], trans_a=True, beta=beta)  # type: ignore
        else:
            assert isinstance(input_layer.synapses, Convolutional)
            self.__conv2d_backward_weight(  # type:ignore
                prev_spikes=input_values,  # type:ignore
                errors=self.errors[0],  # type:ignore
                delta_weights=self.delta_weights[0],  # type:ignore
                stride=input_layer.synapses.stride,
                beta=beta,
            )

    def backward(self, targets: list[cp.ndarray]):  # type: ignore
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

            # Inputs values, weighted input values and spikes for this time step.
            input_values = self.inputs.pop()  # type: ignore
            timestep = len(self.inputs)  # type: ignore
            weighted_input_values = self.weighted_input_values[timestep]  # type: ignore
            spike_values = self.spike_values[timestep]  # type: ignore

            # Output layer
            delta_next = self.__backward_output(  # type: ignore
                target, spike_values, weighted_input_values, beta  # type: ignore
            )

            # Hidden layers
            for li in range(number_of_layers - 2, 0, -1):
                delta_next = self.__backward_hidden(  # type: ignore
                    spike_values,
                    weighted_input_values,
                    beta,
                    delta_next,  # type: ignore
                    li,
                )

            # Input layer
            self.__backward_input(  # type: ignore
                weighted_input_values,
                beta,
                delta_next,  # type: ignore
                input_values,  # type: ignore
            )
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
