from typing import Any
from dataclasses import dataclass
import cudnn

@dataclass
class Linear:
    outputs: int

@dataclass
class Pooling:
    kernel_size: tuple[int, int]
    stride: tuple[int, int]

    def __init__(
        self,
        kernel_size: tuple[int, int] | int,
        stride: tuple[int, int] | int = 1,
    ):
        self.kernel_size = (
            kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        )
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)

@dataclass
class Convolutional:
    out_channels: int
    kernel_size: tuple[int, int]
    stride: tuple[int, int]
    pooling: Pooling | None

    def __init__(
        self,
        out_channels: int,
        kernel_size: tuple[int, int] | int,
        stride: tuple[int, int] | int = 1,
        pooling: Pooling | None = None,
    ):
        self.out_channels = out_channels
        self.kernel_size = (
            kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        )
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.pooling = pooling

SURROGATE_GRADIENT_KERNEL_SRC = r"""
extern "C" __global__
void surrogate_gradient_kernel(const float* x, float* out, int N) {
    const float PI = 3.141592653589793f;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float v = x[i];
        float pix = PI * v;
        out[i] = 1.0f / (1.0f + pix * pix);
    }
}
"""

def launch_surrogate_gradient_kernel(x_tensor, out_tensor):
    # Get device pointers and shape
    x_ptr = cudnn.get_device_pointer(x_tensor)
    out_ptr = cudnn.get_device_pointer(out_tensor)
    N = cudnn.num_elements(x_tensor)
    # Compile and launch the kernel (adapt as needed for your runtime)
    module = cudnn.compile_cuda_kernel(SURROGATE_GRADIENT_KERNEL_SRC, "surrogate_gradient_kernel")
    block = 256
    grid = (N + block - 1) // block
    module.surrogate_gradient_kernel(grid=(grid,1,1), block=(block,1,1), args=[x_ptr, out_ptr, N])

@dataclass
class LIF:
    decay: float
    threshold: float = 1.0

    @staticmethod
    def arctan_surrogate_gradient(membrane_potential_in, gradient_out = None) -> None:
        # Allocate output tensor with same shape
        if gradient_out is None:
            gradient_out = cudnn.tensor_like(membrane_potential_in, fill=0.0, name="sg_out")
        launch_surrogate_gradient_kernel(membrane_potential_in, gradient_out)
        return gradient_out

@dataclass
class Placeholder:
    pass

@dataclass
class FeedForwardLayer:
    neurons: LIF | Placeholder
    neuron_values: Any
    synapses: Linear | Convolutional
    synapse_values: Any
    spike_values: Any
    weighted_input_values: Any

    def __init__(
        self,
        incoming_neurons: tuple[int, ...],
        synapses: Linear | Convolutional,
        neurons: LIF | Placeholder,
    ):
        self.neurons = neurons
        self.synapses = synapses

        if isinstance(synapses, Linear):
            samples, features = incoming_neurons
            self.neuron_values = cudnn.tensor(
                shape=(samples, synapses.outputs),
                dtype=cudnn.data_type.FLOAT,
                name="ffn_neuron_values"
            )
            self.synapse_values = cudnn.tensor(
                shape=(features, synapses.outputs),
                dtype=cudnn.data_type.FLOAT,
                name="ffn_synapse_values"
            )
            self.spike_values = cudnn.tensor(
                shape=(samples, synapses.outputs),
                dtype=cudnn.data_type.FLOAT,
                name="ffn_spike_values"
            )
            self.weighted_input_values = cudnn.tensor(
                shape=(samples, synapses.outputs),
                dtype=cudnn.data_type.FLOAT,
                name="ffn_weighted_input"
            )
        else:
            assert isinstance(synapses, Convolutional)
            kernel_h, kernel_w = synapses.kernel_size
            stride_h, stride_w = synapses.stride

            samples, height, width, channels = incoming_neurons
            pool_kernel = (1, 1)
            pool_stride = (1, 1)
            if synapses.pooling is not None:
                pool_kernel = synapses.pooling.kernel_size
                pool_stride = synapses.pooling.stride

            out_h = (height - kernel_h) // stride_h + 1
            out_w = (width - kernel_w) // stride_w + 1

            self.neuron_values = cudnn.tensor(
                shape=(samples, out_h, out_w, synapses.out_channels),
                dtype=cudnn.data_type.FLOAT,
                name="cnn_neuron_values"
            )
            self.synapse_values = cudnn.tensor(
                shape=(synapses.out_channels, channels, kernel_h, kernel_w),
                dtype=cudnn.data_type.FLOAT,
                name="cnn_synapse_values"
            )
            self.spike_values = cudnn.tensor(
                shape=(samples, out_h, out_w, synapses.out_channels),
                dtype=cudnn.data_type.FLOAT,
                name="cnn_spike_values"
            )
            self.weighted_input_values = cudnn.tensor(
                shape=(samples, out_h, out_w, synapses.out_channels),
                dtype=cudnn.data_type.FLOAT,
                name="cnn_weighted_input"
            )

    def forward(self, inputs: Any):
        self.rforward(inputs, None, None)

    def rforward(self, inputs: Any, weighted_input_values: Any | None, spike_values: Any | None):
        if isinstance(self.synapses, Linear):
            matmul_op = cudnn.matmul(
                a=inputs,
                b=self.synapse_values,
                compute_data_type=cudnn.data_type.FLOAT,
                name="linear_matmul"
            )
            self.weighted_input_values = matmul_op
            if weighted_input_values is not None:
                cudnn.copyto(weighted_input_values, self.weighted_input_values)
        else:
            assert isinstance(self.synapses, Convolutional)
            conv_out = cudnn.conv_fprop(
                image=inputs,
                weight=self.synapse_values,
                padding=[0, 0],
                stride=list(self.synapses.stride),
                dilation=[1, 1],
                compute_data_type=cudnn.data_type.FLOAT,
                name="conv_fprop"
            )
            if self.synapses.pooling is not None:
                pool = self.synapses.pooling
                pool_out = cudnn.resample_fwd(
                    x=conv_out,
                    window_dims=list(pool.kernel_size),
                    pre_paddings=[0, 0],
                    post_paddings=[0, 0],
                    strides=list(pool.stride),
                    mode=cudnn.resample_mode.AVERAGE,  # or MAX, as appropriate
                    compute_data_type=cudnn.data_type.FLOAT,
                    name="pooling"
                )
                self.weighted_input_values = pool_out
            else:
                self.weighted_input_values = conv_out

        if isinstance(self.neurons, LIF):
            spiked, mem_out = LIF.lif_pointwise(
                self.weighted_input_values,
                self.neurons.decay,
                self.neurons.threshold,
                self.neuron_values
            )
            self.spike_values = spiked
            self.neuron_values = mem_out
            if spike_values is not None:
                cudnn.copyto(spike_values, self.spike_values)
        else:
            raise Exception("todo")

@dataclass
class FeedForwardNetwork:
    inputs: int
    layers: list[FeedForwardLayer]

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

    def forward(self, inputs: Any):
        spikes = inputs
        for layer in self.layers:
            layer.forward(spikes)
            spikes = layer.neuron_values
        return spikes

@dataclass
class BackpropagationThroughTime:
    network: FeedForwardNetwork
    weighted_input_values: list[list[Any]]
    spike_values: list[list[Any]]
    delta_weights: list[Any]
    errors: list[Any]
    inputs: list[Any]
    backed: bool
    number_backed: int

    def __init__(self, network: FeedForwardNetwork):
        self.network = network
        self.errors = [cudnn.tensor_like(layer.neuron_values, fill=0.0, name="bptt_error") for layer in network.layers]
        self.delta_weights = [cudnn.tensor_like(layer.synapse_values, fill=0.0, name="bptt_dw") for layer in network.layers]
        self.weighted_input_values = []
        self.spike_values = []
        self.inputs = []
        self.backed = False
        self.number_backed = 0

    def forward(self, inputs: list[Any]):
        self.inputs = inputs
        self.__update_cache()
        for input, timestep_weighted_input_values, timestep_spike_values in zip(inputs, self.weighted_input_values, self.spike_values):
            for layer, layer_timestep_weighted_input_values, layer_timestep_spike_values in zip(self.network.layers, timestep_weighted_input_values, timestep_spike_values):
                layer.rforward(input, layer_timestep_weighted_input_values, layer_timestep_spike_values)

    def __update_cache(self):
        timesteps = len(self.inputs)
        while len(self.weighted_input_values) < timesteps:
            self.weighted_input_values.append([cudnn.tensor_like(layer.weighted_input_values, fill=0.0, name="bptt_cache_wi") for layer in self.network.layers])
        while len(self.spike_values) < timesteps:
            self.spike_values.append([cudnn.tensor_like(layer.spike_values, fill=0.0, name="bptt_cache_spike") for layer in self.network.layers])

    def __surrogate(self, neurons: LIF | Placeholder, membrane_potential: Any):
        if isinstance(neurons, LIF):
            return LIF.arctan_surrogate_gradient(membrane_potential)
        else:
            raise Exception("todo")

    def __conv2d_backward_weight(
        self,
        prev_spikes: Any,
        errors: Any,
        delta_weights: Any,
        stride: tuple[int, int],
        beta: float,
    ) -> None:
        # cuDNN convolution backward filter (weight gradient)
        cudnn.conv_bwd_filter(
            x=prev_spikes,
            dy=errors,
            dw=delta_weights,
            stride=list(stride),
            padding=[0, 0],
            compute_data_type=cudnn.data_type.FLOAT,
            beta=beta,
            name="bptt_conv_bwd_filter"
        )

    def __conv2d_backward_input(
        self,
        weight: Any,
        errors: Any,
        input_shape: tuple[int, int, int, int],
        stride: tuple[int, int],
    ) -> Any:
        grad_input = cudnn.tensor(shape=input_shape, dtype=cudnn.data_type.FLOAT, name="bptt_conv_bwd_input")
        cudnn.conv_bwd_data(
            w=weight,
            dy=errors,
            dx=grad_input,
            stride=list(stride),
            padding=[0, 0],
            compute_data_type=cudnn.data_type.FLOAT,
            name="bptt_conv_bwd_data"
        )
        return grad_input

    def __backward_output(
        self,
        target: Any,
        spike_values: list[Any],
        weighted_input_values: list[Any],
        beta: float,
    ) -> Any:
        number_of_layers = len(self.network.layers)
        oli = number_of_layers - 1
        ol = self.network.layers[oli]
        output_gradients = self.__surrogate(ol.neurons, weighted_input_values[oli])
        self.errors[oli] = cudnn.pointwise(
            a=cudnn.pointwise(a=spike_values[oli], b=target, mode="sub", compute_data_type=cudnn.data_type.FLOAT),
            b=output_gradients,
            mode="mul",
            compute_data_type=cudnn.data_type.FLOAT
        )
        if isinstance(ol.synapses, Linear):
            gemm(
                a=self.spike_values[oli - 1],
                b=self.errors[oli],
                c=self.delta_weights[oli],
                trans_a=True,
                beta=beta
            )
            return self.errors[oli]
        else:
            assert isinstance(ol.synapses, Convolutional)
            self.__conv2d_backward_weight(
                prev_spikes=spike_values[oli - 1],
                errors=self.errors[oli],
                delta_weights=self.delta_weights[oli],
                stride=ol.synapses.stride,
                beta=beta,
            )
            return self.__conv2d_backward_input(
                weight=ol.synapse_values,
                errors=self.errors[oli],
                input_shape=self.network.layers[oli - 1].neuron_values.shape,
                stride=ol.synapses.stride,
            )

    def __backward_hidden(
        self,
        spike_values: list[Any],
        weighted_input_values: list[Any],
        beta: float,
        delta_next: Any,
        li: int,
    ) -> Any:
        layer = self.network.layers[li]
        after_layer = self.network.layers[li + 1]
        gradient = self.__surrogate(layer.neurons, weighted_input_values[li])
        if isinstance(after_layer.synapses, Linear):
            gemm(
                a=delta_next,
                b=weighted_input_values[li + 1],
                c=self.errors[li],
                trans_b=True
            )
        else:
            assert isinstance(after_layer.synapses, Convolutional)
            self.errors[li] = self.__conv2d_backward_input(
                weight=after_layer.synapse_values,
                errors=delta_next,
                input_shape=layer.neuron_values.shape,
                stride=after_layer.synapses.stride,
            )
        self.errors[li] = cudnn.pointwise(
            a=self.errors[li],
            b=gradient,
            mode="mul",
            compute_data_type=cudnn.data_type.FLOAT
        )
        if isinstance(layer.synapses, Linear):
            gemm(
                a=spike_values[li - 1],
                b=self.errors[li],
                c=self.delta_weights[li],
                trans_a=True,
                beta=beta
            )
        else:
            assert isinstance(layer.synapses, Convolutional)
            self.__conv2d_backward_weight(
                prev_spikes=spike_values[li - 1],
                errors=self.errors[li],
                delta_weights=self.delta_weights[li],
                stride=layer.synapses.stride,
                beta=beta,
            )
        return self.errors[li]

    def __backward_input(
        self,
        weighted_input_values: list[Any],
        beta: float,
        delta_next: Any,
        input_values: Any,
    ) -> None:
        input_layer = self.network.layers[0]
        first_hidden_layer = self.network.layers[1]
        gradient = self.__surrogate(input_layer.neurons, weighted_input_values[0])
        if isinstance(first_hidden_layer.synapses, Linear):
            gemm(
                a=delta_next,
                b=weighted_input_values[1],
                c=self.errors[0],
                trans_b=True
            )
        else:
            assert isinstance(first_hidden_layer.synapses, Convolutional)
            self.errors[0] = self.__conv2d_backward_input(
                weight=first_hidden_layer.synapse_values,
                errors=delta_next,
                input_shape=input_layer.neuron_values.shape,
                stride=first_hidden_layer.synapses.stride,
            )
        self.errors[0] = cudnn.pointwise(
            a=self.errors[0],
            b=gradient,
            mode="mul",
            compute_data_type=cudnn.data_type.FLOAT
        )
        if isinstance(input_layer.synapses, Linear):
            gemm(
                a=input_values,
                b=self.errors[0],
                c=self.delta_weights[0],
                trans_a=True,
                beta=beta
            )
        else:
            assert isinstance(input_layer.synapses, Convolutional)
            self.__conv2d_backward_weight(
                prev_spikes=input_values,
                errors=self.errors[0],
                delta_weights=self.delta_weights[0],
                stride=input_layer.synapses.stride,
                beta=beta,
            )

    def backward(self, targets: list[Any]):
        self.number_backed += len(targets)
        assert self.number_backed <= len(self.inputs)
        assert self.number_backed <= len(self.weighted_input_values)
        assert self.number_backed <= len(self.spike_values)
        for target in reversed(targets):
            beta = 1.0 if self.backed else 0.0
            number_of_layers = len(self.network.layers)
            input_values = self.inputs.pop()
            timestep = len(self.inputs)
            weighted_input_values = self.weighted_input_values[timestep]
            spike_values = self.spike_values[timestep]
            delta_next = self.__backward_output(
                target, spike_values, weighted_input_values, beta
            )
            for li in range(number_of_layers - 2, 0, -1):
                delta_next = self.__backward_hidden(
                    spike_values,
                    weighted_input_values,
                    beta,
                    delta_next,
                    li,
                )
            self.__backward_input(
                weighted_input_values,
                beta,
                delta_next,
                input_values,
            )
            self.backed = True

    def update(self, learning_rate: float):
        for layer, delta_weights in zip(self.network.layers, self.delta_weights):
            update = cudnn.pointwise(
                a=delta_weights,
                scale=learning_rate / float(self.number_backed),
                mode="scale",
                compute_data_type=cudnn.data_type.FLOAT
            )
            layer.synapse_values = cudnn.pointwise(
                a=layer.synapse_values,
                b=update,
                mode="sub",
                compute_data_type=cudnn.data_type.FLOAT
            )
        self.backed = False
        self.inputs = []

def gemm(
    a: Any,
    b: Any,
    c: Any,
    trans_a: bool = False,
    trans_b: bool = False,
    beta: float = 0.0,
) -> None:
    # cuDNN matmul, with optional transposes and beta scaling for in-place update
    cudnn.matmul(
        a=a,
        b=b,
        c=c,
        trans_a=trans_a,
        trans_b=trans_b,
        beta=beta,
        compute_data_type=cudnn.data_type.FLOAT,
        name="bptt_gemm"
    )
