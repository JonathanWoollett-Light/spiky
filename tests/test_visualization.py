import torch  # type:ignore
import torch.nn as nn  # type:ignore
import snntorch as snn  # type:ignore
from snntorch import surrogate  # type:ignore
from snntorch import utils  # type:ignore
import spiky.network as sn
import numpy as np
from torch.autograd import grad
import csv
from gin import nmnist
from tqdm import tqdm
from tempfile import NamedTemporaryFile
from pathlib import Path


# type: ignore
def test_against_snntorch():
    # Set fixed seeds for reproducibility
    torch.manual_seed(42)  # type:ignore
    np.random.seed(0)

    BATCH_SIZE = 5000
    INPUT_SIZE = 2312  # flattened polarity (2) x dim (34) and y dim (34)
    TEST_PATH = Path("./tmp/test.npy")
    TEST_LABELS_PATH = Path("./tmp/test_labels.npy")
    TRAIN_PATH = Path("./tmp/train.npy")
    TRAIN_LABELS_PATH = Path("./tmp/train_labels.npy")

    beta = 0.5  # The "decay" factor.

    # Sanity check
    assert 2 * 34 * 34 == INPUT_SIZE

    # Extract and store constant weights/biases
    params = torch.load("./tmp/params.pth")

    # Create net
    layers = [784, 784, 392, 196, 10]
    neuron_type = sn.LIF(np.float32(beta))
    net = sn.FeedForwardNetwork(
        BATCH_SIZE,
        2312,
        [(sn.Linear(n), neuron_type) for n in layers],
    )

    # Set net weights and biases
    for i in range(1, 10, 2):
        weight = params[f"{i}.weight"]  # type: ignore
        bias = params[f"{i}.bias"]  # type: ignore
        j = i // 2
        net.layers[j].synapse_weights = weight.numpy().T  # type: ignore
        net.layers[j].synapse_biases = bias.numpy().T  # type: ignore

    # Get data
    if TEST_PATH.exists() and TRAIN_PATH.exists():
        test = np.load(TEST_PATH, mmap_mode="r")
        train = np.load(TRAIN_PATH, mmap_mode="r")
    else:
        loader = nmnist.nmnist()
        data = loader.frames()
        test, train = data.test, data.train
        np.save(TEST_PATH, test)
        np.save(TRAIN_PATH, train)
        np.save(TEST_LABELS_PATH, data.test_labels)
        np.save(TRAIN_LABELS_PATH, data.train_labels)

    # Sanity checks
    TIMESTEPS = 337
    print(f"test.shape: {test.shape}")
    print(f"train.shape: {train.shape}")
    assert test.shape[0] == TIMESTEPS
    assert train.shape[0] == TIMESTEPS
    test_samples = test.shape[1]
    print(f"test_samples: {test_samples}")
    assert test_samples % BATCH_SIZE == 0

    # Store all spikes across all layers, all timesteps and all samples.

    spike_store: list[npt.NDArray[np.float32]] = [
        np.memmap(
            filename=NamedTemporaryFile(),
            dtype=np.float32,
            mode="w+",
            shape=(test_samples, n, TIMESTEPS),
        )
        for n in layers
    ]

    # Perform forward pass
    #
    # Iterate over batches
    with tqdm(total=test_samples * TIMESTEPS * len(layers), desc="Forward") as bar:
        # TODO This skips the last partial batch, this is fine because currently
        # the net only supports static batch sizes so wouldn't work if given a
        # partial batch. We can add support for this (most likely but simply
        # padding the partial batch with zeros).
        for batch_start_idx in range(0, test_samples, BATCH_SIZE):
            batch_end_idx = batch_start_idx + BATCH_SIZE

            # Iterate over timesteps
            for ts in range(TIMESTEPS):
                input_spikes = train[ts, batch_start_idx:batch_end_idx, :, :, :].copy()
                input_spikes = input_spikes.reshape((BATCH_SIZE, INPUT_SIZE))

                # Iterate over layers
                # So as to store spikes from each layer, rather than just output.
                for layer, layer_store in zip(net.layers, spike_store):
                    # Run forward layer
                    layer.forward(input_spikes)

                    # Store spikes
                    layer_store[batch_start_idx:batch_end_idx, :, ts] = (
                        layer.spike_values.copy()
                    )

                    # Set inputs for next layer
                    input_spikes = layer.spike_values.copy()

                    bar.update(BATCH_SIZE)

                    # Flush to disc layer spikes in this batch to save memory
                    layer_store.flush()

    for i, layer_store in enumerate(spike_store):
        np.save(f"./tmp/layer_spikes_{i}.npy", layer_store)
