import numpy as np
from gin import nmnist
from pathlib import Path
import matplotlib.pyplot as plt
import plotly.express as px
from tqdm import tqdm
from matplotlib import cm

TEST_PATH = Path("./tmp/test.npy")
TEST_LABELS_PATH = Path("./tmp/test_labels.npy")
TRAIN_PATH = Path("./tmp/train.npy")
TRAIN_LABELS_PATH = Path("./tmp/train_labels.npy")

def plot_layer_spikes(layer_spikes, test_labels, label, save, title):
    filtered = layer_spikes[test_labels == label]
    summed = filtered.sum(axis=0)
    
    plt.title(title)
    plt.imshow(summed, cmap="Greys")
    plt.savefig(save, dpi=2000)
    
def analyze_label(
    label,
    test_labels,
    layer_spikes_0,
    layer_spikes_1,
    layer_spikes_2,
    layer_spikes_3,
    layer_spikes_4
):
    
    plot_layer_spikes(layer_spikes_0, test_labels, label, f"./tmp/label_{label}_layer_0.png", f"Spike Recordings for label {label} at layer 0")
    plot_layer_spikes(layer_spikes_1, test_labels, label, f"./tmp/label_{label}_layer_1.png", f"Spike Recordings for label {label} at layer 1")
    plot_layer_spikes(layer_spikes_2, test_labels, label, f"./tmp/label_{label}_layer_2.png", f"Spike Recordings for label {label} at layer 2")
    plot_layer_spikes(layer_spikes_3, test_labels, label, f"./tmp/label_{label}_layer_3.png", f"Spike Recordings for label {label} at layer 3")
    plot_layer_spikes(layer_spikes_4, test_labels, label, f"./tmp/label_{label}_layer_4.png", f"Spike Recordings for label {label} at layer 4")

    l0 = layer_spikes_0[test_labels == label]
    l1 = layer_spikes_1[test_labels == label]
    l2 = layer_spikes_2[test_labels == label]
    l3 = layer_spikes_3[test_labels == label]
    l4 = layer_spikes_4[test_labels == label]
    all_filtered = np.concatenate((l0,l1,l2,l3,l4), axis=1)
    print(f"all_filtered.shape: {all_filtered.shape}")
    all_summed = all_filtered.sum(axis=0)
    plt.title(f"Spike Recordings for label {label}")
    plt.imshow(all_summed, cmap="Greys")
    plt.savefig(f"./tmp/label_{label}_layer_all.png", dpi=2000)
    

    # nonzero = filtered.nonzero()
    # print(f"nonzero shapes: {nonzero[0].shape}, {nonzero[1].shape}, {nonzero[2].shape}")
    # x_coords = nonzero[1]
    # y_coords = nonzero[2]
    # print(f"len(x_coords): {len(x_coords)}")
    # print(f"len(y_coords): {len(y_coords)}")
    # fig = px.scatter(x=x_coords[:1000], y=y_coords[:1000])
    # fig.show()

    # plt.hexbin(x=x_coords[:100], y=y_coords[:100], bins='log', gridsize=())
    # plt.show()

    # y_coords = l0_coords[0]
    # x_coords = l0_coords[1]
    # plt.scatter(x_coords, y_coords, alpha=0.6, s=50)
    # plt.xlabel('Timestep')
    # plt.ylabel('Neuron')
    # plt.title('Spike recordings in layer 1 for 0')
    # plt.grid(True, alpha=0.3)
    # plt.show()
    # plt.save(f"./tmp/label_{}_layer_0.png")

    # l1 = layer_spikes_1[samples, :, :]
    # l2 = layer_spikes_2[samples, :, :]
    # l3 = layer_spikes_3[samples, :, :]
    # l4 = layer_spikes_4[samples, :, :]

def test_analyze():
    if TEST_LABELS_PATH.exists() and TRAIN_LABELS_PATH.exists():
        test_labels = np.load(TEST_LABELS_PATH, mmap_mode="r")
        train_labels = np.load(TRAIN_LABELS_PATH, mmap_mode="r")
    else:
        loader = nmnist.nmnist()
        data = loader.frames()
        test_labels, train_labels = data.test_labels, data.train_labels
        np.save(TEST_PATH, data.test)
        np.save(TRAIN_PATH, data.train)
        np.save(TEST_LABELS_PATH, test_labels)
        np.save(TRAIN_LABELS_PATH, train_labels)

    # This array of shape `[samples]` has the dataset labels which integers 0 to 9
    print(f"test_labels.shape: {test_labels.shape}")

    # These arrays of shape `[sample x neuron x timestep]` record whether spike
    # occurred for a given neuron in a given layer at a given timestep
    layer_spikes_0 = np.load("./tmp/layer_spikes_0.npy", mmap_mode="r")
    layer_spikes_1 = np.load("./tmp/layer_spikes_1.npy", mmap_mode="r")
    layer_spikes_2 = np.load("./tmp/layer_spikes_2.npy", mmap_mode="r")
    layer_spikes_3 = np.load("./tmp/layer_spikes_3.npy", mmap_mode="r")
    layer_spikes_4 = np.load("./tmp/layer_spikes_4.npy", mmap_mode="r")

    print(f"layer_spikes_0.shape: {layer_spikes_0.shape}")
    print(f"layer_spikes_1.shape: {layer_spikes_1.shape}")
    print(f"layer_spikes_2.shape: {layer_spikes_2.shape}")
    print(f"layer_spikes_3.shape: {layer_spikes_3.shape}")
    print(f"layer_spikes_4.shape: {layer_spikes_4.shape}")

    for label in range(10):
        analyze_label(
            label, test_labels, layer_spikes_0, layer_spikes_1, layer_spikes_2,
            layer_spikes_3, layer_spikes_4
        )