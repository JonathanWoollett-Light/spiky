# import pytest
import spiky.network as sn
import cupy as cp  # type: ignore
from numpy import float32

DECAY: float32 = float32(0.8)
THRESHOLD: float32 = float32(1.0)
INPUT: int = 28 * 28
BATCH: int = 10_000
LEARNING_RATE: float32 = float32(0.02)


def test_net():
    neuron = sn.LIF(DECAY, THRESHOLD)
    net = sn.FeedForwardNetwork(
        BATCH,
        INPUT,
        [(sn.Linear(784), neuron), (sn.Linear(800), neuron), (sn.Linear(10), neuron)],
    )
    net.forward(cp.zeros((BATCH, INPUT), float32))  # type: ignore
    assert False


def test_bptt():
    TIMESTEPS = 10
    neuron = sn.LIF(DECAY, THRESHOLD)
    net = sn.FeedForwardNetwork(
        BATCH,
        INPUT,
        [(sn.Linear(784), neuron), (sn.Linear(800), neuron), (sn.Linear(10), neuron)],
    )
    bptt = sn.BackpropagationThroughTime(net)
    bptt.forward([cp.zeros((BATCH, INPUT), float32) for _ in range(TIMESTEPS)])  # type: ignore
    bptt.backward([cp.zeros((BATCH, INPUT), float32) for _ in range(TIMESTEPS)])  # type: ignore
    bptt.update(LEARNING_RATE)
