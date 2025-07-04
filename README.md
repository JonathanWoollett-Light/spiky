# spiky

The general idea is to support training a variety of state of the art models by assmebling state of the art techniques then running these against robotics benchmarks to answer experimental questions e.g. "are more bio-plausible models better at embodied learning?".

## Installation

This package is not published yet.

## Development

- Manages depdencies with [Poetry](https://python-poetry.org/).
- Generate docs with `poetry run pdoc -o docs src/spiky`.
- Format with `poetry run black --check .`
- Check types with `poetry run pyright --warnings`
- Lint with `poetry run pylint .`
- Test with `poetry run pytest`

### Development Prerequisites

1. Python >=3.13
1. [Poetry](https://python-poetry.org/)

### Development Installation

1. Install dependencies

   ```bash
   poetry install
   ```

1. Install PyTorch within the enviroment e.g.

   ```bash
   poetry run pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

### Testing

You can run a specific test with (where `-s` avoids capturing the output):

```
poetry run pytest .\tests\test_against_snntorch.py::test_against_snntorch -s
```

## TODOs

The below table lists all the dimensions I am looking at testing across, for practical concerns I
wouldn't be testing the cartesian product of all combinations but I will try and do as many as I can.

Benchmark|Synapses|Architectures|Neurons|Optimizers|Hardware
---|---|---|---|---|---
[N-MNIST](https://www.garrickorchard.com/datasets/n-mnist)|Linear|Feed forward|Leaky Integrate and Fire (LIF)|Backpropagation through time (BPTT)|CPU
[ST-MNIST](https://hh-see.com/projects/2_project/)|Convolutional|Reservoir|Adaptive LIF|SpikeProp|GPU
[Isaac-Cartpole-v0](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)| | |Resonate and fire|SuperSpike|TPU
[Isaac-Ant-v0](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)| | |Quadratic LIF|SLAYER|FPGA
[Isaac-Humanoid-v0](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)| | |Izhikevich|EventProp|FPAA
| | | |Hodgkin-Huxley|Spike Timing Dependant Plasticity (STDP)|memFPAA[^1]
| | | |Sigma Delta (SDNN)| |

[^1]: [Memristive Field-Programmable Analog Arrays for Analog Computing](https://advanced.onlinelibrary.wiley.com/doi/full/10.1002/adma.202206648)

From these experiments I hope to gather the data of:

- Time to reach X accuracy.
- Operations to reach X accuracy (all variants e.g. floating point operations, synpatic operations, effective operations, etc.).
- Power to reach X accuracy.
- Cost to reach X accuracy.

From these I hope to form a view on:

- Models which can outperform existing models given better hardware optimization.
- Non-standard models which can currently outperform existing standard models.
- Models which are best at task Y (cartpole, ant, etc.).
- etc.

### Misc

- re-write to lower level `cuDNN` to avoid `CuPy`s bad documentation (for `cupy.cudnn`) and worse performance.
  The aim should be to acheive similat performance to snnTorch in n-mnist classification.
- add GitHub action that runs black, pyright, pytest and pdoc (for pdoc it should also post the github page in the action)
- run `snnTorch` to compare and improve until reaching similar performance for n-mnist and st-mnist.
- output the EFLOPs metrics for models to give an idea of their foundational performance.
- run a test including sparsity in the cost function
- look at [ml_genn](https://github.com/genn-team/ml_genn).
- I went to a conferece and heard about "audoAdjoint" which is intended to be a version of automatic differentiation that works for event driven models and SNNs. Look into this.
- paper from Matias Barandiaran and James Stovold on "Developmental Graph Cellular Automata Can Grow Reservoirs" and lookup info on plastic reservoirs and how these could be implemented/used.
