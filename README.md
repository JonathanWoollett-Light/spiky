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
- We can remove all the `#type: ignore`s when `cupy` sorts itself out and adds a type stub.

### Development Prerequisites

1. Python >=3.13
2. [Poetry](https://python-poetry.org/)

### Development Installation

1. Install dependencies

   ```bash
   poetry install
   ```

2. Install PyTorch within the enviroment e.g.

   ```bash
   poetry run pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

### Testing

You can run a specific test with (where `-s` avoids capturing the output):

```
poetry run pytest .\tests\test_against_snntorch.py::test_against_snntorch -s
```

## TODOs

Model|Implemented
---|---
Feedforward network with LIF neurons trained with BPTT|x

Benchmark|Implemented
---|---
[N-MNIST](https://www.garrickorchard.com/datasets/n-mnist)|🚧
[ST-MNIST](https://hh-see.com/projects/2_project/)|x
[Isaac-Cartpole-v0](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)|x
[Isaac-Ant-v0](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)|x
[Isaac-Humanoid-v0](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)|x

Architectures|Implemented
---|---
Feed forward|x
Reservoir|x

Synapses|Implemented
---|---
Linear|✓
Convolutional|🚧

Neurons|Implemented
---|---
Leaky Integrate and Fire (LIF)|✓
Adaptive LIF|x
Resonate and fire|x
Quadratic LIF|x
Izhikevich|x
Hodgkin-Huxley|x
Sigma Delta (SDNN)|x

Optimizers|Implemented
---|---
Backpropagation through time (BPTT)|x
SpikeProp|x
SuperSpike|x
SLAYER|x
EventProp|x
Spike Timing Dependant Plasticity (STDP)|x

### Misc

- add GitHub action that runs black, pyright, pytest and pdoc (for pdoc it should also post the github page in the action)
- run `snnTorch` to compare and improve until reaching similar performance for n-mnist and st-mnist.
- output the EFLOPs metrics for models to give an idea of their foundational performance.
- run a test including sparsity in the cost function
- look at [ml_genn](https://github.com/genn-team/ml_genn).
- I went to a conferece and heard about "audoAdjoint" which is intended to be a version of automatic differentiation that works for event driven models and SNNs. Look into this.
- paper from Matias Barandiaran and James Stovold on "Developmental Graph Cellular Automata Can Grow Reservoirs" and lookup info on plastic reservoirs and how these could be implemented/used.