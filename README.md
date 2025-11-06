# spiky

The general idea is to support training a variety of state of the art models by assmebling state of the art techniques then running these against robotics benchmarks to answer experimental questions e.g. "are more bio-plausible models better at embodied learning?".

## Installation

This package is not published yet.

## Development

- Manages dependencies with [Poetry](https://python-poetry.org/).
- Generate docs with `poetry run pdoc -o docs src/spiky`.
- Format with `poetry run black .`
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

1. Install PyTorch within the environment.

   For GPU:

   ```bash
   poetry run pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

   For CPU:

   ```bash
   poetry run pip install torch torchvision
   ```

   etc.

### Testing

You can run a specific test with (where `-s` avoids capturing the output):

```
poetry run pytest .\tests\test_against_snntorch_foreprop.py -s
```

## TODOs

The below table lists the experimental dimensions I am looking at testing across, for practical
concerns I wouldn't be testing the cartesian product of all combinations but I will try and do a
large number while prioritizing the most important combinations.

Benchmark|Synapses|Architectures|Neurons|Optimizers|Hardware|Noise
---|---|---|---|---|---|---
[N-MNIST](https://www.garrickorchard.com/datasets/n-mnist)|Linear|Feed forward|Leaky Integrate and Fire (LIF)|Backpropagation through time (BPTT)|CPU|None
[ST-MNIST](https://hh-see.com/projects/2_project/)|Convolutional|Reservoir|Adaptive LIF|SpikeProp|GPU|Static
[Isaac-Cartpole-v0](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)| | |Resonate and fire|SuperSpike|TPU|Dynamic
[Isaac-Ant-v0](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)| | |Quadratic LIF|SLAYER|FPGA|Adverserial
[Isaac-Humanoid-v0](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)| | |Izhikevich|EventProp|FPAA|
[NeuroBench](https://github.com/NeuroBench/neurobench)| | |Hodgkin-Huxley|Spike Timing Dependant Plasticity (STDP)|memFPAA[^1]|
| | | |Sigma Delta (SDNN)| | |

Vague implementation order plan:
1. Feedforward LIF BPTT CPU
2. Feedforward LIF STDP CPU
3. Reservoir LIF CPU
4. ...

### Noise

- **Static noise** might be a normal distribution
with a set mean and deviation; this might repre-
sent a sensor having a given degree of inaccuracy.
- **Dynamic noise** might be a normal distribution
with a changing mean and deviation; this might
represent a sensor degrading over time, becom-
ing progressively less accurate.
- **Adverserial noise** might be an SNN which
knows the ground truth and slightly changes the
value within some range; this might represent an
intelligent electronic warfare system attempting
to degrade the system.

### Summary

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

An overarching theme guiding my work is "Can SNNs perform online learning to adapt to degradation?".

I'm thinking I will define models and their training declaratively and then implement executors. A LIF SNN feedforward network might be defined with a JSON file, from this we can gather algorithmic metrics like the number of synaptic operations required for inference and training, we can then implement CPU and GPU (and potentially FPGA, and FPAA etc.) executors to measure system metrics like real-time performance. Importantly the algorithmic metrics give us insight into what could be acheived given optimized hardware and where potentially specilized hardware might be useful. An extreme example would be given an ASIC with memristors the latency of a LIF feedforward SNN in real-time would depend on the material properties of the memrestiors, thus we can directly tie the algorithms potential performance to hardware development e.g. "if hardware with X properties exists this algorithm will be N ms faster than the current approach".

### Misc

- re-write to lower level `cuDNN` to avoid `CuPy`s bad documentation (for `cupy.cudnn`) and worse performance.
  The aim should be to acheive similar performance to snnTorch in n-mnist classification.
- add GitHub action that runs black, pyright, pytest and pdoc (for pdoc it should also post the github page in the action)
- run `snnTorch` to compare and improve until reaching similar performance for n-mnist and st-mnist.
- output the EFLOPs metrics for models to give an idea of their foundational performance.
- run a test including sparsity in the cost function
- look at [ml_genn](https://github.com/genn-team/ml_genn).
- I went to a conferece and heard about "audoAdjoint" which is intended to be a version of automatic differentiation that works for event driven models and SNNs. Look into this.
- paper from Matias Barandiaran and James Stovold on "Developmental Graph Cellular Automata Can Grow Reservoirs" and lookup info on plastic reservoirs and how these could be implemented/used.