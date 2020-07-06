import sys
_module = sys.modules[__name__]
del sys
bindsnet = _module
analysis = _module
pipeline_analysis = _module
plotting = _module
visualization = _module
conversion = _module
conversion = _module
datasets = _module
alov300 = _module
collate = _module
dataloader = _module
davis = _module
preprocess = _module
spoken_mnist = _module
torchvision_wrapper = _module
encoding = _module
encoders = _module
encodings = _module
loaders = _module
environment = _module
environment = _module
evaluation = _module
evaluation = _module
learning = _module
learning = _module
reward = _module
models = _module
models = _module
network = _module
monitors = _module
network = _module
nodes = _module
topology = _module
pipeline = _module
action = _module
base_pipeline = _module
dataloader_pipeline = _module
environment_pipeline = _module
preprocessing = _module
preprocessing = _module
utils = _module
conf = _module
annarchy = _module
benchmark = _module
gpu_annarchy = _module
plot_benchmark = _module
breakout = _module
breakout_stdp = _module
play_breakout_from_ANN = _module
random_baseline = _module
random_network_baseline = _module
batch_eth_mnist = _module
conv_mnist = _module
eth_mnist = _module
reservoir = _module
supervised_mnist = _module
tensorboard = _module
setup = _module
test_analyzers = _module
test_conversion = _module
test_encoding = _module
test_import = _module
test_models = _module
test_connections = _module
test_learning = _module
test_monitors = _module
test_network = _module
test_nodes = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
import numpy as np
from torch import Tensor
patch_functional()
open = mock_open()
yaml = logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
yaml.load.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


from abc import ABC


from abc import abstractmethod


from typing import Dict


from typing import Optional


import numpy as np


import torch


from torchvision.utils import make_grid


from torch.nn.modules.utils import _pair


from typing import Tuple


from typing import List


from typing import Sized


from typing import Union


import torch.nn as nn


import torch.nn.functional as F


from copy import deepcopy


from typing import Sequence


from typing import Iterable


import warnings


import time


from torch.utils.data import Dataset


from torch._six import container_abcs


from torch._six import string_classes


from torch._six import int_classes


from torch.utils.data._utils import collate as pytorch_collate


from collections import defaultdict


import math


from torchvision import transforms


from scipy.io import wavfile


import torchvision


from typing import Iterator


from typing import Any


from itertools import product


from sklearn.linear_model import LogisticRegression


from scipy.spatial.distance import euclidean


from torchvision import models


from typing import Type


from functools import reduce


from torch.nn import Module


from torch.nn import Parameter


import itertools


from typing import Callable


from torch import Tensor


from numpy import ndarray


from time import time as t


class Permute(nn.Module):
    """
    PyTorch module for the explicit permutation of a tensor's dimensions in a parent
    module's ``forward`` pass (as opposed to ``torch.permute``).
    """

    def __init__(self, dims):
        """
        Constructor for ``Permute`` module.

        :param dims: Ordering of dimensions for permutation.
        """
        super(Permute, self).__init__()
        self.dims = dims

    def forward(self, x):
        """
        Forward pass of permutation module.

        :param x: Input tensor to permute.
        :return: Permuted input tensor.
        """
        return x.permute(*self.dims).contiguous()


class FeatureExtractor(nn.Module):
    """
    Special-purpose PyTorch module for the extraction of child module's activations.
    """

    def __init__(self, submodule):
        """
        Constructor for ``FeatureExtractor`` module.

        :param submodule: The module who's children modules are to be extracted.
        """
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule

    def forward(self, x: torch.Tensor) ->Dict[nn.Module, torch.Tensor]:
        """
        Forward pass of the feature extractor.

        :param x: Input data for the ``submodule''.
        :return: A dictionary mapping
        """
        activations = {'input': x}
        for name, module in self.submodule._modules.items():
            if isinstance(module, nn.Linear):
                x = x.view(-1, module.in_features)
            x = module(x)
            activations[name] = x
        return activations


class Nodes(torch.nn.Module):
    """
    Abstract base class for groups of neurons.
    """

    def __init__(self, n: Optional[int]=None, shape: Optional[Iterable[int]]=None, traces: bool=False, traces_additive: bool=False, tc_trace: Union[float, torch.Tensor]=20.0, trace_scale: Union[float, torch.Tensor]=1.0, sum_input: bool=False, learning: bool=True, **kwargs) ->None:
        """
        Abstract base class constructor.

        :param n: The number of neurons in the layer.
        :param shape: The dimensionality of the layer.
        :param traces: Whether to record decaying spike traces.
        :param traces_additive: Whether to record spike traces additively.
        :param tc_trace: Time constant of spike trace decay.
        :param trace_scale: Scaling factor for spike trace.
        :param sum_input: Whether to sum all inputs.
        :param learning: Whether to be in learning or testing.
        """
        super().__init__()
        assert n is not None or shape is not None, 'Must provide either no. of neurons or shape of layer'
        if n is None:
            self.n = reduce(mul, shape)
        else:
            self.n = n
        if shape is None:
            self.shape = [self.n]
        else:
            self.shape = shape
        assert self.n == reduce(mul, self.shape), 'No. of neurons and shape do not match'
        self.traces = traces
        self.traces_additive = traces_additive
        self.register_buffer('s', torch.ByteTensor())
        self.sum_input = sum_input
        if self.traces:
            self.register_buffer('x', torch.Tensor())
            self.register_buffer('tc_trace', torch.tensor(tc_trace))
            if self.traces_additive:
                self.register_buffer('trace_scale', torch.tensor(trace_scale))
            self.register_buffer('trace_decay', torch.empty_like(self.tc_trace))
        if self.sum_input:
            self.register_buffer('summed', torch.FloatTensor())
        self.dt = None
        self.batch_size = None
        self.trace_decay = None
        self.learning = learning

    @abstractmethod
    def forward(self, x: torch.Tensor) ->None:
        """
        Abstract base class method for a single simulation step.

        :param x: Inputs to the layer.
        """
        if self.traces:
            self.x *= self.trace_decay
            if self.traces_additive:
                self.x += self.trace_scale * self.s.float()
            else:
                self.x.masked_fill_(self.s != 0, 1)
        if self.sum_input:
            self.summed += x.float()

    def reset_state_variables(self) ->None:
        """
        Abstract base class method for resetting state variables.
        """
        self.s.zero_()
        if self.traces:
            self.x.zero_()
        if self.sum_input:
            self.summed.zero_()

    def compute_decays(self, dt) ->None:
        """
        Abstract base class method for setting decays.
        """
        self.dt = dt
        if self.traces:
            self.trace_decay = torch.exp(-self.dt / self.tc_trace)

    def set_batch_size(self, batch_size) ->None:
        """
        Sets mini-batch size. Called when layer is added to a network.

        :param batch_size: Mini-batch size.
        """
        self.batch_size = batch_size
        self.s = torch.zeros(batch_size, *self.shape, device=self.s.device)
        if self.traces:
            self.x = torch.zeros(batch_size, *self.shape, device=self.x.device)
        if self.sum_input:
            self.summed = torch.zeros(batch_size, *self.shape, device=self.summed.device)

    def train(self, mode: bool=True) ->'Nodes':
        """
        Sets the layer in training mode.

        :param bool mode: Turn training on or off
        :return: self as specified in `torch.nn.Module`
        """
        self.learning = mode
        return super().train(mode)


class AbstractConnection(ABC, Module):
    """
    Abstract base method for connections between ``Nodes``.
    """

    def __init__(self, source: Nodes, target: Nodes, nu: Optional[Union[float, Sequence[float]]]=None, reduction: Optional[callable]=None, weight_decay: float=0.0, **kwargs) ->None:
        """
        Constructor for abstract base class for connection objects.

        :param source: A layer of nodes from which the connection originates.
        :param target: A layer of nodes to which the connection connects.
        :param nu: Learning rate for both pre- and post-synaptic events.
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension.
        :param weight_decay: Constant multiple to decay weights by on each iteration.

        Keyword arguments:

        :param LearningRule update_rule: Modifies connection parameters according to
            some rule.
        :param float wmin: The minimum value on the connection weights.
        :param float wmax: The maximum value on the connection weights.
        :param float norm: Total weight per target neuron normalization.
        """
        super().__init__()
        assert isinstance(source, Nodes), 'Source is not a Nodes object'
        assert isinstance(target, Nodes), 'Target is not a Nodes object'
        self.source = source
        self.target = target
        self.nu = nu
        self.weight_decay = weight_decay
        self.reduction = reduction
        self.update_rule = kwargs.get('update_rule', NoOp)
        self.wmin = kwargs.get('wmin', -np.inf)
        self.wmax = kwargs.get('wmax', np.inf)
        self.norm = kwargs.get('norm', None)
        self.decay = kwargs.get('decay', None)
        if self.update_rule is None:
            self.update_rule = NoOp
        self.update_rule = self.update_rule(connection=self, nu=nu, reduction=reduction, weight_decay=weight_decay, **kwargs)

    @abstractmethod
    def compute(self, s: torch.Tensor) ->None:
        """
        Compute pre-activations of downstream neurons given spikes of upstream neurons.

        :param s: Incoming spikes.
        """
        pass

    @abstractmethod
    def update(self, **kwargs) ->None:
        """
        Compute connection's update rule.

        Keyword arguments:

        :param bool learning: Whether to allow connection updates.
        :param ByteTensor mask: Boolean mask determining which weights to clamp to zero.
        """
        learning = kwargs.get('learning', True)
        if learning:
            self.update_rule.update(**kwargs)
        mask = kwargs.get('mask', None)
        if mask is not None:
            self.w.masked_fill_(mask, 0)

    @abstractmethod
    def reset_state_variables(self) ->None:
        """
        Contains resetting logic for the connection.
        """
        pass


class AbstractMonitor(ABC):
    """
    Abstract base class for state variable monitors.
    """


class AbstractReward(ABC):
    """
    Abstract base class for reward computation.
    """

    @abstractmethod
    def compute(self, **kwargs) ->None:
        """
        Computes/modifies reward.
        """
        pass

    @abstractmethod
    def update(self, **kwargs) ->None:
        """
        Updates internal variables needed to modify reward. Usually called once per
        episode.
        """
        pass


class Network(torch.nn.Module):
    """
    Central object of the ``bindsnet`` package. Responsible for the simulation and
    interaction of nodes and connections.

    **Example:**

    .. code-block:: python

        import torch
        import matplotlib.pyplot as plt

        from bindsnet         import encoding
        from bindsnet.network import Network, nodes, topology, monitors

        network = Network(dt=1.0)  # Instantiates network.

        X = nodes.Input(100)  # Input layer.
        Y = nodes.LIFNodes(100)  # Layer of LIF neurons.
        C = topology.Connection(source=X, target=Y, w=torch.rand(X.n, Y.n))  # Connection from X to Y.

        # Spike monitor objects.
        M1 = monitors.Monitor(obj=X, state_vars=['s'])
        M2 = monitors.Monitor(obj=Y, state_vars=['s'])

        # Add everything to the network object.
        network.add_layer(layer=X, name='X')
        network.add_layer(layer=Y, name='Y')
        network.add_connection(connection=C, source='X', target='Y')
        network.add_monitor(monitor=M1, name='X')
        network.add_monitor(monitor=M2, name='Y')

        # Create Poisson-distributed spike train inputs.
        data = 15 * torch.rand(100)  # Generate random Poisson rates for 100 input neurons.
        train = encoding.poisson(datum=data, time=5000)  # Encode input as 5000ms Poisson spike trains.

        # Simulate network on generated spike trains.
        inputs = {'X' : train}  # Create inputs mapping.
        network.run(inputs=inputs, time=5000)  # Run network simulation.

        # Plot spikes of input and output layers.
        spikes = {'X' : M1.get('s'), 'Y' : M2.get('s')}

        fig, axes = plt.subplots(2, 1, figsize=(12, 7))
        for i, layer in enumerate(spikes):
            axes[i].matshow(spikes[layer], cmap='binary')
            axes[i].set_title('%s spikes' % layer)
            axes[i].set_xlabel('Time'); axes[i].set_ylabel('Index of neuron')
            axes[i].set_xticks(()); axes[i].set_yticks(())
            axes[i].set_aspect('auto')

        plt.tight_layout(); plt.show()
    """

    def __init__(self, dt: float=1.0, batch_size: int=1, learning: bool=True, reward_fn: Optional[Type[AbstractReward]]=None) ->None:
        """
        Initializes network object.

        :param dt: Simulation timestep.
        :param batch_size: Mini-batch size.
        :param learning: Whether to allow connection updates. True by default.
        :param reward_fn: Optional class allowing for modification of reward in case of
            reward-modulated learning.
        """
        super().__init__()
        self.dt = dt
        self.batch_size = batch_size
        self.layers = {}
        self.connections = {}
        self.monitors = {}
        self.train(learning)
        if reward_fn is not None:
            self.reward_fn = reward_fn()
        else:
            self.reward_fn = None

    def add_layer(self, layer: Nodes, name: str) ->None:
        """
        Adds a layer of nodes to the network.

        :param layer: A subclass of the ``Nodes`` object.
        :param name: Logical name of layer.
        """
        self.layers[name] = layer
        self.add_module(name, layer)
        layer.train(self.learning)
        layer.compute_decays(self.dt)
        layer.set_batch_size(self.batch_size)

    def add_connection(self, connection: AbstractConnection, source: str, target: str) ->None:
        """
        Adds a connection between layers of nodes to the network.

        :param connection: An instance of class ``Connection``.
        :param source: Logical name of the connection's source layer.
        :param target: Logical name of the connection's target layer.
        """
        self.connections[source, target] = connection
        self.add_module(source + '_to_' + target, connection)
        connection.dt = self.dt
        connection.train(self.learning)

    def add_monitor(self, monitor: AbstractMonitor, name: str) ->None:
        """
        Adds a monitor on a network object to the network.

        :param monitor: An instance of class ``Monitor``.
        :param name: Logical name of monitor object.
        """
        self.monitors[name] = monitor
        monitor.network = self
        monitor.dt = self.dt

    def save(self, file_name: str) ->None:
        """
        Serializes the network object to disk.

        :param file_name: Path to store serialized network object on disk.

        **Example:**

        .. code-block:: python

            import torch
            import matplotlib.pyplot as plt

            from pathlib          import Path
            from bindsnet.network import *
            from bindsnet.network import topology

            # Build simple network.
            network = Network(dt=1.0)

            X = nodes.Input(100)  # Input layer.
            Y = nodes.LIFNodes(100)  # Layer of LIF neurons.
            C = topology.Connection(source=X, target=Y, w=torch.rand(X.n, Y.n))  # Connection from X to Y.

            # Add everything to the network object.
            network.add_layer(layer=X, name='X')
            network.add_layer(layer=Y, name='Y')
            network.add_connection(connection=C, source='X', target='Y')

            # Save the network to disk.
            network.save(str(Path.home()) + '/network.pt')
        """
        torch.save(self, open(file_name, 'wb'))

    def clone(self) ->'Network':
        """
        Returns a cloned network object.
        
        :return: A copy of this network.
        """
        virtual_file = tempfile.SpooledTemporaryFile()
        torch.save(self, virtual_file)
        virtual_file.seek(0)
        return torch.load(virtual_file)

    def _get_inputs(self, layers: Iterable=None) ->Dict[str, torch.Tensor]:
        """
        Fetches outputs from network layers to use as input to downstream layers.

        :param layers: Layers to update inputs for. Defaults to all network layers.
        :return: Inputs to all layers for the current iteration.
        """
        inputs = {}
        if layers is None:
            layers = self.layers
        for c in self.connections:
            if c[1] in layers:
                source = self.connections[c].source
                target = self.connections[c].target
                if not c[1] in inputs:
                    inputs[c[1]] = torch.zeros(self.batch_size, *target.shape, device=target.s.device)
                inputs[c[1]] += self.connections[c].compute(source.s)
        return inputs

    def run(self, inputs: Dict[str, torch.Tensor], time: int, one_step=False, **kwargs) ->None:
        """
        Simulate network for given inputs and time.

        :param inputs: Dictionary of ``Tensor``s of shape ``[time, *input_shape]`` or
                      ``[time, batch_size, *input_shape]``.
        :param time: Simulation time.
        :param one_step: Whether to run the network in "feed-forward" mode, where inputs
            propagate all the way through the network in a single simulation time step.
            Layers are updated in the order they are added to the network.

        Keyword arguments:

        :param Dict[str, torch.Tensor] clamp: Mapping of layer names to boolean masks if
            neurons should be clamped to spiking. The ``Tensor``s have shape
            ``[n_neurons]`` or ``[time, n_neurons]``.
        :param Dict[str, torch.Tensor] unclamp: Mapping of layer names to boolean masks
            if neurons should be clamped to not spiking. The ``Tensor``s should have
            shape ``[n_neurons]`` or ``[time, n_neurons]``.
        :param Dict[str, torch.Tensor] injects_v: Mapping of layer names to boolean
            masks if neurons should be added voltage. The ``Tensor``s should have shape
            ``[n_neurons]`` or ``[time, n_neurons]``.
        :param Union[float, torch.Tensor] reward: Scalar value used in reward-modulated
            learning.
        :param Dict[Tuple[str], torch.Tensor] masks: Mapping of connection names to
            boolean masks determining which weights to clamp to zero.

        **Example:**

        .. code-block:: python

            import torch
            import matplotlib.pyplot as plt

            from bindsnet.network import Network
            from bindsnet.network.nodes import Input
            from bindsnet.network.monitors import Monitor

            # Build simple network.
            network = Network()
            network.add_layer(Input(500), name='I')
            network.add_monitor(Monitor(network.layers['I'], state_vars=['s']), 'I')

            # Generate spikes by running Bernoulli trials on Uniform(0, 0.5) samples.
            spikes = torch.bernoulli(0.5 * torch.rand(500, 500))

            # Run network simulation.
            network.run(inputs={'I' : spikes}, time=500)

            # Look at input spiking activity.
            spikes = network.monitors['I'].get('s')
            plt.matshow(spikes, cmap='binary')
            plt.xticks(()); plt.yticks(());
            plt.xlabel('Time'); plt.ylabel('Neuron index')
            plt.title('Input spiking')
            plt.show()
        """
        clamps = kwargs.get('clamp', {})
        unclamps = kwargs.get('unclamp', {})
        masks = kwargs.get('masks', {})
        injects_v = kwargs.get('injects_v', {})
        if self.reward_fn is not None:
            kwargs['reward'] = self.reward_fn.compute(**kwargs)
        if inputs != {}:
            for key in inputs:
                if len(inputs[key].size()) == 1:
                    inputs[key] = inputs[key].unsqueeze(0).unsqueeze(0)
                elif len(inputs[key].size()) == 2:
                    inputs[key] = inputs[key].unsqueeze(1)
            for key in inputs:
                if inputs[key].size(1) != self.batch_size:
                    self.batch_size = inputs[key].size(1)
                    for l in self.layers:
                        self.layers[l].set_batch_size(self.batch_size)
                    for m in self.monitors:
                        self.monitors[m].reset_state_variables()
                break
        timesteps = int(time / self.dt)
        for t in range(timesteps):
            current_inputs = {}
            if not one_step:
                current_inputs.update(self._get_inputs())
            for l in self.layers:
                if l in inputs:
                    if l in current_inputs:
                        current_inputs[l] += inputs[l][t]
                    else:
                        current_inputs[l] = inputs[l][t]
                if one_step:
                    current_inputs.update(self._get_inputs(layers=[l]))
                self.layers[l].forward(x=current_inputs[l])
                clamp = clamps.get(l, None)
                if clamp is not None:
                    if clamp.ndimension() == 1:
                        self.layers[l].s[:, (clamp)] = 1
                    else:
                        self.layers[l].s[:, (clamp[t])] = 1
                unclamp = unclamps.get(l, None)
                if unclamp is not None:
                    if unclamp.ndimension() == 1:
                        self.layers[l].s[unclamp] = 0
                    else:
                        self.layers[l].s[unclamp[t]] = 0
                inject_v = injects_v.get(l, None)
                if inject_v is not None:
                    if inject_v.ndimension() == 1:
                        self.layers[l].v += inject_v
                    else:
                        self.layers[l].v += inject_v[t]
            for c in self.connections:
                self.connections[c].update(mask=masks.get(c, None), learning=self.learning, **kwargs)
            current_inputs.update(self._get_inputs())
            for m in self.monitors:
                self.monitors[m].record()
        for c in self.connections:
            self.connections[c].normalize()

    def reset_state_variables(self) ->None:
        """
        Reset state variables of objects in network.
        """
        for layer in self.layers:
            self.layers[layer].reset_state_variables()
        for connection in self.connections:
            self.connections[connection].reset_state_variables()
        for monitor in self.monitors:
            self.monitors[monitor].reset_state_variables()

    def train(self, mode: bool=True) ->'torch.nn.Module':
        """
        Sets the node in training mode.

        :param mode: Turn training on or off.

        :return: ``self`` as specified in ``torch.nn.Module``.
        """
        self.learning = mode
        return super().train(mode)


class AbstractInput(ABC):
    """
    Abstract base class for groups of input neurons.
    """


class Input(Nodes, AbstractInput):
    """
    Layer of nodes with user-specified spiking behavior.
    """

    def __init__(self, n: Optional[int]=None, shape: Optional[Iterable[int]]=None, traces: bool=False, traces_additive: bool=False, tc_trace: Union[float, torch.Tensor]=20.0, trace_scale: Union[float, torch.Tensor]=1.0, sum_input: bool=False, **kwargs) ->None:
        """
        Instantiates a layer of input neurons.

        :param n: The number of neurons in the layer.
        :param shape: The dimensionality of the layer.
        :param traces: Whether to record decaying spike traces.
        :param traces_additive: Whether to record spike traces additively.
        :param tc_trace: Time constant of spike trace decay.
        :param trace_scale: Scaling factor for spike trace.
        :param sum_input: Whether to sum all inputs.
        """
        super().__init__(n=n, shape=shape, traces=traces, traces_additive=traces_additive, tc_trace=tc_trace, trace_scale=trace_scale, sum_input=sum_input)

    def forward(self, x: torch.Tensor) ->None:
        """
        On each simulation step, set the spikes of the population equal to the inputs.

        :param x: Inputs to the layer.
        """
        self.s = x
        super().forward(x)

    def reset_state_variables(self) ->None:
        """
        Resets relevant state variables.
        """
        super().reset_state_variables()


class McCullochPitts(Nodes):
    """
    Layer of `McCulloch-Pitts neurons
    <http://wwwold.ece.utep.edu/research/webfuzzy/docs/kk-thesis/kk-thesis-html/node12.html>`_.
    """

    def __init__(self, n: Optional[int]=None, shape: Optional[Iterable[int]]=None, traces: bool=False, traces_additive: bool=False, tc_trace: Union[float, torch.Tensor]=20.0, trace_scale: Union[float, torch.Tensor]=1.0, sum_input: bool=False, thresh: Union[float, torch.Tensor]=1.0, **kwargs) ->None:
        """
        Instantiates a McCulloch-Pitts layer of neurons.

        :param n: The number of neurons in the layer.
        :param shape: The dimensionality of the layer.
        :param traces: Whether to record spike traces.
        :param traces_additive: Whether to record spike traces additively.
        :param tc_trace: Time constant of spike trace decay.
        :param trace_scale: Scaling factor for spike trace.
        :param sum_input: Whether to sum all inputs.
        :param thresh: Spike threshold voltage.
        """
        super().__init__(n=n, shape=shape, traces=traces, traces_additive=traces_additive, tc_trace=tc_trace, trace_scale=trace_scale, sum_input=sum_input)
        self.register_buffer('thresh', torch.tensor(thresh, dtype=torch.float))
        self.register_buffer('v', torch.FloatTensor())

    def forward(self, x: torch.Tensor) ->None:
        """
        Runs a single simulation step.

        :param x: Inputs to the layer.
        """
        self.v = x
        self.s = self.v >= self.thresh
        super().forward(x)

    def reset_state_variables(self) ->None:
        """
        Resets relevant state variables.
        """
        super().reset_state_variables()

    def set_batch_size(self, batch_size) ->None:
        """
        Sets mini-batch size. Called when layer is added to a network.

        :param batch_size: Mini-batch size.
        """
        super().set_batch_size(batch_size=batch_size)
        self.v = torch.zeros(batch_size, *self.shape, device=self.v.device)


class IFNodes(Nodes):
    """
    Layer of `integrate-and-fire (IF) neurons <http://neuronaldynamics.epfl.ch/online/Ch1.S3.html>`_.
    """

    def __init__(self, n: Optional[int]=None, shape: Optional[Iterable[int]]=None, traces: bool=False, traces_additive: bool=False, tc_trace: Union[float, torch.Tensor]=20.0, trace_scale: Union[float, torch.Tensor]=1.0, sum_input: bool=False, thresh: Union[float, torch.Tensor]=-52.0, reset: Union[float, torch.Tensor]=-65.0, refrac: Union[int, torch.Tensor]=5, lbound: float=None, **kwargs) ->None:
        """
        Instantiates a layer of IF neurons.

        :param n: The number of neurons in the layer.
        :param shape: The dimensionality of the layer.
        :param traces: Whether to record spike traces.
        :param traces_additive: Whether to record spike traces additively.
        :param tc_trace: Time constant of spike trace decay.
        :param trace_scale: Scaling factor for spike trace.
        :param sum_input: Whether to sum all inputs.
        :param thresh: Spike threshold voltage.
        :param reset: Post-spike reset voltage.
        :param refrac: Refractory (non-firing) period of the neuron.
        :param lbound: Lower bound of the voltage.
        """
        super().__init__(n=n, shape=shape, traces=traces, traces_additive=traces_additive, tc_trace=tc_trace, trace_scale=trace_scale, sum_input=sum_input)
        self.register_buffer('reset', torch.tensor(reset, dtype=torch.float))
        self.register_buffer('thresh', torch.tensor(thresh, dtype=torch.float))
        self.register_buffer('refrac', torch.tensor(refrac))
        self.register_buffer('v', torch.FloatTensor())
        self.register_buffer('refrac_count', torch.FloatTensor())
        self.lbound = lbound

    def forward(self, x: torch.Tensor) ->None:
        """
        Runs a single simulation step.

        :param x: Inputs to the layer.
        """
        self.v += (self.refrac_count == 0).float() * x
        self.refrac_count = (self.refrac_count > 0).float() * (self.refrac_count - self.dt)
        self.s = self.v >= self.thresh
        self.refrac_count.masked_fill_(self.s, self.refrac)
        self.v.masked_fill_(self.s, self.reset)
        if self.lbound is not None:
            self.v.masked_fill_(self.v < self.lbound, self.lbound)
        super().forward(x)

    def reset_state_variables(self) ->None:
        """
        Resets relevant state variables.
        """
        super().reset_state_variables()
        self.v.fill_(self.reset)
        self.refrac_count.zero_()

    def set_batch_size(self, batch_size) ->None:
        """
        Sets mini-batch size. Called when layer is added to a network.

        :param batch_size: Mini-batch size.
        """
        super().set_batch_size(batch_size=batch_size)
        self.v = self.reset * torch.ones(batch_size, *self.shape, device=self.v.device)
        self.refrac_count = torch.zeros_like(self.v, device=self.refrac_count.device)


class LIFNodes(Nodes):
    """
    Layer of `leaky integrate-and-fire (LIF) neurons
    <http://icwww.epfl.ch/~gerstner/SPNM/node26.html#SECTION02311000000000000000>`_.
    """

    def __init__(self, n: Optional[int]=None, shape: Optional[Iterable[int]]=None, traces: bool=False, traces_additive: bool=False, tc_trace: Union[float, torch.Tensor]=20.0, trace_scale: Union[float, torch.Tensor]=1.0, sum_input: bool=False, thresh: Union[float, torch.Tensor]=-52.0, rest: Union[float, torch.Tensor]=-65.0, reset: Union[float, torch.Tensor]=-65.0, refrac: Union[int, torch.Tensor]=5, tc_decay: Union[float, torch.Tensor]=100.0, lbound: float=None, **kwargs) ->None:
        """
        Instantiates a layer of LIF neurons.

        :param n: The number of neurons in the layer.
        :param shape: The dimensionality of the layer.
        :param traces: Whether to record spike traces.
        :param traces_additive: Whether to record spike traces additively.
        :param tc_trace: Time constant of spike trace decay.
        :param trace_scale: Scaling factor for spike trace.
        :param sum_input: Whether to sum all inputs.
        :param thresh: Spike threshold voltage.
        :param rest: Resting membrane voltage.
        :param reset: Post-spike reset voltage.
        :param refrac: Refractory (non-firing) period of the neuron.
        :param tc_decay: Time constant of neuron voltage decay.
        :param lbound: Lower bound of the voltage.
        """
        super().__init__(n=n, shape=shape, traces=traces, traces_additive=traces_additive, tc_trace=tc_trace, trace_scale=trace_scale, sum_input=sum_input)
        self.register_buffer('rest', torch.tensor(rest, dtype=torch.float))
        self.register_buffer('reset', torch.tensor(reset, dtype=torch.float))
        self.register_buffer('thresh', torch.tensor(thresh, dtype=torch.float))
        self.register_buffer('refrac', torch.tensor(refrac))
        self.register_buffer('tc_decay', torch.tensor(tc_decay, dtype=torch.float))
        self.register_buffer('decay', torch.zeros(*self.shape))
        self.register_buffer('v', torch.FloatTensor())
        self.register_buffer('refrac_count', torch.FloatTensor())
        if lbound is None:
            self.lbound = None
        else:
            self.lbound = torch.tensor(lbound, dtype=torch.float)

    def forward(self, x: torch.Tensor) ->None:
        """
        Runs a single simulation step.

        :param x: Inputs to the layer.
        """
        self.v = self.decay * (self.v - self.rest) + self.rest
        self.v += (self.refrac_count == 0).float() * x
        self.refrac_count = (self.refrac_count > 0).float() * (self.refrac_count - self.dt)
        self.s = self.v >= self.thresh
        self.refrac_count.masked_fill_(self.s, self.refrac)
        self.v.masked_fill_(self.s, self.reset)
        if self.lbound is not None:
            self.v.masked_fill_(self.v < self.lbound, self.lbound)
        super().forward(x)

    def reset_state_variables(self) ->None:
        """
        Resets relevant state variables.
        """
        super().reset_state_variables()
        self.v.fill_(self.rest)
        self.refrac_count.zero_()

    def compute_decays(self, dt) ->None:
        """
        Sets the relevant decays.
        """
        super().compute_decays(dt=dt)
        self.decay = torch.exp(-self.dt / self.tc_decay)

    def set_batch_size(self, batch_size) ->None:
        """
        Sets mini-batch size. Called when layer is added to a network.

        :param batch_size: Mini-batch size.
        """
        super().set_batch_size(batch_size=batch_size)
        self.v = self.rest * torch.ones(batch_size, *self.shape, device=self.v.device)
        self.refrac_count = torch.zeros_like(self.v, device=self.refrac_count.device)


class CurrentLIFNodes(Nodes):
    """
    Layer of `current-based leaky integrate-and-fire (LIF) neurons
    <http://icwww.epfl.ch/~gerstner/SPNM/node26.html#SECTION02313000000000000000>`_.
    Total synaptic input current is modeled as a decaying memory of input spikes multiplied by synaptic strengths.
    """

    def __init__(self, n: Optional[int]=None, shape: Optional[Iterable[int]]=None, traces: bool=False, traces_additive: bool=False, tc_trace: Union[float, torch.Tensor]=20.0, trace_scale: Union[float, torch.Tensor]=1.0, sum_input: bool=False, thresh: Union[float, torch.Tensor]=-52.0, rest: Union[float, torch.Tensor]=-65.0, reset: Union[float, torch.Tensor]=-65.0, refrac: Union[int, torch.Tensor]=5, tc_decay: Union[float, torch.Tensor]=100.0, tc_i_decay: Union[float, torch.Tensor]=2.0, lbound: float=None, **kwargs) ->None:
        """
        Instantiates a layer of synaptic input current-based LIF neurons.
        :param n: The number of neurons in the layer.
        :param shape: The dimensionality of the layer.
        :param traces: Whether to record spike traces.
        :param traces_additive: Whether to record spike traces additively.
        :param tc_trace: Time constant of spike trace decay.
        :param trace_scale: Scaling factor for spike trace.
        :param sum_input: Whether to sum all inputs.
        :param thresh: Spike threshold voltage.
        :param rest: Resting membrane voltage.
        :param reset: Post-spike reset voltage.
        :param refrac: Refractory (non-firing) period of the neuron.
        :param tc_decay: Time constant of neuron voltage decay.
        :param tc_i_decay: Time constant of synaptic input current decay.
        :param lbound: Lower bound of the voltage.
        """
        super().__init__(n=n, shape=shape, traces=traces, traces_additive=traces_additive, tc_trace=tc_trace, trace_scale=trace_scale, sum_input=sum_input)
        self.register_buffer('rest', torch.tensor(rest))
        self.register_buffer('reset', torch.tensor(reset))
        self.register_buffer('thresh', torch.tensor(thresh))
        self.register_buffer('refrac', torch.tensor(refrac))
        self.register_buffer('tc_decay', torch.tensor(tc_decay))
        self.register_buffer('decay', torch.empty_like(self.tc_decay))
        self.register_buffer('tc_i_decay', torch.tensor(tc_i_decay))
        self.register_buffer('i_decay', torch.empty_like(self.tc_i_decay))
        self.register_buffer('v', torch.FloatTensor())
        self.register_buffer('i', torch.FloatTensor())
        self.register_buffer('refrac_count', torch.FloatTensor())
        self.lbound = lbound

    def forward(self, x: torch.Tensor) ->None:
        """
        Runs a single simulation step.

        :param x: Inputs to the layer.
        """
        self.v = self.decay * (self.v - self.rest) + self.rest
        self.i *= self.i_decay
        self.refrac_count = (self.refrac_count > 0).float() * (self.refrac_count - self.dt)
        self.i += x
        self.v += (self.refrac_count == 0).float() * self.i
        self.s = self.v >= self.thresh
        self.refrac_count.masked_fill_(self.s, self.refrac)
        self.v.masked_fill_(self.s, self.reset)
        if self.lbound is not None:
            self.v.masked_fill_(self.v < self.lbound, self.lbound)
        super().forward(x)

    def reset_state_variables(self) ->None:
        """
        Resets relevant state variables.
        """
        super().reset_state_variables()
        self.v.fill_(self.rest)
        self.i.zero_()
        self.refrac_count.zero_()

    def compute_decays(self, dt) ->None:
        """
        Sets the relevant decays.
        """
        super().compute_decays(dt=dt)
        self.decay = torch.exp(-self.dt / self.tc_decay)
        self.i_decay = torch.exp(-self.dt / self.tc_i_decay)

    def set_batch_size(self, batch_size) ->None:
        """
        Sets mini-batch size. Called when layer is added to a network.

        :param batch_size: Mini-batch size.
        """
        super().set_batch_size(batch_size=batch_size)
        self.v = self.rest * torch.ones(batch_size, *self.shape, device=self.v.device)
        self.i = torch.zeros_like(self.v, device=self.i.device)
        self.refrac_count = torch.zeros_like(self.v, device=self.refrac_count.device)


class AdaptiveLIFNodes(Nodes):
    """
    Layer of leaky integrate-and-fire (LIF) neurons with adaptive thresholds. A neuron's voltage threshold is increased
    by some constant each time it spikes; otherwise, it is decaying back to its default value.
    """

    def __init__(self, n: Optional[int]=None, shape: Optional[Iterable[int]]=None, traces: bool=False, traces_additive: bool=False, tc_trace: Union[float, torch.Tensor]=20.0, trace_scale: Union[float, torch.Tensor]=1.0, sum_input: bool=False, rest: Union[float, torch.Tensor]=-65.0, reset: Union[float, torch.Tensor]=-65.0, thresh: Union[float, torch.Tensor]=-52.0, refrac: Union[int, torch.Tensor]=5, tc_decay: Union[float, torch.Tensor]=100.0, theta_plus: Union[float, torch.Tensor]=0.05, tc_theta_decay: Union[float, torch.Tensor]=10000000.0, lbound: float=None, **kwargs) ->None:
        """
        Instantiates a layer of LIF neurons with adaptive firing thresholds.

        :param n: The number of neurons in the layer.
        :param shape: The dimensionality of the layer.
        :param traces: Whether to record spike traces.
        :param traces_additive: Whether to record spike traces additively.
        :param tc_trace: Time constant of spike trace decay.
        :param trace_scale: Scaling factor for spike trace.
        :param sum_input: Whether to sum all inputs.
        :param rest: Resting membrane voltage.
        :param reset: Post-spike reset voltage.
        :param thresh: Spike threshold voltage.
        :param refrac: Refractory (non-firing) period of the neuron.
        :param tc_decay: Time constant of neuron voltage decay.
        :param theta_plus: Voltage increase of threshold after spiking.
        :param tc_theta_decay: Time constant of adaptive threshold decay.
        :param lbound: Lower bound of the voltage.
        """
        super().__init__(n=n, shape=shape, traces=traces, traces_additive=traces_additive, tc_trace=tc_trace, trace_scale=trace_scale, sum_input=sum_input)
        self.register_buffer('rest', torch.tensor(rest))
        self.register_buffer('reset', torch.tensor(reset))
        self.register_buffer('thresh', torch.tensor(thresh))
        self.register_buffer('refrac', torch.tensor(refrac))
        self.register_buffer('tc_decay', torch.tensor(tc_decay))
        self.register_buffer('decay', torch.empty_like(self.tc_decay))
        self.register_buffer('theta_plus', torch.tensor(theta_plus))
        self.register_buffer('tc_theta_decay', torch.tensor(tc_theta_decay))
        self.register_buffer('theta_decay', torch.empty_like(self.tc_theta_decay))
        self.register_buffer('v', torch.FloatTensor())
        self.register_buffer('theta', torch.zeros(*self.shape))
        self.register_buffer('refrac_count', torch.FloatTensor())
        self.lbound = lbound

    def forward(self, x: torch.Tensor) ->None:
        """
        Runs a single simulation step.

        :param x: Inputs to the layer.
        """
        self.v = self.decay * (self.v - self.rest) + self.rest
        if self.learning:
            self.theta *= self.theta_decay
        self.v += (self.refrac_count == 0).float() * x
        self.refrac_count = (self.refrac_count > 0).float() * (self.refrac_count - self.dt)
        self.s = self.v >= self.thresh + self.theta
        self.refrac_count.masked_fill_(self.s, self.refrac)
        self.v.masked_fill_(self.s, self.reset)
        if self.learning:
            self.theta += self.theta_plus * self.s.float().sum(0)
        if self.lbound is not None:
            self.v.masked_fill_(self.v < self.lbound, self.lbound)
        super().forward(x)

    def reset_state_variables(self) ->None:
        """
        Resets relevant state variables.
        """
        super().reset_state_variables()
        self.v.fill_(self.rest)
        self.refrac_count.zero_()

    def compute_decays(self, dt) ->None:
        """
        Sets the relevant decays.
        """
        super().compute_decays(dt=dt)
        self.decay = torch.exp(-self.dt / self.tc_decay)
        self.theta_decay = torch.exp(-self.dt / self.tc_theta_decay)

    def set_batch_size(self, batch_size) ->None:
        """
        Sets mini-batch size. Called when layer is added to a network.

        :param batch_size: Mini-batch size.
        """
        super().set_batch_size(batch_size=batch_size)
        self.v = self.rest * torch.ones(batch_size, *self.shape, device=self.v.device)
        self.refrac_count = torch.zeros_like(self.v, device=self.refrac_count.device)


class DiehlAndCookNodes(Nodes):
    """
    Layer of leaky integrate-and-fire (LIF) neurons with adaptive thresholds (modified for Diehl & Cook 2015
    replication).
    """

    def __init__(self, n: Optional[int]=None, shape: Optional[Iterable[int]]=None, traces: bool=False, traces_additive: bool=False, tc_trace: Union[float, torch.Tensor]=20.0, trace_scale: Union[float, torch.Tensor]=1.0, sum_input: bool=False, thresh: Union[float, torch.Tensor]=-52.0, rest: Union[float, torch.Tensor]=-65.0, reset: Union[float, torch.Tensor]=-65.0, refrac: Union[int, torch.Tensor]=5, tc_decay: Union[float, torch.Tensor]=100.0, theta_plus: Union[float, torch.Tensor]=0.05, tc_theta_decay: Union[float, torch.Tensor]=10000000.0, lbound: float=None, one_spike: bool=True, **kwargs) ->None:
        """
        Instantiates a layer of Diehl & Cook 2015 neurons.

        :param n: The number of neurons in the layer.
        :param shape: The dimensionality of the layer.
        :param traces: Whether to record spike traces.
        :param traces_additive: Whether to record spike traces additively.
        :param tc_trace: Time constant of spike trace decay.
        :param trace_scale: Scaling factor for spike trace.
        :param sum_input: Whether to sum all inputs.
        :param thresh: Spike threshold voltage.
        :param rest: Resting membrane voltage.
        :param reset: Post-spike reset voltage.
        :param refrac: Refractory (non-firing) period of the neuron.
        :param tc_decay: Time constant of neuron voltage decay.
        :param theta_plus: Voltage increase of threshold after spiking.
        :param tc_theta_decay: Time constant of adaptive threshold decay.
        :param lbound: Lower bound of the voltage.
        :param one_spike: Whether to allow only one spike per timestep.
        """
        super().__init__(n=n, shape=shape, traces=traces, traces_additive=traces_additive, tc_trace=tc_trace, trace_scale=trace_scale, sum_input=sum_input)
        self.register_buffer('rest', torch.tensor(rest))
        self.register_buffer('reset', torch.tensor(reset))
        self.register_buffer('thresh', torch.tensor(thresh))
        self.register_buffer('refrac', torch.tensor(refrac))
        self.register_buffer('tc_decay', torch.tensor(tc_decay))
        self.register_buffer('decay', torch.empty_like(self.tc_decay))
        self.register_buffer('theta_plus', torch.tensor(theta_plus))
        self.register_buffer('tc_theta_decay', torch.tensor(tc_theta_decay))
        self.register_buffer('theta_decay', torch.empty_like(self.tc_theta_decay))
        self.register_buffer('v', torch.FloatTensor())
        self.register_buffer('theta', torch.zeros(*self.shape))
        self.register_buffer('refrac_count', torch.FloatTensor())
        self.lbound = lbound
        self.one_spike = one_spike

    def forward(self, x: torch.Tensor) ->None:
        """
        Runs a single simulation step.

        :param x: Inputs to the layer.
        """
        self.v = self.decay * (self.v - self.rest) + self.rest
        if self.learning:
            self.theta *= self.theta_decay
        self.v += (self.refrac_count == 0).float() * x
        self.refrac_count = (self.refrac_count > 0).float() * (self.refrac_count - self.dt)
        self.s = self.v >= self.thresh + self.theta
        self.refrac_count.masked_fill_(self.s, self.refrac)
        self.v.masked_fill_(self.s, self.reset)
        if self.learning:
            self.theta += self.theta_plus * self.s.float().sum(0)
        if self.one_spike:
            if self.s.any():
                _any = self.s.view(self.batch_size, -1).any(1)
                ind = torch.multinomial(self.s.float().view(self.batch_size, -1)[_any], 1)
                _any = _any.nonzero()
                self.s.zero_()
                self.s.view(self.batch_size, -1)[_any, ind] = 1
        if self.lbound is not None:
            self.v.masked_fill_(self.v < self.lbound, self.lbound)
        super().forward(x)

    def reset_state_variables(self) ->None:
        """
        Resets relevant state variables.
        """
        super().reset_state_variables()
        self.v.fill_(self.rest)
        self.refrac_count.zero_()

    def compute_decays(self, dt) ->None:
        """
        Sets the relevant decays.
        """
        super().compute_decays(dt=dt)
        self.decay = torch.exp(-self.dt / self.tc_decay)
        self.theta_decay = torch.exp(-self.dt / self.tc_theta_decay)

    def set_batch_size(self, batch_size) ->None:
        """
        Sets mini-batch size. Called when layer is added to a network.

        :param batch_size: Mini-batch size.
        """
        super().set_batch_size(batch_size=batch_size)
        self.v = self.rest * torch.ones(batch_size, *self.shape, device=self.v.device)
        self.refrac_count = torch.zeros_like(self.v, device=self.refrac_count.device)


class IzhikevichNodes(Nodes):
    """
    Layer of Izhikevich neurons.
    """

    def __init__(self, n: Optional[int]=None, shape: Optional[Iterable[int]]=None, traces: bool=False, traces_additive: bool=False, tc_trace: Union[float, torch.Tensor]=20.0, trace_scale: Union[float, torch.Tensor]=1.0, sum_input: bool=False, excitatory: float=1, thresh: Union[float, torch.Tensor]=45.0, rest: Union[float, torch.Tensor]=-65.0, lbound: float=None, **kwargs) ->None:
        """
        Instantiates a layer of Izhikevich neurons.

        :param n: The number of neurons in the layer.
        :param shape: The dimensionality of the layer.
        :param traces: Whether to record spike traces.
        :param traces_additive: Whether to record spike traces additively.
        :param tc_trace: Time constant of spike trace decay.
        :param trace_scale: Scaling factor for spike trace.
        :param sum_input: Whether to sum all inputs.
        :param excitatory: Percent of excitatory (vs. inhibitory) neurons in the layer; in range ``[0, 1]``.
        :param thresh: Spike threshold voltage.
        :param rest: Resting membrane voltage.
        :param lbound: Lower bound of the voltage.
        """
        super().__init__(n=n, shape=shape, traces=traces, traces_additive=traces_additive, tc_trace=tc_trace, trace_scale=trace_scale, sum_input=sum_input)
        self.register_buffer('rest', torch.tensor(rest))
        self.register_buffer('thresh', torch.tensor(thresh))
        self.lbound = lbound
        self.register_buffer('r', None)
        self.register_buffer('a', None)
        self.register_buffer('b', None)
        self.register_buffer('c', None)
        self.register_buffer('d', None)
        self.register_buffer('S', None)
        self.register_buffer('excitatory', None)
        if excitatory > 1:
            excitatory = 1
        elif excitatory < 0:
            excitatory = 0
        if excitatory == 1:
            self.r = torch.rand(n)
            self.a = 0.02 * torch.ones(n)
            self.b = 0.2 * torch.ones(n)
            self.c = -65.0 + 15 * self.r ** 2
            self.d = 8 - 6 * self.r ** 2
            self.S = 0.5 * torch.rand(n, n)
            self.excitatory = torch.ones(n).byte()
        elif excitatory == 0:
            self.r = torch.rand(n)
            self.a = 0.02 + 0.08 * self.r
            self.b = 0.25 - 0.05 * self.r
            self.c = -65.0 * torch.ones(n)
            self.d = 2 * torch.ones(n)
            self.S = -torch.rand(n, n)
            self.excitatory = torch.zeros(n).byte()
        else:
            self.excitatory = torch.zeros(n).byte()
            ex = int(n * excitatory)
            inh = n - ex
            self.r = torch.zeros(n)
            self.a = torch.zeros(n)
            self.b = torch.zeros(n)
            self.c = torch.zeros(n)
            self.d = torch.zeros(n)
            self.S = torch.zeros(n, n)
            self.r[:ex] = torch.rand(ex)
            self.a[:ex] = 0.02 * torch.ones(ex)
            self.b[:ex] = 0.2 * torch.ones(ex)
            self.c[:ex] = -65.0 + 15 * self.r[:ex] ** 2
            self.d[:ex] = 8 - 6 * self.r[:ex] ** 2
            self.S[:, :ex] = 0.5 * torch.rand(n, ex)
            self.excitatory[:ex] = 1
            self.r[ex:] = torch.rand(inh)
            self.a[ex:] = 0.02 + 0.08 * self.r[ex:]
            self.b[ex:] = 0.25 - 0.05 * self.r[ex:]
            self.c[ex:] = -65.0 * torch.ones(inh)
            self.d[ex:] = 2 * torch.ones(inh)
            self.S[:, ex:] = -torch.rand(n, inh)
            self.excitatory[ex:] = 0
        self.register_buffer('v', self.rest * torch.ones(n))
        self.register_buffer('u', self.b * self.v)

    def forward(self, x: torch.Tensor) ->None:
        """
        Runs a single simulation step.

        :param x: Inputs to the layer.
        """
        self.s = self.v >= self.thresh
        self.v = torch.where(self.s, self.c, self.v)
        self.u = torch.where(self.s, self.u + self.d, self.u)
        if self.s.any():
            x += torch.cat([self.S[:, (self.s[i])].sum(dim=1)[None] for i in range(self.s.shape[0])], dim=0)
        self.v += self.dt * 0.5 * (0.04 * self.v ** 2 + 5 * self.v + 140 - self.u + x)
        self.v += self.dt * 0.5 * (0.04 * self.v ** 2 + 5 * self.v + 140 - self.u + x)
        self.u += self.dt * self.a * (self.b * self.v - self.u)
        if self.lbound is not None:
            self.v.masked_fill_(self.v < self.lbound, self.lbound)
        super().forward(x)

    def reset_state_variables(self) ->None:
        """
        Resets relevant state variables.
        """
        super().reset_state_variables()
        self.v.fill_(self.rest)
        self.u = self.b * self.v

    def set_batch_size(self, batch_size) ->None:
        """
        Sets mini-batch size. Called when layer is added to a network.

        :param batch_size: Mini-batch size.
        """
        super().set_batch_size(batch_size=batch_size)
        self.v = self.rest * torch.ones(batch_size, *self.shape, device=self.v.device)
        self.u = self.b * self.v


class SRM0Nodes(Nodes):
    """
    Layer of simplified spike response model (SRM0) neurons with stochastic threshold (escape noise). Adapted from
    `(Vasilaki et al., 2009) <https://intranet.physio.unibe.ch/Publikationen/Dokumente/Vasilaki2009PloSComputBio_1.pdf>`_.
    """

    def __init__(self, n: Optional[int]=None, shape: Optional[Iterable[int]]=None, traces: bool=False, traces_additive: bool=False, tc_trace: Union[float, torch.Tensor]=20.0, trace_scale: Union[float, torch.Tensor]=1.0, sum_input: bool=False, thresh: Union[float, torch.Tensor]=-50.0, rest: Union[float, torch.Tensor]=-70.0, reset: Union[float, torch.Tensor]=-70.0, refrac: Union[int, torch.Tensor]=5, tc_decay: Union[float, torch.Tensor]=10.0, lbound: float=None, eps_0: Union[float, torch.Tensor]=1.0, rho_0: Union[float, torch.Tensor]=1.0, d_thresh: Union[float, torch.Tensor]=5.0, **kwargs) ->None:
        """
        Instantiates a layer of SRM0 neurons.

        :param n: The number of neurons in the layer.
        :param shape: The dimensionality of the layer.
        :param traces: Whether to record spike traces.
        :param traces_additive: Whether to record spike traces additively.
        :param tc_trace: Time constant of spike trace decay.
        :param trace_scale: Scaling factor for spike trace.
        :param sum_input: Whether to sum all inputs.
        :param thresh: Spike threshold voltage.
        :param rest: Resting membrane voltage.
        :param reset: Post-spike reset voltage.
        :param refrac: Refractory (non-firing) period of the neuron.
        :param tc_decay: Time constant of neuron voltage decay.
        :param lbound: Lower bound of the voltage.
        :param eps_0: Scaling factor for pre-synaptic spike contributions.
        :param rho_0: Stochastic intensity at threshold.
        :param d_thresh: Width of the threshold region.
        """
        super().__init__(n=n, shape=shape, traces=traces, traces_additive=traces_additive, tc_trace=tc_trace, trace_scale=trace_scale, sum_input=sum_input)
        self.register_buffer('rest', torch.tensor(rest))
        self.register_buffer('reset', torch.tensor(reset))
        self.register_buffer('thresh', torch.tensor(thresh))
        self.register_buffer('refrac', torch.tensor(refrac))
        self.register_buffer('tc_decay', torch.tensor(tc_decay))
        self.register_buffer('decay', torch.tensor(tc_decay))
        self.register_buffer('eps_0', torch.tensor(eps_0))
        self.register_buffer('rho_0', torch.tensor(rho_0))
        self.register_buffer('d_thresh', torch.tensor(d_thresh))
        self.register_buffer('v', torch.FloatTensor())
        self.register_buffer('refrac_count', torch.FloatTensor())
        self.lbound = lbound

    def forward(self, x: torch.Tensor) ->None:
        """
        Runs a single simulation step.

        :param x: Inputs to the layer.
        """
        self.v = self.decay * (self.v - self.rest) + self.rest
        self.v += (self.refrac_count == 0).float() * self.eps_0 * x
        self.rho = self.rho_0 * torch.exp((self.v - self.thresh) / self.d_thresh)
        self.s_prob = 1.0 - torch.exp(-self.rho * self.dt)
        self.refrac_count = (self.refrac_count > 0).float() * (self.refrac_count - self.dt)
        self.s = torch.rand_like(self.s_prob) < self.s_prob
        self.refrac_count.masked_fill_(self.s, self.refrac)
        self.v.masked_fill_(self.s, self.reset)
        if self.lbound is not None:
            self.v.masked_fill_(self.v < self.lbound, self.lbound)
        super().forward(x)

    def reset_state_variables(self) ->None:
        """
        Resets relevant state variables.
        """
        super().reset_state_variables()
        self.v.fill_(self.rest)
        self.refrac_count.zero_()

    def compute_decays(self, dt) ->None:
        """
        Sets the relevant decays.
        """
        super().compute_decays(dt=dt)
        self.decay = torch.exp(-self.dt / self.tc_decay)

    def set_batch_size(self, batch_size) ->None:
        """
        Sets mini-batch size. Called when layer is added to a network.

        :param batch_size: Mini-batch size.
        """
        super().set_batch_size(batch_size=batch_size)
        self.v = self.rest * torch.ones(batch_size, *self.shape, device=self.v.device)
        self.refrac_count = torch.zeros_like(self.v, device=self.refrac_count.device)


class Connection(AbstractConnection):
    """
    Specifies synapses between one or two populations of neurons.
    """

    def __init__(self, source: Nodes, target: Nodes, nu: Optional[Union[float, Sequence[float]]]=None, reduction: Optional[callable]=None, weight_decay: float=0.0, **kwargs) ->None:
        """
        Instantiates a :code:`Connection` object.

        :param source: A layer of nodes from which the connection originates.
        :param target: A layer of nodes to which the connection connects.
        :param nu: Learning rate for both pre- and post-synaptic events.
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension.
        :param weight_decay: Constant multiple to decay weights by on each iteration.

        Keyword arguments:

        :param LearningRule update_rule: Modifies connection parameters according to
            some rule.
        :param torch.Tensor w: Strengths of synapses.
        :param torch.Tensor b: Target population bias.
        :param float wmin: Minimum allowed value on the connection weights.
        :param float wmax: Maximum allowed value on the connection weights.
        :param float norm: Total weight per target neuron normalization constant.
        """
        super().__init__(source, target, nu, reduction, weight_decay, **kwargs)
        w = kwargs.get('w', None)
        if w is None:
            if self.wmin == -np.inf or self.wmax == np.inf:
                w = torch.clamp(torch.rand(source.n, target.n), self.wmin, self.wmax)
            else:
                w = self.wmin + torch.rand(source.n, target.n) * (self.wmax - self.wmin)
        elif self.wmin != -np.inf or self.wmax != np.inf:
            w = torch.clamp(w, self.wmin, self.wmax)
        self.w = Parameter(w, requires_grad=False)
        self.b = Parameter(kwargs.get('b', torch.zeros(target.n)), requires_grad=False)

    def compute(self, s: torch.Tensor) ->torch.Tensor:
        """
        Compute pre-activations given spikes using connection weights.

        :param s: Incoming spikes.
        :return: Incoming spikes multiplied by synaptic weights (with or without
                 decaying spike activation).
        """
        post = s.float().view(s.size(0), -1) @ self.w + self.b
        return post.view(s.size(0), *self.target.shape)

    def update(self, **kwargs) ->None:
        """
        Compute connection's update rule.
        """
        super().update(**kwargs)

    def normalize(self) ->None:
        """
        Normalize weights so each target neuron has sum of connection weights equal to
        ``self.norm``.
        """
        if self.norm is not None:
            w_abs_sum = self.w.abs().sum(0).unsqueeze(0)
            w_abs_sum[w_abs_sum == 0] = 1.0
            self.w *= self.norm / w_abs_sum

    def reset_state_variables(self) ->None:
        """
        Contains resetting logic for the connection.
        """
        super().reset_state_variables()


class Conv2dConnection(AbstractConnection):
    """
    Specifies convolutional synapses between one or two populations of neurons.
    """

    def __init__(self, source: Nodes, target: Nodes, kernel_size: Union[int, Tuple[int, int]], stride: Union[int, Tuple[int, int]]=1, padding: Union[int, Tuple[int, int]]=0, dilation: Union[int, Tuple[int, int]]=1, nu: Optional[Union[float, Sequence[float]]]=None, reduction: Optional[callable]=None, weight_decay: float=0.0, **kwargs) ->None:
        """
        Instantiates a ``Conv2dConnection`` object.

        :param source: A layer of nodes from which the connection originates.
        :param target: A layer of nodes to which the connection connects.
        :param kernel_size: Horizontal and vertical size of convolutional kernels.
        :param stride: Horizontal and vertical stride for convolution.
        :param padding: Horizontal and vertical padding for convolution.
        :param dilation: Horizontal and vertical dilation for convolution.
        :param nu: Learning rate for both pre- and post-synaptic events.
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension.
        :param weight_decay: Constant multiple to decay weights by on each iteration.

        Keyword arguments:

        :param LearningRule update_rule: Modifies connection parameters according to
            some rule.
        :param torch.Tensor w: Strengths of synapses.
        :param torch.Tensor b: Target population bias.
        :param float wmin: Minimum allowed value on the connection weights.
        :param float wmax: Maximum allowed value on the connection weights.
        :param float norm: Total weight per target neuron normalization constant.
        """
        super().__init__(source, target, nu, reduction, weight_decay, **kwargs)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.in_channels, input_height, input_width = source.shape[0], source.shape[1], source.shape[2]
        self.out_channels, output_height, output_width = target.shape[0], target.shape[1], target.shape[2]
        width = (input_height - self.kernel_size[0] + 2 * self.padding[0]) / self.stride[0] + 1
        height = (input_width - self.kernel_size[1] + 2 * self.padding[1]) / self.stride[1] + 1
        shape = self.in_channels, self.out_channels, int(width), int(height)
        error = 'Target dimensionality must be (out_channels, ?,(input_height - filter_height + 2 * padding_height) / stride_height + 1,(input_width - filter_width + 2 * padding_width) / stride_width + 1'
        assert target.shape[0] == shape[1] and target.shape[1] == shape[2] and target.shape[2] == shape[3], error
        w = kwargs.get('w', None)
        if w is None:
            if self.wmin == -np.inf or self.wmax == np.inf:
                w = torch.clamp(torch.rand(self.out_channels, self.in_channels, *self.kernel_size), self.wmin, self.wmax)
            else:
                w = (self.wmax - self.wmin) * torch.rand(self.out_channels, self.in_channels, *self.kernel_size)
                w += self.wmin
        elif self.wmin != -np.inf or self.wmax != np.inf:
            w = torch.clamp(w, self.wmin, self.wmax)
        self.w = Parameter(w, requires_grad=False)
        self.b = Parameter(kwargs.get('b', torch.zeros(self.out_channels)), requires_grad=False)

    def compute(self, s: torch.Tensor) ->torch.Tensor:
        """
        Compute convolutional pre-activations given spikes using layer weights.

        :param s: Incoming spikes.
        :return: Incoming spikes multiplied by synaptic weights (with or without
            decaying spike activation).
        """
        return F.conv2d(s.float(), self.w, self.b, stride=self.stride, padding=self.padding, dilation=self.dilation)

    def update(self, **kwargs) ->None:
        """
        Compute connection's update rule.
        """
        super().update(**kwargs)

    def normalize(self) ->None:
        """
        Normalize weights along the first axis according to total weight per target
        neuron.
        """
        if self.norm is not None:
            w = self.w.view(self.w.size(0) * self.w.size(1), self.w.size(2) * self.w.size(3))
            for fltr in range(w.size(0)):
                w[fltr] *= self.norm / w[fltr].sum(0)

    def reset_state_variables(self) ->None:
        """
        Contains resetting logic for the connection.
        """
        super().reset_state_variables()


class MaxPool2dConnection(AbstractConnection):
    """
    Specifies max-pooling synapses between one or two populations of neurons by keeping
    online estimates of maximally firing neurons.
    """

    def __init__(self, source: Nodes, target: Nodes, kernel_size: Union[int, Tuple[int, int]], stride: Union[int, Tuple[int, int]]=1, padding: Union[int, Tuple[int, int]]=0, dilation: Union[int, Tuple[int, int]]=1, **kwargs) ->None:
        """
        Instantiates a ``MaxPool2dConnection`` object.

        :param source: A layer of nodes from which the connection originates.
        :param target: A layer of nodes to which the connection connects.
        :param kernel_size: Horizontal and vertical size of convolutional kernels.
        :param stride: Horizontal and vertical stride for convolution.
        :param padding: Horizontal and vertical padding for convolution.
        :param dilation: Horizontal and vertical dilation for convolution.

        Keyword arguments:

        :param decay: Decay rate of online estimates of average firing activity.
        """
        super().__init__(source, target, None, None, 0.0, **kwargs)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.register_buffer('firing_rates', torch.zeros(source.shape))

    def compute(self, s: torch.Tensor) ->torch.Tensor:
        """
        Compute max-pool pre-activations given spikes using online firing rate
        estimates.

        :param s: Incoming spikes.
        :return: Incoming spikes multiplied by synaptic weights (with or without
            decaying spike activation).
        """
        self.firing_rates -= self.decay * self.firing_rates
        self.firing_rates += s.float()
        _, indices = F.max_pool2d(self.firing_rates, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, return_indices=True)
        return s.take(indices).float()

    def update(self, **kwargs) ->None:
        """
        Compute connection's update rule.
        """
        super().update(**kwargs)

    def normalize(self) ->None:
        """
        No weights -> no normalization.
        """
        pass

    def reset_state_variables(self) ->None:
        """
        Contains resetting logic for the connection.
        """
        super().reset_state_variables()
        self.firing_rates = torch.zeros(self.source.shape)


class LocalConnection(AbstractConnection):
    """
    Specifies a locally connected connection between one or two populations of neurons.
    """

    def __init__(self, source: Nodes, target: Nodes, kernel_size: Union[int, Tuple[int, int]], stride: Union[int, Tuple[int, int]], n_filters: int, nu: Optional[Union[float, Sequence[float]]]=None, reduction: Optional[callable]=None, weight_decay: float=0.0, **kwargs) ->None:
        """
        Instantiates a ``LocalConnection`` object. Source population should be
        two-dimensional.

        Neurons in the post-synaptic population are ordered by receptive field; that is,
        if there are ``n_conv`` neurons in each post-synaptic patch, then the first
        ``n_conv`` neurons in the post-synaptic population correspond to the first
        receptive field, the second ``n_conv`` to the second receptive field, and so on.

        :param source: A layer of nodes from which the connection originates.
        :param target: A layer of nodes to which the connection connects.
        :param kernel_size: Horizontal and vertical size of convolutional kernels.
        :param stride: Horizontal and vertical stride for convolution.
        :param n_filters: Number of locally connected filters per pre-synaptic region.
        :param nu: Learning rate for both pre- and post-synaptic events.
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension.
        :param weight_decay: Constant multiple to decay weights by on each iteration.

        Keyword arguments:

        :param LearningRule update_rule: Modifies connection parameters according to
            some rule.
        :param torch.Tensor w: Strengths of synapses.
        :param torch.Tensor b: Target population bias.
        :param float wmin: Minimum allowed value on the connection weights.
        :param float wmax: Maximum allowed value on the connection weights.
        :param float norm: Total weight per target neuron normalization constant.
        :param Tuple[int, int] input_shape: Shape of input population if it's not
            ``[sqrt, sqrt]``.
        """
        super().__init__(source, target, nu, reduction, weight_decay, **kwargs)
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_filters = n_filters
        shape = kwargs.get('input_shape', None)
        if shape is None:
            sqrt = int(np.sqrt(source.n))
            shape = _pair(sqrt)
        if kernel_size == shape:
            conv_size = [1, 1]
        else:
            conv_size = int((shape[0] - kernel_size[0]) / stride[0]) + 1, int((shape[1] - kernel_size[1]) / stride[1]) + 1
        self.conv_size = conv_size
        conv_prod = int(np.prod(conv_size))
        kernel_prod = int(np.prod(kernel_size))
        assert target.n == n_filters * conv_prod, 'Target layer size must be n_filters * (kernel_size ** 2).'
        locations = torch.zeros(kernel_size[0], kernel_size[1], conv_size[0], conv_size[1]).long()
        for c1 in range(conv_size[0]):
            for c2 in range(conv_size[1]):
                for k1 in range(kernel_size[0]):
                    for k2 in range(kernel_size[1]):
                        location = c1 * stride[0] * shape[1] + c2 * stride[1] + k1 * shape[0] + k2
                        locations[k1, k2, c1, c2] = location
        self.register_buffer('locations', locations.view(kernel_prod, conv_prod))
        w = kwargs.get('w', None)
        if w is None:
            w = torch.zeros(source.n, target.n)
            for f in range(n_filters):
                for c in range(conv_prod):
                    for k in range(kernel_prod):
                        if self.wmin == -np.inf or self.wmax == np.inf:
                            w[self.locations[k, c], f * conv_prod + c] = np.clip(np.random.rand(), self.wmin, self.wmax)
                        else:
                            w[self.locations[k, c], f * conv_prod + c] = self.wmin + np.random.rand() * (self.wmax - self.wmin)
        elif self.wmin != -np.inf or self.wmax != np.inf:
            w = torch.clamp(w, self.wmin, self.wmax)
        self.w = Parameter(w, requires_grad=False)
        self.register_buffer('mask', self.w == 0)
        self.b = Parameter(kwargs.get('b', torch.zeros(target.n)), requires_grad=False)
        if self.norm is not None:
            self.norm *= kernel_prod

    def compute(self, s: torch.Tensor) ->torch.Tensor:
        """
        Compute pre-activations given spikes using layer weights.

        :param s: Incoming spikes.
        :return: Incoming spikes multiplied by synaptic weights (with or without
            decaying spike activation).
        """
        if self.w.shape[0] == self.source.n and self.w.shape[1] == self.target.n:
            return s.float().view(s.size(0), -1) @ self.w + self.b
        else:
            a_post = s.float().view(s.size(0), -1) @ self.w.view(self.source.n, self.target.n) + self.b
            return a_post.view(*self.target.shape)

    def update(self, **kwargs) ->None:
        """
        Compute connection's update rule.

        Keyword arguments:

        :param ByteTensor mask: Boolean mask determining which weights to clamp to zero.
        """
        if kwargs['mask'] is None:
            kwargs['mask'] = self.mask
        super().update(**kwargs)

    def normalize(self) ->None:
        """
        Normalize weights so each target neuron has sum of connection weights equal to
        ``self.norm``.
        """
        if self.norm is not None:
            w = self.w.view(self.source.n, self.target.n)
            w *= self.norm / self.w.sum(0).view(1, -1)

    def reset_state_variables(self) ->None:
        """
        Contains resetting logic for the connection.
        """
        super().reset_state_variables()


class MeanFieldConnection(AbstractConnection):
    """
    A connection between one or two populations of neurons which computes a summary of
    the pre-synaptic population to use as weighted input to the post-synaptic
    population.
    """

    def __init__(self, source: Nodes, target: Nodes, nu: Optional[Union[float, Sequence[float]]]=None, weight_decay: float=0.0, **kwargs) ->None:
        """
        Instantiates a :code:`MeanFieldConnection` object.
        :param source: A layer of nodes from which the connection originates.
        :param target: A layer of nodes to which the connection connects.
        :param nu: Learning rate for both pre- and post-synaptic events.
        :param weight_decay: Constant multiple to decay weights by on each iteration.
        Keyword arguments:
        :param LearningRule update_rule: Modifies connection parameters according to
            some rule.
        :param torch.Tensor w: Strengths of synapses.
        :param float wmin: Minimum allowed value on the connection weights.
        :param float wmax: Maximum allowed value on the connection weights.
        :param float norm: Total weight per target neuron normalization constant.
        """
        super().__init__(source, target, nu, weight_decay, **kwargs)
        w = kwargs.get('w', None)
        if w is None:
            if self.wmin == -np.inf or self.wmax == np.inf:
                w = torch.clamp((torch.randn(1)[0] + 1) / 10, self.wmin, self.wmax)
            else:
                w = self.wmin + (torch.randn(1)[0] + 1) / 10 * (self.wmax - self.wmin)
        elif self.wmin != -np.inf or self.wmax != np.inf:
            w = torch.clamp(w, self.wmin, self.wmax)
        self.w = Parameter(w, requires_grad=False)

    def compute(self, s: torch.Tensor) ->torch.Tensor:
        """
        Compute pre-activations given spikes using layer weights.
        :param s: Incoming spikes.
        :return: Incoming spikes multiplied by synaptic weights (with or without
            decaying spike activation).
        """
        return s.float().mean() * self.w

    def update(self, **kwargs) ->None:
        """
        Compute connection's update rule.
        """
        super().update(**kwargs)

    def normalize(self) ->None:
        """
        Normalize weights so each target neuron has sum of connection weights equal to
        ``self.norm``.
        """
        if self.norm is not None:
            self.w = self.w.view(1, self.target.n)
            self.w *= self.norm / self.w.sum()
            self.w = self.w.view(1, *self.target.shape)

    def reset_state_variables(self) ->None:
        """
        Contains resetting logic for the connection.
        """
        super().reset_state_variables()


class SparseConnection(AbstractConnection):
    """
    Specifies sparse synapses between one or two populations of neurons.
    """

    def __init__(self, source: Nodes, target: Nodes, nu: Optional[Union[float, Sequence[float]]]=None, reduction: Optional[callable]=None, weight_decay: float=None, **kwargs) ->None:
        """
        Instantiates a :code:`Connection` object with sparse weights.

        :param source: A layer of nodes from which the connection originates.
        :param target: A layer of nodes to which the connection connects.
        :param nu: Learning rate for both pre- and post-synaptic events.
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension.
        :param weight_decay: Constant multiple to decay weights by on each iteration.

        Keyword arguments:

        :param torch.Tensor w: Strengths of synapses.
        :param float sparsity: Fraction of sparse connections to use.
        :param LearningRule update_rule: Modifies connection parameters according to
            some rule.
        :param float wmin: Minimum allowed value on the connection weights.
        :param float wmax: Maximum allowed value on the connection weights.
        :param float norm: Total weight per target neuron normalization constant.
        """
        super().__init__(source, target, nu, reduction, weight_decay, **kwargs)
        w = kwargs.get('w', None)
        self.sparsity = kwargs.get('sparsity', None)
        assert w is not None and self.sparsity is None or w is None and self.sparsity is not None, 'Only one of "weights" or "sparsity" must be specified'
        if w is None and self.sparsity is not None:
            i = torch.bernoulli(1 - self.sparsity * torch.ones(*source.shape, *target.shape))
            if self.wmin == -np.inf or self.wmax == np.inf:
                v = torch.clamp(torch.rand(*source.shape, *target.shape)[i.byte()], self.wmin, self.wmax)
            else:
                v = self.wmin + torch.rand(*source.shape, *target.shape)[i.byte()] * (self.wmax - self.wmin)
            w = torch.sparse.FloatTensor(i.nonzero().t(), v)
        elif w is not None and self.sparsity is None:
            assert w.is_sparse, 'Weight matrix is not sparse (see torch.sparse module)'
            if self.wmin != -np.inf or self.wmax != np.inf:
                w = torch.clamp(w, self.wmin, self.wmax)
        self.w = Parameter(w, requires_grad=False)

    def compute(self, s: torch.Tensor) ->torch.Tensor:
        """
        Compute convolutional pre-activations given spikes using layer weights.

        :param s: Incoming spikes.
        :return: Incoming spikes multiplied by synaptic weights (with or without
            decaying spike activation).
        """
        return torch.mm(self.w, s.unsqueeze(-1).float()).squeeze(-1)

    def update(self, **kwargs) ->None:
        """
        Compute connection's update rule.
        """
        pass

    def normalize(self) ->None:
        """
        Normalize weights along the first axis according to total weight per target
        neuron.
        """
        pass

    def reset_state_variables(self) ->None:
        """
        Contains resetting logic for the connection.
        """
        super().reset_state_variables()


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(6400, 1000)
        self.fc2 = nn.Linear(1000, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class NN(nn.Module):

    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.linear_1 = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = torch.sigmoid(self.linear_1(x.float().view(-1)))
        return out


class FullyConnectedNetwork(nn.Module):
    """
    Simply fully-connected network implemented in PyTorch.
    """

    def __init__(self):
        super(FullyConnectedNetwork, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (FeatureExtractor,
     lambda: ([], {'submodule': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (FullyConnectedNetwork,
     lambda: ([], {}),
     lambda: ([torch.rand([784, 784])], {}),
     True),
    (Net,
     lambda: ([], {}),
     lambda: ([torch.rand([6400, 6400])], {}),
     True),
]

class Test_BindsNET_bindsnet(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

