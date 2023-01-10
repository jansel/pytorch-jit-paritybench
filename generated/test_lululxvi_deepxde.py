import sys
_module = sys.modules[__name__]
del sys
__about__ = _module
deepxde = _module
backend = _module
backend = _module
jax = _module
tensor = _module
paddle = _module
pytorch = _module
tensor = _module
set_default_backend = _module
tensorflow = _module
tensorflow_compat_v1 = _module
callbacks = _module
config = _module
data = _module
constraint = _module
dataset = _module
fpde = _module
func_constraint = _module
function = _module
function_spaces = _module
helper = _module
ide = _module
mf = _module
pde = _module
pde_operator = _module
quadruple = _module
sampler = _module
triple = _module
display = _module
geometry = _module
csg = _module
geometry_1d = _module
geometry_2d = _module
geometry_3d = _module
geometry_nd = _module
pointcloud = _module
timedomain = _module
gradients = _module
icbc = _module
boundary_conditions = _module
initial_conditions = _module
losses = _module
metrics = _module
model = _module
nn = _module
activations = _module
initializers = _module
fnn = _module
nn = _module
deeponet = _module
msffn = _module
deeponet = _module
fnn = _module
mionet = _module
nn = _module
regularizers = _module
mfnn = _module
resnet = _module
optimizers = _module
config = _module
optimizers = _module
tfp_optimizer = _module
scipy_optimizer = _module
real = _module
utils = _module
array_ops_compat = _module
external = _module
internal = _module
conf = _module
func = _module
func_uncertainty = _module
mf_dataset = _module
mf_func = _module
antiderivative_aligned = _module
antiderivative_unaligned = _module
Allen_Cahn = _module
Beltrami_flow = _module
Burgers = _module
Burgers_RAR = _module
Euler_beam = _module
Helmholtz_Dirichlet_2d = _module
Helmholtz_Neumann_2d_hole = _module
Helmholtz_Sound_hard_ABC_2d = _module
Klein_Gordon = _module
Kovasznay_flow = _module
Laplace_disk = _module
Lotka_Volterra = _module
Poisson_Dirichlet_1d = _module
Poisson_Dirichlet_1d_exactBC = _module
Poisson_Lshape = _module
Poisson_Neumann_1d = _module
Poisson_PointSetOperator_1d = _module
Poisson_Robin_1d = _module
Poisson_multiscale_1d = _module
Poisson_periodic_1d = _module
Volterra_IDE = _module
diffusion_1d = _module
diffusion_1d_exactBC = _module
diffusion_1d_resample = _module
diffusion_reaction = _module
fractional_Poisson_1d = _module
fractional_Poisson_2d = _module
fractional_Poisson_3d = _module
fractional_diffusion_1d = _module
heat = _module
heat_resample = _module
ode_2nd = _module
ode_system = _module
wave_1d = _module
Lorenz_inverse = _module
Lorenz_inverse_forced = _module
Navier_Stokes_inverse = _module
brinkman_forchheimer = _module
diffusion_1d_inverse = _module
diffusion_reaction_rate = _module
elliptic_inverse_field = _module
elliptic_inverse_field_batch = _module
fractional_Poisson_1d_inverse = _module
fractional_Poisson_2d_inverse = _module
reaction_inverse = _module
sample_to_test = _module
setup = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, matplotlib, numbers, numpy, pandas, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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
xrange = range
wraps = functools.wraps


import torch


import time


import numpy as np


import random


from collections import OrderedDict


import math


from scipy.io import loadmat


import matplotlib.pyplot as plt


from scipy import integrate


def sum(input_tensor, dim, keepdims=False):
    return torch.sum(input_tensor, dim, keepdim=keepdims)


class NN(torch.nn.Module):
    """Base class for all neural network modules."""

    def __init__(self):
        super().__init__()
        self.regularizer = None
        self._input_transform = None
        self._output_transform = None

    def apply_feature_transform(self, transform):
        """Compute the features by appling a transform to the network inputs, i.e.,
        features = transform(inputs). Then, outputs = network(features).
        """
        self._input_transform = transform

    def apply_output_transform(self, transform):
        """Apply a transform to the network outputs, i.e.,
        outputs = transform(inputs, outputs).
        """
        self._output_transform = transform

    def num_trainable_parameters(self):
        """Evaluate the number of trainable parameters for the NN."""
        return sum(v.numel() for v in self.parameters() if v.requires_grad)


class FNN(NN):
    """Fully-connected neural network."""

    def __init__(self, layer_sizes, activation, kernel_initializer):
        super().__init__()
        if isinstance(activation, list):
            if not len(layer_sizes) - 1 == len(activation):
                raise ValueError('Total number of activation functions do not match with sum of hidden layers and output layer!')
            self.activation = list(map(activations.get, activation))
        else:
            self.activation = activations.get(activation)
        initializer = initializers.get(kernel_initializer)
        initializer_zero = initializers.get('zeros')
        self.linears = torch.nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            self.linears.append(torch.nn.Linear(layer_sizes[i - 1], layer_sizes[i], dtype=config.real(torch)))
            initializer(self.linears[-1].weight)
            initializer_zero(self.linears[-1].bias)

    def forward(self, inputs):
        x = inputs
        if self._input_transform is not None:
            x = self._input_transform(x)
        for j, linear in enumerate(self.linears[:-1]):
            x = self.activation[j](linear(x)) if isinstance(self.activation, list) else self.activation(linear(x))
        x = self.linears[-1](x)
        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x


class DeepONet(NN):
    """Deep operator network.

    Args:
        layer_sizes_branch: A list of integers as the width of a fully connected network,
            or `(dim, f)` where `dim` is the input dimension and `f` is a network
            function. The width of the last layer in the branch and trunk net should be
            equal.
        layer_sizes_trunk (list): A list of integers as the width of a fully connected
            network.
        activation: If `activation` is a ``string``, then the same activation is used in
            both trunk and branch nets. If `activation` is a ``dict``, then the trunk
            net uses the activation `activation["trunk"]`, and the branch net uses
            `activation["branch"]`.
    """

    def __init__(self, layer_sizes_branch, layer_sizes_trunk, activation, kernel_initializer):
        super().__init__()
        if isinstance(activation, dict):
            activation_branch = activations.get(activation['branch'])
            self.activation_trunk = activations.get(activation['trunk'])
        else:
            activation_branch = self.activation_trunk = activations.get(activation)
        if callable(layer_sizes_branch[1]):
            self.branch = layer_sizes_branch[1]
        else:
            self.branch = FNN(layer_sizes_branch, activation_branch, kernel_initializer)
        self.trunk = FNN(layer_sizes_trunk, self.activation_trunk, kernel_initializer)
        self.b = torch.nn.parameter.Parameter(torch.tensor(0.0))

    def forward(self, inputs):
        x_func = inputs[0]
        x_loc = inputs[1]
        x_func = self.branch(x_func)
        if self._input_transform is not None:
            x_loc = self._input_transform(x_loc)
        x_loc = self.activation_trunk(self.trunk(x_loc))
        if x_func.shape[-1] != x_loc.shape[-1]:
            raise AssertionError('Output sizes of branch net and trunk net do not match.')
        x = torch.einsum('bi,bi->b', x_func, x_loc)
        x = torch.unsqueeze(x, 1)
        x += self.b
        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x


class DeepONetCartesianProd(NN):
    """Deep operator network for dataset in the format of Cartesian product.

    Args:
        layer_sizes_branch: A list of integers as the width of a fully connected network,
            or `(dim, f)` where `dim` is the input dimension and `f` is a network
            function. The width of the last layer in the branch and trunk net should be
            equal.
        layer_sizes_trunk (list): A list of integers as the width of a fully connected
            network.
        activation: If `activation` is a ``string``, then the same activation is used in
            both trunk and branch nets. If `activation` is a ``dict``, then the trunk
            net uses the activation `activation["trunk"]`, and the branch net uses
            `activation["branch"]`.
    """

    def __init__(self, layer_sizes_branch, layer_sizes_trunk, activation, kernel_initializer, regularization=None):
        super().__init__()
        if isinstance(activation, dict):
            activation_branch = activations.get(activation['branch'])
            self.activation_trunk = activations.get(activation['trunk'])
        else:
            activation_branch = self.activation_trunk = activations.get(activation)
        if callable(layer_sizes_branch[1]):
            self.branch = layer_sizes_branch[1]
        else:
            self.branch = FNN(layer_sizes_branch, activation_branch, kernel_initializer)
        self.trunk = FNN(layer_sizes_trunk, self.activation_trunk, kernel_initializer)
        self.b = torch.nn.parameter.Parameter(torch.tensor(0.0))
        self.regularizer = regularization

    def forward(self, inputs):
        x_func = inputs[0]
        x_loc = inputs[1]
        x_func = self.branch(x_func)
        if self._input_transform is not None:
            x_loc = self._input_transform(x_loc)
        x_loc = self.activation_trunk(self.trunk(x_loc))
        if x_func.shape[-1] != x_loc.shape[-1]:
            raise AssertionError('Output sizes of branch net and trunk net do not match.')
        x = torch.einsum('bi,ni->bn', x_func, x_loc)
        x += self.b
        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x


class PODDeepONet(NN):
    """Deep operator network with proper orthogonal decomposition (POD) for dataset in
    the format of Cartesian product.

    Args:
        pod_basis: POD basis used in the trunk net.
        layer_sizes_branch: A list of integers as the width of a fully connected network,
            or `(dim, f)` where `dim` is the input dimension and `f` is a network
            function. The width of the last layer in the branch and trunk net should be
            equal.
        activation: If `activation` is a ``string``, then the same activation is used in
            both trunk and branch nets. If `activation` is a ``dict``, then the trunk
            net uses the activation `activation["trunk"]`, and the branch net uses
            `activation["branch"]`.
        layer_sizes_trunk (list): A list of integers as the width of a fully connected
            network. If ``None``, then only use POD basis as the trunk net.

    References:
        `L. Lu, X. Meng, S. Cai, Z. Mao, S. Goswami, Z. Zhang, & G. E. Karniadakis. A
        comprehensive and fair comparison of two neural operators (with practical
        extensions) based on FAIR data. arXiv preprint arXiv:2111.05512, 2021
        <https://arxiv.org/abs/2111.05512>`_.
    """

    def __init__(self, pod_basis, layer_sizes_branch, activation, kernel_initializer, layer_sizes_trunk=None, regularization=None):
        super().__init__()
        if isinstance(activation, dict):
            activation_branch = activations.get(activation['branch'])
            self.activation_trunk = activations.get(activation['trunk'])
        else:
            activation_branch = self.activation_trunk = activations.get(activation)
        self.pod_basis = torch.as_tensor(pod_basis, dtype=torch.float32)
        if callable(layer_sizes_branch[1]):
            self.branch = layer_sizes_branch[1]
        else:
            self.branch = FNN(layer_sizes_branch, activation_branch, kernel_initializer)
        self.trunk = None
        if layer_sizes_trunk is not None:
            self.trunk = FNN(layer_sizes_trunk, self.activation_trunk, kernel_initializer)
            self.b = torch.nn.parameter.Parameter(torch.tensor(0.0))
        self.regularizer = regularization

    def forward(self, inputs):
        x_func = inputs[0]
        x_loc = inputs[1]
        x_func = self.branch(x_func)
        if self.trunk is None:
            x = torch.einsum('bi,ni->bn', x_func, self.pod_basis)
        else:
            x_loc = self.activation_trunk(self.trunk(x_loc))
            x = torch.einsum('bi,ni->bn', x_func, torch.cat((self.pod_basis, x_loc), 1))
            x += self.b
        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x


class PFNN(NN):
    """Parallel fully-connected network that uses independent sub-networks for each
    network output.

    Args:
        layer_sizes: A nested list that defines the architecture of the neural network
            (how the layers are connected). If `layer_sizes[i]` is an int, it represents
            one layer shared by all the outputs; if `layer_sizes[i]` is a list, it
            represents `len(layer_sizes[i])` sub-layers, each of which is exclusively
            used by one output. Note that `len(layer_sizes[i])` should equal the number
            of outputs. Every number specifies the number of neurons in that layer.
    """

    def __init__(self, layer_sizes, activation, kernel_initializer):
        super().__init__()
        self.activation = activations.get(activation)
        initializer = initializers.get(kernel_initializer)
        initializer_zero = initializers.get('zeros')
        if len(layer_sizes) <= 1:
            raise ValueError('must specify input and output sizes')
        if not isinstance(layer_sizes[0], int):
            raise ValueError('input size must be integer')
        if not isinstance(layer_sizes[-1], int):
            raise ValueError('output size must be integer')
        n_output = layer_sizes[-1]

        def make_linear(n_input, n_output):
            linear = torch.nn.Linear(n_input, n_output, dtype=config.real(torch))
            initializer(linear.weight)
            initializer_zero(linear.bias)
            return linear
        self.layers = torch.nn.ModuleList()
        for i in range(1, len(layer_sizes) - 1):
            prev_layer_size = layer_sizes[i - 1]
            curr_layer_size = layer_sizes[i]
            if isinstance(curr_layer_size, (list, tuple)):
                if len(curr_layer_size) != n_output:
                    raise ValueError('number of sub-layers should equal number of network outputs')
                if isinstance(prev_layer_size, (list, tuple)):
                    self.layers.append(torch.nn.ModuleList([make_linear(prev_layer_size[j], curr_layer_size[j]) for j in range(n_output)]))
                else:
                    self.layers.append(torch.nn.ModuleList([make_linear(prev_layer_size, curr_layer_size[j]) for j in range(n_output)]))
            else:
                if not isinstance(prev_layer_size, int):
                    raise ValueError('cannot rejoin parallel subnetworks after splitting')
                self.layers.append(make_linear(prev_layer_size, curr_layer_size))
        if isinstance(layer_sizes[-2], (list, tuple)):
            self.layers.append(torch.nn.ModuleList([make_linear(layer_sizes[-2][j], 1) for j in range(n_output)]))
        else:
            self.layers.append(make_linear(layer_sizes[-2], n_output))

    def forward(self, inputs):
        x = inputs
        if self._input_transform is not None:
            x = self._input_transform(x)
        for layer in self.layers[:-1]:
            if isinstance(layer, torch.nn.ModuleList):
                if isinstance(x, list):
                    x = [self.activation(f(x_)) for f, x_ in zip(layer, x)]
                else:
                    x = [self.activation(f(x)) for f in layer]
            else:
                x = self.activation(layer(x))
        if isinstance(x, list):
            x = torch.cat([f(x_) for f, x_ in zip(self.layers[-1], x)], dim=1)
        else:
            x = self.layers[-1](x)
        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x


class MIONetCartesianProd(NN):
    """MIONet with two input functions for Cartesian product format."""

    def __init__(self, layer_sizes_branch1, layer_sizes_branch2, layer_sizes_trunk, activation, kernel_initializer, regularization=None, trunk_last_activation=False, merge_operation='mul', layer_sizes_merger=None, output_merge_operation='mul', layer_sizes_output_merger=None):
        super().__init__()
        if isinstance(activation, dict):
            self.activation_branch1 = activations.get(activation['branch1'])
            self.activation_branch2 = activations.get(activation['branch2'])
            self.activation_trunk = activations.get(activation['trunk'])
        else:
            self.activation_branch1 = self.activation_branch2 = self.activation_trunk = activations.get(activation)
        if callable(layer_sizes_branch1[1]):
            self.branch1 = layer_sizes_branch1[1]
        else:
            self.branch1 = FNN(layer_sizes_branch1, self.activation_branch1, kernel_initializer)
        if callable(layer_sizes_branch2[1]):
            self.branch2 = layer_sizes_branch2[1]
        else:
            self.branch2 = FNN(layer_sizes_branch2, self.activation_branch2, kernel_initializer)
        if layer_sizes_merger is not None:
            self.activation_merger = activations.get(activation['merger'])
            if callable(layer_sizes_merger[1]):
                self.merger = layer_sizes_merger[1]
            else:
                self.merger = FNN(layer_sizes_merger, self.activation_merger, kernel_initializer)
        else:
            self.merger = None
        if layer_sizes_output_merger is not None:
            self.activation_output_merger = activations.get(activation['output merger'])
            if callable(layer_sizes_output_merger[1]):
                self.output_merger = layer_sizes_output_merger[1]
            else:
                self.output_merger = FNN(layer_sizes_output_merger, self.activation_output_merger, kernel_initializer)
        else:
            self.output_merger = None
        self.trunk = FNN(layer_sizes_trunk, self.activation_trunk, kernel_initializer)
        self.b = torch.nn.parameter.Parameter(torch.tensor(0.0))
        self.regularizer = regularization
        self.trunk_last_activation = trunk_last_activation
        self.merge_operation = merge_operation
        self.output_merge_operation = output_merge_operation

    def forward(self, inputs):
        x_func1 = inputs[0]
        x_func2 = inputs[1]
        x_loc = inputs[2]
        y_func1 = self.branch1(x_func1)
        y_func2 = self.branch2(x_func2)
        if self.merge_operation == 'cat':
            x_merger = torch.cat((y_func1, y_func2), 1)
        else:
            if y_func1.shape[-1] != y_func2.shape[-1]:
                raise AssertionError('Output sizes of branch1 net and branch2 net do not match.')
            if self.merge_operation == 'add':
                x_merger = y_func1 + y_func2
            elif self.merge_operation == 'mul':
                x_merger = torch.mul(y_func1, y_func2)
            else:
                raise NotImplementedError(f'{self.merge_operation} operation to be implimented')
        if self.merger is not None:
            y_func = self.merger(x_merger)
        else:
            y_func = x_merger
        if self._input_transform is not None:
            y_loc = self._input_transform(x_loc)
        y_loc = self.trunk(x_loc)
        if self.trunk_last_activation:
            y_loc = self.activation_trunk(y_loc)
        if y_func.shape[-1] != y_loc.shape[-1]:
            raise AssertionError('Output sizes of merger net and trunk net do not match.')
        if self.output_merger is None:
            y = torch.einsum('ip,jp->ij', y_func, y_loc)
        else:
            y_func = y_func[:, None, :]
            y_loc = y_loc[None, :]
            if self.output_merge_operation == 'mul':
                y = torch.mul(y_func, y_loc)
            elif self.output_merge_operation == 'add':
                y = y_func + y_loc
            elif self.output_merge_operation == 'cat':
                y_func = y_func.repeat(1, y_loc.shape[1], 1)
                y_loc = y_loc.repeat(y_func.shape[0], 1, 1)
                y = torch.cat((y_func, y_loc), dim=2)
            shape0 = y.shape[0]
            shape1 = y.shape[1]
            y = y.reshape(shape0 * shape1, -1)
            y = self.output_merger(y)
            y = y.reshape(shape0, shape1)
        y += self.b
        if self._output_transform is not None:
            y = self._output_transform(inputs, y)
        return y


class PODMIONet(NN):
    """MIONet with two input functions and proper orthogonal decomposition (POD)
    for Cartesian product format."""

    def __init__(self, pod_basis, layer_sizes_branch1, layer_sizes_branch2, activation, kernel_initializer, layer_sizes_trunk=None, regularization=None, trunk_last_activation=False, merge_operation='mul', layer_sizes_merger=None):
        super().__init__()
        if isinstance(activation, dict):
            self.activation_branch1 = activations.get(activation['branch1'])
            self.activation_branch2 = activations.get(activation['branch2'])
            self.activation_trunk = activations.get(activation['trunk'])
            self.activation_merger = activations.get(activation['merger'])
        else:
            self.activation_branch1 = self.activation_branch2 = self.activation_trunk = activations.get(activation)
        self.pod_basis = torch.as_tensor(pod_basis, dtype=torch.float32)
        if callable(layer_sizes_branch1[1]):
            self.branch1 = layer_sizes_branch1[1]
        else:
            self.branch1 = FNN(layer_sizes_branch1, self.activation_branch1, kernel_initializer)
        if callable(layer_sizes_branch2[1]):
            self.branch2 = layer_sizes_branch2[1]
        else:
            self.branch2 = FNN(layer_sizes_branch2, self.activation_branch2, kernel_initializer)
        if layer_sizes_merger is not None:
            if callable(layer_sizes_merger[1]):
                self.merger = layer_sizes_merger[1]
            else:
                self.merger = FNN(layer_sizes_merger, self.activation_merger, kernel_initializer)
        else:
            self.merger = None
        self.trunk = None
        if layer_sizes_trunk is not None:
            self.trunk = FNN(layer_sizes_trunk, self.activation_trunk, kernel_initializer)
            self.b = torch.nn.parameter.Parameter(torch.tensor(0.0))
        self.regularizer = regularization
        self.trunk_last_activation = trunk_last_activation
        self.merge_operation = merge_operation

    def forward(self, inputs):
        x_func1 = inputs[0]
        x_func2 = inputs[1]
        x_loc = inputs[2]
        y_func1 = self.branch1(x_func1)
        y_func2 = self.branch2(x_func2)
        if self.merge_operation == 'cat':
            x_merger = torch.cat((y_func1, y_func2), 1)
        else:
            if y_func1.shape[-1] != y_func2.shape[-1]:
                raise AssertionError('Output sizes of branch1 net and branch2 net do not match.')
            if self.merge_operation == 'add':
                x_merger = y_func1 + y_func2
            elif self.merge_operation == 'mul':
                x_merger = torch.mul(y_func1, y_func2)
            else:
                raise NotImplementedError(f'{self.merge_operation} operation to be implimented')
        if self.merger is not None:
            y_func = self.merger(x_merger)
        else:
            y_func = x_merger
        if self.trunk is None:
            y = torch.einsum('bi,ni->bn', y_func, self.pod_basis)
        else:
            y_loc = self.trunk(x_loc)
            if self.trunk_last_activation:
                y_loc = self.activation_trunk(y_loc)
            y = torch.einsum('bi,ni->bn', y_func, torch.cat((self.pod_basis, y_loc), 1))
            y += self.b
        if self._output_transform is not None:
            y = self._output_transform(inputs, y)
        return y

