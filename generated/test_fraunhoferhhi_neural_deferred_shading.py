import sys
_module = sys.modules[__name__]
del sys
nds = _module
core = _module
camera = _module
mesh = _module
renderer = _module
view = _module
losses = _module
laplacian = _module
mask = _module
normal_consistency = _module
shading = _module
modules = _module
embedder = _module
fc = _module
gfft = _module
neuralshader = _module
space_normalization = _module
view_sampler = _module
utils = _module
geometry = _module
images = _module
io = _module
profiling = _module
visualization = _module
viewer = _module
controller = _module
primitives = _module
shaders = _module
viewer = _module
reconstruct = _module
view = _module

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


import numpy as np


from typing import Dict


from typing import List


import torch.nn as nn


from typing import Callable


from typing import Tuple


from typing import Union


import time


from queue import Queue


import functools


import math


class Sine(nn.Module):
    """Applies the sine function with frequency scaling element-wise:

    :math:`\\text{Sine}(x)= \\sin(\\omega * x)`

    Args:
        omega: factor used for scaling the frequency

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional dimensions
        - Output: :math:`(N, *)`, same shape as the input
    """

    def __init__(self, omega):
        super().__init__()
        self.omega = omega

    def forward(self, x):
        return torch.sin(self.omega * x)


def make_module(module):
    if isinstance(module, torch.nn.Module):
        return module
    else:
        return module()


class FullyConnectedBlock(torch.nn.Module):

    def __init__(self, dim_in, dim_out, bias=True, activation=torch.nn.ReLU):
        super().__init__()
        self.linear = torch.nn.Linear(dim_in, dim_out, bias=bias)
        self.activation = make_module(activation) if activation is not None else torch.nn.Identity()

    def forward(self, input):
        return self.activation(self.linear(input))


class FullyConnectedResidualBlock(torch.nn.Module):

    def __init__(self, dim_in, dims_hidden, dim_out, bias=True, activation_hidden=torch.nn.ReLU, activation=torch.nn.ReLU):
        super().__init__()
        self.dimensions = [dim_in] + dims_hidden + [dim_out]
        self.num_layers = len(self.dimensions) - 1
        for i in range(self.num_layers):
            if i == self.num_layers - 1:
                layer = FullyConnectedBlock(self.dimensions[i], self.dimensions[i + 1], activation=None)
            else:
                layer = FullyConnectedBlock(self.dimensions[i], self.dimensions[i + 1], activation=make_module(activation_hidden))
            self.add_module(f'Residual{i:d}', layer)
        self.shortcut = torch.nn.Identity() if dim_in == dim_out else torch.nn.Linear(dim_in, dim_out)
        self.activation = torch.nn.Identity() if activation is None else make_module(activation)

    def forward(self, input):
        Fx = input
        for i in range(self.num_layers):
            Fx = self.__getattr__(f'Residual{i:d}')(Fx)
        x = self.shortcut(input)
        return self.activation(Fx + x)


def init_weights_normal(**kwargs):
    module = kwargs['module']
    if isinstance(module, nn.Linear):
        if hasattr(module, 'weight'):
            nn.init.kaiming_normal_(module.weight, a=0.0, nonlinearity='relu', mode='fan_in')
        if hasattr(module, 'bias'):
            nn.init.zeros_(module.bias)


def init_weights_normal_last(**kwargs):
    module = kwargs['module']
    if isinstance(module, nn.Linear):
        if hasattr(module, 'weight'):
            nn.init.xavier_normal_(module.weight, gain=1)
            module.weight.data = -torch.abs(module.weight.data)
        if hasattr(module, 'bias'):
            nn.init.zeros_(module.bias)


def siren_init(**kwargs):
    module = kwargs['module']
    n = kwargs['n']
    omega = kwargs['omega']
    if isinstance(module, nn.Linear):
        module.weight.data.uniform_(-np.sqrt(6 / n) / omega, np.sqrt(6 / n) / omega)


def siren_init_first(**kwargs):
    module = kwargs['module']
    n = kwargs['n']
    if isinstance(module, nn.Linear):
        module.weight.data.uniform_(-1 / n, 1 / n)


class FC(nn.Module):

    def __init__(self, in_features, out_features, hidden_features: List[int], activation='relu', last_activation=None, bias=True, first_omega=30, hidden_omega=30.0):
        super().__init__()
        layers = []
        activations_and_inits = {'sine': (Sine(first_omega), siren_init, siren_init_first, None), 'relu': (nn.ReLU(inplace=True), init_weights_normal, init_weights_normal, None), 'relu2': (nn.ReLU(inplace=True), init_weights_normal, init_weights_normal, init_weights_normal_last), 'softplus': (nn.Softplus(), init_weights_normal, None)}
        activation_fn, weight_init, first_layer_init, last_layer_init = activations_and_inits[activation]
        layer = FullyConnectedBlock(in_features, hidden_features[0], bias=bias, activation=activation_fn)
        if first_layer_init is not None:
            layer.apply(lambda module: first_layer_init(module=module, n=in_features))
        layers.append(layer)
        for i in range(len(hidden_features)):
            n = hidden_features[i]
            layer = FullyConnectedBlock(n, n, bias=bias, activation=activation_fn)
            layer.apply(lambda module: weight_init(module=module, n=n, omega=hidden_omega))
            layers.append(layer)
        layer = FullyConnectedBlock(hidden_features[-1], out_features, bias=bias, activation=last_activation)
        layer.apply(lambda module: weight_init(module=module, n=hidden_features[-1], omega=hidden_omega))
        if last_layer_init is not None:
            layer.apply(lambda module: last_layer_init(module=module, n=in_features))
        layers.append(layer)
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class GaussianFourierFeatureTransform(torch.nn.Module):
    """ Gaussian Fourier Feature Transform

    Input: H,W,C
    Returns: H,W,mapping_size*2
    """

    def __init__(self, in_features, mapping_size=256, scale=5, device='cpu'):
        super().__init__()
        self.in_features = in_features
        self.mapping_size = mapping_size
        self.B = torch.randn((in_features, mapping_size)) * scale

    def forward(self, x):
        x = np.pi * 2 * x @ self.B
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


class Embedder:

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        if self.kwargs['log_sampling']:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.0 ** 0.0, 2.0 ** max_freq, N_freqs)
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d
        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires):
    embed_kwargs = {'include_input': True, 'input_dims': 3, 'max_freq_log2': multires - 1, 'num_freqs': multires, 'log_sampling': True, 'periodic_fns': [torch.sin, torch.cos]}
    embedder_obj = Embedder(**embed_kwargs)

    def embed(x, eo=embedder_obj):
        return eo.embed(x)
    return embed, embedder_obj.out_dim


class NeuralShader(torch.nn.Module):

    def __init__(self, hidden_features_size=256, hidden_features_layers=7, activation='relu', last_activation=None, fourier_features='none', mapping_size=256, fft_scale=10, device='cpu'):
        super().__init__()
        self.fourier_feature_transform = None
        if fourier_features == 'gfft':
            self.fourier_feature_transform = GaussianFourierFeatureTransform(3, mapping_size=mapping_size // 2, scale=fft_scale, device=device)
            self.diffuse = FC(mapping_size, hidden_features_size, [hidden_features_size] * hidden_features_layers, activation, None)
            self.specular = FC(hidden_features_size + 3 + 3, 3, [hidden_features_size // 2], activation, last_activation)
        elif fourier_features == 'positional':
            self.fourier_feature_transform, channels = get_embedder(fft_scale)
            self.diffuse = FC(channels, hidden_features_size, [hidden_features_size] * hidden_features_layers, activation, None)
            self.specular = FC(hidden_features_size + 3 + 3, 3, [hidden_features_size // 2], activation, last_activation)
        elif fourier_features == 'none':
            self.diffuse = FC(3, hidden_features_size, [hidden_features_size] * hidden_features_layers, activation, None)
            self.specular = FC(hidden_features_size + 3 + 3, 3, [hidden_features_size // 2], activation, last_activation)
        self._config = {'hidden_features_size': hidden_features_size, 'hidden_features_layers': hidden_features_layers, 'activation': activation, 'last_activation': last_activation, 'fourier_features': fourier_features, 'mapping_size': mapping_size, 'fft_scale': fft_scale}

    def forward(self, position, normal, view_dir):
        if self.fourier_feature_transform is not None:
            h = self.diffuse(self.fourier_feature_transform(position))
            h = torch.cat([h, normal, view_dir], dim=-1)
        else:
            h = self.diffuse(position)
            h = torch.cat([h, normal, view_dir], dim=-1)
        return self.specular(h)

    @classmethod
    def load(cls, path, device='cpu'):
        data = torch.load(path, map_location=device)
        version = data['version']
        shader = cls(**data['config'], device=device)
        shader.load_state_dict(data['state_dict'])
        if version < 2 and isinstance(shader.fourier_feature_transform, GaussianFourierFeatureTransform):
            None
        elif isinstance(shader.fourier_feature_transform, GaussianFourierFeatureTransform):
            shader.fourier_feature_transform.B = data['B']
        return shader

    def save(self, path):
        data = {'version': 2, 'config': self._config, 'state_dict': self.state_dict()}
        if isinstance(self.fourier_feature_transform, GaussianFourierFeatureTransform):
            data['B'] = self.fourier_feature_transform.B
        torch.save(data, path)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (FullyConnectedBlock,
     lambda: ([], {'dim_in': 4, 'dim_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GaussianFourierFeatureTransform,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Sine,
     lambda: ([], {'omega': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_fraunhoferhhi_neural_deferred_shading(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

