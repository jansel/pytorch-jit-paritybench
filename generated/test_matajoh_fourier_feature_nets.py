import sys
_module = sys.modules[__name__]
del sys
fourier1d = _module
fourier2d = _module
mesh_to_octree = _module
near_orbit = _module
fourier_feature_nets = _module
camera_info = _module
fourier_feature_models = _module
image_dataset = _module
nerf_model = _module
octree = _module
pixel_dataset = _module
ray_caster = _module
ray_dataset = _module
ray_sampler = _module
signal_dataset = _module
utils = _module
version = _module
visualizers = _module
voxels_model = _module
orbit_video = _module
setup = _module
submit_aml_run = _module
submit_param_sweep = _module
test_ray_sampling = _module
train_image_regression = _module
train_nerf = _module
train_signal_regression = _module
train_tiny_nerf = _module
train_voxels = _module
visualizations = _module
camera_to_world = _module
ray_cube_intersection = _module
rendering_equation = _module
view_angle = _module
volume_raycasting = _module
voxels_animation = _module
world_to_camera = _module
voxelize_model = _module

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


import math


from typing import List


import torch


import torch.nn as nn


from typing import Set


from typing import Union


import matplotlib.pyplot as plt


import numpy as np


from torch.utils.data import Dataset


from typing import Sequence


from typing import NamedTuple


import copy


import time


from typing import OrderedDict


from matplotlib.pyplot import get_cmap


import torch.nn.functional as F


from abc import ABC


from abc import abstractmethod


from enum import Enum


from typing import Callable


import re


import torch.optim


from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


class FourierFeatureMLP(nn.Module):
    """MLP which uses Fourier features as a preprocessing step."""

    def __init__(self, num_inputs: int, num_outputs: int, a_values: torch.Tensor, b_values: torch.Tensor, layer_channels: List[int]):
        """Constructor.

        Args:
            num_inputs (int): Number of dimensions in the input
            num_outputs (int): Number of dimensions in the output
            a_values (torch.Tensor): a values for encoding
            b_values (torch.Tensor): b values for encoding
            num_layers (int): Number of layers in the MLP
            layer_channels (List[int]): Number of channels per layer.
        """
        nn.Module.__init__(self)
        self.params = {'num_inputs': num_inputs, 'num_outputs': num_outputs, 'a_values': None if a_values is None else a_values.tolist(), 'b_values': None if b_values is None else b_values.tolist(), 'layer_channels': layer_channels}
        self.num_inputs = num_inputs
        if b_values is None:
            self.a_values = None
            self.b_values = None
            num_inputs = num_inputs
        else:
            assert b_values.shape[0] == num_inputs
            assert a_values.shape[0] == b_values.shape[1]
            self.a_values = nn.Parameter(a_values, requires_grad=False)
            self.b_values = nn.Parameter(b_values, requires_grad=False)
            num_inputs = b_values.shape[1] * 2
        self.layers = nn.ModuleList()
        for num_channels in layer_channels:
            self.layers.append(nn.Linear(num_inputs, num_channels))
            num_inputs = num_channels
        self.layers.append(nn.Linear(num_inputs, num_outputs))
        self.use_view = False
        self.keep_activations = False
        self.activations = []

    def forward(self, inputs: torch.Tensor) ->torch.Tensor:
        """Predicts outputs from the provided uv input."""
        if self.b_values is None:
            output = inputs
        else:
            encoded = math.pi * inputs @ self.b_values
            output = torch.cat([self.a_values * encoded.cos(), self.a_values * encoded.sin()], dim=-1)
        self.activations.clear()
        for layer in self.layers[:-1]:
            output = torch.relu(layer(output))
        if self.keep_activations:
            self.activations.append(output.detach().cpu().numpy())
        output = self.layers[-1](output)
        return output

    def save(self, path: str):
        """Saves the model to the specified path.

        Args:
            path (str): Path to the model file on disk
        """
        state_dict = self.state_dict()
        state_dict['type'] = 'fourier'
        state_dict['params'] = self.params
        torch.save(state_dict, path)


class MLP(FourierFeatureMLP):
    """Unencoded FFN, essentially a standard MLP."""

    def __init__(self, num_inputs: int, num_outputs: int, num_layers=3, num_channels=256):
        """Constructor.

        Args:
            num_inputs (int): Number of dimensions in the input
            num_outputs (int): Number of dimensions in the output
            num_layers (int, optional): Number of layers in the MLP.
                                        Defaults to 4.
            num_channels (int, optional): Number of channels in the MLP.
                                          Defaults to 256.
        """
        FourierFeatureMLP.__init__(self, num_inputs, num_outputs, None, None, [num_channels] * num_layers)


class BasicFourierMLP(FourierFeatureMLP):
    """Basic version of FFN in which inputs are projected onto the unit circle."""

    def __init__(self, num_inputs: int, num_outputs: int, num_layers=3, num_channels=256):
        """Constructor.

        Args:
            num_inputs (int): Number of dimensions in the input
            num_outputs (int): Number of dimensions in the output
            num_layers (int, optional): Number of layers in the MLP.
                                        Defaults to 4.
            num_channels (int, optional): Number of channels in the MLP.
                                          Defaults to 256.
        """
        a_values = torch.ones(num_inputs)
        b_values = torch.eye(num_inputs)
        FourierFeatureMLP.__init__(self, num_inputs, num_outputs, a_values, b_values, [num_channels] * num_layers)


class PositionalFourierMLP(FourierFeatureMLP):
    """Version of FFN with positional encoding."""

    def __init__(self, num_inputs: int, num_outputs: int, max_log_scale: float, num_layers=3, num_channels=256, embedding_size=256):
        """Constructor.

        Args:
            num_inputs (int): Number of dimensions in the input
            num_outputs (int): Number of dimensions in the output
            max_log_scale (float): Maximum log scale for embedding
            num_layers (int, optional): Number of layers in the MLP.
                                        Defaults to 4.
            num_channels (int, optional): Number of channels in the MLP.
                                          Defaults to 256.
            embedding_size (int, optional): The size of the feature embedding.
                                            Defaults to 256.
        """
        b_values = self._encoding(max_log_scale, embedding_size, num_inputs)
        a_values = torch.ones(b_values.shape[1])
        FourierFeatureMLP.__init__(self, num_inputs, num_outputs, a_values, b_values, [num_channels] * num_layers)

    @staticmethod
    def _encoding(max_log_scale: float, embedding_size: int, num_inputs: int):
        """Produces the encoding b_values matrix."""
        embedding_size = embedding_size // num_inputs
        frequencies_matrix = 2.0 ** torch.linspace(0, max_log_scale, embedding_size)
        frequencies_matrix = frequencies_matrix.reshape(-1, 1, 1)
        frequencies_matrix = torch.eye(num_inputs) * frequencies_matrix
        frequencies_matrix = frequencies_matrix.reshape(-1, num_inputs)
        frequencies_matrix = frequencies_matrix.transpose(0, 1)
        return frequencies_matrix


class GaussianFourierMLP(FourierFeatureMLP):
    """Version of a FFN using a full Gaussian matrix for encoding."""

    def __init__(self, num_inputs: int, num_outputs: int, sigma: float, num_layers=3, num_channels=256, embedding_size=256):
        """Constructor.

        Args:
            num_inputs (int): Number of dimensions in the input
            num_outputs (int): Number of dimensions in the output
            sigma (float): Standard deviation of the Gaussian distribution
            num_layers (int, optional): Number of layers in the MLP.
                                        Defaults to 4.
            num_channels (int, optional): Number of channels in the MLP.
                                          Defaults to 256.
            embedding_size (int, optional): Number of frequencies to use for
                                             the encoding. Defaults to 256.
        """
        b_values = torch.normal(0, sigma, size=(num_inputs, embedding_size))
        a_values = torch.ones(b_values.shape[1])
        FourierFeatureMLP.__init__(self, num_inputs, num_outputs, a_values, b_values, [num_channels] * num_layers)


class NeRF(nn.Module):
    """The full NeRF model."""

    def __init__(self, num_layers: int, num_channels: int, max_log_scale_pos: float, num_freq_pos: int, max_log_scale_view: float, num_freq_view: int, skips: Sequence[int], include_inputs: bool):
        """Constructor.

        Args:
            num_layers (int): Number of layers in the main body.
            num_channels (int): Number of channels per layer.
            max_log_scale_pos (float): The maximum log scale for the positional
                                       encoding.
            num_freq_pos (int): The number of frequences to use for encoding
                                position.
            max_log_scale_view (float): The maximum log scale for the view
                                        direction.
            num_freq_view (int): The number of frequencies to use for encoding
                                 view direction.
            skips (Sequence[int]): Skip connection layers.
            include_inputs (bool): Whether to include the inputs in the encoded
                                   input vector.
        """
        nn.Module.__init__(self)
        self.params = {'num_layers': num_layers, 'num_channels': num_channels, 'max_log_scale_pos': max_log_scale_pos, 'num_freq_pos': num_freq_pos, 'max_log_scale_view': max_log_scale_view, 'num_freq_view': num_freq_view, 'skips': list(skips), 'include_inputs': include_inputs}
        pos_encoding = self._encoding(max_log_scale_pos, num_freq_pos, 3)
        self.pos_encoding = nn.Parameter(pos_encoding, requires_grad=False)
        view_encoding = self._encoding(max_log_scale_view, num_freq_view, 3)
        self.view_encoding = nn.Parameter(view_encoding, requires_grad=False)
        self.skips = set(skips)
        self.include_inputs = include_inputs
        self.use_view = True
        self.layers = nn.ModuleList()
        num_inputs = 2 * self.pos_encoding.shape[-1]
        if self.include_inputs:
            num_inputs += 3
        layer_inputs = num_inputs
        for i in range(num_layers):
            if i in self.skips:
                layer_inputs += num_inputs
            self.layers.append(nn.Linear(layer_inputs, num_channels))
            layer_inputs = num_channels
        self.opacity_out = nn.Linear(layer_inputs, 1)
        self.bottleneck = nn.Linear(layer_inputs, num_channels)
        layer_inputs = num_channels + 2 * self.view_encoding.shape[-1]
        if self.include_inputs:
            layer_inputs += 3
        self.hidden_view = nn.Linear(layer_inputs, num_channels // 2)
        layer_inputs = num_channels // 2
        self.color_out = nn.Linear(layer_inputs, 3)

    @staticmethod
    def _encoding(max_log_scale: float, num_freq: int, num_inputs: int):
        frequencies_matrix = 2.0 ** torch.linspace(0, max_log_scale, num_freq)
        frequencies_matrix = frequencies_matrix.reshape(-1, 1, 1)
        frequencies_matrix = torch.eye(num_inputs) * frequencies_matrix
        frequencies_matrix = frequencies_matrix.reshape(-1, num_inputs)
        frequencies_matrix = frequencies_matrix.transpose(0, 1)
        return frequencies_matrix

    def forward(self, position: torch.Tensor, view: torch.Tensor) ->torch.Tensor:
        """Queries the model for the radiance field output.

        Args:
            position (torch.Tensor): a (N,3) tensor of positions.
            view (torch.Tensor): a (N,3) tensor of normalized view directions.

        Returns:
            torch.Tensor: a (N,4) tensor of color and opacity.
        """
        encoded_pos = position @ self.pos_encoding
        encoded_pos = [encoded_pos.cos(), encoded_pos.sin()]
        if self.include_inputs:
            encoded_pos.append(position)
        encoded_pos = torch.cat(encoded_pos, dim=-1)
        encoded_view = view @ self.view_encoding
        encoded_view = [encoded_view.cos(), encoded_view.sin()]
        if self.include_inputs:
            encoded_view.append(view)
        encoded_view = torch.cat(encoded_view, dim=-1)
        outputs = encoded_pos
        for i, layer in enumerate(self.layers):
            if i in self.skips:
                outputs = torch.cat([outputs, encoded_pos], dim=-1)
            outputs = torch.relu(layer(outputs))
        opacity = self.opacity_out(outputs)
        bottleneck = self.bottleneck(outputs)
        outputs = torch.cat([bottleneck, encoded_view], dim=-1)
        outputs = torch.relu(self.hidden_view(outputs))
        color = self.color_out(outputs)
        return torch.cat([color, opacity], dim=-1)

    def save(self, path: str):
        """Saves the model to the specified path.

        Args:
            path (str): Path to the model file on disk
        """
        state_dict = self.state_dict()
        state_dict['type'] = 'nerf'
        state_dict['params'] = self.params
        torch.save(state_dict, path)


class Voxels(nn.Module):
    """A voxel based radiance field model."""

    def __init__(self, side: int, scale: float):
        """Constructor.

        Args:
            side (int): The number of voxels on one side of a cube.
            scale (float): The scale of the voxel volume, equivalent
                           to half of one side of the volume, i.e. a
                           scale of 1 indicates a volume of size 2x2x2.
        """
        nn.Module.__init__(self)
        self.params = {'side': side, 'scale': scale}
        voxels = torch.zeros((1, 4, side, side, side), dtype=torch.float32)
        self.voxels = nn.Parameter(voxels)
        bias = torch.zeros(4, dtype=torch.float32)
        bias[:3] = torch.logit(torch.FloatTensor([1e-05, 1e-05, 1e-05]))
        bias[3] = -2
        self.bias = nn.Parameter(bias.unsqueeze(0))
        self.scale = scale
        self.use_view = False

    def forward(self, positions: torch.Tensor) ->torch.Tensor:
        """Interpolates the positions within the voxel volume."""
        positions = positions.reshape(1, -1, 1, 1, 3)
        positions = positions / self.scale
        output = F.grid_sample(self.voxels, positions, padding_mode='border', align_corners=False)
        output = output.transpose(1, 2)
        output = output.reshape(-1, 4)
        output = output + self.bias
        assert not output.isnan().any()
        return output

    def save(self, path: str):
        """Saves the model to the specified path.

        Args:
            path (str): Path to the model file on disk
        """
        state_dict = self.state_dict()
        state_dict['type'] = 'voxels'
        state_dict['params'] = self.params
        torch.save(state_dict, path)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicFourierMLP,
     lambda: ([], {'num_inputs': 4, 'num_outputs': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (GaussianFourierMLP,
     lambda: ([], {'num_inputs': 4, 'num_outputs': 4, 'sigma': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MLP,
     lambda: ([], {'num_inputs': 4, 'num_outputs': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (PositionalFourierMLP,
     lambda: ([], {'num_inputs': 4, 'num_outputs': 4, 'max_log_scale': 1.0}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_matajoh_fourier_feature_nets(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

