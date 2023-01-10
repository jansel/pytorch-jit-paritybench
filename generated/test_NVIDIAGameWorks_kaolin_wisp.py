import sys
_module = sys.modules[__name__]
del sys
main = _module
app_utils = _module
cuda_guard = _module
main_interactive = _module
demo_app = _module
funny_neural_field = _module
main_demo = _module
browse_spc_app = _module
main_spc_browser = _module
mesh2spc = _module
widget_spc_selector = _module
setup = _module
template_app = _module
template_main = _module
template_neural_field = _module
template_trainer = _module
test_packed_rf_tracer = _module
wisp = _module
accelstructs = _module
aabb_as = _module
octree_as = _module
config_parser = _module
core = _module
channel_fn = _module
channels = _module
colors = _module
primitives = _module
rays = _module
render_buffer = _module
transforms = _module
datasets = _module
formats = _module
nerf_standard = _module
rtmv = _module
multiview_dataset = _module
random_view_dataset = _module
sdf_dataset = _module
ray_sampler = _module
utils = _module
framework = _module
event = _module
state = _module
gfx = _module
datalayers = _module
aabb_datalayers = _module
camera_datalayers = _module
octree_datalayers = _module
models = _module
activations = _module
basic_activations = _module
conditioners = _module
basic_conditioners = _module
decoders = _module
basic_decoders = _module
embedders = _module
positional_embedder = _module
grids = _module
blas_grid = _module
codebook_grid = _module
hash_grid = _module
octree_grid = _module
triplanar_grid = _module
layers = _module
nefs = _module
base_nef = _module
nerf = _module
neural_sdf = _module
spc_field = _module
pipeline = _module
offline_renderer = _module
ops = _module
differential = _module
gradients = _module
geometric = _module
grid = _module
image = _module
io = _module
metrics = _module
processing = _module
mesh = _module
area_weighted_distribution = _module
barycentric_coordinates = _module
closest_point = _module
closest_tex = _module
compute_sdf = _module
load_obj = _module
normalize = _module
per_face_normals = _module
point_sample = _module
random_face = _module
sample_near_surface = _module
sample_surface = _module
sample_tex = _module
sample_uniform = _module
pointcloud = _module
conversions = _module
processing = _module
raygen = _module
raygen = _module
sdf = _module
metrics = _module
shaders = _module
matcap = _module
shadow_rays = _module
spc = _module
constructors = _module
conversions = _module
metrics = _module
processing = _module
sampling = _module
renderer = _module
app = _module
optimization_app = _module
wisp_app = _module
api = _module
base_renderer = _module
decorators = _module
renderers_factory = _module
scenegraph = _module
control = _module
camera_controller_mode = _module
first_person = _module
trackball = _module
turntable = _module
render_core = _module
renderers = _module
radiance_pipeline_renderer = _module
sdf_pipeline_renderer = _module
spc_pipeline_renderer = _module
gizmos = _module
gizmo = _module
ogl = _module
axis_painter = _module
primitives_painter = _module
world_grid = _module
gui = _module
imgui = _module
widget_cameras = _module
widget_dictionary_octree_grid = _module
widget_gpu_stats = _module
widget_imgui = _module
widget_object_transform = _module
widget_octree_grid = _module
widget_optimization = _module
widget_property_editor = _module
widget_radiance_pipeline = _module
widget_radiance_pipeline_renderer = _module
widget_renderer_properties = _module
widget_scene_graph = _module
widget_sdf_pipeline = _module
widget_sdf_pipeline_renderer = _module
widget_triplanar_grid = _module
tracers = _module
base_tracer = _module
packed_rf_tracer = _module
packed_sdf_tracer = _module
packed_spc_tracer = _module
sdf_tracer = _module
trainers = _module
base_trainer = _module
multiview_trainer = _module
sdf_trainer = _module
debug = _module
helper_classes = _module
perf = _module

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


import logging as log


import numpy as np


import logging


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CppExtension


from torch.utils.cpp_extension import CUDAExtension


from torch.utils.cpp_extension import CUDA_HOME


import pandas as pd


from typing import Tuple


from typing import Callable


from typing import Any


import torch.nn.functional as F


from typing import List


from typing import Optional


from typing import Union


from typing import Set


from typing import Dict


from typing import Iterator


from functools import partial


import time


from torch.multiprocessing import Pool


import copy


from torch.multiprocessing import cpu_count


from torch.utils.data import Dataset


from copy import deepcopy


import random


import collections


from torch._six import string_classes


from torch.utils.data._utils.collate import default_convert


from typing import Type


from typing import DefaultDict


from typing import TYPE_CHECKING


from collections import defaultdict


import torch.nn as nn


from scipy.stats import ortho_group


from abc import ABC


from abc import abstractmethod


import inspect


import math


from scipy.interpolate import RegularGridInterpolator


from scipy.ndimage import gaussian_filter


from collections import deque


import abc


from typing import Iterable


from torch.utils.data import DataLoader


from torch.utils.tensorboard import SummaryWriter


class SigDecoder(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim, activation, bias):
        """Initialize the SigDecoder.
        Args:
            input_dim (int): Input dimension of the MLP.
            output_dim (int): Output dimension of the MLP.
            hidden_dim (int): Hidden dimension of the MLP.
            activation (function): The activation function to use.
            bias (bool): If True, use bias.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.hidden_layer = nn.Linear(self.input_dim, hidden_dim, bias=bias)
        self.output_layer = nn.Linear(hidden_dim, self.output_dim, bias=bias)

    def forward_feature(self, x):
        """A specialized forward function for the MLP, to obtain 3 hidden channels, post sigmoid activation.
        after

        Args:
            x (torch.FloatTensor): Some tensor of shape [batch, ..., input_dim]

        Returns:
            (torch.FloatTensor): The output tensor of shape [batch, ..., 3]
        """
        x_h = self.hidden_layer(x)
        x_h[..., :3] = torch.sigmoid(x_h[..., :3])
        return x_h[..., :3]

    def forward(self, x):
        """Run the MLP!

        Args:
            x (torch.FloatTensor): Some tensor of shape [batch, ..., input_dim]

        Returns:
            (torch.FloatTensor): The output tensor of shape [batch, ..., output_dim]
        """
        x_h = self.hidden_layer(x)
        x_h[..., :3] = torch.sigmoid(x_h[..., :3])
        x_h[..., 3:] = self.activation(x_h[..., 3:])
        out = self.output_layer(x_h)
        return out


class FullSort(nn.Module):
    """The "FullSort" activation function from https://arxiv.org/abs/1811.05381.
    """

    def forward(self, x):
        """Sorts the feature dimension.
        
        Args:
            x (torch.FloatTensor): Some tensor of shape [..., feature_size]
        
        Returns:
            (torch.FloatTensor): Activation of shape [..., feature_size]
        """
        return torch.sort(x, dim=-1)[0]


class MinMax(nn.Module):
    """The "MinMax" activation function from https://arxiv.org/abs/1811.05381.
    """

    def forward(self, x):
        """Partially sorts the feature dimension.
        
        The feature dimension needs to be a multiple of 2.

        Args:
            x (torch.FloatTensor): Some tensor of shape [batch, feature_size]
        
        Returns:
            (torch.FloatTensor): Activation of shape [batch, feature_size]
        """
        N, M = x.shape
        x = x.reshape(N, M // 2, 2)
        return torch.cat([x.min(-1, keepdim=True)[0], x.max(-1, keepdim=True)[0]], dim=-1).reshape(N, M)


class Identity(nn.Module):
    """Identity function. Occasionally useful.
    """

    def forward(self, x):
        """Returns the input. :)

        Args:
            x (Any): Anything

        Returns:
            (Any): The input!
        """
        return x


class BasicDecoder(nn.Module):
    """Super basic but super useful MLP class.
    """

    def __init__(self, input_dim, output_dim, activation, bias, layer=nn.Linear, num_layers=1, hidden_dim=128, skip=[]):
        """Initialize the BasicDecoder.

        Args:
            input_dim (int): Input dimension of the MLP.
            output_dim (int): Output dimension of the MLP.
            activation (function): The activation function to use.
            bias (bool): If True, use bias.
            layer (nn.Module): The MLP layer module to use.
            num_layers (int): The number of hidden layers in the MLP.
            hidden_dim (int): The hidden dimension of the MLP.
            skip (List[int]): List of layer indices where the input dimension is concatenated.

        Returns:
            (void): Initializes the class.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.bias = bias
        self.layer = layer
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.skip = skip
        if self.skip is None:
            self.skip = []
        self.make()

    def make(self):
        """Builds the actual MLP.
        """
        layers = []
        for i in range(self.num_layers):
            if i == 0:
                layers.append(self.layer(self.input_dim, self.hidden_dim, bias=self.bias))
            elif i in self.skip:
                layers.append(self.layer(self.hidden_dim + input_dim, self.hidden_dim, bias=self.bias))
            else:
                layers.append(self.layer(self.hidden_dim, self.hidden_dim, bias=self.bias))
        self.layers = nn.ModuleList(layers)
        self.lout = self.layer(self.hidden_dim, self.output_dim, bias=self.bias)

    def forward(self, x, return_h=False):
        """Run the MLP!

        Args:
            x (torch.FloatTensor): Some tensor of shape [batch, ..., input_dim]
            return_h (bool): If True, also returns the last hidden layer.

        Returns:
            (torch.FloatTensor, (optional) torch.FloatTensor):
                - The output tensor of shape [batch, ..., output_dim]
                - The last hidden layer of shape [batch, ..., hidden_dim]
        """
        N = x.shape[0]
        for i, l in enumerate(self.layers):
            if i == 0:
                h = self.activation(l(x))
            elif i in self.skip:
                h = self.activation(l(h))
                h = torch.cat([x, h], dim=-1)
            else:
                h = self.activation(l(h))
        out = self.lout(h)
        if return_h:
            return out, h
        else:
            return out

    def initialize(self, get_weight):
        """Initializes the MLP layers with some initialization functions.

        Args:
            get_weight (function): A function which returns a matrix given a matrix.

        Returns:
            (void): Initializes the layer weights.
        """
        ms = []
        for i, w in enumerate(self.layers):
            m = get_weight(w.weight)
            ms.append(m)
        for i in range(len(self.layers)):
            self.layers[i].weight = nn.Parameter(ms[i])
        m = get_weight(self.lout.weight)
        self.lout.weight = nn.Parameter(m)


class PositionalEmbedder(nn.Module):
    """PyTorch implementation of regular positional embedding, as used in the original NeRF and Transformer papers.
    """

    def __init__(self, num_freq, max_freq_log2, log_sampling=True, include_input=True, input_dim=3):
        """Initialize the module.

        Args:
            num_freq (int): The number of frequency bands to sample. 
            max_freq_log2 (int): The maximum frequency.
                                 The bands will be sampled at regular intervals in [0, 2^max_freq_log2].
            log_sampling (bool): If true, will sample frequency bands in log space.
            include_input (bool): If true, will concatenate the input.
            input_dim (int): The dimension of the input coordinate space.

        Returns:
            (void): Initializes the encoding.
        """
        super().__init__()
        self.num_freq = num_freq
        self.max_freq_log2 = max_freq_log2
        self.log_sampling = log_sampling
        self.include_input = include_input
        self.out_dim = 0
        if include_input:
            self.out_dim += input_dim
        if self.log_sampling:
            self.bands = 2.0 ** torch.linspace(0.0, max_freq_log2, steps=num_freq)
        else:
            self.bands = torch.linspace(1, 2.0 ** max_freq_log2, steps=num_freq)
        self.out_dim += self.bands.shape[0] * input_dim * 2
        self.bands = nn.Parameter(self.bands).requires_grad_(False)

    def forward(self, coords):
        """Embeds the coordinates.

        Args:
            coords (torch.FloatTensor): Coordinates of shape [N, input_dim]

        Returns:
            (torch.FloatTensor): Embeddings of shape [N, input_dim + out_dim] or [N, out_dim].
        """
        N = coords.shape[0]
        winded = (coords[:, None] * self.bands[None, :, None]).reshape(N, coords.shape[1] * self.num_freq)
        encoded = torch.cat([torch.sin(winded), torch.cos(winded)], dim=-1)
        if self.include_input:
            encoded = torch.cat([coords, encoded], dim=-1)
        return encoded


class BLASGrid(nn.Module, ABC):
    """
    BLASGrids (commonly referred in documentation as simply "grids"), represent feature grids in Wisp.
    BLAS: "Bottom Level Acceleration Structure", to signify this structure is the backbone that captures
    a neural field's contents, in terms of both features and occupancy for speeding up queries.

    This is an abstract base class that uses some spatial acceleration structure under the hood, to speed up operations
    such as coordinate based queries or ray tracing.
    Classes which inherit the BLASGrid are generally compatible with BaseTracers to support such operations
    (see: raymarch(), raytrace(), query()).

    Grids are usually employed as building blocks within neural fields (see: BaseNeuralField),
    possibly paired with decoders to form a neural field.
    """

    def raymarch(self, *args, **kwargs):
        """By default, this function will use the equivalent BLAS function unless overridden for custom behaviour.
        """
        return self.blas.raymarch(*args, **kwargs)

    def raytrace(self, *args, **kwargs):
        """By default, this function will use the equivalent BLAS function unless overridden for custom behaviour.
        """
        return self.blas.raytrace(*args, **kwargs)

    def query(self, *args, **kwargs):
        """By default, this function will use the equivalent BLAS function unless overridden for custom behaviour.
        """
        return self.blas.query(*args, **kwargs)

    @abstractmethod
    def interpolate(self, coords, lod_idx):
        """ Interpolates a feature value for the given coords using the grid support, in the given lod_idx
        Args:
            coords (torch.FloatTensor): coords of shape [batch, num_samples, 3] or [batch, 3]
            lod_idx  (int): int specifying the index to the desired level of detail, if supported.
        """
        raise NotImplementedError('A BLASGrid should implement the interpolation functionality according to the grid structure.')

    def name(self) ->str:
        """
        Returns:
            (str) A BLASGrid should be given a meaningful, human readable name.
        """
        return type(self).__name__


class TriplanarFeatureVolume(nn.Module):
    """Triplanar feature volume represents a single triplane, e.g. a single LOD in a TriplanarGrid. """

    def __init__(self, fdim, fsize, std, bias):
        """Initializes the feature triplane.

        Args:
            fdim (int): The feature dimension.
            fsize (int): The height and width of the texture map.
            std (float): The standard deviation for the Gaussian initialization.
            bias (float): The mean for the Gaussian initialization.
        """
        super().__init__()
        self.fsize = fsize
        self.fdim = fdim
        self.fmx = nn.Parameter(torch.randn(1, fdim, fsize + 1, fsize + 1) * std + bias)
        self.fmy = nn.Parameter(torch.randn(1, fdim, fsize + 1, fsize + 1) * std + bias)
        self.fmz = nn.Parameter(torch.randn(1, fdim, fsize + 1, fsize + 1) * std + bias)
        self.padding_mode = 'reflection'

    def forward(self, x):
        """Interpolates from the feature volume.

        Args:
            x (torch.FloatTensor): Coordinates of shape [batch, num_samples, 3] or [batch, 3].

        Returns:
            (torch.FloatTensor): Features of shape [batch, num_samples, fdim] or [batch, fdim].
        """
        N = x.shape[0]
        if len(x.shape) == 3:
            sample_coords = x.reshape(1, N, x.shape[1], 3)
            samplex = F.grid_sample(self.fmx, sample_coords[..., [1, 2]], align_corners=True, padding_mode=self.padding_mode)[0, :, :, :].transpose(0, 1)
            sampley = F.grid_sample(self.fmy, sample_coords[..., [0, 2]], align_corners=True, padding_mode=self.padding_mode)[0, :, :, :].transpose(0, 1)
            samplez = F.grid_sample(self.fmz, sample_coords[..., [0, 1]], align_corners=True, padding_mode=self.padding_mode)[0, :, :, :].transpose(0, 1)
            sample = torch.stack([samplex, sampley, samplez], dim=1).permute(0, 3, 1, 2)
        else:
            sample_coords = x.reshape(1, N, 1, 3)
            samplex = F.grid_sample(self.fmx, sample_coords[..., [1, 2]], align_corners=True, padding_mode=self.padding_mode)[0, :, :, 0].transpose(0, 1)
            sampley = F.grid_sample(self.fmy, sample_coords[..., [0, 2]], align_corners=True, padding_modes=self.padding_mode)[0, :, :, 0].transpose(0, 1)
            samplez = F.grid_sample(self.fmz, sample_coords[..., [0, 1]], align_corners=True, padding_mode=self.padding_mode)[0, :, :, 0].transpose(0, 1)
            sample = torch.stack([samplex, sampley, samplez], dim=1)
        return sample


class TriplanarGrid(BLASGrid):
    """A feature grid where the features are stored on a multiresolution pyramid of triplanes.
    Each LOD consists of a triplane, e.g. a triplet of orthogonal planes.

    The shape of the triplanar feature grid means the support region is bounded by an AABB,
    therefore spatial queries / ray tracing ops can use an AABB as an acceleration structure.
    Hence the class is compatible with BaseTracer implementations.
    """

    def __init__(self, feature_dim: int, base_lod: int, num_lods: int=1, interpolation_type: str='linear', multiscale_type: str='sum', feature_std: float=0.0, feature_bias: float=0.0):
        """Constructs an instance of a TriplanarGrid.

        Args:
            feature_dim (int): The dimension of the features stored on the grid.
            base_lod (int): The base LOD of the feature grid. This is the lowest LOD of the SPC octree
                            for which features are defined.
            num_lods (int): The number of LODs for which features are defined. Starts at base_lod.
            interpolation_type (str): The type of interpolation function.
            multiscale_type (str): The type of multiscale aggregation. Usually 'sum' or 'cat'.
                                   Note that 'cat' will change the decoder input dimension.
            feature_std (float): The features are initialized with a Gaussian distribution with the given
                                 standard deviation.
            feature_bias (float): The mean of the Gaussian distribution.

        Returns:
            (void): Initializes the class.
        """
        super().__init__()
        self.feature_dim = feature_dim * 3
        self.base_lod = base_lod
        self.num_lods = num_lods
        self.interpolation_type = interpolation_type
        self.multiscale_type = multiscale_type
        self.feature_std = feature_std
        self.feature_bias = feature_bias
        self.active_lods = [(self.base_lod + x) for x in range(self.num_lods)]
        self.max_lod = self.num_lods + self.base_lod - 1
        log.info(f'Active LODs: {self.active_lods}')
        self.blas = AxisAlignedBBoxAS()
        self.init_feature_structure()

    def init_feature_structure(self):
        """ Initializes everything related to the features stored in the triplanar grid structure. """
        self.features = nn.ModuleList([])
        self.num_feat = 0
        for i in self.active_lods:
            self.features.append(TriplanarFeatureVolume(self.feature_dim // 3, 2 ** i, self.feature_std, self.feature_bias))
            self.num_feat += (2 ** i + 1) ** 2 * self.feature_dim * 3
        log.info(f'# Feature Vectors: {self.num_feat}')

    def freeze(self):
        """Freezes the feature grid.
        """
        self.features.requires_grad_(False)

    def interpolate(self, coords, lod_idx):
        """Query multiscale features.

        Args:
            coords (torch.FloatTensor): coords of shape [batch, num_samples, 3] or [batch, 3]
            lod_idx  (int): int specifying the index to ``active_lods``

        Returns:
            (torch.FloatTensor): interpolated features of
            shape [batch, num_samples, feature_dim] or [batch, feature_dim]
        """
        output_shape = coords.shape[:-1]
        if coords.ndim < 3:
            coords = coords[:, None]
        feats = []
        for i in range(lod_idx + 1):
            feats.append(self._interpolate(coords, self.features[i], i))
        feats = torch.cat(feats, dim=-1)
        if self.multiscale_type == 'sum':
            feats = feats.reshape(*output_shape, lod_idx + 1, feats.shape[-1] // (lod_idx + 1)).sum(-2)
        return feats

    def _interpolate(self, coords, feats, lod_idx):
        """Interpolates the given feature using the coordinates x.

        This is a more low level interface for optimization.

        Inputs:
            coords     : float tensor of shape [batch, num_samples, 3]
            feats : float tensor of shape [num_feats, feat_dim]
            lod_idx   : int specifying the lod
        Returns:
            float tensor of shape [batch, num_samples, feat_dim]
        """
        batch, num_samples = coords.shape[:2]
        if self.interpolation_type == 'linear':
            fs = feats(coords).reshape(batch, num_samples, 3 * feats.fdim)
        else:
            raise ValueError(f"Interpolation mode '{self.interpolation_type}' is not supported")
        return fs

    def raymarch(self, rays, raymarch_type, num_samples, level=None):
        """Mostly a wrapper over OctreeAS.raymarch. See corresponding function for more details.

        Important detail: this is just used as an AABB tracer.
        """
        return self.blas.raymarch(rays, raymarch_type=raymarch_type, num_samples=num_samples, level=0)

    def raytrace(self, rays, level=None, with_exit=False):
        """By default, this function will use the equivalent BLAS function unless overridden for custom behaviour.

        Important detail: this is just used as an AABB tracer.
        """
        return self.blas.raytrace(rays, level=0, with_exit=with_exit)

    def name(self) ->str:
        return 'Triplanar Grid'


def normalize_frobenius(x):
    """Normalizes the matrix according to the Frobenius norm.

    Args:
        x (torch.FloatTensor): A matrix.

    Returns:
        (torch.FloatTensor): A normalized matrix.
    """
    norm = torch.sqrt((torch.abs(x) ** 2).sum())
    return x / norm


class FrobeniusLinear(nn.Module):
    """A standard Linear layer which applies a Frobenius normalization in the forward pass.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.linear = nn.Linear(*args, **kwargs)

    def forward(self, x):
        weight = normalize_frobenius(self.linear.weight)
        return F.linear(x, weight, self.linear.bias)


def normalize_L_1(x):
    """Normalizes the matrix according to the L1 norm.

    Args:
        x (torch.FloatTensor): A matrix.

    Returns:
        (torch.FloatTensor): A normalized matrix.
    """
    abscolsum = torch.sum(torch.abs(x), dim=0)
    abscolsum = torch.min(torch.stack([1.0 / abscolsum, torch.ones_like(abscolsum)], dim=0), dim=0)[0]
    return x * abscolsum[None, :]


class L_1_Linear(nn.Module):
    """A standard Linear layer which applies a L1 normalization in the forward pass.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.linear = nn.Linear(*args, **kwargs)

    def forward(self, x):
        weight = normalize_L_1(self.linear.weight)
        return F.linear(x, weight, self.linear.bias)


def normalize_L_inf(x):
    """Normalizes the matrix according to the Linf norm.

    Args:
        x (torch.FloatTensor): A matrix.

    Returns:
        (torch.FloatTensor): A normalized matrix.
    """
    absrowsum = torch.sum(torch.abs(x), axis=1)
    absrowsum = torch.min(torch.stack([1.0 / absrowsum, torch.ones_like(absrowsum)], dim=0), dim=0)[0]
    return x * absrowsum[:, None]


class L_inf_Linear(nn.Module):
    """A standard Linear layer which applies a Linf normalization in the forward pass.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.linear = nn.Linear(*args, **kwargs)

    def forward(self, x):
        weight = normalize_L_inf(self.linear.weight)
        return F.linear(x, weight, self.linear.bias)


class BaseNeuralField(nn.Module):
    """The base class for all Neural Fields within Wisp.
    Neural Fields are defined as modules which take coordinates as input and output signals of some form.
    The term "Neural" is loosely used here to imply these modules are generally subject for optimization.

    The domain of neural fields in Wisp is flexible, and left up for the user to decide when implementing the subclass.
    Popular neural fields from the literature, such as Neural Radiance Fields (Mildenhall et al. 2020),
    and Neural Signed Distance Functions (SDFs) can be implemented by creating and registering
    the required forward functions (for i.e. rgb, density, sdf values).

    BaseNeuralField subclasses  usually consist of several optional components:
    - A feature grid (BLASGrid), sometimes also known as 'hybrid representations'.
      These are responsible for querying and interpolating features, often in the context of some 3D volume
      (but not limited to).
      Feature grids often employ some acceleration structure (i.e. OctreeAS),
      which can be used to accelerate spatial queries or raytracing ops,
      hence the term "BLAS" (Bottom Level Acceleration Structure).
    - A decoder (i.e. BasicDecoder) which can feeds on features (or coordinates / pos embeddings) and coverts
      them to output signals.
    - Other components such as positional embedders may be employed.

    BaseNeuralFields are generally meant to be compatible with BaseTracers, thus forming a complete pipeline of
    render-able neural primitives.
    """

    def __init__(self):
        super().__init__()
        self._forward_functions = {}
        self.register_forward_functions()
        self.supported_channels = set([channel for channels in self._forward_functions.values() for channel in channels])

    @property
    def device(self):
        """ Returns the device used to process inputs in this neural field.
        By default, the device is queried from the first registered torch nn.parameter.
        Override this property to explicitly specify the device.

        Returns:
            (torch.device): The expected device for inputs to this neural field.
        """
        return next(self.parameters()).device

    def _register_forward_function(self, fn, channels):
        """Registers a forward function.

        Args:
            fn (function): Function to register.
            channels (list of str): Channel output names.
        """
        if isinstance(channels, str):
            channels = [channels]
        self._forward_functions[fn] = set(channels)

    @abstractmethod
    def register_forward_functions(self):
        """Register forward functions with the channels that they output.
        
        This function should be overrided and call `self._register_forward_function` to 
        tell the class which functions output what output channels. The function can be called
        multiple times to register multiple functions.

        Example:

        ```
        self._register_forward_function(self.rgba, ["density", "rgb"])
        self._register_forward_function(self.sdf, ["sdf"])
        ```
        """
        pass

    def get_forward_function(self, channel):
        """Will return the function that will return the channel.
        
        Args: 
            channel (str): The name of the channel to return.

        Returns:
            (function): Function that will return the function. Will return None if the channel is not supported.
        """
        if channel not in self.get_supported_channels():
            raise Exception(f'Channel {channel} is not supported in {self.__class__.__name__}')
        for fn in self._forward_functions:
            output_channels = self._forward_functions[fn]
            if channel in output_channels:
                return lambda *args, **kwargs: fn(*args, **kwargs)[channel]

    def get_supported_channels(self):
        """Returns the channels that are supported by this class.

        Returns:
            (set): Set of channel strings.
        """
        return self.supported_channels

    def forward(self, channels=None, **kwargs):
        """Queries the neural field with channels.

        Args:
            channels (str or list of str or set of str): Requested channels. See return value for details.
            kwargs: Any keyword argument passed in will be passed into the respective forward functions.

        Returns:
            (list or dict or torch.Tensor): 
                If channels is a string, will return a tensor of the request channel. 
                If channels is a list, will return a list of channels.
                If channels is a set, will return a dictionary of channels.
                If channels is None, will return a dictionary of all channels.
        """
        if not (isinstance(channels, str) or isinstance(channels, list) or isinstance(channels, set) or channels is None):
            raise Exception(f'Channels type invalid, got {type(channels)}.Make sure your arguments for the nef are provided as keyword arguments.')
        if channels is None:
            requested_channels = self.get_supported_channels()
        elif isinstance(channels, str):
            requested_channels = set([channels])
        else:
            requested_channels = set(channels)
        unsupported_channels = requested_channels - self.get_supported_channels()
        if unsupported_channels:
            raise Exception(f'Channels {unsupported_channels} are not supported in {self.__class__.__name__}')
        return_dict = {}
        for fn in self._forward_functions:
            output_channels = self._forward_functions[fn]
            supported_channels = output_channels & requested_channels
            if len(supported_channels) != 0:
                argspec = inspect.getfullargspec(fn)
                required_args = argspec.args[:-len(argspec.defaults)][1:]
                optional_args = argspec.args[-len(argspec.defaults):]
                input_args = {}
                for _arg in required_args:
                    if _arg not in kwargs:
                        raise Exception(f'Argument {_arg} not found as input to in {self.__class__.__name__}.{fn.__name__}()')
                    input_args[_arg] = kwargs[_arg]
                for _arg in optional_args:
                    if _arg in kwargs:
                        input_args[_arg] = kwargs[_arg]
                output = fn(**input_args)
                for channel in supported_channels:
                    return_dict[channel] = output[channel]
        if isinstance(channels, str):
            if channels in return_dict:
                return return_dict[channels]
            else:
                return None
        elif isinstance(channels, list):
            return [return_dict[channel] for channel in channels]
        else:
            return return_dict


def get_activation_class(activation_type):
    """Utility function to return an activation function class based on the string description.

    Args:
        activation_type (str): The name for the activation function.
    
    Returns:
        (Function): The activation function to be used. 
    """
    if activation_type == 'none':
        return Identity()
    elif activation_type == 'fullsort':
        return FullSort()
    elif activation_type == 'minmax':
        return MinMax()
    elif activation_type == 'relu':
        return torch.relu
    elif activation_type == 'sin':
        return torch.sin
    else:
        assert False and 'activation type does not exist'


def spectral_norm_(*args, **kwargs):
    """Initializes a spectral norm layer.
    """
    return nn.utils.spectral_norm(nn.Linear(*args, **kwargs))


def get_layer_class(layer_type):
    """Convenience function to return the layer class name from text.

    Args:
        layer_type (str): Text name for the layer.

    Retunrs:
        (nn.Module): The layer to be used for the decoder.
    """
    if layer_type == 'none' or layer_type == 'linear':
        return nn.Linear
    elif layer_type == 'spectral_norm':
        return spectral_norm_
    elif layer_type == 'frobenius_norm':
        return FrobeniusLinear
    elif layer_type == 'l_1_norm':
        return L_1_Linear
    elif layer_type == 'l_inf_norm':
        return L_inf_Linear
    else:
        assert False and 'layer type does not exist'


def get_positional_embedder(frequencies, input_dim=3, include_input=True):
    """Utility function to get a positional encoding embedding.

    Args:
        frequencies (int): The number of frequencies used to define the PE:
            [2^0, 2^1, 2^2, ... 2^(frequencies - 1)].
        input_dim (int): The input coordinate dimension.
        include_input (bool): If true, will concatenate the input coords.

    Returns:
        (nn.Module, int):
        - The embedding module
        - The output dimension of the embedding.
    """
    encoder = PositionalEmbedder(frequencies, frequencies - 1, input_dim=input_dim, include_input=include_input)
    return encoder, encoder.out_dim


def sample_unif_sphere(n):
    """Sample uniformly random points on a sphere.
    
    Args:
        n (int): Number of samples.

    Returns:
        (np.array): Positions of shape [n, 3]
    """
    u = np.random.rand(2, n)
    z = 1 - 2 * u[0, :]
    r = np.sqrt(1.0 - z * z)
    phi = 2 * np.pi * u[1, :]
    xyz = np.array([r * np.cos(phi), r * np.sin(phi), z]).transpose()
    return xyz


class NeuralRadianceField(BaseNeuralField):
    """Model for encoding Neural Radiance Fields (Mildenhall et al. 2020), e.g., density and view dependent color.
    Different to the original NeRF paper, this implementation uses feature grids for a
    higher quality and more efficient implementation, following later trends in the literature,
    such as Neural Sparse Voxel Fields (Liu et al. 2020), Instant Neural Graphics Primitives (Muller et al. 2022)
    and Variable Bitrate Neural Fields (Takikawa et al. 2022).
    """

    def __init__(self, grid: BLASGrid=None, pos_embedder: str='none', view_embedder: str='none', pos_multires: int=10, view_multires: int=4, position_input: bool=False, activation_type: str='relu', layer_type: str='none', hidden_dim: int=128, num_layers: int=1, prune_density_decay: float=None, prune_min_density: float=None):
        super().__init__()
        self.grid = grid
        self.position_input = position_input
        if self.position_input:
            self.pos_embedder, self.pos_embed_dim = self.init_embedder(pos_embedder, pos_multires)
        else:
            self.pos_embedder, self.pos_embed_dim = None, 0
        self.view_embedder, self.view_embed_dim = self.init_embedder(view_embedder, view_multires)
        self.activation_type = activation_type
        self.layer_type = layer_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.decoder_density, self.decoder_color = self.init_decoders(activation_type, layer_type, num_layers, hidden_dim)
        self.prune_density_decay = prune_density_decay
        self.prune_min_density = prune_min_density
        torch.cuda.empty_cache()

    def init_embedder(self, embedder_type, frequencies=None):
        """Creates positional embedding functions for the position and view direction.
        """
        if embedder_type == 'none':
            embedder, embed_dim = None, 0
        elif embedder_type == 'identity':
            embedder, embed_dim = torch.nn.Identity(), 0
        elif embedder_type == 'positional':
            embedder, embed_dim = get_positional_embedder(frequencies=frequencies)
        else:
            raise NotImplementedError(f'Unsupported embedder type for NeuralRadianceField: {embedder_type}')
        return embedder, embed_dim

    def init_decoders(self, activation_type, layer_type, num_layers, hidden_dim):
        """Initializes the decoder object.
        """
        decoder_density = BasicDecoder(input_dim=self.density_net_input_dim, output_dim=16, activation=get_activation_class(activation_type), bias=True, layer=get_layer_class(layer_type), num_layers=num_layers, hidden_dim=hidden_dim, skip=[])
        decoder_density.lout.bias.data[0] = 1.0
        decoder_color = BasicDecoder(input_dim=self.color_net_input_dim, output_dim=3, activation=get_activation_class(activation_type), bias=True, layer=get_layer_class(layer_type), num_layers=num_layers + 1, hidden_dim=hidden_dim, skip=[])
        return decoder_density, decoder_color

    def prune(self):
        """Prunes the blas based on current state.
        """
        if self.grid is not None:
            if isinstance(self.grid, HashGrid):
                density_decay = self.prune_density_decay
                min_density = self.prune_min_density
                self.grid.occupancy = self.grid.occupancy
                self.grid.occupancy = self.grid.occupancy * density_decay
                points = self.grid.dense_points
                res = 2.0 ** self.grid.blas_level
                samples = torch.rand(points.shape[0], 3, device=points.device)
                samples = points.float() + samples
                samples = samples / res
                samples = samples * 2.0 - 1.0
                sample_views = torch.FloatTensor(sample_unif_sphere(samples.shape[0]))
                with torch.no_grad():
                    density = self.forward(coords=samples, ray_d=sample_views, channels='density')
                self.grid.occupancy = torch.stack([density[:, 0], self.grid.occupancy], -1).max(dim=-1)[0]
                mask = self.grid.occupancy > min_density
                _points = points[mask]
                if _points.shape[0] == 0:
                    return
                octree = spc_ops.unbatched_points_to_octree(_points, self.grid.blas_level, sorted=True)
                self.grid.blas = OctreeAS(octree)
            else:
                raise NotImplementedError(f'Pruning not implemented for grid type {self.grid}')

    def register_forward_functions(self):
        """Register the forward functions.
        """
        self._register_forward_function(self.rgba, ['density', 'rgb'])

    def rgba(self, coords, ray_d, lod_idx=None):
        """Compute color and density [particles / vol] for the provided coordinates.

        Args:
            coords (torch.FloatTensor): tensor of shape [batch, 3]
            ray_d (torch.FloatTensor): tensor of shape [batch, 3]
            lod_idx (int): index into active_lods. If None, will use the maximum LOD.
        
        Returns:
            {"rgb": torch.FloatTensor, "density": torch.FloatTensor}:
                - RGB tensor of shape [batch, 3]
                - Density tensor of shape [batch, 1]
        """
        if lod_idx is None:
            lod_idx = len(self.grid.active_lods) - 1
        batch, _ = coords.shape
        feats = self.grid.interpolate(coords, lod_idx).reshape(-1, self.effective_feature_dim)
        if self.pos_embedder is not None:
            embedded_pos = self.pos_embedder(coords).view(-1, self.pos_embed_dim)
            feats = torch.cat([feats, embedded_pos], dim=-1)
        density_feats = self.decoder_density(feats)
        if self.view_embedder is not None:
            embedded_dir = self.view_embedder(-ray_d).view(-1, self.view_embed_dim)
            fdir = torch.cat([density_feats, embedded_dir], dim=-1)
        else:
            fdir = density_feats
        colors = torch.sigmoid(self.decoder_color(fdir))
        density = torch.relu(density_feats[..., 0:1])
        return dict(rgb=colors, density=density)

    @property
    def effective_feature_dim(self):
        if self.grid.multiscale_type == 'cat':
            effective_feature_dim = self.grid.feature_dim * self.grid.num_lods
        else:
            effective_feature_dim = self.grid.feature_dim
        return effective_feature_dim

    @property
    def density_net_input_dim(self):
        return self.effective_feature_dim + self.pos_embed_dim

    @property
    def color_net_input_dim(self):
        return 16 + self.view_embed_dim


class NeuralSDF(BaseNeuralField):
    """Model for encoding neural signed distance functions (implicit surfaces).
    This field implementation uses feature grids for faster and more efficient queries.
    For example, the usage of Octree follows the idea from Takikawa et al. 2021 (Neural Geometric Level of Detail).
    """

    def __init__(self, grid: BLASGrid=None, pos_embedder: str='none', pos_multires: int=10, position_input: bool=True, activation_type: str='relu', layer_type: str='none', hidden_dim: int=128, num_layers: int=1):
        super().__init__()
        self.grid = grid
        self.pos_multires = pos_multires
        self.position_input = position_input
        self.pos_embedder, self.pos_embed_dim = self.init_embedder(pos_embedder, pos_multires, position_input)
        self.activation_type = activation_type
        self.layer_type = layer_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.decoder = self.init_decoder(activation_type, layer_type, num_layers, hidden_dim)
        torch.cuda.empty_cache()

    def init_embedder(self, embedder_type, frequencies=None, position_input=True):
        """Creates positional embedding functions for the position and view direction.
        """
        if embedder_type == 'none':
            embedder, embed_dim = None, 0
        elif embedder_type == 'identity':
            embedder, embed_dim = torch.nn.Identity(), 0
        elif embedder_type == 'positional':
            embedder, embed_dim = get_positional_embedder(frequencies=frequencies, position_input=position_input)
        else:
            raise NotImplementedError(f'Unsupported embedder type for NeuralSDF: {embedder_type}')
        return embedder, embed_dim

    def init_decoder(self, activation_type, layer_type, num_layers, hidden_dim):
        """Initializes the decoder object.
        """
        decoder = BasicDecoder(input_dim=self.decoder_input_dim, output_dim=1, activation=get_activation_class(activation_type), bias=True, layer=get_layer_class(layer_type), num_layers=num_layers, hidden_dim=hidden_dim, skip=[])
        return decoder

    def register_forward_functions(self):
        """Register the forward functions.
        """
        self._register_forward_function(self.sdf, ['sdf'])

    def sdf(self, coords, lod_idx=None):
        """Computes the Signed Distance Function for input samples.

        Args:
            coords (torch.FloatTensor): tensor of shape [batch, num_samples, 3]
            lod_idx (int): index into active_lods. If None, will use the maximum LOD.
        
        Outputs:
            (torch.FloatTensor):
            - SDF of shape [batch, num_samples, 1]
        """
        shape = coords.shape
        if shape[0] == 0:
            return dict(sdf=torch.zeros_like(coords)[..., 0:1])
        if lod_idx is None:
            lod_idx = self.grid.num_lods - 1
        if len(shape) == 2:
            coords = coords[:, None]
        num_samples = coords.shape[1]
        feats = self.grid.interpolate(coords, lod_idx)
        if self.pos_embedder is not None:
            feats = torch.cat([self.pos_embedder(coords.view(-1, 3)).view(-1, num_samples, self.pos_embed_dim), feats], dim=-1)
        sdf = self.decoder(feats)
        if len(shape) == 2:
            sdf = sdf[:, 0]
        return dict(sdf=sdf)

    @property
    def effective_feature_dim(self):
        if self.grid.multiscale_type == 'cat':
            effective_feature_dim = self.grid.feature_dim * self.grid.num_lods
        else:
            effective_feature_dim = self.grid.feature_dim
        return effective_feature_dim

    @property
    def decoder_input_dim(self):
        input_dim = self.effective_feature_dim
        if self.position_input:
            input_dim += self.pos_embed_dim
        return input_dim


class SPCField(BaseNeuralField):
    """ A field based on Structured Point Clouds (SPC) from kaolin.
    SPC is a hierarchical compressed data structure, which can be interpreted in various ways:
    * Quantized point cloud, where each sparse point is quantized to some (possibly very dense) grid.
      Each point is associated with some feature(s).
    * An Octree, where each cell center is represented by a quantized point.
    Throughout wisp, SPCs are used to implement efficient octrees or grid structures.
    This field class allows wisp to render SPCs directly with their feature content (hence no embedders or decoders
    are assumed).

    When rendered, SPCs behave like octrees which allow for efficient tracing.
    Feature samples per ray may be collected from each intersected "cell" of the structured point cloud.
    """

    def __init__(self, spc_octree, features_dict=None, device='cuda', base_lod=None, num_lods=None, optimizable=False, **kwargs):
        """
        Creates a new Structured Point Cloud (SPC), represented as a Wisp Field.
        In wisp, SPCs are considered neural fields, since their features may be optimized.
        See `examples/spc_browser` for an elaborate description of SPCs.

        Args:
            spc_octree (torch.ByteTensor):
                A tensor which holds the topology of the SPC.
                Each byte represents a single octree cell's occupancy (that is, each bit of that byte represents
                the occupancy status of a child octree cell), yielding 8 bits for 8 cells.
                See also https://kaolin.readthedocs.io/en/latest/notes/spc_summary.html
            features_dict (dict):
                A dictionary holding the features information of the SPC.
                Keys are assumed to be a subset of ('colors', 'normals').
                Values are torch feature tensors containing information per point, of shape
                :math:`(	ext{num_points}, 	ext{feat_dim})`.
                Where `num_points` is the number of occupied cells in the SPC.
                See `kaolin.ops.conversions.pointcloud.unbatched_pointcloud_to_spc` for conversion of point
                cloud information to such features.
            device (torch.device):
                Torch device on which the features and topology of the SPC field will be stored.
            base_lod (int):
                Number of levels of detail without features the SPC will use.
            num_lods (int):
                Number of levels of detail with features the SPC will use.
                The total number of levels the SPC will have is `base_lod + num_lods - 1`.
            optimizable (bool):
                A flag which determines if this SPCField supports optimization or not.
                Toggling optimization off allows for quick creation of SPCField objects.
        """
        self.spc_octree = spc_octree
        self.features_dict = features_dict if features_dict is not None else dict()
        self.spc_device = device
        self.base_lod = base_lod
        self.num_lods = num_lods
        self.optimizable = optimizable
        self.grid = None
        self.colors = None
        self.normals = None
        self.init_grid(spc_octree)
        super().__init__(grid=self.grid, **kwargs)

    def init_grid(self, spc_octree):
        """ Uses the OctreeAS / OctreeGrid mechanism to quickly parse the SPC object into a Wisp Neural Field.

        Args:
            spc_octree (torch.ByteTensor):
                A tensor which holds the topology of the SPC.
                Each byte represents a single octree cell's occupancy (that is, each bit of that byte represents
                the occupancy status of a child octree cell), yielding 8 bits for 8 cells.
                See also https://kaolin.readthedocs.io/en/latest/notes/spc_summary.html
        """
        spc_features = self.features_dict
        if 'colors' in self.features_dict:
            colors = spc_features['colors']
            colors = colors.reshape(-1, 4) / 255.0
            self.colors = colors
        if 'normals' in self.features_dict:
            normals = spc_features['normals']
            normals = normals.reshape(-1, 3)
            self.normals = normals
        if self.colors is None:
            if self.normals is not None:
                colors = 0.5 * (normals + 1.0)
            else:
                lengths = torch.tensor([len(spc_octree)], dtype=torch.int32)
                level, pyramids, exsum = kaolin_ops_spc.scan_octrees(spc_octree, lengths)
                point_hierarchies = kaolin_ops_spc.generate_points(spc_octree, pyramids, exsum)
                colors = point_hierarchies[pyramids[0, 1, level]:]
                colors = colors / np.power(2, level)
            self.colors = colors
        _, pyramid, _ = wisp_spc_ops.octree_to_spc(spc_octree)
        max_level = pyramid.shape[-1] - 2
        self.grid = OctreeGrid.from_spc(spc_octree=spc_octree, feature_dim=3, base_lod=max_level, num_lods=0)

    @property
    def device(self):
        """ Returns the device used to process inputs in this neural field.

        Returns:
            (torch.device): The expected device used for this Structured Point Cloud.
        """
        return self.spc_device

    def register_forward_functions(self):
        """Register the forward functions.
        """
        self._register_forward_function(self.rgba, ['rgb'])

    def rgba(self, ridx_hit=None):
        """Compute color for the provided ray hits.

        Args:
            ridx_hit (torch.LongTensor): SPC point_hierarchy indices of shape [batch].
                                     used to indicate index of first hit voxel.

        Returns:
            {"rgb": torch.FloatTensor}:
                - RGB tensor of shape [batch, 1, 3]
        """
        level = self.grid.blas.max_level
        offset = self.grid.blas.pyramid[1, level]
        ridx_hit = ridx_hit - offset
        colors = self.colors[ridx_hit, :3].unsqueeze(1)
        return dict(rgb=colors)


class BaseTracer(nn.Module, ABC):
    """Base class for all tracers within Wisp.
    Tracers drive the mapping process which takes an input "Neural Field", and outputs a RenderBuffer of pixels.
    Different tracers may employ different algorithms for querying points, or tracing / marching rays through the
    neural field.
    A common paradigm for tracers to employ is as follows:
    1. Take input in the form of rays or coordinates
    2. Generate samples by tracing / marching rays, or querying coordinates over the neural field.
       Possibly make use of the neural field spatial structure for high performance.
    2. Invoke neural field's methods to decode sample features into actual channel values, such as color, density,
       signed distance, and so forth.
    3. Aggregate the sample values to decide on the final pixel value.
       The exact output may depend on the requested channel type, blending mode or other parameters.
    Wisp tracers are therefore flexible, and designed to be compatible with specific neural fields,
    depending on the forward functions they support and internal grid structures they use.
    Tracers are generally expected to be differentiable (e.g. they're part of the training loop),
    though non-differentiable tracers are also allowed.
    """

    def __init__(self):
        """Initializes the tracer class and sets the default arguments for trace.
        This should be overrided and called if you want to pass custom defaults into the renderer.
        If overridden, it should keep the arguments to `self.trace` in `self.` class variables.
        Then, if these variables exist and no function arguments are passed into forward,
        it will override them as the default.
        """
        super().__init__()

    @abstractmethod
    def get_supported_channels(self):
        """Returns the set of channel names this tracer may output.

        Implement the function to return the supported channels, e.g.       
        return set(["depth", "rgb"])

        Returns:
            (set): Set of channel strings.
        """
        pass

    @abstractmethod
    def get_required_nef_channels(self):
        """Returns the channels required by neural fields to be compatible with this tracer.
        
        Implement the function to return the required channels, e.g.
        return set(["rgb", "density"])

        Returns:
            (set): Set of channel strings.
        """
        pass

    @abstractmethod
    def trace(self, nef, channels, extra_channels, *args, **kwargs):
        """Apply the forward map on the nef. 

        This is the function to implement to implement a custom
        This can take any number of arguments, but `nef` always needs to be the first argument and 
        `channels` needs to be the second argument.
        
        Args:
            nef (nn.Module): A neural field that uses a grid class.
            channels (set): The set of requested channels. The trace method can return channels that 
                            were not requested since those channels often had to be computed anyways.
            extra_channels (set): Requested extra channels, which are not first class channels supported by
                the tracer but will still be able to handle with some fallback options.

        Returns:
            (wisp.RenderBuffer): A dataclass which holds the output buffers from the tracer.
        """
        pass

    def forward(self, nef, channels=None, **kwargs):
        """Queries the tracer with channels.

        Args:
            channels (str or list of str or set of str): Requested channels.
            kwargs: Any keyword argument passed in will be passed into the respective forward functions.

        Returns:
            (wisp.RenderBuffer): A dataclass which holds the output buffers from the tracer.
        """
        nef_channels = nef.get_supported_channels()
        unsupported_inputs = self.get_required_nef_channels() - nef_channels
        if unsupported_inputs:
            raise Exception(f'The neural field class {type(nef)} does not output the required channels {unsupported_inputs}.')
        if channels is None:
            requested_channels = self.get_supported_channels()
        elif isinstance(channels, str):
            requested_channels = set([channels])
        else:
            requested_channels = set(channels)
        extra_channels = requested_channels - self.get_supported_channels()
        unsupported_outputs = extra_channels - nef_channels
        if unsupported_outputs:
            raise Exception(f'Channels {unsupported_outputs} are not supported in the tracer {type(self)} or neural field {type(nef)}.')
        if extra_channels is None:
            requested_extra_channels = set()
        elif isinstance(extra_channels, str):
            requested_extra_channels = set([extra_channels])
        else:
            requested_extra_channels = set(extra_channels)
        argspec = inspect.getfullargspec(self.trace)
        required_args = argspec.args[:-len(argspec.defaults)][4:]
        optional_args = argspec.args[-len(argspec.defaults):]
        input_args = {}
        for _arg in required_args:
            if _arg not in kwargs:
                raise Exception(f'Argument {_arg} not found as input to in {type(self)}.trace()')
            input_args[_arg] = kwargs[_arg]
        for _arg in optional_args:
            if _arg in kwargs:
                input_args[_arg] = kwargs[_arg]
            else:
                default_arg = getattr(self, _arg, None)
                if default_arg is not None:
                    input_args[_arg] = default_arg
        return self.trace(nef, requested_channels, requested_extra_channels, **input_args)


class Pipeline(nn.Module):
    """Base class for implementing neural field pipelines.

    Pipelines consist of several components:

        - Neural fields (``self.nef``) which take coordinates as input and outputs signals.
          These usually consist of several optional components:

            - A feature grid (``self.nef.grid``)
              Sometimes also known as 'hybrid representations'.
            - An acceleration structure (``self.nef.grid.blas``) which can be used to accelerate spatial queries.
            - A decoder (``self.net.decoder``) which can take the features (or coordinates, or embeddings) and covert it to signals.

        - A forward map (``self.tracer``) which is a function which will invoke the pipeline in
          some outer loop. Usually this consists of renderers which will output a RenderBuffer object.
    
    The 'Pipeline' classes are responsible for holding and orchestrating these components.
    """

    def __init__(self, nef: BaseNeuralField, tracer: BaseTracer=None):
        """Initialize the Pipeline.

        Args:
            nef (nn.Module): Neural fields module.
            tracer (nn.Module or None): Forward map module.
        """
        super().__init__()
        self.nef: BaseNeuralField = nef
        self.tracer: BaseTracer = tracer

    def forward(self, *args, **kwargs):
        """The forward function will use the tracer (the forward model) if one is available. 
        
        Otherwise, it'll execute the neural field.
        """
        if self.tracer is not None:
            return self.tracer(self.nef, *args, **kwargs)
        else:
            return self.nef(*args, **kwargs)


BlendFunction = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


NormalizeFunction = Callable[[torch.Tensor, Any, Any], torch.Tensor]


def normalize(V: torch.Tensor, F: torch.Tensor, mode: str):
    """Normalizes a mesh.

    Args:
        V (torch.FloatTensor): Vertices of shape [V, 3]
        F (torch.LongTensor): Faces of shape [F, 3]
        mode (str): Different methods of normalization.

    Returns:
        (torch.FloatTensor, torch.LongTensor):
        - Normalized Vertices
        - Faces
    """
    if mode == 'sphere':
        V_max, _ = torch.max(V, dim=0)
        V_min, _ = torch.min(V, dim=0)
        V_center = (V_max + V_min) / 2.0
        V = V - V_center
        max_dist = torch.sqrt(torch.max(torch.sum(V ** 2, dim=-1)))
        V_scale = 1.0 / max_dist
        V *= V_scale
        return V, F
    elif mode == 'aabb':
        V_min, _ = torch.min(V, dim=0)
        V = V - V_min
        max_dist = torch.max(V)
        V *= 1.0 / max_dist
        V = V * 2.0 - 1.0
        return V, F
    elif mode == 'planar':
        V_min, _ = torch.min(V, dim=0)
        V = V - V_min
        x_max = torch.max(V[..., 0])
        z_max = torch.max(V[..., 2])
        V[..., 0] *= 1.0 / x_max
        V[..., 2] *= 1.0 / z_max
        max_dist = torch.max(V)
        V[..., 1] *= 1.0 / max_dist
        V = V * 2.0 - 1.0
        y_min = torch.min(V[..., 1])
        V[..., 1] -= y_min
        return V, F
    elif mode == 'none':
        return V, F


__RB_VARIANTS__ = dict()


def blend_alpha_composite_over(c1: torch.Tensor, c2: torch.Tensor, alpha1: torch.Tensor, alpha2: torch.Tensor):
    """ An alpha compositing op where a front pixel is alpha blended with the background pixel
    (in a usual painter's algorithm manner).
    Useful for blending channels such as RGB.
    See: https://en.wikipedia.org/wiki/Alpha_compositing

    Args:
        c1 (torch.Tensor): first channel tensor of an arbitrary shape.
        c2 (torch.Tensor): second channel tensor, in the shape of c1.
        alpha1 (torch.Tensor): alpha channel tensor, corresponding to first channel, in the shape of c1.
        alpha2 (torch.Tensor): alpha channel tensor, corresponding to second channel, in the shape of c1.

    Returns:
        (torch.Tensor): Blended channel in the shape of c1
    """
    alpha_out = alpha1 + alpha2 * (1.0 - alpha1)
    c_out = torch.where(condition=alpha_out > 0, input=(c1 * alpha1 + c2 * alpha2 * (1.0 - alpha1)) / alpha_out, other=torch.zeros_like(c1))
    return c_out


class PackedRFTracer(BaseTracer):
    """Tracer class for sparse (packed) radiance fields.
    - Packed: each ray yields a custom number of samples, which are therefore packed in a flat form within a tensor,
     see: https://kaolin.readthedocs.io/en/latest/modules/kaolin.ops.batch.html#packed
    - RF: Radiance Field
    PackedRFTracer is differentiable, and can be employed within training loops.

    This tracer class expects the neural field to expose a BLASGrid: a Bottom-Level-Acceleration-Structure Grid,
    i.e. a grid that inherits the BLASGrid class for both a feature structure and an occupancy acceleration structure).
    """

    def __init__(self, raymarch_type='voxel', num_steps=64, step_size=1.0, bg_color='white', **kwargs):
        """Set the default trace() arguments.

        Args:
            raymarch_type (str): Sample generation strategy to use for raymarch.
                'voxel' - intersects the rays with the acceleration structure cells.
                    Then among the intersected cells, each cell is sampled `num_steps` times.
                'ray' - samples `num_steps` along each ray, and then filters out samples which falls outside of occupied
                    cells of the acceleration structure.
            num_steps (int): The number of steps to use for the sampling. The meaning of this parameter changes
                depending on `raymarch_type`:
                'voxel' - each acceleration structure cell which intersects a ray is sampled `num_steps` times.
                'ray' - number of samples generated per ray, before culling away samples which don't fall
                    within occupied cells.
                The exact number of samples generated, therefore, depends on this parameter but also the occupancy
                status of the acceleration structure.
            step_size (float): The step size between samples. Currently unused, but will be used for a new
                               sampling method in the future.
            bg_color (str): The background color to use.
        """
        super().__init__()
        self.raymarch_type = raymarch_type
        self.num_steps = num_steps
        self.step_size = step_size
        self.bg_color = bg_color

    def get_supported_channels(self):
        """Returns the set of channel names this tracer may output.
        
        Returns:
            (set): Set of channel strings.
        """
        return {'depth', 'hit', 'rgb', 'alpha'}

    def get_required_nef_channels(self):
        """Returns the channels required by neural fields to be compatible with this tracer.
        
        Returns:
            (set): Set of channel strings.
        """
        return {'rgb', 'density'}

    def trace(self, nef, channels, extra_channels, rays, lod_idx=None, raymarch_type='voxel', num_steps=64, step_size=1.0, bg_color='white'):
        """Trace the rays against the neural field.

        Args:
            nef (nn.Module): A neural field that uses a grid class.
            channels (set): The set of requested channels. The trace method can return channels that 
                            were not requested since those channels often had to be computed anyways.
            extra_channels (set): If there are any extra channels requested, this tracer will by default
                                  perform volumetric integration on those channels.
            rays (wisp.core.Rays): Ray origins and directions of shape [N, 3]
            lod_idx (int): LOD index to render at. 
            raymarch_type (str): The type of raymarching algorithm to use. Currently we support:
                                 voxel: Finds num_steps # of samples per intersected voxel
                                 ray: Finds num_steps # of samples per ray, and filters them by intersected samples
            num_steps (int): The number of steps to use for the sampling.
            step_size (float): The step size between samples. Currently unused, but will be used for a new
                               sampling method in the future.
            bg_color (str): The background color to use. TODO(ttakikawa): Might be able to simplify / remove

        Returns:
            (wisp.RenderBuffer): A dataclass which holds the output buffers from the render.
        """
        assert nef.grid is not None and 'this tracer requires a grid'
        N = rays.origins.shape[0]
        if 'depth' in channels:
            depth = torch.zeros(N, 1, device=rays.origins.device)
        else:
            depth = None
        if bg_color == 'white':
            rgb = torch.ones(N, 3, device=rays.origins.device)
        else:
            rgb = torch.zeros(N, 3, device=rays.origins.device)
        hit = torch.zeros(N, device=rays.origins.device, dtype=torch.bool)
        out_alpha = torch.zeros(N, 1, device=rays.origins.device)
        if lod_idx is None:
            lod_idx = nef.grid.num_lods - 1
        raymarch_results = nef.grid.raymarch(rays, level=nef.grid.active_lods[lod_idx], num_samples=num_steps, raymarch_type=raymarch_type)
        ridx, samples, depths, deltas, boundary = raymarch_results
        ridx_hit = ridx[spc_render.mark_pack_boundaries(ridx.int())]
        hit_ray_d = rays.dirs.index_select(0, ridx)
        color, density = nef(coords=samples, ray_d=hit_ray_d, lod_idx=lod_idx, channels=['rgb', 'density'])
        density = density.reshape(-1, 1)
        del ridx
        tau = density * deltas
        del density, deltas
        ray_colors, transmittance = spc_render.exponential_integration(color, tau, boundary, exclusive=True)
        if 'depth' in channels:
            ray_depth = spc_render.sum_reduce(depths.reshape(-1, 1) * transmittance, boundary)
            depth[ridx_hit, :] = ray_depth
        alpha = spc_render.sum_reduce(transmittance, boundary)
        out_alpha[ridx_hit] = alpha
        hit[ridx_hit] = alpha[..., 0] > 0.0
        if bg_color == 'white':
            color = 1.0 - alpha + ray_colors
        else:
            color = alpha * ray_colors
        rgb[ridx_hit] = color
        extra_outputs = {}
        for channel in extra_channels:
            feats = nef(coords=samples, ray_d=hit_ray_d, lod_idx=lod_idx, channels=channel)
            num_channels = feats.shape[-1]
            ray_feats, transmittance = spc_render.exponential_integration(feats.view(-1, num_channels), tau, boundary, exclusive=True)
            composited_feats = alpha * ray_feats
            out_feats = torch.zeros(N, num_channels, device=feats.device)
            out_feats[ridx_hit] = composited_feats
            extra_outputs[channel] = out_feats
        return RenderBuffer(depth=depth, hit=hit, rgb=rgb, alpha=out_alpha, **extra_outputs)


class bcolors:
    HEADER = '\x1b[95m'
    OKBLUE = '\x1b[94m'
    OKGREEN = '\x1b[92m'
    WARNING = '\x1b[93m'
    FAIL = '\x1b[91m'
    ENDC = '\x1b[0m'
    BOLD = '\x1b[1m'
    UNDERLINE = '\x1b[4m'


def colorize_time(elapsed):
    """Returns colors based on the significance of the time elapsed.
    """
    if elapsed > 0.001:
        return bcolors.FAIL + '{:.3e}'.format(elapsed) + bcolors.ENDC
    elif elapsed > 0.0001:
        return bcolors.WARNING + '{:.3e}'.format(elapsed) + bcolors.ENDC
    elif elapsed > 1e-05:
        return bcolors.OKBLUE + '{:.3e}'.format(elapsed) + bcolors.ENDC
    else:
        return '{:.3e}'.format(elapsed)


class PerfTimer:
    """Super simple performance timer.
    """

    def __init__(self, activate=False, show_memory=False, print_mode=True):
        self.prev_time = time.process_time()
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.prev_time_gpu = self.start.record()
        self.counter = 0
        self.activate = activate
        self.show_memory = show_memory
        self.print_mode = print_mode

    def reset(self):
        self.counter = 0
        self.prev_time = time.process_time()
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.prev_time_gpu = self.start.record()

    def check(self, name=None):
        if self.activate:
            cpu_time = time.process_time() - self.prev_time
            self.end.record()
            torch.cuda.synchronize()
            gpu_time = self.start.elapsed_time(self.end) / 1000.0
            if self.print_mode:
                cpu_time_disp = colorize_time(cpu_time)
                gpu_time_disp = colorize_time(gpu_time)
                if name:
                    None
                    None
                else:
                    None
                    None
                if self.show_memory:
                    None
            self.prev_time = time.process_time()
            self.prev_time_gpu = self.start.record()
            self.counter += 1
            return cpu_time, gpu_time


def find_depth_bound(query, nug_depth, info, curr_idxes=None):
    """Associate query points to the closest depth bound in-order.
    
    TODO: Document the input.
    """
    if curr_idxes is None:
        curr_idxes = torch.nonzero(info).contiguous()
    return _C.render.find_depth_bound_cuda(query.contiguous(), curr_idxes.contiguous(), nug_depth.contiguous())


def finitediff_gradient(x, f, eps=0.005):
    """Compute 3D gradient using finite difference.

    Args:
        x (torch.FloatTensor): Coordinate tensor of shape [..., 3]
        f (nn.Module): The function to perform autodiff on.
    """
    eps_x = torch.tensor([eps, 0.0, 0.0], device=x.device)
    eps_y = torch.tensor([0.0, eps, 0.0], device=x.device)
    eps_z = torch.tensor([0.0, 0.0, eps], device=x.device)
    grad = torch.cat([f(x + eps_x) - f(x - eps_x), f(x + eps_y) - f(x - eps_y), f(x + eps_z) - f(x - eps_z)], dim=-1)
    grad = grad / (eps * 2.0)
    return grad


class PackedSDFTracer(BaseTracer):
    """Tracer class for sparse SDFs.

    - Packed: each ray yields a custom number of samples, which are therefore packed in a flat form within a tensor,
     see: https://kaolin.readthedocs.io/en/latest/modules/kaolin.ops.batch.html#packed
    - SDF: Signed Distance Function
    PackedSDFTracer is non-differentiable, and follows the sphere-tracer implementation of
    Neural Geometric Level of Detail (Takikawa et al. 2021).

    This tracer class expects the neural field to expose a BLASGrid: a Bottom-Level-Acceleration-Structure Grid,
    i.e. a grid that inherits the BLASGrid class for both a feature structure and an occupancy acceleration structure).
    """

    def __init__(self, num_steps=64, step_size=1.0, min_dis=0.0001, **kwargs):
        """Set the default trace() arguments. """
        super().__init__()
        self.num_steps = num_steps
        self.step_size = step_size
        self.min_dis = min_dis

    def get_supported_channels(self):
        """Returns the set of channel names this tracer may output.
        
        Returns:
            (set): Set of channel strings.
        """
        return {'depth', 'normal', 'xyz', 'hit', 'rgb', 'alpha'}

    def get_required_nef_channels(self):
        """Returns the channels required by neural fields to be compatible with this tracer.
        
        Returns:
            (set): Set of channel strings.
        """
        return {'sdf'}

    def trace(self, nef, channels, extra_channels, rays, lod_idx=None, num_steps=64, step_size=1.0, min_dis=0.0001):
        """Trace the rays against the neural field.

        Args:
            nef (nn.Module): A neural field that uses a grid class.
            channels (set): The set of requested channels. The trace method can return channels that 
                            were not requested since those channels often had to be computed anyways.
            extra_channels (set): If there are any extra channels requested, this tracer will by default
                                  query those extra channels at surface intersection points.
            rays (wisp.core.Rays): Ray origins and directions of shape [N, 3]
            lod_idx (int): LOD index to render at. 
            num_steps (int): The number of steps to use for sphere tracing.
            step_size (float): The multiplier for the sphere tracing steps. 
                               Use a value <1.0 for conservative tracing.
            min_dis (float): The termination distance for sphere tracing.

        Returns:
            (wisp.RenderBuffer): A dataclass which holds the output buffers from the render.
        """
        assert nef.grid is not None and 'this tracer requires a grid'
        if lod_idx is None:
            lod_idx = nef.grid.num_lods - 1
        timer = PerfTimer(activate=False)
        invres = 1.0
        ridx, pidx, depth = nef.grid.raytrace(rays, nef.grid.active_lods[lod_idx], with_exit=True)
        depth[..., 0:1] += 1e-05
        first_hit = spc_render.mark_pack_boundaries(ridx)
        curr_idxes = torch.nonzero(first_hit)[..., 0].int()
        first_ridx = ridx[first_hit].long()
        nug_o = rays.origins[first_ridx]
        nug_d = rays.dirs[first_ridx]
        mask = torch.ones([first_ridx.shape[0]], device=nug_o.device).bool()
        hit = torch.zeros_like(mask).bool()
        t = depth[first_hit][..., 0:1]
        x = torch.addcmul(nug_o, nug_d, t)
        dist = torch.zeros_like(t)
        curr_pidx = pidx[first_hit].long()
        timer.check('initial')
        with torch.no_grad():
            dist[mask] = nef(coords=x[mask], lod_idx=lod_idx, pidx=curr_pidx[mask], channels='sdf') * invres * step_size
            dist[~mask] = 20
            dist_prev = dist.clone()
            timer.check('first')
            for i in range(num_steps):
                t += dist
                x = torch.where(mask.view(mask.shape[0], 1), torch.addcmul(nug_o, nug_d, t), x)
                hit = torch.where(mask, torch.abs(dist)[..., 0] < min_dis * invres, hit)
                hit |= torch.where(mask, torch.abs(dist + dist_prev)[..., 0] * 0.5 < min_dis * 5 * invres, hit)
                mask = torch.where(mask, (t < rays.dist_max)[..., 0], mask)
                mask &= ~hit
                if not mask.any():
                    break
                dist_prev = torch.where(mask.view(mask.shape[0], 1), dist, dist_prev)
                next_idxes = find_depth_bound(t, depth, first_hit, curr_idxes=curr_idxes)
                mask &= next_idxes != -1
                aabb_mask = next_idxes != curr_idxes
                curr_idxes = torch.where(mask, next_idxes, curr_idxes)
                t = torch.where((mask & aabb_mask).view(mask.shape[0], 1), depth[curr_idxes.long(), 0:1], t)
                x = torch.where(mask.view(mask.shape[0], 1), torch.addcmul(nug_o, nug_d, t), x)
                curr_pidx = torch.where(mask, pidx[curr_idxes.long()].long(), curr_pidx)
                if not mask.any():
                    break
                dist[mask] = nef(coords=x[mask], lod_idx=lod_idx, pidx=curr_pidx[mask], channels='sdf') * invres * step_size
            timer.check('step done')
        x_buffer = torch.zeros_like(rays.origins)
        depth_buffer = torch.zeros_like(rays.origins[..., 0:1])
        hit_buffer = torch.zeros_like(rays.origins[..., 0]).bool()
        normal_buffer = torch.zeros_like(rays.origins)
        rgb_buffer = torch.zeros(*rays.origins.shape[:-1], 3, device=rays.origins.device)
        alpha_buffer = torch.zeros(*rays.origins.shape[:-1], 1, device=rays.origins.device)
        hit_buffer[first_ridx] = hit
        extra_outputs = {}
        for channel in extra_channels:
            feats = nef(coords=x[hit], lod_idx=lod_idx, channels=channel)
            extra_buffer = torch.zeros(*rays.origins.shape[:-1], feats.shape[-1], device=feats.device)
            extra_buffer[hit_buffer] = feats
        x_buffer[hit_buffer] = x[hit]
        depth_buffer[hit_buffer] = t[hit]
        if 'rgb' in channels or 'normal' in channels:
            grad = finitediff_gradient(x[hit], nef.get_forward_function('sdf'))
            normal_buffer[hit_buffer] = F.normalize(grad, p=2, dim=-1, eps=1e-05)
            rgb_buffer[..., :3] = (normal_buffer + 1.0) / 2.0
        alpha_buffer[hit_buffer] = 1.0
        timer.check('populate buffers')
        return RenderBuffer(xyz=x_buffer, depth=depth_buffer, hit=hit_buffer, normal=normal_buffer, rgb=rgb_buffer, alpha=alpha_buffer, **extra_outputs)


class PackedSPCTracer(BaseTracer):
    """Tracer class for sparse point clouds (packed rays).
    The logic of this tracer is straightforward and does not involve any neural operations:
    rays are intersected against the SPC points (cell centers).
    Each ray returns the color of the intersected cell, if such exists.

    See: https://github.com/NVIDIAGameWorks/kaolin-wisp/tree/main/examples/spc_browser
    See also: https://kaolin.readthedocs.io/en/latest/notes/spc_summary.html#spc
    """

    def __init__(self, **kwargs):
        """Set the default trace() arguments. """
        super().__init__()

    def get_supported_channels(self):
        """Returns the set of channel names this tracer may output.

        Returns:
            (set): Set of channel strings.
        """
        return {'depth', 'hit', 'rgb', 'alpha'}

    def get_required_nef_channels(self):
        """Returns the channels required by neural fields to be compatible with this tracer.

        Returns:
            (set): Set of channel strings.
        """
        return {'rgb'}

    def trace(self, nef, channels, extra_channels, rays, lod_idx=None):
        """Trace the rays against the neural field.

        Args:
            nef (nn.Module): A neural field that uses a grid class.
            channels (set): The set of requested channels. The trace method can return channels that
                            were not requested since those channels often had to be computed anyways.
            rays (wisp.core.Rays): Ray origins and directions of shape [N, 3]
            lod_idx (int): LOD index to render at.

        Returns:
            (wisp.RenderBuffer): A dataclass which holds the output buffers from the render.
        """
        timer = PerfTimer(activate=False, show_memory=False)
        N = rays.origins.shape[0]
        if lod_idx is None:
            lod_idx = nef.grid.blas.max_level
        ridx, pidx, depths = nef.grid.blas.raytrace(rays, lod_idx, with_exit=False)
        timer.check('Raytrace')
        first_hits_mask = spc_render.mark_pack_boundaries(ridx)
        first_hits_point = pidx[first_hits_mask]
        first_hits_ray = ridx[first_hits_mask]
        first_hits_depth = depths[first_hits_mask]
        color = nef(ridx_hit=first_hits_point.long(), channels='rgb')
        timer.check('RGBA')
        del ridx, pidx, rays
        ray_colors = color.squeeze(1)
        ray_depth = first_hits_depth
        depth = torch.zeros(N, 1, device=ray_depth.device)
        depth[first_hits_ray.long(), :] = ray_depth
        alpha = torch.ones([color.shape[0], 1], device=color.device)
        hit = torch.zeros(N, device=color.device).bool()
        rgb = torch.zeros(N, 3, device=color.device)
        out_alpha = torch.zeros(N, 1, device=color.device)
        color = alpha * ray_colors
        hit[first_hits_ray.long()] = alpha[..., 0] > 0.0
        rgb[first_hits_ray.long(), :3] = color
        out_alpha[first_hits_ray.long()] = alpha
        timer.check('Composit')
        return RenderBuffer(depth=depth, hit=hit, rgb=rgb, alpha=out_alpha)


class SDFTracer(BaseTracer):

    def __init__(self, num_steps=64, step_size=1.0, min_dis=0.0001, raymarch_type='voxel', **kwargs):
        """Set the default trace() arguments. """
        super().__init__(**kwargs)
        self.raymarch_type = raymarch_type
        self.num_steps = num_steps
        self.step_size = step_size
        self.min_dis = min_dis

    def get_supported_channels(self):
        """Returns the set of channel names this tracer may output.
        
        Returns:
            (set): Set of channel strings.
        """
        return {'depth', 'normal', 'xyz', 'hit'}

    def get_required_nef_channels(self):
        """Returns the channels required by neural fields to be compatible with this tracer.
        
        Returns:
            (set): Set of channel strings.
        """
        return {'sdf'}

    def trace(self, nef, channels, extra_channels, rays, num_steps=64, step_size=1.0, min_dis=0.0001):
        """Trace the rays against the neural field.

        Args:
            nef (nn.Module): A neural field that uses a grid class.
            channels (set): The set of requested channels. The trace method can return channels that 
                            were not requested since those channels often had to be computed anyways.
            extra_channels (set): If there are any extra channels requested, this tracer will by default
                                  query those extra channels at surface intersection points.
            rays (wisp.core.Rays): Ray origins and directions of shape [N, 3]
            num_steps (int): The number of steps to use for sphere tracing.
            step_size (float): The multiplier for the sphere tracing steps. 
                               Use a value <1.0 for conservative tracing.
            min_dis (float): The termination distance for sphere tracing.

        Returns:
            (wisp.RenderBuffer): A dataclass which holds the output buffers from the render.
        """
        timer = PerfTimer(activate=False)
        t = torch.zeros(rays.origins.shape[0], 1, device=rays.origins.device)
        x = torch.addcmul(rays.origins, rays.dirs, t)
        cond = torch.ones_like(t).bool()[:, 0]
        normal = torch.zeros_like(x)
        with torch.no_grad():
            d = nef(coords=x, channels='sdf')
            dprev = d.clone()
            hit = torch.zeros_like(d).byte()
            for i in range(num_steps):
                timer.check('start')
                hit = (torch.abs(t) < rays.dist_max)[:, 0]
                cond = cond & (torch.abs(d) > min_dis)[:, 0]
                cond = cond & (torch.abs((d + dprev) / 2.0) > min_dis * 3)[:, 0]
                cond = cond & hit
                if not cond.any():
                    break
                x = torch.where(cond.view(cond.shape[0], 1), torch.addcmul(rays.origins, rays.dirs, t), x)
                dprev = torch.where(cond.unsqueeze(1), d, dprev)
                d[cond] = nef(coords=x[cond], channels='sdf') * step_size
                t = torch.where(cond.view(cond.shape[0], 1), t + d, t)
                timer.check('end')
        hit = hit & ~(torch.abs(x) > 1.0).any(dim=-1)
        extra_outputs = {}
        for channel in extra_channels:
            feats = nef(coords=x[hit], lod_idx=lod_idx, channels=channel)
            extra_buffer = torch.zeros(*x.shape[:-1], feats.shape[-1], device=feats.device)
            extra_buffer[hit] = feats
            extra_outputs[channel] = extra_buffer
        if 'normal' in channels:
            if hit.any():
                grad = finitediff_gradient(x[hit], nef.get_forward_function('sdf'))
                _normal = F.normalize(grad, p=2, dim=-1, eps=1e-05)
                normal[hit] = _normal
        else:
            normal = None
        return RenderBuffer(xyz=x, depth=t, hit=hit, normal=normal, **extra_outputs)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BaseNeuralField,
     lambda: ([], {}),
     lambda: ([], {}),
     False),
    (BasicDecoder,
     lambda: ([], {'input_dim': 4, 'output_dim': 4, 'activation': _mock_layer(), 'bias': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (FrobeniusLinear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FullSort,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Identity,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (L_1_Linear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (L_inf_Linear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MinMax,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (Pipeline,
     lambda: ([], {'nef': _mock_layer()}),
     lambda: ([], {'input': torch.rand([4, 4])}),
     False),
    (PositionalEmbedder,
     lambda: ([], {'num_freq': 4, 'max_freq_log2': 4}),
     lambda: ([torch.rand([4, 16])], {}),
     True),
    (SigDecoder,
     lambda: ([], {'input_dim': 4, 'output_dim': 4, 'hidden_dim': 4, 'activation': _mock_layer(), 'bias': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_NVIDIAGameWorks_kaolin_wisp(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

    def test_004(self):
        self._check(*TESTCASES[4])

    def test_005(self):
        self._check(*TESTCASES[5])

    def test_006(self):
        self._check(*TESTCASES[6])

    def test_007(self):
        self._check(*TESTCASES[7])

    def test_008(self):
        self._check(*TESTCASES[8])

    def test_009(self):
        self._check(*TESTCASES[9])

    def test_010(self):
        self._check(*TESTCASES[10])

