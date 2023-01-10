import sys
_module = sys.modules[__name__]
del sys
conf = _module
datasets = _module
dnerf_synthetic = _module
nerf_360_v2 = _module
nerf_synthetic = _module
utils = _module
radiance_fields = _module
mlp = _module
ngp = _module
train_mlp_dnerf = _module
train_mlp_nerf = _module
train_ngp_nerf = _module
train_ngp_nerf_proposal = _module
utils = _module
nerfacc = _module
cdf = _module
contraction = _module
cuda = _module
_backend = _module
grid = _module
intersection = _module
losses = _module
pack = _module
ray_marching = _module
sampling = _module
version = _module
vol_rendering = _module
run_dev_checks = _module
run_profiler = _module
test_contraction = _module
test_grid = _module
test_intersection = _module
test_loss = _module
test_pack = _module
test_pdf_query = _module
test_ray_marching = _module
test_rendering = _module
test_resampling = _module

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


import numpy as np


import torch


import torch.nn.functional as F


import collections


import functools


import math


from typing import Callable


from typing import Optional


import torch.nn as nn


from typing import List


from typing import Union


from torch.autograd import Function


from torch.cuda.amp import custom_bwd


from torch.cuda.amp import custom_fwd


import time


import random


from typing import Tuple


from torch import Tensor


from enum import Enum


from torch.utils.cpp_extension import _get_build_directory


from torch.utils.cpp_extension import load


from typing import overload


class MLP(nn.Module):

    def __init__(self, input_dim: int, output_dim: int=None, net_depth: int=8, net_width: int=256, skip_layer: int=4, hidden_init: Callable=nn.init.xavier_uniform_, hidden_activation: Callable=nn.ReLU(), output_enabled: bool=True, output_init: Optional[Callable]=nn.init.xavier_uniform_, output_activation: Optional[Callable]=nn.Identity(), bias_enabled: bool=True, bias_init: Callable=nn.init.zeros_):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.net_depth = net_depth
        self.net_width = net_width
        self.skip_layer = skip_layer
        self.hidden_init = hidden_init
        self.hidden_activation = hidden_activation
        self.output_enabled = output_enabled
        self.output_init = output_init
        self.output_activation = output_activation
        self.bias_enabled = bias_enabled
        self.bias_init = bias_init
        self.hidden_layers = nn.ModuleList()
        in_features = self.input_dim
        for i in range(self.net_depth):
            self.hidden_layers.append(nn.Linear(in_features, self.net_width, bias=bias_enabled))
            if self.skip_layer is not None and i % self.skip_layer == 0 and i > 0:
                in_features = self.net_width + self.input_dim
            else:
                in_features = self.net_width
        if self.output_enabled:
            self.output_layer = nn.Linear(in_features, self.output_dim, bias=bias_enabled)
        else:
            self.output_dim = in_features
        self.initialize()

    def initialize(self):

        def init_func_hidden(m):
            if isinstance(m, nn.Linear):
                if self.hidden_init is not None:
                    self.hidden_init(m.weight)
                if self.bias_enabled and self.bias_init is not None:
                    self.bias_init(m.bias)
        self.hidden_layers.apply(init_func_hidden)
        if self.output_enabled:

            def init_func_output(m):
                if isinstance(m, nn.Linear):
                    if self.output_init is not None:
                        self.output_init(m.weight)
                    if self.bias_enabled and self.bias_init is not None:
                        self.bias_init(m.bias)
            self.output_layer.apply(init_func_output)

    def forward(self, x):
        inputs = x
        for i in range(self.net_depth):
            x = self.hidden_layers[i](x)
            x = self.hidden_activation(x)
            if self.skip_layer is not None and i % self.skip_layer == 0 and i > 0:
                x = torch.cat([x, inputs], dim=-1)
        if self.output_enabled:
            x = self.output_layer(x)
            x = self.output_activation(x)
        return x


class DenseLayer(MLP):

    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__(input_dim=input_dim, output_dim=output_dim, net_depth=0, **kwargs)


class NerfMLP(nn.Module):

    def __init__(self, input_dim: int, condition_dim: int, net_depth: int=8, net_width: int=256, skip_layer: int=4, net_depth_condition: int=1, net_width_condition: int=128):
        super().__init__()
        self.base = MLP(input_dim=input_dim, net_depth=net_depth, net_width=net_width, skip_layer=skip_layer, output_enabled=False)
        hidden_features = self.base.output_dim
        self.sigma_layer = DenseLayer(hidden_features, 1)
        if condition_dim > 0:
            self.bottleneck_layer = DenseLayer(hidden_features, net_width)
            self.rgb_layer = MLP(input_dim=net_width + condition_dim, output_dim=3, net_depth=net_depth_condition, net_width=net_width_condition, skip_layer=None)
        else:
            self.rgb_layer = DenseLayer(hidden_features, 3)

    def query_density(self, x):
        x = self.base(x)
        raw_sigma = self.sigma_layer(x)
        return raw_sigma

    def forward(self, x, condition=None):
        x = self.base(x)
        raw_sigma = self.sigma_layer(x)
        if condition is not None:
            if condition.shape[:-1] != x.shape[:-1]:
                num_rays, n_dim = condition.shape
                condition = condition.view([num_rays] + [1] * (x.dim() - condition.dim()) + [n_dim]).expand(list(x.shape[:-1]) + [n_dim])
            bottleneck = self.bottleneck_layer(x)
            x = torch.cat([bottleneck, condition], dim=-1)
        raw_rgb = self.rgb_layer(x)
        return raw_rgb, raw_sigma


class SinusoidalEncoder(nn.Module):
    """Sinusoidal Positional Encoder used in Nerf."""

    def __init__(self, x_dim, min_deg, max_deg, use_identity: bool=True):
        super().__init__()
        self.x_dim = x_dim
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.use_identity = use_identity
        self.register_buffer('scales', torch.tensor([(2 ** i) for i in range(min_deg, max_deg)]))

    @property
    def latent_dim(self) ->int:
        return (int(self.use_identity) + (self.max_deg - self.min_deg) * 2) * self.x_dim

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        """
        Args:
            x: [..., x_dim]
        Returns:
            latent: [..., latent_dim]
        """
        if self.max_deg == self.min_deg:
            return x
        xb = torch.reshape(x[Ellipsis, None, :] * self.scales[:, None], list(x.shape[:-1]) + [(self.max_deg - self.min_deg) * self.x_dim])
        latent = torch.sin(torch.cat([xb, xb + 0.5 * math.pi], dim=-1))
        if self.use_identity:
            latent = torch.cat([x] + [latent], dim=-1)
        return latent


class VanillaNeRFRadianceField(nn.Module):

    def __init__(self, net_depth: int=8, net_width: int=256, skip_layer: int=4, net_depth_condition: int=1, net_width_condition: int=128) ->None:
        super().__init__()
        self.posi_encoder = SinusoidalEncoder(3, 0, 10, True)
        self.view_encoder = SinusoidalEncoder(3, 0, 4, True)
        self.mlp = NerfMLP(input_dim=self.posi_encoder.latent_dim, condition_dim=self.view_encoder.latent_dim, net_depth=net_depth, net_width=net_width, skip_layer=skip_layer, net_depth_condition=net_depth_condition, net_width_condition=net_width_condition)

    def query_opacity(self, x, step_size):
        density = self.query_density(x)
        opacity = density * step_size
        return opacity

    def query_density(self, x):
        x = self.posi_encoder(x)
        sigma = self.mlp.query_density(x)
        return F.relu(sigma)

    def forward(self, x, condition=None):
        x = self.posi_encoder(x)
        if condition is not None:
            condition = self.view_encoder(condition)
        rgb, sigma = self.mlp(x, condition=condition)
        return torch.sigmoid(rgb), F.relu(sigma)


class DNeRFRadianceField(nn.Module):

    def __init__(self) ->None:
        super().__init__()
        self.posi_encoder = SinusoidalEncoder(3, 0, 4, True)
        self.time_encoder = SinusoidalEncoder(1, 0, 4, True)
        self.warp = MLP(input_dim=self.posi_encoder.latent_dim + self.time_encoder.latent_dim, output_dim=3, net_depth=4, net_width=64, skip_layer=2, output_init=functools.partial(torch.nn.init.uniform_, b=0.0001))
        self.nerf = VanillaNeRFRadianceField()

    def query_opacity(self, x, timestamps, step_size):
        idxs = torch.randint(0, len(timestamps), (x.shape[0],), device=x.device)
        t = timestamps[idxs]
        density = self.query_density(x, t)
        opacity = density * step_size
        return opacity

    def query_density(self, x, t):
        x = x + self.warp(torch.cat([self.posi_encoder(x), self.time_encoder(t)], dim=-1))
        return self.nerf.query_density(x)

    def forward(self, x, t, condition=None):
        x = x + self.warp(torch.cat([self.posi_encoder(x), self.time_encoder(t)], dim=-1))
        return self.nerf(x, condition=condition)


def contract_to_unisphere(x: torch.Tensor, aabb: torch.Tensor, eps: float=1e-06, derivative: bool=False):
    aabb_min, aabb_max = torch.split(aabb, 3, dim=-1)
    x = (x - aabb_min) / (aabb_max - aabb_min)
    x = x * 2 - 1
    mag = x.norm(dim=-1, keepdim=True)
    mask = mag.squeeze(-1) > 1
    if derivative:
        dev = (2 * mag - 1) / mag ** 2 + 2 * x ** 2 * (1 / mag ** 3 - (2 * mag - 1) / mag ** 4)
        dev[~mask] = 1.0
        dev = torch.clamp(dev, min=eps)
        return dev
    else:
        x[mask] = (2 - 1 / mag[mask]) * (x[mask] / mag[mask])
        x = x / 4 + 0.5
        return x


class _TruncExp(Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):
        x = ctx.saved_tensors[0]
        return g * torch.exp(torch.clamp(x, max=15))


trunc_exp = _TruncExp.apply


class NGPradianceField(torch.nn.Module):
    """Instance-NGP radiance Field"""

    def __init__(self, aabb: Union[torch.Tensor, List[float]], num_dim: int=3, use_viewdirs: bool=True, density_activation: Callable=lambda x: trunc_exp(x - 1), unbounded: bool=False, geo_feat_dim: int=15, n_levels: int=16, log2_hashmap_size: int=19) ->None:
        super().__init__()
        if not isinstance(aabb, torch.Tensor):
            aabb = torch.tensor(aabb, dtype=torch.float32)
        self.register_buffer('aabb', aabb)
        self.num_dim = num_dim
        self.use_viewdirs = use_viewdirs
        self.density_activation = density_activation
        self.unbounded = unbounded
        self.geo_feat_dim = geo_feat_dim
        per_level_scale = 1.4472692012786865
        if self.use_viewdirs:
            self.direction_encoding = tcnn.Encoding(n_input_dims=num_dim, encoding_config={'otype': 'Composite', 'nested': [{'n_dims_to_encode': 3, 'otype': 'SphericalHarmonics', 'degree': 4}]})
        self.mlp_base = tcnn.NetworkWithInputEncoding(n_input_dims=num_dim, n_output_dims=1 + self.geo_feat_dim, encoding_config={'otype': 'HashGrid', 'n_levels': n_levels, 'n_features_per_level': 2, 'log2_hashmap_size': log2_hashmap_size, 'base_resolution': 16, 'per_level_scale': per_level_scale}, network_config={'otype': 'FullyFusedMLP', 'activation': 'ReLU', 'output_activation': 'None', 'n_neurons': 64, 'n_hidden_layers': 1})
        if self.geo_feat_dim > 0:
            self.mlp_head = tcnn.Network(n_input_dims=(self.direction_encoding.n_output_dims if self.use_viewdirs else 0) + self.geo_feat_dim, n_output_dims=3, network_config={'otype': 'FullyFusedMLP', 'activation': 'ReLU', 'output_activation': 'Sigmoid', 'n_neurons': 64, 'n_hidden_layers': 2})

    def query_density(self, x, return_feat: bool=False):
        if self.unbounded:
            x = contract_to_unisphere(x, self.aabb)
        else:
            aabb_min, aabb_max = torch.split(self.aabb, self.num_dim, dim=-1)
            x = (x - aabb_min) / (aabb_max - aabb_min)
        selector = ((x > 0.0) & (x < 1.0)).all(dim=-1)
        x = self.mlp_base(x.view(-1, self.num_dim)).view(list(x.shape[:-1]) + [1 + self.geo_feat_dim])
        density_before_activation, base_mlp_out = torch.split(x, [1, self.geo_feat_dim], dim=-1)
        density = self.density_activation(density_before_activation) * selector[..., None]
        if return_feat:
            return density, base_mlp_out
        else:
            return density

    def _query_rgb(self, dir, embedding):
        if self.use_viewdirs:
            dir = (dir + 1.0) / 2.0
            d = self.direction_encoding(dir.view(-1, dir.shape[-1]))
            h = torch.cat([d, embedding.view(-1, self.geo_feat_dim)], dim=-1)
        else:
            h = embedding.view(-1, self.geo_feat_dim)
        rgb = self.mlp_head(h).view(list(embedding.shape[:-1]) + [3])
        return rgb

    def forward(self, positions: torch.Tensor, directions: torch.Tensor=None):
        if self.use_viewdirs and directions is not None:
            assert positions.shape == directions.shape, f'{positions.shape} v.s. {directions.shape}'
            density, embedding = self.query_density(positions, return_feat=True)
            rgb = self._query_rgb(directions, embedding=embedding)
        return rgb, density


_C = None


class ContractionType(Enum):
    """Space contraction options.

    This is an enum class that describes how a :class:`nerfacc.Grid` covers the 3D space.
    It is also used by :func:`nerfacc.ray_marching` to determine how to perform ray marching
    within the grid.

    The options in this enum class are:

    Attributes:
        AABB: Linearly map the region of interest :math:`[x_0, x_1]` to a
            unit cube in :math:`[0, 1]`.

            .. math:: f(x) = \\frac{x - x_0}{x_1 - x_0}

        UN_BOUNDED_TANH: Contract an unbounded space into a unit cube in :math:`[0, 1]`
            using tanh. The region of interest :math:`[x_0, x_1]` is first
            mapped into :math:`[-0.5, +0.5]` before applying tanh.

            .. math:: f(x) = \\frac{1}{2}(tanh(\\frac{x - x_0}{x_1 - x_0} - \\frac{1}{2}) + 1)

        UN_BOUNDED_SPHERE: Contract an unbounded space into a unit sphere. Used in
            `Mip-Nerf 360: Unbounded Anti-Aliased Neural Radiance Fields`_.

            .. math:: 
                f(x) = 
                \\begin{cases}
                z(x) & ||z(x)|| \\leq 1 \\\\
                (2 - \\frac{1}{||z(x)||})(\\frac{z(x)}{||z(x)||}) & ||z(x)|| > 1
                \\end{cases}
            
            .. math::
                z(x) = \\frac{x - x_0}{x_1 - x_0} * 2 - 1

            .. _Mip-Nerf 360\\: Unbounded Anti-Aliased Neural Radiance Fields:
                https://arxiv.org/abs/2111.12077

    """
    AABB = 0
    UN_BOUNDED_TANH = 1
    UN_BOUNDED_SPHERE = 2

    def to_cpp_version(self):
        """Convert to the C++ version of the enum class.

        Returns:
            The C++ version of the enum class.

        """
        return _C.ContractionTypeGetter(self.value)


class Grid(nn.Module):
    """An abstract Grid class.

    The grid is used as a cache of the 3D space to indicate whether each voxel
    area is important or not for the differentiable rendering process. The
    ray marching function (see :func:`nerfacc.ray_marching`) would use the
    grid to skip the unimportant voxel areas.

    To work with :func:`nerfacc.ray_marching`, three attributes must exist:

        - :attr:`roi_aabb`: The axis-aligned bounding box of the region of interest.
        - :attr:`binary`: A 3D binarized tensor of shape {resx, resy, resz},             with torch.bool data type.
        - :attr:`contraction_type`: The contraction type of the grid, indicating how             the 3D space is mapped to the grid.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.register_buffer('_dummy', torch.empty(0), persistent=False)

    @property
    def device(self) ->torch.device:
        return self._dummy.device

    @property
    def roi_aabb(self) ->torch.Tensor:
        """The axis-aligned bounding box of the region of interest.

        Its is a shape (6,) tensor in the format of {minx, miny, minz, maxx, maxy, maxz}.
        """
        if hasattr(self, '_roi_aabb'):
            return getattr(self, '_roi_aabb')
        else:
            raise NotImplementedError('please set an attribute named _roi_aabb')

    @property
    def binary(self) ->torch.Tensor:
        """A 3D binarized tensor with torch.bool data type.

        The tensor is of shape (resx, resy, resz), in which each boolen value
        represents whether the corresponding voxel should be kept or not.
        """
        if hasattr(self, '_binary'):
            return getattr(self, '_binary')
        else:
            raise NotImplementedError('please set an attribute named _binary')

    @property
    def contraction_type(self) ->ContractionType:
        """The contraction type of the grid.

        The contraction type is an indicator of how the 3D space is contracted
        to this voxel grid. See :class:`nerfacc.ContractionType` for more details.
        """
        if hasattr(self, '_contraction_type'):
            return getattr(self, '_contraction_type')
        else:
            raise NotImplementedError('please set an attribute named _contraction_type')


def _meshgrid3d(res: torch.Tensor, device: Union[torch.device, str]='cpu') ->torch.Tensor:
    """Create 3D grid coordinates."""
    assert len(res) == 3
    res = res.tolist()
    return torch.stack(torch.meshgrid([torch.arange(res[0], dtype=torch.long), torch.arange(res[1], dtype=torch.long), torch.arange(res[2], dtype=torch.long)], indexing='ij'), dim=-1)


@torch.no_grad()
def contract_inv(x: torch.Tensor, roi: torch.Tensor, type: ContractionType=ContractionType.AABB) ->torch.Tensor:
    """Recover the space from [0, 1]^3 by inverse contraction.

    Args:
        x (torch.Tensor): Contracted points ([0, 1]^3).
        roi (torch.Tensor): Region of interest.
        type (ContractionType): Contraction type.

    Returns:
        torch.Tensor: Un-contracted points.
    """
    ctype = type.to_cpp_version()
    return _C.contract_inv(x.contiguous(), roi.contiguous(), ctype)


@torch.no_grad()
def query_grid(samples: torch.Tensor, grid_roi: torch.Tensor, grid_values: torch.Tensor, grid_type: ContractionType):
    """Query grid values given coordinates.

    Args:
        samples: (n_samples, 3) tensor of coordinates.
        grid_roi: (6,) region of interest of the grid. Usually it should be
            accquired from the grid itself using `grid.roi_aabb`.
        grid_values: A 3D tensor of grid values in the shape of (resx, resy, resz).
        grid_type: Contraction type of the grid. Usually it should be
            accquired from the grid itself using `grid.contraction_type`.

    Returns:
        (n_samples) values for those samples queried from the grid.
    """
    assert samples.dim() == 2 and samples.size(-1) == 3
    assert grid_roi.dim() == 1 and grid_roi.size(0) == 6
    assert grid_values.dim() == 3
    assert isinstance(grid_type, ContractionType)
    return _C.grid_query(samples.contiguous(), grid_roi.contiguous(), grid_values.contiguous(), grid_type.to_cpp_version())


class OccupancyGrid(Grid):
    """Occupancy grid: whether each voxel area is occupied or not.

    Args:
        roi_aabb: The axis-aligned bounding box of the region of interest. Useful for mapping
            the 3D space to the grid.
        resolution: The resolution of the grid. If an integer is given, the grid is assumed to
            be a cube. Otherwise, a list or a tensor of shape (3,) is expected. Default: 128.
        contraction_type: The contraction type of the grid. See :class:`nerfacc.ContractionType`
            for more details. Default: :attr:`nerfacc.ContractionType.AABB`.
    """
    NUM_DIM: int = 3

    def __init__(self, roi_aabb: Union[List[int], torch.Tensor], resolution: Union[int, List[int], torch.Tensor]=128, contraction_type: ContractionType=ContractionType.AABB) ->None:
        super().__init__()
        if isinstance(resolution, int):
            resolution = [resolution] * self.NUM_DIM
        if isinstance(resolution, (list, tuple)):
            resolution = torch.tensor(resolution, dtype=torch.int32)
        assert isinstance(resolution, torch.Tensor), f'Invalid type: {type(resolution)}'
        assert resolution.shape == (self.NUM_DIM,), f'Invalid shape: {resolution.shape}'
        if isinstance(roi_aabb, (list, tuple)):
            roi_aabb = torch.tensor(roi_aabb, dtype=torch.float32)
        assert isinstance(roi_aabb, torch.Tensor), f'Invalid type: {type(roi_aabb)}'
        assert roi_aabb.shape == torch.Size([self.NUM_DIM * 2]), f'Invalid shape: {roi_aabb.shape}'
        self.num_cells = int(resolution.prod().item())
        self.register_buffer('_roi_aabb', roi_aabb)
        self.register_buffer('_binary', torch.zeros(resolution.tolist(), dtype=torch.bool))
        self._contraction_type = contraction_type
        self.register_buffer('resolution', resolution)
        self.register_buffer('occs', torch.zeros(self.num_cells))
        grid_coords = _meshgrid3d(resolution).reshape(self.num_cells, self.NUM_DIM)
        self.register_buffer('grid_coords', grid_coords)
        grid_indices = torch.arange(self.num_cells)
        self.register_buffer('grid_indices', grid_indices)

    @torch.no_grad()
    def _get_all_cells(self) ->torch.Tensor:
        """Returns all cells of the grid."""
        return self.grid_indices

    @torch.no_grad()
    def _sample_uniform_and_occupied_cells(self, n: int) ->torch.Tensor:
        """Samples both n uniform and occupied cells."""
        uniform_indices = torch.randint(self.num_cells, (n,), device=self.device)
        occupied_indices = torch.nonzero(self._binary.flatten())[:, 0]
        if n < len(occupied_indices):
            selector = torch.randint(len(occupied_indices), (n,), device=self.device)
            occupied_indices = occupied_indices[selector]
        indices = torch.cat([uniform_indices, occupied_indices], dim=0)
        return indices

    @torch.no_grad()
    def _update(self, step: int, occ_eval_fn: Callable, occ_thre: float=0.01, ema_decay: float=0.95, warmup_steps: int=256) ->None:
        """Update the occ field in the EMA way."""
        if step < warmup_steps:
            indices = self._get_all_cells()
        else:
            N = self.num_cells // 4
            indices = self._sample_uniform_and_occupied_cells(N)
        grid_coords = self.grid_coords[indices]
        x = (grid_coords + torch.rand_like(grid_coords, dtype=torch.float32)) / self.resolution
        if self._contraction_type == ContractionType.UN_BOUNDED_SPHERE:
            mask = (x - 0.5).norm(dim=1) < 0.5
            x = x[mask]
            indices = indices[mask]
        x = contract_inv(x, roi=self._roi_aabb, type=self._contraction_type)
        occ = occ_eval_fn(x).squeeze(-1)
        self.occs[indices] = torch.maximum(self.occs[indices] * ema_decay, occ)
        self._binary = (self.occs > torch.clamp(self.occs.mean(), max=occ_thre)).view(self._binary.shape)

    @torch.no_grad()
    def every_n_step(self, step: int, occ_eval_fn: Callable, occ_thre: float=0.01, ema_decay: float=0.95, warmup_steps: int=256, n: int=16) ->None:
        """Update the grid every n steps during training.

        Args:
            step: Current training step.
            occ_eval_fn: A function that takes in sample locations :math:`(N, 3)` and
                returns the occupancy values :math:`(N, 1)` at those locations.
            occ_thre: Threshold used to binarize the occupancy grid. Default: 1e-2.
            ema_decay: The decay rate for EMA updates. Default: 0.95.
            warmup_steps: Sample all cells during the warmup stage. After the warmup
                stage we change the sampling strategy to 1/4 uniformly sampled cells
                together with 1/4 occupied cells. Default: 256.
            n: Update the grid every n steps. Default: 16.
        """
        if not self.training:
            raise RuntimeError('You should only call this function only during training. Please call _update() directly if you want to update the field during inference.')
        if step % n == 0 and self.training:
            self._update(step=step, occ_eval_fn=occ_eval_fn, occ_thre=occ_thre, ema_decay=ema_decay, warmup_steps=warmup_steps)

    @torch.no_grad()
    def query_occ(self, samples: torch.Tensor) ->torch.Tensor:
        """Query the occupancy field at the given samples.

        Args:
            samples: Samples in the world coordinates. (n_samples, 3)

        Returns:
            Occupancy values at the given samples. (n_samples,)
        """
        return query_grid(samples, self._roi_aabb, self.binary, self.contraction_type)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (DenseLayer,
     lambda: ([], {'input_dim': 4, 'output_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SinusoidalEncoder,
     lambda: ([], {'x_dim': 4, 'min_deg': 4, 'max_deg': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_KAIR_BAIR_nerfacc(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

