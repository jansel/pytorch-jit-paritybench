import sys
_module = sys.modules[__name__]
del sys
regenerate = _module
conf = _module
utils = _module
camera_visualization = _module
plot_image_grid = _module
pytorch3d = _module
io = _module
mtl_io = _module
obj_io = _module
ply_io = _module
loss = _module
chamfer = _module
mesh_edge_loss = _module
mesh_laplacian_smoothing = _module
mesh_normal_consistency = _module
point_mesh_distance = _module
ops = _module
cubify = _module
graph_conv = _module
knn = _module
mesh_face_areas_normals = _module
packed_to_padded = _module
perspective_n_points = _module
points_alignment = _module
points_normals = _module
sample_points_from_meshes = _module
subdivide_meshes = _module
utils = _module
vert_align = _module
renderer = _module
blending = _module
cameras = _module
compositing = _module
lighting = _module
materials = _module
mesh = _module
rasterize_meshes = _module
rasterizer = _module
renderer = _module
shader = _module
shading = _module
texturing = _module
utils = _module
points = _module
compositor = _module
rasterize_points = _module
rasterizer = _module
renderer = _module
utils = _module
structures = _module
meshes = _module
pointclouds = _module
textures = _module
utils = _module
transforms = _module
rotation_conversions = _module
so3 = _module
transform3d = _module
ico_sphere = _module
torus = _module
parse_tutorials = _module
setup = _module
tests = _module
bm_blending = _module
bm_chamfer = _module
bm_cubify = _module
bm_face_areas_normals = _module
bm_graph_conv = _module
bm_knn = _module
bm_lighting = _module
bm_main = _module
bm_mesh_edge_loss = _module
bm_mesh_io = _module
bm_mesh_laplacian_smoothing = _module
bm_mesh_normal_consistency = _module
bm_mesh_rasterizer_transform = _module
bm_meshes = _module
bm_packed_to_padded = _module
bm_perspective_n_points = _module
bm_point_mesh_distance = _module
bm_pointclouds = _module
bm_points_alignment = _module
bm_rasterize_meshes = _module
bm_rasterize_points = _module
bm_sample_points_from_meshes = _module
bm_so3 = _module
bm_subdivide_meshes = _module
bm_vert_align = _module
common_testing = _module
test_blending = _module
test_build = _module
test_cameras = _module
test_chamfer = _module
test_common_testing = _module
test_compositing = _module
test_cubify = _module
test_face_areas_normals = _module
test_graph_conv = _module
test_knn = _module
test_lighting = _module
test_materials = _module
test_mesh_edge_loss = _module
test_mesh_laplacian_smoothing = _module
test_mesh_normal_consistency = _module
test_mesh_rendering_utils = _module
test_meshes = _module
test_obj_io = _module
test_ops_utils = _module
test_packed_to_padded = _module
test_perspective_n_points = _module
test_ply_io = _module
test_point_mesh_distance = _module
test_pointclouds = _module
test_points_alignment = _module
test_points_normals = _module
test_rasterize_meshes = _module
test_rasterize_points = _module
test_rasterizer = _module
test_render_meshes = _module
test_render_points = _module
test_rendering_utils = _module
test_rotation_conversions = _module
test_sample_points_from_meshes = _module
test_so3 = _module
test_struct_utils = _module
test_subdivide_meshes = _module
test_texturing = _module
test_transforms = _module
test_vert_align = _module

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


import matplotlib.pyplot as plt


import torch


import warnings


from typing import Dict


from typing import List


from typing import Optional


import numpy as np


import torch.nn.functional as F


from collections import namedtuple


from typing import Tuple


from typing import Union


from itertools import islice


from torch.autograd import Function


from torch.autograd.function import once_differentiable


import torch.nn as nn


from typing import NamedTuple


from typing import TYPE_CHECKING


from typing import Sequence


import math


import copy


import inspect


from typing import Any


from torch.nn.functional import interpolate


import functools


from itertools import tee


from math import cos


from math import pi


from math import sin


from typing import Iterator


from torch.utils.cpp_extension import CUDA_HOME


from torch.utils.cpp_extension import CppExtension


from torch.utils.cpp_extension import CUDAExtension


from itertools import product


from typing import Callable


import itertools


class GatherScatter(Function):
    """
    Torch autograd Function wrapper for gather_scatter C++/CUDA implementations.
    """

    @staticmethod
    def forward(ctx, input, edges, directed=False):
        """
        Args:
            ctx: Context object used to calculate gradients.
            input: Tensor of shape (num_vertices, input_dim)
            edges: Tensor of edge indices of shape (num_edges, 2)
            directed: Bool indicating if edges are directed.

        Returns:
            output: Tensor of same shape as input.
        """
        if not input.dim() == 2:
            raise ValueError('input can only have 2 dimensions.')
        if not edges.dim() == 2:
            raise ValueError('edges can only have 2 dimensions.')
        if not edges.shape[1] == 2:
            raise ValueError('edges must be of shape (num_edges, 2).')
        if not input.dtype == torch.float32:
            raise ValueError('input has to be of type torch.float32.')
        ctx.directed = directed
        input, edges = input.contiguous(), edges.contiguous()
        ctx.save_for_backward(edges)
        backward = False
        output = _C.gather_scatter(input, edges, directed, backward)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        edges = ctx.saved_tensors[0]
        directed = ctx.directed
        backward = True
        grad_input = _C.gather_scatter(grad_output, edges, directed, backward)
        grad_edges = None
        grad_directed = None
        return grad_input, grad_edges, grad_directed


gather_scatter = GatherScatter.apply


def gather_scatter_python(input, edges, directed: bool=False):
    """
    Python implementation of gather_scatter for aggregating features of
    neighbor nodes in a graph.

    Given a directed graph: v0 -> v1 -> v2 the updated feature for v1 depends
    on v2 in order to be consistent with Morris et al. AAAI 2019
    (https://arxiv.org/abs/1810.02244). This only affects
    directed graphs; for undirected graphs v1 will depend on both v0 and v2,
    no matter which way the edges are physically stored.

    Args:
        input: Tensor of shape (num_vertices, input_dim).
        edges: Tensor of edge indices of shape (num_edges, 2).
        directed: bool indicating if edges are directed.

    Returns:
        output: Tensor of same shape as input.
    """
    if not input.dim() == 2:
        raise ValueError('input can only have 2 dimensions.')
    if not edges.dim() == 2:
        raise ValueError('edges can only have 2 dimensions.')
    if not edges.shape[1] == 2:
        raise ValueError('edges must be of shape (num_edges, 2).')
    num_vertices, input_feature_dim = input.shape
    num_edges = edges.shape[0]
    output = torch.zeros_like(input)
    idx0 = edges[:, (0)].view(num_edges, 1).expand(num_edges, input_feature_dim)
    idx1 = edges[:, (1)].view(num_edges, 1).expand(num_edges, input_feature_dim)
    output = output.scatter_add(0, idx0, input.gather(0, idx1))
    if not directed:
        output = output.scatter_add(0, idx1, input.gather(0, idx0))
    return output


class GraphConv(nn.Module):
    """A single graph convolution layer."""

    def __init__(self, input_dim: int, output_dim: int, init: str='normal', directed: bool=False):
        """
        Args:
            input_dim: Number of input features per vertex.
            output_dim: Number of output features per vertex.
            init: Weight initialization method. Can be one of ['zero', 'normal'].
            directed: Bool indicating if edges in the graph are directed.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.directed = directed
        self.w0 = nn.Linear(input_dim, output_dim)
        self.w1 = nn.Linear(input_dim, output_dim)
        if init == 'normal':
            nn.init.normal_(self.w0.weight, mean=0, std=0.01)
            nn.init.normal_(self.w1.weight, mean=0, std=0.01)
            self.w0.bias.data.zero_()
            self.w1.bias.data.zero_()
        elif init == 'zero':
            self.w0.weight.data.zero_()
            self.w1.weight.data.zero_()
        else:
            raise ValueError('Invalid GraphConv initialization "%s"' % init)

    def forward(self, verts, edges):
        """
        Args:
            verts: FloatTensor of shape (V, input_dim) where V is the number of
                vertices and input_dim is the number of input features
                per vertex. input_dim has to match the input_dim specified
                in __init__.
            edges: LongTensor of shape (E, 2) where E is the number of edges
                where each edge has the indices of the two vertices which
                form the edge.

        Returns:
            out: FloatTensor of shape (V, output_dim) where output_dim is the
            number of output features per vertex.
        """
        if verts.is_cuda != edges.is_cuda:
            raise ValueError('verts and edges tensors must be on the same device.')
        if verts.shape[0] == 0:
            return verts.new_zeros((0, self.output_dim)) * verts.sum()
        verts_w0 = self.w0(verts)
        verts_w1 = self.w1(verts)
        if torch.cuda.is_available() and verts.is_cuda and edges.is_cuda:
            neighbor_sums = gather_scatter(verts_w1, edges, self.directed)
        else:
            neighbor_sums = gather_scatter_python(verts_w1, edges, self.directed)
        out = verts_w0 + neighbor_sums
        return out

    def __repr__(self):
        Din, Dout, directed = self.input_dim, self.output_dim, self.directed
        return 'GraphConv(%d -> %d, directed=%r)' % (Din, Dout, directed)


def _extend_tensor(input_tensor: torch.Tensor, N: int) ->torch.Tensor:
    """
    Extend a tensor `input_tensor` with ndim > 2, `N` times along the batch
    dimension. This is done in the following sequence of steps (where `B` is
    the batch dimension):

    .. code-block:: python

        input_tensor (B, ...)
        -> add leading empty dimension (1, B, ...)
        -> expand (N, B, ...)
        -> reshape (N * B, ...)

    Args:
        input_tensor: torch.Tensor with ndim > 2 representing a batched input.
        N: number of times to extend each element of the batch.
    """
    if input_tensor.ndim < 2:
        raise ValueError('Input tensor must have ndimensions >= 2.')
    B = input_tensor.shape[0]
    non_batch_dims = tuple(input_tensor.shape[1:])
    constant_dims = (-1,) * input_tensor.ndim
    return input_tensor.clone()[None, ...].expand(N, *constant_dims).transpose(0, 1).reshape(N * B, *non_batch_dims)


def _pad_texture_maps(images: List[torch.Tensor]) ->torch.Tensor:
    """
    Pad all texture images so they have the same height and width.

    Args:
        images: list of N tensors of shape (H, W, 3)

    Returns:
        tex_maps: Tensor of shape (N, max_H, max_W, 3)
    """
    tex_maps = []
    max_H = 0
    max_W = 0
    for im in images:
        h, w, _3 = im.shape
        if h > max_H:
            max_H = h
        if w > max_W:
            max_W = w
        tex_maps.append(im)
    max_shape = max_H, max_W
    for i, image in enumerate(tex_maps):
        if image.shape[:2] != max_shape:
            image_BCHW = image.permute(2, 0, 1)[None]
            new_image_BCHW = interpolate(image_BCHW, size=max_shape, mode='bilinear', align_corners=False)
            tex_maps[i] = new_image_BCHW[0].permute(1, 2, 0)
    tex_maps = torch.stack(tex_maps, dim=0)
    return tex_maps


def padded_to_list(x: torch.Tensor, split_size: Union[list, tuple, None]=None):
    """
    Transforms a padded tensor of shape (N, M, K) into a list of N tensors
    of shape (Mi, Ki) where (Mi, Ki) is specified in split_size(i), or of shape
    (M, K) if split_size is None.
    Support only for 3-dimensional input tensor.

    Args:
      x: tensor
      split_size: list, tuple or int defining the number of items for each tensor
        in the output list.

    Returns:
      x_list: a list of tensors
    """
    if x.ndim != 3:
        raise ValueError('Supports only 3-dimensional input tensors')
    x_list = list(x.unbind(0))
    if split_size is None:
        return x_list
    N = len(split_size)
    if x.shape[0] != N:
        raise ValueError('Split size must be of same length as inputs first dimension')
    for i in range(N):
        if isinstance(split_size[i], int):
            x_list[i] = x_list[i][:split_size[i]]
        elif len(split_size[i]) == 2:
            x_list[i] = x_list[i][:split_size[i][0], :split_size[i][1]]
        else:
            raise ValueError('Support only for 2-dimensional unbinded tensor.                     Split size for more dimensions provided')
    return x_list


def padded_to_packed(x: torch.Tensor, split_size: Union[list, tuple, None]=None, pad_value: Union[float, int, None]=None):
    """
    Transforms a padded tensor of shape (N, M, K) into a packed tensor
    of shape:
     - (sum(Mi), K) where (Mi, K) are the dimensions of
        each of the tensors in the batch and Mi is specified by split_size(i)
     - (N*M, K) if split_size is None

    Support only for 3-dimensional input tensor and 1-dimensional split size.

    Args:
      x: tensor
      split_size: list, tuple or int defining the number of items for each tensor
        in the output list.
      pad_value: optional value to use to filter the padded values in the input
        tensor.

    Only one of split_size or pad_value should be provided, or both can be None.

    Returns:
      x_packed: a packed tensor.
    """
    if x.ndim != 3:
        raise ValueError('Supports only 3-dimensional input tensors')
    N, M, D = x.shape
    if split_size is not None and pad_value is not None:
        raise ValueError('Only one of split_size or pad_value should be provided.')
    x_packed = x.reshape(-1, D)
    if pad_value is None and split_size is None:
        return x_packed
    if pad_value is not None:
        mask = x_packed.ne(pad_value).any(-1)
        x_packed = x_packed[mask]
        return x_packed
    N = len(split_size)
    if x.shape[0] != N:
        raise ValueError('Split size must be of same length as inputs first dimension')
    if not all(isinstance(i, int) for i in split_size):
        raise ValueError('Support only 1-dimensional unbinded tensor.                 Split size for more dimensions provided')
    padded_to_packed_idx = torch.cat([(torch.arange(v, dtype=torch.int64, device=x.device) + i * M) for i, v in enumerate(split_size)], dim=0)
    return x_packed[padded_to_packed_idx]


class Textures(object):

    def __init__(self, maps: Union[List, torch.Tensor, None]=None, faces_uvs: Optional[torch.Tensor]=None, verts_uvs: Optional[torch.Tensor]=None, verts_rgb: Optional[torch.Tensor]=None):
        """
        Args:
            maps: texture map per mesh. This can either be a list of maps
              [(H, W, 3)] or a padded tensor of shape (N, H, W, 3).
            faces_uvs: (N, F, 3) tensor giving the index into verts_uvs for each
                vertex in the face. Padding value is assumed to be -1.
            verts_uvs: (N, V, 2) tensor giving the uv coordinate per vertex.
            verts_rgb: (N, V, 3) tensor giving the rgb color per vertex. Padding
                value is assumed to be -1.

        Note: only the padded representation of the textures is stored
        and the packed/list representations are computed on the fly and
        not cached.
        """
        if faces_uvs is not None and faces_uvs.ndim != 3:
            msg = 'Expected faces_uvs to be of shape (N, F, 3); got %r'
            raise ValueError(msg % repr(faces_uvs.shape))
        if verts_uvs is not None and verts_uvs.ndim != 3:
            msg = 'Expected verts_uvs to be of shape (N, V, 2); got %r'
            raise ValueError(msg % repr(verts_uvs.shape))
        if verts_rgb is not None and verts_rgb.ndim != 3:
            msg = 'Expected verts_rgb to be of shape (N, V, 3); got %r'
            raise ValueError(msg % repr(verts_rgb.shape))
        if maps is not None:
            if torch.is_tensor(maps) and maps.ndim != 4:
                msg = 'Expected maps to be of shape (N, H, W, 3); got %r'
                raise ValueError(msg % repr(maps.shape))
            elif isinstance(maps, list):
                maps = _pad_texture_maps(maps)
            if faces_uvs is None or verts_uvs is None:
                msg = 'To use maps, faces_uvs and verts_uvs are required'
                raise ValueError(msg)
        self._faces_uvs_padded = faces_uvs
        self._verts_uvs_padded = verts_uvs
        self._verts_rgb_padded = verts_rgb
        self._maps_padded = maps
        self._num_faces_per_mesh = None
        self._num_verts_per_mesh = None

    def clone(self):
        other = self.__class__()
        for k in dir(self):
            v = getattr(self, k)
            if torch.is_tensor(v):
                setattr(other, k, v.clone())
        return other

    def to(self, device):
        for k in dir(self):
            v = getattr(self, k)
            if torch.is_tensor(v) and v.device != device:
                setattr(self, k, v)
        return self

    def __getitem__(self, index):
        other = self.__class__()
        for key in dir(self):
            value = getattr(self, key)
            if torch.is_tensor(value):
                if isinstance(index, int):
                    setattr(other, key, value[index][None])
                else:
                    setattr(other, key, value[index])
        return other

    def faces_uvs_padded(self) ->torch.Tensor:
        return self._faces_uvs_padded

    def faces_uvs_list(self) ->Union[List[torch.Tensor], None]:
        if self._faces_uvs_padded is None:
            return None
        return padded_to_list(self._faces_uvs_padded, split_size=self._num_faces_per_mesh)

    def faces_uvs_packed(self) ->Union[torch.Tensor, None]:
        if self._faces_uvs_padded is None:
            return None
        return padded_to_packed(self._faces_uvs_padded, split_size=self._num_faces_per_mesh)

    def verts_uvs_padded(self) ->Union[torch.Tensor, None]:
        return self._verts_uvs_padded

    def verts_uvs_list(self) ->Union[List[torch.Tensor], None]:
        if self._verts_uvs_padded is None:
            return None
        return padded_to_list(self._verts_uvs_padded)

    def verts_uvs_packed(self) ->Union[torch.Tensor, None]:
        if self._verts_uvs_padded is None:
            return None
        return padded_to_packed(self._verts_uvs_padded)

    def verts_rgb_padded(self) ->Union[torch.Tensor, None]:
        return self._verts_rgb_padded

    def verts_rgb_list(self) ->Union[List[torch.Tensor], None]:
        if self._verts_rgb_padded is None:
            return None
        return padded_to_list(self._verts_rgb_padded, split_size=self._num_verts_per_mesh)

    def verts_rgb_packed(self) ->Union[torch.Tensor, None]:
        if self._verts_rgb_padded is None:
            return None
        return padded_to_packed(self._verts_rgb_padded, split_size=self._num_verts_per_mesh)

    def maps_padded(self) ->Union[torch.Tensor, None]:
        return self._maps_padded

    def extend(self, N: int) ->'Textures':
        """
        Create new Textures class which contains each input texture N times

        Args:
            N: number of new copies of each texture.

        Returns:
            new Textures object.
        """
        if not isinstance(N, int):
            raise ValueError('N must be an integer.')
        if N <= 0:
            raise ValueError('N must be > 0.')
        if all(v is not None for v in [self._faces_uvs_padded, self._verts_uvs_padded, self._maps_padded]):
            new_verts_uvs = _extend_tensor(self._verts_uvs_padded, N)
            new_faces_uvs = _extend_tensor(self._faces_uvs_padded, N)
            new_maps = _extend_tensor(self._maps_padded, N)
            return self.__class__(verts_uvs=new_verts_uvs, faces_uvs=new_faces_uvs, maps=new_maps)
        elif self._verts_rgb_padded is not None:
            new_verts_rgb = _extend_tensor(self._verts_rgb_padded, N)
            return self.__class__(verts_rgb=new_verts_rgb)
        else:
            msg = 'Either vertex colors or texture maps are required.'
            raise ValueError(msg)


class Meshes(object):
    """
    This class provides functions for working with batches of triangulated
    meshes with varying numbers of faces and vertices, and converting between
    representations.

    Within Meshes, there are three different representations of the faces and
    verts data:

    List
      - only used for input as a starting point to convert to other representations.
    Padded
      - has specific batch dimension.
    Packed
      - no batch dimension.
      - has auxillary variables used to index into the padded representation.

    Example:

    Input list of verts V_n = [[V_1], [V_2], ... , [V_N]]
    where V_1, ... , V_N are the number of verts in each mesh and N is the
    numer of meshes.

    Input list of faces F_n = [[F_1], [F_2], ... , [F_N]]
    where F_1, ... , F_N are the number of faces in each mesh.

    # SPHINX IGNORE
     List                      | Padded                  | Packed
    ---------------------------|-------------------------|------------------------
    [[V_1], ... , [V_N]]       | size = (N, max(V_n), 3) |  size = (sum(V_n), 3)
                               |                         |
    Example for verts:         |                         |
                               |                         |
    V_1 = 3, V_2 = 4, V_3 = 5  | size = (3, 5, 3)        |  size = (12, 3)
                               |                         |
    List([                     | tensor([                |  tensor([
      [                        |     [                   |    [0.1, 0.3, 0.5],
        [0.1, 0.3, 0.5],       |       [0.1, 0.3, 0.5],  |    [0.5, 0.2, 0.1],
        [0.5, 0.2, 0.1],       |       [0.5, 0.2, 0.1],  |    [0.6, 0.8, 0.7],
        [0.6, 0.8, 0.7],       |       [0.6, 0.8, 0.7],  |    [0.1, 0.3, 0.3],
      ],                       |       [0,    0,    0],  |    [0.6, 0.7, 0.8],
      [                        |       [0,    0,    0],  |    [0.2, 0.3, 0.4],
        [0.1, 0.3, 0.3],       |     ],                  |    [0.1, 0.5, 0.3],
        [0.6, 0.7, 0.8],       |     [                   |    [0.7, 0.3, 0.6],
        [0.2, 0.3, 0.4],       |       [0.1, 0.3, 0.3],  |    [0.2, 0.4, 0.8],
        [0.1, 0.5, 0.3],       |       [0.6, 0.7, 0.8],  |    [0.9, 0.5, 0.2],
      ],                       |       [0.2, 0.3, 0.4],  |    [0.2, 0.3, 0.4],
      [                        |       [0.1, 0.5, 0.3],  |    [0.9, 0.3, 0.8],
        [0.7, 0.3, 0.6],       |       [0,    0,    0],  |  ])
        [0.2, 0.4, 0.8],       |     ],                  |
        [0.9, 0.5, 0.2],       |     [                   |
        [0.2, 0.3, 0.4],       |       [0.7, 0.3, 0.6],  |
        [0.9, 0.3, 0.8],       |       [0.2, 0.4, 0.8],  |
      ]                        |       [0.9, 0.5, 0.2],  |
   ])                          |       [0.2, 0.3, 0.4],  |
                               |       [0.9, 0.3, 0.8],  |
                               |     ]                   |
                               |  ])                     |
    Example for faces:         |                         |
                               |                         |
    F_1 = 1, F_2 = 2, F_3 = 7  | size = (3, 7, 3)        | size = (10, 3)
                               |                         |
    List([                     | tensor([                | tensor([
      [                        |     [                   |    [ 0,  1,  2],
        [0, 1, 2],             |       [0,   1,  2],     |    [ 3,  4,  5],
      ],                       |       [-1, -1, -1],     |    [ 4,  5,  6],
      [                        |       [-1, -1, -1]      |    [ 8,  9,  7],
        [0, 1, 2],             |       [-1, -1, -1]      |    [ 7,  8, 10],
        [1, 2, 3],             |       [-1, -1, -1]      |    [ 9, 10,  8],
      ],                       |       [-1, -1, -1],     |    [11, 10,  9],
      [                        |       [-1, -1, -1],     |    [11,  7,  8],
        [1, 2, 0],             |     ],                  |    [11, 10,  8],
        [0, 1, 3],             |     [                   |    [11,  9,  8],
        [2, 3, 1],             |       [0,   1,  2],     |  ])
        [4, 3, 2],             |       [1,   2,  3],     |
        [4, 0, 1],             |       [-1, -1, -1],     |
        [4, 3, 1],             |       [-1, -1, -1],     |
        [4, 2, 1],             |       [-1, -1, -1],     |
      ],                       |       [-1, -1, -1],     |
    ])                         |       [-1, -1, -1],     |
                               |     ],                  |
                               |     [                   |
                               |       [1,   2,  0],     |
                               |       [0,   1,  3],     |
                               |       [2,   3,  1],     |
                               |       [4,   3,  2],     |
                               |       [4,   0,  1],     |
                               |       [4,   3,  1],     |
                               |       [4,   2,  1],     |
                               |     ]                   |
                               |   ])                    |
    -----------------------------------------------------------------------------

    Auxillary variables for packed representation

    Name                           |   Size              |  Example from above
    -------------------------------|---------------------|-----------------------
                                   |                     |
    verts_packed_to_mesh_idx       |  size = (sum(V_n))  |   tensor([
                                   |                     |     0, 0, 0, 1, 1, 1,
                                   |                     |     1, 2, 2, 2, 2, 2
                                   |                     |   )]
                                   |                     |   size = (12)
                                   |                     |
    mesh_to_verts_packed_first_idx |  size = (N)         |   tensor([0, 3, 7])
                                   |                     |   size = (3)
                                   |                     |
    num_verts_per_mesh             |  size = (N)         |   tensor([3, 4, 5])
                                   |                     |   size = (3)
                                   |                     |
    faces_packed_to_mesh_idx       |  size = (sum(F_n))  |   tensor([
                                   |                     |     0, 1, 1, 2, 2, 2,
                                   |                     |     2, 2, 2, 2
                                   |                     |   )]
                                   |                     |   size = (10)
                                   |                     |
    mesh_to_faces_packed_first_idx |  size = (N)         |   tensor([0, 1, 3])
                                   |                     |   size = (3)
                                   |                     |
    num_faces_per_mesh             |  size = (N)         |   tensor([1, 2, 7])
                                   |                     |   size = (3)
                                   |                     |
    verts_padded_to_packed_idx     |  size = (sum(V_n))  |  tensor([
                                   |                     |     0, 1, 2, 5, 6, 7,
                                   |                     |     8, 10, 11, 12, 13,
                                   |                     |     14
                                   |                     |  )]
                                   |                     |  size = (12)
    -----------------------------------------------------------------------------
    # SPHINX IGNORE

    From the faces, edges are computed and have packed and padded
    representations with auxillary variables.

    E_n = [[E_1], ... , [E_N]]
    where E_1, ... , E_N are the number of unique edges in each mesh.
    Total number of unique edges = sum(E_n)

    # SPHINX IGNORE
    Name                           |   Size                  | Example from above
    -------------------------------|-------------------------|----------------------
                                   |                         |
    edges_packed                   | size = (sum(E_n), 2)    |  tensor([
                                   |                         |     [0, 1],
                                   |                         |     [0, 2],
                                   |                         |     [1, 2],
                                   |                         |       ...
                                   |                         |     [10, 11],
                                   |                         |   )]
                                   |                         |   size = (18, 2)
                                   |                         |
    num_edges_per_mesh             | size = (N)              |  tensor([3, 5, 10])
                                   |                         |  size = (3)
                                   |                         |
    edges_packed_to_mesh_idx       | size = (sum(E_n))       |  tensor([
                                   |                         |    0, 0, 0,
                                   |                         |     . . .
                                   |                         |    2, 2, 2
                                   |                         |   ])
                                   |                         |   size = (18)
                                   |                         |
    faces_packed_to_edges_packed   | size = (sum(F_n), 3)    |  tensor([
                                   |                         |    [2,   1,  0],
                                   |                         |    [5,   4,  3],
                                   |                         |       .  .  .
                                   |                         |    [12, 14, 16],
                                   |                         |   ])
                                   |                         |   size = (10, 3)
                                   |                         |
    mesh_to_edges_packed_first_idx | size = (N)              |  tensor([0, 3, 8])
                                   |                         |  size = (3)
    ----------------------------------------------------------------------------
    # SPHINX IGNORE
    """
    _INTERNAL_TENSORS = ['_verts_packed', '_verts_packed_to_mesh_idx', '_mesh_to_verts_packed_first_idx', '_verts_padded', '_num_verts_per_mesh', '_faces_packed', '_faces_packed_to_mesh_idx', '_mesh_to_faces_packed_first_idx', '_faces_padded', '_faces_areas_packed', '_verts_normals_packed', '_faces_normals_packed', '_num_faces_per_mesh', '_edges_packed', '_edges_packed_to_mesh_idx', '_mesh_to_edges_packed_first_idx', '_faces_packed_to_edges_packed', '_num_edges_per_mesh', '_verts_padded_to_packed_idx', '_laplacian_packed', 'valid', 'equisized']

    def __init__(self, verts=None, faces=None, textures=None):
        """
        Args:
            verts:
                Can be either

                - List where each element is a tensor of shape (num_verts, 3)
                  containing the (x, y, z) coordinates of each vertex.
                - Padded float tensor with shape (num_meshes, max_num_verts, 3).
                  Meshes should be padded with fill value of 0 so they all have
                  the same number of vertices.
            faces:
                Can be either

                - List where each element is a tensor of shape (num_faces, 3)
                  containing the indices of the 3 vertices in the corresponding
                  mesh in verts which form the triangular face.
                - Padded long tensor of shape (num_meshes, max_num_faces, 3).
                  Meshes should be padded with fill value of -1 so they have
                  the same number of faces.
            textures: Optional instance of the Textures class with mesh
                texture properties.

        Refer to comments above for descriptions of List and Padded representations.
        """
        self.device = None
        if textures is not None and not isinstance(textures, Textures):
            msg = 'Expected textures to be of type Textures; got %r'
            raise ValueError(msg % type(textures))
        self.textures = textures
        self.equisized = False
        self.valid = None
        self._N = 0
        self._V = 0
        self._F = 0
        self._verts_list = None
        self._faces_list = None
        self._verts_packed = None
        self._verts_packed_to_mesh_idx = None
        self._verts_padded_to_packed_idx = None
        self._mesh_to_verts_packed_first_idx = None
        self._faces_packed = None
        self._faces_packed_to_mesh_idx = None
        self._mesh_to_faces_packed_first_idx = None
        self._edges_packed = None
        self._edges_packed_to_mesh_idx = None
        self._num_edges_per_mesh = None
        self._mesh_to_edges_packed_first_idx = None
        self._faces_packed_to_edges_packed = None
        self._verts_padded = None
        self._num_verts_per_mesh = None
        self._faces_padded = None
        self._num_faces_per_mesh = None
        self._faces_areas_packed = None
        self._verts_normals_packed = None
        self._faces_normals_packed = None
        self._laplacian_packed = None
        if isinstance(verts, list) and isinstance(faces, list):
            self._verts_list = verts
            self._faces_list = [(f[f.gt(-1).all(1)] if len(f) > 0 else f) for f in faces]
            self._N = len(self._verts_list)
            self.device = torch.device('cpu')
            self.valid = torch.zeros((self._N,), dtype=torch.bool, device=self.device)
            if self._N > 0:
                self.device = self._verts_list[0].device
                self._num_verts_per_mesh = torch.tensor([len(v) for v in self._verts_list], device=self.device)
                self._V = int(self._num_verts_per_mesh.max())
                self._num_faces_per_mesh = torch.tensor([len(f) for f in self._faces_list], device=self.device)
                self._F = int(self._num_faces_per_mesh.max())
                self.valid = torch.tensor([(len(v) > 0 and len(f) > 0) for v, f in zip(self._verts_list, self._faces_list)], dtype=torch.bool, device=self.device)
                if len(self._num_verts_per_mesh.unique()) == 1 and len(self._num_faces_per_mesh.unique()) == 1:
                    self.equisized = True
        elif torch.is_tensor(verts) and torch.is_tensor(faces):
            if verts.size(2) != 3 and faces.size(2) != 3:
                raise ValueError('Verts and Faces tensors have incorrect dimensions.')
            self._verts_padded = verts
            self._faces_padded = faces
            self._N = self._verts_padded.shape[0]
            self._V = self._verts_padded.shape[1]
            self.device = self._verts_padded.device
            self.valid = torch.zeros((self._N,), dtype=torch.bool, device=self.device)
            if self._N > 0:
                faces_not_padded = self._faces_padded.gt(-1).all(2)
                self._num_faces_per_mesh = faces_not_padded.sum(1)
                if (faces_not_padded[:, :-1] < faces_not_padded[:, 1:]).any():
                    raise ValueError('Padding of faces must be at the end')
                self.valid = self._num_faces_per_mesh > 0
                self._F = int(self._num_faces_per_mesh.max())
                if len(self._num_faces_per_mesh.unique()) == 1:
                    self.equisized = True
                self._num_verts_per_mesh = torch.full(size=(self._N,), fill_value=self._V, dtype=torch.int64, device=self.device)
        else:
            raise ValueError('Verts and Faces must be either a list or a tensor with                     shape (batch_size, N, 3) where N is either the maximum                        number of verts or faces respectively.')
        if self.isempty():
            self._num_verts_per_mesh = torch.zeros((0,), dtype=torch.int64, device=self.device)
            self._num_faces_per_mesh = torch.zeros((0,), dtype=torch.int64, device=self.device)
        if self.textures is not None:
            self.textures._num_faces_per_mesh = self._num_faces_per_mesh.tolist()
            self.textures._num_verts_per_mesh = self._num_verts_per_mesh.tolist()

    def __len__(self):
        return self._N

    def __getitem__(self, index):
        """
        Args:
            index: Specifying the index of the mesh to retrieve.
                Can be an int, slice, list of ints or a boolean tensor.

        Returns:
            Meshes object with selected meshes. The mesh tensors are not cloned.
        """
        if isinstance(index, (int, slice)):
            verts = self.verts_list()[index]
            faces = self.faces_list()[index]
        elif isinstance(index, list):
            verts = [self.verts_list()[i] for i in index]
            faces = [self.faces_list()[i] for i in index]
        elif isinstance(index, torch.Tensor):
            if index.dim() != 1 or index.dtype.is_floating_point:
                raise IndexError(index)
            if index.dtype == torch.bool:
                index = index.nonzero()
                index = index.squeeze(1) if index.numel() > 0 else index
                index = index.tolist()
            verts = [self.verts_list()[i] for i in index]
            faces = [self.faces_list()[i] for i in index]
        else:
            raise IndexError(index)
        textures = None if self.textures is None else self.textures[index]
        if torch.is_tensor(verts) and torch.is_tensor(faces):
            return self.__class__(verts=[verts], faces=[faces], textures=textures)
        elif isinstance(verts, list) and isinstance(faces, list):
            return self.__class__(verts=verts, faces=faces, textures=textures)
        else:
            raise ValueError('(verts, faces) not defined correctly')

    def isempty(self) ->bool:
        """
        Checks whether any mesh is valid.

        Returns:
            bool indicating whether there is any data.
        """
        return self._N == 0 or self.valid.eq(False).all()

    def verts_list(self):
        """
        Get the list representation of the vertices.

        Returns:
            list of tensors of vertices of shape (V_n, 3).
        """
        if self._verts_list is None:
            assert self._verts_padded is not None, 'verts_padded is required to compute verts_list.'
            self._verts_list = struct_utils.padded_to_list(self._verts_padded, self.num_verts_per_mesh().tolist())
        return self._verts_list

    def faces_list(self):
        """
        Get the list representation of the faces.

        Returns:
            list of tensors of faces of shape (F_n, 3).
        """
        if self._faces_list is None:
            assert self._faces_padded is not None, 'faces_padded is required to compute faces_list.'
            self._faces_list = struct_utils.padded_to_list(self._faces_padded, self.num_faces_per_mesh().tolist())
        return self._faces_list

    def verts_packed(self):
        """
        Get the packed representation of the vertices.

        Returns:
            tensor of vertices of shape (sum(V_n), 3).
        """
        self._compute_packed()
        return self._verts_packed

    def verts_packed_to_mesh_idx(self):
        """
        Return a 1D tensor with the same first dimension as verts_packed.
        verts_packed_to_mesh_idx[i] gives the index of the mesh which contains
        verts_packed[i].

        Returns:
            1D tensor of indices.
        """
        self._compute_packed()
        return self._verts_packed_to_mesh_idx

    def mesh_to_verts_packed_first_idx(self):
        """
        Return a 1D tensor x with length equal to the number of meshes such that
        the first vertex of the ith mesh is verts_packed[x[i]].

        Returns:
            1D tensor of indices of first items.
        """
        self._compute_packed()
        return self._mesh_to_verts_packed_first_idx

    def num_verts_per_mesh(self):
        """
        Return a 1D tensor x with length equal to the number of meshes giving
        the number of vertices in each mesh.

        Returns:
            1D tensor of sizes.
        """
        return self._num_verts_per_mesh

    def faces_packed(self):
        """
        Get the packed representation of the faces.
        Faces are given by the indices of the three vertices in verts_packed.

        Returns:
            tensor of faces of shape (sum(F_n), 3).
        """
        self._compute_packed()
        return self._faces_packed

    def faces_packed_to_mesh_idx(self):
        """
        Return a 1D tensor with the same first dimension as faces_packed.
        faces_packed_to_mesh_idx[i] gives the index of the mesh which contains
        faces_packed[i].

        Returns:
            1D tensor of indices.
        """
        self._compute_packed()
        return self._faces_packed_to_mesh_idx

    def mesh_to_faces_packed_first_idx(self):
        """
        Return a 1D tensor x with length equal to the number of meshes such that
        the first face of the ith mesh is faces_packed[x[i]].

        Returns:
            1D tensor of indices of first items.
        """
        self._compute_packed()
        return self._mesh_to_faces_packed_first_idx

    def verts_padded(self):
        """
        Get the padded representation of the vertices.

        Returns:
            tensor of vertices of shape (N, max(V_n), 3).
        """
        self._compute_padded()
        return self._verts_padded

    def faces_padded(self):
        """
        Get the padded representation of the faces.

        Returns:
            tensor of faces of shape (N, max(F_n), 3).
        """
        self._compute_padded()
        return self._faces_padded

    def num_faces_per_mesh(self):
        """
        Return a 1D tensor x with length equal to the number of meshes giving
        the number of faces in each mesh.

        Returns:
            1D tensor of sizes.
        """
        return self._num_faces_per_mesh

    def edges_packed(self):
        """
        Get the packed representation of the edges.

        Returns:
            tensor of edges of shape (sum(E_n), 2).
        """
        self._compute_edges_packed()
        return self._edges_packed

    def edges_packed_to_mesh_idx(self):
        """
        Return a 1D tensor with the same first dimension as edges_packed.
        edges_packed_to_mesh_idx[i] gives the index of the mesh which contains
        edges_packed[i].

        Returns:
            1D tensor of indices.
        """
        self._compute_edges_packed()
        return self._edges_packed_to_mesh_idx

    def mesh_to_edges_packed_first_idx(self):
        """
        Return a 1D tensor x with length equal to the number of meshes such that
        the first edge of the ith mesh is edges_packed[x[i]].

        Returns:
            1D tensor of indices of first items.
        """
        self._compute_edges_packed()
        return self._mesh_to_edges_packed_first_idx

    def faces_packed_to_edges_packed(self):
        """
        Get the packed representation of the faces in terms of edges.
        Faces are given by the indices of the three edges in
        the packed representation of the edges.

        Returns:
            tensor of faces of shape (sum(F_n), 3).
        """
        self._compute_edges_packed()
        return self._faces_packed_to_edges_packed

    def num_edges_per_mesh(self):
        """
        Return a 1D tensor x with length equal to the number of meshes giving
        the number of edges in each mesh.

        Returns:
            1D tensor of sizes.
        """
        self._compute_edges_packed()
        return self._num_edges_per_mesh

    def verts_padded_to_packed_idx(self):
        """
        Return a 1D tensor x with length equal to the total number of vertices
        such that verts_packed()[i] is element x[i] of the flattened padded
        representation.
        The packed representation can be calculated as follows.

        .. code-block:: python

            p = verts_padded().reshape(-1, 3)
            verts_packed = p[x]

        Returns:
            1D tensor of indices.
        """
        if self._verts_padded_to_packed_idx is not None:
            return self._verts_padded_to_packed_idx
        self._verts_padded_to_packed_idx = torch.cat([(torch.arange(v, dtype=torch.int64, device=self.device) + i * self._V) for i, v in enumerate(self.num_verts_per_mesh())], dim=0)
        return self._verts_padded_to_packed_idx

    def verts_normals_packed(self):
        """
        Get the packed representation of the vertex normals.

        Returns:
            tensor of normals of shape (sum(V_n), 3).
        """
        self._compute_vertex_normals()
        return self._verts_normals_packed

    def verts_normals_list(self):
        """
        Get the list representation of the vertex normals.

        Returns:
            list of tensors of normals of shape (V_n, 3).
        """
        if self.isempty():
            return [torch.empty((0, 3), dtype=torch.float32, device=self.device)] * self._N
        verts_normals_packed = self.verts_normals_packed()
        split_size = self.num_verts_per_mesh().tolist()
        return struct_utils.packed_to_list(verts_normals_packed, split_size)

    def verts_normals_padded(self):
        """
        Get the padded representation of the vertex normals.

        Returns:
            tensor of normals of shape (N, max(V_n), 3).
        """
        if self.isempty():
            return torch.zeros((self._N, 0, 3), dtype=torch.float32, device=self.device)
        verts_normals_list = self.verts_normals_list()
        return struct_utils.list_to_padded(verts_normals_list, (self._V, 3), pad_value=0.0, equisized=self.equisized)

    def faces_normals_packed(self):
        """
        Get the packed representation of the face normals.

        Returns:
            tensor of normals of shape (sum(F_n), 3).
        """
        self._compute_face_areas_normals()
        return self._faces_normals_packed

    def faces_normals_list(self):
        """
        Get the list representation of the face normals.

        Returns:
            list of tensors of normals of shape (F_n, 3).
        """
        if self.isempty():
            return [torch.empty((0, 3), dtype=torch.float32, device=self.device)] * self._N
        faces_normals_packed = self.faces_normals_packed()
        split_size = self.num_faces_per_mesh().tolist()
        return struct_utils.packed_to_list(faces_normals_packed, split_size)

    def faces_normals_padded(self):
        """
        Get the padded representation of the face normals.

        Returns:
            tensor of normals of shape (N, max(F_n), 3).
        """
        if self.isempty():
            return torch.zeros((self._N, 0, 3), dtype=torch.float32, device=self.device)
        faces_normals_list = self.faces_normals_list()
        return struct_utils.list_to_padded(faces_normals_list, (self._F, 3), pad_value=0.0, equisized=self.equisized)

    def faces_areas_packed(self):
        """
        Get the packed representation of the face areas.

        Returns:
            tensor of areas of shape (sum(F_n),).
        """
        self._compute_face_areas_normals()
        return self._faces_areas_packed

    def laplacian_packed(self):
        self._compute_laplacian_packed()
        return self._laplacian_packed

    def _compute_face_areas_normals(self, refresh: bool=False):
        """
        Compute the area and normal of each face in faces_packed.
        The convention of a normal for a face consisting of verts [v0, v1, v2]
        is normal = (v1 - v0) x (v2 - v0)

        Args:
            refresh: Set to True to force recomputation of face areas.
                     Default: False.
        """
        if not (refresh or any(v is None for v in [self._faces_areas_packed, self._faces_normals_packed])):
            return
        faces_packed = self.faces_packed()
        verts_packed = self.verts_packed()
        face_areas, face_normals = mesh_face_areas_normals(verts_packed, faces_packed)
        self._faces_areas_packed = face_areas
        self._faces_normals_packed = face_normals

    def _compute_vertex_normals(self, refresh: bool=False):
        """Computes the packed version of vertex normals from the packed verts
        and faces. This assumes verts are shared between faces. The normal for
        a vertex is computed as the sum of the normals of all the faces it is
        part of weighed by the face areas.

        Args:
            refresh: Set to True to force recomputation of vertex normals.
                Default: False.
        """
        if not (refresh or any(v is None for v in [self._verts_normals_packed])):
            return
        if self.isempty():
            self._verts_normals_packed = torch.zeros((self._N, 3), dtype=torch.int64, device=self.device)
        else:
            faces_packed = self.faces_packed()
            verts_packed = self.verts_packed()
            verts_normals = torch.zeros_like(verts_packed)
            vertices_faces = verts_packed[faces_packed]
            verts_normals = verts_normals.index_add(0, faces_packed[:, (1)], torch.cross(vertices_faces[:, (2)] - vertices_faces[:, (1)], vertices_faces[:, (0)] - vertices_faces[:, (1)], dim=1))
            verts_normals = verts_normals.index_add(0, faces_packed[:, (2)], torch.cross(vertices_faces[:, (0)] - vertices_faces[:, (2)], vertices_faces[:, (1)] - vertices_faces[:, (2)], dim=1))
            verts_normals = verts_normals.index_add(0, faces_packed[:, (0)], torch.cross(vertices_faces[:, (1)] - vertices_faces[:, (0)], vertices_faces[:, (2)] - vertices_faces[:, (0)], dim=1))
            self._verts_normals_packed = torch.nn.functional.normalize(verts_normals, eps=1e-06, dim=1)

    def _compute_padded(self, refresh: bool=False):
        """
        Computes the padded version of meshes from verts_list and faces_list.
        """
        if not (refresh or any(v is None for v in [self._verts_padded, self._faces_padded])):
            return
        verts_list = self.verts_list()
        faces_list = self.faces_list()
        assert faces_list is not None and verts_list is not None, 'faces_list and verts_list arguments are required'
        if self.isempty():
            self._faces_padded = torch.zeros((self._N, 0, 3), dtype=torch.int64, device=self.device)
            self._verts_padded = torch.zeros((self._N, 0, 3), dtype=torch.float32, device=self.device)
        else:
            self._faces_padded = struct_utils.list_to_padded(faces_list, (self._F, 3), pad_value=-1.0, equisized=self.equisized)
            self._verts_padded = struct_utils.list_to_padded(verts_list, (self._V, 3), pad_value=0.0, equisized=self.equisized)

    def _compute_packed(self, refresh: bool=False):
        """
        Computes the packed version of the meshes from verts_list and faces_list
        and sets the values of auxillary tensors.

        Args:
            refresh: Set to True to force recomputation of packed representations.
                Default: False.
        """
        if not (refresh or any(v is None for v in [self._verts_packed, self._verts_packed_to_mesh_idx, self._mesh_to_verts_packed_first_idx, self._faces_packed, self._faces_packed_to_mesh_idx, self._mesh_to_faces_packed_first_idx])):
            return
        verts_list = self.verts_list()
        faces_list = self.faces_list()
        if self.isempty():
            self._verts_packed = torch.zeros((0, 3), dtype=torch.float32, device=self.device)
            self._verts_packed_to_mesh_idx = torch.zeros((0,), dtype=torch.int64, device=self.device)
            self._mesh_to_verts_packed_first_idx = torch.zeros((0,), dtype=torch.int64, device=self.device)
            self._num_verts_per_mesh = torch.zeros((0,), dtype=torch.int64, device=self.device)
            self._faces_packed = -torch.ones((0, 3), dtype=torch.int64, device=self.device)
            self._faces_packed_to_mesh_idx = torch.zeros((0,), dtype=torch.int64, device=self.device)
            self._mesh_to_faces_packed_first_idx = torch.zeros((0,), dtype=torch.int64, device=self.device)
            self._num_faces_per_mesh = torch.zeros((0,), dtype=torch.int64, device=self.device)
            return
        verts_list_to_packed = struct_utils.list_to_packed(verts_list)
        self._verts_packed = verts_list_to_packed[0]
        if not torch.allclose(self.num_verts_per_mesh(), verts_list_to_packed[1]):
            raise ValueError('The number of verts per mesh should be consistent.')
        self._mesh_to_verts_packed_first_idx = verts_list_to_packed[2]
        self._verts_packed_to_mesh_idx = verts_list_to_packed[3]
        faces_list_to_packed = struct_utils.list_to_packed(faces_list)
        faces_packed = faces_list_to_packed[0]
        if not torch.allclose(self.num_faces_per_mesh(), faces_list_to_packed[1]):
            raise ValueError('The number of faces per mesh should be consistent.')
        self._mesh_to_faces_packed_first_idx = faces_list_to_packed[2]
        self._faces_packed_to_mesh_idx = faces_list_to_packed[3]
        faces_packed_offset = self._mesh_to_verts_packed_first_idx[self._faces_packed_to_mesh_idx]
        self._faces_packed = faces_packed + faces_packed_offset.view(-1, 1)

    def _compute_edges_packed(self, refresh: bool=False):
        """
        Computes edges in packed form from the packed version of faces and verts.
        """
        if not (refresh or any(v is None for v in [self._edges_packed, self._faces_packed_to_mesh_idx, self._edges_packed_to_mesh_idx, self._num_edges_per_mesh, self._mesh_to_edges_packed_first_idx])):
            return
        if self.isempty():
            self._edges_packed = torch.full((0, 2), fill_value=-1, dtype=torch.int64, device=self.device)
            self._edges_packed_to_mesh_idx = torch.zeros((0,), dtype=torch.int64, device=self.device)
            return
        faces = self.faces_packed()
        F = faces.shape[0]
        v0, v1, v2 = faces.chunk(3, dim=1)
        e01 = torch.cat([v0, v1], dim=1)
        e12 = torch.cat([v1, v2], dim=1)
        e20 = torch.cat([v2, v0], dim=1)
        edges = torch.cat([e12, e20, e01], dim=0)
        edge_to_mesh = torch.cat([self._faces_packed_to_mesh_idx, self._faces_packed_to_mesh_idx, self._faces_packed_to_mesh_idx], dim=0)
        edges, _ = edges.sort(dim=1)
        V = self._verts_packed.shape[0]
        edges_hash = V * edges[:, (0)] + edges[:, (1)]
        u, inverse_idxs = torch.unique(edges_hash, return_inverse=True)
        sorted_hash, sort_idx = torch.sort(edges_hash, dim=0)
        unique_mask = torch.ones(edges_hash.shape[0], dtype=torch.bool, device=self.device)
        unique_mask[1:] = sorted_hash[1:] != sorted_hash[:-1]
        unique_idx = sort_idx[unique_mask]
        self._edges_packed = torch.stack([u // V, u % V], dim=1)
        self._edges_packed_to_mesh_idx = edge_to_mesh[unique_idx]
        face_to_edge = torch.arange(3 * F).view(3, F).t()
        face_to_edge = inverse_idxs[face_to_edge]
        self._faces_packed_to_edges_packed = face_to_edge
        num_edges_per_mesh = torch.zeros(self._N, dtype=torch.int32, device=self.device)
        ones = torch.ones(1, dtype=torch.int32, device=self.device).expand(self._edges_packed_to_mesh_idx.shape)
        num_edges_per_mesh = num_edges_per_mesh.scatter_add_(0, self._edges_packed_to_mesh_idx, ones)
        self._num_edges_per_mesh = num_edges_per_mesh
        mesh_to_edges_packed_first_idx = torch.zeros(self._N, dtype=torch.int64, device=self.device)
        num_edges_cumsum = num_edges_per_mesh.cumsum(dim=0)
        mesh_to_edges_packed_first_idx[1:] = num_edges_cumsum[:-1].clone()
        self._mesh_to_edges_packed_first_idx = mesh_to_edges_packed_first_idx

    def _compute_laplacian_packed(self, refresh: bool=False):
        """
        Computes the laplacian in packed form.
        The definition of the laplacian is
        L[i, j] =    -1       , if i == j
        L[i, j] = 1 / deg(i)  , if (i, j) is an edge
        L[i, j] =    0        , otherwise
        where deg(i) is the degree of the i-th vertex in the graph

        Returns:
            Sparse FloatTensor of shape (V, V) where V = sum(V_n)

        """
        if not (refresh or self._laplacian_packed is None):
            return
        if self.isempty():
            self._laplacian_packed = torch.zeros((0, 0), dtype=torch.float32, device=self.device).to_sparse()
            return
        verts_packed = self.verts_packed()
        edges_packed = self.edges_packed()
        V = verts_packed.shape[0]
        e0, e1 = edges_packed.unbind(1)
        idx01 = torch.stack([e0, e1], dim=1)
        idx10 = torch.stack([e1, e0], dim=1)
        idx = torch.cat([idx01, idx10], dim=0).t()
        ones = torch.ones(idx.shape[1], dtype=torch.float32, device=self.device)
        A = torch.sparse.FloatTensor(idx, ones, (V, V))
        deg = torch.sparse.sum(A, dim=1).to_dense()
        deg0 = deg[e0]
        deg0 = torch.where(deg0 > 0.0, 1.0 / deg0, deg0)
        deg1 = deg[e1]
        deg1 = torch.where(deg1 > 0.0, 1.0 / deg1, deg1)
        val = torch.cat([deg0, deg1])
        L = torch.sparse.FloatTensor(idx, val, (V, V))
        idx = torch.arange(V, device=self.device)
        idx = torch.stack([idx, idx], dim=0)
        ones = torch.ones(idx.shape[1], dtype=torch.float32, device=self.device)
        L -= torch.sparse.FloatTensor(idx, ones, (V, V))
        self._laplacian_packed = L

    def clone(self):
        """
        Deep copy of Meshes object. All internal tensors are cloned individually.

        Returns:
            new Meshes object.
        """
        verts_list = self.verts_list()
        faces_list = self.faces_list()
        new_verts_list = [v.clone() for v in verts_list]
        new_faces_list = [f.clone() for f in faces_list]
        other = self.__class__(verts=new_verts_list, faces=new_faces_list)
        for k in self._INTERNAL_TENSORS:
            v = getattr(self, k)
            if torch.is_tensor(v):
                setattr(other, k, v.clone())
        if self.textures is not None:
            other.textures = self.textures.clone()
        return other

    def to(self, device, copy: bool=False):
        """
        Match functionality of torch.Tensor.to()
        If copy = True or the self Tensor is on a different device, the
        returned tensor is a copy of self with the desired torch.device.
        If copy = False and the self Tensor already has the correct torch.device,
        then self is returned.

        Args:
            device: Device id for the new tensor.
            copy: Boolean indicator whether or not to clone self. Default False.

        Returns:
            Meshes object.
        """
        if not copy and self.device == device:
            return self
        other = self.clone()
        if self.device != device:
            other.device = device
            if other._N > 0:
                other._verts_list = [v for v in other._verts_list]
                other._faces_list = [f for f in other._faces_list]
            for k in self._INTERNAL_TENSORS:
                v = getattr(self, k)
                if torch.is_tensor(v):
                    setattr(other, k, v)
            if self.textures is not None:
                other.textures = self.textures
        return other

    def cpu(self):
        return self

    def cuda(self):
        return self

    def get_mesh_verts_faces(self, index: int):
        """
        Get tensors for a single mesh from the list representation.

        Args:
            index: Integer in the range [0, N).

        Returns:
            verts: Tensor of shape (V, 3).
            faces: LongTensor of shape (F, 3).
        """
        if not isinstance(index, int):
            raise ValueError('Mesh index must be an integer.')
        if index < 0 or index > self._N:
            raise ValueError('Mesh index must be in the range [0, N) where             N is the number of meshes in the batch.')
        verts = self.verts_list()
        faces = self.faces_list()
        return verts[index], faces[index]

    def split(self, split_sizes: list):
        """
        Splits Meshes object of size N into a list of Meshes objects of
        size len(split_sizes), where the i-th Meshes object is of size split_sizes[i].
        Similar to torch.split().

        Args:
            split_sizes: List of integer sizes of Meshes objects to be returned.

        Returns:
            list[Meshes].
        """
        if not all(isinstance(x, int) for x in split_sizes):
            raise ValueError('Value of split_sizes must be a list of integers.')
        meshlist = []
        curi = 0
        for i in split_sizes:
            meshlist.append(self[curi:curi + i])
            curi += i
        return meshlist

    def offset_verts_(self, vert_offsets_packed):
        """
        Add an offset to the vertices of this Meshes. In place operation.

        Args:
            vert_offsets_packed: A Tensor of the same shape as self.verts_packed
                               giving offsets to be added to all vertices.
        Returns:
            self.
        """
        verts_packed = self.verts_packed()
        if vert_offsets_packed.shape != verts_packed.shape:
            raise ValueError('Verts offsets must have dimension (all_v, 2).')
        self._verts_packed = verts_packed + vert_offsets_packed
        new_verts_list = list(self._verts_packed.split(self.num_verts_per_mesh().tolist(), 0))
        self._verts_list = new_verts_list
        if self._verts_padded is not None:
            for i, verts in enumerate(new_verts_list):
                if len(verts) > 0:
                    self._verts_padded[(i), :verts.shape[0], :] = verts
        if any(v is not None for v in [self._faces_areas_packed, self._faces_normals_packed]):
            self._compute_face_areas_normals(refresh=True)
        if self._verts_normals_packed is not None:
            self._compute_vertex_normals(refresh=True)
        return self

    def offset_verts(self, vert_offsets_packed):
        """
        Out of place offset_verts.

        Args:
            vert_offsets_packed: A Tensor of the same shape as self.verts_packed
                giving offsets to be added to all vertices.
        Returns:
            new Meshes object.
        """
        new_mesh = self.clone()
        return new_mesh.offset_verts_(vert_offsets_packed)

    def scale_verts_(self, scale):
        """
        Multiply the vertices of this Meshes object by a scalar value.
        In place operation.

        Args:
            scale: A scalar, or a Tensor of shape (N,).

        Returns:
            self.
        """
        if not torch.is_tensor(scale):
            scale = torch.full((len(self),), scale, device=self.device)
        new_verts_list = []
        verts_list = self.verts_list()
        for i, old_verts in enumerate(verts_list):
            new_verts_list.append(scale[i] * old_verts)
        self._verts_list = new_verts_list
        if self._verts_packed is not None:
            self._verts_packed = torch.cat(new_verts_list, dim=0)
        if self._verts_padded is not None:
            for i, verts in enumerate(self._verts_list):
                if len(verts) > 0:
                    self._verts_padded[(i), :verts.shape[0], :] = verts
        if any(v is not None for v in [self._faces_areas_packed, self._faces_normals_packed]):
            self._compute_face_areas_normals(refresh=True)
        if self._verts_normals_packed is not None:
            self._compute_vertex_normals(refresh=True)
        return self

    def scale_verts(self, scale):
        """
        Out of place scale_verts.

        Args:
            scale: A scalar, or a Tensor of shape (N,).

        Returns:
            new Meshes object.
        """
        new_mesh = self.clone()
        return new_mesh.scale_verts_(scale)

    def update_padded(self, new_verts_padded):
        """
        This function allows for an pdate of verts_padded without having to
        explicitly convert it to the list representation for heterogeneous batches.
        Returns a Meshes structure with updated padded tensors and copies of the
        auxiliary tensors at construction time.
        It updates self._verts_padded with new_verts_padded, and does a
        shallow copy of (faces_padded, faces_list, num_verts_per_mesh, num_faces_per_mesh).
        If packed representations are computed in self, they are updated as well.

        Args:
            new_points_padded: FloatTensor of shape (N, V, 3)

        Returns:
            Meshes with updated padded representations
        """

        def check_shapes(x, size):
            if x.shape[0] != size[0]:
                raise ValueError('new values must have the same batch dimension.')
            if x.shape[1] != size[1]:
                raise ValueError('new values must have the same number of points.')
            if x.shape[2] != size[2]:
                raise ValueError('new values must have the same dimension.')
        check_shapes(new_verts_padded, [self._N, self._V, 3])
        new = self.__class__(verts=new_verts_padded, faces=self.faces_padded())
        if new._N != self._N or new._V != self._V or new._F != self._F:
            raise ValueError('Inconsistent sizes after construction.')
        new.equisized = self.equisized
        new.textures = self.textures
        copy_tensors = ['_num_verts_per_mesh', '_num_faces_per_mesh', 'valid']
        for k in copy_tensors:
            v = getattr(self, k)
            if torch.is_tensor(v):
                setattr(new, k, v)
        new._faces_list = self._faces_list
        if self._verts_packed is not None:
            copy_tensors = ['_faces_packed', '_verts_packed_to_mesh_idx', '_faces_packed_to_mesh_idx', '_mesh_to_verts_packed_first_idx', '_mesh_to_faces_packed_first_idx']
            for k in copy_tensors:
                v = getattr(self, k)
                assert torch.is_tensor(v)
                setattr(new, k, v)
            pad_to_packed = self.verts_padded_to_packed_idx()
            new_verts_packed = new_verts_padded.reshape(-1, 3)[(pad_to_packed), :]
            new._verts_packed = new_verts_packed
            new._verts_padded_to_packed_idx = pad_to_packed
        if self._edges_packed is not None:
            copy_tensors = ['_edges_packed', '_edges_packed_to_mesh_idx', '_mesh_to_edges_packed_first_idx', '_faces_packed_to_edges_packed', '_num_edges_per_mesh']
            for k in copy_tensors:
                v = getattr(self, k)
                assert torch.is_tensor(v)
                setattr(new, k, v)
        if self._laplacian_packed is not None:
            new._laplacian_packed = self._laplacian_packed
        assert new._verts_list is None
        assert new._verts_normals_packed is None
        assert new._faces_normals_packed is None
        assert new._faces_areas_packed is None
        return new

    def get_bounding_boxes(self):
        """
        Compute an axis-aligned bounding box for each mesh in this Meshes object.

        Returns:
            bboxes: Tensor of shape (N, 3, 2) where bbox[i, j] gives the
            min and max values of mesh i along the jth coordinate axis.
        """
        all_mins, all_maxes = [], []
        for verts in self.verts_list():
            cur_mins = verts.min(dim=0)[0]
            cur_maxes = verts.max(dim=0)[0]
            all_mins.append(cur_mins)
            all_maxes.append(cur_maxes)
        all_mins = torch.stack(all_mins, dim=0)
        all_maxes = torch.stack(all_maxes, dim=0)
        bboxes = torch.stack([all_mins, all_maxes], dim=2)
        return bboxes

    def extend(self, N: int):
        """
        Create new Meshes class which contains each input mesh N times

        Args:
            N: number of new copies of each mesh.

        Returns:
            new Meshes object.
        """
        if not isinstance(N, int):
            raise ValueError('N must be an integer.')
        if N <= 0:
            raise ValueError('N must be > 0.')
        new_verts_list, new_faces_list = [], []
        for verts, faces in zip(self.verts_list(), self.faces_list()):
            new_verts_list.extend(verts.clone() for _ in range(N))
            new_faces_list.extend(faces.clone() for _ in range(N))
        tex = None
        if self.textures is not None:
            tex = self.textures.extend(N)
        return self.__class__(verts=new_verts_list, faces=new_faces_list, textures=tex)


def create_faces_index(faces_per_mesh, device=None):
    """
    Helper function to group the faces indices for each mesh. New faces are
    stacked at the end of the original faces tensor, so in order to have
    sequential packing, the faces tensor needs to be reordered to that faces
    corresponding to each mesh are grouped together.

    Args:
        faces_per_mesh: Tensor of shape (N,) giving the number of faces
            in each mesh in the batch where N is the batch size.

    Returns:
        faces_idx: A tensor with face indices for each mesh ordered sequentially
            by mesh index.
    """
    F = faces_per_mesh.sum()
    faces_per_mesh_cumsum = faces_per_mesh.cumsum(dim=0)
    switch1_idx = faces_per_mesh_cumsum.clone()
    switch1_idx[1:] += 3 * faces_per_mesh_cumsum[:-1]
    switch2_idx = 2 * faces_per_mesh_cumsum
    switch2_idx[1:] += 2 * faces_per_mesh_cumsum[:-1]
    switch3_idx = 3 * faces_per_mesh_cumsum
    switch3_idx[1:] += faces_per_mesh_cumsum[:-1]
    switch4_idx = 4 * faces_per_mesh_cumsum[:-1]
    switch123_offset = F - faces_per_mesh
    idx_diffs = torch.ones(4 * F, device=device, dtype=torch.int64)
    idx_diffs[switch1_idx] += switch123_offset
    idx_diffs[switch2_idx] += switch123_offset
    idx_diffs[switch3_idx] += switch123_offset
    idx_diffs[switch4_idx] -= 3 * F
    faces_idx = idx_diffs.cumsum(dim=0) - 1
    return faces_idx


def create_verts_index(verts_per_mesh, edges_per_mesh, device=None):
    """
    Helper function to group the vertex indices for each mesh. New vertices are
    stacked at the end of the original verts tensor, so in order to have
    sequential packing, the verts tensor needs to be reordered so that the
    vertices corresponding to each mesh are grouped together.

    Args:
        verts_per_mesh: Tensor of shape (N,) giving the number of vertices
            in each mesh in the batch where N is the batch size.
        edges_per_mesh: Tensor of shape (N,) giving the number of edges
            in each mesh in the batch

    Returns:
        verts_idx: A tensor with vert indices for each mesh ordered sequentially
            by mesh index.
    """
    V = verts_per_mesh.sum()
    E = edges_per_mesh.sum()
    verts_per_mesh_cumsum = verts_per_mesh.cumsum(dim=0)
    edges_per_mesh_cumsum = edges_per_mesh.cumsum(dim=0)
    v_to_e_idx = verts_per_mesh_cumsum.clone()
    v_to_e_idx[1:] += edges_per_mesh_cumsum[:-1]
    v_to_e_offset = V - verts_per_mesh_cumsum
    v_to_e_offset[1:] += edges_per_mesh_cumsum[:-1]
    e_to_v_idx = verts_per_mesh_cumsum[:-1] + edges_per_mesh_cumsum[:-1]
    e_to_v_offset = verts_per_mesh_cumsum[:-1] - edges_per_mesh_cumsum[:-1] - V
    idx_diffs = torch.ones(V + E, device=device, dtype=torch.int64)
    idx_diffs[v_to_e_idx] += v_to_e_offset
    idx_diffs[e_to_v_idx] += e_to_v_offset
    verts_idx = idx_diffs.cumsum(dim=0) - 1
    return verts_idx


class SubdivideMeshes(nn.Module):
    """
    Subdivide a triangle mesh by adding a new vertex at the center of each edge
    and dividing each face into four new faces. Vectors of vertex
    attributes can also be subdivided by averaging the values of the attributes
    at the two vertices which form each edge. This implementation
    preserves face orientation - if the vertices of a face are all ordered
    counter-clockwise, then the faces in the subdivided meshes will also have
    their vertices ordered counter-clockwise.

    If meshes is provided as an input, the initializer performs the relatively
    expensive computation of determining the new face indices. This one-time
    computation can be reused for all meshes with the same face topology
    but different vertex positions.
    """

    def __init__(self, meshes=None):
        """
        Args:
            meshes: Meshes object or None. If a meshes object is provided,
                the first mesh is used to compute the new faces of the
                subdivided topology which can be reused for meshes with
                the same input topology.
        """
        super(SubdivideMeshes, self).__init__()
        self.precomputed = False
        self._N = -1
        if meshes is not None:
            mesh = meshes[0]
            with torch.no_grad():
                subdivided_faces = self.subdivide_faces(mesh)
                if subdivided_faces.shape[1] != 3:
                    raise ValueError('faces can only have three vertices')
                self.register_buffer('_subdivided_faces', subdivided_faces)
                self.precomputed = True

    def subdivide_faces(self, meshes):
        """
        Args:
            meshes: a Meshes object.

        Returns:
            subdivided_faces_packed: (4*sum(F_n), 3) shape LongTensor of
            original and new faces.

        Refer to pytorch3d.structures.meshes.py for more details on packed
        representations of faces.

        Each face is split into 4 faces e.g. Input face
        ::
                   v0
                   /\\
                  /  \\
                 /    \\
             e1 /      \\ e0
               /        \\
              /          \\
             /            \\
            /______________\\
          v2       e2       v1

          faces_packed = [[0, 1, 2]]
          faces_packed_to_edges_packed = [[2, 1, 0]]

        `faces_packed_to_edges_packed` is used to represent all the new
        vertex indices corresponding to the mid-points of edges in the mesh.
        The actual vertex coordinates will be computed in the forward function.
        To get the indices of the new vertices, offset
        `faces_packed_to_edges_packed` by the total number of vertices.
        ::
            faces_packed_to_edges_packed = [[2, 1, 0]] + 3 = [[5, 4, 3]]

        e.g. subdivided face
        ::
                   v0
                   /\\
                  /  \\
                 / f0 \\
             v4 /______\\ v3
               /\\      /\\
              /  \\ f3 /  \\
             / f2 \\  / f1 \\
            /______\\/______\\
           v2       v5       v1

           f0 = [0, 3, 4]
           f1 = [1, 5, 3]
           f2 = [2, 4, 5]
           f3 = [5, 4, 3]

        """
        verts_packed = meshes.verts_packed()
        with torch.no_grad():
            faces_packed = meshes.faces_packed()
            faces_packed_to_edges_packed = meshes.faces_packed_to_edges_packed()
            faces_packed_to_edges_packed += verts_packed.shape[0]
            f0 = torch.stack([faces_packed[:, (0)], faces_packed_to_edges_packed[:, (2)], faces_packed_to_edges_packed[:, (1)]], dim=1)
            f1 = torch.stack([faces_packed[:, (1)], faces_packed_to_edges_packed[:, (0)], faces_packed_to_edges_packed[:, (2)]], dim=1)
            f2 = torch.stack([faces_packed[:, (2)], faces_packed_to_edges_packed[:, (1)], faces_packed_to_edges_packed[:, (0)]], dim=1)
            f3 = faces_packed_to_edges_packed
            subdivided_faces_packed = torch.cat([f0, f1, f2, f3], dim=0)
            return subdivided_faces_packed

    def forward(self, meshes, feats=None):
        """
        Subdivide a batch of meshes by adding a new vertex on each edge, and
        dividing each face into four new faces. New meshes contains two types
        of vertices:
        1) Vertices that appear in the input meshes.
           Data for these vertices are copied from the input meshes.
        2) New vertices at the midpoint of each edge.
           Data for these vertices is the average of the data for the two
           vertices that make up the edge.

        Args:
            meshes: Meshes object representing a batch of meshes.
            feats: Per-vertex features to be subdivided along with the verts.
                Should be parallel to the packed vert representation of the
                input meshes; so it should have shape (V, D) where V is the
                total number of verts in the input meshes. Default: None.

        Returns:
            2-element tuple containing

            - **new_meshes**: Meshes object of a batch of subdivided meshes.
            - **new_feats**: (optional) Tensor of subdivided feats, parallel to the
              (packed) vertices of the subdivided meshes. Only returned
              if feats is not None.

        """
        self._N = len(meshes)
        if self.precomputed:
            return self.subdivide_homogeneous(meshes, feats)
        else:
            return self.subdivide_heterogenerous(meshes, feats)

    def subdivide_homogeneous(self, meshes, feats=None):
        """
        Subdivide verts (and optionally features) of a batch of meshes
        where each mesh has the same topology of faces. The subdivided faces
        are precomputed in the initializer.

        Args:
            meshes: Meshes object representing a batch of meshes.
            feats: Per-vertex features to be subdivided along with the verts.

        Returns:
            2-element tuple containing

            - **new_meshes**: Meshes object of a batch of subdivided meshes.
            - **new_feats**: (optional) Tensor of subdivided feats, parallel to the
              (packed) vertices of the subdivided meshes. Only returned
              if feats is not None.
        """
        verts = meshes.verts_padded()
        edges = meshes[0].edges_packed()
        new_faces = self._subdivided_faces.view(1, -1, 3).expand(self._N, -1, -1)
        new_verts = verts[:, (edges)].mean(dim=2)
        new_verts = torch.cat([verts, new_verts], dim=1)
        new_feats = None
        if feats is not None:
            if feats.dim() == 2:
                feats = feats.view(verts.size(0), verts.size(1), feats.size(1))
            if feats.dim() != 3:
                raise ValueError('features need to be of shape (N, V, D) or (N*V, D)')
            new_feats = feats[:, (edges)].mean(dim=2)
            new_feats = torch.cat([feats, new_feats], dim=1)
        new_meshes = Meshes(verts=new_verts, faces=new_faces)
        if feats is None:
            return new_meshes
        else:
            return new_meshes, new_feats

    def subdivide_heterogenerous(self, meshes, feats=None):
        """
        Subdivide faces, verts (and optionally features) of a batch of meshes
        where each mesh can have different face topologies.

        Args:
            meshes: Meshes object representing a batch of meshes.
            feats: Per-vertex features to be subdivided along with the verts.

        Returns:
            2-element tuple containing

            - **new_meshes**: Meshes object of a batch of subdivided meshes.
            - **new_feats**: (optional) Tensor of subdivided feats, parallel to the
              (packed) vertices of the subdivided meshes. Only returned
              if feats is not None.
        """
        verts = meshes.verts_packed()
        with torch.no_grad():
            new_faces = self.subdivide_faces(meshes)
            edges = meshes.edges_packed()
            face_to_mesh_idx = meshes.faces_packed_to_mesh_idx()
            edge_to_mesh_idx = meshes.edges_packed_to_mesh_idx()
            num_edges_per_mesh = edge_to_mesh_idx.bincount(minlength=self._N)
            num_verts_per_mesh = meshes.num_verts_per_mesh()
            num_faces_per_mesh = meshes.num_faces_per_mesh()
            new_verts_per_mesh = num_verts_per_mesh + num_edges_per_mesh
            new_face_to_mesh_idx = torch.cat([face_to_mesh_idx] * 4, dim=0)
            verts_sort_idx = create_verts_index(num_verts_per_mesh, num_edges_per_mesh, meshes.device)
            verts_ordered_idx_init = torch.zeros(new_verts_per_mesh.sum(), dtype=torch.int64, device=meshes.device)
            verts_ordered_idx = verts_ordered_idx_init.scatter_add(0, verts_sort_idx, torch.arange(new_verts_per_mesh.sum(), device=meshes.device))
            new_faces = verts_ordered_idx[new_faces]
            face_sort_idx = create_faces_index(num_faces_per_mesh, device=meshes.device)
            new_faces = new_faces[face_sort_idx]
            new_face_to_mesh_idx = new_face_to_mesh_idx[face_sort_idx]
            new_faces_per_mesh = new_face_to_mesh_idx.bincount(minlength=self._N)
        new_verts = verts[edges].mean(dim=1)
        new_verts = torch.cat([verts, new_verts], dim=0)
        new_verts = new_verts[verts_sort_idx]
        if feats is not None:
            new_feats = feats[edges].mean(dim=1)
            new_feats = torch.cat([feats, new_feats], dim=0)
            new_feats = new_feats[verts_sort_idx]
        verts_list = list(new_verts.split(new_verts_per_mesh.tolist(), 0))
        faces_list = list(new_faces.split(new_faces_per_mesh.tolist(), 0))
        new_verts_per_mesh_cumsum = torch.cat([new_verts_per_mesh.new_full(size=(1,), fill_value=0.0), new_verts_per_mesh.cumsum(0)[:-1]], dim=0)
        faces_list = [(faces_list[n] - new_verts_per_mesh_cumsum[n]) for n in range(self._N)]
        if feats is not None:
            feats_list = new_feats.split(new_verts_per_mesh.tolist(), 0)
        new_meshes = Meshes(verts=verts_list, faces=faces_list)
        if feats is None:
            return new_meshes
        else:
            new_feats = torch.cat(feats_list, dim=0)
            return new_meshes, new_feats


class Fragments(NamedTuple):
    pix_to_face: torch.Tensor
    zbuf: torch.Tensor
    bary_coords: torch.Tensor
    dists: torch.Tensor


class RasterizationSettings:
    __slots__ = ['image_size', 'blur_radius', 'faces_per_pixel', 'bin_size', 'max_faces_per_bin', 'perspective_correct', 'cull_backfaces']

    def __init__(self, image_size: int=256, blur_radius: float=0.0, faces_per_pixel: int=1, bin_size: Optional[int]=None, max_faces_per_bin: Optional[int]=None, perspective_correct: bool=False, cull_backfaces: bool=False):
        self.image_size = image_size
        self.blur_radius = blur_radius
        self.faces_per_pixel = faces_per_pixel
        self.bin_size = bin_size
        self.max_faces_per_bin = max_faces_per_bin
        self.perspective_correct = perspective_correct
        self.cull_backfaces = cull_backfaces


class _RasterizeFaceVerts(torch.autograd.Function):
    """
    Torch autograd wrapper for forward and backward pass of rasterize_meshes
    implemented in C++/CUDA.

    Args:
        face_verts: Tensor of shape (F, 3, 3) giving (packed) vertex positions
            for faces in all the meshes in the batch. Concretely,
            face_verts[f, i] = [x, y, z] gives the coordinates for the
            ith vertex of the fth face. These vertices are expected to
            be in NDC coordinates in the range [-1, 1].
        mesh_to_face_first_idx: LongTensor of shape (N) giving the index in
            faces_verts of the first face in each mesh in
            the batch.
        num_faces_per_mesh: LongTensor of shape (N) giving the number of faces
            for each mesh in the batch.
        image_size, blur_radius, faces_per_pixel: same as rasterize_meshes.
        perspective_correct: same as rasterize_meshes.
        cull_backfaces: same as rasterize_meshes.

    Returns:
        same as rasterize_meshes function.
    """

    @staticmethod
    def forward(ctx, face_verts, mesh_to_face_first_idx, num_faces_per_mesh, image_size: int=256, blur_radius: float=0.01, faces_per_pixel: int=0, bin_size: int=0, max_faces_per_bin: int=0, perspective_correct: bool=False, cull_backfaces: bool=False):
        pix_to_face, zbuf, barycentric_coords, dists = _C.rasterize_meshes(face_verts, mesh_to_face_first_idx, num_faces_per_mesh, image_size, blur_radius, faces_per_pixel, bin_size, max_faces_per_bin, perspective_correct, cull_backfaces)
        ctx.save_for_backward(face_verts, pix_to_face)
        ctx.mark_non_differentiable(pix_to_face)
        ctx.perspective_correct = perspective_correct
        return pix_to_face, zbuf, barycentric_coords, dists

    @staticmethod
    def backward(ctx, grad_pix_to_face, grad_zbuf, grad_barycentric_coords, grad_dists):
        grad_face_verts = None
        grad_mesh_to_face_first_idx = None
        grad_num_faces_per_mesh = None
        grad_image_size = None
        grad_radius = None
        grad_faces_per_pixel = None
        grad_bin_size = None
        grad_max_faces_per_bin = None
        grad_perspective_correct = None
        grad_cull_backfaces = None
        face_verts, pix_to_face = ctx.saved_tensors
        grad_face_verts = _C.rasterize_meshes_backward(face_verts, pix_to_face, grad_zbuf, grad_barycentric_coords, grad_dists, ctx.perspective_correct)
        grads = grad_face_verts, grad_mesh_to_face_first_idx, grad_num_faces_per_mesh, grad_image_size, grad_radius, grad_faces_per_pixel, grad_bin_size, grad_max_faces_per_bin, grad_perspective_correct, grad_cull_backfaces
        return grads


kMaxFacesPerBin = 22


def rasterize_meshes(meshes, image_size: int=256, blur_radius: float=0.0, faces_per_pixel: int=8, bin_size: Optional[int]=None, max_faces_per_bin: Optional[int]=None, perspective_correct: bool=False, cull_backfaces: bool=False):
    """
    Rasterize a batch of meshes given the shape of the desired output image.
    Each mesh is rasterized onto a separate image of shape
    (image_size, image_size).

    Args:
        meshes: A Meshes object representing a batch of meshes, batch size N.
        image_size: Size in pixels of the output raster image for each mesh
            in the batch. Assumes square images.
        blur_radius: Float distance in the range [0, 2] used to expand the face
            bounding boxes for rasterization. Setting blur radius
            results in blurred edges around the shape instead of a
            hard boundary. Set to 0 for no blur.
        faces_per_pixel (Optional): Number of faces to save per pixel, returning
            the nearest faces_per_pixel points along the z-axis.
        bin_size: Size of bins to use for coarse-to-fine rasterization. Setting
            bin_size=0 uses naive rasterization; setting bin_size=None attempts to
            set it heuristically based on the shape of the input. This should not
            affect the output, but can affect the speed of the forward pass.
        faces_per_bin: Only applicable when using coarse-to-fine rasterization
            (bin_size > 0); this is the maxiumum number of faces allowed within each
            bin. If more than this many faces actually fall into a bin, an error
            will be raised. This should not affect the output values, but can affect
            the memory usage in the forward pass.
        perspective_correct: Bool, Whether to apply perspective correction when computing
            barycentric coordinates for pixels.
        cull_backfaces: Bool, Whether to only rasterize mesh faces which are
            visible to the camera.  This assumes that vertices of
            front-facing triangles are ordered in an anti-clockwise
            fashion, and triangles that face away from the camera are
            in a clockwise order relative to the current view
            direction. NOTE: This will only work if the mesh faces are
            consistently defined with counter-clockwise ordering when
            viewed from the outside.

    Returns:
        4-element tuple containing

        - **pix_to_face**: LongTensor of shape
          (N, image_size, image_size, faces_per_pixel)
          giving the indices of the nearest faces at each pixel,
          sorted in ascending z-order.
          Concretely ``pix_to_face[n, y, x, k] = f`` means that
          ``faces_verts[f]`` is the kth closest face (in the z-direction)
          to pixel (y, x). Pixels that are hit by fewer than
          faces_per_pixel are padded with -1.
        - **zbuf**: FloatTensor of shape (N, image_size, image_size, faces_per_pixel)
          giving the NDC z-coordinates of the nearest faces at each pixel,
          sorted in ascending z-order.
          Concretely, if ``pix_to_face[n, y, x, k] = f`` then
          ``zbuf[n, y, x, k] = face_verts[f, 2]``. Pixels hit by fewer than
          faces_per_pixel are padded with -1.
        - **barycentric**: FloatTensor of shape
          (N, image_size, image_size, faces_per_pixel, 3)
          giving the barycentric coordinates in NDC units of the
          nearest faces at each pixel, sorted in ascending z-order.
          Concretely, if ``pix_to_face[n, y, x, k] = f`` then
          ``[w0, w1, w2] = barycentric[n, y, x, k]`` gives
          the barycentric coords for pixel (y, x) relative to the face
          defined by ``face_verts[f]``. Pixels hit by fewer than
          faces_per_pixel are padded with -1.
        - **pix_dists**: FloatTensor of shape
          (N, image_size, image_size, faces_per_pixel)
          giving the signed Euclidean distance (in NDC units) in the
          x/y plane of each point closest to the pixel. Concretely if
          ``pix_to_face[n, y, x, k] = f`` then ``pix_dists[n, y, x, k]`` is the
          squared distance between the pixel (y, x) and the face given
          by vertices ``face_verts[f]``. Pixels hit with fewer than
          ``faces_per_pixel`` are padded with -1.
    """
    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    face_verts = verts_packed[faces_packed]
    mesh_to_face_first_idx = meshes.mesh_to_faces_packed_first_idx()
    num_faces_per_mesh = meshes.num_faces_per_mesh()
    if bin_size is None:
        if not verts_packed.is_cuda:
            bin_size = 0
        elif image_size <= 64:
            bin_size = 8
        else:
            bin_size = int(2 ** max(np.ceil(np.log2(image_size)) - 4, 4))
    if bin_size != 0:
        faces_per_bin = 1 + (image_size - 1) // bin_size
        if faces_per_bin >= kMaxFacesPerBin:
            raise ValueError('bin_size too small, number of faces per bin must be less than %d; got %d' % (kMaxFacesPerBin, faces_per_bin))
    if max_faces_per_bin is None:
        max_faces_per_bin = int(max(10000, verts_packed.shape[0] / 5))
    return _RasterizeFaceVerts.apply(face_verts, mesh_to_face_first_idx, num_faces_per_mesh, image_size, blur_radius, faces_per_pixel, bin_size, max_faces_per_bin, perspective_correct, cull_backfaces)


class MeshRasterizer(nn.Module):
    """
    This class implements methods for rasterizing a batch of heterogenous
    Meshes.
    """

    def __init__(self, cameras=None, raster_settings=None):
        """
        Args:
            cameras: A cameras object which has a  `transform_points` method
                which returns the transformed points after applying the
                world-to-view and view-to-screen
                transformations.
            raster_settings: the parameters for rasterization. This should be a
                named tuple.

        All these initial settings can be overridden by passing keyword
        arguments to the forward function.
        """
        super().__init__()
        if raster_settings is None:
            raster_settings = RasterizationSettings()
        self.cameras = cameras
        self.raster_settings = raster_settings

    def transform(self, meshes_world, **kwargs) ->torch.Tensor:
        """
        Args:
            meshes_world: a Meshes object representing a batch of meshes with
                vertex coordinates in world space.

        Returns:
            meshes_screen: a Meshes object with the vertex positions in screen
            space

        NOTE: keeping this as a separate function for readability but it could
        be moved into forward.
        """
        cameras = kwargs.get('cameras', self.cameras)
        if cameras is None:
            msg = 'Cameras must be specified either at initialization                 or in the forward pass of MeshRasterizer'
            raise ValueError(msg)
        verts_world = meshes_world.verts_padded()
        verts_view = cameras.get_world_to_view_transform(**kwargs).transform_points(verts_world)
        verts_screen = cameras.get_projection_transform(**kwargs).transform_points(verts_view)
        verts_screen[..., 2] = verts_view[..., 2]
        meshes_screen = meshes_world.update_padded(new_verts_padded=verts_screen)
        return meshes_screen

    def forward(self, meshes_world, **kwargs) ->Fragments:
        """
        Args:
            meshes_world: a Meshes object representing a batch of meshes with
                          coordinates in world space.
        Returns:
            Fragments: Rasterization outputs as a named tuple.
        """
        meshes_screen = self.transform(meshes_world, **kwargs)
        raster_settings = kwargs.get('raster_settings', self.raster_settings)
        pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(meshes_screen, image_size=raster_settings.image_size, blur_radius=raster_settings.blur_radius, faces_per_pixel=raster_settings.faces_per_pixel, bin_size=raster_settings.bin_size, max_faces_per_bin=raster_settings.max_faces_per_bin, perspective_correct=raster_settings.perspective_correct, cull_backfaces=raster_settings.cull_backfaces)
        return Fragments(pix_to_face=pix_to_face, zbuf=zbuf, bary_coords=bary_coords, dists=dists)


def _clip_barycentric_coordinates(bary) ->torch.Tensor:
    """
    Args:
        bary: barycentric coordinates of shape (...., 3) where `...` represents
            an arbitrary number of dimensions

    Returns:
        bary: Barycentric coordinates clipped (i.e any values < 0 are set to 0)
        and renormalized. We only clip  the negative values. Values > 1 will fall
        into the [0, 1] range after renormalization.
        The output is the same shape as the input.
    """
    if bary.shape[-1] != 3:
        msg = 'Expected barycentric coords to have last dim = 3; got %r'
        raise ValueError(msg % (bary.shape,))
    clipped = bary.clamp(min=0.0)
    clipped_sum = torch.clamp(clipped.sum(dim=-1, keepdim=True), min=1e-05)
    clipped = clipped / clipped_sum
    return clipped


def interpolate_face_attributes(pix_to_face: torch.Tensor, barycentric_coords: torch.Tensor, face_attributes: torch.Tensor) ->torch.Tensor:
    """
    Interpolate arbitrary face attributes using the barycentric coordinates
    for each pixel in the rasterized output.

    Args:
        pix_to_face: LongTensor of shape (N, H, W, K) specifying the indices
            of the faces (in the packed representation) which
            overlap each pixel in the image.
        barycentric_coords: FloatTensor of shape (N, H, W, K, 3) specifying
            the barycentric coordianates of each pixel
            relative to the faces (in the packed
            representation) which overlap the pixel.
        face_attributes: packed attributes of shape (total_faces, 3, D),
            specifying the value of the attribute for each
            vertex in the face.

    Returns:
        pixel_vals: tensor of shape (N, H, W, K, D) giving the interpolated
        value of the face attribute for each pixel.
    """
    F, FV, D = face_attributes.shape
    if FV != 3:
        raise ValueError('Faces can only have three vertices; got %r' % FV)
    N, H, W, K, _ = barycentric_coords.shape
    if pix_to_face.shape != (N, H, W, K):
        msg = 'pix_to_face must have shape (batch_size, H, W, K); got %r'
        raise ValueError(msg % (pix_to_face.shape,))
    mask = pix_to_face == -1
    pix_to_face = pix_to_face.clone()
    pix_to_face[mask] = 0
    idx = pix_to_face.view(N * H * W * K, 1, 1).expand(N * H * W * K, 3, D)
    pixel_face_vals = face_attributes.gather(0, idx).view(N, H, W, K, 3, D)
    pixel_vals = (barycentric_coords[..., None] * pixel_face_vals).sum(dim=-2)
    pixel_vals[mask] = 0
    return pixel_vals


def _interpolate_zbuf(pix_to_face: torch.Tensor, barycentric_coords: torch.Tensor, meshes) ->torch.Tensor:
    """
    A helper function to calculate the z buffer for each pixel in the
    rasterized output.

    Args:
        pix_to_face: LongTensor of shape (N, H, W, K) specifying the indices
            of the faces (in the packed representation) which
            overlap each pixel in the image.
        barycentric_coords: FloatTensor of shape (N, H, W, K, 3) specifying
            the barycentric coordianates of each pixel
            relative to the faces (in the packed
            representation) which overlap the pixel.
        meshes: Meshes object representing a batch of meshes.

    Returns:
        zbuffer: (N, H, W, K) FloatTensor
    """
    verts = meshes.verts_packed()
    faces = meshes.faces_packed()
    faces_verts_z = verts[faces][..., 2][..., None]
    return interpolate_face_attributes(pix_to_face, barycentric_coords, faces_verts_z)[..., 0]


class MeshRenderer(nn.Module):
    """
    A class for rendering a batch of heterogeneous meshes. The class should
    be initialized with a rasterizer and shader class which each have a forward
    function.
    """

    def __init__(self, rasterizer, shader):
        super().__init__()
        self.rasterizer = rasterizer
        self.shader = shader

    def forward(self, meshes_world, **kwargs) ->torch.Tensor:
        """
        Render a batch of images from a batch of meshes by rasterizing and then
        shading.

        NOTE: If the blur radius for rasterization is > 0.0, some pixels can
        have one or more barycentric coordinates lying outside the range [0, 1].
        For a pixel with out of bounds barycentric coordinates with respect to a
        face f, clipping is required before interpolating the texture uv
        coordinates and z buffer so that the colors and depths are limited to
        the range for the corresponding face.
        """
        fragments = self.rasterizer(meshes_world, **kwargs)
        raster_settings = kwargs.get('raster_settings', self.rasterizer.raster_settings)
        if raster_settings.blur_radius > 0.0:
            clipped_bary_coords = _clip_barycentric_coordinates(fragments.bary_coords)
            clipped_zbuf = _interpolate_zbuf(fragments.pix_to_face, clipped_bary_coords, meshes_world)
            fragments = Fragments(bary_coords=clipped_bary_coords, zbuf=clipped_zbuf, dists=fragments.dists, pix_to_face=fragments.pix_to_face)
        images = self.shader(fragments, meshes_world, **kwargs)
        return images


class BlendParams(NamedTuple):
    sigma: float = 0.0001
    gamma: float = 0.0001
    background_color: Sequence = (1.0, 1.0, 1.0)


BROADCAST_TYPES = float, int, list, tuple, torch.Tensor, np.ndarray


class TensorAccessor(object):
    """
    A helper class to be used with the __getitem__ method. This can be used for
    getting/setting the values for an attribute of a class at one particular
    index.  This is useful when the attributes of a class are batched tensors
    and one element in the batch needs to be modified.
    """

    def __init__(self, class_object, index: Union[int, slice]):
        """
        Args:
            class_object: this should be an instance of a class which has
                attributes which are tensors representing a batch of
                values.
            index: int/slice, an index indicating the position in the batch.
                In __setattr__ and __getattr__ only the value of class
                attributes at this index will be accessed.
        """
        self.__dict__['class_object'] = class_object
        self.__dict__['index'] = index

    def __setattr__(self, name: str, value: Any):
        """
        Update the attribute given by `name` to the value given by `value`
        at the index specified by `self.index`.

        Args:
            name: str, name of the attribute.
            value: value to set the attribute to.
        """
        v = getattr(self.class_object, name)
        if not torch.is_tensor(v):
            msg = 'Can only set values on attributes which are tensors; got %r'
            raise AttributeError(msg % type(v))
        if not torch.is_tensor(value):
            value = torch.tensor(value, device=v.device, dtype=v.dtype, requires_grad=v.requires_grad)
        if v.dim() > 1 and value.dim() > 1 and value.shape[1:] != v.shape[1:]:
            msg = 'Expected value to have shape %r; got %r'
            raise ValueError(msg % (v.shape, value.shape))
        if v.dim() == 0 and isinstance(self.index, slice) and len(value) != len(self.index):
            msg = 'Expected value to have len %r; got %r'
            raise ValueError(msg % (len(self.index), len(value)))
        self.class_object.__dict__[name][self.index] = value

    def __getattr__(self, name: str):
        """
        Return the value of the attribute given by "name" on self.class_object
        at the index specified in self.index.

        Args:
            name: string of the attribute name
        """
        if hasattr(self.class_object, name):
            return self.class_object.__dict__[name][self.index]
        else:
            msg = 'Attribue %s not found on %r'
            return AttributeError(msg % (name, self.class_object.__name__))


def format_tensor(input, dtype=torch.float32, device: str='cpu') ->torch.Tensor:
    """
    Helper function for converting a scalar value to a tensor.

    Args:
        input: Python scalar, Python list/tuple, torch scalar, 1D torch tensor
        dtype: data type for the input
        device: torch device on which the tensor should be placed.

    Returns:
        input_vec: torch tensor with optional added batch dimension.
    """
    if not torch.is_tensor(input):
        input = torch.tensor(input, dtype=dtype, device=device)
    if input.dim() == 0:
        input = input.view(1)
    if input.device != device:
        input = input
    return input


def convert_to_tensors_and_broadcast(*args, dtype=torch.float32, device: str='cpu'):
    """
    Helper function to handle parsing an arbitrary number of inputs (*args)
    which all need to have the same batch dimension.
    The output is a list of tensors.

    Args:
        *args: an arbitrary number of inputs
            Each of the values in `args` can be one of the following
                - Python scalar
                - Torch scalar
                - Torch tensor of shape (N, K_i) or (1, K_i) where K_i are
                  an arbitrary number of dimensions which can vary for each
                  value in args. In this case each input is broadcast to a
                  tensor of shape (N, K_i)
        dtype: data type to use when creating new tensors.
        device: torch device on which the tensors should be placed.

    Output:
        args: A list of tensors of shape (N, K_i)
    """
    args_1d = [format_tensor(c, dtype, device) for c in args]
    sizes = [c.shape[0] for c in args_1d]
    N = max(sizes)
    args_Nd = []
    for c in args_1d:
        if c.shape[0] != 1 and c.shape[0] != N:
            msg = 'Got non-broadcastable sizes %r' % sizes
            raise ValueError(msg)
        expand_sizes = (N,) + (-1,) * len(c.shape[1:])
        args_Nd.append(c.expand(*expand_sizes))
    if len(args) == 1:
        args_Nd = args_Nd[0]
    return args_Nd


class TensorProperties(object):
    """
    A mix-in class for storing tensors as properties with helper methods.
    """

    def __init__(self, dtype=torch.float32, device='cpu', **kwargs):
        """
        Args:
            dtype: data type to set for the inputs
            device: str or torch.device
            kwargs: any number of keyword arguments. Any arguments which are
                of type (float/int/tuple/tensor/array) are broadcasted and
                other keyword arguments are set as attributes.
        """
        super().__init__()
        self.device = device
        self._N = 0
        if kwargs is not None:
            args_to_broadcast = {}
            for k, v in kwargs.items():
                if isinstance(v, (str, bool)):
                    setattr(self, k, v)
                elif isinstance(v, BROADCAST_TYPES):
                    args_to_broadcast[k] = v
                else:
                    msg = 'Arg %s with type %r is not broadcastable'
                    warnings.warn(msg % (k, type(v)))
            names = args_to_broadcast.keys()
            values = tuple(v for v in args_to_broadcast.values())
            if len(values) > 0:
                broadcasted_values = convert_to_tensors_and_broadcast(*values, device=device)
                for i, n in enumerate(names):
                    setattr(self, n, broadcasted_values[i])
                    if self._N == 0:
                        self._N = broadcasted_values[i].shape[0]

    def __len__(self) ->int:
        return self._N

    def isempty(self) ->bool:
        return self._N == 0

    def __getitem__(self, index: Union[int, slice]):
        """

        Args:
            index: an int or slice used to index all the fields.

        Returns:
            if `index` is an index int/slice return a TensorAccessor class
            with getattribute/setattribute methods which return/update the value
            at the index in the original camera.
        """
        if isinstance(index, (int, slice)):
            return TensorAccessor(class_object=self, index=index)
        msg = 'Expected index of type int or slice; got %r'
        raise ValueError(msg % type(index))

    def to(self, device: str='cpu'):
        """
        In place operation to move class properties which are tensors to a
        specified device. If self has a property "device", update this as well.
        """
        for k in dir(self):
            v = getattr(self, k)
            if k == 'device':
                setattr(self, k, device)
            if torch.is_tensor(v) and v.device != device:
                setattr(self, k, v)
        return self

    def clone(self, other):
        """
        Update the tensor properties of other with the cloned properties of self.
        """
        for k in dir(self):
            v = getattr(self, k)
            if inspect.ismethod(v) or k.startswith('__'):
                continue
            if torch.is_tensor(v):
                v_clone = v.clone()
            else:
                v_clone = copy.deepcopy(v)
            setattr(other, k, v_clone)
        return other

    def gather_props(self, batch_idx):
        """
        This is an in place operation to reformat all tensor class attributes
        based on a set of given indices using torch.gather. This is useful when
        attributes which are batched tensors e.g. shape (N, 3) need to be
        multiplied with another tensor which has a different first dimension
        e.g. packed vertices of shape (V, 3).

        Example

        .. code-block:: python

            self.specular_color = (N, 3) tensor of specular colors for each mesh

        A lighting calculation may use

        .. code-block:: python

            verts_packed = meshes.verts_packed()  # (V, 3)

        To multiply these two tensors the batch dimension needs to be the same.
        To achieve this we can do

        .. code-block:: python

            batch_idx = meshes.verts_packed_to_mesh_idx()  # (V)

        This gives index of the mesh for each vertex in verts_packed.

        .. code-block:: python

            self.gather_props(batch_idx)
            self.specular_color = (V, 3) tensor with the specular color for
                                     each packed vertex.

        torch.gather requires the index tensor to have the same shape as the
        input tensor so this method takes care of the reshaping of the index
        tensor to use with class attributes with arbitrary dimensions.

        Args:
            batch_idx: shape (B, ...) where `...` represents an arbitrary
                number of dimensions

        Returns:
            self with all properties reshaped. e.g. a property with shape (N, 3)
            is transformed to shape (B, 3).
        """
        for k in dir(self):
            v = getattr(self, k)
            if torch.is_tensor(v):
                if v.shape[0] > 1:
                    _batch_idx = batch_idx.clone()
                    idx_dims = _batch_idx.shape
                    tensor_dims = v.shape
                    if len(idx_dims) > len(tensor_dims):
                        msg = 'batch_idx cannot have more dimensions than %s. '
                        msg += 'got shape %r and %s has shape %r'
                        raise ValueError(msg % (k, idx_dims, k, tensor_dims))
                    if idx_dims != tensor_dims:
                        new_dims = len(tensor_dims) - len(idx_dims)
                        new_shape = idx_dims + (1,) * new_dims
                        expand_dims = (-1,) + tensor_dims[1:]
                        _batch_idx = _batch_idx.view(*new_shape)
                        _batch_idx = _batch_idx.expand(*expand_dims)
                    v = v.gather(0, _batch_idx)
                    setattr(self, k, v)
        return self


class Materials(TensorProperties):
    """
    A class for storing a batch of material properties. Currently only one
    material per batch element is supported.
    """

    def __init__(self, ambient_color=((1, 1, 1),), diffuse_color=((1, 1, 1),), specular_color=((1, 1, 1),), shininess=64, device='cpu'):
        """
        Args:
            ambient_color: RGB ambient reflectivity of the material
            diffuse_color: RGB diffuse reflectivity of the material
            specular_color: RGB specular reflectivity of the material
            shininess: The specular exponent for the material. This defines
                the focus of the specular highlight with a high value
                resulting in a concentrated highlight. Shininess values
                can range from 0-1000.
            device: torch.device or string

        ambient_color, diffuse_color and specular_color can be of shape
        (1, 3) or (N, 3). shininess can be of shape (1) or (N).

        The colors and shininess are broadcast against each other so need to
        have either the same batch dimension or batch dimension = 1.
        """
        super().__init__(device=device, diffuse_color=diffuse_color, ambient_color=ambient_color, specular_color=specular_color, shininess=shininess)
        for n in ['ambient_color', 'diffuse_color', 'specular_color']:
            t = getattr(self, n)
            if t.shape[-1] != 3:
                msg = 'Expected %s to have shape (N, 3); got %r'
                raise ValueError(msg % (n, t.shape))
        if self.shininess.shape != torch.Size([self._N]):
            msg = 'shininess should have shape (N); got %r'
            raise ValueError(msg % repr(self.shininess.shape))

    def clone(self):
        other = Materials(device=self.device)
        return super().clone(other)


def _validate_light_properties(obj):
    props = 'ambient_color', 'diffuse_color', 'specular_color'
    for n in props:
        t = getattr(obj, n)
        if t.shape[-1] != 3:
            msg = 'Expected %s to have shape (N, 3); got %r'
            raise ValueError(msg % (n, t.shape))


class PointLights(TensorProperties):

    def __init__(self, ambient_color=((0.5, 0.5, 0.5),), diffuse_color=((0.3, 0.3, 0.3),), specular_color=((0.2, 0.2, 0.2),), location=((0, 1, 0),), device: str='cpu'):
        """
        Args:
            ambient_color: RGB color of the ambient component
            diffuse_color: RGB color of the diffuse component
            specular_color: RGB color of the specular component
            location: xyz position of the light.
            device: torch.device on which the tensors should be located

        The inputs can each be
            - 3 element tuple/list or list of lists
            - torch tensor of shape (1, 3)
            - torch tensor of shape (N, 3)
        The inputs are broadcast against each other so they all have batch
        dimension N.
        """
        super().__init__(device=device, ambient_color=ambient_color, diffuse_color=diffuse_color, specular_color=specular_color, location=location)
        _validate_light_properties(self)
        if self.location.shape[-1] != 3:
            msg = 'Expected location to have shape (N, 3); got %r'
            raise ValueError(msg % repr(self.location.shape))

    def clone(self):
        other = self.__class__(device=self.device)
        return super().clone(other)

    def diffuse(self, normals, points) ->torch.Tensor:
        direction = self.location - points
        return diffuse(normals=normals, color=self.diffuse_color, direction=direction)

    def specular(self, normals, points, camera_position, shininess) ->torch.Tensor:
        direction = self.location - points
        return specular(points=points, normals=normals, color=self.specular_color, direction=direction, camera_position=camera_position, shininess=shininess)


def hard_rgb_blend(colors, fragments, blend_params) ->torch.Tensor:
    """
    Naive blending of top K faces to return an RGBA image
      - **RGB** - choose color of the closest point i.e. K=0
      - **A** - 1.0

    Args:
        colors: (N, H, W, K, 3) RGB color for each of the top K faces per pixel.
        fragments: the outputs of rasterization. From this we use
            - pix_to_face: LongTensor of shape (N, H, W, K) specifying the indices
              of the faces (in the packed representation) which
              overlap each pixel in the image. This is used to
              determine the output shape.
        blend_params: BlendParams instance that contains a background_color
        field specifying the color for the background
    Returns:
        RGBA pixel_colors: (N, H, W, 4)
    """
    N, H, W, K = fragments.pix_to_face.shape
    device = fragments.pix_to_face.device
    is_background = fragments.pix_to_face[..., 0] < 0
    background_color = colors.new_tensor(blend_params.background_color)
    num_background_pixels = is_background.sum()
    pixel_colors = colors[(...), (0), :].masked_scatter(is_background[..., None], background_color[(None), :].expand(num_background_pixels, -1))
    alpha = torch.ones((N, H, W, 1), dtype=colors.dtype, device=device)
    return torch.cat([pixel_colors, alpha], dim=-1)


def interpolate_vertex_colors(fragments, meshes) ->torch.Tensor:
    """
    Detemine the color for each rasterized face. Interpolate the colors for
    vertices which form the face using the barycentric coordinates.
    Args:
        meshes: A Meshes class representing a batch of meshes.
        fragments:
            The outputs of rasterization. From this we use

            - pix_to_face: LongTensor of shape (N, H, W, K) specifying the indices
              of the faces (in the packed representation) which
              overlap each pixel in the image.
            - barycentric_coords: FloatTensor of shape (N, H, W, K, 3) specifying
              the barycentric coordianates of each pixel
              relative to the faces (in the packed
              representation) which overlap the pixel.

    Returns:
        texels: An texture per pixel of shape (N, H, W, K, C).
        There will be one C dimensional value for each element in
        fragments.pix_to_face.
    """
    vertex_textures = meshes.textures.verts_rgb_padded().reshape(-1, 3)
    vertex_textures = vertex_textures[(meshes.verts_padded_to_packed_idx()), :]
    faces_packed = meshes.faces_packed()
    faces_textures = vertex_textures[faces_packed]
    texels = interpolate_face_attributes(fragments.pix_to_face, fragments.bary_coords, faces_textures)
    return texels


def _apply_lighting(points, normals, lights, cameras, materials) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Args:
        points: torch tensor of shape (N, P, 3) or (P, 3).
        normals: torch tensor of shape (N, P, 3) or (P, 3)
        lights: instance of the Lights class.
        cameras: instance of the Cameras class.
        materials: instance of the Materials class.

    Returns:
        ambient_color: same shape as materials.ambient_color
        diffuse_color: same shape as the input points
        specular_color: same shape as the input points
    """
    light_diffuse = lights.diffuse(normals=normals, points=points)
    light_specular = lights.specular(normals=normals, points=points, camera_position=cameras.get_camera_center(), shininess=materials.shininess)
    ambient_color = materials.ambient_color * lights.ambient_color
    diffuse_color = materials.diffuse_color * light_diffuse
    specular_color = materials.specular_color * light_specular
    if normals.dim() == 2 and points.dim() == 2:
        return ambient_color.squeeze(), diffuse_color.squeeze(), specular_color.squeeze()
    return ambient_color, diffuse_color, specular_color


def phong_shading(meshes, fragments, lights, cameras, materials, texels) ->torch.Tensor:
    """
    Apply per pixel shading. First interpolate the vertex normals and
    vertex coordinates using the barycentric coordinates to get the position
    and normal at each pixel. Then compute the illumination for each pixel.
    The pixel color is obtained by multiplying the pixel textures by the ambient
    and diffuse illumination and adding the specular component.

    Args:
        meshes: Batch of meshes
        fragments: Fragments named tuple with the outputs of rasterization
        lights: Lights class containing a batch of lights
        cameras: Cameras class containing a batch of cameras
        materials: Materials class containing a batch of material properties
        texels: texture per pixel of shape (N, H, W, K, 3)

    Returns:
        colors: (N, H, W, K, 3)
    """
    verts = meshes.verts_packed()
    faces = meshes.faces_packed()
    vertex_normals = meshes.verts_normals_packed()
    faces_verts = verts[faces]
    faces_normals = vertex_normals[faces]
    pixel_coords = interpolate_face_attributes(fragments.pix_to_face, fragments.bary_coords, faces_verts)
    pixel_normals = interpolate_face_attributes(fragments.pix_to_face, fragments.bary_coords, faces_normals)
    ambient, diffuse, specular = _apply_lighting(pixel_coords, pixel_normals, lights, cameras, materials)
    colors = (ambient + diffuse) * texels + specular
    return colors


class HardPhongShader(nn.Module):
    """
    Per pixel lighting - the lighting model is applied using the interpolated
    coordinates and normals for each pixel. The blending function hard assigns
    the color of the closest face for each pixel.

    To use the default values, simply initialize the shader with the desired
    device e.g.

    .. code-block::

        shader = HardPhongShader(device=torch.device("cuda:0"))
    """

    def __init__(self, device='cpu', cameras=None, lights=None, materials=None, blend_params=None):
        super().__init__()
        self.lights = lights if lights is not None else PointLights(device=device)
        self.materials = materials if materials is not None else Materials(device=device)
        self.cameras = cameras
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments, meshes, **kwargs) ->torch.Tensor:
        cameras = kwargs.get('cameras', self.cameras)
        if cameras is None:
            msg = 'Cameras must be specified either at initialization                 or in the forward pass of HardPhongShader'
            raise ValueError(msg)
        texels = interpolate_vertex_colors(fragments, meshes)
        lights = kwargs.get('lights', self.lights)
        materials = kwargs.get('materials', self.materials)
        blend_params = kwargs.get('blend_params', self.blend_params)
        colors = phong_shading(meshes=meshes, fragments=fragments, texels=texels, lights=lights, cameras=cameras, materials=materials)
        images = hard_rgb_blend(colors, fragments, blend_params)
        return images


def softmax_rgb_blend(colors, fragments, blend_params, znear: float=1.0, zfar: float=100) ->torch.Tensor:
    """
    RGB and alpha channel blending to return an RGBA image based on the method
    proposed in [1]
      - **RGB** - blend the colors based on the 2D distance based probability map and
        relative z distances.
      - **A** - blend based on the 2D distance based probability map.

    Args:
        colors: (N, H, W, K, 3) RGB color for each of the top K faces per pixel.
        fragments: namedtuple with outputs of rasterization. We use properties
            - pix_to_face: LongTensor of shape (N, H, W, K) specifying the indices
              of the faces (in the packed representation) which
              overlap each pixel in the image.
            - dists: FloatTensor of shape (N, H, W, K) specifying
              the 2D euclidean distance from the center of each pixel
              to each of the top K overlapping faces.
            - zbuf: FloatTensor of shape (N, H, W, K) specifying
              the interpolated depth from each pixel to to each of the
              top K overlapping faces.
        blend_params: instance of BlendParams dataclass containing properties
            - sigma: float, parameter which controls the width of the sigmoid
              function used to calculate the 2D distance based probability.
              Sigma controls the sharpness of the edges of the shape.
            - gamma: float, parameter which controls the scaling of the
              exponential function used to control the opacity of the color.
            - background_color: (3) element list/tuple/torch.Tensor specifying
              the RGB values for the background color.
        znear: float, near clipping plane in the z direction
        zfar: float, far clipping plane in the z direction

    Returns:
        RGBA pixel_colors: (N, H, W, 4)

    [0] Shichen Liu et al, 'Soft Rasterizer: A Differentiable Renderer for
    Image-based 3D Reasoning'
    """
    N, H, W, K = fragments.pix_to_face.shape
    device = fragments.pix_to_face.device
    pixel_colors = torch.ones((N, H, W, 4), dtype=colors.dtype, device=colors.device)
    background = blend_params.background_color
    if not torch.is_tensor(background):
        background = torch.tensor(background, dtype=torch.float32, device=device)
    delta = np.exp(1e-10 / blend_params.gamma) * 1e-10
    delta = torch.tensor(delta, device=device)
    mask = fragments.pix_to_face >= 0
    prob_map = torch.sigmoid(-fragments.dists / blend_params.sigma) * mask
    alpha = torch.prod(1.0 - prob_map, dim=-1)
    z_inv = (zfar - fragments.zbuf) / (zfar - znear) * mask
    z_inv_max = torch.max(z_inv, dim=-1).values[..., None]
    weights_num = prob_map * torch.exp((z_inv - z_inv_max) / blend_params.gamma)
    denom = weights_num.sum(dim=-1)[..., None] + delta
    weights = weights_num / denom
    weighted_colors = (weights[..., None] * colors).sum(dim=-2)
    weighted_background = delta / denom * background
    pixel_colors[(...), :3] = weighted_colors + weighted_background
    pixel_colors[..., 3] = 1.0 - alpha
    return pixel_colors


class SoftPhongShader(nn.Module):
    """
    Per pixel lighting - the lighting model is applied using the interpolated
    coordinates and normals for each pixel. The blending function returns the
    soft aggregated color using all the faces per pixel.

    To use the default values, simply initialize the shader with the desired
    device e.g.

    .. code-block::

        shader = SoftPhongShader(device=torch.device("cuda:0"))
    """

    def __init__(self, device='cpu', cameras=None, lights=None, materials=None, blend_params=None):
        super().__init__()
        self.lights = lights if lights is not None else PointLights(device=device)
        self.materials = materials if materials is not None else Materials(device=device)
        self.cameras = cameras
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments, meshes, **kwargs) ->torch.Tensor:
        cameras = kwargs.get('cameras', self.cameras)
        if cameras is None:
            msg = 'Cameras must be specified either at initialization                 or in the forward pass of SoftPhongShader'
            raise ValueError(msg)
        texels = interpolate_vertex_colors(fragments, meshes)
        lights = kwargs.get('lights', self.lights)
        materials = kwargs.get('materials', self.materials)
        colors = phong_shading(meshes=meshes, fragments=fragments, texels=texels, lights=lights, cameras=cameras, materials=materials)
        images = softmax_rgb_blend(colors, fragments, self.blend_params)
        return images


def gouraud_shading(meshes, fragments, lights, cameras, materials) ->torch.Tensor:
    """
    Apply per vertex shading. First compute the vertex illumination by applying
    ambient, diffuse and specular lighting. If vertex color is available,
    combine the ambient and diffuse vertex illumination with the vertex color
    and add the specular component to determine the vertex shaded color.
    Then interpolate the vertex shaded colors using the barycentric coordinates
    to get a color per pixel.

    Args:
        meshes: Batch of meshes
        fragments: Fragments named tuple with the outputs of rasterization
        lights: Lights class containing a batch of lights parameters
        cameras: Cameras class containing a batch of cameras parameters
        materials: Materials class containing a batch of material properties

    Returns:
        colors: (N, H, W, K, 3)
    """
    faces = meshes.faces_packed()
    verts = meshes.verts_packed()
    vertex_normals = meshes.verts_normals_packed()
    vertex_colors = meshes.textures.verts_rgb_packed()
    vert_to_mesh_idx = meshes.verts_packed_to_mesh_idx()
    if len(meshes) > 1:
        lights = lights.clone().gather_props(vert_to_mesh_idx)
        cameras = cameras.clone().gather_props(vert_to_mesh_idx)
        materials = materials.clone().gather_props(vert_to_mesh_idx)
    ambient, diffuse, specular = _apply_lighting(verts, vertex_normals, lights, cameras, materials)
    verts_colors_shaded = vertex_colors * (ambient + diffuse) + specular
    face_colors = verts_colors_shaded[faces]
    colors = interpolate_face_attributes(fragments.pix_to_face, fragments.bary_coords, face_colors)
    return colors


class HardGouraudShader(nn.Module):
    """
    Per vertex lighting - the lighting model is applied to the vertex colors and
    the colors are then interpolated using the barycentric coordinates to
    obtain the colors for each pixel. The blending function hard assigns
    the color of the closest face for each pixel.

    To use the default values, simply initialize the shader with the desired
    device e.g.

    .. code-block::

        shader = HardGouraudShader(device=torch.device("cuda:0"))
    """

    def __init__(self, device='cpu', cameras=None, lights=None, materials=None, blend_params=None):
        super().__init__()
        self.lights = lights if lights is not None else PointLights(device=device)
        self.materials = materials if materials is not None else Materials(device=device)
        self.cameras = cameras
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments, meshes, **kwargs) ->torch.Tensor:
        cameras = kwargs.get('cameras', self.cameras)
        if cameras is None:
            msg = 'Cameras must be specified either at initialization                 or in the forward pass of HardGouraudShader'
            raise ValueError(msg)
        lights = kwargs.get('lights', self.lights)
        materials = kwargs.get('materials', self.materials)
        blend_params = kwargs.get('blend_params', self.blend_params)
        pixel_colors = gouraud_shading(meshes=meshes, fragments=fragments, lights=lights, cameras=cameras, materials=materials)
        images = hard_rgb_blend(pixel_colors, fragments, blend_params)
        return images


class SoftGouraudShader(nn.Module):
    """
    Per vertex lighting - the lighting model is applied to the vertex colors and
    the colors are then interpolated using the barycentric coordinates to
    obtain the colors for each pixel. The blending function returns the
    soft aggregated color using all the faces per pixel.

    To use the default values, simply initialize the shader with the desired
    device e.g.

    .. code-block::

        shader = SoftGouraudShader(device=torch.device("cuda:0"))
    """

    def __init__(self, device='cpu', cameras=None, lights=None, materials=None, blend_params=None):
        super().__init__()
        self.lights = lights if lights is not None else PointLights(device=device)
        self.materials = materials if materials is not None else Materials(device=device)
        self.cameras = cameras
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments, meshes, **kwargs) ->torch.Tensor:
        cameras = kwargs.get('cameras', self.cameras)
        if cameras is None:
            msg = 'Cameras must be specified either at initialization                 or in the forward pass of SoftGouraudShader'
            raise ValueError(msg)
        lights = kwargs.get('lights', self.lights)
        materials = kwargs.get('materials', self.materials)
        pixel_colors = gouraud_shading(meshes=meshes, fragments=fragments, lights=lights, cameras=cameras, materials=materials)
        images = softmax_rgb_blend(pixel_colors, fragments, self.blend_params)
        return images


def interpolate_texture_map(fragments, meshes) ->torch.Tensor:
    """
    Interpolate a 2D texture map using uv vertex texture coordinates for each
    face in the mesh. First interpolate the vertex uvs using barycentric coordinates
    for each pixel in the rasterized output. Then interpolate the texture map
    using the uv coordinate for each pixel.

    Args:
        fragments:
            The outputs of rasterization. From this we use

            - pix_to_face: LongTensor of shape (N, H, W, K) specifying the indices
              of the faces (in the packed representation) which
              overlap each pixel in the image.
            - barycentric_coords: FloatTensor of shape (N, H, W, K, 3) specifying
              the barycentric coordianates of each pixel
              relative to the faces (in the packed
              representation) which overlap the pixel.
        meshes: Meshes representing a batch of meshes. It is expected that
            meshes has a textures attribute which is an instance of the
            Textures class.

    Returns:
        texels: tensor of shape (N, H, W, K, C) giving the interpolated
        texture for each pixel in the rasterized image.
    """
    if not isinstance(meshes.textures, Textures):
        msg = 'Expected meshes.textures to be an instance of Textures; got %r'
        raise ValueError(msg % type(meshes.textures))
    faces_uvs = meshes.textures.faces_uvs_packed()
    verts_uvs = meshes.textures.verts_uvs_packed()
    faces_verts_uvs = verts_uvs[faces_uvs]
    texture_maps = meshes.textures.maps_padded()
    pixel_uvs = interpolate_face_attributes(fragments.pix_to_face, fragments.bary_coords, faces_verts_uvs)
    N, H_out, W_out, K = fragments.pix_to_face.shape
    N, H_in, W_in, C = texture_maps.shape
    pixel_uvs = pixel_uvs.permute(0, 3, 1, 2, 4).reshape(N * K, H_out, W_out, 2)
    texture_maps = texture_maps.permute(0, 3, 1, 2)[None, ...].expand(K, -1, -1, -1, -1).transpose(0, 1).reshape(N * K, C, H_in, W_in)
    pixel_uvs = pixel_uvs * 2.0 - 1.0
    texture_maps = torch.flip(texture_maps, [2])
    if texture_maps.device != pixel_uvs.device:
        texture_maps = texture_maps
    texels = F.grid_sample(texture_maps, pixel_uvs, align_corners=False)
    texels = texels.reshape(N, K, C, H_out, W_out).permute(0, 3, 4, 1, 2)
    return texels


class TexturedSoftPhongShader(nn.Module):
    """
    Per pixel lighting applied to a texture map. First interpolate the vertex
    uv coordinates and sample from a texture map. Then apply the lighting model
    using the interpolated coords and normals for each pixel.

    The blending function returns the soft aggregated color using all
    the faces per pixel.

    To use the default values, simply initialize the shader with the desired
    device e.g.

    .. code-block::

        shader = TexturedPhongShader(device=torch.device("cuda:0"))
    """

    def __init__(self, device='cpu', cameras=None, lights=None, materials=None, blend_params=None):
        super().__init__()
        self.lights = lights if lights is not None else PointLights(device=device)
        self.materials = materials if materials is not None else Materials(device=device)
        self.cameras = cameras
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments, meshes, **kwargs) ->torch.Tensor:
        cameras = kwargs.get('cameras', self.cameras)
        if cameras is None:
            msg = 'Cameras must be specified either at initialization                 or in the forward pass of TexturedSoftPhongShader'
            raise ValueError(msg)
        texels = interpolate_texture_map(fragments, meshes)
        lights = kwargs.get('lights', self.lights)
        materials = kwargs.get('materials', self.materials)
        blend_params = kwargs.get('blend_params', self.blend_params)
        colors = phong_shading(meshes=meshes, fragments=fragments, texels=texels, lights=lights, cameras=cameras, materials=materials)
        images = softmax_rgb_blend(colors, fragments, blend_params)
        return images


def flat_shading(meshes, fragments, lights, cameras, materials, texels) ->torch.Tensor:
    """
    Apply per face shading. Use the average face position and the face normals
    to compute the ambient, diffuse and specular lighting. Apply the ambient
    and diffuse color to the pixel color and add the specular component to
    determine the final pixel color.

    Args:
        meshes: Batch of meshes
        fragments: Fragments named tuple with the outputs of rasterization
        lights: Lights class containing a batch of lights parameters
        cameras: Cameras class containing a batch of cameras parameters
        materials: Materials class containing a batch of material properties
        texels: texture per pixel of shape (N, H, W, K, 3)

    Returns:
        colors: (N, H, W, K, 3)
    """
    verts = meshes.verts_packed()
    faces = meshes.faces_packed()
    face_normals = meshes.faces_normals_packed()
    faces_verts = verts[faces]
    face_coords = faces_verts.mean(dim=-2)
    mask = fragments.pix_to_face == -1
    pix_to_face = fragments.pix_to_face.clone()
    pix_to_face[mask] = 0
    N, H, W, K = pix_to_face.shape
    idx = pix_to_face.view(N * H * W * K, 1).expand(N * H * W * K, 3)
    pixel_coords = face_coords.gather(0, idx).view(N, H, W, K, 3)
    pixel_coords[mask] = 0.0
    pixel_normals = face_normals.gather(0, idx).view(N, H, W, K, 3)
    pixel_normals[mask] = 0.0
    ambient, diffuse, specular = _apply_lighting(pixel_coords, pixel_normals, lights, cameras, materials)
    colors = (ambient + diffuse) * texels + specular
    return colors


class HardFlatShader(nn.Module):
    """
    Per face lighting - the lighting model is applied using the average face
    position and the face normal. The blending function hard assigns
    the color of the closest face for each pixel.

    To use the default values, simply initialize the shader with the desired
    device e.g.

    .. code-block::

        shader = HardFlatShader(device=torch.device("cuda:0"))
    """

    def __init__(self, device='cpu', cameras=None, lights=None, materials=None, blend_params=None):
        super().__init__()
        self.lights = lights if lights is not None else PointLights(device=device)
        self.materials = materials if materials is not None else Materials(device=device)
        self.cameras = cameras
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments, meshes, **kwargs) ->torch.Tensor:
        cameras = kwargs.get('cameras', self.cameras)
        if cameras is None:
            msg = 'Cameras must be specified either at initialization                 or in the forward pass of HardFlatShader'
            raise ValueError(msg)
        texels = interpolate_vertex_colors(fragments, meshes)
        lights = kwargs.get('lights', self.lights)
        materials = kwargs.get('materials', self.materials)
        blend_params = kwargs.get('blend_params', self.blend_params)
        colors = flat_shading(meshes=meshes, fragments=fragments, texels=texels, lights=lights, cameras=cameras, materials=materials)
        images = hard_rgb_blend(colors, fragments, blend_params)
        return images


def sigmoid_alpha_blend(colors, fragments, blend_params) ->torch.Tensor:
    """
    Silhouette blending to return an RGBA image
      - **RGB** - choose color of the closest point.
      - **A** - blend based on the 2D distance based probability map [1].

    Args:
        colors: (N, H, W, K, 3) RGB color for each of the top K faces per pixel.
        fragments: the outputs of rasterization. From this we use
            - pix_to_face: LongTensor of shape (N, H, W, K) specifying the indices
              of the faces (in the packed representation) which
              overlap each pixel in the image.
            - dists: FloatTensor of shape (N, H, W, K) specifying
              the 2D euclidean distance from the center of each pixel
              to each of the top K overlapping faces.

    Returns:
        RGBA pixel_colors: (N, H, W, 4)

    [1] Liu et al, 'Soft Rasterizer: A Differentiable Renderer for Image-based
        3D Reasoning', ICCV 2019
    """
    N, H, W, K = fragments.pix_to_face.shape
    pixel_colors = torch.ones((N, H, W, 4), dtype=colors.dtype, device=colors.device)
    mask = fragments.pix_to_face >= 0
    prob = torch.sigmoid(-fragments.dists / blend_params.sigma) * mask
    alpha = torch.prod(1.0 - prob, dim=-1)
    pixel_colors[(...), :3] = colors[(...), (0), :]
    pixel_colors[..., 3] = 1.0 - alpha
    return pixel_colors


class SoftSilhouetteShader(nn.Module):
    """
    Calculate the silhouette by blending the top K faces for each pixel based
    on the 2d euclidean distance of the center of the pixel to the mesh face.

    Use this shader for generating silhouettes similar to SoftRasterizer [0].

    .. note::

        To be consistent with SoftRasterizer, initialize the
        RasterizationSettings for the rasterizer with
        `blur_radius = np.log(1. / 1e-4 - 1.) * blend_params.sigma`

    [0] Liu et al, 'Soft Rasterizer: A Differentiable Renderer for Image-based
        3D Reasoning', ICCV 2019
    """

    def __init__(self, blend_params=None):
        super().__init__()
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments, meshes, **kwargs) ->torch.Tensor:
        """"
        Only want to render the silhouette so RGB values can be ones.
        There is no need for lighting or texturing
        """
        colors = torch.ones_like(fragments.bary_coords)
        blend_params = kwargs.get('blend_params', self.blend_params)
        images = sigmoid_alpha_blend(colors, fragments, blend_params)
        return images


class CompositeParams(NamedTuple):
    radius: float = 4.0 / 256.0


class _CompositeAlphaPoints(torch.autograd.Function):
    """
    Composite features within a z-buffer using alpha compositing. Given a z-buffer
    with corresponding features and weights, these values are accumulated according
    to their weights such that features nearer in depth contribute more to the final
    feature than ones further away.

    Concretely this means:
        weighted_fs[b,c,i,j] = sum_k cum_alpha_k * features[c,pointsidx[b,k,i,j]]
        cum_alpha_k = alphas[b,k,i,j] * prod_l=0..k-1 (1 - alphas[b,l,i,j])

    Args:
        features: Packed Tensor of shape (C, P) giving the features of each point.
        alphas: float32 Tensor of shape (N, points_per_pixel, image_size,
            image_size) giving the weight of each point in the z-buffer.
            Values should be in the interval [0, 1].
        pointsidx: int32 Tensor of shape (N, points_per_pixel, image_size, image_size)
            giving the indices of the nearest points at each pixel, sorted in z-order.
            Concretely pointsidx[n, k, y, x] = p means that features[:, p] is the
            feature of the kth closest point (along the z-direction) to pixel (y, x) in
            batch element n. This is weighted by alphas[n, k, y, x].

    Returns:
        weighted_fs: Tensor of shape (N, C, image_size, image_size)
            giving the accumulated features at each point.
    """

    @staticmethod
    def forward(ctx, features, alphas, points_idx):
        pt_cld = _C.accum_alphacomposite(features, alphas, points_idx)
        ctx.save_for_backward(features.clone(), alphas.clone(), points_idx.clone())
        return pt_cld

    @staticmethod
    def backward(ctx, grad_output):
        grad_features = None
        grad_alphas = None
        grad_points_idx = None
        features, alphas, points_idx = ctx.saved_tensors
        grad_features, grad_alphas = _C.accum_alphacomposite_backward(grad_output, features, alphas, points_idx)
        return grad_features, grad_alphas, grad_points_idx, None


def alpha_composite(pointsidx, alphas, pt_clds, blend_params=None) ->torch.Tensor:
    """
    Composite features within a z-buffer using alpha compositing. Given a z-buffer
    with corresponding features and weights, these values are accumulated according
    to their weights such that features nearer in depth contribute more to the final
    feature than ones further away.

    Concretely this means:
        weighted_fs[b,c,i,j] = sum_k cum_alpha_k * features[c,pointsidx[b,k,i,j]]
        cum_alpha_k = alphas[b,k,i,j] * prod_l=0..k-1 (1 - alphas[b,l,i,j])


    Args:
        pt_clds: Tensor of shape (N, C, P) giving the features of each point (can use
            RGB for example).
        alphas: float32 Tensor of shape (N, points_per_pixel, image_size,
            image_size) giving the weight of each point in the z-buffer.
            Values should be in the interval [0, 1].
        pointsidx: int32 Tensor of shape (N, points_per_pixel, image_size, image_size)
            giving the indices of the nearest points at each pixel, sorted in z-order.
            Concretely pointsidx[n, k, y, x] = p means that features[n, :, p] is the
            feature of the kth closest point (along the z-direction) to pixel (y, x) in
            batch element n. This is weighted by alphas[n, k, y, x].

    Returns:
        Combined features: Tensor of shape (N, C, image_size, image_size)
            giving the accumulated features at each point.
    """
    return _CompositeAlphaPoints.apply(pt_clds, alphas, pointsidx)


class AlphaCompositor(nn.Module):
    """
    Accumulate points using alpha compositing.
    """

    def __init__(self, composite_params=None):
        super().__init__()
        self.composite_params = composite_params if composite_params is not None else CompositeParams()

    def forward(self, fragments, alphas, ptclds, **kwargs) ->torch.Tensor:
        images = alpha_composite(fragments, alphas, ptclds, self.composite_params)
        return images


class _CompositeNormWeightedSumPoints(torch.autograd.Function):
    """
    Composite features within a z-buffer using normalized weighted sum. Given a z-buffer
    with corresponding features and weights, these values are accumulated
    according to their weights such that depth is ignored; the weights are used to
    perform a weighted sum.

    Concretely this means:
        weighted_fs[b,c,i,j] =
         sum_k alphas[b,k,i,j] * features[c,pointsidx[b,k,i,j]] / sum_k alphas[b,k,i,j]

    Args:
        features: Packed Tensor of shape (C, P) giving the features of each point.
        alphas: float32 Tensor of shape (N, points_per_pixel, image_size,
            image_size) giving the weight of each point in the z-buffer.
            Values should be in the interval [0, 1].
        pointsidx: int32 Tensor of shape (N, points_per_pixel, image_size, image_size)
            giving the indices of the nearest points at each pixel, sorted in z-order.
            Concretely pointsidx[n, k, y, x] = p means that features[:, p] is the
            feature of the kth closest point (along the z-direction) to pixel (y, x) in
            batch element n. This is weighted by alphas[n, k, y, x].

    Returns:
        weighted_fs: Tensor of shape (N, C, image_size, image_size)
            giving the accumulated features at each point.
    """

    @staticmethod
    def forward(ctx, features, alphas, points_idx):
        pt_cld = _C.accum_weightedsumnorm(features, alphas, points_idx)
        ctx.save_for_backward(features.clone(), alphas.clone(), points_idx.clone())
        return pt_cld

    @staticmethod
    def backward(ctx, grad_output):
        grad_features = None
        grad_alphas = None
        grad_points_idx = None
        features, alphas, points_idx = ctx.saved_tensors
        grad_features, grad_alphas = _C.accum_weightedsumnorm_backward(grad_output, features, alphas, points_idx)
        return grad_features, grad_alphas, grad_points_idx, None


def norm_weighted_sum(pointsidx, alphas, pt_clds, blend_params=None) ->torch.Tensor:
    """
    Composite features within a z-buffer using normalized weighted sum. Given a z-buffer
    with corresponding features and weights, these values are accumulated
    according to their weights such that depth is ignored; the weights are used to
    perform a weighted sum.

    Concretely this means:
        weighted_fs[b,c,i,j] =
         sum_k alphas[b,k,i,j] * features[c,pointsidx[b,k,i,j]] / sum_k alphas[b,k,i,j]

    Args:
        pt_clds: Packed feature tensor of shape (C, P) giving the features of each point
            (can use RGB for example).
        alphas: float32 Tensor of shape (N, points_per_pixel, image_size,
            image_size) giving the weight of each point in the z-buffer.
            Values should be in the interval [0, 1].
        pointsidx: int32 Tensor of shape (N, points_per_pixel, image_size, image_size)
            giving the indices of the nearest points at each pixel, sorted in z-order.
            Concretely pointsidx[n, k, y, x] = p means that features[:, p] is the
            feature of the kth closest point (along the z-direction) to pixel (y, x) in
            batch element n. This is weighted by alphas[n, k, y, x].

    Returns:
        Combined features: Tensor of shape (N, C, image_size, image_size)
            giving the accumulated features at each point.
    """
    return _CompositeNormWeightedSumPoints.apply(pt_clds, alphas, pointsidx)


class NormWeightedCompositor(nn.Module):
    """
    Accumulate points using a normalized weighted sum.
    """

    def __init__(self, composite_params=None):
        super().__init__()
        self.composite_params = composite_params if composite_params is not None else CompositeParams()

    def forward(self, fragments, alphas, ptclds, **kwargs) ->torch.Tensor:
        images = norm_weighted_sum(fragments, alphas, ptclds, self.composite_params)
        return images


class PointFragments(NamedTuple):
    idx: torch.Tensor
    zbuf: torch.Tensor
    dists: torch.Tensor


class PointsRasterizationSettings:
    __slots__ = ['image_size', 'radius', 'points_per_pixel', 'bin_size', 'max_points_per_bin']

    def __init__(self, image_size: int=256, radius: float=0.01, points_per_pixel: int=8, bin_size: Optional[int]=None, max_points_per_bin: Optional[int]=None):
        self.image_size = image_size
        self.radius = radius
        self.points_per_pixel = points_per_pixel
        self.bin_size = bin_size
        self.max_points_per_bin = max_points_per_bin


def _axis_angle_rotation(axis: str, angle):
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)
    if axis == 'X':
        R_flat = one, zero, zero, zero, cos, -sin, zero, sin, cos
    if axis == 'Y':
        R_flat = cos, zero, sin, zero, one, zero, -sin, zero, cos
    if axis == 'Z':
        R_flat = cos, -sin, zero, sin, cos, zero, zero, zero, one
    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def _handle_coord(c, dtype, device):
    """
    Helper function for _handle_input.

    Args:
        c: Python scalar, torch scalar, or 1D torch tensor

    Returns:
        c_vec: 1D torch tensor
    """
    if not torch.is_tensor(c):
        c = torch.tensor(c, dtype=dtype, device=device)
    if c.dim() == 0:
        c = c.view(1)
    return c


def _handle_angle_input(x, dtype, device: str, name: str):
    """
    Helper function for building a rotation function using angles.
    The output is always of shape (N,).

    The input can be one of:
        - Torch tensor of shape (N,)
        - Python scalar
        - Torch scalar
    """
    if torch.is_tensor(x) and x.dim() > 1:
        msg = 'Expected tensor of shape (N,); got %r (in %s)'
        raise ValueError(msg % (x.shape, name))
    else:
        return _handle_coord(x, dtype, device)


def _handle_input(x, y, z, dtype, device, name: str, allow_singleton: bool=False):
    """
    Helper function to handle parsing logic for building transforms. The output
    is always a tensor of shape (N, 3), but there are several types of allowed
    input.

    Case I: Single Matrix
        In this case x is a tensor of shape (N, 3), and y and z are None. Here just
        return x.

    Case II: Vectors and Scalars
        In this case each of x, y, and z can be one of the following
            - Python scalar
            - Torch scalar
            - Torch tensor of shape (N, 1) or (1, 1)
        In this case x, y and z are broadcast to tensors of shape (N, 1)
        and concatenated to a tensor of shape (N, 3)

    Case III: Singleton (only if allow_singleton=True)
        In this case y and z are None, and x can be one of the following:
            - Python scalar
            - Torch scalar
            - Torch tensor of shape (N, 1) or (1, 1)
        Here x will be duplicated 3 times, and we return a tensor of shape (N, 3)

    Returns:
        xyz: Tensor of shape (N, 3)
    """
    if torch.is_tensor(x) and x.dim() == 2:
        if x.shape[1] != 3:
            msg = 'Expected tensor of shape (N, 3); got %r (in %s)'
            raise ValueError(msg % (x.shape, name))
        if y is not None or z is not None:
            msg = 'Expected y and z to be None (in %s)' % name
            raise ValueError(msg)
        return x
    if allow_singleton and y is None and z is None:
        y = x
        z = x
    xyz = [_handle_coord(c, dtype, device) for c in [x, y, z]]
    sizes = [c.shape[0] for c in xyz]
    N = max(sizes)
    for c in xyz:
        if c.shape[0] != 1 and c.shape[0] != N:
            msg = 'Got non-broadcastable sizes %r (in %s)' % (sizes, name)
            raise ValueError(msg)
    xyz = [c.expand(N) for c in xyz]
    xyz = torch.stack(xyz, dim=1)
    return xyz


def _broadcast_bmm(a, b):
    """
    Batch multiply two matrices and broadcast if necessary.

    Args:
        a: torch tensor of shape (P, K) or (M, P, K)
        b: torch tensor of shape (N, K, K)

    Returns:
        a and b broadcast multipled. The output batch dimension is max(N, M).

    To broadcast transforms across a batch dimension if M != N then
    expect that either M = 1 or N = 1. The tensor with batch dimension 1 is
    expanded to have shape N or M.
    """
    if a.dim() == 2:
        a = a[None]
    if len(a) != len(b):
        if not (len(a) == 1 or len(b) == 1):
            msg = 'Expected batch dim for bmm to be equal or 1; got %r, %r'
            raise ValueError(msg % (a.shape, b.shape))
        if len(a) == 1:
            a = a.expand(len(b), -1, -1)
        if len(b) == 1:
            b = b.expand(len(a), -1, -1)
    return a.bmm(b)


class Transform3d:
    """
    A Transform3d object encapsulates a batch of N 3D transformations, and knows
    how to transform points and normal vectors. Suppose that t is a Transform3d;
    then we can do the following:

    .. code-block:: python

        N = len(t)
        points = torch.randn(N, P, 3)
        normals = torch.randn(N, P, 3)
        points_transformed = t.transform_points(points)    # => (N, P, 3)
        normals_transformed = t.transform_points(normals)  # => (N, P, 3)


    BROADCASTING
    Transform3d objects supports broadcasting. Suppose that t1 and tN are
    Transform3D objects with len(t1) == 1 and len(tN) == N respectively. Then we
    can broadcast transforms like this:

    .. code-block:: python

        t1.transform_points(torch.randn(P, 3))     # => (P, 3)
        t1.transform_points(torch.randn(1, P, 3))  # => (1, P, 3)
        t1.transform_points(torch.randn(M, P, 3))  # => (M, P, 3)
        tN.transform_points(torch.randn(P, 3))     # => (N, P, 3)
        tN.transform_points(torch.randn(1, P, 3))  # => (N, P, 3)


    COMBINING TRANSFORMS
    Transform3d objects can be combined in two ways: composing and stacking.
    Composing is function composition. Given Transform3d objects t1, t2, t3,
    the following all compute the same thing:

    .. code-block:: python

        y1 = t3.transform_points(t2.transform_points(t2.transform_points(x)))
        y2 = t1.compose(t2).compose(t3).transform_points()
        y3 = t1.compose(t2, t3).transform_points()


    Composing transforms should broadcast.

    .. code-block:: python

        if len(t1) == 1 and len(t2) == N, then len(t1.compose(t2)) == N.

    We can also stack a sequence of Transform3d objects, which represents
    composition along the batch dimension; then the following should compute the
    same thing.

    .. code-block:: python

        N, M = len(tN), len(tM)
        xN = torch.randn(N, P, 3)
        xM = torch.randn(M, P, 3)
        y1 = torch.cat([tN.transform_points(xN), tM.transform_points(xM)], dim=0)
        y2 = tN.stack(tM).transform_points(torch.cat([xN, xM], dim=0))

    BUILDING TRANSFORMS
    We provide convenience methods for easily building Transform3d objects
    as compositions of basic transforms.

    .. code-block:: python

        # Scale by 0.5, then translate by (1, 2, 3)
        t1 = Transform3d().scale(0.5).translate(1, 2, 3)

        # Scale each axis by a different amount, then translate, then scale
        t2 = Transform3d().scale(1, 3, 3).translate(2, 3, 1).scale(2.0)

        t3 = t1.compose(t2)
        tN = t1.stack(t3, t3)


    BACKPROP THROUGH TRANSFORMS
    When building transforms, we can also parameterize them by Torch tensors;
    in this case we can backprop through the construction and application of
    Transform objects, so they could be learned via gradient descent or
    predicted by a neural network.

    .. code-block:: python

        s1_params = torch.randn(N, requires_grad=True)
        t_params = torch.randn(N, 3, requires_grad=True)
        s2_params = torch.randn(N, 3, requires_grad=True)

        t = Transform3d().scale(s1_params).translate(t_params).scale(s2_params)
        x = torch.randn(N, 3)
        y = t.transform_points(x)
        loss = compute_loss(y)
        loss.backward()

        with torch.no_grad():
            s1_params -= lr * s1_params.grad
            t_params -= lr * t_params.grad
            s2_params -= lr * s2_params.grad

    CONVENTIONS
    We adopt a right-hand coordinate system, meaning that rotation about an axis
    with a positive angle results in a counter clockwise rotation.

    This class assumes that transformations are applied on inputs which
    are row vectors. The internal representation of the Nx4x4 transformation
    matrix is of the form:

    .. code-block:: python

        M = [
                [Rxx, Ryx, Rzx, 0],
                [Rxy, Ryy, Rzy, 0],
                [Rxz, Ryz, Rzz, 0],
                [Tx,  Ty,  Tz,  1],
            ]

    To apply the transformation to points which are row vectors, the M matrix
    can be pre multiplied by the points:

    .. code-block:: python

        points = [[0, 1, 2]]  # (1 x 3) xyz coordinates of a point
        transformed_points = points * M

    """

    def __init__(self, dtype: torch.dtype=torch.float32, device='cpu', matrix: Optional[torch.Tensor]=None):
        """
        Args:
            dtype: The data type of the transformation matrix.
                to be used if `matrix = None`.
            device: The device for storing the implemented transformation.
                If `matrix != None`, uses the device of input `matrix`.
            matrix: A tensor of shape (4, 4) or of shape (minibatch, 4, 4)
                representing the 4x4 3D transformation matrix.
                If `None`, initializes with identity using
                the specified `device` and `dtype`.
        """
        if matrix is None:
            self._matrix = torch.eye(4, dtype=dtype, device=device).view(1, 4, 4)
        else:
            if matrix.ndim not in (2, 3):
                raise ValueError('"matrix" has to be a 2- or a 3-dimensional tensor.')
            if matrix.shape[-2] != 4 or matrix.shape[-1] != 4:
                raise ValueError('"matrix" has to be a tensor of shape (minibatch, 4, 4)')
            device = matrix.device
            self._matrix = matrix.view(-1, 4, 4)
        self._transforms = []
        self._lu = None
        self.device = device

    def __len__(self):
        return self.get_matrix().shape[0]

    def compose(self, *others):
        """
        Return a new Transform3d with the tranforms to compose stored as
        an internal list.

        Args:
            *others: Any number of Transform3d objects

        Returns:
            A new Transform3d with the stored transforms
        """
        out = Transform3d(device=self.device)
        out._matrix = self._matrix.clone()
        for other in others:
            if not isinstance(other, Transform3d):
                msg = 'Only possible to compose Transform3d objects; got %s'
                raise ValueError(msg % type(other))
        out._transforms = self._transforms + list(others)
        return out

    def get_matrix(self):
        """
        Return a matrix which is the result of composing this transform
        with others stored in self.transforms. Where necessary transforms
        are broadcast against each other.
        For example, if self.transforms contains transforms t1, t2, and t3, and
        given a set of points x, the following should be true:

        .. code-block:: python

            y1 = t1.compose(t2, t3).transform(x)
            y2 = t3.transform(t2.transform(t1.transform(x)))
            y1.get_matrix() == y2.get_matrix()

        Returns:
            A transformation matrix representing the composed inputs.
        """
        composed_matrix = self._matrix.clone()
        if len(self._transforms) > 0:
            for other in self._transforms:
                other_matrix = other.get_matrix()
                composed_matrix = _broadcast_bmm(composed_matrix, other_matrix)
        return composed_matrix

    def _get_matrix_inverse(self):
        """
        Return the inverse of self._matrix.
        """
        return torch.inverse(self._matrix)

    def inverse(self, invert_composed: bool=False):
        """
        Returns a new Transform3D object that represents an inverse of the
        current transformation.

        Args:
            invert_composed:
                - True: First compose the list of stored transformations
                  and then apply inverse to the result. This is
                  potentially slower for classes of transformations
                  with inverses that can be computed efficiently
                  (e.g. rotations and translations).
                - False: Invert the individual stored transformations
                  independently without composing them.

        Returns:
            A new Transform3D object contaning the inverse of the original
            transformation.
        """
        tinv = Transform3d(device=self.device)
        if invert_composed:
            tinv._matrix = torch.inverse(self.get_matrix())
        else:
            i_matrix = self._get_matrix_inverse()
            if len(self._transforms) > 0:
                tinv._transforms = [t.inverse() for t in reversed(self._transforms)]
                last = Transform3d(device=self.device)
                last._matrix = i_matrix
                tinv._transforms.append(last)
            else:
                tinv._matrix = i_matrix
        return tinv

    def stack(self, *others):
        transforms = [self] + list(others)
        matrix = torch.cat([t._matrix for t in transforms], dim=0)
        out = Transform3d()
        out._matrix = matrix
        return out

    def transform_points(self, points, eps: Optional[float]=None):
        """
        Use this transform to transform a set of 3D points. Assumes row major
        ordering of the input points.

        Args:
            points: Tensor of shape (P, 3) or (N, P, 3)
            eps: If eps!=None, the argument is used to clamp the
                last coordinate before peforming the final division.
                The clamping corresponds to:
                last_coord := (last_coord.sign() + (last_coord==0)) *
                torch.clamp(last_coord.abs(), eps),
                i.e. the last coordinates that are exactly 0 will
                be clamped to +eps.

        Returns:
            points_out: points of shape (N, P, 3) or (P, 3) depending
            on the dimensions of the transform
        """
        points_batch = points.clone()
        if points_batch.dim() == 2:
            points_batch = points_batch[None]
        if points_batch.dim() != 3:
            msg = 'Expected points to have dim = 2 or dim = 3: got shape %r'
            raise ValueError(msg % repr(points.shape))
        N, P, _3 = points_batch.shape
        ones = torch.ones(N, P, 1, dtype=points.dtype, device=points.device)
        points_batch = torch.cat([points_batch, ones], dim=2)
        composed_matrix = self.get_matrix()
        points_out = _broadcast_bmm(points_batch, composed_matrix)
        denom = points_out[(...), 3:]
        if eps is not None:
            denom_sign = denom.sign() + (denom == 0.0).type_as(denom)
            denom = denom_sign * torch.clamp(denom.abs(), eps)
        points_out = points_out[(...), :3] / denom
        if points_out.shape[0] == 1 and points.dim() == 2:
            points_out = points_out.reshape(points.shape)
        return points_out

    def transform_normals(self, normals):
        """
        Use this transform to transform a set of normal vectors.

        Args:
            normals: Tensor of shape (P, 3) or (N, P, 3)

        Returns:
            normals_out: Tensor of shape (P, 3) or (N, P, 3) depending
            on the dimensions of the transform
        """
        if normals.dim() not in [2, 3]:
            msg = 'Expected normals to have dim = 2 or dim = 3: got shape %r'
            raise ValueError(msg % (normals.shape,))
        composed_matrix = self.get_matrix()
        mat = composed_matrix[:, :3, :3]
        normals_out = _broadcast_bmm(normals, mat.transpose(1, 2).inverse())
        if normals_out.shape[0] == 1 and normals.dim() == 2:
            normals_out = normals_out.reshape(normals.shape)
        return normals_out

    def translate(self, *args, **kwargs):
        return self.compose(Translate(*args, device=self.device, **kwargs))

    def scale(self, *args, **kwargs):
        return self.compose(Scale(*args, device=self.device, **kwargs))

    def rotate_axis_angle(self, *args, **kwargs):
        return self.compose(RotateAxisAngle(*args, device=self.device, **kwargs))

    def clone(self):
        """
        Deep copy of Transforms object. All internal tensors are cloned
        individually.

        Returns:
            new Transforms object.
        """
        other = Transform3d(device=self.device)
        if self._lu is not None:
            other._lu = [l.clone() for l in self._lu]
        other._matrix = self._matrix.clone()
        other._transforms = [t.clone() for t in self._transforms]
        return other

    def to(self, device, copy: bool=False, dtype=None):
        """
        Match functionality of torch.Tensor.to()
        If copy = True or the self Tensor is on a different device, the
        returned tensor is a copy of self with the desired torch.device.
        If copy = False and the self Tensor already has the correct torch.device,
        then self is returned.

        Args:
          device: Device id for the new tensor.
          copy: Boolean indicator whether or not to clone self. Default False.
          dtype: If not None, casts the internal tensor variables
              to a given torch.dtype.

        Returns:
          Transform3d object.
        """
        if not copy and self.device == device:
            return self
        other = self.clone()
        if self.device != device:
            other.device = device
            other._matrix = self._matrix
            for t in other._transforms:
                t
        return other

    def cpu(self):
        return self

    def cuda(self):
        return self


def _check_valid_rotation_matrix(R, tol: float=1e-07):
    """
    Determine if R is a valid rotation matrix by checking it satisfies the
    following conditions:

    ``RR^T = I and det(R) = 1``

    Args:
        R: an (N, 3, 3) matrix

    Returns:
        None

    Emits a warning if R is an invalid rotation matrix.
    """
    N = R.shape[0]
    eye = torch.eye(3, dtype=R.dtype, device=R.device)
    eye = eye.view(1, 3, 3).expand(N, -1, -1)
    orthogonal = torch.allclose(R.bmm(R.transpose(1, 2)), eye, atol=tol)
    det_R = torch.det(R)
    no_distortion = torch.allclose(det_R, torch.ones_like(det_R))
    if not (orthogonal and no_distortion):
        msg = 'R is not a valid rotation matrix'
        warnings.warn(msg)
    return


class Rotate(Transform3d):

    def __init__(self, R, dtype=torch.float32, device: str='cpu', orthogonal_tol: float=1e-05):
        """
        Create a new Transform3d representing 3D rotation using a rotation
        matrix as the input.

        Args:
            R: a tensor of shape (3, 3) or (N, 3, 3)
            orthogonal_tol: tolerance for the test of the orthogonality of R

        """
        super().__init__(device=device)
        if R.dim() == 2:
            R = R[None]
        if R.shape[-2:] != (3, 3):
            msg = 'R must have shape (3, 3) or (N, 3, 3); got %s'
            raise ValueError(msg % repr(R.shape))
        R = R.to(dtype=dtype)
        _check_valid_rotation_matrix(R, tol=orthogonal_tol)
        N = R.shape[0]
        mat = torch.eye(4, dtype=dtype, device=device)
        mat = mat.view(1, 4, 4).repeat(N, 1, 1)
        mat[:, :3, :3] = R
        self._matrix = mat

    def _get_matrix_inverse(self):
        """
        Return the inverse of self._matrix.
        """
        return self._matrix.permute(0, 2, 1).contiguous()


r = np.expand_dims(np.eye(3), axis=0)


t = np.expand_dims(np.zeros(3), axis=0)


def get_world_to_view_transform(R=r, T=t) ->Transform3d:
    """
    This function returns a Transform3d representing the transformation
    matrix to go from world space to view space by applying a rotation and
    a translation.

    PyTorch3D uses the same convention as Hartley & Zisserman.
    I.e., for camera extrinsic parameters R (rotation) and T (translation),
    we map a 3D point `X_world` in world coordinates to
    a point `X_cam` in camera coordinates with:
    `X_cam = X_world R + T`

    Args:
        R: (N, 3, 3) matrix representing the rotation.
        T: (N, 3) matrix representing the translation.

    Returns:
        a Transform3d object which represents the composed RT transformation.

    """
    if T.shape[0] != R.shape[0]:
        msg = 'Expected R, T to have the same batch dimension; got %r, %r'
        raise ValueError(msg % (R.shape[0], T.shape[0]))
    if T.dim() != 2 or T.shape[1:] != (3,):
        msg = 'Expected T to have shape (N, 3); got %r'
        raise ValueError(msg % repr(T.shape))
    if R.dim() != 3 or R.shape[1:] != (3, 3):
        msg = 'Expected R to have shape (N, 3, 3); got %r'
        raise ValueError(msg % repr(R.shape))
    T = Translate(T, device=T.device)
    R = Rotate(R, device=R.device)
    return R.compose(T)


class _RasterizePoints(torch.autograd.Function):

    @staticmethod
    def forward(ctx, points, cloud_to_packed_first_idx, num_points_per_cloud, image_size: int=256, radius: float=0.01, points_per_pixel: int=8, bin_size: int=0, max_points_per_bin: int=0):
        args = points, cloud_to_packed_first_idx, num_points_per_cloud, image_size, radius, points_per_pixel, bin_size, max_points_per_bin
        idx, zbuf, dists = _C.rasterize_points(*args)
        ctx.save_for_backward(points, idx)
        ctx.mark_non_differentiable(idx)
        return idx, zbuf, dists

    @staticmethod
    def backward(ctx, grad_idx, grad_zbuf, grad_dists):
        grad_points = None
        grad_cloud_to_packed_first_idx = None
        grad_num_points_per_cloud = None
        grad_image_size = None
        grad_radius = None
        grad_points_per_pixel = None
        grad_bin_size = None
        grad_max_points_per_bin = None
        points, idx = ctx.saved_tensors
        args = points, idx, grad_zbuf, grad_dists
        grad_points = _C.rasterize_points_backward(*args)
        grads = grad_points, grad_cloud_to_packed_first_idx, grad_num_points_per_cloud, grad_image_size, grad_radius, grad_points_per_pixel, grad_bin_size, grad_max_points_per_bin
        return grads


kMaxPointsPerBin = 22


def rasterize_points(pointclouds, image_size: int=256, radius: float=0.01, points_per_pixel: int=8, bin_size: Optional[int]=None, max_points_per_bin: Optional[int]=None):
    """
    Pointcloud rasterization

    Args:
        pointclouds: A Pointclouds object representing a batch of point clouds to be
            rasterized. This is a batch of N pointclouds, where each point cloud
            can have a different number of points; the coordinates of each point
            are (x, y, z). The coordinates are expected to
            be in normalized device coordinates (NDC): [-1, 1]^3 with the camera at
            (0, 0, 0); In the camera coordinate frame the x-axis goes from right-to-left,
            the y-axis goes from bottom-to-top, and the z-axis goes from back-to-front.
        image_size: Integer giving the resolution of the rasterized image
        radius (Optional): Float giving the radius (in NDC units) of the disk to
            be rasterized for each point.
        points_per_pixel (Optional): We will keep track of this many points per
            pixel, returning the nearest points_per_pixel points along the z-axis
        bin_size: Size of bins to use for coarse-to-fine rasterization. Setting
            bin_size=0 uses naive rasterization; setting bin_size=None attempts to
            set it heuristically based on the shape of the input. This should not
            affect the output, but can affect the speed of the forward pass.
        points_per_bin: Only applicable when using coarse-to-fine rasterization
            (bin_size > 0); this is the maxiumum number of points allowed within each
            bin. If more than this many points actually fall into a bin, an error
            will be raised. This should not affect the output values, but can affect
            the memory usage in the forward pass.

    Returns:
        3-element tuple containing

        - **idx**: int32 Tensor of shape (N, image_size, image_size, points_per_pixel)
          giving the indices of the nearest points at each pixel, in ascending
          z-order. Concretely `idx[n, y, x, k] = p` means that `points[p]` is the kth
          closest point (along the z-direction) to pixel (y, x) - note that points
          represents the packed points of shape (P, 3).
          Pixels that are hit by fewer than points_per_pixel are padded with -1.
        - **zbuf**: Tensor of shape (N, image_size, image_size, points_per_pixel)
          giving the z-coordinates of the nearest points at each pixel, sorted in
          z-order. Concretely, if `idx[n, y, x, k] = p` then
          `zbuf[n, y, x, k] = points[n, p, 2]`. Pixels hit by fewer than
          points_per_pixel are padded with -1
        - **dists2**: Tensor of shape (N, image_size, image_size, points_per_pixel)
          giving the squared Euclidean distance (in NDC units) in the x/y plane
          for each point closest to the pixel. Concretely if `idx[n, y, x, k] = p`
          then `dists[n, y, x, k]` is the squared distance between the pixel (y, x)
          and the point `(points[n, p, 0], points[n, p, 1])`. Pixels hit with fewer
          than points_per_pixel are padded with -1.
    """
    points_packed = pointclouds.points_packed()
    cloud_to_packed_first_idx = pointclouds.cloud_to_packed_first_idx()
    num_points_per_cloud = pointclouds.num_points_per_cloud()
    if bin_size is None:
        if not points_packed.is_cuda:
            bin_size = 0
        elif image_size <= 64:
            bin_size = 8
        elif image_size <= 256:
            bin_size = 16
        elif image_size <= 512:
            bin_size = 32
        elif image_size <= 1024:
            bin_size = 64
    if bin_size != 0:
        points_per_bin = 1 + (image_size - 1) // bin_size
        if points_per_bin >= kMaxPointsPerBin:
            raise ValueError('bin_size too small, number of points per bin must be less than %d; got %d' % (kMaxPointsPerBin, points_per_bin))
    if max_points_per_bin is None:
        max_points_per_bin = int(max(10000, points_packed.shape[0] / 5))
    return _RasterizePoints.apply(points_packed, cloud_to_packed_first_idx, num_points_per_cloud, image_size, radius, points_per_pixel, bin_size, max_points_per_bin)


class PointsRasterizer(nn.Module):
    """
    This class implements methods for rasterizing a batch of pointclouds.
    """

    def __init__(self, cameras=None, raster_settings=None):
        """
        cameras: A cameras object which has a  `transform_points` method
                which returns the transformed points after applying the
                world-to-view and view-to-screen
                transformations.
            raster_settings: the parameters for rasterization. This should be a
                named tuple.

        All these initial settings can be overridden by passing keyword
        arguments to the forward function.
        """
        super().__init__()
        if raster_settings is None:
            raster_settings = PointsRasterizationSettings()
        self.cameras = cameras
        self.raster_settings = raster_settings

    def transform(self, point_clouds, **kwargs) ->torch.Tensor:
        """
        Args:
            point_clouds: a set of point clouds

        Returns:
            points_screen: the points with the vertex positions in screen
            space

        NOTE: keeping this as a separate function for readability but it could
        be moved into forward.
        """
        cameras = kwargs.get('cameras', self.cameras)
        if cameras is None:
            msg = 'Cameras must be specified either at initialization                 or in the forward pass of PointsRasterizer'
            raise ValueError(msg)
        pts_world = point_clouds.points_padded()
        pts_world_packed = point_clouds.points_packed()
        pts_screen = cameras.transform_points(pts_world, **kwargs)
        view_transform = get_world_to_view_transform(R=cameras.R, T=cameras.T)
        verts_view = view_transform.transform_points(pts_world)
        pts_screen[..., 2] = verts_view[..., 2]
        pad_to_packed_idx = point_clouds.padded_to_packed_idx()
        pts_screen_packed = pts_screen.view(-1, 3)[(pad_to_packed_idx), :]
        pts_packed_offset = pts_screen_packed - pts_world_packed
        point_clouds = point_clouds.offset(pts_packed_offset)
        return point_clouds

    def forward(self, point_clouds, **kwargs) ->PointFragments:
        """
        Args:
            point_clouds: a set of point clouds with coordinates in world space.
        Returns:
            PointFragments: Rasterization outputs as a named tuple.
        """
        points_screen = self.transform(point_clouds, **kwargs)
        raster_settings = kwargs.get('raster_settings', self.raster_settings)
        idx, zbuf, dists2 = rasterize_points(points_screen, image_size=raster_settings.image_size, radius=raster_settings.radius, points_per_pixel=raster_settings.points_per_pixel, bin_size=raster_settings.bin_size, max_points_per_bin=raster_settings.max_points_per_bin)
        return PointFragments(idx=idx, zbuf=zbuf, dists=dists2)


class PointsRenderer(nn.Module):
    """
    A class for rendering a batch of points. The class should
    be initialized with a rasterizer and compositor class which each have a forward
    function.
    """

    def __init__(self, rasterizer, compositor):
        super().__init__()
        self.rasterizer = rasterizer
        self.compositor = compositor

    def forward(self, point_clouds, **kwargs) ->torch.Tensor:
        fragments = self.rasterizer(point_clouds, **kwargs)
        r = self.rasterizer.raster_settings.radius
        dists2 = fragments.dists.permute(0, 3, 1, 2)
        weights = 1 - dists2 / (r * r)
        images = self.compositor(fragments.idx.long().permute(0, 3, 1, 2), weights, point_clouds.features_packed().permute(1, 0), **kwargs)
        images = images.permute(0, 2, 3, 1)
        return images

