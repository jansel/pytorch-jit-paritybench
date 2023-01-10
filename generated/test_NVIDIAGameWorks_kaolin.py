import sys
_module = sys.modules[__name__]
del sys
conf = _module
kaolin_ext = _module
camera_coordinate_systems = _module
camera_init_explicit = _module
camera_init_simple = _module
camera_movement = _module
camera_opengl_shaders = _module
camera_properties = _module
camera_ray_tracing = _module
camera_transforms = _module
cameras_differentiable = _module
spc_from_pointcloud = _module
fast_mesh_sampling = _module
occupancy_sampling = _module
spc_basics = _module
spc_conv3d_example = _module
spc_dual_octree = _module
spc_trilinear_interp = _module
dmtet_network = _module
spc_formatting = _module
usd_kitchenset = _module
visualize_main = _module
kaolin = _module
dash3d = _module
run = _module
util = _module
io = _module
dataset = _module
materials = _module
modelnet = _module
obj = _module
off = _module
render = _module
shapenet = _module
shrec = _module
usd = _module
utils = _module
metrics = _module
pointcloud = _module
render = _module
tetmesh = _module
trianglemesh = _module
voxelgrid = _module
ops = _module
batch = _module
conversions = _module
pointcloud = _module
sdf = _module
tetmesh = _module
trianglemesh = _module
voxelgrid = _module
coords = _module
gcn = _module
mesh = _module
check_sign = _module
mesh = _module
tetmesh = _module
trianglemesh = _module
random = _module
reduction = _module
spc = _module
convolution = _module
points = _module
spc = _module
uint8 = _module
voxelgrid = _module
camera = _module
camera = _module
coordinates = _module
extrinsics = _module
extrinsics_backends = _module
intrinsics = _module
intrinsics_ortho = _module
intrinsics_pinhole = _module
legacy = _module
lighting = _module
sg = _module
sh = _module
deftet = _module
dibr = _module
rasterization = _module
utils = _module
raytrace = _module
rep = _module
spc = _module
testing = _module
visualize = _module
timelapse = _module
setup = _module
run_e2e_test = _module
test_client_server_io = _module
test_usd_kitchenset = _module
test_visualize_main = _module
test_dataset = _module
test_materials = _module
test_modelnet = _module
test_obj = _module
test_off = _module
test_render = _module
test_shapenet = _module
test_shrec = _module
test_usd = _module
test_pointcloud = _module
test_render = _module
test_tetmesh = _module
test_trianglemesh = _module
test_voxelgrid = _module
test_pointcloud = _module
test_sdf = _module
test_tetmesh = _module
test_trianglemesh = _module
test_voxelgrid = _module
test_check_sign = _module
test_mesh = _module
test_tetmesh = _module
test_conv = _module
test_points = _module
test_spc = _module
test_uint8 = _module
test_batch = _module
test_coords = _module
test_gcn = _module
test_random = _module
test_reduction = _module
test_voxelgrid = _module
test_extrinsics = _module
test_transform = _module
test_sg = _module
test_sh = _module
test_deftet = _module
test_dibr = _module
test_rasterization = _module
test_utils = _module
test_rayops = _module
test_raytrace = _module
test_camera = _module
test_rep_spc = _module
test_testing = _module
test_timelapse = _module
check_torchlibs_versions = _module
fixNvPe = _module

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


import torch


import numpy as np


from typing import Tuple


import torch.optim as optim


import logging


import random


import warnings


import copy


from collections.abc import Sequence


from abc import abstractmethod


from collections import namedtuple


from torch.multiprocessing import Pool


from torch.utils.data import Dataset


import inspect


from collections.abc import Callable


from typing import Callable


import itertools


import re


import torch.nn.functional as F


from torch import nn


from torch.autograd import Function


from itertools import product


from scipy import ndimage


import functools


from copy import deepcopy


from typing import Sequence


from typing import List


from typing import Dict


from typing import Union


from typing import Type


from typing import FrozenSet


from torch.types import _float


from torch.types import _bool


from typing import Iterable


from abc import ABC


from enum import IntEnum


from typing import Optional


import torch.nn


from numpy import tan


import torch.autograd


import collections


from torch._six import string_classes


import numpy


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CppExtension


from torch.utils.cpp_extension import CUDAExtension


from torch.utils.cpp_extension import CUDA_HOME


import time


from itertools import combinations


from itertools import chain


import torchvision


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
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.0 ** 0.0, 2.0 ** max_freq, steps=N_freqs)
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
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


class Decoder(torch.nn.Module):

    def __init__(self, input_dims=3, internal_dims=128, output_dims=4, hidden=5, multires=2):
        super().__init__()
        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn
            input_dims = input_ch
        net = torch.nn.Linear(input_dims, internal_dims, bias=False), torch.nn.ReLU()
        for i in range(hidden - 1):
            net = net + (torch.nn.Linear(internal_dims, internal_dims, bias=False), torch.nn.ReLU())
        net = net + (torch.nn.Linear(internal_dims, output_dims, bias=False),)
        self.net = torch.nn.Sequential(*net)

    def forward(self, p):
        if self.embed_fn is not None:
            p = self.embed_fn(p)
        out = self.net(p)
        return out

    def pre_train_sphere(self, iter):
        None
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(list(self.parameters()), lr=0.0001)
        for i in tqdm(range(iter)):
            p = torch.rand((1024, 3), device='cuda') - 0.5
            ref_value = torch.sqrt((p ** 2).sum(-1)) - 0.3
            output = self(p)
            loss = loss_fn(output[..., 0], ref_value)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        None


def sparse_bmm(sparse_matrix, dense_matrix_batch):
    """Perform torch.bmm on an unbatched sparse matrix and a batched dense matrix.

    Args:
        sparse_matrix (torch.sparse.FloatTensor):
            Input sparse matrix, of shape :math:`(\\text{M}, \\text{N})`.
        dense_matrix_batch (torch.FloatTensor):
            Input batched dense matrix, of shape
            :math:`(\\text{batch_size}, \\text{N}, \\text{P})`.

    Returns:
        (torch.FloatTensor):
            Result of the batched matrix multiplication, of shape,
            :math:`(\\text{batch_size}, \\text{N}, \\text{P})`.
    """
    m = sparse_matrix.shape[0]
    b, n, p = dense_matrix_batch.shape
    dense_matrix = dense_matrix_batch.transpose(0, 1).reshape(n, b * p)
    result = torch.sparse.mm(sparse_matrix, dense_matrix)
    return result.reshape(m, b, p).transpose(0, 1)


class GraphConv(nn.Module):
    """A simple graph convolution layer, similar to the one defined by *Kipf et al.* in
    `Semi-Supervised Classification with Graph Convolutional Networks`_ ICLR 2017

    This operation with ``self_layer=False`` is equivalent to
    :math:`(A H W)` where:

    - :math:`H` is the node features, of shape
      :math:`(\\text{batch_size}, \\text{num_nodes}, \\text{input_dim})`.

    - :math:`W` is a weight matrix, of shape
      :math:`(\\text{input_dim}, \\text{output_dim})`.

    - :math:`A` is the adjacency matrix, of shape
      :math:`(\\text{num_nodes}, \\text{num_nodes})`.
      It can include self-loop.

    With ``normalize_adj=True``, it is equivalent to :math:`(D^{-1} A H W)`, where:

    - :math:`D` is a diagonal matrix with :math:`D_{ii}` = the sum of the i-th row of :math:`A`.
      In other words, :math:`D` is the incoming degree of each node.

    With ``self_layer=True``, it is equivalent to the above plus :math:`(H W_{\\text{self}})`, where:

    - :math:`W_{\\text{self}}` is a separate weight matrix to filter each node's self features.

    Note that when ``self_layer=True``, A should not include self-loop.

    Args:
        input_dim (int): The number of features in each input node.
        output_dim (int): The number of features in each output node.
        bias (bool): Whether to add bias after the node-wise linear layer.

    Example:
        >>> node_feat = torch.rand(1, 3, 5)
        >>> i = torch.LongTensor(
        ...     [[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]])
        >>> v = torch.FloatTensor([1, 1, 1, 1, 1, 1])
        >>> adj = torch.sparse.FloatTensor(i, v, torch.Size([3, 3]))
        >>> model = GraphConv(5, 10)
        >>> output = model(node_feat, adj)
        >>> # pre-normalize adj
        >>> adj = normalize_adj(adj)
        >>> output = model(node_feat, adj, normalize_adj=False)

    .. _Semi-Supervised Classification with Graph Convolutional Networks:
        https://arxiv.org/abs/1609.02907
    """

    def __init__(self, input_dim, output_dim, self_layer=True, bias=True):
        super(GraphConv, self).__init__()
        self.self_layer = self_layer
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)
        if self_layer:
            self.linear_self = nn.Linear(input_dim, output_dim, bias=bias)
        else:
            self.linear_self = None
        self.initialize()

    def initialize(self):
        nn.init.xavier_uniform_(self.linear.weight.data)
        if self.linear.bias is not None:
            self.linear.bias.data.uniform_(-1.0, 1.0)
        if self.self_layer:
            nn.init.xavier_uniform_(self.linear_self.weight.data)
            if self.linear_self.bias is not None:
                self.linear_self.bias.data.uniform_(-1.0, 1.0)

    def forward(self, node_feat, adj, normalize_adj=True):
        """
        Args:
            node_feat (torch.FloatTensor):
                The input features of each node, of shape
                :math:`(\\text{batch_size}, \\text{num_nodes}, \\text{input_dim})`.
            adj (torch.sparse.FloatTensor or torch.FloatTensor):
                The adjacency matrix. ``adj[i, j]`` is non-zero if there's an
                incoming edge from ``j`` to ``i``. Should not include self-loop if
                ``self_layer`` is ``True``, of shape
                :math:`(\\text{num_nodes}, \\text{num_nodes})`.
            normalize_adj (optional, bool):
                Set this to true to apply normalization to adjacency; that is,
                each output feature will be divided by the number of incoming
                neighbors. If normalization is not desired, or if the adjacency
                matrix is pre-normalized, set this to False to improve
                performance. Default: True.

        Returns:
            (torch.FloatTensor):
                The output features of each node, of shape
                :math:(\\text{batch_size}, \\text{num_nodes}, \\text{output_dim})`.
        """
        if adj.type().endswith('sparse.FloatTensor'):
            if normalize_adj:
                norm = torch.sparse.mm(adj, torch.ones((adj.shape[0], 1), device=node_feat.device))
                result = sparse_bmm(adj, self.linear(node_feat)) / norm
            else:
                result = sparse_bmm(adj, self.linear(node_feat))
        elif normalize_adj:
            norm = torch.matmul(adj, torch.ones((adj.shape[0], 1), device=node_feat.device))
            result = torch.matmul(adj, self.linear(node_feat)) / norm
        else:
            result = torch.matmul(adj, self.linear(node_feat))
        if self.self_layer:
            result += self.linear_self(node_feat)
        return result


def get_shape_per_tensor(tensor_list):
    """Returns the shape of each tensor in the tensor list except the last dimension.

    See shape_per_tensor for :ref:`packed<packed_shape_per_tensor>` or :ref:`padded<padded_shape_per_tensor>`
    for more information.

    Args:
        tensor_list (sequence of torch.Tensor): any python sequence of tensors of the identical type,
            number of dimensions, and last dimension size, e.g. :math:`[(H_0, W_0, C), (H_1, W_1, C)]`.

    Returns:
        (torch.Tensor):
            the shape of each subtensor (except for the last dim),
            of shape :math:`(len(\\text{tensor_list}), \\text{tensor_list[0].ndim} - 1)`.

    Examples:
        >>> tensor_list = [
        ...         torch.zeros((1, 3, 4, 2)),
        ...         torch.ones((2, 5, 3, 2))
        ... ]
        >>> get_shape_per_tensor(tensor_list)
        tensor([[1, 3, 4],
                [2, 5, 3]])
    """
    try:
        shape_per_tensor = torch.tensor([t.shape[:-1] for t in tensor_list], dtype=torch.long)
    except ValueError as err:
        ndim = tensor_list[0].ndim
        for i, t in enumerate(tensor_list):
            if t.ndim != ndim:
                raise ValueError(f'Expected all tensors to have {ndim} dimensions but got {t.ndim} at index {i}')
        raise err
    return shape_per_tensor


def list_to_packed(tensor_list):
    """Converts a sequence of torch.Tensor into a single :ref:`packed tensor<packed>`.

    torch.Tensor of same type, number of dimensions and last dimension size
    will be reshaped to :math:`(-1, \\text{last_dim})` and concatenated on first axis.
    E.g.:
    With input of shapes :math:`[(X_0, Y_0, Z_0, C), (X_1, Y_1, Z_1, C)]` the output packed tensor will be
    of shape :math:`((X_0 * Y_0 * Z_0 + X_1 * Y_1 * Z_1), C)`.
    The output shape_per_tensor will be the tensor: :math:`[[X_0, Y_0, Z_0], [X_1, Y_1, Z_1]]`.

    Args:
        tensor_list (sequence of torch.Tensor): any python sequence of tensors of identical type,
            number of dimensions, and last dimension size, e.g. :math:`[(H_0, W_0, C), (H_1, W_1, C)]`.

    Returns:
        (torch.Tensor, torch.LongTensor):
            the :ref:`packed tensor<packed>` and the associated :ref:`shape_per_tensor<padded_shape_per_tensor>`

    Example:
        >>> a = torch.LongTensor([[0, 1, 2],
        ...                       [1, 2, 3]])
        >>> b = torch.LongTensor([[2, 4, 5]])
        >>> packed_tensor, shape_per_tensor = list_to_packed([a, b])
        >>> packed_tensor
        tensor([[0, 1, 2],
                [1, 2, 3],
                [2, 4, 5]])
        >>> shape_per_tensor
        tensor([[2],
                [1]])
    """
    shape_per_tensor = get_shape_per_tensor(tensor_list)
    try:
        output = torch.cat([t.reshape(-1, t.shape[-1]) for t in tensor_list], dim=0)
    except RuntimeError as err:
        last_dim = tensor_list[0].shape[-1]
        t_type = tensor_list[0].type()
        for i, t in enumerate(tensor_list):
            if t.shape[-1] != last_dim:
                raise ValueError(f'Expected all tensor to have last dimension {last_dim} but got {t.shape[-1]} at index {i}')
            if t.type() != t_type:
                raise ValueError(f'Expected all tensor to have type {t_type} but got {t.type()} at index {i}')
        raise err
    return output, shape_per_tensor


class Spc(object):
    """Data class holding all :ref:`Structured Point Cloud (SPC)<spc>` information.

    This class supports batching through :ref:`packed<packed>` representation:
    a single Spc object can pack multiple SPC structures of variable sizes.

    SPC data structures are represented through the combination various tensors detailed below:

    ``octrees`` compress the information required to build a full SPC.
    In practice, they are a low level structure which also constitute the
    :ref:`core part<spc_octree>` of the SPC data structure.

    ``octrees`` are kept as a torch.ByteTensor, where each byte represents a single octree parent cell,
    and each bit represents the occupancy of a child octree cell.
    e.g: 8 bits for 8 cells.

    Bits describe the octree cells in Morton Order::

         . . . . . . . .
         | .   3  .  7  | .                    3   7
         |   . . . . . . . .           ===>    1   5
         |   | .   1  . | 5   .
         |   |   . . . . . . . .
         |   |    |     |       |              2   6
          . .|. . | . . .       |      ===>    0   4
            .| 2  |.  6   .     |
              . . | . . . . .   |
                . | 0  .  4   . |
                  . . . . . . . .

    If a cell is occupied, an additional cell byte may be generated in the next level,
    up till the argument ``level``.

    For example, a ``SPC.octrees`` field may, look as follows::

            tensor([255, 128,  64,  32,  16,   8,   4,   2,  23], dtype=torch.uint8)

    Here "octrees" represents an octree of 9 nodes.
    The binary representation should be interpreted as follows::

            Level #1, Path*,      11111111    (All cells are occupied, therefore 8 bytes are allocated for level 2)
            Level #2, Path*-1,    10000000
            Level #2, Path*-2,    01000000
            Level #2, Path*-3,    00100000
            Level #2, Path*-4,    00010000
            Level #2, Path*-5,    00001000
            Level #2, Path*-6,    00000100
            Level #2, Path*-7,    00000010
            Level #2, Path*-8,    00010111

    ``lengths`` is a tensor of integers required to support batching. Since we assume a packed representation,
    all octree cells are shaped as a single stacked 1D tensor. ``lengths`` specifies the number of cells (bytes) each
    octree uses.

    ``features`` represent an optional per-point feature vector.
    When ``features`` is not ``None``, a feature is kept for each point at the highest-resolution level in the octree.

    ``max_level`` is an integer which specifies how many recursive levels an octree should have.

    ``point_hierarchies``, ``pyramid``, ``exsum`` are auxilary structures, which are generated upon request and
    enable efficient indexing to SPC entries.
    """
    KEYS = {'octrees', 'lengths', 'max_level', 'pyramids', 'exsum', 'point_hierarchies'}

    def __init__(self, octrees, lengths, max_level=None, pyramids=None, exsum=None, point_hierarchies=None, features=None):
        assert isinstance(octrees, torch.Tensor) and octrees.dtype == torch.uint8 and octrees.ndim == 1, 'octrees must be a 1D ByteTensor.'
        assert isinstance(lengths, torch.Tensor) and lengths.dtype == torch.int and lengths.ndim == 1, 'lengths must be a 1D IntTensor.'
        assert max_level is None or isinstance(max_level, int), 'max_level must an int.'
        if pyramids is not None:
            assert isinstance(pyramids, torch.Tensor) and pyramids.dtype == torch.int, 'pyramids must be an IntTensor.'
            assert pyramids.ndim == 3 and pyramids.shape[0] == lengths.shape[0] and pyramids.shape[1] == 2 and (max_level is None or pyramids.shape[2] == max_level + 2), 'pyramids must be of shape (batch_size, 2, max_level + 2).'
        if exsum is not None:
            assert isinstance(exsum, torch.Tensor) and exsum.dtype == torch.int, 'exsum must be an IntTensor.'
            assert exsum.ndim == 1 and exsum.shape[0] == octrees.shape[0] + lengths.shape[0], 'exsum must be of shape (num_bytes + batch_size).'
            assert exsum.device == octrees.device, 'exsum must be on the same device than octrees.'
        if point_hierarchies is not None:
            assert isinstance(point_hierarchies, torch.Tensor) and point_hierarchies.dtype == torch.short, 'point_hierarchies must be a ShortTensor.'
            assert point_hierarchies.ndim == 2 and point_hierarchies.shape[1] == 3, 'point_hierarchies must be of shape (num_nodes, 3).'
            assert point_hierarchies.device == octrees.device, 'point_hierarchies must be on the same device than octrees.'
        if features is not None:
            assert isinstance(features, torch.Tensor), 'features must be a torch.Tensor'
            assert features.device == octrees.device, 'features must be on the same device as octrees.'
        self.octrees = octrees
        self.lengths = lengths
        self._max_level = max_level
        self._pyramids = pyramids
        self._exsum = exsum
        self._point_hierarchies = point_hierarchies
        self.features = features

    @classmethod
    def make_dense(cls, level, device='cuda'):
        """Creates a dense, fully occupied Spc object.
        The Spc will have ``level`` levels of detail.

        Args:
            level (int):
                Number of levels to use for the dense Spc.
            device (torch.device):
                Torch device to keep the spc octree

        Return:
            (kaolin.rep.Spc): a new fully occupied ``Spc``.
        """
        octree, lengths = create_dense_spc(level, device)
        return Spc(octrees=octree, lengths=lengths)

    @classmethod
    def from_features(cls, feature_grids, masks=None):
        """Creates a sparse Spc object from the feature grid.

        Args:
            feature_grids (torch.Tensor):
                The sparse 3D feature grids, of shape
                :math:`(	ext{batch_size}, 	ext{feature_dim}, X, Y, Z)`
            masks (optional, torch.BoolTensor):
                The topology mask, showing where are the features,
                of shape :math:`(	ext{batch_size}, X, Y, Z)`.
                Default: A feature is determined when not full of zeros.

        Returns:
            (torch.ByteTensor, torch.IntTensor, torch.Tensor):
                a tuple containing:

                    - The octree, of size :math:`(	ext{num_nodes})`

                    - The lengths of each octree, of size :math:`(	ext{batch_size})`

                    - The coalescent features, of same dtype than ``feature_grids``,
                      of shape :math:`(	ext{num_features}, 	ext{feature_dim})`.
        Return:
            (kaolin.rep.Spc): a ``Spc``, with length of :math:`(	ext{batch_size})`,
            an octree of size octree, of size :math:`(	ext{num_nodes})`, and the features field
            of the same dtype as ``feature_grids`` and of shape :math:`(	ext{num_features}, 	ext{feature_dim})`.
        """
        octrees, lengths, coalescent_features = feature_grids_to_spc(feature_grids, masks=masks)
        return Spc(octrees=octrees, lengths=lengths, features=coalescent_features)

    def _apply_scan_octrees(self):
        max_level, pyramids, exsum = scan_octrees(self.octrees, self.lengths)
        self._max_level = max_level
        self._pyramids = pyramids
        self._exsum = exsum

    def _apply_generate_points(self):
        self._point_hierarchies = generate_points(self.octrees, self.pyramids, self.exsum)

    @property
    def max_level(self):
        if self._max_level is None:
            self._apply_scan_octrees()
        return self._max_level

    @property
    def pyramids(self):
        if self._pyramids is None:
            self._apply_scan_octrees()
        return self._pyramids

    @property
    def exsum(self):
        if self._exsum is None:
            self._apply_scan_octrees()
        return self._exsum

    @property
    def point_hierarchies(self):
        if self._point_hierarchies is None:
            self._apply_generate_points()
        return self._point_hierarchies

    @classmethod
    def from_list(cls, octrees_list):
        """Generate an Spc from a list of octrees.

        Args:
            octrees_list (list of torch.ByteTensor):
                list containing multiple 1D torch.ByteTensor,
                each representing an octree.

        Return:
            (kaolin.rep.Spc): a new ``Spc``.
        """
        octrees, lengths = list_to_packed([octree.reshape(-1, 1) for octree in octrees_list])
        return cls(octrees.reshape(-1).contiguous(), lengths.reshape(-1).int())

    def to(self, device, non_blocking=False, memory_format=torch.preserve_format):
        _octrees = self.octrees
        if _octrees.data_ptr() == self.octrees.data_ptr():
            return self
        else:
            if self._exsum is not None:
                _exsum = self._exsum
            else:
                _exsum = None
            if self._point_hierarchies is not None:
                _point_hierarchies = self.point_hierarchies
            else:
                _point_hierarchies = None
            return Spc(_octrees, self.lengths, self._max_level, self._pyramids, _exsum, _point_hierarchies)

    def cuda(self, device='cuda', non_blocking=False, memory_format=torch.preserve_format):
        return self

    def cpu(self, memory_format=torch.preserve_format):
        return self

    @property
    def batch_size(self):
        return self.lengths.shape[0]

    def to_dict(self, keys=None):
        if keys is None:
            return {k: getattr(self, k) for k in self.KEYS}
        else:
            return {k: getattr(self, k) for k in keys}

    def num_points(self, lod: int):
        """
        Returns how many points the SPC holds at a given level of detail.

        Args:
            lod (int):
                Index of a level of detail.
                Level 0 is considered the root and always holds a single point,
                level 1 holds up to :math:`(	ext{num_points}=8)` points,
                level 2 holds up to :math:`(	ext{num_points}=8^{2})`, and so forth.
        Return:
            (torch.Tensor): The number of points each SPC entry holds for the given level of detail.
        """
        return self.pyramids[:, 0, lod]


class Conv3dFunction(Function):

    @staticmethod
    def forward(ctx, octrees, point_hierarchies, level, pyramids, exsum, inputs, params, kernel_vectors, jump):
        octrees = octrees.contiguous()
        point_hierarchies = point_hierarchies.contiguous()
        pyramids = pyramids.contiguous()
        exsum = exsum.contiguous()
        inputs = inputs.contiguous()
        params = params.contiguous()
        kernel_vectors = kernel_vectors.contiguous()
        ctx.save_for_backward(octrees, point_hierarchies, pyramids, exsum, inputs, params, kernel_vectors)
        ctx.jump = jump
        outputs, level = _C.ops.spc.Conv3d_forward(octrees, point_hierarchies, level, pyramids, exsum, inputs, params, kernel_vectors, jump)
        ctx.level = level
        level = torch.tensor([level])
        ctx.mark_non_differentiable(level)
        return outputs, level

    @staticmethod
    def backward(ctx, grad_outputs, grad_level):
        grad_outputs = grad_outputs.contiguous()
        octrees, point_hierarchies, pyramids, exsum, inputs, params, kernel_vectors = ctx.saved_tensors
        d_inputs, d_params = _C.ops.spc.Conv3d_backward(octrees, point_hierarchies, ctx.level, pyramids, exsum, inputs, grad_outputs, params, kernel_vectors, ctx.jump)
        return None, None, None, None, None, d_inputs, d_params, None, None


def conv3d(octrees, point_hierarchies, level, pyramids, exsum, input, weight, kernel_vectors, jump=0, bias=None, **kwargs):
    """Convolution over a structured point cloud. The inputs :math:`X` are mapped
    to outputs :math:`Y` by the following:

    .. math::

        Y_i = \\sum_k w_k \\cdot X_{n(i,k)} + b \\quad\\text{for}\\; i \\in 0,\\ldots,|Y|-1,

    where :math:`w_k` are weights associated with the kernel, and :math:`n(i,k)` is the
    neighborhood function described :ref:`here <neighborhood-text>`.

    Args:
        octrees (torch.ByteTensor):
            :ref:`packed` octrees of shape :math:`(\\text{num_bytes})`.
            See :ref:`octree <spc_octree>`.
        point_hierarchies (torch.ShortTensor):
            :ref:`packed` point hierarchies of shape :math:`(\\text{num_points})`.
            See :ref:`point_hierarchies <spc_points>`.
        level (int):
            level at which the ``input`` features are associated to.
        pyramids (torch.IntTensor):
            Batched tensor containing point hierarchy structural information
            of shape :math:`(\\text{batch_size}, 2, \\text{max_level}+2)`.
            See :ref:`pyramids <spc_pyramids>`.
        exsum (torch.IntTensor):
            Tensor containing the :ref:`packed` exclusive sum of the bit
            counts of individual octrees of shape :math:`(\\text{num_bytes} + \\text{batch_size})`.
            See :ref:`exsum <spc_exsum>`.
        input (torch.FloatTensor):
            :ref:`packed` input feature data of the octrees,
            of shape :math:`(\\text{total_num_inputs}, \\text{in_channels})`,
            where ``total_num_inputs`` correspond to the number of nodes of the octrees at ``level``,
            and ``in_channels`` is the input feature dimension (for instance 3 for RGB color).
        weight (torch.FloatTensor):
            filter of shape :math:`(\\text{kernel_vectors.shape[0]}, \\text{in_channels},
            \\text{self.out_channels})`.
        kernel_vectors (torch.ShortTensor):
            A tensor of 3D offsets that define the shape of the kernel,
            of shape :math:`(\\text{num_weights}, 3)`.
            See :ref:`kernel creation <kernel-text>`.
        jump (int, optional):
            The difference between the input and output levels for the convolution.
            A non-zero value implies downsampling. Value must be positive and refer to a valid level of
            the structured point cloud. Default: 0.
        bias (torch.FloatTensor, optional):
            optional bias tensor of shape :math:`(\\text{out_channel})`.

    Returns:
        (torch.FloatTensor, int):

            - Output of convolution. Number of outputs will correspond
              to level in the hierachy determined by **jump**.

            - the level associated to the output features.
    """
    remaining_kwargs = kwargs.keys() - Spc.KEYS
    if len(remaining_kwargs) > 0:
        raise TypeError(f'conv3d got an unexpected keyword argument {list(remaining_kwargs)[0]}')
    if weight.shape[0] == 1 and jump == 0:
        outputs = input.mm(weight.squeeze(0))
    else:
        outputs, level = Conv3dFunction.apply(octrees, point_hierarchies, level, pyramids, exsum, input, weight, kernel_vectors, jump)
    if bias is not None:
        outputs += bias.unsqueeze(0)
    return outputs, int(level)


class Conv3d(nn.Module):
    """Convolution layer for a structured point cloud. The inputs :math:`X` are mapped
    to outputs :math:`Y` by the following:

    .. math::

        Y_i = \\sum_k w_k \\cdot X_{n(i,k)} + b \\quad\\text{for}\\; i \\in 0,\\ldots,|Y|-1,

    where :math:`w_k` are weights associated with the kernel, and :math:`n(i,k)` is the
    neighborhood function described :ref:`here <neighborhood-text>`.

    Args:
        in_channels (int):
            The number of channels in the input tensor.
        out_channels (int):
            The number of channels in the output tensor.
        kernel_vectors (torch.ShortTensor):
            A tensor of 3D offsets that define the shape of the kernel,
            of shape :math:`(\\text{num_weights}, 3)`.
            See :ref:`kernel creation <kernel-text>`.
        jump (int, optional):
            The difference between the input and output levels for the convolution.
            A non-zero value implies downsampling. Value must be positive and refer to a valid level of
            the structured point cloud. Default: 0.
        bias (bool, optional):
            If True, the convolution layer has a bias. Default: True.
    """

    def __init__(self, in_channels, out_channels, kernel_vectors, jump=0, bias=True):
        super(Conv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_vectors_size = kernel_vectors.shape[0]
        self.jump = jump
        self.register_buffer('kernel_vectors', kernel_vectors)
        self.kernel_shape = self.kernel_vectors_size, self.in_channels, self.out_channels
        self.weight = nn.Parameter(torch.empty(*self.kernel_shape))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels * self.kernel_vectors_size
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, octrees, point_hierarchies, level, pyramids, exsum, input, **kwargs):
        """
        Args:
            octrees (torch.ByteTensor):
                :ref:`packed` octrees of shape :math:`(\\text{num_bytes})`.
                See :ref:`octree <spc_octree>`.

            point_hierarchies (torch.ShortTensor):
                :ref:`packed` point hierarchies of shape :math:`(\\text{num_points})`.
                See :ref:`point_hierarchies <spc_points>`.

            level (int):
                level at which the ``input`` features are associated to.

            pyramids (torch.IntTensor):
                Batched tensor containing point hierarchy structural information
                of shape :math:`(\\text{batch_size}, 2, \\text{max_level}+2)`.
                See :ref:`pyramids <spc_pyramids>`.

            exsum (torch.IntTensor):
                Tensor containing the :ref:`packed` exclusive sum of the bit
                counts of individual octrees of shape :math:`(\\text{num_bytes} + \\text{batch_size})`.
                See :ref:`exsum <spc_exsum>`.

            input (torch.FloatTensor):
                :ref:`packed` input feature data of the octrees,
                of shape :math:`(\\text{total_num_inputs}, \\text{in_channels})`,
                where ``total_num_inputs`` correspond to the number of nodes of the octrees at ``level``,
                and ``in_channels`` is the input feature dimension (for instance 3 for RGB color).

        Returns:
            (torch.FloatTensor, int):

                - Output of convolution. Number of outputs will correspond
                  to level in the hierachy determined by **jump**.

                - the level associated to the output features.
        """
        remaining_kwargs = kwargs.keys() - Spc.KEYS
        if len(remaining_kwargs) > 0:
            raise TypeError(f'Conv3d got an unexpected keyword argument {list(remaining_kwargs)[0]}')
        return conv3d(octrees, point_hierarchies, level, pyramids, exsum, input, self.weight, self.kernel_vectors, self.jump, self.bias)

    def __repr__(self):
        s = '(in={}, out={}, kernel_vector_size={})'.format(self.in_channels, self.out_channels, self.kernel_vectors_size)
        return self.__class__.__name__ + s


class ConvTranspose3dFunction(Function):

    @staticmethod
    def forward(ctx, octrees, point_hierarchies, level, pyramids, exsum, inputs, params, kernel_vectors, jump):
        octrees = octrees.contiguous()
        point_hierarchies = point_hierarchies.contiguous()
        pyramids = pyramids.contiguous()
        exsum = exsum.contiguous()
        inputs = inputs.contiguous()
        params = params.contiguous()
        kernel_vectors = kernel_vectors.contiguous()
        ctx.save_for_backward(octrees, point_hierarchies, pyramids, exsum, inputs, params, kernel_vectors)
        ctx.jump = jump
        outputs, level = _C.ops.spc.ConvTranspose3d_forward(octrees, point_hierarchies, level, pyramids, exsum, inputs, params, kernel_vectors, jump)
        ctx.level = level
        level = torch.tensor([level])
        ctx.mark_non_differentiable(level)
        return outputs, level

    @staticmethod
    def backward(ctx, grad_outputs, grad_level):
        grad_outputs = grad_outputs.contiguous()
        octrees, point_hierarchies, pyramids, exsum, inputs, params, kernel_vectors = ctx.saved_tensors
        d_inputs, d_params = _C.ops.spc.ConvTranspose3d_backward(octrees, point_hierarchies, ctx.level, pyramids, exsum, inputs, grad_outputs, params, kernel_vectors, ctx.jump)
        return None, None, None, None, None, d_inputs, d_params, None, None


def conv_transpose3d(octrees, point_hierarchies, level, pyramids, exsum, input, weight, kernel_vectors, jump=0, bias=None, **kwargs):
    """Transposed convolution over a structured point cloud. The inputs :math:`X` are mapped
    to outputs :math:`Y` by the following:

    .. math::

        Y_i = \\sum_k w_k \\cdot X_{n^T(i,k)} + b \\quad\\text{for}\\; i \\in 0,\\ldots,|Y|-1,

    where :math:`w_k` are weights associated with the kernel, and :math:`n^T(i,k)` is the
    transpose neighborhood function described :ref:`here <neighborhood-text>`.


    Args:
        octrees (torch.ByteTensor):
            :ref:`packed` octrees of shape :math:`(\\text{num_bytes})`.
            See :ref:`octree <spc_octree>`.

        point_hierarchies (torch.ShortTensor):
            :ref:`packed` point hierarchies of shape :math:`(\\text{num_points})`.
            See :ref:`point_hierarchies <spc_points>`.

        level (int):
            level at which the ``input`` features are associated to.

        pyramids (torch.IntTensor):
            Batched tensor containing point hierarchy structural information
            of shape :math:`(\\text{batch_size}, 2, \\text{max_level}+2)`.
            See :ref:`pyramids <spc_pyramids>`.

        exsum (torch.IntTensor):
            Tensor containing the :ref:`packed` exclusive sum of the bit
            counts of individual octrees of shape :math:`(\\text{num_bytes} + \\text{batch_size})`.
            See :ref:`exsum <spc_exsum>`.

        input (torch.FloatTensor):
            :ref:`packed` input feature data of the octrees,
            of shape :math:`(\\text{total_num_inputs}, \\text{in_channels})`,
            where ``total_num_inputs`` correspond to the number of nodes of the octrees at ``level``,
            and ``in_channels`` is the input feature dimension (for instance 3 for RGB color).

        weight (torch.FloatTensor):
            filter of shape :math:`(\\text{kernel_vectors.shape[0]}, \\text{in_channels},
            \\text{self.out_channels})`.

        kernel_vectors (torch.ShortTensor):
            A tensor of 3D offsets that define the shape of the kernel,
            of shape :math:`(\\text{num_weights}, 3)`.
            See :ref:`kernel creation <kernel-text>`.

        jump (int, optional):
            The difference between the input and output levels for the convolution.
            A non-zero value implies downsampling. Value must be positive and refer to a valid level of
            the structured point cloud. Default: 0.

        bias (torch.FloatTensor, optional):
            optional bias tensor of shape :math:`(\\text{out_channel})`.
    """
    remaining_kwargs = kwargs.keys() - Spc.KEYS
    if len(remaining_kwargs) > 0:
        raise TypeError(f'conv_transpose3d got an unexpected keyword argument {list(remaining_kwargs)[0]}')
    if weight.shape[0] == 1 and jump == 0:
        outputs = input.mm(weight.squeeze(0))
    else:
        outputs, level = ConvTranspose3dFunction.apply(octrees, point_hierarchies, level, pyramids, exsum, input, weight, kernel_vectors, jump)
    if bias is not None:
        outputs += bias.unsqueeze(0)
    return outputs, int(level)


class ConvTranspose3d(nn.Module):
    """Transposed convolution layer for a structured point cloud. The inputs :math:`X` are mapped
    to outputs :math:`Y` by the following:

    .. math::

        Y_i = \\sum_k w_k \\cdot X_{n^T(i,k)} + b \\quad\\text{for}\\; i \\in 0,\\ldots,|Y|-1,

    where :math:`w_k` are weights associated with the kernel, and :math:`n^T(i,k)` is the
    transpose neighborhood function described :ref:`here <neighborhood-text>`.

    Args:
        in_channels (int):
            The number of channels in the input tensor.

        out_channels (int):
            The number of channels in the output tensor.

        kernel_vectors (torch.ShortTensor):
            A tensor of 3D offsets that define the shape of the kernel,
            of shape :math:`(\\text{num_weights}, 3)`.
            See :ref:`kernel creation <kernel-text>`.

        jump (int, optional):
            The difference between the input and output levels for the convolution. Default: 0.
            A non-zero value implies upsampling. Value must be positive and refer to a valid level of
            the structured point cloud.

        bias (bool, optional):
            If True, the convolution layer has a bias. Default: True.
    """

    def __init__(self, in_channels, out_channels, kernel_vectors, jump=0, bias=True):
        super(ConvTranspose3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_vectors_size = kernel_vectors.shape[0]
        self.jump = jump
        self.register_buffer('kernel_vectors', kernel_vectors)
        self.kernel_shape = self.kernel_vectors_size, self.in_channels, self.out_channels
        self.weight = nn.Parameter(torch.empty(*self.kernel_shape))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.out_channels * self.kernel_vectors_size
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, octrees, point_hierarchies, level, pyramids, exsum, input, **kwargs):
        """
        Args:
            octrees (torch.ByteTensor):
                :ref:`packed` octrees of shape :math:`(\\text{num_bytes})`.
                See :ref:`octree <spc_octree>`.

            point_hierarchies (torch.ShortTensor):
                :ref:`packed` point hierarchies of shape :math:`(\\text{num_points})`.
                See :ref:`point_hierarchies <spc_points>`.

            level (int):
                level at which the ``input`` features are associated to.

            pyramids (torch.IntTensor):
                Batched tensor containing point hierarchy structural information
                of shape :math:`(\\text{batch_size}, 2, \\text{max_level}+2)`.
                See :ref:`pyramids <spc_pyramids>`.

            exsum (torch.IntTensor):
                Tensor containing the :ref:`packed` exclusive sum of the bit
                counts of individual octrees of shape :math:`(\\text{num_bytes} + \\text{batch_size})`.
                See :ref:`exsum <spc_exsum>`.

            input (torch.FloatTensor):
                :ref:`packed` input feature data of the octrees,
                of shape :math:`(\\text{total_num_inputs}, \\text{in_channels})`,
                where ``total_num_inputs`` correspond to the number of nodes of the octrees at ``level``,
                and ``in_channels`` is the input feature dimension (for instance 3 for RGB color).

        Returns:
            (torch.FloatTensor, int):

                - Output of transpose convolution. Number of outputs will correspond
                  to level in the hierachy determined by **jump**.

                - the level associated to the output features.
        """
        remaining_kwargs = kwargs.keys() - Spc.KEYS
        if len(remaining_kwargs) > 0:
            raise TypeError(f'ConvTranspose3d got an unexpected keyword argument {list(remaining_kwargs)[0]}')
        return conv_transpose3d(octrees, point_hierarchies, level, pyramids, exsum, input, self.weight, self.kernel_vectors, self.jump, self.bias)

    def __repr__(self):
        s = '(in={}, out={}, kernel_vector_size={})'.format(self.in_channels, self.out_channels, self.kernel_vectors_size)
        return self.__class__.__name__ + s


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (GraphConv,
     lambda: ([], {'input_dim': 4, 'output_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_NVIDIAGameWorks_kaolin(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

