import sys
_module = sys.modules[__name__]
del sys
preprocess = _module
src = _module
data = _module
dataset = _module
dictionary = _module
loader = _module
evaluation = _module
evaluator = _module
glue = _module
xnli = _module
logger = _module
model = _module
embedder = _module
memory = _module
memory = _module
query = _module
utils = _module
pretrain = _module
transformer = _module
optim = _module
slurm = _module
trainer = _module
lowercase_and_remove_accent = _module
segment_th = _module
train = _module
translate = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, string, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
import numpy as np
from torch import Tensor
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


from logging import getLogger


import copy


import time


from collections import OrderedDict


import numpy as np


import torch


from torch import nn


import torch.nn.functional as F


from scipy.stats import spearmanr


from scipy.stats import pearsonr


import math


import itertools


from torch.nn import functional as F


import torch.nn as nn


from torch.nn.utils import clip_grad_norm_


def get_gaussian_keys(n_keys, dim, normalized, seed):
    """
    Generate random Gaussian keys.
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n_keys, dim)
    if normalized:
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X.astype(np.float32)


def get_uniform_keys(n_keys, dim, normalized, seed):
    """
    Generate random uniform keys (same initialization as nn.Linear).
    """
    rng = np.random.RandomState(seed)
    bound = 1 / math.sqrt(dim)
    X = rng.uniform(-bound, bound, (n_keys, dim))
    if normalized:
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X.astype(np.float32)


def cartesian_product(a, b):
    """
    Compute the batched cartesian product between two matrices.
    Input:
        a: Tensor(n, d1)
        b: Tensor(n, d2)
    Output:
        output: Tensor(n, d1 * d2, 2)
    """
    n1, d1 = a.shape
    n2, d2 = b.shape
    assert n1 == n2
    return torch.cat([a.unsqueeze(-1).repeat(1, 1, d2).unsqueeze(-1), b.repeat(1, d1).view(n2, d1, d2).unsqueeze(-1)], 3).view(n1, d1 * d2, 2)


def swig_ptr_from_FloatTensor(x):
    assert x.is_contiguous()
    assert x.dtype == torch.float32
    return faiss.cast_integer_to_float_ptr(x.storage().data_ptr() + x.storage_offset() * 4)


def swig_ptr_from_LongTensor(x):
    assert x.is_contiguous()
    assert x.dtype == torch.int64, 'dtype=%s' % x.dtype
    return faiss.cast_integer_to_long_ptr(x.storage().data_ptr() + x.storage_offset() * 8)


def get_knn_faiss(xb, xq, k, distance='dot_product'):
    """
    `metric` can be faiss.METRIC_INNER_PRODUCT or faiss.METRIC_L2
    https://github.com/facebookresearch/faiss/blob/master/gpu/test/test_pytorch_faiss.py
    """
    assert xb.device == xq.device
    assert distance in ['dot_product', 'l2']
    metric = faiss.METRIC_INNER_PRODUCT if distance == 'dot_product' else faiss.METRIC_L2
    xq_ptr = swig_ptr_from_FloatTensor(xq)
    xb_ptr = swig_ptr_from_FloatTensor(xb)
    nq, d1 = xq.size()
    nb, d2 = xb.size()
    assert d1 == d2
    D = torch.empty(nq, k, device=xb.device, dtype=torch.float32)
    I = torch.empty(nq, k, device=xb.device, dtype=torch.int64)
    D_ptr = swig_ptr_from_FloatTensor(D)
    I_ptr = swig_ptr_from_LongTensor(I)
    faiss.bruteForceKnn(FAISS_RES, metric, xb_ptr, nb, xq_ptr, nq, d1, k, D_ptr, I_ptr)
    return D, I


FALSY_STRINGS = {'off', 'false', '0'}


TRUTHY_STRINGS = {'on', 'true', '1'}


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError('Invalid value for a boolean flag!')


logger = getLogger()


class HashingMemory(nn.Module):
    MEM_VALUES_PARAMS = '.values.weight'
    VALUES = None
    EVAL_MEMORY = True
    _ids = itertools.count(0)

    def __init__(self, input_dim, output_dim, params):
        super().__init__()
        self.id = next(self._ids)
        self.input2d = params.mem_input2d
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.size = params.mem_size
        self.modulo_size = params.mem_modulo_size
        self.n_indices = params.n_indices
        self.k_dim = params.mem_k_dim
        self.v_dim = params.mem_v_dim if params.mem_v_dim > 0 else output_dim
        self.heads = params.mem_heads
        self.knn = params.mem_knn
        self.shuffle_indices = params.mem_shuffle_indices
        self.keys_normalized_init = params.mem_keys_normalized_init
        self.product_quantization = params.mem_product_quantization
        assert self.modulo_size == -1 and self.size == self.n_indices or self.n_indices > self.size == self.modulo_size >= 1
        self.keys_type = params.mem_keys_type
        self.learn_keys = params.mem_keys_learn
        self.use_different_keys = params.mem_use_different_keys
        self.query_detach_input = params.mem_query_detach_input
        self.query_net_learn = params.mem_query_net_learn
        self.multi_query_net = params.mem_multi_query_net
        self.shuffle_query = params.mem_shuffle_query
        assert self.use_different_keys is False or self.keys_type in ['gaussian', 'uniform']
        assert self.use_different_keys is False or self.heads >= 2 or self.product_quantization
        assert self.multi_query_net is False or self.heads >= 2 or self.product_quantization
        assert self.shuffle_query is False or self.heads > 1 and params.mem_query_layer_sizes == ''
        assert self.shuffle_query is False or self.input_dim % 2 ** self.heads == 0
        self.normalize_query = params.mem_normalize_query
        self.temperature = params.mem_temperature
        self.score_softmax = params.mem_score_softmax
        self.score_subtract = params.mem_score_subtract
        self.score_normalize = params.mem_score_normalize
        assert self.score_subtract in ['', 'min', 'mean', 'median']
        assert self.score_subtract == '' or self.knn >= 2
        assert not (self.score_normalize and self.score_softmax and self.score_subtract == '')
        self.input_dropout = params.mem_input_dropout
        self.query_dropout = params.mem_query_dropout
        self.value_dropout = params.mem_value_dropout
        self.init_keys()
        self.values = nn.EmbeddingBag(self.size, self.v_dim, mode='sum', sparse=params.mem_sparse)
        if params.mem_share_values:
            if HashingMemory.VALUES is None:
                HashingMemory.VALUES = self.values.weight
            else:
                self.values.weight = HashingMemory.VALUES
        if params.mem_value_zero_init:
            nn.init.zeros_(self.values.weight)
        else:
            nn.init.normal_(self.values.weight, mean=0, std=self.v_dim ** -0.5)
        if len(params.mem_query_layer_sizes) == 0:
            assert self.heads == 1 or self.use_different_keys or self.shuffle_query
            assert self.input_dim == self.k_dim
            self.query_proj = QueryIdentity(self.input_dim, self.heads, self.shuffle_query)
        if len(params.mem_query_layer_sizes) > 0:
            assert not self.shuffle_query
            l_sizes = list(params.mem_query_layer_sizes)
            assert len(l_sizes) >= 2 and l_sizes[0] == l_sizes[-1] == 0
            l_sizes[0] = self.input_dim
            l_sizes[-1] = self.k_dim // 2 if self.multi_query_net else self.heads * self.k_dim
            if self.input2d:
                self.query_proj = QueryConv(self.input_dim, self.heads, self.k_dim, self.product_quantization, self.multi_query_net, l_sizes, params.mem_query_kernel_sizes, bias=params.mem_query_bias, batchnorm=params.mem_query_batchnorm, grouped_conv=params.mem_grouped_conv)
            else:
                assert params.mem_query_kernel_sizes == ''
                assert not params.mem_query_residual
                self.query_proj = QueryMLP(self.input_dim, self.heads, self.k_dim, self.product_quantization, self.multi_query_net, l_sizes, bias=params.mem_query_bias, batchnorm=params.mem_query_batchnorm, grouped_conv=params.mem_grouped_conv)
        if self.shuffle_indices:
            head_permutations = [torch.randperm(self.n_indices).unsqueeze(0) for i in range(self.heads)]
            self.register_buffer('head_permutations', torch.cat(head_permutations, 0))
        if self.query_net_learn is False:
            for p in self.query_proj.parameters():
                p.requires_grad = False

    def forward(self, input):
        """
        Read from the memory.
        """
        if self.query_detach_input:
            input = input.detach()
        if self.input2d:
            assert input.shape[1] == self.input_dim
            n_images, _, height, width = input.shape
            prefix_shape = n_images, width, height
        else:
            assert input.shape[-1] == self.input_dim
            prefix_shape = input.shape[:-1]
        bs = np.prod(prefix_shape)
        input = F.dropout(input, p=self.input_dropout, training=self.training)
        query = self.query_proj(input)
        query = F.dropout(query, p=self.query_dropout, training=self.training)
        assert query.shape == (bs * self.heads, self.k_dim)
        scores, indices = self.get_indices(query, self.knn)
        if self.shuffle_indices:
            indices = indices.view(bs, self.heads, -1).chunk(self.heads, 1)
            indices = [p[idx] for p, idx in zip(self.head_permutations, indices)]
            indices = torch.cat(indices, 1).view(bs * self.heads, -1)
        if self.modulo_size != -1:
            indices = indices % self.modulo_size
        if self.temperature != 1:
            scores = scores / self.temperature
        if self.score_softmax:
            scores = F.softmax(scores.float(), dim=-1).type_as(scores)
        if self.score_subtract != '':
            if self.score_subtract == 'min':
                to_sub = scores.min(1, keepdim=True)[0]
            if self.score_subtract == 'mean':
                to_sub = scores.mean(1, keepdim=True)
            if self.score_subtract == 'median':
                to_sub = scores.median(1, keepdim=True)[0]
            scores = scores - to_sub
        if self.score_normalize:
            scores = scores / scores.norm(p=1, dim=1, keepdim=True)
        indices = indices.view(bs, self.heads * self.knn)
        scores = scores.view(bs, self.heads * self.knn)
        output = self.values(indices, per_sample_weights=scores.to(self.values.weight.data))
        output = F.dropout(output, p=self.value_dropout, training=self.training)
        if self.input2d:
            output = output.view(n_images, width, height, self.v_dim)
            output = output.transpose(1, 3)
        elif len(prefix_shape) >= 2:
            output = output.view(prefix_shape + (self.v_dim,))
        if not self.training and HashingMemory.EVAL_MEMORY:
            self.last_indices = indices.view(bs, self.heads, self.knn).detach().cpu()
            self.last_scores = scores.view(bs, self.heads, self.knn).detach().cpu().float()
        return output

    def init_keys(self):
        raise Exception('Not implemented!')

    def _get_indices(self, query, knn, keys):
        raise Exception('Not implemented!')

    def get_indices(self, query, knn):
        raise Exception('Not implemented!')

    @staticmethod
    def register_args(parser):
        """
        Register memory parameters
        """
        parser.add_argument('--mem_implementation', type=str, default='pq_fast', help='Memory implementation (flat, pq_default, pq_fast)')
        parser.add_argument('--mem_grouped_conv', type=bool_flag, default=False, help='Use grouped convolutions in the query network')
        parser.add_argument('--mem_values_optimizer', type=str, default='adam,lr=0.001', help='Memory values optimizer ( for the same optimizer as the rest of the model)')
        parser.add_argument('--mem_sparse', type=bool_flag, default=False, help='Perform sparse updates for the values')
        parser.add_argument('--mem_input2d', type=bool_flag, default=False, help='Convolutional query network')
        parser.add_argument('--mem_k_dim', type=int, default=256, help='Memory keys dimension')
        parser.add_argument('--mem_v_dim', type=int, default=-1, help='Memory values dimension (-1 for automatic output dimension)')
        parser.add_argument('--mem_heads', type=int, default=4, help='Number of memory reading heads')
        parser.add_argument('--mem_knn', type=int, default=32, help='Number of memory slots to read / update - k-NN to the query')
        parser.add_argument('--mem_share_values', type=bool_flag, default=False, help='Share values across memories')
        parser.add_argument('--mem_shuffle_indices', type=bool_flag, default=False, help='Shuffle indices for different heads')
        parser.add_argument('--mem_shuffle_query', type=bool_flag, default=False, help='Shuffle query dimensions (when the query network is the identity and there are multiple heads)')
        parser.add_argument('--mem_modulo_size', type=int, default=-1, help='Effective memory size: indices are taken modulo this parameter. -1 to disable.')
        parser.add_argument('--mem_keys_type', type=str, default='uniform', help='Memory keys type (binary,gaussian,uniform)')
        parser.add_argument('--mem_n_keys', type=int, default=512, help='Number of keys')
        parser.add_argument('--mem_keys_normalized_init', type=bool_flag, default=False, help='Normalize keys at initialization')
        parser.add_argument('--mem_keys_learn', type=bool_flag, default=True, help='Learn keys')
        parser.add_argument('--mem_use_different_keys', type=bool_flag, default=True, help='Use different keys for each head / product quantization')
        parser.add_argument('--mem_query_detach_input', type=bool_flag, default=False, help='Detach input')
        parser.add_argument('--mem_query_layer_sizes', type=str, default='0,0', help="Query MLP layer sizes ('', '0,0', '0,512,0')")
        parser.add_argument('--mem_query_kernel_sizes', type=str, default='', help='Query MLP kernel sizes (2D inputs only)')
        parser.add_argument('--mem_query_bias', type=bool_flag, default=True, help='Query MLP bias')
        parser.add_argument('--mem_query_batchnorm', type=bool_flag, default=False, help='Query MLP batch norm')
        parser.add_argument('--mem_query_net_learn', type=bool_flag, default=True, help='Query MLP learn')
        parser.add_argument('--mem_query_residual', type=bool_flag, default=False, help='Use a bottleneck with a residual layer in the query MLP')
        parser.add_argument('--mem_multi_query_net', type=bool_flag, default=False, help='Use multiple query MLP (one for each head)')
        parser.add_argument('--mem_value_zero_init', type=bool_flag, default=False, help='Initialize values with zeros')
        parser.add_argument('--mem_normalize_query', type=bool_flag, default=False, help='Normalize queries')
        parser.add_argument('--mem_temperature', type=float, default=1, help='Divide scores by a temperature')
        parser.add_argument('--mem_score_softmax', type=bool_flag, default=True, help='Apply softmax on scores')
        parser.add_argument('--mem_score_subtract', type=str, default='', help="Subtract scores ('', min, mean, median)")
        parser.add_argument('--mem_score_normalize', type=bool_flag, default=False, help='L1 normalization of the scores')
        parser.add_argument('--mem_input_dropout', type=float, default=0, help='Input dropout')
        parser.add_argument('--mem_query_dropout', type=float, default=0, help='Query dropout')
        parser.add_argument('--mem_value_dropout', type=float, default=0, help='Value dropout')

    @staticmethod
    def build(input_dim, output_dim, params):
        if params.mem_implementation == 'flat':
            M = HashingMemoryFlat
        elif params.mem_implementation == 'pq_default':
            M = HashingMemoryProduct
        elif params.mem_implementation == 'pq_fast':
            M = HashingMemoryProductFast
        else:
            raise Exception('Unknown memory implementation!')
        return M(input_dim, output_dim, params)

    @staticmethod
    def check_params(params):
        """
        Check and initialize memory parameters.
        """
        assert params.mem_implementation in ['flat', 'pq_default', 'pq_fast']
        params.mem_product_quantization = params.mem_implementation != 'flat'
        assert params.mem_grouped_conv is False or params.mem_multi_query_net
        params.mem_values_optimizer = params.optimizer if params.mem_values_optimizer == '' else params.mem_values_optimizer
        params.mem_values_optimizer = params.mem_values_optimizer.replace('adam', 'sparseadam') if params.mem_sparse else params.mem_values_optimizer
        assert params.mem_k_dim >= 2
        assert params.mem_product_quantization is False or params.mem_k_dim % 2 == 0
        assert params.mem_keys_type in ['binary', 'gaussian', 'uniform']
        if params.mem_keys_type == 'binary':
            assert params.mem_keys_normalized_init is False
            assert 1 << params.mem_k_dim == params.mem_n_keys
        if params.mem_product_quantization:
            params.n_indices = params.mem_n_keys ** 2
        else:
            params.n_indices = params.mem_n_keys
        if params.mem_modulo_size == -1:
            params.mem_size = params.n_indices
        else:
            assert 1 <= params.mem_modulo_size < params.n_indices
            params.mem_size = params.mem_modulo_size
        assert not params.mem_use_different_keys or params.mem_keys_type in ['gaussian', 'uniform']
        assert not params.mem_use_different_keys or params.mem_heads >= 2 or params.mem_product_quantization
        assert not params.mem_multi_query_net or params.mem_heads >= 2 or params.mem_product_quantization
        assert not params.mem_multi_query_net or params.mem_query_layer_sizes not in ['', '0,0']
        assert not params.mem_shuffle_query or params.mem_heads > 1 and params.mem_query_layer_sizes == ''
        if params.mem_query_layer_sizes == '':
            assert params.mem_heads == 1 or params.mem_use_different_keys or params.mem_shuffle_query
        else:
            s = [int(x) for x in filter(None, params.mem_query_layer_sizes.split(','))]
            assert len(s) >= 2 and s[0] == s[-1] == 0
            params.mem_query_layer_sizes = s
            assert not params.mem_query_residual or params.mem_input2d
        if params.mem_query_kernel_sizes == '':
            assert not params.mem_input2d or params.mem_query_layer_sizes == ''
        else:
            assert params.mem_input2d
            s = [int(x) for x in filter(None, params.mem_query_kernel_sizes.split(','))]
            params.mem_query_kernel_sizes = s
            assert all(ks % 2 == 1 for ks in s)
            assert len(params.mem_query_kernel_sizes) == len(params.mem_query_layer_sizes) - 1 >= 1
        assert params.mem_score_subtract in ['', 'min', 'mean', 'median']
        assert params.mem_score_subtract == '' or params.mem_knn >= 2
        assert not (params.mem_score_normalize and params.mem_score_softmax and params.mem_score_subtract == '')
        assert 0 <= params.mem_input_dropout < 1
        assert 0 <= params.mem_query_dropout < 1
        assert 0 <= params.mem_value_dropout < 1
        if params.mem_query_batchnorm:
            logger.warning('WARNING: if you use batch normalization, be sure that you use batches of sentences with the same size at training time. Otherwise, the padding token will result in incorrect mean/variance estimations in the BatchNorm layer.')


class GroupedLinear(nn.Module):

    def __init__(self, in_features, out_features, bias=True, groups=1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.groups = groups
        self.bias = bias
        assert groups > 1
        self.layer = nn.Conv1d(in_features, out_features, bias=bias, kernel_size=1, groups=groups)

    def forward(self, input):
        assert input.dim() == 2 and input.size(1) == self.in_features
        return self.layer(input.unsqueeze(2)).squeeze(2)

    def extra_repr(self):
        return 'in_features={}, out_features={}, groups={}, bias={}'.format(self.in_features, self.out_features, self.groups, self.bias is not None)


class BottleneckResidualConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, bias=True, batchnorm=True, groups=1):
        super().__init__()
        hidden_channels = min(input_channels, output_channels)
        assert all(k % 2 == 1 for k in kernel_size)
        self.conv1 = nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=[(k // 2) for k in kernel_size], bias=bias, groups=groups)
        self.conv2 = nn.Conv2d(hidden_channels, output_channels, kernel_size, padding=[(k // 2) for k in kernel_size], bias=bias, groups=groups)
        self.act = nn.ReLU()
        self.batchnorm = batchnorm
        if self.batchnorm:
            self.bn1 = nn.BatchNorm2d(hidden_channels)
            self.bn2 = nn.BatchNorm2d(output_channels)
        if input_channels == output_channels:
            self.residual = nn.Sequential()
        else:
            self.residual = nn.Conv2d(input_channels, output_channels, (1, 1), bias=False, groups=groups)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x) if self.batchnorm else x
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x) if self.batchnorm else x
        x = self.act(x + self.residual(input))
        return x


def get_slices(dim, head_id):
    """
    Generate slices of hidden dimensions.
    Used when there are multiple heads and/or different set of keys,
    and that there is no query network.
    """
    if head_id == 0:
        return [(0, dim)]
    offset = dim // 2 ** (head_id + 1)
    starts = np.arange(0, dim, offset)
    slices1 = [(x, x + offset) for i, x in enumerate(starts) if i % 2 == 0]
    slices2 = [(x, x + offset) for i, x in enumerate(starts) if i % 2 == 1]
    return slices1 + slices2


class QueryIdentity(nn.Module):

    def __init__(self, input_dim, heads, shuffle_hidden):
        super().__init__()
        self.input_dim = input_dim
        self.heads = heads
        self.shuffle_query = shuffle_hidden
        assert shuffle_hidden is False or heads > 1
        assert shuffle_hidden is False or self.input_dim % 2 ** self.heads == 0
        if shuffle_hidden:
            self.slices = {head_id: get_slices(input_dim, head_id) for head_id in range(heads)}

    def forward(self, input):
        """
        Generate queries from hidden states by either
        repeating them or creating some shuffled version.
        """
        assert input.shape[-1] == self.input_dim
        input = input.contiguous().view(-1, self.input_dim) if input.dim() > 2 else input
        bs = len(input)
        if self.heads == 1:
            query = input
        elif not self.shuffle_query:
            query = input.unsqueeze(1).repeat(1, self.heads, 1)
            query = query.view(bs * self.heads, self.input_dim)
        else:
            query = torch.cat([input[:, a:b] for head_id in range(self.heads) for a, b in self.slices[head_id]], 1).view(bs * self.heads, self.input_dim)
        assert query.shape == (bs * self.heads, self.input_dim)
        return query


def mlp(sizes, bias=True, batchnorm=True, groups=1):
    """
    Generate a feedforward neural network.
    """
    assert len(sizes) >= 2
    pairs = [(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]
    layers = []
    for i, (dim_in, dim_out) in enumerate(pairs):
        if groups == 1 or i == 0:
            layers.append(nn.Linear(dim_in, groups * dim_out, bias=bias))
        else:
            layers.append(GroupedLinear(groups * dim_in, groups * dim_out, bias=bias, groups=groups))
        if batchnorm:
            layers.append(nn.BatchNorm1d(groups * dim_out))
        if i < len(pairs) - 1:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class QueryMLP(nn.Module):

    def __init__(self, input_dim, heads, k_dim, product_quantization, multi_query_net, sizes, bias=True, batchnorm=True, grouped_conv=False):
        super().__init__()
        self.input_dim = input_dim
        self.heads = heads
        self.k_dim = k_dim
        self.sizes = sizes
        self.grouped_conv = grouped_conv
        assert not multi_query_net or product_quantization or heads >= 2
        assert sizes[0] == input_dim
        assert sizes[-1] == k_dim // 2 if multi_query_net else heads * k_dim
        assert self.grouped_conv is False or len(sizes) > 2
        self.groups = 2 * heads if multi_query_net else 1
        if self.grouped_conv:
            self.query_mlps = mlp(sizes, bias=bias, batchnorm=batchnorm, groups=self.groups)
        elif len(self.sizes) == 2:
            sizes_ = list(sizes)
            sizes_[-1] = sizes_[-1] * self.groups
            self.query_mlps = mlp(sizes_, bias=bias, batchnorm=batchnorm, groups=1)
        else:
            self.query_mlps = nn.ModuleList([mlp(sizes, bias=bias, batchnorm=batchnorm, groups=1) for _ in range(self.groups)])

    def forward(self, input):
        """
        Compute queries using either grouped 1D convolutions or ModuleList + concat.
        """
        assert input.shape[-1] == self.input_dim
        input = input.contiguous().view(-1, self.input_dim) if input.dim() > 2 else input
        bs = len(input)
        if self.grouped_conv or len(self.sizes) == 2:
            query = self.query_mlps(input)
        else:
            outputs = [m(input) for m in self.query_mlps]
            query = torch.cat(outputs, 1) if len(outputs) > 1 else outputs[0]
        assert query.shape == (bs, self.heads * self.k_dim)
        return query.view(bs * self.heads, self.k_dim)


def convs(channel_sizes, kernel_sizes, bias=True, batchnorm=True, residual=False, groups=1):
    """
    Generate a convolutional neural network.
    """
    assert len(channel_sizes) >= 2
    assert len(channel_sizes) == len(kernel_sizes) + 1
    pairs = [(channel_sizes[i], channel_sizes[i + 1]) for i in range(len(channel_sizes) - 1)]
    layers = []
    for i, (dim_in, dim_out) in enumerate(pairs):
        ks = kernel_sizes[i], kernel_sizes[i]
        in_group = 1 if i == 0 else groups
        _dim_in = dim_in * in_group
        _dim_out = dim_out * groups
        if not residual:
            layers.append(nn.Conv2d(_dim_in, _dim_out, ks, padding=[(k // 2) for k in ks], bias=bias, groups=in_group))
            if batchnorm:
                layers.append(nn.BatchNorm2d(_dim_out))
            if i < len(pairs) - 1:
                layers.append(nn.ReLU())
        else:
            layers.append(BottleneckResidualConv2d(_dim_in, _dim_out, ks, bias=bias, batchnorm=batchnorm, groups=in_group))
            if i == len(pairs) - 1:
                layers.append(nn.Conv2d(_dim_out, _dim_out, (1, 1), bias=bias))
    return nn.Sequential(*layers)


class QueryConv(nn.Module):

    def __init__(self, input_dim, heads, k_dim, product_quantization, multi_query_net, sizes, kernel_sizes, bias=True, batchnorm=True, residual=False, grouped_conv=False):
        super().__init__()
        self.input_dim = input_dim
        self.heads = heads
        self.k_dim = k_dim
        self.sizes = sizes
        self.grouped_conv = grouped_conv
        assert not multi_query_net or product_quantization or heads >= 2
        assert sizes[0] == input_dim
        assert sizes[-1] == k_dim // 2 if multi_query_net else heads * k_dim
        assert self.grouped_conv is False or len(sizes) > 2
        assert len(sizes) == len(kernel_sizes) + 1 >= 2 and all(ks % 2 == 1 for ks in kernel_sizes)
        self.groups = 2 * heads if multi_query_net else 1
        if self.grouped_conv:
            self.query_convs = convs(sizes, kernel_sizes, bias=bias, batchnorm=batchnorm, residual=residual, groups=self.groups)
        elif len(self.sizes) == 2:
            sizes_ = list(sizes)
            sizes_[-1] = sizes_[-1] * self.groups
            self.query_convs = convs(sizes_, kernel_sizes, bias=bias, batchnorm=batchnorm, residual=residual, groups=1)
        else:
            self.query_convs = nn.ModuleList([convs(sizes, kernel_sizes, bias=bias, batchnorm=batchnorm, residual=residual, groups=1) for _ in range(self.groups)])

    def forward(self, input):
        bs, nf, h, w = input.shape
        assert nf == self.input_dim
        if self.grouped_conv or len(self.sizes) == 2:
            query = self.query_convs(input)
        else:
            outputs = [m(input) for m in self.query_convs]
            query = torch.cat(outputs, 1) if len(outputs) > 1 else outputs[0]
        assert query.shape == (bs, self.heads * self.k_dim, h, w)
        query = query.transpose(1, 3).contiguous().view(bs * w * h * self.heads, self.k_dim)
        return query


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    return m


class PredLayer(nn.Module):
    """
    Prediction layer (cross_entropy or adaptive_softmax).
    """

    def __init__(self, params):
        super().__init__()
        self.asm = params.asm
        self.n_words = params.n_words
        self.pad_index = params.pad_index
        dim = params.emb_dim
        if params.asm is False:
            self.proj = Linear(dim, params.n_words, bias=True)
        else:
            self.proj = nn.AdaptiveLogSoftmaxWithLoss(in_features=dim, n_classes=params.n_words, cutoffs=params.asm_cutoffs, div_value=params.asm_div_value, head_bias=True)

    def forward(self, x, y, get_scores=False):
        """
        Compute the loss, and optionally the scores.
        """
        assert (y == self.pad_index).sum().item() == 0
        if self.asm is False:
            scores = self.proj(x).view(-1, self.n_words)
            loss = F.cross_entropy(scores, y, reduction='mean')
        else:
            _, loss = self.proj(x, y)
            scores = self.proj.log_prob(x) if get_scores else None
        return scores, loss

    def get_scores(self, x):
        """
        Compute scores.
        """
        assert x.dim() == 2
        return self.proj.log_prob(x) if self.asm else self.proj(x)


class MultiHeadAttention(nn.Module):
    NEW_ID = itertools.count()

    def __init__(self, n_heads, dim, dropout):
        super().__init__()
        self.layer_id = next(MultiHeadAttention.NEW_ID)
        self.dim = dim
        self.n_heads = n_heads
        self.dropout = dropout
        assert self.dim % self.n_heads == 0
        self.q_lin = Linear(dim, dim)
        self.k_lin = Linear(dim, dim)
        self.v_lin = Linear(dim, dim)
        self.out_lin = Linear(dim, dim)

    def forward(self, input, mask, kv=None, cache=None):
        """
        Self-attention (if kv is None) or attention over source sentence (provided by kv).
        """
        bs, qlen, dim = input.size()
        if kv is None:
            klen = qlen if cache is None else cache['slen'] + qlen
        else:
            klen = kv.size(1)
        assert dim == self.dim, 'Dimensions do not match: %s input vs %s configured' % (dim, self.dim)
        n_heads = self.n_heads
        dim_per_head = dim // n_heads
        mask_reshape = (bs, 1, qlen, klen) if mask.dim() == 3 else (bs, 1, 1, klen)

        def shape(x):
            """  projection """
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
        q = shape(self.q_lin(input))
        if kv is None:
            k = shape(self.k_lin(input))
            v = shape(self.v_lin(input))
        elif cache is None or self.layer_id not in cache:
            k = v = kv
            k = shape(self.k_lin(k))
            v = shape(self.v_lin(v))
        if cache is not None:
            if self.layer_id in cache:
                if kv is None:
                    k_, v_ = cache[self.layer_id]
                    k = torch.cat([k_, k], dim=2)
                    v = torch.cat([v_, v], dim=2)
                else:
                    k, v = cache[self.layer_id]
            cache[self.layer_id] = k, v
        q = q / math.sqrt(dim_per_head)
        scores = torch.matmul(q, k.transpose(2, 3))
        mask = (mask == 0).view(mask_reshape).expand_as(scores)
        scores.masked_fill_(mask, -float('inf'))
        weights = F.softmax(scores.float(), dim=-1).type_as(scores)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        context = torch.matmul(weights, v)
        context = unshape(context)
        return self.out_lin(context)


def gelu(x):
    """
    GELU activation
    https://arxiv.org/abs/1606.08415
    https://github.com/huggingface/pytorch-openai-transformer-lm/blob/master/model_pytorch.py#L14
    https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/modeling.py
    """
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


class TransformerFFN(nn.Module):

    def __init__(self, in_dim, dim_hidden, out_dim, dropout, gelu_activation):
        super().__init__()
        self.dropout = dropout
        self.lin1 = Linear(in_dim, dim_hidden)
        self.lin2 = Linear(dim_hidden, out_dim)
        self.act = gelu if gelu_activation else F.relu

    def forward(self, input):
        x = self.lin1(input)
        x = self.act(x)
        x = self.lin2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class BeamHypotheses(object):

    def __init__(self, n_hyp, max_len, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_len = max_len - 1
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.n_hyp = n_hyp
        self.hyp = []
        self.worst_score = 1000000000.0

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.hyp)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty
        if len(self) < self.n_hyp or score > self.worst_score:
            self.hyp.append((score, hyp))
            if len(self) > self.n_hyp:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.hyp)])
                del self.hyp[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """
        if len(self) < self.n_hyp:
            return False
        elif self.early_stopping:
            return True
        else:
            return self.worst_score >= best_sum_logprobs / self.max_len ** self.length_penalty


def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    if padding_idx is not None:
        nn.init.constant_(m.weight[padding_idx], 0)
    return m


N_MAX_POSITIONS = 512


def create_sinusoidal_embeddings(n_pos, dim, out):
    position_enc = np.array([[(pos / np.power(10000, 2 * (j // 2) / dim)) for j in range(dim)] for pos in range(n_pos)])
    out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
    out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
    out.detach_()
    out.requires_grad = False


def get_masks(slen, lengths, causal):
    """
    Generate hidden states mask, and optionally an attention mask.
    """
    assert lengths.max().item() <= slen
    bs = lengths.size(0)
    alen = torch.arange(slen, dtype=torch.long, device=lengths.device)
    mask = alen < lengths[:, (None)]
    if causal:
        attn_mask = alen[(None), (None), :].repeat(bs, slen, 1) <= alen[(None), :, (None)]
    else:
        attn_mask = mask
    assert mask.size() == (bs, slen)
    assert causal is False or attn_mask.size() == (bs, slen, slen)
    return mask, attn_mask


class TransformerModel(nn.Module):
    ATTRIBUTES = ['encoder', 'with_output', 'eos_index', 'pad_index', 'n_langs', 'n_words', 'dim', 'n_layers', 'n_heads', 'hidden_dim', 'dropout', 'attention_dropout', 'asm', 'asm_cutoffs', 'asm_div_value']

    def __init__(self, params, dico, is_encoder, with_output):
        """
        Transformer model (encoder or decoder).
        """
        super().__init__()
        self.is_encoder = is_encoder
        self.is_decoder = not is_encoder
        self.with_output = with_output
        self.n_langs = params.n_langs
        self.n_words = params.n_words
        self.eos_index = params.eos_index
        self.pad_index = params.pad_index
        self.dico = dico
        self.id2lang = params.id2lang
        self.lang2id = params.lang2id
        self.use_lang_emb = getattr(params, 'use_lang_emb', True)
        assert len(self.dico) == self.n_words
        assert len(self.id2lang) == len(self.lang2id) == self.n_langs
        self.dim = params.emb_dim
        self.hidden_dim = self.dim * 4
        self.n_heads = params.n_heads
        self.n_layers = params.n_layers
        self.dropout = params.dropout
        self.attention_dropout = params.attention_dropout
        assert self.dim % self.n_heads == 0, 'transformer dim must be a multiple of n_heads'
        self.position_embeddings = Embedding(N_MAX_POSITIONS, self.dim)
        if params.sinusoidal_embeddings:
            create_sinusoidal_embeddings(N_MAX_POSITIONS, self.dim, out=self.position_embeddings.weight)
        if params.n_langs > 1 and self.use_lang_emb:
            self.lang_embeddings = Embedding(self.n_langs, self.dim)
        self.embeddings = Embedding(self.n_words, self.dim, padding_idx=self.pad_index)
        self.layer_norm_emb = nn.LayerNorm(self.dim, eps=1e-12)
        self.attentions = nn.ModuleList()
        self.layer_norm1 = nn.ModuleList()
        self.ffns = nn.ModuleList()
        self.layer_norm2 = nn.ModuleList()
        if self.is_decoder:
            self.layer_norm15 = nn.ModuleList()
            self.encoder_attn = nn.ModuleList()
        self.memories = nn.ModuleDict()
        if getattr(params, 'use_memory', False):
            mem_positions = params.mem_enc_positions if is_encoder else params.mem_dec_positions
            for layer_id, pos in mem_positions:
                assert 0 <= layer_id <= params.n_layers - 1
                assert pos in ['in', 'after']
                self.memories['%i_%s' % (layer_id, pos)] = HashingMemory.build(self.dim, self.dim, params)
        for layer_id in range(self.n_layers):
            self.attentions.append(MultiHeadAttention(self.n_heads, self.dim, dropout=self.attention_dropout))
            self.layer_norm1.append(nn.LayerNorm(self.dim, eps=1e-12))
            if self.is_decoder:
                self.layer_norm15.append(nn.LayerNorm(self.dim, eps=1e-12))
                self.encoder_attn.append(MultiHeadAttention(self.n_heads, self.dim, dropout=self.attention_dropout))
            if '%i_in' % layer_id in self.memories:
                self.ffns.append(None)
            else:
                self.ffns.append(TransformerFFN(self.dim, self.hidden_dim, self.dim, dropout=self.dropout, gelu_activation=params.gelu_activation))
            self.layer_norm2.append(nn.LayerNorm(self.dim, eps=1e-12))
        if self.with_output:
            self.pred_layer = PredLayer(params)
            if params.share_inout_emb:
                self.pred_layer.proj.weight = self.embeddings.weight

    def forward(self, mode, **kwargs):
        """
        Forward function with different forward modes.
        ### Small hack to handle PyTorch distributed.
        """
        if mode == 'fwd':
            return self.fwd(**kwargs)
        elif mode == 'predict':
            return self.predict(**kwargs)
        else:
            raise Exception('Unknown mode: %s' % mode)

    def fwd(self, x, lengths, causal, src_enc=None, src_len=None, positions=None, langs=None, cache=None):
        """
        Inputs:
            `x` LongTensor(slen, bs), containing word indices
            `lengths` LongTensor(bs), containing the length of each sentence
            `causal` Boolean, if True, the attention is only done over previous hidden states
            `positions` LongTensor(slen, bs), containing word positions
            `langs` LongTensor(slen, bs), containing language IDs
        """
        slen, bs = x.size()
        assert lengths.size(0) == bs
        assert lengths.max().item() <= slen
        x = x.transpose(0, 1)
        assert (src_enc is None) == (src_len is None)
        if src_enc is not None:
            assert self.is_decoder
            assert src_enc.size(0) == bs
        mask, attn_mask = get_masks(slen, lengths, causal)
        if self.is_decoder and src_enc is not None:
            src_mask = torch.arange(src_len.max(), dtype=torch.long, device=lengths.device) < src_len[:, (None)]
        if positions is None:
            positions = x.new(slen).long()
            positions = torch.arange(slen, out=positions).unsqueeze(0)
        else:
            assert positions.size() == (slen, bs)
            positions = positions.transpose(0, 1)
        if langs is not None:
            assert langs.size() == (slen, bs)
            langs = langs.transpose(0, 1)
        if cache is not None:
            _slen = slen - cache['slen']
            x = x[:, -_slen:]
            positions = positions[:, -_slen:]
            if langs is not None:
                langs = langs[:, -_slen:]
            mask = mask[:, -_slen:]
            attn_mask = attn_mask[:, -_slen:]
        tensor = self.embeddings(x)
        tensor = tensor + self.position_embeddings(positions).expand_as(tensor)
        if langs is not None and self.use_lang_emb:
            tensor = tensor + self.lang_embeddings(langs)
        tensor = self.layer_norm_emb(tensor)
        tensor = F.dropout(tensor, p=self.dropout, training=self.training)
        tensor *= mask.unsqueeze(-1)
        for i in range(self.n_layers):
            attn = self.attentions[i](tensor, attn_mask, cache=cache)
            attn = F.dropout(attn, p=self.dropout, training=self.training)
            tensor = tensor + attn
            tensor = self.layer_norm1[i](tensor)
            if self.is_decoder and src_enc is not None:
                attn = self.encoder_attn[i](tensor, src_mask, kv=src_enc, cache=cache)
                attn = F.dropout(attn, p=self.dropout, training=self.training)
                tensor = tensor + attn
                tensor = self.layer_norm15[i](tensor)
            if '%i_in' % i in self.memories:
                tensor = tensor + self.memories['%i_in' % i](tensor)
            else:
                tensor = tensor + self.ffns[i](tensor)
            tensor = self.layer_norm2[i](tensor)
            if '%i_after' % i in self.memories:
                tensor = tensor + self.memories['%i_after' % i](tensor)
            tensor *= mask.unsqueeze(-1)
        if cache is not None:
            cache['slen'] += tensor.size(1)
        tensor = tensor.transpose(0, 1)
        return tensor

    def predict(self, tensor, pred_mask, y, get_scores):
        """
        Given the last hidden state, compute word scores and/or the loss.
            `pred_mask` is a ByteTensor of shape (slen, bs), filled with 1 when
                we need to predict a word
            `y` is a LongTensor of shape (pred_mask.sum(),)
            `get_scores` is a boolean specifying whether we need to return scores
        """
        masked_tensor = tensor[pred_mask.unsqueeze(-1).expand_as(tensor)].view(-1, self.dim)
        scores, loss = self.pred_layer(masked_tensor, y, get_scores)
        return scores, loss

    def generate(self, src_enc, src_len, tgt_lang_id, max_len=200, sample_temperature=None):
        """
        Decode a sentence given initial start.
        `x`:
            - LongTensor(bs, slen)
                <EOS> W1 W2 W3 <EOS> <PAD>
                <EOS> W1 W2 W3   W4  <EOS>
        `lengths`:
            - LongTensor(bs) [5, 6]
        `positions`:
            - False, for regular "arange" positions (LM)
            - True, to reset positions from the new generation (MT)
        `langs`:
            - must be None if the model only supports one language
            - lang_id if only one language is involved (LM)
            - (lang_id1, lang_id2) if two languages are involved (MT)
        """
        bs = len(src_len)
        assert src_enc.size(0) == bs
        generated = src_len.new(max_len, bs)
        generated.fill_(self.pad_index)
        generated[0].fill_(self.eos_index)
        positions = src_len.new(max_len).long()
        positions = torch.arange(max_len, out=positions).unsqueeze(1).expand(max_len, bs)
        langs = src_len.new(max_len).long().fill_(tgt_lang_id)
        langs = langs.unsqueeze(1).expand(max_len, bs)
        cur_len = 1
        gen_len = src_len.clone().fill_(1)
        unfinished_sents = src_len.clone().fill_(1)
        cache = {'slen': 0}
        while cur_len < max_len:
            tensor = self.forward('fwd', x=generated[:cur_len], lengths=gen_len, positions=positions[:cur_len], langs=langs[:cur_len], causal=True, src_enc=src_enc, src_len=src_len, cache=cache)
            assert tensor.size() == (1, bs, self.dim), (cur_len, max_len, src_enc.size(), tensor.size(), (1, bs, self.dim))
            tensor = tensor.data[(-1), :, :].type_as(src_enc)
            scores = self.pred_layer.get_scores(tensor)
            if sample_temperature is None:
                next_words = torch.topk(scores, 1)[1].squeeze(1)
            else:
                next_words = torch.multinomial(F.softmax(scores / sample_temperature, dim=1), 1).squeeze(1)
            assert next_words.size() == (bs,)
            generated[cur_len] = next_words * unfinished_sents + self.pad_index * (1 - unfinished_sents)
            gen_len.add_(unfinished_sents)
            unfinished_sents.mul_(next_words.ne(self.eos_index).long())
            cur_len = cur_len + 1
            if unfinished_sents.max() == 0:
                break
        if cur_len == max_len:
            generated[-1].masked_fill_(unfinished_sents.byte(), self.eos_index)
        assert (generated == self.eos_index).sum() == 2 * bs
        return generated[:cur_len], gen_len

    def generate_beam(self, src_enc, src_len, tgt_lang_id, beam_size, length_penalty, early_stopping, max_len=200):
        """
        Decode a sentence given initial start.
        `x`:
            - LongTensor(bs, slen)
                <EOS> W1 W2 W3 <EOS> <PAD>
                <EOS> W1 W2 W3   W4  <EOS>
        `lengths`:
            - LongTensor(bs) [5, 6]
        `positions`:
            - False, for regular "arange" positions (LM)
            - True, to reset positions from the new generation (MT)
        `langs`:
            - must be None if the model only supports one language
            - lang_id if only one language is involved (LM)
            - (lang_id1, lang_id2) if two languages are involved (MT)
        """
        assert src_enc.size(0) == src_len.size(0)
        assert beam_size >= 1
        bs = len(src_len)
        n_words = self.n_words
        src_enc = src_enc.unsqueeze(1).expand((bs, beam_size) + src_enc.shape[1:]).contiguous().view((bs * beam_size,) + src_enc.shape[1:])
        src_len = src_len.unsqueeze(1).expand(bs, beam_size).contiguous().view(-1)
        generated = src_len.new(max_len, bs * beam_size)
        generated.fill_(self.pad_index)
        generated[0].fill_(self.eos_index)
        generated_hyps = [BeamHypotheses(beam_size, max_len, length_penalty, early_stopping) for _ in range(bs)]
        positions = src_len.new(max_len).long()
        positions = torch.arange(max_len, out=positions).unsqueeze(1).expand_as(generated)
        langs = positions.clone().fill_(tgt_lang_id)
        beam_scores = src_enc.new(bs, beam_size).fill_(0)
        beam_scores[:, 1:] = -1000000000.0
        beam_scores = beam_scores.view(-1)
        cur_len = 1
        cache = {'slen': 0}
        done = [(False) for _ in range(bs)]
        while cur_len < max_len:
            tensor = self.forward('fwd', x=generated[:cur_len], lengths=src_len.new(bs * beam_size).fill_(cur_len), positions=positions[:cur_len], langs=langs[:cur_len], causal=True, src_enc=src_enc, src_len=src_len, cache=cache)
            assert tensor.size() == (1, bs * beam_size, self.dim)
            tensor = tensor.data[(-1), :, :]
            scores = self.pred_layer.get_scores(tensor)
            scores = F.log_softmax(scores, dim=-1)
            assert scores.size() == (bs * beam_size, n_words)
            _scores = scores + beam_scores[:, (None)].expand_as(scores)
            _scores = _scores.view(bs, beam_size * n_words)
            next_scores, next_words = torch.topk(_scores, 2 * beam_size, dim=1, largest=True, sorted=True)
            assert next_scores.size() == next_words.size() == (bs, 2 * beam_size)
            next_batch_beam = []
            for sent_id in range(bs):
                done[sent_id] = done[sent_id] or generated_hyps[sent_id].is_done(next_scores[sent_id].max().item())
                if done[sent_id]:
                    next_batch_beam.extend([(0, self.pad_index, 0)] * beam_size)
                    continue
                next_sent_beam = []
                for idx, value in zip(next_words[sent_id], next_scores[sent_id]):
                    beam_id = idx // n_words
                    word_id = idx % n_words
                    if word_id == self.eos_index or cur_len + 1 == max_len:
                        generated_hyps[sent_id].add(generated[:cur_len, (sent_id * beam_size + beam_id)].clone(), value.item())
                    else:
                        next_sent_beam.append((value, word_id, sent_id * beam_size + beam_id))
                    if len(next_sent_beam) == beam_size:
                        break
                assert len(next_sent_beam) == 0 if cur_len + 1 == max_len else beam_size
                if len(next_sent_beam) == 0:
                    next_sent_beam = [(0, self.pad_index, 0)] * beam_size
                next_batch_beam.extend(next_sent_beam)
                assert len(next_batch_beam) == beam_size * (sent_id + 1)
            assert len(next_batch_beam) == bs * beam_size
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_words = generated.new([x[1] for x in next_batch_beam])
            beam_idx = src_len.new([x[2] for x in next_batch_beam])
            generated = generated[:, (beam_idx)]
            generated[cur_len] = beam_words
            for k in cache.keys():
                if k != 'slen':
                    cache[k] = cache[k][0][beam_idx], cache[k][1][beam_idx]
            cur_len = cur_len + 1
            if all(done):
                break
        tgt_len = src_len.new(bs)
        best = []
        for i, hypotheses in enumerate(generated_hyps):
            best_hyp = max(hypotheses.hyp, key=lambda x: x[0])[1]
            tgt_len[i] = len(best_hyp) + 1
            best.append(best_hyp)
        decoded = src_len.new(tgt_len.max().item(), bs).fill_(self.pad_index)
        for i, hypo in enumerate(best):
            decoded[:tgt_len[i] - 1, (i)] = hypo
            decoded[tgt_len[i] - 1, i] = self.eos_index
        assert (decoded == self.eos_index).sum() == 2 * bs
        return decoded, tgt_len


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (MultiHeadAttention,
     lambda: ([], {'n_heads': 4, 'dim': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 1, 1, 4])], {}),
     False),
    (TransformerFFN,
     lambda: ([], {'in_dim': 4, 'dim_hidden': 4, 'out_dim': 4, 'dropout': 0.5, 'gelu_activation': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_facebookresearch_XLM(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

