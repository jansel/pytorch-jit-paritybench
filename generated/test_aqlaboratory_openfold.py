import sys
_module = sys.modules[__name__]
del sys
openfold = _module
config = _module
data = _module
data_modules = _module
data_pipeline = _module
data_transforms = _module
errors = _module
feature_pipeline = _module
input_pipeline = _module
mmcif_parsing = _module
parsers = _module
templates = _module
tools = _module
hhblits = _module
hhsearch = _module
jackhmmer = _module
kalign = _module
utils = _module
model = _module
dropout = _module
embedders = _module
evoformer = _module
heads = _module
model = _module
msa = _module
outer_product_mean = _module
pair_transition = _module
primitives = _module
structure_module = _module
template = _module
torchscript = _module
triangular_attention = _module
triangular_multiplicative_update = _module
np = _module
protein = _module
relax = _module
amber_minimize = _module
cleanup = _module
residue_constants = _module
resources = _module
argparse = _module
callbacks = _module
checkpointing = _module
chunk_utils = _module
exponential_moving_average = _module
feats = _module
import_weights = _module
kernel = _module
attention_core = _module
logger = _module
loss = _module
lr_schedulers = _module
precision_utils = _module
rigid_utils = _module
script_utils = _module
seed = _module
superimposition = _module
suppress_output = _module
tensor_utils = _module
trace_utils = _module
validation_metrics = _module
run_pretrained_openfold = _module
create_alignment_db = _module
unify_alignment_db_indices = _module
build_deepspeed_config = _module
convert_of_weights_to_jax = _module
data_dir_to_fasta = _module
download_cameo = _module
generate_alphafold_feature_dict = _module
generate_chain_data_cache = _module
generate_mmcif_cache = _module
precompute_alignments = _module
precompute_alignments_mmseqs = _module
prep_proteinnet_msas = _module
unpack_proteinnet = _module
zero_to_fp32 = _module
setup = _module
tests = _module
compare_utils = _module
data_utils = _module
test_data_pipeline = _module
test_data_transforms = _module
test_embedders = _module
test_evoformer = _module
test_feats = _module
test_import_weights = _module
test_kernels = _module
test_loss = _module
test_model = _module
test_msa = _module
test_outer_product_mean = _module
test_pair_transition = _module
test_primitives = _module
test_structure_module = _module
test_template = _module
test_triangular_attention = _module
test_triangular_multiplicative_update = _module
test_utils = _module
thread_sequence = _module
train_openfold = _module

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


import copy


from functools import partial


import logging


from typing import Optional


from typing import Sequence


from typing import List


from typing import Any


import numpy as np


import torch


from torch.utils.data import RandomSampler


import itertools


from functools import reduce


from functools import wraps


from typing import Mapping


from typing import Tuple


from typing import Dict


import torch.nn as nn


from functools import partialmethod


from typing import Union


import math


from typing import Callable


from scipy.stats import truncnorm


import torch.utils.checkpoint


from collections import OrderedDict


from enum import Enum


import time


import torch.cuda.profiler as profiler


from torch.distributions.bernoulli import Bernoulli


from functools import lru_cache


import re


import numpy


import random


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CUDAExtension


from torch.utils.cpp_extension import CUDA_HOME


class Dropout(nn.Module):
    """
    Implementation of dropout with the ability to share the dropout mask
    along a particular dimension.

    If not in training mode, this module computes the identity function.
    """

    def __init__(self, r: float, batch_dim: Union[int, List[int]]):
        """
        Args:
            r:
                Dropout rate
            batch_dim:
                Dimension(s) along which the dropout mask is shared
        """
        super(Dropout, self).__init__()
        self.r = r
        if type(batch_dim) == int:
            batch_dim = [batch_dim]
        self.batch_dim = batch_dim
        self.dropout = nn.Dropout(self.r)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        """
        Args:
            x:
                Tensor to which dropout is applied. Can have any shape
                compatible with self.batch_dim
        """
        shape = list(x.shape)
        if self.batch_dim is not None:
            for bd in self.batch_dim:
                shape[bd] = 1
        mask = x.new_ones(shape)
        mask = self.dropout(mask)
        x *= mask
        return x


class DropoutRowwise(Dropout):
    """
    Convenience class for rowwise dropout as described in subsection
    1.11.6.
    """
    __init__ = partialmethod(Dropout.__init__, batch_dim=-3)


class DropoutColumnwise(Dropout):
    """
    Convenience class for columnwise dropout as described in subsection
    1.11.6.
    """
    __init__ = partialmethod(Dropout.__init__, batch_dim=-2)


def final_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def gating_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def glorot_uniform_init_(weights):
    nn.init.xavier_uniform_(weights, gain=1)


def _calculate_fan(linear_weight_shape, fan='fan_in'):
    fan_out, fan_in = linear_weight_shape
    if fan == 'fan_in':
        f = fan_in
    elif fan == 'fan_out':
        f = fan_out
    elif fan == 'fan_avg':
        f = (fan_in + fan_out) / 2
    else:
        raise ValueError('Invalid fan option')
    return f


def _prod(nums):
    out = 1
    for n in nums:
        out = out * n
    return out


def trunc_normal_init_(weights, scale=1.0, fan='fan_in'):
    shape = weights.shape
    f = _calculate_fan(shape, fan)
    scale = scale / max(1, f)
    a = -2
    b = 2
    std = math.sqrt(scale) / truncnorm.std(a=a, b=b, loc=0, scale=1)
    size = _prod(shape)
    samples = truncnorm.rvs(a=a, b=b, loc=0, scale=std, size=size)
    samples = np.reshape(samples, shape)
    with torch.no_grad():
        weights.copy_(torch.tensor(samples, device=weights.device))


def he_normal_init_(weights):
    trunc_normal_init_(weights, scale=2.0)


def lecun_normal_init_(weights):
    trunc_normal_init_(weights, scale=1.0)


def normal_init_(weights):
    torch.nn.init.kaiming_normal_(weights, nonlinearity='linear')


class Linear(nn.Linear):
    """
    A Linear layer with built-in nonstandard initializations. Called just
    like torch.nn.Linear.

    Implements the initializers in 1.11.4, plus some additional ones found
    in the code.
    """

    def __init__(self, in_dim: int, out_dim: int, bias: bool=True, init: str='default', init_fn: Optional[Callable[[torch.Tensor, torch.Tensor], None]]=None):
        """
        Args:
            in_dim:
                The final dimension of inputs to the layer
            out_dim:
                The final dimension of layer outputs
            bias:
                Whether to learn an additive bias. True by default
            init:
                The initializer to use. Choose from:

                "default": LeCun fan-in truncated normal initialization
                "relu": He initialization w/ truncated normal distribution
                "glorot": Fan-average Glorot uniform initialization
                "gating": Weights=0, Bias=1
                "normal": Normal initialization with std=1/sqrt(fan_in)
                "final": Weights=0, Bias=0

                Overridden by init_fn if the latter is not None.
            init_fn:
                A custom initializer taking weight and bias as inputs.
                Overrides init if not None.
        """
        super(Linear, self).__init__(in_dim, out_dim, bias=bias)
        if bias:
            with torch.no_grad():
                self.bias.fill_(0)
        with torch.no_grad():
            if init_fn is not None:
                init_fn(self.weight, self.bias)
            elif init == 'default':
                lecun_normal_init_(self.weight)
            elif init == 'relu':
                he_normal_init_(self.weight)
            elif init == 'glorot':
                glorot_uniform_init_(self.weight)
            elif init == 'gating':
                gating_init_(self.weight)
                if bias:
                    self.bias.fill_(1.0)
            elif init == 'normal':
                normal_init_(self.weight)
            elif init == 'final':
                final_init_(self.weight)
            else:
                raise ValueError('Invalid init string.')


def add(m1, m2, inplace):
    if not inplace:
        m1 = m1 + m2
    else:
        m1 += m2
    return m1


class InputEmbedder(nn.Module):
    """
    Embeds a subset of the input features.

    Implements Algorithms 3 (InputEmbedder) and 4 (relpos).
    """

    def __init__(self, tf_dim: int, msa_dim: int, c_z: int, c_m: int, relpos_k: int, **kwargs):
        """
        Args:
            tf_dim:
                Final dimension of the target features
            msa_dim:
                Final dimension of the MSA features
            c_z:
                Pair embedding dimension
            c_m:
                MSA embedding dimension
            relpos_k:
                Window size used in relative positional encoding
        """
        super(InputEmbedder, self).__init__()
        self.tf_dim = tf_dim
        self.msa_dim = msa_dim
        self.c_z = c_z
        self.c_m = c_m
        self.linear_tf_z_i = Linear(tf_dim, c_z)
        self.linear_tf_z_j = Linear(tf_dim, c_z)
        self.linear_tf_m = Linear(tf_dim, c_m)
        self.linear_msa_m = Linear(msa_dim, c_m)
        self.relpos_k = relpos_k
        self.no_bins = 2 * relpos_k + 1
        self.linear_relpos = Linear(self.no_bins, c_z)

    def relpos(self, ri: torch.Tensor):
        """
        Computes relative positional encodings

        Implements Algorithm 4.

        Args:
            ri:
                "residue_index" features of shape [*, N]
        """
        d = ri[..., None] - ri[..., None, :]
        boundaries = torch.arange(start=-self.relpos_k, end=self.relpos_k + 1, device=d.device)
        reshaped_bins = boundaries.view((1,) * len(d.shape) + (len(boundaries),))
        d = d[..., None] - reshaped_bins
        d = torch.abs(d)
        d = torch.argmin(d, dim=-1)
        d = nn.functional.one_hot(d, num_classes=len(boundaries)).float()
        d = d
        return self.linear_relpos(d)

    def forward(self, tf: torch.Tensor, ri: torch.Tensor, msa: torch.Tensor, inplace_safe: bool=False) ->Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            tf:
                "target_feat" features of shape [*, N_res, tf_dim]
            ri:
                "residue_index" features of shape [*, N_res]
            msa:
                "msa_feat" features of shape [*, N_clust, N_res, msa_dim]
        Returns:
            msa_emb:
                [*, N_clust, N_res, C_m] MSA embedding
            pair_emb:
                [*, N_res, N_res, C_z] pair embedding

        """
        tf_emb_i = self.linear_tf_z_i(tf)
        tf_emb_j = self.linear_tf_z_j(tf)
        pair_emb = self.relpos(ri.type(tf_emb_i.dtype))
        pair_emb = add(pair_emb, tf_emb_i[..., None, :], inplace=inplace_safe)
        pair_emb = add(pair_emb, tf_emb_j[..., None, :, :], inplace=inplace_safe)
        n_clust = msa.shape[-3]
        tf_m = self.linear_tf_m(tf).unsqueeze(-3).expand((-1,) * len(tf.shape[:-2]) + (n_clust, -1, -1))
        msa_emb = self.linear_msa_m(msa) + tf_m
        return msa_emb, pair_emb


class LayerNorm(nn.Module):

    def __init__(self, c_in, eps=1e-05):
        super(LayerNorm, self).__init__()
        self.c_in = c_in,
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(c_in))
        self.bias = nn.Parameter(torch.zeros(c_in))

    def forward(self, x):
        d = x.dtype
        deepspeed_is_initialized = deepspeed_is_installed and deepspeed.utils.is_initialized()
        if d is torch.bfloat16 and not deepspeed_is_initialized:
            with torch.amp.autocast(enabled=False):
                out = nn.functional.layer_norm(x, self.c_in, self.weight, self.bias, self.eps)
        else:
            out = nn.functional.layer_norm(x, self.c_in, self.weight, self.bias, self.eps)
        return out


class RecyclingEmbedder(nn.Module):
    """
    Embeds the output of an iteration of the model for recycling.

    Implements Algorithm 32.
    """

    def __init__(self, c_m: int, c_z: int, min_bin: float, max_bin: float, no_bins: int, inf: float=100000000.0, **kwargs):
        """
        Args:
            c_m:
                MSA channel dimension
            c_z:
                Pair embedding channel dimension
            min_bin:
                Smallest distogram bin (Angstroms)
            max_bin:
                Largest distogram bin (Angstroms)
            no_bins:
                Number of distogram bins
        """
        super(RecyclingEmbedder, self).__init__()
        self.c_m = c_m
        self.c_z = c_z
        self.min_bin = min_bin
        self.max_bin = max_bin
        self.no_bins = no_bins
        self.inf = inf
        self.linear = Linear(self.no_bins, self.c_z)
        self.layer_norm_m = LayerNorm(self.c_m)
        self.layer_norm_z = LayerNorm(self.c_z)

    def forward(self, m: torch.Tensor, z: torch.Tensor, x: torch.Tensor, inplace_safe: bool=False) ->Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            m:
                First row of the MSA embedding. [*, N_res, C_m]
            z:
                [*, N_res, N_res, C_z] pair embedding
            x:
                [*, N_res, 3] predicted C_beta coordinates
        Returns:
            m:
                [*, N_res, C_m] MSA embedding update
            z:
                [*, N_res, N_res, C_z] pair embedding update
        """
        m_update = self.layer_norm_m(m)
        if inplace_safe:
            m.copy_(m_update)
            m_update = m
        z_update = self.layer_norm_z(z)
        if inplace_safe:
            z.copy_(z_update)
            z_update = z
        bins = torch.linspace(self.min_bin, self.max_bin, self.no_bins, dtype=x.dtype, device=x.device, requires_grad=False)
        squared_bins = bins ** 2
        upper = torch.cat([squared_bins[1:], squared_bins.new_tensor([self.inf])], dim=-1)
        d = torch.sum((x[..., None, :] - x[..., None, :, :]) ** 2, dim=-1, keepdims=True)
        d = ((d > squared_bins) * (d < upper)).type(x.dtype)
        d = self.linear(d)
        z_update = add(z_update, d, inplace_safe)
        return m_update, z_update


class TemplateAngleEmbedder(nn.Module):
    """
    Embeds the "template_angle_feat" feature.

    Implements Algorithm 2, line 7.
    """

    def __init__(self, c_in: int, c_out: int, **kwargs):
        """
        Args:
            c_in:
                Final dimension of "template_angle_feat"
            c_out:
                Output channel dimension
        """
        super(TemplateAngleEmbedder, self).__init__()
        self.c_out = c_out
        self.c_in = c_in
        self.linear_1 = Linear(self.c_in, self.c_out, init='relu')
        self.relu = nn.ReLU()
        self.linear_2 = Linear(self.c_out, self.c_out, init='relu')

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        """
        Args:
            x: [*, N_templ, N_res, c_in] "template_angle_feat" features
        Returns:
            x: [*, N_templ, N_res, C_out] embedding
        """
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        return x


class TemplatePairEmbedder(nn.Module):
    """
    Embeds "template_pair_feat" features.

    Implements Algorithm 2, line 9.
    """

    def __init__(self, c_in: int, c_out: int, **kwargs):
        """
        Args:
            c_in:

            c_out:
                Output channel dimension
        """
        super(TemplatePairEmbedder, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.linear = Linear(self.c_in, self.c_out, init='relu')

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        """
        Args:
            x:
                [*, C_in] input tensor
        Returns:
            [*, C_out] output tensor
        """
        x = self.linear(x)
        return x


class ExtraMSAEmbedder(nn.Module):
    """
    Embeds unclustered MSA sequences.

    Implements Algorithm 2, line 15
    """

    def __init__(self, c_in: int, c_out: int, **kwargs):
        """
        Args:
            c_in:
                Input channel dimension
            c_out:
                Output channel dimension
        """
        super(ExtraMSAEmbedder, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.linear = Linear(self.c_in, self.c_out)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        """
        Args:
            x:
                [*, N_extra_seq, N_res, C_in] "extra_msa_feat" features
        Returns:
            [*, N_extra_seq, N_res, C_out] embedding
        """
        x = self.linear(x)
        return x


@torch.jit.ignore
def _flat_idx_to_idx(flat_idx: int, dims: Tuple[int]) ->Tuple[int]:
    idx = []
    for d in reversed(dims):
        idx.append(flat_idx % d)
        flat_idx = flat_idx // d
    return tuple(reversed(idx))


@torch.jit.ignore
def _get_minimal_slice_set(start: Sequence[int], end: Sequence[int], dims: int, start_edges: Optional[Sequence[bool]]=None, end_edges: Optional[Sequence[bool]]=None) ->Sequence[Tuple[int]]:
    """ 
        Produces an ordered sequence of tensor slices that, when used in
        sequence on a tensor with shape dims, yields tensors that contain every
        leaf in the contiguous range [start, end]. Care is taken to yield a 
        short sequence of slices, and perhaps even the shortest possible (I'm 
        pretty sure it's the latter).
         
        end is INCLUSIVE. 
    """

    def reduce_edge_list(l):
        tally = 1
        for i in range(len(l)):
            reversed_idx = -1 * (i + 1)
            l[reversed_idx] *= tally
            tally = l[reversed_idx]
    if start_edges is None:
        start_edges = [(s == 0) for s in start]
        reduce_edge_list(start_edges)
    if end_edges is None:
        end_edges = [(e == d - 1) for e, d in zip(end, dims)]
        reduce_edge_list(end_edges)
    if len(start) == 0:
        return [tuple()]
    elif len(start) == 1:
        return [(slice(start[0], end[0] + 1),)]
    slices = []
    path = []
    for s, e in zip(start, end):
        if s == e:
            path.append(slice(s, s + 1))
        else:
            break
    path = tuple(path)
    divergence_idx = len(path)
    if divergence_idx == len(dims):
        return [tuple(path)]

    def upper():
        sdi = start[divergence_idx]
        return [(path + (slice(sdi, sdi + 1),) + s) for s in _get_minimal_slice_set(start[divergence_idx + 1:], [(d - 1) for d in dims[divergence_idx + 1:]], dims[divergence_idx + 1:], start_edges=start_edges[divergence_idx + 1:], end_edges=[(1) for _ in end_edges[divergence_idx + 1:]])]

    def lower():
        edi = end[divergence_idx]
        return [(path + (slice(edi, edi + 1),) + s) for s in _get_minimal_slice_set([(0) for _ in start[divergence_idx + 1:]], end[divergence_idx + 1:], dims[divergence_idx + 1:], start_edges=[(1) for _ in start_edges[divergence_idx + 1:]], end_edges=end_edges[divergence_idx + 1:])]
    if start_edges[divergence_idx] and end_edges[divergence_idx]:
        slices.append(path + (slice(start[divergence_idx], end[divergence_idx] + 1),))
    elif start_edges[divergence_idx]:
        slices.append(path + (slice(start[divergence_idx], end[divergence_idx]),))
        slices.extend(lower())
    elif end_edges[divergence_idx]:
        slices.extend(upper())
        slices.append(path + (slice(start[divergence_idx] + 1, end[divergence_idx] + 1),))
    else:
        slices.extend(upper())
        middle_ground = end[divergence_idx] - start[divergence_idx]
        if middle_ground > 1:
            slices.append(path + (slice(start[divergence_idx] + 1, end[divergence_idx]),))
        slices.extend(lower())
    return [tuple(s) for s in slices]


@torch.jit.ignore
def _chunk_slice(t: torch.Tensor, flat_start: int, flat_end: int, no_batch_dims: int) ->torch.Tensor:
    """
        Equivalent to
        
            t.reshape((-1,) + t.shape[no_batch_dims:])[flat_start:flat_end]

        but without the need for the initial reshape call, which can be 
        memory-intensive in certain situations. The only reshape operations
        in this function are performed on sub-tensors that scale with
        (flat_end - flat_start), the chunk size.
    """
    batch_dims = t.shape[:no_batch_dims]
    start_idx = list(_flat_idx_to_idx(flat_start, batch_dims))
    end_idx = list(_flat_idx_to_idx(flat_end - 1, batch_dims))
    slices = _get_minimal_slice_set(start_idx, end_idx, batch_dims)
    sliced_tensors = [t[s] for s in slices]
    return torch.cat([s.view((-1,) + t.shape[no_batch_dims:]) for s in sliced_tensors])


def _fetch_dims(tree):
    shapes = []
    tree_type = type(tree)
    if tree_type is dict:
        for v in tree.values():
            shapes.extend(_fetch_dims(v))
    elif tree_type is list or tree_type is tuple:
        for t in tree:
            shapes.extend(_fetch_dims(t))
    elif tree_type is torch.Tensor:
        shapes.append(tree.shape)
    else:
        raise ValueError('Not supported')
    return shapes


def dict_map(fn, dic, leaf_type):
    new_dict = {}
    for k, v in dic.items():
        if type(v) is dict:
            new_dict[k] = dict_map(fn, v, leaf_type)
        else:
            new_dict[k] = tree_map(fn, v, leaf_type)
    return new_dict


def tree_map(fn, tree, leaf_type):
    if isinstance(tree, dict):
        return dict_map(fn, tree, leaf_type)
    elif isinstance(tree, list):
        return [tree_map(fn, x, leaf_type) for x in tree]
    elif isinstance(tree, tuple):
        return tuple([tree_map(fn, x, leaf_type) for x in tree])
    elif isinstance(tree, leaf_type):
        return fn(tree)
    else:
        None
        raise ValueError('Not supported')


tensor_tree_map = partial(tree_map, leaf_type=torch.Tensor)


def chunk_layer(layer: Callable, inputs: Dict[str, Any], chunk_size: int, no_batch_dims: int, low_mem: bool=False, _out: Any=None, _add_into_out: bool=False) ->Any:
    """
    Implements the "chunking" procedure described in section 1.11.8.

    Layer outputs and inputs are assumed to be simple "pytrees,"
    consisting only of (arbitrarily nested) lists, tuples, and dicts with
    torch.Tensor leaves.

    Args:
        layer:
            The layer to be applied chunk-wise
        inputs:
            A (non-nested) dictionary of keyworded inputs. All leaves must
            be tensors and must share the same batch dimensions.
        chunk_size:
            The number of sub-batches per chunk. If multiple batch
            dimensions are specified, a "sub-batch" is defined as a single
            indexing of all batch dimensions simultaneously (s.t. the
            number of sub-batches is the product of the batch dimensions).
        no_batch_dims:
            How many of the initial dimensions of each input tensor can
            be considered batch dimensions.
        low_mem:
            Avoids flattening potentially large input tensors. Unnecessary
            in most cases, and is ever so slightly slower than the default
            setting.
    Returns:
        The reassembled output of the layer on the inputs.
    """
    if not len(inputs) > 0:
        raise ValueError('Must provide at least one input')
    initial_dims = [shape[:no_batch_dims] for shape in _fetch_dims(inputs)]
    orig_batch_dims = tuple([max(s) for s in zip(*initial_dims)])

    def _prep_inputs(t):
        if not low_mem:
            if not sum(t.shape[:no_batch_dims]) == no_batch_dims:
                t = t.expand(orig_batch_dims + t.shape[no_batch_dims:])
            t = t.reshape(-1, *t.shape[no_batch_dims:])
        else:
            t = t.expand(orig_batch_dims + t.shape[no_batch_dims:])
        return t
    prepped_inputs = tensor_tree_map(_prep_inputs, inputs)
    prepped_outputs = None
    if _out is not None:
        reshape_fn = lambda t: t.view([-1] + list(t.shape[no_batch_dims:]))
        prepped_outputs = tensor_tree_map(reshape_fn, _out)
    flat_batch_dim = 1
    for d in orig_batch_dims:
        flat_batch_dim *= d
    no_chunks = flat_batch_dim // chunk_size + (flat_batch_dim % chunk_size != 0)
    i = 0
    out = prepped_outputs
    for _ in range(no_chunks):
        if not low_mem:
            select_chunk = lambda t: t[i:i + chunk_size] if t.shape[0] != 1 else t
        else:
            select_chunk = partial(_chunk_slice, flat_start=i, flat_end=min(flat_batch_dim, i + chunk_size), no_batch_dims=len(orig_batch_dims))
        chunks = tensor_tree_map(select_chunk, prepped_inputs)
        output_chunk = layer(**chunks)
        if out is None:
            allocate = lambda t: t.new_zeros((flat_batch_dim,) + t.shape[1:])
            out = tensor_tree_map(allocate, output_chunk)
        out_type = type(output_chunk)
        if out_type is dict:

            def assign(d1, d2):
                for k, v in d1.items():
                    if type(v) is dict:
                        assign(v, d2[k])
                    elif _add_into_out:
                        v[i:i + chunk_size] += d2[k]
                    else:
                        v[i:i + chunk_size] = d2[k]
            assign(out, output_chunk)
        elif out_type is tuple:
            for x1, x2 in zip(out, output_chunk):
                if _add_into_out:
                    x1[i:i + chunk_size] += x2
                else:
                    x1[i:i + chunk_size] = x2
        elif out_type is torch.Tensor:
            if _add_into_out:
                out[i:i + chunk_size] += output_chunk
            else:
                out[i:i + chunk_size] = output_chunk
        else:
            raise ValueError('Not supported')
        i += chunk_size
    reshape = lambda t: t.view(orig_batch_dims + t.shape[1:])
    out = tensor_tree_map(reshape, out)
    return out


class MSATransition(nn.Module):
    """
    Feed-forward network applied to MSA activations after attention.

    Implements Algorithm 9
    """

    def __init__(self, c_m, n):
        """
        Args:
            c_m:
                MSA channel dimension
            n:
                Factor multiplied to c_m to obtain the hidden channel
                dimension
        """
        super(MSATransition, self).__init__()
        self.c_m = c_m
        self.n = n
        self.layer_norm = LayerNorm(self.c_m)
        self.linear_1 = Linear(self.c_m, self.n * self.c_m, init='relu')
        self.relu = nn.ReLU()
        self.linear_2 = Linear(self.n * self.c_m, self.c_m, init='final')

    def _transition(self, m, mask):
        m = self.layer_norm(m)
        m = self.linear_1(m)
        m = self.relu(m)
        m = self.linear_2(m) * mask
        return m

    @torch.jit.ignore
    def _chunk(self, m: torch.Tensor, mask: torch.Tensor, chunk_size: int) ->torch.Tensor:
        return chunk_layer(self._transition, {'m': m, 'mask': mask}, chunk_size=chunk_size, no_batch_dims=len(m.shape[:-2]))

    def forward(self, m: torch.Tensor, mask: Optional[torch.Tensor]=None, chunk_size: Optional[int]=None) ->torch.Tensor:
        """
        Args:
            m:
                [*, N_seq, N_res, C_m] MSA activation
            mask:
                [*, N_seq, N_res, C_m] MSA mask
        Returns:
            m:
                [*, N_seq, N_res, C_m] MSA activation update
        """
        if mask is None:
            mask = m.new_ones(m.shape[:-1])
        mask = mask.unsqueeze(-1)
        if chunk_size is not None:
            m = self._chunk(m, mask, chunk_size)
        else:
            m = self._transition(m, mask)
        return m


def is_fp16_enabled():
    fp16_enabled = torch.get_autocast_gpu_dtype() == torch.float16
    fp16_enabled = fp16_enabled and torch.is_autocast_enabled()
    return fp16_enabled


class OuterProductMean(nn.Module):
    """
    Implements Algorithm 10.
    """

    def __init__(self, c_m, c_z, c_hidden, eps=0.001):
        """
        Args:
            c_m:
                MSA embedding channel dimension
            c_z:
                Pair embedding channel dimension
            c_hidden:
                Hidden channel dimension
        """
        super(OuterProductMean, self).__init__()
        self.c_m = c_m
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.eps = eps
        self.layer_norm = nn.LayerNorm(c_m)
        self.linear_1 = Linear(c_m, c_hidden)
        self.linear_2 = Linear(c_m, c_hidden)
        self.linear_out = Linear(c_hidden ** 2, c_z, init='final')

    def _opm(self, a, b):
        outer = torch.einsum('...bac,...dae->...bdce', a, b)
        outer = outer.reshape(outer.shape[:-2] + (-1,))
        outer = self.linear_out(outer)
        return outer

    @torch.jit.ignore
    def _chunk(self, a: torch.Tensor, b: torch.Tensor, chunk_size: int) ->torch.Tensor:
        a_reshape = a.reshape((-1,) + a.shape[-3:])
        b_reshape = b.reshape((-1,) + b.shape[-3:])
        out = []
        for a_prime, b_prime in zip(a_reshape, b_reshape):
            outer = chunk_layer(partial(self._opm, b=b_prime), {'a': a_prime}, chunk_size=chunk_size, no_batch_dims=1)
            out.append(outer)
        if len(out) == 1:
            outer = out[0].unsqueeze(0)
        else:
            outer = torch.stack(out, dim=0)
        outer = outer.reshape(a.shape[:-3] + outer.shape[1:])
        return outer

    def _forward(self, m: torch.Tensor, mask: Optional[torch.Tensor]=None, chunk_size: Optional[int]=None, inplace_safe: bool=False) ->torch.Tensor:
        """
        Args:
            m:
                [*, N_seq, N_res, C_m] MSA embedding
            mask:
                [*, N_seq, N_res] MSA mask
        Returns:
            [*, N_res, N_res, C_z] pair embedding update
        """
        if mask is None:
            mask = m.new_ones(m.shape[:-1])
        ln = self.layer_norm(m)
        mask = mask.unsqueeze(-1)
        a = self.linear_1(ln)
        a = a * mask
        b = self.linear_2(ln)
        b = b * mask
        del ln
        a = a.transpose(-2, -3)
        b = b.transpose(-2, -3)
        if chunk_size is not None:
            outer = self._chunk(a, b, chunk_size)
        else:
            outer = self._opm(a, b)
        norm = torch.einsum('...abc,...adc->...bdc', mask, mask)
        norm = norm + self.eps
        if inplace_safe:
            outer /= norm
        else:
            outer = outer / norm
        return outer

    def forward(self, m: torch.Tensor, mask: Optional[torch.Tensor]=None, chunk_size: Optional[int]=None, inplace_safe: bool=False) ->torch.Tensor:
        if is_fp16_enabled():
            with torch.amp.autocast(enabled=False):
                return self._forward(m.float(), mask, chunk_size, inplace_safe)
        else:
            return self._forward(m, mask, chunk_size, inplace_safe)


class PairTransition(nn.Module):
    """
    Implements Algorithm 15.
    """

    def __init__(self, c_z, n):
        """
        Args:
            c_z:
                Pair transition channel dimension
            n:
                Factor by which c_z is multiplied to obtain hidden channel
                dimension
        """
        super(PairTransition, self).__init__()
        self.c_z = c_z
        self.n = n
        self.layer_norm = LayerNorm(self.c_z)
        self.linear_1 = Linear(self.c_z, self.n * self.c_z, init='relu')
        self.relu = nn.ReLU()
        self.linear_2 = Linear(self.n * self.c_z, c_z, init='final')

    def _transition(self, z, mask):
        z = self.layer_norm(z)
        z = self.linear_1(z)
        z = self.relu(z)
        z = self.linear_2(z)
        z = z * mask
        return z

    @torch.jit.ignore
    def _chunk(self, z: torch.Tensor, mask: torch.Tensor, chunk_size: int) ->torch.Tensor:
        return chunk_layer(self._transition, {'z': z, 'mask': mask}, chunk_size=chunk_size, no_batch_dims=len(z.shape[:-2]))

    def forward(self, z: torch.Tensor, mask: Optional[torch.Tensor]=None, chunk_size: Optional[int]=None) ->torch.Tensor:
        """
        Args:
            z:
                [*, N_res, N_res, C_z] pair embedding
        Returns:
            [*, N_res, N_res, C_z] pair embedding update
        """
        if mask is None:
            mask = z.new_ones(z.shape[:-1])
        mask = mask.unsqueeze(-1)
        if chunk_size is not None:
            z = self._chunk(z, mask, chunk_size)
        else:
            z = self._transition(z=z, mask=mask)
        return z


DEFAULT_LMA_KV_CHUNK_SIZE = 4096


DEFAULT_LMA_Q_CHUNK_SIZE = 1024


def permute_final_dims(tensor: torch.Tensor, inds: List[int]):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [(zero_index + i) for i in inds])


@torch.jit.ignore
def softmax_no_cast(t: torch.Tensor, dim: int=-1) ->torch.Tensor:
    """
        Softmax, but without automatic casting to fp32 when the input is of
        type bfloat16
    """
    d = t.dtype
    deepspeed_is_initialized = deepspeed_is_installed and deepspeed.utils.is_initialized()
    if d is torch.bfloat16 and not deepspeed_is_initialized:
        with torch.amp.autocast(enabled=False):
            s = torch.nn.functional.softmax(t, dim=dim)
    else:
        s = torch.nn.functional.softmax(t, dim=dim)
    return s


def _attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, biases: List[torch.Tensor]) ->torch.Tensor:
    key = permute_final_dims(key, (1, 0))
    a = torch.matmul(query, key)
    for b in biases:
        a += b
    a = softmax_no_cast(a, -1)
    a = torch.matmul(a, value)
    return a


@torch.jit.ignore
def _flash_attn(q, k, v, kv_mask):
    if not fa_is_installed:
        raise ValueError('_flash_attn requires that FlashAttention be installed')
    batch_dims = q.shape[:-3]
    no_heads, n, c = q.shape[-3:]
    dtype = q.dtype
    q = q.half()
    k = k.half()
    v = v.half()
    kv_mask = kv_mask.half()
    q = q.transpose(-2, -3)
    k = k.transpose(-2, -3)
    v = v.transpose(-2, -3)
    q = q.reshape(-1, *q.shape[-3:])
    k = k.reshape(-1, *k.shape[-3:])
    v = v.reshape(-1, *v.shape[-3:])
    batch_size = q.shape[0]
    q = q.reshape(-1, *q.shape[-2:])
    q_max_s = n
    q_cu_seqlens = torch.arange(0, (batch_size + 1) * n, step=n, dtype=torch.int32, device=q.device)
    kv = torch.stack([k, v], dim=-3)
    kv_shape = kv.shape
    kv = kv.reshape(*kv.shape[:-3], -1)
    kv_unpad, _, kv_cu_seqlens, kv_max_s = unpad_input(kv, kv_mask)
    kv_unpad = kv_unpad.reshape(-1, *kv_shape[-3:])
    out = flash_attn_unpadded_kvpacked_func(q, kv_unpad, q_cu_seqlens, kv_cu_seqlens, q_max_s, kv_max_s, dropout_p=0.0, softmax_scale=1.0)
    out = out.reshape(*batch_dims, n, no_heads, c)
    out = out
    return out


def _lma(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, biases: List[torch.Tensor], q_chunk_size: int, kv_chunk_size: int):
    no_q, no_kv = q.shape[-2], k.shape[-2]
    o = q.new_zeros(q.shape)
    for q_s in range(0, no_q, q_chunk_size):
        q_chunk = q[..., q_s:q_s + q_chunk_size, :]
        large_bias_chunks = [b[..., q_s:q_s + q_chunk_size, :] for b in biases]
        maxes = []
        weights = []
        values = []
        for kv_s in range(0, no_kv, kv_chunk_size):
            k_chunk = k[..., kv_s:kv_s + kv_chunk_size, :]
            v_chunk = v[..., kv_s:kv_s + kv_chunk_size, :]
            small_bias_chunks = [b[..., kv_s:kv_s + kv_chunk_size] for b in large_bias_chunks]
            a = torch.einsum('...hqd,...hkd->...hqk', q_chunk, k_chunk)
            for b in small_bias_chunks:
                a += b
            max_a = torch.max(a, dim=-1, keepdim=True)[0]
            exp_a = torch.exp(a - max_a)
            exp_v = torch.einsum('...hvf,...hqv->...hqf', v_chunk, exp_a)
            maxes.append(max_a.detach().squeeze(-1))
            weights.append(torch.sum(exp_a, dim=-1))
            values.append(exp_v)
        chunk_max = torch.stack(maxes, dim=-3)
        chunk_weights = torch.stack(weights, dim=-3)
        chunk_values = torch.stack(values, dim=-4)
        global_max = torch.max(chunk_max, dim=-3, keepdim=True)[0]
        max_diffs = torch.exp(chunk_max - global_max)
        chunk_values = chunk_values * max_diffs.unsqueeze(-1)
        chunk_weights = chunk_weights * max_diffs
        all_values = torch.sum(chunk_values, dim=-4)
        all_weights = torch.sum(chunk_weights.unsqueeze(-1), dim=-4)
        q_chunk_out = all_values / all_weights
        o[..., q_s:q_s + q_chunk_size, :] = q_chunk_out
    return o


SUPPORTED_DTYPES = [torch.float32, torch.bfloat16]


class AttentionCoreFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, bias_1=None, bias_2=None):
        if bias_1 is None and bias_2 is not None:
            raise ValueError('bias_1 must be specified before bias_2')
        if q.dtype not in SUPPORTED_DTYPES:
            raise ValueError('Unsupported datatype')
        q = q.contiguous()
        k = k.contiguous()
        attention_logits = torch.matmul(q, k.transpose(-1, -2))
        if bias_1 is not None:
            attention_logits += bias_1
        if bias_2 is not None:
            attention_logits += bias_2
        attn_core_inplace_cuda.forward_(attention_logits, reduce(mul, attention_logits.shape[:-1]), attention_logits.shape[-1])
        o = torch.matmul(attention_logits, v)
        ctx.bias_1_shape = bias_1.shape if bias_1 is not None else None
        ctx.bias_2_shape = bias_2.shape if bias_2 is not None else None
        ctx.save_for_backward(q, k, v, attention_logits)
        return o

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, attention_logits = ctx.saved_tensors
        grad_q = grad_k = grad_v = grad_bias_1 = grad_bias_2 = None
        grad_v = torch.matmul(attention_logits.transpose(-1, -2), grad_output)
        attn_core_inplace_cuda.backward_(attention_logits, grad_output.contiguous(), v.contiguous(), reduce(mul, attention_logits.shape[:-1]), attention_logits.shape[-1], grad_output.shape[-1])
        if ctx.bias_1_shape is not None:
            grad_bias_1 = torch.sum(attention_logits, dim=tuple(i for i, d in enumerate(ctx.bias_1_shape) if d == 1), keepdim=True)
        if ctx.bias_2_shape is not None:
            grad_bias_2 = torch.sum(attention_logits, dim=tuple(i for i, d in enumerate(ctx.bias_2_shape) if d == 1), keepdim=True)
        grad_q = torch.matmul(attention_logits, k)
        grad_k = torch.matmul(q.transpose(-1, -2), attention_logits).transpose(-1, -2)
        return grad_q, grad_k, grad_v, grad_bias_1, grad_bias_2


attention_core = AttentionCoreFunction.apply


def flatten_final_dims(t: torch.Tensor, no_dims: int):
    return t.reshape(t.shape[:-no_dims] + (-1,))


class Attention(nn.Module):
    """
    Standard multi-head attention using AlphaFold's default layer
    initialization. Allows multiple bias vectors.
    """

    def __init__(self, c_q: int, c_k: int, c_v: int, c_hidden: int, no_heads: int, gating: bool=True):
        """
        Args:
            c_q:
                Input dimension of query data
            c_k:
                Input dimension of key data
            c_v:
                Input dimension of value data
            c_hidden:
                Per-head hidden dimension
            no_heads:
                Number of attention heads
            gating:
                Whether the output should be gated using query data
        """
        super(Attention, self).__init__()
        self.c_q = c_q
        self.c_k = c_k
        self.c_v = c_v
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.gating = gating
        self.linear_q = Linear(self.c_q, self.c_hidden * self.no_heads, bias=False, init='glorot')
        self.linear_k = Linear(self.c_k, self.c_hidden * self.no_heads, bias=False, init='glorot')
        self.linear_v = Linear(self.c_v, self.c_hidden * self.no_heads, bias=False, init='glorot')
        self.linear_o = Linear(self.c_hidden * self.no_heads, self.c_q, init='final')
        self.linear_g = None
        if self.gating:
            self.linear_g = Linear(self.c_q, self.c_hidden * self.no_heads, init='gating')
        self.sigmoid = nn.Sigmoid()

    def _prep_qkv(self, q_x: torch.Tensor, kv_x: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.linear_q(q_x)
        k = self.linear_k(kv_x)
        v = self.linear_v(kv_x)
        q = q.view(q.shape[:-1] + (self.no_heads, -1))
        k = k.view(k.shape[:-1] + (self.no_heads, -1))
        v = v.view(v.shape[:-1] + (self.no_heads, -1))
        q = q.transpose(-2, -3)
        k = k.transpose(-2, -3)
        v = v.transpose(-2, -3)
        q /= math.sqrt(self.c_hidden)
        return q, k, v

    def _wrap_up(self, o: torch.Tensor, q_x: torch.Tensor) ->torch.Tensor:
        if self.linear_g is not None:
            g = self.sigmoid(self.linear_g(q_x))
            g = g.view(g.shape[:-1] + (self.no_heads, -1))
            o = o * g
        o = flatten_final_dims(o, 2)
        o = self.linear_o(o)
        return o

    def forward(self, q_x: torch.Tensor, kv_x: torch.Tensor, biases: Optional[List[torch.Tensor]]=None, use_memory_efficient_kernel: bool=False, use_lma: bool=False, lma_q_chunk_size: int=DEFAULT_LMA_Q_CHUNK_SIZE, lma_kv_chunk_size: int=DEFAULT_LMA_KV_CHUNK_SIZE, use_flash: bool=False, flash_mask: Optional[torch.Tensor]=None) ->torch.Tensor:
        """
        Args:
            q_x:
                [*, Q, C_q] query data
            kv_x:
                [*, K, C_k] key data
            biases:
                List of biases that broadcast to [*, H, Q, K]
            use_memory_efficient_kernel:
                Whether to use a custom memory-efficient attention kernel.
                This should be the default choice for most. If none of the
                "use_<...>" flags are True, a stock PyTorch implementation
                is used instead
            use_lma:
                Whether to use low-memory attention (Staats & Rabe 2021). If
                none of the "use_<...>" flags are True, a stock PyTorch 
                implementation is used instead
            lma_q_chunk_size:
                Query chunk size (for LMA)
            lma_kv_chunk_size:
                Key/Value chunk size (for LMA)
        Returns
            [*, Q, C_q] attention update
        """
        if use_lma and (lma_q_chunk_size is None or lma_kv_chunk_size is None):
            raise ValueError('If use_lma is specified, lma_q_chunk_size and lma_kv_chunk_size must be provided')
        if use_flash and biases is not None:
            raise ValueError('use_flash is incompatible with the bias option. For masking, use flash_mask instead')
        attn_options = [use_memory_efficient_kernel, use_lma, use_flash]
        if sum(attn_options) > 1:
            raise ValueError('Choose at most one alternative attention algorithm')
        if biases is None:
            biases = []
        q, k, v = self._prep_qkv(q_x, kv_x)
        if is_fp16_enabled():
            use_memory_efficient_kernel = False
        if use_memory_efficient_kernel:
            if len(biases) > 2:
                raise ValueError('If use_memory_efficient_kernel is True, you may only provide up to two bias terms')
            o = attention_core(q, k, v, *(biases + [None] * 2)[:2])
            o = o.transpose(-2, -3)
        elif use_lma:
            biases = [b.expand(b.shape[:-2] + (q_x.shape[-2],) + (kv_x.shape[-2],)) for b in biases]
            o = _lma(q, k, v, biases, lma_q_chunk_size, lma_kv_chunk_size)
            o = o.transpose(-2, -3)
        elif use_flash:
            o = _flash_attn(q, k, v, flash_mask)
        else:
            o = _attention(q, k, v, biases)
            o = o.transpose(-2, -3)
        o = self._wrap_up(o, q_x)
        return o


class TriangleAttention(nn.Module):

    def __init__(self, c_in, c_hidden, no_heads, starting=True, inf=1000000000.0):
        """
        Args:
            c_in:
                Input channel dimension
            c_hidden:
                Overall hidden channel dimension (not per-head)
            no_heads:
                Number of attention heads
        """
        super(TriangleAttention, self).__init__()
        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.starting = starting
        self.inf = inf
        self.layer_norm = LayerNorm(self.c_in)
        self.linear = Linear(c_in, self.no_heads, bias=False, init='normal')
        self.mha = Attention(self.c_in, self.c_in, self.c_in, self.c_hidden, self.no_heads)

    @torch.jit.ignore
    def _chunk(self, x: torch.Tensor, biases: List[torch.Tensor], chunk_size: int, use_memory_efficient_kernel: bool=False, use_lma: bool=False, inplace_safe: bool=False) ->torch.Tensor:
        """triangle! triangle!"""
        mha_inputs = {'q_x': x, 'kv_x': x, 'biases': biases}
        return chunk_layer(partial(self.mha, use_memory_efficient_kernel=use_memory_efficient_kernel, use_lma=use_lma), mha_inputs, chunk_size=chunk_size, no_batch_dims=len(x.shape[:-2]), _out=x if inplace_safe else None)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]=None, chunk_size: Optional[int]=None, use_memory_efficient_kernel: bool=False, use_lma: bool=False, inplace_safe: bool=False) ->torch.Tensor:
        """
        Args:
            x:
                [*, I, J, C_in] input tensor (e.g. the pair representation)
        Returns:
            [*, I, J, C_in] output tensor
        """
        if mask is None:
            mask = x.new_ones(x.shape[:-1])
        if not self.starting:
            x = x.transpose(-2, -3)
            mask = mask.transpose(-1, -2)
        x = self.layer_norm(x)
        mask_bias = (self.inf * (mask - 1))[..., :, None, None, :]
        triangle_bias = permute_final_dims(self.linear(x), (2, 0, 1))
        triangle_bias = triangle_bias.unsqueeze(-4)
        biases = [mask_bias, triangle_bias]
        if chunk_size is not None:
            x = self._chunk(x, biases, chunk_size, use_memory_efficient_kernel=use_memory_efficient_kernel, use_lma=use_lma, inplace_safe=inplace_safe)
        else:
            x = self.mha(q_x=x, kv_x=x, biases=biases, use_memory_efficient_kernel=use_memory_efficient_kernel, use_lma=use_lma)
        if not self.starting:
            x = x.transpose(-2, -3)
        return x


class TriangleMultiplicativeUpdate(nn.Module):
    """
    Implements Algorithms 11 and 12.
    """

    def __init__(self, c_z, c_hidden, _outgoing=True):
        """
        Args:
            c_z:
                Input channel dimension
            c:
                Hidden channel dimension
        """
        super(TriangleMultiplicativeUpdate, self).__init__()
        self.c_z = c_z
        self.c_hidden = c_hidden
        self._outgoing = _outgoing
        self.linear_a_p = Linear(self.c_z, self.c_hidden)
        self.linear_a_g = Linear(self.c_z, self.c_hidden, init='gating')
        self.linear_b_p = Linear(self.c_z, self.c_hidden)
        self.linear_b_g = Linear(self.c_z, self.c_hidden, init='gating')
        self.linear_g = Linear(self.c_z, self.c_z, init='gating')
        self.linear_z = Linear(self.c_hidden, self.c_z, init='final')
        self.layer_norm_in = LayerNorm(self.c_z)
        self.layer_norm_out = LayerNorm(self.c_hidden)
        self.sigmoid = nn.Sigmoid()

    def _combine_projections(self, a: torch.Tensor, b: torch.Tensor, _inplace_chunk_size: Optional[int]=None) ->torch.Tensor:
        if self._outgoing:
            a = permute_final_dims(a, (2, 0, 1))
            b = permute_final_dims(b, (2, 1, 0))
        else:
            a = permute_final_dims(a, (2, 1, 0))
            b = permute_final_dims(b, (2, 0, 1))
        if _inplace_chunk_size is not None:
            for i in range(0, a.shape[-3], _inplace_chunk_size):
                a_chunk = a[..., i:i + _inplace_chunk_size, :, :]
                b_chunk = b[..., i:i + _inplace_chunk_size, :, :]
                a[..., i:i + _inplace_chunk_size, :, :] = torch.matmul(a_chunk, b_chunk)
            p = a
        else:
            p = torch.matmul(a, b)
        return permute_final_dims(p, (1, 2, 0))

    def _inference_forward(self, z: torch.Tensor, mask: Optional[torch.Tensor]=None, inplace_chunk_size: Optional[int]=None, with_add: bool=True):
        """
        Args:
            z:
                A [*, N, N, C_z] pair representation
            mask:
                A [*, N, N] pair mask
            inplace_chunk_size:
                Size of chunks used in the main computation. Increase to trade
                memory for speed.
            with_add:
                If True, z is overwritten with (z + update). Otherwise, it is
                overwritten with (update).
        Returns:
            A reference to the overwritten z

        More memory-efficient, inference-only version of the forward function.
        Uses in-place operations, fusion of the addition that happens after
        this module in the Evoformer, a smidge of recomputation, and 
        a cache of overwritten values to lower peak memory consumption of this
        module from 5x the size of the input tensor z to 2.5x its size. Useful
        for inference on extremely long sequences. 
        
        It works as follows. We will make reference to variables used in the
        default forward implementation below. Naively, triangle multiplication
        attention requires the manifestation of 5 tensors the size of z:
        1) z, the "square" input tensor, 2) a, the first projection of z, 
        3) b, the second projection of b, 4) g, a z-sized mask, and 5) a 
        z-sized tensor for intermediate computations. For large N, this is 
        prohibitively expensive; for N=4000, for example, z is more than 8GB 
        alone. To avoid this problem, we compute b, g, and all intermediate
        tensors in small chunks, noting that the chunks required to compute a
        chunk of the output depend only on the tensor a and corresponding 
        vertical and horizontal chunks of z. This suggests an algorithm that 
        loops over pairs of chunks of z: hereafter "columns" and "rows" of
        z, even though each "column" and "row" in fact contains
        inplace_chunk_size contiguous true columns and rows of z. Writing 
        output chunks to a new tensor would bring total memory consumption
        down to 3x the size of z. However, more memory can be saved by writing
        output chunks directly to z in-place. WLOG, we choose to write output
        chunks vertically, overwriting the ith "column" of z at the end of
        the ith iteration of the main loop. Despite this overwriting, the 
        ith column is always one column ahead of previously overwritten columns 
        and can be recovered directly from z. After the first iteration,
        however, the ith row of z is always at least partially overwritten. For
        this reason, we introduce the z-cache, a tensor one-half the size of 
        z. The z-cache initially contains the left half (2nd and 3rd quadrants)
        of z. For 0 < i < N/2, the missing left part of the ith row of z is
        recovered from this cache at the beginning of the ith iteration. Once i 
        exceeds n/2, the cache is "reoriented" to encompass the 3rd and 4th 
        quadrants of z instead. Though the 3rd quadrant of the original z is 
        entirely overwritten at this point, it can be recovered from the z-cache 
        itself. Thereafter, the ith row of z can be recovered in its entirety 
        from the reoriented z-cache. After the final iteration, z has been 
        completely overwritten and contains the triangular multiplicative 
        update. If with_add is True, it instead contains the sum of z and the
        triangular multiplicative update. In either case, peak memory 
        consumption is just 2.5x the size of z, disregarding memory used for 
        chunks and other small variables.
        """
        if mask is None:
            mask = z.new_ones(z.shape[:-1])
        mask = mask.unsqueeze(-1)

        def compute_projection_helper(pair, mask, a=True):
            if a:
                linear_g = self.linear_a_g
                linear_p = self.linear_a_p
            else:
                linear_g = self.linear_b_g
                linear_p = self.linear_b_p
            pair = self.layer_norm_in(pair)
            p = linear_g(pair)
            p.sigmoid_()
            p *= linear_p(pair)
            p *= mask
            p = permute_final_dims(p, (2, 0, 1))
            return p

        def compute_projection(pair, mask, a=True, chunked=True):
            need_transpose = self._outgoing ^ a
            if not chunked:
                p = compute_projection_helper(pair, mask, a)
                if need_transpose:
                    p = p.transpose(-1, -2)
            else:
                linear_g = self.linear_a_g if a else self.linear_b_g
                c = linear_g.bias.shape[-1]
                out_shape = pair.shape[:-3] + (c,) + pair.shape[-3:-1]
                p = pair.new_zeros(out_shape)
                for i in range(0, pair.shape[-3], inplace_chunk_size):
                    pair_chunk = pair[..., i:i + inplace_chunk_size, :, :]
                    mask_chunk = mask[..., i:i + inplace_chunk_size, :, :]
                    pair_chunk = compute_projection_helper(pair[..., i:i + inplace_chunk_size, :, :], mask[..., i:i + inplace_chunk_size, :, :], a)
                    if need_transpose:
                        pair_chunk = pair_chunk.transpose(-1, -2)
                        p[..., i:i + inplace_chunk_size] = pair_chunk
                    else:
                        p[..., i:i + inplace_chunk_size, :] = pair_chunk
                    del pair_chunk
            return p
        a = compute_projection(z, mask, True, chunked=True)
        if inplace_chunk_size is not None:
            n = a.shape[-1]
            half_n = n // 2 + n % 2
            row_dim = -3
            col_dim = -2
            b_chunk_dim = row_dim if self._outgoing else col_dim

            def empty_slicer(t):
                return [slice(None) for _ in t.shape]

            def slice_tensor(t, start, end, dim):
                s = empty_slicer(t)
                s[dim] = slice(start, end)
                return t[s]

            def flip_z_cache_(z_cache, z):
                quadrant_3 = slice_tensor(z_cache, half_n, None, row_dim)
                z_cache = z_cache.transpose(row_dim, col_dim)
                z_cache = z_cache[..., :n // 2, :, :]
                first_half_slicer = empty_slicer(z_cache)
                first_half_slicer[col_dim] = slice(0, half_n)
                z_cache[first_half_slicer] = quadrant_3
                quadrant_4 = slice_tensor(z, half_n, None, row_dim)
                quadrant_4 = slice_tensor(quadrant_4, half_n, None, col_dim)
                quadrant_3_slicer = empty_slicer(z_cache)
                quadrant_3_slicer[col_dim] = slice(half_n, None)
                z_cache[quadrant_3_slicer] = quadrant_4
                return z_cache
            z_cache_shape = list(z.shape)
            z_cache_shape[col_dim] = half_n
            z_cache = z.new_zeros(z_cache_shape)
            z_cache_slicer = empty_slicer(z_cache)
            z_cache_slicer[col_dim] = slice(0, half_n)
            z_cache.copy_(z[z_cache_slicer])
            z_cache_rotated = False
            i_range = list(range(0, half_n, inplace_chunk_size))
            initial_offsets = [(i_2 - i_1) for i_1, i_2 in zip(i_range, i_range[1:] + [half_n])]
            after_half = list(range(half_n, n, inplace_chunk_size))
            after_half_offsets = [inplace_chunk_size for _ in after_half]
            combined_range_with_offsets = zip(i_range + after_half, initial_offsets + after_half_offsets)
            for i, offset in combined_range_with_offsets:
                if not z_cache_rotated and i >= half_n:
                    z_cache = flip_z_cache_(z_cache, z)
                    z_cache_rotated = True
                z_chunk_b = slice_tensor(z, i, i + offset, b_chunk_dim)
                mask_chunk = slice_tensor(mask, i, i + offset, b_chunk_dim)
                z_chunk_b = z_chunk_b.clone()
                if b_chunk_dim == col_dim:
                    z_chunk_b = slice_tensor(z, i, i + offset, col_dim)
                elif not z_cache_rotated:
                    z_chunk_slicer = empty_slicer(z_chunk_b)
                    z_chunk_slicer[col_dim] = slice(0, half_n)
                    z_chunk_b[z_chunk_slicer] = slice_tensor(z_cache, i, i + offset, row_dim)
                else:
                    z_cache_offset = i - half_n
                    z_chunk_b = slice_tensor(z_cache, z_cache_offset, z_cache_offset + offset, row_dim)
                b_chunk = compute_projection(z_chunk_b, mask_chunk, a=False, chunked=False)
                del z_chunk_b
                x_chunk = torch.matmul(a, b_chunk)
                x_chunk = permute_final_dims(x_chunk, (1, 2, 0))
                x_chunk = self.layer_norm_out(x_chunk)
                x_chunk = self.linear_z(x_chunk)
                z_chunk_g = slice_tensor(z, i, i + offset, col_dim)
                g_chunk = self.linear_g(self.layer_norm_in(z_chunk_g))
                g_chunk.sigmoid_()
                del z_chunk_g
                x_chunk *= g_chunk
                z_slicer = empty_slicer(z)
                z_slicer[col_dim] = slice(i, i + offset)
                if with_add:
                    z[z_slicer] += x_chunk
                else:
                    z[z_slicer] = x_chunk
        else:
            b = compute_projection(z, mask, False, False)
            x = torch.matmul(a, b)
            x = self.layer_norm_out(x)
            x = self.linear_z(x)
            g = self.linear_g(z)
            g.sigmoid_()
            x *= g
            if with_add:
                z += x
            else:
                z = x
        return z

    def forward(self, z: torch.Tensor, mask: Optional[torch.Tensor]=None, inplace_safe: bool=False, _add_with_inplace: bool=False, _inplace_chunk_size: Optional[int]=256) ->torch.Tensor:
        """
        Args:
            x:
                [*, N_res, N_res, C_z] input tensor
            mask:
                [*, N_res, N_res] input mask
        Returns:
            [*, N_res, N_res, C_z] output tensor
        """
        if inplace_safe:
            x = self._inference_forward(z, mask, inplace_chunk_size=_inplace_chunk_size, with_add=_add_with_inplace)
            return x
        if mask is None:
            mask = z.new_ones(z.shape[:-1])
        mask = mask.unsqueeze(-1)
        z = self.layer_norm_in(z)
        a = mask
        a = a * self.sigmoid(self.linear_a_g(z))
        a = a * self.linear_a_p(z)
        b = mask
        b = b * self.sigmoid(self.linear_b_g(z))
        b = b * self.linear_b_p(z)
        if is_fp16_enabled():
            with torch.amp.autocast(enabled=False):
                x = self._combine_projections(a.float(), b.float())
        else:
            x = self._combine_projections(a, b)
        del a, b
        x = self.layer_norm_out(x)
        x = self.linear_z(x)
        g = self.sigmoid(self.linear_g(z))
        x = x * g
        return x


class TriangleMultiplicationIncoming(TriangleMultiplicativeUpdate):
    """
    Implements Algorithm 12.
    """
    __init__ = partialmethod(TriangleMultiplicativeUpdate.__init__, _outgoing=False)


class TriangleMultiplicationOutgoing(TriangleMultiplicativeUpdate):
    """
    Implements Algorithm 11.
    """
    __init__ = partialmethod(TriangleMultiplicativeUpdate.__init__, _outgoing=True)


class EvoformerBlockCore(nn.Module):

    def __init__(self, c_m: int, c_z: int, c_hidden_opm: int, c_hidden_mul: int, c_hidden_pair_att: int, no_heads_msa: int, no_heads_pair: int, transition_n: int, pair_dropout: float, inf: float, eps: float, _is_extra_msa_stack: bool=False):
        super(EvoformerBlockCore, self).__init__()
        self.msa_transition = MSATransition(c_m=c_m, n=transition_n)
        self.outer_product_mean = OuterProductMean(c_m, c_z, c_hidden_opm)
        self.tri_mul_out = TriangleMultiplicationOutgoing(c_z, c_hidden_mul)
        self.tri_mul_in = TriangleMultiplicationIncoming(c_z, c_hidden_mul)
        self.tri_att_start = TriangleAttention(c_z, c_hidden_pair_att, no_heads_pair, inf=inf)
        self.tri_att_end = TriangleAttention(c_z, c_hidden_pair_att, no_heads_pair, inf=inf)
        self.pair_transition = PairTransition(c_z, transition_n)
        self.ps_dropout_row_layer = DropoutRowwise(pair_dropout)

    def forward(self, input_tensors: Sequence[torch.Tensor], msa_mask: torch.Tensor, pair_mask: torch.Tensor, chunk_size: Optional[int]=None, use_lma: bool=False, inplace_safe: bool=False, _mask_trans: bool=True, _attn_chunk_size: Optional[int]=None, _offload_inference: bool=False) ->Tuple[torch.Tensor, torch.Tensor]:
        msa_trans_mask = msa_mask if _mask_trans else None
        pair_trans_mask = pair_mask if _mask_trans else None
        if _attn_chunk_size is None:
            _attn_chunk_size = chunk_size
        m, z = input_tensors
        m = add(m, self.msa_transition(m, mask=msa_trans_mask, chunk_size=chunk_size), inplace=inplace_safe)
        if _offload_inference and inplace_safe:
            del m, z
            assert sys.getrefcount(input_tensors[1]) == 2
            input_tensors[1] = input_tensors[1].cpu()
            torch.cuda.empty_cache()
            m, z = input_tensors
        opm = self.outer_product_mean(m, mask=msa_mask, chunk_size=chunk_size, inplace_safe=inplace_safe)
        if _offload_inference and inplace_safe:
            del m, z
            assert sys.getrefcount(input_tensors[0]) == 2
            input_tensors[0] = input_tensors[0].cpu()
            input_tensors[1] = input_tensors[1]
            m, z = input_tensors
        z = add(z, opm, inplace=inplace_safe)
        del opm
        tmu_update = self.tri_mul_out(z, mask=pair_mask, inplace_safe=inplace_safe, _add_with_inplace=True)
        if not inplace_safe:
            z = z + self.ps_dropout_row_layer(tmu_update)
        else:
            z = tmu_update
        del tmu_update
        tmu_update = self.tri_mul_in(z, mask=pair_mask, inplace_safe=inplace_safe, _add_with_inplace=True)
        if not inplace_safe:
            z = z + self.ps_dropout_row_layer(tmu_update)
        else:
            z = tmu_update
        del tmu_update
        z = add(z, self.ps_dropout_row_layer(self.tri_att_start(z, mask=pair_mask, chunk_size=_attn_chunk_size, use_memory_efficient_kernel=False, use_lma=use_lma, inplace_safe=inplace_safe)), inplace=inplace_safe)
        z = z.transpose(-2, -3)
        if inplace_safe:
            input_tensors[1] = z.contiguous()
            z = input_tensors[1]
        z = add(z, self.ps_dropout_row_layer(self.tri_att_end(z, mask=pair_mask.transpose(-1, -2), chunk_size=_attn_chunk_size, use_memory_efficient_kernel=False, use_lma=use_lma, inplace_safe=inplace_safe)), inplace=inplace_safe)
        z = z.transpose(-2, -3)
        if inplace_safe:
            input_tensors[1] = z.contiguous()
            z = input_tensors[1]
        z = add(z, self.pair_transition(z, mask=pair_trans_mask, chunk_size=chunk_size), inplace=inplace_safe)
        if _offload_inference and inplace_safe:
            device = z.device
            del m, z
            assert sys.getrefcount(input_tensors[0]) == 2
            assert sys.getrefcount(input_tensors[1]) == 2
            input_tensors[0] = input_tensors[0]
            input_tensors[1] = input_tensors[1]
            m, z = input_tensors
        return m, z


def get_checkpoint_fn():
    deepspeed_is_configured = deepspeed_is_installed and deepspeed.checkpointing.is_configured()
    if deepspeed_is_configured:
        checkpoint = deepspeed.checkpointing.checkpoint
    else:
        checkpoint = torch.utils.checkpoint.checkpoint
    return checkpoint


@torch.jit.ignore
def _attention_chunked_trainable(query, key, value, biases, chunk_size, chunk_dim, checkpoint):
    if checkpoint and len(biases) > 2:
        raise ValueError('Checkpointed version permits only permits two bias terms')

    def _checkpointable_attention(q, k, v, b1, b2):
        bs = [b for b in [b1, b2] if b is not None]
        a = _attention(q, k, v, bs)
        return a
    o_chunks = []
    checkpoint_fn = get_checkpoint_fn()
    count = query.shape[chunk_dim]
    for start in range(0, count, chunk_size):
        end = start + chunk_size
        idx = [slice(None)] * len(query.shape)
        idx[chunk_dim] = slice(start, end)
        idx_tup = tuple(idx)
        q_chunk = query[idx_tup]
        k_chunk = key[idx_tup]
        v_chunk = value[idx_tup]

        def _slice_bias(b):
            idx[chunk_dim] = slice(start, end) if b.shape[chunk_dim] != 1 else slice(None)
            return b[tuple(idx)]
        if checkpoint:
            bias_1_chunk, bias_2_chunk = [(_slice_bias(b) if b is not None else None) for b in (biases + [None, None])[:2]]
            o_chunk = checkpoint_fn(_checkpointable_attention, q_chunk, k_chunk, v_chunk, bias_1_chunk, bias_2_chunk)
        else:
            bias_chunks = [_slice_bias(b) for b in biases]
            o_chunk = _attention(q_chunk, k_chunk, v_chunk, bias_chunks)
        o_chunk = o_chunk.transpose(-2, -3)
        o_chunks.append(o_chunk)
    o = torch.cat(o_chunks, dim=chunk_dim)
    return o


class MSAAttention(nn.Module):

    def __init__(self, c_in, c_hidden, no_heads, pair_bias=False, c_z=None, inf=1000000000.0):
        """
        Args:
            c_in:
                Input channel dimension
            c_hidden:
                Per-head hidden channel dimension
            no_heads:
                Number of attention heads
            pair_bias:
                Whether to use pair embedding bias
            c_z:
                Pair embedding channel dimension. Ignored unless pair_bias
                is true
            inf:
                A large number to be used in computing the attention mask
        """
        super(MSAAttention, self).__init__()
        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.pair_bias = pair_bias
        self.c_z = c_z
        self.inf = inf
        self.layer_norm_m = LayerNorm(self.c_in)
        self.layer_norm_z = None
        self.linear_z = None
        if self.pair_bias:
            self.layer_norm_z = LayerNorm(self.c_z)
            self.linear_z = Linear(self.c_z, self.no_heads, bias=False, init='normal')
        self.mha = Attention(self.c_in, self.c_in, self.c_in, self.c_hidden, self.no_heads)

    @torch.jit.ignore
    def _chunk(self, m: torch.Tensor, biases: Optional[List[torch.Tensor]], chunk_size: int, use_memory_efficient_kernel: bool, use_lma: bool, use_flash: bool, flash_mask: Optional[torch.Tensor]) ->torch.Tensor:

        def fn(m, biases, flash_mask):
            m = self.layer_norm_m(m)
            return self.mha(q_x=m, kv_x=m, biases=biases, use_memory_efficient_kernel=use_memory_efficient_kernel, use_lma=use_lma, use_flash=use_flash, flash_mask=flash_mask)
        inputs = {'m': m}
        if biases is not None:
            inputs['biases'] = biases
        else:
            fn = partial(fn, biases=None)
        if use_flash and flash_mask is not None:
            inputs['flash_mask'] = flash_mask
        else:
            fn = partial(fn, flash_mask=None)
        return chunk_layer(fn, inputs, chunk_size=chunk_size, no_batch_dims=len(m.shape[:-2]))

    def _prep_inputs(self, m: torch.Tensor, z: Optional[torch.Tensor], mask: Optional[torch.Tensor], inplace_safe: bool=False) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_seq, n_res = m.shape[-3:-1]
        if mask is None:
            mask = m.new_ones(m.shape[:-3] + (n_seq, n_res))
        mask_bias = (self.inf * (mask - 1))[..., :, None, None, :]
        if self.pair_bias and z is not None and self.layer_norm_z is not None and self.linear_z is not None:
            chunks = []
            for i in range(0, z.shape[-3], 256):
                z_chunk = z[..., i:i + 256, :, :]
                z_chunk = self.layer_norm_z(z_chunk)
                z_chunk = self.linear_z(z_chunk)
                chunks.append(z_chunk)
            z = torch.cat(chunks, dim=-3)
            z = permute_final_dims(z, (2, 0, 1)).unsqueeze(-4)
        return m, mask_bias, z

    @torch.jit.ignore
    def _chunked_msa_attn(self, m: torch.Tensor, z: Optional[torch.Tensor], mask: Optional[torch.Tensor], chunk_logits: int, checkpoint: bool, inplace_safe: bool=False) ->torch.Tensor:
        """ 
        MSA attention with training-time chunking of the softmax computation.
        Saves memory in the extra MSA stack. Probably obviated by our fused 
        attention kernel, which is now used by default.
        """
        MSA_DIM = -4

        def _get_qkv(m, z):
            m, mask_bias, z = self._prep_inputs(m, z, mask, inplace_safe=inplace_safe)
            m = self.layer_norm_m(m)
            q, k, v = self.mha._prep_qkv(m, m)
            return m, q, k, v, mask_bias, z
        checkpoint_fn = get_checkpoint_fn()
        if torch.is_grad_enabled() and checkpoint:
            m, q, k, v, mask_bias, z = checkpoint_fn(_get_qkv, m, z)
        else:
            m, q, k, v, mask_bias, z = _get_qkv(m, z)
        o = _attention_chunked_trainable(query=q, key=k, value=v, biases=[mask_bias, z], chunk_size=chunk_logits, chunk_dim=MSA_DIM, checkpoint=checkpoint)
        if torch.is_grad_enabled() and checkpoint:
            m = checkpoint_fn(self.mha._wrap_up, o, m)
        else:
            m = self.mha._wrap_up(o, m)
        return m

    def forward(self, m: torch.Tensor, z: Optional[torch.Tensor]=None, mask: Optional[torch.Tensor]=None, chunk_size: Optional[int]=None, use_memory_efficient_kernel: bool=False, use_lma: bool=False, use_flash: bool=False, inplace_safe: bool=False, _chunk_logits: Optional[int]=None, _checkpoint_chunks: Optional[bool]=None) ->torch.Tensor:
        """
        Args:
            m:
                [*, N_seq, N_res, C_m] MSA embedding
            z:
                [*, N_res, N_res, C_z] pair embedding. Required only if
                pair_bias is True
            mask:
                [*, N_seq, N_res] MSA mask
            chunk_size:
                Size of chunks into which the inputs are split along their
                batch dimensions. A low value decreases memory overhead at the 
                cost of slower execution. Chunking is not performed by default.
                
        """
        if _chunk_logits is not None:
            return self._chunked_msa_attn(m=m, z=z, mask=mask, chunk_logits=_chunk_logits, checkpoint=_checkpoint_chunks, inplace_safe=inplace_safe)
        if use_flash:
            assert z is None
            biases = None
        else:
            m, mask_bias, z = self._prep_inputs(m, z, mask, inplace_safe=inplace_safe)
            biases = [mask_bias]
            if z is not None:
                biases.append(z)
        if chunk_size is not None:
            m = self._chunk(m, biases, chunk_size, use_memory_efficient_kernel=use_memory_efficient_kernel, use_lma=use_lma, use_flash=use_flash, flash_mask=mask)
        else:
            m = self.layer_norm_m(m)
            m = self.mha(q_x=m, kv_x=m, biases=biases, use_memory_efficient_kernel=use_memory_efficient_kernel, use_lma=use_lma, use_flash=use_flash, flash_mask=mask)
        return m


class MSAColumnAttention(nn.Module):
    """
    Implements Algorithm 8.

    By rights, this should also be a subclass of MSAAttention. Alas,
    most inheritance isn't supported by TorchScript.
    """

    def __init__(self, c_m, c_hidden, no_heads, inf=1000000000.0):
        """
        Args:
            c_m:
                MSA channel dimension
            c_hidden:
                Per-head hidden channel dimension
            no_heads:
                Number of attention heads
            inf:
                Large number used to construct attention masks
        """
        super(MSAColumnAttention, self).__init__()
        self.c_m = c_m
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.inf = inf
        self._msa_att = MSAAttention(c_in=c_m, c_hidden=c_hidden, no_heads=no_heads, pair_bias=False, c_z=None, inf=inf)

    def forward(self, m: torch.Tensor, mask: Optional[torch.Tensor]=None, chunk_size: Optional[int]=None, use_lma: bool=False, use_flash: bool=False) ->torch.Tensor:
        """
        Args:
            m:
                [*, N_seq, N_res, C_m] MSA embedding
            mask:
                [*, N_seq, N_res] MSA mask
            chunk_size:
                Size of chunks into which the inputs are split along their
                batch dimensions. A low value decreases memory overhead at the 
                cost of slower execution. Chunking is not performed by default.
        """
        m = m.transpose(-2, -3)
        if mask is not None:
            mask = mask.transpose(-1, -2)
        m = self._msa_att(m, mask=mask, chunk_size=chunk_size, use_lma=use_lma, use_flash=use_flash)
        m = m.transpose(-2, -3)
        if mask is not None:
            mask = mask.transpose(-1, -2)
        return m


class MSARowAttentionWithPairBias(MSAAttention):
    """
    Implements Algorithm 7.
    """

    def __init__(self, c_m, c_z, c_hidden, no_heads, inf=1000000000.0):
        """
        Args:
            c_m:
                Input channel dimension
            c_z:
                Pair embedding channel dimension
            c_hidden:
                Per-head hidden channel dimension
            no_heads:
                Number of attention heads
            inf:
                Large number used to construct attention masks
        """
        super(MSARowAttentionWithPairBias, self).__init__(c_m, c_hidden, no_heads, pair_bias=True, c_z=c_z, inf=inf)


class EvoformerBlock(nn.Module):

    def __init__(self, c_m: int, c_z: int, c_hidden_msa_att: int, c_hidden_opm: int, c_hidden_mul: int, c_hidden_pair_att: int, no_heads_msa: int, no_heads_pair: int, transition_n: int, msa_dropout: float, pair_dropout: float, inf: float, eps: float):
        super(EvoformerBlock, self).__init__()
        self.msa_att_row = MSARowAttentionWithPairBias(c_m=c_m, c_z=c_z, c_hidden=c_hidden_msa_att, no_heads=no_heads_msa, inf=inf)
        self.msa_att_col = MSAColumnAttention(c_m, c_hidden_msa_att, no_heads_msa, inf=inf)
        self.msa_dropout_layer = DropoutRowwise(msa_dropout)
        self.core = EvoformerBlockCore(c_m=c_m, c_z=c_z, c_hidden_opm=c_hidden_opm, c_hidden_mul=c_hidden_mul, c_hidden_pair_att=c_hidden_pair_att, no_heads_msa=no_heads_msa, no_heads_pair=no_heads_pair, transition_n=transition_n, pair_dropout=pair_dropout, inf=inf, eps=eps)

    def forward(self, m: Optional[torch.Tensor], z: Optional[torch.Tensor], msa_mask: torch.Tensor, pair_mask: torch.Tensor, chunk_size: Optional[int]=None, use_lma: bool=False, use_flash: bool=False, inplace_safe: bool=False, _mask_trans: bool=True, _attn_chunk_size: Optional[int]=None, _offload_inference: bool=False, _offloadable_inputs: Optional[Sequence[torch.Tensor]]=None) ->Tuple[torch.Tensor, torch.Tensor]:
        if _attn_chunk_size is None:
            _attn_chunk_size = chunk_size
        if _offload_inference and inplace_safe:
            input_tensors = _offloadable_inputs
            del _offloadable_inputs
        else:
            input_tensors = [m, z]
        m, z = input_tensors
        m = add(m, self.msa_dropout_layer(self.msa_att_row(m, z=z, mask=msa_mask, chunk_size=_attn_chunk_size, use_memory_efficient_kernel=False, use_lma=use_lma)), inplace=inplace_safe)
        m = add(m, self.msa_att_col(m, mask=msa_mask, chunk_size=chunk_size, use_lma=use_lma, use_flash=use_flash), inplace=inplace_safe)
        if not inplace_safe:
            input_tensors = [m, input_tensors[1]]
        del m, z
        m, z = self.core(input_tensors, msa_mask=msa_mask, pair_mask=pair_mask, chunk_size=chunk_size, use_lma=use_lma, inplace_safe=inplace_safe, _mask_trans=_mask_trans, _attn_chunk_size=_attn_chunk_size, _offload_inference=_offload_inference)
        return m, z


class GlobalAttention(nn.Module):

    def __init__(self, c_in, c_hidden, no_heads, inf, eps):
        super(GlobalAttention, self).__init__()
        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.inf = inf
        self.eps = eps
        self.linear_q = Linear(c_in, c_hidden * no_heads, bias=False, init='glorot')
        self.linear_k = Linear(c_in, c_hidden, bias=False, init='glorot')
        self.linear_v = Linear(c_in, c_hidden, bias=False, init='glorot')
        self.linear_g = Linear(c_in, c_hidden * no_heads, init='gating')
        self.linear_o = Linear(c_hidden * no_heads, c_in, init='final')
        self.sigmoid = nn.Sigmoid()

    def forward(self, m: torch.Tensor, mask: torch.Tensor, use_lma: bool=False) ->torch.Tensor:
        q = torch.sum(m * mask.unsqueeze(-1), dim=-2) / (torch.sum(mask, dim=-1)[..., None] + self.eps)
        q = self.linear_q(q)
        q *= self.c_hidden ** -0.5
        q = q.view(q.shape[:-1] + (self.no_heads, -1))
        k = self.linear_k(m)
        v = self.linear_v(m)
        bias = (self.inf * (mask - 1))[..., :, None, :]
        if not use_lma:
            a = torch.matmul(q, k.transpose(-1, -2))
            a += bias
            a = softmax_no_cast(a)
            o = torch.matmul(a, v)
        else:
            o = _lma(q, k, v, [bias], DEFAULT_LMA_Q_CHUNK_SIZE, DEFAULT_LMA_KV_CHUNK_SIZE)
        g = self.sigmoid(self.linear_g(m))
        g = g.view(g.shape[:-1] + (self.no_heads, -1))
        o = o.unsqueeze(-3) * g
        o = o.reshape(o.shape[:-2] + (-1,))
        m = self.linear_o(o)
        return m


class MSAColumnGlobalAttention(nn.Module):

    def __init__(self, c_in, c_hidden, no_heads, inf=1000000000.0, eps=1e-10):
        super(MSAColumnGlobalAttention, self).__init__()
        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.inf = inf
        self.eps = eps
        self.layer_norm_m = nn.LayerNorm(c_in)
        self.global_attention = GlobalAttention(c_in=c_in, c_hidden=c_hidden, no_heads=no_heads, inf=inf, eps=eps)

    @torch.jit.ignore
    def _chunk(self, m: torch.Tensor, mask: torch.Tensor, chunk_size: int, use_lma: bool=False) ->torch.Tensor:
        mha_input = {'m': m, 'mask': mask}

        def fn(m, mask):
            m = self.layer_norm_m(m)
            return self.global_attention(m, mask, use_lma=use_lma)
        return chunk_layer(fn, mha_input, chunk_size=chunk_size, no_batch_dims=len(m.shape[:-2]))

    def forward(self, m: torch.Tensor, mask: Optional[torch.Tensor]=None, chunk_size: Optional[int]=None, use_lma: bool=False) ->torch.Tensor:
        n_seq, n_res, c_in = m.shape[-3:]
        if mask is None:
            mask = torch.ones(m.shape[:-1], dtype=m.dtype, device=m.device).detach()
        m = m.transpose(-2, -3)
        mask = mask.transpose(-1, -2)
        if chunk_size is not None:
            m = self._chunk(m, mask, chunk_size, use_lma=use_lma)
        else:
            m = self.layer_norm_m(m)
            m = self.global_attention(m=m, mask=mask, use_lma=use_lma)
        m = m.transpose(-2, -3)
        return m


class ExtraMSABlock(nn.Module):
    """ 
        Almost identical to the standard EvoformerBlock, except in that the
        ExtraMSABlock uses GlobalAttention for MSA column attention and
        requires more fine-grained control over checkpointing. Separated from
        its twin to preserve the TorchScript-ability of the latter.
    """

    def __init__(self, c_m: int, c_z: int, c_hidden_msa_att: int, c_hidden_opm: int, c_hidden_mul: int, c_hidden_pair_att: int, no_heads_msa: int, no_heads_pair: int, transition_n: int, msa_dropout: float, pair_dropout: float, inf: float, eps: float, ckpt: bool):
        super(ExtraMSABlock, self).__init__()
        self.ckpt = ckpt
        self.msa_att_row = MSARowAttentionWithPairBias(c_m=c_m, c_z=c_z, c_hidden=c_hidden_msa_att, no_heads=no_heads_msa, inf=inf)
        self.msa_att_col = MSAColumnGlobalAttention(c_in=c_m, c_hidden=c_hidden_msa_att, no_heads=no_heads_msa, inf=inf, eps=eps)
        self.msa_dropout_layer = DropoutRowwise(msa_dropout)
        self.core = EvoformerBlockCore(c_m=c_m, c_z=c_z, c_hidden_opm=c_hidden_opm, c_hidden_mul=c_hidden_mul, c_hidden_pair_att=c_hidden_pair_att, no_heads_msa=no_heads_msa, no_heads_pair=no_heads_pair, transition_n=transition_n, pair_dropout=pair_dropout, inf=inf, eps=eps)

    def forward(self, m: Optional[torch.Tensor], z: Optional[torch.Tensor], msa_mask: torch.Tensor, pair_mask: torch.Tensor, chunk_size: Optional[int]=None, use_lma: bool=False, inplace_safe: bool=False, _mask_trans: bool=True, _attn_chunk_size: Optional[int]=None, _offload_inference: bool=False, _offloadable_inputs: Optional[Sequence[torch.Tensor]]=None) ->Tuple[torch.Tensor, torch.Tensor]:
        if _attn_chunk_size is None:
            _attn_chunk_size = chunk_size
        if _offload_inference and inplace_safe:
            input_tensors = _offloadable_inputs
            del _offloadable_inputs
        else:
            input_tensors = [m, z]
        m, z = input_tensors
        m = add(m, self.msa_dropout_layer(self.msa_att_row(m.clone() if torch.is_grad_enabled() else m, z=z.clone() if torch.is_grad_enabled() else z, mask=msa_mask, chunk_size=_attn_chunk_size, use_lma=use_lma, use_memory_efficient_kernel=not use_lma, _checkpoint_chunks=self.ckpt if torch.is_grad_enabled() else False)), inplace=inplace_safe)
        if not inplace_safe:
            input_tensors = [m, z]
        del m, z

        def fn(input_tensors):
            m = add(input_tensors[0], self.msa_att_col(input_tensors[0], mask=msa_mask, chunk_size=chunk_size, use_lma=use_lma), inplace=inplace_safe)
            if not inplace_safe:
                input_tensors = [m, input_tensors[1]]
            del m
            m, z = self.core(input_tensors, msa_mask=msa_mask, pair_mask=pair_mask, chunk_size=chunk_size, use_lma=use_lma, inplace_safe=inplace_safe, _mask_trans=_mask_trans, _attn_chunk_size=_attn_chunk_size, _offload_inference=_offload_inference)
            return m, z
        if torch.is_grad_enabled() and self.ckpt:
            checkpoint_fn = get_checkpoint_fn()
            m, z = checkpoint_fn(fn, input_tensors)
        else:
            m, z = fn(input_tensors)
        return m, z


class ChunkSizeTuner:

    def __init__(self, max_chunk_size=512):
        self.max_chunk_size = max_chunk_size
        self.cached_chunk_size = None
        self.cached_arg_data = None

    def _determine_favorable_chunk_size(self, fn, args, min_chunk_size):
        logging.info('Tuning chunk size...')
        if min_chunk_size >= self.max_chunk_size:
            return min_chunk_size
        candidates = [(2 ** l) for l in range(int(math.log(self.max_chunk_size, 2)) + 1)]
        candidates = [c for c in candidates if c > min_chunk_size]
        candidates = [min_chunk_size] + candidates
        candidates[-1] += 4

        def test_chunk_size(chunk_size):
            try:
                with torch.no_grad():
                    fn(*args, chunk_size=chunk_size)
                return True
            except RuntimeError:
                return False
        min_viable_chunk_size_index = 0
        i = len(candidates) - 1
        while i > min_viable_chunk_size_index:
            viable = test_chunk_size(candidates[i])
            if not viable:
                i = (min_viable_chunk_size_index + i) // 2
            else:
                min_viable_chunk_size_index = i
                i = (i + len(candidates) - 1) // 2
        return candidates[min_viable_chunk_size_index]

    def _compare_arg_caches(self, ac1, ac2):
        consistent = True
        for a1, a2 in zip(ac1, ac2):
            assert type(ac1) == type(ac2)
            if type(ac1) is list or type(ac1) is tuple:
                consistent &= self._compare_arg_caches(a1, a2)
            elif type(ac1) is dict:
                a1_items = [v for _, v in sorted(a1.items(), key=lambda x: x[0])]
                a2_items = [v for _, v in sorted(a2.items(), key=lambda x: x[0])]
                consistent &= self._compare_arg_caches(a1_items, a2_items)
            else:
                consistent &= a1 == a2
        return consistent

    def tune_chunk_size(self, representative_fn: Callable, args: Tuple[Any], min_chunk_size: int) ->int:
        consistent = True
        remove_tensors = lambda a: a.shape if type(a) is torch.Tensor else a
        arg_data = tree_map(remove_tensors, args, object)
        if self.cached_arg_data is not None:
            assert len(self.cached_arg_data) == len(arg_data)
            consistent = self._compare_arg_caches(self.cached_arg_data, arg_data)
        else:
            consistent = False
        if not consistent:
            self.cached_chunk_size = self._determine_favorable_chunk_size(representative_fn, args, min_chunk_size)
            self.cached_arg_data = arg_data
        return self.cached_chunk_size


BLOCK_ARG = Any


BLOCK_ARGS = List[BLOCK_ARG]


@torch.jit.ignore
def checkpoint_blocks(blocks: List[Callable], args: BLOCK_ARGS, blocks_per_ckpt: Optional[int]) ->BLOCK_ARGS:
    """
    Chunk a list of blocks and run each chunk with activation
    checkpointing. We define a "block" as a callable whose only inputs are
    the outputs of the previous block.

    Implements Subsection 1.11.8

    Args:
        blocks:
            List of blocks
        args:
            Tuple of arguments for the first block.
        blocks_per_ckpt:
            Size of each chunk. A higher value corresponds to fewer 
            checkpoints, and trades memory for speed. If None, no checkpointing 
            is performed.
    Returns:
        The output of the final block
    """

    def wrap(a):
        return (a,) if type(a) is not tuple else a

    def exec(b, a):
        for block in b:
            a = wrap(block(*a))
        return a

    def chunker(s, e):

        def exec_sliced(*a):
            return exec(blocks[s:e], a)
        return exec_sliced
    args = wrap(args)
    if blocks_per_ckpt is None or not torch.is_grad_enabled():
        return exec(blocks, args)
    elif blocks_per_ckpt < 1 or blocks_per_ckpt > len(blocks):
        raise ValueError('blocks_per_ckpt must be between 1 and len(blocks)')
    checkpoint = get_checkpoint_fn()
    for s in range(0, len(blocks), blocks_per_ckpt):
        e = s + blocks_per_ckpt
        args = checkpoint(chunker(s, e), *args)
        args = wrap(args)
    return args


class EvoformerStack(nn.Module):
    """
    Main Evoformer trunk.

    Implements Algorithm 6.
    """

    def __init__(self, c_m: int, c_z: int, c_hidden_msa_att: int, c_hidden_opm: int, c_hidden_mul: int, c_hidden_pair_att: int, c_s: int, no_heads_msa: int, no_heads_pair: int, no_blocks: int, transition_n: int, msa_dropout: float, pair_dropout: float, blocks_per_ckpt: int, inf: float, eps: float, clear_cache_between_blocks: bool=False, tune_chunk_size: bool=False, **kwargs):
        """
        Args:
            c_m:
                MSA channel dimension
            c_z:
                Pair channel dimension
            c_hidden_msa_att:
                Hidden dimension in MSA attention
            c_hidden_opm:
                Hidden dimension in outer product mean module
            c_hidden_mul:
                Hidden dimension in multiplicative updates
            c_hidden_pair_att:
                Hidden dimension in triangular attention
            c_s:
                Channel dimension of the output "single" embedding
            no_heads_msa:
                Number of heads used for MSA attention
            no_heads_pair:
                Number of heads used for pair attention
            no_blocks:
                Number of Evoformer blocks in the stack
            transition_n:
                Factor by which to multiply c_m to obtain the MSATransition
                hidden dimension
            msa_dropout:
                Dropout rate for MSA activations
            pair_dropout:
                Dropout used for pair activations
            blocks_per_ckpt:
                Number of Evoformer blocks in each activation checkpoint
            clear_cache_between_blocks:
                Whether to clear CUDA's GPU memory cache between blocks of the
                stack. Slows down each block but can reduce fragmentation
            tune_chunk_size:
                Whether to dynamically tune the module's chunk size
        """
        super(EvoformerStack, self).__init__()
        self.blocks_per_ckpt = blocks_per_ckpt
        self.clear_cache_between_blocks = clear_cache_between_blocks
        self.blocks = nn.ModuleList()
        for _ in range(no_blocks):
            block = EvoformerBlock(c_m=c_m, c_z=c_z, c_hidden_msa_att=c_hidden_msa_att, c_hidden_opm=c_hidden_opm, c_hidden_mul=c_hidden_mul, c_hidden_pair_att=c_hidden_pair_att, no_heads_msa=no_heads_msa, no_heads_pair=no_heads_pair, transition_n=transition_n, msa_dropout=msa_dropout, pair_dropout=pair_dropout, inf=inf, eps=eps)
            self.blocks.append(block)
        self.linear = Linear(c_m, c_s)
        self.tune_chunk_size = tune_chunk_size
        self.chunk_size_tuner = None
        if tune_chunk_size:
            self.chunk_size_tuner = ChunkSizeTuner()

    def _prep_blocks(self, m: torch.Tensor, z: torch.Tensor, chunk_size: int, use_lma: bool, use_flash: bool, msa_mask: Optional[torch.Tensor], pair_mask: Optional[torch.Tensor], inplace_safe: bool, _mask_trans: bool):
        blocks = [partial(b, msa_mask=msa_mask, pair_mask=pair_mask, chunk_size=chunk_size, use_lma=use_lma, use_flash=use_flash, inplace_safe=inplace_safe, _mask_trans=_mask_trans) for b in self.blocks]
        if self.clear_cache_between_blocks:

            def block_with_cache_clear(block, *args, **kwargs):
                torch.cuda.empty_cache()
                return block(*args, **kwargs)
            blocks = [partial(block_with_cache_clear, b) for b in blocks]
        if chunk_size is not None and self.chunk_size_tuner is not None:
            assert not self.training
            tuned_chunk_size = self.chunk_size_tuner.tune_chunk_size(representative_fn=blocks[0], args=(m.clone(), z.clone()), min_chunk_size=chunk_size)
            blocks = [partial(b, chunk_size=tuned_chunk_size, _attn_chunk_size=max(chunk_size, tuned_chunk_size // 4)) for b in blocks]
        return blocks

    def _forward_offload(self, input_tensors: Sequence[torch.Tensor], msa_mask: torch.Tensor, pair_mask: torch.Tensor, chunk_size: int, use_lma: bool=False, use_flash: bool=False, _mask_trans: bool=True) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert not (self.training or torch.is_grad_enabled())
        blocks = self._prep_blocks(m=input_tensors[0], z=input_tensors[1], chunk_size=chunk_size, use_lma=use_lma, use_flash=use_flash, msa_mask=msa_mask, pair_mask=pair_mask, inplace_safe=True, _mask_trans=_mask_trans)
        for b in blocks:
            m, z = b(None, None, _offload_inference=True, _offloadable_inputs=input_tensors)
            input_tensors[0] = m
            input_tensors[1] = z
            del m, z
        m, z = input_tensors
        s = self.linear(m[..., 0, :, :])
        return m, z, s

    def forward(self, m: torch.Tensor, z: torch.Tensor, msa_mask: torch.Tensor, pair_mask: torch.Tensor, chunk_size: int, use_lma: bool=False, use_flash: bool=False, inplace_safe: bool=False, _mask_trans: bool=True) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            m:
                [*, N_seq, N_res, C_m] MSA embedding
            z:
                [*, N_res, N_res, C_z] pair embedding
            msa_mask:
                [*, N_seq, N_res] MSA mask
            pair_mask:
                [*, N_res, N_res] pair mask
            chunk_size: 
                Inference-time subbatch size. Acts as a minimum if 
                self.tune_chunk_size is True
            use_lma: Whether to use low-memory attention during inference
            use_flash: 
                Whether to use FlashAttention where possible. Mutually 
                exclusive with use_lma.
        Returns:
            m:
                [*, N_seq, N_res, C_m] MSA embedding
            z:
                [*, N_res, N_res, C_z] pair embedding
            s:
                [*, N_res, C_s] single embedding (or None if extra MSA stack)
        """
        blocks = self._prep_blocks(m=m, z=z, chunk_size=chunk_size, use_lma=use_lma, use_flash=use_flash, msa_mask=msa_mask, pair_mask=pair_mask, inplace_safe=inplace_safe, _mask_trans=_mask_trans)
        blocks_per_ckpt = self.blocks_per_ckpt
        if not torch.is_grad_enabled():
            blocks_per_ckpt = None
        m, z = checkpoint_blocks(blocks, args=(m, z), blocks_per_ckpt=blocks_per_ckpt)
        s = self.linear(m[..., 0, :, :])
        return m, z, s


class ExtraMSAStack(nn.Module):
    """
    Implements Algorithm 18.
    """

    def __init__(self, c_m: int, c_z: int, c_hidden_msa_att: int, c_hidden_opm: int, c_hidden_mul: int, c_hidden_pair_att: int, no_heads_msa: int, no_heads_pair: int, no_blocks: int, transition_n: int, msa_dropout: float, pair_dropout: float, inf: float, eps: float, ckpt: bool, clear_cache_between_blocks: bool=False, tune_chunk_size: bool=False, **kwargs):
        super(ExtraMSAStack, self).__init__()
        self.ckpt = ckpt
        self.clear_cache_between_blocks = clear_cache_between_blocks
        self.blocks = nn.ModuleList()
        for _ in range(no_blocks):
            block = ExtraMSABlock(c_m=c_m, c_z=c_z, c_hidden_msa_att=c_hidden_msa_att, c_hidden_opm=c_hidden_opm, c_hidden_mul=c_hidden_mul, c_hidden_pair_att=c_hidden_pair_att, no_heads_msa=no_heads_msa, no_heads_pair=no_heads_pair, transition_n=transition_n, msa_dropout=msa_dropout, pair_dropout=pair_dropout, inf=inf, eps=eps, ckpt=False)
            self.blocks.append(block)
        self.tune_chunk_size = tune_chunk_size
        self.chunk_size_tuner = None
        if tune_chunk_size:
            self.chunk_size_tuner = ChunkSizeTuner()

    def _prep_blocks(self, m: torch.Tensor, z: torch.Tensor, chunk_size: int, use_lma: bool, msa_mask: Optional[torch.Tensor], pair_mask: Optional[torch.Tensor], inplace_safe: bool, _mask_trans: bool):
        blocks = [partial(b, msa_mask=msa_mask, pair_mask=pair_mask, chunk_size=chunk_size, use_lma=use_lma, inplace_safe=inplace_safe, _mask_trans=_mask_trans) for b in self.blocks]

        def clear_cache(b, *args, **kwargs):
            torch.cuda.empty_cache()
            return b(*args, **kwargs)
        if self.clear_cache_between_blocks:
            blocks = [partial(clear_cache, b) for b in blocks]
        if chunk_size is not None and self.chunk_size_tuner is not None:
            tuned_chunk_size = self.chunk_size_tuner.tune_chunk_size(representative_fn=blocks[0], args=(m.clone(), z.clone()), min_chunk_size=chunk_size)
            blocks = [partial(b, chunk_size=tuned_chunk_size, _attn_chunk_size=max(chunk_size, tuned_chunk_size // 4)) for b in blocks]
        return blocks

    def _forward_offload(self, input_tensors: Sequence[torch.Tensor], chunk_size: int, use_lma: bool=False, msa_mask: Optional[torch.Tensor]=None, pair_mask: Optional[torch.Tensor]=None, _mask_trans: bool=True) ->torch.Tensor:
        assert not (self.training or torch.is_grad_enabled())
        blocks = self._prep_blocks(m=input_tensors[0], z=input_tensors[1], chunk_size=chunk_size, use_lma=use_lma, msa_mask=msa_mask, pair_mask=pair_mask, inplace_safe=True, _mask_trans=_mask_trans)
        for b in blocks:
            m, z = b(None, None, _offload_inference=True, _offloadable_inputs=input_tensors)
            input_tensors[0] = m
            input_tensors[1] = z
            del m, z
        return input_tensors[1]

    def forward(self, m: torch.Tensor, z: torch.Tensor, msa_mask: Optional[torch.Tensor], pair_mask: Optional[torch.Tensor], chunk_size: int, use_lma: bool=False, inplace_safe: bool=False, _mask_trans: bool=True) ->torch.Tensor:
        """
        Args:
            m:
                [*, N_extra, N_res, C_m] extra MSA embedding
            z:
                [*, N_res, N_res, C_z] pair embedding
            chunk_size: Inference-time subbatch size for Evoformer modules
            use_lma: Whether to use low-memory attention during inference
            msa_mask:
                Optional [*, N_extra, N_res] MSA mask
            pair_mask:
                Optional [*, N_res, N_res] pair mask
        Returns:
            [*, N_res, N_res, C_z] pair update
        """
        checkpoint_fn = get_checkpoint_fn()
        blocks = self._prep_blocks(m=m, z=z, chunk_size=chunk_size, use_lma=use_lma, msa_mask=msa_mask, pair_mask=pair_mask, inplace_safe=inplace_safe, _mask_trans=_mask_trans)
        for b in blocks:
            if self.ckpt and torch.is_grad_enabled():
                m, z = checkpoint_fn(b, m, z)
            else:
                m, z = b(m, z)
        return z


class DistogramHead(nn.Module):
    """
    Computes a distogram probability distribution.

    For use in computation of distogram loss, subsection 1.9.8
    """

    def __init__(self, c_z, no_bins, **kwargs):
        """
        Args:
            c_z:
                Input channel dimension
            no_bins:
                Number of distogram bins
        """
        super(DistogramHead, self).__init__()
        self.c_z = c_z
        self.no_bins = no_bins
        self.linear = Linear(self.c_z, self.no_bins, init='final')

    def _forward(self, z):
        """
        Args:
            z:
                [*, N_res, N_res, C_z] pair embedding
        Returns:
            [*, N, N, no_bins] distogram probability distribution
        """
        logits = self.linear(z)
        logits = logits + logits.transpose(-2, -3)
        return logits

    def forward(self, z):
        if is_fp16_enabled():
            with torch.amp.autocast(enabled=False):
                return self._forward(z.float())
        else:
            return self._forward(z)


class ExperimentallyResolvedHead(nn.Module):
    """
    For use in computation of "experimentally resolved" loss, subsection
    1.9.10
    """

    def __init__(self, c_s, c_out, **kwargs):
        """
        Args:
            c_s:
                Input channel dimension
            c_out:
                Number of distogram bins
        """
        super(ExperimentallyResolvedHead, self).__init__()
        self.c_s = c_s
        self.c_out = c_out
        self.linear = Linear(self.c_s, self.c_out, init='final')

    def forward(self, s):
        """
        Args:
            s:
                [*, N_res, C_s] single embedding
        Returns:
            [*, N, C_out] logits
        """
        logits = self.linear(s)
        return logits


class MaskedMSAHead(nn.Module):
    """
    For use in computation of masked MSA loss, subsection 1.9.9
    """

    def __init__(self, c_m, c_out, **kwargs):
        """
        Args:
            c_m:
                MSA channel dimension
            c_out:
                Output channel dimension
        """
        super(MaskedMSAHead, self).__init__()
        self.c_m = c_m
        self.c_out = c_out
        self.linear = Linear(self.c_m, self.c_out, init='final')

    def forward(self, m):
        """
        Args:
            m:
                [*, N_seq, N_res, C_m] MSA embedding
        Returns:
            [*, N_seq, N_res, C_out] reconstruction
        """
        logits = self.linear(m)
        return logits


class PerResidueLDDTCaPredictor(nn.Module):

    def __init__(self, no_bins, c_in, c_hidden):
        super(PerResidueLDDTCaPredictor, self).__init__()
        self.no_bins = no_bins
        self.c_in = c_in
        self.c_hidden = c_hidden
        self.layer_norm = LayerNorm(self.c_in)
        self.linear_1 = Linear(self.c_in, self.c_hidden, init='relu')
        self.linear_2 = Linear(self.c_hidden, self.c_hidden, init='relu')
        self.linear_3 = Linear(self.c_hidden, self.no_bins, init='final')
        self.relu = nn.ReLU()

    def forward(self, s):
        s = self.layer_norm(s)
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = self.relu(s)
        s = self.linear_3(s)
        return s


class TMScoreHead(nn.Module):
    """
    For use in computation of TM-score, subsection 1.9.7
    """

    def __init__(self, c_z, no_bins, **kwargs):
        """
        Args:
            c_z:
                Input channel dimension
            no_bins:
                Number of bins
        """
        super(TMScoreHead, self).__init__()
        self.c_z = c_z
        self.no_bins = no_bins
        self.linear = Linear(self.c_z, self.no_bins, init='final')

    def forward(self, z):
        """
        Args:
            z:
                [*, N_res, N_res, C_z] pairwise embedding
        Returns:
            [*, N_res, N_res, no_bins] prediction
        """
        logits = self.linear(z)
        return logits


def compute_plddt(logits: torch.Tensor) ->torch.Tensor:
    num_bins = logits.shape[-1]
    bin_width = 1.0 / num_bins
    bounds = torch.arange(start=0.5 * bin_width, end=1.0, step=bin_width, device=logits.device)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    pred_lddt_ca = torch.sum(probs * bounds.view(*((1,) * len(probs.shape[:-1])), *bounds.shape), dim=-1)
    return pred_lddt_ca * 100


def _calculate_bin_centers(boundaries: torch.Tensor):
    step = boundaries[1] - boundaries[0]
    bin_centers = boundaries + step / 2
    bin_centers = torch.cat([bin_centers, (bin_centers[-1] + step).unsqueeze(-1)], dim=0)
    return bin_centers


def _calculate_expected_aligned_error(alignment_confidence_breaks: torch.Tensor, aligned_distance_error_probs: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor]:
    bin_centers = _calculate_bin_centers(alignment_confidence_breaks)
    return torch.sum(aligned_distance_error_probs * bin_centers, dim=-1), bin_centers[-1]


def compute_predicted_aligned_error(logits: torch.Tensor, max_bin: int=31, no_bins: int=64, **kwargs) ->Dict[str, torch.Tensor]:
    """Computes aligned confidence metrics from logits.

    Args:
      logits: [*, num_res, num_res, num_bins] the logits output from
        PredictedAlignedErrorHead.
      max_bin: Maximum bin value
      no_bins: Number of bins
    Returns:
      aligned_confidence_probs: [*, num_res, num_res, num_bins] the predicted
        aligned error probabilities over bins for each residue pair.
      predicted_aligned_error: [*, num_res, num_res] the expected aligned distance
        error for each pair of residues.
      max_predicted_aligned_error: [*] the maximum predicted error possible.
    """
    boundaries = torch.linspace(0, max_bin, steps=no_bins - 1, device=logits.device)
    aligned_confidence_probs = torch.nn.functional.softmax(logits, dim=-1)
    predicted_aligned_error, max_predicted_aligned_error = _calculate_expected_aligned_error(alignment_confidence_breaks=boundaries, aligned_distance_error_probs=aligned_confidence_probs)
    return {'aligned_confidence_probs': aligned_confidence_probs, 'predicted_aligned_error': predicted_aligned_error, 'max_predicted_aligned_error': max_predicted_aligned_error}


def compute_tm(logits: torch.Tensor, residue_weights: Optional[torch.Tensor]=None, max_bin: int=31, no_bins: int=64, eps: float=1e-08, **kwargs) ->torch.Tensor:
    if residue_weights is None:
        residue_weights = logits.new_ones(logits.shape[-2])
    boundaries = torch.linspace(0, max_bin, steps=no_bins - 1, device=logits.device)
    bin_centers = _calculate_bin_centers(boundaries)
    torch.sum(residue_weights)
    n = logits.shape[-2]
    clipped_n = max(n, 19)
    d0 = 1.24 * (clipped_n - 15) ** (1.0 / 3) - 1.8
    probs = torch.nn.functional.softmax(logits, dim=-1)
    tm_per_bin = 1.0 / (1 + bin_centers ** 2 / d0 ** 2)
    predicted_tm_term = torch.sum(probs * tm_per_bin, dim=-1)
    normed_residue_mask = residue_weights / (eps + residue_weights.sum())
    per_alignment = torch.sum(predicted_tm_term * normed_residue_mask, dim=-1)
    weighted = per_alignment * residue_weights
    argmax = (weighted == torch.max(weighted)).nonzero()[0]
    return per_alignment[tuple(argmax)]


class AuxiliaryHeads(nn.Module):

    def __init__(self, config):
        super(AuxiliaryHeads, self).__init__()
        self.plddt = PerResidueLDDTCaPredictor(**config['lddt'])
        self.distogram = DistogramHead(**config['distogram'])
        self.masked_msa = MaskedMSAHead(**config['masked_msa'])
        self.experimentally_resolved = ExperimentallyResolvedHead(**config['experimentally_resolved'])
        if config.tm.enabled:
            self.tm = TMScoreHead(**config.tm)
        self.config = config

    def forward(self, outputs):
        aux_out = {}
        lddt_logits = self.plddt(outputs['sm']['single'])
        aux_out['lddt_logits'] = lddt_logits
        aux_out['plddt'] = compute_plddt(lddt_logits)
        distogram_logits = self.distogram(outputs['pair'])
        aux_out['distogram_logits'] = distogram_logits
        masked_msa_logits = self.masked_msa(outputs['msa'])
        aux_out['masked_msa_logits'] = masked_msa_logits
        experimentally_resolved_logits = self.experimentally_resolved(outputs['single'])
        aux_out['experimentally_resolved_logits'] = experimentally_resolved_logits
        if self.config.tm.enabled:
            tm_logits = self.tm(outputs['pair'])
            aux_out['tm_logits'] = tm_logits
            aux_out['predicted_tm_score'] = compute_tm(tm_logits, **self.config.tm)
            aux_out.update(compute_predicted_aligned_error(tm_logits, **self.config.tm))
        return aux_out


class AngleResnetBlock(nn.Module):

    def __init__(self, c_hidden):
        """
        Args:
            c_hidden:
                Hidden channel dimension
        """
        super(AngleResnetBlock, self).__init__()
        self.c_hidden = c_hidden
        self.linear_1 = Linear(self.c_hidden, self.c_hidden, init='relu')
        self.linear_2 = Linear(self.c_hidden, self.c_hidden, init='final')
        self.relu = nn.ReLU()

    def forward(self, a: torch.Tensor) ->torch.Tensor:
        s_initial = a
        a = self.relu(a)
        a = self.linear_1(a)
        a = self.relu(a)
        a = self.linear_2(a)
        return a + s_initial


class AngleResnet(nn.Module):
    """
    Implements Algorithm 20, lines 11-14
    """

    def __init__(self, c_in, c_hidden, no_blocks, no_angles, epsilon):
        """
        Args:
            c_in:
                Input channel dimension
            c_hidden:
                Hidden channel dimension
            no_blocks:
                Number of resnet blocks
            no_angles:
                Number of torsion angles to generate
            epsilon:
                Small constant for normalization
        """
        super(AngleResnet, self).__init__()
        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_blocks = no_blocks
        self.no_angles = no_angles
        self.eps = epsilon
        self.linear_in = Linear(self.c_in, self.c_hidden)
        self.linear_initial = Linear(self.c_in, self.c_hidden)
        self.layers = nn.ModuleList()
        for _ in range(self.no_blocks):
            layer = AngleResnetBlock(c_hidden=self.c_hidden)
            self.layers.append(layer)
        self.linear_out = Linear(self.c_hidden, self.no_angles * 2)
        self.relu = nn.ReLU()

    def forward(self, s: torch.Tensor, s_initial: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            s:
                [*, C_hidden] single embedding
            s_initial:
                [*, C_hidden] single embedding as of the start of the
                StructureModule
        Returns:
            [*, no_angles, 2] predicted angles
        """
        s_initial = self.relu(s_initial)
        s_initial = self.linear_initial(s_initial)
        s = self.relu(s)
        s = self.linear_in(s)
        s = s + s_initial
        for l in self.layers:
            s = l(s)
        s = self.relu(s)
        s = self.linear_out(s)
        s = s.view(s.shape[:-1] + (-1, 2))
        unnormalized_s = s
        norm_denom = torch.sqrt(torch.clamp(torch.sum(s ** 2, dim=-1, keepdim=True), min=self.eps))
        s = s / norm_denom
        return unnormalized_s, s


class BackboneUpdate(nn.Module):
    """
    Implements part of Algorithm 23.
    """

    def __init__(self, c_s):
        """
        Args:
            c_s:
                Single representation channel dimension
        """
        super(BackboneUpdate, self).__init__()
        self.c_s = c_s
        self.linear = Linear(self.c_s, 6, init='final')

    def forward(self, s: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            [*, N_res, C_s] single representation
        Returns:
            [*, N_res, 6] update vector 
        """
        update = self.linear(s)
        return update


@lru_cache(maxsize=None)
def identity_quats(batch_dims: Tuple[int], dtype: Optional[torch.dtype]=None, device: Optional[torch.device]=None, requires_grad: bool=True) ->torch.Tensor:
    quat = torch.zeros((*batch_dims, 4), dtype=dtype, device=device, requires_grad=requires_grad)
    with torch.no_grad():
        quat[..., 0] = 1
    return quat


@lru_cache(maxsize=None)
def identity_rot_mats(batch_dims: Tuple[int], dtype: Optional[torch.dtype]=None, device: Optional[torch.device]=None, requires_grad: bool=True) ->torch.Tensor:
    rots = torch.eye(3, dtype=dtype, device=device, requires_grad=requires_grad)
    rots = rots.view(*((1,) * len(batch_dims)), 3, 3)
    rots = rots.expand(*batch_dims, -1, -1)
    rots = rots.contiguous()
    return rots


def invert_quat(quat: torch.Tensor):
    quat_prime = quat.clone()
    quat_prime[..., 1:] *= -1
    inv = quat_prime / torch.sum(quat ** 2, dim=-1, keepdim=True)
    return inv


def invert_rot_mat(rot_mat: torch.Tensor):
    return rot_mat.transpose(-1, -2)


_QTR_MAT = np.zeros((4, 4, 3, 3))


_QUAT_MULTIPLY = np.zeros((4, 4, 4))


_QUAT_MULTIPLY_BY_VEC = _QUAT_MULTIPLY[:, 1:, :]


_CACHED_QUATS = {'_QTR_MAT': _QTR_MAT, '_QUAT_MULTIPLY': _QUAT_MULTIPLY, '_QUAT_MULTIPLY_BY_VEC': _QUAT_MULTIPLY_BY_VEC}


@lru_cache(maxsize=None)
def _get_quat(quat_key, dtype, device):
    return torch.tensor(_CACHED_QUATS[quat_key], dtype=dtype, device=device)


def quat_multiply(quat1, quat2):
    """Multiply a quaternion by another quaternion."""
    mat = _get_quat('_QUAT_MULTIPLY', dtype=quat1.dtype, device=quat1.device)
    reshaped_mat = mat.view((1,) * len(quat1.shape[:-1]) + mat.shape)
    return torch.sum(reshaped_mat * quat1[..., :, None, None] * quat2[..., None, :, None], dim=(-3, -2))


def quat_multiply_by_vec(quat, vec):
    """Multiply a quaternion by a pure-vector quaternion."""
    mat = _get_quat('_QUAT_MULTIPLY_BY_VEC', dtype=quat.dtype, device=quat.device)
    reshaped_mat = mat.view((1,) * len(quat.shape[:-1]) + mat.shape)
    return torch.sum(reshaped_mat * quat[..., :, None, None] * vec[..., None, :, None], dim=(-3, -2))


def quat_to_rot(quat: torch.Tensor) ->torch.Tensor:
    """
        Converts a quaternion to a rotation matrix.

        Args:
            quat: [*, 4] quaternions
        Returns:
            [*, 3, 3] rotation matrices
    """
    quat = quat[..., None] * quat[..., None, :]
    mat = _get_quat('_QTR_MAT', dtype=quat.dtype, device=quat.device)
    shaped_qtr_mat = mat.view((1,) * len(quat.shape[:-2]) + mat.shape)
    quat = quat[..., None, None] * shaped_qtr_mat
    return torch.sum(quat, dim=(-3, -4))


def rot_matmul(a: torch.Tensor, b: torch.Tensor) ->torch.Tensor:
    """
        Performs matrix multiplication of two rotation matrix tensors. Written
        out by hand to avoid AMP downcasting.

        Args:
            a: [*, 3, 3] left multiplicand
            b: [*, 3, 3] right multiplicand
        Returns:
            The product ab
    """

    def row_mul(i):
        return torch.stack([a[..., i, 0] * b[..., 0, 0] + a[..., i, 1] * b[..., 1, 0] + a[..., i, 2] * b[..., 2, 0], a[..., i, 0] * b[..., 0, 1] + a[..., i, 1] * b[..., 1, 1] + a[..., i, 2] * b[..., 2, 1], a[..., i, 0] * b[..., 0, 2] + a[..., i, 1] * b[..., 1, 2] + a[..., i, 2] * b[..., 2, 2]], dim=-1)
    return torch.stack([row_mul(0), row_mul(1), row_mul(2)], dim=-2)


def rot_to_quat(rot: torch.Tensor):
    if rot.shape[-2:] != (3, 3):
        raise ValueError('Input rotation is incorrectly shaped')
    rot = [[rot[..., i, j] for j in range(3)] for i in range(3)]
    [[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]] = rot
    k = [[xx + yy + zz, zy - yz, xz - zx, yx - xy], [zy - yz, xx - yy - zz, xy + yx, xz + zx], [xz - zx, xy + yx, yy - xx - zz, yz + zy], [yx - xy, xz + zx, yz + zy, zz - xx - yy]]
    k = 1.0 / 3.0 * torch.stack([torch.stack(t, dim=-1) for t in k], dim=-2)
    _, vectors = torch.linalg.eigh(k)
    return vectors[..., -1]


def rot_vec_mul(r: torch.Tensor, t: torch.Tensor) ->torch.Tensor:
    """
        Applies a rotation to a vector. Written out by hand to avoid transfer
        to avoid AMP downcasting.

        Args:
            r: [*, 3, 3] rotation matrices
            t: [*, 3] coordinate tensors
        Returns:
            [*, 3] rotated coordinates
    """
    x, y, z = torch.unbind(t, dim=-1)
    return torch.stack([r[..., 0, 0] * x + r[..., 0, 1] * y + r[..., 0, 2] * z, r[..., 1, 0] * x + r[..., 1, 1] * y + r[..., 1, 2] * z, r[..., 2, 0] * x + r[..., 2, 1] * y + r[..., 2, 2] * z], dim=-1)


@lru_cache(maxsize=None)
def identity_trans(batch_dims: Tuple[int], dtype: Optional[torch.dtype]=None, device: Optional[torch.device]=None, requires_grad: bool=True) ->torch.Tensor:
    trans = torch.zeros((*batch_dims, 3), dtype=dtype, device=device, requires_grad=requires_grad)
    return trans


def ipa_point_weights_init_(weights):
    with torch.no_grad():
        softplus_inverse_1 = 0.541324854612918
        weights.fill_(softplus_inverse_1)


class StructureModuleTransitionLayer(nn.Module):

    def __init__(self, c):
        super(StructureModuleTransitionLayer, self).__init__()
        self.c = c
        self.linear_1 = Linear(self.c, self.c, init='relu')
        self.linear_2 = Linear(self.c, self.c, init='relu')
        self.linear_3 = Linear(self.c, self.c, init='final')
        self.relu = nn.ReLU()

    def forward(self, s):
        s_initial = s
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = self.relu(s)
        s = self.linear_3(s)
        s = s + s_initial
        return s


class StructureModuleTransition(nn.Module):

    def __init__(self, c, num_layers, dropout_rate):
        super(StructureModuleTransition, self).__init__()
        self.c = c
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            l = StructureModuleTransitionLayer(self.c)
            self.layers.append(l)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.layer_norm = LayerNorm(self.c)

    def forward(self, s):
        for l in self.layers:
            s = l(s)
        s = self.dropout(s)
        s = self.layer_norm(s)
        return s


def dict_multimap(fn, dicts):
    first = dicts[0]
    new_dict = {}
    for k, v in first.items():
        all_v = [d[k] for d in dicts]
        if type(v) is dict:
            new_dict[k] = dict_multimap(fn, all_v)
        else:
            new_dict[k] = fn(all_v)
    return new_dict


restype_atom14_mask = np.zeros([21, 14], dtype=np.float32)


restype_atom14_rigid_group_positions = np.zeros([21, 14, 3], dtype=np.float32)


restype_atom14_to_rigid_group = np.zeros([21, 14], dtype=int)


restype_rigid_group_default_frame = np.zeros([21, 8, 4, 4], dtype=np.float32)


class StructureModule(nn.Module):

    def __init__(self, c_s, c_z, c_ipa, c_resnet, no_heads_ipa, no_qk_points, no_v_points, dropout_rate, no_blocks, no_transition_layers, no_resnet_blocks, no_angles, trans_scale_factor, epsilon, inf, **kwargs):
        """
        Args:
            c_s:
                Single representation channel dimension
            c_z:
                Pair representation channel dimension
            c_ipa:
                IPA hidden channel dimension
            c_resnet:
                Angle resnet (Alg. 23 lines 11-14) hidden channel dimension
            no_heads_ipa:
                Number of IPA heads
            no_qk_points:
                Number of query/key points to generate during IPA
            no_v_points:
                Number of value points to generate during IPA
            dropout_rate:
                Dropout rate used throughout the layer
            no_blocks:
                Number of structure module blocks
            no_transition_layers:
                Number of layers in the single representation transition
                (Alg. 23 lines 8-9)
            no_resnet_blocks:
                Number of blocks in the angle resnet
            no_angles:
                Number of angles to generate in the angle resnet
            trans_scale_factor:
                Scale of single representation transition hidden dimension
            epsilon:
                Small number used in angle resnet normalization
            inf:
                Large number used for attention masking
        """
        super(StructureModule, self).__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.c_ipa = c_ipa
        self.c_resnet = c_resnet
        self.no_heads_ipa = no_heads_ipa
        self.no_qk_points = no_qk_points
        self.no_v_points = no_v_points
        self.dropout_rate = dropout_rate
        self.no_blocks = no_blocks
        self.no_transition_layers = no_transition_layers
        self.no_resnet_blocks = no_resnet_blocks
        self.no_angles = no_angles
        self.trans_scale_factor = trans_scale_factor
        self.epsilon = epsilon
        self.inf = inf
        self.layer_norm_s = LayerNorm(self.c_s)
        self.layer_norm_z = LayerNorm(self.c_z)
        self.linear_in = Linear(self.c_s, self.c_s)
        self.ipa = InvariantPointAttention(self.c_s, self.c_z, self.c_ipa, self.no_heads_ipa, self.no_qk_points, self.no_v_points, inf=self.inf, eps=self.epsilon)
        self.ipa_dropout = nn.Dropout(self.dropout_rate)
        self.layer_norm_ipa = LayerNorm(self.c_s)
        self.transition = StructureModuleTransition(self.c_s, self.no_transition_layers, self.dropout_rate)
        self.bb_update = BackboneUpdate(self.c_s)
        self.angle_resnet = AngleResnet(self.c_s, self.c_resnet, self.no_resnet_blocks, self.no_angles, self.epsilon)

    def forward(self, evoformer_output_dict, aatype, mask=None, inplace_safe=False, _offload_inference=False):
        """
        Args:
            evoformer_output_dict:
                Dictionary containing:
                    "single":
                        [*, N_res, C_s] single representation
                    "pair":
                        [*, N_res, N_res, C_z] pair representation
            aatype:
                [*, N_res] amino acid indices
            mask:
                Optional [*, N_res] sequence mask
        Returns:
            A dictionary of outputs
        """
        s = evoformer_output_dict['single']
        if mask is None:
            mask = s.new_ones(s.shape[:-1])
        s = self.layer_norm_s(s)
        z = self.layer_norm_z(evoformer_output_dict['pair'])
        z_reference_list = None
        if _offload_inference:
            assert sys.getrefcount(evoformer_output_dict['pair']) == 2
            evoformer_output_dict['pair'] = evoformer_output_dict['pair'].cpu()
            z_reference_list = [z]
            z = None
        s_initial = s
        s = self.linear_in(s)
        rigids = Rigid.identity(s.shape[:-1], s.dtype, s.device, self.training, fmt='quat')
        outputs = []
        for i in range(self.no_blocks):
            s = s + self.ipa(s, z, rigids, mask, inplace_safe=inplace_safe, _offload_inference=_offload_inference, _z_reference_list=z_reference_list)
            s = self.ipa_dropout(s)
            s = self.layer_norm_ipa(s)
            s = self.transition(s)
            rigids = rigids.compose_q_update_vec(self.bb_update(s))
            backb_to_global = Rigid(Rotation(rot_mats=rigids.get_rots().get_rot_mats(), quats=None), rigids.get_trans())
            backb_to_global = backb_to_global.scale_translation(self.trans_scale_factor)
            unnormalized_angles, angles = self.angle_resnet(s, s_initial)
            all_frames_to_global = self.torsion_angles_to_frames(backb_to_global, angles, aatype)
            pred_xyz = self.frames_and_literature_positions_to_atom14_pos(all_frames_to_global, aatype)
            scaled_rigids = rigids.scale_translation(self.trans_scale_factor)
            preds = {'frames': scaled_rigids.to_tensor_7(), 'sidechain_frames': all_frames_to_global.to_tensor_4x4(), 'unnormalized_angles': unnormalized_angles, 'angles': angles, 'positions': pred_xyz, 'states': s}
            outputs.append(preds)
            rigids = rigids.stop_rot_gradient()
        del z, z_reference_list
        if _offload_inference:
            evoformer_output_dict['pair'] = evoformer_output_dict['pair']
        outputs = dict_multimap(torch.stack, outputs)
        outputs['single'] = s
        return outputs

    def _init_residue_constants(self, float_dtype, device):
        if not hasattr(self, 'default_frames'):
            self.register_buffer('default_frames', torch.tensor(restype_rigid_group_default_frame, dtype=float_dtype, device=device, requires_grad=False), persistent=False)
        if not hasattr(self, 'group_idx'):
            self.register_buffer('group_idx', torch.tensor(restype_atom14_to_rigid_group, device=device, requires_grad=False), persistent=False)
        if not hasattr(self, 'atom_mask'):
            self.register_buffer('atom_mask', torch.tensor(restype_atom14_mask, dtype=float_dtype, device=device, requires_grad=False), persistent=False)
        if not hasattr(self, 'lit_positions'):
            self.register_buffer('lit_positions', torch.tensor(restype_atom14_rigid_group_positions, dtype=float_dtype, device=device, requires_grad=False), persistent=False)

    def torsion_angles_to_frames(self, r, alpha, f):
        self._init_residue_constants(alpha.dtype, alpha.device)
        return torsion_angles_to_frames(r, alpha, f, self.default_frames)

    def frames_and_literature_positions_to_atom14_pos(self, r, f):
        self._init_residue_constants(r.get_rots().dtype, r.get_rots().device)
        return frames_and_literature_positions_to_atom14_pos(r, f, self.default_frames, self.group_idx, self.atom_mask, self.lit_positions)


class TriangleAttentionEndingNode(TriangleAttention):
    """
    Implements Algorithm 14.
    """
    __init__ = partialmethod(TriangleAttention.__init__, starting=False)


TriangleAttentionStartingNode = TriangleAttention


class TemplatePairStackBlock(nn.Module):

    def __init__(self, c_t: int, c_hidden_tri_att: int, c_hidden_tri_mul: int, no_heads: int, pair_transition_n: int, dropout_rate: float, inf: float, **kwargs):
        super(TemplatePairStackBlock, self).__init__()
        self.c_t = c_t
        self.c_hidden_tri_att = c_hidden_tri_att
        self.c_hidden_tri_mul = c_hidden_tri_mul
        self.no_heads = no_heads
        self.pair_transition_n = pair_transition_n
        self.dropout_rate = dropout_rate
        self.inf = inf
        self.dropout_row = DropoutRowwise(self.dropout_rate)
        self.dropout_col = DropoutColumnwise(self.dropout_rate)
        self.tri_att_start = TriangleAttentionStartingNode(self.c_t, self.c_hidden_tri_att, self.no_heads, inf=inf)
        self.tri_att_end = TriangleAttentionEndingNode(self.c_t, self.c_hidden_tri_att, self.no_heads, inf=inf)
        self.tri_mul_out = TriangleMultiplicationOutgoing(self.c_t, self.c_hidden_tri_mul)
        self.tri_mul_in = TriangleMultiplicationIncoming(self.c_t, self.c_hidden_tri_mul)
        self.pair_transition = PairTransition(self.c_t, self.pair_transition_n)

    def forward(self, z: torch.Tensor, mask: torch.Tensor, chunk_size: Optional[int]=None, use_lma: bool=False, inplace_safe: bool=False, _mask_trans: bool=True, _attn_chunk_size: Optional[int]=None):
        if _attn_chunk_size is None:
            _attn_chunk_size = chunk_size
        single_templates = [t.unsqueeze(-4) for t in torch.unbind(z, dim=-4)]
        single_templates_masks = [m.unsqueeze(-3) for m in torch.unbind(mask, dim=-3)]
        for i in range(len(single_templates)):
            single = single_templates[i]
            single_mask = single_templates_masks[i]
            single = add(single, self.dropout_row(self.tri_att_start(single, chunk_size=_attn_chunk_size, mask=single_mask, use_lma=use_lma, inplace_safe=inplace_safe)), inplace_safe)
            single = add(single, self.dropout_col(self.tri_att_end(single, chunk_size=_attn_chunk_size, mask=single_mask, use_lma=use_lma, inplace_safe=inplace_safe)), inplace_safe)
            tmu_update = self.tri_mul_out(single, mask=single_mask, inplace_safe=inplace_safe, _add_with_inplace=True)
            if not inplace_safe:
                single = single + self.dropout_row(tmu_update)
            else:
                single = tmu_update
            del tmu_update
            tmu_update = self.tri_mul_in(single, mask=single_mask, inplace_safe=inplace_safe, _add_with_inplace=True)
            if not inplace_safe:
                single = single + self.dropout_row(tmu_update)
            else:
                single = tmu_update
            del tmu_update
            single = add(single, self.pair_transition(single, mask=single_mask if _mask_trans else None, chunk_size=chunk_size), inplace_safe)
            if not inplace_safe:
                single_templates[i] = single
        if not inplace_safe:
            z = torch.cat(single_templates, dim=-4)
        return z


class TemplatePairStack(nn.Module):
    """
    Implements Algorithm 16.
    """

    def __init__(self, c_t, c_hidden_tri_att, c_hidden_tri_mul, no_blocks, no_heads, pair_transition_n, dropout_rate, blocks_per_ckpt, tune_chunk_size: bool=False, inf=1000000000.0, **kwargs):
        """
        Args:
            c_t:
                Template embedding channel dimension
            c_hidden_tri_att:
                Per-head hidden dimension for triangular attention
            c_hidden_tri_att:
                Hidden dimension for triangular multiplication
            no_blocks:
                Number of blocks in the stack
            pair_transition_n:
                Scale of pair transition (Alg. 15) hidden dimension
            dropout_rate:
                Dropout rate used throughout the stack
            blocks_per_ckpt:
                Number of blocks per activation checkpoint. None disables
                activation checkpointing
        """
        super(TemplatePairStack, self).__init__()
        self.blocks_per_ckpt = blocks_per_ckpt
        self.blocks = nn.ModuleList()
        for _ in range(no_blocks):
            block = TemplatePairStackBlock(c_t=c_t, c_hidden_tri_att=c_hidden_tri_att, c_hidden_tri_mul=c_hidden_tri_mul, no_heads=no_heads, pair_transition_n=pair_transition_n, dropout_rate=dropout_rate, inf=inf)
            self.blocks.append(block)
        self.layer_norm = LayerNorm(c_t)
        self.tune_chunk_size = tune_chunk_size
        self.chunk_size_tuner = None
        if tune_chunk_size:
            self.chunk_size_tuner = ChunkSizeTuner()

    def forward(self, t: torch.tensor, mask: torch.tensor, chunk_size: int, use_lma: bool=False, inplace_safe: bool=False, _mask_trans: bool=True):
        """
        Args:
            t:
                [*, N_templ, N_res, N_res, C_t] template embedding
            mask:
                [*, N_templ, N_res, N_res] mask
        Returns:
            [*, N_templ, N_res, N_res, C_t] template embedding update
        """
        if mask.shape[-3] == 1:
            expand_idx = list(mask.shape)
            expand_idx[-3] = t.shape[-4]
            mask = mask.expand(*expand_idx)
        blocks = [partial(b, mask=mask, chunk_size=chunk_size, use_lma=use_lma, inplace_safe=inplace_safe, _mask_trans=_mask_trans) for b in self.blocks]
        if chunk_size is not None and self.chunk_size_tuner is not None:
            assert not self.training
            tuned_chunk_size = self.chunk_size_tuner.tune_chunk_size(representative_fn=blocks[0], args=(t.clone(),), min_chunk_size=chunk_size)
            blocks = [partial(b, chunk_size=tuned_chunk_size, _attn_chunk_size=max(chunk_size, tuned_chunk_size // 4)) for b in blocks]
        t, = checkpoint_blocks(blocks=blocks, args=(t,), blocks_per_ckpt=self.blocks_per_ckpt if self.training else None)
        t = self.layer_norm(t)
        return t


class TemplatePointwiseAttention(nn.Module):
    """
    Implements Algorithm 17.
    """

    def __init__(self, c_t, c_z, c_hidden, no_heads, inf, **kwargs):
        """
        Args:
            c_t:
                Template embedding channel dimension
            c_z:
                Pair embedding channel dimension
            c_hidden:
                Hidden channel dimension
        """
        super(TemplatePointwiseAttention, self).__init__()
        self.c_t = c_t
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.inf = inf
        self.mha = Attention(self.c_z, self.c_t, self.c_t, self.c_hidden, self.no_heads, gating=False)

    def _chunk(self, z: torch.Tensor, t: torch.Tensor, biases: List[torch.Tensor], chunk_size: int, use_lma: bool=False) ->torch.Tensor:
        mha_inputs = {'q_x': z, 'kv_x': t, 'biases': biases}
        return chunk_layer(partial(self.mha, use_lma=use_lma), mha_inputs, chunk_size=chunk_size, no_batch_dims=len(z.shape[:-2]))

    def forward(self, t: torch.Tensor, z: torch.Tensor, template_mask: Optional[torch.Tensor]=None, chunk_size: Optional[int]=256, use_lma: bool=False) ->torch.Tensor:
        """
        Args:
            t:
                [*, N_templ, N_res, N_res, C_t] template embedding
            z:
                [*, N_res, N_res, C_t] pair embedding
            template_mask:
                [*, N_templ] template mask
        Returns:
            [*, N_res, N_res, C_z] pair embedding update
        """
        if template_mask is None:
            template_mask = t.new_ones(t.shape[:-3])
        bias = self.inf * (template_mask[..., None, None, None, None, :] - 1)
        z = z.unsqueeze(-2)
        t = permute_final_dims(t, (1, 2, 0, 3))
        biases = [bias]
        if chunk_size is not None and not self.training:
            z = self._chunk(z, t, biases, chunk_size, use_lma=use_lma)
        else:
            z = self.mha(q_x=z, kv_x=t, biases=biases, use_lma=use_lma)
        z = z.squeeze(-2)
        return z


def batched_gather(data, inds, dim=0, no_batch_dims=0):
    ranges = []
    for i, s in enumerate(data.shape[:no_batch_dims]):
        r = torch.arange(s)
        r = r.view(*(*((1,) * i), -1, *((1,) * (len(inds.shape) - i - 1))))
        ranges.append(r)
    remaining_dims = [slice(None) for _ in range(len(data.shape) - no_batch_dims)]
    remaining_dims[dim - no_batch_dims if dim >= 0 else dim] = inds
    ranges.extend(remaining_dims)
    return data[ranges]


def atom14_to_atom37(atom14, batch):
    atom37_data = batched_gather(atom14, batch['residx_atom37_to_atom14'], dim=-2, no_batch_dims=len(atom14.shape[:-2]))
    atom37_data = atom37_data * batch['atom37_atom_exists'][..., None]
    return atom37_data


def build_extra_msa_feat(batch):
    msa_1hot = nn.functional.one_hot(batch['extra_msa'], 23)
    msa_feat = [msa_1hot, batch['extra_has_deletion'].unsqueeze(-1), batch['extra_deletion_value'].unsqueeze(-1)]
    return torch.cat(msa_feat, dim=-1)


def build_template_angle_feat(template_feats):
    template_aatype = template_feats['template_aatype']
    torsion_angles_sin_cos = template_feats['template_torsion_angles_sin_cos']
    alt_torsion_angles_sin_cos = template_feats['template_alt_torsion_angles_sin_cos']
    torsion_angles_mask = template_feats['template_torsion_angles_mask']
    template_angle_feat = torch.cat([nn.functional.one_hot(template_aatype, 22), torsion_angles_sin_cos.reshape(*torsion_angles_sin_cos.shape[:-2], 14), alt_torsion_angles_sin_cos.reshape(*alt_torsion_angles_sin_cos.shape[:-2], 14), torsion_angles_mask], dim=-1)
    return template_angle_feat


def build_template_pair_feat(batch, min_bin, max_bin, no_bins, use_unit_vector=False, eps=1e-20, inf=100000000.0):
    template_mask = batch['template_pseudo_beta_mask']
    template_mask_2d = template_mask[..., None] * template_mask[..., None, :]
    tpb = batch['template_pseudo_beta']
    dgram = torch.sum((tpb[..., None, :] - tpb[..., None, :, :]) ** 2, dim=-1, keepdim=True)
    lower = torch.linspace(min_bin, max_bin, no_bins, device=tpb.device) ** 2
    upper = torch.cat([lower[1:], lower.new_tensor([inf])], dim=-1)
    dgram = ((dgram > lower) * (dgram < upper)).type(dgram.dtype)
    to_concat = [dgram, template_mask_2d[..., None]]
    aatype_one_hot = nn.functional.one_hot(batch['template_aatype'], rc.restype_num + 2)
    n_res = batch['template_aatype'].shape[-1]
    to_concat.append(aatype_one_hot[..., None, :, :].expand(*aatype_one_hot.shape[:-2], n_res, -1, -1))
    to_concat.append(aatype_one_hot[..., None, :].expand(*aatype_one_hot.shape[:-2], -1, n_res, -1))
    n, ca, c = [rc.atom_order[a] for a in ['N', 'CA', 'C']]
    rigids = Rigid.make_transform_from_reference(n_xyz=batch['template_all_atom_positions'][..., n, :], ca_xyz=batch['template_all_atom_positions'][..., ca, :], c_xyz=batch['template_all_atom_positions'][..., c, :], eps=eps)
    points = rigids.get_trans()[..., None, :, :]
    rigid_vec = rigids[..., None].invert_apply(points)
    inv_distance_scalar = torch.rsqrt(eps + torch.sum(rigid_vec ** 2, dim=-1))
    t_aa_masks = batch['template_all_atom_mask']
    template_mask = t_aa_masks[..., n] * t_aa_masks[..., ca] * t_aa_masks[..., c]
    template_mask_2d = template_mask[..., None] * template_mask[..., None, :]
    inv_distance_scalar = inv_distance_scalar * template_mask_2d
    unit_vector = rigid_vec * inv_distance_scalar[..., None]
    if not use_unit_vector:
        unit_vector = unit_vector * 0.0
    to_concat.extend(torch.unbind(unit_vector[..., None, :], dim=-1))
    to_concat.append(template_mask_2d[..., None])
    act = torch.cat(to_concat, dim=-1)
    act = act * template_mask_2d[..., None]
    return act


def embed_templates_average(model, batch, z, pair_mask, templ_dim, templ_group_size=2, inplace_safe=False):
    """
    Args:
        model: 
            An AlphaFold model object
        batch: 
            An AlphaFold input batch. See documentation of AlphaFold.
        z: 
            A [*, N, N, C_z] pair embedding
        pair_mask: 
            A [*, N, N] pair mask
        templ_dim: 
            The template dimension of the template tensors in batch
        templ_group_size: 
            Granularity of the approximation. Larger values trade memory for 
            greater proximity to the original function
    Returns:
        A dictionary of template pair and angle embeddings.

    A memory-efficient approximation of the "embed_templates" method of the 
    AlphaFold class. Instead of running pointwise attention over pair 
    embeddings for all of the templates at the same time, it splits templates 
    into groups of size templ_group_size, computes embeddings for each group 
    normally, and then averages the group embeddings. In our experiments, this 
    approximation has a minimal effect on the quality of the resulting 
    embedding, while its low memory footprint allows the number of templates 
    to scale almost indefinitely.
    """
    n = z.shape[-2]
    n_templ = batch['template_aatype'].shape[templ_dim]
    out_tensor = z.new_zeros(z.shape)
    for i in range(0, n_templ, templ_group_size):

        def slice_template_tensor(t):
            s = [slice(None) for _ in t.shape]
            s[templ_dim] = slice(i, i + templ_group_size)
            return t[s]
        template_feats = tensor_tree_map(slice_template_tensor, batch)
        t = build_template_pair_feat(template_feats, use_unit_vector=model.config.template.use_unit_vector, inf=model.config.template.inf, eps=model.config.template.eps, **model.config.template.distogram)
        t = model.template_pair_embedder(t)
        t = model.template_pair_stack(t, pair_mask.unsqueeze(-3), chunk_size=model.globals.chunk_size, use_lma=model.globals.use_lma, _mask_trans=model.config._mask_trans)
        t = model.template_pointwise_att(t, z, template_mask=template_feats['template_mask'], use_lma=model.globals.use_lma)
        denom = math.ceil(n_templ / templ_group_size)
        if inplace_safe:
            t /= denom
        else:
            t = t / denom
        if inplace_safe:
            out_tensor += t
        else:
            out_tensor = out_tensor + t
        del t
    if inplace_safe:
        out_tensor *= torch.sum(batch['template_mask'], dim=-1) > 0
    else:
        out_tensor = out_tensor * (torch.sum(batch['template_mask'], dim=-1) > 0)
    ret = {}
    if model.config.template.embed_angles:
        template_angle_feat = build_template_angle_feat(batch)
        a = model.template_angle_embedder(template_angle_feat)
        ret['template_angle_embedding'] = a
    ret.update({'template_pair_embedding': out_tensor})
    return ret


def embed_templates_offload(model, batch, z, pair_mask, templ_dim, template_chunk_size=256, inplace_safe=False):
    """
    Args:
        model: 
            An AlphaFold model object
        batch: 
            An AlphaFold input batch. See documentation of AlphaFold.
        z: 
            A [*, N, N, C_z] pair embedding
        pair_mask: 
            A [*, N, N] pair mask
        templ_dim: 
            The template dimension of the template tensors in batch
        template_chunk_size: 
            Integer value controlling how quickly the offloaded pair embedding
            tensor is brought back into GPU memory. In dire straits, can be
            lowered to reduce memory consumption of this function even more.
    Returns:
        A dictionary of template pair and angle embeddings.
    
    A version of the "embed_templates" method of the AlphaFold class that
    offloads the large template pair tensor to CPU. Slower but more frugal 
    with GPU memory than the original. Useful for long-sequence inference.
    """
    pair_embeds_cpu = []
    n = z.shape[-2]
    n_templ = batch['template_aatype'].shape[templ_dim]
    for i in range(n_templ):
        idx = batch['template_aatype'].new_tensor(i)
        single_template_feats = tensor_tree_map(lambda t: torch.index_select(t, templ_dim, idx).squeeze(templ_dim), batch)
        t = build_template_pair_feat(single_template_feats, use_unit_vector=model.config.template.use_unit_vector, inf=model.config.template.inf, eps=model.config.template.eps, **model.config.template.distogram)
        t = model.template_pair_embedder(t)
        t = model.template_pair_stack(t.unsqueeze(templ_dim), pair_mask.unsqueeze(-3), chunk_size=model.globals.chunk_size, use_lma=model.globals.use_lma, _mask_trans=model.config._mask_trans)
        assert sys.getrefcount(t) == 2
        pair_embeds_cpu.append(t.cpu())
        del t
    t = z.new_zeros(z.shape)
    for i in range(0, n, template_chunk_size):
        pair_chunks = [p[..., i:i + template_chunk_size, :, :] for p in pair_embeds_cpu]
        pair_chunk = torch.cat(pair_chunks, dim=templ_dim)
        z_chunk = z[..., i:i + template_chunk_size, :, :]
        att_chunk = model.template_pointwise_att(pair_chunk, z_chunk, template_mask=batch['template_mask'], use_lma=model.globals.use_lma)
        t[..., i:i + template_chunk_size, :, :] = att_chunk
    del pair_chunks
    if inplace_safe:
        t = t * (torch.sum(batch['template_mask'], dim=-1) > 0)
    else:
        t *= torch.sum(batch['template_mask'], dim=-1) > 0
    ret = {}
    if model.config.template.embed_angles:
        template_angle_feat = build_template_angle_feat(batch)
        a = model.template_angle_embedder(template_angle_feat)
        ret['template_angle_embedding'] = a
    ret.update({'template_pair_embedding': t})
    return ret


def pseudo_beta_fn(aatype, all_atom_positions, all_atom_masks):
    is_gly = aatype == rc.restype_order['G']
    ca_idx = rc.atom_order['CA']
    cb_idx = rc.atom_order['CB']
    pseudo_beta = torch.where(is_gly[..., None].expand(*((-1,) * len(is_gly.shape)), 3), all_atom_positions[..., ca_idx, :], all_atom_positions[..., cb_idx, :])
    if all_atom_masks is not None:
        pseudo_beta_mask = torch.where(is_gly, all_atom_masks[..., ca_idx], all_atom_masks[..., cb_idx])
        return pseudo_beta, pseudo_beta_mask
    else:
        return pseudo_beta


class AlphaFold(nn.Module):
    """
    Alphafold 2.

    Implements Algorithm 2 (but with training).
    """

    def __init__(self, config):
        """
        Args:
            config:
                A dict-like config object (like the one in config.py)
        """
        super(AlphaFold, self).__init__()
        self.globals = config.globals
        self.config = config.model
        self.template_config = self.config.template
        self.extra_msa_config = self.config.extra_msa
        self.input_embedder = InputEmbedder(**self.config['input_embedder'])
        self.recycling_embedder = RecyclingEmbedder(**self.config['recycling_embedder'])
        if self.template_config.enabled:
            self.template_angle_embedder = TemplateAngleEmbedder(**self.template_config['template_angle_embedder'])
            self.template_pair_embedder = TemplatePairEmbedder(**self.template_config['template_pair_embedder'])
            self.template_pair_stack = TemplatePairStack(**self.template_config['template_pair_stack'])
            self.template_pointwise_att = TemplatePointwiseAttention(**self.template_config['template_pointwise_attention'])
        if self.extra_msa_config.enabled:
            self.extra_msa_embedder = ExtraMSAEmbedder(**self.extra_msa_config['extra_msa_embedder'])
            self.extra_msa_stack = ExtraMSAStack(**self.extra_msa_config['extra_msa_stack'])
        self.evoformer = EvoformerStack(**self.config['evoformer_stack'])
        self.structure_module = StructureModule(**self.config['structure_module'])
        self.aux_heads = AuxiliaryHeads(self.config['heads'])

    def embed_templates(self, batch, z, pair_mask, templ_dim, inplace_safe):
        if self.template_config.offload_templates:
            return embed_templates_offload(self, batch, z, pair_mask, templ_dim, inplace_safe=inplace_safe)
        elif self.template_config.average_templates:
            return embed_templates_average(self, batch, z, pair_mask, templ_dim, inplace_safe=inplace_safe)
        pair_embeds = []
        n = z.shape[-2]
        n_templ = batch['template_aatype'].shape[templ_dim]
        if inplace_safe:
            t_pair = z.new_zeros(z.shape[:-3] + (n_templ, n, n, self.globals.c_t))
        for i in range(n_templ):
            idx = batch['template_aatype'].new_tensor(i)
            single_template_feats = tensor_tree_map(lambda t: torch.index_select(t, templ_dim, idx).squeeze(templ_dim), batch)
            t = build_template_pair_feat(single_template_feats, use_unit_vector=self.config.template.use_unit_vector, inf=self.config.template.inf, eps=self.config.template.eps, **self.config.template.distogram)
            t = self.template_pair_embedder(t)
            if inplace_safe:
                t_pair[..., i, :, :, :] = t
            else:
                pair_embeds.append(t)
            del t
        if not inplace_safe:
            t_pair = torch.stack(pair_embeds, dim=templ_dim)
        del pair_embeds
        t = self.template_pair_stack(t_pair, pair_mask.unsqueeze(-3), chunk_size=self.globals.chunk_size, use_lma=self.globals.use_lma, inplace_safe=inplace_safe, _mask_trans=self.config._mask_trans)
        del t_pair
        t = self.template_pointwise_att(t, z, template_mask=batch['template_mask'], use_lma=self.globals.use_lma)
        t_mask = torch.sum(batch['template_mask'], dim=-1) > 0
        t_mask = t_mask.reshape(*t_mask.shape, *([1] * (len(t.shape) - len(t_mask.shape))))
        if inplace_safe:
            t *= t_mask
        else:
            t = t * t_mask
        ret = {}
        ret.update({'template_pair_embedding': t})
        del t
        if self.config.template.embed_angles:
            template_angle_feat = build_template_angle_feat(batch)
            a = self.template_angle_embedder(template_angle_feat)
            ret['template_angle_embedding'] = a
        return ret

    def iteration(self, feats, prevs, _recycle=True):
        outputs = {}
        dtype = next(self.parameters()).dtype
        for k in feats:
            if feats[k].dtype == torch.float32:
                feats[k] = feats[k]
        batch_dims = feats['target_feat'].shape[:-2]
        no_batch_dims = len(batch_dims)
        n = feats['target_feat'].shape[-2]
        n_seq = feats['msa_feat'].shape[-3]
        device = feats['target_feat'].device
        inplace_safe = not (self.training or torch.is_grad_enabled())
        seq_mask = feats['seq_mask']
        pair_mask = seq_mask[..., None] * seq_mask[..., None, :]
        msa_mask = feats['msa_mask']
        m, z = self.input_embedder(feats['target_feat'], feats['residue_index'], feats['msa_feat'], inplace_safe=inplace_safe)
        m_1_prev, z_prev, x_prev = reversed([prevs.pop() for _ in range(3)])
        if None in [m_1_prev, z_prev, x_prev]:
            m_1_prev = m.new_zeros((*batch_dims, n, self.config.input_embedder.c_m), requires_grad=False)
            z_prev = z.new_zeros((*batch_dims, n, n, self.config.input_embedder.c_z), requires_grad=False)
            x_prev = z.new_zeros((*batch_dims, n, residue_constants.atom_type_num, 3), requires_grad=False)
        x_prev = pseudo_beta_fn(feats['aatype'], x_prev, None)
        if self.globals.offload_inference and inplace_safe:
            m = m.cpu()
            z = z.cpu()
        m_1_prev_emb, z_prev_emb = self.recycling_embedder(m_1_prev, z_prev, x_prev, inplace_safe=inplace_safe)
        if self.globals.offload_inference and inplace_safe:
            m = m
            z = z
        m[..., 0, :, :] += m_1_prev_emb
        z = add(z, z_prev_emb, inplace=inplace_safe)
        del m_1_prev, z_prev, x_prev, m_1_prev_emb, z_prev_emb
        if self.config.template.enabled:
            template_feats = {k: v for k, v in feats.items() if k.startswith('template_')}
            template_embeds = self.embed_templates(template_feats, z, pair_mask, no_batch_dims, inplace_safe=inplace_safe)
            z = add(z, template_embeds.pop('template_pair_embedding'), inplace_safe)
            if 'template_angle_embedding' in template_embeds:
                m = torch.cat([m, template_embeds['template_angle_embedding']], dim=-3)
                torsion_angles_mask = feats['template_torsion_angles_mask']
                msa_mask = torch.cat([feats['msa_mask'], torsion_angles_mask[..., 2]], dim=-2)
        if self.config.extra_msa.enabled:
            a = self.extra_msa_embedder(build_extra_msa_feat(feats))
            if self.globals.offload_inference:
                input_tensors = [a, z]
                del a, z
                z = self.extra_msa_stack._forward_offload(input_tensors, msa_mask=feats['extra_msa_mask'], chunk_size=self.globals.chunk_size, use_lma=self.globals.use_lma, pair_mask=pair_mask, _mask_trans=self.config._mask_trans)
                del input_tensors
            else:
                z = self.extra_msa_stack(a, z, msa_mask=feats['extra_msa_mask'], chunk_size=self.globals.chunk_size, use_lma=self.globals.use_lma, pair_mask=pair_mask, inplace_safe=inplace_safe, _mask_trans=self.config._mask_trans)
        if self.globals.offload_inference:
            input_tensors = [m, z]
            del m, z
            m, z, s = self.evoformer._forward_offload(input_tensors, msa_mask=msa_mask, pair_mask=pair_mask, chunk_size=self.globals.chunk_size, use_lma=self.globals.use_lma, _mask_trans=self.config._mask_trans)
            del input_tensors
        else:
            m, z, s = self.evoformer(m, z, msa_mask=msa_mask, pair_mask=pair_mask, chunk_size=self.globals.chunk_size, use_lma=self.globals.use_lma, use_flash=self.globals.use_flash, inplace_safe=inplace_safe, _mask_trans=self.config._mask_trans)
        outputs['msa'] = m[..., :n_seq, :, :]
        outputs['pair'] = z
        outputs['single'] = s
        del z
        outputs['sm'] = self.structure_module(outputs, feats['aatype'], mask=feats['seq_mask'], inplace_safe=inplace_safe, _offload_inference=self.globals.offload_inference)
        outputs['final_atom_positions'] = atom14_to_atom37(outputs['sm']['positions'][-1], feats)
        outputs['final_atom_mask'] = feats['atom37_atom_exists']
        outputs['final_affine_tensor'] = outputs['sm']['frames'][-1]
        m_1_prev = m[..., 0, :, :]
        z_prev = outputs['pair']
        x_prev = outputs['final_atom_positions']
        return outputs, m_1_prev, z_prev, x_prev

    def forward(self, batch):
        """
        Args:
            batch:
                Dictionary of arguments outlined in Algorithm 2. Keys must
                include the official names of the features in the
                supplement subsection 1.2.9.

                The final dimension of each input must have length equal to
                the number of recycling iterations.

                Features (without the recycling dimension):

                    "aatype" ([*, N_res]):
                        Contrary to the supplement, this tensor of residue
                        indices is not one-hot.
                    "target_feat" ([*, N_res, C_tf])
                        One-hot encoding of the target sequence. C_tf is
                        config.model.input_embedder.tf_dim.
                    "residue_index" ([*, N_res])
                        Tensor whose final dimension consists of
                        consecutive indices from 0 to N_res.
                    "msa_feat" ([*, N_seq, N_res, C_msa])
                        MSA features, constructed as in the supplement.
                        C_msa is config.model.input_embedder.msa_dim.
                    "seq_mask" ([*, N_res])
                        1-D sequence mask
                    "msa_mask" ([*, N_seq, N_res])
                        MSA mask
                    "pair_mask" ([*, N_res, N_res])
                        2-D pair mask
                    "extra_msa_mask" ([*, N_extra, N_res])
                        Extra MSA mask
                    "template_mask" ([*, N_templ])
                        Template mask (on the level of templates, not
                        residues)
                    "template_aatype" ([*, N_templ, N_res])
                        Tensor of template residue indices (indices greater
                        than 19 are clamped to 20 (Unknown))
                    "template_all_atom_positions"
                        ([*, N_templ, N_res, 37, 3])
                        Template atom coordinates in atom37 format
                    "template_all_atom_mask" ([*, N_templ, N_res, 37])
                        Template atom coordinate mask
                    "template_pseudo_beta" ([*, N_templ, N_res, 3])
                        Positions of template carbon "pseudo-beta" atoms
                        (i.e. C_beta for all residues but glycine, for
                        for which C_alpha is used instead)
                    "template_pseudo_beta_mask" ([*, N_templ, N_res])
                        Pseudo-beta mask
        """
        m_1_prev, z_prev, x_prev = None, None, None
        prevs = [m_1_prev, z_prev, x_prev]
        is_grad_enabled = torch.is_grad_enabled()
        num_iters = batch['aatype'].shape[-1]
        for cycle_no in range(num_iters):
            fetch_cur_batch = lambda t: t[..., cycle_no]
            feats = tensor_tree_map(fetch_cur_batch, batch)
            is_final_iter = cycle_no == num_iters - 1
            with torch.set_grad_enabled(is_grad_enabled and is_final_iter):
                if is_final_iter:
                    if torch.is_autocast_enabled():
                        torch.clear_autocast_cache()
                outputs, m_1_prev, z_prev, x_prev = self.iteration(feats, prevs, _recycle=num_iters > 1)
                if not is_final_iter:
                    del outputs
                    prevs = [m_1_prev, z_prev, x_prev]
                    del m_1_prev, z_prev, x_prev
        outputs.update(self.aux_heads(outputs))
        return outputs


def compute_renamed_ground_truth(batch: Dict[str, torch.Tensor], atom14_pred_positions: torch.Tensor, eps=1e-10) ->Dict[str, torch.Tensor]:
    """
    Find optimal renaming of ground truth based on the predicted positions.

    Alg. 26 "renameSymmetricGroundTruthAtoms"

    This renamed ground truth is then used for all losses,
    such that each loss moves the atoms in the same direction.

    Args:
      batch: Dictionary containing:
        * atom14_gt_positions: Ground truth positions.
        * atom14_alt_gt_positions: Ground truth positions with renaming swaps.
        * atom14_atom_is_ambiguous: 1.0 for atoms that are affected by
            renaming swaps.
        * atom14_gt_exists: Mask for which atoms exist in ground truth.
        * atom14_alt_gt_exists: Mask for which atoms exist in ground truth
            after renaming.
        * atom14_atom_exists: Mask for whether each atom is part of the given
            amino acid type.
      atom14_pred_positions: Array of atom positions in global frame with shape
    Returns:
      Dictionary containing:
        alt_naming_is_better: Array with 1.0 where alternative swap is better.
        renamed_atom14_gt_positions: Array of optimal ground truth positions
          after renaming swaps are performed.
        renamed_atom14_gt_exists: Mask after renaming swap is performed.
    """
    pred_dists = torch.sqrt(eps + torch.sum((atom14_pred_positions[..., None, :, None, :] - atom14_pred_positions[..., None, :, None, :, :]) ** 2, dim=-1))
    atom14_gt_positions = batch['atom14_gt_positions']
    gt_dists = torch.sqrt(eps + torch.sum((atom14_gt_positions[..., None, :, None, :] - atom14_gt_positions[..., None, :, None, :, :]) ** 2, dim=-1))
    atom14_alt_gt_positions = batch['atom14_alt_gt_positions']
    alt_gt_dists = torch.sqrt(eps + torch.sum((atom14_alt_gt_positions[..., None, :, None, :] - atom14_alt_gt_positions[..., None, :, None, :, :]) ** 2, dim=-1))
    lddt = torch.sqrt(eps + (pred_dists - gt_dists) ** 2)
    alt_lddt = torch.sqrt(eps + (pred_dists - alt_gt_dists) ** 2)
    atom14_gt_exists = batch['atom14_gt_exists']
    atom14_atom_is_ambiguous = batch['atom14_atom_is_ambiguous']
    mask = atom14_gt_exists[..., None, :, None] * atom14_atom_is_ambiguous[..., None, :, None] * atom14_gt_exists[..., None, :, None, :] * (1.0 - atom14_atom_is_ambiguous[..., None, :, None, :])
    per_res_lddt = torch.sum(mask * lddt, dim=(-1, -2, -3))
    alt_per_res_lddt = torch.sum(mask * alt_lddt, dim=(-1, -2, -3))
    fp_type = atom14_pred_positions.dtype
    alt_naming_is_better = (alt_per_res_lddt < per_res_lddt).type(fp_type)
    renamed_atom14_gt_positions = (1.0 - alt_naming_is_better[..., None, None]) * atom14_gt_positions + alt_naming_is_better[..., None, None] * atom14_alt_gt_positions
    renamed_atom14_gt_mask = (1.0 - alt_naming_is_better[..., None]) * atom14_gt_exists + alt_naming_is_better[..., None] * batch['atom14_alt_gt_exists']
    return {'alt_naming_is_better': alt_naming_is_better, 'renamed_atom14_gt_positions': renamed_atom14_gt_positions, 'renamed_atom14_gt_exists': renamed_atom14_gt_mask}


def softmax_cross_entropy(logits, labels):
    loss = -1 * torch.sum(labels * torch.nn.functional.log_softmax(logits, dim=-1), dim=-1)
    return loss


def distogram_loss(logits, pseudo_beta, pseudo_beta_mask, min_bin=2.3125, max_bin=21.6875, no_bins=64, eps=1e-06, **kwargs):
    boundaries = torch.linspace(min_bin, max_bin, no_bins - 1, device=logits.device)
    boundaries = boundaries ** 2
    dists = torch.sum((pseudo_beta[..., None, :] - pseudo_beta[..., None, :, :]) ** 2, dim=-1, keepdims=True)
    true_bins = torch.sum(dists > boundaries, dim=-1)
    errors = softmax_cross_entropy(logits, torch.nn.functional.one_hot(true_bins, no_bins))
    square_mask = pseudo_beta_mask[..., None] * pseudo_beta_mask[..., None, :]
    denom = eps + torch.sum(square_mask, dim=(-1, -2))
    mean = errors * square_mask
    mean = torch.sum(mean, dim=-1)
    mean = mean / denom[..., None]
    mean = torch.sum(mean, dim=-1)
    mean = torch.mean(mean)
    return mean


def sigmoid_cross_entropy(logits, labels):
    logits_dtype = logits.dtype
    logits = logits.double()
    labels = labels.double()
    log_p = torch.nn.functional.logsigmoid(logits)
    log_not_p = torch.nn.functional.logsigmoid(-1 * logits)
    loss = -1.0 * labels * log_p - (1.0 - labels) * log_not_p
    loss = loss
    return loss


def experimentally_resolved_loss(logits: torch.Tensor, atom37_atom_exists: torch.Tensor, all_atom_mask: torch.Tensor, resolution: torch.Tensor, min_resolution: float, max_resolution: float, eps: float=1e-08, **kwargs) ->torch.Tensor:
    errors = sigmoid_cross_entropy(logits, all_atom_mask)
    loss = torch.sum(errors * atom37_atom_exists, dim=-1)
    loss = loss / (eps + torch.sum(atom37_atom_exists, dim=(-1, -2)))
    loss = torch.sum(loss, dim=-1)
    loss = loss * ((resolution >= min_resolution) & (resolution <= max_resolution))
    loss = torch.mean(loss)
    return loss


def backbone_loss(backbone_rigid_tensor: torch.Tensor, backbone_rigid_mask: torch.Tensor, traj: torch.Tensor, use_clamped_fape: Optional[torch.Tensor]=None, clamp_distance: float=10.0, loss_unit_distance: float=10.0, eps: float=0.0001, **kwargs) ->torch.Tensor:
    pred_aff = Rigid.from_tensor_7(traj)
    pred_aff = Rigid(Rotation(rot_mats=pred_aff.get_rots().get_rot_mats(), quats=None), pred_aff.get_trans())
    gt_aff = Rigid.from_tensor_4x4(backbone_rigid_tensor)
    fape_loss = compute_fape(pred_aff, gt_aff[None], backbone_rigid_mask[None], pred_aff.get_trans(), gt_aff[None].get_trans(), backbone_rigid_mask[None], l1_clamp_distance=clamp_distance, length_scale=loss_unit_distance, eps=eps)
    if use_clamped_fape is not None:
        unclamped_fape_loss = compute_fape(pred_aff, gt_aff[None], backbone_rigid_mask[None], pred_aff.get_trans(), gt_aff[None].get_trans(), backbone_rigid_mask[None], l1_clamp_distance=None, length_scale=loss_unit_distance, eps=eps)
        fape_loss = fape_loss * use_clamped_fape + unclamped_fape_loss * (1 - use_clamped_fape)
    fape_loss = torch.mean(fape_loss)
    return fape_loss


def sidechain_loss(sidechain_frames: torch.Tensor, sidechain_atom_pos: torch.Tensor, rigidgroups_gt_frames: torch.Tensor, rigidgroups_alt_gt_frames: torch.Tensor, rigidgroups_gt_exists: torch.Tensor, renamed_atom14_gt_positions: torch.Tensor, renamed_atom14_gt_exists: torch.Tensor, alt_naming_is_better: torch.Tensor, clamp_distance: float=10.0, length_scale: float=10.0, eps: float=0.0001, **kwargs) ->torch.Tensor:
    renamed_gt_frames = (1.0 - alt_naming_is_better[..., None, None, None]) * rigidgroups_gt_frames + alt_naming_is_better[..., None, None, None] * rigidgroups_alt_gt_frames
    sidechain_frames = sidechain_frames[-1]
    batch_dims = sidechain_frames.shape[:-4]
    sidechain_frames = sidechain_frames.view(*batch_dims, -1, 4, 4)
    sidechain_frames = Rigid.from_tensor_4x4(sidechain_frames)
    renamed_gt_frames = renamed_gt_frames.view(*batch_dims, -1, 4, 4)
    renamed_gt_frames = Rigid.from_tensor_4x4(renamed_gt_frames)
    rigidgroups_gt_exists = rigidgroups_gt_exists.reshape(*batch_dims, -1)
    sidechain_atom_pos = sidechain_atom_pos[-1]
    sidechain_atom_pos = sidechain_atom_pos.view(*batch_dims, -1, 3)
    renamed_atom14_gt_positions = renamed_atom14_gt_positions.view(*batch_dims, -1, 3)
    renamed_atom14_gt_exists = renamed_atom14_gt_exists.view(*batch_dims, -1)
    fape = compute_fape(sidechain_frames, renamed_gt_frames, rigidgroups_gt_exists, sidechain_atom_pos, renamed_atom14_gt_positions, renamed_atom14_gt_exists, l1_clamp_distance=clamp_distance, length_scale=length_scale, eps=eps)
    return fape


def between_residue_bond_loss(pred_atom_positions: torch.Tensor, pred_atom_mask: torch.Tensor, residue_index: torch.Tensor, aatype: torch.Tensor, tolerance_factor_soft=12.0, tolerance_factor_hard=12.0, eps=1e-06) ->Dict[str, torch.Tensor]:
    """Flat-bottom loss to penalize structural violations between residues.

    This is a loss penalizing any violation of the geometry around the peptide
    bond between consecutive amino acids. This loss corresponds to
    Jumper et al. (2021) Suppl. Sec. 1.9.11, eq 44, 45.

    Args:
      pred_atom_positions: Atom positions in atom37/14 representation
      pred_atom_mask: Atom mask in atom37/14 representation
      residue_index: Residue index for given amino acid, this is assumed to be
        monotonically increasing.
      aatype: Amino acid type of given residue
      tolerance_factor_soft: soft tolerance factor measured in standard deviations
        of pdb distributions
      tolerance_factor_hard: hard tolerance factor measured in standard deviations
        of pdb distributions

    Returns:
      Dict containing:
        * 'c_n_loss_mean': Loss for peptide bond length violations
        * 'ca_c_n_loss_mean': Loss for violations of bond angle around C spanned
            by CA, C, N
        * 'c_n_ca_loss_mean': Loss for violations of bond angle around N spanned
            by C, N, CA
        * 'per_residue_loss_sum': sum of all losses for each residue
        * 'per_residue_violation_mask': mask denoting all residues with violation
            present.
    """
    this_ca_pos = pred_atom_positions[..., :-1, 1, :]
    this_ca_mask = pred_atom_mask[..., :-1, 1]
    this_c_pos = pred_atom_positions[..., :-1, 2, :]
    this_c_mask = pred_atom_mask[..., :-1, 2]
    next_n_pos = pred_atom_positions[..., 1:, 0, :]
    next_n_mask = pred_atom_mask[..., 1:, 0]
    next_ca_pos = pred_atom_positions[..., 1:, 1, :]
    next_ca_mask = pred_atom_mask[..., 1:, 1]
    has_no_gap_mask = residue_index[..., 1:] - residue_index[..., :-1] == 1.0
    c_n_bond_length = torch.sqrt(eps + torch.sum((this_c_pos - next_n_pos) ** 2, dim=-1))
    next_is_proline = aatype[..., 1:] == residue_constants.resname_to_idx['PRO']
    gt_length = ~next_is_proline * residue_constants.between_res_bond_length_c_n[0] + next_is_proline * residue_constants.between_res_bond_length_c_n[1]
    gt_stddev = ~next_is_proline * residue_constants.between_res_bond_length_stddev_c_n[0] + next_is_proline * residue_constants.between_res_bond_length_stddev_c_n[1]
    c_n_bond_length_error = torch.sqrt(eps + (c_n_bond_length - gt_length) ** 2)
    c_n_loss_per_residue = torch.nn.functional.relu(c_n_bond_length_error - tolerance_factor_soft * gt_stddev)
    mask = this_c_mask * next_n_mask * has_no_gap_mask
    c_n_loss = torch.sum(mask * c_n_loss_per_residue, dim=-1) / (torch.sum(mask, dim=-1) + eps)
    c_n_violation_mask = mask * (c_n_bond_length_error > tolerance_factor_hard * gt_stddev)
    ca_c_bond_length = torch.sqrt(eps + torch.sum((this_ca_pos - this_c_pos) ** 2, dim=-1))
    n_ca_bond_length = torch.sqrt(eps + torch.sum((next_n_pos - next_ca_pos) ** 2, dim=-1))
    c_ca_unit_vec = (this_ca_pos - this_c_pos) / ca_c_bond_length[..., None]
    c_n_unit_vec = (next_n_pos - this_c_pos) / c_n_bond_length[..., None]
    n_ca_unit_vec = (next_ca_pos - next_n_pos) / n_ca_bond_length[..., None]
    ca_c_n_cos_angle = torch.sum(c_ca_unit_vec * c_n_unit_vec, dim=-1)
    gt_angle = residue_constants.between_res_cos_angles_ca_c_n[0]
    gt_stddev = residue_constants.between_res_bond_length_stddev_c_n[0]
    ca_c_n_cos_angle_error = torch.sqrt(eps + (ca_c_n_cos_angle - gt_angle) ** 2)
    ca_c_n_loss_per_residue = torch.nn.functional.relu(ca_c_n_cos_angle_error - tolerance_factor_soft * gt_stddev)
    mask = this_ca_mask * this_c_mask * next_n_mask * has_no_gap_mask
    ca_c_n_loss = torch.sum(mask * ca_c_n_loss_per_residue, dim=-1) / (torch.sum(mask, dim=-1) + eps)
    ca_c_n_violation_mask = mask * (ca_c_n_cos_angle_error > tolerance_factor_hard * gt_stddev)
    c_n_ca_cos_angle = torch.sum(-c_n_unit_vec * n_ca_unit_vec, dim=-1)
    gt_angle = residue_constants.between_res_cos_angles_c_n_ca[0]
    gt_stddev = residue_constants.between_res_cos_angles_c_n_ca[1]
    c_n_ca_cos_angle_error = torch.sqrt(eps + torch.square(c_n_ca_cos_angle - gt_angle))
    c_n_ca_loss_per_residue = torch.nn.functional.relu(c_n_ca_cos_angle_error - tolerance_factor_soft * gt_stddev)
    mask = this_c_mask * next_n_mask * next_ca_mask * has_no_gap_mask
    c_n_ca_loss = torch.sum(mask * c_n_ca_loss_per_residue, dim=-1) / (torch.sum(mask, dim=-1) + eps)
    c_n_ca_violation_mask = mask * (c_n_ca_cos_angle_error > tolerance_factor_hard * gt_stddev)
    per_residue_loss_sum = c_n_loss_per_residue + ca_c_n_loss_per_residue + c_n_ca_loss_per_residue
    per_residue_loss_sum = 0.5 * (torch.nn.functional.pad(per_residue_loss_sum, (0, 1)) + torch.nn.functional.pad(per_residue_loss_sum, (1, 0)))
    violation_mask = torch.max(torch.stack([c_n_violation_mask, ca_c_n_violation_mask, c_n_ca_violation_mask], dim=-2), dim=-2)[0]
    violation_mask = torch.maximum(torch.nn.functional.pad(violation_mask, (0, 1)), torch.nn.functional.pad(violation_mask, (1, 0)))
    return {'c_n_loss_mean': c_n_loss, 'ca_c_n_loss_mean': ca_c_n_loss, 'c_n_ca_loss_mean': c_n_ca_loss, 'per_residue_loss_sum': per_residue_loss_sum, 'per_residue_violation_mask': violation_mask}


def between_residue_clash_loss(atom14_pred_positions: torch.Tensor, atom14_atom_exists: torch.Tensor, atom14_atom_radius: torch.Tensor, residue_index: torch.Tensor, overlap_tolerance_soft=1.5, overlap_tolerance_hard=1.5, eps=1e-10) ->Dict[str, torch.Tensor]:
    """Loss to penalize steric clashes between residues.

    This is a loss penalizing any steric clashes due to non bonded atoms in
    different peptides coming too close. This loss corresponds to the part with
    different residues of
    Jumper et al. (2021) Suppl. Sec. 1.9.11, eq 46.

    Args:
      atom14_pred_positions: Predicted positions of atoms in
        global prediction frame
      atom14_atom_exists: Mask denoting whether atom at positions exists for given
        amino acid type
      atom14_atom_radius: Van der Waals radius for each atom.
      residue_index: Residue index for given amino acid.
      overlap_tolerance_soft: Soft tolerance factor.
      overlap_tolerance_hard: Hard tolerance factor.

    Returns:
      Dict containing:
        * 'mean_loss': average clash loss
        * 'per_atom_loss_sum': sum of all clash losses per atom, shape (N, 14)
        * 'per_atom_clash_mask': mask whether atom clashes with any other atom
            shape (N, 14)
    """
    fp_type = atom14_pred_positions.dtype
    dists = torch.sqrt(eps + torch.sum((atom14_pred_positions[..., :, None, :, None, :] - atom14_pred_positions[..., None, :, None, :, :]) ** 2, dim=-1))
    dists_mask = (atom14_atom_exists[..., :, None, :, None] * atom14_atom_exists[..., None, :, None, :]).type(fp_type)
    dists_mask = dists_mask * (residue_index[..., :, None, None, None] < residue_index[..., None, :, None, None])
    c_one_hot = torch.nn.functional.one_hot(residue_index.new_tensor(2), num_classes=14)
    c_one_hot = c_one_hot.reshape(*((1,) * len(residue_index.shape[:-1])), *c_one_hot.shape)
    c_one_hot = c_one_hot.type(fp_type)
    n_one_hot = torch.nn.functional.one_hot(residue_index.new_tensor(0), num_classes=14)
    n_one_hot = n_one_hot.reshape(*((1,) * len(residue_index.shape[:-1])), *n_one_hot.shape)
    n_one_hot = n_one_hot.type(fp_type)
    neighbour_mask = residue_index[..., :, None, None, None] + 1 == residue_index[..., None, :, None, None]
    c_n_bonds = neighbour_mask * c_one_hot[..., None, None, :, None] * n_one_hot[..., None, None, None, :]
    dists_mask = dists_mask * (1.0 - c_n_bonds)
    cys = residue_constants.restype_name_to_atom14_names['CYS']
    cys_sg_idx = cys.index('SG')
    cys_sg_idx = residue_index.new_tensor(cys_sg_idx)
    cys_sg_idx = cys_sg_idx.reshape(*((1,) * len(residue_index.shape[:-1])), 1).squeeze(-1)
    cys_sg_one_hot = torch.nn.functional.one_hot(cys_sg_idx, num_classes=14)
    disulfide_bonds = cys_sg_one_hot[..., None, None, :, None] * cys_sg_one_hot[..., None, None, None, :]
    dists_mask = dists_mask * (1.0 - disulfide_bonds)
    dists_lower_bound = dists_mask * (atom14_atom_radius[..., :, None, :, None] + atom14_atom_radius[..., None, :, None, :])
    dists_to_low_error = dists_mask * torch.nn.functional.relu(dists_lower_bound - overlap_tolerance_soft - dists)
    mean_loss = torch.sum(dists_to_low_error) / (1e-06 + torch.sum(dists_mask))
    per_atom_loss_sum = torch.sum(dists_to_low_error, dim=(-4, -2)) + torch.sum(dists_to_low_error, axis=(-3, -1))
    clash_mask = dists_mask * (dists < dists_lower_bound - overlap_tolerance_hard)
    per_atom_clash_mask = torch.maximum(torch.amax(clash_mask, axis=(-4, -2)), torch.amax(clash_mask, axis=(-3, -1)))
    return {'mean_loss': mean_loss, 'per_atom_loss_sum': per_atom_loss_sum, 'per_atom_clash_mask': per_atom_clash_mask}


def within_residue_violations(atom14_pred_positions: torch.Tensor, atom14_atom_exists: torch.Tensor, atom14_dists_lower_bound: torch.Tensor, atom14_dists_upper_bound: torch.Tensor, tighten_bounds_for_loss=0.0, eps=1e-10) ->Dict[str, torch.Tensor]:
    """Loss to penalize steric clashes within residues.

    This is a loss penalizing any steric violations or clashes of non-bonded atoms
    in a given peptide. This loss corresponds to the part with
    the same residues of
    Jumper et al. (2021) Suppl. Sec. 1.9.11, eq 46.

    Args:
        atom14_pred_positions ([*, N, 14, 3]):
            Predicted positions of atoms in global prediction frame.
        atom14_atom_exists ([*, N, 14]):
            Mask denoting whether atom at positions exists for given
            amino acid type
        atom14_dists_lower_bound ([*, N, 14]):
            Lower bound on allowed distances.
        atom14_dists_upper_bound ([*, N, 14]):
            Upper bound on allowed distances
        tighten_bounds_for_loss ([*, N]):
            Extra factor to tighten loss

    Returns:
      Dict containing:
        * 'per_atom_loss_sum' ([*, N, 14]):
              sum of all clash losses per atom, shape
        * 'per_atom_clash_mask' ([*, N, 14]):
              mask whether atom clashes with any other atom shape
    """
    dists_masks = 1.0 - torch.eye(14, device=atom14_atom_exists.device)[None]
    dists_masks = dists_masks.reshape(*((1,) * len(atom14_atom_exists.shape[:-2])), *dists_masks.shape)
    dists_masks = atom14_atom_exists[..., :, :, None] * atom14_atom_exists[..., :, None, :] * dists_masks
    dists = torch.sqrt(eps + torch.sum((atom14_pred_positions[..., :, :, None, :] - atom14_pred_positions[..., :, None, :, :]) ** 2, dim=-1))
    dists_to_low_error = torch.nn.functional.relu(atom14_dists_lower_bound + tighten_bounds_for_loss - dists)
    dists_to_high_error = torch.nn.functional.relu(dists - (atom14_dists_upper_bound - tighten_bounds_for_loss))
    loss = dists_masks * (dists_to_low_error + dists_to_high_error)
    per_atom_loss_sum = torch.sum(loss, dim=-2) + torch.sum(loss, dim=-1)
    violations = dists_masks * ((dists < atom14_dists_lower_bound) | (dists > atom14_dists_upper_bound))
    per_atom_violations = torch.maximum(torch.max(violations, dim=-2)[0], torch.max(violations, axis=-1)[0])
    return {'per_atom_loss_sum': per_atom_loss_sum, 'per_atom_violations': per_atom_violations}


def find_structural_violations(batch: Dict[str, torch.Tensor], atom14_pred_positions: torch.Tensor, violation_tolerance_factor: float, clash_overlap_tolerance: float, **kwargs) ->Dict[str, torch.Tensor]:
    """Computes several checks for structural violations."""
    connection_violations = between_residue_bond_loss(pred_atom_positions=atom14_pred_positions, pred_atom_mask=batch['atom14_atom_exists'], residue_index=batch['residue_index'], aatype=batch['aatype'], tolerance_factor_soft=violation_tolerance_factor, tolerance_factor_hard=violation_tolerance_factor)
    atomtype_radius = [residue_constants.van_der_waals_radius[name[0]] for name in residue_constants.atom_types]
    atomtype_radius = atom14_pred_positions.new_tensor(atomtype_radius)
    atom14_atom_radius = batch['atom14_atom_exists'] * atomtype_radius[batch['residx_atom14_to_atom37']]
    between_residue_clashes = between_residue_clash_loss(atom14_pred_positions=atom14_pred_positions, atom14_atom_exists=batch['atom14_atom_exists'], atom14_atom_radius=atom14_atom_radius, residue_index=batch['residue_index'], overlap_tolerance_soft=clash_overlap_tolerance, overlap_tolerance_hard=clash_overlap_tolerance)
    restype_atom14_bounds = residue_constants.make_atom14_dists_bounds(overlap_tolerance=clash_overlap_tolerance, bond_length_tolerance_factor=violation_tolerance_factor)
    atom14_atom_exists = batch['atom14_atom_exists']
    atom14_dists_lower_bound = atom14_pred_positions.new_tensor(restype_atom14_bounds['lower_bound'])[batch['aatype']]
    atom14_dists_upper_bound = atom14_pred_positions.new_tensor(restype_atom14_bounds['upper_bound'])[batch['aatype']]
    residue_violations = within_residue_violations(atom14_pred_positions=atom14_pred_positions, atom14_atom_exists=batch['atom14_atom_exists'], atom14_dists_lower_bound=atom14_dists_lower_bound, atom14_dists_upper_bound=atom14_dists_upper_bound, tighten_bounds_for_loss=0.0)
    per_residue_violations_mask = torch.max(torch.stack([connection_violations['per_residue_violation_mask'], torch.max(between_residue_clashes['per_atom_clash_mask'], dim=-1)[0], torch.max(residue_violations['per_atom_violations'], dim=-1)[0]], dim=-1), dim=-1)[0]
    return {'between_residues': {'bonds_c_n_loss_mean': connection_violations['c_n_loss_mean'], 'angles_ca_c_n_loss_mean': connection_violations['ca_c_n_loss_mean'], 'angles_c_n_ca_loss_mean': connection_violations['c_n_ca_loss_mean'], 'connections_per_residue_loss_sum': connection_violations['per_residue_loss_sum'], 'connections_per_residue_violation_mask': connection_violations['per_residue_violation_mask'], 'clashes_mean_loss': between_residue_clashes['mean_loss'], 'clashes_per_atom_loss_sum': between_residue_clashes['per_atom_loss_sum'], 'clashes_per_atom_clash_mask': between_residue_clashes['per_atom_clash_mask']}, 'within_residues': {'per_atom_loss_sum': residue_violations['per_atom_loss_sum'], 'per_atom_violations': residue_violations['per_atom_violations']}, 'total_per_residue_violations_mask': per_residue_violations_mask}


def lddt(all_atom_pred_pos: torch.Tensor, all_atom_positions: torch.Tensor, all_atom_mask: torch.Tensor, cutoff: float=15.0, eps: float=1e-10, per_residue: bool=True) ->torch.Tensor:
    n = all_atom_mask.shape[-2]
    dmat_true = torch.sqrt(eps + torch.sum((all_atom_positions[..., None, :] - all_atom_positions[..., None, :, :]) ** 2, dim=-1))
    dmat_pred = torch.sqrt(eps + torch.sum((all_atom_pred_pos[..., None, :] - all_atom_pred_pos[..., None, :, :]) ** 2, dim=-1))
    dists_to_score = (dmat_true < cutoff) * all_atom_mask * permute_final_dims(all_atom_mask, (1, 0)) * (1.0 - torch.eye(n, device=all_atom_mask.device))
    dist_l1 = torch.abs(dmat_true - dmat_pred)
    score = (dist_l1 < 0.5).type(dist_l1.dtype) + (dist_l1 < 1.0).type(dist_l1.dtype) + (dist_l1 < 2.0).type(dist_l1.dtype) + (dist_l1 < 4.0).type(dist_l1.dtype)
    score = score * 0.25
    dims = (-1,) if per_residue else (-2, -1)
    norm = 1.0 / (eps + torch.sum(dists_to_score, dim=dims))
    score = norm * (eps + torch.sum(dists_to_score * score, dim=dims))
    return score


def lddt_loss(logits: torch.Tensor, all_atom_pred_pos: torch.Tensor, all_atom_positions: torch.Tensor, all_atom_mask: torch.Tensor, resolution: torch.Tensor, cutoff: float=15.0, no_bins: int=50, min_resolution: float=0.1, max_resolution: float=3.0, eps: float=1e-10, **kwargs) ->torch.Tensor:
    n = all_atom_mask.shape[-2]
    ca_pos = residue_constants.atom_order['CA']
    all_atom_pred_pos = all_atom_pred_pos[..., ca_pos, :]
    all_atom_positions = all_atom_positions[..., ca_pos, :]
    all_atom_mask = all_atom_mask[..., ca_pos:ca_pos + 1]
    score = lddt(all_atom_pred_pos, all_atom_positions, all_atom_mask, cutoff=cutoff, eps=eps)
    score = score.detach()
    bin_index = torch.floor(score * no_bins).long()
    bin_index = torch.clamp(bin_index, max=no_bins - 1)
    lddt_ca_one_hot = torch.nn.functional.one_hot(bin_index, num_classes=no_bins)
    errors = softmax_cross_entropy(logits, lddt_ca_one_hot)
    all_atom_mask = all_atom_mask.squeeze(-1)
    loss = torch.sum(errors * all_atom_mask, dim=-1) / (eps + torch.sum(all_atom_mask, dim=-1))
    loss = loss * ((resolution >= min_resolution) & (resolution <= max_resolution))
    loss = torch.mean(loss)
    return loss


def masked_msa_loss(logits, true_msa, bert_mask, eps=1e-08, **kwargs):
    """
    Computes BERT-style masked MSA loss. Implements subsection 1.9.9.

    Args:
        logits: [*, N_seq, N_res, 23] predicted residue distribution
        true_msa: [*, N_seq, N_res] true MSA
        bert_mask: [*, N_seq, N_res] MSA mask
    Returns:
        Masked MSA loss
    """
    errors = softmax_cross_entropy(logits, torch.nn.functional.one_hot(true_msa, num_classes=23))
    loss = errors * bert_mask
    loss = torch.sum(loss, dim=-1)
    scale = 0.5
    denom = eps + torch.sum(scale * bert_mask, dim=(-1, -2))
    loss = loss / denom[..., None]
    loss = torch.sum(loss, dim=-1)
    loss = loss * scale
    loss = torch.mean(loss)
    return loss


def masked_mean(mask, value, dim, eps=0.0001):
    mask = mask.expand(*value.shape)
    return torch.sum(mask * value, dim=dim) / (eps + torch.sum(mask, dim=dim))


def supervised_chi_loss(angles_sin_cos: torch.Tensor, unnormalized_angles_sin_cos: torch.Tensor, aatype: torch.Tensor, seq_mask: torch.Tensor, chi_mask: torch.Tensor, chi_angles_sin_cos: torch.Tensor, chi_weight: float, angle_norm_weight: float, eps=1e-06, **kwargs) ->torch.Tensor:
    """
        Implements Algorithm 27 (torsionAngleLoss)

        Args:
            angles_sin_cos:
                [*, N, 7, 2] predicted angles
            unnormalized_angles_sin_cos:
                The same angles, but unnormalized
            aatype:
                [*, N] residue indices
            seq_mask:
                [*, N] sequence mask
            chi_mask:
                [*, N, 7] angle mask
            chi_angles_sin_cos:
                [*, N, 7, 2] ground truth angles
            chi_weight:
                Weight for the angle component of the loss
            angle_norm_weight:
                Weight for the normalization component of the loss
        Returns:
            [*] loss tensor
    """
    pred_angles = angles_sin_cos[..., 3:, :]
    residue_type_one_hot = torch.nn.functional.one_hot(aatype, residue_constants.restype_num + 1)
    chi_pi_periodic = torch.einsum('...ij,jk->ik', residue_type_one_hot.type(angles_sin_cos.dtype), angles_sin_cos.new_tensor(residue_constants.chi_pi_periodic))
    true_chi = chi_angles_sin_cos[None]
    shifted_mask = (1 - 2 * chi_pi_periodic).unsqueeze(-1)
    true_chi_shifted = shifted_mask * true_chi
    sq_chi_error = torch.sum((true_chi - pred_angles) ** 2, dim=-1)
    sq_chi_error_shifted = torch.sum((true_chi_shifted - pred_angles) ** 2, dim=-1)
    sq_chi_error = torch.minimum(sq_chi_error, sq_chi_error_shifted)
    sq_chi_error = sq_chi_error.permute(*range(len(sq_chi_error.shape))[1:-2], 0, -2, -1)
    sq_chi_loss = masked_mean(chi_mask[..., None, :, :], sq_chi_error, dim=(-1, -2, -3))
    loss = chi_weight * sq_chi_loss
    angle_norm = torch.sqrt(torch.sum(unnormalized_angles_sin_cos ** 2, dim=-1) + eps)
    norm_error = torch.abs(angle_norm - 1.0)
    norm_error = norm_error.permute(*range(len(norm_error.shape))[1:-2], 0, -2, -1)
    angle_norm_loss = masked_mean(seq_mask[..., None, :, None], norm_error, dim=(-1, -2, -3))
    loss = loss + angle_norm_weight * angle_norm_loss
    loss = torch.mean(loss)
    return loss


def tm_loss(logits, final_affine_tensor, backbone_rigid_tensor, backbone_rigid_mask, resolution, max_bin=31, no_bins=64, min_resolution: float=0.1, max_resolution: float=3.0, eps=1e-08, **kwargs):
    pred_affine = Rigid.from_tensor_7(final_affine_tensor)
    backbone_rigid = Rigid.from_tensor_4x4(backbone_rigid_tensor)

    def _points(affine):
        pts = affine.get_trans()[..., None, :, :]
        return affine.invert()[..., None].apply(pts)
    sq_diff = torch.sum((_points(pred_affine) - _points(backbone_rigid)) ** 2, dim=-1)
    sq_diff = sq_diff.detach()
    boundaries = torch.linspace(0, max_bin, steps=no_bins - 1, device=logits.device)
    boundaries = boundaries ** 2
    true_bins = torch.sum(sq_diff[..., None] > boundaries, dim=-1)
    errors = softmax_cross_entropy(logits, torch.nn.functional.one_hot(true_bins, no_bins))
    square_mask = backbone_rigid_mask[..., None] * backbone_rigid_mask[..., None, :]
    loss = torch.sum(errors * square_mask, dim=-1)
    scale = 0.5
    denom = eps + torch.sum(scale * square_mask, dim=(-1, -2))
    loss = loss / denom[..., None]
    loss = torch.sum(loss, dim=-1)
    loss = loss * scale
    loss = loss * ((resolution >= min_resolution) & (resolution <= max_resolution))
    loss = torch.mean(loss)
    return loss


def violation_loss(violations: Dict[str, torch.Tensor], atom14_atom_exists: torch.Tensor, eps=1e-06, **kwargs) ->torch.Tensor:
    num_atoms = torch.sum(atom14_atom_exists)
    l_clash = torch.sum(violations['between_residues']['clashes_per_atom_loss_sum'] + violations['within_residues']['per_atom_loss_sum'])
    l_clash = l_clash / (eps + num_atoms)
    loss = violations['between_residues']['bonds_c_n_loss_mean'] + violations['between_residues']['angles_ca_c_n_loss_mean'] + violations['between_residues']['angles_c_n_ca_loss_mean'] + l_clash
    return loss


class AlphaFoldLoss(nn.Module):
    """Aggregation of the various losses described in the supplement"""

    def __init__(self, config):
        super(AlphaFoldLoss, self).__init__()
        self.config = config

    def forward(self, out, batch, _return_breakdown=False):
        if 'violation' not in out.keys():
            out['violation'] = find_structural_violations(batch, out['sm']['positions'][-1], **self.config.violation)
        if 'renamed_atom14_gt_positions' not in out.keys():
            batch.update(compute_renamed_ground_truth(batch, out['sm']['positions'][-1]))
        loss_fns = {'distogram': lambda : distogram_loss(logits=out['distogram_logits'], **{**batch, **self.config.distogram}), 'experimentally_resolved': lambda : experimentally_resolved_loss(logits=out['experimentally_resolved_logits'], **{**batch, **self.config.experimentally_resolved}), 'fape': lambda : fape_loss(out, batch, self.config.fape), 'plddt_loss': lambda : lddt_loss(logits=out['lddt_logits'], all_atom_pred_pos=out['final_atom_positions'], **{**batch, **self.config.plddt_loss}), 'masked_msa': lambda : masked_msa_loss(logits=out['masked_msa_logits'], **{**batch, **self.config.masked_msa}), 'supervised_chi': lambda : supervised_chi_loss(out['sm']['angles'], out['sm']['unnormalized_angles'], **{**batch, **self.config.supervised_chi}), 'violation': lambda : violation_loss(out['violation'], **batch)}
        if self.config.tm.enabled:
            loss_fns['tm'] = lambda : tm_loss(logits=out['tm_logits'], **{**batch, **out, **self.config.tm})
        cum_loss = 0.0
        losses = {}
        for loss_name, loss_fn in loss_fns.items():
            weight = self.config[loss_name].weight
            loss = loss_fn()
            if torch.isnan(loss) or torch.isinf(loss):
                logging.warning(f'{loss_name} loss is NaN. Skipping...')
                loss = loss.new_tensor(0.0, requires_grad=True)
            cum_loss = cum_loss + weight * loss
            losses[loss_name] = loss.detach().clone()
        losses['unscaled_loss'] = cum_loss.detach().clone()
        seq_len = torch.mean(batch['seq_length'].float())
        crop_len = batch['aatype'].shape[-1]
        cum_loss = cum_loss * torch.sqrt(min(seq_len, crop_len))
        losses['loss'] = cum_loss.detach().clone()
        if not _return_breakdown:
            return cum_loss
        return cum_loss, losses


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AngleResnet,
     lambda: ([], {'c_in': 4, 'c_hidden': 4, 'no_blocks': 4, 'no_angles': 4, 'epsilon': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (AngleResnetBlock,
     lambda: ([], {'c_hidden': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BackboneUpdate,
     lambda: ([], {'c_s': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (DistogramHead,
     lambda: ([], {'c_z': 4, 'no_bins': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (DropoutColumnwise,
     lambda: ([], {'r': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DropoutRowwise,
     lambda: ([], {'r': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ExperimentallyResolvedHead,
     lambda: ([], {'c_s': 4, 'c_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ExtraMSAEmbedder,
     lambda: ([], {'c_in': 4, 'c_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InputEmbedder,
     lambda: ([], {'tf_dim': 4, 'msa_dim': 4, 'c_z': 4, 'c_m': 4, 'relpos_k': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (Linear,
     lambda: ([], {'in_dim': 4, 'out_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MaskedMSAHead,
     lambda: ([], {'c_m': 4, 'c_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (OuterProductMean,
     lambda: ([], {'c_m': 4, 'c_z': 4, 'c_hidden': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (StructureModuleTransitionLayer,
     lambda: ([], {'c': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TMScoreHead,
     lambda: ([], {'c_z': 4, 'no_bins': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TemplateAngleEmbedder,
     lambda: ([], {'c_in': 4, 'c_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TemplatePairEmbedder,
     lambda: ([], {'c_in': 4, 'c_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_aqlaboratory_openfold(_paritybench_base):
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

    def test_011(self):
        self._check(*TESTCASES[11])

    def test_012(self):
        self._check(*TESTCASES[12])

    def test_013(self):
        self._check(*TESTCASES[13])

    def test_014(self):
        self._check(*TESTCASES[14])

    def test_015(self):
        self._check(*TESTCASES[15])

