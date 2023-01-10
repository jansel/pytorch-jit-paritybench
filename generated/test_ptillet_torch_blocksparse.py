import sys
_module = sys.modules[__name__]
del sys
simple = _module
setup = _module
scratch = _module
test_attention = _module
test_matmul = _module
test_permute = _module
test_relu = _module
test_softmax = _module
utils = _module
torch_blocksparse = _module
attention = _module
deepspeedsparseselfattention = _module
matmul = _module
permute = _module
relu = _module
softmax = _module

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


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CppExtension


from torch.utils.cpp_extension import load_inline


from time import time


from collections import OrderedDict


import torch.nn as nn


from torch.nn.functional import *


from collections import namedtuple


import math


class MultiheadAttention(nn.modules.activation.MultiheadAttention):
    ops = dict()

    def get_ops(self, L):
        if L not in MultiheadAttention.ops:
            sparse_dot_sdd_nt = torch_blocksparse.MatMul(self.layout, self.block, 'sdd', trans_a=False, trans_b=True)
            sparse_dot_dsd_nn = torch_blocksparse.MatMul(self.layout, self.block, 'dsd', trans_a=False, trans_b=False)
            sparse_softmax = torch_blocksparse.Softmax(self.layout, self.block)
            MultiheadAttention.ops[L] = sparse_dot_sdd_nt, sparse_dot_dsd_nn, sparse_softmax
        return MultiheadAttention.ops[L]

    def __init__(self, embed_dim, num_heads, layout, block, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, key_padding_mask_mode='add', attn_mask_mode='mul'):
        if dropout != 0:
            raise NotImplementedError('dropout is not supported for now')
        super(MultiheadAttention, self).__init__(embed_dim, num_heads, dropout, bias, add_bias_kv, add_zero_attn, kdim, vdim)
        self.layout = layout
        self.block = block
        self.key_padding_mask_mode = key_padding_mask_mode
        self.attn_mask_mode = attn_mask_mode

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None):
        if query.shape != key.shape or key.shape != value.shape:
            raise NotImplementedError('only self-attention is supported for now')
        sparse_dot_sdd_nt, sparse_dot_dsd_nn, sparse_softmax = self.get_ops(query.shape[0])
        if not self._qkv_same_embed_dim:
            return multi_head_attention_forward(query, key, value, self.embed_dim, self.num_heads, self.in_proj_weight, self.in_proj_bias, self.bias_k, self.bias_v, self.add_zero_attn, self.dropout, self.out_proj.weight, self.out_proj.bias, sparse_dot_sdd_nt, sparse_dot_dsd_nn, sparse_softmax, training=self.training, key_padding_mask=key_padding_mask, need_weights=need_weights, attn_mask=attn_mask, use_separate_proj_weight=True, q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight, v_proj_weight=self.v_proj_weight, key_padding_mask_mode=self.key_padding_mask_mode, attn_mask_mode=self.attn_mask_mode)
        else:
            return multi_head_attention_forward(query, key, value, self.embed_dim, self.num_heads, self.in_proj_weight, self.in_proj_bias, self.bias_k, self.bias_v, self.add_zero_attn, self.dropout, self.out_proj.weight, self.out_proj.bias, sparse_dot_sdd_nt, sparse_dot_dsd_nn, sparse_softmax, training=self.training, key_padding_mask=key_padding_mask, need_weights=need_weights, attn_mask=attn_mask, key_padding_mask_mode=self.key_padding_mask_mode, attn_mask_mode=self.attn_mask_mode)


class SparsityConfig:

    def __init__(self, mode='fixed', block=16, stride=64, attention='unidirectional', numverts=1, vertsize=1):
        if mode != 'dense' and mode != 'fixed':
            raise NotImplementedError('only "dense" and "fixed" modes are supported for now')
        self.mode = mode
        self.block = block
        self.stride = stride
        if attention != 'unidirectional' and attention != 'bidirectional':
            raise NotImplementedError('only "uni/bi-directional" attentions are supported for now')
        self.attention = attention
        self.numverts = numverts
        self.vertsize = vertsize


class DeepSpeedSparseSelfAttention(nn.Module):

    @staticmethod
    def _set_s1_layout(layout, h, num_blocks, block_stride, attention):
        for i in range(0, num_blocks, block_stride):
            for j in range(i, i + block_stride):
                for k in range(i, j + 1 if attention == 'unidirectional' else i + block_stride):
                    layout[h, j, k] = 1
        return layout

    @staticmethod
    def _set_s2_layout(layout, h, num_blocks, block_stride, attention, numverts, vertsize):
        start = block_stride - (1 + h % numverts) * vertsize
        for i in range(0, num_blocks):
            end = i if attention == 'unidirectional' else num_blocks
            for j in range(start, end, block_stride):
                for k in range(j, min(j + vertsize, num_blocks)):
                    layout[h, i, k] = 1
        return layout

    @staticmethod
    def _make_layout(num_heads, num_blocks, mode, block_stride, attention, numverts, vertsize):
        if block_stride / vertsize != block_stride // vertsize:
            raise ValueError(f'Number of blocks in a stride window {block_stride} must be dividable by vertical block size {vertsize}')
        if numverts > block_stride / vertsize:
            raise ValueError(f'Number of layout versions {num_verts} cannot be larger than blocks in a stride window divided by vertical block size {block_stride} / {vertsize} = {block_stride / vertsize}')
        layout = torch.zeros((num_heads, num_blocks, num_blocks), dtype=torch.int64)
        if mode == 'dense':
            layout[:, :, :] = 1
        elif mode == 'fixed':
            for i in range(0, num_heads):
                layout = DeepSpeedSparseSelfAttention._set_s1_layout(layout, i, num_blocks, block_stride, attention)
                layout = DeepSpeedSparseSelfAttention._set_s2_layout(layout, i, num_blocks, block_stride, attention, numverts, vertsize)
        return layout
    ops = dict()

    def get_ops(self, H, L):
        if L not in DeepSpeedSparseSelfAttention.ops:
            spConfig = self.sparsity_config
            num_blocks = L // spConfig.block
            if num_blocks != L / spConfig.block:
                raise ValueError(f'Sequence length {L} must be dividable by block size {spConfig.block}')
            block_stride = spConfig.stride // spConfig.block
            if block_stride != spConfig.stride // spConfig.block:
                raise ValueError(f'Stride {spConfig.stride} must be dividable by block size {spConfig.block}')
            layout = DeepSpeedSparseSelfAttention._make_layout(H, num_blocks, spConfig.mode, block_stride, spConfig.attention, spConfig.numverts, spConfig.vertsize)
            sparse_dot_sdd_nt = torch_blocksparse.MatMul(layout, spConfig.block, 'sdd', trans_a=False, trans_b=True)
            sparse_dot_dsd_nn = torch_blocksparse.MatMul(layout, spConfig.block, 'dsd', trans_a=False, trans_b=False)
            sparse_softmax = torch_blocksparse.Softmax(layout, spConfig.block)
            DeepSpeedSparseSelfAttention.ops[L] = sparse_dot_sdd_nt, sparse_dot_dsd_nn, sparse_softmax
        return DeepSpeedSparseSelfAttention.ops[L]

    def __init__(self, sparsity_config=SparsityConfig(), key_padding_mask_mode='add', attn_mask_mode='mul'):
        super().__init__()
        self.sparsity_config = sparsity_config
        self.key_padding_mask_mode = key_padding_mask_mode
        self.attn_mask_mode = attn_mask_mode

    def transpose_key_for_scores(self, x, L):
        bsz, num_heads, seq_len, head_dim = x.size()
        if seq_len != L:
            return x.permute(0, 1, 3, 2)
        return x

    def transpose_mask_for_sparse(self, qtype, x, is_key_padding_mask=False):
        x = x.type(qtype)
        if is_key_padding_mask:
            xdim = x.dim()
            for d in range(xdim - 1, 0, -1):
                x = x.squeeze(dim=d)
            return x
        return x.squeeze()

    def forward(self, query, key, value, rpe=None, key_padding_mask=None, attn_mask=None):
        bsz, num_heads, tgt_len, head_dim = query.size()
        key = self.transpose_key_for_scores(key, tgt_len)
        if query.shape != key.shape or key.shape != value.shape:
            raise NotImplementedError('only self-attention is supported for now')
        if key_padding_mask is not None:
            key_padding_mask = self.transpose_mask_for_sparse(query.dtype, key_padding_mask, is_key_padding_mask=True)
        if attn_mask is not None:
            attn_mask = self.transpose_mask_for_sparse(query.dtype, attn_mask)
        sparse_dot_sdd_nt, sparse_dot_dsd_nn, sparse_softmax = self.get_ops(num_heads, tgt_len)
        scaling = float(head_dim) ** -0.5
        attn_output_weights = sparse_dot_sdd_nt(query, key)
        attn_output_weights = sparse_softmax(attn_output_weights, scale=scaling, rpe=rpe, key_padding_mask=key_padding_mask, attn_mask=attn_mask, key_padding_mask_mode=self.key_padding_mask_mode, attn_mask_mode=self.attn_mask_mode)
        attn_output = sparse_dot_dsd_nn(attn_output_weights, value)
        return attn_output


src = """
   __global__ void NAME (TYPE* X __readonly __noalias __aligned(16),
                         TYPE* Y __readonly __noalias __aligned(16),
                         int N, int C, int HW,
                         int stride_xn  __multipleof(M_STRIDE_XN), 
                         int stride_xc  __multipleof(M_STRIDE_XC), 
                         int stride_xhw __multipleof(M_STRIDE_XHW),
                         int stride_yn  __multipleof(M_STRIDE_YN),
                         int stride_yc  __multipleof(M_STRIDE_YC), 
                         int stride_yhw __multipleof(M_STRIDE_YHW)){
    // ranges
    int _rn   [TN ] = get_program_id(0)*TN  + 0 ... TN;
    int _rc   [TC ] = get_program_id(1)*TC  + 0 ... TC;
    int _rhw  [THW] = get_program_id(2)*THW + 0 ... THW;
    // broadcast
    int rn  [TN, TC, THW] = _rn[:, newaxis, newaxis];
    int rc  [TN, TC, THW] = _rc[newaxis, :, newaxis];
    int rhw [TN, TC, THW] = _rhw[newaxis, newaxis, :];
    // pointers to x
    TYPE *px  [TN, TC, THW] = X + rn*STRIDE_XN + rc*STRIDE_XC + rhw*STRIDE_XHW;
    TYPE *py  [TN, TC, THW] = Y + rn*STRIDE_YN + rc*STRIDE_YC + rhw*STRIDE_YHW;
    // check in bounds
    bool check[TN, TC, THW] = rn < N && rc < C && rhw < HW;
    *?(check)py = *?(check)px;
}
"""


class MatMul:

    def make_lut(self, dtype, device):
        key = dtype, device
        if key in self.lut_cache:
            return self.lut_cache[key]
        layout, block = self.layout, self.block
        step = 8 if dtype == torch.float32 else 16
        if self.mode == 'sdd':
            c_lut, c_num_locks, c_width, c_packs = _sparse_matmul.make_sdd_lut(layout, block, dtype, device)
        elif self.mode == 'dsd':
            c_lut, c_num_locks, c_width, c_packs = _sparse_matmul.make_dxx_lut(layout, block, step, not self.trans_a, device)
        elif self.mode == 'dds':
            c_lut, c_num_locks, c_width, c_packs = _sparse_matmul.make_dxx_lut(layout, block, step, self.trans_b, device)
        if self.mode == 'sdd':
            da_lut, da_num_locks, da_width, da_packs = _sparse_matmul.make_dxx_lut(layout, block, step, True, device)
        elif self.mode == 'dsd':
            da_lut, da_num_locks, da_width, da_packs = _sparse_matmul.make_sdd_lut(layout, block, dtype, device)
        elif self.mode == 'dds':
            da_lut, da_num_locks, da_width, da_packs = _sparse_matmul.make_dxx_lut(layout, block, step, not self.trans_b, device)
        if self.mode == 'sdd':
            db_lut, db_num_locks, db_width, db_packs = _sparse_matmul.make_dxx_lut(layout, block, step, False, device)
        elif self.mode == 'dsd':
            db_lut, db_num_locks, db_width, db_packs = _sparse_matmul.make_dxx_lut(layout, block, step, self.trans_a, device)
        elif self.mode == 'dds':
            db_lut, db_num_locks, db_width, db_packs = _sparse_matmul.make_sdd_lut(layout, block, dtype, device)
        self.lut_cache[key] = c_lut, c_num_locks, c_width, c_packs, da_lut, da_num_locks, da_width, da_packs, db_lut, db_num_locks, db_width, db_packs
        return self.lut_cache[key]

    def __init__(self, layout, block, mode, trans_a=False, trans_b=False, bench=False):
        if mode not in ['sdd', 'dsd', 'dds']:
            raise NotImplementedError('Supported modes are: sdd, dsd, dds')
        self.lut_cache = dict()
        self.trans_a = trans_a
        self.trans_b = trans_b
        self.mode = mode
        self.spdims = layout.shape
        self.block = block
        self.layout = layout
        self.bench = bench
        self.time_c = None
        self.time_da = None
        self.time_db = None

    @staticmethod
    def _pad_shape(x, is_sparse):
        max_dim = 3 if is_sparse else 4
        for i in range(max_dim - x.dim()):
            x = x.unsqueeze(0)
        return x

    def __call__(self, a, b):
        c_lut, c_num_locks, c_width, c_packs, da_lut, da_num_locks, da_width, da_packs, db_lut, db_num_locks, db_width, db_packs = self.make_lut(a.dtype, a.device)
        time_c = [None]
        time_da = [None]
        time_db = [None]
        a = MatMul._pad_shape(a, self.mode == 'dsd')
        b = MatMul._pad_shape(b, self.mode == 'dds')
        c = _sparse_matmul.apply(a, b, self.trans_a, self.trans_b, False, self.mode, self.spdims, self.block, c_lut, c_num_locks, c_width, c_packs, self.bench, time_c, da_lut, da_num_locks, da_width, da_packs, self.bench, time_da, db_lut, db_num_locks, db_width, db_packs, self.bench, time_db)
        self.time_c = time_c[0]
        self.time_da = time_da[0]
        self.time_db = time_db[0]
        return c


class Linear(torch.nn.Module):

    def __init__(self, in_features, out_features, block, layout, bench=False):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block = block
        self.weight = torch.nn.Parameter(torch.Tensor(layout.sum(), block, block))
        self.reset_parameters()
        self.matmul = MatMul(False, False, 'dds', layout, block, bench)

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, input):
        return self.matmul(input, self.weight)


class _permute(torch.autograd.Function):
    kernels = dict()

    @staticmethod
    def strides(N, C, H, W, order):
        return {'CHWN': [1, N * W * H, N * W, N], 'NCHW': [W * H * C, W * H, W, 1]}[order]

    @staticmethod
    def multiple_of(N):
        if N % 8 == 0:
            return 8
        if N % 4 == 0:
            return 4
        if N % 2 == 0:
            return 2
        return 1

    @staticmethod
    def do_work(x, in_order, out_order):
        x_inner_mul = _permute.multiple_of(x.shape['NCHW'.index(in_order[-1])])
        y_inner_mul = _permute.multiple_of(x.shape['NCHW'.index(out_order[-1])])
        key = x.dtype, in_order, out_order, x_inner_mul, y_inner_mul
        if key not in _permute.kernels:
            TN = [32] if in_order[-1] == 'N' or out_order[-1] == 'N' else 1
            TC = [32] if in_order[-1] == 'C' or out_order[-1] == 'C' else 1
            THW = [32] if in_order[-1] == 'W' or out_order[-1] == 'W' else 1
            defines = {'NAME': f'permute_{in_order}_{out_order}_{x_inner_mul}_{y_inner_mul}', 'TYPE': x.dtype, 'M_STRIDE_XN': 1 if in_order[-1] == 'N' else x_inner_mul, 'M_STRIDE_XC': 1 if in_order[-1] == 'N' else x_inner_mul, 'M_STRIDE_XHW': 1 if in_order[-1] == 'N' else x_inner_mul, 'M_STRIDE_YN': 1 if out_order[-1] == 'N' else y_inner_mul, 'M_STRIDE_YC': 1 if out_order[-1] == 'N' else y_inner_mul, 'M_STRIDE_YHW': 1 if out_order[-1] == 'N' else y_inner_mul, 'STRIDE_XN': 1 if in_order[-1] == 'N' else 'stride_xn', 'STRIDE_XC': 1 if in_order[-1] == 'C' else 'stride_xc', 'STRIDE_XHW': 1 if in_order[-1] == 'W' else 'stride_xhw', 'STRIDE_YN': 1 if out_order[-1] == 'N' else 'stride_yn', 'STRIDE_YC': 1 if out_order[-1] == 'C' else 'stride_yc', 'STRIDE_YHW': 1 if out_order[-1] == 'W' else 'stride_yhw', 'TN': TN, 'TC': TC, 'THW': THW}
            _permute.kernels[key] = triton.kernel(src, defines=defines, num_warps=[4])
        kernel = _permute.kernels[key]
        N, C, H, W = x.shape
        y = torch.empty_strided(x.shape, _permute.strides(N, C, H, W, out_order), device=x.device, dtype=x.dtype)
        stride_xn, stride_xc, _, stride_xhw = x.stride()
        stride_yn, stride_yc, _, stride_yhw = y.stride()
        grid = lambda opt: (triton.cdiv(N, opt.d('TN')), triton.cdiv(C, opt.d('TC')), triton.cdiv(H * W, opt.d('THW')))
        kernel(x, y, N, C, H * W, stride_xn, stride_xc, stride_xhw, stride_yn, stride_yc, stride_yhw, grid=grid)
        return y

    @staticmethod
    def forward(ctx, x, in_order, out_order):
        y = _permute.do_work(x, in_order, out_order)
        ctx.in_order = in_order
        ctx.out_order = out_order
        return y

    @staticmethod
    def backward(ctx, dy):
        dx = _permute.do_work(dy, ctx.out_order, ctx.in_order)
        return dx, None, None


class Permute(torch.nn.Module):

    def __init__(self, in_order, out_order):
        super(Permute, self).__init__()
        self.in_order = in_order
        self.out_order = out_order

    def forward(self, x):
        return _permute.apply(x, self.in_order, self.out_order)


class ReLU(torch.nn.Module):

    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x, scale, bias, residual):
        return relu(x, scale, bias, residual)

