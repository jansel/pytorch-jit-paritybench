import sys
_module = sys.modules[__name__]
del sys
data_utils = _module
eval = _module
mem_transformer = _module
train = _module
adaptive_softmax = _module
data_parallel = _module
exp_utils = _module
log_uniform_sampler = _module
proj_adaptive_softmax = _module
vocabulary = _module
fmoe = _module
balance = _module
distributed = _module
fastermoe = _module
config = _module
expert_utils = _module
schedule = _module
shadow_policy = _module
functions = _module
gates = _module
base_gate = _module
dc_gate = _module
faster_gate = _module
gshard_gate = _module
naive_gate = _module
noisy_gate = _module
swipe_gate = _module
switch_gate = _module
utils = _module
zero_gate = _module
layers = _module
linear = _module
megatron = _module
balance = _module
checkpoint = _module
distributed = _module
layers = _module
patch = _module
transformer = _module
utils = _module
setup = _module
benchmark_mlp = _module
moe = _module
test_ddp = _module
test_dp = _module
test_faster_gate = _module
test_faster_schedule = _module
test_faster_shadow = _module
test_gates = _module
test_local_exchange = _module
test_mimo = _module
test_numerical = _module
test_swipe = _module
test_zero = _module

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


from collections import Counter


from collections import OrderedDict


import numpy as np


import torch


import time


import math


import functools


import torch.nn as nn


import torch.nn.functional as F


import itertools


import torch.optim as optim


from collections import defaultdict


from torch.nn.parallel import DataParallel


from torch.nn.parallel._functions import Scatter


from torch.nn.parallel.parallel_apply import parallel_apply


from torch import nn


from torch._utils import _flatten_dense_tensors


from torch._utils import _unflatten_dense_tensors


from torch.autograd.function import Function


import torch.distributed as dist


from torch.autograd import Function


from torch.distributions.normal import Normal


import random


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CUDAExtension


from typing import Dict


from typing import List


from typing import Type


from typing import Union


from copy import deepcopy


class PositionalEmbedding(nn.Module):

    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()
        self.demb = demb
        inv_freq = 1 / 10000 ** (torch.arange(0.0, demb, 2.0) / demb)
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        if bsz is not None:
            return pos_emb[:, None, :].expand(-1, bsz, -1)
        else:
            return pos_emb[:, None, :]


class PositionwiseFF(nn.Module):

    def __init__(self, d_model, d_inner, dropout, pre_lnorm=False):
        super(PositionwiseFF, self).__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout
        self.CoreNet = nn.Sequential(nn.Linear(d_model, d_inner), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(d_inner, d_model), nn.Dropout(dropout))
        self.layer_norm = nn.LayerNorm(d_model)
        self.pre_lnorm = pre_lnorm

    def forward(self, inp):
        if self.pre_lnorm:
            core_out = self.CoreNet(self.layer_norm(inp))
            output = core_out + inp
        else:
            core_out = self.CoreNet(inp)
            output = self.layer_norm(inp + core_out)
        return output


class MultiHeadAttn(nn.Module):

    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0, pre_lnorm=False):
        super(MultiHeadAttn, self).__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout
        self.q_net = nn.Linear(d_model, n_head * d_head, bias=False)
        self.kv_net = nn.Linear(d_model, 2 * n_head * d_head, bias=False)
        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model)
        self.scale = 1 / d_head ** 0.5
        self.pre_lnorm = pre_lnorm

    def forward(self, h, attn_mask=None, mems=None):
        if mems is not None:
            c = torch.cat([mems, h], 0)
        else:
            c = h
        if self.pre_lnorm:
            c = self.layer_norm(c)
        head_q = self.q_net(h)
        head_k, head_v = torch.chunk(self.kv_net(c), 2, -1)
        head_q = head_q.view(h.size(0), h.size(1), self.n_head, self.d_head)
        head_k = head_k.view(c.size(0), c.size(1), self.n_head, self.d_head)
        head_v = head_v.view(c.size(0), c.size(1), self.n_head, self.d_head)
        attn_score = torch.einsum('ibnd,jbnd->ijbn', (head_q, head_k))
        attn_score.mul_(self.scale)
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[None, :, :, None].bool(), -float('inf'))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:, :, :, None].bool(), -float('inf'))
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, head_v))
        attn_vec = attn_vec.contiguous().view(attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)
        if self.pre_lnorm:
            output = h + attn_out
        else:
            output = self.layer_norm(h + attn_out)
        return output


class RelMultiHeadAttn(nn.Module):

    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0, tgt_len=None, ext_len=None, mem_len=None, pre_lnorm=False, moe=False, moe_num_expert=64, moe_top_k=2):
        super(RelMultiHeadAttn, self).__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout
        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)
        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model)
        self.scale = 1 / d_head ** 0.5
        self.pre_lnorm = pre_lnorm

    def _parallelogram_mask(self, h, w, left=False):
        mask = torch.ones((h, w)).byte()
        m = min(h, w)
        mask[:m, :m] = torch.triu(mask[:m, :m])
        mask[-m:, -m:] = torch.tril(mask[-m:, -m:])
        if left:
            return mask
        else:
            return mask.flip(0)

    def _shift(self, x, qlen, klen, mask, left=False):
        if qlen > 1:
            zero_pad = torch.zeros((x.size(0), qlen - 1, x.size(2), x.size(3)), device=x.device, dtype=x.dtype)
        else:
            zero_pad = torch.zeros(0, device=x.device, dtype=x.dtype)
        if left:
            mask = mask.flip(1)
            x_padded = torch.cat([zero_pad, x], dim=1).expand(qlen, -1, -1, -1)
        else:
            x_padded = torch.cat([x, zero_pad], dim=1).expand(qlen, -1, -1, -1)
        x = x_padded.masked_select(mask[:, :, None, None]).view(qlen, klen, x.size(2), x.size(3))
        return x

    def _rel_shift(self, x, zero_triu=False):
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)
        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])
        x = x_padded[1:].view_as(x)
        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:, :, None, None]
        return x

    def forward(self, w, r, attn_mask=None, mems=None):
        raise NotImplementedError


class RelPartialLearnableMultiHeadAttn(RelMultiHeadAttn):

    def __init__(self, *args, **kwargs):
        super(RelPartialLearnableMultiHeadAttn, self).__init__(*args, **kwargs)
        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)

    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask=None, mems=None):
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)
        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            r_head_k = self.r_net(r)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            r_head_k = self.r_net(r)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
        klen = w_head_k.size(0)
        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)
        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)
        rw_head_q = w_head_q + r_w_bias
        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))
        rr_head_q = w_head_q + r_r_bias
        BD = torch.einsum('ibnd,jnd->ijbn', (rr_head_q, r_head_k))
        BD = self._rel_shift(BD)
        attn_score = AC + BD
        attn_score.mul_(self.scale)
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score = attn_score.float().masked_fill(attn_mask[None, :, :, None].bool(), -float('inf')).type_as(attn_score)
            elif attn_mask.dim() == 3:
                attn_score = attn_score.float().masked_fill(attn_mask[:, :, :, None].bool(), -float('inf')).type_as(attn_score)
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))
        attn_vec = attn_vec.contiguous().view(attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)
        if self.pre_lnorm:
            output = w + attn_out
        else:
            output = self.layer_norm(w + attn_out)
        return output


class RelLearnableMultiHeadAttn(RelMultiHeadAttn):

    def __init__(self, *args, **kwargs):
        super(RelLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

    def forward(self, w, r_emb, r_w_bias, r_bias, attn_mask=None, mems=None):
        qlen, bsz = w.size(0), w.size(1)
        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
        klen = w_head_k.size(0)
        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)
        if klen > r_emb.size(0):
            r_emb_pad = r_emb[0:1].expand(klen - r_emb.size(0), -1, -1)
            r_emb = torch.cat([r_emb_pad, r_emb], 0)
            r_bias_pad = r_bias[0:1].expand(klen - r_bias.size(0), -1)
            r_bias = torch.cat([r_bias_pad, r_bias], 0)
        else:
            r_emb = r_emb[-klen:]
            r_bias = r_bias[-klen:]
        rw_head_q = w_head_q + r_w_bias[None]
        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))
        B_ = torch.einsum('ibnd,jnd->ijbn', (w_head_q, r_emb))
        D_ = r_bias[None, :, None]
        BD = self._rel_shift(B_ + D_)
        attn_score = AC + BD
        attn_score.mul_(self.scale)
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[None, :, :, None].bool(), -float('inf'))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:, :, :, None].bool(), -float('inf'))
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))
        attn_vec = attn_vec.contiguous().view(attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)
        if self.pre_lnorm:
            output = w + attn_out
        else:
            output = self.layer_norm(w + attn_out)
        return output


class AllGather(Function):
    """
    A wrapper for the All-Gather function to support auto-differentiation.
    """

    @staticmethod
    def forward(ctx, inp, rank, world_size, group):
        tensor_list = [torch.empty_like(inp) for _ in range(world_size)]
        torch.distributed.all_gather(tensor_list, inp, group=group)
        torch.cuda.synchronize()
        output = torch.cat(tensor_list, dim=0)
        ctx.args = rank, inp.shape[0]
        return output

    @staticmethod
    def backward(ctx, grad_out):
        rank, dim0 = ctx.args
        return grad_out[rank * dim0:(rank + 1) * dim0], None, None, None


class BaseGate(nn.Module):

    def __init__(self, num_expert, world_size):
        super().__init__()
        self.world_size = world_size
        self.num_expert = num_expert
        self.tot_expert = world_size * num_expert
        self.loss = None

    def forward(self, x):
        raise NotImplementedError('Base gate cannot be directly used for fwd')

    def set_loss(self, loss):
        self.loss = loss

    def get_loss(self, clear=True):
        loss = self.loss
        if clear:
            self.loss = None
        return loss

    @property
    def has_loss(self):
        return self.loss is not None


class NaiveGate(BaseGate):
    """
    A naive gate implementation that defines the standard behavior of the gate
    which determines which experts the tokens are going to.
    Both the indicies and the score, or confidence, are output to the parent
    module.
    The load-balance strategies are also designed to be implemented within the
    `Gate` module.
    """

    def __init__(self, d_model, num_expert, world_size, top_k=2):
        super().__init__(num_expert, world_size)
        self.gate = nn.Linear(d_model, self.tot_expert)
        self.top_k = top_k

    def forward(self, inp, return_all_scores=False):
        """
        The naive implementation simply calculates the top-k of a linear layer's
        output.
        """
        gate = self.gate(inp)
        gate_top_k_val, gate_top_k_idx = torch.topk(gate, k=self.top_k, dim=-1, largest=True, sorted=False)
        gate_top_k_val = gate_top_k_val.view(-1, self.top_k)
        gate_score = F.softmax(gate_top_k_val, dim=-1)
        if return_all_scores:
            return gate_top_k_idx, gate_score, gate
        return gate_top_k_idx, gate_score


class Slice(Function):
    """
    A wrapper for the Slice function to support auto-differentiation.
    """

    @staticmethod
    def forward(ctx, inp, rank, world_size, group):
        B: int = inp.shape[0]
        local_batch_size = B // world_size
        batch_start = local_batch_size * rank
        batch_end = min(batch_start + local_batch_size, B)
        inp = inp[batch_start:batch_end]
        ctx.args = world_size, group
        return inp

    @staticmethod
    def backward(ctx, grad_out):
        world_size, group = ctx.args
        tensor_list = [torch.empty_like(grad_out) for _ in range(world_size)]
        torch.distributed.all_gather(tensor_list, grad_out, group=group)
        torch.cuda.synchronize()
        grad_out = torch.cat(tensor_list, dim=0)
        return grad_out, None, None, None


def _local_gather(inp, pos, out_batch_size, maybe_overlap=True):
    inp_buf = torch.zeros(out_batch_size, inp.shape[-1], dtype=inp.dtype, device=inp.device)
    if maybe_overlap:
        inp_buf.index_add_(0, pos, inp)
    else:
        inp_buf.index_copy_(0, pos, inp)
    return inp_buf


def _local_scatter(inp, pos):
    inp_buf = torch.index_select(inp, 0, pos)
    return inp_buf


class MOEGather(Function):
    """
    Gather output samples from contiguous alone experts back to [batch x
    sequences]. Works symmetrically with MOEScatter.
    """

    @staticmethod
    def forward(ctx, global_output_buf, pos, local_expert_count, global_expert_count, local_batch_size, world_size):
        if world_size > 1:
            local_output_buf = fmoe_cuda.global_gather(global_output_buf, local_expert_count, global_expert_count, pos.shape[0], world_size)
        else:
            local_output_buf = global_output_buf
        output = _local_gather(local_output_buf, pos, local_batch_size, maybe_overlap=False)
        ctx.moe_args = global_output_buf.shape[0], world_size
        variables = pos, local_expert_count, global_expert_count
        ctx.save_for_backward(*variables)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        pos, local_expert_count, global_expert_count = ctx.saved_tensors
        fwd_batch_size, world_size = ctx.moe_args
        grad_out_buf = _local_scatter(grad_out.contiguous(), pos)
        if world_size > 1:
            global_grad_out_buf = fmoe_cuda.global_scatter(grad_out_buf, local_expert_count, global_expert_count, fwd_batch_size, world_size)
        else:
            global_grad_out_buf = grad_out_buf
        return global_grad_out_buf, None, None, None, None, None


class MOEScatter(Function):
    """
    Scatter input samples from [batch x sequences] to contiguous alone experts.
    If `world_size` is greater than 1, the samples will first be locally
    scattered, and then exchanged across workers.
    """

    @staticmethod
    def forward(ctx, inp, pos, local_expert_count, global_expert_count, fwd_batch_size, world_size):
        local_input_buf = _local_scatter(inp, pos)
        if world_size > 1:
            global_input_buf = fmoe_cuda.global_scatter(local_input_buf, local_expert_count, global_expert_count, fwd_batch_size, world_size)
        else:
            global_input_buf = local_input_buf
        ctx.moe_args = inp.shape[0], pos.shape[0], world_size
        variables = pos, local_expert_count, global_expert_count
        ctx.save_for_backward(*variables)
        return global_input_buf

    @staticmethod
    def backward(ctx, global_grad_in):
        pos, local_expert_count, global_expert_count = ctx.saved_tensors
        inp_batch_size, buf_batch_size, world_size = ctx.moe_args
        if world_size > 1:
            local_grad_in = fmoe_cuda.global_gather(global_grad_in, local_expert_count, global_expert_count, buf_batch_size, world_size)
        else:
            local_grad_in = global_grad_in
        grad_in = _local_gather(local_grad_in, pos, inp_batch_size)
        return grad_in, None, None, None, None, None


def count_by_gate(gate, num_expert, world_size, require_pos=True):
    with torch.no_grad():
        local_expert_count = torch.zeros(num_expert * world_size, device=gate.device, dtype=torch.int32)
        fmoe_cuda.expert_count(gate, local_expert_count)
        local_expert_count = local_expert_count.long()
        if world_size > 1:
            global_expert_count = fmoe_cuda.expert_exchange(local_expert_count, num_expert, world_size)
        else:
            global_expert_count = local_expert_count
        if not require_pos:
            pos = None
        else:
            lec_cum = torch.cumsum(local_expert_count, dim=0).int()
            pos_size = lec_cum[-1].item()
            pos = torch.empty((pos_size,), device=gate.device, dtype=torch.long)
            fmoe_cuda.assign_pos(lec_cum, gate, pos)
    return pos, local_expert_count, global_expert_count


def prepare_forward(gate, num_expert, world_size):
    """
    Prepare necessary information from gate output for MoE computation.

    Args:
        gate: a 1-d Long Tensor representing the target expert of each input
        sample.
        num_expert: number of experts on each worker.
        world_size: number of workers that hold different experts.
        comm: the communicator of all workers in the expert-parallel group.
    """
    pos, local_expert_count, global_expert_count = count_by_gate(gate, num_expert, world_size)
    with torch.no_grad():
        fwd_expert_count = global_expert_count.view(world_size, num_expert).sum(dim=0)
        fwd_batch_size = int(fwd_expert_count.sum().item())
    return pos, local_expert_count.cpu(), global_expert_count.cpu(), fwd_expert_count.cpu(), fwd_batch_size


def _fmoe_general_global_forward(inp, gate, expert_fn, num_expert, world_size, **kwargs):
    """
    A private function that performs the following steps to complete the MoE
    computation.
    * Count the number of tokens from each worker to each expert.
    * Send the features to their target position so that input features to each
    expert are contiguous in memory.
    * Perform the forward computation of the experts using `expert_fn`
    * Gather the output features of experts back, and reorder them as sentences.
    Intermediate results like expert counts are hidden from users by this
    function.
    """
    pos, local_expert_count, global_expert_count, fwd_expert_count, fwd_batch_size = prepare_forward(gate, num_expert, world_size)
    topk = 1
    if len(gate.shape) == 2:
        topk = gate.shape[1]

    def scatter_func(tensor):
        return MOEScatter.apply(tensor, torch.div(pos, topk, rounding_mode='floor'), local_expert_count, global_expert_count, fwd_batch_size, world_size)
    x = tree.map_structure(scatter_func, inp)
    x = expert_fn(x, fwd_expert_count)
    out_batch_size = tree.flatten(inp)[0].shape[0]
    if len(gate.shape) == 2:
        out_batch_size *= gate.shape[1]

    def gather_func(tensor):
        return MOEGather.apply(tensor, pos, local_expert_count, global_expert_count, out_batch_size, world_size)
    outp = tree.map_structure(gather_func, x)
    return outp


def get_torch_default_comm():
    """
    The NCCL communicator is needed so that Fast MoE can perform customized
    communication operators in the C code. However, it is not a publicly
    available variable. Therefore, a hacking class of the `ProcessGroupNCCL`
    in Fast MoE's C code takes the `_default_pg` and tries to dig the
    communicator out from the object. As PyTorch's private interface varies from
    time to time, different hacking techniques are tried one-by-one to be
    compatible with various versions of PyTorch.
    """
    try:
        comm = dist.distributed_c10d._get_default_group()
        return comm
    except Exception as _:
        pass
    try:
        comm = dist.distributed_c10d._default_pg
        if comm is not None:
            return comm
    except Exception as _:
        pass
    raise RuntimeError('Unsupported PyTorch version')


def ensure_comm(t, comm):
    if comm is None:
        comm = get_torch_default_comm()
    global _moe_group
    _moe_group = comm
    fmoe_cuda.ensure_nccl(comm, t)


def mark_module_parallel_comm(module, comm):
    """
    Mark all parameters in `module` as doing data parallel in `comm`, where
    `comm` may be one of `'world', 'dp', 'none'`.
    """
    for p in module.parameters():
        setattr(p, 'dp_comm', comm)


class FMoE(nn.Module):
    """
    A general moe implementation that supports an arbitrary module as the
    expert.
    * `num_expert` stands for the number of experts on **each** worker.
    * `world_size` stands for the total number of workers that contains
    different experts.
    * `slice_group` can be a torch's communication group, indicating that
    specific model parallel is applied across the group, and workers in the
    group hold the same copy of input feature, and requires the same copy of
    the output. For each worker, FMoE only computes the output of a certain
    slice of the input batch, and will all-gather the outputs after
    computation.
    * `top_k` stands for the number of experts each token is going to.
    * `gate` is a gate class which can found in `fmoe.gates`.
    * `expert` can be specified as a module class, it is used to generate
    `num_expert` expert modules.
    """

    def __init__(self, num_expert=32, d_model=1024, world_size=1, mp_group=None, slice_group=None, moe_group=None, top_k=2, gate=NaiveGate, expert=None, gate_hook=None, mask=None, mask_dict=None):
        super().__init__()
        self.num_expert = num_expert
        self.d_model = d_model
        self.world_size = world_size
        self.slice_group = slice_group
        if mp_group is not None:
            None
            self.slice_group = mp_group
        if self.slice_group is None:
            self.slice_size = 1
            self.slice_rank = 0
        else:
            self.slice_size = self.slice_group.size()
            self.slice_rank = self.slice_group.rank()
        self.top_k = top_k
        if type(expert) is list:
            self.experts = nn.ModuleList([e(d_model) for e in expert])
            self.experts_fused = False
            self.num_expert = num_expert = len(expert)
        elif expert is not None:
            self.experts = nn.ModuleList([expert(d_model) for _ in range(num_expert)])
            self.experts_fused = False
        else:
            self.experts_fused = True
        self.gate = gate(d_model, num_expert, world_size, top_k)
        self.gate_hook = gate_hook
        self.mask = mask
        self.mask_dict = mask_dict
        self.moe_group = moe_group

    def expert_fn(self, inp, fwd_expert_count):
        """
        The default expert function which either calls the experts as a whole
        or as separate experts.
        """
        if self.experts_fused:
            return self.experts(inp, fwd_expert_count)
        if isinstance(fwd_expert_count, torch.Tensor):
            fwd_expert_count = fwd_expert_count.cpu().numpy()
        outputs = []
        base_idx = 0
        for i in range(self.num_expert):
            batch_size = fwd_expert_count[i]
            inp_slice = inp[base_idx:base_idx + batch_size]
            outputs.append(self.experts[i](inp_slice))
            base_idx += batch_size
        return torch.cat(outputs, dim=0)

    def mark_parallel_comm(self, expert_dp_comm='none'):
        """
        Automatically mark the data parallel comms of the parameters within the
        module. This can be typically called at the end of the __init__ function
        in child classes.
        """
        if self.experts is not None:
            comm = expert_dp_comm
            if isinstance(self.experts, list):
                for e in self.experts:
                    mark_module_parallel_comm(e, comm)
            else:
                mark_module_parallel_comm(self.experts, comm)
        mark_module_parallel_comm(self.gate, 'gate')

    def forward(self, moe_inp):
        """
        The FMoE module first computes gate output, and then conduct MoE forward
        according to the gate.  The score of the selected gate given by the
        expert is multiplied to the experts' output tensors as a weight.
        """
        moe_inp_batch_size = tree.flatten(tree.map_structure(lambda tensor: tensor.shape[0], moe_inp))
        assert all([(batch_size == moe_inp_batch_size[0]) for batch_size in moe_inp_batch_size]), 'MoE inputs must have the same batch size'
        if self.world_size > 1:

            def ensure_comm_func(tensor):
                ensure_comm(tensor, self.moe_group)
            tree.map_structure(ensure_comm_func, moe_inp)
        if self.slice_size > 1:

            def slice_func(tensor):
                return Slice.apply(tensor, self.slice_rank, self.slice_size, self.slice_group)
            moe_inp = tree.map_structure(slice_func, moe_inp)
        gate_top_k_idx, gate_score = self.gate(moe_inp)
        if self.gate_hook is not None:
            self.gate_hook(gate_top_k_idx, gate_score, None)
        if self.mask is not None and self.mask_dict is not None:

            def delete_mask_func(tensor):
                tensor = tensor[mask == 0, :]
                return tensor
            mask = self.mask.view(-1)
            moe_inp = tree.map_structure(delete_mask_func, moe_inp)
            gate_top_k_idx = gate_top_k_idx[mask == 0, :]
        fwd = _fmoe_general_global_forward(moe_inp, gate_top_k_idx, self.expert_fn, self.num_expert, self.world_size, experts=self.experts)
        if self.mask is not None and self.mask_dict is not None:

            def recover_func(tensor):
                dim = tensor.shape[-1]
                tensor = tensor.view(-1, self.top_k, dim)
                x = torch.zeros(mask.shape[0], self.top_k, dim, device=tensor.device, dtype=tensor.dtype)
                x[mask == 0] = tensor
                for k, v in self.mask_dict.items():
                    x[mask == k] = v
                return x
            moe_outp = tree.map_structure(recover_func, fwd)
        else:

            def view_func(tensor):
                dim = tensor.shape[-1]
                tensor = tensor.view(-1, self.top_k, dim)
                return tensor
            moe_outp = tree.map_structure(view_func, fwd)
        gate_score = gate_score.view(-1, 1, self.top_k)

        def bmm_func(tensor):
            dim = tensor.shape[-1]
            tensor = torch.bmm(gate_score, tensor).reshape(-1, dim)
            return tensor
        moe_outp = tree.map_structure(bmm_func, moe_outp)
        if self.slice_size > 1:

            def all_gather_func(tensor):
                return AllGather.apply(tensor, self.slice_rank, self.slice_size, self.slice_group)
            moe_outp = tree.map_structure(all_gather_func, moe_outp)
        moe_outp_batch_size = tree.flatten(tree.map_structure(lambda tensor: tensor.shape[0], moe_outp))
        assert all([(batch_size == moe_outp_batch_size[0]) for batch_size in moe_outp_batch_size]), 'MoE outputs must have the same batch size'
        return moe_outp


class MOELinear(Function):
    """
    Computes linear operators within one GPU on different experts simutaneously.
    """

    @staticmethod
    def forward(ctx, global_input_buf, fwd_expert_count, weight, bias=None):
        global_output_buf = fmoe_cuda.linear_forward(global_input_buf, fwd_expert_count, weight, bias)
        variables = global_input_buf, fwd_expert_count, weight, bias
        ctx.save_for_backward(*variables)
        return global_output_buf

    @staticmethod
    def backward(ctx, grad_out):
        input_buf, fwd_expert_count, weight, bias = ctx.saved_tensors
        grad_inp_buf, grad_weight, grad_bias = fmoe_cuda.linear_backward(grad_out, input_buf, fwd_expert_count, weight, bias)
        if not torch.is_tensor(bias):
            grad_bias = None
        return grad_inp_buf, None, grad_weight, grad_bias


class FMoELinear(nn.Module):
    """
    A linear layer that contains multiple experts.
    As multiple experts can be placed on the same worker, the computation can be
    performed in parallel to increase the performance.
    The FMoELinear module provides such function.
    """

    def __init__(self, num_expert: int, in_feat: int, out_feat: int, bias: bool=True, rank: int=0):
        super().__init__()
        self.num_expert = num_expert
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.rank = rank
        self.weight = nn.Parameter(torch.Tensor(num_expert, out_feat, in_feat))
        if bias:
            self.bias = nn.Parameter(torch.zeros(num_expert, out_feat))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def forward(self, inp, fwd_expert_count):
        """
        Call MOE function
        """
        x = MOELinear.apply(inp, fwd_expert_count, self.weight, self.bias)
        return x

    def extra_repr(self) ->str:
        return 'num_expert={}, in_features={},         out_features={}, bias={}, rank={}'.format(self.num_expert, self.in_feat, self.out_feat, self.bias is not None, self.rank)

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))


class _Expert(nn.Module):
    """
    An expert using 2 FMoELinear modules to speed up the computation of experts
    within one worker.
    """

    def __init__(self, num_expert, d_model, d_hidden, activation, rank=0):
        super().__init__()
        self.htoh4 = FMoELinear(num_expert, d_model, d_hidden, bias=True, rank=rank)
        self.h4toh = FMoELinear(num_expert, d_hidden, d_model, bias=True, rank=rank)
        self.activation = activation

    def forward(self, inp, fwd_expert_count):
        """
        First expand input to 4h (the hidden size is variable, but is called h4
        for convenience). Then perform activation. Finally shirink back to h.
        """
        x = self.htoh4(inp, fwd_expert_count)
        x = self.activation(x)
        x = self.h4toh(x, fwd_expert_count)
        return x


class FMoETransformerMLP(FMoE):
    """
    A complete MoE MLP module in a Transformer block.
    * `activation` is the activation function to be used in MLP in each expert.
    * `d_hidden` is the dimension of the MLP layer.
    """

    def __init__(self, num_expert=32, d_model=1024, d_hidden=4096, activation=torch.nn.GELU(), expert_dp_comm='none', expert_rank=0, **kwargs):
        super().__init__(num_expert=num_expert, d_model=d_model, **kwargs)
        self.experts = _Expert(num_expert, d_model, d_hidden, activation, rank=expert_rank)
        self.mark_parallel_comm(expert_dp_comm)

    def forward(self, inp: torch.Tensor):
        """
        This module wraps up the FMoE module with reshape, residual and layer
        normalization.
        """
        original_shape = inp.shape
        inp = inp.reshape(-1, self.d_model)
        output = super().forward(inp)
        return output.reshape(original_shape)


class CustomizedMoEPositionwiseFF(FMoETransformerMLP):

    def __init__(self, d_model, d_inner, dropout, pre_lnorm=False, moe_num_expert=64, moe_top_k=2):
        activation = nn.Sequential(nn.ReLU(), nn.Dropout(dropout))
        super().__init__(num_expert=moe_num_expert, d_model=d_model, d_hidden=d_inner, top_k=moe_top_k, activation=activation)
        self.pre_lnorm = pre_lnorm
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inp):
        if self.pre_lnorm:
            core_out = super().forward(self.layer_norm(inp))
            core_out = self.dropout(core_out)
            output = core_out + inp
        else:
            core_out = super().forward(inp)
            core_out = self.dropout(core_out)
            output = self.layer_norm(inp + core_out)
        return output


class DecoderLayer(nn.Module):

    def __init__(self, n_head, d_model, d_head, d_inner, dropout, **kwargs):
        super(DecoderLayer, self).__init__()
        self.dec_attn = MultiHeadAttn(n_head, d_model, d_head, dropout, **kwargs)
        if kwargs.get('moe') is False:
            self.pos_ff = PositionwiseFF(d_model, d_inner, dropout, pre_lnorm=kwargs.get('pre_lnorm'))
        else:
            self.pos_ff = CustomizedMoEPositionwiseFF(d_model, d_inner, dropout, pre_lnorm=kwargs.get('pre_lnorm'), moe_num_expert=kwargs.get('moe_num_expert'), moe_top_k=kwargs.get('moe_top_k'))

    def forward(self, dec_inp, dec_attn_mask=None, mems=None):
        output = self.dec_attn(dec_inp, attn_mask=dec_attn_mask, mems=mems)
        output = self.pos_ff(output)
        return output


class RelLearnableDecoderLayer(nn.Module):

    def __init__(self, n_head, d_model, d_head, d_inner, dropout, **kwargs):
        super(RelLearnableDecoderLayer, self).__init__()
        self.dec_attn = RelLearnableMultiHeadAttn(n_head, d_model, d_head, dropout, **kwargs)
        if kwargs.get('moe') is False:
            self.pos_ff = PositionwiseFF(d_model, d_inner, dropout, pre_lnorm=kwargs.get('pre_lnorm'))
        else:
            self.pos_ff = CustomizedMoEPositionwiseFF(d_model, d_inner, dropout, pre_lnorm=kwargs.get('pre_lnorm'), moe_num_expert=kwargs.get('moe_num_expert'), moe_top_k=kwargs.get('moe_top_k'))

    def forward(self, dec_inp, r_emb, r_w_bias, r_bias, dec_attn_mask=None, mems=None):
        output = self.dec_attn(dec_inp, r_emb, r_w_bias, r_bias, attn_mask=dec_attn_mask, mems=mems)
        output = self.pos_ff(output)
        return output


class RelPartialLearnableDecoderLayer(nn.Module):

    def __init__(self, n_head, d_model, d_head, d_inner, dropout, **kwargs):
        super(RelPartialLearnableDecoderLayer, self).__init__()
        self.dec_attn = RelPartialLearnableMultiHeadAttn(n_head, d_model, d_head, dropout, **kwargs)
        if kwargs.get('moe') is False:
            self.pos_ff = PositionwiseFF(d_model, d_inner, dropout, pre_lnorm=kwargs.get('pre_lnorm'))
        else:
            self.pos_ff = CustomizedMoEPositionwiseFF(d_model, d_inner, dropout, pre_lnorm=kwargs.get('pre_lnorm'), moe_num_expert=kwargs.get('moe_num_expert'), moe_top_k=kwargs.get('moe_top_k'))

    def forward(self, dec_inp, r, r_w_bias, r_r_bias, dec_attn_mask=None, mems=None):
        output = self.dec_attn(dec_inp, r, r_w_bias, r_r_bias, attn_mask=dec_attn_mask, mems=mems)
        output = self.pos_ff(output)
        return output


class Projection(nn.Module):

    def __init__(self, out_feat, in_feat):
        self.weight = nn.Parameter(torch.Tensor(out_feat, in_feat))


class AdaptiveEmbedding(nn.Module):

    def __init__(self, n_token, d_embed, d_proj, cutoffs, div_val=1, sample_softmax=False):
        super(AdaptiveEmbedding, self).__init__()
        self.n_token = n_token
        self.d_embed = d_embed
        self.cutoffs = cutoffs + [n_token]
        self.div_val = div_val
        self.d_proj = d_proj
        self.emb_scale = d_proj ** 0.5
        self.cutoff_ends = [0] + self.cutoffs
        self.emb_layers = nn.ModuleList()
        self.emb_projs = nn.ModuleList()
        if div_val == 1:
            self.emb_layers.append(nn.Embedding(n_token, d_embed, sparse=sample_softmax > 0))
            if d_proj != d_embed:
                self.emb_projs.append(Projection(d_proj, d_embed))
        else:
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                d_emb_i = d_embed // div_val ** i
                self.emb_layers.append(nn.Embedding(r_idx - l_idx, d_emb_i))
                self.emb_projs.append(Projection(d_proj, d_emb_i))

    def forward(self, inp):
        if self.div_val == 1:
            embed = self.emb_layers[0](inp)
            if self.d_proj != self.d_embed:
                embed = F.linear(embed, self.emb_projs[0].weight)
        else:
            param = next(self.parameters())
            inp_flat = inp.view(-1)
            emb_flat = torch.zeros([inp_flat.size(0), self.d_proj], dtype=param.dtype, device=param.device)
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                mask_i = (inp_flat >= l_idx) & (inp_flat < r_idx)
                indices_i = mask_i.nonzero().squeeze()
                if indices_i.numel() == 0:
                    continue
                inp_i = inp_flat.index_select(0, indices_i) - l_idx
                emb_i = self.emb_layers[i](inp_i)
                emb_i = F.linear(emb_i, self.emb_projs[i].weight)
                emb_flat.index_copy_(0, indices_i, emb_i)
            embed = emb_flat.view(*inp.size(), self.d_proj)
        embed.mul_(self.emb_scale)
        return embed


class LogUniformSampler(object):

    def __init__(self, range_max, n_sample):
        """
        Reference : https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/ops/candidate_sampling_ops.py
            `P(class) = (log(class + 2) - log(class + 1)) / log(range_max + 1)`

        expected count can be approximated by 1 - (1 - p)^n
        and we use a numerically stable version -expm1(num_tries * log1p(-p))

        Our implementation fixes num_tries at 2 * n_sample, and the actual #samples will vary from run to run
        """
        with torch.no_grad():
            self.range_max = range_max
            log_indices = torch.arange(1.0, range_max + 2.0, 1.0).log_()
            self.dist = (log_indices[1:] - log_indices[:-1]) / log_indices[-1]
            self.log_q = (-(-self.dist.double().log1p_() * 2 * n_sample).expm1_()).log_().float()
        self.n_sample = n_sample

    def sample(self, labels):
        """
            labels: [b1, b2]
        Return
            true_log_probs: [b1, b2]
            samp_log_probs: [n_sample]
            neg_samples: [n_sample]
        """
        n_sample = self.n_sample
        n_tries = 2 * n_sample
        with torch.no_grad():
            neg_samples = torch.multinomial(self.dist, n_tries, replacement=True).unique()
            device = labels.device
            neg_samples = neg_samples
            true_log_probs = self.log_q[labels]
            samp_log_probs = self.log_q[neg_samples]
            return true_log_probs, samp_log_probs, neg_samples


class ProjectedAdaptiveLogSoftmax(nn.Module):

    def __init__(self, n_token, d_embed, d_proj, cutoffs, div_val=1, keep_order=False):
        super(ProjectedAdaptiveLogSoftmax, self).__init__()
        self.n_token = n_token
        self.d_embed = d_embed
        self.d_proj = d_proj
        self.cutoffs = cutoffs + [n_token]
        self.cutoff_ends = [0] + self.cutoffs
        self.div_val = div_val
        self.shortlist_size = self.cutoffs[0]
        self.n_clusters = len(self.cutoffs) - 1
        self.head_size = self.shortlist_size + self.n_clusters
        if self.n_clusters > 0:
            self.cluster_weight = nn.Parameter(torch.zeros(self.n_clusters, self.d_embed))
            self.cluster_bias = nn.Parameter(torch.zeros(self.n_clusters))
        self.out_layers = nn.ModuleList()
        self.out_projs = nn.ModuleList()
        if div_val == 1:
            for i in range(len(self.cutoffs)):
                if d_proj != d_embed:
                    self.out_projs.append(Projection(d_proj, d_embed))
                else:
                    self.out_projs.append(None)
            self.out_layers.append(nn.Linear(d_embed, n_token))
        else:
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                d_emb_i = d_embed // div_val ** i
                self.out_projs.append(Projection(d_proj, d_emb_i))
                self.out_layers.append(nn.Linear(d_emb_i, r_idx - l_idx))
        self.keep_order = keep_order

    def _compute_logit(self, hidden, weight, bias, proj):
        if proj is None:
            logit = F.linear(hidden, weight, bias=bias)
        else:
            proj_hid = F.linear(hidden, proj.t().contiguous())
            logit = F.linear(proj_hid, weight, bias=bias)
        return logit

    def forward(self, hidden, target, keep_order=False):
        """
            hidden :: [len*bsz x d_proj]
            target :: [len*bsz]
        """
        if hidden.size(0) != target.size(0):
            raise RuntimeError('Input and target should have the same size in the batch dimension.')
        if self.n_clusters == 0:
            logit = self._compute_logit(hidden, self.out_layers[0].weight, self.out_layers[0].bias, self.out_projs[0].weight if self.out_projs[0] is not None else None)
            nll = -F.log_softmax(logit, dim=-1).gather(1, target.unsqueeze(1)).squeeze(1)
        else:
            weights, biases = [], []
            for i in range(len(self.cutoffs)):
                if self.div_val == 1:
                    l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                    weight_i = self.out_layers[0].weight[l_idx:r_idx]
                    bias_i = self.out_layers[0].bias[l_idx:r_idx]
                else:
                    weight_i = self.out_layers[i].weight
                    bias_i = self.out_layers[i].bias
                if i == 0:
                    weight_i = torch.cat([weight_i, self.cluster_weight], dim=0)
                    bias_i = torch.cat([bias_i, self.cluster_bias], dim=0)
                weights.append(weight_i)
                biases.append(bias_i)
            head_weight, head_bias, head_proj = weights[0], biases[0], self.out_projs[0].weight if self.out_projs[0] is not None else None
            head_logit = self._compute_logit(hidden, head_weight, head_bias, head_proj)
            head_logprob = F.log_softmax(head_logit, dim=1)
            nll = torch.zeros_like(target, dtype=hidden.dtype, device=hidden.device)
            offset = 0
            cutoff_values = [0] + self.cutoffs
            for i in range(len(cutoff_values) - 1):
                l_idx, r_idx = cutoff_values[i], cutoff_values[i + 1]
                mask_i = (target >= l_idx) & (target < r_idx)
                indices_i = mask_i.nonzero().squeeze()
                if indices_i.numel() == 0:
                    continue
                target_i = target.index_select(0, indices_i) - l_idx
                head_logprob_i = head_logprob.index_select(0, indices_i)
                if i == 0:
                    logprob_i = head_logprob_i.gather(1, target_i[:, None]).squeeze(1)
                else:
                    weight_i, bias_i, proj_i = weights[i], biases[i], self.out_projs[i].weight if self.out_projs[i] is not None else None
                    hidden_i = hidden.index_select(0, indices_i)
                    tail_logit_i = self._compute_logit(hidden_i, weight_i, bias_i, proj_i)
                    tail_logprob_i = F.log_softmax(tail_logit_i, dim=1)
                    logprob_i = head_logprob_i[:, -i] + tail_logprob_i.gather(1, target_i[:, None]).squeeze(1)
                if hasattr(self, 'keep_order') and self.keep_order or keep_order:
                    nll.index_copy_(0, indices_i, -logprob_i)
                else:
                    nll[offset:offset + logprob_i.size(0)].copy_(-logprob_i)
                offset += logprob_i.size(0)
        return nll


def sample_logits(embedding, bias, labels, inputs, sampler):
    """
        embedding: an nn.Embedding layer
        bias: [n_vocab]
        labels: [b1, b2]
        inputs: [b1, b2, n_emb]
        sampler: you may use a LogUniformSampler
    Return
        logits: [b1, b2, 1 + n_sample]
    """
    true_log_probs, samp_log_probs, neg_samples = sampler.sample(labels)
    n_sample = neg_samples.size(0)
    b1, b2 = labels.size(0), labels.size(1)
    all_ids = torch.cat([labels.view(-1), neg_samples])
    all_w = embedding(all_ids)
    true_w = all_w[:-n_sample].view(b1, b2, -1)
    sample_w = all_w[-n_sample:].view(n_sample, -1)
    all_b = bias[all_ids]
    true_b = all_b[:-n_sample].view(b1, b2)
    sample_b = all_b[-n_sample:]
    hit = (labels[:, :, None] == neg_samples).detach()
    true_logits = torch.einsum('ijk,ijk->ij', [true_w, inputs]) + true_b - true_log_probs
    sample_logits = torch.einsum('lk,ijk->ijl', [sample_w, inputs]) + sample_b - samp_log_probs
    sample_logits.masked_fill_(hit, -1e+30)
    logits = torch.cat([true_logits[:, :, None], sample_logits], -1)
    return logits


class MemTransformerLM(nn.Module):

    def __init__(self, n_token, n_layer, n_head, d_model, d_head, d_inner, dropout, dropatt, tie_weight=True, d_embed=None, div_val=1, tie_projs=[False], pre_lnorm=False, tgt_len=None, ext_len=None, mem_len=None, cutoffs=[], adapt_inp=False, same_length=False, attn_type=0, clamp_len=-1, sample_softmax=-1, moe=False, moe_num_expert=64, moe_top_k=2):
        super(MemTransformerLM, self).__init__()
        self.n_token = n_token
        d_embed = d_model if d_embed is None else d_embed
        self.d_embed = d_embed
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head
        self.word_emb = AdaptiveEmbedding(n_token, d_embed, d_model, cutoffs, div_val=div_val)
        self.drop = nn.Dropout(dropout)
        self.n_layer = n_layer
        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len
        self.max_klen = tgt_len + ext_len + mem_len
        self.attn_type = attn_type
        self.layers = nn.ModuleList()
        if attn_type == 0:
            for i in range(n_layer):
                self.layers.append(RelPartialLearnableDecoderLayer(n_head, d_model, d_head, d_inner, dropout, tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len, dropatt=dropatt, pre_lnorm=pre_lnorm, moe=moe, moe_num_expert=moe_num_expert, moe_top_k=moe_top_k))
        elif attn_type == 1:
            for i in range(n_layer):
                self.layers.append(RelLearnableDecoderLayer(n_head, d_model, d_head, d_inner, dropout, tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len, dropatt=dropatt, pre_lnorm=pre_lnorm, moe=moe, moe_num_expert=moe_num_expert, moe_top_k=moe_top_k))
        elif attn_type in [2, 3]:
            for i in range(n_layer):
                self.layers.append(DecoderLayer(n_head, d_model, d_head, d_inner, dropout, dropatt=dropatt, pre_lnorm=pre_lnorm, moe=moe, moe_num_expert=moe_num_expert, moe_top_k=moe_top_k))
        self.sample_softmax = sample_softmax
        if sample_softmax > 0:
            self.out_layer = nn.Linear(d_model, n_token)
            if tie_weight:
                self.out_layer.weight = self.word_emb.weight
            self.tie_weight = tie_weight
            self.sampler = LogUniformSampler(n_token, sample_softmax)
        else:
            self.crit = ProjectedAdaptiveLogSoftmax(n_token, d_embed, d_model, cutoffs, div_val=div_val)
            if tie_weight:
                for i in range(len(self.crit.out_layers)):
                    self.crit.out_layers[i].weight = self.word_emb.emb_layers[i].weight
            if tie_projs:
                for i, tie_proj in enumerate(tie_projs):
                    if tie_proj and div_val == 1 and d_model != d_embed:
                        self.crit.out_projs[i].weight = self.word_emb.emb_projs[0].weight
                    elif tie_proj and div_val != 1:
                        self.crit.out_projs[i].weight = self.word_emb.emb_projs[i].weight
        self.same_length = same_length
        self.clamp_len = clamp_len
        self._create_params()

    def backward_compatible(self):
        self.sample_softmax = -1

    def _create_params(self):
        if self.attn_type == 0:
            self.pos_emb = PositionalEmbedding(self.d_model)
            self.r_w_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
            self.r_r_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
        elif self.attn_type == 1:
            self.r_emb = nn.Parameter(torch.Tensor(self.n_layer, self.max_klen, self.n_head, self.d_head))
            self.r_w_bias = nn.Parameter(torch.Tensor(self.n_layer, self.n_head, self.d_head))
            self.r_bias = nn.Parameter(torch.Tensor(self.n_layer, self.max_klen, self.n_head))
        elif self.attn_type == 2:
            self.pos_emb = PositionalEmbedding(self.d_model)
        elif self.attn_type == 3:
            self.r_emb = nn.Parameter(torch.Tensor(self.n_layer, self.max_klen, self.n_head, self.d_head))

    def reset_length(self, tgt_len, ext_len, mem_len):
        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len

    def init_mems(self, x):
        if self.mem_len > 0:
            mems = []
            for i in range(self.n_layer + 1):
                empty = torch.empty(0, dtype=x.dtype, device=x.device)
                mems.append(empty)
            return mems
        else:
            return None

    def _update_mems(self, hids, mems, qlen, mlen):
        if mems is None:
            return None
        assert len(hids) == len(mems), 'len(hids) != len(mems)'
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + max(0, qlen - 0 - self.ext_len)
            beg_idx = max(0, end_idx - self.mem_len)
            for i in range(len(hids)):
                cat = torch.cat([mems[i], hids[i]], dim=0)
                new_mems.append(cat[beg_idx:end_idx].detach())
        return new_mems

    def _forward(self, dec_inp, mems=None):
        qlen, bsz = dec_inp.size()
        word_emb = self.word_emb(dec_inp)
        mlen = mems[0].size(0) if mems is not None else 0
        klen = mlen + qlen
        if self.same_length:
            all_ones = word_emb.new_ones(qlen, klen)
            mask_len = klen - self.mem_len
            if mask_len > 0:
                mask_shift_len = qlen - mask_len
            else:
                mask_shift_len = qlen
            dec_attn_mask = (torch.triu(all_ones, 1 + mlen) + torch.tril(all_ones, -mask_shift_len)).byte()[:, :, None]
        else:
            dec_attn_mask = torch.triu(word_emb.new_ones(qlen, klen), diagonal=1 + mlen).byte()[:, :, None]
        hids = []
        if self.attn_type == 0:
            pos_seq = torch.arange(klen - 1, -1, -1.0, device=word_emb.device, dtype=word_emb.dtype)
            if self.clamp_len > 0:
                pos_seq.clamp_(max=self.clamp_len)
            pos_emb = self.pos_emb(pos_seq)
            core_out = self.drop(word_emb)
            pos_emb = self.drop(pos_emb)
            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                core_out = layer(core_out, pos_emb, self.r_w_bias, self.r_r_bias, dec_attn_mask=dec_attn_mask, mems=mems_i)
                hids.append(core_out)
        elif self.attn_type == 1:
            core_out = self.drop(word_emb)
            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                if self.clamp_len > 0:
                    r_emb = self.r_emb[i][-self.clamp_len:]
                    r_bias = self.r_bias[i][-self.clamp_len:]
                else:
                    r_emb, r_bias = self.r_emb[i], self.r_bias[i]
                mems_i = None if mems is None else mems[i]
                core_out = layer(core_out, r_emb, self.r_w_bias[i], r_bias, dec_attn_mask=dec_attn_mask, mems=mems_i)
                hids.append(core_out)
        elif self.attn_type == 2:
            pos_seq = torch.arange(klen - 1, -1, -1.0, device=word_emb.device, dtype=word_emb.dtype)
            if self.clamp_len > 0:
                pos_seq.clamp_(max=self.clamp_len)
            pos_emb = self.pos_emb(pos_seq)
            core_out = self.drop(word_emb + pos_emb[-qlen:])
            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                if mems_i is not None and i == 0:
                    mems_i += pos_emb[:mlen]
                core_out = layer(core_out, dec_attn_mask=dec_attn_mask, mems=mems_i)
                hids.append(core_out)
        elif self.attn_type == 3:
            core_out = self.drop(word_emb)
            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                if mems_i is not None and mlen > 0:
                    cur_emb = self.r_emb[i][:-qlen]
                    cur_size = cur_emb.size(0)
                    if cur_size < mlen:
                        cur_emb_pad = cur_emb[0:1].expand(mlen - cur_size, -1, -1)
                        cur_emb = torch.cat([cur_emb_pad, cur_emb], 0)
                    else:
                        cur_emb = cur_emb[-mlen:]
                    mems_i += cur_emb.view(mlen, 1, -1)
                core_out += self.r_emb[i][-qlen:].view(qlen, 1, -1)
                core_out = layer(core_out, dec_attn_mask=dec_attn_mask, mems=mems_i)
                hids.append(core_out)
        core_out = self.drop(core_out)
        new_mems = self._update_mems(hids, mems, mlen, qlen)
        return core_out, new_mems

    def forward(self, data, target, *mems):
        if not mems:
            mems = self.init_mems(data)
        tgt_len = target.size(0)
        hidden, new_mems = self._forward(data, mems=mems)
        pred_hid = hidden[-tgt_len:]
        if self.sample_softmax > 0 and self.training:
            assert self.tie_weight
            logit = sample_logits(self.word_emb, self.out_layer.bias, target, pred_hid, self.sampler)
            loss = -F.log_softmax(logit, -1)[:, :, 0]
        else:
            loss = self.crit(pred_hid.view(-1, pred_hid.size(-1)), target.contiguous().view(-1))
            loss = loss.view(tgt_len, -1)
        if new_mems is None:
            return [loss]
        else:
            return [loss] + new_mems


class AdaptiveLogSoftmax(nn.Module):

    def __init__(self, in_features, n_classes, cutoffs, keep_order=False):
        super(AdaptiveLogSoftmax, self).__init__()
        cutoffs = list(cutoffs)
        if cutoffs != sorted(cutoffs) or min(cutoffs) <= 0 or max(cutoffs) >= n_classes - 1 or len(set(cutoffs)) != len(cutoffs) or any([(int(c) != c) for c in cutoffs]):
            raise ValueError('cutoffs should be a sequence of unique, positive integers sorted in an increasing order, where each value is between 1 and n_classes-1')
        self.in_features = in_features
        self.n_classes = n_classes
        self.cutoffs = cutoffs + [n_classes]
        self.shortlist_size = self.cutoffs[0]
        self.n_clusters = len(self.cutoffs) - 1
        self.head_size = self.shortlist_size + self.n_clusters
        self.cluster_weight = nn.Parameter(torch.zeros(self.n_clusters, self.in_features))
        self.cluster_bias = nn.Parameter(torch.zeros(self.n_clusters))
        self.keep_order = keep_order

    def forward(self, hidden, target, weight, bias, keep_order=False):
        if hidden.size(0) != target.size(0):
            raise RuntimeError('Input and target should have the same size in the batch dimension.')
        head_weight = torch.cat([weight[:self.shortlist_size], self.cluster_weight], dim=0)
        head_bias = torch.cat([bias[:self.shortlist_size], self.cluster_bias], dim=0)
        head_logit = F.linear(hidden, head_weight, bias=head_bias)
        head_logprob = F.log_softmax(head_logit, dim=1)
        nll = torch.zeros_like(target, dtype=hidden.dtype, device=hidden.device)
        offset = 0
        cutoff_values = [0] + self.cutoffs
        for i in range(len(cutoff_values) - 1):
            l_idx, h_idx = cutoff_values[i], cutoff_values[i + 1]
            mask_i = (target >= l_idx) & (target < h_idx)
            indices_i = mask_i.nonzero().squeeze()
            if indices_i.numel() == 0:
                continue
            target_i = target.index_select(0, indices_i) - l_idx
            head_logprob_i = head_logprob.index_select(0, indices_i)
            if i == 0:
                logprob_i = head_logprob_i.gather(1, target_i[:, None]).squeeze(1)
            else:
                weight_i = weight[l_idx:h_idx]
                bias_i = bias[l_idx:h_idx]
                hidden_i = hidden.index_select(0, indices_i)
                tail_logit_i = F.linear(hidden_i, weight_i, bias=bias_i)
                tail_logprob_i = F.log_softmax(tail_logit_i, dim=1)
                logprob_i = head_logprob_i[:, -i] + tail_logprob_i.gather(1, target_i[:, None]).squeeze(1)
            if hasattr(self, 'keep_order') and self.keep_order or keep_order:
                nll.index_copy_(0, indices_i, -logprob_i)
            else:
                nll[offset:offset + logprob_i.size(0)].copy_(-logprob_i)
            offset += logprob_i.size(0)
        return nll


def scatter(inputs, target_gpus, chunk_sizes, dim=0):
    """
    Slices tensors into approximately equal chunks and
    distributes them across given GPUs. Duplicates
    references to objects that are not tensors.
    """

    def scatter_map(obj):
        if isinstance(obj, torch.Tensor):
            try:
                return Scatter.apply(target_gpus, chunk_sizes, dim, obj)
            except:
                None
                None
                None
                quit()
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            return list(map(list, zip(*map(scatter_map, obj))))
        if isinstance(obj, dict) and len(obj) > 0:
            return list(map(type(obj), zip(*map(scatter_map, obj.items()))))
        return [obj for targets in target_gpus]
    try:
        return scatter_map(inputs)
    finally:
        scatter_map = None


def scatter_kwargs(inputs, kwargs, target_gpus, chunk_sizes, dim=0):
    """Scatter with support for kwargs dictionary"""
    inputs = scatter(inputs, target_gpus, chunk_sizes, dim) if inputs else []
    kwargs = scatter(kwargs, target_gpus, chunk_sizes, dim) if kwargs else []
    if len(inputs) < len(kwargs):
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
    elif len(kwargs) < len(inputs):
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs


class BalancedDataParallel(DataParallel):

    def __init__(self, gpu0_bsz, *args, **kwargs):
        self.gpu0_bsz = gpu0_bsz
        super().__init__(*args, **kwargs)

    def forward(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module(*inputs, **kwargs)
        if self.gpu0_bsz == 0:
            device_ids = self.device_ids[1:]
        else:
            device_ids = self.device_ids
        inputs, kwargs = self.scatter(inputs, kwargs, device_ids)
        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])
        replicas = self.replicate(self.module, self.device_ids)
        if self.gpu0_bsz == 0:
            replicas = replicas[1:]
        outputs = self.parallel_apply(replicas, device_ids, inputs, kwargs)
        return self.gather(outputs, self.output_device)

    def parallel_apply(self, replicas, device_ids, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs, device_ids)

    def scatter(self, inputs, kwargs, device_ids):
        bsz = inputs[0].size(self.dim)
        num_dev = len(self.device_ids)
        gpu0_bsz = self.gpu0_bsz
        bsz_unit = (bsz - gpu0_bsz) // (num_dev - 1)
        if gpu0_bsz < bsz_unit:
            chunk_sizes = [gpu0_bsz] + [bsz_unit] * (num_dev - 1)
            delta = bsz - sum(chunk_sizes)
            for i in range(delta):
                chunk_sizes[i + 1] += 1
            if gpu0_bsz == 0:
                chunk_sizes = chunk_sizes[1:]
        else:
            return super().scatter(inputs, kwargs, device_ids)
        return scatter_kwargs(inputs, kwargs, device_ids, chunk_sizes, dim=self.dim)


def get_rank_0_in_comm(comm):
    world_size = dist.get_world_size(comm)
    x = torch.tensor([dist.get_rank()], dtype=torch.int64, device='cuda')
    ys = [torch.empty_like(x) for _ in range(world_size)]
    dist.all_gather(ys, x, group=comm)
    root_rank = ys[0].item()
    return root_rank


class DistributedGroupedDataParallel(nn.Module):
    """
    A customized DDP module to support different all-reduce regions in the
    model.  The all-reduce region is defined as an attribution `dp_comm` in the
    weight object.
    The grads of the weights are identified to be reduced in different groups
    according to the weigths' `dp_comm` attribute.
    If it is set to `dp`, it will only be reduced across the data-parallel
    groups, which means that in the model parallel group, they are not
    synchronized.
    If it is set to `world`, the gradients is synchronized across all workers,
    regardless their model or data parallel group. This is extremely useful for
    shared layers like the gate.
    """

    def __init__(self, module, auto_allreduce=False, need_sync=True, **kwargs):
        assert not auto_allreduce, 'Automatic all-reduce is not implemented yet'
        super().__init__()
        self.module = module
        self.comms = dict()
        for k in kwargs:
            if k.endswith('_group'):
                self.comms[k[:-6]] = kwargs[k]
        for k in ['dp', 'gate', 'moe', 'world']:
            if k not in self.comms:
                self.comms[k] = get_torch_default_comm()

        def allreduce_params(no_scale=False, reduce_after=False, fp32_allreduce=False):
            groups = dict()
            for p in self.module.parameters():
                if not p.requires_grad or p.grad is None:
                    continue
                if hasattr(p, 'dp_comm'):
                    dp_comm = p.dp_comm
                else:
                    dp_comm = 'dp'
                group_key = dp_comm, p.dtype
                if group_key not in groups:
                    groups[group_key] = [p]
                else:
                    groups[group_key].append(p)
            for (dp_comm, dtype), group in groups.items():
                if dp_comm not in self.comms:
                    continue
                comm = self.comms[dp_comm]
                grads = [p.grad.data for p in group]
                coalesced = _flatten_dense_tensors(grads)
                if fp32_allreduce and dtype != torch.float32:
                    coalesced = coalesced.float()
                if not no_scale and not reduce_after:
                    coalesced /= comm.size()
                torch.distributed.all_reduce(coalesced, group=comm)
                torch.cuda.synchronize()
                if not no_scale and reduce_after:
                    coalesced /= comm.size()
                synced = _unflatten_dense_tensors(coalesced, grads)
                for g, s in zip(grads, synced):
                    g.copy_(s)
        self.allreduce_params = allreduce_params
        if need_sync:
            self._sync_params()

    def _sync_params(self):
        groups = dict()
        for p in self.module.parameters():
            if hasattr(p, 'dp_comm'):
                dp_comm = p.dp_comm
            else:
                dp_comm = 'dp'
            group_key = dp_comm, p.dtype
            if group_key not in groups:
                groups[group_key] = [p]
            else:
                groups[group_key].append(p)
        for (dp_comm, _), group in groups.items():
            if dp_comm not in self.comms:
                continue
            comm = self.comms[dp_comm]
            datas = [p.data for p in group]
            coalesced = _flatten_dense_tensors(datas)
            torch.distributed.broadcast(coalesced, get_rank_0_in_comm(comm), group=comm)
            torch.cuda.synchronize()
            synced = _unflatten_dense_tensors(coalesced, datas)
            for d, s in zip(datas, synced):
                d.copy_(s)

    def forward(self, *args, **kwargs):
        """
        Directly call the module's forward function.
        """
        return self.module(*args, **kwargs)


class NoisyGate(BaseGate):

    def __init__(self, d_model, num_expert, world_size, top_k=2):
        super().__init__(num_expert, world_size)
        self.w_gate = nn.Parameter(torch.zeros(d_model, self.tot_expert), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(d_model, self.tot_expert), requires_grad=True)
        self.top_k = top_k
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.noise_epsilon = 0.01
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.w_gate, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.w_noise, a=math.sqrt(5))

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()
        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.top_k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        normal = Normal(torch.tensor([0.0], device=clean_values.device), torch.tensor([1.0], device=clean_values.device))
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.Tensor([0])
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def forward(self, inp):
        clean_logits = inp @ self.w_gate
        raw_noise_stddev = inp @ self.w_noise
        noise_stddev = (self.softplus(raw_noise_stddev) + self.noise_epsilon) * self.training
        noisy_logits = clean_logits + torch.randn_like(clean_logits) * noise_stddev
        logits = noisy_logits
        top_logits, top_indices = logits.topk(min(self.top_k + 1, self.tot_expert), dim=1)
        top_k_logits = top_logits[:, :self.top_k]
        top_k_indices = top_indices[:, :self.top_k]
        top_k_gates = self.softmax(top_k_logits)
        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)
        if self.top_k < self.tot_expert:
            load = self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits).sum(0)
        else:
            load = self._gates_to_load(gates)
        importance = gates.sum(0)
        loss = self.cv_squared(importance) + self.cv_squared(load)
        self.set_loss(loss)
        return top_k_indices.contiguous().view(-1), top_k_gates.contiguous().unsqueeze(1)


class SwipeGate(NaiveGate):

    def __init__(self, d_model, num_expert, world_size, top_k=2):
        super().__init__(d_model, num_expert, world_size, top_k)

    def swipe_once(self, idx, capacity, bias):
        with torch.no_grad():
            idx_new, capacity = fmoe_native.swipe_once(idx, capacity, self.num_expert, self.world_size, bias)
            idx_new = idx_new
        return idx_new, capacity

    def forward(self, inp):
        score = self.gate(inp)
        orig_score, orig_idx = torch.topk(score, k=self.top_k, dim=-1)
        if not self.training:
            topk_val = F.softmax(orig_score, dim=-1)
            return orig_idx, topk_val
        capacity = torch.scalar_tensor(inp.shape[0] * self.top_k, dtype=torch.long)
        topk_idxs = []
        topk_vals = []
        idx_x = torch.arange(inp.shape[0], device=inp.device)
        for k in range(self.top_k):
            idx, capacity = self.swipe_once(orig_idx[:, k], capacity, k % self.num_expert)
            topk_vals.append(score[idx_x, idx])
            topk_idxs.append(idx)
        topk_idx = torch.stack(topk_idxs).transpose(0, 1)
        topk_val = torch.stack(topk_vals).transpose(0, 1)
        topk_val = F.softmax(topk_val, dim=-1)
        return topk_idx, topk_val


def limit_by_capacity(topk_idx, num_expert, world_size, capacity):
    with torch.no_grad():
        capacity = torch.ones(num_expert, dtype=torch.int32, device=topk_idx.device) * capacity
        pos, lec, gec = count_by_gate(topk_idx, num_expert, world_size, require_pos=False)
        new_gec = fmoe_native.limit_by_capacity(gec, capacity, num_expert, world_size)
        if world_size > 1:
            new_lec = fmoe_native.expert_exchange(new_gec, num_expert, world_size)
        else:
            new_lec = new_gec
        topk_idx = fmoe_native.prune_gate_by_capacity(topk_idx, new_lec, num_expert, world_size)
    return new_lec, new_gec, topk_idx


class SwitchGate(NaiveGate):
    """
    A switch gate implementation
    """

    def __init__(self, d_model, num_expert, world_size, topk=1, switch_eps=0.1, capacity=(1.2, 2.4)):
        assert topk == 1, 'topk should be 1 in switch'
        super().__init__(d_model, num_expert, world_size, top_k=1)
        self.switch_eps = switch_eps
        self.capacity = capacity

    def forward(self, inp):
        """
        The switch firstly conduct softmax and then calculates the top-1
        """
        score = self.gate(inp)
        if self.training:
            noise = torch.rand_like(score)
            noise = noise * 2 * self.switch_eps + 1.0 - self.switch_eps
            score += noise
        score = F.softmax(score.float(), dim=-1)
        top1_score, top1_idx = torch.topk(score, k=1, dim=-1, largest=True)
        top1_score = top1_score
        cap_rate = self.capacity[0 if self.training else 1]
        capacity = math.ceil(cap_rate * inp.shape[0])
        _new_lec, _new_gec, top1_idx = limit_by_capacity(top1_idx, self.num_expert, self.world_size, capacity)
        valid_idx = top1_idx[top1_idx > -1]
        fraction_expert = torch.scatter_add(torch.zeros(self.tot_expert, device=valid_idx.device), 0, valid_idx, torch.ones_like(valid_idx, dtype=torch.float)) / valid_idx.numel()
        prob_expert = score.sum(dim=0) / valid_idx.numel()
        loss = (fraction_expert * prob_expert).sum() * self.tot_expert
        self.set_loss(loss)
        return top1_idx, top1_score


class ZeroGate(BaseGate):
    """
    Guide all input samples to gate 0.
    """

    def __init__(self, _1, num_expert, world_size, top_k=2):
        super().__init__(num_expert, world_size)
        self.top_k = top_k

    def forward(self, inp):
        """
        All output to expert 1
        """
        idx = torch.zeros(inp.shape[0] * self.top_k, dtype=torch.int64, device=inp.device)
        gate_score = torch.ones(inp.shape[0] * self.top_k, device=inp.device) / self.top_k
        return idx, gate_score.reshape(-1, 1, self.top_k)


_groups = None


def _set_groups(**kwargs):
    global _groups
    _groups = kwargs


def _init():
    args = get_args()
    stage_size = args.world_size // args.pipeline_model_parallel_size
    for i in range(0, args.world_size, stage_size):
        ranks = range(i, i + stage_size)
        group = torch.distributed.new_group(ranks)
        if args.rank in ranks:
            gate_group = group
    _set_groups(dp_group=mpu.get_data_parallel_group(), moe_group=mpu.get_data_parallel_group(), gate_group=gate_group)


class DistributedDataParallel(DistributedGroupedDataParallel):
    """
    A wrapper that is used to replace the DDP module provided by Megatron, which
    is adapted to enable the sophiscated parallel and reduction strategies in
    Fast MoE.
    """

    def __init__(self, module):
        if _groups is None:
            _init()
        super().__init__(module, **_groups)

    def state_dict(self, *args, **kwargs):
        """
        Keep consitency with Megatron
        """
        return self.module.state_dict(*args, **kwargs)

    def state_dict_for_save_checkpoint(self, *args, **kwargs):
        """
        Keep consitency with Megatron
        """
        return self.module.state_dict_for_save_checkpoint(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        """
        Keep consitency with Megatron
        """
        return self.module.load_state_dict(*args, **kwargs)


class _FakeMegatronMLP(nn.Module):
    """
    A fake mlp without model parallelism for correctness testing
    """

    def __init__(self, args, _):
        super().__init__()
        self.fc1 = nn.Linear(args.hidden_size, args.hidden_hidden_size)
        self.fc2 = nn.Linear(args.hidden_hidden_size, args.hidden_size)

    def forward(self, x):
        """
        Directly use GeLU
        """
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x, torch.zeros_like(x)


class BruteForceMoE(nn.Module):

    def __init__(self, expert, num_expert=32, d_model=1024, world_size=1, top_k=2):
        super(BruteForceMoE, self).__init__()
        self.num_expert = num_expert
        self.d_model = d_model
        self.top_k = top_k
        if type(expert) is list:
            self.experts = [e(d_model) for e in expert]
            self.num_expert = num_expert = len(expert)
        else:
            self.experts = [expert(d_model) for _ in range(num_expert * world_size)]

    def forward(self, inp, gate_idx, gate_score):
        inp = inp.repeat_interleave(repeats=self.top_k, dim=0)
        gate_long = gate_idx.long().view(-1)
        batch_size = inp.size(0)
        x = inp.new_zeros((batch_size, self.d_model))
        for i in range(batch_size):
            x[i] = self.experts[gate_long[i]](inp[i])
        gate_score = gate_score.unsqueeze(1)
        x = torch.bmm(gate_score, x.view(-1, self.top_k, self.d_model)).reshape(-1, self.d_model)
        return x


class BruteForceMoELinear(nn.Module):

    def __init__(self, activation, num_expert=32, d_model=1024, d_hidden=2048, world_size=1, top_k=2):
        super(BruteForceMoELinear, self).__init__()
        self.num_expert = num_expert
        self.d_model = d_model
        self.activation = activation
        self.weight_htoh4 = nn.Parameter(torch.Tensor(num_expert * world_size, d_hidden, d_model))
        self.bias_htoh4 = nn.Parameter(torch.Tensor(num_expert * world_size, d_hidden))
        self.weight_h4toh = nn.Parameter(torch.Tensor(num_expert * world_size, d_model, d_hidden))
        self.bias_h4toh = nn.Parameter(torch.Tensor(num_expert * world_size, d_model))
        self.top_k = top_k

    def forward(self, inp, gate_idx, gate_score):
        inp = inp.repeat_interleave(repeats=self.top_k, dim=0)
        gate_long = gate_idx.long().view(-1)
        batch_size = inp.size(0)
        o = torch.empty(batch_size, self.d_model, dtype=inp.dtype, device=inp.device)
        for i in range(self.weight_htoh4.shape[0]):
            idx = gate_long == i
            x = inp[idx]
            x = x @ self.weight_htoh4[i].t()
            x = x + self.bias_htoh4[i]
            x = self.activation(x)
            x = x @ self.weight_h4toh[i].t()
            x = x + self.bias_h4toh[i]
            o[idx] = x
        gate_score = gate_score.unsqueeze(1)
        x = torch.bmm(gate_score, o.view(-1, self.top_k, self.d_model)).reshape(-1, self.d_model)
        return x


class NaiveExpert(nn.Module):

    def __init__(self, d_model):
        super(NaiveExpert, self).__init__()
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, x):
        return self.linear(x)


class LinearExpert(nn.Module):

    def __init__(self, d_model):
        super(LinearExpert, self).__init__()
        self.model = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.ReLU(), nn.Linear(d_model * 2, d_model))

    def forward(self, x):
        return self.model(x)


def _megatron_init_method(self, rng, sigma):
    """
    Init method based on N(0, sigma).
    Copied from Megatron-LM
    """
    device = self.weight.device
    dtype = self.weight.dtype
    weight = rng.normal(loc=0.0, scale=sigma, size=tuple(self.weight.size()))
    self.weight.data = torch.from_numpy(weight)
    if self.bias is not None:
        with torch.no_grad():
            self.bias.zero_()


class MyMoE(FMoE):

    def __init__(self, num_expert, d_model, d_hidden, world_size, mp_group, top_k, activation):
        super().__init__(num_expert=num_expert, d_model=d_model, gate=NaiveGate, world_size=world_size, slice_group=mp_group, top_k=top_k)
        self.experts = _Expert(num_expert, d_model, d_hidden, activation)
        rng = np.random.default_rng(1234)
        _megatron_init_method(self.experts.htoh4, rng, 1.0)
        _megatron_init_method(self.experts.h4toh, rng, 1.0)


class MyExpert(nn.Module):
    """
    An expert using 2 FMoELinear modules to speed up the computation of experts
    within one worker.
    """

    def __init__(self, num_expert, d_model, d_hidden, activation, rank=0):
        super().__init__()
        self.htoh4 = FMoELinear(num_expert, d_model, d_hidden, bias=True, rank=rank)
        self.h4toh = FMoELinear(num_expert, d_hidden, d_model, bias=True, rank=rank)
        self.activation = activation

    def forward(self, inp, fwd_expert_count):
        """
        First expand input to 4h (the hidden size is variable, but is called h4
        for convenience). Then perform activation. Finally shirink back to h.
        """
        if type(inp) == dict:
            x = inp['x']
            y = inp['y']
        elif type(inp) == list:
            x = inp[0]
            y = inp[1]
        else:
            raise NotImplementedError
        x = self.htoh4(x, fwd_expert_count)
        x = self.activation(x)
        x = self.h4toh(x, fwd_expert_count)
        y = self.htoh4(y, fwd_expert_count)
        y = self.activation(y)
        y = self.h4toh(y, fwd_expert_count)
        if type(inp) == dict:
            ret = {'x': x, 'y': y}
        elif type(inp) == list:
            ret = [x, y]
        return ret


class MyGate(NaiveGate):

    def __init__(self, d_model, num_expert, world_size, top_k=2):
        super().__init__(d_model, num_expert, world_size, top_k)

    def forward(self, inp, return_all_scores=False):
        if type(inp) == dict:
            x = inp['x']
        elif type(inp) == list:
            x = inp[0]
        else:
            raise NotImplementedError
        return super().forward(x, return_all_scores)


class MyModule(nn.Module):

    def __init__(self, dim=8):
        super(MyModule, self).__init__()
        self.model = nn.Sequential(OrderedDict([('linear1', nn.Linear(dim, dim)), ('relu1', nn.ReLU()), ('linear2', nn.Linear(dim, dim)), ('relu2', nn.ReLU()), ('linear3', nn.Linear(dim, dim))]))

    def set_comm(self):
        for p in self.model._modules['linear1'].parameters():
            setattr(p, 'dp_comm', 'mp')
        for p in self.model._modules['linear2'].parameters():
            setattr(p, 'dp_comm', 'dp')
        for p in self.model._modules['linear3'].parameters():
            setattr(p, 'dp_comm', 'world')

    def forward(self, inp):
        return self.model(inp)


class ConstantGate(torch.nn.Module):

    def __init__(self, d_model, num_expert, world_size, top_k=1):
        super().__init__()
        self.top_k = top_k

    def forward(self, inp):
        idx = torch.zeros((inp.shape[0], self.top_k), dtype=torch.int64, device=inp.device)
        score = torch.ones((inp.shape[0], 1, self.top_k), device=inp.device) / 2
        return idx, score


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ConstantGate,
     lambda: ([], {'d_model': 4, 'num_expert': 4, 'world_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LinearExpert,
     lambda: ([], {'d_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MultiHeadAttn,
     lambda: ([], {'n_head': 4, 'd_model': 4, 'd_head': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (NaiveExpert,
     lambda: ([], {'d_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (NaiveGate,
     lambda: ([], {'d_model': 4, 'num_expert': 4, 'world_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (PositionwiseFF,
     lambda: ([], {'d_model': 4, 'd_inner': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SwipeGate,
     lambda: ([], {'d_model': 4, 'num_expert': 4, 'world_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ZeroGate,
     lambda: ([], {'_1': 4, 'num_expert': 4, 'world_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (_FakeMegatronMLP,
     lambda: ([], {'args': _mock_config(hidden_size=4, hidden_hidden_size=4), '_': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_laekov_fastmoe(_paritybench_base):
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

