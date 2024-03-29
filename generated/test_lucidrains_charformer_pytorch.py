import sys
_module = sys.modules[__name__]
del sys
charformer_pytorch = _module
charformer_pytorch = _module
setup = _module

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


from math import gcd


import functools


import torch


import torch.nn.functional as F


from torch import nn


from torch import einsum


class Pad(nn.Module):

    def __init__(self, padding, value=0.0):
        super().__init__()
        self.padding = padding
        self.value = value

    def forward(self, x):
        return F.pad(x, self.padding, value=self.value)


class DepthwiseConv1d(nn.Module):

    def __init__(self, dim_in, dim_out, kernel_size):
        super().__init__()
        self.conv = nn.Conv1d(dim_in, dim_out, kernel_size, groups=dim_in)
        self.proj_out = nn.Conv1d(dim_out, dim_out, 1)

    def forward(self, x):
        x = self.conv(x)
        return self.proj_out(x)


def exists(val):
    return val is not None


def lcm(*numbers):
    return int(functools.reduce(lambda x, y: int(x * y / gcd(x, y)), numbers, 1))


def next_divisible_length(seqlen, multiple):
    return math.ceil(seqlen / multiple) * multiple


def pad_to_multiple(tensor, multiple, *, seq_dim, dim=-1, value=0.0):
    seqlen = tensor.shape[seq_dim]
    length = next_divisible_length(seqlen, multiple)
    if length == seqlen:
        return tensor
    remainder = length - seqlen
    pad_offset = (0,) * (-1 - dim) * 2
    return F.pad(tensor, (*pad_offset, 0, remainder), value=value)


class GBST(nn.Module):

    def __init__(self, *, num_tokens, dim, max_block_size=None, blocks=None, downsample_factor=4, score_consensus_attn=True):
        super().__init__()
        assert exists(max_block_size) ^ exists(blocks), 'either max_block_size or blocks are given on initialization'
        self.token_emb = nn.Embedding(num_tokens, dim)
        if exists(blocks):
            assert isinstance(blocks, tuple), 'blocks must be a tuple of block sizes'
            self.blocks = tuple(map(lambda el: el if isinstance(el, tuple) else (el, 0), blocks))
            assert all([(offset < block_size) for block_size, offset in self.blocks]), 'offset must be always smaller than the block size'
            max_block_size = max(list(map(lambda t: t[0], self.blocks)))
        else:
            self.blocks = tuple(map(lambda el: (el, 0), range(1, max_block_size + 1)))
        self.pos_conv = nn.Sequential(Pad((0, 0, 0, max_block_size - 1)), Rearrange('b n d -> b d n'), DepthwiseConv1d(dim, dim, kernel_size=max_block_size), Rearrange('b d n -> b n d'))
        self.score_fn = nn.Sequential(nn.Linear(dim, 1), Rearrange('... () -> ...'))
        self.score_consensus_attn = score_consensus_attn
        assert downsample_factor <= max_block_size, 'final downsample factor should be less than the maximum block size'
        self.block_pad_multiple = lcm(*[block_size for block_size, _ in self.blocks])
        self.downsample_factor = downsample_factor

    def forward(self, x, mask=None):
        b, n, block_mult, ds_factor, device = *x.shape, self.block_pad_multiple, self.downsample_factor, x.device
        m = next_divisible_length(n, ds_factor)
        x = self.token_emb(x)
        x = self.pos_conv(x)
        x = pad_to_multiple(x, block_mult, seq_dim=1, dim=-2)
        if exists(mask):
            mask = pad_to_multiple(mask, block_mult, seq_dim=1, dim=-1, value=False)
        block_masks = []
        block_reprs = []
        for block_size, offset in self.blocks:
            block_x = x.clone()
            if exists(mask):
                block_mask = mask.clone()
            need_padding = offset > 0
            if need_padding:
                left_offset, right_offset = block_size - offset, offset
                block_x = F.pad(block_x, (0, 0, left_offset, right_offset), value=0.0)
                if exists(mask):
                    block_mask = F.pad(block_mask, (left_offset, right_offset), value=False)
            blocks = rearrange(block_x, 'b (n m) d -> b n m d', m=block_size)
            if exists(mask):
                mask_blocks = rearrange(block_mask, 'b (n m) -> b n m', m=block_size)
                block_repr = masked_mean(blocks, mask_blocks, dim=-2)
            else:
                block_repr = blocks.mean(dim=-2)
            block_repr = repeat(block_repr, 'b n d -> b (n m) d', m=block_size)
            if need_padding:
                block_repr = block_repr[:, left_offset:-right_offset]
            block_reprs.append(block_repr)
            if exists(mask):
                mask_blocks = torch.any(mask_blocks, dim=-1)
                mask_blocks = repeat(mask_blocks, 'b n -> b (n m)', m=block_size)
                if need_padding:
                    mask_blocks = mask_blocks[:, left_offset:-right_offset]
                block_masks.append(mask_blocks)
        block_reprs = torch.stack(block_reprs, dim=2)
        scores = self.score_fn(block_reprs)
        if exists(mask):
            block_masks = torch.stack(block_masks, dim=2)
            max_neg_value = -torch.finfo(scores.dtype).max
            scores = scores.masked_fill(~block_masks, max_neg_value)
        scores = scores.softmax(dim=2)
        if self.score_consensus_attn:
            score_sim = einsum('b i d, b j d -> b i j', scores, scores)
            if exists(mask):
                cross_mask = rearrange(mask, 'b i -> b i ()') * rearrange(mask, 'b j -> b () j')
                max_neg_value = -torch.finfo(score_sim.dtype).max
                score_sim = score_sim.masked_fill(~cross_mask, max_neg_value)
            score_attn = score_sim.softmax(dim=-1)
            scores = einsum('b i j, b j m -> b i m', score_attn, scores)
        scores = rearrange(scores, 'b n m -> b n m ()')
        x = (block_reprs * scores).sum(dim=2)
        x = x[:, :m]
        if exists(mask):
            mask = mask[:, :m]
        x = rearrange(x, 'b (n m) d -> b n m d', m=ds_factor)
        if exists(mask):
            mask = rearrange(mask, 'b (n m) -> b n m', m=ds_factor)
            x = masked_mean(x, mask, dim=2)
            mask = torch.any(mask, dim=-1)
        else:
            x = x.mean(dim=-2)
        return x, mask


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (DepthwiseConv1d,
     lambda: ([], {'dim_in': 4, 'dim_out': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
]

class Test_lucidrains_charformer_pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

