import sys
_module = sys.modules[__name__]
del sys
examples = _module
dcgan = _module
imagenet = _module
mnist = _module
setup = _module
torchdp = _module
autograd_grad_sample = _module
dp_model_inspector = _module
layers = _module
dp_multihead_attention = _module
per_sample_gradient_clip = _module
privacy_analysis = _module
privacy_engine = _module
compute_dp_sgd_privacy = _module
stats = _module
supported_layers_grad_samplers = _module
test = _module
dp_layers_test = _module
dp_model_inspector_test = _module
layers_grad_test = _module
per_sample_gradient_clip_test = _module
privacy_engine_test = _module
utils_test = _module
utils = _module

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


import random


import torch


import torch.nn as nn


import torch.nn.parallel


import torch.backends.cudnn as cudnn


import torch.optim as optim


import torch.utils.data


import torchvision.datasets as dset


import torchvision.transforms as transforms


import torchvision.utils as vutils


import time


import warnings


import torch.distributed as dist


import torch.multiprocessing as mp


import torch.optim


import torch.utils.data.distributed


import torch.utils.tensorboard as tensorboard


import torchvision.datasets as datasets


import torchvision.models as models


import numpy as np


import torch.nn.functional as F


from torchvision import datasets


from torchvision import transforms


from typing import List


from torch import nn


from torch.nn.parameter import Parameter


import types


from typing import Union


from torch.functional import F


from torch.nn.modules.activation import MultiheadAttention


from torchvision import models


from torch.utils.data import DataLoader


from torchvision.datasets import FakeData


from typing import Tuple


import math


from enum import IntEnum


from typing import Callable


from typing import Type


parser = argparse.ArgumentParser(description='PyTorch ImageNet DP Training')


opt = parser.parse_args()


class Generator(nn.Module):

    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False), nn.BatchNorm2d(ngf * 8), nn.ReLU(True), nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 4), nn.ReLU(True), nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 2), nn.ReLU(True), nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf), nn.ReLU(True), nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False), nn.Tanh())

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class Discriminator(nn.Module):

    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(nn.Conv2d(nc, ndf, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 4), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 8), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False), nn.Sigmoid())

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output.view(-1, 1).squeeze(1)


class SampleConvNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 8, 2, padding=3)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.fc1 = nn.Linear(32 * 4 * 4, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 1)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 1)
        x = x.view(-1, 32 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def name(self):
        return 'SampleConvNet'


class SequenceBias(nn.Module):
    """ Adds one bias element to the end of the sequence
    Args:
        embed_dim: Embedding dimension

    Shape:
        - Input: (L, N, E), where
            L - sequence length, N - batch size, E - embedding dimension
        - Output: (L+1, N, E), where
            L - sequence length, N - batch size, E - embedding dimension

    Attributes:
        bias:   the learnable bias of the module of shape (E),
            where E - embedding dimension

    Examples::

        >>> m = SequenceBias(16)
        >>> input = torch.randn(20, 4, 16)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([21, 4, 16])
    """

    def __init__(self, embed_dim):
        super(SequenceBias, self).__init__()
        self.bias = Parameter(torch.empty(embed_dim))
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.bias)

    def forward(self, x):
        _, bsz, _ = x.shape
        return torch.cat([x, self.bias.repeat(1, bsz, 1)])


class DPMultiheadAttention(nn.Module):
    """ This is DP-friendly implementation of nn.MultiheadAttention.
    For full reference see original module:
    https://pytorch.org/docs/stable/nn.html#torch.nn.MultiheadAttention

    Current implementation leverages pytorch modules as building blocks
    to allow DP engine to calculate per-sample gradients.
    This is in contrast with original implementation based on nn.functional.
    """

    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None):
        super(DPMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, 'embed_dim must be divisible by num_heads'
        self.qlinear = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.klinear = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.vlinear = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.add_bias_kv = add_bias_kv
        if self.add_bias_kv:
            self.seq_bias_k = SequenceBias(embed_dim)
            self.seq_bias_v = SequenceBias(embed_dim)
        self.add_zero_attn = add_zero_attn
        self.dropout = nn.Dropout(dropout)

    def load_state_dict(self, state_dict):
        """ Loads module from previously saved state.
        Supports loading from both DPMultiheadAttention
        and nn.MultiheadAttention modules
        """
        if 'in_proj_weight' in state_dict:
            qweight, kweight, vweight = state_dict['in_proj_weight'].chunk(3, dim=0)
            state_dict['qlinear.weight'] = qweight
            state_dict['klinear.weight'] = kweight
            state_dict['vlinear.weight'] = vweight
            del state_dict['in_proj_weight']
        if 'in_proj_bias' in state_dict:
            qbias, kbias, vbias = state_dict['in_proj_bias'].chunk(3, dim=0)
            state_dict['qlinear.bias'] = qbias
            state_dict['klinear.bias'] = kbias
            state_dict['vlinear.bias'] = vbias
            del state_dict['in_proj_bias']
        if 'bias_k' in state_dict:
            state_dict['seq_bias_k.bias'] = state_dict['bias_k'].squeeze()
            del state_dict['bias_k']
        if 'bias_v' in state_dict:
            state_dict['seq_bias_v.bias'] = state_dict['bias_v'].squeeze()
            del state_dict['bias_v']
        if 'q_proj_weight' in state_dict:
            state_dict['qlinear.weight'] = state_dict['q_proj_weight']
            del state_dict['q_proj_weight']
        if 'k_proj_weight' in state_dict:
            state_dict['klinear.weight'] = state_dict['k_proj_weight']
            del state_dict['k_proj_weight']
        if 'v_proj_weight' in state_dict:
            state_dict['vlinear.weight'] = state_dict['v_proj_weight']
            del state_dict['v_proj_weight']
        super(DPMultiheadAttention, self).load_state_dict(state_dict)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None):
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert key.size() == value.size()
        head_dim = embed_dim // self.num_heads
        assert head_dim * self.num_heads == embed_dim, 'embed_dim must be divisible by num_heads'
        scaling = float(head_dim) ** -0.5
        q = self.qlinear(query)
        k = self.klinear(key)
        v = self.vlinear(value)
        q = q * scaling
        if attn_mask is not None:
            assert attn_mask.dtype == torch.float32 or attn_mask.dtype == torch.float64 or attn_mask.dtype == torch.uint8 or attn_mask.dtype == torch.bool, f'Only float, byte, and bool types are supported for attn_mask,'
            """not {attn_mask.dtype}"""
            if attn_mask.dtype == torch.uint8:
                warnings.warn('Byte tensor for attn_mask in nn.MultiheadAttention is deprecated.Use bool tensor instead.')
                attn_mask = attn_mask
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
                if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                    raise RuntimeError('The size of the 2D attn_mask is not correct.')
            elif attn_mask.dim() == 3:
                if list(attn_mask.size()) != [bsz * self.num_heads, query.size(0), key.size(0)]:
                    raise RuntimeError('The size of the 3D attn_mask is not correct.')
            else:
                raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))
        if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
            warnings.warn('Byte tensor for key_padding_mask in nn.MultiheadAttentionis deprecated. Use bool tensor instead.')
            key_padding_mask = key_padding_mask
        if self.add_bias_kv:
            k = self.seq_bias_k(k)
            v = self.seq_bias_v(v)
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
        src_len = k.size(1)
        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len
        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device)], dim=1)
            v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device)], dim=1)
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))
        attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_output_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_output_weights.masked_fill_(attn_mask, float('-inf'))
            else:
                attn_output_weights += attn_mask
        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_output_weights = attn_output_weights.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
            attn_output_weights = attn_output_weights.view(bsz * self.num_heads, tgt_len, src_len)
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = self.dropout(attn_output_weights)
        attn_output = torch.bmm(attn_output_weights, v)
        assert list(attn_output.size()) == [bsz * self.num_heads, tgt_len, head_dim]
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)
        if need_weights:
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
            return attn_output, attn_output_weights.sum(dim=1) / self.num_heads
        else:
            return attn_output, None


class SampleConvNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 8, 3)
        self.conv2 = nn.Conv1d(16, 32, 3, 1)
        self.convf = nn.Conv1d(32, 32, 1, 1)
        for p in self.convf.parameters():
            p.requires_grad = False
        self.fc1 = nn.Linear(23, 17)
        self.fc2 = nn.Linear(32 * 17, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        x = F.relu(self.conv2(x))
        x = self.convf(x)
        x = self.fc1(x)
        x = x.view(-1, x.shape[-2] * x.shape[-1])
        x = self.fc2(x)
        return x

    def name(self):
        return 'SampleConvNet'


class SampleConvNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 8, 3)
        self.gnorm1 = nn.GroupNorm(4, 16)
        self.conv2 = nn.Conv1d(16, 32, 3, 1)
        self.lnorm1 = nn.LayerNorm((32, 23))
        self.conv3 = nn.Conv1d(32, 32, 3, 1)
        self.instnorm1 = nn.InstanceNorm1d(32, affine=True)
        self.convf = nn.Conv1d(32, 32, 1, 1)
        for p in self.convf.parameters():
            p.requires_grad = False
        self.fc1 = nn.Linear(21, 17)
        self.lnorm2 = nn.LayerNorm(17)
        self.fc2 = nn.Linear(32 * 17, 10)
        for layer in (self.gnorm1, self.lnorm1, self.lnorm2, self.instnorm1):
            nn.init.uniform_(layer.weight)
            nn.init.uniform_(layer.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.gnorm1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        x = self.conv2(x)
        x = self.lnorm1(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.instnorm1(x)
        x = self.convf(x)
        x = self.fc1(x)
        x = self.lnorm2(x)
        x = x.view(-1, x.shape[-2] * x.shape[-1])
        x = self.fc2(x)
        return x

    def name(self):
        return 'SampleConvNet'


class BasicModel(nn.Module):

    def __init__(self, imgSize):
        super().__init__()
        self.size = imgSize[0] * imgSize[1] * imgSize[2]
        self.bn = nn.BatchNorm2d(imgSize[0])
        self.fc = nn.Linear(self.size, 2)

    def forward(self, input):
        x = self.bn(input)
        x = x.view(-1, self.size)
        x = self.fc(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicModel,
     lambda: ([], {'imgSize': [4, 4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DPMultiheadAttention,
     lambda: ([], {'embed_dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (SequenceBias,
     lambda: ([], {'embed_dim': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
]

class Test_facebookresearch_pytorch_dp(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

