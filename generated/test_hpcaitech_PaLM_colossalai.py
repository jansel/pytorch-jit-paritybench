import sys
_module = sys.modules[__name__]
del sys
palm_16b_tp1_zero = _module
palm_16b_tp2_zero = _module
palm_16b_tp4 = _module
palm_30b_1d = _module
palm_30b_2d = _module
palm_30b_2p5d = _module
palm_30b_3d = _module
palm_30b_zero = _module
palm_62b_1d = _module
palm_62b_2d = _module
palm_62b_2p5d = _module
palm_62b_3d = _module
palm_8b_1d = _module
palm_8b_2d = _module
palm_8b_2p5d = _module
palm_8b_3d = _module
palm_8b_zero = _module
palm_8b_zero_1d = _module
palm_8b_zero_2d = _module
palm_8b_zero_2p5d = _module
palm_8b_zero_3d = _module
palm_small_1d = _module
palm_small_2d = _module
palm_small_2p5d = _module
palm_small_3d = _module
data = _module
randomtext = _module
wikitext = _module
model = _module
loss = _module
palm_utils = _module
parallel_palm = _module
parallel_utils = _module
test_build = _module
test_parallel_transformer_layer = _module
download_wiki = _module
train = _module
utils = _module

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


import numpy as np


import random


import string


import copy


from itertools import chain


from torch.utils.data import DataLoader


from torch.utils.data import DistributedSampler


from torch.distributed import get_world_size


from torch import nn


import torch.nn.functional as F


from torch import einsum


import torch.nn as nn


from torch import dtype


import torch.multiprocessing as mp


from functools import partial


import torch.distributed as dist


class PaLMLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss = colossalai.nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        return self.loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


class RotaryEmbedding(nn.Module):

    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / 10000 ** (torch.arange(0, dim, 2).float() / dim)
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, max_seq_len, *, device):
        seq = torch.arange(max_seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = einsum('i , j -> i j', seq, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)


class SwiGLU(nn.Module):

    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


class PaLMHead(nn.Module):

    def __init__(self, dim: int, num_tokens: int, word_embedding_weight: nn.Parameter=None, bias: bool=False, dtype: dtype=None) ->None:
        super().__init__()
        self.dense = colossalai.nn.Classifier(dim, num_tokens, word_embedding_weight, bias=bias, dtype=dtype)

    @property
    def weight(self):
        return self.dense.weight

    def forward(self, x):
        x = self.dense(x)
        return x


def eval_decorator(fn):

    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner


def exists(val):
    return val is not None


def top_k(logits, thres=0.9):
    k = int((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs


class AutoregressiveWrapper(nn.Module):

    def __init__(self, net, max_seq_len=2048, pad_value=0):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.pad_value = pad_value
        self.net = net

    @torch.no_grad()
    @eval_decorator
    def generate(self, start_tokens, seq_len, eos_token=None, temperature=1.0, filter_thres=0.9, **kwargs):
        b, t, device = *start_tokens.shape, start_tokens.device
        out = start_tokens
        for _ in range(seq_len):
            logits = self.net(out, **kwargs)[:, -1, :]
            filtered_logits = top_k(logits, thres=filter_thres)
            probs = F.softmax(filtered_logits / temperature, dim=-1)
            sample = torch.multinomial(probs, 1)
            out = torch.cat((out, sample), dim=-1)
            if exists(eos_token):
                is_eos_token = out == eos_token
                if is_eos_token.any(dim=-1).all():
                    shifted_is_eos_tokens = F.pad(is_eos_token, (1, -1))
                    mask = shifted_is_eos_tokens.float().cumsum(dim=-1) >= 1
                    out = out.masked_fill(mask, self.pad_value)
                    break
        out = out[:, t:]
        return out

    def forward(self, x, **kwargs):
        return self.net(x, **kwargs)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AutoregressiveWrapper,
     lambda: ([], {'net': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SwiGLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_hpcaitech_PaLM_colossalai(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

