import sys
_module = sys.modules[__name__]
del sys
prepare_quora = _module
prepare_scitail = _module
prepare_snli = _module
prepare_wikiqa = _module
evaluate = _module
src = _module
evaluator = _module
interface = _module
model = _module
modules = _module
alignment = _module
connection = _module
embedding = _module
encoder = _module
fusion = _module
pooling = _module
prediction = _module
network = _module
trainer = _module
utils = _module
loader = _module
logger = _module
metrics = _module
params = _module
registry = _module
vocab = _module
train = _module

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


import math


import random


import torch


import torch.nn.functional as f


from typing import Collection


import torch.nn as nn


from functools import partial


class Module(nn.Module):

    def __init__(self):
        super().__init__()
        self.summary = {}

    def add_summary(self, name, val):
        if self.training:
            self.summary[name] = val.clone().detach().cpu().numpy()

    def get_summary(self, base_name=''):
        summary = {}
        if base_name:
            base_name += '/'
        if self.summary:
            summary.update({(base_name + name): val for name, val in self.summary.items()})
        for name, child in self.named_children():
            if hasattr(child, 'get_summary'):
                name = base_name + name
                summary.update(child.get_summary(name))
        return summary


class ModuleList(nn.ModuleList):

    def get_summary(self, base_name=''):
        summary = {}
        if base_name:
            base_name += '/'
        for i, module in enumerate(self):
            if hasattr(module, 'get_summary'):
                name = base_name + str(i)
                summary.update(module.get_summary(name))
        return summary


class ModuleDict(nn.ModuleDict):

    def get_summary(self, base_name=''):
        summary = {}
        if base_name:
            base_name += '/'
        for key, module in self.items():
            if hasattr(module, 'get_summary'):
                name = base_name + key
                summary.update(module.get_summary(name))
        return summary


class GeLU(nn.Module):

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))


class Linear(nn.Module):

    def __init__(self, in_features, out_features, activations=False):
        super().__init__()
        linear = nn.Linear(in_features, out_features)
        nn.init.normal_(linear.weight, std=math.sqrt((2.0 if activations else 1.0) / in_features))
        nn.init.zeros_(linear.bias)
        modules = [nn.utils.weight_norm(linear)]
        if activations:
            modules.append(GeLU())
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)


class Conv1d(Module):

    def __init__(self, in_channels, out_channels, kernel_sizes: Collection[int]):
        super().__init__()
        assert all(k % 2 == 1 for k in kernel_sizes), 'only support odd kernel sizes'
        assert out_channels % len(kernel_sizes) == 0, 'out channels must be dividable by kernels'
        out_channels = out_channels // len(kernel_sizes)
        convs = []
        for kernel_size in kernel_sizes:
            conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2)
            nn.init.normal_(conv.weight, std=math.sqrt(2.0 / (in_channels * kernel_size)))
            nn.init.zeros_(conv.bias)
            convs.append(nn.Sequential(nn.utils.weight_norm(conv), GeLU()))
        self.model = nn.ModuleList(convs)

    def forward(self, x):
        return torch.cat([encoder(x) for encoder in self.model], dim=-1)


registry = {}


class Embedding(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.fix_embeddings = args.fix_embeddings
        self.embedding = nn.Embedding(args.num_vocab, args.embedding_dim, padding_idx=0)
        self.dropout = args.dropout

    def set_(self, value):
        self.embedding.weight.requires_grad = not self.fix_embeddings
        self.embedding.load_state_dict({'weight': torch.tensor(value)})

    def forward(self, x):
        x = self.embedding(x)
        x = f.dropout(x, self.dropout, self.training)
        return x


class Encoder(nn.Module):

    def __init__(self, args, input_size):
        super().__init__()
        self.dropout = args.dropout
        self.encoders = nn.ModuleList([Conv1d(in_channels=input_size if i == 0 else args.hidden_size, out_channels=args.hidden_size, kernel_sizes=args.kernel_sizes) for i in range(args.enc_layers)])

    def forward(self, x, mask):
        x = x.transpose(1, 2)
        mask = mask.transpose(1, 2)
        for i, encoder in enumerate(self.encoders):
            x.masked_fill_(~mask, 0.0)
            if i > 0:
                x = f.dropout(x, self.dropout, self.training)
            x = encoder(x)
        x = f.dropout(x, self.dropout, self.training)
        return x.transpose(1, 2)


class Pooling(nn.Module):

    def forward(self, x, mask):
        return x.masked_fill_(~mask, -float('inf')).max(dim=1)[0]


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Embedding,
     lambda: ([], {'args': _mock_config(fix_embeddings=4, num_vocab=4, embedding_dim=4, dropout=0.5)}),
     lambda: ([torch.zeros([4], dtype=torch.int64)], {}),
     True),
    (GeLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Linear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_alibaba_edu_simple_effective_text_matching_pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

