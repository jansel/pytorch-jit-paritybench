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
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


class Alignment(Module):

    def __init__(self, args, __):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(1 / math.sqrt(args.hidden_size)))

    def _attention(self, a, b):
        return torch.matmul(a, b.transpose(1, 2)) * self.temperature

    def forward(self, a, b, mask_a, mask_b):
        attn = self._attention(a, b)
        mask = torch.matmul(mask_a.float(), mask_b.transpose(1, 2).float()).byte()
        attn.masked_fill_(~mask, -10000000.0)
        attn_a = f.softmax(attn, dim=1)
        attn_b = f.softmax(attn, dim=2)
        feature_b = torch.matmul(attn_a.transpose(1, 2), a)
        feature_a = torch.matmul(attn_b, b)
        self.add_summary('temperature', self.temperature)
        self.add_summary('attention_a', attn_a)
        self.add_summary('attention_b', attn_b)
        return feature_a, feature_b


class MappedAlignment(Alignment):

    def __init__(self, args, input_size):
        super().__init__(args, input_size)
        self.projection = nn.Sequential(nn.Dropout(args.dropout), Linear(input_size, args.hidden_size, activations=True))

    def _attention(self, a, b):
        a = self.projection(a)
        b = self.projection(b)
        return super()._attention(a, b)


class NullConnection(nn.Module):

    def forward(self, x, _, __):
        return x


class Residual(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.linear = Linear(args.embedding_dim, args.hidden_size)

    def forward(self, x, res, i):
        if i == 1:
            res = self.linear(res)
        return (x + res) * math.sqrt(0.5)


class AugmentedResidual(nn.Module):

    def forward(self, x, res, i):
        if i == 1:
            return torch.cat([x, res], dim=-1)
        hidden_size = x.size(-1)
        x = (res[:, :, :hidden_size] + x) * math.sqrt(0.5)
        return torch.cat([x, res[:, :, hidden_size:]], dim=-1)


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


class Fusion(nn.Module):

    def __init__(self, args, input_size):
        super().__init__()
        self.fusion = Linear(input_size * 2, args.hidden_size, activations=True)

    def forward(self, x, align):
        return self.fusion(torch.cat([x, align], dim=-1))


class FullFusion(nn.Module):

    def __init__(self, args, input_size):
        super().__init__()
        self.dropout = args.dropout
        self.fusion1 = Linear(input_size * 2, args.hidden_size, activations=True)
        self.fusion2 = Linear(input_size * 2, args.hidden_size, activations=True)
        self.fusion3 = Linear(input_size * 2, args.hidden_size, activations=True)
        self.fusion = Linear(args.hidden_size * 3, args.hidden_size, activations=True)

    def forward(self, x, align):
        x1 = self.fusion1(torch.cat([x, align], dim=-1))
        x2 = self.fusion2(torch.cat([x, x - align], dim=-1))
        x3 = self.fusion3(torch.cat([x, x * align], dim=-1))
        x = torch.cat([x1, x2, x3], dim=-1)
        x = f.dropout(x, self.dropout, self.training)
        return self.fusion(x)


class Pooling(nn.Module):

    def forward(self, x, mask):
        return x.masked_fill_(~mask, -float('inf')).max(dim=1)[0]


class Prediction(nn.Module):

    def __init__(self, args, inp_features=2):
        super().__init__()
        self.dense = nn.Sequential(nn.Dropout(args.dropout), Linear(args.hidden_size * inp_features, args.hidden_size, activations=True), nn.Dropout(args.dropout), Linear(args.hidden_size, args.num_classes))

    def forward(self, a, b):
        return self.dense(torch.cat([a, b], dim=-1))


class AdvancedPrediction(Prediction):

    def __init__(self, args):
        super().__init__(args, inp_features=4)

    def forward(self, a, b):
        return self.dense(torch.cat([a, b, a - b, a * b], dim=-1))


class SymmetricPrediction(AdvancedPrediction):

    def forward(self, a, b):
        return self.dense(torch.cat([a, b, (a - b).abs(), a * b], dim=-1))


class Network(Module):

    def __init__(self, args):
        super().__init__()
        self.dropout = args.dropout
        self.embedding = Embedding(args)
        self.blocks = ModuleList([ModuleDict({'encoder': Encoder(args, args.embedding_dim if i == 0 else args.embedding_dim + args.hidden_size), 'alignment': alignment[args.alignment](args, args.embedding_dim + args.hidden_size if i == 0 else args.embedding_dim + args.hidden_size * 2), 'fusion': fusion[args.fusion](args, args.embedding_dim + args.hidden_size if i == 0 else args.embedding_dim + args.hidden_size * 2)}) for i in range(args.blocks)])
        self.connection = connection[args.connection]()
        self.pooling = Pooling()
        self.prediction = prediction[args.prediction](args)

    def forward(self, inputs):
        a = inputs['text1']
        b = inputs['text2']
        mask_a = inputs['mask1']
        mask_b = inputs['mask2']
        a = self.embedding(a)
        b = self.embedding(b)
        res_a, res_b = a, b
        for i, block in enumerate(self.blocks):
            if i > 0:
                a = self.connection(a, res_a, i)
                b = self.connection(b, res_b, i)
                res_a, res_b = a, b
            a_enc = block['encoder'](a, mask_a)
            b_enc = block['encoder'](b, mask_b)
            a = torch.cat([a, a_enc], dim=-1)
            b = torch.cat([b, b_enc], dim=-1)
            align_a, align_b = block['alignment'](a, b, mask_a, mask_b)
            a = block['fusion'](a, align_a)
            b = block['fusion'](b, align_b)
        a = self.pooling(a, mask_a)
        b = self.pooling(b, mask_b)
        return self.prediction(a, b)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AdvancedPrediction,
     lambda: ([], {'args': _mock_config(dropout=0.5, hidden_size=4, num_classes=4)}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (AugmentedResidual,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 0, 4]), torch.rand([4, 4, 0, 4]), 0], {}),
     True),
    (Embedding,
     lambda: ([], {'args': _mock_config(fix_embeddings=4, num_vocab=4, embedding_dim=4, dropout=0.5)}),
     lambda: ([torch.zeros([4], dtype=torch.int64)], {}),
     True),
    (FullFusion,
     lambda: ([], {'args': _mock_config(dropout=0.5, hidden_size=4), 'input_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Fusion,
     lambda: ([], {'args': _mock_config(hidden_size=4), 'input_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (GeLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Linear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (NullConnection,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Prediction,
     lambda: ([], {'args': _mock_config(dropout=0.5, hidden_size=4, num_classes=4)}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Residual,
     lambda: ([], {'args': _mock_config(embedding_dim=4, hidden_size=4)}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), 0], {}),
     True),
    (SymmetricPrediction,
     lambda: ([], {'args': _mock_config(dropout=0.5, hidden_size=4, num_classes=4)}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_alibaba_edu_simple_effective_text_matching_pytorch(_paritybench_base):
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

