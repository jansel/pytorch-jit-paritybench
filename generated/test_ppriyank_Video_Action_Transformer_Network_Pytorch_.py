import sys
_module = sys.modules[__name__]
del sys
tools = _module
transformer_keras_tensorflow = _module
transformer_v2 = _module
transformer_v3 = _module

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


import torch.nn as nn


import torchvision


import torch.nn.functional as F


from torch import nn


import numpy as np


import math


from torch.autograd import Variable


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('LayerNorm') != -1:
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)


class BNClassifier(nn.Module):

    def __init__(self, in_dim, class_num, initialization=True):
        super(BNClassifier, self).__init__()
        self.in_dim = in_dim
        self.class_num = class_num
        self.bn = nn.BatchNorm1d(self.in_dim)
        self.bn.bias.requires_grad_(False)
        self.classifier = nn.Linear(self.in_dim, self.class_num, bias=False)
        if initialization:
            self.bn.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        feature = self.bn(x)
        cls_score = self.classifier(feature)
        return feature, cls_score


class Bottle(nn.Module):

    def forward(self, input):
        if len(input.size()) <= 2:
            return super(Bottle, self).forward(input)
        size = input.size()[:2]
        out = super(Bottle, self).forward(input.view(size[0] * size[1], -1))
        return out.contiguous().view(size[0], size[1], -1)


class BottleSoftmax(Bottle, nn.Softmax):
    pass


class LayerNorm(nn.Module):

    def __init__(self, features, dim1, dim2, eps=1e-06):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(features, dim1, dim2))
        self.bias = nn.Parameter(torch.zeros(features, dim1, dim2))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(1, keepdim=True)
        mean = mean.expand(x.size(0), x.size(1), x.size(2), x.size(3))
        std = x.std(1, keepdim=True)
        std = std.expand(x.size(0), x.size(1), x.size(2), x.size(3))
        z = (x - mean) / (std + self.eps)
        bias = self.bias.view(1, self.bias.size(0), self.bias.size(1), self.bias.size(2))
        bias = bias.expand(x.size(0), x.size(1), x.size(2), x.size(3))
        weight = self.weight.view(1, self.weight.size(0), self.weight.size(1), self.weight.size(2))
        weight = weight.expand(x.size(0), x.size(1), x.size(2), x.size(3))
        return weight * z + bias


class FeedForward(nn.Module):

    def __init__(self, d_model, d_ff=2048, dropout=0.3):
        super(FeedForward, self).__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
        nn.init.normal(self.linear_1.weight, std=0.001)
        nn.init.normal(self.linear_2.weight, std=0.001)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class Norm(nn.Module):

    def __init__(self, d_model, eps=1e-06, trainable=True):
        super(Norm, self).__init__()
        self.size = d_model
        if trainable:
            self.alpha = nn.Parameter(torch.ones(self.size))
            self.bias = nn.Parameter(torch.zeros(self.size))
        else:
            self.alpha = nn.Parameter(torch.ones(self.size), requires_grad=False)
            self.bias = nn.Parameter(torch.zeros(self.size), requires_grad=False)
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class PositionalEncoder(nn.Module):

    def __init__(self, d_model, max_seq_len=80):
        super(PositionalEncoder, self).__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / 10000 ** (2 * i / d_model))
                pe[pos, i + 1] = math.cos(pos / 10000 ** (2 * (i + 1) / d_model))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        seq_len = x.size(1)
        batch_size = x.size(0)
        num_feature = x.size(2)
        spatial_h = x.size(3)
        spatial_w = x.size(4)
        z = Variable(self.pe[:, :seq_len], requires_grad=False)
        z = z.unsqueeze(-1).unsqueeze(-1)
        z = z.expand(batch_size, seq_len, num_feature, spatial_h, spatial_w)
        x = x + z
        return x


def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.sum(q * k, -1) / math.sqrt(d_k)
    scores = F.softmax(scores, dim=-1)
    scores = scores.unsqueeze(-1).expand(scores.size(0), scores.size(1), v.size(-1))
    output = scores * v
    output = torch.sum(output, 1)
    if dropout:
        output = dropout(output)
    return output


class TX(nn.Module):

    def __init__(self, d_model=64, dropout=0.3):
        super(TX, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.ff = FeedForward(d_model, d_ff=d_model / 2, dropout=dropout)

    def forward(self, q, k, v, mask=None):
        b = q.size(0)
        t = k.size(1)
        dim = q.size(1)
        q_temp = q.unsqueeze(1)
        q_temp = q_temp.expand(b, t, dim)
        A = attention(q_temp, k, v, self.d_model, mask, self.dropout)
        q_ = self.norm_1(A + q)
        new_query = self.norm_2(q_ + self.dropout_2(self.ff(q_)))
        return new_query


class Block_head(nn.Module):

    def __init__(self, d_model=1024, dropout=0.3, head=16):
        super(Block_head, self).__init__()
        self.dropout = dropout
        self.head = head
        self.d_model = d_model
        self.d_k = d_model // head
        self.head_layers = []
        for i in range(self.head):
            self.head_layers.append(TX())
        self.list_layers = nn.ModuleList(self.head_layers)
        self.q_linear = nn.Linear(d_model, d_model)
        nn.init.normal(self.q_linear.weight, std=0.001)
        nn.init.constant(self.q_linear.bias, 0)

    def forward(self, q, k, v, mask=None):
        bs = k.shape[0]
        k = k.view(bs, -1, head, d_k)
        q = F.relu(q_linear(q).view(bs, head, d_k))
        v = v.view(bs, -1, head, d_k)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        outputs = []
        for i in range(self.head):
            outputs.append(self.list_layers[i](q[:, i], k[:, i], v[:, i]))
        q = torch.cat(outputs, 1)
        return q


class Tail(nn.Module):

    def __init__(self, num_classes, num_frames, head=16):
        super(Tail, self).__init__()
        self.spatial_h = 7
        self.spatial_w = 4
        self.head = head
        self.num_features = 2048
        self.num_frames = num_frames
        self.d_model = self.num_features / 2
        self.d_k = self.d_model // self.head
        self.bn1 = nn.BatchNorm2d(self.num_features)
        self.bn2 = Norm(self.d_model, trainable=False)
        self.pos_embd = PositionalEncoder(self.num_features, self.num_frames)
        self.Qpr = nn.Conv2d(self.num_features, self.d_model, kernel_size=(7, 4), stride=1, padding=0, bias=False)
        self.L1 = Block_head()
        self.L3 = Block_head()
        self.L2 = Block_head()
        self.classifier = BNClassifier(self.d_model, num_classes)
        nn.init.kaiming_normal(self.Qpr.weight, mode='fan_out')
        nn.init.constant(self.bn1.weight, 1)
        nn.init.constant(self.bn1.bias, 0)

    def forward(self, x, b, t):
        x = self.bn1(x)
        x = x.view(b, t, self.num_features, self.spatial_h, self.spatial_w)
        x = self.pos_embd(x)
        x = x.view(-1, self.num_features, self.spatial_h, self.spatial_w)
        x = F.relu(self.Qpr(x))
        x = x.view(-1, t, self.d_model)
        x = self.bn2(x)
        q = x[:, t / 2, :]
        v = x
        k = x
        q = self.L1(q, k, v)
        q = self.L2(q, k, v)
        q = self.L3(q, k, v)
        f = F.normalize(q, p=2, dim=1)
        if not self.training:
            f, y = self.classifier(f)
            return f
        f_, y = self.classifier(f)
        return y, f_


class Semi_Transformer(nn.Module):

    def __init__(self, num_classes, seq_len):
        super(Semi_Transformer, self).__init__()
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.tail = Tail(num_classes, seq_len)

    def forward(self, x):
        b = x.size(0)
        t = x.size(1)
        x = x.view(b * t, x.size(2), x.size(3), x.size(4))
        x = self.base(x)
        return self.tail(x, b, t)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BNClassifier,
     lambda: ([], {'in_dim': 4, 'class_num': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (BottleSoftmax,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (FeedForward,
     lambda: ([], {'d_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LayerNorm,
     lambda: ([], {'features': 4, 'dim1': 4, 'dim2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Norm,
     lambda: ([], {'d_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PositionalEncoder,
     lambda: ([], {'d_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     False),
]

class Test_ppriyank_Video_Action_Transformer_Network_Pytorch_(_paritybench_base):
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

