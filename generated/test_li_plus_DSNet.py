import sys
_module = sys.modules[__name__]
del sys
anchor_helper = _module
dsnet = _module
losses = _module
train = _module
anchor_free_helper = _module
dsnet_af = _module
losses = _module
train = _module
evaluate = _module
bbox_helper = _module
data_helper = _module
init_helper = _module
video_helper = _module
vsumm_helper = _module
infer = _module
cpd_auto = _module
cpd_nonlin = _module
demo = _module
make_dataset = _module
make_shots = _module
make_split = _module
model_zoo = _module
models = _module
tests = _module
anchor_based = _module
test_ab_losses = _module
test_anchor_helper = _module
anchor_free = _module
test_af_losses = _module
test_anchor_free_helper = _module
helpers = _module
test_bbox_helper = _module
test_data_helper = _module
test_vsumm_helper = _module
modules = _module
test_models = _module
test_train = _module

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


from torch import nn


from torch.nn import functional as F


import logging


import numpy as np


import random


from numpy import linalg


from torchvision import transforms


from torchvision import models


import math


class ScaledDotProductAttention(nn.Module):

    def __init__(self, d_k):
        super().__init__()
        self.dropout = nn.Dropout(0.5)
        self.sqrt_d_k = math.sqrt(d_k)

    def forward(self, Q, K, V):
        attn = torch.bmm(Q, K.transpose(2, 1))
        attn = attn / self.sqrt_d_k
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        y = torch.bmm(attn, V)
        return y, attn


class MultiHeadAttention(nn.Module):

    def __init__(self, num_head=8, num_feature=1024):
        super().__init__()
        self.num_head = num_head
        self.Q = nn.Linear(num_feature, num_feature, bias=False)
        self.K = nn.Linear(num_feature, num_feature, bias=False)
        self.V = nn.Linear(num_feature, num_feature, bias=False)
        self.d_k = num_feature // num_head
        self.attention = ScaledDotProductAttention(self.d_k)
        self.fc = nn.Sequential(nn.Linear(num_feature, num_feature, bias=False), nn.Dropout(0.5))

    def forward(self, x):
        _, seq_len, num_feature = x.shape
        K = self.K(x)
        Q = self.Q(x)
        V = self.V(x)
        K = K.view(1, seq_len, self.num_head, self.d_k).permute(2, 0, 1, 3).contiguous().view(self.num_head, seq_len, self.d_k)
        Q = Q.view(1, seq_len, self.num_head, self.d_k).permute(2, 0, 1, 3).contiguous().view(self.num_head, seq_len, self.d_k)
        V = V.view(1, seq_len, self.num_head, self.d_k).permute(2, 0, 1, 3).contiguous().view(self.num_head, seq_len, self.d_k)
        y, attn = self.attention(Q, K, V)
        y = y.view(1, self.num_head, seq_len, self.d_k).permute(0, 2, 1, 3).contiguous().view(1, seq_len, num_feature)
        y = self.fc(y)
        return y, attn


class AttentionExtractor(MultiHeadAttention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, *inputs):
        out, _ = super().forward(*inputs)
        return out


class GCNExtractor(nn.Module):

    def __init__(self, num_feature):
        super().__init__()
        self.gcn = GCNConv(num_feature, num_feature)

    def forward(self, x):
        x = x.squeeze(0)
        edge_indices, edge_weights = self.create_graph(x, keep_ratio=0.3)
        out = self.gcn(x, edge_indices, edge_weights)
        out = out.unsqueeze(0)
        return out

    @staticmethod
    def create_graph(x, keep_ratio=0.3):
        seq_len, _ = x.shape
        keep_top_k = int(keep_ratio * seq_len * seq_len)
        edge_weights = torch.matmul(x, x.t())
        edge_weights = edge_weights - torch.eye(seq_len, seq_len)
        edge_weights = edge_weights.view(-1)
        edge_weights, edge_indices = torch.topk(edge_weights, keep_top_k, sorted=False)
        edge_indices = edge_indices.unsqueeze(0)
        edge_indices = torch.cat([edge_indices / seq_len, edge_indices % seq_len])
        return edge_indices, edge_weights


class LSTMExtractor(nn.LSTM):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, *inputs):
        out, _ = super().forward(*inputs)
        return out


def build_base_model(base_type: str, num_feature: int, num_head: int) ->nn.Module:
    if base_type == 'linear':
        base_model = nn.Linear(num_feature, num_feature)
    elif base_type == 'lstm':
        base_model = LSTMExtractor(num_feature, num_feature)
    elif base_type == 'bilstm':
        base_model = LSTMExtractor(num_feature, num_feature // 2, bidirectional=True)
    elif base_type == 'gcn':
        base_model = GCNExtractor(num_feature)
    elif base_type == 'attention':
        base_model = AttentionExtractor(num_head, num_feature)
    else:
        raise ValueError(f'Invalid base model {base_type}')
    return base_model


class DSNet(nn.Module):

    def __init__(self, base_model, num_feature, num_hidden, anchor_scales, num_head):
        super().__init__()
        self.anchor_scales = anchor_scales
        self.num_scales = len(anchor_scales)
        self.base_model = build_base_model(base_model, num_feature, num_head)
        self.roi_poolings = [nn.AvgPool1d(scale, stride=1, padding=scale // 2) for scale in anchor_scales]
        self.layer_norm = nn.LayerNorm(num_feature)
        self.fc1 = nn.Sequential(nn.Linear(num_feature, num_hidden), nn.Tanh(), nn.Dropout(0.5), nn.LayerNorm(num_hidden))
        self.fc_cls = nn.Linear(num_hidden, 1)
        self.fc_loc = nn.Linear(num_hidden, 2)

    def forward(self, x):
        _, seq_len, _ = x.shape
        out = self.base_model(x)
        out = out + x
        out = self.layer_norm(out)
        out = out.transpose(2, 1)
        pool_results = [roi_pooling(out) for roi_pooling in self.roi_poolings]
        out = torch.cat(pool_results, dim=0).permute(2, 0, 1)[:-1]
        out = self.fc1(out)
        pred_cls = self.fc_cls(out).sigmoid().view(seq_len, self.num_scales)
        pred_loc = self.fc_loc(out).view(seq_len, self.num_scales, 2)
        return pred_cls, pred_loc

    def predict(self, seq):
        seq_len = seq.shape[1]
        pred_cls, pred_loc = self(seq)
        pred_cls = pred_cls.cpu().numpy().reshape(-1)
        pred_loc = pred_loc.cpu().numpy().reshape((-1, 2))
        anchors = anchor_helper.get_anchors(seq_len, self.anchor_scales)
        anchors = anchors.reshape((-1, 2))
        pred_bboxes = anchor_helper.offset2bbox(pred_loc, anchors)
        pred_bboxes = bbox_helper.cw2lr(pred_bboxes)
        return pred_cls, pred_bboxes


class DSNetAF(nn.Module):

    def __init__(self, base_model, num_feature, num_hidden, num_head):
        super().__init__()
        self.base_model = build_base_model(base_model, num_feature, num_head)
        self.layer_norm = nn.LayerNorm(num_feature)
        self.fc1 = nn.Sequential(nn.Linear(num_feature, num_hidden), nn.ReLU(inplace=True), nn.Dropout(0.5), nn.LayerNorm(num_hidden))
        self.fc_cls = nn.Linear(num_hidden, 1)
        self.fc_loc = nn.Linear(num_hidden, 2)
        self.fc_ctr = nn.Linear(num_hidden, 1)

    def forward(self, x):
        _, seq_len, _ = x.shape
        out = self.base_model(x)
        out = out + x
        out = self.layer_norm(out)
        out = self.fc1(out)
        pred_cls = self.fc_cls(out).sigmoid().view(seq_len)
        pred_loc = self.fc_loc(out).exp().view(seq_len, 2)
        pred_ctr = self.fc_ctr(out).sigmoid().view(seq_len)
        return pred_cls, pred_loc, pred_ctr

    def predict(self, seq):
        pred_cls, pred_loc, pred_ctr = self(seq)
        pred_cls *= pred_ctr
        pred_cls /= pred_cls.max() + 1e-08
        pred_cls = pred_cls.cpu().numpy()
        pred_loc = pred_loc.cpu().numpy()
        pred_bboxes = anchor_free_helper.offset2bbox(pred_loc)
        return pred_cls, pred_bboxes


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ScaledDotProductAttention,
     lambda: ([], {'d_k': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     True),
]

class Test_li_plus_DSNet(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

