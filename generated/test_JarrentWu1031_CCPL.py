import sys
_module = sys.modules[__name__]
del sys
Loader = _module
function = _module
net = _module
predict = _module
sampler = _module
test = _module
test_video_frame = _module
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


import torch.utils.data as data


import torchvision.transforms as transforms


import torch


import torch.nn as nn


import torch.nn.functional as F


import random


import numpy as np


from torchvision import transforms


from torchvision.utils import save_image


from torch.utils import data


import warnings


import torch.backends.cudnn as cudnn


import itertools


import time


import scipy.misc


import scipy.sparse


import scipy.sparse.linalg


from numpy.lib.stride_tricks import as_strided


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1.0 / self.power)
        out = x.div(norm + 1e-07)
        return out


class CCPL(nn.Module):

    def __init__(self, mlp):
        super(CCPL, self).__init__()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.mlp = mlp

    def NeighborSample(self, feat, layer, num_s, sample_ids=[]):
        b, c, h, w = feat.size()
        feat_r = feat.permute(0, 2, 3, 1).flatten(1, 2)
        if sample_ids == []:
            dic = {(0): -(w + 1), (1): -w, (2): -(w - 1), (3): -1, (4): 1, (5): w - 1, (6): w, (7): w + 1}
            s_ids = torch.randperm((h - 2) * (w - 2), device=feat.device)
            s_ids = s_ids[:int(min(num_s, s_ids.shape[0]))]
            ch_ids = s_ids // (w - 2) + 1
            cw_ids = s_ids % (w - 2) + 1
            c_ids = (ch_ids * w + cw_ids).repeat(8)
            delta = [dic[i // num_s] for i in range(8 * num_s)]
            delta = torch.tensor(delta)
            n_ids = c_ids + delta
            sample_ids += [c_ids]
            sample_ids += [n_ids]
        else:
            c_ids = sample_ids[0]
            n_ids = sample_ids[1]
        feat_c, feat_n = feat_r[:, c_ids, :], feat_r[:, n_ids, :]
        feat_d = feat_c - feat_n
        for i in range(3):
            feat_d = self.mlp[3 * layer + i](feat_d)
        feat_d = Normalize(2)(feat_d.permute(0, 2, 1))
        return feat_d, sample_ids

    def PatchNCELoss(self, f_q, f_k, tau=0.07):
        B, C, S = f_q.shape
        f_k = f_k.detach()
        l_pos = (f_k * f_q).sum(dim=1)[:, :, None]
        l_neg = torch.bmm(f_q.transpose(1, 2), f_k)
        identity_matrix = torch.eye(S, dtype=torch.bool)[None, :, :]
        l_neg.masked_fill_(identity_matrix, -float('inf'))
        logits = torch.cat((l_pos, l_neg), dim=2) / tau
        predictions = logits.flatten(0, 1)
        targets = torch.zeros(B * S, dtype=torch.long)
        return self.cross_entropy_loss(predictions, targets)

    def forward(self, feats_q, feats_k, num_s, start_layer, end_layer, tau=0.07):
        loss_ccp = 0.0
        for i in range(start_layer, end_layer):
            f_q, sample_ids = self.NeighborSample(feats_q[i], i, num_s, [])
            f_k, _ = self.NeighborSample(feats_k[i], i, num_s, sample_ids)
            loss_ccp += self.PatchNCELoss(f_q, f_k, tau)
        return loss_ccp


def calc_mean_std(feat, eps=1e-05):
    size = feat.size()
    assert len(size) == 4
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


mlp = nn.ModuleList([nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 16), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 32), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 64), nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 128)])


class Net(nn.Module):

    def __init__(self, encoder, decoder, training_mode='art'):
        super(Net, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])
        self.enc_2 = nn.Sequential(*enc_layers[4:11])
        self.enc_3 = nn.Sequential(*enc_layers[11:18])
        self.enc_4 = nn.Sequential(*enc_layers[18:31])
        self.decoder = decoder
        self.SCT = SCT(training_mode)
        self.mlp = mlp if training_mode == 'art' else mlp[:9]
        self.CCPL = CCPL(self.mlp)
        self.mse_loss = nn.MSELoss()
        self.end_layer = 4 if training_mode == 'art' else 3
        self.mode = training_mode
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(self.end_layer):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    def encode(self, input):
        for i in range(self.end_layer):
            input = getattr(self, 'enc_{:d}'.format(i + 1))(input)
        return input

    def feature_compress(self, feat):
        feat = feat.flatten(2, 3)
        feat = self.mlp(feat)
        feat = feat.flatten(1, 2)
        feat = Normalize(2)(feat)
        return feat

    def calc_content_loss(self, input, target):
        assert input.size() == target.size()
        assert target.requires_grad is False
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert input.size() == target.size()
        assert target.requires_grad is False
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + self.mse_loss(input_std, target_std)

    def forward(self, content, style, tau, num_s, num_layer):
        style_feats = self.encode_with_intermediate(style)
        content_feats = self.encode_with_intermediate(content)
        gF = self.SCT(content_feats[-1], style_feats[-1])
        gimage = self.decoder(gF)
        g_t_feats = self.encode_with_intermediate(gimage)
        end_layer = self.end_layer
        loss_c = self.calc_content_loss(g_t_feats[-1], content_feats[-1])
        loss_s = self.calc_style_loss(g_t_feats[0], style_feats[0])
        for i in range(1, end_layer):
            loss_s += self.calc_style_loss(g_t_feats[i], style_feats[i])
        start_layer = end_layer - num_layer
        loss_ccp = self.CCPL(g_t_feats, content_feats, num_s, start_layer, end_layer)
        return loss_c, loss_s, loss_ccp


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (CCPL,
     lambda: ([], {'mlp': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), 0, 0], {}),
     False),
    (Normalize,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_JarrentWu1031_CCPL(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

