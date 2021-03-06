import sys
_module = sys.modules[__name__]
del sys
config = _module
datasets = _module
nyudv2 = _module
draw_net = _module
models = _module
run = _module

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


from torch.utils.data import Dataset


import numpy as np


import pandas as pd


import matplotlib.pyplot as plt


import torch


from torch.autograd import Variable


import torch.nn as nn


import torch.nn.functional as F


from torch.nn.modules.utils import _pair


from torch.nn.modules.utils import _quadruple


import time


import logging


import torch.optim


from torch.utils.data import DataLoader


class MedianPool2d(nn.Module):
    """ Median pool (usable as median filter when stride=1) module.

    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """

    def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
        super(MedianPool2d, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - ih % self.stride[0], 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - iw % self.stride[1], 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = pl, pr, pt, pb
        else:
            padding = self.padding
        return padding

    def forward(self, x):
        x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x


class EnetInitialBlock(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(6, 32, (3, 3), stride=2, padding=1, bias=True)
        self.batch_norm = nn.BatchNorm2d(32, eps=0.001)
        self.actf = nn.PReLU()

    def forward(self, input):
        output = self.conv(input)
        output = self.batch_norm(output)
        return self.actf(output)


class EnetEncoderMainPath(nn.Module):

    def __init__(self, internal_scale=None, use_relu=None, asymmetric=None, dilated=None, input_channels=None, output_channels=None, downsample=None, dropout_prob=None):
        super().__init__()
        internal_channels = output_channels // internal_scale
        input_stride = downsample and 2 or 1
        self.__dict__.update(locals())
        del self.self
        self.input_conv = nn.Conv2d(input_channels, internal_channels, input_stride, stride=input_stride, padding=0, bias=False)
        self.input_batch_norm = nn.BatchNorm2d(internal_channels, eps=0.001)
        self.middle_conv = nn.Conv2d(internal_channels, internal_channels, 3, stride=1, bias=True, dilation=1 if not dilated or dilated is None else dilated, padding=1 if not dilated or dilated is None else dilated)
        self.middle_batch_norm = nn.BatchNorm2d(internal_channels, eps=0.001)
        self.output_conv = nn.Conv2d(internal_channels, output_channels, 1, stride=1, padding=0, bias=False)
        self.output_batch_norm = nn.BatchNorm2d(output_channels, eps=0.001)
        self.dropout = nn.Dropout2d(dropout_prob)
        self.input_actf = nn.PReLU()
        self.middle_actf = nn.PReLU()

    def forward(self, input):
        output = self.input_conv(input)
        output = self.input_batch_norm(output)
        output = self.input_actf(output)
        output = self.middle_conv(output)
        output = self.middle_batch_norm(output)
        output = self.middle_actf(output)
        output = self.output_conv(output)
        output = self.output_batch_norm(output)
        output = self.dropout(output)
        return output


class EnetEncoderOtherPath(nn.Module):

    def __init__(self, internal_scale=None, use_relu=None, asymmetric=None, dilated=None, input_channels=None, output_channels=None, downsample=None, **kwargs):
        super().__init__()
        self.__dict__.update(locals())
        del self.self
        if downsample:
            self.pool = nn.MaxPool2d(2, stride=2, return_indices=True)

    def forward(self, input):
        output = input
        if self.downsample:
            output, self.indices = self.pool(input)
        if self.output_channels != self.input_channels:
            new_size = [1, 1, 1, 1]
            new_size[1] = self.output_channels // self.input_channels
            output = output.repeat(*new_size)
        return output


class EnetEncoderModule(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.main = EnetEncoderMainPath(**kwargs)
        self.other = EnetEncoderOtherPath(**kwargs)
        self.actf = nn.PReLU()

    def forward(self, input):
        main = self.main(input)
        other = self.other(input)
        return self.actf(main + other)


class EnetEncoder(nn.Module):

    def __init__(self, params, nclasses):
        super().__init__()
        self.initial_block = EnetInitialBlock()
        self.layers = []
        for i, params in enumerate(params):
            layer_name = 'encoder_{:02d}'.format(i)
            layer = EnetEncoderModule(**params)
            super().__setattr__(layer_name, layer)
            self.layers.append(layer)
        self.output_conv = nn.Conv2d(128, nclasses, 1, stride=1, padding=0, bias=True)

    def forward(self, input):
        output = self.initial_block(input)
        for layer in self.layers:
            output = layer(output)
        return output


class EnetDecoderMainPath(nn.Module):

    def __init__(self, input_channels=None, output_channels=None, upsample=None, pooling_module=None):
        super().__init__()
        internal_channels = output_channels // 4
        input_stride = 2 if upsample is True else 1
        self.__dict__.update(locals())
        del self.self
        self.input_conv = nn.Conv2d(input_channels, internal_channels, 1, stride=1, padding=0, bias=False)
        self.input_batch_norm = nn.BatchNorm2d(internal_channels, eps=0.001)
        if not upsample:
            self.middle_conv = nn.Conv2d(internal_channels, internal_channels, 3, stride=1, padding=1, bias=True)
        else:
            self.middle_conv = nn.ConvTranspose2d(internal_channels, internal_channels, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.middle_batch_norm = nn.BatchNorm2d(internal_channels, eps=0.001)
        self.output_conv = nn.Conv2d(internal_channels, output_channels, 1, stride=1, padding=0, bias=False)
        self.output_batch_norm = nn.BatchNorm2d(output_channels, eps=0.001)
        self.input_actf = nn.PReLU()
        self.middle_actf = nn.PReLU()

    def forward(self, input):
        output = self.input_conv(input)
        output = self.input_batch_norm(output)
        output = self.input_actf(output)
        output = self.middle_conv(output)
        output = self.middle_batch_norm(output)
        output = self.middle_actf(output)
        output = self.output_conv(output)
        output = self.output_batch_norm(output)
        return output


class EnetDecoderOtherPath(nn.Module):

    def __init__(self, input_channels=None, output_channels=None, upsample=None, pooling_module=None):
        super().__init__()
        self.__dict__.update(locals())
        del self.self
        if output_channels != input_channels or upsample:
            self.conv = nn.Conv2d(input_channels, output_channels, 1, stride=1, padding=0, bias=False)
            self.batch_norm = nn.BatchNorm2d(output_channels, eps=0.001)
            if upsample and pooling_module:
                self.unpool = nn.MaxUnpool2d(2, stride=2, padding=0)

    def forward(self, input):
        output = input
        if self.output_channels != self.input_channels or self.upsample:
            output = self.conv(output)
            output = self.batch_norm(output)
            if self.upsample and self.pooling_module:
                output_size = list(output.size())
                output_size[2] *= 2
                output_size[3] *= 2
                output = self.unpool(output, self.pooling_module.indices, output_size=output_size)
        return output


class EnetDecoderModule(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.main = EnetDecoderMainPath(**kwargs)
        self.other = EnetDecoderOtherPath(**kwargs)
        self.actf = nn.PReLU()

    def forward(self, input):
        main = self.main(input)
        other = self.other(input)
        return self.actf(main + other)


class EnetDecoder(nn.Module):

    def __init__(self, params, nclasses, encoder):
        super().__init__()
        self.encoder = encoder
        self.pooling_modules = []
        for mod in self.encoder.modules():
            try:
                if mod.other.downsample:
                    self.pooling_modules.append(mod.other)
            except AttributeError:
                pass
        self.layers = []
        for i, params in enumerate(params):
            if params['upsample']:
                params['pooling_module'] = self.pooling_modules.pop(-1)
            layer = EnetDecoderModule(**params)
            self.layers.append(layer)
            layer_name = 'decoder{:02d}'.format(i)
            super().__setattr__(layer_name, layer)
        self.output_conv = nn.ConvTranspose2d(32, nclasses, 2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, input):
        output = input
        for layer in self.layers:
            output = layer(output)
        output = self.output_conv(output)
        return output


class EnetGnn(nn.Module):

    def __init__(self, mlp_num_layers, use_gpu):
        super().__init__()
        self.median_pool = MedianPool2d(kernel_size=8, stride=8, padding=0, same=False)
        self.g_rnn_layers = nn.ModuleList([nn.Linear(128, 128) for l in range(mlp_num_layers)])
        self.g_rnn_actfs = nn.ModuleList([nn.PReLU() for l in range(mlp_num_layers)])
        self.q_rnn_layer = nn.Linear(256, 128)
        self.q_rnn_actf = nn.PReLU()
        self.output_conv = nn.Conv2d(256, 128, 3, stride=1, padding=1, bias=True)
        self.use_gpu = use_gpu

    def get_knn_indices(self, batch_mat, k):
        r = torch.bmm(batch_mat, batch_mat.permute(0, 2, 1))
        N = r.size()[0]
        HW = r.size()[1]
        if self.use_gpu:
            batch_indices = torch.zeros((N, HW, k))
        else:
            batch_indices = torch.zeros((N, HW, k))
        for idx, val in enumerate(r):
            diag = val.diag().unsqueeze(0)
            diag = diag.expand_as(val)
            D = (diag + diag.t() - 2 * val).sqrt()
            topk, indices = torch.topk(D, k=k, largest=False)
            batch_indices[idx] = indices.data
        return batch_indices

    def forward(self, cnn_encoder_output, original_input, gnn_iterations, k, xy):
        N = cnn_encoder_output.size()[0]
        C = cnn_encoder_output.size()[1]
        H = cnn_encoder_output.size()[2]
        W = cnn_encoder_output.size()[3]
        K = k
        depth = original_input[:, (3), :, :]
        depth = depth.view(depth.size()[0], 1, depth.size()[1], depth.size()[2])
        depth_resize = self.median_pool(depth)
        x_coords = xy[:, (0), :, :]
        x_coords = x_coords.view(x_coords.size()[0], 1, x_coords.size()[1], x_coords.size()[2])
        y_coords = xy[:, (1), :, :]
        y_coords = y_coords.view(y_coords.size()[0], 1, y_coords.size()[1], y_coords.size()[2])
        x_coords = self.median_pool(x_coords)
        y_coords = self.median_pool(y_coords)
        proj_3d = torch.cat((x_coords, y_coords, depth_resize), 1)
        proj_3d = proj_3d.view(N, 3, H * W).permute(0, 2, 1).contiguous()
        knn = self.get_knn_indices(proj_3d, k=K)
        knn = knn.view(N, H * W * K).long()
        h = cnn_encoder_output
        h = h.permute(0, 2, 3, 1).contiguous()
        h = h.view(N, H * W, C)
        m = h.clone()
        for i in range(gnn_iterations):
            for n in range(N):
                neighbor_features = torch.index_select(h[n], 0, Variable(knn[n])).view(H * W, K, C)
                for idx, g_layer in enumerate(self.g_rnn_layers):
                    neighbor_features = self.g_rnn_actfs[idx](g_layer(neighbor_features))
                m[n] = torch.mean(neighbor_features, dim=1)
            concat = torch.cat((h, m), 2)
            h = self.q_rnn_actf(self.q_rnn_layer(concat))
        h = h.view(N, H, W, C).permute(0, 3, 1, 2).contiguous()
        output = self.output_conv(torch.cat((cnn_encoder_output, h), 1))
        return output


class Model(nn.Module):

    def __init__(self, nclasses, mlp_num_layers, use_gpu):
        super().__init__()
        self.encoder = EnetEncoder(ENCODER_PARAMS, nclasses)
        self.gnn = EnetGnn(mlp_num_layers, use_gpu)
        self.decoder = EnetDecoder(DECODER_PARAMS, nclasses, self.encoder)

    def forward(self, input, gnn_iterations, k, xy, use_gnn, only_encode=False):
        x = self.encoder.forward(input)
        if only_encode:
            return x
        if use_gnn:
            x = self.gnn.forward(x, input, gnn_iterations, k, xy)
        return self.decoder.forward(x)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (EnetDecoderOtherPath,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (EnetEncoderOtherPath,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (EnetInitialBlock,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 6, 64, 64])], {}),
     True),
    (MedianPool2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_yanx27_3DGNN_pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

