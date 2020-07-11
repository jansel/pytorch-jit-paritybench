import sys
_module = sys.modules[__name__]
del sys
layers = _module
main = _module
model = _module
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


import torch.nn as nn


import torch.nn.functional as F


from torch.autograd import Variable


from torch.nn.utils import weight_norm as wn


import numpy as np


import time


import torch.optim as optim


from torch.optim import lr_scheduler


from torchvision import datasets


from torchvision import transforms


from torchvision import utils


class nin(nn.Module):

    def __init__(self, dim_in, dim_out):
        super(nin, self).__init__()
        self.lin_a = wn(nn.Linear(dim_in, dim_out))
        self.dim_out = dim_out

    def forward(self, x):
        og_x = x
        """ a network in network layer (1x1 CONV) """
        x = x.permute(0, 2, 3, 1)
        shp = [int(y) for y in x.size()]
        out = self.lin_a(x.contiguous().view(shp[0] * shp[1] * shp[2], shp[3]))
        shp[-1] = self.dim_out
        out = out.view(shp)
        return out.permute(0, 3, 1, 2)


def down_shift(x, pad=None):
    xs = [int(y) for y in x.size()]
    x = x[:, :, :xs[2] - 1, :]
    pad = nn.ZeroPad2d((0, 0, 1, 0)) if pad is None else pad
    return pad(x)


class down_shifted_conv2d(nn.Module):

    def __init__(self, num_filters_in, num_filters_out, filter_size=(2, 3), stride=(1, 1), shift_output_down=False, norm='weight_norm'):
        super(down_shifted_conv2d, self).__init__()
        assert norm in [None, 'batch_norm', 'weight_norm']
        self.conv = nn.Conv2d(num_filters_in, num_filters_out, filter_size, stride)
        self.shift_output_down = shift_output_down
        self.norm = norm
        self.pad = nn.ZeroPad2d((int((filter_size[1] - 1) / 2), int((filter_size[1] - 1) / 2), filter_size[0] - 1, 0))
        if norm == 'weight_norm':
            self.conv = wn(self.conv)
        elif norm == 'batch_norm':
            self.bn = nn.BatchNorm2d(num_filters_out)
        if shift_output_down:
            self.down_shift = lambda x: down_shift(x, pad=nn.ZeroPad2d((0, 0, 1, 0)))

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.bn(x) if self.norm == 'batch_norm' else x
        return self.down_shift(x) if self.shift_output_down else x


class down_shifted_deconv2d(nn.Module):

    def __init__(self, num_filters_in, num_filters_out, filter_size=(2, 3), stride=(1, 1)):
        super(down_shifted_deconv2d, self).__init__()
        self.deconv = wn(nn.ConvTranspose2d(num_filters_in, num_filters_out, filter_size, stride, output_padding=1))
        self.filter_size = filter_size
        self.stride = stride

    def forward(self, x):
        x = self.deconv(x)
        xs = [int(y) for y in x.size()]
        return x[:, :, :xs[2] - self.filter_size[0] + 1, int((self.filter_size[1] - 1) / 2):xs[3] - int((self.filter_size[1] - 1) / 2)]


def right_shift(x, pad=None):
    xs = [int(y) for y in x.size()]
    x = x[:, :, :, :xs[3] - 1]
    pad = nn.ZeroPad2d((1, 0, 0, 0)) if pad is None else pad
    return pad(x)


class down_right_shifted_conv2d(nn.Module):

    def __init__(self, num_filters_in, num_filters_out, filter_size=(2, 2), stride=(1, 1), shift_output_right=False, norm='weight_norm'):
        super(down_right_shifted_conv2d, self).__init__()
        assert norm in [None, 'batch_norm', 'weight_norm']
        self.pad = nn.ZeroPad2d((filter_size[1] - 1, 0, filter_size[0] - 1, 0))
        self.conv = nn.Conv2d(num_filters_in, num_filters_out, filter_size, stride=stride)
        self.shift_output_right = shift_output_right
        self.norm = norm
        if norm == 'weight_norm':
            self.conv = wn(self.conv)
        elif norm == 'batch_norm':
            self.bn = nn.BatchNorm2d(num_filters_out)
        if shift_output_right:
            self.right_shift = lambda x: right_shift(x, pad=nn.ZeroPad2d((1, 0, 0, 0)))

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.bn(x) if self.norm == 'batch_norm' else x
        return self.right_shift(x) if self.shift_output_right else x


class down_right_shifted_deconv2d(nn.Module):

    def __init__(self, num_filters_in, num_filters_out, filter_size=(2, 2), stride=(1, 1), shift_output_right=False):
        super(down_right_shifted_deconv2d, self).__init__()
        self.deconv = wn(nn.ConvTranspose2d(num_filters_in, num_filters_out, filter_size, stride, output_padding=1))
        self.filter_size = filter_size
        self.stride = stride

    def forward(self, x):
        x = self.deconv(x)
        xs = [int(y) for y in x.size()]
        x = x[:, :, :xs[2] - self.filter_size[0] + 1, :xs[3] - self.filter_size[1] + 1]
        return x


def concat_elu(x):
    """ like concatenated ReLU (http://arxiv.org/abs/1603.05201), but then with ELU """
    axis = len(x.size()) - 3
    return F.elu(torch.cat([x, -x], dim=axis))


class gated_resnet(nn.Module):

    def __init__(self, num_filters, conv_op, nonlinearity=concat_elu, skip_connection=0):
        super(gated_resnet, self).__init__()
        self.skip_connection = skip_connection
        self.nonlinearity = nonlinearity
        self.conv_input = conv_op(2 * num_filters, num_filters)
        if skip_connection != 0:
            self.nin_skip = nin(2 * skip_connection * num_filters, num_filters)
        self.dropout = nn.Dropout2d(0.5)
        self.conv_out = conv_op(2 * num_filters, 2 * num_filters)

    def forward(self, og_x, a=None):
        x = self.conv_input(self.nonlinearity(og_x))
        if a is not None:
            x += self.nin_skip(self.nonlinearity(a))
        x = self.nonlinearity(x)
        x = self.dropout(x)
        x = self.conv_out(x)
        a, b = torch.chunk(x, 2, dim=1)
        c3 = a * F.sigmoid(b)
        return og_x + c3


class PixelCNNLayer_up(nn.Module):

    def __init__(self, nr_resnet, nr_filters, resnet_nonlinearity):
        super(PixelCNNLayer_up, self).__init__()
        self.nr_resnet = nr_resnet
        self.u_stream = nn.ModuleList([gated_resnet(nr_filters, down_shifted_conv2d, resnet_nonlinearity, skip_connection=0) for _ in range(nr_resnet)])
        self.ul_stream = nn.ModuleList([gated_resnet(nr_filters, down_right_shifted_conv2d, resnet_nonlinearity, skip_connection=1) for _ in range(nr_resnet)])

    def forward(self, u, ul):
        u_list, ul_list = [], []
        for i in range(self.nr_resnet):
            u = self.u_stream[i](u)
            ul = self.ul_stream[i](ul, a=u)
            u_list += [u]
            ul_list += [ul]
        return u_list, ul_list


class PixelCNNLayer_down(nn.Module):

    def __init__(self, nr_resnet, nr_filters, resnet_nonlinearity):
        super(PixelCNNLayer_down, self).__init__()
        self.nr_resnet = nr_resnet
        self.u_stream = nn.ModuleList([gated_resnet(nr_filters, down_shifted_conv2d, resnet_nonlinearity, skip_connection=1) for _ in range(nr_resnet)])
        self.ul_stream = nn.ModuleList([gated_resnet(nr_filters, down_right_shifted_conv2d, resnet_nonlinearity, skip_connection=2) for _ in range(nr_resnet)])

    def forward(self, u, ul, u_list, ul_list):
        for i in range(self.nr_resnet):
            u = self.u_stream[i](u, a=u_list.pop())
            ul = self.ul_stream[i](ul, a=torch.cat((u, ul_list.pop()), 1))
        return u, ul


class PixelCNN(nn.Module):

    def __init__(self, nr_resnet=5, nr_filters=80, nr_logistic_mix=10, resnet_nonlinearity='concat_elu', input_channels=3):
        super(PixelCNN, self).__init__()
        if resnet_nonlinearity == 'concat_elu':
            self.resnet_nonlinearity = lambda x: concat_elu(x)
        else:
            raise Exception('right now only concat elu is supported as resnet nonlinearity.')
        self.nr_filters = nr_filters
        self.input_channels = input_channels
        self.nr_logistic_mix = nr_logistic_mix
        self.right_shift_pad = nn.ZeroPad2d((1, 0, 0, 0))
        self.down_shift_pad = nn.ZeroPad2d((0, 0, 1, 0))
        down_nr_resnet = [nr_resnet] + [nr_resnet + 1] * 2
        self.down_layers = nn.ModuleList([PixelCNNLayer_down(down_nr_resnet[i], nr_filters, self.resnet_nonlinearity) for i in range(3)])
        self.up_layers = nn.ModuleList([PixelCNNLayer_up(nr_resnet, nr_filters, self.resnet_nonlinearity) for _ in range(3)])
        self.downsize_u_stream = nn.ModuleList([down_shifted_conv2d(nr_filters, nr_filters, stride=(2, 2)) for _ in range(2)])
        self.downsize_ul_stream = nn.ModuleList([down_right_shifted_conv2d(nr_filters, nr_filters, stride=(2, 2)) for _ in range(2)])
        self.upsize_u_stream = nn.ModuleList([down_shifted_deconv2d(nr_filters, nr_filters, stride=(2, 2)) for _ in range(2)])
        self.upsize_ul_stream = nn.ModuleList([down_right_shifted_deconv2d(nr_filters, nr_filters, stride=(2, 2)) for _ in range(2)])
        self.u_init = down_shifted_conv2d(input_channels + 1, nr_filters, filter_size=(2, 3), shift_output_down=True)
        self.ul_init = nn.ModuleList([down_shifted_conv2d(input_channels + 1, nr_filters, filter_size=(1, 3), shift_output_down=True), down_right_shifted_conv2d(input_channels + 1, nr_filters, filter_size=(2, 1), shift_output_right=True)])
        num_mix = 3 if self.input_channels == 1 else 10
        self.nin_out = nin(nr_filters, num_mix * nr_logistic_mix)
        self.init_padding = None

    def forward(self, x, sample=False):
        if self.init_padding is None and not sample:
            xs = [int(y) for y in x.size()]
            padding = Variable(torch.ones(xs[0], 1, xs[2], xs[3]), requires_grad=False)
            self.init_padding = padding if x.is_cuda else padding
        if sample:
            xs = [int(y) for y in x.size()]
            padding = Variable(torch.ones(xs[0], 1, xs[2], xs[3]), requires_grad=False)
            padding = padding if x.is_cuda else padding
            x = torch.cat((x, padding), 1)
        x = x if sample else torch.cat((x, self.init_padding), 1)
        u_list = [self.u_init(x)]
        ul_list = [self.ul_init[0](x) + self.ul_init[1](x)]
        for i in range(3):
            u_out, ul_out = self.up_layers[i](u_list[-1], ul_list[-1])
            u_list += u_out
            ul_list += ul_out
            if i != 2:
                u_list += [self.downsize_u_stream[i](u_list[-1])]
                ul_list += [self.downsize_ul_stream[i](ul_list[-1])]
        u = u_list.pop()
        ul = ul_list.pop()
        for i in range(3):
            u, ul = self.down_layers[i](u, ul, u_list, ul_list)
            if i != 2:
                u = self.upsize_u_stream[i](u)
                ul = self.upsize_ul_stream[i](ul)
        x_out = self.nin_out(F.elu(ul))
        assert len(u_list) == len(ul_list) == 0, pdb.set_trace()
        return x_out


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (PixelCNN,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 4, 4])], {}),
     False),
    (down_right_shifted_conv2d,
     lambda: ([], {'num_filters_in': 4, 'num_filters_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (down_shifted_conv2d,
     lambda: ([], {'num_filters_in': 4, 'num_filters_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (nin,
     lambda: ([], {'dim_in': 4, 'dim_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_pclucas14_pixel_cnn_pp(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

