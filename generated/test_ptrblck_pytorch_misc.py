import sys
_module = sys.modules[__name__]
del sys
LocallyConnected2d = _module
accumulate_gradients = _module
adaptive_batchnorm = _module
adaptive_pooling_torchvision = _module
batch_norm_manual = _module
change_crop_in_dataset = _module
channel_to_patches = _module
conv_rnn = _module
csv_chunk_read = _module
densenet_forwardhook = _module
edge_weighting_segmentation = _module
image_rotation_with_matrix = _module
mnist_autoencoder = _module
mnist_permuted = _module
model_sharding_data_parallel = _module
momentum_update_nograd = _module
pytorch_redis = _module
shared_array = _module
shared_dict = _module
unet_demo = _module
weighted_sampling = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import re, math, string, numpy, torch, torchtext, torchaudio, logging, itertools, numbers, inspect, functools, copy, scipy, types, time, torchvision, enum, random, typing, warnings, abc, collections, uuid
import numpy as np
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch


import torch.nn as nn


from torch.nn.modules.utils import _pair


import torch.optim as optim


import torch.nn.functional as F


from torchvision import datasets


from torchvision import transforms


import torchvision.models as models


from torch.utils.data import Dataset


from torchvision import models


from torch.utils.data import DataLoader


import torchvision.transforms as transforms


import torchvision.datasets as datasets


import numpy as np


from torchvision.utils import make_grid


class LocallyConnected2d(nn.Module):

    def __init__(self, in_channels, out_channels, output_size, kernel_size,
        stride, bias=False):
        super(LocallyConnected2d, self).__init__()
        output_size = _pair(output_size)
        self.weight = nn.Parameter(torch.randn(1, out_channels, in_channels,
            output_size[0], output_size[1], kernel_size ** 2))
        if bias:
            self.bias = nn.Parameter(torch.randn(1, out_channels,
                output_size[0], output_size[1]))
        else:
            self.register_parameter('bias', None)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)

    def forward(self, x):
        _, c, h, w = x.size()
        kh, kw = self.kernel_size
        dh, dw = self.stride
        x = x.unfold(2, kh, dh).unfold(3, kw, dw)
        x = x.contiguous().view(*x.size()[:-2], -1)
        out = (x.unsqueeze(1) * self.weight).sum([2, -1])
        if self.bias is not None:
            out += self.bias
        return out


class AdaptiveBatchNorm2d(nn.Module):
    """
    Adaptive BN implementation using two additional parameters:
    out = a * x + b * bn(x)
    """

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True):
        super(AdaptiveBatchNorm2d, self).__init__()
        self.bn = nn.BatchNorm2d(num_features, eps, momentum, affine)
        self.a = nn.Parameter(torch.FloatTensor(1, 1, 1, 1))
        self.b = nn.Parameter(torch.FloatTensor(1, 1, 1, 1))

    def forward(self, x):
        return self.a * x + self.b * self.bn(x)


class MyNet(nn.Module):

    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        self.conv1_bn = AdaptiveBatchNorm2d(10)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)
        self.conv2_bn = AdaptiveBatchNorm2d(20)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1_bn(self.conv1(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2_bn(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class MyBatchNorm2d(nn.BatchNorm2d):

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True,
        track_running_stats=True):
        super(MyBatchNorm2d, self).__init__(num_features, eps, momentum,
            affine, track_running_stats)

    def forward(self, input):
        self._check_input_dim(input)
        exponential_average_factor = 0.0
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.
                        num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum
        if self.training:
            mean = input.mean([0, 2, 3])
            var = input.var([0, 2, 3], unbiased=False)
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean + (1 -
                    exponential_average_factor) * self.running_mean
                self.running_var = exponential_average_factor * var * n / (n -
                    1) + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var
        input = (input - mean[(None), :, (None), (None)]) / torch.sqrt(var[
            (None), :, (None), (None)] + self.eps)
        if self.affine:
            input = input * self.weight[(None), :, (None), (None)] + self.bias[
                (None), :, (None), (None)]
        return input


class MyModel(nn.Module):

    def __init__(self, window=16):
        super(MyModel, self).__init__()
        self.conv_model = nn.Sequential(nn.Conv3d(in_channels=3,
            out_channels=6, kernel_size=3, stride=1, padding=1), nn.
            MaxPool3d((1, 2, 2)), nn.ReLU())
        self.rnn = nn.RNN(input_size=6 * 16 * 12 * 12, hidden_size=1,
            num_layers=1, batch_first=True)
        self.hidden = torch.zeros(1, 1, 1)
        self.window = window

    def forward(self, x):
        self.hidden = torch.zeros(1, 1, 1)
        activations = []
        for idx in range(0, x.size(2), self.window):
            x_ = x[:, :, idx:idx + self.window]
            x_ = self.conv_model(x_)
            x_ = x_.view(x_.size(0), 1, -1)
            activations.append(x_)
        x = torch.cat(activations, 1)
        out, hidden = self.rnn(x, self.hidden)
        return out, hidden


class MyModel(nn.Module):

    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(3, 6, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(2)
        self.conv_trans1 = nn.ConvTranspose2d(6, 3, 4, 2, 1)
        self.conv_trans2 = nn.ConvTranspose2d(3, 1, 4, 2, 1)

    def forward(self, x):
        x = F.relu(self.pool1(self.conv1(x)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = F.relu(self.conv_trans1(x))
        x = self.conv_trans2(x)
        return x


class MyModel(nn.Module):

    def __init__(self):
        super(MyModel, self).__init__()
        self.act = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 4, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(4, 8, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(7 * 7 * 8, 10)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.pool1(x)
        x = self.act(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = F.log_softmax(self.fc1(x), dim=1)
        return x


class SubModule(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(SubModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        None
        x = self.conv1(x)
        return x


class MyModel(nn.Module):

    def __init__(self, split_gpus, parallel):
        super(MyModel, self).__init__()
        self.module1 = SubModule(3, 6)
        self.module2 = SubModule(6, 1)
        self.split_gpus = split_gpus
        self.parallel = parallel
        if self.split_gpus and self.parallel:
            self.module1 = nn.DataParallel(self.module1, device_ids=[0, 1])
            self.module2 = nn.DataParallel(self.module2, device_ids=[2, 3])

    def forward(self, x):
        None
        x = self.module1(x)
        None
        x = self.module2(x)
        None
        return x


class MyModel(nn.Module):

    def __init__(self):
        super(MyModel, self).__init__()
        self.enc = nn.Linear(64, 10)
        self.dec1 = nn.Linear(10, 64)
        self.dec2 = nn.Linear(10, 64)

    def forward(self, x, decoder_idx):
        x = F.relu(self.enc(x))
        if decoder_idx == 1:
            None
            x = self.dec1(x)
        elif decoder_idx == 2:
            None
            x = self.dec2(x)
        else:
            None
        return x


class BaseConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding, stride
        ):
        super(BaseConv, self).__init__()
        self.act = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size,
            padding, stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size,
            padding, stride)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        return x


class DownConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding, stride
        ):
        super(DownConv, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv_block = BaseConv(in_channels, out_channels, kernel_size,
            padding, stride)

    def forward(self, x):
        x = self.pool1(x)
        x = self.conv_block(x)
        return x


class UpConv(nn.Module):

    def __init__(self, in_channels, in_channels_skip, out_channels,
        kernel_size, padding, stride):
        super(UpConv, self).__init__()
        self.conv_trans1 = nn.ConvTranspose2d(in_channels, in_channels,
            kernel_size=2, padding=0, stride=2)
        self.conv_block = BaseConv(in_channels=in_channels +
            in_channels_skip, out_channels=out_channels, kernel_size=
            kernel_size, padding=padding, stride=stride)

    def forward(self, x, x_skip):
        x = self.conv_trans1(x)
        x = torch.cat((x, x_skip), dim=1)
        x = self.conv_block(x)
        return x


class UNet(nn.Module):

    def __init__(self, in_channels, out_channels, n_class, kernel_size,
        padding, stride):
        super(UNet, self).__init__()
        self.init_conv = BaseConv(in_channels, out_channels, kernel_size,
            padding, stride)
        self.down1 = DownConv(out_channels, 2 * out_channels, kernel_size,
            padding, stride)
        self.down2 = DownConv(2 * out_channels, 4 * out_channels,
            kernel_size, padding, stride)
        self.down3 = DownConv(4 * out_channels, 8 * out_channels,
            kernel_size, padding, stride)
        self.up3 = UpConv(8 * out_channels, 4 * out_channels, 4 *
            out_channels, kernel_size, padding, stride)
        self.up2 = UpConv(4 * out_channels, 2 * out_channels, 2 *
            out_channels, kernel_size, padding, stride)
        self.up1 = UpConv(2 * out_channels, out_channels, out_channels,
            kernel_size, padding, stride)
        self.out = nn.Conv2d(out_channels, n_class, kernel_size, padding,
            stride)

    def forward(self, x):
        x = self.init_conv(x)
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x_up = self.up3(x3, x2)
        x_up = self.up2(x_up, x1)
        x_up = self.up1(x_up, x)
        x_out = F.log_softmax(self.out(x_up), 1)
        return x_out


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_ptrblck_pytorch_misc(_paritybench_base):
    pass
    def test_000(self):
        self._check(AdaptiveBatchNorm2d(*[], **{'num_features': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(BaseConv(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'padding': 4, 'stride': 1}), [torch.rand([4, 4, 64, 64])], {})

    def test_002(self):
        self._check(DownConv(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'padding': 4, 'stride': 1}), [torch.rand([4, 4, 64, 64])], {})

    @_fails_compile()
    def test_003(self):
        self._check(LocallyConnected2d(*[], **{'in_channels': 4, 'out_channels': 4, 'output_size': 4, 'kernel_size': 4, 'stride': 1}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_004(self):
        self._check(MyBatchNorm2d(*[], **{'num_features': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_005(self):
        self._check(SubModule(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_006(self):
        self._check(UpConv(*[], **{'in_channels': 4, 'in_channels_skip': 4, 'out_channels': 4, 'kernel_size': 4, 'padding': 4, 'stride': 1}), [torch.rand([4, 4, 8, 8]), torch.rand([4, 4, 16, 16])], {})

