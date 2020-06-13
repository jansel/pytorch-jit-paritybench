import sys
_module = sys.modules[__name__]
del sys
nn_transfer = _module
test = _module
architectures = _module
lenet = _module
simplenet = _module
unet = _module
vggnet = _module
helpers = _module
test_architectures = _module
test_layers = _module
test_util = _module
transfer = _module
util = _module
setup = _module

from _paritybench_helpers import _mock_config
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch.nn as nn


import torch.nn.functional as F


import torch


import numpy as np


class LeNetPytorch(nn.Module):

    def __init__(self):
        super(LeNetPytorch, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class SimpleNetPytorch(nn.Module):

    def __init__(self):
        super(SimpleNetPytorch, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.bn = nn.BatchNorm2d(6)
        self.fc1 = nn.Linear(6 * 14 * 14, 10)

    def forward(self, x):
        out = F.relu(self.bn(self.conv1(x)))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out


class UNetConvBlock(nn.Module):

    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu):
        super(UNetConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel_size, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size, padding=1)
        self.activation = activation

    def forward(self, x):
        out = self.activation(self.conv(x))
        out = self.activation(self.conv2(out))
        return out


class UNetUpBlock(nn.Module):

    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu,
        space_dropout=False):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, 2, stride=2)
        self.conv = nn.Conv2d(in_size, out_size, kernel_size, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size, padding=1)
        self.activation = activation

    def center_crop(self, layer, target_size):
        batch_size, n_channels, layer_width, layer_height = layer.size()
        xy1 = (layer_width - target_size) // 2
        return layer[:, :, xy1:xy1 + target_size, xy1:xy1 + target_size]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.size()[2])
        out = torch.cat([up, crop1], 1)
        out = self.activation(self.conv(out))
        out = self.activation(self.conv2(out))
        return out


class UNetPytorch(nn.Module):

    def __init__(self):
        super(UNetPytorch, self).__init__()
        self.activation = F.relu
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)
        self.pool4 = nn.MaxPool2d(2)
        self.conv_block1_32 = UNetConvBlock(1, 32)
        self.conv_block32_64 = UNetConvBlock(32, 64)
        self.conv_block64_128 = UNetConvBlock(64, 128)
        self.conv_block128_256 = UNetConvBlock(128, 256)
        self.conv_block256_512 = UNetConvBlock(256, 512)
        self.up_block512_256 = UNetUpBlock(512, 256)
        self.up_block256_128 = UNetUpBlock(256, 128)
        self.up_block128_64 = UNetUpBlock(128, 64)
        self.up_block64_32 = UNetUpBlock(64, 32)
        self.last = nn.Conv2d(32, 2, 1)

    def forward(self, x):
        block1 = self.conv_block1_32(x)
        pool1 = self.pool1(block1)
        block2 = self.conv_block32_64(pool1)
        pool2 = self.pool2(block2)
        block3 = self.conv_block64_128(pool2)
        pool3 = self.pool3(block3)
        block4 = self.conv_block128_256(pool3)
        pool4 = self.pool4(block4)
        block5 = self.conv_block256_512(pool4)
        up1 = self.up_block512_256(block5, block4)
        up2 = self.up_block256_128(up1, block3)
        up3 = self.up_block128_64(up2, block2)
        up4 = self.up_block64_32(up3, block1)
        return self.last(up4)


class BatchNet(nn.Module):

    def __init__(self):
        super(BatchNet, self).__init__()
        self.bn = nn.BatchNorm3d(3)

    def forward(self, x):
        return self.bn(x)


class ELUNet(nn.Module):

    def __init__(self):
        super(ELUNet, self).__init__()
        self.elu = nn.ELU()

    def forward(self, x):
        return self.elu(x)


class TransposeNet(nn.Module):

    def __init__(self):
        super(TransposeNet, self).__init__()
        self.trans = nn.ConvTranspose2d(3, 32, 2, 2)

    def forward(self, x):
        return self.trans(x)


class PReLUNet(nn.Module):

    def __init__(self):
        super(PReLUNet, self).__init__()
        self.prelu = nn.PReLU(3)

    def forward(self, x):
        return self.prelu(x)


class Conv2DNet(nn.Module):

    def __init__(self):
        super(Conv2DNet, self).__init__()
        self.conv = nn.Conv2d(3, 16, 7)

    def forward(self, x):
        return self.conv(x)


class Conv3DNet(nn.Module):

    def __init__(self):
        super(Conv3DNet, self).__init__()
        self.conv = nn.Conv3d(3, 8, 5)

    def forward(self, x):
        return self.conv(x)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_gzuidhof_nn_transfer(_paritybench_base):
    pass
    def test_000(self):
        self._check(Conv2DNet(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

    def test_001(self):
        self._check(Conv3DNet(*[], **{}), [torch.rand([4, 3, 64, 64, 64])], {})

    def test_002(self):
        self._check(ELUNet(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(TransposeNet(*[], **{}), [torch.rand([4, 3, 4, 4])], {})

    def test_004(self):
        self._check(UNetConvBlock(*[], **{'in_size': 4, 'out_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_005(self):
        self._check(UNetPytorch(*[], **{}), [torch.rand([4, 1, 64, 64])], {})

