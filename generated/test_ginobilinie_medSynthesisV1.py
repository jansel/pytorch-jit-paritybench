import sys
_module = sys.modules[__name__]
del sys
ResUnet3d_pytorch = _module
Unet2d_pytorch = _module
Unet3d_pytorch = _module
compute3DSSIM = _module
dicom2Nii = _module
extract23DPatch4MultiModalImg = _module
extract23DPatch4SingleModalImg = _module
loss_functions = _module
nnBuildUnits = _module
runCTRecon3d = _module
runTesting_Recon = _module
shuffleDataAmongSubjects_2d = _module
shuffleDataAmongSubjects_3d = _module
utils = _module

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


import torch


import torch.nn as nn


import torch.nn.functional as F


from torch.autograd import Variable


import torch.nn.init as init


import numpy as np


import torch.optim as optim


import torch.nn.init


import torch.autograd as autograd


from torch.autograd import Function


from itertools import repeat


import torch.utils.data as data_utils


from scipy import signal


from scipy import ndimage


import copy


class UNetConvBlock(nn.Module):

    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu):
        super(UNetConvBlock, self).__init__()
        self.conv = nn.Conv3d(in_size, out_size, kernel_size, stride=1,
            padding=1)
        self.bn = nn.BatchNorm3d(out_size)
        self.conv2 = nn.Conv3d(out_size, out_size, kernel_size, stride=1,
            padding=1)
        self.bn2 = nn.BatchNorm3d(out_size)
        self.activation = activation
        init.xavier_uniform(self.conv.weight, gain=np.sqrt(2.0))
        init.constant(self.conv.bias, 0)
        init.xavier_uniform(self.conv2.weight, gain=np.sqrt(2.0))
        init.constant(self.conv2.bias, 0)

    def forward(self, x):
        out = self.activation(self.bn(self.conv(x)))
        out = self.activation(self.bn2(self.conv2(out)))
        return out


class residualUnit(nn.Module):

    def __init__(self, in_size, out_size, kernel_size=3, stride=1, padding=
        1, activation=F.relu):
        super(residualUnit, self).__init__()
        self.conv1 = nn.Conv3d(in_size, out_size, kernel_size, stride=1,
            padding=1)
        init.xavier_uniform(self.conv1.weight, gain=np.sqrt(2.0))
        init.constant(self.conv1.bias, 0)
        self.conv2 = nn.Conv3d(out_size, out_size, kernel_size, stride=1,
            padding=1)
        init.xavier_uniform(self.conv2.weight, gain=np.sqrt(2.0))
        init.constant(self.conv2.bias, 0)
        self.activation = activation
        self.bn1 = nn.BatchNorm3d(out_size)
        self.bn2 = nn.BatchNorm3d(out_size)
        self.in_size = in_size
        self.out_size = out_size
        if in_size != out_size:
            self.convX = nn.Conv3d(in_size, out_size, kernel_size=1, stride
                =1, padding=0)
            self.bnX = nn.BatchNorm3d(out_size)

    def forward(self, x):
        out1 = self.activation(self.bn1(self.conv1(x)))
        out2 = self.activation(self.bn1(self.conv2(out1)))
        if self.in_size != self.out_size:
            bridge = self.activation(self.bnX(self.convX(x)))
        output = torch.add(out2, bridge)
        return output


class UNetUpBlock(nn.Module):

    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu,
        space_dropout=False):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose3d(in_size, out_size, 2, stride=2)
        self.bnup = nn.BatchNorm3d(out_size)
        self.conv = nn.Conv3d(in_size, out_size, kernel_size, stride=1,
            padding=1)
        self.bn = nn.BatchNorm3d(out_size)
        self.conv2 = nn.Conv3d(out_size, out_size, kernel_size, stride=1,
            padding=1)
        self.bn2 = nn.BatchNorm3d(out_size)
        self.activation = activation
        init.xavier_uniform(self.up.weight, gain=np.sqrt(2.0))
        init.constant(self.up.bias, 0)
        init.xavier_uniform(self.conv.weight, gain=np.sqrt(2.0))
        init.constant(self.conv.bias, 0)
        init.xavier_uniform(self.conv2.weight, gain=np.sqrt(2.0))
        init.constant(self.conv2.bias, 0)

    def center_crop(self, layer, target_size):
        batch_size, n_channels, layer_width, layer_height, layer_depth = (layer
            .size())
        xy1 = (layer_width - target_size) // 2
        return layer[:, :, xy1:xy1 + target_size, xy1:xy1 + target_size]

    def forward(self, x, bridge):
        up = self.up(x)
        up = self.activation(self.bnup(up))
        crop1 = self.center_crop(bridge, up.size()[2])
        out = torch.cat([up, crop1], 1)
        out = self.activation(self.bn(self.conv(out)))
        out = self.activation(self.bn2(self.conv2(out)))
        return out


class UNetUpResBlock(nn.Module):

    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu,
        space_dropout=False):
        super(UNetUpResBlock, self).__init__()
        self.up = nn.ConvTranspose3d(in_size, out_size, 2, stride=2)
        self.bnup = nn.BatchNorm3d(out_size)
        init.xavier_uniform(self.up.weight, gain=np.sqrt(2.0))
        init.constant(self.up.bias, 0)
        self.activation = activation
        self.resUnit = residualUnit(in_size, out_size, kernel_size=kernel_size)

    def center_crop(self, layer, target_size):
        batch_size, n_channels, layer_width, layer_height, layer_depth = (layer
            .size())
        xy1 = (layer_width - target_size) // 2
        return layer[:, :, xy1:xy1 + target_size, xy1:xy1 + target_size,
            xy1:xy1 + target_size]

    def forward(self, x, bridge):
        up = self.activation(self.bnup(self.up(x)))
        crop1 = bridge
        out = torch.cat([up, crop1], 1)
        out = self.resUnit(out)
        return out


class UNet(nn.Module):

    def __init__(self, in_channel=1, n_classes=4):
        super(UNet, self).__init__()
        self.activation = F.relu
        self.pool1 = nn.MaxPool3d(2)
        self.pool2 = nn.MaxPool3d(2)
        self.pool3 = nn.MaxPool3d(2)
        self.conv_block1_64 = UNetConvBlock(in_channel, 32)
        self.conv_block64_128 = UNetConvBlock(32, 64)
        self.conv_block128_256 = UNetConvBlock(64, 128)
        self.conv_block256_512 = UNetConvBlock(128, 256)
        self.up_block512_256 = UNetUpBlock(256, 128)
        self.up_block256_128 = UNetUpBlock(128, 64)
        self.up_block128_64 = UNetUpBlock(64, 32)
        self.last = nn.Conv3d(32, n_classes, 1, stride=1)

    def forward(self, x):
        block1 = self.conv_block1_64(x)
        pool1 = self.pool1(block1)
        block2 = self.conv_block64_128(pool1)
        pool2 = self.pool2(block2)
        block3 = self.conv_block128_256(pool2)
        pool3 = self.pool3(block3)
        block4 = self.conv_block256_512(pool3)
        up2 = self.up_block512_256(block4, block3)
        up3 = self.up_block256_128(up2, block2)
        up4 = self.up_block128_64(up3, block1)
        return self.last(up4)


class ResUNet(nn.Module):

    def __init__(self, in_channel=1, n_classes=4):
        super(ResUNet, self).__init__()
        self.activation = F.relu
        self.pool1 = nn.MaxPool3d(2)
        self.pool2 = nn.MaxPool3d(2)
        self.pool3 = nn.MaxPool3d(2)
        self.conv_block1_64 = UNetConvBlock(in_channel, 32)
        self.conv_block64_128 = residualUnit(32, 64)
        self.conv_block128_256 = residualUnit(64, 128)
        self.conv_block256_512 = residualUnit(128, 256)
        self.up_block512_256 = UNetUpResBlock(256, 128)
        self.up_block256_128 = UNetUpResBlock(128, 64)
        self.up_block128_64 = UNetUpResBlock(64, 32)
        self.last = nn.Conv3d(32, n_classes, 1, stride=1)

    def forward(self, x):
        block1 = self.conv_block1_64(x)
        pool1 = self.pool1(block1)
        block2 = self.conv_block64_128(pool1)
        pool2 = self.pool2(block2)
        block3 = self.conv_block128_256(pool2)
        pool3 = self.pool3(block3)
        block4 = self.conv_block256_512(pool3)
        up2 = self.up_block512_256(block4, block3)
        up3 = self.up_block256_128(up2, block2)
        up4 = self.up_block128_64(up3, block1)
        return self.last(up4)


class UNet_LRes(nn.Module):

    def __init__(self, in_channel=1, n_classes=4):
        super(UNet_LRes, self).__init__()
        self.activation = F.relu
        self.pool1 = nn.MaxPool3d(2)
        self.pool2 = nn.MaxPool3d(2)
        self.pool3 = nn.MaxPool3d(2)
        self.conv_block1_64 = UNetConvBlock(in_channel, 32)
        self.conv_block64_128 = UNetConvBlock(32, 64)
        self.conv_block128_256 = UNetConvBlock(64, 128)
        self.conv_block256_512 = UNetConvBlock(128, 256)
        self.up_block512_256 = UNetUpBlock(256, 128)
        self.up_block256_128 = UNetUpBlock(128, 64)
        self.up_block128_64 = UNetUpBlock(64, 32)
        self.last = nn.Conv3d(32, n_classes, 1, stride=1)

    def forward(self, x, res_x):
        block1 = self.conv_block1_64(x)
        pool1 = self.pool1(block1)
        block2 = self.conv_block64_128(pool1)
        pool2 = self.pool2(block2)
        block3 = self.conv_block128_256(pool2)
        pool3 = self.pool3(block3)
        block4 = self.conv_block256_512(pool3)
        up2 = self.up_block512_256(block4, block3)
        up3 = self.up_block256_128(up2, block2)
        up4 = self.up_block128_64(up3, block1)
        last = self.last(up4)
        if len(res_x.shape) == 3:
            res_x = res_x.unsqueeze(1)
        out = torch.add(last, res_x)
        return out


class ResUNet_LRes(nn.Module):

    def __init__(self, in_channel=1, n_classes=4, dp_prob=0):
        super(ResUNet_LRes, self).__init__()
        self.activation = F.relu
        self.pool1 = nn.MaxPool3d(2)
        self.pool2 = nn.MaxPool3d(2)
        self.pool3 = nn.MaxPool3d(2)
        self.conv_block1_64 = UNetConvBlock(in_channel, 32)
        self.conv_block64_128 = residualUnit(32, 64)
        self.conv_block128_256 = residualUnit(64, 128)
        self.conv_block256_512 = residualUnit(128, 256)
        self.up_block512_256 = UNetUpResBlock(256, 128)
        self.up_block256_128 = UNetUpResBlock(128, 64)
        self.up_block128_64 = UNetUpResBlock(64, 32)
        self.Dropout = nn.Dropout3d(p=dp_prob)
        self.last = nn.Conv3d(32, n_classes, 1, stride=1)

    def forward(self, x, res_x):
        block1 = self.conv_block1_64(x)
        pool1 = self.pool1(block1)
        pool1_dp = self.Dropout(pool1)
        block2 = self.conv_block64_128(pool1_dp)
        pool2 = self.pool2(block2)
        pool2_dp = self.Dropout(pool2)
        block3 = self.conv_block128_256(pool2_dp)
        pool3 = self.pool3(block3)
        pool3_dp = self.Dropout(pool3)
        block4 = self.conv_block256_512(pool3_dp)
        up2 = self.up_block512_256(block4, block3)
        up3 = self.up_block256_128(up2, block2)
        up4 = self.up_block128_64(up3, block1)
        last = self.last(up4)
        if len(res_x.shape) == 3:
            res_x = res_x.unsqueeze(1)
        out = torch.add(last, res_x)
        return out


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, 9)
        self.bn1 = nn.BatchNorm3d(32)
        self.conv2 = nn.Conv3d(32, 64, 5)
        self.bn2 = nn.BatchNorm3d(64)
        self.conv3 = nn.Conv3d(64, 64, 5)
        self.bn3 = nn.BatchNorm3d(64)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.max_pool3d(F.relu(self.conv1(x)), (2, 2, 2))
        x = F.max_pool3d(F.relu(self.conv2(x)), (2, 2, 2))
        x = F.max_pool3d(F.relu(self.conv3(x)), (2, 2, 2))
        x = x.view(-1, self.num_of_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_of_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class UNetConvBlock(nn.Module):

    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu):
        super(UNetConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel_size, stride=1,
            padding=1)
        self.bn = nn.BatchNorm2d(out_size)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size, stride=1,
            padding=1)
        self.bn2 = nn.BatchNorm2d(out_size)
        self.activation = activation
        init.xavier_uniform(self.conv.weight, gain=np.sqrt(2.0))
        init.constant(self.conv.bias, 0)
        init.xavier_uniform(self.conv2.weight, gain=np.sqrt(2.0))
        init.constant(self.conv2.bias, 0)

    def forward(self, x):
        out = self.activation(self.bn(self.conv(x)))
        out = self.activation(self.bn2(self.conv2(out)))
        return out


class residualUnit(nn.Module):

    def __init__(self, in_size, out_size, kernel_size=3, stride=1, padding=
        1, activation=F.relu):
        super(residualUnit, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size, stride=1,
            padding=1)
        init.xavier_uniform(self.conv1.weight, gain=np.sqrt(2.0))
        init.constant(self.conv1.bias, 0)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size, stride=1,
            padding=1)
        init.xavier_uniform(self.conv2.weight, gain=np.sqrt(2.0))
        init.constant(self.conv2.bias, 0)
        self.activation = activation
        self.bn1 = nn.BatchNorm2d(out_size)
        self.bn2 = nn.BatchNorm2d(out_size)
        self.in_size = in_size
        self.out_size = out_size
        if in_size != out_size:
            self.convX = nn.Conv2d(in_size, out_size, kernel_size=1, stride
                =1, padding=0)
            self.bnX = nn.BatchNorm2d(out_size)

    def forward(self, x):
        out1 = self.activation(self.bn1(self.conv1(x)))
        out2 = self.activation(self.bn1(self.conv2(out1)))
        if self.in_size != self.out_size:
            bridge = self.activation(self.bnX(self.convX(x)))
        output = torch.add(out2, bridge)
        return output


class UNetUpBlock(nn.Module):

    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu,
        space_dropout=False):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, 2, stride=2)
        self.bnup = nn.BatchNorm2d(out_size)
        self.conv = nn.Conv2d(in_size, out_size, kernel_size, stride=1,
            padding=1)
        self.bn = nn.BatchNorm2d(out_size)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size, stride=1,
            padding=1)
        self.bn2 = nn.BatchNorm2d(out_size)
        self.activation = activation
        init.xavier_uniform(self.up.weight, gain=np.sqrt(2.0))
        init.constant(self.up.bias, 0)
        init.xavier_uniform(self.conv.weight, gain=np.sqrt(2.0))
        init.constant(self.conv.bias, 0)
        init.xavier_uniform(self.conv2.weight, gain=np.sqrt(2.0))
        init.constant(self.conv2.bias, 0)

    def center_crop(self, layer, target_size):
        batch_size, n_channels, layer_width, layer_height = layer.size()
        xy1 = (layer_width - target_size) // 2
        return layer[:, :, xy1:xy1 + target_size, xy1:xy1 + target_size]

    def forward(self, x, bridge):
        up = self.up(x)
        up = self.activation(self.bnup(up))
        crop1 = self.center_crop(bridge, up.size()[2])
        out = torch.cat([up, crop1], 1)
        out = self.activation(self.bn(self.conv(out)))
        out = self.activation(self.bn2(self.conv2(out)))
        return out


class UNetUpResBlock(nn.Module):

    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu,
        space_dropout=False):
        super(UNetUpResBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, 2, stride=2)
        self.bnup = nn.BatchNorm2d(out_size)
        init.xavier_uniform(self.up.weight, gain=np.sqrt(2.0))
        init.constant(self.up.bias, 0)
        self.activation = activation
        self.resUnit = residualUnit(in_size, out_size, kernel_size=kernel_size)

    def center_crop(self, layer, target_size):
        batch_size, n_channels, layer_width, layer_height = layer.size()
        xy1 = (layer_width - target_size) // 2
        return layer[:, :, xy1:xy1 + target_size, xy1:xy1 + target_size]

    def forward(self, x, bridge):
        up = self.activation(self.bnup(self.up(x)))
        crop1 = self.center_crop(bridge, up.size()[2])
        out = torch.cat([up, crop1], 1)
        out = self.resUnit(out)
        return out


class UNet(nn.Module):

    def __init__(self, in_channel=1, n_classes=4):
        super(UNet, self).__init__()
        self.activation = F.relu
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)
        self.pool4 = nn.MaxPool2d(2)
        self.conv_block1_64 = UNetConvBlock(in_channel, 64)
        self.conv_block64_128 = UNetConvBlock(64, 128)
        self.conv_block128_256 = UNetConvBlock(128, 256)
        self.conv_block256_512 = UNetConvBlock(256, 512)
        self.conv_block512_1024 = UNetConvBlock(512, 1024)
        self.up_block1024_512 = UNetUpBlock(1024, 512)
        self.up_block512_256 = UNetUpBlock(512, 256)
        self.up_block256_128 = UNetUpBlock(256, 128)
        self.up_block128_64 = UNetUpBlock(128, 64)
        self.last = nn.Conv2d(64, n_classes, 1, stride=1)

    def forward(self, x):
        block1 = self.conv_block1_64(x)
        pool1 = self.pool1(block1)
        block2 = self.conv_block64_128(pool1)
        pool2 = self.pool2(block2)
        block3 = self.conv_block128_256(pool2)
        pool3 = self.pool3(block3)
        block4 = self.conv_block256_512(pool3)
        pool4 = self.pool4(block4)
        block5 = self.conv_block512_1024(pool4)
        up1 = self.up_block1024_512(block5, block4)
        up2 = self.up_block512_256(up1, block3)
        up3 = self.up_block256_128(up2, block2)
        up4 = self.up_block128_64(up3, block1)
        return self.last(up4)


class ResUNet(nn.Module):

    def __init__(self, in_channel=1, n_classes=4):
        super(ResUNet, self).__init__()
        self.activation = F.relu
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)
        self.pool4 = nn.MaxPool2d(2)
        self.conv_block1_64 = UNetConvBlock(in_channel, 64)
        self.conv_block64_128 = residualUnit(64, 128)
        self.conv_block128_256 = residualUnit(128, 256)
        self.conv_block256_512 = residualUnit(256, 512)
        self.conv_block512_1024 = residualUnit(512, 1024)
        self.up_block1024_512 = UNetUpResBlock(1024, 512)
        self.up_block512_256 = UNetUpResBlock(512, 256)
        self.up_block256_128 = UNetUpResBlock(256, 128)
        self.up_block128_64 = UNetUpResBlock(128, 64)
        self.last = nn.Conv2d(64, n_classes, 1, stride=1)

    def forward(self, x):
        block1 = self.conv_block1_64(x)
        pool1 = self.pool1(block1)
        block2 = self.conv_block64_128(pool1)
        pool2 = self.pool2(block2)
        block3 = self.conv_block128_256(pool2)
        pool3 = self.pool3(block3)
        block4 = self.conv_block256_512(pool3)
        pool4 = self.pool4(block4)
        block5 = self.conv_block512_1024(pool4)
        up1 = self.up_block1024_512(block5, block4)
        up2 = self.up_block512_256(up1, block3)
        up3 = self.up_block256_128(up2, block2)
        up4 = self.up_block128_64(up3, block1)
        return self.last(up4)


class UNet_LRes(nn.Module):

    def __init__(self, in_channel=1, n_classes=4):
        super(UNet_LRes, self).__init__()
        self.activation = F.relu
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)
        self.pool4 = nn.MaxPool2d(2)
        self.conv_block1_64 = UNetConvBlock(in_channel, 64)
        self.conv_block64_128 = UNetConvBlock(64, 128)
        self.conv_block128_256 = UNetConvBlock(128, 256)
        self.conv_block256_512 = UNetConvBlock(256, 512)
        self.conv_block512_1024 = UNetConvBlock(512, 1024)
        self.up_block1024_512 = UNetUpBlock(1024, 512)
        self.up_block512_256 = UNetUpBlock(512, 256)
        self.up_block256_128 = UNetUpBlock(256, 128)
        self.up_block128_64 = UNetUpBlock(128, 64)
        self.last = nn.Conv2d(64, n_classes, 1, stride=1)

    def forward(self, x, res_x):
        block1 = self.conv_block1_64(x)
        pool1 = self.pool1(block1)
        block2 = self.conv_block64_128(pool1)
        pool2 = self.pool2(block2)
        block3 = self.conv_block128_256(pool2)
        pool3 = self.pool3(block3)
        block4 = self.conv_block256_512(pool3)
        pool4 = self.pool4(block4)
        block5 = self.conv_block512_1024(pool4)
        up1 = self.up_block1024_512(block5, block4)
        up2 = self.up_block512_256(up1, block3)
        up3 = self.up_block256_128(up2, block2)
        up4 = self.up_block128_64(up3, block1)
        last = self.last(up4)
        if len(res_x.shape) == 3:
            res_x = res_x.unsqueeze(1)
        out = torch.add(last, res_x)
        return out


class ResUNet_LRes(nn.Module):

    def __init__(self, in_channel=1, n_classes=4, dp_prob=0):
        super(ResUNet_LRes, self).__init__()
        self.activation = F.relu
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)
        self.pool4 = nn.MaxPool2d(2)
        self.conv_block1_64 = UNetConvBlock(in_channel, 64)
        self.conv_block64_128 = residualUnit(64, 128)
        self.conv_block128_256 = residualUnit(128, 256)
        self.conv_block256_512 = residualUnit(256, 512)
        self.conv_block512_1024 = residualUnit(512, 1024)
        self.up_block1024_512 = UNetUpResBlock(1024, 512)
        self.up_block512_256 = UNetUpResBlock(512, 256)
        self.up_block256_128 = UNetUpResBlock(256, 128)
        self.up_block128_64 = UNetUpResBlock(128, 64)
        self.Dropout = nn.Dropout2d(p=dp_prob)
        self.last = nn.Conv2d(64, n_classes, 1, stride=1)

    def forward(self, x, res_x):
        block1 = self.conv_block1_64(x)
        pool1 = self.pool1(block1)
        pool1_dp = self.Dropout(pool1)
        block2 = self.conv_block64_128(pool1_dp)
        pool2 = self.pool2(block2)
        pool2_dp = self.Dropout(pool2)
        block3 = self.conv_block128_256(pool2_dp)
        pool3 = self.pool3(block3)
        pool3_dp = self.Dropout(pool3)
        block4 = self.conv_block256_512(pool3_dp)
        pool4 = self.pool4(block4)
        pool4_dp = self.Dropout(pool4)
        block5 = self.conv_block512_1024(pool4_dp)
        up1 = self.up_block1024_512(block5, block4)
        up2 = self.up_block512_256(up1, block3)
        up3 = self.up_block256_128(up2, block2)
        up4 = self.up_block128_64(up3, block1)
        last = self.last(up4)
        if len(res_x.shape) == 3:
            res_x = res_x.unsqueeze(1)
        out = torch.add(last, res_x)
        return out


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, (9, 9))
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, (5, 5))
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, (5, 5))
        self.bn3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        x = x.view(-1, self.num_of_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_of_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class UNet3D(nn.Module):

    def __init__(self, in_channel, n_classes):
        self.in_channel = in_channel
        self.n_classes = n_classes
        super(UNet3D, self).__init__()
        self.ec0 = self.encoder(self.in_channel, 32, bias=False, batchnorm=
            False)
        self.ec1 = self.encoder(32, 64, bias=False, batchnorm=False)
        self.ec2 = self.encoder(64, 64, bias=False, batchnorm=False)
        self.ec3 = self.encoder(64, 128, bias=False, batchnorm=False)
        self.ec4 = self.encoder(128, 128, bias=False, batchnorm=False)
        self.ec5 = self.encoder(128, 256, bias=False, batchnorm=False)
        self.ec6 = self.encoder(256, 256, bias=False, batchnorm=False)
        self.ec7 = self.encoder(256, 512, bias=False, batchnorm=False)
        self.pool0 = nn.MaxPool3d(2)
        self.pool1 = nn.MaxPool3d(2)
        self.pool2 = nn.MaxPool3d(2)
        self.dc9 = self.decoder(512, 512, kernel_size=4, stride=2, padding=
            1, bias=False)
        self.dc8 = self.decoder(256 + 512, 256, kernel_size=3, stride=1,
            padding=1, bias=False)
        self.dc7 = self.decoder(256, 256, kernel_size=3, stride=1, padding=
            1, bias=False)
        self.dc6 = self.decoder(256, 256, kernel_size=4, stride=2, padding=
            1, bias=False)
        self.dc5 = self.decoder(128 + 256, 128, kernel_size=3, stride=1,
            padding=1, bias=False)
        self.dc4 = self.decoder(128, 128, kernel_size=3, stride=1, padding=
            1, bias=False)
        self.dc3 = self.decoder(128, 128, kernel_size=4, stride=2, padding=
            1, bias=False)
        self.dc2 = self.decoder(64 + 128, 64, kernel_size=3, stride=1,
            padding=1, bias=False)
        self.dc1 = self.decoder(64, 64, kernel_size=3, stride=1, padding=1,
            bias=False)
        self.dc0 = self.decoder(64, n_classes, kernel_size=1, stride=1,
            bias=False)

    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1,
        padding=1, bias=True, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(nn.Conv3d(in_channels, out_channels,
                kernel_size, stride=stride, padding=padding, bias=bias), nn
                .BatchNorm2d(out_channels), nn.ReLU())
        else:
            layer = nn.Sequential(nn.Conv3d(in_channels, out_channels,
                kernel_size, stride=stride, padding=padding, bias=bias), nn
                .ReLU())
        return layer

    def decoder(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, output_padding=0, bias=True):
        layer = nn.Sequential(nn.ConvTranspose3d(in_channels, out_channels,
            kernel_size, stride=stride, padding=padding, output_padding=
            output_padding, bias=bias), nn.ReLU())
        return layer

    def forward(self, x):
        e0 = self.ec0(x)
        syn0 = self.ec1(e0)
        e1 = self.pool0(syn0)
        e2 = self.ec2(e1)
        syn1 = self.ec3(e2)
        del e0, e1, e2
        e3 = self.pool1(syn1)
        e4 = self.ec4(e3)
        syn2 = self.ec5(e4)
        del e3, e4
        e5 = self.pool2(syn2)
        e6 = self.ec6(e5)
        e7 = self.ec7(e6)
        del e5, e6
        dc9 = self.dc9(e7)
        d9 = torch.cat((self.dc9(e7), syn2), dim=1)
        del e7, syn2
        d8 = self.dc8(d9)
        d7 = self.dc7(d8)
        del d9, d8
        d6 = torch.cat((self.dc6(d7), syn1), dim=1)
        del d7, syn1
        d5 = self.dc5(d6)
        d4 = self.dc4(d5)
        del d6, d5
        d3 = torch.cat((self.dc3(d4), syn0), dim=1)
        del d4, syn0
        d2 = self.dc2(d3)
        d1 = self.dc1(d2)
        del d3, d2
        d0 = self.dc0(d1)
        return d0


class conv23DUnit(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, groups=1, bias=True, dilation=1, nd=2):
        super(conv23DUnit, self).__init__()
        assert nd == 1 or nd == 2 or nd == 3, 'nd is not correctly specified!!!!, it should be {1,2,3}'
        if nd == 2:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, groups=groups, bias=bias,
                dilation=dilation)
        elif nd == 3:
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, groups=groups, bias=bias,
                dilation=dilation)
        else:
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, groups=groups, bias=bias,
                dilation=dilation)
        init.xavier_uniform(self.conv.weight, gain=np.sqrt(2.0))
        init.constant(self.conv.bias, 0)

    def forward(self, x):
        return self.conv(x)


class conv23D_bn_Unit(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, groups=1, bias=True, dilation=1, nd=2):
        super(conv23D_bn_Unit, self).__init__()
        assert nd == 1 or nd == 2 or nd == 3, 'nd is not correctly specified!!!!, it should be {1,2,3}'
        if nd == 2:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, groups=groups, bias=bias,
                dilation=dilation)
            self.bn = nn.BatchNorm2d(out_channels)
        elif nd == 3:
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, groups=groups, bias=bias,
                dilation=dilation)
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, groups=groups, bias=bias,
                dilation=dilation)
            self.bn = nn.BatchNorm1d(out_channels)
        init.xavier_uniform(self.conv.weight, gain=np.sqrt(2.0))
        init.constant(self.conv.bias, 0)

    def forward(self, x):
        return self.bn(self.conv(x))


class conv23D_bn_relu_Unit(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, groups=1, bias=True, dilation=1, nd=2):
        super(conv23D_bn_relu_Unit, self).__init__()
        assert nd == 1 or nd == 2 or nd == 3, 'nd is not correctly specified!!!!, it should be {1,2,3}'
        if nd == 2:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, groups=groups, bias=bias,
                dilation=dilation)
            self.bn = nn.BatchNorm2d(out_channels)
        elif nd == 3:
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, groups=groups, bias=bias,
                dilation=dilation)
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, groups=groups, bias=bias,
                dilation=dilation)
            self.bn = nn.BatchNorm1d(out_channels)
        init.xavier_uniform(self.conv.weight, gain=np.sqrt(2.0))
        init.constant(self.conv.bias, 0)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class convTranspose23DUnit(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, output_padding=0, groups=1, bias=True, dilation=1, nd=2):
        super(convTranspose23DUnit, self).__init__()
        assert nd == 1 or nd == 2 or nd == 3, 'nd is not correctly specified!!!!, it should be {1,2,3}'
        if nd == 2:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels,
                kernel_size, stride=stride, padding=padding, output_padding
                =output_padding, groups=groups, bias=bias, dilation=dilation)
        elif nd == 3:
            self.conv = nn.ConvTranspose3d(in_channels, out_channels,
                kernel_size, stride=stride, padding=padding, output_padding
                =output_padding, groups=groups, bias=bias, dilation=dilation)
        else:
            self.conv = nn.ConvTranspose1d(in_channels, out_channels,
                kernel_size, stride=stride, padding=padding, output_padding
                =output_padding, groups=groups, bias=bias, dilation=dilation)
        init.xavier_uniform(self.conv.weight, gain=np.sqrt(2.0))
        init.constant(self.conv.bias, 0)

    def forward(self, x):
        return self.conv(x)


class convTranspose23D_bn_Unit(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, output_padding=0, groups=1, bias=True, dilation=1, nd=2):
        super(convTranspose23D_bn_Unit, self).__init__()
        assert nd == 1 or nd == 2 or nd == 3, 'nd is not correctly specified!!!!, it should be {1,2,3}'
        if nd == 2:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels,
                kernel_size, stride=stride, padding=padding, output_padding
                =output_padding, groups=groups, bias=bias, dilation=dilation)
            self.bn = nn.BatchNorm2d(out_channels)
        elif nd == 3:
            self.conv = nn.ConvTranspose3d(in_channels, out_channels,
                kernel_size, stride=stride, padding=padding, output_padding
                =output_padding, groups=groups, bias=bias, dilation=dilation)
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            self.conv = nn.ConvTranspose1d(in_channels, out_channels,
                kernel_size, stride=stride, padding=padding, output_padding
                =output_padding, groups=groups, bias=bias, dilation=dilation)
            self.bn = nn.BatchNorm1d(out_channels)
        init.xavier_uniform(self.conv.weight, gain=np.sqrt(2.0))
        init.constant(self.conv.bias, 0)

    def forward(self, x):
        return self.bn(self.conv(x))


class convTranspose23D_bn_relu_Unit(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, output_padding=0, groups=1, bias=True, dilation=1, nd=2):
        super(convTranspose23D_bn_relu_Unit, self).__init__()
        assert nd == 1 or nd == 2 or nd == 3, 'nd is not correctly specified!!!!, it should be {1,2,3}'
        if nd == 2:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels,
                kernel_size, stride=stride, padding=padding, output_padding
                =output_padding, groups=groups, bias=bias, dilation=dilation)
            self.bn = nn.BatchNorm2d(out_channels)
        elif nd == 3:
            self.conv = nn.ConvTranspose3d(in_channels, out_channels,
                kernel_size, stride=stride, padding=padding, output_padding
                =output_padding, groups=groups, bias=bias, dilation=dilation)
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            self.conv = nn.ConvTranspose1d(in_channels, out_channels,
                kernel_size, stride=stride, padding=padding, output_padding
                =output_padding, groups=groups, bias=bias, dilation=dilation)
            self.bn = nn.BatchNorm1d(out_channels)
        init.xavier_uniform(self.conv.weight, gain=np.sqrt(2.0))
        init.constant(self.conv.bias, 0)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class dropout23DUnit(nn.Module):

    def __init__(self, prob=0, nd=2):
        super(dropout23DUnit, self).__init__()
        assert nd == 1 or nd == 2 or nd == 3, 'nd is not correctly specified!!!!, it should be {1,2,3}'
        if nd == 2:
            self.dp = nn.Dropout2d(p=prob)
        elif nd == 3:
            self.dp = nn.Dropout3d(p=prob)
        else:
            self.dp = nn.Dropout(p=prob)

    def forward(self, x):
        return self.dp(x)


class maxPool23DUinit(nn.Module):

    def __init__(self, kernel_size, stride, padding=1, dilation=1, nd=2):
        super(maxPool23DUinit, self).__init__()
        assert nd == 1 or nd == 2 or nd == 3, 'nd is not correctly specified!!!!, it should be {1,2,3}'
        if nd == 2:
            self.pool1 = nn.MaxPool2d(kernel_size=kernel_size, stride=
                stride, padding=padding, dilation=dilation)
        elif nd == 3:
            self.pool1 = nn.MaxPool3d(kernel_size=kernel_size, stride=
                stride, padding=padding, dilation=dilation)
        else:
            self.pool1 = nn.MaxPool1d(kernel_size=kernel_size, stride=
                stride, padding=padding, dilation=dilation)

    def forward(self, x):
        return self.pool1(x)


class convUnit(nn.Module):

    def __init__(self, in_size, out_size, kernel_size=3, stride=1, padding=
        1, activation=F.relu):
        super(convUnit, self).__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel_size, stride, padding)
        init.xavier_uniform(self.conv.weight, gain=np.sqrt(2.0))
        init.constant(self.conv.bias, 0)
        self.bn = nn.BatchNorm2d(out_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class residualUnit(nn.Module):

    def __init__(self, in_size, out_size, kernel_size=3, stride=1, padding=
        1, activation=F.relu, nd=2):
        super(residualUnit, self).__init__()
        self.conv1 = conv23DUnit(in_size, out_size, kernel_size, stride,
            padding, nd=nd)
        self.conv2 = conv23DUnit(out_size, out_size, kernel_size, stride,
            padding, nd=nd)

    def forward(self, x):
        return F.relu(self.conv2(F.elu(self.conv1(x))) + x)


class residualUnit1(nn.Module):

    def __init__(self, in_size, out_size, kernel_size=3, stride=1, padding=
        1, activation=F.relu, nd=2):
        super(residualUnit1, self).__init__()
        self.conv1_bn_relu = conv23D_bn_relu_Unit(in_size, out_size,
            kernel_size, stride, padding, nd=nd)
        self.relu = nn.ReLU()
        self.conv2_bn_relu = nn.conv23D_bn_relu_Unit(out_size, out_size,
            kernel_size, stride, padding, nd=nd)

    def forward(self, x):
        identity_data = x
        output = self.conv1_bn_relu(x)
        output = self.conv2_bn_relu(output)
        output = torch.add(output, identity_data)
        output = self.relu(output)
        return output


class residualUnit3(nn.Module):

    def __init__(self, in_size, out_size, isDilation=None, isEmptyBranch1=
        None, activation=F.relu, nd=2):
        super(residualUnit3, self).__init__()
        mid_size = out_size / 2
        if isDilation:
            self.conv1_bn_relu = conv23D_bn_relu_Unit(in_channels=in_size,
                out_channels=mid_size, kernel_size=1, stride=1, padding=0,
                dilation=2, nd=nd)
        else:
            self.conv1_bn_relu = conv23D_bn_relu_Unit(in_channels=in_size,
                out_channels=mid_size, kernel_size=1, stride=1, padding=0,
                nd=nd)
        self.relu = nn.ReLU()
        if isDilation:
            self.conv2_bn_relu = conv23D_bn_relu_Unit(in_channels=mid_size,
                out_channels=mid_size, kernel_size=3, stride=1, padding=2,
                dilation=2, nd=nd)
        else:
            self.conv2_bn_relu = conv23D_bn_relu_Unit(in_channels=mid_size,
                out_channels=mid_size, kernel_size=3, stride=1, padding=1,
                nd=nd)
        if isDilation:
            self.conv3_bn = conv23D_bn_Unit(in_channels=mid_size,
                out_channels=out_size, kernel_size=1, stride=1, padding=0,
                dilation=2, nd=nd)
        else:
            self.conv3_bn = conv23D_bn_Unit(in_channels=mid_size,
                out_channels=out_size, kernel_size=1, stride=1, padding=0,
                nd=nd)
        self.isEmptyBranch1 = isEmptyBranch1
        if in_size != out_size or isEmptyBranch1 == False:
            if isDilation:
                self.convX_bn = conv23D_bn_Unit(in_channels=in_size,
                    out_channels=out_size, kernel_size=1, stride=1, padding
                    =0, dilation=2, nd=nd)
            else:
                self.convX_bn = conv23D_bn_Unit(in_channels=in_size,
                    out_channels=out_size, kernel_size=1, stride=1, padding
                    =0, nd=nd)

    def forward(self, x):
        identity_data = x
        output = self.conv1_bn_relu(x)
        output = self.conv2_bn_relu(output)
        output = self.conv3_bn(output)
        outSZ = output.size()
        idSZ = identity_data.size()
        if outSZ[1] != idSZ[1] or self.isEmptyBranch1 == False:
            identity_data = self.convX_bn(identity_data)
        output = torch.add(output, identity_data)
        output = self.relu(output)
        return output


class longResidualUnit(nn.Module):

    def __init__(self, in_size, out_size, kernel_size=3, stride=1, padding=
        1, activation=F.relu, nd=2):
        super(residualUnit1, self).__init__()
        self.conv1_bn = conv23D_bn_Unit(in_channels=in_size, out_channels=
            out_size, kernel_size=kernel_size, stride=stride, padding=
            padding, nd=nd)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity_data = x
        output = self.conv1_bn(x)
        output = torch.add(output, identity_data)
        output = self.relu(output)
        return output


class ResUpUnit(nn.Module):

    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu,
        spatial_dropout_rate=0, isConvDilation=None, nd=2):
        super(ResUpUnit, self).__init__()
        self.nd = nd
        self.up = convTranspose23D_bn_relu_Unit(in_size, out_size,
            kernel_size=4, stride=2, padding=1, nd=nd)
        self.conv = residualUnit3(out_size, out_size, isDilation=
            isConvDilation, nd=nd)
        self.dp = dropout23DUnit(prob=spatial_dropout_rate, nd=nd)
        self.spatial_dropout_rate = spatial_dropout_rate
        self.conv2 = residualUnit3(out_size, out_size, isDilation=
            isConvDilation, isEmptyBranch1=False, nd=nd)
        self.relu = nn.ReLU()

    def center_crop(self, layer, target_size):
        if self.nd == 2:
            batch_size, n_channels, layer_width, layer_height = layer.size()
        elif self.nd == 3:
            (batch_size, n_channels, layer_width, layer_height, layer_depth
                ) = layer.size()
        xy1 = (layer_width - target_size) // 2
        if self.nd == 3:
            return layer[:, :, xy1:xy1 + target_size, xy1:xy1 + target_size,
                xy1:xy1 + target_size]
        return layer[:, :, xy1:xy1 + target_size, xy1:xy1 + target_size]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = bridge
        if self.spatial_dropout_rate > 0:
            crop1 = self.dp(crop1)
        out = self.relu(torch.add(up, crop1))
        out = self.conv(out)
        out = self.conv2(out)
        return out


class DilatedResUnit(nn.Module):

    def __init__(self, in_size, out_size, kernel_size=3, stride=1, dilation
        =2, nd=2):
        super(DilatedResUnit, self).__init__()
        self.nd = nd
        mid_size = out_size / 1
        padding = dilation * (kernel_size - 1) / 2
        self.conv1_bn_relu = conv23D_bn_relu_Unit(in_channels=in_size,
            out_channels=mid_size, kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation, nd=nd)
        self.conv2_bn_relu = conv23D_bn_relu_Unit(in_channels=mid_size,
            out_channels=mid_size, kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation, nd=nd)
        self.relu = nn.ReLU()

    def forward(self, x):
        conv1_1 = self.conv1_bn_relu(x)
        conv1_2 = self.conv2_bn_relu(conv1_1)
        out1 = torch.add(x, conv1_2)
        conv2_1 = self.conv1_bn_relu(out1)
        conv2_2 = self.conv2_bn_relu(conv2_1)
        out = torch.add(conv2_1, conv2_2)
        return out


class BaseResUpUnit(nn.Module):

    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu,
        space_dropout=False, nd=2):
        super(BaseResUpUnit, self).__init__()
        self.nd = nd
        self.up = convTranspose23D_bn_relu_Unit(in_size, out_size,
            kernel_size=4, stride=2, padding=1, nd=nd)
        self.relu = nn.ReLU()

    def center_crop(self, layer, target_size):
        if self.nd == 2:
            batch_size, n_channels, layer_width, layer_height = layer.size()
        elif self.nd == 3:
            (batch_size, n_channels, layer_width, layer_height, layer_depth
                ) = layer.size()
        xy1 = (layer_width - target_size) // 2
        if self.nd == 3:
            return layer[:, :, xy1:xy1 + target_size, xy1:xy1 + target_size,
                xy1:xy1 + target_size]
        return layer[:, :, xy1:xy1 + target_size, xy1:xy1 + target_size]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = bridge
        out = self.relu(torch.add(up, crop1))
        return out


class upsampleUnit(nn.Module):

    def __init__(self, in_channels, out_channels, nd=2):
        super(upsampleUnit, self).__init__()
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1_bn_relu = conv23D_bn_relu_Unit(in_channels, out_channels,
            3, stride=1, padding=1, nd=nd)

    def forward(self, x):
        return self.conv1_bn_relu(x)


class unetConvUnit(nn.Module):

    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu,
        nd=2):
        super(unetConvUnit, self).__init__()
        self.conv = conv23DUnit(in_size, out_size, kernel_size=3, stride=1,
            padding=1, nd=nd)
        self.conv2 = conv23DUnit(out_size, out_size, kernel_size=3, stride=
            1, padding=1, nd=nd)
        self.activation = activation

    def forward(self, x):
        out = self.activation(self.conv(x))
        out = self.activation(self.conv2(out))
        return out


class unetUpUnit(nn.Module):

    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu,
        space_dropout=False, nd=2):
        super(unetUpUnit, self).__init__()
        self.up = convTranspose23DUnit(in_size, out_size, kernel_size=4,
            stride=2, padding=1, nd=nd)
        self.conv = conv23DUnit(in_size, out_size, kernel_size=3, stride=1,
            padding=1, nd=nd)
        self.conv2 = conv23DUnit(out_size, out_size, kernel_size=3, stride=
            1, padding=1, nd=nd)
        self.activation = activation
        self.nd = nd

    def center_crop(self, layer, target_size):
        if self.nd == 2:
            batch_size, n_channels, layer_width, layer_height = layer.size()
        elif self.nd == 3:
            (batch_size, n_channels, layer_width, layer_height, layer_depth
                ) = layer.size()
        xy1 = (layer_width - target_size) // 2
        if self.nd == 3:
            return layer[:, :, xy1:xy1 + target_size, xy1:xy1 + target_size,
                xy1:xy1 + target_size]
        return layer[:, :, xy1:xy1 + target_size, xy1:xy1 + target_size]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = bridge
        out = torch.cat([up, crop1], 1)
        out = self.activation(self.conv(out))
        out = self.activation(self.conv2(out))
        return out


class WeightedCrossEntropy3d(nn.Module):

    def __init__(self, weight=None, size_average=True, reduce=True,
        ignore_label=255):
        """weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses\""""
        super(WeightedCrossEntropy3d, self).__init__()
        self.weight = weight
        self.size_average = size_average
        self.ignore_label = ignore_label
        self.nll_loss = nn.NLLLoss(weight, size_average=False, reduce=False)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, predict, target, weight_map=None):
        """
            Args:
                predict:(n, c, h, w, d)
                target:(n, h, w, d): 0,1,...,C-1       
        """
        assert not target.requires_grad
        assert predict.dim() == 5
        assert target.dim() == 4
        assert predict.size(0) == target.size(0), '{0} vs {1} '.format(predict
            .size(0), target.size(0))
        assert predict.size(2) == target.size(1), '{0} vs {1} '.format(predict
            .size(2), target.size(1))
        assert predict.size(3) == target.size(2), '{0} vs {1} '.format(predict
            .size(3), target.size(2))
        assert predict.size(4) == target.size(3), '{0} vs {1} '.format(predict
            .size(4), target.size(3))
        n, c, h, w, d = predict.size()
        logits = self.logsoftmax(predict)
        voxel_loss = self.nll_loss(logits, target)
        weighted_voxel_loss = weight_map * voxel_loss
        loss = torch.sum(weighted_voxel_loss) / (n * h * w * d)
        return loss


class CrossEntropy3d(nn.Module):

    def __init__(self, weight=None, size_average=True, ignore_label=255):
        """weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses\""""
        super(CrossEntropy3d, self).__init__()
        self.weight = weight
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target):
        """
            Args:
                predict:(n, c, h, w, d)
                target:(n, h, w, d): 0,1,...,C-1       
        """
        assert not target.requires_grad
        assert predict.dim() == 5
        assert target.dim() == 4
        assert predict.size(0) == target.size(0), '{0} vs {1} '.format(predict
            .size(0), target.size(0))
        assert predict.size(2) == target.size(1), '{0} vs {1} '.format(predict
            .size(2), target.size(1))
        assert predict.size(3) == target.size(2), '{0} vs {1} '.format(predict
            .size(3), target.size(2))
        assert predict.size(4) == target.size(3), '{0} vs {1} '.format(predict
            .size(4), target.size(3))
        n, c, h, w, d = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        predict = predict.transpose(1, 2).transpose(2, 3).transpose(3, 4
            ).contiguous()
        predict = predict[target_mask.view(n, h, w, d, 1).repeat(1, 1, 1, 1, c)
            ].view(-1, c)
        loss = F.cross_entropy(predict, target, weight=self.weight,
            size_average=self.size_average)
        return loss


class CrossEntropy2d(nn.Module):

    def __init__(self, weight=None, size_average=True, ignore_label=255):
        """weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses\""""
        super(CrossEntropy2d, self).__init__()
        self.weight = weight
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w): 0,1,...,C-1       
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), '{0} vs {1} '.format(predict
            .size(0), target.size(0))
        assert predict.size(2) == target.size(1), '{0} vs {1} '.format(predict
            .size(2), target.size(1))
        assert predict.size(3) == target.size(2), '{0} vs {1} '.format(predict
            .size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)
            ].view(-1, c)
        loss = F.cross_entropy(predict, target, weight=self.weight,
            size_average=self.size_average)
        return loss


class CrossEntropyLoss2d(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)


class myFocalLoss(nn.Module):

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(myFocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        elif isinstance(alpha, Variable):
            self.alpha = alpha
        else:
            self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        assert inputs.dim() == 4, 'inputs size should be 4: NXCXWXH'
        N = inputs.size(0)
        C = inputs.size(1)
        W = inputs.size(2)
        H = inputs.size(3)
        P = F.softmax(inputs, dim=1)
        class_mask = inputs.data.new(N, C, W, H).fill_(0)
        class_mask = Variable(class_mask)
        targets = torch.unsqueeze(targets, 1)
        class_mask.scatter_(1, targets, 1)
        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha
        alpha = 0.25
        probs = (P * class_mask).sum(1).view(-1, 1)
        log_p = probs.log()
        batch_loss = -alpha * torch.pow(1 - probs, self.gamma) * log_p
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class myWeightedDiceLoss4Organs(nn.Module):

    def __init__(self, organIDs=[1], organWeights=[1]):
        super(myWeightedDiceLoss4Organs, self).__init__()
        self.organIDs = organIDs
        self.organWeights = organWeights

    def forward(self, inputs, targets, save=True):
        """
            Args:
                inputs:(n, c, h, w, d)
                targets:(n, h, w, d): 0,1,...,C-1       
        """
        assert not targets.requires_grad
        assert inputs.dim() == 5, inputs.shape
        assert targets.dim() == 4, targets.shape
        assert inputs.size(0) == targets.size(0), '{0} vs {1} '.format(inputs
            .size(0), targets.size(0))
        assert inputs.size(2) == targets.size(1), '{0} vs {1} '.format(inputs
            .size(2), targets.size(1))
        assert inputs.size(3) == targets.size(2), '{0} vs {1} '.format(inputs
            .size(3), targets.size(2))
        assert inputs.size(4) == targets.size(3), '{0} vs {1} '.format(inputs
            .size(4), targets.size(3))
        eps = Variable(torch.cuda.FloatTensor(1).fill_(1e-06))
        one = Variable(torch.cuda.FloatTensor(1).fill_(1.0))
        two = Variable(torch.cuda.FloatTensor(1).fill_(2.0))
        inputSZ = inputs.size()
        inputs = F.softmax(inputs, dim=1)
        numOfCategories = inputSZ[1]
        assert numOfCategories == len(self.organWeights
            ), 'organ weights is not matched with organs (bg should be included)'
        results_one_hot = inputs
        target1 = Variable(torch.unsqueeze(targets.data, 1))
        targets_one_hot = Variable(torch.cuda.FloatTensor(inputSZ).zero_())
        targets_one_hot.scatter_(1, target1, 1)
        out = Variable(torch.cuda.FloatTensor(1).zero_(), requires_grad=True)
        for organID in range(0, numOfCategories):
            target = targets_one_hot[:, (organID), (...)].contiguous().view(
                -1, 1).squeeze(1)
            result = results_one_hot[:, (organID), (...)].contiguous().view(
                -1, 1).squeeze(1)
            intersect_vec = result * target
            intersect = torch.sum(intersect_vec)
            result_sum = torch.sum(result)
            target_sum = torch.sum(target)
            union = result_sum + target_sum + two * eps
            IoU = intersect / union
            out = out + self.organWeights[organID] * (one - two * IoU)
        denominator = Variable(torch.cuda.FloatTensor(1).fill_(sum(self.
            organWeights)))
        out = out / denominator
        return out


class GeneralizedDiceLoss4Organs(nn.Module):

    def __init__(self, organIDs=[1], size_average=True):
        super(GeneralizedDiceLoss4Organs, self).__init__()
        self.organIDs = organIDs
        self.size_average = size_average

    def forward(self, inputs, targets, save=True):
        """
            Args:
                inputs:(n, c, h, w, d)
                targets:(n, h, w, d): 0,1,...,C-1       
        """
        assert not targets.requires_grad
        assert inputs.dim() == 5, inputs.shape
        assert targets.dim() == 4, targets.shape
        assert inputs.size(0) == targets.size(0), '{0} vs {1} '.format(inputs
            .size(0), targets.size(0))
        assert inputs.size(2) == targets.size(1), '{0} vs {1} '.format(inputs
            .size(2), targets.size(1))
        assert inputs.size(3) == targets.size(2), '{0} vs {1} '.format(inputs
            .size(3), targets.size(2))
        assert inputs.size(4) == targets.size(3), '{0} vs {1} '.format(inputs
            .size(4), targets.size(3))
        eps = Variable(torch.cuda.FloatTensor(1).fill_(1e-06))
        one = Variable(torch.cuda.FloatTensor(1).fill_(1.0))
        two = Variable(torch.cuda.FloatTensor(1).fill_(2.0))
        inputSZ = inputs.size()
        inputs = F.softmax(inputs, dim=1)
        numOfCategories = inputSZ[1]
        assert numOfCategories == len(self.organIDs
            ), 'organ weights is not matched with organs (bg should be included)'
        results_one_hot = inputs
        target1 = Variable(torch.unsqueeze(targets.data, 1))
        targets_one_hot = Variable(torch.cuda.FloatTensor(inputSZ).zero_())
        targets_one_hot.scatter_(1, target1, 1)
        out = Variable(torch.cuda.FloatTensor(1).zero_(), requires_grad=True)
        intersect = Variable(torch.cuda.FloatTensor(1).fill_(0.0))
        union = Variable(torch.cuda.FloatTensor(1).fill_(0.0))
        for organID in range(0, numOfCategories):
            target = targets_one_hot[:, (organID), (...)].contiguous().view(
                -1, 1).squeeze(1)
            result = results_one_hot[:, (organID), (...)].contiguous().view(
                -1, 1).squeeze(1)
            if torch.sum(target).cpu().data[0] == 0:
                organWeight = Variable(torch.cuda.FloatTensor(1).fill_(0.0))
            else:
                organWeight = 1 / (torch.sum(target) ** 2 + eps)
            intersect_vec = result * target
            intersect = intersect + organWeight * torch.sum(intersect_vec)
            result_sum = torch.sum(result)
            target_sum = torch.sum(target)
            union = union + organWeight * (result_sum + target_sum) + two * eps
        IoU = intersect / union
        out = one - two * IoU
        return out


class topK_RegLoss(nn.Module):

    def __init__(self, topK, size_average=True):
        super(topK_RegLoss, self).__init__()
        self.size_average = size_average
        self.topK = topK

    def forward(self, preds, targets):
        """
            Args:
                inputs:(n, h, w, d)
                targets:(n, h, w, d)  
        """
        assert not targets.requires_grad
        assert preds.shape == targets.shape, 'dim of preds and targets are different'
        K = torch.abs(preds - targets).view(-1)
        if len(preds.shape) == 4:
            V, I = torch.topk(K, int(preds.size(0) * preds.size(1) * preds.
                size(2) * preds.size(3) * self.topK), largest=True, sorted=True
                )
        else:
            V, I = torch.topk(K, int(preds.size(0) * preds.size(1) * preds.
                size(2) * self.topK), largest=True, sorted=True)
        loss = torch.mean(V)
        return loss


class RelativeThreshold_RegLoss(nn.Module):

    def __init__(self, threshold, size_average=True):
        super(RelativeThreshold_RegLoss, self).__init__()
        self.size_average = size_average
        self.eps = 1e-07
        self.threshold = threshold

    def forward(self, preds, targets):
        """
            Args:
                inputs:(n, h, w, d)
                targets:(n, h, w, d)  
        """
        assert not targets.requires_grad
        assert preds.shape == targets.shape, 'dim of preds and targets are different'
        dist = torch.abs(preds - targets).view(-1)
        baseV = targets.view(-1)
        baseV = torch.abs(baseV + self.eps)
        relativeDist = torch.div(dist, baseV)
        mask = relativeDist.ge(self.threshold)
        largerLossVec = torch.masked_select(dist, mask)
        loss = torch.mean(largerLossVec)
        return loss


class gdl_loss(nn.Module):

    def __init__(self, pNorm=2):
        super(gdl_loss, self).__init__()
        self.convX = nn.Conv2d(1, 1, kernel_size=(1, 2), stride=1, padding=
            (0, 1), bias=False)
        self.convY = nn.Conv2d(1, 1, kernel_size=(2, 1), stride=1, padding=
            (1, 0), bias=False)
        filterX = torch.FloatTensor([[[[-1, 1]]]])
        filterY = torch.FloatTensor([[[[1], [-1]]]])
        self.convX.weight = torch.nn.Parameter(filterX, requires_grad=False)
        self.convY.weight = torch.nn.Parameter(filterY, requires_grad=False)
        self.pNorm = pNorm

    def forward(self, pred, gt):
        assert not gt.requires_grad
        assert pred.dim() == 4
        assert gt.dim() == 4
        assert pred.size() == gt.size(), '{0} vs {1} '.format(pred.size(),
            gt.size())
        pred_dx = torch.abs(self.convX(pred))
        pred_dy = torch.abs(self.convY(pred))
        gt_dx = torch.abs(self.convX(gt))
        gt_dy = torch.abs(self.convY(gt))
        grad_diff_x = torch.abs(gt_dx - pred_dx)
        grad_diff_y = torch.abs(gt_dy - pred_dy)
        mat_loss_x = grad_diff_x ** self.pNorm
        mat_loss_y = grad_diff_y ** self.pNorm
        shape = gt.shape
        mean_loss = (torch.sum(mat_loss_x) + torch.sum(mat_loss_y)) / (
            shape[0] * shape[1] * shape[2] * shape[3])
        return mean_loss


class FeatureExtractor(nn.Module):

    def __init__(self, cnn, feature_layer=8):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(cnn.features.children())[:
            feature_layer + 1])

    def forward(self, x):
        return self.features(x)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_ginobilinie_medSynthesisV1(_paritybench_base):
    pass

    def test_000(self):
        self._check(UNetConvBlock(*[], **{'in_size': 4, 'out_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(residualUnit(*[], **{'in_size': 4, 'out_size': 4}), [torch.rand([4, 4, 4, 4])], {})
    @_fails_compile()

    def test_002(self):
        self._check(UNet(*[], **{}), [torch.rand([4, 1, 64, 64])], {})
    @_fails_compile()

    def test_003(self):
        self._check(UNet_LRes(*[], **{}), [torch.rand([4, 1, 64, 64]), torch.rand([4, 4, 64, 64])], {})

    def test_004(self):
        self._check(Discriminator(*[], **{}), [torch.rand([4, 1, 64, 64])], {})

    def test_005(self):
        self._check(conv23DUnit(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_006(self):
        self._check(conv23D_bn_Unit(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_007(self):
        self._check(conv23D_bn_relu_Unit(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_008(self):
        self._check(convTranspose23DUnit(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_009(self):
        self._check(convTranspose23D_bn_Unit(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_010(self):
        self._check(convTranspose23D_bn_relu_Unit(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})
    @_fails_compile()

    def test_011(self):
        self._check(dropout23DUnit(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_012(self):
        self._check(maxPool23DUinit(*[], **{'kernel_size': 4, 'stride': 1}), [torch.rand([4, 4, 4, 4])], {})

    def test_013(self):
        self._check(convUnit(*[], **{'in_size': 4, 'out_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_014(self):
        self._check(BaseResUpUnit(*[], **{'in_size': 4, 'out_size': 4}), [torch.rand([4, 4, 8, 8]), torch.rand([4, 4, 16, 16])], {})

    def test_015(self):
        self._check(upsampleUnit(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_016(self):
        self._check(unetConvUnit(*[], **{'in_size': 4, 'out_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_017(self):
        self._check(RelativeThreshold_RegLoss(*[], **{'threshold': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})
