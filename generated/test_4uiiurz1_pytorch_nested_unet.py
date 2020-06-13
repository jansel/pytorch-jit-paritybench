import sys
_module = sys.modules[__name__]
del sys
archs = _module
dataset = _module
losses = _module
metrics = _module
preprocess_dsb2018 = _module
train = _module
utils = _module
val = _module

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


from torch import nn


import torch.nn as nn


import torch.nn.functional as F


import numpy as np


from collections import OrderedDict


import torch.backends.cudnn as cudnn


import torch.optim as optim


from torch.optim import lr_scheduler


class VGGBlock(nn.Module):

    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out


class UNet(nn.Module):

    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()
        nb_filter = [32, 64, 128, 256, 512]
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear',
            align_corners=True)
        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])
        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3],
            nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2],
            nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1],
            nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0],
            nb_filter[0])
        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))
        output = self.final(x0_4)
        return output


class NestedUNet(nn.Module):

    def __init__(self, num_classes, input_channels=3, deep_supervision=
        False, **kwargs):
        super().__init__()
        nb_filter = [32, 64, 128, 256, 512]
        self.deep_supervision = deep_supervision
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear',
            align_corners=True)
        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])
        self.conv0_1 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0],
            nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1],
            nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2],
            nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3],
            nb_filter[3])
        self.conv0_2 = VGGBlock(nb_filter[0] * 2 + nb_filter[1], nb_filter[
            0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1] * 2 + nb_filter[2], nb_filter[
            1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2] * 2 + nb_filter[3], nb_filter[
            2], nb_filter[2])
        self.conv0_3 = VGGBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[
            0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1] * 3 + nb_filter[2], nb_filter[
            1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0] * 4 + nb_filter[1], nb_filter[
            0], nb_filter[0])
        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3
            )], 1))
        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]
        else:
            output = self.final(x0_4)
            return output


class BCEDiceLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-05
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = input * target
        dice = (2.0 * intersection.sum(1) + smooth) / (input.sum(1) +
            target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice


class LovaszHingeLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)
        return loss


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_4uiiurz1_pytorch_nested_unet(_paritybench_base):
    pass
    def test_000(self):
        self._check(VGGBlock(*[], **{'in_channels': 4, 'middle_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(UNet(*[], **{'num_classes': 4}), [torch.rand([4, 3, 64, 64])], {})

    @_fails_compile()
    def test_002(self):
        self._check(NestedUNet(*[], **{'num_classes': 4}), [torch.rand([4, 3, 64, 64])], {})

    def test_003(self):
        self._check(BCEDiceLoss(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

