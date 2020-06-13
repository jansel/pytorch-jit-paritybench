import sys
_module = sys.modules[__name__]
del sys
setup = _module
preprocess_coco_person = _module
trt_pose = _module
coco = _module
draw_objects = _module
models = _module
common = _module
densenet = _module
dla = _module
mnasnet = _module
resnet = _module
parse_objects = _module
train = _module
utils = _module
export_for_isaac = _module

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


import torch.utils.data


import torch.nn


import numpy as np


import torch.optim


import torch.nn.functional as F


import re


class UpsampleCBR(torch.nn.Sequential):

    def __init__(self, input_channels, output_channels, count=1, num_flat=0):
        layers = []
        for i in range(count):
            if i == 0:
                inch = input_channels
            else:
                inch = output_channels
            layers += [torch.nn.ConvTranspose2d(inch, output_channels,
                kernel_size=4, stride=2, padding=1), torch.nn.BatchNorm2d(
                output_channels), torch.nn.ReLU()]
            for i in range(num_flat):
                layers += [torch.nn.Conv2d(output_channels, output_channels,
                    kernel_size=3, stride=1, padding=1), torch.nn.
                    BatchNorm2d(output_channels), torch.nn.ReLU()]
        super(UpsampleCBR, self).__init__(*layers)


class SelectInput(torch.nn.Module):

    def __init__(self, index):
        super(SelectInput, self).__init__()
        self.index = index

    def forward(self, inputs):
        return inputs[self.index]


class CmapPafHead(torch.nn.Module):

    def __init__(self, input_channels, cmap_channels, paf_channels,
        upsample_channels=256, num_upsample=0, num_flat=0):
        super(CmapPafHead, self).__init__()
        if num_upsample > 0:
            self.cmap_conv = torch.nn.Sequential(UpsampleCBR(input_channels,
                upsample_channels, num_upsample, num_flat), torch.nn.Conv2d
                (upsample_channels, cmap_channels, kernel_size=1, stride=1,
                padding=0))
            self.paf_conv = torch.nn.Sequential(UpsampleCBR(input_channels,
                upsample_channels, num_upsample, num_flat), torch.nn.Conv2d
                (upsample_channels, paf_channels, kernel_size=1, stride=1,
                padding=0))
        else:
            self.cmap_conv = torch.nn.Conv2d(input_channels, cmap_channels,
                kernel_size=1, stride=1, padding=0)
            self.paf_conv = torch.nn.Conv2d(input_channels, paf_channels,
                kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return self.cmap_conv(x), self.paf_conv(x)


class CmapPafHeadAttention(torch.nn.Module):

    def __init__(self, input_channels, cmap_channels, paf_channels,
        upsample_channels=256, num_upsample=0, num_flat=0):
        super(CmapPafHeadAttention, self).__init__()
        self.cmap_up = UpsampleCBR(input_channels, upsample_channels,
            num_upsample, num_flat)
        self.paf_up = UpsampleCBR(input_channels, upsample_channels,
            num_upsample, num_flat)
        self.cmap_att = torch.nn.Conv2d(upsample_channels,
            upsample_channels, kernel_size=3, stride=1, padding=1)
        self.paf_att = torch.nn.Conv2d(upsample_channels, upsample_channels,
            kernel_size=3, stride=1, padding=1)
        self.cmap_conv = torch.nn.Conv2d(upsample_channels, cmap_channels,
            kernel_size=1, stride=1, padding=0)
        self.paf_conv = torch.nn.Conv2d(upsample_channels, paf_channels,
            kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        xc = self.cmap_up(x)
        ac = torch.sigmoid(self.cmap_att(xc))
        xp = self.paf_up(x)
        ap = torch.tanh(self.paf_att(xp))
        return self.cmap_conv(xc * ac), self.paf_conv(xp * ap)


class DenseNetBackbone(torch.nn.Module):

    def __init__(self, densenet):
        super(DenseNetBackbone, self).__init__()
        self.densenet = densenet

    def forward(self, x):
        x = self.densenet.features(x)
        return x


class DlaWrapper(torch.nn.Module):

    def __init__(self, dla_fn, cmap_channels, paf_channels):
        super(DlaWrapper, self).__init__()
        self.backbone = dla_fn(cmap_channels + paf_channels,
            pretrained_base='imagenet')
        self.cmap_channels = cmap_channels
        self.paf_channels = paf_channels

    def forward(self, x):
        x = self.backbone(x)
        cmap, paf = torch.split(x, [self.cmap_channels, self.paf_channels],
            dim=1)
        return cmap, paf


class MnasnetBackbone(torch.nn.Module):

    def __init__(self, backbone):
        super(MnasnetBackbone, self).__init__()
        self.backbone = backbone

    def forward(self, x):
        x = self.backbone.layers(x)
        return x


class ResNetBackbone(torch.nn.Module):

    def __init__(self, resnet):
        super(ResNetBackbone, self).__init__()
        self.resnet = resnet

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        return x


class InputReNormalization(torch.nn.Module):
    """
        This defines "(input - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]" custom operation
        to conform to "Unit" normalized input RGB data.
    """

    def __init__(self):
        super(InputReNormalization, self).__init__()
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))
        self.std = torch.Tensor([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


class HeatmapMaxpoolAndPermute(torch.nn.Module):
    """
        This defines MaxPool2d(kernel_size = 3, stride = 1) and permute([0,2,3,1]) custom operation
        to conform to [part_affinity_fields, heatmap, maxpool_heatmap] output format.
    """

    def __init__(self):
        super(HeatmapMaxpoolAndPermute, self).__init__()
        self.maxpool = torch.nn.MaxPool2d(3, stride=1, padding=1)

    def forward(self, x):
        heatmap, part_affinity_fields = x
        maxpool_heatmap = self.maxpool(heatmap)
        part_affinity_fields = part_affinity_fields.permute([0, 2, 3, 1])
        heatmap = heatmap.permute([0, 2, 3, 1])
        maxpool_heatmap = maxpool_heatmap.permute([0, 2, 3, 1])
        return [part_affinity_fields, heatmap, maxpool_heatmap]


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_NVIDIA_AI_IOT_trt_pose(_paritybench_base):
    pass
    def test_000(self):
        self._check(UpsampleCBR(*[], **{'input_channels': 4, 'output_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(CmapPafHead(*[], **{'input_channels': 4, 'cmap_channels': 4, 'paf_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(CmapPafHeadAttention(*[], **{'input_channels': 4, 'cmap_channels': 4, 'paf_channels': 4}), [torch.rand([4, 256, 64, 64])], {})

    def test_003(self):
        self._check(InputReNormalization(*[], **{}), [torch.rand([4, 3, 4, 4])], {})

