import sys
_module = sys.modules[__name__]
del sys
pytorch_refinenet = _module
blocks = _module
refinenet = _module
refinenet_4cascade = _module
setup = _module
test_refinenet = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import torch.nn as nn


import torchvision.models as models


import torch


import torch.optim as optim


class ResidualConvUnit(nn.Module):

    def __init__(self, features):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out + x


class MultiResolutionFusion(nn.Module):

    def __init__(self, out_feats, *shapes):
        super().__init__()
        _, max_size = max(shapes, key=lambda x: x[1])
        self.scale_factors = []
        for i, shape in enumerate(shapes):
            feat, size = shape
            if max_size % size != 0:
                raise ValueError('max_size not divisble by shape {}'.format(i))
            self.scale_factors.append(max_size // size)
            self.add_module('resolve{}'.format(i), nn.Conv2d(feat, out_feats, kernel_size=3, stride=1, padding=1, bias=False))

    def forward(self, *xs):
        output = self.resolve0(xs[0])
        if self.scale_factors[0] != 1:
            output = nn.functional.interpolate(output, scale_factor=self.scale_factors[0], mode='bilinear', align_corners=True)
        for i, x in enumerate(xs[1:], 1):
            output += self.__getattr__('resolve{}'.format(i))(x)
            if self.scale_factors[i] != 1:
                output = nn.functional.interpolate(output, scale_factor=self.scale_factors[i], mode='bilinear', align_corners=True)
        return output


class ChainedResidualPool(nn.Module):

    def __init__(self, feats):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        for i in range(1, 4):
            self.add_module('block{}'.format(i), nn.Sequential(nn.MaxPool2d(kernel_size=5, stride=1, padding=2), nn.Conv2d(feats, feats, kernel_size=3, stride=1, padding=1, bias=False)))

    def forward(self, x):
        x = self.relu(x)
        path = x
        for i in range(1, 4):
            path = self.__getattr__('block{}'.format(i))(path)
            x = x + path
        return x


class ChainedResidualPoolImproved(nn.Module):

    def __init__(self, feats):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        for i in range(1, 5):
            self.add_module('block{}'.format(i), nn.Sequential(nn.Conv2d(feats, feats, kernel_size=3, stride=1, padding=1, bias=False), nn.MaxPool2d(kernel_size=5, stride=1, padding=2)))

    def forward(self, x):
        x = self.relu(x)
        path = x
        for i in range(1, 5):
            path = self.__getattr__('block{}'.format(i))(path)
            x += path
        return x


class BaseRefineNetBlock(nn.Module):

    def __init__(self, features, residual_conv_unit, multi_resolution_fusion, chained_residual_pool, *shapes):
        super().__init__()
        for i, shape in enumerate(shapes):
            feats = shape[0]
            self.add_module('rcu{}'.format(i), nn.Sequential(residual_conv_unit(feats), residual_conv_unit(feats)))
        if len(shapes) != 1:
            self.mrf = multi_resolution_fusion(features, *shapes)
        else:
            self.mrf = None
        self.crp = chained_residual_pool(features)
        self.output_conv = residual_conv_unit(features)

    def forward(self, *xs):
        rcu_xs = []
        for i, x in enumerate(xs):
            rcu_xs.append(self.__getattr__('rcu{}'.format(i))(x))
        if self.mrf is not None:
            out = self.mrf(*rcu_xs)
        else:
            out = rcu_xs[0]
        out = self.crp(out)
        return self.output_conv(out)


class RefineNetBlock(BaseRefineNetBlock):

    def __init__(self, features, *shapes):
        super().__init__(features, ResidualConvUnit, MultiResolutionFusion, ChainedResidualPool, *shapes)


class RefineNetBlockImprovedPooling(nn.Module):

    def __init__(self, features, *shapes):
        super().__init__(features, ResidualConvUnit, MultiResolutionFusion, ChainedResidualPoolImproved, *shapes)


class BaseRefineNet4Cascade(nn.Module):

    def __init__(self, input_shape, refinenet_block, num_classes=1, features=256, resnet_factory=models.resnet101, pretrained=True, freeze_resnet=True):
        """Multi-path 4-Cascaded RefineNet for image segmentation

        Args:
            input_shape ((int, int)): (channel, size) assumes input has
                equal height and width
            refinenet_block (block): RefineNet Block
            num_classes (int, optional): number of classes
            features (int, optional): number of features in refinenet
            resnet_factory (func, optional): A Resnet model from torchvision.
                Default: models.resnet101
            pretrained (bool, optional): Use pretrained version of resnet
                Default: True
            freeze_resnet (bool, optional): Freeze resnet model
                Default: True

        Raises:
            ValueError: size of input_shape not divisible by 32
        """
        super().__init__()
        input_channel, input_size = input_shape
        if input_size % 32 != 0:
            raise ValueError('{} not divisble by 32'.format(input_shape))
        resnet = resnet_factory(pretrained=pretrained)
        self.layer1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1)
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        if freeze_resnet:
            layers = [self.layer1, self.layer2, self.layer3, self.layer4]
            for layer in layers:
                for param in layer.parameters():
                    param.requires_grad = False
        self.layer1_rn = nn.Conv2d(256, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer2_rn = nn.Conv2d(512, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3_rn = nn.Conv2d(1024, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer4_rn = nn.Conv2d(2048, 2 * features, kernel_size=3, stride=1, padding=1, bias=False)
        self.refinenet4 = RefineNetBlock(2 * features, (2 * features, input_size // 32))
        self.refinenet3 = RefineNetBlock(features, (2 * features, input_size // 32), (features, input_size // 16))
        self.refinenet2 = RefineNetBlock(features, (features, input_size // 16), (features, input_size // 8))
        self.refinenet1 = RefineNetBlock(features, (features, input_size // 8), (features, input_size // 4))
        self.output_conv = nn.Sequential(ResidualConvUnit(features), ResidualConvUnit(features), nn.Conv2d(features, num_classes, kernel_size=1, stride=1, padding=0, bias=True))

    def forward(self, x):
        layer_1 = self.layer1(x)
        layer_2 = self.layer2(layer_1)
        layer_3 = self.layer3(layer_2)
        layer_4 = self.layer4(layer_3)
        layer_1_rn = self.layer1_rn(layer_1)
        layer_2_rn = self.layer2_rn(layer_2)
        layer_3_rn = self.layer3_rn(layer_3)
        layer_4_rn = self.layer4_rn(layer_4)
        path_4 = self.refinenet4(layer_4_rn)
        path_3 = self.refinenet3(path_4, layer_3_rn)
        path_2 = self.refinenet2(path_3, layer_2_rn)
        path_1 = self.refinenet1(path_2, layer_1_rn)
        out = self.output_conv(path_1)
        return out


class RefineNet4CascadePoolingImproved(BaseRefineNet4Cascade):

    def __init__(self, input_shape, num_classes=1, features=256, resnet_factory=models.resnet101, pretrained=True, freeze_resnet=True):
        """Multi-path 4-Cascaded RefineNet for image segmentation with improved pooling

        Args:
            input_shape ((int, int)): (channel, size) assumes input has
                equal height and width
            refinenet_block (block): RefineNet Block
            num_classes (int, optional): number of classes
            features (int, optional): number of features in refinenet
            resnet_factory (func, optional): A Resnet model from torchvision.
                Default: models.resnet101
            pretrained (bool, optional): Use pretrained version of resnet
                Default: True
            freeze_resnet (bool, optional): Freeze resnet model
                Default: True

        Raises:
            ValueError: size of input_shape not divisible by 32
        """
        super().__init__(input_shape, RefineNetBlockImprovedPooling, num_classes=num_classes, features=features, resnet_factory=resnet_factory, pretrained=pretrained, freeze_resnet=freeze_resnet)


class RefineNet4Cascade(BaseRefineNet4Cascade):

    def __init__(self, input_shape, num_classes=1, features=256, resnet_factory=models.resnet101, pretrained=True, freeze_resnet=True):
        """Multi-path 4-Cascaded RefineNet for image segmentation

        Args:
            input_shape ((int, int)): (channel, size) assumes input has
                equal height and width
            refinenet_block (block): RefineNet Block
            num_classes (int, optional): number of classes
            features (int, optional): number of features in refinenet
            resnet_factory (func, optional): A Resnet model from torchvision.
                Default: models.resnet101
            pretrained (bool, optional): Use pretrained version of resnet
                Default: True
            freeze_resnet (bool, optional): Freeze resnet model
                Default: True

        Raises:
            ValueError: size of input_shape not divisible by 32
        """
        super().__init__(input_shape, RefineNetBlock, num_classes=num_classes, features=features, resnet_factory=resnet_factory, pretrained=pretrained, freeze_resnet=freeze_resnet)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ChainedResidualPool,
     lambda: ([], {'feats': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ChainedResidualPoolImproved,
     lambda: ([], {'feats': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ResidualConvUnit,
     lambda: ([], {'features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_thomasjpfan_pytorch_refinenet(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

