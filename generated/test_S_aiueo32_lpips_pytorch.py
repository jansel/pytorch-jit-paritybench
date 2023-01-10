import sys
_module = sys.modules[__name__]
del sys
lpips_pytorch = _module
lpips = _module
networks = _module
utils = _module
setup = _module
tests = _module
test_allclose = _module

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


from typing import Sequence


from itertools import chain


from torchvision import models


from collections import OrderedDict


import torchvision.transforms.functional as TF


from torch.testing import assert_allclose


class LinLayers(nn.ModuleList):

    def __init__(self, n_channels_list: Sequence[int]):
        super(LinLayers, self).__init__([nn.Sequential(nn.Identity(), nn.Conv2d(nc, 1, 1, 1, 0, bias=False)) for nc in n_channels_list])
        for param in self.parameters():
            param.requires_grad = False


def normalize_activation(x, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


class BaseNet(nn.Module):

    def __init__(self):
        super(BaseNet, self).__init__()
        self.register_buffer('mean', torch.Tensor([-0.03, -0.088, -0.188])[None, :, None, None])
        self.register_buffer('std', torch.Tensor([0.458, 0.448, 0.45])[None, :, None, None])

    def set_requires_grad(self, state: bool):
        for param in chain(self.parameters(), self.buffers()):
            param.requires_grad = state

    def z_score(self, x: torch.Tensor):
        return (x - self.mean) / self.std

    def forward(self, x: torch.Tensor):
        x = self.z_score(x)
        output = []
        for i, (_, layer) in enumerate(self.layers._modules.items(), 1):
            x = layer(x)
            if i in self.target_layers:
                output.append(normalize_activation(x))
            if len(output) == len(self.target_layers):
                break
        return output


class AlexNet(BaseNet):

    def __init__(self):
        super(AlexNet, self).__init__()
        self.layers = models.alexnet(True).features
        self.target_layers = [2, 5, 8, 10, 12]
        self.n_channels_list = [64, 192, 384, 256, 256]
        self.set_requires_grad(False)


class SqueezeNet(BaseNet):

    def __init__(self):
        super(SqueezeNet, self).__init__()
        self.layers = models.squeezenet1_1(True).features
        self.target_layers = [2, 5, 8, 10, 11, 12, 13]
        self.n_channels_list = [64, 128, 256, 384, 384, 512, 512]
        self.set_requires_grad(False)


class VGG16(BaseNet):

    def __init__(self):
        super(VGG16, self).__init__()
        self.layers = models.vgg16(True).features
        self.target_layers = [4, 9, 16, 23, 30]
        self.n_channels_list = [64, 128, 256, 512, 512]
        self.set_requires_grad(False)


def get_network(net_type: str):
    if net_type == 'alex':
        return AlexNet()
    elif net_type == 'squeeze':
        return SqueezeNet()
    elif net_type == 'vgg':
        return VGG16()
    else:
        raise NotImplementedError('choose net_type from [alex, squeeze, vgg].')


def get_state_dict(net_type: str='alex', version: str='0.1'):
    url = 'https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/' + f'master/lpips/weights/v{version}/{net_type}.pth'
    old_state_dict = torch.hub.load_state_dict_from_url(url, progress=True, map_location=None if torch.cuda.is_available() else torch.device('cpu'))
    new_state_dict = OrderedDict()
    for key, val in old_state_dict.items():
        new_key = key
        new_key = new_key.replace('lin', '')
        new_key = new_key.replace('model.', '')
        new_state_dict[new_key] = val
    return new_state_dict


class LPIPS(nn.Module):
    """Creates a criterion that measures
    Learned Perceptual Image Patch Similarity (LPIPS).

    Arguments:
        net_type (str): the network type to compare the features: 
                        'alex' | 'squeeze' | 'vgg'. Default: 'alex'.
        version (str): the version of LPIPS. Default: 0.1.
    """

    def __init__(self, net_type: str='alex', version: str='0.1'):
        assert version in ['0.1'], 'v0.1 is only supported now'
        super(LPIPS, self).__init__()
        self.net = get_network(net_type)
        self.lin = LinLayers(self.net.n_channels_list)
        self.lin.load_state_dict(get_state_dict(net_type, version))

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        feat_x, feat_y = self.net(x), self.net(y)
        diff = [((fx - fy) ** 2) for fx, fy in zip(feat_x, feat_y)]
        res = [l(d).mean((2, 3), True) for d, l in zip(diff, self.lin)]
        return torch.sum(torch.cat(res, 0), 0, True)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AlexNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (SqueezeNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (VGG16,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
]

class Test_S_aiueo32_lpips_pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

