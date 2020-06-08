import sys
_module = sys.modules[__name__]
del sys
hubconf = _module
resnest = _module
gluon = _module
ablation = _module
data_utils = _module
dropblock = _module
model_store = _module
model_zoo = _module
resnet = _module
splat = _module
torch = _module
resnet = _module
splat = _module
transforms = _module
utils = _module
prepare_imagenet = _module
train = _module
verify = _module
verify = _module
setup = _module
test_radix_major = _module
test_torch = _module

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


import math


import torch


import torch.nn as nn


from torch import nn


import torch.nn.functional as F


from torch.nn import Conv2d


from torch.nn import Module


from torch.nn import Linear


from torch.nn import BatchNorm2d


from torch.nn import ReLU


from torch.nn.modules.utils import _pair


import warnings


import numpy as np


class GlobalAvgPool2d(nn.Module):

    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        return nn.functional.adaptive_avg_pool2d(inputs, 1).view(inputs.
            size(0), -1)


class DropBlock2D(object):

    def __init__(self, *args, **kwargs):
        raise NotImplementedError


class rSoftMax(nn.Module):

    def __init__(self, radix, cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_zhanghang1989_ResNeSt(_paritybench_base):
    pass

    def test_000(self):
        self._check(GlobalAvgPool2d(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(rSoftMax(*[], **{'radix': 4, 'cardinality': 4}), [torch.rand([4, 4, 4, 4])], {})
