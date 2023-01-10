import sys
_module = sys.modules[__name__]
del sys
resnet = _module
timings_resnet = _module
conf = _module
nngeometry = _module
generator = _module
dummy = _module
jacobian = _module
grads = _module
grads_conv = _module
layercollection = _module
layers = _module
maths = _module
metrics = _module
object = _module
fspace = _module
map = _module
pspace = _module
vector = _module
utils = _module
setup = _module
conftest = _module
tasks = _module
test_conv_switch = _module
test_jacobian = _module
test_jacobian_ekfac = _module
test_jacobian_kfac = _module
test_layers = _module
test_maths = _module
test_metrics = _module
test_pickle = _module
test_representations = _module
test_utils = _module
test_vector = _module
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


from torchvision import datasets


from torchvision import transforms


from torch.utils.data import Subset


from torch.utils.data import DataLoader


import time


from torch.utils.data import TensorDataset


import numpy as np


from torch._C import unify_type_list


from abc import ABC


from collections import OrderedDict


from functools import reduce


from torch import Tensor


from torch.nn import Linear


from torch.nn import Conv2d


from torch.nn import Module


from torch.nn import init


from torch.nn import functional as F


from torch.nn.parameter import Parameter


from torch.nn.functional import softmax


from abc import abstractmethod


import torch.nn.functional as tF


from torch.nn.modules.conv import ConvTranspose2d


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out = F.relu(out + self.shortcut(x))
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.conv3(out)
        out = F.relu(out + self.shortcut(x))
        return out


class ResNet(nn.Module):

    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class Cosine1d(Linear):
    """Computes the cosine similarity between rows of the weight matrix
    and the incoming data
    """

    def __init__(self, in_features: int, out_features: int, eps=1e-05) ->None:
        super(Cosine1d, self).__init__(in_features=in_features, out_features=out_features, bias=False)
        self.eps = eps

    def forward(self, input: Tensor) ->Tensor:
        norm2_w = (self.weight ** 2).sum(dim=1, keepdim=True) + self.eps
        norm2_x = (input ** 2).sum(dim=1, keepdim=True) + self.eps
        return F.linear(input / torch.sqrt(norm2_x), self.weight / torch.sqrt(norm2_w))


class WeightNorm1d(Linear):
    """Computes an affine mapping of the incoming data using a weight matrix
    with rows normalized with norm 1
    """

    def __init__(self, in_features: int, out_features: int, eps=1e-05) ->None:
        super(WeightNorm1d, self).__init__(in_features=in_features, out_features=out_features, bias=False)
        self.eps = eps

    def forward(self, input: Tensor) ->Tensor:
        norm2 = (self.weight ** 2).sum(dim=1, keepdim=True) + self.eps
        return F.linear(input, self.weight / torch.sqrt(norm2))


class WeightNorm2d(Conv2d):
    """Computes a 2d convolution using a kernel weight matrix
    with rows normalized with norm 1
    """

    def __init__(self, *args, eps=1e-05, **kwargs) ->None:
        assert 'bias' not in kwargs or kwargs['bias'] is False
        super(WeightNorm2d, self).__init__(*args, bias=False, **kwargs)
        self.eps = eps

    def forward(self, input: Tensor) ->Tensor:
        norm2 = (self.weight ** 2).sum(dim=(1, 2, 3), keepdim=True) + self.eps
        return self._conv_forward(input, self.weight / torch.sqrt(norm2), None)


class Affine1d(Module):
    """Computes the transformation out = weight * input + bias
    where * is the elementwise multiplication. This is similar to the
    scaling and translation given by parameters gamma and beta in batch norm

    """

    def __init__(self, num_features: int, bias: bool=True, device=None, dtype=None) ->None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Affine1d, self).__init__()
        self.num_features = num_features
        self.weight = Parameter(torch.empty(num_features, **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(num_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) ->None:
        init.ones_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, input: Tensor) ->Tensor:
        if self.bias is not None:
            return input * self.weight.unsqueeze(0) + self.bias
        else:
            return input * self.weight.unsqueeze(0)

    def extra_repr(self) ->str:
        return 'num_features={}, bias={}'.format(self.num_features, self.bias is not None)


class FCNet(nn.Module):

    def __init__(self, out_size=10, normalization='none'):
        if normalization not in ['none', 'batch_norm', 'weight_norm', 'cosine', 'affine']:
            raise NotImplementedError
        super(FCNet, self).__init__()
        layers = []
        sizes = [18 * 18, 10, 10, out_size]
        for i, (s_in, s_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            if normalization == 'weight_norm':
                layers.append(WeightNorm1d(s_in, s_out))
            elif normalization == 'cosine':
                layers.append(Cosine1d(s_in, s_out))
            else:
                layers.append(nn.Linear(s_in, s_out, bias=normalization == 'none'))
            if normalization == 'batch_norm':
                layers.append(nn.BatchNorm1d(s_out))
            elif normalization == 'affine':
                layers.append(Affine1d(s_out, bias=i % 2 == 0))
            layers.append(nn.ReLU())
        layers.pop()
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x[:, :, 5:-5, 5:-5].contiguous()
        x = x.view(x.size(0), -1)
        return self.net(x)


class FCNetSegmentation(nn.Module):

    def __init__(self, out_size=10):
        super(FCNetSegmentation, self).__init__()
        layers = []
        self.out_size = out_size
        sizes = [18 * 18, 10, 10, 4 * 4 * out_size]
        for s_in, s_out in zip(sizes[:-1], sizes[1:]):
            layers.append(nn.Linear(s_in, s_out))
            layers.append(nn.ReLU())
        layers.pop()
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x[:, :, 5:-5, 5:-5].contiguous()
        x = x.view(x.size(0), -1)
        return self.net(x).view(-1, self.out_size, 4, 4)


class ConvNet(nn.Module):

    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, 3, 1)
        self.conv2 = nn.Conv2d(5, 6, 4, 1, bias=False)
        self.conv3 = nn.Conv2d(6, 7, 3, 1)
        self.fc1 = nn.Linear(1 * 1 * 7, 10)

    def forward(self, x):
        x = tF.relu(self.conv1(x))
        x = tF.max_pool2d(x, 2, 2)
        x = tF.relu(self.conv2(x))
        x = tF.max_pool2d(x, 2, 2)
        x = tF.relu(self.conv3(x))
        x = tF.max_pool2d(x, 2, 2)
        x = x.view(-1, 1 * 1 * 7)
        x = self.fc1(x)
        return tF.log_softmax(x, dim=1)


class SmallConvNet(nn.Module):

    def __init__(self, normalization='none'):
        super(SmallConvNet, self).__init__()
        self.normalization = normalization
        if normalization == 'weight_norm':
            self.l1 = WeightNorm2d(1, 6, 3, 2)
            self.l2 = WeightNorm2d(6, 3, 2, 3)
        elif normalization == 'transpose':
            self.l1 = ConvTranspose2d(1, 6, (3, 2), 2)
            self.l2 = ConvTranspose2d(6, 3, (2, 3), 3, bias=False)
        else:
            raise NotImplementedError

    def forward(self, x):
        x = x[:, :, 5:-5, 5:-5]
        x = tF.relu(self.l1(x))
        x = tF.relu(self.l2(x))
        return x.sum(dim=(2, 3))


class LinearFCNet(nn.Module):

    def __init__(self):
        super(LinearFCNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 3)
        self.fc2 = nn.Linear(28 * 28, 7, bias=False)

    def forward(self, x):
        fc1_out = self.fc1(x.view(x.size(0), -1))
        fc2_out = self.fc2(x.view(x.size(0), -1))
        output = torch.stack([fc1_out.sum(dim=1), fc2_out.sum(dim=1)], dim=1)
        return output


class LinearConvNet(nn.Module):

    def __init__(self):
        super(LinearConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, 3, 1)
        self.conv2 = nn.Conv2d(1, 3, 2, 1, bias=False)

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(x)
        output = torch.stack([conv1_out.sum(dim=(1, 2, 3)), conv2_out.sum(dim=(1, 2, 3))], dim=1)
        return output


class BatchNormFCLinearNet(nn.Module):

    def __init__(self):
        super(BatchNormFCLinearNet, self).__init__()
        self.fc0 = nn.Linear(28 * 28, 100)
        self.bn1 = nn.BatchNorm1d(100)
        self.bn2 = nn.BatchNorm1d(100)

    def forward(self, x):
        x = self.fc0(x.view(x.size(0), -1))
        bn1_out = self.bn1(x)
        bn2_out = self.bn2(-x)
        output = torch.stack([bn1_out.sum(dim=1), bn2_out.sum(dim=1)], dim=1)
        return output


class BatchNormConvLinearNet(nn.Module):

    def __init__(self):
        super(BatchNormConvLinearNet, self).__init__()
        self.conv0 = nn.Conv2d(1, 5, 3, 3)
        self.conv1 = nn.BatchNorm2d(5)
        self.conv2 = nn.BatchNorm2d(5)

    def forward(self, x):
        conv0_out = self.conv0(x)
        conv1_out = self.conv1(conv0_out)
        conv2_out = self.conv2(-conv0_out)
        output = torch.stack([conv1_out.sum(dim=(1, 2, 3)), conv2_out.sum(dim=(1, 2, 3))], dim=1)
        return output


class BatchNormNonLinearNet(nn.Module):
    """
    BN Layer followed by a Linear Layer
    This is used to test jacobians against
    there linerization since this network does not
    suffer from the nonlinearity incurred by stacking
    a Linear Layer then a BN Layer
    """

    def __init__(self):
        super(BatchNormNonLinearNet, self).__init__()
        self.bnconv = nn.BatchNorm2d(2)
        self.bnfc = nn.BatchNorm1d(28 * 28)
        self.fc = nn.Linear(2352, 5)

    def forward(self, x):
        bs = x.size(0)
        two_channels = torch.cat([x, x.permute(0, 1, 3, 2)], dim=1)
        bnconv_out = self.bnconv(two_channels)
        bnfc_out = self.bnfc(x.view(bs, -1))
        stacked = torch.cat([bnconv_out.view(bs, -1), bnfc_out], dim=1)
        output = self.fc(stacked)
        return output


class ConvNetWithSkipConnection(nn.Module):

    def __init__(self):
        super(ConvNetWithSkipConnection, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, 3, 1, 1)
        self.conv2 = nn.Conv2d(2, 2, 3, 1, 1)
        self.conv3 = nn.Conv2d(2, 2, 3, 1, 1)
        self.conv4 = nn.Conv2d(2, 3, 3, 1, 1)

    def forward(self, x):
        x_before_skip = tF.relu(self.conv1(x))
        x_block = tF.relu(self.conv2(x_before_skip))
        x_after_skip = tF.relu(self.conv3(x_block))
        x = tF.relu(self.conv4(x_after_skip + x_before_skip))
        x = x.sum(axis=(2, 3))
        return x


class Net(nn.Module):

    def __init__(self, in_size=10, out_size=10, n_hidden=2, hidden_size=25, nonlinearity=nn.ReLU):
        super(Net, self).__init__()
        layers = []
        sizes = [in_size] + [hidden_size] * n_hidden + [out_size]
        for s_in, s_out in zip(sizes[:-1], sizes[1:]):
            layers.append(nn.Linear(s_in, s_out))
            layers.append(nonlinearity())
        layers.pop()
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.net(x)
        return out


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Affine1d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BasicBlock,
     lambda: ([], {'in_planes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BatchNormConvLinearNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64])], {}),
     True),
    (Bottleneck,
     lambda: ([], {'in_planes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64])], {}),
     True),
    (ConvNetWithSkipConnection,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64])], {}),
     False),
    (Cosine1d,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LinearConvNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64])], {}),
     True),
    (WeightNorm1d,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (WeightNorm2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_tfjgeorge_nngeometry(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

    def test_004(self):
        self._check(*TESTCASES[4])

    def test_005(self):
        self._check(*TESTCASES[5])

    def test_006(self):
        self._check(*TESTCASES[6])

    def test_007(self):
        self._check(*TESTCASES[7])

    def test_008(self):
        self._check(*TESTCASES[8])

    def test_009(self):
        self._check(*TESTCASES[9])

