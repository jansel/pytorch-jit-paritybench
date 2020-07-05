import sys
_module = sys.modules[__name__]
del sys
beauty = _module
data_loaders = _module
datasets = _module
image_net = _module
transforms = _module
lr_schedulers = _module
constant_lr = _module
metrics = _module
accuracy = _module
metric_bundle = _module
networks = _module
classifiers = _module
feature_extractors = _module
_feature_extractor = _module
mobile_net_v2 = _module
res_net = _module
networks = _module
submodules = _module
weight_init = _module
task = _module
utils = _module
meters = _module
os_utils = _module
serialization = _module
tensor_utils = _module
setup = _module
train = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, string, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
import numpy as np
from torch import Tensor
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


from torch import nn


from torch.nn import functional as f


import torch


class Accuracy(nn.Module):
    label = 'Accuracy'

    def forward(self, prediction, truth):
        prediction = prediction.argmax(dim=1)
        correct = prediction == truth
        accuracy = correct.float().mean()
        return accuracy


class MetricBundle(nn.Module):

    def __init__(self, metrics):
        super().__init__()
        self.metrics = {metric.label: metric for metric in metrics}

    def forward(self, prediction, truth):
        metric_values = meters.MeterBundle([meters.Meter(label, metric(prediction, truth)) for label, metric in self.metrics.items()])
        return metric_values

    def create_max_meters(self):
        metric_values = self._create_meters(meters.MaxMeter)
        return metric_values

    def create_average_meters(self):
        metric_values = self._create_meters(meters.AverageMeter)
        return metric_values

    def _create_meters(self, meter_type):
        metric_values = meters.MeterBundle([meter_type(label, 0) for label, metric in self.metrics.items()])
        return metric_values


class SoftmaxClassifier(nn.Module):

    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.linear = nn.Linear(input_channels, output_channels)
        weight_init.init(self.modules())

    def forward(self, input_):
        linear = self.linear(input_)
        softmax = f.softmax(linear, dim=1)
        return softmax


class _FeatureExtractor(nn.Module):

    def __init__(self, feature_channels):
        super().__init__()
        self.feature_channels = feature_channels


class BeautyNet(nn.Module):

    def __init__(self, feature_extractor, classifier):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.classifier = classifier

    def forward(self, input_):
        feature = self.feature_extractor(input_)
        global_pool = f.adaptive_avg_pool2d(feature, 1)
        feature_vector = torch.squeeze(torch.squeeze(global_pool, dim=3), dim=2)
        classification = self.classifier(feature_vector)
        return classification


def default_activation():
    activation = nn.ReLU6(inplace=True)
    return activation


def get_perfect_padding(kernel_size, dilation=1):
    padding = (kernel_size - 1) * dilation // 2
    return padding


def sequential(*modules):
    """
    Returns an nn.Sequential object using modules with None's filtered
    """
    modules = [module for module in modules if module is not None]
    return nn.Sequential(*modules)


def conv(in_channels, out_channels, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, normalization=nn.BatchNorm2d, activation=default_activation()):
    padding = padding or get_perfect_padding(kernel_size, dilation)
    layer = sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False), normalization(out_channels), activation)
    return layer


class InvertedResidual(nn.Module):

    def __init__(self, in_channels, out_channels, expansion, stride):
        super().__init__()
        self.stride = stride
        self.is_residual = self.stride == 1 and in_channels == out_channels
        channels = in_channels * expansion
        self.bottlebody = sequential(conv(in_channels, channels, 1, activation=default_activation()), conv(channels, channels, 3, self.stride, groups=channels, activation=default_activation()), conv(channels, out_channels, 1, activation=None))

    def forward(self, input_):
        bottlebody = self.bottlebody(input_)
        output = bottlebody + input_ if self.is_residual else bottlebody
        return output


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Accuracy,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (BeautyNet,
     lambda: ([], {'feature_extractor': _mock_layer(), 'classifier': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InvertedResidual,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'expansion': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_cmsflash_beauty_net(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

