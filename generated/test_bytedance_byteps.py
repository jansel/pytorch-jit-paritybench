import sys
_module = sys.modules[__name__]
del sys
byteps = _module
__version__ = _module
_keras = _module
callbacks = _module
common = _module
keras = _module
misc = _module
imagenet18 = _module
mxnet = _module
ops = _module
server = _module
tensorflow = _module
compression = _module
util = _module
torch = _module
cross_barrier = _module
parallel = _module
distributed = _module
keras_imagenet_resnet50 = _module
keras_mnist = _module
keras_mnist_advanced = _module
data = _module
data_byteps = _module
find_mxnet = _module
fit = _module
fit_byteps = _module
modelzoo = _module
symbols = _module
alexnet = _module
googlenet = _module
lenet = _module
mlp = _module
mobilenet = _module
mobilenetv2 = _module
resnet = _module
resnetv1 = _module
resnext = _module
vgg = _module
train_gluon_mnist_byteps = _module
train_imagenet_byteps = _module
benchmark_byteps = _module
benchmark_byteps_ddp = _module
benchmark_cross_barrier_byteps = _module
elastic_benchmark_byteps = _module
train_imagenet_resnet50_byteps = _module
train_imagenet_resnet_byteps_ddp = _module
train_mnist_byteps = _module
synthetic_benchmark = _module
synthetic_benchmark_tf2 = _module
tensorflow2_mnist = _module
tensorflow_mnist = _module
dist_launcher = _module
launch = _module
pre_setup = _module
setup = _module
test_mxnet = _module
test_tensorflow_keras = _module

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


import collections


from torch.nn.modules import Module


from torch.cuda._utils import _get_device_index


import torch.backends.cudnn as cudnn


import torch.nn.functional as F


import torch.optim as optim


import torch.utils.data.distributed


import numpy as np


import torch.multiprocessing as mp


import torch.nn as nn


import torch.distributed as dist


import math


import random


import warnings


import torch.nn.parallel


import torch.optim


import torch.utils.data


def declare(name):
    c_lib.byteps_torch_declare_tensor(name.encode())
    return 0


def byteps_torch_set_num_grads(num_grads_):
    c_lib.byteps_torch_set_num_grads(num_grads_)
    return 0


def get_ext_suffix():
    """Determine library extension for various versions of Python."""
    ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')
    if ext_suffix:
        return ext_suffix
    ext_suffix = sysconfig.get_config_var('SO')
    if ext_suffix:
        return ext_suffix
    return '.so'


class ConvNet(nn.Module):

    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(1, 16, kernel_size=5, stride=
            1, padding=2), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(
            kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=5, stride
            =1, padding=2), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(
            kernel_size=2, stride=2))
        self.fc = nn.Linear(7 * 7 * 32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_bytedance_byteps(_paritybench_base):
    pass
