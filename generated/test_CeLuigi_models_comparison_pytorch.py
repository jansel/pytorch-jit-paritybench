import sys
_module = sys.modules[__name__]
del sys
pretrainedmodels = _module
models = _module
bninception = _module
cafferesnet = _module
dpn = _module
fbresnet = _module
resnet152_load = _module
googlenet = _module
inceptionresnetv2 = _module
inceptionv4 = _module
mobilenet = _module
mobilenetv2 = _module
nasnet = _module
nasnet_mobile = _module
resnext = _module
resnext_features = _module
resnext101_32x4d_features = _module
resnext101_64x4d_features = _module
senet = _module
shufflenet = _module
state_dict_surgery = _module
torchvision_models = _module
utils = _module
vggm = _module
wideresnet = _module
xception = _module
utils = _module
version = _module
compute_accuracy_rate = _module
compute_computational_complexity = _module
compute_inference_time = _module
compute_memory_usage = _module

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


import torch.utils.model_zoo as model_zoo


import math


import torch.nn.functional as F


from collections import OrderedDict


from torch import nn as nnl


import collections


import torchvision.transforms as transforms


import numpy as np


from torch.autograd import Variable


from functools import reduce


from torch.utils import model_zoo


from torch.nn import init


import torchvision.models as models


import time


import torch.nn.parallel


import torch.backends.cudnn as cudnn


import torchvision.datasets as datasets


class BNInception(nn.Module):

    def __init__(self, num_classes=1000):
        super(BNInception, self).__init__()
        inplace = True
        self.conv1_7x7_s2 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.conv1_7x7_s2_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.9, affine=True)
        self.conv1_relu_7x7 = nn.ReLU(inplace)
        self.pool1_3x3_s2 = nn.MaxPool2d((3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=True)
        self.conv2_3x3_reduce = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
        self.conv2_3x3_reduce_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.9, affine=True)
        self.conv2_relu_3x3_reduce = nn.ReLU(inplace)
        self.conv2_3x3 = nn.Conv2d(64, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2_3x3_bn = nn.BatchNorm2d(192, eps=1e-05, momentum=0.9, affine=True)
        self.conv2_relu_3x3 = nn.ReLU(inplace)
        self.pool2_3x3_s2 = nn.MaxPool2d((3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=True)
        self.inception_3a_1x1 = nn.Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3a_1x1_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3a_relu_1x1 = nn.ReLU(inplace)
        self.inception_3a_3x3_reduce = nn.Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3a_3x3_reduce_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3a_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_3a_3x3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3a_3x3_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3a_relu_3x3 = nn.ReLU(inplace)
        self.inception_3a_double_3x3_reduce = nn.Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3a_double_3x3_reduce_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3a_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_3a_double_3x3_1 = nn.Conv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3a_double_3x3_1_bn = nn.BatchNorm2d(96, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3a_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_3a_double_3x3_2 = nn.Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3a_double_3x3_2_bn = nn.BatchNorm2d(96, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3a_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_3a_pool = nn.AvgPool2d(3, stride=1, padding=1, ceil_mode=True, count_include_pad=True)
        self.inception_3a_pool_proj = nn.Conv2d(192, 32, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3a_pool_proj_bn = nn.BatchNorm2d(32, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3a_relu_pool_proj = nn.ReLU(inplace)
        self.inception_3b_1x1 = nn.Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3b_1x1_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3b_relu_1x1 = nn.ReLU(inplace)
        self.inception_3b_3x3_reduce = nn.Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3b_3x3_reduce_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3b_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_3b_3x3 = nn.Conv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3b_3x3_bn = nn.BatchNorm2d(96, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3b_relu_3x3 = nn.ReLU(inplace)
        self.inception_3b_double_3x3_reduce = nn.Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3b_double_3x3_reduce_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3b_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_3b_double_3x3_1 = nn.Conv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3b_double_3x3_1_bn = nn.BatchNorm2d(96, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3b_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_3b_double_3x3_2 = nn.Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3b_double_3x3_2_bn = nn.BatchNorm2d(96, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3b_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_3b_pool = nn.AvgPool2d(3, stride=1, padding=1, ceil_mode=True, count_include_pad=True)
        self.inception_3b_pool_proj = nn.Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3b_pool_proj_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3b_relu_pool_proj = nn.ReLU(inplace)
        self.inception_3c_3x3_reduce = nn.Conv2d(320, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3c_3x3_reduce_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3c_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_3c_3x3 = nn.Conv2d(128, 160, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.inception_3c_3x3_bn = nn.BatchNorm2d(160, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3c_relu_3x3 = nn.ReLU(inplace)
        self.inception_3c_double_3x3_reduce = nn.Conv2d(320, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3c_double_3x3_reduce_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3c_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_3c_double_3x3_1 = nn.Conv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3c_double_3x3_1_bn = nn.BatchNorm2d(96, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3c_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_3c_double_3x3_2 = nn.Conv2d(96, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.inception_3c_double_3x3_2_bn = nn.BatchNorm2d(96, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3c_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_3c_pool = nn.MaxPool2d((3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=True)
        self.inception_4a_1x1 = nn.Conv2d(576, 224, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4a_1x1_bn = nn.BatchNorm2d(224, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4a_relu_1x1 = nn.ReLU(inplace)
        self.inception_4a_3x3_reduce = nn.Conv2d(576, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4a_3x3_reduce_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4a_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_4a_3x3 = nn.Conv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4a_3x3_bn = nn.BatchNorm2d(96, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4a_relu_3x3 = nn.ReLU(inplace)
        self.inception_4a_double_3x3_reduce = nn.Conv2d(576, 96, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4a_double_3x3_reduce_bn = nn.BatchNorm2d(96, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4a_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_4a_double_3x3_1 = nn.Conv2d(96, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4a_double_3x3_1_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4a_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_4a_double_3x3_2 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4a_double_3x3_2_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4a_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_4a_pool = nn.AvgPool2d(3, stride=1, padding=1, ceil_mode=True, count_include_pad=True)
        self.inception_4a_pool_proj = nn.Conv2d(576, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4a_pool_proj_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4a_relu_pool_proj = nn.ReLU(inplace)
        self.inception_4b_1x1 = nn.Conv2d(576, 192, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4b_1x1_bn = nn.BatchNorm2d(192, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4b_relu_1x1 = nn.ReLU(inplace)
        self.inception_4b_3x3_reduce = nn.Conv2d(576, 96, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4b_3x3_reduce_bn = nn.BatchNorm2d(96, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4b_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_4b_3x3 = nn.Conv2d(96, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4b_3x3_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4b_relu_3x3 = nn.ReLU(inplace)
        self.inception_4b_double_3x3_reduce = nn.Conv2d(576, 96, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4b_double_3x3_reduce_bn = nn.BatchNorm2d(96, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4b_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_4b_double_3x3_1 = nn.Conv2d(96, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4b_double_3x3_1_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4b_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_4b_double_3x3_2 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4b_double_3x3_2_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4b_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_4b_pool = nn.AvgPool2d(3, stride=1, padding=1, ceil_mode=True, count_include_pad=True)
        self.inception_4b_pool_proj = nn.Conv2d(576, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4b_pool_proj_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4b_relu_pool_proj = nn.ReLU(inplace)
        self.inception_4c_1x1 = nn.Conv2d(576, 160, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4c_1x1_bn = nn.BatchNorm2d(160, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4c_relu_1x1 = nn.ReLU(inplace)
        self.inception_4c_3x3_reduce = nn.Conv2d(576, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4c_3x3_reduce_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4c_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_4c_3x3 = nn.Conv2d(128, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4c_3x3_bn = nn.BatchNorm2d(160, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4c_relu_3x3 = nn.ReLU(inplace)
        self.inception_4c_double_3x3_reduce = nn.Conv2d(576, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4c_double_3x3_reduce_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4c_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_4c_double_3x3_1 = nn.Conv2d(128, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4c_double_3x3_1_bn = nn.BatchNorm2d(160, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4c_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_4c_double_3x3_2 = nn.Conv2d(160, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4c_double_3x3_2_bn = nn.BatchNorm2d(160, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4c_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_4c_pool = nn.AvgPool2d(3, stride=1, padding=1, ceil_mode=True, count_include_pad=True)
        self.inception_4c_pool_proj = nn.Conv2d(576, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4c_pool_proj_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4c_relu_pool_proj = nn.ReLU(inplace)
        self.inception_4d_1x1 = nn.Conv2d(608, 96, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4d_1x1_bn = nn.BatchNorm2d(96, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4d_relu_1x1 = nn.ReLU(inplace)
        self.inception_4d_3x3_reduce = nn.Conv2d(608, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4d_3x3_reduce_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4d_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_4d_3x3 = nn.Conv2d(128, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4d_3x3_bn = nn.BatchNorm2d(192, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4d_relu_3x3 = nn.ReLU(inplace)
        self.inception_4d_double_3x3_reduce = nn.Conv2d(608, 160, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4d_double_3x3_reduce_bn = nn.BatchNorm2d(160, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4d_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_4d_double_3x3_1 = nn.Conv2d(160, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4d_double_3x3_1_bn = nn.BatchNorm2d(192, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4d_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_4d_double_3x3_2 = nn.Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4d_double_3x3_2_bn = nn.BatchNorm2d(192, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4d_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_4d_pool = nn.AvgPool2d(3, stride=1, padding=1, ceil_mode=True, count_include_pad=True)
        self.inception_4d_pool_proj = nn.Conv2d(608, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4d_pool_proj_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4d_relu_pool_proj = nn.ReLU(inplace)
        self.inception_4e_3x3_reduce = nn.Conv2d(608, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4e_3x3_reduce_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4e_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_4e_3x3 = nn.Conv2d(128, 192, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.inception_4e_3x3_bn = nn.BatchNorm2d(192, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4e_relu_3x3 = nn.ReLU(inplace)
        self.inception_4e_double_3x3_reduce = nn.Conv2d(608, 192, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4e_double_3x3_reduce_bn = nn.BatchNorm2d(192, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4e_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_4e_double_3x3_1 = nn.Conv2d(192, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4e_double_3x3_1_bn = nn.BatchNorm2d(256, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4e_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_4e_double_3x3_2 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.inception_4e_double_3x3_2_bn = nn.BatchNorm2d(256, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4e_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_4e_pool = nn.MaxPool2d((3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=True)
        self.inception_5a_1x1 = nn.Conv2d(1056, 352, kernel_size=(1, 1), stride=(1, 1))
        self.inception_5a_1x1_bn = nn.BatchNorm2d(352, eps=1e-05, momentum=0.9, affine=True)
        self.inception_5a_relu_1x1 = nn.ReLU(inplace)
        self.inception_5a_3x3_reduce = nn.Conv2d(1056, 192, kernel_size=(1, 1), stride=(1, 1))
        self.inception_5a_3x3_reduce_bn = nn.BatchNorm2d(192, eps=1e-05, momentum=0.9, affine=True)
        self.inception_5a_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_5a_3x3 = nn.Conv2d(192, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_5a_3x3_bn = nn.BatchNorm2d(320, eps=1e-05, momentum=0.9, affine=True)
        self.inception_5a_relu_3x3 = nn.ReLU(inplace)
        self.inception_5a_double_3x3_reduce = nn.Conv2d(1056, 160, kernel_size=(1, 1), stride=(1, 1))
        self.inception_5a_double_3x3_reduce_bn = nn.BatchNorm2d(160, eps=1e-05, momentum=0.9, affine=True)
        self.inception_5a_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_5a_double_3x3_1 = nn.Conv2d(160, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_5a_double_3x3_1_bn = nn.BatchNorm2d(224, eps=1e-05, momentum=0.9, affine=True)
        self.inception_5a_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_5a_double_3x3_2 = nn.Conv2d(224, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_5a_double_3x3_2_bn = nn.BatchNorm2d(224, eps=1e-05, momentum=0.9, affine=True)
        self.inception_5a_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_5a_pool = nn.AvgPool2d(3, stride=1, padding=1, ceil_mode=True, count_include_pad=True)
        self.inception_5a_pool_proj = nn.Conv2d(1056, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_5a_pool_proj_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
        self.inception_5a_relu_pool_proj = nn.ReLU(inplace)
        self.inception_5b_1x1 = nn.Conv2d(1024, 352, kernel_size=(1, 1), stride=(1, 1))
        self.inception_5b_1x1_bn = nn.BatchNorm2d(352, eps=1e-05, momentum=0.9, affine=True)
        self.inception_5b_relu_1x1 = nn.ReLU(inplace)
        self.inception_5b_3x3_reduce = nn.Conv2d(1024, 192, kernel_size=(1, 1), stride=(1, 1))
        self.inception_5b_3x3_reduce_bn = nn.BatchNorm2d(192, eps=1e-05, momentum=0.9, affine=True)
        self.inception_5b_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_5b_3x3 = nn.Conv2d(192, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_5b_3x3_bn = nn.BatchNorm2d(320, eps=1e-05, momentum=0.9, affine=True)
        self.inception_5b_relu_3x3 = nn.ReLU(inplace)
        self.inception_5b_double_3x3_reduce = nn.Conv2d(1024, 192, kernel_size=(1, 1), stride=(1, 1))
        self.inception_5b_double_3x3_reduce_bn = nn.BatchNorm2d(192, eps=1e-05, momentum=0.9, affine=True)
        self.inception_5b_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_5b_double_3x3_1 = nn.Conv2d(192, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_5b_double_3x3_1_bn = nn.BatchNorm2d(224, eps=1e-05, momentum=0.9, affine=True)
        self.inception_5b_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_5b_double_3x3_2 = nn.Conv2d(224, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_5b_double_3x3_2_bn = nn.BatchNorm2d(224, eps=1e-05, momentum=0.9, affine=True)
        self.inception_5b_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_5b_pool = nn.MaxPool2d((3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), ceil_mode=True)
        self.inception_5b_pool_proj = nn.Conv2d(1024, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_5b_pool_proj_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
        self.inception_5b_relu_pool_proj = nn.ReLU(inplace)
        self.global_pool = nn.AvgPool2d(7, stride=1, padding=0, ceil_mode=True, count_include_pad=True)
        self.last_linear = nn.Linear(1024, num_classes)

    def features(self, input):
        conv1_7x7_s2_out = self.conv1_7x7_s2(input)
        conv1_7x7_s2_bn_out = self.conv1_7x7_s2_bn(conv1_7x7_s2_out)
        conv1_relu_7x7_out = self.conv1_relu_7x7(conv1_7x7_s2_bn_out)
        pool1_3x3_s2_out = self.pool1_3x3_s2(conv1_7x7_s2_bn_out)
        conv2_3x3_reduce_out = self.conv2_3x3_reduce(pool1_3x3_s2_out)
        conv2_3x3_reduce_bn_out = self.conv2_3x3_reduce_bn(conv2_3x3_reduce_out)
        conv2_relu_3x3_reduce_out = self.conv2_relu_3x3_reduce(conv2_3x3_reduce_bn_out)
        conv2_3x3_out = self.conv2_3x3(conv2_3x3_reduce_bn_out)
        conv2_3x3_bn_out = self.conv2_3x3_bn(conv2_3x3_out)
        conv2_relu_3x3_out = self.conv2_relu_3x3(conv2_3x3_bn_out)
        pool2_3x3_s2_out = self.pool2_3x3_s2(conv2_3x3_bn_out)
        inception_3a_1x1_out = self.inception_3a_1x1(pool2_3x3_s2_out)
        inception_3a_1x1_bn_out = self.inception_3a_1x1_bn(inception_3a_1x1_out)
        inception_3a_relu_1x1_out = self.inception_3a_relu_1x1(inception_3a_1x1_bn_out)
        inception_3a_3x3_reduce_out = self.inception_3a_3x3_reduce(pool2_3x3_s2_out)
        inception_3a_3x3_reduce_bn_out = self.inception_3a_3x3_reduce_bn(inception_3a_3x3_reduce_out)
        inception_3a_relu_3x3_reduce_out = self.inception_3a_relu_3x3_reduce(inception_3a_3x3_reduce_bn_out)
        inception_3a_3x3_out = self.inception_3a_3x3(inception_3a_3x3_reduce_bn_out)
        inception_3a_3x3_bn_out = self.inception_3a_3x3_bn(inception_3a_3x3_out)
        inception_3a_relu_3x3_out = self.inception_3a_relu_3x3(inception_3a_3x3_bn_out)
        inception_3a_double_3x3_reduce_out = self.inception_3a_double_3x3_reduce(pool2_3x3_s2_out)
        inception_3a_double_3x3_reduce_bn_out = self.inception_3a_double_3x3_reduce_bn(inception_3a_double_3x3_reduce_out)
        inception_3a_relu_double_3x3_reduce_out = self.inception_3a_relu_double_3x3_reduce(inception_3a_double_3x3_reduce_bn_out)
        inception_3a_double_3x3_1_out = self.inception_3a_double_3x3_1(inception_3a_double_3x3_reduce_bn_out)
        inception_3a_double_3x3_1_bn_out = self.inception_3a_double_3x3_1_bn(inception_3a_double_3x3_1_out)
        inception_3a_relu_double_3x3_1_out = self.inception_3a_relu_double_3x3_1(inception_3a_double_3x3_1_bn_out)
        inception_3a_double_3x3_2_out = self.inception_3a_double_3x3_2(inception_3a_double_3x3_1_bn_out)
        inception_3a_double_3x3_2_bn_out = self.inception_3a_double_3x3_2_bn(inception_3a_double_3x3_2_out)
        inception_3a_relu_double_3x3_2_out = self.inception_3a_relu_double_3x3_2(inception_3a_double_3x3_2_bn_out)
        inception_3a_pool_out = self.inception_3a_pool(pool2_3x3_s2_out)
        inception_3a_pool_proj_out = self.inception_3a_pool_proj(inception_3a_pool_out)
        inception_3a_pool_proj_bn_out = self.inception_3a_pool_proj_bn(inception_3a_pool_proj_out)
        inception_3a_relu_pool_proj_out = self.inception_3a_relu_pool_proj(inception_3a_pool_proj_bn_out)
        inception_3a_output_out = torch.cat([inception_3a_1x1_bn_out, inception_3a_3x3_bn_out, inception_3a_double_3x3_2_bn_out, inception_3a_pool_proj_bn_out], 1)
        inception_3b_1x1_out = self.inception_3b_1x1(inception_3a_output_out)
        inception_3b_1x1_bn_out = self.inception_3b_1x1_bn(inception_3b_1x1_out)
        inception_3b_relu_1x1_out = self.inception_3b_relu_1x1(inception_3b_1x1_bn_out)
        inception_3b_3x3_reduce_out = self.inception_3b_3x3_reduce(inception_3a_output_out)
        inception_3b_3x3_reduce_bn_out = self.inception_3b_3x3_reduce_bn(inception_3b_3x3_reduce_out)
        inception_3b_relu_3x3_reduce_out = self.inception_3b_relu_3x3_reduce(inception_3b_3x3_reduce_bn_out)
        inception_3b_3x3_out = self.inception_3b_3x3(inception_3b_3x3_reduce_bn_out)
        inception_3b_3x3_bn_out = self.inception_3b_3x3_bn(inception_3b_3x3_out)
        inception_3b_relu_3x3_out = self.inception_3b_relu_3x3(inception_3b_3x3_bn_out)
        inception_3b_double_3x3_reduce_out = self.inception_3b_double_3x3_reduce(inception_3a_output_out)
        inception_3b_double_3x3_reduce_bn_out = self.inception_3b_double_3x3_reduce_bn(inception_3b_double_3x3_reduce_out)
        inception_3b_relu_double_3x3_reduce_out = self.inception_3b_relu_double_3x3_reduce(inception_3b_double_3x3_reduce_bn_out)
        inception_3b_double_3x3_1_out = self.inception_3b_double_3x3_1(inception_3b_double_3x3_reduce_bn_out)
        inception_3b_double_3x3_1_bn_out = self.inception_3b_double_3x3_1_bn(inception_3b_double_3x3_1_out)
        inception_3b_relu_double_3x3_1_out = self.inception_3b_relu_double_3x3_1(inception_3b_double_3x3_1_bn_out)
        inception_3b_double_3x3_2_out = self.inception_3b_double_3x3_2(inception_3b_double_3x3_1_bn_out)
        inception_3b_double_3x3_2_bn_out = self.inception_3b_double_3x3_2_bn(inception_3b_double_3x3_2_out)
        inception_3b_relu_double_3x3_2_out = self.inception_3b_relu_double_3x3_2(inception_3b_double_3x3_2_bn_out)
        inception_3b_pool_out = self.inception_3b_pool(inception_3a_output_out)
        inception_3b_pool_proj_out = self.inception_3b_pool_proj(inception_3b_pool_out)
        inception_3b_pool_proj_bn_out = self.inception_3b_pool_proj_bn(inception_3b_pool_proj_out)
        inception_3b_relu_pool_proj_out = self.inception_3b_relu_pool_proj(inception_3b_pool_proj_bn_out)
        inception_3b_output_out = torch.cat([inception_3b_1x1_bn_out, inception_3b_3x3_bn_out, inception_3b_double_3x3_2_bn_out, inception_3b_pool_proj_bn_out], 1)
        inception_3c_3x3_reduce_out = self.inception_3c_3x3_reduce(inception_3b_output_out)
        inception_3c_3x3_reduce_bn_out = self.inception_3c_3x3_reduce_bn(inception_3c_3x3_reduce_out)
        inception_3c_relu_3x3_reduce_out = self.inception_3c_relu_3x3_reduce(inception_3c_3x3_reduce_bn_out)
        inception_3c_3x3_out = self.inception_3c_3x3(inception_3c_3x3_reduce_bn_out)
        inception_3c_3x3_bn_out = self.inception_3c_3x3_bn(inception_3c_3x3_out)
        inception_3c_relu_3x3_out = self.inception_3c_relu_3x3(inception_3c_3x3_bn_out)
        inception_3c_double_3x3_reduce_out = self.inception_3c_double_3x3_reduce(inception_3b_output_out)
        inception_3c_double_3x3_reduce_bn_out = self.inception_3c_double_3x3_reduce_bn(inception_3c_double_3x3_reduce_out)
        inception_3c_relu_double_3x3_reduce_out = self.inception_3c_relu_double_3x3_reduce(inception_3c_double_3x3_reduce_bn_out)
        inception_3c_double_3x3_1_out = self.inception_3c_double_3x3_1(inception_3c_double_3x3_reduce_bn_out)
        inception_3c_double_3x3_1_bn_out = self.inception_3c_double_3x3_1_bn(inception_3c_double_3x3_1_out)
        inception_3c_relu_double_3x3_1_out = self.inception_3c_relu_double_3x3_1(inception_3c_double_3x3_1_bn_out)
        inception_3c_double_3x3_2_out = self.inception_3c_double_3x3_2(inception_3c_double_3x3_1_bn_out)
        inception_3c_double_3x3_2_bn_out = self.inception_3c_double_3x3_2_bn(inception_3c_double_3x3_2_out)
        inception_3c_relu_double_3x3_2_out = self.inception_3c_relu_double_3x3_2(inception_3c_double_3x3_2_bn_out)
        inception_3c_pool_out = self.inception_3c_pool(inception_3b_output_out)
        inception_3c_output_out = torch.cat([inception_3c_3x3_bn_out, inception_3c_double_3x3_2_bn_out, inception_3c_pool_out], 1)
        inception_4a_1x1_out = self.inception_4a_1x1(inception_3c_output_out)
        inception_4a_1x1_bn_out = self.inception_4a_1x1_bn(inception_4a_1x1_out)
        inception_4a_relu_1x1_out = self.inception_4a_relu_1x1(inception_4a_1x1_bn_out)
        inception_4a_3x3_reduce_out = self.inception_4a_3x3_reduce(inception_3c_output_out)
        inception_4a_3x3_reduce_bn_out = self.inception_4a_3x3_reduce_bn(inception_4a_3x3_reduce_out)
        inception_4a_relu_3x3_reduce_out = self.inception_4a_relu_3x3_reduce(inception_4a_3x3_reduce_bn_out)
        inception_4a_3x3_out = self.inception_4a_3x3(inception_4a_3x3_reduce_bn_out)
        inception_4a_3x3_bn_out = self.inception_4a_3x3_bn(inception_4a_3x3_out)
        inception_4a_relu_3x3_out = self.inception_4a_relu_3x3(inception_4a_3x3_bn_out)
        inception_4a_double_3x3_reduce_out = self.inception_4a_double_3x3_reduce(inception_3c_output_out)
        inception_4a_double_3x3_reduce_bn_out = self.inception_4a_double_3x3_reduce_bn(inception_4a_double_3x3_reduce_out)
        inception_4a_relu_double_3x3_reduce_out = self.inception_4a_relu_double_3x3_reduce(inception_4a_double_3x3_reduce_bn_out)
        inception_4a_double_3x3_1_out = self.inception_4a_double_3x3_1(inception_4a_double_3x3_reduce_bn_out)
        inception_4a_double_3x3_1_bn_out = self.inception_4a_double_3x3_1_bn(inception_4a_double_3x3_1_out)
        inception_4a_relu_double_3x3_1_out = self.inception_4a_relu_double_3x3_1(inception_4a_double_3x3_1_bn_out)
        inception_4a_double_3x3_2_out = self.inception_4a_double_3x3_2(inception_4a_double_3x3_1_bn_out)
        inception_4a_double_3x3_2_bn_out = self.inception_4a_double_3x3_2_bn(inception_4a_double_3x3_2_out)
        inception_4a_relu_double_3x3_2_out = self.inception_4a_relu_double_3x3_2(inception_4a_double_3x3_2_bn_out)
        inception_4a_pool_out = self.inception_4a_pool(inception_3c_output_out)
        inception_4a_pool_proj_out = self.inception_4a_pool_proj(inception_4a_pool_out)
        inception_4a_pool_proj_bn_out = self.inception_4a_pool_proj_bn(inception_4a_pool_proj_out)
        inception_4a_relu_pool_proj_out = self.inception_4a_relu_pool_proj(inception_4a_pool_proj_bn_out)
        inception_4a_output_out = torch.cat([inception_4a_1x1_bn_out, inception_4a_3x3_bn_out, inception_4a_double_3x3_2_bn_out, inception_4a_pool_proj_bn_out], 1)
        inception_4b_1x1_out = self.inception_4b_1x1(inception_4a_output_out)
        inception_4b_1x1_bn_out = self.inception_4b_1x1_bn(inception_4b_1x1_out)
        inception_4b_relu_1x1_out = self.inception_4b_relu_1x1(inception_4b_1x1_bn_out)
        inception_4b_3x3_reduce_out = self.inception_4b_3x3_reduce(inception_4a_output_out)
        inception_4b_3x3_reduce_bn_out = self.inception_4b_3x3_reduce_bn(inception_4b_3x3_reduce_out)
        inception_4b_relu_3x3_reduce_out = self.inception_4b_relu_3x3_reduce(inception_4b_3x3_reduce_bn_out)
        inception_4b_3x3_out = self.inception_4b_3x3(inception_4b_3x3_reduce_bn_out)
        inception_4b_3x3_bn_out = self.inception_4b_3x3_bn(inception_4b_3x3_out)
        inception_4b_relu_3x3_out = self.inception_4b_relu_3x3(inception_4b_3x3_bn_out)
        inception_4b_double_3x3_reduce_out = self.inception_4b_double_3x3_reduce(inception_4a_output_out)
        inception_4b_double_3x3_reduce_bn_out = self.inception_4b_double_3x3_reduce_bn(inception_4b_double_3x3_reduce_out)
        inception_4b_relu_double_3x3_reduce_out = self.inception_4b_relu_double_3x3_reduce(inception_4b_double_3x3_reduce_bn_out)
        inception_4b_double_3x3_1_out = self.inception_4b_double_3x3_1(inception_4b_double_3x3_reduce_bn_out)
        inception_4b_double_3x3_1_bn_out = self.inception_4b_double_3x3_1_bn(inception_4b_double_3x3_1_out)
        inception_4b_relu_double_3x3_1_out = self.inception_4b_relu_double_3x3_1(inception_4b_double_3x3_1_bn_out)
        inception_4b_double_3x3_2_out = self.inception_4b_double_3x3_2(inception_4b_double_3x3_1_bn_out)
        inception_4b_double_3x3_2_bn_out = self.inception_4b_double_3x3_2_bn(inception_4b_double_3x3_2_out)
        inception_4b_relu_double_3x3_2_out = self.inception_4b_relu_double_3x3_2(inception_4b_double_3x3_2_bn_out)
        inception_4b_pool_out = self.inception_4b_pool(inception_4a_output_out)
        inception_4b_pool_proj_out = self.inception_4b_pool_proj(inception_4b_pool_out)
        inception_4b_pool_proj_bn_out = self.inception_4b_pool_proj_bn(inception_4b_pool_proj_out)
        inception_4b_relu_pool_proj_out = self.inception_4b_relu_pool_proj(inception_4b_pool_proj_bn_out)
        inception_4b_output_out = torch.cat([inception_4b_1x1_bn_out, inception_4b_3x3_bn_out, inception_4b_double_3x3_2_bn_out, inception_4b_pool_proj_bn_out], 1)
        inception_4c_1x1_out = self.inception_4c_1x1(inception_4b_output_out)
        inception_4c_1x1_bn_out = self.inception_4c_1x1_bn(inception_4c_1x1_out)
        inception_4c_relu_1x1_out = self.inception_4c_relu_1x1(inception_4c_1x1_bn_out)
        inception_4c_3x3_reduce_out = self.inception_4c_3x3_reduce(inception_4b_output_out)
        inception_4c_3x3_reduce_bn_out = self.inception_4c_3x3_reduce_bn(inception_4c_3x3_reduce_out)
        inception_4c_relu_3x3_reduce_out = self.inception_4c_relu_3x3_reduce(inception_4c_3x3_reduce_bn_out)
        inception_4c_3x3_out = self.inception_4c_3x3(inception_4c_3x3_reduce_bn_out)
        inception_4c_3x3_bn_out = self.inception_4c_3x3_bn(inception_4c_3x3_out)
        inception_4c_relu_3x3_out = self.inception_4c_relu_3x3(inception_4c_3x3_bn_out)
        inception_4c_double_3x3_reduce_out = self.inception_4c_double_3x3_reduce(inception_4b_output_out)
        inception_4c_double_3x3_reduce_bn_out = self.inception_4c_double_3x3_reduce_bn(inception_4c_double_3x3_reduce_out)
        inception_4c_relu_double_3x3_reduce_out = self.inception_4c_relu_double_3x3_reduce(inception_4c_double_3x3_reduce_bn_out)
        inception_4c_double_3x3_1_out = self.inception_4c_double_3x3_1(inception_4c_double_3x3_reduce_bn_out)
        inception_4c_double_3x3_1_bn_out = self.inception_4c_double_3x3_1_bn(inception_4c_double_3x3_1_out)
        inception_4c_relu_double_3x3_1_out = self.inception_4c_relu_double_3x3_1(inception_4c_double_3x3_1_bn_out)
        inception_4c_double_3x3_2_out = self.inception_4c_double_3x3_2(inception_4c_double_3x3_1_bn_out)
        inception_4c_double_3x3_2_bn_out = self.inception_4c_double_3x3_2_bn(inception_4c_double_3x3_2_out)
        inception_4c_relu_double_3x3_2_out = self.inception_4c_relu_double_3x3_2(inception_4c_double_3x3_2_bn_out)
        inception_4c_pool_out = self.inception_4c_pool(inception_4b_output_out)
        inception_4c_pool_proj_out = self.inception_4c_pool_proj(inception_4c_pool_out)
        inception_4c_pool_proj_bn_out = self.inception_4c_pool_proj_bn(inception_4c_pool_proj_out)
        inception_4c_relu_pool_proj_out = self.inception_4c_relu_pool_proj(inception_4c_pool_proj_bn_out)
        inception_4c_output_out = torch.cat([inception_4c_1x1_bn_out, inception_4c_3x3_bn_out, inception_4c_double_3x3_2_bn_out, inception_4c_pool_proj_bn_out], 1)
        inception_4d_1x1_out = self.inception_4d_1x1(inception_4c_output_out)
        inception_4d_1x1_bn_out = self.inception_4d_1x1_bn(inception_4d_1x1_out)
        inception_4d_relu_1x1_out = self.inception_4d_relu_1x1(inception_4d_1x1_bn_out)
        inception_4d_3x3_reduce_out = self.inception_4d_3x3_reduce(inception_4c_output_out)
        inception_4d_3x3_reduce_bn_out = self.inception_4d_3x3_reduce_bn(inception_4d_3x3_reduce_out)
        inception_4d_relu_3x3_reduce_out = self.inception_4d_relu_3x3_reduce(inception_4d_3x3_reduce_bn_out)
        inception_4d_3x3_out = self.inception_4d_3x3(inception_4d_3x3_reduce_bn_out)
        inception_4d_3x3_bn_out = self.inception_4d_3x3_bn(inception_4d_3x3_out)
        inception_4d_relu_3x3_out = self.inception_4d_relu_3x3(inception_4d_3x3_bn_out)
        inception_4d_double_3x3_reduce_out = self.inception_4d_double_3x3_reduce(inception_4c_output_out)
        inception_4d_double_3x3_reduce_bn_out = self.inception_4d_double_3x3_reduce_bn(inception_4d_double_3x3_reduce_out)
        inception_4d_relu_double_3x3_reduce_out = self.inception_4d_relu_double_3x3_reduce(inception_4d_double_3x3_reduce_bn_out)
        inception_4d_double_3x3_1_out = self.inception_4d_double_3x3_1(inception_4d_double_3x3_reduce_bn_out)
        inception_4d_double_3x3_1_bn_out = self.inception_4d_double_3x3_1_bn(inception_4d_double_3x3_1_out)
        inception_4d_relu_double_3x3_1_out = self.inception_4d_relu_double_3x3_1(inception_4d_double_3x3_1_bn_out)
        inception_4d_double_3x3_2_out = self.inception_4d_double_3x3_2(inception_4d_double_3x3_1_bn_out)
        inception_4d_double_3x3_2_bn_out = self.inception_4d_double_3x3_2_bn(inception_4d_double_3x3_2_out)
        inception_4d_relu_double_3x3_2_out = self.inception_4d_relu_double_3x3_2(inception_4d_double_3x3_2_bn_out)
        inception_4d_pool_out = self.inception_4d_pool(inception_4c_output_out)
        inception_4d_pool_proj_out = self.inception_4d_pool_proj(inception_4d_pool_out)
        inception_4d_pool_proj_bn_out = self.inception_4d_pool_proj_bn(inception_4d_pool_proj_out)
        inception_4d_relu_pool_proj_out = self.inception_4d_relu_pool_proj(inception_4d_pool_proj_bn_out)
        inception_4d_output_out = torch.cat([inception_4d_1x1_bn_out, inception_4d_3x3_bn_out, inception_4d_double_3x3_2_bn_out, inception_4d_pool_proj_bn_out], 1)
        inception_4e_3x3_reduce_out = self.inception_4e_3x3_reduce(inception_4d_output_out)
        inception_4e_3x3_reduce_bn_out = self.inception_4e_3x3_reduce_bn(inception_4e_3x3_reduce_out)
        inception_4e_relu_3x3_reduce_out = self.inception_4e_relu_3x3_reduce(inception_4e_3x3_reduce_bn_out)
        inception_4e_3x3_out = self.inception_4e_3x3(inception_4e_3x3_reduce_bn_out)
        inception_4e_3x3_bn_out = self.inception_4e_3x3_bn(inception_4e_3x3_out)
        inception_4e_relu_3x3_out = self.inception_4e_relu_3x3(inception_4e_3x3_bn_out)
        inception_4e_double_3x3_reduce_out = self.inception_4e_double_3x3_reduce(inception_4d_output_out)
        inception_4e_double_3x3_reduce_bn_out = self.inception_4e_double_3x3_reduce_bn(inception_4e_double_3x3_reduce_out)
        inception_4e_relu_double_3x3_reduce_out = self.inception_4e_relu_double_3x3_reduce(inception_4e_double_3x3_reduce_bn_out)
        inception_4e_double_3x3_1_out = self.inception_4e_double_3x3_1(inception_4e_double_3x3_reduce_bn_out)
        inception_4e_double_3x3_1_bn_out = self.inception_4e_double_3x3_1_bn(inception_4e_double_3x3_1_out)
        inception_4e_relu_double_3x3_1_out = self.inception_4e_relu_double_3x3_1(inception_4e_double_3x3_1_bn_out)
        inception_4e_double_3x3_2_out = self.inception_4e_double_3x3_2(inception_4e_double_3x3_1_bn_out)
        inception_4e_double_3x3_2_bn_out = self.inception_4e_double_3x3_2_bn(inception_4e_double_3x3_2_out)
        inception_4e_relu_double_3x3_2_out = self.inception_4e_relu_double_3x3_2(inception_4e_double_3x3_2_bn_out)
        inception_4e_pool_out = self.inception_4e_pool(inception_4d_output_out)
        inception_4e_output_out = torch.cat([inception_4e_3x3_bn_out, inception_4e_double_3x3_2_bn_out, inception_4e_pool_out], 1)
        inception_5a_1x1_out = self.inception_5a_1x1(inception_4e_output_out)
        inception_5a_1x1_bn_out = self.inception_5a_1x1_bn(inception_5a_1x1_out)
        inception_5a_relu_1x1_out = self.inception_5a_relu_1x1(inception_5a_1x1_bn_out)
        inception_5a_3x3_reduce_out = self.inception_5a_3x3_reduce(inception_4e_output_out)
        inception_5a_3x3_reduce_bn_out = self.inception_5a_3x3_reduce_bn(inception_5a_3x3_reduce_out)
        inception_5a_relu_3x3_reduce_out = self.inception_5a_relu_3x3_reduce(inception_5a_3x3_reduce_bn_out)
        inception_5a_3x3_out = self.inception_5a_3x3(inception_5a_3x3_reduce_bn_out)
        inception_5a_3x3_bn_out = self.inception_5a_3x3_bn(inception_5a_3x3_out)
        inception_5a_relu_3x3_out = self.inception_5a_relu_3x3(inception_5a_3x3_bn_out)
        inception_5a_double_3x3_reduce_out = self.inception_5a_double_3x3_reduce(inception_4e_output_out)
        inception_5a_double_3x3_reduce_bn_out = self.inception_5a_double_3x3_reduce_bn(inception_5a_double_3x3_reduce_out)
        inception_5a_relu_double_3x3_reduce_out = self.inception_5a_relu_double_3x3_reduce(inception_5a_double_3x3_reduce_bn_out)
        inception_5a_double_3x3_1_out = self.inception_5a_double_3x3_1(inception_5a_double_3x3_reduce_bn_out)
        inception_5a_double_3x3_1_bn_out = self.inception_5a_double_3x3_1_bn(inception_5a_double_3x3_1_out)
        inception_5a_relu_double_3x3_1_out = self.inception_5a_relu_double_3x3_1(inception_5a_double_3x3_1_bn_out)
        inception_5a_double_3x3_2_out = self.inception_5a_double_3x3_2(inception_5a_double_3x3_1_bn_out)
        inception_5a_double_3x3_2_bn_out = self.inception_5a_double_3x3_2_bn(inception_5a_double_3x3_2_out)
        inception_5a_relu_double_3x3_2_out = self.inception_5a_relu_double_3x3_2(inception_5a_double_3x3_2_bn_out)
        inception_5a_pool_out = self.inception_5a_pool(inception_4e_output_out)
        inception_5a_pool_proj_out = self.inception_5a_pool_proj(inception_5a_pool_out)
        inception_5a_pool_proj_bn_out = self.inception_5a_pool_proj_bn(inception_5a_pool_proj_out)
        inception_5a_relu_pool_proj_out = self.inception_5a_relu_pool_proj(inception_5a_pool_proj_bn_out)
        inception_5a_output_out = torch.cat([inception_5a_1x1_bn_out, inception_5a_3x3_bn_out, inception_5a_double_3x3_2_bn_out, inception_5a_pool_proj_bn_out], 1)
        inception_5b_1x1_out = self.inception_5b_1x1(inception_5a_output_out)
        inception_5b_1x1_bn_out = self.inception_5b_1x1_bn(inception_5b_1x1_out)
        inception_5b_relu_1x1_out = self.inception_5b_relu_1x1(inception_5b_1x1_bn_out)
        inception_5b_3x3_reduce_out = self.inception_5b_3x3_reduce(inception_5a_output_out)
        inception_5b_3x3_reduce_bn_out = self.inception_5b_3x3_reduce_bn(inception_5b_3x3_reduce_out)
        inception_5b_relu_3x3_reduce_out = self.inception_5b_relu_3x3_reduce(inception_5b_3x3_reduce_bn_out)
        inception_5b_3x3_out = self.inception_5b_3x3(inception_5b_3x3_reduce_bn_out)
        inception_5b_3x3_bn_out = self.inception_5b_3x3_bn(inception_5b_3x3_out)
        inception_5b_relu_3x3_out = self.inception_5b_relu_3x3(inception_5b_3x3_bn_out)
        inception_5b_double_3x3_reduce_out = self.inception_5b_double_3x3_reduce(inception_5a_output_out)
        inception_5b_double_3x3_reduce_bn_out = self.inception_5b_double_3x3_reduce_bn(inception_5b_double_3x3_reduce_out)
        inception_5b_relu_double_3x3_reduce_out = self.inception_5b_relu_double_3x3_reduce(inception_5b_double_3x3_reduce_bn_out)
        inception_5b_double_3x3_1_out = self.inception_5b_double_3x3_1(inception_5b_double_3x3_reduce_bn_out)
        inception_5b_double_3x3_1_bn_out = self.inception_5b_double_3x3_1_bn(inception_5b_double_3x3_1_out)
        inception_5b_relu_double_3x3_1_out = self.inception_5b_relu_double_3x3_1(inception_5b_double_3x3_1_bn_out)
        inception_5b_double_3x3_2_out = self.inception_5b_double_3x3_2(inception_5b_double_3x3_1_bn_out)
        inception_5b_double_3x3_2_bn_out = self.inception_5b_double_3x3_2_bn(inception_5b_double_3x3_2_out)
        inception_5b_relu_double_3x3_2_out = self.inception_5b_relu_double_3x3_2(inception_5b_double_3x3_2_bn_out)
        inception_5b_pool_out = self.inception_5b_pool(inception_5a_output_out)
        inception_5b_pool_proj_out = self.inception_5b_pool_proj(inception_5b_pool_out)
        inception_5b_pool_proj_bn_out = self.inception_5b_pool_proj_bn(inception_5b_pool_proj_out)
        inception_5b_relu_pool_proj_out = self.inception_5b_relu_pool_proj(inception_5b_pool_proj_bn_out)
        inception_5b_output_out = torch.cat([inception_5b_1x1_bn_out, inception_5b_3x3_bn_out, inception_5b_double_3x3_2_bn_out, inception_5b_pool_proj_bn_out], 1)
        return inception_5b_output_out

    def logits(self, features):
        x = self.global_pool(features)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x


def conv3x3(in_channels, out_channels, stride=1, padding=1, bias=True, groups=1):
    """3x3 convolution with padding
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=bias, groups=groups)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    """
    Base class for bottlenecks that implements `forward()` method.
    """

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.se_module(out) + residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=True), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        self.conv1_input = x.clone()
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class CatBnAct(nn.Module):

    def __init__(self, in_chs, activation_fn=nn.ReLU(inplace=True)):
        super(CatBnAct, self).__init__()
        self.bn = nn.BatchNorm2d(in_chs, eps=0.001)
        self.act = activation_fn

    def forward(self, x):
        x = torch.cat(x, dim=1) if isinstance(x, tuple) else x
        return self.act(self.bn(x))


class BnActConv2d(nn.Module):

    def __init__(self, in_chs, out_chs, kernel_size, stride, padding=0, groups=1, activation_fn=nn.ReLU(inplace=True)):
        super(BnActConv2d, self).__init__()
        self.bn = nn.BatchNorm2d(in_chs, eps=0.001)
        self.act = activation_fn
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride, padding, groups=groups, bias=False)

    def forward(self, x):
        return self.conv(self.act(self.bn(x)))


class InputBlock(nn.Module):

    def __init__(self, num_init_features, kernel_size=7, padding=3, activation_fn=nn.ReLU(inplace=True)):
        super(InputBlock, self).__init__()
        self.conv = nn.Conv2d(3, num_init_features, kernel_size=kernel_size, stride=2, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(num_init_features, eps=0.001)
        self.act = activation_fn
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        return x


class DualPathBlock(nn.Module):

    def __init__(self, in_chs, num_1x1_a, num_3x3_b, num_1x1_c, inc, groups, block_type='normal', b=False):
        super(DualPathBlock, self).__init__()
        self.num_1x1_c = num_1x1_c
        self.inc = inc
        self.b = b
        if block_type is 'proj':
            self.key_stride = 1
            self.has_proj = True
        elif block_type is 'down':
            self.key_stride = 2
            self.has_proj = True
        else:
            assert block_type is 'normal'
            self.key_stride = 1
            self.has_proj = False
        if self.has_proj:
            if self.key_stride == 2:
                self.c1x1_w_s2 = BnActConv2d(in_chs=in_chs, out_chs=num_1x1_c + 2 * inc, kernel_size=1, stride=2)
            else:
                self.c1x1_w_s1 = BnActConv2d(in_chs=in_chs, out_chs=num_1x1_c + 2 * inc, kernel_size=1, stride=1)
        self.c1x1_a = BnActConv2d(in_chs=in_chs, out_chs=num_1x1_a, kernel_size=1, stride=1)
        self.c3x3_b = BnActConv2d(in_chs=num_1x1_a, out_chs=num_3x3_b, kernel_size=3, stride=self.key_stride, padding=1, groups=groups)
        if b:
            self.c1x1_c = CatBnAct(in_chs=num_3x3_b)
            self.c1x1_c1 = nn.Conv2d(num_3x3_b, num_1x1_c, kernel_size=1, bias=False)
            self.c1x1_c2 = nn.Conv2d(num_3x3_b, inc, kernel_size=1, bias=False)
        else:
            self.c1x1_c = BnActConv2d(in_chs=num_3x3_b, out_chs=num_1x1_c + inc, kernel_size=1, stride=1)

    def forward(self, x):
        x_in = torch.cat(x, dim=1) if isinstance(x, tuple) else x
        if self.has_proj:
            if self.key_stride == 2:
                x_s = self.c1x1_w_s2(x_in)
            else:
                x_s = self.c1x1_w_s1(x_in)
            x_s1 = x_s[:, :self.num_1x1_c, :, :]
            x_s2 = x_s[:, self.num_1x1_c:, :, :]
        else:
            x_s1 = x[0]
            x_s2 = x[1]
        x_in = self.c1x1_a(x_in)
        x_in = self.c3x3_b(x_in)
        if self.b:
            x_in = self.c1x1_c(x_in)
            out1 = self.c1x1_c1(x_in)
            out2 = self.c1x1_c2(x_in)
        else:
            x_in = self.c1x1_c(x_in)
            out1 = x_in[:, :self.num_1x1_c, :, :]
            out2 = x_in[:, self.num_1x1_c:, :, :]
        resid = x_s1 + out1
        dense = torch.cat([x_s2, out2], dim=1)
        return resid, dense


def adaptive_avgmax_pool2d(x, pool_type='avg', padding=0, count_include_pad=False):
    """Selectable global pooling function with dynamic input kernel size
    """
    if pool_type == 'avgmaxc':
        x = torch.cat([F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=padding, count_include_pad=count_include_pad), F.max_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=padding)], dim=1)
    elif pool_type == 'avgmax':
        x_avg = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=padding, count_include_pad=count_include_pad)
        x_max = F.max_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=padding)
        x = 0.5 * (x_avg + x_max)
    elif pool_type == 'max':
        x = F.max_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=padding)
    else:
        if pool_type != 'avg':
            None
        x = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=padding, count_include_pad=count_include_pad)
    return x


class DPN(nn.Module):

    def __init__(self, small=False, num_init_features=64, k_r=96, groups=32, b=False, k_sec=(3, 4, 20, 3), inc_sec=(16, 32, 24, 128), num_classes=1000, test_time_pool=False):
        super(DPN, self).__init__()
        self.test_time_pool = test_time_pool
        self.b = b
        bw_factor = 1 if small else 4
        blocks = OrderedDict()
        if small:
            blocks['conv1_1'] = InputBlock(num_init_features, kernel_size=3, padding=1)
        else:
            blocks['conv1_1'] = InputBlock(num_init_features, kernel_size=7, padding=3)
        bw = 64 * bw_factor
        inc = inc_sec[0]
        r = k_r * bw // (64 * bw_factor)
        blocks['conv2_1'] = DualPathBlock(num_init_features, r, r, bw, inc, groups, 'proj', b)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[0] + 1):
            blocks['conv2_' + str(i)] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'normal', b)
            in_chs += inc
        bw = 128 * bw_factor
        inc = inc_sec[1]
        r = k_r * bw // (64 * bw_factor)
        blocks['conv3_1'] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'down', b)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[1] + 1):
            blocks['conv3_' + str(i)] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'normal', b)
            in_chs += inc
        bw = 256 * bw_factor
        inc = inc_sec[2]
        r = k_r * bw // (64 * bw_factor)
        blocks['conv4_1'] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'down', b)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[2] + 1):
            blocks['conv4_' + str(i)] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'normal', b)
            in_chs += inc
        bw = 512 * bw_factor
        inc = inc_sec[3]
        r = k_r * bw // (64 * bw_factor)
        blocks['conv5_1'] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'down', b)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[3] + 1):
            blocks['conv5_' + str(i)] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'normal', b)
            in_chs += inc
        blocks['conv5_bn_ac'] = CatBnAct(in_chs)
        self.features = nn.Sequential(blocks)
        self.classifier = nn.Conv2d(in_chs, num_classes, kernel_size=1, bias=True)

    def logits(self, features):
        if not self.training and self.test_time_pool:
            x = F.avg_pool2d(features, kernel_size=7, stride=1)
            out = self.classifier(x)
            out = adaptive_avgmax_pool2d(out, pool_type='avgmax')
        else:
            x = adaptive_avgmax_pool2d(features, pool_type='avg')
            out = self.classifier(x)
        return out.view(out.size(0), -1)

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x


def pooling_factor(pool_type='avg'):
    return 2 if pool_type == 'avgmaxc' else 1


class AdaptiveAvgMaxPool2d(torch.nn.Module):
    """Selectable global pooling layer with dynamic input kernel size
    """

    def __init__(self, output_size=1, pool_type='avg'):
        super(AdaptiveAvgMaxPool2d, self).__init__()
        self.output_size = output_size
        self.pool_type = pool_type
        if pool_type == 'avgmaxc' or pool_type == 'avgmax':
            self.pool = nn.ModuleList([nn.AdaptiveAvgPool2d(output_size), nn.AdaptiveMaxPool2d(output_size)])
        elif pool_type == 'max':
            self.pool = nn.AdaptiveMaxPool2d(output_size)
        else:
            if pool_type != 'avg':
                None
            self.pool = nn.AdaptiveAvgPool2d(output_size)

    def forward(self, x):
        if self.pool_type == 'avgmaxc':
            x = torch.cat([p(x) for p in self.pool], dim=1)
        elif self.pool_type == 'avgmax':
            x = 0.5 * torch.sum(torch.stack([p(x) for p in self.pool]), 0).squeeze(dim=0)
        else:
            x = self.pool(x)
        return x

    def factor(self):
        return pooling_factor(self.pool_type)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + 'output_size=' + str(self.output_size) + ', pool_type=' + self.pool_type + ')'


class FBResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        self.input_space = None
        self.input_size = 299, 299, 3
        self.mean = None
        self.std = None
        super(FBResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.last_linear = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=True), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def features(self, input):
        x = self.conv1(input)
        self.conv1_input = x.clone()
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def logits(self, features):
        x = self.avgpool(features)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x


class GoogLeNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(GoogLeNet, self).__init__()
        self.conv1_7x7_s2 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.conv1_relu_7x7 = nn.ReLU(inplace=True)
        self.pool1_3x3_s2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), ceil_mode=True)
        self.conv2_3x3_reduce = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
        self.conv2_relu_3x3_reduce = nn.ReLU(inplace=True)
        self.conv2_3x3 = nn.Conv2d(64, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2_relu_3x3 = nn.ReLU(inplace=True)
        self.pool2_3x3_s2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), ceil_mode=True)
        self.inception_3a_1x1 = nn.Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3a_5x5_reduce = nn.Conv2d(192, 16, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3a_3x3_reduce = nn.Conv2d(192, 96, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3a_pool_proj = nn.Conv2d(192, 32, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3a_5x5 = nn.Conv2d(16, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.inception_3a_3x3 = nn.Conv2d(96, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3b_3x3_reduce = nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3b_1x1 = nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3b_5x5_reduce = nn.Conv2d(256, 32, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3b_pool_proj = nn.Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3b_3x3 = nn.Conv2d(128, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3b_5x5 = nn.Conv2d(32, 96, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.inception_4a_1x1 = nn.Conv2d(480, 192, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4a_3x3_reduce = nn.Conv2d(480, 96, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4a_5x5_reduce = nn.Conv2d(480, 16, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4a_pool_proj = nn.Conv2d(480, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4a_3x3 = nn.Conv2d(96, 208, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4a_5x5 = nn.Conv2d(16, 48, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.inception_4b_5x5_reduce = nn.Conv2d(512, 24, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4b_1x1 = nn.Conv2d(512, 160, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4b_3x3_reduce = nn.Conv2d(512, 112, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4b_pool_proj = nn.Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4b_5x5 = nn.Conv2d(24, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.inception_4b_3x3 = nn.Conv2d(112, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4c_5x5_reduce = nn.Conv2d(512, 24, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4c_1x1 = nn.Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4c_3x3_reduce = nn.Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4c_pool_proj = nn.Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4c_5x5 = nn.Conv2d(24, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.inception_4c_3x3 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4d_3x3_reduce = nn.Conv2d(512, 144, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4d_1x1 = nn.Conv2d(512, 112, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4d_5x5_reduce = nn.Conv2d(512, 32, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4d_pool_proj = nn.Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4d_3x3 = nn.Conv2d(144, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4d_5x5 = nn.Conv2d(32, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.inception_4e_5x5_reduce = nn.Conv2d(528, 32, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4e_1x1 = nn.Conv2d(528, 256, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4e_3x3_reduce = nn.Conv2d(528, 160, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4e_pool_proj = nn.Conv2d(528, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4e_5x5 = nn.Conv2d(32, 128, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.inception_4e_3x3 = nn.Conv2d(160, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_5a_1x1 = nn.Conv2d(832, 256, kernel_size=(1, 1), stride=(1, 1))
        self.inception_5a_5x5_reduce = nn.Conv2d(832, 32, kernel_size=(1, 1), stride=(1, 1))
        self.inception_5a_3x3_reduce = nn.Conv2d(832, 160, kernel_size=(1, 1), stride=(1, 1))
        self.inception_5a_pool_proj = nn.Conv2d(832, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_5a_5x5 = nn.Conv2d(32, 128, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.inception_5a_3x3 = nn.Conv2d(160, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_5b_3x3_reduce = nn.Conv2d(832, 192, kernel_size=(1, 1), stride=(1, 1))
        self.inception_5b_5x5_reduce = nn.Conv2d(832, 48, kernel_size=(1, 1), stride=(1, 1))
        self.inception_5b_1x1 = nn.Conv2d(832, 384, kernel_size=(1, 1), stride=(1, 1))
        self.inception_5b_pool_proj = nn.Conv2d(832, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_5b_3x3 = nn.Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_5b_5x5 = nn.Conv2d(48, 128, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.loss3_classifier_1 = nn.Linear(1024, num_classes)

    def forward(self, x):
        conv1_7x7_s2 = self.conv1_relu_7x7(self.conv1_7x7_s2(x))
        pool1_3x3_s2 = self.pool1_3x3_s2(conv1_7x7_s2)
        pool1_norm1 = self.LRN(size=5, alpha=0.0001, beta=0.75)(pool1_3x3_s2)
        conv2_3x3_reduce = self.conv2_relu_3x3_reduce(self.conv2_3x3_reduce(pool1_norm1))
        conv2_3x3 = self.conv2_relu_3x3(self.conv2_3x3(conv2_3x3_reduce))
        conv2_norm2 = self.LRN(size=5, alpha=0.0001, beta=0.75)(conv2_3x3)
        pool2_3x3_s2 = self.pool2_3x3_s2(conv2_norm2)
        inception_3a_1x1 = F.relu(self.inception_3a_1x1(pool2_3x3_s2))
        inception_3a_3x3_reduce = F.relu(self.inception_3a_3x3_reduce(pool2_3x3_s2))
        inception_3a_3x3 = F.relu(self.inception_3a_3x3(inception_3a_3x3_reduce))
        inception_3a_5x5_reduce = F.relu(self.inception_3a_5x5_reduce(pool2_3x3_s2))
        inception_3a_5x5 = F.relu(self.inception_3a_5x5(inception_3a_5x5_reduce))
        inception_3a_pool = self.max_pool(pool2_3x3_s2)
        inception_3a_pool_proj = F.relu(self.inception_3a_pool_proj(inception_3a_pool))
        inception_3a_output = torch.cat((inception_3a_1x1, inception_3a_3x3, inception_3a_5x5, inception_3a_pool_proj), 1)
        inception_3b_1x1 = F.relu(self.inception_3b_1x1(inception_3a_output))
        inception_3b_3x3_reduce = F.relu(self.inception_3b_3x3_reduce(inception_3a_output))
        inception_3b_3x3 = F.relu(self.inception_3b_3x3(inception_3b_3x3_reduce))
        inception_3b_5x5_reduce = F.relu(self.inception_3b_5x5_reduce(inception_3a_output))
        inception_3b_5x5 = F.relu(self.inception_3b_5x5(inception_3b_5x5_reduce))
        inception_3b_pool = self.max_pool(inception_3a_output)
        inception_3b_pool_proj = F.relu(self.inception_3b_pool_proj(inception_3b_pool))
        inception_3b_output = torch.cat((inception_3b_1x1, inception_3b_3x3, inception_3b_5x5, inception_3b_pool_proj), 1)
        pool3_3x3_s2 = F.max_pool2d(inception_3b_output, kernel_size=(3, 3), stride=(2, 2), ceil_mode=True)
        inception_4a_1x1 = F.relu(self.inception_4a_1x1(pool3_3x3_s2))
        inception_4a_3x3_reduce = F.relu(self.inception_4a_3x3_reduce(pool3_3x3_s2))
        inception_4a_3x3 = F.relu(self.inception_4a_3x3(inception_4a_3x3_reduce))
        inception_4a_5x5_reduce = F.relu(self.inception_4a_5x5_reduce(pool3_3x3_s2))
        inception_4a_5x5 = F.relu(self.inception_4a_5x5(inception_4a_5x5_reduce))
        inception_4a_pool = self.max_pool(pool3_3x3_s2)
        inception_4a_pool_proj = F.relu(self.inception_4a_pool_proj(inception_4a_pool))
        inception_4a_output = torch.cat((inception_4a_1x1, inception_4a_3x3, inception_4a_5x5, inception_4a_pool_proj), 1)
        inception_4b_1x1 = F.relu(self.inception_4b_1x1(inception_4a_output))
        inception_4b_3x3_reduce = F.relu(self.inception_4b_3x3_reduce(inception_4a_output))
        inception_4b_3x3 = F.relu(self.inception_4b_3x3(inception_4b_3x3_reduce))
        inception_4b_5x5_reduce = F.relu(self.inception_4b_5x5_reduce(inception_4a_output))
        inception_4b_5x5 = F.relu(self.inception_4b_5x5(inception_4b_5x5_reduce))
        inception_4b_pool = self.max_pool(inception_4a_output)
        inception_4b_pool_proj = F.relu(self.inception_4b_pool_proj(inception_4b_pool))
        inception_4b_output = torch.cat((inception_4b_1x1, inception_4b_3x3, inception_4b_5x5, inception_4b_pool_proj), 1)
        inception_4c_1x1 = F.relu(self.inception_4c_1x1(inception_4b_output))
        inception_4c_3x3_reduce = F.relu(self.inception_4c_3x3_reduce(inception_4b_output))
        inception_4c_3x3 = F.relu(self.inception_4c_3x3(inception_4c_3x3_reduce))
        inception_4c_5x5_reduce = F.relu(self.inception_4c_5x5_reduce(inception_4b_output))
        inception_4c_5x5 = F.relu(self.inception_4c_5x5(inception_4c_5x5_reduce))
        inception_4c_pool = self.max_pool(inception_4b_output)
        inception_4c_pool_proj = F.relu(self.inception_4c_pool_proj(inception_4c_pool))
        inception_4c_output = torch.cat((inception_4c_1x1, inception_4c_3x3, inception_4c_5x5, inception_4c_pool_proj), 1)
        inception_4d_1x1 = F.relu(self.inception_4d_1x1(inception_4c_output))
        inception_4d_3x3_reduce = F.relu(self.inception_4d_3x3_reduce(inception_4c_output))
        inception_4d_3x3 = F.relu(self.inception_4d_3x3(inception_4d_3x3_reduce))
        inception_4d_5x5_reduce = F.relu(self.inception_4d_5x5_reduce(inception_4c_output))
        inception_4d_5x5 = F.relu(self.inception_4d_5x5(inception_4d_5x5_reduce))
        inception_4d_pool = self.max_pool(inception_4c_output)
        inception_4d_pool_proj = F.relu(self.inception_4d_pool_proj(inception_4d_pool))
        inception_4d_output = torch.cat((inception_4d_1x1, inception_4d_3x3, inception_4d_5x5, inception_4d_pool_proj), 1)
        inception_4e_1x1 = F.relu(self.inception_4e_1x1(inception_4d_output))
        inception_4e_3x3_reduce = F.relu(self.inception_4e_3x3_reduce(inception_4d_output))
        inception_4e_3x3 = F.relu(self.inception_4e_3x3(inception_4e_3x3_reduce))
        inception_4e_5x5_reduce = F.relu(self.inception_4e_5x5_reduce(inception_4d_output))
        inception_4e_5x5 = F.relu(self.inception_4e_5x5(inception_4e_5x5_reduce))
        inception_4e_pool = self.max_pool(inception_4d_output)
        inception_4e_pool_proj = F.relu(self.inception_4e_pool_proj(inception_4e_pool))
        inception_4e_output = torch.cat((inception_4e_1x1, inception_4e_3x3, inception_4e_5x5, inception_4e_pool_proj), 1)
        pool4_3x3_s2 = F.max_pool2d(inception_4e_output, kernel_size=(3, 3), stride=(2, 2), ceil_mode=True)
        inception_5a_1x1 = F.relu(self.inception_5a_1x1(pool4_3x3_s2))
        inception_5a_3x3_reduce = F.relu(self.inception_5a_3x3_reduce(pool4_3x3_s2))
        inception_5a_3x3 = F.relu(self.inception_5a_3x3(inception_5a_3x3_reduce))
        inception_5a_5x5_reduce = F.relu(self.inception_5a_5x5_reduce(pool4_3x3_s2))
        inception_5a_5x5 = F.relu(self.inception_5a_5x5(inception_5a_5x5_reduce))
        inception_5a_pool = self.max_pool(pool4_3x3_s2)
        inception_5a_pool_proj = F.relu(self.inception_5a_pool_proj(inception_5a_pool))
        inception_5a_output = torch.cat((inception_5a_1x1, inception_5a_3x3, inception_5a_5x5, inception_5a_pool_proj), 1)
        inception_5b_1x1 = F.relu(self.inception_5b_1x1(inception_5a_output))
        inception_5b_3x3_reduce = F.relu(self.inception_5b_3x3_reduce(inception_5a_output))
        inception_5b_3x3 = F.relu(self.inception_5b_3x3(inception_5b_3x3_reduce))
        inception_5b_5x5_reduce = F.relu(self.inception_5b_5x5_reduce(inception_5a_output))
        inception_5b_5x5 = F.relu(self.inception_5b_5x5(inception_5b_5x5_reduce))
        inception_5b_pool = self.max_pool(inception_5a_output)
        inception_5b_pool_proj = F.relu(self.inception_5b_pool_proj(inception_5b_pool))
        inception_5b_output = torch.cat((inception_5b_1x1, inception_5b_3x3, inception_5b_5x5, inception_5b_pool_proj), 1)
        pool5_7x7_s1 = F.avg_pool2d(inception_5b_output, kernel_size=(7, 7), stride=(1, 1), padding=0)
        pool5_drop_7x7_s1 = F.dropout(input=pool5_7x7_s1, p=0.5, training=self.training, inplace=True)
        loss3_classifier_0 = pool5_drop_7x7_s1.view(pool5_drop_7x7_s1.size(0), -1)
        loss3_classifier_1 = self.loss3_classifier_1(loss3_classifier_0)
        return loss3_classifier_1


    class LRN(nn.Module):

        def __init__(self, size=1, alpha=1.0, beta=0.75, ACROSS_CHANNELS=True):
            super(GoogLeNet.LRN, self).__init__()
            self.ACROSS_CHANNELS = ACROSS_CHANNELS
            if self.ACROSS_CHANNELS:
                self.average = nn.AvgPool3d(kernel_size=(size, 1, 1), stride=1, padding=(int((size - 1.0) / 2), 0, 0))
            else:
                self.average = nn.AvgPool2d(kernel_size=size, stride=1, padding=int((size - 1.0) / 2))
            self.alpha = alpha
            self.beta = beta

        def forward(self, x):
            if self.ACROSS_CHANNELS:
                div = x.pow(2).unsqueeze(1)
                div = self.average(div).squeeze(1)
                div = div.mul(self.alpha).add(1.0).pow(self.beta)
            else:
                div = x.pow(2)
                div = self.average(div)
                div = div.mul(self.alpha).add(1.0).pow(self.beta)
            x = x.div(div)
            return x


class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes, eps=0.001, momentum=0.1, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Mixed_5b(nn.Module):

    def __init__(self):
        super(Mixed_5b, self).__init__()
        self.branch0 = BasicConv2d(192, 96, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(192, 48, kernel_size=1, stride=1), BasicConv2d(48, 64, kernel_size=5, stride=1, padding=2))
        self.branch2 = nn.Sequential(BasicConv2d(192, 64, kernel_size=1, stride=1), BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1), BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1))
        self.branch3 = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False), BasicConv2d(192, 64, kernel_size=1, stride=1))

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Block35(nn.Module):

    def __init__(self, scale=1.0):
        super(Block35, self).__init__()
        self.scale = scale
        self.branch0 = BasicConv2d(320, 32, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(320, 32, kernel_size=1, stride=1), BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1))
        self.branch2 = nn.Sequential(BasicConv2d(320, 32, kernel_size=1, stride=1), BasicConv2d(32, 48, kernel_size=3, stride=1, padding=1), BasicConv2d(48, 64, kernel_size=3, stride=1, padding=1))
        self.conv2d = nn.Conv2d(128, 320, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Mixed_6a(nn.Module):

    def __init__(self):
        super(Mixed_6a, self).__init__()
        self.branch0 = BasicConv2d(320, 384, kernel_size=3, stride=2)
        self.branch1 = nn.Sequential(BasicConv2d(320, 256, kernel_size=1, stride=1), BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1), BasicConv2d(256, 384, kernel_size=3, stride=2))
        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Block17(nn.Module):

    def __init__(self, scale=1.0):
        super(Block17, self).__init__()
        self.scale = scale
        self.branch0 = BasicConv2d(1088, 192, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(1088, 128, kernel_size=1, stride=1), BasicConv2d(128, 160, kernel_size=(1, 7), stride=1, padding=(0, 3)), BasicConv2d(160, 192, kernel_size=(7, 1), stride=1, padding=(3, 0)))
        self.conv2d = nn.Conv2d(384, 1088, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Mixed_7a(nn.Module):

    def __init__(self):
        super(Mixed_7a, self).__init__()
        self.branch0 = nn.Sequential(BasicConv2d(1088, 256, kernel_size=1, stride=1), BasicConv2d(256, 384, kernel_size=3, stride=2))
        self.branch1 = nn.Sequential(BasicConv2d(1088, 256, kernel_size=1, stride=1), BasicConv2d(256, 288, kernel_size=3, stride=2))
        self.branch2 = nn.Sequential(BasicConv2d(1088, 256, kernel_size=1, stride=1), BasicConv2d(256, 288, kernel_size=3, stride=1, padding=1), BasicConv2d(288, 320, kernel_size=3, stride=2))
        self.branch3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Block8(nn.Module):

    def __init__(self, scale=1.0, noReLU=False):
        super(Block8, self).__init__()
        self.scale = scale
        self.noReLU = noReLU
        self.branch0 = BasicConv2d(2080, 192, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(2080, 192, kernel_size=1, stride=1), BasicConv2d(192, 224, kernel_size=(1, 3), stride=1, padding=(0, 1)), BasicConv2d(224, 256, kernel_size=(3, 1), stride=1, padding=(1, 0)))
        self.conv2d = nn.Conv2d(448, 2080, kernel_size=1, stride=1)
        if not self.noReLU:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        if not self.noReLU:
            out = self.relu(out)
        return out


class InceptionResNetV2(nn.Module):

    def __init__(self, num_classes=1001):
        super(InceptionResNetV2, self).__init__()
        self.input_space = None
        self.input_size = 299, 299, 3
        self.mean = None
        self.std = None
        self.conv2d_1a = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1)
        self.conv2d_2b = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool_3a = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1, stride=1)
        self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3, stride=1)
        self.maxpool_5a = nn.MaxPool2d(3, stride=2)
        self.mixed_5b = Mixed_5b()
        self.repeat = nn.Sequential(Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17))
        self.mixed_6a = Mixed_6a()
        self.repeat_1 = nn.Sequential(Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1))
        self.mixed_7a = Mixed_7a()
        self.repeat_2 = nn.Sequential(Block8(scale=0.2), Block8(scale=0.2), Block8(scale=0.2), Block8(scale=0.2), Block8(scale=0.2), Block8(scale=0.2), Block8(scale=0.2), Block8(scale=0.2), Block8(scale=0.2))
        self.block8 = Block8(noReLU=True)
        self.conv2d_7b = BasicConv2d(2080, 1536, kernel_size=1, stride=1)
        self.avgpool_1a = nn.AvgPool2d(8, count_include_pad=False)
        self.last_linear = nn.Linear(1536, num_classes)

    def features(self, input):
        x = self.conv2d_1a(input)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.maxpool_5a(x)
        x = self.mixed_5b(x)
        x = self.repeat(x)
        x = self.mixed_6a(x)
        x = self.repeat_1(x)
        x = self.mixed_7a(x)
        x = self.repeat_2(x)
        x = self.block8(x)
        x = self.conv2d_7b(x)
        return x

    def logits(self, features):
        x = self.avgpool_1a(features)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x


class Mixed_3a(nn.Module):

    def __init__(self):
        super(Mixed_3a, self).__init__()
        self.maxpool = nn.MaxPool2d(3, stride=2)
        self.conv = BasicConv2d(64, 96, kernel_size=3, stride=2)

    def forward(self, x):
        x0 = self.maxpool(x)
        x1 = self.conv(x)
        out = torch.cat((x0, x1), 1)
        return out


class Mixed_4a(nn.Module):

    def __init__(self):
        super(Mixed_4a, self).__init__()
        self.branch0 = nn.Sequential(BasicConv2d(160, 64, kernel_size=1, stride=1), BasicConv2d(64, 96, kernel_size=3, stride=1))
        self.branch1 = nn.Sequential(BasicConv2d(160, 64, kernel_size=1, stride=1), BasicConv2d(64, 64, kernel_size=(1, 7), stride=1, padding=(0, 3)), BasicConv2d(64, 64, kernel_size=(7, 1), stride=1, padding=(3, 0)), BasicConv2d(64, 96, kernel_size=(3, 3), stride=1))

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        return out


class Mixed_5a(nn.Module):

    def __init__(self):
        super(Mixed_5a, self).__init__()
        self.conv = BasicConv2d(192, 192, kernel_size=3, stride=2)
        self.maxpool = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.conv(x)
        x1 = self.maxpool(x)
        out = torch.cat((x0, x1), 1)
        return out


class Inception_A(nn.Module):

    def __init__(self):
        super(Inception_A, self).__init__()
        self.branch0 = BasicConv2d(384, 96, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(384, 64, kernel_size=1, stride=1), BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1))
        self.branch2 = nn.Sequential(BasicConv2d(384, 64, kernel_size=1, stride=1), BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1), BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1))
        self.branch3 = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False), BasicConv2d(384, 96, kernel_size=1, stride=1))

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Reduction_A(nn.Module):

    def __init__(self):
        super(Reduction_A, self).__init__()
        self.branch0 = BasicConv2d(384, 384, kernel_size=3, stride=2)
        self.branch1 = nn.Sequential(BasicConv2d(384, 192, kernel_size=1, stride=1), BasicConv2d(192, 224, kernel_size=3, stride=1, padding=1), BasicConv2d(224, 256, kernel_size=3, stride=2))
        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Inception_B(nn.Module):

    def __init__(self):
        super(Inception_B, self).__init__()
        self.branch0 = BasicConv2d(1024, 384, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(1024, 192, kernel_size=1, stride=1), BasicConv2d(192, 224, kernel_size=(1, 7), stride=1, padding=(0, 3)), BasicConv2d(224, 256, kernel_size=(7, 1), stride=1, padding=(3, 0)))
        self.branch2 = nn.Sequential(BasicConv2d(1024, 192, kernel_size=1, stride=1), BasicConv2d(192, 192, kernel_size=(7, 1), stride=1, padding=(3, 0)), BasicConv2d(192, 224, kernel_size=(1, 7), stride=1, padding=(0, 3)), BasicConv2d(224, 224, kernel_size=(7, 1), stride=1, padding=(3, 0)), BasicConv2d(224, 256, kernel_size=(1, 7), stride=1, padding=(0, 3)))
        self.branch3 = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False), BasicConv2d(1024, 128, kernel_size=1, stride=1))

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Reduction_B(nn.Module):

    def __init__(self):
        super(Reduction_B, self).__init__()
        self.branch0 = nn.Sequential(BasicConv2d(1024, 192, kernel_size=1, stride=1), BasicConv2d(192, 192, kernel_size=3, stride=2))
        self.branch1 = nn.Sequential(BasicConv2d(1024, 256, kernel_size=1, stride=1), BasicConv2d(256, 256, kernel_size=(1, 7), stride=1, padding=(0, 3)), BasicConv2d(256, 320, kernel_size=(7, 1), stride=1, padding=(3, 0)), BasicConv2d(320, 320, kernel_size=3, stride=2))
        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Inception_C(nn.Module):

    def __init__(self):
        super(Inception_C, self).__init__()
        self.branch0 = BasicConv2d(1536, 256, kernel_size=1, stride=1)
        self.branch1_0 = BasicConv2d(1536, 384, kernel_size=1, stride=1)
        self.branch1_1a = BasicConv2d(384, 256, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.branch1_1b = BasicConv2d(384, 256, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.branch2_0 = BasicConv2d(1536, 384, kernel_size=1, stride=1)
        self.branch2_1 = BasicConv2d(384, 448, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.branch2_2 = BasicConv2d(448, 512, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.branch2_3a = BasicConv2d(512, 256, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.branch2_3b = BasicConv2d(512, 256, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.branch3 = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False), BasicConv2d(1536, 256, kernel_size=1, stride=1))

    def forward(self, x):
        x0 = self.branch0(x)
        x1_0 = self.branch1_0(x)
        x1_1a = self.branch1_1a(x1_0)
        x1_1b = self.branch1_1b(x1_0)
        x1 = torch.cat((x1_1a, x1_1b), 1)
        x2_0 = self.branch2_0(x)
        x2_1 = self.branch2_1(x2_0)
        x2_2 = self.branch2_2(x2_1)
        x2_3a = self.branch2_3a(x2_2)
        x2_3b = self.branch2_3b(x2_2)
        x2 = torch.cat((x2_3a, x2_3b), 1)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class InceptionV4(nn.Module):

    def __init__(self, num_classes=1001):
        super(InceptionV4, self).__init__()
        self.input_space = None
        self.input_size = 299, 299, 3
        self.mean = None
        self.std = None
        self.features = nn.Sequential(BasicConv2d(3, 32, kernel_size=3, stride=2), BasicConv2d(32, 32, kernel_size=3, stride=1), BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1), Mixed_3a(), Mixed_4a(), Mixed_5a(), Inception_A(), Inception_A(), Inception_A(), Inception_A(), Reduction_A(), Inception_B(), Inception_B(), Inception_B(), Inception_B(), Inception_B(), Inception_B(), Inception_B(), Reduction_B(), Inception_C(), Inception_C(), Inception_C())
        self.avg_pool = nn.AvgPool2d(8, count_include_pad=False)
        self.last_linear = nn.Linear(1536, num_classes)

    def logits(self, features):
        x = self.avg_pool(features)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x


class MobileNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(MobileNet, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.BatchNorm2d(oup), nn.ReLU(inplace=True))

        def conv_dw(inp, oup, stride):
            return nn.Sequential(nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False), nn.BatchNorm2d(inp), nn.ReLU(inplace=True), nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup), nn.ReLU(inplace=True))
        self.model = nn.Sequential(conv_bn(3, 32, 2), conv_dw(32, 64, 1), conv_dw(64, 128, 2), conv_dw(128, 128, 1), conv_dw(128, 256, 2), conv_dw(256, 256, 1), conv_dw(256, 512, 2), conv_dw(512, 512, 1), conv_dw(512, 512, 1), conv_dw(512, 512, 1), conv_dw(512, 512, 1), conv_dw(512, 512, 1), conv_dw(512, 1024, 2), conv_dw(1024, 1024, 1), nn.AvgPool2d(7))
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x


class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and inp == oup
        self.conv = nn.Sequential(nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False), nn.BatchNorm2d(inp * expand_ratio), nn.ReLU6(inplace=True), nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False), nn.BatchNorm2d(inp * expand_ratio), nn.ReLU6(inplace=True), nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup))

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


def conv_1x1_bn(inp, oup):
    return nn.Sequential(nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup), nn.ReLU6(inplace=True))


def conv_bn(inp, oup, stride):
    return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.BatchNorm2d(oup), nn.ReLU6(inplace=True))


class MobileNetV2(nn.Module):

    def __init__(self, num_classes=1000, width_mult=1.0):
        super(MobileNetV2, self).__init__()
        self.interverted_residual_setting = [[1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2], [6, 64, 4, 2], [6, 96, 3, 1], [6, 160, 3, 2], [6, 320, 1, 1]]
        input_channel = int(32 * width_mult)
        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        self.features = [conv_bn(3, input_channel, 2)]
        for t, c, n, s in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(InvertedResidual(input_channel, output_channel, s, t))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, t))
                input_channel = output_channel
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        self.features.append(nn.AvgPool2d(7))
        self.features = nn.Sequential(*self.features)
        self.classifier = nn.Sequential(nn.Dropout(), nn.Linear(self.last_channel, num_classes))

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.last_channel)
        x = self.classifier(x)
        return x


class MaxPoolPad(nn.Module):

    def __init__(self):
        super(MaxPoolPad, self).__init__()
        self.pad = nn.ZeroPad2d((1, 0, 1, 0))
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        x = self.pad(x)
        x = self.pool(x)
        x = x[:, :, 1:, 1:].contiguous()
        return x


class AvgPoolPad(nn.Module):

    def __init__(self, stride=2, padding=1):
        super(AvgPoolPad, self).__init__()
        self.pad = nn.ZeroPad2d((1, 0, 1, 0))
        self.pool = nn.AvgPool2d(3, stride=stride, padding=padding, count_include_pad=False)

    def forward(self, x):
        x = self.pad(x)
        x = self.pool(x)
        x = x[:, :, 1:, 1:].contiguous()
        return x


class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class BranchSeparables(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, name=None, bias=False):
        super(BranchSeparables, self).__init__()
        self.relu = nn.ReLU()
        self.separable_1 = SeparableConv2d(in_channels, in_channels, kernel_size, stride, padding, bias=bias)
        self.bn_sep_1 = nn.BatchNorm2d(in_channels, eps=0.001, momentum=0.1, affine=True)
        self.relu1 = nn.ReLU()
        self.separable_2 = SeparableConv2d(in_channels, out_channels, kernel_size, 1, padding, bias=bias)
        self.bn_sep_2 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True)
        self.name = name

    def forward(self, x):
        x = self.relu(x)
        if self.name == 'specific':
            x = nn.ZeroPad2d((1, 0, 1, 0))(x)
        x = self.separable_1(x)
        if self.name == 'specific':
            x = x[:, :, 1:, 1:].contiguous()
        x = self.bn_sep_1(x)
        x = self.relu1(x)
        x = self.separable_2(x)
        x = self.bn_sep_2(x)
        return x


class BranchSeparablesStem(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super(BranchSeparablesStem, self).__init__()
        self.relu = nn.ReLU()
        self.separable_1 = SeparableConv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn_sep_1 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True)
        self.relu1 = nn.ReLU()
        self.separable_2 = SeparableConv2d(out_channels, out_channels, kernel_size, 1, padding, bias=bias)
        self.bn_sep_2 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True)

    def forward(self, x):
        x = self.relu(x)
        x = self.separable_1(x)
        x = self.bn_sep_1(x)
        x = self.relu1(x)
        x = self.separable_2(x)
        x = self.bn_sep_2(x)
        return x


class BranchSeparablesReduction(BranchSeparables):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, z_padding=1, bias=False):
        BranchSeparables.__init__(self, in_channels, out_channels, kernel_size, stride, padding, bias)
        self.padding = nn.ZeroPad2d((z_padding, 0, z_padding, 0))

    def forward(self, x):
        x = self.relu(x)
        x = self.padding(x)
        x = self.separable_1(x)
        x = x[:, :, 1:, 1:].contiguous()
        x = self.bn_sep_1(x)
        x = self.relu1(x)
        x = self.separable_2(x)
        x = self.bn_sep_2(x)
        return x


class CellStem0(nn.Module):

    def __init__(self, stem_filters, num_filters=42):
        super(CellStem0, self).__init__()
        self.num_filters = num_filters
        self.stem_filters = stem_filters
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(self.stem_filters, self.num_filters, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(self.num_filters, eps=0.001, momentum=0.1, affine=True))
        self.comb_iter_0_left = BranchSeparables(self.num_filters, self.num_filters, 5, 2, 2)
        self.comb_iter_0_right = BranchSeparablesStem(self.stem_filters, self.num_filters, 7, 2, 3, bias=False)
        self.comb_iter_1_left = nn.MaxPool2d(3, stride=2, padding=1)
        self.comb_iter_1_right = BranchSeparablesStem(self.stem_filters, self.num_filters, 7, 2, 3, bias=False)
        self.comb_iter_2_left = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
        self.comb_iter_2_right = BranchSeparablesStem(self.stem_filters, self.num_filters, 5, 2, 2, bias=False)
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
        self.comb_iter_4_left = BranchSeparables(self.num_filters, self.num_filters, 3, 1, 1, bias=False)
        self.comb_iter_4_right = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        x1 = self.conv_1x1(x)
        x_comb_iter_0_left = self.comb_iter_0_left(x1)
        x_comb_iter_0_right = self.comb_iter_0_right(x)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
        x_comb_iter_1_left = self.comb_iter_1_left(x1)
        x_comb_iter_1_right = self.comb_iter_1_right(x)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
        x_comb_iter_2_left = self.comb_iter_2_left(x1)
        x_comb_iter_2_right = self.comb_iter_2_right(x)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right
        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1
        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x1)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right
        x_out = torch.cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class CellStem1(nn.Module):

    def __init__(self, stem_filters, num_filters):
        super(CellStem1, self).__init__()
        self.num_filters = num_filters
        self.stem_filters = stem_filters
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(2 * self.num_filters, self.num_filters, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(self.num_filters, eps=0.001, momentum=0.1, affine=True))
        self.relu = nn.ReLU()
        self.path_1 = nn.Sequential()
        self.path_1.add_module('avgpool', nn.AvgPool2d(1, stride=2, count_include_pad=False))
        self.path_1.add_module('conv', nn.Conv2d(self.stem_filters, self.num_filters // 2, 1, stride=1, bias=False))
        self.path_2 = nn.ModuleList()
        self.path_2.add_module('pad', nn.ZeroPad2d((0, 1, 0, 1)))
        self.path_2.add_module('avgpool', nn.AvgPool2d(1, stride=2, count_include_pad=False))
        self.path_2.add_module('conv', nn.Conv2d(self.stem_filters, self.num_filters // 2, 1, stride=1, bias=False))
        self.final_path_bn = nn.BatchNorm2d(self.num_filters, eps=0.001, momentum=0.1, affine=True)
        self.comb_iter_0_left = BranchSeparables(self.num_filters, self.num_filters, 5, 2, 2, name='specific', bias=False)
        self.comb_iter_0_right = BranchSeparables(self.num_filters, self.num_filters, 7, 2, 3, name='specific', bias=False)
        self.comb_iter_1_left = MaxPoolPad()
        self.comb_iter_1_right = BranchSeparables(self.num_filters, self.num_filters, 7, 2, 3, name='specific', bias=False)
        self.comb_iter_2_left = AvgPoolPad()
        self.comb_iter_2_right = BranchSeparables(self.num_filters, self.num_filters, 5, 2, 2, name='specific', bias=False)
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
        self.comb_iter_4_left = BranchSeparables(self.num_filters, self.num_filters, 3, 1, 1, name='specific', bias=False)
        self.comb_iter_4_right = MaxPoolPad()

    def forward(self, x_conv0, x_stem_0):
        x_left = self.conv_1x1(x_stem_0)
        x_relu = self.relu(x_conv0)
        x_path1 = self.path_1(x_relu)
        x_path2 = self.path_2.pad(x_relu)
        x_path2 = x_path2[:, :, 1:, 1:]
        x_path2 = self.path_2.avgpool(x_path2)
        x_path2 = self.path_2.conv(x_path2)
        x_right = self.final_path_bn(torch.cat([x_path1, x_path2], 1))
        x_comb_iter_0_left = self.comb_iter_0_left(x_left)
        x_comb_iter_0_right = self.comb_iter_0_right(x_right)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
        x_comb_iter_1_left = self.comb_iter_1_left(x_left)
        x_comb_iter_1_right = self.comb_iter_1_right(x_right)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
        x_comb_iter_2_left = self.comb_iter_2_left(x_left)
        x_comb_iter_2_right = self.comb_iter_2_right(x_right)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right
        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1
        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x_left)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right
        x_out = torch.cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class FirstCell(nn.Module):

    def __init__(self, in_channels_left, out_channels_left, in_channels_right, out_channels_right):
        super(FirstCell, self).__init__()
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(in_channels_right, out_channels_right, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(out_channels_right, eps=0.001, momentum=0.1, affine=True))
        self.relu = nn.ReLU()
        self.path_1 = nn.Sequential()
        self.path_1.add_module('avgpool', nn.AvgPool2d(1, stride=2, count_include_pad=False))
        self.path_1.add_module('conv', nn.Conv2d(in_channels_left, out_channels_left, 1, stride=1, bias=False))
        self.path_2 = nn.ModuleList()
        self.path_2.add_module('pad', nn.ZeroPad2d((0, 1, 0, 1)))
        self.path_2.add_module('avgpool', nn.AvgPool2d(1, stride=2, count_include_pad=False))
        self.path_2.add_module('conv', nn.Conv2d(in_channels_left, out_channels_left, 1, stride=1, bias=False))
        self.final_path_bn = nn.BatchNorm2d(out_channels_left * 2, eps=0.001, momentum=0.1, affine=True)
        self.comb_iter_0_left = BranchSeparables(out_channels_right, out_channels_right, 5, 1, 2, bias=False)
        self.comb_iter_0_right = BranchSeparables(out_channels_right, out_channels_right, 3, 1, 1, bias=False)
        self.comb_iter_1_left = BranchSeparables(out_channels_right, out_channels_right, 5, 1, 2, bias=False)
        self.comb_iter_1_right = BranchSeparables(out_channels_right, out_channels_right, 3, 1, 1, bias=False)
        self.comb_iter_2_left = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
        self.comb_iter_3_left = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
        self.comb_iter_4_left = BranchSeparables(out_channels_right, out_channels_right, 3, 1, 1, bias=False)

    def forward(self, x, x_prev):
        x_relu = self.relu(x_prev)
        x_path1 = self.path_1(x_relu)
        x_path2 = self.path_2.pad(x_relu)
        x_path2 = x_path2[:, :, 1:, 1:]
        x_path2 = self.path_2.avgpool(x_path2)
        x_path2 = self.path_2.conv(x_path2)
        x_left = self.final_path_bn(torch.cat([x_path1, x_path2], 1))
        x_right = self.conv_1x1(x)
        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
        x_comb_iter_1_left = self.comb_iter_1_left(x_left)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2 = x_comb_iter_2_left + x_left
        x_comb_iter_3_left = self.comb_iter_3_left(x_left)
        x_comb_iter_3_right = self.comb_iter_3_right(x_left)
        x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right
        x_comb_iter_4_left = self.comb_iter_4_left(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_right
        x_out = torch.cat([x_left, x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class NormalCell(nn.Module):

    def __init__(self, in_channels_left, out_channels_left, in_channels_right, out_channels_right):
        super(NormalCell, self).__init__()
        self.conv_prev_1x1 = nn.Sequential()
        self.conv_prev_1x1.add_module('relu', nn.ReLU())
        self.conv_prev_1x1.add_module('conv', nn.Conv2d(in_channels_left, out_channels_left, 1, stride=1, bias=False))
        self.conv_prev_1x1.add_module('bn', nn.BatchNorm2d(out_channels_left, eps=0.001, momentum=0.1, affine=True))
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(in_channels_right, out_channels_right, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(out_channels_right, eps=0.001, momentum=0.1, affine=True))
        self.comb_iter_0_left = BranchSeparables(out_channels_right, out_channels_right, 5, 1, 2, bias=False)
        self.comb_iter_0_right = BranchSeparables(out_channels_left, out_channels_left, 3, 1, 1, bias=False)
        self.comb_iter_1_left = BranchSeparables(out_channels_left, out_channels_left, 5, 1, 2, bias=False)
        self.comb_iter_1_right = BranchSeparables(out_channels_left, out_channels_left, 3, 1, 1, bias=False)
        self.comb_iter_2_left = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
        self.comb_iter_3_left = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
        self.comb_iter_4_left = BranchSeparables(out_channels_right, out_channels_right, 3, 1, 1, bias=False)

    def forward(self, x, x_prev):
        x_left = self.conv_prev_1x1(x_prev)
        x_right = self.conv_1x1(x)
        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
        x_comb_iter_1_left = self.comb_iter_1_left(x_left)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2 = x_comb_iter_2_left + x_left
        x_comb_iter_3_left = self.comb_iter_3_left(x_left)
        x_comb_iter_3_right = self.comb_iter_3_right(x_left)
        x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right
        x_comb_iter_4_left = self.comb_iter_4_left(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_right
        x_out = torch.cat([x_left, x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class ReductionCell0(nn.Module):

    def __init__(self, in_channels_left, out_channels_left, in_channels_right, out_channels_right):
        super(ReductionCell0, self).__init__()
        self.conv_prev_1x1 = nn.Sequential()
        self.conv_prev_1x1.add_module('relu', nn.ReLU())
        self.conv_prev_1x1.add_module('conv', nn.Conv2d(in_channels_left, out_channels_left, 1, stride=1, bias=False))
        self.conv_prev_1x1.add_module('bn', nn.BatchNorm2d(out_channels_left, eps=0.001, momentum=0.1, affine=True))
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(in_channels_right, out_channels_right, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(out_channels_right, eps=0.001, momentum=0.1, affine=True))
        self.comb_iter_0_left = BranchSeparablesReduction(out_channels_right, out_channels_right, 5, 2, 2, bias=False)
        self.comb_iter_0_right = BranchSeparablesReduction(out_channels_right, out_channels_right, 7, 2, 3, bias=False)
        self.comb_iter_1_left = MaxPoolPad()
        self.comb_iter_1_right = BranchSeparablesReduction(out_channels_right, out_channels_right, 7, 2, 3, bias=False)
        self.comb_iter_2_left = AvgPoolPad()
        self.comb_iter_2_right = BranchSeparablesReduction(out_channels_right, out_channels_right, 5, 2, 2, bias=False)
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
        self.comb_iter_4_left = BranchSeparablesReduction(out_channels_right, out_channels_right, 3, 1, 1, bias=False)
        self.comb_iter_4_right = MaxPoolPad()

    def forward(self, x, x_prev):
        x_left = self.conv_prev_1x1(x_prev)
        x_right = self.conv_1x1(x)
        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
        x_comb_iter_1_left = self.comb_iter_1_left(x_right)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2_right = self.comb_iter_2_right(x_left)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right
        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1
        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right
        x_out = torch.cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class ReductionCell1(nn.Module):

    def __init__(self, in_channels_left, out_channels_left, in_channels_right, out_channels_right):
        super(ReductionCell1, self).__init__()
        self.conv_prev_1x1 = nn.Sequential()
        self.conv_prev_1x1.add_module('relu', nn.ReLU())
        self.conv_prev_1x1.add_module('conv', nn.Conv2d(in_channels_left, out_channels_left, 1, stride=1, bias=False))
        self.conv_prev_1x1.add_module('bn', nn.BatchNorm2d(out_channels_left, eps=0.001, momentum=0.1, affine=True))
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(in_channels_right, out_channels_right, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(out_channels_right, eps=0.001, momentum=0.1, affine=True))
        self.comb_iter_0_left = BranchSeparables(out_channels_right, out_channels_right, 5, 2, 2, name='specific', bias=False)
        self.comb_iter_0_right = BranchSeparables(out_channels_right, out_channels_right, 7, 2, 3, name='specific', bias=False)
        self.comb_iter_1_left = MaxPoolPad()
        self.comb_iter_1_right = BranchSeparables(out_channels_right, out_channels_right, 7, 2, 3, name='specific', bias=False)
        self.comb_iter_2_left = AvgPoolPad()
        self.comb_iter_2_right = BranchSeparables(out_channels_right, out_channels_right, 5, 2, 2, name='specific', bias=False)
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
        self.comb_iter_4_left = BranchSeparables(out_channels_right, out_channels_right, 3, 1, 1, name='specific', bias=False)
        self.comb_iter_4_right = MaxPoolPad()

    def forward(self, x, x_prev):
        x_left = self.conv_prev_1x1(x_prev)
        x_right = self.conv_1x1(x)
        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
        x_comb_iter_1_left = self.comb_iter_1_left(x_right)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2_right = self.comb_iter_2_right(x_left)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right
        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1
        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right
        x_out = torch.cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class NASNetALarge(nn.Module):
    """NASNetALarge (6 @ 4032) """

    def __init__(self, num_classes=1001, stem_filters=96, penultimate_filters=4032, filters_multiplier=2):
        super(NASNetALarge, self).__init__()
        self.num_classes = num_classes
        self.stem_filters = stem_filters
        self.penultimate_filters = penultimate_filters
        self.filters_multiplier = filters_multiplier
        filters = self.penultimate_filters // 24
        self.conv0 = nn.Sequential()
        self.conv0.add_module('conv', nn.Conv2d(in_channels=3, out_channels=self.stem_filters, kernel_size=3, padding=0, stride=2, bias=False))
        self.conv0.add_module('bn', nn.BatchNorm2d(self.stem_filters, eps=0.001, momentum=0.1, affine=True))
        self.cell_stem_0 = CellStem0(self.stem_filters, num_filters=filters // filters_multiplier ** 2)
        self.cell_stem_1 = CellStem1(self.stem_filters, num_filters=filters // filters_multiplier)
        self.cell_0 = FirstCell(in_channels_left=filters, out_channels_left=filters // 2, in_channels_right=2 * filters, out_channels_right=filters)
        self.cell_1 = NormalCell(in_channels_left=2 * filters, out_channels_left=filters, in_channels_right=6 * filters, out_channels_right=filters)
        self.cell_2 = NormalCell(in_channels_left=6 * filters, out_channels_left=filters, in_channels_right=6 * filters, out_channels_right=filters)
        self.cell_3 = NormalCell(in_channels_left=6 * filters, out_channels_left=filters, in_channels_right=6 * filters, out_channels_right=filters)
        self.cell_4 = NormalCell(in_channels_left=6 * filters, out_channels_left=filters, in_channels_right=6 * filters, out_channels_right=filters)
        self.cell_5 = NormalCell(in_channels_left=6 * filters, out_channels_left=filters, in_channels_right=6 * filters, out_channels_right=filters)
        self.reduction_cell_0 = ReductionCell0(in_channels_left=6 * filters, out_channels_left=2 * filters, in_channels_right=6 * filters, out_channels_right=2 * filters)
        self.cell_6 = FirstCell(in_channels_left=6 * filters, out_channels_left=filters, in_channels_right=8 * filters, out_channels_right=2 * filters)
        self.cell_7 = NormalCell(in_channels_left=8 * filters, out_channels_left=2 * filters, in_channels_right=12 * filters, out_channels_right=2 * filters)
        self.cell_8 = NormalCell(in_channels_left=12 * filters, out_channels_left=2 * filters, in_channels_right=12 * filters, out_channels_right=2 * filters)
        self.cell_9 = NormalCell(in_channels_left=12 * filters, out_channels_left=2 * filters, in_channels_right=12 * filters, out_channels_right=2 * filters)
        self.cell_10 = NormalCell(in_channels_left=12 * filters, out_channels_left=2 * filters, in_channels_right=12 * filters, out_channels_right=2 * filters)
        self.cell_11 = NormalCell(in_channels_left=12 * filters, out_channels_left=2 * filters, in_channels_right=12 * filters, out_channels_right=2 * filters)
        self.reduction_cell_1 = ReductionCell1(in_channels_left=12 * filters, out_channels_left=4 * filters, in_channels_right=12 * filters, out_channels_right=4 * filters)
        self.cell_12 = FirstCell(in_channels_left=12 * filters, out_channels_left=2 * filters, in_channels_right=16 * filters, out_channels_right=4 * filters)
        self.cell_13 = NormalCell(in_channels_left=16 * filters, out_channels_left=4 * filters, in_channels_right=24 * filters, out_channels_right=4 * filters)
        self.cell_14 = NormalCell(in_channels_left=24 * filters, out_channels_left=4 * filters, in_channels_right=24 * filters, out_channels_right=4 * filters)
        self.cell_15 = NormalCell(in_channels_left=24 * filters, out_channels_left=4 * filters, in_channels_right=24 * filters, out_channels_right=4 * filters)
        self.cell_16 = NormalCell(in_channels_left=24 * filters, out_channels_left=4 * filters, in_channels_right=24 * filters, out_channels_right=4 * filters)
        self.cell_17 = NormalCell(in_channels_left=24 * filters, out_channels_left=4 * filters, in_channels_right=24 * filters, out_channels_right=4 * filters)
        self.relu = nn.ReLU()
        self.avg_pool = nn.AvgPool2d(11, stride=1, padding=0)
        self.dropout = nn.Dropout()
        self.last_linear = nn.Linear(24 * filters, self.num_classes)

    def features(self, input):
        x_conv0 = self.conv0(input)
        x_stem_0 = self.cell_stem_0(x_conv0)
        x_stem_1 = self.cell_stem_1(x_conv0, x_stem_0)
        x_cell_0 = self.cell_0(x_stem_1, x_stem_0)
        x_cell_1 = self.cell_1(x_cell_0, x_stem_1)
        x_cell_2 = self.cell_2(x_cell_1, x_cell_0)
        x_cell_3 = self.cell_3(x_cell_2, x_cell_1)
        x_cell_4 = self.cell_4(x_cell_3, x_cell_2)
        x_cell_5 = self.cell_5(x_cell_4, x_cell_3)
        x_reduction_cell_0 = self.reduction_cell_0(x_cell_5, x_cell_4)
        x_cell_6 = self.cell_6(x_reduction_cell_0, x_cell_4)
        x_cell_7 = self.cell_7(x_cell_6, x_reduction_cell_0)
        x_cell_8 = self.cell_8(x_cell_7, x_cell_6)
        x_cell_9 = self.cell_9(x_cell_8, x_cell_7)
        x_cell_10 = self.cell_10(x_cell_9, x_cell_8)
        x_cell_11 = self.cell_11(x_cell_10, x_cell_9)
        x_reduction_cell_1 = self.reduction_cell_1(x_cell_11, x_cell_10)
        x_cell_12 = self.cell_12(x_reduction_cell_1, x_cell_10)
        x_cell_13 = self.cell_13(x_cell_12, x_reduction_cell_1)
        x_cell_14 = self.cell_14(x_cell_13, x_cell_12)
        x_cell_15 = self.cell_15(x_cell_14, x_cell_13)
        x_cell_16 = self.cell_16(x_cell_15, x_cell_14)
        x_cell_17 = self.cell_17(x_cell_16, x_cell_15)
        return x_cell_17

    def logits(self, features):
        x = self.relu(features)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x


class NASNetAMobile(nn.Module):
    """NASNetAMobile (4 @ 1056) """

    def __init__(self, num_classes=1001, stem_filters=32, penultimate_filters=1056, filters_multiplier=2):
        super(NASNetAMobile, self).__init__()
        self.num_classes = num_classes
        self.stem_filters = stem_filters
        self.penultimate_filters = penultimate_filters
        self.filters_multiplier = filters_multiplier
        filters = self.penultimate_filters // 24
        self.conv0 = nn.Sequential()
        self.conv0.add_module('conv', nn.Conv2d(in_channels=3, out_channels=self.stem_filters, kernel_size=3, padding=0, stride=2, bias=False))
        self.conv0.add_module('bn', nn.BatchNorm2d(self.stem_filters, eps=0.001, momentum=0.1, affine=True))
        self.cell_stem_0 = CellStem0(self.stem_filters, num_filters=filters // filters_multiplier ** 2)
        self.cell_stem_1 = CellStem1(self.stem_filters, num_filters=filters // filters_multiplier)
        self.cell_0 = FirstCell(in_channels_left=filters, out_channels_left=filters // 2, in_channels_right=2 * filters, out_channels_right=filters)
        self.cell_1 = NormalCell(in_channels_left=2 * filters, out_channels_left=filters, in_channels_right=6 * filters, out_channels_right=filters)
        self.cell_2 = NormalCell(in_channels_left=6 * filters, out_channels_left=filters, in_channels_right=6 * filters, out_channels_right=filters)
        self.cell_3 = NormalCell(in_channels_left=6 * filters, out_channels_left=filters, in_channels_right=6 * filters, out_channels_right=filters)
        self.reduction_cell_0 = ReductionCell0(in_channels_left=6 * filters, out_channels_left=2 * filters, in_channels_right=6 * filters, out_channels_right=2 * filters)
        self.cell_6 = FirstCell(in_channels_left=6 * filters, out_channels_left=filters, in_channels_right=8 * filters, out_channels_right=2 * filters)
        self.cell_7 = NormalCell(in_channels_left=8 * filters, out_channels_left=2 * filters, in_channels_right=12 * filters, out_channels_right=2 * filters)
        self.cell_8 = NormalCell(in_channels_left=12 * filters, out_channels_left=2 * filters, in_channels_right=12 * filters, out_channels_right=2 * filters)
        self.cell_9 = NormalCell(in_channels_left=12 * filters, out_channels_left=2 * filters, in_channels_right=12 * filters, out_channels_right=2 * filters)
        self.reduction_cell_1 = ReductionCell1(in_channels_left=12 * filters, out_channels_left=4 * filters, in_channels_right=12 * filters, out_channels_right=4 * filters)
        self.cell_12 = FirstCell(in_channels_left=12 * filters, out_channels_left=2 * filters, in_channels_right=16 * filters, out_channels_right=4 * filters)
        self.cell_13 = NormalCell(in_channels_left=16 * filters, out_channels_left=4 * filters, in_channels_right=24 * filters, out_channels_right=4 * filters)
        self.cell_14 = NormalCell(in_channels_left=24 * filters, out_channels_left=4 * filters, in_channels_right=24 * filters, out_channels_right=4 * filters)
        self.cell_15 = NormalCell(in_channels_left=24 * filters, out_channels_left=4 * filters, in_channels_right=24 * filters, out_channels_right=4 * filters)
        self.relu = nn.ReLU()
        self.avg_pool = nn.AvgPool2d(7, stride=1, padding=0)
        self.dropout = nn.Dropout()
        self.last_linear = nn.Linear(24 * filters, self.num_classes)

    def features(self, input):
        x_conv0 = self.conv0(input)
        x_stem_0 = self.cell_stem_0(x_conv0)
        x_stem_1 = self.cell_stem_1(x_conv0, x_stem_0)
        x_cell_0 = self.cell_0(x_stem_1, x_stem_0)
        x_cell_1 = self.cell_1(x_cell_0, x_stem_1)
        x_cell_2 = self.cell_2(x_cell_1, x_cell_0)
        x_cell_3 = self.cell_3(x_cell_2, x_cell_1)
        x_reduction_cell_0 = self.reduction_cell_0(x_cell_3, x_cell_2)
        x_cell_6 = self.cell_6(x_reduction_cell_0, x_cell_3)
        x_cell_7 = self.cell_7(x_cell_6, x_reduction_cell_0)
        x_cell_8 = self.cell_8(x_cell_7, x_cell_6)
        x_cell_9 = self.cell_9(x_cell_8, x_cell_7)
        x_reduction_cell_1 = self.reduction_cell_1(x_cell_9, x_cell_8)
        x_cell_12 = self.cell_12(x_reduction_cell_1, x_cell_9)
        x_cell_13 = self.cell_13(x_cell_12, x_reduction_cell_1)
        x_cell_14 = self.cell_14(x_cell_13, x_cell_12)
        x_cell_15 = self.cell_15(x_cell_14, x_cell_13)
        return x_cell_15

    def logits(self, features):
        x = self.relu(features)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x


class LambdaBase(nn.Sequential):

    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input


class Lambda(LambdaBase):

    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))


class LambdaMap(LambdaBase):

    def forward(self, input):
        return list(map(self.lambda_func, self.forward_prepare(input)))


class LambdaReduce(LambdaBase):

    def forward(self, input):
        return reduce(self.lambda_func, self.forward_prepare(input))


resnext101_32x4d_features = nn.Sequential(nn.Conv2d(3, 64, (7, 7), (2, 2), (3, 3), 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d((3, 3), (2, 2), (1, 1)), nn.Sequential(nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(64, 128, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(), nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(128), nn.ReLU()), nn.Conv2d(128, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(256)), nn.Sequential(nn.Conv2d(64, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(256))), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(256, 128, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(), nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(128), nn.ReLU()), nn.Conv2d(128, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(256)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(256, 128, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(), nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(128), nn.ReLU()), nn.Conv2d(128, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(256)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU())), nn.Sequential(nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(256, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(), nn.Conv2d(256, 256, (3, 3), (2, 2), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(256), nn.ReLU()), nn.Conv2d(256, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512)), nn.Sequential(nn.Conv2d(256, 512, (1, 1), (2, 2), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512))), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(512, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(), nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(256), nn.ReLU()), nn.Conv2d(256, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(512, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(), nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(256), nn.ReLU()), nn.Conv2d(256, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(512, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(), nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(256), nn.ReLU()), nn.Conv2d(256, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU())), nn.Sequential(nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(512, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (2, 2), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(512), nn.ReLU()), nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), nn.Sequential(nn.Conv2d(512, 1024, (1, 1), (2, 2), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024))), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(512), nn.ReLU()), nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(512), nn.ReLU()), nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(512), nn.ReLU()), nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(512), nn.ReLU()), nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(512), nn.ReLU()), nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(512), nn.ReLU()), nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(512), nn.ReLU()), nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(512), nn.ReLU()), nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(512), nn.ReLU()), nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(512), nn.ReLU()), nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(512), nn.ReLU()), nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(512), nn.ReLU()), nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(512), nn.ReLU()), nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(512), nn.ReLU()), nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(512), nn.ReLU()), nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(512), nn.ReLU()), nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(512), nn.ReLU()), nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(512), nn.ReLU()), nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(512), nn.ReLU()), nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(512), nn.ReLU()), nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(512), nn.ReLU()), nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(512), nn.ReLU()), nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU())), nn.Sequential(nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024), nn.ReLU(), nn.Conv2d(1024, 1024, (3, 3), (2, 2), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(1024), nn.ReLU()), nn.Conv2d(1024, 2048, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(2048)), nn.Sequential(nn.Conv2d(1024, 2048, (1, 1), (2, 2), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(2048))), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(2048, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024), nn.ReLU(), nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(1024), nn.ReLU()), nn.Conv2d(1024, 2048, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(2048)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(2048, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024), nn.ReLU(), nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(1024), nn.ReLU()), nn.Conv2d(1024, 2048, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(2048)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU())))


class ResNeXt101_32x4d(nn.Module):

    def __init__(self, num_classes=1000):
        super(ResNeXt101_32x4d, self).__init__()
        self.num_classes = num_classes
        self.features = resnext101_32x4d_features
        self.avg_pool = nn.AvgPool2d((7, 7), (1, 1))
        self.last_linear = nn.Linear(2048, num_classes)

    def logits(self, input):
        x = self.avg_pool(input)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x


resnext101_64x4d_features = nn.Sequential(nn.Conv2d(3, 64, (7, 7), (2, 2), (3, 3), 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d((3, 3), (2, 2), (1, 1)), nn.Sequential(nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(64, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(), nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1), 1, 64, bias=False), nn.BatchNorm2d(256), nn.ReLU()), nn.Conv2d(256, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(256)), nn.Sequential(nn.Conv2d(64, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(256))), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(256, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(), nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1), 1, 64, bias=False), nn.BatchNorm2d(256), nn.ReLU()), nn.Conv2d(256, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(256)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(256, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(), nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1), 1, 64, bias=False), nn.BatchNorm2d(256), nn.ReLU()), nn.Conv2d(256, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(256)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU())), nn.Sequential(nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(256, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (2, 2), (1, 1), 1, 64, bias=False), nn.BatchNorm2d(512), nn.ReLU()), nn.Conv2d(512, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512)), nn.Sequential(nn.Conv2d(256, 512, (1, 1), (2, 2), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512))), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(512, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 64, bias=False), nn.BatchNorm2d(512), nn.ReLU()), nn.Conv2d(512, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(512, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 64, bias=False), nn.BatchNorm2d(512), nn.ReLU()), nn.Conv2d(512, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(512, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 64, bias=False), nn.BatchNorm2d(512), nn.ReLU()), nn.Conv2d(512, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU())), nn.Sequential(nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024), nn.ReLU(), nn.Conv2d(1024, 1024, (3, 3), (2, 2), (1, 1), 1, 64, bias=False), nn.BatchNorm2d(1024), nn.ReLU()), nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), nn.Sequential(nn.Conv2d(512, 1024, (1, 1), (2, 2), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024))), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024), nn.ReLU(), nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias=False), nn.BatchNorm2d(1024), nn.ReLU()), nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024), nn.ReLU(), nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias=False), nn.BatchNorm2d(1024), nn.ReLU()), nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024), nn.ReLU(), nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias=False), nn.BatchNorm2d(1024), nn.ReLU()), nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024), nn.ReLU(), nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias=False), nn.BatchNorm2d(1024), nn.ReLU()), nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024), nn.ReLU(), nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias=False), nn.BatchNorm2d(1024), nn.ReLU()), nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024), nn.ReLU(), nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias=False), nn.BatchNorm2d(1024), nn.ReLU()), nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024), nn.ReLU(), nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias=False), nn.BatchNorm2d(1024), nn.ReLU()), nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024), nn.ReLU(), nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias=False), nn.BatchNorm2d(1024), nn.ReLU()), nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024), nn.ReLU(), nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias=False), nn.BatchNorm2d(1024), nn.ReLU()), nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024), nn.ReLU(), nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias=False), nn.BatchNorm2d(1024), nn.ReLU()), nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024), nn.ReLU(), nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias=False), nn.BatchNorm2d(1024), nn.ReLU()), nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024), nn.ReLU(), nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias=False), nn.BatchNorm2d(1024), nn.ReLU()), nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024), nn.ReLU(), nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias=False), nn.BatchNorm2d(1024), nn.ReLU()), nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024), nn.ReLU(), nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias=False), nn.BatchNorm2d(1024), nn.ReLU()), nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024), nn.ReLU(), nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias=False), nn.BatchNorm2d(1024), nn.ReLU()), nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024), nn.ReLU(), nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias=False), nn.BatchNorm2d(1024), nn.ReLU()), nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024), nn.ReLU(), nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias=False), nn.BatchNorm2d(1024), nn.ReLU()), nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024), nn.ReLU(), nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias=False), nn.BatchNorm2d(1024), nn.ReLU()), nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024), nn.ReLU(), nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias=False), nn.BatchNorm2d(1024), nn.ReLU()), nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024), nn.ReLU(), nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias=False), nn.BatchNorm2d(1024), nn.ReLU()), nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024), nn.ReLU(), nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias=False), nn.BatchNorm2d(1024), nn.ReLU()), nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024), nn.ReLU(), nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias=False), nn.BatchNorm2d(1024), nn.ReLU()), nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU())), nn.Sequential(nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 2048, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(2048), nn.ReLU(), nn.Conv2d(2048, 2048, (3, 3), (2, 2), (1, 1), 1, 64, bias=False), nn.BatchNorm2d(2048), nn.ReLU()), nn.Conv2d(2048, 2048, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(2048)), nn.Sequential(nn.Conv2d(1024, 2048, (1, 1), (2, 2), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(2048))), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(2048, 2048, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(2048), nn.ReLU(), nn.Conv2d(2048, 2048, (3, 3), (1, 1), (1, 1), 1, 64, bias=False), nn.BatchNorm2d(2048), nn.ReLU()), nn.Conv2d(2048, 2048, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(2048)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(2048, 2048, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(2048), nn.ReLU(), nn.Conv2d(2048, 2048, (3, 3), (1, 1), (1, 1), 1, 64, bias=False), nn.BatchNorm2d(2048), nn.ReLU()), nn.Conv2d(2048, 2048, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(2048)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU())))


class ResNeXt101_64x4d(nn.Module):

    def __init__(self, num_classes=1000):
        super(ResNeXt101_64x4d, self).__init__()
        self.num_classes = num_classes
        self.features = resnext101_64x4d_features
        self.avg_pool = nn.AvgPool2d((7, 7), (1, 1))
        self.last_linear = nn.Linear(2048, num_classes)

    def logits(self, input):
        x = self.avg_pool(input)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x


class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class SEBottleneck(Bottleneck):
    """
    Bottleneck for SENet154.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1, downsample=None):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes * 2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes * 2)
        self.conv2 = nn.Conv2d(planes * 2, planes * 4, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes * 4)
        self.conv3 = nn.Conv2d(planes * 4, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNetBottleneck(Bottleneck):
    """
    ResNet bottleneck with a Squeeze-and-Excitation module. It follows Caffe
    implementation and uses `stride=stride` in `conv1` and not in `conv2`
    (the latter is used in the torchvision implementation of ResNet).
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1, downsample=None):
        super(SEResNetBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNeXtBottleneck(Bottleneck):
    """
    ResNeXt bottleneck type C with a Squeeze-and-Excitation module.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1, downsample=None, base_width=4.0):
        super(SEResNeXtBottleneck, self).__init__()
        width = int(math.floor(planes * (base_width / 64)) * groups)
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False, stride=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SENet(nn.Module):

    def __init__(self, block, layers, groups, reduction, dropout_p=0.2, inplanes=128, input_3x3=True, downsample_kernel_size=3, downsample_padding=1, num_classes=1000):
        """
        Parameters
        ----------
        block (nn.Module): Bottleneck class.
            - For SENet154: SEBottleneck
            - For SE-ResNet models: SEResNetBottleneck
            - For SE-ResNeXt models:  SEResNeXtBottleneck
        layers (list of ints): Number of residual blocks for 4 layers of the
            network (layer1...layer4).
        groups (int): Number of groups for the 3x3 convolution in each
            bottleneck block.
            - For SENet154: 64
            - For SE-ResNet models: 1
            - For SE-ResNeXt models:  32
        reduction (int): Reduction ratio for Squeeze-and-Excitation modules.
            - For all models: 16
        dropout_p (float or None): Drop probability for the Dropout layer.
            If `None` the Dropout layer is not used.
            - For SENet154: 0.2
            - For SE-ResNet models: None
            - For SE-ResNeXt models: None
        inplanes (int):  Number of input channels for layer1.
            - For SENet154: 128
            - For SE-ResNet models: 64
            - For SE-ResNeXt models: 64
        input_3x3 (bool): If `True`, use three 3x3 convolutions instead of
            a single 7x7 convolution in layer0.
            - For SENet154: True
            - For SE-ResNet models: False
            - For SE-ResNeXt models: False
        downsample_kernel_size (int): Kernel size for downsampling convolutions
            in layer2, layer3 and layer4.
            - For SENet154: 3
            - For SE-ResNet models: 1
            - For SE-ResNeXt models: 1
        downsample_padding (int): Padding for downsampling convolutions in
            layer2, layer3 and layer4.
            - For SENet154: 1
            - For SE-ResNet models: 0
            - For SE-ResNeXt models: 0
        num_classes (int): Number of outputs in `last_linear` layer.
            - For all models: 1000
        """
        super(SENet, self).__init__()
        self.inplanes = inplanes
        if input_3x3:
            layer0_modules = [('conv1', nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False)), ('bn1', nn.BatchNorm2d(64)), ('relu1', nn.ReLU(inplace=True)), ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False)), ('bn2', nn.BatchNorm2d(64)), ('relu2', nn.ReLU(inplace=True)), ('conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1, bias=False)), ('bn3', nn.BatchNorm2d(inplanes)), ('relu3', nn.ReLU(inplace=True))]
        else:
            layer0_modules = [('conv1', nn.Conv2d(3, inplanes, kernel_size=7, stride=2, padding=3, bias=False)), ('bn1', nn.BatchNorm2d(inplanes)), ('relu1', nn.ReLU(inplace=True))]
        layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2, ceil_mode=True)))
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        self.layer1 = self._make_layer(block, planes=64, blocks=layers[0], groups=groups, reduction=reduction, downsample_kernel_size=1, downsample_padding=0)
        self.layer2 = self._make_layer(block, planes=128, blocks=layers[1], stride=2, groups=groups, reduction=reduction, downsample_kernel_size=downsample_kernel_size, downsample_padding=downsample_padding)
        self.layer3 = self._make_layer(block, planes=256, blocks=layers[2], stride=2, groups=groups, reduction=reduction, downsample_kernel_size=downsample_kernel_size, downsample_padding=downsample_padding)
        self.layer4 = self._make_layer(block, planes=512, blocks=layers[3], stride=2, groups=groups, reduction=reduction, downsample_kernel_size=downsample_kernel_size, downsample_padding=downsample_padding)
        self.avg_pool = nn.AvgPool2d(7, stride=1)
        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        self.last_linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, groups, reduction, stride=1, downsample_kernel_size=1, downsample_padding=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=downsample_kernel_size, stride=stride, padding=downsample_padding, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, groups, reduction, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))
        return nn.Sequential(*layers)

    def features(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def logits(self, x):
        x = self.avg_pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x


def conv1x1(in_channels, out_channels, groups=1):
    """1x1 convolution with padding
    - Normal pointwise convolution When groups == 1
    - Grouped pointwise convolution when groups > 1
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=groups, stride=1)


class ShuffleUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=3, grouped_conv=True, combine='add'):
        super(ShuffleUnit, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grouped_conv = grouped_conv
        self.combine = combine
        self.groups = groups
        self.bottleneck_channels = self.out_channels // 4
        if self.combine == 'add':
            self.depthwise_stride = 1
            self._combine_func = self._add
        elif self.combine == 'concat':
            self.depthwise_stride = 2
            self._combine_func = self._concat
            self.out_channels -= self.in_channels
        else:
            raise ValueError('Cannot combine tensors with "{}"Only "add" and "concat" aresupported'.format(self.combine))
        self.first_1x1_groups = self.groups if grouped_conv else 1
        self.g_conv_1x1_compress = self._make_grouped_conv1x1(self.in_channels, self.bottleneck_channels, self.first_1x1_groups, batch_norm=True, relu=True)
        self.depthwise_conv3x3 = conv3x3(self.bottleneck_channels, self.bottleneck_channels, stride=self.depthwise_stride, groups=self.bottleneck_channels)
        self.bn_after_depthwise = nn.BatchNorm2d(self.bottleneck_channels)
        self.g_conv_1x1_expand = self._make_grouped_conv1x1(self.bottleneck_channels, self.out_channels, self.groups, batch_norm=True, relu=False)

    @staticmethod
    def _add(x, out):
        return x + out

    @staticmethod
    def _concat(x, out):
        return torch.cat((x, out), 1)

    def _make_grouped_conv1x1(self, in_channels, out_channels, groups, batch_norm=True, relu=False):
        modules = OrderedDict()
        conv = conv1x1(in_channels, out_channels, groups=groups)
        modules['conv1x1'] = conv
        if batch_norm:
            modules['batch_norm'] = nn.BatchNorm2d(out_channels)
        if relu:
            modules['relu'] = nn.ReLU()
        if len(modules) > 1:
            return nn.Sequential(modules)
        else:
            return conv

    def forward(self, x):
        residual = x
        if self.combine == 'concat':
            residual = F.avg_pool2d(residual, kernel_size=3, stride=2, padding=1)
        out = self.g_conv_1x1_compress(x)
        out = channel_shuffle(out, self.groups)
        out = self.depthwise_conv3x3(out)
        out = self.bn_after_depthwise(out)
        out = self.g_conv_1x1_expand(out)
        out = self._combine_func(residual, out)
        return F.relu(out)


class ShuffleNet(nn.Module):
    """ShuffleNet implementation.
    """

    def __init__(self, groups=8, in_channels=3, num_classes=1000):
        """ShuffleNet constructor.

        Arguments:
            groups (int, optional): number of groups to be used in grouped 
                1x1 convolutions in each ShuffleUnit. Default is 3 for best
                performance according to original paper.
            in_channels (int, optional): number of channels in the input tensor.
                Default is 3 for RGB image inputs.
            num_classes (int, optional): number of classes to predict. Default
                is 1000 for ImageNet.

        """
        super(ShuffleNet, self).__init__()
        self.groups = groups
        self.stage_repeats = [3, 7, 3]
        self.in_channels = in_channels
        self.num_classes = num_classes
        if groups == 1:
            self.stage_out_channels = [-1, 24, 144, 288, 567]
        elif groups == 2:
            self.stage_out_channels = [-1, 24, 200, 400, 800]
        elif groups == 3:
            self.stage_out_channels = [-1, 24, 240, 480, 960]
        elif groups == 4:
            self.stage_out_channels = [-1, 24, 272, 544, 1088]
        elif groups == 8:
            self.stage_out_channels = [-1, 24, 384, 768, 1536]
        else:
            raise ValueError("""{} groups is not supported for
                   1x1 Grouped Convolutions""".format(num_groups))
        self.conv1 = conv3x3(self.in_channels, self.stage_out_channels[1], stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stage2 = self._make_stage(2)
        self.stage3 = self._make_stage(3)
        self.stage4 = self._make_stage(4)
        num_inputs = self.stage_out_channels[-1]
        self.fc = nn.Linear(num_inputs, self.num_classes)

    def _make_stage(self, stage):
        modules = OrderedDict()
        stage_name = 'ShuffleUnit_Stage{}'.format(stage)
        grouped_conv = stage > 2
        first_module = ShuffleUnit(self.stage_out_channels[stage - 1], self.stage_out_channels[stage], groups=self.groups, grouped_conv=grouped_conv, combine='concat')
        modules[stage_name + '_0'] = first_module
        for i in range(self.stage_repeats[stage - 2]):
            name = stage_name + '_{}'.format(i + 1)
            module = ShuffleUnit(self.stage_out_channels[stage], self.stage_out_channels[stage], groups=self.groups, grouped_conv=True, combine='add')
            modules[name] = module
        return nn.Sequential(modules)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = F.avg_pool2d(x, x.data.size()[-2:])
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


class SpatialCrossMapLRN(nn.Module):

    def __init__(self, local_size=1, alpha=1.0, beta=0.75, k=1, ACROSS_CHANNELS=True):
        super(SpatialCrossMapLRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if ACROSS_CHANNELS:
            self.average = nn.AvgPool3d(kernel_size=(local_size, 1, 1), stride=1, padding=(int((local_size - 1.0) / 2), 0, 0))
        else:
            self.average = nn.AvgPool2d(kernel_size=local_size, stride=1, padding=int((local_size - 1.0) / 2))
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(self.k).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(self.k).pow(self.beta)
        x = x.div(div)
        return x


class VGGM(nn.Module):

    def __init__(self, num_classes=1000):
        super(VGGM, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(nn.Conv2d(3, 96, (7, 7), (2, 2)), nn.ReLU(), SpatialCrossMapLRN(5, 0.0005, 0.75, 2), nn.MaxPool2d((3, 3), (2, 2), (0, 0), ceil_mode=True), nn.Conv2d(96, 256, (5, 5), (2, 2), (1, 1)), nn.ReLU(), SpatialCrossMapLRN(5, 0.0005, 0.75, 2), nn.MaxPool2d((3, 3), (2, 2), (0, 0), ceil_mode=True), nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1)), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)), nn.ReLU(), nn.MaxPool2d((3, 3), (2, 2), (0, 0), ceil_mode=True))
        self.classif = nn.Sequential(nn.Linear(18432, 4096), nn.ReLU(), nn.Dropout(0.5), nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5), nn.Linear(4096, num_classes))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classif(x)
        return x


class WideResNet(nn.Module):

    def __init__(self, pooling):
        super(WideResNet, self).__init__()
        self.pooling = pooling
        self.params = params

    def forward(self, x):
        x = f(x, self.params, self.pooling)
        return x


class Block(nn.Module):

    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()
        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None
        self.relu = nn.ReLU(inplace=True)
        rep = []
        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters
        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))
        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)
        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)
        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp
        x += skip
        return x


class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """

    def __init__(self, num_classes=1000):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.block1 = Block(64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block(256, 728, 2, 2, start_with_relu=True, grow_first=True)
        self.block4 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block5 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block8 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block9 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block10 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block11 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block12 = Block(728, 1024, 2, 2, start_with_relu=True, grow_first=False)
        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)
        self.fc = nn.Linear(2048, num_classes)

    def features(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        return x

    def logits(self, features):
        x = self.relu(features)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AdaptiveAvgMaxPool2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (AvgPoolPad,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BasicConv2d,
     lambda: ([], {'in_planes': 4, 'out_planes': 4, 'kernel_size': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Block,
     lambda: ([], {'in_filters': 4, 'out_filters': 4, 'reps': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Block17,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1088, 64, 64])], {}),
     True),
    (Block35,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 320, 64, 64])], {}),
     True),
    (Block8,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 2080, 64, 64])], {}),
     True),
    (BnActConv2d,
     lambda: ([], {'in_chs': 4, 'out_chs': 4, 'kernel_size': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BranchSeparables,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (BranchSeparablesReduction,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BranchSeparablesStem,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CatBnAct,
     lambda: ([], {'in_chs': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (CellStem0,
     lambda: ([], {'stem_filters': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (CellStem1,
     lambda: ([], {'stem_filters': 4, 'num_filters': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 8, 64, 64])], {}),
     False),
    (DPN,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (Identity,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InceptionResNetV2,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 512, 512])], {}),
     False),
    (InceptionV4,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 512, 512])], {}),
     True),
    (Inception_A,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 384, 64, 64])], {}),
     True),
    (Inception_B,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1024, 64, 64])], {}),
     True),
    (Inception_C,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1536, 64, 64])], {}),
     True),
    (InputBlock,
     lambda: ([], {'num_init_features': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (InvertedResidual,
     lambda: ([], {'inp': 4, 'oup': 4, 'stride': 1, 'expand_ratio': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LambdaBase,
     lambda: ([], {'fn': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MaxPoolPad,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Mixed_3a,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 64, 64, 64])], {}),
     True),
    (Mixed_4a,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 160, 64, 64])], {}),
     True),
    (Mixed_5a,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 192, 64, 64])], {}),
     True),
    (Mixed_5b,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 192, 64, 64])], {}),
     True),
    (Mixed_6a,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 320, 64, 64])], {}),
     True),
    (Mixed_7a,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1088, 64, 64])], {}),
     True),
    (MobileNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 256, 256])], {}),
     True),
    (MobileNetV2,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 256, 256])], {}),
     True),
    (NormalCell,
     lambda: ([], {'in_channels_left': 4, 'out_channels_left': 4, 'in_channels_right': 4, 'out_channels_right': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (ReductionCell0,
     lambda: ([], {'in_channels_left': 4, 'out_channels_left': 4, 'in_channels_right': 4, 'out_channels_right': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (ReductionCell1,
     lambda: ([], {'in_channels_left': 4, 'out_channels_left': 4, 'in_channels_right': 4, 'out_channels_right': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (Reduction_A,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 384, 64, 64])], {}),
     True),
    (Reduction_B,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1024, 64, 64])], {}),
     True),
    (SEModule,
     lambda: ([], {'channels': 4, 'reduction': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SeparableConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ShuffleNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (SpatialCrossMapLRN,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_CeLuigi_models_comparison_pytorch(_paritybench_base):
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

    def test_010(self):
        self._check(*TESTCASES[10])

    def test_011(self):
        self._check(*TESTCASES[11])

    def test_012(self):
        self._check(*TESTCASES[12])

    def test_013(self):
        self._check(*TESTCASES[13])

    def test_014(self):
        self._check(*TESTCASES[14])

    def test_015(self):
        self._check(*TESTCASES[15])

    def test_016(self):
        self._check(*TESTCASES[16])

    def test_017(self):
        self._check(*TESTCASES[17])

    def test_018(self):
        self._check(*TESTCASES[18])

    def test_019(self):
        self._check(*TESTCASES[19])

    def test_020(self):
        self._check(*TESTCASES[20])

    def test_021(self):
        self._check(*TESTCASES[21])

    def test_022(self):
        self._check(*TESTCASES[22])

    def test_023(self):
        self._check(*TESTCASES[23])

    def test_024(self):
        self._check(*TESTCASES[24])

    def test_025(self):
        self._check(*TESTCASES[25])

    def test_026(self):
        self._check(*TESTCASES[26])

    def test_027(self):
        self._check(*TESTCASES[27])

    def test_028(self):
        self._check(*TESTCASES[28])

    def test_029(self):
        self._check(*TESTCASES[29])

    def test_030(self):
        self._check(*TESTCASES[30])

    def test_031(self):
        self._check(*TESTCASES[31])

    def test_032(self):
        self._check(*TESTCASES[32])

    def test_033(self):
        self._check(*TESTCASES[33])

    def test_034(self):
        self._check(*TESTCASES[34])

    def test_035(self):
        self._check(*TESTCASES[35])

    def test_036(self):
        self._check(*TESTCASES[36])

    def test_037(self):
        self._check(*TESTCASES[37])

    def test_038(self):
        self._check(*TESTCASES[38])

    def test_039(self):
        self._check(*TESTCASES[39])

    def test_040(self):
        self._check(*TESTCASES[40])

    def test_041(self):
        self._check(*TESTCASES[41])

    def test_042(self):
        self._check(*TESTCASES[42])

