import sys
_module = sys.modules[__name__]
del sys
imagenet_eval = _module
imagenet_logits = _module
visu_arch = _module
voc2007_extract = _module
pretorched = _module
__version__ = _module
datasets = _module
utils = _module
voc = _module
models = _module
bninception = _module
cafferesnet = _module
densenet3D = _module
dpn = _module
fbresnet = _module
resnet152_load = _module
inceptionresnetv2 = _module
inceptionv4 = _module
mobilenet = _module
multiview = _module
nasnet = _module
nasnet_mobile = _module
nonlocalnet = _module
pnasnet = _module
polynet = _module
pre_act_resnet3D = _module
r2plus1d = _module
resnet3D = _module
resnext = _module
resnext3D = _module
resnext_features = _module
resnext101_32x4d_features = _module
resnext101_64x4d_features = _module
senet = _module
settings = _module
slowfast = _module
soundnet = _module
torchvision_models = _module
trn = _module
utils = _module
vggm = _module
wideresnet = _module
wideresnet3D = _module
xception = _module
transforms = _module
utils = _module
setup = _module

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


import time


import torch


import torch.nn as nn


import torch.nn.parallel


import torch.backends.cudnn as cudnn


import torch.optim


import torch.utils.data


from torch.autograd import Variable


from torch.utils import model_zoo


import torch.utils.model_zoo as model_zoo


import math


import torch.nn.functional as F


from collections import OrderedDict


from torch import nn as nnl


import collections


from functools import partial


from torch.nn.modules.utils import _triple


from torch.nn.modules.utils import _pair


import numpy as np


from collections import defaultdict


from torch import nn


from torch.nn import functional as F


from functools import reduce


import re


import types


import itertools


from torch.nn import init


class BNInception(nn.Module):

    def __init__(self, num_classes=1000):
        super(BNInception, self).__init__()
        inplace = True
        self.conv1_7x7_s2 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2,
            2), padding=(3, 3))
        self.conv1_7x7_s2_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.9,
            affine=True)
        self.conv1_relu_7x7 = nn.ReLU(inplace)
        self.pool1_3x3_s2 = nn.MaxPool2d((3, 3), stride=(2, 2), dilation=(1,
            1), ceil_mode=True)
        self.conv2_3x3_reduce = nn.Conv2d(64, 64, kernel_size=(1, 1),
            stride=(1, 1))
        self.conv2_3x3_reduce_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=
            0.9, affine=True)
        self.conv2_relu_3x3_reduce = nn.ReLU(inplace)
        self.conv2_3x3 = nn.Conv2d(64, 192, kernel_size=(3, 3), stride=(1, 
            1), padding=(1, 1))
        self.conv2_3x3_bn = nn.BatchNorm2d(192, eps=1e-05, momentum=0.9,
            affine=True)
        self.conv2_relu_3x3 = nn.ReLU(inplace)
        self.pool2_3x3_s2 = nn.MaxPool2d((3, 3), stride=(2, 2), dilation=(1,
            1), ceil_mode=True)
        self.inception_3a_1x1 = nn.Conv2d(192, 64, kernel_size=(1, 1),
            stride=(1, 1))
        self.inception_3a_1x1_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=
            0.9, affine=True)
        self.inception_3a_relu_1x1 = nn.ReLU(inplace)
        self.inception_3a_3x3_reduce = nn.Conv2d(192, 64, kernel_size=(1, 1
            ), stride=(1, 1))
        self.inception_3a_3x3_reduce_bn = nn.BatchNorm2d(64, eps=1e-05,
            momentum=0.9, affine=True)
        self.inception_3a_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_3a_3x3 = nn.Conv2d(64, 64, kernel_size=(3, 3),
            stride=(1, 1), padding=(1, 1))
        self.inception_3a_3x3_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=
            0.9, affine=True)
        self.inception_3a_relu_3x3 = nn.ReLU(inplace)
        self.inception_3a_double_3x3_reduce = nn.Conv2d(192, 64,
            kernel_size=(1, 1), stride=(1, 1))
        self.inception_3a_double_3x3_reduce_bn = nn.BatchNorm2d(64, eps=
            1e-05, momentum=0.9, affine=True)
        self.inception_3a_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_3a_double_3x3_1 = nn.Conv2d(64, 96, kernel_size=(3, 
            3), stride=(1, 1), padding=(1, 1))
        self.inception_3a_double_3x3_1_bn = nn.BatchNorm2d(96, eps=1e-05,
            momentum=0.9, affine=True)
        self.inception_3a_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_3a_double_3x3_2 = nn.Conv2d(96, 96, kernel_size=(3, 
            3), stride=(1, 1), padding=(1, 1))
        self.inception_3a_double_3x3_2_bn = nn.BatchNorm2d(96, eps=1e-05,
            momentum=0.9, affine=True)
        self.inception_3a_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_3a_pool = nn.AvgPool2d(3, stride=1, padding=1,
            ceil_mode=True, count_include_pad=True)
        self.inception_3a_pool_proj = nn.Conv2d(192, 32, kernel_size=(1, 1),
            stride=(1, 1))
        self.inception_3a_pool_proj_bn = nn.BatchNorm2d(32, eps=1e-05,
            momentum=0.9, affine=True)
        self.inception_3a_relu_pool_proj = nn.ReLU(inplace)
        self.inception_3b_1x1 = nn.Conv2d(256, 64, kernel_size=(1, 1),
            stride=(1, 1))
        self.inception_3b_1x1_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=
            0.9, affine=True)
        self.inception_3b_relu_1x1 = nn.ReLU(inplace)
        self.inception_3b_3x3_reduce = nn.Conv2d(256, 64, kernel_size=(1, 1
            ), stride=(1, 1))
        self.inception_3b_3x3_reduce_bn = nn.BatchNorm2d(64, eps=1e-05,
            momentum=0.9, affine=True)
        self.inception_3b_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_3b_3x3 = nn.Conv2d(64, 96, kernel_size=(3, 3),
            stride=(1, 1), padding=(1, 1))
        self.inception_3b_3x3_bn = nn.BatchNorm2d(96, eps=1e-05, momentum=
            0.9, affine=True)
        self.inception_3b_relu_3x3 = nn.ReLU(inplace)
        self.inception_3b_double_3x3_reduce = nn.Conv2d(256, 64,
            kernel_size=(1, 1), stride=(1, 1))
        self.inception_3b_double_3x3_reduce_bn = nn.BatchNorm2d(64, eps=
            1e-05, momentum=0.9, affine=True)
        self.inception_3b_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_3b_double_3x3_1 = nn.Conv2d(64, 96, kernel_size=(3, 
            3), stride=(1, 1), padding=(1, 1))
        self.inception_3b_double_3x3_1_bn = nn.BatchNorm2d(96, eps=1e-05,
            momentum=0.9, affine=True)
        self.inception_3b_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_3b_double_3x3_2 = nn.Conv2d(96, 96, kernel_size=(3, 
            3), stride=(1, 1), padding=(1, 1))
        self.inception_3b_double_3x3_2_bn = nn.BatchNorm2d(96, eps=1e-05,
            momentum=0.9, affine=True)
        self.inception_3b_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_3b_pool = nn.AvgPool2d(3, stride=1, padding=1,
            ceil_mode=True, count_include_pad=True)
        self.inception_3b_pool_proj = nn.Conv2d(256, 64, kernel_size=(1, 1),
            stride=(1, 1))
        self.inception_3b_pool_proj_bn = nn.BatchNorm2d(64, eps=1e-05,
            momentum=0.9, affine=True)
        self.inception_3b_relu_pool_proj = nn.ReLU(inplace)
        self.inception_3c_3x3_reduce = nn.Conv2d(320, 128, kernel_size=(1, 
            1), stride=(1, 1))
        self.inception_3c_3x3_reduce_bn = nn.BatchNorm2d(128, eps=1e-05,
            momentum=0.9, affine=True)
        self.inception_3c_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_3c_3x3 = nn.Conv2d(128, 160, kernel_size=(3, 3),
            stride=(2, 2), padding=(1, 1))
        self.inception_3c_3x3_bn = nn.BatchNorm2d(160, eps=1e-05, momentum=
            0.9, affine=True)
        self.inception_3c_relu_3x3 = nn.ReLU(inplace)
        self.inception_3c_double_3x3_reduce = nn.Conv2d(320, 64,
            kernel_size=(1, 1), stride=(1, 1))
        self.inception_3c_double_3x3_reduce_bn = nn.BatchNorm2d(64, eps=
            1e-05, momentum=0.9, affine=True)
        self.inception_3c_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_3c_double_3x3_1 = nn.Conv2d(64, 96, kernel_size=(3, 
            3), stride=(1, 1), padding=(1, 1))
        self.inception_3c_double_3x3_1_bn = nn.BatchNorm2d(96, eps=1e-05,
            momentum=0.9, affine=True)
        self.inception_3c_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_3c_double_3x3_2 = nn.Conv2d(96, 96, kernel_size=(3, 
            3), stride=(2, 2), padding=(1, 1))
        self.inception_3c_double_3x3_2_bn = nn.BatchNorm2d(96, eps=1e-05,
            momentum=0.9, affine=True)
        self.inception_3c_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_3c_pool = nn.MaxPool2d((3, 3), stride=(2, 2),
            dilation=(1, 1), ceil_mode=True)
        self.inception_4a_1x1 = nn.Conv2d(576, 224, kernel_size=(1, 1),
            stride=(1, 1))
        self.inception_4a_1x1_bn = nn.BatchNorm2d(224, eps=1e-05, momentum=
            0.9, affine=True)
        self.inception_4a_relu_1x1 = nn.ReLU(inplace)
        self.inception_4a_3x3_reduce = nn.Conv2d(576, 64, kernel_size=(1, 1
            ), stride=(1, 1))
        self.inception_4a_3x3_reduce_bn = nn.BatchNorm2d(64, eps=1e-05,
            momentum=0.9, affine=True)
        self.inception_4a_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_4a_3x3 = nn.Conv2d(64, 96, kernel_size=(3, 3),
            stride=(1, 1), padding=(1, 1))
        self.inception_4a_3x3_bn = nn.BatchNorm2d(96, eps=1e-05, momentum=
            0.9, affine=True)
        self.inception_4a_relu_3x3 = nn.ReLU(inplace)
        self.inception_4a_double_3x3_reduce = nn.Conv2d(576, 96,
            kernel_size=(1, 1), stride=(1, 1))
        self.inception_4a_double_3x3_reduce_bn = nn.BatchNorm2d(96, eps=
            1e-05, momentum=0.9, affine=True)
        self.inception_4a_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_4a_double_3x3_1 = nn.Conv2d(96, 128, kernel_size=(3,
            3), stride=(1, 1), padding=(1, 1))
        self.inception_4a_double_3x3_1_bn = nn.BatchNorm2d(128, eps=1e-05,
            momentum=0.9, affine=True)
        self.inception_4a_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_4a_double_3x3_2 = nn.Conv2d(128, 128, kernel_size=(3,
            3), stride=(1, 1), padding=(1, 1))
        self.inception_4a_double_3x3_2_bn = nn.BatchNorm2d(128, eps=1e-05,
            momentum=0.9, affine=True)
        self.inception_4a_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_4a_pool = nn.AvgPool2d(3, stride=1, padding=1,
            ceil_mode=True, count_include_pad=True)
        self.inception_4a_pool_proj = nn.Conv2d(576, 128, kernel_size=(1, 1
            ), stride=(1, 1))
        self.inception_4a_pool_proj_bn = nn.BatchNorm2d(128, eps=1e-05,
            momentum=0.9, affine=True)
        self.inception_4a_relu_pool_proj = nn.ReLU(inplace)
        self.inception_4b_1x1 = nn.Conv2d(576, 192, kernel_size=(1, 1),
            stride=(1, 1))
        self.inception_4b_1x1_bn = nn.BatchNorm2d(192, eps=1e-05, momentum=
            0.9, affine=True)
        self.inception_4b_relu_1x1 = nn.ReLU(inplace)
        self.inception_4b_3x3_reduce = nn.Conv2d(576, 96, kernel_size=(1, 1
            ), stride=(1, 1))
        self.inception_4b_3x3_reduce_bn = nn.BatchNorm2d(96, eps=1e-05,
            momentum=0.9, affine=True)
        self.inception_4b_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_4b_3x3 = nn.Conv2d(96, 128, kernel_size=(3, 3),
            stride=(1, 1), padding=(1, 1))
        self.inception_4b_3x3_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=
            0.9, affine=True)
        self.inception_4b_relu_3x3 = nn.ReLU(inplace)
        self.inception_4b_double_3x3_reduce = nn.Conv2d(576, 96,
            kernel_size=(1, 1), stride=(1, 1))
        self.inception_4b_double_3x3_reduce_bn = nn.BatchNorm2d(96, eps=
            1e-05, momentum=0.9, affine=True)
        self.inception_4b_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_4b_double_3x3_1 = nn.Conv2d(96, 128, kernel_size=(3,
            3), stride=(1, 1), padding=(1, 1))
        self.inception_4b_double_3x3_1_bn = nn.BatchNorm2d(128, eps=1e-05,
            momentum=0.9, affine=True)
        self.inception_4b_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_4b_double_3x3_2 = nn.Conv2d(128, 128, kernel_size=(3,
            3), stride=(1, 1), padding=(1, 1))
        self.inception_4b_double_3x3_2_bn = nn.BatchNorm2d(128, eps=1e-05,
            momentum=0.9, affine=True)
        self.inception_4b_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_4b_pool = nn.AvgPool2d(3, stride=1, padding=1,
            ceil_mode=True, count_include_pad=True)
        self.inception_4b_pool_proj = nn.Conv2d(576, 128, kernel_size=(1, 1
            ), stride=(1, 1))
        self.inception_4b_pool_proj_bn = nn.BatchNorm2d(128, eps=1e-05,
            momentum=0.9, affine=True)
        self.inception_4b_relu_pool_proj = nn.ReLU(inplace)
        self.inception_4c_1x1 = nn.Conv2d(576, 160, kernel_size=(1, 1),
            stride=(1, 1))
        self.inception_4c_1x1_bn = nn.BatchNorm2d(160, eps=1e-05, momentum=
            0.9, affine=True)
        self.inception_4c_relu_1x1 = nn.ReLU(inplace)
        self.inception_4c_3x3_reduce = nn.Conv2d(576, 128, kernel_size=(1, 
            1), stride=(1, 1))
        self.inception_4c_3x3_reduce_bn = nn.BatchNorm2d(128, eps=1e-05,
            momentum=0.9, affine=True)
        self.inception_4c_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_4c_3x3 = nn.Conv2d(128, 160, kernel_size=(3, 3),
            stride=(1, 1), padding=(1, 1))
        self.inception_4c_3x3_bn = nn.BatchNorm2d(160, eps=1e-05, momentum=
            0.9, affine=True)
        self.inception_4c_relu_3x3 = nn.ReLU(inplace)
        self.inception_4c_double_3x3_reduce = nn.Conv2d(576, 128,
            kernel_size=(1, 1), stride=(1, 1))
        self.inception_4c_double_3x3_reduce_bn = nn.BatchNorm2d(128, eps=
            1e-05, momentum=0.9, affine=True)
        self.inception_4c_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_4c_double_3x3_1 = nn.Conv2d(128, 160, kernel_size=(3,
            3), stride=(1, 1), padding=(1, 1))
        self.inception_4c_double_3x3_1_bn = nn.BatchNorm2d(160, eps=1e-05,
            momentum=0.9, affine=True)
        self.inception_4c_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_4c_double_3x3_2 = nn.Conv2d(160, 160, kernel_size=(3,
            3), stride=(1, 1), padding=(1, 1))
        self.inception_4c_double_3x3_2_bn = nn.BatchNorm2d(160, eps=1e-05,
            momentum=0.9, affine=True)
        self.inception_4c_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_4c_pool = nn.AvgPool2d(3, stride=1, padding=1,
            ceil_mode=True, count_include_pad=True)
        self.inception_4c_pool_proj = nn.Conv2d(576, 128, kernel_size=(1, 1
            ), stride=(1, 1))
        self.inception_4c_pool_proj_bn = nn.BatchNorm2d(128, eps=1e-05,
            momentum=0.9, affine=True)
        self.inception_4c_relu_pool_proj = nn.ReLU(inplace)
        self.inception_4d_1x1 = nn.Conv2d(608, 96, kernel_size=(1, 1),
            stride=(1, 1))
        self.inception_4d_1x1_bn = nn.BatchNorm2d(96, eps=1e-05, momentum=
            0.9, affine=True)
        self.inception_4d_relu_1x1 = nn.ReLU(inplace)
        self.inception_4d_3x3_reduce = nn.Conv2d(608, 128, kernel_size=(1, 
            1), stride=(1, 1))
        self.inception_4d_3x3_reduce_bn = nn.BatchNorm2d(128, eps=1e-05,
            momentum=0.9, affine=True)
        self.inception_4d_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_4d_3x3 = nn.Conv2d(128, 192, kernel_size=(3, 3),
            stride=(1, 1), padding=(1, 1))
        self.inception_4d_3x3_bn = nn.BatchNorm2d(192, eps=1e-05, momentum=
            0.9, affine=True)
        self.inception_4d_relu_3x3 = nn.ReLU(inplace)
        self.inception_4d_double_3x3_reduce = nn.Conv2d(608, 160,
            kernel_size=(1, 1), stride=(1, 1))
        self.inception_4d_double_3x3_reduce_bn = nn.BatchNorm2d(160, eps=
            1e-05, momentum=0.9, affine=True)
        self.inception_4d_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_4d_double_3x3_1 = nn.Conv2d(160, 192, kernel_size=(3,
            3), stride=(1, 1), padding=(1, 1))
        self.inception_4d_double_3x3_1_bn = nn.BatchNorm2d(192, eps=1e-05,
            momentum=0.9, affine=True)
        self.inception_4d_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_4d_double_3x3_2 = nn.Conv2d(192, 192, kernel_size=(3,
            3), stride=(1, 1), padding=(1, 1))
        self.inception_4d_double_3x3_2_bn = nn.BatchNorm2d(192, eps=1e-05,
            momentum=0.9, affine=True)
        self.inception_4d_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_4d_pool = nn.AvgPool2d(3, stride=1, padding=1,
            ceil_mode=True, count_include_pad=True)
        self.inception_4d_pool_proj = nn.Conv2d(608, 128, kernel_size=(1, 1
            ), stride=(1, 1))
        self.inception_4d_pool_proj_bn = nn.BatchNorm2d(128, eps=1e-05,
            momentum=0.9, affine=True)
        self.inception_4d_relu_pool_proj = nn.ReLU(inplace)
        self.inception_4e_3x3_reduce = nn.Conv2d(608, 128, kernel_size=(1, 
            1), stride=(1, 1))
        self.inception_4e_3x3_reduce_bn = nn.BatchNorm2d(128, eps=1e-05,
            momentum=0.9, affine=True)
        self.inception_4e_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_4e_3x3 = nn.Conv2d(128, 192, kernel_size=(3, 3),
            stride=(2, 2), padding=(1, 1))
        self.inception_4e_3x3_bn = nn.BatchNorm2d(192, eps=1e-05, momentum=
            0.9, affine=True)
        self.inception_4e_relu_3x3 = nn.ReLU(inplace)
        self.inception_4e_double_3x3_reduce = nn.Conv2d(608, 192,
            kernel_size=(1, 1), stride=(1, 1))
        self.inception_4e_double_3x3_reduce_bn = nn.BatchNorm2d(192, eps=
            1e-05, momentum=0.9, affine=True)
        self.inception_4e_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_4e_double_3x3_1 = nn.Conv2d(192, 256, kernel_size=(3,
            3), stride=(1, 1), padding=(1, 1))
        self.inception_4e_double_3x3_1_bn = nn.BatchNorm2d(256, eps=1e-05,
            momentum=0.9, affine=True)
        self.inception_4e_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_4e_double_3x3_2 = nn.Conv2d(256, 256, kernel_size=(3,
            3), stride=(2, 2), padding=(1, 1))
        self.inception_4e_double_3x3_2_bn = nn.BatchNorm2d(256, eps=1e-05,
            momentum=0.9, affine=True)
        self.inception_4e_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_4e_pool = nn.MaxPool2d((3, 3), stride=(2, 2),
            dilation=(1, 1), ceil_mode=True)
        self.inception_5a_1x1 = nn.Conv2d(1056, 352, kernel_size=(1, 1),
            stride=(1, 1))
        self.inception_5a_1x1_bn = nn.BatchNorm2d(352, eps=1e-05, momentum=
            0.9, affine=True)
        self.inception_5a_relu_1x1 = nn.ReLU(inplace)
        self.inception_5a_3x3_reduce = nn.Conv2d(1056, 192, kernel_size=(1,
            1), stride=(1, 1))
        self.inception_5a_3x3_reduce_bn = nn.BatchNorm2d(192, eps=1e-05,
            momentum=0.9, affine=True)
        self.inception_5a_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_5a_3x3 = nn.Conv2d(192, 320, kernel_size=(3, 3),
            stride=(1, 1), padding=(1, 1))
        self.inception_5a_3x3_bn = nn.BatchNorm2d(320, eps=1e-05, momentum=
            0.9, affine=True)
        self.inception_5a_relu_3x3 = nn.ReLU(inplace)
        self.inception_5a_double_3x3_reduce = nn.Conv2d(1056, 160,
            kernel_size=(1, 1), stride=(1, 1))
        self.inception_5a_double_3x3_reduce_bn = nn.BatchNorm2d(160, eps=
            1e-05, momentum=0.9, affine=True)
        self.inception_5a_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_5a_double_3x3_1 = nn.Conv2d(160, 224, kernel_size=(3,
            3), stride=(1, 1), padding=(1, 1))
        self.inception_5a_double_3x3_1_bn = nn.BatchNorm2d(224, eps=1e-05,
            momentum=0.9, affine=True)
        self.inception_5a_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_5a_double_3x3_2 = nn.Conv2d(224, 224, kernel_size=(3,
            3), stride=(1, 1), padding=(1, 1))
        self.inception_5a_double_3x3_2_bn = nn.BatchNorm2d(224, eps=1e-05,
            momentum=0.9, affine=True)
        self.inception_5a_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_5a_pool = nn.AvgPool2d(3, stride=1, padding=1,
            ceil_mode=True, count_include_pad=True)
        self.inception_5a_pool_proj = nn.Conv2d(1056, 128, kernel_size=(1, 
            1), stride=(1, 1))
        self.inception_5a_pool_proj_bn = nn.BatchNorm2d(128, eps=1e-05,
            momentum=0.9, affine=True)
        self.inception_5a_relu_pool_proj = nn.ReLU(inplace)
        self.inception_5b_1x1 = nn.Conv2d(1024, 352, kernel_size=(1, 1),
            stride=(1, 1))
        self.inception_5b_1x1_bn = nn.BatchNorm2d(352, eps=1e-05, momentum=
            0.9, affine=True)
        self.inception_5b_relu_1x1 = nn.ReLU(inplace)
        self.inception_5b_3x3_reduce = nn.Conv2d(1024, 192, kernel_size=(1,
            1), stride=(1, 1))
        self.inception_5b_3x3_reduce_bn = nn.BatchNorm2d(192, eps=1e-05,
            momentum=0.9, affine=True)
        self.inception_5b_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_5b_3x3 = nn.Conv2d(192, 320, kernel_size=(3, 3),
            stride=(1, 1), padding=(1, 1))
        self.inception_5b_3x3_bn = nn.BatchNorm2d(320, eps=1e-05, momentum=
            0.9, affine=True)
        self.inception_5b_relu_3x3 = nn.ReLU(inplace)
        self.inception_5b_double_3x3_reduce = nn.Conv2d(1024, 192,
            kernel_size=(1, 1), stride=(1, 1))
        self.inception_5b_double_3x3_reduce_bn = nn.BatchNorm2d(192, eps=
            1e-05, momentum=0.9, affine=True)
        self.inception_5b_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_5b_double_3x3_1 = nn.Conv2d(192, 224, kernel_size=(3,
            3), stride=(1, 1), padding=(1, 1))
        self.inception_5b_double_3x3_1_bn = nn.BatchNorm2d(224, eps=1e-05,
            momentum=0.9, affine=True)
        self.inception_5b_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_5b_double_3x3_2 = nn.Conv2d(224, 224, kernel_size=(3,
            3), stride=(1, 1), padding=(1, 1))
        self.inception_5b_double_3x3_2_bn = nn.BatchNorm2d(224, eps=1e-05,
            momentum=0.9, affine=True)
        self.inception_5b_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_5b_pool = nn.MaxPool2d((3, 3), stride=(1, 1),
            padding=(1, 1), dilation=(1, 1), ceil_mode=True)
        self.inception_5b_pool_proj = nn.Conv2d(1024, 128, kernel_size=(1, 
            1), stride=(1, 1))
        self.inception_5b_pool_proj_bn = nn.BatchNorm2d(128, eps=1e-05,
            momentum=0.9, affine=True)
        self.inception_5b_relu_pool_proj = nn.ReLU(inplace)
        self.global_pool = nn.AvgPool2d(7, stride=1, padding=0, ceil_mode=
            True, count_include_pad=True)
        self.last_linear = nn.Linear(1024, num_classes)

    def features(self, input):
        conv1_7x7_s2_out = self.conv1_7x7_s2(input)
        conv1_7x7_s2_bn_out = self.conv1_7x7_s2_bn(conv1_7x7_s2_out)
        conv1_relu_7x7_out = self.conv1_relu_7x7(conv1_7x7_s2_bn_out)
        pool1_3x3_s2_out = self.pool1_3x3_s2(conv1_7x7_s2_bn_out)
        conv2_3x3_reduce_out = self.conv2_3x3_reduce(pool1_3x3_s2_out)
        conv2_3x3_reduce_bn_out = self.conv2_3x3_reduce_bn(conv2_3x3_reduce_out
            )
        conv2_relu_3x3_reduce_out = self.conv2_relu_3x3_reduce(
            conv2_3x3_reduce_bn_out)
        conv2_3x3_out = self.conv2_3x3(conv2_3x3_reduce_bn_out)
        conv2_3x3_bn_out = self.conv2_3x3_bn(conv2_3x3_out)
        conv2_relu_3x3_out = self.conv2_relu_3x3(conv2_3x3_bn_out)
        pool2_3x3_s2_out = self.pool2_3x3_s2(conv2_3x3_bn_out)
        inception_3a_1x1_out = self.inception_3a_1x1(pool2_3x3_s2_out)
        inception_3a_1x1_bn_out = self.inception_3a_1x1_bn(inception_3a_1x1_out
            )
        inception_3a_relu_1x1_out = self.inception_3a_relu_1x1(
            inception_3a_1x1_bn_out)
        inception_3a_3x3_reduce_out = self.inception_3a_3x3_reduce(
            pool2_3x3_s2_out)
        inception_3a_3x3_reduce_bn_out = self.inception_3a_3x3_reduce_bn(
            inception_3a_3x3_reduce_out)
        inception_3a_relu_3x3_reduce_out = self.inception_3a_relu_3x3_reduce(
            inception_3a_3x3_reduce_bn_out)
        inception_3a_3x3_out = self.inception_3a_3x3(
            inception_3a_3x3_reduce_bn_out)
        inception_3a_3x3_bn_out = self.inception_3a_3x3_bn(inception_3a_3x3_out
            )
        inception_3a_relu_3x3_out = self.inception_3a_relu_3x3(
            inception_3a_3x3_bn_out)
        inception_3a_double_3x3_reduce_out = (self.
            inception_3a_double_3x3_reduce(pool2_3x3_s2_out))
        inception_3a_double_3x3_reduce_bn_out = (self.
            inception_3a_double_3x3_reduce_bn(
            inception_3a_double_3x3_reduce_out))
        inception_3a_relu_double_3x3_reduce_out = (self.
            inception_3a_relu_double_3x3_reduce(
            inception_3a_double_3x3_reduce_bn_out))
        inception_3a_double_3x3_1_out = self.inception_3a_double_3x3_1(
            inception_3a_double_3x3_reduce_bn_out)
        inception_3a_double_3x3_1_bn_out = self.inception_3a_double_3x3_1_bn(
            inception_3a_double_3x3_1_out)
        inception_3a_relu_double_3x3_1_out = (self.
            inception_3a_relu_double_3x3_1(inception_3a_double_3x3_1_bn_out))
        inception_3a_double_3x3_2_out = self.inception_3a_double_3x3_2(
            inception_3a_double_3x3_1_bn_out)
        inception_3a_double_3x3_2_bn_out = self.inception_3a_double_3x3_2_bn(
            inception_3a_double_3x3_2_out)
        inception_3a_relu_double_3x3_2_out = (self.
            inception_3a_relu_double_3x3_2(inception_3a_double_3x3_2_bn_out))
        inception_3a_pool_out = self.inception_3a_pool(pool2_3x3_s2_out)
        inception_3a_pool_proj_out = self.inception_3a_pool_proj(
            inception_3a_pool_out)
        inception_3a_pool_proj_bn_out = self.inception_3a_pool_proj_bn(
            inception_3a_pool_proj_out)
        inception_3a_relu_pool_proj_out = self.inception_3a_relu_pool_proj(
            inception_3a_pool_proj_bn_out)
        inception_3a_output_out = torch.cat([inception_3a_1x1_bn_out,
            inception_3a_3x3_bn_out, inception_3a_double_3x3_2_bn_out,
            inception_3a_pool_proj_bn_out], 1)
        inception_3b_1x1_out = self.inception_3b_1x1(inception_3a_output_out)
        inception_3b_1x1_bn_out = self.inception_3b_1x1_bn(inception_3b_1x1_out
            )
        inception_3b_relu_1x1_out = self.inception_3b_relu_1x1(
            inception_3b_1x1_bn_out)
        inception_3b_3x3_reduce_out = self.inception_3b_3x3_reduce(
            inception_3a_output_out)
        inception_3b_3x3_reduce_bn_out = self.inception_3b_3x3_reduce_bn(
            inception_3b_3x3_reduce_out)
        inception_3b_relu_3x3_reduce_out = self.inception_3b_relu_3x3_reduce(
            inception_3b_3x3_reduce_bn_out)
        inception_3b_3x3_out = self.inception_3b_3x3(
            inception_3b_3x3_reduce_bn_out)
        inception_3b_3x3_bn_out = self.inception_3b_3x3_bn(inception_3b_3x3_out
            )
        inception_3b_relu_3x3_out = self.inception_3b_relu_3x3(
            inception_3b_3x3_bn_out)
        inception_3b_double_3x3_reduce_out = (self.
            inception_3b_double_3x3_reduce(inception_3a_output_out))
        inception_3b_double_3x3_reduce_bn_out = (self.
            inception_3b_double_3x3_reduce_bn(
            inception_3b_double_3x3_reduce_out))
        inception_3b_relu_double_3x3_reduce_out = (self.
            inception_3b_relu_double_3x3_reduce(
            inception_3b_double_3x3_reduce_bn_out))
        inception_3b_double_3x3_1_out = self.inception_3b_double_3x3_1(
            inception_3b_double_3x3_reduce_bn_out)
        inception_3b_double_3x3_1_bn_out = self.inception_3b_double_3x3_1_bn(
            inception_3b_double_3x3_1_out)
        inception_3b_relu_double_3x3_1_out = (self.
            inception_3b_relu_double_3x3_1(inception_3b_double_3x3_1_bn_out))
        inception_3b_double_3x3_2_out = self.inception_3b_double_3x3_2(
            inception_3b_double_3x3_1_bn_out)
        inception_3b_double_3x3_2_bn_out = self.inception_3b_double_3x3_2_bn(
            inception_3b_double_3x3_2_out)
        inception_3b_relu_double_3x3_2_out = (self.
            inception_3b_relu_double_3x3_2(inception_3b_double_3x3_2_bn_out))
        inception_3b_pool_out = self.inception_3b_pool(inception_3a_output_out)
        inception_3b_pool_proj_out = self.inception_3b_pool_proj(
            inception_3b_pool_out)
        inception_3b_pool_proj_bn_out = self.inception_3b_pool_proj_bn(
            inception_3b_pool_proj_out)
        inception_3b_relu_pool_proj_out = self.inception_3b_relu_pool_proj(
            inception_3b_pool_proj_bn_out)
        inception_3b_output_out = torch.cat([inception_3b_1x1_bn_out,
            inception_3b_3x3_bn_out, inception_3b_double_3x3_2_bn_out,
            inception_3b_pool_proj_bn_out], 1)
        inception_3c_3x3_reduce_out = self.inception_3c_3x3_reduce(
            inception_3b_output_out)
        inception_3c_3x3_reduce_bn_out = self.inception_3c_3x3_reduce_bn(
            inception_3c_3x3_reduce_out)
        inception_3c_relu_3x3_reduce_out = self.inception_3c_relu_3x3_reduce(
            inception_3c_3x3_reduce_bn_out)
        inception_3c_3x3_out = self.inception_3c_3x3(
            inception_3c_3x3_reduce_bn_out)
        inception_3c_3x3_bn_out = self.inception_3c_3x3_bn(inception_3c_3x3_out
            )
        inception_3c_relu_3x3_out = self.inception_3c_relu_3x3(
            inception_3c_3x3_bn_out)
        inception_3c_double_3x3_reduce_out = (self.
            inception_3c_double_3x3_reduce(inception_3b_output_out))
        inception_3c_double_3x3_reduce_bn_out = (self.
            inception_3c_double_3x3_reduce_bn(
            inception_3c_double_3x3_reduce_out))
        inception_3c_relu_double_3x3_reduce_out = (self.
            inception_3c_relu_double_3x3_reduce(
            inception_3c_double_3x3_reduce_bn_out))
        inception_3c_double_3x3_1_out = self.inception_3c_double_3x3_1(
            inception_3c_double_3x3_reduce_bn_out)
        inception_3c_double_3x3_1_bn_out = self.inception_3c_double_3x3_1_bn(
            inception_3c_double_3x3_1_out)
        inception_3c_relu_double_3x3_1_out = (self.
            inception_3c_relu_double_3x3_1(inception_3c_double_3x3_1_bn_out))
        inception_3c_double_3x3_2_out = self.inception_3c_double_3x3_2(
            inception_3c_double_3x3_1_bn_out)
        inception_3c_double_3x3_2_bn_out = self.inception_3c_double_3x3_2_bn(
            inception_3c_double_3x3_2_out)
        inception_3c_relu_double_3x3_2_out = (self.
            inception_3c_relu_double_3x3_2(inception_3c_double_3x3_2_bn_out))
        inception_3c_pool_out = self.inception_3c_pool(inception_3b_output_out)
        inception_3c_output_out = torch.cat([inception_3c_3x3_bn_out,
            inception_3c_double_3x3_2_bn_out, inception_3c_pool_out], 1)
        inception_4a_1x1_out = self.inception_4a_1x1(inception_3c_output_out)
        inception_4a_1x1_bn_out = self.inception_4a_1x1_bn(inception_4a_1x1_out
            )
        inception_4a_relu_1x1_out = self.inception_4a_relu_1x1(
            inception_4a_1x1_bn_out)
        inception_4a_3x3_reduce_out = self.inception_4a_3x3_reduce(
            inception_3c_output_out)
        inception_4a_3x3_reduce_bn_out = self.inception_4a_3x3_reduce_bn(
            inception_4a_3x3_reduce_out)
        inception_4a_relu_3x3_reduce_out = self.inception_4a_relu_3x3_reduce(
            inception_4a_3x3_reduce_bn_out)
        inception_4a_3x3_out = self.inception_4a_3x3(
            inception_4a_3x3_reduce_bn_out)
        inception_4a_3x3_bn_out = self.inception_4a_3x3_bn(inception_4a_3x3_out
            )
        inception_4a_relu_3x3_out = self.inception_4a_relu_3x3(
            inception_4a_3x3_bn_out)
        inception_4a_double_3x3_reduce_out = (self.
            inception_4a_double_3x3_reduce(inception_3c_output_out))
        inception_4a_double_3x3_reduce_bn_out = (self.
            inception_4a_double_3x3_reduce_bn(
            inception_4a_double_3x3_reduce_out))
        inception_4a_relu_double_3x3_reduce_out = (self.
            inception_4a_relu_double_3x3_reduce(
            inception_4a_double_3x3_reduce_bn_out))
        inception_4a_double_3x3_1_out = self.inception_4a_double_3x3_1(
            inception_4a_double_3x3_reduce_bn_out)
        inception_4a_double_3x3_1_bn_out = self.inception_4a_double_3x3_1_bn(
            inception_4a_double_3x3_1_out)
        inception_4a_relu_double_3x3_1_out = (self.
            inception_4a_relu_double_3x3_1(inception_4a_double_3x3_1_bn_out))
        inception_4a_double_3x3_2_out = self.inception_4a_double_3x3_2(
            inception_4a_double_3x3_1_bn_out)
        inception_4a_double_3x3_2_bn_out = self.inception_4a_double_3x3_2_bn(
            inception_4a_double_3x3_2_out)
        inception_4a_relu_double_3x3_2_out = (self.
            inception_4a_relu_double_3x3_2(inception_4a_double_3x3_2_bn_out))
        inception_4a_pool_out = self.inception_4a_pool(inception_3c_output_out)
        inception_4a_pool_proj_out = self.inception_4a_pool_proj(
            inception_4a_pool_out)
        inception_4a_pool_proj_bn_out = self.inception_4a_pool_proj_bn(
            inception_4a_pool_proj_out)
        inception_4a_relu_pool_proj_out = self.inception_4a_relu_pool_proj(
            inception_4a_pool_proj_bn_out)
        inception_4a_output_out = torch.cat([inception_4a_1x1_bn_out,
            inception_4a_3x3_bn_out, inception_4a_double_3x3_2_bn_out,
            inception_4a_pool_proj_bn_out], 1)
        inception_4b_1x1_out = self.inception_4b_1x1(inception_4a_output_out)
        inception_4b_1x1_bn_out = self.inception_4b_1x1_bn(inception_4b_1x1_out
            )
        inception_4b_relu_1x1_out = self.inception_4b_relu_1x1(
            inception_4b_1x1_bn_out)
        inception_4b_3x3_reduce_out = self.inception_4b_3x3_reduce(
            inception_4a_output_out)
        inception_4b_3x3_reduce_bn_out = self.inception_4b_3x3_reduce_bn(
            inception_4b_3x3_reduce_out)
        inception_4b_relu_3x3_reduce_out = self.inception_4b_relu_3x3_reduce(
            inception_4b_3x3_reduce_bn_out)
        inception_4b_3x3_out = self.inception_4b_3x3(
            inception_4b_3x3_reduce_bn_out)
        inception_4b_3x3_bn_out = self.inception_4b_3x3_bn(inception_4b_3x3_out
            )
        inception_4b_relu_3x3_out = self.inception_4b_relu_3x3(
            inception_4b_3x3_bn_out)
        inception_4b_double_3x3_reduce_out = (self.
            inception_4b_double_3x3_reduce(inception_4a_output_out))
        inception_4b_double_3x3_reduce_bn_out = (self.
            inception_4b_double_3x3_reduce_bn(
            inception_4b_double_3x3_reduce_out))
        inception_4b_relu_double_3x3_reduce_out = (self.
            inception_4b_relu_double_3x3_reduce(
            inception_4b_double_3x3_reduce_bn_out))
        inception_4b_double_3x3_1_out = self.inception_4b_double_3x3_1(
            inception_4b_double_3x3_reduce_bn_out)
        inception_4b_double_3x3_1_bn_out = self.inception_4b_double_3x3_1_bn(
            inception_4b_double_3x3_1_out)
        inception_4b_relu_double_3x3_1_out = (self.
            inception_4b_relu_double_3x3_1(inception_4b_double_3x3_1_bn_out))
        inception_4b_double_3x3_2_out = self.inception_4b_double_3x3_2(
            inception_4b_double_3x3_1_bn_out)
        inception_4b_double_3x3_2_bn_out = self.inception_4b_double_3x3_2_bn(
            inception_4b_double_3x3_2_out)
        inception_4b_relu_double_3x3_2_out = (self.
            inception_4b_relu_double_3x3_2(inception_4b_double_3x3_2_bn_out))
        inception_4b_pool_out = self.inception_4b_pool(inception_4a_output_out)
        inception_4b_pool_proj_out = self.inception_4b_pool_proj(
            inception_4b_pool_out)
        inception_4b_pool_proj_bn_out = self.inception_4b_pool_proj_bn(
            inception_4b_pool_proj_out)
        inception_4b_relu_pool_proj_out = self.inception_4b_relu_pool_proj(
            inception_4b_pool_proj_bn_out)
        inception_4b_output_out = torch.cat([inception_4b_1x1_bn_out,
            inception_4b_3x3_bn_out, inception_4b_double_3x3_2_bn_out,
            inception_4b_pool_proj_bn_out], 1)
        inception_4c_1x1_out = self.inception_4c_1x1(inception_4b_output_out)
        inception_4c_1x1_bn_out = self.inception_4c_1x1_bn(inception_4c_1x1_out
            )
        inception_4c_relu_1x1_out = self.inception_4c_relu_1x1(
            inception_4c_1x1_bn_out)
        inception_4c_3x3_reduce_out = self.inception_4c_3x3_reduce(
            inception_4b_output_out)
        inception_4c_3x3_reduce_bn_out = self.inception_4c_3x3_reduce_bn(
            inception_4c_3x3_reduce_out)
        inception_4c_relu_3x3_reduce_out = self.inception_4c_relu_3x3_reduce(
            inception_4c_3x3_reduce_bn_out)
        inception_4c_3x3_out = self.inception_4c_3x3(
            inception_4c_3x3_reduce_bn_out)
        inception_4c_3x3_bn_out = self.inception_4c_3x3_bn(inception_4c_3x3_out
            )
        inception_4c_relu_3x3_out = self.inception_4c_relu_3x3(
            inception_4c_3x3_bn_out)
        inception_4c_double_3x3_reduce_out = (self.
            inception_4c_double_3x3_reduce(inception_4b_output_out))
        inception_4c_double_3x3_reduce_bn_out = (self.
            inception_4c_double_3x3_reduce_bn(
            inception_4c_double_3x3_reduce_out))
        inception_4c_relu_double_3x3_reduce_out = (self.
            inception_4c_relu_double_3x3_reduce(
            inception_4c_double_3x3_reduce_bn_out))
        inception_4c_double_3x3_1_out = self.inception_4c_double_3x3_1(
            inception_4c_double_3x3_reduce_bn_out)
        inception_4c_double_3x3_1_bn_out = self.inception_4c_double_3x3_1_bn(
            inception_4c_double_3x3_1_out)
        inception_4c_relu_double_3x3_1_out = (self.
            inception_4c_relu_double_3x3_1(inception_4c_double_3x3_1_bn_out))
        inception_4c_double_3x3_2_out = self.inception_4c_double_3x3_2(
            inception_4c_double_3x3_1_bn_out)
        inception_4c_double_3x3_2_bn_out = self.inception_4c_double_3x3_2_bn(
            inception_4c_double_3x3_2_out)
        inception_4c_relu_double_3x3_2_out = (self.
            inception_4c_relu_double_3x3_2(inception_4c_double_3x3_2_bn_out))
        inception_4c_pool_out = self.inception_4c_pool(inception_4b_output_out)
        inception_4c_pool_proj_out = self.inception_4c_pool_proj(
            inception_4c_pool_out)
        inception_4c_pool_proj_bn_out = self.inception_4c_pool_proj_bn(
            inception_4c_pool_proj_out)
        inception_4c_relu_pool_proj_out = self.inception_4c_relu_pool_proj(
            inception_4c_pool_proj_bn_out)
        inception_4c_output_out = torch.cat([inception_4c_1x1_bn_out,
            inception_4c_3x3_bn_out, inception_4c_double_3x3_2_bn_out,
            inception_4c_pool_proj_bn_out], 1)
        inception_4d_1x1_out = self.inception_4d_1x1(inception_4c_output_out)
        inception_4d_1x1_bn_out = self.inception_4d_1x1_bn(inception_4d_1x1_out
            )
        inception_4d_relu_1x1_out = self.inception_4d_relu_1x1(
            inception_4d_1x1_bn_out)
        inception_4d_3x3_reduce_out = self.inception_4d_3x3_reduce(
            inception_4c_output_out)
        inception_4d_3x3_reduce_bn_out = self.inception_4d_3x3_reduce_bn(
            inception_4d_3x3_reduce_out)
        inception_4d_relu_3x3_reduce_out = self.inception_4d_relu_3x3_reduce(
            inception_4d_3x3_reduce_bn_out)
        inception_4d_3x3_out = self.inception_4d_3x3(
            inception_4d_3x3_reduce_bn_out)
        inception_4d_3x3_bn_out = self.inception_4d_3x3_bn(inception_4d_3x3_out
            )
        inception_4d_relu_3x3_out = self.inception_4d_relu_3x3(
            inception_4d_3x3_bn_out)
        inception_4d_double_3x3_reduce_out = (self.
            inception_4d_double_3x3_reduce(inception_4c_output_out))
        inception_4d_double_3x3_reduce_bn_out = (self.
            inception_4d_double_3x3_reduce_bn(
            inception_4d_double_3x3_reduce_out))
        inception_4d_relu_double_3x3_reduce_out = (self.
            inception_4d_relu_double_3x3_reduce(
            inception_4d_double_3x3_reduce_bn_out))
        inception_4d_double_3x3_1_out = self.inception_4d_double_3x3_1(
            inception_4d_double_3x3_reduce_bn_out)
        inception_4d_double_3x3_1_bn_out = self.inception_4d_double_3x3_1_bn(
            inception_4d_double_3x3_1_out)
        inception_4d_relu_double_3x3_1_out = (self.
            inception_4d_relu_double_3x3_1(inception_4d_double_3x3_1_bn_out))
        inception_4d_double_3x3_2_out = self.inception_4d_double_3x3_2(
            inception_4d_double_3x3_1_bn_out)
        inception_4d_double_3x3_2_bn_out = self.inception_4d_double_3x3_2_bn(
            inception_4d_double_3x3_2_out)
        inception_4d_relu_double_3x3_2_out = (self.
            inception_4d_relu_double_3x3_2(inception_4d_double_3x3_2_bn_out))
        inception_4d_pool_out = self.inception_4d_pool(inception_4c_output_out)
        inception_4d_pool_proj_out = self.inception_4d_pool_proj(
            inception_4d_pool_out)
        inception_4d_pool_proj_bn_out = self.inception_4d_pool_proj_bn(
            inception_4d_pool_proj_out)
        inception_4d_relu_pool_proj_out = self.inception_4d_relu_pool_proj(
            inception_4d_pool_proj_bn_out)
        inception_4d_output_out = torch.cat([inception_4d_1x1_bn_out,
            inception_4d_3x3_bn_out, inception_4d_double_3x3_2_bn_out,
            inception_4d_pool_proj_bn_out], 1)
        inception_4e_3x3_reduce_out = self.inception_4e_3x3_reduce(
            inception_4d_output_out)
        inception_4e_3x3_reduce_bn_out = self.inception_4e_3x3_reduce_bn(
            inception_4e_3x3_reduce_out)
        inception_4e_relu_3x3_reduce_out = self.inception_4e_relu_3x3_reduce(
            inception_4e_3x3_reduce_bn_out)
        inception_4e_3x3_out = self.inception_4e_3x3(
            inception_4e_3x3_reduce_bn_out)
        inception_4e_3x3_bn_out = self.inception_4e_3x3_bn(inception_4e_3x3_out
            )
        inception_4e_relu_3x3_out = self.inception_4e_relu_3x3(
            inception_4e_3x3_bn_out)
        inception_4e_double_3x3_reduce_out = (self.
            inception_4e_double_3x3_reduce(inception_4d_output_out))
        inception_4e_double_3x3_reduce_bn_out = (self.
            inception_4e_double_3x3_reduce_bn(
            inception_4e_double_3x3_reduce_out))
        inception_4e_relu_double_3x3_reduce_out = (self.
            inception_4e_relu_double_3x3_reduce(
            inception_4e_double_3x3_reduce_bn_out))
        inception_4e_double_3x3_1_out = self.inception_4e_double_3x3_1(
            inception_4e_double_3x3_reduce_bn_out)
        inception_4e_double_3x3_1_bn_out = self.inception_4e_double_3x3_1_bn(
            inception_4e_double_3x3_1_out)
        inception_4e_relu_double_3x3_1_out = (self.
            inception_4e_relu_double_3x3_1(inception_4e_double_3x3_1_bn_out))
        inception_4e_double_3x3_2_out = self.inception_4e_double_3x3_2(
            inception_4e_double_3x3_1_bn_out)
        inception_4e_double_3x3_2_bn_out = self.inception_4e_double_3x3_2_bn(
            inception_4e_double_3x3_2_out)
        inception_4e_relu_double_3x3_2_out = (self.
            inception_4e_relu_double_3x3_2(inception_4e_double_3x3_2_bn_out))
        inception_4e_pool_out = self.inception_4e_pool(inception_4d_output_out)
        inception_4e_output_out = torch.cat([inception_4e_3x3_bn_out,
            inception_4e_double_3x3_2_bn_out, inception_4e_pool_out], 1)
        inception_5a_1x1_out = self.inception_5a_1x1(inception_4e_output_out)
        inception_5a_1x1_bn_out = self.inception_5a_1x1_bn(inception_5a_1x1_out
            )
        inception_5a_relu_1x1_out = self.inception_5a_relu_1x1(
            inception_5a_1x1_bn_out)
        inception_5a_3x3_reduce_out = self.inception_5a_3x3_reduce(
            inception_4e_output_out)
        inception_5a_3x3_reduce_bn_out = self.inception_5a_3x3_reduce_bn(
            inception_5a_3x3_reduce_out)
        inception_5a_relu_3x3_reduce_out = self.inception_5a_relu_3x3_reduce(
            inception_5a_3x3_reduce_bn_out)
        inception_5a_3x3_out = self.inception_5a_3x3(
            inception_5a_3x3_reduce_bn_out)
        inception_5a_3x3_bn_out = self.inception_5a_3x3_bn(inception_5a_3x3_out
            )
        inception_5a_relu_3x3_out = self.inception_5a_relu_3x3(
            inception_5a_3x3_bn_out)
        inception_5a_double_3x3_reduce_out = (self.
            inception_5a_double_3x3_reduce(inception_4e_output_out))
        inception_5a_double_3x3_reduce_bn_out = (self.
            inception_5a_double_3x3_reduce_bn(
            inception_5a_double_3x3_reduce_out))
        inception_5a_relu_double_3x3_reduce_out = (self.
            inception_5a_relu_double_3x3_reduce(
            inception_5a_double_3x3_reduce_bn_out))
        inception_5a_double_3x3_1_out = self.inception_5a_double_3x3_1(
            inception_5a_double_3x3_reduce_bn_out)
        inception_5a_double_3x3_1_bn_out = self.inception_5a_double_3x3_1_bn(
            inception_5a_double_3x3_1_out)
        inception_5a_relu_double_3x3_1_out = (self.
            inception_5a_relu_double_3x3_1(inception_5a_double_3x3_1_bn_out))
        inception_5a_double_3x3_2_out = self.inception_5a_double_3x3_2(
            inception_5a_double_3x3_1_bn_out)
        inception_5a_double_3x3_2_bn_out = self.inception_5a_double_3x3_2_bn(
            inception_5a_double_3x3_2_out)
        inception_5a_relu_double_3x3_2_out = (self.
            inception_5a_relu_double_3x3_2(inception_5a_double_3x3_2_bn_out))
        inception_5a_pool_out = self.inception_5a_pool(inception_4e_output_out)
        inception_5a_pool_proj_out = self.inception_5a_pool_proj(
            inception_5a_pool_out)
        inception_5a_pool_proj_bn_out = self.inception_5a_pool_proj_bn(
            inception_5a_pool_proj_out)
        inception_5a_relu_pool_proj_out = self.inception_5a_relu_pool_proj(
            inception_5a_pool_proj_bn_out)
        inception_5a_output_out = torch.cat([inception_5a_1x1_bn_out,
            inception_5a_3x3_bn_out, inception_5a_double_3x3_2_bn_out,
            inception_5a_pool_proj_bn_out], 1)
        inception_5b_1x1_out = self.inception_5b_1x1(inception_5a_output_out)
        inception_5b_1x1_bn_out = self.inception_5b_1x1_bn(inception_5b_1x1_out
            )
        inception_5b_relu_1x1_out = self.inception_5b_relu_1x1(
            inception_5b_1x1_bn_out)
        inception_5b_3x3_reduce_out = self.inception_5b_3x3_reduce(
            inception_5a_output_out)
        inception_5b_3x3_reduce_bn_out = self.inception_5b_3x3_reduce_bn(
            inception_5b_3x3_reduce_out)
        inception_5b_relu_3x3_reduce_out = self.inception_5b_relu_3x3_reduce(
            inception_5b_3x3_reduce_bn_out)
        inception_5b_3x3_out = self.inception_5b_3x3(
            inception_5b_3x3_reduce_bn_out)
        inception_5b_3x3_bn_out = self.inception_5b_3x3_bn(inception_5b_3x3_out
            )
        inception_5b_relu_3x3_out = self.inception_5b_relu_3x3(
            inception_5b_3x3_bn_out)
        inception_5b_double_3x3_reduce_out = (self.
            inception_5b_double_3x3_reduce(inception_5a_output_out))
        inception_5b_double_3x3_reduce_bn_out = (self.
            inception_5b_double_3x3_reduce_bn(
            inception_5b_double_3x3_reduce_out))
        inception_5b_relu_double_3x3_reduce_out = (self.
            inception_5b_relu_double_3x3_reduce(
            inception_5b_double_3x3_reduce_bn_out))
        inception_5b_double_3x3_1_out = self.inception_5b_double_3x3_1(
            inception_5b_double_3x3_reduce_bn_out)
        inception_5b_double_3x3_1_bn_out = self.inception_5b_double_3x3_1_bn(
            inception_5b_double_3x3_1_out)
        inception_5b_relu_double_3x3_1_out = (self.
            inception_5b_relu_double_3x3_1(inception_5b_double_3x3_1_bn_out))
        inception_5b_double_3x3_2_out = self.inception_5b_double_3x3_2(
            inception_5b_double_3x3_1_bn_out)
        inception_5b_double_3x3_2_bn_out = self.inception_5b_double_3x3_2_bn(
            inception_5b_double_3x3_2_out)
        inception_5b_relu_double_3x3_2_out = (self.
            inception_5b_relu_double_3x3_2(inception_5b_double_3x3_2_bn_out))
        inception_5b_pool_out = self.inception_5b_pool(inception_5a_output_out)
        inception_5b_pool_proj_out = self.inception_5b_pool_proj(
            inception_5b_pool_out)
        inception_5b_pool_proj_bn_out = self.inception_5b_pool_proj_bn(
            inception_5b_pool_proj_out)
        inception_5b_relu_pool_proj_out = self.inception_5b_relu_pool_proj(
            inception_5b_pool_proj_bn_out)
        inception_5b_output_out = torch.cat([inception_5b_1x1_bn_out,
            inception_5b_3x3_bn_out, inception_5b_double_3x3_2_bn_out,
            inception_5b_pool_proj_bn_out], 1)
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


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, bias=True)


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
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=
            stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

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
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0,
            ceil_mode=True)
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
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x


class _DenseLayer(nn.Sequential):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm.1', nn.BatchNorm3d(num_input_features))
        self.add_module('relu.1', nn.ReLU(inplace=True))
        self.add_module('conv.1', nn.Conv3d(num_input_features, bn_size *
            growth_rate, kernel_size=1, stride=1, bias=False))
        self.add_module('norm.2', nn.BatchNorm3d(bn_size * growth_rate))
        self.add_module('relu.2', nn.ReLU(inplace=True))
        self.add_module('conv.2', nn.Conv3d(bn_size * growth_rate,
            growth_rate, kernel_size=3, stride=1, padding=1, bias=False))
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate,
        drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate,
                growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):

    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(num_input_features,
            num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    """Densenet-BC model class
    Args:
        growth_rate (int) - how many filters to add each layer (k in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, sample_size, sample_duration, growth_rate=32,
        block_config=(6, 12, 24, 16), num_init_features=64, bn_size=4,
        drop_rate=0, num_classes=1000):
        super(DenseNet, self).__init__()
        self.sample_size = sample_size
        self.sample_duration = sample_duration
        self.features = nn.Sequential(OrderedDict([('conv0', nn.Conv3d(3,
            num_init_features, kernel_size=7, stride=(1, 2, 2), padding=(3,
            3, 3), bias=False)), ('norm0', nn.BatchNorm3d(num_init_features
            )), ('relu0', nn.ReLU(inplace=True)), ('pool0', nn.MaxPool3d(
            kernel_size=3, stride=2, padding=1))]))
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=
                num_features, bn_size=bn_size, growth_rate=growth_rate,
                drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d
                ):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        last_duration = int(math.ceil(self.sample_duration / 16))
        last_size = int(math.floor(self.sample_size / 32))
        out = F.avg_pool3d(out, kernel_size=(last_duration, last_size,
            last_size)).view(features.size(0), -1)
        out = self.classifier(out)
        return out


class CatBnAct(nn.Module):

    def __init__(self, in_chs, activation_fn=nn.ReLU(inplace=True)):
        super(CatBnAct, self).__init__()
        self.bn = nn.BatchNorm2d(in_chs, eps=0.001)
        self.act = activation_fn

    def forward(self, x):
        x = torch.cat(x, dim=1) if isinstance(x, tuple) else x
        return self.act(self.bn(x))


class BnActConv2d(nn.Module):

    def __init__(self, in_chs, out_chs, kernel_size, stride, padding=0,
        groups=1, activation_fn=nn.ReLU(inplace=True)):
        super(BnActConv2d, self).__init__()
        self.bn = nn.BatchNorm2d(in_chs, eps=0.001)
        self.act = activation_fn
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride, padding,
            groups=groups, bias=False)

    def forward(self, x):
        return self.conv(self.act(self.bn(x)))


class InputBlock(nn.Module):

    def __init__(self, num_init_features, kernel_size=7, padding=3,
        activation_fn=nn.ReLU(inplace=True)):
        super(InputBlock, self).__init__()
        self.conv = nn.Conv2d(3, num_init_features, kernel_size=kernel_size,
            stride=2, padding=padding, bias=False)
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

    def __init__(self, in_chs, num_1x1_a, num_3x3_b, num_1x1_c, inc, groups,
        block_type='normal', b=False):
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
                self.c1x1_w_s2 = BnActConv2d(in_chs=in_chs, out_chs=
                    num_1x1_c + 2 * inc, kernel_size=1, stride=2)
            else:
                self.c1x1_w_s1 = BnActConv2d(in_chs=in_chs, out_chs=
                    num_1x1_c + 2 * inc, kernel_size=1, stride=1)
        self.c1x1_a = BnActConv2d(in_chs=in_chs, out_chs=num_1x1_a,
            kernel_size=1, stride=1)
        self.c3x3_b = BnActConv2d(in_chs=num_1x1_a, out_chs=num_3x3_b,
            kernel_size=3, stride=self.key_stride, padding=1, groups=groups)
        if b:
            self.c1x1_c = CatBnAct(in_chs=num_3x3_b)
            self.c1x1_c1 = nn.Conv2d(num_3x3_b, num_1x1_c, kernel_size=1,
                bias=False)
            self.c1x1_c2 = nn.Conv2d(num_3x3_b, inc, kernel_size=1, bias=False)
        else:
            self.c1x1_c = BnActConv2d(in_chs=num_3x3_b, out_chs=num_1x1_c +
                inc, kernel_size=1, stride=1)

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


def adaptive_avgmax_pool2d(x, pool_type='avg', padding=0, count_include_pad
    =False):
    """Selectable global pooling function with dynamic input kernel size
    """
    if pool_type == 'avgmaxc':
        x = torch.cat([F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)),
            padding=padding, count_include_pad=count_include_pad), F.
            max_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=
            padding)], dim=1)
    elif pool_type == 'avgmax':
        x_avg = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding
            =padding, count_include_pad=count_include_pad)
        x_max = F.max_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding
            =padding)
        x = 0.5 * (x_avg + x_max)
    elif pool_type == 'max':
        x = F.max_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=padding
            )
    else:
        if pool_type != 'avg':
            print(
                'Invalid pool type %s specified. Defaulting to average pooling.'
                 % pool_type)
        x = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=
            padding, count_include_pad=count_include_pad)
    return x


class DPN(nn.Module):

    def __init__(self, small=False, num_init_features=64, k_r=96, groups=32,
        b=False, k_sec=(3, 4, 20, 3), inc_sec=(16, 32, 24, 128),
        num_classes=1000, test_time_pool=False):
        super(DPN, self).__init__()
        self.test_time_pool = test_time_pool
        self.b = b
        bw_factor = 1 if small else 4
        blocks = OrderedDict()
        if small:
            blocks['conv1_1'] = InputBlock(num_init_features, kernel_size=3,
                padding=1)
        else:
            blocks['conv1_1'] = InputBlock(num_init_features, kernel_size=7,
                padding=3)
        bw = 64 * bw_factor
        inc = inc_sec[0]
        r = k_r * bw // (64 * bw_factor)
        blocks['conv2_1'] = DualPathBlock(num_init_features, r, r, bw, inc,
            groups, 'proj', b)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[0] + 1):
            blocks['conv2_' + str(i)] = DualPathBlock(in_chs, r, r, bw, inc,
                groups, 'normal', b)
            in_chs += inc
        bw = 128 * bw_factor
        inc = inc_sec[1]
        r = k_r * bw // (64 * bw_factor)
        blocks['conv3_1'] = DualPathBlock(in_chs, r, r, bw, inc, groups,
            'down', b)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[1] + 1):
            blocks['conv3_' + str(i)] = DualPathBlock(in_chs, r, r, bw, inc,
                groups, 'normal', b)
            in_chs += inc
        bw = 256 * bw_factor
        inc = inc_sec[2]
        r = k_r * bw // (64 * bw_factor)
        blocks['conv4_1'] = DualPathBlock(in_chs, r, r, bw, inc, groups,
            'down', b)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[2] + 1):
            blocks['conv4_' + str(i)] = DualPathBlock(in_chs, r, r, bw, inc,
                groups, 'normal', b)
            in_chs += inc
        bw = 512 * bw_factor
        inc = inc_sec[3]
        r = k_r * bw // (64 * bw_factor)
        blocks['conv5_1'] = DualPathBlock(in_chs, r, r, bw, inc, groups,
            'down', b)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[3] + 1):
            blocks['conv5_' + str(i)] = DualPathBlock(in_chs, r, r, bw, inc,
                groups, 'normal', b)
            in_chs += inc
        blocks['conv5_bn_ac'] = CatBnAct(in_chs)
        self.features = nn.Sequential(blocks)
        self.classifier = nn.Conv2d(in_chs, num_classes, kernel_size=1,
            bias=True)

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
            self.pool = nn.ModuleList([nn.AdaptiveAvgPool2d(output_size),
                nn.AdaptiveMaxPool2d(output_size)])
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
            x = 0.5 * torch.sum(torch.stack([p(x) for p in self.pool]), 0
                ).squeeze(dim=0)
        else:
            x = self.pool(x)
        return x

    def factor(self):
        return pooling_factor(self.pool_type)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + 'output_size=' + str(self.
            output_size) + ', pool_type=' + self.pool_type + ')'


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
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=True)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

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
        out += residual
        out = self.relu(out)
        return out


class FBResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        self.input_space = None
        self.input_size = 299, 299, 3
        self.mean = None
        self.std = None
        super(FBResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
            bias=True)
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
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(planes * block.expansion))
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
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=True)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

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
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
            bias=True)
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
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(planes * block.expansion))
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


class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=
            kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes, eps=0.001, momentum=0.1,
            affine=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Mixed_5b(nn.Module):

    def __init__(self):
        super(Mixed_5b, self).__init__()
        self.branch0 = BasicConv2d(192, 96, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(192, 48, kernel_size=1,
            stride=1), BasicConv2d(48, 64, kernel_size=5, stride=1, padding=2))
        self.branch2 = nn.Sequential(BasicConv2d(192, 64, kernel_size=1,
            stride=1), BasicConv2d(64, 96, kernel_size=3, stride=1, padding
            =1), BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1))
        self.branch3 = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False), BasicConv2d(192, 64, kernel_size=1,
            stride=1))

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
        self.branch1 = nn.Sequential(BasicConv2d(320, 32, kernel_size=1,
            stride=1), BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1))
        self.branch2 = nn.Sequential(BasicConv2d(320, 32, kernel_size=1,
            stride=1), BasicConv2d(32, 48, kernel_size=3, stride=1, padding
            =1), BasicConv2d(48, 64, kernel_size=3, stride=1, padding=1))
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
        self.branch1 = nn.Sequential(BasicConv2d(320, 256, kernel_size=1,
            stride=1), BasicConv2d(256, 256, kernel_size=3, stride=1,
            padding=1), BasicConv2d(256, 384, kernel_size=3, stride=2))
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
        self.branch1 = nn.Sequential(BasicConv2d(1088, 128, kernel_size=1,
            stride=1), BasicConv2d(128, 160, kernel_size=(1, 7), stride=1,
            padding=(0, 3)), BasicConv2d(160, 192, kernel_size=(7, 1),
            stride=1, padding=(3, 0)))
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
        self.branch0 = nn.Sequential(BasicConv2d(1088, 256, kernel_size=1,
            stride=1), BasicConv2d(256, 384, kernel_size=3, stride=2))
        self.branch1 = nn.Sequential(BasicConv2d(1088, 256, kernel_size=1,
            stride=1), BasicConv2d(256, 288, kernel_size=3, stride=2))
        self.branch2 = nn.Sequential(BasicConv2d(1088, 256, kernel_size=1,
            stride=1), BasicConv2d(256, 288, kernel_size=3, stride=1,
            padding=1), BasicConv2d(288, 320, kernel_size=3, stride=2))
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
        self.branch1 = nn.Sequential(BasicConv2d(2080, 192, kernel_size=1,
            stride=1), BasicConv2d(192, 224, kernel_size=(1, 3), stride=1,
            padding=(0, 1)), BasicConv2d(224, 256, kernel_size=(3, 1),
            stride=1, padding=(1, 0)))
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
        self.conv2d_2b = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1
            )
        self.maxpool_3a = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1, stride=1)
        self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3, stride=1)
        self.maxpool_5a = nn.MaxPool2d(3, stride=2)
        self.mixed_5b = Mixed_5b()
        self.repeat = nn.Sequential(Block35(scale=0.17), Block35(scale=0.17
            ), Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17
            ), Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17
            ), Block35(scale=0.17), Block35(scale=0.17))
        self.mixed_6a = Mixed_6a()
        self.repeat_1 = nn.Sequential(Block17(scale=0.1), Block17(scale=0.1
            ), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1),
            Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1),
            Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1),
            Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1),
            Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1),
            Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1))
        self.mixed_7a = Mixed_7a()
        self.repeat_2 = nn.Sequential(Block8(scale=0.2), Block8(scale=0.2),
            Block8(scale=0.2), Block8(scale=0.2), Block8(scale=0.2), Block8
            (scale=0.2), Block8(scale=0.2), Block8(scale=0.2), Block8(scale
            =0.2))
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


class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=
            kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes, eps=0.001, momentum=0.1,
            affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
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
        self.branch0 = nn.Sequential(BasicConv2d(160, 64, kernel_size=1,
            stride=1), BasicConv2d(64, 96, kernel_size=3, stride=1))
        self.branch1 = nn.Sequential(BasicConv2d(160, 64, kernel_size=1,
            stride=1), BasicConv2d(64, 64, kernel_size=(1, 7), stride=1,
            padding=(0, 3)), BasicConv2d(64, 64, kernel_size=(7, 1), stride
            =1, padding=(3, 0)), BasicConv2d(64, 96, kernel_size=(3, 3),
            stride=1))

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
        self.branch1 = nn.Sequential(BasicConv2d(384, 64, kernel_size=1,
            stride=1), BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1))
        self.branch2 = nn.Sequential(BasicConv2d(384, 64, kernel_size=1,
            stride=1), BasicConv2d(64, 96, kernel_size=3, stride=1, padding
            =1), BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1))
        self.branch3 = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False), BasicConv2d(384, 96, kernel_size=1,
            stride=1))

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
        self.branch1 = nn.Sequential(BasicConv2d(384, 192, kernel_size=1,
            stride=1), BasicConv2d(192, 224, kernel_size=3, stride=1,
            padding=1), BasicConv2d(224, 256, kernel_size=3, stride=2))
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
        self.branch1 = nn.Sequential(BasicConv2d(1024, 192, kernel_size=1,
            stride=1), BasicConv2d(192, 224, kernel_size=(1, 7), stride=1,
            padding=(0, 3)), BasicConv2d(224, 256, kernel_size=(7, 1),
            stride=1, padding=(3, 0)))
        self.branch2 = nn.Sequential(BasicConv2d(1024, 192, kernel_size=1,
            stride=1), BasicConv2d(192, 192, kernel_size=(7, 1), stride=1,
            padding=(3, 0)), BasicConv2d(192, 224, kernel_size=(1, 7),
            stride=1, padding=(0, 3)), BasicConv2d(224, 224, kernel_size=(7,
            1), stride=1, padding=(3, 0)), BasicConv2d(224, 256,
            kernel_size=(1, 7), stride=1, padding=(0, 3)))
        self.branch3 = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False), BasicConv2d(1024, 128, kernel_size=1,
            stride=1))

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
        self.branch0 = nn.Sequential(BasicConv2d(1024, 192, kernel_size=1,
            stride=1), BasicConv2d(192, 192, kernel_size=3, stride=2))
        self.branch1 = nn.Sequential(BasicConv2d(1024, 256, kernel_size=1,
            stride=1), BasicConv2d(256, 256, kernel_size=(1, 7), stride=1,
            padding=(0, 3)), BasicConv2d(256, 320, kernel_size=(7, 1),
            stride=1, padding=(3, 0)), BasicConv2d(320, 320, kernel_size=3,
            stride=2))
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
        self.branch1_1a = BasicConv2d(384, 256, kernel_size=(1, 3), stride=
            1, padding=(0, 1))
        self.branch1_1b = BasicConv2d(384, 256, kernel_size=(3, 1), stride=
            1, padding=(1, 0))
        self.branch2_0 = BasicConv2d(1536, 384, kernel_size=1, stride=1)
        self.branch2_1 = BasicConv2d(384, 448, kernel_size=(3, 1), stride=1,
            padding=(1, 0))
        self.branch2_2 = BasicConv2d(448, 512, kernel_size=(1, 3), stride=1,
            padding=(0, 1))
        self.branch2_3a = BasicConv2d(512, 256, kernel_size=(1, 3), stride=
            1, padding=(0, 1))
        self.branch2_3b = BasicConv2d(512, 256, kernel_size=(3, 1), stride=
            1, padding=(1, 0))
        self.branch3 = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False), BasicConv2d(1536, 256, kernel_size=1,
            stride=1))

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
        self.features = nn.Sequential(BasicConv2d(3, 32, kernel_size=3,
            stride=2), BasicConv2d(32, 32, kernel_size=3, stride=1),
            BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1),
            Mixed_3a(), Mixed_4a(), Mixed_5a(), Inception_A(), Inception_A(
            ), Inception_A(), Inception_A(), Reduction_A(), Inception_B(),
            Inception_B(), Inception_B(), Inception_B(), Inception_B(),
            Inception_B(), Inception_B(), Reduction_B(), Inception_C(),
            Inception_C(), Inception_C())
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


class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup
        if expand_ratio == 1:
            self.conv = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, 3,
                stride, 1, groups=hidden_dim, bias=False), nn.BatchNorm2d(
                hidden_dim), nn.ReLU6(inplace=True), nn.Conv2d(hidden_dim,
                oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup))
        else:
            self.conv = nn.Sequential(nn.Conv2d(inp, hidden_dim, 1, 1, 0,
                bias=False), nn.BatchNorm2d(hidden_dim), nn.ReLU6(inplace=
                True), nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1,
                groups=hidden_dim, bias=False), nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True), nn.Conv2d(hidden_dim, oup, 1, 1, 0,
                bias=False), nn.BatchNorm2d(oup))

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


def conv_1x1_bn(inp, oup):
    return nn.Sequential(nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.
        BatchNorm2d(oup), nn.ReLU6(inplace=True))


def conv_bn(inp, oup, stride):
    return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.
        BatchNorm2d(oup), nn.ReLU6(inplace=True))


class MobileNetV2(nn.Module):

    def __init__(self, num_classes=1000, input_size=224, width_mult=1.0):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [[1, 16, 1, 1], [6, 24, 2, 2], [6, 
            32, 3, 2], [6, 64, 4, 2], [6, 96, 3, 1], [6, 160, 3, 2], [6, 
            320, 1, 1]]
        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult
            ) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel,
                        output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel,
                        output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        self.features = nn.Sequential(*self.features)
        self.dropout = nn.Dropout(0.2)
        self.last_linear = nn.Linear(self.last_channel, num_classes)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.dropout(x)
        x = self.last_linear(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class MultiViewConv(nn.Conv2d):
    """TODO

    TODO

    Args:
        in_channels (int): Number of channels in the input tensor
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to the sides of the input during their respective convolutions. Default: 0
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride=
            stride, padding=padding, dilation=dilation, groups=groups, bias
            =bias)
        padding = _triple(padding)
        kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.dilation = _triple(dilation)
        self.channel_shape = out_channels, in_channels // groups
        self.paddings = [(0, padding[1], padding[2]), (padding[0], 0,
            padding[2]), (padding[0], padding[1], 0)]
        self.kernel_sizes = [(1, kernel_size[1], kernel_size[2]), (
            kernel_size[0], 1, kernel_size[2]), (kernel_size[0],
            kernel_size[1], 1)]
        self.linear = nn.Linear(3, 1)

    def forward(self, input):
        x = torch.stack([F.conv3d(input, self.weight.view(*self.
            channel_shape, *kernel_size), self.bias, self.stride, padding,
            self.dilation, self.groups) for kernel_size, padding in zip(
            self.kernel_sizes, self.paddings)], -1)
        x = self.linear(x)[..., 0]
        return x


class MaxPoolPad(nn.Module):

    def __init__(self):
        super(MaxPoolPad, self).__init__()
        self.pad = nn.ZeroPad2d((1, 0, 1, 0))
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        x = self.pad(x)
        x = self.pool(x)
        x = x[:, :, 1:, 1:]
        return x


class AvgPoolPad(nn.Module):

    def __init__(self, stride=2, padding=1):
        super(AvgPoolPad, self).__init__()
        self.pad = nn.ZeroPad2d((1, 0, 1, 0))
        self.pool = nn.AvgPool2d(3, stride=stride, padding=padding,
            count_include_pad=False)

    def forward(self, x):
        x = self.pad(x)
        x = self.pool(x)
        x = x[:, :, 1:, 1:]
        return x


class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, dw_kernel, dw_stride,
        dw_padding, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise_conv2d = nn.Conv2d(in_channels, in_channels,
            dw_kernel, stride=dw_stride, padding=dw_padding, bias=bias,
            groups=in_channels)
        self.pointwise_conv2d = nn.Conv2d(in_channels, out_channels, 1,
            stride=1, bias=bias)

    def forward(self, x):
        x = self.depthwise_conv2d(x)
        x = self.pointwise_conv2d(x)
        return x


class BranchSeparables(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
        padding, bias=False):
        super(BranchSeparables, self).__init__()
        self.relu = nn.ReLU()
        self.separable_1 = SeparableConv2d(in_channels, in_channels,
            kernel_size, stride, padding, bias=bias)
        self.bn_sep_1 = nn.BatchNorm2d(in_channels, eps=0.001, momentum=0.1,
            affine=True)
        self.relu1 = nn.ReLU()
        self.separable_2 = SeparableConv2d(in_channels, out_channels,
            kernel_size, 1, padding, bias=bias)
        self.bn_sep_2 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=
            0.1, affine=True)

    def forward(self, x):
        x = self.relu(x)
        x = self.separable_1(x)
        x = self.bn_sep_1(x)
        x = self.relu1(x)
        x = self.separable_2(x)
        x = self.bn_sep_2(x)
        return x


class BranchSeparablesStem(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
        padding, bias=False):
        super(BranchSeparablesStem, self).__init__()
        self.relu = nn.ReLU()
        self.separable_1 = SeparableConv2d(in_channels, out_channels,
            kernel_size, stride, padding, bias=bias)
        self.bn_sep_1 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=
            0.1, affine=True)
        self.relu1 = nn.ReLU()
        self.separable_2 = SeparableConv2d(out_channels, out_channels,
            kernel_size, 1, padding, bias=bias)
        self.bn_sep_2 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=
            0.1, affine=True)

    def forward(self, x):
        x = self.relu(x)
        x = self.separable_1(x)
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
        self.conv_1x1.add_module('conv', nn.Conv2d(self.stem_filters, self.
            num_filters, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(self.num_filters, eps
            =0.001, momentum=0.1, affine=True))
        self.comb_iter_0_left = BranchSeparables(self.num_filters, self.
            num_filters, 5, 2, 2)
        self.comb_iter_0_right = BranchSeparablesStem(self.stem_filters,
            self.num_filters, 7, 2, 3, bias=False)
        self.comb_iter_1_left = nn.MaxPool2d(3, stride=2, padding=1)
        self.comb_iter_1_right = BranchSeparablesStem(self.stem_filters,
            self.num_filters, 7, 2, 3, bias=False)
        self.comb_iter_2_left = nn.AvgPool2d(3, stride=2, padding=1,
            count_include_pad=False)
        self.comb_iter_2_right = BranchSeparablesStem(self.stem_filters,
            self.num_filters, 5, 2, 2, bias=False)
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False)
        self.comb_iter_4_left = BranchSeparables(self.num_filters, self.
            num_filters, 3, 1, 1, bias=False)
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
        x_out = torch.cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3,
            x_comb_iter_4], 1)
        return x_out


class CellStem1(nn.Module):

    def __init__(self, stem_filters, num_filters):
        super(CellStem1, self).__init__()
        self.num_filters = num_filters
        self.stem_filters = stem_filters
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(2 * self.num_filters,
            self.num_filters, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(self.num_filters, eps
            =0.001, momentum=0.1, affine=True))
        self.relu = nn.ReLU()
        self.path_1 = nn.Sequential()
        self.path_1.add_module('avgpool', nn.AvgPool2d(1, stride=2,
            count_include_pad=False))
        self.path_1.add_module('conv', nn.Conv2d(self.stem_filters, self.
            num_filters // 2, 1, stride=1, bias=False))
        self.path_2 = nn.ModuleList()
        self.path_2.add_module('pad', nn.ZeroPad2d((0, 1, 0, 1)))
        self.path_2.add_module('avgpool', nn.AvgPool2d(1, stride=2,
            count_include_pad=False))
        self.path_2.add_module('conv', nn.Conv2d(self.stem_filters, self.
            num_filters // 2, 1, stride=1, bias=False))
        self.final_path_bn = nn.BatchNorm2d(self.num_filters, eps=0.001,
            momentum=0.1, affine=True)
        self.comb_iter_0_left = BranchSeparables(self.num_filters, self.
            num_filters, 5, 2, 2, bias=False)
        self.comb_iter_0_right = BranchSeparables(self.num_filters, self.
            num_filters, 7, 2, 3, bias=False)
        self.comb_iter_1_left = nn.MaxPool2d(3, stride=2, padding=1)
        self.comb_iter_1_right = BranchSeparables(self.num_filters, self.
            num_filters, 7, 2, 3, bias=False)
        self.comb_iter_2_left = nn.AvgPool2d(3, stride=2, padding=1,
            count_include_pad=False)
        self.comb_iter_2_right = BranchSeparables(self.num_filters, self.
            num_filters, 5, 2, 2, bias=False)
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False)
        self.comb_iter_4_left = BranchSeparables(self.num_filters, self.
            num_filters, 3, 1, 1, bias=False)
        self.comb_iter_4_right = nn.MaxPool2d(3, stride=2, padding=1)

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
        x_out = torch.cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3,
            x_comb_iter_4], 1)
        return x_out


class FirstCell(nn.Module):

    def __init__(self, in_channels_left, out_channels_left,
        in_channels_right, out_channels_right):
        super(FirstCell, self).__init__()
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(in_channels_right,
            out_channels_right, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(out_channels_right,
            eps=0.001, momentum=0.1, affine=True))
        self.relu = nn.ReLU()
        self.path_1 = nn.Sequential()
        self.path_1.add_module('avgpool', nn.AvgPool2d(1, stride=2,
            count_include_pad=False))
        self.path_1.add_module('conv', nn.Conv2d(in_channels_left,
            out_channels_left, 1, stride=1, bias=False))
        self.path_2 = nn.ModuleList()
        self.path_2.add_module('pad', nn.ZeroPad2d((0, 1, 0, 1)))
        self.path_2.add_module('avgpool', nn.AvgPool2d(1, stride=2,
            count_include_pad=False))
        self.path_2.add_module('conv', nn.Conv2d(in_channels_left,
            out_channels_left, 1, stride=1, bias=False))
        self.final_path_bn = nn.BatchNorm2d(out_channels_left * 2, eps=
            0.001, momentum=0.1, affine=True)
        self.comb_iter_0_left = BranchSeparables(out_channels_right,
            out_channels_right, 5, 1, 2, bias=False)
        self.comb_iter_0_right = BranchSeparables(out_channels_right,
            out_channels_right, 3, 1, 1, bias=False)
        self.comb_iter_1_left = BranchSeparables(out_channels_right,
            out_channels_right, 5, 1, 2, bias=False)
        self.comb_iter_1_right = BranchSeparables(out_channels_right,
            out_channels_right, 3, 1, 1, bias=False)
        self.comb_iter_2_left = nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False)
        self.comb_iter_3_left = nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False)
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False)
        self.comb_iter_4_left = BranchSeparables(out_channels_right,
            out_channels_right, 3, 1, 1, bias=False)

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
        x_out = torch.cat([x_left, x_comb_iter_0, x_comb_iter_1,
            x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class NormalCell(nn.Module):

    def __init__(self, in_channels_left, out_channels_left,
        in_channels_right, out_channels_right):
        super(NormalCell, self).__init__()
        self.conv_prev_1x1 = nn.Sequential()
        self.conv_prev_1x1.add_module('relu', nn.ReLU())
        self.conv_prev_1x1.add_module('conv', nn.Conv2d(in_channels_left,
            out_channels_left, 1, stride=1, bias=False))
        self.conv_prev_1x1.add_module('bn', nn.BatchNorm2d(
            out_channels_left, eps=0.001, momentum=0.1, affine=True))
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(in_channels_right,
            out_channels_right, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(out_channels_right,
            eps=0.001, momentum=0.1, affine=True))
        self.comb_iter_0_left = BranchSeparables(out_channels_right,
            out_channels_right, 5, 1, 2, bias=False)
        self.comb_iter_0_right = BranchSeparables(out_channels_left,
            out_channels_left, 3, 1, 1, bias=False)
        self.comb_iter_1_left = BranchSeparables(out_channels_left,
            out_channels_left, 5, 1, 2, bias=False)
        self.comb_iter_1_right = BranchSeparables(out_channels_left,
            out_channels_left, 3, 1, 1, bias=False)
        self.comb_iter_2_left = nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False)
        self.comb_iter_3_left = nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False)
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False)
        self.comb_iter_4_left = BranchSeparables(out_channels_right,
            out_channels_right, 3, 1, 1, bias=False)

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
        x_out = torch.cat([x_left, x_comb_iter_0, x_comb_iter_1,
            x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class BranchSeparablesReduction(BranchSeparables):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
        padding, z_padding=1, bias=False):
        BranchSeparables.__init__(self, in_channels, out_channels,
            kernel_size, stride, padding, bias)
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


class ReductionCell0(nn.Module):

    def __init__(self, in_channels_left, out_channels_left,
        in_channels_right, out_channels_right):
        super(ReductionCell0, self).__init__()
        self.conv_prev_1x1 = nn.Sequential()
        self.conv_prev_1x1.add_module('relu', nn.ReLU())
        self.conv_prev_1x1.add_module('conv', nn.Conv2d(in_channels_left,
            out_channels_left, 1, stride=1, bias=False))
        self.conv_prev_1x1.add_module('bn', nn.BatchNorm2d(
            out_channels_left, eps=0.001, momentum=0.1, affine=True))
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(in_channels_right,
            out_channels_right, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(out_channels_right,
            eps=0.001, momentum=0.1, affine=True))
        self.comb_iter_0_left = BranchSeparablesReduction(out_channels_right,
            out_channels_right, 5, 2, 2, bias=False)
        self.comb_iter_0_right = BranchSeparablesReduction(out_channels_right,
            out_channels_right, 7, 2, 3, bias=False)
        self.comb_iter_1_left = MaxPoolPad()
        self.comb_iter_1_right = BranchSeparablesReduction(out_channels_right,
            out_channels_right, 7, 2, 3, bias=False)
        self.comb_iter_2_left = AvgPoolPad()
        self.comb_iter_2_right = BranchSeparablesReduction(out_channels_right,
            out_channels_right, 5, 2, 2, bias=False)
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False)
        self.comb_iter_4_left = BranchSeparablesReduction(out_channels_right,
            out_channels_right, 3, 1, 1, bias=False)
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
        x_out = torch.cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3,
            x_comb_iter_4], 1)
        return x_out


class ReductionCell1(nn.Module):

    def __init__(self, in_channels_left, out_channels_left,
        in_channels_right, out_channels_right):
        super(ReductionCell1, self).__init__()
        self.conv_prev_1x1 = nn.Sequential()
        self.conv_prev_1x1.add_module('relu', nn.ReLU())
        self.conv_prev_1x1.add_module('conv', nn.Conv2d(in_channels_left,
            out_channels_left, 1, stride=1, bias=False))
        self.conv_prev_1x1.add_module('bn', nn.BatchNorm2d(
            out_channels_left, eps=0.001, momentum=0.1, affine=True))
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(in_channels_right,
            out_channels_right, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(out_channels_right,
            eps=0.001, momentum=0.1, affine=True))
        self.comb_iter_0_left = BranchSeparables(out_channels_right,
            out_channels_right, 5, 2, 2, bias=False)
        self.comb_iter_0_right = BranchSeparables(out_channels_right,
            out_channels_right, 7, 2, 3, bias=False)
        self.comb_iter_1_left = nn.MaxPool2d(3, stride=2, padding=1)
        self.comb_iter_1_right = BranchSeparables(out_channels_right,
            out_channels_right, 7, 2, 3, bias=False)
        self.comb_iter_2_left = nn.AvgPool2d(3, stride=2, padding=1,
            count_include_pad=False)
        self.comb_iter_2_right = BranchSeparables(out_channels_right,
            out_channels_right, 5, 2, 2, bias=False)
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False)
        self.comb_iter_4_left = BranchSeparables(out_channels_right,
            out_channels_right, 3, 1, 1, bias=False)
        self.comb_iter_4_right = nn.MaxPool2d(3, stride=2, padding=1)

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
        x_out = torch.cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3,
            x_comb_iter_4], 1)
        return x_out


class NASNetALarge(nn.Module):
    """NASNetALarge (6 @ 4032) """

    def __init__(self, num_classes=1001, stem_filters=96,
        penultimate_filters=4032, filters_multiplier=2):
        super(NASNetALarge, self).__init__()
        self.num_classes = num_classes
        self.stem_filters = stem_filters
        self.penultimate_filters = penultimate_filters
        self.filters_multiplier = filters_multiplier
        filters = self.penultimate_filters // 24
        self.conv0 = nn.Sequential()
        self.conv0.add_module('conv', nn.Conv2d(in_channels=3, out_channels
            =self.stem_filters, kernel_size=3, padding=0, stride=2, bias=False)
            )
        self.conv0.add_module('bn', nn.BatchNorm2d(self.stem_filters, eps=
            0.001, momentum=0.1, affine=True))
        self.cell_stem_0 = CellStem0(self.stem_filters, num_filters=filters //
            filters_multiplier ** 2)
        self.cell_stem_1 = CellStem1(self.stem_filters, num_filters=filters //
            filters_multiplier)
        self.cell_0 = FirstCell(in_channels_left=filters, out_channels_left
            =filters // 2, in_channels_right=2 * filters,
            out_channels_right=filters)
        self.cell_1 = NormalCell(in_channels_left=2 * filters,
            out_channels_left=filters, in_channels_right=6 * filters,
            out_channels_right=filters)
        self.cell_2 = NormalCell(in_channels_left=6 * filters,
            out_channels_left=filters, in_channels_right=6 * filters,
            out_channels_right=filters)
        self.cell_3 = NormalCell(in_channels_left=6 * filters,
            out_channels_left=filters, in_channels_right=6 * filters,
            out_channels_right=filters)
        self.cell_4 = NormalCell(in_channels_left=6 * filters,
            out_channels_left=filters, in_channels_right=6 * filters,
            out_channels_right=filters)
        self.cell_5 = NormalCell(in_channels_left=6 * filters,
            out_channels_left=filters, in_channels_right=6 * filters,
            out_channels_right=filters)
        self.reduction_cell_0 = ReductionCell0(in_channels_left=6 * filters,
            out_channels_left=2 * filters, in_channels_right=6 * filters,
            out_channels_right=2 * filters)
        self.cell_6 = FirstCell(in_channels_left=6 * filters,
            out_channels_left=filters, in_channels_right=8 * filters,
            out_channels_right=2 * filters)
        self.cell_7 = NormalCell(in_channels_left=8 * filters,
            out_channels_left=2 * filters, in_channels_right=12 * filters,
            out_channels_right=2 * filters)
        self.cell_8 = NormalCell(in_channels_left=12 * filters,
            out_channels_left=2 * filters, in_channels_right=12 * filters,
            out_channels_right=2 * filters)
        self.cell_9 = NormalCell(in_channels_left=12 * filters,
            out_channels_left=2 * filters, in_channels_right=12 * filters,
            out_channels_right=2 * filters)
        self.cell_10 = NormalCell(in_channels_left=12 * filters,
            out_channels_left=2 * filters, in_channels_right=12 * filters,
            out_channels_right=2 * filters)
        self.cell_11 = NormalCell(in_channels_left=12 * filters,
            out_channels_left=2 * filters, in_channels_right=12 * filters,
            out_channels_right=2 * filters)
        self.reduction_cell_1 = ReductionCell1(in_channels_left=12 *
            filters, out_channels_left=4 * filters, in_channels_right=12 *
            filters, out_channels_right=4 * filters)
        self.cell_12 = FirstCell(in_channels_left=12 * filters,
            out_channels_left=2 * filters, in_channels_right=16 * filters,
            out_channels_right=4 * filters)
        self.cell_13 = NormalCell(in_channels_left=16 * filters,
            out_channels_left=4 * filters, in_channels_right=24 * filters,
            out_channels_right=4 * filters)
        self.cell_14 = NormalCell(in_channels_left=24 * filters,
            out_channels_left=4 * filters, in_channels_right=24 * filters,
            out_channels_right=4 * filters)
        self.cell_15 = NormalCell(in_channels_left=24 * filters,
            out_channels_left=4 * filters, in_channels_right=24 * filters,
            out_channels_right=4 * filters)
        self.cell_16 = NormalCell(in_channels_left=24 * filters,
            out_channels_left=4 * filters, in_channels_right=24 * filters,
            out_channels_right=4 * filters)
        self.cell_17 = NormalCell(in_channels_left=24 * filters,
            out_channels_left=4 * filters, in_channels_right=24 * filters,
            out_channels_right=4 * filters)
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
        self.pool = nn.AvgPool2d(3, stride=stride, padding=padding,
            count_include_pad=False)

    def forward(self, x):
        x = self.pad(x)
        x = self.pool(x)
        x = x[:, :, 1:, 1:].contiguous()
        return x


class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, dw_kernel, dw_stride,
        dw_padding, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise_conv2d = nn.Conv2d(in_channels, in_channels,
            dw_kernel, stride=dw_stride, padding=dw_padding, bias=bias,
            groups=in_channels)
        self.pointwise_conv2d = nn.Conv2d(in_channels, out_channels, 1,
            stride=1, bias=bias)

    def forward(self, x):
        x = self.depthwise_conv2d(x)
        x = self.pointwise_conv2d(x)
        return x


class BranchSeparables(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
        padding, name=None, bias=False):
        super(BranchSeparables, self).__init__()
        self.relu = nn.ReLU()
        self.separable_1 = SeparableConv2d(in_channels, in_channels,
            kernel_size, stride, padding, bias=bias)
        self.bn_sep_1 = nn.BatchNorm2d(in_channels, eps=0.001, momentum=0.1,
            affine=True)
        self.relu1 = nn.ReLU()
        self.separable_2 = SeparableConv2d(in_channels, out_channels,
            kernel_size, 1, padding, bias=bias)
        self.bn_sep_2 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=
            0.1, affine=True)
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

    def __init__(self, in_channels, out_channels, kernel_size, stride,
        padding, bias=False):
        super(BranchSeparablesStem, self).__init__()
        self.relu = nn.ReLU()
        self.separable_1 = SeparableConv2d(in_channels, out_channels,
            kernel_size, stride, padding, bias=bias)
        self.bn_sep_1 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=
            0.1, affine=True)
        self.relu1 = nn.ReLU()
        self.separable_2 = SeparableConv2d(out_channels, out_channels,
            kernel_size, 1, padding, bias=bias)
        self.bn_sep_2 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=
            0.1, affine=True)

    def forward(self, x):
        x = self.relu(x)
        x = self.separable_1(x)
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
        self.conv_1x1.add_module('conv', nn.Conv2d(self.stem_filters, self.
            num_filters, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(self.num_filters, eps
            =0.001, momentum=0.1, affine=True))
        self.comb_iter_0_left = BranchSeparables(self.num_filters, self.
            num_filters, 5, 2, 2)
        self.comb_iter_0_right = BranchSeparablesStem(self.stem_filters,
            self.num_filters, 7, 2, 3, bias=False)
        self.comb_iter_1_left = nn.MaxPool2d(3, stride=2, padding=1)
        self.comb_iter_1_right = BranchSeparablesStem(self.stem_filters,
            self.num_filters, 7, 2, 3, bias=False)
        self.comb_iter_2_left = nn.AvgPool2d(3, stride=2, padding=1,
            count_include_pad=False)
        self.comb_iter_2_right = BranchSeparablesStem(self.stem_filters,
            self.num_filters, 5, 2, 2, bias=False)
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False)
        self.comb_iter_4_left = BranchSeparables(self.num_filters, self.
            num_filters, 3, 1, 1, bias=False)
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
        x_out = torch.cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3,
            x_comb_iter_4], 1)
        return x_out


class CellStem1(nn.Module):

    def __init__(self, stem_filters, num_filters):
        super(CellStem1, self).__init__()
        self.num_filters = num_filters
        self.stem_filters = stem_filters
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(2 * self.num_filters,
            self.num_filters, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(self.num_filters, eps
            =0.001, momentum=0.1, affine=True))
        self.relu = nn.ReLU()
        self.path_1 = nn.Sequential()
        self.path_1.add_module('avgpool', nn.AvgPool2d(1, stride=2,
            count_include_pad=False))
        self.path_1.add_module('conv', nn.Conv2d(self.stem_filters, self.
            num_filters // 2, 1, stride=1, bias=False))
        self.path_2 = nn.ModuleList()
        self.path_2.add_module('pad', nn.ZeroPad2d((0, 1, 0, 1)))
        self.path_2.add_module('avgpool', nn.AvgPool2d(1, stride=2,
            count_include_pad=False))
        self.path_2.add_module('conv', nn.Conv2d(self.stem_filters, self.
            num_filters // 2, 1, stride=1, bias=False))
        self.final_path_bn = nn.BatchNorm2d(self.num_filters, eps=0.001,
            momentum=0.1, affine=True)
        self.comb_iter_0_left = BranchSeparables(self.num_filters, self.
            num_filters, 5, 2, 2, name='specific', bias=False)
        self.comb_iter_0_right = BranchSeparables(self.num_filters, self.
            num_filters, 7, 2, 3, name='specific', bias=False)
        self.comb_iter_1_left = MaxPoolPad()
        self.comb_iter_1_right = BranchSeparables(self.num_filters, self.
            num_filters, 7, 2, 3, name='specific', bias=False)
        self.comb_iter_2_left = AvgPoolPad()
        self.comb_iter_2_right = BranchSeparables(self.num_filters, self.
            num_filters, 5, 2, 2, name='specific', bias=False)
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False)
        self.comb_iter_4_left = BranchSeparables(self.num_filters, self.
            num_filters, 3, 1, 1, name='specific', bias=False)
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
        x_out = torch.cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3,
            x_comb_iter_4], 1)
        return x_out


class FirstCell(nn.Module):

    def __init__(self, in_channels_left, out_channels_left,
        in_channels_right, out_channels_right):
        super(FirstCell, self).__init__()
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(in_channels_right,
            out_channels_right, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(out_channels_right,
            eps=0.001, momentum=0.1, affine=True))
        self.relu = nn.ReLU()
        self.path_1 = nn.Sequential()
        self.path_1.add_module('avgpool', nn.AvgPool2d(1, stride=2,
            count_include_pad=False))
        self.path_1.add_module('conv', nn.Conv2d(in_channels_left,
            out_channels_left, 1, stride=1, bias=False))
        self.path_2 = nn.ModuleList()
        self.path_2.add_module('pad', nn.ZeroPad2d((0, 1, 0, 1)))
        self.path_2.add_module('avgpool', nn.AvgPool2d(1, stride=2,
            count_include_pad=False))
        self.path_2.add_module('conv', nn.Conv2d(in_channels_left,
            out_channels_left, 1, stride=1, bias=False))
        self.final_path_bn = nn.BatchNorm2d(out_channels_left * 2, eps=
            0.001, momentum=0.1, affine=True)
        self.comb_iter_0_left = BranchSeparables(out_channels_right,
            out_channels_right, 5, 1, 2, bias=False)
        self.comb_iter_0_right = BranchSeparables(out_channels_right,
            out_channels_right, 3, 1, 1, bias=False)
        self.comb_iter_1_left = BranchSeparables(out_channels_right,
            out_channels_right, 5, 1, 2, bias=False)
        self.comb_iter_1_right = BranchSeparables(out_channels_right,
            out_channels_right, 3, 1, 1, bias=False)
        self.comb_iter_2_left = nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False)
        self.comb_iter_3_left = nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False)
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False)
        self.comb_iter_4_left = BranchSeparables(out_channels_right,
            out_channels_right, 3, 1, 1, bias=False)

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
        x_out = torch.cat([x_left, x_comb_iter_0, x_comb_iter_1,
            x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class NormalCell(nn.Module):

    def __init__(self, in_channels_left, out_channels_left,
        in_channels_right, out_channels_right):
        super(NormalCell, self).__init__()
        self.conv_prev_1x1 = nn.Sequential()
        self.conv_prev_1x1.add_module('relu', nn.ReLU())
        self.conv_prev_1x1.add_module('conv', nn.Conv2d(in_channels_left,
            out_channels_left, 1, stride=1, bias=False))
        self.conv_prev_1x1.add_module('bn', nn.BatchNorm2d(
            out_channels_left, eps=0.001, momentum=0.1, affine=True))
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(in_channels_right,
            out_channels_right, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(out_channels_right,
            eps=0.001, momentum=0.1, affine=True))
        self.comb_iter_0_left = BranchSeparables(out_channels_right,
            out_channels_right, 5, 1, 2, bias=False)
        self.comb_iter_0_right = BranchSeparables(out_channels_left,
            out_channels_left, 3, 1, 1, bias=False)
        self.comb_iter_1_left = BranchSeparables(out_channels_left,
            out_channels_left, 5, 1, 2, bias=False)
        self.comb_iter_1_right = BranchSeparables(out_channels_left,
            out_channels_left, 3, 1, 1, bias=False)
        self.comb_iter_2_left = nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False)
        self.comb_iter_3_left = nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False)
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False)
        self.comb_iter_4_left = BranchSeparables(out_channels_right,
            out_channels_right, 3, 1, 1, bias=False)

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
        x_out = torch.cat([x_left, x_comb_iter_0, x_comb_iter_1,
            x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class ReductionCell0(nn.Module):

    def __init__(self, in_channels_left, out_channels_left,
        in_channels_right, out_channels_right):
        super(ReductionCell0, self).__init__()
        self.conv_prev_1x1 = nn.Sequential()
        self.conv_prev_1x1.add_module('relu', nn.ReLU())
        self.conv_prev_1x1.add_module('conv', nn.Conv2d(in_channels_left,
            out_channels_left, 1, stride=1, bias=False))
        self.conv_prev_1x1.add_module('bn', nn.BatchNorm2d(
            out_channels_left, eps=0.001, momentum=0.1, affine=True))
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(in_channels_right,
            out_channels_right, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(out_channels_right,
            eps=0.001, momentum=0.1, affine=True))
        self.comb_iter_0_left = BranchSeparablesReduction(out_channels_right,
            out_channels_right, 5, 2, 2, bias=False)
        self.comb_iter_0_right = BranchSeparablesReduction(out_channels_right,
            out_channels_right, 7, 2, 3, bias=False)
        self.comb_iter_1_left = MaxPoolPad()
        self.comb_iter_1_right = BranchSeparablesReduction(out_channels_right,
            out_channels_right, 7, 2, 3, bias=False)
        self.comb_iter_2_left = AvgPoolPad()
        self.comb_iter_2_right = BranchSeparablesReduction(out_channels_right,
            out_channels_right, 5, 2, 2, bias=False)
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False)
        self.comb_iter_4_left = BranchSeparablesReduction(out_channels_right,
            out_channels_right, 3, 1, 1, bias=False)
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
        x_out = torch.cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3,
            x_comb_iter_4], 1)
        return x_out


class ReductionCell1(nn.Module):

    def __init__(self, in_channels_left, out_channels_left,
        in_channels_right, out_channels_right):
        super(ReductionCell1, self).__init__()
        self.conv_prev_1x1 = nn.Sequential()
        self.conv_prev_1x1.add_module('relu', nn.ReLU())
        self.conv_prev_1x1.add_module('conv', nn.Conv2d(in_channels_left,
            out_channels_left, 1, stride=1, bias=False))
        self.conv_prev_1x1.add_module('bn', nn.BatchNorm2d(
            out_channels_left, eps=0.001, momentum=0.1, affine=True))
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(in_channels_right,
            out_channels_right, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(out_channels_right,
            eps=0.001, momentum=0.1, affine=True))
        self.comb_iter_0_left = BranchSeparables(out_channels_right,
            out_channels_right, 5, 2, 2, name='specific', bias=False)
        self.comb_iter_0_right = BranchSeparables(out_channels_right,
            out_channels_right, 7, 2, 3, name='specific', bias=False)
        self.comb_iter_1_left = MaxPoolPad()
        self.comb_iter_1_right = BranchSeparables(out_channels_right,
            out_channels_right, 7, 2, 3, name='specific', bias=False)
        self.comb_iter_2_left = AvgPoolPad()
        self.comb_iter_2_right = BranchSeparables(out_channels_right,
            out_channels_right, 5, 2, 2, name='specific', bias=False)
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False)
        self.comb_iter_4_left = BranchSeparables(out_channels_right,
            out_channels_right, 3, 1, 1, name='specific', bias=False)
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
        x_out = torch.cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3,
            x_comb_iter_4], 1)
        return x_out


class NASNetAMobile(nn.Module):
    """NASNetAMobile (4 @ 1056) """

    def __init__(self, num_classes=1001, stem_filters=32,
        penultimate_filters=1056, filters_multiplier=2):
        super(NASNetAMobile, self).__init__()
        self.num_classes = num_classes
        self.stem_filters = stem_filters
        self.penultimate_filters = penultimate_filters
        self.filters_multiplier = filters_multiplier
        filters = self.penultimate_filters // 24
        self.conv0 = nn.Sequential()
        self.conv0.add_module('conv', nn.Conv2d(in_channels=3, out_channels
            =self.stem_filters, kernel_size=3, padding=0, stride=2, bias=False)
            )
        self.conv0.add_module('bn', nn.BatchNorm2d(self.stem_filters, eps=
            0.001, momentum=0.1, affine=True))
        self.cell_stem_0 = CellStem0(self.stem_filters, num_filters=filters //
            filters_multiplier ** 2)
        self.cell_stem_1 = CellStem1(self.stem_filters, num_filters=filters //
            filters_multiplier)
        self.cell_0 = FirstCell(in_channels_left=filters, out_channels_left
            =filters // 2, in_channels_right=2 * filters,
            out_channels_right=filters)
        self.cell_1 = NormalCell(in_channels_left=2 * filters,
            out_channels_left=filters, in_channels_right=6 * filters,
            out_channels_right=filters)
        self.cell_2 = NormalCell(in_channels_left=6 * filters,
            out_channels_left=filters, in_channels_right=6 * filters,
            out_channels_right=filters)
        self.cell_3 = NormalCell(in_channels_left=6 * filters,
            out_channels_left=filters, in_channels_right=6 * filters,
            out_channels_right=filters)
        self.reduction_cell_0 = ReductionCell0(in_channels_left=6 * filters,
            out_channels_left=2 * filters, in_channels_right=6 * filters,
            out_channels_right=2 * filters)
        self.cell_6 = FirstCell(in_channels_left=6 * filters,
            out_channels_left=filters, in_channels_right=8 * filters,
            out_channels_right=2 * filters)
        self.cell_7 = NormalCell(in_channels_left=8 * filters,
            out_channels_left=2 * filters, in_channels_right=12 * filters,
            out_channels_right=2 * filters)
        self.cell_8 = NormalCell(in_channels_left=12 * filters,
            out_channels_left=2 * filters, in_channels_right=12 * filters,
            out_channels_right=2 * filters)
        self.cell_9 = NormalCell(in_channels_left=12 * filters,
            out_channels_left=2 * filters, in_channels_right=12 * filters,
            out_channels_right=2 * filters)
        self.reduction_cell_1 = ReductionCell1(in_channels_left=12 *
            filters, out_channels_left=4 * filters, in_channels_right=12 *
            filters, out_channels_right=4 * filters)
        self.cell_12 = FirstCell(in_channels_left=12 * filters,
            out_channels_left=2 * filters, in_channels_right=16 * filters,
            out_channels_right=4 * filters)
        self.cell_13 = NormalCell(in_channels_left=16 * filters,
            out_channels_left=4 * filters, in_channels_right=24 * filters,
            out_channels_right=4 * filters)
        self.cell_14 = NormalCell(in_channels_left=24 * filters,
            out_channels_left=4 * filters, in_channels_right=24 * filters,
            out_channels_right=4 * filters)
        self.cell_15 = NormalCell(in_channels_left=24 * filters,
            out_channels_left=4 * filters, in_channels_right=24 * filters,
            out_channels_right=4 * filters)
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


class _NonLocalBlockND(nn.Module):

    def __init__(self, in_channels, inter_channels=None, dimension=3, mode=
        'embedded_gaussian', sub_sample=False, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()
        assert dimension in [1, 2, 3]
        assert mode in ['embedded_gaussian', 'gaussian', 'dot_product',
            'concatenation']
        self.mode = mode
        self.dimension = dimension
        self.sub_sample = sub_sample
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1
        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool = nn.MaxPool3d
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool = nn.MaxPool2d
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool = nn.MaxPool1d
            bn = nn.BatchNorm1d
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.
            inter_channels, kernel_size=1, stride=1, padding=0)
        if bn_layer:
            self.W = nn.Sequential(conv_nd(in_channels=self.inter_channels,
                out_channels=self.in_channels, kernel_size=1, stride=1,
                padding=0), bn(self.in_channels))
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=
                self.in_channels, kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)
        self.theta = None
        self.phi = None
        self.concat_project = None
        if mode in ['embedded_gaussian', 'dot_product', 'concatenation']:
            self.theta = conv_nd(in_channels=self.in_channels, out_channels
                =self.inter_channels, kernel_size=1, stride=1, padding=0)
            self.phi = conv_nd(in_channels=self.in_channels, out_channels=
                self.inter_channels, kernel_size=1, stride=1, padding=0)
            if mode == 'embedded_gaussian':
                self.operation_function = self._embedded_gaussian
            elif mode == 'dot_product':
                self.operation_function = self._dot_product
            elif mode == 'concatenation':
                self.operation_function = self._concatenation
                self.concat_project = nn.Sequential(nn.Conv2d(self.
                    inter_channels * 2, 1, 1, 1, 0, bias=False), nn.ReLU())
        elif mode == 'gaussian':
            self.operation_function = self._gaussian
        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool(kernel_size=2))
            if self.phi is None:
                self.phi = max_pool(kernel_size=2)
            else:
                self.phi = nn.Sequential(self.phi, max_pool(kernel_size=2))

    def forward(self, x):
        """
        :param x: (b, c, t, h, w)
        :return:
        """
        func = getattr(self, f'_{self.mode}')
        output = func(x)
        return output

    def _embedded_gaussian(self, x):
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        return z

    def _gaussian(self, x):
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        theta_x = x.view(batch_size, self.in_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        if self.sub_sample:
            phi_x = self.phi(x).view(batch_size, self.in_channels, -1)
        else:
            phi_x = x.view(batch_size, self.in_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        return z

    def _dot_product(self, x):
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        return z

    def _concatenation(self, x):
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, 1, -1)
        h = theta_x.size(2)
        w = phi_x.size(3)
        theta_x = theta_x.repeat(1, 1, 1, w)
        phi_x = phi_x.repeat(1, 1, h, 1)
        concat_feature = torch.cat([theta_x, phi_x], dim=1)
        f = self.concat_project(concat_feature)
        b, _, h, w = f.size()
        f = f.view(b, h, w)
        N = f.size(-1)
        f_div_C = f / N
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        return z


class NonLocalBlock2D(_NonLocalBlockND):

    def __init__(self, in_channels, inter_channels=None, mode=
        'embedded_gaussian', sub_sample=False, bn_layer=True):
        super(NonLocalBlock2D, self).__init__(in_channels, inter_channels=
            inter_channels, dimension=2, mode=mode, sub_sample=sub_sample,
            bn_layer=bn_layer)


class MNISTNonLocalNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.convs = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=32,
            kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(32), nn.
            ReLU(), nn.MaxPool2d(2), NonLocalBlock2D(in_channels=32), nn.
            Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1,
            padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            NonLocalBlock2D(in_channels=64), nn.Conv2d(in_channels=64,
            out_channels=128, kernel_size=3, stride=1, padding=1), nn.
            BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2))
        self.fc = nn.Sequential(nn.Linear(in_features=128 * 3 * 3,
            out_features=256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(
            in_features=256, out_features=10))

    def forward(self, x):
        batch_size = x.size(0)
        output = self.convs(x).view(batch_size, -1)
        output = self.fc(output)
        return output


class NonLocalBlock3D(_NonLocalBlockND):

    def __init__(self, in_channels, inter_channels=None, mode=
        'embedded_gaussian', sub_sample=False, bn_layer=True):
        super(NonLocalBlock3D, self).__init__(in_channels, inter_channels=
            inter_channels, dimension=3, mode=mode, sub_sample=sub_sample,
            bn_layer=bn_layer)


def conv3x3x3(in_planes, out_planes, stride=1):
    """3x3x3 convolution with padding."""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, bias=False)


class NonLocalBasicBlock(nn.Module):
    expansion = 1
    Conv3d = staticmethod(conv3x3x3)

    def __init__(self, inplanes, planes, stride=1, downsample=None,
        nonlocal_layer=False):
        super(NonLocalBasicBlock, self).__init__()
        self.conv1 = self.Conv3d(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = self.Conv3d(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.stride = stride
        self.downsample = downsample
        self.nonlocal_layer = nonlocal_layer
        if nonlocal_layer:
            self.nonlocalblock = NonLocalBlock3D(planes)

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
        if self.nonlocal_layer:
            out = self.nonlocalblock(out)
        return out


class NonLocalBottleneck(nn.Module):
    expansion = 4
    Conv3d = nn.Conv3d

    def __init__(self, inplanes, planes, stride=1, downsample=None,
        nonlocal_layer=False):
        super(NonLocalBottleneck, self).__init__()
        self.conv1 = self.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = self.Conv3d(planes, planes, kernel_size=3, stride=
            stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = self.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.downsample = downsample
        self.nonlocal_layer = nonlocal_layer
        if nonlocal_layer:
            self.nonlocalblock = NonLocalBlock3D(planes * 4)

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
        out += residual
        out = self.relu(out)
        if self.nonlocal_layer:
            out = self.nonlocalblock(out)
        return out


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(out.size(0), planes - out.size(1), out.size(2),
        out.size(3), out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()
    out = Variable(torch.cat([out.data, zero_pads], dim=1))
    return out


class NonLocalResNet3D(nn.Module):
    Conv3d = nn.Conv3d

    def __init__(self, block, layers, nonlocal_layers, shortcut_type='A',
        num_classes=339):
        self.inplanes = 64
        super().__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=7, stride=(1, 2, 2),
            padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0],
            nonlocal_layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1],
            nonlocal_layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2],
            nonlocal_layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3],
            nonlocal_layers[3], shortcut_type, stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.last_linear = nn.Linear(512 * block.expansion, num_classes)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, self.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, nonlocal_blocks,
        shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(downsample_basic_block, planes=planes *
                    block.expansion, stride=stride)
            else:
                downsample = nn.Sequential(self.Conv3d(self.inplanes, 
                    planes * block.expansion, kernel_size=1, stride=stride,
                    bias=False), nn.BatchNorm3d(planes * block.expansion))
        nonlocal_freq = (blocks // nonlocal_blocks if nonlocal_blocks != 0 else
            -1)
        layers = []
        for i in range(0, blocks):
            layers.append(block(self.inplanes, planes, stride=stride,
                downsample=downsample, nonlocal_layer=i % nonlocal_freq == 
                0 and nonlocal_freq > 0))
            if i == 0:
                stride = 1
                downsample = None
                self.inplanes = planes * block.expansion
        return nn.Sequential(*layers)

    def features(self, x):
        x = self.conv1(x)
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


class MaxPool(nn.Module):

    def __init__(self, kernel_size, stride=1, padding=1, zero_pad=False):
        super(MaxPool, self).__init__()
        self.zero_pad = nn.ZeroPad2d((1, 0, 1, 0)) if zero_pad else None
        self.pool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        if self.zero_pad:
            x = self.zero_pad(x)
        x = self.pool(x)
        if self.zero_pad:
            x = x[:, :, 1:, 1:]
        return x


class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, dw_kernel_size, dw_stride,
        dw_padding):
        super(SeparableConv2d, self).__init__()
        self.depthwise_conv2d = nn.Conv2d(in_channels, in_channels,
            kernel_size=dw_kernel_size, stride=dw_stride, padding=
            dw_padding, groups=in_channels, bias=False)
        self.pointwise_conv2d = nn.Conv2d(in_channels, out_channels,
            kernel_size=1, bias=False)

    def forward(self, x):
        x = self.depthwise_conv2d(x)
        x = self.pointwise_conv2d(x)
        return x


class BranchSeparables(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        stem_cell=False, zero_pad=False):
        super(BranchSeparables, self).__init__()
        padding = kernel_size // 2
        middle_channels = out_channels if stem_cell else in_channels
        self.zero_pad = nn.ZeroPad2d((1, 0, 1, 0)) if zero_pad else None
        self.relu_1 = nn.ReLU()
        self.separable_1 = SeparableConv2d(in_channels, middle_channels,
            kernel_size, dw_stride=stride, dw_padding=padding)
        self.bn_sep_1 = nn.BatchNorm2d(middle_channels, eps=0.001)
        self.relu_2 = nn.ReLU()
        self.separable_2 = SeparableConv2d(middle_channels, out_channels,
            kernel_size, dw_stride=1, dw_padding=padding)
        self.bn_sep_2 = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.relu_1(x)
        if self.zero_pad:
            x = self.zero_pad(x)
        x = self.separable_1(x)
        if self.zero_pad:
            x = x[:, :, 1:, 1:].contiguous()
        x = self.bn_sep_1(x)
        x = self.relu_2(x)
        x = self.separable_2(x)
        x = self.bn_sep_2(x)
        return x


class ReluConvBn(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(ReluConvBn, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=
            kernel_size, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.relu(x)
        x = self.conv(x)
        x = self.bn(x)
        return x


class FactorizedReduction(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(FactorizedReduction, self).__init__()
        self.relu = nn.ReLU()
        self.path_1 = nn.Sequential(OrderedDict([('avgpool', nn.AvgPool2d(1,
            stride=2, count_include_pad=False)), ('conv', nn.Conv2d(
            in_channels, out_channels // 2, kernel_size=1, bias=False))]))
        self.path_2 = nn.Sequential(OrderedDict([('pad', nn.ZeroPad2d((0, 1,
            0, 1))), ('avgpool', nn.AvgPool2d(1, stride=2,
            count_include_pad=False)), ('conv', nn.Conv2d(in_channels, 
            out_channels // 2, kernel_size=1, bias=False))]))
        self.final_path_bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.relu(x)
        x_path1 = self.path_1(x)
        x_path2 = self.path_2.pad(x)
        x_path2 = x_path2[:, :, 1:, 1:]
        x_path2 = self.path_2.avgpool(x_path2)
        x_path2 = self.path_2.conv(x_path2)
        out = self.final_path_bn(torch.cat([x_path1, x_path2], 1))
        return out


class CellBase(nn.Module):

    def cell_forward(self, x_left, x_right):
        x_comb_iter_0_left = self.comb_iter_0_left(x_left)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
        x_comb_iter_1_left = self.comb_iter_1_left(x_right)
        x_comb_iter_1_right = self.comb_iter_1_right(x_right)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2_right = self.comb_iter_2_right(x_right)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right
        x_comb_iter_3_left = self.comb_iter_3_left(x_comb_iter_2)
        x_comb_iter_3_right = self.comb_iter_3_right(x_right)
        x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right
        x_comb_iter_4_left = self.comb_iter_4_left(x_left)
        if self.comb_iter_4_right:
            x_comb_iter_4_right = self.comb_iter_4_right(x_right)
        else:
            x_comb_iter_4_right = x_right
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right
        x_out = torch.cat([x_comb_iter_0, x_comb_iter_1, x_comb_iter_2,
            x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class Cell(CellBase):

    def __init__(self, in_channels_left, out_channels_left,
        in_channels_right, out_channels_right, is_reduction=False, zero_pad
        =False, match_prev_layer_dimensions=False):
        super(Cell, self).__init__()
        stride = 2 if is_reduction else 1
        self.match_prev_layer_dimensions = match_prev_layer_dimensions
        if match_prev_layer_dimensions:
            self.conv_prev_1x1 = FactorizedReduction(in_channels_left,
                out_channels_left)
        else:
            self.conv_prev_1x1 = ReluConvBn(in_channels_left,
                out_channels_left, kernel_size=1)
        self.conv_1x1 = ReluConvBn(in_channels_right, out_channels_right,
            kernel_size=1)
        self.comb_iter_0_left = BranchSeparables(out_channels_left,
            out_channels_left, kernel_size=5, stride=stride, zero_pad=zero_pad)
        self.comb_iter_0_right = MaxPool(3, stride=stride, zero_pad=zero_pad)
        self.comb_iter_1_left = BranchSeparables(out_channels_right,
            out_channels_right, kernel_size=7, stride=stride, zero_pad=zero_pad
            )
        self.comb_iter_1_right = MaxPool(3, stride=stride, zero_pad=zero_pad)
        self.comb_iter_2_left = BranchSeparables(out_channels_right,
            out_channels_right, kernel_size=5, stride=stride, zero_pad=zero_pad
            )
        self.comb_iter_2_right = BranchSeparables(out_channels_right,
            out_channels_right, kernel_size=3, stride=stride, zero_pad=zero_pad
            )
        self.comb_iter_3_left = BranchSeparables(out_channels_right,
            out_channels_right, kernel_size=3)
        self.comb_iter_3_right = MaxPool(3, stride=stride, zero_pad=zero_pad)
        self.comb_iter_4_left = BranchSeparables(out_channels_left,
            out_channels_left, kernel_size=3, stride=stride, zero_pad=zero_pad)
        if is_reduction:
            self.comb_iter_4_right = ReluConvBn(out_channels_right,
                out_channels_right, kernel_size=1, stride=stride)
        else:
            self.comb_iter_4_right = None

    def forward(self, x_left, x_right):
        x_left = self.conv_prev_1x1(x_left)
        x_right = self.conv_1x1(x_right)
        x_out = self.cell_forward(x_left, x_right)
        return x_out


class PNASNet5Large(nn.Module):

    def __init__(self, num_classes=1001):
        super().__init__()
        self.num_classes = num_classes
        self.conv_0 = nn.Sequential(OrderedDict([('conv', nn.Conv2d(3, 96,
            kernel_size=3, stride=2, bias=False)), ('bn', nn.BatchNorm2d(96,
            eps=0.001))]))
        self.cell_stem_0 = CellStem0(in_channels_left=96, out_channels_left
            =54, in_channels_right=96, out_channels_right=54)
        self.cell_stem_1 = Cell(in_channels_left=96, out_channels_left=108,
            in_channels_right=270, out_channels_right=108,
            match_prev_layer_dimensions=True, is_reduction=True)
        self.cell_0 = Cell(in_channels_left=270, out_channels_left=216,
            in_channels_right=540, out_channels_right=216,
            match_prev_layer_dimensions=True)
        self.cell_1 = Cell(in_channels_left=540, out_channels_left=216,
            in_channels_right=1080, out_channels_right=216)
        self.cell_2 = Cell(in_channels_left=1080, out_channels_left=216,
            in_channels_right=1080, out_channels_right=216)
        self.cell_3 = Cell(in_channels_left=1080, out_channels_left=216,
            in_channels_right=1080, out_channels_right=216)
        self.cell_4 = Cell(in_channels_left=1080, out_channels_left=432,
            in_channels_right=1080, out_channels_right=432, is_reduction=
            True, zero_pad=True)
        self.cell_5 = Cell(in_channels_left=1080, out_channels_left=432,
            in_channels_right=2160, out_channels_right=432,
            match_prev_layer_dimensions=True)
        self.cell_6 = Cell(in_channels_left=2160, out_channels_left=432,
            in_channels_right=2160, out_channels_right=432)
        self.cell_7 = Cell(in_channels_left=2160, out_channels_left=432,
            in_channels_right=2160, out_channels_right=432)
        self.cell_8 = Cell(in_channels_left=2160, out_channels_left=864,
            in_channels_right=2160, out_channels_right=864, is_reduction=True)
        self.cell_9 = Cell(in_channels_left=2160, out_channels_left=864,
            in_channels_right=4320, out_channels_right=864,
            match_prev_layer_dimensions=True)
        self.cell_10 = Cell(in_channels_left=4320, out_channels_left=864,
            in_channels_right=4320, out_channels_right=864)
        self.cell_11 = Cell(in_channels_left=4320, out_channels_left=864,
            in_channels_right=4320, out_channels_right=864)
        self.relu = nn.ReLU()
        self.avg_pool = nn.AvgPool2d(11, stride=1, padding=0)
        self.dropout = nn.Dropout(0.5)
        self.last_linear = nn.Linear(4320, num_classes)

    def features(self, x):
        x_conv_0 = self.conv_0(x)
        x_stem_0 = self.cell_stem_0(x_conv_0)
        x_stem_1 = self.cell_stem_1(x_conv_0, x_stem_0)
        x_cell_0 = self.cell_0(x_stem_0, x_stem_1)
        x_cell_1 = self.cell_1(x_stem_1, x_cell_0)
        x_cell_2 = self.cell_2(x_cell_0, x_cell_1)
        x_cell_3 = self.cell_3(x_cell_1, x_cell_2)
        x_cell_4 = self.cell_4(x_cell_2, x_cell_3)
        x_cell_5 = self.cell_5(x_cell_3, x_cell_4)
        x_cell_6 = self.cell_6(x_cell_4, x_cell_5)
        x_cell_7 = self.cell_7(x_cell_5, x_cell_6)
        x_cell_8 = self.cell_8(x_cell_6, x_cell_7)
        x_cell_9 = self.cell_9(x_cell_7, x_cell_8)
        x_cell_10 = self.cell_10(x_cell_8, x_cell_9)
        x_cell_11 = self.cell_11(x_cell_9, x_cell_10)
        return x_cell_11

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


class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1,
        padding=0, output_relu=True):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=
            kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU() if output_relu else None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu:
            x = self.relu(x)
        return x


class PolyConv2d(nn.Module):
    """A block that is used inside poly-N (poly-2, poly-3, and so on) modules.
    The Convolution layer is shared between all Inception blocks inside
    a poly-N module. BatchNorm layers are not shared between Inception blocks
    and therefore the number of BatchNorm layers is equal to the number of
    Inception blocks inside a poly-N module.
    """

    def __init__(self, in_planes, out_planes, kernel_size, num_blocks,
        stride=1, padding=0):
        super(PolyConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=
            kernel_size, stride=stride, padding=padding, bias=False)
        self.bn_blocks = nn.ModuleList([nn.BatchNorm2d(out_planes) for _ in
            range(num_blocks)])
        self.relu = nn.ReLU()

    def forward(self, x, block_index):
        x = self.conv(x)
        bn = self.bn_blocks[block_index]
        x = bn(x)
        x = self.relu(x)
        return x


class Stem(nn.Module):

    def __init__(self):
        super(Stem, self).__init__()
        self.conv1 = nn.Sequential(BasicConv2d(3, 32, kernel_size=3, stride
            =2), BasicConv2d(32, 32, kernel_size=3), BasicConv2d(32, 64,
            kernel_size=3, padding=1))
        self.conv1_pool_branch = nn.MaxPool2d(3, stride=2)
        self.conv1_branch = BasicConv2d(64, 96, kernel_size=3, stride=2)
        self.conv2_short = nn.Sequential(BasicConv2d(160, 64, kernel_size=1
            ), BasicConv2d(64, 96, kernel_size=3))
        self.conv2_long = nn.Sequential(BasicConv2d(160, 64, kernel_size=1),
            BasicConv2d(64, 64, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(64, 64, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(64, 96, kernel_size=3))
        self.conv2_pool_branch = nn.MaxPool2d(3, stride=2)
        self.conv2_branch = BasicConv2d(192, 192, kernel_size=3, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x0 = self.conv1_pool_branch(x)
        x1 = self.conv1_branch(x)
        x = torch.cat((x0, x1), 1)
        x0 = self.conv2_short(x)
        x1 = self.conv2_long(x)
        x = torch.cat((x0, x1), 1)
        x0 = self.conv2_pool_branch(x)
        x1 = self.conv2_branch(x)
        out = torch.cat((x0, x1), 1)
        return out


class BlockA(nn.Module):
    """Inception-ResNet-A block."""

    def __init__(self):
        super(BlockA, self).__init__()
        self.path0 = nn.Sequential(BasicConv2d(384, 32, kernel_size=1),
            BasicConv2d(32, 48, kernel_size=3, padding=1), BasicConv2d(48, 
            64, kernel_size=3, padding=1))
        self.path1 = nn.Sequential(BasicConv2d(384, 32, kernel_size=1),
            BasicConv2d(32, 32, kernel_size=3, padding=1))
        self.path2 = BasicConv2d(384, 32, kernel_size=1)
        self.conv2d = BasicConv2d(128, 384, kernel_size=1, output_relu=False)

    def forward(self, x):
        x0 = self.path0(x)
        x1 = self.path1(x)
        x2 = self.path2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        return out


class BlockB(nn.Module):
    """Inception-ResNet-B block."""

    def __init__(self):
        super(BlockB, self).__init__()
        self.path0 = nn.Sequential(BasicConv2d(1152, 128, kernel_size=1),
            BasicConv2d(128, 160, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(160, 192, kernel_size=(7, 1), padding=(3, 0)))
        self.path1 = BasicConv2d(1152, 192, kernel_size=1)
        self.conv2d = BasicConv2d(384, 1152, kernel_size=1, output_relu=False)

    def forward(self, x):
        x0 = self.path0(x)
        x1 = self.path1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        return out


class BlockC(nn.Module):
    """Inception-ResNet-C block."""

    def __init__(self):
        super(BlockC, self).__init__()
        self.path0 = nn.Sequential(BasicConv2d(2048, 192, kernel_size=1),
            BasicConv2d(192, 224, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(224, 256, kernel_size=(3, 1), padding=(1, 0)))
        self.path1 = BasicConv2d(2048, 192, kernel_size=1)
        self.conv2d = BasicConv2d(448, 2048, kernel_size=1, output_relu=False)

    def forward(self, x):
        x0 = self.path0(x)
        x1 = self.path1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        return out


class ReductionA(nn.Module):
    """A dimensionality reduction block that is placed after stage-a
    Inception-ResNet blocks.
    """

    def __init__(self):
        super(ReductionA, self).__init__()
        self.path0 = nn.Sequential(BasicConv2d(384, 256, kernel_size=1),
            BasicConv2d(256, 256, kernel_size=3, padding=1), BasicConv2d(
            256, 384, kernel_size=3, stride=2))
        self.path1 = BasicConv2d(384, 384, kernel_size=3, stride=2)
        self.path2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.path0(x)
        x1 = self.path1(x)
        x2 = self.path2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class ReductionB(nn.Module):
    """A dimensionality reduction block that is placed after stage-b
    Inception-ResNet blocks.
    """

    def __init__(self):
        super(ReductionB, self).__init__()
        self.path0 = nn.Sequential(BasicConv2d(1152, 256, kernel_size=1),
            BasicConv2d(256, 256, kernel_size=3, padding=1), BasicConv2d(
            256, 256, kernel_size=3, stride=2))
        self.path1 = nn.Sequential(BasicConv2d(1152, 256, kernel_size=1),
            BasicConv2d(256, 256, kernel_size=3, stride=2))
        self.path2 = nn.Sequential(BasicConv2d(1152, 256, kernel_size=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2))
        self.path3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.path0(x)
        x1 = self.path1(x)
        x2 = self.path2(x)
        x3 = self.path3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class InceptionResNetBPoly(nn.Module):
    """Base class for constructing poly-N Inception-ResNet-B modules.
    When `num_blocks` is equal to 1, a module will have only a first-order path
    and will be equal to a standard Inception-ResNet-B block.
    When `num_blocks` is equal to 2, a module will have first-order and
    second-order paths and will be called Inception-ResNet-B poly-2 module.
    Increasing value of the `num_blocks` parameter will produce a higher order
    Inception-ResNet-B poly-N modules.
    """

    def __init__(self, scale, num_blocks):
        super(InceptionResNetBPoly, self).__init__()
        assert num_blocks >= 1, 'num_blocks should be greater or equal to 1'
        self.scale = scale
        self.num_blocks = num_blocks
        self.path0_1x1 = PolyConv2d(1152, 128, kernel_size=1, num_blocks=
            self.num_blocks)
        self.path0_1x7 = PolyConv2d(128, 160, kernel_size=(1, 7),
            num_blocks=self.num_blocks, padding=(0, 3))
        self.path0_7x1 = PolyConv2d(160, 192, kernel_size=(7, 1),
            num_blocks=self.num_blocks, padding=(3, 0))
        self.path1 = PolyConv2d(1152, 192, kernel_size=1, num_blocks=self.
            num_blocks)
        self.conv2d_blocks = nn.ModuleList([BasicConv2d(384, 1152,
            kernel_size=1, output_relu=False) for _ in range(self.num_blocks)])
        self.relu = nn.ReLU()

    def forward_block(self, x, block_index):
        x0 = self.path0_1x1(x, block_index)
        x0 = self.path0_1x7(x0, block_index)
        x0 = self.path0_7x1(x0, block_index)
        x1 = self.path1(x, block_index)
        out = torch.cat((x0, x1), 1)
        conv2d_block = self.conv2d_blocks[block_index]
        out = conv2d_block(out)
        return out

    def forward(self, x):
        out = x
        for block_index in range(self.num_blocks):
            x = self.forward_block(x, block_index)
            out = out + x * self.scale
            x = self.relu(x)
        out = self.relu(out)
        return out


class InceptionResNetCPoly(nn.Module):
    """Base class for constructing poly-N Inception-ResNet-C modules.
    When `num_blocks` is equal to 1, a module will have only a first-order path
    and will be equal to a standard Inception-ResNet-C block.
    When `num_blocks` is equal to 2, a module will have first-order and
    second-order paths and will be called Inception-ResNet-C poly-2 module.
    Increasing value of the `num_blocks` parameter will produce a higher order
    Inception-ResNet-C poly-N modules.
    """

    def __init__(self, scale, num_blocks):
        super(InceptionResNetCPoly, self).__init__()
        assert num_blocks >= 1, 'num_blocks should be greater or equal to 1'
        self.scale = scale
        self.num_blocks = num_blocks
        self.path0_1x1 = PolyConv2d(2048, 192, kernel_size=1, num_blocks=
            self.num_blocks)
        self.path0_1x3 = PolyConv2d(192, 224, kernel_size=(1, 3),
            num_blocks=self.num_blocks, padding=(0, 1))
        self.path0_3x1 = PolyConv2d(224, 256, kernel_size=(3, 1),
            num_blocks=self.num_blocks, padding=(1, 0))
        self.path1 = PolyConv2d(2048, 192, kernel_size=1, num_blocks=self.
            num_blocks)
        self.conv2d_blocks = nn.ModuleList([BasicConv2d(448, 2048,
            kernel_size=1, output_relu=False) for _ in range(self.num_blocks)])
        self.relu = nn.ReLU()

    def forward_block(self, x, block_index):
        x0 = self.path0_1x1(x, block_index)
        x0 = self.path0_1x3(x0, block_index)
        x0 = self.path0_3x1(x0, block_index)
        x1 = self.path1(x, block_index)
        out = torch.cat((x0, x1), 1)
        conv2d_block = self.conv2d_blocks[block_index]
        out = conv2d_block(out)
        return out

    def forward(self, x):
        out = x
        for block_index in range(self.num_blocks):
            x = self.forward_block(x, block_index)
            out = out + x * self.scale
            x = self.relu(x)
        out = self.relu(out)
        return out


class MultiWay(nn.Module):
    """Base class for constructing N-way modules (2-way, 3-way, and so on)."""

    def __init__(self, scale, block_cls, num_blocks):
        super(MultiWay, self).__init__()
        assert num_blocks >= 1, 'num_blocks should be greater or equal to 1'
        self.scale = scale
        self.blocks = nn.ModuleList([block_cls() for _ in range(num_blocks)])
        self.relu = nn.ReLU()

    def forward(self, x):
        out = x
        for block in self.blocks:
            out = out + block(x) * self.scale
        out = self.relu(out)
        return out


class InceptionResNetA2Way(MultiWay):

    def __init__(self, scale):
        super(InceptionResNetA2Way, self).__init__(scale, block_cls=BlockA,
            num_blocks=2)


class InceptionResNetB2Way(MultiWay):

    def __init__(self, scale):
        super(InceptionResNetB2Way, self).__init__(scale, block_cls=BlockB,
            num_blocks=2)


class InceptionResNetBPoly3(InceptionResNetBPoly):

    def __init__(self, scale):
        super(InceptionResNetBPoly3, self).__init__(scale, num_blocks=3)


class InceptionResNetC2Way(MultiWay):

    def __init__(self, scale):
        super(InceptionResNetC2Way, self).__init__(scale, block_cls=BlockC,
            num_blocks=2)


class InceptionResNetCPoly3(InceptionResNetCPoly):

    def __init__(self, scale):
        super(InceptionResNetCPoly3, self).__init__(scale, num_blocks=3)


class PolyNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(PolyNet, self).__init__()
        self.stem = Stem()
        self.stage_a = nn.Sequential(InceptionResNetA2Way(scale=1),
            InceptionResNetA2Way(scale=0.992308), InceptionResNetA2Way(
            scale=0.984615), InceptionResNetA2Way(scale=0.976923),
            InceptionResNetA2Way(scale=0.969231), InceptionResNetA2Way(
            scale=0.961538), InceptionResNetA2Way(scale=0.953846),
            InceptionResNetA2Way(scale=0.946154), InceptionResNetA2Way(
            scale=0.938462), InceptionResNetA2Way(scale=0.930769))
        self.reduction_a = ReductionA()
        self.stage_b = nn.Sequential(InceptionResNetBPoly3(scale=0.923077),
            InceptionResNetB2Way(scale=0.915385), InceptionResNetBPoly3(
            scale=0.907692), InceptionResNetB2Way(scale=0.9),
            InceptionResNetBPoly3(scale=0.892308), InceptionResNetB2Way(
            scale=0.884615), InceptionResNetBPoly3(scale=0.876923),
            InceptionResNetB2Way(scale=0.869231), InceptionResNetBPoly3(
            scale=0.861538), InceptionResNetB2Way(scale=0.853846),
            InceptionResNetBPoly3(scale=0.846154), InceptionResNetB2Way(
            scale=0.838462), InceptionResNetBPoly3(scale=0.830769),
            InceptionResNetB2Way(scale=0.823077), InceptionResNetBPoly3(
            scale=0.815385), InceptionResNetB2Way(scale=0.807692),
            InceptionResNetBPoly3(scale=0.8), InceptionResNetB2Way(scale=
            0.792308), InceptionResNetBPoly3(scale=0.784615),
            InceptionResNetB2Way(scale=0.776923))
        self.reduction_b = ReductionB()
        self.stage_c = nn.Sequential(InceptionResNetCPoly3(scale=0.769231),
            InceptionResNetC2Way(scale=0.761538), InceptionResNetCPoly3(
            scale=0.753846), InceptionResNetC2Way(scale=0.746154),
            InceptionResNetCPoly3(scale=0.738462), InceptionResNetC2Way(
            scale=0.730769), InceptionResNetCPoly3(scale=0.723077),
            InceptionResNetC2Way(scale=0.715385), InceptionResNetCPoly3(
            scale=0.707692), InceptionResNetC2Way(scale=0.7))
        self.avg_pool = nn.AvgPool2d(9, stride=1)
        self.dropout = nn.Dropout(0.2)
        self.last_linear = nn.Linear(2048, num_classes)

    def features(self, x):
        x = self.stem(x)
        x = self.stage_a(x)
        x = self.reduction_a(x)
        x = self.stage_b(x)
        x = self.reduction_b(x)
        x = self.stage_c(x)
        return x

    def logits(self, x):
        x = self.avg_pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x


class PreActivationBasicBlock(nn.Module):
    expansion = 1
    Conv3d = staticmethod(conv3x3x3)

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(PreActivationBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm3d(inplanes)
        self.conv1 = self.Conv3d(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv2 = self.Conv3d(planes, planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return out


class PreActivationBottleneck(nn.Module):
    expansion = 4
    Conv3d = nn.Conv3d

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(PreActivationBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm3d(inplanes)
        self.conv1 = self.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv2 = self.Conv3d(planes, planes, kernel_size=3, stride=
            stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes)
        self.conv3 = self.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return out


class SpatioTemporalConv(nn.Module):
    """Applies a factored 3D convolution over an input signal.

    The input signal is composed of several input planes with distinct
    spatial and time axes,by performing a 2D convolution over the spatial
    axes to an intermediate subspace, followed by a 1D convolution over the
    time axis to produce the final output.

    Args:
        in_channels (int): Number of channels in the input tensor
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to the sides of the input during their respective convolutions. Default: 0
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, bias=True):
        super().__init__()
        stride = _triple(stride)
        padding = _triple(padding)
        kernel_size = _triple(kernel_size)
        spatial_stride = [1, stride[1], stride[2]]
        spatial_padding = [0, padding[1], padding[2]]
        spatial_kernel_size = [1, kernel_size[1], kernel_size[2]]
        temporal_stride = [stride[0], 1, 1]
        temporal_padding = [padding[0], 0, 0]
        temporal_kernel_size = [kernel_size[0], 1, 1]
        intermed_channels = int(math.floor(kernel_size[0] * kernel_size[1] *
            kernel_size[2] * in_channels * out_channels / (kernel_size[1] *
            kernel_size[2] * in_channels + kernel_size[0] * out_channels)))
        self.spatial_conv = nn.Conv3d(in_channels, intermed_channels,
            spatial_kernel_size, stride=spatial_stride, padding=
            spatial_padding, bias=bias)
        self.bn = nn.BatchNorm3d(intermed_channels)
        self.relu = nn.ReLU()
        self.temporal_conv = nn.Conv3d(intermed_channels, out_channels,
            temporal_kernel_size, stride=temporal_stride, padding=
            temporal_padding, bias=bias)

    def forward(self, x):
        x = self.relu(self.bn(self.spatial_conv(x)))
        x = self.temporal_conv(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1
    Conv3d = staticmethod(conv3x3x3)

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = self.Conv3d(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = self.Conv3d(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
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
    expansion = 4
    Conv3d = nn.Conv3d

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = self.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = self.Conv3d(planes, planes, kernel_size=3, stride=
            stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = self.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

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
        out += residual
        out = self.relu(out)
        return out


class ResNet3D(nn.Module):
    Conv3d = nn.Conv3d

    def __init__(self, block, layers, shortcut_type='B', num_classes=339):
        self.inplanes = 64
        super(ResNet3D, self).__init__()
        self.conv1 = self.Conv3d(3, 64, kernel_size=7, stride=(1, 2, 2),
            padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type,
            stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type,
            stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type,
            stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.init_weights()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(downsample_basic_block, planes=planes *
                    block.expansion, stride=stride)
            else:
                downsample = nn.Sequential(self.Conv3d(self.inplanes, 
                    planes * block.expansion, kernel_size=1, stride=stride,
                    bias=False), nn.BatchNorm3d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, self.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
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


class ResNeXtBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, cardinality, stride=1, downsample=None
        ):
        super(ResNeXtBottleneck, self).__init__()
        mid_planes = cardinality * int(planes / 32)
        self.conv1 = nn.Conv3d(inplanes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_planes)
        self.conv2 = nn.Conv3d(mid_planes, mid_planes, kernel_size=3,
            stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm3d(mid_planes)
        self.conv3 = nn.Conv3d(mid_planes, planes * self.expansion,
            kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

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
        out += residual
        out = self.relu(out)
        return out


class ResNeXt3D(nn.Module):

    def __init__(self, block, layers, shortcut_type='B', cardinality=32,
        num_classes=400):
        self.inplanes = 64
        super(ResNeXt3D, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=7, stride=(1, 2, 2),
            padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 128, layers[0], shortcut_type,
            cardinality)
        self.layer2 = self._make_layer(block, 256, layers[1], shortcut_type,
            cardinality, stride=2)
        self.layer3 = self._make_layer(block, 512, layers[2], shortcut_type,
            cardinality, stride=2)
        self.layer4 = self._make_layer(block, 1024, layers[3],
            shortcut_type, cardinality, stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(cardinality * 32 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, cardinality,
        stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(downsample_basic_block, planes=planes *
                    block.expansion, stride=stride)
            else:
                downsample = nn.Sequential(nn.Conv3d(self.inplanes, planes *
                    block.expansion, kernel_size=1, stride=stride, bias=
                    False), nn.BatchNorm3d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, cardinality, stride,
            downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, cardinality))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
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


class LambdaBase(nn.Sequential):

    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input


class LambdaBase(nn.Sequential):

    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input


class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
            padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
            padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


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


class SENet(nn.Module):

    def __init__(self, block, layers, groups, reduction, dropout_p=0.2,
        inplanes=128, input_3x3=True, downsample_kernel_size=3,
        downsample_padding=1, num_classes=1000):
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
            layer0_modules = [('conv1', nn.Conv2d(3, 64, 3, stride=2,
                padding=1, bias=False)), ('bn1', nn.BatchNorm2d(64)), (
                'relu1', nn.ReLU(inplace=True)), ('conv2', nn.Conv2d(64, 64,
                3, stride=1, padding=1, bias=False)), ('bn2', nn.
                BatchNorm2d(64)), ('relu2', nn.ReLU(inplace=True)), (
                'conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1,
                bias=False)), ('bn3', nn.BatchNorm2d(inplanes)), ('relu3',
                nn.ReLU(inplace=True))]
        else:
            layer0_modules = [('conv1', nn.Conv2d(3, inplanes, kernel_size=
                7, stride=2, padding=3, bias=False)), ('bn1', nn.
                BatchNorm2d(inplanes)), ('relu1', nn.ReLU(inplace=True))]
        layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2, ceil_mode=
            True)))
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        self.layer1 = self._make_layer(block, planes=64, blocks=layers[0],
            groups=groups, reduction=reduction, downsample_kernel_size=1,
            downsample_padding=0)
        self.layer2 = self._make_layer(block, planes=128, blocks=layers[1],
            stride=2, groups=groups, reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding)
        self.layer3 = self._make_layer(block, planes=256, blocks=layers[2],
            stride=2, groups=groups, reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding)
        self.layer4 = self._make_layer(block, planes=512, blocks=layers[3],
            stride=2, groups=groups, reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding)
        self.avg_pool = nn.AvgPool2d(7, stride=1)
        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        self.last_linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, groups, reduction, stride=
        1, downsample_kernel_size=1, downsample_padding=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=downsample_kernel_size, stride
                =stride, padding=downsample_padding, bias=False), nn.
                BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, groups, reduction,
            stride, downsample))
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


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, head_conv=1
        ):
        super().__init__()
        if head_conv == 1:
            self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(1, 3, 3),
                padding=(0, 1, 1), stride=(1, stride, stride), bias=False)
        elif head_conv == 3:
            self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(3, 1, 1),
                padding=(1, 0, 0), bias=False)
        else:
            raise ValueError('Unsupported head_conv')
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(1, 3, 3),
            padding=(0, 1, 1), stride=(1, stride, stride))
        self.bn2 = nn.BatchNorm3d(planes)
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
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, head_conv=1
        ):
        super(Bottleneck, self).__init__()
        if head_conv == 1:
            self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        elif head_conv == 3:
            self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(3, 1, 1),
                bias=False, padding=(1, 0, 0))
        else:
            raise ValueError('Unsupported head_conv!')
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(1, 3, 3),
            stride=(1, stride, stride), padding=(0, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

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
        out += residual
        out = self.relu(out)
        return out


class Slow(nn.Module):

    def __init__(self, block=Bottleneck, layers=[2, 2, 2, 2]):
        super().__init__()
        self.inplanes = 64 + 64 // 8 * 2
        self._make_layers(block, layers)

    def _make_layers(self, block, layers):
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(1, 7, 7), stride=(1, 2, 
            2), padding=(0, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2),
            padding=(0, 1, 1))
        self.res2 = self._make_layer_slow(block, 64, layers[0], head_conv=1)
        res3_stride = 2 if issubclass(block, Bottleneck) else 1
        self.res3 = self._make_layer_slow(block, 128, layers[1], stride=
            res3_stride, head_conv=1)
        self.res4 = self._make_layer_slow(block, 256, layers[2], stride=2,
            head_conv=3)
        self.res5 = self._make_layer_slow(block, 512, layers[3], stride=2,
            head_conv=3)

    def forward(self, input, lateral):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = torch.cat([x, lateral[0]], dim=1)
        x = self.res2(x)
        x = torch.cat([x, lateral[1]], dim=1)
        x = self.res3(x)
        x = torch.cat([x, lateral[2]], dim=1)
        x = self.res4(x)
        x = torch.cat([x, lateral[3]], dim=1)
        x = self.res5(x)
        x = nn.AdaptiveAvgPool3d(1)(x)
        x = x.view(-1, x.size(1))
        return x

    def _make_layer_slow(self, block, planes, blocks, stride=1, head_conv=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv3d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=(1, stride, stride),
                bias=False), nn.BatchNorm3d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
            head_conv=head_conv))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, head_conv=head_conv))
        self.inplanes = (planes * block.expansion + planes * block.
            expansion // 8 * 2)
        return nn.Sequential(*layers)


class Fast(nn.Module):

    def __init__(self, block=Bottleneck, layers=[2, 2, 2, 2]):
        super().__init__()
        self.inplanes = 8
        self.conv1 = nn.Conv3d(3, 8, kernel_size=(5, 7, 7), stride=(1, 2, 2
            ), padding=(2, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(8)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2),
            padding=(0, 1, 1))
        self.res2 = self._make_layer_fast(block, 8, layers[0], head_conv=3)
        res3_stride = 2 if issubclass(block, Bottleneck) else 1
        self.res3 = self._make_layer_fast(block, 16, layers[1], stride=
            res3_stride, head_conv=3)
        self.res4 = self._make_layer_fast(block, 32, layers[2], stride=2,
            head_conv=3)
        self.res5 = self._make_layer_fast(block, 64, layers[3], stride=2,
            head_conv=3)
        expansion = 4 if issubclass(block, Bottleneck) else 1
        self._make_lateral_layers(expansion)

    def _make_lateral_layers(self, expansion):
        self.lateral_p1 = nn.Conv3d(8, 8 * 2, kernel_size=(5, 1, 1), stride
            =(8, 1, 1), bias=False, padding=(2, 0, 0))
        r2 = 8 * expansion
        self.lateral_res2 = nn.Conv3d(r2, r2 * 2, kernel_size=(5, 1, 1),
            stride=(8, 1, 1), bias=False, padding=(2, 0, 0))
        r3 = 16 * expansion
        self.lateral_res3 = nn.Conv3d(r3, r3 * 2, kernel_size=(5, 1, 1),
            stride=(8, 1, 1), bias=False, padding=(2, 0, 0))
        r4 = 32 * expansion
        self.lateral_res4 = nn.Conv3d(r4, r4 * 2, kernel_size=(5, 1, 1),
            stride=(8, 1, 1), bias=False, padding=(2, 0, 0))

    def forward(self, input):
        lateral = []
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        pool1 = self.maxpool(x)
        lateral.append(self.lateral_p1(pool1))
        res2 = self.res2(pool1)
        lateral.append(self.lateral_res2(res2))
        res3 = self.res3(res2)
        lateral.append(self.lateral_res3(res3))
        res4 = self.res4(res3)
        lateral.append(self.lateral_res4(res4))
        res5 = self.res5(res4)
        x = nn.AdaptiveAvgPool3d(1)(res5)
        x = x.view(-1, x.size(1))
        return x, lateral

    def _make_layer_fast(self, block, planes, blocks, stride=1, head_conv=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv3d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=(1, stride, stride),
                bias=False), nn.BatchNorm3d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
            head_conv=head_conv))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, head_conv=head_conv))
        return nn.Sequential(*layers)


class SlowFast(nn.Module):
    """SlowFast Network.

    Constructed from Slow and Fast nets.

    """

    def __init__(self, block=Bottleneck, layers=[2, 2, 2, 2], num_classes=
        400, dropout=0.5, slow_stride=16, fast_stride=2):
        super().__init__()
        self.slow_stride = slow_stride
        self.fast_stride = fast_stride
        self.expansion = 4 if issubclass(block, Bottleneck) else 1
        self.slow = Slow(block=block, layers=layers)
        self.fast = Fast(block=block, layers=layers)
        self.dropout = nn.Dropout(dropout)
        self.last_linear = nn.Linear(self.fast.inplanes + 512 * self.
            expansion, num_classes, bias=False)

    def forward(self, input):
        fast, lateral = self.fast(input[:, :, ::self.fast_stride, :, :])
        slow = self.slow(input[:, :, ::self.slow_stride, :, :], lateral)
        x = torch.cat([slow, fast], dim=1)
        x = self.dropout(x)
        x = self.last_linear(x)
        return x


class SlowFastV0(nn.Module):
    """Original SlowFast implementation where slow and fast pathways are built
    together.

    Advantages: Clean API.
    Disadvantages: Difficult to use individual pathways separately.

    """

    def __init__(self, block=Bottleneck, layers=[3, 4, 6, 3], num_classes=
        10, dropout=0.5):
        super().__init__()
        self.fast_inplanes = 8
        self.fast_conv1 = nn.Conv3d(3, 8, kernel_size=(5, 7, 7), stride=(1,
            2, 2), padding=(2, 3, 3), bias=False)
        self.fast_bn1 = nn.BatchNorm3d(8)
        self.fast_relu = nn.ReLU(inplace=True)
        self.fast_maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 
            2, 2), padding=(0, 1, 1))
        self.fast_res2 = self._make_layer_fast(block, 8, layers[0], head_conv=3
            )
        self.fast_res3 = self._make_layer_fast(block, 16, layers[1], stride
            =2, head_conv=3)
        self.fast_res4 = self._make_layer_fast(block, 32, layers[2], stride
            =2, head_conv=3)
        self.fast_res5 = self._make_layer_fast(block, 64, layers[3], stride
            =2, head_conv=3)
        self.lateral_p1 = nn.Conv3d(8, 8 * 2, kernel_size=(5, 1, 1), stride
            =(8, 1, 1), bias=False, padding=(2, 0, 0))
        self.lateral_res2 = nn.Conv3d(32, 32 * 2, kernel_size=(5, 1, 1),
            stride=(8, 1, 1), bias=False, padding=(2, 0, 0))
        self.lateral_res3 = nn.Conv3d(64, 64 * 2, kernel_size=(5, 1, 1),
            stride=(8, 1, 1), bias=False, padding=(2, 0, 0))
        self.lateral_res4 = nn.Conv3d(128, 128 * 2, kernel_size=(5, 1, 1),
            stride=(8, 1, 1), bias=False, padding=(2, 0, 0))
        self.slow_inplanes = 64 + 64 // 8 * 2
        self.slow_conv1 = nn.Conv3d(3, 64, kernel_size=(1, 7, 7), stride=(1,
            2, 2), padding=(0, 3, 3), bias=False)
        self.slow_bn1 = nn.BatchNorm3d(64)
        self.slow_relu = nn.ReLU(inplace=True)
        self.slow_maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 
            2, 2), padding=(0, 1, 1))
        self.slow_res2 = self._make_layer_slow(block, 64, layers[0],
            head_conv=1)
        self.slow_res3 = self._make_layer_slow(block, 128, layers[1],
            stride=2, head_conv=1)
        self.slow_res4 = self._make_layer_slow(block, 256, layers[2],
            stride=2, head_conv=3)
        self.slow_res5 = self._make_layer_slow(block, 512, layers[3],
            stride=2, head_conv=3)
        self.dropout = nn.Dropout(dropout)
        self.last_linear = nn.Linear(self.fast_inplanes + 2048, num_classes,
            bias=False)

    def forward(self, input):
        fast, lateral = self.fast_path(input[:, :, ::2, :, :])
        slow = self.slow_path(input[:, :, ::16, :, :], lateral)
        x = torch.cat([slow, fast], dim=1)
        x = self.dropout(x)
        x = self.last_linear(x)
        return x

    def slow_path(self, input, lateral):
        x = self.slow_conv1(input)
        x = self.slow_bn1(x)
        x = self.slow_relu(x)
        x = self.slow_maxpool(x)
        x = torch.cat([x, lateral[0]], dim=1)
        x = self.slow_res2(x)
        x = torch.cat([x, lateral[1]], dim=1)
        x = self.slow_res3(x)
        x = torch.cat([x, lateral[2]], dim=1)
        x = self.slow_res4(x)
        x = torch.cat([x, lateral[3]], dim=1)
        x = self.slow_res5(x)
        x = nn.AdaptiveAvgPool3d(1)(x)
        x = x.view(-1, x.size(1))
        return x

    def fast_path(self, input):
        lateral = []
        x = self.fast_conv1(input)
        x = self.fast_bn1(x)
        x = self.fast_relu(x)
        pool1 = self.fast_maxpool(x)
        lateral_p = self.lateral_p1(pool1)
        lateral.append(lateral_p)
        res2 = self.fast_res2(pool1)
        lateral_res2 = self.lateral_res2(res2)
        lateral.append(lateral_res2)
        res3 = self.fast_res3(res2)
        lateral_res3 = self.lateral_res3(res3)
        lateral.append(lateral_res3)
        res4 = self.fast_res4(res3)
        lateral_res4 = self.lateral_res4(res4)
        lateral.append(lateral_res4)
        res5 = self.fast_res5(res4)
        x = nn.AdaptiveAvgPool3d(1)(res5)
        x = x.view(-1, x.size(1))
        return x, lateral

    def _make_layer_fast(self, block, planes, blocks, stride=1, head_conv=1):
        downsample = None
        if stride != 1 or self.fast_inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv3d(self.fast_inplanes, planes *
                block.expansion, kernel_size=1, stride=(1, stride, stride),
                bias=False), nn.BatchNorm3d(planes * block.expansion))
        layers = []
        layers.append(block(self.fast_inplanes, planes, stride, downsample,
            head_conv=head_conv))
        self.fast_inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.fast_inplanes, planes, head_conv=
                head_conv))
        return nn.Sequential(*layers)

    def _make_layer_slow(self, block, planes, blocks, stride=1, head_conv=1):
        downsample = None
        if stride != 1 or self.slow_inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv3d(self.slow_inplanes, planes *
                block.expansion, kernel_size=1, stride=(1, stride, stride),
                bias=False), nn.BatchNorm3d(planes * block.expansion))
        layers = []
        layers.append(block(self.slow_inplanes, planes, stride, downsample,
            head_conv=head_conv))
        self.slow_inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.slow_inplanes, planes, head_conv=
                head_conv))
        self.slow_inplanes = (planes * block.expansion + planes * block.
            expansion // 8 * 2)
        return nn.Sequential(*layers)


class BranchedSoundNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, 64, stride=2, padding=32)
        self.pool1 = nn.MaxPool1d(8, stride=1, padding=0)
        self.conv2 = nn.Conv1d(16, 32, 32, stride=2, padding=16)
        self.pool2 = nn.MaxPool1d(8, stride=1, padding=0)
        self.conv3 = nn.Conv1d(32, 64, 16, stride=2, padding=8)
        self.conv4 = nn.Conv1d(64, 128, 8, stride=2, padding=4)
        self.conv5 = nn.Conv1d(128, 256, 4, stride=2, padding=2)
        self.pool5 = nn.MaxPool1d(4, stride=1, padding=0)
        self.conv6 = nn.Conv1d(256, 512, 4, stride=2, padding=2)
        self.conv7 = nn.Conv1d(512, 1024, 4, stride=2, padding=2)
        self.conv8_1 = nn.Conv1d(1024, 1000, 8, stride=2, padding=0)
        self.conv8_2 = nn.Conv1d(1024, 401, 8, stride=2, padding=0)
        self.last_linear1 = nn.Linear(262000, 1000)
        self.last_linear2 = nn.Linear(105062, 365)
        self.flatten = Flatten()

    def forward(self, input_wav):
        x = self.pool1(F.relu(nn.BatchNorm1d(16)(self.conv1(input_wav))))
        x = self.pool2(F.relu(nn.BatchNorm1d(32)(self.conv2(x))))
        x = F.relu(nn.BatchNorm1d(64)(self.conv3(x)))
        x = F.relu(nn.BatchNorm1d(128)(self.conv4(x)))
        x = self.pool5(F.relu(nn.BatchNorm1d(256)(self.conv5(x))))
        x = F.relu(nn.BatchNorm1d(512)(self.conv6(x)))
        x = F.relu(nn.BatchNorm1d(1024)(self.conv7(x)))
        x_object = self.flatten(F.relu(self.conv8_1(x)))
        x_place = self.flatten(F.relu(self.conv8_2(x)))
        x_object = self.last_linear1(x_object)
        x_place = self.last_linear2(x_place)
        y = [x_object, x_place]
        return y


class SoundNet(nn.Module):

    def __init__(self, num_classes=1000, feature_dim=262000):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.conv1 = nn.Conv1d(1, 16, 64, stride=2, padding=32)
        self.pool1 = nn.MaxPool1d(8, stride=1, padding=0)
        self.conv2 = nn.Conv1d(16, 32, 32, stride=2, padding=16)
        self.pool2 = nn.MaxPool1d(8, stride=1, padding=0)
        self.conv3 = nn.Conv1d(32, 64, 16, stride=2, padding=8)
        self.conv4 = nn.Conv1d(64, 128, 8, stride=2, padding=4)
        self.conv5 = nn.Conv1d(128, 256, 4, stride=2, padding=2)
        self.pool5 = nn.MaxPool1d(4, stride=1, padding=0)
        self.conv6 = nn.Conv1d(256, 512, 4, stride=2, padding=2)
        self.conv7 = nn.Conv1d(512, 1024, 4, stride=2, padding=2)
        self.conv8 = nn.Conv1d(1024, 1000, 8, stride=2, padding=0)
        self.last_linear = nn.Linear(feature_dim, num_classes)
        self.flatten = Flatten()
        self.fdim = feature_dim

    def features(self, x):
        x = self.pool1(F.relu(nn.BatchNorm1d(16)(self.conv1(x))))
        x = self.pool2(F.relu(nn.BatchNorm1d(32)(self.conv2(x))))
        x = F.relu(nn.BatchNorm1d(64)(self.conv3(x)))
        x = F.relu(nn.BatchNorm1d(128)(self.conv4(x)))
        x = self.pool5(F.relu(nn.BatchNorm1d(256)(self.conv5(x))))
        x = F.relu(nn.BatchNorm1d(512)(self.conv6(x)))
        x = F.relu(nn.BatchNorm1d(1024)(self.conv7(x)))
        x = self.flatten(F.relu(self.conv8(x)))
        return x

    def forward(self, x):
        x = self.features(x)
        return torch.stack([self.last_linear(i) for i in x.split(self.fdim,
            -1)[:-1] + (x[(...), -self.fdim:],)]).mean(0)


class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)


class Relation(torch.nn.Module):
    """Base relation module to model uni-directional relationships.

    A relation maps an ordered set of inputs to a single output representation
    of their uni-directional relationship.

    By convention, the relation is performed on the last two dimensions.

    input[..., num_inputs, in_features] -> output[..., -1, out_features]
    """

    def __init__(self, num_inputs, in_features, out_features,
        bottleneck_dim=512):
        super().__init__()
        self.num_inputs = num_inputs
        self.in_features = in_features
        self.out_features = out_features
        self.bottleneck_dim = bottleneck_dim
        self.relate = self.return_mlp()

    def return_mlp(self):
        return nn.Sequential(nn.ReLU(), nn.Linear(self.num_inputs * self.
            in_features, self.bottleneck_dim), nn.ReLU(), nn.Linear(self.
            bottleneck_dim, self.out_features))

    def func(self, input):
        out = self.reshape(input)
        return self.relate(out).view(input.size(0), -1, self.out_features)

    def reshape(self, input):
        return input.contiguous().view(-1, self.num_inputs * self.in_features)

    def forward(self, input):
        """Pass concatenated inputs through simple MLP."""
        return self.func(input)


class MultiScaleRelation(torch.nn.Module):
    """Multi-Relation module.

    This module applies an mlp to that concatenation of
    [2-input relation, 3-input relation, ..., n-input relation].

    """

    def __init__(self, num_input, in_features, out_features, bottleneck_dim
        =512, num_relations=3):
        super().__init__()
        self.num_input = num_input
        self.in_features = in_features
        self.out_features = out_features
        self.num_relations = num_relations
        self.bottleneck_dim = bottleneck_dim
        self.scales = list(range(num_input, 1, -1))
        self.relations_scales = []
        self.subsample_scales = []
        for scale in self.scales:
            relations_scale = self.return_relationset(num_input, scale)
            self.relations_scales.append(relations_scale)
            self.subsample_scales.append(min(self.num_relations, len(
                relations_scale)))
        self.relations = nn.ModuleList([Relation(scale, self.in_features,
            self.out_features, self.bottleneck_dim) for scale in self.scales])
        None
        None

    def forward(self, input):
        output = []
        for scale in range(len(self.scales)):
            idx_relations = np.random.choice(len(self.relations_scales[
                scale]), self.subsample_scales[scale], replace=False)
            for idx in idx_relations:
                input_relation = input[(...), (self.relations_scales[scale]
                    [idx]), :]
                output.append(self.relations[scale](input_relation))
        return torch.stack(output).sum(0).view(input.size(0), -1, self.
            out_features)

    def return_relationset(self, num_input, num_input_relation):
        return list(itertools.combinations(range(num_input),
            num_input_relation))


class HierarchicalRelation(torch.nn.Module):
    """Hierarchical relation module to model nested uni-directional relationships.
    An n-scale hierarchical relation maps an ordered set of inputs to a single
    output representation by recursively computing n-input relations on neighboring
    elements of the output of the previous level.
    """

    def __init__(self, num_inputs, in_features, out_features, relation_size
        =4, relation_dist=1, bottleneck_dim=1024):
        super().__init__()
        self.num_inputs = num_inputs
        self.in_features = in_features
        self.out_features = out_features
        self.relation_size = relation_size
        self.relation_dist = relation_dist
        self.bottleneck_dim = bottleneck_dim
        self._prepare_module()

    def _prepare_module(self):
        depth = int(np.ceil((self.num_inputs - self.relation_size) / (self.
            relation_size - 1)))
        num_inputs_final = self.num_inputs + depth * (1 - self.relation_size)
        self.relations = nn.ModuleList([Relation(self.relation_size, self.
            in_features, self.in_features) for _ in range(depth)])
        self.linears = nn.ModuleList([nn.Linear(self.in_features, self.
            out_features) for _ in range(depth)])
        self.final_linear = nn.Linear(self.in_features, self.out_features)
        self.final_relation = Relation(num_inputs_final, self.in_features,
            self.out_features)

    def forward(self, input):
        outs = []
        input = input.view(-1, self.num_inputs, self.in_features)
        for relation, linear in zip(self.relations, self.linears):
            num_inputs = range(input.size(1))
            idx_relations = list(zip(*[num_inputs[i:] for i in range(self.
                relation_size)]))
            input = torch.stack([relation(input[:, (idx), :]) for idx in
                idx_relations], 1)
            outs.append(linear(input).sum(-2))
        outs.append(self.final_relation(input))
        out = torch.stack(outs).mean(0)
        return out


class MultiScaleHierarchicalRelation(torch.nn.Module):
    """Multi-scale hierarchical relation module."""

    def __init__(self, num_inputs, in_features, out_features, relation_dist
        =1, bottleneck_dim=512):
        super(MultiScaleHierarchicalRelation, self).__init__()
        self.num_inputs = num_inputs
        self.in_features = in_features
        self.out_features = out_features
        self.relation_dist = relation_dist
        self.bottleneck_dim = bottleneck_dim
        self.scales = range(num_inputs, 1, -1)
        self.num_scales = len(self.scales)
        self.h_relations = nn.ModuleList([HierarchicalRelation(num_inputs,
            in_features, out_features, relation_size=scale, relation_dist=
            relation_dist, bottleneck_dim=bottleneck_dim) for scale in self
            .scales])
        self.final_relation = Relation(self.num_scales, out_features,
            out_features, bottleneck_dim=bottleneck_dim)

    def forward(self, input):
        input = input.contiguous().view(-1, self.num_inputs, self.in_features)
        h_outputs = torch.stack([h_rel(input) for h_rel in self.h_relations], 1
            )
        h_outputs = h_outputs.view(-1, self.num_scales, self.out_features)
        return self.final_relation(h_outputs)


class TRN(nn.Module):

    def __init__(self, num_classes, num_segments=8, arch='resnet50',
        frame_bottleneck_dim=1024, video_feature_dim=1024, consensus='HTRN',
        pretrained='moments', dropout=0.5, partial_bn=True):
        super().__init__()
        self.arch = arch
        self.reshape = True
        self.dropout = dropout
        self._enable_pbn = True
        self.consensus = consensus
        self.num_classes = num_classes
        self.num_segments = num_segments
        self.video_feature_dim = video_feature_dim
        self.frame_bottleneck_dim = frame_bottleneck_dim
        num_pc = 1000 if pretrained == 'imagenet' else 339
        self.base_model = pretrainedmodels.__dict__[arch](num_pc, pretrained)
        self.frame_feature_dim = self.base_model.last_linear.in_features
        self.base_model.last_linear = torch.nn.Dropout(self.dropout)
        self.std = self.base_model.std
        self.mean = self.base_model.mean
        self.input_size = self.base_model.input_size[1:]
        self.input_space = self.base_model.input_space
        consensus_mods = {'TRN': Relation, 'HTRN': HierarchicalRelation,
            'MSTRN': MultiScaleRelation, 'MSHTRN':
            MultiScaleHierarchicalRelation}
        try:
            temporal_relation = consensus_mods[consensus]
        except KeyError:
            raise ValueError('Unrecognized temporal consensus.')
        else:
            self.temporal_relation = temporal_relation(self.num_segments,
                self.frame_feature_dim, self.video_feature_dim, self.
                frame_bottleneck_dim)
            self.last_linear = nn.Linear(self.video_feature_dim, self.
                num_classes)
        None

    def features(self, input):
        batch_size = input.size(0)
        base_rep = self.base_model(input.view((-1, 3) + input.size()[-2:]))
        base_rep = base_rep.view(batch_size, -1, self.num_segments,
            base_rep.size(-1))
        num_inputs = base_rep.size(1)
        t_in = base_rep.view(-1, num_inputs, self.num_segments, base_rep.
            size(-1))
        return self.temporal_relation(t_in).squeeze()

    def logits(self, features):
        return self.last_linear(features)

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super().train(mode)
        count = 0
        if self._enable_pbn:
            None
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable

    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        bn = []
        conv_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d
                ):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.BatchNorm1d):
                bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm2d):
                bn_cnt += 1
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError(
                        'New atomic module type: {}. Need to give it a learning policy'
                        .format(type(m)))
        return [{'params': first_conv_weight, 'lr_mult': 1, 'decay_mult': 1,
            'name': 'first_conv_weight'}, {'params': first_conv_bias,
            'lr_mult': 2, 'decay_mult': 0, 'name': 'first_conv_bias'}, {
            'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1, 'name':
            'normal_weight'}, {'params': normal_bias, 'lr_mult': 2,
            'decay_mult': 0, 'name': 'normal_bias'}, {'params': bn,
            'lr_mult': 1, 'decay_mult': 0, 'name': 'BN scale/shift'}]

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size[0] * 256 // 224


class Identity(torch.nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class SpatialCrossMapLRN(nn.Module):

    def __init__(self, local_size=1, alpha=1.0, beta=0.75, k=1,
        ACROSS_CHANNELS=True):
        super(SpatialCrossMapLRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if ACROSS_CHANNELS:
            self.average = nn.AvgPool3d(kernel_size=(local_size, 1, 1),
                stride=1, padding=(int((local_size - 1.0) / 2), 0, 0))
        else:
            self.average = nn.AvgPool2d(kernel_size=local_size, stride=1,
                padding=int((local_size - 1.0) / 2))
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


class LambdaBase(nn.Sequential):

    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input


class VGGM(nn.Module):

    def __init__(self, num_classes=1000):
        super(VGGM, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(nn.Conv2d(3, 96, (7, 7), (2, 2)), nn.
            ReLU(), SpatialCrossMapLRN(5, 0.0005, 0.75, 2), nn.MaxPool2d((3,
            3), (2, 2), (0, 0), ceil_mode=True), nn.Conv2d(96, 256, (5, 5),
            (2, 2), (1, 1)), nn.ReLU(), SpatialCrossMapLRN(5, 0.0005, 0.75,
            2), nn.MaxPool2d((3, 3), (2, 2), (0, 0), ceil_mode=True), nn.
            Conv2d(256, 512, (3, 3), (1, 1), (1, 1)), nn.ReLU(), nn.Conv2d(
            512, 512, (3, 3), (1, 1), (1, 1)), nn.ReLU(), nn.Conv2d(512, 
            512, (3, 3), (1, 1), (1, 1)), nn.ReLU(), nn.MaxPool2d((3, 3), (
            2, 2), (0, 0), ceil_mode=True))
        self.classif = nn.Sequential(nn.Linear(18432, 4096), nn.ReLU(), nn.
            Dropout(0.5), nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, num_classes))

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


class WideBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(WideBottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size
            =1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

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
        out += residual
        out = self.relu(out)
        return out


class WideResNet(nn.Module):

    def __init__(self, block, layers, k=1, shortcut_type='B', num_classes=400):
        self.inplanes = 64
        super(WideResNet, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=7, stride=(1, 2, 2),
            padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64 * k, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128 * k, layers[1],
            shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256 * k, layers[2],
            shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, 512 * k, layers[3],
            shortcut_type, stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(512 * k * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(downsample_basic_block, planes=planes *
                    block.expansion, stride=stride)
            else:
                downsample = nn.Sequential(nn.Conv3d(self.inplanes, planes *
                    block.expansion, kernel_size=1, stride=stride, bias=
                    False), nn.BatchNorm3d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
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


class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
        padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size,
            stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1,
            bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):

    def __init__(self, in_filters, out_filters, reps, strides=1,
        start_with_relu=True, grow_first=True):
        super(Block, self).__init__()
        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=
                strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None
        self.relu = nn.ReLU(inplace=True)
        rep = []
        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1,
                padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters
        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, stride=1,
                padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))
        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1,
                padding=1, bias=False))
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
        self.block1 = Block(64, 128, 2, 2, start_with_relu=False,
            grow_first=True)
        self.block2 = Block(128, 256, 2, 2, start_with_relu=True,
            grow_first=True)
        self.block3 = Block(256, 728, 2, 2, start_with_relu=True,
            grow_first=True)
        self.block4 = Block(728, 728, 3, 1, start_with_relu=True,
            grow_first=True)
        self.block5 = Block(728, 728, 3, 1, start_with_relu=True,
            grow_first=True)
        self.block6 = Block(728, 728, 3, 1, start_with_relu=True,
            grow_first=True)
        self.block7 = Block(728, 728, 3, 1, start_with_relu=True,
            grow_first=True)
        self.block8 = Block(728, 728, 3, 1, start_with_relu=True,
            grow_first=True)
        self.block9 = Block(728, 728, 3, 1, start_with_relu=True,
            grow_first=True)
        self.block10 = Block(728, 728, 3, 1, start_with_relu=True,
            grow_first=True)
        self.block11 = Block(728, 728, 3, 1, start_with_relu=True,
            grow_first=True)
        self.block12 = Block(728, 1024, 2, 2, start_with_relu=True,
            grow_first=False)
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
