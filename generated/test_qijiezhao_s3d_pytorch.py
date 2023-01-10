import sys
_module = sys.modules[__name__]
del sys
S3DG_Pytorch = _module
config = _module
dataloader_pkl = _module
log = _module
ops = _module
basic_ops = _module
utils = _module
draw_actionness = _module
eval_mAP = _module
eval_mAP_perframe_linear = _module
get_proposals_paper = _module
opts = _module
visualize_tool = _module
watershed = _module
trainval = _module
transforms = _module

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


import torch.nn as nn


import torch


import torch.utils.data as data


import random


import numpy as np


import collections


import math


from sklearn.metrics import confusion_matrix


from sklearn.metrics import average_precision_score


import time


import torchvision


import torch.nn.parallel


import torch.backends.cudnn as cudnn


import torch.optim


from torch.nn.utils import clip_grad_norm


import numbers


class BasicConv3d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm3d(out_planes, eps=0.001, momentum=0.001, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class STConv3d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(STConv3d, self).__init__()
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=(1, kernel_size, kernel_size), stride=(1, stride, stride), padding=(0, padding, padding))
        self.conv2 = nn.Conv3d(out_planes, out_planes, kernel_size=(kernel_size, 1, 1), stride=(stride, 1, 1), padding=(padding, 0, 0))
        self.bn = nn.BatchNorm3d(out_planes, eps=0.001, momentum=0.001, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm3d(out_planes, eps=0.001, momentum=0.001, affine=True)
        self.relu2 = nn.ReLU(inplace=True)
        nn.init.normal(self.conv2.weight, mean=0, std=0.01)
        nn.init.constant(self.conv2.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


class Mixed_3b(nn.Module):

    def __init__(self):
        super(Mixed_3b, self).__init__()
        self.branch0 = nn.Sequential(BasicConv3d(192, 64, kernel_size=1, stride=1))
        self.branch1 = nn.Sequential(BasicConv3d(192, 96, kernel_size=1, stride=1), STConv3d(96, 128, kernel_size=3, stride=1, padding=1))
        self.branch2 = nn.Sequential(BasicConv3d(192, 16, kernel_size=1, stride=1), STConv3d(16, 32, kernel_size=3, stride=1, padding=1))
        self.branch3 = nn.Sequential(nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1), BasicConv3d(192, 32, kernel_size=1, stride=1))

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_3c(nn.Module):

    def __init__(self):
        super(Mixed_3c, self).__init__()
        self.branch0 = nn.Sequential(BasicConv3d(256, 128, kernel_size=1, stride=1))
        self.branch1 = nn.Sequential(BasicConv3d(256, 128, kernel_size=1, stride=1), STConv3d(128, 192, kernel_size=3, stride=1, padding=1))
        self.branch2 = nn.Sequential(BasicConv3d(256, 32, kernel_size=1, stride=1), STConv3d(32, 96, kernel_size=3, stride=1, padding=1))
        self.branch3 = nn.Sequential(nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1), BasicConv3d(256, 64, kernel_size=1, stride=1))

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_4b(nn.Module):

    def __init__(self):
        super(Mixed_4b, self).__init__()
        self.branch0 = nn.Sequential(BasicConv3d(480, 192, kernel_size=1, stride=1))
        self.branch1 = nn.Sequential(BasicConv3d(480, 96, kernel_size=1, stride=1), STConv3d(96, 208, kernel_size=3, stride=1, padding=1))
        self.branch2 = nn.Sequential(BasicConv3d(480, 16, kernel_size=1, stride=1), STConv3d(16, 48, kernel_size=3, stride=1, padding=1))
        self.branch3 = nn.Sequential(nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1), BasicConv3d(480, 64, kernel_size=1, stride=1))

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_4c(nn.Module):

    def __init__(self):
        super(Mixed_4c, self).__init__()
        self.branch0 = nn.Sequential(BasicConv3d(512, 160, kernel_size=1, stride=1))
        self.branch1 = nn.Sequential(BasicConv3d(512, 112, kernel_size=1, stride=1), STConv3d(112, 224, kernel_size=3, stride=1, padding=1))
        self.branch2 = nn.Sequential(BasicConv3d(512, 24, kernel_size=1, stride=1), STConv3d(24, 64, kernel_size=3, stride=1, padding=1))
        self.branch3 = nn.Sequential(nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1), BasicConv3d(512, 64, kernel_size=1, stride=1))

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_4d(nn.Module):

    def __init__(self):
        super(Mixed_4d, self).__init__()
        self.branch0 = nn.Sequential(BasicConv3d(512, 128, kernel_size=1, stride=1))
        self.branch1 = nn.Sequential(BasicConv3d(512, 128, kernel_size=1, stride=1), STConv3d(128, 256, kernel_size=3, stride=1, padding=1))
        self.branch2 = nn.Sequential(BasicConv3d(512, 24, kernel_size=1, stride=1), STConv3d(24, 64, kernel_size=3, stride=1, padding=1))
        self.branch3 = nn.Sequential(nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1), BasicConv3d(512, 64, kernel_size=1, stride=1))

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_4e(nn.Module):

    def __init__(self):
        super(Mixed_4e, self).__init__()
        self.branch0 = nn.Sequential(BasicConv3d(512, 112, kernel_size=1, stride=1))
        self.branch1 = nn.Sequential(BasicConv3d(512, 144, kernel_size=1, stride=1), STConv3d(144, 288, kernel_size=3, stride=1, padding=1))
        self.branch2 = nn.Sequential(BasicConv3d(512, 32, kernel_size=1, stride=1), STConv3d(32, 64, kernel_size=3, stride=1, padding=1))
        self.branch3 = nn.Sequential(nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1), BasicConv3d(512, 64, kernel_size=1, stride=1))

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_4f(nn.Module):

    def __init__(self):
        super(Mixed_4f, self).__init__()
        self.branch0 = nn.Sequential(BasicConv3d(528, 256, kernel_size=1, stride=1))
        self.branch1 = nn.Sequential(BasicConv3d(528, 160, kernel_size=1, stride=1), STConv3d(160, 320, kernel_size=3, stride=1, padding=1))
        self.branch2 = nn.Sequential(BasicConv3d(528, 32, kernel_size=1, stride=1), STConv3d(32, 128, kernel_size=3, stride=1, padding=1))
        self.branch3 = nn.Sequential(nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1), BasicConv3d(528, 128, kernel_size=1, stride=1))

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_5b(nn.Module):

    def __init__(self):
        super(Mixed_5b, self).__init__()
        self.branch0 = nn.Sequential(BasicConv3d(832, 256, kernel_size=1, stride=1))
        self.branch1 = nn.Sequential(BasicConv3d(832, 160, kernel_size=1, stride=1), STConv3d(160, 320, kernel_size=3, stride=1, padding=1))
        self.branch2 = nn.Sequential(BasicConv3d(832, 32, kernel_size=1, stride=1), STConv3d(32, 128, kernel_size=3, stride=1, padding=1))
        self.branch3 = nn.Sequential(nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1), BasicConv3d(832, 128, kernel_size=1, stride=1))

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_5c(nn.Module):

    def __init__(self):
        super(Mixed_5c, self).__init__()
        self.branch0 = nn.Sequential(BasicConv3d(832, 384, kernel_size=1, stride=1))
        self.branch1 = nn.Sequential(BasicConv3d(832, 192, kernel_size=1, stride=1), STConv3d(192, 384, kernel_size=3, stride=1, padding=1))
        self.branch2 = nn.Sequential(BasicConv3d(832, 48, kernel_size=1, stride=1), STConv3d(48, 128, kernel_size=3, stride=1, padding=1))
        self.branch3 = nn.Sequential(nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1), BasicConv3d(832, 128, kernel_size=1, stride=1))

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class S3DG(nn.Module):

    def __init__(self, num_classes=400, dropout_keep_prob=1, input_channel=3, spatial_squeeze=True):
        super(S3DG, self).__init__()
        self.features = nn.Sequential(STConv3d(input_channel, 64, kernel_size=7, stride=2, padding=3), nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)), BasicConv3d(64, 64, kernel_size=1, stride=1), STConv3d(64, 192, kernel_size=3, stride=1, padding=1), nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)), Mixed_3b(), Mixed_3c(), nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)), Mixed_4b(), Mixed_4c(), Mixed_4d(), Mixed_4e(), Mixed_4f(), nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0)), Mixed_5b(), Mixed_5c(), nn.AvgPool3d(kernel_size=(2, 7, 7), stride=1), nn.Dropout3d(dropout_keep_prob), nn.Conv3d(1024, num_classes, kernel_size=1, stride=1, bias=True))
        self.spatial_squeeze = spatial_squeeze
        self.softmax = nn.Softmax()

    def forward(self, x):
        logits = self.features(x)
        if self.spatial_squeeze:
            logits = logits.squeeze(3)
            logits = logits.squeeze(3)
        averaged_logits = torch.mean(logits, 2)
        predictions = self.softmax(averaged_logits)
        return predictions, averaged_logits

    def load_state_dict(self, path):
        target_weights = torch.load(path)
        own_state = self.state_dict()
        for name, param in list(target_weights.items()):
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    if len(param.size()) == 5 and param.size()[3] in [3, 7]:
                        own_state[name][:, :, 0, :, :] = torch.mean(param, 2)
                    else:
                        own_state[name].copy_(param)
                except Exception:
                    raise RuntimeError('While copying the parameter named {}.                                       whose dimensions in the model are {} and                                        whose dimensions in the checkpoint are {}.                                       '.format(name, own_state[name].size(), param.size()))
            else:
                None
        missing = set(own_state.keys()) - set(target_weights.keys())
        None


class Identity(torch.nn.Module):

    def forward(self, input):
        return input


class SegmentConsensus(torch.autograd.Function):

    def __init__(self, consensus_type, dim=1):
        self.consensus_type = consensus_type
        self.dim = dim
        self.shape = None

    def forward(self, input_tensor):
        self.shape = input_tensor.size()
        if self.consensus_type == 'avg':
            output = input_tensor.mean(dim=self.dim, keepdim=True)
        elif self.consensus_type == 'identity':
            output = input_tensor
        else:
            output = None
        return output

    def backward(self, grad_output):
        if self.consensus_type == 'avg':
            grad_in = grad_output.expand(self.shape) / float(self.shape[self.dim])
        elif self.consensus_type == 'identity':
            grad_in = grad_output
        else:
            grad_in = None
        return grad_in


class ConsensusModule(torch.nn.Module):

    def __init__(self, consensus_type, dim=1):
        super(ConsensusModule, self).__init__()
        self.consensus_type = consensus_type if consensus_type != 'rnn' else 'identity'
        self.dim = dim

    def forward(self, input):
        return SegmentConsensus(self.consensus_type, self.dim)(input)

