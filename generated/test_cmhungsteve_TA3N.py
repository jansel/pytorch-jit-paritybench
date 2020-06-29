import sys
_module = sys.modules[__name__]
del sys
TRNmodule = _module
dataset = _module
C3D_model = _module
dataset2split = _module
list_ucf_hmdb_full2DA = _module
video2feature = _module
video_dataset2list = _module
loss = _module
main = _module
models = _module
opts = _module
test_models = _module
video_processing = _module
utils = _module

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


import torch.nn as nn


import numpy as np


from math import ceil


import time


import torch.backends.cudnn as cudnn


from torch.autograd import Variable


import torch.nn.functional as F


import torch.nn.parallel


import torch.optim


from torch.nn.utils import clip_grad_norm_


import math


from torch import nn


from torch.nn.init import *


from torch.autograd import Function


from time import sleep


class RelationModule(torch.nn.Module):

    def __init__(self, img_feature_dim, num_bottleneck, num_frames):
        super(RelationModule, self).__init__()
        self.num_frames = num_frames
        self.img_feature_dim = img_feature_dim
        self.num_bottleneck = num_bottleneck
        self.classifier = self.fc_fusion()

    def fc_fusion(self):
        classifier = nn.Sequential(nn.ReLU(), nn.Linear(self.num_frames *
            self.img_feature_dim, self.num_bottleneck), nn.ReLU())
        return classifier

    def forward(self, input):
        input = input.view(input.size(0), self.num_frames * self.
            img_feature_dim)
        input = self.classifier(input)
        return input


class C3D(nn.Module):
    """
    The C3D network as described in [1].
    """

    def __init__(self):
        super(C3D, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 
            1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1,
            1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1,
            1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1,
            1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1,
            1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1,
            1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1,
            1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2),
            padding=(0, 1, 1))
        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 487)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        h = self.relu(self.conv1(x))
        h = self.pool1(h)
        h = self.relu(self.conv2(h))
        h = self.pool2(h)
        h = self.relu(self.conv3a(h))
        h = self.relu(self.conv3b(h))
        h = self.pool3(h)
        h = self.relu(self.conv4a(h))
        h = self.relu(self.conv4b(h))
        h = self.pool4(h)
        h = self.relu(self.conv5a(h))
        h = self.relu(self.conv5b(h))
        h = self.pool5(h)
        h = h.view(-1, 8192)
        h = self.relu(self.fc6(h))
        h = self.dropout(h)
        h = self.relu(self.fc7(h))
        h = self.dropout(h)
        logits = self.fc8(h)
        probs = self.softmax(logits)
        return probs


class TCL(nn.Module):

    def __init__(self, conv_size, dim):
        super(TCL, self).__init__()
        self.conv2d = nn.Conv2d(dim, dim, kernel_size=(conv_size, 1),
            padding=(conv_size // 2, 0))
        kaiming_normal_(self.conv2d.weight)

    def forward(self, x):
        x = self.conv2d(x)
        return x


class GradReverse(Function):

    @staticmethod
    def forward(ctx, x, beta):
        ctx.beta = beta
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.neg() * ctx.beta
        return grad_input, None


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_cmhungsteve_TA3N(_paritybench_base):
    pass
    def test_000(self):
        self._check(RelationModule(*[], **{'img_feature_dim': 4, 'num_bottleneck': 4, 'num_frames': 4}), [torch.rand([4, 4, 4])], {})

    def test_001(self):
        self._check(TCL(*[], **{'conv_size': 4, 'dim': 4}), [torch.rand([4, 4, 4, 4])], {})

