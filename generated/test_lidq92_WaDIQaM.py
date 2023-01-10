import sys
_module = sys.modules[__name__]
del sys
main = _module
test = _module
test_cross_dataset = _module

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


import numpy as np


import random


from scipy import stats


import torch


from torch import nn


import torch.nn.functional as F


from torch.optim import Adam


from torch.optim import lr_scheduler


from torch.utils.data import Dataset


from torchvision.transforms.functional import to_tensor


class FRnet(nn.Module):
    """
    (Wa)DIQaM-FR Model
    """

    def __init__(self, weighted_average=True):
        """
        :param weighted_average: weighted average or not?
        """
        super(FRnet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv7 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv8 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv9 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv10 = nn.Conv2d(512, 512, 3, padding=1)
        self.fc1_q = nn.Linear(512 * 3, 512)
        self.fc2_q = nn.Linear(512, 1)
        self.fc1_w = nn.Linear(512 * 3, 512)
        self.fc2_w = nn.Linear(512, 1)
        self.dropout = nn.Dropout()
        self.weighted_average = weighted_average

    def extract_features(self, x):
        """
        feature extraction
        :param x: the input image
        :return: the output feature
        """
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.max_pool2d(h, 2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pool2d(h, 2)
        h = F.relu(self.conv5(h))
        h = F.relu(self.conv6(h))
        h = F.max_pool2d(h, 2)
        h = F.relu(self.conv7(h))
        h = F.relu(self.conv8(h))
        h = F.max_pool2d(h, 2)
        h = F.relu(self.conv9(h))
        h = F.relu(self.conv10(h))
        h = F.max_pool2d(h, 2)
        h = h.view(-1, 512)
        return h

    def forward(self, data):
        """
        :param data: distorted and reference patches of images
        :return: quality of images/patches
        """
        x, x_ref = data
        batch_size = x.size(0)
        n_patches = x.size(1)
        if self.weighted_average:
            q = torch.ones((batch_size, 1), device=x.device)
        else:
            q = torch.ones((batch_size * n_patches, 1), device=x.device)
        for i in range(batch_size):
            h = self.extract_features(x[i])
            h_ref = self.extract_features(x_ref[i])
            h = torch.cat((h - h_ref, h, h_ref), 1)
            h_ = h
            h = F.relu(self.fc1_q(h_))
            h = self.dropout(h)
            h = self.fc2_q(h)
            if self.weighted_average:
                w = F.relu(self.fc1_w(h_))
                w = self.dropout(w)
                w = F.relu(self.fc2_w(w)) + 1e-06
                q[i] = torch.sum(h * w) / torch.sum(w)
            else:
                q[i * n_patches:(i + 1) * n_patches] = h
        return q


class NRnet(nn.Module):
    """
    (Wa)DIQaM-NR Model
    """

    def __init__(self, weighted_average=True):
        """
        :param weighted_average: weighted average or not?
        """
        super(NRnet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv7 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv8 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv9 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv10 = nn.Conv2d(512, 512, 3, padding=1)
        self.fc1q_nr = nn.Linear(512, 512)
        self.fc2q_nr = nn.Linear(512, 1)
        self.fc1w_nr = nn.Linear(512, 512)
        self.fc2w_nr = nn.Linear(512, 1)
        self.dropout = nn.Dropout()
        self.weighted_average = weighted_average

    def extract_features(self, x):
        """
        feature extraction
        :param x: the input image
        :return: the output feature
        """
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.max_pool2d(h, 2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pool2d(h, 2)
        h = F.relu(self.conv5(h))
        h = F.relu(self.conv6(h))
        h = F.max_pool2d(h, 2)
        h = F.relu(self.conv7(h))
        h = F.relu(self.conv8(h))
        h = F.max_pool2d(h, 2)
        h = F.relu(self.conv9(h))
        h = F.relu(self.conv10(h))
        h = F.max_pool2d(h, 2)
        h = h.view(-1, 512)
        return h

    def forward(self, x):
        """
        :param data: distorted and reference patches of images
        :return: quality of images/patches
        """
        batch_size = x.size(0)
        n_patches = x.size(1)
        if self.weighted_average:
            q = torch.ones((batch_size, 1), device=x.device)
        else:
            q = torch.ones((batch_size * n_patches, 1), device=x.device)
        for i in range(batch_size):
            h = self.extract_features(x[i])
            h_ = h
            h = F.relu(self.fc1q_nr(h_))
            h = self.dropout(h)
            h = self.fc2q_nr(h)
            if self.weighted_average:
                w = F.relu(self.fc1w_nr(h_))
                w = self.dropout(w)
                w = F.relu(self.fc2w_nr(w)) + 1e-06
                q[i] = torch.sum(h * w) / torch.sum(w)
            else:
                q[i * n_patches:(i + 1) * n_patches] = h
        return q


class IQALoss(torch.nn.Module):

    def __init__(self):
        super(IQALoss, self).__init__()

    def forward(self, y_pred, y):
        """
        loss function, e.g., l1 loss
        :param y_pred: predicted values
        :param y: y[0] is the ground truth label
        :return: the calculated loss
        """
        n = int(y_pred.size(0) / y[0].size(0))
        loss = F.l1_loss(y_pred, y[0].repeat((1, n)).reshape((-1, 1)))
        return loss


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (IQALoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4]), torch.rand([4, 4, 4])], {}),
     True),
    (NRnet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
]

class Test_lidq92_WaDIQaM(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

