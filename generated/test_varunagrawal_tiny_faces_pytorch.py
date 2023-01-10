import sys
_module = sys.modules[__name__]
del sys
datasets = _module
processor = _module
wider_face = _module
evaluate = _module
main = _module
models = _module
loss = _module
model = _module
utils = _module
trainer = _module
cluster = _module
dense_overlap = _module
k_medoids = _module
metrics = _module
nms = _module
test_dense_overlap = _module
test_metrics = _module
visualize = _module

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


from torch.utils import data


from copy import deepcopy


import logging


import torch


from torch.utils.data import dataset


from torchvision import transforms


from torch import optim


from torch import nn


from torchvision.models import resnet50


from torchvision.models import resnet101


from torch.nn import functional as nnfunc


class AvgMeter:

    def __init__(self):
        self.average = 0
        self.num_averaged = 0

    def update(self, loss, size):
        n = self.num_averaged
        m = n + size
        self.average = (n * self.average + float(loss)) / m
        self.num_averaged = m

    def reset(self):
        self.average = 0
        self.num_averaged = 0


def shuffle_index(n, n_out):
    """
    Randomly shuffle the indices and return a subset of them
    :param n: The number of indices to shuffle.
    :param n_out: The number of output indices.
    :return:
    """
    n = int(n)
    n_out = int(n_out)
    if n == 0 or n_out == 0:
        return np.empty(0)
    x = np.random.permutation(n)
    assert n_out <= n
    if n_out != n:
        x = x[:n_out]
    return x


def balance_sampling(label_cls, pos_fraction, sample_size=256):
    """
    Perform balance sampling by always sampling `pos_fraction` positive samples and
    `(1-pos_fraction)` negative samples from the input
    :param label_cls: Class labels as numpy.array.
    :param pos_fraction: The maximum fraction of positive samples to keep.
    :return:
    """
    pos_maxnum = sample_size * pos_fraction
    pos_idx_unraveled = np.where(label_cls == 1)
    pos_idx = np.array(np.ravel_multi_index(pos_idx_unraveled, label_cls.shape))
    if pos_idx.size > pos_maxnum:
        didx = shuffle_index(pos_idx.size, pos_idx.size - pos_maxnum)
        pos_idx_unraveled = np.unravel_index(pos_idx[didx], label_cls.shape)
        label_cls[pos_idx_unraveled] = 0
    neg_maxnum = pos_maxnum * (1 - pos_fraction) / pos_fraction
    neg_idx_unraveled = np.where(label_cls == -1)
    neg_idx = np.array(np.ravel_multi_index(neg_idx_unraveled, label_cls.shape))
    if neg_idx.size > neg_maxnum:
        ridx = shuffle_index(neg_idx.size, neg_maxnum)
        didx = np.arange(0, neg_idx.size)
        didx = np.delete(didx, ridx)
        neg_idx = np.unravel_index(neg_idx[didx], label_cls.shape)
        label_cls[neg_idx] = 0
    return label_cls


class DetectionCriterion(nn.Module):
    """
    The loss for the Tiny Faces detector
    """

    def __init__(self, n_templates=25, reg_weight=1, pos_fraction=0.5):
        super().__init__()
        self.regression_criterion = nn.SmoothL1Loss(reduction='none')
        self.classification_criterion = nn.SoftMarginLoss(reduction='none')
        self.n_templates = n_templates
        self.reg_weight = reg_weight
        self.pos_fraction = pos_fraction
        self.class_average = AvgMeter()
        self.reg_average = AvgMeter()
        self.masked_class_loss = None
        self.masked_reg_loss = None
        self.total_loss = None

    def balance_sample(self, class_map):
        device = class_map.device
        label_class_np = class_map.cpu().numpy()
        for idx in range(label_class_np.shape[0]):
            label_class_np[idx, ...] = balance_sampling(label_class_np[idx, ...], pos_fraction=self.pos_fraction)
        class_map = torch.from_numpy(label_class_np)
        return class_map

    def hard_negative_mining(self, classification, class_map):
        loss_class_map = nn.functional.soft_margin_loss(classification.detach(), class_map, reduction='none')
        class_map[loss_class_map < 0.03] = 0
        return class_map

    def forward(self, output, class_map, regression_map):
        classification = output[:, 0:self.n_templates, :, :]
        regression = output[:, self.n_templates:, :, :]
        class_map = self.hard_negative_mining(classification, class_map)
        class_map = self.balance_sample(class_map)
        class_loss = self.classification_criterion(classification, class_map)
        class_mask = (class_map != 0).type(output.dtype)
        self.masked_class_loss = class_mask * class_loss
        reg_loss = self.regression_criterion(regression, regression_map)
        reg_mask = (class_map > 0).repeat(1, 4, 1, 1).type(output.dtype)
        self.masked_reg_loss = reg_mask * reg_loss
        self.total_loss = self.masked_class_loss.sum() + self.reg_weight * self.masked_reg_loss.sum()
        self.class_average.update(self.masked_class_loss.sum(), output.size(0))
        self.reg_average.update(self.masked_reg_loss.sum(), output.size(0))
        return self.total_loss

    def reset(self):
        self.class_average.reset()
        self.reg_average.reset()


class DetectionModel(nn.Module):
    """
    Hybrid Model from Tiny Faces paper
    """

    def __init__(self, base_model=resnet101, num_templates=1, num_objects=1):
        super().__init__()
        output = (num_objects + 4) * num_templates
        self.model = base_model(pretrained=True)
        del self.model.layer4
        self.score_res3 = nn.Conv2d(in_channels=512, out_channels=output, kernel_size=1, padding=0)
        self.score_res4 = nn.Conv2d(in_channels=1024, out_channels=output, kernel_size=1, padding=0)
        self.score4_upsample = nn.ConvTranspose2d(in_channels=output, out_channels=output, kernel_size=4, stride=2, padding=1, bias=False)
        self._init_bilinear()

    def _init_weights(self):
        pass

    def _init_bilinear(self):
        """
        Initialize the ConvTranspose2d layer with a bilinear interpolation mapping
        :return:
        """
        k = self.score4_upsample.kernel_size[0]
        factor = np.floor((k + 1) / 2)
        if k % 2 == 1:
            center = factor
        else:
            center = factor + 0.5
        C = np.arange(1, 5)
        f = np.zeros((self.score4_upsample.in_channels, self.score4_upsample.out_channels, k, k))
        for i in range(self.score4_upsample.out_channels):
            f[i, i, :, :] = (np.ones((1, k)) - np.abs(C - center) / factor).T @ (np.ones((1, k)) - np.abs(C - center) / factor)
        self.score4_upsample.weight = torch.nn.Parameter(data=torch.Tensor(f))

    def learnable_parameters(self, lr):
        parameters = [{'params': self.model.parameters(), 'lr': lr}, {'params': self.score_res3.parameters(), 'lr': 0.1 * lr}, {'params': self.score_res4.parameters(), 'lr': 1 * lr}, {'params': self.score4_upsample.parameters(), 'lr': 0}]
        return parameters

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        res3 = x
        x = self.model.layer3(x)
        res4 = x
        score_res3 = self.score_res3(res3)
        score_res4 = self.score_res4(res4)
        score4 = self.score4_upsample(score_res4)
        if not self.training:
            cropv = score4.size(2) - score_res3.size(2)
            cropu = score4.size(3) - score_res3.size(3)
            if cropv == 0:
                cropv = -score4.size(2)
            if cropu == 0:
                cropu = -score4.size(3)
            score4 = score4[:, :, 0:-cropv, 0:-cropu]
        else:
            score4 = score4[:, :, 0:score_res3.size(2), 0:score_res3.size(3)]
        score = score_res3 + score4
        return score


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (DetectionModel,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
]

class Test_varunagrawal_tiny_faces_pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

