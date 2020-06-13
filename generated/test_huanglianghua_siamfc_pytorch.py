import sys
_module = sys.modules[__name__]
del sys
siamfc = _module
backbones = _module
datasets = _module
heads = _module
losses = _module
ops = _module
siamfc = _module
transforms = _module
demo = _module
test = _module
train = _module

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


import torch.nn as nn


import torch.nn.functional as F


import torch


import numpy as np


import torch.optim as optim


from collections import namedtuple


from torch.optim.lr_scheduler import ExponentialLR


from torch.utils.data import DataLoader


class _BatchNorm2d(nn.BatchNorm2d):

    def __init__(self, num_features, *args, **kwargs):
        super(_BatchNorm2d, self).__init__(num_features, *args, eps=1e-06,
            momentum=0.05, **kwargs)


class _AlexNet(nn.Module):

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x


class SiamFC(nn.Module):

    def __init__(self, out_scale=0.001):
        super(SiamFC, self).__init__()
        self.out_scale = out_scale

    def forward(self, z, x):
        return self._fast_xcorr(z, x) * self.out_scale

    def _fast_xcorr(self, z, x):
        nz = z.size(0)
        nx, c, h, w = x.size()
        x = x.view(-1, nz * c, h, w)
        out = F.conv2d(x, z, groups=nz)
        out = out.view(nx, -1, out.size(-2), out.size(-1))
        return out


class BalancedLoss(nn.Module):

    def __init__(self, neg_weight=1.0):
        super(BalancedLoss, self).__init__()
        self.neg_weight = neg_weight

    def forward(self, input, target):
        pos_mask = target == 1
        neg_mask = target == 0
        pos_num = pos_mask.sum().float()
        neg_num = neg_mask.sum().float()
        weight = target.new_zeros(target.size())
        weight[pos_mask] = 1 / pos_num
        weight[neg_mask] = 1 / neg_num * self.neg_weight
        weight /= weight.sum()
        return F.binary_cross_entropy_with_logits(input, target, weight,
            reduction='sum')


def log_sigmoid(x):
    return torch.clamp(x, max=0) - torch.log(1 + torch.exp(-torch.abs(x))
        ) + 0.5 * torch.clamp(x, min=0, max=0)


def log_minus_sigmoid(x):
    return torch.clamp(-x, max=0) - torch.log(1 + torch.exp(-torch.abs(x))
        ) + 0.5 * torch.clamp(x, min=0, max=0)


class FocalLoss(nn.Module):

    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, input, target):
        pos_log_sig = log_sigmoid(input)
        neg_log_sig = log_minus_sigmoid(input)
        prob = torch.sigmoid(input)
        pos_weight = torch.pow(1 - prob, self.gamma)
        neg_weight = torch.pow(prob, self.gamma)
        loss = -(target * pos_weight * pos_log_sig + (1 - target) *
            neg_weight * neg_log_sig)
        avg_weight = target * pos_weight + (1 - target) * neg_weight
        loss /= avg_weight.mean()
        return loss.mean()


class GHMCLoss(nn.Module):

    def __init__(self, bins=30, momentum=0.5):
        super(GHMCLoss, self).__init__()
        self.bins = bins
        self.momentum = momentum
        self.edges = [(t / bins) for t in range(bins + 1)]
        self.edges[-1] += 1e-06
        if momentum > 0:
            self.acc_sum = [(0.0) for _ in range(bins)]

    def forward(self, input, target):
        edges = self.edges
        mmt = self.momentum
        weights = torch.zeros_like(input)
        g = torch.abs(input.sigmoid().detach() - target)
        tot = input.numel()
        n = 0
        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i + 1])
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] + (1 - mmt
                        ) * num_in_bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
                n += 1
        if n > 0:
            weights /= weights.mean()
        loss = F.binary_cross_entropy_with_logits(input, target, weights,
            reduction='sum') / tot
        return loss


class OHNMLoss(nn.Module):

    def __init__(self, neg_ratio=3.0):
        super(OHNMLoss, self).__init__()
        self.neg_ratio = neg_ratio

    def forward(self, input, target):
        pos_logits = input[target > 0]
        pos_labels = target[target > 0]
        neg_logits = input[target == 0]
        neg_labels = target[target == 0]
        pos_num = pos_logits.numel()
        neg_num = int(pos_num * self.neg_ratio)
        neg_logits, neg_indices = neg_logits.topk(neg_num)
        neg_labels = neg_labels[neg_indices]
        loss = F.binary_cross_entropy_with_logits(torch.cat([pos_logits,
            neg_logits]), torch.cat([pos_labels, neg_labels]), reduction='mean'
            )
        return loss


class Net(nn.Module):

    def __init__(self, backbone, head):
        super(Net, self).__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, z, x):
        z = self.backbone(z)
        x = self.backbone(x)
        return self.head(z, x)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_huanglianghua_siamfc_pytorch(_paritybench_base):
    pass
    def test_000(self):
        self._check(_BatchNorm2d(*[], **{'num_features': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(SiamFC(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(BalancedLoss(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(FocalLoss(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_004(self):
        self._check(GHMCLoss(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

