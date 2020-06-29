import sys
_module = sys.modules[__name__]
del sys
config = _module
ctpn_model = _module
ctpn_predict = _module
ctpn_train = _module
ctpn_utils = _module
data = _module
dataset = _module

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


import torch.nn.functional as F


import numpy as np


class RPN_REGR_Loss(nn.Module):

    def __init__(self, device, sigma=9.0):
        super(RPN_REGR_Loss, self).__init__()
        self.sigma = sigma
        self.device = device

    def forward(self, input, target):
        """
        smooth L1 loss
        :param input:y_preds
        :param target: y_true
        :return:
        """
        try:
            cls = target[(0), :, (0)]
            regr = target[(0), :, 1:3]
            regr_keep = (cls == 1).nonzero()[:, (0)]
            regr_true = regr[regr_keep]
            regr_pred = input[0][regr_keep]
            diff = torch.abs(regr_true - regr_pred)
            less_one = (diff < 1.0 / self.sigma).float()
            loss = less_one * 0.5 * diff ** 2 * self.sigma + torch.abs(1 -
                less_one) * (diff - 0.5 / self.sigma)
            loss = torch.sum(loss, 1)
            loss = torch.mean(loss) if loss.numel() > 0 else torch.tensor(0.0)
        except Exception as e:
            None
            loss = torch.tensor(0.0)
        return loss


class RPN_CLS_Loss(nn.Module):

    def __init__(self, device):
        super(RPN_CLS_Loss, self).__init__()
        self.device = device

    def forward(self, input, target):
        y_true = target[0][0]
        cls_keep = (y_true != -1).nonzero()[:, (0)]
        cls_true = y_true[cls_keep].long()
        cls_pred = input[0][cls_keep]
        loss = F.nll_loss(F.log_softmax(cls_pred, dim=-1), cls_true)
        loss = torch.clamp(torch.mean(loss), 0, 10) if loss.numel(
            ) > 0 else torch.tensor(0.0)
        return loss


class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, relu=True, bn=True, bias=True):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-05, momentum=0.01,
            affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class CTPN_Model(nn.Module):

    def __init__(self):
        super().__init__()
        base_model = models.vgg16(pretrained=False)
        layers = list(base_model.features)[:-1]
        self.base_layers = nn.Sequential(*layers)
        self.rpn = BasicConv(512, 512, 3, 1, 1, bn=False)
        self.brnn = nn.GRU(512, 128, bidirectional=True, batch_first=True)
        self.lstm_fc = BasicConv(256, 512, 1, 1, relu=True, bn=False)
        self.rpn_class = BasicConv(512, 10 * 2, 1, 1, relu=False, bn=False)
        self.rpn_regress = BasicConv(512, 10 * 2, 1, 1, relu=False, bn=False)

    def forward(self, x):
        x = self.base_layers(x)
        x = self.rpn(x)
        x1 = x.permute(0, 2, 3, 1).contiguous()
        b = x1.size()
        x1 = x1.view(b[0] * b[1], b[2], b[3])
        x2, _ = self.brnn(x1)
        xsz = x.size()
        x3 = x2.view(xsz[0], xsz[2], xsz[3], 256)
        x3 = x3.permute(0, 3, 1, 2).contiguous()
        x3 = self.lstm_fc(x3)
        x = x3
        cls = self.rpn_class(x)
        regr = self.rpn_regress(x)
        cls = cls.permute(0, 2, 3, 1).contiguous()
        regr = regr.permute(0, 2, 3, 1).contiguous()
        cls = cls.view(cls.size(0), cls.size(1) * cls.size(2) * 10, 2)
        regr = regr.view(regr.size(0), regr.size(1) * regr.size(2) * 10, 2)
        return cls, regr


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_opconty_pytorch_ctpn(_paritybench_base):
    pass
    def test_000(self):
        self._check(BasicConv(*[], **{'in_planes': 4, 'out_planes': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(RPN_CLS_Loss(*[], **{'device': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_002(self):
        self._check(RPN_REGR_Loss(*[], **{'device': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

