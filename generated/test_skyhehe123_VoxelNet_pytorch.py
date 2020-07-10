import sys
_module = sys.modules[__name__]
del sys
config = _module
crop = _module
gen_train_val = _module
kitti = _module
data_aug = _module
loss = _module
nms = _module
build = _module
pth_nms = _module
setup = _module
test_utils = _module
train = _module
utils = _module
voxelnet = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import torch.utils.data as data


import numpy as np


import torch


import torch.nn as nn


import torch.nn.functional as F


import torch.backends.cudnn


from torch.autograd import Variable


import time


import torch.optim as optim


import torch.nn.init as init


class VoxelLoss(nn.Module):

    def __init__(self, alpha, beta):
        super(VoxelLoss, self).__init__()
        self.smoothl1loss = nn.SmoothL1Loss(size_average=False)
        self.alpha = alpha
        self.beta = beta

    def forward(self, rm, psm, pos_equal_one, neg_equal_one, targets):
        p_pos = F.sigmoid(psm.permute(0, 2, 3, 1))
        rm = rm.permute(0, 2, 3, 1).contiguous()
        rm = rm.view(rm.size(0), rm.size(1), rm.size(2), -1, 7)
        targets = targets.view(targets.size(0), targets.size(1), targets.size(2), -1, 7)
        pos_equal_one_for_reg = pos_equal_one.unsqueeze(pos_equal_one.dim()).expand(-1, -1, -1, -1, 7)
        rm_pos = rm * pos_equal_one_for_reg
        targets_pos = targets * pos_equal_one_for_reg
        cls_pos_loss = -pos_equal_one * torch.log(p_pos + 1e-06)
        cls_pos_loss = cls_pos_loss.sum() / (pos_equal_one.sum() + 1e-06)
        cls_neg_loss = -neg_equal_one * torch.log(1 - p_pos + 1e-06)
        cls_neg_loss = cls_neg_loss.sum() / (neg_equal_one.sum() + 1e-06)
        reg_loss = self.smoothl1loss(rm_pos, targets_pos)
        reg_loss = reg_loss / (pos_equal_one.sum() + 1e-06)
        conf_loss = self.alpha * cls_pos_loss + self.beta * cls_neg_loss
        return conf_loss, reg_loss


class Conv2d(nn.Module):

    def __init__(self, in_channels, out_channels, k, s, p, activation=True, batch_norm=True):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=s, padding=p)
        if batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.activation:
            return F.relu(x, inplace=True)
        else:
            return x


class Conv3d(nn.Module):

    def __init__(self, in_channels, out_channels, k, s, p, batch_norm=True):
        super(Conv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=k, stride=s, padding=p)
        if batch_norm:
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            self.bn = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        return F.relu(x, inplace=True)


class FCN(nn.Module):

    def __init__(self, cin, cout):
        super(FCN, self).__init__()
        self.cout = cout
        self.linear = nn.Linear(cin, cout)
        self.bn = nn.BatchNorm1d(cout)

    def forward(self, x):
        kk, t, _ = x.shape
        x = self.linear(x.view(kk * t, -1))
        x = F.relu(self.bn(x))
        return x.view(kk, t, -1)


class VFE(nn.Module):

    def __init__(self, cin, cout):
        super(VFE, self).__init__()
        assert cout % 2 == 0
        self.units = cout // 2
        self.fcn = FCN(cin, self.units)

    def forward(self, x, mask):
        pwf = self.fcn(x)
        laf = torch.max(pwf, 1)[0].unsqueeze(1).repeat(1, cfg.T, 1)
        pwcf = torch.cat((pwf, laf), dim=2)
        mask = mask.unsqueeze(2).repeat(1, 1, self.units * 2)
        pwcf = pwcf * mask.float()
        return pwcf


class SVFE(nn.Module):

    def __init__(self):
        super(SVFE, self).__init__()
        self.vfe_1 = VFE(7, 32)
        self.vfe_2 = VFE(32, 128)
        self.fcn = FCN(128, 128)

    def forward(self, x):
        mask = torch.ne(torch.max(x, 2)[0], 0)
        x = self.vfe_1(x, mask)
        x = self.vfe_2(x, mask)
        x = self.fcn(x)
        x = torch.max(x, 1)[0]
        return x


class CML(nn.Module):

    def __init__(self):
        super(CML, self).__init__()
        self.conv3d_1 = Conv3d(128, 64, 3, s=(2, 1, 1), p=(1, 1, 1))
        self.conv3d_2 = Conv3d(64, 64, 3, s=(1, 1, 1), p=(0, 1, 1))
        self.conv3d_3 = Conv3d(64, 64, 3, s=(2, 1, 1), p=(1, 1, 1))

    def forward(self, x):
        x = self.conv3d_1(x)
        x = self.conv3d_2(x)
        x = self.conv3d_3(x)
        return x


class RPN(nn.Module):

    def __init__(self):
        super(RPN, self).__init__()
        self.block_1 = [Conv2d(128, 128, 3, 2, 1)]
        self.block_1 += [Conv2d(128, 128, 3, 1, 1) for _ in range(3)]
        self.block_1 = nn.Sequential(*self.block_1)
        self.block_2 = [Conv2d(128, 128, 3, 2, 1)]
        self.block_2 += [Conv2d(128, 128, 3, 1, 1) for _ in range(5)]
        self.block_2 = nn.Sequential(*self.block_2)
        self.block_3 = [Conv2d(128, 256, 3, 2, 1)]
        self.block_3 += [nn.Conv2d(256, 256, 3, 1, 1) for _ in range(5)]
        self.block_3 = nn.Sequential(*self.block_3)
        self.deconv_1 = nn.Sequential(nn.ConvTranspose2d(256, 256, 4, 4, 0), nn.BatchNorm2d(256))
        self.deconv_2 = nn.Sequential(nn.ConvTranspose2d(128, 256, 2, 2, 0), nn.BatchNorm2d(256))
        self.deconv_3 = nn.Sequential(nn.ConvTranspose2d(128, 256, 1, 1, 0), nn.BatchNorm2d(256))
        self.score_head = Conv2d(768, cfg.anchors_per_position, 1, 1, 0, activation=False, batch_norm=False)
        self.reg_head = Conv2d(768, 7 * cfg.anchors_per_position, 1, 1, 0, activation=False, batch_norm=False)

    def forward(self, x):
        x = self.block_1(x)
        x_skip_1 = x
        x = self.block_2(x)
        x_skip_2 = x
        x = self.block_3(x)
        x_0 = self.deconv_1(x)
        x_1 = self.deconv_2(x_skip_2)
        x_2 = self.deconv_3(x_skip_1)
        x = torch.cat((x_0, x_1, x_2), 1)
        return self.score_head(x), self.reg_head(x)


class VoxelNet(nn.Module):

    def __init__(self):
        super(VoxelNet, self).__init__()
        self.svfe = SVFE()
        self.cml = CML()
        self.rpn = RPN()

    def voxel_indexing(self, sparse_features, coords):
        dim = sparse_features.shape[-1]
        dense_feature = Variable(torch.zeros(dim, cfg.N, cfg.D, cfg.H, cfg.W))
        dense_feature[:, (coords[:, (0)]), (coords[:, (1)]), (coords[:, (2)]), (coords[:, (3)])] = sparse_features
        return dense_feature.transpose(0, 1)

    def forward(self, voxel_features, voxel_coords):
        vwfs = self.svfe(voxel_features)
        vwfs = self.voxel_indexing(vwfs, voxel_coords)
        cml_out = self.cml(vwfs)
        psm, rm = self.rpn(cml_out.view(cfg.N, -1, cfg.H, cfg.W))
        return psm, rm


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Conv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'k': 4, 's': 4, 'p': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Conv3d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'k': 4, 's': 4, 'p': 4}),
     lambda: ([torch.rand([4, 4, 64, 64, 64])], {}),
     True),
    (FCN,
     lambda: ([], {'cin': 4, 'cout': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
]

class Test_skyhehe123_VoxelNet_pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

