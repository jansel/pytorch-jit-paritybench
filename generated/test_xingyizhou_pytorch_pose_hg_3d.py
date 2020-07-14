import sys
_module = sys.modules[__name__]
del sys
_init_paths = _module
demo = _module
datasets = _module
coco = _module
fusion_3d = _module
h36m = _module
h36m_iccv = _module
mpii = _module
logger = _module
model = _module
models = _module
losses = _module
msra_resnet = _module
opts = _module
train = _module
train_3d = _module
utils = _module
debugger = _module
eval = _module
image = _module
main = _module
eval_COCO = _module
eval_PCKh = _module

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


import torch


import torch.utils.data


import torch.utils.data as data


import scipy.io as sio


import time


import torchvision.models as models


import torch.nn as nn


from torch.autograd import Function


import torch.utils.model_zoo as model_zoo


def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _tranpose_and_gather_scalar(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, 1)
    feat = _gather_feat(feat, ind)
    return feat


def reg_loss(regr, gt_regr, mask):
    num = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr)
    regr = regr * mask.float()
    gt_regr = gt_regr * mask.float()
    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
    regr_loss = regr_loss / (num + 0.0001)
    return regr_loss


class RegLoss(nn.Module):

    def __init__(self):
        super(RegLoss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _tranpose_and_gather_scalar(output, ind)
        loss = reg_loss(pred, target, mask)
        return loss


class VarLoss(Function):

    def __init__(self, device, var_weight):
        super(VarLoss, self).__init__()
        self.device = device
        self.var_weight = var_weight
        self.skeleton_idx = [[[0, 1], [1, 2], [3, 4], [4, 5]], [[10, 11], [11, 12], [13, 14], [14, 15]], [[2, 6], [3, 6]], [[12, 8], [13, 8]]]
        self.skeleton_weight = [[1.0085885098415446, 1, 1, 1.0085885098415446], [1.1375361376887123, 1, 1, 1.1375361376887123], [1, 1], [1, 1]]

    def forward(self, input, visible, mask, gt_2d):
        xy = gt_2d.view(gt_2d.size(0), -1, 2)
        batch_size = input.size(0)
        output = torch.FloatTensor(1) * 0
        for t in range(batch_size):
            if mask[t].sum() == 0:
                for g in range(len(self.skeleton_idx)):
                    E, num = 0, 0
                    N = len(self.skeleton_idx[g])
                    l = np.zeros(N)
                    for j in range(N):
                        id1, id2 = self.skeleton_idx[g][j]
                        if visible[t, id1] > 0.5 and visible[t, id2] > 0.5:
                            l[j] = (((xy[t, id1] - xy[t, id2]) ** 2).sum() + (input[t, id1] - input[t, id2]) ** 2) ** 0.5
                            l[j] = l[j] * self.skeleton_weight[g][j]
                            num += 1
                            E += l[j]
                    if num < 0.5:
                        E = 0
                    else:
                        E = E / num
                    loss = 0
                    for j in range(N):
                        if l[j] > 0:
                            loss += (l[j] - E) ** 2 / 2.0 / num
                    output += loss
        output = self.var_weight * output / batch_size
        self.save_for_backward(input, visible, mask, gt_2d)
        output = output
        return output

    def backward(self, grad_output):
        input, visible, mask, gt_2d = self.saved_tensors
        xy = gt_2d.view(gt_2d.size(0), -1, 2)
        grad_input = torch.zeros(input.size())
        batch_size = input.size(0)
        for t in range(batch_size):
            if mask[t].sum() == 0:
                for g in range(len(self.skeleton_idx)):
                    E, num = 0, 0
                    N = len(self.skeleton_idx[g])
                    l = np.zeros(N)
                    for j in range(N):
                        id1, id2 = self.skeleton_idx[g][j]
                        if visible[t, id1] > 0.5 and visible[t, id2] > 0.5:
                            l[j] = (((xy[t, id1] - xy[t, id2]) ** 2).sum() + (input[t, id1] - input[t, id2]) ** 2) ** 0.5
                            l[j] = l[j] * self.skeleton_weight[g][j]
                            num += 1
                            E += l[j]
                    if num < 0.5:
                        E = 0
                    else:
                        E = E / num
                    for j in range(N):
                        if l[j] > 0:
                            id1, id2 = self.skeleton_idx[g][j]
                            grad_input[t][id1] += self.var_weight * self.skeleton_weight[g][j] ** 2 / num * (l[j] - E) / l[j] * (input[t, id1] - input[t, id2]) / batch_size
                            grad_input[t][id2] += self.var_weight * self.skeleton_weight[g][j] ** 2 / num * (l[j] - E) / l[j] * (input[t, id2] - input[t, id1]) / batch_size
        grad_input = grad_input
        return grad_input, None, None, None


class FusionLoss(nn.Module):

    def __init__(self, device, reg_weight, var_weight):
        super(FusionLoss, self).__init__()
        self.reg_weight = reg_weight
        self.var_weight = var_weight
        self.device = device

    def forward(self, output, mask, ind, target, gt_2d):
        pred = _tranpose_and_gather_scalar(output, ind)
        loss = torch.FloatTensor(1)[0] * 0
        if self.reg_weight > 0:
            loss += self.reg_weight * reg_loss(pred, target, mask)
        if self.var_weight > 0:
            loss += VarLoss(self.device, self.var_weight)(pred, target, mask, gt_2d)[0]
        return loss


BN_MOMENTUM = 0.1


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
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
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
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


model_urls = {'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth', 'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth', 'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth', 'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth', 'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'}


class PoseResNet(nn.Module):

    def __init__(self, block, layers, heads, **kwargs):
        self.inplanes = 64
        self.deconv_with_bias = False
        self.heads = heads
        super(PoseResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.deconv_layers = self._make_deconv_layer(3, [256, 256, 256], [4, 4, 4])
        for head in sorted(self.heads):
            num_output = self.heads[head]
            self.__setattr__(head, nn.Conv2d(in_channels=256, out_channels=num_output, kernel_size=1, stride=1, padding=0))

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), 'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), 'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = self._get_deconv_cfg(num_kernels[i], i)
            planes = num_filters[i]
            layers.append(nn.ConvTranspose2d(in_channels=self.inplanes, out_channels=planes, kernel_size=kernel, stride=2, padding=padding, output_padding=output_padding, bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes
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
        x = self.deconv_layers(x)
        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(x)
        return [ret]

    def init_weights(self, num_layers, pretrained=True):
        if pretrained:
            for _, m in self.deconv_layers.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            for head in self.heads:
                final_layer = self.__getattr__(head)
                for m in final_layer.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.normal_(m.weight, std=0.001)
                        nn.init.constant_(m.bias, 0)
            url = model_urls['resnet{}'.format(num_layers)]
            pretrained_state_dict = model_zoo.load_url(url)
            None
            self.load_state_dict(pretrained_state_dict, strict=False)
        else:
            None
            None
            raise ValueError('imagenet pretrained model does not exist')


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (RegLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4]), torch.ones([4, 4], dtype=torch.int64), torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_xingyizhou_pytorch_pose_hg_3d(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

