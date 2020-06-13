import sys
_module = sys.modules[__name__]
del sys
wireframe = _module
york = _module
demo = _module
lcnn = _module
box = _module
config = _module
datasets = _module
metric = _module
models = _module
hourglass_pose = _module
line_vectorizer = _module
multitask_learner = _module
postprocess = _module
trainer = _module
utils = _module
lsd = _module
post = _module
process = _module
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


import torch


import torch.nn as nn


import torch.nn.functional as F


import itertools


import random


from collections import defaultdict


import numpy as np


from collections import OrderedDict


class Bottleneck2D(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck2D, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1)
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


class Hourglass(nn.Module):

    def __init__(self, block, num_blocks, planes, depth):
        super(Hourglass, self).__init__()
        self.depth = depth
        self.block = block
        self.hg = self._make_hour_glass(block, num_blocks, planes, depth)

    def _make_residual(self, block, num_blocks, planes):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(planes * block.expansion, planes))
        return nn.Sequential(*layers)

    def _make_hour_glass(self, block, num_blocks, planes, depth):
        hg = []
        for i in range(depth):
            res = []
            for j in range(3):
                res.append(self._make_residual(block, num_blocks, planes))
            if i == 0:
                res.append(self._make_residual(block, num_blocks, planes))
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _hour_glass_forward(self, n, x):
        up1 = self.hg[n - 1][0](x)
        low1 = F.max_pool2d(x, 2, stride=2)
        low1 = self.hg[n - 1][1](low1)
        if n > 1:
            low2 = self._hour_glass_forward(n - 1, low1)
        else:
            low2 = self.hg[n - 1][3](low1)
        low3 = self.hg[n - 1][2](low2)
        up2 = F.interpolate(low3, scale_factor=2)
        out = up1 + up2
        return out

    def forward(self, x):
        return self._hour_glass_forward(self.depth, x)


class HourglassNet(nn.Module):
    """Hourglass model from Newell et al ECCV 2016"""

    def __init__(self, block, head, depth, num_stacks, num_blocks, num_classes
        ):
        super(HourglassNet, self).__init__()
        self.inplanes = 64
        self.num_feats = 128
        self.num_stacks = num_stacks
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2,
            padding=3)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_residual(block, self.inplanes, 1)
        self.layer2 = self._make_residual(block, self.inplanes, 1)
        self.layer3 = self._make_residual(block, self.num_feats, 1)
        self.maxpool = nn.MaxPool2d(2, stride=2)
        ch = self.num_feats * block.expansion
        hg, res, fc, score, fc_, score_ = [], [], [], [], [], []
        for i in range(num_stacks):
            hg.append(Hourglass(block, num_blocks, self.num_feats, depth))
            res.append(self._make_residual(block, self.num_feats, num_blocks))
            fc.append(self._make_fc(ch, ch))
            score.append(head(ch, num_classes))
            if i < num_stacks - 1:
                fc_.append(nn.Conv2d(ch, ch, kernel_size=1))
                score_.append(nn.Conv2d(num_classes, ch, kernel_size=1))
        self.hg = nn.ModuleList(hg)
        self.res = nn.ModuleList(res)
        self.fc = nn.ModuleList(fc)
        self.score = nn.ModuleList(score)
        self.fc_ = nn.ModuleList(fc_)
        self.score_ = nn.ModuleList(score_)

    def _make_residual(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _make_fc(self, inplanes, outplanes):
        bn = nn.BatchNorm2d(inplanes)
        conv = nn.Conv2d(inplanes, outplanes, kernel_size=1)
        return nn.Sequential(conv, bn, self.relu)

    def forward(self, x):
        out = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.layer3(x)
        for i in range(self.num_stacks):
            y = self.hg[i](x)
            y = self.res[i](y)
            y = self.fc[i](y)
            score = self.score[i](y)
            out.append(score)
            if i < self.num_stacks - 1:
                fc_ = self.fc_[i](y)
                score_ = self.score_[i](score)
                x = x + fc_ + score_
        return out[::-1], y


def _to_json(obj, filename=None, encoding='utf-8', errors='strict', **
    json_kwargs):
    json_dump = json.dumps(obj, ensure_ascii=False, **json_kwargs)
    if filename:
        with open(filename, 'w', encoding=encoding, errors=errors) as f:
            f.write(json_dump if sys.version_info >= (3, 0) else json_dump.
                decode('utf-8'))
    else:
        return json_dump


class BoxError(Exception):
    """Non standard dictionary exceptions"""


class BoxKeyError(BoxError, KeyError, AttributeError):
    """Key does not exist"""


class Bottleneck1D(nn.Module):

    def __init__(self, inplanes, outplanes):
        super(Bottleneck1D, self).__init__()
        planes = outplanes // 2
        self.op = nn.Sequential(nn.BatchNorm1d(inplanes), nn.ReLU(inplace=
            True), nn.Conv1d(inplanes, planes, kernel_size=1), nn.
            BatchNorm1d(planes), nn.ReLU(inplace=True), nn.Conv1d(planes,
            planes, kernel_size=3, padding=1), nn.BatchNorm1d(planes), nn.
            ReLU(inplace=True), nn.Conv1d(planes, outplanes, kernel_size=1))

    def forward(self, x):
        return x + self.op(x)


class MultitaskHead(nn.Module):

    def __init__(self, input_channels, num_class):
        super(MultitaskHead, self).__init__()
        m = int(input_channels / 4)
        heads = []
        for output_channels in sum(M.head_size, []):
            heads.append(nn.Sequential(nn.Conv2d(input_channels, m,
                kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d
                (m, output_channels, kernel_size=1)))
        self.heads = nn.ModuleList(heads)
        assert num_class == sum(sum(M.head_size, []))

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=1)


def sigmoid_l1_loss(logits, target, offset=0.0, mask=None):
    logp = torch.sigmoid(logits) + offset
    loss = torch.abs(logp - target)
    if mask is not None:
        w = mask.mean(2, True).mean(1, True)
        w[w == 0] = 1
        loss = loss * (mask / w)
    return loss.mean(2).mean(1)


def cross_entropy_loss(logits, positive):
    nlogp = -F.log_softmax(logits, dim=0)
    return (positive * nlogp[1] + (1 - positive) * nlogp[0]).mean(2).mean(1)


class MultitaskLearner(nn.Module):

    def __init__(self, backbone):
        super(MultitaskLearner, self).__init__()
        self.backbone = backbone
        head_size = M.head_size
        self.num_class = sum(sum(head_size, []))
        self.head_off = np.cumsum([sum(h) for h in head_size])

    def forward(self, input_dict):
        image = input_dict['image']
        outputs, feature = self.backbone(image)
        result = {'feature': feature}
        batch, channel, row, col = outputs[0].shape
        T = input_dict['target'].copy()
        n_jtyp = T['jmap'].shape[1]
        for task in ['jmap']:
            T[task] = T[task].permute(1, 0, 2, 3)
        for task in ['joff']:
            T[task] = T[task].permute(1, 2, 0, 3, 4)
        offset = self.head_off
        loss_weight = M.loss_weight
        losses = []
        for stack, output in enumerate(outputs):
            output = output.transpose(0, 1).reshape([-1, batch, row, col]
                ).contiguous()
            jmap = output[0:offset[0]].reshape(n_jtyp, 2, batch, row, col)
            lmap = output[offset[0]:offset[1]].squeeze(0)
            joff = output[offset[1]:offset[2]].reshape(n_jtyp, 2, batch,
                row, col)
            if stack == 0:
                result['preds'] = {'jmap': jmap.permute(2, 0, 1, 3, 4).
                    softmax(2)[:, :, (1)], 'lmap': lmap.sigmoid(), 'joff': 
                    joff.permute(2, 0, 1, 3, 4).sigmoid() - 0.5}
                if input_dict['mode'] == 'testing':
                    return result
            L = OrderedDict()
            L['jmap'] = sum(cross_entropy_loss(jmap[i], T['jmap'][i]) for i in
                range(n_jtyp))
            L['lmap'] = F.binary_cross_entropy_with_logits(lmap, T['lmap'],
                reduction='none').mean(2).mean(1)
            L['joff'] = sum(sigmoid_l1_loss(joff[i, j], T['joff'][i, j], -
                0.5, T['jmap'][i]) for i in range(n_jtyp) for j in range(2))
            for loss_name in L:
                L[loss_name].mul_(loss_weight[loss_name])
            losses.append(L)
        result['losses'] = losses
        return result


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_zhou13_lcnn(_paritybench_base):
    pass
    def test_000(self):
        self._check(Bottleneck1D(*[], **{'inplanes': 4, 'outplanes': 4}), [torch.rand([4, 4, 4])], {})

