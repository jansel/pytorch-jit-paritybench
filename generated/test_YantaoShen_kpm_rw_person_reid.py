import sys
_module = sys.modules[__name__]
del sys
main = _module
reid = _module
datasets = _module
cuhk01 = _module
cuhk03 = _module
dukemtmc = _module
market1501 = _module
viper = _module
dist_metric = _module
evaluation_metrics = _module
classification = _module
ranking = _module
evaluators = _module
feature_extraction = _module
cnn = _module
database = _module
loss = _module
oim = _module
triplet = _module
metric_learning = _module
euclidean = _module
kissme = _module
models = _module
embedding = _module
inception = _module
kron = _module
multi_branch = _module
resnet = _module
trainers = _module
utils = _module
data = _module
dataset = _module
preprocessor = _module
sampler = _module
transforms = _module
logging = _module
meters = _module
osutils = _module
serialization = _module
setup = _module
test_cuhk01 = _module
test_cuhk03 = _module
test_dukemtmc = _module
test_market1501 = _module
test_viper = _module
test_cmc = _module
test_database = _module
test_oim = _module
test_inception = _module
test_preprocessor = _module

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


import torch.nn.functional as F


from torch import nn


from torch.backends import cudnn


from torch.utils.data import DataLoader


from torch.autograd import Variable


import time


from collections import OrderedDict


import torch.backends.cudnn as cudnn


from torch.utils.data import Dataset


from torch import autograd


import math


import copy


from torch.nn import functional as F


from torch.nn import init


import torchvision


from collections import defaultdict


from torch.utils.data.sampler import Sampler


from torch.utils.data.sampler import SequentialSampler


from torch.utils.data.sampler import RandomSampler


from torch.utils.data.sampler import SubsetRandomSampler


from torch.utils.data.sampler import WeightedRandomSampler


from torch.nn import Parameter


class OIM(autograd.Function):

    def __init__(self, lut, momentum=0.5):
        super(OIM, self).__init__()
        self.lut = lut
        self.momentum = momentum

    def forward(self, inputs, targets):
        self.save_for_backward(inputs, targets)
        outputs = inputs.mm(self.lut.t())
        return outputs

    def backward(self, grad_outputs):
        inputs, targets = self.saved_tensors
        grad_inputs = None
        if self.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(self.lut)
        for x, y in zip(inputs, targets):
            self.lut[y] = self.momentum * self.lut[y] + (1.0 - self.momentum) * x
            self.lut[y] /= self.lut[y].norm()
        return grad_inputs, None


def oim(inputs, targets, lut, momentum=0.5):
    return OIM(lut, momentum=momentum)(inputs, targets)


class OIMLoss(nn.Module):

    def __init__(self, num_features, num_classes, scalar=1.0, momentum=0.5, weight=None, size_average=True):
        super(OIMLoss, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.momentum = momentum
        self.scalar = scalar
        self.weight = weight
        self.size_average = size_average
        self.register_buffer('lut', torch.zeros(num_classes, num_features))

    def forward(self, inputs, targets):
        inputs = oim(inputs, targets, self.lut, momentum=self.momentum)
        inputs *= self.scalar
        loss = F.cross_entropy(inputs, targets, weight=self.weight, size_average=self.size_average)
        return loss, inputs


class TripletLoss(nn.Module):

    def __init__(self, margin=0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        n = inputs.size(0)
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max())
            dist_an.append(dist[i][mask[i] == 0].min())
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        prec = (dist_an.data > dist_ap.data).sum() * 1.0 / y.size(0)
        return loss, prec


def kron_matching(*inputs):
    assert len(inputs) == 2
    assert inputs[0].dim() == 4 and inputs[1].dim() == 4
    assert inputs[0].size() == inputs[1].size()
    N, C, H, W = inputs[0].size()
    w = inputs[0].permute(0, 2, 3, 1).contiguous().view(-1, C, 1, 1)
    x = inputs[1].view(1, N * C, H, W)
    x = F.conv2d(x, w, groups=N)
    x = x.view(N, H, W, H, W)
    return x


class KronMatching(nn.Module):

    def __init__(self):
        super(KronMatching, self).__init__()

    def forward(self, *inputs):
        return kron_matching(*inputs)


class RandomWalkEmbed(nn.Module):

    def __init__(self, instances_num=4, feat_num=2048, num_classes=0, drop_ratio=0.5):
        super(RandomWalkEmbed, self).__init__()
        self.instances_num = instances_num
        self.feat_num = feat_num
        self.temp = 1
        self.kron = KronMatching()
        self.bn = nn.BatchNorm1d(feat_num)
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()
        self.classifier = nn.Linear(feat_num, num_classes)
        self.classifier.weight.data.normal_(0, 0.001)
        self.classifier.bias.data.zero_()
        self.drop = nn.Dropout(drop_ratio)

    def _kron_matching(self, x1, x2):
        n, c, h, w = x1.size()
        x2_kro = self.kron(x1 / x1.norm(2, 1, keepdim=True).expand_as(x1), x2 / x2.norm(2, 1, keepdim=True).expand_as(x2))
        x2_kro_att = F.softmax((self.temp * x2_kro).view(n * h * w, h * w), dim=1).view(n, h, w, h, w)
        warped_x2 = torch.bmm(x2.view(n, c, h * w), x2_kro_att.view(n, h * w, h * w).transpose(1, 2)).view(n, c, h, w)
        return warped_x2

    def forward(self, probe_x, gallery_x, p2g=True, g2g=False):
        if not self.training and len(probe_x.size()) != len(gallery_x.size()):
            probe_x = probe_x.unsqueeze(0)
        probe_x.contiguous()
        gallery_x.contiguous()
        if p2g is True:
            N_probe, C, H, W = probe_x.size()
            N_gallery = gallery_x.size(0)
            probe_x = probe_x.unsqueeze(1)
            probe_x = probe_x.expand(N_probe, N_gallery, C, H, W)
            probe_x = probe_x.contiguous()
            gallery_x = gallery_x.unsqueeze(0)
            gallery_x = gallery_x.expand(N_probe, N_gallery, C, H, W)
            gallery_x = gallery_x.contiguous()
            probe_x = probe_x.view(N_probe * N_gallery, C, H, W)
            gallery_x = gallery_x.view(N_probe * N_gallery, C, H, W)
            probe_x = self._kron_matching(gallery_x, probe_x)
            diff = F.avg_pool2d(probe_x - gallery_x, (probe_x - gallery_x).size()[2:])
        elif g2g is True:
            N_probe = probe_x.size(0)
            N_gallery = gallery_x.size(0)
            probe_x = F.avg_pool2d(probe_x, probe_x.size()[2:]).view(N_probe, self.feat_num)
            gallery_x = F.avg_pool2d(gallery_x, gallery_x.size()[2:]).view(N_gallery, self.feat_num)
            probe_x = probe_x.unsqueeze(1)
            probe_x = probe_x.expand(N_probe, N_gallery, self.feat_num)
            probe_x = probe_x.contiguous()
            gallery_x = gallery_x.unsqueeze(0)
            gallery_x = gallery_x.expand(N_probe, N_gallery, self.feat_num)
            gallery_x = gallery_x.contiguous()
            diff = gallery_x - probe_x
        diff = torch.pow(diff, 2)
        diff = diff.view(N_probe * N_gallery, -1)
        diff = diff.contiguous()
        bn_diff = self.bn(diff)
        bn_diff = self.drop(bn_diff)
        cls_encode = self.classifier(bn_diff)
        cls_encode = cls_encode.view(N_probe, N_gallery, -1)
        return cls_encode


class EltwiseSubEmbed(nn.Module):

    def __init__(self, nonlinearity='square', use_batch_norm=False, use_classifier=False, num_features=0, num_classes=0):
        super(EltwiseSubEmbed, self).__init__()
        self.nonlinearity = nonlinearity
        if nonlinearity is not None and nonlinearity not in ['square', 'abs']:
            raise KeyError('Unknown nonlinearity:', nonlinearity)
        self.use_batch_norm = use_batch_norm
        self.use_classifier = use_classifier
        if self.use_batch_norm:
            self.bn = nn.BatchNorm1d(num_features)
            self.bn.weight.data.fill_(1)
            self.bn.bias.data.zero_()
        if self.use_classifier:
            assert num_features > 0 and num_classes > 0
            self.classifier = nn.Linear(num_features, num_classes)
            self.classifier.weight.data.normal_(0, 0.001)
            self.classifier.bias.data.zero_()

    def forward(self, x1, x2):
        x = x1 - x2
        if self.nonlinearity == 'square':
            x = x.pow(2)
        elif self.nonlinearity == 'abs':
            x = x.abs()
        if self.use_batch_norm:
            x = self.bn(x)
        if self.use_classifier:
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
        else:
            x = x.sum(1)
        return x


def _make_conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False):
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
    bn = nn.BatchNorm2d(out_planes)
    relu = nn.ReLU(inplace=True)
    return nn.Sequential(conv, bn, relu)


class Block(nn.Module):

    def __init__(self, in_planes, out_planes, pool_method, stride):
        super(Block, self).__init__()
        self.branches = nn.ModuleList([nn.Sequential(_make_conv(in_planes, out_planes, kernel_size=1, padding=0), _make_conv(out_planes, out_planes, stride=stride)), nn.Sequential(_make_conv(in_planes, out_planes, kernel_size=1, padding=0), _make_conv(out_planes, out_planes), _make_conv(out_planes, out_planes, stride=stride))])
        if pool_method == 'Avg':
            assert stride == 1
            self.branches.append(_make_conv(in_planes, out_planes, kernel_size=1, padding=0))
            self.branches.append(nn.Sequential(nn.AvgPool2d(kernel_size=3, stride=1, padding=1), _make_conv(in_planes, out_planes, kernel_size=1, padding=0)))
        else:
            self.branches.append(nn.MaxPool2d(kernel_size=3, stride=stride, padding=1))

    def forward(self, x):
        return torch.cat([b(x) for b in self.branches], 1)


class InceptionNet(nn.Module):

    def __init__(self, cut_at_pooling=False, num_features=256, norm=False, dropout=0, num_classes=0):
        super(InceptionNet, self).__init__()
        self.cut_at_pooling = cut_at_pooling
        self.conv1 = _make_conv(3, 32)
        self.conv2 = _make_conv(32, 32)
        self.conv3 = _make_conv(32, 32)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.in_planes = 32
        self.inception4a = self._make_inception(64, 'Avg', 1)
        self.inception4b = self._make_inception(64, 'Max', 2)
        self.inception5a = self._make_inception(128, 'Avg', 1)
        self.inception5b = self._make_inception(128, 'Max', 2)
        self.inception6a = self._make_inception(256, 'Avg', 1)
        self.inception6b = self._make_inception(256, 'Max', 2)
        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            if self.has_embedding:
                self.feat = nn.Linear(self.in_planes, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
            else:
                self.num_features = self.in_planes
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.classifier = nn.Linear(self.num_features, self.num_classes)
        self.reset_params()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.inception6a(x)
        x = self.inception6b(x)
        if self.cut_at_pooling:
            return x
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if self.has_embedding:
            x = self.feat(x)
            x = self.feat_bn(x)
        if self.norm:
            x = F.normalize(x)
        elif self.has_embedding:
            x = F.relu(x)
        if self.dropout > 0:
            x = self.drop(x)
        if self.num_classes > 0:
            x = self.classifier(x)
        return x

    def _make_inception(self, out_planes, pool_method, stride):
        block = Block(self.in_planes, out_planes, pool_method, stride)
        self.in_planes = out_planes * 4 if pool_method == 'Avg' else out_planes * 2 + self.in_planes
        return block

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)


def random_walk_compute(p_g_score, g_g_score, alpha):
    one_diag = Variable(torch.eye(g_g_score.size(0)), requires_grad=False)
    g_g_score_sm = Variable(g_g_score.data.clone(), requires_grad=False)
    inf_diag = torch.diag(torch.Tensor([-float('Inf')]).expand(g_g_score.size(0))) + g_g_score_sm[:, :, 1].squeeze().data
    A = F.softmax(Variable(inf_diag))
    A = (1 - alpha) * torch.inverse(one_diag - alpha * A)
    A = A.transpose(0, 1)
    p_g_score = torch.matmul(p_g_score.permute(2, 0, 1), A).permute(1, 2, 0).contiguous()
    g_g_score = torch.matmul(g_g_score.permute(2, 0, 1), A).permute(1, 2, 0).contiguous()
    p_g_score = p_g_score.view(-1, 2)
    g_g_score = g_g_score.view(-1, 2)
    outputs = torch.cat((p_g_score, g_g_score), 0)
    outputs = outputs.contiguous()
    return outputs


class RandomWalkKpmNet(nn.Module):

    def __init__(self, instances_num=4, base_model=None, embed_model=None, alpha=0.1):
        super(RandomWalkKpmNet, self).__init__()
        self.instances_num = instances_num
        self.alpha = alpha
        self.base = base_model
        self.embed = embed_model
        for i in range(len(embed_model)):
            setattr(self, 'embed_' + str(i), embed_model[i])

    def forward(self, x):
        x = self.base(x)
        N, C, H, W = x.size()
        probe_num = int(N / self.instances_num)
        gallery_num = int(N - N / self.instances_num)
        x = x.view(probe_num, self.instances_num, C, H, W)
        probe_x = x[:, 0, :, :, :]
        probe_x = probe_x.contiguous()
        probe_x = probe_x.view(probe_num, C, H, W)
        gallery_x = x[:, 1:self.instances_num, :, :, :]
        gallery_x = gallery_x.contiguous()
        gallery_x = gallery_x.view(gallery_num, C, H, W)
        count = 2048 / len(self.embed)
        outputs = []
        for j in range(len(self.embed)):
            for i in range(len(self.embed)):
                p_g_score = self.embed[j](probe_x[:, i * count:(i + 1) * count].contiguous(), gallery_x[:, i * count:(i + 1) * count].contiguous(), p2g=True, g2g=False)
                g_g_score = self.embed[j](gallery_x[:, i * count:(i + 1) * count].contiguous(), gallery_x[:, i * count:(i + 1) * count].contiguous(), p2g=False, g2g=True)
                outputs.append(random_walk_compute(p_g_score, g_g_score, self.alpha))
        outputs = torch.cat(outputs, 0)
        return outputs


class ResNet(nn.Module):
    __factory = {(18): torchvision.models.resnet18, (34): torchvision.models.resnet34, (50): torchvision.models.resnet50, (101): torchvision.models.resnet101, (152): torchvision.models.resnet152}

    def __init__(self, depth, pretrained=True, cut_at_pooling=False, cut_after_embed=False, num_features=0, norm=False, dropout=0, num_classes=0):
        super(ResNet, self).__init__()
        self.depth = depth
        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling
        self.cut_after_embed = cut_after_embed
        if depth not in ResNet.__factory:
            raise KeyError('Unsupported depth:', depth)
        self.base = ResNet.__factory[depth](pretrained=pretrained)
        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes
            out_planes = self.base.fc.in_features
            if self.has_embedding:
                self.feat = nn.Linear(out_planes, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal(self.feat.weight, mode='fan_out')
                init.constant(self.feat.bias, 0)
                init.constant(self.feat_bn.weight, 1)
                init.constant(self.feat_bn.bias, 0)
            else:
                self.num_features = out_planes
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.classifier = nn.Linear(self.num_features, self.num_classes)
                init.normal(self.classifier.weight, std=0.001)
                init.constant(self.classifier.bias, 0)
        if not self.pretrained:
            self.reset_params()

    def forward(self, x):
        for name, module in self.base._modules.items():
            if name == 'avgpool':
                break
            x = module(x)
        if self.cut_at_pooling:
            if self.training is True:
                return x
            else:
                return x, F.avg_pool2d(x, x.size()[2:]).view(x.size(0), -1)
        if self.has_embedding:
            x = self.feat(x)
            x = self.feat_bn(x)
        if self.norm:
            x = F.normalize(x)
        elif self.has_embedding:
            x = F.relu(x)
        if self.dropout > 0:
            x = self.drop(x)
        if self.cut_after_embed:
            return x
        if self.num_classes > 0:
            x = self.classifier(x)
        return x

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Block,
     lambda: ([], {'in_planes': 4, 'out_planes': 4, 'pool_method': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (EltwiseSubEmbed,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (InceptionNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
]

class Test_YantaoShen_kpm_rw_person_reid(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

