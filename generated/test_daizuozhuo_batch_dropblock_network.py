import sys
_module = sys.modules[__name__]
del sys
config = _module
datasets = _module
data_loader = _module
data_manager = _module
samplers = _module
main_reid = _module
models = _module
networks = _module
resnet = _module
trainers = _module
evaluator = _module
re_ranking = _module
trainer = _module
DistWeightDevianceLoss = _module
LiftedStructure = _module
utils = _module
loss = _module
meters = _module
random_erasing = _module
serialization = _module
transforms = _module
validation_metrics = _module

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


from torch.utils.data import Dataset


from collections import defaultdict


import numpy as np


import torch


import random


from torch.utils.data.sampler import Sampler


from torch import nn


from torch.backends import cudnn


from torch.utils.data import DataLoader


import copy


import itertools


import torch.nn.functional as F


import torch.utils.model_zoo as model_zoo


from scipy.spatial.distance import cdist


from sklearn.preprocessing import normalize


from torch import optim


from torch.utils.data import dataloader


from torchvision import transforms


from torchvision.models.resnet import Bottleneck


from torchvision.models.resnet import resnet50


from torchvision.transforms import functional


import math


import torch as th


import time


from torch.autograd import Variable


from torchvision.transforms import *


class SELayer(nn.Module):

    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channel, channel // reduction), nn.ReLU(inplace=True), nn.Linear(channel // reduction, channel), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class BatchDrop(nn.Module):

    def __init__(self, h_ratio, w_ratio):
        super(BatchDrop, self).__init__()
        self.h_ratio = h_ratio
        self.w_ratio = w_ratio

    def forward(self, x):
        if self.training:
            h, w = x.size()[-2:]
            rh = round(self.h_ratio * h)
            rw = round(self.w_ratio * w)
            sx = random.randint(0, h - rh)
            sy = random.randint(0, w - rw)
            mask = x.new_ones(x.size())
            mask[:, :, sx:sx + rh, sy:sy + rw] = 0
            x = x * mask
        return x


class BatchCrop(nn.Module):

    def __init__(self, ratio):
        super(BatchCrop, self).__init__()
        self.ratio = ratio

    def forward(self, x):
        if self.training:
            h, w = x.size()[-2:]
            rw = int(self.ratio * w)
            start = random.randint(0, h - 1)
            if start + rw > h:
                select = list(range(0, start + rw - h)) + list(range(start, h))
            else:
                select = list(range(start, start + rw))
            mask = x.new_zeros(x.size())
            mask[:, :, (select), :] = 1
            x = x * mask
        return x


class ResNet(nn.Module):

    def __init__(self, last_stride=2, block=Bottleneck, layers=[3, 4, 6, 3]):
        self.inplanes = 64
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=last_stride)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
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
        return x

    def load_param(self, param_dict):
        for i in param_dict:
            if 'fc' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])

    def random_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0.0)


class ResNetBuilder(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes=None, last_stride=1, pretrained=False):
        super().__init__()
        self.base = ResNet(last_stride)
        if pretrained:
            model_url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
            self.base.load_param(model_zoo.load_url(model_url))
        self.num_classes = num_classes
        if num_classes is not None:
            self.bottleneck = nn.Sequential(nn.Linear(self.in_planes, 512), nn.BatchNorm1d(512), nn.LeakyReLU(0.1), nn.Dropout(p=0.5))
            self.bottleneck.apply(weights_init_kaiming)
            self.classifier = nn.Linear(512, self.num_classes)
            self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        global_feat = self.base(x)
        global_feat = F.avg_pool2d(global_feat, global_feat.shape[2:])
        global_feat = global_feat.view(global_feat.shape[0], -1)
        if self.training and self.num_classes is not None:
            feat = self.bottleneck(global_feat)
            cls_score = self.classifier(feat)
            return [global_feat], [cls_score]
        else:
            return global_feat

    def get_optim_policy(self):
        base_param_group = self.base.parameters()
        if self.num_classes is not None:
            add_param_group = itertools.chain(self.bottleneck.parameters(), self.classifier.parameters())
            return [{'params': base_param_group}, {'params': add_param_group}]
        else:
            return [{'params': base_param_group}]


class BFE(nn.Module):

    def __init__(self, num_classes, width_ratio=0.5, height_ratio=0.5):
        super(BFE, self).__init__()
        resnet = resnet50(pretrained=True)
        self.backbone = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1, resnet.layer2, resnet.layer3)
        self.res_part = nn.Sequential(Bottleneck(1024, 512, stride=1, downsample=nn.Sequential(nn.Conv2d(1024, 2048, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(2048))), Bottleneck(2048, 512), Bottleneck(2048, 512))
        self.res_part.load_state_dict(resnet.layer4.state_dict())
        reduction = nn.Sequential(nn.Conv2d(2048, 512, 1), nn.BatchNorm2d(512), nn.ReLU())
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_softmax = nn.Linear(512, num_classes)
        self.global_softmax.apply(weights_init_kaiming)
        self.global_reduction = copy.deepcopy(reduction)
        self.global_reduction.apply(weights_init_kaiming)
        self.res_part2 = Bottleneck(2048, 512)
        self.part_maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.batch_crop = BatchDrop(height_ratio, width_ratio)
        self.reduction = nn.Sequential(nn.Linear(2048, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU())
        self.reduction.apply(weights_init_kaiming)
        self.softmax = nn.Linear(1024, num_classes)
        self.softmax.apply(weights_init_kaiming)

    def forward(self, x):
        """
        :param x: input image tensor of (N, C, H, W)
        :return: (prediction, triplet_losses, softmax_losses)
        """
        x = self.backbone(x)
        x = self.res_part(x)
        predict = []
        triplet_features = []
        softmax_features = []
        glob = self.global_avgpool(x)
        global_triplet_feature = self.global_reduction(glob).squeeze()
        global_softmax_class = self.global_softmax(global_triplet_feature)
        softmax_features.append(global_softmax_class)
        triplet_features.append(global_triplet_feature)
        predict.append(global_triplet_feature)
        x = self.res_part2(x)
        x = self.batch_crop(x)
        triplet_feature = self.part_maxpool(x).squeeze()
        feature = self.reduction(triplet_feature)
        softmax_feature = self.softmax(feature)
        triplet_features.append(feature)
        softmax_features.append(softmax_feature)
        predict.append(feature)
        if self.training:
            return triplet_features, softmax_features
        else:
            return torch.cat(predict, 1)

    def get_optim_policy(self):
        params = [{'params': self.backbone.parameters()}, {'params': self.res_part.parameters()}, {'params': self.global_reduction.parameters()}, {'params': self.global_softmax.parameters()}, {'params': self.res_part2.parameters()}, {'params': self.reduction.parameters()}, {'params': self.softmax.parameters()}]
        return params


class Resnet(nn.Module):

    def __init__(self, num_classes, resnet=None):
        super(Resnet, self).__init__()
        if not resnet:
            resnet = resnet50(pretrained=True)
        self.backbone = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.softmax = nn.Linear(2048, num_classes)

    def forward(self, x):
        """
        :param x: input image tensor of (N, C, H, W)
        :return: (prediction, triplet_losses, softmax_losses)
        """
        x = self.backbone(x)
        x = self.global_avgpool(x).squeeze()
        feature = self.softmax(x)
        if self.training:
            return [], [feature]
        else:
            return feature

    def get_optim_policy(self):
        return self.parameters()


class IDE(nn.Module):

    def __init__(self, num_classes, resnet=None):
        super(IDE, self).__init__()
        if not resnet:
            resnet = resnet50(pretrained=True)
        self.backbone = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        self.global_avgpool = nn.AvgPool2d(kernel_size=(12, 4))

    def forward(self, x):
        """
        :param x: input image tensor of (N, C, H, W)
        :return: (prediction, triplet_losses, softmax_losses)
        """
        x = self.backbone(x)
        feature = self.global_avgpool(x).squeeze()
        if self.training:
            return [feature], []
        else:
            return feature

    def get_optim_policy(self):
        return self.parameters()


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
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


class CBAM_Module(nn.Module):

    def __init__(self, channels, reduction):
        super(CBAM_Module, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid_channel = nn.Sigmoid()
        self.conv_after_concat = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        avg = self.avg_pool(x)
        mx = self.max_pool(x)
        avg = self.fc1(avg)
        mx = self.fc1(mx)
        avg = self.relu(avg)
        mx = self.relu(mx)
        avg = self.fc2(avg)
        mx = self.fc2(mx)
        x = avg + mx
        x = self.sigmoid_channel(x)
        x = module_input * x
        module_input = x
        avg = torch.mean(x, 1, True)
        mx, _ = torch.max(x, 1, True)
        x = torch.cat((avg, mx), 1)
        x = self.conv_after_concat(x)
        x = self.sigmoid_spatial(x)
        x = module_input * x
        return x


class CBAMBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(CBAMBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.cbam = CBAM_Module(planes * 4, reduction=16)
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
        out = self.cbam(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


def GaussDistribution(data):
    """
    :param data:
    :return:
    """
    mean_value = torch.mean(data)
    diff = data - mean_value
    std = torch.sqrt(torch.mean(torch.pow(diff, 2)))
    return mean_value, std


def similarity(inputs_):
    sim = torch.matmul(inputs_, inputs_.t())
    return sim


class DistWeightBinDevianceLoss(nn.Module):

    def __init__(self, margin=0.5):
        super(DistWeightBinDevianceLoss, self).__init__()
        self.margin = margin

    def forward(self, inputs, targets):
        n = inputs.size(0)
        sim_mat = similarity(inputs)
        targets = targets
        eyes_ = Variable(torch.eye(n, n))
        pos_mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        neg_mask = eyes_.eq(eyes_) - pos_mask
        pos_mask = pos_mask - eyes_.eq(1)
        pos_sim = torch.masked_select(sim_mat, pos_mask)
        neg_sim = torch.masked_select(sim_mat, neg_mask)
        num_instances = len(pos_sim) // n + 1
        num_neg_instances = n - num_instances
        pos_sim = pos_sim.resize(len(pos_sim) // (num_instances - 1), num_instances - 1)
        neg_sim = neg_sim.resize(len(neg_sim) // num_neg_instances, num_neg_instances)
        loss = list()
        c = 0
        for i, pos_pair in enumerate(pos_sim):
            pos_pair = torch.sort(pos_pair)[0]
            neg_pair = torch.sort(neg_sim[i])[0]
            neg_mean, neg_std = GaussDistribution(neg_pair)
            prob = torch.exp(torch.pow(neg_pair - neg_mean, 2) / (2 * torch.pow(neg_std, 2)))
            neg_index = torch.multinomial(prob, num_instances - 1, replacement=False)
            neg_pair = neg_pair[neg_index]
            if len(neg_pair) < 1:
                c += 1
                continue
            if pos_pair[-1].item() > neg_pair[-1].item() + 0.05:
                c += 1
            neg_pair = torch.sort(neg_pair)[0]
            if i == 1 and np.random.randint(256) == 1:
                None
                None
            pos_loss = torch.mean(torch.log(1 + torch.exp(-2 * (pos_pair - self.margin))))
            neg_loss = 0.04 * torch.mean(torch.log(1 + torch.exp(50 * (neg_pair - self.margin))))
            loss.append(pos_loss + neg_loss)
        loss = [torch.unsqueeze(l, 0) for l in loss]
        loss = torch.sum(torch.cat(loss)) / n
        prec = float(c) / n
        neg_d = torch.mean(neg_sim).item()
        pos_d = torch.mean(pos_sim).item()
        return loss, prec, pos_d, neg_d


def pdist(A, squared=False, eps=0.0001):
    prod = torch.mm(A, A.t())
    norm = prod.diag().unsqueeze(1).expand_as(prod)
    res = (norm + norm.t() - 2 * prod).clamp(min=0)
    return res if squared else res.clamp(min=eps).sqrt()


class LiftedStructureLoss(nn.Module):

    def __init__(self, alpha=10, beta=2, margin=0.5, hard_mining=None, **kwargs):
        super(LiftedStructureLoss, self).__init__()
        self.margin = margin
        self.alpha = alpha
        self.beta = beta
        self.hard_mining = hard_mining

    def forward(self, embeddings, labels):
        """
        score = embeddings
        target = labels
        loss = 0
        counter = 0
        bsz = score.size(0)
        mag = (score ** 2).sum(1).expand(bsz, bsz)
        sim = score.mm(score.transpose(0, 1))
        dist = (mag + mag.transpose(0, 1) - 2 * sim)
        dist = torch.nn.functional.relu(dist).sqrt()
        
        for i in range(bsz):
            t_i = target[i].item()
            for j in range(i + 1, bsz):
                t_j = target[j].item()
                if t_i == t_j:
                    # Negative component
                    # !! Could do other things (like softmax that weights closer negatives)
                    l_ni = (self.margin - dist[i][target != t_i]).exp().sum()
                    l_nj = (self.margin - dist[j][target != t_j]).exp().sum()
                    l_n  = (l_ni + l_nj).log()
                    # Positive component
                    l_p  = dist[i,j]
                    loss += torch.nn.functional.relu(l_n + l_p) ** 2
                    counter += 1
        return loss / (2 * counter), 0
        """
        margin = 1.0
        eps = 0.0001
        d = pdist(embeddings, squared=False, eps=eps)
        pos = torch.eq(*[labels.unsqueeze(dim).expand_as(d) for dim in [0, 1]]).type_as(d)
        neg_i = torch.mul((margin - d).exp(), 1 - pos).sum(1).expand_as(d)
        return torch.sum(F.relu(pos.triu(1) * ((neg_i + neg_i.t()).log() + d)).pow(2)) / (pos.sum() - len(d)), 0


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu:
            targets = targets
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BFE,
     lambda: ([], {'num_classes': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (BatchCrop,
     lambda: ([], {'ratio': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (BatchDrop,
     lambda: ([], {'h_ratio': 4, 'w_ratio': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (CBAM_Module,
     lambda: ([], {'channels': 4, 'reduction': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CrossEntropyLabelSmooth,
     lambda: ([], {'num_classes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.zeros([4, 4, 4], dtype=torch.int64)], {}),
     True),
    (IDE,
     lambda: ([], {'num_classes': 4}),
     lambda: ([torch.rand([4, 3, 512, 512])], {}),
     False),
    (ResNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (ResNetBuilder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (Resnet,
     lambda: ([], {'num_classes': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (SELayer,
     lambda: ([], {'channel': 16}),
     lambda: ([torch.rand([4, 16, 4, 16])], {}),
     True),
]

class Test_daizuozhuo_batch_dropblock_network(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

    def test_004(self):
        self._check(*TESTCASES[4])

    def test_005(self):
        self._check(*TESTCASES[5])

    def test_006(self):
        self._check(*TESTCASES[6])

    def test_007(self):
        self._check(*TESTCASES[7])

    def test_008(self):
        self._check(*TESTCASES[8])

    def test_009(self):
        self._check(*TESTCASES[9])

