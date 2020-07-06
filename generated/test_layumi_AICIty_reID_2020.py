import sys
_module = sys.modules[__name__]
del sys
AugFolder = _module
autoaugment = _module
convert = _module
evaluate_gpu = _module
evaluate_rerank = _module
fast_submit_baidu = _module
losses = _module
model = _module
prepare_2020 = _module
prepare_cam2020 = _module
random_erasing = _module
re_ranking = _module
submit_result_multimodel = _module
test_2020 = _module
train_2020 = _module
train_ft_2020 = _module
utils = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import scipy.io


import torch


import numpy as np


import time


import torch.nn as nn


import torch.optim as optim


from torch.optim import lr_scheduler


from torch.autograd import Variable


import torch.backends.cudnn as cudnn


import torchvision


from torchvision import datasets


from torchvision import models


from torchvision import transforms


import math


from sklearn.cluster import DBSCAN


import torch.nn.functional as F


from torch.nn import Parameter


from torch.nn import init


from torch.nn import functional as F


from torchvision.transforms import *


import random


def myphi(x, m):
    x = x * m
    return 1 - x ** 2 / math.factorial(2) + x ** 4 / math.factorial(4) - x ** 6 / math.factorial(6) + x ** 8 / math.factorial(8) - x ** 9 / math.factorial(9)


class AngleLinear(nn.Module):

    def __init__(self, in_features, out_features, m=4, phiflag=True):
        super(AngleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        init.normal_(self.weight.data, std=0.001)
        self.phiflag = phiflag
        self.m = m
        self.mlambda = [lambda x: x ** 0, lambda x: x ** 1, lambda x: 2 * x ** 2 - 1, lambda x: 4 * x ** 3 - 3 * x, lambda x: 8 * x ** 4 - 8 * x ** 2 + 1, lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x]

    def forward(self, input):
        x = input
        w = self.weight
        ww = w.renorm(2, 1, 1e-05).mul(100000.0)
        xlen = x.pow(2).sum(1).pow(0.5)
        wlen = ww.pow(2).sum(0).pow(0.5)
        cos_theta = x.mm(ww)
        cos_theta = cos_theta / xlen.view(-1, 1) / wlen.view(1, -1)
        cos_theta = cos_theta.clamp(-1, 1)
        if self.phiflag:
            cos_m_theta = self.mlambda[self.m](cos_theta)
            theta = Variable(cos_theta.data.acos())
            k = (self.m * theta / 3.14159265).floor()
            n_one = k * 0.0 - 1
            phi_theta = n_one ** k * cos_m_theta - 2 * k
        else:
            theta = cos_theta.acos()
            phi_theta = myphi(theta, self.m)
            phi_theta = phi_theta.clamp(-1 * self.m, 1)
        cos_theta = cos_theta * xlen.view(-1, 1)
        phi_theta = phi_theta * xlen.view(-1, 1)
        output = cos_theta, phi_theta
        return output


def L2Normalization(ff, dim=1):
    fnorm = torch.norm(ff, p=2, dim=dim, keepdim=True) + 1e-05
    ff = ff.div(fnorm.expand_as(ff))
    return ff


class ArcLinear(nn.Module):

    def __init__(self, in_features, out_features, s=64.0):
        super(ArcLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        init.normal_(self.weight.data, std=0.001)
        self.loss_s = s

    def forward(self, input):
        embedding = input
        nembedding = L2Normalization(embedding, dim=1) * self.loss_s
        _weight = L2Normalization(self.weight, dim=0)
        fc7 = nembedding.mm(_weight)
        output = fc7, _weight, nembedding
        return output


class ArcLoss(nn.Module):

    def __init__(self, m1=1.0, m2=0.5, m3=0.0, s=64.0):
        super(ArcLoss, self).__init__()
        self.loss_m1 = m1
        self.loss_m2 = m2
        self.loss_m3 = m3
        self.loss_s = s

    def forward(self, input, target):
        fc7, _weight, nembedding = input
        index = fc7.data * 0.0
        index.scatter_(1, target.data.view(-1, 1), 1)
        index = index.byte()
        index = Variable(index)
        zy = fc7[index]
        cos_t = zy / self.loss_s
        t = torch.acos(cos_t)
        t = t * self.loss_m1 + self.loss_m2
        body = torch.cos(t) - self.loss_m3
        new_zy = body * self.loss_s
        diff = new_zy - zy
        fc7[index] += diff
        loss = F.cross_entropy(fc7, target)
        return loss


class AngleLoss(nn.Module):

    def __init__(self, gamma=0):
        super(AngleLoss, self).__init__()
        self.gamma = gamma
        self.it = 0
        self.LambdaMin = 5.0
        self.LambdaMax = 1500.0
        self.lamb = 1500.0

    def forward(self, input, target):
        self.it += 1
        cos_theta, phi_theta = input
        target = target.view(-1, 1)
        index = cos_theta.data * 0.0
        index.scatter_(1, target.data.view(-1, 1), 1)
        index = index.byte()
        index = Variable(index)
        self.lamb = max(self.LambdaMin, self.LambdaMax / (1 + 0.1 * self.it))
        output = cos_theta * 1.0
        output[index] -= cos_theta[index] * (1.0 + 0) / (1 + self.lamb)
        output[index] += phi_theta[index] * (1.0 + 0) / (1 + self.lamb)
        logpt = F.log_softmax(output, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())
        loss = -1 * (1 - pt) ** self.gamma * logpt
        loss = loss.mean()
        return loss


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


class ClassBlock(nn.Module):

    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True, return_f=False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate > 0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)
        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)
        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return x, f
        else:
            x = self.classifier(x)
            return x


class ft_net(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, init_model=None, pool='avg'):
        super(ft_net, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = 1, 1
            model_ft.layer4[0].conv2.stride = 1, 1
        self.pool = pool
        if pool == 'avg+max':
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
            model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1, 1))
            self.model = model_ft
            self.classifier = ClassBlock(4096, class_num, droprate)
        elif pool == 'avg':
            model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.model = model_ft
            self.classifier = ClassBlock(2048, class_num, droprate)
        self.flag = False
        if init_model != None:
            self.flag = True
            self.model = init_model.model
            self.pool = init_model.pool
            self.classifier.add_block = init_model.classifier.add_block
            self.new_dropout = nn.Sequential(nn.Dropout(p=droprate))

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        if self.pool == 'avg+max':
            x1 = self.model.avgpool2(x)
            x2 = self.model.maxpool2(x)
            x = torch.cat((x1, x2), dim=1)
            x = x.view(x.size(0), x.size(1))
        elif self.pool == 'avg':
            x = self.model.avgpool(x)
            x = x.view(x.size(0), x.size(1))
        if self.flag:
            x = self.classifier.add_block(x)
            x = self.new_dropout(x)
            x = self.classifier.classifier(x)
        else:
            x = self.classifier(x)
        return x


class ft_net_angle(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2):
        super(ft_net_angle, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = 1, 1
            model_ft.layer4[0].conv2.stride = 1, 1
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft
        self.classifier = ClassBlock(2048, class_num, droprate)
        self.classifier.classifier = AngleLinear(512, class_num)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x


class ft_net_arc(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2):
        super(ft_net_arc, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = 1, 1
            model_ft.layer4[0].conv2.stride = 1, 1
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft
        self.classifier = ClassBlock(2048, class_num, droprate)
        self.classifier.classifier = ArcLinear(512, class_num)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x


class ft_net_dense(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, init_model=None, pool='avg'):
        super().__init__()
        model_ft = models.densenet121(pretrained=True)
        if stride == 1:
            model_ft.features.transition3.pool.stride = 1
        model_ft.fc = nn.Sequential()
        self.pool = pool
        if pool == 'avg+max':
            model_ft.features.avgpool = nn.Sequential()
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
            model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1, 1))
            self.model = model_ft
            self.classifier = ClassBlock(2048, class_num, droprate)
        elif pool == 'avg':
            model_ft.features.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.model = model_ft
            self.classifier = ClassBlock(1024, class_num, droprate)
        self.flag = False
        if init_model != None:
            self.flag = True
            self.model = init_model.model
            self.pool = init_model.pool
            self.classifier.add_block = init_model.classifier.add_block
            self.new_dropout = nn.Sequential(nn.Dropout(p=droprate))

    def forward(self, x):
        if self.pool == 'avg':
            x = self.model.features(x)
        elif self.pool == 'avg+max':
            x = self.model.features(x)
            x1 = self.model.avgpool2(x)
            x2 = self.model.maxpool2(x)
            x = torch.cat((x1, x2), dim=1)
        x = x.view(x.size(0), x.size(1))
        if self.flag:
            x = self.classifier.add_block(x)
            x = self.new_dropout(x)
            x = self.classifier.classifier(x)
        else:
            x = self.classifier(x)
        return x


class ft_net_EF4(nn.Module):

    def __init__(self, class_num, droprate=0.2):
        super().__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b4')
        self.model._fc = nn.Sequential()
        self.classifier = ClassBlock(1792, class_num, droprate)

    def forward(self, x):
        x = self.model.extract_features(x)
        x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        x = self.classifier(x)
        return x


class ft_net_EF5(nn.Module):

    def __init__(self, class_num, droprate=0.2):
        super().__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b5')
        self.model._fc = nn.Sequential()
        self.classifier = ClassBlock(2048, class_num, droprate)

    def forward(self, x):
        x = self.model.extract_features(x)
        x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        x = self.classifier(x)
        return x


class ft_net_EF6(nn.Module):

    def __init__(self, class_num, droprate=0.2):
        super().__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b6')
        self.model._fc = nn.Sequential()
        self.classifier = ClassBlock(2304, class_num, droprate)

    def forward(self, x):
        x = self.model.extract_features(x)
        x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        x = self.classifier(x)
        return x


def fix_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


class ft_net_NAS(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2):
        super().__init__()
        model_name = 'nasnetalarge'
        model_ft = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        model_ft.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        model_ft.dropout = nn.Sequential()
        model_ft.last_linear = nn.Sequential()
        model_ft.cell_17.apply(fix_relu)
        self.model = model_ft
        self.classifier = ClassBlock(4032, class_num, droprate)

    def forward(self, x):
        x = self.model.features(x)
        x = self.model.avg_pool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x


class ft_net_SE(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, pool='avg', init_model=None):
        super().__init__()
        model_name = 'se_resnext101_32x4d'
        model_ft = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        if stride == 1:
            model_ft.layer4[0].conv2.stride = 1, 1
            model_ft.layer4[0].downsample[0].stride = 1, 1
        if pool == 'avg':
            model_ft.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        elif pool == 'max':
            model_ft.avg_pool = nn.AdaptiveMaxPool2d((1, 1))
        elif pool == 'avg+max':
            model_ft.avg_pool2 = nn.AdaptiveAvgPool2d((1, 1))
            model_ft.max_pool2 = nn.AdaptiveMaxPool2d((1, 1))
        else:
            None
        model_ft.last_linear = nn.Sequential()
        self.model = model_ft
        self.pool = pool
        if pool == 'avg+max':
            self.classifier = ClassBlock(4096, class_num, droprate)
        else:
            self.classifier = ClassBlock(2048, class_num, droprate)
        self.flag = False
        if init_model != None:
            self.flag = True
            self.model = init_model.model
            self.classifier.add_block = init_model.classifier.add_block
            self.new_dropout = nn.Sequential(nn.Dropout(p=droprate))

    def forward(self, x):
        x = self.model.features(x)
        if self.pool == 'avg+max':
            x1 = self.model.avg_pool2(x)
            x2 = self.model.max_pool2(x)
            x = torch.cat((x1, x2), dim=1)
        else:
            x = self.model.avg_pool(x)
        x = x.view(x.size(0), x.size(1))
        if self.flag:
            x = self.classifier.add_block(x)
            x = self.new_dropout(x)
            x = self.classifier.classifier(x)
        else:
            x = self.classifier(x)
        return x


class ft_net_DSE(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, pool='avg'):
        super().__init__()
        model_name = 'senet154'
        model_ft = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        if stride == 1:
            model_ft.layer4[0].conv2.stride = 1, 1
            model_ft.layer4[0].downsample[0].stride = 1, 1
        if pool == 'avg':
            model_ft.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        elif pool == 'max':
            model_ft.avg_pool = nn.AdaptiveMaxPool2d((1, 1))
        else:
            None
        model_ft.dropout = nn.Sequential()
        model_ft.last_linear = nn.Sequential()
        self.model = model_ft
        self.classifier = ClassBlock(2048, class_num, droprate)

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x


class ft_net_IR(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2):
        super().__init__()
        model_name = 'inceptionresnetv2'
        model_ft = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        if stride == 1:
            model_ft.mixed_7a.branch0[1].conv.stride = 1, 1
            model_ft.mixed_7a.branch1[1].conv.stride = 1, 1
            model_ft.mixed_7a.branch2[2].conv.stride = 1, 1
            model_ft.mixed_7a.branch3.stride = 1
        model_ft.avgpool_1a = nn.AdaptiveAvgPool2d((1, 1))
        model_ft.last_linear = nn.Sequential()
        self.model = model_ft
        self.classifier = ClassBlock(1536, class_num, droprate)

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x


class ft_net_middle(nn.Module):

    def __init__(self, class_num, droprate=0.5):
        super(ft_net_middle, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft
        self.classifier = ClassBlock(2048 + 1024, class_num, droprate)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x0 = self.model.avgpool(x)
        x = self.model.layer4(x)
        x1 = self.model.avgpool(x)
        x = torch.cat((x0, x1), 1)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x


class PCB(nn.Module):

    def __init__(self, class_num):
        super(PCB, self).__init__()
        self.part = 6
        model_ft = models.resnet50(pretrained=True)
        self.model = model_ft
        self.avgpool = nn.AdaptiveAvgPool2d((self.part, 1))
        self.dropout = nn.Dropout(p=0.5)
        self.model.layer4[0].downsample[0].stride = 1, 1
        self.model.layer4[0].conv2.stride = 1, 1
        for i in range(self.part):
            name = 'classifier' + str(i)
            setattr(self, name, ClassBlock(2048, class_num, droprate=0.5, relu=False, bnorm=True, num_bottleneck=256))

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        part = {}
        predict = {}
        for i in range(self.part):
            part[i] = torch.squeeze(x[:, :, (i)])
            name = 'classifier' + str(i)
            c = getattr(self, name)
            predict[i] = c(part[i])
        y = []
        for i in range(self.part):
            y.append(predict[i])
        return y


class PCB_test(nn.Module):

    def __init__(self, model):
        super(PCB_test, self).__init__()
        self.part = 6
        self.model = model.model
        self.avgpool = nn.AdaptiveAvgPool2d((self.part, 1))
        self.model.layer4[0].downsample[0].stride = 1, 1
        self.model.layer4[0].conv2.stride = 1, 1

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        y = x.view(x.size(0), x.size(1), x.size(2))
        return y


class CPB(nn.Module):

    def __init__(self, class_num):
        super(CPB, self).__init__()
        self.part = 4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model_name = 'se_resnext101_32x4d'
        model_ft = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        model_ft.layer4[0].conv2.stride = 1, 1
        model_ft.layer4[0].downsample[0].stride = 1, 1
        self.model = model_ft
        for i in range(self.part):
            name = 'classifier' + str(i)
            setattr(self, name, ClassBlock(2048, class_num, droprate=0.2, relu=False, bnorm=True, num_bottleneck=512))

    def forward(self, x):
        x = self.model.features(x)
        part = {}
        predict = {}
        d = 2 + 2 + 2
        for i in range(self.part):
            N, C, H, W = x.shape
            p = 2
            if i == 0:
                part_input = x[:, :, d:W - d, d:H - d]
                part[i] = torch.squeeze(self.avgpool(part_input))
                last_input = torch.nn.functional.pad(part_input, (p, p, p, p), mode='constant', value=0)
            else:
                part_input = x[:, :, d:W - d, d:H - d] - last_input
                part[i] = torch.squeeze(self.avgpool(part_input))
                last_input = torch.nn.functional.pad(part_input, (p, p, p, p), mode='constant', value=0)
            name = 'classifier' + str(i)
            c = getattr(self, name)
            predict[i] = c(part[i])
            d = d - p
        y = []
        for i in range(self.part):
            y.append(predict[i])
        return y


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AngleLinear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (ArcLinear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (ClassBlock,
     lambda: ([], {'input_dim': 4, 'class_num': 4, 'droprate': 0.5}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (PCB,
     lambda: ([], {'class_num': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (ft_net,
     lambda: ([], {'class_num': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (ft_net_angle,
     lambda: ([], {'class_num': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (ft_net_arc,
     lambda: ([], {'class_num': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (ft_net_dense,
     lambda: ([], {'class_num': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (ft_net_middle,
     lambda: ([], {'class_num': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
]

class Test_layumi_AICIty_reID_2020(_paritybench_base):
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

