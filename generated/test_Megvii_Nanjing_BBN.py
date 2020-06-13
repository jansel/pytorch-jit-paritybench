import sys
_module = sys.modules[__name__]
del sys
backbone = _module
resnet = _module
resnet_cifar = _module
config = _module
default = _module
core = _module
combiner = _module
evaluate = _module
function = _module
data_transform = _module
transform_wrapper = _module
dataset = _module
baseset = _module
iNaturalist = _module
imbalance_cifar = _module
loss = _module
loss = _module
modules = _module
classifier_ops = _module
pooling_ops = _module
net = _module
network = _module
utils = _module
lr_scheduler = _module
registry = _module
utils = _module
_init_paths = _module
train = _module
valid = _module
convert_from_iNat = _module

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


import math


import torch.nn.init as init


import numpy as np


from torch.nn import functional as F


import logging


from torch.utils.data import DataLoader


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1,
            bias=False, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1,
            bias=False, stride=1)
        self.bn2 = nn.BatchNorm2d(planes)
        if stride != 1 or self.expansion * planes != inplanes:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, self.
                expansion * planes, kernel_size=1, stride=stride, bias=
                False), nn.BatchNorm2d(self.expansion * planes))
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(True)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size
            =1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        if stride != 1 or self.expansion * planes != inplanes:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, self.
                expansion * planes, kernel_size=1, stride=stride, bias=
                False), nn.BatchNorm2d(self.expansion * planes))
        else:
            self.downsample = None
        self.relu = nn.ReLU(True)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample != None:
            residual = self.downsample(x)
        else:
            residual = x
        out = out + residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, cfg, block_type, num_blocks, last_layer_stride=2):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.block = block_type
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(num_blocks[0], 64)
        self.layer2 = self._make_layer(num_blocks[1], 128, stride=2)
        self.layer3 = self._make_layer(num_blocks[2], 256, stride=2)
        self.layer4 = self._make_layer(num_blocks[3], 512, stride=
            last_layer_stride)

    def load_model(self, pretrain):
        None
        model_dict = self.state_dict()
        pretrain_dict = torch.load(pretrain)
        pretrain_dict = pretrain_dict['state_dict'
            ] if 'state_dict' in pretrain_dict else pretrain_dictNone
        new_dict = OrderedDict()
        for k, v in pretrain_dict.items():
            if k.startswith('module'):
                k = k[7:]
            if 'fc' not in k and 'classifier' not in k:
                k = k.replace('backbone.', '')
                new_dict[k] = v
        model_dict.update(new_dict)
        self.load_state_dict(model_dict)
        None

    def _make_layer(self, num_block, planes, stride=1):
        strides = [stride] + [1] * (num_block - 1)
        layers = []
        for now_stride in strides:
            layers.append(self.block(self.inplanes, planes, stride=now_stride))
            self.inplanes = planes * self.block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.pool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out


class BBN_ResNet(nn.Module):

    def __init__(self, cfg, block_type, num_blocks, last_layer_stride=2):
        super(BBN_ResNet, self).__init__()
        self.inplanes = 64
        self.block = block_type
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(num_blocks[0], 64)
        self.layer2 = self._make_layer(num_blocks[1], 128, stride=2)
        self.layer3 = self._make_layer(num_blocks[2], 256, stride=2)
        self.layer4 = self._make_layer(num_blocks[3] - 1, 512, stride=
            last_layer_stride)
        self.cb_block = self.block(self.inplanes, self.inplanes // 4, stride=1)
        self.rb_block = self.block(self.inplanes, self.inplanes // 4, stride=1)

    def load_model(self, pretrain):
        None
        model_dict = self.state_dict()
        pretrain_dict = torch.load(pretrain)
        pretrain_dict = pretrain_dict['state_dict'
            ] if 'state_dict' in pretrain_dict else pretrain_dictNone
        new_dict = OrderedDict()
        for k, v in pretrain_dict.items():
            if k.startswith('module'):
                k = k[7:]
            if 'fc' not in k and 'classifier' not in k:
                k = k.replace('backbone.', '')
                new_dict[k] = v
        model_dict.update(new_dict)
        self.load_state_dict(model_dict)
        None

    def _make_layer(self, num_block, planes, stride=1):
        strides = [stride] + [1] * (num_block - 1)
        layers = []
        for now_stride in strides:
            layers.append(self.block(self.inplanes, planes, stride=now_stride))
            self.inplanes = planes * self.block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, **kwargs):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.pool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        if 'feature_cb' in kwargs:
            out = self.cb_block(out)
            return out
        elif 'feature_rb' in kwargs:
            out = self.rb_block(out)
            return out
        out1 = self.cb_block(out)
        out2 = self.rb_block(out)
        out = torch.cat((out1, out2), dim=1)
        return out


class LambdaLayer(nn.Module):

    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=
            stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::
                    2], (0, 0, 0, 0, planes // 4, planes // 4), 'constant', 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.
                    expansion * planes, kernel_size=1, stride=stride, bias=
                    False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class ResNet_Cifar(nn.Module):

    def __init__(self, block, num_blocks):
        super(ResNet_Cifar, self).__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1,
            bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def load_model(self, pretrain):
        None
        model_dict = self.state_dict()
        pretrain_dict = torch.load(pretrain)
        pretrain_dict = pretrain_dict['state_dict'
            ] if 'state_dict' in pretrain_dict else pretrain_dictNone
        new_dict = OrderedDict()
        for k, v in pretrain_dict.items():
            if k.startswith('module'):
                k = k[7:]
            if ('last_linear' not in k and 'classifier' not in k and 
                'linear' not in k and 'fd' not in k):
                k = k.replace('backbone.', '')
                k = k.replace('fr', 'layer3.4')
                new_dict[k] = v
        model_dict.update(new_dict)
        self.load_state_dict(model_dict)
        None

    def forward(self, x, **kwargs):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        return out


class CrossEntropy(nn.Module):

    def __init__(self, para_dict=None):
        super(CrossEntropy, self).__init__()

    def forward(self, output, target):
        output = output
        loss = F.cross_entropy(output, target)
        return loss


class CSCE(nn.Module):

    def __init__(self, para_dict=None):
        super(CSCE, self).__init__()
        self.num_class_list = para_dict['num_class_list']
        self.device = para_dict['device']
        cfg = para_dict['cfg']
        scheduler = cfg.LOSS.CSCE.SCHEDULER
        self.step_epoch = cfg.LOSS.CSCE.DRW_EPOCH
        if scheduler == 'drw':
            self.betas = [0, 0.999999]
        elif scheduler == 'default':
            self.betas = [0.999999, 0.999999]
        self.weight = None

    def update_weight(self, beta):
        effective_num = 1.0 - np.power(beta, self.num_class_list)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(self
            .num_class_list)
        self.weight = torch.FloatTensor(per_cls_weights).to(self.device)

    def reset_epoch(self, epoch):
        idx = (epoch - 1) // self.step_epoch
        beta = self.betas[idx]
        self.update_weight(beta)

    def forward(self, x, target, **kwargs):
        return F.cross_entropy(x, target, weight=self.weight)


class LDAMLoss(nn.Module):

    def __init__(self, para_dict=None):
        super(LDAMLoss, self).__init__()
        s = 30
        self.num_class_list = para_dict['num_class_list']
        self.device = para_dict['device']
        cfg = para_dict['cfg']
        max_m = cfg.LOSS.LDAM.MAX_MARGIN
        m_list = 1.0 / np.sqrt(np.sqrt(self.num_class_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.FloatTensor(m_list).to(self.device)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.step_epoch = cfg.LOSS.LDAM.DRW_EPOCH
        self.weight = None

    def reset_epoch(self, epoch):
        idx = (epoch - 1) // self.step_epoch
        betas = [0, 0.9999]
        effective_num = 1.0 - np.power(betas[idx], self.num_class_list)
        per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(self
            .num_class_list)
        self.weight = torch.FloatTensor(per_cls_weights).to(self.device)

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        index_float = index.type(torch.FloatTensor)
        index_float = index_float.to(self.device)
        batch_m = torch.matmul(self.m_list[(None), :], index_float.
            transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s * output, target, weight=self.weight)


class FCNorm(nn.Module):

    def __init__(self, num_features, num_classes):
        super(FCNorm, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, num_features)
            )
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-05).mul_(100000.0)

    def forward(self, x):
        out = F.linear(F.normalize(x), F.normalize(self.weight))
        return out


class GAP(nn.Module):
    """Global Average pooling
        Widely used in ResNet, Inception, DenseNet, etc.
     """

    def __init__(self):
        super(GAP, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.avgpool(x)
        return x


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_Megvii_Nanjing_BBN(_paritybench_base):
    pass
    def test_000(self):
        self._check(BasicBlock(*[], **{'in_planes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(BottleNeck(*[], **{'inplanes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(FCNorm(*[], **{'num_features': 4, 'num_classes': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(GAP(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_004(self):
        self._check(Identity(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

