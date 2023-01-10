import sys
_module = sys.modules[__name__]
del sys
resize = _module
resize_short_edge = _module
validating_files = _module
data_loader = _module
augmentations = _module
distributed = _module
general_helper_functions = _module
losses = _module
models = _module
ofa = _module
layers = _module
model_zoo = _module
utils = _module
tresnet = _module
anti_aliasing = _module
avg_pool = _module
general_layers = _module
tresnet = _module
factory = _module
create_optimizer = _module
metrics = _module
semantic_loss = _module
semantics = _module
train_semantic_softmax = _module
train_single_label = _module
train_single_label_from_scratch = _module
visualize_detector = _module

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


import torch


from torchvision import transforms


from torchvision.datasets import ImageFolder


import torch.distributed as dist


import torch.nn as nn


from collections import OrderedDict


import math


import torch.nn.parallel


import numpy as np


import torch.nn.functional as F


from torch.nn import Module as Module


from torch.optim import lr_scheduler


from torch import Tensor


import time


import torch.optim


import torch.utils.data.distributed


from torch.cuda.amp import GradScaler


from torch.cuda.amp import autocast


import matplotlib


import matplotlib.pyplot as plt


class CrossEntropyLS(nn.Module):

    def __init__(self, eps: float=0.2):
        super(CrossEntropyLS, self).__init__()
        self.eps = eps
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, inputs, target):
        num_classes = inputs.size()[-1]
        log_preds = self.logsoftmax(inputs)
        targets_classes = torch.zeros_like(inputs).scatter_(1, target.long().unsqueeze(1), 1)
        targets_classes.mul_(1 - self.eps).add_(self.eps / num_classes)
        cross_entropy_loss_tot = -targets_classes.mul(log_preds)
        cross_entropy_loss = cross_entropy_loss_tot.sum(dim=-1).mean()
        return cross_entropy_loss


class MyModule(nn.Module):

    def forward(self, x):
        raise NotImplementedError

    @property
    def module_str(self):
        raise NotImplementedError

    @property
    def config(self):
        raise NotImplementedError

    @staticmethod
    def build_from_config(config):
        raise NotImplementedError


class MyNetwork(MyModule):

    def forward(self, x):
        raise NotImplementedError

    @property
    def module_str(self):
        raise NotImplementedError

    @property
    def config(self):
        raise NotImplementedError

    @staticmethod
    def build_from_config(config):
        raise NotImplementedError

    def zero_last_gamma(self):
        raise NotImplementedError
    """ implemented methods """

    def set_bn_param(self, momentum, eps):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.momentum = momentum
                m.eps = eps
        return

    def get_bn_param(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                return {'momentum': m.momentum, 'eps': m.eps}
        return None

    def init_model(self, model_init):
        """ Conv2d, BatchNorm2d, BatchNorm1d, Linear, """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if model_init == 'he_fout':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2.0 / n))
                elif model_init == 'he_fin':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                    m.weight.data.normal_(0, math.sqrt(2.0 / n))
                else:
                    raise NotImplementedError
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                stdv = 1.0 / math.sqrt(m.weight.size(1))
                m.weight.data.uniform_(-stdv, stdv)
                if m.bias is not None:
                    m.bias.data.zero_()

    def get_parameters(self, keys=None, mode='include', exclude_set=None):
        if exclude_set is None:
            exclude_set = {}
        if keys is None:
            for name, param in self.named_parameters():
                if name not in exclude_set:
                    yield param
        elif mode == 'include':
            for name, param in self.named_parameters():
                flag = False
                for key in keys:
                    if key in name:
                        flag = True
                        break
                if flag and name not in exclude_set:
                    yield param
        elif mode == 'exclude':
            for name, param in self.named_parameters():
                flag = True
                for key in keys:
                    if key in name:
                        flag = False
                        break
                if flag and name not in exclude_set:
                    yield param
        else:
            raise ValueError('do not support: %s' % mode)

    def weight_parameters(self, exclude_set=None):
        return self.get_parameters(exclude_set=exclude_set)


class ShuffleLayer(nn.Module):

    def __init__(self, groups):
        super(ShuffleLayer, self).__init__()
        self.groups = groups

    def forward(self, x):
        batchsize, num_channels, height, width = x.size()
        channels_per_group = num_channels // self.groups
        x = x.view(batchsize, self.groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)
        return x


class bottleneck_head(nn.Module):

    def __init__(self, num_features, num_classes, bottleneck_features=200):
        super(bottleneck_head, self).__init__()
        self.embedding_generator = nn.ModuleList()
        self.embedding_generator.append(nn.Linear(num_features, bottleneck_features))
        self.embedding_generator = nn.Sequential(*self.embedding_generator)
        self.FC = nn.Linear(bottleneck_features, num_classes)

    def forward(self, x):
        self.embedding = self.embedding_generator(x)
        logits = self.FC(self.embedding)
        return logits


class Hswish(nn.Module):

    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * torch.sigmoid(x)


class Hsigmoid(nn.Module):

    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return torch.sigmoid(x)


class FastAvgPool2d(nn.Module):

    def __init__(self, flatten=False):
        super(FastAvgPool2d, self).__init__()
        self.flatten = flatten

    def forward(self, x):
        if self.flatten:
            in_size = x.size()
            return x.view((in_size[0], in_size[1], -1)).mean(dim=2)
        else:
            return x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1, 1)


class SEModule(nn.Module):

    def __init__(self, channels, reduction_channels, inplace=True):
        super(SEModule, self).__init__()
        self.avg_pool = FastAvgPool2d()
        self.fc1 = nn.Conv2d(channels, reduction_channels, kernel_size=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=inplace)
        self.fc2 = nn.Conv2d(reduction_channels, channels, kernel_size=1, padding=0, bias=True)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se2 = self.fc1(x_se)
        x_se2 = self.relu(x_se2)
        x_se = self.fc2(x_se2)
        x_se = self.activation(x_se)
        return x * x_se


class Downsample(nn.Module):

    def __init__(self, filt_size=3, stride=2, channels=None):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.stride = stride
        self.channels = channels
        assert self.filt_size == 3
        a = torch.tensor([1.0, 2.0, 1.0])
        filt = (a[:, None] * a[None, :]).clone().detach()
        filt = filt / torch.sum(filt)
        self.filt = filt[None, None, :, :].repeat((self.channels, 1, 1, 1))

    def forward(self, input):
        input_pad = F.pad(input, (1, 1, 1, 1), 'reflect')
        return F.conv2d(input_pad, self.filt, stride=self.stride, padding=0, groups=input.shape[1])


class AntiAliasDownsampleLayer(nn.Module):

    def __init__(self, remove_model_jit: bool=False, filt_size: int=3, stride: int=2, channels: int=0):
        super(AntiAliasDownsampleLayer, self).__init__()
        if not remove_model_jit:
            self.op = DownsampleJIT(filt_size, stride, channels)
        else:
            self.op = Downsample(filt_size, stride, channels)

    def forward(self, x):
        return self.op(x)


class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)


class DepthToSpace(nn.Module):

    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.bs, self.bs, C // self.bs ** 2, H, W)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()
        x = x.view(N, C // self.bs ** 2, H * self.bs, W * self.bs)
        return x


class SpaceToDepth(nn.Module):

    def __init__(self, block_size=4):
        super().__init__()
        assert block_size == 4
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, C, H // self.bs, self.bs, W // self.bs, self.bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        x = x.view(N, C * self.bs ** 2, H // self.bs, W // self.bs)
        return x


class SpaceToDepthModule(nn.Module):

    def __init__(self, remove_model_jit=False):
        super().__init__()
        if not remove_model_jit:
            self.op = SpaceToDepthJit()
        else:
            self.op = SpaceToDepth()

    def forward(self, x):
        return self.op(x)


class hard_sigmoid(nn.Module):

    def __init__(self, inplace=True):
        super(hard_sigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            return x.add_(3.0).clamp_(0.0, 6.0).div_(6.0)
        else:
            return F.relu6(x + 3.0) / 6.0


def conv2d_ABN(ni, nf, stride, activation='leaky_relu', kernel_size=3, activation_param=0.01, groups=1):
    return nn.Sequential(nn.Conv2d(ni, nf, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, groups=groups, bias=False), InPlaceABN(num_features=nf, activation=activation, activation_param=activation_param))


class BasicBlock(Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True, anti_alias_layer=None):
        super(BasicBlock, self).__init__()
        if stride == 1:
            self.conv1 = conv2d_ABN(inplanes, planes, stride=1, activation_param=0.001)
        elif anti_alias_layer is None:
            self.conv1 = conv2d_ABN(inplanes, planes, stride=2, activation_param=0.001)
        else:
            self.conv1 = nn.Sequential(conv2d_ABN(inplanes, planes, stride=1, activation_param=0.001), anti_alias_layer(channels=planes, filt_size=3, stride=2))
        self.conv2 = conv2d_ABN(planes, planes, stride=1, activation='identity')
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        reduce_layer_planes = max(planes * self.expansion // 4, 64)
        self.se = SEModule(planes * self.expansion, reduce_layer_planes) if use_se else None

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.se is not None:
            out = self.se(out)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True, anti_alias_layer=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv2d_ABN(inplanes, planes, kernel_size=1, stride=1, activation='leaky_relu', activation_param=0.001)
        if stride == 1:
            self.conv2 = conv2d_ABN(planes, planes, kernel_size=3, stride=1, activation='leaky_relu', activation_param=0.001)
        elif anti_alias_layer is None:
            self.conv2 = conv2d_ABN(planes, planes, kernel_size=3, stride=2, activation='leaky_relu', activation_param=0.001)
        else:
            self.conv2 = nn.Sequential(conv2d_ABN(planes, planes, kernel_size=3, stride=1, activation='leaky_relu', activation_param=0.001), anti_alias_layer(channels=planes, filt_size=3, stride=2))
        self.conv3 = conv2d_ABN(planes, planes * self.expansion, kernel_size=1, stride=1, activation='identity')
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        reduce_layer_planes = max(planes * self.expansion // 8, 64)
        self.se = SEModule(planes, reduce_layer_planes) if use_se else None

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.se is not None:
            out = self.se(out)
        out = self.conv3(out)
        out = out + residual
        out = self.relu(out)
        return out


class TResNet(Module):

    def __init__(self, layers, in_chans=3, num_classes=1000, width_factor=1.0, do_bottleneck_head=False, bottleneck_features=512, first_two_layers=BasicBlock):
        super(TResNet, self).__init__()
        space_to_depth = SpaceToDepthModule()
        anti_alias_layer = AntiAliasDownsampleLayer
        global_pool_layer = FastAvgPool2d(flatten=True)
        self.inplanes = int(64 * width_factor)
        self.planes = int(64 * width_factor)
        conv1 = conv2d_ABN(in_chans * 16, self.planes, stride=1, kernel_size=3)
        layer1 = self._make_layer(first_two_layers, self.planes, layers[0], stride=1, use_se=True, anti_alias_layer=anti_alias_layer)
        layer2 = self._make_layer(first_two_layers, self.planes * 2, layers[1], stride=2, use_se=True, anti_alias_layer=anti_alias_layer)
        layer3 = self._make_layer(Bottleneck, self.planes * 4, layers[2], stride=2, use_se=True, anti_alias_layer=anti_alias_layer)
        layer4 = self._make_layer(Bottleneck, self.planes * 8, layers[3], stride=2, use_se=False, anti_alias_layer=anti_alias_layer)
        self.body = nn.Sequential(OrderedDict([('SpaceToDepth', space_to_depth), ('conv1', conv1), ('layer1', layer1), ('layer2', layer2), ('layer3', layer3), ('layer4', layer4)]))
        self.embeddings = []
        self.global_pool = nn.Sequential(OrderedDict([('global_pool_layer', global_pool_layer)]))
        self.num_features = self.planes * 8 * Bottleneck.expansion
        if do_bottleneck_head:
            fc = bottleneck_head(self.num_features, num_classes, bottleneck_features=bottleneck_features)
        else:
            fc = nn.Linear(self.num_features, num_classes)
        self.head = nn.Sequential(OrderedDict([('fc', fc)]))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, InPlaceABN):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in self.modules():
            if isinstance(m, BasicBlock):
                m.conv2[1].weight = nn.Parameter(torch.zeros_like(m.conv2[1].weight))
            if isinstance(m, Bottleneck):
                m.conv3[1].weight = nn.Parameter(torch.zeros_like(m.conv3[1].weight))
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)

    def _make_layer(self, block, planes, blocks, stride=1, use_se=True, anti_alias_layer=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            layers = []
            if stride == 2:
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True, count_include_pad=False))
            layers += [conv2d_ABN(self.inplanes, planes * block.expansion, kernel_size=1, stride=1, activation='identity')]
            downsample = nn.Sequential(*layers)
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_se=use_se, anti_alias_layer=anti_alias_layer))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_se=use_se, anti_alias_layer=anti_alias_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.body(x)
        self.embeddings = self.global_pool(x)
        logits = self.head(self.embeddings)
        return logits


class SemanticSoftmaxLoss(torch.nn.Module):

    def __init__(self, semantic_softmax_processor):
        super(SemanticSoftmaxLoss, self).__init__()
        self.semantic_softmax_processor = semantic_softmax_processor
        self.args = semantic_softmax_processor.args

    def forward(self, logits, targets):
        """
        Calculates the semantic cross-entropy loss distance between logits and targers
        """
        if not self.training:
            return 0
        semantic_logit_list = self.semantic_softmax_processor.split_logits_to_semantic_logits(logits)
        semantic_targets_tensor = self.semantic_softmax_processor.convert_targets_to_semantic_targets(targets)
        losses_list = []
        for i in range(len(semantic_logit_list)):
            logits_i = semantic_logit_list[i]
            targets_i = semantic_targets_tensor[:, i]
            log_preds = F.log_softmax(logits_i, dim=1)
            targets_i_valid = targets_i.clone()
            targets_i_valid[targets_i_valid < 0] = 0
            num_classes = logits_i.size()[-1]
            targets_classes = torch.zeros_like(logits_i).scatter_(1, targets_i_valid.unsqueeze(1), 1)
            targets_classes.mul_(1 - self.args.label_smooth).add_(self.args.label_smooth / num_classes)
            cross_entropy_loss_tot = -targets_classes.mul(log_preds)
            cross_entropy_loss_tot *= (targets_i >= 0).unsqueeze(1)
            cross_entropy_loss = cross_entropy_loss_tot.sum(dim=-1)
            loss_i = cross_entropy_loss.mean()
            losses_list.append(loss_i)
        total_sum = 0
        for i, loss_h in enumerate(losses_list):
            total_sum += loss_h * self.semantic_softmax_processor.normalization_factor_list[i]
        return total_sum


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (FastAvgPool2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Flatten,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Hsigmoid,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Hswish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SEModule,
     lambda: ([], {'channels': 4, 'reduction_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ShuffleLayer,
     lambda: ([], {'groups': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SpaceToDepth,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (bottleneck_head,
     lambda: ([], {'num_features': 4, 'num_classes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (hard_sigmoid,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_Alibaba_MIIL_ImageNet21K(_paritybench_base):
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

