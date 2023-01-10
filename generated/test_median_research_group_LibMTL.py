import sys
_module = sys.modules[__name__]
del sys
LibMTL = _module
_record = _module
CGC = _module
Cross_stitch = _module
DSelect_k = _module
HPS = _module
LTB = _module
MMoE = _module
MTAN = _module
PLE = _module
architecture = _module
abstract_arch = _module
config = _module
loss = _module
metrics = _module
model = _module
resnet = _module
resnet_dilated = _module
trainer = _module
utils = _module
CAGrad = _module
DWA = _module
EW = _module
GLS = _module
GradDrop = _module
GradNorm = _module
GradVac = _module
IMTL = _module
MGDA = _module
Nash_MTL = _module
PCGrad = _module
RLW = _module
UW = _module
weighting = _module
abstract_weighting = _module
conf = _module
aspp = _module
create_dataset = _module
train_nyu = _module
utils = _module
create_dataset = _module
train_office = _module
setup = _module

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


import time


import torch.nn as nn


import torch.nn.functional as F


import numpy as np


import math


import random


from scipy.optimize import minimize


from torch.utils.data.dataset import Dataset


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


import torchvision.transforms as transforms


class _transform_resnet_cross(nn.Module):

    def __init__(self, encoder_list, task_name, device):
        super(_transform_resnet_cross, self).__init__()
        self.task_name = task_name
        self.task_num = len(task_name)
        self.device = device
        self.resnet_conv = nn.ModuleDict({task: nn.Sequential(encoder_list[tn].conv1, encoder_list[tn].bn1, encoder_list[tn].relu, encoder_list[tn].maxpool) for tn, task in enumerate(self.task_name)})
        self.resnet_layer = nn.ModuleDict({})
        for i in range(4):
            self.resnet_layer[str(i)] = nn.ModuleList([])
            for tn in range(self.task_num):
                encoder = encoder_list[tn]
                self.resnet_layer[str(i)].append(eval('encoder.layer' + str(i + 1)))
        self.cross_unit = nn.Parameter(torch.ones(4, self.task_num))

    def forward(self, inputs):
        s_rep = {task: self.resnet_conv[task](inputs) for task in self.task_name}
        ss_rep = {i: ([0] * self.task_num) for i in range(4)}
        for i in range(4):
            for tn, task in enumerate(self.task_name):
                if i == 0:
                    ss_rep[i][tn] = self.resnet_layer[str(i)][tn](s_rep[task])
                else:
                    cross_rep = sum([(self.cross_unit[i - 1][j] * ss_rep[i - 1][j]) for j in range(self.task_num)])
                    ss_rep[i][tn] = self.resnet_layer[str(i)][tn](cross_rep)
        return ss_rep[3]


class _transform_resnet_ltb(nn.Module):

    def __init__(self, encoder_list, task_name, device):
        super(_transform_resnet_ltb, self).__init__()
        self.task_name = task_name
        self.task_num = len(task_name)
        self.device = device
        self.resnet_conv = nn.ModuleDict({task: nn.Sequential(encoder_list[tn].conv1, encoder_list[tn].bn1, encoder_list[tn].relu, encoder_list[tn].maxpool) for tn, task in enumerate(self.task_name)})
        self.resnet_layer = nn.ModuleDict({})
        for i in range(4):
            self.resnet_layer[str(i)] = nn.ModuleList([])
            for tn in range(self.task_num):
                encoder = encoder_list[tn]
                self.resnet_layer[str(i)].append(eval('encoder.layer' + str(i + 1)))
        self.alpha = nn.Parameter(torch.ones(6, self.task_num, self.task_num))

    def forward(self, inputs, epoch, epochs):
        if epoch < epochs / 100:
            alpha = torch.ones(6, self.task_num, self.task_num)
        else:
            tau = epochs / 20 / np.sqrt(epoch + 1)
            alpha = F.gumbel_softmax(self.alpha, dim=-1, tau=tau, hard=True)
        ss_rep = {i: ([0] * self.task_num) for i in range(5)}
        for i in range(5):
            for tn, task in enumerate(self.task_name):
                if i == 0:
                    ss_rep[i][tn] = self.resnet_conv[task](inputs)
                else:
                    child_rep = sum([(alpha[i, tn, j] * ss_rep[i - 1][j]) for j in range(self.task_num)])
                    ss_rep[i][tn] = self.resnet_layer[str(i - 1)][tn](child_rep)
        return ss_rep[4]


class _transform_resnet_MTAN(nn.Module):

    def __init__(self, resnet_network, task_name, device):
        super(_transform_resnet_MTAN, self).__init__()
        self.task_name = task_name
        self.task_num = len(task_name)
        self.device = device
        self.forward_task = None
        self.expansion = 4 if resnet_network.feature_dim == 2048 else 1
        ch = np.array([64, 128, 256, 512]) * self.expansion
        self.shared_conv = nn.Sequential(resnet_network.conv1, resnet_network.bn1, resnet_network.relu, resnet_network.maxpool)
        self.shared_layer, self.encoder_att, self.encoder_block_att = nn.ModuleDict({}), nn.ModuleDict({}), nn.ModuleList([])
        for i in range(4):
            self.shared_layer[str(i)] = nn.ModuleList([eval('resnet_network.layer' + str(i + 1) + '[:-1]'), eval('resnet_network.layer' + str(i + 1) + '[-1]')])
            if i == 0:
                self.encoder_att[str(i)] = nn.ModuleList([self._att_layer(ch[0], ch[0] // self.expansion, ch[0]) for _ in range(self.task_num)])
            else:
                self.encoder_att[str(i)] = nn.ModuleList([self._att_layer(2 * ch[i], ch[i] // self.expansion, ch[i]) for _ in range(self.task_num)])
            if i < 3:
                self.encoder_block_att.append(self._conv_layer(ch[i], ch[i + 1] // self.expansion))
        self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2)

    def _att_layer(self, in_channel, intermediate_channel, out_channel):
        return nn.Sequential(nn.Conv2d(in_channels=in_channel, out_channels=intermediate_channel, kernel_size=1, padding=0), nn.BatchNorm2d(intermediate_channel), nn.ReLU(inplace=True), nn.Conv2d(in_channels=intermediate_channel, out_channels=out_channel, kernel_size=1, padding=0), nn.BatchNorm2d(out_channel), nn.Sigmoid())

    def _conv_layer(self, in_channel, out_channel):
        downsample = nn.Sequential(conv1x1(in_channel, self.expansion * out_channel, stride=1), nn.BatchNorm2d(self.expansion * out_channel))
        if self.expansion == 4:
            return Bottleneck(in_channel, out_channel, downsample=downsample)
        else:
            return BasicBlock(in_channel, out_channel, downsample=downsample)

    def forward(self, inputs):
        s_rep = self.shared_conv(inputs)
        ss_rep = {i: ([0] * 2) for i in range(4)}
        att_rep = [0] * self.task_num
        for i in range(4):
            for j in range(2):
                if i == 0 and j == 0:
                    sh_rep = s_rep
                elif i != 0 and j == 0:
                    sh_rep = ss_rep[i - 1][1]
                else:
                    sh_rep = ss_rep[i][0]
                ss_rep[i][j] = self.shared_layer[str(i)][j](sh_rep)
            for tn, task in enumerate(self.task_name):
                if self.forward_task is not None and task != self.forward_task:
                    continue
                if i == 0:
                    att_mask = self.encoder_att[str(i)][tn](ss_rep[i][0])
                else:
                    if ss_rep[i][0].size()[-2:] != att_rep[tn].size()[-2:]:
                        att_rep[tn] = self.down_sampling(att_rep[tn])
                    att_mask = self.encoder_att[str(i)][tn](torch.cat([ss_rep[i][0], att_rep[tn]], dim=1))
                att_rep[tn] = att_mask * ss_rep[i][1]
                if i < 3:
                    att_rep[tn] = self.encoder_block_att[i](att_rep[tn])
                if i == 0:
                    att_rep[tn] = self.down_sampling(att_rep[tn])
        if self.forward_task is None:
            return att_rep
        else:
            return att_rep[self.task_name.index(self.forward_task)]


class _transform_resnet_PLE(nn.Module):

    def __init__(self, encoder_dict, task_name, img_size, num_experts, device):
        super(_transform_resnet_PLE, self).__init__()
        self.num_experts = num_experts
        self.img_size = img_size
        self.task_name = task_name
        self.task_num = len(task_name)
        self.device = device
        self.forward_task = None
        self.specific_layer, self.shared_layer = nn.ModuleDict({}), nn.ModuleDict({})
        self.specific_layer['0'], self.shared_layer['0'] = nn.ModuleDict({}), nn.ModuleList({})
        for task in self.task_name:
            self.specific_layer['0'][task] = nn.ModuleList([])
            for k in range(self.num_experts[task]):
                encoder = encoder_dict[task][k]
                self.specific_layer['0'][task].append(nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu, encoder.maxpool))
        for k in range(self.num_experts['share']):
            encoder = encoder_dict['share'][k]
            self.shared_layer['0'].append(nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu, encoder.maxpool))
        for i in range(1, 5):
            self.specific_layer[str(i)] = nn.ModuleDict({})
            for task in self.task_name:
                self.specific_layer[str(i)][task] = nn.ModuleList([])
                for k in range(self.num_experts[task]):
                    encoder = encoder_dict[task][k]
                    self.specific_layer[str(i)][task].append(eval('encoder.layer' + str(i)))
            self.shared_layer[str(i)] = nn.ModuleList([])
            for k in range(self.num_experts['share']):
                encoder = encoder_dict['share'][k]
                self.shared_layer[str(i)].append(eval('encoder.layer' + str(i)))
        input_size = []
        with torch.no_grad():
            x = torch.rand([int(s) for s in self.img_size]).unsqueeze(0)
            input_size.append(x.size().numel())
            for i in range(4):
                x = self.shared_layer[str(i)][0](x)
                input_size.append(x.size().numel())
        self.gate_specific = nn.ModuleDict({task: nn.ModuleList([self._gate_layer(input_size[i], self.num_experts['share'] + self.num_experts[task]) for i in range(5)]) for task in self.task_name})

    def _gate_layer(self, in_channel, out_channel):
        return nn.Sequential(nn.Linear(in_channel, out_channel), nn.Softmax(dim=-1))

    def forward(self, inputs):
        gate_rep = {task: inputs for task in self.task_name}
        for i in range(5):
            for task in self.task_name:
                if self.forward_task is not None and task != self.forward_task:
                    continue
                experts_shared_rep = torch.stack([e(gate_rep[task]) for e in self.shared_layer[str(i)]])
                experts_specific_rep = torch.stack([e(gate_rep[task]) for e in self.specific_layer[str(i)][task]])
                selector = self.gate_specific[task][i](torch.flatten(gate_rep[task], start_dim=1))
                gate_rep[task] = torch.einsum('ij..., ji -> j...', torch.cat([experts_shared_rep, experts_specific_rep], dim=0), selector)
        if self.forward_task is None:
            return gate_rep
        else:
            return gate_rep[self.forward_task]


class AbsArchitecture(nn.Module):
    """An abstract class for MTL architectures.

    Args:
        task_name (list): A list of strings for all tasks.
        encoder_class (class): A neural network class.
        decoders (dict): A dictionary of name-decoder pairs of type (:class:`str`, :class:`torch.nn.Module`).
        rep_grad (bool): If ``True``, the gradient of the representation for each task can be computed.
        multi_input (bool): Is ``True`` if each task has its own input data, otherwise is ``False``. 
        device (torch.device): The device where model and data will be allocated. 
        kwargs (dict): A dictionary of hyperparameters of architectures.
     
    """

    def __init__(self, task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs):
        super(AbsArchitecture, self).__init__()
        self.task_name = task_name
        self.task_num = len(task_name)
        self.encoder_class = encoder_class
        self.decoders = decoders
        self.rep_grad = rep_grad
        self.multi_input = multi_input
        self.device = device
        self.kwargs = kwargs
        if self.rep_grad:
            self.rep_tasks = {}
            self.rep = {}

    def forward(self, inputs, task_name=None):
        """

        Args: 
            inputs (torch.Tensor): The input data.
            task_name (str, default=None): The task name corresponding to ``inputs`` if ``multi_input`` is ``True``.
        
        Returns:
            dict: A dictionary of name-prediction pairs of type (:class:`str`, :class:`torch.Tensor`).
        """
        out = {}
        s_rep = self.encoder(inputs)
        same_rep = True if not isinstance(s_rep, list) and not self.multi_input else False
        for tn, task in enumerate(self.task_name):
            if task_name is not None and task != task_name:
                continue
            ss_rep = s_rep[tn] if isinstance(s_rep, list) else s_rep
            ss_rep = self._prepare_rep(ss_rep, task, same_rep)
            out[task] = self.decoders[task](ss_rep)
        return out

    def get_share_params(self):
        """Return the shared parameters of the model.
        """
        return self.encoder.parameters()

    def zero_grad_share_params(self):
        """Set gradients of the shared parameters to zero.
        """
        self.encoder.zero_grad()

    def _prepare_rep(self, rep, task, same_rep=None):
        if self.rep_grad:
            if not same_rep:
                self.rep[task] = rep
            else:
                self.rep = rep
            self.rep_tasks[task] = rep.detach().clone()
            self.rep_tasks[task].requires_grad = True
            return self.rep_tasks[task]
        else:
            return rep


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError('Dilation > 1 not supported in BasicBlock')
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

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


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError('replace_stride_with_dilation should be None or a 3-element tuple, got {}'.format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.feature_dim = 512 * block.expansion
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride), norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))
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


class ResnetDilated(nn.Module):

    def __init__(self, orig_resnet, dilate_scale=8):
        super(ResnetDilated, self).__init__()
        from functools import partial
        if dilate_scale == 8:
            orig_resnet.layer3.apply(partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(partial(self._nostride_dilate, dilate=2))
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu = orig_resnet.relu
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4
        self.feature_dim = orig_resnet.feature_dim

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = 1, 1
                if m.kernel_size == (3, 3):
                    m.dilation = dilate // 2, dilate // 2
                    m.padding = dilate // 2, dilate // 2
            elif m.kernel_size == (3, 3):
                m.dilation = dilate, dilate
                m.padding = dilate, dilate

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward_stage(self, x, stage):
        assert stage in ['conv', 'layer1', 'layer2', 'layer3', 'layer4', 'layer1_without_conv']
        if stage == 'conv':
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.maxpool(x)
            return x
        elif stage == 'layer1':
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.maxpool(x)
            x = self.layer1(x)
            return x
        elif stage == 'layer1_without_conv':
            x = self.layer1(x)
            return x
        else:
            layer = getattr(self, stage)
            return layer(x)


def count_improvement(base_result, new_result, weight):
    """Calculate the improvement between two results as

    .. math::
        \\Delta_{\\mathrm{p}}=100\\%\\times \\frac{1}{T}\\sum_{t=1}^T 
        \\frac{1}{M_t}\\sum_{m=1}^{M_t}\\frac{(-1)^{w_{t,m}}(B_{t,m}-N_{t,m})}{N_{t,m}}.

    Args:
        base_result (dict): A dictionary of scores of all metrics of all tasks.
        new_result (dict): The same structure with ``base_result``.
        weight (dict): The same structure with ``base_result`` while each element is binary integer representing whether higher or lower score is better.

    Returns:
        float: The improvement between ``new_result`` and ``base_result``.

    Examples::

        base_result = {'A': [96, 98], 'B': [0.2]}
        new_result = {'A': [93, 99], 'B': [0.5]}
        weight = {'A': [1, 0], 'B': [1]}

        print(count_improvement(base_result, new_result, weight))
    """
    improvement = 0
    count = 0
    for task in list(base_result.keys()):
        improvement += ((-1) ** np.array(weight[task]) * (np.array(base_result[task]) - np.array(new_result[task])) / np.array(base_result[task])).mean()
        count += 1
    return improvement / count


def count_parameters(model):
    """Calculate the number of parameters for a model.

    Args:
        model (torch.nn.Module): A neural network module.
    """
    trainable_params = 0
    non_trainable_params = 0
    for p in model.parameters():
        if p.requires_grad:
            trainable_params += p.numel()
        else:
            non_trainable_params += p.numel()
    None
    None
    None
    None


class Trainer(nn.Module):
    """A Multi-Task Learning Trainer.

    This is a unified and extensible training framework for multi-task learning. 

    Args:
        task_dict (dict): A dictionary of name-information pairs of type (:class:`str`, :class:`dict`). \\
                            The sub-dictionary for each task has four entries whose keywords are named **metrics**, \\
                            **metrics_fn**, **loss_fn**, **weight** and each of them corresponds to a :class:`list`.
                            The list of **metrics** has ``m`` strings, repersenting the name of ``m`` metrics \\
                            for this task. The list of **metrics_fn** has two elements, i.e., the updating and score \\
                            functions, meaning how to update thoes objectives in the training process and obtain the final \\
                            scores, respectively. The list of **loss_fn** has ``m`` loss functions corresponding to each \\
                            metric. The list of **weight** has ``m`` binary integers corresponding to each \\
                            metric, where ``1`` means the higher the score is, the better the performance, \\
                            ``0`` means the opposite.                           
        weighting (class): A weighting strategy class based on :class:`LibMTL.weighting.abstract_weighting.AbsWeighting`.
        architecture (class): An architecture class based on :class:`LibMTL.architecture.abstract_arch.AbsArchitecture`.
        encoder_class (class): A neural network class.
        decoders (dict): A dictionary of name-decoder pairs of type (:class:`str`, :class:`torch.nn.Module`).
        rep_grad (bool): If ``True``, the gradient of the representation for each task can be computed.
        multi_input (bool): Is ``True`` if each task has its own input data, otherwise is ``False``. 
        optim_param (dict): A dictionary of configurations for the optimizier.
        scheduler_param (dict): A dictionary of configurations for learning rate scheduler. \\
                                 Set it to ``None`` if you do not use a learning rate scheduler.
        kwargs (dict): A dictionary of hyperparameters of weighting and architecture methods.

    .. note::
            It is recommended to use :func:`LibMTL.config.prepare_args` to return the dictionaries of ``optim_param``, \\
            ``scheduler_param``, and ``kwargs``.

    Examples::
        
        import torch.nn as nn
        from LibMTL import Trainer
        from LibMTL.loss import CE_loss_fn
        from LibMTL.metrics import acc_update_fun, acc_score_fun
        from LibMTL.weighting import EW
        from LibMTL.architecture import HPS
        from LibMTL.model import ResNet18
        from LibMTL.config import prepare_args

        task_dict = {'A': {'metrics': ['Acc'],
                           'metrics_fn': [acc_update_fun, acc_score_fun],
                           'loss_fn': [CE_loss_fn],
                           'weight': [1]}}
        
        decoders = {'A': nn.Linear(512, 31)}
        
        # You can use command-line arguments and return configurations by ``prepare_args``.
        # kwargs, optim_param, scheduler_param = prepare_args(params)
        optim_param = {'optim': 'adam', 'lr': 1e-3, 'weight_decay': 1e-4}
        scheduler_param = {'scheduler': 'step'}
        kwargs = {'weight_args': {}, 'arch_args': {}}

        trainer = Trainer(task_dict=task_dict,
                          weighting=EW,
                          architecture=HPS,
                          encoder_class=ResNet18,
                          decoders=decoders,
                          rep_grad=False,
                          multi_input=False,
                          optim_param=optim_param,
                          scheduler_param=scheduler_param,
                          **kwargs)

    """

    def __init__(self, task_dict, weighting, architecture, encoder_class, decoders, rep_grad, multi_input, optim_param, scheduler_param, **kwargs):
        super(Trainer, self).__init__()
        self.device = torch.device('cuda:0')
        self.kwargs = kwargs
        self.task_dict = task_dict
        self.task_num = len(task_dict)
        self.task_name = list(task_dict.keys())
        self.rep_grad = rep_grad
        self.multi_input = multi_input
        self._prepare_model(weighting, architecture, encoder_class, decoders)
        self._prepare_optimizer(optim_param, scheduler_param)
        self.meter = _PerformanceMeter(self.task_dict, self.multi_input)

    def _prepare_model(self, weighting, architecture, encoder_class, decoders):


        class MTLmodel(architecture, weighting):

            def __init__(self, task_name, encoder_class, decoders, rep_grad, multi_input, device, kwargs):
                super(MTLmodel, self).__init__(task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs)
                self.init_param()
        self.model = MTLmodel(task_name=self.task_name, encoder_class=encoder_class, decoders=decoders, rep_grad=self.rep_grad, multi_input=self.multi_input, device=self.device, kwargs=self.kwargs['arch_args'])
        count_parameters(self.model)

    def _prepare_optimizer(self, optim_param, scheduler_param):
        optim_dict = {'sgd': torch.optim.SGD, 'adam': torch.optim.Adam, 'adagrad': torch.optim.Adagrad, 'rmsprop': torch.optim.RMSprop}
        scheduler_dict = {'exp': torch.optim.lr_scheduler.ExponentialLR, 'step': torch.optim.lr_scheduler.StepLR, 'cos': torch.optim.lr_scheduler.CosineAnnealingLR}
        optim_arg = {k: v for k, v in optim_param.items() if k != 'optim'}
        self.optimizer = optim_dict[optim_param['optim']](self.model.parameters(), **optim_arg)
        if scheduler_param is not None:
            scheduler_arg = {k: v for k, v in scheduler_param.items() if k != 'scheduler'}
            self.scheduler = scheduler_dict[scheduler_param['scheduler']](self.optimizer, **scheduler_arg)
        else:
            self.scheduler = None

    def _process_data(self, loader):
        try:
            data, label = loader[1].next()
        except:
            loader[1] = iter(loader[0])
            data, label = loader[1].next()
        data = data
        if not self.multi_input:
            for task in self.task_name:
                label[task] = label[task]
        else:
            label = label
        return data, label

    def process_preds(self, preds, task_name=None):
        """The processing of prediction for each task. 

        - The default is no processing. If necessary, you can rewrite this function. 
        - If ``multi_input`` is ``True``, ``task_name`` is valid and ``preds`` with type :class:`torch.Tensor` is the prediction of this task.
        - otherwise, ``task_name`` is invalid and ``preds`` is a :class:`dict` of name-prediction pairs of all tasks.

        Args:
            preds (dict or torch.Tensor): The prediction of ``task_name`` or all tasks.
            task_name (str): The string of task name.
        """
        return preds

    def _compute_loss(self, preds, gts, task_name=None):
        if not self.multi_input:
            train_losses = torch.zeros(self.task_num)
            for tn, task in enumerate(self.task_name):
                train_losses[tn] = self.meter.losses[task]._update_loss(preds[task], gts[task])
        else:
            train_losses = self.meter.losses[task_name]._update_loss(preds, gts)
        return train_losses

    def _prepare_dataloaders(self, dataloaders):
        if not self.multi_input:
            loader = [dataloaders, iter(dataloaders)]
            return loader, len(dataloaders)
        else:
            loader = {}
            batch_num = []
            for task in self.task_name:
                loader[task] = [dataloaders[task], iter(dataloaders[task])]
                batch_num.append(len(dataloaders[task]))
            return loader, batch_num

    def train(self, train_dataloaders, test_dataloaders, epochs, val_dataloaders=None, return_weight=False):
        """The training process of multi-task learning.

        Args:
            train_dataloaders (dict or torch.utils.data.DataLoader): The dataloaders used for training. \\
                            If ``multi_input`` is ``True``, it is a dictionary of name-dataloader pairs. \\
                            Otherwise, it is a single dataloader which returns data and a dictionary \\
                            of name-label pairs in each iteration.

            test_dataloaders (dict or torch.utils.data.DataLoader): The dataloaders used for the validation or testing. \\
                            The same structure with ``train_dataloaders``.
            epochs (int): The total training epochs.
            return_weight (bool): if ``True``, the loss weights will be returned.
        """
        train_loader, train_batch = self._prepare_dataloaders(train_dataloaders)
        train_batch = max(train_batch) if self.multi_input else train_batch
        self.batch_weight = np.zeros([self.task_num, epochs, train_batch])
        self.model.train_loss_buffer = np.zeros([self.task_num, epochs])
        self.model.epochs = epochs
        for epoch in range(epochs):
            self.model.epoch = epoch
            self.model.train()
            self.meter.record_time('begin')
            for batch_index in range(train_batch):
                if not self.multi_input:
                    train_inputs, train_gts = self._process_data(train_loader)
                    train_preds = self.model(train_inputs)
                    train_preds = self.process_preds(train_preds)
                    train_losses = self._compute_loss(train_preds, train_gts)
                    self.meter.update(train_preds, train_gts)
                else:
                    train_losses = torch.zeros(self.task_num)
                    for tn, task in enumerate(self.task_name):
                        train_input, train_gt = self._process_data(train_loader[task])
                        train_pred = self.model(train_input, task)
                        train_pred = train_pred[task]
                        train_pred = self.process_preds(train_pred, task)
                        train_losses[tn] = self._compute_loss(train_pred, train_gt, task)
                        self.meter.update(train_pred, train_gt, task)
                self.optimizer.zero_grad()
                w = self.model.backward(train_losses, **self.kwargs['weight_args'])
                if w is not None:
                    self.batch_weight[:, epoch, batch_index] = w
                self.optimizer.step()
            self.meter.record_time('end')
            self.meter.get_score()
            self.model.train_loss_buffer[:, epoch] = self.meter.loss_item
            self.meter.display(epoch=epoch, mode='train')
            self.meter.reinit()
            if val_dataloaders is not None:
                self.meter.has_val = True
                self.test(val_dataloaders, epoch, mode='val')
            self.test(test_dataloaders, epoch, mode='test')
            if self.scheduler is not None:
                self.scheduler.step()
        self.meter.display_best_result()
        if return_weight:
            return self.batch_weight

    def test(self, test_dataloaders, epoch=None, mode='test'):
        """The test process of multi-task learning.

        Args:
            test_dataloaders (dict or torch.utils.data.DataLoader): If ``multi_input`` is ``True``, \\
                            it is a dictionary of name-dataloader pairs. Otherwise, it is a single \\
                            dataloader which returns data and a dictionary of name-label pairs in each iteration.
            epoch (int, default=None): The current epoch. 
        """
        test_loader, test_batch = self._prepare_dataloaders(test_dataloaders)
        self.model.eval()
        self.meter.record_time('begin')
        with torch.no_grad():
            if not self.multi_input:
                for batch_index in range(test_batch):
                    test_inputs, test_gts = self._process_data(test_loader)
                    test_preds = self.model(test_inputs)
                    test_preds = self.process_preds(test_preds)
                    test_losses = self._compute_loss(test_preds, test_gts)
                    self.meter.update(test_preds, test_gts)
            else:
                for tn, task in enumerate(self.task_name):
                    for batch_index in range(test_batch[tn]):
                        test_input, test_gt = self._process_data(test_loader[task])
                        test_pred = self.model(test_input, task)
                        test_pred = test_pred[task]
                        test_pred = self.process_preds(test_pred)
                        test_loss = self._compute_loss(test_pred, test_gt, task)
                        self.meter.update(test_pred, test_gt, task)
        self.meter.record_time('end')
        self.meter.get_score()
        self.meter.display(epoch=epoch, mode=mode)
        self.meter.reinit()


class AbsWeighting(nn.Module):
    """An abstract class for weighting strategies.
    """

    def __init__(self):
        super(AbsWeighting, self).__init__()

    def init_param(self):
        """Define and initialize some trainable parameters required by specific weighting methods. 
        """
        pass

    def _compute_grad_dim(self):
        self.grad_index = []
        for param in self.get_share_params():
            self.grad_index.append(param.data.numel())
        self.grad_dim = sum(self.grad_index)

    def _grad2vec(self):
        grad = torch.zeros(self.grad_dim)
        count = 0
        for param in self.get_share_params():
            if param.grad is not None:
                beg = 0 if count == 0 else sum(self.grad_index[:count])
                end = sum(self.grad_index[:count + 1])
                grad[beg:end] = param.grad.data.view(-1)
            count += 1
        return grad

    def _compute_grad(self, losses, mode, rep_grad=False):
        """
        mode: backward, autograd
        """
        if not rep_grad:
            grads = torch.zeros(self.task_num, self.grad_dim)
            for tn in range(self.task_num):
                if mode == 'backward':
                    losses[tn].backward(retain_graph=True) if tn + 1 != self.task_num else losses[tn].backward()
                    grads[tn] = self._grad2vec()
                elif mode == 'autograd':
                    grad = list(torch.autograd.grad(losses[tn], self.get_share_params(), retain_graph=True))
                    grads[tn] = torch.cat([g.view(-1) for g in grad])
                else:
                    raise ValueError('No support {} mode for gradient computation')
                self.zero_grad_share_params()
        else:
            if not isinstance(self.rep, dict):
                grads = torch.zeros(self.task_num, *self.rep.size())
            else:
                grads = [torch.zeros(*self.rep[task].size()) for task in self.task_name]
            for tn, task in enumerate(self.task_name):
                if mode == 'backward':
                    losses[tn].backward(retain_graph=True) if tn + 1 != self.task_num else losses[tn].backward()
                    grads[tn] = self.rep_tasks[task].grad.data.clone()
        return grads

    def _reset_grad(self, new_grads):
        count = 0
        for param in self.get_share_params():
            if param.grad is not None:
                beg = 0 if count == 0 else sum(self.grad_index[:count])
                end = sum(self.grad_index[:count + 1])
                param.grad.data = new_grads[beg:end].contiguous().view(param.data.size()).data.clone()
            count += 1

    def _get_grads(self, losses, mode='backward'):
        """This function is used to return the gradients of representations or shared parameters.

        If ``rep_grad`` is ``True``, it returns a list with two elements. The first element is \\
        the gradients of the representations with the size of [task_num, batch_size, rep_size]. \\
        The second element is the resized gradients with size of [task_num, -1], which means \\
        the gradient of each task is resized as a vector.

        If ``rep_grad`` is ``False``, it returns the gradients of the shared parameters with size \\
        of [task_num, -1], which means the gradient of each task is resized as a vector.
        """
        if self.rep_grad:
            per_grads = self._compute_grad(losses, mode, rep_grad=True)
            if not isinstance(self.rep, dict):
                grads = per_grads.reshape(self.task_num, self.rep.size()[0], -1).sum(1)
            else:
                try:
                    grads = torch.stack(per_grads).sum(1).view(self.task_num, -1)
                except:
                    raise ValueError('The representation dimensions of different tasks must be consistent')
            return [per_grads, grads]
        else:
            self._compute_grad_dim()
            grads = self._compute_grad(losses, mode)
            return grads

    def _backward_new_grads(self, batch_weight, per_grads=None, grads=None):
        """This function is used to reset the gradients and make a backward.

        Args:
            batch_weight (torch.Tensor): A tensor with size of [task_num].
            per_grad (torch.Tensor): It is needed if ``rep_grad`` is True. The gradients of the representations.
            grads (torch.Tensor): It is needed if ``rep_grad`` is False. The gradients of the shared parameters. 
        """
        if self.rep_grad:
            if not isinstance(self.rep, dict):
                transformed_grad = torch.einsum('i, i... -> ...', batch_weight, per_grads)
                self.rep.backward(transformed_grad)
            else:
                for tn, task in enumerate(self.task_name):
                    rg = True if tn + 1 != self.task_num else False
                    self.rep[task].backward(batch_weight[tn] * per_grads[tn], retain_graph=rg)
        else:
            new_grads = torch.einsum('i, i... -> ...', batch_weight, grads)
            self._reset_grad(new_grads)

    @property
    def backward(self, losses, **kwargs):
        """
        Args:
            losses (list): A list of losses of each task.
            kwargs (dict): A dictionary of hyperparameters of weighting methods.
        """
        pass


class ASPPConv(nn.Sequential):

    def __init__(self, in_channels, out_channels, dilation):
        modules = [nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU()]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):

    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):

    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU()))
        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(nn.Conv2d(5 * out_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(), nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class DeepLabHead(nn.Sequential):

    def __init__(self, in_channels, num_classes):
        super(DeepLabHead, self).__init__(ASPP(in_channels, [12, 24, 36]), nn.Conv2d(256, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(), nn.Conv2d(256, num_classes, 1))


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ASPP,
     lambda: ([], {'in_channels': 4, 'atrous_rates': [4, 4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ASPPConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'dilation': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ASPPPooling,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DeepLabHead,
     lambda: ([], {'in_channels': 4, 'num_classes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_median_research_group_LibMTL(_paritybench_base):
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

