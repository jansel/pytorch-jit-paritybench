import sys
_module = sys.modules[__name__]
del sys
lib = _module
Dataset_Base = _module
datasets = _module
dataset_naiReg = _module
eval = _module
eval_aet_multilevel = _module
regEulerNet = _module
trainval_workdir = _module
dataengine_v2 = _module
eval_official = _module
eval_sn = _module
helper = _module
model = _module
modelSE = _module
modelSEv2 = _module
regNormalNet = _module
trainval_workdir = _module
test = _module
unet_model = _module
unet_parts = _module
Dataset_Base = _module
dataset_regQuatNet = _module
db_type = _module
design = _module
eval_quat_multilevel = _module
helper = _module
regQuatNet = _module
trainval_workdir = _module
Pascal3D = _module
anno_db_v2 = _module
util_v2 = _module
data_info = _module
build_imdb = _module
fix_bad_bbox_anno = _module
gen_gt_box = _module
mat2py = _module
prepare_anno_db = _module
prepare_r4cnn_syn_data = _module
basic = _module
common = _module
util = _module
yaml_netconf = _module
lmdb_util = _module
imagedata_lmdb = _module
load_nested_mat_struct = _module
numpy_db = _module
formater = _module
numpy_db_py3 = _module
ordered_easydict = _module
pytorch_util = _module
libtrain = _module
init_torch = _module
reducer = _module
reducer_v2 = _module
tools = _module
netutil = _module
common_v2 = _module
trunk_alexnet_bvlc = _module
trunk_alexnet_pytorch = _module
trunk_inception = _module
trunk_resnet = _module
trunk_vgg = _module
torch_3rd_funcs = _module
torch_3rd_layers = _module
torch_v4_feature = _module
txt_table_v1 = _module

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


import numpy as np


from math import ceil


from math import floor


from math import pi


import torch


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from torchvision import transforms


from torchvision import utils


import torch.nn as nn


import torch.utils.model_zoo as model_zoo


from torch.autograd import Variable


from collections import OrderedDict as odict


from itertools import product


import torch.optim


from torch.utils.data.sampler import RandomSampler


from torch.utils.data.sampler import BatchSampler


import time


from time import gmtime


from time import strftime


import math


import re


from string import Template


import torch.nn.functional as F


from collections import OrderedDict


import torchvision


from torch.nn.modules.module import Module


from torch.nn.modules.utils import _single


from torch.nn.modules.utils import _pair


from torch.nn.modules.utils import _triple


from torch.nn.parameter import Parameter


from torch.nn.functional import *


def init_weights_by_filling(nn_module_or_seq, gaussian_std=0.01, kaiming_normal=True, silent=False):
    """ Note: gaussian_std is fully connected layer (nn.Linear) only.
        For nn.Conv2d:
           If kaiming_normal is enable, nn.Conv2d is initialized by kaiming_normal.
           Otherwise, initialized based on kernel size.
    """
    if not silent:
        None
    for name, m in nn_module_or_seq.named_modules():
        if isinstance(m, nn.Conv2d):
            if kaiming_normal:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            else:
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            if m.weight is not None:
                m.weight.data.fill_(1)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(mean=0, std=gaussian_std)
            m.bias.data.zero_()
    return nn_module_or_seq


def get_weights_from_caffesnapeshot(proto_file, model_file):
    from collections import OrderedDict
    caffe.set_mode_cpu()
    net = caffe.Net(proto_file, model_file, caffe.TEST)
    model_dict = OrderedDict()
    for layer_name, param in net.params.iteritems():
        learnable_weight = []
        if len(param) == 2:
            learnable_weight.append(param[0].data.copy())
            learnable_weight.append(param[1].data.copy())
        elif len(param) == 1:
            learnable_weight.append(param[0].data.copy())
        else:
            raise NotImplementedError
        model_dict[layer_name] = learnable_weight
    return model_dict


def _copy_weights_from_caffemodel(own_state, pretrained_type='alexnet', ignore_missing_dst=False, src2dsts=dict(conv1='conv1', conv2='conv2', conv3='conv3', conv4='conv4', conv5='conv5', fc6='fc6', fc7='fc7')):
    """ src2dsts = dict(conv1='conv1', conv2='conv2', conv3='conv3', conv4='conv4', conv5='conv5')
    Or in list:
        src2dsts = dict(conv1=['conv1'], conv2=['conv2'], conv3=['conv3'], conv4=['conv4'], conv5=['conv5'])
    """
    None
    None
    None
    if isinstance(pretrained_type, tuple):
        proto_file, model_file = pretrained_type
        pretrained_weights = get_weights_from_caffesnapeshot(proto_file, model_file)
    else:
        None
        pretrained_weights = pickle.load(open(cfg.caffemodel[pretrained_type].pkl, 'rb'), encoding='bytes')
        None
    not_copied = list(own_state.keys())
    src_list = sorted(src2dsts.keys())
    for src in src_list:
        dsts = src2dsts[src]
        if not isinstance(dsts, list):
            dsts = [dsts]
        w, b = pretrained_weights[src.encode('utf-8')]
        w = torch.from_numpy(w)
        b = torch.from_numpy(b)
        for dst in dsts:
            if ignore_missing_dst and dst not in own_state.keys():
                None
                continue
            None
            dst_w_name = '%s.weight' % dst
            dst_b_name = '%s.bias' % dst
            assert dst_w_name in own_state.keys(), '[Error] %s not in %s' % (dst_w_name, own_state.keys())
            assert dst_b_name in own_state.keys(), '[Error] %s not in %s' % (dst_b_name, own_state.keys())
            assert own_state[dst_w_name].shape == w.shape, '[%s] w: dest. %s != src. %s' % (dst_w_name, own_state[dst_w_name].shape, w.shape)
            own_state[dst_w_name].copy_(w)
            not_copied.remove(dst_w_name)
            assert own_state[dst_b_name].shape == b.shape, '[%s] w: dest. %s != src. %s' % (dst_b_name, own_state[dst_b_name].shape, b.shape)
            own_state[dst_b_name].copy_(b)
            not_copied.remove(dst_b_name)
    for name in not_copied:
        if name.endswith('.weight'):
            own_state[name].normal_(mean=0.0, std=0.005)
            None
        elif name.endswith('.bias'):
            own_state[name].fill_(0)
            None
        else:
            None
            raise NotImplementedError
    None


def _copy_weights_from_torchmodel(own_state, pretrained_type='alexnet', strict=True, src2dsts=None):
    from torch.nn.parameter import Parameter
    None
    None
    None
    src_state = model_zoo.load_url(cfg.torchmodel[pretrained_type].model_url)
    not_copied = list(own_state.keys())
    if src2dsts is not None:
        for src, dsts in src2dsts.items():
            if not isinstance(dsts, list):
                dsts = [dsts]
            for dst in dsts:
                if dst in own_state.keys():
                    own_state[dst].copy_(src_state[src])
                    not_copied.remove(dst)
                    None
                else:
                    dst_w_name, src_w_name = '%s.weight' % dst, '%s.weight' % src
                    dst_b_name, src_b_name = '%s.bias' % dst, '%s.bias' % src
                    if (dst_w_name not in own_state.keys() or dst_b_name not in own_state.keys()) and not strict:
                        None
                        continue
                    None
                    assert own_state[dst_w_name].shape == src_state[src_w_name].shape, '[%s] w: dest. %s != src. %s' % (dst_w_name, own_state[dst_w_name].shape, src_state[src_w_name].shape)
                    own_state[dst_w_name].copy_(src_state[src_w_name])
                    not_copied.remove(dst_w_name)
                    assert own_state[dst_b_name].shape == src_state[src_b_name].shape, '[%s] w: dest. %s != src. %s' % (dst_b_name, own_state[dst_b_name].shape, src_state[src_b_name].shape)
                    own_state[dst_b_name].copy_(src_state[src_b_name])
                    not_copied.remove(dst_b_name)
    else:
        for name, param in src_state.items():
            if name in own_state:
                if isinstance(param, Parameter):
                    param = param.data
                try:
                    None
                    own_state[name].copy_(param)
                    not_copied.remove(name)
                except Exception:
                    raise RuntimeError('While copying the parameter named {}, whose dimensions in the model are {} and whose dimensions in the checkpoint are {}.'.format(name, own_state[name].size(), param.size()))
            elif strict:
                raise KeyError('unexpected key "{}" in state_dict'.format(name))
            else:
                None
    for name in not_copied:
        if name.endswith('.weight'):
            own_state[name].normal_(mean=0.0, std=0.005)
            None
        elif name.endswith('.bias'):
            own_state[name].fill_(0)
            None
        elif name.endswith('.running_mean') or name.endswith('.running_var') or name.endswith('num_batches_tracked'):
            None
        else:
            None
            raise NotImplementedError
    if strict:
        missing = set(own_state.keys()) - set(src_state.keys())
        if len(missing) > 0:
            raise KeyError('missing keys in state_dict: "{}"'.format(missing))
    None


def copy_weights(own_state, pretrained_type, **kwargs):
    """ Usage example:
            copy_weights(own_state, 'torchmodel.alexnet',  strict=False)
            copy_weights(own_state, 'caffemodel.alexnet',  ignore_missing_dst=True, src2dsts={})
            copy_weights(own_state, '')
    """
    if isinstance(pretrained_type, str):
        if pretrained_type.startswith('torchmodel.'):
            pretrained_type = pretrained_type[11:]
            copy_func = _copy_weights_from_torchmodel
        elif pretrained_type.startswith('caffemodel.'):
            pretrained_type = pretrained_type[11:]
            copy_func = _copy_weights_from_caffemodel
        else:
            None
            raise NotImplementedError
    elif isinstance(pretrained_type, tuple):
        copy_func = _copy_weights_from_caffemodel
    else:
        None
        raise NotImplementedError
    copy_func(own_state, pretrained_type, **kwargs)


class AlexNet_Trunk(nn.Module):

    def __init__(self):
        super(AlexNet_Trunk, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2), nn.Conv2d(64, 192, kernel_size=5, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2), nn.Conv2d(192, 384, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2))
        self.classifier = nn.Sequential(nn.Dropout(), nn.Linear(256 * 6 * 6, 4096), nn.ReLU(inplace=True), nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(inplace=True))

    def forward_Conv(self, x):
        return self.features(x)

    def forward_Fc(self, x):
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def forward(self, x):
        """ x is input image data.
            x is of shape:  (batchsize, 3, 227, 227)  """
        x = self.features(x)
        batchsize = x.size(0)
        x = x.view(batchsize, 256 * 6 * 6)
        x = self.classifier(x)
        return x

    @staticmethod
    def init_weights(self):
        copy_weights(self.state_dict(), 'torchmodel.alexnet', strict=False)
        return self


def _nn_xNorm(num_channels, xNorm='BN', **kwargs):
    """ E.g.
         Fixed BN:   xNorm='BN', affine=False
    """
    if xNorm == 'BN':
        return nn.BatchNorm2d(num_channels, **kwargs)
    elif xNorm == 'GN':
        return nn.GroupNorm(num_groups=32, num_channels=num_channels, **kwargs)
    elif xNorm == 'IN':
        return nn.InstanceNorm2d(num_channels, **kwargs)
    elif xNorm == 'LN':
        raise NotImplementedError


model_urls = {'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth', 'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth', 'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth', 'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth', 'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class nnAdd(nn.Module):

    def forward(self, x0, x1):
        return x0 + x1


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, xNorm='BN', xNorm_kwargs={}):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = _nn_xNorm(planes, xNorm=xNorm, **xNorm_kwargs)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = _nn_xNorm(planes, xNorm=xNorm, **xNorm_kwargs)
        self.downsample = downsample
        self.stride = stride
        self.merge = nnAdd()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.merge(out, residual)
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, xNorm='BN', xNorm_kwargs={}):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = _nn_xNorm(planes, xNorm=xNorm, **xNorm_kwargs)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = _nn_xNorm(planes, xNorm=xNorm, **xNorm_kwargs)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = _nn_xNorm(planes * 4, xNorm=xNorm, **xNorm_kwargs)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.merge = nnAdd()

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
        out = self.merge(out, residual)
        out = self.relu(out)
        return out


type2blockFunc_layerCfgs = dict(resnet18=(BasicBlock, [2, 2, 2, 2]), resnet34=(BasicBlock, [3, 4, 6, 3]), resnet50=(Bottleneck, [3, 4, 6, 3]), resnet101=(Bottleneck, [3, 4, 23, 3]), resnet152=(Bottleneck, [3, 8, 36, 3]))


class _ResNet_Trunk(nn.Module):

    def __init__(self, res_type='resnet101', pretrained='torchmodel', start=None, end=None, xNorm='BN', xNorm_kwargs={}):
        self.inplanes = 64
        super(_ResNet_Trunk, self).__init__()
        self.net_arch = res_type
        blockFunc, layerCfgs = type2blockFunc_layerCfgs[res_type]
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = _nn_xNorm(64, xNorm=xNorm, **xNorm_kwargs)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(blockFunc, 64, layerCfgs[0], stride=1, xNorm='BN', xNorm_kwargs={})
        self.layer2 = self._make_layer(blockFunc, 128, layerCfgs[1], stride=2, xNorm='BN', xNorm_kwargs={})
        self.layer3 = self._make_layer(blockFunc, 256, layerCfgs[2], stride=2, xNorm='BN', xNorm_kwargs={})
        self.layer4 = self._make_layer(blockFunc, 512, layerCfgs[3], stride=2, xNorm='BN', xNorm_kwargs={})
        self.pool5 = nn.AvgPool2d(7, stride=1)
        self._layer_names = [name for name, module in self.named_children()]
        if pretrained == True:
            pretrained = 'torchmodel'
        if pretrained == False:
            pretrained = None
        assert pretrained in [None, 'caffemodel', 'torchmodel'], 'Unknown pretrained: %s' % pretrained
        None
        self.init_weights(pretrained=pretrained)
        self.truck_seq = self.sub_seq(start=start, end=end)

    def sub_seq(self, start=None, end=None):
        assert start is None or start in self._layer_names, '[Error] %s is not in %s' % (start, self._layer_names)
        assert end is None or end in self._layer_names, '[Error] %s is not in %s' % (end, self._layer_names)
        start_ind = self._layer_names.index(start) if start is not None else 0
        end_ind = self._layer_names.index(end) if end is not None else len(self._layer_names) - 1
        assert start_ind <= end_ind
        self.selected_layer_name = self._layer_names[start_ind:end_ind + 1]
        None
        _seq = nn.Sequential(*[self.__getattr__(x) for x in self.selected_layer_name])
        return _seq

    def _make_layer(self, block, planes, blocks, stride, xNorm, xNorm_kwargs):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), _nn_xNorm(planes * block.expansion, xNorm=xNorm, **xNorm_kwargs))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, xNorm=xNorm, xNorm_kwargs=xNorm_kwargs))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, xNorm=xNorm, xNorm_kwargs=xNorm_kwargs))
        return nn.Sequential(*layers)

    def forward(self, x):
        """ x is input image data.
            x is of shape:  (batchsize, 3, 224, 224)  """
        assert x.size()[1:] == (3, 224, 224), 'resnet need (3,224,224) input data, whereas %s is received.' % str(tuple(x.size()[1:]))
        if self.selected_layer_name[-1] == 'pool5':
            return self.truck_seq(x).view(x.size(0), -1)
        else:
            return self.truck_seq(x)

    def init_weights(self, pretrained='caffemodel'):
        """ Two ways to init weights:
            1) by copying pretrained weights.
            2) by filling empirical  weights. (e.g. gaussian, xavier, uniform, constant, bilinear).
        """
        if pretrained is None:
            None
            init_weights_by_filling(self, silent=False)
        elif pretrained == 'caffemodel':
            model_path = pretrained_cfg.caffemodel.resnet101.model
            None
            state_dict = torch.load(model_path)
            self.load_state_dict({k: v for k, v in state_dict.items() if k in self.state_dict()})
        elif pretrained == 'torchmodel':
            None
            state_dict = model_zoo.load_url(model_urls[self.net_arch], progress=True)
            self.load_state_dict(state_dict, strict=False)
        else:
            raise NotImplementedError
        return self


class ResNet101_Trunk(_ResNet_Trunk):

    def __init__(self, **kwargs):
        _ResNet_Trunk.__init__(self, res_type='resnet101', **kwargs)


class ResNet50_Trunk(_ResNet_Trunk):

    def __init__(self, **kwargs):
        _ResNet_Trunk.__init__(self, res_type='resnet50', **kwargs)


def get_caffeSrc2Dst(net_arch='vgg16'):
    if net_arch == 'vgg16':
        return odict(conv1_1='features.0', conv1_2='features.2', conv2_1='features.5', conv2_2='features.7', conv3_1='features.10', conv3_2='features.12', conv3_3='features.14', conv4_1='features.17', conv4_2='features.19', conv4_3='features.21', conv5_1='features.24', conv5_2='features.26', conv5_3='features.28', fc6='classifier.0', fc7='classifier.3')
    elif net_arch == 'vgg19':
        return odict(conv1_1='features.0', conv1_2='features.2', conv2_1='features.5', conv2_2='features.7', conv3_1='features.10', conv3_2='features.12', conv3_3='features.14', conv3_4='features.16', conv4_1='features.19', conv4_2='features.21', conv4_3='features.23', conv4_4='features.25', conv5_1='features.28', conv5_2='features.30', conv5_3='features.32', conv5_4='features.34', fc6='classifier.0', fc7='classifier.3')
    elif net_arch in ['vgg16_bn', 'vgg19_bn']:
        None
        raise NotImplementedError
    elif net_arch == 'vggm':
        return odict(conv1='features.0', conv2='features.4', conv3='features.8', conv4='features.10', conv5='features.12', fc6='classifier.0', fc7='classifier.3')
    else:
        raise NotImplementedError


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class LocalResponseNorm(Module):

    def __init__(self, size, alpha=0.0001, beta=0.75, k=1):
        """Applies local response normalization over an input signal composed
        of several input planes, where channels occupy the second dimension.
        Applies normalization across channels.

        .. math::

            `b_{c} = a_{c}\\left(k + \\frac{\\alpha}{n}
            \\sum_{c'=\\max(0, c-n/2)}^{\\min(N-1,c+n/2)}a_{c'}^2\\right)^{-\\beta}`

        Args:
            size: amount of neighbouring channels used for normalization
            alpha: multiplicative factor. Default: 0.0001
            beta: exponent. Default: 0.75
            k: additive factor. Default: 1

        Shape:
            - Input: :math:`(N, C, ...)`
            - Output: :math:`(N, C, ...)` (same shape as input)
        Examples::
            >>> lrn = nn.LocalResponseNorm(2)
            >>> signal_2d = autograd.Variable(torch.randn(32, 5, 24, 24))
            >>> signal_4d = autograd.Variable(torch.randn(16, 5, 7, 7, 7, 7))
            >>> output_2d = lrn(signal_2d)
            >>> output_4d = lrn(signal_4d)
        """
        super(LocalResponseNorm, self).__init__()
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, input):
        return local_response_norm(input, self.size, self.alpha, self.beta, self.k)

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.size) + ', alpha=' + str(self.alpha) + ', beta=' + str(self.beta) + ', k=' + str(self.k) + ')'


def make_layers_vggm():
    return nn.Sequential(nn.Conv2d(3, 96, (7, 7), (2, 2)), nn.ReLU(inplace=True), LocalResponseNorm(5, 0.0005, 0.75, 2), nn.MaxPool2d((3, 3), (2, 2), (0, 0), ceil_mode=True), nn.Conv2d(96, 256, (5, 5), (2, 2), (1, 1)), nn.ReLU(inplace=True), LocalResponseNorm(5, 0.0005, 0.75, 2), nn.MaxPool2d((3, 3), (2, 2), (0, 0), ceil_mode=True), nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1)), nn.ReLU(inplace=True), nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)), nn.ReLU(inplace=True), nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)), nn.ReLU(inplace=True), nn.MaxPool2d((3, 3), (2, 2), (0, 0), ceil_mode=True))


class _VGG_Trunk(nn.Module):

    def __init__(self, vgg_type='vgg16', init_weights=True):
        super(_VGG_Trunk, self).__init__()
        self.net_arch = vgg_type
        if vgg_type == 'vggm':
            self.features = make_layers_vggm()
            self.classifier = nn.Sequential(nn.Linear(512 * 6 * 6, 4096), nn.ReLU(True), nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout())
        else:
            self.features = make_layers(type2cfg[vgg_type])
            self.classifier = nn.Sequential(nn.Linear(512 * 7 * 7, 4096), nn.ReLU(True), nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout())
        if init_weights == True:
            assert self.net_arch in ['vgg16', 'vgg19']
            self.init_weights(pretrained='caffemodel')
        elif isinstance(init_weights, str) or init_weights is None:
            self.init_weights(pretrained=init_weights)
        else:
            raise NotImplementedError
    """
    def sub_seq(self, start=None, end=None):   # 'conv1', 'pool5'
        # select sub-sequence from trunk (by creating a shot cut).
        assert start is None or start in self._layer_names, '[Error] %s is not in %s' % (start, self._layer_names)
        assert end   is None or end   in self._layer_names, '[Error] %s is not in %s' % (end  , self._layer_names)
        start_ind = self._layer_names.index(start) if (start is not None) else 0
        end_ind   = self._layer_names.index(end)   if (end   is not None) else len(self._layer_names)-1
        assert start_ind<=end_ind
        self.selected_layer_name = self._layer_names[start_ind:end_ind+1]
        print("Selected sub-sequence: %s" % self.selected_layer_name)
        _seq = nn.Sequential(*[self.__getattr__(x) for x in self.selected_layer_name])
        return _seq     # (self.conv1, self.bn1, self.relu, self.maxpool, self.layer1,self.layer2,self.layer3 )
    """

    def forward(self, x):
        """ x is input image data.
            x is of shape:  (batchsize, 3, 224, 224)  """
        x = self.features(x)
        batchsize = x.size(0)
        x = x.view(batchsize, -1)
        x = self.classifier(x)
        return x

    def init_weights(self, pretrained='caffemodel'):
        """ Two ways to init weights:
            1) by copying pretrained weights.
            2) by filling empirical  weights. (e.g. gaussian, xavier, uniform, constant, bilinear).
        """
        if pretrained is None:
            None
            init_weights_by_filling(self)
        elif pretrained == 'caffemodel':
            None
            src2dsts = get_caffeSrc2Dst(self.net_arch)
            copy_weights(self.state_dict(), 'caffemodel.%s' % self.net_arch, src2dsts=src2dsts)
        elif pretrained == 'torchmodel':
            raise NotImplementedError
        else:
            raise NotImplementedError
        return self

    def fix_conv1_conv2(self):
        for layer in range(10):
            for p in self.features[layer].parameters():
                p.requires_grad = False


class VGG16_Trunk(_VGG_Trunk):

    def __init__(self, init_weights=True):
        super(VGG16_Trunk, self).__init__(vgg_type='vgg16')


net_arch2Trunk = dict(alexnet=AlexNet_Trunk, vgg16=VGG16_Trunk, resnet101=ResNet101_Trunk, resnet50=ResNet50_Trunk)


class _naiReg_Net(nn.Module):

    @staticmethod
    def head_seq(in_size, reg_n_D, nr_cate=12, nr_fc8=334):
        seq = nn.Sequential(nn.Linear(in_size, nr_fc8), nn.ReLU(inplace=True), nn.Linear(nr_fc8, nr_cate * reg_n_D))
        init_weights_by_filling(seq, gaussian_std=0.005)
        return seq
    """BVLC alexnet architecture (Note: slightly different from pytorch implementation.)"""

    def __init__(self, nr_cate=12, net_arch='alexnet', pretrained=True):
        super(_naiReg_Net, self).__init__()
        _Trunk = net_arch2Trunk[net_arch]
        self.trunk = _Trunk(pretrained=pretrained)
        self.nr_cate = nr_cate
        self.top_size = 4096 if not self.trunk.net_arch.startswith('resnet') else 2048

    def forword(self, x, label):
        raise NotImplementedError


class Maskout(nn.Module):

    def __init__(self, nr_cate=3):
        super(Maskout, self).__init__()
        self.nr_cate = nr_cate

    def forward(self, x, label):
        """[input]
             x:    of shape (batchsize, nr_cate, nr_feat)
                or of shape (batchsize, nr_cate)   where nr_feat is treat as 1.
             label:
           [output]
             tensor of shape (batchsize, nr_cate)
        """
        batchsize, _nr_cate = x.size(0), x.size(1)
        assert _nr_cate == self.nr_cate, '2nd dim of x should be self.nr_cate=%s' % self.nr_cate
        assert batchsize == label.size(0)
        assert batchsize == reduce(lambda a1, a2: a1 * a2, label.size())
        item_inds = torch.from_numpy(np.arange(batchsize))
        if x.is_cuda:
            item_inds = item_inds
        cate_ind = label.view(batchsize)
        assert cate_ind.lt(self.nr_cate).all(), '[Exception] All index in cate_ind should be smaller than nr_cate.'
        masked_shape = (batchsize,) + x.size()[2:]
        return x[item_inds, cate_ind].view(*masked_shape)


class Smooth_L1_Loss_Handler:

    def __init__(self):
        self.smooth_l1_loss = nn.SmoothL1Loss()

    def compute_loss(self, tgts, Pred, GT):
        """ tgts: list of target names  e.g. tgts=['a', 'e', 't']
            GT  : dict of ground truth for each target
            Pred: dict of prediction   for each target
        """
        Loss = edict()
        for tgt in tgts:
            Loss[tgt] = self.smooth_l1_loss(Pred[tgt], GT[tgt])
        return Loss


def reg2d_pred2tgt(pr_sin, pr_cos):
    theta = torch.atan2(pr_sin, pr_cos)
    return theta


class reg_Euler2D_flat_Net(_naiReg_Net):
    """BVLC alexnet architecture (Note: slightly different from pytorch implementation.)"""

    def __init__(self, nr_cate=12, net_arch='alexnet', pretrained=True):
        _naiReg_Net.__init__(self, nr_cate=nr_cate, net_arch=net_arch, pretrained=pretrained)
        self.nr_cate = nr_cate
        self.reg_n_D = 2
        self.head_a = self.head_seq(self.top_size, self.reg_n_D, nr_cate=nr_cate)
        self.head_e = self.head_seq(self.top_size, self.reg_n_D, nr_cate=nr_cate)
        self.head_t = self.head_seq(self.top_size, self.reg_n_D, nr_cate=nr_cate)
        self.maskout = Maskout(nr_cate=nr_cate)
        self.loss_handler = Smooth_L1_Loss_Handler()
        self.targets = ['cos_a', 'sin_a', 'cos_e', 'sin_e', 'cos_t', 'sin_t']

    def forward(self, x, label):
        """label shape (batchsize, )  """
        x = self.trunk(x)
        batchsize = x.size(0)
        x_a = self.maskout(self.head_a(x).view(batchsize, self.nr_cate, self.reg_n_D), label)
        x_e = self.maskout(self.head_e(x).view(batchsize, self.nr_cate, self.reg_n_D), label)
        x_t = self.maskout(self.head_t(x).view(batchsize, self.nr_cate, self.reg_n_D), label)
        x_cos_a, x_sin_a = x_a[:, (0)], x_a[:, (1)]
        x_cos_e, x_sin_e = x_e[:, (0)], x_e[:, (1)]
        x_cos_t, x_sin_t = x_t[:, (0)], x_t[:, (1)]
        Prob = edict(cos_a=x_cos_a, sin_a=x_sin_a, cos_e=x_cos_e, sin_e=x_sin_e, cos_t=x_cos_t, sin_t=x_sin_t)
        return Prob

    def compute_loss(self, Prob, GT):
        Loss = self.loss_handler.compute_loss(self.targets, Prob, GT)
        return Loss

    def compute_pred(self, Prob):
        Pred = edict(a=reg2d_pred2tgt(Prob.sin_a, Prob.cos_a), e=reg2d_pred2tgt(Prob.sin_e, Prob.cos_e), t=reg2d_pred2tgt(Prob.sin_t, Prob.cos_t))
        return Pred


class Cross_Entropy_Loss_Handler:

    def __init__(self):
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def compute_loss(self, tgts, Pred, GT):
        """ tgts: list of target names  e.g. tgts=['a', 'e', 't']
            GT  : dict of ground truth for each target
            Pred: dict of prediction   for each target
        """
        Loss = edict()
        for tgt in tgts:
            Loss[tgt] = self.cross_entropy_loss(Pred[tgt], GT[tgt])
        return Loss


class Neg_Dot_Loss_Handler:

    def __init_(self):
        pass

    def compute_loss(self, tgts, Pred, GT):
        Loss = edict()
        for tgt in tgts:
            """ Bug fixed on 22 Aug 2018
                torch.dot can only be applied to 1-dim tensor
                Don't know why there's no error.  """
            Loss[tgt] = torch.mean(-torch.sum(GT[tgt] * Pred[tgt], dim=1))
        return Loss


def cls_pred(output, topk=(1,)):
    maxk = max(topk)
    batch_size = output.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    return pred


loss_balance = 4.0


class reg_Euler2D_Sexp_Net(_naiReg_Net):

    @staticmethod
    def head_seq(in_size, nr_cate=12, nr_fc8=1024):
        seq_fc8 = nn.Sequential(nn.Linear(in_size, nr_fc8), nn.ReLU(inplace=True), nn.Dropout())
        seq_ccss = nn.Linear(nr_fc8, nr_cate * 2)
        seq_sgnc = nn.Linear(nr_fc8, nr_cate * 4)
        init_weights_by_filling(seq_fc8, gaussian_std=0.005)
        init_weights_by_filling(seq_ccss, gaussian_std=0.005)
        init_weights_by_filling(seq_sgnc, gaussian_std=0.005)
        return seq_fc8, seq_ccss, seq_sgnc
    """BVLC alexnet architecture (Note: slightly different from pytorch implementation.)"""

    def __init__(self, nr_cate=12, net_arch='alexnet', pretrained=True, nr_fc8=1024):
        _naiReg_Net.__init__(self, nr_cate=nr_cate, net_arch=net_arch, pretrained=pretrained)
        self.nr_cate = nr_cate
        self.head_fc8_a, self.head_xx_yy_a, self.head_sign_x_y_a = self.head_seq(self.top_size, nr_cate=nr_cate, nr_fc8=nr_fc8)
        self.head_fc8_e, self.head_xx_yy_e, self.head_sign_x_y_e = self.head_seq(self.top_size, nr_cate=nr_cate, nr_fc8=nr_fc8)
        self.head_fc8_t, self.head_xx_yy_t, self.head_sign_x_y_t = self.head_seq(self.top_size, nr_cate=nr_cate, nr_fc8=nr_fc8)
        self.maskout = Maskout(nr_cate=nr_cate)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.loss_handler_ccss = Neg_Dot_Loss_Handler()
        self.loss_handler_sign = Cross_Entropy_Loss_Handler()
        self.targets = ['ccss_a', 'ccss_e', 'ccss_t', 'sign_a', 'sign_e', 'sign_t']

    def forward(self, x, label):
        """label shape (batchsize, )  """
        x = self.trunk(x)
        batchsize = x.size(0)
        x_a = self.head_fc8_a(x)
        x_e = self.head_fc8_e(x)
        x_t = self.head_fc8_t(x)
        log_xx_yy_a = self.maskout(self.head_xx_yy_a(x_a).view(batchsize, self.nr_cate, 2), label)
        log_xx_yy_e = self.maskout(self.head_xx_yy_e(x_e).view(batchsize, self.nr_cate, 2), label)
        log_xx_yy_t = self.maskout(self.head_xx_yy_t(x_t).view(batchsize, self.nr_cate, 2), label)
        logprob_xx_yy_a = self.logsoftmax(log_xx_yy_a)
        logprob_xx_yy_e = self.logsoftmax(log_xx_yy_e)
        logprob_xx_yy_t = self.logsoftmax(log_xx_yy_t)
        sign_x_y_a = self.maskout(self.head_sign_x_y_a(x_a).view(batchsize, self.nr_cate, 4), label)
        sign_x_y_e = self.maskout(self.head_sign_x_y_e(x_e).view(batchsize, self.nr_cate, 4), label)
        sign_x_y_t = self.maskout(self.head_sign_x_y_t(x_t).view(batchsize, self.nr_cate, 4), label)
        Prob = edict(logprob_ccss=edict(a=logprob_xx_yy_a, e=logprob_xx_yy_e, t=logprob_xx_yy_t), sign_cate_cs=edict(a=sign_x_y_a, e=sign_x_y_e, t=sign_x_y_t))
        return Prob

    def compute_loss(self, Prob, GT):
        Loss_ccss = self.loss_handler_ccss.compute_loss(['a', 'e', 't'], Prob.logprob_ccss, dict(a=GT.ccss_a, e=GT.ccss_e, t=GT.ccss_t))
        Loss_sign = self.loss_handler_sign.compute_loss(['a', 'e', 't'], Prob.sign_cate_cs, dict(a=GT.sign_a, e=GT.sign_e, t=GT.sign_t))
        Loss = edict(ccss_a=Loss_ccss.a * loss_balance, ccss_e=Loss_ccss.e * loss_balance, ccss_t=Loss_ccss.t * loss_balance, sign_a=Loss_sign.a, sign_e=Loss_sign.e, sign_t=Loss_sign.t)
        return Loss

    def compute_pred(self, Prob):
        label2signs = torch.FloatTensor([[1, 1], [-1, 1], [-1, -1], [1, -1]])
        label2signs = Variable(label2signs)
        batchsize = Prob.logprob_ccss['a'].size(0)
        Pred = edict()
        for tgt in Prob.logprob_ccss.keys():
            log_xx_yy = Prob.logprob_ccss[tgt]
            abs_cos_sin = torch.sqrt(torch.exp(log_xx_yy))
            sign_ind = cls_pred(Prob.sign_cate_cs[tgt], topk=(1,)).data.view(-1)
            item_inds = torch.from_numpy(np.arange(batchsize))
            sign_cos_sin = label2signs.expand(batchsize, 4, 2)[item_inds, sign_ind]
            cos_sin = abs_cos_sin * sign_cos_sin
            Pred[tgt] = torch.atan2(cos_sin[:, (1)], cos_sin[:, (0)])
        return Pred


class double_conv(nn.Module):
    """(conv => BN => ReLU) * 2"""

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True), nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(nn.MaxPool2d(2), double_conv(in_ch, out_ch))

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):

    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2), diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class _regNormalNet(nn.Module):

    def __init__(self, method, net_arch='vgg16', init_weights=True):
        super(_regNormalNet, self).__init__()
        _Trunk = net_arch2Trunk[net_arch][method]
        self.trunk = _Trunk(init_weights=init_weights)

    def forword(self, x, label):
        raise NotImplementedError


class Cos_Proximity_Loss_Handler:

    def __init__(self):
        self.cos_sim = nn.CosineSimilarity()

    def compute_loss(self, tgts, Pred, GT):
        """ tgts: list of target names. In this case   has to be tgts=['quat']
            GT  : dict of ground truth for each target
            Pred: dict of prediction   for each target
        """
        Loss = edict()
        for tgt in tgts:
            Loss[tgt] = torch.mean(1 - self.cos_sim(Pred[tgt], GT[tgt]))
        return Loss


class _BaseReg_Net(nn.Module):

    @staticmethod
    def head_seq(in_size, reg_n_D, nr_cate=12, nr_fc8=334, init_weights=True):
        seq = nn.Sequential(nn.Linear(in_size, nr_fc8), nn.ReLU(inplace=True), nn.Linear(nr_fc8, nr_cate * reg_n_D))
        if init_weights:
            init_weights_by_filling(seq, gaussian_std=0.005, kaiming_normal=True)
        return seq
    """BVLC alexnet architecture (Note: slightly different from pytorch implementation.)"""

    def __init__(self, nr_cate=12, net_arch='alexnet', init_weights=True):
        super(_BaseReg_Net, self).__init__()
        _Trunk = net_arch2Trunk[net_arch]
        self.trunk = _Trunk(init_weights=init_weights)
        self.nr_cate = nr_cate
        self.top_size = 4096 if not self.trunk.net_arch.startswith('resnet') else 2048

    def forword(self, x, label):
        raise NotImplementedError


def norm2unit(vecs, dim=1, p=2):
    """ vecs is of 2D:  (batchsize x nr_feat) or (batchsize x nr_feat x restDims)
        We normalize it to a unit vector here.
        For example, since mu is the coordinate on the circle (x,y) or sphere (x,y,z),
        we want to make it a unit norm.
    """
    vsize = vecs.size()
    batchsize, nr_feat = vsize[0], vsize[1]
    if hasattr(vecs, 'data'):
        check_inf = torch.abs(vecs.data) == float('inf')
    else:
        check_inf = torch.abs(vecs) == float('inf')
    check_nan = vecs != vecs
    if check_inf.any() or check_nan.any():
        None
        None
        exit()
    signs = torch.sign(vecs)
    signs[signs == 0] = 1
    vecs = vecs + signs * 0.0001
    vecs_in = vecs.clone()
    norm = ((vecs ** p).sum(dim=dim, keepdim=True) ** (1.0 / p)).detach()
    check_bad_norm = torch.abs(norm).lt(0.0001)
    if check_bad_norm.any():
        None
        None
        exit()
    vecs = vecs.div(norm)
    recomputed_norm = torch.sum(vecs ** p, dim=dim)
    check_mu = torch.abs(recomputed_norm - 1.0).lt(1e-06)
    if not check_mu.all():
        np.set_printoptions(threshold=np.inf)
        None
        None
        None
        exit()
    return vecs


class reg_Sflat_Net(_BaseReg_Net):
    """ L2 normalization activation, with cosine proximity loss. """
    """BVLC alexnet architecture (Note: slightly different from pytorch implementation.)"""

    def __init__(self, nr_cate=12, net_arch='alexnet', init_weights=True):
        _BaseReg_Net.__init__(self, nr_cate=nr_cate, net_arch=net_arch, init_weights=init_weights)
        self.nr_cate = nr_cate
        self.reg_n_D = 4
        self.head_quat = self.head_seq(self.top_size, self.reg_n_D, nr_cate=nr_cate, nr_fc8=996, init_weights=init_weights)
        self.maskout = Maskout(nr_cate=nr_cate)
        self.loss_handler = Cos_Proximity_Loss_Handler()
        self.targets = ['quat']

    def forward(self, x, label):
        """label shape (batchsize, )  """
        x = self.trunk(x)
        batchsize = x.size(0)
        x_quat = self.maskout(self.head_quat(x).view(batchsize, self.nr_cate, self.reg_n_D), label)
        x_quat = norm2unit(x_quat)
        Prob = edict(quat=x_quat)
        return Prob

    def compute_loss(self, Prob, GT):
        Loss = self.loss_handler.compute_loss(self.targets, Prob, GT)
        return Loss

    @staticmethod
    def compute_pred(Prob):
        x_quat = Prob['quat']
        batchsize = x_quat.size(0)
        batch_data = x_quat.data.cpu().numpy().copy()
        assert batch_data.shape == (batchsize, 4), batch_data.shape
        Pred = edict(quat=batch_data)
        return Pred


class reg_Sexp_Net(_BaseReg_Net):
    """ Spherical exponential activation + Sign classification, with cosine proximity loss """
    """BVLC alexnet architecture (Note: slightly different from pytorch implementation.)"""

    def __init__(self, nr_cate=12, net_arch='alexnet', init_weights=True):
        _BaseReg_Net.__init__(self, nr_cate=nr_cate, net_arch=net_arch, init_weights=init_weights)
        self.nr_cate = nr_cate
        self.reg_n_D = 4
        dim_need_sign = 3
        _signs = list(product(*([(-1, 1)] * dim_need_sign)))
        self.signs = [((1,) + x) for x in _signs]
        self.signs2label = odict(zip(self.signs, range(len(self.signs))))
        self.label2signs = Variable(torch.FloatTensor(self.signs))
        self.head_sqrdprob_quat = self.head_seq(self.top_size, self.reg_n_D, nr_cate=nr_cate, nr_fc8=996, init_weights=init_weights)
        self.head_signcate_quat = self.head_seq(self.top_size, len(self.signs), nr_cate=nr_cate, nr_fc8=996, init_weights=init_weights)
        self.maskout = Maskout(nr_cate=nr_cate)
        self.softmax = nn.Softmax(dim=1)
        self.maskout_sgc = Maskout(nr_cate=nr_cate)
        self.loss_handler_abs_quat = Cos_Proximity_Loss_Handler()
        self.loss_handler_sgc_quat = Cross_Entropy_Loss_Handler()
        self.targets = ['abs_quat', 'sgc_quat']
        self.gt_targets = ['quat']

    def forward(self, x, label):
        """label shape (batchsize, )  """
        x = self.trunk(x)
        batchsize = x.size(0)
        x_sqr_quat = self.maskout(self.head_sqrdprob_quat(x).view(batchsize, self.nr_cate, self.reg_n_D), label)
        x_sqr_quat = self.softmax(x_sqr_quat)
        x_sgc_quat = self.maskout_sgc(self.head_signcate_quat(x).view(batchsize, self.nr_cate, len(self.signs)), label)
        Prob = edict(abs_quat=torch.sqrt(x_sqr_quat), sgc_quat=x_sgc_quat)
        return Prob

    def compute_loss(self, Prob, GT):
        GT_abs_quat = torch.abs(GT.quat)
        GT_sign_quat = torch.sign(GT.quat)
        GT_sign_quat[GT_sign_quat == 0] = 1
        signs_tuples = [tuple(x) for x in GT_sign_quat.data.cpu().numpy().astype(np.int32).tolist()]
        for signs_tuple in signs_tuples:
            assert signs_tuple[0] > 0, 'Need GT to be all positive on first dim of quaternion: %s' % GT
        GT_sgc_quat = Variable(torch.LongTensor([self.signs2label[signs_tuple] for signs_tuple in signs_tuples]))
        if GT.quat.is_cuda:
            GT_sgc_quat = GT_sgc_quat
        _GT = edict(abs_quat=GT_abs_quat, sgc_quat=GT_sgc_quat)
        Loss_abs_quat = self.loss_handler_abs_quat.compute_loss(['abs_quat'], Prob, _GT)
        Loss_sgc_quat = self.loss_handler_sgc_quat.compute_loss(['sgc_quat'], Prob, _GT)
        Loss = edict(abs_quat=Loss_abs_quat['abs_quat'] * 10, sgc_quat=Loss_sgc_quat['sgc_quat'])
        return Loss

    def compute_pred(self, Prob):
        x_abs_quat = Prob['abs_quat']
        x_sgc_quat = Prob['sgc_quat']
        batchsize = x_abs_quat.size(0)
        sign_ind = cls_pred(x_sgc_quat, topk=(1,)).data.view(-1)
        item_inds = torch.from_numpy(np.arange(batchsize))
        _label_shape = self.label2signs.size()
        x_sign_quat = self.label2signs.expand(batchsize, *_label_shape)[item_inds, sign_ind]
        x_quat = x_abs_quat * x_sign_quat
        batch_quat = x_quat.data.cpu().numpy().copy()
        batchsize = x_quat.size(0)
        assert batch_quat.shape == (batchsize, 4), batch_quat.shape
        Pred = edict(quat=batch_quat)
        return Pred


class inconv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class outconv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet(nn.Module):

    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x4_ = self.up1(x5, x4)
        x3_ = self.up2(x4_, x3)
        x2_ = self.up3(x3_, x2)
        x1_ = self.up4(x2_, x1)
        x_ = self.outc(x1_)
        None
        None
        None
        None
        None
        None
        None
        None
        None
        None
        return x


class reg_Direct_Net(_BaseReg_Net):
    """ No any L2 normalization to guarantee prediction is on n-sphere, smooth l1 loss is used. """
    """BVLC alexnet architecture (Note: slightly different from pytorch implementation.)"""

    def __init__(self, nr_cate=12, net_arch='alexnet', init_weights=True):
        _BaseReg_Net.__init__(self, nr_cate=nr_cate, net_arch=net_arch, init_weights=init_weights)
        self.nr_cate = nr_cate
        self.reg_n_D = 4
        self.head_quat = self.head_seq(self.top_size, self.reg_n_D, nr_cate=nr_cate, nr_fc8=996, init_weights=init_weights)
        self.maskout = Maskout(nr_cate=nr_cate)
        self.loss_handler = Smooth_L1_Loss_Handler()
        self.targets = ['quat']

    def forward(self, x, label):
        """label shape (batchsize, )  """
        x = self.trunk(x)
        batchsize = x.size(0)
        x_quat = self.maskout(self.head_quat(x).view(batchsize, self.nr_cate, self.reg_n_D), label)
        Prob = edict(quat=x_quat)
        return Prob

    def compute_loss(self, Prob, GT):
        Loss = self.loss_handler.compute_loss(self.targets, Prob, GT)
        return Loss

    @staticmethod
    def compute_pred(Prob):
        x_quat = Prob['quat']
        x_quat = norm2unit(x_quat)
        batchsize = x_quat.size(0)
        batch_data = x_quat.data.cpu().numpy().copy()
        assert batch_data.shape == (batchsize, 4), batch_data.shape
        Pred = edict(quat=batch_data)
        return Pred


class Test_AlexNet(nn.Module):

    def __init__(self, nr_cate=3, _Trunk=AlexNet_Trunk):
        super(Test_AlexNet, self).__init__()
        self.truck = _Trunk().copy_weights()
        self.nr_cate = nr_cate
        self.maskout = Maskout(nr_cate=nr_cate)

    def forward(self, x, label):
        x = self.truck(x)
        batchsize = x.size(0)
        self.head_s2 = nn.Sequential(nn.Linear(4096, 84), nn.ReLU(inplace=True), nn.Linear(84, self.nr_cate * 3))
        self.head_s1 = nn.Sequential(nn.Linear(4096, 84), nn.ReLU(inplace=True), nn.Linear(84, self.nr_cate * 2))
        x_s2 = self.maskout(self.head_s2(x).view(batchsize, self.nr_cate, 3), label)
        x_s1 = self.maskout(self.head_s1(x).view(batchsize, self.nr_cate, 2), label)
        x_s2 = norm2unit(x_s2)
        x_s1 = norm2unit(x_s1)
        Pred = edict(s2=x_s2, s1=x_s1)
        return Pred


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class InceptionA(nn.Module):

    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)
        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)
        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.conv0 = BasicConv2d(in_channels, 128, kernel_size=1)
        self.conv1 = BasicConv2d(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01
        self.fc = nn.Linear(768, num_classes)
        self.fc.stddev = 0.001

    def forward(self, x):
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        x = self.conv0(x)
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class InceptionB(nn.Module):

    def __init__(self, in_channels):
        super(InceptionB, self).__init__()
        self.branch3x3 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2)
        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):

    def __init__(self, in_channels, channels_7x7):
        super(InceptionC, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)
        c7 = channels_7x7
        self.branch7x7_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = BasicConv2d(c7, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = BasicConv2d(c7, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)
        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionD(nn.Module):

    def __init__(self, in_channels):
        super(InceptionD, self).__init__()
        self.branch3x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(192, 320, kernel_size=3, stride=2)
        self.branch7x7x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = BasicConv2d(192, 192, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return torch.cat(outputs, 1)


class InceptionE(nn.Module):

    def __init__(self, in_channels):
        super(InceptionE, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1)
        self.branch3x3_1 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))
        self.branch3x3dbl_1 = BasicConv2d(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))
        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [self.branch3x3_2a(branch3x3), self.branch3x3_2b(branch3x3)]
        branch3x3 = torch.cat(branch3x3, 1)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [self.branch3x3dbl_3a(branch3x3dbl), self.branch3x3dbl_3b(branch3x3dbl)]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class _Inception3_Trunk(nn.Module):

    def __init__(self, aux_logits=True, transform_input=False, init_weights=True, start=None, end=None):
        super(_Inception3_Trunk, self).__init__()
        self.net_arch = 'inception_v3'
        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)
        self.Mixed_6a = InceptionB(288)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)
        if aux_logits:
            self.AuxLogits = InceptionAux(768, num_classes)
        self.Mixed_7a = InceptionD(768)
        self.Mixed_7b = InceptionE(1280)
        self.Mixed_7c = InceptionE(2048)
        self.fc = nn.Linear(2048, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                X = stats.truncnorm(-2, 2, scale=stddev)
                values = torch.Tensor(X.rvs(m.weight.numel()))
                values = values.view(m.weight.size())
                m.weight.data.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self._layer_names = [name for name, module in self.named_children()]
        if init_weights:
            self.init_weights(pretrained='caffemodel')
        self.truck_seq = self.sub_seq(start=start, end=end)

    def forward(self, x):
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, (0)], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, (1)], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, (2)], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        if self.training and self.aux_logits:
            aux = self.AuxLogits(x)
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)
        x = F.avg_pool2d(x, kernel_size=8)
        x = F.dropout(x, training=self.training)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        if self.training and self.aux_logits:
            return x, aux
        return x


class ResNet18_Trunk(_ResNet_Trunk):

    def __init__(self, **kwargs):
        _ResNet_Trunk.__init__(self, res_type='resnet18', **kwargs)


class ResNet34_Trunk(_ResNet_Trunk):

    def __init__(self, **kwargs):
        _ResNet_Trunk.__init__(self, res_type='resnet34', **kwargs)


class ResNet152_Trunk(_ResNet_Trunk):

    def __init__(self, **kwargs):
        _ResNet_Trunk.__init__(self, res_type='resnet152', **kwargs)


class Test_Net(nn.Module):

    def __init__(self, nr_cate=3, _Trunk=VGG16_Trunk):
        super(Test_Net, self).__init__()
        self.truck = _Trunk()
        self.nr_cate = nr_cate
        self.maskout = Maskout(nr_cate=nr_cate)

    def forward(self, x, label):
        x = self.truck(x)
        batchsize = x.size(0)
        self.head_s2 = nn.Sequential(nn.Linear(4096, 84), nn.ReLU(inplace=True), nn.Linear(84, self.nr_cate * 3))
        self.head_s1 = nn.Sequential(nn.Linear(4096, 84), nn.ReLU(inplace=True), nn.Linear(84, self.nr_cate * 2))
        x_s2 = self.maskout(self.head_s2(x).view(batchsize, self.nr_cate, 3), label)
        x_s1 = self.maskout(self.head_s1(x).view(batchsize, self.nr_cate, 2), label)
        x_s2 = norm2unit(x_s2)
        x_s1 = norm2unit(x_s1)
        Pred = edict(s2=x_s2, s1=x_s1)
        return Pred


class VGGM_Trunk(_VGG_Trunk):

    def __init__(self, init_weights=True):
        super(VGGM_Trunk, self).__init__(vgg_type='vggm')


class VGG19_Trunk(_VGG_Trunk):

    def __init__(self, init_weights=True):
        super(VGG19_Trunk, self).__init__(vgg_type='vgg19')


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BasicConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InceptionA,
     lambda: ([], {'in_channels': 4, 'pool_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InceptionAux,
     lambda: ([], {'in_channels': 4, 'num_classes': 4}),
     lambda: ([torch.rand([4, 4, 18, 18])], {}),
     True),
    (InceptionB,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InceptionC,
     lambda: ([], {'in_channels': 4, 'channels_7x7': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InceptionD,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InceptionE,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LocalResponseNorm,
     lambda: ([], {'size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (UNet,
     lambda: ([], {'n_channels': 4, 'n_classes': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     True),
    (double_conv,
     lambda: ([], {'in_ch': 4, 'out_ch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (down,
     lambda: ([], {'in_ch': 4, 'out_ch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (inconv,
     lambda: ([], {'in_ch': 4, 'out_ch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (nnAdd,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (outconv,
     lambda: ([], {'in_ch': 4, 'out_ch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (up,
     lambda: ([], {'in_ch': 4, 'out_ch': 4}),
     lambda: ([torch.rand([4, 1, 4, 4]), torch.rand([4, 3, 4, 4])], {}),
     True),
]

class Test_leoshine_Spherical_Regression(_paritybench_base):
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

    def test_010(self):
        self._check(*TESTCASES[10])

    def test_011(self):
        self._check(*TESTCASES[11])

    def test_012(self):
        self._check(*TESTCASES[12])

    def test_013(self):
        self._check(*TESTCASES[13])

    def test_014(self):
        self._check(*TESTCASES[14])

    def test_015(self):
        self._check(*TESTCASES[15])

