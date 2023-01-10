import sys
_module = sys.modules[__name__]
del sys
DeepResNet = _module
Encoder = _module
deeplab_resnet_skip = _module
GCN = _module
GCN_layer = _module
GCN_res_layer = _module
GNN = _module
base_network = _module
grid_def_network = _module
superpixel_grid = _module
Models = _module
deformable_grid = _module
Utils = _module
dense_quad = _module
matrix_utils = _module
mypath = _module
parser = _module
plot_sample = _module
time_utils = _module
utils = _module
dataloaders = _module
change_paths = _module
cityscapes_full = _module
cityscapes_processed = _module
custom_transforms = _module
helpers = _module
evaluation = _module
metrics = _module
DefGrid = _module
check_condition_lattice_bbox = _module
utils = _module
diff_variance = _module
mean_feature = _module
mean_feature = _module
variance_function_atom = _module
line_distance_func_parallel = _module
utils = _module
line_distance_func_topk = _module
utils = _module
layers = _module
train_def_grid_full = _module
train_def_grid_multi_comp = _module

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


import torch.nn as nn


import torchvision.models.resnet as resnet


import torch


import numpy as np


from copy import deepcopy


from torch.nn import functional as F


import torch.nn.functional as F


import torchvision.transforms as transforms


import math


from torch.nn.modules.module import Module


from collections import OrderedDict


import torchvision


from collections import defaultdict


import random


import copy


import matplotlib.pyplot as plt


from matplotlib.path import Path


import time


import scipy.misc as m


from torch.utils import data


from torch.utils.data import Dataset


from scipy import misc


from scipy import ndimage


import numpy.random as random


import torch.autograd


from torch.autograd import Function


from torch.utils.cpp_extension import load


import matplotlib


import torch.optim as optim


from torch.utils.data import DataLoader


from torchvision import transforms


affine_par = True


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation_=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        padding = 1
        if dilation_ == 2:
            padding = 2
        elif dilation_ == 4:
            padding = 4
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=padding, bias=False, dilation=dilation_)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine=affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False
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


class ClassifierModule(nn.Module):

    def __init__(self, dilation_series, padding_series, n_classes):
        super(ClassifierModule, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(nn.Conv2d(2048, n_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))
        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
        return out


class PSPModule(nn.Module):
    """
    Pyramid Scene Parsing module
    """

    def __init__(self, in_channels=128, out_channels=128, sizes=(1, 2, 3, 6), n_classes=1):
        super(PSPModule, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage_1(in_channels, size) for size in sizes])
        self.bottleneck = self._make_stage_2(in_channels * (len(sizes) // 4 + 1), out_channels)
        self.relu = nn.ReLU()
        self.final = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def _make_stage_1(self, in_channels, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1, bias=False)
        bn = nn.InstanceNorm2d(in_channels // 4, affine=False)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)

    def _make_stage_2(self, in_channels, out_channels):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = nn.InstanceNorm2d(in_channels, affine=False)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(conv, bn, relu)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages]
        priors.append(feats)
        bottle = self.relu(self.bottleneck(torch.cat(priors, 1)))
        out = self.final(bottle)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, n_classes, input_channel=3, classifier='psp', dilations=(2, 4), strides=(2, 2, 2, 1, 1), feature_channels=(64, 128, 256, 512)):
        self.inplanes = 64
        self.classifier = classifier
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, self.inplanes, kernel_size=7, stride=strides[0], padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=strides[1], padding=1, ceil_mode=False)
        self.layer1 = self._make_layer(block, feature_channels[0], layers[0])
        self.layer2 = self._make_layer(block, feature_channels[1], layers[1], stride=strides[2])
        self.layer3 = self._make_layer(block, feature_channels[2], layers[2], stride=strides[3], dilation__=dilations[0])
        self.layer4 = self._make_layer(block, feature_channels[3], layers[3], stride=strides[4], dilation__=dilations[1])
        if classifier == 'atrous':
            self.layer5 = self._make_pred_layer(ClassifierModule, [6, 12, 18, 24], [6, 12, 18, 24], n_classes=n_classes)
        elif classifier == 'psp':
            self.layer5 = PSPModule(in_features=2048, out_features=512, sizes=(1, 2, 3, 6), n_classes=n_classes)
        else:
            self.layer5 = None
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation__=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation__ == 2 or dilation__ == 4:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion, affine=affine_par))
        if downsample is not None:
            for i in downsample._modules['1'].parameters():
                i.requires_grad = False
        layers = [block(self.inplanes, planes, stride, dilation_=dilation__, downsample=downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation_=dilation__))
        return nn.Sequential(*layers)

    def _make_pred_layer(self, block, dilation_series, padding_series, n_classes):
        return block(dilation_series, padding_series, n_classes)

    def forward(self, x, bbox=None):
        x = self.conv1(x)
        x = self.bn1(x)
        conv1_f = self.relu(x)
        x = self.maxpool(conv1_f)
        layer1_f = self.layer1(x)
        layer2_f = self.layer2(layer1_f)
        layer3_f = self.layer3(layer2_f)
        layer4_f = self.layer4(layer3_f)
        if self.layer5 is not None:
            layer5_f = self.layer5(layer4_f)
            return conv1_f, layer1_f, layer2_f, layer3_f, layer4_f, layer5_f
        return conv1_f, layer1_f, layer2_f, layer3_f, layer4_f

    def load_pretrained_ms(self, base_network, input_channel=3):
        flag = 0
        for module, module_ori in zip(self.modules(), base_network.Scale.modules()):
            if isinstance(module, nn.Conv2d) and isinstance(module_ori, nn.Conv2d):
                if not flag and input_channel != 3:
                    module.weight[:, :3, :, :].data = deepcopy(module_ori.weight.data)
                    module.bias = deepcopy(module_ori.bias)
                    for i in range(3, int(module.weight.data.shape[1])):
                        module.weight[:, i, :, :].data = deepcopy(module_ori.weight[:, -1, :, :][:, np.newaxis, :, :].data)
                    flag = 1
                elif module.weight.data.shape == module_ori.weight.data.shape:
                    module.weight.data = deepcopy(module_ori.weight.data)
                    module.bias = deepcopy(module_ori.bias)
                else:
                    None
            elif isinstance(module, nn.BatchNorm2d) and isinstance(module_ori, nn.BatchNorm2d) and module.weight.data.shape == module_ori.weight.data.shape:
                module.weight.data = deepcopy(module_ori.weight.data)
                module.bias.data = deepcopy(module_ori.bias.data)

    def reload(self, path, strict=False):
        self.load_state_dict(torch.load(path)['state_dict'], strict=strict)


def outS(i):
    return i // 8


class MS_Deeplab(nn.Module):

    def __init__(self, block, NoLabels, input_channel=3):
        super(MS_Deeplab, self).__init__()
        self.Scale = ResNet(block, [3, 4, 23, 3], NoLabels, input_channel=input_channel)

    def forward(self, x):
        input_size = x.size()[2]
        self.interp1 = nn.Upsample(size=(int(input_size * 0.75) + 1, int(input_size * 0.75) + 1), mode='bilinear', align_corners=True)
        self.interp2 = nn.Upsample(size=(int(input_size * 0.5) + 1, int(input_size * 0.5) + 1), mode='bilinear', align_corners=True)
        self.interp3 = nn.Upsample(size=(outS(input_size), outS(input_size)), mode='bilinear', align_corners=True)
        out = []
        x2 = self.interp1(x)
        x3 = self.interp2(x)
        out.append(self.Scale(x))
        out.append(self.interp3(self.Scale(x2)))
        out.append(self.Scale(x3))
        x2Out_interp = out[1]
        x3Out_interp = self.interp3(out[2])
        temp1 = torch.max(out[0], x2Out_interp)
        out.append(torch.max(temp1, x3Out_interp))
        return out[-1]


class NetHead(nn.Module):

    def __init__(self, input_dim=128, final_dim=128):
        super(NetHead, self).__init__()
        conv_final_1 = nn.Conv2d(input_dim, final_dim, kernel_size=3, padding=1, bias=False)
        bn_final_1 = nn.BatchNorm2d(final_dim)
        relu_final_1 = nn.ReLU(inplace=True)
        self.encoder = nn.Sequential(conv_final_1, bn_final_1, relu_final_1)

    def my_load_state_dict(self, state_dict, strict=True):
        new_state_dict = {}
        for k in state_dict:
            if 'encoder.0' in k:
                new_v = torch.zeros_like(self.encoder[0].weight.data)
                old_v = state_dict[k]
                new_v[:, :old_v.shape[1], :, :] = deepcopy(old_v)
                for i in range(old_v.shape[1], new_v.shape[1]):
                    new_v[:, i, :, :] = deepcopy(old_v[:, -1, :, :])
            else:
                new_v = state_dict[k]
            new_state_dict[k] = new_v
        self.load_state_dict(new_state_dict, strict=strict)

    def forward(self, x):
        return self.encoder(x)


def Res_Deeplab(n_classes=21, pretrained=False, reload_path=''):
    model = MS_Deeplab(Bottleneck, n_classes)
    if pretrained:
        saved_state_dict = torch.load(reload_path, map_location=lambda storage, loc: storage)
        """
        if n_classes != 21:
            for i in saved_state_dict:
                i_parts = i.split('.')
                if i_parts[1] == 'layer5':
                    saved_state_dict[i] = model.state_dict()[i]
        """
        model.load_state_dict(saved_state_dict, strict=False)
    return model


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def resnet101(n_classes, input_channel=3, classifier='atrous', dilations=(2, 4), strides=(2, 2, 2, 1, 1)):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], n_classes=n_classes, input_channel=input_channel, classifier=classifier, dilations=dilations, strides=strides)
    return model


class DeepLabResnet(nn.Module):

    def __init__(self, concat_channels=64, final_dim=128, final_resolution=(224, 224), input_channel=3, classifier='psp', n_classes=1, reload=True, cnn_feature_grids=None, normalize_input=True, use_final=True, update_last=False):
        super(DeepLabResnet, self).__init__()
        self.use_final = use_final
        self.input_channel = input_channel
        self.normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.cnn_feature_grids = cnn_feature_grids
        self.concat_channels = concat_channels
        self.final_resolution = final_resolution
        self.final_dim = final_dim
        self.feat_size = 28
        self.reload = reload
        self.update_last = update_last
        self.image_feature_dim = 256
        self.n_classes = n_classes
        self.classifier = classifier
        self.normalize_input = normalize_input
        None
        self.resnet = resnet101(n_classes, input_channel=input_channel, classifier=classifier)
        concat1 = nn.Conv2d(64, concat_channels, kernel_size=3, padding=1, bias=False)
        bn1 = nn.BatchNorm2d(concat_channels)
        relu1 = nn.ReLU(inplace=True)
        self.conv1_concat = nn.Sequential(concat1, bn1, relu1)
        concat2 = nn.Conv2d(256, concat_channels, kernel_size=3, padding=1, bias=False)
        bn2 = nn.BatchNorm2d(concat_channels)
        relu2 = nn.ReLU(inplace=True)
        self.res1_concat = nn.Sequential(concat2, bn2, relu2)
        concat3 = nn.Conv2d(512, concat_channels, kernel_size=3, padding=1, bias=False)
        bn3 = nn.BatchNorm2d(concat_channels)
        relu3 = nn.ReLU(inplace=True)
        self.res2_concat = nn.Sequential(concat3, bn3, relu3)
        concat4 = nn.Conv2d(2048, concat_channels, kernel_size=3, padding=1, bias=False)
        bn4 = nn.BatchNorm2d(concat_channels)
        relu4 = nn.ReLU(inplace=True)
        self.res4_concat = nn.Sequential(concat4, bn4, relu4)
        self.layer5_concat_channels = concat_channels
        concat5 = nn.Conv2d(512, self.layer5_concat_channels, kernel_size=3, padding=1, bias=False)
        bn5 = nn.BatchNorm2d(self.layer5_concat_channels)
        relu5 = nn.ReLU(inplace=True)
        self.res5_concat = nn.Sequential(concat5, bn5, relu5)
        if self.use_final:
            conv_final_1 = nn.Conv2d(5 * concat_channels, final_dim, kernel_size=3, padding=1, bias=False)
            bn_final_1 = nn.BatchNorm2d(final_dim)
            relu_final_1 = nn.ReLU(inplace=True)
            self.final = nn.Sequential(conv_final_1, bn_final_1, relu_final_1)
        """
        conv_final_1 = nn.Conv2d(4*concat_channels, 128, kernel_size=3, padding=1, stride=2,
            bias=False)
        bn_final_1 = nn.BatchNorm2d(128)
        conv_final_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2, bias=False)
        bn_final_2 = nn.BatchNorm2d(128)
        conv_final_3 = nn.Conv2d(128, final_dim, kernel_size=3, padding=1, bias=False)
        bn_final_3 = nn.BatchNorm2d(final_dim)

        self.conv_final = nn.Sequential(conv_final_1, bn_final_1, conv_final_2, bn_final_2,
            conv_final_3, bn_final_3)

        if self.classifier != 'psp' :
            self.final_dim = 64 * 4
        else:
            self.final_dim = 64 * 5
        """
        self.final_upsample = nn.Upsample(size=self.final_resolution, mode='bilinear', align_corners=True)
        self.mid_upsample = nn.Upsample(size=(self.final_resolution[0] // 2, self.final_resolution[1] // 2), mode='bilinear', align_corners=True)
        if self.reload:
            self.reload_model()

    def reload_model(self, path='/scratch/ssd001/home/jungao/pretrained_model/MS_DeepLab_resnet_trained_VOC.pth'):
        if not os.path.exists(path):
            path = '/scratch/gobi1/jungao/pretrained_model/MS_DeepLab_resnet_trained_VOC.pth'
        if not os.path.exists(path):
            path = '/u/jungao/pretrain_w/MS_DeepLab_resnet_trained_VOC.pth'
        model_full = Res_Deeplab(self.n_classes, pretrained=True, reload_path=path)
        self.resnet.load_pretrained_ms(model_full, input_channel=self.input_channel)
        del model_full

    def forward(self, x):
        batch_size = x.shape[0]
        assert self.classifier == 'psp'
        if self.normalize_input:
            x = self.normalize(x)
        conv1_f, layer1_f, layer2_f, layer3_f, layer4_f, layer5_f = self.resnet(x)
        if self.update_last:
            conv1_f_gcn = self.mid_upsample(self.conv1_concat(conv1_f))
            layer1_f_gcn = self.mid_upsample(self.res1_concat(layer1_f))
            layer2_f_gcn = self.mid_upsample(self.res2_concat(layer2_f))
            layer4_f_gcn = self.mid_upsample(self.res4_concat(layer4_f))
            layer5_f_gcn = self.mid_upsample(self.res5_concat(layer5_f))
        else:
            conv1_f_gcn = self.final_upsample(self.conv1_concat(conv1_f))
            layer1_f_gcn = self.final_upsample(self.res1_concat(layer1_f))
            layer2_f_gcn = self.final_upsample(self.res2_concat(layer2_f))
            layer4_f_gcn = self.final_upsample(self.res4_concat(layer4_f))
            layer5_f_gcn = self.final_upsample(self.res5_concat(layer5_f))
        final_features = [conv1_f_gcn, layer1_f_gcn, layer2_f_gcn, layer4_f_gcn, layer5_f_gcn]
        final_cat_features = torch.cat(final_features, dim=1)
        if self.use_final:
            final_features = self.final(final_cat_features)
        else:
            final_features = final_cat_features
        return final_features, final_cat_features

    def normalize(self, x):
        individual = torch.unbind(x, dim=0)
        out = []
        for x in individual:
            x[:3] = self.normalizer(x[:3])
            out.append(x)
        return torch.stack(out, dim=0)

    def my_load_state_dict(self, state_dict, strict=True):
        new_state_dict = {}
        for k in state_dict:
            if 'resnet.conv1' in k:
                new_v = torch.zeros_like(self.resnet.conv1.weight.data)
                old_v = state_dict[k]
                new_v[:, :old_v.shape[1], :, :] = deepcopy(old_v)
                for i in range(old_v.shape[1], new_v.shape[1]):
                    new_v[:, i, :, :] = deepcopy(old_v[:, -1, :, :])
            else:
                new_v = state_dict[k]
            new_state_dict[k] = new_v
        self.load_state_dict(new_state_dict, strict=strict)


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, state_dim, name='', out_state_dim=None):
        super(GraphConvolution, self).__init__()
        self.state_dim = state_dim
        if out_state_dim == None:
            self.out_state_dim = state_dim
        else:
            self.out_state_dim = out_state_dim
        self.fc1 = nn.Linear(in_features=self.state_dim, out_features=self.out_state_dim)
        self.fc2 = nn.Linear(in_features=self.state_dim, out_features=self.out_state_dim)
        self.name = name

    def forward(self, input, adj):
        state_in = self.fc1(input)
        forward_input = self.fc2(torch.bmm(adj, input))
        return state_in + forward_input

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.name + ')'


class GraphResConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, state_dim, name=''):
        super(GraphResConvolution, self).__init__()
        self.state_dim = state_dim
        self.gcn_1 = GraphConvolution(state_dim, '%s_1' % name)
        self.gcn_2 = GraphConvolution(state_dim, '%s_2' % name)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.name = name

    def forward(self, input, adj):
        output_1 = self.gcn_1(input, adj)
        output_1_relu = self.relu1(output_1)
        output_2 = self.gcn_2(output_1_relu, adj)
        output_2_res = output_2 + input
        output = self.relu2(output_2_res)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.name + ')'


class GCN(nn.Module):

    def __init__(self, state_dim=256, feature_dim=256, out_dim=2, layer_num=8):
        super(GCN, self).__init__()
        self.state_dim = state_dim
        self.layer_num = layer_num
        self.first_gcn = GraphConvolution(feature_dim, 'first_gcn', out_state_dim=self.state_dim)
        self.middle_gcn = nn.ModuleList([])
        for i in range(self.layer_num - 2):
            self.middle_gcn.append(GraphResConvolution(self.state_dim, 'gcn_res_%d' % (i + 1)))
        self.last_gcn = GraphConvolution(self.state_dim, 'last_gcn', out_state_dim=self.state_dim)
        self.fc = nn.Linear(in_features=self.state_dim, out_features=out_dim)

    def forward(self, x, adj):
        out = F.relu(self.first_gcn(x, adj))
        for m_gcn in self.middle_gcn:
            out = m_gcn(out, adj)
        out = F.relu(self.last_gcn(out, adj))
        out = self.fc(out)
        return out


class MyResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, use_instance_norm=True):
        super(MyResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.use_instance_norm = use_instance_norm
        if self.use_instance_norm:
            self.in1 = nn.InstanceNorm2d(out_channels, affine=False, track_running_stats=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        if self.use_instance_norm:
            self.in2 = nn.InstanceNorm2d(out_channels, affine=False, track_running_stats=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        if self.use_instance_norm:
            out = self.in1(out)
        out = F.relu(out)
        out = self.conv2(out)
        if self.use_instance_norm:
            out = self.in2(out)
        out += residual
        out = F.relu(out)
        return out


class MyEncoder(nn.Module):

    def __init__(self, nr_channel=128, input_channel=3, use_instance_norm=True):
        super(MyEncoder, self).__init__()
        None
        self.input_channel = input_channel
        self.additional_input = self.input_channel > 3
        if not use_instance_norm:
            self.first_base_module = nn.Sequential(OrderedDict([('base_conv1', nn.Conv2d(3, nr_channel, kernel_size=7, stride=1, padding=3, bias=False)), ('base_relu1', nn.ReLU()), ('base_res2', MyResBlock(nr_channel, nr_channel, use_instance_norm)), ('base_res3', MyResBlock(nr_channel, nr_channel, use_instance_norm)), ('base_res4', MyResBlock(nr_channel, nr_channel, use_instance_norm)), ('base_res5', MyResBlock(nr_channel, nr_channel, use_instance_norm)), ('base_res6', MyResBlock(nr_channel, nr_channel, use_instance_norm)), ('base_conv5', nn.Conv2d(nr_channel, nr_channel, kernel_size=1, stride=1, padding=0, bias=False))]))
        else:
            self.first_base_module = nn.Sequential(OrderedDict([('base_conv1', nn.Conv2d(3, nr_channel, kernel_size=7, stride=1, padding=3, bias=False)), ('base_in1', nn.InstanceNorm2d(nr_channel, affine=False, track_running_stats=True)), ('base_relu1', nn.ReLU()), ('base_res2', MyResBlock(nr_channel, nr_channel)), ('base_res3', MyResBlock(nr_channel, nr_channel)), ('base_res4', MyResBlock(nr_channel, nr_channel)), ('base_res5', MyResBlock(nr_channel, nr_channel)), ('base_res6', MyResBlock(nr_channel, nr_channel)), ('base_conv5', nn.Conv2d(nr_channel, nr_channel, kernel_size=1, stride=1, padding=0, bias=False)), ('base_in5', nn.InstanceNorm2d(nr_channel, affine=False, track_running_stats=True))]))
        if self.additional_input:
            if not use_instance_norm:
                self.first_add_module = nn.Sequential(OrderedDict([('add_conv6', nn.Conv2d(nr_channel + self.input_channel - 3, nr_channel, kernel_size=1, stride=1, bias=False)), ('add_relu6', nn.ReLU()), ('add_conv7', nn.Conv2d(nr_channel, nr_channel, kernel_size=1, stride=1, bias=False))]))
            else:
                self.first_add_module = nn.Sequential(OrderedDict([('add_conv6', nn.Conv2d(nr_channel + self.input_channel - 3, nr_channel, kernel_size=1, stride=1, bias=False)), ('add_in6', nn.InstanceNorm2d(nr_channel, affine=False, track_running_stats=True)), ('add_relu6', nn.ReLU()), ('add_conv7', nn.Conv2d(nr_channel, nr_channel, kernel_size=1, stride=1, bias=False)), ('add_in7', nn.InstanceNorm2d(nr_channel, affine=False, track_running_stats=True))]))
        if not use_instance_norm:
            self.first_branch = nn.Sequential(OrderedDict([('fb_res8', MyResBlock(nr_channel, nr_channel)), ('fb_conv9', nn.Conv2d(nr_channel, nr_channel, kernel_size=1, stride=1, padding=0, bias=False))]))
        else:
            self.first_branch = nn.Sequential(OrderedDict([('fb_res8', MyResBlock(nr_channel, nr_channel)), ('fb_conv9', nn.Conv2d(nr_channel, nr_channel, kernel_size=1, stride=1, padding=0, bias=False)), ('fb_in9', nn.InstanceNorm2d(nr_channel, affine=False, track_running_stats=True))]))

    def forward(self, x):
        if self.additional_input:
            first_image = x[:, :3, :, :]
        else:
            first_image = x
        first_out = self.first_base_module(first_image)
        if self.additional_input:
            first_add = x[:, 3:, :, :]
            first_out = torch.cat((first_out, first_add), 1)
            first_out = self.first_add_module(first_out)
        first_out = F.relu(first_out)
        first_branch_out = self.first_branch(first_out)
        return first_branch_out, first_branch_out


class PPM(nn.Module):

    def __init__(self, in_dim, reduction_dim, bins, BatchNorm):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(nn.AdaptiveAvgPool2d(bin), nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False), BatchNorm(reduction_dim), nn.ReLU(inplace=True)))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)


class SimpleEncoder(nn.Module):

    def __init__(self):
        super(SimpleEncoder, self).__init__()
        resnet = torchvision.models.resnet50(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        bottleneck = list(resnet.layer1.children())
        self.layer1 = nn.Sequential(bottleneck[0], bottleneck[1], bottleneck[2])
        self.ppm = PPM(256, 64, (1, 2, 3, 6), torch.nn.BatchNorm2d)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.ppm(x)
        return x, x


class EncoderHead(nn.Module):

    def __init__(self, input_feature_channel_num=128):
        super(EncoderHead, self).__init__()
        self.predictor = nn.Sequential(nn.Conv2d(input_feature_channel_num, input_feature_channel_num, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(input_feature_channel_num), nn.ReLU(), nn.Conv2d(input_feature_channel_num, input_feature_channel_num, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(input_feature_channel_num), nn.ReLU(), nn.Conv2d(input_feature_channel_num, input_feature_channel_num, kernel_size=1, stride=1, padding=0, bias=True))

    def forward(self, x):
        return self.predictor(x)


class DeformGNN(nn.Module):

    def __init__(self, state_dim=256, feature_channel_num=128, out_dim=2, layer_num=8, scale_pos=False, use_att=False):
        super(DeformGNN, self).__init__()
        self.state_dim = state_dim
        self.feature_channel_num = feature_channel_num
        self.out_dim = out_dim
        self.layer_num = layer_num
        self.scale_pos = scale_pos
        self.use_att = use_att
        self.gnn = GCN(state_dim=self.state_dim, feature_dim=self.feature_channel_num + 2, out_dim=self.out_dim, layer_num=self.layer_num)
        None
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.01)
                nn.init.constant_(m.bias, 0)
        None

    def forward(self, features, base_point, base_normalized_point_adjacent, base_point_mask, old_point_mask=None, update_last=False, scale=0.1):
        """
        pred_polys: in scale [0,1]
        """
        out_dict = {}
        shape = features.shape
        hw = shape[-2:]
        tmp_features = features.reshape(shape[0], shape[1], -1)
        tmp_features = tmp_features.permute(0, 2, 1).contiguous()
        cnn_feature = self.interpolated_sum([tmp_features], base_point, [hw])
        input_feature = torch.cat((cnn_feature, base_point), 2)
        gcn_pred = self.gnn.forward(input_feature, base_normalized_point_adjacent)
        enforced_gcn_pred = gcn_pred
        gcn_pred_poly = base_point + enforced_gcn_pred[:, :, :2] * base_point_mask.squeeze(1)
        laplacian_coord_1 = base_point - torch.bmm(base_normalized_point_adjacent, base_point)
        laplacian_coord_2 = gcn_pred_poly - torch.bmm(base_normalized_point_adjacent, gcn_pred_poly)
        laplacian_energy = ((laplacian_coord_2 - laplacian_coord_1) ** 2 + 1e-10).sum(-1).sqrt()
        laplacian_energy = laplacian_energy.mean(dim=-1)
        out_dict['laplacian_energy'] = laplacian_energy
        out_dict['pred_points'] = gcn_pred_poly
        out_dict['gcn_pred_points'] = gcn_pred
        return out_dict

    def interpolated_sum(self, cnns, coords, grids, grid_multiplier=None):
        X = coords[:, :, 0]
        Y = coords[:, :, 1]
        cnn_outs = []
        for i in range(len(grids)):
            grid = grids[i]
            if grid_multiplier is None:
                Xs = X * grid[1]
            else:
                Xs = X * grid_multiplier[i][1]
            X0 = torch.floor(Xs)
            X1 = X0 + 1
            if grid_multiplier is None:
                Ys = Y * grid[0]
            else:
                Ys = Y * grid_multiplier[i][1]
            Y0 = torch.floor(Ys)
            Y1 = Y0 + 1
            w_00 = (X1 - Xs) * (Y1 - Ys)
            w_01 = (X1 - Xs) * (Ys - Y0)
            w_10 = (Xs - X0) * (Y1 - Ys)
            w_11 = (Xs - X0) * (Ys - Y0)
            X0 = torch.clamp(X0, 0, grid[1] - 1)
            X1 = torch.clamp(X1, 0, grid[1] - 1)
            Y0 = torch.clamp(Y0, 0, grid[0] - 1)
            Y1 = torch.clamp(Y1, 0, grid[0] - 1)
            N1_id = X0 + Y0 * grid[1]
            N2_id = X0 + Y1 * grid[1]
            N3_id = X1 + Y0 * grid[1]
            N4_id = X1 + Y1 * grid[1]
            M_00 = helpers.gather_feature(N1_id, cnns[i])
            M_01 = helpers.gather_feature(N2_id, cnns[i])
            M_10 = helpers.gather_feature(N3_id, cnns[i])
            M_11 = helpers.gather_feature(N4_id, cnns[i])
            cnn_out = w_00.unsqueeze(2) * M_00 + w_01.unsqueeze(2) * M_01 + w_10.unsqueeze(2) * M_10 + w_11.unsqueeze(2) * M_11
            cnn_outs.append(cnn_out)
        concat_features = torch.cat(cnn_outs, dim=2)
        return concat_features


class CheckCondition(Function):

    @staticmethod
    def forward(ctx, grid_bxkx3x2, img_pos_bxnx2, grid_mask=None):
        n_batch = grid_bxkx3x2.shape[0]
        n_pixel = img_pos_bxnx2.shape[1]
        condition_bxnx1 = -torch.ones(n_batch, n_pixel, 1).float()
        top_left, _ = torch.min(grid_bxkx3x2, dim=2)
        top_left = top_left.unsqueeze(2)
        bottom_right, _ = torch.max(grid_bxkx3x2, dim=2)
        bottom_right = bottom_right.unsqueeze(2)
        bbox_bxkx2x2 = torch.cat([top_left, bottom_right], dim=2)
        check_condition.forward(grid_bxkx3x2, img_pos_bxnx2, condition_bxnx1, bbox_bxkx2x2)
        return condition_bxnx1

    @staticmethod
    def backward(ctx, condition_bxnx1):
        return None, None, None


check_condition_f_bbox = CheckCondition.apply


class MeanFeatureGather(torch.autograd.Function):

    @staticmethod
    def forward(ctx, feature_map, condition, grid_num):
        device = feature_map.device
        if feature_map.dim() != 3:
            raise ValueError('expect feature_map to have exactly 3 dimensions')
        if condition.dim() != 2:
            raise ValueError('expect condition to have exactly 2 dimensions')
        batch_size, _, feature_channel_num = feature_map.shape
        assert feature_channel_num <= 642
        mean_feature = torch.zeros(batch_size, grid_num, feature_channel_num, device=feature_map.device)
        grid_size = torch.zeros(batch_size, grid_num, device=feature_map.device)
        condition = condition.contiguous()
        if feature_map.shape[1] == 100000000:
            for i_batch in range(feature_map.shape[0]):
                tmp_feature = feature_map[i_batch].unsqueeze(0).contiguous()
                mean_feature_func.forward_cuda(mean_feature[i_batch].unsqueeze(0), grid_size[i_batch].unsqueeze(0), tmp_feature.float(), condition[i_batch].unsqueeze(0).float())
        else:
            feature_map = feature_map.contiguous()
            mean_feature_func.forward_cuda(mean_feature, grid_size, feature_map.float(), condition.float())
        ctx.save_for_backward(grid_size, condition)
        return mean_feature, grid_size

    @staticmethod
    def backward(ctx, grad_mean_feature, grad_grid_size):
        grid_size, condition = ctx.saved_tensors
        invalid_mask = condition < 0
        filtered_condition = condition.clone().long()
        filtered_condition[invalid_mask] = 0
        if filtered_condition.shape[-1] == 100000000:
            feature_map_grad_list = []
            for i_batch in range(filtered_condition.shape[0]):
                local_condition = filtered_condition[i_batch].unsqueeze(0)
                feature_map_weights = torch.gather(input=grid_size[i_batch].unsqueeze(0), dim=1, index=local_condition).detach()
                tmp_shape = local_condition.shape
                feature_channel_num = grad_mean_feature[i_batch].shape[-1]
                tmp_condition = local_condition.unsqueeze(-1).expand(tmp_shape[0], tmp_shape[1], feature_channel_num)
                feature_map_grad = torch.gather(input=grad_mean_feature[i_batch].unsqueeze(0), dim=1, index=tmp_condition).detach()
                feature_map_grad = feature_map_grad / feature_map_weights.reshape(tmp_shape[0], tmp_shape[1], 1)
                feature_map_grad[invalid_mask[i_batch].unsqueeze(0)] = 0
                feature_map_grad_list.append(feature_map_grad)
            feature_map_grad = torch.cat(feature_map_grad_list, dim=0)
        else:
            feature_map_weights = torch.gather(input=grid_size, dim=1, index=filtered_condition).detach()
            tmp_shape = filtered_condition.shape
            feature_channel_num = grad_mean_feature.shape[-1]
            tmp_condition = filtered_condition.unsqueeze(-1).expand(tmp_shape[0], tmp_shape[1], feature_channel_num)
            feature_map_grad = torch.gather(input=grad_mean_feature, dim=1, index=tmp_condition).detach()
            feature_map_grad = feature_map_grad / feature_map_weights.reshape(tmp_shape[0], tmp_shape[1], 1)
            feature_map_grad[invalid_mask] = 0
        return feature_map_grad, None, None


get_grid_mean_feature = MeanFeatureGather.apply


class LatticeVariance(nn.Module):

    def __init__(self, output_row, output_column, device, sigma=0.001, add_seg=False, mask_coef=0.3):
        super(LatticeVariance, self).__init__()
        self.output_row = output_row
        self.output_column = output_column
        self.sigma = torch.zeros(1)
        self.sigma[0] = sigma
        self.device = device
        output_x_pos = np.zeros((output_row, output_column))
        output_y_pos = np.zeros((output_row, output_column))
        for i in range(output_row):
            output_x_pos[i] = np.arange(output_column, dtype=np.float) / output_column
        for i in range(output_column):
            output_y_pos[:, i] = np.arange(output_row, dtype=np.float) / output_row
        output_x_pos += 1.0 / output_column * 0.5
        output_y_pos += 1.0 / output_row * 0.5
        output_x_pos = torch.from_numpy(output_x_pos).float().unsqueeze(-1)
        output_y_pos = torch.from_numpy(output_y_pos).float().unsqueeze(-1)
        output_pos = torch.cat((output_x_pos, output_y_pos), -1)
        self.register_buffer('output_pos', output_pos)
        self.add_seg = add_seg
        self.mask_coef = mask_coef

    def forward(self, grid_pos=None, img_fea=None, base_triangle2point=None, base_area_mask=None, base_triangle_mask=None, area_normalize=(20, 20), semantic_mask=None, inference=False, img_pos=None, grid_size=None, grid_mask=None):
        n_batch = grid_pos.shape[0]
        gather_input = grid_pos.unsqueeze(2).expand(n_batch, grid_pos.shape[1], 3, 2)
        gather_index = base_triangle2point.unsqueeze(-1).expand(n_batch, base_triangle2point.shape[1], 3, 2).long()
        lattice = torch.gather(input=gather_input, dim=1, index=gather_index)

        def local_check_f(query):
            return check_condition_f_bbox(lattice, query)
        lab_img_fea = img_fea
        if self.add_seg:
            assert not semantic_mask is None
            lab_img_fea = torch.cat((lab_img_fea, semantic_mask * self.mask_coef), dim=-1)
        variance_bxp, reconstruct_fea, condition = self.variance(lattice, lab_img_fea, inference=inference, img_pos=img_pos, grid_size=grid_size)
        ret = {'condition': condition}
        ret['check_condition'] = local_check_f
        ret['grid_lattice'] = lattice
        if inference:
            return ret
        area_variance = self.area_variance(lattice, base_area_mask, area_normalize)
        ret['variance'] = variance_bxp.mean(dim=-1)
        ret['area_variance'] = area_variance
        reconstruct_loss = self.reconstruct(reconstruct_fea, lab_img_fea)
        ret['reconstruct_loss'] = reconstruct_loss
        ret['reconstruct_img'] = reconstruct_fea
        return ret

    def reconstruct(self, reconstruct_fea, img_fea):
        batch_size, h, w, c = img_fea.shape
        reconstruct_fea = reconstruct_fea.reshape(batch_size, h, w, c)
        reconstruct_loss = torch.abs(reconstruct_fea - img_fea).reshape(batch_size, -1).mean(dim=-1)
        return reconstruct_loss

    def variance(self, lattice, img_fea, inference, img_pos=None, grid_size=None):
        grid_num = lattice.shape[1]
        n_batch = lattice.shape[0]
        if img_pos is None:
            img_pos = self.output_pos.unsqueeze(0).expand(n_batch, self.output_row, self.output_column, 2)
        img_pos = img_pos.view(n_batch, -1, 2)
        img_pos = img_pos.contiguous()
        condition = check_condition_f_bbox(lattice, img_pos)
        if inference:
            return None, None, condition
        img_fea = img_fea.reshape(n_batch, -1, img_fea.shape[-1])
        grid_fea, _ = get_grid_mean_feature(img_fea, condition.squeeze(-1), grid_num)
        grid_fea = grid_fea.detach()
        max_grid_size = max(grid_size[0] - 1, grid_size[1] - 1)
        sigma = 0.001 * 20 / max_grid_size
        new_sigma = torch.zeros(1, dtype=torch.float, device=lattice.device)
        new_sigma[0] = sigma
        variance_bxp, reconstruct_img = line_variance_topk(img_fea, grid_fea, lattice, img_pos, new_sigma)
        return variance_bxp, reconstruct_img, condition

    def convertrgb2lab(self, batch_rgb):
        n_batch = batch_rgb.shape[0]
        batch_rgb_np = batch_rgb.cpu().numpy()
        batch_rgb_np = batch_rgb_np.astype(np.uint8)
        lab_list = []
        for i in range(n_batch):
            rgb = batch_rgb_np[i]
            lab = skimage.color.rgb2lab(rgb)
            lab_list.append(lab)
        lab = np.stack(lab_list, axis=0)
        return torch.from_numpy(lab).float()

    def area_variance(self, lattice, base_area_mask, area_normalize=(20, 20)):
        tmp_lattice = lattice * torch.tensor([area_normalize[1], area_normalize[0]]).float().reshape(1, 1, 1, 2)
        A = tmp_lattice[:, :, 0, :]
        B = tmp_lattice[:, :, 1, :]
        C = tmp_lattice[:, :, 2, :]
        area1 = (A[..., 1] + B[..., 1]) * (B[..., 0] - A[..., 0]) / 2
        area2 = (B[..., 1] + C[..., 1]) * (C[..., 0] - B[..., 0]) / 2
        area3 = (C[..., 1] + A[..., 1]) * (A[..., 0] - C[..., 0]) / 2
        area = area1 + area2 + area3
        area = area.view(area.shape[0], -1)
        area = area * base_area_mask
        var_area = torch.var(area, dim=-1)
        return var_area


def tt(s, batch=1):
    global t0
    torch.cuda.synchronize()
    None
    t0 = time.time()


class DeformableGrid(nn.Module):

    def __init__(self, args, device):
        super(DeformableGrid, self).__init__()
        self.debug = args.debug
        self.device = device
        self.resolution = args.resolution
        self.state_dim = args.state_dim
        self.grid_size = args.grid_size
        self.feature_channel_num = args.feature_channel_num
        self.deform_layer_num = args.deform_layer_num
        self.mlp_expansion = args.mlp_expansion
        self.concat_channels = args.concat_channels
        self.final_dim = args.final_dim
        self.grid_type = args.grid_type
        self.add_mask_variance = args.add_mask_variance
        self.dataset = args.dataset
        self.encoder_backbone = args.encoder_backbone
        self.update_last = args.update_last
        self.grid_pos_layer = args.grid_pos_layer
        self.mask_coef = args.mask_coef
        try:
            self.feature_aggregate_type = args.feature_aggregate_type
        except:
            self.feature_aggregate_type = 'mean'
            None
        self.gamma = args.gamma
        self.sigma = args.sigma
        self.input_channel = 3
        self.use_final_encoder = self.resolution[0] == 512
        if self.encoder_backbone == 'affinity_net':
            self.encoder = MyEncoder(nr_channel=self.feature_channel_num, input_channel=self.input_channel)
            self.out_feature_channel_num = self.feature_channel_num
            self.final_dim = self.out_feature_channel_num
        elif self.encoder_backbone == 'deeplab':
            self.encoder = DeepLabResnet(input_channel=self.input_channel, concat_channels=self.concat_channels, final_dim=self.final_dim, final_resolution=self.resolution, use_final=self.use_final_encoder, update_last=self.update_last)
            self.out_feature_channel_num = self.final_dim
        elif self.encoder_backbone == 'simplenn':
            self.encoder = SimpleEncoder()
            self.out_feature_channel_num = 512
            self.final_dim = 512
        if self.grid_pos_layer == 5:
            self.deformer_encoder = NetHead(self.final_dim, self.final_dim)
        else:
            self.deformer_encoder = NetHead(self.grid_pos_layer * self.concat_channels, self.final_dim)
        self.deformer = superpixel_grid.DeformGNN(state_dim=self.state_dim, feature_channel_num=self.out_feature_channel_num, out_dim=2, layer_num=self.deform_layer_num)
        self.superpixel = LatticeVariance(self.resolution[0], self.resolution[1], sigma=self.sigma, device=device, add_seg=self.add_mask_variance, mask_coef=self.mask_coef)

    def forward(self, net_input=None, base_point=None, base_normalized_point_adjacent=None, base_point_mask=None, base_triangle2point=None, base_area_mask=None, base_triangle_mask=None, crop_gt=None, inference=False, timing=False, grid_size=20):
        sub_batch_size = net_input.shape[0]
        device = net_input.device
        variance = torch.zeros(sub_batch_size, device=device)
        laplacian_loss = torch.zeros(sub_batch_size, device=device)
        area_variance = torch.zeros(sub_batch_size, device=device)
        reconstruct_loss = torch.zeros(sub_batch_size, device=device)
        if timing:
            tt('start', sub_batch_size)
        final_features, final_cat_features = self.encoder(deepcopy(net_input))
        if self.grid_pos_layer == 5:
            deformer_feature = self.deformer_encoder(final_features)
        else:
            deformer_feature = self.deformer_encoder(final_cat_features[:, :self.concat_channels * self.grid_pos_layer, :, :])
        output = self.deformer(deformer_feature, base_point, base_normalized_point_adjacent, base_point_mask)
        if timing:
            tt('get deformation', sub_batch_size)
        pred_points = output['pred_points']
        if timing:
            tt('get curve prediction', sub_batch_size)
        n_row_area_normalize = self.grid_size[0]
        n_column_area_normalize = self.grid_size[1]
        if not inference:
            if self.add_mask_variance:
                tmp_gt_mask = deepcopy(crop_gt)
                tmp_gt_mask = tmp_gt_mask.long()
                gt_mask = helpers.gtmask2onehot(tmp_gt_mask).permute(0, 2, 3, 1)
                superpixel_ret = self.superpixel(grid_pos=pred_points, img_fea=net_input[:, :3, ...].permute(0, 2, 3, 1), base_triangle2point=base_triangle2point, base_area_mask=base_area_mask, base_triangle_mask=base_triangle_mask, area_normalize=(n_row_area_normalize, n_column_area_normalize), semantic_mask=gt_mask, inference=inference, grid_size=self.grid_size)
            else:
                superpixel_ret = self.superpixel(grid_pos=pred_points, img_fea=net_input[:, :3, ...].permute(0, 2, 3, 1), base_triangle2point=base_triangle2point, base_area_mask=base_area_mask, base_triangle_mask=base_triangle_mask, area_normalize=(n_row_area_normalize, n_column_area_normalize), inference=inference, grid_size=self.grid_size)
        else:
            superpixel_ret = defaultdict(None)
        if not inference:
            condition = superpixel_ret['condition']
            laplacian_loss += output['laplacian_energy']
            variance += superpixel_ret['variance']
            area_variance += superpixel_ret['area_variance']
            reconstruct_loss += superpixel_ret['reconstruct_loss']
        else:
            condition = None
        return condition, laplacian_loss, variance, area_variance, reconstruct_loss, pred_points


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ClassifierModule,
     lambda: ([], {'dilation_series': [4, 4], 'padding_series': [4, 4], 'n_classes': 4}),
     lambda: ([torch.rand([4, 2048, 64, 64])], {}),
     False),
    (EncoderHead,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 128, 64, 64])], {}),
     True),
    (GraphConvolution,
     lambda: ([], {'state_dim': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     True),
    (GraphResConvolution,
     lambda: ([], {'state_dim': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     True),
    (MyEncoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (MyResBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (NetHead,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 128, 64, 64])], {}),
     True),
    (PPM,
     lambda: ([], {'in_dim': 4, 'reduction_dim': 4, 'bins': [4, 4], 'BatchNorm': _mock_layer}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SimpleEncoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
]

class Test_fidler_lab_defgrid_release(_paritybench_base):
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

