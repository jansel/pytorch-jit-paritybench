import sys
_module = sys.modules[__name__]
del sys
camvid_bbox = _module
camvid_bbox_rename = _module
cityscapes_bbox = _module
loss_smooth = _module
miou_expand = _module
split_dataset_train_val = _module
transform = _module
visdom_offline_data = _module
performance_table = _module
semseg = _module
caffe_pb2 = _module
dataloader = _module
ade20k_loader = _module
camvid_loader = _module
camvid_lrn_loader = _module
cityscapes_loader = _module
folder2lmdb = _module
freespace_loader = _module
freespacepred_loader = _module
movingmnist_loader = _module
segmpred_loader = _module
tfrecords_loader = _module
utils = _module
yolodataset_loader = _module
loss = _module
metrics = _module
EDANet = _module
modelloader = _module
bisenet = _module
deconvnet = _module
deeplab_resnet = _module
deeplabv3 = _module
drn = _module
drn_a_irb = _module
drn_a_mt = _module
drn_a_refine = _module
drn_pred = _module
duc_hdc = _module
enet = _module
enetv2 = _module
erfnet = _module
fast_segnet = _module
fc_densenet = _module
fcn = _module
fcn_mobilenet = _module
fcn_resnet = _module
fcn_shufflenet = _module
frrn = _module
gcn = _module
lrn = _module
pspnet = _module
segnet = _module
segnet_unet = _module
sqnet = _module
unet = _module
utils = _module
netloader = _module
resnet = _module
resnet_ibn_a = _module
resnet_ibn_b = _module
pytorch_modelsize = _module
schedulers = _module
flops_benchmark = _module
get_class_weights = _module
model_info_eval = _module
visualize = _module
yoloLoss = _module
train = _module
train_lrn = _module
train_mt = _module
train_pred = _module
validate = _module
validate_mt = _module
validate_pred = _module
visualize_test = _module

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


import torch.nn.functional as F


import torch


import torch.nn as nn


from torch.autograd import Variable


import numpy as np


from torch import nn


import math


import torch.utils.model_zoo as model_zoo


from torch.utils import model_zoo


import torch.optim


import torch.nn.init as init


from collections import OrderedDict


from torch.nn import init


from scipy import misc


import functools


import torch.optim as optim


from functools import reduce


class DownsamplerBlock(nn.Module):

    def __init__(self, ninput, noutput):
        super(DownsamplerBlock, self).__init__()
        self.ninput = ninput
        self.noutput = noutput
        if self.ninput < self.noutput:
            self.conv = nn.Conv2d(ninput, noutput - ninput, kernel_size=3,
                stride=2, padding=1)
            self.pool = nn.MaxPool2d(2, stride=2)
        else:
            self.conv = nn.Conv2d(ninput, noutput, kernel_size=3, stride=2,
                padding=1)
        self.bn = nn.BatchNorm2d(noutput)

    def forward(self, x):
        if self.ninput < self.noutput:
            output = torch.cat([self.conv(x), self.pool(x)], 1)
        else:
            output = self.conv(x)
        output = self.bn(output)
        return F.relu(output)


class EDABlock(nn.Module):

    def __init__(self, ninput, dilated, k=40, dropprob=0.02):
        super(EDABlock, self).__init__()
        self.conv1x1 = nn.Conv2d(ninput, k, kernel_size=1)
        self.bn0 = nn.BatchNorm2d(k)
        self.conv3x1_1 = nn.Conv2d(k, k, kernel_size=(3, 1), padding=(1, 0))
        self.conv1x3_1 = nn.Conv2d(k, k, kernel_size=(1, 3), padding=(0, 1))
        self.bn1 = nn.BatchNorm2d(k)
        self.conv3x1_2 = nn.Conv2d(k, k, (3, 1), stride=1, padding=(dilated,
            0), dilation=dilated)
        self.conv1x3_2 = nn.Conv2d(k, k, (1, 3), stride=1, padding=(0,
            dilated), dilation=dilated)
        self.bn2 = nn.BatchNorm2d(k)
        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, x):
        input = x
        output = self.conv1x1(x)
        output = self.bn0(output)
        output = F.relu(output)
        output = self.conv3x1_1(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)
        output = self.conv3x1_2(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)
        output = F.relu(output)
        if self.dropout.p != 0:
            output = self.dropout(output)
        output = torch.cat([output, input], 1)
        return output


class EDANet(nn.Module):

    def __init__(self, n_classes=20, pretrained=False):
        super(EDANet, self).__init__()
        self.layers = nn.ModuleList()
        self.dilation1 = [1, 1, 1, 2, 2]
        self.dilation2 = [2, 2, 4, 4, 8, 8, 16, 16]
        self.layers.append(DownsamplerBlock(3, 15))
        self.layers.append(DownsamplerBlock(15, 60))
        for i in range(5):
            self.layers.append(EDABlock(60 + 40 * i, self.dilation1[i]))
        self.layers.append(DownsamplerBlock(260, 130))
        for j in range(8):
            self.layers.append(EDABlock(130 + 40 * j, self.dilation2[j]))
        self.project_layer = nn.Conv2d(450, n_classes, kernel_size=1)
        self.weights_init()

    def weights_init(self):
        for idx, m in enumerate(self.modules()):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                m.weight.data.normal_(0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x):
        output = x
        for layer in self.layers:
            output = layer(output)
        output = self.project_layer(output)
        output = F.upsample(output, scale_factor=8, mode='bilinear')
        if not self.training:
            output = F.upsample(output, scale_factor=2, mode='bilinear')
        return output


class resnet18(nn.Module):

    def __init__(self, pretrained=True):
        super(resnet18, self).__init__()
        self.features = models.resnet18(pretrained=pretrained)
        self.conv1 = self.features.conv1
        self.bn1 = self.features.bn1
        self.relu = self.features.relu
        self.maxpool1 = self.features.maxpool
        self.layer1 = self.features.layer1
        self.layer2 = self.features.layer2
        self.layer3 = self.features.layer3
        self.layer4 = self.features.layer4

    def forward(self, input):
        x = self.conv1(input)
        x = self.relu(self.bn1(x))
        x = self.maxpool1(x)
        feature1 = self.layer1(x)
        feature2 = self.layer2(feature1)
        feature3 = self.layer3(feature2)
        feature4 = self.layer4(feature3)
        tail = torch.mean(feature4, 3, keepdim=True)
        tail = torch.mean(tail, 2, keepdim=True)
        return feature3, feature4, tail


class resnet101(nn.Module):

    def __init__(self, pretrained=True):
        super(resnet101, self).__init__()
        self.features = models.resnet101(pretrained=pretrained)
        self.conv1 = self.features.conv1
        self.bn1 = self.features.bn1
        self.relu = self.features.relu
        self.maxpool1 = self.features.maxpool
        self.layer1 = self.features.layer1
        self.layer2 = self.features.layer2
        self.layer3 = self.features.layer3
        self.layer4 = self.features.layer4

    def forward(self, input):
        x = self.conv1(input)
        x = self.relu(self.bn1(x))
        x = self.maxpool1(x)
        feature1 = self.layer1(x)
        feature2 = self.layer2(feature1)
        feature3 = self.layer3(feature2)
        feature4 = self.layer4(feature3)
        tail = torch.mean(feature4, 3, keepdim=True)
        tail = torch.mean(tail, 2, keepdim=True)
        return feature3, feature4, tail


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2,
        padding=1):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=
            kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Spatial_path(nn.Module):

    def __init__(self):
        """
        Spatial Path is combined by 3 blocks including Conv+BN+ReLU, and here every block is 2 stride
        """
        super(Spatial_path, self).__init__()
        self.convblock1 = ConvBlock(in_channels=3, out_channels=64)
        self.convblock2 = ConvBlock(in_channels=64, out_channels=128)
        self.convblock3 = ConvBlock(in_channels=128, out_channels=256)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        return x


class AttentionRefinementModule(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(AttentionRefinementModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()
        self.in_channels = in_channels

    def forward(self, input):
        x = torch.mean(input, 3, keepdim=True)
        x = torch.mean(x, 2, keepdim=True)
        assert self.in_channels == x.size(1
            ), 'in_channels and out_channels should all be {}'.format(x.size(1)
            )
        x = self.conv(x)
        x = self.sigmoid(x)
        x = torch.mul(input, x)
        return x


class FeatureFusionModule(nn.Module):

    def __init__(self, num_classes, in_channels):
        super(FeatureFusionModule, self).__init__()
        self.in_channels = in_channels
        self.convblock = ConvBlock(in_channels=self.in_channels,
            out_channels=num_classes, stride=1)
        self.conv1 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_1, input_2):
        x = torch.cat((input_1, input_2), dim=1)
        assert self.in_channels == x.size(1
            ), 'in_channels of ConvBlock should be {}'.format(x.size(1))
        feature = self.convblock(x)
        x = torch.mean(feature, 3, keepdim=True)
        x = torch.mean(x, 2, keepdim=True)
        x = self.relu(self.conv1(x))
        x = self.sigmoid(self.relu(x))
        x = torch.mul(feature, x)
        x = torch.add(x, feature)
        return x


def Context_path(name, pretrained=False):
    if name == 'resnet18':
        return resnet18(pretrained=pretrained)
    elif name == 'resnet101':
        return resnet101(pretrained=pretrained)


class BiSeNet(nn.Module):

    def __init__(self, n_classes=21, pretrained=True, context_path='resnet18'):
        super(BiSeNet, self).__init__()
        self.n_classes = n_classes
        self.saptial_path = Spatial_path()
        self.context_path = Context_path(name=context_path, pretrained=
            pretrained)
        if context_path == 'resnet18':
            self.attention_refinement_module1 = AttentionRefinementModule(
                256, 256)
            self.attention_refinement_module2 = AttentionRefinementModule(
                512, 512)
        elif context_path == 'resnet101':
            self.attention_refinement_module1 = AttentionRefinementModule(
                1024, 1024)
            self.attention_refinement_module2 = AttentionRefinementModule(
                2048, 2048)
        if context_path == 'resnet18':
            self.feature_fusion_module = FeatureFusionModule(self.n_classes,
                in_channels=1024)
        elif context_path == 'resnet101':
            self.feature_fusion_module = FeatureFusionModule(self.n_classes,
                in_channels=3328)
        self.conv = nn.Conv2d(in_channels=self.n_classes, out_channels=self
            .n_classes, kernel_size=1)

    def forward(self, input):
        input_size = input.size()
        sx = self.saptial_path(input)
        cx1, cx2, tail = self.context_path(input)
        cx1 = self.attention_refinement_module1(cx1)
        cx2 = self.attention_refinement_module2(cx2)
        cx2 = torch.mul(cx2, tail)
        cx1 = F.upsample_bilinear(cx1, (input_size[2] // 8, input_size[3] // 8)
            )
        cx2 = F.upsample_bilinear(cx2, (input_size[2] // 8, input_size[3] // 8)
            )
        cx = torch.cat((cx1, cx2), dim=1)
        result = self.feature_fusion_module(sx, cx)
        result = F.upsample_bilinear(result, scale_factor=8)
        result = self.conv(result)
        return result


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
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


def deconv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=
        stride, padding=1, bias=False, output_padding=1)


class DeconvBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None):
        super(DeconvBasicBlock, self).__init__()
        if stride == 1:
            self.conv1 = conv3x3(inplanes, planes, stride)
        else:
            self.conv1 = deconv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.upsample is not None:
            residual = self.upsample(x)
        None
        None
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1,
            bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
            stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
            kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self, x):
        shortcut = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        if self.downsample is not None:
            shortcut = self.downsample(x)
        out += shortcut
        out = self.relu(out)
        return out


class DeconvBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, upsample=None):
        super(DeconvBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1,
            bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        if stride == 1:
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=
                3, stride=stride, bias=False, padding=1)
        else:
            self.conv2 = nn.ConvTranspose2d(out_channels, out_channels,
                kernel_size=3, stride=stride, bias=False, padding=1,
                output_padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
            kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.upsample = upsample

    def forward(self, x):
        shortcut = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        if self.upsample is not None:
            shortcut = self.upsample(x)
        out += shortcut
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, downblock, upblock, num_layers, n_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1,
            return_indices=True)
        self.dlayer1 = self._make_downlayer(downblock, 64, num_layers[0])
        self.dlayer2 = self._make_downlayer(downblock, 128, num_layers[1],
            stride=2)
        self.dlayer3 = self._make_downlayer(downblock, 256, num_layers[2],
            stride=2)
        self.dlayer4 = self._make_downlayer(downblock, 512, num_layers[3],
            stride=2)
        self.uplayer1 = self._make_up_block(upblock, 512, num_layers[3],
            stride=2)
        self.uplayer2 = self._make_up_block(upblock, 256, num_layers[2],
            stride=2)
        self.uplayer3 = self._make_up_block(upblock, 128, num_layers[1],
            stride=2)
        self.uplayer4 = self._make_up_block(upblock, 64, num_layers[0])

    def _make_downlayer(self, block, init_channels, num_layer, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != init_channels * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.in_channels, 
                init_channels * block.expansion, kernel_size=1, stride=
                stride, bias=False), nn.BatchNorm2d(init_channels * block.
                expansion))
        layers = []
        layers.append(block(self.in_channels, init_channels, stride,
            downsample))
        self.in_channels = init_channels * block.expansion
        for i in range(1, num_layer):
            layers.append(block(self.in_channels, init_channels))
        return nn.Sequential(*layers)

    def _make_up_block(self, block, init_channels, num_layer, stride=1):
        upsample = None
        None
        None
        if stride != 1 or self.in_channels != init_channels * block.expansion:
            upsample = nn.Sequential(nn.ConvTranspose2d(self.in_channels, 
                init_channels // block.expansion, kernel_size=1, stride=
                stride, bias=False, output_padding=1), nn.BatchNorm2d(
                init_channels // block.expansion))
        layers = []
        for i in range(1, num_layer):
            layers.append(block(self.in_channels, init_channels))
        layers.append(block(self.in_channels, init_channels // block.
            expansion, stride, upsample))
        self.in_channels = init_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        unpool_shape = x.size()
        x, pool_indices = self.maxpool(x)
        x = self.dlayer1(x)
        None
        x = self.dlayer2(x)
        None
        x = self.dlayer3(x)
        None
        x = self.dlayer4(x)
        None
        x = self.uplayer1(x)
        None
        x = self.uplayer2(x)
        None
        x = self.uplayer3(x)
        None
        x = self.uplayer4(x)
        None
        return x


affine_par = True


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
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

    def __init__(self, inplanes, planes, stride=1, dilation_=1, downsample=None
        ):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=
            stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        padding = 1
        if dilation_ == 2:
            padding = 2
        elif dilation_ == 4:
            padding = 4
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
            padding=padding, bias=False, dilation=dilation_)
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


class Classifier_Module(nn.Module):

    def __init__(self, dilation_series, padding_series, NoLabels):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(nn.Conv2d(2048, NoLabels, kernel_size=3,
                stride=1, padding=padding, dilation=dilation, bias=True))
        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, n_classes):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1,
            ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
            dilation__=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
            dilation__=4)
        self.layer5 = self._make_pred_layer(Classifier_Module, [6, 12, 18, 
            24], [6, 12, 18, 24], n_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation__=1):
        downsample = None
        if (stride != 1 or self.inplanes != planes * block.expansion or 
            dilation__ == 2 or dilation__ == 4):
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation_=
            dilation__, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation_=dilation__))
        return nn.Sequential(*layers)

    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels
        ):
        return block(dilation_series, padding_series, NoLabels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x


def outS(i):
    i = int(i)
    i = (i + 1) / 2
    i = int(np.ceil((i + 1) / 2.0))
    i = (i + 1) / 2
    return i


class MS_Deeplab(nn.Module):

    def __init__(self, block, n_classes):
        super(MS_Deeplab, self).__init__()
        self.Scale = ResNet(block, [3, 4, 23, 3], n_classes)

    def forward(self, x):
        input_size = x.size()[2]
        self.interp1 = nn.UpsamplingBilinear2d(size=(int(input_size * 0.75) +
            1, int(input_size * 0.75) + 1))
        self.interp2 = nn.UpsamplingBilinear2d(size=(int(input_size * 0.5) +
            1, int(input_size * 0.5) + 1))
        self.interp3 = nn.UpsamplingBilinear2d(size=(outS(input_size), outS
            (input_size)))
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
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
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

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None
        ):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=
            stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        for i in self.bn1.parameters():
            i.requires_grad = False
        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
            padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
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


class ResNet(nn.Module):

    def __init__(self, block, layers, n_classes):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1,
            ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
            dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
            dilation=4)
        self.layer5 = self._make_pred_layer(ASPP_Classifier_Module, [6, 12,
            18, 24], [6, 12, 18, 24], n_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if (stride != 1 or self.inplanes != planes * block.expansion or 
            dilation == 2 or dilation == 4):
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=
            dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def _make_pred_layer(self, block, dilation_series, padding_series,
        num_classes):
        return block(dilation_series, padding_series, num_classes)

    def forward(self, x):
        x_size = x.size()[2:]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = F.upsample_bilinear(x, x_size)
        return x


def conv3x3_asymmetric(in_planes, out_planes, stride=1, padding=1, dilation=1):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=(3, 1
        ), stride=stride, padding=(padding, 0), bias=False, dilation=
        dilation), nn.Conv2d(out_planes, out_planes, kernel_size=(1, 3),
        stride=1, padding=(0, padding), bias=False, dilation=dilation))


class BasicBlock_asymmetric(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
        dilation=(1, 1), residual=True):
        super(BasicBlock_asymmetric, self).__init__()
        self.conv1 = conv3x3_asymmetric(inplanes, planes, stride, padding=
            dilation[0], dilation=dilation[0])
        self.bn1 = nn.BatchNorm2d(planes)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3_asymmetric(planes, planes, padding=dilation[1],
            dilation=dilation[1])
        self.bn2 = nn.BatchNorm2d(planes)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.downsample = downsample
        self.stride = stride
        self.residual = residual

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        if self.residual:
            out += residual
        out = self.relu(out)
        return out


class BasicBlock_asymmetric_ibn_a(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
        dilation=(1, 1), residual=True):
        super(BasicBlock_asymmetric_ibn_a, self).__init__()
        self.conv1 = conv3x3_asymmetric(inplanes, planes, stride, padding=
            dilation[0], dilation=dilation[0])
        self.bn1 = IBN(planes)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3_asymmetric(planes, planes, padding=dilation[1],
            dilation=dilation[1])
        self.bn2 = nn.BatchNorm2d(planes)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.downsample = downsample
        self.stride = stride
        self.residual = residual

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        if self.residual:
            out += residual
        out = self.relu(out)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
        dilation=(1, 1), residual=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, padding=dilation[0],
            dilation=dilation[0])
        self.bn1 = nn.BatchNorm2d(planes)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, padding=dilation[1], dilation=
            dilation[1])
        self.bn2 = nn.BatchNorm2d(planes)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.downsample = downsample
        self.stride = stride
        self.residual = residual

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        if self.residual:
            out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
        dilation=(1, 1), residual=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=dilation[1], bias=False, dilation=dilation[1])
        self.bn2 = nn.BatchNorm2d(planes)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
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


class DRN(nn.Module):

    def __init__(self, block, layers, n_classes=21, channels=(16, 32, 64, 
        128, 256, 512, 512, 512), out_map=False, out_middle=False,
        pool_size=28, arch='D'):
        super(DRN, self).__init__()
        self.inplanes = channels[0]
        self.out_map = out_map
        self.out_dim = channels[-1]
        self.out_middle = out_middle
        self.arch = arch
        if arch == 'C':
            self.conv1 = nn.Conv2d(3, channels[0], kernel_size=7, stride=1,
                padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(channels[0])
            self.relu = nn.ReLU(inplace=True)
            self.layer1 = self._make_layer(BasicBlock, channels[0], layers[
                0], stride=1)
            self.layer2 = self._make_layer(BasicBlock, channels[1], layers[
                1], stride=2)
        elif arch == 'D' or arch == 'E':
            self.layer0 = nn.Sequential(nn.Conv2d(3, channels[0],
                kernel_size=7, stride=1, padding=3, bias=False), nn.
                BatchNorm2d(channels[0]), nn.ReLU(inplace=True))
            self.layer1 = self._make_conv_layers(channels[0], layers[0],
                stride=1)
            self.layer2 = self._make_conv_layers(channels[1], layers[1],
                stride=2)
        self.layer3 = self._make_layer(block, channels[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, channels[3], layers[3], stride=2)
        self.layer5 = self._make_layer(block, channels[4], layers[4],
            dilation=2, new_level=False)
        self.layer6 = None if layers[5] == 0 else self._make_layer(block,
            channels[5], layers[5], dilation=4, new_level=False)
        if arch == 'C':
            self.layer7 = None if layers[6] == 0 else self._make_layer(
                BasicBlock, channels[6], layers[6], dilation=2, new_level=
                False, residual=False)
            self.layer8 = None if layers[7] == 0 else self._make_layer(
                BasicBlock, channels[7], layers[7], dilation=1, new_level=
                False, residual=False)
        elif arch == 'D' or arch == 'E':
            self.layer7 = None if layers[6] == 0 else self._make_conv_layers(
                channels[6], layers[6], dilation=2)
            self.layer8 = None if layers[7] == 0 else self._make_conv_layers(
                channels[7], layers[7], dilation=1)
        self.layer9 = None
        if arch == 'E':
            self.layer9 = AlignedResInception(in_planes=512)
        self.layer10 = None
        if self.layer10 is not None:
            self.out_dim = n_classes
            pass
        else:
            self.out_dim = 512 * block.expansion
            pass
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1,
        new_level=True, residual=True):
        assert dilation == 1 or dilation % 2 == 0
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
        layers = list()
        layers.append(block(self.inplanes, planes, stride, downsample,
            dilation=(1, 1) if dilation == 1 else (dilation // 2 if
            new_level else dilation, dilation), residual=residual))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, residual=residual,
                dilation=(dilation, dilation)))
        return nn.Sequential(*layers)

    def _make_conv_layers(self, channels, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([nn.Conv2d(self.inplanes, channels, kernel_size=
                3, stride=stride if i == 0 else 1, padding=dilation, bias=
                False, dilation=dilation), nn.BatchNorm2d(channels), nn.
                ReLU(inplace=True)])
            self.inplanes = channels
        return nn.Sequential(*modules)

    def _make_pred_layer(self, block, dilation_series, padding_series,
        n_classes, in_channels):
        return block(dilation_series, padding_series, n_classes, in_channels)

    def forward(self, x):
        if self.arch == 'C':
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        elif self.arch == 'D' or self.arch == 'E':
            x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        if self.layer6 is not None:
            x = self.layer6(x)
        if self.layer7 is not None:
            x = self.layer7(x)
        if self.layer8 is not None:
            x = self.layer8(x)
        if self.layer9 is not None:
            x = self.layer9(x)
        if self.layer10 is not None:
            x = self.layer10(x)
        return x


class DRN_A(nn.Module):

    def __init__(self, block, layers, n_classes=21, input_channel=3):
        self.inplanes = 64
        super(DRN_A, self).__init__()
        self.block = block
        self.layers = layers
        self.n_classes = n_classes
        self.input_channel = input_channel
        self.out_dim = 512 * block.expansion
        self.conv1 = nn.Conv2d(self.input_channel, 64, kernel_size=7,
            stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
            dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
            dilation=4)
        self.layer5 = self._make_pred_layer(ASPP_Classifier_Module, [6, 12,
            18, 24], [6, 12, 18, 24], n_classes, in_channels=512 * block.
            expansion)
        if self.layer5 is not None:
            self.out_dim = n_classes
            pass
        else:
            self.out_dim = 512 * block.expansion
            pass
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
            for i in downsample._modules['1'].parameters():
                i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=(dilation,
                dilation)))
        return nn.Sequential(*layers)

    def _make_pred_layer(self, block, dilation_series, padding_series,
        n_classes, in_channels):
        return block(dilation_series, padding_series, n_classes, in_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x


def drn_a_n(pretrained=False, depth_n=18, **kwargs):
    model = DRN_A(BasicBlock, [2 + depth_n - 18, 2, 2, 2], **kwargs)
    return model


def drn_a_asymmetric_n(pretrained=False, depth_n=18, **kwargs):
    model = DRN_A(BasicBlock_asymmetric, [2 + depth_n - 18, 2, 2, 2], **kwargs)
    return model


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = (1 - math.fabs(i / f - c)) * (1 - math.fabs(j /
                f - c))
    for c in range(1, w.size(0)):
        w[(c), (0), :, :] = w[(0), (0), :, :]


class DRNSeg(nn.Module):

    def __init__(self, model_name, n_classes, pretrained=False,
        use_torch_up=True, depth_n=-1, input_channel=3):
        super(DRNSeg, self).__init__()
        if model_name == 'drn_a_asymmetric_n':
            model = drn_a_asymmetric_n(pretrained=pretrained, n_classes=
                n_classes, depth_n=depth_n, input_channel=input_channel)
        elif model_name == 'drn_a_n':
            model = drn_a_n(pretrained=pretrained, n_classes=n_classes,
                depth_n=depth_n, input_channel=input_channel)
        else:
            model = eval(model_name)(pretrained=pretrained, n_classes=
                n_classes, input_channel=input_channel)
        self.base = model
        self.seg = nn.Conv2d(model.out_dim, n_classes, kernel_size=1)
        m = self.seg
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2.0 / n))
        m.bias.data.zero_()
        if use_torch_up:
            self.up = nn.UpsamplingBilinear2d(scale_factor=8)
        else:
            up = nn.ConvTranspose2d(n_classes, n_classes, 16, stride=8,
                padding=4, output_padding=0, groups=n_classes, bias=False)
            fill_up_weights(up)
            up.weight.requires_grad = False
            self.up = up
        if pretrained and n_classes == 19 and model_name == 'drn_d_38':
            model_checkpoint_path = os.path.expanduser(
                '~/.torch/models/drn_d_38_cityscapes.pth')
            if os.path.exists(model_checkpoint_path):
                model_dict = self.state_dict()
                pretrained_dict = torch.load(model_checkpoint_path,
                    map_location='cpu')
                new_dict = {}
                for k, v in pretrained_dict.items():
                    if k.find('base.') != -1:
                        new_k = str('base.' + 'layer' + k[k.find('.') + 1:])
                        if new_k not in model_dict.keys():
                            None
                        new_v = v
                        new_dict[new_k] = new_v
                    else:
                        new_k = k
                        new_v = v
                        new_dict[new_k] = new_v
                model_dict.update(new_dict)
                self.load_state_dict(model_dict)
        if pretrained and model_name == 'drn_a_18':
            model_checkpoint_path = os.path.expanduser(
                '~/GitHub/Quick/semseg/best.pth')
            if os.path.exists(model_checkpoint_path):
                model_dict = self.state_dict()
                pretrained_dict = torch.load(model_checkpoint_path,
                    map_location='cpu')
                model_dict_keys = model_dict.keys()
                new_dict = {}
                for k, v in pretrained_dict.items():
                    if 'base.{}'.format(k) in model_dict_keys:
                        new_k = 'base.{}'.format(k)
                        new_v = v
                        new_dict[new_k] = new_v
                    else:
                        pass
                new_dict = {k: v for k, v in new_dict.items() if k in
                    model_dict.keys()}
                model_dict.update(new_dict)
                self.load_state_dict(model_dict)

    def forward(self, x):
        x = self.base(x)
        x = self.seg(x)
        y = self.up(x)
        return y

    def optim_parameters(self, memo=None):
        for param in self.base.parameters():
            yield param
        for param in self.seg.parameters():
            yield param


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
        dilation=(1, 1), residual=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, padding=dilation[0],
            dilation=dilation[0])
        self.bn1 = nn.BatchNorm2d(planes)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, padding=dilation[1], dilation=
            dilation[1])
        self.bn2 = nn.BatchNorm2d(planes)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.downsample = downsample
        self.stride = stride
        self.residual = residual

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        if self.residual:
            out += residual
        out = self.relu(out)
        return out


class DRNSegIRB_A(nn.Module):

    def __init__(self, block, layers, n_classes=21):
        super(DRNSegIRB_A, self).__init__()
        self.inplanes = 64
        self.out_dim = 512 * block.expansion
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
            dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
            dilation=4)
        self.out_conv = nn.Conv2d(self.out_dim, n_classes, kernel_size=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=8)
        self.irbunit_1 = AlignedResInception(512)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
            for i in downsample._modules['1'].parameters():
                i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=(dilation,
                dilation)))
        return nn.Sequential(*layers)

    def forward(self, x):
        x_conv1 = self.conv1(x)
        x = self.bn1(x_conv1)
        x = self.relu(x)
        x_pool = self.maxpool(x)
        x_layer1 = self.layer1(x_pool)
        x_layer2 = self.layer2(x_layer1)
        x = self.layer3(x_layer2)
        x = self.layer4(x)
        x = self.irbunit_1(x)
        x = self.out_conv(x)
        x = self.up(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
        dilation=(1, 1), residual=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, padding=dilation[0],
            dilation=dilation[0])
        self.bn1 = nn.BatchNorm2d(planes)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, padding=dilation[1], dilation=
            dilation[1])
        self.bn2 = nn.BatchNorm2d(planes)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.downsample = downsample
        self.stride = stride
        self.residual = residual

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        if self.residual:
            out += residual
        out = self.relu(out)
        return out


class detnet_bottleneck(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, block_type='A'):
        super(detnet_bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=2, bias=False, dilation=2)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size
            =1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.downsample = nn.Sequential()
        if (stride != 1 or in_planes != self.expansion * planes or 
            block_type == 'B'):
            self.downsample = nn.Sequential(nn.Conv2d(in_planes, self.
                expansion * planes, kernel_size=1, stride=stride, bias=
                False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out


class DRNSegMT_A(nn.Module):

    def __init__(self, block, layers, n_classes=21, det_tensor_num=30):
        """
        :param block: resnet basicblock or bottleblock
        :param layers: [2, 2, 2, 2] resnet block format
        :param n_classes: segment classes
        :param det_tensor_num: object detection num
        """
        super(DRNSegMT_A, self).__init__()
        self.inplanes = 64
        self.det_tensor_num = det_tensor_num
        self.out_dim = 512 * block.expansion
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
            dilation=1)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
            dilation=1)
        self.out_conv = nn.Conv2d(self.out_dim, n_classes, kernel_size=1)
        self.layer5 = self._make_detnet_layer(in_channels=512 * block.expansion
            )
        self.conv_end = nn.Conv2d(256, self.det_tensor_num, kernel_size=3,
            stride=1, padding=1, bias=False)
        self.bn_end = nn.BatchNorm2d(self.det_tensor_num)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
            for i in downsample._modules['1'].parameters():
                i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=(dilation,
                dilation)))
        return nn.Sequential(*layers)

    def _make_detnet_layer(self, in_channels):
        layers = []
        layers.append(detnet_bottleneck(in_planes=in_channels, planes=256,
            block_type='B'))
        layers.append(detnet_bottleneck(in_planes=256, planes=256,
            block_type='A'))
        layers.append(detnet_bottleneck(in_planes=256, planes=256,
            block_type='A'))
        return nn.Sequential(*layers)

    def forward(self, x):
        x_size = x.size()
        x_conv1 = self.conv1(x)
        x = self.bn1(x_conv1)
        x = self.relu(x)
        x_pool = self.maxpool(x)
        x_layer1 = self.layer1(x_pool)
        x_layer2 = self.layer2(x_layer1)
        x = self.layer3(x_layer2)
        x = self.layer4(x)
        x_sem = self.out_conv(x)
        x_sem = F.upsample_bilinear(x_sem, x_size[2:])
        x_det = self.layer5(x)
        x_det = self.conv_end(x_det)
        x_det = self.bn_end(x_det)
        x_det = F.sigmoid(x_det)
        x_det = x_det.permute(0, 2, 3, 1)
        return x_sem, x_det


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
        dilation=(1, 1), residual=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, padding=dilation[0],
            dilation=dilation[0])
        self.bn1 = nn.BatchNorm2d(planes)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, padding=dilation[1], dilation=
            dilation[1])
        self.bn2 = nn.BatchNorm2d(planes)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.downsample = downsample
        self.stride = stride
        self.residual = residual

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        if self.residual:
            out += residual
        out = self.relu(out)
        return out


def conv3x3_bn_relu(in_planes, out_planes, stride=1, padding=1, dilation=1):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=3,
        stride=stride, padding=padding, bias=False, dilation=dilation), nn.
        BatchNorm2d(out_planes), nn.ReLU(inplace=True))


class RefineUnit(nn.Module):

    def __init__(self, f2_channel, n_classes=21):
        super(RefineUnit, self).__init__()
        self.f2_channel = f2_channel
        self.n_classes = n_classes
        self.up = nn.ConvTranspose2d(n_classes, n_classes, 4, stride=2,
            padding=1, output_padding=0, groups=n_classes, bias=False)
        self.out_conv_1 = conv3x3_bn_relu(in_planes=self.f2_channel,
            out_planes=self.n_classes)
        self.out_conv_2 = conv3x3_bn_relu(in_planes=self.n_classes,
            out_planes=self.n_classes)

    def forward(self, f3, f2):
        m1 = self.up(f3)
        f2_1 = self.out_conv_1(f2)
        m2 = m1 + f2_1
        o2 = self.out_conv_2(m2)
        return o2


class DRNSegRefine_A(nn.Module):

    def __init__(self, block, layers, n_classes=21):
        super(DRNSegRefine_A, self).__init__()
        self.inplanes = 64
        self.out_dim = 512 * block.expansion
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
            dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
            dilation=4)
        self.out_conv = nn.Conv2d(self.out_dim, n_classes, kernel_size=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=8)
        self.refineunit_1 = RefineUnit(64, n_classes)
        self.refineunit_2 = RefineUnit(64, n_classes)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
            for i in downsample._modules['1'].parameters():
                i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=(dilation,
                dilation)))
        return nn.Sequential(*layers)

    def forward(self, x):
        x_conv1 = self.conv1(x)
        x = self.bn1(x_conv1)
        x = self.relu(x)
        x_pool = self.maxpool(x)
        x_layer1 = self.layer1(x_pool)
        x_layer2 = self.layer2(x_layer1)
        x = self.layer3(x_layer2)
        x = self.layer4(x)
        x = self.out_conv(x)
        x_refine1 = self.refineunit_1(x, x_layer1)
        x_refine2 = self.refineunit_2(x_refine1, x_conv1)
        x = self.up(x_refine2)
        return x


class ConvLSTMCell(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """
        super(ConvLSTMCell, self).__init__()
        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim, kernel_size=self.kernel_size,
            padding=self.padding, bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim,
            dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size):
        if next(self.parameters()).is_cuda:
            return Variable(torch.zeros(batch_size, self.hidden_dim, self.
                height, self.width)), Variable(torch.zeros(batch_size, self
                .hidden_dim, self.height, self.width))
        else:
            return Variable(torch.zeros(batch_size, self.hidden_dim, self.
                height, self.width)), Variable(torch.zeros(batch_size, self
                .hidden_dim, self.height, self.width))


class ConvLSTM(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size,
        num_layers, batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()
        self._check_kernel_size_consistency(kernel_size)
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')
        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1
                ]
            cell_list.append(ConvLSTMCell(input_size=(self.height, self.
                width), input_dim=cur_input_dim, hidden_dim=self.hidden_dim
                [i], kernel_size=self.kernel_size[i], bias=self.bias))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0))
        layer_output_list = []
        last_state_list = []
        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=
                    cur_layer_input[:, (t), :, :, :], cur_state=[h, c])
                output_inner.append(h)
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output
            layer_output_list.append(layer_output)
            last_state_list.append([h, c])
        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1]
            last_state_list = last_state_list[-1]
        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or isinstance(kernel_size,
            list) and all([isinstance(elem, tuple) for elem in kernel_size])):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
        dilation=(1, 1), residual=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, padding=dilation[0],
            dilation=dilation[0])
        self.bn1 = nn.BatchNorm2d(planes)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, padding=dilation[1], dilation=
            dilation[1])
        self.bn2 = nn.BatchNorm2d(planes)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.downsample = downsample
        self.stride = stride
        self.residual = residual

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        if self.residual:
            out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
        dilation=(1, 1)):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=dilation[1], bias=False, dilation=dilation[1])
        self.bn2 = nn.BatchNorm2d(planes)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
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


class DRNPred_A(nn.Module):

    def __init__(self, block, layers, input_channel=3):
        self.inplanes = 64
        super(DRNPred_A, self).__init__()
        self.block = block
        self.layers = layers
        self.input_channel = input_channel
        self.out_dim = 512 * block.expansion
        self.conv1 = nn.Conv2d(self.input_channel, 64, kernel_size=7,
            stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
            dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
            dilation=4)
        self.layer5 = self._make_pred_layer(ASPP_Classifier_Module, [6, 12,
            18, 24], [6, 12, 18, 24], input_channel, in_channels=512 *
            block.expansion)
        if self.layer5 is not None:
            self.out_dim = input_channel
            pass
        else:
            self.out_dim = 512 * block.expansion
            pass
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
            for i in downsample._modules['1'].parameters():
                i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=(dilation,
                dilation)))
        return nn.Sequential(*layers)

    def _make_pred_layer(self, block, dilation_series, padding_series,
        n_classes, in_channels):
        return block(dilation_series, padding_series, n_classes, in_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x


class DRNSegPred(nn.Module):

    def __init__(self, model_name, pretrained=False, use_torch_up=True,
        input_channel=19, input_shape=(64, 64), n_classes=21):
        super(DRNSegPred, self).__init__()
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.input_channel = input_channel
        model = eval(model_name)(pretrained=pretrained, input_channel=
            input_channel * 4)
        self.base = model
        self.seg = nn.Conv2d(model.out_dim, input_channel * 4, kernel_size=1)
        m = self.seg
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2.0 / n))
        m.bias.data.zero_()
        self.lstm1 = ConvLSTM(input_size=(self.input_shape[0] // 8, self.
            input_shape[1] // 8), input_dim=self.input_channel, hidden_dim=
            [128, 128, self.input_channel], kernel_size=(3, 3), num_layers=
            3, batch_first=True, bias=True, return_all_layers=False)
        self.out_conv = nn.Conv2d(self.input_channel * 4, self.n_classes,
            kernel_size=1)
        if use_torch_up:
            self.up = nn.UpsamplingBilinear2d(scale_factor=8)
        else:
            up = nn.ConvTranspose2d(input_channel * 4, input_channel * 4, 
                16, stride=8, padding=4, output_padding=0, groups=
                input_channel * 4, bias=False)
            fill_up_weights(up)
            up.weight.requires_grad = False
            self.up = up

    def forward(self, x):
        x = self.base(x)
        x = self.seg(x)
        x = x.view(-1, 4, self.input_channel, self.input_shape[0] // 8, 
            self.input_shape[1] // 8)
        x, _ = self.lstm1(x)
        x = x.view(-1, 4 * self.input_channel, self.input_shape[0] // 8, 
            self.input_shape[1] // 8)
        x = self.out_conv(x)
        y = self.up(x)
        return y


class _DenseUpsamplingConvModule(nn.Module):

    def __init__(self, down_factor, in_dim, n_classes):
        super(_DenseUpsamplingConvModule, self).__init__()
        upsample_dim = down_factor ** 2 * n_classes
        self.conv = nn.Conv2d(in_dim, upsample_dim, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(upsample_dim)
        self.relu = nn.ReLU(inplace=True)
        self.pixel_shuffle = nn.PixelShuffle(down_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pixel_shuffle(x)
        return x


class ResNetDUC(nn.Module):

    def __init__(self, n_classes=21, pretrained=False):
        super(ResNetDUC, self).__init__()
        resnet = models.resnet152(pretrained=pretrained)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu,
            resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation = 2, 2
                m.padding = 2, 2
                m.stride = 1, 1
            elif 'downsample.0' in n:
                m.stride = 1, 1
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation = 4, 4
                m.padding = 4, 4
                m.stride = 1, 1
            elif 'downsample.0' in n:
                m.stride = 1, 1
        self.duc = _DenseUpsamplingConvModule(8, 2048, n_classes)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.duc(x)
        return x


class ResNetDUCHDC(nn.Module):

    def __init__(self, n_classes, pretrained=True):
        super(ResNetDUCHDC, self).__init__()
        resnet = models.resnet152(pretrained=pretrained)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu,
            resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        for n, m in self.layer3.named_modules():
            if 'conv2' in n or 'downsample.0' in n:
                m.stride = 1, 1
        for n, m in self.layer4.named_modules():
            if 'conv2' in n or 'downsample.0' in n:
                m.stride = 1, 1
        layer3_group_config = [1, 2, 5, 9]
        for idx in range(len(self.layer3)):
            self.layer3[idx].conv2.dilation = layer3_group_config[idx % 4
                ], layer3_group_config[idx % 4]
            self.layer3[idx].conv2.padding = layer3_group_config[idx % 4
                ], layer3_group_config[idx % 4]
        layer4_group_config = [5, 9, 17]
        for idx in range(len(self.layer4)):
            self.layer4[idx].conv2.dilation = layer4_group_config[idx
                ], layer4_group_config[idx]
            self.layer4[idx].conv2.padding = layer4_group_config[idx
                ], layer4_group_config[idx]
        self.duc = _DenseUpsamplingConvModule(8, 2048, n_classes)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.duc(x)
        return x


class InitialBlock(nn.Module):

    def __init__(self):
        super(InitialBlock, self).__init__()
        self.conv = nn.Conv2d(3, 13, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.batch_norm = nn.BatchNorm2d(16, eps=0.001)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.batch_norm(output)
        return F.relu(output)


class EncoderMainPath(nn.Module):

    def __init__(self, internal_scale=None, use_relu=None, asymmetric=None,
        dilated=None, input_channels=None, output_channels=None, downsample
        =None, dropout_prob=None):
        super(EncoderMainPath, self).__init__()
        internal_channels = output_channels // internal_scale
        input_stride = downsample and 2 or 1
        self.__dict__.update(locals())
        del self.self
        self.input_conv = nn.Conv2d(input_channels, internal_channels,
            input_stride, stride=input_stride, padding=0, bias=False)
        self.input_batch_norm = nn.BatchNorm2d(internal_channels, eps=0.001)
        self.middle_conv = nn.Conv2d(internal_channels, internal_channels, 
            3, stride=1, padding=1, bias=True)
        self.middle_batch_norm = nn.BatchNorm2d(internal_channels, eps=0.001)
        self.output_conv = nn.Conv2d(internal_channels, output_channels, 1,
            stride=1, padding=0, bias=False)
        self.output_batch_norm = nn.BatchNorm2d(output_channels, eps=0.001)
        self.dropout = nn.Dropout2d(dropout_prob)

    def forward(self, input):
        output = self.input_conv(input)
        output = self.input_batch_norm(output)
        output = F.relu(output)
        output = self.middle_conv(output)
        output = self.middle_batch_norm(output)
        output = F.relu(output)
        output = self.output_conv(output)
        output = self.output_batch_norm(output)
        output = self.dropout(output)
        return output


class EncoderOtherPath(nn.Module):

    def __init__(self, internal_scale=None, use_relu=None, asymmetric=None,
        dilated=None, input_channels=None, output_channels=None, downsample
        =None, **kwargs):
        super(EncoderOtherPath, self).__init__()
        self.__dict__.update(locals())
        del self.self
        if downsample:
            self.pool = nn.MaxPool2d(2, stride=2, return_indices=True)

    def forward(self, input):
        output = input
        if self.downsample:
            output, self.indices = self.pool(input)
        if self.output_channels != self.input_channels:
            new_size = [1, 1, 1, 1]
            new_size[1] = self.output_channels // self.input_channels
            output = output.repeat(*new_size)
        return output


class EncoderModule(nn.Module):

    def __init__(self, **kwargs):
        super(EncoderModule, self).__init__()
        self.main = EncoderMainPath(**kwargs)
        self.other = EncoderOtherPath(**kwargs)

    def forward(self, input):
        main = self.main(input)
        other = self.other(input)
        return F.relu(main + other)


class Encoder(nn.Module):

    def __init__(self, params, nclasses):
        super(Encoder, self).__init__()
        self.initial_block = InitialBlock()
        self.layers = []
        for i, params in enumerate(params):
            layer_name = 'encoder_{:02d}'.format(i)
            layer = EncoderModule(**params)
            super(Encoder, self).__setattr__(layer_name, layer)
            self.layers.append(layer)
        self.output_conv = nn.Conv2d(128, nclasses, 1, stride=1, padding=0,
            bias=True)

    def forward(self, input, predict=False):
        output = self.initial_block(input)
        for layer in self.layers:
            output = layer(output)
        if predict:
            output = self.output_conv(output)
        return output


class DecoderMainPath(nn.Module):

    def __init__(self, input_channels=None, output_channels=None, upsample=
        None, pooling_module=None):
        super(DecoderMainPath, self).__init__()
        internal_channels = output_channels // 4
        input_stride = 2 if upsample is True else 1
        self.__dict__.update(locals())
        del self.self
        self.input_conv = nn.Conv2d(input_channels, internal_channels, 1,
            stride=1, padding=0, bias=False)
        self.input_batch_norm = nn.BatchNorm2d(internal_channels, eps=0.001)
        if not upsample:
            self.middle_conv = nn.Conv2d(internal_channels,
                internal_channels, 3, stride=1, padding=1, bias=True)
        else:
            self.middle_conv = nn.ConvTranspose2d(internal_channels,
                internal_channels, 3, stride=2, padding=1, output_padding=1,
                bias=True)
        self.middle_batch_norm = nn.BatchNorm2d(internal_channels, eps=0.001)
        self.output_conv = nn.Conv2d(internal_channels, output_channels, 1,
            stride=1, padding=0, bias=False)
        self.output_batch_norm = nn.BatchNorm2d(output_channels, eps=0.001)

    def forward(self, input):
        output = self.input_conv(input)
        output = self.input_batch_norm(output)
        output = F.relu(output)
        output = self.middle_conv(output)
        output = self.middle_batch_norm(output)
        output = F.relu(output)
        output = self.output_conv(output)
        output = self.output_batch_norm(output)
        return output


class DecoderOtherPath(nn.Module):

    def __init__(self, input_channels=None, output_channels=None, upsample=
        None, pooling_module=None):
        super(DecoderOtherPath, self).__init__()
        self.__dict__.update(locals())
        del self.self
        if output_channels != input_channels or upsample:
            self.conv = nn.Conv2d(input_channels, output_channels, 1,
                stride=1, padding=0, bias=False)
            self.batch_norm = nn.BatchNorm2d(output_channels, eps=0.001)
            if upsample and pooling_module:
                self.unpool = nn.MaxUnpool2d(2, stride=2, padding=0)

    def forward(self, input):
        output = input
        if self.output_channels != self.input_channels or self.upsample:
            output = self.conv(output)
            output = self.batch_norm(output)
            if self.upsample and self.pooling_module:
                output_size = list(output.size())
                output_size[2] *= 2
                output_size[3] *= 2
                output = self.unpool(output, self.pooling_module.indices,
                    output_size=output_size)
        return output


class DecoderModule(nn.Module):

    def __init__(self, **kwargs):
        super(DecoderModule, self).__init__()
        self.main = DecoderMainPath(**kwargs)
        self.other = DecoderOtherPath(**kwargs)

    def forward(self, input):
        main = self.main(input)
        other = self.other(input)
        return F.relu(main + other)


class Decoder(nn.Module):

    def __init__(self, params, nclasses, encoder):
        super(Decoder, self).__init__()
        self.encoder = encoder
        self.pooling_modules = []
        for mod in self.encoder.modules():
            try:
                if mod.other.downsample:
                    self.pooling_modules.append(mod.other)
            except AttributeError:
                pass
        self.layers = []
        for i, params in enumerate(params):
            if params['upsample']:
                params['pooling_module'] = self.pooling_modules.pop(-1)
            layer = DecoderModule(**params)
            self.layers.append(layer)
            layer_name = 'decoder{:02d}'.format(i)
            super(Decoder, self).__setattr__(layer_name, layer)
        self.output_conv = nn.ConvTranspose2d(16, nclasses, 2, stride=2,
            padding=0, output_padding=0, bias=True)

    def forward(self, input):
        output = self.encoder(input, predict=False)
        for layer in self.layers:
            output = layer(output)
        output = self.output_conv(output)
        return output


DECODER_PARAMS = [{'input_channels': 128, 'output_channels': 128,
    'upsample': False, 'pooling_module': None}, {'input_channels': 128,
    'output_channels': 64, 'upsample': True, 'pooling_module': None}, {
    'input_channels': 64, 'output_channels': 64, 'upsample': False,
    'pooling_module': None}, {'input_channels': 64, 'output_channels': 64,
    'upsample': False, 'pooling_module': None}, {'input_channels': 64,
    'output_channels': 16, 'upsample': True, 'pooling_module': None}, {
    'input_channels': 16, 'output_channels': 16, 'upsample': False,
    'pooling_module': None}]


ENCODER_PARAMS = [{'internal_scale': 4, 'use_relu': True, 'asymmetric': 
    False, 'dilated': False, 'input_channels': 16, 'output_channels': 64,
    'downsample': True, 'dropout_prob': 0.01}, {'internal_scale': 4,
    'use_relu': True, 'asymmetric': False, 'dilated': False,
    'input_channels': 64, 'output_channels': 64, 'downsample': False,
    'dropout_prob': 0.01}, {'internal_scale': 4, 'use_relu': True,
    'asymmetric': False, 'dilated': False, 'input_channels': 64,
    'output_channels': 64, 'downsample': False, 'dropout_prob': 0.01}, {
    'internal_scale': 4, 'use_relu': True, 'asymmetric': False, 'dilated': 
    False, 'input_channels': 64, 'output_channels': 64, 'downsample': False,
    'dropout_prob': 0.01}, {'internal_scale': 4, 'use_relu': True,
    'asymmetric': False, 'dilated': False, 'input_channels': 64,
    'output_channels': 64, 'downsample': False, 'dropout_prob': 0.01}, {
    'internal_scale': 4, 'use_relu': True, 'asymmetric': False, 'dilated': 
    False, 'input_channels': 64, 'output_channels': 128, 'downsample': True,
    'dropout_prob': 0.1}, {'internal_scale': 4, 'use_relu': True,
    'asymmetric': False, 'dilated': False, 'input_channels': 128,
    'output_channels': 128, 'downsample': False, 'dropout_prob': 0.1}, {
    'internal_scale': 4, 'use_relu': True, 'asymmetric': False, 'dilated': 
    False, 'input_channels': 128, 'output_channels': 128, 'downsample': 
    False, 'dropout_prob': 0.1}, {'internal_scale': 4, 'use_relu': True,
    'asymmetric': False, 'dilated': False, 'input_channels': 128,
    'output_channels': 128, 'downsample': False, 'dropout_prob': 0.1}, {
    'internal_scale': 4, 'use_relu': True, 'asymmetric': False, 'dilated': 
    False, 'input_channels': 128, 'output_channels': 128, 'downsample': 
    False, 'dropout_prob': 0.1}, {'internal_scale': 4, 'use_relu': True,
    'asymmetric': False, 'dilated': False, 'input_channels': 128,
    'output_channels': 128, 'downsample': False, 'dropout_prob': 0.1}, {
    'internal_scale': 4, 'use_relu': True, 'asymmetric': False, 'dilated': 
    False, 'input_channels': 128, 'output_channels': 128, 'downsample': 
    False, 'dropout_prob': 0.1}, {'internal_scale': 4, 'use_relu': True,
    'asymmetric': False, 'dilated': False, 'input_channels': 128,
    'output_channels': 128, 'downsample': False, 'dropout_prob': 0.1}, {
    'internal_scale': 4, 'use_relu': True, 'asymmetric': False, 'dilated': 
    False, 'input_channels': 128, 'output_channels': 128, 'downsample': 
    False, 'dropout_prob': 0.1}, {'internal_scale': 4, 'use_relu': True,
    'asymmetric': False, 'dilated': False, 'input_channels': 128,
    'output_channels': 128, 'downsample': False, 'dropout_prob': 0.1}, {
    'internal_scale': 4, 'use_relu': True, 'asymmetric': False, 'dilated': 
    False, 'input_channels': 128, 'output_channels': 128, 'downsample': 
    False, 'dropout_prob': 0.1}, {'internal_scale': 4, 'use_relu': True,
    'asymmetric': False, 'dilated': False, 'input_channels': 128,
    'output_channels': 128, 'downsample': False, 'dropout_prob': 0.1}, {
    'internal_scale': 4, 'use_relu': True, 'asymmetric': False, 'dilated': 
    False, 'input_channels': 128, 'output_channels': 128, 'downsample': 
    False, 'dropout_prob': 0.1}, {'internal_scale': 4, 'use_relu': True,
    'asymmetric': False, 'dilated': False, 'input_channels': 128,
    'output_channels': 128, 'downsample': False, 'dropout_prob': 0.1}, {
    'internal_scale': 4, 'use_relu': True, 'asymmetric': False, 'dilated': 
    False, 'input_channels': 128, 'output_channels': 128, 'downsample': 
    False, 'dropout_prob': 0.1}, {'internal_scale': 4, 'use_relu': True,
    'asymmetric': False, 'dilated': False, 'input_channels': 128,
    'output_channels': 128, 'downsample': False, 'dropout_prob': 0.1}, {
    'internal_scale': 4, 'use_relu': True, 'asymmetric': False, 'dilated': 
    False, 'input_channels': 128, 'output_channels': 128, 'downsample': 
    False, 'dropout_prob': 0.1}, {'internal_scale': 4, 'use_relu': True,
    'asymmetric': False, 'dilated': False, 'input_channels': 128,
    'output_channels': 128, 'downsample': False, 'dropout_prob': 0.1}]


class ENet(nn.Module):

    def __init__(self, n_classes, pretrained=False):
        super(ENet, self).__init__()
        self.encoder = Encoder(ENCODER_PARAMS, n_classes)
        self.decoder = Decoder(DECODER_PARAMS, n_classes, self.encoder)

    def forward(self, input, only_encode=False, predict=True):
        if only_encode:
            return self.encoder.forward(input, predict=predict)
        else:
            return self.decoder.forward(input)


class InitialBlock(nn.Module):
    """The initial block is composed of two branches:
    1. a main branch which performs a regular convolution with stride 2;
    2. an extension branch which performs max-pooling.
    Doing both operations in parallel and concatenating their results
    allows for efficient downsampling and expansion. The main branch
    outputs 13 feature maps while the extension branch outputs 3, for a
    total of 16 feature maps after concatenation.
    Keyword arguments:
    - in_channels (int): the number of input channels.
    - out_channels (int): the number output channels.
    - kernel_size (int, optional): the kernel size of the filters used in
    the convolution layer. Default: 3.
    - padding (int, optional): zero-padding added to both sides of the
    input. Default: 0.
    - bias (bool, optional): Adds a learnable bias to the output if
    ``True``. Default: False.
    - relu (bool, optional): When ``True`` ReLU is used as the activation
    function; otherwise, PReLU is used. Default: True.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=0,
        bias=False, relu=True):
        super(InitialBlock, self).__init__()
        if relu:
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()
        self.main_branch = nn.Conv2d(in_channels, out_channels - 3,
            kernel_size=kernel_size, stride=2, padding=padding, bias=bias)
        self.ext_branch = nn.MaxPool2d(kernel_size, stride=2, padding=padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.out_prelu = activation

    def forward(self, x):
        main = self.main_branch(x)
        ext = self.ext_branch(x)
        out = torch.cat((main, ext), 1)
        out = self.batch_norm(out)
        return self.out_prelu(out)


class RegularBottleneck(nn.Module):
    """Regular bottlenecks are the main building block of ENet.
    Main branch:
    1. Shortcut connection.
    Extension branch:
    1. 1x1 convolution which decreases the number of channels by
    ``internal_ratio``, also called a projection;
    2. regular, dilated or asymmetric convolution;
    3. 1x1 convolution which increases the number of channels back to
    ``channels``, also called an expansion;
    4. dropout as a regularizer.
    Keyword arguments:
    - channels (int): the number of input and output channels.
    - internal_ratio (int, optional): a scale factor applied to
    ``channels`` used to compute the number of
    channels after the projection. eg. given ``channels`` equal to 128 and
    internal_ratio equal to 2 the number of channels after the projection
    is 64. Default: 4.
    - kernel_size (int, optional): the kernel size of the filters used in
    the convolution layer described above in item 2 of the extension
    branch. Default: 3.
    - padding (int, optional): zero-padding added to both sides of the
    input. Default: 0.
    - dilation (int, optional): spacing between kernel elements for the
    convolution described in item 2 of the extension branch. Default: 1.
    asymmetric (bool, optional): flags if the convolution described in
    item 2 of the extension branch is asymmetric or not. Default: False.
    - dropout_prob (float, optional): probability of an element to be
    zeroed. Default: 0 (no dropout).
    - bias (bool, optional): Adds a learnable bias to the output if
    ``True``. Default: False.
    - relu (bool, optional): When ``True`` ReLU is used as the activation
    function; otherwise, PReLU is used. Default: True.
    """

    def __init__(self, channels, internal_ratio=4, kernel_size=3, padding=0,
        dilation=1, asymmetric=False, dropout_prob=0, bias=False, relu=True):
        super(RegularBottleneck, self).__init__()
        if internal_ratio <= 1 or internal_ratio > channels:
            raise RuntimeError(
                'Value out of range. Expected value in the interval [1, {0}], got internal_scale={1}.'
                .format(channels, internal_ratio))
        internal_channels = channels // internal_ratio
        if relu:
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()
        self.ext_conv1 = nn.Sequential(nn.Conv2d(channels,
            internal_channels, kernel_size=1, stride=1, bias=bias), nn.
            BatchNorm2d(internal_channels), activation)
        if asymmetric:
            self.ext_conv2 = nn.Sequential(nn.Conv2d(internal_channels,
                internal_channels, kernel_size=(kernel_size, 1), stride=1,
                padding=(padding, 0), dilation=dilation, bias=bias), nn.
                BatchNorm2d(internal_channels), activation, nn.Conv2d(
                internal_channels, internal_channels, kernel_size=(1,
                kernel_size), stride=1, padding=(0, padding), dilation=
                dilation, bias=bias), nn.BatchNorm2d(internal_channels),
                activation)
        else:
            self.ext_conv2 = nn.Sequential(nn.Conv2d(internal_channels,
                internal_channels, kernel_size=kernel_size, stride=1,
                padding=padding, dilation=dilation, bias=bias), nn.
                BatchNorm2d(internal_channels), activation)
        self.ext_conv3 = nn.Sequential(nn.Conv2d(internal_channels,
            channels, kernel_size=1, stride=1, bias=bias), nn.BatchNorm2d(
            channels), activation)
        self.ext_regul = nn.Dropout2d(p=dropout_prob)
        self.out_prelu = activation

    def forward(self, x):
        main = x
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)
        out = main + ext
        return self.out_prelu(out)


class DownsamplingBottleneck(nn.Module):
    """Downsampling bottlenecks further downsample the feature map size.
    Main branch:
    1. max pooling with stride 2; indices are saved to be used for
    unpooling later.
    Extension branch:
    1. 2x2 convolution with stride 2 that decreases the number of channels
    by ``internal_ratio``, also called a projection;
    2. regular convolution (by default, 3x3);
    3. 1x1 convolution which increases the number of channels to
    ``out_channels``, also called an expansion;
    4. dropout as a regularizer.
    Keyword arguments:
    - in_channels (int): the number of input channels.
    - out_channels (int): the number of output channels.
    - internal_ratio (int, optional): a scale factor applied to ``channels``
    used to compute the number of channels after the projection. eg. given
    ``channels`` equal to 128 and internal_ratio equal to 2 the number of
    channels after the projection is 64. Default: 4.
    - kernel_size (int, optional): the kernel size of the filters used in
    the convolution layer described above in item 2 of the extension branch.
    Default: 3.
    - padding (int, optional): zero-padding added to both sides of the
    input. Default: 0.
    - dilation (int, optional): spacing between kernel elements for the
    convolution described in item 2 of the extension branch. Default: 1.
    - asymmetric (bool, optional): flags if the convolution described in
    item 2 of the extension branch is asymmetric or not. Default: False.
    - return_indices (bool, optional):  if ``True``, will return the max
    indices along with the outputs. Useful when unpooling later.
    - dropout_prob (float, optional): probability of an element to be
    zeroed. Default: 0 (no dropout).
    - bias (bool, optional): Adds a learnable bias to the output if
    ``True``. Default: False.
    - relu (bool, optional): When ``True`` ReLU is used as the activation
    function; otherwise, PReLU is used. Default: True.
    """

    def __init__(self, in_channels, out_channels, internal_ratio=4,
        kernel_size=3, padding=0, return_indices=False, dropout_prob=0,
        bias=False, relu=True):
        super(DownsamplingBottleneck, self).__init__()
        self.return_indices = return_indices
        if internal_ratio <= 1 or internal_ratio > in_channels:
            raise RuntimeError(
                'Value out of range. Expected value in the interval [1, {0}], got internal_scale={1}. '
                .format(in_channels, internal_ratio))
        internal_channels = in_channels // internal_ratio
        if relu:
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()
        self.main_max1 = nn.MaxPool2d(kernel_size, stride=2, padding=
            padding, return_indices=return_indices)
        self.ext_conv1 = nn.Sequential(nn.Conv2d(in_channels,
            internal_channels, kernel_size=2, stride=2, bias=bias), nn.
            BatchNorm2d(internal_channels), activation)
        self.ext_conv2 = nn.Sequential(nn.Conv2d(internal_channels,
            internal_channels, kernel_size=kernel_size, stride=1, padding=
            padding, bias=bias), nn.BatchNorm2d(internal_channels), activation)
        self.ext_conv3 = nn.Sequential(nn.Conv2d(internal_channels,
            out_channels, kernel_size=1, stride=1, bias=bias), nn.
            BatchNorm2d(out_channels), activation)
        self.ext_regul = nn.Dropout2d(p=dropout_prob)
        self.out_prelu = activation

    def forward(self, x):
        if self.return_indices:
            main, max_indices = self.main_max1(x)
        else:
            main = self.main_max1(x)
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)
        n, ch_ext, h, w = ext.size()
        ch_main = main.size()[1]
        padding = Variable(torch.zeros(n, ch_ext - ch_main, h, w))
        if main.is_cuda:
            padding = padding
        main = torch.cat((main, padding), 1)
        out = main + ext
        return self.out_prelu(out), max_indices


class UpsamplingBottleneck(nn.Module):
    """The upsampling bottlenecks upsample the feature map resolution using max
    pooling indices stored from the corresponding downsampling bottleneck.
    Main branch:
    1. 1x1 convolution with stride 1 that decreases the number of channels by
    ``internal_ratio``, also called a projection;
    2. max unpool layer using the max pool indices from the corresponding
    downsampling max pool layer.
    Extension branch:
    1. 1x1 convolution with stride 1 that decreases the number of channels by
    ``internal_ratio``, also called a projection;
    2. transposed convolution (by default, 3x3);
    3. 1x1 convolution which increases the number of channels to
    ``out_channels``, also called an expansion;
    4. dropout as a regularizer.
    Keyword arguments:
    - in_channels (int): the number of input channels.
    - out_channels (int): the number of output channels.
    - internal_ratio (int, optional): a scale factor applied to ``in_channels``
     used to compute the number of channels after the projection. eg. given
     ``in_channels`` equal to 128 and ``internal_ratio`` equal to 2 the number
     of channels after the projection is 64. Default: 4.
    - kernel_size (int, optional): the kernel size of the filters used in the
    convolution layer described above in item 2 of the extension branch.
    Default: 3.
    - padding (int, optional): zero-padding added to both sides of the input.
    Default: 0.
    - dropout_prob (float, optional): probability of an element to be zeroed.
    Default: 0 (no dropout).
    - bias (bool, optional): Adds a learnable bias to the output if ``True``.
    Default: False.
    - relu (bool, optional): When ``True`` ReLU is used as the activation
    function; otherwise, PReLU is used. Default: True.
    """

    def __init__(self, in_channels, out_channels, internal_ratio=4,
        kernel_size=3, padding=0, dropout_prob=0, bias=False, relu=True):
        super(UpsamplingBottleneck, self).__init__()
        if internal_ratio <= 1 or internal_ratio > in_channels:
            raise RuntimeError(
                'Value out of range. Expected value in the interval [1, {0}], got internal_scale={1}. '
                .format(in_channels, internal_ratio))
        internal_channels = in_channels // internal_ratio
        if relu:
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()
        self.main_conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels,
            kernel_size=1, bias=bias), nn.BatchNorm2d(out_channels))
        self.main_unpool1 = nn.MaxUnpool2d(kernel_size=2)
        self.ext_conv1 = nn.Sequential(nn.Conv2d(in_channels,
            internal_channels, kernel_size=1, bias=bias), nn.BatchNorm2d(
            internal_channels), activation)
        self.ext_conv2 = nn.Sequential(nn.ConvTranspose2d(internal_channels,
            internal_channels, kernel_size=kernel_size, stride=2, padding=
            padding, output_padding=1, bias=bias), nn.BatchNorm2d(
            internal_channels), activation)
        self.ext_conv3 = nn.Sequential(nn.Conv2d(internal_channels,
            out_channels, kernel_size=1, bias=bias), nn.BatchNorm2d(
            out_channels), activation)
        self.ext_regul = nn.Dropout2d(p=dropout_prob)
        self.out_prelu = activation

    def forward(self, x, max_indices):
        main = self.main_conv1(x)
        main = self.main_unpool1(main, max_indices)
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)
        out = main + ext
        return self.out_prelu(out)


class ENetV2(nn.Module):
    """Generate the ENet model.
    Keyword arguments:
    - n_classes (int): the number of classes to segment.
    - encoder_relu (bool, optional): When ``True`` ReLU is used as the
    activation function in the encoder blocks/layers; otherwise, PReLU
    is used. Default: False.
    - decoder_relu (bool, optional): When ``True`` ReLU is used as the
    activation function in the decoder blocks/layers; otherwise, PReLU
    is used. Default: True.
    """

    def __init__(self, n_classes, encoder_relu=False, decoder_relu=True,
        pretrained=False):
        super(ENetV2, self).__init__()
        self.initial_block = InitialBlock(3, 16, padding=1, relu=encoder_relu)
        self.downsample1_0 = DownsamplingBottleneck(16, 64, padding=1,
            return_indices=True, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_1 = RegularBottleneck(64, padding=1, dropout_prob=
            0.01, relu=encoder_relu)
        self.regular1_2 = RegularBottleneck(64, padding=1, dropout_prob=
            0.01, relu=encoder_relu)
        self.regular1_3 = RegularBottleneck(64, padding=1, dropout_prob=
            0.01, relu=encoder_relu)
        self.regular1_4 = RegularBottleneck(64, padding=1, dropout_prob=
            0.01, relu=encoder_relu)
        self.downsample2_0 = DownsamplingBottleneck(64, 128, padding=1,
            return_indices=True, dropout_prob=0.1, relu=encoder_relu)
        self.regular2_1 = RegularBottleneck(128, padding=1, dropout_prob=
            0.1, relu=encoder_relu)
        self.dilated2_2 = RegularBottleneck(128, dilation=2, padding=2,
            dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric2_3 = RegularBottleneck(128, kernel_size=5, padding=
            2, asymmetric=True, dropout_prob=0.1, relu=encoder_relu)
        self.dilated2_4 = RegularBottleneck(128, dilation=4, padding=4,
            dropout_prob=0.1, relu=encoder_relu)
        self.regular2_5 = RegularBottleneck(128, padding=1, dropout_prob=
            0.1, relu=encoder_relu)
        self.dilated2_6 = RegularBottleneck(128, dilation=8, padding=8,
            dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric2_7 = RegularBottleneck(128, kernel_size=5,
            asymmetric=True, padding=2, dropout_prob=0.1, relu=encoder_relu)
        self.dilated2_8 = RegularBottleneck(128, dilation=16, padding=16,
            dropout_prob=0.1, relu=encoder_relu)
        self.regular3_0 = RegularBottleneck(128, padding=1, dropout_prob=
            0.1, relu=encoder_relu)
        self.dilated3_1 = RegularBottleneck(128, dilation=2, padding=2,
            dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric3_2 = RegularBottleneck(128, kernel_size=5, padding=
            2, asymmetric=True, dropout_prob=0.1, relu=encoder_relu)
        self.dilated3_3 = RegularBottleneck(128, dilation=4, padding=4,
            dropout_prob=0.1, relu=encoder_relu)
        self.regular3_4 = RegularBottleneck(128, padding=1, dropout_prob=
            0.1, relu=encoder_relu)
        self.dilated3_5 = RegularBottleneck(128, dilation=8, padding=8,
            dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric3_6 = RegularBottleneck(128, kernel_size=5,
            asymmetric=True, padding=2, dropout_prob=0.1, relu=encoder_relu)
        self.dilated3_7 = RegularBottleneck(128, dilation=16, padding=16,
            dropout_prob=0.1, relu=encoder_relu)
        self.upsample4_0 = UpsamplingBottleneck(128, 64, padding=1,
            dropout_prob=0.1, relu=decoder_relu)
        self.regular4_1 = RegularBottleneck(64, padding=1, dropout_prob=0.1,
            relu=decoder_relu)
        self.regular4_2 = RegularBottleneck(64, padding=1, dropout_prob=0.1,
            relu=decoder_relu)
        self.upsample5_0 = UpsamplingBottleneck(64, 16, padding=1,
            dropout_prob=0.1, relu=decoder_relu)
        self.regular5_1 = RegularBottleneck(16, padding=1, dropout_prob=0.1,
            relu=decoder_relu)
        self.transposed_conv = nn.ConvTranspose2d(16, n_classes,
            kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)

    def forward(self, x):
        x = self.initial_block(x)
        x, max_indices1_0 = self.downsample1_0(x)
        x = self.regular1_1(x)
        x = self.regular1_2(x)
        x = self.regular1_3(x)
        x = self.regular1_4(x)
        x, max_indices2_0 = self.downsample2_0(x)
        x = self.regular2_1(x)
        x = self.dilated2_2(x)
        x = self.asymmetric2_3(x)
        x = self.dilated2_4(x)
        x = self.regular2_5(x)
        x = self.dilated2_6(x)
        x = self.asymmetric2_7(x)
        x = self.dilated2_8(x)
        x = self.regular3_0(x)
        x = self.dilated3_1(x)
        x = self.asymmetric3_2(x)
        x = self.dilated3_3(x)
        x = self.regular3_4(x)
        x = self.dilated3_5(x)
        x = self.asymmetric3_6(x)
        x = self.dilated3_7(x)
        x = self.upsample4_0(x, max_indices2_0)
        x = self.regular4_1(x)
        x = self.regular4_2(x)
        x = self.upsample5_0(x, max_indices1_0)
        x = self.regular5_1(x)
        x = self.transposed_conv(x)
        return x


class DownsamplerBlock(nn.Module):

    def __init__(self, ninput, noutput):
        super(DownsamplerBlock, self).__init__()
        self.conv = nn.Conv2d(ninput, noutput - ninput, (3, 3), stride=2,
            padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=0.001)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        return F.relu(output)


class non_bottleneck_1d(nn.Module):

    def __init__(self, chann, dropprob, dilated):
        super(non_bottleneck_1d, self).__init__()
        self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=
            (1, 0), bias=True)
        self.conv1x3_1 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=
            (0, 1), bias=True)
        self.bn1 = nn.BatchNorm2d(chann, eps=0.001)
        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=
            (1 * dilated, 0), bias=True, dilation=(dilated, 1))
        self.conv1x3_2 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=
            (0, 1 * dilated), bias=True, dilation=(1, dilated))
        self.bn2 = nn.BatchNorm2d(chann, eps=0.001)
        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, input):
        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)
        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)
        if self.dropout.p != 0:
            output = self.dropout(output)
        return F.relu(output + input)


class Encoder(nn.Module):

    def __init__(self, num_classes):
        super(Encoder, self).__init__()
        self.initial_block = DownsamplerBlock(3, 16)
        self.layers = nn.ModuleList()
        self.layers.append(DownsamplerBlock(16, 64))
        for x in range(0, 5):
            self.layers.append(non_bottleneck_1d(64, 0.03, 1))
        self.layers.append(DownsamplerBlock(64, 128))
        for x in range(0, 2):
            self.layers.append(non_bottleneck_1d(128, 0.3, 2))
            self.layers.append(non_bottleneck_1d(128, 0.3, 4))
            self.layers.append(non_bottleneck_1d(128, 0.3, 8))
            self.layers.append(non_bottleneck_1d(128, 0.3, 16))
        self.output_conv = nn.Conv2d(128, num_classes, 1, stride=1, padding
            =0, bias=True)

    def forward(self, input, predict=False):
        output = self.initial_block(input)
        for layer in self.layers:
            output = layer(output)
        if predict:
            output = self.output_conv(output)
        return output


class UpsamplerBlock(nn.Module):

    def __init__(self, ninput, noutput):
        super(UpsamplerBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2,
            padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=0.001)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)


class Decoder(nn.Module):

    def __init__(self, num_classes):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(UpsamplerBlock(128, 64))
        self.layers.append(non_bottleneck_1d(64, 0, 1))
        self.layers.append(non_bottleneck_1d(64, 0, 1))
        self.layers.append(UpsamplerBlock(64, 16))
        self.layers.append(non_bottleneck_1d(16, 0, 1))
        self.layers.append(non_bottleneck_1d(16, 0, 1))
        self.output_conv = nn.ConvTranspose2d(16, num_classes, 2, stride=2,
            padding=0, output_padding=0, bias=True)

    def forward(self, input):
        output = input
        for layer in self.layers:
            output = layer(output)
        output = self.output_conv(output)
        return output


class erfnet(nn.Module):

    def __init__(self, n_classes, encoder=None, pretrained=False):
        super(erfnet, self).__init__()
        if encoder == None:
            self.encoder = Encoder(n_classes)
        else:
            self.encoder = encoder
        self.decoder = Decoder(n_classes)

    def forward(self, input, only_encode=False):
        if only_encode:
            return self.encoder.forward(input, predict=True)
        else:
            output = self.encoder(input)
            return self.decoder.forward(output)


class DenseBlock(nn.Module):

    def __init__(self, nIn, growth_rate, depth, drop_rate=0, only_new=False,
        bottle_neck=False):
        super(DenseBlock, self).__init__()
        self.only_new = only_new
        self.depth = depth
        self.growth_rate = growth_rate
        self.layers = nn.ModuleList([self.get_transform(nIn + i *
            growth_rate, growth_rate, bottle_neck, drop_rate) for i in
            range(depth)])

    def forward(self, x):
        if self.only_new:
            outputs = []
            for i in range(self.depth):
                tx = self.layers[i](x)
                x = torch.cat((x, tx), 1)
                outputs.append(tx)
            return torch.cat(outputs, 1)
        else:
            for i in range(self.depth):
                x = torch.cat((x, self.layers[i](x)), 1)
            return x

    def get_transform(self, nIn, nOut, bottle_neck=None, drop_rate=0):
        if not bottle_neck or nIn <= nOut * bottle_neck:
            return nn.Sequential(nn.BatchNorm2d(nIn), nn.ReLU(True), nn.
                Conv2d(nIn, nOut, 3, stride=1, padding=1, bias=True), nn.
                Dropout(drop_rate))
        else:
            nBottle = nOut * bottle_neck
            return nn.Sequential(nn.BatchNorm2d(nIn), nn.ReLU(True), nn.
                Conv2d(nIn, nBottle, 1, stride=1, padding=0, bias=True), nn
                .BatchNorm2d(nBottle), nn.ReLU(True), nn.Conv2d(nBottle,
                nOut, 3, stride=1, padding=1, bias=True), nn.Dropout(drop_rate)
                )


class FCDenseNet(nn.Module):

    def __init__(self, depths, growth_rates, n_scales=5, n_channel_start=48,
        n_classes=12, drop_rate=0, bottle_neck=False):
        super(FCDenseNet, self).__init__()
        self.n_scales = n_scales
        self.n_classes = n_classes
        self.n_channel_start = n_channel_start
        self.depths = [depths] * (2 * n_scales + 1) if type(depths
            ) == int else depths
        self.growth_rates = [growth_rates] * (2 * n_scales + 1) if type(
            growth_rates) == int else growth_rates
        self.drop_rate = drop_rate
        assert len(self.depths) == len(self.growth_rates) == 2 * n_scales + 1
        self.conv_first = nn.Conv2d(3, n_channel_start, 3, stride=1,
            padding=1, bias=True)
        self.dense_blocks = nn.ModuleList([])
        self.transition_downs = nn.ModuleList([])
        self.transition_ups = nn.ModuleList([])
        nskip = []
        nIn = self.n_channel_start
        for i in range(n_scales):
            self.dense_blocks.append(DenseBlock(nIn, self.growth_rates[i],
                self.depths[i], drop_rate=drop_rate, bottle_neck=bottle_neck))
            nIn += self.growth_rates[i] * self.depths[i]
            nskip.append(nIn)
            self.transition_downs.append(self.get_TD(nIn, drop_rate))
        self.dense_blocks.append(DenseBlock(nIn, self.growth_rates[n_scales
            ], self.depths[n_scales], only_new=True, drop_rate=drop_rate,
            bottle_neck=bottle_neck))
        nIn = self.growth_rates[n_scales] * self.depths[n_scales]
        for i in range(n_scales - 1):
            self.transition_ups.append(nn.ConvTranspose2d(nIn, nIn, 3,
                stride=2, padding=1, bias=True))
            nIn += nskip.pop()
            self.dense_blocks.append(DenseBlock(nIn, self.growth_rates[
                n_scales + 1 + i], self.depths[n_scales + 1 + i], only_new=
                True, drop_rate=drop_rate, bottle_neck=bottle_neck))
            nIn = self.growth_rates[n_scales + 1 + i] * self.depths[
                n_scales + 1 + i]
        self.transition_ups.append(nn.ConvTranspose2d(nIn, nIn, 3, stride=2,
            padding=1, bias=True))
        nIn += nskip.pop()
        self.dense_blocks.append(DenseBlock(nIn, self.growth_rates[2 *
            n_scales], self.depths[2 * n_scales], drop_rate=drop_rate,
            bottle_neck=bottle_neck))
        nIn += self.growth_rates[2 * n_scales] * self.depths[2 * n_scales]
        self.conv_last = nn.Conv2d(nIn, n_classes, 1, bias=True)
        self.logsoftmax = nn.LogSoftmax()

    def forward(self, x):
        x = self.conv_first(x)
        skip_connects = []
        for i in range(self.n_scales):
            x = self.dense_blocks[i](x)
            skip_connects.append(x)
            x = self.transition_downs[i](x)
        x = self.dense_blocks[self.n_scales](x)
        for i in range(self.n_scales):
            skip = skip_connects.pop()
            TU = self.transition_ups[i]
            TU.padding = ((x.size(2) - 1) * TU.stride[0] - skip.size(2) +
                TU.kernel_size[0] + 1) // 2, ((x.size(3) - 1) * TU.stride[1
                ] - skip.size(3) + TU.kernel_size[1] + 1) // 2
            x = TU(x, output_size=skip.size())
            x = torch.cat((skip, x), 1)
            x = self.dense_blocks[self.n_scales + 1 + i](x)
        x = self.conv_last(x)
        return self.logsoftmax(x)

    def get_TD(self, nIn, drop_rate):
        layers = [nn.BatchNorm2d(nIn), nn.ReLU(True), nn.Conv2d(nIn, nIn, 1,
            bias=True)]
        if drop_rate > 0:
            layers.append(nn.Dropout(drop_rate))
        layers.append(nn.MaxPool2d(2))
        return nn.Sequential(*layers)


class fcn(nn.Module):

    def forward(self, x):
        conv1 = self.conv1_block(x)
        conv2 = self.conv2_block(conv1)
        conv3 = self.conv3_block(conv2)
        conv4 = self.conv4_block(conv3)
        conv5 = self.conv5_block(conv4)
        score = self.classifier(conv5)
        if self.module_type == '16s' or self.module_type == '8s':
            score_pool4 = self.score_pool4(conv4)
        if self.module_type == '8s':
            score_pool3 = self.score_pool3(conv3)
        if self.module_type == '16s' or self.module_type == '8s':
            score = F.upsample_bilinear(score, score_pool4.size()[2:])
            score += score_pool4
        if self.module_type == '8s':
            score = F.upsample_bilinear(score, score_pool3.size()[2:])
            score += score_pool3
        out = F.upsample_bilinear(score, x.size()[2:])
        return out

    def __init__(self, module_type='32s', n_classes=21, pretrained=False):
        super(fcn, self).__init__()
        self.n_classes = n_classes
        self.module_type = module_type
        self.conv1_block = nn.Sequential(nn.Conv2d(3, 64, 3, padding=100),
            nn.ReLU(inplace=True), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU
            (inplace=True), nn.MaxPool2d(2, stride=2, ceil_mode=True))
        self.conv2_block = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True), nn.Conv2d(128, 128, 3, padding=1), nn.
            ReLU(inplace=True), nn.MaxPool2d(2, stride=2, ceil_mode=True))
        self.conv3_block = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True), nn.Conv2d(256, 256, 3, padding=1), nn.
            ReLU(inplace=True), nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(
            inplace=True), nn.MaxPool2d(2, stride=2, ceil_mode=True))
        self.conv4_block = nn.Sequential(nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True), nn.Conv2d(512, 512, 3, padding=1), nn.
            ReLU(inplace=True), nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(
            inplace=True), nn.MaxPool2d(2, stride=2, ceil_mode=True))
        self.conv5_block = nn.Sequential(nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True), nn.Conv2d(512, 512, 3, padding=1), nn.
            ReLU(inplace=True), nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(
            inplace=True), nn.MaxPool2d(2, stride=2, ceil_mode=True))
        self.classifier = nn.Sequential(nn.Conv2d(512, 4096, 7), nn.ReLU(
            inplace=True), nn.Dropout2d(), nn.Conv2d(4096, 4096, 1), nn.
            ReLU(inplace=True), nn.Dropout2d(), nn.Conv2d(4096, self.
            n_classes, 1))
        if self.module_type == '16s' or self.module_type == '8s':
            self.score_pool4 = nn.Conv2d(512, self.n_classes, 1)
        if self.module_type == '8s':
            self.score_pool3 = nn.Conv2d(256, self.n_classes, 1)
        if pretrained:
            self.init_vgg16()

    def init_vgg16(self):
        vgg16 = models.vgg16(pretrained=True)
        vgg16_features = list(vgg16.features.children())
        conv_blocks = [self.conv1_block, self.conv2_block, self.conv3_block,
            self.conv4_block, self.conv5_block]
        conv_ids_vgg = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 30]]
        for conv_block_id, conv_block in enumerate(conv_blocks):
            conv_id_vgg = conv_ids_vgg[conv_block_id]
            for l1, l2 in zip(conv_block, vgg16_features[conv_id_vgg[0]:
                conv_id_vgg[1]]):
                if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                    assert l1.weight.size() == l2.weight.size()
                    assert l1.bias.size() == l2.bias.size()
                    l1.weight.data = l2.weight.data
                    l1.bias.data = l2.bias.data
        vgg16_classifier = list(vgg16.classifier.children())
        for l1, l2 in zip(self.classifier, vgg16_classifier[0:3]):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Linear):
                l1.weight.data = l2.weight.data.view(l1.weight.size())
                l1.bias.data = l2.bias.data.view(l1.bias.size())
        l1 = self.classifier[6]
        l2 = vgg16_classifier[6]
        if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Linear):
            l1.weight.data = l2.weight.data[:self.n_classes, :].view(l1.
                weight.size())
            l1.bias.data = l2.bias.data[:self.n_classes].view(l1.bias.size())


class mobilenet_conv_bn_relu(nn.Module):
    """
    :param
    """

    def __init__(self, in_channels, out_channels, stride):
        super(mobilenet_conv_bn_relu, self).__init__()
        self.cbr_seq = nn.Sequential(nn.Conv2d(in_channels=in_channels,
            out_channels=out_channels, kernel_size=3, stride=stride,
            padding=1, bias=False), nn.BatchNorm2d(num_features=
            out_channels), nn.ReLU(inplace=True))

    def forward(self, x):
        """
        :param x:
        :return:
        """
        x = self.cbr_seq(x)
        return x


class mobilenet_conv_dw_relu(nn.Module):
    """
    :param
    """

    def __init__(self, in_channels, out_channels, stride):
        super(mobilenet_conv_dw_relu, self).__init__()
        self.cbr_seq = nn.Sequential(nn.Conv2d(in_channels=in_channels,
            out_channels=in_channels, kernel_size=3, stride=stride, padding
            =1, groups=in_channels, bias=False), nn.BatchNorm2d(
            num_features=in_channels), nn.ReLU(inplace=True), nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size
            =1, stride=1, padding=0, bias=False), nn.BatchNorm2d(
            num_features=out_channels), nn.ReLU(inplace=True))

    def forward(self, x):
        """
        :param x:
        :return:
        """
        x = self.cbr_seq(x)
        return x


class fcn_MobileNet(nn.Module):
    """
    :param
    """

    def __init__(self, module_type='32s', n_classes=21, pretrained=True):
        super(fcn_MobileNet, self).__init__()
        self.n_classes = n_classes
        self.module_type = module_type
        self.conv1_bn = mobilenet_conv_bn_relu(3, 32, 2)
        self.conv2_dw = mobilenet_conv_dw_relu(32, 64, 1)
        self.conv3_dw = mobilenet_conv_dw_relu(64, 128, 2)
        self.conv4_dw = mobilenet_conv_dw_relu(128, 128, 1)
        self.conv5_dw = mobilenet_conv_dw_relu(128, 256, 2)
        self.conv6_dw = mobilenet_conv_dw_relu(256, 256, 1)
        self.conv7_dw = mobilenet_conv_dw_relu(256, 512, 2)
        self.conv8_dw = mobilenet_conv_dw_relu(512, 512, 1)
        self.conv9_dw = mobilenet_conv_dw_relu(512, 512, 1)
        self.conv10_dw = mobilenet_conv_dw_relu(512, 512, 1)
        self.conv11_dw = mobilenet_conv_dw_relu(512, 512, 1)
        self.conv12_dw = mobilenet_conv_dw_relu(512, 512, 1)
        self.conv13_dw = mobilenet_conv_dw_relu(512, 1024, 2)
        self.conv14_dw = mobilenet_conv_dw_relu(1024, 1024, 1)
        self.classifier = nn.Conv2d(1024, self.n_classes, 1)
        if self.module_type == '16s' or self.module_type == '8s':
            self.score_pool4 = nn.Conv2d(512, self.n_classes, 1)
        if self.module_type == '8s':
            self.score_pool3 = nn.Conv2d(256, self.n_classes, 1)
        if pretrained:
            self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=False):
        model_checkpoint_path = os.path.expanduser(
            '~/.torch/models/mobilenet_sgd_rmsprop_69.526.tar')
        if os.path.exists(model_checkpoint_path):
            model_checkpoint = torch.load(model_checkpoint_path,
                map_location='cpu')
            pretrained_dict = model_checkpoint['state_dict']
            model_dict = self.state_dict()
            model_dict_keys = model_dict.keys()
            new_dict = {}
            for dict_index, (k, v) in enumerate(pretrained_dict.items()):
                if k == 'module.fc.weight':
                    break
                new_k = model_dict_keys[dict_index]
                new_v = v
                new_dict[new_k] = new_v
            model_dict.update(new_dict)
            self.load_state_dict(model_dict)

    def forward(self, x):
        """
        :param x:
        :return:
        """
        x_size = x.size()[2:]
        x_conv1 = self.conv1_bn(x)
        x_conv2 = self.conv2_dw(x_conv1)
        x_conv3 = self.conv3_dw(x_conv2)
        x_conv4 = self.conv4_dw(x_conv3)
        x_conv5 = self.conv5_dw(x_conv4)
        x_conv6 = self.conv6_dw(x_conv5)
        x_conv7 = self.conv7_dw(x_conv6)
        x_conv8 = self.conv8_dw(x_conv7)
        x_conv9 = self.conv9_dw(x_conv8)
        x_conv10 = self.conv10_dw(x_conv9)
        x_conv11 = self.conv11_dw(x_conv10)
        x_conv12 = self.conv12_dw(x_conv11)
        x_conv13 = self.conv13_dw(x_conv12)
        x = self.conv14_dw(x_conv13)
        score = self.classifier(x)
        if self.module_type == '16s' or self.module_type == '8s':
            score_pool4 = self.score_pool4(x_conv12)
        if self.module_type == '8s':
            score_pool3 = self.score_pool3(x_conv6)
        if self.module_type == '16s' or self.module_type == '8s':
            score = F.upsample_bilinear(score, score_pool4.size()[2:])
            score += score_pool4
        if self.module_type == '8s':
            score = F.upsample_bilinear(score, score_pool3.size()[2:])
            score += score_pool3
        out = F.upsample_bilinear(score, x_size)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
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


def conv1x1(in_channels, out_channels, groups=1):
    """1x1 convolution with padding
    - Normal pointwise convolution When groups == 1
    - Grouped pointwise convolution when groups > 1
    1x1卷积，groups==1那么正常卷积，否则grouped卷积
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=
        groups, stride=1)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
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


class fcn_resnet(nn.Module):

    def __init__(self, block, layers, module_type='32s', n_classes=21,
        pretrained=False, upsample_method='upsample_bilinear'):
        """
        :param block:
        :param layers:
        :param module_type:
        :param n_classes:
        :param pretrained:
        :param upsample_method: 'upsample_bilinear' or 'ConvTranspose2d'
        """
        super(fcn_resnet, self).__init__()
        self.n_classes = n_classes
        self.module_type = module_type
        self.upsample_method = upsample_method
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.classifier = nn.Conv2d(512 * block.expansion, self.n_classes, 1)
        if self.upsample_method == 'upsample_bilinear':
            pass
        elif self.upsample_method == 'ConvTranspose2d':
            self.upsample_1 = nn.ConvTranspose2d(self.n_classes, self.
                n_classes, 3, stride=2, padding=1)
            self.upsample_2 = nn.ConvTranspose2d(self.n_classes, self.
                n_classes, 3, stride=2)
        if self.module_type == '16s' or self.module_type == '8s':
            self.score_pool4 = nn.Conv2d(256, self.n_classes, 1)
        if self.module_type == '8s':
            self.score_pool3 = nn.Conv2d(128, self.n_classes, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight, mode='fan_out',
                    nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes *
                block.expansion, stride), nn.BatchNorm2d(planes * block.
                expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x_size = x.size()[2:]
        x_conv1 = self.conv1(x)
        x = self.bn1(x_conv1)
        x = self.relu(x)
        x = self.maxpool(x)
        x_layer1 = self.layer1(x)
        x_layer2 = self.layer2(x_layer1)
        x_layer3 = self.layer3(x_layer2)
        x = self.layer4(x_layer3)
        score = self.classifier(x)
        if self.module_type == '16s' or self.module_type == '8s':
            score_pool4 = self.score_pool4(x_layer3)
        if self.module_type == '8s':
            score_pool3 = self.score_pool3(x_layer2)
        if self.module_type == '16s' or self.module_type == '8s':
            if self.upsample_method == 'upsample_bilinear':
                score = F.upsample_bilinear(score, score_pool4.size()[2:])
            elif self.upsample_method == 'ConvTranspose2d':
                score = self.upsample_1(score)
            score += score_pool4
        if self.module_type == '8s':
            if self.upsample_method == 'upsample_bilinear':
                score = F.upsample_bilinear(score, score_pool3.size()[2:])
            elif self.upsample_method == 'ConvTranspose2d':
                score = self.upsample_2(score)
            score += score_pool3
        out = F.upsample_bilinear(score, x_size)
        return out

    def init_weight(self, model_name):
        pretrain_model = None
        if model_name == 'fcn_resnet18':
            pretrain_model = models.resnet18(pretrained=True)
        elif model_name == 'fcn_resnet34':
            pretrain_model = models.resnet34(pretrained=True)
        elif model_name == 'fcn_resnet50':
            pretrain_model = models.resnet50(pretrained=True)
        elif model_name == 'fcn_resnet101':
            pretrain_model = models.resnet101(pretrained=True)
        elif model_name == 'fcn_resnet152':
            pretrain_model = models.resnet152(pretrained=True)
        if pretrain_model is not None:
            self.conv1.weight.data = pretrain_model.conv1.weight.data
            if self.conv1.bias is not None:
                self.conv1.bias.data = pretrain_model.conv1.bias.data
            initial_convs = []
            pretrain_model_convs = []
            layers = [self.layer1, self.layer2, self.layer3, self.layer3]
            for layer in layers:
                layer1_list = list(layer.children())
                for layer1_list_block in layer1_list:
                    layer1_list_block_list = list(layer1_list_block.children())
                    for layer1_list_item in layer1_list_block_list:
                        if isinstance(layer1_list_item, nn.Conv2d):
                            initial_convs.append(layer1_list_item)
            layers = [pretrain_model.layer1, pretrain_model.layer2,
                pretrain_model.layer3, pretrain_model.layer3]
            for layer in layers:
                layer1_list = list(layer.children())
                for layer1_list_block in layer1_list:
                    layer1_list_block_list = list(layer1_list_block.children())
                    for layer1_list_item in layer1_list_block_list:
                        if isinstance(layer1_list_item, nn.Conv2d):
                            pretrain_model_convs.append(layer1_list_item)
            for l1, l2 in zip(initial_convs, pretrain_model_convs):
                if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                    assert l1.weight.size() == l2.weight.size()
                    l1.weight.data = l2.weight.data
                    if l1.bias is not None and l2.bias is not None:
                        assert l1.bias.size() == l2.bias.size()
                        l1.bias.data = l2.bias.data


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x


class ShuffleUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=3, grouped_conv=
        True, combine='add'):
        """
        :param in_channels: ShuffleUnit输入通道数
        :param out_channels: ShuffleUnit输出通道数
        :param groups: ShuffleUnit分组groups
        :param grouped_conv: 是否在第一个1x1卷积使用groued卷积
        :param combine: combine使用element wise add or concat
        """
        super(ShuffleUnit, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grouped_conv = grouped_conv
        self.combine = combine
        self.groups = groups
        self.bottleneck_channels = self.out_channels // 4
        if self.combine == 'add':
            self.depthwise_stride = 1
            self._combine_func = self._add
        elif self.combine == 'concat':
            self.depthwise_stride = 2
            self._combine_func = self._concat
            self.out_channels -= self.in_channels
        else:
            raise ValueError(
                'Cannot combine tensors with "{}"Only "add" and "concat" aresupported'
                .format(self.combine))
        self.first_1x1_groups = self.groups if grouped_conv else 1
        self.g_conv_1x1_compress = self._make_grouped_conv1x1(self.
            in_channels, self.bottleneck_channels, self.first_1x1_groups,
            batch_norm=True, relu=True)
        self.depthwise_conv3x3 = conv3x3(self.bottleneck_channels, self.
            bottleneck_channels, stride=self.depthwise_stride, groups=self.
            bottleneck_channels)
        self.bn_after_depthwise = nn.BatchNorm2d(self.bottleneck_channels)
        self.g_conv_1x1_expand = self._make_grouped_conv1x1(self.
            bottleneck_channels, self.out_channels, self.groups, batch_norm
            =True, relu=False)

    @staticmethod
    def _add(x, out):
        return x + out

    @staticmethod
    def _concat(x, out):
        return torch.cat((x, out), 1)

    def _make_grouped_conv1x1(self, in_channels, out_channels, groups,
        batch_norm=True, relu=False):
        modules = OrderedDict()
        conv = conv1x1(in_channels, out_channels, groups=groups)
        modules['conv1x1'] = conv
        if batch_norm:
            modules['batch_norm'] = nn.BatchNorm2d(out_channels)
        if relu:
            modules['relu'] = nn.ReLU()
        if len(modules) > 1:
            return nn.Sequential(modules)
        else:
            return conv

    def forward(self, x):
        residual = x
        if self.combine == 'concat':
            residual = F.avg_pool2d(residual, kernel_size=3, stride=2,
                padding=1)
        out = self.g_conv_1x1_compress(x)
        out = channel_shuffle(out, self.groups)
        out = self.depthwise_conv3x3(out)
        out = self.bn_after_depthwise(out)
        out = self.g_conv_1x1_expand(out)
        out = self._combine_func(residual, out)
        return F.relu(out)


class ShuffleNet(nn.Module):
    """ShuffleNet implementation.
    """

    def __init__(self, groups=3, in_channels=3, n_classes=1000, pretrained=
        False):
        """ShuffleNet constructor.
        Arguments:
            groups (int, optional): number of groups to be used in grouped
                1x1 convolutions in each ShuffleUnit. Default is 3 for best
                performance according to original paper.
                在ShuffeleUnit中使用的1x1卷积groups数量
            in_channels (int, optional): number of channels in the input tensor.
                Default is 3 for RGB image inputs.
            n_classes (int, optional): number of classes to predict. Default
                is 1000 for ImageNet.
        """
        super(ShuffleNet, self).__init__()
        self.groups = groups
        self.stage_repeats = [3, 7, 3]
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.pretrained = pretrained
        if groups == 1:
            self.stage_out_channels = [-1, 24, 144, 288, 567]
        elif groups == 2:
            self.stage_out_channels = [-1, 24, 200, 400, 800]
        elif groups == 3:
            self.stage_out_channels = [-1, 24, 240, 480, 960]
        elif groups == 4:
            self.stage_out_channels = [-1, 24, 272, 544, 1088]
        elif groups == 8:
            self.stage_out_channels = [-1, 24, 384, 768, 1536]
        else:
            raise ValueError(
                """{} groups is not supported for
                   1x1 Grouped Convolutions"""
                .format(groups))
        self.conv1 = conv3x3(self.in_channels, self.stage_out_channels[1],
            stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stage2 = self._make_stage(2)
        self.stage3 = self._make_stage(3)
        self.stage4 = self._make_stage(4)
        num_inputs = self.stage_out_channels[-1]
        self.fc = nn.Linear(num_inputs, self.n_classes)
        if self.pretrained:
            self.init_weights()

    def init_weights(self):
        model_checkpoint_path = os.path.expanduser(
            '~/.torch/models/ShuffleNet_1g8_Top1_67.408_Top5_87.258.pth.tar')
        if os.path.exists(model_checkpoint_path):
            pretrained_dict = torch.load(model_checkpoint_path,
                map_location='cpu')
            model_dict = self.state_dict()
            new_dict = {}
            for k, v in pretrained_dict.items():
                new_k = k[k.find('.') + 1:]
                new_v = v
                new_dict[new_k] = new_v
            model_dict.update(new_dict)
            self.load_state_dict(model_dict)

    def _make_stage(self, stage):
        modules = OrderedDict()
        stage_name = 'ShuffleUnit_Stage{}'.format(stage)
        grouped_conv = stage > 2
        first_module = ShuffleUnit(self.stage_out_channels[stage - 1], self
            .stage_out_channels[stage], groups=self.groups, grouped_conv=
            grouped_conv, combine='concat')
        modules[stage_name + '_0'] = first_module
        for i in range(self.stage_repeats[stage - 2]):
            name = stage_name + '_{}'.format(i + 1)
            module = ShuffleUnit(self.stage_out_channels[stage], self.
                stage_out_channels[stage], groups=self.groups, grouped_conv
                =True, combine='add')
            modules[name] = module
        return nn.Sequential(modules)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = F.avg_pool2d(x, x.data.size()[-2:])
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class fcn_shufflenet(nn.Module):

    def __init__(self, module_type='32s', n_classes=21, pretrained=False,
        upsample_method='upsample_bilinear'):
        super(fcn_shufflenet, self).__init__()
        self.n_classes = n_classes
        self.module_type = module_type
        self.pretrained = pretrained
        self.upsample_method = upsample_method
        self.shufflenet = ShuffleNet(groups=8, in_channels=3, n_classes=
            1000, pretrained=self.pretrained)
        self.classifier = nn.Conv2d(self.shufflenet.stage_out_channels[4],
            self.n_classes, 1)
        if self.upsample_method == 'upsample_bilinear':
            pass
        elif self.upsample_method == 'ConvTranspose2d':
            self.upsample_1 = nn.ConvTranspose2d(self.n_classes, self.
                n_classes, 3, stride=2, padding=1)
            self.upsample_2 = nn.ConvTranspose2d(self.n_classes, self.
                n_classes, 3, stride=2)
        if self.module_type == '16s' or self.module_type == '8s':
            self.score_16s_conv = nn.Conv2d(self.shufflenet.
                stage_out_channels[3], self.n_classes, 1)
        if self.module_type == '8s':
            self.score_8s_conv = nn.Conv2d(self.shufflenet.
                stage_out_channels[2], self.n_classes, 1)

    def forward(self, x):
        x_size = x.size()[2:]
        features = []
        for name, module in self.shufflenet._modules.items()[:-1]:
            if name in ['stage3', 'stage4']:
                features.append(x)
            x = module(x)
        score = self.classifier(x)
        if self.module_type == '16s' or self.module_type == '8s':
            score_16s_out = self.score_16s_conv(features[1])
        if self.module_type == '8s':
            score_8s_out = self.score_8s_conv(features[0])
        if self.module_type == '16s' or self.module_type == '8s':
            if self.upsample_method == 'upsample_bilinear':
                score = F.upsample_bilinear(score, score_16s_out.size()[2:])
            elif self.upsample_method == 'ConvTranspose2d':
                score = self.upsample_1(score)
            score += score_16s_out
        if self.module_type == '8s':
            if self.upsample_method == 'upsample_bilinear':
                score = F.upsample_bilinear(score, score_8s_out.size()[2:])
            elif self.upsample_method == 'ConvTranspose2d':
                score = self.upsample_2(score)
            score += score_8s_out
        out = F.upsample_bilinear(score, x_size)
        return out


class RU(nn.Module):
    """
    Residual Unit for FRRN
    """

    def __init__(self, channels, kernel_size=3, strides=1, group_norm=False,
        n_groups=None):
        super(RU, self).__init__()
        self.group_norm = group_norm
        self.n_groups = n_groups
        if self.group_norm:
            self.conv1 = conv2DGroupNormRelu(channels, channels,
                kernel_size=kernel_size, stride=strides, padding=1, bias=
                False, n_groups=self.n_groups)
            self.conv2 = conv2DGroupNorm(channels, channels, kernel_size=
                kernel_size, stride=strides, padding=1, bias=False,
                n_groups=self.n_groups)
        else:
            self.conv1 = conv2DBatchNormRelu(channels, channels,
                kernel_size=kernel_size, stride=strides, padding=1, bias=False)
            self.conv2 = conv2DBatchNorm(channels, channels, kernel_size=
                kernel_size, stride=strides, padding=1, bias=False)

    def forward(self, x):
        incoming = x
        x = self.conv1(x)
        x = self.conv2(x)
        return x + incoming


class FRRU(nn.Module):
    """
    Full Resolution Residual Unit for FRRN
    """

    def __init__(self, prev_channels, out_channels, scale, group_norm=False,
        n_groups=None):
        super(FRRU, self).__init__()
        self.scale = scale
        self.prev_channels = prev_channels
        self.out_channels = out_channels
        self.group_norm = group_norm
        self.n_groups = n_groups
        if self.group_norm:
            conv_unit = conv2DGroupNormRelu
            self.conv1 = conv_unit(prev_channels + 32, out_channels,
                kernel_size=3, stride=1, padding=1, bias=False, n_groups=
                self.n_groups)
            self.conv2 = conv_unit(out_channels, out_channels, kernel_size=
                3, stride=1, padding=1, bias=False, n_groups=self.n_groups)
        else:
            conv_unit = conv2DBatchNormRelu
            self.conv1 = conv_unit(prev_channels + 32, out_channels,
                kernel_size=3, stride=1, padding=1, bias=False)
            self.conv2 = conv_unit(out_channels, out_channels, kernel_size=
                3, stride=1, padding=1, bias=False)
        self.conv_res = nn.Conv2d(out_channels, 32, kernel_size=1, stride=1,
            padding=0)

    def forward(self, y, z):
        x = torch.cat([y, nn.MaxPool2d(self.scale, self.scale)(z)], dim=1)
        y_prime = self.conv1(x)
        y_prime = self.conv2(y_prime)
        x = self.conv_res(y_prime)
        upsample_size = torch.Size([(_s * self.scale) for _s in y_prime.
            shape[-2:]])
        x = F.upsample(x, size=upsample_size, mode='nearest')
        z_prime = z + x
        return y_prime, z_prime


frrn_specs_dic = {'A': {'encoder': [[3, 96, 2], [4, 192, 4], [2, 384, 8], [
    2, 384, 16]], 'decoder': [[2, 192, 8], [2, 192, 4], [2, 48, 2]]}, 'B':
    {'encoder': [[3, 96, 2], [4, 192, 4], [2, 384, 8], [2, 384, 16], [2, 
    384, 32]], 'decoder': [[2, 192, 16], [2, 192, 8], [2, 192, 4], [2, 48, 2]]}
    }


class frrn(nn.Module):
    """
    Full Resolution Residual Networks for Semantic Segmentation
    URL: https://arxiv.org/abs/1611.08323
    References:
    1) Original Author's code: https://github.com/TobyPDE/FRRN
    2) TF implementation by @kiwonjoon: https://github.com/hiwonjoon/tf-frrn
    """

    def __init__(self, n_classes=21, model_type=None, group_norm=False,
        n_groups=16):
        super(frrn, self).__init__()
        self.n_classes = n_classes
        self.model_type = model_type
        self.group_norm = group_norm
        self.n_groups = n_groups
        if self.group_norm:
            self.conv1 = conv2DGroupNormRelu(3, 48, 5, 1, 2)
        else:
            self.conv1 = conv2DBatchNormRelu(3, 48, 5, 1, 2)
        self.up_residual_units = []
        self.down_residual_units = []
        for i in range(3):
            self.up_residual_units.append(RU(channels=48, kernel_size=3,
                strides=1, group_norm=self.group_norm, n_groups=self.n_groups))
            self.down_residual_units.append(RU(channels=48, kernel_size=3,
                strides=1, group_norm=self.group_norm, n_groups=self.n_groups))
        self.up_residual_units = nn.ModuleList(self.up_residual_units)
        self.down_residual_units = nn.ModuleList(self.down_residual_units)
        self.split_conv = nn.Conv2d(48, 32, kernel_size=1, padding=0,
            stride=1, bias=False)
        self.encoder_frru_specs = frrn_specs_dic[self.model_type]['encoder']
        self.decoder_frru_specs = frrn_specs_dic[self.model_type]['decoder']
        prev_channels = 48
        self.encoding_frrus = {}
        for n_blocks, channels, scale in self.encoder_frru_specs:
            for block in range(n_blocks):
                key = '_'.join(map(str, ['encoding_frru', n_blocks,
                    channels, scale, block]))
                setattr(self, key, FRRU(prev_channels=prev_channels,
                    out_channels=channels, scale=scale, group_norm=self.
                    group_norm, n_groups=self.n_groups))
            prev_channels = channels
        self.decoding_frrus = {}
        for n_blocks, channels, scale in self.decoder_frru_specs:
            for block in range(n_blocks):
                key = '_'.join(map(str, ['decoding_frru', n_blocks,
                    channels, scale, block]))
                setattr(self, key, FRRU(prev_channels=prev_channels,
                    out_channels=channels, scale=scale, group_norm=self.
                    group_norm, n_groups=self.n_groups))
            prev_channels = channels
        self.merge_conv = nn.Conv2d(prev_channels + 32, 48, kernel_size=1,
            padding=0, stride=1, bias=False)
        self.classif_conv = nn.Conv2d(48, self.n_classes, kernel_size=1,
            padding=0, stride=1, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        for i in range(3):
            x = self.up_residual_units[i](x)
        y = x
        z = self.split_conv(x)
        prev_channels = 48
        for n_blocks, channels, scale in self.encoder_frru_specs:
            y_pooled = F.max_pool2d(y, stride=2, kernel_size=2, padding=0)
            for block in range(n_blocks):
                key = '_'.join(map(str, ['encoding_frru', n_blocks,
                    channels, scale, block]))
                y, z = getattr(self, key)(y_pooled, z)
            prev_channels = channels
        for n_blocks, channels, scale in self.decoder_frru_specs:
            upsample_size = torch.Size([(_s * 2) for _s in y.size()[-2:]])
            y_upsampled = F.upsample(y, size=upsample_size, mode='bilinear',
                align_corners=True)
            for block in range(n_blocks):
                key = '_'.join(map(str, ['decoding_frru', n_blocks,
                    channels, scale, block]))
                y, z = getattr(self, key)(y_upsampled, z)
            prev_channels = channels
        x = torch.cat([F.upsample(y, scale_factor=2, mode='bilinear',
            align_corners=True), z], dim=1)
        x = self.merge_conv(x)
        for i in range(3):
            x = self.down_residual_units[i](x)
        x = self.classif_conv(x)
        return x


class gcn(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=7):
        super(gcn, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.conv1_1 = nn.Conv2d(in_channels=self.in_channels, out_channels
            =self.out_channels, kernel_size=(self.kernel_size, 1), padding=
            (self.kernel_size // 2, 0))
        self.conv1_2 = nn.Conv2d(in_channels=self.out_channels,
            out_channels=self.out_channels, kernel_size=(1, self.
            kernel_size), padding=(0, self.kernel_size // 2))
        self.conv2_1 = nn.Conv2d(in_channels=self.in_channels, out_channels
            =self.out_channels, kernel_size=(1, self.kernel_size), padding=
            (0, self.kernel_size // 2))
        self.conv2_2 = nn.Conv2d(in_channels=self.out_channels,
            out_channels=self.out_channels, kernel_size=(self.kernel_size, 
            1), padding=(self.kernel_size // 2, 0))

    def forward(self, x):
        x_conv1_1 = self.conv1_1(x)
        x_conv1_2 = self.conv1_2(x_conv1_1)
        x_conv2_1 = self.conv2_1(x)
        x_conv2_2 = self.conv2_2(x_conv2_1)
        x_out = x_conv1_2 + x_conv2_2
        return x_out


class boundary_refine(nn.Module):

    def __init__(self, in_channels):
        super(boundary_refine, self).__init__()
        self.in_channels = in_channels
        self.bn = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=
            self.in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=self.in_channels, out_channels=
            self.in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x_out = residual + x
        return x_out


class gcn_resnet(nn.Module):

    def __init__(self, n_classes=21, pretrained=False, expansion=4, model=
        'resnet18'):
        super(gcn_resnet, self).__init__()
        self.n_classes = n_classes
        self.pretrained = pretrained
        self.expansion = expansion
        backbone = eval(model)(self.pretrained)
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.gcn1 = gcn(in_channels=64 * self.expansion, out_channels=self.
            n_classes)
        self.gcn2 = gcn(in_channels=128 * self.expansion, out_channels=self
            .n_classes)
        self.gcn3 = gcn(in_channels=256 * self.expansion, out_channels=self
            .n_classes)
        self.gcn4 = gcn(in_channels=512 * self.expansion, out_channels=self
            .n_classes)
        self.br1_1 = boundary_refine(in_channels=self.n_classes)
        self.br1_2 = boundary_refine(in_channels=self.n_classes)
        self.br1_3 = boundary_refine(in_channels=self.n_classes)
        self.br1_4 = boundary_refine(in_channels=self.n_classes)
        self.br2_1 = boundary_refine(in_channels=self.n_classes)
        self.br2_2 = boundary_refine(in_channels=self.n_classes)
        self.br2_3 = boundary_refine(in_channels=self.n_classes)
        self.br3_1 = boundary_refine(in_channels=self.n_classes)
        self.br3_2 = boundary_refine(in_channels=self.n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x_1 = self.layer1(x)
        x_2 = self.layer2(x_1)
        x_3 = self.layer3(x_2)
        x_4 = self.layer4(x_3)
        x_4 = self.gcn4(x_4)
        x_4 = self.br1_4(x_4)
        x_4_up = F.upsample_bilinear(x_4, x_3.size()[2:])
        x_3 = self.gcn3(x_3)
        x_3 = self.br1_3(x_3)
        x_3_skip = x_3 + x_4_up
        x_3_skip = self.br2_3(x_3_skip)
        x_3_up = F.upsample_bilinear(x_3_skip, x_2.size()[2:])
        x_2 = self.gcn2(x_2)
        x_2 = self.br1_2(x_2)
        x_2_skip = x_2 + x_3_up
        x_2_skip = self.br2_2(x_2_skip)
        x_2_up = F.upsample_bilinear(x_2_skip, x_1.size()[2:])
        x_1 = self.gcn1(x_1)
        x_1 = self.br1_1(x_1)
        x_1_skip = x_1 + x_2_up
        x_1_skip = self.br2_1(x_1_skip)
        x_1_up = F.upsample_bilinear(x_1_skip, scale_factor=2)
        x_out = self.br3_1(x_1_up)
        x_out = F.upsample_bilinear(x_out, scale_factor=2)
        x_out = self.br3_2(x_out)
        return x_out


class LRNRefineUnit(nn.Module):

    def __init__(self, R_channel, M_channel):
        super(LRNRefineUnit, self).__init__()
        self.R_channel = R_channel
        self.M_channel = M_channel
        self.conv_M2m = conv3x3_bn_relu(in_planes=self.M_channel,
            out_planes=self.M_channel, stride=2)
        self.conv_R2r = nn.Conv2d(in_channels=self.R_channel + self.
            M_channel, out_channels=self.R_channel, kernel_size=3, padding=1)

    def forward(self, Rf, Mf):
        Mf_size = Mf.size()
        mf = self.conv_M2m(Mf)
        rf = torch.cat((mf[:, :, :Rf.shape[2], :Rf.shape[3]], Rf), 1)
        rf = self.conv_R2r(rf)
        out = F.upsample_bilinear(rf, Mf_size[2:])
        return out


class lrn_vgg16(nn.Module):

    def __init__(self, n_classes=21, pretrained=False):
        super(lrn_vgg16, self).__init__()
        self.n_classes = n_classes
        vgg16 = vgg.vgg16(pretrained=pretrained)
        self.encoder = vgg16.features
        self.out_conv = nn.Conv2d(in_channels=512, out_channels=self.
            n_classes, kernel_size=1)
        self.refine_units = []
        self.refine_1 = LRNRefineUnit(self.n_classes, 512)
        self.refine_units.append(self.refine_1)
        self.refine_2 = LRNRefineUnit(self.n_classes, 512)
        self.refine_units.append(self.refine_2)
        self.refine_3 = LRNRefineUnit(self.n_classes, 256)
        self.refine_units.append(self.refine_3)
        self.refine_4 = LRNRefineUnit(self.n_classes, 128)
        self.refine_units.append(self.refine_4)
        self.refine_5 = LRNRefineUnit(self.n_classes, 64)
        self.refine_units.append(self.refine_5)

    def forward(self, x):
        encoder_features = []
        for name, module in self.encoder._modules.items():
            x = module(x)
            if name in ['3', '8', '15', '22', '29']:
                encoder_features.append(x)
        x = self.out_conv(x)
        out_s = []
        out_s.append(x)
        encoder_features.reverse()
        for refine_id, encoder_feature in enumerate(encoder_features):
            x = self.refine_units[refine_id](x, encoder_feature)
            out_s.append(x)
        return out_s


class pspnet(nn.Module):

    def __init__(self, n_classes=21, block_config=[3, 4, 23, 3]):
        super(pspnet, self).__init__()
        self.block_config = block_config
        self.n_classes = n_classes
        self.convbnrelu1_1 = conv2DBatchNormRelu(in_channels=3, kernel_size
            =3, out_channels=64, padding=1, stride=2, bias=False)
        self.convbnrelu1_2 = conv2DBatchNormRelu(in_channels=64,
            kernel_size=3, out_channels=64, padding=1, stride=1, bias=False)
        self.convbnrelu1_3 = conv2DBatchNormRelu(in_channels=64,
            kernel_size=3, out_channels=128, padding=1, stride=1, bias=False)
        self.res_block2 = residualBlockPSP(self.block_config[0], 128, 64, 
            256, 1, 1)
        self.res_block3 = residualBlockPSP(self.block_config[1], 256, 128, 
            512, 2, 1)
        self.res_block4 = residualBlockPSP(self.block_config[2], 512, 256, 
            1024, 1, 2)
        self.res_block5 = residualBlockPSP(self.block_config[3], 1024, 512,
            2048, 1, 4)
        self.pyramid_pooling = pyramidPooling(2048, [6, 3, 2, 1])
        self.cbr_final = conv2DBatchNormRelu(4096, 512, 3, 1, 1, False)
        self.classification = nn.Conv2d(512, n_classes, 1, 1, 0)

    def forward(self, x):
        inp_shape = x.size()[2:]
        x = self.convbnrelu1_3(self.convbnrelu1_2(self.convbnrelu1_1(x)))
        x = F.max_pool2d(x, 3, 2, 1)
        x = self.res_block5(self.res_block4(self.res_block3(self.res_block2
            (x))))
        x = self.pyramid_pooling(x)
        x = F.dropout2d(self.cbr_final(x), p=0.1, inplace=True)
        x = self.classification(x)
        x = F.upsample(x, size=inp_shape, mode='bilinear')
        return x

    def load_pretrained_model(self, model_path):
        """
        Load weights from caffemodel w/o caffe dependency
        and plug them in corresponding modules
        """
        ltypes = ['BNData', 'ConvolutionData', 'HoleConvolutionData']

        def _get_layer_params(layer, ltype):
            if ltype == 'BNData':
                n_channels = layer.blobs[0].shape.dim[1]
                gamma = np.array([w for w in layer.blobs[0].data]).reshape(
                    n_channels)
                beta = np.array([w for w in layer.blobs[1].data]).reshape(
                    n_channels)
                mean = np.array([w for w in layer.blobs[2].data]).reshape(
                    n_channels)
                var = np.array([w for w in layer.blobs[3].data]).reshape(
                    n_channels)
                return [mean, var, gamma, beta]
            elif ltype in ['ConvolutionData', 'HoleConvolutionData']:
                is_bias = layer.convolution_param.bias_term
                shape = [int(d) for d in layer.blobs[0].shape.dim]
                weights = np.array([w for w in layer.blobs[0].data]).reshape(
                    shape)
                bias = []
                if is_bias:
                    bias = np.array([w for w in layer.blobs[1].data]).reshape(
                        shape[0])
                return [weights, bias]
            elif ltype == 'InnerProduct':
                raise Exception('Fully connected layers {}, not supported'.
                    format(ltype))
            else:
                raise Exception('Unkown layer type {}'.format(ltype))
        net = caffe_pb2.NetParameter()
        with open(model_path, 'rb') as model_file:
            net.MergeFromString(model_file.read())
        layer_types = {}
        layer_params = {}
        for l in net.layer:
            lname = l.name
            ltype = l.type
            if ltype in ltypes:
                None
                layer_types[lname] = ltype
                layer_params[lname] = _get_layer_params(l, ltype)

        def _no_affine_bn(module=None):
            if isinstance(module, nn.BatchNorm2d):
                module.affine = False
            if len([m for m in module.children()]) > 0:
                for child in module.children():
                    _no_affine_bn(child)

        def _transfer_conv(layer_name, module):
            weights, bias = layer_params[layer_name]
            w_shape = np.array(module.weight.size())
            np.testing.assert_array_equal(weights.shape, w_shape)
            None
            module.weight.data = torch.from_numpy(weights)
            if len(bias) != 0:
                b_shape = np.array(module.bias.size())
                np.testing.assert_array_equal(bias.shape, b_shape)
                None
                module.bias.data = torch.from_numpy(bias)

        def _transfer_conv_bn(conv_layer_name, mother_module):
            conv_module = mother_module[0]
            bn_module = mother_module[1]
            _transfer_conv(conv_layer_name, conv_module)
            mean, var, gamma, beta = layer_params[conv_layer_name + '/bn']
            None
            bn_module.running_mean = torch.from_numpy(mean)
            bn_module.running_var = torch.from_numpy(var)
            bn_module.weight.data = torch.from_numpy(gamma)
            bn_module.bias.data = torch.from_numpy(beta)

        def _transfer_residual(prefix, block):
            block_module, n_layers = block[0], block[1]
            bottleneck = block_module.layers[0]
            bottleneck_conv_bn_dic = {(prefix + '_1_1x1_reduce'):
                bottleneck.cbr1.cbr_unit, (prefix + '_1_3x3'): bottleneck.
                cbr2.cbr_unit, (prefix + '_1_1x1_proj'): bottleneck.cb4.
                cb_unit, (prefix + '_1_1x1_increase'): bottleneck.cb3.cb_unit}
            for k, v in bottleneck_conv_bn_dic.items():
                _transfer_conv_bn(k, v)
            for layer_idx in range(2, n_layers + 1):
                residual_layer = block_module.layers[layer_idx - 1]
                residual_conv_bn_dic = {'_'.join(map(str, [prefix,
                    layer_idx, '1x1_reduce'])): residual_layer.cbr1.
                    cbr_unit, '_'.join(map(str, [prefix, layer_idx, '3x3'])
                    ): residual_layer.cbr2.cbr_unit, '_'.join(map(str, [
                    prefix, layer_idx, '1x1_increase'])): residual_layer.
                    cb3.cb_unit}
                for k, v in residual_conv_bn_dic.items():
                    _transfer_conv_bn(k, v)
        convbn_layer_mapping = {'conv1_1_3x3_s2': self.convbnrelu1_1.
            cbr_unit, 'conv1_2_3x3': self.convbnrelu1_2.cbr_unit,
            'conv1_3_3x3': self.convbnrelu1_3.cbr_unit,
            'conv5_3_pool6_conv': self.pyramid_pooling.paths[0].cbr_unit,
            'conv5_3_pool3_conv': self.pyramid_pooling.paths[1].cbr_unit,
            'conv5_3_pool2_conv': self.pyramid_pooling.paths[2].cbr_unit,
            'conv5_3_pool1_conv': self.pyramid_pooling.paths[3].cbr_unit,
            'conv5_4': self.cbr_final.cbr_unit}
        residual_layers = {'conv2': [self.res_block2, self.block_config[0]],
            'conv3': [self.res_block3, self.block_config[1]], 'conv4': [
            self.res_block4, self.block_config[2]], 'conv5': [self.
            res_block5, self.block_config[3]]}
        for k, v in convbn_layer_mapping.items():
            _transfer_conv_bn(k, v)
        _transfer_conv('conv6', self.classification)
        for k, v in residual_layers.items():
            _transfer_residual(k, v)

    def tile_predict(self, img, input_size=[713, 713]):
        """
        Predict by takin overlapping tiles from the image.
    
        :param img: np.ndarray with shape [C, H, W] in BGR format
            
        Source: https://github.com/mitmul/chainer-pspnet/blob/master/pspnet.py#L408-L448
        Adapted for PyTorch
        # TODO: Remove artifacts in last window.
        """
        setattr(self, 'input_size', input_size)

        def _pad_img(img):
            if img.shape[1] < self.input_size[0]:
                pad_h = self.input_size[0] - img.shape[1]
                img = np.pad(img, ((0, 0), (0, pad_h), (0, 0)), 'constant')
            else:
                pad_h = 0
            if img.shape[2] < self.input_size[1]:
                pad_w = self.input_size[1] - img.shape[2]
                img = np.pad(img, ((0, 0), (0, 0), (0, pad_w)), 'constant')
            else:
                pad_w = 0
            return img, pad_h, pad_w
        ori_rows, ori_cols = img.shape[1:]
        long_size = max(ori_rows, ori_cols)
        if long_size > max(self.input_size):
            count = np.zeros((ori_rows, ori_cols))
            pred = np.zeros((1, self.n_classes, ori_rows, ori_cols))
            stride_rate = 2 / 3.0
            stride = int(ceil(self.input_size[0] * stride_rate)), int(ceil(
                self.input_size[1] * stride_rate))
            hh = int(ceil((ori_rows - self.input_size[0]) / stride[0])) + 1
            ww = int(ceil((ori_cols - self.input_size[1]) / stride[1])) + 1
            for yy in range(hh):
                for xx in range(ww):
                    sy, sx = yy * stride[0], xx * stride[1]
                    ey, ex = sy + self.input_size[0], sx + self.input_size[1]
                    img_sub = img[:, sy:ey, sx:ex]
                    img_sub, pad_h, pad_w = _pad_img(img_sub)
                    img_sub_flp = np.copy(img_sub[:, :, ::-1])
                    inp = Variable(torch.unsqueeze(torch.from_numpy(img_sub
                        ).float(), 0), volatile=True)
                    flp = Variable(torch.unsqueeze(torch.from_numpy(
                        img_sub_flp).float(), 0), volatile=True)
                    psub1 = F.softmax(self.forward(inp), dim=1).data.cpu(
                        ).numpy()
                    psub2 = F.softmax(self.forward(flp), dim=1).data.cpu(
                        ).numpy()
                    psub = (psub1 + psub2[:, :, :, ::-1]) / 2.0
                    if sy + self.input_size[0] > ori_rows:
                        psub = psub[:, :, :-pad_h, :]
                    if sx + self.input_size[1] > ori_cols:
                        psub = psub[:, :, :, :-pad_w]
                    pred[:, :, sy:ey, sx:ex] = psub
                    count[sy:ey, sx:ex] += 1
            score = (pred / count[None, None, ...]).astype(np.float32)
        else:
            img, pad_h, pad_w = _pad_img(img)
            inp = Variable(torch.unsqueeze(torch.from_numpy(img), 0),
                volatile=True)
            pred1 = F.softmax(self.forward(inp), dim=1).data.cpu().numpy()
            pred2 = F.softmax(self.forward(inp[:, :, :, ::-1]), dim=1
                ).data.cpu().numpy()
            pred = (pred1 + pred2[:, :, :, ::-1]) / 2.0
            score = pred[:, :, :self.input_size[0] - pad_h, :self.
                input_size[1] - pad_w]
        tscore = F.upsample(torch.from_numpy(score), size=(ori_rows,
            ori_cols), mode='bilinear')
        score = tscore[0].data.numpy()
        return score / score.sum(axis=0)


class segnet(nn.Module):

    def __init__(self, n_classes=21, pretrained=False):
        super(segnet, self).__init__()
        self.down1 = segnetDown2(3, 64)
        self.down2 = segnetDown2(64, 128)
        self.down3 = segnetDown3(128, 256)
        self.down4 = segnetDown3(256, 512)
        self.down5 = segnetDown3(512, 512)
        self.up5 = segnetUp3(512, 512)
        self.up4 = segnetUp3(512, 256)
        self.up3 = segnetUp3(256, 128)
        self.up2 = segnetUp2(128, 64)
        self.up1 = segnetUp2(64, n_classes)
        self.init_weights(pretrained=pretrained)

    def forward(self, x):
        x, pool_indices1, unpool_shape1 = self.down1(x)
        x, pool_indices2, unpool_shape2 = self.down2(x)
        x, pool_indices3, unpool_shape3 = self.down3(x)
        x, pool_indices4, unpool_shape4 = self.down4(x)
        x, pool_indices5, unpool_shape5 = self.down5(x)
        x = self.up5(x, pool_indices=pool_indices5, unpool_shape=unpool_shape5)
        x = self.up4(x, pool_indices=pool_indices4, unpool_shape=unpool_shape4)
        x = self.up3(x, pool_indices=pool_indices3, unpool_shape=unpool_shape3)
        x = self.up2(x, pool_indices=pool_indices2, unpool_shape=unpool_shape2)
        x = self.up1(x, pool_indices=pool_indices1, unpool_shape=unpool_shape1)
        return x

    def init_weights(self, pretrained=False):
        vgg16 = models.vgg16_bn(pretrained=pretrained)
        vgg16_features = list(vgg16.features.children())
        vgg16_conv_layers = []
        for layer in vgg16_features:
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.BatchNorm2d
                ):
                vgg16_conv_layers.append(layer)
        conv_blocks = [self.down1, self.down2, self.down3, self.down4, self
            .down5]
        segnet_down_conv_layers = []
        for conv_block_id, conv_block in enumerate(conv_blocks):
            conv_block_children = list(conv_block.children())
            for conv_block_child in conv_block_children:
                if isinstance(conv_block_child, conv2DBatchNormRelu):
                    if hasattr(conv_block_child, 'cbr_seq'):
                        layer_lists = list(conv_block_child.cbr_seq)
                        for layer in conv_block_child.cbr_seq:
                            if isinstance(layer, nn.Conv2d) or isinstance(layer
                                , nn.BatchNorm2d):
                                segnet_down_conv_layers.append(layer)
        for l1, l2 in zip(segnet_down_conv_layers, vgg16_conv_layers):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l1.weight.data = l2.weight.data
                l1.bias.data = l2.bias.data


class segnet_vgg19(nn.Module):

    def __init__(self, n_classes=21, pretrained=False):
        super(segnet_vgg19, self).__init__()
        self.down1 = segnetDown2(3, 64)
        self.down2 = segnetDown2(64, 128)
        self.down3 = segnetDown4(128, 256)
        self.down4 = segnetDown4(256, 512)
        self.down5 = segnetDown4(512, 512)
        self.up5 = segnetUp4(512, 512)
        self.up4 = segnetUp4(512, 256)
        self.up3 = segnetUp4(256, 128)
        self.up2 = segnetUp2(128, 64)
        self.up1 = segnetUp2(64, n_classes)
        self.init_vgg19(pretrained=pretrained)

    def forward(self, x):
        x, pool_indices1, unpool_shape1 = self.down1(x)
        x, pool_indices2, unpool_shape2 = self.down2(x)
        x, pool_indices3, unpool_shape3 = self.down3(x)
        x, pool_indices4, unpool_shape4 = self.down4(x)
        x, pool_indices5, unpool_shape5 = self.down5(x)
        x = self.up5(x, pool_indices=pool_indices5, unpool_shape=unpool_shape5)
        x = self.up4(x, pool_indices=pool_indices4, unpool_shape=unpool_shape4)
        x = self.up3(x, pool_indices=pool_indices3, unpool_shape=unpool_shape3)
        x = self.up2(x, pool_indices=pool_indices2, unpool_shape=unpool_shape2)
        x = self.up1(x, pool_indices=pool_indices1, unpool_shape=unpool_shape1)
        return x

    def init_vgg19(self, pretrained=False):
        vgg19 = models.vgg19(pretrained=pretrained)
        vgg19_features = list(vgg19.features.children())
        vgg19_conv_layers = []
        for layer in vgg19_features:
            if isinstance(layer, nn.Conv2d):
                vgg19_conv_layers.append(layer)
        conv_blocks = [self.down1, self.down2, self.down3, self.down4, self
            .down5]
        segnet_down_conv_layers = []
        for conv_block_id, conv_block in enumerate(conv_blocks):
            conv_block_children = list(conv_block.children())
            for conv_block_child in conv_block_children:
                if isinstance(conv_block_child, conv2DBatchNormRelu):
                    if hasattr(conv_block_child, 'cbr_seq'):
                        layer_lists = list(conv_block_child.cbr_seq)
                        for layer in conv_block_child.cbr_seq:
                            if isinstance(layer, nn.Conv2d):
                                segnet_down_conv_layers.append(layer)
        for l1, l2 in zip(segnet_down_conv_layers, vgg19_conv_layers):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l1.weight.data = l2.weight.data
                l1.bias.data = l2.bias.data


class segnet_alignres(nn.Module):

    def __init__(self, n_classes=21, pretrained=False):
        super(segnet_alignres, self).__init__()
        self.down1 = segnetDown2(3, 64)
        self.down2 = segnetDown2(64, 128)
        self.down3 = segnetDown3(128, 256)
        self.down4 = segnetDown3(256, 512)
        self.down5 = segnetDown3(512, 512)
        self.alignres = AlignedResInception(512)
        self.up5 = segnetUp3(512, 512)
        self.up4 = segnetUp3(512, 256)
        self.up3 = segnetUp3(256, 128)
        self.up2 = segnetUp2(128, 64)
        self.up1 = segnetUp2(64, n_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant(m.weight, 1)
                nn.init.constant(m.bias, 0)
        self.init_weights(pretrained=pretrained)

    def forward(self, x):
        x, pool_indices1, unpool_shape1 = self.down1(x)
        x, pool_indices2, unpool_shape2 = self.down2(x)
        x, pool_indices3, unpool_shape3 = self.down3(x)
        x, pool_indices4, unpool_shape4 = self.down4(x)
        x, pool_indices5, unpool_shape5 = self.down5(x)
        x = self.alignres(x)
        x = self.up5(x, pool_indices=pool_indices5, unpool_shape=unpool_shape5)
        x = self.up4(x, pool_indices=pool_indices4, unpool_shape=unpool_shape4)
        x = self.up3(x, pool_indices=pool_indices3, unpool_shape=unpool_shape3)
        x = self.up2(x, pool_indices=pool_indices2, unpool_shape=unpool_shape2)
        x = self.up1(x, pool_indices=pool_indices1, unpool_shape=unpool_shape1)
        return x

    def init_weights(self, pretrained=False):
        vgg16 = models.vgg16(pretrained=pretrained)
        vgg16_features = list(vgg16.features.children())
        vgg16_conv_layers = []
        for layer in vgg16_features:
            if isinstance(layer, nn.Conv2d):
                vgg16_conv_layers.append(layer)
        conv_blocks = [self.down1, self.down2, self.down3, self.down4, self
            .down5]
        segnet_down_conv_layers = []
        for conv_block_id, conv_block in enumerate(conv_blocks):
            conv_block_children = list(conv_block.children())
            for conv_block_child in conv_block_children:
                if isinstance(conv_block_child, conv2DBatchNormRelu):
                    if hasattr(conv_block_child, 'cbr_seq'):
                        layer_lists = list(conv_block_child.cbr_seq)
                        for layer in conv_block_child.cbr_seq:
                            if isinstance(layer, nn.Conv2d):
                                segnet_down_conv_layers.append(layer)
        for l1, l2 in zip(segnet_down_conv_layers, vgg16_conv_layers):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l1.weight.data = l2.weight.data
                l1.bias.data = l2.bias.data


class segnet_squeeze(nn.Module):
    """
    参考论文
    Squeeze-SegNet: A new fast Deep Convolutional Neural Network for Semantic Segmentation
    """

    def __init__(self, n_classes=21, pretrained=False):
        super(segnet_squeeze, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=7, stride=2)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True,
            return_indices=True)
        self.fire2 = Fire(96, 16, 64, 64)
        self.fire3 = Fire(128, 16, 64, 64)
        self.fire4 = Fire(128, 32, 128, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True,
            return_indices=True)
        self.fire5 = Fire(256, 32, 128, 128)
        self.fire6 = Fire(256, 48, 192, 192)
        self.fire7 = Fire(384, 48, 192, 192)
        self.fire8 = Fire(384, 64, 256, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True,
            return_indices=True)
        self.fire9 = Fire(512, 64, 256, 256)
        self.conv10 = nn.Conv2d(512, 1000, kernel_size=1)
        self.relu10 = nn.ReLU(inplace=True)
        self.conv10_D = nn.Conv2d(1000, 512, kernel_size=1)
        self.relu10_D = nn.ReLU(inplace=True)
        self.fire9_D = Fire(512, 64, 256, 256)
        self.unpool3 = nn.MaxUnpool2d(kernel_size=3, stride=2)
        self.fire8_D = Fire(512, 48, 192, 192)
        self.fire7_D = Fire(384, 48, 192, 192)
        self.fire6_D = Fire(384, 32, 128, 128)
        self.fire5_D = Fire(256, 32, 128, 128)
        self.unpool2 = nn.MaxUnpool2d(kernel_size=3, stride=2)
        self.fire4_D = Fire(256, 16, 64, 64)
        self.fire3_D = Fire(128, 16, 64, 64)
        self.fire2_D = Fire(128, 12, 48, 48)
        self.unpool1 = nn.MaxUnpool2d(kernel_size=3, stride=2)
        self.conv1_D = nn.ConvTranspose2d(96, n_classes, kernel_size=10,
            stride=2, padding=1)
        self.init_weights(pretrained)

    def init_weights(self, pretrained=False):
        sequeeze = models.squeezenet1_0(pretrained=pretrained)
        sequeeze_features = list(sequeeze.features.children())
        sequeeze_conv_layers = []
        fire_counts = 0
        for layer in sequeeze_features:
            if isinstance(layer, nn.Conv2d):
                sequeeze_conv_layers.append(layer)
            if isinstance(layer, Fire):
                fire_children = list(layer.children())
                for fire_children_layer in fire_children:
                    if isinstance(fire_children_layer, nn.Conv2d):
                        pass
                        sequeeze_conv_layers.append(fire_children_layer)
        segnet_squeeze_down_conv_layers = []
        features = list(self.children())
        for layer in features:
            if len(segnet_squeeze_down_conv_layers) == len(sequeeze_conv_layers
                ):
                break
            if isinstance(layer, nn.Conv2d):
                segnet_squeeze_down_conv_layers.append(layer)
            if isinstance(layer, Fire):
                fire_children = list(layer.children())
                for fire_children_layer in fire_children:
                    if isinstance(fire_children_layer, nn.Conv2d):
                        pass
                        segnet_squeeze_down_conv_layers.append(
                            fire_children_layer)
        for l1, l2 in zip(segnet_squeeze_down_conv_layers, sequeeze_conv_layers
            ):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l1.weight.data = l2.weight.data
                l1.bias.data = l2.bias.data

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        unpool_shape1 = x.size()
        x, pool_indices1 = self.pool1(x)
        x = self.fire2(x)
        x = self.fire3(x)
        x = self.fire4(x)
        unpool_shape2 = x.size()
        x, pool_indices2 = self.pool2(x)
        x = self.fire5(x)
        x = self.fire6(x)
        x = self.fire7(x)
        x = self.fire8(x)
        unpool_shape3 = x.size()
        x, pool_indices3 = self.pool3(x)
        x = self.fire9(x)
        x = self.conv10(x)
        x = self.relu10(x)
        x = self.conv10_D(x)
        x = self.relu10_D(x)
        x = self.fire9_D(x)
        x = self.unpool3(x, indices=pool_indices3, output_size=unpool_shape3)
        x = self.fire8_D(x)
        x = self.fire7_D(x)
        x = self.fire6_D(x)
        x = self.fire5_D(x)
        x = self.unpool2(x, indices=pool_indices2, output_size=unpool_shape2)
        x = self.fire4_D(x)
        x = self.fire3_D(x)
        x = self.fire2_D(x)
        x = self.unpool1(x, indices=pool_indices1, output_size=unpool_shape1)
        x = self.conv1_D(x)
        return x


class segnet_unet(nn.Module):

    def __init__(self, n_classes=21, pretrained=False):
        super(segnet_unet, self).__init__()
        self.down1 = segnetDown2(3, 64)
        self.down2 = segnetDown2(64, 128)
        self.down3 = segnetDown3(128, 256)
        self.down4 = segnetUNetDown3(256, 512)
        self.down5 = segnetUNetDown3(512, 512)
        self.up5 = segnetUNetUp3(512, 512)
        self.up4 = segnetUNetUp3(512, 256)
        self.up3 = segnetUp3(256, 128)
        self.up2 = segnetUp2(128, 64)
        self.up1 = segnetUp2(64, n_classes)
        self.init_vgg16(pretrained=pretrained)

    def forward(self, x):
        x_down1, pool_indices1, unpool_shape1 = self.down1(x)
        x_down2, pool_indices2, unpool_shape2 = self.down2(x_down1)
        x_down3, pool_indices3, unpool_shape3 = self.down3(x_down2)
        x_down4, pool_indices4, unpool_shape4, x_undown4 = self.down4(x_down3)
        x_down5, pool_indices5, unpool_shape5, x_undown5 = self.down5(x_down4)
        x_up5 = self.up5(x_down5, pool_indices=pool_indices5, unpool_shape=
            unpool_shape5, concat_net=x_undown5)
        x_up4 = self.up4(x_up5, pool_indices=pool_indices4, unpool_shape=
            unpool_shape4, concat_net=x_undown4)
        x_up3 = self.up3(x_up4, pool_indices=pool_indices3, unpool_shape=
            unpool_shape3)
        x_up2 = self.up2(x_up3, pool_indices=pool_indices2, unpool_shape=
            unpool_shape2)
        x_up1 = self.up1(x_up2, pool_indices=pool_indices1, unpool_shape=
            unpool_shape1)
        return x_up1

    def init_vgg16(self, pretrained=False):
        vgg16 = models.vgg16(pretrained=pretrained)
        vgg16_features = list(vgg16.features.children())
        vgg16_conv_layers = []
        for layer in vgg16_features:
            if isinstance(layer, nn.Conv2d):
                vgg16_conv_layers.append(layer)
        conv_blocks = [self.down1, self.down2, self.down3, self.down4, self
            .down5]
        conv_ids_vgg = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 30]]
        segnet_down_conv_layers = []
        for conv_block_id, conv_block in enumerate(conv_blocks):
            conv_block_children = list(conv_block.children())
            for conv_block_child in conv_block_children:
                if isinstance(conv_block_child, conv2DBatchNormRelu):
                    if hasattr(conv_block_child, 'cbr_seq'):
                        layer_lists = list(conv_block_child.cbr_seq)
                        for layer in conv_block_child.cbr_seq:
                            if isinstance(layer, nn.Conv2d):
                                segnet_down_conv_layers.append(layer)
        for l1, l2 in zip(segnet_down_conv_layers, vgg16_conv_layers):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l1.weight.data = l2.weight.data
                l1.bias.data = l2.bias.data


class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes, expand_planes):
        super(Fire, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1,
            stride=1)
        self.relu1 = nn.ELU(inplace=True)
        self.conv2 = nn.Conv2d(squeeze_planes, expand_planes, kernel_size=1,
            stride=1)
        self.conv3 = nn.Conv2d(squeeze_planes, expand_planes, kernel_size=3,
            stride=1, padding=1)
        self.relu2 = nn.ELU(inplace=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        out1 = self.conv2(x)
        out2 = self.conv3(x)
        out = torch.cat([out1, out2], 1)
        out = self.relu2(out)
        return out


class ParallelDilatedConv(nn.Module):

    def __init__(self, inplanes, planes):
        super(ParallelDilatedConv, self).__init__()
        self.dilated_conv_1 = nn.Conv2d(inplanes, planes, kernel_size=3,
            stride=1, padding=1, dilation=1)
        self.dilated_conv_2 = nn.Conv2d(inplanes, planes, kernel_size=3,
            stride=1, padding=2, dilation=2)
        self.dilated_conv_3 = nn.Conv2d(inplanes, planes, kernel_size=3,
            stride=1, padding=3, dilation=3)
        self.dilated_conv_4 = nn.Conv2d(inplanes, planes, kernel_size=3,
            stride=1, padding=4, dilation=4)
        self.relu1 = nn.ELU(inplace=True)
        self.relu2 = nn.ELU(inplace=True)
        self.relu3 = nn.ELU(inplace=True)
        self.relu4 = nn.ELU(inplace=True)

    def forward(self, x):
        out1 = self.dilated_conv_1(x)
        out2 = self.dilated_conv_2(x)
        out3 = self.dilated_conv_3(x)
        out4 = self.dilated_conv_4(x)
        out1 = self.relu1(out1)
        out2 = self.relu2(out2)
        out3 = self.relu3(out3)
        out4 = self.relu4(out4)
        out = out1 + out2 + out3 + out4
        return out


class FCN(nn.Module):

    def __init__(self, n_classes):
        super(FCN, self).__init__()
        self.num_classes = n_classes
        self.conv1 = nn.Conv2d(3, 96, kernel_size=3, stride=2, padding=1)
        self.relu1 = nn.ELU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fire1_1 = Fire(96, 16, 64)
        self.fire1_2 = Fire(128, 16, 64)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fire2_1 = Fire(128, 32, 128)
        self.fire2_2 = Fire(256, 32, 128)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fire3_1 = Fire(256, 64, 256)
        self.fire3_2 = Fire(512, 64, 256)
        self.fire3_3 = Fire(512, 64, 256)
        self.parallel = ParallelDilatedConv(512, 512)
        self.deconv1 = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1,
            output_padding=1)
        self.relu2 = nn.ELU(inplace=True)
        self.deconv2 = nn.ConvTranspose2d(512, 128, 3, stride=2, padding=1,
            output_padding=1)
        self.relu3 = nn.ELU(inplace=True)
        self.deconv3 = nn.ConvTranspose2d(256, 96, 3, stride=2, padding=1,
            output_padding=1)
        self.relu4 = nn.ELU(inplace=True)
        self.deconv4 = nn.ConvTranspose2d(192, self.num_classes, 3, stride=
            2, padding=1, output_padding=1)
        self.conv3_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv2_1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv1_1 = nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1)
        self.relu1_1 = nn.ELU(inplace=True)
        self.relu1_2 = nn.ELU(inplace=True)
        self.relu2_1 = nn.ELU(inplace=True)
        self.relu2_2 = nn.ELU(inplace=True)
        self.relu3_1 = nn.ELU(inplace=True)
        self.relu3_2 = nn.ELU(inplace=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x_1 = self.relu1(x)
        x = self.maxpool1(x_1)
        x = self.fire1_1(x)
        x_2 = self.fire1_2(x)
        x = self.maxpool2(x_2)
        x = self.fire2_1(x)
        x_3 = self.fire2_2(x)
        x = self.maxpool3(x_3)
        x = self.fire3_1(x)
        x = self.fire3_2(x)
        x = self.fire3_3(x)
        x = self.parallel(x)
        y_3 = self.deconv1(x)
        y_3 = self.relu2(y_3)
        x_3 = self.conv3_1(x_3)
        x_3 = self.relu3_1(x_3)
        x_3 = F.upsample_bilinear(x_3, y_3.size()[2:])
        x = torch.cat([x_3, y_3], 1)
        x = self.conv3_2(x)
        x = self.relu3_2(x)
        y_2 = self.deconv2(x)
        y_2 = self.relu3(y_2)
        x_2 = self.conv2_1(x_2)
        x_2 = self.relu2_1(x_2)
        y_2 = F.upsample_bilinear(y_2, x_2.size()[2:])
        x = torch.cat([x_2, y_2], 1)
        x = self.conv2_2(x)
        x = self.relu2_2(x)
        y_1 = self.deconv3(x)
        y_1 = self.relu4(y_1)
        x_1 = self.conv1_1(x_1)
        x_1 = self.relu1_1(x_1)
        x = torch.cat([x_1, y_1], 1)
        x = self.conv1_2(x)
        x = self.relu1_2(x)
        x = self.deconv4(x)
        return x


class unet(nn.Module):

    def __init__(self, n_classes=21, pretrained=False):
        super(unet, self).__init__()
        self.down1 = unetDown(in_channels=3, out_channels=64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down2 = unetDown(in_channels=64, out_channels=128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down3 = unetDown(in_channels=128, out_channels=256)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down4 = unetDown(in_channels=256, out_channels=512)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.center = unetDown(in_channels=512, out_channels=1024)
        self.up4 = unetUp(in_channels=1024, out_channels=512)
        self.up3 = unetUp(in_channels=512, out_channels=256)
        self.up2 = unetUp(in_channels=256, out_channels=128)
        self.up1 = unetUp(in_channels=128, out_channels=64)
        self.classifier = nn.Conv2d(in_channels=64, out_channels=n_classes,
            kernel_size=1)

    def forward(self, x):
        out_size = x.size()[2:]
        down1_x = self.down1(x)
        maxpool1_x = self.maxpool1(down1_x)
        down2_x = self.down2(maxpool1_x)
        maxpool2_x = self.maxpool2(down2_x)
        down3_x = self.down3(maxpool2_x)
        maxpool3_x = self.maxpool3(down3_x)
        down4_x = self.down4(maxpool3_x)
        maxpool4_x = self.maxpool1(down4_x)
        center_x = self.center(maxpool4_x)
        up4_x = self.up4(center_x, down4_x)
        up3_x = self.up3(up4_x, down3_x)
        up2_x = self.up2(up3_x, down2_x)
        up1_x = self.up1(up2_x, down1_x)
        x = self.classifier(up1_x)
        x = F.upsample_bilinear(x, out_size)
        return x


class conv2DBatchNorm(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
        padding, bias=True):
        super(conv2DBatchNorm, self).__init__()
        self.cb_seq = nn.Sequential(nn.Conv2d(int(in_channels), int(
            out_channels), kernel_size=kernel_size, padding=padding, stride
            =stride, bias=bias), nn.BatchNorm2d(int(out_channels)))

    def forward(self, inputs):
        outputs = self.cb_seq(inputs)
        return outputs


class conv2DBatchNormRelu(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
        padding, bias=True, dilation=1):
        super(conv2DBatchNormRelu, self).__init__()
        self.cbr_seq = nn.Sequential(nn.Conv2d(in_channels=in_channels,
            out_channels=out_channels, kernel_size=kernel_size, stride=
            stride, padding=padding, bias=bias, dilation=dilation), nn.
            BatchNorm2d(num_features=out_channels), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.cbr_seq(x)
        return x


class unetDown(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(unetDown, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=in_channels,
            out_channels=out_channels, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(num_features=out_channels), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=out_channels,
            out_channels=out_channels, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(num_features=out_channels), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class unetUp(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(unetUp, self).__init__()
        self.upConv = nn.ConvTranspose2d(in_channels=in_channels,
            out_channels=out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=in_channels,
            out_channels=out_channels, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=out_channels,
            out_channels=out_channels, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True))

    def forward(self, x_cur, x_prev):
        x = self.upConv(x_cur)
        x = torch.cat([F.upsample_bilinear(x_prev, size=x.size()[2:]), x], 1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class segnetDown2(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(segnetDown2, self).__init__()
        self.conv1 = conv2DBatchNormRelu(in_channels=in_channels,
            out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = conv2DBatchNormRelu(in_channels=out_channels,
            out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2,
            return_indices=True)
        pass

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        unpool_shape = x.size()
        x, pool_indices = self.max_pool(x)
        return x, pool_indices, unpool_shape


class segnetDown3(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(segnetDown3, self).__init__()
        self.conv1 = conv2DBatchNormRelu(in_channels=in_channels,
            out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = conv2DBatchNormRelu(in_channels=out_channels,
            out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = conv2DBatchNormRelu(in_channels=out_channels,
            out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2,
            return_indices=True)
        pass

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        unpool_shape = x.size()
        x, pool_indices = self.max_pool(x)
        return x, pool_indices, unpool_shape


class segnetDown4(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(segnetDown4, self).__init__()
        self.conv1 = conv2DBatchNormRelu(in_channels=in_channels,
            out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = conv2DBatchNormRelu(in_channels=out_channels,
            out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = conv2DBatchNormRelu(in_channels=out_channels,
            out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv4 = conv2DBatchNormRelu(in_channels=out_channels,
            out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2,
            return_indices=True)
        pass

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        unpool_shape = x.size()
        x, pool_indices = self.max_pool(x)
        return x, pool_indices, unpool_shape


class segnetUp2(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(segnetUp2, self).__init__()
        self.max_unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.conv1 = conv2DBatchNormRelu(in_channels=in_channels,
            out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = conv2DBatchNormRelu(in_channels=in_channels,
            out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        pass

    def forward(self, x, pool_indices, unpool_shape):
        x = self.max_unpool(x, indices=pool_indices, output_size=unpool_shape)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class segnetUNetDown2(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(segnetUNetDown2, self).__init__()
        self.conv1 = conv2DBatchNormRelu(in_channels=in_channels,
            out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = conv2DBatchNormRelu(in_channels=out_channels,
            out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2,
            return_indices=True)
        pass

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        unpool_shape = x.size()
        x_pool, pool_indices = self.max_pool(x)
        return x_pool, pool_indices, unpool_shape, x


class segnetUNetDown3(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(segnetUNetDown3, self).__init__()
        self.conv1 = conv2DBatchNormRelu(in_channels=in_channels,
            out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = conv2DBatchNormRelu(in_channels=out_channels,
            out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = conv2DBatchNormRelu(in_channels=out_channels,
            out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2,
            return_indices=True)
        pass

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        unpool_shape = x.size()
        x_pool, pool_indices = self.max_pool(x)
        return x_pool, pool_indices, unpool_shape, x


class segnetUp3(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(segnetUp3, self).__init__()
        self.max_unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.conv1 = conv2DBatchNormRelu(in_channels=in_channels,
            out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = conv2DBatchNormRelu(in_channels=in_channels,
            out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = conv2DBatchNormRelu(in_channels=in_channels,
            out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        pass

    def forward(self, x, pool_indices, unpool_shape):
        x = self.max_unpool(x, indices=pool_indices, output_size=unpool_shape)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class segnetUp4(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(segnetUp4, self).__init__()
        self.max_unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.conv1 = conv2DBatchNormRelu(in_channels=in_channels,
            out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = conv2DBatchNormRelu(in_channels=in_channels,
            out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = conv2DBatchNormRelu(in_channels=in_channels,
            out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.conv4 = conv2DBatchNormRelu(in_channels=in_channels,
            out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        pass

    def forward(self, x, pool_indices, unpool_shape):
        x = self.max_unpool(x, indices=pool_indices, output_size=unpool_shape)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x


class segnetUNetUp2(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(segnetUNetUp2, self).__init__()
        self.max_unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.conv1 = conv2DBatchNormRelu(in_channels=in_channels * 2,
            out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = conv2DBatchNormRelu(in_channels=in_channels,
            out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        pass

    def forward(self, x, pool_indices, unpool_shape, concat_net):
        x = self.max_unpool(x, indices=pool_indices, output_size=unpool_shape)
        x = torch.cat([concat_net, x], 1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class segnetUNetUp3(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(segnetUNetUp3, self).__init__()
        self.max_unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.conv1 = conv2DBatchNormRelu(in_channels=in_channels * 2,
            out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = conv2DBatchNormRelu(in_channels=in_channels,
            out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = conv2DBatchNormRelu(in_channels=in_channels,
            out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        pass

    def forward(self, x, pool_indices, unpool_shape, concat_net):
        x = self.max_unpool(x, indices=pool_indices, output_size=unpool_shape)
        x = torch.cat([concat_net, x], 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class residualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(residualBlock, self).__init__()
        self.convbnrelu1 = conv2DBatchNormRelu(in_channels, out_channels, 3,
            stride, 1, bias=False)
        self.convbn2 = conv2DBatchNorm(out_channels, out_channels, 3, 1, 1,
            bias=False)
        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.convbnrelu1(x)
        out = self.convbn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class linknetUp(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(linknetUp, self).__init__()
        self.convbnrelu1 = conv2DBatchNormRelu(in_channels, out_channels / 
            2, kernel_size=1, stride=1, padding=1)
        self.deconvbnrelu2 = deconv2DBatchNormRelu(out_channels / 2, 
            out_channels / 2, kernel_size=3, stride=2, padding=0)
        self.convbnrelu3 = conv2DBatchNormRelu(out_channels / 2,
            out_channels, kernel_size=1, stride=1, padding=1)

    def forward(self, x):
        x = self.convbnrelu1(x)
        x = self.deconvbnrelu2(x)
        x = self.convbnrelu3(x)
        return x


class deconv2DBatchNormRelu(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
        padding, bias=True):
        super(deconv2DBatchNormRelu, self).__init__()
        self.dcbr_unit = nn.Sequential(nn.ConvTranspose2d(int(in_channels),
            int(out_channels), kernel_size=kernel_size, padding=padding,
            stride=stride, bias=bias), nn.BatchNorm2d(int(out_channels)),
            nn.ReLU(inplace=True))

    def forward(self, inputs):
        outputs = self.dcbr_unit(inputs)
        return outputs


class bottleNeckIdentifyPSP(nn.Module):

    def __init__(self, in_channels, mid_channels, stride, dilation=1):
        super(bottleNeckIdentifyPSP, self).__init__()
        self.cbr1 = conv2DBatchNormRelu(in_channels, mid_channels, 1, 1, 0,
            bias=False)
        if dilation > 1:
            self.cbr2 = conv2DBatchNormRelu(mid_channels, mid_channels, 3, 
                1, padding=dilation, bias=False, dilation=dilation)
        else:
            self.cbr2 = conv2DBatchNormRelu(mid_channels, mid_channels, 3,
                stride=1, padding=1, bias=False, dilation=1)
        self.cb3 = conv2DBatchNorm(mid_channels, in_channels, 1, 1, 0)

    def forward(self, x):
        residual = x
        x = self.cb3(self.cbr2(self.cbr1(x)))
        return F.relu(x + residual, inplace=True)


class bottleNeckPSP(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels, stride,
        dilation=1):
        super(bottleNeckPSP, self).__init__()
        self.cbr1 = conv2DBatchNormRelu(in_channels, mid_channels, 1, 1, 0,
            bias=False)
        if dilation > 1:
            self.cbr2 = conv2DBatchNormRelu(mid_channels, mid_channels, 3, 
                1, padding=dilation, bias=False, dilation=dilation)
        else:
            self.cbr2 = conv2DBatchNormRelu(mid_channels, mid_channels, 3,
                stride=stride, padding=1, bias=False, dilation=1)
        self.cb3 = conv2DBatchNorm(mid_channels, out_channels, 1, 1, 0)
        self.cb4 = conv2DBatchNorm(in_channels, out_channels, 1, stride, 0)

    def forward(self, x):
        conv = self.cb3(self.cbr2(self.cbr1(x)))
        residual = self.cb4(x)
        return F.relu(conv + residual, inplace=True)


class residualBlockPSP(nn.Module):

    def __init__(self, n_blocks, in_channels, mid_channels, out_channels,
        stride, dilation=1):
        super(residualBlockPSP, self).__init__()
        if dilation > 1:
            stride = 1
        layers = [bottleNeckPSP(in_channels, mid_channels, out_channels,
            stride, dilation)]
        for i in range(n_blocks):
            layers.append(bottleNeckIdentifyPSP(out_channels, mid_channels,
                stride, dilation))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class pyramidPooling(nn.Module):
    """
    金字塔池化模块，主要是将特征图通过不同的pool构造池化输出，最后的输出是和输入相同分辨率大小的concat的特征图
    """

    def __init__(self, in_channels, pool_sizes):
        super(pyramidPooling, self).__init__()
        self.paths = []
        for i in range(len(pool_sizes)):
            self.paths.append(conv2DBatchNormRelu(in_channels, int(
                in_channels / len(pool_sizes)), 1, 1, 0, bias=False))
        self.path_module_list = nn.ModuleList(self.paths)
        self.pool_sizes = pool_sizes

    def forward(self, x):
        output_slices = [x]
        h, w = x.view()[2:]
        for module, pool_size in zip(self.path_module_list, self.pool_sizes):
            out = F.avg_pool2d(x, pool_size, stride=1, padding=0)
            out = module(out)
            out = F.upsample(out, size=(h, w), mode='bilinear')
            output_slices.append(out)
        return torch.cat(output_slices, dim=1)


class AlignedResInception(nn.Module):
    """
    Aligned残差Inception结构
    """

    def __init__(self, in_planes, stride=1):
        super(AlignedResInception, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.b1 = nn.Sequential(nn.Conv2d(in_planes, in_planes // 4,
            kernel_size=1, stride=1), nn.BatchNorm2d(in_planes // 4), nn.
            ReLU(True), nn.Conv2d(in_planes // 4, in_planes // 4,
            kernel_size=3, stride=1, padding=1))
        self.b2 = nn.Sequential(nn.Conv2d(in_planes, in_planes // 8,
            kernel_size=1, stride=1), nn.BatchNorm2d(in_planes // 8), nn.
            ReLU(True), nn.Conv2d(in_planes // 8, in_planes // 8,
            kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(in_planes //
            8), nn.ReLU(True), nn.Conv2d(in_planes // 8, in_planes // 8,
            kernel_size=3, stride=1, padding=1))
        self.b3 = nn.Sequential(nn.BatchNorm2d(in_planes // 8 * 3), nn.ReLU
            (True))
        self.b4 = nn.Sequential(nn.Conv2d(in_planes // 8 * 3, in_planes,
            kernel_size=1, stride=stride), nn.BatchNorm2d(in_planes), nn.
            ReLU(True))
        self.downsample = None
        if stride > 1:
            self.downsample = nn.Sequential(nn.Conv2d(in_planes, in_planes,
                kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(
                in_planes))

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = torch.cat([y1, y2], 1)
        y3 = self.b3(y3)
        out = self.b4(y3)
        if self.downsample is not None:
            out = out + self.downsample(x)
        else:
            out = out + x
        out = self.relu(out)
        return out


class Inception(nn.Module):

    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5,
        pool_planes):
        super(Inception, self).__init__()
        self.b1 = nn.Sequential(nn.Conv2d(in_planes, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1), nn.ReLU(True))
        self.b2 = nn.Sequential(nn.Conv2d(in_planes, n3x3red, kernel_size=1
            ), nn.BatchNorm2d(n3x3red), nn.ReLU(True), nn.Conv2d(n3x3red,
            n3x3, kernel_size=3, padding=1), nn.BatchNorm2d(n3x3), nn.ReLU(
            True))
        self.b3 = nn.Sequential(nn.Conv2d(in_planes, n5x5red, kernel_size=1
            ), nn.BatchNorm2d(n5x5red), nn.ReLU(True), nn.Conv2d(n5x5red,
            n5x5, kernel_size=3, padding=1), nn.BatchNorm2d(n5x5), nn.ReLU(
            True), nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1), nn.
            BatchNorm2d(n5x5), nn.ReLU(True))
        self.b4 = nn.Sequential(nn.MaxPool2d(3, stride=1, padding=1), nn.
            Conv2d(in_planes, pool_planes, kernel_size=1), nn.BatchNorm2d(
            pool_planes), nn.ReLU(True))

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1, y2, y3, y4], 1)


class ResInception(nn.Module):

    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5,
        pool_planes, stride=1):
        super(ResInception, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.b1 = nn.Sequential(nn.Conv2d(in_planes, n1x1, kernel_size=1,
            stride=stride), nn.BatchNorm2d(n1x1), nn.ReLU(True))
        self.b2 = nn.Sequential(nn.Conv2d(in_planes, n3x3red, kernel_size=1,
            stride=stride), nn.BatchNorm2d(n3x3red), nn.ReLU(True), nn.
            Conv2d(n3x3red, n3x3, kernel_size=3, padding=1), nn.BatchNorm2d
            (n3x3), nn.ReLU(True))
        self.b3 = nn.Sequential(nn.Conv2d(in_planes, n5x5red, kernel_size=1,
            stride=stride), nn.BatchNorm2d(n5x5red), nn.ReLU(True), nn.
            Conv2d(n5x5red, n5x5, kernel_size=3, padding=1), nn.BatchNorm2d
            (n5x5), nn.ReLU(True), nn.Conv2d(n5x5, n5x5, kernel_size=3,
            padding=1), nn.BatchNorm2d(n5x5), nn.ReLU(True))
        self.b4 = nn.Sequential(nn.MaxPool2d(3, stride=stride, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1), nn.
            BatchNorm2d(pool_planes), nn.ReLU(True))
        self.downsample = None
        if stride > 1:
            self.downsample = nn.Sequential(nn.Conv2d(in_planes, in_planes,
                kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(
                in_planes))

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        out = torch.cat([y1, y2, y3, y4], 1)
        if self.downsample is not None:
            out = out + self.downsample(x)
        else:
            out = out + x
        out = self.relu(out)
        return out


class CascadeResInception(nn.Module):

    def __init__(self):
        super(CascadeResInception, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.res1 = ResInception(512, 128, 128, 256, 24, 64, 64, stride=2)
        self.res2 = ResInception(512, 128, 128, 256, 24, 64, 64, stride=7)
        self.res3 = ResInception(512, 128, 128, 256, 24, 64, 64, stride=14)

    def forward(self, x):
        y1 = self.res1(x)
        y2 = self.res2(x)
        y3 = self.res3(x)
        y1 = F.upsample_bilinear(y1, x.size()[2:])
        y2 = F.upsample_bilinear(y2, x.size()[2:])
        y3 = F.upsample_bilinear(y3, x.size()[2:])
        out = x + y1 + y2 + y3
        out = self.relu(out)
        return out


class CascadeAlignedResInception(nn.Module):

    def __init__(self, in_planes):
        super(CascadeAlignedResInception, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.res1 = AlignedResInception(in_planes=in_planes, stride=2)
        self.res2 = AlignedResInception(in_planes=in_planes, stride=7)
        self.res3 = AlignedResInception(in_planes=in_planes, stride=14)

    def forward(self, x):
        y1 = self.res1(x)
        y2 = self.res2(x)
        y3 = self.res3(x)
        y1 = F.upsample_bilinear(y1, x.size()[2:])
        y2 = F.upsample_bilinear(y2, x.size()[2:])
        y3 = F.upsample_bilinear(y3, x.size()[2:])
        out = x + y1 + y2 + y3
        out = self.relu(out)
        return out


class ASPP_Classifier_Module(nn.Module):

    def __init__(self, dilation_series, padding_series, n_classes,
        in_channels=2048):
        super(ASPP_Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(nn.Conv2d(in_channels, n_classes,
                kernel_size=3, stride=1, padding=padding, dilation=dilation,
                bias=True))
        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
        return out


class IBN(nn.Module):

    def __init__(self, planes):
        super(IBN, self).__init__()
        half1 = int(planes / 2)
        self.half = half1
        half2 = planes - half1
        self.IN = nn.InstanceNorm2d(half1, affine=True)
        self.BN = nn.BatchNorm2d(half2)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out


class conv2DGroupNormRelu(nn.Module):

    def __init__(self, in_channels, n_filters, kernel_size, stride, padding,
        bias=True, dilation=1, n_groups=16):
        super(conv2DGroupNormRelu, self).__init__()
        conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=
            kernel_size, padding=padding, stride=stride, bias=bias,
            dilation=dilation)
        self.cgr_unit = nn.Sequential(conv_mod, nn.GroupNorm(n_groups, int(
            n_filters)), nn.ReLU(inplace=True))

    def forward(self, inputs):
        outputs = self.cgr_unit(inputs)
        return outputs


class conv2DGroupNorm(nn.Module):

    def __init__(self, in_channels, n_filters, kernel_size, stride, padding,
        bias=True, dilation=1, n_groups=16):
        super(conv2DGroupNorm, self).__init__()
        conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=
            kernel_size, padding=padding, stride=stride, bias=bias,
            dilation=dilation)
        self.cgr_unit = nn.Sequential(conv_mod, nn.GroupNorm(n_groups, int(
            n_filters)))

    def forward(self, inputs):
        outputs = self.cgr_unit(inputs)
        return outputs


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
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
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size
            =1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
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


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
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
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
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


class IBN(nn.Module):

    def __init__(self, planes):
        super(IBN, self).__init__()
        half1 = int(planes / 2)
        self.half = half1
        half2 = planes - half1
        self.IN = nn.InstanceNorm2d(half1, affine=True)
        self.BN = nn.BatchNorm2d(half2)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, ibn=False, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        if ibn:
            self.bn1 = IBN(planes)
        else:
            self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size
            =1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
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


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        scale = 64
        self.inplanes = scale
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, scale, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = nn.BatchNorm2d(scale)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, scale, layers[0])
        self.layer2 = self._make_layer(block, scale * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, scale * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, scale * 8, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(scale * 8 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
        layers = []
        ibn = True
        if planes == 512:
            ibn = False
        layers.append(block(self.inplanes, planes, ibn, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, ibn))
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
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
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

    def __init__(self, inplanes, planes, stride=1, downsample=None, IN=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size
            =1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.IN = None
        if IN:
            self.IN = nn.InstanceNorm2d(planes * 4, affine=True)
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
        if self.IN is not None:
            out = self.IN(out)
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        scale = 64
        self.inplanes = scale
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, scale, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = nn.InstanceNorm2d(scale, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, scale, layers[0], stride=1,
            IN=True)
        self.layer2 = self._make_layer(block, scale * 2, layers[1], stride=
            2, IN=True)
        self.layer3 = self._make_layer(block, scale * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, scale * 8, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(scale * 8 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, IN=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks - 1):
            layers.append(block(self.inplanes, planes))
        layers.append(block(self.inplanes, planes, IN=IN))
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
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class yoloLoss(nn.Module):

    def __init__(self, S, B, C, l_coord, l_noobj, use_gpu):
        super(yoloLoss, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.l_coord = l_coord
        self.l_noobj = l_noobj
        self.use_gpu = use_gpu
        self.out_tensor_shape = self.B * 5 + self.C

    def compute_iou(self, box1, box2):
        """Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
        Args:
          box1: (tensor) bounding boxes, sized [N,4].
          box2: (tensor) bounding boxes, sized [M,4].
        Return:
          (tensor) iou, sized [N,M].
        """
        N = box1.size(0)
        M = box2.size(0)
        lt = torch.max(box1[:, :2].unsqueeze(1).expand(N, M, 2), box2[:, :2
            ].unsqueeze(0).expand(N, M, 2))
        rb = torch.min(box1[:, 2:].unsqueeze(1).expand(N, M, 2), box2[:, 2:
            ].unsqueeze(0).expand(N, M, 2))
        wh = rb - lt
        wh[wh < 0] = 0
        inter = wh[:, :, (0)] * wh[:, :, (1)]
        area1 = (box1[:, (2)] - box1[:, (0)]) * (box1[:, (3)] - box1[:, (1)])
        area2 = (box2[:, (2)] - box2[:, (0)]) * (box2[:, (3)] - box2[:, (1)])
        area1 = area1.unsqueeze(1).expand_as(inter)
        area2 = area2.unsqueeze(0).expand_as(inter)
        iou = inter / (area1 + area2 - inter)
        return iou

    def forward(self, pred_tensor, target_tensor):
        """
        pred_tensor: (tensor) size(batchsize,S,S,Bx5+20=30) [x,y,w,h,c]
        target_tensor: (tensor) size(batchsize,S,S,30)
        """
        N = pred_tensor.size()[0]
        coo_mask = target_tensor[:, :, :, (4)] > 0
        noo_mask = target_tensor[:, :, :, (4)] == 0
        coo_mask = coo_mask.unsqueeze(-1).expand_as(target_tensor)
        noo_mask = noo_mask.unsqueeze(-1).expand_as(target_tensor)
        coo_pred = pred_tensor[coo_mask].view(-1, self.out_tensor_shape)
        box_pred = coo_pred[:, :10].contiguous().view(-1, 5)
        class_pred = coo_pred[:, 10:]
        coo_target = target_tensor[coo_mask].view(-1, self.out_tensor_shape)
        box_target = coo_target[:, :10].contiguous().view(-1, 5)
        class_target = coo_target[:, 10:]
        noo_pred = pred_tensor[noo_mask].view(-1, self.out_tensor_shape)
        noo_target = target_tensor[noo_mask].view(-1, self.out_tensor_shape)
        if self.use_gpu:
            noo_pred_mask = torch.cuda.ByteTensor(noo_pred.size())
        else:
            noo_pred_mask = torch.ByteTensor(noo_pred.size())
        noo_pred_mask.zero_()
        noo_pred_mask[:, (4)] = 1
        noo_pred_mask[:, (9)] = 1
        noo_pred_c = noo_pred[noo_pred_mask]
        noo_target_c = noo_target[noo_pred_mask]
        nooobj_loss = F.mse_loss(noo_pred_c, noo_target_c, size_average=False)
        if self.use_gpu:
            coo_response_mask = torch.cuda.ByteTensor(box_target.size())
        else:
            coo_response_mask = torch.ByteTensor(box_target.size())
        coo_response_mask.zero_()
        if self.use_gpu:
            coo_not_response_mask = torch.cuda.ByteTensor(box_target.size())
        else:
            coo_not_response_mask = torch.ByteTensor(box_target.size())
        coo_not_response_mask.zero_()
        if self.use_gpu:
            box_target_iou = torch.zeros(box_target.size())
        else:
            box_target_iou = torch.zeros(box_target.size())
        for i in range(0, box_target.size()[0], 2):
            box1 = box_pred[i:i + 2]
            box1_xyxy = Variable(torch.FloatTensor(box1.size()))
            box1_xyxy[:, :2] = box1[:, :2] / 14.0 - 0.5 * box1[:, 2:4]
            box1_xyxy[:, 2:4] = box1[:, :2] / 14.0 + 0.5 * box1[:, 2:4]
            box2 = box_target[i].view(-1, 5)
            box2_xyxy = Variable(torch.FloatTensor(box2.size()))
            box2_xyxy[:, :2] = box2[:, :2] / 14.0 - 0.5 * box2[:, 2:4]
            box2_xyxy[:, 2:4] = box2[:, :2] / 14.0 + 0.5 * box2[:, 2:4]
            iou = self.compute_iou(box1_xyxy[:, :4], box2_xyxy[:, :4])
            max_iou, max_index = iou.max(0)
            if self.use_gpu:
                max_index = max_index.data
            else:
                max_index = max_index.data
            coo_response_mask[i + max_index] = 1
            coo_not_response_mask[i + 1 - max_index] = 1
            if self.use_gpu:
                box_target_iou[i + max_index, torch.LongTensor([4])
                    ] = max_iou.data
            else:
                box_target_iou[i + max_index, torch.LongTensor([4])
                    ] = max_iou.data
        if self.use_gpu:
            box_target_iou = Variable(box_target_iou)
        else:
            box_target_iou = Variable(box_target_iou)
        box_pred_response = box_pred[coo_response_mask].view(-1, 5)
        box_target_response_iou = box_target_iou[coo_response_mask].view(-1, 5)
        box_target_response = box_target[coo_response_mask].view(-1, 5)
        contain_loss = F.mse_loss(box_pred_response[:, (4)],
            box_target_response_iou[:, (4)], size_average=False)
        loc_loss = F.mse_loss(box_pred_response[:, :2], box_target_response
            [:, :2], size_average=False) + F.mse_loss(torch.sqrt(
            box_pred_response[:, 2:4]), torch.sqrt(box_target_response[:, 2
            :4]), size_average=False)
        box_pred_not_response = box_pred[coo_not_response_mask].view(-1, 5)
        box_target_not_response = box_target[coo_not_response_mask].view(-1, 5)
        box_target_not_response[:, (4)] = 0
        not_contain_loss = F.mse_loss(box_pred_not_response[:, (4)],
            box_target_not_response[:, (4)], size_average=False)
        class_loss = F.mse_loss(class_pred, class_target, size_average=False)
        return (self.l_coord * loc_loss + 2 * contain_loss +
            not_contain_loss + self.l_noobj * nooobj_loss + class_loss) / N

