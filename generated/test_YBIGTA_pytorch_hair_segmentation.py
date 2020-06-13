import sys
_module = sys.modules[__name__]
del sys
data = _module
figaro = _module
lfw = _module
demo = _module
evaluate = _module
main = _module
networks = _module
deeplab_v3_plus = _module
mobile_hair = _module
pspnet = _module
utils = _module
joint_transforms = _module
metrics = _module
trainer_verbose = _module

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


import copy


import logging


import torch


import math


import torch.nn as nn


import torch.nn.functional as F


import torch.utils.model_zoo as model_zoo


from torch import nn


from torch.nn import functional as F


import numpy as np


from torch.nn.init import xavier_normal_


from collections import OrderedDict


def fixed_padding(inputs, kernel_size, rate):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs


class SeparableConv2d(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=
        1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, 0,
            dilation, groups=inplanes, bias=bias)
        self.bn = nn.BatchNorm2d(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = fixed_padding(x, self.conv1.kernel_size[0], dilation=self.conv1
            .dilation[0])
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):

    def __init__(self, inplanes, planes, reps, stride=1, dilation=1,
        start_with_relu=True, grow_first=True, is_last=False):
        super(Block, self).__init__()
        if planes != inplanes or stride != 1:
            self.skip = nn.Conv2d(inplanes, planes, 1, stride=stride, bias=
                False)
            self.skipbn = nn.BatchNorm2d(planes)
        else:
            self.skip = None
        self.relu = nn.ReLU(inplace=True)
        rep = []
        filters = inplanes
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(inplanes, planes, 3, 1, dilation))
            rep.append(nn.BatchNorm2d(planes))
            filters = planes
        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, 1, dilation))
            rep.append(nn.BatchNorm2d(filters))
        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(inplanes, planes, 3, 1, dilation))
            rep.append(nn.BatchNorm2d(planes))
        if stride != 1:
            rep.append(self.relu)
            rep.append(SeparableConv2d(planes, planes, 3, 2))
            rep.append(nn.BatchNorm2d(planes))
        if stride == 1 and is_last:
            rep.append(self.relu)
            rep.append(SeparableConv2d(planes, planes, 3, 1))
            rep.append(nn.BatchNorm2d(planes))
        if not start_with_relu:
            rep = rep[1:]
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)
        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp
        x = x + skip
        return x


class ModifiedAlignedXception(nn.Module):
    """
    Modified Alighed Xception
    """

    def __init__(self, output_stride, pretrained=True):
        super(ModifiedAlignedXception, self).__init__()
        if output_stride == 16:
            entry_block3_stride = 2
            middle_block_dilation = 1
            exit_block_dilations = 1, 2
        elif output_stride == 8:
            entry_block3_stride = 1
            middle_block_dilation = 2
            exit_block_dilations = 2, 4
        else:
            raise NotImplementedError
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.block1 = Block(64, 128, reps=2, stride=2, start_with_relu=False)
        self.block2 = Block(128, 256, reps=2, stride=2, start_with_relu=
            False, grow_first=True)
        self.block3 = Block(256, 728, reps=2, stride=entry_block3_stride,
            start_with_relu=True, grow_first=True, is_last=True)
        self.block4 = Block(728, 728, reps=3, stride=1, dilation=
            middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block5 = Block(728, 728, reps=3, stride=1, dilation=
            middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block6 = Block(728, 728, reps=3, stride=1, dilation=
            middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block7 = Block(728, 728, reps=3, stride=1, dilation=
            middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block8 = Block(728, 728, reps=3, stride=1, dilation=
            middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block9 = Block(728, 728, reps=3, stride=1, dilation=
            middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block10 = Block(728, 728, reps=3, stride=1, dilation=
            middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block11 = Block(728, 728, reps=3, stride=1, dilation=
            middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block12 = Block(728, 728, reps=3, stride=1, dilation=
            middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block13 = Block(728, 728, reps=3, stride=1, dilation=
            middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block14 = Block(728, 728, reps=3, stride=1, dilation=
            middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block15 = Block(728, 728, reps=3, stride=1, dilation=
            middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block16 = Block(728, 728, reps=3, stride=1, dilation=
            middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block17 = Block(728, 728, reps=3, stride=1, dilation=
            middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block18 = Block(728, 728, reps=3, stride=1, dilation=
            middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block19 = Block(728, 728, reps=3, stride=1, dilation=
            middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block20 = Block(728, 1024, reps=2, stride=1, dilation=
            exit_block_dilations[0], start_with_relu=True, grow_first=False,
            is_last=True)
        self.conv3 = SeparableConv2d(1024, 1536, 3, stride=1, dilation=
            exit_block_dilations[1])
        self.bn3 = nn.BatchNorm2d(1536)
        self.conv4 = SeparableConv2d(1536, 1536, 3, stride=1, dilation=
            exit_block_dilations[1])
        self.bn4 = nn.BatchNorm2d(1536)
        self.conv5 = SeparableConv2d(1536, 2048, 3, stride=1, dilation=
            exit_block_dilations[1])
        self.bn5 = nn.BatchNorm2d(2048)
        if pretrained:
            self._load_pretrained_model()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.block1(x)
        x = self.relu(x)
        low_level_feat = x
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        x = self.block17(x)
        x = self.block18(x)
        x = self.block19(x)
        x = self.block20(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        return x, low_level_feat

    def _load_pretrained_model(self):
        pretrain_dict = model_zoo.load_url(
            'http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth'
            )
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in model_dict:
                if 'pointwise' in k:
                    v = v.unsqueeze(-1).unsqueeze(-1)
                if k.startswith('block11'):
                    model_dict[k] = v
                    model_dict[k.replace('block11', 'block12')] = v
                    model_dict[k.replace('block11', 'block13')] = v
                    model_dict[k.replace('block11', 'block14')] = v
                    model_dict[k.replace('block11', 'block15')] = v
                    model_dict[k.replace('block11', 'block16')] = v
                    model_dict[k.replace('block11', 'block17')] = v
                    model_dict[k.replace('block11', 'block18')] = v
                    model_dict[k.replace('block11', 'block19')] = v
                elif k.startswith('block12'):
                    model_dict[k.replace('block12', 'block20')] = v
                elif k.startswith('bn3'):
                    model_dict[k] = v
                    model_dict[k.replace('bn3', 'bn4')] = v
                elif k.startswith('conv4'):
                    model_dict[k.replace('conv4', 'conv5')] = v
                elif k.startswith('bn4'):
                    model_dict[k.replace('bn4', 'bn5')] = v
                else:
                    model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)


class _ASPPModule(nn.Module):

    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=
            kernel_size, stride=1, padding=padding, dilation=dilation, bias
            =False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)
        return self.relu(x)


class ASPP(nn.Module):

    def __init__(self, output_stride):
        super(ASPP, self).__init__()
        inplanes = 2048
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError
        self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=
            dilations[0])
        self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1],
            dilation=dilations[1])
        self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2],
            dilation=dilations[2])
        self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3],
            dilation=dilations[3])
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, 256, 1, stride=1, bias=False), nn.
            BatchNorm2d(256), nn.ReLU())
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear',
            align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return self.dropout(x)


class Decoder(nn.Module):

    def __init__(self, return_with_logits, low_level_inplanes=128):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU()
        layers = [nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1,
            bias=False), nn.BatchNorm2d(256), nn.ReLU(), nn.Dropout(0.5),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=
            False), nn.BatchNorm2d(256), nn.ReLU(), nn.Dropout(0.1), nn.
            Conv2d(256, 1, kernel_size=1, stride=1)]
        if not return_with_logits:
            layers.append(nn.Sigmoid())
        self.last_conv = nn.Sequential(*layers)

    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)
        x = F.interpolate(x, size=low_level_feat.size()[2:], mode=
            'bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(x)
        return x


class DeepLab(nn.Module):

    def __init__(self, return_with_logits=True, output_stride=16):
        super(DeepLab, self).__init__()
        self.backbone = ModifiedAlignedXception(output_stride)
        self.aspp = ASPP(output_stride)
        self.decoder = Decoder(return_with_logits)

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear',
            align_corners=True)
        return x


class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
        padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size,
            stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1,
            bias=bias)

    def forward(self, x):
        x = fixed_padding(x, self.conv1.kernel_size[0], rate=self.conv1.
            dilation[0])
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class GreenBlock(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(GreenBlock, self).__init__()
        self.dconv = nn.Sequential(SeparableConv2d(in_channel), nn.
            BatchNorm2d(in_channel), nn.ReLU())
        self.conv = nn.Sequential(nn.Conv2d(in_channel, out_channel,
            kernel_size=1), nn.BatchNorm2d(out_channel), nn.ReLU())

    def forward(self, input):
        x = self.dconv(input)
        x = self.conv(x)
        return x


class YellowBlock(nn.Module):

    def __init__(self):
        super(YellowBlock, self).__init__()

    def forward(self, input):
        return F.interpolate(input, scale_factor=2)


class OrangeBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
        padding=0, dilation=1, bias=False):
        super(OrangeBlock, self).__init__()
        self.conv = nn.Sequential(SeparableConv2d(in_channels, out_channels,
            kernel_size), nn.ReLU())

    def forward(self, input):
        return self.conv(input)


class MobileMattingFCN(nn.Module):

    def __init__(self):
        super(MobileMattingFCN, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=
                False), nn.BatchNorm2d(oup), nn.ReLU(inplace=True))

        def conv_dw(inp, oup, stride):
            return nn.Sequential(nn.Conv2d(inp, inp, 3, stride, 1, groups=
                inp, bias=False), nn.BatchNorm2d(inp), nn.ReLU(inplace=True
                ), nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d
                (oup), nn.ReLU(inplace=True))
        self.model = nn.Sequential(conv_bn(3, 32, 2), conv_dw(32, 64, 1),
            conv_dw(64, 128, 2), conv_dw(128, 128, 1), conv_dw(128, 256, 2),
            conv_dw(256, 256, 1), conv_dw(256, 512, 2), conv_dw(512, 512, 1
            ), conv_dw(512, 512, 1), conv_dw(512, 512, 1), conv_dw(512, 512,
            1), conv_dw(512, 512, 1), conv_dw(512, 1024, 2), conv_dw(1024, 
            1024, 1), conv_dw(1024, 1024, 1))
        self.upsample0 = YellowBlock()
        self.o0 = OrangeBlock(1024 + 512, 64)
        self.upsample1 = YellowBlock()
        self.o1 = OrangeBlock(64 + 256, 64)
        self.upsample2 = YellowBlock()
        self.o2 = OrangeBlock(64 + 128, 64)
        self.upsample3 = YellowBlock()
        self.o3 = OrangeBlock(64 + 64, 64)
        self.upsample4 = YellowBlock()
        self.o4 = OrangeBlock(64, 64)
        self.red = nn.Sequential(nn.Conv2d(64, 1, 1))

    def forward(self, x):
        skips = []
        for i, model in enumerate(self.model):
            x = model(x)
            if i in {1, 3, 5, 11}:
                skips.append(x)
        x = self.upsample0(x)
        x = torch.cat((x, skips[-1]), dim=1)
        x = self.o0(x)
        x = self.upsample1(x)
        x = torch.cat((x, skips[-2]), dim=1)
        x = self.o1(x)
        x = self.upsample2(x)
        x = torch.cat((x, skips[-3]), dim=1)
        x = self.o2(x)
        x = self.upsample3(x)
        x = torch.cat((x, skips[-4]), dim=1)
        x = self.o3(x)
        x = self.upsample4(x)
        x = self.o4(x)
        return self.red(x)

    def load_pretrained_model(self):
        pass


class HairMattingLoss(nn.modules.loss._Loss):

    def __init__(self, ratio_of_Gradient=0.0, add_gradient=False):
        super(HairMattingLoss, self).__init__()
        self.ratio_of_gradient = ratio_of_Gradient
        self.add_gradient = add_gradient
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, pred, true, image):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        loss2 = None
        if self.ratio_of_gradient > 0:
            sobel_kernel_x = torch.Tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -
                2.0], [1.0, 0.0, -1.0]]).to(device)
            sobel_kernel_x = sobel_kernel_x.view((1, 1, 3, 3))
            I_x = F.conv2d(image, sobel_kernel_x)
            G_x = F.conv2d(pred, sobel_kernel_x)
            sobel_kernel_y = torch.Tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0],
                [-1.0, -2.0, -1.0]]).to(device)
            sobel_kernel_y = sobel_kernel_y.view((1, 1, 3, 3))
            I_y = F.conv2d(image, sobel_kernel_y)
            G_y = F.conv2d(pred, sobel_kernel_y)
            G = torch.sqrt(torch.pow(G_x, 2) + torch.pow(G_y, 2))
            rang_grad = 1 - torch.pow(I_x * G_x + I_y * G_y, 2)
            rang_grad = range_grad if rang_grad > 0 else 0
            loss2 = torch.sum(torch.mul(G, rang_grad)) / torch.sum(G) + 1e-06
        if self.add_gradient:
            loss = (1 - self.ratio_of_gradient) * self.bce_loss(pred, true
                ) + loss2 * self.ratio_of_gradient
        else:
            loss = self.bce_loss(pred, true)
        return loss


class ResNet101Extractor(nn.Module):

    def __init__(self):
        super(ResNet101Extractor, self).__init__()
        model = resnet101(pretrained=True)
        self.features = nn.Sequential(*list(model.children())[:7])

    def forward(self, x):
        return self.features(x)


class SqueezeNetExtractor(nn.Module):

    def __init__(self):
        super(SqueezeNetExtractor, self).__init__()
        model = squeezenet1_1(pretrained=True)
        features = model.features
        self.feature1 = features[:2]
        self.feature2 = features[2:5]
        self.feature3 = features[5:8]
        self.feature4 = features[8:]

    def forward(self, x):
        f1 = self.feature1(x)
        f2 = self.feature2(f1)
        f3 = self.feature3(f2)
        f4 = self.feature4(f3)
        return f4


class PyramidPoolingModule(nn.Module):

    def __init__(self, in_channels, sizes=(1, 2, 3, 6)):
        super(PyramidPoolingModule, self).__init__()
        pyramid_levels = len(sizes)
        out_channels = in_channels // pyramid_levels
        pooling_layers = nn.ModuleList()
        for size in sizes:
            layers = [nn.AdaptiveAvgPool2d(size), nn.Conv2d(in_channels,
                out_channels, kernel_size=1)]
            pyramid_layer = nn.Sequential(*layers)
            pooling_layers.append(pyramid_layer)
        self.pooling_layers = pooling_layers

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        features = [x]
        for pooling_layer in self.pooling_layers:
            pooled = pooling_layer(x)
            upsampled = F.upsample(pooled, size=(h, w), mode='bilinear')
            features.append(upsampled)
        return torch.cat(features, dim=1)


class UpsampleLayer(nn.Module):

    def __init__(self, in_channels, out_channels, upsample_size=None):
        super().__init__()
        self.upsample_size = upsample_size
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels,
            kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(
            out_channels), nn.ReLU())

    def forward(self, x):
        size = 2 * x.size(2), 2 * x.size(3)
        f = F.upsample(x, size=size, mode='bilinear')
        return self.conv(f)


class PSPNet(nn.Module):

    def __init__(self, num_class=1, sizes=(1, 2, 3, 6), base_network=
        'resnet101'):
        super(PSPNet, self).__init__()
        base_network = base_network.lower()
        if base_network == 'resnet101':
            self.base_network = ResNet101Extractor()
            feature_dim = 1024
        elif base_network == 'squeezenet':
            self.base_network = SqueezeNetExtractor()
            feature_dim = 512
        else:
            raise ValueError
        self.psp = PyramidPoolingModule(in_channels=feature_dim, sizes=sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)
        self.up_1 = UpsampleLayer(2 * feature_dim, 256)
        self.up_2 = UpsampleLayer(256, 64)
        self.up_3 = UpsampleLayer(64, 64)
        self.drop_2 = nn.Dropout2d(p=0.15)
        self.final = nn.Sequential(nn.Conv2d(64, num_class, kernel_size=1))
        self._init_weight()

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        f = self.base_network(x)
        p = self.psp(f)
        p = self.drop_1(p)
        p = self.up_1(p)
        p = self.drop_2(p)
        p = self.up_2(p)
        p = self.drop_2(p)
        p = self.up_3(p)
        if p.size(2) != h or p.size(3) != w:
            p = F.interpolate(p, size=(h, w), mode='bilinear')
        p = self.drop_2(p)
        return self.final(p)

    def _init_weight(self):
        layers = [self.up_1, self.up_2, self.up_3, self.final]
        for layer in layers:
            if isinstance(layer, nn.Conv2d):
                xavier_normal_(layer.weight.data)
            elif isinstance(layer, nn.BatchNorm2d):
                layer.weight.data.normal_(1.0, 0.02)
                layer.bias.data.fill_(0)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_YBIGTA_pytorch_hair_segmentation(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(SeparableConv2d(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(_ASPPModule(*[], **{'inplanes': 4, 'planes': 4, 'kernel_size': 4, 'padding': 4, 'dilation': 1}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_002(self):
        self._check(YellowBlock(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_003(self):
        self._check(OrangeBlock(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_004(self):
        self._check(MobileMattingFCN(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

    @_fails_compile()
    def test_005(self):
        self._check(HairMattingLoss(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_006(self):
        self._check(PyramidPoolingModule(*[], **{'in_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_007(self):
        self._check(UpsampleLayer(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

