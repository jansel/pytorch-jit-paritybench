import sys
_module = sys.modules[__name__]
del sys
data_config = _module
dataset = _module
esp_dense_seg = _module
mv2_dilate_unet = _module
residualdense_bisenet = _module
shuffle_seg_skipnet = _module

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


from collections import OrderedDict


from torch.nn import init


class make_dense(nn.Module):

    def __init__(self, nChannels, growthRate, dilation):
        super(make_dense, self).__init__()
        self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=3, padding
            =dilation, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(growthRate)

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


class DenseBlock(nn.Module):

    def __init__(self, nChannels, nDenselayer, growthRate, d, reset_channel
        =False):
        super(DenseBlock, self).__init__()
        nChannels_ = nChannels
        modules = []
        for i in range(nDenselayer):
            modules.append(make_dense(nChannels_, growthRate, dilation=d[i]))
            nChannels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.reset_channel = reset_channel
        if self.reset_channel:
            self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1,
                stride=1, padding=0, bias=False)

    def forward(self, x):
        out = self.dense_layers(x)
        if self.reset_channel:
            out = self.conv_1x1(out)
        return out


class DDRB(nn.Module):

    def __init__(self, nIn, s=4, d=[1, 2, 4], add=True):
        super().__init__()
        n = int(nIn // s)
        self.conv = nn.Conv2d(nIn, n, 1, stride=1, padding=0, bias=False)
        self.dense_block = DenseBlock(n, nDenselayer=s - 1, growthRate=n, d=d)
        self.bn = nn.BatchNorm2d(nIn)
        self.act = nn.ReLU()
        self.add = add

    def forward(self, input):
        inter = self.conv(input)
        dense_out = self.dense_block(inter)
        if self.add:
            combine = input + dense_out
        output = self.act(self.bn(combine))
        return output


class DDB(nn.Module):

    def __init__(self, nIn, d=[1, 2, 4]):
        super().__init__()
        self.dense_block = DenseBlock(nIn, nDenselayer=3, growthRate=nIn, d
            =d, reset_channel=True)
        self.bn = nn.BatchNorm2d(nIn)
        self.act = nn.ReLU()

    def forward(self, input):
        dense_out = self.dense_block(input)
        output = self.act(self.bn(dense_out))
        return output


class DPRB(nn.Module):
    """
    This class defines the ESP block, which is based on the following principle
        Reduce ---> Split ---> Transform --> Merge
    """

    def __init__(self, nIn, add=True):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param add: if true, add a residual connection through identity operation. You can use projection too as
                in ResNet paper, but we avoid to use it if the dimensions are not the same because we do not want to
                increase the module complexity
        """
        super().__init__()
        n = int(nIn // 6)
        self.conv = nn.Conv2d(nIn, n, 3, stride=1, padding=1, bias=False)
        self.d0 = nn.Conv2d(n, n, 1, 1, 0, bias=False)
        self.d1 = nn.Conv2d(n, n, 3, 1, padding=1, dilation=1, bias=False)
        self.d2 = nn.Conv2d(n, n, 3, 1, padding=2, dilation=2, bias=False)
        self.d4 = nn.Conv2d(n, n, 3, 1, padding=4, dilation=4, bias=False)
        self.d8 = nn.Conv2d(n, n, 3, 1, padding=8, dilation=8, bias=False)
        self.d16 = nn.Conv2d(n, n, 3, 1, padding=16, dilation=16, bias=False)
        self.bn = nn.BatchNorm2d(nIn, eps=0.001)
        self.act = nn.PReLU(nIn)
        self.add = add

    def forward(self, input):
        inter = self.conv(input)
        d0 = self.d0(inter)
        d1 = self.d1(inter)
        d2 = self.d2(inter)
        d4 = self.d4(inter)
        d8 = self.d8(inter)
        d16 = self.d16(inter)
        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16
        combine = torch.cat([d0, d1, add1, add2, add3, add4], 1)
        if self.add:
            combine = input + combine
        output = self.act(self.bn(combine))
        return output


class ESPD_SegNet(nn.Module):

    def __init__(self, classes=1, p=1, q=1, r=2, t=2):
        """
        :param classes: number of classes in the dataset.
        """
        super().__init__()
        self.conv0 = nn.Conv2d(3, 12, 3, stride=1, padding=1, bias=False)
        self.b0 = nn.BatchNorm2d(12, eps=0.001)
        self.a0 = nn.PReLU(12)
        self.down_1 = nn.Conv2d(12, 12, 3, stride=2, padding=1, bias=False)
        self.stage_1_0 = DPRB(12, add=True)
        block = [DDRB(12, s=6, d=[1, 2, 4, 6, 8], add=True) for _ in range(p)]
        self.stage_1 = nn.Sequential(*block)
        self.b1 = nn.BatchNorm2d(24, eps=0.001)
        self.a1 = nn.PReLU(24)
        self.down_2 = nn.Conv2d(24, 24, 3, stride=2, padding=1, bias=False)
        self.stage_2_0 = DPRB(24, add=True)
        block = [DDRB(24, s=6, d=[1, 2, 4, 6, 8], add=True) for _ in range(q)]
        self.stage_2 = nn.Sequential(*block)
        self.b2 = nn.BatchNorm2d(48, eps=0.001)
        self.a2 = nn.PReLU(48)
        self.down_3 = nn.Conv2d(48, 48, 3, stride=2, padding=1, bias=False)
        self.stage_3_0 = DPRB(48, add=True)
        block = [DDRB(48, s=6, d=[1, 2, 4, 6, 8], add=True) for _ in range(r)]
        self.stage_3 = nn.Sequential(*block)
        self.b3 = nn.BatchNorm2d(96, eps=0.001)
        self.a3 = nn.PReLU(96)
        self.down_4 = nn.Conv2d(96, 96, 3, stride=2, padding=1, bias=False)
        self.stage_4_0 = DPRB(96, add=True)
        block = [DDRB(96, s=6, d=[1, 2, 4, 6, 8], add=True) for _ in range(t)]
        self.stage_4 = nn.Sequential(*block)
        self.b4 = nn.BatchNorm2d(192, eps=0.001)
        self.a4 = nn.PReLU(192)
        self.classifier = nn.Conv2d(192, 1, 1, stride=1, padding=0, bias=False)
        self.bn_ = nn.BatchNorm2d(1, eps=0.001)
        self.relu_ = nn.ReLU()
        self.stage3_down = nn.Conv2d(96, 1, kernel_size=1, stride=1,
            padding=0, bias=False)
        self.conv_1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1,
            bias=False)
        self.stage2_down = nn.Conv2d(48, 1, kernel_size=1, stride=1,
            padding=0, bias=False)
        self.conv_2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1,
            bias=False)
        self.stage1_down = nn.Conv2d(24, 1, kernel_size=1, stride=1,
            padding=0, bias=False)
        self.conv_3 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1,
            bias=False)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input):
        x = self.conv0(input)
        x = self.a0(self.b0(x))
        s1_pool = self.down_1(x)
        s1_0 = self.stage_1_0(s1_pool)
        s1 = self.stage_1(s1_0)
        concat_1 = torch.cat((s1, s1_0), dim=1)
        concat_1 = self.a1(self.b1(concat_1))
        s2_pool = self.down_2(concat_1)
        s2_0 = self.stage_2_0(s2_pool)
        s2 = self.stage_2(s2_0)
        concat_2 = torch.cat((s2, s2_0), dim=1)
        concat_2 = self.a2(self.b2(concat_2))
        s3_pool = self.down_3(concat_2)
        s3_0 = self.stage_3_0(s3_pool)
        s3 = self.stage_3(s3_0)
        concat_3 = torch.cat((s3, s3_0), dim=1)
        concat_3 = self.a3(self.b3(concat_3))
        s4_pool = self.down_4(concat_3)
        s4_0 = self.stage_4_0(s4_pool)
        s4 = self.stage_4(s4_0)
        concat_4 = torch.cat((s4, s4_0), dim=1)
        concat_4 = self.a4(self.b4(concat_4))
        heatmap = self.classifier(concat_4)
        heatmap_1 = F.upsample(heatmap, scale_factor=2, mode='bilinear',
            align_corners=True)
        s3_heatmap = self.bn_(self.stage3_down(concat_3))
        heatmap_1 = heatmap_1 + s3_heatmap
        heatmap_1 = self.conv_1(heatmap_1)
        heatmap_2 = F.upsample(heatmap_1, scale_factor=2, mode='bilinear',
            align_corners=True)
        s2_heatmap = self.bn_(self.stage2_down(concat_2))
        heatmap_2 = heatmap_2 + s2_heatmap
        heatmap_2 = self.conv_2(heatmap_2)
        heatmap_3 = F.upsample(heatmap_2, scale_factor=2, mode='bilinear',
            align_corners=True)
        s1_heatmap = self.bn_(self.stage1_down(concat_1))
        heatmap_3 = heatmap_3 + s1_heatmap
        heatmap_3 = self.conv_3(heatmap_3)
        out = F.upsample(heatmap_3, scale_factor=2, mode='bilinear',
            align_corners=True)
        return out


class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        self.use_res_connect = self.stride == 1 and inp == oup
        self.conv = nn.Sequential(nn.Conv2d(inp, inp * expand_ratio, 1, 1, 
            0, bias=False), nn.BatchNorm2d(inp * expand_ratio), nn.ReLU6(
            inplace=True), nn.Conv2d(inp * expand_ratio, inp * expand_ratio,
            3, stride, 1, groups=inp * expand_ratio, bias=False), nn.
            BatchNorm2d(inp * expand_ratio), nn.ReLU6(inplace=True), nn.
            Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False), nn.
            BatchNorm2d(oup))

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


def conv_bn(inp, oup, kernel_size=3, padding=1, stride=1):
    return nn.Sequential(nn.Conv2d(inp, oup, kernel_size, stride, padding,
        bias=False), nn.BatchNorm2d(oup), nn.PReLU(oup))


class MobileNet_v2_os_32_MFo(nn.Module):

    def __init__(self, nInputChannels=3):
        super(MobileNet_v2_os_32_MFo, self).__init__()
        self.head_conv = conv_bn(nInputChannels, 32, 2)
        self.block_1 = InvertedResidual(32, 16, 1, 1)
        self.block_2 = nn.Sequential(InvertedResidual(16, 24, 2, 6),
            InvertedResidual(24, 24, 1, 6))
        self.block_3 = nn.Sequential(InvertedResidual(24, 32, 2, 6),
            InvertedResidual(32, 32, 1, 6), InvertedResidual(32, 32, 1, 6))
        self.block_4 = nn.Sequential(InvertedResidual(32, 64, 2, 6),
            InvertedResidual(64, 64, 1, 6), InvertedResidual(64, 64, 1, 6),
            InvertedResidual(64, 64, 1, 6))
        self.block_5 = nn.Sequential(InvertedResidual(64, 96, 1, 6),
            InvertedResidual(96, 96, 1, 6), InvertedResidual(96, 96, 1, 6))
        self.block_6 = nn.Sequential(InvertedResidual(96, 160, 2, 6),
            InvertedResidual(160, 160, 1, 6), InvertedResidual(160, 160, 1, 6))
        self.block_7 = InvertedResidual(160, 320, 1, 6)

    def forward(self, x):
        x = self.head_conv(x)
        x1 = self.block_1(x)
        x2 = self.block_2(x1)
        x3 = self.block_3(x2)
        x4 = self.block_4(x3)
        x4 = self.block_5(x4)
        x5 = self.block_6(x4)
        x5 = self.block_7(x5)
        return x1, x2, x3, x4, x5


class UCD(nn.Module):

    def __init__(self, inplanes, planes, dilation):
        super(UCD, self).__init__()
        self.up = nn.ConvTranspose2d(inplanes, planes, kernel_size=2,
            stride=2, padding=0)
        self.aspp = nn.Sequential(nn.Conv2d(planes * 2, planes * 2,
            kernel_size=3, stride=1, padding=dilation, dilation=dilation,
            bias=False), nn.BatchNorm2d(planes * 2))

    def forward(self, e, x):
        x = self.up(x)
        x = torch.cat((x, e), dim=1)
        x = self.aspp(x)
        return x


class MobileNet_v2_Dilate_Unet(nn.Module):

    def __init__(self, nInputChannels=3, n_classes=1):
        super(MobileNet_v2_Dilate_Unet, self).__init__()
        self.mobilenet_features = MobileNet_v2_os_32_MFo(nInputChannels)
        self.up_concat_dilate_1 = UCD(320, 96, dilation=2)
        self.up_concat_dilate_2 = UCD(192, 32, dilation=6)
        self.up_concat_dilate_3 = UCD(64, 24, dilation=12)
        self.up_concat_dilate_4 = UCD(48, 16, dilation=18)
        self.last_conv = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3,
            stride=1, padding=1, bias=False), nn.BatchNorm2d(32), nn.Conv2d
            (32, n_classes, kernel_size=1, stride=1))
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        e1, e2, e3, e4, feature_map = self.mobilenet_features(x)
        feature_map = self.up_concat_dilate_1(e4, feature_map)
        feature_map = self.up_concat_dilate_2(e3, feature_map)
        feature_map = self.up_concat_dilate_3(e2, feature_map)
        feature_map = self.up_concat_dilate_4(e1, feature_map)
        heat_map = self.last_conv(feature_map)
        heat_map = F.upsample(heat_map, scale_factor=2, mode='bilinear',
            align_corners=True)
        return heat_map


class make_dense(nn.Module):

    def __init__(self, nChannels, growthRate):
        super(make_dense, self).__init__()
        self.bn = nn.BatchNorm2d(nChannels)
        self.act = nn.PReLU(nChannels)
        self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=3, padding
            =1, bias=False)

    def forward(self, input):
        x = self.bn(input)
        x = self.act(x)
        x = self.conv(x)
        out = torch.cat((input, x), 1)
        return out


class DenseBlock(nn.Module):

    def __init__(self, nChannels, nDenselayer, growthRate):
        super(DenseBlock, self).__init__()
        nChannels_ = nChannels
        modules = []
        for i in range(nDenselayer):
            modules.append(make_dense(nChannels_, growthRate))
            nChannels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)

    def forward(self, x):
        out = self.dense_layers(x)
        return out


class DRB(nn.Module):

    def __init__(self, nIn, s=4, add=True):
        super(DRB, self).__init__()
        n = int(nIn // s)
        self.conv = nn.Conv2d(nIn, n, 3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(n)
        self.act = nn.PReLU(n)
        self.dense_block = DenseBlock(n, nDenselayer=s - 1, growthRate=n)
        self.add = add

    def forward(self, input):
        residual = input
        x = self.conv(input)
        x = self.bn(x)
        x = self.act(x)
        x = self.dense_block(x)
        if self.add:
            out = x + residual
        else:
            out = x
        return out


class ARM(nn.Module):

    def __init__(self, in_channels, kernel_size):
        super(ARM, self).__init__()
        self.global_pool = nn.AvgPool2d(kernel_size, stride=kernel_size)
        self.conv_1x1 = nn.Conv2d(in_channels, in_channels, 1, stride=1,
            padding=0, bias=False)
        self.sigmod = nn.Sigmoid()

    def forward(self, input):
        x = self.global_pool(input)
        x = self.conv_1x1(x)
        x = self.sigmod(x)
        out = torch.mul(input, x)
        return out


class FFM(nn.Module):

    def __init__(self, in_channels, kernel_size, alpha=3):
        super(FFM, self).__init__()
        inter_channels = in_channels // alpha
        self.conv_bn_relu = conv_bn(in_channels, inter_channels,
            kernel_size=1, padding=0)
        self.global_pool = nn.AvgPool2d(kernel_size, stride=kernel_size)
        self.conv_1x1_1 = nn.Conv2d(inter_channels, inter_channels, 1,
            stride=1, padding=0, bias=False)
        self.relu = nn.ReLU()
        self.conv_1x1_2 = nn.Conv2d(inter_channels, inter_channels, 1,
            stride=1, padding=0, bias=False)
        self.sigmod = nn.Sigmoid()
        self.classifier = nn.Conv2d(inter_channels, 1, 1, stride=1, padding
            =0, bias=True)

    def forward(self, input):
        input = self.conv_bn_relu(input)
        x = self.global_pool(input)
        x = self.conv_1x1_1(x)
        x = self.relu(x)
        x = self.conv_1x1_2(x)
        x = self.sigmod(x)
        out = torch.mul(input, x)
        out = input + out
        out = self.classifier(out)
        return out


class RD_BiSeNet(nn.Module):

    def __init__(self, classes=1, cfg=None):
        super(RD_BiSeNet, self).__init__()
        self.conv_bn_relu_1 = conv_bn(3, 8, stride=2)
        self.conv_bn_relu_2 = conv_bn(8, 12, stride=2)
        self.conv_bn_relu_3 = conv_bn(12, 16, stride=2)
        self.conv = conv_bn(3, 8, stride=2)
        self.stage_0 = DRB(8, s=2, add=True)
        self.down_1 = conv_bn(8, 12, stride=2)
        self.stage_1 = DRB(12, s=3, add=True)
        self.down_2 = conv_bn(12, 24, stride=2)
        self.stage_2 = nn.Sequential(DRB(24, s=6, add=True), DRB(24, s=6,
            add=True))
        self.down_3 = conv_bn(24, 48, stride=2)
        self.stage_3 = nn.Sequential(DRB(48, s=6, add=True), DRB(48, s=6,
            add=True))
        self.down_4 = conv_bn(48, 64, stride=2)
        self.stage_4 = nn.Sequential(DRB(64, s=8, add=True), DRB(64, s=8,
            add=True))
        self.arm_16 = ARM(48, kernel_size=32)
        self.arm_32 = ARM(64, kernel_size=16)
        self.global_pool = nn.AvgPool2d(kernel_size=16, stride=16)
        self.tail_up = nn.Upsample(scale_factor=16, mode='bilinear',
            align_corners=True)
        self.level_16_up = nn.Upsample(scale_factor=2, mode='bilinear',
            align_corners=True)
        self.level_32_up = nn.Sequential(nn.Upsample(scale_factor=4, mode=
            'bilinear', align_corners=True), nn.Conv2d(64, 48, 1, stride=1,
            padding=0, bias=True))
        self.ffm = FFM(48 + 16, kernel_size=64, alpha=1)
        self.up = nn.Upsample(scale_factor=8, mode='bilinear',
            align_corners=True)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input):
        spatial = self.conv_bn_relu_1(input)
        spatial = self.conv_bn_relu_2(spatial)
        spatial = self.conv_bn_relu_3(spatial)
        x = self.conv(input)
        s0 = self.stage_0(x)
        s1_0 = self.down_1(s0)
        s1 = self.stage_1(s1_0)
        s2_0 = self.down_2(s1)
        s2 = self.stage_2(s2_0)
        s3_0 = self.down_3(s2)
        s3 = self.stage_3(s3_0)
        s4_0 = self.down_4(s3)
        s4 = self.stage_4(s4_0)
        level_global = self.global_pool(s4)
        level_global = self.tail_up(level_global)
        level_32 = self.arm_32(s4)
        level_32 = level_32 + level_global
        level_32 = self.level_32_up(level_32)
        level_16 = self.arm_16(s3)
        level_16 = self.level_16_up(level_16)
        context = level_16 + level_32
        feature = torch.cat((spatial, context), 1)
        heatmap = self.ffm(feature)
        out = self.up(heatmap)
        return out


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x


def conv1x1(in_channels, out_channels, groups=1, bias=True):
    """1x1 convolution with padding
    - Normal pointwise convolution When groups == 1
    - Grouped pointwise convolution when groups > 1
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=
        groups, stride=1, bias=bias)


def conv3x3(in_channels, out_channels, stride=1, padding=1, bias=True, groups=1
    ):
    """3x3 convolution with padding
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=
        stride, padding=padding, bias=bias, groups=groups)


class ShuffleUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=3, grouped_conv=
        True, combine='add'):
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

    def __init__(self, groups=3, in_channels=3):
        """ShuffleNet constructor.

        Arguments:
            groups (int, optional): number of groups to be used in grouped 
                1x1 convolutions in each ShuffleUnit. Default is 3 for best
                performance according to original paper.
            in_channels (int, optional): number of channels in the input tensor.
                Default is 3 for RGB image inputs.
            num_classes (int, optional): number of classes to predict. Default
                is 1000 for ImageNet.

        """
        super(ShuffleNet, self).__init__()
        self.groups = groups
        self.stage_repeats = [3, 7, 3]
        self.in_channels = in_channels
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
                .format(num_groups))
        self.conv1 = conv3x3(self.in_channels, self.stage_out_channels[1],
            stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stage2 = self._make_stage(2)
        self.stage3 = self._make_stage(3)
        self.stage4 = self._make_stage(4)

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
        s0 = self.conv1(x)
        s1 = self.maxpool(s0)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)
        return s0, s1, s2, s3, s4


class Shuffle_Seg_SkipNet(nn.Module):

    def __init__(self, groups=3, in_channels=3, n_classes=1):
        """ShuffleNet constructor.
        """
        super(Shuffle_Seg_SkipNet, self).__init__()
        self.encoder = ShuffleNet(groups=groups, in_channels=in_channels)
        self.scorelayer = conv1x1(960, 1, bias=True)
        self.bn_ = nn.BatchNorm2d(1)
        self.up1 = BilinearConvTranspose2d(1, stride=2, groups=1)
        self.stage3_down = conv1x1(480, 1, groups=1, bias=False)
        self.up2 = BilinearConvTranspose2d(1, stride=2, groups=1)
        self.stage2_down = conv1x1(240, 1, groups=1, bias=False)
        self.up3 = BilinearConvTranspose2d(1, stride=2, groups=1)
        self.stage1_down = conv1x1(24, 1, groups=1, bias=False)
        self.up4 = BilinearConvTranspose2d(1, stride=2, groups=1)
        self.stage0_down = conv1x1(24, 1, groups=1, bias=False)
        self.deconv = BilinearConvTranspose2d(1, stride=2, groups=1)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

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
        s0, s1, s2, s3, s4 = self.encoder(x)
        heat_map = self.bn_(self.scorelayer(s4))
        heat_map = self.bn_(self.up1(heat_map))
        s3_heat_map = self.bn_(self.stage3_down(s3))
        heat_map = heat_map + s3_heat_map
        heat_map = self.bn_(self.up2(heat_map))
        s2_heat_map = self.bn_(self.stage2_down(s2))
        heat_map = heat_map + s2_heat_map
        heat_map = self.bn_(self.up3(heat_map))
        s1_heat_map = self.bn_(self.stage1_down(s1))
        heat_map = heat_map + s1_heat_map
        heat_map = self.bn_(self.up4(heat_map))
        s0_heat_map = self.bn_(self.stage0_down(s0))
        heat_map = heat_map + s0_heat_map
        heat_map = self.deconv(heat_map)
        return heat_map


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_lizhengwei1992_Fast_Portrait_Segmentation(_paritybench_base):
    pass
    def test_000(self):
        self._check(ARM(*[], **{'in_channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(DRB(*[], **{'nIn': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(DenseBlock(*[], **{'nChannels': 4, 'nDenselayer': 1, 'growthRate': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(FFM(*[], **{'in_channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_004(self):
        self._check(InvertedResidual(*[], **{'inp': 4, 'oup': 4, 'stride': 1, 'expand_ratio': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_005(self):
        self._check(MobileNet_v2_os_32_MFo(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

    @_fails_compile()
    def test_006(self):
        self._check(ShuffleNet(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

    def test_007(self):
        self._check(UCD(*[], **{'inplanes': 4, 'planes': 4, 'dilation': 1}), [torch.rand([4, 4, 8, 8]), torch.rand([4, 4, 4, 4])], {})

    def test_008(self):
        self._check(make_dense(*[], **{'nChannels': 4, 'growthRate': 4}), [torch.rand([4, 4, 4, 4])], {})

