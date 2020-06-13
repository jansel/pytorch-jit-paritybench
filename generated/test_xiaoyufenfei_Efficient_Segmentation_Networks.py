import sys
_module = sys.modules[__name__]
del sys
builders = _module
dataset_builder = _module
model_builder = _module
dataset = _module
camvid = _module
cityscape_scripts = _module
generate_mappings = _module
print_utils = _module
process_cityscapes = _module
cityscapes = _module
create_dataset_list = _module
CGNet = _module
ContextNet = _module
DABNet = _module
EDANet = _module
ENet = _module
ERFNet = _module
ESNet = _module
ESPNet = _module
Model = _module
SegmentationModel = _module
cnn_utils = _module
FPENet = _module
FSSNet = _module
FastSCNN = _module
LEDNet = _module
LinkNet = _module
SQNet = _module
SegNet = _module
UNet = _module
predict = _module
test = _module
ENet_Flops_test = _module
ptflops = _module
flops_counter = _module
sample = _module
setup = _module
eval_forward_time = _module
trainID2labelID = _module
train = _module
activations = _module
colorize_mask = _module
convert_state = _module
debug = _module
losses = _module
loss = _module
lovasz_losses = _module
metric = _module
AdamW = _module
Lookahead = _module
RAdam = _module
Ranger = _module
optim = _module
scheduler = _module
lr_scheduler = _module
utils = _module

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


import torch.nn.init as init


import math


from torch.nn import init


from torch.autograd import Variable


import numpy as np


import torch.optim as optim


from torch import optim


import torch.backends.cudnn as cudnn


from torch.nn.modules.loss import _Loss


from torch.nn.modules.loss import _WeightedLoss


from torch.nn import NLLLoss2d


import random


class ConvBNPReLU(nn.Module):

    def __init__(self, nIn, nOut, kSize, stride=1):
        """
        args:
            nIn: number of input channels
            nOut: number of output channels
            kSize: kernel size
            stride: stride rate for down-sampling. Default is 1
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride,
            padding=(padding, padding), bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=0.001)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.conv(input)
        output = self.bn(output)
        output = self.act(output)
        return output


class BNPReLU(nn.Module):

    def __init__(self, nOut):
        """
        args:
           nOut: channels of output feature maps
        """
        super().__init__()
        self.bn = nn.BatchNorm2d(nOut, eps=0.001)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        """
        args:
           input: input feature map
           return: normalized and thresholded feature map
        """
        output = self.bn(input)
        output = self.act(output)
        return output


class ConvBN(nn.Module):

    def __init__(self, nIn, nOut, kSize, stride=1):
        """
        args:
           nIn: number of input channels
           nOut: number of output channels
           kSize: kernel size
           stride: optinal stide for down-sampling
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride,
            padding=(padding, padding), bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=0.001)

    def forward(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.conv(input)
        output = self.bn(output)
        return output


class Conv(nn.Module):

    def __init__(self, nIn, nOut, kSize, stride=1):
        """
        args:
            nIn: number of input channels
            nOut: number of output channels
            kSize: kernel size
            stride: optional stride rate for down-sampling
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride,
            padding=(padding, padding), bias=False)

    def forward(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.conv(input)
        return output


class ChannelWiseConv(nn.Module):

    def __init__(self, nIn, nOut, kSize, stride=1):
        """
        Args:
            nIn: number of input channels
            nOut: number of output channels, default (nIn == nOut)
            kSize: kernel size
            stride: optional stride rate for down-sampling
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride,
            padding=(padding, padding), groups=nIn, bias=False)

    def forward(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.conv(input)
        return output


class DilatedConv(nn.Module):

    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        """
        args:
           nIn: number of input channels
           nOut: number of output channels
           kSize: kernel size
           stride: optional stride rate for down-sampling
           d: dilation rate
        """
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride,
            padding=(padding, padding), bias=False, dilation=d)

    def forward(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.conv(input)
        return output


class ChannelWiseDilatedConv(nn.Module):

    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        """
        args:
           nIn: number of input channels
           nOut: number of output channels, default (nIn == nOut)
           kSize: kernel size
           stride: optional stride rate for down-sampling
           d: dilation rate
        """
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride,
            padding=(padding, padding), groups=nIn, bias=False, dilation=d)

    def forward(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.conv(input)
        return output


class FGlo(nn.Module):
    """
    the FGlo class is employed to refine the joint feature of both local feature and surrounding context.
    """

    def __init__(self, channel, reduction=16):
        super(FGlo, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True), nn.Linear(channel // reduction, channel),
            nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ContextGuidedBlock_Down(nn.Module):
    """
    the size of feature map divided 2, (H,W,C)---->(H/2, W/2, 2C)
    """

    def __init__(self, nIn, nOut, dilation_rate=2, reduction=16):
        """
        args:
           nIn: the channel of input feature map
           nOut: the channel of output feature map, and nOut=2*nIn
        """
        super().__init__()
        self.conv1x1 = ConvBNPReLU(nIn, nOut, 3, 2)
        self.F_loc = ChannelWiseConv(nOut, nOut, 3, 1)
        self.F_sur = ChannelWiseDilatedConv(nOut, nOut, 3, 1, dilation_rate)
        self.bn = nn.BatchNorm2d(2 * nOut, eps=0.001)
        self.act = nn.PReLU(2 * nOut)
        self.reduce = Conv(2 * nOut, nOut, 1, 1)
        self.F_glo = FGlo(nOut, reduction)

    def forward(self, input):
        output = self.conv1x1(input)
        loc = self.F_loc(output)
        sur = self.F_sur(output)
        joi_feat = torch.cat([loc, sur], 1)
        joi_feat = self.bn(joi_feat)
        joi_feat = self.act(joi_feat)
        joi_feat = self.reduce(joi_feat)
        output = self.F_glo(joi_feat)
        return output


class ContextGuidedBlock(nn.Module):

    def __init__(self, nIn, nOut, dilation_rate=2, reduction=16, add=True):
        """
        args:
           nIn: number of input channels
           nOut: number of output channels, 
           add: if true, residual learning
        """
        super().__init__()
        n = int(nOut / 2)
        self.conv1x1 = ConvBNPReLU(nIn, n, 1, 1)
        self.F_loc = ChannelWiseConv(n, n, 3, 1)
        self.F_sur = ChannelWiseDilatedConv(n, n, 3, 1, dilation_rate)
        self.bn_prelu = BNPReLU(nOut)
        self.add = add
        self.F_glo = FGlo(nOut, reduction)

    def forward(self, input):
        output = self.conv1x1(input)
        loc = self.F_loc(output)
        sur = self.F_sur(output)
        joi_feat = torch.cat([loc, sur], 1)
        joi_feat = self.bn_prelu(joi_feat)
        output = self.F_glo(joi_feat)
        if self.add:
            output = input + output
        return output


class InputInjection(nn.Module):

    def __init__(self, downsamplingRatio):
        super().__init__()
        self.pool = nn.ModuleList()
        for i in range(0, downsamplingRatio):
            self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, input):
        for pool in self.pool:
            input = pool(input)
        return input


class CGNet(nn.Module):
    """
    This class defines the proposed Context Guided Network (CGNet) in this work.
    """

    def __init__(self, classes=19, M=3, N=21, dropout_flag=False):
        """
        args:
          classes: number of classes in the dataset. Default is 19 for the cityscapes
          M: the number of blocks in stage 2
          N: the number of blocks in stage 3
        """
        super().__init__()
        self.level1_0 = ConvBNPReLU(3, 32, 3, 2)
        self.level1_1 = ConvBNPReLU(32, 32, 3, 1)
        self.level1_2 = ConvBNPReLU(32, 32, 3, 1)
        self.sample1 = InputInjection(1)
        self.sample2 = InputInjection(2)
        self.b1 = BNPReLU(32 + 3)
        self.level2_0 = ContextGuidedBlock_Down(32 + 3, 64, dilation_rate=2,
            reduction=8)
        self.level2 = nn.ModuleList()
        for i in range(0, M - 1):
            self.level2.append(ContextGuidedBlock(64, 64, dilation_rate=2,
                reduction=8))
        self.bn_prelu_2 = BNPReLU(128 + 3)
        self.level3_0 = ContextGuidedBlock_Down(128 + 3, 128, dilation_rate
            =4, reduction=16)
        self.level3 = nn.ModuleList()
        for i in range(0, N - 1):
            self.level3.append(ContextGuidedBlock(128, 128, dilation_rate=4,
                reduction=16))
        self.bn_prelu_3 = BNPReLU(256)
        if dropout_flag:
            None
            self.classifier = nn.Sequential(nn.Dropout2d(0.1, False), Conv(
                256, classes, 1, 1))
        else:
            self.classifier = nn.Sequential(Conv(256, classes, 1, 1))
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
                elif classname.find('ConvTranspose2d') != -1:
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        m.bias.data.zero_()

    def forward(self, input):
        """
        args:
            input: Receives the input RGB image
            return: segmentation map
        """
        output0 = self.level1_0(input)
        output0 = self.level1_1(output0)
        output0 = self.level1_2(output0)
        inp1 = self.sample1(input)
        inp2 = self.sample2(input)
        output0_cat = self.b1(torch.cat([output0, inp1], 1))
        output1_0 = self.level2_0(output0_cat)
        for i, layer in enumerate(self.level2):
            if i == 0:
                output1 = layer(output1_0)
            else:
                output1 = layer(output1)
        output1_cat = self.bn_prelu_2(torch.cat([output1, output1_0, inp2], 1))
        output2_0 = self.level3_0(output1_cat)
        for i, layer in enumerate(self.level3):
            if i == 0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)
        output2_cat = self.bn_prelu_3(torch.cat([output2_0, output2], 1))
        classifier = self.classifier(output2_cat)
        out = F.interpolate(classifier, input.size()[2:], mode='bilinear',
            align_corners=False)
        return out


class Custom_Conv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
        padding=0, **kwargs):
        super(Custom_Conv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels,
            kernel_size, stride, padding, bias=False), nn.BatchNorm2d(
            out_channels), nn.ReLU(True))

    def forward(self, x):
        return self.conv(x)


class DepthSepConv(nn.Module):

    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(DepthSepConv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(dw_channels, dw_channels, 3,
            stride, 1, groups=dw_channels, bias=False), nn.BatchNorm2d(
            dw_channels), nn.ReLU(True), nn.Conv2d(dw_channels,
            out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.
            ReLU(True))

    def forward(self, x):
        return self.conv(x)


class DepthConv(nn.Module):

    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(DepthConv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(dw_channels, out_channels, 3,
            stride, 1, groups=dw_channels, bias=False), nn.BatchNorm2d(
            out_channels), nn.ReLU(True))

    def forward(self, x):
        return self.conv(x)


class LinearBottleneck(nn.Module):

    def __init__(self, in_channels, out_channels, t=6, stride=2, **kwargs):
        super(LinearBottleneck, self).__init__()
        self.use_shortcut = stride == 1 and in_channels == out_channels
        self.block = nn.Sequential(Custom_Conv(in_channels, in_channels * t,
            1), DepthConv(in_channels * t, in_channels * t, stride), nn.
            Conv2d(in_channels * t, out_channels, 1, bias=False), nn.
            BatchNorm2d(out_channels))

    def forward(self, x):
        out = self.block(x)
        if self.use_shortcut:
            out = x + out
        return out


class Shallow_net(nn.Module):

    def __init__(self, dw_channels1=32, dw_channels2=64, out_channels=128,
        **kwargs):
        super(Shallow_net, self).__init__()
        self.conv = Custom_Conv(3, dw_channels1, 3, 2)
        self.dsconv1 = DepthSepConv(dw_channels1, dw_channels2, 2)
        self.dsconv2 = DepthSepConv(dw_channels2, out_channels, 2)
        self.dsconv3 = DepthSepConv(out_channels, out_channels, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        x = self.dsconv3(x)
        return x


class Deep_net(nn.Module):

    def __init__(self, in_channels, block_channels, t, num_blocks, **kwargs):
        super(Deep_net, self).__init__()
        self.block_channels = block_channels
        self.t = t
        self.num_blocks = num_blocks
        self.conv_ = Custom_Conv(3, in_channels, 3, 2)
        self.bottleneck1 = self._layer(LinearBottleneck, in_channels,
            block_channels[0], num_blocks[0], t[0], 1)
        self.bottleneck2 = self._layer(LinearBottleneck, block_channels[0],
            block_channels[1], num_blocks[1], t[1], 1)
        self.bottleneck3 = self._layer(LinearBottleneck, block_channels[1],
            block_channels[2], num_blocks[2], t[2], 2)
        self.bottleneck4 = self._layer(LinearBottleneck, block_channels[2],
            block_channels[3], num_blocks[3], t[3], 2)
        self.bottleneck5 = self._layer(LinearBottleneck, block_channels[3],
            block_channels[4], num_blocks[4], t[4], 1)
        self.bottleneck6 = self._layer(LinearBottleneck, block_channels[4],
            block_channels[5], num_blocks[5], t[5], 1)

    def _layer(self, block, in_channels, out_channels, blocks, t, stride):
        layers = []
        layers.append(block(in_channels, out_channels, t, stride))
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels, t, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_(x)
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.bottleneck4(x)
        x = self.bottleneck5(x)
        x = self.bottleneck6(x)
        return x


class FeatureFusionModule(nn.Module):

    def __init__(self, highter_in_channels, lower_in_channels, out_channels,
        scale_factor=4, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.scale_factor = scale_factor
        self.dwconv = DepthConv(lower_in_channels, out_channels, 1)
        self.conv_lower_res = nn.Sequential(nn.Conv2d(out_channels,
            out_channels, 1), nn.BatchNorm2d(out_channels))
        self.conv_higher_res = nn.Sequential(nn.Conv2d(highter_in_channels,
            out_channels, 1), nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU(True)

    def forward(self, higher_res_feature, lower_res_feature):
        _, _, h, w = higher_res_feature.size()
        lower_res_feature = F.interpolate(lower_res_feature, size=(h, w),
            mode='bilinear', align_corners=True)
        lower_res_feature = self.dwconv(lower_res_feature)
        lower_res_feature = self.conv_lower_res(lower_res_feature)
        higher_res_feature = self.conv_higher_res(higher_res_feature)
        out = higher_res_feature + lower_res_feature
        return self.relu(out)


class Classifer(nn.Module):

    def __init__(self, dw_channels, num_classes, stride=1, **kwargs):
        super(Classifer, self).__init__()
        self.dsconv1 = DepthSepConv(dw_channels, dw_channels, stride)
        self.dsconv2 = DepthSepConv(dw_channels, dw_channels, stride)
        self.conv = nn.Sequential(nn.Dropout(0.1), nn.Conv2d(dw_channels,
            num_classes, 1))

    def forward(self, x):
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        x = self.conv(x)
        return x


class ContextNet(nn.Module):

    def __init__(self, classes, aux=False, **kwargs):
        super(ContextNet, self).__init__()
        self.aux = aux
        self.spatial_detail = Shallow_net(32, 64, 128)
        self.context_feature_extractor = Deep_net(32, [32, 32, 48, 64, 96, 
            128], [1, 6, 6, 6, 6, 6], [1, 1, 3, 3, 2, 2])
        self.feature_fusion = FeatureFusionModule(128, 128, 128)
        self.classifier = Classifer(128, classes)
        if self.aux:
            self.auxlayer = nn.Sequential(nn.Conv2d(128, 32, 3, padding=1,
                bias=False), nn.BatchNorm2d(32), nn.ReLU(True), nn.Dropout(
                0.1), nn.Conv2d(32, classes, 1))

    def forward(self, x):
        size = x.size()[2:]
        higher_res_features = self.spatial_detail(x)
        x_low = F.interpolate(x, scale_factor=0.25, mode='bilinear',
            align_corners=True)
        x = self.context_feature_extractor(x_low)
        x = self.feature_fusion(higher_res_features, x)
        x = self.classifier(x)
        outputs = []
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        outputs.append(x)
        if self.aux:
            auxout = self.auxlayer(higher_res_features)
            auxout = F.interpolate(auxout, size, mode='bilinear',
                align_corners=True)
            outputs.append(auxout)
        return x


class Conv(nn.Module):

    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1),
        groups=1, bn_acti=False, bias=False):
        super().__init__()
        self.bn_acti = bn_acti
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=bias)
        if self.bn_acti:
            self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv(input)
        if self.bn_acti:
            output = self.bn_prelu(output)
        return output


class BNPReLU(nn.Module):

    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=0.001)
        self.acti = nn.PReLU(nIn)

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)
        return output


class DABModule(nn.Module):

    def __init__(self, nIn, d=1, kSize=3, dkSize=3):
        super().__init__()
        self.bn_relu_1 = BNPReLU(nIn)
        self.conv3x3 = Conv(nIn, nIn // 2, kSize, 1, padding=1, bn_acti=True)
        self.dconv3x1 = Conv(nIn // 2, nIn // 2, (dkSize, 1), 1, padding=(1,
            0), groups=nIn // 2, bn_acti=True)
        self.dconv1x3 = Conv(nIn // 2, nIn // 2, (1, dkSize), 1, padding=(0,
            1), groups=nIn // 2, bn_acti=True)
        self.ddconv3x1 = Conv(nIn // 2, nIn // 2, (dkSize, 1), 1, padding=(
            1 * d, 0), dilation=(d, 1), groups=nIn // 2, bn_acti=True)
        self.ddconv1x3 = Conv(nIn // 2, nIn // 2, (1, dkSize), 1, padding=(
            0, 1 * d), dilation=(1, d), groups=nIn // 2, bn_acti=True)
        self.bn_relu_2 = BNPReLU(nIn // 2)
        self.conv1x1 = Conv(nIn // 2, nIn, 1, 1, padding=0, bn_acti=False)

    def forward(self, input):
        output = self.bn_relu_1(input)
        output = self.conv3x3(output)
        br1 = self.dconv3x1(output)
        br1 = self.dconv1x3(br1)
        br2 = self.ddconv3x1(output)
        br2 = self.ddconv1x3(br2)
        output = br1 + br2
        output = self.bn_relu_2(output)
        output = self.conv1x1(output)
        return output + input


class DownSamplingBlock(nn.Module):

    def __init__(self, nIn, nOut):
        super().__init__()
        self.nIn = nIn
        self.nOut = nOut
        if self.nIn < self.nOut:
            nConv = nOut - nIn
        else:
            nConv = nOut
        self.conv3x3 = Conv(nIn, nConv, kSize=3, stride=2, padding=1)
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv3x3(input)
        if self.nIn < self.nOut:
            max_pool = self.max_pool(input)
            output = torch.cat([output, max_pool], 1)
        output = self.bn_prelu(output)
        return output


class InputInjection(nn.Module):

    def __init__(self, ratio):
        super().__init__()
        self.pool = nn.ModuleList()
        for i in range(0, ratio):
            self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, input):
        for pool in self.pool:
            input = pool(input)
        return input


class DABNet(nn.Module):

    def __init__(self, classes=19, block_1=3, block_2=6):
        super().__init__()
        self.init_conv = nn.Sequential(Conv(3, 32, 3, 2, padding=1, bn_acti
            =True), Conv(32, 32, 3, 1, padding=1, bn_acti=True), Conv(32, 
            32, 3, 1, padding=1, bn_acti=True))
        self.down_1 = InputInjection(1)
        self.down_2 = InputInjection(2)
        self.down_3 = InputInjection(3)
        self.bn_prelu_1 = BNPReLU(32 + 3)
        self.downsample_1 = DownSamplingBlock(32 + 3, 64)
        self.DAB_Block_1 = nn.Sequential()
        for i in range(0, block_1):
            self.DAB_Block_1.add_module('DAB_Module_1_' + str(i), DABModule
                (64, d=2))
        self.bn_prelu_2 = BNPReLU(128 + 3)
        dilation_block_2 = [4, 4, 8, 8, 16, 16]
        self.downsample_2 = DownSamplingBlock(128 + 3, 128)
        self.DAB_Block_2 = nn.Sequential()
        for i in range(0, block_2):
            self.DAB_Block_2.add_module('DAB_Module_2_' + str(i), DABModule
                (128, d=dilation_block_2[i]))
        self.bn_prelu_3 = BNPReLU(256 + 3)
        self.classifier = nn.Sequential(Conv(259, classes, 1, 1, padding=0))

    def forward(self, input):
        output0 = self.init_conv(input)
        down_1 = self.down_1(input)
        down_2 = self.down_2(input)
        down_3 = self.down_3(input)
        output0_cat = self.bn_prelu_1(torch.cat([output0, down_1], 1))
        output1_0 = self.downsample_1(output0_cat)
        output1 = self.DAB_Block_1(output1_0)
        output1_cat = self.bn_prelu_2(torch.cat([output1, output1_0, down_2
            ], 1))
        output2_0 = self.downsample_2(output1_cat)
        output2 = self.DAB_Block_2(output2_0)
        output2_cat = self.bn_prelu_3(torch.cat([output2, output2_0, down_3
            ], 1))
        out = self.classifier(output2_cat)
        out = F.interpolate(out, input.size()[2:], mode='bilinear',
            align_corners=False)
        return out


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


class EDAModule(nn.Module):

    def __init__(self, ninput, dilated, k=40, dropprob=0.02):
        super().__init__()
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


class EDANetBlock(nn.Module):

    def __init__(self, in_channels, num_dense_layer, dilated, growth_rate):
        """
        :param in_channels: input channel size
        :param num_dense_layer: the number of RDB layers
        :param growth_rate: growth_rate
        """
        super().__init__()
        _in_channels = in_channels
        modules = []
        for i in range(num_dense_layer):
            modules.append(EDAModule(_in_channels, dilated[i], growth_rate))
            _in_channels += growth_rate
        self.residual_dense_layers = nn.Sequential(*modules)

    def forward(self, x):
        out = self.residual_dense_layers(x)
        return out


class EDANet(nn.Module):

    def __init__(self, classes=19):
        super(EDANet, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(DownsamplerBlock(3, 15))
        self.layers.append(DownsamplerBlock(15, 60))
        self.layers.append(EDANetBlock(60, 5, [1, 1, 1, 2, 2], 40))
        self.layers.append(DownsamplerBlock(260, 130))
        self.layers.append(EDANetBlock(130, 8, [2, 2, 4, 4, 8, 8, 16, 16], 40))
        self.project_layer = nn.Conv2d(450, classes, kernel_size=1)
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
        output = F.interpolate(output, scale_factor=8, mode='bilinear',
            align_corners=True)
        return output


class InitialBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding=0,
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

    def forward(self, input):
        main = self.main_branch(input)
        ext = self.ext_branch(input)
        out = torch.cat((main, ext), dim=1)
        out = self.batch_norm(out)
        return self.out_prelu(out)


class RegularBottleneck(nn.Module):

    def __init__(self, channels, internal_ratio=4, kernel_size=3, padding=0,
        dilation=1, asymmetric=False, dropout_prob=0.0, bias=False, relu=True):
        super(RegularBottleneck, self).__init__()
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
        self.ext_regu1 = nn.Dropout2d(p=dropout_prob)
        self.out_prelu = activation

    def forward(self, input):
        main = input
        ext = self.ext_conv1(input)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regu1(ext)
        out = main + ext
        return self.out_prelu(out)


class DownsamplingBottleneck(nn.Module):

    def __init__(self, in_channels, out_channels, internal_ratio=4,
        kernel_size=3, padding=0, return_indices=False, dropout_prob=0.0,
        bias=False, relu=True):
        super().__init__()
        self.return_indices = return_indices
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
        padding = torch.zeros(n, ch_ext - ch_main, h, w)
        if main.is_cuda:
            padding = padding
        main = torch.cat((main, padding), 1)
        out = main + ext
        return self.out_prelu(out), max_indices


class UpsamplingBottleneck(nn.Module):

    def __init__(self, in_channels, out_channels, internal_ratio=4,
        kernel_size=3, padding=0, dropout_prob=0.0, bias=False, relu=True):
        super().__init__()
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


class ENet(nn.Module):

    def __init__(self, classes, encoder_relu=False, decoder_relu=True):
        super().__init__()
        self.name = 'BaseLine_ENet_trans'
        self.initial_block = InitialBlock(3, 16, kernel_size=3, padding=1,
            relu=encoder_relu)
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
        self.transposed_conv = nn.ConvTranspose2d(16, classes, kernel_size=
            3, stride=2, padding=1, output_padding=1, bias=False)
        self.project_layer = nn.Conv2d(128, classes, 1, bias=False)

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
        super().__init__()
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
        super().__init__()
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
        super().__init__()
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
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2,
            padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=0.001)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)


class Decoder(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
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


class ERFNet(nn.Module):

    def __init__(self, classes, encoder=None):
        super().__init__()
        if encoder == None:
            self.encoder = Encoder(classes)
        else:
            self.encoder = encoder
        self.decoder = Decoder(classes)

    def forward(self, input, only_encode=False):
        if only_encode:
            return self.encoder.forward(input, predict=True)
        else:
            output = self.encoder(input)
            return self.decoder.forward(output)


class DownsamplerBlock(nn.Module):

    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.Conv2d(ninput, noutput - ninput, (3, 3), stride=2,
            padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=0.001)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        x1 = self.pool(input)
        x2 = self.conv(input)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY -
            diffY // 2])
        output = torch.cat([x2, x1], 1)
        output = self.bn(output)
        output = self.relu(output)
        return output


class UpsamplerBlock(nn.Module):

    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2,
            padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=0.001)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)


class FCU(nn.Module):

    def __init__(self, chann, kernel_size, dropprob, dilated):
        """
        Factorized Convolution Unit

        """
        super(FCU, self).__init__()
        padding = int((kernel_size - 1) // 2) * dilated
        self.conv3x1_1 = nn.Conv2d(chann, chann, (kernel_size, 1), stride=1,
            padding=(int((kernel_size - 1) // 2) * 1, 0), bias=True)
        self.conv1x3_1 = nn.Conv2d(chann, chann, (1, kernel_size), stride=1,
            padding=(0, int((kernel_size - 1) // 2) * 1), bias=True)
        self.bn1 = nn.BatchNorm2d(chann, eps=0.001)
        self.conv3x1_2 = nn.Conv2d(chann, chann, (kernel_size, 1), stride=1,
            padding=(padding, 0), bias=True, dilation=(dilated, 1))
        self.conv1x3_2 = nn.Conv2d(chann, chann, (1, kernel_size), stride=1,
            padding=(0, padding), bias=True, dilation=(1, dilated))
        self.bn2 = nn.BatchNorm2d(chann, eps=0.001)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, input):
        residual = input
        output = self.conv3x1_1(input)
        output = self.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = self.relu(output)
        output = self.conv3x1_2(output)
        output = self.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)
        if self.dropout.p != 0:
            output = self.dropout(output)
        return F.relu(residual + output, inplace=True)


class PFCU(nn.Module):

    def __init__(self, chann):
        """
        Parallel Factorized Convolution Unit

        """
        super(PFCU, self).__init__()
        self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=
            (1, 0), bias=True)
        self.conv1x3_1 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=
            (0, 1), bias=True)
        self.bn1 = nn.BatchNorm2d(chann, eps=0.001)
        self.conv3x1_22 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding
            =(2, 0), bias=True, dilation=(2, 1))
        self.conv1x3_22 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding
            =(0, 2), bias=True, dilation=(1, 2))
        self.conv3x1_25 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding
            =(5, 0), bias=True, dilation=(5, 1))
        self.conv1x3_25 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding
            =(0, 5), bias=True, dilation=(1, 5))
        self.conv3x1_29 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding
            =(9, 0), bias=True, dilation=(9, 1))
        self.conv1x3_29 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding
            =(0, 9), bias=True, dilation=(1, 9))
        self.bn2 = nn.BatchNorm2d(chann, eps=0.001)
        self.dropout = nn.Dropout2d(0.3)

    def forward(self, input):
        residual = input
        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)
        output2 = self.conv3x1_22(output)
        output2 = F.relu(output2)
        output2 = self.conv1x3_22(output2)
        output2 = self.bn2(output2)
        if self.dropout.p != 0:
            output2 = self.dropout(output2)
        output5 = self.conv3x1_25(output)
        output5 = F.relu(output5)
        output5 = self.conv1x3_25(output5)
        output5 = self.bn2(output5)
        if self.dropout.p != 0:
            output5 = self.dropout(output5)
        output9 = self.conv3x1_29(output)
        output9 = F.relu(output9)
        output9 = self.conv1x3_29(output9)
        output9 = self.bn2(output9)
        if self.dropout.p != 0:
            output9 = self.dropout(output9)
        return F.relu(residual + output2 + output5 + output9, inplace=True)


class ESNet(nn.Module):

    def __init__(self, classes):
        super().__init__()
        self.initial_block = DownsamplerBlock(3, 16)
        self.layers = nn.ModuleList()
        for x in range(0, 3):
            self.layers.append(FCU(16, 3, 0.03, 1))
        self.layers.append(DownsamplerBlock(16, 64))
        for x in range(0, 2):
            self.layers.append(FCU(64, 5, 0.03, 1))
        self.layers.append(DownsamplerBlock(64, 128))
        for x in range(0, 3):
            self.layers.append(PFCU(chann=128))
        self.layers.append(UpsamplerBlock(128, 64))
        self.layers.append(FCU(64, 5, 0, 1))
        self.layers.append(FCU(64, 5, 0, 1))
        self.layers.append(UpsamplerBlock(64, 16))
        self.layers.append(FCU(16, 3, 0, 1))
        self.layers.append(FCU(16, 3, 0, 1))
        self.output_conv = nn.ConvTranspose2d(16, classes, 2, stride=2,
            padding=0, output_padding=0, bias=True)

    def forward(self, input):
        output = self.initial_block(input)
        for layer in self.layers:
            output = layer(output)
        output = self.output_conv(output)
        return output


class CBR(nn.Module):
    """
    This class defines the convolution layer with batch normalization and PReLU activation
    """

    def __init__(self, nIn, nOut, kSize, stride=1):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride,
            padding=(padding, padding), bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=0.001)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        """
        :param input: input feature map
        :return: transformed feature map
        """
        output = self.conv(input)
        output = self.bn(output)
        output = self.act(output)
        return output


class BR(nn.Module):
    """
        This class groups the batch normalization and PReLU activation
    """

    def __init__(self, nOut):
        """
        :param nOut: output feature maps
        """
        super().__init__()
        self.bn = nn.BatchNorm2d(nOut, eps=0.001)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        """
        :param input: input feature map
        :return: normalized and thresholded feature map
        """
        output = self.bn(input)
        output = self.act(output)
        return output


class CB(nn.Module):
    """
       This class groups the convolution and batch normalization
    """

    def __init__(self, nIn, nOut, kSize, stride=1):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optinal stide for down-sampling
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride,
            padding=(padding, padding), bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=0.001)

    def forward(self, input):
        """
        :param input: input feature map
        :return: transformed feature map
        """
        output = self.conv(input)
        output = self.bn(output)
        return output


class C(nn.Module):
    """
    This class is for a convolutional layer.
    """

    def __init__(self, nIn, nOut, kSize, stride=1):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride,
            padding=(padding, padding), bias=False)

    def forward(self, input):
        """
        :param input: input feature map
        :return: transformed feature map
        """
        output = self.conv(input)
        return output


class CDilated(nn.Module):
    """
    This class defines the dilated convolution, which can maintain feature map size
    """

    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        """
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride,
            padding=(padding, padding), bias=False, dilation=d)

    def forward(self, input):
        """
        :param input: input feature map
        :return: transformed feature map
        """
        output = self.conv(input)
        return output


class DownSamplerB(nn.Module):

    def __init__(self, nIn, nOut):
        super().__init__()
        n = int(nOut / 5)
        n1 = nOut - 4 * n
        self.c1 = C(nIn, n, 3, 2)
        self.d1 = CDilated(n, n1, 3, 1, 1)
        self.d2 = CDilated(n, n, 3, 1, 2)
        self.d4 = CDilated(n, n, 3, 1, 4)
        self.d8 = CDilated(n, n, 3, 1, 8)
        self.d16 = CDilated(n, n, 3, 1, 16)
        self.bn = nn.BatchNorm2d(nOut, eps=0.001)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        output1 = self.c1(input)
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        d16 = self.d16(output1)
        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16
        combine = torch.cat([d1, add1, add2, add3, add4], 1)
        output = self.bn(combine)
        output = self.act(output)
        return output


class DilatedParllelResidualBlockB(nn.Module):
    """
    This class defines the ESP block, which is based on the following principle
        Reduce ---> Split ---> Transform --> Merge
    """

    def __init__(self, nIn, nOut, add=True):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param add: if true, add a residual connection through identity operation. You can use projection too as
                in ResNet paper, but we avoid to use it if the dimensions are not the same because we do not want to
                increase the module complexity
        """
        super().__init__()
        n = int(nOut / 5)
        n1 = nOut - 4 * n
        self.c1 = C(nIn, n, 1, 1)
        self.d1 = CDilated(n, n1, 3, 1, 1)
        self.d2 = CDilated(n, n, 3, 1, 2)
        self.d4 = CDilated(n, n, 3, 1, 4)
        self.d8 = CDilated(n, n, 3, 1, 8)
        self.d16 = CDilated(n, n, 3, 1, 16)
        self.bn = BR(nOut)
        self.add = add

    def forward(self, input):
        """
        :param input: input feature map
        :return: transformed feature map
        """
        output1 = self.c1(input)
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        d16 = self.d16(output1)
        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16
        combine = torch.cat([d1, add1, add2, add3, add4], 1)
        if self.add:
            combine = input + combine
        output = self.bn(combine)
        return output


class InputProjectionA(nn.Module):
    """
    This class projects the input image to the same spatial dimensions as the feature map.
    For example, if the input image is 512 x512 x3 and spatial dimensions of feature map size are 56x56xF, then
    this class will generate an output of 56x56x3, for input reinforcement, which establishes a direct link between 
    the input image and encoding stage, improving the flow of information.    
    """

    def __init__(self, samplingTimes):
        """
        :param samplingTimes: The rate at which you want to down-sample the image
        """
        super().__init__()
        self.pool = nn.ModuleList()
        for i in range(0, samplingTimes):
            self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, input):
        """
        :param input: Input RGB Image
        :return: down-sampled image (pyramid-based approach)
        """
        for pool in self.pool:
            input = pool(input)
        return input


class ESPNet_Encoder(nn.Module):
    """
    This class defines the ESPNet-C network in the paper
    """

    def __init__(self, classes=19, p=5, q=3):
        """
        :param classes: number of classes in the dataset. Default is 20 for the cityscapes
        :param p: depth multiplier
        :param q: depth multiplier
        """
        super().__init__()
        self.level1 = CBR(3, 16, 3, 2)
        self.sample1 = InputProjectionA(1)
        self.sample2 = InputProjectionA(2)
        self.b1 = BR(16 + 3)
        self.level2_0 = DownSamplerB(16 + 3, 64)
        self.level2 = nn.ModuleList()
        for i in range(0, p):
            self.level2.append(DilatedParllelResidualBlockB(64, 64))
        self.b2 = BR(128 + 3)
        self.level3_0 = DownSamplerB(128 + 3, 128)
        self.level3 = nn.ModuleList()
        for i in range(0, q):
            self.level3.append(DilatedParllelResidualBlockB(128, 128))
        self.b3 = BR(256)
        self.classifier = C(256, classes, 1, 1)

    def forward(self, input):
        """
        :param input: Receives the input RGB image
        :return: the transformed feature map with spatial dimensions 1/8th of the input image
        """
        output0 = self.level1(input)
        inp1 = self.sample1(input)
        inp2 = self.sample2(input)
        output0_cat = self.b1(torch.cat([output0, inp1], 1))
        output1_0 = self.level2_0(output0_cat)
        for i, layer in enumerate(self.level2):
            if i == 0:
                output1 = layer(output1_0)
            else:
                output1 = layer(output1)
        output1_cat = self.b2(torch.cat([output1, output1_0, inp2], 1))
        output2_0 = self.level3_0(output1_cat)
        for i, layer in enumerate(self.level3):
            if i == 0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)
        output2_cat = self.b3(torch.cat([output2_0, output2], 1))
        classifier = self.classifier(output2_cat)
        out = F.upsample(classifier, input.size()[2:], mode='bilinear')
        return out


class ESPNet(nn.Module):
    """
    This class defines the ESPNet network
    """

    def __init__(self, classes=19, p=2, q=3, encoderFile=None):
        """
        :param classes: number of classes in the dataset. Default is 20 for the cityscapes
        :param p: depth multiplier
        :param q: depth multiplier
        :param encoderFile: pretrained encoder weights. Recall that we first trained the ESPNet-C and then attached the
                            RUM-based light weight decoder. See paper for more details.
        """
        super().__init__()
        self.encoder = ESPNet_Encoder(classes, p, q)
        if encoderFile != None:
            self.encoder.load_state_dict(torch.load(encoderFile))
            None
        self.en_modules = []
        for i, m in enumerate(self.encoder.children()):
            self.en_modules.append(m)
        self.level3_C = C(128 + 3, classes, 1, 1)
        self.br = nn.BatchNorm2d(classes, eps=0.001)
        self.conv = CBR(19 + classes, classes, 3, 1)
        self.up_l3 = nn.Sequential(nn.ConvTranspose2d(classes, classes, 2,
            stride=2, padding=0, output_padding=0, bias=False))
        self.combine_l2_l3 = nn.Sequential(BR(2 * classes),
            DilatedParllelResidualBlockB(2 * classes, classes, add=False))
        self.up_l2 = nn.Sequential(nn.ConvTranspose2d(classes, classes, 2,
            stride=2, padding=0, output_padding=0, bias=False), BR(classes))
        self.classifier = nn.ConvTranspose2d(classes, classes, 2, stride=2,
            padding=0, output_padding=0, bias=False)

    def forward(self, input):
        """
        :param input: RGB image
        :return: transformed feature map
        """
        output0 = self.en_modules[0](input)
        inp1 = self.en_modules[1](input)
        inp2 = self.en_modules[2](input)
        output0_cat = self.en_modules[3](torch.cat([output0, inp1], 1))
        output1_0 = self.en_modules[4](output0_cat)
        for i, layer in enumerate(self.en_modules[5]):
            if i == 0:
                output1 = layer(output1_0)
            else:
                output1 = layer(output1)
        output1_cat = self.en_modules[6](torch.cat([output1, output1_0,
            inp2], 1))
        output2_0 = self.en_modules[7](output1_cat)
        for i, layer in enumerate(self.en_modules[8]):
            if i == 0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)
        output2_cat = self.en_modules[9](torch.cat([output2_0, output2], 1))
        output2_c = self.up_l3(self.br(self.en_modules[10](output2_cat)))
        output1_C = self.level3_C(output1_cat)
        comb_l2_l3 = self.up_l2(self.combine_l2_l3(torch.cat([output1_C,
            output2_c], 1)))
        concat_features = self.conv(torch.cat([comb_l2_l3, output0_cat], 1))
        classifier = self.classifier(concat_features)
        return classifier


class EESP(nn.Module):
    """
    This class defines the EESP block, which is based on the following principle
        REDUCE ---> SPLIT ---> TRANSFORM --> MERGE
    """

    def __init__(self, nIn, nOut, stride=1, k=4, r_lim=7, down_method='esp'):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param stride: factor by which we should skip (useful for down-sampling). If 2, then down-samples the feature map by 2
        :param k: # of parallel branches
        :param r_lim: A maximum value of receptive field allowed for EESP block
        :param g: number of groups to be used in the feature map reduction step.
        """
        super().__init__()
        self.stride = stride
        n = int(nOut / k)
        n1 = nOut - (k - 1) * n
        assert down_method in ['avg', 'esp'
            ], 'One of these is suppported (avg or esp)'
        assert n == n1, 'n(={}) and n1(={}) should be equal for Depth-wise Convolution '.format(
            n, n1)
        self.proj_1x1 = CBR(nIn, n, 1, stride=1, groups=k)
        map_receptive_ksize = {(3): 1, (5): 2, (7): 3, (9): 4, (11): 5, (13
            ): 6, (15): 7, (17): 8}
        self.k_sizes = list()
        for i in range(k):
            ksize = int(3 + 2 * i)
            ksize = ksize if ksize <= r_lim else 3
            self.k_sizes.append(ksize)
        self.k_sizes.sort()
        self.spp_dw = nn.ModuleList()
        for i in range(k):
            d_rate = map_receptive_ksize[self.k_sizes[i]]
            self.spp_dw.append(CDilated(n, n, kSize=3, stride=stride,
                groups=n, d=d_rate))
        self.conv_1x1_exp = CB(nOut, nOut, 1, 1, groups=k)
        self.br_after_cat = BR(nOut)
        self.module_act = nn.PReLU(nOut)
        self.downAvg = True if down_method == 'avg' else False

    def forward(self, input):
        """
        :param input: input feature map
        :return: transformed feature map
        """
        output1 = self.proj_1x1(input)
        output = [self.spp_dw[0](output1)]
        for k in range(1, len(self.spp_dw)):
            out_k = self.spp_dw[k](output1)
            out_k = out_k + output[k - 1]
            output.append(out_k)
        expanded = self.conv_1x1_exp(self.br_after_cat(torch.cat(output, 1)))
        del output
        if self.stride == 2 and self.downAvg:
            return expanded
        if expanded.size() == input.size():
            expanded = expanded + input
        return self.module_act(expanded)


class DownSampler(nn.Module):
    """
    Down-sampling fucntion that has two parallel branches: (1) avg pooling
    and (2) EESP block with stride of 2. The output feature maps of these branches
    are then concatenated and thresholded using an activation function (PReLU in our
    case) to produce the final output.
    """

    def __init__(self, nin, nout, k=4, r_lim=9, reinf=True):
        """
            :param nin: number of input channels
            :param nout: number of output channels
            :param k: # of parallel branches
            :param r_lim: A maximum value of receptive field allowed for EESP block
            :param g: number of groups to be used in the feature map reduction step.
        """
        super().__init__()
        nout_new = nout - nin
        self.eesp = EESP(nin, nout_new, stride=2, k=k, r_lim=r_lim,
            down_method='avg')
        self.avg = nn.AvgPool2d(kernel_size=3, padding=1, stride=2)
        if reinf:
            self.inp_reinf = nn.Sequential(CBR(config_inp_reinf,
                config_inp_reinf, 3, 1), CB(config_inp_reinf, nout, 1, 1))
        self.act = nn.PReLU(nout)

    def forward(self, input, input2=None):
        """
        :param input: input feature map
        :return: feature map down-sampled by a factor of 2
        """
        avg_out = self.avg(input)
        eesp_out = self.eesp(input)
        output = torch.cat([avg_out, eesp_out], 1)
        if input2 is not None:
            w1 = avg_out.size(2)
            while True:
                input2 = F.avg_pool2d(input2, kernel_size=3, padding=1,
                    stride=2)
                w2 = input2.size(2)
                if w2 == w1:
                    break
            output = output + self.inp_reinf(input2)
        return self.act(output)


class EESPNet(nn.Module):
    """
    This class defines the ESPNetv2 architecture for the ImageNet classification
    """

    def __init__(self, classes=19, s=1):
        """
        :param classes: number of classes in the dataset. Default is 20 for the cityscapes
        :param s: factor that scales the number of output feature maps
        """
        super().__init__()
        reps = [0, 3, 7, 3]
        channels = 3
        r_lim = [13, 11, 9, 7, 5]
        K = [4] * len(r_lim)
        base = 32
        config_len = 5
        config = [base] * config_len
        base_s = 0
        for i in range(config_len):
            if i == 0:
                base_s = int(base * s)
                base_s = math.ceil(base_s / K[0]) * K[0]
                config[i] = base if base_s > base else base_s
            else:
                config[i] = base_s * pow(2, i)
        if s <= 1.5:
            config.append(1024)
        elif s in [1.5, 2]:
            config.append(1280)
        else:
            ValueError('Configuration not supported')
        global config_inp_reinf
        config_inp_reinf = 3
        self.input_reinforcement = True
        assert len(K) == len(r_lim
            ), 'Length of branching factor array and receptive field array should be the same.'
        self.level1 = CBR(channels, config[0], 3, 2)
        self.level2_0 = DownSampler(config[0], config[1], k=K[0], r_lim=
            r_lim[0], reinf=self.input_reinforcement)
        self.level3_0 = DownSampler(config[1], config[2], k=K[1], r_lim=
            r_lim[1], reinf=self.input_reinforcement)
        self.level3 = nn.ModuleList()
        for i in range(reps[1]):
            self.level3.append(EESP(config[2], config[2], stride=1, k=K[2],
                r_lim=r_lim[2]))
        self.level4_0 = DownSampler(config[2], config[3], k=K[2], r_lim=
            r_lim[2], reinf=self.input_reinforcement)
        self.level4 = nn.ModuleList()
        for i in range(reps[2]):
            self.level4.append(EESP(config[3], config[3], stride=1, k=K[3],
                r_lim=r_lim[3]))
        self.level5_0 = DownSampler(config[3], config[4], k=K[3], r_lim=
            r_lim[3])
        self.level5 = nn.ModuleList()
        for i in range(reps[3]):
            self.level5.append(EESP(config[4], config[4], stride=1, k=K[4],
                r_lim=r_lim[4]))
        self.level5.append(CBR(config[4], config[4], 3, 1, groups=config[4]))
        self.level5.append(CBR(config[4], config[5], 1, 1, groups=K[4]))
        self.classifier = nn.Linear(config[5], classes)
        self.init_params()

    def init_params(self):
        """
        Function to initialze the parameters
        """
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

    def forward(self, input, p=0.2, seg=True):
        """
        :param input: Receives the input RGB image
        :return: a C-dimensional vector, C=# of classes
        """
        out_l1 = self.level1(input)
        if not self.input_reinforcement:
            del input
            input = None
        out_l2 = self.level2_0(out_l1, input)
        out_l3_0 = self.level3_0(out_l2, input)
        for i, layer in enumerate(self.level3):
            if i == 0:
                out_l3 = layer(out_l3_0)
            else:
                out_l3 = layer(out_l3)
        out_l4_0 = self.level4_0(out_l3, input)
        for i, layer in enumerate(self.level4):
            if i == 0:
                out_l4 = layer(out_l4_0)
            else:
                out_l4 = layer(out_l4)
        if not seg:
            out_l5_0 = self.level5_0(out_l4)
            for i, layer in enumerate(self.level5):
                if i == 0:
                    out_l5 = layer(out_l5_0)
                else:
                    out_l5 = layer(out_l5)
            output_g = F.adaptive_avg_pool2d(out_l5, output_size=1)
            output_g = F.dropout(output_g, p=p, training=self.training)
            output_1x1 = output_g.view(output_g.size(0), -1)
            return self.classifier(output_1x1)
        return out_l1, out_l2, out_l3, out_l4


class EESPNet_Seg(nn.Module):

    def __init__(self, classes=19, s=2, pretrained=None, gpus=1):
        super().__init__()
        classificationNet = EESPNet(classes=1000, s=s)
        if gpus >= 1:
            classificationNet = nn.DataParallel(classificationNet)
        if pretrained:
            if not os.path.isfile(pretrained):
                None
            None
            classificationNet.load_state_dict(torch.load(pretrained))
        self.net = classificationNet.module
        del classificationNet
        del self.net.classifier
        del self.net.level5
        del self.net.level5_0
        if s <= 0.5:
            p = 0.1
        else:
            p = 0.2
        self.proj_L4_C = CBR(self.net.level4[-1].module_act.num_parameters,
            self.net.level3[-1].module_act.num_parameters, 1, 1)
        pspSize = 2 * self.net.level3[-1].module_act.num_parameters
        self.pspMod = nn.Sequential(EESP(pspSize, pspSize // 2, stride=1, k
            =4, r_lim=7), PSPModule(pspSize // 2, pspSize // 2))
        self.project_l3 = nn.Sequential(nn.Dropout2d(p=p), C(pspSize // 2,
            classes, 1, 1))
        self.act_l3 = BR(classes)
        self.project_l2 = CBR(self.net.level2_0.act.num_parameters +
            classes, classes, 1, 1)
        self.project_l1 = nn.Sequential(nn.Dropout2d(p=p), C(self.net.
            level1.act.num_parameters + classes, classes, 1, 1))

    def hierarchicalUpsample(self, x, factor=3):
        for i in range(factor):
            x = F.interpolate(x, scale_factor=2, mode='bilinear',
                align_corners=True)
        return x

    def forward(self, input):
        out_l1, out_l2, out_l3, out_l4 = self.net(input, seg=True)
        out_l4_proj = self.proj_L4_C(out_l4)
        up_l4_to_l3 = F.interpolate(out_l4_proj, size=out_l3.size()[2:],
            mode='bilinear', align_corners=True)
        merged_l3_upl4 = self.pspMod(torch.cat([out_l3, up_l4_to_l3], 1))
        proj_merge_l3_bef_act = self.project_l3(merged_l3_upl4)
        proj_merge_l3 = self.act_l3(proj_merge_l3_bef_act)
        out_up_l3 = F.interpolate(proj_merge_l3, scale_factor=2, mode=
            'bilinear', align_corners=True)
        merge_l2 = self.project_l2(torch.cat([out_l2, out_up_l3], 1))
        out_up_l2 = F.interpolate(merge_l2, scale_factor=2, mode='bilinear',
            align_corners=True)
        merge_l1 = self.project_l1(torch.cat([out_l1, out_up_l2], 1))
        output = F.interpolate(merge_l1, scale_factor=2, mode='bilinear',
            align_corners=True)
        return output


class PSPModule(nn.Module):

    def __init__(self, features, out_features=1024, sizes=(1, 2, 4, 8)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([C(features, features, 3, 1, groups=
            features) for size in sizes])
        self.project = CBR(features * (len(sizes) + 1), out_features, 1, 1)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        out = [feats]
        for stage in self.stages:
            feats = F.avg_pool2d(feats, kernel_size=3, stride=2, padding=1)
            upsampled = F.interpolate(input=stage(feats), size=(h, w), mode
                ='bilinear', align_corners=True)
            out.append(upsampled)
        return self.project(torch.cat(out, dim=1))


class CBR(nn.Module):
    """
    This class defines the convolution layer with batch normalization and PReLU activation
    """

    def __init__(self, nIn, nOut, kSize, stride=1, groups=1):
        """

        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=
            padding, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(nOut)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        """
        :param input: input feature map
        :return: transformed feature map
        """
        output = self.conv(input)
        output = self.bn(output)
        output = self.act(output)
        return output


class BR(nn.Module):
    """
        This class groups the batch normalization and PReLU activation
    """

    def __init__(self, nOut):
        """
        :param nOut: output feature maps
        """
        super().__init__()
        self.bn = nn.BatchNorm2d(nOut)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        """
        :param input: input feature map
        :return: normalized and thresholded feature map
        """
        output = self.bn(input)
        output = self.act(output)
        return output


class CB(nn.Module):
    """
       This class groups the convolution and batch normalization
    """

    def __init__(self, nIn, nOut, kSize, stride=1, groups=1):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optinal stide for down-sampling
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=
            padding, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(nOut)

    def forward(self, input):
        """

        :param input: input feature map
        :return: transformed feature map
        """
        output = self.conv(input)
        output = self.bn(output)
        return output


class C(nn.Module):
    """
    This class is for a convolutional layer.
    """

    def __init__(self, nIn, nOut, kSize, stride=1, groups=1):
        """

        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=
            padding, bias=False, groups=groups)

    def forward(self, input):
        """
        :param input: input feature map
        :return: transformed feature map
        """
        output = self.conv(input)
        return output


class CDilated(nn.Module):
    """
    This class defines the dilated convolution.
    """

    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        """
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=
            padding, bias=False, dilation=d, groups=groups)

    def forward(self, input):
        """
        :param input: input feature map
        :return: transformed feature map
        """
        output = self.conv(input)
        return output


class CDilatedB(nn.Module):
    """
    This class defines the dilated convolution with batch normalization.
    """

    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        """
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=
            padding, bias=False, dilation=d, groups=groups)
        self.bn = nn.BatchNorm2d(nOut)

    def forward(self, input):
        """
        :param input: input feature map
        :return: transformed feature map
        """
        return self.bn(self.conv(input))


class SEModule(nn.Module):

    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
            padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
            padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.avg_pool(input)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return input * x


def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1, groups=
    1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
        padding=padding, dilation=dilation, groups=groups, bias=bias)


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
        bias=bias)


class FPEBlock(nn.Module):

    def __init__(self, inplanes, outplanes, dilat, downsample=None, stride=
        1, t=1, scales=4, se=False, norm_layer=None):
        super(FPEBlock, self).__init__()
        if inplanes % scales != 0:
            raise ValueError('Planes must be divisible by scales')
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        bottleneck_planes = inplanes * t
        self.conv1 = conv1x1(inplanes, bottleneck_planes, stride)
        self.bn1 = norm_layer(bottleneck_planes)
        self.conv2 = nn.ModuleList([conv3x3(bottleneck_planes // scales, 
            bottleneck_planes // scales, groups=bottleneck_planes // scales,
            dilation=dilat[i], padding=1 * dilat[i]) for i in range(scales)])
        self.bn2 = nn.ModuleList([norm_layer(bottleneck_planes // scales) for
            _ in range(scales)])
        self.conv3 = conv1x1(bottleneck_planes, outplanes)
        self.bn3 = norm_layer(outplanes)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEModule(outplanes) if se else None
        self.downsample = downsample
        self.stride = stride
        self.scales = scales

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        xs = torch.chunk(out, self.scales, 1)
        ys = []
        for s in range(self.scales):
            if s == 0:
                ys.append(self.relu(self.bn2[s](self.conv2[s](xs[s]))))
            else:
                ys.append(self.relu(self.bn2[s](self.conv2[s](xs[s] + ys[-1])))
                    )
        out = torch.cat(ys, 1)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.se is not None:
            out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out += identity
        out = self.relu(out)
        return out


class MEUModule(nn.Module):

    def __init__(self, channels_high, channels_low, channel_out):
        super(MEUModule, self).__init__()
        self.conv1x1_low = nn.Conv2d(channels_low, channel_out, kernel_size
            =1, bias=False)
        self.bn_low = nn.BatchNorm2d(channel_out)
        self.sa_conv = nn.Conv2d(1, 1, kernel_size=1, bias=False)
        self.conv1x1_high = nn.Conv2d(channels_high, channel_out,
            kernel_size=1, bias=False)
        self.bn_high = nn.BatchNorm2d(channel_out)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca_conv = nn.Conv2d(channel_out, channel_out, kernel_size=1,
            bias=False)
        self.sa_sigmoid = nn.Sigmoid()
        self.ca_sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, fms_high, fms_low):
        """
        :param fms_high:  High level Feature map. Tensor.
        :param fms_low: Low level Feature map. Tensor.
        """
        _, _, h, w = fms_low.shape
        fms_low = self.conv1x1_low(fms_low)
        fms_low = self.bn_low(fms_low)
        sa_avg_out = self.sa_sigmoid(self.sa_conv(torch.mean(fms_low, dim=1,
            keepdim=True)))
        fms_high = self.conv1x1_high(fms_high)
        fms_high = self.bn_high(fms_high)
        ca_avg_out = self.ca_sigmoid(self.relu(self.ca_conv(self.avg_pool(
            fms_high))))
        fms_high_up = F.interpolate(fms_high, size=(h, w), mode='bilinear',
            align_corners=True)
        fms_sa_att = sa_avg_out * fms_high_up
        fms_ca_att = ca_avg_out * fms_low
        out = fms_ca_att + fms_sa_att
        return out


class FPENet(nn.Module):

    def __init__(self, classes=19, zero_init_residual=False, width=16,
        scales=4, se=False, norm_layer=None):
        super(FPENet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        outplanes = [int(width * 2 ** i) for i in range(3)]
        self.block_num = [1, 3, 9]
        self.dilation = [1, 2, 4, 8]
        self.inplanes = outplanes[0]
        self.conv1 = nn.Conv2d(3, outplanes[0], kernel_size=3, stride=2,
            padding=1, bias=False)
        self.bn1 = norm_layer(outplanes[0])
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(FPEBlock, outplanes[0], self.
            block_num[0], dilation=self.dilation, stride=1, t=1, scales=
            scales, se=se, norm_layer=norm_layer)
        self.layer2 = self._make_layer(FPEBlock, outplanes[1], self.
            block_num[1], dilation=self.dilation, stride=2, t=4, scales=
            scales, se=se, norm_layer=norm_layer)
        self.layer3 = self._make_layer(FPEBlock, outplanes[2], self.
            block_num[2], dilation=self.dilation, stride=2, t=4, scales=
            scales, se=se, norm_layer=norm_layer)
        self.meu1 = MEUModule(64, 32, 64)
        self.meu2 = MEUModule(64, 16, 32)
        self.project_layer = nn.Conv2d(32, classes, kernel_size=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                    nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, FPEBlock):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, dilation, stride=1, t=1,
        scales=4, se=False, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes,
                stride), norm_layer(planes))
        layers = []
        layers.append(block(self.inplanes, planes, dilat=dilation,
            downsample=downsample, stride=stride, t=t, scales=scales, se=se,
            norm_layer=norm_layer))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilat=dilation,
                scales=scales, se=se, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x_1 = self.layer1(x)
        x_2_0 = self.layer2[0](x_1)
        x_2_1 = self.layer2[1](x_2_0)
        x_2_2 = self.layer2[2](x_2_1)
        x_2 = x_2_0 + x_2_2
        x_3_0 = self.layer3[0](x_2)
        x_3_1 = self.layer3[1](x_3_0)
        x_3_2 = self.layer3[2](x_3_1)
        x_3_3 = self.layer3[3](x_3_2)
        x_3_4 = self.layer3[4](x_3_3)
        x_3_5 = self.layer3[5](x_3_4)
        x_3_6 = self.layer3[6](x_3_5)
        x_3_7 = self.layer3[7](x_3_6)
        x_3_8 = self.layer3[8](x_3_7)
        x_3 = x_3_0 + x_3_8
        x2 = self.meu1(x_3, x_2)
        x1 = self.meu2(x2, x_1)
        output = self.project_layer(x1)
        output = F.interpolate(output, scale_factor=2, mode='bilinear',
            align_corners=True)
        return output


class DownsamplingBottleneck(nn.Module):

    def __init__(self, in_channels, out_channels, internal_ratio=4,
        kernel_size=3, padding=0, dropout_prob=0.0, bias=False, non_linear=
        'ReLU'):
        super().__init__()
        internal_channels = in_channels // internal_ratio
        self.main_max1 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2
            ), nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1,
            bias=bias))
        self.ext_conv1 = nn.Sequential(nn.Conv2d(in_channels,
            internal_channels, kernel_size=2, stride=2, bias=bias), nn.
            BatchNorm2d(internal_channels), NON_LINEARITY[non_linear])
        self.ext_conv2 = nn.Sequential(nn.Conv2d(internal_channels,
            internal_channels, kernel_size=kernel_size, stride=1, padding=
            padding, bias=bias), nn.BatchNorm2d(internal_channels),
            NON_LINEARITY[non_linear])
        self.ext_conv3 = nn.Sequential(nn.Conv2d(internal_channels,
            out_channels, kernel_size=1, stride=1, bias=bias), nn.
            BatchNorm2d(out_channels), NON_LINEARITY[non_linear])
        self.ext_regul = nn.Dropout2d(p=dropout_prob)
        self.out_prelu = NON_LINEARITY[non_linear]

    def forward(self, x):
        main = self.main_max1(x)
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)
        out = self.out_prelu(main + ext)
        return out


class UpsamplingBottleneck(nn.Module):

    def __init__(self, in_channels, out_channels, internal_ratio=4,
        kernel_size=2, padding=0, dropout_prob=0.0, bias=False, non_linear=
        'ReLU'):
        super().__init__()
        internal_channels = in_channels // internal_ratio
        self.main_conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels,
            kernel_size=1, bias=bias), nn.BatchNorm2d(out_channels))
        self.ext_conv1 = nn.Sequential(nn.Conv2d(in_channels,
            internal_channels, kernel_size=1, bias=bias), nn.BatchNorm2d(
            internal_channels), NON_LINEARITY[non_linear])
        self.ext_conv2 = nn.Sequential(nn.ConvTranspose2d(internal_channels,
            internal_channels, kernel_size=kernel_size, stride=2, padding=
            padding, output_padding=0, bias=bias), nn.BatchNorm2d(
            internal_channels), NON_LINEARITY[non_linear])
        self.ext_conv3 = nn.Sequential(nn.Conv2d(internal_channels,
            out_channels, kernel_size=1, bias=bias), nn.BatchNorm2d(
            out_channels), NON_LINEARITY[non_linear])
        self.ext_regul = nn.Dropout2d(p=dropout_prob)
        self.out_prelu = NON_LINEARITY[non_linear]

    def forward(self, x, x_pre):
        main = x + x_pre
        main = self.main_conv1(main)
        main = F.interpolate(main, scale_factor=2, mode='bilinear',
            align_corners=True)
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)
        out = self.out_prelu(main + ext)
        return out


class DilatedBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
        dilation=1, dropout_prob=0.0, bias=False, non_linear='ReLU'):
        super(DilatedBlock, self).__init__()
        self.relu = NON_LINEARITY[non_linear]
        self.internal_channels = in_channels // 4
        self.conv1 = nn.Conv2d(in_channels, self.internal_channels, 1, bias
            =bias)
        self.conv1_bn = nn.BatchNorm2d(self.internal_channels)
        self.conv2 = nn.Conv2d(self.internal_channels, self.
            internal_channels, kernel_size, stride, padding=int((
            kernel_size - 1) / 2 * dilation), dilation=dilation, groups=1,
            bias=bias)
        self.conv2_bn = nn.BatchNorm2d(self.internal_channels)
        self.conv4 = nn.Conv2d(self.internal_channels, out_channels, 1,
            bias=bias)
        self.conv4_bn = nn.BatchNorm2d(out_channels)
        self.regul = nn.Dropout2d(p=dropout_prob)

    def forward(self, x):
        residual = x
        main = self.relu(self.conv1_bn(self.conv1(x)))
        main = self.relu(self.conv2_bn(self.conv2(main)))
        main = self.conv4_bn(self.conv4(main))
        main = self.regul(main)
        out = self.relu(torch.add(main, residual))
        return out


class Factorized_Block(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
        dilation=1, dropout_prob=0.0, bias=False, non_linear='ReLU'):
        super(Factorized_Block, self).__init__()
        self.relu = NON_LINEARITY[non_linear]
        self.internal_channels = in_channels // 4
        self.compress_conv1 = nn.Conv2d(in_channels, self.internal_channels,
            1, padding=0, bias=bias)
        self.conv1_bn = nn.BatchNorm2d(self.internal_channels)
        self.conv2_1 = nn.Conv2d(self.internal_channels, self.
            internal_channels, (kernel_size, 1), stride=(stride, 1),
            padding=(int((kernel_size - 1) / 2 * dilation), 0), dilation=(
            dilation, 1), bias=bias)
        self.conv2_1_bn = nn.BatchNorm2d(self.internal_channels)
        self.conv2_2 = nn.Conv2d(self.internal_channels, self.
            internal_channels, (1, kernel_size), stride=(1, stride),
            padding=(0, int((kernel_size - 1) / 2 * dilation)), dilation=(1,
            dilation), bias=bias)
        self.conv2_2_bn = nn.BatchNorm2d(self.internal_channels)
        self.extend_conv3 = nn.Conv2d(self.internal_channels, out_channels,
            1, padding=0, bias=bias)
        self.conv3_bn = nn.BatchNorm2d(out_channels)
        self.regul = nn.Dropout2d(p=dropout_prob)

    def forward(self, x):
        residual = x
        main = self.relu(self.conv1_bn(self.compress_conv1(x)))
        main = self.relu(self.conv2_1_bn(self.conv2_1(main)))
        main = self.relu(self.conv2_2_bn(self.conv2_2(main)))
        main = self.conv3_bn(self.extend_conv3(main))
        main = self.regul(main)
        out = self.relu(torch.add(residual, main))
        return out


class FSSNet(nn.Module):

    def __init__(self, classes):
        super().__init__()
        self.initial_block = InitialBlock(3, 16)
        self.downsample1_0 = DownsamplingBottleneck(16, 64, padding=1,
            dropout_prob=0.03)
        self.factorized1_1 = Factorized_Block(64, 64, dropout_prob=0.03)
        self.factorized1_2 = Factorized_Block(64, 64, dropout_prob=0.03)
        self.factorized1_3 = Factorized_Block(64, 64, dropout_prob=0.03)
        self.factorized1_4 = Factorized_Block(64, 64, dropout_prob=0.03)
        self.downsample2_0 = DownsamplingBottleneck(64, 128, padding=1,
            dropout_prob=0.3)
        self.dilated2_1 = DilatedBlock(128, 128, dilation=2, dropout_prob=0.3)
        self.dilated2_2 = DilatedBlock(128, 128, dilation=5, dropout_prob=0.3)
        self.dilated2_3 = DilatedBlock(128, 128, dilation=9, dropout_prob=0.3)
        self.dilated2_4 = DilatedBlock(128, 128, dilation=2, dropout_prob=0.3)
        self.dilated2_5 = DilatedBlock(128, 128, dilation=5, dropout_prob=0.3)
        self.dilated2_6 = DilatedBlock(128, 128, dilation=9, dropout_prob=0.3)
        self.upsample4_0 = UpsamplingBottleneck(128, 64, dropout_prob=0.3)
        self.bottleneck4_1 = DilatedBlock(64, 64, dropout_prob=0.3)
        self.bottleneck4_2 = DilatedBlock(64, 64, dropout_prob=0.3)
        self.upsample5_0 = UpsamplingBottleneck(64, 16, dropout_prob=0.3)
        self.bottleneck5_1 = DilatedBlock(16, 16, dropout_prob=0.3)
        self.bottleneck5_2 = DilatedBlock(16, 16, dropout_prob=0.3)
        self.transposed_conv = nn.ConvTranspose2d(16, classes, kernel_size=
            3, stride=2, padding=1, output_padding=1, bias=False)

    def forward(self, x):
        x = self.initial_block(x)
        x_1 = self.downsample1_0(x)
        x = self.factorized1_1(x_1)
        x = self.factorized1_2(x)
        x = self.factorized1_3(x)
        x = self.factorized1_4(x)
        x_2 = self.downsample2_0(x)
        x = self.dilated2_1(x_2)
        x = self.dilated2_2(x)
        x = self.dilated2_3(x)
        x = self.dilated2_4(x)
        x = self.dilated2_5(x)
        x = self.dilated2_6(x)
        x = self.upsample4_0(x, x_2)
        x = self.bottleneck4_1(x)
        x = self.bottleneck4_2(x)
        x = self.upsample5_0(x, x_1)
        x = self.bottleneck5_1(x)
        x = self.bottleneck5_2(x)
        x = self.transposed_conv(x)
        return x


class _ConvBNReLU(nn.Module):
    """Conv-BN-ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
        padding=0, **kwargs):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels,
            kernel_size, stride, padding, bias=False), nn.BatchNorm2d(
            out_channels), nn.ReLU(True))

    def forward(self, x):
        return self.conv(x)


class _DSConv(nn.Module):
    """Depthwise Separable Convolutions"""

    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(_DSConv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(dw_channels, dw_channels, 3,
            stride, 1, groups=dw_channels, bias=False), nn.BatchNorm2d(
            dw_channels), nn.ReLU(True), nn.Conv2d(dw_channels,
            out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.
            ReLU(True))

    def forward(self, x):
        return self.conv(x)


class _DWConv(nn.Module):
    """Depthwise Convolutions"""

    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(_DWConv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(dw_channels, out_channels, 3,
            stride, 1, groups=dw_channels, bias=False), nn.BatchNorm2d(
            out_channels), nn.ReLU(True))

    def forward(self, x):
        return self.conv(x)


class LinearBottleneck(nn.Module):
    """LinearBottleneck used in MobileNetV2"""

    def __init__(self, in_channels, out_channels, t=6, stride=2, **kwargs):
        super(LinearBottleneck, self).__init__()
        self.use_shortcut = stride == 1 and in_channels == out_channels
        self.block = nn.Sequential(_ConvBNReLU(in_channels, in_channels * t,
            1), _DWConv(in_channels * t, in_channels * t, stride), nn.
            Conv2d(in_channels * t, out_channels, 1, bias=False), nn.
            BatchNorm2d(out_channels))

    def forward(self, x):
        out = self.block(x)
        if self.use_shortcut:
            out = x + out
        return out


class PyramidPooling(nn.Module):
    """Pyramid pooling module"""

    def __init__(self, in_channels, out_channels, **kwargs):
        super(PyramidPooling, self).__init__()
        inter_channels = int(in_channels / 4)
        self.conv1 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv2 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv3 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv4 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.out = _ConvBNReLU(in_channels * 2, out_channels, 1)

    def pool(self, x, size):
        avgpool = nn.AdaptiveAvgPool2d(size)
        return avgpool(x)

    def upsample(self, x, size):
        return F.interpolate(x, size, mode='bilinear', align_corners=True)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = self.upsample(self.conv1(self.pool(x, 1)), size)
        feat2 = self.upsample(self.conv2(self.pool(x, 2)), size)
        feat3 = self.upsample(self.conv3(self.pool(x, 3)), size)
        feat4 = self.upsample(self.conv4(self.pool(x, 6)), size)
        x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
        x = self.out(x)
        return x


class LearningToDownsample(nn.Module):
    """Learning to downsample module"""

    def __init__(self, dw_channels1=32, dw_channels2=48, out_channels=64,
        **kwargs):
        super(LearningToDownsample, self).__init__()
        self.conv = _ConvBNReLU(3, dw_channels1, 3, 2)
        self.dsconv1 = _DSConv(dw_channels1, dw_channels2, 2)
        self.dsconv2 = _DSConv(dw_channels2, out_channels, 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        return x


class GlobalFeatureExtractor(nn.Module):
    """Global feature extractor module"""

    def __init__(self, in_channels=64, block_channels=(64, 96, 128),
        out_channels=128, t=6, num_blocks=(3, 3, 3), **kwargs):
        super(GlobalFeatureExtractor, self).__init__()
        self.bottleneck1 = self._make_layer(LinearBottleneck, in_channels,
            block_channels[0], num_blocks[0], t, 2)
        self.bottleneck2 = self._make_layer(LinearBottleneck,
            block_channels[0], block_channels[1], num_blocks[1], t, 2)
        self.bottleneck3 = self._make_layer(LinearBottleneck,
            block_channels[1], block_channels[2], num_blocks[2], t, 1)
        self.ppm = PyramidPooling(block_channels[2], out_channels)

    def _make_layer(self, block, inplanes, planes, blocks, t=6, stride=1):
        layers = []
        layers.append(block(inplanes, planes, t, stride))
        for i in range(1, blocks):
            layers.append(block(planes, planes, t, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.ppm(x)
        return x


class FeatureFusionModule(nn.Module):
    """Feature fusion module"""

    def __init__(self, highter_in_channels, lower_in_channels, out_channels,
        scale_factor=4, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.scale_factor = scale_factor
        self.dwconv = _DWConv(lower_in_channels, out_channels, 1)
        self.conv_lower_res = nn.Sequential(nn.Conv2d(out_channels,
            out_channels, 1), nn.BatchNorm2d(out_channels))
        self.conv_higher_res = nn.Sequential(nn.Conv2d(highter_in_channels,
            out_channels, 1), nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU(True)

    def forward(self, higher_res_feature, lower_res_feature):
        _, _, h, w = higher_res_feature.size()
        lower_res_feature = F.interpolate(lower_res_feature, size=(h, w),
            mode='bilinear', align_corners=True)
        lower_res_feature = self.dwconv(lower_res_feature)
        lower_res_feature = self.conv_lower_res(lower_res_feature)
        higher_res_feature = self.conv_higher_res(higher_res_feature)
        out = higher_res_feature + lower_res_feature
        return self.relu(out)


class Classifer(nn.Module):
    """Classifer"""

    def __init__(self, dw_channels, num_classes, stride=1, **kwargs):
        super(Classifer, self).__init__()
        self.dsconv1 = _DSConv(dw_channels, dw_channels, stride)
        self.dsconv2 = _DSConv(dw_channels, dw_channels, stride)
        self.conv = nn.Sequential(nn.Dropout(0.1), nn.Conv2d(dw_channels,
            num_classes, 1))

    def forward(self, x):
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        x = self.conv(x)
        return x


class FastSCNN(nn.Module):

    def __init__(self, classes, aux=False, **kwargs):
        super(FastSCNN, self).__init__()
        self.aux = aux
        self.learning_to_downsample = LearningToDownsample(32, 48, 64)
        self.global_feature_extractor = GlobalFeatureExtractor(64, [64, 96,
            128], 128, 6, [3, 3, 3])
        self.feature_fusion = FeatureFusionModule(64, 128, 128)
        self.classifier = Classifer(128, classes)
        if self.aux:
            self.auxlayer = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1,
                bias=False), nn.BatchNorm2d(32), nn.ReLU(True), nn.Dropout(
                0.1), nn.Conv2d(32, classes, 1))

    def forward(self, x):
        size = x.size()[2:]
        higher_res_features = self.learning_to_downsample(x)
        x = self.global_feature_extractor(higher_res_features)
        x = self.feature_fusion(higher_res_features, x)
        x = self.classifier(x)
        outputs = []
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        outputs.append(x)
        if self.aux:
            auxout = self.auxlayer(higher_res_features)
            auxout = F.interpolate(auxout, size, mode='bilinear',
                align_corners=True)
            outputs.append(auxout)
        return x


class PermutationBlock(nn.Module):

    def __init__(self, groups):
        super(PermutationBlock, self).__init__()
        self.groups = groups

    def forward(self, input):
        n, c, h, w = input.size()
        G = self.groups
        output = input.view(n, G, c // G, h, w).permute(0, 2, 1, 3, 4
            ).contiguous().view(n, c, h, w)
        return output


class Conv2dBnRelu(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=0,
        dilation=1, bias=True):
        super(Conv2dBnRelu, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size,
            stride, padding, dilation=dilation, bias=bias), nn.BatchNorm2d(
            out_ch, eps=0.001), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


class DownsamplerBlock(nn.Module):

    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.Conv2d(ninput, noutput - ninput, (3, 3), stride=2,
            padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=0.001)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        x1 = self.pool(input)
        x2 = self.conv(input)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY -
            diffY // 2])
        output = torch.cat([x2, x1], 1)
        output = self.bn(output)
        output = self.relu(output)
        return output


def Channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x


def Split(x):
    c = int(x.size()[1])
    c1 = round(c * 0.5)
    x1 = x[:, :c1, :, :].contiguous()
    x2 = x[:, c1:, :, :].contiguous()
    return x1, x2


def Merge(x1, x2):
    return torch.cat((x1, x2), 1)


class SS_nbt_module_paper(nn.Module):

    def __init__(self, chann, dropprob, dilated):
        super().__init__()
        oup_inc = chann // 2
        self.conv3x1_1_l = nn.Conv2d(oup_inc, oup_inc, (3, 1), stride=1,
            padding=(1, 0), bias=True)
        self.conv1x3_1_l = nn.Conv2d(oup_inc, oup_inc, (1, 3), stride=1,
            padding=(0, 1), bias=True)
        self.bn1_l = nn.BatchNorm2d(oup_inc, eps=0.001)
        self.conv3x1_2_l = nn.Conv2d(oup_inc, oup_inc, (3, 1), stride=1,
            padding=(1 * dilated, 0), bias=True, dilation=(dilated, 1))
        self.conv1x3_2_l = nn.Conv2d(oup_inc, oup_inc, (1, 3), stride=1,
            padding=(0, 1 * dilated), bias=True, dilation=(1, dilated))
        self.bn2_l = nn.BatchNorm2d(oup_inc, eps=0.001)
        self.conv3x1_1_r = nn.Conv2d(oup_inc, oup_inc, (3, 1), stride=1,
            padding=(1, 0), bias=True)
        self.conv1x3_1_r = nn.Conv2d(oup_inc, oup_inc, (1, 3), stride=1,
            padding=(0, 1), bias=True)
        self.bn1_r = nn.BatchNorm2d(oup_inc, eps=0.001)
        self.conv3x1_2_r = nn.Conv2d(oup_inc, oup_inc, (3, 1), stride=1,
            padding=(1 * dilated, 0), bias=True, dilation=(dilated, 1))
        self.conv1x3_2_r = nn.Conv2d(oup_inc, oup_inc, (1, 3), stride=1,
            padding=(0, 1 * dilated), bias=True, dilation=(1, dilated))
        self.bn2_r = nn.BatchNorm2d(oup_inc, eps=0.001)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, x):
        residual = x
        x1, x2 = Split(x)
        output1 = self.conv3x1_1_l(x1)
        output1 = self.relu(output1)
        output1 = self.conv1x3_1_l(output1)
        output1 = self.bn1_l(output1)
        output1_mid = self.relu(output1)
        output2 = self.conv1x3_1_r(x2)
        output2 = self.relu(output2)
        output2 = self.conv3x1_1_r(output2)
        output2 = self.bn1_r(output2)
        output2_mid = self.relu(output2)
        output1 = self.conv3x1_2_l(output1_mid)
        output1 = self.relu(output1)
        output1 = self.conv1x3_2_l(output1)
        output1 = self.bn2_l(output1)
        output2 = self.conv1x3_2_r(output2_mid)
        output2 = self.relu(output2)
        output2 = self.conv3x1_2_r(output2)
        output2 = self.bn2_r(output2)
        if self.dropout.p != 0:
            output1 = self.dropout(output1)
            output2 = self.dropout(output2)
        out = Merge(output1, output2)
        out = F.relu(residual + out)
        out = Channel_shuffle(out, 2)
        return out


class APNModule(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(APNModule, self).__init__()
        self.branch1 = nn.Sequential(nn.AdaptiveAvgPool2d(1), Conv2dBnRelu(
            in_ch, out_ch, kernel_size=1, stride=1, padding=0))
        self.mid = nn.Sequential(Conv2dBnRelu(in_ch, out_ch, kernel_size=1,
            stride=1, padding=0))
        self.down1 = nn.Sequential(nn.Conv2d(in_ch, 1, kernel_size=(7, 1),
            stride=(2, 1), padding=(3, 0), bias=True), nn.Conv2d(1, 1,
            kernel_size=(1, 7), stride=(1, 2), padding=(0, 3), bias=True),
            nn.BatchNorm2d(1, eps=0.001), nn.ReLU(inplace=True))
        self.down2 = nn.Sequential(nn.Conv2d(1, 1, kernel_size=(5, 1),
            stride=(2, 1), padding=(2, 0), bias=True), nn.Conv2d(1, 1,
            kernel_size=(1, 5), stride=(1, 2), padding=(0, 2), bias=True),
            nn.BatchNorm2d(1, eps=0.001), nn.ReLU(inplace=True))
        self.down3 = nn.Sequential(nn.Conv2d(1, 1, kernel_size=(3, 1),
            stride=(2, 1), padding=(1, 0), bias=True), nn.Conv2d(1, 1,
            kernel_size=(1, 3), stride=(1, 2), padding=(0, 1), bias=True),
            nn.BatchNorm2d(1, eps=0.001), nn.ReLU(inplace=True), nn.Conv2d(
            1, 1, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=True),
            nn.Conv2d(1, 1, kernel_size=(1, 3), stride=1, padding=(0, 1),
            bias=True), nn.BatchNorm2d(1, eps=0.001), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(1, 1, kernel_size=(5, 1),
            stride=1, padding=(2, 0), bias=True), nn.Conv2d(1, 1,
            kernel_size=(1, 5), stride=1, padding=(0, 2), bias=True), nn.
            BatchNorm2d(1, eps=0.001), nn.ReLU(inplace=True))
        self.conv1 = nn.Sequential(nn.Conv2d(1, 1, kernel_size=(7, 1),
            stride=1, padding=(3, 0), bias=True), nn.Conv2d(1, 1,
            kernel_size=(1, 7), stride=1, padding=(0, 3), bias=True), nn.
            BatchNorm2d(1, eps=0.001), nn.ReLU(inplace=True))

    def forward(self, x):
        h, w = x.size()[2:]
        b1 = self.branch1(x)
        b1 = F.interpolate(b1, size=(h, w), mode='bilinear', align_corners=True
            )
        mid = self.mid(x)
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x3 = F.interpolate(x3, size=((h + 3) // 4, (w + 3) // 4), mode=
            'bilinear', align_corners=True)
        x2 = self.conv2(x2)
        x = x2 + x3
        x = F.interpolate(x, size=((h + 1) // 2, (w + 1) // 2), mode=
            'bilinear', align_corners=True)
        x1 = self.conv1(x1)
        x = x + x1
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        x = torch.mul(x, mid)
        x = x + b1
        return x


class LEDNet(nn.Module):

    def __init__(self, classes):
        super().__init__()
        self.initial_block = DownsamplerBlock(3, 32)
        self.layers = nn.ModuleList()
        for x in range(0, 3):
            self.layers.append(SS_nbt_module_paper(32, 0.03, 1))
        self.layers.append(DownsamplerBlock(32, 64))
        for x in range(0, 2):
            self.layers.append(SS_nbt_module_paper(64, 0.03, 1))
        self.layers.append(DownsamplerBlock(64, 128))
        for x in range(0, 1):
            self.layers.append(SS_nbt_module_paper(128, 0.3, 1))
            self.layers.append(SS_nbt_module_paper(128, 0.3, 2))
            self.layers.append(SS_nbt_module_paper(128, 0.3, 5))
            self.layers.append(SS_nbt_module_paper(128, 0.3, 9))
        for x in range(0, 1):
            self.layers.append(SS_nbt_module_paper(128, 0.3, 2))
            self.layers.append(SS_nbt_module_paper(128, 0.3, 5))
            self.layers.append(SS_nbt_module_paper(128, 0.3, 9))
            self.layers.append(SS_nbt_module_paper(128, 0.3, 17))
        self.apn = APNModule(in_ch=128, out_ch=classes)

    def forward(self, input):
        output = self.initial_block(input)
        for layer in self.layers:
            output = layer(output)
        output = self.apn(output)
        out = F.interpolate(output, input.size()[2:], mode='bilinear',
            align_corners=True)
        return out


class BasicBlock(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1,
        padding=0, groups=1, bias=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size, stride,
            padding, groups=groups, bias=bias)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size, 1,
            padding, groups=groups, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.downsample = None
        if stride > 1:
            self.downsample = nn.Sequential(nn.Conv2d(in_planes, out_planes,
                kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(
                out_planes))

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.relu(out + residual)
        return out


class Encoder(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1,
        padding=0, groups=1, bias=False):
        super(Encoder, self).__init__()
        self.block1 = BasicBlock(in_planes, out_planes, kernel_size, stride,
            padding, groups, bias)
        self.block2 = BasicBlock(out_planes, out_planes, kernel_size, 1,
            padding, groups, bias)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x


def center_crop(layer, max_height, max_width):
    _, _, h, w = layer.size()
    diffy = (h - max_height) // 2
    diffx = (w - max_width) // 2
    return layer[:, :, diffy:diffy + max_height, diffx:diffx + max_width]


class Decoder(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1,
        padding=0, output_padding=0, groups=1, bias=False):
        super(Decoder, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_planes, in_planes // 4, 1, 
            1, 0, bias=bias), nn.BatchNorm2d(in_planes // 4), nn.ReLU(
            inplace=True))
        self.tp_conv = nn.Sequential(nn.ConvTranspose2d(in_planes // 4, 
            in_planes // 4, kernel_size, stride, padding, output_padding,
            bias=bias), nn.BatchNorm2d(in_planes // 4), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_planes // 4, out_planes, 1,
            1, 0, bias=bias), nn.BatchNorm2d(out_planes), nn.ReLU(inplace=True)
            )

    def forward(self, x_high_level, x_low_level):
        x = self.conv1(x_high_level)
        x = self.tp_conv(x)
        x = center_crop(x, x_low_level.size()[2], x_low_level.size()[3])
        x = self.conv2(x)
        return x


class LinkNetImprove(nn.Module):
    """
    Generate Model Architecture
    """

    def __init__(self, classes=19):
        """
        Model initialization
        :param x_n: number of input neurons
        :type x_n: int
        """
        super().__init__()
        base = resnet.resnet18(pretrained=True)
        self.in_block = nn.Sequential(base.conv1, base.bn1, base.relu, base
            .maxpool)
        self.encoder1 = base.layer1
        self.encoder2 = base.layer2
        self.encoder3 = base.layer3
        self.encoder4 = base.layer4
        self.decoder1 = Decoder(64, 64, 3, 1, 1, 0)
        self.decoder2 = Decoder(128, 64, 3, 2, 1, 1)
        self.decoder3 = Decoder(256, 128, 3, 2, 1, 1)
        self.decoder4 = Decoder(512, 256, 3, 2, 1, 1)
        self.tp_conv1 = nn.Sequential(nn.ConvTranspose2d(64, 32, 3, 2, 1, 1
            ), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1), nn.
            BatchNorm2d(32), nn.ReLU(inplace=True))
        self.tp_conv2 = nn.ConvTranspose2d(32, classes, 2, 2, 0)

    def forward(self, x):
        x = self.in_block(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        d4 = e3 + self.decoder4(e4, e3)
        d3 = e2 + self.decoder3(d4, e2)
        d2 = e1 + self.decoder2(d3, e1)
        d1 = x + self.decoder1(d2, x)
        y = self.tp_conv1(d1)
        y = self.conv2(y)
        y = self.tp_conv2(y)
        return y


class LinkNet(nn.Module):
    """
    Generate model architecture
    """

    def __init__(self, classes=19):
        """
        Model initialization
        :param x_n: number of input neurons
        :type x_n: int
        """
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.encoder1 = Encoder(64, 64, 3, 1, 1)
        self.encoder2 = Encoder(64, 128, 3, 2, 1)
        self.encoder3 = Encoder(128, 256, 3, 2, 1)
        self.encoder4 = Encoder(256, 512, 3, 2, 1)
        self.decoder4 = Decoder(512, 256, 3, 2, 1, 1)
        self.decoder3 = Decoder(256, 128, 3, 2, 1, 1)
        self.decoder2 = Decoder(128, 64, 3, 2, 1, 1)
        self.decoder1 = Decoder(64, 64, 3, 1, 1, 0)
        self.tp_conv1 = nn.Sequential(nn.ConvTranspose2d(64, 32, 3, 2, 1, 1
            ), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1), nn.
            BatchNorm2d(32), nn.ReLU(inplace=True))
        self.tp_conv2 = nn.ConvTranspose2d(32, classes, 2, 2, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        d4 = e3 + self.decoder4(e4, e3)
        d3 = e2 + self.decoder3(d4, e2)
        d2 = e1 + self.decoder2(d3, e1)
        d1 = x + self.decoder1(d2, x)
        y = self.tp_conv1(d1)
        y = self.conv2(y)
        y = self.tp_conv2(y)
        return y


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


class SQNet(nn.Module):

    def __init__(self, classes):
        super().__init__()
        self.num_classes = classes
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
        x_3 = F.interpolate(x_3, y_3.size()[2:], mode='bilinear',
            align_corners=True)
        x = torch.cat([x_3, y_3], 1)
        x = self.conv3_2(x)
        x = self.relu3_2(x)
        y_2 = self.deconv2(x)
        y_2 = self.relu3(y_2)
        x_2 = self.conv2_1(x_2)
        x_2 = self.relu2_1(x_2)
        y_2 = F.interpolate(y_2, x_2.size()[2:], mode='bilinear',
            align_corners=True)
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


class SegNet(nn.Module):

    def __init__(self, classes=19):
        super(SegNet, self).__init__()
        batchNorm_momentum = 0.1
        self.conv11 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(64, momentum=batchNorm_momentum)
        self.conv12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(64, momentum=batchNorm_momentum)
        self.conv21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm2d(128, momentum=batchNorm_momentum)
        self.conv22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(128, momentum=batchNorm_momentum)
        self.conv31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn31 = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32 = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33 = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn41 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv53d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv52d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv51d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv43d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv42d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv41d = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn41d = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv33d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33d = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv32d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32d = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv31d = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn31d = nn.BatchNorm2d(128, momentum=batchNorm_momentum)
        self.conv22d = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22d = nn.BatchNorm2d(128, momentum=batchNorm_momentum)
        self.conv21d = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn21d = nn.BatchNorm2d(64, momentum=batchNorm_momentum)
        self.conv12d = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12d = nn.BatchNorm2d(64, momentum=batchNorm_momentum)
        self.conv11d = nn.Conv2d(64, classes, kernel_size=3, padding=1)

    def forward(self, x):
        x11 = F.relu(self.bn11(self.conv11(x)))
        x12 = F.relu(self.bn12(self.conv12(x11)))
        x1_size = x12.size()
        x1p, id1 = F.max_pool2d(x12, kernel_size=2, stride=2,
            return_indices=True)
        x21 = F.relu(self.bn21(self.conv21(x1p)))
        x22 = F.relu(self.bn22(self.conv22(x21)))
        x2_size = x22.size()
        x2p, id2 = F.max_pool2d(x22, kernel_size=2, stride=2,
            return_indices=True)
        x31 = F.relu(self.bn31(self.conv31(x2p)))
        x32 = F.relu(self.bn32(self.conv32(x31)))
        x33 = F.relu(self.bn33(self.conv33(x32)))
        x3_size = x33.size()
        x3p, id3 = F.max_pool2d(x33, kernel_size=2, stride=2,
            return_indices=True)
        x41 = F.relu(self.bn41(self.conv41(x3p)))
        x42 = F.relu(self.bn42(self.conv42(x41)))
        x43 = F.relu(self.bn43(self.conv43(x42)))
        x4_size = x43.size()
        x4p, id4 = F.max_pool2d(x43, kernel_size=2, stride=2,
            return_indices=True)
        x51 = F.relu(self.bn51(self.conv51(x4p)))
        x52 = F.relu(self.bn52(self.conv52(x51)))
        x53 = F.relu(self.bn53(self.conv53(x52)))
        x5_size = x53.size()
        x5p, id5 = F.max_pool2d(x53, kernel_size=2, stride=2,
            return_indices=True)
        x5d = F.max_unpool2d(x5p, id5, kernel_size=2, stride=2, output_size
            =x5_size)
        x53d = F.relu(self.bn53d(self.conv53d(x5d)))
        x52d = F.relu(self.bn52d(self.conv52d(x53d)))
        x51d = F.relu(self.bn51d(self.conv51d(x52d)))
        x4d = F.max_unpool2d(x51d, id4, kernel_size=2, stride=2,
            output_size=x4_size)
        x43d = F.relu(self.bn43d(self.conv43d(x4d)))
        x42d = F.relu(self.bn42d(self.conv42d(x43d)))
        x41d = F.relu(self.bn41d(self.conv41d(x42d)))
        x3d = F.max_unpool2d(x41d, id3, kernel_size=2, stride=2,
            output_size=x3_size)
        x33d = F.relu(self.bn33d(self.conv33d(x3d)))
        x32d = F.relu(self.bn32d(self.conv32d(x33d)))
        x31d = F.relu(self.bn31d(self.conv31d(x32d)))
        x2d = F.max_unpool2d(x31d, id2, kernel_size=2, stride=2,
            output_size=x2_size)
        x22d = F.relu(self.bn22d(self.conv22d(x2d)))
        x21d = F.relu(self.bn21d(self.conv21d(x22d)))
        x1d = F.max_unpool2d(x21d, id1, kernel_size=2, stride=2,
            output_size=x1_size)
        x12d = F.relu(self.bn12d(self.conv12d(x1d)))
        x11d = self.conv11d(x12d)
        return x11d

    def load_from_segnet(self, model_path):
        s_dict = self.state_dict()
        th = torch.load(model_path).state_dict()
        self.load_state_dict(th)


class double_conv(nn.Module):
    """(conv => BN => ReLU) * 2"""

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True), nn.Conv2d(out_ch,
            out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=
            True))

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(nn.MaxPool2d(2), double_conv(in_ch, out_ch)
            )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):

    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()
        self.bilinear = bilinear
        self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = F.interpolate(x1, scale_factor=2, mode='bilinear',
                align_corners=True)
        else:
            x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY -
            diffY // 2])
        x = torch.cat([x2, x1], dim=1)
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

    def __init__(self, classes):
        super(UNet, self).__init__()
        self.inc = inconv(3, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x


class Mish(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class BetaMish(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        beta = 1.5
        return x * torch.tanh(torch.log(torch.pow(1 + torch.exp(x), beta)))


class Swish(nn.Module):

    def __init__(self):
        super(Swish, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)


class Hswish(nn.Module):

    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3.0, inplace=self.inplace) / 6.0


class Hsigmoid(nn.Module):

    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3.0, inplace=self.inplace) / 6.0


class SEModule(nn.Module):

    def __init__(self, channel, act, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(nn.Conv2d(channel, channel // reduction, 
            1, 1, 0, bias=True), act)
        self.fc = nn.Sequential(nn.Conv2d(channel // reduction, channel, 1,
            1, 0, bias=True), Hsigmoid())

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y)
        y = self.fc(y)
        return torch.mul(x, y)


class CrossEntropyLoss2d(_WeightedLoss):
    """
    Standard pytorch weighted nn.CrossEntropyLoss
    """

    def __init__(self, weight=None, ignore_label=255, reduction='mean'):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.CrossEntropyLoss(weight, ignore_index=
            ignore_label, reduction=reduction)

    def forward(self, output, target):
        """
        Forward pass
        :param output: torch.tensor (NxC)
        :param target: torch.tensor (N)
        :return: scalar
        """
        return self.nll_loss(output, target)


class CrossEntropyLoss2dLabelSmooth(_WeightedLoss):
    """
    Refer from https://arxiv.org/pdf/1512.00567.pdf
    :param target: N,
    :param n_classes: int
    :param eta: float
    :return:
        N x C onehot smoothed vector
    """

    def __init__(self, weight=None, ignore_label=255, epsilon=0.1,
        reduction='mean'):
        super(CrossEntropyLoss2dLabelSmooth, self).__init__()
        self.epsilon = epsilon
        self.nll_loss = nn.CrossEntropyLoss(weight, ignore_index=
            ignore_label, reduction=reduction)

    def forward(self, output, target):
        """
        Forward pass
        :param output: torch.tensor (NxC)
        :param target: torch.tensor (N)
        :return: scalar
        """
        n_classes = output.size(1)
        targets = torch.zeros_like(output).scatter_(1, target.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / n_classes
        return self.nll_loss(output, targets)


class FocalLoss2d(nn.Module):

    def __init__(self, alpha=0.5, gamma=2, weight=None, ignore_index=255,
        size_average=True):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.size_average = size_average
        self.ce_fn = nn.CrossEntropyLoss(weight=self.weight, ignore_index=
            self.ignore_index)

    def forward(self, output, target):
        if output.dim() > 2:
            output = output.contiguous().view(output.size(0), output.size(1
                ), -1)
            output = output.transpose(1, 2)
            output = output.contiguous().view(-1, output.size(2)).squeeze()
        if target.dim() == 4:
            target = target.contiguous().view(target.size(0), target.size(1
                ), -1)
            target = target.transpose(1, 2)
            target = target.contiguous().view(-1, target.size(2)).squeeze()
        elif target.dim() == 3:
            target = target.view(-1)
        else:
            target = target.view(-1, 1)
        logpt = self.ce_fn(output, target)
        pt = torch.exp(-logpt)
        loss = (1 - pt) ** self.gamma * self.alpha * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class LDAMLoss(nn.Module):

    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[(None), :], index_float.
            transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s * output, target, weight=self.weight)


class ProbOhemCrossEntropy2d(nn.Module):

    def __init__(self, ignore_label=255, reduction='mean', thresh=0.6,
        min_kept=256, down_ratio=1, use_weight=False):
        super(ProbOhemCrossEntropy2d, self).__init__()
        self.ignore_label = ignore_label
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.down_ratio = down_ratio
        if use_weight:
            None
            weight = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 
                1.0166, 0.9969, 0.9754, 1.0489, 0.8786, 1.0023, 0.9539, 
                0.9843, 1.1116, 0.9037, 1.0865, 1.0955, 1.0865, 1.1529, 1.0507]
                )
            self.criterion = nn.CrossEntropyLoss(reduction=reduction,
                weight=weight, ignore_index=ignore_label)
        else:
            None
            self.criterion = nn.CrossEntropyLoss(reduction=reduction,
                ignore_index=ignore_label)

    def forward(self, pred, target):
        b, c, h, w = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_label)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()
        prob = F.softmax(pred, dim=1)
        prob = prob.transpose(0, 1).reshape(c, -1)
        if self.min_kept > num_valid:
            None
            pass
        elif num_valid > 0:
            prob = prob.masked_fill_(1 - valid_mask, 1)
            mask_prob = prob[target, torch.arange(len(target), dtype=torch.
                long)]
            threshold = self.thresh
            if self.min_kept > 0:
                index = mask_prob.argsort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
                kept_mask = mask_prob.le(threshold)
                target = target * kept_mask.long()
                valid_mask = valid_mask * kept_mask
                None
        target = target.masked_fill_(1 - valid_mask, self.ignore_label)
        target = target.view(b, h, w)
        return self.criterion(pred, target)


class CrossEntropy2d(nn.Module):

    def __init__(self, size_average=True, ignore_label=255, use_weight=True):
        super(CrossEntropy2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label
        self.use_weight = use_weight

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        if self.use_weight:
            None
            freq = np.zeros(19)
            for k in range(19):
                mask = target[:, :, :] == k
                freq[k] = torch.sum(mask)
                None
            weight = freq / np.sum(freq)
            None
            self.weight = torch.FloatTensor(weight)
            None
        else:
            self.weight = None
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=
            self.ignore_label)
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), '{0} vs {1} '.format(predict
            .size(0), target.size(0))
        assert predict.size(2) == target.size(1), '{0} vs {1} '.format(predict
            .size(2), target.size(1))
        assert predict.size(3) == target.size(2), '{0} vs {1} '.format(predict
            .size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return torch.zeros(1)
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)
            ].view(-1, c)
        loss = criterion(predict, target)
        return loss


def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    if probas.dim() == 3:
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = labels != ignore
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels


def isnan(x):
    return x != x


def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        return probas * 0.0
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float()
        if classes is 'present' and fg.sum() == 0:
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, (0)]
        else:
            class_pred = probas[:, (c)]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(
            fg_sorted))))
    return mean(losses)


def lovasz_softmax(probas, labels, classes='present', per_image=False,
    ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0),
            lab.unsqueeze(0), ignore), classes=classes) for prob, lab in
            zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore),
            classes=classes)
    return loss


class LovaszSoftmax(nn.Module):

    def __init__(self, classes='present', per_image=False, ignore_index=255):
        super(LovaszSoftmax, self).__init__()
        self.smooth = classes
        self.per_image = per_image
        self.ignore_index = ignore_index

    def forward(self, output, target):
        logits = F.softmax(output, dim=1)
        loss = lovasz_softmax(logits, target, ignore=self.ignore_index)
        return loss


class StableBCELoss(torch.nn.modules.Module):

    def __init__(self):
        super(StableBCELoss, self).__init__()

    def forward(self, input, target):
        neg_abs = -input.abs()
        loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
        return loss.mean()


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_xiaoyufenfei_Efficient_Segmentation_Networks(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(ContextNet(*[], **{'classes': 4}), [torch.rand([4, 3, 64, 64])], {})

    @_fails_compile()
    def test_001(self):
        self._check(DABNet(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

    @_fails_compile()
    def test_002(self):
        self._check(ESNet(*[], **{'classes': 4}), [torch.rand([4, 3, 64, 64])], {})

    @_fails_compile()
    def test_003(self):
        self._check(ESPNet(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

    @_fails_compile()
    def test_004(self):
        self._check(FPENet(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

    @_fails_compile()
    def test_005(self):
        self._check(FastSCNN(*[], **{'classes': 4}), [torch.rand([4, 3, 64, 64])], {})

    @_fails_compile()
    def test_006(self):
        self._check(LEDNet(*[], **{'classes': 4}), [torch.rand([4, 3, 64, 64])], {})

    @_fails_compile()
    def test_007(self):
        self._check(LinkNet(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

    def test_008(self):
        self._check(SQNet(*[], **{'classes': 4}), [torch.rand([4, 3, 64, 64])], {})

    def test_009(self):
        self._check(SegNet(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

    @_fails_compile()
    def test_010(self):
        self._check(UNet(*[], **{'classes': 4}), [torch.rand([4, 3, 64, 64])], {})

    def test_011(self):
        self._check(ConvBNPReLU(*[], **{'nIn': 4, 'nOut': 4, 'kSize': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_012(self):
        self._check(BNPReLU(*[], **{'nIn': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_013(self):
        self._check(ConvBN(*[], **{'nIn': 4, 'nOut': 4, 'kSize': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_014(self):
        self._check(Conv(*[], **{'nIn': 4, 'nOut': 4, 'kSize': 4, 'stride': 1, 'padding': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_015(self):
        self._check(ChannelWiseConv(*[], **{'nIn': 4, 'nOut': 4, 'kSize': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_016(self):
        self._check(DilatedConv(*[], **{'nIn': 4, 'nOut': 4, 'kSize': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_017(self):
        self._check(ChannelWiseDilatedConv(*[], **{'nIn': 4, 'nOut': 4, 'kSize': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_018(self):
        self._check(InputInjection(*[], **{'ratio': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_019(self):
        self._check(Custom_Conv(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_020(self):
        self._check(DepthSepConv(*[], **{'dw_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_021(self):
        self._check(DepthConv(*[], **{'dw_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_022(self):
        self._check(LinearBottleneck(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_023(self):
        self._check(Shallow_net(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

    def test_024(self):
        self._check(FeatureFusionModule(*[], **{'highter_in_channels': 4, 'lower_in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_025(self):
        self._check(Classifer(*[], **{'dw_channels': 4, 'num_classes': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_026(self):
        self._check(DABModule(*[], **{'nIn': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_027(self):
        self._check(DownSamplingBlock(*[], **{'nIn': 4, 'nOut': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_028(self):
        self._check(EDAModule(*[], **{'ninput': 4, 'dilated': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_029(self):
        self._check(EDANetBlock(*[], **{'in_channels': 4, 'num_dense_layer': 1, 'dilated': [4, 4], 'growth_rate': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_030(self):
        self._check(UpsamplerBlock(*[], **{'ninput': 4, 'noutput': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_031(self):
        self._check(Decoder(*[], **{'in_planes': 4, 'out_planes': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_032(self):
        self._check(PFCU(*[], **{'chann': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_033(self):
        self._check(CBR(*[], **{'nIn': 4, 'nOut': 4, 'kSize': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_034(self):
        self._check(BR(*[], **{'nOut': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_035(self):
        self._check(CB(*[], **{'nIn': 4, 'nOut': 4, 'kSize': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_036(self):
        self._check(C(*[], **{'nIn': 4, 'nOut': 4, 'kSize': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_037(self):
        self._check(CDilated(*[], **{'nIn': 4, 'nOut': 4, 'kSize': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_038(self):
        self._check(DownSamplerB(*[], **{'nIn': 64, 'nOut': 64}), [torch.rand([4, 64, 64, 64])], {})

    def test_039(self):
        self._check(DilatedParllelResidualBlockB(*[], **{'nIn': 64, 'nOut': 64}), [torch.rand([4, 64, 64, 64])], {})

    def test_040(self):
        self._check(InputProjectionA(*[], **{'samplingTimes': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_041(self):
        self._check(ESPNet_Encoder(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

    @_fails_compile()
    def test_042(self):
        self._check(EESP(*[], **{'nIn': 64, 'nOut': 64}), [torch.rand([4, 64, 64, 64])], {})

    @_fails_compile()
    def test_043(self):
        self._check(EESPNet(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

    def test_044(self):
        self._check(PSPModule(*[], **{'features': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_045(self):
        self._check(CDilatedB(*[], **{'nIn': 4, 'nOut': 4, 'kSize': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_046(self):
        self._check(FPEBlock(*[], **{'inplanes': 4, 'outplanes': 4, 'dilat': [4, 4, 4, 4]}), [torch.rand([4, 4, 4, 4])], {})

    def test_047(self):
        self._check(MEUModule(*[], **{'channels_high': 4, 'channels_low': 4, 'channel_out': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_048(self):
        self._check(_ConvBNReLU(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_049(self):
        self._check(_DSConv(*[], **{'dw_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_050(self):
        self._check(_DWConv(*[], **{'dw_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_051(self):
        self._check(PyramidPooling(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_052(self):
        self._check(LearningToDownsample(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

    @_fails_compile()
    def test_053(self):
        self._check(GlobalFeatureExtractor(*[], **{}), [torch.rand([4, 64, 64, 64])], {})

    def test_054(self):
        self._check(PermutationBlock(*[], **{'groups': 1}), [torch.rand([4, 4, 4, 4])], {})

    def test_055(self):
        self._check(Conv2dBnRelu(*[], **{'in_ch': 4, 'out_ch': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_056(self):
        self._check(APNModule(*[], **{'in_ch': 4, 'out_ch': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_057(self):
        self._check(Fire(*[], **{'inplanes': 4, 'squeeze_planes': 4, 'expand_planes': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_058(self):
        self._check(ParallelDilatedConv(*[], **{'inplanes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_059(self):
        self._check(double_conv(*[], **{'in_ch': 4, 'out_ch': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_060(self):
        self._check(inconv(*[], **{'in_ch': 4, 'out_ch': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_061(self):
        self._check(down(*[], **{'in_ch': 4, 'out_ch': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_062(self):
        self._check(up(*[], **{'in_ch': 4, 'out_ch': 4}), [torch.rand([4, 1, 4, 4]), torch.rand([4, 3, 4, 4])], {})

    def test_063(self):
        self._check(outconv(*[], **{'in_ch': 4, 'out_ch': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_064(self):
        self._check(Mish(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_065(self):
        self._check(BetaMish(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_066(self):
        self._check(Swish(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_067(self):
        self._check(Hswish(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_068(self):
        self._check(Hsigmoid(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_069(self):
        self._check(StableBCELoss(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

