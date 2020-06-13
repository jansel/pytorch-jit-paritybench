import sys
_module = sys.modules[__name__]
del sys
KITTIloader2012 = _module
KITTIloader2015 = _module
MiddleburyLoader = _module
dataloader = _module
flow_transforms = _module
listfiles = _module
listsceneflow = _module
eval_disp = _module
eval_mb = _module
models = _module
hsm = _module
submodule = _module
utils = _module
submission = _module
train = _module
eval = _module
logger = _module
preprocess = _module
readpfm = _module

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


import torch.utils.data


from torch.autograd import Variable


import torch.nn.functional as F


import math


import numpy as np


import torch.backends.cudnn as cudnn


import random


import torch.optim as optim


class HSMNet(nn.Module):

    def __init__(self, maxdisp, clean, level=1):
        super(HSMNet, self).__init__()
        self.maxdisp = maxdisp
        self.clean = clean
        self.feature_extraction = unet()
        self.level = level
        self.decoder6 = decoderBlock(6, 32, 32, up=True, pool=True)
        if self.level > 2:
            self.decoder5 = decoderBlock(6, 32, 32, up=False, pool=True)
        else:
            self.decoder5 = decoderBlock(6, 32, 32, up=True, pool=True)
            if self.level > 1:
                self.decoder4 = decoderBlock(6, 32, 32, up=False)
            else:
                self.decoder4 = decoderBlock(6, 32, 32, up=True)
                self.decoder3 = decoderBlock(5, 32, 32, stride=(2, 1, 1),
                    up=False, nstride=1)
        self.disp_reg8 = disparityregression(self.maxdisp, 16)
        self.disp_reg16 = disparityregression(self.maxdisp, 16)
        self.disp_reg32 = disparityregression(self.maxdisp, 32)
        self.disp_reg64 = disparityregression(self.maxdisp, 64)

    def feature_vol(self, refimg_fea, targetimg_fea, maxdisp, leftview=True):
        """
        diff feature volume
        """
        width = refimg_fea.shape[-1]
        cost = Variable(torch.cuda.FloatTensor(refimg_fea.size()[0],
            refimg_fea.size()[1], maxdisp, refimg_fea.size()[2], refimg_fea
            .size()[3]).fill_(0.0))
        for i in range(maxdisp):
            feata = refimg_fea[:, :, :, i:width]
            featb = targetimg_fea[:, :, :, :width - i]
            if leftview:
                cost[:, :refimg_fea.size()[1], (i), :, i:] = torch.abs(
                    feata - featb)
            else:
                cost[:, :refimg_fea.size()[1], (i), :, :width - i] = torch.abs(
                    featb - feata)
        cost = cost.contiguous()
        return cost

    def forward(self, left, right):
        nsample = left.shape[0]
        conv4, conv3, conv2, conv1 = self.feature_extraction(torch.cat([
            left, right], 0))
        conv40, conv30, conv20, conv10 = conv4[:nsample], conv3[:nsample
            ], conv2[:nsample], conv1[:nsample]
        conv41, conv31, conv21, conv11 = conv4[nsample:], conv3[nsample:
            ], conv2[nsample:], conv1[nsample:]
        feat6 = self.feature_vol(conv40, conv41, self.maxdisp // 64)
        feat5 = self.feature_vol(conv30, conv31, self.maxdisp // 32)
        feat4 = self.feature_vol(conv20, conv21, self.maxdisp // 16)
        feat3 = self.feature_vol(conv10, conv11, self.maxdisp // 8)
        feat6_2x, cost6 = self.decoder6(feat6)
        feat5 = torch.cat((feat6_2x, feat5), dim=1)
        feat5_2x, cost5 = self.decoder5(feat5)
        if self.level > 2:
            cost3 = F.upsample(cost5, [left.size()[2], left.size()[3]],
                mode='bilinear')
        else:
            feat4 = torch.cat((feat5_2x, feat4), dim=1)
            feat4_2x, cost4 = self.decoder4(feat4)
            if self.level > 1:
                cost3 = F.upsample(cost4.unsqueeze(1), [self.disp_reg8.disp
                    .shape[1], left.size()[2], left.size()[3]], mode=
                    'trilinear').squeeze(1)
            else:
                feat3 = torch.cat((feat4_2x, feat3), dim=1)
                feat3_2x, cost3 = self.decoder3(feat3)
                cost3 = F.upsample(cost3, [left.size()[2], left.size()[3]],
                    mode='bilinear')
        if self.level > 2:
            final_reg = self.disp_reg32
        else:
            final_reg = self.disp_reg8
        if self.training or self.clean == -1:
            pred3 = final_reg(F.softmax(cost3, 1))
            entropy = pred3
        else:
            pred3, entropy = final_reg(F.softmax(cost3, 1), ifent=True)
            pred3[entropy > self.clean] = np.inf
        if self.training:
            cost6 = F.upsample(cost6.unsqueeze(1), [self.disp_reg8.disp.
                shape[1], left.size()[2], left.size()[3]], mode='trilinear'
                ).squeeze(1)
            cost5 = F.upsample(cost5.unsqueeze(1), [self.disp_reg8.disp.
                shape[1], left.size()[2], left.size()[3]], mode='trilinear'
                ).squeeze(1)
            cost4 = F.upsample(cost4, [left.size()[2], left.size()[3]],
                mode='bilinear')
            pred6 = self.disp_reg16(F.softmax(cost6, 1))
            pred5 = self.disp_reg16(F.softmax(cost5, 1))
            pred4 = self.disp_reg16(F.softmax(cost4, 1))
            stacked = [pred3, pred4, pred5, pred6]
            return stacked, entropy
        else:
            return pred3, torch.squeeze(entropy)


def sepConv3d(in_planes, out_planes, kernel_size, stride, pad, bias=False):
    if bias:
        return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=
            kernel_size, padding=pad, stride=stride, bias=bias))
    else:
        return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=
            kernel_size, padding=pad, stride=stride, bias=bias), nn.
            BatchNorm3d(out_planes))


class sepConv3dBlock(nn.Module):
    """
    Separable 3d convolution block as 2 separable convolutions and a projection
    layer
    """

    def __init__(self, in_planes, out_planes, stride=(1, 1, 1)):
        super(sepConv3dBlock, self).__init__()
        if in_planes == out_planes and stride == (1, 1, 1):
            self.downsample = None
        else:
            self.downsample = projfeat3d(in_planes, out_planes, stride)
        self.conv1 = sepConv3d(in_planes, out_planes, 3, stride, 1)
        self.conv2 = sepConv3d(out_planes, out_planes, 3, (1, 1, 1), 1)

    def forward(self, x):
        out = F.relu(self.conv1(x), inplace=True)
        if self.downsample:
            x = self.downsample(x)
        out = F.relu(x + self.conv2(out), inplace=True)
        return out


class projfeat3d(nn.Module):
    """
    Turn 3d projection into 2d projection
    """

    def __init__(self, in_planes, out_planes, stride):
        super(projfeat3d, self).__init__()
        self.stride = stride
        self.conv1 = nn.Conv2d(in_planes, out_planes, (1, 1), padding=(0, 0
            ), stride=stride[:2], bias=False)
        self.bn = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        b, c, d, h, w = x.size()
        x = self.conv1(x.view(b, c, d, h * w))
        x = self.bn(x)
        x = x.view(b, -1, d // self.stride[0], h, w)
        return x


class disparityregression(nn.Module):

    def __init__(self, maxdisp, divisor):
        super(disparityregression, self).__init__()
        maxdisp = int(maxdisp / divisor)
        self.register_buffer('disp', torch.Tensor(np.reshape(np.array(range
            (maxdisp)), [1, maxdisp, 1, 1])))
        self.divisor = divisor

    def forward(self, x, ifent=False):
        disp = self.disp.repeat(x.size()[0], 1, x.size()[2], x.size()[3])
        out = torch.sum(x * disp, 1) * self.divisor
        if ifent:
            x = x + 1e-12
            ent = (-x * x.log()).sum(dim=1)
            return out, ent
        else:
            return out


class decoderBlock(nn.Module):

    def __init__(self, nconvs, inchannelF, channelF, stride=(1, 1, 1), up=
        False, nstride=1, pool=False):
        super(decoderBlock, self).__init__()
        self.pool = pool
        stride = [stride] * nstride + [(1, 1, 1)] * (nconvs - nstride)
        self.convs = [sepConv3dBlock(inchannelF, channelF, stride=stride[0])]
        for i in range(1, nconvs):
            self.convs.append(sepConv3dBlock(channelF, channelF, stride=
                stride[i]))
        self.convs = nn.Sequential(*self.convs)
        self.classify = nn.Sequential(sepConv3d(channelF, channelF, 3, (1, 
            1, 1), 1), nn.ReLU(inplace=True), sepConv3d(channelF, 1, 3, (1,
            1, 1), 1, bias=True))
        self.up = False
        if up:
            self.up = True
            self.up = nn.Sequential(nn.Upsample(scale_factor=(2, 2, 2),
                mode='trilinear'), sepConv3d(channelF, channelF // 2, 3, (1,
                1, 1), 1, bias=False), nn.ReLU(inplace=True))
        if pool:
            self.pool_convs = torch.nn.ModuleList([sepConv3d(channelF,
                channelF, 1, (1, 1, 1), 0), sepConv3d(channelF, channelF, 1,
                (1, 1, 1), 0), sepConv3d(channelF, channelF, 1, (1, 1, 1), 
                0), sepConv3d(channelF, channelF, 1, (1, 1, 1), 0)])
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2
                    ] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if hasattr(m.bias, 'data'):
                    m.bias.data.zero_()

    def forward(self, fvl):
        fvl = self.convs(fvl)
        if self.pool:
            fvl_out = fvl
            _, _, d, h, w = fvl.shape
            for i, pool_size in enumerate(np.linspace(1, min(d, h, w) // 2,
                4, dtype=int)):
                kernel_size = int(d / pool_size), int(h / pool_size), int(w /
                    pool_size)
                out = F.avg_pool3d(fvl, kernel_size, stride=kernel_size)
                out = self.pool_convs[i](out)
                out = F.upsample(out, size=(d, h, w), mode='trilinear')
                fvl_out = fvl_out + 0.25 * out
            fvl = F.relu(fvl_out / 2.0, inplace=True)
        if self.training:
            costl = self.classify(fvl)
            if self.up:
                fvl = self.up(fvl)
        elif self.up:
            fvl = self.up(fvl)
            costl = fvl
        else:
            costl = self.classify(fvl)
        return fvl, costl.squeeze(1)


class unet(nn.Module):

    def __init__(self):
        super(unet, self).__init__()
        self.inplanes = 32
        self.convbnrelu1_1 = conv2DBatchNormRelu(in_channels=3, k_size=3,
            n_filters=16, padding=1, stride=2, bias=False)
        self.convbnrelu1_2 = conv2DBatchNormRelu(in_channels=16, k_size=3,
            n_filters=16, padding=1, stride=1, bias=False)
        self.convbnrelu1_3 = conv2DBatchNormRelu(in_channels=16, k_size=3,
            n_filters=32, padding=1, stride=1, bias=False)
        self.res_block3 = self._make_layer(residualBlock, 64, 1, stride=2)
        self.res_block5 = self._make_layer(residualBlock, 128, 1, stride=2)
        self.res_block6 = self._make_layer(residualBlock, 128, 1, stride=2)
        self.res_block7 = self._make_layer(residualBlock, 128, 1, stride=2)
        self.pyramid_pooling = pyramidPooling(128, None, fusion_mode='sum',
            model_name='icnet')
        self.upconv6 = nn.Sequential(nn.Upsample(scale_factor=2),
            conv2DBatchNormRelu(in_channels=128, k_size=3, n_filters=64,
            padding=1, stride=1, bias=False))
        self.iconv5 = conv2DBatchNormRelu(in_channels=192, k_size=3,
            n_filters=128, padding=1, stride=1, bias=False)
        self.upconv5 = nn.Sequential(nn.Upsample(scale_factor=2),
            conv2DBatchNormRelu(in_channels=128, k_size=3, n_filters=64,
            padding=1, stride=1, bias=False))
        self.iconv4 = conv2DBatchNormRelu(in_channels=192, k_size=3,
            n_filters=128, padding=1, stride=1, bias=False)
        self.upconv4 = nn.Sequential(nn.Upsample(scale_factor=2),
            conv2DBatchNormRelu(in_channels=128, k_size=3, n_filters=64,
            padding=1, stride=1, bias=False))
        self.iconv3 = conv2DBatchNormRelu(in_channels=128, k_size=3,
            n_filters=64, padding=1, stride=1, bias=False)
        self.proj6 = conv2DBatchNormRelu(in_channels=128, k_size=1,
            n_filters=32, padding=0, stride=1, bias=False)
        self.proj5 = conv2DBatchNormRelu(in_channels=128, k_size=1,
            n_filters=16, padding=0, stride=1, bias=False)
        self.proj4 = conv2DBatchNormRelu(in_channels=128, k_size=1,
            n_filters=16, padding=0, stride=1, bias=False)
        self.proj3 = conv2DBatchNormRelu(in_channels=64, k_size=1,
            n_filters=16, padding=0, stride=1, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if hasattr(m.bias, 'data'):
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
        conv1 = self.convbnrelu1_1(x)
        conv1 = self.convbnrelu1_2(conv1)
        conv1 = self.convbnrelu1_3(conv1)
        pool1 = F.max_pool2d(conv1, 3, 2, 1)
        conv3 = self.res_block3(pool1)
        conv4 = self.res_block5(conv3)
        conv5 = self.res_block6(conv4)
        conv6 = self.res_block7(conv5)
        conv6 = self.pyramid_pooling(conv6)
        concat5 = torch.cat((conv5, self.upconv6(conv6)), dim=1)
        conv5 = self.iconv5(concat5)
        concat4 = torch.cat((conv4, self.upconv5(conv5)), dim=1)
        conv4 = self.iconv4(concat4)
        concat3 = torch.cat((conv3, self.upconv4(conv4)), dim=1)
        conv3 = self.iconv3(concat3)
        proj6 = self.proj6(conv6)
        proj5 = self.proj5(conv5)
        proj4 = self.proj4(conv4)
        proj3 = self.proj3(conv3)
        return proj6, proj5, proj4, proj3


class conv2DBatchNorm(nn.Module):

    def __init__(self, in_channels, n_filters, k_size, stride, padding,
        bias=True, dilation=1, with_bn=True):
        super(conv2DBatchNorm, self).__init__()
        if dilation > 1:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters),
                kernel_size=k_size, padding=padding, stride=stride, bias=
                bias, dilation=dilation)
        else:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters),
                kernel_size=k_size, padding=padding, stride=stride, bias=
                bias, dilation=1)
        if with_bn:
            self.cb_unit = nn.Sequential(conv_mod, nn.BatchNorm2d(int(
                n_filters)))
        else:
            self.cb_unit = nn.Sequential(conv_mod)

    def forward(self, inputs):
        outputs = self.cb_unit(inputs)
        return outputs


class conv2DBatchNormRelu(nn.Module):

    def __init__(self, in_channels, n_filters, k_size, stride, padding,
        bias=True, dilation=1, with_bn=True):
        super(conv2DBatchNormRelu, self).__init__()
        if dilation > 1:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters),
                kernel_size=k_size, padding=padding, stride=stride, bias=
                bias, dilation=dilation)
        else:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters),
                kernel_size=k_size, padding=padding, stride=stride, bias=
                bias, dilation=1)
        if with_bn:
            self.cbr_unit = nn.Sequential(conv_mod, nn.BatchNorm2d(int(
                n_filters)), nn.LeakyReLU(0.1, inplace=True))
        else:
            self.cbr_unit = nn.Sequential(conv_mod, nn.LeakyReLU(0.1,
                inplace=True))

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs


class residualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, n_filters, stride=1, downsample=None,
        dilation=1):
        super(residualBlock, self).__init__()
        if dilation > 1:
            padding = dilation
        else:
            padding = 1
        self.convbnrelu1 = conv2DBatchNormRelu(in_channels, n_filters, 3,
            stride, padding, bias=False, dilation=dilation)
        self.convbn2 = conv2DBatchNorm(n_filters, n_filters, 3, 1, 1, bias=
            False)
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
        return out


class pyramidPooling(nn.Module):

    def __init__(self, in_channels, pool_sizes, model_name='pspnet',
        fusion_mode='cat', with_bn=True):
        super(pyramidPooling, self).__init__()
        bias = not with_bn
        self.paths = []
        if pool_sizes is None:
            for i in range(4):
                self.paths.append(conv2DBatchNormRelu(in_channels,
                    in_channels, 1, 1, 0, bias=bias, with_bn=with_bn))
        else:
            for i in range(len(pool_sizes)):
                self.paths.append(conv2DBatchNormRelu(in_channels, int(
                    in_channels / len(pool_sizes)), 1, 1, 0, bias=bias,
                    with_bn=with_bn))
        self.path_module_list = nn.ModuleList(self.paths)
        self.pool_sizes = pool_sizes
        self.model_name = model_name
        self.fusion_mode = fusion_mode

    def forward(self, x):
        h, w = x.shape[2:]
        k_sizes = []
        strides = []
        if self.pool_sizes is None:
            for pool_size in np.linspace(1, min(h, w) // 2, 4, dtype=int):
                k_sizes.append((int(h / pool_size), int(w / pool_size)))
                strides.append((int(h / pool_size), int(w / pool_size)))
            k_sizes = k_sizes[::-1]
            strides = strides[::-1]
        else:
            k_sizes = [(self.pool_sizes[0], self.pool_sizes[0]), (self.
                pool_sizes[1], self.pool_sizes[1]), (self.pool_sizes[2],
                self.pool_sizes[2]), (self.pool_sizes[3], self.pool_sizes[3])]
            strides = k_sizes
        if self.fusion_mode == 'cat':
            output_slices = [x]
            for i, (module, pool_size) in enumerate(zip(self.
                path_module_list, self.pool_sizes)):
                out = F.avg_pool2d(x, k_sizes[i], stride=strides[i], padding=0)
                if self.model_name != 'icnet':
                    out = module(out)
                out = F.upsample(out, size=(h, w), mode='bilinear')
                output_slices.append(out)
            return torch.cat(output_slices, dim=1)
        else:
            pp_sum = x
            for i, module in enumerate(self.path_module_list):
                out = F.avg_pool2d(x, k_sizes[i], stride=strides[i], padding=0)
                out = module(out)
                out = F.upsample(out, size=(h, w), mode='bilinear')
                pp_sum = pp_sum + 0.25 * out
            pp_sum = F.relu(pp_sum / 2.0, inplace=True)
            return pp_sum


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_gengshan_y_high_res_stereo(_paritybench_base):
    pass
    def test_000(self):
        self._check(conv2DBatchNorm(*[], **{'in_channels': 4, 'n_filters': 4, 'k_size': 4, 'stride': 1, 'padding': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(conv2DBatchNormRelu(*[], **{'in_channels': 4, 'n_filters': 4, 'k_size': 4, 'stride': 1, 'padding': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_002(self):
        self._check(disparityregression(*[], **{'maxdisp': 4, 'divisor': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(projfeat3d(*[], **{'in_planes': 4, 'out_planes': 4, 'stride': [4, 4]}), [torch.rand([4, 4, 4, 4, 4])], {})

    def test_004(self):
        self._check(residualBlock(*[], **{'in_channels': 4, 'n_filters': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_005(self):
        self._check(sepConv3dBlock(*[], **{'in_planes': 4, 'out_planes': 4}), [torch.rand([4, 4, 64, 64, 64])], {})

