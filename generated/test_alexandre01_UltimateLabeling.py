import sys
_module = sys.modules[__name__]
del sys
setup = _module
test_polygon = _module
ultimatelabeling = _module
class_names = _module
config = _module
main = _module
models = _module
detector = _module
hungarian_tracker = _module
keyboard_listener = _module
polygon = _module
ssh_credentials = _module
state = _module
track_info = _module
tracker = _module
siamMask = _module
custom = _module
features = _module
mask = _module
resnet = _module
rpn = _module
siammask = _module
test = _module
utils = _module
anchors = _module
bbox_helper = _module
benchmark_helper = _module
config_helper = _module
load_helper = _module
log_helper = _module
tracker_config = _module
styles = _module
palettes = _module
theme = _module
views = _module
class_editor = _module
detection_manager = _module
hungarian_manager = _module
image_viewer = _module
info_detection = _module
io = _module
message_progress = _module
options = _module
player = _module
slider = _module
ssh_login = _module
theme_picker = _module
tracking_manager = _module
video_list = _module

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


from torch.autograd import Variable


import math


import torch.utils.model_zoo as model_zoo


import numpy as np


class ResDownS(nn.Module):

    def __init__(self, inplane, outplane):
        super(ResDownS, self).__init__()
        self.downsample = nn.Sequential(nn.Conv2d(inplane, outplane,
            kernel_size=1, bias=False), nn.BatchNorm2d(outplane))

    def forward(self, x):
        x = self.downsample(x)
        if x.size(3) < 20:
            l, r = 4, -4
            x = x[:, :, l:r, l:r]
        return x


class Refine(nn.Module):

    def __init__(self):
        """
        Mask refinement module
        Please refer SiamMask (Appendix A)
        https://arxiv.org/abs/1812.05050
        """
        super(Refine, self).__init__()
        self.v0 = nn.Sequential(nn.Conv2d(64, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 4, 3, padding=1), nn.ReLU())
        self.v1 = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 16, 3, padding=1), nn.ReLU())
        self.v2 = nn.Sequential(nn.Conv2d(512, 128, 3, padding=1), nn.ReLU(
            ), nn.Conv2d(128, 32, 3, padding=1), nn.ReLU())
        self.h2 = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())
        self.h1 = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU())
        self.h0 = nn.Sequential(nn.Conv2d(4, 4, 3, padding=1), nn.ReLU(),
            nn.Conv2d(4, 4, 3, padding=1), nn.ReLU())
        self.deconv = nn.ConvTranspose2d(256, 32, 15, 15)
        self.post0 = nn.Conv2d(32, 16, 3, padding=1)
        self.post1 = nn.Conv2d(16, 4, 3, padding=1)
        self.post2 = nn.Conv2d(4, 1, 3, padding=1)

    def forward(self, f, corr_feature, pos=None):
        p0 = torch.nn.functional.pad(f[0], [16, 16, 16, 16])[:, :, 4 * pos[
            0]:4 * pos[0] + 61, 4 * pos[1]:4 * pos[1] + 61]
        p1 = torch.nn.functional.pad(f[1], [8, 8, 8, 8])[:, :, 2 * pos[0]:2 *
            pos[0] + 31, 2 * pos[1]:2 * pos[1] + 31]
        p2 = torch.nn.functional.pad(f[2], [4, 4, 4, 4])[:, :, pos[0]:pos[0
            ] + 15, pos[1]:pos[1] + 15]
        p3 = corr_feature[:, :, (pos[0]), (pos[1])].view(-1, 256, 1, 1)
        out = self.deconv(p3)
        out = self.post0(F.upsample(self.h2(out) + self.v2(p2), size=(31, 31)))
        out = self.post1(F.upsample(self.h1(out) + self.v1(p1), size=(61, 61)))
        out = self.post2(F.upsample(self.h0(out) + self.v0(p0), size=(127, 
            127)))
        out = out.view(-1, 127 * 127)
        return out


class Features(nn.Module):

    def __init__(self):
        super(Features, self).__init__()
        self.feature_size = -1

    def forward(self, x):
        raise NotImplementedError


class Mask(nn.Module):

    def __init__(self):
        super(Mask, self).__init__()

    def forward(self, z_f, x_f):
        raise NotImplementedError

    def template(self, template):
        raise NotImplementedError

    def track(self, search):
        raise NotImplementedError


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


class Bottleneck_nop(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck_nop, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
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
        s = residual.size(3)
        residual = residual[:, :, 1:s - 1, 1:s - 1]
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, layer4=False, layer3=False):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=0,
            bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.feature_size = 128 * block.expansion
        if layer3:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                dilation=2)
            self.feature_size = (256 + 128) * block.expansion
        else:
            self.layer3 = lambda x: x
        if layer4:
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                dilation=4)
            self.feature_size = 512 * block.expansion
        else:
            self.layer4 = lambda x: x
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        dd = dilation
        if stride != 1 or self.inplanes != planes * block.expansion:
            if stride == 1 and dilation == 1:
                downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                    block.expansion, kernel_size=1, stride=stride, bias=
                    False), nn.BatchNorm2d(planes * block.expansion))
            else:
                if dilation > 1:
                    dd = dilation // 2
                    padding = dd
                else:
                    dd = 1
                    padding = 0
                downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                    block.expansion, kernel_size=3, stride=stride, bias=
                    False, padding=padding, dilation=dd), nn.BatchNorm2d(
                    planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
            dilation=dd))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        p0 = self.relu(x)
        x = self.maxpool(p0)
        p1 = self.layer1(x)
        p2 = self.layer2(p1)
        p3 = self.layer3(p2)
        return p0, p1, p2, p3


class Bottleneck(Features):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1
        ):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        padding = 2 - stride
        assert stride == 1 or dilation == 1, 'stride and dilation must have one equals to zero at least'
        if dilation > 1:
            padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
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
        if out.size() != residual.size():
            print(out.size(), residual.size())
        out += residual
        out = self.relu(out)
        return out


class ResAdjust(nn.Module):

    def __init__(self, block=Bottleneck, out_channels=256, adjust_number=1,
        fuse_layers=[2, 3, 4]):
        super(ResAdjust, self).__init__()
        self.fuse_layers = set(fuse_layers)
        if 2 in self.fuse_layers:
            self.layer2 = self._make_layer(block, 128, 1, out_channels,
                adjust_number)
        if 3 in self.fuse_layers:
            self.layer3 = self._make_layer(block, 256, 2, out_channels,
                adjust_number)
        if 4 in self.fuse_layers:
            self.layer4 = self._make_layer(block, 512, 4, out_channels,
                adjust_number)
        self.feature_size = out_channels * len(self.fuse_layers)

    def _make_layer(self, block, plances, dilation, out, number=1):
        layers = []
        for _ in range(number):
            layer = block(plances * block.expansion, plances, dilation=dilation
                )
            layers.append(layer)
        downsample = nn.Sequential(nn.Conv2d(plances * block.expansion, out,
            kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(out))
        layers.append(downsample)
        return nn.Sequential(*layers)

    def forward(self, p2, p3, p4):
        outputs = []
        if 2 in self.fuse_layers:
            outputs.append(self.layer2(p2))
        if 3 in self.fuse_layers:
            outputs.append(self.layer3(p3))
        if 4 in self.fuse_layers:
            outputs.append(self.layer4(p4))
        return outputs


class RPN(nn.Module):

    def __init__(self):
        super(RPN, self).__init__()

    def forward(self, z_f, x_f):
        raise NotImplementedError

    def template(self, template):
        raise NotImplementedError

    def track(self, search):
        raise NotImplementedError


def conv2d_dw_group(x, kernel):
    batch, channel = kernel.shape[:2]
    x = x.view(1, batch * channel, x.size(2), x.size(3))
    kernel = kernel.view(batch * channel, 1, kernel.size(2), kernel.size(3))
    out = F.conv2d(x, kernel, groups=batch * channel)
    out = out.view(batch, channel, out.size(2), out.size(3))
    return out


class DepthCorr(nn.Module):

    def __init__(self, in_channels, hidden, out_channels, kernel_size=3):
        super(DepthCorr, self).__init__()
        self.conv_kernel = nn.Sequential(nn.Conv2d(in_channels, hidden,
            kernel_size=kernel_size, bias=False), nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True))
        self.conv_search = nn.Sequential(nn.Conv2d(in_channels, hidden,
            kernel_size=kernel_size, bias=False), nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True))
        self.head = nn.Sequential(nn.Conv2d(hidden, hidden, kernel_size=1,
            bias=False), nn.BatchNorm2d(hidden), nn.ReLU(inplace=True), nn.
            Conv2d(hidden, out_channels, kernel_size=1))

    def forward_corr(self, kernel, input):
        kernel = self.conv_kernel(kernel)
        input = self.conv_search(input)
        feature = conv2d_dw_group(input, kernel)
        return feature

    def forward(self, kernel, search):
        feature = self.forward_corr(kernel, search)
        out = self.head(feature)
        return out


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_alexandre01_UltimateLabeling(_paritybench_base):
    pass
    def test_000(self):
        self._check(ResDownS(*[], **{'inplane': 4, 'outplane': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(BasicBlock(*[], **{'inplanes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_002(self):
        self._check(ResAdjust(*[], **{}), [torch.rand([4, 512, 64, 64]), torch.rand([4, 1024, 64, 64]), torch.rand([4, 2048, 64, 64])], {})

    def test_003(self):
        self._check(DepthCorr(*[], **{'in_channels': 4, 'hidden': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

