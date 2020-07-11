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


import torch.nn as nn


import torch.nn.functional as F


from torch.autograd import Variable


import math


import torch.utils.model_zoo as model_zoo


import numpy as np


import logging


class ResDownS(nn.Module):

    def __init__(self, inplane, outplane):
        super(ResDownS, self).__init__()
        self.downsample = nn.Sequential(nn.Conv2d(inplane, outplane, kernel_size=1, bias=False), nn.BatchNorm2d(outplane))

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
        self.v0 = nn.Sequential(nn.Conv2d(64, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 4, 3, padding=1), nn.ReLU())
        self.v1 = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 16, 3, padding=1), nn.ReLU())
        self.v2 = nn.Sequential(nn.Conv2d(512, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 32, 3, padding=1), nn.ReLU())
        self.h2 = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())
        self.h1 = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 16, 3, padding=1), nn.ReLU())
        self.h0 = nn.Sequential(nn.Conv2d(4, 4, 3, padding=1), nn.ReLU(), nn.Conv2d(4, 4, 3, padding=1), nn.ReLU())
        self.deconv = nn.ConvTranspose2d(256, 32, 15, 15)
        self.post0 = nn.Conv2d(32, 16, 3, padding=1)
        self.post1 = nn.Conv2d(16, 4, 3, padding=1)
        self.post2 = nn.Conv2d(4, 1, 3, padding=1)

    def forward(self, f, corr_feature, pos=None):
        p0 = torch.nn.functional.pad(f[0], [16, 16, 16, 16])[:, :, 4 * pos[0]:4 * pos[0] + 61, 4 * pos[1]:4 * pos[1] + 61]
        p1 = torch.nn.functional.pad(f[1], [8, 8, 8, 8])[:, :, 2 * pos[0]:2 * pos[0] + 31, 2 * pos[1]:2 * pos[1] + 31]
        p2 = torch.nn.functional.pad(f[2], [4, 4, 4, 4])[:, :, pos[0]:pos[0] + 15, pos[1]:pos[1] + 15]
        p3 = corr_feature[:, :, (pos[0]), (pos[1])].view(-1, 256, 1, 1)
        out = self.deconv(p3)
        out = self.post0(F.upsample(self.h2(out) + self.v2(p2), size=(31, 31)))
        out = self.post1(F.upsample(self.h1(out) + self.v1(p1), size=(61, 61)))
        out = self.post2(F.upsample(self.h0(out) + self.v0(p0), size=(127, 127)))
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
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


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


class Bottleneck(Features):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        padding = 2 - stride
        assert stride == 1 or dilation == 1, 'stride and dilation must have one equals to zero at least'
        if dilation > 1:
            padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=padding, bias=False, dilation=dilation)
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
            None
        out += residual
        out = self.relu(out)
        return out


class Bottleneck_nop(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck_nop, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=0, bias=False)
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
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.feature_size = 128 * block.expansion
        if layer3:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
            self.feature_size = (256 + 128) * block.expansion
        else:
            self.layer3 = lambda x: x
        if layer4:
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
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
                downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
            else:
                if dilation > 1:
                    dd = dilation // 2
                    padding = dd
                else:
                    dd = 1
                    padding = 0
                downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=3, stride=stride, bias=False, padding=padding, dilation=dd), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dilation=dd))
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


class ResAdjust(nn.Module):

    def __init__(self, block=Bottleneck, out_channels=256, adjust_number=1, fuse_layers=[2, 3, 4]):
        super(ResAdjust, self).__init__()
        self.fuse_layers = set(fuse_layers)
        if 2 in self.fuse_layers:
            self.layer2 = self._make_layer(block, 128, 1, out_channels, adjust_number)
        if 3 in self.fuse_layers:
            self.layer3 = self._make_layer(block, 256, 2, out_channels, adjust_number)
        if 4 in self.fuse_layers:
            self.layer4 = self._make_layer(block, 512, 4, out_channels, adjust_number)
        self.feature_size = out_channels * len(self.fuse_layers)

    def _make_layer(self, block, plances, dilation, out, number=1):
        layers = []
        for _ in range(number):
            layer = block(plances * block.expansion, plances, dilation=dilation)
            layers.append(layer)
        downsample = nn.Sequential(nn.Conv2d(plances * block.expansion, out, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(out))
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
        self.conv_kernel = nn.Sequential(nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False), nn.BatchNorm2d(hidden), nn.ReLU(inplace=True))
        self.conv_search = nn.Sequential(nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False), nn.BatchNorm2d(hidden), nn.ReLU(inplace=True))
        self.head = nn.Sequential(nn.Conv2d(hidden, hidden, kernel_size=1, bias=False), nn.BatchNorm2d(hidden), nn.ReLU(inplace=True), nn.Conv2d(hidden, out_channels, kernel_size=1))

    def forward_corr(self, kernel, input):
        kernel = self.conv_kernel(kernel)
        input = self.conv_search(input)
        feature = conv2d_dw_group(input, kernel)
        return feature

    def forward(self, kernel, search):
        feature = self.forward_corr(kernel, search)
        out = self.head(feature)
        return out


def center2corner(center):
    """
    :param center: Center or np.array 4*N
    :return: Corner or np.array 4*N
    """
    if isinstance(center, Center):
        x, y, w, h = center
        return Corner(x - w * 0.5, y - h * 0.5, x + w * 0.5, y + h * 0.5)
    else:
        x, y, w, h = center[0], center[1], center[2], center[3]
        x1 = x - w * 0.5
        y1 = y - h * 0.5
        x2 = x + w * 0.5
        y2 = y + h * 0.5
        return x1, y1, x2, y2


def corner2center(corner):
    """
    :param corner: Corner or np.array 4*N
    :return: Center or 4 np.array N
    """
    if isinstance(corner, Corner):
        x1, y1, x2, y2 = corner
        return Center((x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1)
    else:
        x1, y1, x2, y2 = corner[0], corner[1], corner[2], corner[3]
        x = (x1 + x2) * 0.5
        y = (y1 + y2) * 0.5
        w = x2 - x1
        h = y2 - y1
        return x, y, w, h


class Anchors:

    def __init__(self, cfg):
        self.stride = 8
        self.ratios = [0.33, 0.5, 1, 2, 3]
        self.scales = [8]
        self.round_dight = 0
        self.image_center = 0
        self.size = 0
        self.__dict__.update(cfg)
        self.anchor_num = len(self.scales) * len(self.ratios)
        self.anchors = None
        self.all_anchors = None
        self.generate_anchors()

    def generate_anchors(self):
        self.anchors = np.zeros((self.anchor_num, 4), dtype=np.float32)
        size = self.stride * self.stride
        count = 0
        for r in self.ratios:
            if self.round_dight > 0:
                ws = round(math.sqrt(size * 1.0 / r), self.round_dight)
                hs = round(ws * r, self.round_dight)
            else:
                ws = int(math.sqrt(size * 1.0 / r))
                hs = int(ws * r)
            for s in self.scales:
                w = ws * s
                h = hs * s
                self.anchors[count][:] = [-w * 0.5, -h * 0.5, w * 0.5, h * 0.5][:]
                count += 1

    def generate_all_anchors(self, im_c, size):
        if self.image_center == im_c and self.size == size:
            return False
        self.image_center = im_c
        self.size = size
        a0x = im_c - size // 2 * self.stride
        ori = np.array([a0x] * 4, dtype=np.float32)
        zero_anchors = self.anchors + ori
        x1 = zero_anchors[:, (0)]
        y1 = zero_anchors[:, (1)]
        x2 = zero_anchors[:, (2)]
        y2 = zero_anchors[:, (3)]
        x1, y1, x2, y2 = map(lambda x: x.reshape(self.anchor_num, 1, 1), [x1, y1, x2, y2])
        cx, cy, w, h = corner2center([x1, y1, x2, y2])
        disp_x = np.arange(0, size).reshape(1, 1, -1) * self.stride
        disp_y = np.arange(0, size).reshape(1, -1, 1) * self.stride
        cx = cx + disp_x
        cy = cy + disp_y
        zero = np.zeros((self.anchor_num, size, size), dtype=np.float32)
        cx, cy, w, h = map(lambda x: x + zero, [cx, cy, w, h])
        x1, y1, x2, y2 = center2corner([cx, cy, w, h])
        self.all_anchors = np.stack([x1, y1, x2, y2]), np.stack([cx, cy, w, h])
        return True


class SiamMask(nn.Module):

    def __init__(self, anchors=None, o_sz=127, g_sz=127):
        super(SiamMask, self).__init__()
        self.anchors = anchors
        self.anchor_num = len(self.anchors['ratios']) * len(self.anchors['scales'])
        self.anchor = Anchors(anchors)
        self.features = None
        self.rpn_model = None
        self.mask_model = None
        self.o_sz = o_sz
        self.g_sz = g_sz
        self.all_anchors = None

    def set_all_anchors(self, image_center, size):
        if not self.anchor.generate_all_anchors(image_center, size):
            return
        all_anchors = self.anchor.all_anchors[1]
        self.all_anchors = torch.from_numpy(all_anchors).float()
        self.all_anchors = [self.all_anchors[i] for i in range(4)]

    def feature_extractor(self, x):
        return self.features(x)

    def rpn(self, template, search):
        pred_cls, pred_loc = self.rpn_model(template, search)
        return pred_cls, pred_loc

    def mask(self, template, search):
        pred_mask = self.mask_model(template, search)
        return pred_mask

    def template(self, z):
        self.zf = self.feature_extractor(z)
        cls_kernel, loc_kernel = self.rpn_model.template(self.zf)
        return cls_kernel, loc_kernel

    def track(self, x, cls_kernel=None, loc_kernel=None, softmax=False):
        xf = self.feature_extractor(x)
        rpn_pred_cls, rpn_pred_loc = self.rpn_model.track(xf, cls_kernel, loc_kernel)
        if softmax:
            rpn_pred_cls = self.softmax(rpn_pred_cls)
        return rpn_pred_cls, rpn_pred_loc


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DepthCorr,
     lambda: ([], {'in_channels': 4, 'hidden': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResAdjust,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 512, 64, 64]), torch.rand([4, 1024, 64, 64]), torch.rand([4, 2048, 64, 64])], {}),
     False),
    (ResDownS,
     lambda: ([], {'inplane': 4, 'outplane': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_alexandre01_UltimateLabeling(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

