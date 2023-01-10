import sys
_module = sys.modules[__name__]
del sys
dataset = _module
davis_test_dataset = _module
range_transform = _module
reseed = _module
static_dataset = _module
tps = _module
util = _module
vos_dataset = _module
yv_test_dataset = _module
download_bl30k = _module
download_datasets = _module
download_model = _module
eval_davis = _module
eval_davis_2016 = _module
eval_youtube = _module
inference_core = _module
inference_core_yv = _module
model = _module
aggregate = _module
corr_network = _module
eval_network = _module
losses = _module
mod_resnet = _module
model = _module
modules = _module
network = _module
scripts = _module
resize_length = _module
resize_youtube = _module
train = _module
try_correspondence = _module
hyper_para = _module
image_saver = _module
load_subset = _module
log_integrator = _module
logger = _module
tensor_util = _module

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


import numpy as np


import torch


import torch.nn.functional as F


from torchvision import transforms


from torch.utils.data.dataset import Dataset


import random


import time


from collections import defaultdict


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


import math


import torch.nn as nn


from collections import OrderedDict


from torch.utils import model_zoo


import torch.optim as optim


from torchvision import models


from torch.utils.data import ConcatDataset


import torch.distributed as distributed


import warnings


import torchvision.transforms as transforms


from torch.utils.tensorboard import SummaryWriter


class AttentionMemory(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, Mk, Qk):
        B, CK, H, W = Mk.shape
        Mk = Mk.view(B, CK, H * W)
        Mk = torch.transpose(Mk, 1, 2)
        Qk = Qk.view(B, CK, H * W).expand(B, -1, -1) / math.sqrt(CK)
        affinity = torch.bmm(Mk, Qk)
        affinity = F.softmax(affinity, dim=1)
        return affinity

    def readout(self, affinity, mv):
        B, CV, H, W = mv.shape
        mv = mv.flatten(start_dim=2)
        readout = torch.bmm(mv, affinity)
        readout = readout.view(B, CV, H, W)
        return readout


class KeyValue(nn.Module):

    def __init__(self, indim, keydim, valdim):
        super().__init__()
        self.key_proj = nn.Conv2d(indim, keydim, kernel_size=3, padding=1)
        self.val_proj = nn.Conv2d(indim, valdim, kernel_size=3, padding=1)

    def forward(self, x):
        return self.key_proj(x), self.val_proj(x)


class MaskRGBEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        resnet = mod_resnet.resnet50(pretrained=True, extra_chan=2)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3

    def forward(self, f, m, o):
        f = torch.cat([f, m, o], 1)
        x = self.conv1(f)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class RGBEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.res2 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3

    def forward(self, f):
        x = self.conv1(f)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        f4 = self.res2(x)
        f8 = self.layer2(f4)
        f16 = self.layer3(f8)
        return f16, f8, f4


class CorrespondenceNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.mask_rgb_encoder = MaskRGBEncoder()
        self.rgb_encoder = RGBEncoder()
        self.kv_m_f16 = KeyValue(1024, keydim=128, valdim=512)
        self.kv_q_f16 = KeyValue(1024, keydim=128, valdim=512)
        self.attn_memory = AttentionMemory()

    def get_query_key(self, frame):
        f16, _, _ = self.rgb_encoder(frame)
        k16, _ = self.kv_q_f16(f16)
        return k16

    def get_mem_key(self, frame, mask, mask2):
        f16 = self.mask_rgb_encoder(frame, mask, mask2)
        k16, _ = self.kv_m_f16(f16)
        return k16

    def get_W(self, mk16, qk16):
        W = self.attn_memory(mk16, qk16)
        return W

    def transfer(self, W, val):
        return self.attn_memory.readout(W, val)


def make_gaussian(y_idx, x_idx, height, width, sigma=7):
    yv, xv = torch.meshgrid([torch.arange(0, height), torch.arange(0, width)])
    yv = yv.reshape(height * width).unsqueeze(0).float()
    xv = xv.reshape(height * width).unsqueeze(0).float()
    y_idx = y_idx.transpose(0, 1)
    x_idx = x_idx.transpose(0, 1)
    g = torch.exp(-((yv - y_idx) ** 2 + (xv - x_idx) ** 2) / (2 * sigma ** 2))
    return g


def softmax_w_g_top(x, top=None, gauss=None):
    if top is not None:
        if gauss is not None:
            maxes = torch.max(x, dim=1, keepdim=True)[0]
            x_exp = torch.exp(x - maxes) * gauss
            x_exp, indices = torch.topk(x_exp, k=top, dim=1)
        else:
            values, indices = torch.topk(x, k=top, dim=1)
            x_exp = torch.exp(values - values[:, 0])
        x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
        x_exp /= x_exp_sum
        x.zero_().scatter_(1, indices, x_exp)
        output = x
    else:
        maxes = torch.max(x, dim=1, keepdim=True)[0]
        if gauss is not None:
            x_exp = torch.exp(x - maxes) * gauss
        x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
        x_exp /= x_exp_sum
        output = x_exp
    return output


class EvalMemoryReader(nn.Module):

    def __init__(self, top_k, km):
        super().__init__()
        self.top_k = top_k
        self.km = km

    def forward(self, mk, mv, qk):
        B, CK, T, H, W = mk.shape
        _, CV, _, _, _ = mv.shape
        mi = mk.view(B, CK, T * H * W).transpose(1, 2)
        qi = qk.view(1, CK, H * W).expand(B, -1, -1) / math.sqrt(CK)
        affinity = torch.bmm(mi, qi)
        if self.km is not None:
            argmax_idx = affinity.max(2)[1]
            y_idx, x_idx = argmax_idx // W, argmax_idx % W
            g = make_gaussian(y_idx, x_idx, H, W, sigma=self.km)
            g = g.view(B, T * H * W, H * W)
            affinity = softmax_w_g_top(affinity, top=self.top_k, gauss=g)
        elif self.top_k is not None:
            affinity = softmax_w_g_top(affinity, top=self.top_k, gauss=None)
        else:
            affinity = F.softmax(affinity, dim=1)
        mv = mv.view(B, CV, T * H * W)
        mem = torch.bmm(mv, affinity)
        mem = mem.view(B, CV, H, W)
        return mem


class ResBlock(nn.Module):

    def __init__(self, indim, outdim=None):
        super(ResBlock, self).__init__()
        if outdim == None:
            outdim = indim
        if indim == outdim:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(indim, outdim, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)

    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))
        if self.downsample is not None:
            x = self.downsample(x)
        return x + r


class UpsampleBlock(nn.Module):

    def __init__(self, skip_c, up_c, out_c, scale_factor=2):
        super().__init__()
        self.skip_conv1 = nn.Conv2d(skip_c, up_c, kernel_size=3, padding=1)
        self.skip_conv2 = ResBlock(up_c, up_c)
        self.out_conv = ResBlock(up_c, out_c)
        self.scale_factor = scale_factor

    def forward(self, skip_f, up_f):
        x = self.skip_conv2(self.skip_conv1(skip_f))
        x = x + F.interpolate(up_f, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        x = self.out_conv(x)
        return x


class Decoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.compress = ResBlock(1024, 512)
        self.up_16_8 = UpsampleBlock(512, 512, 256)
        self.up_8_4 = UpsampleBlock(256, 256, 256)
        self.pred = nn.Conv2d(256, 1, kernel_size=(3, 3), padding=(1, 1), stride=1)

    def forward(self, f16, f8, f4):
        x = self.compress(f16)
        x = self.up_16_8(f8, x)
        x = self.up_8_4(f4, x)
        x = self.pred(F.relu(x))
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        return x


class MaskRGBEncoderSO(nn.Module):

    def __init__(self):
        super().__init__()
        resnet = mod_resnet.resnet50(pretrained=True, extra_chan=1)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3

    def forward(self, f, m):
        f = torch.cat([f, m], 1)
        x = self.conv1(f)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class MemoryReader(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, mk, mv, qk, qv):
        B, CK, T, H, W = mk.shape
        _, CV, _, _, _ = mv.shape
        mi = mk.view(B, CK, T * H * W)
        mi = torch.transpose(mi, 1, 2)
        qi = qk.view(B, CK, H * W) / math.sqrt(CK)
        affinity = torch.bmm(mi, qi)
        affinity = F.softmax(affinity, dim=1)
        mv = mv.view(B, CV, T * H * W)
        mem = torch.bmm(mv, affinity)
        mem = mem.view(B, CV, H, W)
        mem_out = torch.cat([mem, qv], dim=1)
        return mem_out


class PropagationNetwork(nn.Module):

    def __init__(self, single_object):
        super().__init__()
        self.single_object = single_object
        if single_object:
            self.mask_rgb_encoder = MaskRGBEncoderSO()
        else:
            self.mask_rgb_encoder = MaskRGBEncoder()
        self.rgb_encoder = RGBEncoder()
        self.kv_m_f16 = KeyValue(1024, keydim=128, valdim=512)
        self.kv_q_f16 = KeyValue(1024, keydim=128, valdim=512)
        self.memory = MemoryReader()
        self.decoder = Decoder()

    def aggregate(self, prob):
        new_prob = torch.cat([torch.prod(1 - prob, dim=1, keepdim=True), prob], 1).clamp(1e-07, 1 - 1e-07)
        logits = torch.log(new_prob / (1 - new_prob))
        return logits

    def memorize(self, frame, mask, other_mask=None):
        if self.single_object:
            f16 = self.mask_rgb_encoder(frame, mask)
        else:
            f16 = self.mask_rgb_encoder(frame, mask, other_mask)
        k16, v16 = self.kv_m_f16(f16)
        return k16.unsqueeze(2), v16.unsqueeze(2)

    def segment(self, frame, keys, values, selector=None):
        b, k = keys.shape[:2]
        f16, f8, f4 = self.rgb_encoder(frame)
        k16, v16 = self.kv_q_f16(f16)
        if self.single_object:
            logits = self.decoder(self.memory(keys, values, k16, v16), f8, f4)
            prob = torch.sigmoid(logits)
        else:
            logits = torch.cat([self.decoder(self.memory(keys[:, 0], values[:, 0], k16, v16), f8, f4), self.decoder(self.memory(keys[:, 1], values[:, 1], k16, v16), f8, f4)], 1)
            prob = torch.sigmoid(logits)
            prob = prob * selector.unsqueeze(2).unsqueeze(2)
        logits = self.aggregate(prob)
        prob = F.softmax(logits, dim=1)[:, 1:]
        return logits, prob

    def forward(self, *args, **kwargs):
        if args[1].dim() > 4:
            return self.segment(*args, **kwargs)
        else:
            return self.memorize(*args, **kwargs)


class BootstrappedCE(nn.Module):

    def __init__(self, start_warm=20000, end_warm=70000, top_p=0.15):
        super().__init__()
        self.start_warm = start_warm
        self.end_warm = end_warm
        self.top_p = top_p

    def forward(self, input, target, it):
        if it < self.start_warm:
            return F.cross_entropy(input, target), 1.0
        raw_loss = F.cross_entropy(input, target, reduction='none').view(-1)
        num_pixels = raw_loss.numel()
        if it > self.end_warm:
            this_p = self.top_p
        else:
            this_p = self.top_p + (1 - self.top_p) * ((self.end_warm - it) / (self.end_warm - self.start_warm))
        loss, _ = torch.topk(raw_loss, int(num_pixels * this_p), sorted=False)
        return loss.mean(), this_p


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride=1, dilation=dilation)
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

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1)
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
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers=(3, 4, 23, 3), extra_chan=1):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3 + extra_chan, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride), nn.BatchNorm2d(planes * block.expansion))
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AttentionMemory,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BootstrappedCE,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), 0], {}),
     False),
    (Decoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1024, 64, 64]), torch.rand([4, 512, 128, 128]), torch.rand([4, 256, 256, 256])], {}),
     False),
    (KeyValue,
     lambda: ([], {'indim': 4, 'keydim': 4, 'valdim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MemoryReader,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4, 4]), torch.rand([4, 4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (RGBEncoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (ResBlock,
     lambda: ([], {'indim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (UpsampleBlock,
     lambda: ([], {'skip_c': 4, 'up_c': 4, 'out_c': 4}),
     lambda: ([torch.rand([4, 4, 16, 16]), torch.rand([4, 4, 8, 8])], {}),
     False),
]

class Test_hkchengrex_Mask_Propagation(_paritybench_base):
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

