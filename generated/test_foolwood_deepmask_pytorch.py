import sys
_module = sys.modules[__name__]
del sys
loader = _module
coco_loader = _module
pycocotools = _module
coco = _module
cocoeval = _module
mask = _module
setup = _module
DeepMask = _module
SharpMask = _module
models = _module
InferDeepMask = _module
computeProposals = _module
dasiamrpn_deepmask = _module
evalPerImage = _module
train = _module
yolo_deepmask = _module
utils = _module
load_helper = _module
log_helper = _module

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


import numpy as np


import random


from collections import namedtuple


from torch.utils import data


import torch


import torch.nn as nn


import logging


import torch.backends.cudnn as cudnn


from torch.optim.lr_scheduler import MultiStepLR


class Reshape(nn.Module):

    def __init__(self, oSz):
        super(Reshape, self).__init__()
        self.oSz = oSz

    def forward(self, x):
        b = x.shape[0]
        return x.permute(0, 2, 3, 1).view(b, -1, self.oSz, self.oSz)


class SymmetricPad2d(nn.Module):

    def __init__(self, padding):
        super(SymmetricPad2d, self).__init__()
        self.padding = padding
        try:
            self.pad_l, self.pad_b, self.pad_r, self.pad_t = padding
        except:
            self.pad_l, self.pad_b, self.pad_r, self.pad_t = [padding] * 4

    def forward(self, input):
        assert len(input.shape) == 4, 'only Dimension=4 implemented'
        h = input.shape[2] + self.pad_t + self.pad_b
        w = input.shape[3] + self.pad_l + self.pad_r
        assert w >= 1 and h >= 1, 'input is too small'
        output = torch.zeros(input.shape[0], input.shape[1], h, w).to(input
            .device)
        c_input = input
        if self.pad_t < 0:
            c_input = c_input.narrow(2, -self.pad_t, c_input.shape[2] +
                self.pad_t)
        if self.pad_b < 0:
            c_input = c_input.narrow(2, 0, c_input.shape[2] + self.pad_b)
        if self.pad_l < 0:
            c_input = c_input.narrow(3, -self.pad_l, c_input.shape[3] +
                self.pad_l)
        if self.pad_r < 0:
            c_input = c_input.narrow(3, 0, c_input.shape[3] + self.pad_r)
        c_output = output
        if self.pad_t > 0:
            c_output = c_output.narrow(2, self.pad_t, c_output.shape[2] -
                self.pad_t)
        if self.pad_b > 0:
            c_output = c_output.narrow(2, 0, c_output.shape[2] - self.pad_b)
        if self.pad_l > 0:
            c_output = c_output.narrow(3, self.pad_l, c_output.shape[3] -
                self.pad_l)
        if self.pad_r > 0:
            c_output = c_output.narrow(3, 0, c_output.shape[3] - self.pad_r)
        c_output.copy_(c_input)
        assert w >= 2 * self.pad_l and w >= 2 * self.pad_r and h >= 2 * self.pad_t and h >= 2 * self.pad_b
        """input is too small"""
        for i in range(self.pad_t):
            output.narrow(2, self.pad_t - i - 1, 1).copy_(output.narrow(2, 
                self.pad_t + i, 1))
        for i in range(self.pad_b):
            output.narrow(2, output.shape[2] - self.pad_b + i, 1).copy_(output
                .narrow(2, output.shape[2] - self.pad_b - i - 1, 1))
        for i in range(self.pad_l):
            output.narrow(3, self.pad_l - i - 1, 1).copy_(output.narrow(3, 
                self.pad_l + i, 1))
        for i in range(self.pad_r):
            output.narrow(3, output.shape[3] - self.pad_r + i, 1).copy_(output
                .narrow(3, output.shape[3] - self.pad_r - i - 1, 1))
        return output


Config = namedtuple('Config', ['iSz', 'oSz', 'gSz', 'km', 'ks'])


default_config = Config(iSz=160, oSz=56, gSz=160, km=32, ks=32)


def updatePadding(net, nn_padding):
    typename = torch.typename(net)
    if typename.find('Sequential') >= 0 or typename.find('Bottleneck') >= 0:
        modules_keys = list(net._modules.keys())
        for i in reversed(range(len(modules_keys))):
            subnet = net._modules[modules_keys[i]]
            out = updatePadding(subnet, nn_padding)
            if out != -1:
                p = out
                in_c, out_c, k, s, _, d, g, b = (subnet.in_channels, subnet
                    .out_channels, subnet.kernel_size[0], subnet.stride[0],
                    subnet.padding[0], subnet.dilation[0], subnet.groups,
                    subnet.bias)
                conv_temple = nn.Conv2d(in_c, out_c, k, stride=s, padding=0,
                    dilation=d, groups=g, bias=b)
                conv_temple.weight = subnet.weight
                conv_temple.bias = subnet.bias
                if p > 1:
                    net._modules[modules_keys[i]] = nn.Sequential(
                        SymmetricPad2d(p), conv_temple)
                else:
                    net._modules[modules_keys[i]] = nn.Sequential(nn_padding
                        (p), conv_temple)
    elif typename.find('torch.nn.modules.conv.Conv2d') >= 0:
        k_sz, p_sz = net.kernel_size[0], net.padding[0]
        if (k_sz == 3 or k_sz == 7) and p_sz != 0:
            return p_sz
    return -1


class DeepMask(nn.Module):

    def __init__(self, config=default_config, context=True):
        super(DeepMask, self).__init__()
        self.config = config
        self.context = context
        self.strides = 16
        self.fSz = -(-self.config.iSz // self.strides)
        self.trunk = self.creatTrunk()
        updatePadding(self.trunk, nn.ReflectionPad2d)
        self.crop_trick = nn.ZeroPad2d(-16 // self.strides)
        self.maskBranch = self.createMaskBranch()
        self.scoreBranch = self.createScoreBranch()
        npt = sum(p.numel() for p in self.trunk.parameters()) / 1000000.0
        npm = sum(p.numel() for p in self.maskBranch.parameters()) / 1000000.0
        nps = sum(p.numel() for p in self.scoreBranch.parameters()) / 1000000.0
        None
        None
        None
        None

    def forward(self, x):
        feat = self.trunk(x)
        if self.context:
            feat = self.crop_trick(feat)
        mask = self.maskBranch(feat)
        score = self.scoreBranch(feat)
        return mask, score

    def creatTrunk(self):
        resnet50 = torchvision.models.resnet50(pretrained=True)
        trunk1 = nn.Sequential(*list(resnet50.children())[:-3])
        trunk2 = nn.Sequential(nn.Conv2d(1024, 128, 1), nn.ReLU(inplace=
            True), nn.Conv2d(128, 512, self.fSz))
        return nn.Sequential(trunk1, trunk2)

    def createMaskBranch(self):
        maskBranch = nn.Sequential(nn.Conv2d(512, self.config.oSz ** 2, 1),
            Reshape(self.config.oSz))
        if self.config.gSz > self.config.oSz:
            upSample = nn.UpsamplingBilinear2d(size=[self.config.gSz, self.
                config.gSz])
            maskBranch = nn.Sequential(maskBranch, upSample)
        return maskBranch

    def createScoreBranch(self):
        scoreBranch = nn.Sequential(nn.Dropout(0.5), nn.Conv2d(512, 1024, 1
            ), nn.Threshold(0, 1e-06), nn.Dropout(0.5), nn.Conv2d(1024, 1, 1))
        return scoreBranch


class RefineModule(nn.Module):

    def __init__(self, l1, l2, l3):
        super(RefineModule, self).__init__()
        self.layer1 = l1
        self.layer2 = l2
        self.layer3 = l3

    def forward(self, x):
        x1 = self.layer1(x[0])
        x2 = self.layer2(x[1])
        y = x1 + x2
        y = self.layer3(y)
        return y


logger = logging.getLogger('global')


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    logger.info('missing keys:{}'.format(len(missing_keys)))
    logger.info('unused checkpoint keys:{}'.format(len(unused_pretrained_keys))
        )
    logger.info('used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys
        ) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    """ Old style model is stored with all names of parameters share common prefix 'module.' """
    logger.info("remove prefix '{}'".format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_pretrain(model, pretrained_path):
    logger.info('load pretrained model from {}'.format(pretrained_path))
    device = torch.cuda.current_device()
    pretrained_dict = torch.load(pretrained_path, map_location=lambda
        storage, loc: storage.cuda(device))
    if 'state_dict' in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'],
            'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


class SharpMask(nn.Module):

    def __init__(self, config=default_config, context=True):
        super(SharpMask, self).__init__()
        self.context = context
        self.km, self.ks = config.km, config.ks
        self.skpos = [6, 5, 4, 2]
        deepmask = DeepMask(config)
        deeomask_resume = os.path.join('exps', 'deepmask', 'train',
            'model_best.pth.tar')
        assert os.path.exists(deeomask_resume), 'Please train DeepMask first'
        deepmask = load_pretrain(deepmask, deeomask_resume)
        self.trunk = deepmask.trunk
        self.crop_trick = deepmask.crop_trick
        self.scoreBranch = deepmask.scoreBranch
        self.maskBranchDM = deepmask.maskBranch
        self.fSz = deepmask.fSz
        self.refs = self.createTopDownRefinement()
        nph = sum(p.numel() for h in self.neths for p in h.parameters()
            ) / 1000000.0
        npv = sum(p.numel() for h in self.netvs for p in h.parameters()
            ) / 1000000.0
        None
        None
        None

    def forward(self, x):
        inps = list()
        for i, l in enumerate(self.trunk.children()):
            for j, ll in enumerate(l.children()):
                x = ll(x)
                if i == 0 and j == len(l) - 1 and self.context:
                    x = self.crop_trick(x)
                if i == 0 and j in self.skpos:
                    inps.append(x)
        currentOutput = self.refs[0](x)
        for k in range(len(self.refs) - 2):
            x_f = inps[-(k + 1)]
            currentOutput = self.refs[k + 1]((x_f, currentOutput))
        currentOutput = self.refs[-1](currentOutput)
        return currentOutput, self.scoreBranch(x)

    def train(self, mode=True):
        self.training = mode
        if mode:
            for module in self.children():
                module.train(False)
            for module in self.refs.children():
                module.train(mode)
        else:
            for module in self.children():
                module.train(mode)
        return self

    def createHorizontal(self):
        neths = nn.ModuleList()
        nhu1, nhu2, crop = 0, 0, 0
        for i in range(len(self.skpos)):
            h = []
            nInps = self.ks // 2 ** i
            if i == 0:
                nhu1, nhu2, crop = 1024, 64, 0 if self.context else 0
            elif i == 1:
                nhu1, nhu2, crop = 512, 64, -2 if self.context else 0
            elif i == 2:
                nhu1, nhu2, crop = 256, 64, -4 if self.context else 0
            elif i == 3:
                nhu1, nhu2, crop = 64, 64, -8 if self.context else 0
            if crop != 0:
                h.append(nn.ZeroPad2d(crop))
            h.append(nn.ReflectionPad2d(1))
            h.append(nn.Conv2d(nhu1, nhu2, 3))
            h.append(nn.ReLU(inplace=True))
            h.append(nn.ReflectionPad2d(1))
            h.append(nn.Conv2d(nhu2, nInps, 3))
            h.append(nn.ReLU(inplace=True))
            h.append(nn.ReflectionPad2d(1))
            h.append(nn.Conv2d(nInps, nInps // 2, 3))
            neths.append(nn.Sequential(*h))
        return neths

    def createVertical(self):
        netvs = nn.ModuleList()
        netvs.append(nn.ConvTranspose2d(512, self.km, self.fSz))
        for i in range(len(self.skpos)):
            netv = []
            nInps = self.km // 2 ** i
            netv.append(nn.ReflectionPad2d(1))
            netv.append(nn.Conv2d(nInps, nInps, 3))
            netv.append(nn.ReLU(inplace=True))
            netv.append(nn.ReflectionPad2d(1))
            netv.append(nn.Conv2d(nInps, nInps // 2, 3))
            netvs.append(nn.Sequential(*netv))
        return netvs

    def refinement(self, neth, netv):
        return RefineModule(neth, netv, nn.Sequential(nn.ReLU(inplace=True),
            nn.UpsamplingNearest2d(scale_factor=2)))

    def createTopDownRefinement(self):
        self.neths = self.createHorizontal()
        self.netvs = self.createVertical()
        refs = nn.ModuleList()
        refs.append(self.netvs[0])
        for i in range(len(self.skpos)):
            refs.append(self.refinement(self.neths[i], self.netvs[i + 1]))
        refs.append(nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(self.km //
            2 ** (len(refs) - 1), 1, 3)))
        return refs


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_foolwood_deepmask_pytorch(_paritybench_base):
    pass

    def test_000(self):
        self._check(Reshape(*[], **{'oSz': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(SymmetricPad2d(*[], **{'padding': 4}), [torch.rand([4, 4, 4, 4])], {})
