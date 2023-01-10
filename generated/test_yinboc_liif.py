import sys
_module = sys.modules[__name__]
del sys
datasets = _module
image_folder = _module
wrappers = _module
demo = _module
models = _module
edsr = _module
liif = _module
misc = _module
mlp = _module
rcan = _module
rdn = _module
resize = _module
test = _module
train_liif = _module
utils = _module

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


from torch.utils.data import Dataset


from torchvision import transforms


import functools


import random


import math


import torch.nn as nn


import torch.nn.functional as F


from functools import partial


from torch.utils.data import DataLoader


from torch.optim.lr_scheduler import MultiStepLR


import time


from torch.optim import SGD


from torch.optim import Adam


class MeanShift(nn.Conv2d):

    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False


class ResBlock(nn.Module):

    def __init__(self, conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


class Upsampler(nn.Sequential):

    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):
        m = []
        if scale & scale - 1 == 0:
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feat))
                if act:
                    m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feat))
            if act:
                m.append(act())
        else:
            raise NotImplementedError
        super(Upsampler, self).__init__(*m)


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=bias)


url = {'r16f64x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt', 'r16f64x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x3-abf2a44e.pt', 'r16f64x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x4-6b446fab.pt', 'r32f256x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x2-0edfb8a3.pt', 'r32f256x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x3-ea3ef2c6.pt', 'r32f256x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x4-4f62e9ef.pt'}


class EDSR(nn.Module):

    def __init__(self, args, conv=default_conv):
        super(EDSR, self).__init__()
        self.args = args
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale[0]
        act = nn.ReLU(True)
        url_name = 'r{}f{}x{}'.format(n_resblocks, n_feats, scale)
        if url_name in url:
            self.url = url[url_name]
        else:
            self.url = None
        self.sub_mean = MeanShift(args.rgb_range)
        self.add_mean = MeanShift(args.rgb_range, sign=1)
        m_head = [conv(args.n_colors, n_feats, kernel_size)]
        m_body = [ResBlock(conv, n_feats, kernel_size, act=act, res_scale=args.res_scale) for _ in range(n_resblocks)]
        m_body.append(conv(n_feats, n_feats, kernel_size))
        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        if args.no_upsampling:
            self.out_dim = n_feats
        else:
            self.out_dim = args.n_colors
            m_tail = [Upsampler(conv, scale, n_feats, act=False), conv(n_feats, args.n_colors, kernel_size)]
            self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x
        if self.args.no_upsampling:
            x = res
        else:
            x = self.tail(res)
        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, whose dimensions in the model are {} and whose dimensions in the checkpoint are {}.'.format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'.format(name))


def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + 2 * r * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


models = {}


datasets = {}


def register(name):

    def decorator(cls):
        datasets[name] = cls
        return cls
    return decorator


class LIIF(nn.Module):

    def __init__(self, encoder_spec, imnet_spec=None, local_ensemble=True, feat_unfold=True, cell_decode=True):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode
        self.encoder = models.make(encoder_spec)
        if imnet_spec is not None:
            imnet_in_dim = self.encoder.out_dim
            if self.feat_unfold:
                imnet_in_dim *= 9
            imnet_in_dim += 2
            if self.cell_decode:
                imnet_in_dim += 2
            self.imnet = models.make(imnet_spec, args={'in_dim': imnet_in_dim})
        else:
            self.imnet = None

    def gen_feat(self, inp):
        self.feat = self.encoder(inp)
        return self.feat

    def query_rgb(self, coord, cell=None):
        feat = self.feat
        if self.imnet is None:
            ret = F.grid_sample(feat, coord.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)
            return ret
        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).view(feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])
        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-06
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2
        feat_coord = make_coord(feat.shape[-2:], flatten=False).permute(2, 0, 1).unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])
        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-06, 1 - 1e-06)
                q_feat = F.grid_sample(feat, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)
                q_coord = F.grid_sample(feat_coord, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([q_feat, rel_coord], dim=-1)
                if self.cell_decode:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= feat.shape[-2]
                    rel_cell[:, :, 1] *= feat.shape[-1]
                    inp = torch.cat([inp, rel_cell], dim=-1)
                bs, q = coord.shape[:2]
                pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)
                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-09)
        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            t = areas[0]
            areas[0] = areas[3]
            areas[3] = t
            t = areas[1]
            areas[1] = areas[2]
            areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        return ret

    def forward(self, inp, coord, cell):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)


class MetaSR(nn.Module):

    def __init__(self, encoder_spec):
        super().__init__()
        self.encoder = models.make(encoder_spec)
        imnet_spec = {'name': 'mlp', 'args': {'in_dim': 3, 'out_dim': self.encoder.out_dim * 9 * 3, 'hidden_list': [256]}}
        self.imnet = models.make(imnet_spec)

    def gen_feat(self, inp):
        self.feat = self.encoder(inp)
        return self.feat

    def query_rgb(self, coord, cell=None):
        feat = self.feat
        feat = F.unfold(feat, 3, padding=1).view(feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])
        feat_coord = make_coord(feat.shape[-2:], flatten=False)
        feat_coord[:, :, 0] -= 2 / feat.shape[-2] / 2
        feat_coord[:, :, 1] -= 2 / feat.shape[-1] / 2
        feat_coord = feat_coord.permute(2, 0, 1).unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])
        coord_ = coord.clone()
        coord_[:, :, 0] -= cell[:, :, 0] / 2
        coord_[:, :, 1] -= cell[:, :, 1] / 2
        coord_q = (coord_ + 1e-06).clamp(-1 + 1e-06, 1 - 1e-06)
        q_feat = F.grid_sample(feat, coord_q.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)
        q_coord = F.grid_sample(feat_coord, coord_q.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)
        rel_coord = coord_ - q_coord
        rel_coord[:, :, 0] *= feat.shape[-2] / 2
        rel_coord[:, :, 1] *= feat.shape[-1] / 2
        r_rev = cell[:, :, 0] * (feat.shape[-2] / 2)
        inp = torch.cat([rel_coord, r_rev.unsqueeze(-1)], dim=-1)
        bs, q = coord.shape[:2]
        pred = self.imnet(inp.view(bs * q, -1)).view(bs * q, feat.shape[1], 3)
        pred = torch.bmm(q_feat.contiguous().view(bs * q, 1, -1), pred)
        pred = pred.view(bs, q, 3)
        return pred

    def forward(self, inp, coord, cell):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)


class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)


class CALayer(nn.Module):

    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True), nn.ReLU(inplace=True), nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True), nn.Sigmoid())

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class RCAB(nn.Module):

    def __init__(self, conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn:
                modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class ResidualGroup(nn.Module):

    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [RCAB(conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class RCAN(nn.Module):

    def __init__(self, args, conv=default_conv):
        super(RCAN, self).__init__()
        self.args = args
        n_resgroups = args.n_resgroups
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        reduction = args.reduction
        scale = args.scale[0]
        act = nn.ReLU(True)
        rgb_mean = 0.4488, 0.4371, 0.404
        rgb_std = 1.0, 1.0, 1.0
        self.sub_mean = MeanShift(args.rgb_range, rgb_mean, rgb_std)
        modules_head = [conv(args.n_colors, n_feats, kernel_size)]
        modules_body = [ResidualGroup(conv, n_feats, kernel_size, reduction, act=act, res_scale=args.res_scale, n_resblocks=n_resblocks) for _ in range(n_resgroups)]
        modules_body.append(conv(n_feats, n_feats, kernel_size))
        self.add_mean = MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)
        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        if args.no_upsampling:
            self.out_dim = n_feats
        else:
            self.out_dim = args.n_colors
            modules_tail = [Upsampler(conv, scale, n_feats, act=False), conv(n_feats, args.n_colors, kernel_size)]
            self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x
        if self.args.no_upsampling:
            x = res
        else:
            x = self.tail(res)
        return x

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        None
                    else:
                        raise RuntimeError('While copying the parameter named {}, whose dimensions in the model are {} and whose dimensions in the checkpoint are {}.'.format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'.format(name))
        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))


class RDB_Conv(nn.Module):

    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G = growRate
        self.conv = nn.Sequential(*[nn.Conv2d(Cin, G, kSize, padding=(kSize - 1) // 2, stride=1), nn.ReLU()])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)


class RDB(nn.Module):

    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G = growRate
        C = nConvLayers
        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c * G, G))
        self.convs = nn.Sequential(*convs)
        self.LFF = nn.Conv2d(G0 + C * G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x


class RDN(nn.Module):

    def __init__(self, args):
        super(RDN, self).__init__()
        self.args = args
        r = args.scale[0]
        G0 = args.G0
        kSize = args.RDNkSize
        self.D, C, G = {'A': (20, 6, 32), 'B': (16, 8, 64)}[args.RDNconfig]
        self.SFENet1 = nn.Conv2d(args.n_colors, G0, kSize, padding=(kSize - 1) // 2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(RDB(growRate0=G0, growRate=G, nConvLayers=C))
        self.GFF = nn.Sequential(*[nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1), nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)])
        if args.no_upsampling:
            self.out_dim = G0
        else:
            self.out_dim = args.n_colors
            if r == 2 or r == 3:
                self.UPNet = nn.Sequential(*[nn.Conv2d(G0, G * r * r, kSize, padding=(kSize - 1) // 2, stride=1), nn.PixelShuffle(r), nn.Conv2d(G, args.n_colors, kSize, padding=(kSize - 1) // 2, stride=1)])
            elif r == 4:
                self.UPNet = nn.Sequential(*[nn.Conv2d(G0, G * 4, kSize, padding=(kSize - 1) // 2, stride=1), nn.PixelShuffle(2), nn.Conv2d(G, G * 4, kSize, padding=(kSize - 1) // 2, stride=1), nn.PixelShuffle(2), nn.Conv2d(G, args.n_colors, kSize, padding=(kSize - 1) // 2, stride=1)])
            else:
                raise ValueError('scale must be 2 or 3 or 4.')

    def forward(self, x):
        f__1 = self.SFENet1(x)
        x = self.SFENet2(f__1)
        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)
        x = self.GFF(torch.cat(RDBs_out, 1))
        x += f__1
        if self.args.no_upsampling:
            return x
        else:
            return self.UPNet(x)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (MLP,
     lambda: ([], {'in_dim': 4, 'out_dim': 4, 'hidden_list': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (RCAB,
     lambda: ([], {'conv': _mock_layer, 'n_feat': 4, 'kernel_size': 4, 'reduction': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (RDB,
     lambda: ([], {'growRate0': 4, 'growRate': 4, 'nConvLayers': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (RDB_Conv,
     lambda: ([], {'inChannels': 4, 'growRate': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResBlock,
     lambda: ([], {'conv': _mock_layer, 'n_feats': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_yinboc_liif(_paritybench_base):
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

