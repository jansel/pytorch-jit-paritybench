import sys
_module = sys.modules[__name__]
del sys
ann_to_snn = _module
detect = _module
models = _module
ann_parser = _module
snn_evaluate = _module
snn_transformer = _module
spike_dag = _module
spike_layer = _module
spike_tensor = _module
test = _module
train = _module
utils = _module
adabound = _module
datasets = _module
google_utils = _module
layers = _module
parse_config = _module
torch_utils = _module
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


from torch.utils.data import DataLoader


from collections import OrderedDict


import torch


import torch.nn as nn


import numpy as np


import torch.nn.functional as F


from torch.nn.modules.utils import _pair


import torch.distributed as dist


import torch.optim as optim


import torch.optim.lr_scheduler as lr_scheduler


from torch.utils.tensorboard import SummaryWriter


import math


from torch.optim.optimizer import Optimizer


import random


import time


from torch.utils.data import Dataset


from copy import deepcopy


import torch.backends.cudnn as cudnn


from copy import copy


import matplotlib


import matplotlib.pyplot as plt


import torchvision


class ListWrapper(nn.Module):
    """
    partial model(without route & conv[-1] & yolo layers) to transform
    """

    def __init__(self, modulelist):
        super().__init__()
        self.list = modulelist

    def forward(self, x):
        for i in range(9):
            x = self.list[i](x)
        x1 = x
        for i in range(9, 13):
            x = self.list[i](x)
        x2 = x
        y1 = self.list[13](x)
        c = self.list[17](x2)
        c = self.list[18](c)
        x = torch.cat((c, x1), 1)
        y2 = self.list[20](x)
        return y1, y2


ONNX_EXPORT = False


class YOLOLayer(nn.Module):

    def __init__(self, anchors, nc, img_size, yolo_index, layers, stride):
        super(YOLOLayer, self).__init__()
        self.anchors = torch.Tensor(anchors)
        self.index = yolo_index
        self.layers = layers
        self.stride = stride
        self.nl = len(layers)
        self.na = len(anchors)
        self.nc = nc
        self.no = nc + 5
        self.nx, self.ny, self.ng = 0, 0, 0
        self.anchor_vec = self.anchors / self.stride
        self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2)
        if ONNX_EXPORT:
            self.training = False
            self.create_grids((img_size[1] // stride, img_size[0] // stride))

    def create_grids(self, ng=(13, 13), device='cpu'):
        self.nx, self.ny = ng
        self.ng = torch.tensor(ng, dtype=torch.float)
        if not self.training:
            yv, xv = torch.meshgrid([torch.arange(self.ny, device=device), torch.arange(self.nx, device=device)])
            self.grid = torch.stack((xv, yv), 2).view((1, 1, self.ny, self.nx, 2)).float()
        if self.anchor_vec.device != device:
            self.anchor_vec = self.anchor_vec
            self.anchor_wh = self.anchor_wh

    def forward(self, p, out):
        ASFF = False
        if ASFF:
            i, n = self.index, self.nl
            p = out[self.layers[i]]
            bs, _, ny, nx = p.shape
            if (self.nx, self.ny) != (nx, ny):
                self.create_grids((nx, ny), p.device)
            w = torch.sigmoid(p[:, -n:]) * (2 / n)
            p = out[self.layers[i]][:, :-n] * w[:, i:i + 1]
            for j in range(n):
                if j != i:
                    p += w[:, j:j + 1] * F.interpolate(out[self.layers[j]][:, :-n], size=[ny, nx], mode='bilinear', align_corners=False)
        elif ONNX_EXPORT:
            bs = 1
        else:
            bs, _, ny, nx = p.shape
            if (self.nx, self.ny) != (nx, ny):
                self.create_grids((nx, ny), p.device)
        p = p.view(bs, self.na, self.no, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()
        if self.training:
            return p
        elif ONNX_EXPORT:
            m = self.na * self.nx * self.ny
            ng = 1.0 / self.ng.repeat(m, 1)
            grid = self.grid.repeat(1, self.na, 1, 1, 1).view(m, 2)
            anchor_wh = self.anchor_wh.repeat(1, 1, self.nx, self.ny, 1).view(m, 2) * ng
            p = p.view(m, self.no)
            xy = torch.sigmoid(p[:, 0:2]) + grid
            wh = torch.exp(p[:, 2:4]) * anchor_wh
            p_cls = torch.sigmoid(p[:, 4:5]) if self.nc == 1 else torch.sigmoid(p[:, 5:self.no]) * torch.sigmoid(p[:, 4:5])
            return p_cls, xy * ng, wh
        else:
            io = p.clone()
            io[..., :2] = torch.sigmoid(io[..., :2]) + self.grid
            io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh
            io[..., :4] *= self.stride
            torch.sigmoid_(io[..., 4:])
            return io.view(bs, -1, self.no), p


class FeatureConcat(nn.Module):

    def __init__(self, layers):
        super(FeatureConcat, self).__init__()
        self.layers = layers
        self.multiple = len(layers) > 1

    def forward(self, x, outputs):
        return torch.cat([outputs[i] for i in self.layers], 1) if self.multiple else outputs[self.layers[0]]


class Mish(nn.Module):

    def forward(self, x):
        return x * F.softplus(x).tanh()


class MixConv2d(nn.Module):

    def __init__(self, in_ch, out_ch, k=(3, 5, 7), stride=1, dilation=1, bias=True, method='equal_params'):
        super(MixConv2d, self).__init__()
        groups = len(k)
        if method == 'equal_ch':
            i = torch.linspace(0, groups - 1e-06, out_ch).floor()
            ch = [(i == g).sum() for g in range(groups)]
        else:
            b = [out_ch] + [0] * groups
            a = np.eye(groups + 1, groups, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            ch = np.linalg.lstsq(a, b, rcond=None)[0].round().astype(int)
        self.m = nn.ModuleList([nn.Conv2d(in_channels=in_ch, out_channels=ch[g], kernel_size=k[g], stride=stride, padding=k[g] // 2, dilation=dilation, bias=bias) for g in range(groups)])

    def forward(self, x):
        return torch.cat([m(x) for m in self.m], 1)


class Swish(nn.Module):

    def forward(self, x):
        return x * torch.sigmoid(x)


class WeightedFeatureFusion(nn.Module):

    def __init__(self, layers, weight=False):
        super(WeightedFeatureFusion, self).__init__()
        self.layers = layers
        self.weight = weight
        self.n = len(layers) + 1
        if weight:
            self.w = nn.Parameter(torch.zeros(self.n), requires_grad=True)

    def forward(self, x, outputs):
        if self.weight:
            w = torch.sigmoid(self.w) * (2 / self.n)
            x = x * w[0]
        nx = x.shape[1]
        for i in range(self.n - 1):
            a = outputs[self.layers[i]] * w[i + 1] if self.weight else outputs[self.layers[i]]
            na = a.shape[1]
            if nx == na:
                x = x + a
            elif nx > na:
                x[:, :na] = x[:, :na] + a
            else:
                x = x + a[:, :nx]
        return x


def create_modules(module_defs, img_size, cfg):
    img_size = [img_size] * 2 if isinstance(img_size, int) else img_size
    _ = module_defs.pop(0)
    output_filters = [3]
    module_list = nn.ModuleList()
    routs = []
    yolo_index = -1
    for i, mdef in enumerate(module_defs):
        modules = nn.Sequential()
        if mdef['type'] == 'convolutional':
            bn = mdef['batch_normalize']
            filters = mdef['filters']
            k = mdef['size']
            stride = mdef['stride'] if 'stride' in mdef else (mdef['stride_y'], mdef['stride_x'])
            if isinstance(k, int):
                modules.add_module('Conv2d', nn.Conv2d(in_channels=output_filters[-1], out_channels=filters, kernel_size=k, stride=stride, padding=k // 2 if mdef['pad'] else 0, groups=mdef['groups'] if 'groups' in mdef else 1, bias=not bn))
            else:
                modules.add_module('MixConv2d', MixConv2d(in_ch=output_filters[-1], out_ch=filters, k=k, stride=stride, bias=not bn))
            if bn:
                modules.add_module('BatchNorm2d', nn.BatchNorm2d(filters, momentum=0.03, eps=0.0001))
            else:
                routs.append(i)
            if mdef['activation'] == 'leaky':
                modules.add_module('activation', nn.LeakyReLU(0.1, inplace=True))
            elif mdef['activation'] == 'swish':
                modules.add_module('activation', Swish())
            elif mdef['activation'] == 'mish':
                modules.add_module('activation', Mish())
            elif mdef['activation'] == 'relu':
                modules.add_module('activation', nn.ReLU(inplace=True))
        elif mdef['type'] == 'transposed_convolutional':
            bn = mdef['batch_normalize']
            filters = mdef['filters']
            k = mdef['size']
            stride = mdef['stride'] if 'stride' in mdef else (mdef['stride_y'], mdef['stride_x'])
            pad = k // 2
            output_padding = (2 * pad - k) % stride
            modules.add_module('ConvTranspose2d', nn.ConvTranspose2d(in_channels=output_filters[-1], out_channels=filters, kernel_size=k, stride=stride, padding=pad if mdef['pad'] else 0, output_padding=output_padding, groups=mdef['groups'] if 'groups' in mdef else 1, bias=not bn))
            if bn:
                modules.add_module('BatchNorm2d', nn.BatchNorm2d(filters, momentum=0.03, eps=0.0001))
            else:
                routs.append(i)
            if mdef['activation'] == 'leaky':
                modules.add_module('activation', nn.LeakyReLU(0.1, inplace=True))
            elif mdef['activation'] == 'swish':
                modules.add_module('activation', Swish())
            elif mdef['activation'] == 'mish':
                modules.add_module('activation', Mish())
            elif mdef['activation'] == 'relu':
                modules.add_module('activation', nn.ReLU(inplace=True))
        elif mdef['type'] == 'BatchNorm2d':
            filters = output_filters[-1]
            modules = nn.BatchNorm2d(filters, momentum=0.03, eps=0.0001)
            if i == 0 and filters == 3:
                modules.running_mean = torch.tensor([0.485, 0.456, 0.406])
                modules.running_var = torch.tensor([0.0524, 0.0502, 0.0506])
        elif mdef['type'] == 'maxpool':
            k = mdef['size']
            stride = mdef['stride']
            maxpool = nn.MaxPool2d(kernel_size=k, stride=stride, padding=(k - 1) // 2)
            if k == 2 and stride == 1:
                modules.add_module('ZeroPad2d', nn.ZeroPad2d((0, 1, 0, 1)))
                modules.add_module('MaxPool2d', maxpool)
            else:
                modules = maxpool
        elif mdef['type'] == 'upsample':
            if ONNX_EXPORT:
                g = (yolo_index + 1) * 2 / 32
                modules = nn.Upsample(size=tuple(int(x * g) for x in img_size))
            else:
                modules = nn.Upsample(scale_factor=mdef['stride'])
        elif mdef['type'] == 'route':
            layers = mdef['layers']
            filters = sum([output_filters[l + 1 if l > 0 else l] for l in layers])
            routs.extend([(i + l if l < 0 else l) for l in layers])
            modules = FeatureConcat(layers=layers)
        elif mdef['type'] == 'shortcut':
            layers = mdef['from']
            filters = output_filters[-1]
            routs.extend([(i + l if l < 0 else l) for l in layers])
            modules = WeightedFeatureFusion(layers=layers, weight='weights_type' in mdef)
        elif mdef['type'] == 'reorg3d':
            pass
        elif mdef['type'] == 'yolo':
            yolo_index += 1
            stride = [32, 16, 8]
            if any(x in cfg for x in ['panet', 'yolov4', 'cd53']):
                stride = list(reversed(stride))
            layers = mdef['from'] if 'from' in mdef else []
            modules = YOLOLayer(anchors=mdef['anchors'][mdef['mask']], nc=mdef['classes'], img_size=img_size, yolo_index=yolo_index, layers=layers, stride=stride[yolo_index])
            try:
                j = layers[yolo_index] if 'from' in mdef else -1
                if module_list[j].__class__.__name__ == 'Dropout':
                    j -= 1
                bias_ = module_list[j][0].bias
                bias = bias_[:modules.no * modules.na].view(modules.na, -1)
                bias[:, 4] += -4.5
                bias[:, 5:] += math.log(0.6 / (modules.nc - 0.99))
                module_list[j][0].bias = torch.nn.Parameter(bias_, requires_grad=bias_.requires_grad)
            except:
                None
        elif mdef['type'] == 'dropout':
            perc = float(mdef['probability'])
            modules = nn.Dropout(p=perc)
        else:
            None
        module_list.append(modules)
        output_filters.append(filters)
    routs_binary = [False] * (i + 1)
    for i in routs:
        routs_binary[i] = True
    return module_list, routs_binary


def get_yolo_layers(model):
    bool_vec = [(x['type'] == 'yolo') for x in model.module_defs]
    return [i for i, x in enumerate(bool_vec) if x]


def parse_model_cfg(path):
    if not path.endswith('.cfg'):
        path += '.cfg'
    if not os.path.exists(path) and os.path.exists('cfg' + os.sep + path):
        path = 'cfg' + os.sep + path
    with open(path, 'r') as f:
        lines = f.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]
    mdefs = []
    for line in lines:
        if line.startswith('['):
            mdefs.append({})
            mdefs[-1]['type'] = line[1:-1].rstrip()
            if mdefs[-1]['type'] == 'convolutional':
                mdefs[-1]['batch_normalize'] = 0
        else:
            key, val = line.split('=')
            key = key.rstrip()
            if key == 'anchors':
                mdefs[-1][key] = np.array([float(x) for x in val.split(',')]).reshape((-1, 2))
            elif key in ['from', 'layers', 'mask'] or key == 'size' and ',' in val:
                mdefs[-1][key] = [int(x) for x in val.split(',')]
            else:
                val = val.strip()
                if val.isnumeric():
                    mdefs[-1][key] = int(val) if int(val) - float(val) == 0 else float(val)
                else:
                    mdefs[-1][key] = val
    supported = ['type', 'batch_normalize', 'filters', 'size', 'stride', 'pad', 'activation', 'layers', 'groups', 'from', 'mask', 'anchors', 'classes', 'num', 'jitter', 'ignore_thresh', 'truth_thresh', 'random', 'stride_x', 'stride_y', 'weights_type', 'weights_normalization', 'scale_x_y', 'beta_nms', 'nms_kind', 'iou_loss', 'iou_normalizer', 'cls_normalizer', 'iou_thresh', 'probability']
    f = []
    for x in mdefs[1:]:
        [f.append(k) for k in x if k not in f]
    u = [x for x in f if x not in supported]
    assert not any(u), 'Unsupported fields %s in %s. See https://github.com/ultralytics/yolov3/issues/631' % (u, path)
    return mdefs


class Darknet(nn.Module):

    def __init__(self, cfg, img_size=(416, 416), verbose=False):
        super(Darknet, self).__init__()
        self.module_defs = parse_model_cfg(cfg)
        self.module_list, self.routs = create_modules(self.module_defs, img_size, cfg)
        self.yolo_layers = get_yolo_layers(self)
        self.version = np.array([0, 2, 5], dtype=np.int32)
        self.seen = np.array([0], dtype=np.int64)
        self.info(verbose) if not ONNX_EXPORT else None

    def forward(self, x, augment=False, verbose=False):
        if not augment:
            return self.forward_once(x)
        else:
            img_size = x.shape[-2:]
            s = [0.83, 0.67]
            y = []
            for i, xi in enumerate((x, torch_utils.scale_img(x.flip(3), s[0], same_shape=False), torch_utils.scale_img(x, s[1], same_shape=False))):
                y.append(self.forward_once(xi)[0])
            y[1][..., :4] /= s[0]
            y[1][..., 0] = img_size[1] - y[1][..., 0]
            y[2][..., :4] /= s[1]
            y = torch.cat(y, 1)
            return y, None

    def forward_once(self, x, augment=False, verbose=False):
        img_size = x.shape[-2:]
        yolo_out, out = [], []
        if verbose:
            None
            str = ''
        if augment:
            nb = x.shape[0]
            s = [0.83, 0.67]
            x = torch.cat((x, torch_utils.scale_img(x.flip(3), s[0]), torch_utils.scale_img(x, s[1])), 0)
        for i, module in enumerate(self.module_list):
            name = module.__class__.__name__
            if name in ['WeightedFeatureFusion', 'FeatureConcat']:
                if verbose:
                    l = [i - 1] + module.layers
                    sh = [list(x.shape)] + [list(out[i].shape) for i in module.layers]
                    str = ' >> ' + ' + '.join([('layer %g %s' % x) for x in zip(l, sh)])
                x = module(x, out)
            elif name == 'YOLOLayer':
                yolo_out.append(module(x, out))
            else:
                x = module(x)
            out.append(x if self.routs[i] else [])
            if verbose:
                None
                str = ''
        if self.training:
            return yolo_out
        elif ONNX_EXPORT:
            x = [torch.cat(x, 0) for x in zip(*yolo_out)]
            return x[0], torch.cat(x[1:3], 1)
        else:
            x, p = zip(*yolo_out)
            x = torch.cat(x, 1)
            if augment:
                x = torch.split(x, nb, dim=0)
                x[1][..., :4] /= s[0]
                x[1][..., 0] = img_size[1] - x[1][..., 0]
                x[2][..., :4] /= s[1]
                x = torch.cat(x, 1)
            return x, p

    def fuse(self):
        None
        fused_list = nn.ModuleList()
        for a in list(self.children())[0]:
            if isinstance(a, nn.Sequential):
                for i, b in enumerate(a):
                    if isinstance(b, nn.modules.batchnorm.BatchNorm2d):
                        conv = a[i - 1]
                        fused = torch_utils.fuse_conv_and_bn(conv, b)
                        a = nn.Sequential(fused, *list(a.children())[i + 1:])
                        break
            fused_list.append(a)
        self.module_list = fused_list
        self.info() if not ONNX_EXPORT else None

    def info(self, verbose=False):
        torch_utils.model_info(self, verbose)


firing_ratio_record = False


firing_ratios = []


class SpikeTensor:

    def __init__(self, data, timesteps, scale_factor):
        """
        data shape: [t*batch, channel, height, width]
        """
        self.data = data
        self.timesteps = timesteps
        self.b = self.data.size(0) // timesteps
        self.chw = self.data.size()[1:]
        self.scale_factor = torch.ones([*self.data.size()[1:]])
        if isinstance(scale_factor, torch.Tensor):
            dim = scale_factor.dim()
            if dim == 1:
                self.scale_factor *= scale_factor.view(-1, *([1] * (len(self.chw) - 1)))
            else:
                self.scale_factor *= scale_factor
        else:
            self.scale_factor.fill_(scale_factor)
        if firing_ratio_record:
            firing_ratios.append(self.firing_ratio())

    def firing_ratio(self):
        firing_ratio = torch.mean(self.data.view(self.timesteps, -1, *self.chw), 0)
        return firing_ratio

    def timestep_dim_tensor(self):
        return self.data.view(self.timesteps, -1, *self.chw)

    def size(self, *args):
        return self.data.size(*args)

    def view(self, *args):
        return SpikeTensor(self.data.view(*args), self.timesteps, self.scale_factor.view(*args[1:]))

    def to_float(self):
        assert self.scale_factor is not None
        firing_ratio = self.firing_ratio()
        scaled_float_tensor = firing_ratio * self.scale_factor.unsqueeze(0)
        return scaled_float_tensor

    def __str__(self):
        return f'Spiketensor T{self.timesteps} Shape({self.b} {self.chw}) ScaleFactor {self.scale_factor} \n{self.data.shape}'


class SpikeDAGModule(nn.Module):
    """
    This is a module contains `nodes` dict and `ops` dict.
    It can do inference under the topological ordering of the ops given the input nodes.
    - Attributes
        - `nodes` OrderedDict, in which the key is the name of the input (`<op_name>_out_<id>`)
        - `ops` OrderedDict, in which the key is the name of the op (`<op_type>_<id>`)
            and the value is a dict `{'op':op, 'in_nodes':tuple, 'out_nodes':tuple}`
        - `inputs_nodes` tuple, specifying the input nodes for forward
    - forward
        - copy the inputs to nodes in `inputs_nodes`
        - for `op` in `ops`:
            - get inputs from `nodes`
            - forward `op`
            - save outputs to `nodes`
    """

    def __init__(self):
        super().__init__()
        self.nodes = OrderedDict()
        self.ops = OrderedDict()
        self.inputs_nodes = []
        self.outputs_nodes = []

    def add_op(self, name, op, in_nodes, out_nodes):
        assert name not in self.ops, AssertionError(f'Error: op {name} already in dag.ops')
        assert len(in_nodes) and len(out_nodes)
        self.ops[name] = {'op': op, 'in_nodes': in_nodes, 'out_nodes': out_nodes}
        if isinstance(op, nn.Module):
            self.add_module(name, op)
        None

    def add_node(self, name, tensor=None):
        assert name not in self.nodes, AssertionError(f'Error: node {name} already in dag.nodes')
        self.nodes[name] = tensor

    def clear_nodes(self):
        for key in self.nodes:
            self.nodes[key] = None

    def find_end_nodes(self):
        in_nodes = []
        for _, op in self.ops.items():
            in_nodes.extend(op['in_nodes'])
        end_nodes = []
        for node in self.nodes:
            if node not in in_nodes:
                end_nodes.append(node)
        None
        return tuple(end_nodes)

    def forward(self, *inputs):
        assert len(self.inputs_nodes) == len(inputs)
        assert len(self.outputs_nodes)
        for inp, name in zip(inputs, self.inputs_nodes):
            self.nodes[name] = inp
        for op_name, op in self.ops.items():
            op_inputs = tuple(self.nodes[_] for _ in op['in_nodes'])
            op_outputs = op['op'](*op_inputs)
            if isinstance(op_outputs, torch.Tensor) or isinstance(op_outputs, SpikeTensor):
                op_outputs = [op_outputs]
            for node_name, op_output in zip(op['out_nodes'], op_outputs):
                self.nodes[node_name] = op_output
        outputs = tuple(self.nodes[_] for _ in self.outputs_nodes)
        return outputs if len(outputs) > 1 else outputs[0]


class SpikeReLU(nn.Module):

    def __init__(self, quantize=False):
        super().__init__()
        self.max_val = 1
        self.quantize = quantize
        self.activation_bitwidth = None

    def forward(self, x):
        if isinstance(x, SpikeTensor):
            return x
        else:
            x_ = F.relu(x)
            if self.quantize:
                bits = self.activation_bitwidth
                if self.training:
                    xv = x_.view(-1)
                    max_val = xv.max()
                    if self.max_val is 1:
                        self.max_val = max_val.detach()
                    else:
                        self.max_val = (self.max_val * 0.95 + max_val * 0.05).detach()
                rst = torch.clamp(torch.round(x_ / self.max_val * 2 ** bits), 0, 2 ** bits) * (self.max_val / 2 ** bits)
                return rst
            else:
                return x_


def generate_spike_mem_potential(out_s, mem_potential, Vthr, reset_mode):
    """
    out_s: is a Tensor of the output of different timesteps [timesteps, *sizes]
    mem_potential: is a placeholder Tensor with [*sizes]
    """
    spikes = []
    for t in range(out_s.size(0)):
        mem_potential += out_s[t]
        spike = (mem_potential >= Vthr).float()
        if reset_mode == 'zero':
            mem_potential *= 1 - spike
        elif reset_mode == 'subtraction':
            mem_potential -= spike * Vthr
        else:
            raise NotImplementedError
        spikes.append(spike)
    return spikes


class SpikeConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', bn=None):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.mem_potential = None
        self.register_buffer('out_scales', torch.ones(out_channels))
        self.register_buffer('Vthr', torch.ones(1))
        self.register_buffer('leakage', torch.zeros(out_channels))
        self.reset_mode = 'subtraction'
        self.bn = bn

    def forward(self, x):
        if isinstance(x, SpikeTensor):
            Vthr = self.Vthr.view(1, -1, 1, 1)
            out = F.conv2d(x.data, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
            if self.bn is not None:
                out = self.bn(out)
            chw = out.size()[1:]
            out_s = out.view(x.timesteps, -1, *chw)
            self.mem_potential = torch.zeros(out_s.size(1), *chw)
            spikes = generate_spike_mem_potential(out_s, self.mem_potential, Vthr, self.reset_mode)
            out = SpikeTensor(torch.cat(spikes, 0), x.timesteps, self.out_scales)
            return out
        else:
            out = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
            if self.bn is not None:
                out = self.bn(out)
            return out


class SpikeConvTranspose2d(nn.ConvTranspose2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', bn=None):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation, padding_mode)
        self.mem_potential = None
        self.register_buffer('out_scales', torch.ones(out_channels))
        self.register_buffer('Vthr', torch.ones(1))
        self.register_buffer('leakage', torch.zeros(out_channels))
        self.reset_mode = 'subtraction'
        self.bn = bn

    def forward(self, x):
        if isinstance(x, SpikeTensor):
            Vthr = self.Vthr.view(1, -1, 1, 1)
            out = F.conv_transpose2d(x.data, self.weight, self.bias, self.stride, self.padding, self.output_padding, self.groups, self.dilation)
            if self.bn is not None:
                out = self.bn(out)
            chw = out.size()[1:]
            out_s = out.view(x.timesteps, -1, *chw)
            self.mem_potential = torch.zeros(out_s.size(1), *chw)
            spikes = generate_spike_mem_potential(out_s, self.mem_potential, Vthr, self.reset_mode)
            out = SpikeTensor(torch.cat(spikes, 0), x.timesteps, self.out_scales)
            return out
        else:
            out = F.conv_transpose2d(x.data, self.weight, self.bias, self.stride, self.padding, self.output_padding, self.groups, self.dilation)
            if self.bn is not None:
                out = self.bn(out)
            return out


class SpikeAvgPool2d(nn.Module):
    """
    substitute all 1 Depthwise Convolution for AvgPooling
    """

    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        if stride is None:
            stride = kernel_size
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.register_buffer('Vthr', torch.ones(1) * np.prod(self.kernel_size))
        self.reset_mode = 'subtraction'

    def forward(self, x):
        Vthr = self.Vthr
        kernel_size = self.kernel_size
        stride = self.stride
        padding = self.padding
        if isinstance(x, SpikeTensor):
            if stride is None:
                stride = kernel_size
            weight = torch.ones([x.chw[0], 1, *_pair(kernel_size)])
            out = F.conv2d(x.data, weight, None, _pair(stride), _pair(padding), 1, groups=x.chw[0])
            chw = out.size()[1:]
            out_s = out.view(x.timesteps, -1, *chw)
            self.mem_potential = torch.zeros(out_s.size(1), *chw)
            spikes = generate_spike_mem_potential(out_s, self.mem_potential, Vthr, self.reset_mode)
            out = SpikeTensor(torch.cat(spikes, 0), x.timesteps, F.avg_pool2d(x.scale_factor.unsqueeze(0), kernel_size, stride, padding).squeeze(0))
            return out
        else:
            out = F.avg_pool2d(x, kernel_size, stride, padding)
            return out


class SpikeLinear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, last_layer=False):
        super().__init__(in_features, out_features, bias)
        self.last_layer = last_layer
        self.register_buffer('out_scales', torch.ones(out_features))
        self.register_buffer('Vthr', torch.ones(1))
        self.register_buffer('leakage', torch.zeros(out_features))
        self.reset_mode = 'subtraction'

    def forward(self, x):
        if isinstance(x, SpikeTensor):
            Vthr = self.Vthr.view(1, -1)
            out = F.linear(x.data, self.weight, self.bias)
            chw = out.size()[1:]
            out_s = out.view(x.timesteps, -1, *chw)
            self.mem_potential = torch.zeros(out_s.size(1), *chw)
            spikes = generate_spike_mem_potential(out_s, self.mem_potential, Vthr, self.reset_mode)
            out = SpikeTensor(torch.cat(spikes, 0), x.timesteps, self.out_scales)
            return out
        else:
            out = F.linear(x, self.weight, self.bias)
            if not self.last_layer:
                out = F.relu(out)
            return out


class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)


class Concat(nn.Module):

    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class SwishImplementation(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * torch.sigmoid(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        sx = torch.sigmoid(x)
        return grad_output * (sx * (1 + x * (1 - sx)))


class MemoryEfficientSwish(nn.Module):

    def forward(self, x):
        return SwishImplementation.apply(x)


class MishImplementation(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.mul(torch.tanh(F.softplus(x)))

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        sx = torch.sigmoid(x)
        fx = F.softplus(x).tanh()
        return grad_output * (fx + x * sx * (1 - fx * fx))


class MemoryEfficientMish(nn.Module):

    def forward(self, x):
        return MishImplementation.apply(x)


class HardSwish(nn.Module):

    def forward(self, x):
        return x * F.hardtanh(x + 3, 0.0, 6.0, True) / 6.0


class FocalLoss(nn.Module):

    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred_prob = torch.sigmoid(pred)
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (FeatureConcat,
     lambda: ([], {'layers': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([5, 4, 4, 4])], {}),
     True),
    (Flatten,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FocalLoss,
     lambda: ([], {'loss_fcn': MSELoss()}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (HardSwish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MemoryEfficientMish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MemoryEfficientSwish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Mish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SpikeAvgPool2d,
     lambda: ([], {'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SpikeConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SpikeConvTranspose2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SpikeLinear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SpikeReLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Swish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (WeightedFeatureFusion,
     lambda: ([], {'layers': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([5, 4, 4, 4])], {}),
     False),
]

class Test_cwq159_PyTorch_Spiking_YOLOv3(_paritybench_base):
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

    def test_009(self):
        self._check(*TESTCASES[9])

    def test_010(self):
        self._check(*TESTCASES[10])

    def test_011(self):
        self._check(*TESTCASES[11])

    def test_012(self):
        self._check(*TESTCASES[12])

    def test_013(self):
        self._check(*TESTCASES[13])

