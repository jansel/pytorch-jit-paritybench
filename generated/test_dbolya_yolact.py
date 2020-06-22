import sys
_module = sys.modules[__name__]
del sys
backbone = _module
data = _module
coco = _module
config = _module
mix_sets = _module
eval = _module
DCNv2 = _module
dcn_v2 = _module
setup = _module
test = _module
layers = _module
box_utils = _module
functions = _module
detection = _module
interpolate = _module
modules = _module
multibox_loss = _module
output_utils = _module
run_coco_eval = _module
augment_bbox = _module
bbox_recall = _module
cluster_bbox_sizes = _module
compute_masks = _module
convert_darknet = _module
convert_sbd = _module
make_grid = _module
optimize_bboxes = _module
parse_eval = _module
plot_loss = _module
save_bboxes = _module
unpack_statedict = _module
train = _module
utils = _module
augmentations = _module
functions = _module
logger = _module
nvinfo = _module
timer = _module
server = _module
yolact = _module

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


from collections import OrderedDict


import torch.utils.data as data


import torch.nn.functional as F


import numpy as np


import random


from math import sqrt


import torch.backends.cudnn as cudnn


from torch.autograd import Variable


import time


from collections import defaultdict


import math


from torch import nn


from torch.autograd import Function


from torch.nn.modules.utils import _pair


from torch.autograd.function import once_differentiable


from torch.autograd import gradcheck


import torch.optim as optim


import torch.nn.init as init


import types


from numpy import random


from collections import deque


from itertools import product


from typing import List


class _DCNv2(Function):

    @staticmethod
    def forward(ctx, input, offset, mask, weight, bias, stride, padding,
        dilation, deformable_groups):
        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.kernel_size = _pair(weight.shape[2:4])
        ctx.deformable_groups = deformable_groups
        output = _backend.dcn_v2_forward(input, weight, bias, offset, mask,
            ctx.kernel_size[0], ctx.kernel_size[1], ctx.stride[0], ctx.
            stride[1], ctx.padding[0], ctx.padding[1], ctx.dilation[0], ctx
            .dilation[1], ctx.deformable_groups)
        ctx.save_for_backward(input, offset, mask, weight, bias)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, offset, mask, weight, bias = ctx.saved_tensors
        grad_input, grad_offset, grad_mask, grad_weight, grad_bias = (_backend
            .dcn_v2_backward(input, weight, bias, offset, mask, grad_output,
            ctx.kernel_size[0], ctx.kernel_size[1], ctx.stride[0], ctx.
            stride[1], ctx.padding[0], ctx.padding[1], ctx.dilation[0], ctx
            .dilation[1], ctx.deformable_groups))
        return (grad_input, grad_offset, grad_mask, grad_weight, grad_bias,
            None, None, None, None)


dcn_v2_conv = _DCNv2.apply


def darknetconvlayer(in_channels, out_channels, *args, **kwdargs):
    """
    Implements a conv, activation, then batch norm.
    Arguments are passed into the conv layer.
    """
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, *args, **
        kwdargs, bias=False), nn.BatchNorm2d(out_channels), nn.LeakyReLU(
        0.1, inplace=True))


class DarkNetBlock(nn.Module):
    """ Note: channels is the lesser of the two. The output will be expansion * channels. """
    expansion = 2

    def __init__(self, in_channels, channels):
        super().__init__()
        self.conv1 = darknetconvlayer(in_channels, channels, kernel_size=1)
        self.conv2 = darknetconvlayer(channels, channels * self.expansion,
            kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv2(self.conv1(x)) + x


class DarkNetBackbone(nn.Module):
    """
    An implementation of YOLOv3's Darnet53 in
    https://pjreddie.com/media/files/papers/YOLOv3.pdf

    This is based off of the implementation of Resnet above.
    """

    def __init__(self, layers=[1, 2, 8, 8, 4], block=DarkNetBlock):
        super().__init__()
        self.num_base_layers = len(layers)
        self.layers = nn.ModuleList()
        self.channels = []
        self._preconv = darknetconvlayer(3, 32, kernel_size=3, padding=1)
        self.in_channels = 32
        self._make_layer(block, 32, layers[0])
        self._make_layer(block, 64, layers[1])
        self._make_layer(block, 128, layers[2])
        self._make_layer(block, 256, layers[3])
        self._make_layer(block, 512, layers[4])
        self.backbone_modules = [m for m in self.modules() if isinstance(m,
            nn.Conv2d)]

    def _make_layer(self, block, channels, num_blocks, stride=2):
        """ Here one layer means a string of n blocks. """
        layer_list = []
        layer_list.append(darknetconvlayer(self.in_channels, channels *
            block.expansion, kernel_size=3, padding=1, stride=stride))
        self.in_channels = channels * block.expansion
        layer_list += [block(self.in_channels, channels) for _ in range(
            num_blocks)]
        self.channels.append(self.in_channels)
        self.layers.append(nn.Sequential(*layer_list))

    def forward(self, x):
        """ Returns a list of convouts for each layer. """
        x = self._preconv(x)
        outs = []
        for layer in self.layers:
            x = layer(x)
            outs.append(x)
        return tuple(outs)

    def add_layer(self, conv_channels=1024, stride=2, depth=1, block=
        DarkNetBlock):
        """ Add a downsample layer to the backbone as per what SSD does. """
        self._make_layer(block, conv_channels // block.expansion,
            num_blocks=depth, stride=stride)

    def init_backbone(self, path):
        """ Initializes the backbone weights for training. """
        self.load_state_dict(torch.load(path), strict=False)


class VGGBackbone(nn.Module):
    """
    Args:
        - cfg: A list of layers given as lists. Layers can be either 'M' signifying
                a max pooling layer, a number signifying that many feature maps in
                a conv layer, or a tuple of 'M' or a number and a kwdargs dict to pass
                into the function that creates the layer (e.g. nn.MaxPool2d for 'M').
        - extra_args: A list of lists of arguments to pass into add_layer.
        - norm_layers: Layers indices that need to pass through an l2norm layer.
    """

    def __init__(self, cfg, extra_args=[], norm_layers=[]):
        super().__init__()
        self.channels = []
        self.layers = nn.ModuleList()
        self.in_channels = 3
        self.extra_args = list(reversed(extra_args))
        self.total_layer_count = 0
        self.state_dict_lookup = {}
        for idx, layer_cfg in enumerate(cfg):
            self._make_layer(layer_cfg)
        self.norms = nn.ModuleList([nn.BatchNorm2d(self.channels[l]) for l in
            norm_layers])
        self.norm_lookup = {l: idx for idx, l in enumerate(norm_layers)}
        self.backbone_modules = [m for m in self.modules() if isinstance(m,
            nn.Conv2d)]

    def _make_layer(self, cfg):
        """
        Each layer is a sequence of conv layers usually preceded by a max pooling.
        Adapted from torchvision.models.vgg.make_layers.
        """
        layers = []
        for v in cfg:
            args = None
            if isinstance(v, tuple):
                args = v[1]
                v = v[0]
            if v == 'M':
                if args is None:
                    args = {'kernel_size': 2, 'stride': 2}
                layers.append(nn.MaxPool2d(**args))
            else:
                cur_layer_idx = self.total_layer_count + len(layers)
                self.state_dict_lookup[cur_layer_idx] = '%d.%d' % (len(self
                    .layers), len(layers))
                if args is None:
                    args = {'kernel_size': 3, 'padding': 1}
                layers.append(nn.Conv2d(self.in_channels, v, **args))
                layers.append(nn.ReLU(inplace=True))
                self.in_channels = v
        self.total_layer_count += len(layers)
        self.channels.append(self.in_channels)
        self.layers.append(nn.Sequential(*layers))

    def forward(self, x):
        """ Returns a list of convouts for each layer. """
        outs = []
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx in self.norm_lookup:
                x = self.norms[self.norm_lookup[idx]](x)
            outs.append(x)
        return tuple(outs)

    def transform_key(self, k):
        """ Transform e.g. features.24.bias to layers.4.1.bias """
        vals = k.split('.')
        layerIdx = self.state_dict_lookup[int(vals[0])]
        return 'layers.%s.%s' % (layerIdx, vals[1])

    def init_backbone(self, path):
        """ Initializes the backbone weights for training. """
        state_dict = torch.load(path)
        state_dict = OrderedDict([(self.transform_key(k), v) for k, v in
            state_dict.items()])
        self.load_state_dict(state_dict, strict=False)

    def add_layer(self, conv_channels=128, downsample=2):
        """ Add a downsample layer to the backbone as per what SSD does. """
        if len(self.extra_args) > 0:
            conv_channels, downsample = self.extra_args.pop()
        padding = 1 if downsample > 1 else 0
        layer = nn.Sequential(nn.Conv2d(self.in_channels, conv_channels,
            kernel_size=1), nn.ReLU(inplace=True), nn.Conv2d(conv_channels,
            conv_channels * 2, kernel_size=3, stride=downsample, padding=
            padding), nn.ReLU(inplace=True))
        self.in_channels = conv_channels * 2
        self.channels.append(self.in_channels)
        self.layers.append(layer)


class CustomDataParallel(torch.nn.DataParallel):
    """ A Custom Data Parallel class that properly gathers lists of dictionaries. """

    def gather(self, outputs, output_device):
        return sum(outputs, [])


class DCNv2(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
        padding, dilation=1, deformable_groups=1):
        super(DCNv2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.deformable_groups = deformable_groups
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels,
            *self.kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.zero_()

    def forward(self, input, offset, mask):
        assert 2 * self.deformable_groups * self.kernel_size[0
            ] * self.kernel_size[1] == offset.shape[1]
        assert self.deformable_groups * self.kernel_size[0] * self.kernel_size[
            1] == mask.shape[1]
        return dcn_v2_conv(input, offset, mask, self.weight, self.bias,
            self.stride, self.padding, self.dilation, self.deformable_groups)


class _DCNv2Pooling(Function):

    @staticmethod
    def forward(ctx, input, rois, offset, spatial_scale, pooled_size,
        output_dim, no_trans, group_size=1, part_size=None, sample_per_part
        =4, trans_std=0.0):
        ctx.spatial_scale = spatial_scale
        ctx.no_trans = int(no_trans)
        ctx.output_dim = output_dim
        ctx.group_size = group_size
        ctx.pooled_size = pooled_size
        ctx.part_size = pooled_size if part_size is None else part_size
        ctx.sample_per_part = sample_per_part
        ctx.trans_std = trans_std
        output, output_count = _backend.dcn_v2_psroi_pooling_forward(input,
            rois, offset, ctx.no_trans, ctx.spatial_scale, ctx.output_dim,
            ctx.group_size, ctx.pooled_size, ctx.part_size, ctx.
            sample_per_part, ctx.trans_std)
        ctx.save_for_backward(input, rois, offset, output_count)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, rois, offset, output_count = ctx.saved_tensors
        grad_input, grad_offset = _backend.dcn_v2_psroi_pooling_backward(
            grad_output, input, rois, offset, output_count, ctx.no_trans,
            ctx.spatial_scale, ctx.output_dim, ctx.group_size, ctx.
            pooled_size, ctx.part_size, ctx.sample_per_part, ctx.trans_std)
        return (grad_input, None, grad_offset, None, None, None, None, None,
            None, None, None)


dcn_v2_pooling = _DCNv2Pooling.apply


class DCNv2Pooling(nn.Module):

    def __init__(self, spatial_scale, pooled_size, output_dim, no_trans,
        group_size=1, part_size=None, sample_per_part=4, trans_std=0.0):
        super(DCNv2Pooling, self).__init__()
        self.spatial_scale = spatial_scale
        self.pooled_size = pooled_size
        self.output_dim = output_dim
        self.no_trans = no_trans
        self.group_size = group_size
        self.part_size = pooled_size if part_size is None else part_size
        self.sample_per_part = sample_per_part
        self.trans_std = trans_std

    def forward(self, input, rois, offset):
        assert input.shape[1] == self.output_dim
        if self.no_trans:
            offset = input.new()
        return dcn_v2_pooling(input, rois, offset, self.spatial_scale, self
            .pooled_size, self.output_dim, self.no_trans, self.group_size,
            self.part_size, self.sample_per_part, self.trans_std)


class InterpolateModule(nn.Module):
    """
	This is a module version of F.interpolate (rip nn.Upsampling).
	Any arguments you give it just get passed along for the ride.
	"""

    def __init__(self, *args, **kwdargs):
        super().__init__()
        self.args = args
        self.kwdargs = kwdargs

    def forward(self, x):
        return F.interpolate(x, *self.args, **self.kwdargs)


class Config(object):
    """
    Holds the configuration for anything you want it to.
    To get the currently active config, call get_cfg().

    To use, just do cfg.x instead of cfg['x'].
    I made this because doing cfg['x'] all the time is dumb.
    """

    def __init__(self, config_dict):
        for key, val in config_dict.items():
            self.__setattr__(key, val)

    def copy(self, new_config_dict={}):
        """
        Copies this config into a new config object, making
        the changes given by new_config_dict.
        """
        ret = Config(vars(self))
        for key, val in new_config_dict.items():
            ret.__setattr__(key, val)
        return ret

    def replace(self, new_config_dict):
        """
        Copies new_config_dict into this config object.
        Note: new_config_dict can also be a config object.
        """
        if isinstance(new_config_dict, Config):
            new_config_dict = vars(new_config_dict)
        for key, val in new_config_dict.items():
            self.__setattr__(key, val)

    def print(self):
        for k, v in vars(self).items():
            print(k, ' = ', v)


activation_func = Config({'tanh': torch.tanh, 'sigmoid': torch.sigmoid,
    'softmax': lambda x: torch.nn.functional.softmax(x, dim=-1), 'relu': lambda
    x: torch.nn.functional.relu(x, inplace=True), 'none': lambda x: x})


@torch.jit.script
def center_size(boxes):
    """ Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat(((boxes[:, 2:] + boxes[:, :2]) / 2, boxes[:, 2:] -
        boxes[:, :2]), 1)


@torch.jit.script
def sanitize_coordinates(_x1, _x2, img_size: int, padding: int=0, cast:
    bool=True):
    """
    Sanitizes the input coordinates so that x1 < x2, x1 != x2, x1 >= 0, and x2 <= image_size.
    Also converts from relative to absolute coordinates and casts the results to long tensors.

    If cast is false, the result won't be cast to longs.
    Warning: this does things in-place behind the scenes so copy if necessary.
    """
    _x1 = _x1 * img_size
    _x2 = _x2 * img_size
    if cast:
        _x1 = _x1.long()
        _x2 = _x2.long()
    x1 = torch.min(_x1, _x2)
    x2 = torch.max(_x1, _x2)
    x1 = torch.clamp(x1 - padding, min=0)
    x2 = torch.clamp(x2 + padding, max=img_size)
    return x1, x2


@torch.jit.script
def crop(masks, boxes, padding: int=1):
    """
    "Crop" predicted masks by zeroing out everything not in the predicted bbox.
    Vectorized by Chong (thanks Chong).

    Args:
        - masks should be a size [h, w, n] tensor of masks
        - boxes should be a size [n, 4] tensor of bbox coords in relative point form
    """
    h, w, n = masks.size()
    x1, x2 = sanitize_coordinates(boxes[:, (0)], boxes[:, (2)], w, padding,
        cast=False)
    y1, y2 = sanitize_coordinates(boxes[:, (1)], boxes[:, (3)], h, padding,
        cast=False)
    rows = torch.arange(w, device=masks.device, dtype=x1.dtype).view(1, -1, 1
        ).expand(h, w, n)
    cols = torch.arange(h, device=masks.device, dtype=x1.dtype).view(-1, 1, 1
        ).expand(h, w, n)
    masks_left = rows >= x1.view(1, 1, -1)
    masks_right = rows < x2.view(1, 1, -1)
    masks_up = cols >= y1.view(1, 1, -1)
    masks_down = cols < y2.view(1, 1, -1)
    crop_mask = masks_left * masks_right * masks_up * masks_down
    return masks * crop_mask.float()


@torch.jit.script
def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes
        [:, 2:] / 2), 1)


@torch.jit.script
def decode(loc, priors, use_yolo_regressors: bool=False):
    """
    Decode predicted bbox coordinates using the same scheme
    employed by Yolov2: https://arxiv.org/pdf/1612.08242.pdf

        b_x = (sigmoid(pred_x) - .5) / conv_w + prior_x
        b_y = (sigmoid(pred_y) - .5) / conv_h + prior_y
        b_w = prior_w * exp(loc_w)
        b_h = prior_h * exp(loc_h)
    
    Note that loc is inputed as [(s(x)-.5)/conv_w, (s(y)-.5)/conv_h, w, h]
    while priors are inputed as [x, y, w, h] where each coordinate
    is relative to size of the image (even sigmoid(x)). We do this
    in the network by dividing by the 'cell size', which is just
    the size of the convouts.
    
    Also note that prior_x and prior_y are center coordinates which
    is why we have to subtract .5 from sigmoid(pred_x and pred_y).
    
    Args:
        - loc:    The predicted bounding boxes of size [num_priors, 4]
        - priors: The priorbox coords with size [num_priors, 4]
    
    Returns: A tensor of decoded relative coordinates in point form 
             form with size [num_priors, 4]
    """
    if use_yolo_regressors:
        boxes = torch.cat((loc[:, :2] + priors[:, :2], priors[:, 2:] *
            torch.exp(loc[:, 2:])), 1)
        boxes = point_form(boxes)
    else:
        variances = [0.1, 0.2]
        boxes = torch.cat((priors[:, :2] + loc[:, :2] * variances[0] *
            priors[:, 2:], priors[:, 2:] * torch.exp(loc[:, 2:] * variances
            [1])), 1)
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
    return boxes


def elemwise_box_iou(box_a, box_b):
    """ Does the same as above but instead of pairwise, elementwise along the inner dimension. """
    max_xy = torch.min(box_a[:, 2:], box_b[:, 2:])
    min_xy = torch.max(box_a[:, :2], box_b[:, :2])
    inter = torch.clamp(max_xy - min_xy, min=0)
    inter = inter[:, (0)] * inter[:, (1)]
    area_a = (box_a[:, (2)] - box_a[:, (0)]) * (box_a[:, (3)] - box_a[:, (1)])
    area_b = (box_b[:, (2)] - box_b[:, (0)]) * (box_b[:, (3)] - box_b[:, (1)])
    union = area_a + area_b - inter
    union = torch.clamp(union, min=0.1)
    return torch.clamp(inter / union, max=1)


def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x - x_max), 1)) + x_max


mask_type = Config({'direct': 0, 'lincomb': 1})


def change(gt, priors):
    """
    Compute the d_change metric proposed in Box2Pix:
    https://lmb.informatik.uni-freiburg.de/Publications/2018/UB18/paper-box2pix.pdf
    
    Input should be in point form (xmin, ymin, xmax, ymax).

    Output is of shape [num_gt, num_priors]
    Note this returns -change so it can be a drop in replacement for 
    """
    num_priors = priors.size(0)
    num_gt = gt.size(0)
    gt_w = (gt[:, (2)] - gt[:, (0)])[:, (None)].expand(num_gt, num_priors)
    gt_h = (gt[:, (3)] - gt[:, (1)])[:, (None)].expand(num_gt, num_priors)
    gt_mat = gt[:, (None), :].expand(num_gt, num_priors, 4)
    pr_mat = priors[(None), :, :].expand(num_gt, num_priors, 4)
    diff = gt_mat - pr_mat
    diff[:, :, (0)] /= gt_w
    diff[:, :, (2)] /= gt_w
    diff[:, :, (1)] /= gt_h
    diff[:, :, (3)] /= gt_h
    return -torch.sqrt((diff ** 2).sum(dim=2))


@torch.jit.script
def encode(matched, priors, use_yolo_regressors: bool=False):
    """
    Encode bboxes matched with each prior into the format
    produced by the network. See decode for more details on
    this format. Note that encode(decode(x, p), p) = x.
    
    Args:
        - matched: A tensor of bboxes in point form with shape [num_priors, 4]
        - priors:  The tensor of all priors with shape [num_priors, 4]
    Return: A tensor with encoded relative coordinates in the format
            outputted by the network (see decode). Size: [num_priors, 4]
    """
    if use_yolo_regressors:
        boxes = center_size(matched)
        loc = torch.cat((boxes[:, :2] - priors[:, :2], torch.log(boxes[:, 2
            :] / priors[:, 2:])), 1)
    else:
        variances = [0.1, 0.2]
        g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - priors[:, :2]
        g_cxcy /= variances[0] * priors[:, 2:]
        g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
        g_wh = torch.log(g_wh) / variances[1]
        loc = torch.cat([g_cxcy, g_wh], 1)
    return loc


def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip(max_xy - min_xy, a_min=0, a_max=np.inf)
    return inter[:, (0)] * inter[:, (1)]


def jaccard(box_a, box_b, iscrowd: bool=False):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes. If iscrowd=True, put the crowd in box_b.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    use_batch = True
    if box_a.dim() == 2:
        use_batch = False
        box_a = box_a[None, ...]
        box_b = box_b[None, ...]
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, :, (2)] - box_a[:, :, (0)]) * (box_a[:, :, (3)] -
        box_a[:, :, (1)])).unsqueeze(2).expand_as(inter)
    area_b = ((box_b[:, :, (2)] - box_b[:, :, (0)]) * (box_b[:, :, (3)] -
        box_b[:, :, (1)])).unsqueeze(1).expand_as(inter)
    union = area_a + area_b - inter
    out = inter / area_a if iscrowd else inter / union
    return out if use_batch else out.squeeze(0)


_global_config['crowd_iou_threshold'] = 4


_global_config['use_prediction_matching'] = 4


_global_config['use_yolo_regressors'] = 4


_global_config['use_change_matching'] = 4


def match(pos_thresh, neg_thresh, truths, priors, labels, crowd_boxes,
    loc_t, conf_t, idx_t, idx, loc_data):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        pos_thresh: (float) IoU > pos_thresh ==> positive.
        neg_thresh: (float) IoU < neg_thresh ==> negative.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        crowd_boxes: (tensor) All the crowd box annotations or None if there are none.
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds. Note: -1 means neutral.
        idx_t: (tensor) Tensor to be filled w/ the index of the matched gt box for each prior.
        idx: (int) current batch index.
        loc_data: (tensor) The predicted bbox regression coordinates for this batch.
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    decoded_priors = decode(loc_data, priors, cfg.use_yolo_regressors
        ) if cfg.use_prediction_matching else point_form(priors)
    overlaps = jaccard(truths, decoded_priors
        ) if not cfg.use_change_matching else change(truths, decoded_priors)
    best_truth_overlap, best_truth_idx = overlaps.max(0)
    for _ in range(overlaps.size(0)):
        best_prior_overlap, best_prior_idx = overlaps.max(1)
        j = best_prior_overlap.max(0)[1]
        i = best_prior_idx[j]
        overlaps[:, (i)] = -1
        overlaps[(j), :] = -1
        best_truth_overlap[i] = 2
        best_truth_idx[i] = j
    matches = truths[best_truth_idx]
    conf = labels[best_truth_idx] + 1
    conf[best_truth_overlap < pos_thresh] = -1
    conf[best_truth_overlap < neg_thresh] = 0
    if crowd_boxes is not None and cfg.crowd_iou_threshold < 1:
        crowd_overlaps = jaccard(decoded_priors, crowd_boxes, iscrowd=True)
        best_crowd_overlap, best_crowd_idx = crowd_overlaps.max(1)
        conf[(conf <= 0) & (best_crowd_overlap > cfg.crowd_iou_threshold)] = -1
    loc = encode(matches, priors, cfg.use_yolo_regressors)
    loc_t[idx] = loc
    conf_t[idx] = conf
    idx_t[idx] = best_truth_idx


_global_config['bbox_alpha'] = 4


class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, pos_threshold, neg_threshold, negpos_ratio
        ):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold
        self.negpos_ratio = negpos_ratio
        self.l1_expected_area = 20 * 20 / 70 / 70
        self.l1_alpha = 0.1
        if cfg.use_class_balanced_conf:
            self.class_instances = None
            self.total_instances = 0

    def forward(self, net, predictions, targets, masks, num_crowds):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            mask preds, and prior boxes from SSD net.
                loc shape: torch.size(batch_size,num_priors,4)
                conf shape: torch.size(batch_size,num_priors,num_classes)
                masks shape: torch.size(batch_size,num_priors,mask_dim)
                priors shape: torch.size(num_priors,4)
                proto* shape: torch.size(batch_size,mask_h,mask_w,mask_dim)

            targets (list<tensor>): Ground truth boxes and labels for a batch,
                shape: [batch_size][num_objs,5] (last idx is the label).

            masks (list<tensor>): Ground truth masks for each object in each image,
                shape: [batch_size][num_objs,im_height,im_width]

            num_crowds (list<int>): Number of crowd annotations per batch. The crowd
                annotations should be the last num_crowds elements of targets and masks.
            
            * Only if mask_type == lincomb
        """
        loc_data = predictions['loc']
        conf_data = predictions['conf']
        mask_data = predictions['mask']
        priors = predictions['priors']
        if cfg.mask_type == mask_type.lincomb:
            proto_data = predictions['proto']
        score_data = predictions['score'] if cfg.use_mask_scoring else None
        inst_data = predictions['inst'] if cfg.use_instance_coeff else None
        labels = [None] * len(targets)
        batch_size = loc_data.size(0)
        num_priors = priors.size(0)
        num_classes = self.num_classes
        loc_t = loc_data.new(batch_size, num_priors, 4)
        gt_box_t = loc_data.new(batch_size, num_priors, 4)
        conf_t = loc_data.new(batch_size, num_priors).long()
        idx_t = loc_data.new(batch_size, num_priors).long()
        if cfg.use_class_existence_loss:
            class_existence_t = loc_data.new(batch_size, num_classes - 1)
        for idx in range(batch_size):
            truths = targets[idx][:, :-1].data
            labels[idx] = targets[idx][:, (-1)].data.long()
            if cfg.use_class_existence_loss:
                class_existence_t[(idx), :] = torch.eye(num_classes - 1,
                    device=conf_t.get_device())[labels[idx]].max(dim=0)[0]
            cur_crowds = num_crowds[idx]
            if cur_crowds > 0:
                split = lambda x: (x[-cur_crowds:], x[:-cur_crowds])
                crowd_boxes, truths = split(truths)
                _, labels[idx] = split(labels[idx])
                _, masks[idx] = split(masks[idx])
            else:
                crowd_boxes = None
            match(self.pos_threshold, self.neg_threshold, truths, priors.
                data, labels[idx], crowd_boxes, loc_t, conf_t, idx_t, idx,
                loc_data[idx])
            gt_box_t[(idx), :, :] = truths[idx_t[idx]]
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)
        idx_t = Variable(idx_t, requires_grad=False)
        pos = conf_t > 0
        num_pos = pos.sum(dim=1, keepdim=True)
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        losses = {}
        if cfg.train_boxes:
            loc_p = loc_data[pos_idx].view(-1, 4)
            loc_t = loc_t[pos_idx].view(-1, 4)
            losses['B'] = F.smooth_l1_loss(loc_p, loc_t, reduction='sum'
                ) * cfg.bbox_alpha
        if cfg.train_masks:
            if cfg.mask_type == mask_type.direct:
                if cfg.use_gt_bboxes:
                    pos_masks = []
                    for idx in range(batch_size):
                        pos_masks.append(masks[idx][idx_t[idx, pos[idx]]])
                    masks_t = torch.cat(pos_masks, 0)
                    masks_p = mask_data[(pos), :].view(-1, cfg.mask_dim)
                    losses['M'] = F.binary_cross_entropy(torch.clamp(
                        masks_p, 0, 1), masks_t, reduction='sum'
                        ) * cfg.mask_alpha
                else:
                    losses['M'] = self.direct_mask_loss(pos_idx, idx_t,
                        loc_data, mask_data, priors, masks)
            elif cfg.mask_type == mask_type.lincomb:
                ret = self.lincomb_mask_loss(pos, idx_t, loc_data,
                    mask_data, priors, proto_data, masks, gt_box_t,
                    score_data, inst_data, labels)
                if cfg.use_maskiou:
                    loss, maskiou_targets = ret
                else:
                    loss = ret
                losses.update(loss)
                if cfg.mask_proto_loss is not None:
                    if cfg.mask_proto_loss == 'l1':
                        losses['P'] = torch.mean(torch.abs(proto_data)
                            ) / self.l1_expected_area * self.l1_alpha
                    elif cfg.mask_proto_loss == 'disj':
                        losses['P'] = -torch.mean(torch.max(F.log_softmax(
                            proto_data, dim=-1), dim=-1)[0])
        if cfg.use_focal_loss:
            if cfg.use_sigmoid_focal_loss:
                losses['C'] = self.focal_conf_sigmoid_loss(conf_data, conf_t)
            elif cfg.use_objectness_score:
                losses['C'] = self.focal_conf_objectness_loss(conf_data, conf_t
                    )
            else:
                losses['C'] = self.focal_conf_loss(conf_data, conf_t)
        elif cfg.use_objectness_score:
            losses['C'] = self.conf_objectness_loss(conf_data, conf_t,
                batch_size, loc_p, loc_t, priors)
        else:
            losses['C'] = self.ohem_conf_loss(conf_data, conf_t, pos,
                batch_size)
        if cfg.use_maskiou and maskiou_targets is not None:
            losses['I'] = self.mask_iou_loss(net, maskiou_targets)
        if cfg.use_class_existence_loss:
            losses['E'] = self.class_existence_loss(predictions['classes'],
                class_existence_t)
        if cfg.use_semantic_segmentation_loss:
            losses['S'] = self.semantic_segmentation_loss(predictions[
                'segm'], masks, labels)
        total_num_pos = num_pos.data.sum().float()
        for k in losses:
            if k not in ('P', 'E', 'S'):
                losses[k] /= total_num_pos
            else:
                losses[k] /= batch_size
        return losses

    def class_existence_loss(self, class_data, class_existence_t):
        return cfg.class_existence_alpha * F.binary_cross_entropy_with_logits(
            class_data, class_existence_t, reduction='sum')

    def semantic_segmentation_loss(self, segment_data, mask_t, class_t,
        interpolation_mode='bilinear'):
        batch_size, num_classes, mask_h, mask_w = segment_data.size()
        loss_s = 0
        for idx in range(batch_size):
            cur_segment = segment_data[idx]
            cur_class_t = class_t[idx]
            with torch.no_grad():
                downsampled_masks = F.interpolate(mask_t[idx].unsqueeze(0),
                    (mask_h, mask_w), mode=interpolation_mode,
                    align_corners=False).squeeze(0)
                downsampled_masks = downsampled_masks.gt(0.5).float()
                segment_t = torch.zeros_like(cur_segment, requires_grad=False)
                for obj_idx in range(downsampled_masks.size(0)):
                    segment_t[cur_class_t[obj_idx]] = torch.max(segment_t[
                        cur_class_t[obj_idx]], downsampled_masks[obj_idx])
            loss_s += F.binary_cross_entropy_with_logits(cur_segment,
                segment_t, reduction='sum')
        return loss_s / mask_h / mask_w * cfg.semantic_segmentation_alpha

    def ohem_conf_loss(self, conf_data, conf_t, pos, num):
        batch_conf = conf_data.view(-1, self.num_classes)
        if cfg.ohem_use_most_confident:
            batch_conf = F.softmax(batch_conf, dim=1)
            loss_c, _ = batch_conf[:, 1:].max(dim=1)
        else:
            loss_c = log_sum_exp(batch_conf) - batch_conf[:, (0)]
        loss_c = loss_c.view(num, -1)
        loss_c[pos] = 0
        loss_c[conf_t < 0] = 0
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio * num_pos, max=pos.size(1) - 1)
        neg = idx_rank < num_neg.expand_as(idx_rank)
        neg[pos] = 0
        neg[conf_t < 0] = 0
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes
            )
        targets_weighted = conf_t[(pos + neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='none')
        if cfg.use_class_balanced_conf:
            if self.class_instances is None:
                self.class_instances = torch.zeros(self.num_classes, device
                    =targets_weighted.device)
            classes, counts = targets_weighted.unique(return_counts=True)
            for _cls, _cnt in zip(classes.cpu().numpy(), counts.cpu().numpy()):
                self.class_instances[_cls] += _cnt
            self.total_instances += targets_weighted.size(0)
            weighting = 1 - self.class_instances[targets_weighted
                ] / self.total_instances
            weighting = torch.clamp(weighting, min=1 / self.num_classes)
            avg_weight = (self.num_classes - 1) / self.num_classes
            loss_c = (loss_c * weighting).sum() / avg_weight
        else:
            loss_c = loss_c.sum()
        return cfg.conf_alpha * loss_c

    def focal_conf_loss(self, conf_data, conf_t):
        """
        Focal loss as described in https://arxiv.org/pdf/1708.02002.pdf
        Adapted from https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
        Note that this uses softmax and not the original sigmoid from the paper.
        """
        conf_t = conf_t.view(-1)
        conf_data = conf_data.view(-1, conf_data.size(-1))
        keep = (conf_t >= 0).float()
        conf_t[conf_t < 0] = 0
        logpt = F.log_softmax(conf_data, dim=-1)
        logpt = logpt.gather(1, conf_t.unsqueeze(-1))
        logpt = logpt.view(-1)
        pt = logpt.exp()
        background = (conf_t == 0).float()
        at = (1 - cfg.focal_loss_alpha) * background + cfg.focal_loss_alpha * (
            1 - background)
        loss = -at * (1 - pt) ** cfg.focal_loss_gamma * logpt
        return cfg.conf_alpha * (loss * keep).sum()

    def focal_conf_sigmoid_loss(self, conf_data, conf_t):
        """
        Focal loss but using sigmoid like the original paper.
        Note: To make things mesh easier, the network still predicts 81 class confidences in this mode.
              Because retinanet originally only predicts 80, we simply just don't use conf_data[..., 0]
        """
        num_classes = conf_data.size(-1)
        conf_t = conf_t.view(-1)
        conf_data = conf_data.view(-1, num_classes)
        keep = (conf_t >= 0).float()
        conf_t[conf_t < 0] = 0
        conf_one_t = torch.eye(num_classes, device=conf_t.get_device())[conf_t]
        conf_pm_t = conf_one_t * 2 - 1
        logpt = F.logsigmoid(conf_data * conf_pm_t)
        pt = logpt.exp()
        at = cfg.focal_loss_alpha * conf_one_t + (1 - cfg.focal_loss_alpha) * (
            1 - conf_one_t)
        at[..., 0] = 0
        loss = -at * (1 - pt) ** cfg.focal_loss_gamma * logpt
        loss = keep * loss.sum(dim=-1)
        return cfg.conf_alpha * loss.sum()

    def focal_conf_objectness_loss(self, conf_data, conf_t):
        """
        Instead of using softmax, use class[0] to be the objectness score and do sigmoid focal loss on that.
        Then for the rest of the classes, softmax them and apply CE for only the positive examples.

        If class[0] = 1 implies forground and class[0] = 0 implies background then you achieve something
        similar during test-time to softmax by setting class[1:] = softmax(class[1:]) * class[0] and invert class[0].
        """
        conf_t = conf_t.view(-1)
        conf_data = conf_data.view(-1, conf_data.size(-1))
        keep = (conf_t >= 0).float()
        conf_t[conf_t < 0] = 0
        background = (conf_t == 0).float()
        at = (1 - cfg.focal_loss_alpha) * background + cfg.focal_loss_alpha * (
            1 - background)
        logpt = F.logsigmoid(conf_data[:, (0)]) * (1 - background
            ) + F.logsigmoid(-conf_data[:, (0)]) * background
        pt = logpt.exp()
        obj_loss = -at * (1 - pt) ** cfg.focal_loss_gamma * logpt
        pos_mask = conf_t > 0
        conf_data_pos = conf_data[:, 1:][pos_mask]
        conf_t_pos = conf_t[pos_mask] - 1
        class_loss = F.cross_entropy(conf_data_pos, conf_t_pos, reduction='sum'
            )
        return cfg.conf_alpha * (class_loss + (obj_loss * keep).sum())

    def conf_objectness_loss(self, conf_data, conf_t, batch_size, loc_p,
        loc_t, priors):
        """
        Instead of using softmax, use class[0] to be p(obj) * p(IoU) as in YOLO.
        Then for the rest of the classes, softmax them and apply CE for only the positive examples.
        """
        conf_t = conf_t.view(-1)
        conf_data = conf_data.view(-1, conf_data.size(-1))
        pos_mask = conf_t > 0
        neg_mask = conf_t == 0
        obj_data = conf_data[:, (0)]
        obj_data_pos = obj_data[pos_mask]
        obj_data_neg = obj_data[neg_mask]
        obj_neg_loss = -F.logsigmoid(-obj_data_neg).sum()
        with torch.no_grad():
            pos_priors = priors.unsqueeze(0).expand(batch_size, -1, -1
                ).reshape(-1, 4)[(pos_mask), :]
            boxes_pred = decode(loc_p, pos_priors, cfg.use_yolo_regressors)
            boxes_targ = decode(loc_t, pos_priors, cfg.use_yolo_regressors)
            iou_targets = elemwise_box_iou(boxes_pred, boxes_targ)
        obj_pos_loss = -iou_targets * F.logsigmoid(obj_data_pos) - (1 -
            iou_targets) * F.logsigmoid(-obj_data_pos)
        obj_pos_loss = obj_pos_loss.sum()
        conf_data_pos = conf_data[:, 1:][pos_mask]
        conf_t_pos = conf_t[pos_mask] - 1
        class_loss = F.cross_entropy(conf_data_pos, conf_t_pos, reduction='sum'
            )
        return cfg.conf_alpha * (class_loss + obj_pos_loss + obj_neg_loss)

    def direct_mask_loss(self, pos_idx, idx_t, loc_data, mask_data, priors,
        masks):
        """ Crops the gt masks using the predicted bboxes, scales them down, and outputs the BCE loss. """
        loss_m = 0
        for idx in range(mask_data.size(0)):
            with torch.no_grad():
                cur_pos_idx = pos_idx[(idx), :, :]
                cur_pos_idx_squeezed = cur_pos_idx[:, (1)]
                pos_bboxes = decode(loc_data[(idx), :, :], priors.data, cfg
                    .use_yolo_regressors)
                pos_bboxes = pos_bboxes[cur_pos_idx].view(-1, 4).clamp(0, 1)
                pos_lookup = idx_t[idx, cur_pos_idx_squeezed]
                cur_masks = masks[idx]
                pos_masks = cur_masks[(pos_lookup), :, :]
                num_pos, img_height, img_width = pos_masks.size()
                x1, x2 = sanitize_coordinates(pos_bboxes[:, (0)],
                    pos_bboxes[:, (2)], img_width)
                y1, y2 = sanitize_coordinates(pos_bboxes[:, (1)],
                    pos_bboxes[:, (3)], img_height)
                scaled_masks = []
                for jdx in range(num_pos):
                    tmp_mask = pos_masks[(jdx), y1[jdx]:y2[jdx], x1[jdx]:x2
                        [jdx]]
                    while tmp_mask.dim() < 2:
                        tmp_mask = tmp_mask.unsqueeze(0)
                    new_mask = F.adaptive_avg_pool2d(tmp_mask.unsqueeze(0),
                        cfg.mask_size)
                    scaled_masks.append(new_mask.view(1, -1))
                mask_t = torch.cat(scaled_masks, 0).gt(0.5).float()
            pos_mask_data = mask_data[(idx), (cur_pos_idx_squeezed), :]
            loss_m += F.binary_cross_entropy(torch.clamp(pos_mask_data, 0, 
                1), mask_t, reduction='sum') * cfg.mask_alpha
        return loss_m

    def coeff_diversity_loss(self, coeffs, instance_t):
        """
        coeffs     should be size [num_pos, num_coeffs]
        instance_t should be size [num_pos] and be values from 0 to num_instances-1
        """
        num_pos = coeffs.size(0)
        instance_t = instance_t.view(-1)
        coeffs_norm = F.normalize(coeffs, dim=1)
        cos_sim = coeffs_norm @ coeffs_norm.t()
        inst_eq = (instance_t[:, (None)].expand_as(cos_sim) == instance_t[(
            None), :].expand_as(cos_sim)).float()
        cos_sim = (cos_sim + 1) / 2
        loss = (1 - cos_sim) * inst_eq + cos_sim * (1 - inst_eq)
        return cfg.mask_proto_coeff_diversity_alpha * loss.sum() / num_pos

    def lincomb_mask_loss(self, pos, idx_t, loc_data, mask_data, priors,
        proto_data, masks, gt_box_t, score_data, inst_data, labels,
        interpolation_mode='bilinear'):
        mask_h = proto_data.size(1)
        mask_w = proto_data.size(2)
        process_gt_bboxes = (cfg.mask_proto_normalize_emulate_roi_pooling or
            cfg.mask_proto_crop)
        if cfg.mask_proto_remove_empty_masks:
            pos = pos.clone()
        loss_m = 0
        loss_d = 0
        maskiou_t_list = []
        maskiou_net_input_list = []
        label_t_list = []
        for idx in range(mask_data.size(0)):
            with torch.no_grad():
                downsampled_masks = F.interpolate(masks[idx].unsqueeze(0),
                    (mask_h, mask_w), mode=interpolation_mode,
                    align_corners=False).squeeze(0)
                downsampled_masks = downsampled_masks.permute(1, 2, 0
                    ).contiguous()
                if cfg.mask_proto_binarize_downsampled_gt:
                    downsampled_masks = downsampled_masks.gt(0.5).float()
                if cfg.mask_proto_remove_empty_masks:
                    very_small_masks = downsampled_masks.sum(dim=(0, 1)
                        ) <= 0.0001
                    for i in range(very_small_masks.size(0)):
                        if very_small_masks[i]:
                            pos[idx, idx_t[idx] == i] = 0
                if cfg.mask_proto_reweight_mask_loss:
                    if not cfg.mask_proto_binarize_downsampled_gt:
                        bin_gt = downsampled_masks.gt(0.5).float()
                    else:
                        bin_gt = downsampled_masks
                    gt_foreground_norm = bin_gt / (torch.sum(bin_gt, dim=(0,
                        1), keepdim=True) + 0.0001)
                    gt_background_norm = (1 - bin_gt) / (torch.sum(1 -
                        bin_gt, dim=(0, 1), keepdim=True) + 0.0001)
                    mask_reweighting = (gt_foreground_norm * cfg.
                        mask_proto_reweight_coeff + gt_background_norm)
                    mask_reweighting *= mask_h * mask_w
            cur_pos = pos[idx]
            pos_idx_t = idx_t[idx, cur_pos]
            if process_gt_bboxes:
                if cfg.mask_proto_crop_with_pred_box:
                    pos_gt_box_t = decode(loc_data[(idx), :, :], priors.
                        data, cfg.use_yolo_regressors)[cur_pos]
                else:
                    pos_gt_box_t = gt_box_t[idx, cur_pos]
            if pos_idx_t.size(0) == 0:
                continue
            proto_masks = proto_data[idx]
            proto_coef = mask_data[(idx), (cur_pos), :]
            if cfg.use_mask_scoring:
                mask_scores = score_data[(idx), (cur_pos), :]
            if cfg.mask_proto_coeff_diversity_loss:
                if inst_data is not None:
                    div_coeffs = inst_data[(idx), (cur_pos), :]
                else:
                    div_coeffs = proto_coef
                loss_d += self.coeff_diversity_loss(div_coeffs, pos_idx_t)
            old_num_pos = proto_coef.size(0)
            if old_num_pos > cfg.masks_to_train:
                perm = torch.randperm(proto_coef.size(0))
                select = perm[:cfg.masks_to_train]
                proto_coef = proto_coef[(select), :]
                pos_idx_t = pos_idx_t[select]
                if process_gt_bboxes:
                    pos_gt_box_t = pos_gt_box_t[(select), :]
                if cfg.use_mask_scoring:
                    mask_scores = mask_scores[(select), :]
            num_pos = proto_coef.size(0)
            mask_t = downsampled_masks[:, :, (pos_idx_t)]
            label_t = labels[idx][pos_idx_t]
            pred_masks = proto_masks @ proto_coef.t()
            pred_masks = cfg.mask_proto_mask_activation(pred_masks)
            if cfg.mask_proto_double_loss:
                if cfg.mask_proto_mask_activation == activation_func.sigmoid:
                    pre_loss = F.binary_cross_entropy(torch.clamp(
                        pred_masks, 0, 1), mask_t, reduction='sum')
                else:
                    pre_loss = F.smooth_l1_loss(pred_masks, mask_t,
                        reduction='sum')
                loss_m += cfg.mask_proto_double_loss_alpha * pre_loss
            if cfg.mask_proto_crop:
                pred_masks = crop(pred_masks, pos_gt_box_t)
            if cfg.mask_proto_mask_activation == activation_func.sigmoid:
                pre_loss = F.binary_cross_entropy(torch.clamp(pred_masks, 0,
                    1), mask_t, reduction='none')
            else:
                pre_loss = F.smooth_l1_loss(pred_masks, mask_t, reduction=
                    'none')
            if cfg.mask_proto_normalize_mask_loss_by_sqrt_area:
                gt_area = torch.sum(mask_t, dim=(0, 1), keepdim=True)
                pre_loss = pre_loss / (torch.sqrt(gt_area) + 0.0001)
            if cfg.mask_proto_reweight_mask_loss:
                pre_loss = pre_loss * mask_reweighting[:, :, (pos_idx_t)]
            if cfg.mask_proto_normalize_emulate_roi_pooling:
                weight = mask_h * mask_w if cfg.mask_proto_crop else 1
                pos_gt_csize = center_size(pos_gt_box_t)
                gt_box_width = pos_gt_csize[:, (2)] * mask_w
                gt_box_height = pos_gt_csize[:, (3)] * mask_h
                pre_loss = pre_loss.sum(dim=(0, 1)
                    ) / gt_box_width / gt_box_height * weight
            if old_num_pos > num_pos:
                pre_loss *= old_num_pos / num_pos
            loss_m += torch.sum(pre_loss)
            if cfg.use_maskiou:
                if cfg.discard_mask_area > 0:
                    gt_mask_area = torch.sum(mask_t, dim=(0, 1))
                    select = gt_mask_area > cfg.discard_mask_area
                    if torch.sum(select) < 1:
                        continue
                    pos_gt_box_t = pos_gt_box_t[(select), :]
                    pred_masks = pred_masks[:, :, (select)]
                    mask_t = mask_t[:, :, (select)]
                    label_t = label_t[select]
                maskiou_net_input = pred_masks.permute(2, 0, 1).contiguous(
                    ).unsqueeze(1)
                pred_masks = pred_masks.gt(0.5).float()
                maskiou_t = self._mask_iou(pred_masks, mask_t)
                maskiou_net_input_list.append(maskiou_net_input)
                maskiou_t_list.append(maskiou_t)
                label_t_list.append(label_t)
        losses = {'M': loss_m * cfg.mask_alpha / mask_h / mask_w}
        if cfg.mask_proto_coeff_diversity_loss:
            losses['D'] = loss_d
        if cfg.use_maskiou:
            if len(maskiou_t_list) == 0:
                return losses, None
            maskiou_t = torch.cat(maskiou_t_list)
            label_t = torch.cat(label_t_list)
            maskiou_net_input = torch.cat(maskiou_net_input_list)
            num_samples = maskiou_t.size(0)
            if (cfg.maskious_to_train > 0 and num_samples > cfg.
                maskious_to_train):
                perm = torch.randperm(num_samples)
                select = perm[:cfg.masks_to_train]
                maskiou_t = maskiou_t[select]
                label_t = label_t[select]
                maskiou_net_input = maskiou_net_input[select]
            return losses, [maskiou_net_input, maskiou_t, label_t]
        return losses

    def _mask_iou(self, mask1, mask2):
        intersection = torch.sum(mask1 * mask2, dim=(0, 1))
        area1 = torch.sum(mask1, dim=(0, 1))
        area2 = torch.sum(mask2, dim=(0, 1))
        union = area1 + area2 - intersection
        ret = intersection / union
        return ret

    def mask_iou_loss(self, net, maskiou_targets):
        maskiou_net_input, maskiou_t, label_t = maskiou_targets
        maskiou_p = net.maskiou_net(maskiou_net_input)
        label_t = label_t[:, (None)]
        maskiou_p = torch.gather(maskiou_p, dim=1, index=label_t).view(-1)
        loss_i = F.smooth_l1_loss(maskiou_p, maskiou_t, reduction='sum')
        return loss_i * cfg.maskiou_alpha


def enforce_size(img, targets, masks, num_crowds, new_w, new_h):
    """ Ensures that the image is the given size without distorting aspect ratio. """
    with torch.no_grad():
        _, h, w = img.size()
        if h == new_h and w == new_w:
            return img, targets, masks, num_crowds
        w_prime = new_w
        h_prime = h * new_w / w
        if h_prime > new_h:
            w_prime *= new_h / h_prime
            h_prime = new_h
        w_prime = int(w_prime)
        h_prime = int(h_prime)
        img = F.interpolate(img.unsqueeze(0), (h_prime, w_prime), mode=
            'bilinear', align_corners=False)
        img.squeeze_(0)
        masks = F.interpolate(masks.unsqueeze(0), (h_prime, w_prime), mode=
            'bilinear', align_corners=False)
        masks.squeeze_(0)
        targets[:, ([0, 2])] *= w_prime / new_w
        targets[:, ([1, 3])] *= h_prime / new_h
        pad_dims = 0, new_w - w_prime, 0, new_h - h_prime
        img = F.pad(img, pad_dims, mode='constant', value=0)
        masks = F.pad(masks, pad_dims, mode='constant', value=0)
        return img, targets, masks, num_crowds


def gradinator(x):
    x.requires_grad = False
    return x


_global_config['batch_size'] = 4


_global_config['cuda'] = 4


_global_config['preserve_aspect_ratio'] = 4


def prepare_data(datum, devices: list=None, allocation: list=None):
    with torch.no_grad():
        if devices is None:
            devices = ['cuda:0'] if args.cuda else ['cpu']
        if allocation is None:
            allocation = [args.batch_size // len(devices)] * (len(devices) - 1)
            allocation.append(args.batch_size - sum(allocation))
        images, (targets, masks, num_crowds) = datum
        cur_idx = 0
        for device, alloc in zip(devices, allocation):
            for _ in range(alloc):
                images[cur_idx] = gradinator(images[cur_idx].to(device))
                targets[cur_idx] = gradinator(targets[cur_idx].to(device))
                masks[cur_idx] = gradinator(masks[cur_idx].to(device))
                cur_idx += 1
        if cfg.preserve_aspect_ratio:
            _, h, w = images[random.randint(0, len(images) - 1)].size()
            for idx, (image, target, mask, num_crowd) in enumerate(zip(
                images, targets, masks, num_crowds)):
                images[idx], targets[idx], masks[idx], num_crowds[idx
                    ] = enforce_size(image, target, mask, num_crowd, w, h)
        cur_idx = 0
        split_images, split_targets, split_masks, split_numcrowds = [[None for
            alloc in allocation] for _ in range(4)]
        for device_idx, alloc in enumerate(allocation):
            split_images[device_idx] = torch.stack(images[cur_idx:cur_idx +
                alloc], dim=0)
            split_targets[device_idx] = targets[cur_idx:cur_idx + alloc]
            split_masks[device_idx] = masks[cur_idx:cur_idx + alloc]
            split_numcrowds[device_idx] = num_crowds[cur_idx:cur_idx + alloc]
            cur_idx += alloc
        return split_images, split_targets, split_masks, split_numcrowds


_global_config['batch_alloc'] = 4


class CustomDataParallel(nn.DataParallel):
    """
    This is a custom version of DataParallel that works better with our training data.
    It should also be faster than the general case.
    """

    def scatter(self, inputs, kwargs, device_ids):
        devices = [('cuda:' + str(x)) for x in device_ids]
        splits = prepare_data(inputs[0], devices, allocation=args.batch_alloc)
        return [[split[device_idx] for split in splits] for device_idx in
            range(len(devices))], [kwargs] * len(devices)

    def gather(self, outputs, output_device):
        out = {}
        for k in outputs[0]:
            out[k] = torch.stack([output[k].to(output_device) for output in
                outputs])
        return out


MEANS = 103.94, 116.78, 123.68


_global_config['discard_box_height'] = 4


_global_config['max_size'] = 4


_global_config['discard_box_width'] = 4


class Resize(object):
    """ If preserve_aspect_ratio is true, this resizes to an approximate area of max_size * max_size """

    @staticmethod
    def calc_size_preserve_ar(img_w, img_h, max_size):
        """ I mathed this one out on the piece of paper. Resulting width*height = approx max_size^2 """
        ratio = sqrt(img_w / img_h)
        w = max_size * ratio
        h = max_size / ratio
        return int(w), int(h)

    def __init__(self, resize_gt=True):
        self.resize_gt = resize_gt
        self.max_size = cfg.max_size
        self.preserve_aspect_ratio = cfg.preserve_aspect_ratio

    def __call__(self, image, masks, boxes, labels=None):
        img_h, img_w, _ = image.shape
        if self.preserve_aspect_ratio:
            width, height = Resize.calc_size_preserve_ar(img_w, img_h, self
                .max_size)
        else:
            width, height = self.max_size, self.max_size
        image = cv2.resize(image, (width, height))
        if self.resize_gt:
            masks = masks.transpose((1, 2, 0))
            masks = cv2.resize(masks, (width, height))
            if len(masks.shape) == 2:
                masks = np.expand_dims(masks, 0)
            else:
                masks = masks.transpose((2, 0, 1))
            boxes[:, ([0, 2])] *= width / img_w
            boxes[:, ([1, 3])] *= height / img_h
        w = boxes[:, (2)] - boxes[:, (0)]
        h = boxes[:, (3)] - boxes[:, (1)]
        keep = (w > cfg.discard_box_width) * (h > cfg.discard_box_height)
        masks = masks[keep]
        boxes = boxes[keep]
        labels['labels'] = labels['labels'][keep]
        labels['num_crowds'] = (labels['labels'] < 0).sum()
        return image, masks, boxes, labels


STD = 57.38, 57.12, 58.4


_global_config['backbone'] = 4


class FastBaseTransform(torch.nn.Module):
    """
    Transform that does all operations on the GPU for super speed.
    This doesn't suppport a lot of config settings and should only be used for production.
    Maintain this as necessary.
    """

    def __init__(self):
        super().__init__()
        self.mean = torch.Tensor(MEANS).float()[(None), :, (None), (None)]
        self.std = torch.Tensor(STD).float()[(None), :, (None), (None)]
        self.transform = cfg.backbone.transform

    def forward(self, img):
        self.mean = self.mean.to(img.device)
        self.std = self.std.to(img.device)
        if cfg.preserve_aspect_ratio:
            _, h, w, _ = img.size()
            img_size = Resize.calc_size_preserve_ar(w, h, cfg.max_size)
            img_size = img_size[1], img_size[0]
        else:
            img_size = cfg.max_size, cfg.max_size
        img = img.permute(0, 3, 1, 2).contiguous()
        img = F.interpolate(img, img_size, mode='bilinear', align_corners=False
            )
        if self.transform.normalize:
            img = (img - self.mean) / self.std
        elif self.transform.subtract_means:
            img = img - self.mean
        elif self.transform.to_float:
            img = img / 255
        if self.transform.channel_order != 'RGB':
            raise NotImplementedError
        img = img[:, (2, 1, 0), :, :].contiguous()
        return img


class Concat(nn.Module):

    def __init__(self, nets, extra_params):
        super().__init__()
        self.nets = nn.ModuleList(nets)
        self.extra_params = extra_params

    def forward(self, x):
        return torch.cat([net(x) for net in self.nets], dim=1, **self.
            extra_params)


def make_net(in_channels, conf, include_last_relu=True):
    """
    A helper function to take a config setting and turn it into a network.
    Used by protonet and extrahead. Returns (network, out_channels)
    """

    def make_layer(layer_cfg):
        nonlocal in_channels
        if isinstance(layer_cfg[0], str):
            layer_name = layer_cfg[0]
            if layer_name == 'cat':
                nets = [make_net(in_channels, x) for x in layer_cfg[1]]
                layer = Concat([net[0] for net in nets], layer_cfg[2])
                num_channels = sum([net[1] for net in nets])
        else:
            num_channels = layer_cfg[0]
            kernel_size = layer_cfg[1]
            if kernel_size > 0:
                layer = nn.Conv2d(in_channels, num_channels, kernel_size,
                    **layer_cfg[2])
            elif num_channels is None:
                layer = InterpolateModule(scale_factor=-kernel_size, mode=
                    'bilinear', align_corners=False, **layer_cfg[2])
            else:
                layer = nn.ConvTranspose2d(in_channels, num_channels, -
                    kernel_size, **layer_cfg[2])
        in_channels = num_channels if num_channels is not None else in_channels
        return [layer, nn.ReLU(inplace=True)]
    net = sum([make_layer(x) for x in conf], [])
    if not include_last_relu:
        net = net[:-1]
    return nn.Sequential(*net), in_channels


_global_config['use_prediction_module'] = 4


_global_config['extra_layers'] = 1


_global_config['_tmp_img_w'] = 4


_global_config['use_mask_scoring'] = 4


_global_config['num_classes'] = 4


_global_config['num_heads'] = 4


_global_config['use_instance_coeff'] = 4


_global_config['eval_mask_branch'] = 4


_global_config['mask_proto_coeff_gate'] = 4


_global_config['head_layer_params'] = 1


_global_config['mask_proto_split_prototypes_by_head'] = 4


_global_config['mask_type'] = 4


_global_config['num_instance_coeffs'] = 4


_global_config['extra_head_net'] = 4


_global_config['mask_proto_prototypes_as_features'] = 4


_global_config['_tmp_img_h'] = 4


_global_config['mask_proto_coeff_activation'] = 4


_global_config['mask_dim'] = 4


class PredictionModule(nn.Module):
    """
    The (c) prediction module adapted from DSSD:
    https://arxiv.org/pdf/1701.06659.pdf

    Note that this is slightly different to the module in the paper
    because the Bottleneck block actually has a 3x3 convolution in
    the middle instead of a 1x1 convolution. Though, I really can't
    be arsed to implement it myself, and, who knows, this might be
    better.

    Args:
        - in_channels:   The input feature size.
        - out_channels:  The output feature size (must be a multiple of 4).
        - aspect_ratios: A list of lists of priorbox aspect ratios (one list per scale).
        - scales:        A list of priorbox scales relative to this layer's convsize.
                         For instance: If this layer has convouts of size 30x30 for
                                       an image of size 600x600, the 'default' (scale
                                       of 1) for this layer would produce bounding
                                       boxes with an area of 20x20px. If the scale is
                                       .5 on the other hand, this layer would consider
                                       bounding boxes with area 10x10px, etc.
        - parent:        If parent is a PredictionModule, this module will use all the layers
                         from parent instead of from this module.
    """

    def __init__(self, in_channels, out_channels=1024, aspect_ratios=[[1]],
        scales=[1], parent=None, index=0):
        super().__init__()
        self.num_classes = cfg.num_classes
        self.mask_dim = cfg.mask_dim
        self.num_priors = sum(len(x) * len(scales) for x in aspect_ratios)
        self.parent = [parent]
        self.index = index
        self.num_heads = cfg.num_heads
        if (cfg.mask_proto_split_prototypes_by_head and cfg.mask_type ==
            mask_type.lincomb):
            self.mask_dim = self.mask_dim // self.num_heads
        if cfg.mask_proto_prototypes_as_features:
            in_channels += self.mask_dim
        if parent is None:
            if cfg.extra_head_net is None:
                out_channels = in_channels
            else:
                self.upfeature, out_channels = make_net(in_channels, cfg.
                    extra_head_net)
            if cfg.use_prediction_module:
                self.block = Bottleneck(out_channels, out_channels // 4)
                self.conv = nn.Conv2d(out_channels, out_channels,
                    kernel_size=1, bias=True)
                self.bn = nn.BatchNorm2d(out_channels)
            self.bbox_layer = nn.Conv2d(out_channels, self.num_priors * 4,
                **cfg.head_layer_params)
            self.conf_layer = nn.Conv2d(out_channels, self.num_priors *
                self.num_classes, **cfg.head_layer_params)
            self.mask_layer = nn.Conv2d(out_channels, self.num_priors *
                self.mask_dim, **cfg.head_layer_params)
            if cfg.use_mask_scoring:
                self.score_layer = nn.Conv2d(out_channels, self.num_priors,
                    **cfg.head_layer_params)
            if cfg.use_instance_coeff:
                self.inst_layer = nn.Conv2d(out_channels, self.num_priors *
                    cfg.num_instance_coeffs, **cfg.head_layer_params)

            def make_extra(num_layers):
                if num_layers == 0:
                    return lambda x: x
                else:
                    return nn.Sequential(*sum([[nn.Conv2d(out_channels,
                        out_channels, kernel_size=3, padding=1), nn.ReLU(
                        inplace=True)] for _ in range(num_layers)], []))
            self.bbox_extra, self.conf_extra, self.mask_extra = [make_extra
                (x) for x in cfg.extra_layers]
            if (cfg.mask_type == mask_type.lincomb and cfg.
                mask_proto_coeff_gate):
                self.gate_layer = nn.Conv2d(out_channels, self.num_priors *
                    self.mask_dim, kernel_size=3, padding=1)
        self.aspect_ratios = aspect_ratios
        self.scales = scales
        self.priors = None
        self.last_conv_size = None
        self.last_img_size = None

    def forward(self, x):
        """
        Args:
            - x: The convOut from a layer in the backbone network
                 Size: [batch_size, in_channels, conv_h, conv_w])

        Returns a tuple (bbox_coords, class_confs, mask_output, prior_boxes) with sizes
            - bbox_coords: [batch_size, conv_h*conv_w*num_priors, 4]
            - class_confs: [batch_size, conv_h*conv_w*num_priors, num_classes]
            - mask_output: [batch_size, conv_h*conv_w*num_priors, mask_dim]
            - prior_boxes: [conv_h*conv_w*num_priors, 4]
        """
        src = self if self.parent[0] is None else self.parent[0]
        conv_h = x.size(2)
        conv_w = x.size(3)
        if cfg.extra_head_net is not None:
            x = src.upfeature(x)
        if cfg.use_prediction_module:
            a = src.block(x)
            b = src.conv(x)
            b = src.bn(b)
            b = F.relu(b)
            x = a + b
        bbox_x = src.bbox_extra(x)
        conf_x = src.conf_extra(x)
        mask_x = src.mask_extra(x)
        bbox = src.bbox_layer(bbox_x).permute(0, 2, 3, 1).contiguous().view(x
            .size(0), -1, 4)
        conf = src.conf_layer(conf_x).permute(0, 2, 3, 1).contiguous().view(x
            .size(0), -1, self.num_classes)
        if cfg.eval_mask_branch:
            mask = src.mask_layer(mask_x).permute(0, 2, 3, 1).contiguous(
                ).view(x.size(0), -1, self.mask_dim)
        else:
            mask = torch.zeros(x.size(0), bbox.size(1), self.mask_dim,
                device=bbox.device)
        if cfg.use_mask_scoring:
            score = src.score_layer(x).permute(0, 2, 3, 1).contiguous().view(x
                .size(0), -1, 1)
        if cfg.use_instance_coeff:
            inst = src.inst_layer(x).permute(0, 2, 3, 1).contiguous().view(x
                .size(0), -1, cfg.num_instance_coeffs)
        if cfg.use_yolo_regressors:
            bbox[:, :, :2] = torch.sigmoid(bbox[:, :, :2]) - 0.5
            bbox[:, :, (0)] /= conv_w
            bbox[:, :, (1)] /= conv_h
        if cfg.eval_mask_branch:
            if cfg.mask_type == mask_type.direct:
                mask = torch.sigmoid(mask)
            elif cfg.mask_type == mask_type.lincomb:
                mask = cfg.mask_proto_coeff_activation(mask)
                if cfg.mask_proto_coeff_gate:
                    gate = src.gate_layer(x).permute(0, 2, 3, 1).contiguous(
                        ).view(x.size(0), -1, self.mask_dim)
                    mask = mask * torch.sigmoid(gate)
        if (cfg.mask_proto_split_prototypes_by_head and cfg.mask_type ==
            mask_type.lincomb):
            mask = F.pad(mask, (self.index * self.mask_dim, (self.num_heads -
                self.index - 1) * self.mask_dim), mode='constant', value=0)
        priors = self.make_priors(conv_h, conv_w, x.device)
        preds = {'loc': bbox, 'conf': conf, 'mask': mask, 'priors': priors}
        if cfg.use_mask_scoring:
            preds['score'] = score
        if cfg.use_instance_coeff:
            preds['inst'] = inst
        return preds

    def make_priors(self, conv_h, conv_w, device):
        """ Note that priors are [x,y,width,height] where (x,y) is the center of the box. """
        global prior_cache
        size = conv_h, conv_w
        with timer.env('makepriors'):
            if self.last_img_size != (cfg._tmp_img_w, cfg._tmp_img_h):
                prior_data = []
                for j, i in product(range(conv_h), range(conv_w)):
                    x = (i + 0.5) / conv_w
                    y = (j + 0.5) / conv_h
                    for ars in self.aspect_ratios:
                        for scale in self.scales:
                            for ar in ars:
                                if not cfg.backbone.preapply_sqrt:
                                    ar = sqrt(ar)
                                if cfg.backbone.use_pixel_scales:
                                    w = scale * ar / cfg.max_size
                                    h = scale / ar / cfg.max_size
                                else:
                                    w = scale * ar / conv_w
                                    h = scale / ar / conv_h
                                if cfg.backbone.use_square_anchors:
                                    h = w
                                prior_data += [x, y, w, h]
                self.priors = torch.Tensor(prior_data, device=device).view(
                    -1, 4).detach()
                self.priors.requires_grad = False
                self.last_img_size = cfg._tmp_img_w, cfg._tmp_img_h
                self.last_conv_size = conv_w, conv_h
                prior_cache[size] = None
            elif self.priors.device != device:
                if prior_cache[size] is None:
                    prior_cache[size] = {}
                if device not in prior_cache[size]:
                    prior_cache[size][device] = self.priors.to(device)
                self.priors = prior_cache[size][device]
        return self.priors


_global_config['max_num_detections'] = 4


class Detect(object):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations, as the predicted masks.
    """

    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.use_cross_class_nms = False
        self.use_fast_nms = False

    def __call__(self, predictions, net):
        """
        Args:
             loc_data: (tensor) Loc preds from loc layers
                Shape: [batch, num_priors, 4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch, num_priors, num_classes]
            mask_data: (tensor) Mask preds from mask layers
                Shape: [batch, num_priors, mask_dim]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [num_priors, 4]
            proto_data: (tensor) If using mask_type.lincomb, the prototype masks
                Shape: [batch, mask_h, mask_w, mask_dim]
        
        Returns:
            output of shape (batch_size, top_k, 1 + 1 + 4 + mask_dim)
            These outputs are in the order: class idx, confidence, bbox coords, and mask.

            Note that the outputs are sorted only if cross_class_nms is False
        """
        loc_data = predictions['loc']
        conf_data = predictions['conf']
        mask_data = predictions['mask']
        prior_data = predictions['priors']
        proto_data = predictions['proto'] if 'proto' in predictions else None
        inst_data = predictions['inst'] if 'inst' in predictions else None
        out = []
        with timer.env('Detect'):
            batch_size = loc_data.size(0)
            num_priors = prior_data.size(0)
            conf_preds = conf_data.view(batch_size, num_priors, self.
                num_classes).transpose(2, 1).contiguous()
            for batch_idx in range(batch_size):
                decoded_boxes = decode(loc_data[batch_idx], prior_data)
                result = self.detect(batch_idx, conf_preds, decoded_boxes,
                    mask_data, inst_data)
                if result is not None and proto_data is not None:
                    result['proto'] = proto_data[batch_idx]
                out.append({'detection': result, 'net': net})
        return out

    def detect(self, batch_idx, conf_preds, decoded_boxes, mask_data, inst_data
        ):
        """ Perform nms for only the max scoring class that isn't background (class 0) """
        cur_scores = conf_preds[(batch_idx), 1:, :]
        conf_scores, _ = torch.max(cur_scores, dim=0)
        keep = conf_scores > self.conf_thresh
        scores = cur_scores[:, (keep)]
        boxes = decoded_boxes[(keep), :]
        masks = mask_data[(batch_idx), (keep), :]
        if inst_data is not None:
            inst = inst_data[(batch_idx), (keep), :]
        if scores.size(1) == 0:
            return None
        if self.use_fast_nms:
            if self.use_cross_class_nms:
                boxes, masks, classes, scores = self.cc_fast_nms(boxes,
                    masks, scores, self.nms_thresh, self.top_k)
            else:
                boxes, masks, classes, scores = self.fast_nms(boxes, masks,
                    scores, self.nms_thresh, self.top_k)
        else:
            boxes, masks, classes, scores = self.traditional_nms(boxes,
                masks, scores, self.nms_thresh, self.conf_thresh)
            if self.use_cross_class_nms:
                print(
                    'Warning: Cross Class Traditional NMS is not implemented.')
        return {'box': boxes, 'mask': masks, 'class': classes, 'score': scores}

    def cc_fast_nms(self, boxes, masks, scores, iou_threshold: float=0.5,
        top_k: int=200):
        scores, classes = scores.max(dim=0)
        _, idx = scores.sort(0, descending=True)
        idx = idx[:top_k]
        boxes_idx = boxes[idx]
        iou = jaccard(boxes_idx, boxes_idx)
        iou.triu_(diagonal=1)
        iou_max, _ = torch.max(iou, dim=0)
        idx_out = idx[iou_max <= iou_threshold]
        return boxes[idx_out], masks[idx_out], classes[idx_out], scores[idx_out
            ]

    def fast_nms(self, boxes, masks, scores, iou_threshold: float=0.5,
        top_k: int=200, second_threshold: bool=False):
        scores, idx = scores.sort(1, descending=True)
        idx = idx[:, :top_k].contiguous()
        scores = scores[:, :top_k]
        num_classes, num_dets = idx.size()
        boxes = boxes[(idx.view(-1)), :].view(num_classes, num_dets, 4)
        masks = masks[(idx.view(-1)), :].view(num_classes, num_dets, -1)
        iou = jaccard(boxes, boxes)
        iou.triu_(diagonal=1)
        iou_max, _ = iou.max(dim=1)
        keep = iou_max <= iou_threshold
        if second_threshold:
            keep *= scores > self.conf_thresh
        classes = torch.arange(num_classes, device=boxes.device)[:, (None)
            ].expand_as(keep)
        classes = classes[keep]
        boxes = boxes[keep]
        masks = masks[keep]
        scores = scores[keep]
        scores, idx = scores.sort(0, descending=True)
        idx = idx[:cfg.max_num_detections]
        scores = scores[:cfg.max_num_detections]
        classes = classes[idx]
        boxes = boxes[idx]
        masks = masks[idx]
        return boxes, masks, classes, scores

    def traditional_nms(self, boxes, masks, scores, iou_threshold=0.5,
        conf_thresh=0.05):
        import pyximport
        pyximport.install(setup_args={'include_dirs': np.get_include()},
            reload_support=True)
        from utils.cython_nms import nms as cnms
        num_classes = scores.size(0)
        idx_lst = []
        cls_lst = []
        scr_lst = []
        boxes = boxes * cfg.max_size
        for _cls in range(num_classes):
            cls_scores = scores[(_cls), :]
            conf_mask = cls_scores > conf_thresh
            idx = torch.arange(cls_scores.size(0), device=boxes.device)
            cls_scores = cls_scores[conf_mask]
            idx = idx[conf_mask]
            if cls_scores.size(0) == 0:
                continue
            preds = torch.cat([boxes[conf_mask], cls_scores[:, (None)]], dim=1
                ).cpu().numpy()
            keep = cnms(preds, iou_threshold)
            keep = torch.Tensor(keep, device=boxes.device).long()
            idx_lst.append(idx[keep])
            cls_lst.append(keep * 0 + _cls)
            scr_lst.append(cls_scores[keep])
        idx = torch.cat(idx_lst, dim=0)
        classes = torch.cat(cls_lst, dim=0)
        scores = torch.cat(scr_lst, dim=0)
        scores, idx2 = scores.sort(0, descending=True)
        idx2 = idx2[:cfg.max_num_detections]
        scores = scores[:cfg.max_num_detections]
        idx = idx[idx2]
        classes = classes[idx2]
        return boxes[idx] / cfg.max_size, masks[idx], classes, scores


use_jit = torch.cuda.device_count() <= 1


ScriptModuleWrapper = torch.jit.ScriptModule if use_jit else nn.Module


script_method_wrapper = (torch.jit.script_method if use_jit else lambda fn,
    _rcn=None: fn)


_global_config['fpn'] = 4


class FPN(ScriptModuleWrapper):
    """
    Implements a general version of the FPN introduced in
    https://arxiv.org/pdf/1612.03144.pdf

    Parameters (in cfg.fpn):
        - num_features (int): The number of output features in the fpn layers.
        - interpolation_mode (str): The mode to pass to F.interpolate.
        - num_downsample (int): The number of downsampled layers to add onto the selected layers.
                                These extra layers are downsampled from the last selected layer.

    Args:
        - in_channels (list): For each conv layer you supply in the forward pass,
                              how many features will it have?
    """
    __constants__ = ['interpolation_mode', 'num_downsample',
        'use_conv_downsample', 'relu_pred_layers', 'lat_layers',
        'pred_layers', 'downsample_layers', 'relu_downsample_layers']

    def __init__(self, in_channels):
        super().__init__()
        self.lat_layers = nn.ModuleList([nn.Conv2d(x, cfg.fpn.num_features,
            kernel_size=1) for x in reversed(in_channels)])
        padding = 1 if cfg.fpn.pad else 0
        self.pred_layers = nn.ModuleList([nn.Conv2d(cfg.fpn.num_features,
            cfg.fpn.num_features, kernel_size=3, padding=padding) for _ in
            in_channels])
        if cfg.fpn.use_conv_downsample:
            self.downsample_layers = nn.ModuleList([nn.Conv2d(cfg.fpn.
                num_features, cfg.fpn.num_features, kernel_size=3, padding=
                1, stride=2) for _ in range(cfg.fpn.num_downsample)])
        self.interpolation_mode = cfg.fpn.interpolation_mode
        self.num_downsample = cfg.fpn.num_downsample
        self.use_conv_downsample = cfg.fpn.use_conv_downsample
        self.relu_downsample_layers = cfg.fpn.relu_downsample_layers
        self.relu_pred_layers = cfg.fpn.relu_pred_layers

    @script_method_wrapper
    def forward(self, convouts: List[torch.Tensor]):
        """
        Args:
            - convouts (list): A list of convouts for the corresponding layers in in_channels.
        Returns:
            - A list of FPN convouts in the same order as x with extra downsample layers if requested.
        """
        out = []
        x = torch.zeros(1, device=convouts[0].device)
        for i in range(len(convouts)):
            out.append(x)
        j = len(convouts)
        for lat_layer in self.lat_layers:
            j -= 1
            if j < len(convouts) - 1:
                _, _, h, w = convouts[j].size()
                x = F.interpolate(x, size=(h, w), mode=self.
                    interpolation_mode, align_corners=False)
            x = x + lat_layer(convouts[j])
            out[j] = x
        j = len(convouts)
        for pred_layer in self.pred_layers:
            j -= 1
            out[j] = pred_layer(out[j])
            if self.relu_pred_layers:
                F.relu(out[j], inplace=True)
        cur_idx = len(out)
        if self.use_conv_downsample:
            for downsample_layer in self.downsample_layers:
                out.append(downsample_layer(out[-1]))
        else:
            for idx in range(self.num_downsample):
                out.append(nn.functional.max_pool2d(out[-1], 1, stride=2))
        if self.relu_downsample_layers:
            for idx in range(len(out) - cur_idx):
                out[idx] = F.relu(out[idx + cur_idx], inplace=False)
        return out


_global_config['maskiou_net'] = 4


class FastMaskIoUNet(ScriptModuleWrapper):

    def __init__(self):
        super().__init__()
        input_channels = 1
        last_layer = [(cfg.num_classes - 1, 1, {})]
        self.maskiou_net, _ = make_net(input_channels, cfg.maskiou_net +
            last_layer, include_last_relu=True)

    def forward(self, x):
        x = self.maskiou_net(x)
        maskiou_p = F.max_pool2d(x, kernel_size=x.size()[2:]).squeeze(-1
            ).squeeze(-1)
        return maskiou_p


def construct_backbone(cfg):
    """ Constructs a backbone given a backbone config object (see config.py). """
    backbone = cfg.type(*cfg.args)
    num_layers = max(cfg.selected_layers) + 1
    while len(backbone.layers) < num_layers:
        backbone.add_layer()
    return backbone


_global_config['mask_proto_net'] = 4


_global_config['freeze_bn'] = 4


_global_config['mask_proto_src'] = 4


_global_config['nms_thresh'] = 4


_global_config['mask_size'] = 4


_global_config['nms_top_k'] = 4


_global_config['use_objectness_score'] = 4


class Yolact(nn.Module):
    """


    ██╗   ██╗ ██████╗ ██╗      █████╗  ██████╗████████╗
    ╚██╗ ██╔╝██╔═══██╗██║     ██╔══██╗██╔════╝╚══██╔══╝
     ╚████╔╝ ██║   ██║██║     ███████║██║        ██║   
      ╚██╔╝  ██║   ██║██║     ██╔══██║██║        ██║   
       ██║   ╚██████╔╝███████╗██║  ██║╚██████╗   ██║   
       ╚═╝    ╚═════╝ ╚══════╝╚═╝  ╚═╝ ╚═════╝   ╚═╝ 


    You can set the arguments by changing them in the backbone config object in config.py.

    Parameters (in cfg.backbone):
        - selected_layers: The indices of the conv layers to use for prediction.
        - pred_scales:     A list with len(selected_layers) containing tuples of scales (see PredictionModule)
        - pred_aspect_ratios: A list of lists of aspect ratios with len(selected_layers) (see PredictionModule)
    """

    def __init__(self):
        super().__init__()
        self.backbone = construct_backbone(cfg.backbone)
        if cfg.freeze_bn:
            self.freeze_bn()
        if cfg.mask_type == mask_type.direct:
            cfg.mask_dim = cfg.mask_size ** 2
        elif cfg.mask_type == mask_type.lincomb:
            if cfg.mask_proto_use_grid:
                self.grid = torch.Tensor(np.load(cfg.mask_proto_grid_file))
                self.num_grids = self.grid.size(0)
            else:
                self.num_grids = 0
            self.proto_src = cfg.mask_proto_src
            if self.proto_src is None:
                in_channels = 3
            elif cfg.fpn is not None:
                in_channels = cfg.fpn.num_features
            else:
                in_channels = self.backbone.channels[self.proto_src]
            in_channels += self.num_grids
            self.proto_net, cfg.mask_dim = make_net(in_channels, cfg.
                mask_proto_net, include_last_relu=False)
            if cfg.mask_proto_bias:
                cfg.mask_dim += 1
        self.selected_layers = cfg.backbone.selected_layers
        src_channels = self.backbone.channels
        if cfg.use_maskiou:
            self.maskiou_net = FastMaskIoUNet()
        if cfg.fpn is not None:
            self.fpn = FPN([src_channels[i] for i in self.selected_layers])
            self.selected_layers = list(range(len(self.selected_layers) +
                cfg.fpn.num_downsample))
            src_channels = [cfg.fpn.num_features] * len(self.selected_layers)
        self.prediction_layers = nn.ModuleList()
        cfg.num_heads = len(self.selected_layers)
        for idx, layer_idx in enumerate(self.selected_layers):
            parent = None
            if cfg.share_prediction_module and idx > 0:
                parent = self.prediction_layers[0]
            pred = PredictionModule(src_channels[layer_idx], src_channels[
                layer_idx], aspect_ratios=cfg.backbone.pred_aspect_ratios[
                idx], scales=cfg.backbone.pred_scales[idx], parent=parent,
                index=idx)
            self.prediction_layers.append(pred)
        if cfg.use_class_existence_loss:
            self.class_existence_fc = nn.Linear(src_channels[-1], cfg.
                num_classes - 1)
        if cfg.use_semantic_segmentation_loss:
            self.semantic_seg_conv = nn.Conv2d(src_channels[0], cfg.
                num_classes - 1, kernel_size=1)
        self.detect = Detect(cfg.num_classes, bkg_label=0, top_k=cfg.
            nms_top_k, conf_thresh=cfg.nms_conf_thresh, nms_thresh=cfg.
            nms_thresh)

    def save_weights(self, path):
        """ Saves the model's weights using compression because the file sizes were getting too big. """
        torch.save(self.state_dict(), path)

    def load_weights(self, path):
        """ Loads weights from a compressed save file. """
        state_dict = torch.load(path)
        for key in list(state_dict.keys()):
            if key.startswith('backbone.layer') and not key.startswith(
                'backbone.layers'):
                del state_dict[key]
            if key.startswith('fpn.downsample_layers.'):
                if cfg.fpn is not None and int(key.split('.')[2]
                    ) >= cfg.fpn.num_downsample:
                    del state_dict[key]
        self.load_state_dict(state_dict)

    def init_weights(self, backbone_path):
        """ Initialize weights for training. """
        self.backbone.init_backbone(backbone_path)
        conv_constants = getattr(nn.Conv2d(1, 1, 1), '__constants__')

        def all_in(x, y):
            for _x in x:
                if _x not in y:
                    return False
            return True
        for name, module in self.named_modules():
            is_script_conv = False
            if 'Script' in type(module).__name__:
                if hasattr(module, 'original_name'):
                    is_script_conv = 'Conv' in module.original_name
                else:
                    is_script_conv = all_in(module.__dict__[
                        '_constants_set'], conv_constants) and all_in(
                        conv_constants, module.__dict__['_constants_set'])
            is_conv_layer = isinstance(module, nn.Conv2d) or is_script_conv
            if is_conv_layer and module not in self.backbone.backbone_modules:
                nn.init.xavier_uniform_(module.weight.data)
                if module.bias is not None:
                    if cfg.use_focal_loss and 'conf_layer' in name:
                        if not cfg.use_sigmoid_focal_loss:
                            module.bias.data[0] = np.log((1 - cfg.
                                focal_loss_init_pi) / cfg.focal_loss_init_pi)
                            module.bias.data[1:] = -np.log(module.bias.size
                                (0) - 1)
                        else:
                            module.bias.data[0] = -np.log(cfg.
                                focal_loss_init_pi / (1 - cfg.
                                focal_loss_init_pi))
                            module.bias.data[1:] = -np.log((1 - cfg.
                                focal_loss_init_pi) / cfg.focal_loss_init_pi)
                    else:
                        module.bias.data.zero_()

    def train(self, mode=True):
        super().train(mode)
        if cfg.freeze_bn:
            self.freeze_bn()

    def freeze_bn(self, enable=False):
        """ Adapted from https://discuss.pytorch.org/t/how-to-train-with-frozen-batchnorm/12106/8 """
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.train() if enable else module.eval()
                module.weight.requires_grad = enable
                module.bias.requires_grad = enable

    def forward(self, x):
        """ The input should be of size [batch_size, 3, img_h, img_w] """
        _, _, img_h, img_w = x.size()
        cfg._tmp_img_h = img_h
        cfg._tmp_img_w = img_w
        with timer.env('backbone'):
            outs = self.backbone(x)
        if cfg.fpn is not None:
            with timer.env('fpn'):
                outs = [outs[i] for i in cfg.backbone.selected_layers]
                outs = self.fpn(outs)
        proto_out = None
        if cfg.mask_type == mask_type.lincomb and cfg.eval_mask_branch:
            with timer.env('proto'):
                proto_x = x if self.proto_src is None else outs[self.proto_src]
                if self.num_grids > 0:
                    grids = self.grid.repeat(proto_x.size(0), 1, 1, 1)
                    proto_x = torch.cat([proto_x, grids], dim=1)
                proto_out = self.proto_net(proto_x)
                proto_out = cfg.mask_proto_prototype_activation(proto_out)
                if cfg.mask_proto_prototypes_as_features:
                    proto_downsampled = proto_out.clone()
                    if cfg.mask_proto_prototypes_as_features_no_grad:
                        proto_downsampled = proto_out.detach()
                proto_out = proto_out.permute(0, 2, 3, 1).contiguous()
                if cfg.mask_proto_bias:
                    bias_shape = [x for x in proto_out.size()]
                    bias_shape[-1] = 1
                    proto_out = torch.cat([proto_out, torch.ones(*
                        bias_shape)], -1)
        with timer.env('pred_heads'):
            pred_outs = {'loc': [], 'conf': [], 'mask': [], 'priors': []}
            if cfg.use_mask_scoring:
                pred_outs['score'] = []
            if cfg.use_instance_coeff:
                pred_outs['inst'] = []
            for idx, pred_layer in zip(self.selected_layers, self.
                prediction_layers):
                pred_x = outs[idx]
                if (cfg.mask_type == mask_type.lincomb and cfg.
                    mask_proto_prototypes_as_features):
                    proto_downsampled = F.interpolate(proto_downsampled,
                        size=outs[idx].size()[2:], mode='bilinear',
                        align_corners=False)
                    pred_x = torch.cat([pred_x, proto_downsampled], dim=1)
                if (cfg.share_prediction_module and pred_layer is not self.
                    prediction_layers[0]):
                    pred_layer.parent = [self.prediction_layers[0]]
                p = pred_layer(pred_x)
                for k, v in p.items():
                    pred_outs[k].append(v)
        for k, v in pred_outs.items():
            pred_outs[k] = torch.cat(v, -2)
        if proto_out is not None:
            pred_outs['proto'] = proto_out
        if self.training:
            if cfg.use_class_existence_loss:
                pred_outs['classes'] = self.class_existence_fc(outs[-1].
                    mean(dim=(2, 3)))
            if cfg.use_semantic_segmentation_loss:
                pred_outs['segm'] = self.semantic_seg_conv(outs[0])
            return pred_outs
        else:
            if cfg.use_mask_scoring:
                pred_outs['score'] = torch.sigmoid(pred_outs['score'])
            if cfg.use_focal_loss:
                if cfg.use_sigmoid_focal_loss:
                    pred_outs['conf'] = torch.sigmoid(pred_outs['conf'])
                    if cfg.use_mask_scoring:
                        pred_outs['conf'] *= pred_outs['score']
                elif cfg.use_objectness_score:
                    objectness = torch.sigmoid(pred_outs['conf'][:, :, (0)])
                    pred_outs['conf'][:, :, 1:] = objectness[:, :, (None)
                        ] * F.softmax(pred_outs['conf'][:, :, 1:], -1)
                    pred_outs['conf'][:, :, (0)] = 1 - objectness
                else:
                    pred_outs['conf'] = F.softmax(pred_outs['conf'], -1)
            elif cfg.use_objectness_score:
                objectness = torch.sigmoid(pred_outs['conf'][:, :, (0)])
                pred_outs['conf'][:, :, 1:] = (objectness > 0.1)[..., None
                    ] * F.softmax(pred_outs['conf'][:, :, 1:], dim=-1)
            else:
                pred_outs['conf'] = F.softmax(pred_outs['conf'], -1)
            return self.detect(pred_outs, self)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_dbolya_yolact(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(DarkNetBackbone(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

    @_fails_compile()
    def test_001(self):
        self._check(VGGBackbone(*[], **{'cfg': _mock_config()}), [torch.rand([4, 4, 4, 4])], {})

