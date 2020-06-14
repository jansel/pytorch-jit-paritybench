import sys
_module = sys.modules[__name__]
del sys
maskrcnn_benchmark = _module
config = _module
defaults = _module
paths_catalog = _module
data = _module
build = _module
collate_batch = _module
datasets = _module
coco = _module
concat_dataset = _module
evaluation = _module
coco_eval = _module
voc = _module
voc_eval = _module
list_dataset = _module
samplers = _module
distributed = _module
grouped_batch_sampler = _module
iteration_based_batch_sampler = _module
transforms = _module
engine = _module
inference = _module
trainer = _module
layers = _module
_utils = _module
batch_norm = _module
dcn_v2 = _module
iou_loss = _module
misc = _module
nms = _module
roi_align = _module
roi_pool = _module
scale = _module
sigmoid_focal_loss = _module
smooth_l1_loss = _module
modeling = _module
backbone = _module
fbnet = _module
fbnet_builder = _module
fbnet_modeldef = _module
fpn = _module
mobilenet = _module
resnet = _module
balanced_positive_negative_sampler = _module
box_coder = _module
detector = _module
detectors = _module
generalized_rcnn = _module
single_stage_detector = _module
make_layers = _module
matcher = _module
poolers = _module
registry = _module
roi_heads = _module
box_head = _module
box_head = _module
inference = _module
loss = _module
roi_box_feature_extractors = _module
roi_box_predictors = _module
mask_head = _module
inference = _module
loss = _module
mask_head = _module
roi_mask_feature_extractors = _module
roi_mask_predictors = _module
roi_heads = _module
rpn = _module
anchor_generator = _module
densebox = _module
densebox = _module
inference = _module
loss = _module
inference = _module
loss = _module
nas_head = _module
retinanet_nas_head = _module
retinanet = _module
inference = _module
loss = _module
retinanet = _module
rpn = _module
utils = _module
nas = _module
layer_factory = _module
micro_decoders = _module
micro_heads = _module
rl = _module
genotypes = _module
solver = _module
lr_scheduler = _module
structures = _module
bounding_box = _module
boxlist_ops = _module
image_list = _module
segmentation_mask = _module
c2_model_loading = _module
checkpoint = _module
collect_env = _module
comm = _module
cv2_util = _module
env = _module
imports = _module
logger = _module
metric_logger = _module
miscellaneous = _module
model_serialization = _module
model_zoo = _module
setup = _module
checkpoint = _module
test_data_samplers = _module
test_metric_logger = _module
convert_cityscapes_to_coco = _module
instances2dict_with_polygons = _module
test_net = _module
train_net = _module

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


import math


import torch


import torch.distributed as dist


from torch.utils.data.sampler import Sampler


from torch import nn


from torch.autograd import Function


from torch.nn.modules.utils import _pair


from torch.autograd.function import once_differentiable


from torch.nn.modules.utils import _ntuple


import copy


import logging


from collections import OrderedDict


import torch.nn as nn


import torch.nn.functional as F


from torch.nn import BatchNorm2d


from collections import namedtuple


from torch.nn import functional as F


import numpy as np


class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters
    are fixed
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer('weight', torch.ones(n))
        self.register_buffer('bias', torch.zeros(n))
        self.register_buffer('running_mean', torch.zeros(n))
        self.register_buffer('running_var', torch.ones(n))

    def forward(self, x):
        if x.dtype == torch.float16:
            self.weight = self.weight.half()
            self.bias = self.bias.half()
            self.running_mean = self.running_mean.half()
            self.running_var = self.running_var.half()
        scale = self.weight * self.running_var.rsqrt()
        bias = self.bias - self.running_mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return x * scale + bias


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
        output, output_count = _C.dcn_v2_psroi_pooling_forward(input, rois,
            offset, ctx.no_trans, ctx.spatial_scale, ctx.output_dim, ctx.
            group_size, ctx.pooled_size, ctx.part_size, ctx.sample_per_part,
            ctx.trans_std)
        ctx.save_for_backward(input, rois, offset, output_count)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, rois, offset, output_count = ctx.saved_tensors
        grad_input, grad_offset = _C.dcn_v2_psroi_pooling_backward(grad_output,
            input, rois, offset, output_count, ctx.no_trans, ctx.
            spatial_scale, ctx.output_dim, ctx.group_size, ctx.pooled_size,
            ctx.part_size, ctx.sample_per_part, ctx.trans_std)
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


class IOULoss(nn.Module):

    def forward(self, pred, target, weight=None):
        pred_left = pred[:, (0)]
        pred_top = pred[:, (1)]
        pred_right = pred[:, (2)]
        pred_bottom = pred[:, (3)]
        target_left = target[:, (0)]
        target_top = target[:, (1)]
        target_right = target[:, (2)]
        target_bottom = target[:, (3)]
        target_aera = (target_left + target_right) * (target_top +
            target_bottom)
        pred_aera = (pred_left + pred_right) * (pred_top + pred_bottom)
        w_intersect = torch.min(pred_left, target_left) + torch.min(pred_right,
            target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(
            pred_top, target_top)
        area_intersect = w_intersect * h_intersect
        area_union = target_aera + pred_aera - area_intersect
        losses = -torch.log((area_intersect + 1.0) / (area_union + 1.0))
        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum() / weight.sum()
        else:
            assert losses.numel() != 0
            return losses.mean()


class _NewEmptyTensorOp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, new_shape):
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, grad):
        shape = ctx.shape
        return _NewEmptyTensorOp.apply(grad, shape), None


class BatchNorm2d(torch.nn.BatchNorm2d):

    def forward(self, x):
        if x.numel() > 0:
            return super(BatchNorm2d, self).forward(x)
        output_shape = x.shape
        return _NewEmptyTensorOp.apply(x, output_shape)


class Conv2d(torch.nn.Conv2d):

    def forward(self, x):
        if x.numel() > 0:
            return super(Conv2d, self).forward(x)
        output_shape = [((i + 2 * p - (di * (k - 1) + 1)) // d + 1) for i,
            p, di, k, d in zip(x.shape[-2:], self.padding, self.dilation,
            self.kernel_size, self.stride)]
        output_shape = [x.shape[0], self.weight.shape[0]] + output_shape
        return _NewEmptyTensorOp.apply(x, output_shape)


class ConvTranspose2d(torch.nn.ConvTranspose2d):

    def forward(self, x):
        if x.numel() > 0:
            return super(ConvTranspose2d, self).forward(x)
        output_shape = [((i - 1) * d - 2 * p + (di * (k - 1) + 1) + op) for
            i, p, di, k, d, op in zip(x.shape[-2:], self.padding, self.
            dilation, self.kernel_size, self.stride, self.output_padding)]
        output_shape = [x.shape[0], self.bias.shape[0]] + output_shape
        return _NewEmptyTensorOp.apply(x, output_shape)


class _ROIAlign(Function):

    @staticmethod
    def forward(ctx, input, roi, output_size, spatial_scale, sampling_ratio):
        ctx.save_for_backward(roi)
        ctx.output_size = _pair(output_size)
        ctx.spatial_scale = spatial_scale
        ctx.sampling_ratio = sampling_ratio
        ctx.input_shape = input.size()
        output = _C.roi_align_forward(input, roi, spatial_scale,
            output_size[0], output_size[1], sampling_ratio)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        rois, = ctx.saved_tensors
        output_size = ctx.output_size
        spatial_scale = ctx.spatial_scale
        sampling_ratio = ctx.sampling_ratio
        bs, ch, h, w = ctx.input_shape
        grad_input = _C.roi_align_backward(grad_output, rois, spatial_scale,
            output_size[0], output_size[1], bs, ch, h, w, sampling_ratio)
        return grad_input, None, None, None, None


roi_align = _ROIAlign.apply


class ROIAlign(nn.Module):

    def __init__(self, output_size, spatial_scale, sampling_ratio):
        super(ROIAlign, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio

    def forward(self, input, rois):
        return roi_align(input, rois, self.output_size, self.spatial_scale,
            self.sampling_ratio)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '('
        tmpstr += 'output_size=' + str(self.output_size)
        tmpstr += ', spatial_scale=' + str(self.spatial_scale)
        tmpstr += ', sampling_ratio=' + str(self.sampling_ratio)
        tmpstr += ')'
        return tmpstr


class _ROIPool(Function):

    @staticmethod
    def forward(ctx, input, roi, output_size, spatial_scale):
        ctx.output_size = _pair(output_size)
        ctx.spatial_scale = spatial_scale
        ctx.input_shape = input.size()
        output, argmax = _C.roi_pool_forward(input, roi, spatial_scale,
            output_size[0], output_size[1])
        ctx.save_for_backward(input, roi, argmax)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, rois, argmax = ctx.saved_tensors
        output_size = ctx.output_size
        spatial_scale = ctx.spatial_scale
        bs, ch, h, w = ctx.input_shape
        grad_input = _C.roi_pool_backward(grad_output, input, rois, argmax,
            spatial_scale, output_size[0], output_size[1], bs, ch, h, w)
        return grad_input, None, None, None


roi_pool = _ROIPool.apply


class ROIPool(nn.Module):

    def __init__(self, output_size, spatial_scale):
        super(ROIPool, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale

    def forward(self, input, rois):
        return roi_pool(input, rois, self.output_size, self.spatial_scale)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '('
        tmpstr += 'output_size=' + str(self.output_size)
        tmpstr += ', spatial_scale=' + str(self.spatial_scale)
        tmpstr += ')'
        return tmpstr


class Scale(nn.Module):

    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


class _SigmoidFocalLoss(Function):

    @staticmethod
    def forward(ctx, logits, targets, gamma, alpha):
        ctx.save_for_backward(logits, targets)
        num_classes = logits.shape[1]
        ctx.num_classes = num_classes
        ctx.gamma = gamma
        ctx.alpha = alpha
        losses = _C.sigmoid_focalloss_forward(logits, targets, num_classes,
            gamma, alpha)
        return losses

    @staticmethod
    @once_differentiable
    def backward(ctx, d_loss):
        logits, targets = ctx.saved_tensors
        num_classes = ctx.num_classes
        gamma = ctx.gamma
        alpha = ctx.alpha
        d_loss = d_loss.contiguous()
        d_logits = _C.sigmoid_focalloss_backward(logits, targets, d_loss,
            num_classes, gamma, alpha)
        return d_logits, None, None, None, None


sigmoid_focal_loss_cuda = _SigmoidFocalLoss.apply


def sigmoid_focal_loss_cpu(logits, targets, gamma, alpha):
    num_classes = logits.shape[1]
    gamma = gamma[0]
    alpha = alpha[0]
    dtype = targets.dtype
    device = targets.device
    class_range = torch.arange(1, num_classes + 1, dtype=dtype, device=device
        ).unsqueeze(0)
    t = targets.unsqueeze(1)
    p = torch.sigmoid(logits)
    term1 = (1 - p) ** gamma * torch.log(p)
    term2 = p ** gamma * torch.log(1 - p)
    return -(t == class_range).float() * term1 * alpha - ((t != class_range
        ) * (t >= 0)).float() * term2 * (1 - alpha)


class SigmoidFocalLoss(nn.Module):

    def __init__(self, gamma, alpha):
        super(SigmoidFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        device = logits.device
        if logits.is_cuda:
            loss_func = sigmoid_focal_loss_cuda
        else:
            loss_func = sigmoid_focal_loss_cpu
        loss = loss_func(logits, targets, self.gamma, self.alpha)
        return loss.sum()

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '('
        tmpstr += 'gamma=' + str(self.gamma)
        tmpstr += ', alpha=' + str(self.alpha)
        tmpstr += ')'
        return tmpstr


def _get_trunk_cfg(arch_def):
    """ Get all stages except the last one """
    num_stages = mbuilder.get_num_stages(arch_def)
    trunk_stages = arch_def.get('backbone', range(num_stages - 1))
    ret = mbuilder.get_blocks(arch_def, stage_indices=trunk_stages)
    return ret


class FBNetTrunk(nn.Module):

    def __init__(self, builder, arch_def, dim_in):
        super(FBNetTrunk, self).__init__()
        self.first = builder.add_first(arch_def['first'], dim_in=dim_in)
        trunk_cfg = _get_trunk_cfg(arch_def)
        self.stages = builder.add_blocks(trunk_cfg['stages'])

    def forward(self, x):
        y = self.first(x)
        ret = []
        for index, stages_sub_children in enumerate(self.stages):
            y = stages_sub_children(y)
            if index in (2, 7, 19, 24):
                ret.append(y)
        return ret


logger = logging.getLogger(__name__)


def _get_rpn_stage(arch_def, num_blocks):
    rpn_stage = arch_def.get('rpn')
    ret = mbuilder.get_blocks(arch_def, stage_indices=rpn_stage)
    if num_blocks > 0:
        logger.warn('Use last {} blocks in {} as rpn'.format(num_blocks, ret))
        block_count = len(ret['stages'])
        assert num_blocks <= block_count, 'use block {}, block count {}'.format(
            num_blocks, block_count)
        blocks = range(block_count - num_blocks, block_count)
        ret = mbuilder.get_blocks(ret, block_indices=blocks)
    return ret['stages']


class FBNetRPNHead(nn.Module):

    def __init__(self, cfg, in_channels, builder, arch_def):
        super(FBNetRPNHead, self).__init__()
        assert in_channels == builder.last_depth
        rpn_bn_type = cfg.MODEL.FBNET.RPN_BN_TYPE
        if len(rpn_bn_type) > 0:
            builder.bn_type = rpn_bn_type
        use_blocks = cfg.MODEL.FBNET.RPN_HEAD_BLOCKS
        stages = _get_rpn_stage(arch_def, use_blocks)
        self.head = builder.add_blocks(stages)
        self.out_channels = builder.last_depth

    def forward(self, x):
        x = [self.head(y) for y in x]
        return x


def _get_head_stage(arch, head_name, blocks):
    if head_name not in arch:
        head_name = 'head'
    head_stage = arch.get(head_name)
    ret = mbuilder.get_blocks(arch, stage_indices=head_stage, block_indices
        =blocks)
    return ret['stages']


ARCH_CFG_NAME_MAPPING = {'bbox': 'ROI_BOX_HEAD', 'kpts':
    'ROI_KEYPOINT_HEAD', 'mask': 'ROI_MASK_HEAD'}


class FBNetROIHead(nn.Module):

    def __init__(self, cfg, in_channels, builder, arch_def, head_name,
        use_blocks, stride_init, last_layer_scale):
        super(FBNetROIHead, self).__init__()
        assert in_channels == builder.last_depth
        assert isinstance(use_blocks, list)
        head_cfg_name = ARCH_CFG_NAME_MAPPING[head_name]
        self.pooler = poolers.make_pooler(cfg, head_cfg_name)
        stage = _get_head_stage(arch_def, head_name, use_blocks)
        assert stride_init in [0, 1, 2]
        if stride_init != 0:
            stage[0]['block'][3] = stride_init
        blocks = builder.add_blocks(stage)
        last_info = copy.deepcopy(arch_def['last'])
        last_info[1] = last_layer_scale
        last = builder.add_last(last_info)
        self.head = nn.Sequential(OrderedDict([('blocks', blocks), ('last',
            last)]))
        self.out_channels = builder.last_depth

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = self.head(x)
        return x


class Identity(nn.Module):

    def __init__(self, C_in, C_out, stride):
        super(Identity, self).__init__()
        self.conv = ConvBNRelu(C_in, C_out, kernel=1, stride=stride, pad=0,
            no_bias=1, use_relu='relu', bn_type='bn'
            ) if C_in != C_out or stride != 1 else None

    def forward(self, x):
        if self.conv:
            out = self.conv(x)
        else:
            out = x
        return out


class CascadeConv3x3(nn.Sequential):

    def __init__(self, C_in, C_out, stride):
        assert stride in [1, 2]
        ops = [Conv2d(C_in, C_in, 3, stride, 1, bias=False), BatchNorm2d(
            C_in), nn.ReLU(inplace=True), Conv2d(C_in, C_out, 3, 1, 1, bias
            =False), BatchNorm2d(C_out)]
        super(CascadeConv3x3, self).__init__(*ops)
        self.res_connect = stride == 1 and C_in == C_out

    def forward(self, x):
        y = super(CascadeConv3x3, self).forward(x)
        if self.res_connect:
            y += x
        return y


class Shift(nn.Module):

    def __init__(self, C, kernel_size, stride, padding):
        super(Shift, self).__init__()
        self.C = C
        kernel = torch.zeros((C, 1, kernel_size, kernel_size), dtype=torch.
            float32)
        ch_idx = 0
        assert stride in [1, 2]
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.dilation = 1
        hks = kernel_size // 2
        ksq = kernel_size ** 2
        for i in range(kernel_size):
            for j in range(kernel_size):
                if i == hks and j == hks:
                    num_ch = C // ksq + C % ksq
                else:
                    num_ch = C // ksq
                kernel[ch_idx:ch_idx + num_ch, (0), (i), (j)] = 1
                ch_idx += num_ch
        self.register_parameter('bias', None)
        self.kernel = nn.Parameter(kernel, requires_grad=False)

    def forward(self, x):
        if x.numel() > 0:
            return nn.functional.conv2d(x, self.kernel, self.bias, (self.
                stride, self.stride), (self.padding, self.padding), self.
                dilation, self.C)
        output_shape = [((i + 2 * p - (di * (k - 1) + 1)) // d + 1) for i,
            p, di, k, d in zip(x.shape[-2:], (self.padding, self.dilation),
            (self.dilation, self.dilation), (self.kernel_size, self.
            kernel_size), (self.stride, self.stride))]
        output_shape = [x.shape[0], self.C] + output_shape
        return _NewEmptyTensorOp.apply(x, output_shape)


def _py2_round(x):
    return math.floor(x + 0.5) if x >= 0.0 else math.ceil(x - 0.5)


def _get_divisible_by(num, divisible_by, min_val):
    ret = int(num)
    if divisible_by > 0 and num % divisible_by != 0:
        ret = int((_py2_round(num / divisible_by) or min_val) * divisible_by)
    return ret


class ShiftBlock5x5(nn.Sequential):

    def __init__(self, C_in, C_out, expansion, stride):
        assert stride in [1, 2]
        self.res_connect = stride == 1 and C_in == C_out
        C_mid = _get_divisible_by(C_in * expansion, 8, 8)
        ops = [Conv2d(C_in, C_mid, 1, 1, 0, bias=False), BatchNorm2d(C_mid),
            nn.ReLU(inplace=True), Shift(C_mid, 5, stride, 2), Conv2d(C_mid,
            C_out, 1, 1, 0, bias=False), BatchNorm2d(C_out)]
        super(ShiftBlock5x5, self).__init__(*ops)

    def forward(self, x):
        y = super(ShiftBlock5x5, self).forward(x)
        if self.res_connect:
            y += x
        return y


class ChannelShuffle(nn.Module):

    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        """Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]"""
        N, C, H, W = x.size()
        g = self.groups
        assert C % g == 0, 'Incompatible group size {} for input channel {}'.format(
            g, C)
        return x.view(N, g, int(C / g), H, W).permute(0, 2, 1, 3, 4
            ).contiguous().view(N, C, H, W)


class ConvBNRelu(nn.Sequential):

    def __init__(self, input_depth, output_depth, kernel, stride, pad,
        no_bias, use_relu, bn_type, group=1, *args, **kwargs):
        super(ConvBNRelu, self).__init__()
        assert use_relu in ['relu', None]
        if isinstance(bn_type, (list, tuple)):
            assert len(bn_type) == 2
            assert bn_type[0] == 'gn'
            gn_group = bn_type[1]
            bn_type = bn_type[0]
        assert bn_type in ['bn', 'af', 'gn', None]
        assert stride in [1, 2, 4]
        op = Conv2d(input_depth, output_depth, *args, kernel_size=kernel,
            stride=stride, padding=pad, bias=not no_bias, groups=group, **
            kwargs)
        nn.init.kaiming_normal_(op.weight, mode='fan_out', nonlinearity='relu')
        if op.bias is not None:
            nn.init.constant_(op.bias, 0.0)
        self.add_module('conv', op)
        if bn_type == 'bn':
            bn_op = BatchNorm2d(output_depth)
        elif bn_type == 'gn':
            bn_op = nn.GroupNorm(num_groups=gn_group, num_channels=output_depth
                )
        elif bn_type == 'af':
            bn_op = FrozenBatchNorm2d(output_depth)
        if bn_type is not None:
            self.add_module('bn', bn_op)
        if use_relu == 'relu':
            self.add_module('relu', nn.ReLU(inplace=True))


class SEModule(nn.Module):
    reduction = 4

    def __init__(self, C):
        super(SEModule, self).__init__()
        mid = max(C // self.reduction, 8)
        conv1 = Conv2d(C, mid, 1, 1, 0)
        conv2 = Conv2d(mid, C, 1, 1, 0)
        self.op = nn.Sequential(nn.AdaptiveAvgPool2d(1), conv1, nn.ReLU(
            inplace=True), conv2, nn.Sigmoid())

    def forward(self, x):
        return x * self.op(x)


def interpolate(input, size=None, scale_factor=None, mode='nearest',
    align_corners=None):
    if input.numel() > 0:
        return torch.nn.functional.interpolate(input, size, scale_factor,
            mode, align_corners)

    def _check_size_scale_factor(dim):
        if size is None and scale_factor is None:
            raise ValueError('either size or scale_factor should be defined')
        if size is not None and scale_factor is not None:
            raise ValueError(
                'only one of size or scale_factor should be defined')
        if scale_factor is not None and isinstance(scale_factor, tuple
            ) and len(scale_factor) != dim:
            raise ValueError(
                'scale_factor shape must match input shape. Input is {}D, scale_factor size is {}'
                .format(dim, len(scale_factor)))

    def _output_size(dim):
        _check_size_scale_factor(dim)
        if size is not None:
            return size
        scale_factors = _ntuple(dim)(scale_factor)
        return [int(math.floor(input.size(i + 2) * scale_factors[i])) for i in
            range(dim)]
    output_shape = tuple(_output_size(2))
    output_shape = input.shape[:-2] + output_shape
    return _NewEmptyTensorOp.apply(input, output_shape)


class Upsample(nn.Module):

    def __init__(self, scale_factor, mode, align_corners=None):
        super(Upsample, self).__init__()
        self.scale = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return interpolate(x, scale_factor=self.scale, mode=self.mode,
            align_corners=self.align_corners)


def _get_upsample_op(stride):
    assert stride in [1, 2, 4] or stride in [-1, -2, -4] or isinstance(stride,
        tuple) and all(x in [-1, -2, -4] for x in stride)
    scales = stride
    ret = None
    if isinstance(stride, tuple) or stride < 0:
        scales = [(-x) for x in stride] if isinstance(stride, tuple
            ) else -stride
        stride = 1
        ret = Upsample(scale_factor=scales, mode='nearest', align_corners=None)
    return ret, stride


class IRFBlock(nn.Module):

    def __init__(self, input_depth, output_depth, expansion, stride,
        bn_type='bn', kernel=3, width_divisor=1, shuffle_type=None,
        pw_group=1, se=False, cdw=False, dw_skip_bn=False, dw_skip_relu=False):
        super(IRFBlock, self).__init__()
        assert kernel in [1, 3, 5, 7], kernel
        self.use_res_connect = stride == 1 and input_depth == output_depth
        self.output_depth = output_depth
        mid_depth = int(input_depth * expansion)
        mid_depth = _get_divisible_by(mid_depth, width_divisor, width_divisor)
        self.pw = ConvBNRelu(input_depth, mid_depth, kernel=1, stride=1,
            pad=0, no_bias=1, use_relu='relu', bn_type=bn_type, group=pw_group)
        self.upscale, stride = _get_upsample_op(stride)
        if kernel == 1:
            self.dw = nn.Sequential()
        elif cdw:
            dw1 = ConvBNRelu(mid_depth, mid_depth, kernel=kernel, stride=
                stride, pad=kernel // 2, group=mid_depth, no_bias=1,
                use_relu='relu', bn_type=bn_type)
            dw2 = ConvBNRelu(mid_depth, mid_depth, kernel=kernel, stride=1,
                pad=kernel // 2, group=mid_depth, no_bias=1, use_relu=
                'relu' if not dw_skip_relu else None, bn_type=bn_type if 
                not dw_skip_bn else None)
            self.dw = nn.Sequential(OrderedDict([('dw1', dw1), ('dw2', dw2)]))
        else:
            self.dw = ConvBNRelu(mid_depth, mid_depth, kernel=kernel,
                stride=stride, pad=kernel // 2, group=mid_depth, no_bias=1,
                use_relu='relu' if not dw_skip_relu else None, bn_type=
                bn_type if not dw_skip_bn else None)
        self.pwl = ConvBNRelu(mid_depth, output_depth, kernel=1, stride=1,
            pad=0, no_bias=1, use_relu=None, bn_type=bn_type, group=pw_group)
        self.shuffle_type = shuffle_type
        if shuffle_type is not None:
            self.shuffle = ChannelShuffle(pw_group)
        self.se4 = SEModule(output_depth) if se else nn.Sequential()
        self.output_depth = output_depth

    def forward(self, x):
        y = self.pw(x)
        if self.shuffle_type == 'mid':
            y = self.shuffle(y)
        if self.upscale is not None:
            y = self.upscale(y)
        y = self.dw(y)
        y = self.pwl(y)
        if self.use_res_connect:
            y += x
        y = self.se4(y)
        return y


class FPN(nn.Module):
    """
    Module that adds FPN on top of a list of feature maps.
    The feature maps are currently supposed to be in increasing depth
    order, and must be consecutive
    """

    def __init__(self, in_channels_list, out_channels, conv_block,
        top_blocks=None):
        """
        Arguments:
            in_channels_list (list[int]): number of channels for each feature map that
                will be fed
            out_channels (int): number of channels of the FPN representation
            top_blocks (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                FPN output, and the result will extend the result list
        """
        super(FPN, self).__init__()
        self.inner_blocks = []
        self.layer_blocks = []
        for idx, in_channels in enumerate(in_channels_list, 1):
            inner_block = 'fpn_inner{}'.format(idx)
            layer_block = 'fpn_layer{}'.format(idx)
            if in_channels == 0:
                continue
            inner_block_module = conv_block(in_channels, out_channels, 1)
            layer_block_module = conv_block(out_channels, out_channels, 3, 1)
            self.add_module(inner_block, inner_block_module)
            self.add_module(layer_block, layer_block_module)
            self.inner_blocks.append(inner_block)
            self.layer_blocks.append(layer_block)
        self.top_blocks = top_blocks

    def forward(self, x):
        """
        Arguments:
            x (list[Tensor]): feature maps for each feature level.
        Returns:
            results (tuple[Tensor]): feature maps after FPN layers.
                They are ordered from highest resolution first.
        """
        last_inner = getattr(self, self.inner_blocks[-1])(x[-1])
        results = []
        results.append(getattr(self, self.layer_blocks[-1])(last_inner))
        for feature, inner_block, layer_block in zip(x[:-1][::-1], self.
            inner_blocks[:-1][::-1], self.layer_blocks[:-1][::-1]):
            if not inner_block:
                continue
            inner_top_down = F.interpolate(last_inner, scale_factor=2, mode
                ='nearest')
            inner_lateral = getattr(self, inner_block)(feature)
            last_inner = inner_lateral + inner_top_down
            results.insert(0, getattr(self, layer_block)(last_inner))
        if isinstance(self.top_blocks, LastLevelP6P7):
            last_results = self.top_blocks(x[-1], results[-1])
            results.extend(last_results)
        elif isinstance(self.top_blocks, LastLevelMaxPool):
            last_results = self.top_blocks(results[-1])
            results.extend(last_results)
        return tuple(results)


class ASPP(nn.Module):

    def __init__(self, inplanes, planes, rates):
        super(ASPP, self).__init__()
        self.atrous_convs = nn.ModuleList()
        for rate in rates:
            self.atrous_convs.append(nn.Sequential(nn.Conv2d(inplanes,
                inplanes, kernel_size=3, stride=1, padding=rate, groups=
                inplanes, dilation=rate, bias=False), nn.BatchNorm2d(
                inplanes), nn.ReLU(inplace=True)))
        self._init_weight()

    def forward(self, x):
        outs = []
        temp = 0
        for atrous_conv in self.atrous_convs:
            temp = temp + atrous_conv(x)
            outs.append(temp)
        return torch.cat(outs, dim=1)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class InnerBlock(nn.Module):
    """This block is applied before merging."""

    def __init__(self, in_c, out_c, rates):
        super(InnerBlock, self).__init__()
        self.pre_aspp = nn.Sequential(nn.Conv2d(in_c, out_c, kernel_size=3,
            stride=1, padding=1, bias=False), nn.BatchNorm2d(out_c), nn.
            ReLU(inplace=True))
        self.aspp = ASPP(out_c, out_c, rates)
        self.post_aspp = nn.Sequential(nn.Conv2d(out_c * len(rates), out_c,
            kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d
            (out_c), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.pre_aspp(x)
        aspp = self.aspp(x)
        return x + self.post_aspp(aspp)


class DFPN(nn.Module):
    """
    Dilated FPN
    Module that adds FPN on top of a list of feature maps.
    The feature maps are currently supposed to be in increasing depth
    order, and must be consecutive
    """

    def __init__(self, in_channels_list, out_channels, dilations_list=None,
        top_blocks=None):
        """
        Arguments:
            in_channels_list (list[int]): number of channels for each feature map that
                will be fed
            out_channels (int): number of channels of the FPN representation
            top_blocks (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                FPN output, and the result will extend the result list
        """
        super(DFPN, self).__init__()
        self.inner_blocks = []
        self.layer_blocks = []
        for idx, (in_channels, rates) in enumerate(zip(in_channels_list,
            dilations_list), 1):
            inner_block = 'dfpn_inner{}'.format(idx)
            layer_block = 'dfpn_layer{}'.format(idx)
            if in_channels == 0:
                continue
            inner_block_module = InnerBlock(in_channels, out_channels, rates)
            layer_block_module = InnerBlock(out_channels, out_channels, rates)
            self.add_module(inner_block, inner_block_module)
            self.add_module(layer_block, layer_block_module)
            self.inner_blocks.append(inner_block)
            self.layer_blocks.append(layer_block)
        self.top_blocks = top_blocks

    def forward(self, x):
        """
        Arguments:
            x (list[Tensor]): feature maps for each feature level.
        Returns:
            results (tuple[Tensor]): feature maps after FPN layers.
                They are ordered from highest resolution first.
        """
        last_inner = getattr(self, self.inner_blocks[-1])(x[-1])
        results = []
        results.append(getattr(self, self.layer_blocks[-1])(last_inner))
        for feature, inner_block, layer_block in zip(x[:-1][::-1], self.
            inner_blocks[:-1][::-1], self.layer_blocks[:-1][::-1]):
            if len(inner_block):
                inner_lateral = getattr(self, inner_block)(feature)
                inner_top_down = F.interpolate(last_inner, size=
                    inner_lateral.shape[-2:], mode='bilinear',
                    align_corners=False)
                last_inner = inner_lateral + inner_top_down
                results.insert(0, getattr(self, layer_block)(last_inner))
        if self.top_blocks is not None:
            last_results = self.top_blocks(results[-1])
            results.extend(last_results)
        return tuple(results)


class LastLevelMaxPool(nn.Module):

    def forward(self, x):
        return [F.max_pool2d(x, 1, 2, 0)]


class LastLevelP6P7(nn.Module):
    """
    This module is used in RetinaNet to generate extra layers, P6 and P7.
    """

    def __init__(self, in_channels, out_channels):
        super(LastLevelP6P7, self).__init__()
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        for module in [self.p6, self.p7]:
            nn.init.kaiming_uniform_(module.weight, a=1)
            nn.init.constant_(module.bias, 0)
        self.use_P5 = in_channels == out_channels

    def forward(self, c5, p5):
        x = p5 if self.use_P5 else c5
        p6 = self.p6(x)
        p7 = self.p7(F.relu(p6))
        return [p6, p7]


class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup
        if expand_ratio == 1:
            self.conv = nn.Sequential(Conv2d(hidden_dim, hidden_dim, 3,
                stride, 1, groups=hidden_dim, bias=False), BatchNorm2d(
                hidden_dim), nn.ReLU6(inplace=True), Conv2d(hidden_dim, oup,
                1, 1, 0, bias=False), BatchNorm2d(oup))
        else:
            self.conv = nn.Sequential(Conv2d(inp, hidden_dim, 1, 1, 0, bias
                =False), BatchNorm2d(hidden_dim), nn.ReLU6(inplace=True),
                Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=
                hidden_dim, bias=False), BatchNorm2d(hidden_dim), nn.ReLU6(
                inplace=True), Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                BatchNorm2d(oup))

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


def conv_bn(C_in, C_out, kernel_size, stride, padding, affine=True):
    return nn.Sequential(nn.Conv2d(C_in, C_out, kernel_size, stride=stride,
        padding=padding, bias=False), nn.BatchNorm2d(C_out, affine=affine))


class MobileNetV2(nn.Module):
    """
    Should freeze bn
    """

    def __init__(self, cfg, n_class=1000, input_size=224, width_mult=1.0):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        interverted_residual_setting = [[1, 16, 1, 1], [6, 24, 2, 2], [6, 
            32, 3, 2], [6, 64, 4, 2], [6, 96, 3, 1], [6, 160, 3, 2], [6, 
            320, 1, 1]]
        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.features = nn.ModuleList([conv_bn(3, input_channel, 2)])
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel,
                        output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel,
                        output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        self._initialize_weights()
        self._freeze_backbone(cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT)

    def _freeze_backbone(self, freeze_at):
        for layer_index in range(freeze_at):
            for p in self.features[layer_index].parameters():
                p.requires_grad = False

    def forward(self, x):
        res = []
        for i, m in enumerate(self.features):
            x = m(x)
            if i in (3, 6, 13, 17):
                res.append(x)
        return res

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2.0 / n) ** 0.5)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


StageSpec = namedtuple('StageSpec', ['index', 'block_count',
    'return_features', 'deformable'])


def _make_stage(transformation_module, in_channels, bottleneck_channels,
    out_channels, block_count, num_groups, stride_in_1x1, first_stride,
    dilation=1, deformable=False):
    blocks = []
    stride = first_stride
    for _ in range(block_count):
        blocks.append(transformation_module(in_channels,
            bottleneck_channels, out_channels, num_groups, stride_in_1x1,
            stride, dilation=dilation))
        stride = 1
        in_channels = out_channels
    return nn.Sequential(*blocks)


def _register_generic(module_dict, module_name, module):
    assert module_name not in module_dict
    module_dict[module_name] = module


class Registry(dict):
    """
    A helper class for managing registering modules, it extends a dictionary
    and provides a register functions.

    Eg. creeting a registry:
        some_registry = Registry({"default": default_module})

    There're two ways of registering new modules:
    1): normal way is just calling register function:
        def foo():
            ...
        some_registry.register("foo_module", foo)
    2): used as decorator when declaring the module:
        @some_registry.register("foo_module")
        @some_registry.register("foo_modeul_nickname")
        def foo():
            ...

    Access of module is just like using a dictionary, eg:
        f = some_registry["foo_modeul"]
    """

    def __init__(self, *args, **kwargs):
        super(Registry, self).__init__(*args, **kwargs)

    def register(self, module_name, module=None):
        if module is not None:
            _register_generic(self, module_name, module)
            return

        def register_fn(fn):
            _register_generic(self, module_name, fn)
            return fn
        return register_fn


class Bottleneck(nn.Module):

    def __init__(self, in_channels, bottleneck_channels, out_channels,
        num_groups, stride_in_1x1, stride, dilation, norm_func):
        super(Bottleneck, self).__init__()
        self.downsample = None
        if in_channels != out_channels:
            down_stride = stride if dilation == 1 else 1
            self.downsample = nn.Sequential(Conv2d(in_channels,
                out_channels, kernel_size=1, stride=down_stride, bias=False
                ), norm_func(out_channels))
            for modules in [self.downsample]:
                for l in modules.modules():
                    if isinstance(l, Conv2d):
                        nn.init.kaiming_uniform_(l.weight, a=1)
        if dilation > 1:
            stride = 1
        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)
        self.conv1 = Conv2d(in_channels, bottleneck_channels, kernel_size=1,
            stride=stride_1x1, bias=False)
        self.bn1 = norm_func(bottleneck_channels)
        self.conv2 = Conv2d(bottleneck_channels, bottleneck_channels,
            kernel_size=3, stride=stride_3x3, padding=dilation, bias=False,
            groups=num_groups, dilation=dilation)
        self.bn2 = norm_func(bottleneck_channels)
        self.conv3 = Conv2d(bottleneck_channels, out_channels, kernel_size=
            1, bias=False)
        self.bn3 = norm_func(out_channels)
        for l in [self.conv1, self.conv2, self.conv3]:
            nn.init.kaiming_uniform_(l.weight, a=1)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu_(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu_(out)
        out0 = self.conv3(out)
        out = self.bn3(out0)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = F.relu_(out)
        return out


class BaseStem(nn.Module):

    def __init__(self, cfg, norm_func):
        super(BaseStem, self).__init__()
        out_channels = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
        self.conv1 = Conv2d(3, out_channels, kernel_size=7, stride=2,
            padding=3, bias=False)
        self.bn1 = norm_func(out_channels)
        for l in [self.conv1]:
            nn.init.kaiming_uniform_(l.weight, a=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu_(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        return x


class ImageList(object):
    """
    Structure that holds a list of images (of possibly
    varying sizes) as a single tensor.
    This works by padding the images to the same size,
    and storing in a field the original sizes of each image
    """

    def __init__(self, tensors, image_sizes):
        """
        Arguments:
            tensors (tensor)
            image_sizes (list[tuple[int, int]])
        """
        self.tensors = tensors
        self.image_sizes = image_sizes

    def to(self, *args, **kwargs):
        cast_tensor = self.tensors.to(*args, **kwargs)
        return ImageList(cast_tensor, self.image_sizes)


def to_image_list(tensors, size_divisible=0):
    """
    tensors can be an ImageList, a torch.Tensor or
    an iterable of Tensors. It can't be a numpy array.
    When tensors is an iterable of Tensors, it pads
    the Tensors with zeros so that they have the same
    shape
    """
    if isinstance(tensors, torch.Tensor) and size_divisible > 0:
        tensors = [tensors]
    if isinstance(tensors, ImageList):
        return tensors
    elif isinstance(tensors, torch.Tensor):
        if tensors.dim() == 3:
            tensors = tensors[None]
        assert tensors.dim() == 4
        image_sizes = [tensor.shape[-2:] for tensor in tensors]
        return ImageList(tensors, image_sizes)
    elif isinstance(tensors, (tuple, list)):
        max_size = tuple(max(s) for s in zip(*[img.shape for img in tensors]))
        if size_divisible > 0:
            import math
            stride = size_divisible
            max_size = list(max_size)
            max_size[1] = int(math.ceil(max_size[1] / stride) * stride)
            max_size[2] = int(math.ceil(max_size[2] / stride) * stride)
            max_size = tuple(max_size)
        batch_shape = (len(tensors),) + max_size
        batched_imgs = tensors[0].new(*batch_shape).zero_()
        for img, pad_img in zip(tensors, batched_imgs):
            pad_img[:img.shape[0], :img.shape[1], :img.shape[2]].copy_(img)
        image_sizes = [im.shape[-2:] for im in tensors]
        return ImageList(batched_imgs, image_sizes)
    else:
        raise TypeError('Unsupported type for to_image_list: {}'.format(
            type(tensors)))


def build_densebox(cfg):
    return DenseBoxModule(cfg)


def build_backbone(cfg):
    """
    For Generalized_RCNN
    """
    assert cfg.MODEL.BACKBONE.CONV_BODY in registry.BACKBONES, 'cfg.MODEL.BACKBONE.CONV_BODY: {} are not registered in registry'.format(
        cfg.MODEL.BACKBONE.CONV_BODY)
    return registry.BACKBONES[cfg.MODEL.BACKBONE.CONV_BODY](cfg)


def build_retinanet(cfg):
    return RetinaNetModule(cfg)


def build_roi_box_head(cfg):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIBoxHead(cfg)


def build_roi_mask_head(cfg):
    return ROIMaskHead(cfg)


def build_roi_heads(cfg):
    roi_heads = []
    if not cfg.MODEL.RPN_ONLY:
        roi_heads.append(('box', build_roi_box_head(cfg)))
    if cfg.MODEL.MASK_ON:
        roi_heads.append(('mask', build_roi_mask_head(cfg)))
    if roi_heads:
        roi_heads = CombinedROIHeads(cfg, roi_heads)
    return roi_heads


def build_rpn(cfg):
    """
    This gives the gist of it. Not super important because it doesn't change as much
    """
    return RPNModule(cfg)


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()
        self.backbone = build_backbone(cfg)
        if cfg.MODEL.RETINANET_ON:
            self.rpn = build_retinanet(cfg)
            self.roi_heads = []
        elif cfg.MODEL.DENSEBOX_ON:
            self.rpn = build_densebox(cfg)
            self.roi_heads = []
        else:
            self.rpn = build_rpn(cfg)
            self.roi_heads = build_roi_heads(cfg)

    def generate_anchors(self, cfg):
        if cfg.INPUT.CROP_SIZE_TRAIN > 0:
            mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // cfg.SOLVER.NUM_GPUS
            self.rpn.generate_anchors(mini_batch_size, cfg.INPUT.
                CROP_SIZE_TRAIN)

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError('In training mode, targets should be passed')
        images = to_image_list(images)
        features = self.backbone(images.tensors)
        proposals, proposal_losses = self.rpn(images, features, targets)
        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals,
                targets)
        else:
            x = features
            result = proposals
            detector_losses = {}
        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses
        return result


class Head(nn.Module):

    def __init__(self, fpn, rpn):
        super(Head, self).__init__()
        self.fpn = fpn
        self.rpn = rpn

    def forward(self, images, features, targets):
        features = self.fpn(features)
        proposals, proposal_losses = self.rpn(images, features, targets)
        if self.training:
            losses = {}
            losses.update(proposal_losses)
            return losses
        return proposals


def build_retinanet_nas_head(cfg):
    return RetinaNet_NasHeadModule(cfg)


def build_nas_head(cfg):
    return NasHeadModule(cfg)


def build_decoder(cfg):
    if cfg.SEARCH.DECODER.VERSION == 2:
        Decoder = MicroDecoder_v2
    top_blocks = LastLevelP6P7(cfg.SEARCH.DECODER.AGG_SIZE, cfg.SEARCH.
        DECODER.AGG_SIZE)
    arch_info = cfg.SEARCH.DECODER.CONFIG
    decoder_layer_num = cfg.SEARCH.DECODER.NUM_CELLS
    if cfg.SEARCH.NAS_DECODER_ON and cfg.SEARCH.NAS_HEAD_ON:
        sample_arch = arch_info[:decoder_layer_num]
    elif cfg.SEARCH.NAS_DECODER_ON and not cfg.SEARCH.NAS_HEAD_ON:
        sample_arch = arch_info
    return Decoder(cfg.MODEL.BACKBONE.ENCODER_OUT_CHANNELS, sample_arch,
        agg_size=cfg.SEARCH.DECODER.AGG_SIZE, repeats=cfg.SEARCH.DECODER.
        REPEATS, top_blocks=top_blocks)


class SingleStageDetector(nn.Module):
    """
    Main class for Single Stage Detector. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - fpn
    = rpn
    """

    def __init__(self, cfg):
        super(SingleStageDetector, self).__init__()
        self.encoder = build_backbone(cfg)
        if cfg.MODEL.RETINANET_ON:
            if cfg.SEARCH.NAS_HEAD_ON:
                rpn = build_retinanet_nas_head(cfg)
                self.rpn_name = 'nas_head'
            else:
                rpn = build_retinanet(cfg)
                self.rpn_name = 'retinanet'
        elif cfg.MODEL.DENSEBOX_ON:
            if cfg.SEARCH.NAS_HEAD_ON:
                rpn = build_nas_head(cfg)
                self.rpn_name = 'nas_head'
            else:
                rpn = build_densebox(cfg)
                self.rpn_name = 'densebox'
        else:
            rpn = build_rpn(cfg)
        fpn = build_decoder(cfg)
        self.decoder = Head(fpn, rpn)

    def generate_anchors(self, mini_batch_size, img_size):
        if self.rpn_name == 'retinanet':
            self.decoder.rpn.generate_anchors(mini_batch_size, img_size)

    def reset_anchors(self):
        self.decoder.rpn.anchors = None

    def forward(self, images, targets=None, features=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)
            features (list[Tensor]): encoder output features (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError('In training mode, targets should be passed')
        if features is None:
            images = to_image_list(images)
            features = self.encoder(images.tensors)
        return self.decoder(images, features, targets)


def cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


class LevelMapper(object):
    """Determine which FPN level each RoI in a set of RoIs should map to based
    on the heuristic in the FPN paper.
    """

    def __init__(self, k_min, k_max, canonical_scale=224, canonical_level=4,
        eps=1e-06):
        """
        Arguments:
            k_min (int)
            k_max (int)
            canonical_scale (int)
            canonical_level (int)
            eps (float)
        """
        self.k_min = k_min
        self.k_max = k_max
        self.s0 = canonical_scale
        self.lvl0 = canonical_level
        self.eps = eps

    def __call__(self, boxlists):
        """
        Arguments:
            boxlists (list[BoxList])
        """
        s = torch.sqrt(cat([boxlist.area() for boxlist in boxlists]))
        target_lvls = torch.floor(self.lvl0 + torch.log2(s / self.s0 + self
            .eps))
        target_lvls = torch.clamp(target_lvls, min=self.k_min, max=self.k_max)
        return target_lvls.to(torch.int64) - self.k_min


class Pooler(nn.Module):
    """
    Pooler for Detection with or without FPN.
    It currently hard-code ROIAlign in the implementation,
    but that can be made more generic later on.
    Also, the requirement of passing the scales is not strictly necessary, as they
    can be inferred from the size of the feature map / size of original image,
    which is available thanks to the BoxList.
    """

    def __init__(self, output_size, scales, sampling_ratio):
        """
        Arguments:
            output_size (list[tuple[int]] or list[int]): output size for the pooled region
            scales (list[float]): scales for each Pooler
            sampling_ratio (int): sampling ratio for ROIAlign
        """
        super(Pooler, self).__init__()
        poolers = []
        for scale in scales:
            poolers.append(ROIAlign(output_size, spatial_scale=scale,
                sampling_ratio=sampling_ratio))
        self.poolers = nn.ModuleList(poolers)
        self.output_size = output_size
        lvl_min = -torch.log2(torch.tensor(scales[0], dtype=torch.float32)
            ).item()
        lvl_max = -torch.log2(torch.tensor(scales[-1], dtype=torch.float32)
            ).item()
        self.map_levels = LevelMapper(lvl_min, lvl_max)

    def convert_to_roi_format(self, boxes):
        concat_boxes = cat([b.bbox for b in boxes], dim=0)
        device, dtype = concat_boxes.device, concat_boxes.dtype
        ids = cat([torch.full((len(b), 1), i, dtype=dtype, device=device) for
            i, b in enumerate(boxes)], dim=0)
        rois = torch.cat([ids, concat_boxes], dim=1)
        return rois

    def forward(self, x, boxes):
        """
        Arguments:
            x (list[Tensor]): feature maps for each level
            boxes (list[BoxList]): boxes to be used to perform the pooling operation.
        Returns:
            result (Tensor)
        """
        num_levels = len(self.poolers)
        rois = self.convert_to_roi_format(boxes)
        if num_levels == 1:
            return self.poolers[0](x[0], rois)
        levels = self.map_levels(boxes)
        num_rois = len(rois)
        num_channels = x[0].shape[1]
        output_size = self.output_size[0]
        dtype, device = x[0].dtype, x[0].device
        result = torch.zeros((num_rois, num_channels, output_size,
            output_size), dtype=dtype, device=device)
        for level, (per_level_feature, pooler) in enumerate(zip(x, self.
            poolers)):
            idx_in_level = torch.nonzero(levels == level).squeeze(1)
            rois_per_level = rois[idx_in_level]
            result[idx_in_level] = pooler(per_level_feature, rois_per_level
                ).to(dtype)
        return result


class Matcher(object):
    """
    This class assigns to each predicted "element" (e.g., a box) a ground-truth
    element. Each predicted element will have exactly zero or one matches; each
    ground-truth element may be assigned to zero or more predicted elements.

    Matching is based on the MxN match_quality_matrix, that characterizes how well
    each (ground-truth, predicted)-pair match. For example, if the elements are
    boxes, the matrix may contain box IoU overlap values.

    The matcher returns a tensor of size N containing the index of the ground-truth
    element m that matches to prediction n. If there is no match, a negative value
    is returned.
    """
    BELOW_LOW_THRESHOLD = -1
    BETWEEN_THRESHOLDS = -2

    def __init__(self, high_threshold, low_threshold,
        allow_low_quality_matches=False):
        """
        Args:
            high_threshold (float): quality values greater than or equal to
                this value are candidate matches.
            low_threshold (float): a lower quality threshold used to stratify
                matches into three levels:
                1) matches >= high_threshold
                2) BETWEEN_THRESHOLDS matches in [low_threshold, high_threshold)
                3) BELOW_LOW_THRESHOLD matches in [0, low_threshold)
            allow_low_quality_matches (bool): if True, produce additional matches
                for predictions that have only low-quality match candidates. See
                set_low_quality_matches_ for more details.
        """
        assert low_threshold <= high_threshold
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, match_quality_matrix):
        """
        Args:
            match_quality_matrix (Tensor[float]): an MxN tensor, containing the
            pairwise quality between M ground-truth elements and N predicted elements.

        Returns:
            matches (Tensor[int64]): an N tensor where N[i] is a matched gt in
            [0, M - 1] or a negative value indicating that prediction i could not
            be matched.
        """
        if match_quality_matrix.numel() == 0:
            if match_quality_matrix.shape[0] == 0:
                raise ValueError(
                    'No ground-truth boxes available for one of the images during training'
                    )
            else:
                raise ValueError(
                    'No proposal boxes available for one of the images during training'
                    )
        matched_vals, matches = match_quality_matrix.max(dim=0)
        if self.allow_low_quality_matches:
            all_matches = matches.clone()
        below_low_threshold = matched_vals < self.low_threshold
        between_thresholds = (matched_vals >= self.low_threshold) & (
            matched_vals < self.high_threshold)
        above_high_thresholds = matched_vals >= self.high_threshold
        matches = matches * above_high_thresholds.long()
        matches = matches + below_low_threshold.long(
            ) * Matcher.BELOW_LOW_THRESHOLD
        matches = matches + between_thresholds.long(
            ) * Matcher.BETWEEN_THRESHOLDS
        if self.allow_low_quality_matches:
            self.set_low_quality_matches_(matches, all_matches,
                match_quality_matrix)
        return matches

    def set_low_quality_matches_(self, matches, all_matches,
        match_quality_matrix):
        """
        Produce additional matches for predictions that have only low-quality matches.
        Specifically, for each ground-truth find the set of predictions that have
        maximum overlap with it (including ties); for each prediction in that set, if
        it is unmatched, then match it to the ground-truth with which it has the highest
        quality value.
        """
        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)
        gt_pred_pairs_of_highest_quality = torch.nonzero(
            match_quality_matrix == highest_quality_foreach_gt[:, (None)])
        pred_inds_to_update = gt_pred_pairs_of_highest_quality[:, (1)]
        matches[pred_inds_to_update] = all_matches[pred_inds_to_update]


class BalancedPositiveNegativeSampler(object):
    """
    This class samples batches, ensuring that they contain a fixed proportion of positives
    """

    def __init__(self, batch_size_per_image, positive_fraction):
        """
        Arguments:
            batch_size_per_image (int): number of elements to be selected per image
            positive_fraction (float): percentace of positive elements per batch
        """
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction

    def __call__(self, matched_idxs):
        """
        Arguments:
            matched idxs: list of tensors containing -1, 0 or positive values.
                Each tensor corresponds to a specific image.
                -1 values are ignored, 0 are considered as negatives and > 0 as
                positives.

        Returns:
            pos_idx (list[tensor])
            neg_idx (list[tensor])

        Returns two lists of binary masks for each image.
        The first list contains the positive elements that were selected,
        and the second list the negative example.
        """
        pos_idx = []
        neg_idx = []
        for matched_idxs_per_image in matched_idxs:
            positive = torch.nonzero(matched_idxs_per_image >= 1).squeeze(1)
            negative = torch.nonzero(matched_idxs_per_image == 0).squeeze(1)
            num_pos = int(self.batch_size_per_image * self.positive_fraction)
            num_pos = min(positive.numel(), num_pos)
            num_neg = self.batch_size_per_image - num_pos
            num_neg = min(negative.numel(), num_neg)
            perm1 = torch.randperm(positive.numel(), device=positive.device)[:
                num_pos]
            perm2 = torch.randperm(negative.numel(), device=negative.device)[:
                num_neg]
            pos_idx_per_image = positive[perm1]
            neg_idx_per_image = negative[perm2]
            pos_idx_per_image_mask = torch.zeros_like(matched_idxs_per_image,
                dtype=torch.uint8)
            neg_idx_per_image_mask = torch.zeros_like(matched_idxs_per_image,
                dtype=torch.uint8)
            pos_idx_per_image_mask[pos_idx_per_image] = 1
            neg_idx_per_image_mask[neg_idx_per_image] = 1
            pos_idx.append(pos_idx_per_image_mask)
            neg_idx.append(neg_idx_per_image_mask)
        return pos_idx, neg_idx


def smooth_l1_loss(input, target, beta=1.0 / 9, size_average=True):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    n = torch.abs(input - target)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if size_average:
        return loss.mean()
    return loss.sum()


def boxlist_iou(boxlist1, boxlist2):
    """Compute the intersection over union of two set of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Arguments:
      box1: (BoxList) bounding boxes, sized [N,4].
      box2: (BoxList) bounding boxes, sized [M,4].

    Returns:
      (tensor) iou, sized [N,M].

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    """
    if boxlist1.size != boxlist2.size:
        raise RuntimeError('boxlists should have same image size, got {}, {}'
            .format(boxlist1, boxlist2))
    N = len(boxlist1)
    M = len(boxlist2)
    area1 = boxlist1.area()
    area2 = boxlist2.area()
    box1, box2 = boxlist1.bbox, boxlist2.bbox
    lt = torch.max(box1[:, (None), :2], box2[:, :2])
    rb = torch.min(box1[:, (None), 2:], box2[:, 2:])
    TO_REMOVE = 1
    wh = (rb - lt + TO_REMOVE).clamp(min=0)
    inter = wh[:, :, (0)] * wh[:, :, (1)]
    iou = inter / (area1[:, (None)] + area2 - inter)
    return iou


class FastRCNNLossComputation(object):
    """
    Computes the loss for Faster R-CNN.
    Also supports FPN
    """

    def __init__(self, proposal_matcher, fg_bg_sampler, box_coder):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.box_coder = box_coder

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        target = target.copy_with_fields('labels')
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field('matched_idxs', matched_idxs)
        return matched_targets

    def prepare_targets(self, proposals, targets):
        labels = []
        regression_targets = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image)
            matched_idxs = matched_targets.get_field('matched_idxs')
            labels_per_image = matched_targets.get_field('labels')
            labels_per_image = labels_per_image.to(dtype=torch.int64)
            bg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_inds] = 0
            ignore_inds = matched_idxs == Matcher.BETWEEN_THRESHOLDS
            labels_per_image[ignore_inds] = -1
            regression_targets_per_image = self.box_coder.encode(
                matched_targets.bbox, proposals_per_image.bbox)
            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)
        return labels, regression_targets

    def subsample(self, proposals, targets):
        """
        This method performs the positive/negative sampling, and return
        the sampled proposals.
        Note: this function keeps a state.

        Arguments:
            proposals (list[BoxList])
            targets (list[BoxList])
        """
        labels, regression_targets = self.prepare_targets(proposals, targets)
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        proposals = list(proposals)
        for labels_per_image, regression_targets_per_image, proposals_per_image in zip(
            labels, regression_targets, proposals):
            proposals_per_image.add_field('labels', labels_per_image)
            proposals_per_image.add_field('regression_targets',
                regression_targets_per_image)
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(zip(
            sampled_pos_inds, sampled_neg_inds)):
            img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img
                ).squeeze(1)
            proposals_per_image = proposals[img_idx][img_sampled_inds]
            proposals[img_idx] = proposals_per_image
        self._proposals = proposals
        return proposals

    def __call__(self, class_logits, box_regression):
        """
        Computes the loss for Faster R-CNN.
        This requires that the subsample method has been called beforehand.

        Arguments:
            class_logits (list[Tensor])
            box_regression (list[Tensor])

        Returns:
            classification_loss (Tensor)
            box_loss (Tensor)
        """
        class_logits = cat(class_logits, dim=0)
        box_regression = cat(box_regression, dim=0)
        device = class_logits.device
        if not hasattr(self, '_proposals'):
            raise RuntimeError('subsample needs to be called before')
        proposals = self._proposals
        labels = cat([proposal.get_field('labels') for proposal in
            proposals], dim=0)
        regression_targets = cat([proposal.get_field('regression_targets') for
            proposal in proposals], dim=0)
        classification_loss = F.cross_entropy(class_logits, labels)
        sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
        labels_pos = labels[sampled_pos_inds_subset]
        map_inds = 4 * labels_pos[:, (None)] + torch.tensor([0, 1, 2, 3],
            device=device)
        box_loss = smooth_l1_loss(box_regression[sampled_pos_inds_subset[:,
            (None)], map_inds], regression_targets[sampled_pos_inds_subset],
            size_average=False, beta=1)
        box_loss = box_loss / labels.numel()
        return classification_loss, box_loss


class BoxCoder(object):
    """
    This class encodes and decodes a set of bounding boxes into
    the representation used for training the regressors.
    """

    def __init__(self, weights, bbox_xform_clip=math.log(1000.0 / 16)):
        """
        Arguments:
            weights (4-element tuple)
            bbox_xform_clip (float)
        """
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def encode(self, reference_boxes, proposals):
        """
        Encode a set of proposals with respect to some
        reference boxes

        Arguments:
            reference_boxes (Tensor): reference boxes
            proposals (Tensor): boxes to be encoded
        """
        TO_REMOVE = 1
        ex_widths = proposals[:, (2)] - proposals[:, (0)] + TO_REMOVE
        ex_heights = proposals[:, (3)] - proposals[:, (1)] + TO_REMOVE
        ex_ctr_x = proposals[:, (0)] + 0.5 * ex_widths
        ex_ctr_y = proposals[:, (1)] + 0.5 * ex_heights
        gt_widths = reference_boxes[:, (2)] - reference_boxes[:, (0)
            ] + TO_REMOVE
        gt_heights = reference_boxes[:, (3)] - reference_boxes[:, (1)
            ] + TO_REMOVE
        gt_ctr_x = reference_boxes[:, (0)] + 0.5 * gt_widths
        gt_ctr_y = reference_boxes[:, (1)] + 0.5 * gt_heights
        wx, wy, ww, wh = self.weights
        targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
        targets_dw = ww * torch.log(gt_widths / ex_widths)
        targets_dh = wh * torch.log(gt_heights / ex_heights)
        targets = torch.stack((targets_dx, targets_dy, targets_dw,
            targets_dh), dim=1)
        return targets

    def decode(self, rel_codes, boxes):
        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.

        Arguments:
            rel_codes (Tensor): encoded boxes
            boxes (Tensor): reference boxes.
        """
        boxes = boxes.to(rel_codes.dtype)
        TO_REMOVE = 1
        widths = boxes[:, (2)] - boxes[:, (0)] + TO_REMOVE
        heights = boxes[:, (3)] - boxes[:, (1)] + TO_REMOVE
        ctr_x = boxes[:, (0)] + 0.5 * widths
        ctr_y = boxes[:, (1)] + 0.5 * heights
        wx, wy, ww, wh = self.weights
        dx = rel_codes[:, 0::4] / wx
        dy = rel_codes[:, 1::4] / wy
        dw = rel_codes[:, 2::4] / ww
        dh = rel_codes[:, 3::4] / wh
        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        dh = torch.clamp(dh, max=self.bbox_xform_clip)
        pred_ctr_x = dx * widths[:, (None)] + ctr_x[:, (None)]
        pred_ctr_y = dy * heights[:, (None)] + ctr_y[:, (None)]
        pred_w = torch.exp(dw) * widths[:, (None)]
        pred_h = torch.exp(dh) * heights[:, (None)]
        pred_boxes = torch.zeros_like(rel_codes)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w - 1
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h - 1
        return pred_boxes


def make_roi_box_loss_evaluator(cfg):
    matcher = Matcher(cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD, cfg.MODEL.
        ROI_HEADS.BG_IOU_THRESHOLD, allow_low_quality_matches=False)
    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights)
    fg_bg_sampler = BalancedPositiveNegativeSampler(cfg.MODEL.ROI_HEADS.
        BATCH_SIZE_PER_IMAGE, cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION)
    loss_evaluator = FastRCNNLossComputation(matcher, fg_bg_sampler, box_coder)
    return loss_evaluator


def boxlist_nms(boxlist, nms_thresh, max_proposals=-1, score_field='scores'):
    """
    Performs non-maximum suppression on a boxlist, with scores specified
    in a boxlist field via score_field.

    Arguments:
        boxlist(BoxList)
        nms_thresh (float)
        max_proposals (int): if > 0, then only the top max_proposals are kept
            after non-maximum suppression
        score_field (str)
    """
    if nms_thresh <= 0:
        return boxlist
    mode = boxlist.mode
    boxlist = boxlist.convert('xyxy')
    boxes = boxlist.bbox
    score = boxlist.get_field(score_field)
    keep = _box_nms(boxes, score, nms_thresh)
    if max_proposals > 0:
        keep = keep[:max_proposals]
    boxlist = boxlist[keep]
    return boxlist.convert(mode)


FLIP_TOP_BOTTOM = 1


FLIP_LEFT_RIGHT = 0


class BoxList(object):
    """
    This class represents a set of bounding boxes.
    The bounding boxes are represented as a Nx4 Tensor.
    In order to uniquely determine the bounding boxes with respect
    to an image, we also store the corresponding image dimensions.
    They can contain extra information that is specific to each bounding box, such as
    labels.
    """

    def __init__(self, bbox, image_size, mode='xyxy'):
        device = bbox.device if isinstance(bbox, torch.Tensor
            ) else torch.device('cpu')
        bbox = torch.as_tensor(bbox, dtype=torch.float32, device=device)
        if bbox.ndimension() != 2:
            raise ValueError('bbox should have 2 dimensions, got {}'.format
                (bbox.ndimension()))
        if bbox.size(-1) != 4:
            raise ValueError(
                'last dimension of bbox should have a size of 4, got {}'.
                format(bbox.size(-1)))
        if mode not in ('xyxy', 'xywh'):
            raise ValueError("mode should be 'xyxy' or 'xywh'")
        self.bbox = bbox
        self.size = image_size
        self.mode = mode
        self.extra_fields = {}

    def add_field(self, field, field_data):
        self.extra_fields[field] = field_data

    def get_field(self, field):
        return self.extra_fields[field]

    def has_field(self, field):
        return field in self.extra_fields

    def fields(self):
        return list(self.extra_fields.keys())

    def _copy_extra_fields(self, bbox):
        for k, v in bbox.extra_fields.items():
            self.extra_fields[k] = v

    def convert(self, mode):
        if mode not in ('xyxy', 'xywh'):
            raise ValueError("mode should be 'xyxy' or 'xywh'")
        if mode == self.mode:
            return self
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        if mode == 'xyxy':
            bbox = torch.cat((xmin, ymin, xmax, ymax), dim=-1)
            bbox = BoxList(bbox, self.size, mode=mode)
        else:
            TO_REMOVE = 1
            bbox = torch.cat((xmin, ymin, xmax - xmin + TO_REMOVE, ymax -
                ymin + TO_REMOVE), dim=-1)
            bbox = BoxList(bbox, self.size, mode=mode)
        bbox._copy_extra_fields(self)
        return bbox

    def _split_into_xyxy(self):
        if self.mode == 'xyxy':
            xmin, ymin, xmax, ymax = self.bbox.split(1, dim=-1)
            return xmin, ymin, xmax, ymax
        elif self.mode == 'xywh':
            TO_REMOVE = 1
            xmin, ymin, w, h = self.bbox.split(1, dim=-1)
            return xmin, ymin, xmin + (w - TO_REMOVE).clamp(min=0), ymin + (h -
                TO_REMOVE).clamp(min=0)
        else:
            raise RuntimeError('Should not be here')

    def resize(self, size, *args, **kwargs):
        """
        Returns a resized copy of this bounding box

        :param size: The requested size in pixels, as a 2-tuple:
            (width, height).
        """
        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(size,
            self.size))
        if ratios[0] == ratios[1]:
            ratio = ratios[0]
            scaled_box = self.bbox * ratio
            bbox = BoxList(scaled_box, size, mode=self.mode)
            for k, v in self.extra_fields.items():
                if not isinstance(v, torch.Tensor):
                    v = v.resize(size, *args, **kwargs)
                bbox.add_field(k, v)
            return bbox
        ratio_width, ratio_height = ratios
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        scaled_xmin = xmin * ratio_width
        scaled_xmax = xmax * ratio_width
        scaled_ymin = ymin * ratio_height
        scaled_ymax = ymax * ratio_height
        scaled_box = torch.cat((scaled_xmin, scaled_ymin, scaled_xmax,
            scaled_ymax), dim=-1)
        bbox = BoxList(scaled_box, size, mode='xyxy')
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.resize(size, *args, **kwargs)
            bbox.add_field(k, v)
        return bbox.convert(self.mode)

    def transpose(self, method):
        """
        Transpose bounding box (flip or rotate in 90 degree steps)
        :param method: One of :py:attr:`PIL.Image.FLIP_LEFT_RIGHT`,
          :py:attr:`PIL.Image.FLIP_TOP_BOTTOM`, :py:attr:`PIL.Image.ROTATE_90`,
          :py:attr:`PIL.Image.ROTATE_180`, :py:attr:`PIL.Image.ROTATE_270`,
          :py:attr:`PIL.Image.TRANSPOSE` or :py:attr:`PIL.Image.TRANSVERSE`.
        """
        if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM):
            raise NotImplementedError(
                'Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented')
        image_width, image_height = self.size
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        if method == FLIP_LEFT_RIGHT:
            TO_REMOVE = 1
            transposed_xmin = image_width - xmax - TO_REMOVE
            transposed_xmax = image_width - xmin - TO_REMOVE
            transposed_ymin = ymin
            transposed_ymax = ymax
        elif method == FLIP_TOP_BOTTOM:
            transposed_xmin = xmin
            transposed_xmax = xmax
            transposed_ymin = image_height - ymax
            transposed_ymax = image_height - ymin
        transposed_boxes = torch.cat((transposed_xmin, transposed_ymin,
            transposed_xmax, transposed_ymax), dim=-1)
        bbox = BoxList(transposed_boxes, self.size, mode='xyxy')
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.transpose(method)
            bbox.add_field(k, v)
        return bbox.convert(self.mode)

    def crop(self, box, remove_empty=False):
        """
        Crops a rectangular region from this bounding box. The box is a
        4-tuple defining the left, upper, right, and lower pixel
        coordinate.
        """
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        w, h = box[2] - box[0], box[3] - box[1]
        cropped_xmin = (xmin - box[0]).clamp(min=0, max=w)
        cropped_ymin = (ymin - box[1]).clamp(min=0, max=h)
        cropped_xmax = (xmax - box[0]).clamp(min=0, max=w)
        cropped_ymax = (ymax - box[1]).clamp(min=0, max=h)
        cropped_box = torch.cat((cropped_xmin, cropped_ymin, cropped_xmax,
            cropped_ymax), dim=-1)
        bbox = BoxList(cropped_box, (w, h), mode='xyxy')
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.crop(box)
            bbox.add_field(k, v)
        if remove_empty:
            box = bbox.bbox
            keep = (box[:, (3)] > box[:, (1)]) & (box[:, (2)] > box[:, (0)])
            bbox = bbox[keep]
        return bbox.convert(self.mode)

    def pad(self, padding):
        l, t, r, b = padding
        w = l + self.size[0] + r
        h = t + self.size[1] + b
        self.size = w, h
        new_fields = {}
        for k, v in self.extra_fields.items():
            if hasattr(v, 'pad'):
                v = v.pad(padding)
                new_fields[k] = v
        for k, v in new_fields.items():
            self.add_field(k, v)
        return self

    def to(self, device):
        bbox = BoxList(self.bbox.to(device), self.size, self.mode)
        for k, v in self.extra_fields.items():
            if hasattr(v, 'to'):
                v = v.to(device)
            bbox.add_field(k, v)
        return bbox

    def to_tensor(self):
        bbox = BoxList(self.bbox, self.size, self.mode)
        for k, v in self.extra_fields.items():
            if hasattr(v, 'to_tensor'):
                v = v.to_tensor()
            bbox.add_field(k, v)
        return bbox

    def __getitem__(self, item):
        bbox = BoxList(self.bbox[item], self.size, self.mode)
        for k, v in self.extra_fields.items():
            bbox.add_field(k, v[item])
        return bbox

    def __len__(self):
        return self.bbox.shape[0]

    def clip_to_image(self, remove_empty=True):
        TO_REMOVE = 1
        self.bbox[:, (0)].clamp_(min=0, max=self.size[0] - TO_REMOVE)
        self.bbox[:, (1)].clamp_(min=0, max=self.size[1] - TO_REMOVE)
        self.bbox[:, (2)].clamp_(min=0, max=self.size[0] - TO_REMOVE)
        self.bbox[:, (3)].clamp_(min=0, max=self.size[1] - TO_REMOVE)
        if remove_empty:
            box = self.bbox
            keep = (box[:, (3)] > box[:, (1)]) & (box[:, (2)] > box[:, (0)])
            return self[keep]
        return self

    def area(self):
        box = self.bbox
        if self.mode == 'xyxy':
            TO_REMOVE = 1
            area = (box[:, (2)] - box[:, (0)] + TO_REMOVE) * (box[:, (3)] -
                box[:, (1)] + TO_REMOVE)
        elif self.mode == 'xywh':
            area = box[:, (2)] * box[:, (3)]
        else:
            raise RuntimeError('Should not be here')
        return area

    def copy_with_fields(self, fields):
        bbox = BoxList(self.bbox, self.size, self.mode)
        if not isinstance(fields, (list, tuple)):
            fields = [fields]
        for field in fields:
            bbox.add_field(field, self.get_field(field))
        return bbox

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += 'num_boxes={}, '.format(len(self))
        s += 'image_width={}, '.format(self.size[0])
        s += 'image_height={}, '.format(self.size[1])
        s += 'mode={})'.format(self.mode)
        return s


def _cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def cat_boxlist(bboxes):
    """
    Concatenates a list of BoxList (having the same image size) into a
    single BoxList

    Arguments:
        bboxes (list[BoxList])
    """
    assert isinstance(bboxes, (list, tuple))
    assert all(isinstance(bbox, BoxList) for bbox in bboxes)
    size = bboxes[0].size
    assert all(bbox.size == size for bbox in bboxes)
    mode = bboxes[0].mode
    assert all(bbox.mode == mode for bbox in bboxes)
    fields = set(bboxes[0].fields())
    assert all(set(bbox.fields()) == fields for bbox in bboxes)
    cat_boxes = BoxList(_cat([bbox.bbox for bbox in bboxes], dim=0), size, mode
        )
    for field in fields:
        data = _cat([bbox.get_field(field) for bbox in bboxes], dim=0)
        cat_boxes.add_field(field, data)
    return cat_boxes


class PostProcessor(nn.Module):
    """
    From a set of classification scores, box regression and proposals,
    computes the post-processed boxes, and applies NMS to obtain the
    final results
    """

    def __init__(self, score_thresh=0.05, nms=0.5, detections_per_img=100,
        box_coder=None):
        """
        Arguments:
            score_thresh (float)
            nms (float)
            detections_per_img (int)
            box_coder (BoxCoder)
        """
        super(PostProcessor, self).__init__()
        self.score_thresh = score_thresh
        self.nms = nms
        self.detections_per_img = detections_per_img
        if box_coder is None:
            box_coder = BoxCoder(weights=(10.0, 10.0, 5.0, 5.0))
        self.box_coder = box_coder

    def forward(self, x, boxes):
        """
        Arguments:
            x (tuple[tensor, tensor]): x contains the class logits
                and the box_regression from the model.
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for ech image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra fields labels and scores
        """
        class_logits, box_regression = x
        class_prob = F.softmax(class_logits, -1)
        image_shapes = [box.size for box in boxes]
        boxes_per_image = [len(box) for box in boxes]
        concat_boxes = torch.cat([a.bbox for a in boxes], dim=0)
        proposals = self.box_coder.decode(box_regression.view(sum(
            boxes_per_image), -1), concat_boxes)
        num_classes = class_prob.shape[1]
        proposals = proposals.split(boxes_per_image, dim=0)
        class_prob = class_prob.split(boxes_per_image, dim=0)
        results = []
        for prob, boxes_per_img, image_shape in zip(class_prob, proposals,
            image_shapes):
            boxlist = self.prepare_boxlist(boxes_per_img, prob, image_shape)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = self.filter_results(boxlist, num_classes)
            results.append(boxlist)
        return results

    def prepare_boxlist(self, boxes, scores, image_shape):
        """
        Returns BoxList from `boxes` and adds probability scores information
        as an extra field
        `boxes` has shape (#detections, 4 * #classes), where each row represents
        a list of predicted bounding boxes for each of the object classes in the
        dataset (including the background class). The detections in each row
        originate from the same object proposal.
        `scores` has shape (#detection, #classes), where each row represents a list
        of object detection confidence scores for each of the object classes in the
        dataset (including the background class). `scores[i, j]`` corresponds to the
        box at `boxes[i, j * 4:(j + 1) * 4]`.
        """
        boxes = boxes.reshape(-1, 4)
        scores = scores.reshape(-1)
        boxlist = BoxList(boxes, image_shape, mode='xyxy')
        boxlist.add_field('scores', scores)
        return boxlist

    def filter_results(self, boxlist, num_classes):
        """Returns bounding-box detection results by thresholding on scores and
        applying non-maximum suppression (NMS).
        """
        boxes = boxlist.bbox.reshape(-1, num_classes * 4)
        scores = boxlist.get_field('scores').reshape(-1, num_classes)
        device = scores.device
        result = []
        inds_all = scores > self.score_thresh
        for j in range(1, num_classes):
            inds = inds_all[:, (j)].nonzero().squeeze(1)
            scores_j = scores[inds, j]
            boxes_j = boxes[(inds), j * 4:(j + 1) * 4]
            boxlist_for_class = BoxList(boxes_j, boxlist.size, mode='xyxy')
            boxlist_for_class.add_field('scores', scores_j)
            boxlist_for_class = boxlist_nms(boxlist_for_class, self.nms,
                score_field='scores')
            num_labels = len(boxlist_for_class)
            boxlist_for_class.add_field('labels', torch.full((num_labels,),
                j, dtype=torch.int64, device=device))
            result.append(boxlist_for_class)
        result = cat_boxlist(result)
        number_of_detections = len(result)
        if number_of_detections > self.detections_per_img > 0:
            cls_scores = result.get_field('scores')
            image_thresh, _ = torch.kthvalue(cls_scores.cpu(), 
                number_of_detections - self.detections_per_img + 1)
            keep = cls_scores >= image_thresh.item()
            keep = torch.nonzero(keep).squeeze(1)
            result = result[keep]
        return result


def get_group_gn(dim, dim_per_gp, num_groups):
    """get number of groups used by GroupNorm, based on number of channels."""
    assert dim_per_gp == -1 or num_groups == -1, 'GroupNorm: can only specify G or C/G.'
    if dim_per_gp > 0:
        assert dim % dim_per_gp == 0, 'dim: {}, dim_per_gp: {}'.format(dim,
            dim_per_gp)
        group_gn = dim // dim_per_gp
    else:
        assert dim % num_groups == 0, 'dim: {}, num_groups: {}'.format(dim,
            num_groups)
        group_gn = num_groups
    return group_gn


_global_config['MODEL'] = 4


def group_norm(out_channels, affine=True, divisor=1):
    out_channels = out_channels // divisor
    dim_per_gp = cfg.MODEL.GROUP_NORM.DIM_PER_GP // divisor
    num_groups = cfg.MODEL.GROUP_NORM.NUM_GROUPS // divisor
    eps = cfg.MODEL.GROUP_NORM.EPSILON
    return torch.nn.GroupNorm(get_group_gn(out_channels, dim_per_gp,
        num_groups), out_channels, eps, affine)


def make_fc(dim_in, hidden_dim, use_gn=False):
    """
        Caffe2 implementation uses XavierFill, which in fact
        corresponds to kaiming_uniform_ in PyTorch
    """
    if use_gn:
        fc = nn.Linear(dim_in, hidden_dim, bias=False)
        nn.init.kaiming_uniform_(fc.weight, a=1)
        return nn.Sequential(fc, group_norm(hidden_dim))
    fc = nn.Linear(dim_in, hidden_dim)
    nn.init.kaiming_uniform_(fc.weight, a=1)
    nn.init.constant_(fc.bias, 0)
    return fc


class FastRCNNPredictor(nn.Module):

    def __init__(self, config, pretrained=None):
        super(FastRCNNPredictor, self).__init__()
        stage_index = 4
        stage2_relative_factor = 2 ** (stage_index - 1)
        res2_out_channels = config.MODEL.RESNETS.RES2_OUT_CHANNELS
        num_inputs = res2_out_channels * stage2_relative_factor
        num_classes = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=7)
        self.cls_score = nn.Linear(num_inputs, num_classes)
        self.bbox_pred = nn.Linear(num_inputs, num_classes * 4)
        nn.init.normal_(self.cls_score.weight, mean=0, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)
        nn.init.normal_(self.bbox_pred.weight, mean=0, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        cls_logit = self.cls_score(x)
        bbox_pred = self.bbox_pred(x)
        return cls_logit, bbox_pred


class FPNPredictor(nn.Module):

    def __init__(self, cfg):
        super(FPNPredictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        self.cls_score = nn.Linear(representation_size, num_classes)
        self.bbox_pred = nn.Linear(representation_size, num_classes * 4)
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

    def forward(self, x):
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        return scores, bbox_deltas


class MaskPostProcessor(nn.Module):
    """
    From the results of the CNN, post process the masks
    by taking the mask corresponding to the class with max
    probability (which are of fixed size and directly output
    by the CNN) and return the masks in the mask field of the BoxList.

    If a masker object is passed, it will additionally
    project the masks in the image according to the locations in boxes,
    """

    def __init__(self, masker=None):
        super(MaskPostProcessor, self).__init__()
        self.masker = masker

    def forward(self, x, boxes):
        """
        Arguments:
            x (Tensor): the mask logits
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for ech image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra field mask
        """
        mask_prob = x.sigmoid()
        num_masks = x.shape[0]
        labels = [bbox.get_field('labels') for bbox in boxes]
        labels = torch.cat(labels)
        index = torch.arange(num_masks, device=labels.device)
        mask_prob = mask_prob[index, labels][:, (None)]
        boxes_per_image = [len(box) for box in boxes]
        mask_prob = mask_prob.split(boxes_per_image, dim=0)
        if self.masker:
            mask_prob = self.masker(mask_prob, boxes)
        results = []
        for prob, box in zip(mask_prob, boxes):
            bbox = BoxList(box.bbox, box.size, mode='xyxy')
            for field in box.fields():
                bbox.add_field(field, box.get_field(field))
            bbox.add_field('mask', prob)
            results.append(bbox)
        return results


def project_masks_on_boxes(segmentation_masks, proposals, discretization_size):
    """
    Given segmentation masks and the bounding boxes corresponding
    to the location of the masks in the image, this function
    crops and resizes the masks in the position defined by the
    boxes. This prepares the masks for them to be fed to the
    loss computation as the targets.

    Arguments:
        segmentation_masks: an instance of SegmentationMask
        proposals: an instance of BoxList
    """
    masks = []
    M = discretization_size
    device = proposals.bbox.device
    proposals = proposals.convert('xyxy')
    assert segmentation_masks.size == proposals.size, '{}, {}'.format(
        segmentation_masks, proposals)
    proposals = proposals.bbox.to(torch.device('cpu'))
    for segmentation_mask, proposal in zip(segmentation_masks, proposals):
        cropped_mask = segmentation_mask.crop(proposal)
        scaled_mask = cropped_mask.resize((M, M))
        mask = scaled_mask.convert(mode='mask')
        masks.append(mask)
    if len(masks) == 0:
        return torch.empty(0, dtype=torch.float32, device=device)
    return torch.stack(masks, dim=0).to(device, dtype=torch.float32)


class MaskRCNNLossComputation(object):

    def __init__(self, proposal_matcher, discretization_size):
        """
        Arguments:
            proposal_matcher (Matcher)
            discretization_size (int)
        """
        self.proposal_matcher = proposal_matcher
        self.discretization_size = discretization_size

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        target = target.copy_with_fields(['labels', 'masks'])
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field('matched_idxs', matched_idxs)
        return matched_targets

    def prepare_targets(self, proposals, targets):
        labels = []
        masks = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image)
            matched_idxs = matched_targets.get_field('matched_idxs')
            labels_per_image = matched_targets.get_field('labels')
            labels_per_image = labels_per_image.to(dtype=torch.int64)
            neg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[neg_inds] = 0
            positive_inds = torch.nonzero(labels_per_image > 0).squeeze(1)
            segmentation_masks = matched_targets.get_field('masks')
            segmentation_masks = segmentation_masks[positive_inds]
            positive_proposals = proposals_per_image[positive_inds]
            masks_per_image = project_masks_on_boxes(segmentation_masks,
                positive_proposals, self.discretization_size)
            labels.append(labels_per_image)
            masks.append(masks_per_image)
        return labels, masks

    def __call__(self, proposals, mask_logits, targets):
        """
        Arguments:
            proposals (list[BoxList])
            mask_logits (Tensor)
            targets (list[BoxList])

        Return:
            mask_loss (Tensor): scalar tensor containing the loss
        """
        labels, mask_targets = self.prepare_targets(proposals, targets)
        labels = cat(labels, dim=0)
        mask_targets = cat(mask_targets, dim=0)
        positive_inds = torch.nonzero(labels > 0).squeeze(1)
        labels_pos = labels[positive_inds]
        if mask_targets.numel() == 0:
            return mask_logits.sum() * 0
        mask_loss = F.binary_cross_entropy_with_logits(mask_logits[
            positive_inds, labels_pos], mask_targets)
        return mask_loss


def make_roi_mask_loss_evaluator(cfg):
    matcher = Matcher(cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD, cfg.MODEL.
        ROI_HEADS.BG_IOU_THRESHOLD, allow_low_quality_matches=False)
    loss_evaluator = MaskRCNNLossComputation(matcher, cfg.MODEL.
        ROI_MASK_HEAD.RESOLUTION)
    return loss_evaluator


def expand_masks(mask, padding):
    N = mask.shape[0]
    M = mask.shape[-1]
    pad2 = 2 * padding
    scale = float(M + pad2) / M
    padded_mask = mask.new_zeros((N, 1, M + pad2, M + pad2))
    padded_mask[:, :, padding:-padding, padding:-padding] = mask
    return padded_mask, scale


def expand_boxes(boxes, scale):
    w_half = (boxes[:, (2)] - boxes[:, (0)]) * 0.5
    h_half = (boxes[:, (3)] - boxes[:, (1)]) * 0.5
    x_c = (boxes[:, (2)] + boxes[:, (0)]) * 0.5
    y_c = (boxes[:, (3)] + boxes[:, (1)]) * 0.5
    w_half *= scale
    h_half *= scale
    boxes_exp = torch.zeros_like(boxes)
    boxes_exp[:, (0)] = x_c - w_half
    boxes_exp[:, (2)] = x_c + w_half
    boxes_exp[:, (1)] = y_c - h_half
    boxes_exp[:, (3)] = y_c + h_half
    return boxes_exp


def paste_mask_in_image(mask, box, im_h, im_w, thresh=0.5, padding=1):
    padded_mask, scale = expand_masks(mask[None], padding=padding)
    mask = padded_mask[0, 0]
    box = expand_boxes(box[None], scale)[0]
    box = box.to(dtype=torch.int32)
    TO_REMOVE = 1
    w = int(box[2] - box[0] + TO_REMOVE)
    h = int(box[3] - box[1] + TO_REMOVE)
    w = max(w, 1)
    h = max(h, 1)
    mask = mask.expand((1, 1, -1, -1))
    mask = mask.to(torch.float32)
    mask = F.interpolate(mask, size=(h, w), mode='bilinear', align_corners=
        False)
    mask = mask[0][0]
    if thresh >= 0:
        mask = mask > thresh
    else:
        mask = (mask * 255).to(torch.uint8)
    im_mask = torch.zeros((im_h, im_w), dtype=torch.uint8)
    x_0 = max(box[0], 0)
    x_1 = min(box[2] + 1, im_w)
    y_0 = max(box[1], 0)
    y_1 = min(box[3] + 1, im_h)
    im_mask[y_0:y_1, x_0:x_1] = mask[y_0 - box[1]:y_1 - box[1], x_0 - box[0
        ]:x_1 - box[0]]
    return im_mask


class Masker(object):
    """
    Projects a set of masks in an image on the locations
    specified by the bounding boxes
    """

    def __init__(self, threshold=0.5, padding=1):
        self.threshold = threshold
        self.padding = padding

    def forward_single_image(self, masks, boxes):
        boxes = boxes.convert('xyxy')
        im_w, im_h = boxes.size
        res = [paste_mask_in_image(mask, box, im_h, im_w, self.threshold,
            self.padding) for mask, box in zip(masks, boxes.bbox)]
        if len(res) > 0:
            res = torch.stack(res, dim=0)[:, (None)]
        else:
            res = masks.new_empty((0, 1, masks.shape[-2], masks.shape[-1]))
        return res

    def __call__(self, masks, boxes):
        if isinstance(boxes, BoxList):
            boxes = [boxes]
        assert len(boxes) == len(masks
            ), 'Masks and boxes should have the same length.'
        results = []
        for mask, box in zip(masks, boxes):
            assert mask.shape[0] == len(box
                ), 'Number of objects should be the same.'
            result = self.forward_single_image(mask, box)
            results.append(result)
        return results


def make_roi_mask_post_processor(cfg):
    if cfg.MODEL.ROI_MASK_HEAD.POSTPROCESS_MASKS:
        mask_threshold = cfg.MODEL.ROI_MASK_HEAD.POSTPROCESS_MASKS_THRESHOLD
        masker = Masker(threshold=mask_threshold, padding=1)
    else:
        masker = None
    mask_post_processor = MaskPostProcessor(masker)
    return mask_post_processor


def make_conv3x3(in_channels, out_channels, dilation=1, stride=1, use_gn=
    False, use_relu=False, kaiming_init=True):
    conv = Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
        padding=dilation, dilation=dilation, bias=False if use_gn else True)
    if kaiming_init:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity=
            'relu')
    else:
        torch.nn.init.normal_(conv.weight, std=0.01)
    if not use_gn:
        nn.init.constant_(conv.bias, 0)
    module = [conv]
    if use_gn:
        module.append(group_norm(out_channels))
    if use_relu:
        module.append(nn.ReLU(inplace=True))
    if len(module) > 1:
        return nn.Sequential(*module)
    return conv


class MaskRCNNFPNFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg):
        """
        Arguments:
            num_classes (int): number of output classes
            input_size (int): number of channels of the input once it's flattened
            representation_size (int): size of the intermediate representation
        """
        super(MaskRCNNFPNFeatureExtractor, self).__init__()
        resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_MASK_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(output_size=(resolution, resolution), scales=scales,
            sampling_ratio=sampling_ratio)
        input_size = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.pooler = pooler
        use_gn = cfg.MODEL.ROI_MASK_HEAD.USE_GN
        layers = cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS
        dilation = cfg.MODEL.ROI_MASK_HEAD.DILATION
        next_feature = input_size
        self.blocks = []
        for layer_idx, layer_features in enumerate(layers, 1):
            layer_name = 'mask_fcn{}'.format(layer_idx)
            module = make_conv3x3(next_feature, layer_features, dilation=
                dilation, stride=1, use_gn=use_gn)
            self.add_module(layer_name, module)
            next_feature = layer_features
            self.blocks.append(layer_name)

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        for layer_name in self.blocks:
            x = F.relu(getattr(self, layer_name)(x))
        return x


class MaskRCNNC4Predictor(nn.Module):

    def __init__(self, cfg):
        super(MaskRCNNC4Predictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        dim_reduced = cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS[-1]
        if cfg.MODEL.ROI_HEADS.USE_FPN:
            num_inputs = dim_reduced
        else:
            stage_index = 4
            stage2_relative_factor = 2 ** (stage_index - 1)
            res2_out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
            num_inputs = res2_out_channels * stage2_relative_factor
        self.conv5_mask = ConvTranspose2d(num_inputs, dim_reduced, 2, 2, 0)
        self.mask_fcn_logits = Conv2d(dim_reduced, num_classes, 1, 1, 0)
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity
                    ='relu')

    def forward(self, x):
        x = F.relu(self.conv5_mask(x))
        return self.mask_fcn_logits(x)


class CombinedROIHeads(torch.nn.ModuleDict):
    """
    Combines a set of individual heads (for box prediction or masks) into a single
    head.
    """

    def __init__(self, cfg, heads):
        super(CombinedROIHeads, self).__init__(heads)
        self.cfg = cfg.clone()
        if (cfg.MODEL.MASK_ON and cfg.MODEL.ROI_MASK_HEAD.
            SHARE_BOX_FEATURE_EXTRACTOR):
            self.mask.feature_extractor = self.box.feature_extractor

    def forward(self, features, proposals, targets=None):
        losses = {}
        x, detections, loss_box = self.box(features, proposals, targets)
        losses.update(loss_box)
        if self.cfg.MODEL.MASK_ON:
            mask_features = features
            if (self.training and self.cfg.MODEL.ROI_MASK_HEAD.
                SHARE_BOX_FEATURE_EXTRACTOR):
                mask_features = x
            x, detections, loss_mask = self.mask(mask_features, detections,
                targets)
            losses.update(loss_mask)
        return x, detections, losses


class BufferList(nn.Module):
    """
    Similar to nn.ParameterList, but for buffers
    """

    def __init__(self, buffers=None):
        super(BufferList, self).__init__()
        if buffers is not None:
            self.extend(buffers)

    def extend(self, buffers):
        offset = len(self)
        for i, buffer in enumerate(buffers):
            self.register_buffer(str(offset + i), buffer)
        return self

    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return iter(self._buffers.values())


def _whctrs(anchor):
    """Return width, height, x center, and y center for an anchor (window)."""
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr


def _mkanchors(ws, hs, x_ctr, y_ctr):
    """Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """
    ws = ws[:, (np.newaxis)]
    hs = hs[:, (np.newaxis)]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1), y_ctr - 0.5 * (hs - 1), 
        x_ctr + 0.5 * (ws - 1), y_ctr + 0.5 * (hs - 1)))
    return anchors


def _ratio_enum(anchor, ratios):
    """Enumerate a set of anchors for each aspect ratio wrt an anchor."""
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def _scale_enum(anchor, scales):
    """Enumerate a set of anchors for each scale wrt an anchor."""
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def _generate_anchors(base_size, scales, aspect_ratios):
    """Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, base_size - 1, base_size - 1) window.
    """
    anchor = np.array([1, 1, base_size, base_size], dtype=np.float) - 1
    anchors = _ratio_enum(anchor, aspect_ratios)
    anchors = np.vstack([_scale_enum(anchors[(i), :], scales) for i in
        range(anchors.shape[0])])
    return torch.from_numpy(anchors)


def generate_anchors(stride=16, sizes=(32, 64, 128, 256, 512),
    aspect_ratios=(0.5, 1, 2)):
    """Generates a matrix of anchor boxes in (x1, y1, x2, y2) format. Anchors
    are centered on stride / 2, have (approximate) sqrt areas of the specified
    sizes, and aspect ratios as given.
    """
    return _generate_anchors(stride, np.array(sizes, dtype=np.float) /
        stride, np.array(aspect_ratios, dtype=np.float))


class AnchorGenerator(nn.Module):
    """
    For a set of image sizes and feature maps, computes a set
    of anchors
    """

    def __init__(self, sizes=(128, 256, 512), aspect_ratios=(0.5, 1.0, 2.0),
        anchor_strides=(8, 16, 32), straddle_thresh=0):
        super(AnchorGenerator, self).__init__()
        if len(anchor_strides) == 1:
            anchor_stride = anchor_strides[0]
            cell_anchors = [generate_anchors(anchor_stride, sizes,
                aspect_ratios).float()]
        else:
            if len(anchor_strides) != len(sizes):
                raise RuntimeError('FPN should have #anchor_strides == #sizes')
            cell_anchors = [generate_anchors(anchor_stride, size if type(
                size) is tuple else (size,), aspect_ratios).float() for 
                anchor_stride, size in zip(anchor_strides, sizes)]
        self.strides = anchor_strides
        self.cell_anchors = BufferList(cell_anchors)
        self.straddle_thresh = straddle_thresh

    def num_anchors_per_location(self):
        return [len(cell_anchors) for cell_anchors in self.cell_anchors]

    def grid_anchors(self, grid_sizes):
        anchors = []
        for size, stride, base_anchors in zip(grid_sizes, self.strides,
            self.cell_anchors):
            grid_height, grid_width = size
            device = base_anchors.device
            shifts_x = torch.arange(0, grid_width * stride, step=stride,
                dtype=torch.float32, device=device)
            shifts_y = torch.arange(0, grid_height * stride, step=stride,
                dtype=torch.float32, device=device)
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)
            anchors.append((shifts.view(-1, 1, 4) + base_anchors.view(1, -1,
                4)).reshape(-1, 4))
        return anchors

    def add_visibility_to(self, boxlist):
        image_width, image_height = boxlist.size
        anchors = boxlist.bbox
        if self.straddle_thresh >= 0:
            inds_inside = (anchors[..., 0] >= -self.straddle_thresh) & (anchors
                [..., 1] >= -self.straddle_thresh) & (anchors[..., 2] < 
                image_width + self.straddle_thresh) & (anchors[..., 3] < 
                image_height + self.straddle_thresh)
        else:
            device = anchors.device
            inds_inside = torch.ones(anchors.shape[0], dtype=torch.uint8,
                device=device)
        boxlist.add_field('visibility', inds_inside)

    def forward(self, image_list, feature_maps):
        grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]
        anchors_over_all_feature_maps = self.grid_anchors(grid_sizes)
        anchors = []
        for _, (image_height, image_width) in enumerate(image_list.image_sizes
            ):
            anchors_in_image = []
            for anchors_per_feature_map in anchors_over_all_feature_maps:
                boxlist = BoxList(anchors_per_feature_map, (image_width,
                    image_height), mode='xyxy')
                self.add_visibility_to(boxlist)
                anchors_in_image.append(boxlist)
            anchors.append(anchors_in_image)
        return anchors

    def get_anchors(self, mini_batch_size, img_size, grid_sizes):
        """
        Arguments:
            mini_batch_size (int): number of images on one GPU
            img_size (int): weight and height of image
            grid_sizes (list[feature_map size]):
        """
        anchors_over_all_feature_maps = self.grid_anchors(grid_sizes)
        anchors = []
        for i in range(mini_batch_size):
            image_height, image_width = img_size, img_size
            anchors_in_image = []
            for anchors_per_feature_map in anchors_over_all_feature_maps:
                boxlist = BoxList(anchors_per_feature_map, (image_width,
                    image_height), mode='xyxy')
                self.add_visibility_to(boxlist)
                anchors_in_image.append(boxlist)
            anchors.append(anchors_in_image)
        return anchors


class DenseBoxHead(torch.nn.Module):

    def __init__(self, cfg):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(DenseBoxHead, self).__init__()
        num_classes = cfg.MODEL.RETINANET.NUM_CLASSES - 1
        in_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        cls_tower = []
        bbox_tower = []
        for i in range(cfg.MODEL.RETINANET.NUM_CONVS):
            cls_tower.append(nn.Conv2d(in_channels, in_channels,
                kernel_size=3, stride=1, padding=1))
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU())
            bbox_tower.append(nn.Conv2d(in_channels, in_channels,
                kernel_size=3, stride=1, padding=1))
            bbox_tower.append(nn.GroupNorm(32, in_channels))
            bbox_tower.append(nn.ReLU())
        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        self.cls_logits = nn.Conv2d(in_channels, num_classes, kernel_size=3,
            stride=1, padding=1)
        self.bbox_pred = nn.Conv2d(in_channels, 4, kernel_size=3, stride=1,
            padding=1)
        self.centerness = nn.Conv2d(in_channels, 1, kernel_size=3, stride=1,
            padding=1)
        for modules in [self.cls_tower, self.bbox_tower, self.cls_logits,
            self.bbox_pred, self.centerness]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)
        prior_prob = cfg.MODEL.RETINANET.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)
        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])

    def forward(self, x):
        logits = []
        bbox_reg = []
        centerness = []
        for l, feature in enumerate(x):
            cls_tower = self.cls_tower(feature)
            logits.append(self.cls_logits(cls_tower))
            centerness.append(self.centerness(cls_tower))
            bbox_reg.append(torch.exp(self.scales[l](self.bbox_pred(self.
                bbox_tower(feature)))))
        return logits, bbox_reg, centerness


class RetinaNetLossComputation(object):
    """
    This class computes the RetinaNet loss.
    """

    def __init__(self, cfg, proposal_matcher, box_coder):
        """
        Arguments:
            proposal_matcher (Matcher) 
                a tensor of size N containing the index of the gt element m the 
                matches to prediction n. If there is no match, a negative value is returned
            box_coder (BoxCoder)
        """
        self.proposal_matcher = proposal_matcher
        self.box_coder = box_coder
        self.num_classes = cfg.MODEL.RETINANET.NUM_CLASSES - 1
        self.box_cls_loss_func = SigmoidFocalLoss(cfg.MODEL.RETINANET.
            LOSS_GAMMA, cfg.MODEL.RETINANET.LOSS_ALPHA)
        self.bbox_reg_weight = cfg.MODEL.RETINANET.BBOX_REG_WEIGHT
        self.bbox_reg_beta = cfg.MODEL.RETINANET.BBOX_REG_BETA
        self.weight = cfg.MODEL.PANOPTIC.DET_WEIGHT

    def match_targets_to_anchors(self, anchor, target):
        match_quality_matrix = boxlist_iou(target, anchor)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        target = target.copy_with_fields(['labels'])
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field('matched_idxs', matched_idxs)
        return matched_targets

    def prepare_targets(self, anchors, targets):
        labels = []
        regression_targets = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            if len(targets_per_image) <= 0:
                device = anchors_per_image.bbox.device
                dummy_labels = torch.zeros(len(anchors_per_image), dtype=
                    torch.float32, device=device)
                dummy_regression = torch.zeros((len(anchors_per_image), 4),
                    dtype=torch.float32, device=device)
                labels.append(dummy_labels)
                regression_targets.append(dummy_regression)
                continue
            matched_targets = self.match_targets_to_anchors(anchors_per_image,
                targets_per_image)
            matched_idxs = matched_targets.get_field('matched_idxs')
            labels_per_image = matched_targets.get_field('labels').clone()
            bg_indices = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_indices] = 0
            inds_to_discard = matched_idxs == Matcher.BETWEEN_THRESHOLDS
            labels_per_image[inds_to_discard] = -1
            labels_per_image = labels_per_image.to(dtype=torch.float32)
            regression_targets_per_image = self.box_coder.encode(
                matched_targets.bbox, anchors_per_image.bbox)
            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)
        return labels, regression_targets

    def __call__(self, anchors, box_cls, box_regression, targets):
        """
        Arguments:
            anchors (list[BoxList])
            box_cls (list[Tensor])
            box_regression (list[Tensor])
            targets (list[BoxList])

        Returns:
            retinanet_cls_loss (Tensor)
            retinanet_regression_loss (Tensor)
        """
        if isinstance(targets, dict):
            labels = targets['labels']
            regression_targets = targets['regression_targets']
        else:
            anchors = [cat_boxlist(anchors_per_image) for anchors_per_image in
                anchors]
            labels, regression_targets = self.prepare_targets(anchors, targets)
        num_layers = len(box_cls)
        box_cls_flattened = []
        box_regression_flattened = []
        for box_cls_per_level, box_regression_per_level in zip(box_cls,
            box_regression):
            N, A, H, W = box_cls_per_level.shape
            C = self.num_classes
            box_cls_per_level = box_cls_per_level.view(N, -1, C, H, W)
            box_cls_per_level = box_cls_per_level.permute(0, 3, 4, 1, 2)
            box_cls_per_level = box_cls_per_level.reshape(N, -1, C)
            box_regression_per_level = box_regression_per_level.view(N, -1,
                4, H, W)
            box_regression_per_level = box_regression_per_level.permute(0, 
                3, 4, 1, 2)
            box_regression_per_level = box_regression_per_level.reshape(N, 
                -1, 4)
            box_cls_flattened.append(box_cls_per_level)
            box_regression_flattened.append(box_regression_per_level)
        box_cls = cat(box_cls_flattened, dim=1).reshape(-1, C)
        box_regression = cat(box_regression_flattened, dim=1).reshape(-1, 4)
        if not isinstance(targets, dict):
            labels = torch.cat(labels, dim=0)
            regression_targets = torch.cat(regression_targets, dim=0)
        pos_inds = labels > 0
        retinanet_regression_loss = smooth_l1_loss(box_regression[pos_inds],
            regression_targets[pos_inds], beta=self.bbox_reg_beta,
            size_average=False) / (pos_inds.sum() * 4)
        retinanet_regression_loss *= self.bbox_reg_weight
        labels = labels.int()
        retinanet_cls_loss = self.box_cls_loss_func(box_cls, labels) / ((
            labels > 0).sum() + N)
        return (retinanet_cls_loss * self.weight, retinanet_regression_loss *
            self.weight)


def make_retinanet_loss_evaluator(cfg, box_coder):
    matcher = Matcher(cfg.MODEL.RETINANET.FG_IOU_THRESHOLD, cfg.MODEL.
        RETINANET.BG_IOU_THRESHOLD, allow_low_quality_matches=True)
    loss_evaluator = RetinaNetLossComputation(cfg, matcher, box_coder)
    return loss_evaluator


def make_retinanet_postprocessor(config, rpn_box_coder, is_train):
    pre_nms_thresh = config.MODEL.RETINANET.INFERENCE_TH
    pre_nms_top_n = config.MODEL.RETINANET.PRE_NMS_TOP_N
    nms_thresh = config.MODEL.RETINANET.NMS_TH
    fpn_post_nms_top_n = config.TEST.DETECTIONS_PER_IMG
    min_size = 0
    box_selector = RetinaNetPostProcessor(pre_nms_thresh=pre_nms_thresh,
        pre_nms_top_n=pre_nms_top_n, nms_thresh=nms_thresh,
        fpn_post_nms_top_n=fpn_post_nms_top_n, box_coder=rpn_box_coder,
        min_size=min_size)
    return box_selector


class DenseBoxModule(torch.nn.Module):
    """
    Module for RetinaNet computation. Takes feature maps from the backbone and RPN
    proposals and losses.
    """

    def __init__(self, cfg):
        super(DenseBoxModule, self).__init__()
        head = DenseBoxHead(cfg)
        box_selector_test = make_retinanet_postprocessor(cfg)
        loss_evaluator = make_retinanet_loss_evaluator(cfg)
        self.head = head
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator
        self.strides = cfg.MODEL.RETINANET.ANCHOR_STRIDES

    def forward(self, images, features, targets=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        box_cls, box_regression, centerness = self.head(features)
        if not isinstance(targets, dict):
            points = self.generate_points(features)
        else:
            points = None
        if self.training:
            return self._forward_train(points, box_cls, box_regression,
                centerness, targets)
        else:
            return self._forward_test(points, box_cls, box_regression,
                centerness, images.image_sizes)

    def _forward_train(self, points, box_cls, box_regression, centerness,
        targets):
        loss_box_cls, loss_box_reg, loss_reg_weights = self.loss_evaluator(
            points, box_cls, box_regression, centerness, targets)
        losses = {'loss_densebox_cls': loss_box_cls, 'loss_densebox_reg':
            loss_box_reg, 'loss_reg_weights': loss_reg_weights}
        return None, losses

    def _forward_test(self, points, box_cls, box_regression, centerness,
        image_sizes):
        boxes = self.box_selector_test(points, box_cls, box_regression,
            centerness, image_sizes)
        return boxes, {}

    def generate_points(self, features):
        points = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            points_per_level = self.generate_points_per_level(h, w, self.
                strides[level], feature.device)
            points.append(points_per_level)
        return points

    def generate_points_per_level(self, h, w, stride, device):
        shifts_x = torch.arange(0, w * stride, step=stride, dtype=torch.
            float32, device=device)
        shifts_y = torch.arange(0, h * stride, step=stride, dtype=torch.
            float32, device=device)
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        points = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        return points


def remove_small_boxes(boxlist, min_size):
    """
    Only keep boxes with both sides >= min_size

    Arguments:
        boxlist (Boxlist)
        min_size (int)
    """
    xywh_boxes = boxlist.convert('xywh').bbox
    _, _, ws, hs = xywh_boxes.unbind(dim=1)
    keep = ((ws >= min_size) & (hs >= min_size)).nonzero().squeeze(1)
    return boxlist[keep]


class RetinaNetPostProcessor(torch.nn.Module):
    """
    Performs post-processing on the outputs of the RetinaNet boxes.
    This is only used in the testing.
    """

    def __init__(self, pre_nms_thresh, pre_nms_top_n, nms_thresh,
        fpn_post_nms_top_n, min_size, num_classes):
        """
        Arguments:
            pre_nms_thresh (float)
            pre_nms_top_n (int)
            nms_thresh (float)
            fpn_post_nms_top_n (int)
            min_size (int)
            num_classes (int)
            box_coder (BoxCoder)
        """
        super(RetinaNetPostProcessor, self).__init__()
        self.pre_nms_thresh = pre_nms_thresh
        self.pre_nms_top_n = pre_nms_top_n
        self.nms_thresh = nms_thresh
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.min_size = min_size
        self.num_classes = num_classes

    def forward_for_single_feature_map(self, points, box_cls,
        box_regression, centerness, image_sizes):
        """
        Arguments:
            anchors: list[BoxList]
            box_cls: tensor of size N, A * C, H, W
            box_regression: tensor of size N, A * 4, H, W
        """
        N, C, H, W = box_cls.shape
        box_cls = box_cls.view(N, C, H, W).permute(0, 2, 3, 1)
        box_cls = box_cls.reshape(N, -1, C).sigmoid()
        box_regression = box_regression.view(N, 4, H, W).permute(0, 2, 3, 1)
        box_regression = box_regression.reshape(N, -1, 4)
        centerness = centerness.view(N, 1, H, W).permute(0, 2, 3, 1)
        centerness = centerness.reshape(N, -1).sigmoid()
        candidate_inds = box_cls > self.pre_nms_thresh
        pre_nms_top_n = candidate_inds.view(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)
        box_cls = box_cls * centerness[:, :, (None)]
        results = []
        for i in range(N):
            per_box_cls = box_cls[i]
            per_candidate_inds = candidate_inds[i]
            per_box_cls = per_box_cls[per_candidate_inds]
            per_candidate_nonzeros = per_candidate_inds.nonzero()
            per_box_loc = per_candidate_nonzeros[:, (0)]
            per_class = per_candidate_nonzeros[:, (1)] + 1
            per_box_regression = box_regression[i]
            per_box_regression = per_box_regression[per_box_loc]
            per_points = points[per_box_loc]
            per_pre_nms_top_n = pre_nms_top_n[i]
            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                per_box_cls, top_k_indices = per_box_cls.topk(per_pre_nms_top_n
                    , sorted=False)
                per_class = per_class[top_k_indices]
                per_box_regression = per_box_regression[top_k_indices]
                per_points = per_points[top_k_indices]
            detections = torch.stack([per_points[:, (0)] -
                per_box_regression[:, (0)], per_points[:, (1)] -
                per_box_regression[:, (1)], per_points[:, (0)] +
                per_box_regression[:, (2)], per_points[:, (1)] +
                per_box_regression[:, (3)]], dim=1)
            h, w = image_sizes[i]
            boxlist = BoxList(detections, (int(w), int(h)), mode='xyxy')
            boxlist.add_field('labels', per_class)
            boxlist.add_field('scores', per_box_cls)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = remove_small_boxes(boxlist, self.min_size)
            results.append(boxlist)
        return results

    def forward(self, points, box_cls, box_regression, centerness, image_sizes
        ):
        """
        Arguments:
            anchors: list[list[BoxList]]
            box_cls: list[tensor]
            box_regression: list[tensor]
            image_sizes: list[(h, w)]
        Returns:
            boxlists (list[BoxList]): the post-processed anchors, after
                applying box decoding and NMS
        """
        sampled_boxes = []
        for _, (p, o, b, r) in enumerate(zip(points, box_cls,
            box_regression, centerness)):
            sampled_boxes.append(self.forward_for_single_feature_map(p, o,
                b, r, image_sizes))
        boxlists = list(zip(*sampled_boxes))
        boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]
        boxlists = self.select_over_all_levels(boxlists)
        return boxlists

    def select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        results = []
        for i in range(num_images):
            scores = boxlists[i].get_field('scores')
            labels = boxlists[i].get_field('labels')
            boxes = boxlists[i].bbox
            boxlist = boxlists[i]
            result = []
            for j in range(1, self.num_classes):
                inds = (labels == j).nonzero().view(-1)
                scores_j = scores[inds]
                boxes_j = boxes[(inds), :].view(-1, 4)
                boxlist_for_class = BoxList(boxes_j, boxlist.size, mode='xyxy')
                boxlist_for_class.add_field('scores', scores_j)
                boxlist_for_class = boxlist_nms(boxlist_for_class, self.
                    nms_thresh, score_field='scores')
                num_labels = len(boxlist_for_class)
                boxlist_for_class.add_field('labels', torch.full((
                    num_labels,), j, dtype=torch.int64, device=scores.device))
                result.append(boxlist_for_class)
            result = cat_boxlist(result)
            number_of_detections = len(result)
            if number_of_detections > self.fpn_post_nms_top_n > 0:
                cls_scores = result.get_field('scores')
                image_thresh, _ = torch.kthvalue(cls_scores.cpu(), 
                    number_of_detections - self.fpn_post_nms_top_n + 1)
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]
            results.append(result)
        return results


class RPNPostProcessor(torch.nn.Module):
    """
    Performs post-processing on the outputs of the RPN boxes, before feeding the
    proposals to the heads
    """

    def __init__(self, pre_nms_top_n, post_nms_top_n, nms_thresh, min_size,
        box_coder=None, fpn_post_nms_top_n=None):
        """
        Arguments:
            pre_nms_top_n (int)
            post_nms_top_n (int)
            nms_thresh (float)
            min_size (int)
            box_coder (BoxCoder)
            fpn_post_nms_top_n (int)
        """
        super(RPNPostProcessor, self).__init__()
        self.pre_nms_top_n = pre_nms_top_n
        self.post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.min_size = min_size
        if box_coder is None:
            box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        self.box_coder = box_coder
        if fpn_post_nms_top_n is None:
            fpn_post_nms_top_n = post_nms_top_n
        self.fpn_post_nms_top_n = fpn_post_nms_top_n

    def add_gt_proposals(self, proposals, targets):
        """
        Arguments:
            proposals: list[BoxList]
            targets: list[BoxList]
        """
        device = proposals[0].bbox.device
        gt_boxes = [target.copy_with_fields([]) for target in targets]
        for gt_box in gt_boxes:
            gt_box.add_field('objectness', torch.ones(len(gt_box), device=
                device))
        proposals = [cat_boxlist((proposal, gt_box)) for proposal, gt_box in
            zip(proposals, gt_boxes)]
        return proposals

    def forward_for_single_feature_map(self, anchors, objectness,
        box_regression):
        """
        Arguments:
            anchors: list[BoxList]
            objectness: tensor of size N, A, H, W
            box_regression: tensor of size N, A * 4, H, W
        """
        device = objectness.device
        N, A, H, W = objectness.shape
        objectness = objectness.permute(0, 2, 3, 1).reshape(N, -1)
        objectness = objectness.sigmoid()
        box_regression = box_regression.view(N, -1, 4, H, W).permute(0, 3, 
            4, 1, 2)
        box_regression = box_regression.reshape(N, -1, 4)
        num_anchors = A * H * W
        pre_nms_top_n = min(self.pre_nms_top_n, num_anchors)
        objectness, topk_idx = objectness.topk(pre_nms_top_n, dim=1, sorted
            =True)
        batch_idx = torch.arange(N, device=device)[:, (None)]
        box_regression = box_regression[batch_idx, topk_idx]
        image_shapes = [box.size for box in anchors]
        concat_anchors = torch.cat([a.bbox for a in anchors], dim=0)
        concat_anchors = concat_anchors.reshape(N, -1, 4)[batch_idx, topk_idx]
        proposals = self.box_coder.decode(box_regression.view(-1, 4),
            concat_anchors.view(-1, 4))
        proposals = proposals.view(N, -1, 4)
        result = []
        for proposal, score, im_shape in zip(proposals, objectness,
            image_shapes):
            boxlist = BoxList(proposal, im_shape, mode='xyxy')
            boxlist.add_field('objectness', score)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = remove_small_boxes(boxlist, self.min_size)
            boxlist = boxlist_nms(boxlist, self.nms_thresh, max_proposals=
                self.post_nms_top_n, score_field='objectness')
            result.append(boxlist)
        return result

    def forward(self, anchors, objectness, box_regression, targets=None):
        """
        Arguments:
            anchors: list[list[BoxList]]
            objectness: list[tensor]
            box_regression: list[tensor]

        Returns:
            boxlists (list[BoxList]): the post-processed anchors, after
                applying box decoding and NMS
        """
        sampled_boxes = []
        num_levels = len(objectness)
        anchors = list(zip(*anchors))
        for a, o, b in zip(anchors, objectness, box_regression):
            sampled_boxes.append(self.forward_for_single_feature_map(a, o, b))
        boxlists = list(zip(*sampled_boxes))
        boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]
        if num_levels > 1:
            boxlists = self.select_over_all_levels(boxlists)
        if self.training and targets is not None:
            boxlists = self.add_gt_proposals(boxlists, targets)
        return boxlists

    def select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        if self.training:
            objectness = torch.cat([boxlist.get_field('objectness') for
                boxlist in boxlists], dim=0)
            box_sizes = [len(boxlist) for boxlist in boxlists]
            post_nms_top_n = min(self.fpn_post_nms_top_n, len(objectness))
            _, inds_sorted = torch.topk(objectness, post_nms_top_n, dim=0,
                sorted=True)
            inds_mask = torch.zeros_like(objectness, dtype=torch.uint8)
            inds_mask[inds_sorted] = 1
            inds_mask = inds_mask.split(box_sizes)
            for i in range(num_images):
                boxlists[i] = boxlists[i][inds_mask[i]]
        else:
            for i in range(num_images):
                objectness = boxlists[i].get_field('objectness')
                post_nms_top_n = min(self.fpn_post_nms_top_n, len(objectness))
                _, inds_sorted = torch.topk(objectness, post_nms_top_n, dim
                    =0, sorted=True)
                boxlists[i] = boxlists[i][inds_sorted]
        return boxlists


class NasHeadModule(torch.nn.Module):
    """
    Module for NAS Head computation. Take features from the backbone and 
    Decoder proposals and losses
    """

    def __init__(self, cfg):
        super(NasHeadModule, self).__init__()
        config_arch = cfg.SEARCH.DECODER.CONFIG
        decoder_layer_num = cfg.SEARCH.DECODER.NUM_CELLS
        head_repeats = cfg.SEARCH.HEAD.REPEATS
        share_weight_search = cfg.SEARCH.HEAD.WEIGHT_SHARE_SEARCH
        nas_decoder_on = cfg.SEARCH.NAS_DECODER_ON
        logger = logging.getLogger('maskrcnn_benchmark.nas_head')
        if nas_decoder_on:
            if share_weight_search:
                sample_weight_layer = config_arch[decoder_layer_num]
                sample_head_arch = config_arch[decoder_layer_num + 1]
                assert len(config_arch) == decoder_layer_num + 2
            else:
                sample_weight_layer = [0]
                sample_head_arch = config_arch[decoder_layer_num]
                assert len(config_arch) == decoder_layer_num + 1
        elif share_weight_search:
            sample_weight_layer = config_arch[0]
            sample_head_arch = config_arch[1]
            assert len(config_arch) == 2
        else:
            sample_weight_layer = [0]
            sample_head_arch = config_arch[0]
            assert len(config_arch) == 1
        assert len(sample_weight_layer) == 1
        logger.info('Share Weight Level: {}'.format(sample_weight_layer[0]))
        logger.info('Sample Head Arch : {}'.format(sample_head_arch))
        head = MicroHead_v2(sample_weight_layer[0], sample_head_arch,
            head_repeats, cfg)
        box_selector_test = make_retinanet_postprocessor(cfg)
        loss_evaluator = make_retinanet_loss_evaluator(cfg)
        self.head = head
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator
        self.strides = cfg.MODEL.RETINANET.ANCHOR_STRIDES
        self.dense_points = 1

    def forward(self, images, features, targets=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are 
                used for computing the predictions. Each tensor in the list 
                correspond to different feature levels
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During 
                testing, it is an empty dict.
        """
        box_cls, box_regression, centerness = self.head(features)
        if not isinstance(targets, dict):
            points = self.generate_points(features)
        else:
            points = None
        if self.training:
            return self._forward_train(points, box_cls, box_regression,
                centerness, targets)
        else:
            return self._forward_test(points, box_cls, box_regression,
                centerness, images.image_sizes)

    def _forward_train(self, points, box_cls, box_regression, centerness,
        targets):
        loss_box_cls, loss_box_reg, loss_reg_weights = self.loss_evaluator(
            points, box_cls, box_regression, centerness, targets)
        losses = {'loss_densebox_cls': loss_box_cls, 'loss_densebox_reg':
            loss_box_reg, 'loss_reg_weights': loss_reg_weights}
        return None, losses

    def _forward_test(self, points, box_cls, box_regression, centerness,
        image_sizes):
        boxes = self.box_selector_test(points, box_cls, box_regression,
            centerness, image_sizes)
        return boxes, {}

    def generate_points(self, features):
        points = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            points_per_level = self.generate_points_per_level(h, w, self.
                strides[level], feature.device)
            points.append(points_per_level)
        return points

    def generate_points_per_level(self, h, w, stride, device):
        shifts_x = torch.arange(0, w * stride, step=stride, dtype=torch.
            float32, device=device)
        shifts_y = torch.arange(0, h * stride, step=stride, dtype=torch.
            float32, device=device)
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        points = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        points = self.get_dense_locations(points, stride, device)
        return points

    def get_dense_locations(self, locations, stride, device):
        if self.dense_points <= 1:
            return locations
        center = 0
        step = stride // 4
        l_t = [center - step, center - step]
        r_t = [center + step, center - step]
        l_b = [center - step, center + step]
        r_b = [center + step, center + step]
        if self.dense_points == 4:
            points = torch.cuda.FloatTensor([l_t, r_t, l_b, r_b], device=device
                )
        elif self.dense_points == 5:
            points = torch.cuda.FloatTensor([l_t, r_t, [center, center],
                l_b, r_b], device=device)
        else:
            None
        points.reshape(1, -1, 2)
        locations = locations.reshape(-1, 1, 2).to(points)
        dense_locations = points + locations
        dense_locations = dense_locations.view(-1, 2)
        return dense_locations


def make_anchor_generator_retinanet(config):
    anchor_sizes = config.MODEL.RETINANET.ANCHOR_SIZES
    aspect_ratios = config.MODEL.RETINANET.ASPECT_RATIOS
    anchor_strides = config.MODEL.RETINANET.ANCHOR_STRIDES
    straddle_thresh = config.MODEL.RETINANET.STRADDLE_THRESH
    octave = config.MODEL.RETINANET.OCTAVE
    scales_per_octave = config.MODEL.RETINANET.SCALES_PER_OCTAVE
    assert len(anchor_strides) == len(anchor_sizes), 'Only support FPN now'
    new_anchor_sizes = []
    for size in anchor_sizes:
        per_layer_anchor_sizes = []
        for scale_per_octave in range(scales_per_octave):
            octave_scale = octave ** (scale_per_octave / float(
                scales_per_octave))
            per_layer_anchor_sizes.append(octave_scale * size)
        new_anchor_sizes.append(tuple(per_layer_anchor_sizes))
    anchor_generator = AnchorGenerator(tuple(new_anchor_sizes),
        aspect_ratios, anchor_strides, straddle_thresh)
    return anchor_generator


class RetinaNet_NasHeadModule(torch.nn.Module):
    """
    Module for NAS Head computation based on retina-net. Take features 
    from the backbone and Decoder proposals and losses
    """

    def __init__(self, cfg):
        super(RetinaNet_NasHeadModule, self).__init__()
        config_arch = cfg.SEARCH.DECODER.CONFIG
        decoder_layer_num = cfg.SEARCH.DECODER.NUM_CELLS
        head_repeats = cfg.SEARCH.HEAD.REPEATS
        share_weight_search = cfg.SEARCH.HEAD.WEIGHT_SHARE_SEARCH
        nas_decoder_on = cfg.SEARCH.NAS_DECODER_ON
        logger = logging.getLogger('maskrcnn_benchmark.nas_head')
        if nas_decoder_on:
            if share_weight_search:
                sample_weight_layer = config_arch[decoder_layer_num]
                sample_head_arch = config_arch[decoder_layer_num + 1]
                assert len(config_arch) == decoder_layer_num + 2
            else:
                sample_weight_layer = [0]
                sample_head_arch = config_arch[decoder_layer_num]
                assert len(config_arch) == decoder_layer_num + 1
        elif share_weight_search:
            sample_weight_layer = config_arch[0]
            sample_head_arch = config_arch[1]
            assert len(config_arch) == 2
        else:
            sample_weight_layer = [0]
            sample_head_arch = config_arch[0]
            assert len(config_arch) == 1
        assert len(sample_weight_layer) == 1
        logger.info('Share Weight Level: {}'.format(sample_weight_layer[0]))
        logger.info('Sample Head Arch : {}'.format(sample_head_arch))
        head = MicroHead_v2_retinanet(sample_weight_layer[0],
            sample_head_arch, head_repeats, cfg)
        anchor_generator = make_anchor_generator_retinanet(cfg)
        box_coder = BoxCoder(weights=(10.0, 10.0, 5.0, 5.0))
        box_selector_test = make_retinanet_postprocessor(cfg, box_coder,
            is_train=False)
        loss_evaluator = make_retinanet_loss_evaluator(cfg, box_coder)
        self.head = head
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator
        self.anchor_generator = anchor_generator
        self.anchors = None

    def forward(self, images, features, targets=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are 
                used for computing the predictions. Each tensor in the list 
                correspond to different feature levels
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During 
                testing, it is an empty dict.
        """
        box_cls, box_regression = self.head(features)
        if self.anchors is None and not isinstance(targets, dict):
            anchors = self.anchor_generator(images, features)
        elif self.anchors is None and isinstance(targets, dict):
            anchors = None
        else:
            anchors = self.anchors
        if self.training:
            return self._forward_train(anchors, box_cls, box_regression,
                targets)
        else:
            return self._forward_test(anchors, box_cls, box_regression)

    def generate_anchors(self, mini_batch_size, crop_size):
        grid_sizes = [(math.ceil(crop_size / r), math.ceil(crop_size / r)) for
            r in (8, 16, 32, 64, 128)]
        self.anchors = self.anchor_generator.get_anchors(mini_batch_size,
            crop_size, grid_sizes)

    def _forward_train(self, anchors, box_cls, box_regression, targets):
        loss_box_cls, loss_box_reg = self.loss_evaluator(anchors, box_cls,
            box_regression, targets)
        losses = {'loss_retina_cls': loss_box_cls, 'loss_retina_reg':
            loss_box_reg}
        return anchors, losses

    def _forward_test(self, anchors, box_cls, box_regression):
        boxes = self.box_selector_test(anchors, box_cls, box_regression)
        return boxes, {}


class RetinaNetPostProcessor(torch.nn.Module):
    """
    Performs post-processing on the outputs of the RetinaNet boxes.
    This is only used in the testing.
    """

    def __init__(self, pre_nms_thresh, pre_nms_top_n, nms_thresh,
        fpn_post_nms_top_n, min_size, box_coder=None):
        """
        Arguments:
            pre_nms_thresh (float)
            pre_nms_top_n (int)
            nms_thresh (float)
            fpn_post_nms_top_n (int)
            min_size (int)
            box_coder (BoxCoder)
        """
        super(RetinaNetPostProcessor, self).__init__()
        self.pre_nms_thresh = pre_nms_thresh
        self.pre_nms_top_n = pre_nms_top_n
        self.nms_thresh = nms_thresh
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.min_size = min_size
        if box_coder is None:
            box_coder = BoxCoder(weights=(10.0, 10.0, 5.0, 5.0))
        self.box_coder = box_coder

    def forward_for_single_feature_map(self, anchors, box_cls,
        box_regression, pre_nms_thresh=0.05):
        """
        Arguments:
            anchors: list[BoxList]
            box_cls: tensor of size N, A * C, H, W
            box_regression: tensor of size N, A * 4, H, W
        """
        device = box_cls.device
        N, _, H, W = box_cls.shape
        A = int(box_regression.size(1) / 4)
        C = int(box_cls.size(1) / A)
        box_cls = box_cls.view(N, -1, C, H, W).permute(0, 3, 4, 1, 2)
        box_cls = box_cls.reshape(N, -1, C)
        box_cls = box_cls.sigmoid()
        box_regression = box_regression.view(N, -1, 4, H, W)
        box_regression = box_regression.permute(0, 3, 4, 1, 2)
        box_regression = box_regression.reshape(N, -1, 4)
        num_anchors = A * H * W
        results = [[] for _ in range(N)]
        candidate_inds = box_cls > pre_nms_thresh
        if candidate_inds.sum().item() == 0:
            empty_boxlists = []
            for a in anchors:
                empty_boxlist = BoxList(torch.Tensor(0, 4).to(device), a.size)
                empty_boxlist.add_field('labels', torch.LongTensor([]).to(
                    device))
                empty_boxlist.add_field('scores', torch.Tensor([]).to(device))
                empty_boxlists.append(empty_boxlist)
            return empty_boxlists
        pre_nms_top_n = candidate_inds.view(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)
        for batch_idx, (per_box_cls, per_box_regression, per_pre_nms_top_n,
            per_candidate_inds, per_anchors) in enumerate(zip(box_cls,
            box_regression, pre_nms_top_n, candidate_inds, anchors)):
            per_box_cls = per_box_cls[per_candidate_inds]
            per_candidate_nonzeros = per_candidate_inds.nonzero()
            per_box_loc = per_candidate_nonzeros[:, (0)]
            per_class = per_candidate_nonzeros[:, (1)]
            per_class += 1
            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                per_box_cls, top_k_indices = per_box_cls.topk(per_pre_nms_top_n
                    , sorted=False)
                per_box_loc = per_box_loc[top_k_indices]
                per_class = per_class[top_k_indices]
            detections = self.box_coder.decode(per_box_regression[(
                per_box_loc), :].view(-1, 4), per_anchors.bbox[(per_box_loc
                ), :].view(-1, 4))
            boxlist = BoxList(detections, per_anchors.size, mode='xyxy')
            boxlist.add_field('labels', per_class)
            boxlist.add_field('scores', per_box_cls)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = remove_small_boxes(boxlist, self.min_size)
            results[batch_idx] = boxlist
        return results

    def forward(self, anchors, box_cls, box_regression, targets=None):
        """
        Arguments:
            anchors: list[list[BoxList]]
            box_cls: list[tensor]
            box_regression: list[tensor]

        Returns:
            boxlists (list[BoxList]): the post-processed anchors, after
                applying box decoding and NMS
        """
        sampled_boxes = []
        num_levels = len(box_cls)
        anchors = list(zip(*anchors))
        for l, (a, o, b) in enumerate(zip(anchors, box_cls, box_regression)):
            sampled_boxes.append(self.forward_for_single_feature_map(a, o,
                b, self.pre_nms_thresh))
        boxlists = list(zip(*sampled_boxes))
        boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]
        boxlists = self.select_over_all_levels(boxlists)
        return boxlists

    def select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        results = []
        for i in range(num_images):
            scores = boxlists[i].get_field('scores')
            labels = boxlists[i].get_field('labels')
            boxes = boxlists[i].bbox
            boxlist = boxlists[i]
            result = []
            for j in range(1, 81):
                inds = (labels == j).nonzero().view(-1)
                if len(inds) == 0:
                    continue
                scores_j = scores[inds]
                boxes_j = boxes[(inds), :].view(-1, 4)
                boxlist_for_class = BoxList(boxes_j, boxlist.size, mode='xyxy')
                boxlist_for_class.add_field('scores', scores_j)
                boxlist_for_class = boxlist_nms(boxlist_for_class, self.
                    nms_thresh, score_field='scores')
                num_labels = len(boxlist_for_class)
                boxlist_for_class.add_field('labels', torch.full((
                    num_labels,), j, dtype=torch.int64, device=scores.device))
                result.append(boxlist_for_class)
            if len(result):
                result = cat_boxlist(result)
                number_of_detections = len(result)
                if number_of_detections > self.fpn_post_nms_top_n > 0:
                    cls_scores = result.get_field('scores')
                    image_thresh, _ = torch.kthvalue(cls_scores.cpu(), 
                        number_of_detections - self.fpn_post_nms_top_n + 1)
                    keep = cls_scores >= image_thresh.item()
                    keep = torch.nonzero(keep).squeeze(1)
                    result = result[keep]
                results.append(result)
            else:
                device = boxlist.bbox.device
                empty_boxlist = BoxList(torch.zeros(1, 4).to(device),
                    boxlist.size)
                empty_boxlist.add_field('labels', torch.LongTensor([1]).to(
                    device))
                empty_boxlist.add_field('scores', torch.Tensor([0.01]).to(
                    device))
                results.append(empty_boxlist)
        return results


class RetinaNetHead(nn.Module):
    """
    Adds a RetinNet head with classification and regression heads
    """

    def __init__(self, cfg):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
            num_anchors (int): number of anchors to be predicted
        """
        super(RetinaNetHead, self).__init__()
        assert cfg.MODEL.BACKBONE.OUT_CHANNELS == cfg.SEARCH.DECODER.AGG_SIZE
        num_classes = cfg.MODEL.RETINANET.NUM_CLASSES - 1
        in_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        num_anchors = len(cfg.MODEL.RETINANET.ASPECT_RATIOS
            ) * cfg.MODEL.RETINANET.SCALES_PER_OCTAVE
        cls_tower = []
        bbox_tower = []
        for i in range(cfg.MODEL.RETINANET.NUM_CONVS):
            cls_tower.append(nn.Conv2d(in_channels, in_channels,
                kernel_size=3, stride=1, padding=1))
            cls_tower.append(nn.ReLU())
            bbox_tower.append(nn.Conv2d(in_channels, in_channels,
                kernel_size=3, stride=1, padding=1))
            bbox_tower.append(nn.ReLU())
        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        self.cls_logits = nn.Conv2d(in_channels, num_anchors * num_classes,
            kernel_size=3, stride=1, padding=1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4,
            kernel_size=3, stride=1, padding=1)
        for modules in [self.cls_tower, self.bbox_tower, self.cls_logits,
            self.bbox_pred]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    nn.init.normal_(l.weight, std=0.01)
                    nn.init.constant_(l.bias, 0)
        prior_prob = cfg.MODEL.RETINANET.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_logits.bias, bias_value)
        self.anchors = None

    def forward(self, x):
        logits = []
        bbox_reg = []
        for feature in x:
            logits.append(self.cls_logits(self.cls_tower(feature)))
            bbox_reg.append(self.bbox_pred(self.bbox_tower(feature)))
        return logits, bbox_reg


class RetinaNetModule(nn.Module):
    """
    Module for RetinaNet computation. Takes feature maps from the backbone and RPN
    proposals and losses.
    """

    def __init__(self, cfg):
        super(RetinaNetModule, self).__init__()
        self.cfg = cfg.clone()
        anchor_generator = make_anchor_generator_retinanet(cfg)
        head = RetinaNetHead(cfg)
        box_coder = BoxCoder(weights=(10.0, 10.0, 5.0, 5.0))
        box_selector_test = make_retinanet_postprocessor(cfg, box_coder,
            is_train=False)
        loss_evaluator = make_retinanet_loss_evaluator(cfg, box_coder)
        self.anchor_generator = anchor_generator
        self.head = head
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator
        self.anchors = None

    def forward(self, images, features, targets=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        box_cls, box_regression = self.head(features)
        if self.anchors is None and not isinstance(targets, dict):
            anchors = self.anchor_generator(images, features)
        elif self.anchors is None and isinstance(targets, dict):
            anchors = None
        else:
            anchors = self.anchors
        if self.training:
            return self._forward_train(anchors, box_cls, box_regression,
                targets)
        else:
            return self._forward_test(anchors, box_cls, box_regression)

    def generate_anchors(self, mini_batch_size, crop_size):
        grid_sizes = [(math.ceil(crop_size / r), math.ceil(crop_size / r)) for
            r in (8, 16, 32, 64, 128)]
        self.anchors = self.anchor_generator.get_anchors(mini_batch_size,
            crop_size, grid_sizes)

    def _forward_train(self, anchors, box_cls, box_regression, targets):
        loss_box_cls, loss_box_reg = self.loss_evaluator(anchors, box_cls,
            box_regression, targets)
        losses = {'loss_retina_cls': loss_box_cls, 'loss_retina_reg':
            loss_box_reg}
        return anchors, losses

    def _forward_test(self, anchors, box_cls, box_regression):
        boxes = self.box_selector_test(anchors, box_cls, box_regression)
        return boxes, {}


def make_anchor_generator(config):
    anchor_sizes = config.MODEL.RPN.ANCHOR_SIZES
    aspect_ratios = config.MODEL.RPN.ASPECT_RATIOS
    anchor_stride = config.MODEL.RPN.ANCHOR_STRIDE
    straddle_thresh = config.MODEL.RPN.STRADDLE_THRESH
    if config.MODEL.RPN.USE_FPN:
        assert len(anchor_stride) == len(anchor_sizes
            ), 'FPN should have len(ANCHOR_STRIDE) == len(ANCHOR_SIZES)'
    else:
        assert len(anchor_stride
            ) == 1, 'Non-FPN should have a single ANCHOR_STRIDE'
    anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios,
        anchor_stride, straddle_thresh)
    return anchor_generator


class RPNLossComputation(object):
    """
    This class computes the RPN loss.
    """

    def __init__(self, proposal_matcher, fg_bg_sampler, box_coder):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.box_coder = box_coder

    def match_targets_to_anchors(self, anchor, target):
        match_quality_matrix = boxlist_iou(target, anchor)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        target = target.copy_with_fields([])
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field('matched_idxs', matched_idxs)
        return matched_targets

    def prepare_targets(self, anchors, targets):
        labels = []
        regression_targets = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            matched_targets = self.match_targets_to_anchors(anchors_per_image,
                targets_per_image)
            matched_idxs = matched_targets.get_field('matched_idxs')
            labels_per_image = matched_idxs >= 0
            labels_per_image = labels_per_image.to(dtype=torch.float32)
            labels_per_image[~anchors_per_image.get_field('visibility')] = -1
            inds_to_discard = matched_idxs == Matcher.BETWEEN_THRESHOLDS
            labels_per_image[inds_to_discard] = -1
            regression_targets_per_image = self.box_coder.encode(
                matched_targets.bbox, anchors_per_image.bbox)
            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)
        return labels, regression_targets

    def __call__(self, anchors, objectness, box_regression, targets):
        """
        Arguments:
            anchors (list[BoxList])
            objectness (list[Tensor])
            box_regression (list[Tensor])
            targets (list[BoxList])

        Returns:
            objectness_loss (Tensor)
            box_loss (Tensor
        """
        anchors = [cat_boxlist(anchors_per_image) for anchors_per_image in
            anchors]
        labels, regression_targets = self.prepare_targets(anchors, targets)
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_pos_inds = torch.nonzero(torch.cat(sampled_pos_inds, dim=0)
            ).squeeze(1)
        sampled_neg_inds = torch.nonzero(torch.cat(sampled_neg_inds, dim=0)
            ).squeeze(1)
        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)
        objectness_flattened = []
        box_regression_flattened = []
        for objectness_per_level, box_regression_per_level in zip(objectness,
            box_regression):
            N, A, H, W = objectness_per_level.shape
            objectness_per_level = objectness_per_level.permute(0, 2, 3, 1
                ).reshape(N, -1)
            box_regression_per_level = box_regression_per_level.view(N, -1,
                4, H, W)
            box_regression_per_level = box_regression_per_level.permute(0, 
                3, 4, 1, 2)
            box_regression_per_level = box_regression_per_level.reshape(N, 
                -1, 4)
            objectness_flattened.append(objectness_per_level)
            box_regression_flattened.append(box_regression_per_level)
        objectness = cat(objectness_flattened, dim=1).reshape(-1)
        box_regression = cat(box_regression_flattened, dim=1).reshape(-1, 4)
        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)
        box_loss = smooth_l1_loss(box_regression[sampled_pos_inds],
            regression_targets[sampled_pos_inds], beta=1.0 / 9,
            size_average=False) / sampled_inds.numel()
        objectness_loss = F.binary_cross_entropy_with_logits(objectness[
            sampled_inds], labels[sampled_inds])
        return objectness_loss, box_loss


def make_rpn_loss_evaluator(cfg, box_coder):
    matcher = Matcher(cfg.MODEL.RPN.FG_IOU_THRESHOLD, cfg.MODEL.RPN.
        BG_IOU_THRESHOLD, allow_low_quality_matches=True)
    fg_bg_sampler = BalancedPositiveNegativeSampler(cfg.MODEL.RPN.
        BATCH_SIZE_PER_IMAGE, cfg.MODEL.RPN.POSITIVE_FRACTION)
    loss_evaluator = RPNLossComputation(matcher, fg_bg_sampler, box_coder)
    return loss_evaluator


def make_rpn_postprocessor(config, rpn_box_coder, is_train):
    fpn_post_nms_top_n = config.MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN
    if not is_train:
        fpn_post_nms_top_n = config.MODEL.RPN.FPN_POST_NMS_TOP_N_TEST
    pre_nms_top_n = config.MODEL.RPN.PRE_NMS_TOP_N_TRAIN
    post_nms_top_n = config.MODEL.RPN.POST_NMS_TOP_N_TRAIN
    if not is_train:
        pre_nms_top_n = config.MODEL.RPN.PRE_NMS_TOP_N_TEST
        post_nms_top_n = config.MODEL.RPN.POST_NMS_TOP_N_TEST
    nms_thresh = config.MODEL.RPN.NMS_THRESH
    min_size = config.MODEL.RPN.MIN_SIZE
    box_selector = RPNPostProcessor(pre_nms_top_n=pre_nms_top_n,
        post_nms_top_n=post_nms_top_n, nms_thresh=nms_thresh, min_size=
        min_size, box_coder=rpn_box_coder, fpn_post_nms_top_n=
        fpn_post_nms_top_n)
    return box_selector


class RPNModule(torch.nn.Module):
    """
    Module for RPN computation. Takes feature maps from the backbone and RPN
    proposals and losses. Works for both FPN and non-FPN.
    """

    def __init__(self, cfg):
        super(RPNModule, self).__init__()
        self.cfg = cfg.clone()
        anchor_generator = make_anchor_generator(cfg)
        in_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        rpn_head = registry.RPN_HEADS[cfg.MODEL.RPN.RPN_HEAD]
        head = rpn_head(cfg, in_channels, anchor_generator.
            num_anchors_per_location()[0])
        rpn_box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        box_selector_train = make_rpn_postprocessor(cfg, rpn_box_coder,
            is_train=True)
        box_selector_test = make_rpn_postprocessor(cfg, rpn_box_coder,
            is_train=False)
        loss_evaluator = make_rpn_loss_evaluator(cfg, rpn_box_coder)
        self.anchor_generator = anchor_generator
        self.head = head
        self.box_selector_train = box_selector_train
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator

    def forward(self, images, features, targets=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        objectness, rpn_box_regression = self.head(features)
        anchors = self.anchor_generator(images, features)
        if self.training:
            return self._forward_train(anchors, objectness,
                rpn_box_regression, targets)
        else:
            return self._forward_test(anchors, objectness, rpn_box_regression)

    def _forward_train(self, anchors, objectness, rpn_box_regression, targets):
        if self.cfg.MODEL.RPN_ONLY:
            boxes = anchors
        else:
            with torch.no_grad():
                boxes = self.box_selector_train(anchors, objectness,
                    rpn_box_regression, targets)
        loss_objectness, loss_rpn_box_reg = self.loss_evaluator(anchors,
            objectness, rpn_box_regression, targets)
        losses = {'loss_objectness': loss_objectness, 'loss_rpn_box_reg':
            loss_rpn_box_reg}
        return boxes, losses

    def _forward_test(self, anchors, objectness, rpn_box_regression):
        boxes = self.box_selector_test(anchors, objectness, rpn_box_regression)
        if self.cfg.MODEL.RPN_ONLY:
            inds = [box.get_field('objectness').sort(descending=True)[1] for
                box in boxes]
            boxes = [box[ind] for box, ind in zip(boxes, inds)]
        return boxes, {}


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


def conv_bn_relu(C_in, C_out, kernel_size, stride, padding, affine=True):
    return nn.Sequential(nn.Conv2d(C_in, C_out, kernel_size, stride=stride,
        padding=padding, bias=False), nn.BatchNorm2d(C_out, affine=affine),
        nn.ReLU(inplace=False))


class GAPConv1x1(nn.Module):
    """Global Average Pooling + conv1x1"""

    def __init__(self, C_in, C_out):
        super(GAPConv1x1, self).__init__()
        self.conv1x1 = conv_bn_relu(C_in, C_out, 1, stride=1, padding=0)

    def forward(self, x):
        size = x.size()[2:]
        out = x.mean(2, keepdim=True).mean(3, keepdim=True)
        out = self.conv1x1(out)
        out = nn.functional.interpolate(out, size=size, mode='bilinear',
            align_corners=False)
        return out


class DilConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation,
        affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(nn.Conv2d(C_in, C_in, kernel_size=
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=C_in, bias=False), nn.Conv2d(C_in, C_out, kernel_size=1,
            padding=0, bias=False), nn.BatchNorm2d(C_out, affine=affine),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation=
        1, affine=True, repeats=1):
        super(SepConv, self).__init__()
        if C_in != C_out:
            assert repeats == 1, 'SepConv with C_in != C_out must have only 1 repeat'
        basic_op = lambda : nn.Sequential(nn.Conv2d(C_in, C_in, kernel_size
            =kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=C_in, bias=False), nn.Conv2d(C_in, C_out, kernel_size=1,
            padding=0, bias=False), nn.BatchNorm2d(C_out, affine=affine),
            nn.ReLU(inplace=True))
        self.op = nn.Sequential()
        for idx in range(repeats):
            self.op.add_module('sep_{}'.format(idx), basic_op())

    def forward(self, x):
        return self.op(x)


class GN_SepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation=
        1, affine=True, repeats=1):
        super(GN_SepConv, self).__init__()
        if C_in != C_out:
            assert repeats == 1, 'SepConv with C_in != C_out must have only 1 repeat'
        basic_op = lambda : nn.Sequential(nn.Conv2d(C_in, C_in, kernel_size
            =kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=C_in, bias=False), nn.Conv2d(C_in, C_out, kernel_size=1,
            padding=0, bias=False), nn.GroupNorm(32, C_out), nn.ReLU(
            inplace=True))
        self.op = nn.Sequential()
        for idx in range(repeats):
            self.op.add_module('sep_{}'.format(idx), basic_op())

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.0)
        return x[:, :, ::self.stride, ::self.stride].mul(0.0)


class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0,
            bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0,
            bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


class GN_FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, affine=True):
        super(GN_FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0,
            bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0,
            bias=False)
        self.gn = nn.GroupNorm(32, C_out)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.gn(out)
        return out


class GN_DefConv(nn.Module):

    def __init__(self, C_in, C_out, ksize):
        super(GN_DefConv, self).__init__()
        self.dcn = nn.Sequential(DCN(C_in, C_out, ksize, stride=1, padding=
            ksize // 2, deformable_groups=2), nn.GroupNorm(32, C_out), nn.
            ReLU(inplace=True))

    def forward(self, x):
        return self.dcn(x)


def resize(x1, x2, largest=True):
    if largest:
        if x1.size()[2:] > x2.size()[2:]:
            x2 = nn.Upsample(size=x1.size()[2:], mode='bilinear')(x2)
        elif x1.size()[2:] < x2.size()[2:]:
            x1 = nn.Upsample(size=x2.size()[2:], mode='bilinear')(x1)
        return x1, x2
    else:
        raise NotImplementedError


class ParamSum(nn.Module):

    def __init__(self, C):
        super(ParamSum, self).__init__()
        self.a = nn.Parameter(torch.ones(C))
        self.b = nn.Parameter(torch.ones(C))

    def forward(self, x, y):
        bsize = x.size(0)
        x, y = resize(x, y)
        return self.a.expand(bsize, -1)[:, :, (None), (None)
            ] * x + self.b.expand(bsize, -1)[:, :, (None), (None)] * y


class ConcatReduce(nn.Module):

    def __init__(self, C, affine=True, repeats=1):
        super(ConcatReduce, self).__init__()
        self.conv1x1 = nn.Sequential(nn.BatchNorm2d(2 * C, affine=affine),
            nn.ReLU(inplace=False), nn.Conv2d(2 * C, C, 1, stride=1, groups
            =C, padding=0, bias=False))

    def forward(self, x, y):
        x, y = resize(x, y)
        z = torch.cat([x, y], 1)
        return self.conv1x1(z)


AGG_NAMES = ['psum', 'cat']


OPS = {'skip_connect': lambda C, stride, affine, repeats=1: Identity() if 
    stride == 1 else FactorizedReduce(C, C, affine=affine), 'sep_conv_3x3':
    lambda C, stride, affine, repeats=1: SepConv(C, C, 3, stride, 1, affine
    =affine, repeats=repeats), 'sep_conv_3x3_dil3': lambda C, stride,
    affine, repeats=1: SepConv(C, C, 3, stride, 3, affine=affine, dilation=
    3, repeats=repeats), 'sep_conv_5x5_dil6': lambda C, stride, affine,
    repeats=1: SepConv(C, C, 5, stride, 12, affine=affine, dilation=6,
    repeats=repeats), 'def_conv_3x3': lambda C, stride, affine, repeats=1:
    DefConv(C, C, 3)}


AGG_OPS = {'psum': lambda C, stride, affine, repeats=1: ParamSum(C), 'cat':
    lambda C, stride, affine, repeats=1: ConcatReduce(C, affine=affine,
    repeats=repeats)}


OP_NAMES = ['sep_conv_3x3', 'sep_conv_3x3_dil3', 'sep_conv_5x5_dil6',
    'skip_connect', 'def_conv_3x3']


class MicroDecoder_v2(nn.Module):

    def __init__(self, inp_sizes, config, agg_size=48, num_pools=4, repeats
        =1, top_blocks=None, inter=False):
        super(MicroDecoder_v2, self).__init__()
        inp_sizes = list(inp_sizes)
        for out_idx, size in enumerate(inp_sizes):
            setattr(self, 'adapt{}'.format(out_idx + 1), conv_bn_relu(size,
                agg_size, 1, 1, 0, affine=True))
            inp_sizes[out_idx] = agg_size
        init_pool = len(inp_sizes)
        self.inter = inter
        mult = 3 if self.inter else 1
        self._ops = nn.ModuleList()
        self._pos = []
        self._pos_stride = []
        self._collect_inds = []
        self._collect_inds_stride = []
        self._pools = ['l1', 'l2', 'l3', 'l4']
        self._pools_stride = [4, 8, 16, 32]
        for ind, cell in enumerate(config):
            _ops = nn.ModuleList()
            _pos = []
            l1, l2, op0, op1, op_agg = cell[0]
            for pos, op_id in zip([l1, l2], [op0, op1]):
                if pos in self._collect_inds:
                    pos_index = self._collect_inds.index(pos)
                    self._collect_inds.remove(pos)
                    self._collect_inds_stride.pop(pos_index)
                op_name = OP_NAMES[op_id]
                _ops.append(OPS[op_name](agg_size, 1, True, repeats=repeats))
                _pos.append(pos)
                self._pools.append('{}({})'.format(op_name, self._pools[pos]))
                self._pools_stride.append(self._pools_stride[pos])
            op_name = AGG_NAMES[op_agg]
            _ops.append(AGG_OPS[op_name](agg_size, 1, True, repeats=repeats))
            self._ops.append(_ops)
            self._pos.append(_pos)
            self._collect_inds.append(init_pool - 1 + (ind + 1) * mult)
            self._collect_inds_stride.append(min(self._pools_stride[l1],
                self._pools_stride[l2]))
            self._pools.append('TODO')
            self._pools_stride.append(min(self._pools_stride[l1], self.
                _pools_stride[l2]))
        self.info = ' + '.join(self._pools[i] for i in self._collect_inds)
        self.top_blocks = top_blocks

    def judge(self):
        """
        Judge whether the sampled arch can be used to train
        """
        metric = len(self._collect_inds)
        if metric < 3:
            return False
        else:
            return True

    def prettify(self, n_params):
        """ Encoder config: None
            Dec Config:
              ctx: (index, op) x 4
              conn: [index_1, index_2] x 3
        """
        header = '#PARAMS\n\n {:3.2f}M'.format(n_params / 1000000.0)
        conn_desc = '#Connections:\n' + self.info
        return header + '\n\n' + conn_desc

    def forward(self, x):
        results = []
        feats = []
        for idx, xx in enumerate(x):
            feats.append(getattr(self, 'adapt{}'.format(idx + 1))(xx))
        for pos, ops in zip(self._pos, self._ops):
            assert isinstance(pos, list), 'Must be list of pos'
            out0 = ops[0](feats[pos[0]])
            out1 = ops[1](feats[pos[1]])
            if self.inter:
                feats.append(out0)
                feats.append(out1)
            out2 = ops[2](out0, out1)
            feats.append(out2)
        unused_collect_inds = self._collect_inds[:-3]
        unused_collect_inds_stride = self._collect_inds_stride[:-3]
        for block_idx, i in enumerate(self._collect_inds[-3:]):
            feats_mid = feats[i]
            for unused_index in unused_collect_inds:
                feats_unused = feats[unused_index]
                feats_resize = F.interpolate(feats_unused, size=feats_mid.
                    size()[2:], mode='bilinear')
                feats_mid = feats_mid + feats_resize
            cell_out = F.interpolate(feats_mid, size=x[3 - block_idx].size(
                )[2:], mode='bilinear')
            results.insert(0, cell_out)
        if self.top_blocks is not None:
            last_results = self.top_blocks(x[-1], results[-1])
            results.extend(last_results)
        return results


def conv3x3(in_planes, out_planes, stride=1, groups=1, bias=False, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
        groups=groups, padding=dilation, dilation=dilation, bias=bias)


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
        padding=0, bias=bias)


HEAD_OPS = {'skip_connect': lambda C, stride, affine, repeats=1: Identity() if
    stride == 1 else GN_FactorizedReduce(C, C, affine=affine),
    'sep_conv_3x3': lambda C, stride, affine, repeats=1: GN_SepConv(C, C, 3,
    stride, 1, affine=affine, repeats=repeats), 'conv1x1': lambda C, stride,
    affine, repeats=1: nn.Sequential(conv1x1(C, C, stride=stride), nn.
    GroupNorm(32, C), nn.ReLU(inplace=False)), 'conv3x3': lambda C, stride,
    affine, repeats=1: nn.Sequential(conv3x3(C, C, stride=stride), nn.
    GroupNorm(32, C), nn.ReLU(inplace=False)), 'sep_conv_3x3_dil3': lambda
    C, stride, affine, repeats=1: GN_SepConv(C, C, 3, stride, 3, affine=
    affine, dilation=3, repeats=repeats), 'def_conv_3x3': lambda C, stride,
    affine, repeats=1: GN_DefConv(C, C, 3)}


HEAD_OP_NAMES = ['conv1x1', 'conv3x3', 'sep_conv_3x3', 'sep_conv_3x3_dil3',
    'skip_connect', 'def_conv_3x3']


class MicroHead_v2(nn.Module):
    """
    Simplified head arch which is used to search and construct single-stage
    detector head part
    """

    def __init__(self, share_weights_layer, head_config, repeats, cfg):
        """
        Arguments:
            head_config: head arch sampled by controller
            cfg: global setting info
        """
        super(MicroHead_v2, self).__init__()
        self.num_classes = cfg.MODEL.RETINANET.NUM_CLASSES - 1
        self.in_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.num_head_layers = cfg.SEARCH.HEAD.NUM_HEAD_LAYERS
        self.num_head = 5
        self.output_concat = cfg.SEARCH.HEAD.OUTPUT_CONCAT
        self.fpn_strides = cfg.MODEL.RETINANET.ANCHOR_STRIDES
        self.dense_points = 1
        self.norm_reg_targets = False
        self.centerness_on_reg = False
        self.share_weights_layer = share_weights_layer
        assert self.share_weights_layer >= 0
        assert self.share_weights_layer <= self.num_head_layers
        if self.share_weights_layer == 0:
            self.has_split_weights = False
        else:
            self.has_split_weights = True
        if self.share_weights_layer == self.num_head_layers:
            self.has_shared_weights = False
        else:
            self.has_shared_weights = True
        if self.has_split_weights:
            self._cls_head_split_ops = nn.ModuleList()
            self._reg_head_split_ops = nn.ModuleList()
            for ind in range(self.num_head):
                cls_empty_head_layer = nn.ModuleList()
                reg_empty_head_layer = nn.ModuleList()
                self._cls_head_split_ops.append(cls_empty_head_layer)
                self._reg_head_split_ops.append(reg_empty_head_layer)
        if self.has_shared_weights:
            self._cls_head_global_ops = nn.ModuleList()
            self._reg_head_global_ops = nn.ModuleList()
        agg_size = self.in_channels
        for ind, cell in enumerate(head_config):
            op_index = cell
            op_name = HEAD_OP_NAMES[op_index]
            _cls_ops = HEAD_OPS[op_name](agg_size, 1, True, repeats=repeats)
            _reg_ops = HEAD_OPS[op_name](agg_size, 1, True, repeats=repeats)
            if ind < self.share_weights_layer:
                for ind2 in range(self.num_head):
                    self._cls_head_split_ops[ind2].append(copy.deepcopy(
                        _cls_ops))
                    self._reg_head_split_ops[ind2].append(copy.deepcopy(
                        _reg_ops))
            else:
                self._cls_head_global_ops.append(_cls_ops)
                self._reg_head_global_ops.append(_reg_ops)
        final_channel = self.in_channels
        self.cls_logits = nn.Conv2d(final_channel, self.num_classes * self.
            dense_points, kernel_size=3, stride=1, padding=1)
        self.bbox_pred = nn.Conv2d(final_channel, 4 * self.dense_points,
            kernel_size=3, stride=1, padding=1)
        self.centerness = nn.Conv2d(final_channel, 1 * self.dense_points,
            kernel_size=3, stride=1, padding=1)
        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])
        for modules in [self.cls_logits, self.bbox_pred, self.centerness]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)
        prior_prob = cfg.MODEL.RETINANET.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

    def forward(self, x):
        logits = []
        bbox_reg = []
        centerness = []
        for l, feature in enumerate(x):
            cls_out = feature
            reg_out = feature
            if self.has_split_weights:
                for ops in self._cls_head_split_ops[l]:
                    cls_out = ops(cls_out)
                for ops in self._reg_head_split_ops[l]:
                    reg_out = ops(reg_out)
            if self.has_shared_weights:
                for ops in self._cls_head_global_ops:
                    cls_out = ops(cls_out)
                for ops in self._reg_head_global_ops:
                    reg_out = ops(reg_out)
            logits.append(self.cls_logits(cls_out))
            if self.centerness_on_reg:
                centerness.append(self.centerness(reg_out))
            else:
                centerness.append(self.centerness(cls_out))
            bbox_pred = self.scales[l](self.bbox_pred(reg_out))
            if self.norm_reg_targets:
                if self.training:
                    bbox_pred = F.relu(bbox_pred)
                    bbox_reg.append(bbox_pred)
                else:
                    bbox_reg.append(bbox_pred * self.fpn_strides[l])
            else:
                bbox_reg.append(torch.exp(bbox_pred))
        return logits, bbox_reg, centerness


class MicroHead_v2_retinanet(nn.Module):
    """
    Simplified head arch which is used to search and construct single-stage
    detector head part
    """

    def __init__(self, share_weights_layer, head_config, repeats, cfg):
        """
        Arguments:
            head_config: head arch sampled by controller
            cfg: global setting info
        """
        super(MicroHead_v2_retinanet, self).__init__()
        self.num_head = 5
        self.num_classes = cfg.MODEL.RETINANET.NUM_CLASSES - 1
        self.num_head_layers = cfg.SEARCH.HEAD.NUM_HEAD_LAYERS
        self.in_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.share_weights_layer = share_weights_layer
        self.num_anchors = len(cfg.MODEL.RETINANET.ASPECT_RATIOS
            ) * cfg.MODEL.RETINANET.SCALES_PER_OCTAVE
        assert self.share_weights_layer >= 0
        assert self.share_weights_layer <= self.num_head_layers
        if self.share_weights_layer == 0:
            self.has_split_weights = False
        else:
            self.has_split_weights = True
        if self.share_weights_layer == self.num_head_layers:
            self.has_shared_weights = False
        else:
            self.has_shared_weights = True
        if self.has_split_weights:
            self._cls_head_split_ops = nn.ModuleList()
            self._reg_head_split_ops = nn.ModuleList()
            for ind in range(self.num_head):
                cls_empty_head_layer = nn.ModuleList()
                reg_empty_head_layer = nn.ModuleList()
                self._cls_head_split_ops.append(cls_empty_head_layer)
                self._reg_head_split_ops.append(reg_empty_head_layer)
        if self.has_shared_weights:
            self._cls_head_global_ops = nn.ModuleList()
            self._reg_head_global_ops = nn.ModuleList()
        agg_size = self.in_channels
        for ind, cell in enumerate(head_config):
            op_index = cell
            op_name = HEAD_OP_NAMES[op_index]
            _cls_ops = HEAD_OPS[op_name](agg_size, 1, True, repeats=repeats)
            _reg_ops = HEAD_OPS[op_name](agg_size, 1, True, repeats=repeats)
            if ind < self.share_weights_layer:
                for ind2 in range(self.num_head):
                    self._cls_head_split_ops[ind2].append(copy.deepcopy(
                        _cls_ops))
                    self._reg_head_split_ops[ind2].append(copy.deepcopy(
                        _reg_ops))
            else:
                self._cls_head_global_ops.append(_cls_ops)
                self._reg_head_global_ops.append(_reg_ops)
        final_channel = self.in_channels
        self.cls_logits = nn.Conv2d(final_channel, self.num_classes * self.
            num_anchors, kernel_size=3, stride=1, padding=1)
        self.bbox_pred = nn.Conv2d(final_channel, 4 * self.num_anchors,
            kernel_size=3, stride=1, padding=1)
        for modules in [self.cls_logits, self.bbox_pred]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)
        prior_prob = cfg.MODEL.RETINANET.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

    def forward(self, x):
        logits = []
        bbox_reg = []
        for l, feature in enumerate(x):
            cls_out = feature
            reg_out = feature
            if self.has_split_weights:
                for ops in self._cls_head_split_ops[l]:
                    cls_out = ops(cls_out)
                for ops in self._reg_head_split_ops[l]:
                    reg_out = ops(reg_out)
            if self.has_shared_weights:
                for ops in self._cls_head_global_ops:
                    cls_out = ops(cls_out)
                for ops in self._reg_head_global_ops:
                    reg_out = ops(reg_out)
            logits.append(self.cls_logits(cls_out))
            bbox_reg.append(self.bbox_pred(reg_out))
        return logits, bbox_reg


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_Lausannen_NAS_FCOS(_paritybench_base):
    pass
    def test_000(self):
        self._check(ASPP(*[], **{'inplanes': 4, 'planes': 4, 'rates': [4, 4]}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(BatchNorm2d(*[], **{'num_features': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_002(self):
        self._check(CascadeConv3x3(*[], **{'C_in': 4, 'C_out': 4, 'stride': 1}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(ChannelShuffle(*[], **{'groups': 1}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_004(self):
        self._check(ConcatReduce(*[], **{'C': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_005(self):
        self._check(Conv2d(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_006(self):
        self._check(ConvTranspose2d(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_007(self):
        self._check(DilConv(*[], **{'C_in': 4, 'C_out': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4, 'dilation': 1}), [torch.rand([4, 4, 4, 4])], {})

    def test_008(self):
        self._check(FactorizedReduce(*[], **{'C_in': 4, 'C_out': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_009(self):
        self._check(FrozenBatchNorm2d(*[], **{'n': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_010(self):
        self._check(GAPConv1x1(*[], **{'C_in': 4, 'C_out': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_011(self):
        self._check(IOULoss(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_012(self):
        self._check(IRFBlock(*[], **{'input_depth': 1, 'output_depth': 1, 'expansion': 4, 'stride': 1}), [torch.rand([4, 1, 64, 64])], {})

    def test_013(self):
        self._check(Identity(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_014(self):
        self._check(InnerBlock(*[], **{'in_c': 4, 'out_c': 4, 'rates': [4, 4]}), [torch.rand([4, 4, 4, 4])], {})

    def test_015(self):
        self._check(InvertedResidual(*[], **{'inp': 4, 'oup': 4, 'stride': 1, 'expand_ratio': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_016(self):
        self._check(LastLevelMaxPool(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_017(self):
        self._check(LastLevelP6P7(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_018(self):
        self._check(ParamSum(*[], **{'C': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_019(self):
        self._check(SEModule(*[], **{'C': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_020(self):
        self._check(Scale(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_021(self):
        self._check(SepConv(*[], **{'C_in': 4, 'C_out': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_022(self):
        self._check(Shift(*[], **{'C': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_023(self):
        self._check(ShiftBlock5x5(*[], **{'C_in': 4, 'C_out': 4, 'expansion': 4, 'stride': 1}), [torch.rand([4, 4, 4, 4])], {})

    def test_024(self):
        self._check(Zero(*[], **{'stride': 1}), [torch.rand([4, 4, 4, 4])], {})

