import sys
_module = sys.modules[__name__]
del sys
predict = _module
webcam = _module
config = _module
defaults = _module
paths_catalog = _module
build = _module
collate_batch = _module
evaluation = _module
coco = _module
coco_eval = _module
gqa_coco = _module
gqa_coco_eval = _module
gqa_voc = _module
gqa_voc_eval = _module
sg = _module
evaluator = _module
sg_eval = _module
voc = _module
voc_eval = _module
factory = _module
samplers = _module
distributed = _module
grouped_batch_sampler = _module
iteration_based_batch_sampler = _module
transforms = _module
vg_eval = _module
vg_hdf5 = _module
model = _module
scene_parser = _module
parser = _module
rcnn = _module
datasets = _module
concat_dataset = _module
list_dataset = _module
engine = _module
bbox_aug = _module
inference = _module
trainer = _module
layers = _module
_utils = _module
batch_norm = _module
dcn = _module
deform_conv_func = _module
deform_conv_module = _module
deform_pool_func = _module
deform_pool_module = _module
misc = _module
nms = _module
roi_align = _module
roi_pool = _module
sigmoid_focal_loss = _module
smooth_l1_loss = _module
modeling = _module
backbone = _module
fbnet = _module
fbnet_builder = _module
fbnet_modeldef = _module
fpn = _module
resnet = _module
balanced_positive_negative_pair_sampler = _module
balanced_positive_negative_sampler = _module
box_coder = _module
detector = _module
generalized_rcnn = _module
make_layers = _module
matcher = _module
pair_matcher = _module
poolers = _module
registry = _module
relation_heads = _module
auxilary = _module
multi_head_att = _module
baseline = _module
baseline = _module
grcnn = _module
agcn = _module
agcn = _module
grcnn = _module
imp = _module
imp = _module
inference = _module
loss = _module
msdn = _module
msdn = _module
msdn_base = _module
relation_heads = _module
reldn = _module
reldn = _module
semantic = _module
spatial = _module
visual = _module
relpn = _module
relationshipness = _module
relpn = _module
utils = _module
roi_relation_box_feature_extractors = _module
roi_relation_box_predictors = _module
roi_relation_feature_extractors = _module
roi_relation_predictors = _module
sparse_targets = _module
roi_heads = _module
box_head = _module
box_head = _module
inference = _module
loss = _module
roi_box_feature_extractors = _module
roi_box_predictors = _module
roi_heads = _module
rpn = _module
anchor_generator = _module
inference = _module
loss = _module
retinanet = _module
loss = _module
retinanet = _module
rpn = _module
setup = _module
solver = _module
lr_scheduler = _module
structures = _module
bounding_box = _module
bounding_box_pair = _module
boxlist_ops = _module
image_list = _module
keypoint = _module
segmentation_mask = _module
boxes = _module
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
timer = _module
visualize = _module
box = _module
pytorch_misc = _module
main = _module

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


import logging


import copy


import torch.nn as nn


from torch import nn


from torch.autograd import Function


from torch.autograd.function import once_differentiable


from torch.nn.modules.utils import _pair


from torch.nn.modules.utils import _ntuple


from collections import OrderedDict


import torch.nn.functional as F


from collections import namedtuple


from torch.nn import functional as F


import numpy as np


from torch.autograd import Variable


from torch.nn import Parameter


from itertools import tee


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


class ModulatedDeformConvFunction(Function):

    @staticmethod
    def forward(ctx, input, offset, mask, weight, bias=None, stride=1,
        padding=0, dilation=1, groups=1, deformable_groups=1):
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.deformable_groups = deformable_groups
        ctx.with_bias = bias is not None
        if not ctx.with_bias:
            bias = input.new_empty(1)
        if not input.is_cuda:
            raise NotImplementedError
        if (weight.requires_grad or mask.requires_grad or offset.
            requires_grad or input.requires_grad):
            ctx.save_for_backward(input, offset, mask, weight, bias)
        output = input.new_empty(ModulatedDeformConvFunction._infer_shape(
            ctx, input, weight))
        ctx._bufs = [input.new_empty(0), input.new_empty(0)]
        _C.modulated_deform_conv_forward(input, weight, bias, ctx._bufs[0],
            offset, mask, output, ctx._bufs[1], weight.shape[2], weight.
            shape[3], ctx.stride, ctx.stride, ctx.padding, ctx.padding, ctx
            .dilation, ctx.dilation, ctx.groups, ctx.deformable_groups, ctx
            .with_bias)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        if not grad_output.is_cuda:
            raise NotImplementedError
        input, offset, mask, weight, bias = ctx.saved_tensors
        grad_input = torch.zeros_like(input)
        grad_offset = torch.zeros_like(offset)
        grad_mask = torch.zeros_like(mask)
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(bias)
        _C.modulated_deform_conv_backward(input, weight, bias, ctx._bufs[0],
            offset, mask, ctx._bufs[1], grad_input, grad_weight, grad_bias,
            grad_offset, grad_mask, grad_output, weight.shape[2], weight.
            shape[3], ctx.stride, ctx.stride, ctx.padding, ctx.padding, ctx
            .dilation, ctx.dilation, ctx.groups, ctx.deformable_groups, ctx
            .with_bias)
        if not ctx.with_bias:
            grad_bias = None
        return (grad_input, grad_offset, grad_mask, grad_weight, grad_bias,
            None, None, None, None, None)

    @staticmethod
    def _infer_shape(ctx, input, weight):
        n = input.size(0)
        channels_out = weight.size(0)
        height, width = input.shape[2:4]
        kernel_h, kernel_w = weight.shape[2:4]
        height_out = (height + 2 * ctx.padding - (ctx.dilation * (kernel_h -
            1) + 1)) // ctx.stride + 1
        width_out = (width + 2 * ctx.padding - (ctx.dilation * (kernel_w - 
            1) + 1)) // ctx.stride + 1
        return n, channels_out, height_out, width_out


modulated_deform_conv = ModulatedDeformConvFunction.apply


class ModulatedDeformConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, deformable_groups=1, bias=True):
        super(ModulatedDeformConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.with_bias = bias
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels //
            groups, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, input, offset, mask):
        return modulated_deform_conv(input, offset, mask, self.weight, self
            .bias, self.stride, self.padding, self.dilation, self.groups,
            self.deformable_groups)

    def __repr__(self):
        return ''.join(['{}('.format(self.__class__.__name__),
            'in_channels={}, '.format(self.in_channels),
            'out_channels={}, '.format(self.out_channels),
            'kernel_size={}, '.format(self.kernel_size), 'stride={}, '.
            format(self.stride), 'dilation={}, '.format(self.dilation),
            'padding={}, '.format(self.padding), 'groups={}, '.format(self.
            groups), 'deformable_groups={}, '.format(self.deformable_groups
            ), 'bias={})'.format(self.with_bias)])


class DeformRoIPoolingFunction(Function):

    @staticmethod
    def forward(ctx, data, rois, offset, spatial_scale, out_size,
        out_channels, no_trans, group_size=1, part_size=None,
        sample_per_part=4, trans_std=0.0):
        ctx.spatial_scale = spatial_scale
        ctx.out_size = out_size
        ctx.out_channels = out_channels
        ctx.no_trans = no_trans
        ctx.group_size = group_size
        ctx.part_size = out_size if part_size is None else part_size
        ctx.sample_per_part = sample_per_part
        ctx.trans_std = trans_std
        assert 0.0 <= ctx.trans_std <= 1.0
        if not data.is_cuda:
            raise NotImplementedError
        n = rois.shape[0]
        output = data.new_empty(n, out_channels, out_size, out_size)
        output_count = data.new_empty(n, out_channels, out_size, out_size)
        _C.deform_psroi_pooling_forward(data, rois, offset, output,
            output_count, ctx.no_trans, ctx.spatial_scale, ctx.out_channels,
            ctx.group_size, ctx.out_size, ctx.part_size, ctx.
            sample_per_part, ctx.trans_std)
        if data.requires_grad or rois.requires_grad or offset.requires_grad:
            ctx.save_for_backward(data, rois, offset)
        ctx.output_count = output_count
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        if not grad_output.is_cuda:
            raise NotImplementedError
        data, rois, offset = ctx.saved_tensors
        output_count = ctx.output_count
        grad_input = torch.zeros_like(data)
        grad_rois = None
        grad_offset = torch.zeros_like(offset)
        _C.deform_psroi_pooling_backward(grad_output, data, rois, offset,
            output_count, grad_input, grad_offset, ctx.no_trans, ctx.
            spatial_scale, ctx.out_channels, ctx.group_size, ctx.out_size,
            ctx.part_size, ctx.sample_per_part, ctx.trans_std)
        return (grad_input, grad_rois, grad_offset, None, None, None, None,
            None, None, None, None)


deform_roi_pooling = DeformRoIPoolingFunction.apply


class DeformRoIPooling(nn.Module):

    def __init__(self, spatial_scale, out_size, out_channels, no_trans,
        group_size=1, part_size=None, sample_per_part=4, trans_std=0.0):
        super(DeformRoIPooling, self).__init__()
        self.spatial_scale = spatial_scale
        self.out_size = out_size
        self.out_channels = out_channels
        self.no_trans = no_trans
        self.group_size = group_size
        self.part_size = out_size if part_size is None else part_size
        self.sample_per_part = sample_per_part
        self.trans_std = trans_std

    def forward(self, data, rois, offset):
        if self.no_trans:
            offset = data.new_empty(0)
        return deform_roi_pooling(data, rois, offset, self.spatial_scale,
            self.out_size, self.out_channels, self.no_trans, self.
            group_size, self.part_size, self.sample_per_part, self.trans_std)


class _NewEmptyTensorOp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, new_shape):
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, grad):
        shape = ctx.shape
        return _NewEmptyTensorOp.apply(grad, shape), None


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


class BatchNorm2d(torch.nn.BatchNorm2d):

    def forward(self, x):
        if x.numel() > 0:
            return super(BatchNorm2d, self).forward(x)
        output_shape = x.shape
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
        y = self.stages(y)
        ret = [y]
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


ARCH_CFG_NAME_MAPPING = {'bbox': 'ROI_BOX_HEAD', 'kpts':
    'ROI_KEYPOINT_HEAD', 'mask': 'ROI_MASK_HEAD'}


def _get_head_stage(arch, head_name, blocks):
    if head_name not in arch:
        head_name = 'head'
    head_stage = arch.get(head_name)
    ret = mbuilder.get_blocks(arch, stage_indices=head_stage, block_indices
        =blocks)
    return ret['stages']


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


def _make_stage(transformation_module, in_channels, bottleneck_channels,
    out_channels, block_count, num_groups, stride_in_1x1, first_stride,
    dilation=1, dcn_config={}):
    blocks = []
    stride = first_stride
    for _ in range(block_count):
        blocks.append(transformation_module(in_channels,
            bottleneck_channels, out_channels, num_groups, stride_in_1x1,
            stride, dilation=dilation, dcn_config=dcn_config))
        stride = 1
        in_channels = out_channels
    return nn.Sequential(*blocks)


class ResNetHead(nn.Module):

    def __init__(self, block_module, stages, num_groups=1, width_per_group=
        64, stride_in_1x1=True, stride_init=None, res2_out_channels=256,
        dilation=1, dcn_config={}):
        super(ResNetHead, self).__init__()
        stage2_relative_factor = 2 ** (stages[0].index - 1)
        stage2_bottleneck_channels = num_groups * width_per_group
        out_channels = res2_out_channels * stage2_relative_factor
        in_channels = out_channels // 2
        bottleneck_channels = (stage2_bottleneck_channels *
            stage2_relative_factor)
        block_module = _TRANSFORMATION_MODULES[block_module]
        self.stages = []
        stride = stride_init
        for stage in stages:
            name = 'layer' + str(stage.index)
            if not stride:
                stride = int(stage.index > 1) + 1
            module = _make_stage(block_module, in_channels,
                bottleneck_channels, out_channels, stage.block_count,
                num_groups, stride_in_1x1, first_stride=stride, dilation=
                dilation, dcn_config=dcn_config)
            stride = None
            self.add_module(name, module)
            self.stages.append(name)
        self.out_channels = out_channels

    def forward(self, x):
        for stage in self.stages:
            x = getattr(self, stage)(x)
        return x


class Bottleneck(nn.Module):

    def __init__(self, in_channels, bottleneck_channels, out_channels,
        num_groups, stride_in_1x1, stride, dilation, norm_func, dcn_config):
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
        with_dcn = dcn_config.get('stage_with_dcn', False)
        if with_dcn:
            deformable_groups = dcn_config.get('deformable_groups', 1)
            with_modulated_dcn = dcn_config.get('with_modulated_dcn', False)
            self.conv2 = DFConv2d(bottleneck_channels, bottleneck_channels,
                with_modulated_dcn=with_modulated_dcn, kernel_size=3,
                stride=stride_3x3, groups=num_groups, dilation=dilation,
                deformable_groups=deformable_groups, bias=False)
        else:
            self.conv2 = Conv2d(bottleneck_channels, bottleneck_channels,
                kernel_size=3, stride=stride_3x3, padding=dilation, bias=
                False, groups=num_groups, dilation=dilation)
            nn.init.kaiming_uniform_(self.conv2.weight, a=1)
        self.bn2 = norm_func(bottleneck_channels)
        self.conv3 = Conv2d(bottleneck_channels, out_channels, kernel_size=
            1, bias=False)
        self.bn3 = norm_func(out_channels)
        for l in [self.conv1, self.conv3]:
            nn.init.kaiming_uniform_(l.weight, a=1)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu_(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu_(out)
        out = self.conv3(out)
        out = self.bn3(out)
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


def build_roi_box_head(cfg, in_channels):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIBoxHead(cfg, in_channels)


def build_roi_heads(cfg, in_channels):
    roi_heads = []
    if cfg.MODEL.RETINANET_ON:
        return []
    if not cfg.MODEL.RPN_ONLY:
        roi_heads.append(('box', build_roi_box_head(cfg, in_channels)))
    if roi_heads:
        roi_heads = CombinedROIHeads(cfg, roi_heads)
    return roi_heads


def build_retinanet(cfg, in_channels):
    return RetinaNetModule(cfg, in_channels)


def build_rpn(cfg, in_channels):
    """
    This gives the gist of it. Not super important because it doesn't change as much
    """
    if cfg.MODEL.RETINANET_ON:
        return build_retinanet(cfg, in_channels)
    return RPNModule(cfg, in_channels)


def build_backbone(cfg):
    assert cfg.MODEL.BACKBONE.CONV_BODY in registry.BACKBONES, 'cfg.MODEL.BACKBONE.CONV_BODY: {} are not registered in registry'.format(
        cfg.MODEL.BACKBONE.CONV_BODY)
    return registry.BACKBONES[cfg.MODEL.BACKBONE.CONV_BODY](cfg)


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
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)

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


def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1000000000.0)
    scores = F.softmax(scores, dim=-1)
    if dropout is not None:
        scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output


class MultiHeadAttention(nn.Module):

    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        return output


def make_roi_relation_feature_extractor(cfg, in_channels):
    func = registry.ROI_RELATION_FEATURE_EXTRACTORS[cfg.MODEL.
        ROI_RELATION_HEAD.FEATURE_EXTRACTOR]
    return func(cfg, in_channels)


def make_roi_relation_predictor(cfg, in_channels):
    func = registry.ROI_RELATION_PREDICTOR[cfg.MODEL.ROI_RELATION_HEAD.
        PREDICTOR]
    return func(cfg, in_channels)


class Baseline(nn.Module):

    def __init__(self, cfg, in_channels):
        super(Baseline, self).__init__()
        self.cfg = cfg
        self.pred_feature_extractor = make_roi_relation_feature_extractor(cfg,
            in_channels)
        self.predictor = make_roi_relation_predictor(cfg, self.
            pred_feature_extractor.out_channels)

    def forward(self, features, proposals, proposal_pairs):
        obj_class_logits = None
        if self.training:
            x, rel_inds = self.pred_feature_extractor(features, proposals,
                proposal_pairs)
            rel_class_logits = self.predictor(x)
        else:
            with torch.no_grad():
                x, rel_inds = self.pred_feature_extractor(features,
                    proposals, proposal_pairs)
                rel_class_logits = self.predictor(x)
        if obj_class_logits is None:
            logits = torch.cat([proposal.get_field('logits') for proposal in
                proposals], 0)
            obj_class_labels = logits[:, 1:].max(1)[1] + 1
        else:
            obj_class_labels = obj_class_logits[:, 1:].max(1)[1] + 1
        return (x, obj_class_logits, rel_class_logits, obj_class_labels,
            rel_inds)


def normal_init(m, mean, stddev, truncated=False):
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()


class _Collection_Unit(nn.Module):

    def __init__(self, dim_in, dim_out):
        super(_Collection_Unit, self).__init__()
        self.fc = nn.Linear(dim_in, dim_out, bias=True)
        normal_init(self.fc, 0, 0.01)

    def forward(self, target, source, attention_base):
        fc_out = F.relu(self.fc(source))
        collect = torch.mm(attention_base, fc_out)
        collect_avg = collect / (attention_base.sum(1).view(collect.size(0),
            1) + 1e-07)
        return collect_avg


class _Update_Unit(nn.Module):

    def __init__(self, dim):
        super(_Update_Unit, self).__init__()

    def forward(self, target, source):
        assert target.size() == source.size(
            ), 'source dimension must be equal to target dimension'
        update = target + source
        return update


class _GraphConvolutionLayer_Collect(nn.Module):
    """ graph convolutional layer """
    """ collect information from neighbors """

    def __init__(self, dim_obj, dim_rel):
        super(_GraphConvolutionLayer_Collect, self).__init__()
        self.collect_units = nn.ModuleList()
        self.collect_units.append(_Collection_Unit(dim_rel, dim_obj))
        self.collect_units.append(_Collection_Unit(dim_rel, dim_obj))
        self.collect_units.append(_Collection_Unit(dim_obj, dim_rel))
        self.collect_units.append(_Collection_Unit(dim_obj, dim_rel))
        self.collect_units.append(_Collection_Unit(dim_obj, dim_obj))

    def forward(self, target, source, attention, unit_id):
        collection = self.collect_units[unit_id](target, source, attention)
        return collection


class _GraphConvolutionLayer_Update(nn.Module):
    """ graph convolutional layer """
    """ update target nodes """

    def __init__(self, dim_obj, dim_rel):
        super(_GraphConvolutionLayer_Update, self).__init__()
        self.update_units = nn.ModuleList()
        self.update_units.append(_Update_Unit(dim_obj))
        self.update_units.append(_Update_Unit(dim_rel))

    def forward(self, target, source, unit_id):
        update = self.update_units[unit_id](target, source)
        return update


def make_roi_relation_box_predictor(cfg, in_channels):
    func = registry.ROI_RELATION_BOX_PREDICTOR[cfg.MODEL.ROI_BOX_HEAD.PREDICTOR
        ]
    return func(cfg, in_channels)


class GRCNN(nn.Module):

    def __init__(self, cfg, in_channels):
        super(GRCNN, self).__init__()
        self.cfg = cfg
        self.dim = 1024
        self.feat_update_step = (cfg.MODEL.ROI_RELATION_HEAD.
            GRCNN_FEATURE_UPDATE_STEP)
        self.score_update_step = (cfg.MODEL.ROI_RELATION_HEAD.
            GRCNN_SCORE_UPDATE_STEP)
        num_classes_obj = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        num_classes_pred = cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.pred_feature_extractor = make_roi_relation_feature_extractor(cfg,
            in_channels)
        self.obj_embedding = nn.Sequential(nn.Linear(self.
            pred_feature_extractor.out_channels, self.dim), nn.ReLU(True),
            nn.Linear(self.dim, self.dim))
        self.rel_embedding = nn.Sequential(nn.Linear(self.
            pred_feature_extractor.out_channels, self.dim), nn.ReLU(True),
            nn.Linear(self.dim, self.dim))
        if self.feat_update_step > 0:
            self.gcn_collect_feat = _GraphConvolutionLayer_Collect(self.dim,
                self.dim)
            self.gcn_update_feat = _GraphConvolutionLayer_Update(self.dim,
                self.dim)
        if self.score_update_step > 0:
            self.gcn_collect_score = _GraphConvolutionLayer_Collect(
                num_classes_obj, num_classes_pred)
            self.gcn_update_score = _GraphConvolutionLayer_Update(
                num_classes_obj, num_classes_pred)
        self.obj_predictor = make_roi_relation_box_predictor(cfg, self.dim)
        self.pred_predictor = make_roi_relation_predictor(cfg, self.dim)

    def _get_map_idxs(self, proposals, proposal_pairs):
        rel_inds = []
        offset = 0
        obj_num = sum([len(proposal) for proposal in proposals])
        obj_obj_map = torch.FloatTensor(obj_num, obj_num).fill_(0)
        for proposal, proposal_pair in zip(proposals, proposal_pairs):
            rel_ind_i = proposal_pair.get_field('idx_pairs').detach()
            obj_obj_map_i = (1 - torch.eye(len(proposal))).float()
            obj_obj_map[offset:offset + len(proposal), offset:offset + len(
                proposal)] = obj_obj_map_i
            rel_ind_i += offset
            offset += len(proposal)
            rel_inds.append(rel_ind_i)
        rel_inds = torch.cat(rel_inds, 0)
        subj_pred_map = rel_inds.new(obj_num, rel_inds.shape[0]).fill_(0
            ).float().detach()
        obj_pred_map = rel_inds.new(obj_num, rel_inds.shape[0]).fill_(0).float(
            ).detach()
        subj_pred_map.scatter_(0, rel_inds[:, (0)].contiguous().view(1, -1), 1)
        obj_pred_map.scatter_(0, rel_inds[:, (1)].contiguous().view(1, -1), 1)
        obj_obj_map = obj_obj_map.type_as(obj_pred_map)
        return rel_inds, obj_obj_map, subj_pred_map, obj_pred_map

    def forward(self, features, proposals, proposal_pairs):
        rel_inds, obj_obj_map, subj_pred_map, obj_pred_map = (self.
            _get_map_idxs(proposals, proposal_pairs))
        x_obj = torch.cat([proposal.get_field('features').detach() for
            proposal in proposals], 0)
        obj_class_logits = torch.cat([proposal.get_field('logits').detach() for
            proposal in proposals], 0)
        x_pred, _ = self.pred_feature_extractor(features, proposals,
            proposal_pairs)
        x_pred = self.avgpool(x_pred)
        x_obj = x_obj.view(x_obj.size(0), -1)
        x_obj = self.obj_embedding(x_obj)
        x_pred = x_pred.view(x_pred.size(0), -1)
        x_pred = self.rel_embedding(x_pred)
        """feature level agcn"""
        obj_feats = [x_obj]
        pred_feats = [x_pred]
        for t in range(self.feat_update_step):
            source_obj = self.gcn_collect_feat(obj_feats[t], obj_feats[t],
                obj_obj_map, 4)
            source_rel_sub = self.gcn_collect_feat(obj_feats[t], pred_feats
                [t], subj_pred_map, 0)
            source_rel_obj = self.gcn_collect_feat(obj_feats[t], pred_feats
                [t], obj_pred_map, 1)
            source2obj_all = (source_obj + source_rel_sub + source_rel_obj) / 3
            obj_feats.append(self.gcn_update_feat(obj_feats[t],
                source2obj_all, 0))
            """update predicate logits"""
            source_obj_sub = self.gcn_collect_feat(pred_feats[t], obj_feats
                [t], subj_pred_map.t(), 2)
            source_obj_obj = self.gcn_collect_feat(pred_feats[t], obj_feats
                [t], obj_pred_map.t(), 3)
            source2rel_all = (source_obj_sub + source_obj_obj) / 2
            pred_feats.append(self.gcn_update_feat(pred_feats[t],
                source2rel_all, 1))
        obj_class_logits = self.obj_predictor(obj_feats[-1].unsqueeze(2).
            unsqueeze(3))
        pred_class_logits = self.pred_predictor(pred_feats[-1].unsqueeze(2)
            .unsqueeze(3))
        """score level agcn"""
        obj_scores = [obj_class_logits]
        pred_scores = [pred_class_logits]
        for t in range(self.score_update_step):
            """update object logits"""
            source_obj = self.gcn_collect_score(obj_scores[t], obj_scores[t
                ], obj_obj_map, 4)
            source_rel_sub = self.gcn_collect_score(obj_scores[t],
                pred_scores[t], subj_pred_map, 0)
            source_rel_obj = self.gcn_collect_score(obj_scores[t],
                pred_scores[t], obj_pred_map, 1)
            source2obj_all = (source_obj + source_rel_sub + source_rel_obj) / 3
            obj_scores.append(self.gcn_update_score(obj_scores[t],
                source2obj_all, 0))
            """update predicate logits"""
            source_obj_sub = self.gcn_collect_score(pred_scores[t],
                obj_scores[t], subj_pred_map.t(), 2)
            source_obj_obj = self.gcn_collect_score(pred_scores[t],
                obj_scores[t], obj_pred_map.t(), 3)
            source2rel_all = (source_obj_sub + source_obj_obj) / 2
            pred_scores.append(self.gcn_update_score(pred_scores[t],
                source2rel_all, 1))
        obj_class_logits = obj_scores[-1]
        pred_class_logits = pred_scores[-1]
        if obj_class_logits is None:
            logits = torch.cat([proposal.get_field('logits') for proposal in
                proposals], 0)
            obj_class_labels = logits[:, 1:].max(1)[1] + 1
        else:
            obj_class_labels = obj_class_logits[:, 1:].max(1)[1] + 1
        return (x_pred, obj_class_logits, pred_class_logits,
            obj_class_labels, rel_inds)


class IMP(nn.Module):

    def __init__(self, cfg, in_channels):
        super(IMP, self).__init__()
        self.cfg = cfg
        self.dim = 512
        self.update_step = cfg.MODEL.ROI_RELATION_HEAD.IMP_FEATURE_UPDATE_STEP
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.pred_feature_extractor = make_roi_relation_feature_extractor(cfg,
            in_channels)
        self.obj_embedding = nn.Sequential(nn.Linear(self.
            pred_feature_extractor.out_channels, self.dim), nn.ReLU(True),
            nn.Linear(self.dim, self.dim))
        self.pred_embedding = nn.Sequential(nn.Linear(self.
            pred_feature_extractor.out_channels, self.dim), nn.ReLU(True),
            nn.Linear(self.dim, self.dim))
        if self.update_step > 0:
            self.edge_gru = nn.GRUCell(input_size=self.dim, hidden_size=
                self.dim)
            self.node_gru = nn.GRUCell(input_size=self.dim, hidden_size=
                self.dim)
            self.subj_node_gate = nn.Sequential(nn.Linear(self.dim * 2, 1),
                nn.Sigmoid())
            self.obj_node_gate = nn.Sequential(nn.Linear(self.dim * 2, 1),
                nn.Sigmoid())
            self.subj_edge_gate = nn.Sequential(nn.Linear(self.dim * 2, 1),
                nn.Sigmoid())
            self.obj_edge_gate = nn.Sequential(nn.Linear(self.dim * 2, 1),
                nn.Sigmoid())
        self.obj_predictor = make_roi_relation_box_predictor(cfg, 512)
        self.pred_predictor = make_roi_relation_predictor(cfg, 512)

    def _get_map_idxs(self, proposals, proposal_pairs):
        rel_inds = []
        offset = 0
        for proposal, proposal_pair in zip(proposals, proposal_pairs):
            rel_ind_i = proposal_pair.get_field('idx_pairs').detach()
            rel_ind_i += offset
            offset += len(proposal)
            rel_inds.append(rel_ind_i)
        rel_inds = torch.cat(rel_inds, 0)
        subj_pred_map = rel_inds.new(sum([len(proposal) for proposal in
            proposals]), rel_inds.shape[0]).fill_(0).float().detach()
        obj_pred_map = rel_inds.new(sum([len(proposal) for proposal in
            proposals]), rel_inds.shape[0]).fill_(0).float().detach()
        subj_pred_map.scatter_(0, rel_inds[:, (0)].contiguous().view(1, -1), 1)
        obj_pred_map.scatter_(0, rel_inds[:, (1)].contiguous().view(1, -1), 1)
        return rel_inds, subj_pred_map, obj_pred_map

    def forward(self, features, proposals, proposal_pairs):
        rel_inds, subj_pred_map, obj_pred_map = self._get_map_idxs(proposals,
            proposal_pairs)
        x_obj = torch.cat([proposal.get_field('features') for proposal in
            proposals], 0)
        x_pred, _ = self.pred_feature_extractor(features, proposals,
            proposal_pairs)
        x_pred = self.avgpool(x_pred)
        x_obj = x_obj.view(x_obj.size(0), -1)
        x_pred = x_pred.view(x_pred.size(0), -1)
        x_obj = self.obj_embedding(x_obj)
        x_pred = self.pred_embedding(x_pred)
        hx_obj = [x_obj]
        hx_edge = [x_pred]
        for t in range(self.update_step):
            sub_vert = hx_obj[t][rel_inds[:, (0)]]
            obj_vert = hx_obj[t][rel_inds[:, (1)]]
            """update object features"""
            message_pred_to_subj = self.subj_node_gate(torch.cat([sub_vert,
                hx_edge[t]], 1)) * hx_edge[t]
            message_pred_to_obj = self.obj_node_gate(torch.cat([obj_vert,
                hx_edge[t]], 1)) * hx_edge[t]
            node_message = (torch.mm(subj_pred_map, message_pred_to_subj) /
                (subj_pred_map.sum(1, keepdim=True) + 1e-05) + torch.mm(
                obj_pred_map, message_pred_to_obj) / (obj_pred_map.sum(1,
                keepdim=True) + 1e-05)) / 2.0
            hx_obj.append(self.node_gru(node_message, hx_obj[t]))
            """update predicat features"""
            message_subj_to_pred = self.subj_edge_gate(torch.cat([sub_vert,
                hx_edge[t]], 1)) * sub_vert
            message_obj_to_pred = self.obj_edge_gate(torch.cat([obj_vert,
                hx_edge[t]], 1)) * obj_vert
            edge_message = (message_subj_to_pred + message_obj_to_pred) / 2.0
            hx_edge.append(self.edge_gru(edge_message, hx_edge[t]))
        """compute results and losses"""
        obj_class_logits = self.obj_predictor(hx_obj[-1].unsqueeze(2).
            unsqueeze(3))
        pred_class_logits = self.pred_predictor(hx_edge[-1].unsqueeze(2).
            unsqueeze(3))
        if obj_class_logits is None:
            logits = torch.cat([proposal.get_field('logits') for proposal in
                proposals], 0)
            obj_class_labels = logits[:, 1:].max(1)[1] + 1
        else:
            obj_class_labels = obj_class_logits[:, 1:].max(1)[1] + 1
        return (hx_obj[-1], hx_edge[-1]
            ), obj_class_logits, pred_class_logits, obj_class_labels, rel_inds


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

    def crop(self, box):
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
        if False:
            is_empty = (cropped_xmin == cropped_xmax) | (cropped_ymin ==
                cropped_ymax)
        cropped_box = torch.cat((cropped_xmin, cropped_ymin, cropped_xmax,
            cropped_ymax), dim=-1)
        bbox = BoxList(cropped_box, (w, h), mode='xyxy')
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.crop(box)
            bbox.add_field(k, v)
        return bbox.convert(self.mode)

    def to(self, device):
        bbox = BoxList(self.bbox.to(device), self.size, self.mode)
        for k, v in self.extra_fields.items():
            if hasattr(v, 'to'):
                v = v.to(device)
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

    def copy_with_fields(self, fields, skip_missing=False):
        bbox = BoxList(self.bbox, self.size, self.mode)
        if not isinstance(fields, (list, tuple)):
            fields = [fields]
        for field in fields:
            if self.has_field(field):
                bbox.add_field(field, self.get_field(field))
            elif not skip_missing:
                raise KeyError("Field '{}' not found in {}".format(field, self)
                    )
        return bbox

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += 'num_boxes={}, '.format(len(self))
        s += 'image_width={}, '.format(self.size[0])
        s += 'image_height={}, '.format(self.size[1])
        s += 'mode={})'.format(self.mode)
        return s


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


class BoxPairList(object):
    """
    This class represents a set of bounding boxes.
    The bounding boxes are represented as a Nx4 Tensor.
    In order to uniquely determine the bounding boxes with respect
    to an image, we also store the corresponding image dimensions.
    They can contain extra information that is specific to each bounding box, such as
    labels.
    """

    def __init__(self, bbox_pair, image_size, mode='xyxy'):
        device = bbox_pair.device if isinstance(bbox_pair, torch.Tensor
            ) else torch.device('cpu')
        bbox_pair = torch.as_tensor(bbox_pair, dtype=torch.float32, device=
            device)
        if bbox_pair.ndimension() != 2:
            raise ValueError('bbox should have 2 dimensions, got {}'.format
                (bbox_pair.ndimension()))
        if bbox_pair.size(-1) != 8:
            raise ValueError(
                'last dimension of bbox should have a size of 8, got {}'.
                format(bbox_pair.size(-1)))
        if mode not in ('xyxy', 'xywh'):
            raise ValueError("mode should be 'xyxy' or 'xywh'")
        self.bbox = bbox_pair
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
            bbox = BoxPairList(bbox, self.size, mode=mode)
        else:
            TO_REMOVE = 1
            bbox = torch.cat((xmin, ymin, xmax - xmin + TO_REMOVE, ymax -
                ymin + TO_REMOVE), dim=-1)
            bbox = BoxPairList(bbox, self.size, mode=mode)
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
            bbox = BoxPairList(scaled_box, size, mode=self.mode)
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
        bbox = BoxPairList(scaled_box, size, mode='xyxy')
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
        bbox = BoxPairList(transposed_boxes, self.size, mode='xyxy')
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.transpose(method)
            bbox.add_field(k, v)
        return bbox.convert(self.mode)

    def crop(self, box):
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
        if False:
            is_empty = (cropped_xmin == cropped_xmax) | (cropped_ymin ==
                cropped_ymax)
        cropped_box = torch.cat((cropped_xmin, cropped_ymin, cropped_xmax,
            cropped_ymax), dim=-1)
        bbox = BoxPairList(cropped_box, (w, h), mode='xyxy')
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.crop(box)
            bbox.add_field(k, v)
        return bbox.convert(self.mode)

    def to(self, device):
        bbox = BoxPairList(self.bbox.to(device), self.size, self.mode)
        for k, v in self.extra_fields.items():
            if hasattr(v, 'to'):
                v = v.to(device)
            bbox.add_field(k, v)
        return bbox

    def __getitem__(self, item):
        bbox = BoxPairList(self.bbox[item], self.size, self.mode)
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

    def copy_with_fields(self, fields, skip_missing=False):
        bbox = BoxPairList(self.bbox, self.size, self.mode)
        if not isinstance(fields, (list, tuple)):
            fields = [fields]
        for field in fields:
            if self.has_field(field):
                bbox.add_field(field, self.get_field(field))
            elif not skip_missing:
                raise KeyError("Field '{}' not found in {}".format(field, self)
                    )
        return bbox

    def copy_with_subject(self):
        bbox = BoxList(self.bbox[:, :4], self.size, self.mode)
        return bbox

    def copy_with_object(self):
        bbox = BoxList(self.bbox[:, 4:], self.size, self.mode)
        return bbox

    def copy_with_union(self):
        x1 = self.bbox[:, 0::4].min(1)[0].view(-1, 1)
        y1 = self.bbox[:, 1::4].min(1)[0].view(-1, 1)
        x2 = self.bbox[:, 2::4].max(1)[0].view(-1, 1)
        y2 = self.bbox[:, 3::4].max(1)[0].view(-1, 1)
        bbox = BoxList(torch.cat((x1, y1, x2, y2), 1), self.size, self.mode)
        return bbox

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += 'num_boxes={}, '.format(len(self))
        s += 'image_width={}, '.format(self.size[0])
        s += 'image_height={}, '.format(self.size[1])
        s += 'mode={})'.format(self.mode)
        return s


class PostProcessor(nn.Module):
    """
    From a set of classification scores, box regression and proposals,
    computes the post-processed boxes, and applies NMS to obtain the
    final results
    """

    def __init__(self, score_thresh=0.05, nms=0.5, detections_per_img=100,
        box_coder=None, cls_agnostic_bbox_reg=False, bbox_aug_enabled=False):
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
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg
        self.bbox_aug_enabled = bbox_aug_enabled

    def forward(self, x, boxes, use_freq_prior=False):
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
        class_logits = x
        class_prob = class_logits if use_freq_prior else F.softmax(class_logits
            , -1)
        image_shapes = [box.size for box in boxes]
        boxes_per_image = [len(box) for box in boxes]
        num_classes = class_prob.shape[1]
        proposals = boxes
        class_prob = class_prob.split(boxes_per_image, dim=0)
        results = []
        for prob, boxes_per_img, image_shape in zip(class_prob, proposals,
            image_shapes):
            boxes_per_img.add_field('scores', prob)
            results.append(boxes_per_img)
        return results

    def prepare_boxpairlist(self, boxes, scores, image_shape):
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
        boxes = boxes.reshape(-1, 8)
        scores = scores.reshape(-1)
        boxlist = BoxPairList(boxes, image_shape, mode='xyxy')
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
            boxlist_for_class = boxlist_nms(boxlist_for_class, self.nms)
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


class Message_Passing_Unit_v2(nn.Module):

    def __init__(self, fea_size, filter_size=128):
        super(Message_Passing_Unit_v2, self).__init__()
        self.w = nn.Linear(fea_size, filter_size, bias=True)
        self.fea_size = fea_size
        self.filter_size = filter_size

    def forward(self, unary_term, pair_term):
        if unary_term.size()[0] == 1 and pair_term.size()[0] > 1:
            unary_term = unary_term.expand(pair_term.size()[0], unary_term.
                size()[1])
        if unary_term.size()[0] > 1 and pair_term.size()[0] == 1:
            pair_term = pair_term.expand(unary_term.size()[0], pair_term.
                size()[1])
        gate = self.w(F.relu(unary_term)) * self.w(F.relu(pair_term))
        gate = torch.sigmoid(gate.sum(1))
        output = pair_term * gate.expand(gate.size()[0], pair_term.size()[1])
        return output


class Message_Passing_Unit_v1(nn.Module):

    def __init__(self, fea_size, filter_size=128):
        super(Message_Passing_Unit_v1, self).__init__()
        self.w = nn.Linear(fea_size * 2, filter_size, bias=True)
        self.fea_size = fea_size
        self.filter_size = filter_size

    def forward(self, unary_term, pair_term):
        if unary_term.size()[0] == 1 and pair_term.size()[0] > 1:
            unary_term = unary_term.expand(pair_term.size()[0], unary_term.
                size()[1])
        if unary_term.size()[0] > 1 and pair_term.size()[0] == 1:
            pair_term = pair_term.expand(unary_term.size()[0], pair_term.
                size()[1])
        gate = torch.cat([unary_term, pair_term], 1)
        gate = F.relu(gate)
        gate = torch.sigmoid(self.w(gate)).mean(1)
        output = pair_term * gate.view(-1, 1).expand(gate.size()[0],
            pair_term.size()[1])
        return output


class Gated_Recurrent_Unit(nn.Module):

    def __init__(self, fea_size, dropout):
        super(Gated_Recurrent_Unit, self).__init__()
        self.wih = nn.Linear(fea_size, fea_size, bias=True)
        self.whh = nn.Linear(fea_size, fea_size, bias=True)
        self.dropout = dropout

    def forward(self, input, hidden):
        output = self.wih(F.relu(input)) + self.whh(F.relu(hidden))
        if self.dropout:
            output = F.dropout(output, training=self.training)
        return output


class MSDN_BASE(nn.Module):

    def __init__(self, fea_size, dropout=False, gate_width=128, use_region=
        False, use_kernel_function=False):
        super(MSDN_BASE, self).__init__()
        if use_kernel_function:
            Message_Passing_Unit = Message_Passing_Unit_v2
        else:
            Message_Passing_Unit = Message_Passing_Unit_v1
        self.gate_sub2pred = Message_Passing_Unit(fea_size, gate_width)
        self.gate_obj2pred = Message_Passing_Unit(fea_size, gate_width)
        self.gate_pred2sub = Message_Passing_Unit(fea_size, gate_width)
        self.gate_pred2obj = Message_Passing_Unit(fea_size, gate_width)
        self.GRU_object = Gated_Recurrent_Unit(fea_size, dropout)
        self.GRU_pred = Gated_Recurrent_Unit(fea_size, dropout)

    def forward(self, feature_obj, feature_phrase, feature_region,
        mps_object, mps_phrase, mps_region):
        raise Exception('Please implement the forward function')

    def prepare_message(self, target_features, source_features, select_mat,
        gate_module):
        feature_data = []
        if select_mat.data.sum() == 0:
            temp = Variable(torch.zeros(target_features.size()[1:]),
                requires_grad=True).type_as(target_features)
            feature_data.append(temp)
        else:
            transfer_list = (select_mat.data > 0).nonzero()
            source_indices = Variable(transfer_list[:, (1)])
            target_indices = Variable(transfer_list[:, (0)])
            source_f = torch.index_select(source_features, 0, source_indices)
            target_f = torch.index_select(target_features, 0, target_indices)
            transferred_features = gate_module(target_f, source_f)
            for f_id in range(target_features.size()[0]):
                if select_mat[(f_id), :].data.sum() > 0:
                    feature_indices = (transfer_list[:, (0)] == f_id).nonzero(
                        )[0]
                    indices = Variable(feature_indices)
                    features = torch.index_select(transferred_features, 0,
                        indices).mean(0).view(-1)
                    feature_data.append(features)
                else:
                    temp = Variable(torch.zeros(target_features.size()[1:]),
                        requires_grad=True).type_as(target_features)
                    feature_data.append(temp)
        return torch.stack(feature_data, 0)


class BalancedPositiveNegativePairSampler(object):
    """
    This class samples batches, ensuring that they contain a fixed proportion of positives
    """

    def __init__(self, batch_size_per_image, positive_fraction):
        """
        Arguments:
            batch_size_per_image (int): number of elements to be selected per image
            positive_fraction (float): percentage of positive elements per batch
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


class PairMatcher(object):
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

    def __call__(self, match_pair_quality_matrix):
        """
        Args:
            match_pair_quality_matrix (Tensor[float]): an MxN tensor, containing the
            pairwise quality between M ground-truth pairs and N predicted pairs.

        Returns:
            matches (Tensor[int64]): an N tensor where N[i] is a matched gt in
            [0, M - 1] or a negative value indicating that prediction i could not
            be matched.
        """
        if match_pair_quality_matrix.numel() == 0:
            if match_quality_matrix.shape[0] == 0:
                raise ValueError(
                    'No ground-truth boxes available for one of the images during training'
                    )
            else:
                raise ValueError(
                    'No proposal boxes available for one of the images during training'
                    )
        matched_vals, matches = match_pair_quality_matrix.max(dim=0)
        if self.allow_low_quality_matches:
            all_matches = matches.clone()
        below_low_threshold = matched_vals < self.low_threshold
        between_thresholds = (matched_vals >= self.low_threshold) & (
            matched_vals < self.high_threshold)
        matches[below_low_threshold] = PairMatcher.BELOW_LOW_THRESHOLD
        matches[between_thresholds] = PairMatcher.BETWEEN_THRESHOLDS
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


def make_relation_proposal_network(cfg):
    matcher = PairMatcher(cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD, cfg.MODEL.
        ROI_HEADS.BG_IOU_THRESHOLD, allow_low_quality_matches=False)
    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights)
    fg_bg_sampler = BalancedPositiveNegativePairSampler(cfg.MODEL.
        ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE, cfg.MODEL.ROI_RELATION_HEAD
        .POSITIVE_FRACTION)
    cls_agnostic_bbox_reg = cfg.MODEL.CLS_AGNOSTIC_BBOX_REG
    relpn = RelPN(cfg, matcher, fg_bg_sampler, box_coder, cls_agnostic_bbox_reg
        )
    return relpn


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
    boxlist1 = boxlist1.convert('xyxy')
    boxlist2 = boxlist2.convert('xyxy')
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


def build_imp_model(cfg, in_channels):
    return IMP(cfg, in_channels)


def make_roi_relation_post_processor(cfg):
    use_fpn = cfg.MODEL.ROI_HEADS.USE_FPN
    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights)
    score_thresh = cfg.MODEL.ROI_HEADS.SCORE_THRESH
    nms_thresh = cfg.MODEL.ROI_HEADS.NMS
    detections_per_img = cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG
    cls_agnostic_bbox_reg = cfg.MODEL.CLS_AGNOSTIC_BBOX_REG
    bbox_aug_enabled = cfg.TEST.BBOX_AUG.ENABLED
    postprocessor = PostProcessor(score_thresh, nms_thresh,
        detections_per_img, box_coder, cls_agnostic_bbox_reg, bbox_aug_enabled)
    return postprocessor


def make_roi_box_feature_extractor(cfg, in_channels):
    func = registry.ROI_BOX_FEATURE_EXTRACTORS[cfg.MODEL.ROI_BOX_HEAD.
        FEATURE_EXTRACTOR]
    return func(cfg, in_channels)


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
        matches[below_low_threshold] = Matcher.BELOW_LOW_THRESHOLD
        matches[between_thresholds] = Matcher.BETWEEN_THRESHOLDS
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


class FastRCNNLossComputation(object):
    """
    Computes the loss for Faster R-CNN.
    Also supports FPN
    """

    def __init__(self, proposal_matcher, fg_bg_sampler, box_coder,
        cls_agnostic_bbox_reg=False):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.box_coder = box_coder
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg

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
            img_sampled_inds = torch.nonzero(pos_inds_img.view(-1) |
                neg_inds_img.view(-1)).squeeze(1)
            proposals_per_image = proposals[img_idx][img_sampled_inds]
            proposals[img_idx] = proposals_per_image
        self._proposals = proposals
        return proposals

    def prepare_labels(self, proposals, targets):
        """
        This method prepares the ground-truth labels for each bounding box, and return
        the sampled proposals.
        Note: this function keeps a state.

        Arguments:
            proposals (list[BoxList])
            targets (list[BoxList])
        """
        labels, regression_targets = self.prepare_targets(proposals, targets)
        proposals = list(proposals)
        for labels_per_image, regression_targets_per_image, proposals_per_image in zip(
            labels, regression_targets, proposals):
            proposals_per_image.add_field('labels', labels_per_image)
            proposals_per_image.add_field('regression_targets',
                regression_targets_per_image)
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
        if self.cls_agnostic_bbox_reg:
            map_inds = torch.tensor([4, 5, 6, 7], device=device)
        else:
            map_inds = 4 * labels_pos[:, (None)] + torch.tensor([0, 1, 2, 3
                ], device=device)
        box_loss = smooth_l1_loss(box_regression[sampled_pos_inds_subset[:,
            (None)], map_inds], regression_targets[sampled_pos_inds_subset],
            size_average=False, beta=1)
        box_loss = box_loss / labels.numel()
        return classification_loss, box_loss


def make_roi_relation_loss_evaluator(cfg):
    matcher = PairMatcher(cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD, cfg.MODEL.
        ROI_HEADS.BG_IOU_THRESHOLD, allow_low_quality_matches=False)
    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights)
    fg_bg_sampler = BalancedPositiveNegativePairSampler(cfg.MODEL.
        ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE, cfg.MODEL.ROI_RELATION_HEAD
        .POSITIVE_FRACTION)
    cls_agnostic_bbox_reg = cfg.MODEL.CLS_AGNOSTIC_BBOX_REG
    loss_evaluator = FastRCNNLossComputation(cfg, matcher, fg_bg_sampler,
        box_coder, cls_agnostic_bbox_reg)
    return loss_evaluator


def build_baseline_model(cfg, in_channels):
    return Baseline(cfg, in_channels)


def make_roi_box_predictor(cfg, in_channels):
    func = registry.ROI_BOX_PREDICTOR[cfg.MODEL.ROI_BOX_HEAD.PREDICTOR]
    return func(cfg, in_channels)


def _get_tensor_from_boxlist(proposals, field='labels'):
    assert proposals[0].extra_fields[field] is not None
    for im_ind, prop_per_im in enumerate(proposals):
        if im_ind == 0:
            num_proposals_im = prop_per_im.bbox.size(0)
            bbox_batch = prop_per_im.bbox
            output_batch = prop_per_im.extra_fields[field]
            im_inds = im_ind * torch.ones(num_proposals_im, 1)
        else:
            num_proposals_im = prop_per_im.bbox.size(0)
            bbox_batch = torch.cat((bbox_batch, prop_per_im.bbox), dim=0)
            output_batch = torch.cat((output_batch, prop_per_im.
                extra_fields[field]), dim=0)
            im_inds = torch.cat((im_inds, im_ind * torch.ones(
                num_proposals_im, 1)), dim=0)
    im_inds_batch = torch.Tensor(im_inds).long().cuda()
    return bbox_batch, output_batch, im_inds_batch


def build_grcnn_model(cfg, in_channels):
    return GRCNN(cfg, in_channels)


def build_reldn_model(cfg, in_channels):
    return RelDN(cfg, in_channels)


def make_roi_box_post_processor(cfg):
    use_fpn = cfg.MODEL.ROI_HEADS.USE_FPN
    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights)
    score_thresh = cfg.MODEL.ROI_HEADS.SCORE_THRESH
    nms_thresh = cfg.MODEL.ROI_HEADS.NMS
    detections_per_img = cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG
    min_detections_per_img = cfg.MODEL.ROI_HEADS.MIN_DETECTIONS_PER_IMG
    cls_agnostic_bbox_reg = cfg.MODEL.CLS_AGNOSTIC_BBOX_REG
    bbox_aug_enabled = cfg.TEST.BBOX_AUG.ENABLED
    postprocessor = PostProcessor(score_thresh, nms_thresh,
        detections_per_img, min_detections_per_img, box_coder,
        cls_agnostic_bbox_reg, bbox_aug_enabled, relation_on=cfg.MODEL.
        RELATION_ON)
    return postprocessor


def _get_rel_inds(im_inds, im_inds_pairs, proposal_idx_pairs):
    rel_ind_sub = proposal_idx_pairs[:, (0)]
    rel_ind_obj = proposal_idx_pairs[:, (1)]
    num_obj_im = torch.unique(im_inds)
    num_obj_im = torch.cumsum(num_obj_im, dim=0)
    rel_ind_offset_im = num_obj_im[im_inds_pairs - 1]
    num_rels_im = torch.unique(im_inds_pairs)
    rel_ind_offset_im[:num_rels_im[0]] = 0
    rel_ind_offset_im = torch.squeeze(rel_ind_offset_im)
    rel_ind_sub += rel_ind_offset_im
    rel_ind_obj += rel_ind_offset_im
    return torch.cat((rel_ind_sub[:, (None)], rel_ind_obj[:, (None)]), 1)


class MSDN(MSDN_BASE):

    def __init__(self, cfg, in_channels, dim=1024, dropout=False,
        gate_width=128, use_kernel_function=False):
        super(MSDN, self).__init__(dim, dropout, gate_width, use_region=
            True, use_kernel_function=use_kernel_function)
        self.cfg = cfg
        self.dim = dim
        self.update_step = cfg.MODEL.ROI_RELATION_HEAD.MSDN_FEATURE_UPDATE_STEP
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.pred_feature_extractor = make_roi_relation_feature_extractor(cfg,
            in_channels)
        self.obj_embedding = nn.Sequential(nn.Linear(self.
            pred_feature_extractor.out_channels, self.dim), nn.ReLU(True),
            nn.Linear(self.dim, self.dim))
        self.rel_embedding = nn.Sequential(nn.Linear(self.
            pred_feature_extractor.out_channels, self.dim), nn.ReLU(True),
            nn.Linear(self.dim, self.dim))
        self.obj_predictor = make_roi_relation_box_predictor(cfg, dim)
        self.pred_predictor = make_roi_relation_predictor(cfg, dim)

    def _get_map_idxs(self, proposals, proposal_pairs):
        rel_inds = []
        offset = 0
        for proposal, proposal_pair in zip(proposals, proposal_pairs):
            rel_ind_i = proposal_pair.get_field('idx_pairs').detach()
            rel_ind_i += offset
            offset += len(proposal)
            rel_inds.append(rel_ind_i)
        rel_inds = torch.cat(rel_inds, 0)
        subj_pred_map = rel_inds.new(sum([len(proposal) for proposal in
            proposals]), rel_inds.shape[0]).fill_(0).float().detach()
        obj_pred_map = rel_inds.new(sum([len(proposal) for proposal in
            proposals]), rel_inds.shape[0]).fill_(0).float().detach()
        subj_pred_map.scatter_(0, rel_inds[:, (0)].contiguous().view(1, -1), 1)
        obj_pred_map.scatter_(0, rel_inds[:, (1)].contiguous().view(1, -1), 1)
        return rel_inds, subj_pred_map, obj_pred_map

    def forward(self, features, proposals, proposal_pairs):
        rel_inds, subj_pred_map, obj_pred_map = self._get_map_idxs(proposals,
            proposal_pairs)
        x_obj = torch.cat([proposal.get_field('features').detach() for
            proposal in proposals], 0)
        x_pred, _ = self.pred_feature_extractor(features, proposals,
            proposal_pairs)
        x_pred = self.avgpool(x_pred)
        x_obj = x_obj.view(x_obj.size(0), -1)
        x_pred = x_pred.view(x_pred.size(0), -1)
        x_obj = self.obj_embedding(x_obj)
        x_pred = self.rel_embedding(x_pred)
        x_obj = [x_obj]
        x_pred = [x_pred]
        for t in range(self.update_step):
            """update object features"""
            object_sub = self.prepare_message(x_obj[t], x_pred[t],
                subj_pred_map, self.gate_pred2sub)
            object_obj = self.prepare_message(x_obj[t], x_pred[t],
                obj_pred_map, self.gate_pred2obj)
            GRU_input_feature_object = (object_sub + object_obj) / 2.0
            x_obj.append(x_obj[t] + self.GRU_object(
                GRU_input_feature_object, x_obj[t]))
            """update predicate features"""
            indices_sub = rel_inds[:, (0)]
            indices_obj = rel_inds[:, (1)]
            feat_sub2pred = torch.index_select(x_obj[t], 0, indices_sub)
            feat_obj2pred = torch.index_select(x_obj[t], 0, indices_obj)
            phrase_sub = self.gate_sub2pred(x_pred[t], feat_sub2pred)
            phrase_obj = self.gate_obj2pred(x_pred[t], feat_obj2pred)
            GRU_input_feature_phrase = phrase_sub / 2.0 + phrase_obj / 2.0
            x_pred.append(x_pred[t] + self.GRU_pred(
                GRU_input_feature_phrase, x_pred[t]))
        """compute results and losses"""
        obj_class_logits = self.obj_predictor(x_obj[-1].unsqueeze(2).
            unsqueeze(3))
        pred_class_logits = self.pred_predictor(x_pred[-1].unsqueeze(2).
            unsqueeze(3))
        if obj_class_logits is None:
            logits = torch.cat([proposal.get_field('logits') for proposal in
                proposals], 0)
            obj_class_labels = logits[:, 1:].max(1)[1] + 1
        else:
            obj_class_labels = obj_class_logits[:, 1:].max(1)[1] + 1
        return (x_obj[-1], x_pred[-1]
            ), obj_class_logits, pred_class_logits, obj_class_labels, rel_inds


def build_msdn_model(cfg, in_channels):
    return MSDN(cfg, in_channels)


class ROIRelationHead(torch.nn.Module):
    """
    Generic Relation Head class.
    """

    def __init__(self, cfg, in_channels):
        super(ROIRelationHead, self).__init__()
        self.cfg = cfg
        if cfg.MODEL.ALGORITHM == 'sg_baseline':
            self.rel_predictor = build_baseline_model(cfg, in_channels)
        elif cfg.MODEL.ALGORITHM == 'sg_imp':
            self.rel_predictor = build_imp_model(cfg, in_channels)
        elif cfg.MODEL.ALGORITHM == 'sg_msdn':
            self.rel_predictor = build_msdn_model(cfg, in_channels)
        elif cfg.MODEL.ALGORITHM == 'sg_grcnn':
            self.rel_predictor = build_grcnn_model(cfg, in_channels)
        elif cfg.MODEL.ALGORITHM == 'sg_reldn':
            self.rel_predictor = build_reldn_model(cfg, in_channels)
        self.post_processor = make_roi_relation_post_processor(cfg)
        self.loss_evaluator = make_roi_relation_loss_evaluator(cfg)
        if self.cfg.MODEL.USE_RELPN:
            self.relpn = make_relation_proposal_network(cfg)
        self.freq_dist = None
        self.use_bias = self.cfg.MODEL.ROI_RELATION_HEAD.USE_BIAS
        self.use_gt_boxes = self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOXES
        if self.use_gt_boxes:
            self.box_avgpool = nn.AdaptiveAvgPool2d(1)
            self.box_feature_extractor = make_roi_box_feature_extractor(cfg,
                in_channels)
            self.box_predictor = make_roi_box_predictor(cfg, self.
                box_feature_extractor.out_channels)
            self.box_post_processor = make_roi_box_post_processor(cfg)
            self._freeze_components(cfg)
        self.freq_dist_file = 'freq_prior.npy'
        self.freq_dist = np.load(self.freq_dist_file)
        if self.cfg.MODEL.USE_FREQ_PRIOR:
            self.freq_dist[:, :, (0)] = 0
            self.freq_bias = FrequencyBias(self.freq_dist)
        else:
            self.freq_dist[:, :, (0)] = 0
            self.freq_dist = np.log(self.freq_dist + 0.001)
            self.freq_dist = torch.from_numpy(self.freq_dist)

    def _freeze_components(self, cfg):
        for param in self.box_feature_extractor.parameters():
            param.requires_grad = False
        for param in self.box_predictor.parameters():
            param.requires_grad = False

    def _get_proposal_pairs(self, proposals):
        proposal_pairs = []
        for i, proposals_per_image in enumerate(proposals):
            box_subj = proposals_per_image.bbox
            box_obj = proposals_per_image.bbox
            box_subj = box_subj.unsqueeze(1).repeat(1, box_subj.shape[0], 1)
            box_obj = box_obj.unsqueeze(0).repeat(box_obj.shape[0], 1, 1)
            proposal_box_pairs = torch.cat((box_subj.view(-1, 4), box_obj.
                view(-1, 4)), 1)
            idx_subj = torch.arange(box_subj.shape[0]).view(-1, 1, 1).repeat(
                1, box_obj.shape[0], 1).to(proposals_per_image.bbox.device)
            idx_obj = torch.arange(box_obj.shape[0]).view(1, -1, 1).repeat(
                box_subj.shape[0], 1, 1).to(proposals_per_image.bbox.device)
            proposal_idx_pairs = torch.cat((idx_subj.view(-1, 1), idx_obj.
                view(-1, 1)), 1)
            keep_idx = (proposal_idx_pairs[:, (0)] != proposal_idx_pairs[:,
                (1)]).nonzero().view(-1)
            if self.cfg.MODEL.ROI_RELATION_HEAD.FILTER_NON_OVERLAP:
                ious = boxlist_iou(proposals_per_image, proposals_per_image
                    ).view(-1)
                ious = ious[keep_idx]
                keep_idx = keep_idx[(ious > 0).nonzero().view(-1)]
            proposal_idx_pairs = proposal_idx_pairs[keep_idx]
            proposal_box_pairs = proposal_box_pairs[keep_idx]
            proposal_pairs_per_image = BoxPairList(proposal_box_pairs,
                proposals_per_image.size, proposals_per_image.mode)
            proposal_pairs_per_image.add_field('idx_pairs', proposal_idx_pairs)
            proposal_pairs.append(proposal_pairs_per_image)
        return proposal_pairs

    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """
        if self.training and self.use_gt_boxes:
            targets_cp = [target.copy_with_fields(target.fields()) for
                target in targets]
            with torch.no_grad():
                x = self.box_feature_extractor(features, targets_cp)
                class_logits, box_regression = self.box_predictor(x)
            boxes_per_image = [len(proposal) for proposal in targets_cp]
            target_features = x.split(boxes_per_image, dim=0)
            for proposal, target_feature in zip(targets_cp, target_features):
                proposal.add_field('features', self.box_avgpool(target_feature)
                    )
            proposals_gt = self.box_post_processor((class_logits,
                box_regression), targets_cp, skip_nms=True)
            proposals = [cat_boxlist([proposal, proposal_gt]) for proposal,
                proposal_gt in zip(proposals, proposals_gt)]
        if self.training:
            if self.cfg.MODEL.USE_RELPN:
                proposal_pairs, loss_relpn = self.relpn(proposals, targets)
            else:
                proposal_pairs = self.loss_evaluator.subsample(proposals,
                    targets)
        else:
            with torch.no_grad():
                if self.cfg.MODEL.USE_RELPN:
                    proposal_pairs, relnesses = self.relpn(proposals)
                else:
                    proposal_pairs = self.loss_evaluator.subsample(proposals)
        if self.cfg.MODEL.USE_FREQ_PRIOR:
            """
            if use frequency prior, we directly use the statistics
            """
            x = None
            obj_class_logits = None
            _, obj_labels, im_inds = _get_tensor_from_boxlist(proposals,
                'labels')
            _, proposal_idx_pairs, im_inds_pairs = _get_tensor_from_boxlist(
                proposal_pairs, 'idx_pairs')
            rel_inds = _get_rel_inds(im_inds, im_inds_pairs, proposal_idx_pairs
                )
            pred_class_logits = self.freq_bias.index_with_labels(torch.
                stack((obj_labels[rel_inds[:, (0)]], obj_labels[rel_inds[:,
                (1)]]), 1))
        else:
            (x, obj_class_logits, pred_class_logits, obj_class_labels, rel_inds
                ) = self.rel_predictor(features, proposals, proposal_pairs)
            if self.use_bias:
                pred_class_logits = (pred_class_logits + self.freq_bias.
                    index_with_labels(torch.stack((obj_class_labels[
                    rel_inds[:, (0)]], obj_class_labels[rel_inds[:, (1)]]), 1))
                    )
        if not self.training:
            result = self.post_processor(pred_class_logits, proposal_pairs,
                use_freq_prior=self.cfg.MODEL.USE_FREQ_PRIOR)
            return x, result, {}
        loss_obj_classifier = 0
        if obj_class_logits is not None:
            loss_obj_classifier = self.loss_evaluator.obj_classification_loss(
                proposals, [obj_class_logits])
        if self.cfg.MODEL.USE_RELPN:
            idx = obj_class_labels[rel_inds[:, (0)]] * 151 + obj_class_labels[
                rel_inds[:, (1)]]
            freq_prior = self.freq_dist.view(-1, 51)[idx]
            loss_pred_classifier = self.relpn.pred_classification_loss([
                pred_class_logits], freq_prior=freq_prior)
            return x, proposal_pairs, dict(loss_obj_classifier=
                loss_obj_classifier, loss_relpn=loss_relpn,
                loss_pred_classifier=loss_pred_classifier)
        else:
            loss_pred_classifier = self.loss_evaluator([pred_class_logits])
            return x, proposal_pairs, dict(loss_obj_classifier=
                loss_obj_classifier, loss_pred_classifier=loss_pred_classifier)


def build_spatial_feature(cfg, dim=0):
    return SpatialFeature(cfg, dim)


class RelDN(nn.Module):

    def __init__(self, cfg, in_channels, eps=1e-10):
        super(RelDN, self).__init__()
        self.cfg = cfg
        self.dim = 512
        self.update_step = cfg.MODEL.ROI_RELATION_HEAD.IMP_FEATURE_UPDATE_STEP
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.pred_feature_extractor = make_roi_relation_feature_extractor(cfg,
            in_channels)
        num_classes = cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.obj_embedding = nn.Sequential(nn.Linear(self.
            pred_feature_extractor.out_channels, self.dim), nn.ReLU(True),
            nn.Linear(self.dim, self.dim))
        self.pred_embedding = nn.Sequential(nn.Linear(self.
            pred_feature_extractor.out_channels, self.dim), nn.ReLU(True),
            nn.Linear(self.dim, self.dim))
        self.rel_embedding = nn.Sequential(nn.Linear(3 * self.dim, self.dim
            ), nn.ReLU(True), nn.Linear(self.dim, self.dim), nn.ReLU(True))
        self.rel_spatial_feat = build_spatial_feature(cfg, self.dim)
        self.rel_subj_predictor = make_roi_relation_predictor(cfg, 512)
        self.rel_obj_predictor = make_roi_relation_predictor(cfg, 512)
        self.rel_pred_predictor = make_roi_relation_predictor(cfg, 512)
        self.rel_spt_predictor = nn.Linear(64, num_classes)
        self.freq_dist = torch.from_numpy(np.load('freq_prior.npy'))
        self.pred_dist = 10 * self.freq_dist
        self.num_objs = self.pred_dist.shape[0]
        self.pred_dist = torch.FloatTensor(self.pred_dist).view(-1, self.
            pred_dist.shape[2])

    def _get_map_idxs(self, proposals, proposal_pairs):
        rel_inds = []
        offset = 0
        for proposal, proposal_pair in zip(proposals, proposal_pairs):
            rel_ind_i = proposal_pair.get_field('idx_pairs').detach().clone()
            rel_ind_i += offset
            offset += len(proposal)
            rel_inds.append(rel_ind_i)
        rel_inds = torch.cat(rel_inds, 0)
        subj_pred_map = rel_inds.new(sum([len(proposal) for proposal in
            proposals]), rel_inds.shape[0]).fill_(0).float().detach()
        obj_pred_map = rel_inds.new(sum([len(proposal) for proposal in
            proposals]), rel_inds.shape[0]).fill_(0).float().detach()
        subj_pred_map.scatter_(0, rel_inds[:, (0)].contiguous().view(1, -1), 1)
        obj_pred_map.scatter_(0, rel_inds[:, (1)].contiguous().view(1, -1), 1)
        return rel_inds, subj_pred_map, obj_pred_map

    def forward(self, features, proposals, proposal_pairs):
        obj_class_logits = None
        rel_inds, subj_pred_map, obj_pred_map = self._get_map_idxs(proposals,
            proposal_pairs)
        x_obj = torch.cat([proposal.get_field('features').detach() for
            proposal in proposals], 0)
        x_pred, _ = self.pred_feature_extractor(features, proposals,
            proposal_pairs)
        x_pred = self.avgpool(x_pred)
        x_obj = x_obj.view(x_obj.size(0), -1)
        x_pred = x_pred.view(x_pred.size(0), -1)
        x_obj = self.obj_embedding(x_obj)
        x_pred = self.pred_embedding(x_pred)
        sub_vert = x_obj[rel_inds[:, (0)]]
        obj_vert = x_obj[rel_inds[:, (1)]]
        """compute visual scores"""
        rel_subj_class_logits = self.rel_subj_predictor(sub_vert.unsqueeze(
            2).unsqueeze(3))
        rel_obj_class_logits = self.rel_obj_predictor(obj_vert.unsqueeze(2)
            .unsqueeze(3))
        x_rel = torch.cat([sub_vert, obj_vert, x_pred], 1)
        x_rel = self.rel_embedding(x_rel)
        rel_pred_class_logits = self.rel_pred_predictor(x_rel.unsqueeze(2).
            unsqueeze(3))
        rel_vis_class_logits = (rel_pred_class_logits +
            rel_subj_class_logits + rel_obj_class_logits)
        """compute spatial scores"""
        edge_spt_feats = self.rel_spatial_feat(proposal_pairs)
        rel_spt_class_logits = self.rel_spt_predictor(edge_spt_feats)
        """compute semantic scores"""
        rel_sem_class_logits = []
        for proposal_per_image, proposal_pairs_per_image in zip(proposals,
            proposal_pairs):
            obj_labels = proposal_per_image.get_field('labels').detach()
            rel_ind_i = proposal_pairs_per_image.get_field('idx_pairs').detach(
                )
            subj_vert_labels = obj_labels[rel_ind_i[:, (0)]]
            obj_vert_labels = obj_labels[rel_ind_i[:, (1)]]
            class_logits_per_image = self.pred_dist[subj_vert_labels * self
                .num_objs + obj_vert_labels]
            rel_sem_class_logits.append(class_logits_per_image)
        rel_sem_class_logits = torch.cat(rel_sem_class_logits, 0)
        rel_class_logits = (rel_vis_class_logits + rel_sem_class_logits +
            rel_spt_class_logits)
        if obj_class_logits is None:
            logits = torch.cat([proposal.get_field('logits') for proposal in
                proposals], 0)
            obj_class_labels = logits[:, 1:].max(1)[1] + 1
        else:
            obj_class_labels = obj_class_logits[:, 1:].max(1)[1] + 1
        return (x_obj, x_pred
            ), obj_class_logits, rel_class_logits, obj_class_labels, rel_inds


def bbox_transform_inv(boxes, gt_boxes, weights=(1.0, 1.0, 1.0, 1.0)):
    """Inverse transform that computes target bounding-box regression deltas
    given proposal boxes and ground-truth boxes. The weights argument should be
    a 4-tuple of multiplicative weights that are applied to the regression
    target.

    In older versions of this code (and in py-faster-rcnn), the weights were set
    such that the regression deltas would have unit standard deviation on the
    training dataset. Presently, rather than computing these statistics exactly,
    we use a fixed set of weights (10., 10., 5., 5.) by default. These are
    approximately the weights one would get from COCO using the previous unit
    stdev heuristic.
    """
    ex_widths = boxes[:, (2)] - boxes[:, (0)] + 1.0
    ex_heights = boxes[:, (3)] - boxes[:, (1)] + 1.0
    ex_ctr_x = boxes[:, (0)] + 0.5 * ex_widths
    ex_ctr_y = boxes[:, (1)] + 0.5 * ex_heights
    gt_widths = gt_boxes[:, (2)] - gt_boxes[:, (0)] + 1.0
    gt_heights = gt_boxes[:, (3)] - gt_boxes[:, (1)] + 1.0
    gt_ctr_x = gt_boxes[:, (0)] + 0.5 * gt_widths
    gt_ctr_y = gt_boxes[:, (1)] + 0.5 * gt_heights
    wx, wy, ww, wh = weights
    targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = ww * np.log(gt_widths / ex_widths)
    targets_dh = wh * np.log(gt_heights / ex_heights)
    targets = np.vstack((targets_dx, targets_dy, targets_dw, targets_dh)
        ).transpose()
    return targets


def boxes_union(boxes1, boxes2):
    assert boxes1.shape == boxes2.shape
    xmin = np.minimum(boxes1[:, (0)], boxes2[:, (0)])
    ymin = np.minimum(boxes1[:, (1)], boxes2[:, (1)])
    xmax = np.maximum(boxes1[:, (2)], boxes2[:, (2)])
    ymax = np.maximum(boxes1[:, (3)], boxes2[:, (3)])
    return np.vstack((xmin, ymin, xmax, ymax)).transpose()


class SpatialFeature(nn.Module):

    def __init__(self, cfg, dim):
        super(SpatialFeature, self).__init__()
        self.model = nn.Sequential(nn.Linear(28, 64), nn.LeakyReLU(0.1), nn
            .Linear(64, 64), nn.LeakyReLU(0.1))

    def _get_pair_feature(self, boxes1, boxes2):
        delta_1 = bbox_transform_inv(boxes1, boxes2)
        delta_2 = bbox_transform_inv(boxes2, boxes1)
        spt_feat = np.hstack((delta_1, delta_2[:, :2]))
        return spt_feat

    def _get_box_feature(self, boxes, width, height):
        f1 = boxes[:, (0)] / width
        f2 = boxes[:, (1)] / height
        f3 = boxes[:, (2)] / width
        f4 = boxes[:, (3)] / height
        f5 = (boxes[:, (2)] - boxes[:, (0)] + 1) * (boxes[:, (3)] - boxes[:,
            (1)] + 1) / (width * height)
        return np.vstack((f1, f2, f3, f4, f5)).transpose()

    def _get_spt_features(self, boxes1, boxes2, width, height):
        boxes_u = boxes_union(boxes1, boxes2)
        spt_feat_1 = self._get_box_feature(boxes1, width, height)
        spt_feat_2 = self._get_box_feature(boxes2, width, height)
        spt_feat_12 = self._get_pair_feature(boxes1, boxes2)
        spt_feat_1u = self._get_pair_feature(boxes1, boxes_u)
        spt_feat_u2 = self._get_pair_feature(boxes_u, boxes2)
        return np.hstack((spt_feat_12, spt_feat_1u, spt_feat_u2, spt_feat_1,
            spt_feat_2))

    def forward(self, proposal_pairs):
        spt_feats = []
        for proposal_pair in proposal_pairs:
            boxes_subj = proposal_pair.bbox[:, :4]
            boxes_obj = proposal_pair.bbox[:, 4:]
            spt_feat = self._get_spt_features(boxes_subj.cpu().numpy(),
                boxes_obj.cpu().numpy(), proposal_pair.size[0],
                proposal_pair.size[1])
            spt_feat = torch.from_numpy(spt_feat).to(boxes_subj.device)
            spt_feats.append(spt_feat)
        spt_feats = torch.cat(spt_feats, 0).float()
        spt_feats = self.model(spt_feats)
        return spt_feats


class VisualFeature(nn.Module):

    def __init__(self, dim):
        self.subj_branch = nn.Sequential(nn.Linear())

    def forward(self, subj_feat, obj_feat, rel_feat):
        pass


def box_pos_encoder(bboxes, width, height):
    """
    bounding box encoding
    """
    bboxes_enc = bboxes.clone()
    dim0 = bboxes_enc[:, (0)] / width
    dim1 = bboxes_enc[:, (1)] / height
    dim2 = bboxes_enc[:, (2)] / width
    dim3 = bboxes_enc[:, (3)] / height
    dim4 = (bboxes_enc[:, (2)] - bboxes_enc[:, (0)]) * (bboxes_enc[:, (3)] -
        bboxes_enc[:, (1)]) / height / width
    dim5 = (bboxes_enc[:, (3)] - bboxes_enc[:, (1)]) / (bboxes_enc[:, (2)] -
        bboxes_enc[:, (0)] + 1)
    return torch.stack((dim0, dim1, dim2, dim3, dim4, dim5), 1)


class Relationshipness(nn.Module):
    """
    compute relationshipness between subjects and objects
    """

    def __init__(self, dim, pos_encoding=False):
        super(Relationshipness, self).__init__()
        self.subj_proj = nn.Sequential(nn.Linear(dim, 64), nn.ReLU(True),
            nn.Linear(64, 64))
        self.obj_prof = nn.Sequential(nn.Linear(dim, 64), nn.ReLU(True), nn
            .Linear(64, 64))
        self.pos_encoding = False
        if pos_encoding:
            self.pos_encoding = True
            self.sub_pos_encoder = nn.Sequential(nn.Linear(6, 64), nn.ReLU(
                True), nn.Linear(64, 64))
            self.obj_pos_encoder = nn.Sequential(nn.Linear(6, 64), nn.ReLU(
                True), nn.Linear(64, 64))

    def forward(self, x, bbox=None, imsize=None):
        x_subj = self.subj_proj(x)
        x_obj = self.obj_prof(x)
        scores = torch.mm(x_subj, x_obj.t())
        if self.pos_encoding:
            pos = box_pos_encoder(bbox, imsize[0], imsize[1])
            pos_subj = self.sub_pos_encoder(pos)
            pos_obj = self.obj_pos_encoder(pos)
            pos_scores = torch.mm(pos_subj, pos_obj.t())
            scores = scores + pos_scores
        relness = torch.sigmoid(scores)
        return relness


class Relationshipnessv2(nn.Module):
    """
    compute relationshipness between subjects and objects
    """

    def __init__(self, dim, pos_encoding=False):
        super(Relationshipnessv2, self).__init__()
        self.subj_proj = nn.Sequential(nn.Linear(dim, 64), nn.ReLU(True),
            nn.Linear(64, 64))
        self.obj_proj = nn.Sequential(nn.Linear(dim, 64), nn.ReLU(True), nn
            .Linear(64, 64))
        self.pos_encoding = False
        if pos_encoding:
            self.pos_encoding = True
            self.sub_pos_encoder = nn.Sequential(nn.Linear(6, 64), nn.ReLU(
                True), nn.Linear(64, 64))
            self.obj_pos_encoder = nn.Sequential(nn.Linear(6, 64), nn.ReLU(
                True), nn.Linear(64, 64))
        self.self_att_subj = MultiHeadAttention(8, 64)
        self.self_att_obj = MultiHeadAttention(8, 64)
        self.self_att_pos_subj = MultiHeadAttention(8, 64)
        self.self_att_pos_obj = MultiHeadAttention(8, 64)

    def forward(self, x, bbox=None, imsize=None):
        x_subj = self.subj_proj(x)
        x_subj = self.self_att_subj(x_subj, x_subj, x_subj).squeeze(1)
        x_obj = self.obj_proj(x)
        x_obj = self.self_att_obj(x_obj, x_obj, x_obj).squeeze(1)
        scores = torch.mm(x_subj, x_obj.t())
        if self.pos_encoding:
            pos = box_pos_encoder(bbox, imsize[0], imsize[1])
            pos_subj = self.sub_pos_encoder(pos)
            pos_subj = self.self_att_pos_subj(pos_subj, pos_subj, pos_subj
                ).squeeze(1)
            pos_obj = self.obj_pos_encoder(pos)
            pos_obj = self.self_att_pos_obj(pos_obj, pos_obj, pos_obj).squeeze(
                1)
            pos_scores = torch.mm(pos_subj, pos_obj.t())
            scores = scores + pos_scores
        relness = torch.sigmoid(scores)
        return relness


class RelPN(nn.Module):

    def __init__(self, cfg, proposal_matcher, fg_bg_pair_sampler, box_coder,
        cls_agnostic_bbox_reg=False, use_matched_pairs_only=False,
        minimal_matched_pairs=0):
        super(RelPN, self).__init__()
        self.cfg = cfg
        self.proposal_pair_matcher = proposal_matcher
        self.fg_bg_pair_sampler = fg_bg_pair_sampler
        self.box_coder = box_coder
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg
        self.use_matched_pairs_only = use_matched_pairs_only
        self.minimal_matched_pairs = minimal_matched_pairs
        self.relationshipness = Relationshipness(self.cfg.MODEL.
            ROI_BOX_HEAD.NUM_CLASSES, pos_encoding=True)

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        temp = []
        target_box_pairs = []
        for i in range(match_quality_matrix.shape[0]):
            for j in range(match_quality_matrix.shape[0]):
                match_i = match_quality_matrix[i].view(-1, 1)
                match_j = match_quality_matrix[j].view(1, -1)
                match_ij = (match_i + match_j) / 2
                match_ij = match_ij.view(-1)
                temp.append(match_ij)
                boxi = target.bbox[i]
                boxj = target.bbox[j]
                box_pair = torch.cat((boxi, boxj), 0)
                target_box_pairs.append(box_pair)
        match_pair_quality_matrix = torch.stack(temp, 0).view(len(temp), -1)
        target_box_pairs = torch.stack(target_box_pairs, 0)
        target_pair = BoxPairList(target_box_pairs, target.size, target.mode)
        target_pair.add_field('labels', target.get_field('pred_labels').
            view(-1))
        box_subj = proposal.bbox
        box_obj = proposal.bbox
        box_subj = box_subj.unsqueeze(1).repeat(1, box_subj.shape[0], 1)
        box_obj = box_obj.unsqueeze(0).repeat(box_obj.shape[0], 1, 1)
        proposal_box_pairs = torch.cat((box_subj.view(-1, 4), box_obj.view(
            -1, 4)), 1)
        idx_subj = torch.arange(box_subj.shape[0]).view(-1, 1, 1).repeat(1,
            box_obj.shape[0], 1).to(proposal.bbox.device)
        idx_obj = torch.arange(box_obj.shape[0]).view(1, -1, 1).repeat(box_subj
            .shape[0], 1, 1).to(proposal.bbox.device)
        proposal_idx_pairs = torch.cat((idx_subj.view(-1, 1), idx_obj.view(
            -1, 1)), 1)
        proposal_pairs = BoxPairList(proposal_box_pairs, proposal.size,
            proposal.mode)
        proposal_pairs.add_field('idx_pairs', proposal_idx_pairs)
        matched_idxs = self.proposal_pair_matcher(match_pair_quality_matrix)
        if self.use_matched_pairs_only and (matched_idxs >= 0).sum(
            ) > self.minimal_matched_pairs:
            proposal_pairs = proposal_pairs[matched_idxs >= 0]
            matched_idxs = matched_idxs[matched_idxs >= 0]
        matched_targets = target_pair[matched_idxs.clamp(min=0)]
        matched_targets.add_field('matched_idxs', matched_idxs)
        return matched_targets, proposal_pairs

    def prepare_targets(self, proposals, targets):
        labels = []
        proposal_pairs = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets, proposal_pairs_per_image = (self.
                match_targets_to_proposals(proposals_per_image,
                targets_per_image))
            matched_idxs = matched_targets.get_field('matched_idxs')
            labels_per_image = matched_targets.get_field('labels')
            labels_per_image = labels_per_image.to(dtype=torch.int64)
            bg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_inds] = 0
            ignore_inds = matched_idxs == Matcher.BETWEEN_THRESHOLDS
            labels_per_image[ignore_inds] = -1
            labels.append(labels_per_image)
            proposal_pairs.append(proposal_pairs_per_image)
        return labels, proposal_pairs

    def _relpnsample_train(self, proposals, targets):
        """
        perform relpn based sampling during training
        """
        labels, proposal_pairs = self.prepare_targets(proposals, targets)
        proposal_pairs = list(proposal_pairs)
        for labels_per_image, proposal_pairs_per_image in zip(labels,
            proposal_pairs):
            proposal_pairs_per_image.add_field('labels', labels_per_image)
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_pair_sampler(labels)
        losses = 0
        for img_idx, (proposals_per_image, pos_inds_img, neg_inds_img
            ) in enumerate(zip(proposals, sampled_pos_inds, sampled_neg_inds)):
            obj_logits = proposals_per_image.get_field('logits')
            obj_bboxes = proposals_per_image.bbox
            relness = self.relationshipness(obj_logits, obj_bboxes,
                proposals_per_image.size)
            relness_sorted, order = torch.sort(relness.view(-1), descending
                =True)
            img_sampled_inds = order[:self.cfg.MODEL.ROI_RELATION_HEAD.
                BATCH_SIZE_PER_IMAGE].view(-1)
            proposal_pairs_per_image = proposal_pairs[img_idx][img_sampled_inds
                ]
            proposal_pairs[img_idx] = proposal_pairs_per_image
            losses += F.binary_cross_entropy(relness.view(-1, 1), (labels[
                img_idx] > 0).view(-1, 1).float())
        self._proposal_pairs = proposal_pairs
        return proposal_pairs, losses

    def _fullsample_test(self, proposals):
        """
        This method get all subject-object pairs, and return the proposals.
        Note: this function keeps a state.

        Arguments:
            proposals (list[BoxList])
        """
        proposal_pairs = []
        for i, proposals_per_image in enumerate(proposals):
            box_subj = proposals_per_image.bbox
            box_obj = proposals_per_image.bbox
            box_subj = box_subj.unsqueeze(1).repeat(1, box_subj.shape[0], 1)
            box_obj = box_obj.unsqueeze(0).repeat(box_obj.shape[0], 1, 1)
            proposal_box_pairs = torch.cat((box_subj.view(-1, 4), box_obj.
                view(-1, 4)), 1)
            idx_subj = torch.arange(box_subj.shape[0]).view(-1, 1, 1).repeat(
                1, box_obj.shape[0], 1).to(proposals_per_image.bbox.device)
            idx_obj = torch.arange(box_obj.shape[0]).view(1, -1, 1).repeat(
                box_subj.shape[0], 1, 1).to(proposals_per_image.bbox.device)
            proposal_idx_pairs = torch.cat((idx_subj.view(-1, 1), idx_obj.
                view(-1, 1)), 1)
            keep_idx = (proposal_idx_pairs[:, (0)] != proposal_idx_pairs[:,
                (1)]).nonzero().view(-1)
            if self.cfg.MODEL.ROI_RELATION_HEAD.FILTER_NON_OVERLAP:
                ious = boxlist_iou(proposals_per_image, proposals_per_image
                    ).view(-1)
                ious = ious[keep_idx]
                keep_idx = keep_idx[(ious > 0).nonzero().view(-1)]
            proposal_idx_pairs = proposal_idx_pairs[keep_idx]
            proposal_box_pairs = proposal_box_pairs[keep_idx]
            proposal_pairs_per_image = BoxPairList(proposal_box_pairs,
                proposals_per_image.size, proposals_per_image.mode)
            proposal_pairs_per_image.add_field('idx_pairs', proposal_idx_pairs)
            proposal_pairs.append(proposal_pairs_per_image)
        return proposal_pairs

    def _relpnsample_test(self, proposals):
        """
        perform relpn based sampling during testing
        """
        proposals[0] = proposals[0]
        proposal_pairs = self._fullsample_test(proposals)
        proposal_pairs = list(proposal_pairs)
        relnesses = []
        for img_idx, proposals_per_image in enumerate(proposals):
            obj_logits = proposals_per_image.get_field('logits')
            obj_bboxes = proposals_per_image.bbox
            relness = self.relationshipness(obj_logits, obj_bboxes,
                proposals_per_image.size)
            keep_idx = (1 - torch.eye(obj_logits.shape[0]).to(relness.device)
                ).view(-1).nonzero().view(-1)
            if self.cfg.MODEL.ROI_RELATION_HEAD.FILTER_NON_OVERLAP:
                ious = boxlist_iou(proposals_per_image, proposals_per_image
                    ).view(-1)
                ious = ious[keep_idx]
                keep_idx = keep_idx[(ious > 0).nonzero().view(-1)]
            relness = relness.view(-1)[keep_idx]
            relness_sorted, order = torch.sort(relness.view(-1), descending
                =True)
            img_sampled_inds = order[:self.cfg.MODEL.ROI_RELATION_HEAD.
                BATCH_SIZE_PER_IMAGE].view(-1)
            relness = relness_sorted[:self.cfg.MODEL.ROI_RELATION_HEAD.
                BATCH_SIZE_PER_IMAGE].view(-1)
            proposal_pairs_per_image = proposal_pairs[img_idx][img_sampled_inds
                ]
            proposal_pairs[img_idx] = proposal_pairs_per_image
            relnesses.append(relness)
        self._proposal_pairs = proposal_pairs
        return proposal_pairs, relnesses

    def forward(self, proposals, targets=None):
        """
        This method performs the positive/negative sampling, and return
        the sampled proposals.
        Note: this function keeps a state.

        Arguments:
            proposals (list[BoxList])
            targets (list[BoxList])
        """
        if self.training:
            return self._relpnsample_train(proposals, targets)
        else:
            return self._relpnsample_test(proposals)

    def pred_classification_loss(self, class_logits, freq_prior=None):
        """
        Computes the loss for Faster R-CNN.
        This requires that the subsample method has been called beforehand.

        Arguments:
            class_logits (list[Tensor])

        Returns:
            classification_loss (Tensor)
        """
        class_logits = cat(class_logits, dim=0)
        device = class_logits.device
        if not hasattr(self, '_proposal_pairs'):
            raise RuntimeError('subsample needs to be called before')
        proposals = self._proposal_pairs
        labels = cat([proposal.get_field('labels') for proposal in
            proposals], dim=0)
        rel_fg_cnt = len(labels.nonzero())
        rel_bg_cnt = labels.shape[0] - rel_fg_cnt
        ce_weights = labels.new(class_logits.size(1)).fill_(1).float()
        ce_weights[0] = float(rel_fg_cnt) / (rel_bg_cnt + 1e-05)
        classification_loss = F.cross_entropy(class_logits, labels, weight=
            ce_weights)
        return classification_loss


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


class FrequencyBias(nn.Module):
    """
    The goal of this is to provide a simplified way of computing
    P(predicate | obj1, obj2, img).
    """

    def __init__(self, pred_dist):
        super(FrequencyBias, self).__init__()
        self.num_objs = pred_dist.shape[0]
        pred_dist = torch.FloatTensor(pred_dist).view(-1, pred_dist.shape[2])
        self.obj_baseline = nn.Embedding(pred_dist.size(0), pred_dist.size(1))
        self.obj_baseline.weight.data = pred_dist

    def index_with_labels(self, labels):
        """
        :param labels: [batch_size, 2]
        :return:
        """
        return self.obj_baseline(labels[:, (0)] * self.num_objs + labels[:,
            (1)])

    def forward(self, obj_cands0, obj_cands1):
        """
        :param obj_cands0: [batch_size, 151] prob distibution over cands.
        :param obj_cands1: [batch_size, 151] prob distibution over cands.
        :return: [batch_size, #predicates] array, which contains potentials for
        each possibility
        """
        joint_cands = obj_cands0[:, :, (None)] * obj_cands1[:, (None)]
        baseline = joint_cands.view(joint_cands.size(0), -1
            ) @ self.obj_baseline.weight
        return baseline


class BalancedPositiveNegativeSampler(object):
    """
    This class samples batches, ensuring that they contain a fixed proportion of positives
    """

    def __init__(self, batch_size_per_image, positive_fraction):
        """
        Arguments:
            batch_size_per_image (int): number of elements to be selected per image
            positive_fraction (float): percentage of positive elements per batch
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


def make_roi_box_loss_evaluator(cfg):
    matcher = Matcher(cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD, cfg.MODEL.
        ROI_HEADS.BG_IOU_THRESHOLD, allow_low_quality_matches=False)
    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights)
    fg_bg_sampler = BalancedPositiveNegativeSampler(cfg.MODEL.ROI_HEADS.
        BATCH_SIZE_PER_IMAGE, cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION)
    cls_agnostic_bbox_reg = cfg.MODEL.CLS_AGNOSTIC_BBOX_REG
    loss_evaluator = FastRCNNLossComputation(matcher, fg_bg_sampler,
        box_coder, cls_agnostic_bbox_reg)
    return loss_evaluator


class ROIBoxHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg, in_channels):
        super(ROIBoxHead, self).__init__()
        self.cfg = cfg
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.feature_extractor = make_roi_box_feature_extractor(cfg,
            in_channels)
        self.predictor = make_roi_box_predictor(cfg, self.feature_extractor
            .out_channels)
        self.post_processor = make_roi_box_post_processor(cfg)
        self.loss_evaluator = make_roi_box_loss_evaluator(cfg)

    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """
        if self.training:
            with torch.no_grad():
                proposals = self.loss_evaluator.subsample(proposals, targets)
        x = self.feature_extractor(features, proposals)
        class_logits, box_regression = self.predictor(x)
        boxes_per_image = [len(proposal) for proposal in proposals]
        features = x.split(boxes_per_image, dim=0)
        for proposal, feature in zip(proposals, features):
            proposal.add_field('features', self.avgpool(feature))
        if not self.training:
            result = self.post_processor((class_logits, box_regression),
                proposals)
            if targets:
                result = self.loss_evaluator.prepare_labels(result, targets)
            return x, result, {}
        loss_classifier, loss_box_reg = self.loss_evaluator([class_logits],
            [box_regression])
        class_logits = class_logits.split(boxes_per_image, dim=0)
        for proposal, class_logit in zip(proposals, class_logits):
            proposal.add_field('logits', class_logit)
        return x, proposals, dict(loss_classifier=loss_classifier,
            loss_box_reg=loss_box_reg)


class PostProcessor(nn.Module):
    """
    From a set of classification scores, box regression and proposals,
    computes the post-processed boxes, and applies NMS to obtain the
    final results
    """

    def __init__(self, score_thresh=0.05, nms=0.5, detections_per_img=100,
        min_detections_per_img=0, box_coder=None, cls_agnostic_bbox_reg=
        False, bbox_aug_enabled=False, relation_on=False):
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
        self.min_detections_per_img = min_detections_per_img
        if box_coder is None:
            box_coder = BoxCoder(weights=(10.0, 10.0, 5.0, 5.0))
        self.box_coder = box_coder
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg
        self.bbox_aug_enabled = bbox_aug_enabled
        self.relation_on = relation_on

    def forward(self, x, boxes, skip_nms=False):
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
        class_logit, box_regression = x
        class_prob = F.softmax(class_logit, -1)
        image_shapes = [box.size for box in boxes]
        boxes_per_image = [len(box) for box in boxes]
        features = [box.get_field('features') for box in boxes]
        concat_boxes = torch.cat([a.bbox for a in boxes], dim=0)
        if not skip_nms:
            if self.cls_agnostic_bbox_reg:
                box_regression = box_regression[:, -4:]
            proposals = self.box_coder.decode(box_regression.view(sum(
                boxes_per_image), -1), concat_boxes)
            if self.cls_agnostic_bbox_reg:
                proposals = proposals.repeat(1, class_prob.shape[1])
            proposals = proposals.split(boxes_per_image, dim=0)
        else:
            proposals = concat_boxes.split(boxes_per_image, dim=0)
        num_classes = class_prob.shape[1]
        class_prob = class_prob.split(boxes_per_image, dim=0)
        class_logit = class_logit.split(boxes_per_image, dim=0)
        results = []
        idx = 0
        for prob, logit, boxes_per_img, features_per_img, image_shape in zip(
            class_prob, class_logit, proposals, features, image_shapes):
            if not self.bbox_aug_enabled and not skip_nms:
                boxlist = self.prepare_boxlist(boxes_per_img,
                    features_per_img, prob, logit, image_shape)
                boxlist = boxlist.clip_to_image(remove_empty=False)
                if not self.relation_on:
                    boxlist_filtered = self.filter_results(boxlist, num_classes
                        )
                else:
                    boxlist_filtered = self.filter_results_nm(boxlist,
                        num_classes)
                    score_thresh = 0.05
                    while len(boxlist_filtered) < self.min_detections_per_img:
                        score_thresh /= 2.0
                        None
                        boxlist_filtered = self.filter_results_nm(boxlist,
                            num_classes, thresh=score_thresh)
            else:
                boxlist = BoxList(boxes_per_img, image_shape, mode='xyxy')
                boxlist.add_field('scores', prob[:, 1:].max(1)[0])
                boxlist.add_field('logits', logit)
                boxlist.add_field('features', features_per_img)
                boxlist.add_field('labels', boxes[idx].get_field('labels'))
                boxlist.add_field('regression_targets', boxes[idx].bbox.
                    clone().fill_(0.0))
                boxlist_filtered = boxlist
                idx += 1
            if len(boxlist) == 0:
                raise ValueError('boxlist shoud not be empty!')
            results.append(boxlist_filtered)
        return results

    def prepare_boxlist(self, boxes, features, scores, logits, image_shape):
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
        boxlist.add_field('logits', logits)
        boxlist.add_field('features', features)
        return boxlist

    def filter_results(self, boxlist, num_classes):
        """Returns bounding-box detection results by thresholding on scores and
        applying non-maximum suppression (NMS).
        """
        boxes = boxlist.bbox.reshape(-1, num_classes * 4)
        scores = boxlist.get_field('scores').reshape(-1, num_classes)
        logits = boxlist.get_field('logits').reshape(-1, num_classes)
        features = boxlist.get_field('features')
        device = scores.device
        result = []
        inds_all = scores > self.score_thresh
        for j in range(1, num_classes):
            inds = inds_all[:, (j)].nonzero().squeeze(1)
            scores_j = scores[inds, j]
            features_j = features[inds]
            boxes_j = boxes[(inds), j * 4:(j + 1) * 4]
            boxlist_for_class = BoxList(boxes_j, boxlist.size, mode='xyxy')
            boxlist_for_class.add_field('scores', scores_j)
            boxlist_for_class.add_field('features', features_j)
            boxlist_for_class = boxlist_nms(boxlist_for_class, self.nms)
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

    def filter_results_nm(self, boxlist, num_classes, thresh=0.05):
        """Returns bounding-box detection results by thresholding on scores and
        applying non-maximum suppression (NMS). Similar to Neural-Motif Network
        """
        boxes = boxlist.bbox.reshape(-1, num_classes * 4)
        scores = boxlist.get_field('scores').reshape(-1, num_classes)
        logits = boxlist.get_field('logits').reshape(-1, num_classes)
        features = boxlist.get_field('features')
        valid_cls = (scores[:, 1:].max(0)[0] > thresh).nonzero() + 1
        nms_mask = scores.clone()
        nms_mask.zero_()
        device = scores.device
        result = []
        inds_all = scores > self.score_thresh
        for j in valid_cls.view(-1).cpu():
            scores_j = scores[:, (j)]
            boxes_j = boxes[:, j * 4:(j + 1) * 4]
            boxlist_for_class = BoxList(boxes_j, boxlist.size, mode='xyxy')
            boxlist_for_class.add_field('scores', scores_j)
            boxlist_for_class.add_field('idxs', torch.arange(0, scores.
                shape[0]).long())
            boxlist_for_class = boxlist_nms(boxlist_for_class, 0.3)
            nms_mask[:, (j)][boxlist_for_class.get_field('idxs')] = 1
            num_labels = len(boxlist_for_class)
            boxlist_for_class.add_field('labels', torch.full((num_labels,),
                j, dtype=torch.int64, device=device))
            result.append(boxlist_for_class)
        dists_all = nms_mask * scores
        scores_pre, labels_pre = dists_all.max(1)
        inds_all = scores_pre.nonzero()
        assert inds_all.dim() != 0
        inds_all = inds_all.squeeze(1)
        labels_all = labels_pre[inds_all]
        scores_all = scores_pre[inds_all]
        features_all = features[inds_all]
        logits_all = logits[inds_all]
        box_inds_all = inds_all * scores.shape[1] + labels_all
        result = BoxList(boxlist.bbox.view(-1, 4)[box_inds_all], boxlist.
            size, mode='xyxy')
        result.add_field('labels', labels_all)
        result.add_field('scores', scores_all)
        result.add_field('logits', logits_all)
        result.add_field('features', features_all)
        number_of_detections = len(result)
        vs, idx = torch.sort(scores_all, dim=0, descending=True)
        idx = idx[vs > thresh]
        if self.detections_per_img < idx.size(0):
            idx = idx[:self.detections_per_img]
        result = result[idx]
        return result


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
        if (cfg.MODEL.KEYPOINT_ON and cfg.MODEL.ROI_KEYPOINT_HEAD.
            SHARE_BOX_FEATURE_EXTRACTOR):
            self.keypoint.feature_extractor = self.box.feature_extractor

    def forward(self, features, proposals, targets=None):
        x, detections, loss_box = self.box(features, proposals, targets)
        return x, detections, loss_box


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
            cell_anchors = [generate_anchors(anchor_stride, size if
                isinstance(size, (tuple, list)) else (size,), aspect_ratios
                ).float() for anchor_stride, size in zip(anchor_strides, sizes)
                ]
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
        for i, (image_height, image_width) in enumerate(image_list.image_sizes
            ):
            anchors_in_image = []
            for anchors_per_feature_map in anchors_over_all_feature_maps:
                boxlist = BoxList(anchors_per_feature_map, (image_width,
                    image_height), mode='xyxy')
                self.add_visibility_to(boxlist)
                anchors_in_image.append(boxlist)
            anchors.append(anchors_in_image)
        return anchors


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


def permute_and_flatten(layer, N, A, C, H, W):
    layer = layer.view(N, -1, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2)
    layer = layer.reshape(N, -1, C)
    return layer


class RPNPostProcessor(torch.nn.Module):
    """
    Performs post-processing on the outputs of the RPN boxes, before feeding the
    proposals to the heads
    """

    def __init__(self, pre_nms_top_n, post_nms_top_n, nms_thresh, min_size,
        box_coder=None, fpn_post_nms_top_n=None, fpn_post_nms_per_batch=True):
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
        self.fpn_post_nms_per_batch = fpn_post_nms_per_batch

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
        objectness = permute_and_flatten(objectness, N, A, 1, H, W).view(N, -1)
        objectness = objectness.sigmoid()
        box_regression = permute_and_flatten(box_regression, N, A, 4, H, W)
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
        if self.training and self.fpn_post_nms_per_batch:
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


class RetinaNetHead(torch.nn.Module):
    """
    Adds a RetinNet head with classification and regression heads
    """

    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
            num_anchors (int): number of anchors to be predicted
        """
        super(RetinaNetHead, self).__init__()
        num_classes = cfg.MODEL.RETINANET.NUM_CLASSES - 1
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
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)
        prior_prob = cfg.MODEL.RETINANET.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

    def forward(self, x):
        logits = []
        bbox_reg = []
        for feature in x:
            logits.append(self.cls_logits(self.cls_tower(feature)))
            bbox_reg.append(self.bbox_pred(self.bbox_tower(feature)))
        return logits, bbox_reg


class RetinaNetPostProcessor(RPNPostProcessor):
    """
    Performs post-processing on the outputs of the RetinaNet boxes.
    This is only used in the testing.
    """

    def __init__(self, pre_nms_thresh, pre_nms_top_n, nms_thresh,
        fpn_post_nms_top_n, min_size, num_classes, box_coder=None):
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
        super(RetinaNetPostProcessor, self).__init__(pre_nms_thresh, 0,
            nms_thresh, min_size)
        self.pre_nms_thresh = pre_nms_thresh
        self.pre_nms_top_n = pre_nms_top_n
        self.nms_thresh = nms_thresh
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.min_size = min_size
        self.num_classes = num_classes
        if box_coder is None:
            box_coder = BoxCoder(weights=(10.0, 10.0, 5.0, 5.0))
        self.box_coder = box_coder

    def add_gt_proposals(self, proposals, targets):
        """
        This function is not used in RetinaNet
        """
        pass

    def forward_for_single_feature_map(self, anchors, box_cls, box_regression):
        """
        Arguments:
            anchors: list[BoxList]
            box_cls: tensor of size N, A * C, H, W
            box_regression: tensor of size N, A * 4, H, W
        """
        device = box_cls.device
        N, _, H, W = box_cls.shape
        A = box_regression.size(1) // 4
        C = box_cls.size(1) // A
        box_cls = permute_and_flatten(box_cls, N, A, C, H, W)
        box_cls = box_cls.sigmoid()
        box_regression = permute_and_flatten(box_regression, N, A, 4, H, W)
        box_regression = box_regression.reshape(N, -1, 4)
        num_anchors = A * H * W
        candidate_inds = box_cls > self.pre_nms_thresh
        pre_nms_top_n = candidate_inds.view(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)
        results = []
        for per_box_cls, per_box_regression, per_pre_nms_top_n, per_candidate_inds, per_anchors in zip(
            box_cls, box_regression, pre_nms_top_n, candidate_inds, anchors):
            per_box_cls = per_box_cls[per_candidate_inds]
            per_box_cls, top_k_indices = per_box_cls.topk(per_pre_nms_top_n,
                sorted=False)
            per_candidate_nonzeros = per_candidate_inds.nonzero()[(
                top_k_indices), :]
            per_box_loc = per_candidate_nonzeros[:, (0)]
            per_class = per_candidate_nonzeros[:, (1)]
            per_class += 1
            detections = self.box_coder.decode(per_box_regression[(
                per_box_loc), :].view(-1, 4), per_anchors.bbox[(per_box_loc
                ), :].view(-1, 4))
            boxlist = BoxList(detections, per_anchors.size, mode='xyxy')
            boxlist.add_field('labels', per_class)
            boxlist.add_field('scores', per_box_cls)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = remove_small_boxes(boxlist, self.min_size)
            results.append(boxlist)
        return results

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


def make_retinanet_postprocessor(config, rpn_box_coder, is_train):
    pre_nms_thresh = config.MODEL.RETINANET.INFERENCE_TH
    pre_nms_top_n = config.MODEL.RETINANET.PRE_NMS_TOP_N
    nms_thresh = config.MODEL.RETINANET.NMS_TH
    fpn_post_nms_top_n = config.TEST.DETECTIONS_PER_IMG
    min_size = 0
    box_selector = RetinaNetPostProcessor(pre_nms_thresh=pre_nms_thresh,
        pre_nms_top_n=pre_nms_top_n, nms_thresh=nms_thresh,
        fpn_post_nms_top_n=fpn_post_nms_top_n, min_size=min_size,
        num_classes=config.MODEL.RETINANET.NUM_CLASSES, box_coder=rpn_box_coder
        )
    return box_selector


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


def generate_retinanet_labels(matched_targets):
    labels_per_image = matched_targets.get_field('labels')
    return labels_per_image


def concat_box_prediction_layers(box_cls, box_regression):
    box_cls_flattened = []
    box_regression_flattened = []
    for box_cls_per_level, box_regression_per_level in zip(box_cls,
        box_regression):
        N, AxC, H, W = box_cls_per_level.shape
        Ax4 = box_regression_per_level.shape[1]
        A = Ax4 // 4
        C = AxC // A
        box_cls_per_level = permute_and_flatten(box_cls_per_level, N, A, C,
            H, W)
        box_cls_flattened.append(box_cls_per_level)
        box_regression_per_level = permute_and_flatten(box_regression_per_level
            , N, A, 4, H, W)
        box_regression_flattened.append(box_regression_per_level)
    box_cls = cat(box_cls_flattened, dim=1).reshape(-1, C)
    box_regression = cat(box_regression_flattened, dim=1).reshape(-1, 4)
    return box_cls, box_regression


class RPNLossComputation(object):
    """
    This class computes the RPN loss.
    """

    def __init__(self, proposal_matcher, fg_bg_sampler, box_coder,
        generate_labels_func):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.box_coder = box_coder
        self.copied_fields = []
        self.generate_labels_func = generate_labels_func
        self.discard_cases = ['not_visibility', 'between_thresholds']

    def match_targets_to_anchors(self, anchor, target, copied_fields=[]):
        match_quality_matrix = boxlist_iou(target, anchor)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        target = target.copy_with_fields(copied_fields)
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field('matched_idxs', matched_idxs)
        return matched_targets

    def prepare_targets(self, anchors, targets):
        labels = []
        regression_targets = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            matched_targets = self.match_targets_to_anchors(anchors_per_image,
                targets_per_image, self.copied_fields)
            matched_idxs = matched_targets.get_field('matched_idxs')
            labels_per_image = self.generate_labels_func(matched_targets)
            labels_per_image = labels_per_image.to(dtype=torch.float32)
            bg_indices = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_indices] = 0
            if 'not_visibility' in self.discard_cases:
                labels_per_image[~anchors_per_image.get_field('visibility')
                    ] = -1
            if 'between_thresholds' in self.discard_cases:
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
        objectness, box_regression = concat_box_prediction_layers(objectness,
            box_regression)
        objectness = objectness.squeeze()
        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)
        box_loss = smooth_l1_loss(box_regression[sampled_pos_inds],
            regression_targets[sampled_pos_inds], beta=1.0 / 9,
            size_average=False) / sampled_inds.numel()
        objectness_loss = F.binary_cross_entropy_with_logits(objectness[
            sampled_inds], labels[sampled_inds])
        return objectness_loss, box_loss


class RetinaNetLossComputation(RPNLossComputation):
    """
    This class computes the RetinaNet loss.
    """

    def __init__(self, proposal_matcher, box_coder, generate_labels_func,
        sigmoid_focal_loss, bbox_reg_beta=0.11, regress_norm=1.0):
        """
        Arguments:
            proposal_matcher (Matcher)
            box_coder (BoxCoder)
        """
        self.proposal_matcher = proposal_matcher
        self.box_coder = box_coder
        self.box_cls_loss_func = sigmoid_focal_loss
        self.bbox_reg_beta = bbox_reg_beta
        self.copied_fields = ['labels']
        self.generate_labels_func = generate_labels_func
        self.discard_cases = ['between_thresholds']
        self.regress_norm = regress_norm

    def __call__(self, anchors, box_cls, box_regression, targets):
        """
        Arguments:
            anchors (list[BoxList])
            box_cls (list[Tensor])
            box_regression (list[Tensor])
            targets (list[BoxList])

        Returns:
            retinanet_cls_loss (Tensor)
            retinanet_regression_loss (Tensor
        """
        anchors = [cat_boxlist(anchors_per_image) for anchors_per_image in
            anchors]
        labels, regression_targets = self.prepare_targets(anchors, targets)
        N = len(labels)
        box_cls, box_regression = concat_box_prediction_layers(box_cls,
            box_regression)
        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)
        pos_inds = torch.nonzero(labels > 0).squeeze(1)
        retinanet_regression_loss = smooth_l1_loss(box_regression[pos_inds],
            regression_targets[pos_inds], beta=self.bbox_reg_beta,
            size_average=False) / max(1, pos_inds.numel() * self.regress_norm)
        labels = labels.int()
        retinanet_cls_loss = self.box_cls_loss_func(box_cls, labels) / (
            pos_inds.numel() + N)
        return retinanet_cls_loss, retinanet_regression_loss


def make_retinanet_loss_evaluator(cfg, box_coder):
    matcher = Matcher(cfg.MODEL.RETINANET.FG_IOU_THRESHOLD, cfg.MODEL.
        RETINANET.BG_IOU_THRESHOLD, allow_low_quality_matches=True)
    sigmoid_focal_loss = SigmoidFocalLoss(cfg.MODEL.RETINANET.LOSS_GAMMA,
        cfg.MODEL.RETINANET.LOSS_ALPHA)
    loss_evaluator = RetinaNetLossComputation(matcher, box_coder,
        generate_retinanet_labels, sigmoid_focal_loss, bbox_reg_beta=cfg.
        MODEL.RETINANET.BBOX_REG_BETA, regress_norm=cfg.MODEL.RETINANET.
        BBOX_REG_WEIGHT)
    return loss_evaluator


class RetinaNetModule(torch.nn.Module):
    """
    Module for RetinaNet computation. Takes feature maps from the backbone and
    RetinaNet outputs and losses. Only Test on FPN now.
    """

    def __init__(self, cfg, in_channels):
        super(RetinaNetModule, self).__init__()
        self.cfg = cfg.clone()
        anchor_generator = make_anchor_generator_retinanet(cfg)
        head = RetinaNetHead(cfg, in_channels)
        box_coder = BoxCoder(weights=(10.0, 10.0, 5.0, 5.0))
        box_selector_test = make_retinanet_postprocessor(cfg, box_coder,
            is_train=False)
        loss_evaluator = make_retinanet_loss_evaluator(cfg, box_coder)
        self.anchor_generator = anchor_generator
        self.head = head
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
        box_cls, box_regression = self.head(features)
        anchors = self.anchor_generator(images, features)
        if self.training:
            return self._forward_train(anchors, box_cls, box_regression,
                targets)
        else:
            return self._forward_test(anchors, box_cls, box_regression)

    def _forward_train(self, anchors, box_cls, box_regression, targets):
        loss_box_cls, loss_box_reg = self.loss_evaluator(anchors, box_cls,
            box_regression, targets)
        losses = {'loss_retina_cls': loss_box_cls, 'loss_retina_reg':
            loss_box_reg}
        return anchors, losses

    def _forward_test(self, anchors, box_cls, box_regression):
        boxes = self.box_selector_test(anchors, box_cls, box_regression)
        return boxes, {}


class RPNHeadConvRegressor(nn.Module):
    """
    A simple RPN Head for classification and bbox regression
    """

    def __init__(self, cfg, in_channels, num_anchors):
        """
        Arguments:
            cfg              : config
            in_channels (int): number of channels of the input feature
            num_anchors (int): number of anchors to be predicted
        """
        super(RPNHeadConvRegressor, self).__init__()
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1,
            stride=1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4,
            kernel_size=1, stride=1)
        for l in [self.cls_logits, self.bbox_pred]:
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        assert isinstance(x, (list, tuple))
        logits = [self.cls_logits(y) for y in x]
        bbox_reg = [self.bbox_pred(y) for y in x]
        return logits, bbox_reg


class RPNHeadFeatureSingleConv(nn.Module):
    """
    Adds a simple RPN Head with one conv to extract the feature
    """

    def __init__(self, cfg, in_channels):
        """
        Arguments:
            cfg              : config
            in_channels (int): number of channels of the input feature
        """
        super(RPNHeadFeatureSingleConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3,
            stride=1, padding=1)
        for l in [self.conv]:
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)
        self.out_channels = in_channels

    def forward(self, x):
        assert isinstance(x, (list, tuple))
        x = [F.relu(self.conv(z)) for z in x]
        return x


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


def generate_rpn_labels(matched_targets):
    matched_idxs = matched_targets.get_field('matched_idxs')
    labels_per_image = matched_idxs >= 0
    return labels_per_image


def make_rpn_loss_evaluator(cfg, box_coder):
    matcher = Matcher(cfg.MODEL.RPN.FG_IOU_THRESHOLD, cfg.MODEL.RPN.
        BG_IOU_THRESHOLD, allow_low_quality_matches=True)
    fg_bg_sampler = BalancedPositiveNegativeSampler(cfg.MODEL.RPN.
        BATCH_SIZE_PER_IMAGE, cfg.MODEL.RPN.POSITIVE_FRACTION)
    loss_evaluator = RPNLossComputation(matcher, fg_bg_sampler, box_coder,
        generate_rpn_labels)
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
    fpn_post_nms_per_batch = config.MODEL.RPN.FPN_POST_NMS_PER_BATCH
    nms_thresh = config.MODEL.RPN.NMS_THRESH
    min_size = config.MODEL.RPN.MIN_SIZE
    box_selector = RPNPostProcessor(pre_nms_top_n=pre_nms_top_n,
        post_nms_top_n=post_nms_top_n, nms_thresh=nms_thresh, min_size=
        min_size, box_coder=rpn_box_coder, fpn_post_nms_top_n=
        fpn_post_nms_top_n, fpn_post_nms_per_batch=fpn_post_nms_per_batch)
    return box_selector


class RPNModule(torch.nn.Module):
    """
    Module for RPN computation. Takes feature maps from the backbone and outputs
    RPN proposals and losses. Works for both FPN and non-FPN.
    """

    def __init__(self, cfg, in_channels):
        super(RPNModule, self).__init__()
        self.cfg = cfg.clone()
        anchor_generator = make_anchor_generator(cfg)
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


class Flattener(nn.Module):

    def __init__(self):
        """
        Flattens last 3 dimensions to make it only batch size, -1
        """
        super(Flattener, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_jwyang_graph_rcnn_pytorch(_paritybench_base):
    pass

    def test_000(self):
        self._check(FrozenBatchNorm2d(*[], **{'n': 4}), [torch.rand([4, 4, 4, 4])], {})
    @_fails_compile()

    def test_001(self):
        self._check(Conv2d(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})
    @_fails_compile()

    def test_002(self):
        self._check(ConvTranspose2d(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})
    @_fails_compile()

    def test_003(self):
        self._check(BatchNorm2d(*[], **{'num_features': 4}), [torch.rand([4, 4, 4, 4])], {})
    @_fails_compile()

    def test_004(self):
        self._check(Identity(*[], **{'C_in': 4, 'C_out': 4, 'stride': 1}), [torch.rand([4, 4, 4, 4])], {})
    @_fails_compile()

    def test_005(self):
        self._check(CascadeConv3x3(*[], **{'C_in': 4, 'C_out': 4, 'stride': 1}), [torch.rand([4, 4, 4, 4])], {})
    @_fails_compile()

    def test_006(self):
        self._check(Shift(*[], **{'C': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4}), [torch.rand([4, 4, 4, 4])], {})
    @_fails_compile()

    def test_007(self):
        self._check(ShiftBlock5x5(*[], **{'C_in': 4, 'C_out': 4, 'expansion': 4, 'stride': 1}), [torch.rand([4, 4, 4, 4])], {})

    def test_008(self):
        self._check(ChannelShuffle(*[], **{'groups': 1}), [torch.rand([4, 4, 4, 4])], {})
    @_fails_compile()

    def test_009(self):
        self._check(SEModule(*[], **{'C': 4}), [torch.rand([4, 4, 4, 4])], {})
    @_fails_compile()

    def test_010(self):
        self._check(IRFBlock(*[], **{'input_depth': 1, 'output_depth': 1, 'expansion': 4, 'stride': 1}), [torch.rand([4, 1, 64, 64])], {})

    def test_011(self):
        self._check(LastLevelMaxPool(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_012(self):
        self._check(LastLevelP6P7(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})
    @_fails_compile()

    def test_013(self):
        self._check(MultiHeadAttention(*[], **{'heads': 4, 'd_model': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_014(self):
        self._check(_Collection_Unit(*[], **{'dim_in': 4, 'dim_out': 4}), [torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])], {})

    def test_015(self):
        self._check(_Update_Unit(*[], **{'dim': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})
    @_fails_compile()

    def test_016(self):
        self._check(_GraphConvolutionLayer_Update(*[], **{'dim_obj': 4, 'dim_rel': 4}), [torch.rand([4, 4]), torch.rand([4, 4]), 0], {})

    def test_017(self):
        self._check(Message_Passing_Unit_v2(*[], **{'fea_size': 4}), [torch.rand([4, 4]), torch.rand([4, 4])], {})

    def test_018(self):
        self._check(Message_Passing_Unit_v1(*[], **{'fea_size': 4}), [torch.rand([4, 4]), torch.rand([4, 4])], {})

    def test_019(self):
        self._check(Gated_Recurrent_Unit(*[], **{'fea_size': 4, 'dropout': 0.5}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})
    @_fails_compile()

    def test_020(self):
        self._check(Relationshipness(*[], **{'dim': 4}), [torch.rand([4, 4])], {})
    @_fails_compile()

    def test_021(self):
        self._check(Relationshipnessv2(*[], **{'dim': 4}), [torch.rand([4, 4])], {})

    def test_022(self):
        self._check(Flattener(*[], **{}), [torch.rand([4, 4, 4, 4])], {})
