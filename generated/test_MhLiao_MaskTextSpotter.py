import sys
_module = sys.modules[__name__]
del sys
prepare_results = _module
rrc_evaluation_funcs = _module
script = _module
weighted_editdistance = _module
config = _module
defaults = _module
paths_catalog = _module
data = _module
build = _module
collate_batch = _module
datasets = _module
coco = _module
concat_dataset = _module
icdar = _module
list_dataset = _module
scut = _module
synthtext = _module
total_text = _module
samplers = _module
distributed = _module
grouped_batch_sampler = _module
iteration_based_batch_sampler = _module
transforms = _module
inference = _module
text_inference = _module
trainer = _module
layers = _module
_utils = _module
batch_norm = _module
misc = _module
nms = _module
roi_align = _module
roi_pool = _module
smooth_l1_loss = _module
modeling = _module
backbone = _module
fpn = _module
resnet = _module
balanced_positive_negative_sampler = _module
box_coder = _module
detector = _module
detectors = _module
generalized_rcnn = _module
matcher = _module
poolers = _module
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
roi_seq_predictors = _module
roi_heads = _module
rpn = _module
anchor_generator = _module
inference = _module
loss = _module
rpn = _module
utils = _module
solver = _module
lr_scheduler = _module
structures = _module
bounding_box = _module
boxlist_ops = _module
image_list = _module
segmentation_mask = _module
c2_model_loading = _module
chars = _module
checkpoint = _module
collect_env = _module
comm = _module
env = _module
imports = _module
logging = _module
metric_logger = _module
miscellaneous = _module
model_serialization = _module
model_zoo = _module
setup = _module
checkpoint = _module
test_data_samplers = _module
demo = _module
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


import logging


from torch import nn


from torch.nn.modules.utils import _ntuple


from torch.autograd import Function


from torch.autograd.function import once_differentiable


from torch.nn.modules.utils import _pair


import torch.nn.functional as F


from collections import namedtuple


from torch.nn import functional as F


import numpy as np


import random


from collections import OrderedDict


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
        scale = self.weight * self.running_var.rsqrt()
        bias = self.bias - self.running_mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return x * scale + bias


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


class FPN(nn.Module):
    """
    Module that adds FPN on top of a list of feature maps.
    The feature maps are currently supposed to be in increasing depth
    order, and must be consecutive
    """

    def __init__(self, in_channels_list, out_channels, top_blocks=None):
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
            inner_block_module = nn.Conv2d(in_channels, out_channels, 1)
            layer_block_module = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
            for module in [inner_block_module, layer_block_module]:
                nn.init.kaiming_uniform_(module.weight, a=1)
                nn.init.constant_(module.bias, 0)
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
            inner_top_down = F.interpolate(last_inner, scale_factor=2, mode
                ='nearest')
            inner_lateral = getattr(self, inner_block)(feature)
            last_inner = inner_lateral + inner_top_down
            results.insert(0, getattr(self, layer_block)(last_inner))
        if self.top_blocks is not None:
            last_results = self.top_blocks(results[-1])
            results.extend(last_results)
        return tuple(results)


class LastLevelMaxPool(nn.Module):

    def forward(self, x):
        return [F.max_pool2d(x, 1, 2, 0)]


class BottleneckWithFixedBatchNorm(nn.Module):

    def __init__(self, in_channels, bottleneck_channels, out_channels,
        num_groups=1, stride_in_1x1=True, stride=1):
        super(BottleneckWithFixedBatchNorm, self).__init__()
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(Conv2d(in_channels,
                out_channels, kernel_size=1, stride=stride, bias=False),
                FrozenBatchNorm2d(out_channels))
        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)
        self.conv1 = Conv2d(in_channels, bottleneck_channels, kernel_size=1,
            stride=stride_1x1, bias=False)
        self.bn1 = FrozenBatchNorm2d(bottleneck_channels)
        self.conv2 = Conv2d(bottleneck_channels, bottleneck_channels,
            kernel_size=3, stride=stride_3x3, padding=1, bias=False, groups
            =num_groups)
        self.bn2 = FrozenBatchNorm2d(bottleneck_channels)
        self.conv3 = Conv2d(bottleneck_channels, out_channels, kernel_size=
            1, bias=False)
        self.bn3 = FrozenBatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu_(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu_(out)
        out0 = self.conv3(out)
        out = self.bn3(out0)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = F.relu_(out)
        return out


class StemWithFixedBatchNorm(nn.Module):

    def __init__(self, cfg):
        super(StemWithFixedBatchNorm, self).__init__()
        out_channels = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
        self.conv1 = Conv2d(3, out_channels, kernel_size=7, stride=2,
            padding=3, bias=False)
        self.bn1 = FrozenBatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu_(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        return x


def build_rpn(cfg):
    """
    This gives the gist of it. Not super important because it doesn't change as much
    """
    return RPNModule(cfg)


def build_resnet_backbone(cfg):
    body = resnet.ResNet(cfg)
    model = nn.Sequential(OrderedDict([('body', body)]))
    return model


def build_resnet_fpn_backbone(cfg):
    body = resnet.ResNet(cfg)
    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    out_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
    fpn = fpn_module.FPN(in_channels_list=[in_channels_stage2, 
        in_channels_stage2 * 2, in_channels_stage2 * 4, in_channels_stage2 *
        8], out_channels=out_channels, top_blocks=fpn_module.LastLevelMaxPool()
        )
    model = nn.Sequential(OrderedDict([('body', body), ('fpn', fpn)]))
    return model


def build_backbone(cfg):
    assert cfg.MODEL.BACKBONE.CONV_BODY.startswith('R-'
        ), 'Only ResNet and ResNeXt models are currently implemented'
    if cfg.MODEL.BACKBONE.CONV_BODY.endswith('-FPN'):
        return build_resnet_fpn_backbone(cfg)
    return build_resnet_backbone(cfg)


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
            device = match_quality_matrix.device
            return torch.empty((0,), dtype=torch.int64, device=device)
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


def build_roi_mask_head(cfg):
    matcher = Matcher(cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD, cfg.MODEL.
        ROI_HEADS.BG_IOU_THRESHOLD, allow_low_quality_matches=False)
    return ROIMaskHead(cfg, matcher, (cfg.MODEL.ROI_MASK_HEAD.RESOLUTION_H,
        cfg.MODEL.ROI_MASK_HEAD.RESOLUTION_W))


def build_roi_box_head(cfg):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIBoxHead(cfg)


def build_roi_heads(cfg):
    roi_heads = []
    if not cfg.MODEL.RPN_ONLY:
        roi_heads.append(('box', build_roi_box_head(cfg)))
    if cfg.MODEL.MASK_ON:
        roi_heads.append(('mask', build_roi_mask_head(cfg)))
    if roi_heads:
        roi_heads = CombinedROIHeads(cfg, roi_heads)
    return roi_heads


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


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    = rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()
        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg)
        self.roi_heads = build_roi_heads(cfg)

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
            scales (list[flaot]): scales for each Pooler
            sampling_ratio (int): sampling ratio for ROIAlign
        """
        super(Pooler, self).__init__()
        poolers = []
        for scale in scales:
            poolers.append(ROIAlign(output_size, spatial_scale=scale,
                sampling_ratio=sampling_ratio))
        self.poolers = nn.ModuleList(poolers)
        self.output_size = output_size
        lvl_min = -math.log2(scales[0])
        lvl_max = -math.log2(scales[-1])
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
        output_size_h = self.output_size[0]
        output_size_w = self.output_size[1]
        dtype, device = x[0].dtype, x[0].device
        result = torch.zeros((num_rois, num_channels, output_size_h,
            output_size_w), dtype=dtype, device=device)
        for level, (per_level_feature, pooler) in enumerate(zip(x, self.
            poolers)):
            idx_in_level = torch.nonzero(levels == level).squeeze(1)
            rois_per_level = rois[idx_in_level]
            result[idx_in_level] = pooler(per_level_feature, rois_per_level)
        return result


FLIP_TOP_BOTTOM = 1


FLIP_LEFT_RIGHT = 0


class BoxList(object):
    """
    This class represents a set of bounding boxes.
    The bounding boxes are represented as a Nx4 Tensor.
    In order ot uniquely determine the bounding boxes with respect
    to an image, we also store the corresponding image dimensions.
    They can contain extra information that is specific to each bounding box, such as
    labels.
    """

    def __init__(self, bbox, image_size, mode='xyxy', use_char_ann=True):
        device = bbox.device if isinstance(bbox, torch.Tensor
            ) else torch.device('cpu')
        bbox = torch.as_tensor(bbox, dtype=torch.float32, device=device)
        if bbox.ndimension() != 2:
            raise ValueError('bbox should have 2 dimensions, got {}'.format
                (bbox.ndimension()))
        if bbox.size(-1) != 4:
            raise ValueError(
                'last dimenion of bbox should have a size of 4, got {}'.
                format(bbox.size(-1)))
        if mode not in ('xyxy', 'xywh'):
            raise ValueError("mode should be 'xyxy' or 'xywh'")
        self.bbox = bbox
        self.size = image_size
        self.mode = mode
        self.extra_fields = {}
        self.use_char_ann = use_char_ann

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
            bbox = BoxList(bbox, self.size, mode=mode, use_char_ann=self.
                use_char_ann)
        else:
            TO_REMOVE = 1
            bbox = torch.cat((xmin, ymin, xmax - xmin + TO_REMOVE, ymax -
                ymin + TO_REMOVE), dim=-1)
            bbox = BoxList(bbox, self.size, mode=mode, use_char_ann=self.
                use_char_ann)
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
            bbox = BoxList(scaled_box, size, mode=self.mode, use_char_ann=
                self.use_char_ann)
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
        bbox = BoxList(scaled_box, size, mode='xyxy', use_char_ann=self.
            use_char_ann)
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.resize(size, *args, **kwargs)
            bbox.add_field(k, v)
        return bbox.convert(self.mode)

    def poly2box(self, poly):
        xmin = min(poly[0::2])
        xmax = max(poly[0::2])
        ymin = min(poly[1::2])
        ymax = max(poly[1::2])
        return [xmin, ymin, xmax, ymax]

    def rotate(self, angle, r_c, start_h, start_w):
        masks = self.extra_fields['masks']
        masks = masks.rotate(angle, r_c, start_h, start_w)
        polys = masks.polygons
        boxes = []
        for poly in polys:
            box = self.poly2box(poly.polygons[0].numpy())
            boxes.append(box)
        self.size = r_c[0] * 2, r_c[1] * 2
        bbox = BoxList(boxes, self.size, mode='xyxy', use_char_ann=self.
            use_char_ann)
        for k, v in self.extra_fields.items():
            if k == 'masks':
                v = masks
            elif self.use_char_ann:
                if not isinstance(v, torch.Tensor):
                    v = v.rotate(angle, r_c, start_h, start_w)
            elif not isinstance(v, torch.Tensor) and k != 'char_masks':
                v = v.rotate(angle, r_c, start_h, start_w)
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
        bbox = BoxList(transposed_boxes, self.size, mode='xyxy',
            use_char_ann=self.use_char_ann)
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.transpose(method)
            bbox.add_field(k, v)
        return bbox.convert(self.mode)

    def crop(self, box):
        """
        Cropss a rectangular region from this bounding box. The box is a
        4-tuple defining the left, upper, right, and lower pixel
        coordinate.
        """
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        w, h = box[2] - box[0], box[3] - box[1]
        cropped_xmin = (xmin - box[0]).clamp(min=0, max=w)
        cropped_ymin = (ymin - box[1]).clamp(min=0, max=h)
        cropped_xmax = (xmax - box[0]).clamp(min=0, max=w)
        cropped_ymax = (ymax - box[1]).clamp(min=0, max=h)
        keep_ind = None
        not_empty = np.where((cropped_xmin != cropped_xmax) & (cropped_ymin !=
            cropped_ymax))[0]
        if len(not_empty) > 0:
            keep_ind = not_empty
        cropped_box = torch.cat((cropped_xmin, cropped_ymin, cropped_xmax,
            cropped_ymax), dim=-1)
        cropped_box = cropped_box[not_empty]
        bbox = BoxList(cropped_box, (w, h), mode='xyxy', use_char_ann=self.
            use_char_ann)
        for k, v in self.extra_fields.items():
            if self.use_char_ann:
                if not isinstance(v, torch.Tensor):
                    v = v.crop(box, keep_ind)
            elif not isinstance(v, torch.Tensor) and k != 'char_masks':
                v = v.crop(box, keep_ind)
            bbox.add_field(k, v)
        return bbox.convert(self.mode)

    def to(self, device):
        bbox = BoxList(self.bbox.to(device), self.size, self.mode, self.
            use_char_ann)
        for k, v in self.extra_fields.items():
            if hasattr(v, 'to'):
                v = v.to(device)
            bbox.add_field(k, v)
        return bbox

    def __getitem__(self, item):
        bbox = BoxList(self.bbox[item], self.size, self.mode, self.use_char_ann
            )
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
        TO_REMOVE = 1
        box = self.bbox
        area = (box[:, (2)] - box[:, (0)] + TO_REMOVE) * (box[:, (3)] - box
            [:, (1)] + TO_REMOVE)
        return area

    def copy_with_fields(self, fields):
        bbox = BoxList(self.bbox, self.size, self.mode, self.use_char_ann)
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


def boxlist_nms(boxlist, nms_thresh, max_proposals=-1, score_field='score'):
    """
    Performs non-maximum suppression on a boxlist, with scores specified
    in a boxlist field via score_field.

    Arguments:
        boxlist(BoxList)
        nms_thresh (float)
        max_proposals (int): if > 0, then only the top max_proposals are kept
            after non-maxium suppression
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


class ResNet50Conv5ROIFeatureExtractor(nn.Module):

    def __init__(self, config):
        super(ResNet50Conv5ROIFeatureExtractor, self).__init__()
        resolution = config.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = config.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = config.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(output_size=(resolution, resolution), scales=scales,
            sampling_ratio=sampling_ratio)
        stage = resnet.StageSpec(index=4, block_count=3, return_features=False)
        head = resnet.ResNetHead(block_module=config.MODEL.RESNETS.
            TRANS_FUNC, stages=(stage,), num_groups=config.MODEL.RESNETS.
            NUM_GROUPS, width_per_group=config.MODEL.RESNETS.
            WIDTH_PER_GROUP, stride_in_1x1=config.MODEL.RESNETS.
            STRIDE_IN_1X1, stride_init=None, res2_out_channels=config.MODEL
            .RESNETS.RES2_OUT_CHANNELS)
        self.pooler = pooler
        self.head = head

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = self.head(x)
        return x


class FPN2MLPFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg):
        super(FPN2MLPFeatureExtractor, self).__init__()
        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(output_size=(resolution, resolution), scales=scales,
            sampling_ratio=sampling_ratio)
        input_size = cfg.MODEL.BACKBONE.OUT_CHANNELS * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        self.pooler = pooler
        self.fc6 = nn.Linear(input_size, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)
        for l in [self.fc6, self.fc7]:
            nn.init.kaiming_uniform_(l.weight, a=1)
            nn.init.constant_(l.bias, 0)

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        return x


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
        if self.masker:
            mask_prob = self.masker(mask_prob, boxes)
        boxes_per_image = [len(box) for box in boxes]
        mask_prob = mask_prob.split(boxes_per_image, dim=0)
        results = []
        for prob, box in zip(mask_prob, boxes):
            bbox = BoxList(box.bbox, box.size, mode='xyxy')
            for field in box.fields():
                bbox.add_field(field, box.get_field(field))
            bbox.add_field('mask', prob)
            results.append(bbox)
        return results


class CharMaskPostProcessor(nn.Module):
    """
    From the results of the CNN, post process the masks
    by taking the mask corresponding to the class with max
    probability (which are of fixed size and directly output
    by the CNN) and return the masks in the mask field of the BoxList.

    If a masker object is passed, it will additionally
    project the masks in the image according to the locations in boxes,
    """

    def __init__(self, cfg, masker=None):
        super(CharMaskPostProcessor, self).__init__()
        self.masker = masker
        self.cfg = cfg

    def forward(self, x, char_mask, boxes, seq_outputs=None, seq_scores=
        None, detailed_seq_scores=None):
        """
        Arguments:
            x (Tensor): the mask logits
            char_mask (Tensor): the char mask logits
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for ech image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra field mask
        """
        mask_prob = x.sigmoid()
        char_mask_softmax = F.softmax(char_mask, dim=1)
        image_width, image_height = boxes[0].size
        char_results = {'char_mask': char_mask_softmax.cpu().numpy(),
            'boxes': boxes[0].bbox.cpu().numpy(), 'seq_outputs':
            seq_outputs, 'seq_scores': seq_scores, 'detailed_seq_scores':
            detailed_seq_scores}
        num_masks = x.shape[0]
        mask_prob = mask_prob.squeeze(dim=1)[:, (None)]
        if self.masker:
            mask_prob = self.masker(mask_prob, boxes)
        boxes_per_image = [len(box) for box in boxes]
        mask_prob = mask_prob.split(boxes_per_image, dim=0)
        results = []
        for prob, box in zip(mask_prob, boxes):
            bbox = BoxList(box.bbox, box.size, mode='xyxy')
            for field in box.fields():
                bbox.add_field(field, box.get_field(field))
            bbox.add_field('mask', prob)
            results.append(bbox)
        return [results, char_results]


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
        if cfg.MODEL.CHAR_MASK_ON:
            resolution_h = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION_H
            resolution_w = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION_W
        else:
            resolution_h = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
            resolution_w = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_MASK_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(output_size=(resolution_h, resolution_w), scales=
            scales, sampling_ratio=sampling_ratio)
        input_size = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.pooler = pooler
        layers = cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS
        next_feature = input_size
        self.blocks = []
        for layer_idx, layer_features in enumerate(layers, 1):
            layer_name = 'mask_fcn{}'.format(layer_idx)
            module = Conv2d(next_feature, layer_features, 3, stride=1,
                padding=1)
            nn.init.kaiming_normal_(module.weight, mode='fan_out',
                nonlinearity='relu')
            nn.init.constant_(module.bias, 0)
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


class CharMaskRCNNC4Predictor(nn.Module):

    def __init__(self, cfg):
        super(CharMaskRCNNC4Predictor, self).__init__()
        num_classes = 1
        char_num_classes = cfg.MODEL.ROI_MASK_HEAD.CHAR_NUM_CLASSES
        dim_reduced = cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS[-1]
        if cfg.MODEL.ROI_HEADS.USE_FPN:
            num_inputs = dim_reduced
        else:
            stage_index = 4
            stage2_relative_factor = 2 ** (stage_index - 1)
            res2_out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
            num_inputs = res2_out_channels * stage2_relative_factor
        self.conv5_mask = ConvTranspose2d(num_inputs, dim_reduced, 2, 2, 0)
        if cfg.MODEL.CHAR_MASK_ON:
            self.mask_fcn_logits = Conv2d(dim_reduced, num_classes, 1, 1, 0)
            self.char_mask_fcn_logits = Conv2d(dim_reduced,
                char_num_classes, 1, 1, 0)
        else:
            self.mask_fcn_logits = Conv2d(dim_reduced, num_classes, 1, 1, 0)
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity
                    ='relu')

    def forward(self, x):
        x = F.relu(self.conv5_mask(x))
        return self.mask_fcn_logits(x), self.char_mask_fcn_logits(x)


def make_roi_seq_predictor(cfg, dim_in):
    return SequencePredictor(cfg, dim_in)


class SeqCharMaskRCNNC4Predictor(nn.Module):

    def __init__(self, cfg):
        super(SeqCharMaskRCNNC4Predictor, self).__init__()
        num_classes = 1
        char_num_classes = cfg.MODEL.ROI_MASK_HEAD.CHAR_NUM_CLASSES
        dim_reduced = cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS[-1]
        if cfg.MODEL.ROI_HEADS.USE_FPN:
            num_inputs = dim_reduced
        else:
            stage_index = 4
            stage2_relative_factor = 2 ** (stage_index - 1)
            res2_out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
            num_inputs = res2_out_channels * stage2_relative_factor
        self.conv5_mask = ConvTranspose2d(num_inputs, dim_reduced, 2, 2, 0)
        if cfg.MODEL.CHAR_MASK_ON:
            self.mask_fcn_logits = Conv2d(dim_reduced, num_classes, 1, 1, 0)
            self.char_mask_fcn_logits = Conv2d(dim_reduced,
                char_num_classes, 1, 1, 0)
            self.seq = make_roi_seq_predictor(cfg, dim_reduced)
        else:
            self.mask_fcn_logits = Conv2d(dim_reduced, num_classes, 1, 1, 0)
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity
                    ='relu')

    def forward(self, x, decoder_targets=None, word_targets=None):
        x = F.relu(self.conv5_mask(x))
        if self.training:
            loss_seq_decoder = self.seq(x, decoder_targets=decoder_targets,
                word_targets=word_targets)
            return self.mask_fcn_logits(x), self.char_mask_fcn_logits(x
                ), loss_seq_decoder
        else:
            decoded_chars, decoded_scores, detailed_decoded_scores = self.seq(x
                , use_beam_search=True)
            return self.mask_fcn_logits(x), self.char_mask_fcn_logits(x
                ), decoded_chars, decoded_scores, detailed_decoded_scores


def check_all_done(seqs):
    for seq in seqs:
        if not seq[-1]:
            return False
    return True


def reduce_mul(l):
    out = 1.0
    for x in l:
        out *= x
    return out


cpu_device = torch.device('cpu')


def num2char(num):
    chars = '_0123456789abcdefghijklmnopqrstuvwxyz'
    char = chars[num]
    return char


gpu_device = torch.device('cuda')


class SequencePredictor(nn.Module):

    def __init__(self, cfg, dim_in):
        super(SequencePredictor, self).__init__()
        self.cfg = cfg
        if cfg.SEQUENCE.TWO_CONV:
            self.seq_encoder = nn.Sequential(nn.Conv2d(dim_in, dim_in, 3,
                padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, stride=2,
                ceil_mode=True), nn.Conv2d(dim_in, 256, 3, padding=1), nn.
                ReLU(inplace=True))
        else:
            self.seq_encoder = nn.Sequential(nn.Conv2d(dim_in, 256, 3,
                padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, stride=2,
                ceil_mode=True))
        self.seq_decoder = BahdanauAttnDecoderRNN(256, cfg.SEQUENCE.
            NUM_CHAR, cfg.SEQUENCE.NUM_CHAR, n_layers=1, dropout_p=0.1)
        self.criterion_seq_decoder = nn.NLLLoss(ignore_index=-1, reduction=
            'none')
        self.rescale = nn.Upsample(size=(16, 64), mode='bilinear',
            align_corners=False)
        self.x_onehot = nn.Embedding(32, 32)
        self.x_onehot.weight.data = torch.eye(32)
        self.y_onehot = nn.Embedding(8, 8)
        self.y_onehot.weight.data = torch.eye(8)
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity
                    ='relu')

    def forward(self, x, decoder_targets=None, word_targets=None,
        use_beam_search=False):
        rescale_out = self.rescale(x)
        seq_decoder_input = self.seq_encoder(rescale_out)
        x_t, y_t = np.meshgrid(np.linspace(0, 31, 32), np.linspace(0, 7, 8))
        x_t = torch.LongTensor(x_t, device=cpu_device)
        y_t = torch.LongTensor(y_t, device=cpu_device)
        x_onehot_embedding = self.x_onehot(x_t).transpose(0, 2).transpose(1, 2
            ).repeat(seq_decoder_input.size(0), 1, 1, 1)
        y_onehot_embedding = self.y_onehot(y_t).transpose(0, 2).transpose(1, 2
            ).repeat(seq_decoder_input.size(0), 1, 1, 1)
        seq_decoder_input_loc = torch.cat([seq_decoder_input,
            x_onehot_embedding, y_onehot_embedding], 1)
        seq_decoder_input_reshape = seq_decoder_input_loc.view(
            seq_decoder_input_loc.size(0), seq_decoder_input_loc.size(1), -1
            ).transpose(0, 2).transpose(1, 2)
        if self.training:
            bos_onehot = np.zeros((seq_decoder_input_reshape.size(1), 1),
                dtype=np.int32)
            bos_onehot[:, (0)] = self.cfg.SEQUENCE.BOS_TOKEN
            decoder_input = torch.tensor(bos_onehot.tolist(), device=gpu_device
                )
            decoder_hidden = torch.zeros((seq_decoder_input_reshape.size(1),
                256), device=gpu_device)
            use_teacher_forcing = True if random.random(
                ) < self.cfg.SEQUENCE.TEACHER_FORCE_RATIO else False
            target_length = decoder_targets.size(1)
            if use_teacher_forcing:
                for di in range(target_length):
                    decoder_output, decoder_hidden, decoder_attention = (self
                        .seq_decoder(decoder_input, decoder_hidden,
                        seq_decoder_input_reshape))
                    if di == 0:
                        loss_seq_decoder = self.criterion_seq_decoder(
                            decoder_output, word_targets[:, (di)])
                    else:
                        loss_seq_decoder += self.criterion_seq_decoder(
                            decoder_output, word_targets[:, (di)])
                    decoder_input = decoder_targets[:, (di)]
            else:
                for di in range(target_length):
                    decoder_output, decoder_hidden, decoder_attention = (self
                        .seq_decoder(decoder_input, decoder_hidden,
                        seq_decoder_input_reshape))
                    topv, topi = decoder_output.topk(1)
                    decoder_input = topi.squeeze(1).detach()
                    if di == 0:
                        loss_seq_decoder = self.criterion_seq_decoder(
                            decoder_output, word_targets[:, (di)])
                    else:
                        loss_seq_decoder += self.criterion_seq_decoder(
                            decoder_output, word_targets[:, (di)])
            loss_seq_decoder = loss_seq_decoder.sum() / loss_seq_decoder.size(0
                )
            loss_seq_decoder = 0.2 * loss_seq_decoder
            return loss_seq_decoder
        else:
            words = []
            decoded_scores = []
            detailed_decoded_scores = []
            real_length = 0
            if use_beam_search:
                for batch_index in range(seq_decoder_input_reshape.size(1)):
                    decoder_hidden = torch.zeros((1, 256), device=gpu_device)
                    word = []
                    char_scores = []
                    detailed_char_scores = []
                    top_seqs = self.beam_search(seq_decoder_input_reshape[:,
                        batch_index:batch_index + 1, :], decoder_hidden,
                        beam_size=6, max_len=self.cfg.SEQUENCE.MAX_LENGTH)
                    top_seq = top_seqs[0]
                    for character in top_seq[1:]:
                        character_index = character[0]
                        if character_index == self.cfg.SEQUENCE.NUM_CHAR - 1:
                            char_scores.append(character[1])
                            detailed_char_scores.append(character[2])
                            break
                        elif character_index == 0:
                            word.append('~')
                            char_scores.append(0.0)
                        else:
                            word.append(num2char(character_index))
                            char_scores.append(character[1])
                            detailed_char_scores.append(character[2])
                    words.append(''.join(word))
                    decoded_scores.append(char_scores)
                    detailed_decoded_scores.append(detailed_char_scores)
            else:
                for batch_index in range(seq_decoder_input_reshape.size(1)):
                    bos_onehot = np.zeros((1, 1), dtype=np.int32)
                    bos_onehot[:, (0)] = self.cfg.SEQUENCE.BOS_TOKEN
                    decoder_input = torch.tensor(bos_onehot.tolist(),
                        device=gpu_device)
                    decoder_hidden = torch.zeros((1, 256), device=gpu_device)
                    word = []
                    char_scores = []
                    for di in range(self.cfg.SEQUENCE.MAX_LENGTH):
                        (decoder_output, decoder_hidden, decoder_attention) = (
                            self.seq_decoder(decoder_input, decoder_hidden,
                            seq_decoder_input_reshape[:, batch_index:
                            batch_index + 1, :]))
                        topv, topi = decoder_output.data.topk(1)
                        char_scores.append(topv.item())
                        if topi.item() == self.cfg.SEQUENCE.NUM_CHAR - 1:
                            break
                        elif topi.item() == 0:
                            word.append('~')
                        else:
                            word.append(num2char(topi.item()))
                        real_length = di
                        decoder_input = topi.squeeze(1).detach()
                    words.append(''.join(word))
                    decoded_scores.append(char_scores)
            return words, decoded_scores, detailed_decoded_scores

    def beam_search_step(self, encoder_context, top_seqs, k):
        all_seqs = []
        for seq in top_seqs:
            seq_score = reduce_mul([_score for _, _score, _, _ in seq])
            if seq[-1][0] == self.cfg.SEQUENCE.NUM_CHAR - 1:
                all_seqs.append((seq, seq_score, seq[-1][2], True))
                continue
            decoder_hidden = seq[-1][-1][0]
            onehot = np.zeros((1, 1), dtype=np.int32)
            onehot[:, (0)] = seq[-1][0]
            decoder_input = torch.tensor(onehot.tolist(), device=gpu_device)
            decoder_output, decoder_hidden, decoder_attention = (self.
                seq_decoder(decoder_input, decoder_hidden, encoder_context))
            detailed_char_scores = decoder_output.cpu().numpy()
            scores, candidates = decoder_output.data[:, 1:].topk(k)
            for i in range(k):
                character_score = scores[:, (i)]
                character_index = candidates[:, (i)]
                score = seq_score * character_score.item()
                char_score = seq_score * detailed_char_scores
                rs_seq = seq + [(character_index.item() + 1,
                    character_score.item(), char_score, [decoder_hidden])]
                done = character_index.item(
                    ) + 1 == self.cfg.SEQUENCE.NUM_CHAR - 1
                all_seqs.append((rs_seq, score, char_score, done))
        all_seqs = sorted(all_seqs, key=lambda seq: seq[1], reverse=True)
        topk_seqs = [seq for seq, _, _, _ in all_seqs[:k]]
        all_done = check_all_done(all_seqs[:k])
        return topk_seqs, all_done

    def beam_search(self, encoder_context, decoder_hidden, beam_size=6,
        max_len=32):
        char_score = np.zeros(38)
        top_seqs = [[(self.cfg.SEQUENCE.BOS_TOKEN, 1.0, char_score, [
            decoder_hidden])]]
        for _ in range(max_len):
            top_seqs, all_done = self.beam_search_step(encoder_context,
                top_seqs, beam_size)
            if all_done:
                break
        return top_seqs


class Attn(nn.Module):

    def __init__(self, method, hidden_size, embed_size):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.attn = nn.Linear(2 * self.hidden_size + 32 + 8, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1.0 / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, hidden, encoder_outputs):
        """
        :param hidden: 
            previous hidden state of the decoder, in shape (B, hidden_size)
        :param encoder_outputs:
            encoder outputs from Encoder, in shape (H*W, B, hidden_size)
        :return
            attention energies in shape (B, H*W)
        """
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)
        H = hidden.repeat(max_len, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)
        attn_energies = self.score(H, encoder_outputs)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, encoder_outputs):
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy = energy.transpose(2, 1)
        v = self.v.repeat(encoder_outputs.data.shape[0], 1).unsqueeze(1)
        energy = torch.bmm(v, energy)
        return energy.squeeze(1)


class BahdanauAttnDecoderRNN(nn.Module):

    def __init__(self, hidden_size, embed_size, output_size, n_layers=1,
        dropout_p=0, bidirectional=False):
        super(BahdanauAttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.embedding = nn.Embedding(output_size, embed_size)
        self.embedding.weight.data = torch.eye(embed_size)
        self.word_linear = nn.Linear(embed_size, hidden_size)
        self.attn = Attn('concat', hidden_size, embed_size)
        self.rnn = nn.GRUCell(2 * hidden_size + 32 + 8, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, word_input, last_hidden, encoder_outputs):
        """
        :param word_input:
            word input for current time step, in shape (B)
        :param last_hidden:
            last hidden stat of the decoder, in shape (layers*direction*B, hidden_size)
        :param encoder_outputs:
            encoder outputs in shape (H*W, B, C)
        :return
            decoder output
        """
        word_embedded_onehot = self.embedding(word_input).view(1,
            word_input.size(0), -1)
        word_embedded = self.word_linear(word_embedded_onehot)
        attn_weights = self.attn(last_hidden, encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        context = context.transpose(0, 1)
        rnn_input = torch.cat((word_embedded, context), 2)
        last_hidden = last_hidden.view(last_hidden.size(0), -1)
        rnn_input = rnn_input.view(word_input.size(0), -1)
        hidden = self.rnn(rnn_input, last_hidden)
        if not self.training:
            output = F.softmax(self.out(hidden), dim=1)
        else:
            output = F.log_softmax(self.out(hidden), dim=1)
        return output, hidden, attn_weights


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
            cell_anchors = [generate_anchors(anchor_stride, (size,),
                aspect_ratios).float() for anchor_stride, size in zip(
                anchor_strides, sizes)]
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
            inds_inside = torch.ones(anchors.shape[0], dtype=torch.bool,
                device=device)
        boxlist.add_field('visibility', inds_inside)

    def forward(self, image_list, feature_maps):
        grid_height, grid_width = feature_maps[0].shape[-2:]
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
            inds_mask = torch.zeros_like(objectness, dtype=torch.bool)
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


class RPNHead(nn.Module):
    """
    Adds a simple RPN Head with classification and regression heads
    """

    def __init__(self, in_channels, num_anchors):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
            num_anchors (int): number of anchors to be predicted
        """
        super(RPNHead, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3,
            stride=1, padding=1)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1,
            stride=1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4,
            kernel_size=1, stride=1)
        for l in [self.conv, self.cls_logits, self.bbox_pred]:
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        logits = []
        bbox_reg = []
        for feature in x:
            t = F.relu(self.conv(feature))
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg


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
            perm1 = torch.randperm(positive.numel())[:num_pos]
            perm2 = torch.randperm(negative.numel())[:num_neg]
            pos_idx_per_image = positive[perm1]
            neg_idx_per_image = negative[perm2]
            pos_idx_per_image_mask = torch.zeros_like(matched_idxs_per_image,
                dtype=torch.bool)
            neg_idx_per_image_mask = torch.zeros_like(matched_idxs_per_image,
                dtype=torch.bool)
            pos_idx_per_image_mask[pos_idx_per_image] = 1
            neg_idx_per_image_mask[neg_idx_per_image] = 1
            pos_idx.append(pos_idx_per_image_mask)
            neg_idx.append(neg_idx_per_image_mask)
        return pos_idx, neg_idx


def make_rpn_loss_evaluator(cfg, box_coder):
    matcher = Matcher(cfg.MODEL.RPN.FG_IOU_THRESHOLD, cfg.MODEL.RPN.
        BG_IOU_THRESHOLD, allow_low_quality_matches=True)
    fg_bg_sampler = BalancedPositiveNegativeSampler(cfg.MODEL.RPN.
        BATCH_SIZE_PER_IMAGE, cfg.MODEL.RPN.POSITIVE_FRACTION)
    loss_evaluator = RPNLossComputation(matcher, fg_bg_sampler, box_coder)
    return loss_evaluator


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
        head = RPNHead(in_channels, anchor_generator.
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


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_MhLiao_MaskTextSpotter(_paritybench_base):
    pass
    def test_000(self):
        self._check(FrozenBatchNorm2d(*[], **{'n': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(Conv2d(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_002(self):
        self._check(ConvTranspose2d(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(LastLevelMaxPool(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_004(self):
        self._check(BottleneckWithFixedBatchNorm(*[], **{'in_channels': 4, 'bottleneck_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_005(self):
        self._check(RPNHead(*[], **{'in_channels': 4, 'num_anchors': 4}), [torch.rand([4, 4, 4, 64, 64])], {})

