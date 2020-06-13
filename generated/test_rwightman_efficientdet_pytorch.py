import sys
_module = sys.modules[__name__]
del sys
avg_checkpoints = _module
data = _module
dataset = _module
loader = _module
transforms = _module
effdet = _module
anchors = _module
bench = _module
config = _module
model_config = _module
train_config = _module
distributed = _module
efficientdet = _module
evaluator = _module
factory = _module
helpers = _module
loss = _module
object_detection = _module
argmax_matcher = _module
box_coder = _module
box_list = _module
matcher = _module
region_similarity_calculator = _module
target_assigner = _module
version = _module
setup = _module
sotabench = _module
train = _module
validate = _module

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


import collections


import numpy as np


import torch


import torch.nn as nn


import logging


import math


from collections import OrderedDict


from typing import List


import torch.nn.functional as F


from typing import Optional


from torch.nn.functional import one_hot


import torch.nn.parallel


def _generate_anchor_configs(min_level, max_level, num_scales, aspect_ratios):
    """Generates mapping from output level to a list of anchor configurations.

    A configuration is a tuple of (num_anchors, scale, aspect_ratio).

    Args:
        min_level: integer number of minimum level of the output feature pyramid.

        max_level: integer number of maximum level of the output feature pyramid.

        num_scales: integer number representing intermediate scales added on each level.
            For instances, num_scales=2 adds two additional anchor scales [2^0, 2^0.5] on each level.

        aspect_ratios: list of tuples representing the aspect ratio anchors added on each level.
            For instances, aspect_ratios = [(1, 1), (1.4, 0.7), (0.7, 1.4)] adds three anchors on each level.

    Returns:
        anchor_configs: a dictionary with keys as the levels of anchors and
            values as a list of anchor configuration.
    """
    anchor_configs = {}
    for level in range(min_level, max_level + 1):
        anchor_configs[level] = []
        for scale_octave in range(num_scales):
            for aspect in aspect_ratios:
                anchor_configs[level].append((2 ** level, scale_octave /
                    float(num_scales), aspect))
    return anchor_configs


def _generate_anchor_boxes(image_size, anchor_scale, anchor_configs):
    """Generates multiscale anchor boxes.

    Args:
        image_size: integer number of input image size. The input image has the same dimension for
            width and height. The image_size should be divided by the largest feature stride 2^max_level.

        anchor_scale: float number representing the scale of size of the base
            anchor to the feature stride 2^level.

        anchor_configs: a dictionary with keys as the levels of anchors and
            values as a list of anchor configuration.

    Returns:
        anchor_boxes: a numpy array with shape [N, 4], which stacks anchors on all feature levels.

    Raises:
        ValueError: input size must be the multiple of largest feature stride.
    """
    boxes_all = []
    for _, configs in anchor_configs.items():
        boxes_level = []
        for config in configs:
            stride, octave_scale, aspect = config
            if image_size % stride != 0:
                raise ValueError('input size must be divided by the stride.')
            base_anchor_size = anchor_scale * stride * 2 ** octave_scale
            anchor_size_x_2 = base_anchor_size * aspect[0] / 2.0
            anchor_size_y_2 = base_anchor_size * aspect[1] / 2.0
            x = np.arange(stride / 2, image_size, stride)
            y = np.arange(stride / 2, image_size, stride)
            xv, yv = np.meshgrid(x, y)
            xv = xv.reshape(-1)
            yv = yv.reshape(-1)
            boxes = np.vstack((yv - anchor_size_y_2, xv - anchor_size_x_2, 
                yv + anchor_size_y_2, xv + anchor_size_x_2))
            boxes = np.swapaxes(boxes, 0, 1)
            boxes_level.append(np.expand_dims(boxes, axis=1))
        boxes_level = np.concatenate(boxes_level, axis=1)
        boxes_all.append(boxes_level.reshape([-1, 4]))
    anchor_boxes = np.vstack(boxes_all)
    return anchor_boxes


class Anchors(nn.Module):
    """RetinaNet Anchors class."""

    def __init__(self, min_level, max_level, num_scales, aspect_ratios,
        anchor_scale, image_size):
        """Constructs multiscale RetinaNet anchors.

        Args:
            min_level: integer number of minimum level of the output feature pyramid.

            max_level: integer number of maximum level of the output feature pyramid.

            num_scales: integer number representing intermediate scales added
                on each level. For instances, num_scales=2 adds two additional
                anchor scales [2^0, 2^0.5] on each level.

            aspect_ratios: list of tuples representing the aspect ratio anchors added
                on each level. For instances, aspect_ratios =
                [(1, 1), (1.4, 0.7), (0.7, 1.4)] adds three anchors on each level.

            anchor_scale: float number representing the scale of size of the base
                anchor to the feature stride 2^level.

            image_size: integer number of input image size. The input image has the
                same dimension for width and height. The image_size should be divided by
                the largest feature stride 2^max_level.
        """
        super(Anchors, self).__init__()
        self.min_level = min_level
        self.max_level = max_level
        self.num_scales = num_scales
        self.aspect_ratios = aspect_ratios
        self.anchor_scale = anchor_scale
        self.image_size = image_size
        self.config = self._generate_configs()
        self.register_buffer('boxes', self._generate_boxes())

    def _generate_configs(self):
        """Generate configurations of anchor boxes."""
        return _generate_anchor_configs(self.min_level, self.max_level,
            self.num_scales, self.aspect_ratios)

    def _generate_boxes(self):
        """Generates multiscale anchor boxes."""
        boxes = _generate_anchor_boxes(self.image_size, self.anchor_scale,
            self.config)
        boxes = torch.from_numpy(boxes).float()
        return boxes

    def get_anchors_per_location(self):
        return self.num_scales * len(self.aspect_ratios)


MAX_DETECTION_POINTS = 5000


def _post_process(config, cls_outputs, box_outputs):
    """Selects top-k predictions.

    Post-proc code adapted from Tensorflow version at: https://github.com/google/automl/tree/master/efficientdet
    and optimized for PyTorch.

    Args:
        config: a parameter dictionary that includes `min_level`, `max_level`,  `batch_size`, and `num_classes`.

        cls_outputs: an OrderDict with keys representing levels and values
            representing logits in [batch_size, height, width, num_anchors].

        box_outputs: an OrderDict with keys representing levels and values
            representing box regression targets in [batch_size, height, width, num_anchors * 4].
    """
    batch_size = cls_outputs[0].shape[0]
    cls_outputs_all = torch.cat([cls_outputs[level].permute(0, 2, 3, 1).
        reshape([batch_size, -1, config.num_classes]) for level in range(
        config.num_levels)], 1)
    box_outputs_all = torch.cat([box_outputs[level].permute(0, 2, 3, 1).
        reshape([batch_size, -1, 4]) for level in range(config.num_levels)], 1)
    _, cls_topk_indices_all = torch.topk(cls_outputs_all.reshape(batch_size,
        -1), dim=1, k=MAX_DETECTION_POINTS)
    indices_all = cls_topk_indices_all / config.num_classes
    classes_all = cls_topk_indices_all % config.num_classes
    box_outputs_all_after_topk = torch.gather(box_outputs_all, 1,
        indices_all.unsqueeze(2).expand(-1, -1, 4))
    cls_outputs_all_after_topk = torch.gather(cls_outputs_all, 1,
        indices_all.unsqueeze(2).expand(-1, -1, config.num_classes))
    cls_outputs_all_after_topk = torch.gather(cls_outputs_all_after_topk, 2,
        classes_all.unsqueeze(2))
    return (cls_outputs_all_after_topk, box_outputs_all_after_topk,
        indices_all, classes_all)


def decode_box_outputs(rel_codes, anchors, output_xyxy: bool=False):
    """Transforms relative regression coordinates to absolute positions.

    Network predictions are normalized and relative to a given anchor; this
    reverses the transformation and outputs absolute coordinates for the input image.

    Args:
        rel_codes: box regression targets.

        anchors: anchors on all feature levels.

    Returns:
        outputs: bounding boxes.

    """
    ycenter_a = (anchors[:, (0)] + anchors[:, (2)]) / 2
    xcenter_a = (anchors[:, (1)] + anchors[:, (3)]) / 2
    ha = anchors[:, (2)] - anchors[:, (0)]
    wa = anchors[:, (3)] - anchors[:, (1)]
    ty, tx, th, tw = rel_codes.unbind(dim=1)
    w = torch.exp(tw) * wa
    h = torch.exp(th) * ha
    ycenter = ty * ha + ycenter_a
    xcenter = tx * wa + xcenter_a
    ymin = ycenter - h / 2.0
    xmin = xcenter - w / 2.0
    ymax = ycenter + h / 2.0
    xmax = xcenter + w / 2.0
    if output_xyxy:
        out = torch.stack([xmin, ymin, xmax, ymax], dim=1)
    else:
        out = torch.stack([ymin, xmin, ymax, xmax], dim=1)
    return out


MAX_DETECTIONS_PER_IMAGE = 100


def clip_boxes_xyxy(boxes: torch.Tensor, size: torch.Tensor):
    boxes = boxes.clamp(min=0)
    size = torch.cat([size, size], dim=0)
    boxes = boxes.min(size)
    return boxes


def generate_detections(cls_outputs, box_outputs, anchor_boxes, indices,
    classes, img_scale, img_size, max_det_per_image: int=
    MAX_DETECTIONS_PER_IMAGE):
    """Generates detections with RetinaNet model outputs and anchors.

    Args:
        cls_outputs: a torch tensor with shape [N, 1], which has the highest class
            scores on all feature levels. The N is the number of selected
            top-K total anchors on all levels.  (k being MAX_DETECTION_POINTS)

        box_outputs: a torch tensor with shape [N, 4], which stacks box regression
            outputs on all feature levels. The N is the number of selected top-k
            total anchors on all levels. (k being MAX_DETECTION_POINTS)

        anchor_boxes: a torch tensor with shape [N, 4], which stacks anchors on all
            feature levels. The N is the number of selected top-k total anchors on all levels.

        indices: a torch tensor with shape [N], which is the indices from top-k selection.

        classes: a torch tensor with shape [N], which represents the class
            prediction on all selected anchors from top-k selection.

        img_scale: a float tensor representing the scale between original image
            and input image for the detector. It is used to rescale detections for
            evaluating with the original groundtruth annotations.

        max_det_per_image: an int constant, added as argument to make torchscript happy

    Returns:
        detections: detection results in a tensor with shape [MAX_DETECTION_POINTS, 6],
            each row representing [x, y, width, height, score, class]
    """
    anchor_boxes = anchor_boxes[(indices), :]
    boxes = decode_box_outputs(box_outputs.float(), anchor_boxes,
        output_xyxy=True)
    boxes = clip_boxes_xyxy(boxes, img_size / img_scale)
    scores = cls_outputs.sigmoid().squeeze(1).float()
    top_detection_idx = batched_nms(boxes, scores, classes, iou_threshold=0.5)
    top_detection_idx = top_detection_idx[:max_det_per_image]
    boxes = boxes[top_detection_idx]
    scores = scores[top_detection_idx, None]
    classes = classes[top_detection_idx, None]
    boxes[:, (2)] -= boxes[:, (0)]
    boxes[:, (3)] -= boxes[:, (1)]
    boxes *= img_scale
    classes += 1
    detections = torch.cat([boxes, scores, classes.float()], dim=1)
    if len(top_detection_idx) < max_det_per_image:
        detections = torch.cat([detections, torch.zeros((max_det_per_image -
            len(top_detection_idx), 6), device=detections.device, dtype=
            detections.dtype)], dim=0)
    return detections


class SequentialAppend(nn.Sequential):

    def __init__(self, *args):
        super(SequentialAppend, self).__init__(*args)

    def forward(self, x: List[torch.Tensor]):
        for module in self:
            x.append(module(x))
        return x


class SequentialAppendLast(nn.Sequential):

    def __init__(self, *args):
        super(SequentialAppendLast, self).__init__(*args)

    def forward(self, x: List[torch.Tensor]):
        for module in self:
            x.append(module(x[-1]))
        return x


class ResampleFeatureMap(nn.Sequential):

    def __init__(self, in_channels, out_channels, reduction_ratio=1.0,
        pad_type='', pooling_type='max', norm_layer=nn.BatchNorm2d,
        norm_kwargs=None, apply_bn=False, conv_after_downsample=False,
        redundant_bias=False):
        super(ResampleFeatureMap, self).__init__()
        pooling_type = pooling_type or 'max'
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reduction_ratio = reduction_ratio
        self.conv_after_downsample = conv_after_downsample
        conv = None
        if in_channels != out_channels:
            conv = ConvBnAct2d(in_channels, out_channels, kernel_size=1,
                padding=pad_type, norm_layer=norm_layer if apply_bn else
                None, norm_kwargs=norm_kwargs, bias=not apply_bn or
                redundant_bias, act_layer=None)
        if reduction_ratio > 1:
            stride_size = int(reduction_ratio)
            if conv is not None and not self.conv_after_downsample:
                self.add_module('conv', conv)
            self.add_module('downsample', create_pool2d(pooling_type,
                kernel_size=stride_size + 1, stride=stride_size, padding=
                pad_type))
            if conv is not None and self.conv_after_downsample:
                self.add_module('conv', conv)
        else:
            if conv is not None:
                self.add_module('conv', conv)
            if reduction_ratio < 1:
                scale = int(1 // reduction_ratio)
                self.add_module('upsample', nn.UpsamplingNearest2d(
                    scale_factor=scale))


class FpnCombine(nn.Module):

    def __init__(self, feature_info, fpn_config, fpn_channels,
        inputs_offsets, target_reduction, pad_type='', pooling_type='max',
        norm_layer=nn.BatchNorm2d, norm_kwargs=None,
        apply_bn_for_resampling=False, conv_after_downsample=False,
        redundant_bias=False, weight_method='attn'):
        super(FpnCombine, self).__init__()
        self.inputs_offsets = inputs_offsets
        self.weight_method = weight_method
        self.resample = nn.ModuleDict()
        for idx, offset in enumerate(inputs_offsets):
            in_channels = fpn_channels
            if offset < len(feature_info):
                in_channels = feature_info[offset]['num_chs']
                input_reduction = feature_info[offset]['reduction']
            else:
                node_idx = offset - len(feature_info)
                input_reduction = fpn_config.nodes[node_idx]['reduction']
            reduction_ratio = target_reduction / input_reduction
            self.resample[str(offset)] = ResampleFeatureMap(in_channels,
                fpn_channels, reduction_ratio=reduction_ratio, pad_type=
                pad_type, pooling_type=pooling_type, norm_layer=norm_layer,
                norm_kwargs=norm_kwargs, apply_bn=apply_bn_for_resampling,
                conv_after_downsample=conv_after_downsample, redundant_bias
                =redundant_bias)
        if weight_method == 'attn' or weight_method == 'fastattn':
            self.edge_weights = nn.Parameter(torch.ones(len(inputs_offsets)
                ), requires_grad=True)
        else:
            self.edge_weights = None

    def forward(self, x):
        dtype = x[0].dtype
        nodes = []
        for offset in self.inputs_offsets:
            input_node = x[offset]
            input_node = self.resample[str(offset)](input_node)
            nodes.append(input_node)
        if self.weight_method == 'attn':
            normalized_weights = torch.softmax(self.edge_weights.type(dtype
                ), dim=0)
            x = torch.stack(nodes, dim=-1) * normalized_weights
        elif self.weight_method == 'fastattn':
            edge_weights = nn.functional.relu(self.edge_weights.type(dtype))
            weights_sum = torch.sum(edge_weights)
            x = torch.stack([(nodes[i] * edge_weights[i] / (weights_sum + 
                0.0001)) for i in range(len(nodes))], dim=-1)
        elif self.weight_method == 'sum':
            x = torch.stack(nodes, dim=-1)
        else:
            raise ValueError('unknown weight_method {}'.format(self.
                weight_method))
        x = torch.sum(x, dim=-1)
        return x


def bifpn_sum_config(base_reduction=8):
    """BiFPN config with sum."""
    p = OmegaConf.create()
    p.nodes = [{'reduction': base_reduction << 3, 'inputs_offsets': [3, 4]},
        {'reduction': base_reduction << 2, 'inputs_offsets': [2, 5]}, {
        'reduction': base_reduction << 1, 'inputs_offsets': [1, 6]}, {
        'reduction': base_reduction, 'inputs_offsets': [0, 7]}, {
        'reduction': base_reduction << 1, 'inputs_offsets': [1, 7, 8]}, {
        'reduction': base_reduction << 2, 'inputs_offsets': [2, 6, 9]}, {
        'reduction': base_reduction << 3, 'inputs_offsets': [3, 5, 10]}, {
        'reduction': base_reduction << 4, 'inputs_offsets': [4, 11]}]
    p.weight_method = 'sum'
    return p


def bifpn_attn_config():
    """BiFPN config with fast weighted sum."""
    p = bifpn_sum_config()
    p.weight_method = 'attn'
    return p


def bifpn_fa_config():
    """BiFPN config with fast weighted sum."""
    p = bifpn_sum_config()
    p.weight_method = 'fastattn'
    return p


def get_fpn_config(fpn_name):
    if not fpn_name:
        fpn_name = 'bifpn_fa'
    name_to_config = {'bifpn_sum': bifpn_sum_config(), 'bifpn_attn':
        bifpn_attn_config(), 'bifpn_fa': bifpn_fa_config()}
    return name_to_config[fpn_name]


def _init_weight_alt(m, n=''):
    """ Weight initialization alternative, based on EfficientNet bacbkone init w/ class bias addition
    NOTE: this will likely be removed after some experimentation
    """
    if isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        fan_out //= m.groups
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            if 'class_net.predict' in n:
                m.bias.data.fill_(-math.log((1 - 0.01) / 0.01))
            else:
                m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()


def _init_weight(m, n=''):
    """ Weight initialization as per Tensorflow official implementations.
    """

    def _fan_in_out(w, groups=1):
        dimensions = w.dim()
        if dimensions < 2:
            raise ValueError(
                'Fan in and fan out can not be computed for tensor with fewer than 2 dimensions'
                )
        num_input_fmaps = w.size(1)
        num_output_fmaps = w.size(0)
        receptive_field_size = 1
        if w.dim() > 2:
            receptive_field_size = w[0][0].numel()
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size
        fan_out //= groups
        return fan_in, fan_out

    def _glorot_uniform(w, gain=1, groups=1):
        fan_in, fan_out = _fan_in_out(w, groups)
        gain /= max(1.0, (fan_in + fan_out) / 2.0)
        limit = math.sqrt(3.0 * gain)
        w.data.uniform_(-limit, limit)

    def _variance_scaling(w, gain=1, groups=1):
        fan_in, fan_out = _fan_in_out(w, groups)
        gain /= max(1.0, fan_in)
        std = math.sqrt(gain)
        w.data.normal_(std=std)
    if isinstance(m, SeparableConv2d):
        if 'box_net' in n or 'class_net' in n:
            _variance_scaling(m.conv_dw.weight, groups=m.conv_dw.groups)
            _variance_scaling(m.conv_pw.weight)
            if m.conv_pw.bias is not None:
                if 'class_net.predict' in n:
                    m.conv_pw.bias.data.fill_(-math.log((1 - 0.01) / 0.01))
                else:
                    m.conv_pw.bias.data.zero_()
        else:
            _glorot_uniform(m.conv_dw.weight, groups=m.conv_dw.groups)
            _glorot_uniform(m.conv_pw.weight)
            if m.conv_pw.bias is not None:
                m.conv_pw.bias.data.zero_()
    elif isinstance(m, ConvBnAct2d):
        if 'box_net' in n or 'class_net' in n:
            m.conv.weight.data.normal_(std=0.01)
            if m.conv.bias is not None:
                if 'class_net.predict' in n:
                    m.conv.bias.data.fill_(-math.log((1 - 0.01) / 0.01))
                else:
                    m.conv.bias.data.zero_()
        else:
            _glorot_uniform(m.conv.weight)
            if m.conv.bias is not None:
                m.conv.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()


def load_pretrained(model, url, filter_fn=None, strict=True):
    if not url:
        logging.warning(
            'Pretrained model URL is empty, using random initialization. Did you intend to use a `tf_` variant of the model?'
            )
        return
    state_dict = load_state_dict_from_url(url, progress=False, map_location
        ='cpu')
    if filter_fn is not None:
        state_dict = filter_fn(state_dict)
    model.load_state_dict(state_dict, strict=strict)


efficientdet_model_param_dict = dict(efficientdet_d0=dict(name=
    'efficientdet_d0', backbone_name='efficientnet_b0', image_size=512,
    fpn_channels=64, fpn_cell_repeats=3, box_class_repeats=3, pad_type='',
    redundant_bias=False, backbone_args=dict(drop_path_rate=0.2), url=''),
    efficientdet_d1=dict(name='efficientdet_d1', backbone_name=
    'efficientnet_b1', image_size=640, fpn_channels=88, fpn_cell_repeats=4,
    box_class_repeats=3, pad_type='', redundant_bias=False, backbone_args=
    dict(drop_path_rate=0.2), url=''), efficientdet_d2=dict(name=
    'efficientdet_d2', backbone_name='efficientnet_b2', image_size=768,
    fpn_channels=112, fpn_cell_repeats=5, box_class_repeats=3, pad_type='',
    redundant_bias=False, backbone_args=dict(drop_path_rate=0.2), url=''),
    efficientdet_d3=dict(name='efficientdet_d3', backbone_name=
    'efficientnet_b3', image_size=768, fpn_channels=112, fpn_cell_repeats=5,
    box_class_repeats=3, pad_type='', redundant_bias=False, backbone_args=
    dict(drop_path_rate=0.2), url=''), mixdet_m=dict(name='mixdet_m',
    backbone_name='mixnet_m', image_size=512, fpn_channels=64,
    fpn_cell_repeats=3, box_class_repeats=3, pad_type='', redundant_bias=
    False, backbone_args=dict(drop_path_rate=0.2), url=''), mixdet_l=dict(
    name='mixdet_l', backbone_name='mixnet_l', image_size=640, fpn_channels
    =88, fpn_cell_repeats=4, box_class_repeats=3, pad_type='',
    redundant_bias=False, backbone_args=dict(drop_path_rate=0.2), url=''),
    tf_efficientdet_d0=dict(name='tf_efficientdet_d0', backbone_name=
    'tf_efficientnet_b0', image_size=512, fpn_channels=64, fpn_cell_repeats
    =3, box_class_repeats=3, backbone_args=dict(drop_path_rate=0.2), url=
    'https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d0-d92fd44f.pth'
    ), tf_efficientdet_d1=dict(name='tf_efficientdet_d1', backbone_name=
    'tf_efficientnet_b1', image_size=640, fpn_channels=88, fpn_cell_repeats
    =4, box_class_repeats=3, backbone_args=dict(drop_path_rate=0.2), url=
    'https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d1-4c7ebaf2.pth'
    ), tf_efficientdet_d2=dict(name='tf_efficientdet_d2', backbone_name=
    'tf_efficientnet_b2', image_size=768, fpn_channels=112,
    fpn_cell_repeats=5, box_class_repeats=3, backbone_args=dict(
    drop_path_rate=0.2), url=
    'https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d2-cb4ce77d.pth'
    ), tf_efficientdet_d3=dict(name='tf_efficientdet_d3', backbone_name=
    'tf_efficientnet_b3', image_size=896, fpn_channels=160,
    fpn_cell_repeats=6, box_class_repeats=4, backbone_args=dict(
    drop_path_rate=0.2), url=
    'https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d3-b0ea2cbc.pth'
    ), tf_efficientdet_d4=dict(name='tf_efficientdet_d4', backbone_name=
    'tf_efficientnet_b4', image_size=1024, fpn_channels=224,
    fpn_cell_repeats=7, box_class_repeats=4, backbone_args=dict(
    drop_path_rate=0.2), url=
    'https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d4-5b370b7a.pth'
    ), tf_efficientdet_d5=dict(name='tf_efficientdet_d5', backbone_name=
    'tf_efficientnet_b5', image_size=1280, fpn_channels=288,
    fpn_cell_repeats=7, box_class_repeats=4, backbone_args=dict(
    drop_path_rate=0.2), url=
    'https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d5-ef44aea8.pth'
    ), tf_efficientdet_d6=dict(name='tf_efficientdet_d6', backbone_name=
    'tf_efficientnet_b6', image_size=1280, fpn_channels=384,
    fpn_cell_repeats=8, box_class_repeats=5, fpn_name='bifpn_sum',
    backbone_args=dict(drop_path_rate=0.2), url=
    'https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d6-51cb0132.pth'
    ), tf_efficientdet_d7=dict(name='tf_efficientdet_d7', backbone_name=
    'tf_efficientnet_b6', image_size=1536, fpn_channels=384,
    fpn_cell_repeats=8, box_class_repeats=5, anchor_scale=5.0, fpn_name=
    'bifpn_sum', backbone_args=dict(drop_path_rate=0.2), url=
    'https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d7-f05bf714.pth'
    ))


def default_detection_model_configs():
    """Returns a default detection configs."""
    h = OmegaConf.create()
    h.name = 'tf_efficientdet_d1'
    h.backbone_name = 'tf_efficientnet_b1'
    h.backbone_args = None
    h.image_size = 640
    h.num_classes = 90
    h.min_level = 3
    h.max_level = 7
    h.num_levels = h.max_level - h.min_level + 1
    h.num_scales = 3
    h.aspect_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
    h.anchor_scale = 4.0
    h.pad_type = 'same'
    h.box_class_repeats = 3
    h.fpn_cell_repeats = 3
    h.fpn_channels = 88
    h.separable_conv = True
    h.apply_bn_for_resampling = True
    h.conv_after_downsample = False
    h.conv_bn_relu_pattern = False
    h.use_native_resize_op = False
    h.pooling_type = None
    h.redundant_bias = True
    h.fpn_name = None
    h.fpn_config = None
    h.fpn_drop_path_rate = 0.0
    h.alpha = 0.25
    h.gamma = 1.5
    h.delta = 0.1
    h.box_loss_weight = 50.0
    return h


def get_efficientdet_config(model_name='tf_efficientdet_d1'):
    """Get the default config for EfficientDet based on model name."""
    h = default_detection_model_configs()
    h.update(efficientdet_model_param_dict[model_name])
    return h


def create_model(model_name, bench_task='', pretrained=False,
    checkpoint_path='', **kwargs):
    config = get_efficientdet_config(model_name)
    pretrained_backbone = kwargs.pop('pretrained_backbone', True)
    if pretrained or checkpoint_path:
        pretrained_backbone = False
    redundant_bias = kwargs.pop('redundant_bias', None)
    if redundant_bias is not None:
        config.redundant_bias = redundant_bias
    model = EfficientDet(config, pretrained_backbone=pretrained_backbone,
        **kwargs)
    if checkpoint_path:
        load_checkpoint(model, checkpoint_path)
    elif pretrained:
        load_pretrained(model, config.url)
    if bench_task == 'train':
        model = DetBenchTrain(model, config)
    elif bench_task == 'predict':
        model = DetBenchPredict(model, config)
    return model


class EfficientDet(nn.Module):

    def __init__(self, config, norm_kwargs=None, pretrained_backbone=True,
        alternate_init=False):
        super(EfficientDet, self).__init__()
        norm_kwargs = norm_kwargs or dict(eps=0.001, momentum=0.01)
        self.backbone = create_model(config.backbone_name, features_only=
            True, out_indices=(2, 3, 4), pretrained=pretrained_backbone, **
            config.backbone_args)
        feature_info = [dict(num_chs=f['num_chs'], reduction=f['reduction']
            ) for i, f in enumerate(self.backbone.feature_info())]
        self.fpn = BiFpn(config, feature_info, norm_kwargs=norm_kwargs)
        self.class_net = HeadNet(config, num_outputs=config.num_classes,
            norm_kwargs=norm_kwargs)
        self.box_net = HeadNet(config, num_outputs=4, norm_kwargs=norm_kwargs)
        for n, m in self.named_modules():
            if 'backbone' not in n:
                if alternate_init:
                    _init_weight_alt(m, n)
                else:
                    _init_weight(m, n)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fpn(x)
        x_class = self.class_net(x)
        x_box = self.box_net(x)
        return x_class, x_box


def huber_loss(input, target, delta: float=1.0, weights: Optional[torch.
    Tensor]=None, size_average: bool=True):
    """
    """
    err = input - target
    abs_err = err.abs()
    quadratic = torch.clamp(abs_err, max=delta)
    linear = abs_err - quadratic
    loss = 0.5 * quadratic.pow(2) + delta * linear
    if weights is not None:
        loss *= weights
    return loss.mean() if size_average else loss.sum()


def _box_loss(box_outputs, box_targets, num_positives, delta: float=0.1):
    """Computes box regression loss."""
    normalizer = num_positives * 4.0
    mask = box_targets != 0.0
    box_loss = huber_loss(box_targets, box_outputs, weights=mask, delta=
        delta, size_average=False)
    box_loss /= normalizer
    return box_loss


def focal_loss(logits, targets, alpha: float, gamma: float, normalizer):
    """Compute the focal loss between `logits` and the golden `target` values.

    Focal loss = -(1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.

    Args:
        logits: A float32 tensor of size [batch, height_in, width_in, num_predictions].

        targets: A float32 tensor of size [batch, height_in, width_in, num_predictions].

        alpha: A float32 scalar multiplying alpha to the loss from positive examples
            and (1-alpha) to the loss from negative examples.

        gamma: A float32 scalar modulating loss from hard and easy examples.

         normalizer: A float32 scalar normalizes the total loss from all examples.

    Returns:
        loss: A float32 scalar representing normalized total loss.
    """
    positive_label_mask = targets == 1.0
    cross_entropy = F.binary_cross_entropy_with_logits(logits, targets.to(
        logits.dtype), reduction='none')
    neg_logits = -1.0 * logits
    modulator = torch.exp(gamma * targets * neg_logits - gamma * torch.
        log1p(torch.exp(neg_logits)))
    loss = modulator * cross_entropy
    weighted_loss = torch.where(positive_label_mask, alpha * loss, (1.0 -
        alpha) * loss)
    weighted_loss /= normalizer
    return weighted_loss


def _classification_loss(cls_outputs, cls_targets, num_positives, alpha:
    float=0.25, gamma: float=2.0):
    """Computes classification loss."""
    normalizer = num_positives
    classification_loss = focal_loss(cls_outputs, cls_targets, alpha, gamma,
        normalizer)
    return classification_loss


class DetectionLoss(nn.Module):

    def __init__(self, config):
        super(DetectionLoss, self).__init__()
        self.config = config
        self.num_classes = config.num_classes
        self.alpha = config.alpha
        self.gamma = config.gamma
        self.delta = config.delta
        self.box_loss_weight = config.box_loss_weight

    def forward(self, cls_outputs: List[torch.Tensor], box_outputs: List[
        torch.Tensor], cls_targets: List[torch.Tensor], box_targets: List[
        torch.Tensor], num_positives: torch.Tensor):
        """Computes total detection loss.
        Computes total detection loss including box and class loss from all levels.
        Args:
            cls_outputs: a List with values representing logits in [batch_size, height, width, num_anchors].
                at each feature level (index)

            box_outputs: a List with values representing box regression targets in
                [batch_size, height, width, num_anchors * 4] at each feature level (index)

            cls_targets: groundtruth class targets.

            box_targets: groundtrusth box targets.

            num_positives: num positive grountruth anchors

        Returns:
            total_loss: an integer tensor representing total loss reducing from class and box losses from all levels.

            cls_loss: an integer tensor representing total class loss.

            box_loss: an integer tensor representing total box regression loss.
        """
        num_positives_sum = num_positives.sum() + 1.0
        levels = len(cls_outputs)
        cls_losses = []
        box_losses = []
        for l in range(levels):
            cls_targets_at_level = cls_targets[l]
            box_targets_at_level = box_targets[l]
            cls_targets_non_neg = cls_targets_at_level >= 0
            cls_targets_at_level_oh = F.one_hot(cls_targets_at_level *
                cls_targets_non_neg, self.num_classes)
            cls_targets_at_level_oh = torch.where(cls_targets_non_neg.
                unsqueeze(-1), cls_targets_at_level_oh, torch.zeros_like(
                cls_targets_at_level_oh))
            bs, height, width, _, _ = cls_targets_at_level_oh.shape
            cls_targets_at_level_oh = cls_targets_at_level_oh.view(bs,
                height, width, -1)
            cls_loss = _classification_loss(cls_outputs[l].permute(0, 2, 3,
                1), cls_targets_at_level_oh, num_positives_sum, alpha=self.
                alpha, gamma=self.gamma)
            cls_loss = cls_loss.view(bs, height, width, -1, self.num_classes)
            cls_loss *= (cls_targets_at_level != -2).unsqueeze(-1).float()
            cls_losses.append(cls_loss.sum())
            box_losses.append(_box_loss(box_outputs[l].permute(0, 2, 3, 1),
                box_targets_at_level, num_positives_sum, delta=self.delta))
        cls_loss = torch.sum(torch.stack(cls_losses, dim=-1), dim=-1)
        box_loss = torch.sum(torch.stack(box_losses, dim=-1), dim=-1)
        total_loss = cls_loss + self.box_loss_weight * box_loss
        return total_loss, cls_loss, box_loss


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_rwightman_efficientdet_pytorch(_paritybench_base):
    pass
    def test_000(self):
        self._check(ResampleFeatureMap(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(SequentialAppend(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(SequentialAppendLast(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

