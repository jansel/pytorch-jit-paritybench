import sys
_module = sys.modules[__name__]
del sys
realtime_panoptic = _module
config = _module
defaults = _module
data = _module
panoptic_transform = _module
layers = _module
scale = _module
models = _module
backbones = _module
panoptic_from_dense_box = _module
rt_pano_net = _module
utils = _module
bounding_box = _module
boxlist_ops = _module
visualization = _module
demo = _module

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


from torch import nn


from torchvision.models import resnet


from torchvision.models._utils import IntermediateLayerGetter


from torchvision.ops import misc as misc_nn_ops


from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork


from torchvision.ops.feature_pyramid_network import LastLevelP6P7


import torch.nn.functional as F


import math


from collections import OrderedDict


from torchvision.ops.boxes import nms as _box_nms


import copy


import numpy as np


import warnings


from torchvision.models.detection.image_list import ImageList


class Scale(nn.Module):

    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        """Scale layer with trainable scale factor.

        Parameters
        ----------
        init_value: float
            Initial value of the scale factor.
        """
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


class ResNetWithModifiedFPN(nn.Module):
    """Adds a p67-FPN on top of a ResNet model with more options.

    We adopt this function from torchvision.models.detection.backbone_utils.
    Modification has been added to enable RetinaNet style FPN with P6 P7 as extra blocks.

    Parameters
    ----------
    backbone_name: string 
        Resnet architecture supported by torchvision. Possible values are 'ResNet', 'resnet18', 'resnet34', 'resnet50',
         'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2'

    norm_layer: torchvision.ops
        It is recommended to use the default value. For details visit:
        (https://github.com/facebookresearch/maskrcnn-benchmark/issues/267)

    pretrained: bool
        If True, returns a model with backbone pre-trained on Imagenet. Default: False

    trainable_layers: int
        Number of trainable (not frozen) resnet layers starting from final block.
        Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable.

    out_channels: int
        number of channels in the FPN.
    """

    def __init__(self, backbone_name, pretrained=False, norm_layer=misc_nn_ops.FrozenBatchNorm2d, trainable_layers=3, out_channels=256):
        super().__init__()
        backbone = resnet.__dict__[backbone_name](pretrained=pretrained, norm_layer=norm_layer)
        assert 0 <= trainable_layers <= 5
        layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:trainable_layers]
        for name, parameter in backbone.named_parameters():
            if all([(not name.startswith(layer)) for layer in layers_to_train]):
                parameter.requires_grad_(False)
        return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
        in_channels_stage2 = backbone.inplanes // 8
        self.in_channels_list = [0, in_channels_stage2 * 2, in_channels_stage2 * 4, in_channels_stage2 * 8]
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.fpn = FeaturePyramidNetwork(in_channels_list=self.in_channels_list[1:], out_channels=out_channels, extra_blocks=LastLevelP6P7(out_channels, out_channels))
        self.out_channels = out_channels

    def forward(self, x):
        x = self.body(x)
        keys = list(x.keys())
        for idx, key in enumerate(keys):
            if self.in_channels_list[idx] == 0:
                del x[key]
        x = self.fpn(x)
        return x


FLIP_LEFT_RIGHT = 0


FLIP_TOP_BOTTOM = 1


ROTATE_90 = 2


class BoxList:
    """This class represents a set of bounding boxes.
    The bounding boxes are represented as a Nx4 Tensor.
    In order to uniquely determine the bounding boxes with respect
    to an image, we also store the corresponding image dimensions.
    They can contain extra information that is specific to each bounding box, such as
    labels.
    """

    def __init__(self, bbox, image_size, mode='xyxy'):
        """Initial function.

        Parameters
        ----------
        bbox: tensor
            Nx4 tensor following bounding box parameterization defined by "mode".

        image_size: list
            [W,H] Image size.

        mode: str
            Bounding box parameterization. 'xyxy' or 'xyhw'.
        """
        device = bbox.device if isinstance(bbox, torch.Tensor) else torch.device('cpu')
        bbox = torch.as_tensor(bbox, dtype=torch.float32, device=device)
        if bbox.ndimension() != 2:
            raise ValueError('bbox should have 2 dimensions, got {}'.format(bbox.ndimension()), bbox)
        if bbox.size(-1) != 4:
            raise ValueError('last dimension of bbox should have a size of 4, got {}'.format(bbox.size(-1)))
        if mode not in ('xyxy', 'xywh'):
            raise ValueError("mode should be 'xyxy' or 'xywh'")
        self.bbox = bbox
        self.size = image_size
        self.mode = mode
        self.extra_fields = {}

    def add_field(self, field, field_data):
        """Add a field to boxlist.
        """
        self.extra_fields[field] = field_data

    def get_field(self, field):
        """Get a field from boxlist.
        """
        return self.extra_fields[field]

    def has_field(self, field):
        """Check if certain field exist in boxlist
        """
        return field in self.extra_fields

    def fields(self):
        """Get all available field names.
        """
        return list(self.extra_fields.keys())

    def _copy_extra_fields(self, bbox):
        """Copy extra fields from given boxlist to current boxlist.
        """
        for k, v in bbox.extra_fields.items():
            self.extra_fields[k] = v

    def convert(self, mode):
        """Convert bounding box parameterization mode.
        """
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
            bbox = torch.cat((xmin, ymin, xmax - xmin + TO_REMOVE, ymax - ymin + TO_REMOVE), dim=-1)
            bbox = BoxList(bbox, self.size, mode=mode)
        bbox._copy_extra_fields(self)
        return bbox

    def _split_into_xyxy(self):
        """split lists of bounding box corners. 
        """
        if self.mode == 'xyxy':
            xmin, ymin, xmax, ymax = self.bbox.split(1, dim=-1)
            return xmin, ymin, xmax, ymax
        elif self.mode == 'xywh':
            TO_REMOVE = 1
            xmin, ymin, w, h = self.bbox.split(1, dim=-1)
            return xmin, ymin, xmin + (w - TO_REMOVE).clamp(min=0), ymin + (h - TO_REMOVE).clamp(min=0)
        else:
            raise RuntimeError('Should not be here')

    def resize(self, size, *args, **kwargs):
        """Returns a resized copy of this bounding box.

        Parameters
        ----------
        size: list or tuple
            The requested image size in pixels, as a 2-tuple:
            (width, height).
        """
        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(size, self.size))
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
        scaled_box = torch.cat((scaled_xmin, scaled_ymin, scaled_xmax, scaled_ymax), dim=-1)
        bbox = BoxList(scaled_box, size, mode='xyxy')
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.resize(size, *args, **kwargs)
            bbox.add_field(k, v)
        return bbox.convert(self.mode)

    def transpose(self, method):
        """Transpose bounding box (flip or rotate in 90 degree steps)

        Parameters
        ----------
        method: str
            One of:py:attr:`PIL.Image.FLIP_LEFT_RIGHT`,
           :py:attr:`PIL.Image.FLIP_TOP_BOTTOM`,:py:attr:`PIL.Image.ROTATE_90`,
           :py:attr:`PIL.Image.ROTATE_180`,:py:attr:`PIL.Image.ROTATE_270`,
           :py:attr:`PIL.Image.TRANSPOSE` or:py:attr:`PIL.Image.TRANSVERSE`.
        """
        if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM, ROTATE_90):
            raise NotImplementedError('Only FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM and ROTATE_90 implemented')
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
        elif method == ROTATE_90:
            transposed_xmin = ymin * image_width / image_height
            transposed_xmax = ymax * image_width / image_height
            transposed_ymin = (image_width - xmax) * image_height / image_width
            transposed_ymax = (image_width - xmin) * image_height / image_width
        transposed_boxes = torch.cat((transposed_xmin, transposed_ymin, transposed_xmax, transposed_ymax), dim=-1)
        bbox = BoxList(transposed_boxes, self.size, mode='xyxy')
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.transpose(method)
            bbox.add_field(k, v)
        return bbox.convert(self.mode)

    def translate(self, x_offset, y_offset):
        """Translate bounding box.

        Parameters
        ----------
        x_offseflt: float
            x offset
        y_offset: float
            y offset
        """
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        translated_xmin = xmin + x_offset
        translated_xmax = xmax + x_offset
        translated_ymin = ymin + y_offset
        translated_ymax = ymax + y_offset
        translated_boxes = torch.cat((translated_xmin, translated_ymin, translated_xmax, translated_ymax), dim=-1)
        bbox = BoxList(translated_boxes, self.size, mode='xyxy')
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.translate(x_offset, y_offset)
            bbox.add_field(k, v)
        return bbox.convert(self.mode)

    def crop(self, box):
        """Crop a rectangular region from this bounding box.
        
        Parameters
        ----------
        box: tuple
            The box is a 4-tuple defining the left, upper, right, and lower pixel
            coordinate.
        """
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        w, h = box[2] - box[0], box[3] - box[1]
        cropped_xmin = (xmin - box[0]).clamp(min=0, max=w)
        cropped_ymin = (ymin - box[1]).clamp(min=0, max=h)
        cropped_xmax = (xmax - box[0]).clamp(min=0, max=w)
        cropped_ymax = (ymax - box[1]).clamp(min=0, max=h)
        if False:
            is_empty = (cropped_xmin == cropped_xmax) | (cropped_ymin == cropped_ymax)
        cropped_box = torch.cat((cropped_xmin, cropped_ymin, cropped_xmax, cropped_ymax), dim=-1)
        bbox = BoxList(cropped_box, (w, h), mode='xyxy')
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.crop(box)
            bbox.add_field(k, v)
        return bbox.convert(self.mode)

    def augmentation_crop(self, top, left, crop_height, crop_width):
        """Random cropping of the bounding box (bbox).
        This function is created for label box to be crop at training time.

        Parameters:
        -----------
        top: int
            Top pixel position of crop area

        left: int
            left pixel position of crop area

        crop_height: int
            Height of crop area

        crop_width: int
            Width of crop area

        Returns:
        --------
        bbox_cropped: BoxList
            A BoxList object with instances after cropping. If no valid instance is left after
            cropping, return None.


        """
        masks = self.extra_fields['masks']
        masks_cropped, keep_ids = masks.augmentation_crop(top, left, crop_height, crop_width)
        if not keep_ids:
            return None
        assert masks_cropped.mode == 'poly'
        bbox_cropped = []
        labels = self.extra_fields['labels']
        labels_cropped = [labels[idx] for idx in keep_ids]
        labels_cropped = torch.as_tensor(labels_cropped, dtype=torch.long)
        crop_box_xyxy = [float(left), float(top), float(left + crop_width), float(top + crop_height)]
        self.extra_fields.pop('masks', None)
        new_bbox = self.crop(crop_box_xyxy).convert('xyxy')
        for mask_id, box_id in enumerate(keep_ids):
            x1, y1, x2, y2 = new_bbox.bbox[box_id].numpy()
            if x1 > 0 and y1 > 0 and x2 < crop_width - 1 and y2 < crop_height - 1:
                bbox_cropped.append([x1, y1, x2, y2])
            else:
                current_polygon_instance = masks_cropped.instances.polygons[mask_id]
                x_ids = []
                y_ids = []
                for poly in current_polygon_instance.polygons:
                    p = poly.clone()
                    x_ids.extend(p[0::2])
                    y_ids.extend(p[1::2])
                bbox_cropped.append([min(x_ids), min(y_ids), max(x_ids), max(y_ids)])
        bbox_cropped = BoxList(bbox_cropped, (crop_width, crop_height), mode='xyxy')
        bbox_cropped = bbox_cropped.convert(self.mode)
        bbox_cropped.add_field('masks', masks_cropped)
        bbox_cropped.add_field('labels', labels_cropped)
        return bbox_cropped

    def to(self, device):
        """Move object to torch device.
        """
        bbox = BoxList(self.bbox, self.size, self.mode)
        for k, v in self.extra_fields.items():
            if hasattr(v, 'to'):
                v = v
            bbox.add_field(k, v)
        return bbox

    def __getitem__(self, item):
        """Get a sub-list of Boxlist as a new Boxlist
        """
        item_bbox = self.bbox[item]
        if len(item_bbox.shape) < 2:
            item_bbox.unsqueeze(0)
        bbox = BoxList(item_bbox, self.size, self.mode)
        for k, v in self.extra_fields.items():
            bbox.add_field(k, v[item])
        return bbox

    def __len__(self):
        return self.bbox.shape[0]

    def clip_to_image(self, remove_empty=True):
        """Clip bounding box coordinates according to the image range. 
        """
        TO_REMOVE = 1
        self.bbox[:, 0].clamp_(min=0, max=self.size[0] - TO_REMOVE)
        self.bbox[:, 1].clamp_(min=0, max=self.size[1] - TO_REMOVE)
        self.bbox[:, 2].clamp_(min=0, max=self.size[0] - TO_REMOVE)
        self.bbox[:, 3].clamp_(min=0, max=self.size[1] - TO_REMOVE)
        if remove_empty:
            box = self.bbox
            keep = (box[:, 3] > box[:, 1]) & (box[:, 2] > box[:, 0])
            return self[keep]
        return self

    def area(self, idx=None):
        """Get bounding box area.
        """
        box = self.bbox if idx is None else self.bbox[idx].unsqueeze(0)
        if self.mode == 'xyxy':
            TO_REMOVE = 1
            area = (box[:, 2] - box[:, 0] + TO_REMOVE) * (box[:, 3] - box[:, 1] + TO_REMOVE)
        elif self.mode == 'xywh':
            area = box[:, 2] * box[:, 3]
        else:
            raise RuntimeError('Should not be here')
        return area

    def copy_with_fields(self, fields, skip_missing=False):
        """Provide deep copy of Boxlist with requested fields.
        """
        bbox = BoxList(self.bbox, self.size, self.mode)
        if not isinstance(fields, (list, tuple)):
            fields = [fields]
        for field in fields:
            if self.has_field(field):
                bbox.add_field(field, self.get_field(field))
            elif not skip_missing:
                raise KeyError("Field '{}' not found in {}".format(field, self))
        return bbox

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += 'num_boxes={}, '.format(len(self))
        s += 'image_width={}, '.format(self.size[0])
        s += 'image_height={}, '.format(self.size[1])
        s += 'mode={})'.format(self.mode)
        return s


def boxlist_nms(boxlist, nms_thresh, max_proposals=-1, score_field='scores'):
    """Performs non-maximum suppression on a boxlist.
    The ranking scores are specified in a boxlist field via score_field.

    Parameters
    ----------
    boxlist : BoxList
        Original boxlist

    nms_thresh : float
        NMS threshold

    max_proposals :  int
        If > 0, then only the top max_proposals are kept after non-maximum suppression

    score_field : str
        Boxlist field to use during NMS score ranking. Field value needs to be numeric.
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


def cat_boxlist(bboxes):
    """Concatenates a list of BoxList  into a single BoxList
    image sizes needs to be same in this operation.

    Parameters
    ----------
    bboxes : list[BoxList]
    """
    assert isinstance(bboxes, (list, tuple))
    assert all(isinstance(bbox, BoxList) for bbox in bboxes)
    size = bboxes[0].size
    assert all(bbox.size == size for bbox in bboxes)
    mode = bboxes[0].mode
    assert all(bbox.mode == mode for bbox in bboxes)
    fields = set(bboxes[0].fields())
    assert all(set(bbox.fields()) == fields for bbox in bboxes)
    cat_boxes = BoxList(torch.cat([bbox.bbox for bbox in bboxes], dim=0), size, mode)
    for field in fields:
        data = torch.cat([bbox.get_field(field) for bbox in bboxes], dim=0)
        cat_boxes.add_field(field, data)
    return cat_boxes


def remove_small_boxes(boxlist, min_size):
    """Only keep boxes with both sides >= min_size

    Parameters
    ----------
    boxlist : Boxlist
        Original boxlist

    min_size : int
        Max edge dimension of boxes to be kept.
    """
    xywh_boxes = boxlist.convert('xywh').bbox
    _, _, ws, hs = xywh_boxes.unbind(dim=1)
    keep = ((ws >= min_size) & (hs >= min_size)).nonzero().squeeze(1)
    return boxlist[keep]


class PanopticFromDenseBox:
    """Performs post-processing on the outputs of the RTPanonet.

    Parameters
    ----------
    pre_nms_thresh: float
        Acceptance class probability threshold for bounding box candidates before NMS.

    pre_nms_top_n: int
        Maximum number of accepted bounding box candidates before NMS.

    nms_thresh: float
        NMS threshold.

    fpn_post_nms_top_n: int
        Maximum number of detected object per image.

    min_size: int
        Minimum dimension of accepted detection.

    num_classes: int
        Number of total semantic classes (stuff and things).

    mask_thresh: float
        Bounding box IoU threshold to determined 'similar bounding box' in mask reconstruction.

    instance_id_range: list of int
        [min_id, max_id] defines the range of id in 1:num_classes that corresponding to thing classes.

    is_training: bool
        Whether the current process is during training process.
    """

    def __init__(self, pre_nms_thresh, pre_nms_top_n, nms_thresh, fpn_post_nms_top_n, min_size, num_classes, mask_thresh, instance_id_range, is_training):
        super(PanopticFromDenseBox, self).__init__()
        self.pre_nms_thresh = pre_nms_thresh
        self.pre_nms_top_n = pre_nms_top_n
        self.nms_thresh = nms_thresh
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.min_size = min_size
        self.num_classes = num_classes
        self.mask_thresh = mask_thresh
        self.instance_id_range = instance_id_range
        self.is_training = is_training

    def process(self, locations, box_cls, box_regression, centerness, levelness_logits, semantic_logits, image_sizes):
        """ Reconstruct panoptic segmentation result from raw predictions.

        This function conduct post processing of panoptic head raw prediction, including bounding box
        prediction, semantic segmentation and levelness to reconstruct instance segmentation results.

        Parameters
        ----------
        locations: list of torch.Tensor
            Corresponding pixel locations of each FPN predictions.

        box_cls: list of torch.Tensor
            Predicted bounding box class from each FPN layers.

        box_regression: list of torch.Tensor
            Predicted bounding box offsets from each FPN layers.

        centerness: list of torch.Tensor
            Predicted object centerness from each FPN layers.

        levelness_logits:
            Global prediction of best source FPN layer for each pixel location.

        semantic_logits:
            Global prediction of semantic segmentation.

        image_sizes: list of [int,int]
            Image sizes.

        Returns:
        --------
        boxlists: list of BoxList
            reconstructed instances with masks.
        """
        num_locs_per_level = [len(loc_per_level) for loc_per_level in locations]
        sampled_boxes = []
        for i, (l, o, b, c) in enumerate(zip(locations[:-1], box_cls, box_regression, centerness)):
            if self.is_training:
                layer_boxes = self.forward_for_single_feature_map(l, o, b, c, image_sizes)
                for layer_box in layer_boxes:
                    pred_indices = layer_box.get_field('indices')
                    pred_indices = pred_indices + sum(num_locs_per_level[:i])
                    layer_box.add_field('indices', pred_indices)
                sampled_boxes.append(layer_boxes)
            else:
                sampled_boxes.append(self.forward_for_single_feature_map(l, o, b, c, image_sizes))
        boxlists = list(zip(*sampled_boxes))
        try:
            boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]
            boxlists = self.select_over_all_levels(boxlists)
        except Exception as e:
            None
            for boxlist in boxlists:
                for box in boxlist:
                    None
        levelness_locations = locations[-1]
        _, c_semantic, _, _ = semantic_logits.shape
        N, _, h_map, w_map = levelness_logits.shape
        bounding_box_feature_map = self.generate_box_feature_map(levelness_locations, box_regression, levelness_logits)
        semantic_logits = F.interpolate(semantic_logits, size=(h_map, w_map), mode='bilinear')
        semantic_logits = semantic_logits.view(N, c_semantic, h_map, w_map).permute(0, 2, 3, 1)
        semantic_logits = semantic_logits.reshape(N, -1, c_semantic)
        semantic_probability = F.softmax(semantic_logits, dim=2)
        semantic_probability = semantic_probability[:, :, self.instance_id_range[0]:]
        boxlists = self.mask_reconstruction(boxlists=boxlists, box_feature_map=bounding_box_feature_map, semantic_prob=semantic_probability, box_feature_map_location=levelness_locations, h_map=h_map, w_map=w_map)
        if not self.is_training:
            for boxlist in boxlists:
                masks = boxlist.get_field('mask')
                w, h = boxlist.size
                if len(masks.shape) == 3 and masks.shape[0] != 0:
                    masks = F.interpolate(masks.unsqueeze(0), size=(h_map * 4, w_map * 4), mode='bilinear').squeeze()
                else:
                    masks = masks.view([-1, h_map * 4, w_map * 4])
                masks = masks >= self.mask_thresh
                if len(masks.shape) < 3:
                    masks = masks.unsqueeze(0)
                masks = masks[:, 0:h, 0:w].contiguous()
                boxlist.add_field('mask', masks)
        return boxlists

    def forward_for_single_feature_map(self, locations, box_cls, box_regression, centerness, image_sizes):
        """Recover dense bounding box detection results from raw predictions for each FPN layer.

        Parameters
        ----------
        locations: torch.Tensor
            Corresponding pixel location of FPN feature map with size of (N, H * W, 2).

        box_cls: torch.Tensor
            Predicted bounding box class probability with size of (N, C, H, W).

        box_regression: torch.Tensor
            Predicted bounding box offset centered at corresponding pixel with size of (N, 4, H, W).

        centerness: torch.Tensor
            Predicted centerness of corresponding pixel with size of (N, 1, H, W).

        Note: N is the number of FPN level.

        Returns
        -------
        results: List of BoxList
            A list of dense bounding boxes from each FPN layer.
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
        box_cls = box_cls * centerness[:, :, None]
        results = []
        for i in range(N):
            per_box_cls = box_cls[i]
            per_candidate_inds = candidate_inds[i]
            per_box_cls = per_box_cls[per_candidate_inds]
            per_candidate_nonzeros = per_candidate_inds.nonzero()
            per_box_loc = per_candidate_nonzeros[:, 0]
            per_class = per_candidate_nonzeros[:, 1] + 1
            per_box_regression = box_regression[i]
            per_box_regression = per_box_regression[per_box_loc]
            per_locations = locations[per_box_loc]
            per_pre_nms_top_n = pre_nms_top_n[i]
            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                per_box_cls, top_k_indices = per_box_cls.topk(per_pre_nms_top_n, sorted=False)
                per_class = per_class[top_k_indices]
                per_box_regression = per_box_regression[top_k_indices]
                per_locations = per_locations[top_k_indices]
                if self.is_training:
                    per_box_loc = per_box_loc[top_k_indices]
            detections = torch.stack([per_locations[:, 0] - per_box_regression[:, 0], per_locations[:, 1] - per_box_regression[:, 1], per_locations[:, 0] + per_box_regression[:, 2], per_locations[:, 1] + per_box_regression[:, 3]], dim=1)
            h, w = image_sizes[i]
            boxlist = BoxList(detections, (int(w), int(h)), mode='xyxy')
            boxlist.add_field('labels', per_class)
            boxlist.add_field('scores', per_box_cls)
            if self.is_training:
                boxlist.add_field('indices', per_box_loc)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = remove_small_boxes(boxlist, self.min_size)
            results.append(boxlist)
        return results

    def generate_box_feature_map(self, location, box_regression, levelness_logits):
        """Generate bounding box feature aggregating dense bounding box predictions.

        Parameters
        ----------
        location: torch.Tensor
            Pixel location of levelness.

        box_regression: list of torch.Tensor
            Bounding box offsets from each FPN.

        levelness_logits: torch.Tenor
            Global prediction of best source FPN layer for each pixel location.
            Predict at the resolution of (H/4, W/4).

        Returns
        -------
        bounding_box_feature_map: torch.Tensor
            Aggregated bounding box feature map.
        """
        upscaled_box_reg = []
        N, _, h_map, w_map = levelness_logits.shape
        downsampled_shape = torch.Size((h_map, w_map))
        for box_reg in box_regression:
            upscaled_box_reg.append(F.interpolate(box_reg, size=downsampled_shape, mode='bilinear').unsqueeze(1))
        upscaled_box_reg = torch.cat(upscaled_box_reg, 1)
        max_v, level = torch.max(levelness_logits[:, 1:, :, :], dim=1)
        box_feature_map = torch.gather(upscaled_box_reg, dim=1, index=level.unsqueeze(1).expand([N, 4, h_map, w_map]).unsqueeze(1))
        box_feature_map = box_feature_map.view(N, 4, h_map, w_map).permute(0, 2, 3, 1)
        box_feature_map = box_feature_map.reshape(N, -1, 4)
        levelness_locations_repeat = location.repeat(N, 1, 1)
        bounding_box_feature_map = torch.stack([levelness_locations_repeat[:, :, 0] - box_feature_map[:, :, 0], levelness_locations_repeat[:, :, 1] - box_feature_map[:, :, 1], levelness_locations_repeat[:, :, 0] + box_feature_map[:, :, 2], levelness_locations_repeat[:, :, 1] + box_feature_map[:, :, 3]], dim=2)
        return bounding_box_feature_map

    def mask_reconstruction(self, boxlists, box_feature_map, semantic_prob, box_feature_map_location, h_map, w_map):
        """Reconstruct instance mask from dense bounding box and semantic smoothing.

        Parameters
        ----------
        boxlists: List of Boxlist
            Object detection result after NMS.

        box_feature_map: torch.Tensor
            Aggregated bounding box feature map.

        semantic_prob: torch.Tensor
            Prediction semantic probability.

        box_feature_map_location: torch.Tensor
            Corresponding pixel location of bounding box feature map.

        h_map: int
            Height of bounding box feature map.

        w_map: int
            Width of bounding box feature map.
        """
        for i, (boxlist, per_image_bounding_box_feature_map, per_image_semantic_prob, box_feature_map_loc) in enumerate(zip(boxlists, box_feature_map, semantic_prob, box_feature_map_location)):
            if len(boxlist) > 0:
                query_boxes = boxlist.bbox
                propose_cls = boxlist.get_field('labels')
                propose_bbx = query_boxes.unsqueeze(2).repeat(1, 1, per_image_bounding_box_feature_map.shape[0]).permute(0, 2, 1)
                voting_bbx = per_image_bounding_box_feature_map.permute(1, 0).unsqueeze(0).repeat(query_boxes.shape[0], 1, 1).permute(0, 2, 1)
                proposal_area = (propose_bbx[:, :, 2] - propose_bbx[:, :, 0]) * (propose_bbx[:, :, 3] - propose_bbx[:, :, 1])
                voting_area = (voting_bbx[:, :, 2] - voting_bbx[:, :, 0]) * (voting_bbx[:, :, 3] - voting_bbx[:, :, 1])
                w_intersect = torch.min(voting_bbx[:, :, 2], propose_bbx[:, :, 2]) - torch.max(voting_bbx[:, :, 0], propose_bbx[:, :, 0])
                h_intersect = torch.min(voting_bbx[:, :, 3], propose_bbx[:, :, 3]) - torch.max(voting_bbx[:, :, 1], propose_bbx[:, :, 1])
                w_intersect = w_intersect.clamp(min=0.0)
                h_intersect = h_intersect.clamp(min=0.0)
                w_general = torch.max(voting_bbx[:, :, 2], propose_bbx[:, :, 2]) - torch.min(voting_bbx[:, :, 0], propose_bbx[:, :, 0])
                h_general = torch.max(voting_bbx[:, :, 3], propose_bbx[:, :, 3]) - torch.min(voting_bbx[:, :, 1], propose_bbx[:, :, 1])
                area_intersect = w_intersect * h_intersect
                area_union = proposal_area + voting_area - area_intersect
                torch.cuda.synchronize()
                area_general = w_general * h_general + 1e-07
                bbox_correlation_map = (area_intersect + 1.0) / (area_union + 1.0) - (area_general - area_union) / area_general
                per_image_cls_prob = per_image_semantic_prob[:, propose_cls - 1].permute(1, 0)
                bbox_correlation_map = bbox_correlation_map * per_image_cls_prob
                masks = bbox_correlation_map.view(query_boxes.shape[0], h_map, w_map)
                if len(masks.shape) < 3:
                    masks = masks.unsqueeze(0)
                boxlist.add_field('mask', masks)
            else:
                dummy_masks = torch.zeros(len(boxlist), h_map, w_map).float().to(boxlist.bbox.device)
                boxlist.add_field('mask', dummy_masks)
        return boxlists

    def select_over_all_levels(self, boxlists):
        """NMS of bounding box candidates.

        Parameters
        ----------
        boxlists: list of Boxlist
            Pre-NMS bounding boxes.

        Returns
        -------
        results: list of Boxlist
            Final detection result.
        """
        num_images = len(boxlists)
        results = []
        for i in range(num_images):
            boxlist = boxlists[i]
            scores = boxlist.get_field('scores')
            labels = boxlist.get_field('labels')
            if self.is_training:
                indices = boxlist.get_field('indices')
            boxes = boxlist.bbox
            result = []
            w, h = boxlist.size
            if boxes.shape[0] < 1:
                results.append(boxlist)
                continue
            for j in range(1, self.num_classes):
                inds = (labels == j).nonzero().view(-1)
                if len(inds) > 0:
                    scores_j = scores[inds]
                    boxes_j = boxes[inds, :].view(-1, 4)
                    boxlist_for_class = BoxList(boxes_j, boxlist.size, mode='xyxy')
                    boxlist_for_class.add_field('scores', scores_j)
                    if self.is_training:
                        indices_j = indices[inds]
                        boxlist_for_class.add_field('indices', indices_j)
                    boxlist_for_class = boxlist_nms(boxlist_for_class, self.nms_thresh, score_field='scores')
                    num_labels = len(boxlist_for_class)
                    boxlist_for_class.add_field('labels', torch.full((num_labels,), j, dtype=torch.int64, device=scores.device))
                    result.append(boxlist_for_class)
            result = cat_boxlist(result)
            result = boxlist_nms(result, 0.97, score_field='scores')
            number_of_detections = len(result)
            if number_of_detections > self.fpn_post_nms_top_n > 0:
                cls_scores = result.get_field('scores')
                image_thresh, _ = torch.kthvalue(cls_scores.cpu(), number_of_detections - self.fpn_post_nms_top_n + 1)
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]
            results.append(result)
        return results


class PanopticHead(torch.nn.Module):
    """Network module of Panoptic Head extracting features from FPN feature maps.

    Parameters
    ----------
    num_classes: int
        Number of total classes, including 'things' and 'stuff'.

    things_num_classes: int
        number of thing classes. 
    
    num_fpn_levels: int
        Number of FPN levels.

    fpn_strides: list 
        FPN strides at each FPN scale.

    in_channels: int
        Number of channels of the input features (output of FPN)

    norm_reg_targets: bool
        If true, train on normalized target.

    centerness_on_reg: bool
        If true, regress centerness on box tower of FCOS.

    fcos_num_convs: int
        number of convolution modules used in FCOS towers.

    fcos_norm: str
        Normalization layer type used in FCOS modules. 

    prior_prob: float
        Initial probability for focal loss. See `https://arxiv.org/pdf/1708.02002.pdf` for more details.
    """

    def __init__(self, num_classes, things_num_classes, num_fpn_levels, fpn_strides, in_channels, norm_reg_targets=False, centerness_on_reg=True, fcos_num_convs=4, fcos_norm='GN', prior_prob=0.01):
        super(PanopticHead, self).__init__()
        self.fpn_strides = fpn_strides
        self.norm_reg_targets = norm_reg_targets
        self.centerness_on_reg = centerness_on_reg
        cls_tower = []
        bbox_tower = []
        mid_channels = in_channels // 2
        for i in range(fcos_num_convs):
            if i == 0:
                cls_tower.append(nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1))
            else:
                cls_tower.append(nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1))
            if fcos_norm == 'GN':
                cls_tower.append(nn.GroupNorm(mid_channels // 8, mid_channels))
            elif fcos_norm == 'BN':
                cls_tower.append(nn.BatchNorm2d(mid_channels))
            elif fcos_norm == 'SBN':
                cls_tower.append(apex.parallel.SyncBatchNorm(mid_channels))
            cls_tower.append(nn.ReLU())
            bbox_tower.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            if fcos_norm == 'GN':
                bbox_tower.append(nn.GroupNorm(in_channels // 8, in_channels))
            elif fcos_norm == 'BN':
                bbox_tower.append(nn.BatchNorm2d(in_channels))
            elif fcos_norm == 'SBN':
                bbox_tower.append(apex.parallel.SyncBatchNorm(in_channels))
            bbox_tower.append(nn.ReLU())
        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        self.cls_logits = nn.Conv2d(mid_channels * 5, num_classes, kernel_size=3, stride=1, padding=1)
        self.box_cls_logits = nn.Conv2d(mid_channels, things_num_classes, kernel_size=3, stride=1, padding=1)
        self.bbox_pred = nn.Conv2d(in_channels, 4, kernel_size=3, stride=1, padding=1)
        self.centerness = nn.Conv2d(in_channels, 1, kernel_size=3, stride=1, padding=1)
        self.levelness = nn.Conv2d(in_channels * 5, num_fpn_levels + 1, kernel_size=3, stride=1, padding=1)
        to_initialize = [self.bbox_tower, self.cls_logits, self.cls_tower, self.bbox_pred, self.centerness, self.levelness, self.box_cls_logits]
        for modules in to_initialize:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)
        prior_prob = prior_prob
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)
        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])

    def forward(self, x):
        box_cls = []
        logits = []
        bbox_reg = []
        centerness = []
        levelness = []
        downsampled_shape = x[0].shape[2:]
        box_feature_map_downsampled_shape = torch.Size((downsampled_shape[0] * 2, downsampled_shape[1] * 2))
        for l, feature in enumerate(x):
            box_tower = self.bbox_tower(feature)
            cls_tower = self.cls_tower(feature)
            box_cls.append(self.box_cls_logits(cls_tower))
            logits.append(F.interpolate(cls_tower, size=downsampled_shape, mode='bilinear'))
            if self.centerness_on_reg:
                centerness.append(self.centerness(box_tower))
            else:
                centerness.append(self.centerness(cls_tower))
            bbox_pred = self.scales[l](self.bbox_pred(box_tower))
            if self.norm_reg_targets:
                bbox_pred = F.relu(bbox_pred)
                if self.training:
                    bbox_reg.append(bbox_pred)
                else:
                    bbox_reg.append(bbox_pred * self.fpn_strides[l])
            else:
                bbox_reg.append(torch.exp(bbox_pred.clamp(max=math.log(10000))))
            levelness.append(F.interpolate(box_tower, size=box_feature_map_downsampled_shape, mode='bilinear'))
        levelness = torch.cat(levelness, dim=1)
        levelness_logits = self.levelness(levelness)
        logits = torch.cat(logits, 1)
        semantic_logits = self.cls_logits(logits)
        return semantic_logits, box_cls, bbox_reg, centerness, levelness_logits


class RTPanoNet(torch.nn.Module):
    """Real-Time Panoptic Network
    This module takes the input from a FPN backbone and conducts feature extraction
    through a panoptic head, which can be then fed into post processing for final panoptic
    results including semantic segmentation and instance segmentation.
    NOTE: Currently only the inference functionality is supported.

    Parameters
    ----------
    backbone: str
        backbone type.

    num_classes: int
        Number of total classes, including 'things' and 'stuff'.

    things_num_classes: int
        number of thing classes

    pre_nms_thresh: float
        Acceptance class probability threshold for bounding box candidates before NMS.

    pre_nms_top_n: int
        Maximum number of accepted bounding box candidates before NMS.

    nms_thresh: float
        NMS threshold.

    fpn_post_nms_top_n: int
        Maximum number of detected object per image.

    instance_id_range: list of int
        [min_id, max_id] defines the range of id in 1:num_classes that corresponding to thing classes.
    """

    def __init__(self, backbone, num_classes, things_num_classes, pre_nms_thresh, pre_nms_top_n, nms_thresh, fpn_post_nms_top_n, instance_id_range):
        super(RTPanoNet, self).__init__()
        if backbone == 'R-50-FPN-RETINANET':
            self.backbone = ResNetWithModifiedFPN('resnet50')
            backbone_out_channels = 256
            fpn_strides = [8, 16, 32, 64, 128]
            num_fpn_levels = 5
        else:
            raise NotImplementedError('Backbone type: {} is not supported yet.'.format(backbone))
        self.panoptic_head = PanopticHead(num_classes, things_num_classes, num_fpn_levels, fpn_strides, backbone_out_channels)
        self.fpn_strides = fpn_strides
        self.panoptic_from_dense_bounding_box = PanopticFromDenseBox(pre_nms_thresh=pre_nms_thresh, pre_nms_top_n=pre_nms_top_n, nms_thresh=nms_thresh, fpn_post_nms_top_n=fpn_post_nms_top_n, min_size=0, num_classes=num_classes, mask_thresh=0.4, instance_id_range=instance_id_range, is_training=False)

    def forward(self, images, detection_targets=None, segmentation_targets=None):
        """ Forward function.

        Parameters
        ----------
        images: torchvision.models.detection.ImageList
            Images for which we want to compute the predictions

        detection_targets: list of BoxList
            Ground-truth boxes present in the image

        segmentation_targets: List of torch.Tensor
            semantic segmentation target for each image in the batch.

        Returns
        -------
        panoptic_result: Dict
            'instance_segmentation_result': list of BoxList
                The predicted boxes (including instance masks), one BoxList per image.
            'semantic_segmentation_result': torch.Tensor
                semantic logits interpolated to input data size. 
                NOTE: this might not be the original input image size due to paddings. 
        losses: dict of torch.ScalarTensor
            the losses for the model during training. During testing, it is an empty dict.
        """
        features = self.backbone(torch.stack(images.tensors))
        locations = self.compute_locations(list(features.values()))
        semantic_logits, box_cls, box_regression, centerness, levelness_logits = self.panoptic_head(list(features.values()))
        downsampled_level = images.tensors[0].shape[-1] // semantic_logits.shape[-1]
        interpolated_semantic_logits_padded = F.interpolate(semantic_logits, scale_factor=downsampled_level, mode='bilinear')
        interpolated_semantic_logits = interpolated_semantic_logits_padded[:, :, :images.tensors[0].shape[-2], :images.tensors[0].shape[-1]]
        h, w = levelness_logits.size()[-2:]
        levelness_location = self.compute_locations_per_level(h, w, self.fpn_strides[0] // 2, levelness_logits.device)
        locations.append(levelness_location)
        panoptic_result = OrderedDict()
        boxes = self.panoptic_from_dense_bounding_box.process(locations, box_cls, box_regression, centerness, levelness_logits, semantic_logits, images.image_sizes)
        panoptic_result['instance_segmentation_result'] = boxes
        panoptic_result['semantic_segmentation_result'] = interpolated_semantic_logits
        return panoptic_result, {}

    def compute_locations(self, features):
        """Compute corresponding pixel location for feature maps.

        Parameters
        ----------
        features: list of torch.Tensor
            List of feature maps.

        Returns
        -------
        locations: list of torch.Tensor
            List of pixel location corresponding to the list of features.
        """
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = self.compute_locations_per_level(h, w, self.fpn_strides[level], feature.device)
            locations.append(locations_per_level)
        return locations

    def compute_locations_per_level(self, h, w, stride, device):
        """Compute corresponding pixel location for a feature map in pyramid space with certain stride.

        Parameters
        ----------
        h: int
            height of current feature map.

        w: int
            width of current feature map.

        stride: int
            stride level of current feature map with respect to original input.

        device: torch.device
            device to create return tensor.

        Returns
        -------
        locations: torch.Tensor
            pixel location map.
        """
        shifts_x = torch.arange(0, w * stride, step=stride, dtype=torch.float32, device=device)
        shifts_y = torch.arange(0, h * stride, step=stride, dtype=torch.float32, device=device)
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        return locations


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Scale,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_TRI_ML_realtime_panoptic(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

