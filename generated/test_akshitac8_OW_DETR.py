import sys
_module = sys.modules[__name__]
del sys
benchmark = _module
datasets = _module
coco = _module
coco2voc = _module
coco_eval = _module
coco_panoptic = _module
create_imagenets_t1 = _module
create_imagenets_t2 = _module
create_imagenets_t3 = _module
create_imagenets_t4 = _module
data_prefetcher = _module
open_world_eval = _module
panoptic_eval = _module
samplers = _module
torchvision_datasets = _module
open_world = _module
transforms = _module
engine = _module
main_open_world = _module
models = _module
backbone = _module
deformable_detr = _module
deformable_transformer = _module
matcher = _module
functions = _module
ms_deform_attn_func = _module
modules = _module
ms_deform_attn = _module
setup = _module
test = _module
position_encoding = _module
segmentation = _module
launch = _module
util = _module
box_ops = _module
misc = _module
plot_utils = _module

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


import time


import torch


import torch.utils.data


import copy


import numpy as np


import functools


import logging


from collections import OrderedDict


from collections import defaultdict


import math


import torch.distributed as dist


from torch.utils.data.sampler import Sampler


import collections


from torchvision.datasets import VisionDataset


import itertools


from torchvision.datasets.utils import download_url


from torchvision.datasets.utils import check_integrity


from torchvision.datasets.utils import verify_str_arg


import random


import torchvision.transforms as T


import torchvision.transforms.functional as F


from typing import Iterable


import matplotlib.pyplot as plt


from copy import deepcopy


from torch.utils.data import DataLoader


from torchvision.models.resnet import resnet50


import torch.nn.functional as F


import torchvision


from torch import nn


from torchvision.models._utils import IntermediateLayerGetter


from typing import Dict


from typing import List


from typing import Optional


from torch import Tensor


from torch.nn.init import xavier_uniform_


from torch.nn.init import constant_


from torch.nn.init import uniform_


from torch.nn.init import normal_


from scipy.optimize import linear_sum_assignment


from torch.autograd import Function


from torch.autograd.function import once_differentiable


import warnings


from torch.utils.cpp_extension import CUDA_HOME


from torch.utils.cpp_extension import CppExtension


from torch.utils.cpp_extension import CUDAExtension


import torch.nn as nn


from torch.autograd import gradcheck


from torchvision.ops.boxes import box_area


from collections import deque


import pandas as pd


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n, eps=1e-05):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer('weight', torch.ones(n))
        self.register_buffer('bias', torch.zeros(n))
        self.register_buffer('running_mean', torch.zeros(n))
        self.register_buffer('running_var', torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]
        super(FrozenBatchNorm2d, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class NestedTensor(object):

    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device, non_blocking=False):
        cast_tensor = self.tensors
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def record_stream(self, *args, **kwargs):
        self.tensors.record_stream(*args, **kwargs)
        if self.mask is not None:
            self.mask.record_stream(*args, **kwargs)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {'layer2': '0', 'layer3': '1', 'layer4': '2'}
            self.strides = [8, 16, 32]
            self.num_channels = [512, 1024, 2048]
        else:
            return_layers = {'layer4': '0'}
            self.strides = [32]
            self.num_channels = [2048]
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:])[0]
            out[name] = NestedTensor(x, mask)
        return out


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(self, name: str, train_backbone: bool, return_interm_layers: bool, dilation: bool):
        norm_layer = FrozenBatchNorm2d
        if name == 'resnet50':
            None
            backbone = getattr(torchvision.models, name)(replace_stride_with_dilation=[False, False, dilation], pretrained=is_main_process(), norm_layer=norm_layer)
        else:
            None
            backbone = resnet50(pretrained=False, replace_stride_with_dilation=[False, False, dilation], norm_layer=norm_layer)
            if is_main_process():
                state_dict = torch.load('/proj/cvl/users/x_fahkh/akshita/Deformable-DETR/models/dino_resnet50_pretrain/dino_resnet50_pretrain.pth')
                backbone.load_state_dict(state_dict, strict=False)
        assert name not in ('resnet18', 'resnet34'), 'number of channels are hard coded'
        super().__init__(backbone, train_backbone, return_interm_layers)
        if dilation:
            self.strides[-1] = self.strides[-1] // 2


class Joiner(nn.Sequential):

    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in sorted(xs.items()):
            out.append(x)
        for x in out:
            pos.append(self[1](x))
        return out, pos


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def inverse_sigmoid(x, eps=1e-05):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def _max_by_axis(the_list):
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    if tensor_list[0].ndim == 3:
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[:img.shape[0], :img.shape[1], :img.shape[2]].copy_(img)
            m[:img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)


class DeformableDETR(nn.Module):
    """ This is the Deformable DETR module that performs object detection """

    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels, aux_loss=True, with_box_refine=False, two_stage=False, unmatched_boxes=False, novelty_cls=False, featdim=1024):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.num_feature_levels = num_feature_levels
        self.featdim = featdim
        self.unmatched_boxes = unmatched_boxes
        self.novelty_cls = novelty_cls
        if self.novelty_cls:
            self.nc_class_embed = nn.Linear(hidden_dim, 1)
        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(nn.Conv2d(in_channels, hidden_dim, kernel_size=1), nn.GroupNorm(32, hidden_dim)))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1), nn.GroupNorm(32, hidden_dim)))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([nn.Sequential(nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1), nn.GroupNorm(32, hidden_dim))])
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        if self.novelty_cls:
            self.nc_class_embed.bias.data = torch.ones(1) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
        num_pred = transformer.decoder.num_layers + 1 if two_stage else transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            if self.novelty_cls:
                self.nc_class_embed = nn.ModuleList([self.nc_class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
        if two_stage:
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        srcs = []
        masks = []
        if self.featdim == 512:
            dim_index = 0
        elif self.featdim == 1024:
            dim_index = 1
        else:
            dim_index = 2
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            if self.unmatched_boxes:
                if l == dim_index:
                    resnet_1024_feature = src.clone()
            else:
                resnet_1024_feature = None
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:])[0]
                pos_l = self.backbone[1](NestedTensor(src, mask))
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)
        query_embeds = None
        if not self.two_stage:
            query_embeds = self.query_embed.weight
        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = self.transformer(srcs, masks, pos, query_embeds)
        outputs_classes = []
        outputs_coords = []
        outputs_classes_nc = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            if self.novelty_cls:
                outputs_class_nc = self.nc_class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            if self.novelty_cls:
                outputs_classes_nc.append(outputs_class_nc)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        if self.novelty_cls:
            output_class_nc = torch.stack(outputs_classes_nc)
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1], 'resnet_1024_feat': resnet_1024_feature}
        if self.novelty_cls:
            out = {'pred_logits': outputs_class[-1], 'pred_nc_logits': output_class_nc[-1], 'pred_boxes': outputs_coord[-1], 'resnet_1024_feat': resnet_1024_feature}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, output_class_nc=None)
            if self.novelty_cls:
                out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, output_class_nc=output_class_nc)
        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, output_class_nc=None):
        if output_class_nc is not None:
            xx = [{'pred_logits': a, 'pred_nc_logits': c, 'pred_boxes': b} for a, c, b in zip(outputs_class[:-1], output_class_nc[:-1], outputs_coord[:-1])]
        else:
            xx = [{'pred_logits': a, 'pred_boxes': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
        return xx


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def dice_loss(inputs, targets, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None):
    """
    Equivalent to nn.functional.interpolate, but with support for empty batch sizes.
    This will eventually be supported natively by PyTorch, and this
    class can go away.
    """
    if float(torchvision.__version__[:3]) < 0.7:
        if input.numel() > 0:
            return torch.nn.functional.interpolate(input, size, scale_factor, mode, align_corners)
        output_shape = _output_size(2, input, size, scale_factor)
        output_shape = list(input.shape[:-2]) + list(output_shape)
        if float(torchvision.__version__[:3]) < 0.5:
            return _NewEmptyTensorOp.apply(input, output_shape)
        return _new_empty_tensor(input, output_shape)
    else:
        return torchvision.ops.misc.interpolate(input, size, scale_factor, mode, align_corners)


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float=0.25, gamma: float=2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * (1 - p_t) ** gamma
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean(1).sum() / num_boxes


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, args, num_classes, matcher, weight_dict, losses, invalid_cls_logits, focal_alpha=0.25):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.nc_epoch = args.nc_epoch
        self.output_dir = args.output_dir
        self.invalid_cls_logits = invalid_cls_logits
        self.unmatched_boxes = args.unmatched_boxes
        self.top_unk = args.top_unk
        self.bbox_thresh = args.bbox_thresh
        self.num_seen_classes = args.PREV_INTRODUCED_CLS + args.CUR_INTRODUCED_CLS

    def loss_NC_labels(self, outputs, targets, indices, num_boxes, current_epoch, owod_targets, owod_indices, log=True):
        """Novelty classification loss
        target labels will contain class as 1
        owod_indices -> indices combining matched indices + psuedo labeled indices
        owod_targets -> targets combining GT targets + psuedo labeled unknown targets
        target_classes_o -> contains all 1's
        """
        assert 'pred_nc_logits' in outputs
        src_logits = outputs['pred_nc_logits']
        idx = self._get_src_permutation_idx(owod_indices)
        target_classes_o = torch.cat([torch.full_like(t['labels'][J], 0) for t, (_, J) in zip(owod_targets, owod_indices)])
        target_classes = torch.full(src_logits.shape[:2], 1, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1], dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_NC': loss_ce}
        return losses

    def loss_labels(self, outputs, targets, indices, num_boxes, current_epoch, owod_targets, owod_indices, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        temp_src_logits = outputs['pred_logits'].clone()
        temp_src_logits[:, :, self.invalid_cls_logits] = -100000000000.0
        src_logits = temp_src_logits
        if self.unmatched_boxes:
            idx = self._get_src_permutation_idx(owod_indices)
            target_classes_o = torch.cat([t['labels'][J] for t, (_, J) in zip(owod_targets, owod_indices)])
        else:
            idx = self._get_src_permutation_idx(indices)
            target_classes_o = torch.cat([t['labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1], dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}
        if log:
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes, current_epoch, owod_targets, owod_indices):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        temp_pred_logits = outputs['pred_logits'].clone()
        temp_pred_logits[:, :, self.invalid_cls_logits] = -100000000000.0
        pred_logits = temp_pred_logits
        device = pred_logits.device
        if self.unmatched_boxes:
            tgt_lengths = torch.as_tensor([len(v['labels']) for v in owod_targets], device=device)
        else:
            tgt_lengths = torch.as_tensor([len(v['labels']) for v in targets], device=device)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes, current_epoch, owod_targets, owod_indices):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(box_ops.box_cxcywh_to_xyxy(src_boxes), box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes, current_epoch, owod_targets, owod_indices):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert 'pred_masks' in outputs
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs['pred_masks']
        target_masks, valid = nested_tensor_from_tensor_list([t['masks'] for t in targets]).decompose()
        target_masks = target_masks
        src_masks = src_masks[src_idx]
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:], mode='bilinear', align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)
        target_masks = target_masks[tgt_idx].flatten(1)
        losses = {'loss_mask': sigmoid_focal_loss(src_masks, target_masks, num_boxes), 'loss_dice': dice_loss(src_masks, target_masks, num_boxes)}
        return losses

    def save_dict(self, di_, filename_):
        with open(filename_, 'wb') as f:
            pickle.dump(di_, f)

    def load_dict(self, filename_):
        with open(filename_, 'rb') as f:
            ret_dict = pickle.load(f)
        return ret_dict

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for src, _ in indices])
        return batch_idx, src_idx

    def _get_src_single_permutation_idx(self, indices, index):
        batch_idx = [torch.full_like(src, i) for i, src in enumerate(indices)][0]
        src_idx = indices[0]
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for _, tgt in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, epoch, owod_targets, owod_indices, **kwargs):
        loss_map = {'labels': self.loss_labels, 'NC_labels': self.loss_NC_labels, 'cardinality': self.loss_cardinality, 'boxes': self.loss_boxes, 'masks': self.loss_masks}
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, epoch, owod_targets, owod_indices, **kwargs)

    def forward(self, samples, outputs, targets, epoch):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        if self.nc_epoch > 0:
            loss_epoch = 9
        else:
            loss_epoch = 0
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}
        indices = self.matcher(outputs_without_aux, targets)
        owod_targets = deepcopy(targets)
        owod_indices = deepcopy(indices)
        owod_outputs = outputs_without_aux.copy()
        owod_device = owod_outputs['pred_boxes'].device
        if self.unmatched_boxes and epoch >= loss_epoch:
            res_feat = torch.mean(outputs['resnet_1024_feat'], 1)
            queries = torch.arange(outputs['pred_logits'].shape[1])
            for i in range(len(indices)):
                combined = torch.cat((queries, self._get_src_single_permutation_idx(indices[i], i)[-1]))
                uniques, counts = combined.unique(return_counts=True)
                unmatched_indices = uniques[counts == 1]
                boxes = outputs_without_aux['pred_boxes'][i]
                img = samples.tensors[i].cpu().permute(1, 2, 0).numpy()
                h, w = img.shape[:-1]
                img_w = torch.tensor(w, device=owod_device)
                img_h = torch.tensor(h, device=owod_device)
                unmatched_boxes = box_ops.box_cxcywh_to_xyxy(boxes)
                unmatched_boxes = unmatched_boxes * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
                means_bb = torch.zeros(queries.shape[0])
                bb = unmatched_boxes
                for j, _ in enumerate(means_bb):
                    if j in unmatched_indices:
                        upsaple = nn.Upsample(size=(img_h, img_w), mode='bilinear')
                        img_feat = upsaple(res_feat[i].unsqueeze(0).unsqueeze(0))
                        img_feat = img_feat.squeeze(0).squeeze(0)
                        xmin = bb[j, :][0].long()
                        ymin = bb[j, :][1].long()
                        xmax = bb[j, :][2].long()
                        ymax = bb[j, :][3].long()
                        means_bb[j] = torch.mean(img_feat[ymin:ymax, xmin:xmax])
                        if torch.isnan(means_bb[j]):
                            means_bb[j] = -100000000000.0
                    else:
                        means_bb[j] = -100000000000.0
                _, topk_inds = torch.topk(means_bb, self.top_unk)
                topk_inds = torch.as_tensor(topk_inds)
                topk_inds = topk_inds.cpu()
                unk_label = torch.as_tensor([self.num_classes - 1], device=owod_device)
                owod_targets[i]['labels'] = torch.cat((owod_targets[i]['labels'], unk_label.repeat_interleave(self.top_unk)))
                owod_indices[i] = torch.cat((owod_indices[i][0], topk_inds)), torch.cat((owod_indices[i][1], (owod_targets[i]['labels'] == unk_label).nonzero(as_tuple=True)[0].cpu()))
        num_boxes = sum(len(t['labels']) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, epoch, owod_targets, owod_indices, **kwargs))
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                owod_targets = deepcopy(targets)
                owod_indices = deepcopy(indices)
                aux_owod_outputs = aux_outputs.copy()
                owod_device = aux_owod_outputs['pred_boxes'].device
                if self.unmatched_boxes and epoch >= loss_epoch:
                    res_feat = torch.mean(outputs['resnet_1024_feat'], 1)
                    queries = torch.arange(aux_owod_outputs['pred_logits'].shape[1])
                    for i in range(len(indices)):
                        combined = torch.cat((queries, self._get_src_single_permutation_idx(indices[i], i)[-1]))
                        uniques, counts = combined.unique(return_counts=True)
                        unmatched_indices = uniques[counts == 1]
                        boxes = aux_owod_outputs['pred_boxes'][i]
                        img = samples.tensors[i].cpu().permute(1, 2, 0).numpy()
                        h, w = img.shape[:-1]
                        img_w = torch.tensor(w, device=owod_device)
                        img_h = torch.tensor(h, device=owod_device)
                        unmatched_boxes = box_ops.box_cxcywh_to_xyxy(boxes)
                        unmatched_boxes = unmatched_boxes * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
                        means_bb = torch.zeros(queries.shape[0])
                        bb = unmatched_boxes
                        for j, _ in enumerate(means_bb):
                            if j in unmatched_indices:
                                upsaple = nn.Upsample(size=(img_h, img_w), mode='bilinear')
                                img_feat = upsaple(res_feat[i].unsqueeze(0).unsqueeze(0))
                                img_feat = img_feat.squeeze(0).squeeze(0)
                                xmin = bb[j, :][0].long()
                                ymin = bb[j, :][1].long()
                                xmax = bb[j, :][2].long()
                                ymax = bb[j, :][3].long()
                                means_bb[j] = torch.mean(img_feat[ymin:ymax, xmin:xmax])
                                if torch.isnan(means_bb[j]):
                                    means_bb[j] = -100000000000.0
                            else:
                                means_bb[j] = -100000000000.0
                        _, topk_inds = torch.topk(means_bb, self.top_unk)
                        topk_inds = torch.as_tensor(topk_inds)
                        topk_inds = topk_inds.cpu()
                        unk_label = torch.as_tensor([self.num_classes - 1], device=owod_device)
                        owod_targets[i]['labels'] = torch.cat((owod_targets[i]['labels'], unk_label.repeat_interleave(self.top_unk)))
                        owod_indices[i] = torch.cat((owod_indices[i][0], topk_inds)), torch.cat((owod_indices[i][1], (owod_targets[i]['labels'] == unk_label).nonzero(as_tuple=True)[0].cpu()))
                for loss in self.losses:
                    if loss == 'masks':
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, epoch, owod_targets, owod_indices, **kwargs)
                    l_dict = {(k + f'_{i}'): v for k, v in l_dict.items()}
                    losses.update(l_dict)
        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt['labels'] = torch.zeros_like(bt['labels'])
            indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                if loss == 'masks':
                    continue
                kwargs = {}
                if loss == 'labels':
                    kwargs['log'] = False
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
                l_dict = {(k + f'_enc'): v for k, v in l_dict.items()}
                losses.update(l_dict)
        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2
        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]
        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
        return results


class DeformableTransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.bbox_embed = None
        self.class_embed = None

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios, query_pos=None, src_padding_mask=None):
        output = tgt
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
            output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index, src_padding_mask)
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)
        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)
        return output, reference_points


class MSDeformAttnFunction(Function):

    @staticmethod
    def forward(ctx, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step):
        ctx.im2col_step = im2col_step
        output = MSDA.ms_deform_attn_forward(value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, ctx.im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = MSDA.ms_deform_attn_backward(value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, grad_output, ctx.im2col_step)
        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


def _is_power_of_2(n):
    if not isinstance(n, int) or n < 0:
        raise ValueError('invalid input for _is_power_of_2: {} (type: {})'.format(n, type(n)))
    return n & n - 1 == 0 and n != 0


class MSDeformAttn(nn.Module):

    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 which is more efficient in our CUDA implementation.")
        self.im2col_step = 64
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.0)
        constant_(self.attention_weights.bias.data, 0.0)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \\sum_{l=0}^{L-1} H_l \\cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \\sum_{l=0}^{L-1} H_l \\cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in
        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError('Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        output = MSDeformAttnFunction.apply(value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)
        output = self.output_proj(output)
        return output


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == 'relu':
        return F.relu
    if activation == 'gelu':
        return F.gelu
    if activation == 'glu':
        return F.glu
    raise RuntimeError(f'activation should be relu/gelu, not {activation}.')


class DeformableTransformerDecoderLayer(nn.Module):

    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, activation='relu', n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask=None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos), reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt = self.forward_ffn(tgt)
        return tgt


class DeformableTransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device), torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)
        return output


class DeformableTransformerEncoderLayer(nn.Module):

    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, activation='relu', n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src = self.forward_ffn(src)
        return src


class DeformableTransformer(nn.Module):

    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1, activation='relu', return_intermediate_dec=False, num_feature_levels=4, dec_n_points=4, enc_n_points=4, two_stage=False, two_stage_num_proposals=300):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals
        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward, dropout, activation, num_feature_levels, nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)
        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward, dropout, activation, num_feature_levels, nhead, dec_n_points)
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        if two_stage:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        else:
            self.reference_points = nn.Linear(d_model, 2)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if not self.two_stage:
            xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            constant_(self.reference_points.bias.data, 0.0)
        normal_(self.level_embed)

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        proposals = proposals.sigmoid() * scale
        pos = proposals[:, :, :, None] / dim_t
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:_cur + H_ * W_].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)
            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device), torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)
            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * 2.0 ** lvl
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += H_ * W_
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))
        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def to_4d(self, x, h, w):
        return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

    def forward(self, srcs, masks, pos_embeds, query_embed=None):
        assert self.two_stage or query_embed is not None
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = h, w
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)
        bs, _, c = memory.shape
        if self.two_stage:
            output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)
            enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
            enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals
            topk = self.two_stage_num_proposals
            topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
            topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            init_reference_out = reference_points
            pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
            query_embed, tgt = torch.split(pos_trans_out, c, dim=2)
        else:
            query_embed, tgt = torch.split(query_embed, c, dim=1)
            query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
            reference_points = self.reference_points(query_embed).sigmoid()
            init_reference_out = reference_points
        hs, inter_references = self.decoder(tgt, reference_points, memory, spatial_shapes, level_start_index, valid_ratios, query_embed, mask_flatten)
        inter_references_out = inter_references
        if self.two_stage:
            return hs, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord_unact
        return hs, init_reference_out, inter_references_out, None, None


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h]
    return torch.stack(b, dim=-1)


def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    area = wh[:, :, 0] * wh[:, :, 1]
    return iou - (area - union) / area


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float=1, cost_bbox: float=1, cost_giou: float=1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, 'all costs cant be 0'

    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        with torch.no_grad():
            bs, num_queries = outputs['pred_logits'].shape[:2]
            out_prob = outputs['pred_logits'].flatten(0, 1).sigmoid()
            out_bbox = outputs['pred_boxes'].flatten(0, 1)
            tgt_ids = torch.cat([v['labels'] for v in targets])
            tgt_bbox = torch.cat([v['boxes'] for v in targets])
            alpha = 0.25
            gamma = 2.0
            neg_cost_class = (1 - alpha) * out_prob ** gamma * -(1 - out_prob + 1e-08).log()
            pos_cost_class = alpha * (1 - out_prob) ** gamma * -(out_prob + 1e-08).log()
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            C = C.view(bs, num_queries, -1).cpu()
            sizes = [len(v['boxes']) for v in targets]
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError('normalize should be True if scale is passed')
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-06
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([x_emb.unsqueeze(0).repeat(h, 1, 1), y_emb.unsqueeze(1).repeat(1, w, 1)], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos


class MHAttentionMap(nn.Module):
    """This is a 2D attention module, which only returns the attention softmax (no multiplication by value)"""

    def __init__(self, query_dim, hidden_dim, num_heads, dropout=0, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.q_linear = nn.Linear(query_dim, hidden_dim, bias=bias)
        self.k_linear = nn.Linear(query_dim, hidden_dim, bias=bias)
        nn.init.zeros_(self.k_linear.bias)
        nn.init.zeros_(self.q_linear.bias)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.xavier_uniform_(self.q_linear.weight)
        self.normalize_fact = float(hidden_dim / self.num_heads) ** -0.5

    def forward(self, q, k, mask=None):
        q = self.q_linear(q)
        k = F.conv2d(k, self.k_linear.weight.unsqueeze(-1).unsqueeze(-1), self.k_linear.bias)
        qh = q.view(q.shape[0], q.shape[1], self.num_heads, self.hidden_dim // self.num_heads)
        kh = k.view(k.shape[0], self.num_heads, self.hidden_dim // self.num_heads, k.shape[-2], k.shape[-1])
        weights = torch.einsum('bqnc,bnchw->bqnhw', qh * self.normalize_fact, kh)
        if mask is not None:
            weights.masked_fill_(mask.unsqueeze(1).unsqueeze(1), float('-inf'))
        weights = F.softmax(weights.flatten(2), dim=-1).view_as(weights)
        weights = self.dropout(weights)
        return weights


class MaskHeadSmallConv(nn.Module):
    """
    Simple convolutional head, using group norm.
    Upsampling is done using a FPN approach
    """

    def __init__(self, dim, fpn_dims, context_dim):
        super().__init__()
        inter_dims = [dim, context_dim // 2, context_dim // 4, context_dim // 8, context_dim // 16, context_dim // 64]
        self.lay1 = torch.nn.Conv2d(dim, dim, 3, padding=1)
        self.gn1 = torch.nn.GroupNorm(8, dim)
        self.lay2 = torch.nn.Conv2d(dim, inter_dims[1], 3, padding=1)
        self.gn2 = torch.nn.GroupNorm(8, inter_dims[1])
        self.lay3 = torch.nn.Conv2d(inter_dims[1], inter_dims[2], 3, padding=1)
        self.gn3 = torch.nn.GroupNorm(8, inter_dims[2])
        self.lay4 = torch.nn.Conv2d(inter_dims[2], inter_dims[3], 3, padding=1)
        self.gn4 = torch.nn.GroupNorm(8, inter_dims[3])
        self.lay5 = torch.nn.Conv2d(inter_dims[3], inter_dims[4], 3, padding=1)
        self.gn5 = torch.nn.GroupNorm(8, inter_dims[4])
        self.out_lay = torch.nn.Conv2d(inter_dims[4], 1, 3, padding=1)
        self.dim = dim
        self.adapter1 = torch.nn.Conv2d(fpn_dims[0], inter_dims[1], 1)
        self.adapter2 = torch.nn.Conv2d(fpn_dims[1], inter_dims[2], 1)
        self.adapter3 = torch.nn.Conv2d(fpn_dims[2], inter_dims[3], 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, bbox_mask, fpns):

        def expand(tensor, length):
            return tensor.unsqueeze(1).repeat(1, int(length), 1, 1, 1).flatten(0, 1)
        x = torch.cat([expand(x, bbox_mask.shape[1]), bbox_mask.flatten(0, 1)], 1)
        x = self.lay1(x)
        x = self.gn1(x)
        x = F.relu(x)
        x = self.lay2(x)
        x = self.gn2(x)
        x = F.relu(x)
        cur_fpn = self.adapter1(fpns[0])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = expand(cur_fpn, x.size(0) / cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode='nearest')
        x = self.lay3(x)
        x = self.gn3(x)
        x = F.relu(x)
        cur_fpn = self.adapter2(fpns[1])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = expand(cur_fpn, x.size(0) / cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode='nearest')
        x = self.lay4(x)
        x = self.gn4(x)
        x = F.relu(x)
        cur_fpn = self.adapter3(fpns[2])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = expand(cur_fpn, x.size(0) / cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode='nearest')
        x = self.lay5(x)
        x = self.gn5(x)
        x = F.relu(x)
        x = self.out_lay(x)
        return x


class DETRsegm(nn.Module):

    def __init__(self, detr, freeze_detr=False):
        super().__init__()
        self.detr = detr
        if freeze_detr:
            for p in self.parameters():
                p.requires_grad_(False)
        hidden_dim, nheads = detr.transformer.d_model, detr.transformer.nhead
        self.bbox_attention = MHAttentionMap(hidden_dim, hidden_dim, nheads, dropout=0)
        self.mask_head = MaskHeadSmallConv(hidden_dim + nheads, [1024, 512, 256], hidden_dim)

    def forward(self, samples: NestedTensor):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.detr.backbone(samples)
        bs = features[-1].tensors.shape[0]
        src, mask = features[-1].decompose()
        src_proj = self.detr.input_proj(src)
        hs, memory = self.detr.transformer(src_proj, mask, self.detr.query_embed.weight, pos[-1])
        outputs_class = self.detr.class_embed(hs)
        outputs_coord = self.detr.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.detr.aux_loss:
            out['aux_outputs'] = [{'pred_logits': a, 'pred_boxes': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
        bbox_mask = self.bbox_attention(hs[-1], memory, mask=mask)
        seg_masks = self.mask_head(src_proj, bbox_mask, [features[2].tensors, features[1].tensors, features[0].tensors])
        outputs_seg_masks = seg_masks.view(bs, self.detr.num_queries, seg_masks.shape[-2], seg_masks.shape[-1])
        out['pred_masks'] = outputs_seg_masks
        return out


class PostProcessSegm(nn.Module):

    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    @torch.no_grad()
    def forward(self, results, outputs, orig_target_sizes, max_target_sizes):
        assert len(orig_target_sizes) == len(max_target_sizes)
        max_h, max_w = max_target_sizes.max(0)[0].tolist()
        outputs_masks = outputs['pred_masks'].squeeze(2)
        outputs_masks = F.interpolate(outputs_masks, size=(max_h, max_w), mode='bilinear', align_corners=False)
        outputs_masks = (outputs_masks.sigmoid() > self.threshold).cpu()
        for i, (cur_mask, t, tt) in enumerate(zip(outputs_masks, max_target_sizes, orig_target_sizes)):
            img_h, img_w = t[0], t[1]
            results[i]['masks'] = cur_mask[:, :img_h, :img_w].unsqueeze(1)
            results[i]['masks'] = F.interpolate(results[i]['masks'].float(), size=tuple(tt.tolist()), mode='nearest').byte()
        return results


class PostProcessPanoptic(nn.Module):
    """This class converts the output of the model to the final panoptic result, in the format expected by the
    coco panoptic API """

    def __init__(self, is_thing_map, threshold=0.85):
        """
        Parameters:
           is_thing_map: This is a whose keys are the class ids, and the values a boolean indicating whether
                          the class is  a thing (True) or a stuff (False) class
           threshold: confidence threshold: segments with confidence lower than this will be deleted
        """
        super().__init__()
        self.threshold = threshold
        self.is_thing_map = is_thing_map

    def forward(self, outputs, processed_sizes, target_sizes=None):
        """ This function computes the panoptic prediction from the model's predictions.
        Parameters:
            outputs: This is a dict coming directly from the model. See the model doc for the content.
            processed_sizes: This is a list of tuples (or torch tensors) of sizes of the images that were passed to the
                             model, ie the size after data augmentation but before batching.
            target_sizes: This is a list of tuples (or torch tensors) corresponding to the requested final size
                          of each prediction. If left to None, it will default to the processed_sizes
            """
        if target_sizes is None:
            target_sizes = processed_sizes
        assert len(processed_sizes) == len(target_sizes)
        out_logits, raw_masks, raw_boxes = outputs['pred_logits'], outputs['pred_masks'], outputs['pred_boxes']
        assert len(out_logits) == len(raw_masks) == len(target_sizes)
        preds = []

        def to_tuple(tup):
            if isinstance(tup, tuple):
                return tup
            return tuple(tup.cpu().tolist())
        for cur_logits, cur_masks, cur_boxes, size, target_size in zip(out_logits, raw_masks, raw_boxes, processed_sizes, target_sizes):
            scores, labels = cur_logits.softmax(-1).max(-1)
            keep = labels.ne(outputs['pred_logits'].shape[-1] - 1) & (scores > self.threshold)
            cur_scores, cur_classes = cur_logits.softmax(-1).max(-1)
            cur_scores = cur_scores[keep]
            cur_classes = cur_classes[keep]
            cur_masks = cur_masks[keep]
            cur_masks = interpolate(cur_masks[None], to_tuple(size), mode='bilinear').squeeze(0)
            cur_boxes = box_ops.box_cxcywh_to_xyxy(cur_boxes[keep])
            h, w = cur_masks.shape[-2:]
            assert len(cur_boxes) == len(cur_classes)
            cur_masks = cur_masks.flatten(1)
            stuff_equiv_classes = defaultdict(lambda : [])
            for k, label in enumerate(cur_classes):
                if not self.is_thing_map[label.item()]:
                    stuff_equiv_classes[label.item()].append(k)

            def get_ids_area(masks, scores, dedup=False):
                m_id = masks.transpose(0, 1).softmax(-1)
                if m_id.shape[-1] == 0:
                    m_id = torch.zeros((h, w), dtype=torch.long, device=m_id.device)
                else:
                    m_id = m_id.argmax(-1).view(h, w)
                if dedup:
                    for equiv in stuff_equiv_classes.values():
                        if len(equiv) > 1:
                            for eq_id in equiv:
                                m_id.masked_fill_(m_id.eq(eq_id), equiv[0])
                final_h, final_w = to_tuple(target_size)
                seg_img = Image.fromarray(id2rgb(m_id.view(h, w).cpu().numpy()))
                seg_img = seg_img.resize(size=(final_w, final_h), resample=Image.NEAREST)
                np_seg_img = torch.ByteTensor(torch.ByteStorage.from_buffer(seg_img.tobytes())).view(final_h, final_w, 3).numpy()
                m_id = torch.from_numpy(rgb2id(np_seg_img))
                area = []
                for i in range(len(scores)):
                    area.append(m_id.eq(i).sum().item())
                return area, seg_img
            area, seg_img = get_ids_area(cur_masks, cur_scores, dedup=True)
            if cur_classes.numel() > 0:
                while True:
                    filtered_small = torch.as_tensor([(area[i] <= 4) for i, c in enumerate(cur_classes)], dtype=torch.bool, device=keep.device)
                    if filtered_small.any().item():
                        cur_scores = cur_scores[~filtered_small]
                        cur_classes = cur_classes[~filtered_small]
                        cur_masks = cur_masks[~filtered_small]
                        area, seg_img = get_ids_area(cur_masks, cur_scores)
                    else:
                        break
            else:
                cur_classes = torch.ones(1, dtype=torch.long, device=cur_classes.device)
            segments_info = []
            for i, a in enumerate(area):
                cat = cur_classes[i].item()
                segments_info.append({'id': i, 'isthing': self.is_thing_map[cat], 'category_id': cat, 'area': a})
            del cur_classes
            with io.BytesIO() as out:
                seg_img.save(out, format='PNG')
                predictions = {'png_string': out.getvalue(), 'segments_info': segments_info}
            preds.append(predictions)
        return preds


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (FrozenBatchNorm2d,
     lambda: ([], {'n': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MLP,
     lambda: ([], {'input_dim': 4, 'hidden_dim': 4, 'output_dim': 4, 'num_layers': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_akshitac8_OW_DETR(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

