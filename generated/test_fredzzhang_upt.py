import sys
_module = sys.modules[__name__]
del sys
inference = _module
interaction_head = _module
main = _module
ops = _module
upt = _module
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


import torch


import warnings


import numpy as np


import matplotlib.pyplot as plt


import matplotlib.patches as patches


import matplotlib.patheffects as peff


import torch.nn.functional as F


from torch import nn


from torch import Tensor


from typing import List


from typing import Optional


from typing import Tuple


from collections import OrderedDict


import random


import torch.distributed as dist


import torch.multiprocessing as mp


from torch.utils.data import DataLoader


from torch.utils.data import DistributedSampler


import math


import torchvision.ops.boxes as box_ops


from typing import Dict


from scipy.optimize import linear_sum_assignment


from torchvision.ops.boxes import batched_nms


from torchvision.ops.boxes import box_iou


import scipy.io as sio


from collections import defaultdict


from torch.utils.data import Dataset


class MultiBranchFusion(nn.Module):
    """
    Multi-branch fusion module

    Parameters:
    -----------
    appearance_size: int
        Size of the appearance features
    spatial_size: int
        Size of the spatial features
    hidden_state_size: int
        Size of the intermediate representations
    cardinality: int
        The number of homogeneous branches
    """

    def __init__(self, appearance_size: int, spatial_size: int, hidden_state_size: int, cardinality: int) ->None:
        super().__init__()
        self.cardinality = cardinality
        sub_repr_size = int(hidden_state_size / cardinality)
        assert sub_repr_size * cardinality == hidden_state_size, 'The given representation size should be divisible by cardinality'
        self.fc_1 = nn.ModuleList([nn.Linear(appearance_size, sub_repr_size) for _ in range(cardinality)])
        self.fc_2 = nn.ModuleList([nn.Linear(spatial_size, sub_repr_size) for _ in range(cardinality)])
        self.fc_3 = nn.ModuleList([nn.Linear(sub_repr_size, hidden_state_size) for _ in range(cardinality)])

    def forward(self, appearance: Tensor, spatial: Tensor) ->Tensor:
        return F.relu(torch.stack([fc_3(F.relu(fc_1(appearance) * fc_2(spatial))) for fc_1, fc_2, fc_3 in zip(self.fc_1, self.fc_2, self.fc_3)]).sum(dim=0))


class ModifiedEncoderLayer(nn.Module):

    def __init__(self, hidden_size: int=256, representation_size: int=512, num_heads: int=8, dropout_prob: float=0.1, return_weights: bool=False) ->None:
        super().__init__()
        if representation_size % num_heads != 0:
            raise ValueError(f'The given representation size {representation_size} should be divisible by the number of attention heads {num_heads}.')
        self.sub_repr_size = int(representation_size / num_heads)
        self.hidden_size = hidden_size
        self.representation_size = representation_size
        self.num_heads = num_heads
        self.return_weights = return_weights
        self.unary = nn.Linear(hidden_size, representation_size)
        self.pairwise = nn.Linear(representation_size, representation_size)
        self.attn = nn.ModuleList([nn.Linear(3 * self.sub_repr_size, 1) for _ in range(num_heads)])
        self.message = nn.ModuleList([nn.Linear(self.sub_repr_size, self.sub_repr_size) for _ in range(num_heads)])
        self.aggregate = nn.Linear(representation_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.ffn = pocket.models.FeedForwardNetwork(hidden_size, hidden_size * 4, dropout_prob)

    def reshape(self, x: Tensor) ->Tensor:
        new_x_shape = x.size()[:-1] + (self.num_heads, self.sub_repr_size)
        x = x.view(*new_x_shape)
        if len(new_x_shape) == 3:
            return x.permute(1, 0, 2)
        elif len(new_x_shape) == 4:
            return x.permute(2, 0, 1, 3)
        else:
            raise ValueError('Incorrect tensor shape')

    def forward(self, x: Tensor, y: Tensor) ->Tuple[Tensor, Optional[Tensor]]:
        device = x.device
        n = len(x)
        u = F.relu(self.unary(x))
        p = F.relu(self.pairwise(y))
        u_r = self.reshape(u)
        p_r = self.reshape(p)
        i, j = torch.meshgrid(torch.arange(n, device=device), torch.arange(n, device=device))
        attn_features = torch.cat([u_r[:, i], u_r[:, j], p_r], dim=-1)
        weights = [F.softmax(l(f), dim=0) for f, l in zip(attn_features, self.attn)]
        u_r_repeat = u_r.unsqueeze(dim=2).repeat(1, 1, n, 1)
        messages = [l(f_1 * f_2) for f_1, f_2, l in zip(u_r_repeat, p_r, self.message)]
        aggregated_messages = self.aggregate(F.relu(torch.cat([(w * m).sum(dim=0) for w, m in zip(weights, messages)], dim=-1)))
        aggregated_messages = self.dropout(aggregated_messages)
        x = self.norm(x + aggregated_messages)
        x = self.ffn(x)
        if self.return_weights:
            attn = weights
        else:
            attn = None
        return x, attn


class ModifiedEncoder(nn.Module):

    def __init__(self, hidden_size: int=256, representation_size: int=512, num_heads: int=8, num_layers: int=2, dropout_prob: float=0.1, return_weights: bool=False) ->None:
        super().__init__()
        self.num_layers = num_layers
        self.mod_enc = nn.ModuleList([ModifiedEncoderLayer(hidden_size=hidden_size, representation_size=representation_size, num_heads=num_heads, dropout_prob=dropout_prob, return_weights=return_weights) for _ in range(num_layers)])

    def forward(self, x: Tensor, y: Tensor) ->Tuple[Tensor, List[Optional[Tensor]]]:
        attn_weights = []
        for layer in self.mod_enc:
            x, attn = layer(x, y)
            attn_weights.append(attn)
        return x, attn_weights


def compute_spatial_encodings(boxes_1: List[Tensor], boxes_2: List[Tensor], shapes: List[Tuple[int, int]], eps: float=1e-10) ->Tensor:
    """
    Parameters:
    -----------
    boxes_1: List[Tensor]
        First set of bounding boxes (M, 4)
    boxes_1: List[Tensor]
        Second set of bounding boxes (M, 4)
    shapes: List[Tuple[int, int]]
        Image shapes, heights followed by widths
    eps: float
        A small constant used for numerical stability

    Returns:
    --------
    Tensor
        Computed spatial encodings between the boxes (N, 36)
    """
    features = []
    for b1, b2, shape in zip(boxes_1, boxes_2, shapes):
        h, w = shape
        c1_x = (b1[:, 0] + b1[:, 2]) / 2
        c1_y = (b1[:, 1] + b1[:, 3]) / 2
        c2_x = (b2[:, 0] + b2[:, 2]) / 2
        c2_y = (b2[:, 1] + b2[:, 3]) / 2
        b1_w = b1[:, 2] - b1[:, 0]
        b1_h = b1[:, 3] - b1[:, 1]
        b2_w = b2[:, 2] - b2[:, 0]
        b2_h = b2[:, 3] - b2[:, 1]
        d_x = torch.abs(c2_x - c1_x) / (b1_w + eps)
        d_y = torch.abs(c2_y - c1_y) / (b1_h + eps)
        iou = torch.diag(box_ops.box_iou(b1, b2))
        f = torch.stack([c1_x / w, c1_y / h, c2_x / w, c2_y / h, b1_w / w, b1_h / h, b2_w / w, b2_h / h, b1_w * b1_h / (h * w), b2_w * b2_h / (h * w), b2_w * b2_h / (b1_w * b1_h + eps), b1_w / (b1_h + eps), b2_w / (b2_h + eps), iou, (c2_x > c1_x).float() * d_x, (c2_x < c1_x).float() * d_x, (c2_y > c1_y).float() * d_y, (c2_y < c1_y).float() * d_y], 1)
        features.append(torch.cat([f, torch.log(f + eps)], 1))
    return torch.cat(features)


class InteractionHead(nn.Module):
    """
    Interaction head that constructs and classifies box pairs

    Parameters:
    -----------
    box_pair_predictor: nn.Module
        Module that classifies box pairs
    hidden_state_size: int
        Size of the object features
    representation_size: int
        Size of the human-object pair features
    num_channels: int
        Number of channels in the global image features
    num_classes: int
        Number of target classes
    human_idx: int
        The index of human/person class
    object_class_to_target_class: List[list]
        The set of valid action classes for each object type
    """

    def __init__(self, box_pair_predictor: nn.Module, hidden_state_size: int, representation_size: int, num_channels: int, num_classes: int, human_idx: int, object_class_to_target_class: List[list]) ->None:
        super().__init__()
        self.box_pair_predictor = box_pair_predictor
        self.hidden_state_size = hidden_state_size
        self.representation_size = representation_size
        self.num_classes = num_classes
        self.human_idx = human_idx
        self.object_class_to_target_class = object_class_to_target_class
        self.spatial_head = nn.Sequential(nn.Linear(36, 128), nn.ReLU(), nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, representation_size), nn.ReLU())
        self.coop_layer = ModifiedEncoder(hidden_size=hidden_state_size, representation_size=representation_size, num_layers=2, return_weights=True)
        self.comp_layer = pocket.models.TransformerEncoderLayer(hidden_size=representation_size * 2, return_weights=True)
        self.mbf = MultiBranchFusion(hidden_state_size * 2, representation_size, representation_size, cardinality=16)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.mbf_g = MultiBranchFusion(num_channels, representation_size, representation_size, cardinality=16)

    def compute_prior_scores(self, x: Tensor, y: Tensor, scores: Tensor, object_class: Tensor) ->Tensor:
        prior_h = torch.zeros(len(x), self.num_classes, device=scores.device)
        prior_o = torch.zeros_like(prior_h)
        p = 1.0 if self.training else 2.8
        s_h = scores[x].pow(p)
        s_o = scores[y].pow(p)
        target_cls_idx = [self.object_class_to_target_class[obj.item()] for obj in object_class[y]]
        pair_idx = [i for i, tar in enumerate(target_cls_idx) for _ in tar]
        flat_target_idx = [t for tar in target_cls_idx for t in tar]
        prior_h[pair_idx, flat_target_idx] = s_h[pair_idx]
        prior_o[pair_idx, flat_target_idx] = s_o[pair_idx]
        return torch.stack([prior_h, prior_o])

    def forward(self, features: OrderedDict, image_shapes: Tensor, region_props: List[dict]):
        """
        Parameters:
        -----------
        features: OrderedDict
            Feature maps returned by FPN
        image_shapes: Tensor
            (B, 2) Image shapes, heights followed by widths
        region_props: List[dict]
            Region proposals with the following keys
            `boxes`: Tensor
                (N, 4) Bounding boxes
            `scores`: Tensor
                (N,) Object confidence scores
            `labels`: Tensor
                (N,) Object class indices
            `hidden_states`: Tensor
                (N, 256) Object features
        """
        device = features.device
        global_features = self.avg_pool(features).flatten(start_dim=1)
        boxes_h_collated = []
        boxes_o_collated = []
        prior_collated = []
        object_class_collated = []
        pairwise_tokens_collated = []
        attn_maps_collated = []
        for b_idx, props in enumerate(region_props):
            boxes = props['boxes']
            scores = props['scores']
            labels = props['labels']
            unary_tokens = props['hidden_states']
            is_human = labels == self.human_idx
            n_h = torch.sum(is_human)
            n = len(boxes)
            if not torch.all(labels[:n_h] == self.human_idx):
                h_idx = torch.nonzero(is_human).squeeze(1)
                o_idx = torch.nonzero(is_human == 0).squeeze(1)
                perm = torch.cat([h_idx, o_idx])
                boxes = boxes[perm]
                scores = scores[perm]
                labels = labels[perm]
                unary_tokens = unary_tokens[perm]
            if n_h == 0 or n <= 1:
                pairwise_tokens_collated.append(torch.zeros(0, 2 * self.representation_size, device=device))
                boxes_h_collated.append(torch.zeros(0, device=device, dtype=torch.int64))
                boxes_o_collated.append(torch.zeros(0, device=device, dtype=torch.int64))
                object_class_collated.append(torch.zeros(0, device=device, dtype=torch.int64))
                prior_collated.append(torch.zeros(2, 0, self.num_classes, device=device))
                continue
            x, y = torch.meshgrid(torch.arange(n, device=device), torch.arange(n, device=device))
            x_keep, y_keep = torch.nonzero(torch.logical_and(x != y, x < n_h)).unbind(1)
            if len(x_keep) == 0:
                raise ValueError('There are no valid human-object pairs')
            x = x.flatten()
            y = y.flatten()
            box_pair_spatial = compute_spatial_encodings([boxes[x]], [boxes[y]], [image_shapes[b_idx]])
            box_pair_spatial = self.spatial_head(box_pair_spatial)
            box_pair_spatial_reshaped = box_pair_spatial.reshape(n, n, -1)
            unary_tokens, unary_attn = self.coop_layer(unary_tokens, box_pair_spatial_reshaped)
            pairwise_tokens = torch.cat([self.mbf(torch.cat([unary_tokens[x_keep], unary_tokens[y_keep]], 1), box_pair_spatial_reshaped[x_keep, y_keep]), self.mbf_g(global_features[b_idx, None], box_pair_spatial_reshaped[x_keep, y_keep])], dim=1)
            pairwise_tokens, pairwise_attn = self.comp_layer(pairwise_tokens)
            pairwise_tokens_collated.append(pairwise_tokens)
            boxes_h_collated.append(x_keep)
            boxes_o_collated.append(y_keep)
            object_class_collated.append(labels[y_keep])
            prior_collated.append(self.compute_prior_scores(x_keep, y_keep, scores, labels))
            attn_maps_collated.append((unary_attn, pairwise_attn))
        pairwise_tokens_collated = torch.cat(pairwise_tokens_collated)
        logits = self.box_pair_predictor(pairwise_tokens_collated)
        return logits, prior_collated, boxes_h_collated, boxes_o_collated, object_class_collated, attn_maps_collated


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h]
    return torch.stack(b, dim=-1)


class HungarianMatcher(nn.Module):

    def __init__(self, cost_object: float=1.0, cost_verb: float=1.0, cost_bbox: float=1.0, cost_giou: float=1.0) ->None:
        """
        Parameters:
        ----------
        cost_object: float
            Weight on the object classification term
        cost_verb: float
            Weight on the verb classification term
        cost_bbox:
            Weight on the L1 regression error
        cost_giou:
            Weight on the GIoU term
        """
        super().__init__()
        self.cost_object = cost_object
        self.cost_verb = cost_verb
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_object + cost_verb + cost_bbox + cost_giou, 'At least one cost coefficient should be non zero.'

    @torch.no_grad()
    def forward(self, bx_h: List[Tensor], bx_o: List[Tensor], objects: List[Tensor], prior: List[Tensor], logits: Tensor, targets: List[dict]) ->List[Tensor]:
        """
        Parameters:
        ----------
        bh: List[Tensor]
            (M, 4) Human bounding boxes in detected pairs
        bo: List[Tensor]
            (M, 4) Object bounding boxes in detected pairs
        objects: List[Tensor]
            (M,) Object class indices in each pair 
        prior: List[Tensor]
            (2, M, K) Object detection scores for the human and object boxes in each pair
        logits: Tensor
            (M_, K) Classification logits for all boxes pairs
        targets: List[dict]
            Targets for each image with the following keys, `boxes_h` (G, 4), `boxes_o` (G, 4),
            `labels` (G, 117), `objects` (G,)

        Returns:
        --------
        List[Tensor]
            A list of tuples for matched indices between detected pairs and ground truth pairs.

        """
        eps = 1e-06
        n = [len(p) for p in bx_h]
        gt_bx_h = [t['boxes_h'] for t in targets]
        gt_bx_o = [t['boxes_o'] for t in targets]
        scores = [(torch.sigmoid(lg) * p.prod(0)) for lg, p in zip(logits.split(n), prior)]
        gt_labels = [t['labels'] for t in targets]
        cost_verb = [(-0.5 * (s.matmul(l.T) / (l.sum(dim=1).unsqueeze(0) + eps) + (1 - s).matmul(1 - l.T) / (torch.sum(1 - l, dim=1).unsqueeze(0) + eps))) for s, l in zip(scores, gt_labels)]
        cost_bbox = [torch.max(torch.cdist(h, gt_h, p=1), torch.cdist(o, gt_o, p=1)) for h, o, gt_h, gt_o in zip(bx_h, bx_o, gt_bx_h, gt_bx_o)]
        cost_giou = [torch.max(-generalized_box_iou(box_cxcywh_to_xyxy(h), box_cxcywh_to_xyxy(gt_h)), -generalized_box_iou(box_cxcywh_to_xyxy(o), box_cxcywh_to_xyxy(gt_o))) for h, o, gt_h, gt_o in zip(bx_h, bx_o, gt_bx_h, gt_bx_o)]
        cost_object = [(-torch.log(obj.unsqueeze(1).eq(t['object']) * p[0].max(-1)[0].unsqueeze(1) + eps)) for obj, p, t in zip(objects, prior, targets)]
        C = [(c_v * self.cost_verb + c_b * self.cost_bbox + c_g * self.cost_giou + c_o * self.cost_object) for c_v, c_b, c_g, c_o in zip(cost_verb, cost_bbox, cost_giou, cost_object)]
        indices = [linear_sum_assignment(c.cpu()) for c in C]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class BoxPairCoder:

    def __init__(self, weights: Optional[List[float]]=None, bbox_xform_clip: float=math.log(1000.0 / 16)) ->None:
        if weights is None:
            weights = [10.0, 10.0, 5.0, 5.0]
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def encode(self, props_h: Tensor, props_o: Tensor, target_h: Tensor, target_o: Tensor) ->Tensor:
        """
        Compute the regression targets based on proposed boxes pair and target box pairs.
        NOTE that all boxes are presumed to have been normalised by image width and height
        and are in (c_x, c_y, w, h) format.

        Parameters:
        -----------
        props_h: Tensor
            (N, 4) Human box proposals
        props_o: Tensor
            (N, 4) Object box proposals
        target_h: Tensor
            (N, 4) Human box targets
        target_o: Tensor
            (N, 4) Object box targets

        Returns:
        --------
        box_deltas: Tensor
            (N, 8) Regression targets for proposed box pairs
        """
        wx, wy, ww, wh = self.weights
        dx_h = wx * (target_h[:, 0] - props_h[:, 0])
        dy_h = wy * (target_h[:, 1] - props_h[:, 1])
        dw_h = ww * torch.log(target_h[:, 2] / props_h[:, 2])
        dh_h = wh * torch.log(target_h[:, 3] / props_h[:, 3])
        dx_o = wx * (target_o[:, 0] - props_o[:, 0])
        dy_o = wy * (target_o[:, 1] - props_o[:, 1])
        dw_o = ww * torch.log(target_o[:, 2] / props_o[:, 2])
        dh_o = wh * torch.log(target_o[:, 3] / props_o[:, 3])
        box_deltas = torch.stack([dx_h, dy_h, dw_h, dh_h, dx_o, dy_o, dw_o, dh_o], dim=1)
        return box_deltas

    def decode(self, props_h: Tensor, props_o: Tensor, box_deltas: Tensor) ->Tuple[Tensor, Tensor]:
        """
        Recover the regressed box pairs based on the proposed pairs and the box deltas.
        NOTE that the proposed box pairs are presumed to have been normalised by image
        width and height and are in (c_x, c_y, w, h) format.

        Parameters:
        -----------
        props_h: Tensor
            (N, 4) Human box proposals
        props_o: Tensor
            (N, 4) Object box proposals
        box_deltas: Tensor
            (N, 8) Predicted regression values for proposed box pairs

        Returns:
        --------
        regressed_h: Tensor
            (N, 4) Regressed human boxes
        regressed_o: Tensor
            (N, 4) Regressed object boxes
        """
        weights = torch.as_tensor(self.weights).repeat(2)
        box_deltas = box_deltas / weights
        dx_h, dy_h, dw_h, dh_h, dx_o, dy_o, dw_o, dh_o = box_deltas.unbind(1)
        dw_h = torch.clamp(dw_h, max=self.bbox_xform_clip)
        dh_h = torch.clamp(dh_h, max=self.bbox_xform_clip)
        dw_o = torch.clamp(dw_o, max=self.bbox_xform_clip)
        dh_o = torch.clamp(dh_o, max=self.bbox_xform_clip)
        regressed_h = torch.stack([props_h[:, 0] + dx_h, props_h[:, 1] + dy_h, props_h[:, 2] * torch.exp(dw_h), props_h[:, 3] * torch.exp(dh_h)], dim=1)
        regressed_o = torch.stack([props_o[:, 0] + dx_o, props_o[:, 1] + dy_o, props_o[:, 2] * torch.exp(dw_o), props_o[:, 3] * torch.exp(dh_o)], dim=1)
        return regressed_h, regressed_o


def binary_focal_loss_with_logits(x: Tensor, y: Tensor, alpha: float=0.5, gamma: float=2.0, reduction: str='mean', eps: float=1e-06) ->Tensor:
    """
    Focal loss by Lin et al.
    https://arxiv.org/pdf/1708.02002.pdf

    L = - |1-y-alpha| * |y-x|^{gamma} * log(|1-y-x|)

    Parameters:
    -----------
    x: Tensor[N, K]
        Post-normalisation scores
    y: Tensor[N, K]
        Binary labels
    alpha: float
        Hyper-parameter that balances between postive and negative examples
    gamma: float
        Hyper-paramter suppresses well-classified examples
    reduction: str
        Reduction methods
    eps: float
        A small constant to avoid NaN values from 'PowBackward'

    Returns:
    --------
    loss: Tensor
        Computed loss tensor
    """
    loss = (1 - y - alpha).abs() * ((y - torch.sigmoid(x)).abs() + eps) ** gamma * torch.nn.functional.binary_cross_entropy_with_logits(x, y, reduction='none')
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise ValueError('Unsupported reduction method {}'.format(reduction))


class SetCriterion(nn.Module):

    def __init__(self, args) ->None:
        super().__init__()
        self.args = args
        self.matcher = HungarianMatcher(cost_object=args.set_cost_object, cost_verb=args.set_cost_verb, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)
        self.box_pair_coder = BoxPairCoder()

    def focal_loss(self, bx_h: List[Tensor], bx_o: List[Tensor], indices: List[Tensor], prior: List[Tensor], logits: Tensor, targets: List[dict]) ->Tensor:
        collated_labels = []
        for bh, bo, idx, tgt in zip(bx_h, bx_o, indices, targets):
            idx_h, idx_o = idx
            mask = torch.diag(torch.min(box_ops.box_iou(box_cxcywh_to_xyxy(bh[idx_h]), box_cxcywh_to_xyxy(tgt['boxes_h'][idx_o])), box_ops.box_iou(box_cxcywh_to_xyxy(bo[idx_h]), box_cxcywh_to_xyxy(tgt['boxes_o'][idx_o]))) > 0.5).unsqueeze(1)
            matched_labels = tgt['labels'][idx_o] * mask
            labels = torch.zeros(len(bh), self.args.num_classes, device=matched_labels.device)
            labels[idx_h] = matched_labels
            collated_labels.append(labels)
        collated_labels = torch.cat(collated_labels)
        prior = torch.cat(prior, dim=1).prod(0)
        x, y = torch.nonzero(prior).unbind(1)
        logits = logits[x, y]
        prior = prior[x, y]
        labels = collated_labels[x, y]
        n_p = len(torch.nonzero(labels))
        if dist.is_initialized():
            world_size = dist.get_world_size()
            n_p = torch.as_tensor([n_p], device='cuda')
            dist.barrier()
            dist.all_reduce(n_p)
            n_p = (n_p / world_size).item()
        loss = binary_focal_loss_with_logits(torch.log((prior + 1e-08) / (1 + torch.exp(-logits) - prior)), labels, reduction='sum', alpha=self.args.alpha, gamma=self.args.gamma)
        return loss / n_p

    def regression_loss(self, props_h: List[Tensor], props_o: List[Tensor], reg_h: List[Tensor], reg_o: List[Tensor], indices: List[Tensor], targets: List[dict], bbox_deltas: List[Tensor]) ->Tensor:
        props_h = torch.cat([b[i].view(-1, 4) for (i, _), b in zip(indices, props_h)])
        props_o = torch.cat([b[i].view(-1, 4) for (i, _), b in zip(indices, props_o)])
        reg_h = torch.cat([b[i].view(-1, 4) for (i, _), b in zip(indices, reg_h)])
        reg_o = torch.cat([b[i].view(-1, 4) for (i, _), b in zip(indices, reg_o)])
        tgt_h = torch.cat([t['boxes_h'][j].view(-1, 4) for (_, j), t in zip(indices, targets)])
        tgt_o = torch.cat([t['boxes_o'][j].view(-1, 4) for (_, j), t in zip(indices, targets)])
        bbox_deltas = torch.cat([d[i].view(-1, 8) for (i, _), d in zip(indices, bbox_deltas)])
        reg_targets = self.box_pair_coder.encode(props_h, props_o, tgt_h, tgt_o)
        huber_loss = F.smooth_l1_loss(bbox_deltas, reg_targets, beta=1 / 9, reduction='sum')
        huber_loss = huber_loss / len(bbox_deltas)
        giou_loss = 2 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(reg_h), box_cxcywh_to_xyxy(tgt_h))) - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(reg_o), box_cxcywh_to_xyxy(tgt_o)))
        giou_loss = giou_loss.sum() / len(bbox_deltas)
        return dict(huber_loss=huber_loss, giou_loss=giou_loss)

    def forward(self, boxes: List[Tensor], bh: List[Tensor], bo: List[Tensor], objects: List[Tensor], prior: List[Tensor], logits: Tensor, bbox_deltas: Tensor, targets: List[dict]) ->Dict[str, Tensor]:
        bx_h = [b[h] for b, h in zip(boxes, bh)]
        bx_o = [b[o] for b, o in zip(boxes, bo)]
        indices = self.matcher(bx_h, bx_o, objects, prior, logits, targets)
        loss_dict = {'focal_loss': self.focal_loss(bx_h, bx_o, indices, prior, logits, targets)}
        return loss_dict


class UPT(nn.Module):
    """
    Unary-pairwise transformer

    Parameters:
    -----------
    detector: nn.Module
        Object detector (DETR)
    postprocessor: nn.Module
        Postprocessor for the object detector
    interaction_head: nn.Module
        Interaction head of the network
    human_idx: int
        Index of the human class
    num_classes: int
        Number of action classes
    alpha: float
        Hyper-parameter in the focal loss
    gamma: float
        Hyper-parameter in the focal loss
    box_score_thresh: float
        Threshold used to eliminate low-confidence objects
    fg_iou_thresh: float
        Threshold used to associate detections with ground truth
    min_instances: float
        Minimum number of instances (human or object) to sample
    max_instances: float
        Maximum number of instances (human or object) to sample
    """

    def __init__(self, detector: nn.Module, postprocessor: nn.Module, interaction_head: nn.Module, human_idx: int, num_classes: int, alpha: float=0.5, gamma: float=2.0, box_score_thresh: float=0.2, fg_iou_thresh: float=0.5, min_instances: int=3, max_instances: int=15) ->None:
        super().__init__()
        self.detector = detector
        self.postprocessor = postprocessor
        self.interaction_head = interaction_head
        self.human_idx = human_idx
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.box_score_thresh = box_score_thresh
        self.fg_iou_thresh = fg_iou_thresh
        self.min_instances = min_instances
        self.max_instances = max_instances

    def recover_boxes(self, boxes, size):
        boxes = box_ops.box_cxcywh_to_xyxy(boxes)
        h, w = size
        scale_fct = torch.stack([w, h, w, h])
        boxes = boxes * scale_fct
        return boxes

    def associate_with_ground_truth(self, boxes_h, boxes_o, targets):
        n = boxes_h.shape[0]
        labels = torch.zeros(n, self.num_classes, device=boxes_h.device)
        gt_bx_h = self.recover_boxes(targets['boxes_h'], targets['size'])
        gt_bx_o = self.recover_boxes(targets['boxes_o'], targets['size'])
        x, y = torch.nonzero(torch.min(box_iou(boxes_h, gt_bx_h), box_iou(boxes_o, gt_bx_o)) >= self.fg_iou_thresh).unbind(1)
        labels[x, targets['labels'][y]] = 1
        return labels

    def compute_interaction_loss(self, boxes, bh, bo, logits, prior, targets):
        labels = torch.cat([self.associate_with_ground_truth(bx[h], bx[o], target) for bx, h, o, target in zip(boxes, bh, bo, targets)])
        prior = torch.cat(prior, dim=1).prod(0)
        x, y = torch.nonzero(prior).unbind(1)
        logits = logits[x, y]
        prior = prior[x, y]
        labels = labels[x, y]
        n_p = len(torch.nonzero(labels))
        if dist.is_initialized():
            world_size = dist.get_world_size()
            n_p = torch.as_tensor([n_p], device='cuda')
            dist.barrier()
            dist.all_reduce(n_p)
            n_p = (n_p / world_size).item()
        loss = binary_focal_loss_with_logits(torch.log(prior / (1 + torch.exp(-logits) - prior) + 1e-08), labels, reduction='sum', alpha=self.alpha, gamma=self.gamma)
        return loss / n_p

    def prepare_region_proposals(self, results, hidden_states):
        region_props = []
        for res, hs in zip(results, hidden_states):
            sc, lb, bx = res.values()
            keep = batched_nms(bx, sc, lb, 0.5)
            sc = sc[keep].view(-1)
            lb = lb[keep].view(-1)
            bx = bx[keep].view(-1, 4)
            hs = hs[keep].view(-1, 256)
            keep = torch.nonzero(sc >= self.box_score_thresh).squeeze(1)
            is_human = lb == self.human_idx
            hum = torch.nonzero(is_human).squeeze(1)
            obj = torch.nonzero(is_human == 0).squeeze(1)
            n_human = is_human[keep].sum()
            n_object = len(keep) - n_human
            if n_human < self.min_instances:
                keep_h = sc[hum].argsort(descending=True)[:self.min_instances]
                keep_h = hum[keep_h]
            elif n_human > self.max_instances:
                keep_h = sc[hum].argsort(descending=True)[:self.max_instances]
                keep_h = hum[keep_h]
            else:
                keep_h = torch.nonzero(is_human[keep]).squeeze(1)
                keep_h = keep[keep_h]
            if n_object < self.min_instances:
                keep_o = sc[obj].argsort(descending=True)[:self.min_instances]
                keep_o = obj[keep_o]
            elif n_object > self.max_instances:
                keep_o = sc[obj].argsort(descending=True)[:self.max_instances]
                keep_o = obj[keep_o]
            else:
                keep_o = torch.nonzero(is_human[keep] == 0).squeeze(1)
                keep_o = keep[keep_o]
            keep = torch.cat([keep_h, keep_o])
            region_props.append(dict(boxes=bx[keep], scores=sc[keep], labels=lb[keep], hidden_states=hs[keep]))
        return region_props

    def postprocessing(self, boxes, bh, bo, logits, prior, objects, attn_maps, image_sizes):
        n = [len(b) for b in bh]
        logits = logits.split(n)
        detections = []
        for bx, h, o, lg, pr, obj, attn, size in zip(boxes, bh, bo, logits, prior, objects, attn_maps, image_sizes):
            pr = pr.prod(0)
            x, y = torch.nonzero(pr).unbind(1)
            scores = torch.sigmoid(lg[x, y])
            detections.append(dict(boxes=bx, pairing=torch.stack([h[x], o[x]]), scores=scores * pr[x, y], labels=y, objects=obj[x], attn_maps=attn, size=size))
        return detections

    def forward(self, images: List[Tensor], targets: Optional[List[dict]]=None) ->List[dict]:
        """
        Parameters:
        -----------
        images: List[Tensor]
            Input images in format (C, H, W)
        targets: List[dict], optional
            Human-object interaction targets

        Returns:
        --------
        results: List[dict]
            Detected human-object interactions. Each dict has the following keys:
            `boxes`: torch.Tensor
                (N, 4) Bounding boxes for detected human and object instances
            `pairing`: torch.Tensor
                (2, M) Pairing indices, with human instance preceding the object instance
            `scores`: torch.Tensor
                (M,) Interaction score for each pair
            `labels`: torch.Tensor
                (M,) Predicted action class for each pair
            `objects`: torch.Tensor
                (M,) Predicted object class for each pair
            `attn_maps`: list
                Attention weights in the cooperative and competitive layers
            `size`: torch.Tensor
                (2,) Image height and width
        """
        if self.training and targets is None:
            raise ValueError('In training mode, targets should be passed')
        image_sizes = torch.as_tensor([im.size()[-2:] for im in images], device=images[0].device)
        if isinstance(images, (list, torch.Tensor)):
            images = nested_tensor_from_tensor_list(images)
        features, pos = self.detector.backbone(images)
        src, mask = features[-1].decompose()
        assert mask is not None
        hs = self.detector.transformer(self.detector.input_proj(src), mask, self.detector.query_embed.weight, pos[-1])[0]
        outputs_class = self.detector.class_embed(hs)
        outputs_coord = self.detector.bbox_embed(hs).sigmoid()
        results = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        results = self.postprocessor(results, image_sizes)
        region_props = self.prepare_region_proposals(results, hs[-1])
        logits, prior, bh, bo, objects, attn_maps = self.interaction_head(features[-1].tensors, image_sizes, region_props)
        boxes = [r['boxes'] for r in region_props]
        if self.training:
            interaction_loss = self.compute_interaction_loss(boxes, bh, bo, logits, prior, targets)
            loss_dict = dict(interaction_loss=interaction_loss)
            return loss_dict
        detections = self.postprocessing(boxes, bh, bo, logits, prior, objects, attn_maps, image_sizes)
        return detections


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (MultiBranchFusion,
     lambda: ([], {'appearance_size': 4, 'spatial_size': 4, 'hidden_state_size': 4, 'cardinality': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_fredzzhang_upt(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

