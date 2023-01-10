import sys
_module = sys.modules[__name__]
del sys
eval_detection = _module
utils = _module
datasets = _module
data_utils = _module
tad_dataset = _module
tad_eval = _module
demo = _module
engine = _module
main = _module
models = _module
custom_loss = _module
matcher = _module
roi_align = _module
roi_align = _module
setup = _module
temporal_deform_attn = _module
temporal_deform_attn = _module
position_encoding = _module
tadtr = _module
transformer = _module
opts = _module
util = _module
logger = _module
misc = _module
segment_ops = _module

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


import math


import logging


import pandas as pd


import numpy as np


import torch


import torch.nn.functional as F


import torch.utils.data


import time


from typing import Iterable


import random


import re


from torch.utils.data import DataLoader


from torch.utils.data import DistributedSampler


from scipy.optimize import linear_sum_assignment


from torch import nn


from torch.autograd import Function


from torch.autograd.function import once_differentiable


from torch.nn.modules.utils import _pair


from torch.utils.cpp_extension import CUDA_HOME


from torch.utils.cpp_extension import CppExtension


from torch.utils.cpp_extension import CUDAExtension


import warnings


from torch.nn.init import xavier_uniform_


from torch.nn.init import constant_


import copy


from torch import Tensor


from torch.nn.init import uniform_


from torch.nn.init import normal_


from collections import defaultdict


from collections import deque


from typing import Optional


from typing import List


import torch.distributed as dist


import torchvision


def segment_cw_to_t1t2(x):
    """corresponds to box_cxcywh_to_xyxy in detr
    Params:
        x: segments in (center, width) format, shape=(*, 2)
    Returns:
        segments in (t_start, t_end) format, shape=(*, 2)
    """
    if not isinstance(x, np.ndarray):
        x_c, w = x.unbind(-1)
        b = [x_c - 0.5 * w, x_c + 0.5 * w]
        return torch.stack(b, dim=-1)
    else:
        x_c, w = x[..., 0], x[..., 1]
        b = [(x_c - 0.5 * w)[..., None], (x_c + 0.5 * w)[..., None]]
        return np.concatenate(b, axis=-1)


def segment_length(segments):
    return (segments[:, 1] - segments[:, 0]).clamp(min=0)


def segment_iou(segments1, segments2):
    """
    Temporal IoU between 

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(segments1)
    and M = len(segments2)
    """
    assert (segments1[:, 1] >= segments1[:, 0]).all()
    area1 = segment_length(segments1)
    area2 = segment_length(segments2)
    l = torch.max(segments1[:, None, 0], segments2[:, 0])
    r = torch.min(segments1[:, None, 1], segments2[:, 1])
    inter = (r - l).clamp(min=0)
    union = area1[:, None] + area2 - inter
    iou = inter / union
    return iou


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float=1, cost_seg: float=1, cost_iou: float=1):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_seg: This is the relative weight of the L1 error of the segment coordinates in the matching cost
            cost_iou: This is the relative weight of the iou loss of the segment in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_seg = cost_seg
        self.cost_iou = cost_iou
        assert cost_class != 0 or cost_seg != 0 or cost_iou != 0, 'all costs cant be 0'

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_segments": Tensor of dim [batch_size, num_queries, 2] with the predicted segment coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_segments] (where num_target_segments is the number of ground-truth
                           objects in the target) containing the class labels
                 "segments": Tensor of dim [num_target_segments, 2] containing the target segment coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_segments)
        """
        bs, num_queries = outputs['pred_logits'].shape[:2]
        out_prob = outputs['pred_logits'].flatten(0, 1).sigmoid()
        out_seg = outputs['pred_segments'].flatten(0, 1)
        tgt_ids = torch.cat([v['labels'] for v in targets])
        tgt_seg = torch.cat([v['segments'] for v in targets])
        alpha = 0.25
        gamma = 2.0
        neg_cost_class = (1 - alpha) * out_prob ** gamma * -(1 - out_prob + 1e-08).log()
        pos_cost_class = alpha * (1 - out_prob) ** gamma * -(out_prob + 1e-08).log()
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
        cost_seg = torch.cdist(out_seg, tgt_seg, p=1)
        cost_iou = -segment_iou(segment_cw_to_t1t2(out_seg), segment_cw_to_t1t2(tgt_seg))
        C = self.cost_seg * cost_seg + self.cost_class * cost_class + self.cost_iou * cost_iou
        C = C.view(bs, num_queries, -1).cpu()
        sizes = [len(v['segments']) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class _Align1D(Function):

    @staticmethod
    def forward(ctx, input, roi, feature_dim, ratio):
        ctx.save_for_backward(roi)
        ctx.feature_dim = feature_dim
        ctx.input_shape = input.size()
        ctx.sampling_ratio = ratio
        output = _align_1d.forward(input, roi, feature_dim, ratio)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        rois, = ctx.saved_tensors
        feature_dim = ctx.feature_dim
        bs, ch, t = ctx.input_shape
        ratio = ctx.sampling_ratio
        grad_input = _align_1d.backward(grad_output, rois, feature_dim, bs, ch, t, ratio)
        return grad_input, None, None, None, None


align1d = _Align1D.apply


class ROIAlign(nn.Module):

    def __init__(self, feature_dim, ratio=0):
        super(ROIAlign, self).__init__()
        self.feature_dim = feature_dim
        self.ratio = ratio

    def forward(self, input, rois):
        assert input.device == rois.device, 'Align operation requires ' + 'both feature and roi are on the same device! ' + 'Get feature on {} but roi on {}'.format(input.device, rois.device)
        out = align1d(input, rois, self.feature_dim, self.ratio)
        return out

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '('
        tmpstr += 'feature_dim=' + str(self.feature_dim)
        tmpstr += 'sampling_ratio=' + str(self.ratio)
        tmpstr += ')'
        return tmpstr


def _is_power_of_2(n):
    if not isinstance(n, int) or n < 0:
        raise ValueError('invalid input for _is_power_of_2: {} (type: {})'.format(n, type(n)))
    return n & n - 1 == 0 and n != 0


def deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    """deformable attention implemeted with grid_sample."""
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    value_list = value.split([(H_ * W_) for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_ * M_, D_, H_, W_)
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_, mode='bilinear', padding_mode='zeros', align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    attention_weights = attention_weights.transpose(1, 2).reshape(N_ * M_, 1, Lq_, L_ * P_)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(N_, M_ * D_, Lq_)
    return output.transpose(1, 2).contiguous()


class DeformAttn(nn.Module):

    def __init__(self, d_model=256, n_levels=1, n_heads=8, n_points=4):
        """
        Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in DeformAttn to make the dimension of each attention head a power of 2 which is more efficient in our CUDA implementation.")
        assert n_levels == 1, 'multi-level attention is not supported!'
        self.seq2col_step = 64
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (4.0 * math.pi / self.n_heads)
        grid_init = thetas.cos()[:, None]
        grid_init = grid_init.view(self.n_heads, 1, 1, 1).repeat(1, self.n_levels, self.n_points, 1)
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

    def forward(self, query, reference_points, input_flatten, input_temporal_lens, input_level_start_index, input_padding_mask=None):
        """
        :param query (= src + pos)         (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 1), range in [0, 1], left (0), right (1), including padding area
                                        or (N, Length_{query}, n_levels, 2), add additional (t) to form reference segments
        :param input_flatten (=src)        (N, \\sum_{l=0}^{L-1} T_l, C)
        :param input_temporal_lens         (n_levels), [T_0, T_1, ..., T_(L-1)]
        :param input_level_start_index     (n_levels, ), [0, T_0, T_1, T_2, ..., T_{L-1}]
        :param input_padding_mask          (N, \\sum_{l=0}^{L-1} T_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert input_temporal_lens.sum() == Len_in
        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 1)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        if reference_points.shape[-1] == 1:
            offset_normalizer = input_temporal_lens[..., None]
            sampling_locations = reference_points[:, :, None, :, None, :] + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 2:
            sampling_locations = reference_points[:, :, None, :, None, :1] + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 1:] * 0.5
        else:
            raise ValueError('Last dim of reference_points must be 1 or 2, but get {} instead.'.format(reference_points.shape[-1]))
        if cfg.dfm_att_backend == 'pytorch' or cfg.disable_cuda:
            sampling_locations = torch.cat((sampling_locations, torch.ones_like(sampling_locations) * 0.5), dim=-1)
            input_spatial_shapes = torch.stack((torch.ones_like(input_temporal_lens), input_temporal_lens), dim=-1)
            output = deform_attn_core_pytorch(value, input_spatial_shapes, sampling_locations, attention_weights)
        else:
            raise NotImplementedError
        output = self.output_proj(output)
        return output, (sampling_locations, attention_weights)


class NestedTensor(object):

    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        cast_tensor = self.tensors
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on videos.
    """

    def __init__(self, num_pos_feats=256, temperature=10000, normalize=False, scale=None):
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
        x_embed = not_mask.cumsum(1, dtype=torch.float32)
        if self.normalize:
            eps = 1e-06
            x_embed = x_embed / (x_embed[:, -1:] + eps) * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos = pos_x.permute(0, 2, 1)
        return pos


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


@torch.jit.unused
def _onnx_nested_tensor_from_tensor_list(tensor_list: List[Tensor]) ->NestedTensor:
    max_size = []
    for i in range(tensor_list[0].dim()):
        max_size_i = torch.max(torch.stack([img.shape[i] for img in tensor_list]).to(torch.float32))
        max_size.append(max_size_i)
    max_size = tuple(max_size)
    padded_imgs = []
    padded_masks = []
    for img in tensor_list:
        padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
        padded_img = torch.nn.functional.pad(img, (0, padding[2], 0, padding[1], 0, padding[0]))
        padded_imgs.append(padded_img)
        m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
        padded_mask = torch.nn.functional.pad(m, (0, padding[2], 0, padding[1]), 'constant', 1)
        padded_masks.append(padded_mask)
    tensor = torch.stack(padded_imgs)
    mask = torch.stack(padded_masks)
    return NestedTensor(tensor, mask=mask)


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    if tensor_list[0].ndim == 3:
        if torchvision._is_tracing():
            return _onnx_nested_tensor_from_tensor_list(tensor_list)
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
    elif tensor_list[0].ndim == 2 or tensor_list[0].ndim == 4:
        max_size = max([video_ft.shape[1] for video_ft in tensor_list])
        if tensor_list[0].ndim == 2:
            batch_shape = [len(tensor_list), tensor_list[0].shape[0], max_size]
        else:
            batch_shape = [len(tensor_list), tensor_list[0].shape[0], max_size, tensor_list[0].shape[2], tensor_list[0].shape[3]]
        b, c, t = batch_shape[:3]
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, t), dtype=torch.bool, device=device)
        for video_ft, pad_video_ft, m in zip(tensor_list, tensor, mask):
            pad_video_ft[:video_ft.shape[0], :video_ft.shape[1]].copy_(video_ft)
            m[:video_ft.shape[1]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)


class TadTR(nn.Module):
    """ This is the TadTR module that performs temporal action detection """

    def __init__(self, position_embedding, transformer, num_classes, num_queries, aux_loss=True, with_segment_refine=True, with_act_reg=True):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See deformable_transformer.py
            num_classes: number of action classes
            num_queries: number of action queries, ie detection slot. This is the maximal number of actions
                         TadTR can detect in a single video.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_segment_refine: iterative segment refinement
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.segment_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)
        self.input_proj = nn.ModuleList([nn.Sequential(nn.Conv1d(2048, hidden_dim, kernel_size=1), nn.GroupNorm(32, hidden_dim))])
        self.position_embedding = position_embedding
        self.aux_loss = aux_loss
        self.with_segment_refine = with_segment_refine
        self.with_act_reg = with_act_reg
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.segment_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.segment_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
        num_pred = transformer.decoder.num_layers
        if with_segment_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.segment_embed = _get_clones(self.segment_embed, num_pred)
            nn.init.constant_(self.segment_embed[0].layers[-1].bias.data[1:], -2.0)
            self.transformer.decoder.segment_embed = self.segment_embed
        else:
            nn.init.constant_(self.segment_embed.layers[-1].bias.data[1:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.segment_embed = nn.ModuleList([self.segment_embed for _ in range(num_pred)])
            self.transformer.decoder.segment_embed = None
        if with_act_reg:
            self.roi_size = 16
            self.roi_scale = 0
            self.roi_extractor = ROIAlign(self.roi_size, self.roi_scale)
            self.actionness_pred = nn.Sequential(nn.Linear(self.roi_size * hidden_dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def _to_roi_align_format(self, rois, T, scale_factor=1):
        """Convert RoIs to RoIAlign format.
        Params:
            RoIs: normalized segments coordinates, shape (batch_size, num_segments, 4)
            T: length of the video feature sequence
        """
        B, N = rois.shape[:2]
        rois_center = rois[:, :, 0:1]
        rois_size = rois[:, :, 1:2] * scale_factor
        rois_abs = torch.cat((rois_center - rois_size / 2, rois_center + rois_size / 2), dim=2) * T
        rois_abs = torch.clamp(rois_abs, min=0, max=T)
        batch_ind = torch.arange(0, B).view((B, 1, 1))
        batch_ind = batch_ind.repeat(1, N, 1)
        rois_abs = torch.cat((batch_ind, rois_abs), dim=2)
        return rois_abs.view((B * N, 3)).detach()

    def forward(self, samples):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensors: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
            or a tuple of tensors and mask

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-action) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_segments": The normalized segments coordinates for all queries, represented as
                               (center, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized segment.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            if isinstance(samples, (list, tuple)):
                samples = NestedTensor(*samples)
            else:
                samples = nested_tensor_from_tensor_list(samples)
        pos = [self.position_embedding(samples)]
        src, mask = samples.tensors, samples.mask
        srcs = [self.input_proj[0](src)]
        masks = [mask]
        query_embeds = self.query_embed.weight
        hs, init_reference, inter_references, memory = self.transformer(srcs, masks, pos, query_embeds)
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.segment_embed[lvl](hs[lvl])
            if reference.shape[-1] == 2:
                tmp += reference
            else:
                assert reference.shape[-1] == 1
                tmp[..., 0] += reference[..., 0]
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        if not self.with_act_reg:
            out = {'pred_logits': outputs_class[-1], 'pred_segments': outputs_coord[-1]}
        else:
            B, N = outputs_coord[-1].shape[:2]
            origin_feat = memory
            rois = self._to_roi_align_format(outputs_coord[-1], origin_feat.shape[2], scale_factor=1.5)
            roi_features = self.roi_extractor(origin_feat, rois)
            roi_features = roi_features.view((B, N, -1))
            pred_actionness = self.actionness_pred(roi_features)
            last_layer_cls = outputs_class[-1]
            last_layer_reg = outputs_coord[-1]
            out = {'pred_logits': last_layer_cls, 'pred_segments': last_layer_reg, 'pred_actionness': pred_actionness}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{'pred_logits': a, 'pred_segments': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


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


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


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
    """ This class computes the loss for TadTR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth segments and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and segment)
    """

    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25):
        """ Create the criterion.
        Parameters:
            num_classes: number of action categories, omitting the special no-action category
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

    def loss_labels(self, outputs, targets, indices, num_segments, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_segments]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1], dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_segments, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}
        if log:
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_segments(self, outputs, targets, indices, num_segments):
        """Compute the losses related to the segmentes, the L1 regression loss and the IoU loss
           targets dicts must contain the key "segments" containing a tensor of dim [nb_target_segments, 2]
           The target segments are expected in format (center, width), normalized by the video length.
        """
        assert 'pred_segments' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_segments = outputs['pred_segments'][idx]
        target_segments = torch.cat([t['segments'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_segment = F.l1_loss(src_segments, target_segments, reduction='none')
        losses = {}
        losses['loss_segments'] = loss_segment.sum() / num_segments
        loss_iou = 1 - torch.diag(segment_ops.segment_iou(segment_ops.segment_cw_to_t1t2(src_segments), segment_ops.segment_cw_to_t1t2(target_segments)))
        losses['loss_iou'] = loss_iou.sum() / num_segments
        return losses

    def loss_actionness(self, outputs, targets, indices, num_segments):
        """Compute the actionness regression loss
           targets dicts must contain the key "segments" containing a tensor of dim [nb_target_segments, 2]
           The target segments are expected in format (center, width), normalized by the video length.
        """
        assert 'pred_segments' in outputs
        assert 'pred_actionness' in outputs
        src_segments = outputs['pred_segments'].view((-1, 2))
        target_segments = torch.cat([t['segments'] for t in targets], dim=0)
        losses = {}
        iou_mat = segment_ops.segment_iou(segment_ops.segment_cw_to_t1t2(src_segments), segment_ops.segment_cw_to_t1t2(target_segments))
        gt_iou = iou_mat.max(dim=1)[0]
        pred_actionness = outputs['pred_actionness']
        loss_actionness = F.l1_loss(pred_actionness.view(-1), gt_iou.view(-1).detach())
        losses['loss_iou'] = loss_actionness
        return losses

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for src, _ in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for _, tgt in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_segments, **kwargs):
        loss_map = {'labels': self.loss_labels, 'segments': self.loss_segments, 'actionness': self.loss_actionness}
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_segments, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        indices = self.matcher(outputs_without_aux, targets)
        num_segments = sum(len(t['labels']) for t in targets)
        num_segments = torch.as_tensor([num_segments], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_segments)
        num_segments = torch.clamp(num_segments / get_world_size(), min=1).item()
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_segments, **kwargs))
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if 'actionness' in loss:
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_segments, **kwargs)
                    l_dict = {(k + f'_{i}'): v for k, v in l_dict.items()}
                    losses.update(l_dict)
        self.indices = indices
        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the TADEvaluator"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes, fuse_score=True):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size] containing the duration of each video of the batch
        """
        out_logits, out_segments = outputs['pred_logits'], outputs['pred_segments']
        assert len(out_logits) == len(target_sizes)
        prob = out_logits.sigmoid()
        if fuse_score:
            prob *= outputs['pred_actionness']
        segments = segment_ops.segment_cw_to_t1t2(out_segments)
        if cfg.postproc_rank == 1:
            topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), min(cfg.postproc_ins_topk, prob.shape[1] * prob.shape[2]), dim=1)
            scores = topk_values
            topk_segments = topk_indexes // out_logits.shape[2]
            labels = topk_indexes % out_logits.shape[2]
            segments = torch.gather(segments, 1, topk_segments.unsqueeze(-1).repeat(1, 1, 2))
            query_ids = topk_segments
        else:
            scores, labels = torch.topk(prob, cfg.postproc_cls_topk, dim=-1)
            scores, labels = scores.flatten(1), labels.flatten(1)
            segments = segments[:, [(i // cfg.postproc_cls_topk) for i in range(cfg.postproc_cls_topk * segments.shape[1])], :]
            query_ids = (torch.arange(0, cfg.postproc_cls_topk * segments.shape[1], 1, dtype=labels.dtype, device=labels.device) // cfg.postproc_cls_topk)[None, :].repeat(labels.shape[0], 1)
        vid_length = target_sizes
        scale_fct = torch.stack([vid_length, vid_length], dim=1)
        segments = segments * scale_fct[:, None, :]
        results = [{'scores': s, 'labels': l, 'segments': b, 'query_ids': q} for s, l, b, q in zip(scores, labels, segments, query_ids)]
        return results


class DeformableTransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.segment_embed = None
        self.class_embed = None

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios, query_pos=None, src_padding_mask=None):
        """
        tgt: [bs, nq, C]
        reference_points: [bs, nq, 1 or 2]
        src: [bs, T, C]
        src_valid_ratios: [bs, levels]
        """
        output = tgt
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None, :, None]
            output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index, src_padding_mask)
            if self.segment_embed is not None:
                tmp = self.segment_embed[lid](output)
                if reference_points.shape[-1] == 2:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 1
                    new_reference_points = tmp
                    new_reference_points[..., :1] = tmp[..., :1] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)
        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)
        return output, reference_points


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == 'relu':
        return F.relu
    if activation == 'gelu':
        return F.gelu
    if activation == 'glu':
        return F.glu
    if activation == 'leaky_relu':
        return F.leaky_relu
    raise RuntimeError(f'activation should be relu/gelu, not {activation}.')


class DeformableTransformerDecoderLayer(nn.Module):

    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, activation='relu', n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        self.cross_attn = DeformAttn(d_model, n_levels, n_heads, n_points)
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
        if not cfg.disable_query_self_att:
            q = k = self.with_pos_embed(tgt, query_pos)
            tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)
        else:
            pass
        tgt2, _ = self.cross_attn(self.with_pos_embed(tgt, query_pos), reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask)
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
        for lvl, T_ in enumerate(spatial_shapes):
            ref = torch.linspace(0.5, T_ - 0.5, T_, dtype=torch.float32, device=device)
            ref = ref[None] / (valid_ratios[:, None, lvl] * T_)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points[..., None]

    def forward(self, src, temporal_lens, level_start_index, valid_ratios, pos=None, padding_mask=None):
        """
        src: shape=(bs, t, c)
        temporal_lens: shape=(n_levels). content: [t1, t2, t3, ...]
        level_start_index: shape=(n_levels,). [0, t1, t1+t2, ...]
        valid_ratios: shape=(bs, n_levels).
        """
        output = src
        reference_points = self.get_reference_points(temporal_lens, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, temporal_lens, level_start_index, padding_mask)
        return output


class DeformableTransformerEncoderLayer(nn.Module):

    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, activation='relu', n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        self.self_attn = DeformAttn(d_model, n_levels, n_heads, n_points)
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
        src2, _ = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src = self.forward_ffn(src)
        return src


class DeformableTransformer(nn.Module):

    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1, activation='relu', return_intermediate_dec=False, num_feature_levels=4, dec_n_points=4, enc_n_points=4):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward, dropout, activation, num_feature_levels, nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)
        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward, dropout, activation, num_feature_levels, nhead, dec_n_points)
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        self.reference_points = nn.Linear(d_model, 1)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, DeformAttn):
                m._reset_parameters()
        xavier_uniform_(self.reference_points.weight.data, gain=1.0)
        constant_(self.reference_points.bias.data, 0.0)
        normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, T = mask.shape
        valid_T = torch.sum(~mask, 1)
        valid_ratio = valid_T.float() / T
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds, query_embed=None):
        """
        Params:
            srcs: list of Tensor with shape (bs, c, t)
            masks: list of Tensor with shape (bs, t)
            pos_embeds: list of Tensor with shape (bs, c, t)
            query_embed: list of Tensor with shape (nq, 2c)
        Returns:
            hs: list, per layer output of decoder
            init_reference_out: reference points predicted from query embeddings
            inter_references_out: reference points predicted from each decoder layer
            memory: (bs, c, t), final output of the encoder
        """
        assert query_embed is not None
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        temporal_lens = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, t = src.shape
            temporal_lens.append(t)
            src = src.transpose(1, 2)
            pos_embed = pos_embed.transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        temporal_lens = torch.as_tensor(temporal_lens, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((temporal_lens.new_zeros((1,)), temporal_lens.cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        memory = self.encoder(src_flatten, temporal_lens, level_start_index, valid_ratios, lvl_pos_embed_flatten if cfg.use_pos_embed else None, mask_flatten)
        bs, _, c = memory.shape
        query_embed, tgt = torch.split(query_embed, c, dim=1)
        query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
        tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_embed).sigmoid()
        init_reference_out = reference_points
        hs, inter_references = self.decoder(tgt, reference_points, memory, temporal_lens, level_start_index, valid_ratios, query_embed, mask_flatten)
        inter_references_out = inter_references
        return hs, init_reference_out, inter_references_out, memory.transpose(1, 2)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (MLP,
     lambda: ([], {'input_dim': 4, 'hidden_dim': 4, 'output_dim': 4, 'num_layers': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_xlliu7_TadTR(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

