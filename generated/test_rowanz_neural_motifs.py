import sys
_module = sys.modules[__name__]
del sys
config = _module
dataloaders = _module
blob = _module
image_transforms = _module
mscoco = _module
visual_genome = _module
lib = _module
setup = _module
evaluation = _module
sg_eval = _module
sg_eval_all_rel_cates = _module
sg_eval_slow = _module
test_sg_eval = _module
anchor_targets = _module
box_utils = _module
generate_anchors = _module
build = _module
nms = _module
proposal_assignments_det = _module
proposal_assignments_gtbox = _module
proposal_assignments_postnms = _module
proposal_assignments_rel = _module
rel_assignments = _module
roi_align = _module
_ext = _module
functions = _module
modules = _module
roi_align = _module
get_dataset_counts = _module
get_union_boxes = _module
lstm = _module
decoder_rnn = _module
highway_lstm_cuda = _module
highway_lstm_layer = _module
alternating_highway_lstm = _module
object_detector = _module
pytorch_misc = _module
rel_model = _module
rel_model_stanford = _module
resnet = _module
sparse_targets = _module
surgery = _module
word_vectors = _module
misc = _module
motifs = _module
_visualize = _module
eval_rel_count = _module
eval_rels = _module
train_detector = _module
train_rels = _module

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


import numpy as np


import torch


from torch.autograd import Variable


from torch.nn import functional as F


from torch.nn.modules.module import Module


from torch.nn.functional import avg_pool2d


from torch.nn.functional import max_pool2d


from torch import nn


import torch.nn as nn


import torch.nn.functional as F


from torch.nn.utils.rnn import PackedSequence


from typing import Optional


from typing import Tuple


from torch.autograd import Function


from torch.nn import Parameter


from torch.nn.utils.rnn import pad_packed_sequence


from torch.nn.utils.rnn import pack_padded_sequence


import itertools


import torch.nn.parallel


from torch.nn.parallel._functions import Gather


from itertools import tee


import math


import torch.utils.model_zoo as model_zoo


from torch import optim


import time


import torch.backends.cudnn as cudnn


from torch.optim.lr_scheduler import ReduceLROnPlateau


class RoIAlignFunction(Function):

    def __init__(self, aligned_height, aligned_width, spatial_scale):
        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)
        self.feature_size = None

    def forward(self, features, rois):
        self.save_for_backward(rois)
        rois_normalized = rois.clone()
        self.feature_size = features.size()
        batch_size, num_channels, data_height, data_width = self.feature_size
        height = (data_height - 1) / self.spatial_scale
        width = (data_width - 1) / self.spatial_scale
        rois_normalized[:, (1)] /= width
        rois_normalized[:, (2)] /= height
        rois_normalized[:, (3)] /= width
        rois_normalized[:, (4)] /= height
        num_rois = rois.size(0)
        output = features.new(num_rois, num_channels, self.aligned_height,
            self.aligned_width).zero_()
        if features.is_cuda:
            res = roi_align.roi_align_forward_cuda(self.aligned_height,
                self.aligned_width, self.spatial_scale, features,
                rois_normalized, output)
            assert res == 1
        else:
            raise ValueError
        return output

    def backward(self, grad_output):
        assert self.feature_size is not None and grad_output.is_cuda
        rois = self.saved_tensors[0]
        rois_normalized = rois.clone()
        batch_size, num_channels, data_height, data_width = self.feature_size
        height = (data_height - 1) / self.spatial_scale
        width = (data_width - 1) / self.spatial_scale
        rois_normalized[:, (1)] /= width
        rois_normalized[:, (2)] /= height
        rois_normalized[:, (3)] /= width
        rois_normalized[:, (4)] /= height
        grad_input = rois_normalized.new(batch_size, num_channels,
            data_height, data_width).zero_()
        res = roi_align.roi_align_backward_cuda(self.aligned_height, self.
            aligned_width, self.spatial_scale, grad_output, rois_normalized,
            grad_input)
        assert res == 1
        return grad_input, None


class RoIAlign(Module):

    def __init__(self, aligned_height, aligned_width, spatial_scale):
        super(RoIAlign, self).__init__()
        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        return RoIAlignFunction(self.aligned_height, self.aligned_width,
            self.spatial_scale)(features, rois)


class RoIAlignAvg(Module):

    def __init__(self, aligned_height, aligned_width, spatial_scale):
        super(RoIAlignAvg, self).__init__()
        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        x = RoIAlignFunction(self.aligned_height + 1, self.aligned_width + 
            1, self.spatial_scale)(features, rois)
        return avg_pool2d(x, kernel_size=2, stride=1)


class RoIAlignMax(Module):

    def __init__(self, aligned_height, aligned_width, spatial_scale):
        super(RoIAlignMax, self).__init__()
        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        x = RoIAlignFunction(self.aligned_height + 1, self.aligned_width + 
            1, self.spatial_scale)(features, rois)
        return max_pool2d(x, kernel_size=2, stride=1)


BATCHNORM_MOMENTUM = 0.01


def union_boxes(fmap, rois, union_inds, pooling_size=14, stride=16):
    """
    :param fmap: (batch_size, d, IM_SIZE/stride, IM_SIZE/stride)
    :param rois: (num_rois, 5) with [im_ind, x1, y1, x2, y2]
    :param union_inds: (num_urois, 2) with [roi_ind1, roi_ind2]
    :param pooling_size: we'll resize to this
    :param stride:
    :return:
    """
    assert union_inds.size(1) == 2
    im_inds = rois[:, (0)][union_inds[:, (0)]]
    assert (im_inds.data == rois.data[:, (0)][union_inds[:, (1)]]).sum(
        ) == union_inds.size(0)
    union_rois = torch.cat((im_inds[:, (None)], torch.min(rois[:, 1:3][
        union_inds[:, (0)]], rois[:, 1:3][union_inds[:, (1)]]), torch.max(
        rois[:, 3:5][union_inds[:, (0)]], rois[:, 3:5][union_inds[:, (1)]])), 1
        )
    union_pools = RoIAlignFunction(pooling_size, pooling_size,
        spatial_scale=1 / stride)(fmap, union_rois)
    return union_pools


class UnionBoxesAndFeats(Module):

    def __init__(self, pooling_size=7, stride=16, dim=256, concat=False,
        use_feats=True):
        """
        :param pooling_size: Pool the union boxes to this dimension
        :param stride: pixel spacing in the entire image
        :param dim: Dimension of the feats
        :param concat: Whether to concat (yes) or add (False) the representations
        """
        super(UnionBoxesAndFeats, self).__init__()
        self.pooling_size = pooling_size
        self.stride = stride
        self.dim = dim
        self.use_feats = use_feats
        self.conv = nn.Sequential(nn.Conv2d(2, dim // 2, kernel_size=7,
            stride=2, padding=3, bias=True), nn.ReLU(inplace=True), nn.
            BatchNorm2d(dim // 2, momentum=BATCHNORM_MOMENTUM), nn.
            MaxPool2d(kernel_size=3, stride=2, padding=1), nn.Conv2d(dim //
            2, dim, kernel_size=3, stride=1, padding=1, bias=True), nn.ReLU
            (inplace=True), nn.BatchNorm2d(dim, momentum=BATCHNORM_MOMENTUM))
        self.concat = concat

    def forward(self, fmap, rois, union_inds):
        union_pools = union_boxes(fmap, rois, union_inds, pooling_size=self
            .pooling_size, stride=self.stride)
        if not self.use_feats:
            return union_pools.detach()
        pair_rois = torch.cat((rois[:, 1:][union_inds[:, (0)]], rois[:, 1:]
            [union_inds[:, (1)]]), 1).data.cpu().numpy()
        rects_np = draw_union_boxes(pair_rois, self.pooling_size * 4 - 1) - 0.5
        rects = Variable(torch.FloatTensor(rects_np), volatile=fmap.volatile)
        if self.concat:
            return torch.cat((union_pools, self.conv(rects)), 1)
        return union_pools + self.conv(rects)


def block_orthogonal(tensor, split_sizes, gain=1.0):
    """
    An initializer which allows initializing model parameters in "blocks". This is helpful
    in the case of recurrent models which use multiple gates applied to linear projections,
    which can be computed efficiently if they are concatenated together. However, they are
    separate parameters which should be initialized independently.
    Parameters
    ----------
    tensor : ``torch.Tensor``, required.
        A tensor to initialize.
    split_sizes : List[int], required.
        A list of length ``tensor.ndim()`` specifying the size of the
        blocks along that particular dimension. E.g. ``[10, 20]`` would
        result in the tensor being split into chunks of size 10 along the
        first dimension and 20 along the second.
    gain : float, optional (default = 1.0)
        The gain (scaling) applied to the orthogonal initialization.
    """
    if isinstance(tensor, Variable):
        block_orthogonal(tensor.data, split_sizes, gain)
        return tensor
    sizes = list(tensor.size())
    if any([(a % b != 0) for a, b in zip(sizes, split_sizes)]):
        raise ValueError(
            'tensor dimensions must be divisible by their respective split_sizes. Found size: {} and split_sizes: {}'
            .format(sizes, split_sizes))
    indexes = [list(range(0, max_size, split)) for max_size, split in zip(
        sizes, split_sizes)]
    for block_start_indices in itertools.product(*indexes):
        index_and_step_tuples = zip(block_start_indices, split_sizes)
        block_slice = tuple([slice(start_index, start_index + step) for 
            start_index, step in index_and_step_tuples])
        assert len(block_slice) == 2
        sizes = [(x.stop - x.start) for x in block_slice]
        tensor_copy = tensor.new(max(sizes), max(sizes))
        torch.nn.init.orthogonal(tensor_copy, gain=gain)
        tensor[block_slice] = tensor_copy[0:sizes[0], 0:sizes[1]]


def get_dropout_mask(dropout_probability: float, tensor_for_masking: torch.
    autograd.Variable):
    """
    Computes and returns an element-wise dropout mask for a given tensor, where
    each element in the mask is dropped out with probability dropout_probability.
    Note that the mask is NOT applied to the tensor - the tensor is passed to retain
    the correct CUDA tensor type for the mask.

    Parameters
    ----------
    dropout_probability : float, required.
        Probability of dropping a dimension of the input.
    tensor_for_masking : torch.Variable, required.


    Returns
    -------
    A torch.FloatTensor consisting of the binary mask scaled by 1/ (1 - dropout_probability).
    This scaling ensures expected values and variances of the output of applying this mask
     and the original tensor are the same.
    """
    binary_mask = tensor_for_masking.clone()
    binary_mask.data.copy_(torch.rand(tensor_for_masking.size()) >
        dropout_probability)
    dropout_mask = binary_mask.float().div(1.0 - dropout_probability)
    return dropout_mask


def nms_overlaps(boxes):
    """ get overlaps for each channel"""
    assert boxes.dim() == 3
    N = boxes.size(0)
    nc = boxes.size(1)
    max_xy = torch.min(boxes[:, (None), :, 2:].expand(N, N, nc, 2), boxes[(
        None), :, :, 2:].expand(N, N, nc, 2))
    min_xy = torch.max(boxes[:, (None), :, :2].expand(N, N, nc, 2), boxes[(
        None), :, :, :2].expand(N, N, nc, 2))
    inter = torch.clamp(max_xy - min_xy + 1.0, min=0)
    inters = inter[:, :, :, (0)] * inter[:, :, :, (1)]
    boxes_flat = boxes.view(-1, 4)
    areas_flat = (boxes_flat[:, (2)] - boxes_flat[:, (0)] + 1.0) * (
        boxes_flat[:, (3)] - boxes_flat[:, (1)] + 1.0)
    areas = areas_flat.view(boxes.size(0), boxes.size(1))
    union = -inters + areas[None] + areas[:, (None)]
    return inters / union


class Result(object):
    """ little container class for holding the detection result
        od: object detector, rm: rel model"""

    def __init__(self, od_obj_dists=None, rm_obj_dists=None, obj_scores=
        None, obj_preds=None, obj_fmap=None, od_box_deltas=None,
        rm_box_deltas=None, od_box_targets=None, rm_box_targets=None,
        od_box_priors=None, rm_box_priors=None, boxes_assigned=None,
        boxes_all=None, od_obj_labels=None, rm_obj_labels=None, rpn_scores=
        None, rpn_box_deltas=None, rel_labels=None, im_inds=None, fmap=None,
        rel_dists=None, rel_inds=None, rel_rep=None):
        self.__dict__.update(locals())
        del self.__dict__['self']

    def is_none(self):
        return all([(v is None) for k, v in self.__dict__.items() if k !=
            'self'])


def bbox_intersections(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    if isinstance(box_a, np.ndarray):
        assert isinstance(box_b, np.ndarray)
        return bbox_intersections_np(box_a, box_b)
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2), box_b[:, 
        2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2), box_b[:,
        :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp(max_xy - min_xy + 1.0, min=0)
    return inter[:, :, (0)] * inter[:, :, (1)]


def bbox_overlaps(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    if isinstance(box_a, np.ndarray):
        assert isinstance(box_b, np.ndarray)
        return bbox_overlaps_np(box_a, box_b)
    inter = bbox_intersections(box_a, box_b)
    area_a = ((box_a[:, (2)] - box_a[:, (0)] + 1.0) * (box_a[:, (3)] -
        box_a[:, (1)] + 1.0)).unsqueeze(1).expand_as(inter)
    area_b = ((box_b[:, (2)] - box_b[:, (0)] + 1.0) * (box_b[:, (3)] -
        box_b[:, (1)] + 1.0)).unsqueeze(0).expand_as(inter)
    union = area_a + area_b - inter
    return inter / union


def center_size(boxes):
    """ Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    wh = boxes[:, 2:] - boxes[:, :2] + 1.0
    if isinstance(boxes, np.ndarray):
        return np.column_stack((boxes[:, :2] + 0.5 * wh, wh))
    return torch.cat((boxes[:, :2] + 0.5 * wh, wh), 1)


def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    if isinstance(boxes, np.ndarray):
        return np.column_stack((boxes[:, :2] - 0.5 * boxes[:, 2:], boxes[:,
            :2] + 0.5 * (boxes[:, 2:] - 2.0)))
    return torch.cat((boxes[:, :2] - 0.5 * boxes[:, 2:], boxes[:, :2] + 0.5 *
        (boxes[:, 2:] - 2.0)), 1)


def bbox_preds(boxes, deltas):
    """
    Converts "deltas" (predicted by the network) along with prior boxes
    into (x1, y1, x2, y2) representation.
    :param boxes: Prior boxes, represented as (x1, y1, x2, y2)
    :param deltas: Offsets (tx, ty, tw, th)
    :param box_strides [num_boxes,] distance apart between boxes. anchor box can't go more than
       \\pm box_strides/2 from its current position. If None then we'll use the widths
       and heights
    :return: Transformed boxes
    """
    if boxes.size(0) == 0:
        return boxes
    prior_centers = center_size(boxes)
    xys = prior_centers[:, :2] + prior_centers[:, 2:] * deltas[:, :2]
    whs = torch.exp(deltas[:, 2:]) * prior_centers[:, 2:]
    return point_form(torch.cat((xys, whs), 1))


def enumerate_by_image(im_inds):
    im_inds_np = im_inds.cpu().numpy()
    initial_ind = int(im_inds_np[0])
    s = 0
    for i, val in enumerate(im_inds_np):
        if val != initial_ind:
            yield initial_ind, s, i
            initial_ind = int(val)
            s = i
    yield initial_ind, s, len(im_inds_np)


def _nms_single_im(scores, boxes, pre_nms_topn=12000, post_nms_topn=2000,
    nms_thresh=0.7):
    keep = torch.IntTensor(scores.size(0))
    vs, idx = torch.sort(scores, dim=0, descending=True)
    if idx.size(0) > pre_nms_topn:
        idx = idx[:pre_nms_topn]
    boxes_sorted = boxes[idx].contiguous()
    num_out = nms.nms_apply(keep, boxes_sorted, nms_thresh)
    num_out = min(num_out, post_nms_topn)
    keep = keep[:num_out].long()
    keep = idx[keep.cuda(scores.get_device())]
    return keep


def apply_nms(scores, boxes, pre_nms_topn=12000, post_nms_topn=2000,
    boxes_per_im=None, nms_thresh=0.7):
    """
    Note - this function is non-differentiable so everything is assumed to be a tensor, not
    a variable.
        """
    just_inds = boxes_per_im is None
    if boxes_per_im is None:
        boxes_per_im = [boxes.size(0)]
    s = 0
    keep = []
    im_per = []
    for bpi in boxes_per_im:
        e = s + int(bpi)
        keep_im = _nms_single_im(scores[s:e], boxes[s:e], pre_nms_topn,
            post_nms_topn, nms_thresh)
        keep.append(keep_im + s)
        im_per.append(keep_im.size(0))
        s = e
    inds = torch.cat(keep, 0)
    if just_inds:
        return inds
    return inds, im_per


def filter_det(scores, boxes, start_ind=0, max_per_img=100, thresh=0.001,
    pre_nms_topn=6000, post_nms_topn=300, nms_thresh=0.3,
    nms_filter_duplicates=True):
    """
    Filters the detections for a single image
    :param scores: [num_rois, num_classes]
    :param boxes: [num_rois, num_classes, 4]. Assumes the boxes have been clamped
    :param max_per_img: Max detections per image
    :param thresh: Threshold for calling it a good box
    :param nms_filter_duplicates: True if we shouldn't allow for mulitple detections of the
           same box (with different labels)
    :return: A numpy concatenated array with up to 100 detections/img [num_im, x1, y1, x2, y2, score, cls]
    """
    valid_cls = (scores[:, 1:].data.max(0)[0] > thresh).nonzero() + 1
    if valid_cls.dim() == 0:
        return None
    nms_mask = scores.data.clone()
    nms_mask.zero_()
    for c_i in valid_cls.squeeze(1).cpu():
        scores_ci = scores.data[:, (c_i)]
        boxes_ci = boxes.data[:, (c_i)]
        keep = apply_nms(scores_ci, boxes_ci, pre_nms_topn=pre_nms_topn,
            post_nms_topn=post_nms_topn, nms_thresh=nms_thresh)
        nms_mask[:, (c_i)][keep] = 1
    dists_all = Variable(nms_mask * scores.data, volatile=True)
    if nms_filter_duplicates:
        scores_pre, labels_pre = dists_all.data.max(1)
        inds_all = scores_pre.nonzero()
        assert inds_all.dim() != 0
        inds_all = inds_all.squeeze(1)
        labels_all = labels_pre[inds_all]
        scores_all = scores_pre[inds_all]
    else:
        nz = nms_mask.nonzero()
        assert nz.dim() != 0
        inds_all = nz[:, (0)]
        labels_all = nz[:, (1)]
        scores_all = scores.data.view(-1)[inds_all * scores.data.size(1) +
            labels_all]
    vs, idx = torch.sort(scores_all, dim=0, descending=True)
    idx = idx[vs > thresh]
    if max_per_img < idx.size(0):
        idx = idx[:max_per_img]
    inds_all = inds_all[idx] + start_ind
    scores_all = Variable(scores_all[idx], volatile=True)
    labels_all = Variable(labels_all[idx], volatile=True)
    return inds_all, scores_all, labels_all


def filter_roi_proposals(box_preds, class_preds, boxes_per_im, nms_thresh=
    0.7, pre_nms_topn=12000, post_nms_topn=2000):
    inds, im_per = apply_nms(class_preds, box_preds, pre_nms_topn=
        pre_nms_topn, post_nms_topn=post_nms_topn, boxes_per_im=
        boxes_per_im, nms_thresh=nms_thresh)
    img_inds = torch.cat([(val * torch.ones(i)) for val, i in enumerate(
        im_per)], 0).cuda(box_preds.get_device())
    rois = torch.cat((img_inds[:, (None)], box_preds[inds]), 1)
    return rois


def gather_res(outputs, target_device, dim=0):
    """
    Assuming the signatures are the same accross results!
    """
    out = outputs[0]
    args = {field: Gather.apply(target_device, dim, *[getattr(o, field) for
        o in outputs]) for field, v in out.__dict__.items() if v is not None}
    return type(out)(**args)


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def load_resnet():
    model = resnet101(pretrained=True)
    del model.layer4
    del model.avgpool
    del model.fc
    return model


def load_vgg(use_dropout=True, use_relu=True, use_linear=True, pretrained=True
    ):
    model = vgg16(pretrained=pretrained)
    del model.features._modules['30']
    del model.classifier._modules['6']
    if not use_dropout:
        del model.classifier._modules['5']
        if not use_relu:
            del model.classifier._modules['4']
            if not use_linear:
                del model.classifier._modules['3']
    return model


FG_FRACTION = 0.25


ROIS_PER_IMG = 256


BG_THRESH_HI = 0.5


BG_THRESH_LO = 0.0


def _sel_inds(max_overlaps, fg_thresh=0.5, fg_rois_per_image=128,
    rois_per_image=256):
    fg_inds = np.where(max_overlaps >= fg_thresh)[0]
    fg_rois_per_this_image = min(fg_rois_per_image, fg_inds.shape[0])
    if fg_inds.size > 0:
        fg_inds = npr.choice(fg_inds, size=fg_rois_per_this_image, replace=
            False)
    bg_inds = np.where((max_overlaps < BG_THRESH_HI) & (max_overlaps >=
        BG_THRESH_LO))[0]
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size)
    if bg_inds.size > 0:
        bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image, replace=
            False)
    return np.append(fg_inds, bg_inds), fg_rois_per_this_image


def to_variable(f):
    """
    Decorator that pushes all the outputs to a variable
    :param f: 
    :return: 
    """

    def variable_wrapper(*args, **kwargs):
        rez = f(*args, **kwargs)
        if isinstance(rez, tuple):
            return tuple([Variable(x) for x in rez])
        return Variable(rez)
    return variable_wrapper


@to_variable
def proposal_assignments_det(rpn_rois, gt_boxes, gt_classes, image_offset,
    fg_thresh=0.5):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    :param rpn_rois: [img_ind, x1, y1, x2, y2]
    :param gt_boxes:   [num_boxes, 4] array of x0, y0, x1, y1
    :param gt_classes: [num_boxes, 2] array of [img_ind, class]
    :param Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)
    :return:
        rois: [num_rois, 5]
        labels: [num_rois] array of labels
        bbox_targets [num_rois, 4] array of targets for the labels.
    """
    fg_rois_per_image = int(np.round(ROIS_PER_IMG * FG_FRACTION))
    gt_img_inds = gt_classes[:, (0)] - image_offset
    all_boxes = torch.cat([rpn_rois[:, 1:], gt_boxes], 0)
    ims_per_box = torch.cat([rpn_rois[:, (0)].long(), gt_img_inds], 0)
    im_sorted, idx = torch.sort(ims_per_box, 0)
    all_boxes = all_boxes[idx]
    num_images = int(im_sorted[-1]) + 1
    labels = []
    rois = []
    bbox_targets = []
    for im_ind in range(num_images):
        g_inds = (gt_img_inds == im_ind).nonzero()
        if g_inds.dim() == 0:
            continue
        g_inds = g_inds.squeeze(1)
        g_start = g_inds[0]
        g_end = g_inds[-1] + 1
        t_inds = (im_sorted == im_ind).nonzero().squeeze(1)
        t_start = t_inds[0]
        t_end = t_inds[-1] + 1
        ious = bbox_overlaps(all_boxes[t_start:t_end], gt_boxes[g_start:g_end])
        max_overlaps, gt_assignment = ious.max(1)
        max_overlaps = max_overlaps.cpu().numpy()
        gt_assignment += g_start
        keep_inds_np, num_fg = _sel_inds(max_overlaps, fg_thresh,
            fg_rois_per_image, ROIS_PER_IMG)
        if keep_inds_np.size == 0:
            continue
        keep_inds = torch.LongTensor(keep_inds_np).cuda(rpn_rois.get_device())
        labels_ = gt_classes[:, (1)][gt_assignment[keep_inds]]
        bbox_target_ = gt_boxes[gt_assignment[keep_inds]]
        if num_fg < labels_.size(0):
            labels_[num_fg:] = 0
        rois_ = torch.cat((im_sorted[t_start:t_end, (None)][keep_inds].
            float(), all_boxes[t_start:t_end][keep_inds]), 1)
        labels.append(labels_)
        rois.append(rois_)
        bbox_targets.append(bbox_target_)
    rois = torch.cat(rois, 0)
    labels = torch.cat(labels, 0)
    bbox_targets = torch.cat(bbox_targets, 0)
    return rois, labels, bbox_targets


RELS_PER_IMG = 256


REL_FG_FRACTION = 0.25


def diagonal_inds(tensor):
    """
    Returns the indices required to go along first 2 dims of tensor in diag fashion
    :param tensor: thing
    :return: 
    """
    assert tensor.dim() >= 2
    assert tensor.size(0) == tensor.size(1)
    size = tensor.size(0)
    arange_inds = tensor.new(size).long()
    torch.arange(0, tensor.size(0), out=arange_inds)
    return (size + 1) * arange_inds


def random_choose(tensor, num):
    """randomly choose indices"""
    num_choose = min(tensor.size(0), num)
    if num_choose == tensor.size(0):
        return tensor
    rand_idx = np.random.choice(tensor.size(0), size=num, replace=False)
    rand_idx = torch.LongTensor(rand_idx).cuda(tensor.get_device())
    chosen = tensor[rand_idx].contiguous()
    return chosen


@to_variable
def proposal_assignments_gtbox(rois, gt_boxes, gt_classes, gt_rels,
    image_offset, fg_thresh=0.5):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    :param rpn_rois: [img_ind, x1, y1, x2, y2]
    :param gt_boxes:   [num_boxes, 4] array of x0, y0, x1, y1]. Not needed it seems
    :param gt_classes: [num_boxes, 2] array of [img_ind, class]
        Note, the img_inds here start at image_offset
    :param gt_rels     [num_boxes, 4] array of [img_ind, box_0, box_1, rel type].
        Note, the img_inds here start at image_offset
    :param Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)
    :return:
        rois: [num_rois, 5]
        labels: [num_rois] array of labels
        bbox_targets [num_rois, 4] array of targets for the labels.
        rel_labels: [num_rels, 4] (img ind, box0 ind, box1ind, rel type)
    """
    im_inds = rois[:, (0)].long()
    num_im = im_inds[-1] + 1
    fg_rels = gt_rels.clone()
    fg_rels[:, (0)] -= image_offset
    offset = {}
    for i, s, e in enumerate_by_image(im_inds):
        offset[i] = s
    for i, s, e in enumerate_by_image(fg_rels[:, (0)]):
        fg_rels[s:e, 1:3] += offset[i]
    is_cand = im_inds[:, (None)] == im_inds[None]
    is_cand.view(-1)[diagonal_inds(is_cand)] = 0
    is_cand.view(-1)[fg_rels[:, (1)] * im_inds.size(0) + fg_rels[:, (2)]] = 0
    is_bgcand = is_cand.nonzero()
    num_fg = min(fg_rels.size(0), int(RELS_PER_IMG * REL_FG_FRACTION * num_im))
    if num_fg < fg_rels.size(0):
        fg_rels = random_choose(fg_rels, num_fg)
    num_bg = min(is_bgcand.size(0) if is_bgcand.dim() > 0 else 0, int(
        RELS_PER_IMG * num_im) - num_fg)
    if num_bg > 0:
        bg_rels = torch.cat((im_inds[is_bgcand[:, (0)]][:, (None)],
            is_bgcand, (is_bgcand[:, (0), (None)] < -10).long()), 1)
        if num_bg < is_bgcand.size(0):
            bg_rels = random_choose(bg_rels, num_bg)
        rel_labels = torch.cat((fg_rels, bg_rels), 0)
    else:
        rel_labels = fg_rels
    _, perm = torch.sort(rel_labels[:, (0)] * gt_boxes.size(0) ** 2 + 
        rel_labels[:, (1)] * gt_boxes.size(0) + rel_labels[:, (2)])
    rel_labels = rel_labels[perm].contiguous()
    labels = gt_classes[:, (1)].contiguous()
    return rois, labels, rel_labels


class ObjectDetector(nn.Module):
    """
    Core model for doing object detection + getting the visual features. This could be the first step in
    a pipeline. We can provide GT rois or use the RPN (which would then be classification!)
    """
    MODES = 'rpntrain', 'gtbox', 'refinerels', 'proposals'

    def __init__(self, classes, mode='rpntrain', num_gpus=1,
        nms_filter_duplicates=True, max_per_img=64, use_resnet=False,
        thresh=0.05):
        """
        :param classes: Object classes
        :param rel_classes: Relationship classes. None if were not using rel mode
        :param num_gpus: how many GPUS 2 use
        """
        super(ObjectDetector, self).__init__()
        if mode not in self.MODES:
            raise ValueError('invalid mode')
        self.mode = mode
        self.classes = classes
        self.num_gpus = num_gpus
        self.pooling_size = 7
        self.nms_filter_duplicates = nms_filter_duplicates
        self.max_per_img = max_per_img
        self.use_resnet = use_resnet
        self.thresh = thresh
        if not self.use_resnet:
            vgg_model = load_vgg()
            self.features = vgg_model.features
            self.roi_fmap = vgg_model.classifier
            rpn_input_dim = 512
            output_dim = 4096
        else:
            self.features = load_resnet()
            self.compress = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=
                1), nn.ReLU(inplace=True), nn.BatchNorm2d(256))
            self.roi_fmap = nn.Sequential(nn.Linear(256 * 7 * 7, 2048), nn.
                SELU(inplace=True), nn.AlphaDropout(p=0.05), nn.Linear(2048,
                2048), nn.SELU(inplace=True), nn.AlphaDropout(p=0.05))
            rpn_input_dim = 1024
            output_dim = 2048
        self.score_fc = nn.Linear(output_dim, self.num_classes)
        self.bbox_fc = nn.Linear(output_dim, self.num_classes * 4)
        self.rpn_head = RPNHead(dim=512, input_dim=rpn_input_dim)

    @property
    def num_classes(self):
        return len(self.classes)

    def feature_map(self, x):
        """
        Produces feature map from the input image
        :param x: [batch_size, 3, size, size] float32 padded image
        :return: Feature maps at 1/16 the original size.
        Each one is [batch_size, dim, IM_SIZE/k, IM_SIZE/k].
        """
        if not self.use_resnet:
            return self.features(x)
        x = self.features.conv1(x)
        x = self.features.bn1(x)
        x = self.features.relu(x)
        x = self.features.maxpool(x)
        c2 = self.features.layer1(x)
        c3 = self.features.layer2(c2)
        c4 = self.features.layer3(c3)
        return c4

    def obj_feature_map(self, features, rois):
        """
        Gets the ROI features
        :param features: [batch_size, dim, IM_SIZE/4, IM_SIZE/4] (features at level p2)
        :param rois: [num_rois, 5] array of [img_num, x0, y0, x1, y1].
        :return: [num_rois, #dim] array
        """
        feature_pool = RoIAlignFunction(self.pooling_size, self.
            pooling_size, spatial_scale=1 / 16)(self.compress(features) if
            self.use_resnet else features, rois)
        return self.roi_fmap(feature_pool.view(rois.size(0), -1))

    def rpn_boxes(self, fmap, im_sizes, image_offset, gt_boxes=None,
        gt_classes=None, gt_rels=None, train_anchor_inds=None, proposals=None):
        """
        Gets boxes from the RPN
        :param fmap:
        :param im_sizes:
        :param image_offset:
        :param gt_boxes:
        :param gt_classes:
        :param gt_rels:
        :param train_anchor_inds:
        :return:
        """
        rpn_feats = self.rpn_head(fmap)
        rois = self.rpn_head.roi_proposals(rpn_feats, im_sizes, nms_thresh=
            0.7, pre_nms_topn=12000 if self.training and self.mode ==
            'rpntrain' else 6000, post_nms_topn=2000 if self.training and 
            self.mode == 'rpntrain' else 1000)
        if self.training:
            if (gt_boxes is None or gt_classes is None or train_anchor_inds is
                None):
                raise ValueError(
                    'Must supply GT boxes, GT classes, trainanchors when in train mode'
                    )
            rpn_scores, rpn_box_deltas = self.rpn_head.anchor_preds(rpn_feats,
                train_anchor_inds, image_offset)
            if gt_rels is not None and self.mode == 'rpntrain':
                raise ValueError(
                    "Training the object detector and the relationship model with detectionat the same time isn't supported"
                    )
            if self.mode == 'refinerels':
                all_rois = Variable(rois)
                labels = None
                bbox_targets = None
                rel_labels = None
            else:
                all_rois, labels, bbox_targets = proposal_assignments_det(rois,
                    gt_boxes.data, gt_classes.data, image_offset, fg_thresh=0.5
                    )
                rel_labels = None
        else:
            all_rois = Variable(rois, volatile=True)
            labels = None
            bbox_targets = None
            rel_labels = None
            rpn_box_deltas = None
            rpn_scores = None
        return (all_rois, labels, bbox_targets, rpn_scores, rpn_box_deltas,
            rel_labels)

    def gt_boxes(self, fmap, im_sizes, image_offset, gt_boxes=None,
        gt_classes=None, gt_rels=None, train_anchor_inds=None, proposals=None):
        """
        Gets GT boxes!
        :param fmap:
        :param im_sizes:
        :param image_offset:
        :param gt_boxes:
        :param gt_classes:
        :param gt_rels:
        :param train_anchor_inds:
        :return:
        """
        assert gt_boxes is not None
        im_inds = gt_classes[:, (0)] - image_offset
        rois = torch.cat((im_inds.float()[:, (None)], gt_boxes), 1)
        if gt_rels is not None and self.training:
            rois, labels, rel_labels = proposal_assignments_gtbox(rois.data,
                gt_boxes.data, gt_classes.data, gt_rels.data, image_offset,
                fg_thresh=0.5)
        else:
            labels = gt_classes[:, (1)]
            rel_labels = None
        return rois, labels, None, None, None, rel_labels

    def proposal_boxes(self, fmap, im_sizes, image_offset, gt_boxes=None,
        gt_classes=None, gt_rels=None, train_anchor_inds=None, proposals=None):
        """
        Gets boxes from the RPN
        :param fmap:
        :param im_sizes:
        :param image_offset:
        :param gt_boxes:
        :param gt_classes:
        :param gt_rels:
        :param train_anchor_inds:
        :return:
        """
        assert proposals is not None
        rois = filter_roi_proposals(proposals[:, 2:].data.contiguous(),
            proposals[:, (1)].data.contiguous(), np.array([2000] * len(
            im_sizes)), nms_thresh=0.7, pre_nms_topn=12000 if self.training and
            self.mode == 'rpntrain' else 6000, post_nms_topn=2000 if self.
            training and self.mode == 'rpntrain' else 1000)
        if self.training:
            all_rois, labels, bbox_targets = proposal_assignments_det(rois,
                gt_boxes.data, gt_classes.data, image_offset, fg_thresh=0.5)
            all_rois = torch.cat((all_rois, Variable(rois)), 0)
        else:
            all_rois = Variable(rois, volatile=True)
            labels = None
            bbox_targets = None
        rpn_scores = None
        rpn_box_deltas = None
        rel_labels = None
        return (all_rois, labels, bbox_targets, rpn_scores, rpn_box_deltas,
            rel_labels)

    def get_boxes(self, *args, **kwargs):
        if self.mode == 'gtbox':
            fn = self.gt_boxes
        elif self.mode == 'proposals':
            assert kwargs['proposals'] is not None
            fn = self.proposal_boxes
        else:
            fn = self.rpn_boxes
        return fn(*args, **kwargs)

    def forward(self, x, im_sizes, image_offset, gt_boxes=None, gt_classes=
        None, gt_rels=None, proposals=None, train_anchor_inds=None,
        return_fmap=False):
        """
        Forward pass for detection
        :param x: Images@[batch_size, 3, IM_SIZE, IM_SIZE]
        :param im_sizes: A numpy array of (h, w, scale) for each image.
        :param image_offset: Offset onto what image we're on for MGPU training (if single GPU this is 0)
        :param gt_boxes:

        Training parameters:
        :param gt_boxes: [num_gt, 4] GT boxes over the batch.
        :param gt_classes: [num_gt, 2] gt boxes where each one is (img_id, class)
        :param proposals: things
        :param train_anchor_inds: a [num_train, 2] array of indices for the anchors that will
                                  be used to compute the training loss. Each (img_ind, fpn_idx)
        :return: If train:
        """
        fmap = self.feature_map(x)
        (rois, obj_labels, bbox_targets, rpn_scores, rpn_box_deltas, rel_labels
            ) = (self.get_boxes(fmap, im_sizes, image_offset, gt_boxes,
            gt_classes, gt_rels, train_anchor_inds, proposals=proposals))
        obj_fmap = self.obj_feature_map(fmap, rois)
        od_obj_dists = self.score_fc(obj_fmap)
        od_box_deltas = self.bbox_fc(obj_fmap).view(-1, len(self.classes), 4
            ) if self.mode != 'gtbox' else None
        od_box_priors = rois[:, 1:]
        if not self.training and not self.mode == 'gtbox' or self.mode in (
            'proposals', 'refinerels'):
            (nms_inds, nms_scores, nms_preds, nms_boxes_assign, nms_boxes,
                nms_imgs) = (self.nms_boxes(od_obj_dists, rois,
                od_box_deltas, im_sizes))
            im_inds = nms_imgs + image_offset
            obj_dists = od_obj_dists[nms_inds]
            obj_fmap = obj_fmap[nms_inds]
            box_deltas = od_box_deltas[nms_inds]
            box_priors = nms_boxes[:, (0)]
            if self.training and not self.mode == 'gtbox':
                pred_to_gtbox = bbox_overlaps(box_priors, gt_boxes).data
                pred_to_gtbox[im_inds.data[:, (None)] != gt_classes.data[(
                    None), :, (0)]] = 0.0
                max_overlaps, argmax_overlaps = pred_to_gtbox.max(1)
                rm_obj_labels = gt_classes[:, (1)][argmax_overlaps]
                rm_obj_labels[max_overlaps < 0.5] = 0
            else:
                rm_obj_labels = None
        else:
            im_inds = rois[:, (0)].long().contiguous() + image_offset
            nms_scores = None
            nms_preds = None
            nms_boxes_assign = None
            nms_boxes = None
            box_priors = rois[:, 1:]
            rm_obj_labels = obj_labels
            box_deltas = od_box_deltas
            obj_dists = od_obj_dists
        return Result(od_obj_dists=od_obj_dists, rm_obj_dists=obj_dists,
            obj_scores=nms_scores, obj_preds=nms_preds, obj_fmap=obj_fmap,
            od_box_deltas=od_box_deltas, rm_box_deltas=box_deltas,
            od_box_targets=bbox_targets, rm_box_targets=bbox_targets,
            od_box_priors=od_box_priors, rm_box_priors=box_priors,
            boxes_assigned=nms_boxes_assign, boxes_all=nms_boxes,
            od_obj_labels=obj_labels, rm_obj_labels=rm_obj_labels,
            rpn_scores=rpn_scores, rpn_box_deltas=rpn_box_deltas,
            rel_labels=rel_labels, im_inds=im_inds, fmap=fmap if
            return_fmap else None)

    def nms_boxes(self, obj_dists, rois, box_deltas, im_sizes):
        """
        Performs NMS on the boxes
        :param obj_dists: [#rois, #classes]
        :param rois: [#rois, 5]
        :param box_deltas: [#rois, #classes, 4]
        :param im_sizes: sizes of images
        :return
            nms_inds [#nms]
            nms_scores [#nms]
            nms_labels [#nms]
            nms_boxes_assign [#nms, 4]
            nms_boxes  [#nms, #classes, 4]. classid=0 is the box prior.
        """
        boxes = bbox_preds(rois[:, (None), 1:].expand_as(box_deltas).
            contiguous().view(-1, 4), box_deltas.view(-1, 4)).view(*
            box_deltas.size())
        inds = rois[:, (0)].long().contiguous()
        dets = []
        for i, s, e in enumerate_by_image(inds.data):
            h, w = im_sizes[(i), :2]
            boxes[s:e, :, (0)].data.clamp_(min=0, max=w - 1)
            boxes[s:e, :, (1)].data.clamp_(min=0, max=h - 1)
            boxes[s:e, :, (2)].data.clamp_(min=0, max=w - 1)
            boxes[s:e, :, (3)].data.clamp_(min=0, max=h - 1)
            d_filtered = filter_det(F.softmax(obj_dists[s:e], 1), boxes[s:e
                ], start_ind=s, nms_filter_duplicates=self.
                nms_filter_duplicates, max_per_img=self.max_per_img, thresh
                =self.thresh)
            if d_filtered is not None:
                dets.append(d_filtered)
        if len(dets) == 0:
            None
            return None
        nms_inds, nms_scores, nms_labels = [torch.cat(x, 0) for x in zip(*dets)
            ]
        twod_inds = nms_inds * boxes.size(1) + nms_labels.data
        nms_boxes_assign = boxes.view(-1, 4)[twod_inds]
        nms_boxes = torch.cat((rois[:, 1:][nms_inds][:, (None)], boxes[
            nms_inds][:, 1:]), 1)
        return (nms_inds, nms_scores, nms_labels, nms_boxes_assign,
            nms_boxes, inds[nms_inds])

    def __getitem__(self, batch):
        """ Hack to do multi-GPU training"""
        batch.scatter()
        if self.num_gpus == 1:
            return self(*batch[0])
        replicas = nn.parallel.replicate(self, devices=list(range(self.
            num_gpus)))
        outputs = nn.parallel.parallel_apply(replicas, [batch[i] for i in
            range(self.num_gpus)])
        if any([x.is_none() for x in outputs]):
            assert not self.training
            return None
        return gather_res(outputs, 0, dim=0)


ANCHOR_RATIOS = 0.23232838, 0.63365731, 1.28478321, 3.15089189


ANCHOR_SCALES = 2.22152954, 4.12315647, 7.21692515, 12.60263013, 22.7102731


ANCHOR_SIZE = 16


def gather_nd(x, index):
    """

    :param x: n dimensional tensor [x0, x1, x2, ... x{n-1}, dim]
    :param index: [num, n-1] where each row contains the indices we'll use
    :return: [num, dim]
    """
    nd = x.dim() - 1
    assert nd > 0
    assert index.dim() == 2
    assert index.size(1) == nd
    dim = x.size(-1)
    sel_inds = index[:, (nd - 1)].clone()
    mult_factor = x.size(nd - 1)
    for col in range(nd - 2, -1, -1):
        sel_inds += index[:, (col)] * mult_factor
        mult_factor *= x.size(col)
    grouped = x.view(-1, dim)[sel_inds]
    return grouped


IM_SCALE = 592


def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
  Given a vector of widths (ws) and heights (hs) around a center
  (x_ctr, y_ctr), output a set of anchors (windows).
  """
    ws = ws[:, (np.newaxis)]
    hs = hs[:, (np.newaxis)]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1), y_ctr - 0.5 * (hs - 1), 
        x_ctr + 0.5 * (ws - 1), y_ctr + 0.5 * (hs - 1)))
    return anchors


def _whctrs(anchor):
    """
  Return width, height, x center, and y center for an anchor (window).
  """
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr


def _ratio_enum(anchor, ratios):
    """
  Enumerate a set of anchors for each aspect ratio wrt an anchor.
  """
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.sqrt(size_ratios)
    hs = ws * ratios
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def _scale_enum(anchor, scales):
    """
  Enumerate a set of anchors for each scale wrt an anchor.
  """
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def generate_base_anchors(base_size=16, ratios=[0.5, 1, 2], scales=2 ** np.
    arange(3, 6)):
    """
  Generate anchor (reference) windows by enumerating aspect ratios X
  scales wrt a reference (0, 0, 15, 15) window.
  """
    base_anchor = np.array([1, 1, base_size, base_size]) - 1
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    anchors = np.vstack([_scale_enum(ratio_anchors[(i), :], scales) for i in
        range(ratio_anchors.shape[0])])
    return anchors


def generate_anchors(base_size=16, feat_stride=16, anchor_scales=(8, 16, 32
    ), anchor_ratios=(0.5, 1, 2)):
    """ A wrapper function to generate anchors given different scales
    Also return the number of anchors in variable 'length'
  """
    anchors = generate_base_anchors(base_size=base_size, ratios=np.array(
        anchor_ratios), scales=np.array(anchor_scales))
    A = anchors.shape[0]
    shift_x = np.arange(0, IM_SCALE // feat_stride) * feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_x)
    shifts = np.stack([shift_x, shift_y, shift_x, shift_y], -1)
    all_anchors = shifts[:, :, (None)] + anchors[None, None]
    return all_anchors


class RPNHead(nn.Module):
    """
    Serves as the class + box outputs for each level in the FPN.
    """

    def __init__(self, dim=512, input_dim=1024):
        """
        :param aspect_ratios: Aspect ratios for the anchors. NOTE - this can't be changed now
               as it depends on other things in the C code...
        """
        super(RPNHead, self).__init__()
        self.anchor_target_dim = 6
        self.stride = 16
        self.conv = nn.Sequential(nn.Conv2d(input_dim, dim, kernel_size=3,
            padding=1), nn.ReLU6(inplace=True), nn.Conv2d(dim, self.
            anchor_target_dim * self._A, kernel_size=1))
        ans_np = generate_anchors(base_size=ANCHOR_SIZE, feat_stride=self.
            stride, anchor_scales=ANCHOR_SCALES, anchor_ratios=ANCHOR_RATIOS)
        self.register_buffer('anchors', torch.FloatTensor(ans_np))

    @property
    def _A(self):
        return len(ANCHOR_RATIOS) * len(ANCHOR_SCALES)

    def forward(self, fmap):
        """
        Gets the class / noclass predictions over all the scales

        :param fmap: [batch_size, dim, IM_SIZE/16, IM_SIZE/16] featuremap
        :return: [batch_size, IM_SIZE/16, IM_SIZE/16, A, 6]
        """
        rez = self._reshape_channels(self.conv(fmap))
        rez = rez.view(rez.size(0), rez.size(1), rez.size(2), self._A, self
            .anchor_target_dim)
        return rez

    def anchor_preds(self, preds, train_anchor_inds, image_offset):
        """
        Get predictions for the training indices
        :param preds: [batch_size, IM_SIZE/16, IM_SIZE/16, A, 6]
        :param train_anchor_inds: [num_train, 4] indices into the predictions
        :return: class_preds: [num_train, 2] array of yes/no
                 box_preds:   [num_train, 4] array of predicted boxes
        """
        assert train_anchor_inds.size(1) == 4
        tai = train_anchor_inds.data.clone()
        tai[:, (0)] -= image_offset
        train_regions = gather_nd(preds, tai)
        class_preds = train_regions[:, :2]
        box_preds = train_regions[:, 2:]
        return class_preds, box_preds

    @staticmethod
    def _reshape_channels(x):
        """ [batch_size, channels, h, w] -> [batch_size, h, w, channels] """
        assert x.dim() == 4
        batch_size, nc, h, w = x.size()
        x_t = x.view(batch_size, nc, -1).transpose(1, 2).contiguous()
        x_t = x_t.view(batch_size, h, w, nc)
        return x_t

    def roi_proposals(self, fmap, im_sizes, nms_thresh=0.7, pre_nms_topn=
        12000, post_nms_topn=2000):
        """
        :param fmap: [batch_size, IM_SIZE/16, IM_SIZE/16, A, 6]
        :param im_sizes:        [batch_size, 3] numpy array of (h, w, scale)
        :return: ROIS: shape [a <=post_nms_topn, 5] array of ROIS.
        """
        class_fmap = fmap[:, :, :, :, :2].contiguous()
        class_preds = F.softmax(class_fmap, 4)[..., 1].data.contiguous()
        box_fmap = fmap[:, :, :, :, 2:].data.contiguous()
        anchor_stacked = torch.cat([self.anchors[None]] * fmap.size(0), 0)
        box_preds = bbox_preds(anchor_stacked.view(-1, 4), box_fmap.view(-1, 4)
            ).view(*box_fmap.size())
        for i, (h, w, scale) in enumerate(im_sizes):
            h_end = int(h) // self.stride
            w_end = int(w) // self.stride
            if h_end < class_preds.size(1):
                class_preds[(i), h_end:] = -0.01
            if w_end < class_preds.size(2):
                class_preds[(i), :, w_end:] = -0.01
            box_preds[(i), :, :, :, (0)].clamp_(min=0, max=w - 1)
            box_preds[(i), :, :, :, (1)].clamp_(min=0, max=h - 1)
            box_preds[(i), :, :, :, (2)].clamp_(min=0, max=w - 1)
            box_preds[(i), :, :, :, (3)].clamp_(min=0, max=h - 1)
        sizes = center_size(box_preds.view(-1, 4))
        class_preds.view(-1)[(sizes[:, (2)] < 4) | (sizes[:, (3)] < 4)] = -0.01
        return filter_roi_proposals(box_preds.view(-1, 4), class_preds.view
            (-1), boxes_per_im=np.array([np.prod(box_preds.size()[1:-1])] *
            fmap.size(0)), nms_thresh=nms_thresh, pre_nms_topn=pre_nms_topn,
            post_nms_topn=post_nms_topn)


class Flattener(nn.Module):

    def __init__(self):
        """
        Flattens last 3 dimensions to make it only batch size, -1
        """
        super(Flattener, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


MODES = 'sgdet', 'sgcls', 'predcls'


def transpose_packed_sequence_inds(lengths):
    """
    Goes from a TxB packed sequence to a BxT or vice versa. Assumes that nothing is a variable
    :param ps: PackedSequence
    :return:
    """
    new_inds = []
    new_lens = []
    cum_add = np.cumsum([0] + lengths)
    max_len = lengths[0]
    length_pointer = len(lengths) - 1
    for i in range(max_len):
        while length_pointer > 0 and lengths[length_pointer] <= i:
            length_pointer -= 1
        new_inds.append(cum_add[:length_pointer + 1].copy())
        cum_add[:length_pointer + 1] += 1
        new_lens.append(length_pointer + 1)
    new_inds = np.concatenate(new_inds, 0)
    return new_inds, new_lens


def _sort_by_score(im_inds, scores):
    """
    We'll sort everything scorewise from Hi->low, BUT we need to keep images together
    and sort LSTM from l
    :param im_inds: Which im we're on
    :param scores: Goodness ranging between [0, 1]. Higher numbers come FIRST
    :return: Permutation to put everything in the right order for the LSTM
             Inverse permutation
             Lengths for the TxB packed sequence.
    """
    num_im = im_inds[-1] + 1
    rois_per_image = scores.new(num_im)
    lengths = []
    for i, s, e in enumerate_by_image(im_inds):
        rois_per_image[i] = 2 * (s - e) * num_im + i
        lengths.append(e - s)
    lengths = sorted(lengths, reverse=True)
    inds, ls_transposed = transpose_packed_sequence_inds(lengths)
    inds = torch.LongTensor(inds).cuda(im_inds.get_device())
    roi_order = scores - 2 * rois_per_image[im_inds]
    _, perm = torch.sort(roi_order, 0, descending=True)
    perm = perm[inds]
    _, inv_perm = torch.sort(perm)
    return perm, inv_perm, ls_transposed


def arange(base_tensor, n=None):
    new_size = base_tensor.size(0) if n is None else n
    new_vec = base_tensor.new(new_size).long()
    torch.arange(0, new_size, out=new_vec)
    return new_vec


def to_onehot(vec, num_classes, fill=1000):
    """
    Creates a [size, num_classes] torch FloatTensor where
    one_hot[i, vec[i]] = fill
    
    :param vec: 1d torch tensor
    :param num_classes: int
    :param fill: value that we want + and - things to be.
    :return: 
    """
    onehot_result = vec.new(vec.size(0), num_classes).float().fill_(-fill)
    arange_inds = vec.new(vec.size(0)).long()
    torch.arange(0, vec.size(0), out=arange_inds)
    onehot_result.view(-1)[vec + num_classes * arange_inds] = fill
    return onehot_result


class LinearizedContext(nn.Module):
    """
    Module for computing the object contexts and edge contexts
    """

    def __init__(self, classes, rel_classes, mode='sgdet', embed_dim=200,
        hidden_dim=256, obj_dim=2048, nl_obj=2, nl_edge=2, dropout_rate=0.2,
        order='confidence', pass_in_obj_feats_to_decoder=True,
        pass_in_obj_feats_to_edge=True):
        super(LinearizedContext, self).__init__()
        self.classes = classes
        self.rel_classes = rel_classes
        assert mode in MODES
        self.mode = mode
        self.nl_obj = nl_obj
        self.nl_edge = nl_edge
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.obj_dim = obj_dim
        self.dropout_rate = dropout_rate
        self.pass_in_obj_feats_to_decoder = pass_in_obj_feats_to_decoder
        self.pass_in_obj_feats_to_edge = pass_in_obj_feats_to_edge
        assert order in ('size', 'confidence', 'random', 'leftright')
        self.order = order
        embed_vecs = obj_edge_vectors(self.classes, wv_dim=self.embed_dim)
        self.obj_embed = nn.Embedding(self.num_classes, self.embed_dim)
        self.obj_embed.weight.data = embed_vecs.clone()
        self.obj_embed2 = nn.Embedding(self.num_classes, self.embed_dim)
        self.obj_embed2.weight.data = embed_vecs.clone()
        self.pos_embed = nn.Sequential(*[nn.BatchNorm1d(4, momentum=
            BATCHNORM_MOMENTUM / 10.0), nn.Linear(4, 128), nn.ReLU(inplace=
            True), nn.Dropout(0.1)])
        if self.nl_obj > 0:
            self.obj_ctx_rnn = AlternatingHighwayLSTM(input_size=self.
                obj_dim + self.embed_dim + 128, hidden_size=self.hidden_dim,
                num_layers=self.nl_obj, recurrent_dropout_probability=
                dropout_rate)
            decoder_inputs_dim = self.hidden_dim
            if self.pass_in_obj_feats_to_decoder:
                decoder_inputs_dim += self.obj_dim + self.embed_dim
            self.decoder_rnn = DecoderRNN(self.classes, embed_dim=self.
                embed_dim, inputs_dim=decoder_inputs_dim, hidden_dim=self.
                hidden_dim, recurrent_dropout_probability=dropout_rate)
        else:
            self.decoder_lin = nn.Linear(self.obj_dim + self.embed_dim + 
                128, self.num_classes)
        if self.nl_edge > 0:
            input_dim = self.embed_dim
            if self.nl_obj > 0:
                input_dim += self.hidden_dim
            if self.pass_in_obj_feats_to_edge:
                input_dim += self.obj_dim
            self.edge_ctx_rnn = AlternatingHighwayLSTM(input_size=input_dim,
                hidden_size=self.hidden_dim, num_layers=self.nl_edge,
                recurrent_dropout_probability=dropout_rate)

    def sort_rois(self, batch_idx, confidence, box_priors):
        """
        :param batch_idx: tensor with what index we're on
        :param confidence: tensor with confidences between [0,1)
        :param boxes: tensor with (x1, y1, x2, y2)
        :return: Permutation, inverse permutation, and the lengths transposed (same as _sort_by_score)
        """
        cxcywh = center_size(box_priors)
        if self.order == 'size':
            sizes = cxcywh[:, (2)] * cxcywh[:, (3)]
            assert sizes.min() > 0.0
            scores = sizes / (sizes.max() + 1)
        elif self.order == 'confidence':
            scores = confidence
        elif self.order == 'random':
            scores = torch.FloatTensor(np.random.rand(batch_idx.size(0)))
        elif self.order == 'leftright':
            centers = cxcywh[:, (0)]
            scores = centers / (centers.max() + 1)
        else:
            raise ValueError('invalid mode {}'.format(self.order))
        return _sort_by_score(batch_idx, scores)

    @property
    def num_classes(self):
        return len(self.classes)

    @property
    def num_rels(self):
        return len(self.rel_classes)

    def edge_ctx(self, obj_feats, obj_dists, im_inds, obj_preds, box_priors
        =None):
        """
        Object context and object classification.
        :param obj_feats: [num_obj, img_dim + object embedding0 dim]
        :param obj_dists: [num_obj, #classes]
        :param im_inds: [num_obj] the indices of the images
        :return: edge_ctx: [num_obj, #feats] For later!
        """
        obj_embed2 = self.obj_embed2(obj_preds)
        inp_feats = torch.cat((obj_embed2, obj_feats), 1)
        confidence = F.softmax(obj_dists, dim=1).data.view(-1)[obj_preds.
            data + arange(obj_preds.data) * self.num_classes]
        perm, inv_perm, ls_transposed = self.sort_rois(im_inds.data,
            confidence, box_priors)
        edge_input_packed = PackedSequence(inp_feats[perm], ls_transposed)
        edge_reps = self.edge_ctx_rnn(edge_input_packed)[0][0]
        edge_ctx = edge_reps[inv_perm]
        return edge_ctx

    def obj_ctx(self, obj_feats, obj_dists, im_inds, obj_labels=None,
        box_priors=None, boxes_per_cls=None):
        """
        Object context and object classification.
        :param obj_feats: [num_obj, img_dim + object embedding0 dim]
        :param obj_dists: [num_obj, #classes]
        :param im_inds: [num_obj] the indices of the images
        :param obj_labels: [num_obj] the GT labels of the image
        :param boxes: [num_obj, 4] boxes. We'll use this for NMS
        :return: obj_dists: [num_obj, #classes] new probability distribution.
                 obj_preds: argmax of that distribution.
                 obj_final_ctx: [num_obj, #feats] For later!
        """
        confidence = F.softmax(obj_dists, dim=1).data[:, 1:].max(1)[0]
        perm, inv_perm, ls_transposed = self.sort_rois(im_inds.data,
            confidence, box_priors)
        obj_inp_rep = obj_feats[perm].contiguous()
        input_packed = PackedSequence(obj_inp_rep, ls_transposed)
        encoder_rep = self.obj_ctx_rnn(input_packed)[0][0]
        if self.mode != 'predcls':
            decoder_inp = PackedSequence(torch.cat((obj_inp_rep,
                encoder_rep), 1) if self.pass_in_obj_feats_to_decoder else
                encoder_rep, ls_transposed)
            obj_dists, obj_preds = self.decoder_rnn(decoder_inp, labels=
                obj_labels[perm] if obj_labels is not None else None,
                boxes_for_nms=boxes_per_cls[perm] if boxes_per_cls is not
                None else None)
            obj_preds = obj_preds[inv_perm]
            obj_dists = obj_dists[inv_perm]
        else:
            assert obj_labels is not None
            obj_preds = obj_labels
            obj_dists = Variable(to_onehot(obj_preds.data, self.num_classes))
        encoder_rep = encoder_rep[inv_perm]
        return obj_dists, obj_preds, encoder_rep

    def forward(self, obj_fmaps, obj_logits, im_inds, obj_labels=None,
        box_priors=None, boxes_per_cls=None):
        """
        Forward pass through the object and edge context
        :param obj_priors:
        :param obj_fmaps:
        :param im_inds:
        :param obj_labels:
        :param boxes:
        :return:
        """
        obj_embed = F.softmax(obj_logits, dim=1) @ self.obj_embed.weight
        pos_embed = self.pos_embed(Variable(center_size(box_priors)))
        obj_pre_rep = torch.cat((obj_fmaps, obj_embed, pos_embed), 1)
        if self.nl_obj > 0:
            obj_dists2, obj_preds, obj_ctx = self.obj_ctx(obj_pre_rep,
                obj_logits, im_inds, obj_labels, box_priors, boxes_per_cls)
        else:
            if self.mode == 'predcls':
                obj_dists2 = Variable(to_onehot(obj_labels.data, self.
                    num_classes))
            else:
                obj_dists2 = self.decoder_lin(obj_pre_rep)
            if self.mode == 'sgdet' and not self.training:
                probs = F.softmax(obj_dists2, 1)
                nms_mask = obj_dists2.data.clone()
                nms_mask.zero_()
                for c_i in range(1, obj_dists2.size(1)):
                    scores_ci = probs.data[:, (c_i)]
                    boxes_ci = boxes_per_cls.data[:, (c_i)]
                    keep = apply_nms(scores_ci, boxes_ci, pre_nms_topn=
                        scores_ci.size(0), post_nms_topn=scores_ci.size(0),
                        nms_thresh=0.3)
                    nms_mask[:, (c_i)][keep] = 1
                obj_preds = Variable(nms_mask * probs.data, volatile=True)[:,
                    1:].max(1)[1] + 1
            else:
                obj_preds = (obj_labels if obj_labels is not None else 
                    obj_dists2[:, 1:].max(1)[1] + 1)
            obj_ctx = obj_pre_rep
        edge_ctx = None
        if self.nl_edge > 0:
            edge_ctx = self.edge_ctx(torch.cat((obj_fmaps, obj_ctx), 1) if
                self.pass_in_obj_feats_to_edge else obj_ctx, obj_dists=
                obj_dists2.detach(), im_inds=im_inds, obj_preds=obj_preds,
                box_priors=box_priors)
        return obj_dists2, obj_preds, edge_ctx


def filter_dets(boxes, obj_scores, obj_classes, rel_inds, pred_scores):
    """
    Filters detections....
    :param boxes: [num_box, topk, 4] if bbox regression else [num_box, 4]
    :param obj_scores: [num_box] probabilities for the scores
    :param obj_classes: [num_box] class labels for the topk
    :param rel_inds: [num_rel, 2] TENSOR consisting of (im_ind0, im_ind1)
    :param pred_scores: [topk, topk, num_rel, num_predicates]
    :param use_nms: True if use NMS to filter dets.
    :return: boxes, objs, rels, pred_scores

    """
    if boxes.dim() != 2:
        raise ValueError('Boxes needs to be [num_box, 4] but its {}'.format
            (boxes.size()))
    num_box = boxes.size(0)
    assert obj_scores.size(0) == num_box
    assert obj_classes.size() == obj_scores.size()
    num_rel = rel_inds.size(0)
    assert rel_inds.size(1) == 2
    assert pred_scores.size(0) == num_rel
    obj_scores0 = obj_scores.data[rel_inds[:, (0)]]
    obj_scores1 = obj_scores.data[rel_inds[:, (1)]]
    pred_scores_max, pred_classes_argmax = pred_scores.data[:, 1:].max(1)
    pred_classes_argmax = pred_classes_argmax + 1
    rel_scores_argmaxed = pred_scores_max * obj_scores0 * obj_scores1
    rel_scores_vs, rel_scores_idx = torch.sort(rel_scores_argmaxed.view(-1),
        dim=0, descending=True)
    rels = rel_inds[rel_scores_idx].cpu().numpy()
    pred_scores_sorted = pred_scores[rel_scores_idx].data.cpu().numpy()
    obj_scores_np = obj_scores.data.cpu().numpy()
    objs_np = obj_classes.data.cpu().numpy()
    boxes_out = boxes.data.cpu().numpy()
    return boxes_out, objs_np, obj_scores_np, rels, pred_scores_sorted


@to_variable
def rel_assignments(im_inds, rpn_rois, roi_gtlabels, gt_boxes, gt_classes,
    gt_rels, image_offset, fg_thresh=0.5, num_sample_per_gt=4,
    filter_non_overlap=True):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    :param rpn_rois: [img_ind, x1, y1, x2, y2]
    :param gt_boxes:   [num_boxes, 4] array of x0, y0, x1, y1]
    :param gt_classes: [num_boxes, 2] array of [img_ind, class]
    :param gt_rels     [num_boxes, 4] array of [img_ind, box_0, box_1, rel type]
    :param Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)
    :return:
        rois: [num_rois, 5]
        labels: [num_rois] array of labels
        bbox_targets [num_rois, 4] array of targets for the labels.
        rel_labels: [num_rels, 4] (img ind, box0 ind, box1ind, rel type)
    """
    fg_rels_per_image = int(np.round(REL_FG_FRACTION * 64))
    pred_inds_np = im_inds.cpu().numpy()
    pred_boxes_np = rpn_rois.cpu().numpy()
    pred_boxlabels_np = roi_gtlabels.cpu().numpy()
    gt_boxes_np = gt_boxes.cpu().numpy()
    gt_classes_np = gt_classes.cpu().numpy()
    gt_rels_np = gt_rels.cpu().numpy()
    gt_classes_np[:, (0)] -= image_offset
    gt_rels_np[:, (0)] -= image_offset
    num_im = gt_classes_np[:, (0)].max() + 1
    rel_labels = []
    num_box_seen = 0
    for im_ind in range(num_im):
        pred_ind = np.where(pred_inds_np == im_ind)[0]
        gt_ind = np.where(gt_classes_np[:, (0)] == im_ind)[0]
        gt_boxes_i = gt_boxes_np[gt_ind]
        gt_classes_i = gt_classes_np[gt_ind, 1]
        gt_rels_i = gt_rels_np[(gt_rels_np[:, (0)] == im_ind), 1:]
        pred_boxes_i = pred_boxes_np[pred_ind]
        pred_boxlabels_i = pred_boxlabels_np[pred_ind]
        ious = bbox_overlaps(pred_boxes_i, gt_boxes_i)
        is_match = (pred_boxlabels_i[:, (None)] == gt_classes_i[None]) & (ious
             >= fg_thresh)
        pbi_iou = bbox_overlaps(pred_boxes_i, pred_boxes_i)
        if filter_non_overlap:
            rel_possibilities = (pbi_iou < 1) & (pbi_iou > 0)
            rels_intersect = rel_possibilities
        else:
            rel_possibilities = np.ones((pred_boxes_i.shape[0],
                pred_boxes_i.shape[0]), dtype=np.int64) - np.eye(pred_boxes_i
                .shape[0], dtype=np.int64)
            rels_intersect = (pbi_iou < 1) & (pbi_iou > 0)
        rel_possibilities[pred_boxlabels_i == 0] = 0
        rel_possibilities[:, (pred_boxlabels_i == 0)] = 0
        fg_rels = []
        p_size = []
        for i, (from_gtind, to_gtind, rel_id) in enumerate(gt_rels_i):
            fg_rels_i = []
            fg_scores_i = []
            for from_ind in np.where(is_match[:, (from_gtind)])[0]:
                for to_ind in np.where(is_match[:, (to_gtind)])[0]:
                    if from_ind != to_ind:
                        fg_rels_i.append((from_ind, to_ind, rel_id))
                        fg_scores_i.append(ious[from_ind, from_gtind] *
                            ious[to_ind, to_gtind])
                        rel_possibilities[from_ind, to_ind] = 0
            if len(fg_rels_i) == 0:
                continue
            p = np.array(fg_scores_i)
            p = p / p.sum()
            p_size.append(p.shape[0])
            num_to_add = min(p.shape[0], num_sample_per_gt)
            for rel_to_add in npr.choice(p.shape[0], p=p, size=num_to_add,
                replace=False):
                fg_rels.append(fg_rels_i[rel_to_add])
        fg_rels = np.array(fg_rels, dtype=np.int64)
        if fg_rels.size > 0 and fg_rels.shape[0] > fg_rels_per_image:
            fg_rels = fg_rels[npr.choice(fg_rels.shape[0], size=
                fg_rels_per_image, replace=False)]
        elif fg_rels.size == 0:
            fg_rels = np.zeros((0, 3), dtype=np.int64)
        bg_rels = np.column_stack(np.where(rel_possibilities))
        bg_rels = np.column_stack((bg_rels, np.zeros(bg_rels.shape[0],
            dtype=np.int64)))
        num_bg_rel = min(64 - fg_rels.shape[0], bg_rels.shape[0])
        if bg_rels.size > 0:
            bg_rels = bg_rels[np.random.choice(bg_rels.shape[0], size=
                num_bg_rel, replace=False)]
        else:
            bg_rels = np.zeros((0, 3), dtype=np.int64)
        if fg_rels.size == 0 and bg_rels.size == 0:
            bg_rels = np.array([[0, 0, 0]], dtype=np.int64)
        all_rels_i = np.concatenate((fg_rels, bg_rels), 0)
        all_rels_i[:, 0:2] += num_box_seen
        all_rels_i = all_rels_i[np.lexsort((all_rels_i[:, (1)], all_rels_i[
            :, (0)]))]
        rel_labels.append(np.column_stack((im_ind * np.ones(all_rels_i.
            shape[0], dtype=np.int64), all_rels_i)))
        num_box_seen += pred_boxes_i.shape[0]
    rel_labels = torch.LongTensor(np.concatenate(rel_labels, 0)).cuda(rpn_rois
        .get_device(), non_blocking=True)
    return rel_labels


def resnet_l4(relu_end=True):
    model = resnet101(pretrained=True)
    l4 = model.layer4
    if not relu_end:
        l4[-1].relu_end = False
    l4[0].conv2.stride = 1, 1
    l4[0].downsample[0].stride = 1, 1
    return l4


class RelModel(nn.Module):
    """
    RELATIONSHIPS
    """

    def __init__(self, classes, rel_classes, mode='sgdet', num_gpus=1,
        use_vision=True, require_overlap_det=True, embed_dim=200,
        hidden_dim=256, pooling_dim=2048, nl_obj=1, nl_edge=2, use_resnet=
        False, order='confidence', thresh=0.01, use_proposals=False,
        pass_in_obj_feats_to_decoder=True, pass_in_obj_feats_to_edge=True,
        rec_dropout=0.0, use_bias=True, use_tanh=True, limit_vision=True):
        """
        :param classes: Object classes
        :param rel_classes: Relationship classes. None if were not using rel mode
        :param mode: (sgcls, predcls, or sgdet)
        :param num_gpus: how many GPUS 2 use
        :param use_vision: Whether to use vision in the final product
        :param require_overlap_det: Whether two objects must intersect
        :param embed_dim: Dimension for all embeddings
        :param hidden_dim: LSTM hidden size
        :param obj_dim:
        """
        super(RelModel, self).__init__()
        self.classes = classes
        self.rel_classes = rel_classes
        self.num_gpus = num_gpus
        assert mode in MODES
        self.mode = mode
        self.pooling_size = 7
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.obj_dim = 2048 if use_resnet else 4096
        self.pooling_dim = pooling_dim
        self.use_bias = use_bias
        self.use_vision = use_vision
        self.use_tanh = use_tanh
        self.limit_vision = limit_vision
        self.require_overlap = require_overlap_det and self.mode == 'sgdet'
        self.detector = ObjectDetector(classes=classes, mode=('proposals' if
            use_proposals else 'refinerels') if mode == 'sgdet' else
            'gtbox', use_resnet=use_resnet, thresh=thresh, max_per_img=64)
        self.context = LinearizedContext(self.classes, self.rel_classes,
            mode=self.mode, embed_dim=self.embed_dim, hidden_dim=self.
            hidden_dim, obj_dim=self.obj_dim, nl_obj=nl_obj, nl_edge=
            nl_edge, dropout_rate=rec_dropout, order=order,
            pass_in_obj_feats_to_decoder=pass_in_obj_feats_to_decoder,
            pass_in_obj_feats_to_edge=pass_in_obj_feats_to_edge)
        self.union_boxes = UnionBoxesAndFeats(pooling_size=self.
            pooling_size, stride=16, dim=1024 if use_resnet else 512)
        if use_resnet:
            self.roi_fmap = nn.Sequential(resnet_l4(relu_end=False), nn.
                AvgPool2d(self.pooling_size), Flattener())
        else:
            roi_fmap = [Flattener(), load_vgg(use_dropout=False, use_relu=
                False, use_linear=pooling_dim == 4096, pretrained=False).
                classifier]
            if pooling_dim != 4096:
                roi_fmap.append(nn.Linear(4096, pooling_dim))
            self.roi_fmap = nn.Sequential(*roi_fmap)
            self.roi_fmap_obj = load_vgg(pretrained=False).classifier
        self.post_lstm = nn.Linear(self.hidden_dim, self.pooling_dim * 2)
        self.post_lstm.weight.data.normal_(0, 10.0 * math.sqrt(1.0 / self.
            hidden_dim))
        self.post_lstm.bias.data.zero_()
        if nl_edge == 0:
            self.post_emb = nn.Embedding(self.num_classes, self.pooling_dim * 2
                )
            self.post_emb.weight.data.normal_(0, math.sqrt(1.0))
        self.rel_compress = nn.Linear(self.pooling_dim, self.num_rels, bias
            =True)
        self.rel_compress.weight = torch.nn.init.xavier_normal(self.
            rel_compress.weight, gain=1.0)
        if self.use_bias:
            self.freq_bias = FrequencyBias()

    @property
    def num_classes(self):
        return len(self.classes)

    @property
    def num_rels(self):
        return len(self.rel_classes)

    def visual_rep(self, features, rois, pair_inds):
        """
        Classify the features
        :param features: [batch_size, dim, IM_SIZE/4, IM_SIZE/4]
        :param rois: [num_rois, 5] array of [img_num, x0, y0, x1, y1].
        :param pair_inds inds to use when predicting
        :return: score_pred, a [num_rois, num_classes] array
                 box_pred, a [num_rois, num_classes, 4] array
        """
        assert pair_inds.size(1) == 2
        uboxes = self.union_boxes(features, rois, pair_inds)
        return self.roi_fmap(uboxes)

    def get_rel_inds(self, rel_labels, im_inds, box_priors):
        if self.training:
            rel_inds = rel_labels[:, :3].data.clone()
        else:
            rel_cands = im_inds.data[:, (None)] == im_inds.data[None]
            rel_cands.view(-1)[diagonal_inds(rel_cands)] = 0
            if self.require_overlap:
                rel_cands = rel_cands & (bbox_overlaps(box_priors.data,
                    box_priors.data) > 0)
                amt_to_add = 100 - rel_cands.long().sum()
            rel_cands = rel_cands.nonzero()
            if rel_cands.dim() == 0:
                rel_cands = im_inds.data.new(1, 2).fill_(0)
            rel_inds = torch.cat((im_inds.data[rel_cands[:, (0)]][:, (None)
                ], rel_cands), 1)
        return rel_inds

    def obj_feature_map(self, features, rois):
        """
        Gets the ROI features
        :param features: [batch_size, dim, IM_SIZE/4, IM_SIZE/4] (features at level p2)
        :param rois: [num_rois, 5] array of [img_num, x0, y0, x1, y1].
        :return: [num_rois, #dim] array
        """
        feature_pool = RoIAlignFunction(self.pooling_size, self.
            pooling_size, spatial_scale=1 / 16)(features, rois)
        return self.roi_fmap_obj(feature_pool.view(rois.size(0), -1))

    def forward(self, x, im_sizes, image_offset, gt_boxes=None, gt_classes=
        None, gt_rels=None, proposals=None, train_anchor_inds=None,
        return_fmap=False):
        """
        Forward pass for detection
        :param x: Images@[batch_size, 3, IM_SIZE, IM_SIZE]
        :param im_sizes: A numpy array of (h, w, scale) for each image.
        :param image_offset: Offset onto what image we're on for MGPU training (if single GPU this is 0)
        :param gt_boxes:

        Training parameters:
        :param gt_boxes: [num_gt, 4] GT boxes over the batch.
        :param gt_classes: [num_gt, 2] gt boxes where each one is (img_id, class)
        :param train_anchor_inds: a [num_train, 2] array of indices for the anchors that will
                                  be used to compute the training loss. Each (img_ind, fpn_idx)
        :return: If train:
            scores, boxdeltas, labels, boxes, boxtargets, rpnscores, rpnboxes, rellabels
            
            if test:
            prob dists, boxes, img inds, maxscores, classes
            
        """
        result = self.detector(x, im_sizes, image_offset, gt_boxes,
            gt_classes, gt_rels, proposals, train_anchor_inds, return_fmap=True
            )
        if result.is_none():
            return ValueError('heck')
        im_inds = result.im_inds - image_offset
        boxes = result.rm_box_priors
        if self.training and result.rel_labels is None:
            assert self.mode == 'sgdet'
            result.rel_labels = rel_assignments(im_inds.data, boxes.data,
                result.rm_obj_labels.data, gt_boxes.data, gt_classes.data,
                gt_rels.data, image_offset, filter_non_overlap=True,
                num_sample_per_gt=1)
        rel_inds = self.get_rel_inds(result.rel_labels, im_inds, boxes)
        rois = torch.cat((im_inds[:, (None)].float(), boxes), 1)
        result.obj_fmap = self.obj_feature_map(result.fmap.detach(), rois)
        result.rm_obj_dists, result.obj_preds, edge_ctx = self.context(result
            .obj_fmap, result.rm_obj_dists.detach(), im_inds, result.
            rm_obj_labels if self.training or self.mode == 'predcls' else
            None, boxes.data, result.boxes_all)
        if edge_ctx is None:
            edge_rep = self.post_emb(result.obj_preds)
        else:
            edge_rep = self.post_lstm(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.pooling_dim)
        subj_rep = edge_rep[:, (0)]
        obj_rep = edge_rep[:, (1)]
        prod_rep = subj_rep[rel_inds[:, (1)]] * obj_rep[rel_inds[:, (2)]]
        if self.use_vision:
            vr = self.visual_rep(result.fmap.detach(), rois, rel_inds[:, 1:])
            if self.limit_vision:
                prod_rep = torch.cat((prod_rep[:, :2048] * vr[:, :2048],
                    prod_rep[:, 2048:]), 1)
            else:
                prod_rep = prod_rep * vr
        if self.use_tanh:
            prod_rep = F.tanh(prod_rep)
        result.rel_dists = self.rel_compress(prod_rep)
        if self.use_bias:
            result.rel_dists = (result.rel_dists + self.freq_bias.
                index_with_labels(torch.stack((result.obj_preds[rel_inds[:,
                (1)]], result.obj_preds[rel_inds[:, (2)]]), 1)))
        if self.training:
            return result
        twod_inds = arange(result.obj_preds.data
            ) * self.num_classes + result.obj_preds.data
        result.obj_scores = F.softmax(result.rm_obj_dists, dim=1).view(-1)[
            twod_inds]
        if self.mode == 'sgdet':
            bboxes = result.boxes_all.view(-1, 4)[twod_inds].view(result.
                boxes_all.size(0), 4)
        else:
            bboxes = result.rm_box_priors
        rel_rep = F.softmax(result.rel_dists, dim=1)
        return filter_dets(bboxes, result.obj_scores, result.obj_preds,
            rel_inds[:, 1:], rel_rep)

    def __getitem__(self, batch):
        """ Hack to do multi-GPU training"""
        batch.scatter()
        if self.num_gpus == 1:
            return self(*batch[0])
        replicas = nn.parallel.replicate(self, devices=list(range(self.
            num_gpus)))
        outputs = nn.parallel.parallel_apply(replicas, [batch[i] for i in
            range(self.num_gpus)])
        if self.training:
            return gather_res(outputs, 0, dim=0)
        return outputs


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
        relu_end=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BATCHNORM_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BATCHNORM_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, momentum=BATCHNORM_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.relu_end = relu_end

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        if self.relu_end:
            out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BATCHNORM_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=
                BATCHNORM_MOMENTUM))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


BOX_SCALE = 1024


def stanford_path(fn):
    return os.path.join(DATA_PATH, 'stanford_filtered', fn)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_rowanz_neural_motifs(_paritybench_base):
    pass
    def test_000(self):
        self._check(Flattener(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(RPNHead(*[], **{}), [torch.rand([4, 1024, 64, 64])], {})

