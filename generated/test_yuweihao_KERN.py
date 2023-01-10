import sys
_module = sys.modules[__name__]
del sys
config = _module
dataloaders = _module
blob = _module
image_transforms = _module
movies = _module
visual_genome = _module
draw_figures = _module
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
roi_align = _module
build = _module
functions = _module
roi_align = _module
modules = _module
roi_align = _module
get_union_boxes = _module
ggnn = _module
kern_model = _module
object_detector = _module
pytorch_misc = _module
resnet = _module
surgery = _module
eval = _module
eval_rels = _module
train_detector = _module
train_rels = _module
generate_knowledge = _module
visualize_sgcls = _module

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


import numpy as np


import torch


from torch.autograd import Variable


import torch.utils.data


import torchvision.transforms


from torch.utils.data import Dataset


from torchvision.transforms import Resize


from torchvision.transforms import Compose


from torchvision.transforms import ToTensor


from torchvision.transforms import Normalize


from collections import defaultdict


from torch.nn import functional as F


import numpy.random as npr


from torch.autograd import Function


from torch.nn.modules.module import Module


from torch.nn.functional import avg_pool2d


from torch.nn.functional import max_pool2d


from torch import nn


import torch.nn as nn


import torch.nn.parallel


from torch.nn.utils.rnn import PackedSequence


from torchvision.models.vgg import vgg16


from torchvision.models.resnet import resnet101


from torch.nn.parallel._functions import Gather


from itertools import tee


import math


import torch.utils.model_zoo as model_zoo


from torchvision.models.resnet import model_urls


from torchvision.models.resnet import conv3x3


from torchvision.models.resnet import BasicBlock


from time import time


import matplotlib.pyplot as plt


from torch import optim


import pandas as pd


import time


import torch.backends.cudnn as cudnn


from torch.optim.lr_scheduler import ReduceLROnPlateau


import matplotlib.image as mpimg


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
        rois_normalized[:, 1] /= width
        rois_normalized[:, 2] /= height
        rois_normalized[:, 3] /= width
        rois_normalized[:, 4] /= height
        num_rois = rois.size(0)
        output = features.new(num_rois, num_channels, self.aligned_height, self.aligned_width).zero_()
        if features.is_cuda:
            res = roi_align.roi_align_forward_cuda(self.aligned_height, self.aligned_width, self.spatial_scale, features, rois_normalized, output)
            assert res == 1
        else:
            raise ValueError
        return output

    def backward(self, grad_output):
        rois = self.saved_tensors[0]
        rois_normalized = rois.clone()
        batch_size, num_channels, data_height, data_width = self.feature_size
        height = (data_height - 1) / self.spatial_scale
        width = (data_width - 1) / self.spatial_scale
        rois_normalized[:, 1] /= width
        rois_normalized[:, 2] /= height
        rois_normalized[:, 3] /= width
        rois_normalized[:, 4] /= height
        grad_input = rois_normalized.new(batch_size, num_channels, data_height, data_width).zero_()
        res = roi_align.roi_align_backward_cuda(self.aligned_height, self.aligned_width, self.spatial_scale, grad_output, rois_normalized, grad_input)
        assert res == 1
        return grad_input, None


class RoIAlign(Module):

    def __init__(self, aligned_height, aligned_width, spatial_scale):
        super(RoIAlign, self).__init__()
        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        return RoIAlignFunction(self.aligned_height, self.aligned_width, self.spatial_scale)(features, rois)


class RoIAlignAvg(Module):

    def __init__(self, aligned_height, aligned_width, spatial_scale):
        super(RoIAlignAvg, self).__init__()
        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        x = RoIAlignFunction(self.aligned_height + 1, self.aligned_width + 1, self.spatial_scale)(features, rois)
        return avg_pool2d(x, kernel_size=2, stride=1)


class RoIAlignMax(Module):

    def __init__(self, aligned_height, aligned_width, spatial_scale):
        super(RoIAlignMax, self).__init__()
        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        x = RoIAlignFunction(self.aligned_height + 1, self.aligned_width + 1, self.spatial_scale)(features, rois)
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
    im_inds = rois[:, 0][union_inds[:, 0]]
    assert (im_inds.data == rois.data[:, 0][union_inds[:, 1]]).sum() == union_inds.size(0)
    union_rois = torch.cat((im_inds[:, None], torch.min(rois[:, 1:3][union_inds[:, 0]], rois[:, 1:3][union_inds[:, 1]]), torch.max(rois[:, 3:5][union_inds[:, 0]], rois[:, 3:5][union_inds[:, 1]])), 1)
    union_pools = RoIAlignFunction(pooling_size, pooling_size, spatial_scale=1 / stride)(fmap, union_rois)
    return union_pools


class UnionBoxesAndFeats(Module):

    def __init__(self, pooling_size=7, stride=16, dim=256, concat=False, use_feats=True):
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
        self.conv = nn.Sequential(nn.Conv2d(2, dim // 2, kernel_size=7, stride=2, padding=3, bias=True), nn.ReLU(inplace=True), nn.BatchNorm2d(dim // 2, momentum=BATCHNORM_MOMENTUM), nn.MaxPool2d(kernel_size=3, stride=2, padding=1), nn.Conv2d(dim // 2, dim, kernel_size=3, stride=1, padding=1, bias=True), nn.ReLU(inplace=True), nn.BatchNorm2d(dim, momentum=BATCHNORM_MOMENTUM))
        self.concat = concat

    def forward(self, fmap, rois, union_inds):
        union_pools = union_boxes(fmap, rois, union_inds, pooling_size=self.pooling_size, stride=self.stride)
        if not self.use_feats:
            return union_pools.detach()
        pair_rois = torch.cat((rois[:, 1:][union_inds[:, 0]], rois[:, 1:][union_inds[:, 1]]), 1).data.cpu().numpy()
        rects_np = draw_union_boxes(pair_rois, self.pooling_size * 4 - 1) - 0.5
        rects = Variable(torch.FloatTensor(rects_np), volatile=fmap.volatile)
        if self.concat:
            return torch.cat((union_pools, self.conv(rects)), 1)
        return union_pools + self.conv(rects)


class GGNNObj(nn.Module):

    def __init__(self, num_obj_cls=151, time_step_num=3, hidden_dim=512, output_dim=512, use_knowledge=True, prior_matrix=''):
        super(GGNNObj, self).__init__()
        self.num_obj_cls = num_obj_cls
        self.time_step_num = time_step_num
        self.output_dim = output_dim
        if use_knowledge:
            matrix_np = np.load(prior_matrix).astype(np.float32)
        else:
            matrix_np = np.ones((num_obj_cls, num_obj_cls)).astype(np.float32) / num_obj_cls
        self.matrix = Variable(torch.from_numpy(matrix_np), requires_grad=False)
        self.fc_eq3_w = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc_eq3_u = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq4_w = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc_eq4_u = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq5_w = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc_eq5_u = nn.Linear(hidden_dim, hidden_dim)
        self.fc_output = nn.Linear(2 * hidden_dim, output_dim)
        self.ReLU = nn.ReLU(True)
        self.fc_obj_cls = nn.Linear(self.num_obj_cls * output_dim, self.num_obj_cls)

    def forward(self, input_ggnn):
        num_object = input_ggnn.size()[0]
        hidden = input_ggnn.repeat(1, self.num_obj_cls).view(num_object, self.num_obj_cls, -1)
        for t in range(self.time_step_num):
            hidden_sum = torch.sum(hidden, 0)
            av = torch.cat([torch.cat([(self.matrix.transpose(0, 1) @ (hidden_sum - hidden_i)) for hidden_i in hidden], 0), torch.cat([(self.matrix @ (hidden_sum - hidden_i)) for hidden_i in hidden], 0)], 1)
            hidden = hidden.view(num_object * self.num_obj_cls, -1)
            zv = torch.sigmoid(self.fc_eq3_w(av) + self.fc_eq3_u(hidden))
            rv = torch.sigmoid(self.fc_eq4_w(av) + self.fc_eq3_u(hidden))
            hv = torch.tanh(self.fc_eq5_w(av) + self.fc_eq5_u(rv * hidden))
            hidden = (1 - zv) * hidden + zv * hv
            hidden = hidden.view(num_object, self.num_obj_cls, -1)
        output = torch.cat((hidden.view(num_object * self.num_obj_cls, -1), input_ggnn.repeat(1, self.num_obj_cls).view(num_object * self.num_obj_cls, -1)), 1)
        output = self.fc_output(output)
        output = self.ReLU(output)
        obj_dists = self.fc_obj_cls(output.view(-1, self.num_obj_cls * self.output_dim))
        return obj_dists


class GGNNRel(nn.Module):

    def __init__(self, num_rel_cls=51, time_step_num=3, hidden_dim=512, output_dim=512, use_knowledge=True, prior_matrix=''):
        super(GGNNRel, self).__init__()
        self.num_rel_cls = num_rel_cls
        self.time_step_num = time_step_num
        self.matrix = np.load(prior_matrix).astype(np.float32)
        self.use_knowledge = use_knowledge
        self.fc_eq3_w = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc_eq3_u = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq4_w = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc_eq4_u = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq5_w = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc_eq5_u = nn.Linear(hidden_dim, hidden_dim)
        self.fc_output = nn.Linear(2 * hidden_dim, output_dim)
        self.ReLU = nn.ReLU(True)
        self.fc_rel_cls = nn.Linear((self.num_rel_cls + 2) * output_dim, self.num_rel_cls)

    def forward(self, rel_inds, sub_obj_preds, input_ggnn):
        input_rel_num, node_num, _ = input_ggnn.size()
        assert input_rel_num == len(rel_inds)
        batch_in_matrix_sub = np.zeros((input_rel_num, 2, self.num_rel_cls), dtype=np.float32)
        if self.use_knowledge:
            for index, rel in enumerate(rel_inds):
                batch_in_matrix_sub[index][0] = self.matrix[sub_obj_preds[index, 0].cpu().data, sub_obj_preds[index, 1].cpu().data]
                batch_in_matrix_sub[index][1] = batch_in_matrix_sub[index][0]
        else:
            for index, rel in enumerate(rel_inds):
                batch_in_matrix_sub[index][0] = 1.0 / float(self.num_rel_cls)
                batch_in_matrix_sub[index][1] = batch_in_matrix_sub[index][0]
        batch_in_matrix_sub_gpu = Variable(torch.from_numpy(batch_in_matrix_sub), requires_grad=False)
        del batch_in_matrix_sub
        hidden = input_ggnn
        for t in range(self.time_step_num):
            av = torch.cat((torch.bmm(batch_in_matrix_sub_gpu, hidden[:, 2:]), torch.bmm(batch_in_matrix_sub_gpu.transpose(1, 2), hidden[:, :2])), 1).repeat(1, 1, 2)
            av = av.view(input_rel_num * node_num, -1)
            flatten_hidden = hidden.view(input_rel_num * node_num, -1)
            zv = torch.sigmoid(self.fc_eq3_w(av) + self.fc_eq3_u(flatten_hidden))
            rv = torch.sigmoid(self.fc_eq4_w(av) + self.fc_eq3_u(flatten_hidden))
            hv = torch.tanh(self.fc_eq5_w(av) + self.fc_eq5_u(rv * flatten_hidden))
            flatten_hidden = (1 - zv) * flatten_hidden + zv * hv
            hidden = flatten_hidden.view(input_rel_num, node_num, -1)
        output = torch.cat((flatten_hidden, input_ggnn.view(input_rel_num * node_num, -1)), 1)
        output = self.fc_output(output)
        output = self.ReLU(output)
        rel_dists = self.fc_rel_cls(output.view(input_rel_num, -1))
        return rel_dists


MODES = 'sgdet', 'sgcls', 'predcls'


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


class GGNNObjReason(nn.Module):
    """
    Module for object classification
    """

    def __init__(self, mode='sgdet', num_obj_cls=151, obj_dim=4096, time_step_num=3, hidden_dim=512, output_dim=512, use_knowledge=True, knowledge_matrix=''):
        super(GGNNObjReason, self).__init__()
        assert mode in MODES
        self.mode = mode
        self.num_obj_cls = num_obj_cls
        self.obj_proj = nn.Linear(obj_dim, hidden_dim)
        self.ggnn_obj = GGNNObj(num_obj_cls=num_obj_cls, time_step_num=time_step_num, hidden_dim=hidden_dim, output_dim=output_dim, use_knowledge=use_knowledge, prior_matrix=knowledge_matrix)

    def forward(self, im_inds, obj_fmaps, obj_labels):
        """
        Reason object classes using knowledge of object cooccurrence
        """
        if self.mode == 'predcls':
            obj_dists = Variable(to_onehot(obj_labels.data, self.num_obj_cls))
            return obj_dists
        else:
            input_ggnn = self.obj_proj(obj_fmaps)
            lengths = []
            for i, s, e in enumerate_by_image(im_inds.data):
                lengths.append(e - s)
            obj_cum_add = np.cumsum([0] + lengths)
            obj_dists = torch.cat([self.ggnn_obj(input_ggnn[obj_cum_add[i]:obj_cum_add[i + 1]]) for i in range(len(lengths))], 0)
            return obj_dists


def _nms_single_im(scores, boxes, pre_nms_topn=12000, post_nms_topn=2000, nms_thresh=0.7):
    keep = torch.IntTensor(scores.size(0))
    vs, idx = torch.sort(scores, dim=0, descending=True)
    if idx.size(0) > pre_nms_topn:
        idx = idx[:pre_nms_topn]
    boxes_sorted = boxes[idx].contiguous()
    num_out = nms.nms_apply(keep, boxes_sorted, nms_thresh)
    num_out = min(num_out, post_nms_topn)
    keep = keep[:num_out].long()
    keep = idx[keep]
    return keep


def apply_nms(scores, boxes, pre_nms_topn=12000, post_nms_topn=2000, boxes_per_im=None, nms_thresh=0.7):
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
        keep_im = _nms_single_im(scores[s:e], boxes[s:e], pre_nms_topn, post_nms_topn, nms_thresh)
        keep.append(keep_im + s)
        im_per.append(keep_im.size(0))
        s = e
    inds = torch.cat(keep, 0)
    if just_inds:
        return inds
    return inds, im_per


class GGNNRelReason(nn.Module):
    """
    Module for relationship classification.
    """

    def __init__(self, mode='sgdet', num_obj_cls=151, num_rel_cls=51, obj_dim=4096, rel_dim=4096, time_step_num=3, hidden_dim=512, output_dim=512, use_knowledge=True, knowledge_matrix=''):
        super(GGNNRelReason, self).__init__()
        assert mode in MODES
        self.mode = mode
        self.num_obj_cls = num_obj_cls
        self.num_rel_cls = num_rel_cls
        self.obj_dim = obj_dim
        self.rel_dim = rel_dim
        self.obj_proj = nn.Linear(self.obj_dim, hidden_dim)
        self.rel_proj = nn.Linear(self.rel_dim, hidden_dim)
        self.ggnn_rel = GGNNRel(num_rel_cls=num_rel_cls, time_step_num=time_step_num, hidden_dim=hidden_dim, output_dim=output_dim, use_knowledge=use_knowledge, prior_matrix=knowledge_matrix)

    def forward(self, obj_fmaps, obj_logits, rel_inds, vr, obj_labels=None, boxes_per_cls=None):
        """
        Reason relationship classes using knowledge of object and relationship coccurrence.
        """
        if self.mode == 'predcls':
            obj_dists2 = Variable(to_onehot(obj_labels.data, self.num_obj_cls))
        else:
            obj_dists2 = obj_logits
        if self.mode == 'sgdet' and not self.training:
            probs = F.softmax(obj_dists2, 1)
            nms_mask = obj_dists2.data.clone()
            nms_mask.zero_()
            for c_i in range(1, obj_dists2.size(1)):
                scores_ci = probs.data[:, c_i]
                boxes_ci = boxes_per_cls.data[:, c_i]
                keep = apply_nms(scores_ci, boxes_ci, pre_nms_topn=scores_ci.size(0), post_nms_topn=scores_ci.size(0), nms_thresh=0.3)
                nms_mask[:, c_i][keep] = 1
            obj_preds = Variable(nms_mask * probs.data, volatile=True)[:, 1:].max(1)[1] + 1
        else:
            obj_preds = obj_labels if obj_labels is not None else obj_dists2[:, 1:].max(1)[1] + 1
        sub_obj_preds = torch.cat((obj_preds[rel_inds[:, 1]].view(-1, 1), obj_preds[rel_inds[:, 2]].view(-1, 1)), 1)
        obj_fmaps = self.obj_proj(obj_fmaps)
        vr = self.rel_proj(vr)
        input_ggnn = torch.stack([torch.cat([obj_fmaps[rel_ind[1]].unsqueeze(0), obj_fmaps[rel_ind[2]].unsqueeze(0), vr[index].repeat(self.num_rel_cls, 1)], 0) for index, rel_ind in enumerate(rel_inds)])
        rel_dists = self.ggnn_rel(rel_inds[:, 1:], sub_obj_preds, input_ggnn)
        return obj_dists2, obj_preds, rel_dists


class VRFC(nn.Module):
    """
    Module for relationship classification just using a fully connected layer.
    """

    def __init__(self, mode, rel_dim, num_obj_cls, num_rel_cls):
        super(VRFC, self).__init__()
        self.mode = mode
        self.rel_dim = rel_dim
        self.num_obj_cls = num_obj_cls
        self.num_rel_cls = num_rel_cls
        self.vr_fc = nn.Linear(self.rel_dim, self.num_rel_cls)

    def forward(self, obj_logits, vr, obj_labels=None, boxes_per_cls=None):
        if self.mode == 'predcls':
            obj_dists2 = Variable(to_onehot(obj_labels.data, self.num_obj_cls))
        else:
            obj_dists2 = obj_logits
        if self.mode == 'sgdet' and not self.training:
            probs = F.softmax(obj_dists2, 1)
            nms_mask = obj_dists2.data.clone()
            nms_mask.zero_()
            for c_i in range(1, obj_dists2.size(1)):
                scores_ci = probs.data[:, c_i]
                boxes_ci = boxes_per_cls.data[:, c_i]
                keep = apply_nms(scores_ci, boxes_ci, pre_nms_topn=scores_ci.size(0), post_nms_topn=scores_ci.size(0), nms_thresh=0.3)
                nms_mask[:, c_i][keep] = 1
            obj_preds = Variable(nms_mask * probs.data, volatile=True)[:, 1:].max(1)[1] + 1
        else:
            obj_preds = obj_labels if obj_labels is not None else obj_dists2[:, 1:].max(1)[1] + 1
        rel_dists = self.vr_fc(vr)
        return obj_dists2, obj_preds, rel_dists


class Flattener(nn.Module):

    def __init__(self):
        """
        Flattens last 3 dimensions to make it only batch size, -1
        """
        super(Flattener, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


ANCHOR_RATIOS = 0.23232838, 0.63365731, 1.28478321, 3.15089189


ANCHOR_SCALES = 2.22152954, 4.12315647, 7.21692515, 12.60263013, 22.7102731


ANCHOR_SIZE = 16


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
        return np.column_stack((boxes[:, :2] - 0.5 * boxes[:, 2:], boxes[:, :2] + 0.5 * (boxes[:, 2:] - 2.0)))
    return torch.cat((boxes[:, :2] - 0.5 * boxes[:, 2:], boxes[:, :2] + 0.5 * (boxes[:, 2:] - 2.0)), 1)


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


def filter_roi_proposals(box_preds, class_preds, boxes_per_im, nms_thresh=0.7, pre_nms_topn=12000, post_nms_topn=2000):
    inds, im_per = apply_nms(class_preds, box_preds, pre_nms_topn=pre_nms_topn, post_nms_topn=post_nms_topn, boxes_per_im=boxes_per_im, nms_thresh=nms_thresh)
    img_inds = torch.cat([(val * torch.ones(i)) for val, i in enumerate(im_per)], 0)
    rois = torch.cat((img_inds[:, None], box_preds[inds]), 1)
    return rois


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
    sel_inds = index[:, nd - 1].clone()
    mult_factor = x.size(nd - 1)
    for col in range(nd - 2, -1, -1):
        sel_inds += index[:, col] * mult_factor
        mult_factor *= x.size(col)
    grouped = x.view(-1, dim)[sel_inds]
    return grouped


IM_SCALE = 592


def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
  Given a vector of widths (ws) and heights (hs) around a center
  (x_ctr, y_ctr), output a set of anchors (windows).
  """
    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1), y_ctr - 0.5 * (hs - 1), x_ctr + 0.5 * (ws - 1), y_ctr + 0.5 * (hs - 1)))
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


def generate_base_anchors(base_size=16, ratios=[0.5, 1, 2], scales=2 ** np.arange(3, 6)):
    """
  Generate anchor (reference) windows by enumerating aspect ratios X
  scales wrt a reference (0, 0, 15, 15) window.
  """
    base_anchor = np.array([1, 1, base_size, base_size]) - 1
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales) for i in range(ratio_anchors.shape[0])])
    return anchors


def generate_anchors(base_size=16, feat_stride=16, anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2)):
    """ A wrapper function to generate anchors given different scales
    Also return the number of anchors in variable 'length'
  """
    anchors = generate_base_anchors(base_size=base_size, ratios=np.array(anchor_ratios), scales=np.array(anchor_scales))
    A = anchors.shape[0]
    shift_x = np.arange(0, IM_SCALE // feat_stride) * feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_x)
    shifts = np.stack([shift_x, shift_y, shift_x, shift_y], -1)
    all_anchors = shifts[:, :, None] + anchors[None, None]
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
        self.conv = nn.Sequential(nn.Conv2d(input_dim, dim, kernel_size=3, padding=1), nn.ReLU6(inplace=True), nn.Conv2d(dim, self.anchor_target_dim * self._A, kernel_size=1))
        ans_np = generate_anchors(base_size=ANCHOR_SIZE, feat_stride=self.stride, anchor_scales=ANCHOR_SCALES, anchor_ratios=ANCHOR_RATIOS)
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
        rez = rez.view(rez.size(0), rez.size(1), rez.size(2), self._A, self.anchor_target_dim)
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
        tai[:, 0] -= image_offset
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

    def roi_proposals(self, fmap, im_sizes, nms_thresh=0.7, pre_nms_topn=12000, post_nms_topn=2000):
        """
        :param fmap: [batch_size, IM_SIZE/16, IM_SIZE/16, A, 6]
        :param im_sizes:        [batch_size, 3] numpy array of (h, w, scale)
        :return: ROIS: shape [a <=post_nms_topn, 5] array of ROIS.
        """
        class_fmap = fmap[:, :, :, :, :2].contiguous()
        class_preds = F.softmax(class_fmap, 4)[..., 1].data.contiguous()
        box_fmap = fmap[:, :, :, :, 2:].data.contiguous()
        anchor_stacked = torch.cat([self.anchors[None]] * fmap.size(0), 0)
        box_preds = bbox_preds(anchor_stacked.view(-1, 4), box_fmap.view(-1, 4)).view(*box_fmap.size())
        for i, (h, w, scale) in enumerate(im_sizes):
            h_end = int(h) // self.stride
            w_end = int(w) // self.stride
            if h_end < class_preds.size(1):
                class_preds[i, h_end:] = -0.01
            if w_end < class_preds.size(2):
                class_preds[i, :, w_end:] = -0.01
            box_preds[i, :, :, :, 0].clamp_(min=0, max=w - 1)
            box_preds[i, :, :, :, 1].clamp_(min=0, max=h - 1)
            box_preds[i, :, :, :, 2].clamp_(min=0, max=w - 1)
            box_preds[i, :, :, :, 3].clamp_(min=0, max=h - 1)
        sizes = center_size(box_preds.view(-1, 4))
        class_preds.view(-1)[(sizes[:, 2] < 4) | (sizes[:, 3] < 4)] = -0.01
        return filter_roi_proposals(box_preds.view(-1, 4), class_preds.view(-1), boxes_per_im=np.array([np.prod(box_preds.size()[1:-1])] * fmap.size(0)), nms_thresh=nms_thresh, pre_nms_topn=pre_nms_topn, post_nms_topn=post_nms_topn)


class Result(object):
    """ little container class for holding the detection result
        od: object detector, rm: rel model"""

    def __init__(self, od_obj_dists=None, rm_obj_dists=None, obj_scores=None, obj_preds=None, obj_fmap=None, od_box_deltas=None, rm_box_deltas=None, od_box_targets=None, rm_box_targets=None, od_box_priors=None, rm_box_priors=None, boxes_assigned=None, boxes_all=None, od_obj_labels=None, rm_obj_labels=None, rpn_scores=None, rpn_box_deltas=None, rel_labels=None, im_inds=None, fmap=None, rel_dists=None, rel_inds=None, rel_rep=None):
        self.__dict__.update(locals())
        del self.__dict__['self']

    def is_none(self):
        return all([(v is None) for k, v in self.__dict__.items() if k != 'self'])


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
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2), box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2), box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp(max_xy - min_xy + 1.0, min=0)
    return inter[:, :, 0] * inter[:, :, 1]


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
    area_a = ((box_a[:, 2] - box_a[:, 0] + 1.0) * (box_a[:, 3] - box_a[:, 1] + 1.0)).unsqueeze(1).expand_as(inter)
    area_b = ((box_b[:, 2] - box_b[:, 0] + 1.0) * (box_b[:, 3] - box_b[:, 1] + 1.0)).unsqueeze(0).expand_as(inter)
    union = area_a + area_b - inter
    return inter / union


def filter_det(scores, boxes, start_ind=0, max_per_img=100, thresh=0.001, pre_nms_topn=6000, post_nms_topn=300, nms_thresh=0.3, nms_filter_duplicates=True):
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
        scores_ci = scores.data[:, c_i]
        boxes_ci = boxes.data[:, c_i]
        keep = apply_nms(scores_ci, boxes_ci, pre_nms_topn=pre_nms_topn, post_nms_topn=post_nms_topn, nms_thresh=nms_thresh)
        nms_mask[:, c_i][keep] = 1
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
        inds_all = nz[:, 0]
        labels_all = nz[:, 1]
        scores_all = scores.data.view(-1)[inds_all * scores.data.size(1) + labels_all]
    vs, idx = torch.sort(scores_all, dim=0, descending=True)
    idx = idx[vs > thresh]
    if max_per_img < idx.size(0):
        idx = idx[:max_per_img]
    inds_all = inds_all[idx] + start_ind
    scores_all = Variable(scores_all[idx], volatile=True)
    labels_all = Variable(labels_all[idx], volatile=True)
    return inds_all, scores_all, labels_all


def gather_res(outputs, target_device, dim=0):
    """
    Assuming the signatures are the same accross results!
    """
    out = outputs[0]
    args = {field: Gather.apply(target_device, dim, *[getattr(o, field) for o in outputs]) for field, v in out.__dict__.items() if v is not None}
    return type(out)(**args)


def load_resnet():
    model = resnet101(pretrained=True)
    del model.layer4
    del model.avgpool
    del model.fc
    return model


def load_vgg(use_dropout=True, use_relu=True, use_linear=True, pretrained=True):
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


def _sel_inds(ious, gt_classes_i, fg_thresh=0.5, fg_rois_per_image=128, rois_per_image=256, n_sample_per=1):
    fg_ious = ious.T >= fg_thresh
    fg_inds = []
    for i, (ious_i, cls_i) in enumerate(zip(fg_ious, gt_classes_i)):
        n_sample_this_roi = min(n_sample_per, ious_i.sum())
        if n_sample_this_roi > 0:
            p = ious_i.astype(np.float64) / ious_i.sum()
            for ind in npr.choice(ious_i.shape[0], p=p, size=n_sample_this_roi, replace=False):
                fg_inds.append((ind, i))
    fg_inds = np.array(fg_inds, dtype=np.int64)
    if fg_inds.size == 0:
        fg_inds = np.zeros((0, 2), dtype=np.int64)
    elif fg_inds.shape[0] > fg_rois_per_image:
        fg_inds = fg_inds[npr.choice(fg_inds.shape[0], size=fg_rois_per_image, replace=False)]
    max_overlaps = ious.max(1)
    bg_inds = np.where((max_overlaps < BG_THRESH_HI) & (max_overlaps >= BG_THRESH_LO))[0]
    bg_rois_per_this_image = min(rois_per_image - fg_inds.shape[0], bg_inds.size)
    if bg_inds.size > 0:
        bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image, replace=False)
    obj_inds = np.concatenate((fg_inds[:, 0], bg_inds), 0)
    obj_assignments_i = np.concatenate((fg_inds[:, 1], np.zeros(bg_inds.shape[0], dtype=np.int64)))
    obj_labels_i = gt_classes_i[obj_assignments_i]
    obj_labels_i[fg_inds.shape[0]:] = 0
    return obj_inds, obj_labels_i, obj_assignments_i


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
def proposal_assignments_det(rpn_rois, gt_boxes, gt_classes, image_offset, fg_thresh=0.5):
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
    gt_img_inds = gt_classes[:, 0] - image_offset
    all_boxes = torch.cat([rpn_rois[:, 1:], gt_boxes], 0)
    ims_per_box = torch.cat([rpn_rois[:, 0].long(), gt_img_inds], 0)
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
        keep_inds_np, num_fg = _sel_inds(max_overlaps, fg_thresh, fg_rois_per_image, ROIS_PER_IMG)
        if keep_inds_np.size == 0:
            continue
        keep_inds = torch.LongTensor(keep_inds_np)
        labels_ = gt_classes[:, 1][gt_assignment[keep_inds]]
        bbox_target_ = gt_boxes[gt_assignment[keep_inds]]
        if num_fg < labels_.size(0):
            labels_[num_fg:] = 0
        rois_ = torch.cat((im_sorted[t_start:t_end, None][keep_inds].float(), all_boxes[t_start:t_end][keep_inds]), 1)
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
    rand_idx = torch.LongTensor(rand_idx)
    chosen = tensor[rand_idx].contiguous()
    return chosen


@to_variable
def proposal_assignments_gtbox(rois, gt_boxes, gt_classes, gt_rels, image_offset, fg_thresh=0.5):
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
    im_inds = rois[:, 0].long()
    num_im = im_inds[-1] + 1
    fg_rels = gt_rels.clone()
    fg_rels[:, 0] -= image_offset
    offset = {}
    for i, s, e in enumerate_by_image(im_inds):
        offset[i] = s
    for i, s, e in enumerate_by_image(fg_rels[:, 0]):
        fg_rels[s:e, 1:3] += offset[i]
    is_cand = im_inds[:, None] == im_inds[None]
    is_cand.view(-1)[diagonal_inds(is_cand)] = 0
    is_cand.view(-1)[fg_rels[:, 1] * im_inds.size(0) + fg_rels[:, 2]] = 0
    is_bgcand = is_cand.nonzero()
    num_fg = min(fg_rels.size(0), int(RELS_PER_IMG * REL_FG_FRACTION * num_im))
    if num_fg < fg_rels.size(0):
        fg_rels = random_choose(fg_rels, num_fg)
    num_bg = min(is_bgcand.size(0) if is_bgcand.dim() > 0 else 0, int(RELS_PER_IMG * num_im) - num_fg)
    if num_bg > 0:
        bg_rels = torch.cat((im_inds[is_bgcand[:, 0]][:, None], is_bgcand, (is_bgcand[:, 0, None] < -10).long()), 1)
        if num_bg < is_bgcand.size(0):
            bg_rels = random_choose(bg_rels, num_bg)
        rel_labels = torch.cat((fg_rels, bg_rels), 0)
    else:
        rel_labels = fg_rels
    _, perm = torch.sort(rel_labels[:, 0] * gt_boxes.size(0) ** 2 + rel_labels[:, 1] * gt_boxes.size(0) + rel_labels[:, 2])
    rel_labels = rel_labels[perm].contiguous()
    labels = gt_classes[:, 1].contiguous()
    return rois, labels, rel_labels


class ObjectDetector(nn.Module):
    """
    Core model for doing object detection + getting the visual features. This could be the first step in
    a pipeline. We can provide GT rois or use the RPN (which would then be classification!)
    """
    MODES = 'rpntrain', 'gtbox', 'refinerels', 'proposals'

    def __init__(self, classes, mode='rpntrain', num_gpus=1, nms_filter_duplicates=True, max_per_img=64, use_resnet=False, thresh=0.05):
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
            self.compress = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=1), nn.ReLU(inplace=True), nn.BatchNorm2d(256))
            self.roi_fmap = nn.Sequential(nn.Linear(256 * 7 * 7, 2048), nn.SELU(inplace=True), nn.AlphaDropout(p=0.05), nn.Linear(2048, 2048), nn.SELU(inplace=True), nn.AlphaDropout(p=0.05))
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
        feature_pool = RoIAlignFunction(self.pooling_size, self.pooling_size, spatial_scale=1 / 16)(self.compress(features) if self.use_resnet else features, rois)
        return self.roi_fmap(feature_pool.view(rois.size(0), -1))

    def rpn_boxes(self, fmap, im_sizes, image_offset, gt_boxes=None, gt_classes=None, gt_rels=None, train_anchor_inds=None, proposals=None):
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
        rois = self.rpn_head.roi_proposals(rpn_feats, im_sizes, nms_thresh=0.7, pre_nms_topn=12000 if self.training and self.mode == 'rpntrain' else 6000, post_nms_topn=2000 if self.training and self.mode == 'rpntrain' else 1000)
        if self.training:
            if gt_boxes is None or gt_classes is None or train_anchor_inds is None:
                raise ValueError('Must supply GT boxes, GT classes, trainanchors when in train mode')
            rpn_scores, rpn_box_deltas = self.rpn_head.anchor_preds(rpn_feats, train_anchor_inds, image_offset)
            if gt_rels is not None and self.mode == 'rpntrain':
                raise ValueError("Training the object detector and the relationship model with detectionat the same time isn't supported")
            if self.mode == 'refinerels':
                all_rois = Variable(rois)
                labels = None
                bbox_targets = None
                rel_labels = None
            else:
                all_rois, labels, bbox_targets = proposal_assignments_det(rois, gt_boxes.data, gt_classes.data, image_offset, fg_thresh=0.5)
                rel_labels = None
        else:
            all_rois = Variable(rois, volatile=True)
            labels = None
            bbox_targets = None
            rel_labels = None
            rpn_box_deltas = None
            rpn_scores = None
        return all_rois, labels, bbox_targets, rpn_scores, rpn_box_deltas, rel_labels

    def gt_boxes(self, fmap, im_sizes, image_offset, gt_boxes=None, gt_classes=None, gt_rels=None, train_anchor_inds=None, proposals=None):
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
        im_inds = gt_classes[:, 0] - image_offset
        rois = torch.cat((im_inds.float()[:, None], gt_boxes), 1)
        if gt_rels is not None and self.training:
            rois, labels, rel_labels = proposal_assignments_gtbox(rois.data, gt_boxes.data, gt_classes.data, gt_rels.data, image_offset, fg_thresh=0.5)
        else:
            labels = gt_classes[:, 1]
            rel_labels = None
        return rois, labels, None, None, None, rel_labels

    def proposal_boxes(self, fmap, im_sizes, image_offset, gt_boxes=None, gt_classes=None, gt_rels=None, train_anchor_inds=None, proposals=None):
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
        rois = filter_roi_proposals(proposals[:, 2:].data.contiguous(), proposals[:, 1].data.contiguous(), np.array([2000] * len(im_sizes)), nms_thresh=0.7, pre_nms_topn=12000 if self.training and self.mode == 'rpntrain' else 6000, post_nms_topn=2000 if self.training and self.mode == 'rpntrain' else 1000)
        if self.training:
            all_rois, labels, bbox_targets = proposal_assignments_det(rois, gt_boxes.data, gt_classes.data, image_offset, fg_thresh=0.5)
            all_rois = torch.cat((all_rois, Variable(rois)), 0)
        else:
            all_rois = Variable(rois, volatile=True)
            labels = None
            bbox_targets = None
        rpn_scores = None
        rpn_box_deltas = None
        rel_labels = None
        return all_rois, labels, bbox_targets, rpn_scores, rpn_box_deltas, rel_labels

    def get_boxes(self, *args, **kwargs):
        if self.mode == 'gtbox':
            fn = self.gt_boxes
        elif self.mode == 'proposals':
            assert kwargs['proposals'] is not None
            fn = self.proposal_boxes
        else:
            fn = self.rpn_boxes
        return fn(*args, **kwargs)

    def forward(self, x, im_sizes, image_offset, gt_boxes=None, gt_classes=None, gt_rels=None, proposals=None, train_anchor_inds=None, return_fmap=False):
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
        rois, obj_labels, bbox_targets, rpn_scores, rpn_box_deltas, rel_labels = self.get_boxes(fmap, im_sizes, image_offset, gt_boxes, gt_classes, gt_rels, train_anchor_inds, proposals=proposals)
        obj_fmap = self.obj_feature_map(fmap, rois)
        od_obj_dists = self.score_fc(obj_fmap)
        od_box_deltas = self.bbox_fc(obj_fmap).view(-1, len(self.classes), 4) if self.mode != 'gtbox' else None
        od_box_priors = rois[:, 1:]
        if not self.training and not self.mode == 'gtbox' or self.mode in ('proposals', 'refinerels'):
            nms_inds, nms_scores, nms_preds, nms_boxes_assign, nms_boxes, nms_imgs = self.nms_boxes(od_obj_dists, rois, od_box_deltas, im_sizes)
            im_inds = nms_imgs + image_offset
            obj_dists = od_obj_dists[nms_inds]
            obj_fmap = obj_fmap[nms_inds]
            box_deltas = od_box_deltas[nms_inds]
            box_priors = nms_boxes[:, 0]
            if self.training and not self.mode == 'gtbox':
                pred_to_gtbox = bbox_overlaps(box_priors, gt_boxes).data
                pred_to_gtbox[im_inds.data[:, None] != gt_classes.data[None, :, 0]] = 0.0
                max_overlaps, argmax_overlaps = pred_to_gtbox.max(1)
                rm_obj_labels = gt_classes[:, 1][argmax_overlaps]
                rm_obj_labels[max_overlaps < 0.5] = 0
            else:
                rm_obj_labels = None
        else:
            im_inds = rois[:, 0].long().contiguous() + image_offset
            nms_scores = None
            nms_preds = None
            nms_boxes_assign = None
            nms_boxes = None
            box_priors = rois[:, 1:]
            rm_obj_labels = obj_labels
            box_deltas = od_box_deltas
            obj_dists = od_obj_dists
        return Result(od_obj_dists=od_obj_dists, rm_obj_dists=obj_dists, obj_scores=nms_scores, obj_preds=nms_preds, obj_fmap=obj_fmap, od_box_deltas=od_box_deltas, rm_box_deltas=box_deltas, od_box_targets=bbox_targets, rm_box_targets=bbox_targets, od_box_priors=od_box_priors, rm_box_priors=box_priors, boxes_assigned=nms_boxes_assign, boxes_all=nms_boxes, od_obj_labels=obj_labels, rm_obj_labels=rm_obj_labels, rpn_scores=rpn_scores, rpn_box_deltas=rpn_box_deltas, rel_labels=rel_labels, im_inds=im_inds, fmap=fmap if return_fmap else None)

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
        boxes = bbox_preds(rois[:, None, 1:].expand_as(box_deltas).contiguous().view(-1, 4), box_deltas.view(-1, 4)).view(*box_deltas.size())
        inds = rois[:, 0].long().contiguous()
        dets = []
        for i, s, e in enumerate_by_image(inds.data):
            h, w = im_sizes[i, :2]
            boxes[s:e, :, 0].data.clamp_(min=0, max=w - 1)
            boxes[s:e, :, 1].data.clamp_(min=0, max=h - 1)
            boxes[s:e, :, 2].data.clamp_(min=0, max=w - 1)
            boxes[s:e, :, 3].data.clamp_(min=0, max=h - 1)
            d_filtered = filter_det(F.softmax(obj_dists[s:e], 1), boxes[s:e], start_ind=s, nms_filter_duplicates=self.nms_filter_duplicates, max_per_img=self.max_per_img, thresh=self.thresh)
            if d_filtered is not None:
                dets.append(d_filtered)
        if len(dets) == 0:
            None
            return None
        nms_inds, nms_scores, nms_labels = [torch.cat(x, 0) for x in zip(*dets)]
        twod_inds = nms_inds * boxes.size(1) + nms_labels.data
        nms_boxes_assign = boxes.view(-1, 4)[twod_inds]
        nms_boxes = torch.cat((rois[:, 1:][nms_inds][:, None], boxes[nms_inds][:, 1:]), 1)
        return nms_inds, nms_scores, nms_labels, nms_boxes_assign, nms_boxes, inds[nms_inds]

    def __getitem__(self, batch):
        """ Hack to do multi-GPU training"""
        batch.scatter()
        if self.num_gpus == 1:
            return self(*batch[0])
        replicas = nn.parallel.replicate(self, devices=list(range(self.num_gpus)))
        outputs = nn.parallel.parallel_apply(replicas, [batch[i] for i in range(self.num_gpus)])
        if any([x.is_none() for x in outputs]):
            assert not self.training
            return None
        return gather_res(outputs, 0, dim=0)


def arange(base_tensor, n=None):
    new_size = base_tensor.size(0) if n is None else n
    new_vec = base_tensor.new(new_size).long()
    torch.arange(0, new_size, out=new_vec)
    return new_vec


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
        raise ValueError('Boxes needs to be [num_box, 4] but its {}'.format(boxes.size()))
    num_box = boxes.size(0)
    assert obj_scores.size(0) == num_box
    assert obj_classes.size() == obj_scores.size()
    num_rel = rel_inds.size(0)
    assert rel_inds.size(1) == 2
    assert pred_scores.size(0) == num_rel
    obj_scores0 = obj_scores.data[rel_inds[:, 0]]
    obj_scores1 = obj_scores.data[rel_inds[:, 1]]
    pred_scores_max, pred_classes_argmax = pred_scores.data[:, 1:].max(1)
    pred_classes_argmax = pred_classes_argmax + 1
    rel_scores_argmaxed = pred_scores_max * obj_scores0 * obj_scores1
    rel_scores_vs, rel_scores_idx = torch.sort(rel_scores_argmaxed.view(-1), dim=0, descending=True)
    rels = rel_inds[rel_scores_idx].cpu().numpy()
    pred_scores_sorted = pred_scores[rel_scores_idx].data.cpu().numpy()
    obj_scores_np = obj_scores.data.cpu().numpy()
    objs_np = obj_classes.data.cpu().numpy()
    boxes_out = boxes.data.cpu().numpy()
    return boxes_out, objs_np, obj_scores_np, rels, pred_scores_sorted


@to_variable
def rel_assignments(im_inds, rpn_rois, roi_gtlabels, gt_boxes, gt_classes, gt_rels, image_offset, fg_thresh=0.5, num_sample_per_gt=4, filter_non_overlap=True):
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
    gt_classes_np[:, 0] -= image_offset
    gt_rels_np[:, 0] -= image_offset
    num_im = gt_classes_np[:, 0].max() + 1
    rel_labels = []
    num_box_seen = 0
    for im_ind in range(num_im):
        pred_ind = np.where(pred_inds_np == im_ind)[0]
        gt_ind = np.where(gt_classes_np[:, 0] == im_ind)[0]
        gt_boxes_i = gt_boxes_np[gt_ind]
        gt_classes_i = gt_classes_np[gt_ind, 1]
        gt_rels_i = gt_rels_np[gt_rels_np[:, 0] == im_ind, 1:]
        pred_boxes_i = pred_boxes_np[pred_ind]
        pred_boxlabels_i = pred_boxlabels_np[pred_ind]
        ious = bbox_overlaps(pred_boxes_i, gt_boxes_i)
        is_match = (pred_boxlabels_i[:, None] == gt_classes_i[None]) & (ious >= fg_thresh)
        pbi_iou = bbox_overlaps(pred_boxes_i, pred_boxes_i)
        if filter_non_overlap:
            rel_possibilities = (pbi_iou < 1) & (pbi_iou > 0)
            rels_intersect = rel_possibilities
        else:
            rel_possibilities = np.ones((pred_boxes_i.shape[0], pred_boxes_i.shape[0]), dtype=np.int64) - np.eye(pred_boxes_i.shape[0], dtype=np.int64)
            rels_intersect = (pbi_iou < 1) & (pbi_iou > 0)
        rel_possibilities[pred_boxlabels_i == 0] = 0
        rel_possibilities[:, pred_boxlabels_i == 0] = 0
        fg_rels = []
        p_size = []
        for i, (from_gtind, to_gtind, rel_id) in enumerate(gt_rels_i):
            fg_rels_i = []
            fg_scores_i = []
            for from_ind in np.where(is_match[:, from_gtind])[0]:
                for to_ind in np.where(is_match[:, to_gtind])[0]:
                    if from_ind != to_ind:
                        fg_rels_i.append((from_ind, to_ind, rel_id))
                        fg_scores_i.append(ious[from_ind, from_gtind] * ious[to_ind, to_gtind])
                        rel_possibilities[from_ind, to_ind] = 0
            if len(fg_rels_i) == 0:
                continue
            p = np.array(fg_scores_i)
            p = p / p.sum()
            p_size.append(p.shape[0])
            num_to_add = min(p.shape[0], num_sample_per_gt)
            for rel_to_add in npr.choice(p.shape[0], p=p, size=num_to_add, replace=False):
                fg_rels.append(fg_rels_i[rel_to_add])
        fg_rels = np.array(fg_rels, dtype=np.int64)
        if fg_rels.size > 0 and fg_rels.shape[0] > fg_rels_per_image:
            fg_rels = fg_rels[npr.choice(fg_rels.shape[0], size=fg_rels_per_image, replace=False)]
        elif fg_rels.size == 0:
            fg_rels = np.zeros((0, 3), dtype=np.int64)
        bg_rels = np.column_stack(np.where(rel_possibilities))
        bg_rels = np.column_stack((bg_rels, np.zeros(bg_rels.shape[0], dtype=np.int64)))
        num_bg_rel = min(64 - fg_rels.shape[0], bg_rels.shape[0])
        if bg_rels.size > 0:
            bg_rels = bg_rels[np.random.choice(bg_rels.shape[0], size=num_bg_rel, replace=False)]
        else:
            bg_rels = np.zeros((0, 3), dtype=np.int64)
        if fg_rels.size == 0 and bg_rels.size == 0:
            bg_rels = np.array([[0, 0, 0]], dtype=np.int64)
        all_rels_i = np.concatenate((fg_rels, bg_rels), 0)
        all_rels_i[:, 0:2] += num_box_seen
        all_rels_i = all_rels_i[np.lexsort((all_rels_i[:, 1], all_rels_i[:, 0]))]
        rel_labels.append(np.column_stack((im_ind * np.ones(all_rels_i.shape[0], dtype=np.int64), all_rels_i)))
        num_box_seen += pred_boxes_i.shape[0]
    rel_labels = torch.LongTensor(np.concatenate(rel_labels, 0))
    return rel_labels


def resnet_l4(relu_end=True):
    model = resnet101(pretrained=True)
    l4 = model.layer4
    if not relu_end:
        l4[-1].relu_end = False
    l4[0].conv2.stride = 1, 1
    l4[0].downsample[0].stride = 1, 1
    return l4


class KERN(nn.Module):
    """
    Knowledge-Embedded Routing Network 
    """

    def __init__(self, classes, rel_classes, mode='sgdet', num_gpus=1, require_overlap_det=True, pooling_dim=4096, use_resnet=False, thresh=0.01, use_proposals=False, use_ggnn_obj=False, ggnn_obj_time_step_num=3, ggnn_obj_hidden_dim=512, ggnn_obj_output_dim=512, use_ggnn_rel=False, ggnn_rel_time_step_num=3, ggnn_rel_hidden_dim=512, ggnn_rel_output_dim=512, use_obj_knowledge=True, use_rel_knowledge=True, obj_knowledge='', rel_knowledge=''):
        """
        :param classes: Object classes
        :param rel_classes: Relationship classes. None if were not using rel mode
        :param mode: (sgcls, predcls, or sgdet)
        :param num_gpus: how many GPUS 2 use
        :param require_overlap_det: Whether two objects must intersect
        """
        super(KERN, self).__init__()
        self.classes = classes
        self.rel_classes = rel_classes
        self.num_gpus = num_gpus
        assert mode in MODES
        self.mode = mode
        self.pooling_size = 7
        self.obj_dim = 2048 if use_resnet else 4096
        self.rel_dim = self.obj_dim
        self.pooling_dim = pooling_dim
        self.use_ggnn_obj = use_ggnn_obj
        self.use_ggnn_rel = use_ggnn_rel
        self.require_overlap = require_overlap_det and self.mode == 'sgdet'
        self.detector = ObjectDetector(classes=classes, mode=('proposals' if use_proposals else 'refinerels') if mode == 'sgdet' else 'gtbox', use_resnet=use_resnet, thresh=thresh, max_per_img=64)
        self.union_boxes = UnionBoxesAndFeats(pooling_size=self.pooling_size, stride=16, dim=1024 if use_resnet else 512)
        if use_resnet:
            self.roi_fmap = nn.Sequential(resnet_l4(relu_end=False), nn.AvgPool2d(self.pooling_size), Flattener())
        else:
            roi_fmap = [Flattener(), load_vgg(use_dropout=False, use_relu=False, use_linear=pooling_dim == 4096, pretrained=False).classifier]
            if pooling_dim != 4096:
                roi_fmap.append(nn.Linear(4096, pooling_dim))
            self.roi_fmap = nn.Sequential(*roi_fmap)
            self.roi_fmap_obj = load_vgg(pretrained=False).classifier
        if self.use_ggnn_obj:
            self.ggnn_obj_reason = GGNNObjReason(mode=self.mode, num_obj_cls=len(self.classes), obj_dim=self.obj_dim, time_step_num=ggnn_obj_time_step_num, hidden_dim=ggnn_obj_hidden_dim, output_dim=ggnn_obj_output_dim, use_knowledge=use_obj_knowledge, knowledge_matrix=obj_knowledge)
        if self.use_ggnn_rel:
            self.ggnn_rel_reason = GGNNRelReason(mode=self.mode, num_obj_cls=len(self.classes), num_rel_cls=len(rel_classes), obj_dim=self.obj_dim, rel_dim=self.rel_dim, time_step_num=ggnn_rel_time_step_num, hidden_dim=ggnn_rel_hidden_dim, output_dim=ggnn_obj_output_dim, use_knowledge=use_rel_knowledge, knowledge_matrix=rel_knowledge)
        else:
            self.vr_fc_cls = VRFC(self.mode, self.rel_dim, len(self.classes), len(self.rel_classes))

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
            rel_cands = im_inds.data[:, None] == im_inds.data[None]
            rel_cands.view(-1)[diagonal_inds(rel_cands)] = 0
            if self.require_overlap:
                rel_cands = rel_cands & (bbox_overlaps(box_priors.data, box_priors.data) > 0)
                amt_to_add = 100 - rel_cands.long().sum()
            rel_cands = rel_cands.nonzero()
            if rel_cands.dim() == 0:
                rel_cands = im_inds.data.new(1, 2).fill_(0)
            rel_inds = torch.cat((im_inds.data[rel_cands[:, 0]][:, None], rel_cands), 1)
        return rel_inds

    def obj_feature_map(self, features, rois):
        """
        Gets the ROI features
        :param features: [batch_size, dim, IM_SIZE/4, IM_SIZE/4] (features at level p2)
        :param rois: [num_rois, 5] array of [img_num, x0, y0, x1, y1].
        :return: [num_rois, #dim] array
        """
        feature_pool = RoIAlignFunction(self.pooling_size, self.pooling_size, spatial_scale=1 / 16)(features, rois)
        return self.roi_fmap_obj(feature_pool.view(rois.size(0), -1))

    def forward(self, x, im_sizes, image_offset, gt_boxes=None, gt_classes=None, gt_rels=None, proposals=None, train_anchor_inds=None, return_fmap=False):
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
        result = self.detector(x, im_sizes, image_offset, gt_boxes, gt_classes, gt_rels, proposals, train_anchor_inds, return_fmap=True)
        if result.is_none():
            return ValueError('heck')
        im_inds = result.im_inds - image_offset
        boxes = result.rm_box_priors
        if self.training and result.rel_labels is None:
            assert self.mode == 'sgdet'
            result.rel_labels = rel_assignments(im_inds.data, boxes.data, result.rm_obj_labels.data, gt_boxes.data, gt_classes.data, gt_rels.data, image_offset, filter_non_overlap=True, num_sample_per_gt=1)
        rel_inds = self.get_rel_inds(result.rel_labels, im_inds, boxes)
        rois = torch.cat((im_inds[:, None].float(), boxes), 1)
        result.obj_fmap = self.obj_feature_map(result.fmap.detach(), rois)
        if self.use_ggnn_obj:
            result.rm_obj_dists = self.ggnn_obj_reason(im_inds, result.obj_fmap, result.rm_obj_labels if self.training or self.mode == 'predcls' else None)
        vr = self.visual_rep(result.fmap.detach(), rois, rel_inds[:, 1:])
        if self.use_ggnn_rel:
            result.rm_obj_dists, result.obj_preds, result.rel_dists = self.ggnn_rel_reason(obj_fmaps=result.obj_fmap, obj_logits=result.rm_obj_dists, vr=vr, rel_inds=rel_inds, obj_labels=result.rm_obj_labels if self.training or self.mode == 'predcls' else None, boxes_per_cls=result.boxes_all)
        else:
            result.rm_obj_dists, result.obj_preds, result.rel_dists = self.vr_fc_cls(obj_logits=result.rm_obj_dists, vr=vr, obj_labels=result.rm_obj_labels if self.training or self.mode == 'predcls' else None, boxes_per_cls=result.boxes_all)
        if self.training:
            return result
        twod_inds = arange(result.obj_preds.data) * self.num_classes + result.obj_preds.data
        result.obj_scores = F.softmax(result.rm_obj_dists, dim=1).view(-1)[twod_inds]
        if self.mode == 'sgdet':
            bboxes = result.boxes_all.view(-1, 4)[twod_inds].view(result.boxes_all.size(0), 4)
        else:
            bboxes = result.rm_box_priors
        rel_rep = F.softmax(result.rel_dists, dim=1)
        return filter_dets(bboxes, result.obj_scores, result.obj_preds, rel_inds[:, 1:], rel_rep)

    def __getitem__(self, batch):
        """ Hack to do multi-GPU training"""
        batch.scatter()
        if self.num_gpus == 1:
            return self(*batch[0])
        replicas = nn.parallel.replicate(self, devices=list(range(self.num_gpus)))
        outputs = nn.parallel.parallel_apply(replicas, [batch[i] for i in range(self.num_gpus)])
        if self.training:
            return gather_res(outputs, 0, dim=0)
        return outputs


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, relu_end=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BATCHNORM_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
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
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
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
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion, momentum=BATCHNORM_MOMENTUM))
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


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Flattener,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (RPNHead,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1024, 64, 64])], {}),
     False),
    (VRFC,
     lambda: ([], {'mode': 4, 'rel_dim': 4, 'num_obj_cls': 4, 'num_rel_cls': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_yuweihao_KERN(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

