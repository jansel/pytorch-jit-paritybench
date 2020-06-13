import sys
_module = sys.modules[__name__]
del sys
pcdet = _module
config = _module
datasets = _module
augmentation_utils = _module
dbsampler = _module
dataset = _module
kitti_dataset = _module
kitti_eval = _module
eval = _module
kitti_common = _module
rotate_iou = _module
models = _module
bbox_heads = _module
anchor_target_assigner = _module
rpn_head = _module
PartA2_net = _module
detectors = _module
detector3d = _module
pointpillar = _module
second_net = _module
model_utils = _module
proposal_layer = _module
proposal_target_layer = _module
pytorch_utils = _module
resnet_utils = _module
rcnn = _module
partA2_rcnn_net = _module
rpn = _module
pillar_scatter = _module
rpn_backbone = _module
rpn_unet = _module
vfe = _module
vfe_utils = _module
iou3d_nms_utils = _module
setup = _module
roiaware_pool3d_utils = _module
utils = _module
box_coder_utils = _module
box_utils = _module
calibration = _module
common_utils = _module
loss_utils = _module
object3d_utils = _module
eval_utils = _module
test = _module
train = _module
optimization = _module
fastai_optim = _module
learning_schedules_fastai = _module
train_utils = _module

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


import torch


import torch.nn as nn


import numpy as np


from functools import partial


from collections import OrderedDict


from typing import List


from typing import Tuple


import torch.nn.functional as F


from torch.autograd import Function


from abc import ABCMeta


from abc import abstractmethod


import torch.distributed as dist


import torch.optim as optim


import torch.optim.lr_scheduler as lr_sched


from collections import Iterable


from torch import nn


from torch.nn.utils import parameters_to_vector


from torch._utils import _unflatten_dense_tensors


from torch.nn.utils import clip_grad_norm_


def center_to_minmax_2d_0_5(centers, dims):
    return np.concatenate([centers - dims / 2, centers + dims / 2], axis=-1)


def corners_nd(dims, origin=0.5):
    """generate relative box corners based on length per dim and
    origin point.

    Args:
        dims (float array, shape=[N, ndim]): array of length per dim
        origin (list or array or float): origin point relate to smallest point.

    Returns:
        float array, shape=[N, 2 ** ndim, ndim]: returned corners.
        point layout example: (2d) x0y0, x0y1, x1y0, x1y1;
            (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
            where x0 < x1, y0 < y1, z0 < z1
    """
    ndim = int(dims.shape[1])
    corners_norm = np.stack(np.unravel_index(np.arange(2 ** ndim), [2] *
        ndim), axis=1).astype(dims.dtype)
    if ndim == 2:
        corners_norm = corners_norm[[0, 1, 3, 2]]
    elif ndim == 3:
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array(origin, dtype=dims.dtype)
    corners = dims.reshape([-1, 1, ndim]) * corners_norm.reshape([1, 2 **
        ndim, ndim])
    return corners


def rotation_2d(points, angles):
    """rotation 2d points based on origin point clockwise when angle positive.

    Args:
        points (float array, shape=[N, point_size, 2]): points to be rotated.
        angles (float array, shape=[N]): rotation angle.

    Returns:
        float array: same shape as points
    """
    rot_sin = np.sin(angles)
    rot_cos = np.cos(angles)
    rot_mat_T = np.stack([[rot_cos, -rot_sin], [rot_sin, rot_cos]])
    return np.einsum('aij,jka->aik', points, rot_mat_T)


def center_to_corner_box2d(centers, dims, angles=None, origin=0.5):
    """convert kitti locations, dimensions and angles to corners.
    format: center(xy), dims(xy), angles(clockwise when positive)

    Args:
        centers (float array, shape=[N, 2]): locations in kitti label file.
        dims (float array, shape=[N, 2]): dimensions in kitti label file.
        angles (float array, shape=[N]): rotation_y in kitti label file.

    Returns:
        [type]: [description]
    """
    corners = corners_nd(dims, origin=origin)
    if angles is not None:
        corners = rotation_2d(corners, angles)
    corners += centers.reshape([-1, 1, 2])
    return corners


def center_to_minmax_2d(centers, dims, origin=0.5):
    if origin == 0.5:
        return center_to_minmax_2d_0_5(centers, dims)
    corners = center_to_corner_box2d(centers, dims, origin=origin)
    return corners[:, ([0, 2])].reshape([-1, 4])


def rbbox2d_to_near_bbox(rbboxes):
    """convert rotated bbox to nearest 'standing' or 'lying' bbox.
    Args:
        rbboxes: [N, 5(x, y, xdim, ydim, rad)] rotated bboxes
    Returns:
        bboxes: [N, 4(xmin, ymin, xmax, ymax)] bboxes
    """
    rots = rbboxes[..., -1]
    rots_0_pi_div_2 = np.abs(common_utils.limit_period(rots, 0.5, np.pi))
    cond = (rots_0_pi_div_2 > np.pi / 4)[..., np.newaxis]
    bboxes_center = np.where(cond, rbboxes[:, ([0, 1, 3, 2])], rbboxes[:, :4])
    bboxes = center_to_minmax_2d(bboxes_center[:, :2], bboxes_center[:, 2:])
    return bboxes


_global_config['LOCAL_RANK'] = 4


_global_config['MODEL'] = 4


class Empty(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super(Empty, self).__init__()

    def forward(self, *args, **kwargs):
        if len(args) == 1:
            return args[0]
        elif len(args) == 0:
            return None
        return args


class Sequential(torch.nn.Module):
    """A sequential container.
    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can also be passed in.

    To make it easier to understand, given is a small example::

        # Example of using Sequential
        model = Sequential(
                  nn.Conv2d(1,20,5),
                  nn.ReLU(),
                  nn.Conv2d(20,64,5),
                  nn.ReLU()
                )

        # Example of using Sequential with OrderedDict
        model = Sequential(OrderedDict([
                  ('conv1', nn.Conv2d(1,20,5)),
                  ('relu1', nn.ReLU()),
                  ('conv2', nn.Conv2d(20,64,5)),
                  ('relu2', nn.ReLU())
                ]))

        # Example of using Sequential with kwargs(python 3.6+)
        model = Sequential(
                  conv1=nn.Conv2d(1,20,5),
                  relu1=nn.ReLU(),
                  conv2=nn.Conv2d(20,64,5),
                  relu2=nn.ReLU()
                )
    """

    def __init__(self, *args, **kwargs):
        super(Sequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
        for name, module in kwargs.items():
            if sys.version_info < (3, 6):
                raise ValueError('kwargs only supported in py36+')
            if name in self._modules:
                raise ValueError('name exists.')
            self.add_module(name, module)

    def __getitem__(self, idx):
        if not -len(self) <= idx < len(self):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx += len(self)
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __len__(self):
        return len(self._modules)

    def add(self, module, name=None):
        if name is None:
            name = str(len(self._modules))
            if name in self._modules:
                raise KeyError('name exists')
        self.add_module(name, module)

    def forward(self, input):
        for module in self._modules.values():
            input = module(input)
        return input


class _ConvBase(nn.Sequential):

    def __init__(self, in_size, out_size, kernel_size, stride, padding,
        activation, bn, init, conv=None, batch_norm=None, bias=True, preact
        =False, name='', instance_norm=False, instance_norm_func=None):
        super().__init__()
        bias = bias and not bn
        conv_unit = conv(in_size, out_size, kernel_size=kernel_size, stride
            =stride, padding=padding, bias=bias)
        init(conv_unit.weight)
        if bias:
            nn.init.constant_(conv_unit.bias, 0)
        if bn:
            if not preact:
                bn_unit = batch_norm(out_size)
            else:
                bn_unit = batch_norm(in_size)
        if instance_norm:
            if not preact:
                in_unit = instance_norm_func(out_size, affine=False,
                    track_running_stats=False)
            else:
                in_unit = instance_norm_func(in_size, affine=False,
                    track_running_stats=False)
        if preact:
            if bn:
                self.add_module(name + 'bn', bn_unit)
            if activation is not None:
                self.add_module(name + 'activation', activation)
            if not bn and instance_norm:
                self.add_module(name + 'in', in_unit)
        self.add_module(name + 'conv', conv_unit)
        if not preact:
            if bn:
                self.add_module(name + 'bn', bn_unit)
            if activation is not None:
                self.add_module(name + 'activation', activation)
            if not bn and instance_norm:
                self.add_module(name + 'in', in_unit)


class _BNBase(nn.Sequential):

    def __init__(self, in_size, batch_norm=None, name=''):
        super().__init__()
        self.add_module(name + 'bn', batch_norm(in_size))
        nn.init.constant_(self[0].weight, 1.0)
        nn.init.constant_(self[0].bias, 0)


class BatchNorm1d(_BNBase):

    def __init__(self, in_size: int, *, name: str=''):
        super().__init__(in_size, batch_norm=nn.BatchNorm1d, name=name)


class FC(nn.Sequential):

    def __init__(self, in_size: int, out_size: int, *, activation=nn.ReLU(
        inplace=True), bn: bool=False, init=None, preact: bool=False, name:
        str=''):
        super().__init__()
        fc = nn.Linear(in_size, out_size, bias=not bn)
        if init is not None:
            init(fc.weight)
        if not bn:
            nn.init.constant(fc.bias, 0)
        if preact:
            if bn:
                self.add_module(name + 'bn', BatchNorm1d(in_size))
            if activation is not None:
                self.add_module(name + 'activation', activation)
        self.add_module(name + 'fc', fc)
        if not preact:
            if bn:
                self.add_module(name + 'bn', BatchNorm1d(out_size))
            if activation is not None:
                self.add_module(name + 'activation', activation)


def get_maxiou3d_with_same_class(rois, roi_labels, gt_boxes, gt_labels):
    """
    :param rois: (N, 7)
    :param roi_labels: (N)
    :param gt_boxes: (N, 8)
    :return:
    """
    max_overlaps = rois.new_zeros(rois.shape[0])
    gt_assignment = roi_labels.new_zeros(roi_labels.shape[0])
    for k in range(gt_labels.min().item(), gt_labels.max().item() + 1):
        roi_mask = roi_labels == k
        gt_mask = gt_labels == k
        if roi_mask.sum() > 0 and gt_mask.sum() > 0:
            cur_roi = rois[roi_mask]
            cur_gt = gt_boxes[gt_mask]
            original_gt_assignment = gt_mask.nonzero().view(-1)
            iou3d = iou3d_nms_utils.boxes_iou3d_gpu(cur_roi, cur_gt)
            cur_max_overlaps, cur_gt_assignment = torch.max(iou3d, dim=1)
            max_overlaps[roi_mask] = cur_max_overlaps
            gt_assignment[roi_mask] = original_gt_assignment[cur_gt_assignment]
    return max_overlaps, gt_assignment


def sample_bg_inds(hard_bg_inds, easy_bg_inds, bg_rois_per_this_image,
    roi_sampler_cfg):
    if hard_bg_inds.numel() > 0 and easy_bg_inds.numel() > 0:
        hard_bg_rois_num = int(bg_rois_per_this_image * roi_sampler_cfg.
            HARD_BG_RATIO)
        easy_bg_rois_num = bg_rois_per_this_image - hard_bg_rois_num
        rand_idx = torch.randint(low=0, high=hard_bg_inds.numel(), size=(
            hard_bg_rois_num,)).long()
        hard_bg_inds = hard_bg_inds[rand_idx]
        rand_idx = torch.randint(low=0, high=easy_bg_inds.numel(), size=(
            easy_bg_rois_num,)).long()
        easy_bg_inds = easy_bg_inds[rand_idx]
        bg_inds = torch.cat([hard_bg_inds, easy_bg_inds], dim=0)
    elif hard_bg_inds.numel() > 0 and easy_bg_inds.numel() == 0:
        hard_bg_rois_num = bg_rois_per_this_image
        rand_idx = torch.randint(low=0, high=hard_bg_inds.numel(), size=(
            hard_bg_rois_num,)).long()
        bg_inds = hard_bg_inds[rand_idx]
    elif hard_bg_inds.numel() == 0 and easy_bg_inds.numel() > 0:
        easy_bg_rois_num = bg_rois_per_this_image
        rand_idx = torch.randint(low=0, high=easy_bg_inds.numel(), size=(
            easy_bg_rois_num,)).long()
        bg_inds = easy_bg_inds[rand_idx]
    else:
        raise NotImplementedError
    return bg_inds


_global_config['CLASS_NAMES'] = 4


def sample_rois_for_rcnn(roi_boxes3d, gt_boxes3d, roi_raw_scores,
    roi_labels, roi_sampler_cfg):
    """
    :param roi_boxes3d: (B, M, 7 + ?) [x, y, z, w, l, h, ry] in LiDAR coords
    :param gt_boxes3d: (B, N, 7 + ? + 1) [x, y, z, w, l, h, ry, class]
    :param roi_raw_scores: (B, N)
    :param roi_labels: (B, N)
    :return
        batch_rois: (B, N, 7)
        batch_gt_of_rois: (B, N, 7 + 1)
        batch_roi_iou: (B, N)
    """
    batch_size = roi_boxes3d.size(0)
    fg_rois_per_image = int(np.round(roi_sampler_cfg.FG_RATIO *
        roi_sampler_cfg.ROI_PER_IMAGE))
    code_size = roi_boxes3d.shape[-1]
    batch_rois = gt_boxes3d.new(batch_size, roi_sampler_cfg.ROI_PER_IMAGE,
        code_size).zero_()
    batch_gt_of_rois = gt_boxes3d.new(batch_size, roi_sampler_cfg.
        ROI_PER_IMAGE, code_size + 1).zero_()
    batch_roi_iou = gt_boxes3d.new(batch_size, roi_sampler_cfg.ROI_PER_IMAGE
        ).zero_()
    batch_roi_raw_scores = gt_boxes3d.new(batch_size, roi_sampler_cfg.
        ROI_PER_IMAGE).zero_()
    batch_roi_labels = gt_boxes3d.new(batch_size, roi_sampler_cfg.ROI_PER_IMAGE
        ).zero_().long()
    for idx in range(batch_size):
        cur_roi, cur_gt, cur_roi_raw_scores, cur_roi_labels = roi_boxes3d[idx
            ], gt_boxes3d[idx], roi_raw_scores[idx], roi_labels[idx]
        k = cur_gt.__len__() - 1
        while k > 0 and cur_gt[k].sum() == 0:
            k -= 1
        cur_gt = cur_gt[:k + 1]
        if len(cfg.CLASS_NAMES) == 1:
            iou3d = iou3d_nms_utils.boxes_iou3d_gpu(cur_roi, cur_gt[:, 0:7])
            max_overlaps, gt_assignment = torch.max(iou3d, dim=1)
        else:
            cur_gt_labels = cur_gt[:, (-1)].long()
            max_overlaps, gt_assignment = get_maxiou3d_with_same_class(cur_roi,
                cur_roi_labels, cur_gt[:, 0:7], cur_gt_labels)
        fg_thresh = min(roi_sampler_cfg.REG_FG_THRESH, roi_sampler_cfg.
            CLS_FG_THRESH)
        fg_inds = torch.nonzero(max_overlaps >= fg_thresh).view(-1)
        easy_bg_inds = torch.nonzero(max_overlaps < roi_sampler_cfg.
            CLS_BG_THRESH_LO).view(-1)
        hard_bg_inds = torch.nonzero((max_overlaps < roi_sampler_cfg.
            REG_FG_THRESH) & (max_overlaps >= roi_sampler_cfg.CLS_BG_THRESH_LO)
            ).view(-1)
        fg_num_rois = fg_inds.numel()
        bg_num_rois = hard_bg_inds.numel() + easy_bg_inds.numel()
        if fg_num_rois > 0 and bg_num_rois > 0:
            fg_rois_per_this_image = min(fg_rois_per_image, fg_num_rois)
            rand_num = torch.from_numpy(np.random.permutation(fg_num_rois)
                ).type_as(gt_boxes3d).long()
            fg_inds = fg_inds[rand_num[:fg_rois_per_this_image]]
            bg_rois_per_this_image = (roi_sampler_cfg.ROI_PER_IMAGE -
                fg_rois_per_this_image)
            bg_inds = sample_bg_inds(hard_bg_inds, easy_bg_inds,
                bg_rois_per_this_image, roi_sampler_cfg)
        elif fg_num_rois > 0 and bg_num_rois == 0:
            rand_num = np.floor(np.random.rand(roi_sampler_cfg.
                ROI_PER_IMAGE) * fg_num_rois)
            rand_num = torch.from_numpy(rand_num).type_as(gt_boxes3d).long()
            fg_inds = fg_inds[rand_num]
            fg_rois_per_this_image = roi_sampler_cfg.ROI_PER_IMAGE
            bg_rois_per_this_image = 0
        elif bg_num_rois > 0 and fg_num_rois == 0:
            bg_rois_per_this_image = roi_sampler_cfg.ROI_PER_IMAGE
            bg_inds = sample_bg_inds(hard_bg_inds, easy_bg_inds,
                bg_rois_per_this_image, roi_sampler_cfg)
            fg_rois_per_this_image = 0
        else:
            print('maxoverlaps:(min=%f, max=%f)' % (max_overlaps.min().item
                (), max_overlaps.max().item()))
            print('ERROR: FG=%d, BG=%d' % (fg_num_rois, bg_num_rois))
            raise NotImplementedError
        (roi_list, roi_iou_list, roi_gt_list, roi_score_list, roi_labels_list
            ) = [], [], [], [], []
        if fg_rois_per_this_image > 0:
            fg_rois = cur_roi[fg_inds]
            gt_of_fg_rois = cur_gt[gt_assignment[fg_inds]]
            fg_iou3d = max_overlaps[fg_inds]
            roi_list.append(fg_rois)
            roi_iou_list.append(fg_iou3d)
            roi_gt_list.append(gt_of_fg_rois)
            roi_score_list.append(cur_roi_raw_scores[fg_inds])
            roi_labels_list.append(cur_roi_labels[fg_inds])
        if bg_rois_per_this_image > 0:
            bg_rois = cur_roi[bg_inds]
            gt_of_bg_rois = cur_gt[gt_assignment[bg_inds]]
            bg_iou3d = max_overlaps[bg_inds]
            roi_list.append(bg_rois)
            roi_iou_list.append(bg_iou3d)
            roi_gt_list.append(gt_of_bg_rois)
            roi_score_list.append(cur_roi_raw_scores[bg_inds])
            roi_labels_list.append(cur_roi_labels[bg_inds])
        rois = torch.cat(roi_list, dim=0)
        iou_of_rois = torch.cat(roi_iou_list, dim=0)
        gt_of_rois = torch.cat(roi_gt_list, dim=0)
        cur_roi_raw_scores = torch.cat(roi_score_list, dim=0)
        cur_roi_labels = torch.cat(roi_labels_list, dim=0)
        batch_rois[idx] = rois
        batch_gt_of_rois[idx] = gt_of_rois
        batch_roi_iou[idx] = iou_of_rois
        batch_roi_raw_scores[idx] = cur_roi_raw_scores
        batch_roi_labels[idx] = cur_roi_labels
    return (batch_rois, batch_gt_of_rois, batch_roi_iou,
        batch_roi_raw_scores, batch_roi_labels)


def proposal_target_layer(input_dict, roi_sampler_cfg):
    rois = input_dict['rois']
    roi_raw_scores = input_dict['roi_raw_scores']
    roi_labels = input_dict['roi_labels']
    gt_boxes = input_dict['gt_boxes']
    (batch_rois, batch_gt_of_rois, batch_roi_iou, batch_roi_raw_scores,
        batch_roi_labels) = (sample_rois_for_rcnn(rois, gt_boxes,
        roi_raw_scores, roi_labels, roi_sampler_cfg))
    reg_valid_mask = (batch_roi_iou > roi_sampler_cfg.REG_FG_THRESH).long()
    if roi_sampler_cfg.CLS_SCORE_TYPE == 'cls':
        batch_cls_label = (batch_roi_iou > roi_sampler_cfg.CLS_FG_THRESH).long(
            )
        invalid_mask = (batch_roi_iou > roi_sampler_cfg.CLS_BG_THRESH) & (
            batch_roi_iou < roi_sampler_cfg.CLS_FG_THRESH)
        batch_cls_label[invalid_mask > 0] = -1
    elif roi_sampler_cfg.CLS_SCORE_TYPE == 'roi_iou':
        fg_mask = batch_roi_iou > roi_sampler_cfg.CLS_FG_THRESH
        bg_mask = batch_roi_iou < roi_sampler_cfg.CLS_BG_THRESH
        interval_mask = (fg_mask == 0) & (bg_mask == 0)
        batch_cls_label = (fg_mask > 0).float()
        batch_cls_label[interval_mask] = batch_roi_iou[interval_mask] * 2 - 0.5
    else:
        raise NotImplementedError
    output_dict = {'rcnn_cls_labels': batch_cls_label.view(-1),
        'reg_valid_mask': reg_valid_mask.view(-1), 'gt_of_rois':
        batch_gt_of_rois, 'gt_iou': batch_roi_iou, 'rois': batch_rois,
        'roi_raw_scores': batch_roi_raw_scores, 'roi_labels': batch_roi_labels}
    return output_dict


class RCNNHead(nn.Module):

    def __init__(self, rcnn_target_config):
        super().__init__()
        self.forward_ret_dict = None
        self.rcnn_target_config = rcnn_target_config
        self.box_coder = getattr(box_coder_utils, rcnn_target_config.BOX_CODER
            )()
        losses_cfg = cfg.MODEL.LOSSES
        code_weights = losses_cfg.LOSS_WEIGHTS['code_weights']
        self.reg_loss_func = loss_utils.WeightedSmoothL1LocalizationLoss(sigma
            =3.0, code_weights=code_weights)

    def assign_targets(self, batch_size, rcnn_dict):
        with torch.no_grad():
            targets_dict = proposal_target_layer(rcnn_dict, roi_sampler_cfg
                =self.rcnn_target_config)
        rois = targets_dict['rois']
        gt_of_rois = targets_dict['gt_of_rois']
        targets_dict['gt_of_rois_src'] = gt_of_rois.clone().detach()
        roi_center = rois[:, :, 0:3]
        roi_ry = rois[:, :, (6)] % (2 * np.pi)
        gt_of_rois[:, :, 0:3] = gt_of_rois[:, :, 0:3] - roi_center
        gt_of_rois[:, :, (6)] = gt_of_rois[:, :, (6)] - roi_ry
        for k in range(batch_size):
            gt_of_rois[k] = common_utils.rotate_pc_along_z_torch(gt_of_rois
                [k].unsqueeze(dim=1), -(roi_ry[k] + np.pi / 2)).squeeze(dim=1)
        ry_label = gt_of_rois[:, :, (6)] % (2 * np.pi)
        opposite_flag = (ry_label > np.pi * 0.5) & (ry_label < np.pi * 1.5)
        ry_label[opposite_flag] = (ry_label[opposite_flag] + np.pi) % (2 *
            np.pi)
        flag = ry_label > np.pi
        ry_label[flag] = ry_label[flag] - np.pi * 2
        ry_label = torch.clamp(ry_label, min=-np.pi / 2, max=np.pi / 2)
        gt_of_rois[:, :, (6)] = ry_label
        targets_dict['gt_of_rois'] = gt_of_rois
        return targets_dict

    def get_loss(self, forward_ret_dict=None):
        loss_cfgs = cfg.MODEL.LOSSES
        LOSS_WEIGHTS = loss_cfgs.LOSS_WEIGHTS
        forward_ret_dict = (self.forward_ret_dict if forward_ret_dict is
            None else forward_ret_dict)
        code_size = self.box_coder.code_size
        rcnn_cls = forward_ret_dict['rcnn_cls']
        rcnn_cls_labels = forward_ret_dict['rcnn_cls_labels'].float().view(-1)
        reg_valid_mask = forward_ret_dict['reg_valid_mask']
        gt_boxes3d_ct = forward_ret_dict['gt_of_rois'][(...), 0:code_size]
        gt_of_rois_src = forward_ret_dict['gt_of_rois_src'][(...), 0:code_size
            ].view(-1, code_size)
        rcnn_reg = forward_ret_dict['rcnn_reg']
        roi_boxes3d = forward_ret_dict['rois']
        rcnn_batch_size = rcnn_cls_labels.shape[0]
        rcnn_loss = 0
        if loss_cfgs.RCNN_CLS_LOSS == 'BinaryCrossEntropy':
            rcnn_cls_flat = rcnn_cls.view(-1)
            batch_loss_cls = F.binary_cross_entropy(torch.sigmoid(
                rcnn_cls_flat), rcnn_cls_labels, reduction='none')
            cls_valid_mask = (rcnn_cls_labels >= 0).float()
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum(
                ) / torch.clamp(cls_valid_mask.sum(), min=1.0)
            rcnn_loss_cls = rcnn_loss_cls * LOSS_WEIGHTS['rcnn_cls_weight']
        else:
            raise NotImplementedError
        rcnn_loss += rcnn_loss_cls
        tb_dict = {'rcnn_loss_cls': rcnn_loss_cls.item()}
        fg_mask = reg_valid_mask > 0
        fg_sum = fg_mask.long().sum().item()
        if fg_sum == 0:
            temp_rcnn_reg = rcnn_reg.view(rcnn_batch_size, -1)[0].unsqueeze(dim
                =0)
            faked_reg_target = temp_rcnn_reg.detach()
            rcnn_loss_reg = self.reg_loss_func(temp_rcnn_reg, faked_reg_target)
            rcnn_loss_reg = rcnn_loss_reg.sum() / 1.0
            tb_dict['rcnn_loss_reg'] = rcnn_loss_reg.item()
        else:
            fg_rcnn_reg = rcnn_reg.view(rcnn_batch_size, -1)[fg_mask]
            fg_roi_boxes3d = roi_boxes3d.view(-1, code_size)[fg_mask]
            if loss_cfgs.RCNN_REG_LOSS == 'smooth-l1':
                rois_anchor = roi_boxes3d.clone().detach().view(-1, code_size)
                rois_anchor[:, 0:3] = 0
                rois_anchor[:, (6)] = 0
                reg_targets = self.box_coder.encode_torch(gt_boxes3d_ct.
                    view(rcnn_batch_size, code_size)[fg_mask], rois_anchor[
                    fg_mask])
                rcnn_loss_reg = self.reg_loss_func(rcnn_reg.view(
                    rcnn_batch_size, -1)[fg_mask].unsqueeze(dim=0),
                    reg_targets.unsqueeze(dim=0))
                rcnn_loss_reg = rcnn_loss_reg.sum() / max(fg_sum, 0)
                rcnn_loss_reg = rcnn_loss_reg * LOSS_WEIGHTS['rcnn_reg_weight']
                tb_dict['rcnn_loss_reg'] = rcnn_loss_reg.item()
                if loss_cfgs.CORNER_LOSS_REGULARIZATION:
                    fg_roi_boxes3d = fg_roi_boxes3d.view(1, -1, code_size)
                    batch_anchors = fg_roi_boxes3d.clone().detach()
                    roi_ry = fg_roi_boxes3d[:, :, (6)].view(-1)
                    roi_xyz = fg_roi_boxes3d[:, :, 0:3].view(-1, 3)
                    batch_anchors[:, :, 0:3] = 0
                    rcnn_boxes3d = self.box_coder.decode_torch(fg_rcnn_reg.
                        view(batch_anchors.shape[0], -1, code_size),
                        batch_anchors).view(-1, code_size)
                    rcnn_boxes3d = common_utils.rotate_pc_along_z_torch(
                        rcnn_boxes3d.unsqueeze(dim=1), roi_ry + np.pi / 2
                        ).squeeze(dim=1)
                    rcnn_boxes3d[:, 0:3] += roi_xyz
                    loss_corner = loss_utils.get_corner_loss_lidar(rcnn_boxes3d
                        [:, 0:7], gt_of_rois_src[fg_mask][:, 0:7])
                    loss_corner = loss_corner.mean()
                    loss_corner = loss_corner * LOSS_WEIGHTS[
                        'rcnn_corner_weight']
                    rcnn_loss_reg += loss_corner
                    tb_dict['rcnn_loss_corner'] = loss_corner
            else:
                raise NotImplementedError
        rcnn_loss += rcnn_loss_reg
        tb_dict['rcnn_loss'] = rcnn_loss.item()
        return rcnn_loss, tb_dict


class PointPillarsScatter(nn.Module):

    def __init__(self, input_channels=64, **kwargs):
        """
        Point Pillar's Scatter.
        Converts learned features from dense tensor to sparse pseudo image.
        :param output_shape: ([int]: 4). Required output shape of features.
        :param num_input_features: <int>. Number of input features.
        """
        super().__init__()
        self.nchannels = input_channels

    def forward(self, voxel_features, coords, batch_size, **kwargs):
        output_shape = kwargs['output_shape']
        nz, ny, nx = output_shape
        batch_canvas = []
        for batch_itt in range(batch_size):
            canvas = torch.zeros(self.nchannels, nz * nx * ny, dtype=
                voxel_features.dtype, device=voxel_features.device)
            batch_mask = coords[:, (0)] == batch_itt
            this_coords = coords[(batch_mask), :]
            indices = this_coords[:, (1)] * nz + this_coords[:, (2)
                ] * nx + this_coords[:, (3)]
            indices = indices.type(torch.long)
            voxels = voxel_features[(batch_mask), :]
            voxels = voxels.t()
            canvas[:, (indices)] = voxels
            batch_canvas.append(canvas)
        batch_canvas = torch.stack(batch_canvas, 0)
        batch_canvas = batch_canvas.view(batch_size, self.nchannels * nz,
            ny, nx)
        return batch_canvas


_global_config['DATA_CONFIG'] = 4


class BackBone8x(nn.Module):

    def __init__(self, input_channels):
        super().__init__()
        norm_fn = partial(nn.BatchNorm1d, eps=0.001, momentum=0.01)
        self.conv_input = spconv.SparseSequential(spconv.SubMConv3d(
            input_channels, 16, 3, padding=1, bias=False, indice_key=
            'subm1'), norm_fn(16), nn.ReLU())
        block = self.post_act_block
        self.conv1 = spconv.SparseSequential(block(16, 16, 3, norm_fn=
            norm_fn, padding=1, indice_key='subm1'))
        self.conv2 = spconv.SparseSequential(block(16, 32, 3, norm_fn=
            norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type=
            'spconv'), block(32, 32, 3, norm_fn=norm_fn, padding=1,
            indice_key='subm2'), block(32, 32, 3, norm_fn=norm_fn, padding=
            1, indice_key='subm2'))
        self.conv3 = spconv.SparseSequential(block(32, 64, 3, norm_fn=
            norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type=
            'spconv'), block(64, 64, 3, norm_fn=norm_fn, padding=1,
            indice_key='subm3'), block(64, 64, 3, norm_fn=norm_fn, padding=
            1, indice_key='subm3'))
        self.conv4 = spconv.SparseSequential(block(64, 64, 3, norm_fn=
            norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4',
            conv_type='spconv'), block(64, 64, 3, norm_fn=norm_fn, padding=
            1, indice_key='subm4'), block(64, 64, 3, norm_fn=norm_fn,
            padding=1, indice_key='subm4'))
        last_pad = 0 if cfg.DATA_CONFIG.VOXEL_GENERATOR.VOXEL_SIZE[-1] in [
            0.1, 0.2] else (1, 0, 0)
        self.conv_out = spconv.SparseSequential(spconv.SparseConv3d(64, 128,
            (3, 1, 1), stride=(2, 1, 1), padding=last_pad, bias=False,
            indice_key='spconv_down2'), norm_fn(128), nn.ReLU())

    def forward(self, input_sp_tensor, **kwargs):
        """
        :param voxel_features:  (N, C)
        :param coors:   (N, 4)  [batch_idx, z_idx, y_idx, x_idx],  sparse_shape: (z_size, y_size, x_size)
        :param batch_size:
        :return:
        """
        x = self.conv_input(input_sp_tensor)
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        out = self.conv_out(x_conv4)
        spatial_features = out.dense()
        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)
        ret = {'spatial_features': spatial_features}
        return ret

    def post_act_block(self, in_channels, out_channels, kernel_size,
        indice_key, stride=1, padding=0, conv_type='subm', norm_fn=None):
        if conv_type == 'subm':
            m = spconv.SparseSequential(spconv.SubMConv3d(in_channels,
                out_channels, kernel_size, bias=False, indice_key=
                indice_key), norm_fn(out_channels), nn.ReLU())
        elif conv_type == 'spconv':
            m = spconv.SparseSequential(spconv.SparseConv3d(in_channels,
                out_channels, kernel_size, stride=stride, padding=padding,
                bias=False, indice_key=indice_key), norm_fn(out_channels),
                nn.ReLU())
        elif conv_type == 'inverseconv':
            m = spconv.SparseSequential(spconv.SparseInverseConv3d(
                in_channels, out_channels, kernel_size, indice_key=
                indice_key, bias=False), norm_fn(out_channels), nn.ReLU())
        else:
            raise NotImplementedError
        return m


class UNetHead(nn.Module):

    def __init__(self, unet_target_cfg):
        super().__init__()
        self.gt_extend_width = unet_target_cfg.GT_EXTEND_WIDTH
        if 'MEAN_SIZE' in unet_target_cfg:
            self.mean_size = unet_target_cfg.MEAN_SIZE
        self.target_generated_on = unet_target_cfg.GENERATED_ON
        self.cls_loss_func = loss_utils.SigmoidFocalClassificationLoss(alpha
            =0.25, gamma=2.0)
        self.forward_ret_dict = None

    def assign_targets(self, batch_points, gt_boxes,
        generate_bbox_reg_labels=False):
        """
        :param points: [(N1, 3), (N2, 3), ...]
        :param gt_boxes: (B, M, 8)
        :param gt_classes: (B, M)
        :param gt_names: (B, M)
        :return:
        """
        batch_size = gt_boxes.shape[0]
        cls_labels_list, part_reg_labels_list, bbox_reg_labels_list = [], [], [
            ]
        for k in range(batch_size):
            if True or self.target_generated_on == 'head_cpu':
                (cur_cls_labels, cur_part_reg_labels, cur_bbox_reg_labels) = (
                    self.generate_part_targets_cpu(points=batch_points[k],
                    gt_boxes=gt_boxes[k][:, 0:7], gt_classes=gt_boxes[k][:,
                    (7)], generate_bbox_reg_labels=generate_bbox_reg_labels))
            else:
                raise NotImplementedError
            cls_labels_list.append(cur_cls_labels)
            part_reg_labels_list.append(cur_part_reg_labels)
            bbox_reg_labels_list.append(cur_bbox_reg_labels)
        cls_labels = torch.cat(cls_labels_list, dim=0)
        part_reg_labels = torch.cat(part_reg_labels_list, dim=0)
        bbox_reg_labels = torch.cat(bbox_reg_labels_list, dim=0
            ) if generate_bbox_reg_labels else None
        targets_dict = {'seg_labels': cls_labels, 'part_labels':
            part_reg_labels, 'bbox_reg_labels': bbox_reg_labels}
        return targets_dict

    def generate_part_targets_cpu(self, points, gt_boxes, gt_classes,
        generate_bbox_reg_labels=False):
        """
        :param voxel_centers: (N, 3) [x, y, z]
        :param gt_boxes: (M, 7) [x, y, z, w, l, h, ry] in LiDAR coords
        :return:
        """
        k = gt_boxes.__len__() - 1
        while k > 0 and gt_boxes[k].sum() == 0:
            k -= 1
        gt_boxes = gt_boxes[:k + 1]
        gt_classes = gt_classes[:k + 1]
        extend_gt_boxes = common_utils.enlarge_box3d(gt_boxes, extra_width=
            self.gt_extend_width)
        cls_labels = torch.zeros(points.shape[0]).int()
        part_reg_labels = torch.zeros((points.shape[0], 3)).float()
        bbox_reg_labels = torch.zeros((points.shape[0], 7)).float(
            ) if generate_bbox_reg_labels else None
        point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(points,
            gt_boxes).long()
        extend_point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(points
            , extend_gt_boxes).long()
        for k in range(gt_boxes.shape[0]):
            fg_pt_flag = point_indices[k] > 0
            fg_points = points[fg_pt_flag]
            cls_labels[fg_pt_flag] = gt_classes[k]
            fg_enlarge_flag = extend_point_indices[k] > 0
            ignore_flag = fg_pt_flag ^ fg_enlarge_flag
            cls_labels[ignore_flag] = -1
            transformed_points = fg_points - gt_boxes[(k), 0:3]
            transformed_points = common_utils.rotate_pc_along_z_torch(
                transformed_points.view(1, -1, 3), -gt_boxes[k, 6])
            part_reg_labels[fg_pt_flag] = transformed_points / gt_boxes[(k),
                3:6] + torch.tensor([0.5, 0.5, 0]).float()
            if generate_bbox_reg_labels:
                center3d = gt_boxes[(k), 0:3].clone()
                center3d[2] += gt_boxes[k][5] / 2
                bbox_reg_labels[(fg_pt_flag), 0:3] = center3d - fg_points
                bbox_reg_labels[fg_pt_flag, 6] = gt_boxes[k, 6]
                cur_mean_size = torch.tensor(self.mean_size[cfg.CLASS_NAMES
                    [gt_classes[k] - 1]])
                bbox_reg_labels[(fg_pt_flag), 3:6] = (gt_boxes[(k), 3:6] -
                    cur_mean_size) / cur_mean_size
        return cls_labels, part_reg_labels, bbox_reg_labels

    def get_loss(self, forward_ret_dict=None):
        forward_ret_dict = (self.forward_ret_dict if forward_ret_dict is
            None else forward_ret_dict)
        tb_dict = {}
        u_seg_preds = forward_ret_dict['u_seg_preds'].squeeze(dim=-1)
        u_reg_preds = forward_ret_dict['u_reg_preds']
        u_cls_labels, u_reg_labels = forward_ret_dict['seg_labels'
            ], forward_ret_dict['part_labels']
        u_cls_target = (u_cls_labels > 0).float()
        pos_mask = u_cls_labels > 0
        pos = pos_mask.float()
        neg = (u_cls_labels == 0).float()
        u_cls_weights = pos + neg
        pos_normalizer = pos.sum()
        u_cls_weights = u_cls_weights / torch.clamp(pos_normalizer, min=1.0)
        u_loss_cls = self.cls_loss_func(u_seg_preds, u_cls_target, weights=
            u_cls_weights)
        u_loss_cls_pos = (u_loss_cls * pos).sum()
        u_loss_cls_neg = (u_loss_cls * neg).sum()
        u_loss_cls = u_loss_cls.sum()
        loss_unet = u_loss_cls
        if pos_normalizer > 0:
            u_loss_reg = F.binary_cross_entropy(torch.sigmoid(u_reg_preds[
                pos_mask]), u_reg_labels[pos_mask])
            loss_unet += u_loss_reg
            tb_dict['rpn_u_loss_reg'] = u_loss_reg.item()
        tb_dict['rpn_loss_u_cls'] = u_loss_cls.item()
        tb_dict['rpn_loss_u_cls_pos'] = u_loss_cls_pos.item()
        tb_dict['rpn_loss_u_cls_neg'] = u_loss_cls_neg.item()
        tb_dict['rpn_loss_unet'] = loss_unet.item()
        tb_dict['rpn_pos_num'] = pos_normalizer.item()
        return loss_unet, tb_dict


class VoxelFeatureExtractor(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

    def get_output_feature_dim(self):
        raise NotImplementedError

    def forward(self, **kwargs):
        raise NotImplementedError


class PFNLayer(nn.Module):

    def __init__(self, in_channels, out_channels, use_norm=True, last_layer
        =False):
        """
        Pillar Feature Net Layer.
        The Pillar Feature Net could be composed of a series of these layers, but the PointPillars paper results only
        used a single PFNLayer.
        :param in_channels: <int>. Number of input channels.
        :param out_channels: <int>. Number of output channels.
        :param use_norm: <bool>. Whether to include BatchNorm.
        :param last_layer: <bool>. If last_layer, there is no concatenation of features.
        """
        super().__init__()
        self.name = 'PFNLayer'
        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels
        if use_norm:
            self.linear = nn.Linear(in_channels, self.units, bias=False)
            self.norm = nn.BatchNorm1d(self.units, eps=0.001, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, self.units, bias=True)
            self.norm = Empty(self.units)

    def forward(self, inputs):
        x = self.linear(inputs)
        total_points, voxel_points, channels = x.shape
        x = self.norm(x.view(-1, channels)).view(total_points, voxel_points,
            channels)
        x = F.relu(x)
        x_max = torch.max(x, dim=1, keepdim=True)[0]
        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


class RoIAwarePool3dFunction(Function):

    @staticmethod
    def forward(ctx, rois, pts, pts_feature, out_size, max_pts_each_voxel,
        pool_method):
        """
        :param rois: (N, 7) [x, y, z, w, l, h, ry] in LiDAR coordinate, (x, y, z) is the bottom center of rois
        :param pts: (npoints, 3)
        :param pts_feature: (npoints, C)
        :param out_size: int or tuple, like 7 or (7, 7, 7)
        :param pool_method: 'max' or 'avg'
        :return
            pooled_features: (N, out_x, out_y, out_z, C)
        """
        if isinstance(out_size, int):
            out_x = out_y = out_z = out_size
        else:
            assert len(out_size) == 3
            for k in range(3):
                assert isinstance(out_size[k], int)
            out_x, out_y, out_z = out_size
        num_rois = rois.shape[0]
        num_channels = pts_feature.shape[-1]
        num_pts = pts.shape[0]
        pooled_features = pts_feature.new_zeros((num_rois, out_x, out_y,
            out_z, num_channels))
        argmax = pts_feature.new_zeros((num_rois, out_x, out_y, out_z,
            num_channels), dtype=torch.int)
        pts_idx_of_voxels = pts_feature.new_zeros((num_rois, out_x, out_y,
            out_z, max_pts_each_voxel), dtype=torch.int)
        pool_method_map = {'max': 0, 'avg': 1}
        pool_method = pool_method_map[pool_method]
        roiaware_pool3d_cuda.forward(rois, pts, pts_feature, argmax,
            pts_idx_of_voxels, pooled_features, pool_method)
        ctx.roiaware_pool3d_for_backward = (pts_idx_of_voxels, argmax,
            pool_method, num_pts, num_channels)
        return pooled_features

    @staticmethod
    def backward(ctx, grad_out):
        """
        :param grad_out: (N, out_x, out_y, out_z, C)
        :return:
            grad_in: (npoints, C)
        """
        pts_idx_of_voxels, argmax, pool_method, num_pts, num_channels = (ctx
            .roiaware_pool3d_for_backward)
        grad_in = grad_out.new_zeros((num_pts, num_channels))
        roiaware_pool3d_cuda.backward(pts_idx_of_voxels, argmax, grad_out.
            contiguous(), grad_in, pool_method)
        return None, None, grad_in, None, None, None


class RoIAwarePool3d(nn.Module):

    def __init__(self, out_size, max_pts_each_voxel=128):
        super().__init__()
        self.out_size = out_size
        self.max_pts_each_voxel = max_pts_each_voxel

    def forward(self, rois, pts, pts_feature, pool_method='max'):
        assert pool_method in ['max', 'avg']
        return RoIAwarePool3dFunction.apply(rois, pts, pts_feature, self.
            out_size, self.max_pts_each_voxel, pool_method)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_sshaoshuai_PCDet(_paritybench_base):
    pass
    def test_000(self):
        self._check(BatchNorm1d(*[], **{'in_size': 4}), [torch.rand([4, 4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(Empty(*[], **{}), [], {})

    def test_002(self):
        self._check(FC(*[], **{'in_size': 4, 'out_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(PFNLayer(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4])], {})

    @_fails_compile()
    def test_004(self):
        self._check(Sequential(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

