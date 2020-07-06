import sys
_module = sys.modules[__name__]
del sys
i3d_kinetics400_3d_rgb_r50_c3d_inflate3x1x1_seg1_f32s2 = _module
i3d_kinetics400_3d_rgb_r50_c3d_inflate3x1x1_seg1_f32s2_video = _module
slowonly_kinetics400_se_rgb_r50_seg1_4x16_finetune = _module
slowonly_kinetics400_se_rgb_r50_seg1_4x16_scratch = _module
slowonly_kinetics400_se_rgb_r50_seg1_8x8_finetune = _module
slowonly_kinetics400_se_rgb_r50_seg1_8x8_scratch = _module
tsn_kinetics400_2d_rgb_r50_seg3_f1s1 = _module
tsn_flow_bninception = _module
tsn_rgb_bninception = _module
ava_fast_rcnn_nl_r50_c4_1x_kinetics_pretrain_crop = _module
ssn_thumos14_rgb_bn_inception = _module
build_file_list = _module
build_rawframes = _module
mmaction = _module
apis = _module
env = _module
train = _module
core = _module
anchor2d = _module
anchor_generator = _module
anchor_target = _module
bbox1d = _module
geometry = _module
bbox2d = _module
assign_sampling = _module
assigners = _module
assign_result = _module
base_assigner = _module
max_iou_assigner = _module
bbox_target = _module
geometry = _module
samplers = _module
base_sampler = _module
pseudo_sampler = _module
random_sampler = _module
sampling_result = _module
transforms = _module
evaluation = _module
accuracy = _module
ava_utils = _module
bbox_overlaps = _module
class_names = _module
eval_hooks = _module
localize_utils = _module
recall = _module
post_processing = _module
bbox_nms = _module
merge_augs = _module
utils = _module
dist_utils = _module
datasets = _module
ava_dataset = _module
feature_dataset = _module
lmdbframes_dataset = _module
loader = _module
build_loader = _module
sampler = _module
rawframes_dataset = _module
ssn_dataset = _module
utils = _module
video_dataset = _module
losses = _module
flow_losses = _module
losses = _module
ssn_losses = _module
models = _module
builder = _module
detectors = _module
base = _module
fast_rcnn = _module
faster_rcnn = _module
test_mixins = _module
two_stage = _module
SSN2D = _module
localizers = _module
base = _module
TSN2D = _module
TSN3D = _module
recognizers = _module
base = _module
registry = _module
anchor_heads = _module
anchor_head = _module
rpn_head = _module
backbones = _module
bninception = _module
inception_v1_i3d = _module
resnet = _module
resnet_i3d = _module
resnet_i3d_slowfast = _module
resnet_r3d = _module
resnet_s3d = _module
bbox_heads = _module
bbox_head = _module
cls_heads = _module
cls_head = _module
ssn_head = _module
flownets = _module
motionnet = _module
necks = _module
fpn = _module
roi_extractors = _module
single_level = _module
single_level_straight3d = _module
segmental_consensuses = _module
simple_consensus = _module
stpp = _module
shared_heads = _module
res_i3d_layer = _module
res_layer = _module
spatial_temporal_modules = _module
non_local = _module
simple_spatial_module = _module
simple_spatial_temporal_module = _module
slowfast_spatial_temporal_module = _module
conv_module = _module
nonlocal_block = _module
norm = _module
resnet_r3d_utils = _module
ops = _module
nms = _module
nms_wrapper = _module
setup = _module
resample2d_package = _module
resample2d = _module
setup = _module
roi_align = _module
functions = _module
roi_align = _module
gradcheck = _module
modules = _module
roi_align = _module
setup = _module
roi_pool = _module
roi_pool = _module
gradcheck = _module
roi_pool = _module
setup = _module
trajectory_conv_package = _module
gradcheck = _module
setup = _module
traj_conv = _module
misc = _module
ipcsn_kinetics400_se_rgb_r152_seg1_32x2 = _module
ircsn_kinetics400_se_rgb_r152_seg1_32x2 = _module
i3d_hmdb51_3d_tvl1_inception_v1_seg1_f64s1 = _module
i3d_kinetics400_3d_tvl1_inception_v1_seg1_f64s1 = _module
i3d_ucf101_3d_tvl1_inception_v1_seg1_f64s1 = _module
i3d_hmdb51_3d_rgb_inception_v1_seg1_f64s1 = _module
i3d_kinetics400_3d_rgb_inception_v1_seg1_f64s1 = _module
i3d_ucf101_3d_rgb_inception_v1_seg1_f64s1 = _module
r2plus1d_kinetics400_se_rgb_r34_seg1_32x2 = _module
r2plus1d_kinetics400_se_rgb_r34_seg1_8x8 = _module
slowfast_kinetics400_se_rgb_r50_seg1_4x16 = _module
slowonly_kinetics400_se_rgb_r101_seg1_8x8 = _module
slowonly_kinetics400_se_rgb_r50_seg1_4x16 = _module
slowonly_kinetics400_se_rgb_r50_seg1_8x8 = _module
eval_localize_results = _module
generate_lmdb = _module
test_detector = _module
test_localizer = _module
test_recognizer = _module
test_recognizer_heavy = _module
train_detector = _module
train_localizer = _module
train_recognizer = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import logging


import random


import numpy as np


import torch


import torch.distributed as dist


import torch.multiprocessing as mp


from collections import OrderedDict


from abc import ABCMeta


from abc import abstractmethod


import time


from torch.utils.data import Dataset


from torch._utils import _flatten_dense_tensors


from torch._utils import _unflatten_dense_tensors


from torch._utils import _take_tensors


from functools import partial


from torch.utils.data import DataLoader


import math


from torch.distributed import get_world_size


from torch.distributed import get_rank


from torch.utils.data.sampler import Sampler


from torch.utils.data import DistributedSampler as _DistributedSampler


import copy


from collections import Sequence


import torch.nn as nn


import torch.nn.functional as F


from torch import nn


import torch.utils.checkpoint as cp


import warnings


import string


import itertools


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CUDAExtension


from torch.nn.modules.module import Module


from torch.autograd import Function


from torch.autograd import Variable


from torch.autograd import gradcheck


from torch.nn.modules.utils import _triple


from functools import reduce


dataset_aliases = {'ava': ['ava', 'ava2.1', 'ava2.2']}


def get_classes(dataset):
    """Get class names of a dataset."""
    alias2name = {}
    for name, aliases in dataset_aliases.items():
        for alias in aliases:
            alias2name[alias] = name
    if mmcv.is_str(dataset):
        if dataset in alias2name:
            labels = eval(alias2name[dataset] + '_classes()')
        else:
            raise ValueError('Unrecognized dataset: {}'.format(dataset))
    else:
        raise TypeError('dataset must a str, but got {}'.format(type(dataset)))
    return labels


def tensor2video_snaps(tensor, mean=(0, 0, 0), std=(1, 1, 1), to_rgb=True):
    num_videos = tensor.size(0)
    num_frames = tensor.size(2)
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    video_snaps = []
    for vid_id in range(num_videos):
        img = tensor[(vid_id), :, (num_frames // 2), (...)].cpu().numpy().transpose(1, 2, 0)
        img = mmcv.imdenormalize(img, mean, std, to_bgr=to_rgb).astype(np.uint8)
        video_snaps.append(np.ascontiguousarray(img))
    return video_snaps


class BaseDetector(nn.Module):
    """Base class for detectors"""
    __metaclass__ = ABCMeta

    def __init__(self):
        super(BaseDetector, self).__init__()

    @property
    def with_neck(self):
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_shared_head(self):
        return hasattr(self, 'shared_head') and self.shared_head is not None

    @property
    def with_bbox(self):
        return hasattr(self, 'bbox_head') and self.bbox_head is not None

    @abstractmethod
    def extract_feat(self, img_group):
        pass

    def extract_feats(self, img_groups):
        assert isinstance(img_groups, list)
        for img_group in img_groups:
            yield self.extract_feat(img_group)

    @abstractmethod
    def forward_train(self, num_modalities, img_metas, **kwargs):
        pass

    @abstractmethod
    def simple_test(self, num_modalities, img_metas, **kwargs):
        pass

    @abstractmethod
    def aug_test(self, num_modalities, img_metas, **kwargs):
        pass

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            logger = logging.getLogger()
            logger.info('load model from: {}'.format(pretrained))

    def forward_test(self, num_modalities, img_metas, **kwargs):
        if not isinstance(img_metas, list):
            raise TypeError('{} must be a list, but got {}'.format(img_metas, type(img_metas)))
        num_augs = len(kwargs['img_group_0'])
        if num_augs != len(img_metas):
            raise ValueError('num of augmentations ({}) != num of image meta ({})'.format(num_augs, len(img_metas)))
        videos_per_gpu = kwargs['img_group_0'][0].size(0)
        assert videos_per_gpu == 1
        if num_augs == 1:
            return self.simple_test(num_modalities, img_metas, **kwargs)
        else:
            return self.aug_test(num_modalities, img_metas, **kwargs)

    def forward(self, num_modalities, img_meta, return_loss=True, **kwargs):
        num_modalities = int(num_modalities[0])
        if return_loss:
            return self.forward_train(num_modalities, img_meta, **kwargs)
        else:
            return self.forward_test(num_modalities, img_meta, **kwargs)

    def show_result(self, data, bbox_result, img_norm_cfg, dataset='ava', score_thr=0.3):
        img_group_tensor = data['img_group_0'][0]
        img_metas = data['img_meta'][0].data[0]
        imgs = tensor2video_snaps(img_group_tensor, **img_norm_cfg)
        assert len(imgs) == len(img_metas)
        if isinstance(dataset, str):
            class_names = get_classes(dataset)
        elif isinstance(dataset, (list, tuple)) or dataset is None:
            class_names = dataset
        else:
            raise TypeError('dataset must be a valid dataset name or a sequence of class names, not {}'.format(type(dataset)))
        for img, img_meta in zip(imgs, img_metas):
            h, w, _ = img_meta['img_shape']
            img_show = img[:h, :w, :]
            bboxes = np.vstack(bbox_result)
            labels = [np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(bbox_result)]
            labels = np.concatenate(labels)
            mmcv.imshow_det_bboxes(img_show, bboxes, labels, class_names=class_names, score_thr=score_thr)


def bbox2roi(bbox_list):
    """Convert a list of bboxes to roi format

    Args:
        bbox_list (list[Tensor]): a list of bboxes
        corresponding to a batch of images

    Returns:
        Tensor: shape (n, 5), [batch_ind, x1, y1, x2, y2]
    """
    rois_list = []
    for img_id, bboxes in enumerate(bbox_list):
        if bboxes.size(0) > 0:
            img_inds = bboxes.new_full((bboxes.size(0), 1), img_id)
            rois = torch.cat([img_inds, bboxes[:, :4]], dim=-1)
        else:
            rois = bboxes.new_zeros((0, 5))
        rois_list.append(rois)
    rois = torch.cat(rois_list, 0)
    return rois


def bbox_flip(bboxes, img_shape):
    """Flip bboxes horizontally.

    Args:
        bboxes(Tensor or ndarray): Shape (..., 4*k)
        img_shape(tuple): Image shape.

    Returns:
        Same type as `bboxes`: Flipped bboxes.
    """
    if isinstance(bboxes, torch.Tensor):
        assert bboxes.shape[-1] % 4 == 0
        flipped = bboxes.clone()
        flipped[:, 0::4] = img_shape[1] - bboxes[:, 2::4] - 1
        flipped[:, 2::4] = img_shape[1] - bboxes[:, 0::4] - 1
        return flipped
    elif isinstance(bboxes, np.ndarray):
        return mmcv.bbox_flip(bboxes, img_shape)


def bbox_mapping(bboxes, img_shape, scale_factor, flip):
    """Map bboxes from the original image scale to testing scale"""
    new_bboxes = bboxes * scale_factor
    if flip:
        new_bboxes = bbox_flip(new_bboxes, img_shape)
    return new_bboxes


def bbox_mapping_back(bboxes, img_shape, scale_factor, flip):
    """Map bboxes from testing scale to original image scale"""
    new_bboxes = bbox_flip(bboxes, img_shape) if flip else bboxes
    new_bboxes = new_bboxes / scale_factor
    return new_bboxes


def merge_aug_bboxes(aug_bboxes, aug_scores, img_metas, rcnn_test_cfg):
    """Merge augmented detection bboxes and scores.

    Args:
        aug_bboxes (list[Tensor]): shape (n, 4*#class)
        aug_scores (list[Tensor] or None): shape (n, #class)
        img_shapes (list[Tensor]): shape (3, ).
        rcnn_test_cfg (dict): rcnn test config.

    Returns:
        tuple: (bboxes, scores)
    """
    recovered_bboxes = []
    for bboxes, img_info in zip(aug_bboxes, img_metas):
        img_shape = img_info[0]['img_shape']
        scale_factor = img_info[0]['scale_factor']
        flip = img_info[0]['flip']
        bboxes = bbox_mapping_back(bboxes, img_shape, scale_factor, flip)
        recovered_bboxes.append(bboxes)
    bboxes = torch.stack(recovered_bboxes).mean(dim=0)
    if aug_scores is None:
        return bboxes
    else:
        scores = torch.stack(aug_scores).mean(dim=0)
        return bboxes, scores


def multiclass_nms(multi_bboxes, multi_scores, score_thr, nms_cfg, max_num=-1):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class)
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). labels
            are 0-based.
    """
    num_classes = multi_scores.shape[1]
    bboxes, labels = [], []
    nms_cfg_ = nms_cfg.copy()
    nms_type = nms_cfg_.pop('type', 'nms')
    nms_op = getattr(nms_wrapper, nms_type)
    for i in range(1, num_classes):
        cls_inds = multi_scores[:, (i)] > score_thr
        if not cls_inds.any():
            continue
        if multi_bboxes.shape[1] == 4:
            _bboxes = multi_bboxes[(cls_inds), :]
        else:
            _bboxes = multi_bboxes[(cls_inds), i * 4:(i + 1) * 4]
        _scores = multi_scores[cls_inds, i]
        cls_dets = torch.cat([_bboxes, _scores[:, (None)]], dim=1)
        cls_dets, _ = nms_op(cls_dets, **nms_cfg_)
        cls_labels = multi_bboxes.new_full((cls_dets.shape[0],), i - 1, dtype=torch.long)
        bboxes.append(cls_dets)
        labels.append(cls_labels)
    if bboxes:
        bboxes = torch.cat(bboxes)
        labels = torch.cat(labels)
        if bboxes.shape[0] > max_num:
            _, inds = bboxes[:, (-1)].sort(descending=True)
            inds = inds[:max_num]
            bboxes = bboxes[inds]
            labels = labels[inds]
    else:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0,), dtype=torch.long)
    return bboxes, labels


class BBoxTestMixin(object):

    def simple_test_bboxes(self, x, img_meta, proposals, rcnn_test_cfg, rescale=False):
        """Test only det bboxes without augmentation."""
        rois = bbox2roi(proposals)
        roi_feats = self.bbox_roi_extractor(x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
        if self.with_shared_head:
            roi_feats = self.shared_head(roi_feats)
        cls_score, bbox_pred = self.bbox_head(roi_feats)
        img_shape = img_meta[0]['img_shape']
        scale_factor = img_meta[0]['scale_factor']
        det_bboxes, det_labels = self.bbox_head.get_det_bboxes(rois, cls_score, bbox_pred, img_shape, scale_factor, rescale=rescale, cfg=rcnn_test_cfg, crop_quadruple=img_meta[0]['crop_quadruple'])
        return det_bboxes, det_labels

    def aug_test_bboxes(self, feats, img_metas, proposal_list, rcnn_test_cfg):
        aug_bboxes = []
        aug_scores = []
        for x, img_meta in zip(feats, img_metas):
            img_shape = img_meta[0]['img_shape']
            scale_factor = img_meta[0]['scale_factor']
            flip = img_meta[0]['flip']
            proposals = bbox_mapping(proposal_list[0][:, :4], img_shape, scale_factor, flip)
            rois = bbox2roi([proposals])
            roi_feats = self.bbox_roi_extractor(x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
            if self.with_shared_head:
                roi_feats = self.shared_head(roi_feats)
            cls_score, bbox_pred = self.bbox_head(roi_feats)
            bboxes, scores = self.bbox_head.get_det_bboxes(rois, cls_score, bbox_pred, img_shape, scale_factor, rescale=False, cfg=None)
            aug_bboxes.append(bboxes)
            aug_scores.append(scores)
        merged_bboxes, merged_scores = merge_aug_bboxes(aug_bboxes, aug_scores, img_metas, rcnn_test_cfg)
        det_bboxes, det_labels = multiclass_nms(merged_bboxes, merged_scores, rcnn_test_cfg.score_thr, rcnn_test_cfg.nms, rcnn_test_cfg.max_per_img)
        return det_bboxes, det_labels


class Registry(object):

    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def _register_module(self, module_class):
        """Register a module

        Args:
            module (:obj:`nn.Module`): Module to be registered.
        """
        if not issubclass(module_class, nn.Module):
            raise TypeError('module must be a child of nn.Module, but got {}'.format(module_class))
        module_name = module_class.__name__
        if module_name in self._module_dict:
            raise KeyError('{} is already registered in {}'.format(module_name, self.name))
        self._module_dict[module_name] = module_class

    def register_module(self, cls):
        self._register_module(cls)
        return cls


DETECTORS = Registry('detector')


def nms(dets, iou_thr, device_id=None):
    """Dispatch to either CPU or GPU NMS implementations.

    The input can be either a torch tensor or numpy array. GPU NMS will be used
    if the input is a gpu tensor or device_id is specified, otherwise CPU NMS
    will be used. The returned type will always be the same as inputs.

    Arguments:
        dets (torch.Tensor or np.ndarray): bboxes with scores.
        iou_thr (float): IoU threshold for NMS.
        device_id (int, optional): when `dets` is a numpy array, if `device_id`
            is None, then cpu nms is used, otherwise gpu_nms will be used.

    Returns:
        tuple: kept bboxes and indice, which is always the same data type as
            the input.
    """
    if isinstance(dets, torch.Tensor):
        is_numpy = False
        dets_th = dets
    elif isinstance(dets, np.ndarray):
        is_numpy = True
        device = 'cpu' if device_id is None else 'cuda:{}'.format(device_id)
        dets_th = torch.from_numpy(dets)
    else:
        raise TypeError('dets must be either a Tensor or numpy array, but got {}'.format(type(dets)))
    if dets_th.shape[0] == 0:
        inds = dets_th.new_zeros(0, dtype=torch.long)
    elif dets_th.is_cuda:
        inds = nms_cuda.nms(dets_th, iou_thr)
    else:
        inds = nms_cpu.nms(dets_th, iou_thr)
    if is_numpy:
        inds = inds.cpu().numpy()
    return dets[(inds), :], inds


def merge_aug_proposals(aug_proposals, img_metas, rpn_test_cfg):
    """Merge augmented proposals (multiscale, flip, etc.)

    Args:
        aug_proposals (list[Tensor]): proposals from different testing
            schemes, shape (n, 5). Note that they are not rescaled to the
            original image size.
        img_metas (list[dict]): image info including "shape_scale" and "flip".
        rpn_test_cfg (dict): rpn test config.

    Returns:
        Tensor: shape (n, 4), proposals corresponding to original image scale.
    """
    recovered_proposals = []
    for proposals, img_info in zip(aug_proposals, img_metas):
        img_shape = img_info['img_shape']
        scale_factor = img_info['scale_factor']
        flip = img_info['flip']
        _proposals = proposals.clone()
        _proposals[:, :4] = bbox_mapping_back(_proposals[:, :4], img_shape, scale_factor, flip)
        recovered_proposals.append(_proposals)
    aug_proposals = torch.cat(recovered_proposals, dim=0)
    merged_proposals, _ = nms(aug_proposals, rpn_test_cfg.nms_thr)
    scores = merged_proposals[:, (4)]
    _, order = scores.sort(0, descending=True)
    num = min(rpn_test_cfg.max_num, merged_proposals.shape[0])
    order = order[:num]
    merged_proposals = merged_proposals[(order), :]
    return merged_proposals


class RPNTestMixin(object):

    def simple_test_rpn(self, x, img_meta, rpn_test_cfg):
        x_slice = (xx[:, :, (xx.size(2) // 2), :, :] for xx in x)
        rpn_outs = self.rpn_head(x_slice)
        proposal_inputs = rpn_outs + (img_meta, rpn_test_cfg)
        proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        return proposal_list

    def aug_test_rpn(self, feats, img_metas, rpn_test_cfg):
        imgs_per_gpu = len(img_metas[0])
        aug_proposals = [[] for _ in range(imgs_per_gpu)]
        for x, img_meta in zip(feats, img_metas):
            proposal_list = self.simple_test_rpn(x, img_meta, rpn_test_cfg)
            for i, proposals in enumerate(proposal_list):
                aug_proposals[i].append(proposals)
        merged_proposals = [merge_aug_proposals(proposals, img_meta, rpn_test_cfg) for proposals, img_meta in zip(aug_proposals, img_metas)]
        return merged_proposals


def bbox2result(bboxes, labels, num_classes, thr=0.01):
    """Convert detection results to a list of numpy arrays.
    Args:
        bboxes (Tensor): shape (n, 5)
        labels (Tensor): shape (n, ) or shape (n, #num_classes)
        num_classes (int): class number, including background class
    Returns:
        list(ndarray): bbox results of each class
    """
    if bboxes.shape[0] == 0:
        return [np.zeros((0, 5), dtype=np.float32) for i in range(num_classes - 1)]
    else:
        bboxes = bboxes.cpu().numpy()
        labels = labels.cpu().numpy()
        if labels.ndim == 1:
            return [bboxes[(labels == i), :] for i in range(num_classes - 1)]
        else:
            scores = labels
            thr = (thr,) * num_classes if isinstance(thr, float) else thr
            assert scores.shape[1] == num_classes
            assert len(thr) == num_classes
            result = []
            for i in range(num_classes - 1):
                where = scores[:, (i + 1)] > thr[i + 1]
                result.append(np.concatenate((bboxes[(where), :4], scores[(where), i + 1:i + 2]), axis=1))
            return result


def build_assigner(cfg, **kwargs):
    if isinstance(cfg, assigners.BaseAssigner):
        return cfg
    elif isinstance(cfg, dict):
        return mmcv.runner.obj_from_dict(cfg, assigners, default_args=kwargs)
    else:
        raise TypeError('Invalid type {} for building a sampler'.format(type(cfg)))


def build_sampler(cfg, **kwargs):
    if isinstance(cfg, samplers.BaseSampler):
        return cfg
    elif isinstance(cfg, dict):
        return mmcv.runner.obj_from_dict(cfg, samplers, default_args=kwargs)
    else:
        raise TypeError('Invalid type {} for building a sampler'.format(type(cfg)))


class TwoStageDetector(BaseDetector, RPNTestMixin, BBoxTestMixin):

    def __init__(self, backbone, neck=None, shared_head=None, rpn_head=None, bbox_roi_extractor=None, dropout_ratio=0, bbox_head=None, train_cfg=None, test_cfg=None, pretrained=None):
        super(TwoStageDetector, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        if shared_head is not None:
            self.shared_head = builder.build_head(shared_head)
        if rpn_head is not None:
            self.rpn_head = builder.build_head(rpn_head)
        if bbox_head is not None:
            self.bbox_roi_extractor = builder.build_roi_extractor(bbox_roi_extractor)
            self.bbox_head = builder.build_head(bbox_head)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout(p=dropout_ratio)
        else:
            self.dropout = None
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights()

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    def init_weights(self):
        super(TwoStageDetector, self).init_weights()
        self.backbone.init_weights()
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
        if self.with_shared_head:
            self.shared_head.init_weights()
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_bbox:
            self.bbox_roi_extractor.init_weights()
            self.bbox_head.init_weights()

    def extract_feat(self, image_group):
        x = self.backbone(image_group)
        if self.with_neck:
            x = self.neck()
        elif not isinstance(x, (list, tuple)):
            x = x,
        return x

    def forward_train(self, num_modalities, img_meta, gt_bboxes, gt_labels, gt_bboxes_ignore=None, proposals=None, **kwargs):
        assert num_modalities == 1
        img_group = kwargs['img_group_0']
        x = self.extract_feat(img_group)
        losses = dict()
        if self.with_rpn:
            x_slice = (xx[:, :, (xx.size(2) // 2), :, :] for xx in x)
            rpn_outs = self.rpn_head(x_slice)
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta, self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(*rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)
            proposal_inputs = rpn_outs + (img_meta, self.test_cfg.rpn)
            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        else:
            proposal_list = proposals
        if not self.train_cfg.train_detector:
            proposal_list = []
            for proposal in proposals:
                select_inds = proposal[:, (4)] >= min(self.train_cfg.person_det_score_thr, max(proposal[:, (4)]))
                proposal_list.append(proposal[select_inds])
        if self.with_bbox:
            bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
            bbox_sampler = build_sampler(self.train_cfg.rcnn.sampler, context=self)
            num_imgs = img_group.size(0)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = bbox_assigner.assign(proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i], gt_labels[i])
                sampling_result = bbox_sampler.sample(assign_result, proposal_list[i], gt_bboxes[i], gt_labels[i], feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)
        if self.with_bbox:
            rois = bbox2roi([res.bboxes for res in sampling_results])
            bbox_feats = self.bbox_roi_extractor(x[:self.bbox_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
            if self.dropout is not None:
                bbox_feats = self.dropout(bbox_feats)
            cls_score, bbox_pred = self.bbox_head(bbox_feats)
            bbox_targets = self.bbox_head.get_target(sampling_results, gt_bboxes, gt_labels, self.train_cfg.rcnn)
            loss_bbox = self.bbox_head.loss(cls_score, bbox_pred, *bbox_targets)
            if not self.train_cfg.train_detector:
                loss_bbox.pop('loss_person_cls')
            losses.update(loss_bbox)
        return losses

    def simple_test(self, num_modalities, img_meta, proposals=None, rescale=False, **kwargs):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        assert num_modalities == 1
        img_group = kwargs['img_group_0'][0]
        x = self.extract_feat(img_group)
        if proposals is None:
            proposal_list = self.simple_test_rpn(x, img_meta, self.test_cfg.rpn)
        else:
            proposal_list = []
            for proposal in proposals:
                proposal = proposal[0, ...]
                if not self.test_cfg.train_detector:
                    select_inds = proposal[:, (4)] >= min(self.test_cfg.person_det_score_thr, max(proposal[:, (4)]))
                    proposal = proposal[select_inds]
                proposal_list.append(proposal)
        img_meta = img_meta[0]
        det_bboxes, det_labels = self.simple_test_bboxes(x, img_meta, proposal_list, self.test_cfg.rcnn, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes, thr=self.test_cfg.rcnn.action_thr)
        return bbox_results

    def aug_test(self, num_modalities, img_metas, proposals=None, rescale=False, **kwargs):
        """Test with augmentations.

        If rescale is False, then returned bboxes will fit the scale
        of imgs[0]
        """
        assert num_modalities == 1
        img_groups = kwargs['img_group_0']
        if proposals is None:
            proposal_list = self.aug_test_rpn(self.extract_feats(img_groups), img_metas, self.test_cfg.rpn)
        else:
            proposal_list = []
            for proposal in proposals:
                proposal = proposal[0, ...]
                if not self.test_cfg.train_detector:
                    select_inds = proposal[:, (4)] >= min(self.test_cfg.person_det_score_thr, max(proposal[:, (4)]))
                    proposal = proposal[select_inds]
                proposal_list.append(proposal)
        det_bboxes, det_labels = self.aug_test_bboxes(self.extract_feats(img_groups), img_metas, proposal_list, self.test_cfg.rcnn)
        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= img_metas[0][0]['scale_factor']
        bbox_results = bbox2result(_det_bboxes, det_labels, self.bbox_head.num_classes, thr=self.test_cfg.rcnn.action_thr)
        return bbox_results


class BaseLocalizer(nn.Module):
    """Base class for localizers"""
    __metaclass__ = ABCMeta

    def __init__(self):
        super(BaseLocalizer, self).__init__()

    @abstractmethod
    def forward_train(self, num_modalities, **kwargs):
        pass

    @abstractmethod
    def forward_test(self, num_modalities, **kwargs):
        pass

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            logger = logging.getLogger()
            logger.info('load model from: {}'.format(pretrained))

    def forward(self, num_modalities, img_meta, return_loss=True, **kwargs):
        num_modalities = int(num_modalities[0])
        if return_loss:
            return self.forward_train(num_modalities, img_meta, **kwargs)
        else:
            return self.forward_test(num_modalities, img_meta, **kwargs)


class BaseRecognizer(nn.Module):
    """Base class for recognizers"""
    __metaclass__ = ABCMeta

    def __init__(self):
        super(BaseRecognizer, self).__init__()

    @property
    def with_tenon_list(self):
        return hasattr(self, 'tenon_list') and self.tenon_list is not None

    @property
    def with_cls(self):
        return hasattr(self, 'cls_head') and self.cls_head is not None

    @abstractmethod
    def forward_train(self, num_modalities, **kwargs):
        pass

    @abstractmethod
    def forward_test(self, num_modalities, **kwargs):
        pass

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            logger = logging.getLogger()
            logger.info('load model from: {}'.format(pretrained))

    def forward(self, num_modalities, img_meta, return_loss=True, **kwargs):
        num_modalities = int(num_modalities[0])
        if return_loss:
            return self.forward_train(num_modalities, img_meta, **kwargs)
        else:
            return self.forward_test(num_modalities, img_meta, **kwargs)


class AnchorGenerator(object):

    def __init__(self, base_size, scales, ratios, scale_major=True, ctr=None):
        self.base_size = base_size
        self.scales = torch.Tensor(scales)
        self.ratios = torch.Tensor(ratios)
        self.scale_major = scale_major
        self.ctr = ctr
        self.base_anchors = self.gen_base_anchors()

    @property
    def num_base_anchors(self):
        return self.base_anchors.size(0)

    def gen_base_anchors(self):
        w = self.base_size
        h = self.base_size
        if self.ctr is None:
            x_ctr = 0.5 * (w - 1)
            y_ctr = 0.5 * (h - 1)
        else:
            x_ctr, y_ctr = self.ctr
        h_ratios = torch.sqrt(self.ratios)
        w_ratios = 1 / h_ratios
        if self.scale_major:
            ws = (w * w_ratios[:, (None)] * self.scales[(None), :]).view(-1)
            hs = (h * h_ratios[:, (None)] * self.scales[(None), :]).view(-1)
        else:
            ws = (w * self.scales[:, (None)] * w_ratios[(None), :]).view(-1)
            hs = (h * self.scales[:, (None)] * h_ratios[(None), :]).view(-1)
        base_anchors = torch.stack([x_ctr - 0.5 * (ws - 1), y_ctr - 0.5 * (hs - 1), x_ctr + 0.5 * (ws - 1), y_ctr + 0.5 * (hs - 1)], dim=-1).round()
        return base_anchors

    def _meshgrid(self, x, y, row_major=True):
        xx = x.repeat(len(y))
        yy = y.view(-1, 1).repeat(1, len(x)).view(-1)
        if row_major:
            return xx, yy
        else:
            return yy, xx

    def grid_anchors(self, featmap_size, stride=16, device='cuda'):
        base_anchors = self.base_anchors
        feat_h, feat_w = featmap_size
        shift_x = torch.arange(0, feat_w, device=device) * stride
        shift_y = torch.arange(0, feat_h, device=device) * stride
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
        shifts = shifts.type_as(base_anchors)
        all_anchors = base_anchors[(None), :, :] + shifts[:, (None), :]
        all_anchors = all_anchors.view(-1, 4)
        return all_anchors

    def valid_flags(self, featmap_size, valid_size, device='cuda'):
        feat_h, feat_w = featmap_size
        valid_h, valid_w = valid_size
        assert valid_h <= feat_h and valid_w <= feat_w
        valid_x = torch.zeros(feat_w, dtype=torch.uint8, device=device)
        valid_y = torch.zeros(feat_h, dtype=torch.uint8, device=device)
        valid_x[:valid_w] = 1
        valid_y[:valid_h] = 1
        valid_xx, valid_yy = self._meshgrid(valid_x, valid_y)
        valid = valid_xx & valid_yy
        valid = valid[:, (None)].expand(valid.size(0), self.num_base_anchors).contiguous().view(-1)
        return valid


HEADS = Registry('head')


class SamplingResult(object):

    def __init__(self, pos_inds, neg_inds, bboxes, gt_bboxes, assign_result, gt_flags):
        self.pos_inds = pos_inds
        self.neg_inds = neg_inds
        self.pos_bboxes = bboxes[pos_inds]
        self.neg_bboxes = bboxes[neg_inds]
        self.pos_is_gt = gt_flags[pos_inds]
        self.num_gts = gt_bboxes.shape[0]
        self.pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1
        self.pos_gt_bboxes = gt_bboxes[(self.pos_assigned_gt_inds), :]
        if assign_result.labels is not None:
            self.pos_gt_labels = assign_result.labels[pos_inds]
        else:
            self.pos_gt_labels = None

    @property
    def bboxes(self):
        return torch.cat([self.pos_bboxes, self.neg_bboxes])


class BaseSampler(metaclass=ABCMeta):

    def __init__(self, num, pos_fraction, neg_pos_ub=-1, add_gt_as_proposals=True, **kwargs):
        self.num = num
        self.pos_fraction = pos_fraction
        self.neg_pos_ub = neg_pos_ub
        self.add_gt_as_proposals = add_gt_as_proposals
        self.pos_sampler = self
        self.neg_sampler = self

    @abstractmethod
    def _sample_pos(self, assign_result, num_expected, **kwargs):
        pass

    @abstractmethod
    def _sample_neg(self, assign_result, num_expected, **kwargs):
        pass

    def sample(self, assign_result, bboxes, gt_bboxes, gt_labels=None, **kwargs):
        """Sample positive and negative bboxes.
        This is a simple implementation of bbox sampling given candidates,
        assigning results and ground truth bboxes.
        Args:
            assign_result (:obj:`AssignResult`): Bbox assigning results.
            bboxes (Tensor): Boxes to be sampled from.
            gt_bboxes (Tensor): Ground truth bboxes.
            gt_labels (Tensor, optional): Class labels of ground truth bboxes.
        Returns:
            :obj:`SamplingResult`: Sampling result.
        """
        bboxes = bboxes[:, :4]
        gt_flags = bboxes.new_zeros((bboxes.shape[0],), dtype=torch.uint8)
        if self.add_gt_as_proposals:
            bboxes = torch.cat([gt_bboxes, bboxes], dim=0)
            assign_result.add_gt_(gt_labels)
            gt_ones = bboxes.new_ones(gt_bboxes.shape[0], dtype=torch.uint8)
            gt_flags = torch.cat([gt_ones, gt_flags])
        num_expected_pos = int(self.num * self.pos_fraction)
        pos_inds = self.pos_sampler._sample_pos(assign_result, num_expected_pos, bboxes=bboxes, **kwargs)
        pos_inds = pos_inds.unique()
        num_sampled_pos = pos_inds.numel()
        num_expected_neg = self.num - num_sampled_pos
        if self.neg_pos_ub >= 0:
            _pos = max(1, num_sampled_pos)
            neg_upper_bound = int(self.neg_pos_ub * _pos)
            if num_expected_neg > neg_upper_bound:
                num_expected_neg = neg_upper_bound
        neg_inds = self.neg_sampler._sample_neg(assign_result, num_expected_neg, bboxes=bboxes, **kwargs)
        neg_inds = neg_inds.unique()
        return SamplingResult(pos_inds, neg_inds, bboxes, gt_bboxes, assign_result, gt_flags)


class PseudoSampler(BaseSampler):

    def __init__(self, **kwargs):
        pass

    def _sample_pos(self, **kwargs):
        raise NotImplementedError

    def _sample_neg(self, **kwargs):
        raise NotImplementedError

    def sample(self, assign_result, bboxes, gt_bboxes, **kwargs):
        pos_inds = torch.nonzero(assign_result.gt_inds > 0).squeeze(-1).unique()
        neg_inds = torch.nonzero(assign_result.gt_inds == 0).squeeze(-1).unique()
        gt_flags = bboxes.new_zeros(bboxes.shape[0], dtype=torch.uint8)
        sampling_result = SamplingResult(pos_inds, neg_inds, bboxes, gt_bboxes, assign_result, gt_flags)
        return sampling_result


def anchor_inside_flags(flat_anchors, valid_flags, img_shape, allowed_border=0):
    img_h, img_w = img_shape[:2]
    if allowed_border >= 0:
        inside_flags = valid_flags & (flat_anchors[:, (0)] >= -allowed_border) & (flat_anchors[:, (1)] >= -allowed_border) & (flat_anchors[:, (2)] < img_w + allowed_border) & (flat_anchors[:, (3)] < img_h + allowed_border)
    else:
        inside_flags = valid_flags
    return inside_flags


def assign_and_sample(bboxes, gt_bboxes, gt_bboxes_ignore, gt_labels, cfg):
    bbox_assigner = build_assigner(cfg.assigner)
    bbox_sampler = build_sampler(cfg.sampler)
    assign_result = bbox_assigner.assign(bboxes, gt_bboxes, gt_bboxes_ignore, gt_labels)
    sampling_result = bbox_sampler.sample(assign_result, bboxes, gt_bboxes, gt_labels)
    return assign_result, sampling_result


def bbox2delta(proposals, gt, means=[0, 0, 0, 0], stds=[1, 1, 1, 1]):
    assert proposals.size() == gt.size()
    proposals = proposals.float()
    gt = gt.float()
    px = (proposals[..., 0] + proposals[..., 2]) * 0.5
    py = (proposals[..., 1] + proposals[..., 3]) * 0.5
    pw = proposals[..., 2] - proposals[..., 0] + 1.0
    ph = proposals[..., 3] - proposals[..., 1] + 1.0
    gx = (gt[..., 0] + gt[..., 2]) * 0.5
    gy = (gt[..., 1] + gt[..., 3]) * 0.5
    gw = gt[..., 2] - gt[..., 0] + 1.0
    gh = gt[..., 3] - gt[..., 1] + 1.0
    dx = (gx - px) / pw
    dy = (gy - py) / ph
    dw = torch.log(gw / pw)
    dh = torch.log(gh / ph)
    deltas = torch.stack([dx, dy, dw, dh], dim=-1)
    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)
    return deltas


def unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if data.dim() == 1:
        ret = data.new_full((count,), fill)
        ret[inds] = data
    else:
        new_size = (count,) + data.size()[1:]
        ret = data.new_full(new_size, fill)
        ret[(inds), :] = data
    return ret


def anchor_target_single(flat_anchors, valid_flags, gt_bboxes, gt_bboxes_ignore, gt_labels, img_meta, target_means, target_stds, cfg, label_channels=1, sampling=True, unmap_outputs=True):
    inside_flags = anchor_inside_flags(flat_anchors, valid_flags, img_meta['img_shape'][:2], cfg.allowed_border)
    if not inside_flags.any():
        return (None,) * 6
    anchors = flat_anchors[(inside_flags), :]
    if sampling:
        assign_result, sampling_result = assign_and_sample(anchors, gt_bboxes, gt_bboxes_ignore, None, cfg)
    else:
        bbox_assigner = build_assigner(cfg.assigner)
        assign_result = bbox_assigner.assign(anchors, gt_bboxes, gt_bboxes_ignore, gt_labels)
        bbox_sampler = PseudoSampler()
        sampling_result = bbox_sampler.sample(assign_result, anchors, gt_bboxes)
    num_valid_anchors = anchors.shape[0]
    bbox_targets = torch.zeros_like(anchors)
    bbox_weights = torch.zeros_like(anchors)
    labels = anchors.new_zeros(num_valid_anchors, dtype=torch.long)
    label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)
    pos_inds = sampling_result.pos_inds
    neg_inds = sampling_result.neg_inds
    if len(pos_inds) > 0:
        pos_bbox_targets = bbox2delta(sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes, target_means, target_stds)
        bbox_targets[(pos_inds), :] = pos_bbox_targets
        bbox_weights[(pos_inds), :] = 1.0
        if gt_labels is None:
            labels[pos_inds] = 1
        else:
            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        if cfg.pos_weight <= 0:
            label_weights[pos_inds] = 1.0
        else:
            label_weights[pos_inds] = cfg.pos_weight
    if len(neg_inds) > 0:
        label_weights[neg_inds] = 1.0
    if unmap_outputs:
        num_total_anchors = flat_anchors.size(0)
        labels = unmap(labels, num_total_anchors, inside_flags)
        label_weights = unmap(label_weights, num_total_anchors, inside_flags)
        bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
        bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)
    return labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds


def images_to_levels(target, num_level_anchors):
    """Convert targets by image to targets by feature level.

    [target_img0, target_img1] -> [target_level0, target_level1, ...]
    """
    target = torch.stack(target, 0)
    level_targets = []
    start = 0
    for n in num_level_anchors:
        end = start + n
        level_targets.append(target[:, start:end].squeeze(0))
        start = end
    return level_targets


def multi_apply(func, *args, **kwargs):
    pfunc = functools.partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def anchor_target(anchor_list, valid_flag_list, gt_bboxes_list, img_metas, target_means, target_stds, cfg, gt_bboxes_ignore_list=None, gt_labels_list=None, label_channels=1, sampling=True, unmap_outputs=True):
    """Compute regression and classification targets for anchors.

    Args:
        anchor_list (list[list]): Multi level anchors of each image.
        valid_flag_list (list[list]): Multi level valid flags of each image.
        gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
        img_metas (list[dict]): Meta info of each image.
        target_means (Iterable): Mean value of regression targets.
        target_stds (Iterable): Std value of regression targets.
        cfg (dict): RPN train configs.

    Returns:
        tuple
    """
    num_imgs = len(img_metas)
    assert len(anchor_list) == len(valid_flag_list) == num_imgs
    num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
    for i in range(num_imgs):
        assert len(anchor_list[i]) == len(valid_flag_list[i])
        anchor_list[i] = torch.cat(anchor_list[i])
        valid_flag_list[i] = torch.cat(valid_flag_list[i])
    if gt_bboxes_ignore_list is None:
        gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
    if gt_labels_list is None:
        gt_labels_list = [None for _ in range(num_imgs)]
    all_labels, all_label_weights, all_bbox_targets, all_bbox_weights, pos_inds_list, neg_inds_list = multi_apply(anchor_target_single, anchor_list, valid_flag_list, gt_bboxes_list, gt_bboxes_ignore_list, gt_labels_list, img_metas, target_means=target_means, target_stds=target_stds, cfg=cfg, label_channels=label_channels, sampling=sampling, unmap_outputs=unmap_outputs)
    if any([(labels is None) for labels in all_labels]):
        return None
    num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
    num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
    labels_list = images_to_levels(all_labels, num_level_anchors)
    label_weights_list = images_to_levels(all_label_weights, num_level_anchors)
    bbox_targets_list = images_to_levels(all_bbox_targets, num_level_anchors)
    bbox_weights_list = images_to_levels(all_bbox_weights, num_level_anchors)
    return labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg


def delta2bbox(rois, deltas, means=[0, 0, 0, 0], stds=[1, 1, 1, 1], max_shape=None, wh_ratio_clip=16 / 1000):
    means = deltas.new_tensor(means).repeat(1, deltas.size(1) // 4)
    stds = deltas.new_tensor(stds).repeat(1, deltas.size(1) // 4)
    denorm_deltas = deltas * stds + means
    dx = denorm_deltas[:, 0::4]
    dy = denorm_deltas[:, 1::4]
    dw = denorm_deltas[:, 2::4]
    dh = denorm_deltas[:, 3::4]
    max_ratio = np.abs(np.log(wh_ratio_clip))
    dw = dw.clamp(min=-max_ratio, max=max_ratio)
    dh = dh.clamp(min=-max_ratio, max=max_ratio)
    px = ((rois[:, (0)] + rois[:, (2)]) * 0.5).unsqueeze(1).expand_as(dx)
    py = ((rois[:, (1)] + rois[:, (3)]) * 0.5).unsqueeze(1).expand_as(dy)
    pw = (rois[:, (2)] - rois[:, (0)] + 1.0).unsqueeze(1).expand_as(dw)
    ph = (rois[:, (3)] - rois[:, (1)] + 1.0).unsqueeze(1).expand_as(dh)
    gw = pw * dw.exp()
    gh = ph * dh.exp()
    gx = torch.addcmul(px, 1, pw, dx)
    gy = torch.addcmul(py, 1, ph, dy)
    x1 = gx - gw * 0.5 + 0.5
    y1 = gy - gh * 0.5 + 0.5
    x2 = gx + gw * 0.5 - 0.5
    y2 = gy + gh * 0.5 - 0.5
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1] - 1)
        y1 = y1.clamp(min=0, max=max_shape[0] - 1)
        x2 = x2.clamp(min=0, max=max_shape[1] - 1)
        y2 = y2.clamp(min=0, max=max_shape[0] - 1)
    bboxes = torch.stack([x1, y1, x2, y2], dim=-1).view_as(deltas)
    return bboxes


def _expand_binary_labels(labels, label_weights, label_channels):
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = torch.nonzero(labels >= 1).squeeze()
    if inds.numel() > 0:
        bin_labels[inds, labels[inds] - 1] = 1
    bin_label_weights = label_weights.view(-1, 1).expand(label_weights.size(0), label_channels)
    return bin_labels, bin_label_weights


def weighted_binary_cross_entropy(pred, label, weight, avg_factor=None):
    if pred.dim() != label.dim():
        label, weight = _expand_binary_labels(label, weight, pred.size(-1))
    if avg_factor is None:
        avg_factor = max(torch.sum(weight > 0).float().item(), 1.0)
    return F.binary_cross_entropy_with_logits(pred, label.float(), weight.float(), reduction='sum')[None] / avg_factor


def weighted_cross_entropy(pred, label, weight, avg_factor=None, reduce=True):
    if avg_factor is None:
        avg_factor = max(torch.sum(weight > 0).float().item(), 1.0)
    raw = F.cross_entropy(pred, label, reduction='none')
    if reduce:
        return torch.sum(raw * weight)[None] / avg_factor
    else:
        return raw * weight / avg_factor


def smooth_l1_loss(pred, target, beta=1.0, reduction='mean'):
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)
    reduction_enum = F._Reduction.get_enum(reduction)
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.sum() / pred.numel()
    elif reduction_enum == 2:
        return loss.sum()


def weighted_smoothl1(pred, target, weight, beta=1.0, avg_factor=None):
    if avg_factor is None:
        avg_factor = torch.sum(weight > 0).float().item() / 4 + 1e-06
    loss = smooth_l1_loss(pred, target, beta, reduction='none')
    return torch.sum(loss * weight)[None] / avg_factor


class AnchorHead(nn.Module):
    """Anchor-based head (RPN, etc.).

    """

    def __init__(self, num_classes, in_channels, feat_channels=256, anchor_scales=[8, 16, 32], anchor_ratios=[0.5, 1.0, 2.0], anchor_strides=[4, 8, 16, 32, 64], anchor_base_sizes=None, target_means=(0.0, 0.0, 0.0, 0.0), target_stds=(1.0, 1.0, 1.0, 1.0), use_sigmoid_cls=False, use_focal_loss=False):
        super(AnchorHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.anchor_strides = anchor_strides
        self.anchor_base_sizes = list(anchor_strides) if anchor_base_sizes is None else anchor_base_sizes
        self.target_means = target_means
        self.target_stds = target_stds
        self.use_sigmoid_cls = use_sigmoid_cls
        self.use_focal_loss = use_focal_loss
        self.anchor_generators = []
        for anchor_base in self.anchor_base_sizes:
            self.anchor_generators.append(AnchorGenerator(anchor_base, anchor_scales, anchor_ratios))
        self.num_anchors = len(self.anchor_ratios) * len(self.anchor_scales)
        if self.use_sigmoid_cls:
            self.cls_out_channels = self.num_classes - 1
        else:
            self.cls_out_channels = self.num_classes
        self._init_layers()

    def _init_layers(self):
        self.conv_cls = nn.Conv2d(self.feat_channels, self.num_anchors * self.cls_out_channels, 1)
        self.conv_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 1)

    def init_weights(self):
        normal_init(self.conv_cls, std=0.01)
        normal_init(self.conv_reg, std=0.01)

    def forward_single(self, x):
        cls_score = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        return cls_score, bbox_pred

    def forward(self, feats):
        return multi_apply(self.forward_single, feats)

    def get_anchors(self, featmap_sizes, img_metas):
        """Get anchors according to feature map sizes.
        
        """
        num_imgs = len(img_metas)
        num_levels = len(featmap_sizes)
        multi_level_anchors = []
        for i in range(num_levels):
            anchors = self.anchor_generators[i].grid_anchors(featmap_sizes[i], self.anchor_strides[i])
            multi_level_anchors.append(anchors)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = []
            for i in range(num_levels):
                anchor_stride = self.anchor_strides[i]
                feat_h, feat_w = featmap_sizes[i]
                h, w, _ = img_meta['pad_shape']
                valid_feat_h = min(int(np.ceil(h / anchor_stride)), feat_h)
                valid_feat_w = min(int(np.ceil(w / anchor_stride)), feat_w)
                flags = self.anchor_generators[i].valid_flags((feat_h, feat_w), (valid_feat_h, valid_feat_w))
                multi_level_flags.append(flags)
            valid_flag_list.append(multi_level_flags)
        return anchor_list, valid_flag_list

    def loss_single(self, cls_score, bbox_pred, labels, label_weights, bbox_targets, bbox_weights, num_total_samples, cfg):
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        if self.use_sigmoid_cls:
            if self.use_focal_loss:
                raise NotImplementedError
            else:
                cls_criterion = weighted_binary_cross_entropy
        elif self.use_focal_loss:
            raise NotImplementedError
        else:
            cls_criterion = weighted_cross_entropy
        if self.use_focal_loss:
            raise NotImplementedError
        else:
            loss_cls = cls_criterion(cls_score, labels, label_weights, avg_factor=num_total_samples)
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        loss_reg = weighted_smoothl1(bbox_pred, bbox_targets, bbox_weights, beta=cfg.smoothl1_beta, avg_factor=num_total_samples)
        return loss_cls, loss_reg

    def loss(self, cls_scores, bbox_preds, gt_bboxes, gt_labels, img_metas, cfg, gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == len(self.anchor_generators)
        anchor_list, valid_flag_list = self.get_anchors(featmap_sizes, img_metas)
        sampling = False if self.use_focal_loss else True
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = anchor_target(anchor_list, valid_flag_list, gt_bboxes, img_metas, self.target_means, self.target_stds, cfg, gt_bboxes_ignore_list=gt_bboxes_ignore, gt_labels_list=gt_labels, label_channels=label_channels, sampling=sampling)
        if cls_reg_targets is None:
            return None
        labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg = cls_reg_targets
        num_total_samples = num_total_pos if self.use_focal_loss else num_total_pos + num_total_neg
        losses_cls, losses_reg = multi_apply(self.loss_single, cls_scores, bbox_preds, labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_samples=num_total_samples, cfg=cfg)
        return dict(loss_cls=losses_cls, loss_reg=losses_reg)

    def get_bboxes(self, cls_scores, bbox_preds, img_metas, cfg, rescale=False):
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)
        mlvl_anchors = [self.anchor_generators[i].grid_anchors(cls_scores[i].size()[-2:], self.anchor_strides[i]) for i in range(num_levels)]
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [cls_scores[i][img_id].detach() for i in range(num_levels)]
            bbox_pred_list = [bbox_preds[i][img_id].detach() for i in range(num_levels)]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self.get_bboxes_single(cls_score_list, bbox_pred_list, mlvl_anchors, img_shape, scale_factor, cfg, rescale)
            result_list.append(proposals)
        return result_list

    def get_bboxes_single(self, cls_scores, bbox_preds, mlvl_anchors, img_shape, scale_factor, cfg, rescale=False):
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, anchors in zip(cls_scores, bbox_preds, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    max_scores, _ = scores[:, 1:].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[(topk_inds), :]
                bbox_pred = bbox_pred[(topk_inds), :]
                scores = scores[(topk_inds), :]
            bboxes = delta2bbox(anchors, bbox_pred, self.target_means, self.target_stds, img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        if self.use_sigmoid_cls:
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)
        det_bboxes, det_labels = multiclass_nms(mlvl_bboxes, mlvl_scores, cfg.score_thr, cfg.nms, cfg.max_per_img)
        return det_bboxes, det_labels


class RPNHead(AnchorHead):

    def __init__(self, in_channels, **kwargs):
        super(RPNHead, self).__init__(2, in_channels, **kwargs)

    def _init_layers(self):
        self.rpn_conv = nn.Conv2d(self.in_channels, self.feat_channels, 3, padding=1)
        self.rpn_cls = nn.Conv2d(self.feat_channels, self.num_anchors * self.cls_out_channels, 1)
        self.rpn_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 1)

    def init_weights(self):
        normal_init(self.rpn_conv, std=0.01)
        normal_init(self.rpn_cls, std=0.01)
        normal_init(self.rpn_reg, std=0.01)

    def forward_single(self, x):
        x = self.rpn_conv(x)
        x = F.relu(x, inplace=True)
        rpn_cls_score = self.rpn_cls(x)
        rpn_bbox_pred = self.rpn_reg(x)
        return rpn_cls_score, rpn_bbox_pred

    def loss(self, cls_scores, bbox_preds, gt_bboxes, img_metas, cfg, gt_bboxes_ignore=None):
        losses = super(RPNHead, self).loss(cls_scores, bbox_preds, gt_bboxes, None, img_metas, cfg, gt_bboxes_ignore=gt_bboxes_ignore)
        return dict(loss_rpn_cls=losses['loss_cls'], loss_rpn_reg=losses['loss_reg'])

    def get_bboxes_single(self, cls_scores, bbox_preds, mlvl_anchors, img_shape, scale_factor, cfg, rescale=False):
        mlvl_proposals = []
        for idx in range(len(cls_scores)):
            rpn_cls_score = cls_scores[idx]
            rpn_bbox_pred = bbox_preds[idx]
            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
            anchors = mlvl_anchors[idx]
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
            if self.use_sigmoid_cls:
                rpn_cls_score = rpn_cls_score.reshape(-1)
                scores = rpn_cls_score.sigmoid()
            else:
                rpn_cls_score = rpn_cls_score.reshape(-1, 2)
                scores = rpn_cls_score.softmax(dim=1)[:, (1)]
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            if cfg.nms_pre > 0 and scores.shape[0] > cfg.nms_pre:
                _, topk_inds = scores.topk(cfg.nms_pre)
                rpn_bbox_pred = rpn_bbox_pred[(topk_inds), :]
                anchors = anchors[(topk_inds), :]
                scores = scores[topk_inds]
            proposals = delta2bbox(anchors, rpn_bbox_pred, self.target_means, self.target_stds, img_shape)
            if cfg.min_bbox_size > 0:
                w = proposals[:, (2)] - proposals[:, (0)] + 1
                h = proposals[:, (3)] - proposals[:, (1)] + 1
                valid_inds = torch.nonzero((w >= cfg.min_bbox_size) & (h >= cfg.min_bbox_size)).squeeze()
                proposals = proposals[(valid_inds), :]
                scores = scores[valid_inds]
            proposals = torch.cat([proposals, scores.unsqueeze(-1)], dim=-1)
            proposals, _ = nms(proposals, cfg.nms_thr)
            proposals = proposals[:cfg.nms_post, :]
            mlvl_proposals.append(proposals)
        proposals = torch.cat(mlvl_proposals, 0)
        if cfg.nms_across_levels:
            proposals, _ = nms(proposals, cfg.nms_thr)
            proposals = proposals[:cfg.max_num, :]
        else:
            scores = proposals[:, (4)]
            num = min(cfg.max_num, proposals.shape[0])
            _, topk_inds = scores.topk(num)
            proposals = proposals[(topk_inds), :]
        return proposals


BACKBONES = Registry('backbone')


class BNInception(nn.Module):

    def __init__(self, pretrained=None, bn_eval=True, bn_frozen=False, partial_bn=False):
        super(BNInception, self).__init__()
        self.pretrained = pretrained
        self.bn_eval = bn_eval
        self.bn_frozen = bn_frozen
        self.partial_bn = partial_bn
        inplace = True
        self.conv1_7x7_s2 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.conv1_7x7_s2_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.9, affine=True)
        self.conv1_relu_7x7 = nn.ReLU(inplace)
        self.pool1_3x3_s2 = nn.MaxPool2d((3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=True)
        self.conv2_3x3_reduce = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
        self.conv2_3x3_reduce_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.9, affine=True)
        self.conv2_relu_3x3_reduce = nn.ReLU(inplace)
        self.conv2_3x3 = nn.Conv2d(64, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2_3x3_bn = nn.BatchNorm2d(192, eps=1e-05, momentum=0.9, affine=True)
        self.conv2_relu_3x3 = nn.ReLU(inplace)
        self.pool2_3x3_s2 = nn.MaxPool2d((3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=True)
        self.inception_3a_1x1 = nn.Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3a_1x1_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3a_relu_1x1 = nn.ReLU(inplace)
        self.inception_3a_3x3_reduce = nn.Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3a_3x3_reduce_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3a_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_3a_3x3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3a_3x3_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3a_relu_3x3 = nn.ReLU(inplace)
        self.inception_3a_double_3x3_reduce = nn.Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3a_double_3x3_reduce_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3a_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_3a_double_3x3_1 = nn.Conv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3a_double_3x3_1_bn = nn.BatchNorm2d(96, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3a_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_3a_double_3x3_2 = nn.Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3a_double_3x3_2_bn = nn.BatchNorm2d(96, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3a_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_3a_pool = nn.AvgPool2d(3, stride=1, padding=1, ceil_mode=True, count_include_pad=True)
        self.inception_3a_pool_proj = nn.Conv2d(192, 32, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3a_pool_proj_bn = nn.BatchNorm2d(32, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3a_relu_pool_proj = nn.ReLU(inplace)
        self.inception_3b_1x1 = nn.Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3b_1x1_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3b_relu_1x1 = nn.ReLU(inplace)
        self.inception_3b_3x3_reduce = nn.Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3b_3x3_reduce_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3b_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_3b_3x3 = nn.Conv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3b_3x3_bn = nn.BatchNorm2d(96, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3b_relu_3x3 = nn.ReLU(inplace)
        self.inception_3b_double_3x3_reduce = nn.Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3b_double_3x3_reduce_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3b_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_3b_double_3x3_1 = nn.Conv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3b_double_3x3_1_bn = nn.BatchNorm2d(96, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3b_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_3b_double_3x3_2 = nn.Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3b_double_3x3_2_bn = nn.BatchNorm2d(96, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3b_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_3b_pool = nn.AvgPool2d(3, stride=1, padding=1, ceil_mode=True, count_include_pad=True)
        self.inception_3b_pool_proj = nn.Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3b_pool_proj_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3b_relu_pool_proj = nn.ReLU(inplace)
        self.inception_3c_3x3_reduce = nn.Conv2d(320, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3c_3x3_reduce_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3c_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_3c_3x3 = nn.Conv2d(128, 160, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.inception_3c_3x3_bn = nn.BatchNorm2d(160, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3c_relu_3x3 = nn.ReLU(inplace)
        self.inception_3c_double_3x3_reduce = nn.Conv2d(320, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3c_double_3x3_reduce_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3c_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_3c_double_3x3_1 = nn.Conv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3c_double_3x3_1_bn = nn.BatchNorm2d(96, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3c_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_3c_double_3x3_2 = nn.Conv2d(96, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.inception_3c_double_3x3_2_bn = nn.BatchNorm2d(96, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3c_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_3c_pool = nn.MaxPool2d((3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=True)
        self.inception_4a_1x1 = nn.Conv2d(576, 224, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4a_1x1_bn = nn.BatchNorm2d(224, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4a_relu_1x1 = nn.ReLU(inplace)
        self.inception_4a_3x3_reduce = nn.Conv2d(576, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4a_3x3_reduce_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4a_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_4a_3x3 = nn.Conv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4a_3x3_bn = nn.BatchNorm2d(96, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4a_relu_3x3 = nn.ReLU(inplace)
        self.inception_4a_double_3x3_reduce = nn.Conv2d(576, 96, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4a_double_3x3_reduce_bn = nn.BatchNorm2d(96, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4a_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_4a_double_3x3_1 = nn.Conv2d(96, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4a_double_3x3_1_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4a_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_4a_double_3x3_2 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4a_double_3x3_2_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4a_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_4a_pool = nn.AvgPool2d(3, stride=1, padding=1, ceil_mode=True, count_include_pad=True)
        self.inception_4a_pool_proj = nn.Conv2d(576, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4a_pool_proj_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4a_relu_pool_proj = nn.ReLU(inplace)
        self.inception_4b_1x1 = nn.Conv2d(576, 192, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4b_1x1_bn = nn.BatchNorm2d(192, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4b_relu_1x1 = nn.ReLU(inplace)
        self.inception_4b_3x3_reduce = nn.Conv2d(576, 96, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4b_3x3_reduce_bn = nn.BatchNorm2d(96, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4b_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_4b_3x3 = nn.Conv2d(96, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4b_3x3_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4b_relu_3x3 = nn.ReLU(inplace)
        self.inception_4b_double_3x3_reduce = nn.Conv2d(576, 96, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4b_double_3x3_reduce_bn = nn.BatchNorm2d(96, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4b_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_4b_double_3x3_1 = nn.Conv2d(96, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4b_double_3x3_1_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4b_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_4b_double_3x3_2 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4b_double_3x3_2_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4b_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_4b_pool = nn.AvgPool2d(3, stride=1, padding=1, ceil_mode=True, count_include_pad=True)
        self.inception_4b_pool_proj = nn.Conv2d(576, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4b_pool_proj_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4b_relu_pool_proj = nn.ReLU(inplace)
        self.inception_4c_1x1 = nn.Conv2d(576, 160, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4c_1x1_bn = nn.BatchNorm2d(160, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4c_relu_1x1 = nn.ReLU(inplace)
        self.inception_4c_3x3_reduce = nn.Conv2d(576, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4c_3x3_reduce_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4c_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_4c_3x3 = nn.Conv2d(128, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4c_3x3_bn = nn.BatchNorm2d(160, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4c_relu_3x3 = nn.ReLU(inplace)
        self.inception_4c_double_3x3_reduce = nn.Conv2d(576, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4c_double_3x3_reduce_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4c_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_4c_double_3x3_1 = nn.Conv2d(128, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4c_double_3x3_1_bn = nn.BatchNorm2d(160, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4c_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_4c_double_3x3_2 = nn.Conv2d(160, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4c_double_3x3_2_bn = nn.BatchNorm2d(160, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4c_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_4c_pool = nn.AvgPool2d(3, stride=1, padding=1, ceil_mode=True, count_include_pad=True)
        self.inception_4c_pool_proj = nn.Conv2d(576, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4c_pool_proj_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4c_relu_pool_proj = nn.ReLU(inplace)
        self.inception_4d_1x1 = nn.Conv2d(608, 96, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4d_1x1_bn = nn.BatchNorm2d(96, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4d_relu_1x1 = nn.ReLU(inplace)
        self.inception_4d_3x3_reduce = nn.Conv2d(608, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4d_3x3_reduce_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4d_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_4d_3x3 = nn.Conv2d(128, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4d_3x3_bn = nn.BatchNorm2d(192, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4d_relu_3x3 = nn.ReLU(inplace)
        self.inception_4d_double_3x3_reduce = nn.Conv2d(608, 160, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4d_double_3x3_reduce_bn = nn.BatchNorm2d(160, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4d_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_4d_double_3x3_1 = nn.Conv2d(160, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4d_double_3x3_1_bn = nn.BatchNorm2d(192, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4d_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_4d_double_3x3_2 = nn.Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4d_double_3x3_2_bn = nn.BatchNorm2d(192, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4d_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_4d_pool = nn.AvgPool2d(3, stride=1, padding=1, ceil_mode=True, count_include_pad=True)
        self.inception_4d_pool_proj = nn.Conv2d(608, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4d_pool_proj_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4d_relu_pool_proj = nn.ReLU(inplace)
        self.inception_4e_3x3_reduce = nn.Conv2d(608, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4e_3x3_reduce_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4e_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_4e_3x3 = nn.Conv2d(128, 192, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.inception_4e_3x3_bn = nn.BatchNorm2d(192, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4e_relu_3x3 = nn.ReLU(inplace)
        self.inception_4e_double_3x3_reduce = nn.Conv2d(608, 192, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4e_double_3x3_reduce_bn = nn.BatchNorm2d(192, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4e_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_4e_double_3x3_1 = nn.Conv2d(192, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4e_double_3x3_1_bn = nn.BatchNorm2d(256, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4e_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_4e_double_3x3_2 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.inception_4e_double_3x3_2_bn = nn.BatchNorm2d(256, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4e_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_4e_pool = nn.MaxPool2d((3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=True)
        self.inception_5a_1x1 = nn.Conv2d(1056, 352, kernel_size=(1, 1), stride=(1, 1))
        self.inception_5a_1x1_bn = nn.BatchNorm2d(352, eps=1e-05, momentum=0.9, affine=True)
        self.inception_5a_relu_1x1 = nn.ReLU(inplace)
        self.inception_5a_3x3_reduce = nn.Conv2d(1056, 192, kernel_size=(1, 1), stride=(1, 1))
        self.inception_5a_3x3_reduce_bn = nn.BatchNorm2d(192, eps=1e-05, momentum=0.9, affine=True)
        self.inception_5a_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_5a_3x3 = nn.Conv2d(192, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_5a_3x3_bn = nn.BatchNorm2d(320, eps=1e-05, momentum=0.9, affine=True)
        self.inception_5a_relu_3x3 = nn.ReLU(inplace)
        self.inception_5a_double_3x3_reduce = nn.Conv2d(1056, 160, kernel_size=(1, 1), stride=(1, 1))
        self.inception_5a_double_3x3_reduce_bn = nn.BatchNorm2d(160, eps=1e-05, momentum=0.9, affine=True)
        self.inception_5a_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_5a_double_3x3_1 = nn.Conv2d(160, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_5a_double_3x3_1_bn = nn.BatchNorm2d(224, eps=1e-05, momentum=0.9, affine=True)
        self.inception_5a_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_5a_double_3x3_2 = nn.Conv2d(224, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_5a_double_3x3_2_bn = nn.BatchNorm2d(224, eps=1e-05, momentum=0.9, affine=True)
        self.inception_5a_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_5a_pool = nn.AvgPool2d(3, stride=1, padding=1, ceil_mode=True, count_include_pad=True)
        self.inception_5a_pool_proj = nn.Conv2d(1056, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_5a_pool_proj_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
        self.inception_5a_relu_pool_proj = nn.ReLU(inplace)
        self.inception_5b_1x1 = nn.Conv2d(1024, 352, kernel_size=(1, 1), stride=(1, 1))
        self.inception_5b_1x1_bn = nn.BatchNorm2d(352, eps=1e-05, momentum=0.9, affine=True)
        self.inception_5b_relu_1x1 = nn.ReLU(inplace)
        self.inception_5b_3x3_reduce = nn.Conv2d(1024, 192, kernel_size=(1, 1), stride=(1, 1))
        self.inception_5b_3x3_reduce_bn = nn.BatchNorm2d(192, eps=1e-05, momentum=0.9, affine=True)
        self.inception_5b_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_5b_3x3 = nn.Conv2d(192, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_5b_3x3_bn = nn.BatchNorm2d(320, eps=1e-05, momentum=0.9, affine=True)
        self.inception_5b_relu_3x3 = nn.ReLU(inplace)
        self.inception_5b_double_3x3_reduce = nn.Conv2d(1024, 192, kernel_size=(1, 1), stride=(1, 1))
        self.inception_5b_double_3x3_reduce_bn = nn.BatchNorm2d(192, eps=1e-05, momentum=0.9, affine=True)
        self.inception_5b_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_5b_double_3x3_1 = nn.Conv2d(192, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_5b_double_3x3_1_bn = nn.BatchNorm2d(224, eps=1e-05, momentum=0.9, affine=True)
        self.inception_5b_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_5b_double_3x3_2 = nn.Conv2d(224, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_5b_double_3x3_2_bn = nn.BatchNorm2d(224, eps=1e-05, momentum=0.9, affine=True)
        self.inception_5b_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_5b_pool = nn.MaxPool2d((3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), ceil_mode=True)
        self.inception_5b_pool_proj = nn.Conv2d(1024, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_5b_pool_proj_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
        self.inception_5b_relu_pool_proj = nn.ReLU(inplace)

    def init_weights(self):
        if isinstance(self.pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, self.pretrained, strict=False, logger=logger)
        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)

    def conv1(self, input):
        conv1_7x7_s2_out = self.conv1_7x7_s2(input)
        conv1_7x7_s2_bn_out = self.conv1_7x7_s2_bn(conv1_7x7_s2_out)
        conv1_relu_7x7_out = self.conv1_relu_7x7(conv1_7x7_s2_bn_out)
        return conv1_7x7_s2_bn_out

    def forward(self, input):
        conv1_7x7_s2_bn_out = self.conv1(input)
        pool1_3x3_s2_out = self.pool1_3x3_s2(conv1_7x7_s2_bn_out)
        conv2_3x3_reduce_out = self.conv2_3x3_reduce(pool1_3x3_s2_out)
        conv2_3x3_reduce_bn_out = self.conv2_3x3_reduce_bn(conv2_3x3_reduce_out)
        conv2_relu_3x3_reduce_out = self.conv2_relu_3x3_reduce(conv2_3x3_reduce_bn_out)
        conv2_3x3_out = self.conv2_3x3(conv2_3x3_reduce_bn_out)
        conv2_3x3_bn_out = self.conv2_3x3_bn(conv2_3x3_out)
        conv2_relu_3x3_out = self.conv2_relu_3x3(conv2_3x3_bn_out)
        pool2_3x3_s2_out = self.pool2_3x3_s2(conv2_3x3_bn_out)
        inception_3a_1x1_out = self.inception_3a_1x1(pool2_3x3_s2_out)
        inception_3a_1x1_bn_out = self.inception_3a_1x1_bn(inception_3a_1x1_out)
        inception_3a_relu_1x1_out = self.inception_3a_relu_1x1(inception_3a_1x1_bn_out)
        inception_3a_3x3_reduce_out = self.inception_3a_3x3_reduce(pool2_3x3_s2_out)
        inception_3a_3x3_reduce_bn_out = self.inception_3a_3x3_reduce_bn(inception_3a_3x3_reduce_out)
        inception_3a_relu_3x3_reduce_out = self.inception_3a_relu_3x3_reduce(inception_3a_3x3_reduce_bn_out)
        inception_3a_3x3_out = self.inception_3a_3x3(inception_3a_3x3_reduce_bn_out)
        inception_3a_3x3_bn_out = self.inception_3a_3x3_bn(inception_3a_3x3_out)
        inception_3a_relu_3x3_out = self.inception_3a_relu_3x3(inception_3a_3x3_bn_out)
        inception_3a_double_3x3_reduce_out = self.inception_3a_double_3x3_reduce(pool2_3x3_s2_out)
        inception_3a_double_3x3_reduce_bn_out = self.inception_3a_double_3x3_reduce_bn(inception_3a_double_3x3_reduce_out)
        inception_3a_relu_double_3x3_reduce_out = self.inception_3a_relu_double_3x3_reduce(inception_3a_double_3x3_reduce_bn_out)
        inception_3a_double_3x3_1_out = self.inception_3a_double_3x3_1(inception_3a_double_3x3_reduce_bn_out)
        inception_3a_double_3x3_1_bn_out = self.inception_3a_double_3x3_1_bn(inception_3a_double_3x3_1_out)
        inception_3a_relu_double_3x3_1_out = self.inception_3a_relu_double_3x3_1(inception_3a_double_3x3_1_bn_out)
        inception_3a_double_3x3_2_out = self.inception_3a_double_3x3_2(inception_3a_double_3x3_1_bn_out)
        inception_3a_double_3x3_2_bn_out = self.inception_3a_double_3x3_2_bn(inception_3a_double_3x3_2_out)
        inception_3a_relu_double_3x3_2_out = self.inception_3a_relu_double_3x3_2(inception_3a_double_3x3_2_bn_out)
        inception_3a_pool_out = self.inception_3a_pool(pool2_3x3_s2_out)
        inception_3a_pool_proj_out = self.inception_3a_pool_proj(inception_3a_pool_out)
        inception_3a_pool_proj_bn_out = self.inception_3a_pool_proj_bn(inception_3a_pool_proj_out)
        inception_3a_relu_pool_proj_out = self.inception_3a_relu_pool_proj(inception_3a_pool_proj_bn_out)
        inception_3a_output_out = torch.cat([inception_3a_1x1_bn_out, inception_3a_3x3_bn_out, inception_3a_double_3x3_2_bn_out, inception_3a_pool_proj_bn_out], 1)
        inception_3b_1x1_out = self.inception_3b_1x1(inception_3a_output_out)
        inception_3b_1x1_bn_out = self.inception_3b_1x1_bn(inception_3b_1x1_out)
        inception_3b_relu_1x1_out = self.inception_3b_relu_1x1(inception_3b_1x1_bn_out)
        inception_3b_3x3_reduce_out = self.inception_3b_3x3_reduce(inception_3a_output_out)
        inception_3b_3x3_reduce_bn_out = self.inception_3b_3x3_reduce_bn(inception_3b_3x3_reduce_out)
        inception_3b_relu_3x3_reduce_out = self.inception_3b_relu_3x3_reduce(inception_3b_3x3_reduce_bn_out)
        inception_3b_3x3_out = self.inception_3b_3x3(inception_3b_3x3_reduce_bn_out)
        inception_3b_3x3_bn_out = self.inception_3b_3x3_bn(inception_3b_3x3_out)
        inception_3b_relu_3x3_out = self.inception_3b_relu_3x3(inception_3b_3x3_bn_out)
        inception_3b_double_3x3_reduce_out = self.inception_3b_double_3x3_reduce(inception_3a_output_out)
        inception_3b_double_3x3_reduce_bn_out = self.inception_3b_double_3x3_reduce_bn(inception_3b_double_3x3_reduce_out)
        inception_3b_relu_double_3x3_reduce_out = self.inception_3b_relu_double_3x3_reduce(inception_3b_double_3x3_reduce_bn_out)
        inception_3b_double_3x3_1_out = self.inception_3b_double_3x3_1(inception_3b_double_3x3_reduce_bn_out)
        inception_3b_double_3x3_1_bn_out = self.inception_3b_double_3x3_1_bn(inception_3b_double_3x3_1_out)
        inception_3b_relu_double_3x3_1_out = self.inception_3b_relu_double_3x3_1(inception_3b_double_3x3_1_bn_out)
        inception_3b_double_3x3_2_out = self.inception_3b_double_3x3_2(inception_3b_double_3x3_1_bn_out)
        inception_3b_double_3x3_2_bn_out = self.inception_3b_double_3x3_2_bn(inception_3b_double_3x3_2_out)
        inception_3b_relu_double_3x3_2_out = self.inception_3b_relu_double_3x3_2(inception_3b_double_3x3_2_bn_out)
        inception_3b_pool_out = self.inception_3b_pool(inception_3a_output_out)
        inception_3b_pool_proj_out = self.inception_3b_pool_proj(inception_3b_pool_out)
        inception_3b_pool_proj_bn_out = self.inception_3b_pool_proj_bn(inception_3b_pool_proj_out)
        inception_3b_relu_pool_proj_out = self.inception_3b_relu_pool_proj(inception_3b_pool_proj_bn_out)
        inception_3b_output_out = torch.cat([inception_3b_1x1_bn_out, inception_3b_3x3_bn_out, inception_3b_double_3x3_2_bn_out, inception_3b_pool_proj_bn_out], 1)
        inception_3c_3x3_reduce_out = self.inception_3c_3x3_reduce(inception_3b_output_out)
        inception_3c_3x3_reduce_bn_out = self.inception_3c_3x3_reduce_bn(inception_3c_3x3_reduce_out)
        inception_3c_relu_3x3_reduce_out = self.inception_3c_relu_3x3_reduce(inception_3c_3x3_reduce_bn_out)
        inception_3c_3x3_out = self.inception_3c_3x3(inception_3c_3x3_reduce_bn_out)
        inception_3c_3x3_bn_out = self.inception_3c_3x3_bn(inception_3c_3x3_out)
        inception_3c_relu_3x3_out = self.inception_3c_relu_3x3(inception_3c_3x3_bn_out)
        inception_3c_double_3x3_reduce_out = self.inception_3c_double_3x3_reduce(inception_3b_output_out)
        inception_3c_double_3x3_reduce_bn_out = self.inception_3c_double_3x3_reduce_bn(inception_3c_double_3x3_reduce_out)
        inception_3c_relu_double_3x3_reduce_out = self.inception_3c_relu_double_3x3_reduce(inception_3c_double_3x3_reduce_bn_out)
        inception_3c_double_3x3_1_out = self.inception_3c_double_3x3_1(inception_3c_double_3x3_reduce_bn_out)
        inception_3c_double_3x3_1_bn_out = self.inception_3c_double_3x3_1_bn(inception_3c_double_3x3_1_out)
        inception_3c_relu_double_3x3_1_out = self.inception_3c_relu_double_3x3_1(inception_3c_double_3x3_1_bn_out)
        inception_3c_double_3x3_2_out = self.inception_3c_double_3x3_2(inception_3c_double_3x3_1_bn_out)
        inception_3c_double_3x3_2_bn_out = self.inception_3c_double_3x3_2_bn(inception_3c_double_3x3_2_out)
        inception_3c_relu_double_3x3_2_out = self.inception_3c_relu_double_3x3_2(inception_3c_double_3x3_2_bn_out)
        inception_3c_pool_out = self.inception_3c_pool(inception_3b_output_out)
        inception_3c_output_out = torch.cat([inception_3c_3x3_bn_out, inception_3c_double_3x3_2_bn_out, inception_3c_pool_out], 1)
        inception_4a_1x1_out = self.inception_4a_1x1(inception_3c_output_out)
        inception_4a_1x1_bn_out = self.inception_4a_1x1_bn(inception_4a_1x1_out)
        inception_4a_relu_1x1_out = self.inception_4a_relu_1x1(inception_4a_1x1_bn_out)
        inception_4a_3x3_reduce_out = self.inception_4a_3x3_reduce(inception_3c_output_out)
        inception_4a_3x3_reduce_bn_out = self.inception_4a_3x3_reduce_bn(inception_4a_3x3_reduce_out)
        inception_4a_relu_3x3_reduce_out = self.inception_4a_relu_3x3_reduce(inception_4a_3x3_reduce_bn_out)
        inception_4a_3x3_out = self.inception_4a_3x3(inception_4a_3x3_reduce_bn_out)
        inception_4a_3x3_bn_out = self.inception_4a_3x3_bn(inception_4a_3x3_out)
        inception_4a_relu_3x3_out = self.inception_4a_relu_3x3(inception_4a_3x3_bn_out)
        inception_4a_double_3x3_reduce_out = self.inception_4a_double_3x3_reduce(inception_3c_output_out)
        inception_4a_double_3x3_reduce_bn_out = self.inception_4a_double_3x3_reduce_bn(inception_4a_double_3x3_reduce_out)
        inception_4a_relu_double_3x3_reduce_out = self.inception_4a_relu_double_3x3_reduce(inception_4a_double_3x3_reduce_bn_out)
        inception_4a_double_3x3_1_out = self.inception_4a_double_3x3_1(inception_4a_double_3x3_reduce_bn_out)
        inception_4a_double_3x3_1_bn_out = self.inception_4a_double_3x3_1_bn(inception_4a_double_3x3_1_out)
        inception_4a_relu_double_3x3_1_out = self.inception_4a_relu_double_3x3_1(inception_4a_double_3x3_1_bn_out)
        inception_4a_double_3x3_2_out = self.inception_4a_double_3x3_2(inception_4a_double_3x3_1_bn_out)
        inception_4a_double_3x3_2_bn_out = self.inception_4a_double_3x3_2_bn(inception_4a_double_3x3_2_out)
        inception_4a_relu_double_3x3_2_out = self.inception_4a_relu_double_3x3_2(inception_4a_double_3x3_2_bn_out)
        inception_4a_pool_out = self.inception_4a_pool(inception_3c_output_out)
        inception_4a_pool_proj_out = self.inception_4a_pool_proj(inception_4a_pool_out)
        inception_4a_pool_proj_bn_out = self.inception_4a_pool_proj_bn(inception_4a_pool_proj_out)
        inception_4a_relu_pool_proj_out = self.inception_4a_relu_pool_proj(inception_4a_pool_proj_bn_out)
        inception_4a_output_out = torch.cat([inception_4a_1x1_bn_out, inception_4a_3x3_bn_out, inception_4a_double_3x3_2_bn_out, inception_4a_pool_proj_bn_out], 1)
        inception_4b_1x1_out = self.inception_4b_1x1(inception_4a_output_out)
        inception_4b_1x1_bn_out = self.inception_4b_1x1_bn(inception_4b_1x1_out)
        inception_4b_relu_1x1_out = self.inception_4b_relu_1x1(inception_4b_1x1_bn_out)
        inception_4b_3x3_reduce_out = self.inception_4b_3x3_reduce(inception_4a_output_out)
        inception_4b_3x3_reduce_bn_out = self.inception_4b_3x3_reduce_bn(inception_4b_3x3_reduce_out)
        inception_4b_relu_3x3_reduce_out = self.inception_4b_relu_3x3_reduce(inception_4b_3x3_reduce_bn_out)
        inception_4b_3x3_out = self.inception_4b_3x3(inception_4b_3x3_reduce_bn_out)
        inception_4b_3x3_bn_out = self.inception_4b_3x3_bn(inception_4b_3x3_out)
        inception_4b_relu_3x3_out = self.inception_4b_relu_3x3(inception_4b_3x3_bn_out)
        inception_4b_double_3x3_reduce_out = self.inception_4b_double_3x3_reduce(inception_4a_output_out)
        inception_4b_double_3x3_reduce_bn_out = self.inception_4b_double_3x3_reduce_bn(inception_4b_double_3x3_reduce_out)
        inception_4b_relu_double_3x3_reduce_out = self.inception_4b_relu_double_3x3_reduce(inception_4b_double_3x3_reduce_bn_out)
        inception_4b_double_3x3_1_out = self.inception_4b_double_3x3_1(inception_4b_double_3x3_reduce_bn_out)
        inception_4b_double_3x3_1_bn_out = self.inception_4b_double_3x3_1_bn(inception_4b_double_3x3_1_out)
        inception_4b_relu_double_3x3_1_out = self.inception_4b_relu_double_3x3_1(inception_4b_double_3x3_1_bn_out)
        inception_4b_double_3x3_2_out = self.inception_4b_double_3x3_2(inception_4b_double_3x3_1_bn_out)
        inception_4b_double_3x3_2_bn_out = self.inception_4b_double_3x3_2_bn(inception_4b_double_3x3_2_out)
        inception_4b_relu_double_3x3_2_out = self.inception_4b_relu_double_3x3_2(inception_4b_double_3x3_2_bn_out)
        inception_4b_pool_out = self.inception_4b_pool(inception_4a_output_out)
        inception_4b_pool_proj_out = self.inception_4b_pool_proj(inception_4b_pool_out)
        inception_4b_pool_proj_bn_out = self.inception_4b_pool_proj_bn(inception_4b_pool_proj_out)
        inception_4b_relu_pool_proj_out = self.inception_4b_relu_pool_proj(inception_4b_pool_proj_bn_out)
        inception_4b_output_out = torch.cat([inception_4b_1x1_bn_out, inception_4b_3x3_bn_out, inception_4b_double_3x3_2_bn_out, inception_4b_pool_proj_bn_out], 1)
        inception_4c_1x1_out = self.inception_4c_1x1(inception_4b_output_out)
        inception_4c_1x1_bn_out = self.inception_4c_1x1_bn(inception_4c_1x1_out)
        inception_4c_relu_1x1_out = self.inception_4c_relu_1x1(inception_4c_1x1_bn_out)
        inception_4c_3x3_reduce_out = self.inception_4c_3x3_reduce(inception_4b_output_out)
        inception_4c_3x3_reduce_bn_out = self.inception_4c_3x3_reduce_bn(inception_4c_3x3_reduce_out)
        inception_4c_relu_3x3_reduce_out = self.inception_4c_relu_3x3_reduce(inception_4c_3x3_reduce_bn_out)
        inception_4c_3x3_out = self.inception_4c_3x3(inception_4c_3x3_reduce_bn_out)
        inception_4c_3x3_bn_out = self.inception_4c_3x3_bn(inception_4c_3x3_out)
        inception_4c_relu_3x3_out = self.inception_4c_relu_3x3(inception_4c_3x3_bn_out)
        inception_4c_double_3x3_reduce_out = self.inception_4c_double_3x3_reduce(inception_4b_output_out)
        inception_4c_double_3x3_reduce_bn_out = self.inception_4c_double_3x3_reduce_bn(inception_4c_double_3x3_reduce_out)
        inception_4c_relu_double_3x3_reduce_out = self.inception_4c_relu_double_3x3_reduce(inception_4c_double_3x3_reduce_bn_out)
        inception_4c_double_3x3_1_out = self.inception_4c_double_3x3_1(inception_4c_double_3x3_reduce_bn_out)
        inception_4c_double_3x3_1_bn_out = self.inception_4c_double_3x3_1_bn(inception_4c_double_3x3_1_out)
        inception_4c_relu_double_3x3_1_out = self.inception_4c_relu_double_3x3_1(inception_4c_double_3x3_1_bn_out)
        inception_4c_double_3x3_2_out = self.inception_4c_double_3x3_2(inception_4c_double_3x3_1_bn_out)
        inception_4c_double_3x3_2_bn_out = self.inception_4c_double_3x3_2_bn(inception_4c_double_3x3_2_out)
        inception_4c_relu_double_3x3_2_out = self.inception_4c_relu_double_3x3_2(inception_4c_double_3x3_2_bn_out)
        inception_4c_pool_out = self.inception_4c_pool(inception_4b_output_out)
        inception_4c_pool_proj_out = self.inception_4c_pool_proj(inception_4c_pool_out)
        inception_4c_pool_proj_bn_out = self.inception_4c_pool_proj_bn(inception_4c_pool_proj_out)
        inception_4c_relu_pool_proj_out = self.inception_4c_relu_pool_proj(inception_4c_pool_proj_bn_out)
        inception_4c_output_out = torch.cat([inception_4c_1x1_bn_out, inception_4c_3x3_bn_out, inception_4c_double_3x3_2_bn_out, inception_4c_pool_proj_bn_out], 1)
        inception_4d_1x1_out = self.inception_4d_1x1(inception_4c_output_out)
        inception_4d_1x1_bn_out = self.inception_4d_1x1_bn(inception_4d_1x1_out)
        inception_4d_relu_1x1_out = self.inception_4d_relu_1x1(inception_4d_1x1_bn_out)
        inception_4d_3x3_reduce_out = self.inception_4d_3x3_reduce(inception_4c_output_out)
        inception_4d_3x3_reduce_bn_out = self.inception_4d_3x3_reduce_bn(inception_4d_3x3_reduce_out)
        inception_4d_relu_3x3_reduce_out = self.inception_4d_relu_3x3_reduce(inception_4d_3x3_reduce_bn_out)
        inception_4d_3x3_out = self.inception_4d_3x3(inception_4d_3x3_reduce_bn_out)
        inception_4d_3x3_bn_out = self.inception_4d_3x3_bn(inception_4d_3x3_out)
        inception_4d_relu_3x3_out = self.inception_4d_relu_3x3(inception_4d_3x3_bn_out)
        inception_4d_double_3x3_reduce_out = self.inception_4d_double_3x3_reduce(inception_4c_output_out)
        inception_4d_double_3x3_reduce_bn_out = self.inception_4d_double_3x3_reduce_bn(inception_4d_double_3x3_reduce_out)
        inception_4d_relu_double_3x3_reduce_out = self.inception_4d_relu_double_3x3_reduce(inception_4d_double_3x3_reduce_bn_out)
        inception_4d_double_3x3_1_out = self.inception_4d_double_3x3_1(inception_4d_double_3x3_reduce_bn_out)
        inception_4d_double_3x3_1_bn_out = self.inception_4d_double_3x3_1_bn(inception_4d_double_3x3_1_out)
        inception_4d_relu_double_3x3_1_out = self.inception_4d_relu_double_3x3_1(inception_4d_double_3x3_1_bn_out)
        inception_4d_double_3x3_2_out = self.inception_4d_double_3x3_2(inception_4d_double_3x3_1_bn_out)
        inception_4d_double_3x3_2_bn_out = self.inception_4d_double_3x3_2_bn(inception_4d_double_3x3_2_out)
        inception_4d_relu_double_3x3_2_out = self.inception_4d_relu_double_3x3_2(inception_4d_double_3x3_2_bn_out)
        inception_4d_pool_out = self.inception_4d_pool(inception_4c_output_out)
        inception_4d_pool_proj_out = self.inception_4d_pool_proj(inception_4d_pool_out)
        inception_4d_pool_proj_bn_out = self.inception_4d_pool_proj_bn(inception_4d_pool_proj_out)
        inception_4d_relu_pool_proj_out = self.inception_4d_relu_pool_proj(inception_4d_pool_proj_bn_out)
        inception_4d_output_out = torch.cat([inception_4d_1x1_bn_out, inception_4d_3x3_bn_out, inception_4d_double_3x3_2_bn_out, inception_4d_pool_proj_bn_out], 1)
        inception_4e_3x3_reduce_out = self.inception_4e_3x3_reduce(inception_4d_output_out)
        inception_4e_3x3_reduce_bn_out = self.inception_4e_3x3_reduce_bn(inception_4e_3x3_reduce_out)
        inception_4e_relu_3x3_reduce_out = self.inception_4e_relu_3x3_reduce(inception_4e_3x3_reduce_bn_out)
        inception_4e_3x3_out = self.inception_4e_3x3(inception_4e_3x3_reduce_bn_out)
        inception_4e_3x3_bn_out = self.inception_4e_3x3_bn(inception_4e_3x3_out)
        inception_4e_relu_3x3_out = self.inception_4e_relu_3x3(inception_4e_3x3_bn_out)
        inception_4e_double_3x3_reduce_out = self.inception_4e_double_3x3_reduce(inception_4d_output_out)
        inception_4e_double_3x3_reduce_bn_out = self.inception_4e_double_3x3_reduce_bn(inception_4e_double_3x3_reduce_out)
        inception_4e_relu_double_3x3_reduce_out = self.inception_4e_relu_double_3x3_reduce(inception_4e_double_3x3_reduce_bn_out)
        inception_4e_double_3x3_1_out = self.inception_4e_double_3x3_1(inception_4e_double_3x3_reduce_bn_out)
        inception_4e_double_3x3_1_bn_out = self.inception_4e_double_3x3_1_bn(inception_4e_double_3x3_1_out)
        inception_4e_relu_double_3x3_1_out = self.inception_4e_relu_double_3x3_1(inception_4e_double_3x3_1_bn_out)
        inception_4e_double_3x3_2_out = self.inception_4e_double_3x3_2(inception_4e_double_3x3_1_bn_out)
        inception_4e_double_3x3_2_bn_out = self.inception_4e_double_3x3_2_bn(inception_4e_double_3x3_2_out)
        inception_4e_relu_double_3x3_2_out = self.inception_4e_relu_double_3x3_2(inception_4e_double_3x3_2_bn_out)
        inception_4e_pool_out = self.inception_4e_pool(inception_4d_output_out)
        inception_4e_output_out = torch.cat([inception_4e_3x3_bn_out, inception_4e_double_3x3_2_bn_out, inception_4e_pool_out], 1)
        inception_5a_1x1_out = self.inception_5a_1x1(inception_4e_output_out)
        inception_5a_1x1_bn_out = self.inception_5a_1x1_bn(inception_5a_1x1_out)
        inception_5a_relu_1x1_out = self.inception_5a_relu_1x1(inception_5a_1x1_bn_out)
        inception_5a_3x3_reduce_out = self.inception_5a_3x3_reduce(inception_4e_output_out)
        inception_5a_3x3_reduce_bn_out = self.inception_5a_3x3_reduce_bn(inception_5a_3x3_reduce_out)
        inception_5a_relu_3x3_reduce_out = self.inception_5a_relu_3x3_reduce(inception_5a_3x3_reduce_bn_out)
        inception_5a_3x3_out = self.inception_5a_3x3(inception_5a_3x3_reduce_bn_out)
        inception_5a_3x3_bn_out = self.inception_5a_3x3_bn(inception_5a_3x3_out)
        inception_5a_relu_3x3_out = self.inception_5a_relu_3x3(inception_5a_3x3_bn_out)
        inception_5a_double_3x3_reduce_out = self.inception_5a_double_3x3_reduce(inception_4e_output_out)
        inception_5a_double_3x3_reduce_bn_out = self.inception_5a_double_3x3_reduce_bn(inception_5a_double_3x3_reduce_out)
        inception_5a_relu_double_3x3_reduce_out = self.inception_5a_relu_double_3x3_reduce(inception_5a_double_3x3_reduce_bn_out)
        inception_5a_double_3x3_1_out = self.inception_5a_double_3x3_1(inception_5a_double_3x3_reduce_bn_out)
        inception_5a_double_3x3_1_bn_out = self.inception_5a_double_3x3_1_bn(inception_5a_double_3x3_1_out)
        inception_5a_relu_double_3x3_1_out = self.inception_5a_relu_double_3x3_1(inception_5a_double_3x3_1_bn_out)
        inception_5a_double_3x3_2_out = self.inception_5a_double_3x3_2(inception_5a_double_3x3_1_bn_out)
        inception_5a_double_3x3_2_bn_out = self.inception_5a_double_3x3_2_bn(inception_5a_double_3x3_2_out)
        inception_5a_relu_double_3x3_2_out = self.inception_5a_relu_double_3x3_2(inception_5a_double_3x3_2_bn_out)
        inception_5a_pool_out = self.inception_5a_pool(inception_4e_output_out)
        inception_5a_pool_proj_out = self.inception_5a_pool_proj(inception_5a_pool_out)
        inception_5a_pool_proj_bn_out = self.inception_5a_pool_proj_bn(inception_5a_pool_proj_out)
        inception_5a_relu_pool_proj_out = self.inception_5a_relu_pool_proj(inception_5a_pool_proj_bn_out)
        inception_5a_output_out = torch.cat([inception_5a_1x1_bn_out, inception_5a_3x3_bn_out, inception_5a_double_3x3_2_bn_out, inception_5a_pool_proj_bn_out], 1)
        inception_5b_1x1_out = self.inception_5b_1x1(inception_5a_output_out)
        inception_5b_1x1_bn_out = self.inception_5b_1x1_bn(inception_5b_1x1_out)
        inception_5b_relu_1x1_out = self.inception_5b_relu_1x1(inception_5b_1x1_bn_out)
        inception_5b_3x3_reduce_out = self.inception_5b_3x3_reduce(inception_5a_output_out)
        inception_5b_3x3_reduce_bn_out = self.inception_5b_3x3_reduce_bn(inception_5b_3x3_reduce_out)
        inception_5b_relu_3x3_reduce_out = self.inception_5b_relu_3x3_reduce(inception_5b_3x3_reduce_bn_out)
        inception_5b_3x3_out = self.inception_5b_3x3(inception_5b_3x3_reduce_bn_out)
        inception_5b_3x3_bn_out = self.inception_5b_3x3_bn(inception_5b_3x3_out)
        inception_5b_relu_3x3_out = self.inception_5b_relu_3x3(inception_5b_3x3_bn_out)
        inception_5b_double_3x3_reduce_out = self.inception_5b_double_3x3_reduce(inception_5a_output_out)
        inception_5b_double_3x3_reduce_bn_out = self.inception_5b_double_3x3_reduce_bn(inception_5b_double_3x3_reduce_out)
        inception_5b_relu_double_3x3_reduce_out = self.inception_5b_relu_double_3x3_reduce(inception_5b_double_3x3_reduce_bn_out)
        inception_5b_double_3x3_1_out = self.inception_5b_double_3x3_1(inception_5b_double_3x3_reduce_bn_out)
        inception_5b_double_3x3_1_bn_out = self.inception_5b_double_3x3_1_bn(inception_5b_double_3x3_1_out)
        inception_5b_relu_double_3x3_1_out = self.inception_5b_relu_double_3x3_1(inception_5b_double_3x3_1_bn_out)
        inception_5b_double_3x3_2_out = self.inception_5b_double_3x3_2(inception_5b_double_3x3_1_bn_out)
        inception_5b_double_3x3_2_bn_out = self.inception_5b_double_3x3_2_bn(inception_5b_double_3x3_2_out)
        inception_5b_relu_double_3x3_2_out = self.inception_5b_relu_double_3x3_2(inception_5b_double_3x3_2_bn_out)
        inception_5b_pool_out = self.inception_5b_pool(inception_5a_output_out)
        inception_5b_pool_proj_out = self.inception_5b_pool_proj(inception_5b_pool_out)
        inception_5b_pool_proj_bn_out = self.inception_5b_pool_proj_bn(inception_5b_pool_proj_out)
        inception_5b_relu_pool_proj_out = self.inception_5b_relu_pool_proj(inception_5b_pool_proj_bn_out)
        inception_5b_output_out = torch.cat([inception_5b_1x1_bn_out, inception_5b_3x3_bn_out, inception_5b_double_3x3_2_bn_out, inception_5b_pool_proj_bn_out], 1)
        return inception_5b_output_out

    def train(self, mode=True):
        super(BNInception, self).train(mode)
        if self.bn_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    if self.bn_frozen:
                        for params in m.parameters():
                            params.requires_grad = False
        if self.partial_bn:
            for n, m in self.named_modules():
                if 'conv1' not in n and isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False


class InceptionV1_I3D(nn.Module):

    def __init__(self, pretrained=None, bn_eval=True, bn_frozen=False, partial_bn=False, modality='RGB'):
        super(InceptionV1_I3D, self).__init__()
        self.pretrained = pretrained
        self.bn_eval = bn_eval
        self.bn_frozen = bn_frozen
        self.partial_bn = partial_bn
        self.modality = modality
        inplace = True
        assert modality in ['RGB', 'Flow']
        if modality == 'RGB':
            self.conv1_7x7_s2 = nn.Conv3d(3, 64, kernel_size=(7, 7, 7), stride=(2, 2, 2), padding=(0, 0, 0), bias=False)
        else:
            self.conv1_7x7_s2 = nn.Conv3d(2, 64, kernel_size=(7, 7, 7), stride=(2, 2, 2), padding=(0, 0, 0), bias=False)
        self.conv1_7x7_s2_bn = nn.BatchNorm3d(64, eps=1e-05, affine=True)
        self.conv1_relu_7x7 = nn.ReLU(inplace)
        self.pool1_3x3_s2 = nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2), dilation=(1, 1, 1), ceil_mode=True)
        self.conv2_3x3_reduce = nn.Conv3d(64, 64, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.conv2_3x3_reduce_bn = nn.BatchNorm3d(64, eps=1e-05, affine=True)
        self.conv2_relu_3x3_reduce = nn.ReLU(inplace)
        self.conv2_3x3 = nn.Conv3d(64, 192, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        self.conv2_3x3_bn = nn.BatchNorm3d(192, eps=1e-05, affine=True)
        self.conv2_relu_3x3 = nn.ReLU(inplace)
        self.pool2_3x3_s2 = nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2), dilation=(1, 1, 1), ceil_mode=True)
        self.inception_3a_1x1 = nn.Conv3d(192, 64, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.inception_3a_1x1_bn = nn.BatchNorm3d(64, eps=1e-05, affine=True)
        self.inception_3a_relu_1x1 = nn.ReLU(inplace)
        self.inception_3a_branch1_3x3_reduce = nn.Conv3d(192, 96, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.inception_3a_branch1_3x3_reduce_bn = nn.BatchNorm3d(96, eps=1e-05, affine=True)
        self.inception_3a_branch1_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_3a_branch1_3x3 = nn.Conv3d(96, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        self.inception_3a_branch1_3x3_bn = nn.BatchNorm3d(128, eps=1e-05, affine=True)
        self.inception_3a_branch1_relu_3x3 = nn.ReLU(inplace)
        self.inception_3a_branch2_3x3_reduce = nn.Conv3d(192, 16, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.inception_3a_branch2_3x3_reduce_bn = nn.BatchNorm3d(16, eps=1e-05, affine=True)
        self.inception_3a_branch2_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_3a_branch2_3x3 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        self.inception_3a_branch2_3x3_bn = nn.BatchNorm3d(32, eps=1e-05, affine=True)
        self.inception_3a_branch2_relu_3x3 = nn.ReLU(inplace)
        self.inception_3a_pool = nn.MaxPool3d(3, stride=1, padding=1, ceil_mode=True)
        self.inception_3a_pool_proj = nn.Conv3d(192, 32, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.inception_3a_pool_proj_bn = nn.BatchNorm3d(32, eps=1e-05, affine=True)
        self.inception_3a_relu_pool_proj = nn.ReLU(inplace)
        self.inception_3b_1x1 = nn.Conv3d(256, 128, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.inception_3b_1x1_bn = nn.BatchNorm3d(128, eps=1e-05, affine=True)
        self.inception_3b_relu_1x1 = nn.ReLU(inplace)
        self.inception_3b_branch1_3x3_reduce = nn.Conv3d(256, 128, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.inception_3b_branch1_3x3_reduce_bn = nn.BatchNorm3d(128, eps=1e-05, affine=True)
        self.inception_3b_branch1_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_3b_branch1_3x3 = nn.Conv3d(128, 192, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        self.inception_3b_branch1_3x3_bn = nn.BatchNorm3d(192, eps=1e-05, affine=True)
        self.inception_3b_branch1_relu_3x3 = nn.ReLU(inplace)
        self.inception_3b_branch2_3x3_reduce = nn.Conv3d(256, 32, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.inception_3b_branch2_3x3_reduce_bn = nn.BatchNorm3d(32, eps=1e-05, affine=True)
        self.inception_3b_branch2_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_3b_branch2_3x3 = nn.Conv3d(32, 96, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        self.inception_3b_branch2_3x3_bn = nn.BatchNorm3d(96, eps=1e-05, affine=True)
        self.inception_3b_branch2_relu_3x3 = nn.ReLU(inplace)
        self.inception_3b_pool = nn.MaxPool3d(3, stride=1, padding=1, ceil_mode=True)
        self.inception_3b_pool_proj = nn.Conv3d(256, 64, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.inception_3b_pool_proj_bn = nn.BatchNorm3d(64, eps=1e-05, affine=True)
        self.inception_3b_relu_pool_proj = nn.ReLU(inplace)
        self.inception_3c_pool = nn.MaxPool3d((3, 3, 3), stride=(2, 2, 2), dilation=(1, 1, 1), ceil_mode=True)
        self.inception_4a_1x1 = nn.Conv3d(480, 192, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.inception_4a_1x1_bn = nn.BatchNorm3d(192, eps=1e-05, affine=True)
        self.inception_4a_relu_1x1 = nn.ReLU(inplace)
        self.inception_4a_branch1_3x3_reduce = nn.Conv3d(480, 96, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.inception_4a_branch1_3x3_reduce_bn = nn.BatchNorm3d(96, eps=1e-05, affine=True)
        self.inception_4a_branch1_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_4a_branch1_3x3 = nn.Conv3d(96, 208, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        self.inception_4a_branch1_3x3_bn = nn.BatchNorm3d(208, eps=1e-05, affine=True)
        self.inception_4a_branch1_relu_3x3 = nn.ReLU(inplace)
        self.inception_4a_branch2_3x3_reduce = nn.Conv3d(480, 16, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.inception_4a_branch2_3x3_reduce_bn = nn.BatchNorm3d(16, eps=1e-05, affine=True)
        self.inception_4a_branch2_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_4a_branch2_3x3 = nn.Conv3d(16, 48, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        self.inception_4a_branch2_3x3_bn = nn.BatchNorm3d(48, eps=1e-05, affine=True)
        self.inception_4a_branch2_relu_3x3 = nn.ReLU(inplace)
        self.inception_4a_pool = nn.MaxPool3d(3, stride=1, padding=1, ceil_mode=True)
        self.inception_4a_pool_proj = nn.Conv3d(480, 64, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.inception_4a_pool_proj_bn = nn.BatchNorm3d(64, eps=1e-05, affine=True)
        self.inception_4a_relu_pool_proj = nn.ReLU(inplace)
        self.inception_4b_1x1 = nn.Conv3d(512, 160, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.inception_4b_1x1_bn = nn.BatchNorm3d(160, eps=1e-05, affine=True)
        self.inception_4b_relu_1x1 = nn.ReLU(inplace)
        self.inception_4b_branch1_3x3_reduce = nn.Conv3d(512, 112, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.inception_4b_branch1_3x3_reduce_bn = nn.BatchNorm3d(112, eps=1e-05, affine=True)
        self.inception_4b_branch1_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_4b_branch1_3x3 = nn.Conv3d(112, 224, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        self.inception_4b_branch1_3x3_bn = nn.BatchNorm3d(224, eps=1e-05, affine=True)
        self.inception_4b_branch1_relu_3x3 = nn.ReLU(inplace)
        self.inception_4b_branch2_3x3_reduce = nn.Conv3d(512, 24, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.inception_4b_branch2_3x3_reduce_bn = nn.BatchNorm3d(24, eps=1e-05, affine=True)
        self.inception_4b_branch2_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_4b_branch2_3x3 = nn.Conv3d(24, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        self.inception_4b_branch2_3x3_bn = nn.BatchNorm3d(64, eps=1e-05, affine=True)
        self.inception_4b_branch2_relu_3x3 = nn.ReLU(inplace)
        self.inception_4b_pool = nn.MaxPool3d(3, stride=1, padding=1, ceil_mode=True)
        self.inception_4b_pool_proj = nn.Conv3d(512, 64, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.inception_4b_pool_proj_bn = nn.BatchNorm3d(64, eps=1e-05, affine=True)
        self.inception_4b_relu_pool_proj = nn.ReLU(inplace)
        self.inception_4c_1x1 = nn.Conv3d(512, 128, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.inception_4c_1x1_bn = nn.BatchNorm3d(128, eps=1e-05, affine=True)
        self.inception_4c_relu_1x1 = nn.ReLU(inplace)
        self.inception_4c_branch1_3x3_reduce = nn.Conv3d(512, 128, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.inception_4c_branch1_3x3_reduce_bn = nn.BatchNorm3d(128, eps=1e-05, affine=True)
        self.inception_4c_branch1_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_4c_branch1_3x3 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        self.inception_4c_branch1_3x3_bn = nn.BatchNorm3d(256, eps=1e-05, affine=True)
        self.inception_4c_branch1_relu_3x3 = nn.ReLU(inplace)
        self.inception_4c_branch2_3x3_reduce = nn.Conv3d(512, 24, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.inception_4c_branch2_3x3_reduce_bn = nn.BatchNorm3d(24, eps=1e-05, affine=True)
        self.inception_4c_branch2_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_4c_branch2_3x3 = nn.Conv3d(24, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        self.inception_4c_branch2_3x3_bn = nn.BatchNorm3d(64, eps=1e-05, affine=True)
        self.inception_4c_branch2_relu_3x3 = nn.ReLU(inplace)
        self.inception_4c_pool = nn.MaxPool3d(3, stride=1, padding=1, ceil_mode=True)
        self.inception_4c_pool_proj = nn.Conv3d(512, 64, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.inception_4c_pool_proj_bn = nn.BatchNorm3d(64, eps=1e-05, affine=True)
        self.inception_4c_relu_pool_proj = nn.ReLU(inplace)
        self.inception_4d_1x1 = nn.Conv3d(512, 112, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.inception_4d_1x1_bn = nn.BatchNorm3d(112, eps=1e-05, affine=True)
        self.inception_4d_relu_1x1 = nn.ReLU(inplace)
        self.inception_4d_branch1_3x3_reduce = nn.Conv3d(512, 144, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.inception_4d_branch1_3x3_reduce_bn = nn.BatchNorm3d(144, eps=1e-05, affine=True)
        self.inception_4d_branch1_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_4d_branch1_3x3 = nn.Conv3d(144, 288, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        self.inception_4d_branch1_3x3_bn = nn.BatchNorm3d(288, eps=1e-05, affine=True)
        self.inception_4d_branch1_relu_3x3 = nn.ReLU(inplace)
        self.inception_4d_branch2_3x3_reduce = nn.Conv3d(512, 32, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.inception_4d_branch2_3x3_reduce_bn = nn.BatchNorm3d(32, eps=1e-05, affine=True)
        self.inception_4d_branch2_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_4d_branch2_3x3 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        self.inception_4d_branch2_3x3_bn = nn.BatchNorm3d(64, eps=1e-05, affine=True)
        self.inception_4d_branch2_relu_3x3 = nn.ReLU(inplace)
        self.inception_4d_pool = nn.MaxPool3d(3, stride=1, padding=1, ceil_mode=True)
        self.inception_4d_pool_proj = nn.Conv3d(512, 64, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.inception_4d_pool_proj_bn = nn.BatchNorm3d(64, eps=1e-05, affine=True)
        self.inception_4d_relu_pool_proj = nn.ReLU(inplace)
        self.inception_4e_1x1 = nn.Conv3d(528, 256, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.inception_4e_1x1_bn = nn.BatchNorm3d(256, eps=1e-05, affine=True)
        self.inception_4e_relu_1x1 = nn.ReLU(inplace)
        self.inception_4e_branch1_3x3_reduce = nn.Conv3d(528, 160, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.inception_4e_branch1_3x3_reduce_bn = nn.BatchNorm3d(160, eps=1e-05, affine=True)
        self.inception_4e_branch1_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_4e_branch1_3x3 = nn.Conv3d(160, 320, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        self.inception_4e_branch1_3x3_bn = nn.BatchNorm3d(320, eps=1e-05, affine=True)
        self.inception_4e_branch1_relu_3x3 = nn.ReLU(inplace)
        self.inception_4e_branch2_3x3_reduce = nn.Conv3d(528, 32, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.inception_4e_branch2_3x3_reduce_bn = nn.BatchNorm3d(32, eps=1e-05, affine=True)
        self.inception_4e_branch2_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_4e_branch2_3x3 = nn.Conv3d(32, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        self.inception_4e_branch2_3x3_bn = nn.BatchNorm3d(128, eps=1e-05, affine=True)
        self.inception_4e_branch2_relu_3x3 = nn.ReLU(inplace)
        self.inception_4e_pool = nn.MaxPool3d(3, stride=1, padding=1, ceil_mode=True)
        self.inception_4e_pool_proj = nn.Conv3d(528, 128, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.inception_4e_pool_proj_bn = nn.BatchNorm3d(128, eps=1e-05, affine=True)
        self.inception_4e_relu_pool_proj = nn.ReLU(inplace)
        self.inception_4f_pool = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2), dilation=(1, 1, 1), ceil_mode=True)
        self.inception_5a_1x1 = nn.Conv3d(832, 256, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.inception_5a_1x1_bn = nn.BatchNorm3d(256, eps=1e-05, affine=True)
        self.inception_5a_relu_1x1 = nn.ReLU(inplace)
        self.inception_5a_branch1_3x3_reduce = nn.Conv3d(832, 160, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.inception_5a_branch1_3x3_reduce_bn = nn.BatchNorm3d(160, eps=1e-05, affine=True)
        self.inception_5a_branch1_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_5a_branch1_3x3 = nn.Conv3d(160, 320, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        self.inception_5a_branch1_3x3_bn = nn.BatchNorm3d(320, eps=1e-05, affine=True)
        self.inception_5a_branch1_relu_3x3 = nn.ReLU(inplace)
        self.inception_5a_branch2_3x3_reduce = nn.Conv3d(832, 32, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.inception_5a_branch2_3x3_reduce_bn = nn.BatchNorm3d(32, eps=1e-05, affine=True)
        self.inception_5a_branch2_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_5a_branch2_3x3 = nn.Conv3d(32, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        self.inception_5a_branch2_3x3_bn = nn.BatchNorm3d(128, eps=1e-05, affine=True)
        self.inception_5a_branch2_relu_3x3 = nn.ReLU(inplace)
        self.inception_5a_pool = nn.MaxPool3d(3, stride=1, padding=1, ceil_mode=True)
        self.inception_5a_pool_proj = nn.Conv3d(832, 128, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.inception_5a_pool_proj_bn = nn.BatchNorm3d(128, eps=1e-05, affine=True)
        self.inception_5a_relu_pool_proj = nn.ReLU(inplace)
        self.inception_5b_1x1 = nn.Conv3d(832, 384, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.inception_5b_1x1_bn = nn.BatchNorm3d(384, eps=1e-05, affine=True)
        self.inception_5b_relu_1x1 = nn.ReLU(inplace)
        self.inception_5b_branch1_3x3_reduce = nn.Conv3d(832, 192, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.inception_5b_branch1_3x3_reduce_bn = nn.BatchNorm3d(192, eps=1e-05, affine=True)
        self.inception_5b_branch1_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_5b_branch1_3x3 = nn.Conv3d(192, 384, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        self.inception_5b_branch1_3x3_bn = nn.BatchNorm3d(384, eps=1e-05, affine=True)
        self.inception_5b_branch1_relu_3x3 = nn.ReLU(inplace)
        self.inception_5b_branch2_3x3_reduce = nn.Conv3d(832, 48, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.inception_5b_branch2_3x3_reduce_bn = nn.BatchNorm3d(48, eps=1e-05, affine=True)
        self.inception_5b_branch2_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_5b_branch2_3x3 = nn.Conv3d(48, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        self.inception_5b_branch2_3x3_bn = nn.BatchNorm3d(128, eps=1e-05, affine=True)
        self.inception_5b_branch2_relu_3x3 = nn.ReLU(inplace)
        self.inception_5b_pool = nn.MaxPool3d(3, stride=1, padding=1, dilation=1, ceil_mode=True)
        self.inception_5b_pool_proj = nn.Conv3d(832, 128, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.inception_5b_pool_proj_bn = nn.BatchNorm3d(128, eps=1e-05, affine=True)
        self.inception_5b_relu_pool_proj = nn.ReLU(inplace)

    def init_weights(self):
        if isinstance(self.pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, self.pretrained, strict=False, logger=logger)
        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm3d):
                    constant_init(m, 1)

    def forward(self, input):
        conv1_7x7_s2_out = self.conv1_7x7_s2(F.pad(input, (2, 4, 2, 4, 2, 4)))
        conv1_7x7_s2_bn_out = self.conv1_7x7_s2_bn(conv1_7x7_s2_out)
        conv1_relu_7x7_out = self.conv1_relu_7x7(conv1_7x7_s2_bn_out)
        pool1_3x3_s2_out = self.pool1_3x3_s2(conv1_7x7_s2_bn_out)
        conv2_3x3_reduce_out = self.conv2_3x3_reduce(pool1_3x3_s2_out)
        conv2_3x3_reduce_bn_out = self.conv2_3x3_reduce_bn(conv2_3x3_reduce_out)
        conv2_relu_3x3_reduce_out = self.conv2_relu_3x3_reduce(conv2_3x3_reduce_bn_out)
        conv2_3x3_out = self.conv2_3x3(conv2_3x3_reduce_bn_out)
        conv2_3x3_bn_out = self.conv2_3x3_bn(conv2_3x3_out)
        conv2_relu_3x3_out = self.conv2_relu_3x3(conv2_3x3_bn_out)
        pool2_3x3_s2_out = self.pool2_3x3_s2(conv2_3x3_bn_out)
        inception_3a_1x1_out = self.inception_3a_1x1(pool2_3x3_s2_out)
        inception_3a_1x1_bn_out = self.inception_3a_1x1_bn(inception_3a_1x1_out)
        inception_3a_relu_1x1_out = self.inception_3a_relu_1x1(inception_3a_1x1_bn_out)
        inception_3a_branch1_3x3_reduce_out = self.inception_3a_branch1_3x3_reduce(pool2_3x3_s2_out)
        inception_3a_branch1_3x3_reduce_bn_out = self.inception_3a_branch1_3x3_reduce_bn(inception_3a_branch1_3x3_reduce_out)
        inception_3a_branch1_relu_3x3_reduce_out = self.inception_3a_branch1_relu_3x3_reduce(inception_3a_branch1_3x3_reduce_bn_out)
        inception_3a_branch1_3x3_out = self.inception_3a_branch1_3x3(inception_3a_branch1_3x3_reduce_bn_out)
        inception_3a_branch1_3x3_bn_out = self.inception_3a_branch1_3x3_bn(inception_3a_branch1_3x3_out)
        inception_3a_branch1_relu_3x3_out = self.inception_3a_branch1_relu_3x3(inception_3a_branch1_3x3_bn_out)
        inception_3a_branch2_3x3_reduce_out = self.inception_3a_branch2_3x3_reduce(pool2_3x3_s2_out)
        inception_3a_branch2_3x3_reduce_bn_out = self.inception_3a_branch2_3x3_reduce_bn(inception_3a_branch2_3x3_reduce_out)
        inception_3a_branch2_relu_3x3_reduce_out = self.inception_3a_branch2_relu_3x3_reduce(inception_3a_branch2_3x3_reduce_bn_out)
        inception_3a_branch2_3x3_out = self.inception_3a_branch2_3x3(inception_3a_branch2_3x3_reduce_bn_out)
        inception_3a_branch2_3x3_bn_out = self.inception_3a_branch2_3x3_bn(inception_3a_branch2_3x3_out)
        inception_3a_branch2_relu_3x3_out = self.inception_3a_branch2_relu_3x3(inception_3a_branch2_3x3_bn_out)
        inception_3a_pool_out = self.inception_3a_pool(pool2_3x3_s2_out)
        inception_3a_pool_proj_out = self.inception_3a_pool_proj(inception_3a_pool_out)
        inception_3a_pool_proj_bn_out = self.inception_3a_pool_proj_bn(inception_3a_pool_proj_out)
        inception_3a_relu_pool_proj_out = self.inception_3a_relu_pool_proj(inception_3a_pool_proj_bn_out)
        inception_3a_output_out = torch.cat([inception_3a_1x1_bn_out, inception_3a_branch1_3x3_bn_out, inception_3a_branch2_3x3_bn_out, inception_3a_pool_proj_bn_out], 1)
        inception_3b_1x1_out = self.inception_3b_1x1(inception_3a_output_out)
        inception_3b_1x1_bn_out = self.inception_3b_1x1_bn(inception_3b_1x1_out)
        inception_3b_relu_1x1_out = self.inception_3b_relu_1x1(inception_3b_1x1_bn_out)
        inception_3b_branch1_3x3_reduce_out = self.inception_3b_branch1_3x3_reduce(inception_3a_output_out)
        inception_3b_branch1_3x3_reduce_bn_out = self.inception_3b_branch1_3x3_reduce_bn(inception_3b_branch1_3x3_reduce_out)
        inception_3b_branch1_relu_3x3_reduce_out = self.inception_3b_branch1_relu_3x3_reduce(inception_3b_branch1_3x3_reduce_bn_out)
        inception_3b_branch1_3x3_out = self.inception_3b_branch1_3x3(inception_3b_branch1_3x3_reduce_bn_out)
        inception_3b_branch1_3x3_bn_out = self.inception_3b_branch1_3x3_bn(inception_3b_branch1_3x3_out)
        inception_3b_branch1_relu_3x3_out = self.inception_3b_branch1_relu_3x3(inception_3b_branch1_3x3_bn_out)
        inception_3b_branch2_3x3_reduce_out = self.inception_3b_branch2_3x3_reduce(inception_3a_output_out)
        inception_3b_branch2_3x3_reduce_bn_out = self.inception_3b_branch2_3x3_reduce_bn(inception_3b_branch2_3x3_reduce_out)
        inception_3b_branch2_relu_3x3_reduce_out = self.inception_3b_branch2_relu_3x3_reduce(inception_3b_branch2_3x3_reduce_bn_out)
        inception_3b_branch2_3x3_out = self.inception_3b_branch2_3x3(inception_3b_branch2_3x3_reduce_bn_out)
        inception_3b_branch2_3x3_bn_out = self.inception_3b_branch2_3x3_bn(inception_3b_branch2_3x3_out)
        inception_3b_branch2_relu_3x3_out = self.inception_3b_branch2_relu_3x3(inception_3b_branch2_3x3_bn_out)
        inception_3b_pool_out = self.inception_3b_pool(inception_3a_output_out)
        inception_3b_pool_proj_out = self.inception_3b_pool_proj(inception_3b_pool_out)
        inception_3b_pool_proj_bn_out = self.inception_3b_pool_proj_bn(inception_3b_pool_proj_out)
        inception_3b_relu_pool_proj_out = self.inception_3b_relu_pool_proj(inception_3b_pool_proj_bn_out)
        inception_3b_output_out = torch.cat([inception_3b_1x1_bn_out, inception_3b_branch1_3x3_bn_out, inception_3b_branch2_3x3_bn_out, inception_3b_pool_proj_bn_out], 1)
        inception_3c_pool_out = self.inception_3c_pool(inception_3b_output_out)
        inception_4a_1x1_out = self.inception_4a_1x1(inception_3c_pool_out)
        inception_4a_1x1_bn_out = self.inception_4a_1x1_bn(inception_4a_1x1_out)
        inception_4a_relu_1x1_out = self.inception_4a_relu_1x1(inception_4a_1x1_bn_out)
        inception_4a_branch1_3x3_reduce_out = self.inception_4a_branch1_3x3_reduce(inception_3c_pool_out)
        inception_4a_branch1_3x3_reduce_bn_out = self.inception_4a_branch1_3x3_reduce_bn(inception_4a_branch1_3x3_reduce_out)
        inception_4a_branch1_relu_3x3_reduce_out = self.inception_4a_branch1_relu_3x3_reduce(inception_4a_branch1_3x3_reduce_bn_out)
        inception_4a_branch1_3x3_out = self.inception_4a_branch1_3x3(inception_4a_branch1_3x3_reduce_bn_out)
        inception_4a_branch1_3x3_bn_out = self.inception_4a_branch1_3x3_bn(inception_4a_branch1_3x3_out)
        inception_4a_branch1_relu_3x3_out = self.inception_4a_branch1_relu_3x3(inception_4a_branch1_3x3_bn_out)
        inception_4a_branch2_3x3_reduce_out = self.inception_4a_branch2_3x3_reduce(inception_3c_pool_out)
        inception_4a_branch2_3x3_reduce_bn_out = self.inception_4a_branch2_3x3_reduce_bn(inception_4a_branch2_3x3_reduce_out)
        inception_4a_branch2_relu_3x3_reduce_out = self.inception_4a_branch2_relu_3x3_reduce(inception_4a_branch2_3x3_reduce_bn_out)
        inception_4a_branch2_3x3_out = self.inception_4a_branch2_3x3(inception_4a_branch2_3x3_reduce_bn_out)
        inception_4a_branch2_3x3_bn_out = self.inception_4a_branch2_3x3_bn(inception_4a_branch2_3x3_out)
        inception_4a_branch2_relu_3x3_out = self.inception_4a_branch2_relu_3x3(inception_4a_branch2_3x3_bn_out)
        inception_4a_pool_out = self.inception_4a_pool(inception_3c_pool_out)
        inception_4a_pool_proj_out = self.inception_4a_pool_proj(inception_4a_pool_out)
        inception_4a_pool_proj_bn_out = self.inception_4a_pool_proj_bn(inception_4a_pool_proj_out)
        inception_4a_relu_pool_proj_out = self.inception_4a_relu_pool_proj(inception_4a_pool_proj_bn_out)
        inception_4a_output_out = torch.cat([inception_4a_1x1_bn_out, inception_4a_branch1_3x3_bn_out, inception_4a_branch2_3x3_bn_out, inception_4a_pool_proj_bn_out], 1)
        inception_4b_1x1_out = self.inception_4b_1x1(inception_4a_output_out)
        inception_4b_1x1_bn_out = self.inception_4b_1x1_bn(inception_4b_1x1_out)
        inception_4b_relu_1x1_out = self.inception_4b_relu_1x1(inception_4b_1x1_bn_out)
        inception_4b_branch1_3x3_reduce_out = self.inception_4b_branch1_3x3_reduce(inception_4a_output_out)
        inception_4b_branch1_3x3_reduce_bn_out = self.inception_4b_branch1_3x3_reduce_bn(inception_4b_branch1_3x3_reduce_out)
        inception_4b_branch1_relu_3x3_reduce_out = self.inception_4b_branch1_relu_3x3_reduce(inception_4b_branch1_3x3_reduce_bn_out)
        inception_4b_branch1_3x3_out = self.inception_4b_branch1_3x3(inception_4b_branch1_3x3_reduce_bn_out)
        inception_4b_branch1_3x3_bn_out = self.inception_4b_branch1_3x3_bn(inception_4b_branch1_3x3_out)
        inception_4b_branch1_relu_3x3_out = self.inception_4b_branch1_relu_3x3(inception_4b_branch1_3x3_bn_out)
        inception_4b_branch2_3x3_reduce_out = self.inception_4b_branch2_3x3_reduce(inception_4a_output_out)
        inception_4b_branch2_3x3_reduce_bn_out = self.inception_4b_branch2_3x3_reduce_bn(inception_4b_branch2_3x3_reduce_out)
        inception_4b_branch2_relu_3x3_reduce_out = self.inception_4b_branch2_relu_3x3_reduce(inception_4b_branch2_3x3_reduce_bn_out)
        inception_4b_branch2_3x3_out = self.inception_4b_branch2_3x3(inception_4b_branch2_3x3_reduce_bn_out)
        inception_4b_branch2_3x3_bn_out = self.inception_4b_branch2_3x3_bn(inception_4b_branch2_3x3_out)
        inception_4b_branch2_relu_3x3_out = self.inception_4b_branch2_relu_3x3(inception_4b_branch2_3x3_bn_out)
        inception_4b_pool_out = self.inception_4b_pool(inception_4a_output_out)
        inception_4b_pool_proj_out = self.inception_4b_pool_proj(inception_4b_pool_out)
        inception_4b_pool_proj_bn_out = self.inception_4b_pool_proj_bn(inception_4b_pool_proj_out)
        inception_4b_relu_pool_proj_out = self.inception_4b_relu_pool_proj(inception_4b_pool_proj_bn_out)
        inception_4b_output_out = torch.cat([inception_4b_1x1_bn_out, inception_4b_branch1_3x3_bn_out, inception_4b_branch2_3x3_bn_out, inception_4b_pool_proj_bn_out], 1)
        inception_4c_1x1_out = self.inception_4c_1x1(inception_4b_output_out)
        inception_4c_1x1_bn_out = self.inception_4c_1x1_bn(inception_4c_1x1_out)
        inception_4c_relu_1x1_out = self.inception_4c_relu_1x1(inception_4c_1x1_bn_out)
        inception_4c_branch1_3x3_reduce_out = self.inception_4c_branch1_3x3_reduce(inception_4b_output_out)
        inception_4c_branch1_3x3_reduce_bn_out = self.inception_4c_branch1_3x3_reduce_bn(inception_4c_branch1_3x3_reduce_out)
        inception_4c_branch1_relu_3x3_reduce_out = self.inception_4c_branch1_relu_3x3_reduce(inception_4c_branch1_3x3_reduce_bn_out)
        inception_4c_branch1_3x3_out = self.inception_4c_branch1_3x3(inception_4c_branch1_3x3_reduce_bn_out)
        inception_4c_branch1_3x3_bn_out = self.inception_4c_branch1_3x3_bn(inception_4c_branch1_3x3_out)
        inception_4c_branch1_relu_3x3_out = self.inception_4c_branch1_relu_3x3(inception_4c_branch1_3x3_bn_out)
        inception_4c_branch2_3x3_reduce_out = self.inception_4c_branch2_3x3_reduce(inception_4b_output_out)
        inception_4c_branch2_3x3_reduce_bn_out = self.inception_4c_branch2_3x3_reduce_bn(inception_4c_branch2_3x3_reduce_out)
        inception_4c_branch2_relu_3x3_reduce_out = self.inception_4c_branch2_relu_3x3_reduce(inception_4c_branch2_3x3_reduce_bn_out)
        inception_4c_branch2_3x3_out = self.inception_4c_branch2_3x3(inception_4c_branch2_3x3_reduce_bn_out)
        inception_4c_branch2_3x3_bn_out = self.inception_4c_branch2_3x3_bn(inception_4c_branch2_3x3_out)
        inception_4c_branch2_relu_3x3_out = self.inception_4c_branch2_relu_3x3(inception_4c_branch2_3x3_bn_out)
        inception_4c_pool_out = self.inception_4c_pool(inception_4b_output_out)
        inception_4c_pool_proj_out = self.inception_4c_pool_proj(inception_4c_pool_out)
        inception_4c_pool_proj_bn_out = self.inception_4c_pool_proj_bn(inception_4c_pool_proj_out)
        inception_4c_relu_pool_proj_out = self.inception_4c_relu_pool_proj(inception_4c_pool_proj_bn_out)
        inception_4c_output_out = torch.cat([inception_4c_1x1_bn_out, inception_4c_branch1_3x3_bn_out, inception_4c_branch2_3x3_bn_out, inception_4c_pool_proj_bn_out], 1)
        inception_4d_1x1_out = self.inception_4d_1x1(inception_4c_output_out)
        inception_4d_1x1_bn_out = self.inception_4d_1x1_bn(inception_4d_1x1_out)
        inception_4d_relu_1x1_out = self.inception_4d_relu_1x1(inception_4d_1x1_bn_out)
        inception_4d_branch1_3x3_reduce_out = self.inception_4d_branch1_3x3_reduce(inception_4c_output_out)
        inception_4d_branch1_3x3_reduce_bn_out = self.inception_4d_branch1_3x3_reduce_bn(inception_4d_branch1_3x3_reduce_out)
        inception_4d_branch1_relu_3x3_reduce_out = self.inception_4d_branch1_relu_3x3_reduce(inception_4d_branch1_3x3_reduce_bn_out)
        inception_4d_branch1_3x3_out = self.inception_4d_branch1_3x3(inception_4d_branch1_3x3_reduce_bn_out)
        inception_4d_branch1_3x3_bn_out = self.inception_4d_branch1_3x3_bn(inception_4d_branch1_3x3_out)
        inception_4d_branch1_relu_3x3_out = self.inception_4d_branch1_relu_3x3(inception_4d_branch1_3x3_bn_out)
        inception_4d_branch2_3x3_reduce_out = self.inception_4d_branch2_3x3_reduce(inception_4c_output_out)
        inception_4d_branch2_3x3_reduce_bn_out = self.inception_4d_branch2_3x3_reduce_bn(inception_4d_branch2_3x3_reduce_out)
        inception_4d_branch2_relu_3x3_reduce_out = self.inception_4d_branch2_relu_3x3_reduce(inception_4d_branch2_3x3_reduce_bn_out)
        inception_4d_branch2_3x3_out = self.inception_4d_branch2_3x3(inception_4d_branch2_3x3_reduce_bn_out)
        inception_4d_branch2_3x3_bn_out = self.inception_4d_branch2_3x3_bn(inception_4d_branch2_3x3_out)
        inception_4d_branch2_relu_3x3_out = self.inception_4d_branch2_relu_3x3(inception_4d_branch2_3x3_bn_out)
        inception_4d_pool_out = self.inception_4d_pool(inception_4c_output_out)
        inception_4d_pool_proj_out = self.inception_4d_pool_proj(inception_4d_pool_out)
        inception_4d_pool_proj_bn_out = self.inception_4d_pool_proj_bn(inception_4d_pool_proj_out)
        inception_4d_relu_pool_proj_out = self.inception_4d_relu_pool_proj(inception_4d_pool_proj_bn_out)
        inception_4d_output_out = torch.cat([inception_4d_1x1_bn_out, inception_4d_branch1_3x3_bn_out, inception_4d_branch2_3x3_bn_out, inception_4d_pool_proj_bn_out], 1)
        inception_4e_1x1_out = self.inception_4e_1x1(inception_4d_output_out)
        inception_4e_1x1_bn_out = self.inception_4e_1x1_bn(inception_4e_1x1_out)
        inception_4e_relu_1x1_out = self.inception_4e_relu_1x1(inception_4e_1x1_bn_out)
        inception_4e_branch1_3x3_reduce_out = self.inception_4e_branch1_3x3_reduce(inception_4d_output_out)
        inception_4e_branch1_3x3_reduce_bn_out = self.inception_4e_branch1_3x3_reduce_bn(inception_4e_branch1_3x3_reduce_out)
        inception_4e_branch1_relu_3x3_reduce_out = self.inception_4e_branch1_relu_3x3_reduce(inception_4e_branch1_3x3_reduce_bn_out)
        inception_4e_branch1_3x3_out = self.inception_4e_branch1_3x3(inception_4e_branch1_3x3_reduce_bn_out)
        inception_4e_branch1_3x3_bn_out = self.inception_4e_branch1_3x3_bn(inception_4e_branch1_3x3_out)
        inception_4e_branch1_relu_3x3_out = self.inception_4e_branch1_relu_3x3(inception_4e_branch1_3x3_bn_out)
        inception_4e_branch2_3x3_reduce_out = self.inception_4e_branch2_3x3_reduce(inception_4d_output_out)
        inception_4e_branch2_3x3_reduce_bn_out = self.inception_4e_branch2_3x3_reduce_bn(inception_4e_branch2_3x3_reduce_out)
        inception_4e_branch2_relu_3x3_reduce_out = self.inception_4e_branch2_relu_3x3_reduce(inception_4e_branch2_3x3_reduce_bn_out)
        inception_4e_branch2_3x3_out = self.inception_4e_branch2_3x3(inception_4e_branch2_3x3_reduce_bn_out)
        inception_4e_branch2_3x3_bn_out = self.inception_4e_branch2_3x3_bn(inception_4e_branch2_3x3_out)
        inception_4e_branch2_relu_3x3_out = self.inception_4e_branch2_relu_3x3(inception_4e_branch2_3x3_bn_out)
        inception_4e_pool_out = self.inception_4e_pool(inception_4d_output_out)
        inception_4e_pool_proj_out = self.inception_4e_pool_proj(inception_4e_pool_out)
        inception_4e_pool_proj_bn_out = self.inception_4e_pool_proj_bn(inception_4e_pool_proj_out)
        inception_4e_relu_pool_proj_out = self.inception_4e_relu_pool_proj(inception_4e_pool_proj_bn_out)
        inception_4e_output_out = torch.cat([inception_4e_1x1_bn_out, inception_4e_branch1_3x3_bn_out, inception_4e_branch2_3x3_bn_out, inception_4e_pool_proj_bn_out], 1)
        inception_4f_pool_out = self.inception_4f_pool(inception_4e_output_out)
        inception_5a_1x1_out = self.inception_5a_1x1(inception_4f_pool_out)
        inception_5a_1x1_bn_out = self.inception_5a_1x1_bn(inception_5a_1x1_out)
        inception_5a_relu_1x1_out = self.inception_5a_relu_1x1(inception_5a_1x1_bn_out)
        inception_5a_branch1_3x3_reduce_out = self.inception_5a_branch1_3x3_reduce(inception_4f_pool_out)
        inception_5a_branch1_3x3_reduce_bn_out = self.inception_5a_branch1_3x3_reduce_bn(inception_5a_branch1_3x3_reduce_out)
        inception_5a_branch1_relu_3x3_reduce_out = self.inception_5a_branch1_relu_3x3_reduce(inception_5a_branch1_3x3_reduce_bn_out)
        inception_5a_branch1_3x3_out = self.inception_5a_branch1_3x3(inception_5a_branch1_3x3_reduce_bn_out)
        inception_5a_branch1_3x3_bn_out = self.inception_5a_branch1_3x3_bn(inception_5a_branch1_3x3_out)
        inception_5a_branch1_relu_3x3_out = self.inception_5a_branch1_relu_3x3(inception_5a_branch1_3x3_bn_out)
        inception_5a_branch2_3x3_reduce_out = self.inception_5a_branch2_3x3_reduce(inception_4f_pool_out)
        inception_5a_branch2_3x3_reduce_bn_out = self.inception_5a_branch2_3x3_reduce_bn(inception_5a_branch2_3x3_reduce_out)
        inception_5a_branch2_relu_3x3_reduce_out = self.inception_5a_branch2_relu_3x3_reduce(inception_5a_branch2_3x3_reduce_bn_out)
        inception_5a_branch2_3x3_out = self.inception_5a_branch2_3x3(inception_5a_branch2_3x3_reduce_bn_out)
        inception_5a_branch2_3x3_bn_out = self.inception_5a_branch2_3x3_bn(inception_5a_branch2_3x3_out)
        inception_5a_branch2_relu_3x3_out = self.inception_5a_branch2_relu_3x3(inception_5a_branch2_3x3_bn_out)
        inception_5a_pool_out = self.inception_5a_pool(inception_4f_pool_out)
        inception_5a_pool_proj_out = self.inception_5a_pool_proj(inception_5a_pool_out)
        inception_5a_pool_proj_bn_out = self.inception_5a_pool_proj_bn(inception_5a_pool_proj_out)
        inception_5a_relu_pool_proj_out = self.inception_5a_relu_pool_proj(inception_5a_pool_proj_bn_out)
        inception_5a_output_out = torch.cat([inception_5a_1x1_bn_out, inception_5a_branch1_3x3_bn_out, inception_5a_branch2_3x3_bn_out, inception_5a_pool_proj_bn_out], 1)
        inception_5b_1x1_out = self.inception_5b_1x1(inception_5a_output_out)
        inception_5b_1x1_bn_out = self.inception_5b_1x1_bn(inception_5b_1x1_out)
        inception_5b_relu_1x1_out = self.inception_5b_relu_1x1(inception_5b_1x1_bn_out)
        inception_5b_branch1_3x3_reduce_out = self.inception_5b_branch1_3x3_reduce(inception_5a_output_out)
        inception_5b_branch1_3x3_reduce_bn_out = self.inception_5b_branch1_3x3_reduce_bn(inception_5b_branch1_3x3_reduce_out)
        inception_5b_branch1_relu_3x3_reduce_out = self.inception_5b_branch1_relu_3x3_reduce(inception_5b_branch1_3x3_reduce_bn_out)
        inception_5b_branch1_3x3_out = self.inception_5b_branch1_3x3(inception_5b_branch1_3x3_reduce_bn_out)
        inception_5b_branch1_3x3_bn_out = self.inception_5b_branch1_3x3_bn(inception_5b_branch1_3x3_out)
        inception_5b_branch1_relu_3x3_out = self.inception_5b_branch1_relu_3x3(inception_5b_branch1_3x3_bn_out)
        inception_5b_branch2_3x3_reduce_out = self.inception_5b_branch2_3x3_reduce(inception_5a_output_out)
        inception_5b_branch2_3x3_reduce_bn_out = self.inception_5b_branch2_3x3_reduce_bn(inception_5b_branch2_3x3_reduce_out)
        inception_5b_branch2_relu_3x3_reduce_out = self.inception_5b_branch2_relu_3x3_reduce(inception_5b_branch2_3x3_reduce_bn_out)
        inception_5b_branch2_3x3_out = self.inception_5b_branch2_3x3(inception_5b_branch2_3x3_reduce_bn_out)
        inception_5b_branch2_3x3_bn_out = self.inception_5b_branch2_3x3_bn(inception_5b_branch2_3x3_out)
        inception_5b_branch2_relu_3x3_out = self.inception_5b_branch2_relu_3x3(inception_5b_branch2_3x3_bn_out)
        inception_5b_pool_out = self.inception_5b_pool(inception_5a_output_out)
        inception_5b_pool_proj_out = self.inception_5b_pool_proj(inception_5b_pool_out)
        inception_5b_pool_proj_bn_out = self.inception_5b_pool_proj_bn(inception_5b_pool_proj_out)
        inception_5b_relu_pool_proj_out = self.inception_5b_relu_pool_proj(inception_5b_pool_proj_bn_out)
        inception_5b_output_out = torch.cat([inception_5b_1x1_bn_out, inception_5b_branch1_3x3_bn_out, inception_5b_branch2_3x3_bn_out, inception_5b_pool_proj_bn_out], 1)
        return inception_5b_output_out

    def train(self, mode=True):
        super(InceptionV1_I3D, self).train(mode)
        if self.bn_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm3d):
                    m.eval()
                    if self.bn_frozen:
                        for params in m.parameters():
                            params.requires_grad = False
        if self.partial_bn:
            for n, m in self.named_modules():
                if 'conv1' not in n and isinstance(m, nn.BatchNorm3d):
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False


def conv1x3x3(in_planes, out_planes, spatial_stride=1, temporal_stride=1, dilation=1):
    """1x3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=(1, 3, 3), stride=(temporal_stride, spatial_stride, spatial_stride), padding=(0, dilation, dilation), dilation=dilation, bias=False)


def conv3x1x1(in_planes, out_planes, spatial_stride=1, temporal_stride=1, dilation=1, bias=False):
    """3x1x1 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=(3, 1, 1), stride=(temporal_stride, spatial_stride, spatial_stride), padding=(dilation, 0, 0), dilation=dilation, bias=bias)


class TrajConvFunction(Function):

    @staticmethod
    def forward(ctx, input, offset, weight, bias, stride=1, padding=0, dilation=1, deformable_groups=1, im2col_step=64):
        if input is not None and input.dim() != 5:
            raise ValueError('Expected 5D tensor as input, got {}D tensor instead.'.format(input.dim()))
        ctx.stride = _triple(stride)
        ctx.padding = _triple(padding)
        ctx.dilation = _triple(dilation)
        ctx.deformable_groups = deformable_groups
        ctx.im2col_step = im2col_step
        ctx.save_for_backward(input, offset, weight, bias)
        output = input.new(*TrajConvFunction._output_size(input, weight, ctx.padding, ctx.dilation, ctx.stride))
        ctx.bufs_ = [input.new(), input.new()]
        if not input.is_cuda:
            raise NotImplementedError
        else:
            if isinstance(input, torch.autograd.Variable):
                if not (isinstance(input.data, torch.FloatTensor) or isinstance(input.data, torch.DoubleTensor)):
                    raise NotImplementedError
            elif not (isinstance(input, torch.FloatTensor) or isinstance(input, torch.DoubleTensor)):
                raise NotImplementedError
            cur_im2col_step = min(ctx.im2col_step, input.shape[0])
            assert input.shape[0] % cur_im2col_step == 0, 'im2col step must divide batchsize'
            traj_conv_cuda.deform_3d_conv_forward_cuda(input, weight, bias, offset, output, ctx.bufs_[0], ctx.bufs_[1], weight.size(2), weight.size(3), weight.size(4), ctx.stride[0], ctx.stride[1], ctx.stride[2], ctx.padding[0], ctx.padding[1], ctx.padding[2], ctx.dilation[0], ctx.dilation[1], ctx.dilation[2], ctx.deformable_groups, cur_im2col_step)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, offset, weight, bias = ctx.saved_tensors
        grad_input = grad_offset = grad_weight = grad_bias = None
        if not grad_output.is_cuda:
            raise NotImplementedError
        else:
            if isinstance(grad_output, torch.autograd.Variable):
                if not (isinstance(grad_output.data, torch.FloatTensor) or isinstance(grad_output.data, torch.DoubleTensor)):
                    raise NotImplementedError
            elif not (isinstance(grad_output, torch.FloatTensor) or isinstance(grad_output, torch.DoubleTensor)):
                raise NotImplementedError
            cur_im2col_step = min(ctx.im2col_step, input.shape[0])
            assert input.shape[0] % cur_im2col_step == 0, 'im2col step must divide batchsize'
            if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
                grad_input = torch.zeros_like(input)
                grad_offset = torch.zeros_like(offset)
                traj_conv_cuda.deform_3d_conv_backward_input_cuda(input, offset, grad_output, grad_input, grad_offset, weight, bias, ctx.bufs_[0], weight.size(2), weight.size(3), weight.size(4), ctx.stride[0], ctx.stride[1], ctx.stride[2], ctx.padding[0], ctx.padding[1], ctx.padding[2], ctx.dilation[0], ctx.dilation[1], ctx.dilation[2], ctx.deformable_groups, cur_im2col_step)
            if ctx.needs_input_grad[2]:
                grad_weight = torch.zeros_like(weight)
                grad_bias = torch.zeros_like(bias)
                traj_conv_cuda.deform_3d_conv_backward_parameters_cuda(input, offset, grad_output, grad_weight, grad_bias, ctx.bufs_[0], ctx.bufs_[1], weight.size(2), weight.size(3), weight.size(4), ctx.stride[0], ctx.stride[1], ctx.stride[2], ctx.padding[0], ctx.padding[1], ctx.padding[2], ctx.dilation[0], ctx.dilation[1], ctx.dilation[2], ctx.deformable_groups, 1, cur_im2col_step)
        return grad_input, grad_offset, grad_weight, grad_bias, None, None, None, None, None

    @staticmethod
    def _output_size(input, weight, padding, dilation, stride):
        channels = weight.size(0)
        output_size = input.size(0), channels
        for d in range(input.dim() - 2):
            in_size = input.size(d + 2)
            pad = padding[d]
            kernel = dilation[d] * (weight.size(d + 2) - 1) + 1
            stride_ = stride[d]
            output_size += (in_size + 2 * pad - kernel) // stride_ + 1,
        if not all(map(lambda s: s > 0, output_size)):
            raise ValueError('convolution input is too small (output would be {})'.format('x'.join(map(str, output_size))))
        return output_size


class TrajConv(Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, num_deformable_groups=1, im2col_step=64, bias=True):
        super(TrajConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.padding = _triple(padding)
        self.dilation = _triple(dilation)
        self.num_deformable_groups = num_deformable_groups
        self.im2col_step = im2col_step
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = nn.Parameter(torch.zeros(0))
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias.nelement() != 0:
            self.bias.data.fill_(0.0)

    def forward(self, input, offset):
        return TrajConvFunction.apply(input, offset, self.weight, self.bias, self.stride, self.padding, self.dilation, self.num_deformable_groups, self.im2col_step)


def trajconv3x1x1(in_planes, out_planes, spatial_stride=1, temporal_stride=1, dilation=1, bias=False):
    """3x1x1 convolution with padding"""
    return TrajConv(in_planes, out_planes, kernel_size=(3, 1, 1), stride=(temporal_stride, spatial_stride, spatial_stride), padding=(dilation, 0, 0), dilation=dilation, bias=bias)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, spatial_stride=1, temporal_stride=1, dilation=1, downsample=None, style='pytorch', if_inflate=True, with_cp=False, with_trajectory=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv1x3x3(inplanes, planes, spatial_stride, 1, dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.if_inflate = if_inflate
        if self.if_inflate:
            self.conv1_t = conv3x1x1(planes, planes, 1, temporal_stride, dilation, bias=True)
            self.bn1_t = nn.BatchNorm3d(planes)
            if with_trajectory:
                self.conv2_t = trajconv3x1x1(planes, planes, bias=True)
            else:
                self.conv2_t = conv3x1x1(planes, planes, bias=True)
            self.bn2_t = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.spatial_stride = spatial_stride
        self.temporal_stride = temporal_stride
        self.dilation = dilation
        assert not with_cp
        self.with_trajectory = with_trajectory

    def forward(self, input):
        x, traj_src = input
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if self.if_inflate:
            out = self.conv1_t(out)
            out = self.bn1_t(out)
            out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.if_inflate:
            out = self.relu(out)
            if self.with_trajectory:
                assert traj_src[0] is not None
                out = self.conv2_t(out, traj_src[0])
            else:
                out = self.conv2_t(out)
            out = self.bn2_t(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out, traj_src[1:]


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, spatial_stride=1, temporal_stride=1, dilation=1, downsample=None, style='pytorch', if_inflate=True, with_cp=False, with_trajectory=False):
        """Bottleneck block for ResNet.
        If style is "pytorch", the stride-two layer is the 3x3 conv layer,
        if it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(Bottleneck, self).__init__()
        assert style in ['pytorch', 'caffe']
        self.inplanes = inplanes
        self.planes = planes
        if style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = spatial_stride
            self.conv1_stride_t = 1
            self.conv2_stride_t = temporal_stride
        else:
            self.conv1_stride = spatial_stride
            self.conv2_stride = 1
            self.conv1_stride_t = temporal_stride
            self.conv2_stride_t = 1
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, stride=(self.conv1_stride_t, self.conv1_stride, self.conv1_stride), bias=False)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(1, 3, 3), stride=(1, self.conv2_stride, self.conv2_stride), padding=(0, dilation, dilation), dilation=(1, dilation, dilation), bias=False)
        self.if_inflate = if_inflate
        if self.if_inflate:
            self.conv2_t = nn.Conv3d(planes, planes, kernel_size=(3, 1, 1), stride=(self.conv2_stride_t, 1, 1), padding=(1, 0, 0), dilation=1, bias=True)
            self.bn2_t = nn.BatchNorm3d(planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.spatial_tride = spatial_stride
        self.temporal_tride = temporal_stride
        self.dilation = dilation
        self.with_cp = with_cp
        self.with_trajectory = with_trajectory

    def forward(self, x):

        def _inner_forward(xx):
            x, traj_src = xx
            identity = x
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)
            if self.if_inflate:
                if self.with_trajectory:
                    assert traj_src is not None
                    out = self.conv2_t(out, traj_src[0])
                else:
                    out = self.conv2_t(out)
                out = self.bn2_t(out)
            out = self.conv3(out)
            out = self.bn3(out)
            if self.downsample is not None:
                identity = self.downsample(x)
            out += identity
            return out, traj_src[1:]
        if self.with_cp and x.requires_grad:
            out, traj_remains = cp.checkpoint(_inner_forward, x)
        else:
            out, traj_remains = _inner_forward(x)
        out = self.relu(out)
        return out, traj_remains


def make_res_layer(block, inplanes, planes, blocks, spatial_stride=1, temporal_stride=1, dilation=1, style='pytorch', inflate_freq=1, with_cp=False, traj_src_indices=-1):
    traj_src_indices = traj_src_indices if not isinstance(traj_src_indices, int) else (traj_src_indices,) * blocks
    inflate_freq = inflate_freq if not isinstance(inflate_freq, int) else (inflate_freq,) * blocks
    assert len(inflate_freq) == blocks
    downsample = None
    if spatial_stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(nn.Conv3d(inplanes, planes * block.expansion, kernel_size=1, stride=(temporal_stride, spatial_stride, spatial_stride), bias=False), nn.BatchNorm3d(planes * block.expansion))
    layers = []
    layers.append(block(inplanes, planes, spatial_stride, temporal_stride, dilation, downsample, style=style, if_inflate=inflate_freq[0] == 1, with_trajectory=traj_src_indices[0] > -1, with_cp=with_cp))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(block(inplanes, planes, 1, 1, dilation, style=style, if_inflate=inflate_freq[i] == 1, with_trajectory=traj_src_indices[i] > -1, with_cp=with_cp))
    return nn.Sequential(*layers)


class ResNet(nn.Module):
    """ResNet backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        num_stages (int): Resnet stages, normally 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means
            not freezing any parameters.
        bn_eval (bool): Whether to set BN layers to eval mode, namely, freeze
            running stats (mean and var).
        bn_frozen (bool): Whether to freeze weight and bias of BN layers.
        partial_bn (bool): Whether to freeze weight and bias of **all but the first** BN layers.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
    """
    arch_settings = {(18): (BasicBlock, (2, 2, 2, 2)), (34): (BasicBlock, (3, 4, 6, 3)), (50): (Bottleneck, (3, 4, 6, 3)), (101): (Bottleneck, (3, 4, 23, 3)), (152): (Bottleneck, (3, 8, 36, 3))}

    def __init__(self, depth, pretrained=None, num_stages=4, strides=(1, 2, 2, 2), dilations=(1, 1, 1, 1), out_indices=(0, 1, 2, 3), style='pytorch', frozen_stages=-1, bn_eval=True, bn_frozen=False, partial_bn=False, with_cp=False):
        super(ResNet, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError('invalid depth {} for resnet'.format(depth))
        self.depth = depth
        self.pretrained = pretrained
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.frozen_stages = frozen_stages
        self.bn_eval = bn_eval
        self.bn_frozen = bn_frozen
        self.partial_bn = partial_bn
        self.with_cp = with_cp
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            planes = 64 * 2 ** i
            res_layer = make_res_layer(self.block, self.inplanes, planes, num_blocks, stride=stride, dilation=dilation, style=self.style, with_cp=with_cp)
            self.inplanes = planes * self.block.expansion
            layer_name = 'layer{}'.format(i + 1)
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)
        self.feat_dim = self.block.expansion * 64 * 2 ** (len(self.stage_blocks) - 1)

    def init_weights(self):
        if isinstance(self.pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, self.pretrained, strict=False, logger=logger)
        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)

    def train(self, mode=True):
        super(ResNet, self).train(mode)
        if self.bn_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    if self.bn_frozen:
                        for params in m.parameters():
                            params.requires_grad = False
        if self.partial_bn:
            for i in range(1, self.frozen_stages + 1):
                mod = getattr(self, 'layer{}'.format(i))
                for m in mod.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.eval()
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False
        if mode and self.frozen_stages >= 0:
            for param in self.conv1.parameters():
                param.requires_grad = False
            for param in self.bn1.parameters():
                param.requires_grad = False
            self.bn1.eval()
            self.bn1.weight.requires_grad = False
            self.bn1.bias.requires_grad = False
            for i in range(1, self.frozen_stages + 1):
                mod = getattr(self, 'layer{}'.format(i))
                mod.eval()
                for param in mod.parameters():
                    param.requires_grad = False


SPATIAL_TEMPORAL_MODULES = Registry('spatial_temporal_module')


class NonLocalModule(nn.Module):

    def __init__(self, in_channels=1024, nonlocal_type='gaussian', dim=3, embed=True, embed_dim=None, sub_sample=True, use_bn=True):
        super(NonLocalModule, self).__init__()
        assert nonlocal_type in ['gaussian', 'dot', 'concat']
        assert dim == 2 or dim == 3
        self.nonlocal_type = nonlocal_type
        self.embed = embed
        self.embed_dim = embed_dim if embed_dim is not None else in_channels // 2
        self.sub_sample = sub_sample
        self.use_bn = use_bn
        if self.embed:
            if dim == 2:
                self.theta = nn.Conv2d(in_channels, self.embed_dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
                self.phi = nn.Conv2d(in_channels, self.embed_dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
                self.g = nn.Conv2d(in_channels, self.embed_dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
            elif dim == 3:
                self.theta = nn.Conv3d(in_channels, self.embed_dim, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
                self.phi = nn.Conv3d(in_channels, self.embed_dim, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
                self.g = nn.Conv3d(in_channels, self.embed_dim, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        if self.nonlocal_type == 'gaussian':
            self.softmax = nn.Softmax(dim=2)
        elif self.nonlocal_type == 'concat':
            if dim == 2:
                self.concat_proj = nn.Sequential(nn.Conv2d(self.embed_dim * 2, 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)), nn.ReLU())
            elif dim == 3:
                self.concat_proj = nn.Sequential(nn.Conv3d(self.embed_dim * 2, 1, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0)), nn.ReLU())
        if sub_sample:
            if dim == 2:
                self.max_pool = nn.MaxPool2d(kernel_size=(2, 2))
            elif dim == 3:
                self.max_pool = nn.MaxPool3d(kernel_size=(1, 2, 2))
            self.g = nn.Sequential(self.max_pool, self.g)
            self.phi = nn.Sequential(self.max_pool, self.phi)
        if dim == 2:
            self.W = nn.Conv2d(self.embed_dim, in_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        elif dim == 3:
            self.W = nn.Conv3d(self.embed_dim, in_channels, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        if use_bn:
            if dim == 2:
                self.bn = nn.BatchNorm2d(in_channels, eps=1e-05, momentum=0.9, affine=True)
            elif dim == 3:
                self.bn = nn.BatchNorm3d(in_channels, eps=1e-05, momentum=0.9, affine=True)
            self.W = nn.Sequential(self.W, self.bn)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                kaiming_init(m)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                constant_init(m, 0)

    def forward(self, input):
        if self.embed:
            theta = self.theta(input)
            phi = self.phi(input)
            g = self.g(input)
        else:
            theta = input
            phi = input
            g = input
        if self.nonlocal_type in ['gaussian', 'dot']:
            theta = theta.reshape(theta.shape[:2] + (-1,))
            phi = phi.reshape(theta.shape[:2] + (-1,))
            g = g.reshape(theta.shape[:2] + (-1,))
            theta_phi = torch.matmul(theta.transpose(1, 2), phi)
            if self.nonlocal_type == 'gaussian':
                p = self.softmax(theta_phi)
            elif self.nonlocal_type == 'dot':
                N = theta_phi.size(-1)
                p = theta_phi / N
        elif self.non_local_type == 'concat':
            theta = theta.reshape(theta.shape[:2] + (-1, 1))
            phi = phi.reshape(theta.shape[:2] + (1, -1))
            theta_x = theta.repeat(1, 1, 1, phi.size(3))
            phi_x = phi.repeat(1, 1, theta.size(2), 1)
            theta_phi = torch.cat([theta_x, phi_x], dim=1)
            theta_phi = self.concat_proj(theta_phi)
            theta_phi = theta_phi.squeeze()
            N = theta_phi.size(-1)
            p = theta_phi / N
        else:
            NotImplementedError
        y = torch.matmul(g, p.transpose(1, 2))
        y = y.reshape(y.shape[:2] + input.shape[2:])
        z = self.W(y) + input
        return z


def rgetattr(obj, attr, *args):

    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


def rhasattr(obj, attr, *args):

    def _hasattr(obj, attr):
        if hasattr(obj, attr):
            return getattr(obj, attr)
        else:
            return None
    return functools.reduce(_hasattr, [obj] + attr.split('.')) is not None


class ResNet_I3D(nn.Module):
    """ResNet_I3D backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        num_stages (int): Resnet stages, normally 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means
            not freezing any parameters.
        bn_eval (bool): Whether to set BN layers to eval mode, namely, freeze
            running stats (mean and var).
        bn_frozen (bool): Whether to freeze weight and bias of BN layers.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
    """
    arch_settings = {(18): (BasicBlock, (2, 2, 2, 2)), (34): (BasicBlock, (3, 4, 6, 3)), (50): (Bottleneck, (3, 4, 6, 3)), (101): (Bottleneck, (3, 4, 23, 3)), (152): (Bottleneck, (3, 8, 36, 3))}

    def __init__(self, depth, pretrained=None, pretrained2d=True, num_stages=4, spatial_strides=(1, 2, 2, 2), temporal_strides=(1, 1, 1, 1), dilations=(1, 1, 1, 1), out_indices=(0, 1, 2, 3), conv1_kernel_t=5, conv1_stride_t=2, pool1_kernel_t=1, pool1_stride_t=2, style='pytorch', frozen_stages=-1, inflate_freq=(1, 1, 1, 1), inflate_stride=(1, 1, 1, 1), inflate_style='3x1x1', nonlocal_stages=(-1,), nonlocal_freq=(0, 1, 1, 0), nonlocal_cfg=None, no_pool2=False, bn_eval=True, bn_frozen=False, partial_bn=False, with_cp=False):
        super(ResNet_I3D, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError('invalid depth {} for resnet'.format(depth))
        self.depth = depth
        self.pretrained = pretrained
        self.pretrained2d = pretrained2d
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.spatial_strides = spatial_strides
        self.temporal_strides = temporal_strides
        self.dilations = dilations
        assert len(spatial_strides) == len(temporal_strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.frozen_stages = frozen_stages
        self.inflate_freqs = inflate_freq if not isinstance(inflate_freq, int) else (inflate_freq,) * num_stages
        self.inflate_style = inflate_style
        self.nonlocal_stages = nonlocal_stages
        self.nonlocal_freqs = nonlocal_freq if not isinstance(nonlocal_freq, int) else (nonlocal_freq,) * num_stages
        self.nonlocal_cfg = nonlocal_cfg
        self.bn_eval = bn_eval
        self.bn_frozen = bn_frozen
        self.partial_bn = partial_bn
        self.with_cp = with_cp
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = 64
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(conv1_kernel_t, 7, 7), stride=(conv1_stride_t, 2, 2), padding=((conv1_kernel_t - 1) // 2, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(pool1_kernel_t, 3, 3), stride=(pool1_stride_t, 2, 2), padding=(pool1_kernel_t // 2, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
        self.no_pool2 = no_pool2
        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            spatial_stride = spatial_strides[i]
            temporal_stride = temporal_strides[i]
            dilation = dilations[i]
            planes = 64 * 2 ** i
            res_layer = make_res_layer(self.block, self.inplanes, planes, num_blocks, spatial_stride=spatial_stride, temporal_stride=temporal_stride, dilation=dilation, style=self.style, inflate_freq=self.inflate_freqs[i], inflate_style=self.inflate_style, nonlocal_freq=self.nonlocal_freqs[i], nonlocal_cfg=self.nonlocal_cfg if i in self.nonlocal_stages else None, with_cp=with_cp)
            self.inplanes = planes * self.block.expansion
            layer_name = 'layer{}'.format(i + 1)
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)
        self.feat_dim = self.block.expansion * 64 * 2 ** (len(self.stage_blocks) - 1)

    def init_weights(self):
        if isinstance(self.pretrained, str):
            logger = logging.getLogger()
            if self.pretrained2d:
                resnet2d = ResNet(self.depth)
                load_checkpoint(resnet2d, self.pretrained, strict=False, logger=logger)
                for name, module in self.named_modules():
                    if isinstance(module, NonLocalModule):
                        module.init_weights()
                    elif isinstance(module, nn.Conv3d) and rhasattr(resnet2d, name):
                        new_weight = rgetattr(resnet2d, name).weight.data.unsqueeze(2).expand_as(module.weight) / module.weight.data.shape[2]
                        module.weight.data.copy_(new_weight)
                        logging.info('{}.weight loaded from weights file into {}'.format(name, new_weight.shape))
                        if hasattr(module, 'bias') and module.bias is not None:
                            new_bias = rgetattr(resnet2d, name).bias.data
                            module.bias.data.copy_(new_bias)
                            logging.info('{}.bias loaded from weights file into {}'.format(name, new_bias.shape))
                    elif isinstance(module, nn.BatchNorm3d) and rhasattr(resnet2d, name):
                        for attr in ['weight', 'bias', 'running_mean', 'running_var']:
                            logging.info('{}.{} loaded from weights file into {}'.format(name, attr, getattr(rgetattr(resnet2d, name), attr).shape))
                            setattr(module, attr, getattr(rgetattr(resnet2d, name), attr))
            else:
                load_checkpoint(self, self.pretrained, strict=False, logger=logger)
        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm3d):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
            if self.no_pool2:
                pass
            elif i == 0:
                x = self.pool2(x)
        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)

    def train(self, mode=True):
        super(ResNet_I3D, self).train(mode)
        if self.bn_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm3d):
                    m.eval()
                    if self.bn_frozen:
                        for params in m.parameters():
                            params.requires_grad = False
        if self.partial_bn:
            for i in range(1, self.frozen_stages + 1):
                mod = getattr(self, 'layer{}'.format(i))
                for m in mod.modules():
                    if isinstance(m, nn.BatchNorm3d):
                        m.eval()
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False
        if mode and self.frozen_stages >= 0:
            for param in self.conv1.parameters():
                param.requires_grad = False
            for param in self.bn1.parameters():
                param.requires_grad = False
            self.bn1.eval()
            self.bn1.weight.requires_grad = False
            self.bn1.bias.requires_grad = False
            for i in range(1, self.frozen_stages + 1):
                mod = getattr(self, 'layer{}'.format(i))
                mod.eval()
                for param in mod.parameters():
                    param.requires_grad = False


class pathway(nn.Module):
    arch_settings = {(18): (BasicBlock, (2, 2, 2, 2)), (34): (BasicBlock, (3, 4, 6, 3)), (50): (Bottleneck, (3, 4, 6, 3)), (101): (Bottleneck, (3, 4, 23, 3)), (152): (Bottleneck, (3, 8, 36, 3))}

    def __init__(self, depth, num_stages=4, channel_mul_inv=1, lateral=True, alpha=8, beta_inv=8, lateral_type='conv', lateral_op='concat', conv1_kernel_t=1, conv1_stride_t=1, pool1_kernel_t=1, pool1_stride_t=1, spatial_strides=(1, 2, 2, 2), dilations=(1, 1, 1, 1), style='pytorch', inflate_freqs=(1, 1, 1, 1), inflate_style='3x1x1', nonlocal_stages=(-1,), nonlocal_freqs=(0, 1, 1, 0), nonlocal_cfg=None, with_cp=False):
        super(pathway, self).__init__()
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = 64 // channel_mul_inv
        if lateral:
            if lateral_type == 'toC':
                lateral_inplanes = self.inplanes * alpha // beta_inv
            elif lateral_type == 'sampling':
                lateral_inplanes = self.inplanes // beta_inv
            elif lateral_type == 'conv':
                lateral_inplanes = self.inplanes * 2 // beta_inv
                self.conv1_lateral = nn.Conv3d(self.inplanes // beta_inv, self.inplanes * 2 // beta_inv, kernel_size=(5, 1, 1), stride=(alpha, 1, 1), padding=(2, 0, 0), bias=False)
            else:
                raise NotImplementedError
        else:
            lateral_inplanes = 0
        self.conv1 = nn.Conv3d(3, 64 // channel_mul_inv, kernel_size=(conv1_kernel_t, 7, 7), stride=(conv1_stride_t, 2, 2), padding=((conv1_kernel_t - 1) // 2, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64 // channel_mul_inv)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(pool1_kernel_t, 3, 3), stride=(pool1_stride_t, 2, 2), padding=(pool1_kernel_t // 2, 1, 1))
        self.res_layers = []
        self.lateral_connections = []
        for i, num_blocks in enumerate(self.stage_blocks):
            spatial_stride = spatial_strides[i]
            temporal_stride = 1
            dilation = dilations[i]
            planes = 64 * 2 ** i // channel_mul_inv
            res_layer = make_res_layer(self.block, self.inplanes, planes, num_blocks, lateral_inplanes=lateral_inplanes, spatial_stride=spatial_stride, temporal_stride=temporal_stride, dilation=dilation, style=style, inflate_freq=inflate_freqs[i], inflate_style=inflate_style, nonlocal_freq=nonlocal_freqs[i], nonlocal_cfg=nonlocal_cfg if i in nonlocal_stages else None, with_cp=with_cp)
            self.inplanes = planes * self.block.expansion
            if lateral:
                if lateral_type == 'toC':
                    lateral_inplanes = self.inplanes * alpha // beta_inv
                elif lateral_type == 'sampling':
                    lateral_inplanes = self.inplanes // beta_inv
                elif lateral_type == 'conv':
                    lateral_inplanes = self.inplanes * 2 // beta_inv
                    lateral_name = 'layer{}_lateral'.format(i + 1)
                    setattr(self, lateral_name, nn.Conv3d(self.inplanes // beta_inv, self.inplanes * 2 // beta_inv, kernel_size=(5, 1, 1), stride=(alpha, 1, 1), padding=(2, 0, 0), bias=False))
                    self.lateral_connections.append(lateral_name)
            else:
                lateral_inplanes = 0
            layer_name = 'layer{}'.format(i + 1)
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)
        self.feat_dim = self.block.expansion * 64 * 2 ** (len(self.stage_blocks) - 1)


class ResNet_I3D_SlowFast(nn.Module):
    """ResNet_I3D backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        num_stages (int): Resnet stages, normally 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means
            not freezing any parameters.
        bn_eval (bool): Whether to set BN layers to eval mode, namely, freeze
            running stats (mean and var).
        bn_frozen (bool): Whether to freeze weight and bias of BN layers.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
    """
    arch_settings = {(18): (BasicBlock, (2, 2, 2, 2)), (34): (BasicBlock, (3, 4, 6, 3)), (50): (Bottleneck, (3, 4, 6, 3)), (101): (Bottleneck, (3, 4, 23, 3)), (152): (Bottleneck, (3, 8, 36, 3))}

    def __init__(self, depth, tau=16, alpha=8, beta_inv=8, pretrained_slow=None, pretrained_fast=None, num_stages=4, slow_only=False, fast_only=False, lateral_type='conv', lateral_op='concat', spatial_strides=(1, 2, 2, 2), dilations=(1, 1, 1, 1), out_indices=(0, 1, 2, 3), slow_conv1_kernel_t=1, slow_conv1_stride_t=1, slow_pool1_kernel_t=1, slow_pool1_stride_t=1, fast_conv1_kernel_t=5, fast_conv1_stride_t=1, fast_pool1_kernel_t=1, fast_pool1_stride_t=1, style='pytorch', frozen_stages=-1, slow_inflate_freq=(0, 0, 1, 1), fast_inflate_freq=(1, 1, 1, 1), inflate_stride=(1, 1, 1, 1), inflate_style='3x1x1', nonlocal_stages=(-1,), nonlocal_freq=(0, 1, 1, 0), nonlocal_cfg=None, bn_eval=True, bn_frozen=False, partial_bn=False, with_cp=False):
        super(ResNet_I3D_SlowFast, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError('invalid depth {} for resnet'.format(depth))
        self.depth = depth
        self.tau = tau
        self.alpha = alpha
        self.beta_inv = beta_inv
        self.pretrained_slow = pretrained_slow
        self.pretrained_fast = pretrained_fast
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.slow_only = slow_only
        self.fast_only = fast_only
        assert not (self.slow_only and self.fast_only)
        self.lateral_type = lateral_type
        self.lateral_op = lateral_op
        assert lateral_type in ['conv']
        assert lateral_op in ['concat']
        self.spatial_strides = spatial_strides
        self.dilations = dilations
        assert len(spatial_strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.frozen_stages = frozen_stages
        self.slow_inflate_freq = slow_inflate_freq
        if isinstance(slow_inflate_freq, int):
            self.slow_inflate_freq = (slow_inflate_freq,) * num_stages
        self.fast_inflate_freq = fast_inflate_freq
        if isinstance(fast_inflate_freq, int):
            self.fast_inflate_freq = (fast_inflate_freq,) * num_stages
        self.inflate_style = inflate_style
        self.nonlocal_stages = nonlocal_stages
        self.nonlocal_freqs = nonlocal_freq if not isinstance(nonlocal_freq, int) else (nonlocal_freq,) * num_stages
        self.nonlocal_cfg = nonlocal_cfg
        self.bn_eval = bn_eval
        self.bn_frozen = bn_frozen
        self.partial_bn = partial_bn
        self.with_cp = with_cp
        if not self.fast_only:
            self.slow_path = pathway(depth, num_stages=num_stages, channel_mul_inv=1, lateral=not self.slow_only, alpha=alpha, beta_inv=beta_inv, lateral_type=lateral_type, lateral_op=lateral_op, conv1_kernel_t=slow_conv1_kernel_t, conv1_stride_t=slow_conv1_stride_t, pool1_kernel_t=slow_pool1_kernel_t, pool1_stride_t=slow_pool1_stride_t, spatial_strides=spatial_strides, dilations=dilations, style=style, inflate_freqs=self.slow_inflate_freq, inflate_style=inflate_style, nonlocal_stages=nonlocal_stages, nonlocal_freqs=nonlocal_freq, nonlocal_cfg=nonlocal_cfg, with_cp=with_cp)
        if not self.slow_only:
            self.fast_path = pathway(depth, num_stages=num_stages, channel_mul_inv=beta_inv, lateral=False, conv1_kernel_t=fast_conv1_kernel_t, conv1_stride_t=fast_conv1_stride_t, pool1_kernel_t=fast_pool1_kernel_t, pool1_stride_t=fast_pool1_stride_t, spatial_strides=spatial_strides, dilations=dilations, style=style, inflate_freqs=self.fast_inflate_freq, inflate_style=inflate_style, nonlocal_stages=nonlocal_stages, nonlocal_freqs=nonlocal_freq, nonlocal_cfg=nonlocal_cfg, with_cp=with_cp)

    def init_weights(self):
        logger = logging.getLogger()
        if not self.fast_only:
            if self.pretrained_slow:
                resnet2d = ResNet(self.depth)
                load_checkpoint(resnet2d, self.pretrained_slow, strict=False, logger=logger)
                for name, module in self.slow_path.named_modules():
                    if isinstance(module, NonLocalModule):
                        module.init_weights()
                    elif isinstance(module, nn.Conv3d) and rhasattr(resnet2d, name):
                        old_weight = rgetattr(resnet2d, name).weight.data
                        old_shape = old_weight.shape
                        new_shape = module.weight.data.shape
                        if new_shape[1] != old_shape[1]:
                            new_ch = new_shape[1] - old_shape[1]
                            pad_shape = old_shape
                            pad_shape = pad_shape[:1] + (new_ch,) + pad_shape[2:]
                            old_weight = torch.cat((old_weight, torch.zeros(pad_shape).type_as(old_weight)), dim=1)
                        new_weight = old_weight.unsqueeze(2).expand_as(module.weight.data) / new_shape[2]
                        module.weight.data.copy_(new_weight)
                        logging.info('{}.weight loaded from weights file into {}'.format(name, new_weight.shape))
                        if hasattr(module, 'bias') and module.bias is not None:
                            new_bias = rgetattr(resnet2d, name).bias.data
                            module.bias.data.copy_(new_bias)
                            logging.info('{}.bias loaded from weights file into {}'.format(name, new_bias.shape))
                    elif isinstance(module, nn.BatchNorm3d) and rhasattr(resnet2d, name):
                        for attr in ['weight', 'bias', 'running_mean', 'running_var']:
                            logging.info('{}.{} loaded from weights file into {}'.format(name, attr, getattr(rgetattr(resnet2d, name), attr).shape))
                            setattr(module, attr, getattr(rgetattr(resnet2d, name), attr))
                    else:
                        None
            else:
                for m in self.slow_path.modules():
                    if isinstance(m, nn.Conv3d):
                        kaiming_init(m)
                    elif isinstance(m, nn.BatchNorm3d):
                        constant_init(m, 1)
        if not self.slow_only:
            if self.pretrained_fast:
                resnet2d = ResNet(self.depth, base_channels=64 // self.beta_inv)
                load_checkpoint(resnet2d, self.pretrained_fast, strict=False, logger=logger)
                for name, module in self.fast_path.named_modules():
                    if isinstance(module, NonLocalModule):
                        module.init_weights()
                    elif isinstance(module, nn.Conv3d) and rhasattr(resnet2d, name):
                        old_weight = rgetattr(resnet2d, name).weight.data
                        old_shape = old_weight.shape
                        new_shape = module.weight.data.shape
                        if new_shape[1] != old_shape[1]:
                            new_ch = new_shape[1] - old_shape[1]
                            pad_shape = old_shape
                            pad_shape = pad_shape[:1] + (new_ch,) + pad_shape[2:]
                            old_weight = torch.cat((old_weight, torch.zeros(pad_shape).type_as(old_weight)), dim=1)
                        new_weight = old_weight.unsqueeze(2).expand_as(module.weight.data) / new_shape[2]
                        module.weight.data.copy_(new_weight)
                        logging.info('{}.weight loaded from weights file into {}'.format(name, new_weight.shape))
                        if hasattr(module, 'bias') and module.bias is not None:
                            new_bias = rgetattr(resnet2d, name).bias.data
                            module.bias.data.copy_(new_bias)
                            logging.info('{}.bias loaded from weights file into {}'.format(name, new_bias.shape))
                    elif isinstance(module, nn.BatchNorm3d) and rhasattr(resnet2d, name):
                        for attr in ['weight', 'bias', 'running_mean', 'running_var']:
                            logging.info('{}.{} loaded from weights file into {}'.format(name, attr, getattr(rgetattr(resnet2d, name), attr).shape))
                            setattr(module, attr, getattr(rgetattr(resnet2d, name), attr))
                    else:
                        None
            else:
                for m in self.fast_path.modules():
                    if isinstance(m, nn.Conv3d):
                        kaiming_init(m)
                    elif isinstance(m, nn.BatchNorm3d):
                        constant_init(m, 1)

    def forward(self, x):
        if not self.fast_only:
            x_slow = x[:, :, ::self.tau, :, :]
            x_slow = self.slow_path.conv1(x_slow)
            x_slow = self.slow_path.bn1(x_slow)
            x_slow = self.slow_path.relu(x_slow)
            x_slow = self.slow_path.maxpool(x_slow)
        if not self.slow_only:
            x_fast = x[:, :, ::self.tau // self.alpha, :, :]
            x_fast = self.fast_path.conv1(x_fast)
            x_fast = self.fast_path.bn1(x_fast)
            x_fast = self.fast_path.relu(x_fast)
            x_fast = self.fast_path.maxpool(x_fast)
        if not self.fast_only and not self.slow_only:
            x_fast_lateral = self.slow_path.conv1_lateral(x_fast)
            x_slow = torch.cat((x_slow, x_fast_lateral), dim=1)
        outs = []
        if not self.fast_only:
            for i, layer_name in enumerate(self.slow_path.res_layers):
                res_layer = getattr(self.slow_path, layer_name)
                x_slow = res_layer(x_slow)
                if not self.slow_only:
                    res_layer_fast = getattr(self.fast_path, layer_name)
                    x_fast = res_layer_fast(x_fast)
                    if self.lateral_type == 'conv' and i != 3:
                        lateral_name = self.slow_path.lateral_connections[i]
                        conv_lateral = getattr(self.slow_path, lateral_name)
                        x_fast_lateral = conv_lateral(x_fast)
                        x_slow = torch.cat((x_slow, x_fast_lateral), dim=1)
                if i in self.out_indices:
                    if not self.slow_only:
                        outs.append((x_slow, x_fast))
                    else:
                        outs.append(x_slow)
        else:
            for i, layer_name in enumerate(self.fast_path.res_layers):
                res_layer = getattr(self.fast_path, layer_name)
                x_fast = res_layer(x_fast)
                if i in self.out_indices:
                    outs.append(x_fast)
        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)

    def train(self, mode=True):
        super(ResNet_I3D_SlowFast, self).train(mode)
        if self.bn_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm3d):
                    m.eval()
                    if self.bn_frozen:
                        for params in m.parameters():
                            params.requires_grad = False
        if self.partial_bn:
            for i in range(1, self.frozen_stages + 1):
                mod = getattr(self, 'layer{}'.format(i))
                for m in mod.modules():
                    if isinstance(m, nn.BatchNorm3d):
                        m.eval()
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False
        if mode and self.frozen_stages >= 0:
            for param in self.conv1.parameters():
                param.requires_grad = False
            for param in self.bn1.parameters():
                param.requires_grad = False
            self.bn1.eval()
            self.bn1.weight.requires_grad = False
            self.bn1.bias.requires_grad = False
            for i in range(1, self.frozen_stages + 1):
                mod = getattr(self, 'layer{}'.format(i))
                mod.eval()
                for param in mod.parameters():
                    param.requires_grad = False


BLOCK_CONFIG = {(10): (1, 1, 1, 1), (16): (2, 2, 2, 1), (18): (2, 2, 2, 2), (26): (2, 2, 2, 2), (34): (3, 4, 6, 3), (50): (3, 4, 6, 3), (101): (3, 4, 23, 3), (152): (3, 8, 36, 3)}


DEEP_FILTER_CONFIG = [[256, 64], [512, 128], [1024, 256], [2048, 512]]


SHALLOW_FILTER_CONFIG = [[64, 64], [128, 128], [256, 256], [512, 512]]


def conv3d_wbias(in_planes, out_planes, kernel, stride, pad, groups=1):
    assert len(kernel) == 3
    assert len(stride) == 3
    assert len(pad) == 3
    return nn.Conv3d(in_planes, out_planes, kernel_size=kernel, stride=stride, padding=pad, groups=groups, bias=True)


def conv3d_wobias(in_planes, out_planes, kernel, stride, pad, groups=1):
    assert len(kernel) == 3
    assert len(stride) == 3
    assert len(pad) == 3
    return nn.Conv3d(in_planes, out_planes, kernel_size=kernel, stride=stride, padding=pad, groups=groups, bias=False)


class module_list(nn.Module):

    def __init__(self, modules, names=None):
        super(module_list, self).__init__()
        self.num = len(modules)
        self.modules = modules
        if names is None:
            alphabet = string.ascii_lowercase
            alphabet = list(alphabet)
            if self.num < 26:
                self.names = alphabet[:self.num]
            else:
                alphabet2 = itertools.product(alphabet, alphabet)
                alphabet2 = list(map(lambda x: x[0] + x[1], alphabet2))
                self.names = alphabet2[:self.num]
        else:
            assert len(names) == self.num
            self.names = names
        for m, n in zip(self.modules, self.names):
            setattr(self, n, m)

    def forward(self, inp):
        for n in self.names:
            inp = getattr(self, n)(inp)
        return inp


def make_plain_res_layer(block, num_blocks, in_filters, num_filters, base_filters, block_type='3d', down_sampling=False, down_sampling_temporal=None, is_real_3d=True, with_bn=True):
    layers = []
    layers.append(block(in_filters, num_filters, base_filters, down_sampling=down_sampling, down_sampling_temporal=down_sampling_temporal, block_type=block_type, is_real_3d=is_real_3d, with_bn=with_bn))
    for i in range(num_blocks - 1):
        layers.append(block(num_filters, num_filters, base_filters, block_type=block_type, is_real_3d=is_real_3d, with_bn=with_bn))
    return module_list(layers)


class ResNet_R3D(nn.Module):

    def __init__(self, pretrained=None, num_input_channels=3, depth=34, block_type='2.5d', channel_multiplier=1.0, bottleneck_multiplier=1.0, conv1_kernel_t=3, conv1_stride_t=1, use_pool1=False, bn_eval=True, bn_frozen=True, with_bn=True):
        super(ResNet_R3D, self).__init__()
        self.pretrained = pretrained
        self.num_input_channels = num_input_channels
        self.depth = depth
        self.block_type = block_type
        self.channel_multiplier = channel_multiplier
        self.bottleneck_multiplier = bottleneck_multiplier
        self.conv1_kernel_t = conv1_kernel_t
        self.conv1_stride_t = conv1_stride_t
        self.use_pool1 = use_pool1
        self.relu = nn.ReLU()
        self.bn_eval = bn_eval
        self.bn_frozen = bn_frozen
        self.with_bn = with_bn
        global comp_count, comp_idx
        comp_idx = 0
        comp_count = 0
        if self.with_bn:
            conv3d = conv3d_wobias
        else:
            conv3d = conv3d_wbias
        if self.block_type in ['2.5d', '2.5d-sep']:
            self.conv1_s = conv3d(self.num_input_channels, 45, [1, 7, 7], [1, 2, 2], [0, 3, 3])
            if self.with_bn:
                self.bn1_s = nn.BatchNorm3d(45, eps=0.001)
            self.conv1_t = conv3d(45, 64, [self.conv1_kernel_t, 1, 1], [self.conv1_stride_t, 1, 1], [(self.conv1_kernel_t - 1) // 2, 0, 0])
            if self.with_bn:
                self.bn1_t = nn.BatchNorm3d(64, eps=0.001)
        else:
            self.conv1 = conv3d(self.num_input_channels, 64, [self.conv1_kernel_t, 7, 7], [self.conv1_stride_t, 2, 2], [(self.conv1_kernel_t - 1) // 2, 3, 3])
            if self.with_bn:
                self.bn1 = nn.BatchNorm3d(64, eps=0.001)
        if self.use_pool1:
            self.pool1 = nn.MaxPool3d(kernel_size=[1, 3, 3], stride=[1, 2, 2], padding=[0, 1, 1])
        self.stage_blocks = BLOCK_CONFIG[self.depth]
        if self.depth <= 18 or self.depth == 34:
            self.block = BasicBlock
        else:
            self.block = Bottleneck
        if self.depth <= 34:
            self.filter_config = SHALLOW_FILTER_CONFIG
        else:
            self.filter_config = DEEP_FILTER_CONFIG
        self.filter_config = np.multiply(self.filter_config, self.channel_multiplier).astype(np.int)
        layer1 = make_plain_res_layer(self.block, self.stage_blocks[0], 64, self.filter_config[0][0], int(self.filter_config[0][1] * self.bottleneck_multiplier), block_type=self.block_type, with_bn=self.with_bn)
        self.add_module('layer1', layer1)
        layer2 = make_plain_res_layer(self.block, self.stage_blocks[1], self.filter_config[0][0], self.filter_config[1][0], int(self.filter_config[1][1] * self.bottleneck_multiplier), block_type=self.block_type, down_sampling=True, with_bn=self.with_bn)
        self.add_module('layer2', layer2)
        layer3 = make_plain_res_layer(self.block, self.stage_blocks[2], self.filter_config[1][0], self.filter_config[2][0], int(self.filter_config[2][1] * self.bottleneck_multiplier), block_type=self.block_type, down_sampling=True, with_bn=self.with_bn)
        self.add_module('layer3', layer3)
        layer4 = make_plain_res_layer(self.block, self.stage_blocks[3], self.filter_config[2][0], self.filter_config[3][0], int(self.filter_config[3][1] * self.bottleneck_multiplier), block_type=self.block_type, down_sampling=True, with_bn=self.with_bn)
        self.add_module('layer4', layer4)
        self.res_layers = ['layer1', 'layer2', 'layer3', 'layer4']

    def forward(self, x):
        if self.block_type in ['2.5d', '2.5d-sep']:
            if self.with_bn:
                x = self.relu(self.bn1_s(self.conv1_s(x)))
                x = self.relu(self.bn1_t(self.conv1_t(x)))
            else:
                x = self.relu(self.conv1_s(x))
                x = self.relu(self.conv1_t(x))
        elif self.with_bn:
            x = self.relu(self.bn1(self.conv1(x)))
        else:
            x = self.relu(self.conv1(x))
        if self.use_pool1:
            x = self.pool1(x)
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
        return x

    def init_weights(self):
        if isinstance(self.pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, self.pretrained, strict=False, logger=logger)
        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm3d):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def train(self, mode=True):
        super(ResNet_R3D, self).train(mode)
        if self.bn_eval and self.with_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm3d):
                    m.eval()
                    if self.bn_frozen:
                        for params in m.parameters():
                            params.requires_grad = False


class ResNet_S3D(nn.Module):
    """ResNet_S3D backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        num_stages (int): Resnet stages, normally 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means
            not freezing any parameters.
        bn_eval (bool): Whether to set BN layers to eval mode, namely, freeze
            running stats (mean and var).
        bn_frozen (bool): Whether to freeze weight and bias of BN layers.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
    """
    arch_settings = {(18): (BasicBlock, (2, 2, 2, 2)), (34): (BasicBlock, (3, 4, 6, 3)), (50): (Bottleneck, (3, 4, 6, 3)), (101): (Bottleneck, (3, 4, 23, 3)), (152): (Bottleneck, (3, 8, 36, 3))}

    def __init__(self, depth, pretrained=None, num_stages=4, spatial_strides=(1, 2, 2, 2), temporal_strides=(1, 1, 1, 1), dilations=(1, 1, 1, 1), out_indices=(0, 1, 2, 3), conv1_kernel_t=5, conv1_stride_t=2, pool1_kernel_t=1, pool1_stride_t=2, use_pool2=True, style='pytorch', frozen_stages=-1, inflate_freq=(1, 1, 1, 1), bn_eval=True, bn_frozen=False, partial_bn=False, with_cp=False, with_trajectory=False, trajectory_source_indices=-1, trajectory_downsample_method='ave', conv_bias=0.2):
        super(ResNet_S3D, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError('invalid depth {} for resnet'.format(depth))
        self.depth = depth
        self.pretrained = pretrained
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.spatial_strides = spatial_strides
        self.temporal_strides = temporal_strides
        self.dilations = dilations
        assert len(spatial_strides) == len(temporal_strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.frozen_stages = frozen_stages
        self.inflate_freqs = inflate_freq if not isinstance(inflate_freq, int) else (inflate_freq,) * num_stages
        self.bn_eval = bn_eval
        self.bn_frozen = bn_frozen
        self.partial_bn = partial_bn
        self.with_cp = with_cp
        self.with_trajectory = with_trajectory
        self.trajectory_source_indices = trajectory_source_indices if not isinstance(trajectory_source_indices, int) else [trajectory_source_indices] * num_stages
        self.trajectory_downsample_method = trajectory_downsample_method
        self.conv_bias = conv_bias
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        for stage in range(num_stages):
            self.trajectory_source_indices[stage] = self.trajectory_source_indices[stage] if not isinstance(self.trajectory_source_indices[stage], int) else (self.trajectory_source_indices[stage],) * self.stage_blocks[stage]
        self.inplanes = 64
        if conv1_kernel_t > 1:
            self.conv1 = nn.Conv3d(3, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
            self.conv1_t = nn.Conv3d(64, 64, kernel_size=(conv1_kernel_t, 1, 1), stride=(conv1_stride_t, 1, 1), padding=((conv1_kernel_t - 1) // 2, 1, 1), bias=True)
            self.bn1_t = nn.BatchNorm3d(64)
        else:
            self.conv1 = nn.Conv3d(3, 64, kernel_size=(1, 7, 7), stride=(conv1_stride_t, 2, 2), padding=(0, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(pool1_kernel_t, 3, 3), stride=(pool1_stride_t, 2, 2), padding=(pool1_kernel_t // 2, 1, 1))
        self.use_pool2 = use_pool2
        if self.use_pool2:
            self.pool2 = nn.MaxPool3d(kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))
        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            traj_src_indices = self.trajectory_source_indices[i] if not isinstance(self.trajectory_source_indices[i], int) else (self.trajectory_source_indices[i],) * num_blocks
            spatial_stride = spatial_strides[i]
            temporal_stride = temporal_strides[i]
            dilation = dilations[i]
            planes = 64 * 2 ** i
            res_layer = make_res_layer(self.block, self.inplanes, planes, num_blocks, spatial_stride=spatial_stride, temporal_stride=temporal_stride, dilation=dilation, style=self.style, inflate_freq=self.inflate_freqs[i], with_cp=with_cp, traj_src_indices=traj_src_indices)
            self.inplanes = planes * self.block.expansion
            layer_name = 'layer{}'.format(i + 1)
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)
        self.feat_dim = self.block.expansion * 64 * 2 ** (len(self.stage_blocks) - 1)

    def init_weights(self):
        if isinstance(self.pretrained, str):
            logger = logging.getLogger()
            resnet2d = ResNet(self.depth)
            load_checkpoint(resnet2d, self.pretrained, strict=False, logger=logger)
            for name, module in self.named_modules():
                if isinstance(module, nn.Conv3d) or isinstance(module, TrajConv):
                    if rhasattr(resnet2d, name):
                        new_weight = rgetattr(resnet2d, name).weight.data.unsqueeze(2).expand_as(module.weight) / module.weight.data.shape[2]
                        module.weight.data.copy_(new_weight)
                        if hasattr(module, 'bias') and module.bias is not None:
                            new_bias = rgetattr(resnet2d, name).bias.data
                            module.bias.data.copy_(new_bias)
                    else:
                        kaiming_init(module, bias=self.conv_bias)
                elif isinstance(module, nn.BatchNorm3d):
                    if rhasattr(resnet2d, name):
                        for attr in ['weight', 'bias', 'running_mean', 'running_var']:
                            setattr(module, attr, getattr(rgetattr(resnet2d, name), attr))
                    else:
                        constant_init(module, 1)
        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    kaiming_init(m, bias=self.conv_bias)
                elif isinstance(m, nn.BatchNorm3d):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x, trajectory_forward=None, trajectory_backward=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            y = []
            for j in self.trajectory_source_indices[i]:
                if j > -1:
                    flow_forward = trajectory_forward[j]
                    flow_backward = trajectory_backward[j]
                    flow_forward = flow_forward.view((flow_forward.size(0), -1, 2, flow_forward.size(2), flow_forward.size(3)))
                    flow_backward = flow_backward.view((flow_backward.size(0), -1, 2, flow_backward.size(2), flow_backward.size(3)))
                    flow_forward_x, flow_forward_y = torch.split(flow_forward, 1, 2)
                    flow_backward_x, flow_backward_y = torch.split(flow_backward, 1, 2)
                    flow_backward_x = flow_backward_x.flip(1).view((flow_backward_x.size(0), 1, flow_backward_x.size(1), flow_backward_x.size(3), flow_backward_x.size(4)))
                    flow_backward_y = flow_backward_y.flip(1).view((flow_backward_y.size(0), 1, flow_backward_y.size(1), flow_backward_y.size(3), flow_backward_y.size(4)))
                    flow_forward_x = flow_forward_x.view((flow_forward_x.size(0), 1, flow_forward_x.size(1), flow_forward_x.size(3), flow_forward_x.size(4)))
                    flow_forward_y = flow_forward_y.view((flow_forward_y.size(0), 1, flow_forward_y.size(1), flow_forward_y.size(3), flow_forward_y.size(4)))
                    flow_zero = torch.zeros_like(flow_forward_x)
                    y.append(torch.cat((flow_backward_y, flow_backward_x, flow_zero, flow_zero, flow_forward_y, flow_forward_x), 1))
                else:
                    y.append(None)
            x, remains = res_layer((x, y))
            assert len(remains) == 0
            if i in self.out_indices:
                outs.append(x)
            if self.use_pool2 and i == 0:
                x = self.pool2(x)
        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)

    def train(self, mode=True):
        super(ResNet_S3D, self).train(mode)
        if self.bn_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm3d):
                    m.eval()
                    if self.bn_frozen:
                        for params in m.parameters():
                            params.requires_grad = False
        if self.partial_bn:
            for i in range(1, self.frozen_stages + 1):
                mod = getattr(self, 'layer{}'.format(i))
                for m in mod.modules():
                    if isinstance(m, nn.BatchNorm3d):
                        m.eval()
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False
        if mode and self.frozen_stages >= 0:
            for param in self.conv1.parameters():
                param.requires_grad = False
            for param in self.bn1.parameters():
                param.requires_grad = False
            self.bn1.eval()
            self.bn1.weight.requires_grad = False
            self.bn1.bias.requires_grad = False
            for i in range(1, self.frozen_stages + 1):
                mod = getattr(self, 'layer{}'.format(i))
                mod.eval()
                for param in mod.parameters():
                    param.requires_grad = False


def accuracy(pred, target, topk=1):
    if isinstance(topk, int):
        topk = topk,
        return_single = True
    else:
        return_single = False
    maxk = max(topk)
    _, pred_label = pred.topk(maxk, 1, True, True)
    pred_label = pred_label.t()
    correct = pred_label.eq(target.view(1, -1).expand_as(pred_label))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / pred.size(0)))
    return res[0] if return_single else res


def bbox_target_single(pos_bboxes, neg_bboxes, pos_gt_bboxes, pos_gt_labels, cfg, reg_classes=1, target_means=[0.0, 0.0, 0.0, 0.0], target_stds=[1.0, 1.0, 1.0, 1.0]):
    num_pos = pos_bboxes.size(0)
    num_neg = neg_bboxes.size(0)
    num_samples = num_pos + num_neg
    if len(pos_gt_labels[0]) == 1:
        labels = pos_bboxes.new_zeros(num_samples, dtype=torch.long)
    else:
        labels = pos_bboxes.new_zeros((num_samples, len(pos_gt_labels[0])), dtype=torch.long)
    label_weights = pos_bboxes.new_zeros(num_samples)
    if len(pos_gt_labels[0]) == 1:
        class_weights = pos_bboxes.new_zeros(num_samples)
    else:
        class_weights = pos_bboxes.new_zeros(num_samples, len(pos_gt_labels[0]))
    bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
    bbox_weights = pos_bboxes.new_zeros(num_samples, 4)
    if num_pos > 0:
        labels[:num_pos] = pos_gt_labels
        pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
        label_weights[:num_pos] = pos_weight
        class_weight = 1.0 if not hasattr(cfg, 'cls_weight') or cfg.cls_weight <= 0 else cfg.cls_weight
        class_weights[:num_pos] = class_weight
        pos_bbox_targets = bbox2delta(pos_bboxes, pos_gt_bboxes, target_means, target_stds)
        bbox_targets[:num_pos, :] = pos_bbox_targets
        bbox_weights[:num_pos, :] = 1
    if num_neg > 0:
        label_weights[-num_neg:] = 1.0
        class_weights[-num_neg:] = 0.0
    return labels, label_weights, bbox_targets, bbox_weights, class_weights


def bbox_target(pos_bboxes_list, neg_bboxes_list, pos_gt_bboxes_list, pos_gt_labels_list, cfg, reg_classes=1, target_means=[0.0, 0.0, 0.0, 0.0], target_stds=[1.0, 1.0, 1.0, 1.0], concat=True):
    labels, label_weights, bbox_targets, bbox_weights, class_weights = multi_apply(bbox_target_single, pos_bboxes_list, neg_bboxes_list, pos_gt_bboxes_list, pos_gt_labels_list, cfg=cfg, reg_classes=reg_classes, target_means=target_means, target_stds=target_stds)
    if concat:
        labels = torch.cat(labels, 0)
        label_weights = torch.cat(label_weights, 0)
        bbox_targets = torch.cat(bbox_targets, 0)
        bbox_weights = torch.cat(bbox_weights, 0)
        class_weights = torch.cat(class_weights, 0)
    return labels, label_weights, bbox_targets, bbox_weights, class_weights


def recall_prec(pred_vec, target_vec):
    """
    Args:
        pred_vec: <torch.tensor> (n, C+1), each element is either 0 or 1
        target_vec: <torch.tensor> (n, C+1), each element is either 0 or 1

    Returns:
        recall
        prec
    """
    recall = pred_vec.new_full((pred_vec.size(0),), 0).float()
    prec = pred_vec.new_full((pred_vec.size(0),), 0).float()
    num_pos = 0
    for i in range(target_vec.size(0)):
        if target_vec[(i), :].float().sum(0) == 0:
            continue
        correct_labels = pred_vec[(i), :] & target_vec[(i), :]
        recall[i] = correct_labels.float().sum(0, keepdim=True) / target_vec[(i), :].float().sum(0, keepdim=True)
        prec[i] = correct_labels.float().sum(0, keepdim=True) / (pred_vec[(i), :].float().sum(0, keepdim=True) + 1e-06)
        num_pos += 1
    recall = recall.float().sum(0, keepdim=True).mul_(100.0 / num_pos)
    prec = prec.float().sum(0, keepdim=True).mul_(100.0 / num_pos)
    return recall, prec


def multilabel_accuracy(pred, target, topk=1, thr=0.5):
    if topk is None:
        topk = ()
    elif isinstance(topk, int):
        topk = topk,
    pred = pred.sigmoid()
    pred_bin_labels = pred.new_full((pred.size(0),), 0, dtype=torch.long)
    pred_vec_labels = pred.new_full(pred.size(), 0, dtype=torch.long)
    for i in range(pred.size(0)):
        inds = torch.nonzero(pred[(i), 1:] > thr).squeeze() + 1
        if inds.numel() > 0:
            pred_vec_labels[i, inds] = 1
        if pred[i, 0] > thr:
            pred_bin_labels[i] = 1
    target_bin_labels = target.new_full((target.size(0),), 0, dtype=torch.long)
    target_vec_labels = target.new_full(target.size(), 0, dtype=torch.long)
    for i in range(target.size(0)):
        inds = torch.nonzero(target[(i), :] >= 1).squeeze()
        if inds.numel() > 0:
            target_vec_labels[i, target[i, inds]] = 1
            target_bin_labels[i] = 1
    correct = pred_bin_labels.eq(target_bin_labels)
    acc = correct.float().sum(0, keepdim=True).mul_(100.0 / correct.size(0))
    recall_thr, prec_thr = recall_prec(pred_vec_labels, target_vec_labels)
    recalls = []
    precs = []
    for k in topk:
        _, pred_label = pred.topk(k, 1, True, True)
        pred_vec_labels = pred.new_full(pred.size(), 0, dtype=torch.long)
        for i in range(pred.size(0)):
            pred_vec_labels[i, pred_label[i]] = 1
        recall_k, prec_k = recall_prec(pred_vec_labels, target_vec_labels)
        recalls.append(recall_k)
        precs.append(prec_k)
    return acc, recall_thr, prec_thr, recalls, precs


def singleclass_nms(multi_bboxes, multi_scores, score_thr, nms_cfg, max_num=-1):
    """NMS for single-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, 4)
        multi_scores (Tensor): shape (n, #class)
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.

    Returns:
        tuple: (bboxes, scores), tensors of shape (k, 5) and (k, #class).
            labels are 0-based.
    """
    bboxes, scores = [], []
    nms_cfg_ = nms_cfg.copy()
    nms_type = nms_cfg_.pop('type', 'nms')
    nms_op = getattr(nms_wrapper, nms_type)
    cls_inds = multi_scores[:, (0)] > score_thr
    if not cls_inds.any():
        bboxes = multi_bboxes.new_zeros((0, 5))
        scores = multi_bboxes.new_zeros((0, multi_scores.size(1)))
        return bboxes, scores
    _bboxes = multi_bboxes[(cls_inds), :]
    _scores = multi_scores[(cls_inds), :]
    cls_dets = torch.cat([_bboxes, _scores[:, 0:1]], dim=1)
    cls_dets, nms_keep = nms_op(cls_dets, **nms_cfg_)
    cls_scores = _scores[(nms_keep), :]
    bboxes.append(cls_dets)
    scores.append(cls_scores)
    if bboxes:
        bboxes = torch.cat(bboxes)
        scores = torch.cat(scores)
        if bboxes.shape[0] > max_num:
            _, inds = bboxes[:, (-1)].sort(descending=True)
            inds = inds[:max_num]
            bboxes = bboxes[inds]
            scores = scores[inds]
    else:
        bboxes = multi_bboxes.new_zeros((0, 5))
        scores = multi_bboxes.new_zeros((0, multi_scores.size(1)))
    return bboxes, scores


def _expand_multilabel_binary_labels(labels, label_weights, label_channels):
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = torch.nonzero(labels >= 1)
    if inds.numel() > 0:
        for ind in inds:
            bin_labels[ind[0], labels[ind[0], ind[1]] - 1] = 1
    bin_label_weights = label_weights
    return bin_labels, bin_label_weights


def weighted_multilabel_binary_cross_entropy(pred, label, weight, avg_factor=None):
    label, weight = _expand_multilabel_binary_labels(label, weight, pred.size(-1))
    if avg_factor is None:
        avg_factor = max(torch.sum(weight > 0).float().item(), 1.0)
    return F.binary_cross_entropy_with_logits(pred, label.float(), weight.float(), reduction='sum')[None] / avg_factor


class BBoxHead(nn.Module):
    """Simplest RoI head, with only two fc layers for classification and
    regression respectively"""

    def __init__(self, with_temporal_pool=False, with_spatial_pool=False, temporal_pool_type='avg', spatial_pool_type='max', with_cls=True, with_reg=True, roi_feat_size=7, in_channels=256, num_classes=81, target_means=[0.0, 0.0, 0.0, 0.0], target_stds=[0.1, 0.1, 0.2, 0.2], multilabel_classification=True, reg_class_agnostic=True, nms_class_agnostic=True):
        super(BBoxHead, self).__init__()
        assert with_cls or with_reg
        self.with_temporal_pool = with_temporal_pool
        self.with_spatial_pool = with_spatial_pool
        assert temporal_pool_type in ['max', 'avg']
        assert spatial_pool_type in ['max', 'avg']
        self.temporal_pool_type = temporal_pool_type
        self.spatial_pool_type = spatial_pool_type
        self.with_cls = with_cls
        self.with_reg = with_reg
        self.roi_feat_size = roi_feat_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.target_means = target_means
        self.target_stds = target_stds
        self.multilabel_classification = multilabel_classification
        self.reg_class_agnostic = reg_class_agnostic
        self.nms_class_agnostic = nms_class_agnostic
        in_channels = self.in_channels
        if self.with_temporal_pool:
            if self.temporal_pool_type == 'avg':
                self.temporal_pool = nn.AvgPool3d((roi_feat_size[0], 1, 1))
            else:
                self.temporal_pool = nn.MaxPool3d((roi_feat_size[0], 1, 1))
        if self.with_spatial_pool:
            if self.spatial_pool_type == 'avg':
                self.spatial_pool = nn.AvgPool3d((1, roi_feat_size[1], roi_feat_size[2]))
            else:
                self.spatial_pool = nn.MaxPool3d((1, roi_feat_size[1], roi_feat_size[2]))
        if not self.with_temporal_pool and not self.with_spatial_pool:
            in_channels *= self.roi_feat_size * self.roi_feat_size
        if self.with_cls:
            self.fc_cls = nn.Linear(in_channels, num_classes)
        if self.with_reg:
            out_dim_reg = 4 if reg_class_agnostic else 4 * num_classes
            self.fc_reg = nn.Linear(in_channels, out_dim_reg)
        self.debug_imgs = None

    def init_weights(self):
        if self.with_cls:
            nn.init.normal_(self.fc_cls.weight, 0, 0.01)
            nn.init.constant_(self.fc_cls.bias, 0)
        if self.with_reg:
            nn.init.normal_(self.fc_reg.weight, 0, 0.001)
            nn.init.constant_(self.fc_reg.bias, 0)

    def forward(self, x):
        if self.with_temporal_pool:
            x = self.temporal_pool(x)
        if self.with_spatial_pool:
            x = self.spatial_pool(x)
        x = x.view(x.size(0), -1)
        cls_score = self.fc_cls(x) if self.with_cls else None
        bbox_pred = self.fc_reg(x) if self.with_reg else None
        return cls_score, bbox_pred

    def get_target(self, sampling_results, gt_bboxes, gt_labels, rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        neg_proposals = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels = [res.pos_gt_labels for res in sampling_results]
        reg_classes = 1 if self.reg_class_agnostic else self.num_classes
        cls_reg_targets = bbox_target(pos_proposals, neg_proposals, pos_gt_bboxes, pos_gt_labels, rcnn_train_cfg, reg_classes, target_means=self.target_means, target_stds=self.target_stds)
        return cls_reg_targets

    def loss(self, cls_score, bbox_pred, labels, label_weights, bbox_targets, bbox_weights, class_weights, reduce=True):
        losses = dict()
        if cls_score is not None:
            if not self.multilabel_classification:
                assert len(labels[0]) == 1
                losses['loss_cls'] = weighted_cross_entropy(cls_score, labels, label_weights, reduce=reduce)
                losses['acc'] = accuracy(cls_score, labels)
            else:
                losses['loss_person_cls'] = weighted_binary_cross_entropy(cls_score[:, (0)], labels[:, (0)] >= 1, label_weights)
                pos_inds = torch.nonzero(labels[:, (0)] > 0).squeeze(1)
                losses['loss_action_cls'] = weighted_multilabel_binary_cross_entropy(cls_score[(pos_inds), 1:], labels[(pos_inds), :], class_weights[(pos_inds), 1:])
                acc, recall_thr, prec_thr, recall_k, prec_k = multilabel_accuracy(cls_score, labels, topk=(3, 5), thr=0.5)
                losses['acc'] = acc
                losses['recall@thr=0.5'] = recall_thr
                losses['prec@thr=0.5'] = prec_thr
                losses['recall@top3'] = recall_k[0]
                losses['prec@top3'] = prec_k[0]
                losses['recall@top5'] = recall_k[1]
                losses['prec@top5'] = prec_k[1]
        if bbox_pred is not None:
            pos_inds = labels > 0
            if self.reg_class_agnostic:
                pos_inds = labels[:, (0)] > 0
                pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), 4)[pos_inds]
            else:
                pos_inds = labels > 0
                pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), -1, 4)[pos_inds, labels[pos_inds]]
            losses['loss_reg'] = weighted_smoothl1(pos_bbox_pred, bbox_targets[pos_inds], bbox_weights[pos_inds], avg_factor=bbox_targets.size(0))
        return losses

    def get_det_bboxes(self, rois, cls_score, bbox_pred, img_shape, scale_factor, rescale=False, cfg=None, crop_quadruple=None):
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        if not self.multilabel_classification:
            scores = F.softmax(cls_score, dim=1) if cls_score is not None else None
        else:
            scores = cls_score.sigmoid() if cls_score is not None else None
        if bbox_pred is not None:
            bboxes = delta2bbox(rois[:, 1:], bbox_pred, self.target_means, self.target_stds, img_shape)
        else:
            bboxes = rois[:, 1:]

        def _bbox_crop_undo(bboxes, crop_quadruple):
            assert bboxes.shape[-1] % 4 == 0
            assert crop_quadruple is not None
            decropped = bboxes.clone()
            x1, y1, tw, th = crop_quadruple
            decropped[(...), 0::2] = bboxes[(...), 0::2] + x1
            decropped[(...), 1::2] = bboxes[(...), 1::2] + y1
            return decropped
        if crop_quadruple is not None:
            bboxes = _bbox_crop_undo(bboxes, crop_quadruple)
        if rescale:
            bboxes /= scale_factor
        if cfg is None:
            return bboxes, scores
        else:
            if self.nms_class_agnostic:
                det_bboxes, det_labels = singleclass_nms(bboxes, scores, cfg.score_thr, cfg.nms, cfg.max_per_img)
            else:
                det_bboxes, det_labels = multiclass_nms(bboxes, scores, cfg.score_thr, cfg.nms, cfg.max_per_img)
            return det_bboxes, det_labels

    def refine_bboxes(self, rois, labels, bbox_preds, pos_is_gts, img_metas):
        """Refine bboxes during training.
        Args:
            rois (Tensor): Shape (n*bs, 5), where n is image number per GPU,
                and bs is the sampled RoIs per image.
            labels (Tensor): Shape (n*bs, ).
            bbox_preds (Tensor): Shape (n*bs, 4) or (n*bs, 4*#class).
            pos_is_gts (list[Tensor]): Flags indicating if each positive bbox
                is a gt bbox.
            img_metas (list[dict]): Meta info of each image.
        Returns:
            list[Tensor]: Refined bboxes of each image in a mini-batch.
        """
        img_ids = rois[:, (0)].long().unique(sorted=True)
        assert img_ids.numel() == len(img_metas)
        bboxes_list = []
        for i in range(len(img_metas)):
            inds = torch.nonzero(rois[:, (0)] == i).squeeze()
            num_rois = inds.numel()
            bboxes_ = rois[(inds), 1:]
            label_ = labels[inds]
            bbox_pred_ = bbox_preds[inds]
            img_meta_ = img_metas[i]
            pos_is_gts_ = pos_is_gts[i]
            bboxes = self.regress_by_class(bboxes_, label_, bbox_pred_, img_meta_)
            pos_keep = 1 - pos_is_gts_
            keep_inds = pos_is_gts_.new_ones(num_rois)
            keep_inds[:len(pos_is_gts_)] = pos_keep
            bboxes_list.append(bboxes[keep_inds])
        return bboxes_list

    def regress_by_class(self, rois, label, bbox_pred, img_meta):
        """Regress the bbox for the predicted class. Used in Cascade R-CNN.
        Args:
            rois (Tensor): shape (n, 4) or (n, 5)
            label (Tensor): shape (n, )
            bbox_pred (Tensor): shape (n, 4*(#class+1)) or (n, 4)
            img_meta (dict): Image meta info.
        Returns:
            Tensor: Regressed bboxes, the same shape as input rois.
        """
        assert rois.size(1) == 4 or rois.size(1) == 5
        if not self.reg_class_agnostic:
            label = label * 4
            inds = torch.stack((label, label + 1, label + 2, label + 3), 1)
            bbox_pred = torch.gather(bbox_pred, 1, inds)
        assert bbox_pred.size(1) == 4
        if rois.size(1) == 4:
            new_rois = delta2bbox(rois, bbox_pred, self.target_means, self.target_stds, img_meta['img_shape'])
        else:
            bboxes = delta2bbox(rois[:, 1:], bbox_pred, self.target_means, self.target_stds, img_meta['img_shape'])
            new_rois = torch.cat((rois[:, ([0])], bboxes), dim=1)
        return new_rois


class ClsHead(nn.Module):
    """Simplest classification head"""

    def __init__(self, with_avg_pool=True, temporal_feature_size=1, spatial_feature_size=7, dropout_ratio=0.8, in_channels=2048, num_classes=101, init_std=0.01, fcn_testing=False):
        super(ClsHead, self).__init__()
        self.with_avg_pool = with_avg_pool
        self.dropout_ratio = dropout_ratio
        self.in_channels = in_channels
        self.dropout_ratio = dropout_ratio
        self.temporal_feature_size = temporal_feature_size
        self.spatial_feature_size = spatial_feature_size
        self.init_std = init_std
        self.fcn_testing = fcn_testing
        self.num_classes = num_classes
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool3d((temporal_feature_size, spatial_feature_size, spatial_feature_size))
        self.fc_cls = nn.Linear(in_channels, num_classes)
        self.new_cls = None

    def init_weights(self):
        nn.init.normal_(self.fc_cls.weight, 0, self.init_std)
        nn.init.constant_(self.fc_cls.bias, 0)

    def forward(self, x):
        if not self.fcn_testing:
            if x.ndimension() == 4:
                x = x.unsqueeze(2)
            assert x.shape[1] == self.in_channels
            assert x.shape[2] == self.temporal_feature_size
            assert x.shape[3] == self.spatial_feature_size
            assert x.shape[4] == self.spatial_feature_size
            if self.with_avg_pool:
                x = self.avg_pool(x)
            if self.dropout is not None:
                x = self.dropout(x)
            x = x.view(x.size(0), -1)
            cls_score = self.fc_cls(x)
            return cls_score
        else:
            if x.ndimension() == 4:
                x = x.unsqueeze(2)
            if self.with_avg_pool:
                x = self.avg_pool(x)
            if self.new_cls is None:
                self.new_cls = nn.Conv3d(self.in_channels, self.num_classes, 1, 1, 0)
                self.new_cls.load_state_dict({'weight': self.fc_cls.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), 'bias': self.fc_cls.bias})
            class_map = self.new_cls(x)
            return class_map

    def loss(self, cls_score, labels):
        losses = dict()
        losses['loss_cls'] = F.cross_entropy(cls_score, labels)
        return losses


def classwise_regression_loss(pred, labels, targets):
    indexer = labels.data - 1
    prep = pred[:, (indexer), :]
    class_pred = torch.cat((torch.diag(prep[:, :, (0)]).view(-1, 1), torch.diag(prep[:, :, (1)]).view(-1, 1)), dim=1)
    loss = F.smooth_l1_loss(class_pred.view(-1), targets.view(-1)) * 2
    return loss


class OHEMHingeLoss(torch.autograd.Function):
    """
    This class is the core implementation for the completeness loss in paper.
    It compute class-wise hinge loss and performs online hard negative mining
    (OHEM).
    """

    @staticmethod
    def forward(ctx, pred, labels, is_positive, ohem_ratio, group_size):
        n_sample = pred.size()[0]
        assert n_sample == len(labels), 'mismatch between sample size and label size'
        losses = torch.zeros(n_sample)
        slopes = torch.zeros(n_sample)
        for i in range(n_sample):
            losses[i] = max(0, 1 - is_positive * pred[i, labels[i] - 1])
            slopes[i] = -is_positive if losses[i] != 0 else 0
        losses = losses.view(-1, group_size).contiguous()
        sorted_losses, indices = torch.sort(losses, dim=1, descending=True)
        keep_num = int(group_size * ohem_ratio)
        loss = torch.zeros(1)
        for i in range(losses.size(0)):
            loss += sorted_losses[(i), :keep_num].sum()
        ctx.loss_ind = indices[:, :keep_num]
        ctx.labels = labels
        ctx.slopes = slopes
        ctx.shape = pred.size()
        ctx.group_size = group_size
        ctx.num_group = losses.size(0)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        labels = ctx.labels
        slopes = ctx.slopes
        grad_in = torch.zeros(ctx.shape)
        for group in range(ctx.num_group):
            for idx in ctx.loss_ind[group]:
                loc = idx + group * ctx.group_size
                grad_in[loc, labels[loc] - 1] = slopes[loc] * grad_output.data[0]
        return torch.autograd.Variable(grad_in), None, None, None, None


def completeness_loss(pred, labels, sample_split, sample_group_size, ohem_ratio=0.17):
    pred_dim = pred.size()[1]
    pred = pred.view(-1, sample_group_size, pred_dim)
    labels = labels.view(-1, sample_group_size)
    pos_group_size = sample_split
    neg_group_size = sample_group_size - sample_split
    pos_prob = pred[:, :sample_split, :].contiguous().view(-1, pred_dim)
    neg_prob = pred[:, sample_split:, :].contiguous().view(-1, pred_dim)
    pos_ls = OHEMHingeLoss.apply(pos_prob, labels[:, :sample_split].contiguous().view(-1), 1, 1.0, pos_group_size)
    neg_ls = OHEMHingeLoss.apply(neg_prob, labels[:, sample_split:].contiguous().view(-1), -1, ohem_ratio, neg_group_size)
    pos_cnt = pos_prob.size(0)
    neg_cnt = int(neg_prob.size()[0] * ohem_ratio)
    return pos_ls / float(pos_cnt + neg_cnt) + neg_ls / float(pos_cnt + neg_cnt)


class SSNHead(nn.Module):
    """SSN's classification head"""

    def __init__(self, dropout_ratio=0.8, in_channels_activity=3072, in_channels_complete=3072, num_classes=20, with_bg=False, with_reg=True, init_std=0.001):
        super(SSNHead, self).__init__()
        self.dropout_ratio = dropout_ratio
        self.in_channels_activity = in_channels_activity
        self.in_channels_complete = in_channels_complete
        self.num_classes = num_classes - 1 if with_bg else num_classes
        self.with_reg = with_reg
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.activity_fc = nn.Linear(in_channels_activity, num_classes + 1)
        self.completeness_fc = nn.Linear(in_channels_complete, num_classes)
        if self.with_reg:
            self.regressor_fc = nn.Linear(in_channels_complete, num_classes * 2)

    def init_weights(self):
        nn.init.normal_(self.activity_fc.weight, 0, self.init_std)
        nn.init.constant_(self.activity_fc.bias, 0)
        nn.init.normal_(self.completeness_fc.weight, 0, self.init_std)
        nn.init.constant_(self.completeness_fc.bias, 0)
        if self.with_reg:
            nn.init.normal_(self.regressor_fc.weight, 0, self.init_std)
            nn.init.constant_(self.regressor_fc.bias, 0)

    def prepare_test_fc(self, stpp_feat_multiplier):
        self.test_fc = nn.Linear(self.activity_fc.in_features, self.activity_fc.out_features + self.completeness_fc.out_features * stpp_feat_multiplier + (self.regressor_fc.out_features * stpp_feat_multiplier if self.with_reg else 0))
        reorg_comp_weight = self.completeness_fc.weight.data.view(self.completeness_fc.out_features, stpp_feat_multiplier, self.activity_fc.in_features).transpose(0, 1).contiguous().view(-1, self.activity_fc.in_features)
        reorg_comp_bias = self.completeness_fc.bias.data.view(1, -1).expand(stpp_feat_multiplier, self.completeness_fc.out_features).contiguous().view(-1) / stpp_feat_multiplier
        weight = torch.cat((self.activity_fc.weight.data, reorg_comp_weight))
        bias = torch.cat((self.activity_fc.bias.data, reorg_comp_bias))
        if self.with_reg:
            reorg_reg_weight = self.regressor_fc.weight.data.view(self.regressor_fc.out_features, stpp_feat_multiplier, self.activity_fc.in_features).transpose(0, 1).contiguous().view(-1, self.activity_fc.in_features)
            reorg_reg_bias = self.regressor_fc.bias.data.view(1, -1).expand(stpp_feat_multiplier, self.regressor_fc.out_features).contiguous().view(-1) / stpp_feat_multiplier
            weight = torch.cat((weight, reorg_reg_weight))
            bias = torch.cat((bias, reorg_reg_bias))
        self.test_fc.weight.data = weight
        self.test_fc.bias.data = bias
        return True

    def forward(self, input, test_mode=False):
        if not test_mode:
            activity_feat, completeness_feat = input
            if self.dropout is not None:
                activity_feat = self.dropout(activity_feat)
                completeness_feat = self.dropout(completeness_feat)
            act_score = self.activity_fc(activity_feat)
            comp_score = self.completeness_fc(completeness_feat)
            bbox_pred = self.regressor_fc(completeness_feat) if self.with_reg else None
            return act_score, comp_score, bbox_pred
        else:
            test_score = self.test_fc(input)
            return test_score

    def loss(self, act_score, comp_score, bbox_pred, prop_type, labels, bbox_targets, train_cfg):
        losses = dict()
        prop_type = prop_type.view(-1)
        labels = labels.view(-1)
        act_indexer = ((prop_type == 0) + (prop_type == 2)).nonzero().squeeze()
        comp_indexer = ((prop_type == 0) + (prop_type == 1)).nonzero().squeeze()
        denum = train_cfg.ssn.sampler.fg_ratio + train_cfg.ssn.sampler.bg_ratio + train_cfg.ssn.sampler.incomplete_ratio
        fg_per_video = int(train_cfg.ssn.sampler.num_per_video * (train_cfg.ssn.sampler.fg_ratio / denum))
        bg_per_video = int(train_cfg.ssn.sampler.num_per_video * (train_cfg.ssn.sampler.bg_ratio / denum))
        incomplete_per_video = train_cfg.ssn.sampler.num_per_video - fg_per_video - bg_per_video
        losses['loss_act'] = F.cross_entropy(act_score[(act_indexer), :], labels[act_indexer])
        losses['loss_comp'] = completeness_loss(comp_score[(comp_indexer), :], labels[comp_indexer], fg_per_video, fg_per_video + incomplete_per_video, ohem_ratio=fg_per_video / incomplete_per_video)
        losses['loss_comp'] = losses['loss_comp'] * train_cfg.ssn.loss_weight.comp_loss_weight
        if bbox_pred is not None:
            reg_indexer = (prop_type == 0).nonzero().squeeze()
            bbox_targets = bbox_targets.view(-1, 2)
            bbox_pred = bbox_pred.view(-1, self.completeness_fc.out_features, 2)
            losses['loss_reg'] = classwise_regression_loss(bbox_pred[(reg_indexer), :, :], labels[reg_indexer], bbox_targets[(reg_indexer), :])
            losses['loss_reg'] = losses['loss_reg'] * train_cfg.ssn.loss_weight.reg_loss_weight
        return losses


FLOWNETS = Registry('flownet')


class Resample2dFunction(Function):

    @staticmethod
    def forward(ctx, input1, input2, kernel_size=1):
        assert input1.is_contiguous()
        assert input2.is_contiguous()
        ctx.save_for_backward(input1, input2)
        ctx.kernel_size = kernel_size
        _, d, _, _ = input1.size()
        b, _, h, w = input2.size()
        output = input1.new(b, d, h, w).zero_()
        resample2d_cuda.forward(input1, input2, output, kernel_size)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        assert grad_output.is_contiguous()
        input1, input2 = ctx.saved_tensors
        grad_input1 = Variable(input1.new(input1.size()).zero_())
        grad_input2 = Variable(input1.new(input2.size()).zero_())
        resample2d_cuda.backward(input1, input2, grad_output.data, grad_input1.data, grad_input2.data, ctx.kernel_size)
        return grad_input1, grad_input2, None


class Resample2d(Module):

    def __init__(self, kernel_size=1):
        super(Resample2d, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, input1, input2):
        input1_c = input1.contiguous()
        return Resample2dFunction.apply(input1_c, input2, self.kernel_size)


def SSIM_loss(img1, img2, kernel_size=8, stride=8, c1=1e-05, c2=1e-05):
    num = img1.size(0)
    channels = img1.size(1)
    kernel_h = kernel_w = kernel_size
    sigma = (kernel_w + kernel_h) / 12.0
    gauss_kernel = torch.zeros((1, 1, kernel_h, kernel_w)).type(img1.type())
    for h in range(kernel_h):
        for w in range(kernel_w):
            gauss_kernel[0, 0, h, w] = math.exp(-(math.pow(h - kernel_h / 2.0, 2) + math.pow(-kernel_w / 2.0, 2)) / (2.0 * sigma ** 2)) / (2 * 3.14159 * sigma ** 2)
    gauss_kernel = gauss_kernel / torch.sum(gauss_kernel)
    gauss_kernel = gauss_kernel.repeat(channels, 1, 1, 1)
    gauss_filter = nn.Conv2d(channels, channels, kernel_size, stride=stride, padding=0, groups=channels, bias=False)
    gauss_filter.weight.data = gauss_kernel
    gauss_filter.weight.requires_grad = False
    ux = gauss_filter(img1)
    uy = gauss_filter(img2)
    sx2 = gauss_filter(img1 ** 2)
    sy2 = gauss_filter(img2 ** 2)
    sxy = gauss_filter(img1 * img2)
    ux2 = ux ** 2
    uy2 = uy ** 2
    sx2 = sx2 - ux2
    sy2 = sy2 - uy2
    sxy = sxy - ux * uy
    lp = (2 * ux * uy + c1) / (ux2 + uy2 + c1)
    sc = (2 * sxy + c2) / (sx2 + sy2 + c2)
    ssim = lp * sc
    return (lp.numel() - torch.sum(ssim)) / num


def charbonnier_loss(difference, mask, alpha=1, beta=1.0, epsilon=0.001):
    """
    : sum( (x*beta)^2 + epsilon^2)^alpha
    """
    if mask is not None:
        assert difference.size(0) == mask.size(0)
        assert difference.size(2) == mask.size(2)
        assert difference.size(3) == mask.size(3)
    res = torch.pow(torch.pow(difference * beta, 2) + epsilon ** 2, alpha)
    if mask is not None:
        batch_pixels = torch.sum(mask)
        return torch.sum(res * mask) / batch_pixels
    else:
        batch_pixels = torch.numel(res)
        return torch.sum(res) / batch_pixels


def make_border_mask(batch, channels, height, width, tensor_type, border_ratio=0.1):
    border_width = round(border_ratio * min(height, width))
    mask = torch.ones(batch, channels, height, width).type(tensor_type)
    mask[:, :, :border_width, :] = 0
    mask[:, :, -border_width:, :] = 0
    mask[:, :, :border_width, :] = 0
    mask[:, :, -border_width:, :] = 0
    return mask


def make_smoothness_mask(batch, height, width, tensor_type):
    mask = torch.ones(batch, 2, height, width).type(tensor_type)
    mask[:1, (-1), :] = 0
    mask[:0, :, (-1)] = 0
    return mask


class MotionNet(nn.Module):

    def __init__(self, num_frames=1, rgb_disorder=False, scale=0.0039216, out_loss_indices=(0, 1, 2, 3, 4), out_prediction_indices=(0, 1, 2, 3, 4), out_prediction_rescale=True, frozen=False, use_photometric_loss=True, use_ssim_loss=True, use_smoothness_loss=True, photometric_loss_weights=(1, 1, 1, 1, 1), ssim_loss_weights=(0.16, 0.08, 0.04, 0.02, 0.01), smoothness_loss_weights=(1, 1, 1, 1, 1), pretrained=None):
        super(MotionNet, self).__init__()
        self.num_frames = num_frames
        self.rgb_disorder = rgb_disorder
        self.scale = scale
        self.out_loss_indices = out_loss_indices
        self.out_prediction_indices = out_prediction_indices
        self.out_prediction_rescale = out_prediction_rescale
        self.use_photometric_loss = use_photometric_loss
        self.use_ssim_loss = use_ssim_loss
        self.use_smoothness_loss = use_smoothness_loss
        if frozen:
            self.use_photometric_loss = False
            self.use_ssim_loss = False
            self.use_smoothness_loss = False
        self.frozen = frozen
        self.photometric_loss_weights = photometric_loss_weights
        self.ssim_loss_weights = ssim_loss_weights
        self.smoothness_loss_weights = smoothness_loss_weights
        if use_photometric_loss:
            assert len(out_prediction_indices) == len(photometric_loss_weights)
        if use_ssim_loss:
            assert len(out_prediction_indices) == len(ssim_loss_weights)
        if use_smoothness_loss:
            assert len(out_prediction_indices) == len(smoothness_loss_weights)
        self.pretrained = pretrained
        inplace = True
        self.lrn = nn.LocalResponseNorm(9, alpha=1, beta=0.5)
        self.conv1 = nn.Conv2d(3 * (num_frames + 1), 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
        self.conv1_1 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.relu1_1 = nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=True)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
        self.conv2_1 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.relu2_1 = nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=True)
        self.relu3 = nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
        self.conv3_1 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.relu3_1 = nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=True)
        self.relu4 = nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
        self.conv4_1 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.relu4_1 = nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=True)
        self.relu5 = nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.relu5_1 = nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=True)
        self.relu6 = nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
        self.conv6_1 = nn.Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.relu6_1 = nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
        self.conv_pr6 = nn.Conv2d(1024, 2 * num_frames, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.warp6 = Resample2d()
        self.warp5 = Resample2d()
        self.warp4 = Resample2d()
        self.warp3 = Resample2d()
        self.warp2 = Resample2d()
        self.conv_FlowDelta = nn.Conv2d(1, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.conv_FlowDelta.weight = nn.Parameter(torch.Tensor([[[[0, 0, 0], [0, 1, -1], [0, 0, 0]]], [[[0, 0, 0], [0, 1, 0], [0, -1, 0]]]]))
        self.conv_FlowDelta.weight.requires_grad = False
        self.deconv5 = nn.ConvTranspose2d(1024, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        self.relu_up5 = nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
        self.upsample_flow6to5 = nn.ConvTranspose2d(2 * self.num_frames, 2 * self.num_frames, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        self.smooth_conv5 = nn.Conv2d(2 * (self.num_frames + 512), 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.conv_pr5 = nn.Conv2d(512, 2 * num_frames, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.deconv4 = nn.ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        self.relu_up4 = nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
        self.upsample_flow5to4 = nn.ConvTranspose2d(2 * self.num_frames, 2 * self.num_frames, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        self.smooth_conv4 = nn.Conv2d(2 * self.num_frames + 512 + 256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.conv_pr4 = nn.Conv2d(256, 2 * num_frames, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        self.relu_up3 = nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
        self.upsample_flow4to3 = nn.ConvTranspose2d(2 * self.num_frames, 2 * self.num_frames, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        self.smooth_conv3 = nn.Conv2d(2 * self.num_frames + 256 + 128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.conv_pr3 = nn.Conv2d(128, 2 * num_frames, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        self.relu_up2 = nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
        self.upsample_flow3to2 = nn.ConvTranspose2d(2 * self.num_frames, 2 * self.num_frames, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        self.smooth_conv2 = nn.Conv2d(2 * self.num_frames + 128 + 64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.conv_pr2 = nn.Conv2d(64, 2 * num_frames, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.init_weights()

    @property
    def flip_rgb(self):
        return self.rgb_disorder

    @property
    def multiframe(self):
        return self.num_frames > 1

    def forward(self, x, train=True):
        assert x.ndimension() == 4
        scaling = torch.tensor(self.scale * (self.num_frames + 1)).type(x.type()).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        x = x * scaling
        imgs = torch.split(x, 3, 1)
        assert len(imgs) == self.num_frames + 1
        imgs_norm = [self.lrn(imgs[i]) for i in range(self.num_frames + 1)]
        conv1 = self.conv1(x)
        conv1_relu = self.relu1(conv1)
        conv1_1 = self.conv1_1(conv1_relu)
        conv1_1_relu = self.relu1_1(conv1_1)
        conv2 = self.conv2(conv1_1_relu)
        conv2_relu = self.relu2(conv2)
        conv2_1 = self.conv2_1(conv2_relu)
        conv2_1_relu = self.relu2_1(conv2_1)
        conv3 = self.conv3(conv2_1_relu)
        conv3_relu = self.relu3(conv3)
        conv3_1 = self.conv3_1(conv3_relu)
        conv3_1_relu = self.relu3_1(conv3_1)
        conv4 = self.conv4(conv3_1_relu)
        conv4_relu = self.relu4(conv4)
        conv4_1 = self.conv4_1(conv4_relu)
        conv4_1_relu = self.relu4_1(conv4_1)
        conv5 = self.conv5(conv4_1_relu)
        conv5_relu = self.relu5(conv5)
        conv5_1 = self.conv5_1(conv5_relu)
        conv5_1_relu = self.relu5_1(conv5_1)
        conv6 = self.conv6(conv5_1_relu)
        conv6_relu = self.relu6(conv6)
        conv6_1 = self.conv6_1(conv6_relu)
        conv6_1_relu = self.relu6_1(conv6_1)
        predict_flow6 = self.conv_pr6(conv6_1_relu)
        predictions_outs = []
        photometric_loss_outs = []
        ssim_loss_outs = []
        smoothness_loss_outs = []
        FlowScale6 = predict_flow6 * 0.625
        if train:
            predict_flow6_xs = torch.split(predict_flow6, 2, 1)
            FlowScale6_xs = torch.split(FlowScale6, 2, 1)
            downsampled_imgs_6 = [F.interpolate(img_norm, size=predict_flow6_xs[0].size()[-2:], mode='bilinear') for img_norm in imgs_norm]
            Warped6_xs = [self.warp6(downsampled_imgs_6[i + 1].contiguous(), FlowScale6_xs[i].contiguous()) for i in range(self.num_frames)]
            downsampled6_input_concat = torch.cat(downsampled_imgs_6[:self.num_frames], 1)
            warped6_concat = torch.cat(Warped6_xs, 1)
            PhotoDifference6 = downsampled6_input_concat - warped6_concat
            U6 = predict_flow6[:, ::2, (...)]
            V6 = predict_flow6[:, 1::2, (...)]
            FlowDeltasU6 = self.conv_FlowDelta(U6.view(-1, 1, U6.size(2), U6.size(3))).view(-1, self.num_frames, 2, U6.size(2), U6.size(3))
            FlowDeltasU6_xs = torch.split(FlowDeltasU6, 1, 1)
            FlowDeltasV6 = self.conv_FlowDelta(V6.view(-1, 1, V6.size(2), V6.size(3))).view(-1, self.num_frames, 2, V6.size(2), V6.size(3))
            FlowDeltasV6_xs = torch.split(FlowDeltasV6, 1, 1)
            SmoothnessMask6 = make_smoothness_mask(U6.size(0), U6.size(2), U6.size(3), U6.type())
            FlowDeltasUClean6_xs = [(FlowDeltasU6_x.squeeze() * SmoothnessMask6) for FlowDeltasU6_x in FlowDeltasU6_xs]
            FlowDeltasVClean6_xs = [(FlowDeltasV6_x.squeeze() * SmoothnessMask6) for FlowDeltasV6_x in FlowDeltasV6_xs]
            FlowDeltasUClean6 = torch.cat(FlowDeltasUClean6_xs, 1)
            FlowDeltasVClean6 = torch.cat(FlowDeltasVClean6_xs, 1)
            BorderMask6 = make_border_mask(U6.size(0), 3 * U6.size(1), U6.size(2), U6.size(3), U6.type(), border_ratio=0.1)
            photometric_loss_outs.append((PhotoDifference6, BorderMask6))
            ssim_loss_outs.append((warped6_concat, downsampled6_input_concat))
            BorderMask6 = make_border_mask(U6.size(0), 2 * U6.size(1), U6.size(2), U6.size(3), U6.type(), border_ratio=0.1)
            smoothness_loss_outs.append((FlowDeltasUClean6, FlowDeltasVClean6, BorderMask6))
        if self.out_prediction_rescale:
            predictions_outs.append(FlowScale6)
        else:
            predictions_outs.append(predict_flow6)
        deconv5 = self.deconv5(conv6_1_relu)
        deconv5_relu = self.relu_up5(deconv5)
        upsampled_flow6_to_5 = self.upsample_flow6to5(predict_flow6)
        concat5 = torch.cat((conv5_1_relu, deconv5_relu, upsampled_flow6_to_5), 1)
        smooth_conv5 = self.smooth_conv5(concat5)
        predict_flow5 = self.conv_pr5(smooth_conv5)
        FlowScale5 = predict_flow5 * 1.25
        if train:
            predict_flow5_xs = torch.split(predict_flow5, 2, 1)
            FlowScale5_xs = torch.split(FlowScale5, 2, 1)
            downsampled_imgs_5 = [F.interpolate(img_norm, size=predict_flow5_xs[0].size()[-2:], mode='bilinear') for img_norm in imgs_norm]
            Warped5_xs = [self.warp5(downsampled_imgs_5[i + 1].contiguous(), FlowScale5_xs[i].contiguous()) for i in range(self.num_frames)]
            downsampled5_input_concat = torch.cat(downsampled_imgs_5[:self.num_frames], 1)
            warped5_concat = torch.cat(Warped5_xs, 1)
            PhotoDifference5 = downsampled5_input_concat - warped5_concat
            U5 = predict_flow5[:, ::2, (...)]
            V5 = predict_flow5[:, 1::2, (...)]
            FlowDeltasU5 = self.conv_FlowDelta(U5.view(-1, 1, U5.size(2), U5.size(3))).view(-1, self.num_frames, 2, U5.size(2), U5.size(3))
            FlowDeltasU5_xs = torch.split(FlowDeltasU5, 1, 1)
            FlowDeltasV5 = self.conv_FlowDelta(V5.view(-1, 1, V5.size(2), V5.size(3))).view(-1, self.num_frames, 2, V5.size(2), V5.size(3))
            FlowDeltasV5_xs = torch.split(FlowDeltasV5, 1, 1)
            SmoothnessMask5 = make_smoothness_mask(U5.size(0), U5.size(2), U5.size(3), U5.type())
            FlowDeltasUClean5_xs = [(FlowDeltasU5_x.squeeze() * SmoothnessMask5) for FlowDeltasU5_x in FlowDeltasU5_xs]
            FlowDeltasVClean5_xs = [(FlowDeltasV5_x.squeeze() * SmoothnessMask5) for FlowDeltasV5_x in FlowDeltasV5_xs]
            FlowDeltasUClean5 = torch.cat(FlowDeltasUClean5_xs, 1)
            FlowDeltasVClean5 = torch.cat(FlowDeltasVClean5_xs, 1)
            BorderMask5 = make_border_mask(U5.size(0), 3 * U5.size(1), U5.size(2), U5.size(3), U5.type(), border_ratio=0.1)
            photometric_loss_outs.append((PhotoDifference5, BorderMask5))
            ssim_loss_outs.append((warped5_concat, downsampled5_input_concat))
            BorderMask5 = make_border_mask(U5.size(0), 2 * U5.size(1), U5.size(2), U5.size(3), U5.type(), border_ratio=0.1)
            smoothness_loss_outs.append((FlowDeltasUClean5, FlowDeltasVClean5, BorderMask5))
        if self.out_prediction_rescale:
            predictions_outs.append(FlowScale5)
        else:
            predictions_outs.append(predict_flow5)
        deconv4 = self.deconv4(smooth_conv5)
        deconv4_relu = self.relu_up4(deconv4)
        upsampled_flow5_to_4 = self.upsample_flow5to4(predict_flow5)
        concat4 = torch.cat((conv4_1_relu, deconv4_relu, upsampled_flow5_to_4), 1)
        smooth_conv4 = self.smooth_conv4(concat4)
        predict_flow4 = self.conv_pr4(smooth_conv4)
        FlowScale4 = predict_flow4 * 2.5
        if train:
            predict_flow4_xs = torch.split(predict_flow4, 2, 1)
            FlowScale4_xs = torch.split(FlowScale4, 2, 1)
            downsampled_imgs_4 = [F.interpolate(img_norm, size=predict_flow4_xs[0].size()[-2:], mode='bilinear') for img_norm in imgs_norm]
            Warped4_xs = [self.warp4(downsampled_imgs_4[i + 1].contiguous(), FlowScale4_xs[i].contiguous()) for i in range(self.num_frames)]
            downsampled4_input_concat = torch.cat(downsampled_imgs_4[:self.num_frames], 1)
            warped4_concat = torch.cat(Warped4_xs, 1)
            PhotoDifference4 = downsampled4_input_concat - warped4_concat
            U4 = predict_flow4[:, ::2, (...)]
            V4 = predict_flow4[:, 1::2, (...)]
            FlowDeltasU4 = self.conv_FlowDelta(U4.view(-1, 1, U4.size(2), U4.size(3))).view(-1, self.num_frames, 2, U4.size(2), U4.size(3))
            FlowDeltasU4_xs = torch.split(FlowDeltasU4, 1, 1)
            FlowDeltasV4 = self.conv_FlowDelta(V4.view(-1, 1, V4.size(2), V4.size(3))).view(-1, self.num_frames, 2, V4.size(2), V4.size(3))
            FlowDeltasV4_xs = torch.split(FlowDeltasV4, 1, 1)
            SmoothnessMask4 = make_smoothness_mask(U4.size(0), U4.size(2), U4.size(3), U4.type())
            FlowDeltasUClean4_xs = [(FlowDeltasU4_x.squeeze() * SmoothnessMask4) for FlowDeltasU4_x in FlowDeltasU4_xs]
            FlowDeltasVClean4_xs = [(FlowDeltasV4_x.squeeze() * SmoothnessMask4) for FlowDeltasV4_x in FlowDeltasV4_xs]
            FlowDeltasUClean4 = torch.cat(FlowDeltasUClean4_xs, 1)
            FlowDeltasVClean4 = torch.cat(FlowDeltasVClean4_xs, 1)
            BorderMask4 = make_border_mask(U4.size(0), 3 * U4.size(1), U4.size(2), U4.size(3), U4.type(), border_ratio=0.1)
            photometric_loss_outs.append((PhotoDifference4, BorderMask4))
            ssim_loss_outs.append((warped4_concat, downsampled4_input_concat))
            BorderMask4 = make_border_mask(U4.size(0), 2 * U4.size(1), U4.size(2), U4.size(3), U4.type(), border_ratio=0.1)
            smoothness_loss_outs.append((FlowDeltasUClean4, FlowDeltasVClean4, BorderMask4))
        if self.out_prediction_rescale:
            predictions_outs.append(FlowScale4)
        else:
            predictions_outs.append(predict_flow4)
        deconv3 = self.deconv3(smooth_conv4)
        deconv3_relu = self.relu_up3(deconv3)
        upsampled_flow4_to_3 = self.upsample_flow4to3(predict_flow4)
        concat3 = torch.cat((conv3_1_relu, deconv3_relu, upsampled_flow4_to_3), 1)
        smooth_conv3 = self.smooth_conv3(concat3)
        predict_flow3 = self.conv_pr3(smooth_conv3)
        FlowScale3 = predict_flow3 * 5.0
        if train:
            predict_flow3_xs = torch.split(predict_flow3, 2, 1)
            FlowScale3_xs = torch.split(FlowScale3, 2, 1)
            downsampled_imgs_3 = [F.interpolate(img_norm, size=predict_flow3_xs[0].size()[-2:], mode='bilinear') for img_norm in imgs_norm]
            Warped3_xs = [self.warp3(downsampled_imgs_3[i + 1].contiguous(), FlowScale3_xs[i].contiguous()) for i in range(self.num_frames)]
            downsampled3_input_concat = torch.cat(downsampled_imgs_3[:self.num_frames], 1)
            warped3_concat = torch.cat(Warped3_xs, 1)
            PhotoDifference3 = downsampled3_input_concat - warped3_concat
            U3 = predict_flow3[:, ::2, (...)]
            V3 = predict_flow3[:, 1::2, (...)]
            FlowDeltasU3 = self.conv_FlowDelta(U3.view(-1, 1, U3.size(2), U3.size(3))).view(-1, self.num_frames, 2, U3.size(2), U3.size(3))
            FlowDeltasU3_xs = torch.split(FlowDeltasU3, 1, 1)
            FlowDeltasV3 = self.conv_FlowDelta(V3.view(-1, 1, V3.size(2), V3.size(3))).view(-1, self.num_frames, 2, V3.size(2), V3.size(3))
            FlowDeltasV3_xs = torch.split(FlowDeltasV3, 1, 1)
            SmoothnessMask3 = make_smoothness_mask(U3.size(0), U3.size(2), U3.size(3), U3.type())
            FlowDeltasUClean3_xs = [(FlowDeltasU3_x.squeeze() * SmoothnessMask3) for FlowDeltasU3_x in FlowDeltasU3_xs]
            FlowDeltasVClean3_xs = [(FlowDeltasV3_x.squeeze() * SmoothnessMask3) for FlowDeltasV3_x in FlowDeltasV3_xs]
            FlowDeltasUClean3 = torch.cat(FlowDeltasUClean3_xs, 1)
            FlowDeltasVClean3 = torch.cat(FlowDeltasVClean3_xs, 1)
            BorderMask3 = make_border_mask(U3.size(0), 3 * U3.size(1), U3.size(2), U3.size(3), U3.type(), border_ratio=0.1)
            photometric_loss_outs.append((PhotoDifference3, BorderMask3))
            ssim_loss_outs.append((warped3_concat, downsampled3_input_concat))
            BorderMask3 = make_border_mask(U3.size(0), 2 * U3.size(1), U3.size(2), U3.size(3), U3.type(), border_ratio=0.1)
            smoothness_loss_outs.append((FlowDeltasUClean3, FlowDeltasVClean3, BorderMask3))
        if self.out_prediction_rescale:
            predictions_outs.append(FlowScale3)
        else:
            predictions_outs.append(predict_flow3)
        deconv2 = self.deconv2(smooth_conv3)
        deconv2_relu = self.relu_up2(deconv2)
        upsampled_flow3_to_2 = self.upsample_flow3to2(predict_flow3)
        concat2 = torch.cat((conv2_1_relu, deconv2_relu, upsampled_flow3_to_2), 1)
        smooth_conv2 = self.smooth_conv2(concat2)
        predict_flow2 = self.conv_pr2(smooth_conv2)
        FlowScale2 = predict_flow2 * 10.0
        if train:
            predict_flow2_xs = torch.split(predict_flow2, 2, 1)
            FlowScale2_xs = torch.split(FlowScale2, 2, 1)
            downsampled_imgs_2 = [F.interpolate(img_norm, size=predict_flow2_xs[0].size()[-2:], mode='bilinear') for img_norm in imgs_norm]
            Warped2_xs = [self.warp2(downsampled_imgs_2[i + 1].contiguous(), FlowScale2_xs[i].contiguous()) for i in range(self.num_frames)]
            downsampled2_input_concat = torch.cat(downsampled_imgs_2[:self.num_frames], 1)
            warped2_concat = torch.cat(Warped2_xs, 1)
            PhotoDifference2 = downsampled2_input_concat - warped2_concat
            U2 = predict_flow2[:, ::2, (...)]
            V2 = predict_flow2[:, 1::2, (...)]
            FlowDeltasU2 = self.conv_FlowDelta(U2.view(-1, 1, U2.size(2), U2.size(3))).view(-1, self.num_frames, 2, U2.size(2), U2.size(3))
            FlowDeltasU2_xs = torch.split(FlowDeltasU2, 1, 1)
            FlowDeltasV2 = self.conv_FlowDelta(V2.view(-1, 1, V2.size(2), V2.size(3))).view(-1, self.num_frames, 2, V2.size(2), V2.size(3))
            FlowDeltasV2_xs = torch.split(FlowDeltasV2, 1, 1)
            SmoothnessMask2 = make_smoothness_mask(U2.size(0), U2.size(2), U2.size(3), U2.type())
            FlowDeltasUClean2_xs = [(FlowDeltasU2_x.squeeze() * SmoothnessMask2) for FlowDeltasU2_x in FlowDeltasU2_xs]
            FlowDeltasVClean2_xs = [(FlowDeltasV2_x.squeeze() * SmoothnessMask2) for FlowDeltasV2_x in FlowDeltasV2_xs]
            FlowDeltasUClean2 = torch.cat(FlowDeltasUClean2_xs, 1)
            FlowDeltasVClean2 = torch.cat(FlowDeltasVClean2_xs, 1)
            BorderMask2 = make_border_mask(U2.size(0), 3 * U2.size(1), U2.size(2), U2.size(3), U2.type(), border_ratio=0.1)
            if self.out_prediction_rescale:
                predictions_outs.append(FlowScale2)
            else:
                predictions_outs.append(predict_flow2)
            photometric_loss_outs.append((PhotoDifference2, BorderMask2))
            ssim_loss_outs.append((warped2_concat, downsampled2_input_concat))
            BorderMask2 = make_border_mask(U2.size(0), 2 * U2.size(1), U2.size(2), U2.size(3), U2.type(), border_ratio=0.1)
            smoothness_loss_outs.append((FlowDeltasUClean2, FlowDeltasVClean2, BorderMask2))
        outs_predictions = [predictions_outs[i] for i in self.out_prediction_indices]
        if train:
            return tuple(outs_predictions), photometric_loss_outs, ssim_loss_outs, smoothness_loss_outs
        else:
            return tuple(outs_predictions), None, None, None

    def init_weights(self):
        if isinstance(self.pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, self.pretrained, strict=False, logger=logger)
        elif self.pretrained is None:
            for name, m in self.named_modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                    if name != 'conv_FlowDelta':
                        kaiming_init(m)
                    else:
                        None
        else:
            raise TypeError('pretrained must be a str or None')

    def loss(self, photometric_loss_outs, ssim_loss_outs, smoothness_loss_outs, direction='forward'):
        assert direction in ['forward', 'backward']
        losses = dict()
        if self.use_photometric_loss:
            for i, ind in enumerate(self.out_loss_indices):
                losses['photometric_loss_{}_{}'.format(ind, direction)] = self.photometric_loss_weights[i] * charbonnier_loss(photometric_loss_outs[i][0], photometric_loss_outs[i][1], alpha=0.4, beta=255)
        if self.use_ssim_loss:
            for i, ind in enumerate(self.out_loss_indices):
                losses['ssim_loss_{}_{}'.format(ind, direction)] = self.ssim_loss_weights[i] * SSIM_loss(ssim_loss_outs[i][0], ssim_loss_outs[i][1], kernel_size=8, stride=8, c1=0.0001, c2=0.001)
        if self.use_smoothness_loss:
            for i, ind in enumerate(self.out_loss_indices):
                losses['smoothness_loss_{}_{}'.format(ind, direction)] = self.smoothness_loss_weights[i] * charbonnier_loss(smoothness_loss_outs[i][0], smoothness_loss_outs[i][2], alpha=0.3, beta=5) + self.smoothness_loss_weights[i] * charbonnier_loss(smoothness_loss_outs[i][1], smoothness_loss_outs[i][2], alpha=0.3, beta=5)
        return losses

    def train(self, mode=True):
        super(MotionNet, self).train(mode)
        if self.frozen:
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                    for param in m.parameters():
                        param.requires_grad = False


norm_cfg = {'BN': ('bn', nn.BatchNorm2d), 'SyncBN': ('bn', None), 'GN': ('gn', nn.GroupNorm)}


def build_norm_layer(cfg, num_features, postfix=''):
    """ Build normalization layer
    Args:
        cfg (dict): cfg should contain:
            type (str): identify norm layer type.
            layer args: args needed to instantiate a norm layer.
            frozen (bool): [optional] whether stop gradient updates
                of norm layer, it is helpful to set frozen mode
                in backbone's norms.
        num_features (int): number of channels from input
        postfix (int, str): appended into norm abbreation to
            create named layer.
    Returns:
        name (str): abbreation + postfix
        layer (nn.Module): created norm layer
    """
    assert isinstance(cfg, dict) and 'type' in cfg
    cfg_ = cfg.copy()
    layer_type = cfg_.pop('type')
    if layer_type not in norm_cfg:
        raise KeyError('Unrecognized norm type {}'.format(layer_type))
    else:
        abbr, norm_layer = norm_cfg[layer_type]
        if norm_layer is None:
            raise NotImplementedError
    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)
    frozen = cfg_.pop('frozen', False)
    cfg_.setdefault('eps', 1e-05)
    if layer_type != 'GN':
        layer = norm_layer(num_features, **cfg_)
    else:
        assert 'num_groups' in cfg_
        layer = norm_layer(num_channels=num_features, **cfg_)
    if frozen:
        for param in layer.parameters():
            param.requires_grad = False
    return name, layer


class ConvModule(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, normalize=None, activation='relu', inplace=True, activate_last=True):
        super(ConvModule, self).__init__()
        self.with_norm = normalize is not None
        self.with_activatation = activation is not None
        self.with_bias = bias
        self.activation = activation
        self.activate_last = activate_last
        if self.with_norm and self.with_bias:
            warnings.warn('ConvModule has norm and bias at the same time')
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias)
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = self.conv.padding
        self.dilation = self.conv.dilation
        self.transposed = self.conv.transposed
        self.output_padding = self.conv.output_padding
        self.groups = self.conv.groups
        if self.with_norm:
            norm_channels = out_channels if self.activate_last else in_channels
            self.norm_name, norm = build_norm_layer(normalize, norm_channels)
            self.add_module(self.norm_name, norm)
        if self.with_activatation:
            assert activation in ['relu'], 'Only ReLU supported.'
            if self.activation == 'relu':
                self.activate = nn.ReLU(inplace=inplace)
        self.init_weights()

    @property
    def norm(self):
        return getattr(self, self.norm_name)

    def init_weights(self):
        nonlinearity = 'relu' if self.activation is None else self.activation
        kaiming_init(self.conv, nonlinearity=nonlinearity)
        if self.with_norm:
            constant_init(self.norm, 1, bias=0)

    def forward(self, x, activate=True, norm=True):
        if self.activate_last:
            x = self.conv(x)
            if norm and self.with_norm:
                x = self.norm(x)
            if activate and self.with_activatation:
                x = self.activate(x)
        else:
            if norm and self.with_norm:
                x = self.norm(x)
            if activate and self.with_activatation:
                x = self.activate(x)
            x = self.conv(x)
        return x


NECKS = Registry('neck')


class FPN(nn.Module):

    def __init__(self, in_channels, out_channels, num_outs, start_level=0, end_level=-1, add_extra_convs=False, extra_convs_on_inputs=True, normalize=None, activation=None):
        super(FPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.activation = activation
        self.with_bias = normalize is None
        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        self.extra_convs_on_inputs = extra_convs_on_inputs
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(in_channels[i], out_channels, 1, normalize=normalize, bias=self.with_bias, activation=self.activation, inplace=False)
            fpn_conv = ConvModule(out_channels, out_channels, 3, padding=1, normalize=normalize, bias=self.with_bias, activation=self.activation, inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.extra_convs_on_inputs:
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(in_channels, out_channels, 3, stride=2, padding=1, normalize=normalize, bias=self.with_bias, activation=self.activation, inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        laterals = [lateral_conv(inputs[i + self.start_level]) for i, lateral_conv in enumerate(self.lateral_convs)]
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] += F.interpolate(laterals[i], scale_factor=2, mode='nearest')
        outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)]
        if self.num_outs > len(outs):
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            else:
                if self.extra_convs_on_inputs:
                    orig = inputs[self.backbone_end_level - 1]
                    outs.append(self.fpn_convs[used_backbone_levels](orig))
                else:
                    outs.append(self.fpn_convs[used_backbone_levels](outs[-1]))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)


ROI_EXTRACTORS = Registry('roi_extractor')


class SingleRoIExtractor(nn.Module):
    """Extract RoI features from a single level feature map.

    If there are multiple input feature levels, each RoI is mapped to a level
    according to its scale.

    Args:
        roi_layer (dict): Specify RoI layer type and arguments.
        out_channels (int): Output channels of RoI layers.
        featmap_strides (int): Strides of input feature maps.
        finest_scale (int): Scale threshold of mapping to level 0.
    """

    def __init__(self, roi_layer, out_channels, featmap_strides, finest_scale=56):
        super(SingleRoIExtractor, self).__init__()
        self.roi_layers = self.build_roi_layers(roi_layer, featmap_strides)
        self.out_channels = out_channels
        self.featmap_strides = featmap_strides
        self.finest_scale = finest_scale

    @property
    def num_inputs(self):
        """int: Input feature map levels."""
        return len(self.featmap_strides)

    def init_weights(self):
        pass

    def build_roi_layers(self, layer_cfg, featmap_strides):
        cfg = layer_cfg.copy()
        layer_type = cfg.pop('type')
        assert hasattr(ops, layer_type)
        layer_cls = getattr(ops, layer_type)
        roi_layers = nn.ModuleList([layer_cls(spatial_scale=1 / s, **cfg) for s in featmap_strides])
        return roi_layers

    def map_roi_levels(self, rois, num_levels):
        """Map rois to corresponding feature levels by scales.

        - scale < finest_scale: level 0
        - finest_scale <= scale < finest_scale * 2: level 1
        - finest_scale * 2 <= scale < finest_scale * 4: level 2
        - scale >= finest_scale * 4: level 3

        Args:
            rois (Tensor): Input RoIs, shape (k, 5).
            num_levels (int): Total level number.

        Returns:
            Tensor: Level index (0-based) of each RoI, shape (k, )
        """
        scale = torch.sqrt((rois[:, (3)] - rois[:, (1)] + 1) * (rois[:, (4)] - rois[:, (2)] + 1))
        target_lvls = torch.floor(torch.log2(scale / self.finest_scale + 1e-06))
        target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long()
        return target_lvls

    def forward(self, feats, rois):
        if len(feats) == 1:
            return self.roi_layers[0](feats[0], rois)
        out_size = self.roi_layers[0].out_size
        num_levels = len(feats)
        target_lvls = self.map_roi_levels(rois, num_levels)
        roi_feats = torch.FloatTensor(rois.size()[0], self.out_channels, out_size, out_size).fill_(0)
        for i in range(num_levels):
            inds = target_lvls == i
            if inds.any():
                rois_ = rois[(inds), :]
                roi_feats_t = self.roi_layers[i](feats[i], rois_)
                roi_feats[inds] += roi_feats_t
        return roi_feats


class SingleRoIStraight3DExtractor(nn.Module):
    """Extract RoI features from a single level feature map.

    If there are multiple input feature levels, each RoI is mapped to a level
    according to its scale.

    Args:
        roi_layer (dict): Specify RoI layer type and arguments.
        out_channels (int): Output channels of RoI layers.
        featmap_strides (int): Strides of input feature maps.
        finest_scale (int): Scale threshold of mapping to level 0.
    """

    def __init__(self, roi_layer, out_channels, featmap_strides, finest_scale=56, with_temporal_pool=False):
        super(SingleRoIStraight3DExtractor, self).__init__()
        self.roi_layers = self.build_roi_layers(roi_layer, featmap_strides)
        self.out_channels = out_channels
        self.featmap_strides = featmap_strides
        self.finest_scale = finest_scale
        self.with_temporal_pool = with_temporal_pool

    @property
    def num_inputs(self):
        """int: Input feature map levels."""
        return len(self.featmap_strides)

    def init_weights(self):
        pass

    def build_roi_layers(self, layer_cfg, featmap_strides):
        cfg = layer_cfg.copy()
        layer_type = cfg.pop('type')
        assert hasattr(ops, layer_type)
        layer_cls = getattr(ops, layer_type)
        roi_layers = nn.ModuleList([layer_cls(spatial_scale=1 / s, **cfg) for s in featmap_strides])
        return roi_layers

    def map_roi_levels(self, rois, num_levels):
        """Map rois to corresponding feature levels by scales.

        - scale < finest_scale: level 0
        - finest_scale <= scale < finest_scale * 2: level 1
        - finest_scale * 2 <= scale < finest_scale * 4: level 2
        - scale >= finest_scale * 4: level 3

        Args:
            rois (Tensor): Input RoIs, shape (k, 5).
            num_levels (int): Total level number.

        Returns:
            Tensor: Level index (0-based) of each RoI, shape (k, )
        """
        scale = torch.sqrt((rois[:, (3)] - rois[:, (1)] + 1) * (rois[:, (4)] - rois[:, (2)] + 1))
        target_lvls = torch.floor(torch.log2(scale / self.finest_scale + 1e-06))
        target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long()
        return target_lvls

    def forward(self, feats, rois):
        feats = list(feats)
        if len(feats) == 1:
            if self.with_temporal_pool:
                feats[0] = torch.mean(feats[0], 2, keepdim=True)
            roi_feats = []
            for t in range(feats[0].size(2)):
                feat = feats[0][:, :, (t), :, :].contiguous()
                roi_feats.append(self.roi_layers[0](feat, rois))
            return torch.stack(roi_feats, dim=2)
        if self.with_temporal_pool:
            for i in range(len(feats)):
                feats[i] = torch.mean(feats[i], 2, keepdim=True)
        t_size = feats[0].size(2)
        out_size = self.roi_layers[0].out_size
        num_levels = len(feats)
        target_lvls = self.map_roi_levels(rois, num_levels)
        roi_feats = torch.FloatTensor(rois.size()[0], self.out_channels, t_size, out_size, out_size).fill_(0)
        for i in range(num_levels):
            inds = target_lvls == i
            if inds.any():
                rois_ = rois[(inds), :]
                for t in range(t_size):
                    feat_ = feats[i][:, :, (t), :, :].contiguous()
                    roi_feats_t = self.roi_layers[i](feat_, rois_)
                    roi_feats[(inds), :, (t), :, :] += roi_feats_t
        return roi_feats


SEGMENTAL_CONSENSUSES = Registry('segmental_consensus')


class _SimpleConsensus(torch.autograd.Function):
    """Simplest segmental consensus module"""

    def __init__(self, consensus_type='avg', dim=1):
        super(_SimpleConsensus, self).__init__()
        assert consensus_type in ['avg']
        self.consensus_type = consensus_type
        self.dim = dim
        self.shape = None

    def forward(self, x):
        self.shape = x.size()
        if self.consensus_type == 'avg':
            output = x.mean(dim=self.dim, keepdim=True)
        else:
            output = None
        return output

    def backward(self, grad_output):
        if self.consensus_type == 'avg':
            grad_in = grad_output.expand(self.shape) / float(self.shape[self.dim])
        else:
            grad_in = None
        return grad_in


class SimpleConsensus(nn.Module):

    def __init__(self, consensus_type, dim=1):
        super(SimpleConsensus, self).__init__()
        assert consensus_type in ['avg']
        self.consensus_type = consensus_type
        self.dim = dim

    def init_weights(self):
        pass

    def forward(self, input):
        return _SimpleConsensus(self.consensus_type, self.dim)(input)


def parse_stage_config(stage_cfg):
    if isinstance(stage_cfg, int):
        return (stage_cfg,), stage_cfg
    elif isinstance(stage_cfg, tuple) or isinstance(stage_cfg, list):
        return stage_cfg, sum(stage_cfg)
    else:
        raise ValueError('Incorrect STPP config {}'.format(stage_cfg))


class StructuredTemporalPyramidPooling(nn.Module):

    def __init__(self, standalong_classifier=False, stpp_cfg=(1, (1, 2), 1), num_seg=(2, 5, 2)):
        super(StructuredTemporalPyramidPooling, self).__init__()
        self.sc = standalong_classifier
        starting_parts, starting_mult = parse_stage_config(stpp_cfg[0])
        course_parts, course_mult = parse_stage_config(stpp_cfg[1])
        ending_parts, ending_mult = parse_stage_config(stpp_cfg[2])
        self.feat_multiplier = starting_mult + course_mult + ending_mult
        self.parts = starting_parts, course_parts, ending_parts
        self.norm_num = starting_mult, course_mult, ending_mult
        self.num_seg = num_seg

    def init_weights(self):
        pass

    def forward(self, input, scaling):
        x1 = self.num_seg[0]
        x2 = x1 + self.num_seg[1]
        n_seg = x2 + self.num_seg[2]
        feat_dim = input.size(1)
        src = input.view(-1, n_seg, feat_dim)
        num_sample = src.size(0)
        scaling = scaling.view(-1, 2)

        def get_stage_stpp(stage_feat, stage_parts, norm_num, scaling):
            stage_stpp = []
            stage_len = stage_feat.size(1)
            for n_part in stage_parts:
                ticks = torch.arange(0, stage_len + 1e-05, stage_len / n_part)
                for i in range(n_part):
                    part_feat = stage_feat[:, int(ticks[i]):int(ticks[i + 1]), :].mean(dim=1) / norm_num
                    if scaling is not None:
                        part_feat = part_feat * scaling.view(num_sample, 1)
                    stage_stpp.append(part_feat)
            return stage_stpp
        feature_parts = []
        feature_parts.extend(get_stage_stpp(src[:, :x1, :], self.parts[0], self.norm_num[0], scaling[:, (0)]))
        feature_parts.extend(get_stage_stpp(src[:, x1:x2, :], self.parts[1], self.norm_num[1], None))
        feature_parts.extend(get_stage_stpp(src[:, x2:, :], self.parts[2], self.norm_num[2], scaling[:, (1)]))
        stpp_feat = torch.cat(feature_parts, dim=1)
        if not self.sc:
            return stpp_feat, stpp_feat
        else:
            course_feat = src[:, x1:x2, :].mean(dim=1)
            return course_feat, stpp_feat


class STPPReorganized(nn.Module):

    def __init__(self, feat_dim, act_score_len, comp_score_len, reg_score_len, standalong_classifier=False, with_regression=True, stpp_cfg=(1, (1, 2), 1)):
        super(STPPReorganized, self).__init__()
        self.sc = standalong_classifier
        self.feat_dim = feat_dim
        self.act_score_len = act_score_len
        self.comp_score_len = comp_score_len
        self.reg_score_len = reg_score_len
        self.with_regression = with_regression
        starting_parts, starting_mult = parse_stage_config(stpp_cfg[0])
        course_parts, course_mult = parse_stage_config(stpp_cfg[1])
        ending_parts, ending_mult = parse_stage_config(stpp_cfg[2])
        self.feat_multiplier = starting_mult + course_mult + ending_mult
        self.stpp_cfg = starting_parts, course_parts, ending_parts
        self.act_slice = slice(0, self.act_score_len if self.sc else self.act_score_len * self.feat_multiplier)
        self.comp_slice = slice(self.act_slice.stop, self.act_slice.stop + self.comp_score_len * self.feat_multiplier)
        self.reg_slice = slice(self.comp_slice.stop, self.comp_slice.stop + self.reg_score_len * self.feat_multiplier)

    def init_weights(self):
        pass

    def forward(self, input, proposal_ticks, scaling):
        assert input.size(1) == self.feat_dim
        n_ticks = proposal_ticks.size(0)
        out_act_scores = torch.zeros((n_ticks, self.act_score_len)).type_as(input)
        raw_act_scores = input[:, (self.act_slice)]
        out_comp_scores = torch.zeros((n_ticks, self.comp_score_len)).type_as(input)
        raw_comp_scores = input[:, (self.comp_slice)]
        if self.with_regression:
            out_reg_scores = torch.zeros((n_ticks, self.reg_score_len)).type_as(input)
            raw_reg_scores = input[:, (self.reg_slice)]
        else:
            out_reg_scores = None
            raw_reg_scores = None

        def pspool(out_scores, index, raw_scores, ticks, scaling, score_len, stpp_cfg):
            offset = 0
            for stage_idx, stage_cfg in enumerate(stpp_cfg):
                if stage_idx == 0:
                    s = scaling[0]
                elif stage_idx == len(stpp_cfg) - 1:
                    s = scaling[1]
                else:
                    s = 1.0
                stage_cnt = sum(stage_cfg)
                left = ticks[stage_idx]
                right = max(ticks[stage_idx] + 1, ticks[stage_idx + 1])
                if right <= 0 or left >= raw_scores.size(0):
                    offset += stage_cnt
                    continue
                for n_part in stage_cfg:
                    part_ticks = np.arange(left, right + 1e-05, (right - left) / n_part)
                    for i in range(n_part):
                        pl = int(part_ticks[i])
                        pr = int(part_ticks[i + 1])
                        if pr - pl >= 1:
                            out_scores[(index), :] += raw_scores[pl:pr, offset * score_len:(offset + 1) * score_len].mean(dim=0) * s
                        offset += 1
        for i in range(n_ticks):
            ticks = proposal_ticks[i].cpu().numpy()
            if self.sc:
                out_act_scores[(i), :] = raw_act_scores[ticks[1]:max(ticks[1] + 1, ticks[2]), :].mean(dim=0)
            else:
                pspool(out_act_scores, i, raw_act_scores, ticks, scaling[i], self.act_score_len, self.stpp_cfg)
            pspool(out_comp_scores, i, raw_comp_scores, ticks, scaling[i], self.comp_score_len, self.stpp_cfg)
            if self.with_regression:
                pspool(out_reg_scores, i, raw_reg_scores, ticks, scaling[i], self.reg_score_len, self.stpp_cfg)
        return out_act_scores, out_comp_scores, out_reg_scores


class ResI3DLayer(nn.Module):

    def __init__(self, depth, pretrained=None, pretrained2d=True, stage=3, spatial_stride=2, temporal_stride=1, dilation=1, style='pytorch', inflate_freq=1, inflate_style='3x1x1', bn_eval=True, bn_frozen=True, all_frozen=False, with_cp=False):
        super(ResI3DLayer, self).__init__()
        self.bn_eval = bn_eval
        self.bn_frozen = bn_frozen
        self.all_frozen = all_frozen
        self.stage = stage
        block, stage_blocks = ResNet_I3D.arch_settings[depth]
        self.pretrained = pretrained
        self.pretrained2d = pretrained2d
        stage_block = stage_blocks[stage]
        planes = 64 * 2 ** stage
        inplanes = 64 * 2 ** (stage - 1) * block.expansion
        self.inflate_freq = inflate_freq if not isinstance(inflate_freq, int) else (inflate_freq,) * stage
        res_layer = make_res_layer(block, inplanes, planes, stage_block, spatial_stride=spatial_stride, temporal_stride=temporal_stride, dilation=dilation, style=style, inflate_freq=self.inflate_freq, inflate_style='3x1x1', with_cp=with_cp)
        self.add_module('layer{}'.format(stage + 1), res_layer)

    def init_weights(self):
        if isinstance(self.pretrained, str):
            logger = logging.getLogger()
            if self.pretrained2d:
                resnet2d = ResNet(self.depth)
                load_checkpoint(resnet2d, self.pretrained, strict=False, logger=logger)
                for name, module in self.named_modules():
                    if isinstance(module, NonLocalModule):
                        module.init_weights()
                    elif isinstance(module, nn.Conv3d) and rhasattr(resnet2d, name):
                        new_weight = rgetattr(resnet2d, name).weight.data.unsqueeze(2).expand_as(module.weight) / module.weight.data.shape[2]
                        module.weight.data.copy_(new_weight)
                        logging.info('{}.weight loaded from weights file into {}'.format(name, new_weight.shape))
                        if hasattr(module, 'bias') and module.bias is not None:
                            new_bias = rgetattr(resnet2d, name).bias.data
                            module.bias.data.copy_(new_bias)
                            logging.info('{}.bias loaded from weights file into {}'.format(name, new_bias.shape))
                    elif isinstance(module, nn.BatchNorm3d) and rhasattr(resnet2d, name):
                        for attr in ['weight', 'bias', 'running_mean', 'running_var']:
                            logging.info('{}.{} loaded from weights file into {}'.format(name, attr, getattr(rgetattr(resnet2d, name), attr).shape))
                            setattr(module, attr, getattr(rgetattr(resnet2d, name), attr))
            else:
                load_checkpoint(self, self.pretrained, strict=False, logger=logger)
        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        res_layer = getattr(self, 'layer{}'.format(self.stage + 1))
        out = res_layer(x)
        return out

    def train(self, mode=True):
        super(ResI3DLayer, self).train(mode)
        if self.bn_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm3d):
                    m.eval()
                    if self.bn_frozen:
                        for params in m.parameters():
                            params.requires_grad = False
        if self.bn_frozen:
            res_layer = getattr(self, 'layer{}'.format(self.stage + 1))
            for m in res_layer:
                if isinstance(m, nn.BatchNorm3d):
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False
        if self.all_frozen:
            res_layer = getattr(self, 'layer{}'.format(self.stage + 1))
            res_layer.eval()
            for param in res_layer.parameters():
                param.requires_grad = False


class ResLayer(nn.Module):

    def __init__(self, depth, pretrained=None, stage=3, stride=2, dilation=1, style='pytorch', bn_eval=True, bn_frozen=True, all_frozen=False, with_cp=False):
        super(ResLayer, self).__init__()
        self.bn_eval = bn_eval
        self.bn_frozen = bn_frozen
        self.all_frozen = all_frozen
        self.stage = stage
        block, stage_blocks = ResNet.arch_settings[depth]
        self.pretrained = pretrained
        stage_block = stage_blocks[stage]
        planes = 64 * 2 ** stage
        inplanes = 64 * 2 ** (stage - 1) * block.expansion
        res_layer = make_res_layer(block, inplanes, planes, stage_block, stride=stride, dilation=dilation, style=style, with_cp=with_cp)
        self.add_module('layer{}'.format(stage + 1), res_layer)

    def init_weights(self):
        if isinstance(self.pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, self.pretrained, strict=False, logger=logger)
        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        res_layer = getattr(self, 'layer{}'.format(self.stage + 1))
        out = res_layer(x)
        return out

    def train(self, mode=True):
        super(ResLayer, self).train(mode)
        if self.bn_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    if self.bn_frozen:
                        for params in m.parameters():
                            params.requires_grad = False
        if self.bn_frozen:
            res_layer = getattr(self, 'layer{}'.format(self.stage + 1))
            for m in res_layer:
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False
        if self.all_frozen:
            res_layer = getattr(self, 'layer{}'.format(self.stage + 1))
            res_layer.eval()
            for param in mod.parameters():
                param.requires_grad = False


class SimpleSpatialModule(nn.Module):

    def __init__(self, spatial_type='avg', spatial_size=7):
        super(SimpleSpatialModule, self).__init__()
        assert spatial_type in ['avg']
        self.spatial_type = spatial_type
        self.spatial_size = spatial_size if not isinstance(spatial_size, int) else (spatial_size, spatial_size)
        if self.spatial_type == 'avg':
            self.op = nn.AvgPool2d(self.spatial_size, stride=1, padding=0)

    def init_weights(self):
        pass

    def forward(self, input):
        return self.op(input)


class SimpleSpatialTemporalModule(nn.Module):

    def __init__(self, spatial_type='avg', spatial_size=7, temporal_size=1):
        super(SimpleSpatialTemporalModule, self).__init__()
        assert spatial_type in ['avg', 'max']
        self.spatial_type = spatial_type
        self.spatial_size = spatial_size
        if spatial_size != -1:
            self.spatial_size = spatial_size, spatial_size
        self.temporal_size = temporal_size
        assert not (self.spatial_size == -1) ^ (self.temporal_size == -1)
        if self.temporal_size == -1 and self.spatial_size == -1:
            self.pool_size = 1, 1, 1
            if self.spatial_type == 'avg':
                self.pool_func = nn.AdaptiveAvgPool3d(self.pool_size)
            if self.spatial_type == 'max':
                self.pool_func = nn.AdaptiveMaxPool3d(self.pool_size)
        else:
            self.pool_size = (self.temporal_size,) + self.spatial_size
            if self.spatial_type == 'avg':
                self.pool_func = nn.AvgPool3d(self.pool_size, stride=1, padding=0)
            if self.spatial_type == 'max':
                self.pool_func = nn.MaxPool3d(self.pool_size, stride=1, padding=0)

    def init_weights(self):
        pass

    def forward(self, input):
        return self.pool_func(input)


class SlowFastSpatialTemporalModule(nn.Module):

    def __init__(self, adaptive_pool=True, spatial_type='avg', spatial_size=1, temporal_size=1):
        super(SlowFastSpatialTemporalModule, self).__init__()
        self.adaptive_pool = adaptive_pool
        assert spatial_type in ['avg']
        self.spatial_type = spatial_type
        self.spatial_size = spatial_size if not isinstance(spatial_size, int) else (spatial_size, spatial_size)
        self.temporal_size = temporal_size
        self.pool_size = (self.temporal_size,) + self.spatial_size
        if self.adaptive_pool:
            if self.spatial_type == 'avg':
                self.op = nn.AdaptiveAvgPool3d(self.pool_size)
        else:
            raise NotImplementedError

    def init_weights(self):
        pass

    def forward(self, input):
        x_slow, x_fast = input
        x_slow = self.op(x_slow)
        x_fast = self.op(x_fast)
        return torch.cat((x_slow, x_fast), dim=1)


class RoIAlignFunction(Function):

    @staticmethod
    def forward(ctx, features, rois, out_size, spatial_scale, sample_num=0):
        if isinstance(out_size, int):
            out_h = out_size
            out_w = out_size
        elif isinstance(out_size, tuple):
            assert len(out_size) == 2
            assert isinstance(out_size[0], int)
            assert isinstance(out_size[1], int)
            out_h, out_w = out_size
        else:
            raise TypeError('"out_size" must be an integer or tuple of integers')
        ctx.spatial_scale = spatial_scale
        ctx.sample_num = sample_num
        ctx.save_for_backward(rois)
        ctx.feature_size = features.size()
        batch_size, num_channels, data_height, data_width = features.size()
        num_rois = rois.size(0)
        output = features.new_zeros(num_rois, num_channels, out_h, out_w)
        if features.is_cuda:
            roi_align_cuda.forward(features, rois, out_h, out_w, spatial_scale, sample_num, output)
        else:
            raise NotImplementedError
        return output

    @staticmethod
    def backward(ctx, grad_output):
        feature_size = ctx.feature_size
        spatial_scale = ctx.spatial_scale
        sample_num = ctx.sample_num
        rois = ctx.saved_tensors[0]
        assert feature_size is not None and grad_output.is_cuda
        batch_size, num_channels, data_height, data_width = feature_size
        out_w = grad_output.size(3)
        out_h = grad_output.size(2)
        grad_input = grad_rois = None
        if ctx.needs_input_grad[0]:
            grad_input = rois.new_zeros(batch_size, num_channels, data_height, data_width)
            roi_align_cuda.backward(grad_output.contiguous(), rois, out_h, out_w, spatial_scale, sample_num, grad_input)
        return grad_input, grad_rois, None, None, None


class RoIAlign(Module):

    def __init__(self, out_size, spatial_scale, sample_num=0):
        super(RoIAlign, self).__init__()
        self.out_size = out_size
        self.spatial_scale = float(spatial_scale)
        self.sample_num = int(sample_num)

    def forward(self, features, rois):
        return RoIAlignFunction.apply(features, rois, self.out_size, self.spatial_scale, self.sample_num)


class RoIPoolFunction(Function):

    @staticmethod
    def forward(ctx, features, rois, out_size, spatial_scale):
        if isinstance(out_size, int):
            out_h = out_size
            out_w = out_size
        elif isinstance(out_size, tuple):
            assert len(out_size) == 2
            assert isinstance(out_size[0], int)
            assert isinstance(out_size[1], int)
            out_h, out_w = out_size
        else:
            raise TypeError('"out_size" must be an integer or tuple of integers')
        assert features.is_cuda
        ctx.save_for_backward(rois)
        num_channels = features.size(1)
        num_rois = rois.size(0)
        out_size = num_rois, num_channels, out_h, out_w
        output = features.new_zeros(out_size)
        argmax = features.new_zeros(out_size, dtype=torch.int)
        roi_pool_cuda.forward(features, rois, out_h, out_w, spatial_scale, output, argmax)
        ctx.spatial_scale = spatial_scale
        ctx.feature_size = features.size()
        ctx.argmax = argmax
        return output

    @staticmethod
    def backward(ctx, grad_output):
        assert grad_output.is_cuda
        spatial_scale = ctx.spatial_scale
        feature_size = ctx.feature_size
        argmax = ctx.argmax
        rois = ctx.saved_tensors[0]
        assert feature_size is not None
        grad_input = grad_rois = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.new_zeros(feature_size)
            roi_pool_cuda.backward(grad_output.contiguous(), rois, argmax, spatial_scale, grad_input)
        return grad_input, grad_rois, None, None


roi_pool = RoIPoolFunction.apply


class RoIPool(Module):

    def __init__(self, out_size, spatial_scale):
        super(RoIPool, self).__init__()
        self.out_size = out_size
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        return roi_pool(features, rois, self.out_size, self.spatial_scale)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AnchorHead,
     lambda: ([], {'num_classes': 4, 'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 256, 64, 64])], {}),
     False),
    (BBoxHead,
     lambda: ([], {}),
     lambda: ([torch.rand([12544, 12544])], {}),
     False),
    (BNInception,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (BaseDetector,
     lambda: ([], {}),
     lambda: ([torch.rand([4]), torch.rand([4, 4])], {}),
     False),
    (RPNHead,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 64, 64])], {}),
     False),
    (STPPReorganized,
     lambda: ([], {'feat_dim': 4, 'act_score_len': 4, 'comp_score_len': 4, 'reg_score_len': 4}),
     lambda: ([torch.rand([0, 4]), torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
    (SimpleSpatialModule,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     True),
]

class Test_open_mmlab_mmaction(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

    def test_004(self):
        self._check(*TESTCASES[4])

    def test_005(self):
        self._check(*TESTCASES[5])

    def test_006(self):
        self._check(*TESTCASES[6])

