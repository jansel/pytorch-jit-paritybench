import sys
_module = sys.modules[__name__]
del sys
auto_assign = _module
config = _module
net = _module

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


import logging


import math


from typing import List


import torch


import torch.distributed as dist


import torch.nn.functional as F


from torch import nn


def negative_bag_loss(logits, gamma):
    return logits ** gamma * F.binary_cross_entropy(logits, torch.zeros_like(logits), reduction='none')


def normal_distribution(x, mu=0, sigma=1):
    return (-(x - mu) ** 2 / (2 * sigma ** 2)).exp()


def normalize(x):
    return (x - x.min() + 1e-12) / (x.max() - x.min() + 1e-12)


def positive_bag_loss(logits, mask, gaussian_probs):
    weight = (3 * logits).exp() * gaussian_probs * mask
    w = weight / weight.sum(dim=1, keepdim=True).clamp(min=1e-12)
    bag_prob = (w * logits).sum(dim=1)
    return F.binary_cross_entropy(bag_prob, torch.ones_like(bag_prob), reduction='none')


class AutoAssign(nn.Module):
    """
    Implement AutoAssign (https://arxiv.org/abs/2007.03496).
    """

    def __init__(self, cfg):
        super(AutoAssign, self).__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.num_classes = cfg.MODEL.FCOS.NUM_CLASSES
        self.in_features = cfg.MODEL.FCOS.IN_FEATURES
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.focal_loss_alpha = cfg.MODEL.FCOS.FOCAL_LOSS_ALPHA
        self.focal_loss_gamma = cfg.MODEL.FCOS.FOCAL_LOSS_GAMMA
        self.iou_loss_type = cfg.MODEL.FCOS.IOU_LOSS_TYPE
        self.reg_weight = cfg.MODEL.FCOS.REG_WEIGHT
        self.score_threshold = cfg.MODEL.FCOS.SCORE_THRESH_TEST
        self.topk_candidates = cfg.MODEL.FCOS.TOPK_CANDIDATES_TEST
        self.nms_threshold = cfg.MODEL.FCOS.NMS_THRESH_TEST
        self.max_detections_per_image = cfg.TEST.DETECTIONS_PER_IMAGE
        self.backbone = cfg.build_backbone(cfg, input_shape=ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN)))
        backbone_shape = self.backbone.output_shape()
        feature_shapes = [backbone_shape[f] for f in self.in_features]
        self.head = AutoAssignHead(cfg, feature_shapes)
        self.shift_generator = cfg.build_shift_generator(cfg, feature_shapes)
        self.shift2box_transform = Shift2BoxTransform(weights=cfg.MODEL.FCOS.BBOX_REG_WEIGHTS)
        self.mu = nn.Parameter(torch.zeros(80, 2))
        self.sigma = nn.Parameter(torch.ones(80, 2))
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        images = self.preprocess_image(batched_inputs)
        if 'instances' in batched_inputs[0]:
            gt_instances = [x['instances'] for x in batched_inputs]
        elif 'targets' in batched_inputs[0]:
            log_first_n(logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10)
            gt_instances = [x['targets'] for x in batched_inputs]
        else:
            gt_instances = None
        features = self.backbone(images.tensor)
        features = [features[f] for f in self.in_features]
        box_cls, box_delta, box_center = self.head(features)
        shifts = self.shift_generator(features)
        if self.training:
            return self.losses(shifts, gt_instances, box_cls, box_delta, box_center)
        else:
            results = self.inference(box_cls, box_delta, box_center, shifts, images)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                height = input_per_image.get('height', image_size[0])
                width = input_per_image.get('width', image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({'instances': r})
            return processed_results

    def losses(self, shifts, gt_instances, box_cls, box_delta, box_center):
        box_cls_flattened = [permute_to_N_HWA_K(x, self.num_classes) for x in box_cls]
        box_delta_flattened = [permute_to_N_HWA_K(x, 4) for x in box_delta]
        box_center_flattened = [permute_to_N_HWA_K(x, 1) for x in box_center]
        pred_class_logits = cat(box_cls_flattened, dim=1)
        pred_shift_deltas = cat(box_delta_flattened, dim=1)
        pred_obj_logits = cat(box_center_flattened, dim=1)
        pred_class_probs = pred_class_logits.sigmoid()
        pred_obj_probs = pred_obj_logits.sigmoid()
        pred_box_probs = []
        num_foreground = pred_class_logits.new_zeros(1)
        num_background = pred_class_logits.new_zeros(1)
        positive_losses = []
        gaussian_norm_losses = []
        for shifts_per_image, gt_instances_per_image, pred_class_probs_per_image, pred_shift_deltas_per_image, pred_obj_probs_per_image in zip(shifts, gt_instances, pred_class_probs, pred_shift_deltas, pred_obj_probs):
            locations = torch.cat(shifts_per_image, dim=0)
            labels = gt_instances_per_image.gt_classes
            gt_boxes = gt_instances_per_image.gt_boxes
            target_shift_deltas = self.shift2box_transform.get_deltas(locations, gt_boxes.tensor.unsqueeze(1))
            is_in_boxes = target_shift_deltas.min(dim=-1).values > 0
            foreground_idxs = torch.nonzero(is_in_boxes, as_tuple=True)
            with torch.no_grad():
                predicted_boxes_per_image = self.shift2box_transform.apply_deltas(pred_shift_deltas_per_image, locations)
                gt_pred_iou = pairwise_iou(gt_boxes, Boxes(predicted_boxes_per_image)).max(dim=0, keepdim=True).values.repeat(len(gt_instances_per_image), 1)
                pred_box_prob_per_image = torch.zeros_like(pred_class_probs_per_image)
                box_prob = 1 / (1 - gt_pred_iou[foreground_idxs]).clamp_(1e-12)
                for i in range(len(gt_instances_per_image)):
                    idxs = foreground_idxs[0] == i
                    if idxs.sum() > 0:
                        box_prob[idxs] = normalize(box_prob[idxs])
                pred_box_prob_per_image[foreground_idxs[1], labels[foreground_idxs[0]]] = box_prob
                pred_box_probs.append(pred_box_prob_per_image)
            normal_probs = []
            for stride, shifts_i in zip(self.fpn_strides, shifts_per_image):
                gt_shift_deltas = self.shift2box_transform.get_deltas(shifts_i, gt_boxes.tensor.unsqueeze(1))
                distances = (gt_shift_deltas[..., :2] - gt_shift_deltas[..., 2:]) / 2
                normal_probs.append(normal_distribution(distances / stride, self.mu[labels].unsqueeze(1), self.sigma[labels].unsqueeze(1)))
            normal_probs = torch.cat(normal_probs, dim=1).prod(dim=-1)
            composed_cls_prob = pred_class_probs_per_image[:, labels] * pred_obj_probs_per_image
            loss_box_reg = iou_loss(pred_shift_deltas_per_image.unsqueeze(0), target_shift_deltas, box_mode='ltrb', loss_type=self.iou_loss_type, reduction='none') * self.reg_weight
            pred_reg_probs = (-loss_box_reg).exp()
            positive_losses.append(positive_bag_loss(composed_cls_prob.permute(1, 0) * pred_reg_probs, is_in_boxes.float(), normal_probs))
            num_foreground += len(gt_instances_per_image)
            num_background += normal_probs[foreground_idxs].sum().item()
            gaussian_norm_losses.append(len(gt_instances_per_image) / normal_probs[foreground_idxs].sum().clamp_(1e-12))
        if dist.is_initialized():
            dist.all_reduce(num_foreground)
            num_foreground /= dist.get_world_size()
            dist.all_reduce(num_background)
            num_background /= dist.get_world_size()
        positive_loss = torch.cat(positive_losses).sum() / max(1, num_foreground)
        pred_box_probs = torch.stack(pred_box_probs, dim=0)
        negative_loss = negative_bag_loss(pred_class_probs * pred_obj_probs * (1 - pred_box_probs), self.focal_loss_gamma).sum() / max(1, num_background)
        loss_pos = positive_loss * self.focal_loss_alpha
        loss_neg = negative_loss * (1 - self.focal_loss_alpha)
        loss_norm = torch.stack(gaussian_norm_losses).mean() * (1 - self.focal_loss_alpha)
        return {'loss_pos': loss_pos, 'loss_neg': loss_neg, 'loss_norm': loss_norm}

    def inference(self, box_cls, box_delta, box_center, shifts, images):
        """
        Arguments:
            box_cls, box_delta, box_center: Same as the output of :meth:`AutoAssignHead.forward`
            shifts (list[list[Tensor]): a list of #images elements. Each is a
                list of #feature level tensor. The tensor contain shifts of this
                image on the specific feature level.
            images (ImageList): the input images

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(shifts) == len(images)
        results = []
        box_cls = [permute_to_N_HWA_K(x, self.num_classes) for x in box_cls]
        box_delta = [permute_to_N_HWA_K(x, 4) for x in box_delta]
        box_center = [permute_to_N_HWA_K(x, 1) for x in box_center]
        for img_idx, shifts_per_image in enumerate(shifts):
            image_size = images.image_sizes[img_idx]
            box_cls_per_image = [box_cls_per_level[img_idx] for box_cls_per_level in box_cls]
            box_reg_per_image = [box_reg_per_level[img_idx] for box_reg_per_level in box_delta]
            box_ctr_per_image = [box_ctr_per_level[img_idx] for box_ctr_per_level in box_center]
            results_per_image = self.inference_single_image(box_cls_per_image, box_reg_per_image, box_ctr_per_image, shifts_per_image, tuple(image_size))
            results.append(results_per_image)
        return results

    def inference_single_image(self, box_cls, box_delta, box_center, shifts, image_size):
        """
        Single-image inference. Return bounding-box detection results by thresholding
        on scores and applying non-maximum suppression (NMS).

        Arguments:
            box_cls (list[Tensor]): list of #feature levels. Each entry contains
                tensor of size (H x W, K)
            box_delta (list[Tensor]): Same shape as 'box_cls' except that K becomes 4.
            box_center (list[Tensor]): Same shape as 'box_cls' except that K becomes 1.
            shifts (list[Tensor]): list of #feature levels. Each entry contains
                a tensor, which contains all the shifts for that
                image in that feature level.
            image_size (tuple(H, W)): a tuple of the image height and width.

        Returns:
            Same as `inference`, but for only one image.
        """
        boxes_all = []
        scores_all = []
        class_idxs_all = []
        for box_cls_i, box_reg_i, box_ctr_i, shifts_i in zip(box_cls, box_delta, box_center, shifts):
            box_cls_i = (box_cls_i.sigmoid_() * box_ctr_i.sigmoid_()).flatten()
            num_topk = min(self.topk_candidates, box_reg_i.size(0))
            predicted_prob, topk_idxs = box_cls_i.sort(descending=True)
            predicted_prob = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[:num_topk]
            keep_idxs = predicted_prob > self.score_threshold
            predicted_prob = predicted_prob[keep_idxs]
            topk_idxs = topk_idxs[keep_idxs]
            shift_idxs = topk_idxs // self.num_classes
            classes_idxs = topk_idxs % self.num_classes
            box_reg_i = box_reg_i[shift_idxs]
            shifts_i = shifts_i[shift_idxs]
            predicted_boxes = self.shift2box_transform.apply_deltas(box_reg_i, shifts_i)
            boxes_all.append(predicted_boxes)
            scores_all.append(predicted_prob)
            class_idxs_all.append(classes_idxs)
        boxes_all, scores_all, class_idxs_all = [cat(x) for x in [boxes_all, scores_all, class_idxs_all]]
        keep = batched_nms(boxes_all, scores_all, class_idxs_all, self.nms_threshold)
        keep = keep[:self.max_detections_per_image]
        result = Instances(image_size)
        result.pred_boxes = Boxes(boxes_all[keep])
        result.scores = scores_all[keep]
        result.pred_classes = class_idxs_all[keep]
        return result

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x['image'] for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

