import sys
_module = sys.modules[__name__]
del sys
calibrator = _module
demo = _module
demo_MLSD_flask = _module
mlsd_pytorch = _module
default = _module
data = _module
utils = _module
wireframe_dset = _module
learner = _module
loss = _module
_func = _module
mlsd_multi_loss = _module
metric = _module
models = _module
build_model = _module
layers = _module
mbv2_mlsd = _module
mbv2_mlsd_large = _module
optim = _module
lr_scheduler = _module
pred_and_eval_sAP = _module
tf_pred_and_eval_sAP = _module
train = _module
comm = _module
decode = _module
logger = _module
meter = _module
mbv2_mlsd_large = _module
mbv2_mlsd_tiny = _module
trt_converter = _module
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


import numpy as np


import torch


from torchvision.datasets import ImageFolder


import uuid


import time


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


import torch.nn as nn


from torch.nn import functional as F


import random


from torch.optim.optimizer import Optimizer


import math


import torch.nn.functional as F


import torch.utils.model_zoo as model_zoo


def deccode_lines_TP(tpMap, score_thresh=0.1, len_thresh=2, topk_n=1000, ksize=3):
    """
    tpMap:
        center: tpMap[1, 0, :, :]
        displacement: tpMap[1, 1:5, :, :]
    """
    b, c, h, w = tpMap.shape
    assert b == 1, 'only support bsize==1'
    displacement = tpMap[:, 1:5, :, :]
    center = tpMap[:, 0, :, :]
    heat = torch.sigmoid(center)
    hmax = F.max_pool2d(heat, (ksize, ksize), stride=1, padding=(ksize - 1) // 2)
    keep = (hmax == heat).float()
    heat = heat * keep
    heat = heat.reshape(-1)
    scores, indices = torch.topk(heat, topk_n, dim=-1, largest=True)
    valid_inx = torch.where(scores > score_thresh)
    scores = scores[valid_inx]
    indices = indices[valid_inx]
    yy = torch.floor_divide(indices, w).unsqueeze(-1)
    xx = torch.fmod(indices, w).unsqueeze(-1)
    center_ptss = torch.cat((xx, yy), dim=-1)
    start_point = center_ptss + displacement[0, :2, yy, xx].reshape(2, -1).permute(1, 0)
    end_point = center_ptss + displacement[0, 2:, yy, xx].reshape(2, -1).permute(1, 0)
    lines = torch.cat((start_point, end_point), dim=-1)
    lines_swap = torch.cat((end_point, start_point), dim=-1)
    all_lens = (end_point - start_point) ** 2
    all_lens = all_lens.sum(dim=-1)
    all_lens = torch.sqrt(all_lens)
    valid_inx = torch.where(all_lens > len_thresh)
    center_ptss = center_ptss[valid_inx]
    lines = lines[valid_inx]
    lines_swap = lines_swap[valid_inx]
    scores = scores[valid_inx]
    return center_ptss, lines, lines_swap, scores


def displacement_loss_func(pred_dis, gt_dis, gt_center_mask=None):
    x0 = gt_dis[:, 0, :, :]
    y0 = gt_dis[:, 1, :, :]
    x1 = gt_dis[:, 2, :, :]
    y1 = gt_dis[:, 3, :, :]
    pos_v = x0.abs() + y0.abs() + x1.abs() + y1.abs()
    pos_mask = torch.where(pos_v != 0, torch.ones_like(x0), torch.zeros_like(x0))
    pos_mask_sum = pos_mask.sum()
    pos_mask = pos_mask.unsqueeze(1)
    pred_dis = pred_dis * pos_mask
    gt_dis = gt_dis * pos_mask
    displacement_loss1 = F.smooth_l1_loss(pred_dis, gt_dis, reduction='none').sum(axis=[1])
    pred_dis2 = torch.cat((pred_dis[:, 2:, :, :], pred_dis[:, :2, :, :]), dim=1)
    displacement_loss2 = F.smooth_l1_loss(pred_dis2, gt_dis, reduction='none').sum(axis=[1])
    displacement_loss = displacement_loss1.min(displacement_loss2)
    displacement_loss = displacement_loss.sum() / pos_mask_sum
    return displacement_loss


def focal_neg_loss_with_logits(preds, gt, alpha=2, belta=4):
    """
    borrow from https://github.com/princeton-vl/CornerNet
    """
    preds = torch.sigmoid(preds)
    pos_inds = gt.eq(1)
    neg_inds = gt.lt(1)
    neg_weights = torch.pow(1 - gt[neg_inds], belta)
    loss = 0
    pos_pred = preds[pos_inds]
    neg_pred = preds[neg_inds]
    pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, alpha)
    neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, alpha) * neg_weights
    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    if pos_pred.nelement() == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


def len_and_angle_loss_func(pred_len, pred_angle, gt_len, gt_angle):
    pred_len = torch.sigmoid(pred_len)
    pred_angle = torch.sigmoid(pred_angle)
    pos_mask = torch.where(gt_len != 0, torch.ones_like(gt_len), torch.zeros_like(gt_len))
    pos_mask_sum = pos_mask.sum()
    len_loss = F.smooth_l1_loss(pred_len, gt_len, reduction='none')
    len_loss = len_loss * pos_mask
    len_loss = len_loss.sum() / pos_mask_sum
    angle_loss = F.smooth_l1_loss(pred_angle, gt_angle, reduction='none')
    angle_loss = angle_loss * pos_mask
    angle_loss = angle_loss.sum() / pos_mask_sum
    return len_loss, angle_loss


def weighted_bce_with_logits(out, gt, pos_w=1.0, neg_w=30.0):
    pos_mask = torch.where(gt != 0, torch.ones_like(gt), torch.zeros_like(gt))
    neg_mask = torch.ones_like(pos_mask) - pos_mask
    loss = F.binary_cross_entropy_with_logits(out, gt, reduction='none')
    loss_pos = (loss * pos_mask).sum() / torch.sum(pos_mask)
    loss_neg = (loss * neg_mask).sum() / torch.sum(neg_mask)
    loss = loss_pos * pos_w + loss_neg * neg_w
    return loss


class LineSegmentLoss(nn.Module):

    def __init__(self, cfg):
        super(LineSegmentLoss, self).__init__()
        self.input_size = cfg.datasets.input_size
        self.with_SOL_loss = cfg.loss.with_sol_loss
        self.with_match_loss = cfg.loss.with_match_loss
        self.with_focal_loss = cfg.loss.with_focal_loss
        self.focal_loss_level = cfg.loss.focal_loss_level
        self.match_sap_thresh = cfg.loss.match_sap_thresh
        self.decode_score_thresh = cfg.decode.score_thresh
        self.decode_len_thresh = cfg.decode.len_thresh
        self.decode_top_k = cfg.decode.top_k
        self.loss_w_dict = {'tp_center_loss': 10.0, 'tp_displacement_loss': 1.0, 'tp_len_loss': 1.0, 'tp_angle_loss': 1.0, 'tp_match_loss': 1.0, 'tp_centerness_loss': 1.0, 'sol_center_loss': 1.0, 'sol_displacement_loss': 1.0, 'sol_len_loss': 1.0, 'sol_angle_loss': 1.0, 'sol_match_loss': 1.0, 'sol_centerness_loss': 1.0, 'line_seg_loss': 1.0, 'junc_seg_loss': 1.0}
        if len(cfg.loss.loss_weight_dict_list) > 0:
            self.loss_w_dict.update(cfg.loss.loss_weight_dict_list[0])
        None

    def _m_gt_matched_n(self, p_lines, gt_lines, thresh):
        gt_lines = gt_lines
        distance1 = torch.cdist(gt_lines[:, :2], p_lines[:, :2], p=2)
        distance2 = torch.cdist(gt_lines[:, 2:], p_lines[:, 2:], p=2)
        distance = distance1 + distance2
        near_inx = torch.argsort(distance, 1)[:, 0]
        matched_pred_lines = p_lines[near_inx]
        distance1 = F.pairwise_distance(gt_lines[:, :2], matched_pred_lines[:, :2], p=2)
        distance2 = F.pairwise_distance(gt_lines[:, 2:], matched_pred_lines[:, 2:], p=2)
        inx = torch.where((distance1 < thresh) & (distance2 < thresh))[0]
        return len(inx)

    def _m_match_loss_fn(self, p_lines, p_centers, p_scores, gt_lines, thresh):
        gt_lines = gt_lines
        distance1 = torch.cdist(p_lines[:, :2], gt_lines[:, :2], p=2)
        distance2 = torch.cdist(p_lines[:, 2:], gt_lines[:, 2:], p=2)
        distance = distance1 + distance2
        near_inx = torch.argsort(distance, 1)[:, 0]
        matched_gt_lines = gt_lines[near_inx]
        distance1 = F.pairwise_distance(matched_gt_lines[:, :2], p_lines[:, :2], p=2)
        distance2 = F.pairwise_distance(matched_gt_lines[:, 2:], p_lines[:, 2:], p=2)
        inx = torch.where((distance1 < thresh) & (distance2 < thresh))[0]
        match_n = len(inx)
        loss = 4 * thresh
        if match_n > 0:
            mathed_gt_lines = matched_gt_lines[inx]
            mathed_pred_lines = p_lines[inx]
            mathed_pred_centers = p_centers[inx]
            endpoint_loss = F.l1_loss(mathed_pred_lines, mathed_gt_lines, reduction='mean')
            gt_centers = (mathed_gt_lines[:, :2] + mathed_gt_lines[:, 2:]) / 2
            center_dis_loss = F.l1_loss(mathed_pred_centers, gt_centers, reduction='mean')
            loss = 1.0 * endpoint_loss + 1.0 * center_dis_loss
        return loss, match_n

    def matching_loss_func(self, pred_tp_mask, gt_line_512_tensor_list):
        match_loss_all = 0.0
        match_ratio_all = 0.0
        for pred, gt_line_512 in zip(pred_tp_mask, gt_line_512_tensor_list):
            gt_line_128 = gt_line_512 / 4
            n_gt = gt_line_128.shape[0]
            pred_center_ptss, pred_lines, pred_lines_swap, pred_scores = deccode_lines_TP(pred.unsqueeze(0), score_thresh=self.decode_score_thresh, len_thresh=self.decode_len_thresh, topk_n=self.decode_top_k, ksize=3)
            n_pred = pred_center_ptss.shape[0]
            if n_pred == 0:
                match_loss_all += 4 * self.match_sap_thresh
                match_ratio_all += 0.0
                continue
            pred_lines_128 = 128 * pred_lines / (self.input_size / 2)
            pred_lines_128_swap = 128 * pred_lines_swap / (self.input_size / 2)
            pred_center_ptss_128 = 128 * pred_center_ptss / (self.input_size / 2)
            pred_lines_128 = torch.cat((pred_lines_128, pred_lines_128_swap), dim=0)
            pred_center_ptss_128 = torch.cat((pred_center_ptss_128, pred_center_ptss_128), dim=0)
            pred_scores = torch.cat((pred_scores, pred_scores), dim=0)
            mloss, match_n_pred = self._m_match_loss_fn(pred_lines_128, pred_center_ptss_128, pred_scores, gt_line_128, self.match_sap_thresh)
            match_n = self._m_gt_matched_n(pred_lines_128, gt_line_128, self.match_sap_thresh)
            match_ratio = match_n / n_gt
            match_loss_all += mloss
            match_ratio_all += match_ratio
        return match_loss_all / pred_tp_mask.shape[0], match_ratio_all / pred_tp_mask.shape[0]

    def tp_mask_loss(self, out, gt, gt_lines_tensor_512_list):
        out_center = out[:, 7, :, :]
        gt_center = gt[:, 7, :, :]
        if self.with_focal_loss:
            center_loss = focal_neg_loss_with_logits(out_center, gt_center)
        else:
            center_loss = weighted_bce_with_logits(out_center, gt_center, 1.0, 30.0)
        out_displacement = out[:, 8:12, :, :]
        gt_displacement = gt[:, 8:12, :, :]
        displacement_loss = displacement_loss_func(out_displacement, gt_displacement, gt_center)
        len_loss, angle_loss = len_and_angle_loss_func(pred_len=out[:, 12, :, :], pred_angle=out[:, 13, :, :], gt_len=gt[:, 12, :, :], gt_angle=gt[:, 13, :, :])
        match_loss, match_ratio = 0, 0
        if self.with_match_loss:
            match_loss, match_ratio = self.matching_loss_func(out[:, 7:12], gt_lines_tensor_512_list)
        return {'tp_center_loss': center_loss, 'tp_displacement_loss': displacement_loss, 'tp_len_loss': len_loss, 'tp_angle_loss': angle_loss, 'tp_match_loss': match_loss, 'tp_match_ratio': match_ratio}

    def sol_mask_loss(self, out, gt, sol_lines_512_all_tensor_list):
        out_center = out[:, 0, :, :]
        gt_center = gt[:, 0, :, :]
        if self.with_focal_loss and self.focal_loss_level >= 1:
            center_loss = focal_neg_loss_with_logits(out_center, gt_center)
        else:
            center_loss = weighted_bce_with_logits(out_center, gt_center, 1.0, 30.0)
        out_displacement = out[:, 1:5, :, :]
        gt_displacement = gt[:, 1:5, :, :]
        displacement_loss = displacement_loss_func(out_displacement, gt_displacement, gt_center)
        len_loss, angle_loss = len_and_angle_loss_func(pred_len=out[:, 5, :, :], pred_angle=out[:, 6, :, :], gt_len=gt[:, 5, :, :], gt_angle=gt[:, 6, :, :])
        match_loss, match_ratio = 0, 0
        if self.with_match_loss:
            match_loss, match_ratio = self.matching_loss_func(out[:, 0:5], sol_lines_512_all_tensor_list)
        return {'sol_center_loss': center_loss, 'sol_displacement_loss': displacement_loss, 'sol_len_loss': len_loss, 'sol_angle_loss': angle_loss, 'sol_match_loss': match_loss}

    def line_and_juc_seg_loss(self, out, gt):
        out_line_seg = out[:, 15, :, :]
        gt_line_seg = gt[:, 15, :, :]
        if self.with_focal_loss and self.focal_loss_level >= 3:
            line_seg_loss = focal_neg_loss_with_logits(out_line_seg, gt_line_seg)
        else:
            line_seg_loss = weighted_bce_with_logits(out_line_seg, gt_line_seg, 1.0, 1.0)
        out_junc_seg = out[:, 14, :, :]
        gt_junc_seg = gt[:, 14, :, :]
        if self.with_focal_loss and self.focal_loss_level >= 2:
            junc_seg_loss = focal_neg_loss_with_logits(out_junc_seg, gt_junc_seg)
        else:
            junc_seg_loss = weighted_bce_with_logits(out_junc_seg, gt_junc_seg, 1.0, 30.0)
        return line_seg_loss, junc_seg_loss

    def forward(self, preds, gts, tp_gt_lines_512_list, sol_gt_lines_512_list):
        line_seg_loss, junc_seg_loss = self.line_and_juc_seg_loss(preds, gts)
        loss_dict = {'line_seg_loss': line_seg_loss, 'junc_seg_loss': junc_seg_loss}
        if self.with_SOL_loss:
            sol_loss_dict = self.sol_mask_loss(preds, gts, sol_gt_lines_512_list)
            loss_dict.update(sol_loss_dict)
        tp_loss_dict = self.tp_mask_loss(preds, gts, tp_gt_lines_512_list)
        loss_dict.update(tp_loss_dict)
        loss = 0.0
        for k, v in loss_dict.items():
            if not self.with_SOL_loss and 'sol_' in k:
                continue
            if k in self.loss_w_dict.keys():
                v = v * self.loss_w_dict[k]
                loss_dict[k] = v
                loss += v
        loss_dict['loss'] = loss
        if self.with_SOL_loss:
            loss_dict['center_loss'] = loss_dict['sol_center_loss'] + loss_dict['tp_center_loss']
            loss_dict['displacement_loss'] = loss_dict['sol_displacement_loss'] + loss_dict['tp_displacement_loss']
            loss_dict['match_loss'] = loss_dict['tp_match_loss'] + loss_dict['sol_match_loss']
            loss_dict['match_ratio'] = loss_dict['tp_match_ratio']
        else:
            loss_dict['center_loss'] = loss_dict['tp_center_loss']
            loss_dict['displacement_loss'] = loss_dict['tp_displacement_loss']
            loss_dict['match_loss'] = loss_dict['tp_match_loss']
            loss_dict['match_ratio'] = loss_dict['tp_match_ratio']
        return loss_dict


class BlockTypeA(nn.Module):

    def __init__(self, in_c1, in_c2, out_c1, out_c2, upscale=True):
        super(BlockTypeA, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_c2, out_c2, kernel_size=1), nn.BatchNorm2d(out_c2), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_c1, out_c1, kernel_size=1), nn.BatchNorm2d(out_c1), nn.ReLU(inplace=True))
        self.upscale = upscale

    def forward(self, a, b):
        b = self.conv1(b)
        a = self.conv2(a)
        b = F.interpolate(b, scale_factor=2.0, mode='bilinear', align_corners=True)
        return torch.cat((a, b), dim=1)


class BlockTypeB(nn.Module):

    def __init__(self, in_c, out_c):
        super(BlockTypeB, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_c, in_c, kernel_size=3, padding=1), nn.BatchNorm2d(in_c), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(in_c, out_c, kernel_size=3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU())

    def forward(self, x):
        x = self.conv1(x) + x
        x = self.conv2(x)
        return x


class BlockTypeC(nn.Module):

    def __init__(self, in_c, out_c):
        super(BlockTypeC, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_c, in_c, kernel_size=3, padding=5, dilation=5), nn.BatchNorm2d(in_c), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(in_c, in_c, kernel_size=3, padding=1), nn.BatchNorm2d(in_c), nn.ReLU())
        self.conv3 = nn.Conv2d(in_c, out_c, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class ConvBNReLU(nn.Sequential):

    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        self.channel_pad = out_planes - in_planes
        self.stride = stride
        if stride == 2:
            padding = 0
        else:
            padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False), nn.BatchNorm2d(out_planes), nn.ReLU6(inplace=True))
        self.max_pool = nn.MaxPool2d(kernel_size=stride, stride=stride)

    def forward(self, x):
        if self.stride == 2:
            x = F.pad(x, (0, 1, 0, 1), 'constant', 0)
        for module in self:
            if not isinstance(module, nn.MaxPool2d):
                x = module(x)
        return x


class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup
        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim), nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup)])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class MobileNetV2(nn.Module):

    def __init__(self, pretrained=True):
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
        """
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        width_mult = 1.0
        round_nearest = 8
        inverted_residual_setting = [[1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2], [6, 64, 4, 2]]
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError('inverted_residual_setting should be non-empty or a 4-element list, got {}'.format(inverted_residual_setting))
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(4, input_channel, stride=2)]
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        self.features = nn.Sequential(*features)
        self.fpn_selected = [3, 6, 10]
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x):
        fpn_features = []
        for i, f in enumerate(self.features):
            if i > self.fpn_selected[-1]:
                break
            x = f(x)
            if i in self.fpn_selected:
                fpn_features.append(x)
        c2, c3, c4 = fpn_features
        return c2, c3, c4

    def forward(self, x):
        return self._forward_impl(x)

    def _load_pretrained_model(self):
        pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/mobilenet_v2-b0353104.pth')
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)


class BilinearConvTranspose2d(nn.ConvTranspose2d):
    """A conv transpose initialized to bilinear interpolation."""

    def __init__(self, channels, stride, groups=1):
        """Set up the layer.

        Parameters
        ----------
        channels: int
            The number of input and output channels

        stride: int or tuple
            The amount of upsampling to do

        groups: int
            Set to 1 for a standard convolution. Set equal to channels to
            make sure there is no cross-talk between channels.
        """
        if isinstance(stride, int):
            stride = stride, stride
        assert groups in (1, channels), 'Must use no grouping, ' + 'or one group per channel'
        kernel_size = 2 * stride[0] - 1, 2 * stride[1] - 1
        padding = stride[0] - 1, stride[1] - 1
        super().__init__(channels, channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=padding, groups=groups)

    def reset_parameters(self):
        """Reset the weight and bias."""
        nn.init.constant(self.bias, 0)
        nn.init.constant(self.weight, 0)
        bilinear_kernel = self.bilinear_kernel(self.stride)
        for i in range(self.in_channels):
            if self.groups == 1:
                j = i
            else:
                j = 0
            self.weight.data[i, j] = bilinear_kernel

    @staticmethod
    def bilinear_kernel(stride):
        """Generate a bilinear upsampling kernel."""
        num_dims = len(stride)
        shape = (1,) * num_dims
        bilinear_kernel = torch.ones(*shape)
        for channel in range(num_dims):
            channel_stride = stride[channel]
            kernel_size = 2 * channel_stride - 1
            delta = torch.arange(1 - channel_stride, channel_stride)
            channel_filter = 1 - torch.abs(delta / channel_stride)
            shape = [1] * num_dims
            shape[channel] = kernel_size
            bilinear_kernel = bilinear_kernel * channel_filter.view(shape)
        return bilinear_kernel


class MobileV2_MLSD(nn.Module):

    def __init__(self, cfg):
        super(MobileV2_MLSD, self).__init__()
        self.backbone = MobileNetV2(pretrained=True)
        self.block12 = BlockTypeA(in_c1=32, in_c2=64, out_c1=64, out_c2=64)
        self.block13 = BlockTypeB(128, 64)
        self.block14 = BlockTypeA(in_c1=24, in_c2=64, out_c1=32, out_c2=32)
        self.block15 = BlockTypeB(64, 64)
        self.block16 = BlockTypeC(64, 16)
        self.with_deconv = cfg.model.with_deconv
        if self.with_deconv:
            self.block17 = BilinearConvTranspose2d(16, 2, 1)
            self.block17.reset_parameters()

    def forward(self, x):
        c1, c2, c3, c4 = self.backbone(x)
        x = self.block12(c3, c4)
        x = self.block13(x)
        x = self.block14(c2, x)
        x = self.block15(x)
        x = self.block16(x)
        if self.with_deconv:
            x = self.block17(x)
        else:
            x = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=True)
        return x


class MobileV2_MLSD_Large(nn.Module):

    def __init__(self):
        super(MobileV2_MLSD_Large, self).__init__()
        self.backbone = MobileNetV2(pretrained=False)
        self.block15 = BlockTypeA(in_c1=64, in_c2=96, out_c1=64, out_c2=64, upscale=False)
        self.block16 = BlockTypeB(128, 64)
        self.block17 = BlockTypeA(in_c1=32, in_c2=64, out_c1=64, out_c2=64)
        self.block18 = BlockTypeB(128, 64)
        self.block19 = BlockTypeA(in_c1=24, in_c2=64, out_c1=64, out_c2=64)
        self.block20 = BlockTypeB(128, 64)
        self.block21 = BlockTypeA(in_c1=16, in_c2=64, out_c1=64, out_c2=64)
        self.block22 = BlockTypeB(128, 64)
        self.block23 = BlockTypeC(64, 16)

    def forward(self, x):
        c1, c2, c3, c4, c5 = self.backbone(x)
        x = self.block15(c4, c5)
        x = self.block16(x)
        x = self.block17(c3, x)
        x = self.block18(x)
        x = self.block19(c2, x)
        x = self.block20(x)
        x = self.block21(c1, x)
        x = self.block22(x)
        x = self.block23(x)
        x = x[:, 7:, :, :]
        return x


class MobileV2_MLSD_Tiny(nn.Module):

    def __init__(self):
        super(MobileV2_MLSD_Tiny, self).__init__()
        self.backbone = MobileNetV2(pretrained=True)
        self.block12 = BlockTypeA(in_c1=32, in_c2=64, out_c1=64, out_c2=64)
        self.block13 = BlockTypeB(128, 64)
        self.block14 = BlockTypeA(in_c1=24, in_c2=64, out_c1=32, out_c2=32)
        self.block15 = BlockTypeB(64, 64)
        self.block16 = BlockTypeC(64, 16)

    def forward(self, x):
        c2, c3, c4 = self.backbone(x)
        x = self.block12(c3, c4)
        x = self.block13(x)
        x = self.block14(c2, x)
        x = self.block15(x)
        x = self.block16(x)
        x = x[:, 7:, :, :]
        x = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=True)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BilinearConvTranspose2d,
     lambda: ([], {'channels': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BlockTypeB,
     lambda: ([], {'in_c': 4, 'out_c': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BlockTypeC,
     lambda: ([], {'in_c': 4, 'out_c': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvBNReLU,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (InvertedResidual,
     lambda: ([], {'inp': 4, 'oup': 4, 'stride': 1, 'expand_ratio': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MobileNetV2,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     False),
    (MobileV2_MLSD_Tiny,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     False),
]

class Test_lhwcv_mlsd_pytorch(_paritybench_base):
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

