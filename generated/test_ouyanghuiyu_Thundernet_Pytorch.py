import sys
_module = sys.modules[__name__]
del sys
_init_paths = _module
demo = _module
lib = _module
datasets = _module
coco = _module
ds_utils = _module
factory = _module
imagenet = _module
imdb = _module
pascal_voc = _module
pascal_voc_rbg = _module
mcg_munge = _module
vg = _module
vg_eval = _module
voc_eval = _module
external = _module
setup = _module
Snet = _module
faster_rcnn = _module
faster_rcnn = _module
modules = _module
loss = _module
losses = _module
rpn = _module
anchor_target_layer = _module
bbox_transform = _module
centernet_rpn = _module
generate_anchors = _module
proposal_layer = _module
proposal_target_layer_cascade = _module
rpn = _module
utils = _module
blob = _module
cente_decode = _module
config = _module
layer_utils = _module
logger = _module
net_utils = _module
PSROIAlign = _module
model = _module
example = _module
roi_layers = _module
ps_roi_align = _module
ps_roi_pool = _module
psroialign = _module
pollers = _module
psroialign = _module
roi_data_layer = _module
augmentation = _module
minibatch = _module
roibatchLoader = _module
roidb = _module
onnx = _module
onnx_infer = _module
rcnn_head_to_onnx = _module
rpn_to_onnx = _module
test_net = _module
trainval_net = _module

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


import time


import torch


from torch.autograd import Variable


import torch.nn as nn


import torch.optim as optim


import random


import torch.nn.functional as F


import functools


from torch.nn import functional as F


import math


from torch import nn


import numpy.random as npr


from torch.autograd import Function


from torch.autograd.function import once_differentiable


from torch.utils.data import RandomSampler


from torch.utils.data.sampler import Sampler


def conv_bn(inp, oup, stride):
    return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.
        BatchNorm2d(oup), nn.ReLU6(inplace=True))


class SnetExtractor(nn.Module):
    cfg = {(49): [24, 60, 120, 240, 512], (146): [24, 132, 264, 528], (535):
        [48, 248, 496, 992]}

    def __init__(self, version=146, model_path=None, **kwargs):
        super(SnetExtractor, self).__init__()
        num_layers = [4, 8, 4]
        self.model_path = model_path
        self.num_layers = num_layers
        channels = self.cfg[version]
        self.channels = channels
        self.conv1 = conv_bn(3, channels[0], kernel_size=3, stride=2, pad=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stage1 = self._make_layer(num_layers[0], channels[0], channels
            [1], **kwargs)
        self.stage2 = self._make_layer(num_layers[1], channels[1], channels
            [2], **kwargs)
        self.stage3 = self._make_layer(num_layers[2], channels[2], channels
            [3], **kwargs)
        if len(self.channels) == 5:
            self.conv5 = conv_bn(channels[3], channels[4], kernel_size=1,
                stride=1, pad=0)
        if len(channels) == 5:
            self.cem = CEM(channels[-3], channels[-1], channels[-1], cfg.
                FEAT_STRIDE)
        else:
            self.cem = CEM(channels[-2], channels[-1], channels[-1], cfg.
                FEAT_STRIDE)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self._initialize_weights()

    def _make_layer(self, num_layers, in_channels, out_channels, **kwargs):
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(ShuffleV2Block(in_channels, out_channels,
                    mid_channels=out_channels // 2, ksize=5, stride=2))
            else:
                layers.append(ShuffleV2Block(in_channels // 2, out_channels,
                    mid_channels=out_channels // 2, ksize=5, stride=1))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def _initialize_weights(self):

        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters():
                    p.requires_grad = False
        if self.model_path is not None:
            None
            if torch.cuda.is_available():
                state_dict = torch.load(self.model_path)['state_dict']
            else:
                state_dict = torch.load(self.model_path, map_location=lambda
                    storage, loc: storage)['state_dict']
            keys = []
            for k, v in state_dict.items():
                keys.append(k)
            for k in keys:
                state_dict[k.replace('module.', '')] = state_dict.pop(k)
            self.load_state_dict(state_dict, strict=False)
            for para in self.conv1.parameters():
                para.requires_grad = False
            None
            for para in self.stage1.parameters():
                para.requires_grad = False
            None
            set_bn_fix(self.conv1)
            set_bn_fix(self.stage1)
            set_bn_fix(self.stage2)
            set_bn_fix(self.stage3)
        else:
            for name, m in self.named_modules():
                if isinstance(m, nn.Conv2d):
                    if 'first' in name:
                        nn.init.normal_(m.weight, 0, 0.01)
                    else:
                        nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0001)
                    nn.init.constant_(m.running_mean, 0)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0001)
                    nn.init.constant_(m.running_mean, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        c3 = self.stage1(x)
        c4 = self.stage2(c3)
        c5 = self.stage3(c4)
        if len(self.channels) == 5:
            c5 = self.conv5(c5)
        Cglb_lat = self.avgpool(c5)
        if cfg.FEAT_STRIDE == 16:
            cem_out = self.cem([c4, c5, Cglb_lat])
        elif cfg.FEAT_STRIDE == 8:
            cem_out = self.cem([c3, c4, c5, Cglb_lat])
        return cem_out


def _smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights,
    bbox_outside_weights, sigma=1.0, dim=[1], reduce='mean'):
    sigma_2 = sigma ** 2
    box_diff = bbox_pred - bbox_targets
    in_box_diff = bbox_inside_weights * box_diff
    abs_in_box_diff = torch.abs(in_box_diff)
    smoothL1_sign = (abs_in_box_diff < 1.0 / sigma_2).detach().float()
    in_loss_box = torch.pow(in_box_diff, 2) * (sigma_2 / 2.0
        ) * smoothL1_sign + (abs_in_box_diff - 0.5 / sigma_2) * (1.0 -
        smoothL1_sign)
    out_loss_box = bbox_outside_weights * in_loss_box
    loss_box = out_loss_box
    for i in sorted(dim, reverse=True):
        loss_box = loss_box.sum(i)
    if reduce == 'mean':
        loss_box = loss_box.mean()
    elif reduce == 'sum':
        loss_box = loss_box.sum()
    return loss_box


def hard_negative_mining(loss, labels, neg_pos_ratio=3):
    """
    It used to suppress the presence of a large number of negative prediction.
    It works on image level not batch level.
    For any example/image, it keeps all the positive predictions and
     cut the number of negative predictions to make sure the ratio
     between the negative examples and positive examples is no more
     the given ratio for an image.
    Args:
        loss (N, num_anchors): the loss for each example.
        labels (N, num_anchors): the labels.
        neg_pos_ratio:  the ratio between the negative examples and positive examples.
    """
    pos_mask = labels > 0
    num_pos = pos_mask.long().sum(dim=0, keepdim=True)
    num_neg = num_pos * neg_pos_ratio
    loss[pos_mask] = -math.inf
    _, indexes = loss.sort(descending=True)
    _, orders = indexes.sort()
    neg_mask = orders < num_neg
    return pos_mask | neg_mask, num_pos.cpu().numpy()[0]


_global_config['TRAIN'] = 4


_global_config['FEAT_STRIDE'] = 4


_global_config['POOLING_MODE'] = 4


class _fasterRCNN(nn.Module):
    """ faster RCNN """

    def __init__(self, classes, class_agnostic, compact_mode=False):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0
        self.rpn = RPN(in_channels=245, f_channels=256)
        self.sam = SAM(256, 245)
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.rpn_time = None
        self.pre_roi_time = None
        self.roi_pooling_time = None
        self.subnet_time = None
        self.psroiAlign = PSROIAlignhandle(1.0 / cfg.FEAT_STRIDE, 7, 2, 5)
        self.psroiPool = PSROIPoolhandle(7, 7, 1.0 / cfg.FEAT_STRIDE, 7, 5)

    def _roi_pool_layer(self, bottom, rois):
        return self.psroiPool.forward(bottom, rois)

    def _roi_align_layer(self, bottom, rois):
        return self.psroiAlign.forward(bottom, rois)

    def forward(self, im_data, im_info, gt_boxes, num_boxes):
        batch_size = im_data.size(0)
        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data
        start = time.time()
        basefeat = self.RCNN_base(im_data)
        rpn_feat = self.rpn(basefeat)
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(rpn_feat, im_info,
            gt_boxes, num_boxes)
        rpn_time = time.time()
        self.rpn_time = rpn_time - start
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            (rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws
                ) = roi_data
            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1,
                rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1,
                rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0
        rois = Variable(rois)
        pre_roi_time = time.time()
        self.pre_roi_time = pre_roi_time - rpn_time
        base_feat = self.sam([basefeat, rpn_feat])
        if cfg.POOLING_MODE == 'align':
            pooled_feat = self._roi_align_layer(base_feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self._roi_pool_layer(base_feat, rois.view(-1, 5))
        roi_pool_time = time.time()
        self.roi_pooling_time = roi_pool_time - pre_roi_time
        pooled_feat = self._head_to_tail(pooled_feat)
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        if self.training and not self.class_agnostic:
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(
                bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.
                view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4)
                )
            bbox_pred = bbox_pred_select.squeeze(1)
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, 1)
        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0
        if self.training:
            loss = -F.log_softmax(cls_score, dim=1)[:, (0)]
            mask, num_pos = hard_negative_mining(loss, rois_label)
            confidence = cls_score[(mask), :]
            RCNN_loss_cls = F.cross_entropy(confidence, rois_label[mask],
                reduction='mean')
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target,
                rois_inside_ws, rois_outside_ws)
            RCNN_loss_bbox = RCNN_loss_bbox * 2
        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)
        subnet_time = time.time()
        self.subnet_time = subnet_time - roi_pool_time
        time_measure = [self.rpn_time, self.pre_roi_time, self.
            roi_pooling_time, self.subnet_time]
        return (time_measure, rois, cls_prob, bbox_pred, rpn_loss_cls,
            rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label)

    def _init_weights(self):

        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()


anchor_number = 25


class RPN(nn.Module):
    """region proposal network"""

    def __init__(self, in_channels=245, f_channels=256):
        super(RPN, self).__init__()
        self.num_anchors = anchor_number
        self.dw5_5 = nn.Conv2d(in_channels, in_channels, kernel_size=5,
            stride=1, padding=2, groups=in_channels)
        self.bn0 = nn.BatchNorm2d(in_channels)
        self.relu0 = nn.ReLU(inplace=True)
        self.con1x1 = nn.Conv2d(in_channels, f_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(f_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self._initialize_weights()

    def forward(self, x):
        x = self.dw5_5(x)
        x = self.bn0(x)
        x = self.relu0(x)
        x = self.con1x1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        return x

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class SAM(torch.nn.Module):

    def __init__(self, f_channels, CEM_FILTER):
        super(SAM, self).__init__()
        self.conv1 = nn.Conv2d(f_channels, CEM_FILTER, kernel_size=1)
        self.bn = nn.BatchNorm2d(CEM_FILTER)
        self._initialize_weights()

    def forward(self, input):
        cem = input[0]
        rpn = input[1]
        sam = self.conv1(rpn)
        sam = self.bn(sam)
        sam = F.sigmoid(sam)
        out = cem * sam
        return out

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class ShuffleV2Block(nn.Module):

    def __init__(self, inp, oup, mid_channels, *, ksize, stride):
        super(ShuffleV2Block, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        self.mid_channels = mid_channels
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inp = inp
        outputs = oup - inp
        branch_main = [nn.Conv2d(inp, mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=True), nn.Conv2d(
            mid_channels, mid_channels, ksize, stride, pad, groups=
            mid_channels, bias=False), nn.BatchNorm2d(mid_channels), nn.
            Conv2d(mid_channels, outputs, 1, 1, 0, bias=False), nn.
            BatchNorm2d(outputs), nn.ReLU(inplace=True)]
        self.branch_main = nn.Sequential(*branch_main)
        if stride == 2:
            branch_proj = [nn.Conv2d(inp, inp, ksize, stride, pad, groups=
                inp, bias=False), nn.BatchNorm2d(inp), nn.Conv2d(inp, inp, 
                1, 1, 0, bias=False), nn.BatchNorm2d(inp), nn.ReLU(inplace=
                True)]
            self.branch_proj = nn.Sequential(*branch_proj)
        else:
            self.branch_proj = None

    def forward(self, old_x):
        if self.stride == 1:
            x_proj, x = self.channel_shuffle(old_x)
            return torch.cat((x_proj, self.branch_main(x)), 1)
        elif self.stride == 2:
            x_proj = old_x
            x = old_x
            return torch.cat((self.branch_proj(x_proj), self.branch_main(x)), 1
                )

    def channel_shuffle(self, x):
        g = 2
        x = x.reshape(x.shape[0], g, x.shape[1] // g, x.shape[2], x.shape[3])
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(x.shape[0], -1, x.shape[3], x.shape[4])
        x_proj = x[:, :x.shape[1] // 2, :, :]
        x = x[:, x.shape[1] // 2:, :, :]
        return x_proj, x


def _neg_loss(pred, gt):
    """ Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
      Arguments:
        pred (batch x c x h x w)
        gt_regr (batch x c x h x w)
    """
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()
    neg_weights = torch.pow(1 - gt, 4)
    loss = 0
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2
        ) * neg_weights * neg_inds
    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


class FocalLoss(nn.Module):
    """nn.Module warpper for focal loss"""

    def __init__(self):
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target):
        return self.neg_loss(out, target)


def _reg_loss(regr, gt_regr, mask):
    """ L1 regression loss
      Arguments:
        regr (batch x max_objects x dim)
        gt_regr (batch x max_objects x dim)
        mask (batch x max_objects)
    """
    num = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr).float()
    regr = regr * mask
    gt_regr = gt_regr * mask
    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
    regr_loss = regr_loss / (num + 0.0001)
    return regr_loss


def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


class RegLoss(nn.Module):
    """Regression loss for an output tensor
      Arguments:
        output (batch x dim x h x w)
        mask (batch x max_objects)
        ind (batch x max_objects)
        target (batch x max_objects x dim)
    """

    def __init__(self):
        super(RegLoss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _tranpose_and_gather_feat(output, ind)
        loss = _reg_loss(pred, target, mask)
        return loss


class RegL1Loss(nn.Module):

    def __init__(self):
        super(RegL1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _tranpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 0.0001)
        return loss


class NormRegL1Loss(nn.Module):

    def __init__(self):
        super(NormRegL1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _tranpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        pred = pred / (target + 0.0001)
        target = target * 0 + 1
        loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 0.0001)
        return loss


class RegWeightedL1Loss(nn.Module):

    def __init__(self):
        super(RegWeightedL1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _tranpose_and_gather_feat(output, ind)
        mask = mask.float()
        loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 0.0001)
        return loss


class L1Loss(nn.Module):

    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _tranpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = F.l1_loss(pred * mask, target * mask, reduction=
            'elementwise_mean')
        return loss


def compute_bin_loss(output, target, mask):
    mask = mask.expand_as(output)
    output = output * mask.float()
    return F.cross_entropy(output, target, reduction='elementwise_mean')


def compute_res_loss(output, target):
    return F.smooth_l1_loss(output, target, reduction='elementwise_mean')


def compute_rot_loss(output, target_bin, target_res, mask):
    output = output.view(-1, 8)
    target_bin = target_bin.view(-1, 2)
    target_res = target_res.view(-1, 2)
    mask = mask.view(-1, 1)
    loss_bin1 = compute_bin_loss(output[:, 0:2], target_bin[:, (0)], mask)
    loss_bin2 = compute_bin_loss(output[:, 4:6], target_bin[:, (1)], mask)
    loss_res = torch.zeros_like(loss_bin1)
    if target_bin[:, (0)].nonzero().shape[0] > 0:
        idx1 = target_bin[:, (0)].nonzero()[:, (0)]
        valid_output1 = torch.index_select(output, 0, idx1.long())
        valid_target_res1 = torch.index_select(target_res, 0, idx1.long())
        loss_sin1 = compute_res_loss(valid_output1[:, (2)], torch.sin(
            valid_target_res1[:, (0)]))
        loss_cos1 = compute_res_loss(valid_output1[:, (3)], torch.cos(
            valid_target_res1[:, (0)]))
        loss_res += loss_sin1 + loss_cos1
    if target_bin[:, (1)].nonzero().shape[0] > 0:
        idx2 = target_bin[:, (1)].nonzero()[:, (0)]
        valid_output2 = torch.index_select(output, 0, idx2.long())
        valid_target_res2 = torch.index_select(target_res, 0, idx2.long())
        loss_sin2 = compute_res_loss(valid_output2[:, (6)], torch.sin(
            valid_target_res2[:, (1)]))
        loss_cos2 = compute_res_loss(valid_output2[:, (7)], torch.cos(
            valid_target_res2[:, (1)]))
        loss_res += loss_sin2 + loss_cos2
    return loss_bin1 + loss_bin2 + loss_res


class BinRotLoss(nn.Module):

    def __init__(self):
        super(BinRotLoss, self).__init__()

    def forward(self, output, mask, ind, rotbin, rotres):
        pred = _tranpose_and_gather_feat(output, ind)
        loss = compute_rot_loss(pred, rotbin, rotres, mask)
        return loss


def bbox_transform_batch(ex_rois, gt_rois):
    if ex_rois.dim() == 2:
        ex_widths = ex_rois[:, (2)] - ex_rois[:, (0)] + 1.0
        ex_heights = ex_rois[:, (3)] - ex_rois[:, (1)] + 1.0
        ex_ctr_x = ex_rois[:, (0)] + 0.5 * ex_widths
        ex_ctr_y = ex_rois[:, (1)] + 0.5 * ex_heights
        gt_widths = gt_rois[:, :, (2)] - gt_rois[:, :, (0)] + 1.0
        gt_heights = gt_rois[:, :, (3)] - gt_rois[:, :, (1)] + 1.0
        gt_ctr_x = gt_rois[:, :, (0)] + 0.5 * gt_widths
        gt_ctr_y = gt_rois[:, :, (1)] + 0.5 * gt_heights
        targets_dx = (gt_ctr_x - ex_ctr_x.view(1, -1).expand_as(gt_ctr_x)
            ) / ex_widths
        targets_dy = (gt_ctr_y - ex_ctr_y.view(1, -1).expand_as(gt_ctr_y)
            ) / ex_heights
        targets_dw = torch.log(gt_widths / ex_widths.view(1, -1).expand_as(
            gt_widths))
        targets_dh = torch.log(gt_heights / ex_heights.view(1, -1).
            expand_as(gt_heights))
    elif ex_rois.dim() == 3:
        ex_widths = ex_rois[:, :, (2)] - ex_rois[:, :, (0)] + 1.0
        ex_heights = ex_rois[:, :, (3)] - ex_rois[:, :, (1)] + 1.0
        ex_ctr_x = ex_rois[:, :, (0)] + 0.5 * ex_widths
        ex_ctr_y = ex_rois[:, :, (1)] + 0.5 * ex_heights
        gt_widths = gt_rois[:, :, (2)] - gt_rois[:, :, (0)] + 1.0
        gt_heights = gt_rois[:, :, (3)] - gt_rois[:, :, (1)] + 1.0
        gt_ctr_x = gt_rois[:, :, (0)] + 0.5 * gt_widths
        gt_ctr_y = gt_rois[:, :, (1)] + 0.5 * gt_heights
        targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
        targets_dw = torch.log(gt_widths / ex_widths)
        targets_dh = torch.log(gt_heights / ex_heights)
    else:
        raise ValueError('ex_roi input dimension is not correct.')
    targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh), 2)
    return targets


def _compute_targets_batch(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""
    return bbox_transform_batch(ex_rois, gt_rois[:, :, :4])


def _unmap(data, count, inds, batch_size, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if data.dim() == 2:
        ret = torch.Tensor(batch_size, count).fill_(fill).type_as(data)
        ret[:, (inds)] = data
    else:
        ret = torch.Tensor(batch_size, count, data.size(2)).fill_(fill
            ).type_as(data)
        ret[:, (inds), :] = data
    return ret


def bbox_overlaps_batch(anchors, gt_boxes):
    """
    anchors: (N, 4) ndarray of float
    gt_boxes: (b, K, 5) ndarray of float

    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    batch_size = gt_boxes.size(0)
    if anchors.dim() == 2:
        N = anchors.size(0)
        K = gt_boxes.size(1)
        anchors = anchors.view(1, N, 4).expand(batch_size, N, 4).contiguous()
        gt_boxes = gt_boxes[:, :, :4].contiguous()
        gt_boxes_x = gt_boxes[:, :, (2)] - gt_boxes[:, :, (0)] + 1
        gt_boxes_y = gt_boxes[:, :, (3)] - gt_boxes[:, :, (1)] + 1
        gt_boxes_area = (gt_boxes_x * gt_boxes_y).view(batch_size, 1, K)
        anchors_boxes_x = anchors[:, :, (2)] - anchors[:, :, (0)] + 1
        anchors_boxes_y = anchors[:, :, (3)] - anchors[:, :, (1)] + 1
        anchors_area = (anchors_boxes_x * anchors_boxes_y).view(batch_size,
            N, 1)
        gt_area_zero = (gt_boxes_x == 1) & (gt_boxes_y == 1)
        anchors_area_zero = (anchors_boxes_x == 1) & (anchors_boxes_y == 1)
        boxes = anchors.view(batch_size, N, 1, 4).expand(batch_size, N, K, 4)
        query_boxes = gt_boxes.view(batch_size, 1, K, 4).expand(batch_size,
            N, K, 4)
        iw = torch.min(boxes[:, :, :, (2)], query_boxes[:, :, :, (2)]
            ) - torch.max(boxes[:, :, :, (0)], query_boxes[:, :, :, (0)]) + 1
        iw[iw < 0] = 0
        ih = torch.min(boxes[:, :, :, (3)], query_boxes[:, :, :, (3)]
            ) - torch.max(boxes[:, :, :, (1)], query_boxes[:, :, :, (1)]) + 1
        ih[ih < 0] = 0
        ua = anchors_area + gt_boxes_area - iw * ih
        overlaps = iw * ih / ua
        overlaps.masked_fill_(gt_area_zero.view(batch_size, 1, K).expand(
            batch_size, N, K), 0)
        overlaps.masked_fill_(anchors_area_zero.view(batch_size, N, 1).
            expand(batch_size, N, K), -1)
    elif anchors.dim() == 3:
        N = anchors.size(1)
        K = gt_boxes.size(1)
        if anchors.size(2) == 4:
            anchors = anchors[:, :, :4].contiguous()
        else:
            anchors = anchors[:, :, 1:5].contiguous()
        gt_boxes = gt_boxes[:, :, :4].contiguous()
        gt_boxes_x = gt_boxes[:, :, (2)] - gt_boxes[:, :, (0)] + 1
        gt_boxes_y = gt_boxes[:, :, (3)] - gt_boxes[:, :, (1)] + 1
        gt_boxes_area = (gt_boxes_x * gt_boxes_y).view(batch_size, 1, K)
        anchors_boxes_x = anchors[:, :, (2)] - anchors[:, :, (0)] + 1
        anchors_boxes_y = anchors[:, :, (3)] - anchors[:, :, (1)] + 1
        anchors_area = (anchors_boxes_x * anchors_boxes_y).view(batch_size,
            N, 1)
        gt_area_zero = (gt_boxes_x == 1) & (gt_boxes_y == 1)
        anchors_area_zero = (anchors_boxes_x == 1) & (anchors_boxes_y == 1)
        boxes = anchors.view(batch_size, N, 1, 4).expand(batch_size, N, K, 4)
        query_boxes = gt_boxes.view(batch_size, 1, K, 4).expand(batch_size,
            N, K, 4)
        iw = torch.min(boxes[:, :, :, (2)], query_boxes[:, :, :, (2)]
            ) - torch.max(boxes[:, :, :, (0)], query_boxes[:, :, :, (0)]) + 1
        iw[iw < 0] = 0
        ih = torch.min(boxes[:, :, :, (3)], query_boxes[:, :, :, (3)]
            ) - torch.max(boxes[:, :, :, (1)], query_boxes[:, :, :, (1)]) + 1
        ih[ih < 0] = 0
        ua = anchors_area + gt_boxes_area - iw * ih
        overlaps = iw * ih / ua
        overlaps.masked_fill_(gt_area_zero.view(batch_size, 1, K).expand(
            batch_size, N, K), 0)
        overlaps.masked_fill_(anchors_area_zero.view(batch_size, N, 1).
            expand(batch_size, N, K), -1)
    else:
        raise ValueError('anchors input dimension is not correct.')
    return overlaps


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
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
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


def generate_anchors(base_size=16, ratios=[0.5, 1, 2], scales=2 ** np.
    arange(3, 6)):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.
    """
    base_anchor = np.array([1, 1, base_size, base_size]) - 1
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    anchors = np.vstack([_scale_enum(ratio_anchors[(i), :], scales) for i in
        xrange(ratio_anchors.shape[0])])
    return anchors


class _AnchorTargetLayer(nn.Module):
    """
        Assign anchors to ground-truth targets. Produces anchor classification
        labels and bounding-box regression targets.
    """

    def __init__(self, feat_stride, scales, ratios):
        super(_AnchorTargetLayer, self).__init__()
        self._feat_stride = feat_stride
        self._scales = scales
        anchor_scales = scales
        self._anchors = torch.from_numpy(generate_anchors(scales=np.array(
            anchor_scales), ratios=np.array(ratios))).float()
        self._num_anchors = self._anchors.size(0)
        self._allowed_border = 0
        self.bbox_inside_weights = 1.0
        self.rpn_negative_overlap = cfg.TRAIN.RPN_NEGATIVE_OVERLAP
        self.rpn_positive_overlap = cfg.TRAIN.RPN_POSITIVE_OVERLAP
        self.rpn_fg_fraction = cfg.TRAIN.RPN_FG_FRACTION
        self.rpn_batch_size = cfg.TRAIN.RPN_BATCHSIZE

    def forward(self, input):
        rpn_cls_score = input[0]
        gt_boxes = input[1]
        im_info = input[2]
        num_boxes = input[3]
        height, width = rpn_cls_score.size(2), rpn_cls_score.size(3)
        batch_size = gt_boxes.size(0)
        feat_height, feat_width = rpn_cls_score.size(2), rpn_cls_score.size(3)
        shift_x = np.arange(0, feat_width) * self._feat_stride
        shift_y = np.arange(0, feat_height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel
            (), shift_x.ravel(), shift_y.ravel())).transpose())
        shifts = shifts.contiguous().type_as(rpn_cls_score).float()
        A = self._num_anchors
        K = shifts.size(0)
        self._anchors = self._anchors.type_as(gt_boxes)
        all_anchors = self._anchors.view(1, A, 4) + shifts.view(K, 1, 4)
        all_anchors = all_anchors.view(K * A, 4)
        total_anchors = int(K * A)
        keep = (all_anchors[:, (0)] >= -self._allowed_border) & (all_anchors
            [:, (1)] >= -self._allowed_border) & (all_anchors[:, (2)] < 
            long(im_info[0][1]) + self._allowed_border) & (all_anchors[:, (
            3)] < long(im_info[0][0]) + self._allowed_border)
        inds_inside = torch.nonzero(keep).view(-1)
        anchors = all_anchors[(inds_inside), :]
        labels = gt_boxes.new(batch_size, inds_inside.size(0)).fill_(-1)
        bbox_inside_weights = gt_boxes.new(batch_size, inds_inside.size(0)
            ).zero_()
        bbox_outside_weights = gt_boxes.new(batch_size, inds_inside.size(0)
            ).zero_()
        overlaps = bbox_overlaps_batch(anchors, gt_boxes)
        max_overlaps, argmax_overlaps = torch.max(overlaps, 2)
        gt_max_overlaps, _ = torch.max(overlaps, 1)
        if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0
        gt_max_overlaps[gt_max_overlaps == 0] = 1e-05
        keep = torch.sum(overlaps.eq(gt_max_overlaps.view(batch_size, 1, -1
            ).expand_as(overlaps)), 2)
        if torch.sum(keep) > 0:
            labels[keep > 0] = 1
        labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1
        if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0
        num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)
        sum_fg = torch.sum((labels == 1).int(), 1)
        sum_bg = torch.sum((labels == 0).int(), 1)
        for i in range(batch_size):
            if sum_fg[i] > num_fg:
                fg_inds = torch.nonzero(labels[i] == 1).view(-1)
                rand_num = torch.from_numpy(np.random.permutation(fg_inds.
                    size(0))).type_as(gt_boxes).long()
                disable_inds = fg_inds[rand_num[:fg_inds.size(0) - num_fg]]
                labels[i][disable_inds] = -1
            num_bg = cfg.TRAIN.RPN_BATCHSIZE - torch.sum((labels == 1).int(), 1
                )[i]
            if sum_bg[i] > num_bg:
                bg_inds = torch.nonzero(labels[i] == 0).view(-1)
                rand_num = torch.from_numpy(np.random.permutation(bg_inds.
                    size(0))).type_as(gt_boxes).long()
                disable_inds = bg_inds[rand_num[:bg_inds.size(0) - num_bg]]
                labels[i][disable_inds] = -1
        offset = torch.arange(0, batch_size) * gt_boxes.size(1)
        argmax_overlaps = argmax_overlaps + offset.view(batch_size, 1).type_as(
            argmax_overlaps)
        bbox_targets = _compute_targets_batch(anchors, gt_boxes.view(-1, 5)
            [(argmax_overlaps.view(-1)), :].view(batch_size, -1, 5))
        bbox_inside_weights[labels == 1] = cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS[0]
        if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
            num_examples = torch.sum(labels[i] >= 0)
            positive_weights = 1.0 / num_examples.item()
            negative_weights = 1.0 / num_examples.item()
        else:
            assert (cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) & (cfg.TRAIN.
                RPN_POSITIVE_WEIGHT < 1)
        bbox_outside_weights[labels == 1] = positive_weights
        bbox_outside_weights[labels == 0] = negative_weights
        labels = _unmap(labels, total_anchors, inds_inside, batch_size, fill=-1
            )
        bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside,
            batch_size, fill=0)
        bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors,
            inds_inside, batch_size, fill=0)
        bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors,
            inds_inside, batch_size, fill=0)
        outputs = []
        labels = labels.view(batch_size, height, width, A).permute(0, 3, 1, 2
            ).contiguous()
        labels = labels.view(batch_size, 1, A * height, width)
        outputs.append(labels)
        bbox_targets = bbox_targets.view(batch_size, height, width, A * 4
            ).permute(0, 3, 1, 2).contiguous()
        outputs.append(bbox_targets)
        anchors_count = bbox_inside_weights.size(1)
        bbox_inside_weights = bbox_inside_weights.view(batch_size,
            anchors_count, 1).expand(batch_size, anchors_count, 4)
        bbox_inside_weights = bbox_inside_weights.contiguous().view(batch_size,
            height, width, 4 * A).permute(0, 3, 1, 2).contiguous()
        outputs.append(bbox_inside_weights)
        bbox_outside_weights = bbox_outside_weights.view(batch_size,
            anchors_count, 1).expand(batch_size, anchors_count, 4)
        bbox_outside_weights = bbox_outside_weights.contiguous().view(
            batch_size, height, width, 4 * A).permute(0, 3, 1, 2).contiguous()
        outputs.append(bbox_outside_weights)
        return outputs

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2
    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1,
        padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def _topk(scores, K=40):
    batch, cat, height, width = scores.size()
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)
    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch
        , K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)
    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def ctdet_decode(heat, wh, reg=None, cat_spec_wh=False, K=100):
    batch, cat, height, width = heat.size()
    heat = _nms(heat)
    scores, inds, clses, ys, xs = _topk(heat, K=K)
    if reg is not None:
        reg = _tranpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1) + 0.5
        ys = ys.view(batch, K, 1) + 0.5
    wh = _tranpose_and_gather_feat(wh, inds)
    if cat_spec_wh:
        wh = wh.view(batch, K, cat, 2)
        clses_ind = clses.view(batch, K, 1, 1).expand(batch, K, 1, 2).long()
        wh = wh.gather(2, clses_ind).view(batch, K, 2)
    else:
        wh = wh.view(batch, K, 2)
    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    bboxes = torch.cat([xs - wh[(...), 0:1] / 2, ys - wh[(...), 1:2] / 2, 
        xs + wh[(...), 0:1] / 2, ys + wh[(...), 1:2] / 2], dim=2)
    detections = torch.cat([bboxes, scores, clses], dim=2)
    return detections


class _ProposalLayer(nn.Module):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """

    def __init__(self, feat_stride):
        super(_ProposalLayer, self).__init__()
        self._feat_stride = feat_stride

    def forward(self, input):
        scores = input[0]
        wh_deltas = input[1]
        offset_deltas = input[2]
        im_info = input[3]
        cfg_key = input[4]
        post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
        detections = ctdet_decode(scores, wh_deltas, offset_deltas, K=
            post_nms_topN)
        detections[:, :, :4] *= self._feat_stride
        batch_size = scores.size(0)
        output = scores.new(batch_size, post_nms_topN, 5).zero_()
        for i in range(batch_size):
            output[(i), :, (0)] = i
            output[(i), :, 1:] = detections[(i), :, :4]
        return output

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    def _filter_boxes(self, boxes, min_size):
        """Remove all boxes with any side smaller than min_size."""
        ws = boxes[:, :, (2)] - boxes[:, :, (0)] + 1
        hs = boxes[:, :, (3)] - boxes[:, :, (1)] + 1
        keep = (ws >= min_size.view(-1, 1).expand_as(ws)) & (hs >= min_size
            .view(-1, 1).expand_as(hs))
        return keep


class _RPN(nn.Module):
    """ region proposal network """

    def __init__(self, din):
        super(_RPN, self).__init__()
        self.din = din
        self.feat_stride = cfg.FEAT_STRIDE
        self.RPN_hm_score = nn.Conv2d(self.din, 1, 1, 1, 0)
        self.PRN_wh_score = nn.Conv2d(self.din, 2, 1, 1, 0)
        self.PRN_offset_score = nn.Conv2d(self.din, 2, 1, 1, 0)
        self.RPN_proposal = _ProposalLayer(self.feat_stride)
        self.crit = FocalLoss()
        self.crit_offset = RegL1Loss()
        self.crit_wh = RegL1Loss()
        self.rpn_loss_hm = 0
        self.rpn_loss_wh = 0
        self.rpn_loss_offset = 0

    @staticmethod
    def reshape(x, d):
        input_shape = x.size()
        x = x.view(input_shape[0], int(d), int(float(input_shape[1] *
            input_shape[2]) / float(d)), input_shape[3])
        return x

    def forward(self, base_feat, im_info, gt_boxes, num_boxes, hm, reg_mask,
        wh, offset, ind):
        batch_size = base_feat.size(0)
        rpn_hm_score = self.RPN_hm_score(base_feat)
        rpn_cls_prob = F.sigmoid(rpn_hm_score)
        rpn_wh_pred = self.PRN_wh_score(base_feat)
        rpn_offset_pred = self.PRN_offset_score(base_feat)
        cfg_key = 'TRAIN' if self.training else 'TEST'
        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0
        if self.training:
            assert gt_boxes is not None
            hm_loss = self.crit(rpn_cls_prob, hm)
            offset_loss = self.crit_offset(rpn_offset_pred, reg_mask, ind,
                offset)
            wh_loss = self.crit_wh(rpn_wh_pred, reg_mask, ind, wh)
            self.rpn_loss_cls = hm_loss + offset_loss
            self.rpn_loss_box = wh_loss
        rois = self.RPN_proposal((rpn_cls_prob, rpn_wh_pred,
            rpn_offset_pred, im_info, cfg_key))
        return rois, self.rpn_loss_cls, self.rpn_loss_box


def bbox_transform_inv(boxes, deltas, batch_size):
    widths = boxes[:, :, (2)] - boxes[:, :, (0)] + 1.0
    heights = boxes[:, :, (3)] - boxes[:, :, (1)] + 1.0
    ctr_x = boxes[:, :, (0)] + 0.5 * widths
    ctr_y = boxes[:, :, (1)] + 0.5 * heights
    dx = deltas[:, :, 0::4]
    dy = deltas[:, :, 1::4]
    dw = deltas[:, :, 2::4]
    dh = deltas[:, :, 3::4]
    pred_ctr_x = dx * widths.unsqueeze(2) + ctr_x.unsqueeze(2)
    pred_ctr_y = dy * heights.unsqueeze(2) + ctr_y.unsqueeze(2)
    pred_w = torch.exp(dw) * widths.unsqueeze(2)
    pred_h = torch.exp(dh) * heights.unsqueeze(2)
    pred_boxes = deltas.clone()
    pred_boxes[:, :, 0::4] = pred_ctr_x - 0.5 * pred_w
    pred_boxes[:, :, 1::4] = pred_ctr_y - 0.5 * pred_h
    pred_boxes[:, :, 2::4] = pred_ctr_x + 0.5 * pred_w
    pred_boxes[:, :, 3::4] = pred_ctr_y + 0.5 * pred_h
    return pred_boxes


def clip_boxes(boxes, im_shape, batch_size):
    for i in range(batch_size):
        boxes[(i), :, 0::4].clamp_(0, im_shape[i, 1] - 1)
        boxes[(i), :, 1::4].clamp_(0, im_shape[i, 0] - 1)
        boxes[(i), :, 2::4].clamp_(0, im_shape[i, 1] - 1)
        boxes[(i), :, 3::4].clamp_(0, im_shape[i, 0] - 1)
    return boxes


class _ProposalLayer(nn.Module):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """

    def __init__(self, feat_stride, scales, ratios):
        super(_ProposalLayer, self).__init__()
        self._feat_stride = feat_stride
        self._anchors = torch.from_numpy(generate_anchors(scales=np.array(
            scales), ratios=np.array(ratios))).float()
        self._num_anchors = self._anchors.size(0)

    def forward(self, input):
        scores = input[0][:, self._num_anchors:, :, :]
        bbox_deltas = input[1]
        im_info = input[2]
        cfg_key = input[3]
        pre_nms_topN = cfg[cfg_key].RPN_PRE_NMS_TOP_N
        post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
        nms_thresh = cfg[cfg_key].RPN_NMS_THRESH
        min_size = cfg[cfg_key].RPN_MIN_SIZE
        batch_size = bbox_deltas.size(0)
        feat_height, feat_width = scores.size(2), scores.size(3)
        shift_x = np.arange(0, feat_width) * self._feat_stride
        shift_y = np.arange(0, feat_height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel
            (), shift_x.ravel(), shift_y.ravel())).transpose())
        shifts = shifts.contiguous().type_as(scores).float()
        A = self._num_anchors
        K = shifts.size(0)
        self._anchors = self._anchors.type_as(scores)
        anchors = self._anchors.view(1, A, 4) + shifts.view(K, 1, 4)
        anchors = anchors.view(1, K * A, 4).expand(batch_size, K * A, 4)
        bbox_deltas = bbox_deltas.permute(0, 2, 3, 1).contiguous()
        bbox_deltas = bbox_deltas.view(batch_size, -1, 4)
        scores = scores.permute(0, 2, 3, 1).contiguous()
        scores = scores.view(batch_size, -1)
        proposals = bbox_transform_inv(anchors, bbox_deltas, batch_size)
        proposals = clip_boxes(proposals, im_info, batch_size)
        scores_keep = scores
        proposals_keep = proposals
        _, order = torch.sort(scores_keep, 1, True)
        output = scores.new(batch_size, post_nms_topN, 5).zero_()
        for i in range(batch_size):
            proposals_single = proposals_keep[i]
            scores_single = scores_keep[i]
            order_single = order[i]
            if pre_nms_topN > 0 and pre_nms_topN < scores_keep.numel():
                order_single = order_single[:pre_nms_topN]
            proposals_single = proposals_single[(order_single), :]
            scores_single = scores_single[order_single].view(-1, 1)
            keep_idx_i = nms(proposals_single, scores_single.squeeze(1),
                nms_thresh)
            if post_nms_topN > 0:
                keep_idx_i = keep_idx_i[:post_nms_topN]
            proposals_single = proposals_single[(keep_idx_i), :]
            scores_single = scores_single[(keep_idx_i), :]
            num_proposal = proposals_single.size(0)
            output[(i), :, (0)] = i
            output[(i), :num_proposal, 1:] = proposals_single
        return output

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    def _filter_boxes(self, boxes, min_size):
        """Remove all boxes with any side smaller than min_size."""
        ws = boxes[:, :, (2)] - boxes[:, :, (0)] + 1
        hs = boxes[:, :, (3)] - boxes[:, :, (1)] + 1
        keep = (ws >= min_size.view(-1, 1).expand_as(ws)) & (hs >= min_size
            .view(-1, 1).expand_as(hs))
        return keep


class _ProposalTargetLayer(nn.Module):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    def __init__(self, nclasses):
        super(_ProposalTargetLayer, self).__init__()
        self._num_classes = nclasses
        self.BBOX_NORMALIZE_MEANS = torch.FloatTensor(cfg.TRAIN.
            BBOX_NORMALIZE_MEANS)
        self.BBOX_NORMALIZE_STDS = torch.FloatTensor(cfg.TRAIN.
            BBOX_NORMALIZE_STDS)
        self.BBOX_INSIDE_WEIGHTS = torch.FloatTensor(cfg.TRAIN.
            BBOX_INSIDE_WEIGHTS)

    def forward(self, all_rois, gt_boxes, num_boxes):
        self.BBOX_NORMALIZE_MEANS = self.BBOX_NORMALIZE_MEANS.type_as(gt_boxes)
        self.BBOX_NORMALIZE_STDS = self.BBOX_NORMALIZE_STDS.type_as(gt_boxes)
        self.BBOX_INSIDE_WEIGHTS = self.BBOX_INSIDE_WEIGHTS.type_as(gt_boxes)
        gt_boxes_append = gt_boxes.new(gt_boxes.size()).zero_()
        gt_boxes_append[:, :, 1:5] = gt_boxes[:, :, :4]
        all_rois = torch.cat([all_rois, gt_boxes_append], 1)
        num_images = 1
        rois_per_image = int(cfg.TRAIN.BATCH_SIZE / num_images)
        fg_rois_per_image = int(np.round(cfg.TRAIN.FG_FRACTION *
            rois_per_image))
        fg_rois_per_image = 1 if fg_rois_per_image == 0 else fg_rois_per_image
        labels, rois, bbox_targets, bbox_inside_weights = (self.
            _sample_rois_pytorch(all_rois, gt_boxes, fg_rois_per_image,
            rois_per_image, self._num_classes))
        bbox_outside_weights = (bbox_inside_weights > 0).float()
        return (rois, labels, bbox_targets, bbox_inside_weights,
            bbox_outside_weights)

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    def _get_bbox_regression_labels_pytorch(self, bbox_target_data,
        labels_batch, num_classes):
        """Bounding-box regression targets (bbox_target_data) are stored in a
        compact form b x N x (class, tx, ty, tw, th)

        This function expands those targets into the 4-of-4*K representation used
        by the network (i.e. only one class has non-zero targets).

        Returns:
            bbox_target (ndarray): b x N x 4K blob of regression targets
            bbox_inside_weights (ndarray): b x N x 4K blob of loss weights
        """
        batch_size = labels_batch.size(0)
        rois_per_image = labels_batch.size(1)
        clss = labels_batch
        bbox_targets = bbox_target_data.new(batch_size, rois_per_image, 4
            ).zero_()
        bbox_inside_weights = bbox_target_data.new(bbox_targets.size()).zero_()
        for b in range(batch_size):
            if clss[b].sum() == 0:
                continue
            inds = torch.nonzero(clss[b] > 0).view(-1)
            for i in range(inds.numel()):
                ind = inds[i]
                bbox_targets[(b), (ind), :] = bbox_target_data[(b), (ind), :]
                bbox_inside_weights[(b), (ind), :] = self.BBOX_INSIDE_WEIGHTS
        return bbox_targets, bbox_inside_weights

    def _compute_targets_pytorch(self, ex_rois, gt_rois):
        """Compute bounding-box regression targets for an image."""
        assert ex_rois.size(1) == gt_rois.size(1)
        assert ex_rois.size(2) == 4
        assert gt_rois.size(2) == 4
        batch_size = ex_rois.size(0)
        rois_per_image = ex_rois.size(1)
        targets = bbox_transform_batch(ex_rois, gt_rois)
        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            targets = (targets - self.BBOX_NORMALIZE_MEANS.expand_as(targets)
                ) / self.BBOX_NORMALIZE_STDS.expand_as(targets)
        return targets

    def _sample_rois_pytorch(self, all_rois, gt_boxes, fg_rois_per_image,
        rois_per_image, num_classes):
        """Generate a random sample of RoIs comprising foreground and background
        examples.
        """
        overlaps = bbox_overlaps_batch(all_rois, gt_boxes)
        max_overlaps, gt_assignment = torch.max(overlaps, 2)
        batch_size = overlaps.size(0)
        num_proposal = overlaps.size(1)
        num_boxes_per_img = overlaps.size(2)
        offset = torch.arange(0, batch_size) * gt_boxes.size(1)
        offset = offset.view(-1, 1).type_as(gt_assignment) + gt_assignment
        labels = gt_boxes[:, :, (4)].contiguous().view(-1)[offset.view(-1),
            ].view(batch_size, -1)
        labels_batch = labels.new(batch_size, rois_per_image).zero_()
        rois_batch = all_rois.new(batch_size, rois_per_image, 5).zero_()
        gt_rois_batch = all_rois.new(batch_size, rois_per_image, 5).zero_()
        for i in range(batch_size):
            fg_inds = torch.nonzero(max_overlaps[i] >= cfg.TRAIN.FG_THRESH
                ).view(-1)
            fg_num_rois = fg_inds.numel()
            bg_inds = torch.nonzero((max_overlaps[i] < cfg.TRAIN.
                BG_THRESH_HI) & (max_overlaps[i] >= cfg.TRAIN.BG_THRESH_LO)
                ).view(-1)
            bg_num_rois = bg_inds.numel()
            if fg_num_rois > 0 and bg_num_rois > 0:
                fg_rois_per_this_image = min(fg_rois_per_image, fg_num_rois)
                rand_num = torch.from_numpy(np.random.permutation(fg_num_rois)
                    ).type_as(gt_boxes).long()
                fg_inds = fg_inds[rand_num[:fg_rois_per_this_image]]
                bg_rois_per_this_image = (rois_per_image -
                    fg_rois_per_this_image)
                rand_num = np.floor(np.random.rand(bg_rois_per_this_image) *
                    bg_num_rois)
                rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()
                bg_inds = bg_inds[rand_num]
            elif fg_num_rois > 0 and bg_num_rois == 0:
                rand_num = np.floor(np.random.rand(rois_per_image) *
                    fg_num_rois)
                rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()
                fg_inds = fg_inds[rand_num]
                fg_rois_per_this_image = rois_per_image
                bg_rois_per_this_image = 0
            elif bg_num_rois > 0 and fg_num_rois == 0:
                rand_num = np.floor(np.random.rand(rois_per_image) *
                    bg_num_rois)
                rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()
                bg_inds = bg_inds[rand_num]
                bg_rois_per_this_image = rois_per_image
                fg_rois_per_this_image = 0
            else:
                raise ValueError(
                    'bg_num_rois = 0 and fg_num_rois = 0, this should not happen!'
                    )
            keep_inds = torch.cat([fg_inds, bg_inds], 0)
            labels_batch[i].copy_(labels[i][keep_inds])
            if fg_rois_per_this_image < rois_per_image:
                labels_batch[i][fg_rois_per_this_image:] = 0
            rois_batch[i] = all_rois[i][keep_inds]
            rois_batch[(i), :, (0)] = i
            gt_rois_batch[i] = gt_boxes[i][gt_assignment[i][keep_inds]]
        bbox_target_data = self._compute_targets_pytorch(rois_batch[:, :, 1
            :5], gt_rois_batch[:, :, :4])
        bbox_targets, bbox_inside_weights = (self.
            _get_bbox_regression_labels_pytorch(bbox_target_data,
            labels_batch, num_classes))
        return labels_batch, rois_batch, bbox_targets, bbox_inside_weights


_global_config['ANCHOR_SCALES'] = 4


_global_config['ANCHOR_RATIOS'] = 4


class _RPN(nn.Module):
    """ region proposal network """

    def __init__(self, din):
        super(_RPN, self).__init__()
        self.din = din
        self.anchor_scales = cfg.ANCHOR_SCALES
        self.anchor_ratios = cfg.ANCHOR_RATIOS
        self.feat_stride = cfg.FEAT_STRIDE
        self.nc_score_out = len(self.anchor_scales) * len(self.anchor_ratios
            ) * 2
        self.RPN_cls_score = nn.Conv2d(self.din, self.nc_score_out, 1, 1, 0)
        self.nc_bbox_out = len(self.anchor_scales) * len(self.anchor_ratios
            ) * 4
        self.RPN_bbox_pred = nn.Conv2d(self.din, self.nc_bbox_out, 1, 1, 0)
        self.RPN_proposal = _ProposalLayer(self.feat_stride, self.
            anchor_scales, self.anchor_ratios)
        self.RPN_anchor_target = _AnchorTargetLayer(self.feat_stride, self.
            anchor_scales, self.anchor_ratios)
        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

    @staticmethod
    def reshape(x, d):
        input_shape = x.size()
        x = x.view(input_shape[0], int(d), int(float(input_shape[1] *
            input_shape[2]) / float(d)), input_shape[3])
        return x

    def forward(self, base_feat, im_info, gt_boxes, num_boxes):
        batch_size = base_feat.size(0)
        rpn_cls_score = self.RPN_cls_score(base_feat)
        rpn_cls_score_reshape = self.reshape(rpn_cls_score, 2)
        rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape, 1)
        rpn_cls_prob = self.reshape(rpn_cls_prob_reshape, self.nc_score_out)
        rpn_bbox_pred = self.RPN_bbox_pred(base_feat)
        cfg_key = 'TRAIN' if self.training else 'TEST'
        rois = self.RPN_proposal((rpn_cls_prob.data, rpn_bbox_pred.data,
            im_info, cfg_key))
        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0
        if self.training:
            assert gt_boxes is not None
            rpn_data = self.RPN_anchor_target((rpn_cls_score.data, gt_boxes,
                im_info, num_boxes))
            rpn_cls_score = rpn_cls_score_reshape.permute(0, 2, 3, 1
                ).contiguous().view(batch_size, -1, 2)
            rpn_label = rpn_data[0].view(batch_size, -1)
            rpn_keep = Variable(rpn_label.view(-1).ne(-1).nonzero().view(-1))
            rpn_cls_score = torch.index_select(rpn_cls_score.view(-1, 2), 0,
                rpn_keep)
            rpn_label = torch.index_select(rpn_label.view(-1), 0, rpn_keep.data
                )
            rpn_label = Variable(rpn_label.long())
            loss = -F.log_softmax(rpn_cls_score, dim=1)[:, (0)]
            mask, num_pos = hard_negative_mining(loss, rpn_label)
            confidence = rpn_cls_score[(mask), :]
            self.rpn_loss_cls = F.cross_entropy(confidence.reshape(-1, 2),
                rpn_label[mask], reduction='mean')
            fg_cnt = torch.sum(rpn_label.data.ne(0))
            (rpn_bbox_targets, rpn_bbox_inside_weights,
                rpn_bbox_outside_weights) = rpn_data[1:]
            rpn_bbox_inside_weights = Variable(rpn_bbox_inside_weights)
            rpn_bbox_outside_weights = Variable(rpn_bbox_outside_weights)
            rpn_bbox_targets = Variable(rpn_bbox_targets)
            self.rpn_loss_box = _smooth_l1_loss(rpn_bbox_pred,
                rpn_bbox_targets, rpn_bbox_inside_weights,
                rpn_bbox_outside_weights, sigma=3, dim=[1, 2, 3])
        return rois, self.rpn_loss_cls, self.rpn_loss_box


class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and inp == oup
        self.conv = nn.Sequential(nn.Conv2d(inp, inp * expand_ratio, 1, 1, 
            0, bias=False), nn.BatchNorm2d(inp * expand_ratio), nn.ReLU6(
            inplace=True), nn.Conv2d(inp * expand_ratio, inp * expand_ratio,
            3, stride, 1, groups=inp * expand_ratio, bias=False), nn.
            BatchNorm2d(inp * expand_ratio), nn.ReLU6(inplace=True), nn.
            Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False), nn.
            BatchNorm2d(oup))

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=
            stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

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
        out = self.relu(out)
        return out


class LargeSeparableConv2d(nn.Module):

    def __init__(self, c_in, kernel_size=15, bias=False, bn=False, setting='L'
        ):
        super(LargeSeparableConv2d, self).__init__()
        dim_out = 10 * 7 * 7
        c_mid = 64 if setting == 'S' else 256
        self.din = c_in
        self.c_mid = c_mid
        self.c_out = dim_out
        self.k_width = kernel_size, 1
        self.k_height = 1, kernel_size
        self.pad = 0
        self.bias = bias
        self.bn = bn
        self.block1_1 = nn.Conv2d(self.din, self.c_mid, self.k_width, 1,
            padding=self.pad, bias=self.bias)
        self.bn1_1 = nn.BatchNorm2d(self.c_mid)
        self.block1_2 = nn.Conv2d(self.c_mid, self.c_out, self.k_height, 1,
            padding=self.pad, bias=self.bias)
        self.bn1_2 = nn.BatchNorm2d(self.c_out)
        self.block2_1 = nn.Conv2d(self.din, self.c_mid, self.k_height, 1,
            padding=self.pad, bias=self.bias)
        self.bn2_1 = nn.BatchNorm2d(self.c_mid)
        self.block2_2 = nn.Conv2d(self.c_mid, self.c_out, self.k_width, 1,
            padding=self.pad, bias=self.bias)
        self.bn2_2 = nn.BatchNorm2d(self.c_out)

    def forward(self, x):
        x1 = self.block1_1(x)
        x1 = self.bn1_1(x1) if self.bn else x1
        x1 = self.block1_2(x1)
        x1 = self.bn1_2(x1) if self.bn else x1
        x2 = self.block2_1(x)
        x2 = self.bn2_1(x2) if self.bn else x2
        x2 = self.block2_2(x2)
        x2 = self.bn2_2(x2) if self.bn else x2
        return x1 + x2


class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
        padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size,
            stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1,
            bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class _Block(nn.Module):

    def __init__(self, in_filters, out_filters, reps, strides=1,
        start_with_relu=True, grow_first=True):
        super(_Block, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        rep = []
        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=
                strides, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters
        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, stride=1,
                padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))
        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=
                strides, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)
        return x


class PSROIAlignExample(nn.Module):

    def __init__(self, spatial_scale=1.0 / 16.0, roi_size=7, sample_ratio=2,
        pooled_dim=10):
        super(PSROIAlignExample, self).__init__()
        self.psroialign = PSROIAlign(spatial_scale=spatial_scale, roi_size=
            roi_size, sampling_ratio=sample_ratio, pooled_dim=pooled_dim)

    def forward(self, feat, rois):
        None
        None
        None
        pooled_feat = self.psroialign(feat, rois)
        None
        return pooled_feat


class _PSROIAlign(Function):

    @staticmethod
    def forward(ctx, bottom_data, bottom_rois, spatial_scale, roi_size,
        sampling_ratio, pooled_dim):
        ctx.spatial_scale = spatial_scale
        ctx.roi_size = roi_size
        ctx.sampling_ratio = sampling_ratio
        ctx.pooled_dim = pooled_dim
        ctx.feature_size = bottom_data.size()
        num_rois = bottom_rois.size(0)
        top_data = torch.zeros([num_rois, pooled_dim, roi_size, roi_size],
            dtype=torch.float32).to(bottom_data.device)
        argmax_data = torch.zeros([num_rois, pooled_dim, roi_size, roi_size
            ], dtype=torch.int32).to(bottom_data.device)
        if bottom_data.is_cuda:
            _C.ps_roi_align_forward(bottom_data, bottom_rois, top_data,
                argmax_data, spatial_scale, roi_size, sampling_ratio)
            ctx.save_for_backward(bottom_rois, argmax_data)
        else:
            raise NotImplementedError
        return top_data

    @staticmethod
    @once_differentiable
    def backward(ctx, top_diff):
        spatial_scale = ctx.spatial_scale
        roi_size = ctx.roi_size
        sampling_ratio = ctx.sampling_ratio
        batch_size, channels, height, width = ctx.feature_size
        [bottom_rois, argmax_data] = ctx.saved_tensors
        bottom_diff = None
        if ctx.needs_input_grad[0]:
            bottom_diff = torch.zeros([batch_size, channels, height, width],
                dtype=torch.float32).to(top_diff.device)
            _C.ps_roi_align_backward(top_diff, argmax_data, bottom_rois,
                bottom_diff, spatial_scale, roi_size, sampling_ratio)
        return bottom_diff, None, None, None, None, None


ps_roi_align = _PSROIAlign.apply


class PSROIAlign(nn.Module):

    def __init__(self, spatial_scale, roi_size, sampling_ratio, pooled_dim):
        super(PSROIAlign, self).__init__()
        self.spatial_scale = spatial_scale
        self.roi_size = roi_size
        self.sampling_ratio = sampling_ratio
        self.pooled_dim = pooled_dim

    def forward(self, bottom_data, bottom_rois):
        return ps_roi_align(bottom_data, bottom_rois, self.spatial_scale,
            self.roi_size, self.sampling_ratio, self.pooled_dim)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '('
        tmpstr += 'spatial_scale=' + str(self.spatial_scale)
        tmpstr += ', roi_size=' + str(self.roi_size)
        tmpstr += ', sampling_ratio=' + str(self.sampling_ratio)
        tmpstr += ', pooled_dim=' + str(self.pooled_dim)
        tmpstr += ')'
        return tmpstr


class _PSROIPool(Function):

    @staticmethod
    def forward(ctx, features, rois, pooled_height, pooled_width,
        spatial_scale, group_size, output_dim):
        ctx.pooled_height = int(pooled_height)
        ctx.pooled_width = int(pooled_width)
        ctx.spatial_scale = float(spatial_scale)
        ctx.group_size = int(group_size)
        ctx.output_dim = int(output_dim)
        num_rois = rois.size()[0]
        output = torch.zeros(num_rois, ctx.output_dim, ctx.pooled_height,
            ctx.pooled_width).to(features.device)
        mappingchannel = torch.IntTensor(num_rois, ctx.output_dim, ctx.
            pooled_height, ctx.pooled_width).zero_().to(features.device)
        _C.ps_roi_pool_forward(ctx.pooled_height, ctx.pooled_width, ctx.
            spatial_scale, ctx.group_size, ctx.output_dim, features, rois,
            output, mappingchannel)
        ctx.save_for_backward(rois, mappingchannel)
        ctx.feature_size = features.size()
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        assert ctx.feature_size is not None and grad_output.is_cuda
        batch_size, num_channels, data_height, data_width = ctx.feature_size
        [rois, mappingchannel] = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = torch.zeros(batch_size, num_channels, data_height,
                data_width).to(grad_output.device)
            _C.ps_roi_pool_backward(ctx.pooled_height, ctx.pooled_width,
                ctx.spatial_scale, ctx.output_dim, grad_output, rois,
                grad_input, mappingchannel)
        return grad_input, None, None, None, None, None, None


ps_roi_pool = _PSROIPool.apply


class PSROIPool(nn.Module):

    def __init__(self, pooled_height, pooled_width, spatial_scale,
        group_size, output_dim):
        super(PSROIPool, self).__init__()
        self.pooled_width = int(pooled_width)
        self.pooled_height = int(pooled_height)
        self.spatial_scale = float(spatial_scale)
        self.group_size = int(group_size)
        self.output_dim = int(output_dim)

    def forward(self, features, rois):
        return ps_roi_pool(features, rois, self.pooled_height, self.
            pooled_width, self.spatial_scale, self.group_size, self.output_dim)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '('
        tmpstr += 'pooled_width=' + str(self.pooled_width)
        tmpstr += ', pooled_height=' + str(self.pooled_height)
        tmpstr += ', spatial_scale=' + str(self.spatial_scale)
        tmpstr += ', group_size=' + str(self.group_size)
        tmpstr += ', output_dim=' + str(self.output_dim)
        tmpstr += ')'
        return tmpstr


class PSROIAlignhandle(nn.Module):

    def __init__(self, spatial_scale=1.0 / 16.0, roi_size=7, sampling_ratio
        =2, pooled_dim=5):
        super(PSROIAlignhandle, self).__init__()
        self.psroialign = PSROIAlign(spatial_scale=spatial_scale, roi_size=
            roi_size, sampling_ratio=sampling_ratio, pooled_dim=pooled_dim)

    def forward(self, feat, rois):
        pooled_feat = self.psroialign(feat, rois)
        return pooled_feat


class PSROIPoolhandle(nn.Module):

    def __init__(self, pooled_height=7, pooled_width=7, spatial_scale=1.0 /
        16.0, group_size=7, output_dim=5):
        super(PSROIPoolhandle, self).__init__()
        self.psroipool = PSROIPool(pooled_height=pooled_height,
            pooled_width=pooled_width, spatial_scale=spatial_scale,
            group_size=group_size, output_dim=output_dim)

    def forward(self, feat, rois):
        pooled_feat = self.psroipool(feat, rois)
        return pooled_feat


class _fasterRCNN(nn.Module):
    """ faster RCNN """

    def __init__(self, n_classes):
        self.n_classes = n_classes
        super(_fasterRCNN, self).__init__()
        c_in = 1024
        self.RCNN_top = nn.Sequential(nn.Linear(5 * 7 * 7, c_in), nn.ReLU(
            inplace=True))
        self.RCNN_cls_score = nn.Linear(c_in, self.n_classes)
        self.RCNN_bbox_pred = nn.Linear(c_in, 4 * self.n_classes)

    def forward(self, pool5):
        pool5_flat = pool5.view(pool5.size(0), -1)
        None
        fc7 = self.RCNN_top(pool5_flat)
        RCNN_cls_score = self.RCNN_cls_score(fc7)
        cls_prob = F.softmax(RCNN_cls_score, 1)
        bbox_pred = self.RCNN_bbox_pred(fc7)
        return [cls_prob, bbox_pred]


class _RPN(nn.Module):
    """ region proposal network """

    def __init__(self, din):
        super(_RPN, self).__init__()
        self.din = din
        self.nc_score_out = 25 * 2
        self.RPN_cls_score = nn.Conv2d(self.din, self.nc_score_out, 1, 1, 0)
        self.nc_bbox_out = 25 * 4
        self.RPN_bbox_pred = nn.Conv2d(self.din, self.nc_bbox_out, 1, 1, 0)
        self.softmax = nn.Softmax(1)

    @staticmethod
    def reshape(x, d):
        input_shape = x.size()
        x = x.view(input_shape[0], int(d), int(float(input_shape[1] *
            input_shape[2]) / float(d)), input_shape[3])
        return x

    def forward(self, base_feat):
        rpn_cls_score = self.RPN_cls_score(base_feat)
        rpn_cls_score_reshape = self.reshape(rpn_cls_score, 2)
        rpn_cls_prob_reshape = self.softmax(rpn_cls_score_reshape)
        rpn_cls_prob = self.reshape(rpn_cls_prob_reshape, self.nc_score_out)
        rpn_bbox_pred = self.RPN_bbox_pred(base_feat)
        return rpn_cls_prob, rpn_bbox_pred


class _fasterRCNN(nn.Module):
    """ faster RCNN """

    def __init__(self):
        super(_fasterRCNN, self).__init__()
        self.RCNN_base = SnetExtractor(146)
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0
        self.rpn = RPN(in_channels=245, f_channels=256)
        self.sam = SAM(256, 245)
        self.RCNN_rpn = _RPN(256)

    def forward(self, im_data):
        basefeat = self.RCNN_base(im_data)
        rpn_feat = self.rpn(basefeat)
        rpn_cls_prob, rpn_bbox_pred = self.RCNN_rpn(rpn_feat)
        base_feat = self.sam([basefeat, rpn_feat])
        return [rpn_cls_prob, rpn_bbox_pred, base_feat]


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_ouyanghuiyu_Thundernet_Pytorch(_paritybench_base):
    pass
    def test_000(self):
        self._check(BasicBlock(*[], **{'inplanes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(FocalLoss(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(InvertedResidual(*[], **{'inp': 4, 'oup': 4, 'stride': 1, 'expand_ratio': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(LargeSeparableConv2d(*[], **{'c_in': 4}), [torch.rand([4, 4, 64, 64])], {})

    def test_004(self):
        self._check(RPN(*[], **{}), [torch.rand([4, 245, 64, 64])], {})

    def test_005(self):
        self._check(SAM(*[], **{'f_channels': 4, 'CEM_FILTER': 4}), [torch.rand([4, 4, 4, 64, 64])], {})

    def test_006(self):
        self._check(SeparableConv2d(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_007(self):
        self._check(_Block(*[], **{'in_filters': 4, 'out_filters': 4, 'reps': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_008(self):
        self._check(_RPN(*[], **{'din': 4}), [torch.rand([4, 4, 4, 4])], {})

