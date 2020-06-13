import sys
_module = sys.modules[__name__]
del sys
datasets = _module
coco = _module
COCO_data_pipeline = _module
ImageAugmentation = _module
coco_data = _module
heatmap = _module
preprocessing = _module
prn_data_pipeline = _module
prn_gaussian = _module
data_parallel = _module
dataloader = _module
evaluate = _module
multipose_coco_eval = _module
multipose_detection_val = _module
multipose_keypoint_val = _module
multipose_prn_val = _module
multipose_test = _module
tester = _module
lib = _module
core = _module
config = _module
nms = _module
_ext = _module
build = _module
pth_nms = _module
utils = _module
log = _module
meter = _module
path = _module
timer = _module
network = _module
anchors = _module
fpn = _module
joint_utils = _module
losses = _module
net_utils = _module
posenet = _module
utils = _module
training = _module
batch_processor = _module
multipose_detection_train = _module
multipose_keypoint_train = _module
multipose_prn_train = _module
trainer = _module

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


import itertools


import torch


from torch.nn import DataParallel


from torch.autograd import Variable


from torch.nn.parallel._functions import Scatter


from torch.nn.parallel._functions import Gather


import math


import numpy as np


from collections import OrderedDict


import torch.nn as nn


import torch.nn.functional as F


from copy import deepcopy


from torch.nn import init


from torch.optim.lr_scheduler import ReduceLROnPlateau


from torch.optim.lr_scheduler import _LRScheduler


from torch.optim.optimizer import Optimizer


class ScatterList(list):
    pass


def scatter(inputs, target_gpus, dim=0):
    """
    Slices variables into approximately equal chunks and
    distributes them across given GPUs. Duplicates
    references to objects that are not variables. Does not
    support Tensors.
    """

    def scatter_map(obj):
        if isinstance(obj, Variable):
            return Scatter.apply(target_gpus, None, dim, obj)
        assert not torch.is_tensor(obj), 'Tensors not supported in scatter.'
        if isinstance(obj, ScatterList):
            assert len(obj) == len(target_gpus)
            return [obj[i] for i in range(len(target_gpus))]
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            return list(map(list, zip(*map(scatter_map, obj))))
        if isinstance(obj, dict) and len(obj) > 0:
            return list(map(type(obj), zip(*map(scatter_map, obj.items()))))
        return [obj for targets in target_gpus]
    return scatter_map(inputs)


def pose_scatter_kwargs(inputs, kwargs, target_gpus, dim=0):
    """Scatter with support for kwargs dictionary"""
    inputs = scatter(inputs, target_gpus, dim) if inputs else []
    kwargs = scatter(kwargs, target_gpus, dim) if kwargs else []
    if len(inputs) < len(kwargs):
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
    elif len(kwargs) < len(inputs):
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs


class ConstList(list):
    pass


def pose_gather(outputs, target_device, dim=0):
    """
    Gathers variables from different GPUs on a specified device
      (-1 means the CPU).
    """

    def gather_map(outputs):
        if isinstance(outputs, Variable):
            if target_device == -1:
                return outputs.cpu()
            return outputs.cuda(target_device)
        out = outputs[0]
        if isinstance(out, Variable):
            return Gather.apply(target_device, dim, *outputs)
        if out is None:
            return None
        if isinstance(out, str):
            return out
        if isinstance(out, ConstList):
            return out
        if isinstance(out, ScatterList):
            return tuple(map(gather_map, itertools.chain(*outputs)))
        return type(out)(map(gather_map, zip(*outputs)))
    return gather_map(outputs)


class ListDataParallel(DataParallel):

    def scatter(self, inputs, kwargs, device_ids):
        return pose_scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def gather(self, outputs, output_device):
        return pose_gather(outputs, output_device, dim=self.dim)


def shift(shape, stride, anchors):
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(),
        shift_y.ravel())).transpose()
    A = anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)
        ).transpose((1, 0, 2))
    all_anchors = all_anchors.reshape((K * A, 4))
    return all_anchors


def generate_anchors(base_size=16, ratios=None, scales=None):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales w.r.t. a reference window.
    """
    if ratios is None:
        ratios = np.array([0.5, 1, 2])
    if scales is None:
        scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])
    num_anchors = len(ratios) * len(scales)
    anchors = np.zeros((num_anchors, 4))
    anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T
    areas = anchors[:, (2)] * anchors[:, (3)]
    anchors[:, (2)] = np.sqrt(areas / np.repeat(ratios, len(scales)))
    anchors[:, (3)] = anchors[:, (2)] * np.repeat(ratios, len(scales))
    anchors[:, 0::2] -= np.tile(anchors[:, (2)] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, (3)] * 0.5, (2, 1)).T
    return anchors


class Anchors(nn.Module):

    def __init__(self, pyramid_levels=None, strides=None, sizes=None,
        ratios=None, scales=None):
        super(Anchors, self).__init__()
        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]
        if strides is None:
            self.strides = [(2 ** x) for x in self.pyramid_levels]
        if sizes is None:
            self.sizes = [(2 ** (x + 2)) for x in self.pyramid_levels]
        if ratios is None:
            self.ratios = np.array([0.5, 1, 2])
        if scales is None:
            self.scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
                )

    def forward(self, image):
        image_shape = image.shape[2:]
        image_shape = np.array(image_shape)
        image_shapes = [((image_shape + 2 ** x - 1) // 2 ** x) for x in
            self.pyramid_levels]
        all_anchors = np.zeros((0, 4)).astype(np.float32)
        for idx, p in enumerate(self.pyramid_levels):
            anchors = generate_anchors(base_size=self.sizes[idx], ratios=
                self.ratios, scales=self.scales)
            shifted_anchors = shift(image_shapes[idx], self.strides[idx],
                anchors)
            all_anchors = np.append(all_anchors, shifted_anchors, axis=0)
        all_anchors = np.expand_dims(all_anchors, axis=0)
        return torch.from_numpy(all_anchors.astype(np.float32))


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size
            =1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(nn.Conv2d(in_planes, self.
                expansion * planes, kernel_size=1, stride=stride, bias=
                False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out


class FPN(nn.Module):

    def __init__(self, block, num_blocks):
        super(FPN, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.conv6 = nn.Conv2d(2048, 256, kernel_size=3, stride=2, padding=1)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.latlayer1 = nn.Conv2d(2048, 256, kernel_size=1, stride=1,
            padding=0)
        self.latlayer2 = nn.Conv2d(1024, 256, kernel_size=1, stride=1,
            padding=0)
        self.latlayer3 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0
            )
        self.toplayer0 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1
            )
        self.toplayer1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1
            )
        self.toplayer2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1
            )
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0
            )
        self.flatlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1,
            padding=0)
        self.flatlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1,
            padding=0)
        self.flatlayer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1,
            padding=0)
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        """Upsample and add two feature maps.

        Args:
          x: top feature map to be upsampled.
          y: lateral feature map.

        Returns:
          added feature map.
        """
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='nearest', align_corners=None
            ) + y

    def forward(self, x):
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        p6 = self.conv6(c5)
        p7 = self.conv7(F.relu(p6))
        p5 = self.latlayer1(c5)
        p4 = self._upsample_add(p5, self.latlayer2(c4))
        p3 = self._upsample_add(p4, self.latlayer3(c3))
        p5 = self.toplayer0(p5)
        p4 = self.toplayer1(p4)
        p3 = self.toplayer2(p3)
        fp5 = self.toplayer(c5)
        fp4 = self._upsample_add(fp5, self.flatlayer1(c4))
        fp3 = self._upsample_add(fp4, self.flatlayer2(c3))
        fp2 = self._upsample_add(fp3, self.flatlayer3(c2))
        fp4 = self.smooth1(fp4)
        fp3 = self.smooth2(fp3)
        fp2 = self.smooth3(fp2)
        return [[fp2, fp3, fp4, fp5], [p3, p4, p5, p6, p7]]


def calc_iou(a, b):
    area = (b[:, (2)] - b[:, (0)]) * (b[:, (3)] - b[:, (1)])
    iw = torch.min(torch.unsqueeze(a[:, (2)], dim=1), b[:, (2)]) - torch.max(
        torch.unsqueeze(a[:, (0)], 1), b[:, (0)])
    ih = torch.min(torch.unsqueeze(a[:, (3)], dim=1), b[:, (3)]) - torch.max(
        torch.unsqueeze(a[:, (1)], 1), b[:, (1)])
    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)
    ua = torch.unsqueeze((a[:, (2)] - a[:, (0)]) * (a[:, (3)] - a[:, (1)]),
        dim=1) + area - iw * ih
    ua = torch.clamp(ua, min=1e-08)
    intersection = iw * ih
    IoU = intersection / ua
    return IoU


class FocalLoss(nn.Module):

    def forward(self, classifications, regressions, anchors, annotations):
        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []
        anchor = anchors[(0), :, :]
        anchor_widths = anchor[:, (2)] - anchor[:, (0)]
        anchor_heights = anchor[:, (3)] - anchor[:, (1)]
        anchor_ctr_x = anchor[:, (0)] + 0.5 * anchor_widths
        anchor_ctr_y = anchor[:, (1)] + 0.5 * anchor_heights
        for j in range(batch_size):
            classification = classifications[(j), :, :]
            regression = regressions[(j), :, :]
            bbox_annotation = annotations[(j), :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, (4)] != -1]
            if bbox_annotation.shape[0] == 0:
                regression_losses.append(torch.tensor(0, requires_grad=True
                    ).float())
                classification_losses.append(torch.tensor(0, requires_grad=
                    True).float())
                continue
            classification = torch.clamp(classification, 0.0001, 1.0 - 0.0001)
            IoU = calc_iou(anchors[(0), :, :], bbox_annotation[:, :4])
            IoU_max, IoU_argmax = torch.max(IoU, dim=1)
            targets = torch.ones(classification.shape) * -1
            targets = targets
            targets[(torch.lt(IoU_max, 0.4)), :] = 0
            positive_indices = torch.ge(IoU_max, 0.5)
            num_positive_anchors = positive_indices.sum()
            assigned_annotations = bbox_annotation[(IoU_argmax), :]
            targets[(positive_indices), :] = 0
            targets[positive_indices, assigned_annotations[positive_indices,
                4].long()] = 1
            alpha_factor = torch.ones(targets.shape) * alpha
            alpha_factor = torch.where(torch.eq(targets, 1.0), alpha_factor,
                1.0 - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.0), 1.0 -
                classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)
            bce = -(targets * torch.log(classification) + (1.0 - targets) *
                torch.log(1.0 - classification))
            cls_loss = focal_weight * bce
            cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch
                .zeros(cls_loss.shape))
            classification_losses.append(cls_loss.sum() / torch.clamp(
                num_positive_anchors.float(), min=1.0))
            if positive_indices.sum() > 0:
                assigned_annotations = assigned_annotations[(
                    positive_indices), :]
                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]
                gt_widths = assigned_annotations[:, (2)
                    ] - assigned_annotations[:, (0)]
                gt_heights = assigned_annotations[:, (3)
                    ] - assigned_annotations[:, (1)]
                gt_ctr_x = assigned_annotations[:, (0)] + 0.5 * gt_widths
                gt_ctr_y = assigned_annotations[:, (1)] + 0.5 * gt_heights
                gt_widths = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)
                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)
                targets = torch.stack((targets_dx, targets_dy, targets_dw,
                    targets_dh))
                targets = targets.t()
                targets = targets / torch.Tensor([[0.1, 0.1, 0.2, 0.2]])
                negative_indices = 1 - positive_indices
                regression_diff = torch.abs(targets - regression[(
                    positive_indices), :])
                regression_loss = torch.where(torch.le(regression_diff, 1.0 /
                    9.0), 0.5 * 9.0 * torch.pow(regression_diff, 2), 
                    regression_diff - 0.5 / 9.0)
                regression_losses.append(regression_loss.mean())
            else:
                regression_losses.append(torch.tensor(0).float())
        return torch.stack(classification_losses).mean(dim=0, keepdim=True
            ), torch.stack(regression_losses).mean(dim=0, keepdim=True)


class Concat(nn.Module):

    def __init__(self):
        super(Concat, self).__init__()

    def forward(self, up1, up2, up3, up4):
        return torch.cat((up1, up2, up3, up4), 1)


class RegressionModel(nn.Module):

    def __init__(self, num_features_in, num_anchors=9, feature_size=256):
        super(RegressionModel, self).__init__()
        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3,
            padding=1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3,
            padding=1)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3,
            padding=1)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3,
            padding=1)
        self.act4 = nn.ReLU()
        self.output = nn.Conv2d(feature_size, num_anchors * 4, kernel_size=
            3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.act2(out)
        out = self.conv3(out)
        out = self.act3(out)
        out = self.conv4(out)
        out = self.act4(out)
        out = self.output(out)
        out = out.permute(0, 2, 3, 1)
        return out.contiguous().view(out.shape[0], -1, 4)


class ClassificationModel(nn.Module):

    def __init__(self, num_features_in, num_anchors=9, num_classes=80,
        prior=0.01, feature_size=256):
        super(ClassificationModel, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3,
            padding=1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3,
            padding=1)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3,
            padding=1)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3,
            padding=1)
        self.act4 = nn.ReLU()
        self.output = nn.Conv2d(feature_size, num_anchors * num_classes,
            kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.act2(out)
        out = self.conv3(out)
        out = self.act3(out)
        out = self.conv4(out)
        out = self.act4(out)
        out = self.output(out)
        out = self.output_act(out)
        out1 = out.permute(0, 2, 3, 1)
        batch_size, width, height, channels = out1.shape
        out2 = out1.view(batch_size, width, height, self.num_anchors, self.
            num_classes)
        return out2.contiguous().view(x.shape[0], -1, self.num_classes)


class Flatten(nn.Module):

    def forward(self, input):
        return input.view(input.size(0), -1)


class Add(nn.Module):

    def forward(self, input1, input2):
        return torch.add(input1, input2)


class PRN(nn.Module):

    def __init__(self, node_count, coeff):
        super(PRN, self).__init__()
        self.flatten = Flatten()
        self.height = coeff * 28
        self.width = coeff * 18
        self.dens1 = nn.Linear(self.height * self.width * 17, node_count)
        self.bneck = nn.Linear(node_count, node_count)
        self.dens2 = nn.Linear(node_count, self.height * self.width * 17)
        self.drop = nn.Dropout()
        self.add = Add()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        res = self.flatten(x)
        out = self.drop(F.relu(self.dens1(res)))
        out = self.drop(F.relu(self.bneck(out)))
        out = F.relu(self.dens2(out))
        out = self.add(out, res)
        out = self.softmax(out)
        out = out.view(out.size()[0], self.height, self.width, 17)
        return out


def pth_nms(dets, thresh):
    """
  dets has to be a tensor
  """
    if not dets.is_cuda:
        x1 = dets[:, (0)]
        y1 = dets[:, (1)]
        x2 = dets[:, (2)]
        y2 = dets[:, (3)]
        scores = dets[:, (4)]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.sort(0, descending=True)[1]
        keep = torch.LongTensor(dets.size(0))
        num_out = torch.LongTensor(1)
        nms.cpu_nms(keep, num_out, dets, order, areas, thresh)
        return keep[:num_out[0]]
    else:
        x1 = dets[:, (0)]
        y1 = dets[:, (1)]
        x2 = dets[:, (2)]
        y2 = dets[:, (3)]
        scores = dets[:, (4)]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.sort(0, descending=True)[1]
        dets = dets[order].contiguous()
        keep = torch.LongTensor(dets.size(0))
        num_out = torch.LongTensor(1)
        nms.gpu_nms(keep, num_out, dets, thresh)
        return order[keep[:num_out[0]].cuda()].contiguous()


def nms(dets, thresh):
    """Dispatch to either CPU or GPU NMS implementations.    Accept dets as tensor"""
    return pth_nms(dets, thresh)


def FPN50():
    return FPN(Bottleneck, [3, 4, 6, 3])


def build_detection_loss(saved_for_loss, anno):
    """
    :param saved_for_loss: [classifications, regressions, anchors]
    :param anno: annotations
    :return: classification_loss, regression_loss
    """
    saved_for_log = OrderedDict()
    focalLoss = losses.FocalLoss()
    classification_loss, regression_loss = focalLoss(*saved_for_loss, anno)
    classification_loss = classification_loss.mean()
    regression_loss = regression_loss.mean()
    total_loss = classification_loss + regression_loss
    saved_for_log['total_loss'] = total_loss.item()
    saved_for_log['classification_loss'] = classification_loss.item()
    saved_for_log['regression_loss'] = regression_loss.item()
    return total_loss, saved_for_log


def FPN101():
    return FPN(Bottleneck, [3, 4, 23, 3])


def build_prn_loss(saved_for_loss, label):
    """
    :param saved_for_loss: [out]
    :param label: label
    :return: prn loss
    """
    saved_for_log = OrderedDict()
    criterion = nn.BCELoss(size_average=True).cuda()
    total_loss = 0
    loss1 = criterion(saved_for_loss[0], label)
    total_loss += loss1
    saved_for_log['PRN loss'] = loss1.item()
    return total_loss, saved_for_log


def build_names():
    names = []
    for j in range(2, 6):
        names.append('heatmap_loss_k%d' % j)
        names.append('seg_loss_k%d' % j)
    names.append('heatmap_loss')
    names.append('seg_loss')
    return names


def build_keypoint_loss(saved_for_loss, heat_temp, heat_weight):
    names = build_names()
    saved_for_log = OrderedDict()
    criterion = nn.MSELoss(size_average=True).cuda()
    total_loss = 0
    div1 = 1.0
    for j in range(5):
        pred1 = saved_for_loss[j][:, :18, :, :] * heat_weight
        gt1 = heat_weight * heat_temp
        loss1 = criterion(pred1, gt1) / div1
        total_loss += loss1
        saved_for_log[names[j * 2]] = loss1.item()
    saved_for_log['max_ht'] = torch.max(saved_for_loss[-1].data[:, :18, :, :]
        ).item()
    saved_for_log['min_ht'] = torch.min(saved_for_loss[-1].data[:, :18, :, :]
        ).item()
    return total_loss, saved_for_log


class poseNet(nn.Module):

    def __init__(self, layers, prn_node_count=1024, prn_coeff=2):
        super(poseNet, self).__init__()
        if layers == 101:
            self.fpn = FPN101()
        if layers == 50:
            self.fpn = FPN50()
        self.convfin_k2 = nn.Conv2d(256, 19, kernel_size=1, stride=1, padding=0
            )
        self.convfin_k3 = nn.Conv2d(256, 19, kernel_size=1, stride=1, padding=0
            )
        self.convfin_k4 = nn.Conv2d(256, 19, kernel_size=1, stride=1, padding=0
            )
        self.convfin_k5 = nn.Conv2d(256, 19, kernel_size=1, stride=1, padding=0
            )
        self.convt1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.convt2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.convt3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.convt4 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.convs1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.convs2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.convs3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.convs4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.upsample1 = nn.Upsample(scale_factor=8, mode='nearest',
            align_corners=None)
        self.upsample2 = nn.Upsample(scale_factor=4, mode='nearest',
            align_corners=None)
        self.upsample3 = nn.Upsample(scale_factor=2, mode='nearest',
            align_corners=None)
        self.concat = Concat()
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.convfin = nn.Conv2d(256, 18, kernel_size=1, stride=1, padding=0)
        self.regressionModel = RegressionModel(256)
        self.classificationModel = ClassificationModel(256, num_classes=1)
        self.anchors = Anchors()
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()
        self.focalLoss = losses.FocalLoss()
        self.prn = PRN(prn_node_count, prn_coeff)
        self._initialize_weights_norm()
        prior = 0.01
        self.classificationModel.output.weight.data.fill_(0)
        self.classificationModel.output.bias.data.fill_(-math.log((1.0 -
            prior) / prior))
        self.regressionModel.output.weight.data.fill_(0)
        self.regressionModel.output.bias.data.fill_(0)
        self.freeze_bn()

    def _initialize_weights_norm(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0.0)

    def freeze_bn(self):
        """Freeze BatchNorm layers."""
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, x):
        img_batch, subnet_name = x
        if subnet_name == 'keypoint_subnet':
            return self.keypoint_forward(img_batch)
        elif subnet_name == 'detection_subnet':
            return self.detection_forward(img_batch)
        elif subnet_name == 'prn_subnet':
            return self.prn_forward(img_batch)
        else:
            features = self.fpn(img_batch)
            p2, p3, p4, p5 = features[0]
            features = features[1]
            p5 = self.convt1(p5)
            p5 = self.convs1(p5)
            p4 = self.convt2(p4)
            p4 = self.convs2(p4)
            p3 = self.convt3(p3)
            p3 = self.convs3(p3)
            p2 = self.convt4(p2)
            p2 = self.convs4(p2)
            p5 = self.upsample1(p5)
            p4 = self.upsample2(p4)
            p3 = self.upsample3(p3)
            concat = self.concat(p5, p4, p3, p2)
            predict_keypoint = self.convfin(F.relu(self.conv2(concat)))
            del p5, p4, p3, p2, concat
            regression = torch.cat([self.regressionModel(feature) for
                feature in features], dim=1)
            classification = torch.cat([self.classificationModel(feature) for
                feature in features], dim=1)
            anchors = self.anchors(img_batch)
            transformed_anchors = self.regressBoxes(anchors, regression)
            transformed_anchors = self.clipBoxes(transformed_anchors, img_batch
                )
            scores = torch.max(classification, dim=2, keepdim=True)[0]
            scores_over_thresh = (scores > 0.05)[(0), :, (0)]
            if scores_over_thresh.sum() == 0:
                return predict_keypoint, [torch.zeros(0), torch.zeros(0),
                    torch.zeros(0, 4)]
            classification = classification[:, (scores_over_thresh), :]
            transformed_anchors = transformed_anchors[:, (
                scores_over_thresh), :]
            scores = scores[:, (scores_over_thresh), :]
            anchors_nms_idx = nms(torch.cat([transformed_anchors, scores],
                dim=2)[(0), :, :], 0.5)
            nms_scores, nms_class = classification[(0), (anchors_nms_idx), :
                ].max(dim=1)
            return predict_keypoint, [nms_scores, nms_class,
                transformed_anchors[(0), (anchors_nms_idx), :]]

    def keypoint_forward(self, img_batch):
        saved_for_loss = []
        p2, p3, p4, p5 = self.fpn(img_batch)[0]
        saved_for_loss.append(self.convfin_k2(p2))
        saved_for_loss.append(self.upsample3(self.convfin_k3(p3)))
        saved_for_loss.append(self.upsample2(self.convfin_k4(p4)))
        saved_for_loss.append(self.upsample1(self.convfin_k5(p5)))
        p5 = self.convt1(p5)
        p5 = self.convs1(p5)
        p4 = self.convt2(p4)
        p4 = self.convs2(p4)
        p3 = self.convt3(p3)
        p3 = self.convs3(p3)
        p2 = self.convt4(p2)
        p2 = self.convs4(p2)
        p5 = self.upsample1(p5)
        p4 = self.upsample2(p4)
        p3 = self.upsample3(p3)
        predict_keypoint = self.convfin(F.relu(self.conv2(self.concat(p5,
            p4, p3, p2))))
        saved_for_loss.append(predict_keypoint)
        return predict_keypoint, saved_for_loss

    def detection_forward(self, img_batch):
        saved_for_loss = []
        features = self.fpn(img_batch)[1]
        regression = torch.cat([self.regressionModel(feature) for feature in
            features], dim=1)
        classification = torch.cat([self.classificationModel(feature) for
            feature in features], dim=1)
        anchors = self.anchors(img_batch)
        saved_for_loss.append(classification)
        saved_for_loss.append(regression)
        saved_for_loss.append(anchors)
        return [], saved_for_loss

    def prn_forward(self, img_batch):
        saved_for_loss = []
        res = self.prn.flatten(img_batch)
        out = self.prn.drop(F.relu(self.prn.dens1(res)))
        out = self.prn.drop(F.relu(self.prn.bneck(out)))
        out = F.relu(self.prn.dens2(out))
        out = self.prn.add(out, res)
        out = self.prn.softmax(out)
        out = out.view(out.size()[0], self.prn.height, self.prn.width, 17)
        saved_for_loss.append(out)
        return out, saved_for_loss

    @staticmethod
    def build_loss(saved_for_loss, *args):
        subnet_name = args[0]
        if subnet_name == 'keypoint_subnet':
            return build_keypoint_loss(saved_for_loss, args[1], args[2])
        elif subnet_name == 'detection_subnet':
            return build_detection_loss(saved_for_loss, args[1])
        elif subnet_name == 'prn_subnet':
            return build_prn_loss(saved_for_loss, args[1])
        else:
            return 0


class BBoxTransform(nn.Module):

    def __init__(self, mean=None, std=None):
        super(BBoxTransform, self).__init__()
        if mean is None:
            self.mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.
                float32))
        else:
            self.mean = mean
        if std is None:
            self.std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).
                astype(np.float32))
        else:
            self.std = std

    def forward(self, boxes, deltas):
        widths = boxes[:, :, (2)] - boxes[:, :, (0)]
        heights = boxes[:, :, (3)] - boxes[:, :, (1)]
        ctr_x = boxes[:, :, (0)] + 0.5 * widths
        ctr_y = boxes[:, :, (1)] + 0.5 * heights
        dx = deltas[:, :, (0)] * self.std[0] + self.mean[0]
        dy = deltas[:, :, (1)] * self.std[1] + self.mean[1]
        dw = deltas[:, :, (2)] * self.std[2] + self.mean[2]
        dh = deltas[:, :, (3)] * self.std[3] + self.mean[3]
        pred_ctr_x = ctr_x + dx * widths
        pred_ctr_y = ctr_y + dy * heights
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights
        pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
        pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
        pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
        pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h
        pred_boxes = torch.stack([pred_boxes_x1, pred_boxes_y1,
            pred_boxes_x2, pred_boxes_y2], dim=2)
        return pred_boxes


class ClipBoxes(nn.Module):

    def __init__(self, width=None, height=None):
        super(ClipBoxes, self).__init__()

    def forward(self, boxes, img):
        batch_size, num_channels, height, width = img.shape
        boxes[:, :, (0)] = torch.clamp(boxes[:, :, (0)], min=0)
        boxes[:, :, (1)] = torch.clamp(boxes[:, :, (1)], min=0)
        boxes[:, :, (2)] = torch.clamp(boxes[:, :, (2)], max=width)
        boxes[:, :, (3)] = torch.clamp(boxes[:, :, (3)], max=height)
        return boxes


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_LiMeng95_MultiPoseNet_pytorch(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(Anchors(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(Bottleneck(*[], **{'in_planes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(Concat(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(RegressionModel(*[], **{'num_features_in': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_004(self):
        self._check(ClassificationModel(*[], **{'num_features_in': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_005(self):
        self._check(Flatten(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_006(self):
        self._check(Add(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_007(self):
        self._check(BBoxTransform(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_008(self):
        self._check(ClipBoxes(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

