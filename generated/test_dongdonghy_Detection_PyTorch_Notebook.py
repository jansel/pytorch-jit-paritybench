import sys
_module = sys.modules[__name__]
del sys
evaluation = _module
Evaluator = _module
detection = _module
utils = _module
mlp = _module
perception = _module
perception_sequential = _module
visdom = _module
densenet_block = _module
detnet_bottleneck = _module
fpn = _module
inceptionv1 = _module
inceptionv2 = _module
resnet_bottleneck = _module
vgg = _module
_init_paths = _module
demo = _module
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
model = _module
faster_rcnn = _module
faster_rcnn = _module
resnet = _module
vgg16 = _module
nms = _module
_ext = _module
build = _module
nms_cpu = _module
nms_gpu = _module
nms_wrapper = _module
roi_align = _module
functions = _module
modules = _module
roi_align = _module
roi_crop = _module
crop_resize = _module
gridgen = _module
gridgen = _module
roi_crop = _module
roi_pooling = _module
roi_pool = _module
roi_pool = _module
rpn = _module
anchor_target_layer = _module
bbox_transform = _module
generate_anchors = _module
proposal_layer = _module
proposal_target_layer_cascade = _module
rpn = _module
blob = _module
config = _module
net_utils = _module
pycocotools = _module
cocoeval = _module
mask = _module
roi_data_layer = _module
minibatch = _module
roibatchLoader = _module
roidb = _module
setup = _module
test_net = _module
trainval_net = _module
arm = _module
tcb = _module
data = _module
voc0712 = _module
live = _module
eval = _module
layers = _module
box_utils = _module
prior_box = _module
l2norm = _module
multibox_loss = _module
ssd = _module
test = _module
train = _module
augmentations = _module
cfgs = _module
config_voc = _module
exps = _module
darknet19_exp1 = _module
darknet19_exp2 = _module
darknet = _module
reorg = _module
reorg_layer = _module
reorg_layer = _module
roi_pool = _module
roi_pool_py = _module
im_transform = _module
network = _module
py_cpu_nms = _module
timer = _module
yolo = _module
mobilenet_v1 = _module
mobilenet_v2 = _module
mobilenet_v2_block = _module
shufflenet_v1 = _module
squeezenet_fire = _module
retinanet = _module

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


from torch import nn


import torch


import torch.nn.functional as F


import torch.nn as nn


import math


import numpy as np


from torch.autograd import Variable


import torch.optim as optim


import random


import torch.utils.model_zoo as model_zoo


from torch.nn.modules.module import Module


from torch.nn.functional import avg_pool2d


from torch.nn.functional import max_pool2d


import numpy.random as npr


from torch.utils.data.sampler import Sampler


import torch.backends.cudnn as cudnn


import torch.utils.data as data


from torch.autograd import Function


import torch.nn.init as init


from functools import partial


class MLP(nn.Module):

    def __init__(self, in_dim, hid_dim1, hid_dim2, out_dim):
        super(MLP, self).__init__()
        self.layer = nn.Sequential(nn.Linear(in_dim, hid_dim1), nn.ReLU(),
            nn.Linear(hid_dim1, hid_dim2), nn.ReLU(), nn.Linear(hid_dim2,
            out_dim), nn.ReLU())

    def forward(self, x):
        x = self.layer(x)
        return x


class Linear(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(Linear, self).__init__()
        self.w = nn.Parameter(torch.randn(in_dim, out_dim))
        self.b = nn.Parameter(torch.randn(out_dim))

    def forward(self, x):
        x = x.matmul(self.w)
        y = x + self.b.expand_as(x)
        return y


class Perception(nn.Module):

    def __init__(self, in_dim, hid_dim, out_dim):
        super(Perception, self).__init__()
        self.layer1 = Linear(in_dim, hid_dim)
        self.layer2 = Linear(hid_dim, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        y = torch.sigmoid(x)
        y = self.layer2(y)
        y = torch.sigmoid(y)
        return y


class Perception(nn.Module):

    def __init__(self, in_dim, hid_dim, out_dim):
        super(Perception, self).__init__()
        self.layer = nn.Sequential(nn.Linear(in_dim, hid_dim), nn.Sigmoid(),
            nn.Linear(hid_dim, out_dim), nn.Sigmoid())

    def forward(self, x):
        y = self.layer(x)
        return y


class Bottleneck(nn.Module):

    def __init__(self, nChannels, growthRate):
        super(Bottleneck, self).__init__()
        interChannels = 4 * growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1,
            bias=False)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,
            padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat((x, out), 1)
        return out


class Denseblock(nn.Module):

    def __init__(self, nChannels, growthRate, nDenseBlocks):
        super(Denseblock, self).__init__()
        layers = []
        for i in range(int(nDenseBlocks)):
            layers.append(Bottleneck(nChannels, growthRate))
            nChannels += growthRate
        self.denseblock = nn.Sequential(*layers)

    def forward(self, x):
        return self.denseblock(x)


class DetBottleneck(nn.Module):

    def __init__(self, inplanes, planes, stride=1, extra=False):
        super(DetBottleneck, self).__init__()
        self.bottleneck = nn.Sequential(nn.Conv2d(inplanes, planes, 1, bias
            =False), nn.BatchNorm2d(planes), nn.ReLU(inplace=True), nn.
            Conv2d(planes, planes, kernel_size=3, stride=1, padding=2,
            dilation=2, bias=False), nn.BatchNorm2d(planes), nn.ReLU(
            inplace=True), nn.Conv2d(planes, planes, 1, bias=False), nn.
            BatchNorm2d(planes))
        self.relu = nn.ReLU(inplace=True)
        self.extra = extra
        if self.extra:
            self.extra_conv = nn.Sequential(nn.Conv2d(inplanes, planes, 1,
                bias=False), nn.BatchNorm2d(planes))

    def forward(self, x):
        if self.extra:
            identity = self.extra_conv(x)
        else:
            identity = x
        out = self.bottleneck(x)
        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.bottleneck = nn.Sequential(nn.Conv2d(in_planes, planes, 1,
            bias=False), nn.BatchNorm2d(planes), nn.ReLU(inplace=True), nn.
            Conv2d(planes, planes, 3, stride, 1, bias=False), nn.
            BatchNorm2d(planes), nn.ReLU(inplace=True), nn.Conv2d(planes, 
            self.expansion * planes, 1, bias=False), nn.BatchNorm2d(self.
            expansion * planes))
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.bottleneck(x)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class FPN(nn.Module):

    def __init__(self, layers):
        super(FPN, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], 2)
        self.layer3 = self._make_layer(256, layers[2], 2)
        self.layer4 = self._make_layer(512, layers[3], 2)
        self.toplayer = nn.Conv2d(2048, 256, 1, 1, 0)
        self.smooth1 = nn.Conv2d(256, 256, 3, 1, 1)
        self.smooth2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.smooth3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.latlayer1 = nn.Conv2d(1024, 256, 1, 1, 0)
        self.latlayer2 = nn.Conv2d(512, 256, 1, 1, 0)
        self.latlayer3 = nn.Conv2d(256, 256, 1, 1, 0)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != Bottleneck.expansion * planes:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, Bottleneck.
                expansion * planes, 1, stride, bias=False), nn.BatchNorm2d(
                Bottleneck.expansion * planes))
        layers = []
        layers.append(Bottleneck(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * Bottleneck.expansion
        for i in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        _, _, H, W = y.shape
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def forward(self, x):
        c1 = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)
        return p2, p3, p4, p5


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
            padding=padding)

    def forward(self, x):
        x = self.conv(x)
        return F.relu(x, inplace=True)


class Inceptionv1(nn.Module):

    def __init__(self, in_dim, hid_1_1, hid_2_1, hid_2_3, hid_3_1, out_3_5,
        out_4_1):
        super(Inceptionv1, self).__init__()
        self.branch1x1 = BasicConv2d(in_dim, hid_1_1, 1)
        self.branch3x3 = nn.Sequential(BasicConv2d(in_dim, hid_2_1, 1),
            BasicConv2d(hid_2_1, hid_2_3, 3, padding=1))
        self.branch5x5 = nn.Sequential(BasicConv2d(in_dim, hid_3_1, 1),
            BasicConv2d(hid_3_1, out_3_5, 5, padding=2))
        self.branch_pool = nn.Sequential(nn.MaxPool2d(3, stride=1, padding=
            1), BasicConv2d(in_dim, out_4_1, 1))

    def forward(self, x):
        b1 = self.branch1x1(x)
        b2 = self.branch3x3(x)
        b3 = self.branch5x5(x)
        b4 = self.branch_pool(x)
        output = torch.cat((b1, b2, b3, b4), dim=1)
        return output


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
            padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class Inceptionv2(nn.Module):

    def __init__(self):
        super(Inceptionv2, self).__init__()
        self.branch1 = BasicConv2d(192, 96, 1, 0)
        self.branch2 = nn.Sequential(BasicConv2d(192, 48, 1, 0),
            BasicConv2d(48, 64, 3, 1))
        self.branch3 = nn.Sequential(BasicConv2d(192, 64, 1, 0),
            BasicConv2d(64, 96, 3, 1), BasicConv2d(96, 96, 3, 1))
        self.branch4 = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False), BasicConv2d(192, 64, 1, 0))

    def forward(self, x):
        x0 = self.branch1(x)
        x1 = self.branch2(x)
        x2 = self.branch3(x)
        x3 = self.branch4(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Bottleneck(nn.Module):

    def __init__(self, in_dim, out_dim, stride=1):
        super(Bottleneck, self).__init__()
        self.bottleneck = nn.Sequential(nn.Conv2d(in_dim, in_dim, 1, bias=
            False), nn.BatchNorm2d(in_dim), nn.ReLU(inplace=True), nn.
            Conv2d(in_dim, in_dim, 3, stride, 1, bias=False), nn.
            BatchNorm2d(in_dim), nn.ReLU(inplace=True), nn.Conv2d(in_dim,
            out_dim, 1, bias=False), nn.BatchNorm2d(out_dim))
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(nn.Conv2d(in_dim, out_dim, 1, 1),
            nn.BatchNorm2d(out_dim))

    def forward(self, x):
        identity = x
        out = self.bottleneck(x)
        identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class VGG(nn.Module):

    def __init__(self, num_classes=1000):
        super(VGG, self).__init__()
        layers = []
        in_dim = 3
        out_dim = 64
        for i in range(13):
            layers += [nn.Conv2d(in_dim, out_dim, 3, 1, 1), nn.ReLU(inplace
                =True)]
            in_dim = out_dim
            if i == 1 or i == 3 or i == 6 or i == 9 or i == 12:
                layers += [nn.MaxPool2d(2, 2)]
                if i != 9:
                    out_dim *= 2
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.Linear(512 * 7 * 7, 4096), nn.
            ReLU(True), nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(True),
            nn.Dropout(), nn.Linear(4096, num_classes))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def _smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights,
    bbox_outside_weights, sigma=1.0, dim=[1]):
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
    loss_box = loss_box.mean()
    return loss_box


def _affine_grid_gen(rois, input_size, grid_size):
    rois = rois.detach()
    x1 = rois[:, 1::4] / 16.0
    y1 = rois[:, 2::4] / 16.0
    x2 = rois[:, 3::4] / 16.0
    y2 = rois[:, 4::4] / 16.0
    height = input_size[0]
    width = input_size[1]
    zero = Variable(rois.data.new(rois.size(0), 1).zero_())
    theta = torch.cat([(x2 - x1) / (width - 1), zero, (x1 + x2 - width + 1) /
        (width - 1), zero, (y2 - y1) / (height - 1), (y1 + y2 - height + 1) /
        (height - 1)], 1).view(-1, 2, 3)
    grid = F.affine_grid(theta, torch.Size((rois.size(0), 1, grid_size,
        grid_size)))
    return grid


_global_config['TRAIN'] = 4


_global_config['POOLING_MODE'] = 4


_global_config['POOLING_SIZE'] = 4


_global_config['CROP_RESIZE_WITH_MAX_POOL'] = 4


class _fasterRCNN(nn.Module):
    """ faster RCNN """

    def __init__(self, classes, class_agnostic):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE,
            1.0 / 16.0)
        self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.
            POOLING_SIZE, 1.0 / 16.0)
        self.grid_size = (cfg.POOLING_SIZE * 2 if cfg.
            CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE)
        self.RCNN_roi_crop = _RoICrop()

    def forward(self, im_data, im_info, gt_boxes, num_boxes):
        batch_size = im_data.size(0)
        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data
        base_feat = self.RCNN_base(im_data)
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat,
            im_info, gt_boxes, num_boxes)
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
        if cfg.POOLING_MODE == 'crop':
            grid_xy = _affine_grid_gen(rois.view(-1, 5), base_feat.size()[2
                :], self.grid_size)
            grid_yx = torch.stack([grid_xy.data[:, :, :, (1)], grid_xy.data
                [:, :, :, (0)]], 3).contiguous()
            pooled_feat = self.RCNN_roi_crop(base_feat, Variable(grid_yx).
                detach())
            if cfg.CROP_RESIZE_WITH_MAX_POOL:
                pooled_feat = F.max_pool2d(pooled_feat, 2, 2)
        elif cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1, 5))
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
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target,
                rois_inside_ws, rois_outside_ws)
        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)
        return (rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox,
            RCNN_loss_cls, RCNN_loss_bbox, rois_label)

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
        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()


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


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0,
            ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
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
                nn.BatchNorm2d(planes * block.expansion))
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


class RoIAlignFunction(Function):

    def __init__(self, aligned_height, aligned_width, spatial_scale):
        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)
        self.rois = None
        self.feature_size = None

    def forward(self, features, rois):
        self.rois = rois
        self.feature_size = features.size()
        batch_size, num_channels, data_height, data_width = features.size()
        num_rois = rois.size(0)
        output = features.new(num_rois, num_channels, self.aligned_height,
            self.aligned_width).zero_()
        if features.is_cuda:
            roi_align.roi_align_forward_cuda(self.aligned_height, self.
                aligned_width, self.spatial_scale, features, rois, output)
        else:
            roi_align.roi_align_forward(self.aligned_height, self.
                aligned_width, self.spatial_scale, features, rois, output)
        return output

    def backward(self, grad_output):
        assert self.feature_size is not None and grad_output.is_cuda
        batch_size, num_channels, data_height, data_width = self.feature_size
        grad_input = self.rois.new(batch_size, num_channels, data_height,
            data_width).zero_()
        roi_align.roi_align_backward_cuda(self.aligned_height, self.
            aligned_width, self.spatial_scale, grad_output, self.rois,
            grad_input)
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


class AffineGridGenFunction(Function):

    def __init__(self, height, width, lr=1):
        super(AffineGridGenFunction, self).__init__()
        self.lr = lr
        self.height, self.width = height, width
        self.grid = np.zeros([self.height, self.width, 3], dtype=np.float32)
        self.grid[:, :, (0)] = np.expand_dims(np.repeat(np.expand_dims(np.
            arange(-1, 1, 2.0 / self.height), 0), repeats=self.width, axis=
            0).T, 0)
        self.grid[:, :, (1)] = np.expand_dims(np.repeat(np.expand_dims(np.
            arange(-1, 1, 2.0 / self.width), 0), repeats=self.height, axis=
            0), 0)
        self.grid[:, :, (2)] = np.ones([self.height, width])
        self.grid = torch.from_numpy(self.grid.astype(np.float32))

    def forward(self, input1):
        self.input1 = input1
        output = input1.new(torch.Size([input1.size(0)]) + self.grid.size()
            ).zero_()
        self.batchgrid = input1.new(torch.Size([input1.size(0)]) + self.
            grid.size()).zero_()
        for i in range(input1.size(0)):
            self.batchgrid[i] = self.grid.astype(self.batchgrid[i])
        for i in range(input1.size(0)):
            output = torch.bmm(self.batchgrid.view(-1, self.height * self.
                width, 3), torch.transpose(input1, 1, 2)).view(-1, self.
                height, self.width, 2)
        return output

    def backward(self, grad_output):
        grad_input1 = self.input1.new(self.input1.size()).zero_()
        grad_input1 = torch.baddbmm(grad_input1, torch.transpose(
            grad_output.view(-1, self.height * self.width, 2), 1, 2), self.
            batchgrid.view(-1, self.height * self.width, 3))
        return grad_input1


class _AffineGridGen(Module):

    def __init__(self, height, width, lr=1, aux_loss=False):
        super(_AffineGridGen, self).__init__()
        self.height, self.width = height, width
        self.aux_loss = aux_loss
        self.f = AffineGridGenFunction(self.height, self.width, lr=lr)
        self.lr = lr

    def forward(self, input):
        return self.f(input)


class CylinderGridGen(Module):

    def __init__(self, height, width, lr=1, aux_loss=False):
        super(CylinderGridGen, self).__init__()
        self.height, self.width = height, width
        self.aux_loss = aux_loss
        self.f = CylinderGridGenFunction(self.height, self.width, lr=lr)
        self.lr = lr

    def forward(self, input):
        if not self.aux_loss:
            return self.f(input)
        else:
            return self.f(input), torch.mul(input, input).view(-1, 1)


class AffineGridGenV2(Module):

    def __init__(self, height, width, lr=1, aux_loss=False):
        super(AffineGridGenV2, self).__init__()
        self.height, self.width = height, width
        self.aux_loss = aux_loss
        self.lr = lr
        self.grid = np.zeros([self.height, self.width, 3], dtype=np.float32)
        self.grid[:, :, (0)] = np.expand_dims(np.repeat(np.expand_dims(np.
            arange(-1, 1, 2.0 / self.height), 0), repeats=self.width, axis=
            0).T, 0)
        self.grid[:, :, (1)] = np.expand_dims(np.repeat(np.expand_dims(np.
            arange(-1, 1, 2.0 / self.width), 0), repeats=self.height, axis=
            0), 0)
        self.grid[:, :, (2)] = np.ones([self.height, width])
        self.grid = torch.from_numpy(self.grid.astype(np.float32))

    def forward(self, input1):
        self.batchgrid = torch.zeros(torch.Size([input1.size(0)]) + self.
            grid.size())
        for i in range(input1.size(0)):
            self.batchgrid[i] = self.grid
        self.batchgrid = Variable(self.batchgrid)
        if input1.is_cuda:
            self.batchgrid = self.batchgrid
        output = torch.bmm(self.batchgrid.view(-1, self.height * self.width,
            3), torch.transpose(input1, 1, 2)).view(-1, self.height, self.
            width, 2)
        return output


class CylinderGridGenV2(Module):

    def __init__(self, height, width, lr=1):
        super(CylinderGridGenV2, self).__init__()
        self.height, self.width = height, width
        self.lr = lr
        self.grid = np.zeros([self.height, self.width, 3], dtype=np.float32)
        self.grid[:, :, (0)] = np.expand_dims(np.repeat(np.expand_dims(np.
            arange(-1, 1, 2.0 / self.height), 0), repeats=self.width, axis=
            0).T, 0)
        self.grid[:, :, (1)] = np.expand_dims(np.repeat(np.expand_dims(np.
            arange(-1, 1, 2.0 / self.width), 0), repeats=self.height, axis=
            0), 0)
        self.grid[:, :, (2)] = np.ones([self.height, width])
        self.grid = torch.from_numpy(self.grid.astype(np.float32))

    def forward(self, input):
        self.batchgrid = torch.zeros(torch.Size([input.size(0)]) + self.
            grid.size())
        for i in range(input.size(0)):
            self.batchgrid[(i), :, :, :] = self.grid
        self.batchgrid = Variable(self.batchgrid)
        input_u = input.view(-1, 1, 1, 1).repeat(1, self.height, self.width, 1)
        output0 = self.batchgrid[:, :, :, 0:1]
        output1 = torch.atan(torch.tan(np.pi / 2.0 * (self.batchgrid[:, :,
            :, 1:2] + self.batchgrid[:, :, :, 2:] * input_u[:, :, :, :]))) / (
            np.pi / 2)
        output = torch.cat([output0, output1], 3)
        return output


class DenseAffineGridGen(Module):

    def __init__(self, height, width, lr=1, aux_loss=False):
        super(DenseAffineGridGen, self).__init__()
        self.height, self.width = height, width
        self.aux_loss = aux_loss
        self.lr = lr
        self.grid = np.zeros([self.height, self.width, 3], dtype=np.float32)
        self.grid[:, :, (0)] = np.expand_dims(np.repeat(np.expand_dims(np.
            arange(-1, 1, 2.0 / self.height), 0), repeats=self.width, axis=
            0).T, 0)
        self.grid[:, :, (1)] = np.expand_dims(np.repeat(np.expand_dims(np.
            arange(-1, 1, 2.0 / self.width), 0), repeats=self.height, axis=
            0), 0)
        self.grid[:, :, (2)] = np.ones([self.height, width])
        self.grid = torch.from_numpy(self.grid.astype(np.float32))

    def forward(self, input1):
        self.batchgrid = torch.zeros(torch.Size([input1.size(0)]) + self.
            grid.size())
        for i in range(input1.size(0)):
            self.batchgrid[i] = self.grid
        self.batchgrid = Variable(self.batchgrid)
        x = torch.mul(self.batchgrid, input1[:, :, :, 0:3])
        y = torch.mul(self.batchgrid, input1[:, :, :, 3:6])
        output = torch.cat([torch.sum(x, 3), torch.sum(y, 3)], 3)
        return output


class DenseAffine3DGridGen(Module):

    def __init__(self, height, width, lr=1, aux_loss=False):
        super(DenseAffine3DGridGen, self).__init__()
        self.height, self.width = height, width
        self.aux_loss = aux_loss
        self.lr = lr
        self.grid = np.zeros([self.height, self.width, 3], dtype=np.float32)
        self.grid[:, :, (0)] = np.expand_dims(np.repeat(np.expand_dims(np.
            arange(-1, 1, 2.0 / self.height), 0), repeats=self.width, axis=
            0).T, 0)
        self.grid[:, :, (1)] = np.expand_dims(np.repeat(np.expand_dims(np.
            arange(-1, 1, 2.0 / self.width), 0), repeats=self.height, axis=
            0), 0)
        self.grid[:, :, (2)] = np.ones([self.height, width])
        self.grid = torch.from_numpy(self.grid.astype(np.float32))
        self.theta = self.grid[:, :, (0)] * np.pi / 2 + np.pi / 2
        self.phi = self.grid[:, :, (1)] * np.pi
        self.x = torch.sin(self.theta) * torch.cos(self.phi)
        self.y = torch.sin(self.theta) * torch.sin(self.phi)
        self.z = torch.cos(self.theta)
        self.grid3d = torch.from_numpy(np.zeros([self.height, self.width, 4
            ], dtype=np.float32))
        self.grid3d[:, :, (0)] = self.x
        self.grid3d[:, :, (1)] = self.y
        self.grid3d[:, :, (2)] = self.z
        self.grid3d[:, :, (3)] = self.grid[:, :, (2)]

    def forward(self, input1):
        self.batchgrid3d = torch.zeros(torch.Size([input1.size(0)]) + self.
            grid3d.size())
        for i in range(input1.size(0)):
            self.batchgrid3d[i] = self.grid3d
        self.batchgrid3d = Variable(self.batchgrid3d)
        x = torch.sum(torch.mul(self.batchgrid3d, input1[:, :, :, 0:4]), 3)
        y = torch.sum(torch.mul(self.batchgrid3d, input1[:, :, :, 4:8]), 3)
        z = torch.sum(torch.mul(self.batchgrid3d, input1[:, :, :, 8:]), 3)
        r = torch.sqrt(x ** 2 + y ** 2 + z ** 2) + 1e-05
        theta = torch.acos(z / r) / (np.pi / 2) - 1
        phi = torch.atan(y / (x + 1e-05)) + np.pi * x.lt(0).type(torch.
            FloatTensor) * (y.ge(0).type(torch.FloatTensor) - y.lt(0).type(
            torch.FloatTensor))
        phi = phi / np.pi
        output = torch.cat([theta, phi], 3)
        return output


class DenseAffine3DGridGen_rotate(Module):

    def __init__(self, height, width, lr=1, aux_loss=False):
        super(DenseAffine3DGridGen_rotate, self).__init__()
        self.height, self.width = height, width
        self.aux_loss = aux_loss
        self.lr = lr
        self.grid = np.zeros([self.height, self.width, 3], dtype=np.float32)
        self.grid[:, :, (0)] = np.expand_dims(np.repeat(np.expand_dims(np.
            arange(-1, 1, 2.0 / self.height), 0), repeats=self.width, axis=
            0).T, 0)
        self.grid[:, :, (1)] = np.expand_dims(np.repeat(np.expand_dims(np.
            arange(-1, 1, 2.0 / self.width), 0), repeats=self.height, axis=
            0), 0)
        self.grid[:, :, (2)] = np.ones([self.height, width])
        self.grid = torch.from_numpy(self.grid.astype(np.float32))
        self.theta = self.grid[:, :, (0)] * np.pi / 2 + np.pi / 2
        self.phi = self.grid[:, :, (1)] * np.pi
        self.x = torch.sin(self.theta) * torch.cos(self.phi)
        self.y = torch.sin(self.theta) * torch.sin(self.phi)
        self.z = torch.cos(self.theta)
        self.grid3d = torch.from_numpy(np.zeros([self.height, self.width, 4
            ], dtype=np.float32))
        self.grid3d[:, :, (0)] = self.x
        self.grid3d[:, :, (1)] = self.y
        self.grid3d[:, :, (2)] = self.z
        self.grid3d[:, :, (3)] = self.grid[:, :, (2)]

    def forward(self, input1, input2):
        self.batchgrid3d = torch.zeros(torch.Size([input1.size(0)]) + self.
            grid3d.size())
        for i in range(input1.size(0)):
            self.batchgrid3d[i] = self.grid3d
        self.batchgrid3d = Variable(self.batchgrid3d)
        self.batchgrid = torch.zeros(torch.Size([input1.size(0)]) + self.
            grid.size())
        for i in range(input1.size(0)):
            self.batchgrid[i] = self.grid
        self.batchgrid = Variable(self.batchgrid)
        x = torch.sum(torch.mul(self.batchgrid3d, input1[:, :, :, 0:4]), 3)
        y = torch.sum(torch.mul(self.batchgrid3d, input1[:, :, :, 4:8]), 3)
        z = torch.sum(torch.mul(self.batchgrid3d, input1[:, :, :, 8:]), 3)
        r = torch.sqrt(x ** 2 + y ** 2 + z ** 2) + 1e-05
        theta = torch.acos(z / r) / (np.pi / 2) - 1
        phi = torch.atan(y / (x + 1e-05)) + np.pi * x.lt(0).type(torch.
            FloatTensor) * (y.ge(0).type(torch.FloatTensor) - y.lt(0).type(
            torch.FloatTensor))
        phi = phi / np.pi
        input_u = input2.view(-1, 1, 1, 1).repeat(1, self.height, self.width, 1
            )
        output = torch.cat([theta, phi], 3)
        output1 = torch.atan(torch.tan(np.pi / 2.0 * (output[:, :, :, 1:2] +
            self.batchgrid[:, :, :, 2:] * input_u[:, :, :, :]))) / (np.pi / 2)
        output2 = torch.cat([output[:, :, :, 0:1], output1], 3)
        return output2


class Depth3DGridGen(Module):

    def __init__(self, height, width, lr=1, aux_loss=False):
        super(Depth3DGridGen, self).__init__()
        self.height, self.width = height, width
        self.aux_loss = aux_loss
        self.lr = lr
        self.grid = np.zeros([self.height, self.width, 3], dtype=np.float32)
        self.grid[:, :, (0)] = np.expand_dims(np.repeat(np.expand_dims(np.
            arange(-1, 1, 2.0 / self.height), 0), repeats=self.width, axis=
            0).T, 0)
        self.grid[:, :, (1)] = np.expand_dims(np.repeat(np.expand_dims(np.
            arange(-1, 1, 2.0 / self.width), 0), repeats=self.height, axis=
            0), 0)
        self.grid[:, :, (2)] = np.ones([self.height, width])
        self.grid = torch.from_numpy(self.grid.astype(np.float32))
        self.theta = self.grid[:, :, (0)] * np.pi / 2 + np.pi / 2
        self.phi = self.grid[:, :, (1)] * np.pi
        self.x = torch.sin(self.theta) * torch.cos(self.phi)
        self.y = torch.sin(self.theta) * torch.sin(self.phi)
        self.z = torch.cos(self.theta)
        self.grid3d = torch.from_numpy(np.zeros([self.height, self.width, 4
            ], dtype=np.float32))
        self.grid3d[:, :, (0)] = self.x
        self.grid3d[:, :, (1)] = self.y
        self.grid3d[:, :, (2)] = self.z
        self.grid3d[:, :, (3)] = self.grid[:, :, (2)]

    def forward(self, depth, trans0, trans1, rotate):
        self.batchgrid3d = torch.zeros(torch.Size([depth.size(0)]) + self.
            grid3d.size())
        for i in range(depth.size(0)):
            self.batchgrid3d[i] = self.grid3d
        self.batchgrid3d = Variable(self.batchgrid3d)
        self.batchgrid = torch.zeros(torch.Size([depth.size(0)]) + self.
            grid.size())
        for i in range(depth.size(0)):
            self.batchgrid[i] = self.grid
        self.batchgrid = Variable(self.batchgrid)
        x = self.batchgrid3d[:, :, :, 0:1] * depth + trans0.view(-1, 1, 1, 1
            ).repeat(1, self.height, self.width, 1)
        y = self.batchgrid3d[:, :, :, 1:2] * depth + trans1.view(-1, 1, 1, 1
            ).repeat(1, self.height, self.width, 1)
        z = self.batchgrid3d[:, :, :, 2:3] * depth
        r = torch.sqrt(x ** 2 + y ** 2 + z ** 2) + 1e-05
        theta = torch.acos(z / r) / (np.pi / 2) - 1
        phi = torch.atan(y / (x + 1e-05)) + np.pi * x.lt(0).type(torch.
            FloatTensor) * (y.ge(0).type(torch.FloatTensor) - y.lt(0).type(
            torch.FloatTensor))
        phi = phi / np.pi
        input_u = rotate.view(-1, 1, 1, 1).repeat(1, self.height, self.width, 1
            )
        output = torch.cat([theta, phi], 3)
        output1 = torch.atan(torch.tan(np.pi / 2.0 * (output[:, :, :, 1:2] +
            self.batchgrid[:, :, :, 2:] * input_u[:, :, :, :]))) / (np.pi / 2)
        output2 = torch.cat([output[:, :, :, 0:1], output1], 3)
        return output2


class Depth3DGridGen_with_mask(Module):

    def __init__(self, height, width, lr=1, aux_loss=False, ray_tracing=False):
        super(Depth3DGridGen_with_mask, self).__init__()
        self.height, self.width = height, width
        self.aux_loss = aux_loss
        self.lr = lr
        self.ray_tracing = ray_tracing
        self.grid = np.zeros([self.height, self.width, 3], dtype=np.float32)
        self.grid[:, :, (0)] = np.expand_dims(np.repeat(np.expand_dims(np.
            arange(-1, 1, 2.0 / self.height), 0), repeats=self.width, axis=
            0).T, 0)
        self.grid[:, :, (1)] = np.expand_dims(np.repeat(np.expand_dims(np.
            arange(-1, 1, 2.0 / self.width), 0), repeats=self.height, axis=
            0), 0)
        self.grid[:, :, (2)] = np.ones([self.height, width])
        self.grid = torch.from_numpy(self.grid.astype(np.float32))
        self.theta = self.grid[:, :, (0)] * np.pi / 2 + np.pi / 2
        self.phi = self.grid[:, :, (1)] * np.pi
        self.x = torch.sin(self.theta) * torch.cos(self.phi)
        self.y = torch.sin(self.theta) * torch.sin(self.phi)
        self.z = torch.cos(self.theta)
        self.grid3d = torch.from_numpy(np.zeros([self.height, self.width, 4
            ], dtype=np.float32))
        self.grid3d[:, :, (0)] = self.x
        self.grid3d[:, :, (1)] = self.y
        self.grid3d[:, :, (2)] = self.z
        self.grid3d[:, :, (3)] = self.grid[:, :, (2)]

    def forward(self, depth, trans0, trans1, rotate):
        self.batchgrid3d = torch.zeros(torch.Size([depth.size(0)]) + self.
            grid3d.size())
        for i in range(depth.size(0)):
            self.batchgrid3d[i] = self.grid3d
        self.batchgrid3d = Variable(self.batchgrid3d)
        self.batchgrid = torch.zeros(torch.Size([depth.size(0)]) + self.
            grid.size())
        for i in range(depth.size(0)):
            self.batchgrid[i] = self.grid
        self.batchgrid = Variable(self.batchgrid)
        if depth.is_cuda:
            self.batchgrid = self.batchgrid
            self.batchgrid3d = self.batchgrid3d
        x_ = self.batchgrid3d[:, :, :, 0:1] * depth + trans0.view(-1, 1, 1, 1
            ).repeat(1, self.height, self.width, 1)
        y_ = self.batchgrid3d[:, :, :, 1:2] * depth + trans1.view(-1, 1, 1, 1
            ).repeat(1, self.height, self.width, 1)
        z = self.batchgrid3d[:, :, :, 2:3] * depth
        rotate_z = rotate.view(-1, 1, 1, 1).repeat(1, self.height, self.
            width, 1) * np.pi
        x = x_ * torch.cos(rotate_z) - y_ * torch.sin(rotate_z)
        y = x_ * torch.sin(rotate_z) + y_ * torch.cos(rotate_z)
        r = torch.sqrt(x ** 2 + y ** 2 + z ** 2) + 1e-05
        theta = torch.acos(z / r) / (np.pi / 2) - 1
        if depth.is_cuda:
            phi = torch.atan(y / (x + 1e-05)) + np.pi * x.lt(0).type(torch.
                cuda.FloatTensor) * (y.ge(0).type(torch.cuda.FloatTensor) -
                y.lt(0).type(torch.cuda.FloatTensor))
        else:
            phi = torch.atan(y / (x + 1e-05)) + np.pi * x.lt(0).type(torch.
                FloatTensor) * (y.ge(0).type(torch.FloatTensor) - y.lt(0).
                type(torch.FloatTensor))
        phi = phi / np.pi
        output = torch.cat([theta, phi], 3)
        return output


sources = []


extra_objects = ['src/nms_cuda_kernel.cu.o']


with_cuda = False


headers = []


defines = []


class RoIPoolFunction(Function):

    def __init__(self, pooled_height, pooled_width, spatial_scale):
        self.pooled_width = int(pooled_width)
        self.pooled_height = int(pooled_height)
        self.spatial_scale = float(spatial_scale)
        self.output = None
        self.argmax = None
        self.rois = None
        self.feature_size = None

    def forward(self, features, rois):
        batch_size, num_channels, data_height, data_width = features.size()
        num_rois = rois.size()[0]
        output = torch.zeros(num_rois, num_channels, self.pooled_height,
            self.pooled_width)
        argmax = torch.IntTensor(num_rois, num_channels, self.pooled_height,
            self.pooled_width).zero_()
        if not features.is_cuda:
            _features = features.permute(0, 2, 3, 1)
            roi_pooling.roi_pooling_forward(self.pooled_height, self.
                pooled_width, self.spatial_scale, _features, rois, output)
        else:
            output = output.cuda()
            argmax = argmax.cuda()
            roi_pooling.roi_pooling_forward_cuda(self.pooled_height, self.
                pooled_width, self.spatial_scale, features, rois, output,
                argmax)
            self.output = output
            self.argmax = argmax
            self.rois = rois
            self.feature_size = features.size()
        return output

    def backward(self, grad_output):
        assert self.feature_size is not None and grad_output.is_cuda
        batch_size, num_channels, data_height, data_width = self.feature_size
        grad_input = torch.zeros(batch_size, num_channels, data_height,
            data_width).cuda()
        roi_pooling.roi_pooling_backward_cuda(self.pooled_height, self.
            pooled_width, self.spatial_scale, grad_output, self.rois,
            grad_input, self.argmax)
        return grad_input, None


class _RoIPooling(Module):

    def __init__(self, pooled_height, pooled_width, spatial_scale):
        super(_RoIPooling, self).__init__()
        self.pooled_width = int(pooled_width)
        self.pooled_height = int(pooled_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        return RoIPoolFunction(self.pooled_height, self.pooled_width, self.
            spatial_scale)(features, rois)


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


def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr


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


def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


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


def nms_gpu(dets, thresh):
    keep = dets.new(dets.size(0), 1).zero_().int()
    num_out = dets.new(1).zero_().int()
    nms.nms_cuda(keep, dets, num_out, thresh)
    keep = keep[:num_out[0]]
    return keep


def nms_cpu(dets, thresh):
    dets = dets.numpy()
    x1 = dets[:, (0)]
    y1 = dets[:, (1)]
    x2 = dets[:, (2)]
    y2 = dets[:, (3)]
    scores = dets[:, (4)]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order.item(0)
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return torch.IntTensor(keep)


def nms(dets, thresh, force_cpu=False):
    """Dispatch to either CPU or GPU NMS implementations."""
    if dets.shape[0] == 0:
        return []
    return nms_gpu(dets, thresh) if force_cpu == False else nms_cpu(dets,
        thresh)


def clip_boxes(boxes, im_shape, batch_size):
    for i in range(batch_size):
        boxes[(i), :, 0::4].clamp_(0, im_shape[i, 1] - 1)
        boxes[(i), :, 1::4].clamp_(0, im_shape[i, 0] - 1)
        boxes[(i), :, 2::4].clamp_(0, im_shape[i, 1] - 1)
        boxes[(i), :, 3::4].clamp_(0, im_shape[i, 0] - 1)
    return boxes


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


_global_config['USE_GPU_NMS'] = 4


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
            keep_idx_i = nms(torch.cat((proposals_single, scores_single), 1
                ), nms_thresh, force_cpu=not cfg.USE_GPU_NMS)
            keep_idx_i = keep_idx_i.long().view(-1)
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


_global_config['FEAT_STRIDE'] = 4


_global_config['ANCHOR_SCALES'] = 4


_global_config['ANCHOR_RATIOS'] = 4


class _RPN(nn.Module):
    """ region proposal network """

    def __init__(self, din):
        super(_RPN, self).__init__()
        self.din = din
        self.anchor_scales = cfg.ANCHOR_SCALES
        self.anchor_ratios = cfg.ANCHOR_RATIOS
        self.feat_stride = cfg.FEAT_STRIDE[0]
        self.RPN_Conv = nn.Conv2d(self.din, 512, 3, 1, 1, bias=True)
        self.nc_score_out = len(self.anchor_scales) * len(self.anchor_ratios
            ) * 2
        self.RPN_cls_score = nn.Conv2d(512, self.nc_score_out, 1, 1, 0)
        self.nc_bbox_out = len(self.anchor_scales) * len(self.anchor_ratios
            ) * 4
        self.RPN_bbox_pred = nn.Conv2d(512, self.nc_bbox_out, 1, 1, 0)
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
        rpn_conv1 = F.relu(self.RPN_Conv(base_feat), inplace=True)
        rpn_cls_score = self.RPN_cls_score(rpn_conv1)
        rpn_cls_score_reshape = self.reshape(rpn_cls_score, 2)
        rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape, 1)
        rpn_cls_prob = self.reshape(rpn_cls_prob_reshape, self.nc_score_out)
        rpn_bbox_pred = self.RPN_bbox_pred(rpn_conv1)
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
            rpn_keep = rpn_label.view(-1).ne(-1).nonzero().view(-1)
            rpn_cls_score = torch.index_select(rpn_cls_score.view(-1, 2), 0,
                rpn_keep)
            rpn_label = torch.index_select(rpn_label.view(-1), 0, rpn_keep.data
                )
            rpn_label = rpn_label.long()
            self.rpn_loss_cls = F.cross_entropy(rpn_cls_score, rpn_label)
            fg_cnt = torch.sum(rpn_label.data.ne(0))
            (rpn_bbox_targets, rpn_bbox_inside_weights,
                rpn_bbox_outside_weights) = rpn_data[1:]
            self.rpn_loss_box = _smooth_l1_loss(rpn_bbox_pred,
                rpn_bbox_targets, rpn_bbox_inside_weights,
                rpn_bbox_outside_weights, sigma=3, dim=[1, 2, 3])
        return rois, self.rpn_loss_cls, self.rpn_loss_box


class L2Norm(nn.Module):

    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x
            ) * x
        return out


def intersect(box_a, box_b):
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
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2), box_b[:, 
        2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2), box_b[:,
        :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp(max_xy - min_xy, min=0)
    return inter[:, :, (0)] * inter[:, :, (1)]


def jaccard(box_a, box_b):
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
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, (2)] - box_a[:, (0)]) * (box_a[:, (3)] - box_a[:, (1)])
        ).unsqueeze(1).expand_as(inter)
    area_b = ((box_b[:, (2)] - box_b[:, (0)]) * (box_b[:, (3)] - box_b[:, (1)])
        ).unsqueeze(0).expand_as(inter)
    union = area_a + area_b - inter
    return inter / union


voc = {'num_classes': 21, 'lr_steps': (80000, 100000, 120000), 'max_iter': 
    120000, 'feature_maps': [38, 19, 10, 5, 3, 1], 'min_dim': 300, 'steps':
    [8, 16, 32, 64, 100, 300], 'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315], 'aspect_ratios': [[2], [2, 
    3], [2, 3], [2, 3], [2], [2]], 'variance': [0.1, 0.2], 'clip': True,
    'name': 'VOC'}


def _make_layers(in_channels, net_cfg):
    layers = []
    if len(net_cfg) > 0 and isinstance(net_cfg[0], list):
        for sub_cfg in net_cfg:
            layer, in_channels = _make_layers(in_channels, sub_cfg)
            layers.append(layer)
    else:
        for item in net_cfg:
            if item == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                out_channels, ksize = item
                layers.append(net_utils.Conv2d_BatchNorm(in_channels,
                    out_channels, ksize, same_padding=True))
                in_channels = out_channels
    return nn.Sequential(*layers), in_channels


_global_config['multi_scale_inp_size'] = 1.0


_global_config['iou_thresh'] = 4


_global_config['coord_scale'] = 1.0


_global_config['multi_scale_out_size'] = 1.0


_global_config['class_scale'] = 1.0


_global_config['anchors'] = 4


_global_config['object_scale'] = 1.0


_global_config['noobject_scale'] = 1.0


_global_config['num_classes'] = 4


def _process_batch(data, size_index):
    W, H = cfg.multi_scale_out_size[size_index]
    inp_size = cfg.multi_scale_inp_size[size_index]
    out_size = cfg.multi_scale_out_size[size_index]
    bbox_pred_np, gt_boxes, gt_classes, dontcares, iou_pred_np = data
    hw, num_anchors, _ = bbox_pred_np.shape
    _classes = np.zeros([hw, num_anchors, cfg.num_classes], dtype=np.float)
    _class_mask = np.zeros([hw, num_anchors, 1], dtype=np.float)
    _ious = np.zeros([hw, num_anchors, 1], dtype=np.float)
    _iou_mask = np.zeros([hw, num_anchors, 1], dtype=np.float)
    _boxes = np.zeros([hw, num_anchors, 4], dtype=np.float)
    _boxes[:, :, 0:2] = 0.5
    _boxes[:, :, 2:4] = 1.0
    _box_mask = np.zeros([hw, num_anchors, 1], dtype=np.float) + 0.01
    anchors = np.ascontiguousarray(cfg.anchors, dtype=np.float)
    bbox_pred_np = np.expand_dims(bbox_pred_np, 0)
    bbox_np = yolo_to_bbox(np.ascontiguousarray(bbox_pred_np, dtype=np.
        float), anchors, H, W)
    bbox_np = bbox_np[0]
    bbox_np[:, :, 0::2] *= float(inp_size[0])
    bbox_np[:, :, 1::2] *= float(inp_size[1])
    gt_boxes_b = np.asarray(gt_boxes, dtype=np.float)
    bbox_np_b = np.reshape(bbox_np, [-1, 4])
    ious = bbox_ious(np.ascontiguousarray(bbox_np_b, dtype=np.float), np.
        ascontiguousarray(gt_boxes_b, dtype=np.float))
    best_ious = np.max(ious, axis=1).reshape(_iou_mask.shape)
    iou_penalty = 0 - iou_pred_np[best_ious < cfg.iou_thresh]
    _iou_mask[best_ious <= cfg.iou_thresh] = cfg.noobject_scale * iou_penalty
    cell_w = float(inp_size[0]) / W
    cell_h = float(inp_size[1]) / H
    cx = (gt_boxes_b[:, (0)] + gt_boxes_b[:, (2)]) * 0.5 / cell_w
    cy = (gt_boxes_b[:, (1)] + gt_boxes_b[:, (3)]) * 0.5 / cell_h
    cell_inds = np.floor(cy) * W + np.floor(cx)
    cell_inds = cell_inds.astype(np.int)
    target_boxes = np.empty(gt_boxes_b.shape, dtype=np.float)
    target_boxes[:, (0)] = cx - np.floor(cx)
    target_boxes[:, (1)] = cy - np.floor(cy)
    target_boxes[:, (2)] = (gt_boxes_b[:, (2)] - gt_boxes_b[:, (0)]
        ) / inp_size[0] * out_size[0]
    target_boxes[:, (3)] = (gt_boxes_b[:, (3)] - gt_boxes_b[:, (1)]
        ) / inp_size[1] * out_size[1]
    gt_boxes_resize = np.copy(gt_boxes_b)
    gt_boxes_resize[:, 0::2] *= out_size[0] / float(inp_size[0])
    gt_boxes_resize[:, 1::2] *= out_size[1] / float(inp_size[1])
    anchor_ious = anchor_intersections(anchors, np.ascontiguousarray(
        gt_boxes_resize, dtype=np.float))
    anchor_inds = np.argmax(anchor_ious, axis=0)
    ious_reshaped = np.reshape(ious, [hw, num_anchors, len(cell_inds)])
    for i, cell_ind in enumerate(cell_inds):
        if cell_ind >= hw or cell_ind < 0:
            print('cell inds size {}'.format(len(cell_inds)))
            print('cell over {} hw {}'.format(cell_ind, hw))
            continue
        a = anchor_inds[i]
        iou_pred_cell_anchor = iou_pred_np[(cell_ind), (a), :]
        _iou_mask[(cell_ind), (a), :] = cfg.object_scale * (1 -
            iou_pred_cell_anchor)
        _ious[(cell_ind), (a), :] = ious_reshaped[cell_ind, a, i]
        _box_mask[(cell_ind), (a), :] = cfg.coord_scale
        target_boxes[(i), 2:4] /= anchors[a]
        _boxes[(cell_ind), (a), :] = target_boxes[i]
        _class_mask[(cell_ind), (a), :] = cfg.class_scale
        _classes[cell_ind, a, gt_classes[i]] = 1.0
    return _boxes, _ious, _classes, _box_mask, _iou_mask, _class_mask


_global_config['num_anchors'] = 4


class Darknet19(nn.Module):

    def __init__(self):
        super(Darknet19, self).__init__()
        net_cfgs = [[(32, 3)], ['M', (64, 3)], ['M', (128, 3), (64, 1), (
            128, 3)], ['M', (256, 3), (128, 1), (256, 3)], ['M', (512, 3),
            (256, 1), (512, 3), (256, 1), (512, 3)], ['M', (1024, 3), (512,
            1), (1024, 3), (512, 1), (1024, 3)], [(1024, 3), (1024, 3)], [(
            1024, 3)]]
        self.conv1s, c1 = _make_layers(3, net_cfgs[0:5])
        self.conv2, c2 = _make_layers(c1, net_cfgs[5])
        self.conv3, c3 = _make_layers(c2, net_cfgs[6])
        stride = 2
        self.reorg = ReorgLayer(stride=2)
        self.conv4, c4 = _make_layers(c1 * (stride * stride) + c3, net_cfgs[7])
        out_channels = cfg.num_anchors * (cfg.num_classes + 5)
        self.conv5 = net_utils.Conv2d(c4, out_channels, 1, 1, relu=False)
        self.global_average_pool = nn.AvgPool2d((1, 1))
        self.bbox_loss = None
        self.iou_loss = None
        self.cls_loss = None
        self.pool = Pool(processes=10)

    @property
    def loss(self):
        return self.bbox_loss + self.iou_loss + self.cls_loss

    def forward(self, im_data, gt_boxes=None, gt_classes=None, dontcare=
        None, size_index=0):
        conv1s = self.conv1s(im_data)
        conv2 = self.conv2(conv1s)
        conv3 = self.conv3(conv2)
        conv1s_reorg = self.reorg(conv1s)
        cat_1_3 = torch.cat([conv1s_reorg, conv3], 1)
        conv4 = self.conv4(cat_1_3)
        conv5 = self.conv5(conv4)
        global_average_pool = self.global_average_pool(conv5)
        bsize, _, h, w = global_average_pool.size()
        global_average_pool_reshaped = global_average_pool.permute(0, 2, 3, 1
            ).contiguous().view(bsize, -1, cfg.num_anchors, cfg.num_classes + 5
            )
        xy_pred = F.sigmoid(global_average_pool_reshaped[:, :, :, 0:2])
        wh_pred = torch.exp(global_average_pool_reshaped[:, :, :, 2:4])
        bbox_pred = torch.cat([xy_pred, wh_pred], 3)
        iou_pred = F.sigmoid(global_average_pool_reshaped[:, :, :, 4:5])
        score_pred = global_average_pool_reshaped[:, :, :, 5:].contiguous()
        prob_pred = F.softmax(score_pred.view(-1, score_pred.size()[-1]), dim=1
            ).view_as(score_pred)
        if self.training:
            bbox_pred_np = bbox_pred.data.cpu().numpy()
            iou_pred_np = iou_pred.data.cpu().numpy()
            _boxes, _ious, _classes, _box_mask, _iou_mask, _class_mask = (self
                ._build_target(bbox_pred_np, gt_boxes, gt_classes, dontcare,
                iou_pred_np, size_index))
            _boxes = net_utils.np_to_variable(_boxes)
            _ious = net_utils.np_to_variable(_ious)
            _classes = net_utils.np_to_variable(_classes)
            box_mask = net_utils.np_to_variable(_box_mask, dtype=torch.
                FloatTensor)
            iou_mask = net_utils.np_to_variable(_iou_mask, dtype=torch.
                FloatTensor)
            class_mask = net_utils.np_to_variable(_class_mask, dtype=torch.
                FloatTensor)
            num_boxes = sum(len(boxes) for boxes in gt_boxes)
            box_mask = box_mask.expand_as(_boxes)
            self.bbox_loss = nn.MSELoss(size_average=False)(bbox_pred *
                box_mask, _boxes * box_mask) / num_boxes
            self.iou_loss = nn.MSELoss(size_average=False)(iou_pred *
                iou_mask, _ious * iou_mask) / num_boxes
            class_mask = class_mask.expand_as(prob_pred)
            self.cls_loss = nn.MSELoss(size_average=False)(prob_pred *
                class_mask, _classes * class_mask) / num_boxes
        return bbox_pred, iou_pred, prob_pred

    def _build_target(self, bbox_pred_np, gt_boxes, gt_classes, dontcare,
        iou_pred_np, size_index):
        """
        :param bbox_pred: shape: (bsize, h x w, num_anchors, 4) :
                          (sig(tx), sig(ty), exp(tw), exp(th))
        """
        bsize = bbox_pred_np.shape[0]
        targets = self.pool.map(partial(_process_batch, size_index=
            size_index), ((bbox_pred_np[b], gt_boxes[b], gt_classes[b],
            dontcare[b], iou_pred_np[b]) for b in range(bsize)))
        _boxes = np.stack(tuple(row[0] for row in targets))
        _ious = np.stack(tuple(row[1] for row in targets))
        _classes = np.stack(tuple(row[2] for row in targets))
        _box_mask = np.stack(tuple(row[3] for row in targets))
        _iou_mask = np.stack(tuple(row[4] for row in targets))
        _class_mask = np.stack(tuple(row[5] for row in targets))
        return _boxes, _ious, _classes, _box_mask, _iou_mask, _class_mask

    def load_from_npz(self, fname, num_conv=None):
        dest_src = {'conv.weight': 'kernel', 'conv.bias': 'biases',
            'bn.weight': 'gamma', 'bn.bias': 'biases', 'bn.running_mean':
            'moving_mean', 'bn.running_var': 'moving_variance'}
        params = np.load(fname)
        own_dict = self.state_dict()
        keys = list(own_dict.keys())
        for i, start in enumerate(range(0, len(keys), 5)):
            if num_conv is not None and i >= num_conv:
                break
            end = min(start + 5, len(keys))
            for key in keys[start:end]:
                list_key = key.split('.')
                ptype = dest_src['{}.{}'.format(list_key[-2], list_key[-1])]
                src_key = '{}-convolutional/{}:0'.format(i, ptype)
                None
                param = torch.from_numpy(params[src_key])
                if ptype == 'kernel':
                    param = param.permute(3, 2, 0, 1)
                own_dict[key].copy_(param)


class ReorgFunction(Function):

    def __init__(self, stride=2):
        self.stride = stride

    def forward(self, x):
        stride = self.stride
        bsize, c, h, w = x.size()
        out_w, out_h, out_c = int(w / stride), int(h / stride), c * (stride *
            stride)
        out = torch.FloatTensor(bsize, out_c, out_h, out_w)
        if x.is_cuda:
            out = out.cuda()
            reorg_layer.reorg_cuda(x, out_w, out_h, out_c, bsize, stride, 0,
                out)
        else:
            reorg_layer.reorg_cpu(x, out_w, out_h, out_c, bsize, stride, 0, out
                )
        return out

    def backward(self, grad_top):
        stride = self.stride
        bsize, c, h, w = grad_top.size()
        out_w, out_h, out_c = w * stride, h * stride, c / (stride * stride)
        grad_bottom = torch.FloatTensor(bsize, int(out_c), out_h, out_w)
        if grad_top.is_cuda:
            grad_bottom = grad_bottom.cuda()
            reorg_layer.reorg_cuda(grad_top, w, h, c, bsize, stride, 1,
                grad_bottom)
        else:
            reorg_layer.reorg_cpu(grad_top, w, h, c, bsize, stride, 1,
                grad_bottom)
        return grad_bottom


class ReorgLayer(torch.nn.Module):

    def __init__(self, stride):
        super(ReorgLayer, self).__init__()
        self.stride = stride

    def forward(self, x):
        x = ReorgFunction(self.stride)(x)
        return x


class RoIPool(torch.nn.Module):

    def __init__(self, pooled_height, pooled_width, spatial_scale):
        super(RoIPool, self).__init__()
        self.pooled_width = int(pooled_width)
        self.pooled_height = int(pooled_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        return RoIPoolFunction(self.pooled_height, self.pooled_width, self.
            spatial_scale)(features, rois)


class RoIPool(nn.Module):

    def __init__(self, pooled_height, pooled_width, spatial_scale):
        super(RoIPool, self).__init__()
        self.pooled_width = int(pooled_width)
        self.pooled_height = int(pooled_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        batch_size, num_channels, data_height, data_width = features.size()
        num_rois = rois.size()[0]
        outputs = Variable(torch.zeros(num_rois, num_channels, self.
            pooled_height, self.pooled_width))
        for roi_ind, roi in enumerate(rois):
            batch_ind = int(roi[0].data[0])
            roi_start_w, roi_start_h, roi_end_w, roi_end_h = np.round(roi[1
                :].data.cpu().numpy() * self.spatial_scale).astype(int)
            roi_width = max(roi_end_w - roi_start_w + 1, 1)
            roi_height = max(roi_end_h - roi_start_h + 1, 1)
            bin_size_w = float(roi_width) / float(self.pooled_width)
            bin_size_h = float(roi_height) / float(self.pooled_height)
            for ph in range(self.pooled_height):
                hstart = int(np.floor(ph * bin_size_h))
                hend = int(np.ceil((ph + 1) * bin_size_h))
                hstart = min(data_height, max(0, hstart + roi_start_h))
                hend = min(data_height, max(0, hend + roi_start_h))
                for pw in range(self.pooled_width):
                    wstart = int(np.floor(pw * bin_size_w))
                    wend = int(np.ceil((pw + 1) * bin_size_w))
                    wstart = min(data_width, max(0, wstart + roi_start_w))
                    wend = min(data_width, max(0, wend + roi_start_w))
                    is_empty = hend <= hstart or wend <= wstart
                    if is_empty:
                        outputs[(roi_ind), :, (ph), (pw)] = 0
                    else:
                        data = features[batch_ind]
                        outputs[(roi_ind), :, (ph), (pw)] = torch.max(torch
                            .max(data[:, hstart:hend, wstart:wend], 1)[0], 2)[0
                            ].view(-1)
        return outputs


class Conv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        relu=True, same_padding=False):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
            stride, padding=padding)
        self.relu = nn.LeakyReLU(0.1, inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Conv2d_BatchNorm(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        relu=True, same_padding=False):
        super(Conv2d_BatchNorm, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
            stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.01)
        self.relu = nn.LeakyReLU(0.1, inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class FC(nn.Module):

    def __init__(self, in_features, out_features, relu=True):
        super(FC, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.fc(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class MobileNet(nn.Module):

    def __init__(self):
        super(MobileNet, self).__init__()

        def conv_bn(dim_in, dim_out, stride):
            return nn.Sequential(nn.Conv2d(dim_in, dim_out, 3, stride, 1,
                bias=False), nn.BatchNorm2d(dim_out), nn.ReLU(inplace=True))

        def conv_dw(dim_in, dim_out, stride):
            return nn.Sequential(nn.Conv2d(dim_in, dim_in, 3, stride, 1,
                groups=dim_in, bias=False), nn.BatchNorm2d(dim_in), nn.ReLU
                (inplace=True), nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=
                False), nn.BatchNorm2d(dim_out), nn.ReLU(inplace=True))
        self.model = nn.Sequential(conv_bn(3, 32, 2), conv_dw(32, 64, 1),
            conv_dw(64, 128, 2), conv_dw(128, 128, 1), conv_dw(128, 256, 2),
            conv_dw(256, 256, 1), conv_dw(256, 512, 2), conv_dw(512, 512, 1
            ), conv_dw(512, 512, 1), conv_dw(512, 512, 1), conv_dw(512, 512,
            1), conv_dw(512, 512, 1), conv_dw(512, 1024, 2), conv_dw(1024, 
            1024, 1), nn.AvgPool2d(7))
        self.fc = nn.Linear(1024, 1000)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x


class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup
        if expand_ratio == 1:
            self.conv = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, 3,
                stride, 1, groups=hidden_dim, bias=False), nn.BatchNorm2d(
                hidden_dim), nn.ReLU6(inplace=True), nn.Conv2d(hidden_dim,
                oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup))
        else:
            self.conv = nn.Sequential(nn.Conv2d(inp, hidden_dim, 1, 1, 0,
                bias=False), nn.BatchNorm2d(hidden_dim), nn.ReLU6(inplace=
                True), nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1,
                groups=hidden_dim, bias=False), nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True), nn.Conv2d(hidden_dim, oup, 1, 1, 0,
                bias=False), nn.BatchNorm2d(oup))

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


def conv_1x1_bn(inp, oup):
    return nn.Sequential(nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.
        BatchNorm2d(oup), nn.ReLU6(inplace=True))


def conv_bn(inp, oup, stride):
    return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.
        BatchNorm2d(oup), nn.ReLU6(inplace=True))


class MobileNetV2(nn.Module):

    def __init__(self, n_class=1000, input_size=224, width_mult=1.0):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [[1, 16, 1, 1], [6, 24, 2, 2], [6, 
            32, 3, 2], [6, 64, 4, 2], [6, 96, 3, 1], [6, 160, 3, 2], [6, 
            320, 1, 1]]
        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult
            ) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel,
                        output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel,
                        output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        self.features = nn.Sequential(*self.features)
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(self.
            last_channel, n_class))
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        hidden_dim = round(inp * expand_ratio)
        self.conv = nn.Sequential(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=
            False), nn.BatchNorm2d(hidden_dim), nn.ReLU6(inplace=True), nn.
            Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim,
            bias=False), nn.BatchNorm2d(hidden_dim), nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), nn.BatchNorm2d
            (oup))

    def forward(self, x):
        return x + self.conv(x)


class ShuffleNet(nn.Module):

    def __init__(self, groups=3, in_channels=3, num_classes=1000):
        super(ShuffleNet, self).__init__()
        self.groups = groups
        self.stage_repeats = [3, 7, 3]
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.stage_out_channels = [-1, 24, 240, 480, 960]
        self.conv1 = conv3x3(self.in_channels, self.stage_out_channels[1],
            stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stage2 = self._make_stage(2)
        self.stage3 = self._make_stage(3)
        self.stage4 = self._make_stage(4)
        num_inputs = self.stage_out_channels[-1]
        self.fc = nn.Linear(num_inputs, self.num_classes)

    def _make_stage(self, stage):
        modules = OrderedDict()
        stage_name = 'ShuffleUnit_Stage{}'.format(stage)
        grouped_conv = stage > 2
        first_module = ShuffleUnit(self.stage_out_channels[stage - 1], self
            .stage_out_channels[stage], groups=self.groups, grouped_conv=
            grouped_conv, combine='concat')
        modules[stage_name + '_0'] = first_module
        for i in range(self.stage_repeats[stage - 2]):
            name = stage_name + '_{}'.format(i + 1)
            module = ShuffleUnit(self.stage_out_channels[stage], self.
                stage_out_channels[stage], groups=self.groups, grouped_conv
                =True, combine='add')
            modules[name] = module
        return nn.Sequential(modules)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = F.avg_pool2d(x, x.data.size()[-2:])
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes, expand_planes):
        super(Fire, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1,
            stride=1)
        self.bn1 = nn.BatchNorm2d(squeeze_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(squeeze_planes, expand_planes, kernel_size=1,
            stride=1)
        self.bn2 = nn.BatchNorm2d(expand_planes)
        self.conv3 = nn.Conv2d(squeeze_planes, expand_planes, kernel_size=3,
            stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(expand_planes)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        out1 = self.conv2(x)
        out1 = self.bn2(out1)
        out2 = self.conv3(x)
        out2 = self.bn3(out2)
        out = torch.cat([out1, out2], 1)
        out = self.relu2(out)
        return out


class RetinaNet(nn.Module):
    num_anchors = 9

    def __init__(self, num_classes=20):
        super(RetinaNet, self).__init__()
        self.fpn = FPN50()
        self.num_classes = num_classes
        self.loc_head = self._make_head(self.num_anchors * 4)
        self.cls_head = self._make_head(self.num_anchors * self.num_classes)

    def forward(self, x):
        fms = self.fpn(x)
        loc_preds = []
        cls_preds = []
        for fm in fms:
            loc_pred = self.loc_head(fm)
            cls_pred = self.cls_head(fm)
            loc_pred = loc_pred.permute(0, 2, 3, 1).contiguous().view(x.
                size(0), -1, 4)
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(x.
                size(0), -1, self.num_classes)
            loc_preds.append(loc_pred)
            cls_preds.append(cls_pred)
        return torch.cat(loc_preds, 1), torch.cat(cls_preds, 1)

    def _make_head(self, out_planes):
        layers = []
        for _ in range(4):
            layers.append(nn.Conv2d(256, 256, kernel_size=3, stride=1,
                padding=1))
            layers.append(nn.ReLU(True))
        layers.append(nn.Conv2d(256, out_planes, kernel_size=3, stride=1,
            padding=1))
        return nn.Sequential(*layers)

    def freeze_bn(self):
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_dongdonghy_Detection_PyTorch_Notebook(_paritybench_base):
    pass
    def test_000(self):
        self._check(MLP(*[], **{'in_dim': 4, 'hid_dim1': 4, 'hid_dim2': 4, 'out_dim': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(Linear(*[], **{'in_dim': 4, 'out_dim': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(Perception(*[], **{'in_dim': 4, 'hid_dim': 4, 'out_dim': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_003(self):
        self._check(DetBottleneck(*[], **{'inplanes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_004(self):
        self._check(FPN(*[], **{'layers': [4, 4, 4, 4]}), [torch.rand([4, 3, 64, 64])], {})

    def test_005(self):
        self._check(BasicConv2d(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_006(self):
        self._check(Inceptionv1(*[], **{'in_dim': 4, 'hid_1_1': 4, 'hid_2_1': 4, 'hid_2_3': 4, 'hid_3_1': 4, 'out_3_5': 4, 'out_4_1': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_007(self):
        self._check(Inceptionv2(*[], **{}), [torch.rand([4, 192, 64, 64])], {})

    def test_008(self):
        self._check(BasicBlock(*[], **{'inplanes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_009(self):
        self._check(Depth3DGridGen(*[], **{'height': 4, 'width': 4}), [torch.rand([256, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_010(self):
        self._check(Depth3DGridGen_with_mask(*[], **{'height': 4, 'width': 4}), [torch.rand([256, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_011(self):
        self._check(L2Norm(*[], **{'n_channels': 4, 'scale': 1.0}), [torch.rand([4, 4, 4, 4])], {})

    def test_012(self):
        self._check(Conv2d(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_013(self):
        self._check(Conv2d_BatchNorm(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_014(self):
        self._check(FC(*[], **{'in_features': 4, 'out_features': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_015(self):
        self._check(InvertedResidual(*[], **{'inp': 4, 'oup': 4, 'stride': 1, 'expand_ratio': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_016(self):
        self._check(Fire(*[], **{'inplanes': 4, 'squeeze_planes': 4, 'expand_planes': 4}), [torch.rand([4, 4, 4, 4])], {})

