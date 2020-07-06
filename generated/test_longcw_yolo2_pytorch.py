import sys
_module = sys.modules[__name__]
del sys
cfgs = _module
config = _module
config_voc = _module
exps = _module
darknet19_exp1 = _module
darknet19_exp2 = _module
darknet = _module
datasets = _module
imdb = _module
pascal_voc = _module
voc_eval = _module
demo = _module
layers = _module
reorg = _module
_ext = _module
reorg_layer = _module
build = _module
reorg_layer = _module
roi_pooling = _module
roi_pooling = _module
build = _module
roi_pool = _module
roi_pool_py = _module
test = _module
train = _module
utils = _module
im_transform = _module
network = _module
nms = _module
py_cpu_nms = _module
nms_wrapper = _module
pycocotools = _module
coco = _module
cocoeval = _module
mask = _module
timer = _module
yolo = _module

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


import torch


import numpy as np


import torch.nn as nn


import torch.nn.functional as F


from functools import partial


from torch.multiprocessing import Pool


from torch.autograd import Function


from torch.autograd import Variable


from random import randint


class ReorgFunction(Function):

    def __init__(self, stride=2):
        self.stride = stride

    def forward(self, x):
        stride = self.stride
        bsize, c, h, w = x.size()
        out_w, out_h, out_c = int(w / stride), int(h / stride), c * (stride * stride)
        out = torch.FloatTensor(bsize, out_c, out_h, out_w)
        if x.is_cuda:
            out = out
            reorg_layer.reorg_cuda(x, out_w, out_h, out_c, bsize, stride, 0, out)
        else:
            reorg_layer.reorg_cpu(x, out_w, out_h, out_c, bsize, stride, 0, out)
        return out

    def backward(self, grad_top):
        stride = self.stride
        bsize, c, h, w = grad_top.size()
        out_w, out_h, out_c = w * stride, h * stride, c / (stride * stride)
        grad_bottom = torch.FloatTensor(bsize, int(out_c), out_h, out_w)
        if grad_top.is_cuda:
            grad_bottom = grad_bottom
            reorg_layer.reorg_cuda(grad_top, w, h, c, bsize, stride, 1, grad_bottom)
        else:
            reorg_layer.reorg_cpu(grad_top, w, h, c, bsize, stride, 1, grad_bottom)
        return grad_bottom


class ReorgLayer(torch.nn.Module):

    def __init__(self, stride):
        super(ReorgLayer, self).__init__()
        self.stride = stride

    def forward(self, x):
        x = ReorgFunction(self.stride)(x)
        return x


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
                layers.append(net_utils.Conv2d_BatchNorm(in_channels, out_channels, ksize, same_padding=True))
                in_channels = out_channels
    return nn.Sequential(*layers), in_channels


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
    bbox_np = yolo_to_bbox(np.ascontiguousarray(bbox_pred_np, dtype=np.float), anchors, H, W)
    bbox_np = bbox_np[0]
    bbox_np[:, :, 0::2] *= float(inp_size[0])
    bbox_np[:, :, 1::2] *= float(inp_size[1])
    gt_boxes_b = np.asarray(gt_boxes, dtype=np.float)
    bbox_np_b = np.reshape(bbox_np, [-1, 4])
    ious = bbox_ious(np.ascontiguousarray(bbox_np_b, dtype=np.float), np.ascontiguousarray(gt_boxes_b, dtype=np.float))
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
    target_boxes[:, (2)] = (gt_boxes_b[:, (2)] - gt_boxes_b[:, (0)]) / inp_size[0] * out_size[0]
    target_boxes[:, (3)] = (gt_boxes_b[:, (3)] - gt_boxes_b[:, (1)]) / inp_size[1] * out_size[1]
    gt_boxes_resize = np.copy(gt_boxes_b)
    gt_boxes_resize[:, 0::2] *= out_size[0] / float(inp_size[0])
    gt_boxes_resize[:, 1::2] *= out_size[1] / float(inp_size[1])
    anchor_ious = anchor_intersections(anchors, np.ascontiguousarray(gt_boxes_resize, dtype=np.float))
    anchor_inds = np.argmax(anchor_ious, axis=0)
    ious_reshaped = np.reshape(ious, [hw, num_anchors, len(cell_inds)])
    for i, cell_ind in enumerate(cell_inds):
        if cell_ind >= hw or cell_ind < 0:
            None
            None
            continue
        a = anchor_inds[i]
        iou_pred_cell_anchor = iou_pred_np[(cell_ind), (a), :]
        _iou_mask[(cell_ind), (a), :] = cfg.object_scale * (1 - iou_pred_cell_anchor)
        _ious[(cell_ind), (a), :] = ious_reshaped[cell_ind, a, i]
        _box_mask[(cell_ind), (a), :] = cfg.coord_scale
        target_boxes[(i), 2:4] /= anchors[a]
        _boxes[(cell_ind), (a), :] = target_boxes[i]
        _class_mask[(cell_ind), (a), :] = cfg.class_scale
        _classes[cell_ind, a, gt_classes[i]] = 1.0
    return _boxes, _ious, _classes, _box_mask, _iou_mask, _class_mask


class Darknet19(nn.Module):

    def __init__(self):
        super(Darknet19, self).__init__()
        net_cfgs = [[(32, 3)], ['M', (64, 3)], ['M', (128, 3), (64, 1), (128, 3)], ['M', (256, 3), (128, 1), (256, 3)], ['M', (512, 3), (256, 1), (512, 3), (256, 1), (512, 3)], ['M', (1024, 3), (512, 1), (1024, 3), (512, 1), (1024, 3)], [(1024, 3), (1024, 3)], [(1024, 3)]]
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

    def forward(self, im_data, gt_boxes=None, gt_classes=None, dontcare=None, size_index=0):
        conv1s = self.conv1s(im_data)
        conv2 = self.conv2(conv1s)
        conv3 = self.conv3(conv2)
        conv1s_reorg = self.reorg(conv1s)
        cat_1_3 = torch.cat([conv1s_reorg, conv3], 1)
        conv4 = self.conv4(cat_1_3)
        conv5 = self.conv5(conv4)
        global_average_pool = self.global_average_pool(conv5)
        bsize, _, h, w = global_average_pool.size()
        global_average_pool_reshaped = global_average_pool.permute(0, 2, 3, 1).contiguous().view(bsize, -1, cfg.num_anchors, cfg.num_classes + 5)
        xy_pred = F.sigmoid(global_average_pool_reshaped[:, :, :, 0:2])
        wh_pred = torch.exp(global_average_pool_reshaped[:, :, :, 2:4])
        bbox_pred = torch.cat([xy_pred, wh_pred], 3)
        iou_pred = F.sigmoid(global_average_pool_reshaped[:, :, :, 4:5])
        score_pred = global_average_pool_reshaped[:, :, :, 5:].contiguous()
        prob_pred = F.softmax(score_pred.view(-1, score_pred.size()[-1])).view_as(score_pred)
        if self.training:
            bbox_pred_np = bbox_pred.data.cpu().numpy()
            iou_pred_np = iou_pred.data.cpu().numpy()
            _boxes, _ious, _classes, _box_mask, _iou_mask, _class_mask = self._build_target(bbox_pred_np, gt_boxes, gt_classes, dontcare, iou_pred_np, size_index)
            _boxes = net_utils.np_to_variable(_boxes)
            _ious = net_utils.np_to_variable(_ious)
            _classes = net_utils.np_to_variable(_classes)
            box_mask = net_utils.np_to_variable(_box_mask, dtype=torch.FloatTensor)
            iou_mask = net_utils.np_to_variable(_iou_mask, dtype=torch.FloatTensor)
            class_mask = net_utils.np_to_variable(_class_mask, dtype=torch.FloatTensor)
            num_boxes = sum(len(boxes) for boxes in gt_boxes)
            box_mask = box_mask.expand_as(_boxes)
            self.bbox_loss = nn.MSELoss(size_average=False)(bbox_pred * box_mask, _boxes * box_mask) / num_boxes
            self.iou_loss = nn.MSELoss(size_average=False)(iou_pred * iou_mask, _ious * iou_mask) / num_boxes
            class_mask = class_mask.expand_as(prob_pred)
            self.cls_loss = nn.MSELoss(size_average=False)(prob_pred * class_mask, _classes * class_mask) / num_boxes
        return bbox_pred, iou_pred, prob_pred

    def _build_target(self, bbox_pred_np, gt_boxes, gt_classes, dontcare, iou_pred_np, size_index):
        """
        :param bbox_pred: shape: (bsize, h x w, num_anchors, 4) :
                          (sig(tx), sig(ty), exp(tw), exp(th))
        """
        bsize = bbox_pred_np.shape[0]
        targets = self.pool.map(partial(_process_batch, size_index=size_index), ((bbox_pred_np[b], gt_boxes[b], gt_classes[b], dontcare[b], iou_pred_np[b]) for b in range(bsize)))
        _boxes = np.stack(tuple(row[0] for row in targets))
        _ious = np.stack(tuple(row[1] for row in targets))
        _classes = np.stack(tuple(row[2] for row in targets))
        _box_mask = np.stack(tuple(row[3] for row in targets))
        _iou_mask = np.stack(tuple(row[4] for row in targets))
        _class_mask = np.stack(tuple(row[5] for row in targets))
        return _boxes, _ious, _classes, _box_mask, _iou_mask, _class_mask

    def load_from_npz(self, fname, num_conv=None):
        dest_src = {'conv.weight': 'kernel', 'conv.bias': 'biases', 'bn.weight': 'gamma', 'bn.bias': 'biases', 'bn.running_mean': 'moving_mean', 'bn.running_var': 'moving_variance'}
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


class RoIPool(nn.Module):

    def __init__(self, pooled_height, pooled_width, spatial_scale):
        super(RoIPool, self).__init__()
        self.pooled_width = int(pooled_width)
        self.pooled_height = int(pooled_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        batch_size, num_channels, data_height, data_width = features.size()
        num_rois = rois.size()[0]
        outputs = Variable(torch.zeros(num_rois, num_channels, self.pooled_height, self.pooled_width))
        for roi_ind, roi in enumerate(rois):
            batch_ind = int(roi[0].data[0])
            roi_start_w, roi_start_h, roi_end_w, roi_end_h = np.round(roi[1:].data.cpu().numpy() * self.spatial_scale).astype(int)
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
                        outputs[(roi_ind), :, (ph), (pw)] = torch.max(torch.max(data[:, hstart:hend, wstart:wend], 1)[0], 2)[0].view(-1)
        return outputs


class Conv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, relu=True, same_padding=False):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.relu = nn.LeakyReLU(0.1, inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Conv2d_BatchNorm(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, relu=True, same_padding=False):
        super(Conv2d_BatchNorm, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=False)
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


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Conv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Conv2d_BatchNorm,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FC,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_longcw_yolo2_pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

