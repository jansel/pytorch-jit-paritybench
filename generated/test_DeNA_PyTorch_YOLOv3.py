import sys
_module = sys.modules[__name__]
del sys
cocodataset = _module
demo = _module
yolo_layer = _module
yolov3 = _module
train = _module
cocoapi_evaluator = _module
parse_yolo_weights = _module
utils = _module
vis_bbox = _module

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


from collections import defaultdict


def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.
    IoU is calculated as a ratio of area of the intersection
    and area of the union.

    Args:
        bbox_a (array): An array whose shape is :math:`(N, 4)`.
            :math:`N` is the number of bounding boxes.
            The dtype should be :obj:`numpy.float32`.
        bbox_b (array): An array similar to :obj:`bbox_a`,
            whose shape is :math:`(K, 4)`.
            The dtype should be :obj:`numpy.float32`.
    Returns:
        array:
        An array whose shape is :math:`(N, K)`.         An element at index :math:`(n, k)` contains IoUs between         :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding         box in :obj:`bbox_b`.

    from: https://github.com/chainer/chainercv
    """
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError
    if xyxy:
        tl = torch.max(bboxes_a[:, (None), :2], bboxes_b[:, :2])
        br = torch.min(bboxes_a[:, (None), 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max(bboxes_a[:, (None), :2] - bboxes_a[:, (None), 2:] / 
            2, bboxes_b[:, :2] - bboxes_b[:, 2:] / 2)
        br = torch.min(bboxes_a[:, (None), :2] + bboxes_a[:, (None), 2:] / 
            2, bboxes_b[:, :2] + bboxes_b[:, 2:] / 2)
        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en
    return area_i / (area_a[:, (None)] + area_b - area_i)


class YOLOLayer(nn.Module):
    """
    detection layer corresponding to yolo_layer.c of darknet
    """

    def __init__(self, config_model, layer_no, in_ch, ignore_thre=0.7):
        """
        Args:
            config_model (dict) : model configuration.
                ANCHORS (list of tuples) :
                ANCH_MASK:  (list of int list): index indicating the anchors to be
                    used in YOLO layers. One of the mask group is picked from the list.
                N_CLASSES (int): number of classes
            layer_no (int): YOLO layer number - one from (0, 1, 2).
            in_ch (int): number of input channels.
            ignore_thre (float): threshold of IoU above which objectness training is ignored.
        """
        super(YOLOLayer, self).__init__()
        strides = [32, 16, 8]
        self.anchors = config_model['ANCHORS']
        self.anch_mask = config_model['ANCH_MASK'][layer_no]
        self.n_anchors = len(self.anch_mask)
        self.n_classes = config_model['N_CLASSES']
        self.ignore_thre = ignore_thre
        self.l2_loss = nn.MSELoss(size_average=False)
        self.bce_loss = nn.BCELoss(size_average=False)
        self.stride = strides[layer_no]
        self.all_anchors_grid = [(w / self.stride, h / self.stride) for w,
            h in self.anchors]
        self.masked_anchors = [self.all_anchors_grid[i] for i in self.anch_mask
            ]
        self.ref_anchors = np.zeros((len(self.all_anchors_grid), 4))
        self.ref_anchors[:, 2:] = np.array(self.all_anchors_grid)
        self.ref_anchors = torch.FloatTensor(self.ref_anchors)
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=self.
            n_anchors * (self.n_classes + 5), kernel_size=1, stride=1,
            padding=0)

    def forward(self, xin, labels=None):
        """
        In this
        Args:
            xin (torch.Tensor): input feature map whose size is :math:`(N, C, H, W)`,                 where N, C, H, W denote batchsize, channel width, height, width respectively.
            labels (torch.Tensor): label data whose size is :math:`(N, K, 5)`.                 N and K denote batchsize and number of labels.
                Each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
        Returns:
            loss (torch.Tensor): total loss - the target of backprop.
            loss_xy (torch.Tensor): x, y loss - calculated by binary cross entropy (BCE)                 with boxsize-dependent weights.
            loss_wh (torch.Tensor): w, h loss - calculated by l2 without size averaging and                 with boxsize-dependent weights.
            loss_obj (torch.Tensor): objectness loss - calculated by BCE.
            loss_cls (torch.Tensor): classification loss - calculated by BCE for each class.
            loss_l2 (torch.Tensor): total l2 loss - only for logging.
        """
        output = self.conv(xin)
        batchsize = output.shape[0]
        fsize = output.shape[2]
        n_ch = 5 + self.n_classes
        dtype = torch.cuda.FloatTensor if xin.is_cuda else torch.FloatTensor
        output = output.view(batchsize, self.n_anchors, n_ch, fsize, fsize)
        output = output.permute(0, 1, 3, 4, 2)
        output[..., np.r_[:2, 4:n_ch]] = torch.sigmoid(output[..., np.r_[:2,
            4:n_ch]])
        x_shift = dtype(np.broadcast_to(np.arange(fsize, dtype=np.float32),
            output.shape[:4]))
        y_shift = dtype(np.broadcast_to(np.arange(fsize, dtype=np.float32).
            reshape(fsize, 1), output.shape[:4]))
        masked_anchors = np.array(self.masked_anchors)
        w_anchors = dtype(np.broadcast_to(np.reshape(masked_anchors[:, (0)],
            (1, self.n_anchors, 1, 1)), output.shape[:4]))
        h_anchors = dtype(np.broadcast_to(np.reshape(masked_anchors[:, (1)],
            (1, self.n_anchors, 1, 1)), output.shape[:4]))
        pred = output.clone()
        pred[..., 0] += x_shift
        pred[..., 1] += y_shift
        pred[..., 2] = torch.exp(pred[..., 2]) * w_anchors
        pred[..., 3] = torch.exp(pred[..., 3]) * h_anchors
        if labels is None:
            pred[(...), :4] *= self.stride
            return pred.view(batchsize, -1, n_ch).data
        pred = pred[(...), :4].data
        tgt_mask = torch.zeros(batchsize, self.n_anchors, fsize, fsize, 4 +
            self.n_classes).type(dtype)
        obj_mask = torch.ones(batchsize, self.n_anchors, fsize, fsize).type(
            dtype)
        tgt_scale = torch.zeros(batchsize, self.n_anchors, fsize, fsize, 2
            ).type(dtype)
        target = torch.zeros(batchsize, self.n_anchors, fsize, fsize, n_ch
            ).type(dtype)
        labels = labels.cpu().data
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)
        truth_x_all = labels[:, :, (1)] * fsize
        truth_y_all = labels[:, :, (2)] * fsize
        truth_w_all = labels[:, :, (3)] * fsize
        truth_h_all = labels[:, :, (4)] * fsize
        truth_i_all = truth_x_all.numpy()
        truth_j_all = truth_y_all.numpy()
        for b in range(batchsize):
            n = int(nlabel[b])
            if n == 0:
                continue
            truth_box = dtype(np.zeros((n, 4)))
            truth_box[:n, (2)] = truth_w_all[(b), :n]
            truth_box[:n, (3)] = truth_h_all[(b), :n]
            truth_i = truth_i_all[(b), :n]
            truth_j = truth_j_all[(b), :n]
            anchor_ious_all = bboxes_iou(truth_box.cpu(), self.ref_anchors)
            best_n_all = np.argmax(anchor_ious_all, axis=1)
            best_n = best_n_all % 3
            best_n_mask = (best_n_all == self.anch_mask[0]) | (best_n_all ==
                self.anch_mask[1]) | (best_n_all == self.anch_mask[2])
            truth_box[:n, (0)] = truth_x_all[(b), :n]
            truth_box[:n, (1)] = truth_y_all[(b), :n]
            pred_ious = bboxes_iou(pred[b].view(-1, 4), truth_box, xyxy=False)
            pred_best_iou, _ = pred_ious.max(dim=1)
            pred_best_iou = pred_best_iou > self.ignore_thre
            pred_best_iou = pred_best_iou.view(pred[b].shape[:3])
            obj_mask[b] = 1 - pred_best_iou
            if sum(best_n_mask) == 0:
                continue
            for ti in range(best_n.shape[0]):
                if best_n_mask[ti] == 1:
                    i, j = truth_i[ti], truth_j[ti]
                    a = best_n[ti]
                    obj_mask[b, a, j, i] = 1
                    tgt_mask[(b), (a), (j), (i), :] = 1
                    target[b, a, j, i, 0] = truth_x_all[b, ti] - truth_x_all[
                        b, ti].to(torch.int16)
                    target[b, a, j, i, 1] = truth_y_all[b, ti] - truth_y_all[
                        b, ti].to(torch.int16)
                    target[b, a, j, i, 2] = torch.log(truth_w_all[b, ti] /
                        torch.Tensor(self.masked_anchors)[best_n[ti], 0] + 
                        1e-16)
                    target[b, a, j, i, 3] = torch.log(truth_h_all[b, ti] /
                        torch.Tensor(self.masked_anchors)[best_n[ti], 1] + 
                        1e-16)
                    target[b, a, j, i, 4] = 1
                    target[b, a, j, i, 5 + labels[b, ti, 0].numpy()] = 1
                    tgt_scale[(b), (a), (j), (i), :] = torch.sqrt(2 - 
                        truth_w_all[b, ti] * truth_h_all[b, ti] / fsize / fsize
                        )
        output[..., 4] *= obj_mask
        output[..., np.r_[0:4, 5:n_ch]] *= tgt_mask
        output[(...), 2:4] *= tgt_scale
        target[..., 4] *= obj_mask
        target[..., np.r_[0:4, 5:n_ch]] *= tgt_mask
        target[(...), 2:4] *= tgt_scale
        bceloss = nn.BCELoss(weight=tgt_scale * tgt_scale, size_average=False)
        loss_xy = bceloss(output[(...), :2], target[(...), :2])
        loss_wh = self.l2_loss(output[(...), 2:4], target[(...), 2:4]) / 2
        loss_obj = self.bce_loss(output[..., 4], target[..., 4])
        loss_cls = self.bce_loss(output[(...), 5:], target[(...), 5:])
        loss_l2 = self.l2_loss(output, target)
        loss = loss_xy + loss_wh + loss_obj + loss_cls
        return loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2


def add_conv(in_ch, out_ch, ksize, stride):
    """
    Add a conv2d / batchnorm / leaky ReLU block.
    Args:
        in_ch (int): number of input channels of the convolution layer.
        out_ch (int): number of output channels of the convolution layer.
        ksize (int): kernel size of the convolution layer.
        stride (int): stride of the convolution layer.
    Returns:
        stage (Sequential) : Sequential layers composing a convolution block.
    """
    stage = nn.Sequential()
    pad = (ksize - 1) // 2
    stage.add_module('conv', nn.Conv2d(in_channels=in_ch, out_channels=
        out_ch, kernel_size=ksize, stride=stride, padding=pad, bias=False))
    stage.add_module('batch_norm', nn.BatchNorm2d(out_ch))
    stage.add_module('leaky', nn.LeakyReLU(0.1))
    return stage


class resblock(nn.Module):
    """
    Sequential residual blocks each of which consists of     two convolution layers.
    Args:
        ch (int): number of input and output channels.
        nblocks (int): number of residual blocks.
        shortcut (bool): if True, residual tensor addition is enabled.
    """

    def __init__(self, ch, nblocks=1, shortcut=True):
        super().__init__()
        self.shortcut = shortcut
        self.module_list = nn.ModuleList()
        for i in range(nblocks):
            resblock_one = nn.ModuleList()
            resblock_one.append(add_conv(ch, ch // 2, 1, 1))
            resblock_one.append(add_conv(ch // 2, ch, 3, 1))
            self.module_list.append(resblock_one)

    def forward(self, x):
        for module in self.module_list:
            h = x
            for res in module:
                h = res(h)
            x = x + h if self.shortcut else h
        return x


def create_yolov3_modules(config_model, ignore_thre):
    """
    Build yolov3 layer modules.
    Args:
        config_model (dict): model configuration.
            See YOLOLayer class for details.
        ignore_thre (float): used in YOLOLayer.
    Returns:
        mlist (ModuleList): YOLOv3 module list.
    """
    mlist = nn.ModuleList()
    mlist.append(add_conv(in_ch=3, out_ch=32, ksize=3, stride=1))
    mlist.append(add_conv(in_ch=32, out_ch=64, ksize=3, stride=2))
    mlist.append(resblock(ch=64))
    mlist.append(add_conv(in_ch=64, out_ch=128, ksize=3, stride=2))
    mlist.append(resblock(ch=128, nblocks=2))
    mlist.append(add_conv(in_ch=128, out_ch=256, ksize=3, stride=2))
    mlist.append(resblock(ch=256, nblocks=8))
    mlist.append(add_conv(in_ch=256, out_ch=512, ksize=3, stride=2))
    mlist.append(resblock(ch=512, nblocks=8))
    mlist.append(add_conv(in_ch=512, out_ch=1024, ksize=3, stride=2))
    mlist.append(resblock(ch=1024, nblocks=4))
    mlist.append(resblock(ch=1024, nblocks=2, shortcut=False))
    mlist.append(add_conv(in_ch=1024, out_ch=512, ksize=1, stride=1))
    mlist.append(add_conv(in_ch=512, out_ch=1024, ksize=3, stride=1))
    mlist.append(YOLOLayer(config_model, layer_no=0, in_ch=1024,
        ignore_thre=ignore_thre))
    mlist.append(add_conv(in_ch=512, out_ch=256, ksize=1, stride=1))
    mlist.append(nn.Upsample(scale_factor=2, mode='nearest'))
    mlist.append(add_conv(in_ch=768, out_ch=256, ksize=1, stride=1))
    mlist.append(add_conv(in_ch=256, out_ch=512, ksize=3, stride=1))
    mlist.append(resblock(ch=512, nblocks=1, shortcut=False))
    mlist.append(add_conv(in_ch=512, out_ch=256, ksize=1, stride=1))
    mlist.append(add_conv(in_ch=256, out_ch=512, ksize=3, stride=1))
    mlist.append(YOLOLayer(config_model, layer_no=1, in_ch=512, ignore_thre
        =ignore_thre))
    mlist.append(add_conv(in_ch=256, out_ch=128, ksize=1, stride=1))
    mlist.append(nn.Upsample(scale_factor=2, mode='nearest'))
    mlist.append(add_conv(in_ch=384, out_ch=128, ksize=1, stride=1))
    mlist.append(add_conv(in_ch=128, out_ch=256, ksize=3, stride=1))
    mlist.append(resblock(ch=256, nblocks=2, shortcut=False))
    mlist.append(YOLOLayer(config_model, layer_no=2, in_ch=256, ignore_thre
        =ignore_thre))
    return mlist


class YOLOv3(nn.Module):
    """
    YOLOv3 model module. The module list is defined by create_yolov3_modules function.     The network returns loss values from three YOLO layers during training     and detection results during test.
    """

    def __init__(self, config_model, ignore_thre=0.7):
        """
        Initialization of YOLOv3 class.
        Args:
            config_model (dict): used in YOLOLayer.
            ignore_thre (float): used in YOLOLayer.
        """
        super(YOLOv3, self).__init__()
        if config_model['TYPE'] == 'YOLOv3':
            self.module_list = create_yolov3_modules(config_model, ignore_thre)
        else:
            raise Exception('Model name {} is not available'.format(
                config_model['TYPE']))

    def forward(self, x, targets=None):
        """
        Forward path of YOLOv3.
        Args:
            x (torch.Tensor) : input data whose shape is :math:`(N, C, H, W)`,                 where N, C are batchsize and num. of channels.
            targets (torch.Tensor) : label array whose shape is :math:`(N, 50, 5)`

        Returns:
            training:
                output (torch.Tensor): loss tensor for backpropagation.
            test:
                output (torch.Tensor): concatenated detection results.
        """
        train = targets is not None
        output = []
        self.loss_dict = defaultdict(float)
        route_layers = []
        for i, module in enumerate(self.module_list):
            if i in [14, 22, 28]:
                if train:
                    x, *loss_dict = module(x, targets)
                    for name, loss in zip(['xy', 'wh', 'conf', 'cls', 'l2'],
                        loss_dict):
                        self.loss_dict[name] += loss
                else:
                    x = module(x)
                output.append(x)
            else:
                x = module(x)
            if i in [6, 8, 12, 20]:
                route_layers.append(x)
            if i == 14:
                x = route_layers[2]
            if i == 22:
                x = route_layers[3]
            if i == 16:
                x = torch.cat((x, route_layers[1]), 1)
            if i == 24:
                x = torch.cat((x, route_layers[0]), 1)
        if train:
            return sum(output)
        else:
            return torch.cat(output, 1)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_DeNA_PyTorch_YOLOv3(_paritybench_base):
    pass
    def test_000(self):
        self._check(resblock(*[], **{'ch': 4}), [torch.rand([4, 4, 4, 4])], {})

