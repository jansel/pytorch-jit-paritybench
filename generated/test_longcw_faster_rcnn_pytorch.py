import sys
_module = sys.modules[__name__]
del sys
master = _module
demo = _module
faster_rcnn = _module
datasets = _module
coco = _module
ds_utils = _module
factory = _module
imagenet3d = _module
imdb = _module
imdb2 = _module
kitti = _module
kitti_tracking = _module
kittivoc = _module
nissan = _module
nthu = _module
pascal3d = _module
pascal_voc = _module
pascal_voc2 = _module
voc_eval = _module
fast_rcnn = _module
bbox_transform = _module
config = _module
config2 = _module
nms_wrapper = _module
faster_rcnn = _module
network = _module
nms = _module
py_cpu_nms = _module
pycocotools = _module
cocoeval = _module
mask = _module
roi_data_layer = _module
layer = _module
minibatch = _module
minibatch2 = _module
roidb = _module
roidb2 = _module
roi_pooling = _module
_ext = _module
build = _module
functions = _module
roi_pool = _module
modules = _module
roi_pool = _module
roi_pool_py = _module
rpn_msr = _module
anchor_target_layer = _module
generate = _module
generate_anchors = _module
proposal_layer = _module
proposal_target_layer = _module
setup = _module
utils = _module
blob = _module
boxes_grid = _module
timer = _module
vgg16 = _module
test = _module
train = _module

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


import torch


import torch.nn as nn


import torch.nn.functional as F


from torch.autograd import Variable


from torch.nn.modules.module import Module


class RPN(nn.Module):
    _feat_stride = [16]
    anchor_scales = [8, 16, 32]

    def __init__(self):
        super(RPN, self).__init__()
        self.features = VGG16(bn=False)
        self.conv1 = Conv2d(512, 512, 3, same_padding=True)
        self.score_conv = Conv2d(512, len(self.anchor_scales) * 3 * 2, 1,
            relu=False, same_padding=False)
        self.bbox_conv = Conv2d(512, len(self.anchor_scales) * 3 * 4, 1,
            relu=False, same_padding=False)
        self.cross_entropy = None
        self.los_box = None

    @property
    def loss(self):
        return self.cross_entropy + self.loss_box * 10

    def forward(self, im_data, im_info, gt_boxes=None, gt_ishard=None,
        dontcare_areas=None):
        im_data = network.np_to_variable(im_data, is_cuda=True)
        im_data = im_data.permute(0, 3, 1, 2)
        features = self.features(im_data)
        rpn_conv1 = self.conv1(features)
        rpn_cls_score = self.score_conv(rpn_conv1)
        rpn_cls_score_reshape = self.reshape_layer(rpn_cls_score, 2)
        rpn_cls_prob = F.softmax(rpn_cls_score_reshape)
        rpn_cls_prob_reshape = self.reshape_layer(rpn_cls_prob, len(self.
            anchor_scales) * 3 * 2)
        rpn_bbox_pred = self.bbox_conv(rpn_conv1)
        cfg_key = 'TRAIN' if self.training else 'TEST'
        rois = self.proposal_layer(rpn_cls_prob_reshape, rpn_bbox_pred,
            im_info, cfg_key, self._feat_stride, self.anchor_scales)
        if self.training:
            assert gt_boxes is not None
            rpn_data = self.anchor_target_layer(rpn_cls_score, gt_boxes,
                gt_ishard, dontcare_areas, im_info, self._feat_stride, self
                .anchor_scales)
            self.cross_entropy, self.loss_box = self.build_loss(
                rpn_cls_score_reshape, rpn_bbox_pred, rpn_data)
        return features, rois

    def build_loss(self, rpn_cls_score_reshape, rpn_bbox_pred, rpn_data):
        rpn_cls_score = rpn_cls_score_reshape.permute(0, 2, 3, 1).contiguous(
            ).view(-1, 2)
        rpn_label = rpn_data[0].view(-1)
        rpn_keep = Variable(rpn_label.data.ne(-1).nonzero().squeeze())
        rpn_cls_score = torch.index_select(rpn_cls_score, 0, rpn_keep)
        rpn_label = torch.index_select(rpn_label, 0, rpn_keep)
        fg_cnt = torch.sum(rpn_label.data.ne(0))
        rpn_cross_entropy = F.cross_entropy(rpn_cls_score, rpn_label)
        (rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights
            ) = rpn_data[1:]
        rpn_bbox_targets = torch.mul(rpn_bbox_targets, rpn_bbox_inside_weights)
        rpn_bbox_pred = torch.mul(rpn_bbox_pred, rpn_bbox_inside_weights)
        rpn_loss_box = F.smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets,
            size_average=False) / (fg_cnt + 0.0001)
        return rpn_cross_entropy, rpn_loss_box

    @staticmethod
    def reshape_layer(x, d):
        input_shape = x.size()
        x = x.view(input_shape[0], int(d), int(float(input_shape[1] *
            input_shape[2]) / float(d)), input_shape[3])
        return x

    @staticmethod
    def proposal_layer(rpn_cls_prob_reshape, rpn_bbox_pred, im_info,
        cfg_key, _feat_stride, anchor_scales):
        rpn_cls_prob_reshape = rpn_cls_prob_reshape.data.cpu().numpy()
        rpn_bbox_pred = rpn_bbox_pred.data.cpu().numpy()
        x = proposal_layer_py(rpn_cls_prob_reshape, rpn_bbox_pred, im_info,
            cfg_key, _feat_stride, anchor_scales)
        x = network.np_to_variable(x, is_cuda=True)
        return x.view(-1, 5)

    @staticmethod
    def anchor_target_layer(rpn_cls_score, gt_boxes, gt_ishard,
        dontcare_areas, im_info, _feat_stride, anchor_scales):
        """
        rpn_cls_score: for pytorch (1, Ax2, H, W) bg/fg scores of previous conv layer
        gt_boxes: (G, 5) vstack of [x1, y1, x2, y2, class]
        gt_ishard: (G, 1), 1 or 0 indicates difficult or not
        dontcare_areas: (D, 4), some areas may contains small objs but no labelling. D may be 0
        im_info: a list of [image_height, image_width, scale_ratios]
        _feat_stride: the downsampling ratio of feature map to the original input image
        anchor_scales: the scales to the basic_anchor (basic anchor is [16, 16])
        ----------
        Returns
        ----------
        rpn_labels : (1, 1, HxA, W), for each anchor, 0 denotes bg, 1 fg, -1 dontcare
        rpn_bbox_targets: (1, 4xA, H, W), distances of the anchors to the gt_boxes(may contains some transform)
                        that are the regression objectives
        rpn_bbox_inside_weights: (1, 4xA, H, W) weights of each boxes, mainly accepts hyper param in cfg
        rpn_bbox_outside_weights: (1, 4xA, H, W) used to balance the fg/bg,
        beacuse the numbers of bgs and fgs mays significiantly different
        """
        rpn_cls_score = rpn_cls_score.data.cpu().numpy()
        (rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights,
            rpn_bbox_outside_weights) = (anchor_target_layer_py(
            rpn_cls_score, gt_boxes, gt_ishard, dontcare_areas, im_info,
            _feat_stride, anchor_scales))
        rpn_labels = network.np_to_variable(rpn_labels, is_cuda=True, dtype
            =torch.LongTensor)
        rpn_bbox_targets = network.np_to_variable(rpn_bbox_targets, is_cuda
            =True)
        rpn_bbox_inside_weights = network.np_to_variable(
            rpn_bbox_inside_weights, is_cuda=True)
        rpn_bbox_outside_weights = network.np_to_variable(
            rpn_bbox_outside_weights, is_cuda=True)
        return (rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights,
            rpn_bbox_outside_weights)

    def load_from_npz(self, params):
        self.features.load_from_npz(params)
        pairs = {'conv1.conv': 'rpn_conv/3x3', 'score_conv.conv':
            'rpn_cls_score', 'bbox_conv.conv': 'rpn_bbox_pred'}
        own_dict = self.state_dict()
        for k, v in pairs.items():
            key = '{}.weight'.format(k)
            param = torch.from_numpy(params['{}/weights:0'.format(v)]).permute(
                3, 2, 0, 1)
            own_dict[key].copy_(param)
            key = '{}.bias'.format(k)
            param = torch.from_numpy(params['{}/biases:0'.format(v)])
            own_dict[key].copy_(param)


def bbox_transform_inv(boxes, deltas):
    if boxes.shape[0] == 0:
        return np.zeros((0,), dtype=deltas.dtype)
    boxes = boxes.astype(deltas.dtype, copy=False)
    widths = boxes[:, (2)] - boxes[:, (0)] + 1.0
    heights = boxes[:, (3)] - boxes[:, (1)] + 1.0
    ctr_x = boxes[:, (0)] + 0.5 * widths
    ctr_y = boxes[:, (1)] + 0.5 * heights
    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]
    pred_ctr_x = dx * widths[:, (np.newaxis)] + ctr_x[:, (np.newaxis)]
    pred_ctr_y = dy * heights[:, (np.newaxis)] + ctr_y[:, (np.newaxis)]
    pred_w = np.exp(dw) * widths[:, (np.newaxis)]
    pred_h = np.exp(dh) * heights[:, (np.newaxis)]
    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h
    return pred_boxes


def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """
    if boxes.shape[0] == 0:
        return boxes
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes


def im_list_to_blob(ims):
    """Convert a list of images into a network input.

    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3), dtype=np.
        float32)
    for i in xrange(num_images):
        im = ims[i]
        blob[(i), 0:im.shape[0], 0:im.shape[1], :] = im
    return blob


_global_config['USE_GPU_NMS'] = 4


_global_config['GPU_ID'] = 4


def nms(dets, thresh, force_cpu=False):
    """Dispatch to either CPU or GPU NMS implementations."""
    if dets.shape[0] == 0:
        return []
    if cfg.USE_GPU_NMS and not force_cpu:
        return gpu_nms(dets, thresh, device_id=cfg.GPU_ID)
    else:
        return cpu_nms(dets, thresh)


def nms_detections(pred_boxes, scores, nms_thresh, inds=None):
    dets = np.hstack((pred_boxes, scores[:, (np.newaxis)])).astype(np.float32)
    keep = nms(dets, nms_thresh)
    if inds is None:
        return pred_boxes[keep], scores[keep]
    return pred_boxes[keep], scores[keep], inds[keep]


class FasterRCNN(nn.Module):
    n_classes = 21
    classes = np.asarray(['__background__', 'aeroplane', 'bicycle', 'bird',
        'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train', 'tvmonitor'])
    PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])
    SCALES = 600,
    MAX_SIZE = 1000

    def __init__(self, classes=None, debug=False):
        super(FasterRCNN, self).__init__()
        if classes is not None:
            self.classes = classes
            self.n_classes = len(classes)
        self.rpn = RPN()
        self.roi_pool = RoIPool(7, 7, 1.0 / 16)
        self.fc6 = FC(512 * 7 * 7, 4096)
        self.fc7 = FC(4096, 4096)
        self.score_fc = FC(4096, self.n_classes, relu=False)
        self.bbox_fc = FC(4096, self.n_classes * 4, relu=False)
        self.cross_entropy = None
        self.loss_box = None
        self.debug = debug

    @property
    def loss(self):
        return self.cross_entropy + self.loss_box * 10

    def forward(self, im_data, im_info, gt_boxes=None, gt_ishard=None,
        dontcare_areas=None):
        features, rois = self.rpn(im_data, im_info, gt_boxes, gt_ishard,
            dontcare_areas)
        if self.training:
            roi_data = self.proposal_target_layer(rois, gt_boxes, gt_ishard,
                dontcare_areas, self.n_classes)
            rois = roi_data[0]
        pooled_features = self.roi_pool(features, rois)
        x = pooled_features.view(pooled_features.size()[0], -1)
        x = self.fc6(x)
        x = F.dropout(x, training=self.training)
        x = self.fc7(x)
        x = F.dropout(x, training=self.training)
        cls_score = self.score_fc(x)
        cls_prob = F.softmax(cls_score)
        bbox_pred = self.bbox_fc(x)
        if self.training:
            self.cross_entropy, self.loss_box = self.build_loss(cls_score,
                bbox_pred, roi_data)
        return cls_prob, bbox_pred, rois

    def build_loss(self, cls_score, bbox_pred, roi_data):
        label = roi_data[1].squeeze()
        fg_cnt = torch.sum(label.data.ne(0))
        bg_cnt = label.data.numel() - fg_cnt
        if self.debug:
            maxv, predict = cls_score.data.max(1)
            self.tp = torch.sum(predict[:fg_cnt].eq(label.data[:fg_cnt])
                ) if fg_cnt > 0 else 0
            self.tf = torch.sum(predict[fg_cnt:].eq(label.data[fg_cnt:]))
            self.fg_cnt = fg_cnt
            self.bg_cnt = bg_cnt
        ce_weights = torch.ones(cls_score.size()[1])
        ce_weights[0] = float(fg_cnt) / bg_cnt
        ce_weights = ce_weights
        cross_entropy = F.cross_entropy(cls_score, label, weight=ce_weights)
        bbox_targets, bbox_inside_weights, bbox_outside_weights = roi_data[2:]
        bbox_targets = torch.mul(bbox_targets, bbox_inside_weights)
        bbox_pred = torch.mul(bbox_pred, bbox_inside_weights)
        loss_box = F.smooth_l1_loss(bbox_pred, bbox_targets, size_average=False
            ) / (fg_cnt + 0.0001)
        return cross_entropy, loss_box

    @staticmethod
    def proposal_target_layer(rpn_rois, gt_boxes, gt_ishard, dontcare_areas,
        num_classes):
        """
        ----------
        rpn_rois:  (1 x H x W x A, 5) [0, x1, y1, x2, y2]
        gt_boxes: (G, 5) [x1 ,y1 ,x2, y2, class] int
        # gt_ishard: (G, 1) {0 | 1} 1 indicates hard
        dontcare_areas: (D, 4) [ x1, y1, x2, y2]
        num_classes
        ----------
        Returns
        ----------
        rois: (1 x H x W x A, 5) [0, x1, y1, x2, y2]
        labels: (1 x H x W x A, 1) {0,1,...,_num_classes-1}
        bbox_targets: (1 x H x W x A, K x4) [dx1, dy1, dx2, dy2]
        bbox_inside_weights: (1 x H x W x A, Kx4) 0, 1 masks for the computing loss
        bbox_outside_weights: (1 x H x W x A, Kx4) 0, 1 masks for the computing loss
        """
        rpn_rois = rpn_rois.data.cpu().numpy()
        (rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights
            ) = (proposal_target_layer_py(rpn_rois, gt_boxes, gt_ishard,
            dontcare_areas, num_classes))
        rois = network.np_to_variable(rois, is_cuda=True)
        labels = network.np_to_variable(labels, is_cuda=True, dtype=torch.
            LongTensor)
        bbox_targets = network.np_to_variable(bbox_targets, is_cuda=True)
        bbox_inside_weights = network.np_to_variable(bbox_inside_weights,
            is_cuda=True)
        bbox_outside_weights = network.np_to_variable(bbox_outside_weights,
            is_cuda=True)
        return (rois, labels, bbox_targets, bbox_inside_weights,
            bbox_outside_weights)

    def interpret_faster_rcnn(self, cls_prob, bbox_pred, rois, im_info,
        im_shape, nms=True, clip=True, min_score=0.0):
        scores, inds = cls_prob.data.max(1)
        scores, inds = scores.cpu().numpy(), inds.cpu().numpy()
        keep = np.where((inds > 0) & (scores >= min_score))
        scores, inds = scores[keep], inds[keep]
        keep = keep[0]
        box_deltas = bbox_pred.data.cpu().numpy()[keep]
        box_deltas = np.asarray([box_deltas[(i), inds[i] * 4:inds[i] * 4 + 
            4] for i in range(len(inds))], dtype=np.float)
        boxes = rois.data.cpu().numpy()[(keep), 1:5] / im_info[0][2]
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        if clip:
            pred_boxes = clip_boxes(pred_boxes, im_shape)
        if nms and pred_boxes.shape[0] > 0:
            pred_boxes, scores, inds = nms_detections(pred_boxes, scores, 
                0.3, inds=inds)
        return pred_boxes, scores, self.classes[inds]

    def detect(self, image, thr=0.3):
        im_data, im_scales = self.get_image_blob(image)
        im_info = np.array([[im_data.shape[1], im_data.shape[2], im_scales[
            0]]], dtype=np.float32)
        cls_prob, bbox_pred, rois = self(im_data, im_info)
        pred_boxes, scores, classes = self.interpret_faster_rcnn(cls_prob,
            bbox_pred, rois, im_info, image.shape, min_score=thr)
        return pred_boxes, scores, classes

    def get_image_blob_noscale(self, im):
        im_orig = im.astype(np.float32, copy=True)
        im_orig -= self.PIXEL_MEANS
        processed_ims = [im]
        im_scale_factors = [1.0]
        blob = im_list_to_blob(processed_ims)
        return blob, np.array(im_scale_factors)

    def get_image_blob(self, im):
        """Converts an image into a network input.
        Arguments:
            im (ndarray): a color image in BGR order
        Returns:
            blob (ndarray): a data blob holding an image pyramid
            im_scale_factors (list): list of image scales (relative to im) used
                in the image pyramid
        """
        im_orig = im.astype(np.float32, copy=True)
        im_orig -= self.PIXEL_MEANS
        im_shape = im_orig.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        processed_ims = []
        im_scale_factors = []
        for target_size in self.SCALES:
            im_scale = float(target_size) / float(im_size_min)
            if np.round(im_scale * im_size_max) > self.MAX_SIZE:
                im_scale = float(self.MAX_SIZE) / float(im_size_max)
            im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                interpolation=cv2.INTER_LINEAR)
            im_scale_factors.append(im_scale)
            processed_ims.append(im)
        blob = im_list_to_blob(processed_ims)
        return blob, np.array(im_scale_factors)

    def load_from_npz(self, params):
        self.rpn.load_from_npz(params)
        pairs = {'fc6.fc': 'fc6', 'fc7.fc': 'fc7', 'score_fc.fc':
            'cls_score', 'bbox_fc.fc': 'bbox_pred'}
        own_dict = self.state_dict()
        for k, v in pairs.items():
            key = '{}.weight'.format(k)
            param = torch.from_numpy(params['{}/weights:0'.format(v)]).permute(
                1, 0)
            own_dict[key].copy_(param)
            key = '{}.bias'.format(k)
            param = torch.from_numpy(params['{}/biases:0'.format(v)])
            own_dict[key].copy_(param)


class Conv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        relu=True, same_padding=False, bn=False):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
            stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0,
            affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
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


class RoIPool(Module):

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


class VGG16(nn.Module):

    def __init__(self, bn=False):
        super(VGG16, self).__init__()
        self.conv1 = nn.Sequential(Conv2d(3, 64, 3, same_padding=True, bn=
            bn), Conv2d(64, 64, 3, same_padding=True, bn=bn), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(Conv2d(64, 128, 3, same_padding=True, bn
            =bn), Conv2d(128, 128, 3, same_padding=True, bn=bn), nn.
            MaxPool2d(2))
        network.set_trainable(self.conv1, requires_grad=False)
        network.set_trainable(self.conv2, requires_grad=False)
        self.conv3 = nn.Sequential(Conv2d(128, 256, 3, same_padding=True,
            bn=bn), Conv2d(256, 256, 3, same_padding=True, bn=bn), Conv2d(
            256, 256, 3, same_padding=True, bn=bn), nn.MaxPool2d(2))
        self.conv4 = nn.Sequential(Conv2d(256, 512, 3, same_padding=True,
            bn=bn), Conv2d(512, 512, 3, same_padding=True, bn=bn), Conv2d(
            512, 512, 3, same_padding=True, bn=bn), nn.MaxPool2d(2))
        self.conv5 = nn.Sequential(Conv2d(512, 512, 3, same_padding=True,
            bn=bn), Conv2d(512, 512, 3, same_padding=True, bn=bn), Conv2d(
            512, 512, 3, same_padding=True, bn=bn))

    def forward(self, im_data):
        x = self.conv1(im_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

    def load_from_npz(self, params):
        own_dict = self.state_dict()
        for name, val in own_dict.items():
            i, j = int(name[4]), int(name[6]) + 1
            ptype = 'weights' if name[-1] == 't' else 'biases'
            key = 'conv{}_{}/{}:0'.format(i, j, ptype)
            param = torch.from_numpy(params[key])
            if ptype == 'weights':
                param = param.permute(3, 2, 0, 1)
            val.copy_(param)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_longcw_faster_rcnn_pytorch(_paritybench_base):
    pass
    def test_000(self):
        self._check(Conv2d(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(FC(*[], **{'in_features': 4, 'out_features': 4}), [torch.rand([4, 4, 4, 4])], {})

