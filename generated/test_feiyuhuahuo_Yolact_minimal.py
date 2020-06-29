import sys
_module = sys.modules[__name__]
del sys
coco = _module
config = _module
detect = _module
eval = _module
backbone = _module
build_yolact = _module
multi_loss = _module
train = _module
augmentations = _module
box_utils = _module
functions = _module
labelme2coco = _module
output_utils = _module
pascal2coco = _module
timer = _module

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


import torch.nn.functional as F


from torch.autograd import Variable


import time


import torch.optim as optim


import torch.backends.cudnn as cudnn


import torch.utils.data as data


import numpy as np


from numpy import random


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
        norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * 4)
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


class ResNetBackbone(nn.Module):
    """ Adapted from torchvision.models.resnet """

    def __init__(self, layers, block=Bottleneck, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.num_base_layers = len(layers)
        self.layers = nn.ModuleList()
        self.channels = []
        self.norm_layer = norm_layer
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self._make_layer(block, 64, layers[0])
        self._make_layer(block, 128, layers[1], stride=2)
        self._make_layer(block, 256, layers[2], stride=2)
        self._make_layer(block, 512, layers[3], stride=2)
        self.backbone_modules = [m for m in self.modules() if isinstance(m,
            nn.Conv2d)]

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                self.norm_layer(planes * block.expansion))
        layers = [block(self.inplanes, planes, stride, downsample, self.
            norm_layer)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=self.
                norm_layer))
        layer = nn.Sequential(*layers)
        self.channels.append(planes * block.expansion)
        self.layers.append(layer)

    def forward(self, x):
        """ Returns a list of convouts for each layer. """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            outs.append(x)
        return tuple(outs)

    def init_backbone(self, path):
        """ Initializes the backbone weights for training. """
        state_dict = torch.load(path)
        keys = list(state_dict)
        for key in keys:
            if key.startswith('layer'):
                idx = int(key[5])
                new_key = 'layers.' + str(idx - 1) + key[6:]
                state_dict[new_key] = state_dict.pop(key)
        self.load_state_dict(state_dict, strict=False)

    def add_layer(self, conv_channels=1024, downsample=2, depth=1, block=
        Bottleneck):
        """ Add a downsample layer to the backbone as per what SSD does. """
        self._make_layer(block, conv_channels // block.expansion, blocks=
            depth, stride=downsample)


class Concat(nn.Module):

    def __init__(self, nets, extra_params):
        super().__init__()
        self.nets = nn.ModuleList(nets)
        self.extra_params = extra_params

    def forward(self, x):
        return torch.cat([net(x) for net in self.nets], dim=1, **self.
            extra_params)


class InterpolateModule(nn.Module):
    """
    A module version of F.interpolate.
    """

    def __init__(self, *args, **kwdargs):
        super().__init__()
        self.args = args
        self.kwdargs = kwdargs

    def forward(self, x):
        return F.interpolate(x, *self.args, **self.kwdargs)


extra_head_net = [(256, 3, {'padding': 1})]


def make_net(in_channels, cfg_net, include_last_relu=True):

    def make_layer(layer_cfg):
        nonlocal in_channels
        if isinstance(layer_cfg[0], str):
            layer_name = layer_cfg[0]
            if layer_name == 'cat':
                nets = [make_net(in_channels, x) for x in layer_cfg[1]]
                layer = Concat([net[0] for net in nets], layer_cfg[2])
                num_channels = sum([net[1] for net in nets])
        else:
            num_channels = layer_cfg[0]
            kernel_size = layer_cfg[1]
            if kernel_size > 0:
                layer = nn.Conv2d(in_channels, num_channels, kernel_size,
                    **layer_cfg[2])
            elif num_channels is None:
                layer = InterpolateModule(scale_factor=-kernel_size, mode=
                    'bilinear', align_corners=False, **layer_cfg[2])
            else:
                layer = nn.ConvTranspose2d(in_channels, num_channels, -
                    kernel_size, **layer_cfg[2])
        in_channels = num_channels if num_channels is not None else in_channels
        return [layer, nn.ReLU(inplace=True)]
    net = sum([make_layer(x) for x in cfg_net], [])
    if not include_last_relu:
        net = net[:-1]
    return nn.Sequential(*net), in_channels


_global_config['num_classes'] = 4


_global_config['aspect_ratios'] = 4


class PredictionModule(nn.Module):

    def __init__(self, in_channels, coef_dim):
        super().__init__()
        self.num_classes = cfg.num_classes
        self.coef_dim = coef_dim
        self.num_priors = len(cfg.aspect_ratios)
        self.upfeature, out_channels = make_net(in_channels, extra_head_net)
        self.bbox_layer = nn.Conv2d(out_channels, self.num_priors * 4,
            kernel_size=3, padding=1)
        self.conf_layer = nn.Conv2d(out_channels, self.num_priors * self.
            num_classes, kernel_size=3, padding=1)
        self.mask_layer = nn.Conv2d(out_channels, self.num_priors * self.
            coef_dim, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.upfeature(x)
        conf = self.conf_layer(x).permute(0, 2, 3, 1).reshape(x.size(0), -1,
            self.num_classes)
        bbox = self.bbox_layer(x).permute(0, 2, 3, 1).reshape(x.size(0), -1, 4)
        coef = self.mask_layer(x).permute(0, 2, 3, 1).reshape(x.size(0), -1,
            self.coef_dim)
        coef = torch.tanh(coef)
        return {'box': bbox, 'class': conf, 'coef': coef}


class FPN(nn.Module):
    """
    The FPN here is slightly different from the FPN introduced in https://arxiv.org/pdf/1612.03144.pdf.
    """

    def __init__(self, in_channels):
        super().__init__()
        self.num_downsample = 2
        self.in_channels = in_channels
        self.lat_layers = nn.ModuleList([nn.Conv2d(x, 256, kernel_size=1) for
            x in reversed(self.in_channels)])
        self.pred_layers = nn.ModuleList([nn.Conv2d(256, 256, kernel_size=3,
            padding=1) for _ in self.in_channels])
        self.downsample_layers = nn.ModuleList([nn.Conv2d(256, 256,
            kernel_size=3, padding=1, stride=2) for _ in range(self.
            num_downsample)])

    def forward(self, backbone_outs):
        out = []
        x = torch.zeros(1, device=backbone_outs[0].device)
        for i in range(len(backbone_outs)):
            out.append(x)
        j = len(backbone_outs)
        for lat_layer in self.lat_layers:
            j -= 1
            if j < len(backbone_outs) - 1:
                _, _, h, w = backbone_outs[j].size()
                x = F.interpolate(x, size=(h, w), mode='bilinear',
                    align_corners=False)
            x = x + lat_layer(backbone_outs[j])
            out[j] = x
        j = len(backbone_outs)
        for pred_layer in self.pred_layers:
            j -= 1
            out[j] = F.relu(pred_layer(out[j]))
        for layer in self.downsample_layers:
            out.append(layer(out[-1]))
        return out


def construct_backbone(cfg_backbone):
    backbone = cfg_backbone.type(*cfg_backbone.args)
    num_layers = max(cfg_backbone.selected_layers) + 1
    while len(backbone.layers) < num_layers:
        backbone.add_layer()
    return backbone


_global_config['img_size'] = 4


_global_config['use_square_anchors'] = 4


def make_anchors(conv_h, conv_w, scale):
    prior_data = []
    for j, i in product(range(conv_h), range(conv_w)):
        x = (i + 0.5) / conv_w
        y = (j + 0.5) / conv_h
        for ar in cfg.aspect_ratios:
            ar = sqrt(ar)
            w = scale * ar / cfg.img_size
            h = scale / ar / cfg.img_size
            if cfg.use_square_anchors:
                h = w
            prior_data += [x, y, w, h]
    return prior_data


mask_proto_net = [(256, 3, {'padding': 1}), (256, 3, {'padding': 1}), (256,
    3, {'padding': 1}), (None, -2, {}), (256, 3, {'padding': 1}), (32, 1, {})]


_global_config['backbone'] = 4


_global_config['scales'] = 1.0


_global_config['freeze_bn'] = 4


_global_config['train_semantic'] = False


class Yolact(nn.Module):

    def __init__(self):
        super().__init__()
        self.anchors = []
        self.backbone = construct_backbone(cfg.backbone)
        if cfg.freeze_bn:
            self.freeze_bn()
        self.proto_net, coef_dim = make_net(256, mask_proto_net,
            include_last_relu=False)
        """  
        self.proto_net:
        Sequential((0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                   (1): ReLU(inplace)
                   (2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                   (3): ReLU(inplace)
                   (4): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                   (5): ReLU(inplace)
                   (6): InterpolateModule()
                   (7): ReLU(inplace)
                   (8): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                   (9): ReLU(inplace)
                   (10): Conv2d(256, 32, kernel_size=(1, 1), stride=(1, 1)))
        """
        self.fpn = FPN([512, 1024, 2048])
        self.selected_layers = [0, 1, 2, 3, 4]
        self.prediction_layers = nn.ModuleList()
        self.prediction_layers.append(PredictionModule(in_channels=256,
            coef_dim=coef_dim))
        """  
        self.prediction_layers:
        ModuleList(
          (0): PredictionModule((upfeature): Sequential((0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                                                        (1): ReLU(inplace))
                                (bbox_layer): Conv2d(256, 12, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                                (conf_layer): Conv2d(256, 243, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                                (mask_layer): Conv2d(256, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))))
        """
        if cfg.train_semantic:
            self.semantic_seg_conv = nn.Conv2d(256, cfg.num_classes - 1,
                kernel_size=1)

    def load_weights(self, path, cuda):
        if cuda:
            state_dict = torch.load(path)
        else:
            state_dict = torch.load(path, map_location='cpu')
        for key in list(state_dict.keys()):
            if key.startswith('fpn.downsample_layers.'):
                if int(key.split('.')[2]) >= 2:
                    del state_dict[key]
        self.load_state_dict(state_dict)

    def init_weights(self, backbone_path):
        self.backbone.init_backbone(backbone_path)
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d
                ) and module not in self.backbone.backbone_modules:
                nn.init.xavier_uniform_(module.weight.data)
                if module.bias is not None:
                    module.bias.data.zero_()

    def train(self, mode=True):
        super().train(mode)
        if cfg.freeze_bn:
            self.freeze_bn()

    def freeze_bn(self):
        """ Adapted from https://discuss.pytorch.org/t/how-to-train-with-frozen-batchnorm/12106/8 """
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
                module.weight.requires_grad = False
                module.bias.requires_grad = False

    def forward(self, x):
        with timer.env('backbone'):
            outs = self.backbone(x)
        with timer.env('fpn'):
            outs = [outs[i] for i in cfg.backbone.selected_layers]
            outs = self.fpn(outs)
            """
            outs:
            (n, 3, 550, 550) -> backbone -> (n, 256, 138, 138) -> fpn -> (n, 256, 69, 69) P3
                                            (n, 512, 69, 69)             (n, 256, 35, 35) P4
                                            (n, 1024, 35, 35)            (n, 256, 18, 18) P5
                                            (n, 2048, 18, 18)            (n, 256, 9, 9)   P6
                                                                         (n, 256, 5, 5)   P7
            """
        if isinstance(self.anchors, list):
            for i, shape in enumerate([list(aa.shape) for aa in outs]):
                self.anchors += make_anchors(shape[2], shape[3], cfg.scales[i])
            self.anchors = torch.Tensor(self.anchors).view(-1, 4)
        with timer.env('proto'):
            proto_out = self.proto_net(outs[0])
            proto_out = F.relu(proto_out, inplace=True)
            proto_out = proto_out.permute(0, 2, 3, 1).contiguous()
        with timer.env('pred_heads'):
            predictions = {'box': [], 'class': [], 'coef': []}
            for i in self.selected_layers:
                p = self.prediction_layers[0](outs[i])
                for k, v in p.items():
                    predictions[k].append(v)
        for k, v in predictions.items():
            predictions[k] = torch.cat(v, -2)
        predictions['proto'] = proto_out
        predictions['anchors'] = self.anchors
        if self.training:
            if cfg.train_semantic:
                predictions['segm'] = self.semantic_seg_conv(outs[0])
            return predictions
        else:
            predictions['class'] = F.softmax(predictions['class'], -1)
            return predictions


def center_size(boxes):
    """ Convert prior_boxes to format: (cx, cy, w, h)."""
    return torch.cat(((boxes[:, 2:] + boxes[:, :2]) / 2, boxes[:, 2:] -
        boxes[:, :2]), 1)


def sanitize_coordinates(_x1, _x2, img_size: int, padding: int=0):
    """
    Sanitizes the input coordinates so that x1 < x2, x1 != x2, x1 >= 0, and x2 <= image_size.
    Also converts from relative to absolute coordinates and casts the results to long tensors.

    Warning: this does things in-place behind the scenes so copy if necessary.
    """
    _x1 = _x1 * img_size
    _x2 = _x2 * img_size
    x1 = torch.min(_x1, _x2)
    x2 = torch.max(_x1, _x2)
    x1 = torch.clamp(x1 - padding, min=0)
    x2 = torch.clamp(x2 + padding, max=img_size)
    return x1, x2


def crop(masks, boxes, padding: int=1):
    """
    "Crop" predicted masks by zeroing out everything not in the predicted bbox.
    Args:
        - masks should be a size [h, w, n] tensor of masks
        - boxes should be a size [n, 4] tensor of bbox coords in relative point form
    """
    h, w, n = masks.size()
    x1, x2 = sanitize_coordinates(boxes[:, (0)], boxes[:, (2)], w, padding)
    y1, y2 = sanitize_coordinates(boxes[:, (1)], boxes[:, (3)], h, padding)
    rows = torch.arange(w, device=masks.device, dtype=x1.dtype).view(1, -1, 1
        ).expand(h, w, n)
    cols = torch.arange(h, device=masks.device, dtype=x1.dtype).view(-1, 1, 1
        ).expand(h, w, n)
    masks_left = rows >= x1.view(1, 1, -1)
    masks_right = rows < x2.view(1, 1, -1)
    masks_up = cols >= y1.view(1, 1, -1)
    masks_down = cols < y2.view(1, 1, -1)
    crop_mask = masks_left * masks_right * masks_up * masks_down
    return masks * crop_mask.float()


def encode(matched, priors):
    variances = [0.1, 0.2]
    g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - priors[:, :2]
    g_cxcy /= variances[0] * priors[:, 2:]
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    offsets = torch.cat([g_cxcy, g_wh], 1)
    return offsets


def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip(max_xy - min_xy, a_min=0, a_max=np.inf)
    return inter[:, (0)] * inter[:, (1)]


def jaccard(box_a, box_b, iscrowd: bool=False):
    """
    Compute the IoU of two sets of boxes.
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
        iscrowd: if True, put the crowd in box_b
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    use_batch = True
    if box_a.dim() == 2:
        use_batch = False
        box_a = box_a[None, ...]
        box_b = box_b[None, ...]
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, :, (2)] - box_a[:, :, (0)]) * (box_a[:, :, (3)] -
        box_a[:, :, (1)])).unsqueeze(2).expand_as(inter)
    area_b = ((box_b[:, :, (2)] - box_b[:, :, (0)]) * (box_b[:, :, (3)] -
        box_b[:, :, (1)])).unsqueeze(1).expand_as(inter)
    union = area_a + area_b - inter
    out = inter / area_a if iscrowd else inter / union
    return out if use_batch else out.squeeze(0)


_global_config['crowd_iou_threshold'] = 4


def match(pos_thresh, neg_thresh, box_gt, priors, class_gt, crowd_boxes):
    """
    Match each prior box with the ground truth box of the highest jaccard overlap, encode the bounding boxes,
    then return the matched indices corresponding to both confidence and location preds.
    Args:
        pos_thresh: (float) IoU > pos_thresh ==> positive.
        neg_thresh: (float) IoU < neg_thresh ==> negative.
        box_gt: (tensor) Ground truth boxes, Shape: [num_obj, 4], (xmin, ymin, xmax, ymax).
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors, 4], (center_x, center_y, w, h).
        class_gt: (tensor) All the class labels for the image, Shape: [num_obj].
        crowd_boxes: (tensor) All the crowd box annotations or None if there are none.

    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    priors = priors.data
    decoded_priors = torch.cat((priors[:, :2] - priors[:, 2:] / 2, priors[:,
        :2] + priors[:, 2:] / 2), 1)
    overlaps = jaccard(box_gt, decoded_priors)
    each_box_max, each_box_index = overlaps.max(1)
    each_prior_max, each_prior_index = overlaps.max(0)
    each_prior_max.index_fill_(0, each_box_index, 2)
    for j in range(each_box_index.size(0)):
        each_prior_index[each_box_index[j]] = j
    each_prior_box = box_gt[each_prior_index]
    conf = class_gt[each_prior_index] + 1
    conf[each_prior_max < pos_thresh] = -1
    conf[each_prior_max < neg_thresh] = 0
    if crowd_boxes is not None and cfg.crowd_iou_threshold < 1:
        crowd_overlaps = jaccard(decoded_priors, crowd_boxes, iscrowd=True)
        best_crowd_overlap, best_crowd_idx = crowd_overlaps.max(1)
        conf[(conf <= 0) & (best_crowd_overlap > cfg.crowd_iou_threshold)] = -1
    offsets = encode(each_prior_box, priors)
    return offsets, conf, each_prior_box, each_prior_index


_global_config['mask_alpha'] = 4


_global_config['conf_alpha'] = 4


_global_config['bbox_alpha'] = 4


_global_config['masks_to_train'] = False


_global_config['semantic_alpha'] = 4


class Multi_Loss(nn.Module):

    def __init__(self, num_classes, pos_thre, neg_thre, np_ratio):
        super().__init__()
        self.num_classes = num_classes
        self.pos_thre = pos_thre
        self.neg_thre = neg_thre
        self.negpos_ratio = np_ratio

    def ohem_conf_loss(self, class_p, conf_gt, positive_bool):
        batch_conf = class_p.view(-1, self.num_classes)
        batch_conf_max = batch_conf.data.max()
        mark = torch.log(torch.sum(torch.exp(batch_conf - batch_conf_max), 1)
            ) + batch_conf_max - batch_conf[:, (0)]
        mark = mark.view(class_p.size(0), -1)
        mark[positive_bool] = 0
        mark[conf_gt < 0] = 0
        _, idx = mark.sort(1, descending=True)
        _, idx_rank = idx.sort(1)
        num_pos = positive_bool.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio * num_pos, max=
            positive_bool.size(1) - 1)
        negative_bool = idx_rank < num_neg.expand_as(idx_rank)
        negative_bool[positive_bool] = 0
        negative_bool[conf_gt < 0] = 0
        class_p_selected = class_p[positive_bool + negative_bool].view(-1,
            self.num_classes)
        class_gt_selected = conf_gt[positive_bool + negative_bool]
        loss_c = F.cross_entropy(class_p_selected, class_gt_selected,
            reduction='sum')
        return cfg.conf_alpha * loss_c

    @staticmethod
    def bbox_loss(pos_box_p, pos_offsets):
        loss_b = F.smooth_l1_loss(pos_box_p, pos_offsets, reduction='sum'
            ) * cfg.bbox_alpha
        return loss_b

    @staticmethod
    def lincomb_mask_loss(positive_bool, prior_max_index, coef_p, proto_p,
        mask_gt, prior_max_box):
        proto_h = proto_p.size(1)
        proto_w = proto_p.size(2)
        loss_m = 0
        for i in range(coef_p.size(0)):
            with torch.no_grad():
                downsampled_masks = F.interpolate(mask_gt[i].unsqueeze(0),
                    (proto_h, proto_w), mode='bilinear', align_corners=False
                    ).squeeze(0)
                downsampled_masks = downsampled_masks.permute(1, 2, 0
                    ).contiguous()
                downsampled_masks = downsampled_masks.gt(0.5).float()
            pos_prior_index = prior_max_index[i, positive_bool[i]]
            pos_prior_box = prior_max_box[i, positive_bool[i]]
            pos_coef = coef_p[i, positive_bool[i]]
            if pos_prior_index.size(0) == 0:
                continue
            old_num_pos = pos_coef.size(0)
            if old_num_pos > cfg.masks_to_train:
                perm = torch.randperm(pos_coef.size(0))
                select = perm[:cfg.masks_to_train]
                pos_coef = pos_coef[select]
                pos_prior_index = pos_prior_index[select]
                pos_prior_box = pos_prior_box[select]
            num_pos = pos_coef.size(0)
            pos_mask_gt = downsampled_masks[:, :, (pos_prior_index)]
            mask_p = torch.sigmoid(proto_p[i] @ pos_coef.t())
            mask_p = crop(mask_p, pos_prior_box)
            mask_loss = F.binary_cross_entropy(torch.clamp(mask_p, 0, 1),
                pos_mask_gt, reduction='none')
            pos_get_csize = center_size(pos_prior_box)
            mask_loss = mask_loss.sum(dim=(0, 1)) / pos_get_csize[:, (2)
                ] / pos_get_csize[:, (3)]
            if old_num_pos > num_pos:
                mask_loss *= old_num_pos / num_pos
            loss_m += torch.sum(mask_loss)
        loss_m *= cfg.mask_alpha / proto_h / proto_w
        return loss_m

    @staticmethod
    def semantic_segmentation_loss(segmentation_p, mask_gt, class_gt):
        batch_size, num_classes, mask_h, mask_w = segmentation_p.size()
        loss_s = 0
        for i in range(batch_size):
            cur_segment = segmentation_p[i]
            cur_class_gt = class_gt[i]
            with torch.no_grad():
                downsampled_masks = F.interpolate(mask_gt[i].unsqueeze(0),
                    (mask_h, mask_w), mode='bilinear', align_corners=False
                    ).squeeze(0)
                downsampled_masks = downsampled_masks.gt(0.5).float()
                segment_gt = torch.zeros_like(cur_segment, requires_grad=False)
                for i_obj in range(downsampled_masks.size(0)):
                    segment_gt[cur_class_gt[i_obj]] = torch.max(segment_gt[
                        cur_class_gt[i_obj]], downsampled_masks[i_obj])
            loss_s += F.binary_cross_entropy_with_logits(cur_segment,
                segment_gt, reduction='sum')
        return loss_s / mask_h / mask_w * cfg.semantic_alpha

    def forward(self, predictions, box_class, mask_gt, num_crowds):
        box_p = predictions['box']
        class_p = predictions['class']
        coef_p = predictions['coef']
        anchors = predictions['anchors']
        proto_p = predictions['proto']
        class_gt = [None] * len(box_class)
        batch_size = box_p.size(0)
        anchors = anchors[:box_p.size(1), :]
        num_priors = anchors.size(0)
        all_offsets = box_p.new(batch_size, num_priors, 4)
        conf_gt = box_p.new(batch_size, num_priors).long()
        prior_max_box = box_p.new(batch_size, num_priors, 4)
        prior_max_index = box_p.new(batch_size, num_priors).long()
        for i in range(batch_size):
            box_gt = box_class[i][:, :-1].data
            class_gt[i] = box_class[i][:, (-1)].data.long()
            cur_crowds = num_crowds[i]
            if cur_crowds > 0:
                split = lambda x: (x[-cur_crowds:], x[:-cur_crowds])
                crowd_boxes, box_gt = split(box_gt)
                _, class_gt[i] = split(class_gt[i])
                _, mask_gt[i] = split(mask_gt[i])
            else:
                crowd_boxes = None
            all_offsets[i], conf_gt[i], prior_max_box[i], prior_max_index[i
                ] = match(self.pos_thre, self.neg_thre, box_gt, anchors,
                class_gt[i], crowd_boxes)
        all_offsets = Variable(all_offsets, requires_grad=False)
        conf_gt = Variable(conf_gt, requires_grad=False)
        prior_max_index = Variable(prior_max_index, requires_grad=False)
        losses = {}
        positive_bool = conf_gt > 0
        num_pos = positive_bool.sum(dim=1, keepdim=True)
        pos_box_p = box_p[(positive_bool), :]
        pos_offsets = all_offsets[(positive_bool), :]
        losses['B'] = self.bbox_loss(pos_box_p, pos_offsets)
        losses['M'] = self.lincomb_mask_loss(positive_bool, prior_max_index,
            coef_p, proto_p, mask_gt, prior_max_box)
        losses['C'] = self.ohem_conf_loss(class_p, conf_gt, positive_bool)
        if cfg.train_semantic:
            losses['S'] = self.semantic_segmentation_loss(predictions[
                'segm'], mask_gt, class_gt)
        total_num_pos = num_pos.data.sum().float()
        for aa in losses:
            if aa != 'S':
                losses[aa] /= total_num_pos
            else:
                losses[aa] /= batch_size
        return losses


MEANS = 103.94, 116.78, 123.68


STD = 57.38, 57.12, 58.4


class FastBaseTransform(torch.nn.Module):
    """
    Transform that does all operations on the GPU for super speed.
    This doesn't suppport a lot of config settings and should only be used for production.
    Maintain this as necessary.
    """

    def __init__(self):
        super().__init__()
        self.mean = torch.Tensor(MEANS).float()[(None), :, (None), (None)]
        self.std = torch.Tensor(STD).float()[(None), :, (None), (None)]
        self.transform = cfg.backbone.transform

    def forward(self, img):
        self.mean = self.mean
        self.std = self.std
        img = img.permute(0, 3, 1, 2).contiguous()
        img = F.interpolate(img, (cfg.img_size, cfg.img_size), mode=
            'bilinear', align_corners=False)
        if self.transform.normalize:
            img = (img - self.mean) / self.std
        elif self.transform.subtract_means:
            img = img - self.mean
        elif self.transform.to_float:
            img = img / 255
        if self.transform.channel_order != 'RGB':
            raise NotImplementedError
        img = img[:, (2, 1, 0), :, :].contiguous()
        return img


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_feiyuhuahuo_Yolact_minimal(_paritybench_base):
    pass
