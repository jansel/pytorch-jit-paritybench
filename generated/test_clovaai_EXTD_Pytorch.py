import sys
_module = sys.modules[__name__]
del sys
EXTD_32 = _module
EXTD_48 = _module
EXTD_64 = _module
bbox_setup = _module
config = _module
egohand = _module
factory = _module
vochead = _module
widerface = _module
demo = _module
evaluation = _module
bbox_utils = _module
detection = _module
prior_box = _module
l2norm = _module
multibox_loss = _module
logger = _module
mobileFacenet_32_PReLU = _module
mobileFacenet_48_PReLU = _module
mobileFacenet_64_PReLU = _module
prepare_hand_dataset = _module
prepare_wider_data = _module
setup = _module
afw_test = _module
anchor_matching_test = _module
detect = _module
eval_hand = _module
eval_head = _module
fddb_test = _module
pascal_test = _module
wider_test = _module
train = _module
utils = _module
augmentations = _module
wider_test = _module

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


import torch.nn.init as init


import torch.nn.functional as F


from torch.autograd import Variable


import numpy as np


import torch.utils.data as data


import torch.backends.cudnn as cudnn


from torch.autograd import Function


import math


import scipy.io as sio


import torch.optim as optim


def nms(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """
    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, (0)]
    y1 = boxes[:, (1)]
    x2 = boxes[:, (2)]
    y2 = boxes[:, (3)]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)
    idx = idx[-top_k:]
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w * h
        rem_areas = torch.index_select(area, 0, idx)
        union = rem_areas - inter + area[i]
        IoU = inter / union
        idx = idx[IoU.le(overlap)]
    return keep, count


def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """
    boxes = torch.cat((priors[:, :2] + loc[:, :2] * variances[0] * priors[:,
        2:], priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


class Detect(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """

    def __init__(self, cfg):
        self.num_classes = cfg.NUM_CLASSES
        self.top_k = cfg.TOP_K
        self.nms_thresh = cfg.NMS_THRESH
        self.conf_thresh = cfg.CONF_THRESH
        self.variance = cfg.VARIANCE
        self.nms_top_k = cfg.NMS_TOP_K

    def forward(self, loc_data, conf_data, prior_data):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4] 
        """
        num = loc_data.size(0)
        num_priors = prior_data.size(0)
        conf_preds = conf_data.view(num, num_priors, self.num_classes
            ).transpose(2, 1)
        batch_priors = prior_data.view(-1, num_priors, 4).expand(num,
            num_priors, 4)
        batch_priors = batch_priors.contiguous().view(-1, 4)
        decoded_boxes = decode(loc_data.view(-1, 4), batch_priors, self.
            variance)
        decoded_boxes = decoded_boxes.view(num, num_priors, 4)
        output = torch.zeros(num, self.num_classes, self.top_k, 5)
        for i in range(num):
            boxes = decoded_boxes[i].clone()
            conf_scores = conf_preds[i].clone()
            for cl in range(1, self.num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]
                if scores.dim() == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(boxes)
                boxes_ = boxes[l_mask].view(-1, 4)
                try:
                    ids, count = nms(boxes_, scores, self.nms_thresh, self.
                        nms_top_k)
                    count = count if count < self.top_k else self.top_k
                    output[(i), (cl), :count] = torch.cat((scores[ids[:
                        count]].unsqueeze(1), boxes_[ids[:count]]), 1)
                except:
                    print('zero')
        return output


def upsample(in_channels, out_channels):
    return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=
        in_channels, kernel_size=(3, 3), stride=1, padding=1, groups=
        in_channels, bias=False), nn.Conv2d(in_channels=in_channels,
        out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias
        =False), nn.BatchNorm2d(out_channels), nn.ReLU())


class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """

    def __init__(self, input_size, feature_maps, cfg):
        super(PriorBox, self).__init__()
        self.imh = input_size[0]
        self.imw = input_size[1]
        self.variance = cfg.VARIANCE or [0.1]
        self.min_sizes = cfg.ANCHOR_SIZES
        self.steps = cfg.STEPS
        self.clip = cfg.CLIP
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')
        self.feature_maps = feature_maps

    def forward(self):
        mean = []
        for k in range(len(self.feature_maps)):
            feath = self.feature_maps[k][0]
            featw = self.feature_maps[k][1]
            for i, j in product(range(feath), range(featw)):
                f_kw = self.imw / self.steps[k]
                f_kh = self.imh / self.steps[k]
                cx = (j + 0.5) / f_kw
                cy = (i + 0.5) / f_kh
                s_kw = self.min_sizes[k] / self.imw
                s_kh = self.min_sizes[k] / self.imh
                mean += [cx, cy, s_kw, s_kh]
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


class EXTD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.


    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, base, head, num_classes):
        super(EXTD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        """
        self.priorbox = PriorBox(size,cfg)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        """
        self.base = nn.ModuleList(base)
        self.upfeat = []
        for it in range(5):
            self.upfeat.append(upsample(in_channels=32, out_channels=32))
        self.upfeat = nn.ModuleList(self.upfeat)
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(cfg)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        size = x.size()[2:]
        sources = list()
        loc = list()
        conf = list()
        for k in range(8):
            x = self.base[k](x)
        s1 = x
        for k in range(2, 8):
            x = self.base[k](x)
        s2 = x
        for k in range(2, 8):
            x = self.base[k](x)
        s3 = x
        for k in range(2, 8):
            x = self.base[k](x)
        s4 = x
        for k in range(2, 8):
            x = self.base[k](x)
        s5 = x
        for k in range(2, 8):
            x = self.base[k](x)
        s6 = x
        sources.append(s6)
        u1 = self.upfeat[0](F.interpolate(s6, size=(s5.size()[2], s5.size()
            [3]), mode='bilinear')) + s5
        sources.append(u1)
        u2 = self.upfeat[1](F.interpolate(u1, size=(s4.size()[2], s4.size()
            [3]), mode='bilinear')) + s4
        sources.append(u2)
        u3 = self.upfeat[2](F.interpolate(u2, size=(s3.size()[2], s3.size()
            [3]), mode='bilinear')) + s3
        sources.append(u3)
        u4 = self.upfeat[3](F.interpolate(u3, size=(s2.size()[2], s2.size()
            [3]), mode='bilinear')) + s2
        sources.append(u4)
        u5 = self.upfeat[4](F.interpolate(u4, size=(s1.size()[2], s1.size()
            [3]), mode='bilinear')) + s1
        sources.append(u5)
        sources = sources[::-1]
        loc_x = self.loc[0](sources[0])
        conf_x = self.conf[0](sources[0])
        max_conf, _ = torch.max(conf_x[:, 0:3, :, :], dim=1, keepdim=True)
        conf_x = torch.cat((max_conf, conf_x[:, 3:, :, :]), dim=1)
        loc.append(loc_x.permute(0, 2, 3, 1).contiguous())
        conf.append(conf_x.permute(0, 2, 3, 1).contiguous())
        for i in range(1, len(sources)):
            x = sources[i]
            conf.append(self.conf[i](x).permute(0, 2, 3, 1).contiguous())
            loc.append(self.loc[i](x).permute(0, 2, 3, 1).contiguous())
        """
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        """
        features_maps = []
        for i in range(len(loc)):
            feat = []
            feat += [loc[i].size(1), loc[i].size(2)]
            features_maps += [feat]
        self.priorbox = PriorBox(size, features_maps, cfg)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == 'test':
            output = self.detect(loc.view(loc.size(0), -1, 4), self.softmax
                (conf.view(conf.size(0), -1, self.num_classes)), self.
                priors.type(type(x.data)))
        else:
            output = loc.view(loc.size(0), -1, 4), conf.view(conf.size(0), 
                -1, self.num_classes), self.priors
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            None
            mdata = torch.load(base_file, map_location=lambda storage, loc:
                storage)
            weights = mdata['weight']
            epoch = mdata['epoch']
            self.load_state_dict(weights)
            None
        else:
            None
        return epoch

    def xavier(self, param):
        init.xavier_uniform(param)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            self.xavier(m.weight.data)
            if torch.is_tensor(m.bias):
                m.bias.data.zero_()


class EXTD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.


    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, base, head, num_classes):
        super(EXTD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.base = nn.ModuleList(base)
        self.upfeat = []
        for it in range(5):
            self.upfeat.append(upsample(in_channels=48, out_channels=48))
        self.upfeat = nn.ModuleList(self.upfeat)
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(cfg)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        size = x.size()[2:]
        sources = list()
        loc = list()
        conf = list()
        for k in range(6):
            x = self.base[k](x)
        s1 = x
        for k in range(2, 6):
            x = self.base[k](x)
        s2 = x
        for k in range(2, 6):
            x = self.base[k](x)
        s3 = x
        for k in range(2, 6):
            x = self.base[k](x)
        s4 = x
        for k in range(2, 6):
            x = self.base[k](x)
        s5 = x
        for k in range(2, 6):
            x = self.base[k](x)
        s6 = x
        sources.append(s6)
        u1 = self.upfeat[0](F.interpolate(s6, size=(s5.size()[2], s5.size()
            [3]), mode='bilinear')) + s5
        sources.append(u1)
        u2 = self.upfeat[1](F.interpolate(u1, size=(s4.size()[2], s4.size()
            [3]), mode='bilinear')) + s4
        sources.append(u2)
        u3 = self.upfeat[2](F.interpolate(u2, size=(s3.size()[2], s3.size()
            [3]), mode='bilinear')) + s3
        sources.append(u3)
        u4 = self.upfeat[3](F.interpolate(u3, size=(s2.size()[2], s2.size()
            [3]), mode='bilinear')) + s2
        sources.append(u4)
        u5 = self.upfeat[4](F.interpolate(u4, size=(s1.size()[2], s1.size()
            [3]), mode='bilinear')) + s1
        sources.append(u5)
        sources = sources[::-1]
        loc_x = self.loc[0](sources[0])
        conf_x = self.conf[0](sources[0])
        max_conf, _ = torch.max(conf_x[:, 0:3, :, :], dim=1, keepdim=True)
        conf_x = torch.cat((max_conf, conf_x[:, 3:, :, :]), dim=1)
        loc.append(loc_x.permute(0, 2, 3, 1).contiguous())
        conf.append(conf_x.permute(0, 2, 3, 1).contiguous())
        for i in range(1, len(sources)):
            x = sources[i]
            conf.append(self.conf[i](x).permute(0, 2, 3, 1).contiguous())
            loc.append(self.loc[i](x).permute(0, 2, 3, 1).contiguous())
        features_maps = []
        for i in range(len(loc)):
            feat = []
            feat += [loc[i].size(1), loc[i].size(2)]
            features_maps += [feat]
        self.priorbox = PriorBox(size, features_maps, cfg)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == 'test':
            output = self.detect(loc.view(loc.size(0), -1, 4), self.softmax
                (conf.view(conf.size(0), -1, self.num_classes)), self.
                priors.type(type(x.data)))
        else:
            output = loc.view(loc.size(0), -1, 4), conf.view(conf.size(0), 
                -1, self.num_classes), self.priors
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            None
            mdata = torch.load(base_file, map_location=lambda storage, loc:
                storage)
            weights = mdata['weight']
            epoch = mdata['epoch']
            self.load_state_dict(weights)
            None
        else:
            None
        return epoch

    def xavier(self, param):
        init.xavier_uniform(param)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            self.xavier(m.weight.data)
            if torch.is_tensor(m.bias):
                m.bias.data.zero_()


class EXTD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.


    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, base, head, num_classes):
        super(EXTD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.base = nn.ModuleList(base)
        self.upfeat = []
        for it in range(5):
            self.upfeat.append(upsample(in_channels=64, out_channels=64))
        self.upfeat = nn.ModuleList(self.upfeat)
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(cfg)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        size = x.size()[2:]
        sources = list()
        loc = list()
        conf = list()
        for k in range(6):
            x = self.base[k](x)
        s1 = x
        for k in range(2, 6):
            x = self.base[k](x)
        s2 = x
        for k in range(2, 6):
            x = self.base[k](x)
        s3 = x
        for k in range(2, 6):
            x = self.base[k](x)
        s4 = x
        for k in range(2, 6):
            x = self.base[k](x)
        s5 = x
        for k in range(2, 6):
            x = self.base[k](x)
        s6 = x
        sources.append(s6)
        u1 = self.upfeat[0](F.interpolate(s6, size=(s5.size()[2], s5.size()
            [3]), mode='bilinear')) + s5
        sources.append(u1)
        u2 = self.upfeat[1](F.interpolate(u1, size=(s4.size()[2], s4.size()
            [3]), mode='bilinear')) + s4
        sources.append(u2)
        u3 = self.upfeat[2](F.interpolate(u2, size=(s3.size()[2], s3.size()
            [3]), mode='bilinear')) + s3
        sources.append(u3)
        u4 = self.upfeat[3](F.interpolate(u3, size=(s2.size()[2], s2.size()
            [3]), mode='bilinear')) + s2
        sources.append(u4)
        u5 = self.upfeat[4](F.interpolate(u4, size=(s1.size()[2], s1.size()
            [3]), mode='bilinear')) + s1
        sources.append(u5)
        sources = sources[::-1]
        loc_x = self.loc[0](sources[0])
        conf_x = self.conf[0](sources[0])
        max_conf, _ = torch.max(conf_x[:, 0:3, :, :], dim=1, keepdim=True)
        conf_x = torch.cat((max_conf, conf_x[:, 3:, :, :]), dim=1)
        loc.append(loc_x.permute(0, 2, 3, 1).contiguous())
        conf.append(conf_x.permute(0, 2, 3, 1).contiguous())
        for i in range(1, len(sources)):
            x = sources[i]
            conf.append(self.conf[i](x).permute(0, 2, 3, 1).contiguous())
            loc.append(self.loc[i](x).permute(0, 2, 3, 1).contiguous())
        features_maps = []
        for i in range(len(loc)):
            feat = []
            feat += [loc[i].size(1), loc[i].size(2)]
            features_maps += [feat]
        self.priorbox = PriorBox(size, features_maps, cfg)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == 'test':
            output = self.detect(loc.view(loc.size(0), -1, 4), self.softmax
                (conf.view(conf.size(0), -1, self.num_classes)), self.
                priors.type(type(x.data)))
        else:
            output = loc.view(loc.size(0), -1, 4), conf.view(conf.size(0), 
                -1, self.num_classes), self.priors
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            None
            mdata = torch.load(base_file, map_location=lambda storage, loc:
                storage)
            weights = mdata['weight']
            epoch = mdata['epoch']
            self.load_state_dict(weights)
            None
        else:
            None
        return epoch

    def xavier(self, param):
        init.xavier_uniform(param)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            self.xavier(m.weight.data)
            if torch.is_tensor(m.bias):
                m.bias.data.zero_()


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


def encode(matched, priors, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 4].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded boxes (tensor), Shape: [num_priors, 4]
    """
    g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - priors[:, :2]
    g_cxcy /= variances[0] * priors[:, 2:]
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    return torch.cat([g_cxcy, g_wh], 1)


def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes
        [:, 2:] / 2), 1)


def match_ssd(threshold, truths, priors, variances, labels, loc_t, conf_t, idx
    ):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    overlaps = jaccard(truths, point_form(priors))
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    matches = truths[best_truth_idx]
    conf = labels[best_truth_idx]
    conf[best_truth_overlap < threshold] = 0
    loc = encode(matches, priors, variances)
    loc_t[idx] = loc
    conf_t[idx] = conf


def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x - x_max), 1, keepdim=True)) + x_max


def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    overlaps = jaccard(truths, point_form(priors))
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    _th1, _th2, _th3 = threshold
    N = (torch.sum(best_prior_overlap >= _th2) + torch.sum(
        best_prior_overlap >= _th3)) // 2
    matches = truths[best_truth_idx]
    conf = labels[best_truth_idx]
    conf[best_truth_overlap < _th2] = 0
    best_truth_overlap_clone = best_truth_overlap.clone()
    add_idx = best_truth_overlap_clone.gt(_th1).eq(best_truth_overlap_clone
        .lt(_th2))
    best_truth_overlap_clone[1 - add_idx] = 0
    stage2_overlap, stage2_idx = best_truth_overlap_clone.sort(descending=True)
    stage2_overlap = stage2_overlap.gt(_th1)
    if N > 0:
        N = torch.sum(stage2_overlap[:N]) if torch.sum(stage2_overlap[:N]
            ) < N else N
        conf[stage2_idx[:N]] += 1
    loc = encode(matches, priors, variances)
    loc_t[idx] = loc
    conf_t[idx] = conf


class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, cfg, dataset, use_gpu=True):
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = cfg.NUM_CLASSES
        self.negpos_ratio = cfg.NEG_POS_RATIOS
        self.variance = cfg.VARIANCE
        self.dataset = dataset
        if dataset == 'face':
            self.threshold = cfg.FACE.OVERLAP_THRESH
            self.match = match
        elif dataset == 'hand':
            self.threshold = cfg.HAND.OVERLAP_THRESH
            self.match = match_ssd
        else:
            self.threshold = cfg.HEAD.OVERLAP_THRESH
            self.match = match

    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            targets (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        loc_data, conf_data, priors = predictions
        num = loc_data.size(0)
        priors = priors[:loc_data.size(1), :]
        num_priors = priors.size(0)
        num_classes = self.num_classes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, (-1)].data
            defaults = priors.data
            self.match(self.threshold, truths, defaults, self.variance,
                labels, loc_t, conf_t, idx)
        if self.use_gpu:
            loc_t = loc_t
            conf_t = conf_t
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)
        pos = conf_t > 0
        num_pos = pos.sum(dim=1, keepdim=True)
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view
            (-1, 1))
        loss_c[pos.view(-1, 1)] = 0
        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio * num_pos, max=pos.size(1) - 1)
        neg = idx_rank < num_neg.expand_as(idx_rank)
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes
            )
        targets_weighted = conf_t[(pos + neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)
        N = num_pos.data.sum() if num_pos.data.sum() > 0 else num
        loss_l /= N.float()
        loss_c /= N.float()
        return loss_l, loss_c


class DWC(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DWC, self).__init__()
        self.batch_norm_in = nn.BatchNorm2d(in_channels)
        self.depthwise = nn.AvgPool2d((7, 6), stride=1, padding=0)
        self.pointwise = nn.Conv2d(in_channels=in_channels, out_channels=
            out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class Max_AvgPool(nn.Module):

    def __init__(self, kernel_size=(3, 3), stride=2, padding=1, dim=128):
        super(Max_AvgPool, self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride,
            padding=padding)
        self.Avgpool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride,
            padding=padding)

    def forward(self, x):
        x = self.Maxpool(x) + self.Avgpool(x)
        return x


class Max_AvgPool(nn.Module):

    def __init__(self, kernel_size=(3, 3), stride=2, padding=1, dim=128):
        super(Max_AvgPool, self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride,
            padding=padding)
        self.Avgpool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride,
            padding=padding)

    def forward(self, x):
        x = self.Maxpool(x) + self.Avgpool(x)
        return x


class gated_conv1x1(nn.Module):

    def __init__(self, inc=128, outc=128):
        super(gated_conv1x1, self).__init__()
        self.inp = int(inc / 2)
        self.oup = int(outc / 2)
        self.conv1x1_1 = nn.Conv2d(self.inp, self.oup, 1, 1, 0, bias=False)
        self.gate_1 = nn.Conv2d(self.inp, self.oup, 1, 1, 0, bias=True)
        self.conv1x1_2 = nn.Conv2d(self.inp, self.oup, 1, 1, 0, bias=False)
        self.gate_2 = nn.Conv2d(self.inp, self.oup, 1, 1, 0, bias=True)

    def forward(self, x):
        x_1 = x[:, :self.inp, :, :]
        x_2 = x[:, self.inp:, :, :]
        a_1 = self.conv1x1_1(x_1)
        g_1 = F.sigmoid(self.gate_1(x_1))
        a_2 = self.conv1x1_2(x_2)
        g_2 = F.sigmoid(self.gate_2(x_2))
        ret = torch.cat((a_1 * g_1, a_2 * g_2), 1)
        return ret


class InvertedResidual_dwc(nn.Module):

    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual_dwc, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup
        self.conv = []
        if expand_ratio == 1:
            self.conv.append(nn.Conv2d(inp, hidden_dim, kernel_size=(3, 3),
                stride=stride, padding=1, groups=hidden_dim))
            self.conv.append(nn.BatchNorm2d(hidden_dim))
            self.conv.append(nn.PReLU())
            self.conv.append(nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False))
            self.conv.append(nn.BatchNorm2d(oup))
            self.conv.append(nn.PReLU())
        else:
            self.conv.append(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False))
            self.conv.append(nn.BatchNorm2d(hidden_dim))
            self.conv.append(nn.PReLU())
            self.conv.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=
                (3, 3), stride=stride, padding=1, groups=hidden_dim))
            self.conv.append(nn.BatchNorm2d(hidden_dim))
            self.conv.append(nn.PReLU())
            self.conv.append(nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False))
            self.conv.append(nn.BatchNorm2d(oup))
            self.conv.append(nn.PReLU())
        self.conv = nn.Sequential(*self.conv)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup
        self.conv = []
        if expand_ratio == 1:
            self.conv.append(nn.MaxPool2d(kernel_size=(3, 3), stride=stride,
                padding=1))
            self.conv.append(nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False))
            self.conv.append(nn.BatchNorm2d(oup))
        else:
            self.conv.append(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False))
            self.conv.append(nn.BatchNorm2d(hidden_dim))
            self.conv.append(nn.PReLU())
            self.conv.append(nn.MaxPool2d(kernel_size=(3, 3), stride=stride,
                padding=1))
            self.conv.append(nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False))
            self.conv.append(nn.BatchNorm2d(oup))
        self.conv = nn.Sequential(*self.conv)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


def conv_bn(inp, oup, stride, k_size=3):
    return nn.Sequential(nn.Conv2d(inp, oup, k_size, stride, 1, bias=False),
        nn.BatchNorm2d(oup), nn.PReLU())


class Net(nn.Module):

    def __init__(self, embedding_size=128, input_size=224, width_mult=1.0):
        super(Net, self).__init__()
        block = InvertedResidual
        block_dwc = InvertedResidual_dwc
        input_channel = 64
        last_channel = 256
        interverted_residual_setting = [[1, 32, 1, 1], [2, 32, 2, 1], [4, 
            32, 2, 1], [2, 32, 2, 2], [4, 32, 5, 1], [2, 32, 2, 2], [2, 32,
            6, 2]]
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult
            ) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        cnt = 0
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if cnt > 1:
                    if i == n - 1:
                        self.features.append(block_dwc(input_channel,
                            output_channel, s, expand_ratio=t))
                    else:
                        self.features.append(block_dwc(input_channel,
                            output_channel, 1, expand_ratio=t))
                    input_channel = output_channel
                else:
                    if i == n - 1:
                        self.features.append(block_dwc(input_channel,
                            output_channel, s, expand_ratio=t))
                    else:
                        self.features.append(block_dwc(input_channel,
                            output_channel, 1, expand_ratio=t))
                    input_channel = output_channel
            cnt += 1
        self.features.append(gated_conv1x1(input_channel, self.last_channel))
        self.features_sequential = nn.Sequential(*self.features)
        self._initialize_weights()

    def forward(self, x):
        x = self.features_sequential(x).view(-1, 256 * 4)
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


class DWC(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DWC, self).__init__()
        self.batch_norm_in = nn.BatchNorm2d(in_channels)
        self.depthwise = nn.AvgPool2d((7, 6), stride=1, padding=0)
        self.pointwise = nn.Conv2d(in_channels=in_channels, out_channels=
            out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class Max_AvgPool(nn.Module):

    def __init__(self, kernel_size=(3, 3), stride=2, padding=1, dim=128):
        super(Max_AvgPool, self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride,
            padding=padding)
        self.Avgpool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride,
            padding=padding)

    def forward(self, x):
        x = self.Maxpool(x) + self.Avgpool(x)
        return x


class Max_AvgPool(nn.Module):

    def __init__(self, kernel_size=(3, 3), stride=2, padding=1, dim=128):
        super(Max_AvgPool, self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride,
            padding=padding)
        self.Avgpool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride,
            padding=padding)

    def forward(self, x):
        x = self.Maxpool(x) + self.Avgpool(x)
        return x


class gated_conv1x1(nn.Module):

    def __init__(self, inc=128, outc=128):
        super(gated_conv1x1, self).__init__()
        self.inp = int(inc / 2)
        self.oup = int(outc / 2)
        self.conv1x1_1 = nn.Conv2d(self.inp, self.oup, 1, 1, 0, bias=False)
        self.gate_1 = nn.Conv2d(self.inp, self.oup, 1, 1, 0, bias=True)
        self.conv1x1_2 = nn.Conv2d(self.inp, self.oup, 1, 1, 0, bias=False)
        self.gate_2 = nn.Conv2d(self.inp, self.oup, 1, 1, 0, bias=True)

    def forward(self, x):
        x_1 = x[:, :self.inp, :, :]
        x_2 = x[:, self.inp:, :, :]
        a_1 = self.conv1x1_1(x_1)
        g_1 = F.sigmoid(self.gate_1(x_1))
        a_2 = self.conv1x1_2(x_2)
        g_2 = F.sigmoid(self.gate_2(x_2))
        ret = torch.cat((a_1 * g_1, a_2 * g_2), 1)
        return ret


class InvertedResidual_dwc(nn.Module):

    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual_dwc, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup
        self.conv = []
        if expand_ratio == 1:
            self.conv.append(nn.Conv2d(inp, hidden_dim, kernel_size=(3, 3),
                stride=stride, padding=1, groups=hidden_dim))
            self.conv.append(nn.BatchNorm2d(hidden_dim))
            self.conv.append(nn.PReLU())
            self.conv.append(nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False))
            self.conv.append(nn.BatchNorm2d(oup))
        else:
            self.conv.append(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False))
            self.conv.append(nn.BatchNorm2d(hidden_dim))
            self.conv.append(nn.PReLU())
            self.conv.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=
                (3, 3), stride=stride, padding=1, groups=hidden_dim))
            self.conv.append(nn.BatchNorm2d(hidden_dim))
            self.conv.append(nn.PReLU())
            self.conv.append(nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False))
            self.conv.append(nn.BatchNorm2d(oup))
        self.conv = nn.Sequential(*self.conv)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup
        self.conv = []
        if expand_ratio == 1:
            self.conv.append(nn.MaxPool2d(kernel_size=(3, 3), stride=stride,
                padding=1))
            self.conv.append(nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False))
            self.conv.append(nn.BatchNorm2d(oup))
        else:
            self.conv.append(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False))
            self.conv.append(nn.BatchNorm2d(hidden_dim))
            self.conv.append(nn.PReLU())
            self.conv.append(nn.MaxPool2d(kernel_size=(3, 3), stride=stride,
                padding=1))
            self.conv.append(nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False))
            self.conv.append(nn.BatchNorm2d(oup))
        self.conv = nn.Sequential(*self.conv)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class Net(nn.Module):

    def __init__(self, embedding_size=128, input_size=224, width_mult=1.0):
        super(Net, self).__init__()
        block = InvertedResidual
        block_dwc = InvertedResidual_dwc
        input_channel = 64
        last_channel = 256
        interverted_residual_setting = [[1, 48, 1, 1], [2, 48, 2, 1], [4, 
            48, 2, 2], [2, 48, 2, 1], [4, 48, 5, 1], [2, 48, 2, 2], [2, 48,
            6, 2]]
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult
            ) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        cnt = 0
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if cnt > 1:
                    if i == n - 1:
                        self.features.append(block_dwc(input_channel,
                            output_channel, s, expand_ratio=t))
                    else:
                        self.features.append(block_dwc(input_channel,
                            output_channel, 1, expand_ratio=t))
                    input_channel = output_channel
                else:
                    if i == n - 1:
                        self.features.append(block_dwc(input_channel,
                            output_channel, s, expand_ratio=t))
                    else:
                        self.features.append(block_dwc(input_channel,
                            output_channel, 1, expand_ratio=t))
                    input_channel = output_channel
            cnt += 1
        self.features.append(gated_conv1x1(input_channel, self.last_channel))
        self.features_sequential = nn.Sequential(*self.features)
        self._initialize_weights()

    def forward(self, x):
        x = self.features_sequential(x).view(-1, 256 * 4)
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


class DWC(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DWC, self).__init__()
        self.batch_norm_in = nn.BatchNorm2d(in_channels)
        self.depthwise = nn.AvgPool2d((7, 6), stride=1, padding=0)
        self.pointwise = nn.Conv2d(in_channels=in_channels, out_channels=
            out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class Max_AvgPool(nn.Module):

    def __init__(self, kernel_size=(3, 3), stride=2, padding=1, dim=128):
        super(Max_AvgPool, self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride,
            padding=padding)
        self.Avgpool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride,
            padding=padding)

    def forward(self, x):
        x = self.Maxpool(x) + self.Avgpool(x)
        return x


class Max_AvgPool(nn.Module):

    def __init__(self, kernel_size=(3, 3), stride=2, padding=1, dim=128):
        super(Max_AvgPool, self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride,
            padding=padding)
        self.Avgpool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride,
            padding=padding)

    def forward(self, x):
        x = self.Maxpool(x) + self.Avgpool(x)
        return x


class gated_conv1x1(nn.Module):

    def __init__(self, inc=128, outc=128):
        super(gated_conv1x1, self).__init__()
        self.inp = int(inc / 2)
        self.oup = int(outc / 2)
        self.conv1x1_1 = nn.Conv2d(self.inp, self.oup, 1, 1, 0, bias=False)
        self.gate_1 = nn.Conv2d(self.inp, self.oup, 1, 1, 0, bias=True)
        self.conv1x1_2 = nn.Conv2d(self.inp, self.oup, 1, 1, 0, bias=False)
        self.gate_2 = nn.Conv2d(self.inp, self.oup, 1, 1, 0, bias=True)

    def forward(self, x):
        x_1 = x[:, :self.inp, :, :]
        x_2 = x[:, self.inp:, :, :]
        a_1 = self.conv1x1_1(x_1)
        g_1 = F.sigmoid(self.gate_1(x_1))
        a_2 = self.conv1x1_2(x_2)
        g_2 = F.sigmoid(self.gate_2(x_2))
        ret = torch.cat((a_1 * g_1, a_2 * g_2), 1)
        return ret


class InvertedResidual_dwc(nn.Module):

    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual_dwc, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup
        self.conv = []
        if expand_ratio == 1:
            self.conv.append(nn.Conv2d(inp, hidden_dim, kernel_size=(3, 3),
                stride=stride, padding=1, groups=hidden_dim))
            self.conv.append(nn.BatchNorm2d(hidden_dim))
            self.conv.append(nn.PReLU())
            self.conv.append(nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False))
            self.conv.append(nn.BatchNorm2d(oup))
        else:
            self.conv.append(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False))
            self.conv.append(nn.BatchNorm2d(hidden_dim))
            self.conv.append(nn.PReLU())
            self.conv.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=
                (3, 3), stride=stride, padding=1, groups=hidden_dim))
            self.conv.append(nn.BatchNorm2d(hidden_dim))
            self.conv.append(nn.PReLU())
            self.conv.append(nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False))
            self.conv.append(nn.BatchNorm2d(oup))
        self.conv = nn.Sequential(*self.conv)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup
        self.conv = []
        if expand_ratio == 1:
            self.conv.append(nn.MaxPool2d(kernel_size=(3, 3), stride=stride,
                padding=1))
            self.conv.append(nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False))
            self.conv.append(nn.BatchNorm2d(oup))
        else:
            self.conv.append(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False))
            self.conv.append(nn.BatchNorm2d(hidden_dim))
            self.conv.append(nn.PReLU())
            self.conv.append(nn.MaxPool2d(kernel_size=(3, 3), stride=stride,
                padding=1))
            self.conv.append(nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False))
            self.conv.append(nn.BatchNorm2d(oup))
        self.conv = nn.Sequential(*self.conv)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class Net(nn.Module):

    def __init__(self, embedding_size=128, input_size=224, width_mult=1.0):
        super(Net, self).__init__()
        block = InvertedResidual
        block_dwc = InvertedResidual_dwc
        input_channel = 64
        last_channel = 256
        interverted_residual_setting = [[1, 64, 1, 1], [2, 64, 2, 1], [4, 
            64, 2, 2], [2, 64, 2, 1], [4, 64, 5, 1], [2, 64, 2, 2], [2, 64,
            6, 2]]
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult
            ) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        cnt = 0
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if cnt > 1:
                    if i == n - 1:
                        self.features.append(block_dwc(input_channel,
                            output_channel, s, expand_ratio=t))
                    else:
                        self.features.append(block_dwc(input_channel,
                            output_channel, 1, expand_ratio=t))
                    input_channel = output_channel
                else:
                    if i == n - 1:
                        self.features.append(block_dwc(input_channel,
                            output_channel, s, expand_ratio=t))
                    else:
                        self.features.append(block_dwc(input_channel,
                            output_channel, 1, expand_ratio=t))
                    input_channel = output_channel
            cnt += 1
        self.features.append(gated_conv1x1(input_channel, self.last_channel))
        self.features_sequential = nn.Sequential(*self.features)
        self._initialize_weights()

    def forward(self, x):
        x = self.features_sequential(x).view(-1, 256 * 4)
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


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_clovaai_EXTD_Pytorch(_paritybench_base):
    pass

    def test_000(self):
        self._check(L2Norm(*[], **{'n_channels': 4, 'scale': 1.0}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(Max_AvgPool(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(InvertedResidual_dwc(*[], **{'inp': 4, 'oup': 4, 'stride': 1, 'expand_ratio': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(InvertedResidual(*[], **{'inp': 4, 'oup': 4, 'stride': 1, 'expand_ratio': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_004(self):
        self._check(Net(*[], **{}), [torch.rand([4, 3, 64, 64])], {})
