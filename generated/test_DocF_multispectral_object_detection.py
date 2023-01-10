import sys
_module = sys.modules[__name__]
del sys
detect_twostream = _module
global_var = _module
hubconf = _module
models = _module
common = _module
experimental = _module
export = _module
yolo = _module
yolo_test = _module
test = _module
train = _module
utils = _module
activations = _module
autoanchor = _module
aws = _module
resume = _module
datasets = _module
ds_fusion = _module
example_request = _module
restapi = _module
general = _module
google_utils = _module
gradcam = _module
loss = _module
metrics = _module
plots = _module
torch_utils = _module
wandb_logging = _module
log_dataset = _module
wandb_utils = _module

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


import time


import torch


import torch.backends.cudnn as cudnn


import numpy as np


from numpy import random


import math


from copy import copy


import pandas as pd


import torch.nn as nn


from torch.cuda import amp


import torch.nn.functional as F


from torch.nn import init


from torch.nn import Sequential


from torch.utils.mobile_optimizer import optimize_for_mobile


import logging


from copy import deepcopy


import random


import torch.distributed as dist


import torch.optim as optim


import torch.optim.lr_scheduler as lr_scheduler


import torch.utils.data


from torch.nn.parallel import DistributedDataParallel as DDP


from torch.utils.tensorboard import SummaryWriter


from scipy.cluster.vq import kmeans


from itertools import repeat


from torch.utils.data import Dataset


import re


import torchvision


from torch.autograd import Variable


from torch.autograd import Function


from torchvision import models


from torchvision import utils


import warnings


import matplotlib.pyplot as plt


import matplotlib


from scipy.signal import butter


from scipy.signal import filtfilt


def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [(x // 2) for x in k]
    return p


class Conv(nn.Module):

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class TransformerLayer(nn.Module):

    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):

    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)
        self.tr = nn.Sequential(*[TransformerLayer(c2, num_heads) for _ in range(num_layers)])
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2)
        p = p.unsqueeze(0)
        p = p.transpose(0, 3)
        p = p.squeeze(3)
        e = self.linear(p)
        x = p + e
        x = self.tr(x)
        x = x.unsqueeze(3)
        x = x.transpose(0, 3)
        x = x.reshape(b, self.c2, w, h)
        return x


class Bottleneck(nn.Module):

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class C3(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super(C3, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class C3TR(C3):

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class SPP(nn.Module):

    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class Focus(nn.Module):

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)

    def forward(self, x):
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))


class Contract(nn.Module):

    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()
        s = self.gain
        x = x.view(N, C, H // s, s, W // s, s)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        return x.view(N, C * s * s, H // s, W // s)


class Expand(nn.Module):

    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()
        s = self.gain
        x = x.view(N, s, s, C // s ** 2, H, W)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()
        return x.view(N, C // s ** 2, H * s, W * s)


class Concat(nn.Module):

    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class Add(nn.Module):

    def __init__(self, arg):
        super(Add, self).__init__()
        self.arg = arg

    def forward(self, x):
        return torch.add(x[0], x[1])


class Add2(nn.Module):

    def __init__(self, c1, index):
        super().__init__()
        self.index = index

    def forward(self, x):
        if self.index == 0:
            return torch.add(x[0], x[1][0])
        elif self.index == 1:
            return torch.add(x[0], x[1][1])


def box_iou(box1, box2):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])
    area1 = box_area(box1.T)
    area2 = box_area(box2.T)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)


def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False, labels=()):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    nc = prediction.shape[2] - 5
    xc = prediction[..., 4] > conf_thres
    min_wh, max_wh = 2, 4096
    max_det = 300
    max_nms = 30000
    time_limit = 10.0
    redundant = True
    multi_label &= nc > 1
    merge = False
    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]
            v[:, 4] = 1.0
            v[range(len(l)), l[:, 0].long() + 5] = 1.0
            x = torch.cat((x, v), 0)
        if not x.shape[0]:
            continue
        x[:, 5:] *= x[:, 4:5]
        box = xywh2xyxy(x[:, :4])
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]
        n = x.shape[0]
        if not n:
            continue
        elif n > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]
        c = x[:, 5:6] * (0 if agnostic else max_wh)
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:
            i = i[:max_det]
        if merge and 1 < n < 3000.0:
            iou = box_iou(boxes[i], boxes) > iou_thres
            weights = iou * scores[None]
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)
            if redundant:
                i = i[iou.sum(1) > 1]
        output[xi] = x[i]
        if time.time() - t > time_limit:
            None
            break
    return output


class NMS(nn.Module):
    conf = 0.25
    iou = 0.45
    classes = None

    def __init__(self):
        super(NMS, self).__init__()

    def forward(self, x):
        return non_max_suppression(x[0], conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)


class Colors:

    def __init__(self):
        self.palette = [self.hex2rgb(c) for c in matplotlib.colors.TABLEAU_COLORS.values()]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()


def increment_path(path, exist_ok=False, sep='', mkdir=False):
    path = Path(path)
    if path.exists() and not exist_ok:
        suffix = path.suffix
        path = path.with_suffix('')
        dirs = glob.glob(f'{path}{sep}*')
        matches = [re.search(f'%s{sep}(\\d+)' % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        path = Path(f'{path}{sep}{n}{suffix}')
    dir = path if path.suffix == '' else path.parent
    if not dir.exists() and mkdir:
        dir.mkdir(parents=True, exist_ok=True)
    return path


def plot_one_box(x, im, color=None, label=None, line_thickness=3):
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)
        cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def clip_coords(boxes, img_shape):
    boxes[:, 0].clamp_(0, img_shape[1])
    boxes[:, 1].clamp_(0, img_shape[0])
    boxes[:, 2].clamp_(0, img_shape[1])
    boxes[:, 3].clamp_(0, img_shape[0])


def xyxy2xywh(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y


def save_one_box(xyxy, im, file='image.jpg', gain=1.02, pad=10, square=False, BGR=False):
    xyxy = torch.tensor(xyxy).view(-1, 4)
    b = xyxy2xywh(xyxy)
    if square:
        b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)
    b[:, 2:] = b[:, 2:] * gain + pad
    xyxy = xywh2xyxy(b).long()
    clip_coords(xyxy, im.shape)
    crop = im[int(xyxy[0, 1]):int(xyxy[0, 3]), int(xyxy[0, 0]):int(xyxy[0, 2])]
    cv2.imwrite(str(increment_path(file, mkdir=True).with_suffix('.jpg')), crop if BGR else crop[..., ::-1])


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = new_shape, new_shape
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = new_shape[1], new_shape[0]
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)


def make_divisible(x, divisor):
    return math.ceil(x / divisor) * divisor


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def time_synchronized():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


class autoShape(nn.Module):
    conf = 0.25
    iou = 0.45
    classes = None

    def __init__(self, model):
        super(autoShape, self).__init__()
        self.model = model.eval()

    def autoshape(self):
        None
        return self

    @torch.no_grad()
    def forward(self, imgs, size=640, augment=False, profile=False):
        t = [time_synchronized()]
        p = next(self.model.parameters())
        if isinstance(imgs, torch.Tensor):
            with amp.autocast(enabled=p.device.type != 'cpu'):
                return self.model(imgs.type_as(p), augment, profile)
        n, imgs = (len(imgs), imgs) if isinstance(imgs, list) else (1, [imgs])
        shape0, shape1, files = [], [], []
        for i, im in enumerate(imgs):
            f = f'image{i}'
            if isinstance(im, str):
                im, f = np.asarray(Image.open(requests.get(im, stream=True).raw if im.startswith('http') else im)), im
            elif isinstance(im, Image.Image):
                im, f = np.asarray(im), getattr(im, 'filename', f) or f
            files.append(Path(f).with_suffix('.jpg').name)
            if im.shape[0] < 5:
                im = im.transpose((1, 2, 0))
            im = im[:, :, :3] if im.ndim == 3 else np.tile(im[:, :, None], 3)
            s = im.shape[:2]
            shape0.append(s)
            g = size / max(s)
            shape1.append([(y * g) for y in s])
            imgs[i] = im if im.data.contiguous else np.ascontiguousarray(im)
        shape1 = [make_divisible(x, int(self.stride.max())) for x in np.stack(shape1, 0).max(0)]
        x = [letterbox(im, new_shape=shape1, auto=False)[0] for im in imgs]
        x = np.stack(x, 0) if n > 1 else x[0][None]
        x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))
        x = torch.from_numpy(x).type_as(p) / 255.0
        t.append(time_synchronized())
        with amp.autocast(enabled=p.device.type != 'cpu'):
            y = self.model(x, augment, profile)[0]
            t.append(time_synchronized())
            y = non_max_suppression(y, conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)
            for i in range(n):
                scale_coords(shape1, y[i][:, :4], shape0[i])
            t.append(time_synchronized())
            return Detections(imgs, y, files, t, self.names, x.shape)


class Classify(nn.Module):

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        super(Classify, self).__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g)
        self.flat = nn.Flatten()

    def forward(self, x):
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)
        return self.flat(self.conv(z))


class SelfAttention(nn.Module):
    """
     Multi-head masked self-attention layer
    """

    def __init__(self, d_model, d_k, d_v, h, attn_pdrop=0.1, resid_pdrop=0.1):
        """
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        """
        super(SelfAttention, self).__init__()
        assert d_k % h == 0
        self.d_model = d_model
        self.d_k = d_model // h
        self.d_v = d_model // h
        self.h = h
        self.que_proj = nn.Linear(d_model, h * self.d_k)
        self.key_proj = nn.Linear(d_model, h * self.d_k)
        self.val_proj = nn.Linear(d_model, h * self.d_v)
        self.out_proj = nn.Linear(h * self.d_v, d_model)
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, attention_mask=None, attention_weights=None):
        """
        Computes Self-Attention
        Args:
            x (tensor): input (token) dim:(b_s, nx, c),
                b_s means batch size
                nx means length, for CNN, equals H*W, i.e. the length of feature maps
                c means channel, i.e. the channel of feature maps
            attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
            attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        Return:
            output (tensor): dim:(b_s, nx, c)
        """
        b_s, nq = x.shape[:2]
        nk = x.shape[1]
        q = self.que_proj(x).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)
        k = self.key_proj(x).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)
        v = self.val_proj(x).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)
        att = torch.matmul(q, k) / np.sqrt(self.d_k)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att = self.attn_drop(att)
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)
        out = self.resid_drop(self.out_proj(out))
        return out


class myTransformerBlock(nn.Module):
    """ Transformer block """

    def __init__(self, d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop):
        """
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        :param block_exp: Expansion factor for MLP (feed foreword network)

        """
        super().__init__()
        self.ln_input = nn.LayerNorm(d_model)
        self.ln_output = nn.LayerNorm(d_model)
        self.sa = SelfAttention(d_model, d_k, d_v, h, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(nn.Linear(d_model, block_exp * d_model), nn.GELU(), nn.Linear(block_exp * d_model, d_model), nn.Dropout(resid_pdrop))

    def forward(self, x):
        bs, nx, c = x.size()
        x = x + self.sa(self.ln_input(x))
        x = x + self.mlp(self.ln_output(x))
        return x


class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, d_model, h=8, block_exp=4, n_layer=8, vert_anchors=8, horz_anchors=8, embd_pdrop=0.1, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()
        self.n_embd = d_model
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors
        d_k = d_model
        d_v = d_model
        self.pos_emb = nn.Parameter(torch.zeros(1, 2 * vert_anchors * horz_anchors, self.n_embd))
        self.trans_blocks = nn.Sequential(*[myTransformerBlock(d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop) for layer in range(n_layer)])
        self.ln_f = nn.LayerNorm(self.n_embd)
        self.drop = nn.Dropout(embd_pdrop)
        self.avgpool = nn.AdaptiveAvgPool2d((self.vert_anchors, self.horz_anchors))
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        """
        Args:
            x (tuple?)

        """
        rgb_fea = x[0]
        ir_fea = x[1]
        assert rgb_fea.shape[0] == ir_fea.shape[0]
        bs, c, h, w = rgb_fea.shape
        rgb_fea = self.avgpool(rgb_fea)
        ir_fea = self.avgpool(ir_fea)
        rgb_fea_flat = rgb_fea.view(bs, c, -1)
        ir_fea_flat = ir_fea.view(bs, c, -1)
        token_embeddings = torch.cat([rgb_fea_flat, ir_fea_flat], dim=2)
        token_embeddings = token_embeddings.permute(0, 2, 1).contiguous()
        x = self.drop(self.pos_emb + token_embeddings)
        x = self.trans_blocks(x)
        x = self.ln_f(x)
        x = x.view(bs, 2, self.vert_anchors, self.horz_anchors, self.n_embd)
        x = x.permute(0, 1, 4, 2, 3)
        rgb_fea_out = x[:, 0, :, :, :].contiguous().view(bs, self.n_embd, self.vert_anchors, self.horz_anchors)
        ir_fea_out = x[:, 1, :, :, :].contiguous().view(bs, self.n_embd, self.vert_anchors, self.horz_anchors)
        rgb_fea_out = F.interpolate(rgb_fea_out, size=[h, w], mode='bilinear')
        ir_fea_out = F.interpolate(ir_fea_out, size=[h, w], mode='bilinear')
        return rgb_fea_out, ir_fea_out


class CrossConv(nn.Module):

    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
        super(CrossConv, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, (1, k), (1, s))
        self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class Sum(nn.Module):

    def __init__(self, n, weight=False):
        super(Sum, self).__init__()
        self.weight = weight
        self.iter = range(n - 1)
        if weight:
            self.w = nn.Parameter(-torch.arange(1.0, n) / 2, requires_grad=True)

    def forward(self, x):
        y = x[0]
        if self.weight:
            w = torch.sigmoid(self.w) * 2
            for i in self.iter:
                y = y + x[i + 1] * w[i]
        else:
            for i in self.iter:
                y = y + x[i + 1]
        return y


class GhostConv(nn.Module):

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super(GhostConv, self).__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, None, g, act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat([y, self.cv2(y)], 1)


def DWConv(c1, c2, k=1, s=1, act=True):
    return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class GhostBottleneck(nn.Module):

    def __init__(self, c1, c2, k=3, s=1):
        super(GhostBottleneck, self).__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(GhostConv(c1, c_, 1, 1), DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(), GhostConv(c_, c2, 1, 1, act=False))
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class MixConv2d(nn.Module):

    def __init__(self, c1, c2, k=(1, 3), s=1, equal_ch=True):
        super(MixConv2d, self).__init__()
        groups = len(k)
        if equal_ch:
            i = torch.linspace(0, groups - 1e-06, c2).floor()
            c_ = [(i == g).sum() for g in range(groups)]
        else:
            b = [c2] + [0] * groups
            a = np.eye(groups + 1, groups, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            c_ = np.linalg.lstsq(a, b, rcond=None)[0].round()
        self.m = nn.ModuleList([nn.Conv2d(c1, int(c_[g]), k[g], s, k[g] // 2, bias=False) for g in range(groups)])
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return x + self.act(self.bn(torch.cat([m(x) for m in self.m], 1)))


class Ensemble(nn.ModuleList):

    def __init__(self):
        super(Ensemble, self).__init__()

    def forward(self, x, augment=False):
        y = []
        for module in self:
            y.append(module(x, augment)[0])
        y = torch.cat(y, 1)
        return y, None


class Detect(nn.Module):
    stride = None
    export = False

    def __init__(self, nc=80, anchors=(), ch=()):
        super(Detect, self).__init__()
        self.nc = nc
        self.no = nc + 5
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.grid = [torch.zeros(1)] * self.nl
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)

    def forward(self, x):
        z = []
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            if not self.training:
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny)
                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2.0 - 0.5 + self.grid[i]) * self.stride[i]
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
                z.append(y.view(bs, -1, self.no))
        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


def check_anchor_order(m):
    a = m.anchor_grid.prod(-1).view(-1)
    da = a[-1] - a[0]
    ds = m.stride[-1] - m.stride[0]
    if da.sign() != ds.sign():
        None
        m.anchors[:] = m.anchors.flip(0)
        m.anchor_grid[:] = m.anchor_grid.flip(0)


def copy_attr(a, b, include=(), exclude=()):
    for k, v in b.__dict__.items():
        if len(include) and k not in include or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)


def fuse_conv_and_bn(conv, bn):
    fusedconv = nn.Conv2d(conv.in_channels, conv.out_channels, kernel_size=conv.kernel_size, stride=conv.stride, padding=conv.padding, groups=conv.groups, bias=True).requires_grad_(False)
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
    return fusedconv


def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass
        elif t is nn.BatchNorm2d:
            m.eps = 0.001
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True


logger = logging.getLogger(__name__)


def model_info(model, verbose=False, img_size=640):
    n_p = sum(x.numel() for x in model.parameters())
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)
    if verbose:
        None
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            None
    try:
        stride = max(int(model.stride.max()), 32) if hasattr(model, 'stride') else 32
        img = torch.zeros((1, model.yaml.get('ch', 3), stride, stride), device=next(model.parameters()).device)
        flops = profile(deepcopy(model), inputs=(img,), verbose=False)[0] / 1000000000.0 * 2
        img_size = img_size if isinstance(img_size, list) else [img_size, img_size]
        fs = ', %.1f GFLOPS' % (flops * img_size[0] / stride * img_size[1] / stride)
    except (ImportError, Exception):
        fs = ''
    logger.info(f'Model Summary: {len(list(model.modules()))} layers, {n_p} parameters, {n_g} gradients{fs}')


def parse_model(d, ch):
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = len(anchors[0]) // 2 if isinstance(anchors, list) else anchors
    no = na * (nc + 5)
    layers, save, c2 = [], [], ch[-1]
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):
        m = eval(m) if isinstance(m, str) else m
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a
            except:
                pass
        n = max(round(n * gd), 1) if n > 1 else n
        if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, DWConv, MixConv2d, Focus, CrossConv, BottleneckCSP, C3, C3TR]:
            if m is Focus:
                c1, c2 = 3, args[0]
                if c2 != no:
                    c2 = make_divisible(c2 * gw, 8)
                args = [c1, c2, *args[1:]]
            else:
                c1, c2 = ch[f], args[0]
                if c2 != no:
                    c2 = make_divisible(c2 * gw, 8)
                args = [c1, c2, *args[1:]]
                if m in [BottleneckCSP, C3, C3TR]:
                    args.insert(2, n)
                    n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        elif m is Add:
            c2 = ch[f[0]]
            args = [c2]
        elif m is Add2:
            c2 = ch[f[0]]
            args = [c2, args[1]]
        elif m is GPT:
            c2 = ch[f[0]]
            args = [c2]
        elif m is Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]
        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)
        t = str(m)[8:-2].replace('__main__.', '')
        np = sum([x.numel() for x in m_.parameters()])
        m_.i, m_.f, m_.type, m_.np = i, f, t, np
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


def scale_img(img, ratio=1.0, same_shape=False, gs=32):
    if ratio == 1.0:
        return img
    else:
        h, w = img.shape[2:]
        s = int(h * ratio), int(w * ratio)
        img = F.interpolate(img, size=s, mode='bilinear', align_corners=False)
        if not same_shape:
            h, w = [(math.ceil(x * ratio / gs) * gs) for x in (h, w)]
        return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)


class Model(nn.Module):

    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):
        super(Model, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg
        else:
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.safe_load(f)
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)
        if nc and nc != self.yaml['nc']:
            logger.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc
        if anchors:
            logger.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])
        self.names = [str(i) for i in range(self.yaml['nc'])]
        m = self.model[-1]
        if isinstance(m, Detect):
            s = 256
            m.stride = torch.Tensor([8.0, 16.0, 32.0])
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()
        initialize_weights(self)
        self.info()
        logger.info('')

    def forward(self, x, x2, augment=False, profile=False):
        if augment:
            img_size = x.shape[-2:]
            s = [1, 0.83, 0.67]
            f = [None, 3, None]
            y = []
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
                yi = self.forward_once(xi)[0]
                yi[..., :4] /= si
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]
                y.append(yi)
            return torch.cat(y, 1), None
        else:
            return self.forward_once(x, x2, profile)

    def forward_once(self, x, x2, profile=False):
        """

        :param x:          RGB Inputs
        :param x2:         IR  Inputs
        :param profile:
        :return:
        """
        y, dt = [], []
        i = 0
        for m in self.model:
            if m.f != -1:
                if m.f != -4:
                    x = y[m.f] if isinstance(m.f, int) else [(x if j == -1 else y[j]) for j in m.f]
            if profile:
                o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1000000000.0 * 2 if thop else 0
                t = time_synchronized()
                for _ in range(10):
                    _ = m(x)
                dt.append((time_synchronized() - t) * 100)
                if m == self.model[0]:
                    logger.info(f"{'time (ms)':>10s} {'GFLOPS':>10s} {'params':>10s}  {'module'}")
                logger.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
            if m.f == -4:
                x = m(x2)
            else:
                x = m(x)
            y.append(x if m.i in self.save else None)
            i += 1
        if profile:
            logger.info('%.1fms total' % sum(dt))
        return x

    def _initialize_biases(self, cf=None):
        m = self.model[-1]
        for mi, s in zip(m.m, m.stride):
            b = mi.bias.view(m.na, -1)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]
        for mi in m.m:
            b = mi.bias.detach().view(m.na, -1).T
            logger.info(('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    def fuse(self):
        logger.info('Fusing layers... ')
        for m in self.model.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)
                delattr(m, 'bn')
                m.forward = m.fuseforward
        self.info()
        return self

    def nms(self, mode=True):
        present = type(self.model[-1]) is NMS
        if mode and not present:
            logger.info('Adding NMS... ')
            m = NMS()
            m.f = -1
            m.i = self.model[-1].i + 1
            self.model.add_module(name='%s' % m.i, module=m)
            self.eval()
        elif not mode and present:
            logger.info('Removing NMS... ')
            self.model = self.model[:-1]
        return self

    def autoshape(self):
        logger.info('Adding autoShape... ')
        m = autoShape(self)
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())
        return m

    def info(self, verbose=False, img_size=640):
        model_info(self, verbose, img_size)


class SiLU(nn.Module):

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


class Hardswish(nn.Module):

    @staticmethod
    def forward(x):
        return x * F.hardtanh(x + 3, 0.0, 6.0) / 6.0


class Mish(nn.Module):

    @staticmethod
    def forward(x):
        return x * F.softplus(x).tanh()


class MemoryEfficientMish(nn.Module):


    class F(torch.autograd.Function):

        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x.mul(torch.tanh(F.softplus(x)))

        @staticmethod
        def backward(ctx, grad_output):
            x = ctx.saved_tensors[0]
            sx = torch.sigmoid(x)
            fx = F.softplus(x).tanh()
            return grad_output * (fx + x * sx * (1 - fx * fx))

    def forward(self, x):
        return self.F.apply(x)


class FReLU(nn.Module):

    def __init__(self, c1, k=3):
        super().__init__()
        self.conv = nn.Conv2d(c1, c1, k, 1, 1, groups=c1, bias=False)
        self.bn = nn.BatchNorm2d(c1)

    def forward(self, x):
        return torch.max(x, self.bn(self.conv(x)))


class AconC(nn.Module):
    """ ACON activation (activate or not).
    AconC: (p1*x-p2*x) * sigmoid(beta*(p1*x-p2*x)) + p2*x, beta is a learnable parameter
    according to "Activate or Not: Learning Customized Activation" <https://arxiv.org/pdf/2009.04759.pdf>.
    """

    def __init__(self, c1):
        super().__init__()
        self.p1 = nn.Parameter(torch.randn(1, c1, 1, 1))
        self.p2 = nn.Parameter(torch.randn(1, c1, 1, 1))
        self.beta = nn.Parameter(torch.ones(1, c1, 1, 1))

    def forward(self, x):
        dpx = (self.p1 - self.p2) * x
        return dpx * torch.sigmoid(self.beta * dpx) + self.p2 * x


class MetaAconC(nn.Module):
    """ ACON activation (activate or not).
    MetaAconC: (p1*x-p2*x) * sigmoid(beta*(p1*x-p2*x)) + p2*x, beta is generated by a small network
    according to "Activate or Not: Learning Customized Activation" <https://arxiv.org/pdf/2009.04759.pdf>.
    """

    def __init__(self, c1, k=1, s=1, r=16):
        super().__init__()
        c2 = max(r, c1 // r)
        self.p1 = nn.Parameter(torch.randn(1, c1, 1, 1))
        self.p2 = nn.Parameter(torch.randn(1, c1, 1, 1))
        self.fc1 = nn.Conv2d(c1, c2, k, s, bias=True)
        self.fc2 = nn.Conv2d(c2, c1, k, s, bias=True)

    def forward(self, x):
        y = x.mean(dim=2, keepdims=True).mean(dim=3, keepdims=True)
        beta = torch.sigmoid(self.fc2(self.fc1(y)))
        dpx = (self.p1 - self.p2) * x
        return dpx * torch.sigmoid(beta * dpx) + self.p2 * x


class BCEBlurWithLogitsLoss(nn.Module):

    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)
        dx = pred - true
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 0.0001))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):

    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred_prob = torch.sigmoid(pred)
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class QFocalLoss(nn.Module):

    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(QFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred_prob = torch.sigmoid(pred)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AconC,
     lambda: ([], {'c1': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Add,
     lambda: ([], {'arg': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Add2,
     lambda: ([], {'c1': 4, 'index': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BCEBlurWithLogitsLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Bottleneck,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BottleneckCSP,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (C3,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Classify,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Contract,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Conv,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CrossConv,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Expand,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (FReLU,
     lambda: ([], {'c1': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FocalLoss,
     lambda: ([], {'loss_fcn': MSELoss()}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Focus,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GhostBottleneck,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GhostConv,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Hardswish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MemoryEfficientMish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MetaAconC,
     lambda: ([], {'c1': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Mish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MixConv2d,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (QFocalLoss,
     lambda: ([], {'loss_fcn': MSELoss()}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (SPP,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SelfAttention,
     lambda: ([], {'d_model': 4, 'd_k': 4, 'd_v': 4, 'h': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (SiLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Sum,
     lambda: ([], {'n': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (TransformerBlock,
     lambda: ([], {'c1': 4, 'c2': 4, 'num_heads': 4, 'num_layers': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TransformerLayer,
     lambda: ([], {'c': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (myTransformerBlock,
     lambda: ([], {'d_model': 4, 'd_k': 4, 'd_v': 4, 'h': 4, 'block_exp': 4, 'attn_pdrop': 0.5, 'resid_pdrop': 0.5}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
]

class Test_DocF_multispectral_object_detection(_paritybench_base):
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

    def test_007(self):
        self._check(*TESTCASES[7])

    def test_008(self):
        self._check(*TESTCASES[8])

    def test_009(self):
        self._check(*TESTCASES[9])

    def test_010(self):
        self._check(*TESTCASES[10])

    def test_011(self):
        self._check(*TESTCASES[11])

    def test_012(self):
        self._check(*TESTCASES[12])

    def test_013(self):
        self._check(*TESTCASES[13])

    def test_014(self):
        self._check(*TESTCASES[14])

    def test_015(self):
        self._check(*TESTCASES[15])

    def test_016(self):
        self._check(*TESTCASES[16])

    def test_017(self):
        self._check(*TESTCASES[17])

    def test_018(self):
        self._check(*TESTCASES[18])

    def test_019(self):
        self._check(*TESTCASES[19])

    def test_020(self):
        self._check(*TESTCASES[20])

    def test_021(self):
        self._check(*TESTCASES[21])

    def test_022(self):
        self._check(*TESTCASES[22])

    def test_023(self):
        self._check(*TESTCASES[23])

    def test_024(self):
        self._check(*TESTCASES[24])

    def test_025(self):
        self._check(*TESTCASES[25])

    def test_026(self):
        self._check(*TESTCASES[26])

    def test_027(self):
        self._check(*TESTCASES[27])

    def test_028(self):
        self._check(*TESTCASES[28])

    def test_029(self):
        self._check(*TESTCASES[29])

