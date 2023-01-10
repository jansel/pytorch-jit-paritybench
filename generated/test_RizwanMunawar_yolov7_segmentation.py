import sys
_module = sys.modules[__name__]
del sys
export = _module
models = _module
common = _module
experimental = _module
tf = _module
yolo = _module
predict = _module
sort_count = _module
train = _module
val = _module
train = _module
utils = _module
activations = _module
add_nms = _module
augmentations = _module
autoanchor = _module
autobatch = _module
benchmarks = _module
callbacks = _module
dataloaders = _module
datasets = _module
downloads = _module
general = _module
google_utils = _module
loggers = _module
clearml = _module
clearml_utils = _module
hpo = _module
wandb = _module
log_dataset = _module
sweep = _module
wandb_utils = _module
loss = _module
metrics = _module
plots = _module
segment = _module
dataloaders = _module
general = _module
loss = _module
plots = _module
torch_utils = _module
wandb_logging = _module
wandb_utils = _module
val = _module

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


import warnings


import pandas as pd


import torch


from torch.utils.mobile_optimizer import optimize_for_mobile


import math


from collections import OrderedDict


from collections import namedtuple


from copy import copy


import numpy as np


import torch.nn as nn


import torch.nn.functional as F


from torch.cuda import amp


from copy import deepcopy


import tensorflow as tf


from tensorflow import keras


import torch.backends.cudnn as cudnn


import random


import torch.distributed as dist


from torch.optim import lr_scheduler


from itertools import repeat


import torchvision


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


from torch.utils.data import dataloader


from torch.utils.data import distributed


import logging


from torchvision.utils import save_image


from torchvision.ops import roi_pool


from torchvision.ops import roi_align


from torchvision.ops import ps_roi_pool


from torchvision.ops import ps_roi_align


import inspect


import re


from typing import Optional


from torch.utils.tensorboard import SummaryWriter


import matplotlib.pyplot as plt


import matplotlib


from torch.nn.parallel import DistributedDataParallel as DDP


class ImplicitA(nn.Module):

    def __init__(self, channel):
        super(ImplicitA, self).__init__()
        self.channel = channel
        self.implicit = nn.Parameter(torch.zeros(1, channel, 1, 1))
        nn.init.normal_(self.implicit, std=0.02)

    def forward(self, x):
        return self.implicit + x


class ImplicitM(nn.Module):

    def __init__(self, channel):
        super(ImplicitM, self).__init__()
        self.channel = channel
        self.implicit = nn.Parameter(torch.ones(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=1.0, std=0.02)

    def forward(self, x):
        return self.implicit * x


class MP(nn.Module):

    def __init__(self, k=2):
        super(MP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=k)

    def forward(self, x):
        return self.m(x)


class SP(nn.Module):

    def __init__(self, k=3, s=1):
        super(SP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=s, padding=k // 2)

    def forward(self, x):
        return self.m(x)


class ReOrg(nn.Module):

    def __init__(self):
        super(ReOrg, self).__init__()

    def forward(self, x):
        return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)


def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [(x // 2) for x in k]
    return p


class Conv(nn.Module):

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class DWConv(Conv):

    def __init__(self, c1, c2, k=1, s=1, act=True):
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


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
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).permute(2, 0, 1)
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, w, h)


class Bottleneck(nn.Module):

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class CrossConv(nn.Module):

    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, (1, k), (1, s))
        self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)))


class C3TR(C3):

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class SPP(nn.Module):

    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class C3SPP(C3):

    def __init__(self, c1, c2, k=(5, 9, 13), n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = SPP(c_, c_, k)


class GhostConv(nn.Module):

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, None, g, act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class GhostBottleneck(nn.Module):

    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(GhostConv(c1, c_, 1, 1), DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(), GhostConv(c_, c2, 1, 1, act=False))
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class C3Ghost(C3):

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class SPPF(nn.Module):

    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class SPPCSPC(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(SPPCSPC, self).__init__()
        c_ = int(2 * c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))


class Focus(nn.Module):

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)

    def forward(self, x):
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))


class Contract(nn.Module):

    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()
        s = self.gain
        x = x.view(b, c, h // s, s, w // s, s)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        return x.view(b, c * s * s, h // s, w // s)


class Expand(nn.Module):

    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()
        s = self.gain
        x = x.view(b, s, s, c // s ** 2, h, w)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()
        return x.view(b, c // s ** 2, h * s, w * s)


class Concat(nn.Module):

    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


LOGGER = logging.getLogger('yolov5')


def check_online():
    try:
        socket.create_connection(('1.1.1.1', 443), 5)
        return True
    except OSError:
        return False


def check_version(current='0.0.0', minimum='0.0.0', name='version ', pinned=False, hard=False, verbose=False):
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    result = current == minimum if pinned else current >= minimum
    s = f'{name}{minimum} required by YOLOv5, but {name}{current} is currently installed'
    if hard:
        assert result, s
    if verbose and not result:
        LOGGER.warning(s)
    return result


def check_python(minimum='3.7.0'):
    check_version(platform.python_version(), minimum, name='Python ', hard=True)


def colorstr(*input):
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])
    colors = {'black': '\x1b[30m', 'red': '\x1b[31m', 'green': '\x1b[32m', 'yellow': '\x1b[33m', 'blue': '\x1b[34m', 'magenta': '\x1b[35m', 'cyan': '\x1b[36m', 'white': '\x1b[37m', 'bright_black': '\x1b[90m', 'bright_red': '\x1b[91m', 'bright_green': '\x1b[92m', 'bright_yellow': '\x1b[93m', 'bright_blue': '\x1b[94m', 'bright_magenta': '\x1b[95m', 'bright_cyan': '\x1b[96m', 'bright_white': '\x1b[97m', 'end': '\x1b[0m', 'bold': '\x1b[1m', 'underline': '\x1b[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


def check_suffix(file='yolov5s.pt', suffix=('.pt',), msg=''):
    if file and suffix:
        if isinstance(suffix, str):
            suffix = [suffix]
        for f in (file if isinstance(file, (list, tuple)) else [file]):
            s = Path(f).suffix.lower()
            if len(s):
                assert s in suffix, f'{msg}{f} acceptable suffix is {suffix}'


def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def yaml_load(file='data.yaml'):
    with open(file, errors='ignore') as f:
        return yaml.safe_load(f)


def is_writeable(dir, test=False):
    if not test:
        return os.access(dir, os.W_OK)
    file = Path(dir) / 'tmp.txt'
    try:
        with open(file, 'w'):
            pass
        file.unlink()
        return True
    except OSError:
        return False


def user_config_dir(dir='Ultralytics', env_var='YOLOV5_CONFIG_DIR'):
    env = os.getenv(env_var)
    if env:
        path = Path(env)
    else:
        cfg = {'Windows': 'AppData/Roaming', 'Linux': '.config', 'Darwin': 'Library/Application Support'}
        path = Path.home() / cfg.get(platform.system(), '')
        path = (path if is_writeable(path) else Path('/tmp')) / dir
    path.mkdir(exist_ok=True)
    return path


FONT = 'Arial.ttf'


def check_font(font=FONT, progress=False):
    font = Path(font)
    file = CONFIG_DIR / font.name
    if not font.exists() and not file.exists():
        url = 'https://ultralytics.com/assets/' + font.name
        LOGGER.info(f'Downloading {url} to {file}...')
        torch.hub.download_url_to_file(url, str(file), progress=progress)


def check_pil_font(font=FONT, size=10):
    font = Path(font)
    font = font if font.exists() else CONFIG_DIR / font.name
    try:
        return ImageFont.truetype(str(font) if font.exists() else font.name, size)
    except Exception:
        try:
            check_font(font)
            return ImageFont.truetype(str(font), size)
        except TypeError:
            check_requirements('Pillow>=8.4.0')
        except URLError:
            return ImageFont.load_default()


def is_ascii(s=''):
    s = str(s)
    return len(s.encode().decode('ascii', 'ignore')) == len(s)


class Annotator:

    def __init__(self, im, line_width=None, font_size=None, font='Arial.ttf', pil=False, example='abc'):
        assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to Annotator() input images.'
        non_ascii = not is_ascii(example)
        self.pil = pil or non_ascii
        if self.pil:
            self.im = im if isinstance(im, Image.Image) else Image.fromarray(im)
            self.draw = ImageDraw.Draw(self.im)
            self.font = check_pil_font(font='Arial.Unicode.ttf' if non_ascii else font, size=font_size or max(round(sum(self.im.size) / 2 * 0.035), 12))
        else:
            self.im = im
        self.lw = line_width or max(round(sum(im.shape) / 2 * 0.003), 2)

    def box_label(self, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
        if self.pil or not is_ascii(label):
            self.draw.rectangle(box, width=self.lw, outline=color)
            if label:
                w, h = self.font.getsize(label)
                outside = box[1] - h >= 0
                self.draw.rectangle((box[0], box[1] - h if outside else box[1], box[0] + w + 1, box[1] + 1 if outside else box[1] + h + 1), fill=color)
                self.draw.text((box[0], box[1] - h if outside else box[1]), label, fill=txt_color, font=self.font)
        else:
            p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            cv2.rectangle(self.im, p1, p2, color, thickness=self.lw, lineType=cv2.LINE_AA)
            if label:
                tf = max(self.lw - 1, 1)
                w, h = cv2.getTextSize(label, 0, fontScale=self.lw / 3, thickness=tf)[0]
                outside = p1[1] - h >= 3
                p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)
                cv2.putText(self.im, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, self.lw / 3, txt_color, thickness=tf, lineType=cv2.LINE_AA)

    def draw_trk(self, thickness, centroids):
        [cv2.line(self.im, (int(centroids.centroids[i][0]), int(centroids.centroids[i][1])), (int(centroids.centroids[i + 1][0]), int(centroids.centroids[i + 1][1])), (255, 144, 30), thickness=thickness) for i, _ in enumerate(centroids.centroids) if i < len(centroids.centroids) - 1]

    def draw_id(self, bbox, identities=None, categories=None, names=None, offset=(0, 0)):
        for i, box in enumerate(bbox):
            x1, y1, x2, y2 = [int(i) for i in box]
            x1 += offset[0]
            x2 += offset[0]
            y1 += offset[1]
            y2 += offset[1]
            cat = int(categories[i]) if categories is not None else 0
            id = int(identities[i]) if identities is not None else 0
            data = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
            label = str(id)
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(self.im, (x1, y1 - 20), (x1 + w, y1), (255, 144, 30), -1)
            cv2.putText(self.im, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, [255, 255, 255], 1)
            cv2.circle(self.im, data, 4, (255, 0, 255), -1)

    def rectangle(self, xy, fill=None, outline=None, width=1):
        self.draw.rectangle(xy, fill, outline, width)

    def text(self, xy, text, txt_color=(255, 255, 255), anchor='top'):
        if anchor == 'bottom':
            w, h = self.font.getsize(text)
            xy[1] += 1 - h
        self.draw.text(xy, text, fill=txt_color, font=self.font)

    def fromarray(self, im):
        self.im = im if isinstance(im, Image.Image) else Image.fromarray(im)
        self.draw = ImageDraw.Draw(self.im)

    def result(self):
        return np.asarray(self.im)


class Colors:

    def __init__(self):
        hexs = 'FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB', '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7'
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
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
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        for n in range(2, 9999):
            p = f'{path}{sep}{n}{suffix}'
            if not os.path.exists(p):
                break
        path = Path(p)
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)
    return path


def clip_coords(boxes, shape):
    if isinstance(boxes, torch.Tensor):
        boxes[:, 0].clamp_(0, shape[1])
        boxes[:, 1].clamp_(0, shape[0])
        boxes[:, 2].clamp_(0, shape[1])
        boxes[:, 3].clamp_(0, shape[0])
    else:
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])


def xyxy2xywh(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y


def copy_attr(a, b, include=(), exclude=()):
    for k, v in b.__dict__.items():
        if len(include) and k not in include or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)


def exif_transpose(image):
    """
    Transpose a PIL image accordingly if it has an EXIF Orientation tag.
    Inplace version of https://github.com/python-pillow/Pillow/blob/master/src/PIL/ImageOps.py exif_transpose()

    :param image: The image to transpose.
    :return: An image.
    """
    exif = image.getexif()
    orientation = exif.get(274, 1)
    if orientation > 1:
        method = {(2): Image.FLIP_LEFT_RIGHT, (3): Image.ROTATE_180, (4): Image.FLIP_TOP_BOTTOM, (5): Image.TRANSPOSE, (6): Image.ROTATE_270, (7): Image.TRANSVERSE, (8): Image.ROTATE_90}.get(orientation)
        if method is not None:
            image = image.transpose(method)
            del exif[274]
            image.info['exif'] = exif.tobytes()
    return image


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
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())
    return math.ceil(x / divisor) * divisor


def box_area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def box_iou(box1, box2, eps=1e-07):
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
    (a1, a2), (b1, b2) = box1[:, None].chunk(2, 2), box2.chunk(2, 1)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)
    return inter / (box_area(box1.T)[:, None] + box_area(box2.T) - inter + eps)


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False, labels=(), max_det=300, nm=0):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    bs = prediction.shape[0]
    nc = prediction.shape[2] - nm - 5
    xc = prediction[..., 4] > conf_thres
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
    max_wh = 7680
    max_nms = 30000
    time_limit = 0.5 + 0.05 * bs
    redundant = True
    multi_label &= nc > 1
    merge = False
    t = time.time()
    mi = 5 + nc
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]
            v[:, 4] = 1.0
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0
            x = torch.cat((x, v), 0)
        if not x.shape[0]:
            continue
        x[:, 5:] *= x[:, 4:5]
        box = xywh2xyxy(x[:, :4])
        mask = x[:, mi:]
        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
        else:
            conf, j = x[:, 5:mi].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]
        n = x.shape[0]
        if not n:
            continue
        elif n > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]
        else:
            x = x[x[:, 4].argsort(descending=True)]
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
            LOGGER.warning(f'WARNING: NMS time limit {time_limit:.3f}s exceeded')
            break
    return output


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


class Proto(nn.Module):

    def __init__(self, c1, c_=256, c2=32):
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class Refine(nn.Module):

    def __init__(self, ch, c_=256, c2=32, k=3):
        super().__init__()
        self.refine = nn.ModuleList()
        for c in ch:
            self.refine.append(Conv(c, c_, k))
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        for i, f in enumerate(x):
            if i == 0:
                r = self.refine[i](f)
            else:
                r_p = self.refine[i](f)
                r_p = F.interpolate(r_p, r.size()[2:], mode='nearest')
                r = r + r_p
        return self.cv3(self.cv2(self.upsample(r)))


class Classify(nn.Module):

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        super().__init__()
        c_ = 1280
        self.conv = Conv(c1, c_, k, s, autopad(k, p), g)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = nn.Linear(c_, c2)

    def forward(self, x):
        if isinstance(x, list):
            x = torch.cat(x, 1)
        return self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))


class Sum(nn.Module):

    def __init__(self, n, weight=False):
        super().__init__()
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


class MixConv2d(nn.Module):

    def __init__(self, c1, c2, k=(1, 3), s=1, equal_ch=True):
        super().__init__()
        n = len(k)
        if equal_ch:
            i = torch.linspace(0, n - 1e-06, c2).floor()
            c_ = [(i == g).sum() for g in range(n)]
        else:
            b = [c2] + [0] * n
            a = np.eye(n + 1, n, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            c_ = np.linalg.lstsq(a, b, rcond=None)[0].round()
        self.m = nn.ModuleList([nn.Conv2d(c1, int(c_), k, s, k // 2, groups=math.gcd(c1, int(c_)), bias=False) for k, c_ in zip(k, c_)])
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(torch.cat([m(x) for m in self.m], 1)))


class Ensemble(nn.ModuleList):

    def __init__(self):
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        y = [module(x, augment, profile, visualize)[0] for module in self]
        y = torch.cat(y, 1)
        return y, None


def fuse_conv_and_bn(conv, bn):
    fusedconv = nn.Conv2d(conv.in_channels, conv.out_channels, kernel_size=conv.kernel_size, stride=conv.stride, padding=conv.padding, groups=conv.groups, bias=True).requires_grad_(False)
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
    return fusedconv


def model_info(model, verbose=False, imgsz=640):
    n_p = sum(x.numel() for x in model.parameters())
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)
    if verbose:
        None
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            None
    try:
        p = next(model.parameters())
        stride = max(int(model.stride.max()), 32) if hasattr(model, 'stride') else 32
        im = torch.empty((1, p.shape[1], stride, stride), device=p.device)
        flops = thop.profile(deepcopy(model), inputs=(im,), verbose=False)[0] / 1000000000.0 * 2
        imgsz = imgsz if isinstance(imgsz, list) else [imgsz, imgsz]
        fs = f', {flops * imgsz[0] / stride * imgsz[1] / stride:.1f} GFLOPs'
    except Exception:
        fs = ''
    name = Path(model.yaml_file).stem.replace('yolov5', 'YOLOv5') if hasattr(model, 'yaml_file') else 'Model'
    LOGGER.info(f'{name} summary: {len(list(model.modules()))} layers, {n_p} parameters, {n_g} gradients{fs}')


def time_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


class BaseModel(nn.Module):

    def forward(self, x, profile=False, visualize=False):
        return self._forward_once(x, profile, visualize)

    def _forward_once(self, x, profile=False, visualize=False):
        y, dt = [], []
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [(x if j == -1 else y[j]) for j in m.f]
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)
            y.append(x if m.i in self.save else None)
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x

    def _profile_one_layer(self, m, x, dt):
        c = m == self.model[-1]
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1000000000.0 * 2 if thop else 0
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self):
        LOGGER.info('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)
                delattr(m, 'bn')
                m.forward = m.forward_fuse
        self.info()
        return self

    def info(self, verbose=False, img_size=640):
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        self = super()._apply(fn)
        m = self.model[-1]
        if isinstance(m, (Detect, IDetect, Segment, ISegment, IRSegment)):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self


PREFIX = colorstr('AutoAnchor: ')


def check_anchor_order(m):
    a = m.anchors.prod(-1).mean(-1).view(-1)
    da = a[-1] - a[0]
    ds = m.stride[-1] - m.stride[0]
    if da and da.sign() != ds.sign():
        LOGGER.info(f'{PREFIX}Reversing anchor order')
        m.anchors[:] = m.anchors.flip(0)


def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass
        elif t is nn.BatchNorm2d:
            m.eps = 0.001
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True


def parse_model(d, ch):
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = len(anchors[0]) // 2 if isinstance(anchors, list) else anchors
    no = na * (nc + 5)
    layers, save, c2 = [], [], ch[-1]
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):
        m = eval(m) if isinstance(m, str) else m
        for j, a in enumerate(args):
            with contextlib.suppress(NameError):
                args[j] = eval(a) if isinstance(a, str) else a
        n = n_ = max(round(n * gd), 1) if n > 1 else n
        if m in {Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv, BottleneckCSP, C3, C3TR, C3SPP, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x, SPPCSPC}:
            c1, c2 = ch[f], args[0]
            if c2 != no:
                c2 = make_divisible(c2 * gw, 8)
            args = [c1, c2, *args[1:]]
            if m in {BottleneckCSP, C3, C3TR, C3Ghost, C3x, SPPCSPC}:
                args.insert(2, n)
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m in {Detect, IDetect, Segment, ISegment, IRSegment}:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):
                args[1] = [list(range(args[1] * 2))] * len(f)
            if m in {Segment, ISegment, IRSegment}:
                args[3] = make_divisible(args[3] * gw, 8)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
        t = str(m)[8:-2].replace('__main__.', '')
        np = sum(x.numel() for x in m_.parameters())
        m_.i, m_.f, m_.type, m_.np = i, f, t, np
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


def scale_img(img, ratio=1.0, same_shape=False, gs=32):
    if ratio == 1.0:
        return img
    h, w = img.shape[2:]
    s = int(h * ratio), int(w * ratio)
    img = F.interpolate(img, size=s, mode='bilinear', align_corners=False)
    if not same_shape:
        h, w = (math.ceil(x * ratio / gs) * gs for x in (h, w))
    return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)


class DetectionModel(BaseModel):

    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg
        else:
            self.yaml_file = Path(cfg).name
            with open(cfg, encoding='ascii', errors='ignore') as f:
                self.yaml = yaml.safe_load(f)
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc
        if anchors:
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])
        self.names = [str(i) for i in range(self.yaml['nc'])]
        self.inplace = self.yaml.get('inplace', True)
        m = self.model[-1]
        if isinstance(m, (Detect, IDetect, Segment, ISegment, IRSegment)):
            s = 256
            m.inplace = self.inplace
            forward = lambda x: self.forward(x)[0] if isinstance(m, (Segment, ISegment, IRSegment)) else self.forward(x)
            m.stride = torch.tensor([(s / x.shape[-2]) for x in forward(torch.zeros(1, ch, s, s))])
            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_biases()
        initialize_weights(self)
        self.info()
        LOGGER.info('')

    def forward(self, x, augment=False, profile=False, visualize=False):
        if augment:
            return self._forward_augment(x)
        return self._forward_once(x, profile, visualize)

    def _forward_augment(self, x):
        img_size = x.shape[-2:]
        s = [1, 0.83, 0.67]
        f = [None, 3, None]
        y = []
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)
        return torch.cat(y, 1), None

    def _descale_pred(self, p, flips, scale, img_size):
        if self.inplace:
            p[..., :4] /= scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale
            if flips == 2:
                y = img_size[0] - y
            elif flips == 3:
                x = img_size[1] - x
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        nl = self.model[-1].nl
        g = sum(4 ** x for x in range(nl))
        e = 1
        i = y[0].shape[1] // g * sum(4 ** x for x in range(e))
        y[0] = y[0][:, :-i]
        i = y[-1].shape[1] // g * sum(4 ** (nl - 1 - x) for x in range(e))
        y[-1] = y[-1][:, i:]
        return y

    def _initialize_biases(self, cf=None):
        m = self.model[-1]
        for mi, s in zip(m.m, m.stride):
            b = mi.bias.view(m.na, -1)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)
            b.data[:, 5:5 + m.nc] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


class SegmentationModel(DetectionModel):

    def __init__(self, cfg='yolov5s-seg.yaml', ch=3, nc=None, anchors=None):
        super().__init__(cfg, ch, nc, anchors)


class ClassificationModel(BaseModel):

    def __init__(self, cfg=None, model=None, nc=1000, cutoff=10):
        super().__init__()
        self._from_detection_model(model, nc, cutoff) if model is not None else self._from_yaml(cfg)

    def _from_detection_model(self, model, nc=1000, cutoff=10):
        if isinstance(model, DetectMultiBackend):
            model = model.model
        model.model = model.model[:cutoff]
        m = model.model[-1]
        ch = m.conv.in_channels if hasattr(m, 'conv') else m.cv1.conv.in_channels
        c = Classify(ch, nc)
        c.i, c.f, c.type = m.i, m.f, 'models.common.Classify'
        model.model[-1] = c
        self.model = model.model
        self.stride = model.stride
        self.save = []
        self.nc = nc

    def _from_yaml(self, cfg):
        self.model = None


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
    """ ACON activation (activate or not)
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
    """ ACON activation (activate or not)
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
        super().__init__()
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
        super().__init__()
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
        super().__init__()
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
    (C3SPP,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (C3x,
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
    (DWConv,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DWConvTranspose2d,
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
    (ImplicitA,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ImplicitM,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MP,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
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
    (Proto,
     lambda: ([], {'c1': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (QFocalLoss,
     lambda: ([], {'loss_fcn': MSELoss()}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (ReOrg,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SP,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SPP,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SPPCSPC,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SPPF,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
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
]

class Test_RizwanMunawar_yolov7_segmentation(_paritybench_base):
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

    def test_030(self):
        self._check(*TESTCASES[30])

    def test_031(self):
        self._check(*TESTCASES[31])

    def test_032(self):
        self._check(*TESTCASES[32])

    def test_033(self):
        self._check(*TESTCASES[33])

    def test_034(self):
        self._check(*TESTCASES[34])

    def test_035(self):
        self._check(*TESTCASES[35])

    def test_036(self):
        self._check(*TESTCASES[36])

    def test_037(self):
        self._check(*TESTCASES[37])

