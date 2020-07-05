import sys
_module = sys.modules[__name__]
del sys
alfred = _module
dl = _module
common = _module
coco_dataset = _module
meta = _module
concatenated_dataset = _module
dataset_mixin = _module
getter_dataset = _module
sliceable_dataset = _module
inference = _module
image_inference = _module
torch = _module
utils = _module
env = _module
metrics = _module
model_summary = _module
nn = _module
functional = _module
modules = _module
common = _module
normalization = _module
weights_init = _module
ops = _module
array_ops = _module
checkpoint = _module
tools = _module
train = _module
checkpoint = _module
fastai_optim = _module
learning_schedules = _module
learning_schedules_fastai = _module
optim = _module
fusion = _module
geometry = _module
kitti_fusion = _module
nuscenes_fusion = _module
cabinet = _module
count_file = _module
license = _module
split_txt = _module
stack_imgs = _module
data = _module
coco2voc = _module
convert_csv2voc = _module
convert_cvat2voc = _module
convert_labelone2voc = _module
eval_coco = _module
eval_voc = _module
extract_voc = _module
gather_voclabels = _module
labelone_view = _module
split = _module
txt2voc = _module
view_coco = _module
view_txt = _module
view_voc = _module
voc2coco = _module
voc2yolo = _module
scrap = _module
image_scraper = _module
scraper_images = _module
text = _module
vision = _module
face_extractor = _module
to_video = _module
video_extractor = _module
video_reducer = _module
vis_kit = _module
labelmap_pb2 = _module
tests = _module
cv_box_fancy = _module
cv_wrapper = _module
file_io = _module
image_convertor = _module
log = _module
mana = _module
timer = _module
vis = _module
image = _module
det = _module
get_dataset_color_map = _module
get_dataset_label_map = _module
mask = _module
process = _module
seg = _module
pointcloud = _module
pointcloud_vis = _module
alfred_show_box_gt = _module
draw_3d_box_on_image = _module
draw_3d_pointcloud = _module
pykitti_test = _module
vis_coco = _module
setup = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, string, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
import numpy as np
from torch import Tensor
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import numpy as np


import torch


import torch.nn.functional as F


from torch import nn


import torch.nn as nn


from torch.autograd import Variable


from collections import OrderedDict


from torch.nn import functional as F


import time


import warnings


import torchvision


from torch.utils import model_zoo


import logging


from collections import Iterable


from collections import defaultdict


from copy import deepcopy


from itertools import chain


from torch._utils import _unflatten_dense_tensors


from torch.nn.utils import parameters_to_vector


class Scalar(nn.Module):

    def __init__(self):
        super().__init__()
        self.register_buffer('total', torch.FloatTensor([0.0]))
        self.register_buffer('count', torch.FloatTensor([0.0]))

    def forward(self, scalar):
        if not scalar.eq(0.0):
            self.count += 1
            self.total += scalar.data.float()
        return self.value.cpu()

    @property
    def value(self):
        return self.total / self.count

    def clear(self):
        self.total.zero_()
        self.count.zero_()


class Accuracy(nn.Module):

    def __init__(self, dim=1, ignore_idx=-1, threshold=0.5, encode_background_as_zeros=True):
        super().__init__()
        self.register_buffer('total', torch.FloatTensor([0.0]))
        self.register_buffer('count', torch.FloatTensor([0.0]))
        self._ignore_idx = ignore_idx
        self._dim = dim
        self._threshold = threshold
        self._encode_background_as_zeros = encode_background_as_zeros

    def forward(self, labels, preds, weights=None):
        if self._encode_background_as_zeros:
            scores = torch.sigmoid(preds)
            labels_pred = torch.max(preds, dim=self._dim)[1] + 1
            pred_labels = torch.where((scores > self._threshold).any(self._dim), labels_pred, torch.tensor(0).type_as(labels_pred))
        else:
            pred_labels = torch.max(preds, dim=self._dim)[1]
        N, *Ds = labels.shape
        labels = labels.view(N, int(np.prod(Ds)))
        pred_labels = pred_labels.view(N, int(np.prod(Ds)))
        if weights is None:
            weights = (labels != self._ignore_idx).float()
        else:
            weights = weights.float()
        num_examples = torch.sum(weights)
        num_examples = torch.clamp(num_examples, min=1.0).float()
        total = torch.sum((pred_labels == labels.long()).float())
        self.count += num_examples
        self.total += total
        return self.value.cpu()

    @property
    def value(self):
        return self.total / self.count

    def clear(self):
        self.total.zero_()
        self.count.zero_()


class Precision(nn.Module):

    def __init__(self, dim=1, ignore_idx=-1, threshold=0.5):
        super().__init__()
        self.register_buffer('total', torch.FloatTensor([0.0]))
        self.register_buffer('count', torch.FloatTensor([0.0]))
        self._ignore_idx = ignore_idx
        self._dim = dim
        self._threshold = threshold

    def forward(self, labels, preds, weights=None):
        if preds.shape[self._dim] == 1:
            pred_labels = (torch.sigmoid(preds) > self._threshold).long().squeeze(self._dim)
        else:
            assert preds.shape[self._dim] == 2, 'precision only support 2 class'
            pred_labels = torch.max(preds, dim=self._dim)[1]
        N, *Ds = labels.shape
        labels = labels.view(N, int(np.prod(Ds)))
        pred_labels = pred_labels.view(N, int(np.prod(Ds)))
        if weights is None:
            weights = (labels != self._ignore_idx).float()
        else:
            weights = weights.float()
        pred_trues = pred_labels > 0
        pred_falses = pred_labels == 0
        trues = labels > 0
        falses = labels == 0
        true_positives = (weights * (trues & pred_trues).float()).sum()
        true_negatives = (weights * (falses & pred_falses).float()).sum()
        false_positives = (weights * (falses & pred_trues).float()).sum()
        false_negatives = (weights * (trues & pred_falses).float()).sum()
        count = true_positives + false_positives
        if count > 0:
            self.count += count
            self.total += true_positives
        return self.value.cpu()

    @property
    def value(self):
        return self.total / self.count

    def clear(self):
        self.total.zero_()
        self.count.zero_()


class Recall(nn.Module):

    def __init__(self, dim=1, ignore_idx=-1, threshold=0.5):
        super().__init__()
        self.register_buffer('total', torch.FloatTensor([0.0]))
        self.register_buffer('count', torch.FloatTensor([0.0]))
        self._ignore_idx = ignore_idx
        self._dim = dim
        self._threshold = threshold

    def forward(self, labels, preds, weights=None):
        if preds.shape[self._dim] == 1:
            pred_labels = (torch.sigmoid(preds) > self._threshold).long().squeeze(self._dim)
        else:
            assert preds.shape[self._dim] == 2, 'precision only support 2 class'
            pred_labels = torch.max(preds, dim=self._dim)[1]
        N, *Ds = labels.shape
        labels = labels.view(N, int(np.prod(Ds)))
        pred_labels = pred_labels.view(N, int(np.prod(Ds)))
        if weights is None:
            weights = (labels != self._ignore_idx).float()
        else:
            weights = weights.float()
        pred_trues = pred_labels == 1
        pred_falses = pred_labels == 0
        trues = labels == 1
        falses = labels == 0
        true_positives = (weights * (trues & pred_trues).float()).sum()
        true_negatives = (weights * (falses & pred_falses).float()).sum()
        false_positives = (weights * (falses & pred_trues).float()).sum()
        false_negatives = (weights * (trues & pred_falses).float()).sum()
        count = true_positives + false_negatives
        if count > 0:
            self.count += count
            self.total += true_positives
        return self.value.cpu()

    @property
    def value(self):
        return self.total / self.count

    def clear(self):
        self.total.zero_()
        self.count.zero_()


def _calc_binary_metrics(labels, scores, weights=None, ignore_idx=-1, threshold=0.5):
    pred_labels = (scores > threshold).long()
    N, *Ds = labels.shape
    labels = labels.view(N, int(np.prod(Ds)))
    pred_labels = pred_labels.view(N, int(np.prod(Ds)))
    pred_trues = pred_labels > 0
    pred_falses = pred_labels == 0
    trues = labels > 0
    falses = labels == 0
    true_positives = (weights * (trues & pred_trues).float()).sum()
    true_negatives = (weights * (falses & pred_falses).float()).sum()
    false_positives = (weights * (falses & pred_trues).float()).sum()
    false_negatives = (weights * (trues & pred_falses).float()).sum()
    return true_positives, true_negatives, false_positives, false_negatives


class PrecisionRecall(nn.Module):

    def __init__(self, dim=1, ignore_idx=-1, thresholds=0.5, use_sigmoid_score=False, encode_background_as_zeros=True):
        super().__init__()
        if not isinstance(thresholds, (list, tuple)):
            thresholds = [thresholds]
        self.register_buffer('prec_total', torch.FloatTensor(len(thresholds)).zero_())
        self.register_buffer('prec_count', torch.FloatTensor(len(thresholds)).zero_())
        self.register_buffer('rec_total', torch.FloatTensor(len(thresholds)).zero_())
        self.register_buffer('rec_count', torch.FloatTensor(len(thresholds)).zero_())
        self._ignore_idx = ignore_idx
        self._dim = dim
        self._thresholds = thresholds
        self._use_sigmoid_score = use_sigmoid_score
        self._encode_background_as_zeros = encode_background_as_zeros

    def forward(self, labels, preds, weights=None):
        if self._encode_background_as_zeros:
            assert self._use_sigmoid_score is True
            total_scores = torch.sigmoid(preds)
        elif self._use_sigmoid_score:
            total_scores = torch.sigmoid(preds)[(...), 1:]
        else:
            total_scores = F.softmax(preds, dim=-1)[(...), 1:]
        """
        if preds.shape[self._dim] == 1:  # BCE
            scores = torch.sigmoid(preds)
        else:
            # assert preds.shape[
            #     self._dim] == 2, "precision only support 2 class"
            # TODO: add support for [N, C, ...] format.
            # TODO: add multiclass support
            if self._use_sigmoid_score:
                scores = torch.sigmoid(preds)[:, ..., 1:].sum(-1)
            else:
                scores = F.softmax(preds, dim=self._dim)[:, ..., 1:].sum(-1)
        """
        scores = torch.max(total_scores, dim=-1)[0]
        if weights is None:
            weights = (labels != self._ignore_idx).float()
        else:
            weights = weights.float()
        for i, thresh in enumerate(self._thresholds):
            tp, tn, fp, fn = _calc_binary_metrics(labels, scores, weights, self._ignore_idx, thresh)
            rec_count = tp + fn
            prec_count = tp + fp
            if rec_count > 0:
                self.rec_count[i] += rec_count
                self.rec_total[i] += tp
            if prec_count > 0:
                self.prec_count[i] += prec_count
                self.prec_total[i] += tp
        return self.value

    @property
    def value(self):
        prec_count = torch.clamp(self.prec_count, min=1.0)
        rec_count = torch.clamp(self.rec_count, min=1.0)
        return (self.prec_total / prec_count).cpu(), (self.rec_total / rec_count).cpu()

    @property
    def thresholds(self):
        return self._thresholds

    def clear(self):
        self.rec_count.zero_()
        self.prec_count.zero_()
        self.prec_total.zero_()
        self.rec_total.zero_()


class Empty(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super(Empty, self).__init__()

    def forward(self, *args, **kwargs):
        if len(args) == 1:
            return args[0]
        elif len(args) == 0:
            return None
        return args


class Sequential(torch.nn.Module):
    """A sequential container.
    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can also be passed in.

    To make it easier to understand, given is a small example::

        # Example of using Sequential
        model = Sequential(
                  nn.Conv2d(1,20,5),
                  nn.ReLU(),
                  nn.Conv2d(20,64,5),
                  nn.ReLU()
                )

        # Example of using Sequential with OrderedDict
        model = Sequential(OrderedDict([
                  ('conv1', nn.Conv2d(1,20,5)),
                  ('relu1', nn.ReLU()),
                  ('conv2', nn.Conv2d(20,64,5)),
                  ('relu2', nn.ReLU())
                ]))
        
        # Example of using Sequential with kwargs(python 3.6+)
        model = Sequential(
                  conv1=nn.Conv2d(1,20,5),
                  relu1=nn.ReLU(),
                  conv2=nn.Conv2d(20,64,5),
                  relu2=nn.ReLU()
                )
    """

    def __init__(self, *args, **kwargs):
        super(Sequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
        for name, module in kwargs.items():
            if sys.version_info < (3, 6):
                raise ValueError('kwargs only supported in py36+')
            if name in self._modules:
                raise ValueError('name exists.')
            self.add_module(name, module)

    def __getitem__(self, idx):
        if not -len(self) <= idx < len(self):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx += len(self)
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __len__(self):
        return len(self._modules)

    def add(self, module, name=None):
        if name is None:
            name = str(len(self._modules))
            if name in self._modules:
                raise KeyError('name exists')
        self.add_module(name, module)

    def forward(self, input):
        for module in self._modules.values():
            input = module(input)
        return input


class GroupNorm(torch.nn.GroupNorm):

    def __init__(self, num_channels, num_groups, eps=1e-05, affine=True):
        super().__init__(num_groups=num_groups, num_channels=num_channels, eps=eps, affine=affine)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Empty,
     lambda: ([], {}),
     lambda: ([], {}),
     False),
    (GroupNorm,
     lambda: ([], {'num_channels': 4, 'num_groups': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Sequential,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_jinfagang_alfred(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

