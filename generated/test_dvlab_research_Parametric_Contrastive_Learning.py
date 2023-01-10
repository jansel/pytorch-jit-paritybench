import sys
_module = sys.modules[__name__]
del sys
autoaug = _module
imagenet = _module
imagenet_moco = _module
imbalance_cifar = _module
inat = _module
inat_moco = _module
places_moco = _module
imagenet_r = _module
losses = _module
moco = _module
builder = _module
loader = _module
models = _module
resnet_cifar = _module
resnet_imagenet = _module
multitask_cifar = _module
multitask_lt = _module
paco_cifar = _module
paco_fp16_places = _module
paco_imagenet = _module
paco_lt = _module
paco_lt_ensemble = _module
randaugment = _module
utils = _module
engine_finetune = _module
engine_finetune_paco = _module
engine_pretrain = _module
main_finetune = _module
main_finetune_gpaco = _module
main_finetune_gpaco_eval = _module
main_finetune_multitask = _module
main_finetune_singleview = _module
main_linprobe = _module
main_pretrain = _module
builder = _module
models_mae = _module
models_vit = _module
models_vit_paco = _module
paco = _module
randaugment = _module
submitit_finetune = _module
submitit_linprobe = _module
submitit_pretrain = _module
crop = _module
datasets = _module
lars = _module
lr_decay = _module
lr_sched = _module
misc = _module
pos_embed = _module
ade20k = _module
ade20k_640x640 = _module
chase_db1 = _module
cityscapes = _module
cityscapes_1024x1024 = _module
cityscapes_769x769 = _module
drive = _module
hrf = _module
pascal_context = _module
pascal_context_59 = _module
pascal_voc12 = _module
pascal_voc12_aug = _module
stare = _module
default_runtime = _module
bisenetv2 = _module
cgnet = _module
fast_scnn = _module
fcn_hr18 = _module
fpn_r50 = _module
ocrnet_hr18 = _module
pointrend_r50 = _module
setr_mla = _module
setr_naive = _module
setr_pup = _module
upernet_r50 = _module
upernet_swin = _module
schedule_160k = _module
schedule_20k = _module
schedule_320k = _module
schedule_40k = _module
schedule_80k = _module
bisenetv2_fcn_4x4_1024x1024_160k_cityscapes = _module
bisenetv2_fcn_4x8_1024x1024_160k_cityscapes = _module
bisenetv2_fcn_fp16_4x4_1024x1024_160k_cityscapes = _module
bisenetv2_fcn_ohem_4x4_1024x1024_160k_cityscapes = _module
cgnet_512x1024_60k_cityscapes = _module
cgnet_680x680_60k_cityscapes = _module
fcn_hr18_480x480_40k_pascal_context = _module
fcn_hr18_480x480_40k_pascal_context_59 = _module
fcn_hr18_480x480_80k_pascal_context = _module
fcn_hr18_480x480_80k_pascal_context_59 = _module
fcn_hr18_512x1024_160k_cityscapes = _module
fcn_hr18_512x1024_40k_cityscapes = _module
fcn_hr18_512x1024_80k_cityscapes = _module
fcn_hr18_512x512_160k_ade20k = _module
fcn_hr18_512x512_20k_voc12aug = _module
fcn_hr18_512x512_40k_voc12aug = _module
fcn_hr18_512x512_80k_ade20k = _module
fcn_hr18s_480x480_40k_pascal_context = _module
fcn_hr18s_480x480_40k_pascal_context_59 = _module
fcn_hr18s_480x480_80k_pascal_context = _module
fcn_hr18s_480x480_80k_pascal_context_59 = _module
fcn_hr18s_512x1024_160k_cityscapes = _module
fcn_hr18s_512x1024_40k_cityscapes = _module
fcn_hr18s_512x1024_80k_cityscapes = _module
fcn_hr18s_512x512_160k_ade20k = _module
fcn_hr18s_512x512_20k_voc12aug = _module
fcn_hr18s_512x512_40k_voc12aug = _module
fcn_hr18s_512x512_80k_ade20k = _module
fcn_hr48_480x480_40k_pascal_context = _module
fcn_hr48_480x480_40k_pascal_context_59 = _module
fcn_hr48_480x480_80k_pascal_context = _module
fcn_hr48_480x480_80k_pascal_context_59 = _module
fcn_hr48_512x1024_160k_cityscapes = _module
fcn_hr48_512x1024_40k_cityscapes = _module
fcn_hr48_512x1024_80k_cityscapes = _module
fcn_hr48_512x512_160k_ade20k = _module
fcn_hr48_512x512_20k_voc12aug = _module
fcn_hr48_512x512_40k_voc12aug = _module
fcn_hr48_512x512_80k_ade20k = _module
ocrnet_hr18_512x1024_160k_cityscapes = _module
ocrnet_hr18_512x1024_40k_cityscapes = _module
ocrnet_hr18_512x1024_80k_cityscapes = _module
ocrnet_hr18_512x512_160k_ade20k = _module
ocrnet_hr18_512x512_20k_voc12aug = _module
ocrnet_hr18_512x512_40k_voc12aug = _module
ocrnet_hr18_512x512_80k_ade20k = _module
ocrnet_hr18s_512x1024_160k_cityscapes = _module
ocrnet_hr18s_512x1024_40k_cityscapes = _module
ocrnet_hr18s_512x1024_80k_cityscapes = _module
ocrnet_hr18s_512x512_160k_ade20k = _module
ocrnet_hr18s_512x512_20k_voc12aug = _module
ocrnet_hr18s_512x512_40k_voc12aug = _module
ocrnet_hr18s_512x512_80k_ade20k = _module
ocrnet_hr48_512x1024_160k_cityscapes = _module
ocrnet_hr48_512x1024_40k_cityscapes = _module
ocrnet_hr48_512x1024_80k_cityscapes = _module
ocrnet_hr48_512x512_160k_ade20k = _module
ocrnet_hr48_512x512_20k_voc12aug = _module
ocrnet_hr48_512x512_40k_voc12aug = _module
ocrnet_hr48_512x512_80k_ade20k = _module
upernet_swin_base_patch4_window12_512x512_160k_ade20k_pretrain_384x384_22K_paco = _module
upernet_swin_base_patch4_window12_512x512_160k_ade20k_pretrain_384x384_22K_paco11 = _module
upernet_swin_large_patch4_window12_512x512_160k_ade20k_pretrain_384x384_22K_paco = _module
upernet_swin_large_patch4_window12_512x512_160k_ade20k_pretrain_384x384_22K_paco11 = _module
upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K_paco = _module
upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K_paco11 = _module
pointrend_r101_512x1024_80k_cityscapes = _module
pointrend_r101_512x512_160k_ade20k = _module
pointrend_r50_512x1024_80k_cityscapes = _module
pointrend_r50_512x512_160k_ade20k = _module
fpn_r101_512x1024_80k_cityscapes = _module
fpn_r101_512x512_160k_ade20k = _module
fpn_r50_512x1024_80k_cityscapes = _module
fpn_r50_512x512_160k_ade20k = _module
setr_mla_512x512_160k_b16_ade20k = _module
setr_mla_512x512_160k_b8_ade20k = _module
setr_naive_512x512_160k_b16_ade20k = _module
setr_pup_512x512_160k_b16_ade20k = _module
upernet_swin_base_patch4_window12_512x512_160k_ade20k_pretrain_384x384_1K = _module
upernet_swin_base_patch4_window12_512x512_160k_ade20k_pretrain_384x384_22K = _module
upernet_swin_base_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K = _module
upernet_swin_base_patch4_window7_512x512_160k_ade20k_pretrain_224x224_22K = _module
upernet_swin_small_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K = _module
upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K = _module
upernet_r101_512x1024_40k_cityscapes = _module
upernet_r101_512x1024_80k_cityscapes = _module
upernet_r101_512x512_160k_ade20k = _module
upernet_r101_512x512_20k_voc12aug = _module
upernet_r101_512x512_40k_voc12aug = _module
upernet_r101_512x512_80k_ade20k = _module
upernet_r101_769x769_40k_cityscapes = _module
upernet_r101_769x769_80k_cityscapes = _module
upernet_r50_512x1024_40k_cityscapes = _module
upernet_r50_512x1024_80k_cityscapes = _module
upernet_r50_512x512_160k_ade20k = _module
upernet_r50_512x512_20k_voc12aug = _module
upernet_r50_512x512_40k_voc12aug = _module
upernet_r50_512x512_80k_ade20k = _module
upernet_r50_769x769_40k_cityscapes = _module
upernet_r50_769x769_80k_cityscapes = _module
image_demo = _module
conf = _module
stat = _module
mmseg = _module
apis = _module
inference = _module
test = _module
train = _module
core = _module
evaluation = _module
class_names = _module
eval_hooks = _module
metrics = _module
seg = _module
sampler = _module
base_pixel_sampler = _module
ohem_pixel_sampler = _module
ade = _module
builder = _module
coco_stuff = _module
custom = _module
dark_zurich = _module
dataset_wrappers = _module
night_driving = _module
pipelines = _module
compose = _module
formating = _module
loading = _module
test_time_aug = _module
transforms = _module
voc = _module
backbones = _module
bisenetv1 = _module
bisenetv2 = _module
cgnet = _module
fast_scnn = _module
hrnet = _module
mit = _module
mobilenet_v2 = _module
mobilenet_v3 = _module
resnest = _module
resnet = _module
resnext = _module
swin = _module
unet = _module
vit = _module
decode_heads = _module
ann_head = _module
apc_head = _module
aspp_head = _module
cascade_decode_head = _module
cc_head = _module
da_head = _module
decode_head = _module
dm_head = _module
dnl_head = _module
dpt_head = _module
ema_head = _module
enc_head = _module
fcn_head = _module
focal = _module
fpn_head = _module
gc_head = _module
isa_head = _module
lovasz_loss = _module
lraspp_head = _module
nl_head = _module
ocr_head = _module
paco = _module
point_head = _module
psa_head = _module
psp_head = _module
rr = _module
segformer_head = _module
sep_aspp_head = _module
sep_fcn_head = _module
setr_mla_head = _module
setr_up_head = _module
uper_head = _module
uper_head_lovasz = _module
uper_head_paco = _module
accuracy = _module
cross_entropy_loss = _module
dice_loss = _module
lovasz_loss = _module
utils = _module
necks = _module
fpn = _module
mla_neck = _module
multilevel_neck = _module
segmentors = _module
base = _module
cascade_encoder_decoder = _module
encoder_decoder = _module
embed = _module
inverted_residual = _module
make_divisible = _module
res_layer = _module
se_layer = _module
self_attention_block = _module
shape_convert = _module
up_conv_block = _module
ops = _module
encoding = _module
wrappers = _module
collect_env = _module
logger = _module
version = _module
setup = _module
stats_cityscapes = _module
stats_cocostuff10k = _module
stats_cocostuff164k = _module
stats_pix_ade20k = _module
stats_pix_cityscapes = _module
stats_pix_cocostuff10k = _module
stats_pix_cocostuff164k = _module
tests = _module
test_single_gpu = _module
test_config = _module
test_dataset = _module
test_dataset_builder = _module
test_loading = _module
test_transform = _module
test_tta = _module
test_digit_version = _module
test_eval_hook = _module
test_inference = _module
test_metrics = _module
test_models = _module
test_backbones = _module
test_bisenetv1 = _module
test_bisenetv2 = _module
test_blocks = _module
test_cgnet = _module
test_fast_scnn = _module
test_hrnet = _module
test_mit = _module
test_mobilenet_v3 = _module
test_resnest = _module
test_resnet = _module
test_resnext = _module
test_swin = _module
test_unet = _module
test_vit = _module
utils = _module
test_forward = _module
test_heads = _module
test_ann_head = _module
test_apc_head = _module
test_aspp_head = _module
test_cc_head = _module
test_da_head = _module
test_decode_head = _module
test_dm_head = _module
test_dnl_head = _module
test_dpt_head = _module
test_ema_head = _module
test_enc_head = _module
test_fcn_head = _module
test_gc_head = _module
test_isa_head = _module
test_lraspp_head = _module
test_nl_head = _module
test_ocr_head = _module
test_point_head = _module
test_psa_head = _module
test_psp_head = _module
test_segformer_head = _module
test_setr_mla_head = _module
test_setr_up_head = _module
test_uper_head = _module
test_losses = _module
test_ce_loss = _module
test_dice_loss = _module
test_lovasz_loss = _module
test_utils = _module
test_necks = _module
test_fpn = _module
test_mla_neck = _module
test_multilevel_neck = _module
test_segmentors = _module
test_cascade_encoder_decoder = _module
test_encoder_decoder = _module
utils = _module
test_embed = _module
test_sampler = _module
analyze_logs = _module
benchmark = _module
browse_dataset = _module
coco_stuff10k = _module
coco_stuff164k = _module
voc_aug = _module
deploy_test = _module
get_flops = _module
mit2mmseg = _module
swin2mmseg = _module
vit2mmseg = _module
onnx2tensorrt = _module
print_config = _module
publish_model = _module
pytorch2onnx = _module
pytorch2torchscript = _module
test = _module
mmseg2torchserve = _module
mmseg_handler = _module
test_torchserve = _module
train = _module
txt_names = _module
ade_names = _module
losses = _module
builder = _module
resnet = _module
paco_imagenet = _module
randaugment = _module
autoaug = _module
imagenet = _module
imagenet_due = _module
imagenet_moco = _module
inat = _module
inat_due = _module
inat_moco = _module
places = _module
places_due = _module
places_moco = _module
losses = _module
builder = _module
resnet_cifar = _module
resnet_imagenet = _module
paco_cifar = _module
paco_fp16_places = _module
paco_lt = _module
randaugment = _module
utils = _module

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


import numpy as np


import random


import torch


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from torchvision import transforms


import torchvision


from torch.utils.data import ConcatDataset


import torch.nn as nn


import torch.nn.functional as F


from torch.nn import Parameter


import torch.nn.init as init


from torch import Tensor


from typing import Type


from typing import Any


from typing import Callable


from typing import Union


from typing import List


from typing import Optional


import math


import time


import warnings


import torch.nn.parallel


import torch.backends.cudnn as cudnn


import torch.distributed as dist


import torch.optim


import torch.multiprocessing as mp


import torch.utils.data


import torch.utils.data.distributed


import torchvision.transforms as transforms


import torchvision.models as models


import torchvision.datasets as datasets


from torch.cuda.amp import GradScaler


from torch.cuda.amp import autocast


import re


import matplotlib.pyplot as plt


from sklearn.metrics import f1_score


from typing import Iterable


from torchvision import datasets


from torch.utils.tensorboard import SummaryWriter


from functools import partial


from torchvision.transforms import functional as F


from collections import defaultdict


from collections import deque


from torch._six import inf


from torch.nn.modules.batchnorm import _BatchNorm


from collections import OrderedDict


import copy


from torch.utils.data import DistributedSampler


from itertools import chain


from torch.utils.data.dataset import ConcatDataset as _ConcatDataset


from collections.abc import Sequence


import torch.utils.checkpoint as cp


from copy import deepcopy


from torch.nn.modules.utils import _pair as to_2tuple


from torch import nn


from abc import ABCMeta


from abc import abstractmethod


from torch.autograd import Variable


import functools


from typing import Sequence


from torch.utils import checkpoint as cp


from torch import nn as nn


from torch.nn import functional as F


from torch.utils.data import dataloader


from typing import Generator


from torch.utils.data import RandomSampler


from torch.utils.data import SequentialSampler


import logging


from torch.nn.modules import AvgPool2d


from torch.nn.modules import GroupNorm


import torch._C


import torch.serialization


class PaCoLoss(nn.Module):

    def __init__(self, alpha, beta=1.0, gamma=1.0, supt=1.0, temperature=1.0, base_temperature=None, K=128, num_classes=1000):
        super(PaCoLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = temperature if base_temperature is None else base_temperature
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.supt = supt
        self.num_classes = num_classes

    def cal_weight_for_classes(self, cls_num_list):
        cls_num_list = torch.Tensor(cls_num_list).view(1, self.num_classes)
        self.weight = cls_num_list / cls_num_list.sum()
        self.weight = self.weight

    def forward(self, features, labels=None, sup_logits=None, mask=None, epoch=None):
        device = torch.device('cuda') if features.is_cuda else torch.device('cpu')
        ss = features.shape[0]
        batch_size = (features.shape[0] - self.K) // 2
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels[:batch_size], labels.T).float()
        anchor_dot_contrast = torch.div(torch.matmul(features[:batch_size], features.T), self.temperature)
        anchor_dot_contrast = torch.cat(((sup_logits + torch.log(self.weight + 1e-09)) / self.supt, anchor_dot_contrast), dim=1)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1), 0)
        mask = mask * logits_mask
        one_hot_label = torch.nn.functional.one_hot(labels[:batch_size,].view(-1), num_classes=self.num_classes)
        mask = torch.cat((one_hot_label * self.beta, mask * self.alpha), dim=1)
        logits_mask = torch.cat((torch.ones(batch_size, self.num_classes), self.gamma * logits_mask), dim=1)
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()
        return loss


class MultiTaskLoss(nn.Module):

    def __init__(self, alpha, beta=1.0, gamma=1.0, supt=1.0, temperature=1.0, base_temperature=None, K=8192, num_classes=1000, smooth=0.1):
        super(MultiTaskLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = temperature if base_temperature is None else base_temperature
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.supt = supt
        self.num_classes = num_classes
        self.effective_num_beta = 0.999
        self.criterion = LabelSmoothingCrossEntropy(smoothing=smooth)

    def cal_weight_for_classes(self, cls_num_list):
        cls_num_list = torch.Tensor(cls_num_list).view(1, self.num_classes)
        self.weight = cls_num_list / cls_num_list.sum()
        self.weight = self.weight
        if self.effective_num_beta != 0:
            effective_num = np.array(1.0 - np.power(self.effective_num_beta, cls_num_list)) / (1.0 - self.effective_num_beta)
            per_cls_weights = sum(effective_num) / len(effective_num) / effective_num
            self.class_weight = torch.FloatTensor(per_cls_weights)

    def forward(self, features, labels=None, sup_logits=None, mask=None, epoch=None):
        device = torch.device('cuda') if features.is_cuda else torch.device('cpu')
        batch_size = features.shape[0] - self.K
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels[:batch_size], labels.T).float()
        anchor_dot_contrast = torch.div(torch.matmul(features[:batch_size], features.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1), 0)
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()
        loss_ce = self.criterion(sup_logits, labels[:batch_size].squeeze())
        return loss_ce + self.alpha * loss


class MultiTaskBLoss(nn.Module):

    def __init__(self, alpha, beta=1.0, gamma=1.0, supt=1.0, temperature=1.0, base_temperature=None, K=8192, num_classes=1000):
        super(MultiTaskBLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = temperature if base_temperature is None else base_temperature
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.supt = supt
        self.num_classes = num_classes
        self.effective_num_beta = 0.999

    def cal_weight_for_classes(self, cls_num_list):
        cls_num_list = torch.Tensor(cls_num_list).view(1, self.num_classes)
        self.weight = cls_num_list / cls_num_list.sum()
        self.weight = self.weight
        if self.effective_num_beta != 0:
            effective_num = np.array(1.0 - np.power(self.effective_num_beta, cls_num_list)) / (1.0 - self.effective_num_beta)
            per_cls_weights = sum(effective_num) / len(effective_num) / effective_num
            self.class_weight = torch.FloatTensor(per_cls_weights)

    def forward(self, features, labels=None, sup_logits=None, mask=None, epoch=None):
        device = torch.device('cuda') if features.is_cuda else torch.device('cpu')
        batch_size = features.shape[0] - self.K
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels[:batch_size], labels.T).float()
        anchor_dot_contrast = torch.div(torch.matmul(features[:batch_size], features.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1), 0)
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()
        loss_ce = F.cross_entropy(sup_logits, labels[:batch_size].squeeze())
        return loss_ce + self.alpha * loss


class NormedLinear_Classifier(nn.Module):

    def __init__(self, num_classes=1000, feat_dim=2048):
        super(NormedLinear_Classifier, self).__init__()
        self.weight = Parameter(torch.Tensor(feat_dim, num_classes))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-05).mul_(100000.0)

    def forward(self, x, *args):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output


def flatten(t):
    return t.reshape(t.shape[0], -1)


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.2, mlp=False, feat_dim=2048, normalize=False, num_classes=1000):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()
        self.K = K
        self.m = m
        self.T = T
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)
        self.linear = nn.Linear(feat_dim, num_classes)
        self.linear_k = nn.Linear(feat_dim, num_classes)
        if mlp:
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.BatchNorm1d(dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.BatchNorm1d(dim_mlp), nn.ReLU(), self.encoder_k.fc)
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        for param_q, param_k in zip(self.linear.parameters(), self.linear_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        self.register_buffer('queue', torch.randn(K, dim))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer('queue_l', torch.randint(0, num_classes, (K,)))
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
        self.layer = -2
        self.feat_after_avg_k = None
        self.feat_after_avg_q = None
        self._register_hook()
        self.normalize = normalize

    def _find_layer(self, module):
        if type(self.layer) == str:
            modules = dict([*module.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*module.children()]
            return children[self.layer]
        return None

    def _hook_k(self, _, __, output):
        self.feat_after_avg_k = flatten(output)
        if self.normalize:
            self.feat_after_avg_k = nn.functional.normalize(self.feat_after_avg_k, dim=1)

    def _hook_q(self, _, __, output):
        self.feat_after_avg_q = flatten(output)
        if self.normalize:
            self.feat_after_avg_q = nn.functional.normalize(self.feat_after_avg_q, dim=1)

    def _register_hook(self):
        layer_k = self._find_layer(self.encoder_k)
        assert layer_k is not None, f'hidden layer ({self.layer}) not found'
        handle = layer_k.register_forward_hook(self._hook_k)
        layer_q = self._find_layer(self.encoder_q)
        assert layer_q is not None, f'hidden layer ({self.layer}) not found'
        handle = layer_q.register_forward_hook(self._hook_q)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)
        for param_q, param_k in zip(self.linear.parameters(), self.linear_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels):
        keys = concat_all_gather(keys)
        labels = concat_all_gather(labels)
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0
        self.queue[ptr:ptr + batch_size, :] = keys
        self.queue_l[ptr:ptr + batch_size] = labels
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x, y):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        y_gather = concat_all_gather(y)
        batch_size_all = x_gather.shape[0]
        num_gpus = batch_size_all // batch_size_this
        idx_shuffle = torch.randperm(batch_size_all)
        torch.distributed.broadcast(idx_shuffle, src=0)
        idx_unshuffle = torch.argsort(idx_shuffle)
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]
        return x_gather[idx_this], y_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, y, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        y_gather = concat_all_gather(y)
        batch_size_all = x_gather.shape[0]
        num_gpus = batch_size_all // batch_size_this
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]
        return x_gather[idx_this], y_gather[idx_this]

    def _train(self, im_q, im_k, labels):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        q = self.encoder_q(im_q)
        q = nn.functional.normalize(q, dim=1)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            im_k, labels, idx_unshuffle = self._batch_shuffle_ddp(im_k, labels)
            k = self.encoder_k(im_k)
            k = nn.functional.normalize(k, dim=1)
            k, labels = self._batch_unshuffle_ddp(k, labels, idx_unshuffle)
        features = torch.cat((q, k, self.queue.clone().detach()), dim=0)
        target = torch.cat((labels, labels, self.queue_l.clone().detach()), dim=0)
        self._dequeue_and_enqueue(k, labels)
        logits_q = self.linear(self.feat_after_avg_q)
        return features, target, logits_q

    def _inference(self, image):
        q = self.encoder_q(image)
        q = nn.functional.normalize(q, dim=1)
        encoder_q_logits = self.linear(self.feat_after_avg_q)
        return encoder_q_logits

    def forward(self, im_q, im_k=None, labels=None):
        if self.training:
            return self._train(im_q, im_k, labels)
        else:
            return self._inference(im_q)


class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-05).mul_(100000.0)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out


class LambdaLayer(nn.Module):

    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


def conv3x3(in_planes: int, out_planes: int, stride: int=1, groups: int=1, dilation: int=1) ->nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self, inplanes: int, planes: int, stride: int=1, downsample: Optional[nn.Module]=None, groups: int=1, base_width: int=64, dilation: int=1, norm_layer: Optional[Callable[..., nn.Module]]=None) ->None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError('Dilation > 1 not supported in BasicBlock')
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) ->Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class ResNet_s(nn.Module):

    def __init__(self, block, num_blocks, num_classes=10, use_norm=False, return_features=False):
        super(ResNet_s, self).__init__()
        factor = 1
        self.in_planes = 16 * factor
        self.conv1 = nn.Conv2d(3, 16 * factor, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16 * factor)
        self.layer1 = self._make_layer(block, 16 * factor, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32 * factor, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64 * factor, num_blocks[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if use_norm:
            self.fc = NormedLinear(64 * factor, num_classes)
        else:
            self.fc = nn.Linear(64 * factor, num_classes)
        self.apply(_weights_init)
        self.return_encoding = return_features

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        encoding = out.view(out.size(0), -1)
        out = self.fc(encoding)
        if self.return_encoding:
            return out, encoding
        else:
            return out


def conv1x1(in_planes: int, out_planes: int, stride: int=1) ->nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(self, inplanes: int, planes: int, stride: int=1, downsample: Optional[nn.Module]=None, groups: int=1, base_width: int=64, dilation: int=1, norm_layer: Optional[Callable[..., nn.Module]]=None) ->None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) ->Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block: Type[Union[BasicBlock, Bottleneck]], layers: List[int], num_classes: int=1000, zero_init_residual: bool=False, groups: int=1, width_per_group: int=64, replace_stride_with_dilation: Optional[List[bool]]=None, norm_layer: Optional[Callable[..., nn.Module]]=None) ->None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError('replace_stride_with_dilation should be None or a 3-element tuple, got {}'.format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int, stride: int=1, dilate: bool=False) ->nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride), norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) ->Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, x: Tensor) ->Tensor:
        return self._forward_impl(x)


class AdaptivePadding(nn.Module):
    """Applies padding to input (if needed) so that input can get fully covered
    by filter you specified. It support two modes "same" and "corner". The
    "same" mode is same with "SAME" padding mode in TensorFlow, pad zero around
    input. The "corner"  mode would pad zero to bottom right.

    Args:
        kernel_size (int | tuple): Size of the kernel:
        stride (int | tuple): Stride of the filter. Default: 1:
        dilation (int | tuple): Spacing between kernel elements.
            Default: 1.
        padding (str): Support "same" and "corner", "corner" mode
            would pad zero to bottom right, and "same" mode would
            pad zero around input. Default: "corner".
    Example:
        >>> kernel_size = 16
        >>> stride = 16
        >>> dilation = 1
        >>> input = torch.rand(1, 1, 15, 17)
        >>> adap_pad = AdaptivePadding(
        >>>     kernel_size=kernel_size,
        >>>     stride=stride,
        >>>     dilation=dilation,
        >>>     padding="corner")
        >>> out = adap_pad(input)
        >>> assert (out.shape[2], out.shape[3]) == (16, 32)
        >>> input = torch.rand(1, 1, 16, 17)
        >>> out = adap_pad(input)
        >>> assert (out.shape[2], out.shape[3]) == (16, 32)
    """

    def __init__(self, kernel_size=1, stride=1, dilation=1, padding='corner'):
        super(AdaptivePadding, self).__init__()
        assert padding in ('same', 'corner')
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)
        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

    def get_pad_shape(self, input_shape):
        input_h, input_w = input_shape
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        output_h = math.ceil(input_h / stride_h)
        output_w = math.ceil(input_w / stride_w)
        pad_h = max((output_h - 1) * stride_h + (kernel_h - 1) * self.dilation[0] + 1 - input_h, 0)
        pad_w = max((output_w - 1) * stride_w + (kernel_w - 1) * self.dilation[1] + 1 - input_w, 0)
        return pad_h, pad_w

    def forward(self, x):
        pad_h, pad_w = self.get_pad_shape(x.size()[-2:])
        if pad_h > 0 or pad_w > 0:
            if self.padding == 'corner':
                x = F.pad(x, [0, pad_w, 0, pad_h])
            elif self.padding == 'same':
                x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return x


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=1024, depth=24, num_heads=16, decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, mlp_ratio=4.0, norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer) for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)
        self.decoder_blocks = nn.ModuleList([Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer) for i in range(decoder_depth)])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)
        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** 0.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches ** 0.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)
        x = x + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        x = x[:, 1:, :]
        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-06) ** 0.5
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


class GlobalContextExtractor(nn.Module):
    """Global Context Extractor for CGNet.

    This class is employed to refine the joint feature of both local feature
    and surrounding context.

    Args:
        channel (int): Number of input feature channels.
        reduction (int): Reductions for global context extractor. Default: 16.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
    """

    def __init__(self, channel, reduction=16, with_cp=False):
        super(GlobalContextExtractor, self).__init__()
        self.channel = channel
        self.reduction = reduction
        assert reduction >= 1 and channel >= reduction
        self.with_cp = with_cp
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channel, channel // reduction), nn.ReLU(inplace=True), nn.Linear(channel // reduction, channel), nn.Sigmoid())

    def forward(self, x):

        def _inner_forward(x):
            num_batch, num_channel = x.size()[:2]
            y = self.avg_pool(x).view(num_batch, num_channel)
            y = self.fc(y).view(num_batch, num_channel, 1, 1)
            return x * y
        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)
        return out


class ContextGuidedBlock(nn.Module):
    """Context Guided Block for CGNet.

    This class consists of four components: local feature extractor,
    surrounding feature extractor, joint feature extractor and global
    context extractor.

    Args:
        in_channels (int): Number of input feature channels.
        out_channels (int): Number of output feature channels.
        dilation (int): Dilation rate for surrounding context extractor.
            Default: 2.
        reduction (int): Reduction for global context extractor. Default: 16.
        skip_connect (bool): Add input to output or not. Default: True.
        downsample (bool): Downsample the input to 1/2 or not. Default: False.
        conv_cfg (dict): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN', requires_grad=True).
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='PReLU').
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
    """

    def __init__(self, in_channels, out_channels, dilation=2, reduction=16, skip_connect=True, downsample=False, conv_cfg=None, norm_cfg=dict(type='BN', requires_grad=True), act_cfg=dict(type='PReLU'), with_cp=False):
        super(ContextGuidedBlock, self).__init__()
        self.with_cp = with_cp
        self.downsample = downsample
        channels = out_channels if downsample else out_channels // 2
        if 'type' in act_cfg and act_cfg['type'] == 'PReLU':
            act_cfg['num_parameters'] = channels
        kernel_size = 3 if downsample else 1
        stride = 2 if downsample else 1
        padding = (kernel_size - 1) // 2
        self.conv1x1 = ConvModule(in_channels, channels, kernel_size, stride, padding, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.f_loc = build_conv_layer(conv_cfg, channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.f_sur = build_conv_layer(conv_cfg, channels, channels, kernel_size=3, padding=dilation, groups=channels, dilation=dilation, bias=False)
        self.bn = build_norm_layer(norm_cfg, 2 * channels)[1]
        self.activate = nn.PReLU(2 * channels)
        if downsample:
            self.bottleneck = build_conv_layer(conv_cfg, 2 * channels, out_channels, kernel_size=1, bias=False)
        self.skip_connect = skip_connect and not downsample
        self.f_glo = GlobalContextExtractor(out_channels, reduction, with_cp)

    def forward(self, x):

        def _inner_forward(x):
            out = self.conv1x1(x)
            loc = self.f_loc(out)
            sur = self.f_sur(out)
            joi_feat = torch.cat([loc, sur], 1)
            joi_feat = self.bn(joi_feat)
            joi_feat = self.activate(joi_feat)
            if self.downsample:
                joi_feat = self.bottleneck(joi_feat)
            out = self.f_glo(joi_feat)
            if self.skip_connect:
                return x + out
            else:
                return out
        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)
        return out


class InputInjection(nn.Module):
    """Downsampling module for CGNet."""

    def __init__(self, num_downsampling):
        super(InputInjection, self).__init__()
        self.pool = nn.ModuleList()
        for i in range(num_downsampling):
            self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, x):
        for pool in self.pool:
            x = pool(x)
        return x


class LearningToDownsample(nn.Module):
    """Learning to downsample module.

    Args:
        in_channels (int): Number of input channels.
        dw_channels (tuple[int]): Number of output channels of the first and
            the second depthwise conv (dwconv) layers.
        out_channels (int): Number of output channels of the whole
            'learning to downsample' module.
        conv_cfg (dict | None): Config of conv layers. Default: None
        norm_cfg (dict | None): Config of norm layers. Default:
            dict(type='BN')
        act_cfg (dict): Config of activation layers. Default:
            dict(type='ReLU')
        dw_act_cfg (dict): In DepthwiseSeparableConvModule, activation config
            of depthwise ConvModule. If it is 'default', it will be the same
            as `act_cfg`. Default: None.
    """

    def __init__(self, in_channels, dw_channels, out_channels, conv_cfg=None, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU'), dw_act_cfg=None):
        super(LearningToDownsample, self).__init__()
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.dw_act_cfg = dw_act_cfg
        dw_channels1 = dw_channels[0]
        dw_channels2 = dw_channels[1]
        self.conv = ConvModule(in_channels, dw_channels1, 3, stride=2, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        self.dsconv1 = DepthwiseSeparableConvModule(dw_channels1, dw_channels2, kernel_size=3, stride=2, padding=1, norm_cfg=self.norm_cfg, dw_act_cfg=self.dw_act_cfg)
        self.dsconv2 = DepthwiseSeparableConvModule(dw_channels2, out_channels, kernel_size=3, stride=2, padding=1, norm_cfg=self.norm_cfg, dw_act_cfg=self.dw_act_cfg)

    def forward(self, x):
        x = self.conv(x)
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        return x


class InvertedResidual(nn.Module):
    """InvertedResidual block for MobileNetV2.

    Args:
        in_channels (int): The input channels of the InvertedResidual block.
        out_channels (int): The output channels of the InvertedResidual block.
        stride (int): Stride of the middle (first) 3x3 convolution.
        expand_ratio (int): Adjusts number of channels of the hidden layer
            in InvertedResidual by this amount.
        dilation (int): Dilation rate of depthwise conv. Default: 1
        conv_cfg (dict): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU6').
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.

    Returns:
        Tensor: The output tensor.
    """

    def __init__(self, in_channels, out_channels, stride, expand_ratio, dilation=1, conv_cfg=None, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU6'), with_cp=False, **kwargs):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2], f'stride must in [1, 2]. But received {stride}.'
        self.with_cp = with_cp
        self.use_res_connect = self.stride == 1 and in_channels == out_channels
        hidden_dim = int(round(in_channels * expand_ratio))
        layers = []
        if expand_ratio != 1:
            layers.append(ConvModule(in_channels=in_channels, out_channels=hidden_dim, kernel_size=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg, **kwargs))
        layers.extend([ConvModule(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, groups=hidden_dim, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg, **kwargs), ConvModule(in_channels=hidden_dim, out_channels=out_channels, kernel_size=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=None, **kwargs)])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):

        def _inner_forward(x):
            if self.use_res_connect:
                return x + self.conv(x)
            else:
                return self.conv(x)
        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)
        return out


def resize(input, size=None, scale_factor=None, mode='nearest', align_corners=None, warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if (output_h > 1 and output_w > 1 and input_h > 1 and input_w > 1) and (output_h - 1) % (input_h - 1) and (output_w - 1) % (input_w - 1):
                    warnings.warn(f'When align_corners={align_corners}, the output would more aligned if input size {input_h, input_w} is `x+1` and out size {output_h, output_w} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class PPM(nn.ModuleList):
    """Pooling Pyramid Module used in PSPNet.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
        align_corners (bool): align_corners argument of F.interpolate.
    """

    def __init__(self, pool_scales, in_channels, channels, conv_cfg, norm_cfg, act_cfg, align_corners, **kwargs):
        super(PPM, self).__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        for pool_scale in pool_scales:
            self.append(nn.Sequential(nn.AdaptiveAvgPool2d(pool_scale), ConvModule(self.in_channels, self.channels, 1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg, **kwargs)))

    def forward(self, x):
        """Forward function."""
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(x)
            upsampled_ppm_out = resize(ppm_out, size=x.size()[2:], mode='bilinear', align_corners=self.align_corners)
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs


class GlobalFeatureExtractor(nn.Module):
    """Global feature extractor module.

    Args:
        in_channels (int): Number of input channels of the GFE module.
            Default: 64
        block_channels (tuple[int]): Tuple of ints. Each int specifies the
            number of output channels of each Inverted Residual module.
            Default: (64, 96, 128)
        out_channels(int): Number of output channels of the GFE module.
            Default: 128
        expand_ratio (int): Adjusts number of channels of the hidden layer
            in InvertedResidual by this amount.
            Default: 6
        num_blocks (tuple[int]): Tuple of ints. Each int specifies the
            number of times each Inverted Residual module is repeated.
            The repeated Inverted Residual modules are called a 'group'.
            Default: (3, 3, 3)
        strides (tuple[int]): Tuple of ints. Each int specifies
            the downsampling factor of each 'group'.
            Default: (2, 2, 1)
        pool_scales (tuple[int]): Tuple of ints. Each int specifies
            the parameter required in 'global average pooling' within PPM.
            Default: (1, 2, 3, 6)
        conv_cfg (dict | None): Config of conv layers. Default: None
        norm_cfg (dict | None): Config of norm layers. Default:
            dict(type='BN')
        act_cfg (dict): Config of activation layers. Default:
            dict(type='ReLU')
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False
    """

    def __init__(self, in_channels=64, block_channels=(64, 96, 128), out_channels=128, expand_ratio=6, num_blocks=(3, 3, 3), strides=(2, 2, 1), pool_scales=(1, 2, 3, 6), conv_cfg=None, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU'), align_corners=False):
        super(GlobalFeatureExtractor, self).__init__()
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        assert len(block_channels) == len(num_blocks) == 3
        self.bottleneck1 = self._make_layer(in_channels, block_channels[0], num_blocks[0], strides[0], expand_ratio)
        self.bottleneck2 = self._make_layer(block_channels[0], block_channels[1], num_blocks[1], strides[1], expand_ratio)
        self.bottleneck3 = self._make_layer(block_channels[1], block_channels[2], num_blocks[2], strides[2], expand_ratio)
        self.ppm = PPM(pool_scales, block_channels[2], block_channels[2] // 4, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg, align_corners=align_corners)
        self.out = ConvModule(block_channels[2] * 2, out_channels, 3, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1, expand_ratio=6):
        layers = [InvertedResidual(in_channels, out_channels, stride, expand_ratio, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)]
        for i in range(1, blocks):
            layers.append(InvertedResidual(out_channels, out_channels, 1, expand_ratio, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = torch.cat([x, *self.ppm(x)], dim=1)
        x = self.out(x)
        return x


class FeatureFusionModule(nn.Module):
    """Feature fusion module.

    Args:
        higher_in_channels (int): Number of input channels of the
            higher-resolution branch.
        lower_in_channels (int): Number of input channels of the
            lower-resolution branch.
        out_channels (int): Number of output channels.
        conv_cfg (dict | None): Config of conv layers. Default: None
        norm_cfg (dict | None): Config of norm layers. Default:
            dict(type='BN')
        dwconv_act_cfg (dict): Config of activation layers in 3x3 conv.
            Default: dict(type='ReLU').
        conv_act_cfg (dict): Config of activation layers in the two 1x1 conv.
            Default: None.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
    """

    def __init__(self, higher_in_channels, lower_in_channels, out_channels, conv_cfg=None, norm_cfg=dict(type='BN'), dwconv_act_cfg=dict(type='ReLU'), conv_act_cfg=None, align_corners=False):
        super(FeatureFusionModule, self).__init__()
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.dwconv_act_cfg = dwconv_act_cfg
        self.conv_act_cfg = conv_act_cfg
        self.align_corners = align_corners
        self.dwconv = ConvModule(lower_in_channels, out_channels, 3, padding=1, groups=out_channels, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.dwconv_act_cfg)
        self.conv_lower_res = ConvModule(out_channels, out_channels, 1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.conv_act_cfg)
        self.conv_higher_res = ConvModule(higher_in_channels, out_channels, 1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.conv_act_cfg)
        self.relu = nn.ReLU(True)

    def forward(self, higher_res_feature, lower_res_feature):
        lower_res_feature = resize(lower_res_feature, size=higher_res_feature.size()[2:], mode='bilinear', align_corners=self.align_corners)
        lower_res_feature = self.dwconv(lower_res_feature)
        lower_res_feature = self.conv_lower_res(lower_res_feature)
        higher_res_feature = self.conv_higher_res(higher_res_feature)
        out = higher_res_feature + lower_res_feature
        return self.relu(out)


class RSoftmax(nn.Module):
    """Radix Softmax module in ``SplitAttentionConv2d``.

    Args:
        radix (int): Radix of input.
        groups (int): Groups of input.
    """

    def __init__(self, radix, groups):
        super().__init__()
        self.radix = radix
        self.groups = groups

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.groups, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x


class SplitAttentionConv2d(nn.Module):
    """Split-Attention Conv2d in ResNeSt.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int | tuple[int]): Same as nn.Conv2d.
        stride (int | tuple[int]): Same as nn.Conv2d.
        padding (int | tuple[int]): Same as nn.Conv2d.
        dilation (int | tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        radix (int): Radix of SpltAtConv2d. Default: 2
        reduction_factor (int): Reduction factor of inter_channels. Default: 4.
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        dcn (dict): Config dict for DCN. Default: None.
    """

    def __init__(self, in_channels, channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, radix=2, reduction_factor=4, conv_cfg=None, norm_cfg=dict(type='BN'), dcn=None):
        super(SplitAttentionConv2d, self).__init__()
        inter_channels = max(in_channels * radix // reduction_factor, 32)
        self.radix = radix
        self.groups = groups
        self.channels = channels
        self.with_dcn = dcn is not None
        self.dcn = dcn
        fallback_on_stride = False
        if self.with_dcn:
            fallback_on_stride = self.dcn.pop('fallback_on_stride', False)
        if self.with_dcn and not fallback_on_stride:
            assert conv_cfg is None, 'conv_cfg must be None for DCN'
            conv_cfg = dcn
        self.conv = build_conv_layer(conv_cfg, in_channels, channels * radix, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups * radix, bias=False)
        self.norm0_name, norm0 = build_norm_layer(norm_cfg, channels * radix, postfix=0)
        self.add_module(self.norm0_name, norm0)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = build_conv_layer(None, channels, inter_channels, 1, groups=self.groups)
        self.norm1_name, norm1 = build_norm_layer(norm_cfg, inter_channels, postfix=1)
        self.add_module(self.norm1_name, norm1)
        self.fc2 = build_conv_layer(None, inter_channels, channels * radix, 1, groups=self.groups)
        self.rsoftmax = RSoftmax(radix, groups)

    @property
    def norm0(self):
        """nn.Module: the normalization layer named "norm0" """
        return getattr(self, self.norm0_name)

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm0(x)
        x = self.relu(x)
        batch, rchannel = x.shape[:2]
        batch = x.size(0)
        if self.radix > 1:
            splits = x.view(batch, self.radix, -1, *x.shape[2:])
            gap = splits.sum(dim=1)
        else:
            gap = x
        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)
        gap = self.norm1(gap)
        gap = self.relu(gap)
        atten = self.fc2(gap)
        atten = self.rsoftmax(atten).view(batch, -1, 1, 1)
        if self.radix > 1:
            attens = atten.view(batch, self.radix, -1, *atten.shape[2:])
            out = torch.sum(attens * splits, dim=1)
        else:
            out = atten * x
        return out.contiguous()


class ResNetV1c(ResNet):
    """ResNetV1c variant described in [1]_.

    Compared with default ResNet(ResNetV1b), ResNetV1c replaces the 7x7 conv in
    the input stem with three 3x3 convs. For more details please refer to `Bag
    of Tricks for Image Classification with Convolutional Neural Networks
    <https://arxiv.org/abs/1812.01187>`_.
    """

    def __init__(self, **kwargs):
        super(ResNetV1c, self).__init__(deep_stem=True, avg_down=False, **kwargs)


class ResNetV1d(ResNet):
    """ResNetV1d variant described in [1]_.

    Compared with default ResNet(ResNetV1b), ResNetV1d replaces the 7x7 conv in
    the input stem with three 3x3 convs. And in the downsampling block, a 2x2
    avg_pool with stride 2 is added before conv, whose stride is changed to 1.
    """

    def __init__(self, **kwargs):
        super(ResNetV1d, self).__init__(deep_stem=True, avg_down=True, **kwargs)


class ResNeXt(ResNet):
    """ResNeXt backbone.

    This backbone is the implementation of `Aggregated
    Residual Transformations for Deep Neural
    Networks <https://arxiv.org/abs/1611.05431>`_.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        in_channels (int): Number of input image channels. Normally 3.
        num_stages (int): Resnet stages, normally 4.
        groups (int): Group of resnext.
        base_width (int): Base width of resnext.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means
            not freezing any parameters.
        norm_cfg (dict): dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): whether to use zero init for last norm layer
            in resblocks to let them behave as identity.

    Example:
        >>> from mmseg.models import ResNeXt
        >>> import torch
        >>> self = ResNeXt(depth=50)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 32, 32)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 256, 8, 8)
        (1, 512, 4, 4)
        (1, 1024, 2, 2)
        (1, 2048, 1, 1)
    """
    arch_settings = {(50): (Bottleneck, (3, 4, 6, 3)), (101): (Bottleneck, (3, 4, 23, 3)), (152): (Bottleneck, (3, 8, 36, 3))}

    def __init__(self, groups=1, base_width=4, **kwargs):
        self.groups = groups
        self.base_width = base_width
        super(ResNeXt, self).__init__(**kwargs)

    def make_res_layer(self, **kwargs):
        """Pack all blocks in a stage into a ``ResLayer``"""
        return ResLayer(groups=self.groups, base_width=self.base_width, base_channels=self.base_channels, **kwargs)


class BasicConvBlock(nn.Module):
    """Basic convolutional block for UNet.

    This module consists of several plain convolutional layers.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        num_convs (int): Number of convolutional layers. Default: 2.
        stride (int): Whether use stride convolution to downsample
            the input feature map. If stride=2, it only uses stride convolution
            in the first convolutional layer to downsample the input feature
            map. Options are 1 or 2. Default: 1.
        dilation (int): Whether use dilated convolution to expand the
            receptive field. Set dilation rate of each convolutional layer and
            the dilation rate of the first convolutional layer is always 1.
            Default: 1.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        conv_cfg (dict | None): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict | None): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict | None): Config dict for activation layer in ConvModule.
            Default: dict(type='ReLU').
        dcn (bool): Use deformable convolution in convolutional layer or not.
            Default: None.
        plugins (dict): plugins for convolutional layers. Default: None.
    """

    def __init__(self, in_channels, out_channels, num_convs=2, stride=1, dilation=1, with_cp=False, conv_cfg=None, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU'), dcn=None, plugins=None):
        super(BasicConvBlock, self).__init__()
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'
        self.with_cp = with_cp
        convs = []
        for i in range(num_convs):
            convs.append(ConvModule(in_channels=in_channels if i == 0 else out_channels, out_channels=out_channels, kernel_size=3, stride=stride if i == 0 else 1, dilation=1 if i == 0 else dilation, padding=1 if i == 0 else dilation, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg))
        self.convs = nn.Sequential(*convs)

    def forward(self, x):
        """Forward function."""
        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(self.convs, x)
        else:
            out = self.convs(x)
        return out


class DeconvModule(nn.Module):
    """Deconvolution upsample module in decoder for UNet (2X upsample).

    This module uses deconvolution to upsample feature map in the decoder
    of UNet.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        norm_cfg (dict | None): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict | None): Config dict for activation layer in ConvModule.
            Default: dict(type='ReLU').
        kernel_size (int): Kernel size of the convolutional layer. Default: 4.
    """

    def __init__(self, in_channels, out_channels, with_cp=False, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU'), *, kernel_size=4, scale_factor=2):
        super(DeconvModule, self).__init__()
        assert kernel_size - scale_factor >= 0 and (kernel_size - scale_factor) % 2 == 0, f'kernel_size should be greater than or equal to scale_factor and (kernel_size - scale_factor) should be even numbers, while the kernel size is {kernel_size} and scale_factor is {scale_factor}.'
        stride = scale_factor
        padding = (kernel_size - scale_factor) // 2
        self.with_cp = with_cp
        deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        norm_name, norm = build_norm_layer(norm_cfg, out_channels)
        activate = build_activation_layer(act_cfg)
        self.deconv_upsamping = nn.Sequential(deconv, norm, activate)

    def forward(self, x):
        """Forward function."""
        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(self.deconv_upsamping, x)
        else:
            out = self.deconv_upsamping(x)
        return out


class Upsample(nn.Module):

    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None):
        super(Upsample, self).__init__()
        self.size = size
        if isinstance(scale_factor, tuple):
            self.scale_factor = tuple(float(factor) for factor in scale_factor)
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        if not self.size:
            size = [int(t * self.scale_factor) for t in x.shape[-2:]]
        else:
            size = self.size
        return resize(x, size, None, self.mode, self.align_corners)


class InterpConv(nn.Module):
    """Interpolation upsample module in decoder for UNet.

    This module uses interpolation to upsample feature map in the decoder
    of UNet. It consists of one interpolation upsample layer and one
    convolutional layer. It can be one interpolation upsample layer followed
    by one convolutional layer (conv_first=False) or one convolutional layer
    followed by one interpolation upsample layer (conv_first=True).

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        norm_cfg (dict | None): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict | None): Config dict for activation layer in ConvModule.
            Default: dict(type='ReLU').
        conv_cfg (dict | None): Config dict for convolution layer.
            Default: None.
        conv_first (bool): Whether convolutional layer or interpolation
            upsample layer first. Default: False. It means interpolation
            upsample layer followed by one convolutional layer.
        kernel_size (int): Kernel size of the convolutional layer. Default: 1.
        stride (int): Stride of the convolutional layer. Default: 1.
        padding (int): Padding of the convolutional layer. Default: 1.
        upsample_cfg (dict): Interpolation config of the upsample layer.
            Default: dict(
                scale_factor=2, mode='bilinear', align_corners=False).
    """

    def __init__(self, in_channels, out_channels, with_cp=False, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU'), *, conv_cfg=None, conv_first=False, kernel_size=1, stride=1, padding=0, upsample_cfg=dict(scale_factor=2, mode='bilinear', align_corners=False)):
        super(InterpConv, self).__init__()
        self.with_cp = with_cp
        conv = ConvModule(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        upsample = Upsample(**upsample_cfg)
        if conv_first:
            self.interp_upsample = nn.Sequential(conv, upsample)
        else:
            self.interp_upsample = nn.Sequential(upsample, conv)

    def forward(self, x):
        """Forward function."""
        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(self.interp_upsample, x)
        else:
            out = self.interp_upsample(x)
        return out


class PPMConcat(nn.ModuleList):
    """Pyramid Pooling Module that only concat the features of each layer.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module.
    """

    def __init__(self, pool_scales=(1, 3, 6, 8)):
        super(PPMConcat, self).__init__([nn.AdaptiveAvgPool2d(pool_scale) for pool_scale in pool_scales])

    def forward(self, feats):
        """Forward function."""
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(feats)
            ppm_outs.append(ppm_out.view(*feats.shape[:2], -1))
        concat_outs = torch.cat(ppm_outs, dim=2)
        return concat_outs


class SelfAttentionBlock(nn.Module):
    """General self-attention block/non-local block.

    Please refer to https://arxiv.org/abs/1706.03762 for details about key,
    query and value.

    Args:
        key_in_channels (int): Input channels of key feature.
        query_in_channels (int): Input channels of query feature.
        channels (int): Output channels of key/query transform.
        out_channels (int): Output channels.
        share_key_query (bool): Whether share projection weight between key
            and query projection.
        query_downsample (nn.Module): Query downsample module.
        key_downsample (nn.Module): Key downsample module.
        key_query_num_convs (int): Number of convs for key/query projection.
        value_num_convs (int): Number of convs for value projection.
        matmul_norm (bool): Whether normalize attention map with sqrt of
            channels
        with_out (bool): Whether use out projection.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict|None): Config of activation layers.
    """

    def __init__(self, key_in_channels, query_in_channels, channels, out_channels, share_key_query, query_downsample, key_downsample, key_query_num_convs, value_out_num_convs, key_query_norm, value_out_norm, matmul_norm, with_out, conv_cfg, norm_cfg, act_cfg):
        super(SelfAttentionBlock, self).__init__()
        if share_key_query:
            assert key_in_channels == query_in_channels
        self.key_in_channels = key_in_channels
        self.query_in_channels = query_in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.share_key_query = share_key_query
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.key_project = self.build_project(key_in_channels, channels, num_convs=key_query_num_convs, use_conv_module=key_query_norm, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        if share_key_query:
            self.query_project = self.key_project
        else:
            self.query_project = self.build_project(query_in_channels, channels, num_convs=key_query_num_convs, use_conv_module=key_query_norm, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.value_project = self.build_project(key_in_channels, channels if with_out else out_channels, num_convs=value_out_num_convs, use_conv_module=value_out_norm, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        if with_out:
            self.out_project = self.build_project(channels, out_channels, num_convs=value_out_num_convs, use_conv_module=value_out_norm, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        else:
            self.out_project = None
        self.query_downsample = query_downsample
        self.key_downsample = key_downsample
        self.matmul_norm = matmul_norm
        self.init_weights()

    def init_weights(self):
        """Initialize weight of later layer."""
        if self.out_project is not None:
            if not isinstance(self.out_project, ConvModule):
                constant_init(self.out_project, 0)

    def build_project(self, in_channels, channels, num_convs, use_conv_module, conv_cfg, norm_cfg, act_cfg):
        """Build projection layer for key/query/value/out."""
        if use_conv_module:
            convs = [ConvModule(in_channels, channels, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)]
            for _ in range(num_convs - 1):
                convs.append(ConvModule(channels, channels, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg))
        else:
            convs = [nn.Conv2d(in_channels, channels, 1)]
            for _ in range(num_convs - 1):
                convs.append(nn.Conv2d(channels, channels, 1))
        if len(convs) > 1:
            convs = nn.Sequential(*convs)
        else:
            convs = convs[0]
        return convs

    def forward(self, query_feats, key_feats):
        """Forward function."""
        batch_size = query_feats.size(0)
        query = self.query_project(query_feats)
        if self.query_downsample is not None:
            query = self.query_downsample(query)
        query = query.reshape(*query.shape[:2], -1)
        query = query.permute(0, 2, 1).contiguous()
        key = self.key_project(key_feats)
        value = self.value_project(key_feats)
        if self.key_downsample is not None:
            key = self.key_downsample(key)
            value = self.key_downsample(value)
        key = key.reshape(*key.shape[:2], -1)
        value = value.reshape(*value.shape[:2], -1)
        value = value.permute(0, 2, 1).contiguous()
        sim_map = torch.matmul(query, key)
        if self.matmul_norm:
            sim_map = self.channels ** -0.5 * sim_map
        sim_map = F.softmax(sim_map, dim=-1)
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.reshape(batch_size, -1, *query_feats.shape[2:])
        if self.out_project is not None:
            context = self.out_project(context)
        return context


class AFNB(nn.Module):
    """Asymmetric Fusion Non-local Block(AFNB)

    Args:
        low_in_channels (int): Input channels of lower level feature,
            which is the key feature for self-attention.
        high_in_channels (int): Input channels of higher level feature,
            which is the query feature for self-attention.
        channels (int): Output channels of key/query transform.
        out_channels (int): Output channels.
            and query projection.
        query_scales (tuple[int]): The scales of query feature map.
            Default: (1,)
        key_pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module of key feature.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict|None): Config of activation layers.
    """

    def __init__(self, low_in_channels, high_in_channels, channels, out_channels, query_scales, key_pool_scales, conv_cfg, norm_cfg, act_cfg):
        super(AFNB, self).__init__()
        self.stages = nn.ModuleList()
        for query_scale in query_scales:
            self.stages.append(SelfAttentionBlock(low_in_channels=low_in_channels, high_in_channels=high_in_channels, channels=channels, out_channels=out_channels, share_key_query=False, query_scale=query_scale, key_pool_scales=key_pool_scales, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg))
        self.bottleneck = ConvModule(out_channels + high_in_channels, out_channels, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=None)

    def forward(self, low_feats, high_feats):
        """Forward function."""
        priors = [stage(high_feats, low_feats) for stage in self.stages]
        context = torch.stack(priors, dim=0).sum(dim=0)
        output = self.bottleneck(torch.cat([context, high_feats], 1))
        return output


class APNB(nn.Module):
    """Asymmetric Pyramid Non-local Block (APNB)

    Args:
        in_channels (int): Input channels of key/query feature,
            which is the key feature for self-attention.
        channels (int): Output channels of key/query transform.
        out_channels (int): Output channels.
        query_scales (tuple[int]): The scales of query feature map.
            Default: (1,)
        key_pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module of key feature.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict|None): Config of activation layers.
    """

    def __init__(self, in_channels, channels, out_channels, query_scales, key_pool_scales, conv_cfg, norm_cfg, act_cfg):
        super(APNB, self).__init__()
        self.stages = nn.ModuleList()
        for query_scale in query_scales:
            self.stages.append(SelfAttentionBlock(low_in_channels=in_channels, high_in_channels=in_channels, channels=channels, out_channels=out_channels, share_key_query=True, query_scale=query_scale, key_pool_scales=key_pool_scales, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg))
        self.bottleneck = ConvModule(2 * in_channels, out_channels, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, feats):
        """Forward function."""
        priors = [stage(feats, feats) for stage in self.stages]
        context = torch.stack(priors, dim=0).sum(dim=0)
        output = self.bottleneck(torch.cat([context, feats], 1))
        return output


class ACM(nn.Module):
    """Adaptive Context Module used in APCNet.

    Args:
        pool_scale (int): Pooling scale used in Adaptive Context
            Module to extract region features.
        fusion (bool): Add one conv to fuse residual feature.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        conv_cfg (dict | None): Config of conv layers.
        norm_cfg (dict | None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
    """

    def __init__(self, pool_scale, fusion, in_channels, channels, conv_cfg, norm_cfg, act_cfg):
        super(ACM, self).__init__()
        self.pool_scale = pool_scale
        self.fusion = fusion
        self.in_channels = in_channels
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.pooled_redu_conv = ConvModule(self.in_channels, self.channels, 1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        self.input_redu_conv = ConvModule(self.in_channels, self.channels, 1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        self.global_info = ConvModule(self.channels, self.channels, 1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        self.gla = nn.Conv2d(self.channels, self.pool_scale ** 2, 1, 1, 0)
        self.residual_conv = ConvModule(self.channels, self.channels, 1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        if self.fusion:
            self.fusion_conv = ConvModule(self.channels, self.channels, 1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)

    def forward(self, x):
        """Forward function."""
        pooled_x = F.adaptive_avg_pool2d(x, self.pool_scale)
        x = self.input_redu_conv(x)
        pooled_x = self.pooled_redu_conv(pooled_x)
        batch_size = x.size(0)
        pooled_x = pooled_x.view(batch_size, self.channels, -1).permute(0, 2, 1).contiguous()
        affinity_matrix = self.gla(x + resize(self.global_info(F.adaptive_avg_pool2d(x, 1)), size=x.shape[2:])).permute(0, 2, 3, 1).reshape(batch_size, -1, self.pool_scale ** 2)
        affinity_matrix = F.sigmoid(affinity_matrix)
        z_out = torch.matmul(affinity_matrix, pooled_x)
        z_out = z_out.permute(0, 2, 1).contiguous()
        z_out = z_out.view(batch_size, self.channels, x.size(2), x.size(3))
        z_out = self.residual_conv(z_out)
        z_out = F.relu(z_out + x)
        if self.fusion:
            z_out = self.fusion_conv(z_out)
        return z_out


class ASPPModule(nn.ModuleList):
    """Atrous Spatial Pyramid Pooling (ASPP) Module.

    Args:
        dilations (tuple[int]): Dilation rate of each layer.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
    """

    def __init__(self, dilations, in_channels, channels, conv_cfg, norm_cfg, act_cfg):
        super(ASPPModule, self).__init__()
        self.dilations = dilations
        self.in_channels = in_channels
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        for dilation in dilations:
            self.append(ConvModule(self.in_channels, self.channels, 1 if dilation == 1 else 3, dilation=dilation, padding=0 if dilation == 1 else dilation, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg))

    def forward(self, x):
        """Forward function."""
        aspp_outs = []
        for aspp_module in self:
            aspp_outs.append(aspp_module(x))
        return aspp_outs


class CAM(nn.Module):
    """Channel Attention Module (CAM)"""

    def __init__(self):
        super(CAM, self).__init__()
        self.gamma = Scale(0)

    def forward(self, x):
        """Forward function."""
        batch_size, channels, height, width = x.size()
        proj_query = x.view(batch_size, channels, -1)
        proj_key = x.view(batch_size, channels, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = F.softmax(energy_new, dim=-1)
        proj_value = x.view(batch_size, channels, -1)
        out = torch.bmm(attention, proj_value)
        out = out.view(batch_size, channels, height, width)
        out = self.gamma(out) + x
        return out


class DCM(nn.Module):
    """Dynamic Convolutional Module used in DMNet.

    Args:
        filter_size (int): The filter size of generated convolution kernel
            used in Dynamic Convolutional Module.
        fusion (bool): Add one conv to fuse DCM output feature.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        conv_cfg (dict | None): Config of conv layers.
        norm_cfg (dict | None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
    """

    def __init__(self, filter_size, fusion, in_channels, channels, conv_cfg, norm_cfg, act_cfg):
        super(DCM, self).__init__()
        self.filter_size = filter_size
        self.fusion = fusion
        self.in_channels = in_channels
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.filter_gen_conv = nn.Conv2d(self.in_channels, self.channels, 1, 1, 0)
        self.input_redu_conv = ConvModule(self.in_channels, self.channels, 1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        if self.norm_cfg is not None:
            self.norm = build_norm_layer(self.norm_cfg, self.channels)[1]
        else:
            self.norm = None
        self.activate = build_activation_layer(self.act_cfg)
        if self.fusion:
            self.fusion_conv = ConvModule(self.channels, self.channels, 1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)

    def forward(self, x):
        """Forward function."""
        generated_filter = self.filter_gen_conv(F.adaptive_avg_pool2d(x, self.filter_size))
        x = self.input_redu_conv(x)
        b, c, h, w = x.shape
        x = x.view(1, b * c, h, w)
        generated_filter = generated_filter.view(b * c, 1, self.filter_size, self.filter_size)
        pad = (self.filter_size - 1) // 2
        if (self.filter_size - 1) % 2 == 0:
            p2d = pad, pad, pad, pad
        else:
            p2d = pad + 1, pad, pad + 1, pad
        x = F.pad(input=x, pad=p2d, mode='constant', value=0)
        output = F.conv2d(input=x, weight=generated_filter, groups=b * c)
        output = output.view(b, c, h, w)
        if self.norm is not None:
            output = self.norm(output)
        output = self.activate(output)
        if self.fusion:
            output = self.fusion_conv(output)
        return output


def reduce_mean(tensor):
    """Reduce mean when distributed training."""
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor.div_(dist.get_world_size()), op=dist.ReduceOp.SUM)
    return tensor


class EMAModule(nn.Module):
    """Expectation Maximization Attention Module used in EMANet.

    Args:
        channels (int): Channels of the whole module.
        num_bases (int): Number of bases.
        num_stages (int): Number of the EM iterations.
    """

    def __init__(self, channels, num_bases, num_stages, momentum):
        super(EMAModule, self).__init__()
        assert num_stages >= 1, 'num_stages must be at least 1!'
        self.num_bases = num_bases
        self.num_stages = num_stages
        self.momentum = momentum
        bases = torch.zeros(1, channels, self.num_bases)
        bases.normal_(0, math.sqrt(2.0 / self.num_bases))
        bases = F.normalize(bases, dim=1, p=2)
        self.register_buffer('bases', bases)

    def forward(self, feats):
        """Forward function."""
        batch_size, channels, height, width = feats.size()
        feats = feats.view(batch_size, channels, height * width)
        bases = self.bases.repeat(batch_size, 1, 1)
        with torch.no_grad():
            for i in range(self.num_stages):
                attention = torch.einsum('bcn,bck->bnk', feats, bases)
                attention = F.softmax(attention, dim=2)
                attention_normed = F.normalize(attention, dim=1, p=1)
                bases = torch.einsum('bcn,bnk->bck', feats, attention_normed)
                bases = F.normalize(bases, dim=1, p=2)
        feats_recon = torch.einsum('bck,bnk->bcn', bases, attention)
        feats_recon = feats_recon.view(batch_size, channels, height, width)
        if self.training:
            bases = bases.mean(dim=0, keepdim=True)
            bases = reduce_mean(bases)
            bases = F.normalize(bases, dim=1, p=2)
            self.bases = (1 - self.momentum) * self.bases + self.momentum * bases
        return feats_recon


class Encoding(nn.Module):
    """Encoding Layer: a learnable residual encoder.

    Input is of shape  (batch_size, channels, height, width).
    Output is of shape (batch_size, num_codes, channels).

    Args:
        channels: dimension of the features or feature channels
        num_codes: number of code words
    """

    def __init__(self, channels, num_codes):
        super(Encoding, self).__init__()
        self.channels, self.num_codes = channels, num_codes
        std = 1.0 / (num_codes * channels) ** 0.5
        self.codewords = nn.Parameter(torch.empty(num_codes, channels, dtype=torch.float).uniform_(-std, std), requires_grad=True)
        self.scale = nn.Parameter(torch.empty(num_codes, dtype=torch.float).uniform_(-1, 0), requires_grad=True)

    @staticmethod
    def scaled_l2(x, codewords, scale):
        num_codes, channels = codewords.size()
        batch_size = x.size(0)
        reshaped_scale = scale.view((1, 1, num_codes))
        expanded_x = x.unsqueeze(2).expand((batch_size, x.size(1), num_codes, channels))
        reshaped_codewords = codewords.view((1, 1, num_codes, channels))
        scaled_l2_norm = reshaped_scale * (expanded_x - reshaped_codewords).pow(2).sum(dim=3)
        return scaled_l2_norm

    @staticmethod
    def aggregate(assignment_weights, x, codewords):
        num_codes, channels = codewords.size()
        reshaped_codewords = codewords.view((1, 1, num_codes, channels))
        batch_size = x.size(0)
        expanded_x = x.unsqueeze(2).expand((batch_size, x.size(1), num_codes, channels))
        encoded_feat = (assignment_weights.unsqueeze(3) * (expanded_x - reshaped_codewords)).sum(dim=1)
        return encoded_feat

    def forward(self, x):
        assert x.dim() == 4 and x.size(1) == self.channels
        batch_size = x.size(0)
        x = x.view(batch_size, self.channels, -1).transpose(1, 2).contiguous()
        assignment_weights = F.softmax(self.scaled_l2(x, self.codewords, self.scale), dim=2)
        encoded_feat = self.aggregate(assignment_weights, x, self.codewords)
        return encoded_feat

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(Nx{self.channels}xHxW =>Nx{self.num_codes}x{self.channels})'
        return repr_str


class EncModule(nn.Module):
    """Encoding Module used in EncNet.

    Args:
        in_channels (int): Input channels.
        num_codes (int): Number of code words.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
    """

    def __init__(self, in_channels, num_codes, conv_cfg, norm_cfg, act_cfg):
        super(EncModule, self).__init__()
        self.encoding_project = ConvModule(in_channels, in_channels, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        if norm_cfg is not None:
            encoding_norm_cfg = norm_cfg.copy()
            if encoding_norm_cfg['type'] in ['BN', 'IN']:
                encoding_norm_cfg['type'] += '1d'
            else:
                encoding_norm_cfg['type'] = encoding_norm_cfg['type'].replace('2d', '1d')
        else:
            encoding_norm_cfg = dict(type='BN1d')
        self.encoding = nn.Sequential(Encoding(channels=in_channels, num_codes=num_codes), build_norm_layer(encoding_norm_cfg, num_codes)[1], nn.ReLU(inplace=True))
        self.fc = nn.Sequential(nn.Linear(in_channels, in_channels), nn.Sigmoid())

    def forward(self, x):
        """Forward function."""
        encoding_projection = self.encoding_project(x)
        encoding_feat = self.encoding(encoding_projection).mean(dim=1)
        batch_size, channels, _, _ = x.size()
        gamma = self.fc(encoding_feat)
        y = gamma.view(batch_size, channels, 1, 1)
        output = F.relu_(x + x * y)
        return encoding_feat, output


class FocalLoss(nn.Module):

    def __init__(self, gamma=2.0, weight=None, size_average=True, num_classes=150, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def forward(self, input, target):
        mask = target != self.ignore_index
        gt = torch.where(mask, target, self.num_classes)
        one_hot_gt = F.one_hot(gt, num_classes=self.num_classes + 1)[:, :, :, :-1].permute(0, 3, 1, 2).contiguous()
        score = F.softmax(input, dim=1)
        score_gt = (one_hot_gt * score).sum(1)
        loss = -(1 - score_gt) ** self.gamma * (one_hot_gt * torch.log(score + 1e-12)).sum(1)
        loss = (loss * mask.float()).sum() / (mask.float().sum() + 1e-12)
        return loss


class StableBCELoss(torch.nn.modules.Module):

    def __init__(self):
        super(StableBCELoss, self).__init__()

    def forward(self, input, target):
        neg_abs = -input.abs()
        loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
        return loss.mean()


def get_class_weight(class_weight):
    """Get class weight for loss function.

    Args:
        class_weight (list[float] | str | None): If class_weight is a str,
            take it as a file name and read from it.
    """
    if isinstance(class_weight, str):
        if class_weight.endswith('.npy'):
            class_weight = np.load(class_weight)
        else:
            class_weight = mmcv.load(class_weight)
    return class_weight


def flatten_binary_logits(logits, labels, ignore_index=None):
    """Flattens predictions in the batch (binary case) Remove labels equal to
    'ignore_index'."""
    logits = logits.view(-1)
    labels = labels.view(-1)
    if ignore_index is None:
        return logits, labels
    valid = labels != ignore_index
    vlogits = logits[valid]
    vlabels = labels[valid]
    return vlogits, vlabels


def lovasz_grad(gt_sorted):
    """Computes gradient of the Lovasz extension w.r.t sorted errors.

    See Alg. 1 in paper.
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_hinge_flat(logits, labels):
    """Binary Lovasz hinge loss.

    Args:
        logits (torch.Tensor): [P], logits at each prediction
            (between -infty and +infty).
        labels (torch.Tensor): [P], binary ground truth labels (0 or 1).

    Returns:
        torch.Tensor: The calculated loss.
    """
    if len(labels) == 0:
        return logits.sum() * 0.0
    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss


def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Avarage factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    if weight is not None:
        assert weight.dim() == loss.dim()
        if weight.dim() > 1:
            assert weight.size(1) == 1 or weight.size(1) == loss.size(1)
        loss = loss * weight
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    elif reduction == 'mean':
        loss = loss.sum() / avg_factor
    elif reduction != 'none':
        raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def lovasz_hinge(logits, labels, classes='present', per_image=False, class_weight=None, reduction='mean', avg_factor=None, ignore_index=255):
    """Binary Lovasz hinge loss.

    Args:
        logits (torch.Tensor): [B, H, W], logits at each pixel
            (between -infty and +infty).
        labels (torch.Tensor): [B, H, W], binary ground truth masks (0 or 1).
        classes (str | list[int], optional): Placeholder, to be consistent with
            other loss. Default: None.
        per_image (bool, optional): If per_image is True, compute the loss per
            image instead of per batch. Default: False.
        class_weight (list[float], optional): Placeholder, to be consistent
            with other loss. Default: None.
        reduction (str, optional): The method used to reduce the loss. Options
            are "none", "mean" and "sum". This parameter only works when
            per_image is True. Default: 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. This parameter only works when per_image is True.
            Default: None.
        ignore_index (int | None): The label index to be ignored. Default: 255.

    Returns:
        torch.Tensor: The calculated loss.
    """
    if per_image:
        loss = [lovasz_hinge_flat(*flatten_binary_logits(logit.unsqueeze(0), label.unsqueeze(0), ignore_index)) for logit, label in zip(logits, labels)]
        loss = weight_reduce_loss(torch.stack(loss), None, reduction, avg_factor)
    else:
        loss = lovasz_hinge_flat(*flatten_binary_logits(logits, labels, ignore_index))
    return loss


def flatten_probs(probs, labels, ignore_index=None):
    """Flattens predictions in the batch."""
    if probs.dim() == 3:
        B, H, W = probs.size()
        probs = probs.view(B, 1, H, W)
    B, C, H, W = probs.size()
    probs = probs.permute(0, 2, 3, 1).contiguous().view(-1, C)
    labels = labels.view(-1)
    if ignore_index is None:
        return probs, labels
    valid = labels != ignore_index
    vprobs = probs[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobs, vlabels


def lovasz_softmax_flat(probs, labels, classes='present', class_weight=None):
    """Multi-class Lovasz-Softmax loss.

    Args:
        probs (torch.Tensor): [P, C], class probabilities at each prediction
            (between 0 and 1).
        labels (torch.Tensor): [P], ground truth labels (between 0 and C - 1).
        classes (str | list[int], optional): Classes chosen to calculate loss.
            'all' for all classes, 'present' for classes present in labels, or
            a list of classes to average. Default: 'present'.
        class_weight (list[float], optional): The weight for each class.
            Default: None.

    Returns:
        torch.Tensor: The calculated loss.
    """
    if probs.numel() == 0:
        return probs * 0.0
    C = probs.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float()
        if classes == 'present' and fg.sum() == 0:
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probs[:, 0]
        else:
            class_pred = probs[:, c]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        loss = torch.dot(errors_sorted, lovasz_grad(fg_sorted))
        if class_weight is not None:
            loss *= class_weight[c]
        losses.append(loss)
    return torch.stack(losses).mean()


def lovasz_softmax(probs, labels, classes='present', per_image=False, class_weight=None, reduction='mean', avg_factor=None, ignore_index=255):
    """Multi-class Lovasz-Softmax loss.

    Args:
        probs (torch.Tensor): [B, C, H, W], class probabilities at each
            prediction (between 0 and 1).
        labels (torch.Tensor): [B, H, W], ground truth labels (between 0 and
            C - 1).
        classes (str | list[int], optional): Classes chosen to calculate loss.
            'all' for all classes, 'present' for classes present in labels, or
            a list of classes to average. Default: 'present'.
        per_image (bool, optional): If per_image is True, compute the loss per
            image instead of per batch. Default: False.
        class_weight (list[float], optional): The weight for each class.
            Default: None.
        reduction (str, optional): The method used to reduce the loss. Options
            are "none", "mean" and "sum". This parameter only works when
            per_image is True. Default: 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. This parameter only works when per_image is True.
            Default: None.
        ignore_index (int | None): The label index to be ignored. Default: 255.

    Returns:
        torch.Tensor: The calculated loss.
    """
    if per_image:
        loss = [lovasz_softmax_flat(*flatten_probs(prob.unsqueeze(0), label.unsqueeze(0), ignore_index), classes=classes, class_weight=class_weight) for prob, label in zip(probs, labels)]
        loss = weight_reduce_loss(torch.stack(loss), None, reduction, avg_factor)
    else:
        loss = lovasz_softmax_flat(*flatten_probs(probs, labels, ignore_index), classes=classes, class_weight=class_weight)
    return loss


class LovaszLoss(nn.Module):
    """LovaszLoss.

    This loss is proposed in `The Lovasz-Softmax loss: A tractable surrogate
    for the optimization of the intersection-over-union measure in neural
    networks <https://arxiv.org/abs/1705.08790>`_.

    Args:
        loss_type (str, optional): Binary or multi-class loss.
            Default: 'multi_class'. Options are "binary" and "multi_class".
        classes (str | list[int], optional): Classes chosen to calculate loss.
            'all' for all classes, 'present' for classes present in labels, or
            a list of classes to average. Default: 'present'.
        per_image (bool, optional): If per_image is True, compute the loss per
            image instead of per batch. Default: False.
        reduction (str, optional): The method used to reduce the loss. Options
            are "none", "mean" and "sum". This parameter only works when
            per_image is True. Default: 'mean'.
        class_weight (list[float] | str, optional): Weight of each class. If in
            str format, read them from a file. Defaults to None.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_lovasz'.
    """

    def __init__(self, loss_type='multi_class', classes='present', per_image=False, reduction='mean', class_weight=None, loss_weight=1.0, loss_name='loss_lovasz'):
        super(LovaszLoss, self).__init__()
        assert loss_type in ('binary', 'multi_class'), "loss_type should be                                                     'binary' or 'multi_class'."
        if loss_type == 'binary':
            self.cls_criterion = lovasz_hinge
        else:
            self.cls_criterion = lovasz_softmax
        assert classes in ('all', 'present') or mmcv.is_list_of(classes, int)
        if not per_image:
            assert reduction == 'none', "reduction should be 'none' when                                                         per_image is False."
        self.classes = classes
        self.per_image = per_image
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = get_class_weight(class_weight)
        self._loss_name = loss_name

    def forward(self, cls_score, label, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        """Forward function."""
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None
        if self.cls_criterion == lovasz_softmax:
            cls_score = F.softmax(cls_score, dim=1)
        loss_cls = self.loss_weight * self.cls_criterion(cls_score, label, self.classes, self.per_image, class_weight=class_weight, reduction=reduction, avg_factor=avg_factor, **kwargs)
        return loss_cls

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name


class SpatialGatherModule(nn.Module):
    """Aggregate the context features according to the initial predicted
    probability distribution.
    Employ the soft-weighted method to aggregate the context.
    """

    def __init__(self, scale):
        super(SpatialGatherModule, self).__init__()
        self.scale = scale

    def forward(self, feats, probs):
        """Forward function."""
        batch_size, num_classes, height, width = probs.size()
        channels = feats.size(1)
        probs = probs.view(batch_size, num_classes, -1)
        feats = feats.view(batch_size, channels, -1)
        feats = feats.permute(0, 2, 1)
        probs = F.softmax(self.scale * probs, dim=2)
        ocr_context = torch.matmul(probs, feats)
        ocr_context = ocr_context.permute(0, 2, 1).contiguous().unsqueeze(3)
        return ocr_context


class DepthwiseSeparableASPPModule(ASPPModule):
    """Atrous Spatial Pyramid Pooling (ASPP) Module with depthwise separable
    conv."""

    def __init__(self, **kwargs):
        super(DepthwiseSeparableASPPModule, self).__init__(**kwargs)
        for i, dilation in enumerate(self.dilations):
            if dilation > 1:
                self[i] = DepthwiseSeparableConvModule(self.in_channels, self.channels, 3, dilation=dilation, padding=dilation, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred)).contiguous()
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class Accuracy(nn.Module):
    """Accuracy calculation module."""

    def __init__(self, topk=(1,), thresh=None):
        """Module to calculate the accuracy.

        Args:
            topk (tuple, optional): The criterion used to calculate the
                accuracy. Defaults to (1,).
            thresh (float, optional): If not None, predictions with scores
                under this threshold are considered incorrect. Default to None.
        """
        super().__init__()
        self.topk = topk
        self.thresh = thresh

    def forward(self, pred, target):
        """Forward function to calculate accuracy.

        Args:
            pred (torch.Tensor): Prediction of models.
            target (torch.Tensor): Target for each prediction.

        Returns:
            tuple[float]: The accuracies under different topk criterions.
        """
        return accuracy(pred, target, self.topk, self.thresh)


def _expand_onehot_labels(labels, label_weights, target_shape, ignore_index):
    """Expand onehot labels to match the size of prediction."""
    bin_labels = labels.new_zeros(target_shape)
    valid_mask = (labels >= 0) & (labels != ignore_index)
    inds = torch.nonzero(valid_mask, as_tuple=True)
    if inds[0].numel() > 0:
        if labels.dim() == 3:
            bin_labels[inds[0], labels[valid_mask], inds[1], inds[2]] = 1
        else:
            bin_labels[inds[0], labels[valid_mask]] = 1
    valid_mask = valid_mask.unsqueeze(1).expand(target_shape).float()
    if label_weights is None:
        bin_label_weights = valid_mask
    else:
        bin_label_weights = label_weights.unsqueeze(1).expand(target_shape)
        bin_label_weights *= valid_mask
    return bin_labels, bin_label_weights


def binary_cross_entropy(pred, label, weight=None, reduction='mean', avg_factor=None, class_weight=None, ignore_index=255):
    """Calculate the binary CrossEntropy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, 1).
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
        ignore_index (int | None): The label index to be ignored. Default: 255

    Returns:
        torch.Tensor: The calculated loss
    """
    if pred.dim() != label.dim():
        assert pred.dim() == 2 and label.dim() == 1 or pred.dim() == 4 and label.dim() == 3, 'Only pred shape [N, C], label shape [N] or pred shape [N, C, H, W], label shape [N, H, W] are supported'
        label, weight = _expand_onehot_labels(label, weight, pred.shape, ignore_index)
    if weight is not None:
        weight = weight.float()
    loss = F.binary_cross_entropy_with_logits(pred, label.float(), pos_weight=class_weight, reduction='none')
    loss = weight_reduce_loss(loss, weight, reduction=reduction, avg_factor=avg_factor)
    return loss


def cross_entropy(pred, label, weight=None, class_weight=None, reduction='mean', avg_factor=None, ignore_index=-100):
    """The wrapper function for :func:`F.cross_entropy`"""
    loss = F.cross_entropy(pred, label, weight=class_weight, reduction='none', ignore_index=ignore_index)
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
    return loss


def mask_cross_entropy(pred, target, label, reduction='mean', avg_factor=None, class_weight=None, ignore_index=None):
    """Calculate the CrossEntropy loss for masks.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        target (torch.Tensor): The learning label of the prediction.
        label (torch.Tensor): ``label`` indicates the class label of the mask'
            corresponding object. This will be used to select the mask in the
            of the class which the object belongs to when the mask prediction
            if not class-agnostic.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
        ignore_index (None): Placeholder, to be consistent with other loss.
            Default: None.

    Returns:
        torch.Tensor: The calculated loss
    """
    assert ignore_index is None, 'BCE loss does not support ignore_index'
    assert reduction == 'mean' and avg_factor is None
    num_rois = pred.size()[0]
    inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
    pred_slice = pred[inds, label].squeeze(1)
    return F.binary_cross_entropy_with_logits(pred_slice, target, weight=class_weight, reduction='mean')[None]


class CrossEntropyLoss(nn.Module):
    """CrossEntropyLoss.

    Args:
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        use_mask (bool, optional): Whether to use mask cross entropy loss.
            Defaults to False.
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        class_weight (list[float] | str, optional): Weight of each class. If in
            str format, read them from a file. Defaults to None.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_ce'.
    """

    def __init__(self, use_sigmoid=False, use_mask=False, reduction='mean', class_weight=None, loss_weight=1.0, loss_name='loss_ce'):
        super(CrossEntropyLoss, self).__init__()
        assert use_sigmoid is False or use_mask is False
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = get_class_weight(class_weight)
        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        elif self.use_mask:
            self.cls_criterion = mask_cross_entropy
        else:
            self.cls_criterion = cross_entropy
        self._loss_name = loss_name

    def forward(self, cls_score, label, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        """Forward function."""
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None
        loss_cls = self.loss_weight * self.cls_criterion(cls_score, label, weight, class_weight=class_weight, reduction=reduction, avg_factor=avg_factor, **kwargs)
        return loss_cls

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name


def weighted_loss(loss_func):
    """Create a weighted version of a given loss function.

    To use this decorator, the loss function must have the signature like
    `loss_func(pred, target, **kwargs)`. The function only needs to compute
    element-wise loss without any reduction. This decorator will add weight
    and reduction arguments to the function. The decorated function will have
    the signature like `loss_func(pred, target, weight=None, reduction='mean',
    avg_factor=None, **kwargs)`.

    :Example:

    >>> import torch
    >>> @weighted_loss
    >>> def l1_loss(pred, target):
    >>>     return (pred - target).abs()

    >>> pred = torch.Tensor([0, 2, 3])
    >>> target = torch.Tensor([1, 1, 1])
    >>> weight = torch.Tensor([1, 0, 1])

    >>> l1_loss(pred, target)
    tensor(1.3333)
    >>> l1_loss(pred, target, weight)
    tensor(1.)
    >>> l1_loss(pred, target, reduction='none')
    tensor([1., 1., 2.])
    >>> l1_loss(pred, target, weight, avg_factor=2)
    tensor(1.5000)
    """

    @functools.wraps(loss_func)
    def wrapper(pred, target, weight=None, reduction='mean', avg_factor=None, **kwargs):
        loss = loss_func(pred, target, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss
    return wrapper


@weighted_loss
def binary_dice_loss(pred, target, valid_mask, smooth=1, exponent=2, **kwards):
    assert pred.shape[0] == target.shape[0]
    pred = pred.reshape(pred.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    valid_mask = valid_mask.reshape(valid_mask.shape[0], -1)
    num = torch.sum(torch.mul(pred, target) * valid_mask, dim=1) * 2 + smooth
    den = torch.sum(pred.pow(exponent) + target.pow(exponent), dim=1) + smooth
    return 1 - num / den


@weighted_loss
def dice_loss(pred, target, valid_mask, smooth=1, exponent=2, class_weight=None, ignore_index=255):
    assert pred.shape[0] == target.shape[0]
    total_loss = 0
    num_classes = pred.shape[1]
    for i in range(num_classes):
        if i != ignore_index:
            dice_loss = binary_dice_loss(pred[:, i], target[..., i], valid_mask=valid_mask, smooth=smooth, exponent=exponent)
            if class_weight is not None:
                dice_loss *= class_weight[i]
            total_loss += dice_loss
    return total_loss / num_classes


class DiceLoss(nn.Module):
    """DiceLoss.

    This loss is proposed in `V-Net: Fully Convolutional Neural Networks for
    Volumetric Medical Image Segmentation <https://arxiv.org/abs/1606.04797>`_.

    Args:
        loss_type (str, optional): Binary or multi-class loss.
            Default: 'multi_class'. Options are "binary" and "multi_class".
        smooth (float): A float number to smooth loss, and avoid NaN error.
            Default: 1
        exponent (float): An float number to calculate denominator
            value: \\sum{x^exponent} + \\sum{y^exponent}. Default: 2.
        reduction (str, optional): The method used to reduce the loss. Options
            are "none", "mean" and "sum". This parameter only works when
            per_image is True. Default: 'mean'.
        class_weight (list[float] | str, optional): Weight of each class. If in
            str format, read them from a file. Defaults to None.
        loss_weight (float, optional): Weight of the loss. Default to 1.0.
        ignore_index (int | None): The label index to be ignored. Default: 255.
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_dice'.
    """

    def __init__(self, smooth=1, exponent=2, reduction='mean', class_weight=None, loss_weight=1.0, ignore_index=255, loss_name='loss_dice', **kwards):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.exponent = exponent
        self.reduction = reduction
        self.class_weight = get_class_weight(class_weight)
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self._loss_name = loss_name

    def forward(self, pred, target, avg_factor=None, reduction_override=None, **kwards):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        if self.class_weight is not None:
            class_weight = pred.new_tensor(self.class_weight)
        else:
            class_weight = None
        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        one_hot_target = F.one_hot(torch.clamp(target.long(), 0, num_classes - 1), num_classes=num_classes)
        valid_mask = (target != self.ignore_index).long()
        loss = self.loss_weight * dice_loss(pred, one_hot_target, valid_mask=valid_mask, reduction=reduction, avg_factor=avg_factor, smooth=self.smooth, exponent=self.exponent, class_weight=class_weight, ignore_index=self.ignore_index)
        return loss

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name


class MLAModule(nn.Module):

    def __init__(self, in_channels=[1024, 1024, 1024, 1024], out_channels=256, norm_cfg=None, act_cfg=None):
        super(MLAModule, self).__init__()
        self.channel_proj = nn.ModuleList()
        for i in range(len(in_channels)):
            self.channel_proj.append(ConvModule(in_channels=in_channels[i], out_channels=out_channels, kernel_size=1, norm_cfg=norm_cfg, act_cfg=act_cfg))
        self.feat_extract = nn.ModuleList()
        for i in range(len(in_channels)):
            self.feat_extract.append(ConvModule(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg))

    def forward(self, inputs):
        feat_list = []
        for x, conv in zip(inputs, self.channel_proj):
            feat_list.append(conv(x))
        feat_list = feat_list[::-1]
        mid_list = []
        for feat in feat_list:
            if len(mid_list) == 0:
                mid_list.append(feat)
            else:
                mid_list.append(mid_list[-1] + feat)
        out_list = []
        for mid, conv in zip(mid_list, self.feat_extract):
            out_list.append(conv(mid))
        return tuple(out_list)


class MLANeck(nn.Module):
    """Multi-level Feature Aggregation.

    This neck is `The Multi-level Feature Aggregation construction of
    SETR <https://arxiv.org/abs/2012.15840>`_.


    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        norm_layer (dict): Config dict for input normalization.
            Default: norm_layer=dict(type='LN', eps=1e-6, requires_grad=True).
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer in ConvModule.
            Default: None.
    """

    def __init__(self, in_channels, out_channels, norm_layer=dict(type='LN', eps=1e-06, requires_grad=True), norm_cfg=None, act_cfg=None):
        super(MLANeck, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm = nn.ModuleList([build_norm_layer(norm_layer, in_channels[i])[1] for i in range(len(in_channels))])
        self.mla = MLAModule(in_channels=in_channels, out_channels=out_channels, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        outs = []
        for i in range(len(inputs)):
            x = inputs[i]
            n, c, h, w = x.shape
            x = x.reshape(n, c, h * w).transpose(2, 1).contiguous()
            x = self.norm[i](x)
            x = x.transpose(1, 2).reshape(n, c, h, w).contiguous()
            outs.append(x)
        outs = self.mla(outs)
        return tuple(outs)


class MultiLevelNeck(nn.Module):
    """MultiLevelNeck.

    A neck structure connect vit backbone and decoder_heads.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        scales (List[float]): Scale factors for each input feature map.
            Default: [0.5, 1, 2, 4]
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer in ConvModule.
            Default: None.
    """

    def __init__(self, in_channels, out_channels, scales=[0.5, 1, 2, 4], norm_cfg=None, act_cfg=None):
        super(MultiLevelNeck, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scales = scales
        self.num_outs = len(scales)
        self.lateral_convs = nn.ModuleList()
        self.convs = nn.ModuleList()
        for in_channel in in_channels:
            self.lateral_convs.append(ConvModule(in_channel, out_channels, kernel_size=1, norm_cfg=norm_cfg, act_cfg=act_cfg))
        for _ in range(self.num_outs):
            self.convs.append(ConvModule(out_channels, out_channels, kernel_size=3, padding=1, stride=1, norm_cfg=norm_cfg, act_cfg=act_cfg))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        inputs = [lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)]
        if len(inputs) == 1:
            inputs = [inputs[0] for _ in range(self.num_outs)]
        outs = []
        for i in range(self.num_outs):
            x_resize = resize(inputs[i], scale_factor=self.scales[i], mode='bilinear')
            outs.append(self.convs[i](x_resize))
        return tuple(outs)


def make_divisible(value, divisor, min_value=None, min_ratio=0.9):
    """Make divisible function.

    This function rounds the channel number to the nearest value that can be
    divisible by the divisor. It is taken from the original tf repo. It ensures
    that all layers have a channel number that is divisible by divisor. It can
    be seen here: https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py  # noqa

    Args:
        value (int): The original channel number.
        divisor (int): The divisor to fully divide the channel number.
        min_value (int): The minimum value of the output channel.
            Default: None, means that the minimum value equal to the divisor.
        min_ratio (float): The minimum ratio of the rounded channel number to
            the original channel number. Default: 0.9.

    Returns:
        int: The modified output channel number.
    """
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if new_value < min_ratio * value:
        new_value += divisor
    return new_value


class SELayer(nn.Module):
    """Squeeze-and-Excitation Module.

    Args:
        channels (int): The input (and output) channels of the SE layer.
        ratio (int): Squeeze ratio in SELayer, the intermediate channel will be
            ``int(channels/ratio)``. Default: 16.
        conv_cfg (None or dict): Config dict for convolution layer.
            Default: None, which means using conv2d.
        act_cfg (dict or Sequence[dict]): Config dict for activation layer.
            If act_cfg is a dict, two activation layers will be configured
            by this dict. If act_cfg is a sequence of dicts, the first
            activation layer will be configured by the first dict and the
            second activation layer will be configured by the second dict.
            Default: (dict(type='ReLU'), dict(type='HSigmoid', bias=3.0,
            divisor=6.0)).
    """

    def __init__(self, channels, ratio=16, conv_cfg=None, act_cfg=(dict(type='ReLU'), dict(type='HSigmoid', bias=3.0, divisor=6.0))):
        super(SELayer, self).__init__()
        if isinstance(act_cfg, dict):
            act_cfg = act_cfg, act_cfg
        assert len(act_cfg) == 2
        assert mmcv.is_tuple_of(act_cfg, dict)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = ConvModule(in_channels=channels, out_channels=make_divisible(channels // ratio, 8), kernel_size=1, stride=1, conv_cfg=conv_cfg, act_cfg=act_cfg[0])
        self.conv2 = ConvModule(in_channels=make_divisible(channels // ratio, 8), out_channels=channels, kernel_size=1, stride=1, conv_cfg=conv_cfg, act_cfg=act_cfg[1])

    def forward(self, x):
        out = self.global_avgpool(x)
        out = self.conv1(out)
        out = self.conv2(out)
        return x * out


class InvertedResidualV3(nn.Module):
    """Inverted Residual Block for MobileNetV3.

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        mid_channels (int): The input channels of the depthwise convolution.
        kernel_size (int): The kernel size of the depthwise convolution.
            Default: 3.
        stride (int): The stride of the depthwise convolution. Default: 1.
        se_cfg (dict): Config dict for se layer. Default: None, which means no
            se layer.
        with_expand_conv (bool): Use expand conv or not. If set False,
            mid_channels must be the same with in_channels. Default: True.
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.

    Returns:
        Tensor: The output tensor.
    """

    def __init__(self, in_channels, out_channels, mid_channels, kernel_size=3, stride=1, se_cfg=None, with_expand_conv=True, conv_cfg=None, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU'), with_cp=False):
        super(InvertedResidualV3, self).__init__()
        self.with_res_shortcut = stride == 1 and in_channels == out_channels
        assert stride in [1, 2]
        self.with_cp = with_cp
        self.with_se = se_cfg is not None
        self.with_expand_conv = with_expand_conv
        if self.with_se:
            assert isinstance(se_cfg, dict)
        if not self.with_expand_conv:
            assert mid_channels == in_channels
        if self.with_expand_conv:
            self.expand_conv = ConvModule(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=1, padding=0, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.depthwise_conv = ConvModule(in_channels=mid_channels, out_channels=mid_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, groups=mid_channels, conv_cfg=dict(type='Conv2dAdaptivePadding') if stride == 2 else conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        if self.with_se:
            self.se = SELayer(**se_cfg)
        self.linear_conv = ConvModule(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=None)

    def forward(self, x):

        def _inner_forward(x):
            out = x
            if self.with_expand_conv:
                out = self.expand_conv(out)
            out = self.depthwise_conv(out)
            if self.with_se:
                out = self.se(out)
            out = self.linear_conv(out)
            if self.with_res_shortcut:
                return x + out
            else:
                return out
        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)
        return out


class UpConvBlock(nn.Module):
    """Upsample convolution block in decoder for UNet.

    This upsample convolution block consists of one upsample module
    followed by one convolution block. The upsample module expands the
    high-level low-resolution feature map and the convolution block fuses
    the upsampled high-level low-resolution feature map and the low-level
    high-resolution feature map from encoder.

    Args:
        conv_block (nn.Sequential): Sequential of convolutional layers.
        in_channels (int): Number of input channels of the high-level
        skip_channels (int): Number of input channels of the low-level
        high-resolution feature map from encoder.
        out_channels (int): Number of output channels.
        num_convs (int): Number of convolutional layers in the conv_block.
            Default: 2.
        stride (int): Stride of convolutional layer in conv_block. Default: 1.
        dilation (int): Dilation rate of convolutional layer in conv_block.
            Default: 1.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        conv_cfg (dict | None): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict | None): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict | None): Config dict for activation layer in ConvModule.
            Default: dict(type='ReLU').
        upsample_cfg (dict): The upsample config of the upsample module in
            decoder. Default: dict(type='InterpConv'). If the size of
            high-level feature map is the same as that of skip feature map
            (low-level feature map from encoder), it does not need upsample the
            high-level feature map and the upsample_cfg is None.
        dcn (bool): Use deformable convolution in convolutional layer or not.
            Default: None.
        plugins (dict): plugins for convolutional layers. Default: None.
    """

    def __init__(self, conv_block, in_channels, skip_channels, out_channels, num_convs=2, stride=1, dilation=1, with_cp=False, conv_cfg=None, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU'), upsample_cfg=dict(type='InterpConv'), dcn=None, plugins=None):
        super(UpConvBlock, self).__init__()
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'
        self.conv_block = conv_block(in_channels=2 * skip_channels, out_channels=out_channels, num_convs=num_convs, stride=stride, dilation=dilation, with_cp=with_cp, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg, dcn=None, plugins=None)
        if upsample_cfg is not None:
            self.upsample = build_upsample_layer(cfg=upsample_cfg, in_channels=in_channels, out_channels=skip_channels, with_cp=with_cp, norm_cfg=norm_cfg, act_cfg=act_cfg)
        else:
            self.upsample = ConvModule(in_channels, skip_channels, kernel_size=1, stride=1, padding=0, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, skip, x):
        """Forward function."""
        x = self.upsample(x)
        out = torch.cat([skip, x], dim=1)
        out = self.conv_block(out)
        return out


class ExampleModel(nn.Module):

    def __init__(self):
        super(ExampleModel, self).__init__()
        self.test_cfg = None
        self.conv = nn.Conv2d(3, 3, 3)

    def forward(self, img, img_metas, test_mode=False, **kwargs):
        return img

    def train_step(self, data_batch, optimizer):
        loss = self.forward(**data_batch)
        return dict(loss=loss)


class ExampleBackbone(nn.Module):

    def __init__(self):
        super(ExampleBackbone, self).__init__()
        self.conv = nn.Conv2d(3, 3, 3)

    def init_weights(self, pretrained=None):
        pass

    def forward(self, x):
        return [self.conv(x)]


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AdaptivePadding,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CrossEntropyLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (DiceLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
    (EMAModule,
     lambda: ([], {'channels': 4, 'num_bases': 4, 'num_stages': 4, 'momentum': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Encoding,
     lambda: ([], {'channels': 4, 'num_codes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ExampleBackbone,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (ExampleModel,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (InputInjection,
     lambda: ([], {'num_downsampling': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LambdaLayer,
     lambda: ([], {'lambd': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (NormedLinear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (PPMConcat,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (RSoftmax,
     lambda: ([], {'radix': 4, 'groups': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SpatialGatherModule,
     lambda: ([], {'scale': 1.0}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (StableBCELoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_dvlab_research_Parametric_Contrastive_Learning(_paritybench_base):
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

