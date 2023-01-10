import sys
_module = sys.modules[__name__]
del sys
clip = _module
adapter_config = _module
clip = _module
model = _module
simple_tokenizer = _module
adapters = _module
adapter_configuration = _module
adapter_controller = _module
adapter_hypernetwork = _module
adapter_modeling = _module
adapter_outputs = _module
adapter_utils = _module
config = _module
hypercomplex = _module
inits = _module
kronecker = _module
layers = _module
low_rank_layer = _module
entry = _module
file_utils = _module
modeling = _module
optimization = _module
tokenization = _module
visual_transformers = _module
param = _module
lxmert_data = _module
lxmert_pretrain = _module
qa_answer_table = _module
gqa = _module
gqa_data = _module
gqa_model = _module
snli = _module
snli_data = _module
vision_helpers = _module
vqa = _module
vqa_data = _module
vqa_model = _module
lmdb_dataset = _module
load_stagte_dict = _module
resize_images = _module
sharearray = _module
vision_helpers = _module
utils = _module
extracting_data = _module
modeling_frcnn = _module
processing_image = _module
utils = _module
visualizing_image = _module
activitynet = _module
activitynet_data = _module
activitynet_model = _module
adapter_configuration = _module
adapter_controller = _module
adapter_hypernetwork = _module
adapter_modeling = _module
adapter_outputs = _module
adapter_utils = _module
inits = _module
kronecker = _module
layers = _module
low_rank_layer = _module
caption = _module
caption_clip_data = _module
caption_data = _module
caption_model = _module
caption_raw_data = _module
classification = _module
classification_clip_data = _module
classification_model = _module
classification_raw_data = _module
clip = _module
model = _module
clip_prepro_feats = _module
dist_utils = _module
gqa = _module
gqa_clip_data = _module
gqa_data = _module
gqa_model = _module
gqa_raw_data = _module
how2qa = _module
lora = _module
controller = _module
layers = _module
utils = _module
mmt = _module
mmt_data = _module
mmt_model = _module
modeling_bart = _module
modeling_prefix_bart = _module
modeling_t5 = _module
multitask = _module
multitask_data = _module
multitask_model = _module
multitask_video = _module
my_deepspeed = _module
my_transformers = _module
modeling_bart = _module
modeling_t5 = _module
nlvr = _module
nlvr_clip_data = _module
nlvr_data = _module
nlvr_model = _module
nlvr_raw_data = _module
param = _module
preprocess = _module
pretrain = _module
pretrain_data = _module
pretrain_model = _module
pretrain_raw_data = _module
pretrain_vcr = _module
pretrain_vcr_data = _module
prompt = _module
prompt_controller = _module
prompt_modeling = _module
qa_answer_table = _module
refcoco = _module
refcoco_data = _module
refcoco_model = _module
refcoco_utils = _module
trainer_base = _module
tvc = _module
tvqa = _module
utils = _module
vcr = _module
vcr_data = _module
vcr_model = _module
how2qa_data = _module
tvc_data = _module
tvqa_data = _module
tvqa_matching_data = _module
tvr_data = _module
video_matching_model = _module
video_model = _module
yc2c_data = _module
vis_encoder = _module
vqa = _module
vqa_clip_data = _module
vqa_data = _module
vqa_model = _module
vqa_raw_data = _module
yc2c = _module
download_backbones = _module
coco_CLIP = _module
coco_gt = _module
coco_proposal = _module
coco_val_compact = _module
detectron2_given_box_maxnms = _module
detectron2_proposal_maxnms = _module
flickr30k_proposal = _module
refcocog_gt = _module
refcocog_mattnet = _module
tsv_to_h5 = _module
vcr_gt = _module
vcr_proposal = _module

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


import warnings


from typing import Union


from typing import List


import torch


from torchvision.transforms import Compose


from torchvision.transforms import Resize


from torchvision.transforms import CenterCrop


from torchvision.transforms import ToTensor


from torchvision.transforms import Normalize


from collections import OrderedDict


from typing import Tuple


import torch.nn.functional as F


from torch import nn


import torch.nn as nn


import math


from typing import Optional


import copy


import logging


import numpy as np


from torch.nn import CrossEntropyLoss


from torch.nn import SmoothL1Loss


from torch.optim import Optimizer


from torch.optim.optimizer import required


import random


from collections import defaultdict


from torch.utils.data import Dataset


import matplotlib.pyplot as plt


from torchvision.transforms import ColorJitter


import collections


from torch.utils.data import DataLoader


import torch.distributed as dist


from torch.utils.data.distributed import DistributedSampler


from torch.utils.data.dataloader import DataLoader


import torchvision


from torchvision.transforms import functional as F


from itertools import repeat


from itertools import chain


import torch.utils.data


from torch.utils.data.sampler import BatchSampler


from torch.utils.data.sampler import Sampler


from torch.utils.model_zoo import tqdm


from torch.nn.utils.rnn import pad_sequence


import itertools


from abc import ABCMeta


from abc import abstractmethod


from collections import namedtuple


from typing import Dict


from torch.nn.modules.batchnorm import BatchNorm2d


from torchvision.ops import RoIPool


from torchvision.ops.boxes import batched_nms


from torchvision.ops.boxes import nms


from functools import partial


import matplotlib as mpl


import matplotlib.colors as mplc


import matplotlib.figure as mplfigure


from matplotlib.backends.backend_agg import FigureCanvasAgg


import torch.backends.cudnn as cudnn


import torch.multiprocessing as mp


from torch.nn.parallel import DistributedDataParallel as DDP


from torch.utils.data import Sampler


import pandas as pd


from copy import deepcopy


import re


import torchvision as tv


from random import shuffle


from random import seed


import string


import torchvision.models as models


import functools


from typing import Any


from typing import Callable


from typing import Iterable


from torchvision.transforms import RandomCrop


from torchvision.transforms import RandomHorizontalFlip


from torchvision.transforms import RandomErasing


from torchvision.ops import nms


class VisualAdapter(nn.Module):
    """Conventional Adapter layer, in which the weights of up and down sampler modules
    are parameters and are optimized."""

    def __init__(self, input_dim, output_dim, adapter_kind, reduction_factor=16, use_bn=True):
        super().__init__()
        self.adapter_kind = adapter_kind
        self.use_bn = use_bn
        if adapter_kind == 'bottleneck':
            self.down_sample_size = input_dim // reduction_factor
            self.activation = nn.ReLU(inplace=True)
            self.down_sampler = nn.Conv2d(input_dim, self.down_sample_size, 1, bias=False)
            self.up_sampler = nn.Conv2d(self.down_sample_size, output_dim, 1, bias=False)
            if use_bn:
                self.bn1 = nn.BatchNorm2d(self.down_sample_size)
                self.bn2 = nn.BatchNorm2d(output_dim)
        elif adapter_kind == 'basic':
            self.activation = nn.ReLU(inplace=True)
            self.conv = nn.Conv2d(input_dim, output_dim, 1, bias=False)
            if use_bn:
                self.bn = nn.BatchNorm2d(output_dim)
        else:
            raise NotImplementedError

    def forward(self, x):
        if self.adapter_kind == 'bottleneck':
            z = self.down_sampler(x)
            z = self.bn1(z) if self.use_bn else z
            z = self.activation(z)
            output = self.up_sampler(z)
            output = self.bn2(output) if self.use_bn else output
        elif self.adapter_kind == 'basic':
            output = self.conv(x)
            output = self.bn(output) if self.use_bn else output
        return output


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, adapter_type=None, reduction_factor=16, use_bn=True):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride
        self.adapter_type = adapter_type
        self.front_adapter = None
        self.middle_adapter = None
        self.back_adapter = None
        self.transition_adapter = None
        self.use_bn = use_bn
        if self.adapter_type is not None:
            adapter_pos, adapter_kind = self.adapter_type.split('-')
            if 'front' in adapter_pos:
                self.front_adapter = VisualAdapter(inplanes, planes, adapter_kind, reduction_factor, use_bn)
            if 'middle' in adapter_pos:
                self.middle_adapter = VisualAdapter(planes, planes, adapter_kind, reduction_factor, use_bn)
            if 'back' in adapter_pos:
                self.back_adapter = VisualAdapter(planes, planes * self.expansion, adapter_kind, reduction_factor, use_bn)
            if 'transition' in adapter_pos:
                self.transition_adapter = VisualAdapter(planes * self.expansion, planes * self.expansion, adapter_kind, reduction_factor, use_bn)
        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            self.downsample = nn.Sequential(OrderedDict([('-1', nn.AvgPool2d(stride)), ('0', nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)), ('1', nn.BatchNorm2d(planes * self.expansion))]))

    def forward(self, x: torch.Tensor):
        identity = x
        if self.front_adapter is not None:
            adapter_out = self.front_adapter(x)
            if self.use_bn:
                out = self.bn1(self.conv1(x))
                out = self.relu(adapter_out + out)
            else:
                out = self.conv1(x)
                out = self.relu(self.bn1(adapter_out + out))
        else:
            out = self.relu(self.bn1(self.conv1(x)))
        if self.middle_adapter is not None:
            adapter_out = self.middle_adapter(out)
            if self.use_bn:
                out = self.bn2(self.conv2(out))
                out = self.relu(adapter_out + out)
            else:
                out = self.conv2(out)
                out = self.relu(self.bn2(adapter_out + out))
        else:
            out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        if self.back_adapter is not None:
            adapter_out = self.back_adapter(out)
            if self.use_bn:
                out = self.bn3(self.conv3(out))
                out = adapter_out + out
            else:
                out = self.conv3(out)
                out = self.bn3(adapter_out + out)
        else:
            out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        if self.transition_adapter is not None:
            adapter_out = self.transition_adapter(out)
            out = self.relu(adapter_out + out)
        return out


class AttentionPool2d(nn.Module):

    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int=None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)
        x = x + self.positional_embedding[0, :, None, :]
        x, _ = F.multi_head_attention_forward(query=x, key=x, value=x, embed_dim_to_check=x.shape[-1], num_heads=self.num_heads, q_proj_weight=self.q_proj.weight, k_proj_weight=self.k_proj.weight, v_proj_weight=self.v_proj.weight, in_proj_weight=None, in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]), bias_k=None, bias_v=None, add_zero_attn=False, dropout_p=0, out_proj_weight=self.c_proj.weight, out_proj_bias=self.c_proj.bias, use_separate_proj_weight=True, training=self.training, need_weights=False)
        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64, adapter_type=None, reduction_factor=1, use_bn=True):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution
        self.adapter_type = adapter_type
        self.reduction_factor = reduction_factor
        self.use_bn = use_bn
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)
        self._inplanes = width
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)
        embed_dim = width * 32
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride, adapter_type=self.adapter_type, reduction_factor=self.reduction_factor, use_bn=self.use_bn)]
        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes, adapter_type=self.adapter_type, reduction_factor=self.reduction_factor))
        return nn.Sequential(*layers)

    def forward(self, x):

        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x
        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        attnpool = self.attnpool(x)
        return x, attnpool


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):

    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):

    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor=None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([('c_fc', nn.Linear(d_model, d_model * 4)), ('gelu', QuickGELU()), ('c_proj', nn.Linear(d_model * 4, d_model))]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):

    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor=None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisualTransformer(nn.Module):

    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads)
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([self.class_embedding + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.positional_embedding
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_post(x)
        return x


class CLIP(nn.Module):

    def __init__(self, embed_dim: int, image_resolution: int, vision_layers: Union[Tuple[int, int, int, int], int], vision_width: int, vision_patch_size: int, context_length: int, vocab_size: int, transformer_width: int, transformer_heads: int, transformer_layers: int, adapter_type: str, reduction_factor: int, use_bn: bool):
        super().__init__()
        self.context_length = context_length
        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(layers=vision_layers, output_dim=embed_dim, heads=vision_heads, input_resolution=image_resolution, width=vision_width, adapter_type=adapter_type, reduction_factor=reduction_factor, use_bn=use_bn)
        else:
            vision_heads = vision_width // 64
            self.visual = VisualTransformer(input_resolution=image_resolution, patch_size=vision_patch_size, width=vision_width, layers=vision_layers, heads=vision_heads, output_dim=embed_dim)
        self.transformer = Transformer(width=transformer_width, layers=transformer_layers, heads=transformer_heads, attn_mask=self.build_attention_mask())
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]))
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)
            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith('bn3.weight'):
                        nn.init.zeros_(param)
        proj_std = self.transformer.width ** -0.5 * (2 * self.transformer.layers) ** -0.5
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float('-inf'))
        mask.triu_(1)
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()
        return logits_per_image, logits_per_text


class AutoAdapterConfig(nn.Module):
    """Generic Adapter config class to instantiate different adapter configs."""

    @classmethod
    def get(cls, config_name: str):
        if config_name in ADAPTER_CONFIG_MAPPING:
            return ADAPTER_CONFIG_MAPPING[config_name]()
        raise ValueError('Unrecognized adapter config type identifier: {}. Should contain one of {}'.format(config_name, ', '.join(ADAPTER_CONFIG_MAPPING.keys())))


class Activations(nn.Module):

    def __init__(self, activation_type):
        super().__init__()
        self.f = get_activation(activation_type)

    def forward(self, x):
        return self.f(x)


class Adapter(nn.Module):
    """Conventional Adapter layer, in which the weights of up and down sampler modules
    are parameters and are optimized."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_dim = config.d_model
        reduction_factor = config.reduction_factor
        self.down_sample_size = self.input_dim // reduction_factor
        self.activation = Activations(config.non_linearity.lower())
        self.down_sampler = nn.Linear(self.input_dim, self.down_sample_size)
        self.up_sampler = nn.Linear(self.down_sample_size, self.input_dim)
        self.track_z = config.track_z

    def forward(self, x):
        z = self.down_sampler(x)
        z = self.activation(z)
        if self.track_z:
            self.z = z
        output = self.up_sampler(z)
        return output


def glorot_normal(tensor: torch.Tensor):
    return torch.nn.init.xavier_normal_(tensor, gain=math.sqrt(2))


def glorot_uniform(tensor: torch.Tensor):
    return torch.nn.init.xavier_uniform_(tensor, gain=math.sqrt(2))


def kronecker_product(a, b):
    """
    Kronecker product of matrices a and b with leading batch dimensions.
    Batch dimensions are broadcast. The number of them mush
    :type a: torch.Tensor
    :type b: torch.Tensor
    :rtype: torch.Tensor
    """
    siz1 = torch.Size(torch.tensor(a.shape[-2:]) * torch.tensor(b.shape[-2:]))
    res = a.unsqueeze(-1).unsqueeze(-3) * b.unsqueeze(-2).unsqueeze(-4)
    siz0 = res.shape[:-4]
    out = res.reshape(siz0 + siz1)
    return out


def kronecker_product_einsum_batched(A: torch.Tensor, B: torch.Tensor):
    """
    Batched Version of Kronecker Products
    :param A: has shape (b, a, c)
    :param B: has shape (b, k, p)
    :return: (b, ak, cp)
    """
    assert A.dim() == 3 and B.dim() == 3
    res = torch.einsum('bac,bkp->bakcp', A, B).view(A.size(0), A.size(1) * B.size(1), A.size(2) * B.size(2))
    return res


def matvec_product(W: torch.Tensor, x: torch.Tensor, bias: Optional[torch.Tensor], phm_rule: Union[torch.Tensor], kronecker_prod=False) ->torch.Tensor:
    """
    Functional method to compute the generalized matrix-vector product based on the paper
    "Parameterization of Hypercomplex Multiplications (2020)"
    https://openreview.net/forum?id=rcQdycl0zyk
    y = Hx + b , where W is generated through the sum of kronecker products from the Parameterlist W, i.e.
    W is a an order-3 tensor of size (phm_dim, in_features, out_features)
    x has shape (batch_size, phm_dim*in_features)
    phm_rule is an order-3 tensor of shape (phm_dim, phm_dim, phm_dim)
    H = sum_{i=0}^{d} mul_rule \\otimes W[i], where \\otimes is the kronecker product
    """
    if kronecker_prod:
        H = kronecker_product(phm_rule, W).sum(0)
    else:
        H = kronecker_product_einsum_batched(phm_rule, W).sum(0)
    y = torch.matmul(input=x, other=H)
    if bias is not None:
        y += bias
    return y


class PHMLinear(torch.nn.Module):

    def __init__(self, in_features: int, out_features: int, phm_dim: int, phm_rule: Union[None, torch.Tensor]=None, bias: bool=True, w_init: str='phm', c_init: str='random', learn_phm: bool=True, shared_phm_rule=False, factorized_phm=False, shared_W_phm=False, factorized_phm_rule=False, phm_rank=1, phm_init_range=0.0001, kronecker_prod=False) ->None:
        super(PHMLinear, self).__init__()
        assert w_init in ['phm', 'glorot-normal', 'glorot-uniform', 'normal']
        assert c_init in ['normal', 'uniform']
        assert in_features % phm_dim == 0, f'Argument `in_features`={in_features} is not divisble be `phm_dim`{phm_dim}'
        assert out_features % phm_dim == 0, f'Argument `out_features`={out_features} is not divisble be `phm_dim`{phm_dim}'
        self.in_features = in_features
        self.out_features = out_features
        self.learn_phm = learn_phm
        self.phm_dim = phm_dim
        self._in_feats_per_axis = in_features // phm_dim
        self._out_feats_per_axis = out_features // phm_dim
        self.phm_rank = phm_rank
        self.phm_init_range = phm_init_range
        self.kronecker_prod = kronecker_prod
        self.shared_phm_rule = shared_phm_rule
        self.factorized_phm_rule = factorized_phm_rule
        if not self.shared_phm_rule:
            if self.factorized_phm_rule:
                self.phm_rule_left = nn.Parameter(torch.FloatTensor(phm_dim, phm_dim, 1), requires_grad=learn_phm)
                self.phm_rule_right = nn.Parameter(torch.FloatTensor(phm_dim, 1, phm_dim), requires_grad=learn_phm)
            else:
                self.phm_rule = nn.Parameter(torch.FloatTensor(phm_dim, phm_dim, phm_dim), requires_grad=learn_phm)
        self.bias_flag = bias
        self.w_init = w_init
        self.c_init = c_init
        self.shared_W_phm = shared_W_phm
        self.factorized_phm = factorized_phm
        if not self.shared_W_phm:
            if self.factorized_phm:
                self.W_left = nn.Parameter(torch.Tensor(size=(phm_dim, self._in_feats_per_axis, self.phm_rank)), requires_grad=True)
                self.W_right = nn.Parameter(torch.Tensor(size=(phm_dim, self.phm_rank, self._out_feats_per_axis)), requires_grad=True)
            else:
                self.W = nn.Parameter(torch.Tensor(size=(phm_dim, self._in_feats_per_axis, self._out_feats_per_axis)), requires_grad=True)
        if self.bias_flag:
            self.b = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('b', None)
        self.reset_parameters()

    def init_W(self):
        if self.w_init == 'glorot-normal':
            if self.factorized_phm:
                for i in range(self.phm_dim):
                    self.W_left.data[i] = glorot_normal(self.W_left.data[i])
                    self.W_right.data[i] = glorot_normal(self.W_right.data[i])
            else:
                for i in range(self.phm_dim):
                    self.W.data[i] = glorot_normal(self.W.data[i])
        elif self.w_init == 'glorot-uniform':
            if self.factorized_phm:
                for i in range(self.phm_dim):
                    self.W_left.data[i] = glorot_uniform(self.W_left.data[i])
                    self.W_right.data[i] = glorot_uniform(self.W_right.data[i])
            else:
                for i in range(self.phm_dim):
                    self.W.data[i] = glorot_uniform(self.W.data[i])
        elif self.w_init == 'normal':
            if self.factorized_phm:
                for i in range(self.phm_dim):
                    self.W_left.data[i].normal_(mean=0, std=self.phm_init_range)
                    self.W_right.data[i].normal_(mean=0, std=self.phm_init_range)
            else:
                for i in range(self.phm_dim):
                    self.W.data[i].normal_(mean=0, std=self.phm_init_range)
        else:
            raise ValueError

    def reset_parameters(self):
        if not self.shared_W_phm:
            self.init_W()
        if self.bias_flag:
            self.b.data = torch.zeros_like(self.b.data)
        if not self.shared_phm_rule:
            if self.factorized_phm_rule:
                if self.c_init == 'uniform':
                    self.phm_rule_left.data.uniform_(-0.01, 0.01)
                    self.phm_rule_right.data.uniform_(-0.01, 0.01)
                elif self.c_init == 'normal':
                    self.phm_rule_left.data.normal_(std=self.phm_init_range)
                    self.phm_rule_right.data.normal_(std=self.phm_init_range)
                else:
                    raise NotImplementedError
            elif self.c_init == 'uniform':
                self.phm_rule.data.uniform_(-0.01, 0.01)
            elif self.c_init == 'normal':
                self.phm_rule.data.normal_(mean=0, std=self.phm_init_range)
            else:
                raise NotImplementedError

    def set_phm_rule(self, phm_rule=None, phm_rule_left=None, phm_rule_right=None):
        """If factorized_phm_rules is set, phm_rule is a tuple, showing the left and right
        phm rules, and if this is not set, this is showing  the phm_rule."""
        if self.factorized_phm_rule:
            self.phm_rule_left = phm_rule_left
            self.phm_rule_right = phm_rule_right
        else:
            self.phm_rule = phm_rule

    def set_W(self, W=None, W_left=None, W_right=None):
        if self.factorized_phm:
            self.W_left = W_left
            self.W_right = W_right
        else:
            self.W = W

    def forward(self, x: torch.Tensor, phm_rule: Union[None, nn.ParameterList]=None) ->torch.Tensor:
        if self.factorized_phm:
            W = torch.bmm(self.W_left, self.W_right)
        if self.factorized_phm_rule:
            phm_rule = torch.bmm(self.phm_rule_left, self.phm_rule_right)
        return matvec_product(W=W if self.factorized_phm else self.W, x=x, bias=self.b, phm_rule=phm_rule if self.factorized_phm_rule else self.phm_rule, kronecker_prod=self.kronecker_prod)


class HyperComplexAdapter(nn.Module):
    """Hypercomplex Adapter layer, in which the weights of up and down sampler modules
    are parameters are 1/n times of the conventional adapter layers, where n is
    hypercomplex division number."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.down_sample_size = self.input_dim // config.reduction_factor
        self.activation = Activations(config.non_linearity.lower())
        self.down_sampler = PHMLinear(in_features=self.input_dim, out_features=self.down_sample_size, bias=True, c_init=config.phm_c_init, phm_dim=config.hypercomplex_division, learn_phm=config.learn_phm, w_init=config.hypercomplex_nonlinearity, shared_phm_rule=config.shared_phm_rule, factorized_phm=config.factorized_phm, shared_W_phm=config.shared_W_phm, factorized_phm_rule=config.factorized_phm_rule, phm_rank=config.phm_rank, phm_init_range=config.phm_init_range, kronecker_prod=config.kronecker_prod)
        self.up_sampler = PHMLinear(in_features=self.down_sample_size, out_features=self.input_dim, bias=True, c_init=config.phm_c_init, phm_dim=config.hypercomplex_division, learn_phm=config.learn_phm, w_init=config.hypercomplex_nonlinearity, shared_phm_rule=config.shared_phm_rule, factorized_phm=config.factorized_phm, shared_W_phm=config.shared_W_phm, factorized_phm_rule=config.factorized_phm_rule, phm_rank=config.phm_rank, phm_init_range=config.phm_init_range, kronecker_prod=config.kronecker_prod)
        self.track_z = config.track_z

    def forward(self, x):
        z = self.down_sampler(x)
        z = self.activation(z)
        if self.track_z:
            self.z = z
        return self.up_sampler(z)


class LowRankLinear(torch.nn.Module):

    def __init__(self, input_dim: int, output_dim: int, rank: int=1, bias: bool=True, w_init: str='glorot-uniform'):
        super(LowRankLinear, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rank = rank
        self.bias = bias
        self.w_init = w_init
        self.W_left = nn.Parameter(torch.Tensor(size=(input_dim, rank)), requires_grad=True)
        self.W_right = nn.Parameter(torch.Tensor(size=(rank, output_dim)), requires_grad=True)
        if bias:
            self.b = nn.Parameter(torch.Tensor(output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        if self.bias:
            self.b.data = torch.zeros_like(self.b.data)
        if self.w_init == 'glorot-uniform':
            self.W_left.data = glorot_uniform(self.W_left.data)
            self.W_right.data = glorot_uniform(self.W_right.data)
        elif self.w_init == 'glorot-normal':
            self.W_left.data = glorot_normal(self.W_left.data)
            self.W_right.data = glorot_normal(self.W_right.data)
        else:
            raise ValueError

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        W = self.W_left.matmul(self.W_right)
        output = torch.matmul(input=x, other=W)
        if self.bias:
            output += self.b
        return output


class LowRankAdapter(nn.Module):
    """This is the low-rank adapter, in which each adapter is composed of two rank-one matrices.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.down_sample_size = self.input_dim // config.reduction_factor
        self.activation = Activations(config.non_linearity.lower())
        self.down_sampler = LowRankLinear(self.input_dim, self.down_sample_size, w_init=config.low_rank_w_init, rank=config.low_rank_rank)
        self.up_sampler = LowRankLinear(self.down_sample_size, self.input_dim, w_init=config.low_rank_w_init, rank=config.low_rank_rank)
        self.track_z = config.track_z

    def forward(self, x):
        z = self.down_sampler(x)
        z = self.activation(z)
        if self.track_z:
            self.z = z
        output = self.up_sampler(z)
        return output


class AdapterController(nn.Module):
    """Implements Adapter controller module which controls the logics of
    putting adapter layers within transformer's layers."""

    def __init__(self, config):
        super().__init__()
        self.low_rank_adapters = config.low_rank_adapters
        self.config = config
        self.adapters = nn.ModuleDict(dict())
        self.tasks = config.tasks
        self.shared_phm_rule = config.shared_phm_rule
        self.hypercomplex_adapters = config.hypercomplex_adapters
        self.use_single_adapter = config.use_single_adapter
        self.share_up_sampler = config.share_up_sampler
        self.share_down_sampler = config.share_down_sampler
        self.shared_phm_rule_over_tasks = config.shared_phm_rule_over_tasks
        self.adapters = self.construct_adapters(self.tasks)
        self.add_layer_norm_before_adapter = config.add_layer_norm_before_adapter
        self.add_layer_norm_after_adapter = config.add_layer_norm_after_adapter
        if self.add_layer_norm_before_adapter:
            self.pre_layer_norm = nn.LayerNorm(config.input_dim)
        if self.add_layer_norm_after_adapter:
            self.post_layer_norm = nn.LayerNorm(config.input_dim)

    def get_task(self, task):
        return task

    def construct_adapters(self, tasks):
        """
        Constructs adapter layers and adds them to a dictionary for the given
        tasks.
        Args:
            tasks: A list of string containing the task names.
        """
        if self.use_single_adapter:
            if self.hypercomplex_adapters:
                adapter = HyperComplexAdapter(self.config)
            elif self.low_rank_adapters:
                adapter = LowRankAdapter(self.config)
            else:
                adapter = Adapter(self.config)
            for task in tasks:
                self.adapters[task] = adapter
        else:
            for task in tasks:
                if self.hypercomplex_adapters:
                    self.adapters[task] = HyperComplexAdapter(self.config)
                elif self.low_rank_adapters:
                    self.adapters[task] = LowRankAdapter(self.config)
                else:
                    self.adapters[task] = Adapter(self.config)
            if self.share_up_sampler:
                layer = self.adapters[tasks[0]].up_sampler
                for task in tasks:
                    self.adapters[task].up_sampler = layer
            if self.share_down_sampler:
                layer = self.adapters[tasks[0]].down_sampler
                for task in tasks:
                    self.adapters[task].down_sampler = layer
            if self.hypercomplex_adapters and self.shared_phm_rule_over_tasks and not self.shared_phm_rule:
                up_phm_rule = self.adapters[tasks[0]].up_sampler.phm_rule
                down_phm_rule = self.adapters[tasks[0]].down_sampler.phm_rule
                for task in tasks:
                    self.adapters[task].up_sampler.phm_rule = up_phm_rule
                    self.adapters[task].down_sampler.phm_rule = down_phm_rule
        return self.adapters

    def disable_adapters(self, tasks):
        """
        Given a list of tasks, it freezes their corresponding adapter layers'
        parameters.
        Args:
           tasks: List of tasks.
        """
        tasks = self.convert_to_list(tasks)
        for task in tasks:
            adapter = self.get_adapter(task)
            for param in adapter.parameters():
                param.requires_grad = False

    def convert_to_list(self, tasks):
        if isinstance(tasks, list):
            return tasks
        return [tasks]

    def enable_adapters(self, tasks):
        """
        Given a list of tasks, it unfreezes their corresponding adapter layers.
        Args:
            tasks: Given list of tasks.
        """
        tasks = self.convert_to_list(tasks)
        for task in tasks:
            adapter = self.get_adapter(task)
            for name, param in adapter.named_parameters():
                if self.config.hypercomplex_adapters and not self.config.learn_phm:
                    if not 'phm_rule' in name:
                        param.requires_grad = True
                else:
                    param.requires_grad = True

    def get_adapter(self, task):
        """Given a task returns its corresponding adapter layer.
        Args:
            task: Input task name.
        Returns:
            Adapter layer corresponding to the given task.
        """
        return self.adapters[task]

    def forward(self, inputs, task):
        """
        Retrieves the adapter layer corresponding to the given
        task. It freezes the adapter layers for all the other tasks
        and call the selected adapter layer.
        Args:
            task: the name of the current task.
            inputs: the inputs to feed in in the adapter layer.
        Returns:
            outputs of the adapter layer.
        """
        task = self.get_task(task)
        adapter = self.get_adapter(task)
        z = self.pre_layer_norm(inputs) if self.add_layer_norm_before_adapter else inputs
        outputs = adapter(z)
        if self.add_layer_norm_after_adapter:
            outputs = self.post_layer_norm(outputs)
        outputs = outputs + inputs
        return outputs


class AdapterLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.adapter = Adapter(config)
        self.add_layer_norm_before_adapter = config.add_layer_norm_before_adapter
        self.add_layer_norm_after_adapter = config.add_layer_norm_after_adapter
        if self.add_layer_norm_before_adapter:
            self.pre_layer_norm = nn.LayerNorm(config.d_model)
        if self.add_layer_norm_after_adapter:
            self.post_layer_norm = nn.LayerNorm(config.d_model)

    def forward(self, inputs):
        z = self.pre_layer_norm(inputs) if self.add_layer_norm_before_adapter else inputs
        outputs = self.adapter(z)
        if self.add_layer_norm_after_adapter:
            outputs = self.post_layer_norm(outputs)
        outputs = outputs + inputs
        return outputs


class OutputAdapter(nn.Module):
    """Conventional Adapter layer, in which the weights of up and down sampler modules
    are parameters and are optimized."""

    def __init__(self, config, output_dim):
        super().__init__()
        self.config = config
        self.input_dim = config.d_model
        reduction_factor = 16
        self.down_sample_size = self.input_dim // reduction_factor
        self.activation = Activations(config.non_linearity.lower())
        self.down_sampler = nn.Linear(self.input_dim, self.down_sample_size)
        self.up_sampler = nn.Linear(self.down_sample_size, output_dim)

    def forward(self, x):
        z = self.down_sampler(x)
        z = self.activation(z)
        output = self.up_sampler(z)
        return output

    def resize_up_sampler(self, resized_size):
        self.up_sampler = nn.Linear(self.down_sample_size, resized_size)


class OutputParallelAdapterLayer(nn.Module):

    def __init__(self, config, output_dim):
        super().__init__()
        self.adapter = OutputAdapter(config, output_dim)
        self.add_layer_norm_before_adapter = config.add_layer_norm_before_adapter
        self.add_layer_norm_after_adapter = config.add_layer_norm_after_adapter
        if self.add_layer_norm_before_adapter:
            self.pre_layer_norm = nn.LayerNorm(config.d_model)
        if self.add_layer_norm_after_adapter:
            self.post_layer_norm = nn.LayerNorm(output_dim)

    def forward(self, inputs):
        z = self.pre_layer_norm(inputs) if self.add_layer_norm_before_adapter else inputs
        outputs = self.adapter(z)
        if self.add_layer_norm_after_adapter:
            outputs = self.post_layer_norm(outputs)
        outputs = outputs
        return outputs

    def resize_output_dim(self, resized_size):
        self.adapter.resize_up_sampler(resized_size)
        if self.add_layer_norm_after_adapter:
            self.post_layer_norm = nn.LayerNorm(resized_size)


class MetaLayersAdapterController(nn.Module):
    """Implements Meta Adapter controller module, in which
    the adapter layers' weights are generated from a unique hyper-network."""

    def __init__(self, config):
        super().__init__()
        self.activation_type = config.non_linearity.lower()
        self.input_dim = config.input_dim
        self.add_layer_norm_before_adapter = config.add_layer_norm_before_adapter
        self.add_layer_norm_after_adapter = config.add_layer_norm_after_adapter
        self.track_z = config.track_z

    def apply_layer_norm(self, inputs, layer_norm_weights):
        """Applies layer norm to the inputs."""
        return torch.nn.functional.layer_norm(inputs, (self.input_dim,), weight=layer_norm_weights.weight, bias=layer_norm_weights.bias)

    def call_adapter(self, inputs, adapter_weights):
        """Computes the output of the adapter layers."""
        down = F.linear(inputs, weight=adapter_weights.down.weight, bias=adapter_weights.down.bias)
        middle = get_activation(self.activation_type)(down)
        if self.track_z:
            self.z = middle
        output = F.linear(middle, weight=adapter_weights.up.weight, bias=adapter_weights.up.bias)
        return output

    def forward(self, inputs, adapter_weights):
        z = self.apply_layer_norm(inputs, adapter_weights.pre_norm) if self.add_layer_norm_before_adapter else inputs
        outputs = self.call_adapter(z, adapter_weights)
        if self.add_layer_norm_after_adapter:
            outputs = self.apply_layer_norm(outputs, adapter_weights.post_norm)
        outputs = outputs + inputs
        return outputs


def init_linear_layer(linear_layer, std=0.01):
    """Initializes the given linear module as explained in adapter paper."""
    nn.init.normal_(linear_layer.weight, std=std)
    nn.init.zeros_(linear_layer.bias)


def linear_layer(input_dim, output_dim, std=0.01):
    """Generates a linear module and initializes it."""
    linear = nn.Linear(input_dim, output_dim)
    init_linear_layer(linear, std=std)
    return linear


class AdapterHyperNet(nn.Module):
    """This module generates the weights for the meta adapter layers."""

    def __init__(self, config, input_dim, output_dim):
        super(AdapterHyperNet, self).__init__()
        self.hidden_dim = config.hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.train_task_embeddings = config.train_task_embeddings
        self.task_embedding_dim = config.projected_task_embedding_dim if config.train_task_embeddings else config.task_embedding_dim
        self.weight_generator = nn.Sequential(linear_layer(self.task_embedding_dim, self.input_dim * self.output_dim))
        self.bias_generator = nn.Sequential(linear_layer(self.task_embedding_dim, self.input_dim))

    def forward(self, task_embedding):
        task_embedding = task_embedding.view(-1)
        weight = self.weight_generator(task_embedding).view(self.input_dim, self.output_dim)
        bias = self.bias_generator(task_embedding).view(-1)
        return weight, bias


class AdapterLayersHyperNet(nn.Module):
    """This module generates the weights for all the meta adapter layers
    given the task embeddings and layer id."""

    def __init__(self, config, input_dim, output_dim):
        super(AdapterLayersHyperNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight_generator = nn.Sequential(linear_layer(config.projected_task_embedding_dim, self.input_dim * self.output_dim))
        self.bias_generator = nn.Sequential(linear_layer(config.projected_task_embedding_dim, self.input_dim))

    def forward(self, embeddings):
        weight = self.weight_generator(embeddings).view(self.input_dim, self.output_dim)
        bias = self.bias_generator(embeddings).view(-1)
        return SamplerOutput(weight=weight, bias=bias)


class LayerNormHyperNet(nn.Module):
    """This module generates the weight and bias for the task conditioned layer norm."""

    def __init__(self, config):
        super(LayerNormHyperNet, self).__init__()
        self.task_embedding_dim = config.projected_task_embedding_dim if config.train_task_embeddings else config.task_embedding_dim
        self.weight_generator = linear_layer(self.task_embedding_dim, config.input_dim)
        self.bias_generator = linear_layer(self.task_embedding_dim, config.input_dim)

    def forward(self, input):
        return self.weight_generator(input), self.bias_generator(input)


class TaskHyperNet(nn.Module):
    """This module generates the task-embeddings from the initial feeded task embeddings."""

    def __init__(self, config, input_dim):
        super(TaskHyperNet, self).__init__()
        self.task_hidden_dim = config.task_hidden_dim
        self.projected_task_embedding_dim = config.projected_task_embedding_dim
        self.task_embeding_generator = nn.Sequential(linear_layer(input_dim, self.task_hidden_dim), nn.ReLU(), linear_layer(self.task_hidden_dim, self.projected_task_embedding_dim))

    def forward(self, task_embedding):
        task_embedding = task_embedding.view(-1)
        return self.task_embeding_generator(task_embedding).view(-1)


class AdapterLayersHyperNetController(nn.Module):
    """This modules contains the hyper-nets for the feed forward
    and self-attention modules and it generates the adapter's weights and
    layer norm's weights for all the layers of transformers."""

    def __init__(self, config, num_layers=6, include_cross_attention=False):
        super(AdapterLayersHyperNetController, self).__init__()
        self.num_layers = num_layers
        self.layer_norm_epsilon = 1e-06
        self.max_position_embeddings = 2
        self.task_embedding_dim = config.task_embedding_dim
        self.layer_id_embeddings = nn.Embedding(self.num_layers, self.task_embedding_dim)
        self.task_hypernet = TaskHyperNet(config, input_dim=self.task_embedding_dim * 2)
        self.unique_hyper_net_layer_norm = config.unique_hyper_net_layer_norm
        if self.unique_hyper_net_layer_norm:
            self.LayerNorm = nn.LayerNorm(config.projected_task_embedding_dim, eps=self.layer_norm_epsilon)
        self.input_dim = config.input_dim
        self.down_sample_size = self.input_dim // config.reduction_factor
        self.feed_forward_up_sampler_hyper_net = AdapterLayersHyperNet(config, self.input_dim, self.down_sample_size)
        self.feed_forward_down_sampler_hyper_net = AdapterLayersHyperNet(config, self.down_sample_size, self.input_dim)
        self.self_attention_up_sampler_hyper_net = AdapterLayersHyperNet(config, self.input_dim, self.down_sample_size)
        self.self_attention_down_sampler_hyper_net = AdapterLayersHyperNet(config, self.down_sample_size, self.input_dim)
        self.include_cross_attention = include_cross_attention
        if include_cross_attention:
            self.cross_attention_up_sampler_hyper_net = AdapterLayersHyperNet(config, self.input_dim, self.down_sample_size)
            self.cross_attention_down_sampler_hyper_net = AdapterLayersHyperNet(config, self.down_sample_size, self.input_dim)
        self.add_layer_norm_before_adapter = config.add_layer_norm_before_adapter
        self.add_layer_norm_after_adapter = config.add_layer_norm_after_adapter
        self.train_task_embeddings = config.train_task_embeddings
        config.train_task_embeddings = True
        if self.add_layer_norm_before_adapter:
            self.feed_forward_pre_layernorm_hypernet = LayerNormHyperNet(config)
            self.self_attention_pre_layernorm_hypernet = LayerNormHyperNet(config)
            self.cross_attention_pre_layernorm_hypernet = LayerNormHyperNet(config)
        if self.add_layer_norm_after_adapter:
            self.feed_forward_post_layernorm_hypernet = LayerNormHyperNet(config)
            self.self_attention_post_layernorm_hypernet = LayerNormHyperNet(config)
            self.cross_attention_post_layernorm_hypernet = LayerNormHyperNet(config)
        config.train_task_embeddings = self.train_task_embeddings

    def get_embedding(self, task_embedding, layer_id):
        """Concatenates the task embedding with the embedding for the layer id and
        returns the final joint embedding."""
        layer_id_tensor = torch.tensor([layer_id], dtype=torch.long, device=task_embedding.device)
        layer_embedding = self.layer_id_embeddings(layer_id_tensor)
        layer_embedding = layer_embedding.view(-1)
        embeddings = torch.cat([task_embedding.view(1, -1), layer_embedding.view(1, -1)], axis=0)
        embeddings = self.task_hypernet(embeddings.view(-1))
        if self.unique_hyper_net_layer_norm:
            embeddings = self.LayerNorm(embeddings)
        return embeddings

    def forward(self, task_embedding, layer_id):
        embeddings = self.get_embedding(task_embedding, layer_id)
        feed_forward_down = self.feed_forward_down_sampler_hyper_net(embeddings)
        feed_forward_up = self.feed_forward_up_sampler_hyper_net(embeddings)
        self_attention_down = self.self_attention_down_sampler_hyper_net(embeddings)
        self_attention_up = self.self_attention_up_sampler_hyper_net(embeddings)
        if self.include_cross_attention:
            cross_attention_down = self.cross_attention_down_sampler_hyper_net(embeddings)
            cross_attention_up = self.cross_attention_up_sampler_hyper_net(embeddings)
            cross_attention_output = AdapterOutput(up=cross_attention_up, down=cross_attention_down)
        else:
            cross_attention_output = None
        feed_forward_output = AdapterOutput(up=feed_forward_up, down=feed_forward_down)
        self_attention_output = AdapterOutput(up=self_attention_up, down=self_attention_down)
        if self.add_layer_norm_before_adapter:
            weight, bias = self.feed_forward_pre_layernorm_hypernet(embeddings)
            feed_forward_output.pre_norm = LayerNormOutput(weight=weight, bias=bias)
            weight, bias = self.self_attention_pre_layernorm_hypernet(embeddings)
            self_attention_output.pre_norm = LayerNormOutput(weight=weight, bias=bias)
            if self.include_cross_attention:
                weight, bias = self.cross_attention_pre_layernorm_hypernet(embeddings)
                cross_attention_output.pre_norm = LayerNormOutput(weight=weight, bias=bias)
        if self.add_layer_norm_after_adapter:
            weight, bias = self.feed_forward_post_layernorm_hypernet(embeddings)
            feed_forward_output.post_norm = LayerNormOutput(weight=weight, bias=bias)
            weight, bias = self.self_attention_post_layernorm_hypernet(embeddings)
            self_attention_output.post_norm = LayerNormOutput(weight=weight, bias=bias)
            if self.include_cross_attention:
                weight, bias = self.cross_attention_post_layernorm_hypernet(embeddings)
                cross_attention_output.post_norm = LayerNormOutput(weight=weight, bias=bias)
        return AdapterT5BlockOutput(feed_forward=feed_forward_output, self_attention=self_attention_output, cross_attention=cross_attention_output)


class AdapterLayersOneHyperNetController(nn.Module):
    """This modules contains the hyper-nets for the feed forward
    and self-attention modules and it generates the adapter's weights and
    layer norm's weights for all the layers of transformers."""

    def __init__(self, config, num_layers=6, include_cross_attention=False):
        super(AdapterLayersOneHyperNetController, self).__init__()
        self.num_layers = num_layers
        self.layer_norm_epsilon = 1e-06
        self.max_position_embeddings = 2
        self.task_embedding_dim = config.task_embedding_dim
        self.layer_id_embeddings = nn.Embedding(self.num_layers, self.task_embedding_dim)
        self.adapters_block_type = nn.Embedding(3, self.task_embedding_dim)
        self.task_hypernet = TaskHyperNet(config, self.task_embedding_dim * 3)
        self.unique_hyper_net_layer_norm = config.unique_hyper_net_layer_norm
        if self.unique_hyper_net_layer_norm:
            self.LayerNorm = nn.LayerNorm(config.projected_task_embedding_dim, eps=self.layer_norm_epsilon)
        self.input_dim = config.input_dim
        self.down_sample_size = self.input_dim // config.reduction_factor
        self.include_cross_attention = include_cross_attention
        self.up_sampler_hyper_net = AdapterLayersHyperNet(config, self.input_dim, self.down_sample_size)
        self.down_sampler_hyper_net = AdapterLayersHyperNet(config, self.down_sample_size, self.input_dim)
        self.add_layer_norm_before_adapter = config.add_layer_norm_before_adapter
        self.add_layer_norm_after_adapter = config.add_layer_norm_after_adapter
        self.train_task_embeddings = config.train_task_embeddings
        config.train_task_embeddings = True
        if self.add_layer_norm_before_adapter:
            self.pre_layernorm_hypernet = LayerNormHyperNet(config)
        if self.add_layer_norm_after_adapter:
            self.post_layernorm_hypernet = LayerNormHyperNet(config)
        config.train_task_embeddings = self.train_task_embeddings

    def get_embedding(self, task_embedding, layer_id, block_type):
        """Concatenates the task embedding with the embedding for the layer id and
        returns the final joint embedding."""
        layer_id_tensor = torch.tensor([layer_id], dtype=torch.long, device=task_embedding.device)
        layer_embedding = self.layer_id_embeddings(layer_id_tensor)
        type_id_tensor = torch.tensor([block_type], dtype=torch.long, device=task_embedding.device)
        type_embedding = self.adapters_block_type(type_id_tensor)
        layer_embedding = layer_embedding.view(-1)
        type_embedding = type_embedding.view(-1)
        embeddings = torch.cat([task_embedding.view(1, -1), layer_embedding.view(1, -1), type_embedding.view(1, -1)], axis=0)
        embeddings = self.task_hypernet(embeddings.view(-1))
        if self.unique_hyper_net_layer_norm:
            embeddings = self.LayerNorm(embeddings)
        return embeddings

    def forward(self, task_embedding, layer_id):
        feed_forward_embeddings = self.get_embedding(task_embedding, layer_id, 0)
        self_attention_embeddings = self.get_embedding(task_embedding, layer_id, 1)
        feed_forward_down = self.down_sampler_hyper_net(feed_forward_embeddings)
        feed_forward_up = self.up_sampler_hyper_net(feed_forward_embeddings)
        self_attention_down = self.down_sampler_hyper_net(self_attention_embeddings)
        self_attention_up = self.up_sampler_hyper_net(self_attention_embeddings)
        feed_forward_output = AdapterOutput(up=feed_forward_up, down=feed_forward_down)
        self_attention_output = AdapterOutput(up=self_attention_up, down=self_attention_down)
        if self.include_cross_attention:
            cross_attention_embeddings = self.get_embedding(task_embedding, layer_id, 2)
            cross_attention_down = self.down_sampler_hyper_net(cross_attention_embeddings)
            cross_attention_up = self.up_sampler_hyper_net(cross_attention_embeddings)
            cross_attention_output = AdapterOutput(up=cross_attention_up, down=cross_attention_down)
        else:
            cross_attention_output = None
        if self.add_layer_norm_before_adapter:
            weight, bias = self.pre_layernorm_hypernet(feed_forward_embeddings)
            feed_forward_output.pre_norm = LayerNormOutput(weight=weight, bias=bias)
            weight, bias = self.pre_layernorm_hypernet(self_attention_embeddings)
            self_attention_output.pre_norm = LayerNormOutput(weight=weight, bias=bias)
            if self.include_cross_attention:
                weight, bias = self.pre_layernorm_hypernet(cross_attention_embeddings)
                cross_attention_output.pre_norm = LayerNormOutput(weight=weight, bias=bias)
        if self.add_layer_norm_after_adapter:
            weight, bias = self.post_layernorm_hypernet(feed_forward_embeddings)
            feed_forward_output.post_norm = LayerNormOutput(weight=weight, bias=bias)
            weight, bias = self.post_layernorm_hypernet(self_attention_embeddings)
            self_attention_output.post_norm = LayerNormOutput(weight=weight, bias=bias)
            if self.include_cross_attention:
                weight, bias = self.post_layernorm_hypernet(cross_attention_embeddings)
                cross_attention_output.post_norm = LayerNormOutput(weight=weight, bias=bias)
        return AdapterT5BlockOutput(feed_forward=feed_forward_output, self_attention=self_attention_output, cross_attention=cross_attention_output)


class TaskEmbeddingController(nn.Module):
    """Main module controlling task embeddings."""

    def __init__(self, config):
        super(TaskEmbeddingController, self).__init__()
        self.task_embedding_dim = config.task_embedding_dim
        self.tasks = config.tasks
        self.task_to_task_embeddings = {task: task for task in self.tasks}
        if config.task_to_embeddings is not None:
            self.task_to_task_embeddings = config.task_to_embeddings
            self.tasks = self.task_to_task_embeddings.values()
        self.set_task_embeddings(self.tasks)
        self.train_task_embeddings = config.train_task_embeddings
        if self.train_task_embeddings:
            self.task_hyper_net = TaskHyperNet(config)

    def get_task(self, task):
        return self.task_to_task_embeddings[task]

    def set_task_embeddings(self, tasks):
        self.task_to_embeddings = nn.ParameterDict(dict())
        for task in tasks:
            task_embedding = torch.Tensor(torch.randn(self.task_embedding_dim))
            self.task_to_embeddings[task] = nn.Parameter(task_embedding)

    def forward(self, task):
        task_mapped = self.get_task(task)
        task_embedding = self.task_to_embeddings[task_mapped]
        if self.train_task_embeddings:
            return self.task_hyper_net(task_embedding)
        return task_embedding


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {'gelu': gelu, 'relu': torch.nn.functional.relu, 'swish': swish}


class BertIntermediate(nn.Module):

    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str) or sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


BertLayerNorm = torch.nn.LayerNorm


class BertOutput(nn.Module):

    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if config.adapter_config is not None:
            self.adapter = AdapterController(config.adapter_config)
        else:
            self.adapter = None

    def forward(self, hidden_states, input_tensor, task):
        hidden_states = self.dense(hidden_states)
        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states, task)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttOutput(nn.Module):

    def __init__(self, config):
        super(BertAttOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if config.adapter_config is not None:
            self.adapter = AdapterController(config.adapter_config)
        else:
            self.adapter = None

    def forward(self, hidden_states, input_tensor, task):
        hidden_states = self.dense(hidden_states)
        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states, task)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):

    def __init__(self, config, ctx_dim=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError('The hidden size (%d) is not a multiple of the number of attention heads (%d)' % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        if ctx_dim is None:
            ctx_dim = config.hidden_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(ctx_dim, self.all_head_size)
        self.value = nn.Linear(ctx_dim, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, context, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(context)
        mixed_value_layer = self.value(context)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertSelfattLayer(nn.Module):

    def __init__(self, config):
        super(BertSelfattLayer, self).__init__()
        self.self = BertAttention(config)
        self.output = BertAttOutput(config)

    def forward(self, input_tensor, attention_mask, task=None):
        self_output = self.self(input_tensor, input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor, task)
        return attention_output


class BertLayer(nn.Module):

    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertSelfattLayer(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, task=None):
        attention_output = self.attention(hidden_states, attention_mask, task)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output, task)
        return layer_output


class BertCrossattLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.att = BertAttention(config)
        self.output = BertAttOutput(config)

    def forward(self, input_tensor, ctx_tensor, ctx_att_mask=None, task=None):
        output = self.att(input_tensor, ctx_tensor, ctx_att_mask)
        attention_output = self.output(output, input_tensor, task)
        return attention_output


class LXRTXLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.visual_attention = BertCrossattLayer(config)
        self.lang_self_att = BertSelfattLayer(config)
        self.visn_self_att = BertSelfattLayer(config)
        self.lang_inter = BertIntermediate(config)
        self.lang_output = BertOutput(config)
        self.visn_inter = BertIntermediate(config)
        self.visn_output = BertOutput(config)

    def cross_att(self, lang_input, lang_attention_mask, visn_input, visn_attention_mask, task=None):
        lang_att_output = self.visual_attention(lang_input, visn_input, ctx_att_mask=visn_attention_mask, task=task)
        visn_att_output = self.visual_attention(visn_input, lang_input, ctx_att_mask=lang_attention_mask, task=task)
        return lang_att_output, visn_att_output

    def self_att(self, lang_input, lang_attention_mask, visn_input, visn_attention_mask, task=None):
        lang_att_output = self.lang_self_att(lang_input, lang_attention_mask, task)
        visn_att_output = self.visn_self_att(visn_input, visn_attention_mask, task)
        return lang_att_output, visn_att_output

    def output_fc(self, lang_input, visn_input, task=None):
        lang_inter_output = self.lang_inter(lang_input)
        visn_inter_output = self.visn_inter(visn_input)
        lang_output = self.lang_output(lang_inter_output, lang_input, task)
        visn_output = self.visn_output(visn_inter_output, visn_input, task)
        return lang_output, visn_output

    def forward(self, lang_feats, lang_attention_mask, visn_feats, visn_attention_mask, task=None):
        lang_att_output = lang_feats
        visn_att_output = visn_feats
        lang_att_output, visn_att_output = self.cross_att(lang_att_output, lang_attention_mask, visn_att_output, visn_attention_mask, task)
        lang_att_output, visn_att_output = self.self_att(lang_att_output, lang_attention_mask, visn_att_output, visn_attention_mask, task)
        lang_output, visn_output = self.output_fc(lang_att_output, visn_att_output, task)
        return lang_output, visn_output


class VisualConfig(object):
    VISUAL_LOSSES = ['obj', 'attr', 'feat']

    def __init__(self, l_layers=12, x_layers=5, r_layers=0, use_clip=False, visualbert_style=False, freeze_clip=False, clip_model_name='ViT-B/32', drop_boxes=False, vilt_style=False, use_vit=False, reset_pos_embedding=False, sub_sampling=False, sub_feat_num=36, use_positional_embedding=False, use_max_pooling=False, pos_num=25):
        self.l_layers = l_layers
        self.x_layers = x_layers
        self.r_layers = r_layers
        if use_clip and clip_model_name == 'ViT-B/32':
            self.visual_feat_dim = 768
        elif use_vit:
            self.visual_feat_dim = 768
        elif use_clip and clip_model_name == 'RN50x4':
            self.visual_feat_dim = 2560
        else:
            self.visual_feat_dim = 2048
        self.visual_pos_dim = 4
        self.obj_id_num = 1600
        self.attr_id_num = 400
        self.visual_losses = self.VISUAL_LOSSES
        self.visual_loss_config = {'obj': (self.obj_id_num, 'ce', (-1,), 1 / 0.15), 'attr': (self.attr_id_num, 'ce', (-1,), 1 / 0.15), 'feat': (self.visual_feat_dim, 'l2', (-1, self.visual_feat_dim), 1 / 0.15)}
        self.use_clip = use_clip
        self.visualbert_style = visualbert_style
        self.freeze_clip = freeze_clip
        self.clip_model_name = clip_model_name
        self.drop_boxes = drop_boxes
        self.vilt_style = vilt_style
        self.use_vit = use_vit
        self.reset_pos_embedding = reset_pos_embedding
        self.sub_sampling = sub_sampling
        self.sub_feat_num = sub_feat_num
        self.use_positional_embedding = use_positional_embedding
        self.use_max_pooling = use_max_pooling
        self.pos_num = pos_num

    def set_visual_dims(self, feat_dim, pos_dim):
        self.visual_feat_dim = feat_dim
        self.visual_pos_dim = pos_dim


class LinearPositionEmbedding(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.x_position_embedding = nn.Embedding(VISUAL_CONFIG.pos_num, VISUAL_CONFIG.visual_feat_dim)
        self.y_position_embedding = nn.Embedding(VISUAL_CONFIG.pos_num, VISUAL_CONFIG.visual_feat_dim)
        self.hidden_size = VISUAL_CONFIG.visual_feat_dim

    def forward(self, visn_feats):
        width = visn_feats.size(2)
        width_ids = torch.arange(width, dtype=torch.long, device=visn_feats.device)
        width_ids = width_ids.unsqueeze(0)
        x_embedding = self.x_position_embedding(width_ids).unsqueeze(-2)
        height = visn_feats.size(3)
        height_ids = torch.arange(height, dtype=torch.long, device=visn_feats.device)
        height_ids = height_ids.unsqueeze(0)
        y_embedding = self.y_position_embedding(height_ids).unsqueeze(-3)
        position_embedding = x_embedding + y_embedding
        position_embedding = position_embedding.view(1, -1, self.hidden_size)
        visn_feats = visn_feats.permute(0, 2, 3, 1).view(visn_feats.size(0), -1, visn_feats.size(1))
        visn_feats += position_embedding
        return visn_feats


class VisualFeatEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        feat_dim = VISUAL_CONFIG.visual_feat_dim
        pos_dim = VISUAL_CONFIG.visual_pos_dim
        self.visn_fc = nn.Linear(feat_dim, config.hidden_size)
        self.visn_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.box_fc = nn.Linear(pos_dim, config.hidden_size)
        self.box_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, visn_input):
        if isinstance(visn_input, tuple):
            feats, boxes = visn_input
            x = self.visn_fc(feats)
            x = self.visn_layer_norm(x)
            y = self.box_fc(boxes)
            y = self.box_layer_norm(y)
            output = (x + y) / 2
            output = self.dropout(output)
            return output
        else:
            feats = visn_input
            x = self.visn_fc(feats)
            x = self.visn_layer_norm(x)
            x = self.dropout(x)
            return x


def _cat_with_none(feat_1, feat_2, dim):
    if feat_1 is None:
        return feat_2
    if feat_2 is None:
        return feat_1
    return torch.cat((feat_1, feat_2), dim=dim)


def _split_with_none(lang_feats, visn_feats, joint_feats):
    if lang_feats is None:
        assert visn_feats.size(1) == joint_feats.size(1)
        return None, joint_feats
    if visn_feats is None:
        assert lang_feats.size(1) == joint_feats.size(1)
        return joint_feats, None
    return joint_feats[:, :lang_feats.size(1), :].contiguous(), joint_feats[:, lang_feats.size(1):, :].contiguous()


class LXRTEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.visualbert_style = VISUAL_CONFIG.visualbert_style
        self.visn_fc = VisualFeatEncoder(config)
        self.num_l_layers = VISUAL_CONFIG.l_layers
        self.num_x_layers = VISUAL_CONFIG.x_layers
        self.num_r_layers = VISUAL_CONFIG.r_layers
        if self.visualbert_style:
            layers = [BertLayer(config) for _ in range(self.num_l_layers)]
            self.layer = nn.ModuleList(layers)
            None
        else:
            None
            self.layer = nn.ModuleList([BertLayer(config) for _ in range(self.num_l_layers)])
            self.x_layers = nn.ModuleList([LXRTXLayer(config) for _ in range(self.num_x_layers)])
            self.r_layers = nn.ModuleList([BertLayer(config) for _ in range(self.num_r_layers)])
        if VISUAL_CONFIG.use_clip:
            self.visual_model = initialize_clip(VISUAL_CONFIG, adapter_config=config.vis_adapter_config)
            del self.visual_model.transformer
        elif VISUAL_CONFIG.use_vit:
            self.visual_model = initialize_vit(VISUAL_CONFIG)
        if VISUAL_CONFIG.use_positional_embedding:
            self.visual_pos = LinearPositionEmbedding(config)
        if VISUAL_CONFIG.use_max_pooling:
            self.max_pooling = nn.MaxPool2d(2, stride=2)

    def forward(self, lang_feats, lang_attention_mask, visn_feats, visn_attention_mask=None, task=None):
        if VISUAL_CONFIG.vilt_style:
            assert not VISUAL_CONFIG.freeze_clip
            if VISUAL_CONFIG.use_clip:
                images, boxes = visn_feats
                lang_attention_mask = lang_attention_mask.squeeze(1).squeeze(1)
                lang_attention_mask[lang_attention_mask != 0] = float('-inf')
                joint_feats = self.visual_model.visual(images.type(self.visual_model.dtype), skip_last_layer=True, text_embedding=lang_feats, text_mask=lang_attention_mask)
                return _split_with_none(lang_feats, images, joint_feats)
            elif VISUAL_CONFIG.use_vit:
                images, boxes = visn_feats
                joint_feats = self.visual_model(images, return_features=True, text_embedding=lang_feats, text_mask=lang_attention_mask)
                return _split_with_none(lang_feats, images, joint_feats)
        if VISUAL_CONFIG.use_clip:
            images, boxes = visn_feats
            visn_feats = self.visual_model.visual(images.type(self.visual_model.dtype), skip_last_layer=True)
            if 'RN' in VISUAL_CONFIG.clip_model_name:
                if VISUAL_CONFIG.use_max_pooling:
                    visn_feats = self.max_pooling(visn_feats)
                if VISUAL_CONFIG.use_positional_embedding:
                    visn_feats = self.visual_pos(visn_feats)
                else:
                    visn_feats = visn_feats.permute(0, 2, 3, 1).view(visn_feats.size(0), -1, visn_feats.size(1))
            visn_feats = visn_feats
        elif VISUAL_CONFIG.use_vit:
            images, boxes = visn_feats
            visn_feats = self.visual_model(images, return_features=True)
            visn_feats = visn_feats
        elif VISUAL_CONFIG.drop_boxes:
            visn_feats, boxes = visn_feats
        if VISUAL_CONFIG.sub_sampling:
            sub_feat_num = VISUAL_CONFIG.sub_feat_num
            sampled_index = []
            for i in range(visn_feats.size(0)):
                sampled_index.append(torch.from_numpy(np.random.choice(visn_feats.size(1), sub_feat_num, replace=False)))
            sampled_index = torch.stack(sampled_index, dim=0).unsqueeze(-1).expand(visn_feats.size(0), sub_feat_num, visn_feats.size(2)).long()
            visn_feats = torch.gather(visn_feats, 1, sampled_index)
        visn_feats = self.visn_fc(visn_feats)
        if visn_attention_mask is None:
            visn_attention_mask = torch.zeros(visn_feats.size(0), visn_feats.size(1)).to(dtype=next(self.visn_fc.parameters()).dtype)
            visn_attention_mask = visn_attention_mask.unsqueeze(1).unsqueeze(2)
        if VISUAL_CONFIG.visualbert_style:
            joint_feats = _cat_with_none(lang_feats, visn_feats, dim=1)
            joint_mask = _cat_with_none(lang_attention_mask, visn_attention_mask, dim=-1)
            all_attention_weights = []
            for idx, layer_module in enumerate(self.layer):
                joint_feats = layer_module(joint_feats, joint_mask, task)
            return _split_with_none(lang_feats, visn_feats, joint_feats)
        else:
            for idx, layer_module in enumerate(self.layer):
                lang_feats = layer_module(lang_feats, lang_attention_mask, task)
            for idx, layer_module in enumerate(self.r_layers):
                visn_feats = layer_module(visn_feats, visn_attention_mask, task)
            for idx, layer_module in enumerate(self.x_layers):
                lang_feats, visn_feats = layer_module(lang_feats, lang_attention_mask, visn_feats, visn_attention_mask, task)
        return lang_feats, visn_feats


class GeLU(nn.Module):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu(x)


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size, padding_idx=0)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size, padding_idx=0)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertPooler(nn.Module):

    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):

    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str) or sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):

    def __init__(self, config, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1), bert_model_embedding_weights.size(0), bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertVisualAnswerHead(nn.Module):

    def __init__(self, config, num_answers):
        super().__init__()
        hid_dim = config.hidden_size
        self.logit_fc = nn.Sequential(nn.Linear(hid_dim, hid_dim * 2), GeLU(), BertLayerNorm(hid_dim * 2, eps=1e-12), nn.Linear(hid_dim * 2, num_answers))

    def forward(self, hidden_states):
        return self.logit_fc(hidden_states)


class BertVisualObjHead(nn.Module):

    def __init__(self, config, visual_losses):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        visual_losses = visual_losses.split(',')
        for loss in visual_losses:
            assert loss in VISUAL_CONFIG.VISUAL_LOSSES
        self.visual_losses = visual_losses
        self.decoder_dict = nn.ModuleDict({key: nn.Linear(config.hidden_size, VISUAL_CONFIG.visual_loss_config[key][0]) for key in self.visual_losses})

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        output = {}
        for key in self.visual_losses:
            output[key] = self.decoder_dict[key](hidden_states)
        return output


class BertPreTrainingHeads(nn.Module):

    def __init__(self, config, bert_model_embedding_weights):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """

    def __init__(self, vocab_size_or_config_json_file, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, hidden_act='gelu', hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=2, initializer_range=0.02):
        """Constructs BertConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        if isinstance(vocab_size_or_config_json_file, str) or sys.version_info[0] == 2 and isinstance(vocab_size_or_config_json_file, unicode):
            with open(vocab_size_or_config_json_file, 'r', encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
        else:
            raise ValueError('First argument must be either a vocabulary size (int)or the path to a pretrained model config file (str)')

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, 'r', encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + '\n'


CONFIG_NAME = 'config.yaml'


PRETRAINED_MODEL_ARCHIVE_MAP = {'bert-base-uncased': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz', 'bert-large-uncased': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased.tar.gz', 'bert-base-cased': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz', 'bert-large-cased': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased.tar.gz', 'bert-base-multilingual-uncased': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased.tar.gz', 'bert-base-multilingual-cased': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased.tar.gz', 'bert-base-chinese': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz'}


TF_WEIGHTS_NAME = 'model.ckpt'


WEIGHTS_NAME = 'pytorch_model.bin'


def http_get(url, temp_file, proxies=None, resume_size=0, user_agent=None):
    ua = 'python/{}'.format(sys.version.split()[0])
    if _torch_available:
        ua += '; torch/{}'.format(torch.__version__)
    if isinstance(user_agent, dict):
        ua += '; ' + '; '.join('{}/{}'.format(k, v) for k, v in user_agent.items())
    elif isinstance(user_agent, str):
        ua += '; ' + user_agent
    headers = {'user-agent': ua}
    if resume_size > 0:
        headers['Range'] = 'bytes=%d-' % (resume_size,)
    response = requests.get(url, stream=True, proxies=proxies, headers=headers)
    if response.status_code == 416:
        return
    content_length = response.headers.get('Content-Length')
    total = resume_size + int(content_length) if content_length is not None else None
    progress = tqdm(unit='B', unit_scale=True, total=total, initial=resume_size, desc='Downloading')
    for chunk in response.iter_content(chunk_size=1024):
        if chunk:
            progress.update(len(chunk))
            temp_file.write(chunk)
    progress.close()


def url_to_filename(url, etag=None):
    url_bytes = url.encode('utf-8')
    url_hash = sha256(url_bytes)
    filename = url_hash.hexdigest()
    if etag:
        etag_bytes = etag.encode('utf-8')
        etag_hash = sha256(etag_bytes)
        filename += '.' + etag_hash.hexdigest()
    if url.endswith('.h5'):
        filename += '.h5'
    return filename


def get_from_cache(url, cache_dir=None, force_download=False, proxies=None, etag_timeout=10, resume_download=False, user_agent=None, local_files_only=False):
    if cache_dir is None:
        cache_dir = TRANSFORMERS_CACHE
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    etag = None
    if not local_files_only:
        try:
            response = requests.head(url, allow_redirects=True, proxies=proxies, timeout=etag_timeout)
            if response.status_code == 200:
                etag = response.headers.get('ETag')
        except (EnvironmentError, requests.exceptions.Timeout):
            pass
    filename = url_to_filename(url, etag)
    cache_path = os.path.join(cache_dir, filename)
    if etag is None:
        if os.path.exists(cache_path):
            return cache_path
        else:
            matching_files = [file for file in fnmatch.filter(os.listdir(cache_dir), filename + '.*') if not file.endswith('.json') and not file.endswith('.lock')]
            if len(matching_files) > 0:
                return os.path.join(cache_dir, matching_files[-1])
            else:
                if local_files_only:
                    raise ValueError("Cannot find the requested files in the cached path and outgoing traffic has been disabled. To enable model look-ups and downloads online, set 'local_files_only' to False.")
                return None
    if os.path.exists(cache_path) and not force_download:
        return cache_path
    lock_path = cache_path + '.lock'
    with FileLock(lock_path):
        if os.path.exists(cache_path) and not force_download:
            return cache_path
        if resume_download:
            incomplete_path = cache_path + '.incomplete'

            @contextmanager
            def _resumable_file_manager():
                with open(incomplete_path, 'a+b') as f:
                    yield f
            temp_file_manager = _resumable_file_manager
            if os.path.exists(incomplete_path):
                resume_size = os.stat(incomplete_path).st_size
            else:
                resume_size = 0
        else:
            temp_file_manager = partial(tempfile.NamedTemporaryFile, dir=cache_dir, delete=False)
            resume_size = 0
        with temp_file_manager() as temp_file:
            None
            http_get(url, temp_file, proxies=proxies, resume_size=resume_size, user_agent=user_agent)
        os.replace(temp_file.name, cache_path)
        meta = {'url': url, 'etag': etag}
        meta_path = cache_path + '.json'
        with open(meta_path, 'w') as meta_file:
            json.dump(meta, meta_file)
    return cache_path


def is_remote_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ('http', 'https')


def cached_path(url_or_filename, cache_dir=None, force_download=False, proxies=None, resume_download=False, user_agent=None, extract_compressed_file=False, force_extract=False, local_files_only=False):
    if cache_dir is None:
        cache_dir = TRANSFORMERS_CACHE
    if isinstance(url_or_filename, Path):
        url_or_filename = str(url_or_filename)
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)
    if is_remote_url(url_or_filename):
        output_path = get_from_cache(url_or_filename, cache_dir=cache_dir, force_download=force_download, proxies=proxies, resume_download=resume_download, user_agent=user_agent, local_files_only=local_files_only)
    elif os.path.exists(url_or_filename):
        output_path = url_or_filename
    elif urlparse(url_or_filename).scheme == '':
        raise EnvironmentError('file {} not found'.format(url_or_filename))
    else:
        raise ValueError('unable to parse {} as a URL or as a local path'.format(url_or_filename))
    if extract_compressed_file:
        if not is_zipfile(output_path) and not tarfile.is_tarfile(output_path):
            return output_path
        output_dir, output_file = os.path.split(output_path)
        output_extract_dir_name = output_file.replace('.', '-') + '-extracted'
        output_path_extracted = os.path.join(output_dir, output_extract_dir_name)
        if os.path.isdir(output_path_extracted) and os.listdir(output_path_extracted) and not force_extract:
            return output_path_extracted
        lock_path = output_path + '.lock'
        with FileLock(lock_path):
            shutil.rmtree(output_path_extracted, ignore_errors=True)
            os.makedirs(output_path_extracted)
            if is_zipfile(output_path):
                with ZipFile(output_path, 'r') as zip_file:
                    zip_file.extractall(output_path_extracted)
                    zip_file.close()
            elif tarfile.is_tarfile(output_path):
                tar_file = tarfile.open(output_path)
                tar_file.extractall(output_path_extracted)
                tar_file.close()
            else:
                raise EnvironmentError('Archive format of {} could not be identified'.format(output_path))
        return output_path_extracted
    return output_path


def load_tf_weights_in_bert(model, tf_checkpoint_path):
    """ Load tf checkpoints in a pytorch model
    """
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except Importtokenization:
        None
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    None
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        None
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)
    for name, array in zip(names, arrays):
        name = name.split('/')
        if any(n in ['adam_v', 'adam_m'] for n in name):
            None
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch('[A-Za-z]+_\\d+', m_name):
                l = re.split('_(\\d+)', m_name)
            else:
                l = [m_name]
            if l[0] == 'kernel' or l[0] == 'gamma':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'output_bias' or l[0] == 'beta':
                pointer = getattr(pointer, 'bias')
            elif l[0] == 'output_weights':
                pointer = getattr(pointer, 'weight')
            else:
                pointer = getattr(pointer, l[0])
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        if m_name[-11:] == '_embeddings':
            pointer = getattr(pointer, 'weight')
        elif m_name == 'kernel':
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += pointer.shape, array.shape
            raise
        None
        pointer.data = torch.from_numpy(array)
    return model


class BertPreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """

    def __init__(self, config, *inputs, **kwargs):
        super(BertPreTrainedModel, self).__init__()
        if not isinstance(config, BertConfig):
            raise ValueError('Parameter config in `{}(config)` should be an instance of class `BertConfig`. To create a model from a Google pretrained model use `model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`'.format(self.__class__.__name__, self.__class__.__name__))
        self.config = config

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, state_dict=None, cache_dir=None, from_tf=False, *inputs, **kwargs):
        """
        Instantiate a BertPreTrainedModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.

        Params:
            pretrained_model_name_or_path: either:
                - a str with the name of a pre-trained model to load selected in the list of:
                    . `bert-base-uncased`
                    . `bert-large-uncased`
                    . `bert-base-cased`
                    . `bert-large-cased`
                    . `bert-base-multilingual-uncased`
                    . `bert-base-multilingual-cased`
                    . `bert-base-chinese`
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `pytorch_model.bin` a PyTorch dump of a BertForPreTraining instance
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `model.chkpt` a TensorFlow checkpoint
            from_tf: should we load the weights from a locally saved TensorFlow checkpoint
            cache_dir: an optional path to a folder in which the pre-trained models will be cached.
            state_dict: an optional state dictionnary (collections.OrderedDict object) to use instead of Google pre-trained models
            *inputs, **kwargs: additional input for the specific Bert class
                (ex: num_labels for BertForSequenceClassification)
        """
        if pretrained_model_name_or_path in PRETRAINED_MODEL_ARCHIVE_MAP:
            archive_file = PRETRAINED_MODEL_ARCHIVE_MAP[pretrained_model_name_or_path]
        else:
            archive_file = pretrained_model_name_or_path
        try:
            resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir)
        except EnvironmentError:
            if pretrained_model_name_or_path == 'bert-base-uncased':
                try:
                    None
                    archive_file = 'https://nlp.cs.unc.edu/data/bert/bert-base-uncased.tar.gz'
                    resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir)
                except EnvironmentError:
                    None
                    return None
            else:
                logger.error("Model name '{}' was not found in model name list ({}). We assumed '{}' was a path or url but couldn't find any file associated to this path or url.".format(pretrained_model_name_or_path, ', '.join(PRETRAINED_MODEL_ARCHIVE_MAP.keys()), archive_file))
        if resolved_archive_file == archive_file:
            logger.info('loading archive file {}'.format(archive_file))
        else:
            logger.info('loading archive file {} from cache at {}'.format(archive_file, resolved_archive_file))
        tempdir = None
        if os.path.isdir(resolved_archive_file) or from_tf:
            serialization_dir = resolved_archive_file
        else:
            tempdir = tempfile.mkdtemp()
            logger.info('extracting archive file {} to temp dir {}'.format(resolved_archive_file, tempdir))
            with tarfile.open(resolved_archive_file, 'r:gz') as archive:
                archive.extractall(tempdir)
            serialization_dir = tempdir
        config_file = os.path.join(serialization_dir, CONFIG_NAME)
        config = BertConfig.from_json_file(config_file)
        logger.info('Model config {}'.format(config))
        model = cls(config, *inputs, **kwargs)
        if state_dict is None and not from_tf:
            weights_path = os.path.join(serialization_dir, WEIGHTS_NAME)
            state_dict = torch.load(weights_path, map_location='cpu' if not torch.cuda.is_available() else None)
        if tempdir:
            shutil.rmtree(tempdir)
        if from_tf:
            weights_path = os.path.join(serialization_dir, TF_WEIGHTS_NAME)
            return load_tf_weights_in_bert(model, weights_path)
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)
        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')
        start_prefix = ''
        if not hasattr(model, 'bert') and any(s.startswith('bert.') for s in state_dict.keys()):
            start_prefix = 'bert.'
        load(model, prefix=start_prefix)
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(model.__class__.__name__, '\n\t'.join(error_msgs)))
        return model


class LXRTModel(BertPreTrainedModel):
    """LXRT Model."""

    def __init__(self, config):
        super().__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = LXRTEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, visual_feats=None, visual_attention_mask=None, task=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        if visual_attention_mask is not None:
            extended_visual_attention_mask = visual_attention_mask.unsqueeze(1).unsqueeze(2)
            extended_visual_attention_mask = extended_visual_attention_mask
            extended_visual_attention_mask = (1.0 - extended_visual_attention_mask) * -10000.0
        else:
            extended_visual_attention_mask = None
        embedding_output = self.embeddings(input_ids, token_type_ids)
        lang_feats, visn_feats = self.encoder(embedding_output, extended_attention_mask, visn_feats=visual_feats, visn_attention_mask=extended_visual_attention_mask, task=task)
        pooled_output = self.pooler(lang_feats)
        return (lang_feats, visn_feats), pooled_output


class LXRTPretraining(BertPreTrainedModel):

    def __init__(self, config, task_mask_lm=True, task_matched=True, task_obj_predict=True, visual_losses='', task_qa=True, num_answers=2):
        super().__init__(config)
        self.config = config
        self.num_answers = num_answers
        self.task_mask_lm = task_mask_lm
        self.task_obj_predict = task_obj_predict
        self.task_matched = task_matched
        self.task_qa = task_qa
        self.bert = LXRTModel(config)
        self.cls = BertPreTrainingHeads(config, self.bert.embeddings.word_embeddings.weight)
        if self.task_obj_predict:
            self.obj_predict_head = BertVisualObjHead(config, visual_losses)
        if self.task_qa:
            self.answer_head = BertVisualAnswerHead(config, self.num_answers)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None, visual_feats=None, pos=None, obj_labels=None, matched_label=None, ans=None):
        (lang_output, visn_output), pooled_output = self.bert(input_ids, token_type_ids, attention_mask, visual_feats=(visual_feats, pos))
        lang_prediction_scores, cross_relationship_score = self.cls(lang_output, pooled_output)
        if self.task_qa:
            answer_score = self.answer_head(pooled_output)
        else:
            answer_score = pooled_output[0][0]
        total_loss = 0.0
        loss_fct = CrossEntropyLoss(ignore_index=-1)
        losses = ()
        if masked_lm_labels is not None and self.task_mask_lm:
            masked_lm_loss = loss_fct(lang_prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            total_loss += masked_lm_loss
            losses += masked_lm_loss.detach(),
        if matched_label is not None and self.task_matched:
            matched_loss = loss_fct(cross_relationship_score.view(-1, 2), matched_label.view(-1))
            total_loss += matched_loss
            losses += matched_loss.detach(),
        if obj_labels is not None and self.task_obj_predict:
            loss_fcts = {'l2': SmoothL1Loss(reduction='none'), 'ce': CrossEntropyLoss(ignore_index=-1, reduction='none')}
            total_visn_loss = 0.0
            visn_prediction_scores_dict = self.obj_predict_head(visn_output)
            for key in VISUAL_CONFIG.visual_losses:
                label, mask_conf = obj_labels[key]
                output_dim, loss_fct_name, label_shape, weight = VISUAL_CONFIG.visual_loss_config[key]
                visn_loss_fct = loss_fcts[loss_fct_name]
                visn_prediction_scores = visn_prediction_scores_dict[key]
                visn_loss = visn_loss_fct(visn_prediction_scores.view(-1, output_dim), label.view(*label_shape))
                if visn_loss.dim() > 1:
                    visn_loss = visn_loss.mean(1)
                visn_loss = (visn_loss * mask_conf.view(-1)).mean() * weight
                total_visn_loss += visn_loss
                losses += visn_loss.detach(),
            total_loss += total_visn_loss
        if ans is not None and self.task_qa:
            answer_loss = loss_fct(answer_score.view(-1, self.num_answers), ans.view(-1))
            total_loss += answer_loss
            losses += answer_loss.detach(),
        return total_loss, torch.stack(losses).unsqueeze(0), answer_score.detach()


class LXRTFeatureExtraction(BertPreTrainedModel):
    """
    BERT model for classification.
    """

    def __init__(self, config, mode='lxr', adapter_config=None, vis_adapter_config=None):
        """

        :param config:
        :param mode:  Number of visual layers
        """
        config.adapter_config = adapter_config
        config.vis_adapter_config = vis_adapter_config
        super().__init__(config)
        self.bert = LXRTModel(config)
        self.mode = mode
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, visual_feats=None, visual_attention_mask=None, task=None):
        feat_seq, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, visual_feats=visual_feats, visual_attention_mask=visual_attention_mask, task=task)
        if 'x' == self.mode:
            return pooled_output
        elif 'x' in self.mode and ('l' in self.mode or 'r' in self.mode):
            return feat_seq, pooled_output
        elif 'l' in self.mode or 'r' in self.mode:
            return feat_seq


MAX_GQA_LENGTH = 20


class GQAModel(nn.Module):

    def __init__(self, num_answers):
        super().__init__()
        self.lxrt_encoder = LXRTEncoder(args, max_seq_length=MAX_GQA_LENGTH)
        hid_dim = self.lxrt_encoder.dim
        self.logit_fc = nn.Sequential(nn.Linear(hid_dim, hid_dim * 2), GeLU(), BertLayerNorm(hid_dim * 2, eps=1e-12), nn.Linear(hid_dim * 2, num_answers))
        self.logit_fc.apply(self.lxrt_encoder.model.init_bert_weights)
        self.task = 'vqa'

    def forward(self, feat, pos, sent):
        """
        b -- batch_size, o -- object_number, f -- visual_feature_size

        :param feat: (b, o, f)
        :param pos:  (b, o, 4)
        :param sent: (b,) Type -- list of string
        :param leng: (b,) Type -- int numpy array
        :return: (b, num_answer) The logit of each answers.
        """
        x = self.lxrt_encoder(sent, (feat, pos), task=self.task)
        logit = self.logit_fc(x)
        return logit


MAX_VQA_LENGTH = 20


class VQAModel(nn.Module):

    def __init__(self, num_answers):
        super().__init__()
        self.lxrt_encoder = LXRTEncoder(args, max_seq_length=MAX_VQA_LENGTH)
        hid_dim = self.lxrt_encoder.dim
        self.logit_fc = nn.Sequential(nn.Linear(hid_dim, hid_dim * 2), GeLU(), BertLayerNorm(hid_dim * 2, eps=1e-12), nn.Linear(hid_dim * 2, num_answers))
        self.logit_fc.apply(self.lxrt_encoder.model.init_bert_weights)
        self.task = 'vqa'

    def forward(self, feat, pos, sent):
        """
        b -- batch_size, o -- object_number, f -- visual_feature_size

        :param feat: (b, o, f)
        :param pos:  (b, o, 4)
        :param sent: (b,) Type -- list of string
        :param leng: (b,) Type -- int numpy array
        :return: (b, num_answer) The logit of each answers.
        """
        x = self.lxrt_encoder(sent, (feat, pos), task=self.task)
        logit = self.logit_fc(x)
        return logit


class LoRALayer:

    def __init__(self, r: int, lora_alpha: int, lora_dropout: float, merge_weights: bool):
        self.r = r
        self.lora_alpha = lora_alpha
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        self.merged = False
        self.merge_weights = merge_weights


class Conv2d(nn.Conv2d, LoRALayer):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, r: int=0, lora_alpha: int=1, lora_dropout: float=0.0, merge_weights: bool=True, **kwargs):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        assert type(kernel_size) is int
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r * kernel_size, in_channels * kernel_size)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_channels * kernel_size, r * kernel_size)))
            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.Conv2d.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool=True):
        nn.Conv2d.train(self, mode)
        if self.merge_weights and self.merged:
            self.weight.data -= (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling
            self.merged = False

    def eval(self):
        nn.Conv2d.eval(self)
        if self.merge_weights and not self.merged:
            self.weight.data += (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            return F.conv2d(x, self.weight + (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return nn.Conv2d.forward(self, x)


class LastLevelMaxPool(nn.Module):
    """
    This module is used in the original FPN to generate a downsampled P6 feature from P5.
    """

    def __init__(self):
        super().__init__()
        self.num_levels = 1
        self.in_feature = 'p5'

    def forward(self, x):
        return [nn.functional.max_pool2d(x, kernel_size=1, stride=2, padding=0)]


class LastLevelP6P7(nn.Module):
    """
    This module is used in RetinaNet to generate extra layers, P6 and P7 from C5 feature.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.num_levels = 2
        self.in_feature = 'res5'
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)

    def forward(self, c5):
        p6 = self.p6(c5)
        p7 = self.p7(nn.functional.relu(p6))
        return [p6, p7]


def get_norm(norm, out_channels):
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {'BN': BatchNorm2d, 'GN': lambda channels: nn.GroupNorm(32, channels), 'nnSyncBN': nn.SyncBatchNorm, '': lambda x: x}[norm]
    return norm(out_channels)


class BasicStem(nn.Module):

    def __init__(self, in_channels=3, out_channels=64, norm='BN', caffe_maxpool=False):
        super().__init__()
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False, norm=get_norm(norm, out_channels))
        self.caffe_maxpool = caffe_maxpool

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu_(x)
        if self.caffe_maxpool:
            x = nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=0, ceil_mode=True)
        else:
            x = nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        return x

    @property
    def out_channels(self):
        return self.conv1.out_channels

    @property
    def stride(self):
        return 4


class ResNetBlockBase(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        return self


class BottleneckBlock(ResNetBlockBase):

    def __init__(self, in_channels, out_channels, bottleneck_channels, stride=1, num_groups=1, norm='BN', stride_in_1x1=False, dilation=1):
        super().__init__(in_channels, out_channels, stride)
        if in_channels != out_channels:
            self.shortcut = Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False, norm=get_norm(norm, out_channels))
        else:
            self.shortcut = None
        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)
        self.conv1 = Conv2d(in_channels, bottleneck_channels, kernel_size=1, stride=stride_1x1, bias=False, norm=get_norm(norm, bottleneck_channels))
        self.conv2 = Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=stride_3x3, padding=1 * dilation, bias=False, groups=num_groups, dilation=dilation, norm=get_norm(norm, bottleneck_channels))
        self.conv3 = Conv2d(bottleneck_channels, out_channels, kernel_size=1, bias=False, norm=get_norm(norm, out_channels))

    def forward(self, x):
        out = self.conv1(x)
        out = nn.functional.relu_(out)
        out = self.conv2(out)
        out = nn.functional.relu_(out)
        out = self.conv3(out)
        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x
        out += shortcut
        out = nn.functional.relu_(out)
        return out


class ShapeSpec(namedtuple('_ShapeSpec', ['channels', 'height', 'width', 'stride'])):

    def __new__(cls, *, channels=None, height=None, width=None, stride=None):
        return super().__new__(cls, channels, height, width, stride)


class Backbone(nn.Module, metaclass=ABCMeta):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self):
        pass

    @property
    def size_divisibility(self):
        """
        Some backbones require the input height and width to be divisible by a specific integer. This is
        typically true for encoder / decoder type networks with lateral connection (e.g., FPN) for which feature maps need to match
        dimension in the "bottom up" and "top down" paths. Set to 0 if no specific input size divisibility is required.
        """
        return 0

    def output_shape(self):
        return {name: ShapeSpec(channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]) for name in self._out_features}

    @property
    def out_features(self):
        """deprecated"""
        return self._out_features

    @property
    def out_feature_strides(self):
        """deprecated"""
        return {f: self._out_feature_strides[f] for f in self._out_features}

    @property
    def out_feature_channels(self):
        """deprecated"""
        return {f: self._out_feature_channels[f] for f in self._out_features}


class ResNet(Backbone):

    def __init__(self, stem, stages, num_classes=None, out_features=None):
        """
        Args:
            stem (nn.Module): a stem module
            stages (list[list[ResNetBlock]]): several (typically 4) stages, each contains multiple :class:`ResNetBlockBase`.
            num_classes (None or int): if None, will not perform classification.
            out_features (list[str]): name of the layers whose outputs should be returned in forward. Can be anything in:
            "stem", "linear", or "res2" ... If None, will return the output of the last layer.
        """
        super(ResNet, self).__init__()
        self.stem = stem
        self.num_classes = num_classes
        current_stride = self.stem.stride
        self._out_feature_strides = {'stem': current_stride}
        self._out_feature_channels = {'stem': self.stem.out_channels}
        self.stages_and_names = []
        for i, blocks in enumerate(stages):
            for block in blocks:
                assert isinstance(block, ResNetBlockBase), block
                curr_channels = block.out_channels
            stage = nn.Sequential(*blocks)
            name = 'res' + str(i + 2)
            self.add_module(name, stage)
            self.stages_and_names.append((stage, name))
            self._out_feature_strides[name] = current_stride = int(current_stride * np.prod([k.stride for k in blocks]))
            self._out_feature_channels[name] = blocks[-1].out_channels
        if num_classes is not None:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.linear = nn.Linear(curr_channels, num_classes)
            nn.init.normal_(self.linear.weight, stddev=0.01)
            name = 'linear'
        if out_features is None:
            out_features = [name]
        self._out_features = out_features
        assert len(self._out_features)
        children = [x[0] for x in self.named_children()]
        for out_feature in self._out_features:
            assert out_feature in children, 'Available children: {}'.format(', '.join(children))

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        if 'stem' in self._out_features:
            outputs['stem'] = x
        for stage, name in self.stages_and_names:
            x = stage(x)
            if name in self._out_features:
                outputs[name] = x
        if self.num_classes is not None:
            x = self.avgpool(x)
            x = self.linear(x)
            if 'linear' in self._out_features:
                outputs['linear'] = x
        return outputs

    def output_shape(self):
        return {name: ShapeSpec(channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]) for name in self._out_features}

    @staticmethod
    def make_stage(block_class, num_blocks, first_stride=None, *, in_channels, out_channels, **kwargs):
        """
        Usually, layers that produce the same feature map spatial size
        are defined as one "stage".
        Under such definition, stride_per_block[1:] should all be 1.
        """
        if first_stride is not None:
            assert 'stride' not in kwargs and 'stride_per_block' not in kwargs
            kwargs['stride_per_block'] = [first_stride] + [1] * (num_blocks - 1)
        blocks = []
        for i in range(num_blocks):
            curr_kwargs = {}
            for k, v in kwargs.items():
                if k.endswith('_per_block'):
                    assert len(v) == num_blocks, f"Argument '{k}' of make_stage should have the same length as num_blocks={num_blocks}."
                    newk = k[:-len('_per_block')]
                    assert newk not in kwargs, f'Cannot call make_stage with both {k} and {newk}!'
                    curr_kwargs[newk] = v[i]
                else:
                    curr_kwargs[k] = v
            blocks.append(block_class(in_channels=in_channels, out_channels=out_channels, **curr_kwargs))
            in_channels = out_channels
        return blocks


def assign_boxes_to_levels(box_lists: List[torch.Tensor], min_level: int, max_level: int, canonical_box_size: int, canonical_level: int):
    box_sizes = torch.sqrt(torch.cat([boxes.area() for boxes in box_lists]))
    level_assignments = torch.floor(canonical_level + torch.log2(box_sizes / canonical_box_size + 1e-08))
    level_assignments = torch.clamp(level_assignments, min=min_level, max=max_level)
    return level_assignments - min_level


def _fmt_box_list(box_tensor, batch_index: int):
    repeated_index = torch.full((len(box_tensor), 1), batch_index, dtype=box_tensor.dtype, device=box_tensor.device)
    return torch.cat((repeated_index, box_tensor), dim=1)


def convert_boxes_to_pooler_format(box_lists: List[torch.Tensor]):
    pooler_fmt_boxes = torch.cat([_fmt_box_list(box_list, i) for i, box_list in enumerate(box_lists)], dim=0)
    return pooler_fmt_boxes


class ROIPooler(nn.Module):
    """
    Region of interest feature map pooler that supports pooling from one or more
    feature maps.
    """

    def __init__(self, output_size, scales, sampling_ratio, canonical_box_size=224, canonical_level=4):
        super().__init__()
        min_level = -math.log2(scales[0])
        max_level = -math.log2(scales[-1])
        assert math.isclose(min_level, int(min_level)) and math.isclose(max_level, int(max_level))
        assert len(scales) == max_level - min_level + 1, 'not pyramid'
        assert 0 < min_level and min_level <= max_level
        if isinstance(output_size, int):
            output_size = output_size, output_size
        assert len(output_size) == 2 and isinstance(output_size[0], int) and isinstance(output_size[1], int)
        if len(scales) > 1:
            assert min_level <= canonical_level and canonical_level <= max_level
        assert canonical_box_size > 0
        self.output_size = output_size
        self.min_level = int(min_level)
        self.max_level = int(max_level)
        self.level_poolers = nn.ModuleList(RoIPool(output_size, spatial_scale=scale) for scale in scales)
        self.canonical_level = canonical_level
        self.canonical_box_size = canonical_box_size

    def forward(self, feature_maps, boxes):
        """
        Args:
            feature_maps: List[torch.Tensor(N,C,W,H)]
            box_lists: list[torch.Tensor])
        Returns:
            A tensor of shape(N*B, Channels, output_size, output_size)
        """
        x = [v for v in feature_maps.values()]
        num_level_assignments = len(self.level_poolers)
        assert len(x) == num_level_assignments and len(boxes) == x[0].size(0)
        pooler_fmt_boxes = convert_boxes_to_pooler_format(boxes)
        if num_level_assignments == 1:
            return self.level_poolers[0](x[0], pooler_fmt_boxes)
        level_assignments = assign_boxes_to_levels(boxes, self.min_level, self.max_level, self.canonical_box_size, self.canonical_level)
        num_boxes = len(pooler_fmt_boxes)
        num_channels = x[0].shape[1]
        output_size = self.output_size[0]
        dtype, device = x[0].dtype, x[0].device
        output = torch.zeros((num_boxes, num_channels, output_size, output_size), dtype=dtype, device=device)
        for level, (x_level, pooler) in enumerate(zip(x, self.level_poolers)):
            inds = torch.nonzero(level_assignments == level).squeeze(1)
            pooler_fmt_boxes_level = pooler_fmt_boxes[inds]
            output[inds] = pooler(x_level, pooler_fmt_boxes_level)
        return output


class FastRCNNOutputLayers(nn.Module):
    """
    Two linear layers for predicting Fast R-CNN outputs:
      (1) proposal-to-detection box regression deltas
      (2) classification scores
    """

    def __init__(self, input_size, num_classes, cls_agnostic_bbox_reg, box_dim=4, use_attr=False, num_attrs=-1):
        """
        Args:
            input_size (int): channels, or (channels, height, width)
            num_classes (int)
            cls_agnostic_bbox_reg (bool)
            box_dim (int)
        """
        super().__init__()
        if not isinstance(input_size, int):
            input_size = np.prod(input_size)
        self.cls_score = nn.Linear(input_size, num_classes + 1)
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        self.bbox_pred = nn.Linear(input_size, num_bbox_reg_classes * box_dim)
        self.use_attr = use_attr
        if use_attr:
            """
            Modifications for VG in RoI heads
            Embedding: {num_classes + 1} --> {input_size // 8}
            Linear: {input_size + input_size // 8} --> {input_size // 4}
            Linear: {input_size // 4} --> {num_attrs + 1}
            """
            self.cls_embedding = nn.Embedding(num_classes + 1, input_size // 8)
            self.fc_attr = nn.Linear(input_size + input_size // 8, input_size // 4)
            self.attr_score = nn.Linear(input_size // 4, num_attrs + 1)
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for item in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(item.bias, 0)

    def forward(self, roi_features):
        if roi_features.dim() > 2:
            roi_features = torch.flatten(roi_features, start_dim=1)
        scores = self.cls_score(roi_features)
        proposal_deltas = self.bbox_pred(roi_features)
        if self.use_attr:
            _, max_class = scores.max(-1)
            cls_emb = self.cls_embedding(max_class)
            roi_features = torch.cat([roi_features, cls_emb], -1)
            roi_features = self.fc_attr(roi_features)
            roi_features = nn.functional.relu(roi_features)
            attr_scores = self.attr_score(roi_features)
            return scores, attr_scores, proposal_deltas
        else:
            return scores, proposal_deltas


class Res5ROIHeads(nn.Module):
    """
    ROIHeads perform all per-region computation in an R-CNN.
    It contains logic of cropping the regions, extract per-region features
    (by the res-5 block in this case), and make per-region predictions.
    """

    def __init__(self, cfg, input_shape):
        super().__init__()
        self.batch_size_per_image = cfg.RPN.BATCH_SIZE_PER_IMAGE
        self.positive_sample_fraction = cfg.ROI_HEADS.POSITIVE_FRACTION
        self.in_features = cfg.ROI_HEADS.IN_FEATURES
        self.num_classes = cfg.ROI_HEADS.NUM_CLASSES
        self.proposal_append_gt = cfg.ROI_HEADS.PROPOSAL_APPEND_GT
        self.feature_strides = {k: v.stride for k, v in input_shape.items()}
        self.feature_channels = {k: v.channels for k, v in input_shape.items()}
        self.cls_agnostic_bbox_reg = cfg.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG
        self.stage_channel_factor = 2 ** 3
        self.out_channels = cfg.RESNETS.RES2_OUT_CHANNELS * self.stage_channel_factor
        pooler_resolution = cfg.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales = 1.0 / self.feature_strides[self.in_features[0]],
        sampling_ratio = cfg.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        res5_halve = cfg.ROI_BOX_HEAD.RES5HALVE
        use_attr = cfg.ROI_BOX_HEAD.ATTR
        num_attrs = cfg.ROI_BOX_HEAD.NUM_ATTRS
        self.pooler = ROIPooler(output_size=pooler_resolution, scales=pooler_scales, sampling_ratio=sampling_ratio)
        self.res5 = self._build_res5_block(cfg)
        if not res5_halve:
            """
            Modifications for VG in RoI heads:
            1. Change the stride of conv1 and shortcut in Res5.Block1 from 2 to 1
            2. Modifying all conv2 with (padding: 1 --> 2) and (dilation: 1 --> 2)
            """
            self.res5[0].conv1.stride = 1, 1
            self.res5[0].shortcut.stride = 1, 1
            for i in range(3):
                self.res5[i].conv2.padding = 2, 2
                self.res5[i].conv2.dilation = 2, 2
        self.box_predictor = FastRCNNOutputLayers(self.out_channels, self.num_classes, self.cls_agnostic_bbox_reg, use_attr=use_attr, num_attrs=num_attrs)

    def _build_res5_block(self, cfg):
        stage_channel_factor = self.stage_channel_factor
        num_groups = cfg.RESNETS.NUM_GROUPS
        width_per_group = cfg.RESNETS.WIDTH_PER_GROUP
        bottleneck_channels = num_groups * width_per_group * stage_channel_factor
        out_channels = self.out_channels
        stride_in_1x1 = cfg.RESNETS.STRIDE_IN_1X1
        norm = cfg.RESNETS.NORM
        blocks = ResNet.make_stage(BottleneckBlock, 3, first_stride=2, in_channels=out_channels // 2, bottleneck_channels=bottleneck_channels, out_channels=out_channels, num_groups=num_groups, norm=norm, stride_in_1x1=stride_in_1x1)
        return nn.Sequential(*blocks)

    def _shared_roi_transform(self, features, boxes):
        x = self.pooler(features, boxes)
        return self.res5(x)

    def forward(self, features, proposal_boxes, gt_boxes=None):
        if self.training:
            """
            see https://github.com/airsplay/py-bottom-up-attention/                    blob/master/detectron2/modeling/roi_heads/roi_heads.py
            """
            raise NotImplementedError()
        assert not proposal_boxes[0].requires_grad
        box_features = self._shared_roi_transform(features, proposal_boxes)
        feature_pooled = box_features.mean(dim=[2, 3])
        obj_logits, attr_logits, pred_proposal_deltas = self.box_predictor(feature_pooled)
        return obj_logits, attr_logits, pred_proposal_deltas, feature_pooled


def _create_grid_offsets(size: List[int], stride: int, offset: float, device):
    grid_height, grid_width = size
    shifts_x = torch.arange(offset * stride, grid_width * stride, step=stride, dtype=torch.float32, device=device)
    shifts_y = torch.arange(offset * stride, grid_height * stride, step=stride, dtype=torch.float32, device=device)
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    return shift_x, shift_y


class AnchorGenerator(nn.Module):
    """
    For a set of image sizes and feature maps, computes a set of anchors.
    """

    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()
        sizes = cfg.ANCHOR_GENERATOR.SIZES
        aspect_ratios = cfg.ANCHOR_GENERATOR.ASPECT_RATIOS
        self.strides = [x.stride for x in input_shape]
        self.offset = cfg.ANCHOR_GENERATOR.OFFSET
        assert 0.0 <= self.offset < 1.0, self.offset
        """
        sizes (list[list[int]]): sizes[i] is the list of anchor sizes for feat map i
            1. given in absolute lengths in units of the input image;
            2. they do not dynamically scale if the input image size changes.
        aspect_ratios (list[list[float]])
        strides (list[int]): stride of each input feature.
        """
        self.num_features = len(self.strides)
        self.cell_anchors = nn.ParameterList(self._calculate_anchors(sizes, aspect_ratios))
        self._spacial_feat_dim = 4

    def _calculate_anchors(self, sizes, aspect_ratios):
        if len(sizes) == 1:
            sizes *= self.num_features
        if len(aspect_ratios) == 1:
            aspect_ratios *= self.num_features
        assert self.num_features == len(sizes)
        assert self.num_features == len(aspect_ratios)
        cell_anchors = [self.generate_cell_anchors(s, a).float() for s, a in zip(sizes, aspect_ratios)]
        return cell_anchors

    @property
    def box_dim(self):
        return self._spacial_feat_dim

    @property
    def num_cell_anchors(self):
        """
        Returns:
            list[int]: Each int is the number of anchors at every pixel location, on that feature map.
        """
        return [len(cell_anchors) for cell_anchors in self.cell_anchors]

    def grid_anchors(self, grid_sizes):
        anchors = []
        for size, stride, base_anchors in zip(grid_sizes, self.strides, self.cell_anchors):
            shift_x, shift_y = _create_grid_offsets(size, stride, self.offset, base_anchors.device)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)
            anchors.append((shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4))
        return anchors

    def generate_cell_anchors(self, sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1, 2)):
        """
        anchors are continuous geometric rectangles
        centered on one feature map point sample.
        We can later build the set of anchors
        for the entire feature map by tiling these tensors
        """
        anchors = []
        for size in sizes:
            area = size ** 2.0
            for aspect_ratio in aspect_ratios:
                w = math.sqrt(area / aspect_ratio)
                h = aspect_ratio * w
                x0, y0, x1, y1 = -w / 2.0, -h / 2.0, w / 2.0, h / 2.0
                anchors.append([x0, y0, x1, y1])
        return nn.Parameter(torch.tensor(anchors))

    def forward(self, features):
        """
        Args:
            features List[torch.Tensor]: list of feature maps on which to generate anchors.
        Returns:
            torch.Tensor: a list of #image elements.
        """
        num_images = features[0].size(0)
        grid_sizes = [feature_map.shape[-2:] for feature_map in features]
        anchors_over_all_feature_maps = self.grid_anchors(grid_sizes)
        anchors_over_all_feature_maps = torch.stack(anchors_over_all_feature_maps)
        return anchors_over_all_feature_maps.unsqueeze(0).repeat_interleave(num_images, dim=0)


class RPNHead(nn.Module):
    """
    RPN classification and regression heads. Uses a 3x3 conv to produce a shared
    hidden state from which one 1x1 conv predicts objectness logits for each anchor
    and a second 1x1 conv predicts bounding-box deltas specifying how to deform
    each anchor into an object proposal.
    """

    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()
        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, 'Each level must have the same channel!'
        in_channels = in_channels[0]
        anchor_generator = AnchorGenerator(cfg, input_shape)
        num_cell_anchors = anchor_generator.num_cell_anchors
        box_dim = anchor_generator.box_dim
        assert len(set(num_cell_anchors)) == 1, 'Each level must have the same number of cell anchors'
        num_cell_anchors = num_cell_anchors[0]
        if cfg.PROPOSAL_GENERATOR.HIDDEN_CHANNELS == -1:
            hid_channels = in_channels
        else:
            hid_channels = cfg.PROPOSAL_GENERATOR.HIDDEN_CHANNELS
        self.conv = nn.Conv2d(in_channels, hid_channels, kernel_size=3, stride=1, padding=1)
        self.objectness_logits = nn.Conv2d(hid_channels, num_cell_anchors, kernel_size=1, stride=1)
        self.anchor_deltas = nn.Conv2d(hid_channels, num_cell_anchors * box_dim, kernel_size=1, stride=1)
        for layer in [self.conv, self.objectness_logits, self.anchor_deltas]:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)

    def forward(self, features):
        """
        Args:
            features (list[Tensor]): list of feature maps
        """
        pred_objectness_logits = []
        pred_anchor_deltas = []
        for x in features:
            t = nn.functional.relu(self.conv(x))
            pred_objectness_logits.append(self.objectness_logits(t))
            pred_anchor_deltas.append(self.anchor_deltas(t))
        return pred_objectness_logits, pred_anchor_deltas


class Box2BoxTransform(object):
    """
    This R-CNN transformation scales the box's width and height
    by exp(dw), exp(dh) and shifts a box's center by the offset
    (dx * width, dy * height).
    """

    def __init__(self, weights: Tuple[float, float, float, float], scale_clamp: float=None):
        """
        Args:
            weights (4-element tuple): Scaling factors that are applied to the
                (dx, dy, dw, dh) deltas. In Fast R-CNN, these were originally set
                such that the deltas have unit variance; now they are treated as
                hyperparameters of the system.
            scale_clamp (float): When predicting deltas, the predicted box scaling
                factors (dw and dh) are clamped such that they are <= scale_clamp.
        """
        self.weights = weights
        if scale_clamp is not None:
            self.scale_clamp = scale_clamp
        else:
            """
            Value for clamping large dw and dh predictions.
            The heuristic is that we clamp such that dw and dh are no larger
            than what would transform a 16px box into a 1000px box
            (based on a small anchor, 16px, and a typical image size, 1000px).
            """
            self.scale_clamp = math.log(1000.0 / 16)

    def get_deltas(self, src_boxes, target_boxes):
        """
        Get box regression transformation deltas (dx, dy, dw, dh) that can be used
        to transform the `src_boxes` into the `target_boxes`. That is, the relation
        ``target_boxes == self.apply_deltas(deltas, src_boxes)`` is true (unless
        any delta is too large and is clamped).
        Args:
            src_boxes (Tensor): source boxes, e.g., object proposals
            target_boxes (Tensor): target of the transformation, e.g., ground-truth
                boxes.
        """
        assert isinstance(src_boxes, torch.Tensor), type(src_boxes)
        assert isinstance(target_boxes, torch.Tensor), type(target_boxes)
        src_widths = src_boxes[:, 2] - src_boxes[:, 0]
        src_heights = src_boxes[:, 3] - src_boxes[:, 1]
        src_ctr_x = src_boxes[:, 0] + 0.5 * src_widths
        src_ctr_y = src_boxes[:, 1] + 0.5 * src_heights
        target_widths = target_boxes[:, 2] - target_boxes[:, 0]
        target_heights = target_boxes[:, 3] - target_boxes[:, 1]
        target_ctr_x = target_boxes[:, 0] + 0.5 * target_widths
        target_ctr_y = target_boxes[:, 1] + 0.5 * target_heights
        wx, wy, ww, wh = self.weights
        dx = wx * (target_ctr_x - src_ctr_x) / src_widths
        dy = wy * (target_ctr_y - src_ctr_y) / src_heights
        dw = ww * torch.log(target_widths / src_widths)
        dh = wh * torch.log(target_heights / src_heights)
        deltas = torch.stack((dx, dy, dw, dh), dim=1)
        assert (src_widths > 0).all().item(), 'Input boxes to Box2BoxTransform are not valid!'
        return deltas

    def apply_deltas(self, deltas, boxes):
        """
        Apply transformation `deltas` (dx, dy, dw, dh) to `boxes`.
        Args:
            deltas (Tensor): transformation deltas of shape (N, k*4), where k >= 1.
                deltas[i] represents k potentially different class-specific
                box transformations for the single box boxes[i].
            boxes (Tensor): boxes to transform, of shape (N, 4)
        """
        boxes = boxes
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights
        wx, wy, ww, wh = self.weights
        dx = deltas[:, 0::4] / wx
        dy = deltas[:, 1::4] / wy
        dw = deltas[:, 2::4] / ww
        dh = deltas[:, 3::4] / wh
        dw = torch.clamp(dw, max=self.scale_clamp)
        dh = torch.clamp(dh, max=self.scale_clamp)
        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]
        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h
        return pred_boxes


class Matcher(object):
    """
    This class assigns to each predicted "element" (e.g., a box) a ground-truth
    element. Each predicted element will have exactly zero or one matches; each
    ground-truth element may be matched to zero or more predicted elements.
    The matching is determined by the MxN match_quality_matrix, that characterizes
    how well each (ground-truth, prediction)-pair match each other. For example,
    if the elements are boxes, this matrix may contain box intersection-over-union
    overlap values.
    The matcher returns (a) a vector of length N containing the index of the
    ground-truth element m in [0, M) that matches to prediction n in [0, N).
    (b) a vector of length N containing the labels for each prediction.
    """

    def __init__(self, thresholds: List[float], labels: List[int], allow_low_quality_matches: bool=False):
        """
        Args:
            thresholds (list): a list of thresholds used to stratify predictions
                into levels.
            labels (list): a list of values to label predictions belonging at
                each level. A label can be one of {-1, 0, 1} signifying
                {ignore, negative class, positive class}, respectively.
            allow_low_quality_matches (bool): if True, produce additional matches or predictions with maximum match quality lower than high_threshold.
                For example, thresholds = [0.3, 0.5] labels = [0, -1, 1] All predictions with iou < 0.3 will be marked with 0 and
                thus will be considered as false positives while training. All predictions with 0.3 <= iou < 0.5 will be marked with -1 and
                thus will be ignored. All predictions with 0.5 <= iou will be marked with 1 and thus will be considered as true positives.
        """
        thresholds = thresholds[:]
        assert thresholds[0] > 0
        thresholds.insert(0, -float('inf'))
        thresholds.append(float('inf'))
        assert all([(low <= high) for low, high in zip(thresholds[:-1], thresholds[1:])])
        assert all([(label_i in [-1, 0, 1]) for label_i in labels])
        assert len(labels) == len(thresholds) - 1
        self.thresholds = thresholds
        self.labels = labels
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, match_quality_matrix):
        """
        Args:
            match_quality_matrix (Tensor[float]): an MxN tensor, containing the pairwise quality between M ground-truth elements and N predicted
                elements. All elements must be >= 0 (due to the us of `torch.nonzero` for selecting indices in :meth:`set_low_quality_matches_`).
        Returns:
            matches (Tensor[int64]): a vector of length N, where matches[i] is a matched ground-truth index in [0, M)
            match_labels (Tensor[int8]): a vector of length N, where pred_labels[i] indicates true or false positive or ignored
        """
        assert match_quality_matrix.dim() == 2
        if match_quality_matrix.numel() == 0:
            default_matches = match_quality_matrix.new_full((match_quality_matrix.size(1),), 0, dtype=torch.int64)
            default_match_labels = match_quality_matrix.new_full((match_quality_matrix.size(1),), self.labels[0], dtype=torch.int8)
            return default_matches, default_match_labels
        assert torch.all(match_quality_matrix >= 0)
        matched_vals, matches = match_quality_matrix.max(dim=0)
        match_labels = matches.new_full(matches.size(), 1, dtype=torch.int8)
        for l, low, high in zip(self.labels, self.thresholds[:-1], self.thresholds[1:]):
            low_high = (matched_vals >= low) & (matched_vals < high)
            match_labels[low_high] = l
        if self.allow_low_quality_matches:
            self.set_low_quality_matches_(match_labels, match_quality_matrix)
        return matches, match_labels

    def set_low_quality_matches_(self, match_labels, match_quality_matrix):
        """
        Produce additional matches for predictions that have only low-quality matches.
        Specifically, for each ground-truth G find the set of predictions that have
        maximum overlap with it (including ties); for each prediction in that set, if
        it is unmatched, then match it to the ground-truth G.
        This function implements the RPN assignment case (i)
        in Sec. 3.1.2 of Faster R-CNN.
        """
        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)
        of_quality_inds = match_quality_matrix == highest_quality_foreach_gt[:, None]
        if of_quality_inds.dim() == 0:
            _, pred_inds_with_highest_quality = of_quality_inds.unsqueeze(0).nonzero().unbind(1)
        else:
            _, pred_inds_with_highest_quality = of_quality_inds.nonzero().unbind(1)
        match_labels[pred_inds_with_highest_quality] = 1


class RPNOutputs(object):

    def __init__(self, box2box_transform, anchor_matcher, batch_size_per_image, positive_fraction, images, pred_objectness_logits, pred_anchor_deltas, anchors, boundary_threshold=0, gt_boxes=None, smooth_l1_beta=0.0):
        """
        Args:
            box2box_transform (Box2BoxTransform): :class:`Box2BoxTransform` instance for anchor-proposal transformations.
            anchor_matcher (Matcher): :class:`Matcher` instance for matching anchors to ground-truth boxes; used to determine training labels.
            batch_size_per_image (int): number of proposals to sample when training
            positive_fraction (float): target fraction of sampled proposals that should be positive
            images (ImageList): :class:`ImageList` instance representing N input images
            pred_objectness_logits (list[Tensor]): A list of L elements. Element i is a tensor of shape (N, A, Hi, W)
            pred_anchor_deltas (list[Tensor]): A list of L elements. Element i is a tensor of shape (N, A*4, Hi, Wi)
            anchors (list[torch.Tensor]): nested list of boxes. anchors[i][j] at (n, l) stores anchor array for feature map l
            boundary_threshold (int): if >= 0, then anchors that extend beyond the image boundary by more than boundary_thresh are not used in training.
            gt_boxes (list[Boxes], optional): A list of N elements.
            smooth_l1_beta (float): The transition point between L1 and L2 lossn. When set to 0, the loss becomes L1. When +inf, it is ignored
        """
        self.box2box_transform = box2box_transform
        self.anchor_matcher = anchor_matcher
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        self.pred_objectness_logits = pred_objectness_logits
        self.pred_anchor_deltas = pred_anchor_deltas
        self.anchors = anchors
        self.gt_boxes = gt_boxes
        self.num_feature_maps = len(pred_objectness_logits)
        self.num_images = len(images)
        self.boundary_threshold = boundary_threshold
        self.smooth_l1_beta = smooth_l1_beta

    def _get_ground_truth(self):
        raise NotImplementedError()

    def predict_proposals(self):
        proposals = []
        anchors = self.anchors.transpose(0, 1)
        for anchors_i, pred_anchor_deltas_i in zip(anchors, self.pred_anchor_deltas):
            B = anchors_i.size(-1)
            N, _, Hi, Wi = pred_anchor_deltas_i.shape
            anchors_i = anchors_i.flatten(start_dim=0, end_dim=1)
            pred_anchor_deltas_i = pred_anchor_deltas_i.view(N, -1, B, Hi, Wi).permute(0, 3, 4, 1, 2).reshape(-1, B)
            proposals_i = self.box2box_transform.apply_deltas(pred_anchor_deltas_i, anchors_i)
            proposals.append(proposals_i.view(N, -1, B))
        proposals = torch.stack(proposals)
        return proposals

    def predict_objectness_logits(self):
        """
        Returns:
            pred_objectness_logits (list[Tensor]) -> (N, Hi*Wi*A).
        """
        pred_objectness_logits = [score.permute(0, 2, 3, 1).reshape(self.num_images, -1) for score in self.pred_objectness_logits]
        return pred_objectness_logits


def _clip_box(tensor, box_size: Tuple[int, int]):
    assert torch.isfinite(tensor).all(), 'Box tensor contains infinite or NaN!'
    h, w = box_size
    tensor[:, 0].clamp_(min=0, max=w)
    tensor[:, 1].clamp_(min=0, max=h)
    tensor[:, 2].clamp_(min=0, max=w)
    tensor[:, 3].clamp_(min=0, max=h)


def _nonempty_boxes(box, threshold: float=0.0) ->torch.Tensor:
    widths = box[:, 2] - box[:, 0]
    heights = box[:, 3] - box[:, 1]
    keep = (widths > threshold) & (heights > threshold)
    return keep


def find_top_rpn_proposals(proposals, pred_objectness_logits, images, image_sizes, nms_thresh, pre_nms_topk, post_nms_topk, min_box_side_len, training):
    """Args:
        proposals (list[Tensor]): (L, N, Hi*Wi*A, 4).
        pred_objectness_logits: tensors of length L.
        nms_thresh (float): IoU threshold to use for NMS
        pre_nms_topk (int): before nms
        post_nms_topk (int): after nms
        min_box_side_len (float): minimum proposal box side
        training (bool): True if proposals are to be used in training,
    Returns:
        results (List[Dict]): stores post_nms_topk object proposals for image i.
    """
    num_images = len(images)
    device = proposals[0].device
    topk_scores = []
    topk_proposals = []
    level_ids = []
    batch_idx = torch.arange(num_images, device=device)
    for level_id, proposals_i, logits_i in zip(itertools.count(), proposals, pred_objectness_logits):
        Hi_Wi_A = logits_i.shape[1]
        num_proposals_i = min(pre_nms_topk, Hi_Wi_A)
        logits_i, idx = logits_i.sort(descending=True, dim=1)
        topk_scores_i = logits_i[batch_idx, :num_proposals_i]
        topk_idx = idx[batch_idx, :num_proposals_i]
        topk_proposals_i = proposals_i[batch_idx[:, None], topk_idx]
        topk_proposals.append(topk_proposals_i)
        topk_scores.append(topk_scores_i)
        level_ids.append(torch.full((num_proposals_i,), level_id, dtype=torch.int64, device=device))
    topk_scores = torch.cat(topk_scores, dim=1)
    topk_proposals = torch.cat(topk_proposals, dim=1)
    level_ids = torch.cat(level_ids, dim=0)
    results = []
    for n, image_size in enumerate(image_sizes):
        boxes = topk_proposals[n]
        scores_per_img = topk_scores[n]
        _clip_box(boxes, image_size)
        keep = _nonempty_boxes(boxes, threshold=min_box_side_len)
        lvl = level_ids
        if keep.sum().item() != len(boxes):
            boxes, scores_per_img, lvl = boxes[keep], scores_per_img[keep], level_ids[keep]
        keep = batched_nms(boxes, scores_per_img, lvl, nms_thresh)
        keep = keep[:post_nms_topk]
        res = boxes[keep], scores_per_img[keep]
        results.append(res)
    return results


class RPN(nn.Module):
    """
    Region Proposal Network, introduced by the Faster R-CNN paper.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()
        self.min_box_side_len = cfg.PROPOSAL_GENERATOR.MIN_SIZE
        self.in_features = cfg.RPN.IN_FEATURES
        self.nms_thresh = cfg.RPN.NMS_THRESH
        self.batch_size_per_image = cfg.RPN.BATCH_SIZE_PER_IMAGE
        self.positive_fraction = cfg.RPN.POSITIVE_FRACTION
        self.smooth_l1_beta = cfg.RPN.SMOOTH_L1_BETA
        self.loss_weight = cfg.RPN.LOSS_WEIGHT
        self.pre_nms_topk = {(True): cfg.RPN.PRE_NMS_TOPK_TRAIN, (False): cfg.RPN.PRE_NMS_TOPK_TEST}
        self.post_nms_topk = {(True): cfg.RPN.POST_NMS_TOPK_TRAIN, (False): cfg.RPN.POST_NMS_TOPK_TEST}
        self.boundary_threshold = cfg.RPN.BOUNDARY_THRESH
        self.anchor_generator = AnchorGenerator(cfg, [input_shape[f] for f in self.in_features])
        self.box2box_transform = Box2BoxTransform(weights=cfg.RPN.BBOX_REG_WEIGHTS)
        self.anchor_matcher = Matcher(cfg.RPN.IOU_THRESHOLDS, cfg.RPN.IOU_LABELS, allow_low_quality_matches=True)
        self.rpn_head = RPNHead(cfg, [input_shape[f] for f in self.in_features])

    def training(self, images, image_shapes, features, gt_boxes):
        pass

    def inference(self, outputs, images, image_shapes, features, gt_boxes=None):
        outputs = find_top_rpn_proposals(outputs.predict_proposals(), outputs.predict_objectness_logits(), images, image_shapes, self.nms_thresh, self.pre_nms_topk[self.training], self.post_nms_topk[self.training], self.min_box_side_len, self.training)
        results = []
        for img in outputs:
            im_boxes, img_box_logits = img
            img_box_logits, inds = img_box_logits.sort(descending=True)
            im_boxes = im_boxes[inds]
            results.append((im_boxes, img_box_logits))
        proposal_boxes, logits = tuple(map(list, zip(*results)))
        return proposal_boxes, logits

    def forward(self, images, image_shapes, features, gt_boxes=None):
        """
        Args:
            images (torch.Tensor): input images of length `N`
            features (dict[str: Tensor])
            gt_instances
        """
        features = [features[f] for f in self.in_features]
        pred_objectness_logits, pred_anchor_deltas = self.rpn_head(features)
        anchors = self.anchor_generator(features)
        outputs = RPNOutputs(self.box2box_transform, self.anchor_matcher, self.batch_size_per_image, self.positive_fraction, images, pred_objectness_logits, pred_anchor_deltas, anchors, self.boundary_threshold, gt_boxes, self.smooth_l1_beta)
        if self.training:
            raise NotImplementedError()
            return self.training(outputs, images, image_shapes, features, gt_boxes)
        else:
            return self.inference(outputs, images, image_shapes, features, gt_boxes)


class Config(object):

    def __init__(self, **kwargs):
        """Configuration Class: set kwargs as class attributes with setattr"""
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def config_str(self):
        return pprint.pformat(self.__dict__)

    def __repr__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += self.config_str
        return config_str

    def save(self, path):
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            kwargs = yaml.load(f)
        return Config(**kwargs)


def do_nms(boxes, scores, image_shape, score_thresh, nms_thresh, mind, maxd):
    scores = scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    boxes = boxes.reshape(-1, 4)
    _clip_box(boxes, image_shape)
    boxes = boxes.view(-1, num_bbox_reg_classes, 4)
    max_scores, max_classes = scores.max(1)
    num_objs = boxes.size(0)
    boxes = boxes.view(-1, 4)
    idxs = torch.arange(num_objs) * num_bbox_reg_classes + max_classes
    max_boxes = boxes[idxs]
    keep = nms(max_boxes, max_scores, nms_thresh)
    keep = keep[:maxd]
    if keep.shape[-1] >= mind and keep.shape[-1] <= maxd:
        max_boxes, max_scores = max_boxes[keep], max_scores[keep]
        classes = max_classes[keep]
        return max_boxes, max_scores, classes, keep
    else:
        return None


class ROIOutputs(object):

    def __init__(self, cfg, training=False):
        self.smooth_l1_beta = cfg.ROI_BOX_HEAD.SMOOTH_L1_BETA
        self.box2box_transform = Box2BoxTransform(weights=cfg.ROI_BOX_HEAD.BBOX_REG_WEIGHTS)
        self.training = training
        self.score_thresh = cfg.ROI_HEADS.SCORE_THRESH_TEST
        self.min_detections = cfg.MIN_DETECTIONS
        self.max_detections = cfg.MAX_DETECTIONS
        nms_thresh = cfg.ROI_HEADS.NMS_THRESH_TEST
        if not isinstance(nms_thresh, list):
            nms_thresh = [nms_thresh]
        self.nms_thresh = nms_thresh

    def _predict_boxes(self, proposals, box_deltas, preds_per_image):
        num_pred = box_deltas.size(0)
        B = proposals[0].size(-1)
        K = box_deltas.size(-1) // B
        box_deltas = box_deltas.view(num_pred * K, B)
        proposals = torch.cat(proposals, dim=0).unsqueeze(-2).expand(num_pred, K, B)
        proposals = proposals.reshape(-1, B)
        boxes = self.box2box_transform.apply_deltas(box_deltas, proposals)
        return boxes.view(num_pred, K * B).split(preds_per_image, dim=0)

    def _predict_objs(self, obj_logits, preds_per_image):
        probs = nn.functional.softmax(obj_logits, dim=-1)
        probs = probs.split(preds_per_image, dim=0)
        return probs

    def _predict_attrs(self, attr_logits, preds_per_image):
        attr_logits = attr_logits[..., :-1].softmax(-1)
        attr_probs, attrs = attr_logits.max(-1)
        return attr_probs.split(preds_per_image, dim=0), attrs.split(preds_per_image, dim=0)

    @torch.no_grad()
    def inference(self, obj_logits, attr_logits, box_deltas, pred_boxes, features, sizes, scales=None):
        preds_per_image = [p.size(0) for p in pred_boxes]
        boxes_all = self._predict_boxes(pred_boxes, box_deltas, preds_per_image)
        obj_scores_all = self._predict_objs(obj_logits, preds_per_image)
        attr_probs_all, attrs_all = self._predict_attrs(attr_logits, preds_per_image)
        features = features.split(preds_per_image, dim=0)
        final_results = []
        zipped = zip(boxes_all, obj_scores_all, attr_probs_all, attrs_all, sizes)
        for i, (boxes, obj_scores, attr_probs, attrs, size) in enumerate(zipped):
            for nms_t in self.nms_thresh:
                outputs = do_nms(boxes, obj_scores, size, self.score_thresh, nms_t, self.min_detections, self.max_detections)
                if outputs is not None:
                    max_boxes, max_scores, classes, ids = outputs
                    break
            if scales is not None:
                scale_yx = scales[i]
                max_boxes[:, 0::2] *= scale_yx[1]
                max_boxes[:, 1::2] *= scale_yx[0]
            final_results.append((max_boxes, classes, max_scores, attrs[ids], attr_probs[ids], features[i][ids]))
        boxes, classes, class_probs, attrs, attr_probs, roi_features = map(list, zip(*final_results))
        return boxes, classes, class_probs, attrs, attr_probs, roi_features

    def training(self, obj_logits, attr_logits, box_deltas, pred_boxes, features, sizes):
        pass

    def __call__(self, obj_logits, attr_logits, box_deltas, pred_boxes, features, sizes, scales=None):
        if self.training:
            raise NotImplementedError()
        return self.inference(obj_logits, attr_logits, box_deltas, pred_boxes, features, sizes, scales=scales)


def build_backbone(cfg):
    input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))
    norm = cfg.RESNETS.NORM
    stem = BasicStem(in_channels=input_shape.channels, out_channels=cfg.RESNETS.STEM_OUT_CHANNELS, norm=norm, caffe_maxpool=cfg.MODEL.MAX_POOL)
    freeze_at = cfg.BACKBONE.FREEZE_AT
    if freeze_at >= 1:
        for p in stem.parameters():
            p.requires_grad = False
    out_features = cfg.RESNETS.OUT_FEATURES
    depth = cfg.RESNETS.DEPTH
    num_groups = cfg.RESNETS.NUM_GROUPS
    width_per_group = cfg.RESNETS.WIDTH_PER_GROUP
    bottleneck_channels = num_groups * width_per_group
    in_channels = cfg.RESNETS.STEM_OUT_CHANNELS
    out_channels = cfg.RESNETS.RES2_OUT_CHANNELS
    stride_in_1x1 = cfg.RESNETS.STRIDE_IN_1X1
    res5_dilation = cfg.RESNETS.RES5_DILATION
    assert res5_dilation in {1, 2}, 'res5_dilation cannot be {}.'.format(res5_dilation)
    num_blocks_per_stage = {(50): [3, 4, 6, 3], (101): [3, 4, 23, 3], (152): [3, 8, 36, 3]}[depth]
    stages = []
    out_stage_idx = [{'res2': 2, 'res3': 3, 'res4': 4, 'res5': 5}[f] for f in out_features]
    max_stage_idx = max(out_stage_idx)
    for idx, stage_idx in enumerate(range(2, max_stage_idx + 1)):
        dilation = res5_dilation if stage_idx == 5 else 1
        first_stride = 1 if idx == 0 or stage_idx == 5 and dilation == 2 else 2
        stage_kargs = {'num_blocks': num_blocks_per_stage[idx], 'first_stride': first_stride, 'in_channels': in_channels, 'bottleneck_channels': bottleneck_channels, 'out_channels': out_channels, 'num_groups': num_groups, 'norm': norm, 'stride_in_1x1': stride_in_1x1, 'dilation': dilation}
        stage_kargs['block_class'] = BottleneckBlock
        blocks = ResNet.make_stage(**stage_kargs)
        in_channels = out_channels
        out_channels *= 2
        bottleneck_channels *= 2
        if freeze_at >= stage_idx:
            for block in blocks:
                block.freeze()
        stages.append(blocks)
    return ResNet(stem, stages, out_features=out_features)


CLOUDFRONT_DISTRIB_PREFIX = 'https://cdn.huggingface.co'


S3_BUCKET_PREFIX = 'https://s3.amazonaws.com/models.huggingface.co/bert'


def hf_bucket_url(model_id: str, filename: str, use_cdn=True) ->str:
    endpoint = CLOUDFRONT_DISTRIB_PREFIX if use_cdn else S3_BUCKET_PREFIX
    legacy_format = '/' not in model_id
    if legacy_format:
        return f'{endpoint}/{model_id}-{filename}'
    else:
        return f'{endpoint}/{model_id}/{filename}'


def load_checkpoint(ckp):
    r = OrderedDict()
    with open(ckp, 'rb') as f:
        ckp = pkl.load(f)['model']
    for k in copy.deepcopy(list(ckp.keys())):
        v = ckp.pop(k)
        if isinstance(v, np.ndarray):
            v = torch.tensor(v)
        else:
            assert isinstance(v, torch.tensor), type(v)
        r[k] = v
    return r


def norm_box(boxes, raw_sizes):
    if not isinstance(boxes, torch.Tensor):
        normalized_boxes = boxes.copy()
    else:
        normalized_boxes = boxes.clone()
    normalized_boxes[:, :, (0, 2)] /= raw_sizes[:, 1]
    normalized_boxes[:, :, (1, 3)] /= raw_sizes[:, 0]
    return normalized_boxes


def pad_list_tensors(list_tensors, preds_per_image, max_detections=None, return_tensors=None, padding=None, pad_value=0, location=None):
    """
    location will always be cpu for np tensors
    """
    if location is None:
        location = 'cpu'
    assert return_tensors in {'pt', 'np', None}
    assert padding in {'max_detections', 'max_batch', None}
    new = []
    if padding is None:
        if return_tensors is None:
            return list_tensors
        elif return_tensors == 'pt':
            if not isinstance(list_tensors, torch.Tensor):
                return torch.stack(list_tensors)
            else:
                return list_tensors
        elif not isinstance(list_tensors, list):
            return np.array(list_tensors)
        else:
            return list_tensors
    if padding == 'max_detections':
        assert max_detections is not None, 'specify max number of detections per batch'
    elif padding == 'max_batch':
        max_detections = max(preds_per_image)
    for i in range(len(list_tensors)):
        too_small = False
        tensor_i = list_tensors.pop(0)
        if tensor_i.ndim < 2:
            too_small = True
            tensor_i = tensor_i.unsqueeze(-1)
        assert isinstance(tensor_i, torch.Tensor)
        tensor_i = nn.functional.pad(input=tensor_i, pad=(0, 0, 0, max_detections - preds_per_image[i]), mode='constant', value=pad_value)
        if too_small:
            tensor_i = tensor_i.squeeze(-1)
        if return_tensors is None:
            if location == 'cpu':
                tensor_i = tensor_i.cpu()
            tensor_i = tensor_i.tolist()
        if return_tensors == 'np':
            if location == 'cpu':
                tensor_i = tensor_i.cpu()
            tensor_i = tensor_i.numpy()
        elif location == 'cpu':
            tensor_i = tensor_i.cpu()
        new.append(tensor_i)
    if return_tensors == 'np':
        return np.stack(new, axis=0)
    elif return_tensors == 'pt' and not isinstance(new, torch.Tensor):
        return torch.stack(new, dim=0)
    else:
        return list_tensors


class GeneralizedRCNN(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone = build_backbone(cfg)
        self.proposal_generator = RPN(cfg, self.backbone.output_shape())
        self.roi_heads = Res5ROIHeads(cfg, self.backbone.output_shape())
        self.roi_outputs = ROIOutputs(cfg)
        self

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop('config', None)
        state_dict = kwargs.pop('state_dict', None)
        cache_dir = kwargs.pop('cache_dir', None)
        from_tf = kwargs.pop('from_tf', False)
        force_download = kwargs.pop('force_download', False)
        resume_download = kwargs.pop('resume_download', False)
        proxies = kwargs.pop('proxies', None)
        local_files_only = kwargs.pop('local_files_only', False)
        use_cdn = kwargs.pop('use_cdn', True)
        if not isinstance(config, Config):
            config_path = config if config is not None else pretrained_model_name_or_path
            config = Config.from_pretrained(config_path, cache_dir=cache_dir, force_download=force_download, resume_download=resume_download, proxies=proxies, local_files_only=local_files_only)
        if pretrained_model_name_or_path is not None:
            if os.path.isdir(pretrained_model_name_or_path):
                if os.path.isfile(os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)):
                    archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
                else:
                    raise EnvironmentError('Error no file named {} found in directory {} '.format(WEIGHTS_NAME, pretrained_model_name_or_path))
            elif os.path.isfile(pretrained_model_name_or_path) or is_remote_url(pretrained_model_name_or_path):
                archive_file = pretrained_model_name_or_path
            elif os.path.isfile(pretrained_model_name_or_path + '.index'):
                assert from_tf, 'We found a TensorFlow checkpoint at {}, please set from_tf to True to load from this checkpoint'.format(pretrained_model_name_or_path + '.index')
                archive_file = pretrained_model_name_or_path + '.index'
            else:
                archive_file = hf_bucket_url(pretrained_model_name_or_path, filename=WEIGHTS_NAME, use_cdn=use_cdn)
            try:
                resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir, force_download=force_download, proxies=proxies, resume_download=resume_download, local_files_only=local_files_only)
                if resolved_archive_file is None:
                    raise EnvironmentError
            except EnvironmentError:
                msg = f"Can't load weights for '{pretrained_model_name_or_path}'."
                raise EnvironmentError(msg)
            if resolved_archive_file == archive_file:
                None
            else:
                None
        else:
            resolved_archive_file = None
        model = cls(config)
        if state_dict is None:
            try:
                try:
                    state_dict = torch.load(resolved_archive_file, map_location='cpu')
                except Exception:
                    state_dict = load_checkpoint(resolved_archive_file)
            except Exception:
                raise OSError('Unable to load weights from pytorch checkpoint file. If you tried to load a PyTorch model from a TF 2.0 checkpoint, please set from_tf=True. ')
        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata
        model_to_load = model
        model_to_load.load_state_dict(state_dict)
        if model.__class__.__name__ != model_to_load.__class__.__name__:
            base_model_state_dict = model_to_load.state_dict().keys()
            head_model_state_dict_without_base_prefix = [key.split(cls.base_model_prefix + '.')[-1] for key in model.state_dict().keys()]
            missing_keys.extend(head_model_state_dict_without_base_prefix - base_model_state_dict)
        if len(unexpected_keys) > 0:
            None
        else:
            None
        if len(missing_keys) > 0:
            None
        else:
            None
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(model.__class__.__name__, '\n\t'.join(error_msgs)))
        model.eval()
        return model

    def forward(self, images, image_shapes, gt_boxes=None, proposals=None, scales_yx=None, **kwargs):
        """
        kwargs:
            max_detections (int), return_tensors {"np", "pt", None}, padding {None,
            "max_detections"}, pad_value (int), location = {"cuda", "cpu"}
        """
        if self.training:
            raise NotImplementedError()
        return self.inference(images=images, image_shapes=image_shapes, gt_boxes=gt_boxes, proposals=proposals, scales_yx=scales_yx, **kwargs)

    @torch.no_grad()
    def inference(self, images, image_shapes, gt_boxes=None, proposals=None, scales_yx=None, **kwargs):
        original_sizes = image_shapes * scales_yx
        features = self.backbone(images)
        if proposals is None:
            proposal_boxes, _ = self.proposal_generator(images, image_shapes, features, gt_boxes)
        else:
            assert proposals is not None
        obj_logits, attr_logits, box_deltas, feature_pooled = self.roi_heads(features, proposal_boxes, gt_boxes)
        boxes, classes, class_probs, attrs, attr_probs, roi_features = self.roi_outputs(obj_logits=obj_logits, attr_logits=attr_logits, box_deltas=box_deltas, pred_boxes=proposal_boxes, features=feature_pooled, sizes=image_shapes, scales=scales_yx)
        subset_kwargs = {'max_detections': kwargs.get('max_detections', None), 'return_tensors': kwargs.get('return_tensors', None), 'pad_value': kwargs.get('pad_value', 0), 'padding': kwargs.get('padding', None)}
        preds_per_image = torch.tensor([p.size(0) for p in boxes])
        boxes = pad_list_tensors(boxes, preds_per_image, **subset_kwargs)
        classes = pad_list_tensors(classes, preds_per_image, **subset_kwargs)
        class_probs = pad_list_tensors(class_probs, preds_per_image, **subset_kwargs)
        attrs = pad_list_tensors(attrs, preds_per_image, **subset_kwargs)
        attr_probs = pad_list_tensors(attr_probs, preds_per_image, **subset_kwargs)
        roi_features = pad_list_tensors(roi_features, preds_per_image, **subset_kwargs)
        subset_kwargs['padding'] = None
        preds_per_image = pad_list_tensors(preds_per_image, None, **subset_kwargs)
        sizes = pad_list_tensors(image_shapes, None, **subset_kwargs)
        normalized_boxes = norm_box(boxes, original_sizes)
        return OrderedDict({'obj_ids': classes, 'obj_probs': class_probs, 'attr_ids': attrs, 'attr_probs': attr_probs, 'boxes': boxes, 'sizes': sizes, 'preds_per_image': preds_per_image, 'roi_features': roi_features, 'normalized_boxes': normalized_boxes})


class LoRALinearController(nn.Linear, LoRALayer):
    """Implements Adapter controller module which controls the logics of
    putting adapter layers within transformer's layers."""

    def __init__(self, in_features: int, out_features: int, fan_in_fan_out: bool=False, config=None, **kwargs):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        self.tasks = config.tasks
        self.use_single_lora = config.use_single_lora
        r = config.lora_dim
        lora_alpha = config.lora_alpha
        lora_dropout = config.lora_dropout
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=True)
        self.fan_in_fan_out = fan_in_fan_out
        self.lora_As = nn.ParameterDict(dict())
        self.lora_Bs = nn.ParameterDict(dict())
        if r > 0:
            self.lora_As, self.lora_Bs = self.construct_lora_weights(self.tasks)
            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_As'):
            for task in self.tasks:
                nn.init.kaiming_uniform_(self.lora_As[task], a=math.sqrt(5))
                nn.init.zeros_(self.lora_Bs[task])

    def forward(self, x, task):

        def T(w):
            return w.T if self.fan_in_fan_out else w
        result = F.linear(x, T(self.weight), bias=self.bias)
        lora_A = self.lora_As[task]
        lora_B = self.lora_Bs[task]
        if self.training:
            result += self.lora_dropout(x) @ lora_A.T @ lora_B.T * self.scaling
        else:
            result += x @ lora_A.T @ lora_B.T * self.scaling
        return result

    def get_task(self, task):
        return task

    def construct_lora_weights(self, tasks):
        if self.use_single_lora:
            lora_A = nn.Parameter(self.weight.new_zeros((self.r, self.in_features)))
            lora_B = nn.Parameter(self.weight.new_zeros((self.out_features, self.r)))
            for task in tasks:
                self.lora_As[task] = lora_A
                self.lora_Bs[task] = lora_B
        else:
            for task in tasks:
                self.lora_As[task] = nn.Parameter(self.weight.new_zeros((self.r, self.in_features)))
                self.lora_Bs[task] = nn.Parameter(self.weight.new_zeros((self.out_features, self.r)))
        return self.lora_As, self.lora_Bs


class Embedding(nn.Embedding, LoRALayer):

    def __init__(self, num_embeddings: int, embedding_dim: int, r: int=0, lora_alpha: int=1, merge_weights: bool=True, **kwargs):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=0, merge_weights=merge_weights)
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, num_embeddings)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((embedding_dim, r)))
            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.Embedding.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            nn.init.zeros_(self.lora_A)
            nn.init.normal_(self.lora_B)

    def train(self, mode: bool=True):
        nn.Embedding.train(self, mode)
        if self.merge_weights and self.merged:
            if self.r > 0:
                self.weight.data -= (self.lora_B @ self.lora_A).T * self.scaling
            self.merged = False

    def eval(self):
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            if self.r > 0:
                self.weight.data += self.lora_B @ self.lora_A * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            result = nn.Embedding.forward(self, x)
            if self.r > 0:
                after_A = F.embedding(x, self.lora_A.T, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)
                result += after_A @ self.lora_B.T * self.scaling
            return result
        else:
            return nn.Embedding.forward(self, x)


class Linear(nn.Linear, LoRALayer):

    def __init__(self, in_features: int, out_features: int, r: int=0, lora_alpha: int=1, lora_dropout: float=0.0, fan_in_fan_out: bool=False, merge_weights: bool=True, **kwargs):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        self.fan_in_fan_out = fan_in_fan_out
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool=True):

        def T(w):
            return w.T if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            if self.r > 0:
                self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
            self.merged = False

    def eval(self):

        def T(w):
            return w.T if self.fan_in_fan_out else w
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            if self.r > 0:
                self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor):

        def T(w):
            return w.T if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                result += self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)


class MergedLinear(nn.Linear, LoRALayer):

    def __init__(self, in_features: int, out_features: int, r: int=0, lora_alpha: int=1, lora_dropout: float=0.0, enable_lora: List[bool]=[False], fan_in_fan_out: bool=False, merge_weights: bool=True, **kwargs):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        assert out_features % len(enable_lora) == 0, 'The length of enable_lora must divide out_features'
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        if r > 0 and any(enable_lora):
            self.lora_A = nn.Parameter(self.weight.new_zeros((r * sum(enable_lora), in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features // len(enable_lora) * sum(enable_lora), r)))
            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False
            self.lora_ind = self.weight.new_zeros((out_features,), dtype=torch.bool).view(len(enable_lora), -1)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def zero_pad(self, x):
        result = x.new_zeros((*x.shape[:-1], self.out_features))
        result = result.view(-1, self.out_features)
        result[:, self.lora_ind] = x.reshape(-1, self.out_features // len(self.enable_lora) * sum(self.enable_lora))
        return result.view((*x.shape[:-1], self.out_features))

    def train(self, mode: bool=True):

        def T(w):
            return w.T if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            if self.r > 0 and any(self.enable_lora):
                delta_w = F.conv1d(self.lora_A.data.unsqueeze(0), self.lora_B.data.unsqueeze(-1), groups=sum(self.enable_lora)).squeeze(0)
                self.weight.data -= self.zero_pad(T(delta_w * self.scaling))
            self.merged = False

    def eval(self):

        def T(w):
            return w.T if self.fan_in_fan_out else w
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            if self.r > 0 and any(self.enable_lora):
                delta_w = F.conv1d(self.lora_A.data.unsqueeze(0), self.lora_B.data.unsqueeze(-1), groups=sum(self.enable_lora)).squeeze(0)
                self.weight.data += self.zero_pad(T(delta_w * self.scaling))
            self.merged = True

    def forward(self, x: torch.Tensor):

        def T(w):
            return w.T if self.fan_in_fan_out else w
        if self.merged:
            return F.linear(x, T(self.weight), bias=self.bias)
        else:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                after_A = F.linear(self.lora_dropout(x), self.lora_A)
                after_B = F.conv1d(after_A.transpose(-2, -1), self.lora_B.unsqueeze(-1), groups=sum(self.enable_lora)).transpose(-2, -1)
                result += self.zero_pad(after_B) * self.scaling
            return result


class CustomForward(nn.Module):

    def __init__(self, bert_module):
        super().__init__()
        self.bert_module = bert_module

    def forward(self, inputs):
        return self.bert_module(inputs_embeds=inputs).last_hidden_state


class T5LayerNorm(nn.Module):

    def __init__(self, hidden_size, eps=1e-06):
        """
        Construct a layernorm module in the T5 style No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        if self.weight.dtype == torch.float16:
            hidden_states = hidden_states
        return self.weight * hidden_states


class VisualEmbedding(nn.Module):

    def __init__(self, config, obj_order_embedding):
        super().__init__()
        self.config = config
        feat_dim = config.feat_dim
        pos_dim = config.pos_dim
        n_images = config.n_images
        if self.config.individual_vis_layer_norm:
            feat_embedding = [nn.Linear(feat_dim, config.d_model)]
            if self.config.use_vis_layer_norm:
                feat_embedding.append(T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon))
            for i in range(config.additional_visual_embedding_layers):
                feat_embedding.append(nn.Linear(config.d_model, config.d_model))
                feat_embedding.append(nn.ReLU(True))
                if self.config.use_vis_layer_norm:
                    feat_embedding.append(T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon))
            self.feat_embedding = nn.Sequential(*feat_embedding)
            absolute_vis_pos_embedding = [nn.Linear(pos_dim + 1, config.d_model)]
            if self.config.use_vis_layer_norm:
                absolute_vis_pos_embedding.append(T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon))
            self.absolute_vis_pos_embedding = nn.Sequential(*absolute_vis_pos_embedding)
            if self.config.use_vis_order_embedding:
                self.obj_order_embedding = obj_order_embedding
                self.img_order_embedding = nn.Embedding(n_images, config.d_model)
        else:
            feat_embedding = [nn.Linear(feat_dim, config.d_model)]
            for i in range(config.additional_visual_embedding_layers):
                feat_embedding.append(nn.Linear(config.d_model, config.d_model))
                feat_embedding.append(nn.ReLU(True))
            self.feat_embedding = nn.Sequential(*feat_embedding)
            absolute_vis_pos_embedding = [nn.Linear(pos_dim + 1, config.d_model)]
            self.absolute_vis_pos_embedding = nn.Sequential(*absolute_vis_pos_embedding)
            if self.config.use_vis_order_embedding:
                self.obj_order_embedding = obj_order_embedding
                self.img_order_embedding = nn.Embedding(n_images, config.d_model)
            if self.config.use_vis_layer_norm:
                self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

    def get_area(self, pos):
        """
        Args
            pos: [B, N, 4]
                (x1, x2, y1, y2)
        Return
            area : [B, N]
        """
        height = pos[:, :, 3] - pos[:, :, 2]
        width = pos[:, :, 1] - pos[:, :, 0]
        area = height * width
        return area

    def forward(self, feats, pos, img_order_ids=None, obj_order_ids=None):
        """
        Args
            feats: [B, N, feat_dim]
            pos: [B, N, 4]
                (x1, x2, y1, y2)
        Return
            relative_vis_pos_embedding: [B, N, N, n_heads]
            absolute_vis_pos_embedding: # [B, N, d_model]
        """
        B, N, _ = feats.size()
        assert pos.size() == (B, N, 4)
        feat_embedding = self.feat_embedding(feats)
        device = feats.device
        dtype = feats.dtype
        area = self.get_area(pos).unsqueeze(2)
        pos = torch.cat([pos, area], dim=2)
        absolute_vis_pos_embedding = self.absolute_vis_pos_embedding(pos)
        if self.config.use_vis_order_embedding:
            if img_order_ids is None:
                img_order_ids = torch.zeros(N, dtype=torch.long, device=device)
                img_order_ids = img_order_ids.unsqueeze(0)
            img_order_embedding = self.img_order_embedding(img_order_ids)
            if obj_order_ids is None:
                obj_order_ids = torch.arange(N, dtype=torch.long, device=device)
                obj_order_ids = obj_order_ids.unsqueeze(0)
            obj_order_ids = self.obj_order_embedding.num_embeddings - obj_order_ids - 1
            obj_order_embedding = self.obj_order_embedding(obj_order_ids)
            vis_embedding = feat_embedding + absolute_vis_pos_embedding + img_order_embedding + obj_order_embedding
        else:
            vis_embedding = feat_embedding + absolute_vis_pos_embedding
        if not self.config.individual_vis_layer_norm:
            if self.config.use_vis_layer_norm:
                vis_embedding = self.layer_norm(vis_embedding)
        return vis_embedding


class ExpandVisualEmbedding(nn.Module):

    def __init__(self, config, obj_order_embedding):
        super().__init__()
        self.config = config
        feat_dim = config.feat_dim
        pos_dim = config.pos_dim
        n_images = config.n_images
        n_image_tokens = config.n_image_tokens
        self.n_image_tokens = n_image_tokens
        feat_embedding = [nn.Linear(feat_dim, n_image_tokens * config.d_model)]
        if self.config.use_vis_layer_norm and self.config.individual_vis_layer_norm:
            feat_embedding.append(nn.LayerNorm(n_image_tokens * config.d_model))
        self.feat_embedding = nn.Sequential(*feat_embedding)
        absolute_vis_pos_embedding = [nn.Linear(pos_dim + 1, n_image_tokens * config.d_model)]
        if self.config.use_vis_layer_norm and self.config.individual_vis_layer_norm:
            absolute_vis_pos_embedding.append(nn.LayerNorm(n_image_tokens * config.d_model))
        self.absolute_vis_pos_embedding = nn.Sequential(*absolute_vis_pos_embedding)
        if self.config.use_vis_order_embedding:
            self.obj_order_embedding = nn.Embedding(config.n_boxes, n_image_tokens * config.d_model)
            self.img_order_embedding = nn.Embedding(n_images, n_image_tokens * config.d_model)
            self.default_obj_order_ids = self.config.default_obj_order_ids
        if self.config.use_vis_layer_norm and not self.config.individual_vis_layer_norm:
            self.layer_norm = nn.LayerNorm(n_image_tokens * config.d_model)

    def get_area(self, pos):
        """
        Args
            pos: [B, N, 4]
                (x1, x2, y1, y2)
        Return
            area : [B, N]
        """
        height = pos[:, :, 3] - pos[:, :, 2]
        width = pos[:, :, 1] - pos[:, :, 0]
        area = height * width
        return area

    def forward(self, feats, pos, img_order_ids=None, obj_order_ids=None):
        """
        Args
            feats: [B, N, feat_dim]
            pos: [B, N, 4]
                (x1, x2, y1, y2)
        Return
            relative_vis_pos_embedding: [B, N, N, n_heads]
            absolute_vis_pos_embedding: # [B, N, d_model]
        """
        B, N, _ = feats.size()
        assert pos.size() == (B, N, 4)
        feat_embedding = self.feat_embedding(feats)
        device = feats.device
        dtype = feats.dtype
        area = self.get_area(pos).unsqueeze(2)
        pos = torch.cat([pos, area], dim=2)
        absolute_vis_pos_embedding = self.absolute_vis_pos_embedding(pos)
        if self.config.use_vis_order_embedding:
            if img_order_ids is None:
                img_order_ids = torch.zeros(N, dtype=torch.long, device=device)
                img_order_ids = img_order_ids.unsqueeze(0)
            img_order_embedding = self.img_order_embedding(img_order_ids)
            if obj_order_ids is None:
                obj_order_ids = torch.arange(N, dtype=torch.long, device=device)
                obj_order_ids = obj_order_ids.unsqueeze(0)
            obj_order_embedding = self.obj_order_embedding(obj_order_ids)
            vis_embedding = feat_embedding + absolute_vis_pos_embedding + img_order_embedding + obj_order_embedding
        else:
            vis_embedding = feat_embedding + absolute_vis_pos_embedding
        if not self.config.individual_vis_layer_norm:
            if self.config.use_vis_layer_norm:
                vis_embedding = self.layer_norm(vis_embedding)
        vis_embedding = vis_embedding.reshape(B, self.n_image_tokens, -1)
        return vis_embedding


class ResidualVisualEmbedding(nn.Module):

    def __init__(self, config, obj_order_embedding):
        super().__init__()
        self.config = config
        feat_dim = config.feat_dim
        pos_dim = config.pos_dim
        n_images = config.n_images
        self.vis_use_transformer = config.vis_use_transformer
        LinearModule = nn.Linear
        feat_embedding = [LinearModule(feat_dim, config.d_model * config.encoder_layers * 4)]
        if self.config.use_vis_layer_norm and self.config.individual_vis_layer_norm:
            feat_embedding.append(nn.LayerNorm(config.d_model * config.encoder_layers * 4))
        self.feat_embedding = nn.Sequential(*feat_embedding)
        absolute_vis_pos_embedding = [LinearModule(pos_dim + 1, config.d_model * config.encoder_layers * 4)]
        if self.config.use_vis_layer_norm and self.config.individual_vis_layer_norm:
            absolute_vis_pos_embedding.append(nn.LayerNorm(config.d_model * config.encoder_layers * 4))
        self.absolute_vis_pos_embedding = nn.Sequential(*absolute_vis_pos_embedding)
        if self.config.use_vis_order_embedding:
            self.obj_num_embeddings = config.n_boxes
            self.obj_order_embedding = nn.Sequential(nn.Embedding(config.n_boxes, config.d_model), LinearModule(config.d_model, config.d_model * config.encoder_layers * 4))
            self.img_order_embedding = nn.Sequential(nn.Embedding(n_images, config.d_model), LinearModule(config.d_model, config.d_model * config.encoder_layers * 4))
            self.default_obj_order_ids = self.config.default_obj_order_ids
        if self.config.use_vis_layer_norm and not self.config.individual_vis_layer_norm:
            self.layer_norm = nn.LayerNorm(config.d_model * config.encoder_layers * 4 * encoder_attention_heads)

    def get_area(self, pos):
        """
        Args
            pos: [B, N, 4]
                (x1, x2, y1, y2)
        Return
            area : [B, N]
        """
        height = pos[:, :, 3] - pos[:, :, 2]
        width = pos[:, :, 1] - pos[:, :, 0]
        area = height * width
        return area

    def forward(self, feats, pos, img_order_ids=None, obj_order_ids=None):
        """
        Args
            feats: [B, N, feat_dim]
            pos: [B, N, 4]
                (x1, x2, y1, y2)
        Return
            relative_vis_pos_embedding: [B, N, N, n_heads]
            absolute_vis_pos_embedding: # [B, N, d_model]
        """
        B, N, _ = feats.size()
        assert pos.size() == (B, N, 4)
        feat_embedding = self.feat_embedding(feats)
        device = feats.device
        dtype = feats.dtype
        area = self.get_area(pos).unsqueeze(2)
        pos = torch.cat([pos, area], dim=2)
        absolute_vis_pos_embedding = self.absolute_vis_pos_embedding(pos)
        if self.config.use_vis_order_embedding:
            if img_order_ids is None:
                img_order_ids = torch.zeros(N, dtype=torch.long, device=device)
                img_order_ids = img_order_ids.unsqueeze(0)
            img_order_embedding = self.img_order_embedding(img_order_ids)
            if obj_order_ids is None:
                obj_order_ids = torch.arange(N, dtype=torch.long, device=device)
                obj_order_ids = obj_order_ids.unsqueeze(0)
            obj_order_ids = self.obj_num_embeddings - obj_order_ids - 1
            obj_order_embedding = self.obj_order_embedding(obj_order_ids)
            vis_embedding = feat_embedding + absolute_vis_pos_embedding + img_order_embedding + obj_order_embedding
        else:
            vis_embedding = feat_embedding + absolute_vis_pos_embedding
        if not self.config.individual_vis_layer_norm:
            if self.config.use_vis_layer_norm:
                vis_embedding = self.layer_norm(vis_embedding)
        return vis_embedding


class Downsample(nn.Module):

    def __init__(self, output_size):
        super().__init__()
        """
        output size: list of 1-D size, such as (6, 6), (16, 16)
        """
        self.output_size = output_size
        self.pool = nn.AdaptiveMaxPool2d(output_size)

    def forward(self, inputs_tuple):
        inputs = inputs_tuple
        B, L, dim = inputs.shape
        inputs = inputs.permute(0, 2, 1)
        sqrt_L = int(L ** 0.5)
        inputs = inputs.reshape(B, dim, sqrt_L, sqrt_L)
        outputs = self.pool(inputs)
        outputs = outputs.reshape(B, dim, -1)
        outputs = outputs.permute(0, 2, 1)
        return outputs


class OneDDownsample(nn.Module):

    def __init__(self, output_size):
        super().__init__()
        """
        output size: list of 1-D size, such as (6, 6), (16, 16)
        """
        self.output_size = output_size
        self.pool = nn.AdaptiveMaxPool1d(output_size)

    def downsample_inputs(self, inputs):
        B, L, dim = inputs.shape
        inputs = inputs.permute(0, 2, 1)
        inputs = self.pool(inputs)
        inputs = inputs.reshape(B, dim, -1)
        inputs = inputs.permute(0, 2, 1)
        return inputs

    def forward(self, inputs_tuple):
        if len(inputs_tuple) == 4:
            inputs, boxes, img_order_ids, obj_order_ids = inputs_tuple
            inputs = torch.cat(torch.chunk(inputs, 2, 1), 0)
            inputs = self.downsample_inputs(inputs)
            inputs = torch.cat(torch.chunk(inputs, 2, 0), 1)
            boxes = torch.cat(torch.chunk(boxes, 2, 1), 0)
            boxes = boxes[:, :inputs.shape[1] // 2]
            boxes = torch.cat(torch.chunk(boxes, 2, 0), 1)
            img_order_ids = torch.cat(torch.chunk(img_order_ids, 2, 1), 0)
            img_order_ids = img_order_ids[:, :inputs.shape[1] // 2]
            img_order_ids = torch.cat(torch.chunk(img_order_ids, 2, 0), 1)
            obj_order_ids = torch.cat(torch.chunk(obj_order_ids, 2, 1), 0)
            obj_order_ids = obj_order_ids[:, :inputs.shape[1] // 2]
            obj_order_ids = torch.cat(torch.chunk(obj_order_ids, 2, 0), 1)
            outputs_tuple = inputs, boxes, img_order_ids, obj_order_ids
        else:
            inputs, boxes = inputs_tuple
            inputs = self.downsample_inputs(inputs)
            boxes = boxes[:, :inputs.shape[1]]
            outputs_tuple = inputs, boxes
        return outputs_tuple


class SparseSample(nn.Module):

    def __init__(self, output_size):
        super().__init__()
        """
        output size: list of 1-D size, such as (6, 6), (16, 16)
        """
        self.output_size = output_size

    def forward(self, inputs):
        if self.training:
            B, L, _ = inputs.shape
            x = torch.rand(B, L)
            indices = torch.argsort(torch.rand(*x.shape), dim=-1)
            indices = indices[:, :self.output_size]
            indices = torch.sort(indices)[0]
            return inputs[torch.arange(B).unsqueeze(1), indices]
        else:
            return inputs


class BartLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int):
        assert padding_idx is not None, '`padding_idx` should not be None, but of type int'
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim, padding_idx=padding_idx)

    def forward(self, input_ids_shape: torch.Size, past_key_values_length: int=0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        bsz, seq_len = input_ids_shape[:2]
        positions = torch.arange(past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device)
        return super().forward(positions + self.offset)


class BartAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float=0.0, is_decoder: bool=False, bias: bool=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, f'embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads}).'
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, hidden_states: torch.Tensor, key_value_states: Optional[torch.Tensor]=None, past_key_value: Optional[Tuple[torch.Tensor]]=None, attention_mask: Optional[torch.Tensor]=None, output_attentions: bool=False) ->Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, embed_dim = hidden_states.size()
        query_states = self.q_proj(hidden_states) * self.scaling
        if is_cross_attention and past_key_value is not None:
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        if self.is_decoder:
            past_key_value = key_states, value_states
        proj_shape = bsz * self.num_heads, -1, self.head_dim
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)
        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        assert attn_weights.size() == (bsz * self.num_heads, tgt_len, src_len), f'Attention weights should be of size {bsz * self.num_heads, tgt_len, src_len}, but is {attn_weights.size()}'
        if attention_mask is not None:
            assert attention_mask.size() == (bsz, 1, tgt_len, src_len), f'Attention mask should be of size {bsz, 1, tgt_len, src_len}, but is {attention_mask.size()}'
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        attn_weights = F.softmax(attn_weights, dim=-1)
        if output_attentions:
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None
        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.bmm(attn_probs, value_states)
        assert attn_output.size() == (bsz * self.num_heads, tgt_len, self.head_dim), f'`attn_output` should be of size {bsz, self.num_heads, tgt_len, self.head_dim}, but is {attn_output.size()}'
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim).transpose(1, 2).reshape(bsz, tgt_len, embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights_reshaped, past_key_value


class LoraBartAttention(BartAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float=0.0, is_decoder: bool=False, bias: bool=True, config=None):
        super().__init__(embed_dim, num_heads, dropout, is_decoder, bias)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, f'embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads}).'
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = lora.LoRALinearController(embed_dim, embed_dim, config=config, bias=bias)
        self.q_proj = lora.LoRALinearController(embed_dim, embed_dim, config=config, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, hidden_states: torch.Tensor, key_value_states: Optional[torch.Tensor]=None, past_key_value: Optional[Tuple[torch.Tensor]]=None, attention_mask: Optional[torch.Tensor]=None, output_attentions: bool=False, task=None) ->Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, embed_dim = hidden_states.size()
        query_states = self.q_proj(hidden_states, task) * self.scaling
        if is_cross_attention and past_key_value is not None:
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states, task), -1, bsz)
        elif past_key_value is not None:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states, task), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states, task), -1, bsz)
        if self.is_decoder:
            past_key_value = key_states, value_states
        proj_shape = bsz * self.num_heads, -1, self.head_dim
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)
        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        assert attn_weights.size() == (bsz * self.num_heads, tgt_len, src_len), f'Attention weights should be of size {bsz * self.num_heads, tgt_len, src_len}, but is {attn_weights.size()}'
        if attention_mask is not None:
            assert attention_mask.size() == (bsz, 1, tgt_len, src_len), f'Attention mask should be of size {bsz, 1, tgt_len, src_len}, but is {attention_mask.size()}'
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        attn_weights = F.softmax(attn_weights, dim=-1)
        if output_attentions:
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None
        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.bmm(attn_probs, value_states)
        assert attn_output.size() == (bsz * self.num_heads, tgt_len, self.head_dim), f'`attn_output` should be of size {bsz, self.num_heads, tgt_len, self.head_dim}, but is {attn_output.size()}'
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim).transpose(1, 2).reshape(bsz, tgt_len, embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights_reshaped, past_key_value


class BartClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, input_dim: int, inner_dim: int, num_classes: int, pooler_dropout: float):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


class T5DenseReluDense(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        hidden_states = self.wi(hidden_states)
        hidden_states = F.relu(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5DenseGatedGeluDense(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.gelu_act = ACT2FN['gelu_new']

    def forward(self, hidden_states):
        hidden_gelu = self.gelu_act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5LayerFF(nn.Module):

    def __init__(self, config):
        super().__init__()
        if config.feed_forward_proj == 'relu':
            self.DenseReluDense = T5DenseReluDense(config)
        elif config.feed_forward_proj == 'gated-gelu':
            self.DenseReluDense = T5DenseGatedGeluDense(config)
        else:
            raise ValueError(f'{self.config.feed_forward_proj} is not supported. Choose between `relu` and `gated-gelu`')
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)
        if config.use_adapter or config.use_compacter or config.use_lradapter:
            self.ff_adapter = AdapterController(config.adapter_config)
        else:
            self.ff_adapter = None
        self.adapter_hypernet = None
        if config.use_hyperformer:
            self.adapter_hypernet = MetaLayersAdapterController(config.adapter_config)

    def forward(self, hidden_states, block_adapters, task):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        if self.ff_adapter is not None:
            forwarded_states = self.ff_adapter(forwarded_states, task)
        if self.adapter_hypernet is not None:
            forwarded_states = self.adapter_hypernet(forwarded_states, block_adapters.feed_forward)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states


class T5LayerSelfAttention(nn.Module):

    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.SelfAttention = T5Attention(config, has_relative_attention_bias=has_relative_attention_bias)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)
        if config.use_adapter or config.use_compacter or config.use_lradapter:
            self.attn_adapter = AdapterController(config.adapter_config)
        else:
            self.attn_adapter = None
        self.adapter_hypernet = None
        if config.use_hyperformer:
            self.adapter_hypernet = MetaLayersAdapterController(config.adapter_config)

    def forward(self, hidden_states, attention_mask=None, position_bias=None, head_mask=None, past_key_value=None, use_cache=False, output_attentions=False, block_adapters=None, task=None):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(normed_hidden_states, mask=attention_mask, position_bias=position_bias, head_mask=head_mask, past_key_value=past_key_value, use_cache=use_cache, output_attentions=output_attentions)
        y = attention_output[0]
        if self.attn_adapter is not None:
            y = self.attn_adapter(y, task)
        if self.adapter_hypernet is not None:
            y = self.adapter_hypernet(y, block_adapters.self_attention)
        hidden_states = hidden_states + self.dropout(y)
        outputs = (hidden_states,) + attention_output[1:]
        return outputs


class T5LayerCrossAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.EncDecAttention = T5Attention(config, has_relative_attention_bias=False)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)
        if (config.use_adapter or config.use_compacter or config.use_lradapter) and config.add_adapter_cross_attn:
            self.enc_attn_adapter = AdapterController(config.adapter_config)
        else:
            self.enc_attn_adapter = None
        self.adapter_hypernet = None
        if config.use_hyperformer and config.add_adapter_cross_attn:
            self.adapter_hypernet = MetaLayersAdapterController(config.adapter_config)

    def forward(self, hidden_states, key_value_states, attention_mask=None, position_bias=None, head_mask=None, past_key_value=None, use_cache=False, query_length=None, output_attentions=False, block_adapters=None, task=None):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(normed_hidden_states, mask=attention_mask, key_value_states=key_value_states, position_bias=position_bias, head_mask=head_mask, past_key_value=past_key_value, use_cache=use_cache, query_length=query_length, output_attentions=output_attentions)
        y = attention_output[0]
        if self.enc_attn_adapter is not None:
            y = self.enc_attn_adapter(y, task)
        if self.adapter_hypernet is not None:
            y = self.adapter_hypernet(y, block_adapters.cross_attention)
        layer_output = hidden_states + self.dropout(y)
        outputs = (layer_output,) + attention_output[1:]
        return outputs


class T5Block(nn.Module):

    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(T5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias))
        if self.is_decoder:
            self.layer.append(T5LayerCrossAttention(config))
        self.layer.append(T5LayerFF(config))

    def forward(self, hidden_states, attention_mask=None, position_bias=None, encoder_hidden_states=None, encoder_attention_mask=None, encoder_decoder_position_bias=None, head_mask=None, past_key_value=None, use_cache=False, output_attentions=False, return_dict=True, block_adapters=None, task=None):
        if past_key_value is not None:
            assert self.is_decoder, 'Only decoder can use `past_key_values`'
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4
            error_message = 'There should be {} past states. 2 (past / key) for self attention.{} Got {} past key / value states'.format(expected_num_past_key_values, '2 (past / key) for cross attention' if expected_num_past_key_values == 4 else '', len(past_key_value))
            assert len(past_key_value) == expected_num_past_key_values, error_message
            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None
        self_attention_outputs = self.layer[0](hidden_states, attention_mask=attention_mask, position_bias=position_bias, head_mask=head_mask, past_key_value=self_attn_past_key_value, use_cache=use_cache, output_attentions=output_attentions, block_adapters=block_adapters, task=task)
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]
        if torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None
            cross_attention_outputs = self.layer[1](hidden_states, key_value_states=encoder_hidden_states, attention_mask=encoder_attention_mask, position_bias=encoder_decoder_position_bias, head_mask=head_mask, past_key_value=cross_attn_past_key_value, query_length=query_length, use_cache=use_cache, output_attentions=output_attentions, block_adapters=block_adapters, task=task)
            hidden_states = cross_attention_outputs[0]
            if torch.isinf(hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]
            attention_outputs = attention_outputs + cross_attention_outputs[2:]
        hidden_states = self.layer[-1](hidden_states, block_adapters, task)
        if torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        outputs = hidden_states,
        outputs = outputs + (present_key_value_state,) + attention_outputs
        return outputs


class InputPrompts(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.prompt_len = config.prompt_len
        self.input_dim = config.input_dim
        self.mid_dim = config.mid_dim
        self.prefix_tokens = torch.arange(self.prompt_len).long()
        self.prefix_embedding = nn.Sequential(nn.Embedding(self.prompt_len, self.input_dim), nn.Linear(self.input_dim, self.mid_dim), nn.Tanh(), nn.Linear(self.mid_dim, self.input_dim))

    def get_prompt(self, bsz, device):
        input_tokens = self.prefix_tokens.unsqueeze(0).expand(bsz, -1)
        prefix_prompt = self.prefix_embedding(input_tokens)
        return prefix_prompt


class PromptController(nn.Module):
    """Implements Adapter controller module which controls the logics of
    putting adapter layers within transformer's layers."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.prompts = nn.ModuleDict(dict())
        self.tasks = config.tasks
        self.use_input_prompt = config.use_input_prompt
        self.use_single_prompt = config.use_single_prompt
        self.prompts = self.construct_prompts(self.tasks)

    def get_task(self, task):
        return task

    def construct_prompts(self, tasks):
        """
        Constructs adapter layers and adds them to a dictionary for the given
        tasks.
        Args:
            tasks: A list of string containing the task names.
        """
        if self.use_single_prompt:
            if self.use_input_prompt:
                prompt = InputPrompts(self.config)
            for task in tasks:
                self.prompts[task] = prompt
        else:
            for task in tasks:
                if self.use_input_prompt:
                    prompt = InputPrompts(self.config)
                    self.prompts[task] = prompt
        return self.prompts

    def convert_to_list(self, tasks):
        if isinstance(tasks, list):
            return tasks
        return [tasks]

    def get_prompt(self, task):
        """Given a task returns its corresponding adapter layer.
        Args:
            task: Input task name.
        Returns:
            Adapter layer corresponding to the given task.
        """
        return self.prompts[task]

    def forward(self, bsz, device, task):
        """
        Retrieves the adapter layer corresponding to the given
        task. It freezes the adapter layers for all the other tasks
        and call the selected adapter layer.
        Args:
            task: the name of the current task.
            inputs: the inputs to feed in in the adapter layer.
        Returns:
            outputs of the adapter layer.
        """
        task = self.get_task(task)
        prompt_module = self.get_prompt(task)
        trainable_prompt = prompt_module.get_prompt(bsz, device)
        return trainable_prompt


_MODELS = {'RN50': 'https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt', 'RN101': 'https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt', 'RN50x4': 'https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt', 'ViT-B/32': 'https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt'}


def _transform(n_px):
    return Compose([Resize(n_px, interpolation=Image.BICUBIC), CenterCrop(n_px), lambda image: image.convert('RGB'), ToTensor(), Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])


def available_models() ->List[str]:
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())


def build_model():
    cfg = get_cfg()
    cfg.merge_from_file(os.path.join(D2_ROOT, 'configs/VG-Detection/faster_rcnn_R_101_C4_attr_caffemaxpool.yaml'))
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 300
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
    cfg.MODEL.WEIGHTS = str(Path.home().joinpath('.torch/fvcore_cache/models/faster_rcnn_from_caffe_attr.pkl'))
    detector = DefaultPredictor(cfg)
    return detector


def load(name: str, device: Union[str, torch.device]='cuda' if torch.cuda.is_available() else 'cpu', jit=True, adapter_type=None, reduction_factor=1, use_bn=True):
    """Load a CLIP model

    Parameters
    ----------
    name : str
        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict

    device : Union[str, torch.device]
        The device to put the loaded model

    jit : bool
        Whether to load the optimized JIT model (default) or more hackable non-JIT model.

    Returns
    -------
    model : torch.nn.Module
        The CLIP model

    preprocess : Callable[[PIL.Image], torch.Tensor]
        A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input
    """
    if name in _MODELS:
        model_path = _download(_MODELS[name])
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f'Model {name} not found; available models = {available_models()}')
    try:
        model = torch.jit.load(model_path, map_location=device if jit else 'cpu').eval()
        state_dict = None
    except RuntimeError:
        if jit:
            warnings.warn(f'File {model_path} is not a JIT archive. Loading as a state dict instead')
            jit = False
        state_dict = torch.load(model_path, map_location='cpu')
    if not jit:
        model = build_model(state_dict or model.state_dict(), adapter_type, reduction_factor, use_bn)
        if str(device) == 'cpu':
            model.float()
        return model, _transform(model.visual.input_resolution)
    device_holder = torch.jit.trace(lambda : torch.ones([]), example_inputs=[])
    device_node = [n for n in device_holder.graph.findAllNodes('prim::Constant') if 'Device' in repr(n)][-1]

    def patch_device(module):
        graphs = [module.graph] if hasattr(module, 'graph') else []
        if hasattr(module, 'forward1'):
            graphs.append(module.forward1.graph)
        for graph in graphs:
            for node in graph.findAllNodes('prim::Constant'):
                if 'value' in node.attributeNames() and str(node['value']).startswith('cuda'):
                    node.copyAttributes(device_node)
    model.apply(patch_device)
    patch_device(model.encode_image)
    patch_device(model.encode_text)
    if str(device) == 'cpu':
        float_holder = torch.jit.trace(lambda : torch.ones([]).float(), example_inputs=[])
        float_input = list(float_holder.graph.findNode('aten::to').inputs())[1]
        float_node = float_input.node()

        def patch_float(module):
            graphs = [module.graph] if hasattr(module, 'graph') else []
            if hasattr(module, 'forward1'):
                graphs.append(module.forward1.graph)
            for graph in graphs:
                for node in graph.findAllNodes('aten::to'):
                    inputs = list(node.inputs())
                    for i in [1, 2]:
                        if inputs[i].node()['value'] == 5:
                            inputs[i].node().copyAttributes(float_node)
        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)
        model.float()
    return model, _transform(model.input_resolution.item())


def resize_pos_embed(posemb, posemb_new):
    ntok_new = posemb_new.shape[1]
    if True:
        posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
        ntok_new -= 1
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    gs_new = int(math.sqrt(ntok_new))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(gs_new, gs_new), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


class CLIPResNetEncoder(nn.Module):

    def __init__(self, backbone='RN50x4', image_size=224, adapter_type=None, reduction_factor=1, use_bn=True):
        super().__init__()
        self.model, transform = load(backbone, device='cpu', jit=False, adapter_type=adapter_type, reduction_factor=reduction_factor, use_bn=use_bn)
        del self.model.transformer
        self.config = PretrainedConfig(image_size=image_size, patch_size=32, hidden_size=self.model.visual.attnpool.positional_embedding.shape[-1])
        num_patches = int(image_size / 32) ** 2
        pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.model.visual.attnpool.positional_embedding.shape[-1]))
        pos_embed.weight = resize_pos_embed(self.model.visual.attnpool.positional_embedding.unsqueeze(0), pos_embed)
        self.model.visual.attnpool.positional_embedding = pos_embed

    def reset_image_size(self, image_size=224, patch_size=32):
        old_image_size = self.config.image_size
        new_image_size = image_size
        old_grid = old_image_size // self.config.patch_size
        new_grid = new_image_size // patch_size
        previous_position_embedding_weight = self.model.visual.attnpool.positional_embedding.data[0, 1:]
        previous_position_embedding_weight = previous_position_embedding_weight.transpose(1, 0)
        previous_position_embedding_weight = previous_position_embedding_weight.view(1, self.config.hidden_size, old_grid, old_grid)
        new_position_embedding_weight = torch.nn.functional.interpolate(previous_position_embedding_weight, size=(new_grid, new_grid), mode='bicubic', align_corners=False)
        new_position_embedding_weight = new_position_embedding_weight.view(self.config.hidden_size, new_grid ** 2).transpose(1, 0)
        new_position_embedding = nn.Parameter(torch.zeros(1 + new_grid ** 2, self.config.hidden_size))
        new_position_embedding.data[0] = self.model.visual.attnpool.positional_embedding.data[0, 0]
        new_position_embedding.data[1:] = new_position_embedding_weight
        self.model.visual.attnpool.positional_embedding = nn.Parameter(new_position_embedding.unsqueeze(0))
        self.config.image_size = new_image_size
        self.config.patch_size = patch_size

    def forward(self, image):
        x, attnpool = self.model.encode_image(image)
        B, C, H, W = x.size()
        x = x.permute(0, 2, 3, 1)
        x = x.contiguous().view(B, H * W, C)
        x = x
        return x, attnpool.unsqueeze(1)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BartClassificationHead,
     lambda: ([], {'input_dim': 4, 'inner_dim': 4, 'num_classes': 4, 'pooler_dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BartLearnedPositionalEmbedding,
     lambda: ([], {'num_embeddings': 4, 'embedding_dim': 4, 'padding_idx': 4}),
     lambda: ([[4, 4, 4, 4]], {}),
     False),
    (BertAttention,
     lambda: ([], {'config': _mock_config(hidden_size=4, num_attention_heads=4, attention_probs_dropout_prob=0.5)}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (BertIntermediate,
     lambda: ([], {'config': _mock_config(hidden_size=4, intermediate_size=4, hidden_act=_mock_layer())}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BertPooler,
     lambda: ([], {'config': _mock_config(hidden_size=4)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BertPredictionHeadTransform,
     lambda: ([], {'config': _mock_config(hidden_size=4, hidden_act=_mock_layer())}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BertVisualAnswerHead,
     lambda: ([], {'config': _mock_config(hidden_size=4), 'num_answers': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Bottleneck,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Conv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Downsample,
     lambda: ([], {'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (ExpandVisualEmbedding,
     lambda: ([], {'config': _mock_config(feat_dim=4, pos_dim=4, n_images=4, n_image_tokens=4, d_model=4, use_vis_layer_norm=1, individual_vis_layer_norm=1, use_vis_order_embedding=4, n_boxes=4, default_obj_order_ids=4), 'obj_order_embedding': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (FastRCNNOutputLayers,
     lambda: ([], {'input_size': 4, 'num_classes': 4, 'cls_agnostic_bbox_reg': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (GeLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LastLevelMaxPool,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LastLevelP6P7,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LayerNorm,
     lambda: ([], {'normalized_shape': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (LayerNormHyperNet,
     lambda: ([], {'config': _mock_config(train_task_embeddings=False, task_embedding_dim=4, input_dim=4)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LowRankLinear,
     lambda: ([], {'input_dim': 4, 'output_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (OneDDownsample,
     lambda: ([], {'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (QuickGELU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResidualAttentionBlock,
     lambda: ([], {'d_model': 4, 'n_head': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (ResidualVisualEmbedding,
     lambda: ([], {'config': _mock_config(feat_dim=4, pos_dim=4, n_images=4, vis_use_transformer=4, d_model=4, encoder_layers=1, use_vis_layer_norm=1, individual_vis_layer_norm=1, use_vis_order_embedding=4, n_boxes=4, default_obj_order_ids=4), 'obj_order_embedding': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (SparseSample,
     lambda: ([], {'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (T5LayerNorm,
     lambda: ([], {'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Transformer,
     lambda: ([], {'width': 4, 'layers': 1, 'heads': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
]

class Test_ylsung_VL_adapter(_paritybench_base):
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

