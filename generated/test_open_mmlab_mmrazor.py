import sys
_module = sys.modules[__name__]
del sys
cifar100_bs16 = _module
cifar10_bs16 = _module
cifar10_bs96_cutout = _module
imagenet_bs128_colorjittor = _module
imagenet_bs256_autoslim = _module
imagenet_bs32 = _module
imagenet_bs32_pil_resize = _module
imagenet_bs64 = _module
imagenet_bs64_autoaug = _module
imagenet_bs64_pil_resize = _module
imagenet_bs64_pil_resize_autoaug = _module
imagenet_bs64_swin_224 = _module
imagenet_bs64_swin_384 = _module
cityscapes_detection = _module
cityscapes_instance = _module
coco_detection = _module
coco_instance = _module
coco_instance_semantic = _module
coco_panoptic = _module
deepfashion = _module
lvis_v1_instance = _module
voc0712 = _module
wider_face = _module
ade20k = _module
chase_db1 = _module
cityscapes = _module
cityscapes_1024x1024 = _module
cityscapes_769x769 = _module
cityscapes_832x832 = _module
drive = _module
hrf = _module
pascal_context = _module
pascal_context_59 = _module
pascal_voc12 = _module
pascal_voc12_aug = _module
stare = _module
mmcls_runtime = _module
mmdet_runtime = _module
mmseg_runtime = _module
cifar10_bs128 = _module
imagenet_bs1024_adamw_swin = _module
imagenet_bs1024_linearlr_bn_nowd = _module
imagenet_bs1024_spos = _module
imagenet_bs2048 = _module
imagenet_bs2048_AdamW = _module
imagenet_bs2048_autoslim = _module
imagenet_bs2048_coslr = _module
imagenet_bs256 = _module
imagenet_bs256_140e = _module
imagenet_bs256_200e_coslr_warmup = _module
imagenet_bs256_coslr = _module
imagenet_bs256_epochstep = _module
imagenet_bs4096_AdamW = _module
schedule_1x = _module
schedule_2x = _module
schedule_160k = _module
schedule_20k = _module
schedule_320k = _module
schedule_40k = _module
schedule_80k = _module
cwd_cls_head_gfl_r101_fpn_gfl_r50_fpn_1x_coco = _module
cwd_cls_head_pspnet_r101_d8_pspnet_r18_d8_512x1024_cityscapes_80k = _module
rkd_neck_resnet34_resnet18_8xb32_in1k = _module
wsld_cls_head_resnet34_resnet18_8xb32_in1k = _module
darts_subnet_1xb96_cifar10 = _module
darts_supernet_unroll_1xb64_cifar10 = _module
detnas_evolution_search_frcnn_shufflenetv2_fpn_coco = _module
detnas_subnet_frcnn_shufflenetv2_fpn_1x_coco = _module
detnas_subnet_shufflenetv2_8xb128_in1k = _module
detnas_supernet_frcnn_shufflenetv2_fpn_1x_coco = _module
detnas_supernet_shufflenetv2_8xb128_in1k = _module
spos_evolution_search_mobilenet_proxyless_gpu_flops465_8xb512_in1k = _module
spos_evolution_search_shufflenetv2_8xb2048_in1k = _module
spos_mobilenet_for_check_ckpt_from_anglenas = _module
spos_subnet_mobilenet_proxyless_gpu_8xb128_in1k = _module
spos_subnet_shufflenetv2_8xb128_in1k = _module
spos_supernet_mobilenet_proxyless_gpu_8xb128_in1k = _module
spos_supernet_shufflenetv2_8xb128_in1k = _module
autoslim_mbv2_search_8xb1024_in1k = _module
autoslim_mbv2_subnet_8xb256_in1k = _module
autoslim_mbv2_supernet_8xb256_in1k = _module
conf = _module
gen_model_zoo = _module
mmrazor = _module
apis = _module
mmcls = _module
inference = _module
train = _module
mmdet = _module
inference = _module
train = _module
mmseg = _module
inference = _module
train = _module
utils = _module
core = _module
builder = _module
distributed_wrapper = _module
hooks = _module
drop_path_prob = _module
sampler_seed = _module
search_subnet = _module
optimizer = _module
builder = _module
runners = _module
epoch_based_runner = _module
iter_based_runner = _module
searcher = _module
evolution_search = _module
greedy_search = _module
broadcast = _module
lr = _module
utils = _module
datasets = _module
utils = _module
models = _module
algorithms = _module
align_method_kd = _module
autoslim = _module
base = _module
darts = _module
detnas = _module
general_distill = _module
spos = _module
architectures = _module
base = _module
components = _module
backbones = _module
darts_backbone = _module
searchable_mobilenet = _module
searchable_shufflenet_v2 = _module
heads = _module
darts_head = _module
no_bias_fc_head = _module
necks = _module
placeholder = _module
distillers = _module
base = _module
self_distiller = _module
single_teacher = _module
losses = _module
cwd = _module
kl_divergence = _module
relational_kd = _module
weighted_soft_label_distillation = _module
mutables = _module
mutable_edge = _module
mutable_module = _module
mutable_op = _module
mutators = _module
base = _module
darts_mutator = _module
differentiable_mutator = _module
one_shot_mutator = _module
ops = _module
common = _module
darts_series = _module
mobilenet_series = _module
shufflenet_series = _module
pruners = _module
ratio_pruning = _module
structure_pruning = _module
switchable_bn = _module
misc = _module
setup_env = _module
version = _module
setup = _module
cwd_pspnet = _module
detnas_frcnn_shufflenet_fpn = _module
retinanet = _module
test_inference = _module
test_utils = _module
test_searcher = _module
test_algorithm = _module
utils = _module
test_architecture = _module
test_mutable = _module
test_mutator = _module
test_op = _module
test_pruner = _module
test_misc = _module
test_setup_env = _module
get_flops = _module
print_config = _module
search_mmcls = _module
test_mmcls = _module
train_mmcls = _module
search_mmdet = _module
test_mmdet = _module
train_mmdet = _module
test_mmseg = _module
train_mmseg = _module
publish_model = _module
split_checkpoint = _module

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


from typing import Dict


from typing import Optional


from typing import Union


from torch import nn


import random


import numpy as np


import torch


import torch.distributed as dist


import torch.nn as nn


from torch.cuda._utils import _get_device_index


import time


import copy


from typing import Any


from typing import List


from typing import Tuple


from torch import Tensor


from torch import distributed as dist


from torch.utils.data import random_split


from torch.nn.modules.batchnorm import _BatchNorm


from collections import OrderedDict


from functools import partial


from abc import ABCMeta


from abc import abstractmethod


import torch.nn.functional as F


from torch.nn import functional as F


import torch.utils.checkpoint as cp


from torch.nn.modules import GroupNorm


from types import MethodType


from torch.nn.modules.instancenorm import _InstanceNorm


import torch.multiprocessing as mp


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


from copy import deepcopy


class DistributedDataParallelWrapper(nn.Module):
    """A DistributedDataParallel wrapper for models in MMediting.

    In MMedting, there is a need to wrap different modules in the models
    with separate DistributedDataParallel. Otherwise, it will cause
    errors for GAN training.
    More specific, the GAN model, usually has two sub-modules:
    generator and discriminator. If we wrap both of them in one
    standard DistributedDataParallel, it will cause errors during training,
    because when we update the parameters of the generator (or discriminator),
    the parameters of the discriminator (or generator) is not updated, which is
    not allowed for DistributedDataParallel.
    So we design this wrapper to separately wrap DistributedDataParallel
    for generator and discriminator.
    In this wrapper, we perform two operations:
    1. Wrap the modules in the models with separate MMDistributedDataParallel.
        Note that only modules with parameters will be wrapped.
    2. Do scatter operation for 'forward', 'train_step' and 'val_step'.
    Note that the arguments of this wrapper is the same as those in
    ``torch.nn.parallel.distributed.DistributedDataParallel``.

    Args:
        module (nn.Module): Module that needs to be wrapped.
        device_ids (list[int | ``torch.device``]): Same as that in
            ``torch.nn.parallel.distributed.DistributedDataParallel``.
        dim (int, optional): Same as that in the official scatter function in
            pytorch. Defaults to 0.
        broadcast_buffers (bool): Same as that in
            ``torch.nn.parallel.distributed.DistributedDataParallel``.
            Defaults to False.
        find_unused_parameters (bool, optional): Same as that in
            ``torch.nn.parallel.distributed.DistributedDataParallel``.
            Traverse the autograd graph of all tensors contained in returned
            value of the wrapped module’s forward function. Defaults to False.
        kwargs (dict): Other arguments used in
            ``torch.nn.parallel.distributed.DistributedDataParallel``.
    """

    def __init__(self, module, device_ids, dim=0, broadcast_buffers=False, find_unused_parameters=False, **kwargs):
        super().__init__()
        assert len(device_ids) == 1, f'Currently, DistributedDataParallelWrapper only supports onesingle CUDA device for each process.The length of device_ids must be 1, but got {len(device_ids)}.'
        self.module = module
        self.dim = dim
        self.to_ddp(device_ids=device_ids, dim=dim, broadcast_buffers=broadcast_buffers, find_unused_parameters=find_unused_parameters, **kwargs)
        self.output_device = _get_device_index(device_ids[0], True)

    def to_ddp(self, device_ids, dim, broadcast_buffers, find_unused_parameters, **kwargs):
        """Wrap models with separate MMDistributedDataParallel.

        It only wraps the modules with parameters.
        """
        for name, module in self.module._modules.items():
            if next(module.parameters(), None) is None:
                module = module
            elif all(not p.requires_grad for p in module.parameters()):
                module = module
            else:
                module = MMDistributedDataParallel(module, device_ids=device_ids, dim=dim, broadcast_buffers=broadcast_buffers, find_unused_parameters=find_unused_parameters, **kwargs)
            self.module._modules[name] = module

    def scatter(self, inputs, kwargs, device_ids):
        """Scatter function.

        Args:
            inputs (Tensor): Input Tensor.
            kwargs (dict): Args for
                ``mmcv.parallel.scatter_gather.scatter_kwargs``.
            device_ids (int): Device id.
        """
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def forward(self, *inputs, **kwargs):
        """Forward function.

        Args:
            inputs (tuple): Input data.
            kwargs (dict): Args for
                ``mmcv.parallel.scatter_gather.scatter_kwargs``.
        """
        inputs, kwargs = self.scatter(inputs, kwargs, [torch.cuda.current_device()])
        return self.module(*inputs[0], **kwargs[0])

    def train_step(self, *inputs, **kwargs):
        """Train step function.

        Args:
            inputs (Tensor): Input Tensor.
            kwargs (dict): Args for
                ``mmcv.parallel.scatter_gather.scatter_kwargs``.
        """
        inputs, kwargs = self.scatter(inputs, kwargs, [torch.cuda.current_device()])
        output = self.module.train_step(*inputs[0], **kwargs[0])
        return output

    def val_step(self, *inputs, **kwargs):
        """Validation step function.

        Args:
            inputs (tuple): Input data.
            kwargs (dict): Args for ``scatter_kwargs``.
        """
        inputs, kwargs = self.scatter(inputs, kwargs, [torch.cuda.current_device()])
        output = self.module.val_step(*inputs[0], **kwargs[0])
        return output


class FactorizedReduce(nn.Module):
    """Reduce feature map size by factorized pointwise (stride=2)."""

    def __init__(self, in_channels, out_channels, act_cfg=dict(type='ReLU'), norm_cfg=dict(type='BN')):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act_cfg = act_cfg
        self.norm_cfg = norm_cfg
        self.relu = build_activation_layer(self.act_cfg)
        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels // 2, 1, stride=2, padding=0, bias=False)
        self.conv2 = nn.Conv2d(self.in_channels, self.out_channels // 2, 1, stride=2, padding=0, bias=False)
        self.bn = build_norm_layer(self.norm_cfg, self.out_channels)[1]

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv1(x), self.conv2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


class StandardConv(nn.Module):
    """
    Standard conv: ReLU - Conv - BN
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, act_cfg=dict(type='ReLU'), norm_cfg=dict(type='BN')):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.act_cfg = act_cfg
        self.norm_cfg = norm_cfg
        self.net = nn.Sequential(build_activation_layer(self.act_cfg), nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, bias=False), build_norm_layer(self.norm_cfg, self.out_channels)[1])

    def forward(self, x):
        return self.net(x)


class Placeholder(nn.Module):
    """Used for build searchable network.

    Args:
        group (str): Placeholder_group, as group name in searchable network.
        space_id (str): It is one and only index for each ``Placeholder``.
        choices (dict): Consist of the registered ``OPS``, used to combine
            ``MUTABLES``, the ``Placeholder`` will be replace with the
            ``MUTABLES``.
        choice_args (dict): The configuration of ``OPS`` used in choices.
    """

    def __init__(self, group, space_id, choices=None, choice_args=None):
        super(Placeholder, self).__init__()
        self.placeholder_group = group
        self.placeholder_kwargs = dict(space_id=space_id)
        if choices is not None:
            self.placeholder_kwargs.update(dict(choices=choices))
        if choice_args is not None:
            self.placeholder_kwargs.update(dict(choice_args=choice_args))


class Node(nn.Module):

    def __init__(self, node_id, num_prev_nodes, channels, num_downsample_nodes):
        super().__init__()
        edges = nn.ModuleDict()
        for i in range(num_prev_nodes):
            if i < num_downsample_nodes:
                stride = 2
            else:
                stride = 1
            edge_id = '{}_p{}'.format(node_id, i)
            edges.add_module(edge_id, nn.Sequential(Placeholder(group='node', space_id=edge_id, choice_args=dict(stride=stride, in_channels=channels, out_channels=channels))))
        self.edges = Placeholder(group='node_edge', space_id=node_id, choices=edges)

    def forward(self, prev_nodes):
        return self.edges(prev_nodes)


class Cell(nn.Module):

    def __init__(self, num_nodes, channels, prev_channels, prev_prev_channels, reduction, prev_reduction, act_cfg=dict(type='ReLU'), norm_cfg=dict(type='BN')):
        super().__init__()
        self.act_cfg = act_cfg
        self.norm_cfg = norm_cfg
        self.reduction = reduction
        self.num_nodes = num_nodes
        if prev_reduction:
            self.preproc0 = FactorizedReduce(prev_prev_channels, channels, self.act_cfg, self.norm_cfg)
        else:
            self.preproc0 = StandardConv(prev_prev_channels, channels, 1, 1, 0, self.act_cfg, self.norm_cfg)
        self.preproc1 = StandardConv(prev_channels, channels, 1, 1, 0, self.act_cfg, self.norm_cfg)
        self.nodes = nn.ModuleList()
        for depth in range(2, self.num_nodes + 2):
            if reduction:
                node_id = f'reduce_n{depth}'
                num_downsample_nodes = 2
            else:
                node_id = f'normal_n{depth}'
                num_downsample_nodes = 0
            self.nodes.append(Node(node_id, depth, channels, num_downsample_nodes))

    def forward(self, s0, s1):
        tensors = [self.preproc0(s0), self.preproc1(s1)]
        for node in self.nodes:
            cur_tensor = node(tensors)
            tensors.append(cur_tensor)
        output = torch.cat(tensors[2:], dim=1)
        return output


class AuxiliaryModule(nn.Module):
    """Auxiliary head in 2/3 place of network to let the gradient flow well."""

    def __init__(self, in_channels, base_channels, out_channels, norm_cfg=dict(type='BN')):
        super().__init__()
        self.norm_cfg = norm_cfg
        self.net = nn.Sequential(nn.ReLU(), nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False), nn.Conv2d(in_channels, base_channels, kernel_size=1, bias=False), build_norm_layer(self.norm_cfg, base_channels)[1], nn.ReLU(inplace=True), nn.Conv2d(base_channels, out_channels, kernel_size=2, bias=False), build_norm_layer(self.norm_cfg, out_channels)[1], nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)


class DartsBackbone(nn.Module):

    def __init__(self, in_channels, base_channels, num_layers=8, num_nodes=4, stem_multiplier=3, out_indices=(7,), auxliary=False, aux_channels=None, aux_out_channels=None, act_cfg=dict(type='ReLU'), norm_cfg=dict(type='BN')):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_layers = num_layers
        self.num_nodes = num_nodes
        self.stem_multiplier = stem_multiplier
        self.out_indices = out_indices
        assert self.out_indices[-1] == self.num_layers - 1
        if auxliary:
            assert aux_channels is not None
            assert aux_out_channels is not None
            self.aux_channels = aux_channels
            self.aux_out_channels = aux_out_channels
            self.auxliary_indice = 2 * self.num_layers // 3
        else:
            self.auxliary_indice = -1
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.out_channels = self.stem_multiplier * self.base_channels
        stem_norm_cfg = copy.deepcopy(self.norm_cfg)
        stem_norm_cfg.update(dict(affine=True))
        self.stem = nn.Sequential(nn.Conv2d(self.in_channels, self.out_channels, 3, 1, 1, bias=False), build_norm_layer(self.norm_cfg, self.out_channels)[1])
        prev_prev_channels = self.out_channels
        prev_channels = self.out_channels
        self.out_channels = self.base_channels
        self.cells = nn.ModuleList()
        prev_reduction, reduction = False, False
        for i in range(self.num_layers):
            prev_reduction, reduction = reduction, False
            if i == self.num_layers // 3 or i == 2 * self.num_layers // 3:
                self.out_channels *= 2
                reduction = True
            cell = Cell(self.num_nodes, self.out_channels, prev_channels, prev_prev_channels, reduction, prev_reduction, self.act_cfg, self.norm_cfg)
            self.cells.append(cell)
            prev_prev_channels = prev_channels
            prev_channels = self.out_channels * self.num_nodes
            if i == self.auxliary_indice:
                self.auxliary_module = AuxiliaryModule(prev_channels, self.aux_channels, self.aux_out_channels, self.norm_cfg)

    def forward(self, x):
        outs = []
        s0 = s1 = self.stem(x)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1)
            if i in self.out_indices:
                outs.append(s1)
            if i == self.auxliary_indice and self.training:
                aux_feature = self.auxliary_module(s1)
                outs.insert(0, aux_feature)
        return tuple(outs)


class ChannelWiseDivergence(nn.Module):
    """PyTorch version of `Channel-wise Distillation for Semantic Segmentation.

    <https://arxiv.org/abs/2011.13256>`_.

    Args:
        tau (float): Temperature coefficient. Defaults to 1.0.
        loss_weight (float): Weight of loss. Defaults to 1.0.
    """

    def __init__(self, tau=1.0, loss_weight=1.0):
        super(ChannelWiseDivergence, self).__init__()
        self.tau = tau
        self.loss_weight = loss_weight

    def forward(self, preds_S, preds_T):
        """Forward computation.

        Args:
            preds_S (torch.Tensor): The student model prediction with
                shape (N, C, H, W).
            preds_T (torch.Tensor): The teacher model prediction with
                shape (N, C, H, W).

        Return:
            torch.Tensor: The calculated loss value.
        """
        assert preds_S.shape[-2:] == preds_T.shape[-2:]
        N, C, H, W = preds_S.shape
        softmax_pred_T = F.softmax(preds_T.view(-1, W * H) / self.tau, dim=1)
        logsoftmax = torch.nn.LogSoftmax(dim=1)
        loss = torch.sum(softmax_pred_T * logsoftmax(preds_T.view(-1, W * H) / self.tau) - softmax_pred_T * logsoftmax(preds_S.view(-1, W * H) / self.tau)) * self.tau ** 2
        loss = self.loss_weight * loss / (C * N)
        return loss


class KLDivergence(nn.Module):
    """A measure of how one probability distribution Q is different from a
    second, reference probability distribution P.

    Args:
        tau (float): Temperature coefficient. Defaults to 1.0.
        reduction (str): Specifies the reduction to apply to the loss:
            ``'none'`` | ``'batchmean'`` | ``'sum'`` | ``'mean'``.
            ``'none'``: no reduction will be applied,
            ``'batchmean'``: the sum of the output will be divided by
                the batchsize,
            ``'sum'``: the output will be summed,
            ``'mean'``: the output will be divided by the number of
                elements in the output.
            Default: ``'batchmean'``
        loss_weight (float): Weight of loss. Defaults to 1.0.
    """

    def __init__(self, tau=1.0, reduction='batchmean', loss_weight=1.0):
        super(KLDivergence, self).__init__()
        self.tau = tau
        self.loss_weight = loss_weight
        accept_reduction = {'none', 'batchmean', 'sum', 'mean'}
        assert reduction in accept_reduction, f'KLDivergence supports reduction {accept_reduction}, but gets {reduction}.'
        self.reduction = reduction

    def forward(self, preds_S, preds_T):
        """Forward computation.

        Args:
            preds_S (torch.Tensor): The student model prediction with
                shape (N, C, H, W) or shape (N, C).
            preds_T (torch.Tensor): The teacher model prediction with
                shape (N, C, H, W) or shape (N, C).

        Return:
            torch.Tensor: The calculated loss value.
        """
        preds_T = preds_T.detach()
        softmax_pred_T = F.softmax(preds_T / self.tau, dim=1)
        logsoftmax_preds_S = F.log_softmax(preds_S / self.tau, dim=1)
        loss = self.tau ** 2 * F.kl_div(logsoftmax_preds_S, softmax_pred_T, reduction=self.reduction)
        return self.loss_weight * loss


def euclidean_distance(pred, squared=False, eps=1e-12):
    """Calculate the Euclidean distance between the two examples in the output
    representation space.

    Args:
        pred (torch.Tensor): The prediction of the teacher or student with
            shape (N, C).
        squared (bool): Whether to calculate the squared Euclidean
            distance. Defaults to False.
        eps (float): The minimum Euclidean distance between the two
            examples. Defaults to 1e-12.
    """
    pred_square = pred.pow(2).sum(dim=-1)
    prod = torch.mm(pred, pred.t())
    distance = (pred_square.unsqueeze(1) + pred_square.unsqueeze(0) - 2 * prod).clamp(min=eps)
    if not squared:
        distance = distance.sqrt()
    distance = distance.clone()
    distance[range(len(prod)), range(len(prod))] = 0
    return distance


class DistanceWiseRKD(nn.Module):
    """PyTorch version of distance-wise loss of `Relational Knowledge
    Distillation.

    <https://arxiv.org/abs/1904.05068>`_.

    Args:
        loss_weight (float): Weight of distance-wise distillation loss.
            Defaults to 25.0.
        with_l2_norm (bool): Whether to normalize the model predictions before
            calculating the loss. Defaults to True.
    """

    def __init__(self, loss_weight=25.0, with_l2_norm=True):
        super(DistanceWiseRKD, self).__init__()
        self.loss_weight = loss_weight
        self.with_l2_norm = with_l2_norm

    def distance_loss(self, preds_S, preds_T):
        """Calculate distance-wise distillation loss."""
        d_T = euclidean_distance(preds_T, squared=False)
        mean_d_T = d_T[d_T > 0].mean()
        d_T = d_T / mean_d_T
        d_S = euclidean_distance(preds_S, squared=False)
        mean_d_S = d_S[d_S > 0].mean()
        d_S = d_S / mean_d_S
        return F.smooth_l1_loss(d_S, d_T)

    def forward(self, preds_S, preds_T):
        """Forward computation.

        Args:
            preds_S (torch.Tensor): The student model prediction with
                shape (N, C, H, W) or shape (N, C).
            preds_T (torch.Tensor): The teacher model prediction with
                shape (N, C, H, W) or shape (N, C).
        Return:
            torch.Tensor: The calculated loss value.
        """
        preds_S = preds_S.view(preds_S.shape[0], -1)
        preds_T = preds_T.view(preds_T.shape[0], -1)
        if self.with_l2_norm:
            preds_S = F.normalize(preds_S, p=2, dim=1)
            preds_T = F.normalize(preds_T, p=2, dim=1)
        loss = self.distance_loss(preds_S, preds_T) * self.loss_weight
        return loss


def angle(pred):
    """Calculate the angle-wise relational potential which measures the angle
    formed by the three examples in the output representation space.

    Args:
        pred (torch.Tensor): The prediction of the teacher or student with
            shape (N, C).
    """
    pred_vec = pred.unsqueeze(0) - pred.unsqueeze(1)
    norm_pred_vec = F.normalize(pred_vec, p=2, dim=2)
    angle = torch.bmm(norm_pred_vec, norm_pred_vec.transpose(1, 2)).view(-1)
    return angle


class AngleWiseRKD(nn.Module):
    """PyTorch version of angle-wise loss of `Relational Knowledge
    Distillation.

    <https://arxiv.org/abs/1904.05068>`_.

    Args:
        loss_weight (float): Weight of angle-wise distillation loss.
            Defaults to 50.0.
        with_l2_norm (bool): Whether to normalize the model predictions before
            calculating the loss. Defaults to True.
    """

    def __init__(self, loss_weight=50.0, with_l2_norm=True):
        super(AngleWiseRKD, self).__init__()
        self.loss_weight = loss_weight
        self.with_l2_norm = with_l2_norm

    def angle_loss(self, preds_S, preds_T):
        """Calculate the angle-wise distillation loss."""
        angle_T = angle(preds_T)
        angle_S = angle(preds_S)
        return F.smooth_l1_loss(angle_S, angle_T)

    def forward(self, preds_S, preds_T):
        """Forward computation.

        Args:
            preds_S (torch.Tensor): The student model prediction with
                shape (N, C, H, W) or shape (N, C).
            preds_T (torch.Tensor): The teacher model prediction with
                shape (N, C, H, W) or shape (N, C).
        Return:
            torch.Tensor: The calculated loss value.
        """
        preds_S = preds_S.view(preds_S.shape[0], -1)
        preds_T = preds_T.view(preds_T.shape[0], -1)
        if self.with_l2_norm:
            preds_S = F.normalize(preds_S, p=2, dim=-1)
            preds_T = F.normalize(preds_T, p=2, dim=-1)
        loss = self.angle_loss(preds_S, preds_T) * self.loss_weight
        return loss


class WSLD(nn.Module):
    """PyTorch version of `Rethinking Soft Labels for Knowledge
    Distillation: A Bias-Variance Tradeoff Perspective
    <https://arxiv.org/abs/2102.00650>`_.

    Args:
        tau (float): Temperature coefficient. Defaults to 1.0.
        loss_weight (float): Weight of loss. Defaults to 1.0.
        num_classes (int): Defaults to 1000.
    """

    def __init__(self, tau=1.0, loss_weight=1.0, num_classes=1000):
        super(WSLD, self).__init__()
        self.tau = tau
        self.loss_weight = loss_weight
        self.num_classes = num_classes
        self.softmax = nn.Softmax(dim=1)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, student, teacher):
        gt_labels = self.current_data['gt_label']
        student_logits = student / self.tau
        teacher_logits = teacher / self.tau
        teacher_probs = self.softmax(teacher_logits)
        ce_loss = -torch.sum(teacher_probs * self.logsoftmax(student_logits), 1, keepdim=True)
        student_detach = student.detach()
        teacher_detach = teacher.detach()
        log_softmax_s = self.logsoftmax(student_detach)
        log_softmax_t = self.logsoftmax(teacher_detach)
        one_hot_labels = F.one_hot(gt_labels, num_classes=self.num_classes).float()
        ce_loss_s = -torch.sum(one_hot_labels * log_softmax_s, 1, keepdim=True)
        ce_loss_t = -torch.sum(one_hot_labels * log_softmax_t, 1, keepdim=True)
        focal_weight = ce_loss_s / (ce_loss_t + 1e-07)
        ratio_lower = torch.zeros(1)
        focal_weight = torch.max(focal_weight, ratio_lower)
        focal_weight = 1 - torch.exp(-focal_weight)
        ce_loss = focal_weight * ce_loss
        loss = self.tau ** 2 * torch.mean(ce_loss)
        loss = self.loss_weight * loss
        return loss


class SwitchableBatchNorm2d(nn.Module):
    """Employs independent batch normalization for different switches in a
    slimmable network.

    To train slimmable networks, ``SwitchableBatchNorm2d`` privatizes all
    batch normalization layers for each switch in a slimmable network.
    Compared with the naive training approach, it solves the problem of feature
    aggregation inconsistency between different switches by independently
    normalizing the feature mean and variance during testing.

    Args:
        max_num_features (int): The maximum ``num_features`` among BatchNorm2d
            in all the switches.
        num_bns (int): The number of different switches in the slimmable
            networks.
    """

    def __init__(self, max_num_features, num_bns):
        super(SwitchableBatchNorm2d, self).__init__()
        self.max_num_features = max_num_features
        self.num_bns = num_bns
        bns = []
        for _ in range(num_bns):
            bns.append(nn.BatchNorm2d(max_num_features))
        self.bns = nn.ModuleList(bns)
        self.index = 0

    def forward(self, input):
        """Forward computation according to the current switch of the slimmable
        networks."""
        return self.bns[self.index](input)


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
    (AngleWiseRKD,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (ChannelWiseDivergence,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (DistanceWiseRKD,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (ExampleBackbone,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (KLDivergence,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (SwitchableBatchNorm2d,
     lambda: ([], {'max_num_features': 4, 'num_bns': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_open_mmlab_mmrazor(_paritybench_base):
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

