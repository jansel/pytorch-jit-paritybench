import sys
_module = sys.modules[__name__]
del sys
setup = _module
tests = _module
test_data = _module
test_metric = _module
test_nn = _module
test_op = _module
test_summary = _module
test_tools = _module
test_transform = _module
torchtoolbox = _module
data = _module
dataprefetcher = _module
datasets = _module
lmdb_dataset = _module
utils = _module
metric = _module
feature_verification = _module
nn = _module
activation = _module
conv = _module
functional = _module
init = _module
loss = _module
norm = _module
operators = _module
EncodingDataParallel = _module
parallel = _module
sequential = _module
optimizer = _module
lookahead = _module
lr_scheduler = _module
tools = _module
convert_lmdb = _module
mixup = _module
reset_model_setting = _module
summary = _module
transform = _module
cutout = _module
transforms = _module

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


import torch


from torch import nn


import numpy as np


from torch.nn import functional as F


import numbers


import math


from torch.nn.init import xavier_normal_


from torch.nn.init import xavier_uniform_


from torch.nn.init import kaiming_normal_


from torch.nn.init import kaiming_uniform_


from torch.nn.init import zeros_


from torch.nn.modules.loss import _WeightedLoss


import functools


import torch.cuda.comm as comm


from torch.nn import Module


from itertools import chain


from torch.autograd import Function


from torch.nn.parallel.parallel_apply import get_a_var


from torch.nn.parallel.parallel_apply import parallel_apply


from torch.nn.parallel._functions import ReduceAddCoalesced


from torch.nn.parallel._functions import Broadcast


from torch.nn.parallel.scatter_gather import scatter_kwargs


from torch.nn.parallel.scatter_gather import gather


from torch.nn.parallel.replicate import replicate


from torch.nn.parallel.data_parallel import _check_balance


from torch.cuda._utils import _get_device_index


from torch._utils import ExceptionWrapper


from collections import OrderedDict


import torch.nn as nn


class n_to_n(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(3, 3, 1, 1, bias=False)

    def forward(self, x1, x2):
        y1 = self.conv1(x1)
        y2 = self.conv2(x2)
        return y1, y2


class n_to_one(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(3, 3, 1, 1, bias=False)

    def forward(self, x1, x2):
        y1 = self.conv1(x1)
        y2 = self.conv2(x2)
        return y1 + y2


class one_to_n(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(3, 3, 1, 1, bias=False)

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        return y1, y2


class SwishOP(Function):

    @staticmethod
    def forward(ctx, tensor, beta=1.0):
        ctx.save_for_backward(tensor)
        ctx.beta = beta
        swish = tensor / (1 + torch.exp(-beta * tensor))
        return swish

    @staticmethod
    def backward(ctx, grad_outputs):
        tensor = ctx.saved_tensors[0]
        beta = ctx.beta
        grad_swish = (torch.exp(-beta * tensor) * (1 + beta * tensor) + 1) / (1 + torch.exp(-beta * tensor)) ** 2
        grad_swish = grad_outputs * grad_swish
        return grad_swish, None


def swish(x, beta=1.0):
    """Swish activation.
    'https://arxiv.org/pdf/1710.05941.pdf'
    Args:
        x: Input tensor.
        beta:
    """
    return SwishOP.apply(x, beta)


class Swish(nn.Module):
    """Switch activation from 'SEARCHING FOR ACTIVATION FUNCTIONS'
        https://arxiv.org/pdf/1710.05941.pdf

        swish =  x / (1 + e^-beta*x)
        d_swish = (1 + (1+beta*x)) / ((1 + e^-beta*x)^2)

    """

    def __init__(self, beta=1.0):
        super(Swish, self).__init__()
        self.beta = beta

    def forward(self, x):
        return swish(x, self.beta)


class HardSwish(nn.Module):

    def __init__(self, inplace=False):
        super(HardSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3.0, inplace=self.inplace) / 6.0


class HardSigmoid(nn.Module):

    def __init__(self, inplace=False):
        super(HardSigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3.0, inplace=self.inplace) / 6.0


class Activation(nn.Module):

    def __init__(self, act_type, auto_optimize=True, **kwargs):
        super(Activation, self).__init__()
        if act_type == 'relu':
            self.act = nn.ReLU(inplace=True) if auto_optimize else nn.ReLU(**kwargs)
        elif act_type == 'relu6':
            self.act = nn.ReLU6(inplace=True) if auto_optimize else nn.ReLU6(**kwargs)
        elif act_type == 'h_swish':
            self.act = HardSwish(inplace=True) if auto_optimize else HardSwish(**kwargs)
        elif act_type == 'h_sigmoid':
            self.act = HardSigmoid(inplace=True) if auto_optimize else HardSigmoid(**kwargs)
        elif act_type == 'swish':
            self.act = Swish(**kwargs)
        elif act_type == 'sigmoid':
            self.act = nn.Sigmoid()
        elif act_type == 'lrelu':
            self.act = nn.LeakyReLU(inplace=True, **kwargs) if auto_optimize else nn.LeakyReLU(**kwargs)
        elif act_type == 'prelu':
            self.act = nn.PReLU(**kwargs)
        else:
            raise NotImplementedError('{} activation is not implemented.'.format(act_type))

    def forward(self, x):
        return self.act(x)


class DeformConv2d(nn.Module):

    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
        """
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)
        self.p_conv = nn.Conv2d(inc, 2 * kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_backward_hook(self._set_lr)
        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        offset = self.p_conv(x)
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))
        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2
        if self.padding:
            x = self.zero_padding(x)
        p = self._get_p(offset, dtype)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1
        q_lt = torch.cat([torch.clamp(q_lt[(...), :N], 0, x.size(2) - 1), torch.clamp(q_lt[(...), N:], 0, x.size(3) - 1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[(...), :N], 0, x.size(2) - 1), torch.clamp(q_rb[(...), N:], 0, x.size(3) - 1)], dim=-1).long()
        q_lb = torch.cat([q_lt[(...), :N], q_rb[(...), N:]], dim=-1)
        q_rt = torch.cat([q_rb[(...), :N], q_lt[(...), N:]], dim=-1)
        p = torch.cat([torch.clamp(p[(...), :N], 0, x.size(2) - 1), torch.clamp(p[(...), N:], 0, x.size(3) - 1)], dim=-1)
        g_lt = (1 + (q_lt[(...), :N].type_as(p) - p[(...), :N])) * (1 + (q_lt[(...), N:].type_as(p) - p[(...), N:]))
        g_rb = (1 - (q_rb[(...), :N].type_as(p) - p[(...), :N])) * (1 - (q_rb[(...), N:].type_as(p) - p[(...), N:]))
        g_lb = (1 + (q_lb[(...), :N].type_as(p) - p[(...), :N])) * (1 - (q_lb[(...), N:].type_as(p) - p[(...), N:]))
        g_rt = (1 - (q_rt[(...), :N].type_as(p) - p[(...), :N])) * (1 + (q_rt[(...), N:].type_as(p) - p[(...), N:]))
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + g_rb.unsqueeze(dim=1) * x_q_rb + g_lb.unsqueeze(dim=1) * x_q_lb + g_rt.unsqueeze(dim=1) * x_q_rt
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m
        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)
        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1), torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1))
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)
        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(torch.arange(1, h * self.stride + 1, self.stride), torch.arange(1, w * self.stride + 1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)
        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)
        p_n = self._get_p_n(N, dtype)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        x = x.contiguous().view(b, c, -1)
        index = q[(...), :N] * padded_w + q[(...), N:]
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)
        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)
        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[(...), s:s + ks].contiguous().view(b, c, h, w * ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h * ks, w * ks)
        return x_offset


class SigmoidCrossEntropy(_WeightedLoss):

    def __init__(self, classes, weight=None, reduction='mean'):
        super(SigmoidCrossEntropy, self).__init__(weight=weight, reduction=reduction)
        self.classes = classes

    def forward(self, pred, target):
        zt = BF.logits_distribution(pred, target, self.classes)
        return BF.logits_nll_loss(-F.logsigmoid(zt), target, self.weight, self.reduction)


class FocalLoss(_WeightedLoss):

    def __init__(self, classes, gamma, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__(weight=weight, reduction=reduction)
        self.classes = classes
        self.gamma = gamma

    def forward(self, pred, target):
        zt = BF.logits_distribution(pred, target, self.classes)
        ret = -(1 - torch.sigmoid(zt)).pow(self.gamma) * F.logsigmoid(zt)
        return BF.logits_nll_loss(ret, target, self.weight, self.reduction)


class L0Loss(nn.Module):
    """L0loss from
    "Noise2Noise: Learning Image Restoration without Clean Data"
    <https://arxiv.org/pdf/1803.04189>`_ paper.

    """

    def __init__(self, gamma=2, eps=1e-08):
        super(L0Loss, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, pred, target):
        loss = (torch.abs(pred - target) + self.eps).pow(self.gamma)
        return torch.mean(loss)


class LabelSmoothingLoss(nn.Module):
    """This is label smoothing loss function.
    """

    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        true_dist = BF.smooth_one_hot(target, self.cls, self.smoothing)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class L2Softmax(_WeightedLoss):
    """L2Softmax from
    `"L2-constrained Softmax Loss for Discriminative Face Verification"
    <https://arxiv.org/abs/1703.09507>`_ paper.

    Parameters
    ----------
    classes: int.
        Number of classes.
    alpha: float.
        The scaling parameter, a hypersphere with small alpha
        will limit surface area for embedding features.
    p: float, default is 0.9.
        The expected average softmax probability for correctly
        classifying a feature.
    from_normx: bool, default is False.
         Whether input has already been normalized.

    Outputs:
        - **loss**: loss tensor with shape (1,). Dimensions other than
          batch_axis are averaged out.
    """

    def __init__(self, classes, alpha, p=0.9, from_normx=False, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
        super(L2Softmax, self).__init__(weight, size_average, reduce, reduction)
        alpha_low = math.log(p * (classes - 2) / (1 - p))
        assert alpha > alpha_low, 'For given probability of p={}, alpha should higher than {}.'.format(p, alpha_low)
        self.ignore_index = ignore_index
        self.alpha = alpha
        self.from_normx = from_normx

    def forward(self, x, target):
        if not self.from_normx:
            x = F.normalize(x, 2, dim=-1)
        x = x * self.alpha
        return F.cross_entropy(x, target, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction)


class CosLoss(_WeightedLoss):
    """CosLoss from
       `"CosFace: Large Margin Cosine Loss for Deep Face Recognition"
       <https://arxiv.org/abs/1801.09414>`_ paper.

       It is also AM-Softmax from
       `"Additive Margin Softmax for Face Verification"
       <https://arxiv.org/abs/1801.05599>`_ paper.

    Parameters
    ----------
    classes: int.
        Number of classes.
    m: float, default 0.4
        Margin parameter for loss.
    s: int, default 64
        Scale parameter for loss.


    Outputs:
        - **loss**: loss tensor with shape (1,). Dimensions other than
          batch_axis are averaged out.
    """

    def __init__(self, classes, m, s, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
        super(CosLoss, self).__init__(weight, size_average, reduce, reduction)
        assert m > 0 and s > 0
        self.ignore_index = ignore_index
        self.classes = classes
        self.scale = s
        self.margin = m

    def forward(self, x, target):
        sparse_target = F.one_hot(target, num_classes=self.classes)
        x = x - sparse_target * self.margin
        x = x * self.scale
        return F.cross_entropy(x, target, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction)


class ArcLoss(_WeightedLoss):
    """ArcLoss from
    `"ArcFace: Additive Angular Margin Loss for Deep Face Recognition"
    <https://arxiv.org/abs/1801.07698>`_ paper.

    Parameters
    ----------
    classes: int.
        Number of classes.
    m: float.
        Margin parameter for loss.
    s: int.
        Scale parameter for loss.

    Outputs:
        - **loss**:
    """

    def __init__(self, classes, m=0.5, s=64, easy_margin=True, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
        super(ArcLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        assert s > 0.0
        assert 0 <= m <= math.pi / 2
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.mm = math.sin(math.pi - m) * m
        self.threshold = math.cos(math.pi - m)
        self.classes = classes
        self.easy_margin = easy_margin

    @torch.no_grad()
    def _get_body(self, x, target):
        cos_t = torch.gather(x, 1, target.unsqueeze(1))
        if self.easy_margin:
            cond = torch.relu(cos_t)
        else:
            cond_v = cos_t - self.threshold
            cond = torch.relu(cond_v)
        cond = cond.bool()
        new_zy = torch.cos(torch.acos(cos_t) + self.m).type(cos_t.dtype)
        if self.easy_margin:
            zy_keep = cos_t
        else:
            zy_keep = cos_t - self.mm
        new_zy = torch.where(cond, new_zy, zy_keep)
        diff = new_zy - cos_t
        gt_one_hot = F.one_hot(target, num_classes=self.classes)
        body = gt_one_hot * diff
        return body

    def forward(self, x, target):
        body = self._get_body(x, target)
        x = x + body
        x = x * self.s
        return F.cross_entropy(x, target, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction)


class CircleLoss(nn.Module):
    """CircleLoss from
    `"Circle Loss: A Unified Perspective of Pair Similarity Optimization"
    <https://arxiv.org/pdf/2002.10857>`_ paper.

    Parameters
    ----------
    m: float.
        Margin parameter for loss.
    gamma: int.
        Scale parameter for loss.

    Outputs:
        - **loss**: scalar.
    """

    def __init__(self, m, gamma):
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.dp = 1 - m
        self.dn = m

    def forward(self, x, target):
        similarity_matrix = x @ x.T
        label_matrix = target.unsqueeze(1) == target.unsqueeze(0)
        negative_matrix = label_matrix.logical_not()
        positive_matrix = label_matrix.fill_diagonal_(False)
        sp = torch.where(positive_matrix, similarity_matrix, torch.zeros_like(similarity_matrix))
        sn = torch.where(negative_matrix, similarity_matrix, torch.zeros_like(similarity_matrix))
        ap = torch.clamp_min(1 + self.m - sp.detach(), min=0.0)
        an = torch.clamp_min(sn.detach() + self.m, min=0.0)
        logit_p = -self.gamma * ap * (sp - self.dp)
        logit_n = self.gamma * an * (sn - self.dn)
        logit_p = torch.where(positive_matrix, logit_p, torch.zeros_like(logit_p))
        logit_n = torch.where(negative_matrix, logit_n, torch.zeros_like(logit_n))
        loss = F.softplus(torch.logsumexp(logit_p, dim=1) + torch.logsumexp(logit_n, dim=1)).mean()
        return loss


class RingLoss(nn.Module):
    """Computes the Ring Loss from
    `"Ring loss: Convex Feature Normalization for Face Recognition"

    Parameters
    ----------
    lamda: float
        The loss weight enforcing a trade-off between the softmax loss and ring loss.
    l2_norm: bool
        Whether use l2 norm to embedding.
    weight_initializer (None or torch.Tensor): If not None a torch.Tensor should be provided.

    Outputs:
        - **loss**: scalar.
    """

    def __init__(self, lamda, l2_norm=True, weight_initializer=None):
        super(RingLoss, self).__init__()
        self.lamda = lamda
        self.l2_norm = l2_norm
        if weight_initializer is None:
            self.R = self.parameters(torch.rand(1))
        else:
            assert torch.is_tensor(weight_initializer), 'weight_initializer should be a Tensor.'
            self.R = self.parameters(weight_initializer)

    def forward(self, embedding):
        if self.l2_norm:
            embedding = F.normalize(embedding, 2, dim=-1)
        loss = (embedding - self.R).pow(2).sum(1).mean(0) * self.lamda * 0.5
        return loss


class CenterLoss(nn.Module):
    """Computes the Center Loss from
    `"A Discriminative Feature Learning Approach for Deep Face Recognition"
    <http://ydwen.github.io/papers/WenECCV16.pdf>`_paper.
    Implementation is refer to
    'https://github.com/lyakaap/image-feature-learning-pytorch/blob/master/code/center_loss.py'

    Parameters
    ----------
    classes: int.
        Number of classes.
    embedding_dim: int
        embedding_dim.
    lamda: float
        The loss weight enforcing a trade-off between the softmax loss and center loss.

    Outputs:
        - **loss**: loss tensor with shape (batch_size,). Dimensions other than
          batch_axis are averaged out.
    """

    def __init__(self, classes, embedding_dim, lamda):
        super(CenterLoss, self).__init__()
        self.lamda = lamda
        self.centers = nn.Parameter(torch.randn(classes, embedding_dim))

    def forward(self, embedding, target):
        expanded_centers = self.centers.index_select(0, target)
        intra_distances = embedding.dist(expanded_centers)
        loss = self.lamda * 0.5 * intra_distances / target.size()[0]
        return loss


class _SwitchNorm(nn.Module):
    """
    Avoid to feed 1xCxHxW and NxCx1x1 data to this.
    """
    _version = 2

    def __init__(self, num_features, eps=1e-05, momentum=0.9, affine=True):
        super(_SwitchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features))
            self.bias = nn.Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.mean_weight = nn.Parameter(torch.ones(3))
        self.var_weight = nn.Parameter(torch.ones(3))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def _check_input_dim(self, x):
        raise NotImplementedError

    def forward(self, x):
        self._check_input_dim(x)
        return F.switch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, self.mean_weight, self.var_weight, self.training, self.momentum, self.eps)


class _EvoNorm(nn.Module):

    def __init__(self, prefix, num_features, eps=1e-05, momentum=0.9, groups=32, affine=True):
        super(_EvoNorm, self).__init__()
        assert prefix in ('s0', 'b0')
        self.prefix = prefix
        self.groups = groups
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
            self.bias = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
            self.v = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
            self.register_parameter('v', None)
        self.register_buffer('running_var', torch.ones(1, num_features, 1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)
            torch.nn.init.ones_(self.v)

    def _check_input_dim(self, x):
        if x.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(x.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        return F.evo_norm(x, self.prefix, self.running_var, self.v, self.weight, self.bias, self.training, self.momentum, self.eps, self.groups)


class EncodingParallel(Module):

    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super(EncodingParallel, self).__init__()
        if not torch.is_available():
            self.module = module
            self.device_ids = []
            return
        if device_ids is None:
            device_ids = list(range(torch.device_count()))
        if output_device is None:
            output_device = device_ids[0]
        self.dim = dim
        self.module = module
        self.device_ids = list(map(lambda x: _get_device_index(x, True), device_ids))
        self.output_device = _get_device_index(output_device, True)
        self.src_device_obj = torch.device('cuda {}'.format(self.device_ids[0]))
        _check_balance(self.device_ids)
        if len(self.device_ids) == 1:
            self.module

    def replicate(self, module, device_ids):
        return replicate(module, device_ids, not torch.is_grad_enabled())

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)


class AdaptiveSequential(nn.Sequential):
    """Make Sequential could handle multiple input/output layer.

    Example:
        class n_to_n(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 3, 1, 1, bias=False)
                self.conv2 = nn.Conv2d(3, 3, 1, 1, bias=False)

            def forward(self, x1, x2):
                y1 = self.conv1(x1)
                y2 = self.conv2(x2)
                return y1, y2


        class n_to_one(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 3, 1, 1, bias=False)
                self.conv2 = nn.Conv2d(3, 3, 1, 1, bias=False)

            def forward(self, x1, x2):
                y1 = self.conv1(x1)
                y2 = self.conv2(x2)
                return y1 + y2


        class one_to_n(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 3, 1, 1, bias=False)
                self.conv2 = nn.Conv2d(3, 3, 1, 1, bias=False)

            def forward(self, x):
                y1 = self.conv1(x)
                y2 = self.conv2(x)
                return y1, y2

        seq = AdaptiveSequential(one_to_n(), n_to_n(), n_to_one()).cuda()
        td = torch.rand(1, 3, 32, 32).cuda()

        out = seq(td)
        print(out.size())
        # torch.Size([1, 3, 32, 32])

    """

    def forward(self, *inputs):
        for module in self._modules.values():
            if isinstance(inputs, tuple):
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AdaptiveSequential,
     lambda: ([], {}),
     lambda: ([], {}),
     False),
    (CircleLoss,
     lambda: ([], {'m': 4, 'gamma': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (DeformConv2d,
     lambda: ([], {'inc': 4, 'outc': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (HardSigmoid,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (HardSwish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (L0Loss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Swish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (n_to_n,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64]), torch.rand([4, 3, 64, 64])], {}),
     True),
    (n_to_one,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64]), torch.rand([4, 3, 64, 64])], {}),
     True),
    (one_to_n,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
]

class Test_PistonY_torch_toolbox(_paritybench_base):
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

