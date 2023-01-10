import sys
_module = sys.modules[__name__]
del sys
affinity_loss = _module
amsoftmax = _module
conv_ops = _module
dice_loss = _module
dual_focal_loss = _module
ema = _module
focal_loss = _module
generalized_iou_loss = _module
hswish = _module
info_nce_dist = _module
iou_loss = _module
label_smooth = _module
large_margin_softmax = _module
lovasz_softmax = _module
mish = _module
one_hot = _module
partial_fc_amsoftmax = _module
pc_softmax = _module
pytorch_loss = _module
affinity_loss = _module
amsoftmax = _module
conv_ops = _module
dice_loss = _module
dual_focal_loss = _module
ema = _module
focal_loss = _module
focal_loss_old = _module
frelu = _module
group_loss = _module
hswish = _module
info_nce_dist = _module
iou_loss = _module
label_smooth = _module
large_margin_softmax = _module
layer_norm = _module
lovasz_softmax = _module
mish = _module
ohem_loss = _module
one_hot = _module
partial_fc_amsoftmax = _module
pc_softmax = _module
soft_dice_loss = _module
swish = _module
taylor_softmax = _module
test = _module
triplet_loss = _module
setup = _module
soft_dice_loss = _module
swish = _module
taylor_softmax = _module
triplet_loss = _module

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


import torch


import torch.nn as nn


import torch.nn.functional as F


import torch.cuda.amp as amp


import torch.distributed as dist


import math


import torchvision


import numpy as np


import random


from torch.utils import cpp_extension


@torch.no_grad()
def convert_to_one_hot(x, minleng, ignore_idx=-1):
    """
    encode input x into one hot
    inputs:
        x: tensor of shape (N, ...) with type long
        minleng: minimum length of one hot code, this should be larger than max value in x
        ignore_idx: the index in x that should be ignored, default is 255

    return:
        tensor of shape (N, minleng, ...) with type float
    """
    device = x.device
    size = list(x.size())
    size.insert(1, minleng)
    assert x[x != ignore_idx].max() < minleng, 'minleng should larger than max value in x'
    if ignore_idx < 0:
        out = torch.zeros(size, device=device).scatter_(1, x.unsqueeze(1), 1)
    else:
        x = x.clone().detach()
        ignore = x == ignore_idx
        x[ignore] = 0
        out = torch.zeros(size, device=device).scatter_(1, x.unsqueeze(1), 1)
        ignore = ignore.nonzero(as_tuple=False)
        _, M = ignore.size()
        a, *b = ignore.chunk(M, dim=1)
        out[[a, torch.arange(minleng), *b]] = 0
    return out


class AffinityLoss(nn.Module):

    def __init__(self, kernel_size=3, ignore_index=-100):
        super(AffinityLoss, self).__init__()
        self.kernel_size = kernel_size
        self.ignore_index = ignore_index
        self.unfold = nn.Unfold(kernel_size=kernel_size)
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, logits, labels):
        """
        usage similar to nn.CrossEntropyLoss:
            >>> criteria = AffinityLoss(kernel_size=3, ignore_index=255)
            >>> logits = torch.randn(8, 19, 384, 384) # nchw
            >>> lbs = torch.randint(0, 19, (8, 384, 384)) # nhw
            >>> loss = criteria(logits, lbs)
        """
        n, c, h, w = logits.size()
        context_size = self.kernel_size * self.kernel_size
        lb_one_hot = convert_to_one_hot(labels, c, self.ignore_index).detach()
        logits_unfold = self.unfold(logits).view(n, c, context_size, -1)
        lbs_unfold = self.unfold(lb_one_hot).view(n, c, context_size, -1)
        aff_map = torch.einsum('ncal,ncbl->nabl', logits_unfold, logits_unfold)
        lb_map = torch.einsum('ncal,ncbl->nabl', lbs_unfold, lbs_unfold)
        loss = self.bce(aff_map, lb_map)
        return loss


class AffinityFieldLoss(nn.Module):
    """
        loss proposed in the paper: https://arxiv.org/abs/1803.10335
        used for sigmentation tasks
    """

    def __init__(self, kl_margin, lambda_edge=1.0, lambda_not_edge=1.0, ignore_lb=255):
        super(AffinityFieldLoss, self).__init__()
        self.kl_margin = kl_margin
        self.ignore_lb = ignore_lb
        self.lambda_edge = lambda_edge
        self.lambda_not_edge = lambda_not_edge
        self.kldiv = nn.KLDivLoss(reduction='none')

    def forward(self, logits, labels):
        ignore_mask = labels.cpu() == self.ignore_lb
        n_valid = ignore_mask.numel() - ignore_mask.sum().item()
        indices = [((1, None, None, None), (None, -1, None, None)), ((None, -1, None, None), (1, None, None, None)), ((None, None, 1, None), (None, None, None, -1)), ((None, None, None, -1), (None, None, 1, None)), ((1, None, 1, None), (None, -1, None, -1)), ((1, None, None, -1), (None, -1, 1, None)), ((None, -1, 1, None), (1, None, None, -1)), ((None, -1, None, -1), (1, None, 1, None))]
        losses = []
        probs = torch.softmax(logits, dim=1)
        log_probs = torch.log_softmax(logits, dim=1)
        for idx_c, idx_e in indices:
            lbcenter = labels[:, idx_c[0]:idx_c[1], idx_c[2]:idx_c[3]].detach()
            lbedge = labels[:, idx_e[0]:idx_e[1], idx_e[2]:idx_e[3]].detach()
            igncenter = ignore_mask[:, idx_c[0]:idx_c[1], idx_c[2]:idx_c[3]].detach()
            ignedge = ignore_mask[:, idx_e[0]:idx_e[1], idx_e[2]:idx_e[3]].detach()
            lgp_center = probs[:, :, idx_c[0]:idx_c[1], idx_c[2]:idx_c[3]]
            lgp_edge = probs[:, :, idx_e[0]:idx_e[1], idx_e[2]:idx_e[3]]
            prob_edge = probs[:, :, idx_e[0]:idx_e[1], idx_e[2]:idx_e[3]]
            kldiv = (prob_edge * (lgp_edge - lgp_center)).sum(dim=1)
            kldiv[ignedge | igncenter] = 0
            loss = torch.where(lbcenter == lbedge, self.lambda_edge * kldiv, self.lambda_not_edge * F.relu(self.kl_margin - kldiv, inplace=True)).sum() / n_valid
            losses.append(loss)
        return sum(losses) / 8


class AMSoftmax(nn.Module):

    def __init__(self, in_feats, n_classes=10, m=0.3, s=15):
        super(AMSoftmax, self).__init__()
        self.m = m
        self.s = s
        self.in_feats = in_feats
        self.W = torch.nn.Parameter(torch.randn(in_feats, n_classes), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.W, gain=1)

    def forward(self, x, lb):
        assert x.size()[0] == lb.size()[0]
        assert x.size()[1] == self.in_feats
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-09)
        x_norm = torch.div(x, x_norm)
        w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-09)
        w_norm = torch.div(self.W, w_norm)
        costh = torch.mm(x_norm, w_norm)
        delt_costh = torch.zeros_like(costh).scatter_(1, lb.unsqueeze(1), self.m)
        costh_m = costh - delt_costh
        costh_m_s = self.s * costh_m
        loss = self.ce(costh_m_s, lb)
        return loss


class CoordConv2d(nn.Conv2d):

    def __init__(self, in_chan, out_chan, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
        super(CoordConv2d, self).__init__(in_chan + 2, out_chan, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        batchsize, H, W = x.size(0), x.size(2), x.size(3)
        h_range = torch.linspace(-1, 1, H, device=x.device, dtype=x.dtype)
        w_range = torch.linspace(-1, 1, W, device=x.device, dtype=x.dtype)
        h_chan, w_chan = torch.meshgrid(h_range, w_range)
        h_chan = h_chan.expand([batchsize, 1, -1, -1])
        w_chan = w_chan.expand([batchsize, 1, -1, -1])
        feat = torch.cat([h_chan, w_chan, x], dim=1)
        return F.conv2d(feat, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class DY_Conv2d(nn.Conv2d):

    def __init__(self, in_chan, out_chan, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False, act=nn.ReLU(inplace=True), K=4, temperature=30, temp_anneal_steps=3000):
        super(DY_Conv2d, self).__init__(in_chan, out_chan * K, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        assert in_chan // 4 > 0
        self.K = K
        self.act = act
        self.se_conv1 = nn.Conv2d(in_chan, in_chan // 4, 1, 1, 0, bias=True)
        self.se_conv2 = nn.Conv2d(in_chan // 4, K, 1, 1, 0, bias=True)
        self.temperature = temperature
        self.temp_anneal_steps = temp_anneal_steps
        self.temp_interval = (temperature - 1) / temp_anneal_steps

    def get_atten(self, x):
        bs, _, h, w = x.size()
        atten = torch.mean(x, dim=(2, 3), keepdim=True)
        atten = self.se_conv1(atten)
        atten = self.act(atten)
        atten = self.se_conv2(atten)
        if self.training and self.temp_anneal_steps > 0:
            atten = atten / self.temperature
            self.temperature -= self.temp_interval
            self.temp_anneal_steps -= 1
        atten = atten.softmax(dim=1).view(bs, -1)
        return atten

    def forward(self, x):
        bs, _, h, w = x.size()
        atten = self.get_atten(x)
        out_chan, in_chan, k1, k2 = self.weight.size()
        W = self.weight.view(1, self.K, -1, in_chan, k1, k2)
        W = (W * atten.view(bs, self.K, 1, 1, 1, 1)).sum(dim=1)
        W = W.view(-1, in_chan, k1, k2)
        b = self.bias
        if not b is None:
            b = b.view(1, self.K, -1)
            b = (b * atten.view(bs, self.K, 1)).sum(dim=1).view(-1)
        x = x.view(1, -1, h, w)
        out = F.conv2d(x, W, b, self.stride, self.padding, self.dilation, self.groups * bs)
        out = out.view(bs, -1, out.size(2), out.size(3))
        return out


class GeneralizedSoftDiceLoss(nn.Module):

    def __init__(self, p=1, smooth=1, reduction='mean', weight=None, ignore_lb=255):
        super(GeneralizedSoftDiceLoss, self).__init__()
        self.p = p
        self.smooth = smooth
        self.reduction = reduction
        self.weight = None if weight is None else torch.tensor(weight)
        self.ignore_lb = ignore_lb

    def forward(self, logits, label):
        """
        args: logits: tensor of shape (N, C, H, W)
        args: label: tensor of shape(N, H, W)
        """
        logits = logits.float()
        ignore = label.data.cpu() == self.ignore_lb
        label = label.clone()
        label[ignore] = 0
        lb_one_hot = torch.zeros_like(logits).scatter_(1, label.unsqueeze(1), 1)
        ignore = ignore.nonzero()
        _, M = ignore.size()
        a, *b = ignore.chunk(M, dim=1)
        lb_one_hot[[a, torch.arange(lb_one_hot.size(1)).long(), *b]] = 0
        lb_one_hot = lb_one_hot.detach()
        probs = torch.sigmoid(logits)
        numer = torch.sum(probs * lb_one_hot, dim=(2, 3))
        denom = torch.sum(probs.pow(self.p) + lb_one_hot.pow(self.p), dim=(2, 3))
        if not self.weight is None:
            numer = numer * self.weight.view(1, -1)
            denom = denom * self.weight.view(1, -1)
        numer = torch.sum(numer, dim=1)
        denom = torch.sum(denom, dim=1)
        loss = 1 - (2 * numer + self.smooth) / (denom + self.smooth)
        if self.reduction == 'mean':
            loss = loss.mean()
        return loss


class BatchSoftDiceLoss(nn.Module):

    def __init__(self, p=1, smooth=1, weight=None, ignore_lb=255):
        super(BatchSoftDiceLoss, self).__init__()
        self.p = p
        self.smooth = smooth
        self.weight = None if weight is None else torch.tensor(weight)
        self.ignore_lb = ignore_lb

    def forward(self, logits, label):
        """
        args: logits: tensor of shape (N, C, H, W)
        args: label: tensor of shape(N, H, W)
        """
        logits = logits.float()
        ignore = label.data.cpu() == self.ignore_lb
        label = label.clone()
        label[ignore] = 0
        lb_one_hot = torch.zeros_like(logits).scatter_(1, label.unsqueeze(1), 1)
        ignore = ignore.nonzero()
        _, M = ignore.size()
        a, *b = ignore.chunk(M, dim=1)
        lb_one_hot[[a, torch.arange(lb_one_hot.size(1)).long(), *b]] = 0
        lb_one_hot = lb_one_hot.detach()
        probs = torch.sigmoid(logits)
        numer = torch.sum(probs * lb_one_hot, dim=(2, 3))
        denom = torch.sum(probs.pow(self.p) + lb_one_hot.pow(self.p), dim=(2, 3))
        if not self.weight is None:
            numer = numer * self.weight.view(1, -1)
            denom = denom * self.weight.view(1, -1)
        numer = torch.sum(numer)
        denom = torch.sum(denom)
        loss = 1 - (2 * numer + self.smooth) / (denom + self.smooth)
        return loss


class Dual_Focal_loss(nn.Module):
    """
    This loss is proposed in this paper: https://arxiv.org/abs/1909.11932
    It does not work in my projects, hope it will work well in your projects.
    Hope you can correct me if there are any mistakes in the implementation.
    """

    def __init__(self, ignore_lb=255, eps=1e-05, reduction='mean'):
        super(Dual_Focal_loss, self).__init__()
        self.ignore_lb = ignore_lb
        self.eps = eps
        self.reduction = reduction
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, logits, label):
        ignore = label.data.cpu() == self.ignore_lb
        n_valid = (ignore == 0).sum()
        label = label.clone()
        label[ignore] = 0
        lb_one_hot = logits.data.clone().zero_().scatter_(1, label.unsqueeze(1), 1).detach()
        pred = torch.softmax(logits, dim=1)
        loss = -torch.log(self.eps + 1.0 - self.mse(pred, lb_one_hot)).sum(dim=1)
        loss[ignore] = 0
        if self.reduction == 'mean':
            loss = loss.sum() / n_valid
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'none':
            loss = loss
        return loss


class FocalLossV1(nn.Module):

    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLossV1, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, label):
        """
        logits and label have same shape, and label data type is long
        args:
            logits: tensor of shape (N, ...)
            label: tensor of shape(N, ...)
        Usage is like this:
            >>> criteria = FocalLossV1()
            >>> logits = torch.randn(8, 19, 384, 384)# nchw, float/half
            >>> lbs = torch.randint(0, 19, (8, 384, 384)) # nchw, int64_t
            >>> loss = criteria(logits, lbs)
        """
        logits = logits.float()
        with torch.no_grad():
            alpha = torch.empty_like(logits).fill_(1 - self.alpha)
            alpha[label == 1] = self.alpha
        probs = torch.sigmoid(logits)
        pt = torch.where(label == 1, probs, 1 - probs)
        ce_loss = self.crit(logits, label.float())
        loss = alpha * torch.pow(1 - pt, self.gamma) * ce_loss
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss


class FocalSigmoidLossFuncV2(torch.autograd.Function):
    """
    compute backward directly for better numeric stability
    """

    @staticmethod
    @amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, logits, label, alpha, gamma):
        logits = logits.float()
        coeff = torch.empty_like(logits).fill_(1 - alpha)
        coeff[label == 1] = alpha
        probs = torch.sigmoid(logits)
        log_probs = torch.where(logits >= 0, F.softplus(logits, -1, 50), logits - F.softplus(logits, 1, 50))
        log_1_probs = torch.where(logits >= 0, -logits + F.softplus(logits, -1, 50), -F.softplus(logits, 1, 50))
        probs_gamma = probs ** gamma
        probs_1_gamma = (1.0 - probs) ** gamma
        ctx.vars = coeff, probs, log_probs, log_1_probs, probs_gamma, probs_1_gamma, label, gamma
        term1 = probs_1_gamma * log_probs
        term2 = probs_gamma * log_1_probs
        loss = torch.where(label == 1, term1, term2).mul_(coeff).neg_()
        return loss

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_output):
        """
        compute gradient of focal loss
        """
        coeff, probs, log_probs, log_1_probs, probs_gamma, probs_1_gamma, label, gamma = ctx.vars
        term1 = (1.0 - probs - gamma * probs * log_probs).mul_(probs_1_gamma).neg_()
        term2 = (probs - gamma * (1.0 - probs) * log_1_probs).mul_(probs_gamma)
        grads = torch.where(label == 1, term1, term2).mul_(coeff).mul_(grad_output)
        return grads, None, None, None


class FocalLossV2(nn.Module):
    """
    This use better formula to compute the gradient, which has better numeric stability
    Usage is like this:
        >>> criteria = FocalLossV2()
        >>> logits = torch.randn(8, 19, 384, 384)# nchw, float/half
        >>> lbs = torch.randint(0, 19, (8, 384, 384)) # nchw, int64_t
        >>> loss = criteria(logits, lbs)
    """

    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLossV2, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, label):
        loss = FocalSigmoidLossFuncV2.apply(logits, label, self.alpha, self.gamma)
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss


class FocalSigmoidLossFuncV3(torch.autograd.Function):
    """
    use cpp/cuda to accelerate and shrink memory usage
    """

    @staticmethod
    @amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, logits, labels, alpha, gamma):
        logits = logits.float()
        loss = focal_cpp.focalloss_forward(logits, labels, gamma, alpha)
        ctx.variables = logits, labels, alpha, gamma
        return loss

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_output):
        """
        compute gradient of focal loss
        """
        logits, labels, alpha, gamma = ctx.variables
        grads = focal_cpp.focalloss_backward(grad_output, logits, labels, gamma, alpha)
        return grads, None, None, None


class FocalLossV3(nn.Module):
    """
    This use better formula to compute the gradient, which has better numeric stability
    Usage is like this:
        >>> criteria = FocalLossV3()
        >>> logits = torch.randn(8, 19, 384, 384)# nchw, float/half
        >>> lbs = torch.randint(0, 19, (8, 384, 384)) # nchw, int64_t
        >>> loss = criteria(logits, lbs)
    """

    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLossV3, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, label):
        loss = FocalSigmoidLossFuncV3.apply(logits, label, self.alpha, self.gamma)
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss


class HSwishV1(nn.Module):

    def __init__(self):
        super(HSwishV1, self).__init__()

    def forward(self, feat):
        return feat * F.relu6(feat + 3) / 6


class HSwishFunctionV2(torch.autograd.Function):

    @staticmethod
    @amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, feat):
        act = F.relu6(feat + 3).mul_(feat).div_(6)
        ctx.variables = feat
        return act

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_output):
        feat = ctx.variables
        grad = F.relu6(feat + 3).div_(6)
        grad.add_(torch.where(torch.eq(-3 < feat, feat < 3), torch.ones_like(feat).div_(6), torch.zeros_like(feat)).mul_(feat))
        grad *= grad_output
        return grad


class HSwishV2(nn.Module):

    def __init__(self):
        super(HSwishV2, self).__init__()

    def forward(self, feat):
        return HSwishFunctionV2.apply(feat)


class HSwishFunctionV3(torch.autograd.Function):

    @staticmethod
    @amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, feat):
        ctx.feat = feat
        return swish_cpp.hswish_forward(feat)

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_output):
        feat = ctx.feat
        return swish_cpp.hswish_backward(grad_output, feat)


class HSwishV3(nn.Module):

    def __init__(self):
        super(HSwishV3, self).__init__()

    def forward(self, feat):
        return HSwishFunctionV3.apply(feat)


class InfoNceFunction(torch.autograd.Function):

    @staticmethod
    @amp.custom_fwd
    def forward(ctx, embs1, embs2, temper_factor, margin):
        assert embs1.size() == embs2.size()
        N, C = embs1.size()
        dtype = embs1.dtype
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        device = embs1.device
        all_embs1 = torch.zeros(size=[N * world_size, C], dtype=dtype)
        dist.all_gather(list(all_embs1.chunk(world_size, dim=0)), embs1)
        all_embs2 = torch.zeros(size=[N * world_size, C], dtype=dtype)
        dist.all_gather(list(all_embs2.chunk(world_size, dim=0)), embs2)
        all_embs = torch.cat([all_embs1, all_embs2], dim=0)
        embs12 = torch.cat([embs1, embs2], dim=0)
        logits = torch.einsum('ac,bc->ab', embs12, all_embs)
        inds1 = torch.arange(N * 2)
        inds2 = torch.cat([torch.arange(N) + rank * N, torch.arange(N) + (rank + world_size) * N], dim=0)
        logits[inds1, inds2] = -10000.0
        labels = inds2.view(2, -1).flip(dims=(0,)).reshape(-1)
        logits[inds1, labels] -= margin
        logits *= temper_factor
        ctx.vars = inds1, inds2, embs12, all_embs, temper_factor
        return logits, labels

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_logits, grad_label):
        inds1, inds2, embs12, all_embs, temper_factor = ctx.vars
        grad_logits = grad_logits * temper_factor
        grad_logits[inds1, inds2] = 0
        grad_embs12 = torch.einsum('ab,bc->ac', grad_logits, all_embs)
        grad_all_embs = torch.einsum('ab,ac->bc', grad_logits, embs12)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        N = int(all_embs.size(0) / (world_size * 2))
        grad_embs1 = grad_embs12[:N] + grad_all_embs[rank * N:(rank + 1) * N]
        grad_embs2 = grad_embs12[N:] + grad_all_embs[(rank + world_size) * N:(rank + world_size + 1) * N]
        return grad_embs1, grad_embs2, None, None


class InfoNceDist(nn.Module):

    def __init__(self, temper=0.1, margin=0.0):
        super(InfoNceDist, self).__init__()
        self.crit = nn.CrossEntropyLoss()
        self.margin = margin
        self.temp_factor = 1.0 / temper

    def forward(self, embs1, embs2):
        """
        embs1, embs2: n x c, one by one pairs
            1 positive, 2n - 2 negative
            distributed mode, no need to wrap with nn.DistributedParallel
        """
        embs1 = F.normalize(embs1, dim=1)
        embs2 = F.normalize(embs2, dim=1)
        logits, labels = InfoNceFunction.apply(embs1, embs2, self.temp_factor, self.margin)
        loss = self.crit(logits, labels.detach())
        return loss


def iou_func(gt_bboxes, pr_bboxes, eps=1e-05):
    """
    input:
        gt_bboxes: tensor (N, 4) xyxy
        pr_bboxes: tensor (N, 4) xyxy
    output:
        gious: tensor (N, )
    """
    gt_area = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (gt_bboxes[:, 3] - gt_bboxes[:, 1])
    pr_area = (pr_bboxes[:, 2] - pr_bboxes[:, 0]) * (pr_bboxes[:, 3] - pr_bboxes[:, 1])
    lt = torch.max(gt_bboxes[:, :2], pr_bboxes[:, :2])
    rb = torch.min(gt_bboxes[:, 2:], pr_bboxes[:, 2:])
    wh = (rb - lt + eps).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    union = gt_area + pr_area - inter
    iou = inter / union
    return iou


def giou_func(gt_bboxes, pr_bboxes, eps=1e-05):
    """
    input:
        gt_bboxes: tensor (N, 4) xyxy
        pr_bboxes: tensor (N, 4) xyxy
    output:
        gious: tensor (N, )
    """
    iou = iou_func(gt_bboxes, pr_bboxes, eps)
    lt = torch.min(gt_bboxes[:, :2], pr_bboxes[:, :2])
    rb = torch.max(gt_bboxes[:, 2:], pr_bboxes[:, 2:])
    wh = (rb - lt + eps).clamp(min=0)
    enclosure = wh[:, 0] * wh[:, 1]
    giou = iou - (enclosure - union) / enclosure
    return giou


class GIOULoss(nn.Module):

    def __init__(self, eps=1e-05, reduction='mean'):
        super(GIOULoss, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, pr_bboxes, gt_bboxes):
        """
        pr_bboxes: tensor (-1, 4) xyxy, predicted bbox
        gt_bboxes: tensor (-1, 4) xyxy, ground truth bbox
        loss proposed in the paper of giou
        """
        giou = giou_func(gt_bboxes, pr_bboxes, self.eps)
        loss = 1.0 - giou
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'none':
            pass
        return loss


def diou_func(gt_bboxes, pr_bboxes, eps=1e-05):
    """
    input:
        gt_bboxes: tensor (N, 4) xyxy
        pr_bboxes: tensor (N, 4) xyxy
    output:
        dious: tensor (N, )
    """
    iou = iou_func(gt_bboxes, pr_bboxes, eps)
    gt_cent_x = gt_bboxes[:, 0::2].mean(dim=-1)
    gt_cent_y = gt_bboxes[:, 1::2].mean(dim=-1)
    pr_cent_x = pr_bboxes[:, 0::2].mean(dim=-1)
    pr_cent_y = pr_bboxes[:, 1::2].mean(dim=-1)
    cent_dis = (gt_cent_x - pr_cent_x).pow(2.0) + (gt_cent_y - pr_cent_y).pow(2.0)
    lt = torch.min(gt_bboxes[:, :2], pr_bboxes[:, :2])
    rb = torch.max(gt_bboxes[:, 2:], pr_bboxes[:, 2:])
    diag_dis = (lt - rb).pow(2).sum(dim=-1)
    reg = cent_dis / (diag_dis + eps)
    diou = iou - reg
    return diou


class DIOULoss(nn.Module):

    def __init__(self, eps=1e-05, reduction='mean'):
        super(DIOULoss, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, pr_bboxes, gt_bboxes):
        """
        pr_bboxes: tensor (-1, 4) xyxy, predicted bbox
        gt_bboxes: tensor (-1, 4) xyxy, ground truth bbox
        loss proposed in the paper of giou
        """
        diou = diou_func(gt_bboxes, pr_bboxes, self.eps)
        loss = 1.0 - diou
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'none':
            pass
        return loss


class CIOURegFunc(torch.autograd.Function):
    """
    forward and backward of CIOU regularization term
    """

    @staticmethod
    @amp.custom_fwd
    def forward(ctx, gt_bboxes, pr_bboxes, eps=1e-05):
        gt_w = gt_bboxes[:, 2] - gt_bboxes[:, 0]
        gt_h = gt_bboxes[:, 3] - gt_bboxes[:, 1]
        pr_w = pr_bboxes[:, 2] - pr_bboxes[:, 0]
        pr_h = pr_bboxes[:, 3] - pr_bboxes[:, 1]
        coef = 4.0 / math.pi ** 2
        atan_diff = torch.atan(gt_w / gt_h) - torch.atan(pr_w / pr_h)
        v = atan_diff.pow(2.0)
        v = coef * v
        iou = iou_func(gt_bboxes, pr_bboxes, eps)
        alpha = v / (1 - iou + v)
        reg = alpha * v
        h2_w2 = 1.0
        dv = 2 * coef * atan_diff * h2_w2 * alpha
        dv_dh = dv * pr_w
        dv_dw = -dv * pr_h
        dx1, dx2 = -dv_dw.view(-1, 1), dv_dw.view(-1, 1)
        dy1, dy2 = -dv_dh.view(-1, 1), dv_dh.view(-1, 1)
        d_pr_bbox = torch.cat([dx1, dy1, dx2, dy2], dim=-1)
        h2_w2 = 1.0
        dv = 2 * coef * atan_diff * h2_w2 * alpha
        dv_dh = dv * gt_w
        dv_dw = -dv * gt_h
        dx1, dx2 = -dv_dw.view(-1, 1), dv_dw.view(-1, 1)
        dy1, dy2 = -dv_dh.view(-1, 1), dv_dh.view(-1, 1)
        d_gt_bbox = -torch.cat([dx1, dy1, dx2, dy2], dim=-1)
        ctx.variables = d_gt_bbox, d_pr_bbox
        return reg

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_output):
        d_gt_bbox, d_pr_bbox = ctx.variables
        return d_gt_bbox, d_pr_bbox, None


def ciou_func(gt_bboxes, pr_bboxes, eps=1e-05):
    """
    input:
        gt_bboxes: tensor (N, 4) xyxy
        pr_bboxes: tensor (N, 4) xyxy
    output:
        cious: tensor (N, )
    """
    diou = diou_func(gt_bboxes, pr_bboxes, eps)
    creg = CIOURegFunc.apply(gt_bboxes, pr_bboxes, eps)
    ciou = diou - creg
    return ciou


class CIOULoss(nn.Module):

    def __init__(self, eps=1e-05, reduction='sum'):
        super(CIOULoss, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, pr_bboxes, gt_bboxes):
        """
        pr_bboxes: tensor (-1, 4) xyxy, predicted bbox
        gt_bboxes: tensor (-1, 4) xyxy, ground truth bbox
        loss proposed in the paper of giou
        """
        ciou = ciou_func(gt_bboxes, pr_bboxes, self.eps)
        loss = 1.0 - ciou
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'none':
            pass
        return loss


class LabelSmoothSoftmaxCEV1(nn.Module):
    """
    This is the autograd version, you can also try the LabelSmoothSoftmaxCEV2 that uses derived gradients
    """

    def __init__(self, lb_smooth=0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothSoftmaxCEV1, self).__init__()
        self.lb_smooth = lb_smooth
        self.reduction = reduction
        self.lb_ignore = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, logits, label):
        """
        Same usage method as nn.CrossEntropyLoss:
            >>> criteria = LabelSmoothSoftmaxCEV1()
            >>> logits = torch.randn(8, 19, 384, 384) # nchw, float/half
            >>> lbs = torch.randint(0, 19, (8, 384, 384)) # nhw, int64_t
            >>> loss = criteria(logits, lbs)
        """
        logits = logits.float()
        with torch.no_grad():
            num_classes = logits.size(1)
            label = label.clone().detach()
            ignore = label.eq(self.lb_ignore)
            n_valid = ignore.eq(0).sum()
            label[ignore] = 0
            lb_pos, lb_neg = 1.0 - self.lb_smooth, self.lb_smooth / num_classes
            lb_one_hot = torch.empty_like(logits).fill_(lb_neg).scatter_(1, label.unsqueeze(1), lb_pos).detach()
        logs = self.log_softmax(logits)
        loss = -torch.sum(logs * lb_one_hot, dim=1)
        loss[ignore] = 0
        if self.reduction == 'mean':
            loss = loss.sum() / n_valid
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss


class LSRCrossEntropyFunctionV2(torch.autograd.Function):

    @staticmethod
    @amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, logits, label, lb_smooth, lb_ignore):
        num_classes = logits.size(1)
        lb_pos, lb_neg = 1.0 - lb_smooth, lb_smooth / num_classes
        label = label.clone().detach()
        ignore = label.eq(lb_ignore)
        n_valid = ignore.eq(0).sum()
        label[ignore] = 0
        lb_one_hot = torch.empty_like(logits).fill_(lb_neg).scatter_(1, label.unsqueeze(1), lb_pos).detach()
        ignore = ignore.nonzero(as_tuple=False)
        _, M = ignore.size()
        a, *b = ignore.chunk(M, dim=1)
        mask = [a, torch.arange(logits.size(1)), *b]
        lb_one_hot[mask] = 0
        coeff = (num_classes - 1) * lb_neg + lb_pos
        ctx.variables = coeff, mask, logits, lb_one_hot
        loss = torch.log_softmax(logits, dim=1).neg_().mul_(lb_one_hot).sum(dim=1)
        return loss

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_output):
        coeff, mask, logits, lb_one_hot = ctx.variables
        scores = torch.softmax(logits, dim=1).mul_(coeff)
        grad = scores.sub_(lb_one_hot).mul_(grad_output.unsqueeze(1))
        grad[mask] = 0
        return grad, None, None, None


class LabelSmoothSoftmaxCEV2(nn.Module):

    def __init__(self, lb_smooth=0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothSoftmaxCEV2, self).__init__()
        self.lb_smooth = lb_smooth
        self.reduction = reduction
        self.lb_ignore = ignore_index

    def forward(self, logits, labels):
        """
        Same usage method as nn.CrossEntropyLoss:
            >>> criteria = LabelSmoothSoftmaxCEV2()
            >>> logits = torch.randn(8, 19, 384, 384) # nchw, float/half
            >>> lbs = torch.randint(0, 19, (8, 384, 384)) # nhw, int64_t
            >>> loss = criteria(logits, lbs)
        """
        losses = LSRCrossEntropyFunctionV2.apply(logits, labels, self.lb_smooth, self.lb_ignore)
        if self.reduction == 'sum':
            losses = losses.sum()
        elif self.reduction == 'mean':
            n_valid = (labels != self.lb_ignore).sum()
            losses = losses.sum() / n_valid
        return losses


class LSRCrossEntropyFunctionV3(torch.autograd.Function):
    """
    use cpp/cuda to accelerate and shrink memory usage
    """

    @staticmethod
    @amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, logits, labels, lb_smooth, lb_ignore):
        losses = lsr_cpp.lsr_forward(logits, labels, lb_ignore, lb_smooth)
        ctx.variables = logits, labels, lb_ignore, lb_smooth
        return losses

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_output):
        logits, labels, lb_ignore, lb_smooth = ctx.variables
        grad = lsr_cpp.lsr_backward(logits, labels, lb_ignore, lb_smooth)
        grad.mul_(grad_output.unsqueeze(1))
        return grad, None, None, None


class LabelSmoothSoftmaxCEV3(nn.Module):

    def __init__(self, lb_smooth=0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothSoftmaxCEV3, self).__init__()
        self.lb_smooth = lb_smooth
        self.reduction = reduction
        self.lb_ignore = ignore_index

    def forward(self, logits, labels):
        """
        Same usage method as nn.CrossEntropyLoss:
            >>> criteria = LabelSmoothSoftmaxCEV3()
            >>> logits = torch.randn(8, 19, 384, 384) # nchw, float/half
            >>> lbs = torch.randint(0, 19, (8, 384, 384)) # nhw, int64_t
            >>> loss = criteria(logits, lbs)
        """
        losses = LSRCrossEntropyFunctionV3.apply(logits, labels, self.lb_smooth, self.lb_ignore)
        if self.reduction == 'sum':
            losses = losses.sum()
        elif self.reduction == 'mean':
            n_valid = (labels != self.lb_ignore).sum()
            losses = losses.sum() / n_valid
        return losses


class LargeMarginSoftmaxV1(nn.Module):

    def __init__(self, lam=0.3, reduction='mean', ignore_index=255):
        super(LargeMarginSoftmaxV1, self).__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.lam = lam
        self.ce_crit = nn.CrossEntropyLoss(reduction='none', ignore_index=ignore_index)

    def forward(self, logits, label):
        """
        Same usage method as nn.CrossEntropyLoss:
            >>> criteria = LargeMarginSoftmaxV1()
            >>> logits = torch.randn(8, 19, 384, 384) # nchw, float/half
            >>> lbs = torch.randint(0, 19, (8, 384, 384)) # nhw, int64_t
            >>> loss = criteria(logits, lbs)
        """
        logits = logits.float()
        logits.retain_grad()
        logits.register_hook(lambda grad: grad)
        with torch.no_grad():
            num_classes = logits.size(1)
            coeff = 1.0 / (num_classes - 1.0)
            lb = label.clone().detach()
            mask = label == self.ignore_index
            lb[mask] = 0
            idx = torch.zeros_like(logits).scatter_(1, lb.unsqueeze(1), 1.0)
        lgts = logits - idx * 1000000.0
        q = lgts.softmax(dim=1)
        q = q * (1.0 - idx)
        log_q = lgts.log_softmax(dim=1)
        log_q = log_q * (1.0 - idx)
        mg_loss = (q - coeff) * log_q * (self.lam / 2)
        mg_loss = mg_loss * (1.0 - idx)
        mg_loss = mg_loss.sum(dim=1)
        ce_loss = self.ce_crit(logits, label)
        loss = ce_loss + mg_loss
        loss = loss[mask == 0]
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss


class LargeMarginSoftmaxFuncV2(torch.autograd.Function):

    @staticmethod
    @amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, logits, labels, lam=0.3):
        num_classes = logits.size(1)
        coeff = 1.0 / (num_classes - 1.0)
        idx = torch.zeros_like(logits).scatter_(1, labels.unsqueeze(1), 1.0)
        lgts = logits.clone()
        lgts[idx.bool()] = -1000000.0
        q = lgts.softmax(dim=1)
        log_q = lgts.log_softmax(dim=1)
        losses = q.sub_(coeff).mul_(log_q).mul_(lam / 2.0)
        losses[idx.bool()] = 0
        losses = losses.sum(dim=1).add_(F.cross_entropy(logits, labels, reduction='none'))
        ctx.variables = logits, labels, idx, coeff, lam
        return losses

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_output):
        """
        compute gradient
        """
        logits, labels, idx, coeff, lam = ctx.variables
        num_classes = logits.size(1)
        p = logits.softmax(dim=1)
        lgts = logits.clone()
        lgts[idx.bool()] = -1000000.0
        q = lgts.softmax(dim=1)
        qx = q * lgts
        qx[idx.bool()] = 0
        grad = qx + q - q * qx.sum(dim=1).unsqueeze(1) - coeff
        grad = grad * lam / 2.0
        grad[idx.bool()] = -1
        grad = grad + p
        grad.mul_(grad_output.unsqueeze(1))
        return grad, None, None


class LargeMarginSoftmaxV2(nn.Module):

    def __init__(self, lam=0.3, reduction='mean', ignore_index=255):
        super(LargeMarginSoftmaxV2, self).__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.lam = lam

    def forward(self, logits, labels):
        """
        Same usage method as nn.CrossEntropyLoss:
            >>> criteria = LargeMarginSoftmaxV2()
            >>> logits = torch.randn(8, 19, 384, 384) # nchw, float/half
            >>> lbs = torch.randint(0, 19, (8, 384, 384)) # nhw, int64_t
            >>> loss = criteria(logits, lbs)
        """
        logits = logits.float()
        mask = labels == self.ignore_index
        lb = labels.clone().detach()
        lb[mask] = 0
        loss = LargeMarginSoftmaxFuncV2.apply(logits, lb, self.lam)
        loss = loss[mask == 0]
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


class LargeMarginSoftmaxFuncV3(torch.autograd.Function):
    """
    use cpp/cuda to accelerate and shrink memory usage
    """

    @staticmethod
    @amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, logits, labels, lam=0.3, ignore_index=255):
        losses = large_margin_cpp.l_margin_forward(logits, labels, lam, ignore_index)
        ctx.variables = logits, labels, lam, ignore_index
        return losses

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_output):
        """
        compute gradient
        """
        logits, labels, lam, ignore_index = ctx.variables
        grads = large_margin_cpp.l_margin_backward(logits, labels, lam, ignore_index)
        grads.mul_(grad_output.unsqueeze(1))
        return grads, None, None, None


class LargeMarginSoftmaxV3(nn.Module):

    def __init__(self, lam=0.3, reduction='mean', ignore_index=255):
        super(LargeMarginSoftmaxV3, self).__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.lam = lam

    def forward(self, logits, labels):
        """
        Same usage method as nn.CrossEntropyLoss:
            >>> criteria = LargeMarginSoftmaxV3()
            >>> logits = torch.randn(8, 19, 384, 384) # nchw, float/half
            >>> lbs = torch.randint(0, 19, (8, 384, 384)) # nhw, int64_t
            >>> loss = criteria(logits, lbs)
        """
        logits = logits.float()
        losses = LargeMarginSoftmaxFuncV3.apply(logits, labels, self.lam, self.ignore_index)
        if self.reduction == 'mean':
            n_valid = (labels != self.ignore_index).sum()
            losses = losses.sum() / n_valid
        elif self.reduction == 'sum':
            losses = losses.sum()
        return losses


class LovaszSoftmaxV1(nn.Module):
    """
    This is the autograd version, used in the multi-category classification case
    """

    def __init__(self, reduction='mean', ignore_index=-100):
        super(LovaszSoftmaxV1, self).__init__()
        self.reduction = reduction
        self.lb_ignore = ignore_index

    def forward(self, logits, label):
        """
        Same usage method as nn.CrossEntropyLoss:
            >>> criteria = LovaszSoftmaxV1()
            >>> logits = torch.randn(8, 19, 384, 384) # nchw, float/half
            >>> lbs = torch.randint(0, 19, (8, 384, 384)) # nhw, int64_t
            >>> loss = criteria(logits, lbs)
        """
        n, c, h, w = logits.size()
        logits = logits.transpose(0, 1).reshape(c, -1).float()
        label = label.view(-1)
        idx = label.ne(self.lb_ignore).nonzero(as_tuple=False).squeeze()
        probs = logits.softmax(dim=0)[:, idx]
        label = label[idx]
        lb_one_hot = torch.zeros_like(probs).scatter_(0, label.unsqueeze(0), 1).detach()
        errs = (lb_one_hot - probs).abs()
        errs_sort, errs_order = torch.sort(errs, dim=1, descending=True)
        n_samples = errs.size(1)
        with torch.no_grad():
            lb_one_hot_sort = torch.cat([lb_one_hot[i, ord].unsqueeze(0) for i, ord in enumerate(errs_order)], dim=0)
            n_pos = lb_one_hot_sort.sum(dim=1, keepdim=True)
            inter = n_pos - lb_one_hot_sort.cumsum(dim=1)
            union = n_pos + (1.0 - lb_one_hot_sort).cumsum(dim=1)
            jacc = 1.0 - inter / union
            if n_samples > 1:
                jacc[:, 1:] = jacc[:, 1:] - jacc[:, :-1]
        losses = torch.einsum('ab,ab->a', errs_sort, jacc)
        if self.reduction == 'sum':
            losses = losses.sum()
        elif self.reduction == 'mean':
            losses = losses.mean()
        return losses, errs


class LovaszSoftmaxFunctionV3(torch.autograd.Function):

    @staticmethod
    @amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, logits, labels, ignore_index):
        losses, jacc = lovasz_softmax_cpp.lovasz_softmax_forward(logits, labels, ignore_index)
        ctx.vars = logits, labels, jacc, ignore_index
        return losses

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_output):
        logits, labels, jacc, ignore_index = ctx.vars
        grad = lovasz_softmax_cpp.lovasz_softmax_backward(grad_output, logits, labels, jacc, ignore_index)
        return grad, None, None


class LovaszSoftmaxV3(nn.Module):
    """
    """

    def __init__(self, reduction='mean', ignore_index=-100):
        super(LovaszSoftmaxV3, self).__init__()
        self.reduction = reduction
        self.lb_ignore = ignore_index

    def forward(self, logits, label):
        """
        Same usage method as nn.CrossEntropyLoss:
            >>> criteria = LovaszSoftmaxV3()
            >>> logits = torch.randn(8, 19, 384, 384) # nchw, float/half
            >>> lbs = torch.randint(0, 19, (8, 384, 384)) # nhw, int64_t
            >>> loss = criteria(logits, lbs)
        """
        losses = LovaszSoftmaxFunctionV3.apply(logits, label, self.lb_ignore)
        if self.reduction == 'sum':
            losses = losses.sum()
        elif self.reduction == 'mean':
            losses = losses.mean()
        return losses


class MishV1(nn.Module):

    def __init__(self):
        super(MishV1, self).__init__()

    def forward(self, feat):
        return feat * torch.tanh(F.softplus(feat))


class MishFunctionV2(torch.autograd.Function):

    @staticmethod
    @amp.custom_fwd
    def forward(ctx, feat):
        tanhX = torch.tanh(F.softplus(feat))
        out = feat * tanhX
        grad = tanhX + feat * (1 - torch.pow(tanhX, 2)) * torch.sigmoid(feat)
        ctx.grad = grad
        return out

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_output):
        grad = ctx.grad
        grad *= grad_output
        return grad


class MishV2(nn.Module):

    def __init__(self):
        super(MishV2, self).__init__()

    def forward(self, feat):
        return MishFunctionV2.apply(feat)


class MishFunctionV3(torch.autograd.Function):

    @staticmethod
    @amp.custom_fwd
    def forward(ctx, feat):
        ctx.feat = feat
        return mish_cpp.mish_forward(feat)

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_output):
        feat = ctx.feat
        return mish_cpp.mish_backward(grad_output, feat)


class MishV3(nn.Module):

    def __init__(self):
        super(MishV3, self).__init__()

    def forward(self, feat):
        return MishFunctionV3.apply(feat)


def convert_to_one_hot_cu(x, minleng, smooth=0.0, ignore_idx=-1):
    """
    cuda version of encoding x into one hot, the difference from above is that, this support label smooth.
    inputs:
        x: tensor of shape (N, ...) with type long
        minleng: minimum length of one hot code, this should be larger than max value in x
        smooth: sets positive to **1. - smooth**, while sets negative to **smooth / minleng**
        ignore_idx: the index in x that should be ignored, default is 255

    return:
        tensor of shape (N, minleng, ...) with type float32
    """
    return one_hot_cpp.label_one_hot(x, ignore_idx, smooth, minleng)


class OnehotEncoder(nn.Module):

    def __init__(self, n_classes, lb_smooth=0.0, ignore_idx=-1):
        super(OnehotEncoder, self).__init__()
        self.n_classes = n_classes
        self.lb_smooth = lb_smooth
        self.ignore_idx = ignore_idx

    @torch.no_grad()
    def forward(self, label):
        return convert_to_one_hot_cu(label, self.n_classes, self.lb_smooth, self.ignore_idx).detach()


class GatherFunction(torch.autograd.Function):

    @staticmethod
    @amp.custom_fwd
    def forward(ctx, embs, lbs):
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        N, C = embs.size()
        e_dtype = embs.dtype
        l_dtype = lbs.dtype
        device = embs.device
        embs = embs.contiguous()
        all_embs = torch.zeros(size=[N * world_size, C], dtype=e_dtype, device=device)
        dist.all_gather(list(all_embs.chunk(world_size, dim=0)), embs)
        lbs = lbs.contiguous()
        all_lbs = torch.zeros(size=[N * world_size], dtype=l_dtype, device=device)
        dist.all_gather(list(all_lbs.chunk(world_size, dim=0)), lbs)
        return all_embs, all_lbs

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_all_embs, grad_all_lbs):
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        N = int(grad_all_embs.size(0) / world_size)
        grads_embs = grad_all_embs[rank * N:(rank + 1) * N]
        return grads_embs, None


class PartialFCFunction(torch.autograd.Function):

    @staticmethod
    @amp.custom_fwd
    def forward(ctx, all_embs, W, ind1, ind2, n_pos, s, m):
        assert all_embs.size(1) == W.size(0)
        N, C = all_embs.size()
        n_ids = W.size(1)
        e_dtype = all_embs.dtype
        device = all_embs.device
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        logits = torch.einsum('ab,bc->ac', all_embs, W)
        if n_pos > 0:
            logits[ind1, ind2] -= m
        logits *= s
        logits = logits.float()
        l_max = logits.max(dim=1, keepdim=True)[0]
        dist.all_reduce(l_max, dist.ReduceOp.MAX)
        logits -= l_max
        l_exp = logits.exp_()
        l_exp_sum = l_exp.sum(dim=1, keepdim=True)
        dist.all_reduce(l_exp_sum, dist.ReduceOp.SUM)
        softmax = l_exp.div_(l_exp_sum)
        softmax = softmax
        loss = torch.zeros(all_embs.size(0), dtype=e_dtype, device=device)
        if n_pos > 0:
            prob = softmax[ind1, ind2]
            loss[ind1] = prob.log().neg()
        dist.all_reduce(loss, dist.ReduceOp.SUM)
        ctx.vars = softmax, ind1, ind2, n_pos, s, W, all_embs
        return loss

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_output):
        softmax, ind1, ind2, n_pos, s, W, all_embs = ctx.vars
        world_size = dist.get_world_size()
        grads = softmax
        if n_pos > 0:
            grads[ind1, ind2] -= 1
        grads *= grad_output.view(-1, 1)
        grads *= s
        grads_embs = torch.einsum('ac,bc->ab', grads, W).mul_(world_size)
        dist.all_reduce(grads_embs, dist.ReduceOp.SUM)
        grads_W = torch.einsum('ac,ab->cb', all_embs, grads)
        return grads_embs, grads_W, None, None, None, None, None


class SampleFunction(torch.autograd.Function):

    @staticmethod
    @amp.custom_fwd
    def forward(ctx, W, lb, ratio):
        assert ratio < 1.0, 'do not call this unless ratio should less than 1.'
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        device = lb.device
        Wshape = W.size()
        n_ids = W.size(1)
        n_sample = int(n_ids * ratio) + 1
        lb_unq = lb.unique(sorted=True)
        pos_ind1 = lb_unq.div(n_ids, rounding_mode='trunc') == rank
        pos_ind2 = lb_unq[pos_ind1] % n_ids
        id_n_pos = pos_ind1.sum()
        id_n_neg = max(0, n_sample - id_n_pos)
        ind1 = lb.div(n_ids, rounding_mode='trunc') == rank
        ind2 = lb[ind1] % n_ids
        n_pos = ind1.sum()
        if id_n_pos == n_ids:
            keep_ind = torch.arange(n_ids, device=device)
            ctx.vars = keep_ind, Wshape
            return W, ind1, ind2, n_pos
        if id_n_neg == 0:
            keep_ind = ind2
        elif id_n_pos == 0:
            keep_ind = torch.randperm(n_ids, device=device)[:id_n_neg]
        else:
            neg_mask = torch.ones(n_ids, device=device)
            neg_mask[pos_ind2] = 0
            neg_ind = neg_mask.nonzero()[:, 0]
            neg_mask = torch.randperm(neg_ind.size(0), device=device)[:id_n_neg]
            neg_ind = neg_ind[neg_mask]
            keep_ind = torch.cat([pos_ind2, neg_ind], dim=0)
        W = W[:, keep_ind]
        if n_pos > 0:
            ind2 = (ind2.unsqueeze(1) == pos_ind2.unsqueeze(0)).nonzero()[:, 1]
        ctx.vars = keep_ind, Wshape
        return W, ind1, ind2, n_pos

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_W, grad_ind1, grad_ind2, grad_n_pos):
        keep_ind, Wshape = ctx.vars
        grad = torch.zeros(Wshape, dtype=grad_W.dtype, device=grad_W.device)
        grad[:, keep_ind] = grad_W
        return grad, None, None


class PartialFCAMSoftmax(nn.Module):

    def __init__(self, emb_dim, n_ids=10, m=0.3, s=15, ratio=1.0, reduction='mean'):
        super(PartialFCAMSoftmax, self).__init__()
        assert dist.is_initialized(), 'must initialize distributed before create this'
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        assert n_ids % world_size == 0, 'number of ids should be divisible among gpus. please drop some ids, which should make trivial differences'
        self.n_ids = int(n_ids / world_size)
        self.emb_dim = emb_dim
        assert ratio > 0.0 and ratio <= 1.0, 'sample ratio should be in (0., 1.]'
        self.m, self.s, self.ratio = m, s, ratio
        self.W = torch.nn.Parameter(torch.randn(emb_dim, self.n_ids), requires_grad=True)
        nn.init.xavier_normal_(self.W, gain=1)
        self.reduction = reduction

    def forward(self, x, lb):
        assert x.size()[0] == lb.size()[0]
        assert x.size()[1] == self.emb_dim
        x, lb = GatherFunction.apply(x, lb)
        if self.ratio < 1.0:
            W, ind1, ind2, n_pos = SampleFunction.apply(self.W, lb, self.ratio)
        else:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            W = self.W
            ind1 = lb.div(self.n_ids, rounding_mode='trunc') == rank
            ind2 = lb[ind1] % self.n_ids
            n_pos = ind1.sum()
        x_norm = F.normalize(x, dim=1)
        w_norm = F.normalize(W, dim=0)
        loss = PartialFCFunction.apply(x_norm, w_norm, ind1, ind2, n_pos, self.s, self.m)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


def pc_softmax_func(logits, lb_proportion):
    assert logits.size(1) == len(lb_proportion)
    shape = [1, -1] + [(1) for _ in range(len(logits.size()) - 2)]
    W = torch.tensor(lb_proportion).view(*shape).detach()
    logits = logits - logits.max(dim=1, keepdim=True)[0]
    exp = torch.exp(logits)
    pc_softmax = exp.div_((W * exp).sum(dim=1, keepdim=True))
    return pc_softmax


class PCSoftmax(nn.Module):

    def __init__(self, lb_proportion):
        super(PCSoftmax, self).__init__()
        self.weight = lb_proportion

    def forward(self, logits):
        return pc_softmax_func(logits, self.weight)


class PCSoftmaxCrossEntropyV1(nn.Module):

    def __init__(self, lb_proportion, ignore_index=255, reduction='mean'):
        super(PCSoftmaxCrossEntropyV1, self).__init__()
        self.weight = torch.tensor(lb_proportion).detach()
        self.nll = nn.NLLLoss(reduction=reduction, ignore_index=ignore_index)

    def forward(self, logits, label):
        shape = [1, -1] + [(1) for _ in range(len(logits.size()) - 2)]
        W = self.weight.view(*shape).detach()
        logits = logits - logits.max(dim=1, keepdim=True)[0]
        wexp_sum = torch.exp(logits).mul(W).sum(dim=1, keepdim=True)
        log_wsoftmax = logits - torch.log(wexp_sum)
        loss = self.nll(log_wsoftmax, label)
        return loss


class PCSoftmaxCrossEntropyFunction(torch.autograd.Function):

    @staticmethod
    @amp.custom_fwd
    def forward(ctx, logits, label, lb_proportion, reduction, ignore_index):
        label = label.clone().detach()
        ignore = label == ignore_index
        n_valid = (ignore == 0).sum()
        label[ignore] = 0
        lb_one_hot = torch.zeros_like(logits).scatter_(1, label.unsqueeze(1), 1).detach()
        shape = [1, -1] + [(1) for _ in range(len(logits.size()) - 2)]
        W = torch.tensor(lb_proportion).view(*shape).detach()
        logits = logits - logits.max(dim=1, keepdim=True)[0]
        exp_wsum = torch.exp(logits).mul_(W).sum(dim=1, keepdim=True)
        ignore = ignore.nonzero()
        _, M = ignore.size()
        a, *b = ignore.chunk(M, dim=1)
        mask = [a, torch.arange(lb_one_hot.size(1)), *b]
        lb_one_hot[mask] = 0
        ctx.mask = mask
        ctx.W = W
        ctx.lb_one_hot = lb_one_hot
        ctx.logits = logits
        ctx.exp_wsum = exp_wsum
        ctx.reduction = reduction
        ctx.n_valid = n_valid
        log_wsoftmax = logits - torch.log(exp_wsum)
        loss = -log_wsoftmax.mul_(lb_one_hot).sum(dim=1)
        if reduction == 'mean':
            loss = loss.sum().div_(n_valid)
        if reduction == 'sum':
            loss = loss.sum()
        return loss

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_output):
        mask = ctx.mask
        W = ctx.W
        lb_one_hot = ctx.lb_one_hot
        logits = ctx.logits
        exp_wsum = ctx.exp_wsum
        reduction = ctx.reduction
        n_valid = ctx.n_valid
        wlabel = torch.sum(W * lb_one_hot, dim=1, keepdim=True)
        wscores = torch.exp(logits).div_(exp_wsum).mul_(wlabel)
        wscores[mask] = 0
        grad = wscores.sub_(lb_one_hot)
        if reduction == 'none':
            grad.mul_(grad_output.unsqueeze(1))
        elif reduction == 'sum':
            grad.mul_(grad_output)
        elif reduction == 'mean':
            grad.div_(n_valid).mul_(grad_output)
        return grad, None, None, None, None, None


class PCSoftmaxCrossEntropyV2(nn.Module):

    def __init__(self, lb_proportion, reduction='mean', ignore_index=-100):
        super(PCSoftmaxCrossEntropyV2, self).__init__()
        self.lb_proportion = lb_proportion
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, logits, label):
        return PCSoftmaxCrossEntropyFunction.apply(logits, label, self.lb_proportion, self.reduction, self.ignore_index)


class FReLU(nn.Module):

    def __init__(self, in_chan):
        super(FReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan, in_chan, 3, 1, 1, groups=in_chan)
        self.bn = nn.BatchNorm2d(in_chan)
        nn.init.xavier_normal_(self.conv.weight, gain=1.0)

    def forward(self, x):
        branch = self.bn(self.conv(x))
        out = torch.max(x, branch)
        return out


class GroupLoss(nn.Module):

    def __init__(self, in_feats=2048, n_ids=100, n_iters=2, n_lbs_per_cls=2, has_fc=True):
        super(GroupLoss, self).__init__()
        self.n_lbs_per_cls = n_lbs_per_cls
        self.n_iters = n_iters
        self.has_fc = has_fc
        self.clip = nn.ReLU(inplace=True)
        self.fc = nn.Identity()
        if has_fc:
            self.fc = nn.Linear(in_feats, n_ids)

    def forward(self, emb, lbs, logits=None):
        if self.has_fc:
            logits = self.fc(emb)
        n, c = emb.size()
        n_cls = logits.size()[1]
        device = logits.device
        emb_norm = emb - emb.mean(dim=1, keepdims=True)
        emb_norm = F.normalize(emb_norm, dim=1)
        W = torch.einsum('ab,cb->ac', emb_norm, emb_norm)
        W = W.fill_diagonal_(0)
        W = self.clip(W)
        inds_shuf = torch.randperm(n)
        n_select = n_cls * self.n_lbs_per_cls
        i_onehot = inds_shuf[:n_select]
        i_prob = inds_shuf[n_select:]
        j = lbs[i_onehot]
        X = torch.zeros_like(logits)
        probs = logits.softmax(dim=1)
        X[i_onehot, j] = 1.0
        X[i_prob] = probs[i_prob]
        for _ in range(self.n_iters):
            X = torch.einsum('ab,bc->ac', W, X)
            Xsum = X.sum(dim=1, keepdims=True) + 1e-06
            X = X / Xsum
        X = X.log()
        loss = F.nll_loss(X, lbs, reduction='mean')
        return loss


class LayerNormV1(nn.Module):
    """
    """

    def __init__(self, n_chan, affine=True, eps=1e-06):
        super(LayerNormV1, self).__init__()
        self.n_chan, self.affine = n_chan, affine
        self.weight, self.bias = None, None
        self.eps = eps
        if affine:
            self.weight = nn.Parameter(torch.ones(1, n_chan, 1))
            self.bias = nn.Parameter(torch.zeros(1, n_chan, 1))

    def forward(self, x):
        """
        input is NCHW, norm along C
        """
        N, C, H, W = x.size()
        x = x.view(N, C, -1)
        mean = x.mean(dim=1, keepdim=True)
        std = (x.var(dim=1, keepdim=True, unbiased=False) + self.eps).rsqrt()
        x = (x - mean) * std
        if self.affine:
            x = self.weight * x + self.bias
        x = x.view(N, C, H, W)
        return x


class LayerNormV2Func(torch.autograd.Function):
    """
    compute backward directly for better numeric stability
    """

    @staticmethod
    @amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x, eps):
        """
        inputs:
            x: (N, C, M)
            eps: float
        outpus:
            x: (N, C, M)
        """
        mean = x.mean(dim=1, keepdim=True)
        std = (x.var(dim=1, keepdim=True, unbiased=False) + eps).rsqrt()
        out = (x - mean).mul_(std)
        ctx.vars = x, eps
        return out

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_output):
        """
        """
        x, eps = ctx.vars
        N, C, M = x.size()
        mean = x.mean(dim=1, keepdim=True)
        var_plus_eps = x.var(dim=1, keepdim=True, unbiased=False) + eps
        grads = (x - mean).mul_(x - 1 / C).mul_(x.sum(dim=1, keepdim=True)).mul_(var_plus_eps).add_(1).mul_(var_plus_eps.rsqrt()).mul_(1.0 / C).mul_(grad_output)
        return grads, None


class LayerNormV2(nn.Module):
    """
    """

    def __init__(self, n_chan, affine=True, eps=1e-06):
        super(LayerNormV2, self).__init__()
        self.n_chan, self.affine = n_chan, affine
        self.weight, self.bias = None, None
        self.eps = eps
        if affine:
            self.weight = nn.Parameter(torch.ones(1, n_chan, 1))
            self.bias = nn.Parameter(torch.zeros(1, n_chan, 1))

    def forward(self, x):
        """
        input is NCHW, norm along C
        """
        N, C, H, W = x.size()
        x = x.view(N, C, -1)
        dt = x.dtype
        if dt == torch.float16:
            x = x.float()
        x = LayerNormV2Func.apply(x, self.eps)
        if self.affine:
            x = self.weight * x + self.bias
        x = x.view(N, C, H, W)
        return x


class LayerNormV3Func(torch.autograd.Function):
    """
    compute backward directly for better numeric stability
    """

    @staticmethod
    @amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x, eps):
        """
        inputs:
            x: (N, C, M)
            eps: float
        outpus:
            x: (N, C, M)
        """
        out = layer_norm_cpp.layer_norm_forward(x, eps)
        ctx.vars = x, eps
        return out

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_output):
        """
        """
        x, eps = ctx.vars
        grads = layer_norm_cpp.layer_norm_backward(grad_output, x, eps)
        return grads, None


class LayerNormV3(nn.Module):
    """
    """

    def __init__(self, n_chan, affine=True, eps=1e-06):
        super(LayerNormV3, self).__init__()
        self.n_chan, self.affine = n_chan, affine
        self.weight, self.bias = None, None
        self.eps = eps
        if affine:
            self.weight = nn.Parameter(torch.ones(1, n_chan, 1))
            self.bias = nn.Parameter(torch.zeros(1, n_chan, 1))

    def forward(self, x):
        """
        input is NCHW, norm along C
        """
        N, C, H, W = x.size()
        x = x.view(N, C, -1)
        dt = x.dtype
        if dt == torch.float16:
            x = x.float()
        x = LayerNormV3Func.apply(x, self.eps)
        if self.affine:
            x = self.weight * x + self.bias
        x = x.view(N, C, H, W)
        return x


class OhemCELoss(nn.Module):

    def __init__(self, score_thresh, n_min=None, ignore_index=255):
        super(OhemCELoss, self).__init__()
        self.score_thresh = score_thresh
        self.ignore_lb = ignore_index
        self.n_min = n_min
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='mean')

    def forward(self, logits, labels):
        n_min = labels.numel() // 16 if self.n_min is None else self.n_min
        labels = ohem_cpp.score_ohem_label(logits.float(), labels, self.ignore_lb, self.score_thresh, n_min).detach()
        loss = self.criteria(logits, labels)
        return loss


class OhemLargeMarginLoss(nn.Module):

    def __init__(self, score_thresh, n_min=None, ignore_index=255):
        super(OhemLargeMarginLoss, self).__init__()
        self.score_thresh = score_thresh
        self.ignore_lb = ignore_index
        self.n_min = n_min
        self.criteria = LargeMarginSoftmaxV3(ignore_index=ignore_index, reduction='mean')

    def forward(self, logits, labels):
        n_min = labels.numel() // 16 if self.n_min is None else self.n_min
        labels = ohem_cpp.score_ohem_label(logits.float(), labels, self.ignore_lb, self.score_thresh, n_min).detach()
        loss = self.criteria(logits, labels)
        return loss


class SoftDiceLossV1(nn.Module):
    """
    soft-dice loss, useful in binary segmentation
    """

    def __init__(self, p=1, smooth=1):
        super(SoftDiceLossV1, self).__init__()
        self.p = p
        self.smooth = smooth

    def forward(self, logits, labels):
        """
        inputs:
            logits: tensor of shape (N, H, W, ...)
            label: tensor of shape(N, H, W, ...)
        output:
            loss: tensor of shape(1, )
        """
        probs = torch.sigmoid(logits)
        numer = (probs * labels).sum()
        denor = (probs.pow(self.p) + labels.pow(self.p)).sum()
        loss = 1.0 - (2 * numer + self.smooth) / (denor + self.smooth)
        return loss


class SoftDiceLossV2Func(torch.autograd.Function):
    """
    compute backward directly for better numeric stability
    """

    @staticmethod
    @amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, logits, labels, p, smooth):
        """
        inputs:
            logits: (N, L)
            labels: (N, L)
        outpus:
            loss: (N,)
        """
        probs = torch.sigmoid(logits)
        numer = 2 * (probs * labels).sum(dim=1) + smooth
        denor = (probs.pow(p) + labels.pow(p)).sum(dim=1) + smooth
        loss = 1.0 - numer / denor
        ctx.vars = probs, labels, numer, denor, p, smooth
        return loss

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_output):
        """
        compute gradient of soft-dice loss
        """
        probs, labels, numer, denor, p, smooth = ctx.vars
        numer, denor = numer.view(-1, 1), denor.view(-1, 1)
        term1 = (1.0 - probs).mul_(2).mul_(labels).mul_(probs).div_(denor)
        term2 = probs.pow(p).mul_(1.0 - probs).mul_(numer).mul_(p).div_(denor.pow_(2))
        grads = term2.sub_(term1).mul_(grad_output)
        return grads, None, None, None


class SoftDiceLossV2(nn.Module):
    """
    soft-dice loss, useful in binary segmentation
    """

    def __init__(self, p=1, smooth=1):
        super(SoftDiceLossV2, self).__init__()
        self.p = p
        self.smooth = smooth

    def forward(self, logits, labels):
        """
        inputs:
            logits: tensor of shape (N, H, W, ...)
            label: tensor of shape(N, H, W, ...)
        output:
            loss: tensor of shape(1, )
        """
        logits = logits.view(1, -1)
        labels = labels.view(1, -1)
        loss = SoftDiceLossV2Func.apply(logits, labels, self.p, self.smooth)
        return loss


class SoftDiceLossV3Func(torch.autograd.Function):
    """
    compute backward directly for better numeric stability
    """

    @staticmethod
    @amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, logits, labels, p, smooth):
        """
        inputs:
            logits: (N, L)
            labels: (N, L)
        outpus:
            loss: (N,)
        """
        assert logits.size() == labels.size() and logits.dim() == 2
        loss = soft_dice_cpp.soft_dice_forward(logits, labels, p, smooth)
        ctx.vars = logits, labels, p, smooth
        return loss

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_output):
        """
        compute gradient of soft-dice loss
        """
        logits, labels, p, smooth = ctx.vars
        grads = soft_dice_cpp.soft_dice_backward(grad_output, logits, labels, p, smooth)
        return grads, None, None, None


class SoftDiceLossV3(nn.Module):
    """
    soft-dice loss, useful in binary segmentation
    """

    def __init__(self, p=1, smooth=1.0):
        super(SoftDiceLossV3, self).__init__()
        self.p = p
        self.smooth = smooth

    def forward(self, logits, labels):
        """
        inputs:
            logits: tensor of shape (N, H, W, ...)
            label: tensor of shape(N, H, W, ...)
        output:
            loss: tensor of shape(1, )
        """
        logits = logits.view(1, -1)
        labels = labels.view(1, -1)
        loss = SoftDiceLossV3Func.apply(logits, labels, self.p, self.smooth)
        return loss


class SwishV1(nn.Module):

    def __init__(self):
        super(SwishV1, self).__init__()

    def forward(self, feat):
        return feat * torch.sigmoid(feat)


class SwishFunction(torch.autograd.Function):

    @staticmethod
    @amp.custom_fwd
    def forward(ctx, feat):
        sig = torch.sigmoid(feat)
        out = feat * torch.sigmoid(feat)
        grad = sig * (1 + feat * (1 - sig))
        ctx.grad = grad
        return out

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_output):
        grad = ctx.grad
        grad *= grad_output
        return grad


class SwishV2(nn.Module):

    def __init__(self):
        super(SwishV2, self).__init__()

    def forward(self, feat):
        return SwishFunction.apply(feat)


class SwishFunctionV3(torch.autograd.Function):

    @staticmethod
    @amp.custom_fwd
    def forward(ctx, feat):
        ctx.feat = feat
        return swish_cpp.swish_forward(feat)

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_output):
        feat = ctx.feat
        return swish_cpp.swish_backward(grad_output, feat)


class SwishV3(nn.Module):

    def __init__(self):
        super(SwishV3, self).__init__()

    def forward(self, feat):
        return SwishFunctionV3.apply(feat)


def taylor_softmax_v1(x, dim=1, n=4, use_log=False):
    assert n % 2 == 0 and n > 0
    fn = torch.ones_like(x)
    denor = 1.0
    for i in range(1, n + 1):
        denor *= i
        fn = fn + x.pow(i) / denor
    out = fn / fn.sum(dim=dim, keepdims=True)
    if use_log:
        out = out.log()
    return out


class TaylorSoftmaxV1(nn.Module):
    """
    This is the autograd version
    """

    def __init__(self, dim=1, n=2):
        super(TaylorSoftmaxV1, self).__init__()
        self.dim = dim
        self.n = n

    def forward(self, x):
        """
        usage similar to nn.Softmax:
            >>> mod = TaylorSoftmaxV1(dim=1, n=4)
            >>> inten = torch.randn(1, 32, 64, 64)
            >>> out = mod(inten)
        """
        return taylor_softmax_v1(x, self.dim, self.n, use_log=False)


class TaylorSoftmaxFunc(torch.autograd.Function):
    """
    use cpp/cuda to accelerate and shrink memory usage
    """

    @staticmethod
    @amp.custom_fwd
    def forward(ctx, feat, dim=1, n=2, use_log=False):
        ctx.vars = feat, dim, n, use_log
        return taylor_softmax_cpp.taylor_softmax_forward(feat, dim, n, use_log)

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_output):
        """
        compute gradient of focal loss
        """
        feat, dim, n, use_log = ctx.vars
        return taylor_softmax_cpp.taylor_softmax_backward(grad_output, feat, dim, n, use_log), None, None, None


def taylor_softmax_v3(inten, dim=1, n=4, use_log=False):
    assert n % 2 == 0 and n > 0
    return TaylorSoftmaxFunc.apply(inten, dim, n, use_log)


class TaylorSoftmaxV3(nn.Module):
    """
    This is the autograd version
    """

    def __init__(self, dim=1, n=2):
        super(TaylorSoftmaxV3, self).__init__()
        assert n % 2 == 0 and n > 0
        self.dim = dim
        self.n = n

    def forward(self, x):
        """
        usage similar to nn.Softmax:
            >>> mod = TaylorSoftmaxV3(dim=1, n=4)
            >>> inten = torch.randn(1, 32, 64, 64)
            >>> out = mod(inten)
        """
        return taylor_softmax_v3(x, self.dim, self.n, use_log=False)


class LogTaylorSoftmaxV1(nn.Module):
    """
    This is the autograd version
    """

    def __init__(self, dim=1, n=2):
        super(LogTaylorSoftmaxV1, self).__init__()
        assert n % 2 == 0
        self.dim = dim
        self.n = n

    def forward(self, x):
        """
        usage similar to nn.Softmax:
            >>> mod = LogTaylorSoftmaxV1(dim=1, n=4)
            >>> inten = torch.randn(1, 32, 64, 64)
            >>> out = mod(inten)
        """
        return taylor_softmax_v1(x, self.dim, self.n, use_log=True)


class LogTaylorSoftmaxV3(nn.Module):
    """
    This is the autograd version
    """

    def __init__(self, dim=1, n=2):
        super(LogTaylorSoftmaxV3, self).__init__()
        assert n % 2 == 0
        self.dim = dim
        self.n = n

    def forward(self, x):
        """
        usage similar to nn.Softmax:
            >>> mod = LogTaylorSoftmaxV3(dim=1, n=4)
            >>> inten = torch.randn(1, 32, 64, 64)
            >>> out = mod(inten)
        """
        return taylor_softmax_v3(x, self.dim, self.n, use_log=True)


class TaylorCrossEntropyLossV1(nn.Module):
    """
    This is the autograd version
    """

    def __init__(self, n=2, ignore_index=-1, reduction='mean'):
        super(TaylorCrossEntropyLossV1, self).__init__()
        assert n % 2 == 0
        self.taylor_softmax = LogTaylorSoftmaxV1(dim=1, n=n)
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, logits, labels):
        """
        usage similar to nn.CrossEntropyLoss:
            >>> crit = TaylorCrossEntropyLossV1(n=4)
            >>> inten = torch.randn(1, 10, 64, 64)
            >>> label = torch.randint(0, 10, (1, 64, 64))
            >>> out = crit(inten, label)
        """
        log_probs = self.taylor_softmax(logits)
        loss = F.nll_loss(log_probs, labels, reduction=self.reduction, ignore_index=self.ignore_index)
        return loss


class TaylorCrossEntropyLossV3(nn.Module):
    """
    This is the autograd version
    """

    def __init__(self, n=2, ignore_index=-1, reduction='mean'):
        super(TaylorCrossEntropyLossV3, self).__init__()
        assert n % 2 == 0
        self.taylor_softmax = LogTaylorSoftmaxV3(dim=1, n=n)
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, logits, labels):
        """
        usage similar to nn.CrossEntropyLoss:
            >>> crit = TaylorCrossEntropyLossV3(n=4)
            >>> inten = torch.randn(1, 10, 64, 64)
            >>> label = torch.randint(0, 10, (1, 64, 64))
            >>> out = crit(inten, label)
        """
        log_probs = self.taylor_softmax(logits)
        loss = F.nll_loss(log_probs, labels, reduction=self.reduction, ignore_index=self.ignore_index)
        return loss


class Model(nn.Module):

    def __init__(self, n_classes):
        super(Model, self).__init__()
        net = torchvision.models.resnet18(pretrained=False)
        self.conv1 = net.conv1
        self.bn1 = net.bn1
        self.maxpool = net.maxpool
        self.relu = net.relu
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        self.fc = nn.Conv2d(512, n_classes, 3, 1, 1)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.bn1(feat)
        feat = self.relu(feat)
        feat = self.maxpool(feat)
        feat = self.layer1(feat)
        feat = self.layer2(feat)
        feat = self.layer3(feat)
        feat = self.layer4(feat)
        feat = self.fc(feat)
        out = torch.mean(feat, dim=(2, 3))
        return out


class TripletLoss(nn.Module):
    """
    Compute normal triplet loss or soft margin triplet loss given triplets
    """

    def __init__(self, margin=None):
        super(TripletLoss, self).__init__()
        self.margin = margin
        if self.margin is None:
            self.Loss = nn.SoftMarginLoss()
        else:
            self.Loss = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, anchor, pos, neg):
        if self.margin is None:
            num_samples = anchor.shape[0]
            y = torch.ones((num_samples, 1)).view(-1)
            if anchor.is_cuda:
                y = y
            ap_dist = torch.norm(anchor - pos, 2, dim=1).view(-1)
            an_dist = torch.norm(anchor - neg, 2, dim=1).view(-1)
            loss = self.Loss(an_dist - ap_dist, y)
        else:
            loss = self.Loss(anchor, pos, neg)
        return loss


class TaylorCrossEntropyLoss(nn.Module):
    """
    This is the autograd version
    """

    def __init__(self, n=2, ignore_index=-1, reduction='mean'):
        super(TaylorCrossEntropyLoss, self).__init__()
        assert n % 2 == 0
        self.taylor_softmax = LogTaylorSoftmaxV1(dim=1, n=n)
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, logits, labels):
        """
        usage similar to nn.CrossEntropyLoss:
            >>> crit = TaylorCrossEntropyLoss(n=4)
            >>> inten = torch.randn(1, 10, 64, 64)
            >>> label = torch.randint(0, 10, (1, 64, 64))
            >>> out = crit(inten, label)
        """
        log_probs = self.taylor_softmax(logits)
        loss = F.nll_loss(log_probs, labels, reduction=self.reduction, ignore_index=self.ignore_index)
        return loss


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AMSoftmax,
     lambda: ([], {'in_feats': 4}),
     lambda: ([torch.rand([4, 4]), torch.ones([4], dtype=torch.int64)], {}),
     True),
    (AffinityLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.ones([4, 4, 4], dtype=torch.int64)], {}),
     False),
    (CIOULoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
    (CoordConv2d,
     lambda: ([], {'in_chan': 4, 'out_chan': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DIOULoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
    (DY_Conv2d,
     lambda: ([], {'in_chan': 4, 'out_chan': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (FReLU,
     lambda: ([], {'in_chan': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FocalLossV1,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (FocalLossV2,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (HSwishV1,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (HSwishV2,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (LabelSmoothSoftmaxCEV1,
     lambda: ([], {}),
     lambda: ([torch.ones([4, 4], dtype=torch.int64), torch.ones([4], dtype=torch.int64)], {}),
     True),
    (LargeMarginSoftmaxV2,
     lambda: ([], {}),
     lambda: ([torch.ones([4, 4], dtype=torch.int64), torch.ones([4], dtype=torch.int64)], {}),
     False),
    (LayerNormV1,
     lambda: ([], {'n_chan': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LayerNormV2,
     lambda: ([], {'n_chan': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (LogTaylorSoftmaxV1,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (LovaszSoftmaxV1,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.ones([4], dtype=torch.int64)], {}),
     False),
    (MishV1,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MishV2,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Model,
     lambda: ([], {'n_classes': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (PCSoftmaxCrossEntropyV2,
     lambda: ([], {'lb_proportion': 4}),
     lambda: ([torch.ones([4, 4], dtype=torch.int64), torch.ones([4], dtype=torch.int64)], {}),
     False),
    (SoftDiceLossV1,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (SoftDiceLossV2,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (SwishV1,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SwishV2,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (TaylorSoftmaxV1,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (TripletLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])], {}),
     True),
]

class Test_CoinCheung_pytorch_loss(_paritybench_base):
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

