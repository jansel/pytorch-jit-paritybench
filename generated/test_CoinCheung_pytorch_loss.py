import sys
_module = sys.modules[__name__]
del sys
affinity_loss = _module
amsoftmax = _module
dice_loss = _module
dual_focal_loss = _module
ema = _module
focal_loss = _module
generalized_iou_loss = _module
label_smooth = _module
mish = _module
one_hot = _module
pc_softmax = _module
setup = _module
soft_dice_loss = _module
swish = _module
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


from torch.utils import cpp_extension


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
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        x_norm = torch.div(x, x_norm)
        w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        w_norm = torch.div(self.W, w_norm)
        costh = torch.mm(x_norm, w_norm)
        lb_view = lb.view(-1, 1)
        if lb_view.is_cuda:
            lb_view = lb_view.cpu()
        delt_costh = torch.zeros(costh.size()).scatter_(1, lb_view, self.m)
        if x.is_cuda:
            delt_costh = delt_costh
        costh_m = costh - delt_costh
        costh_m_s = self.s * costh_m
        loss = self.ce(costh_m_s, lb)
        return loss


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
        args:
            logits: tensor of shape (N, ...)
            label: tensor of shape(N, ...)
        """
        logits = logits.float()
        with torch.no_grad():
            alpha = torch.empty_like(logits).fill_(1 - self.alpha)
            alpha[label == 1] = self.alpha
        probs = torch.sigmoid(logits)
        pt = torch.where(label == 1, probs, 1 - probs)
        ce_loss = self.crit(logits, label.double())
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
    def forward(ctx, logits, label, alpha, gamma):
        logits = logits.float()
        coeff = torch.empty_like(logits).fill_(1 - alpha)
        coeff[label == 1] = alpha
        probs = torch.sigmoid(logits)
        log_probs = torch.where(logits >= 0, F.softplus(logits, -1, 50), logits - F.softplus(logits, 1, 50))
        log_1_probs = torch.where(logits >= 0, -logits + F.softplus(logits, -1, 50), -F.softplus(logits, 1, 50))
        probs_gamma = probs ** gamma
        probs_1_gamma = (1.0 - probs) ** gamma
        ctx.coeff = coeff
        ctx.probs = probs
        ctx.log_probs = log_probs
        ctx.log_1_probs = log_1_probs
        ctx.probs_gamma = probs_gamma
        ctx.probs_1_gamma = probs_1_gamma
        ctx.label = label
        ctx.gamma = gamma
        term1 = probs_1_gamma * log_probs
        term2 = probs_gamma * log_1_probs
        loss = torch.where(label == 1, term1, term2).mul_(coeff).neg_()
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        """
        compute gradient of focal loss
        """
        coeff = ctx.coeff
        probs = ctx.probs
        log_probs = ctx.log_probs
        log_1_probs = ctx.log_1_probs
        probs_gamma = ctx.probs_gamma
        probs_1_gamma = ctx.probs_1_gamma
        label = ctx.label
        gamma = ctx.gamma
        term1 = (1.0 - probs - gamma * probs * log_probs).mul_(probs_1_gamma).neg_()
        term2 = (probs - gamma * (1.0 - probs) * log_1_probs).mul_(probs_gamma)
        grads = torch.where(label == 1, term1, term2).mul_(coeff).mul_(grad_output)
        return grads, None, None, None


class FocalLossV2(nn.Module):
    """
    This use better formula to compute the gradient, which has better numeric stability
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
    def forward(ctx, logits, labels, alpha, gamma):
        logits = logits.float()
        loss = focal_cpp.focalloss_forward(logits, labels, gamma, alpha)
        ctx.variables = logits, labels, alpha, gamma
        return loss

    @staticmethod
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
        args: logits: tensor of shape (N, C, H, W)
        args: label: tensor of shape(N, H, W)
        """
        logits = logits.float()
        with torch.no_grad():
            num_classes = logits.size(1)
            label = label.clone().detach()
            ignore = label == self.lb_ignore
            n_valid = (ignore == 0).sum()
            label[ignore] = 0
            lb_pos, lb_neg = 1.0 - self.lb_smooth, self.lb_smooth / num_classes
            label = torch.empty_like(logits).fill_(lb_neg).scatter_(1, label.unsqueeze(1), lb_pos).detach()
        logs = self.log_softmax(logits)
        loss = -torch.sum(logs * label, dim=1)
        loss[ignore] = 0
        if self.reduction == 'mean':
            loss = loss.sum() / n_valid
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss


class LSRCrossEntropyFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, logits, label, lb_smooth, reduction, lb_ignore):
        num_classes = logits.size(1)
        label = label.clone().detach()
        ignore = label == lb_ignore
        n_valid = (ignore == 0).sum()
        label[ignore] = 0
        lb_pos, lb_neg = 1.0 - lb_smooth, lb_smooth / num_classes
        label = torch.empty_like(logits).fill_(lb_neg).scatter_(1, label.unsqueeze(1), lb_pos).detach()
        ignore = ignore.nonzero()
        _, M = ignore.size()
        a, *b = ignore.chunk(M, dim=1)
        mask = [a, torch.arange(label.size(1)), *b]
        label[mask] = 0
        coeff = (num_classes - 1) * lb_neg + lb_pos
        ctx.coeff = coeff
        ctx.mask = mask
        ctx.logits = logits
        ctx.label = label
        ctx.reduction = reduction
        ctx.n_valid = n_valid
        loss = torch.log_softmax(logits, dim=1).neg_().mul_(label).sum(dim=1)
        if reduction == 'mean':
            loss = loss.sum().div_(n_valid)
        if reduction == 'sum':
            loss = loss.sum()
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        coeff = ctx.coeff
        mask = ctx.mask
        logits = ctx.logits
        label = ctx.label
        reduction = ctx.reduction
        n_valid = ctx.n_valid
        scores = torch.softmax(logits, dim=1).mul_(coeff)
        scores[mask] = 0
        if reduction == 'none':
            grad = scores.sub_(label).mul_(grad_output.unsqueeze(1))
        elif reduction == 'sum':
            grad = scores.sub_(label).mul_(grad_output)
        elif reduction == 'mean':
            grad = scores.sub_(label).mul_(grad_output.div_(n_valid))
        return grad, None, None, None, None, None


class LabelSmoothSoftmaxCEV2(nn.Module):

    def __init__(self, lb_smooth=0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothSoftmaxCEV2, self).__init__()
        self.lb_smooth = lb_smooth
        self.reduction = reduction
        self.lb_ignore = ignore_index

    def forward(self, logits, label):
        return LSRCrossEntropyFunction.apply(logits, label, self.lb_smooth, self.reduction, self.lb_ignore)


class MishV1(nn.Module):

    def __init__(self):
        super(MishV1, self).__init__()

    def forward(self, feat):
        return feat * torch.tanh(F.softplus(feat))


class MishFunctionV2(torch.autograd.Function):

    @staticmethod
    def forward(ctx, feat):
        tanhX = torch.tanh(F.softplus(feat))
        out = feat * tanhX
        grad = tanhX + feat * (1 - torch.pow(tanhX, 2)) * torch.sigmoid(feat)
        ctx.grad = grad
        return out

    @staticmethod
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
    def forward(ctx, feat):
        ctx.feat = feat
        return mish_cpp.mish_forward(feat)

    @staticmethod
    def backward(ctx, grad_output):
        feat = ctx.feat
        return mish_cpp.mish_backward(grad_output, feat)


class MishV3(nn.Module):

    def __init__(self):
        super(MishV3, self).__init__()

    def forward(self, feat):
        return MishFunctionV3.apply(feat)


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


class SoftDiceLossV1(nn.Module):
    """
    soft-dice loss, useful in binary segmentation
    """

    def __init__(self, p=1, smooth=1, reduction='mean'):
        super(SoftDiceLossV1, self).__init__()
        self.p = p
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, logits, labels):
        """
        args: logits: tensor of shape (N, H, W)
        args: label: tensor of shape(N, H, W)
        """
        probs = torch.sigmoid(logits)
        numer = (probs * labels).sum(dim=(1, 2))
        denor = (probs.pow(self.p) + labels).sum(dim=(1, 2))
        loss = 1.0 - (2 * numer + self.smooth) / (denor + self.smooth)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


class SoftDiceLossV2Func(torch.autograd.Function):
    """
    compute backward directly for better numeric stability
    """

    @staticmethod
    def forward(ctx, logits, labels, p, smooth):
        logits = logits.float()
        probs = torch.sigmoid(logits)
        numer = 2 * (probs * labels).sum(dim=(1, 2)) + smooth
        denor = (probs.pow(p) + labels).sum(dim=(1, 2)) + smooth
        loss = 1.0 - numer / denor
        ctx.vars = probs, labels, numer, denor, p, smooth
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        """
        compute gradient of soft-dice loss
        """
        probs, labels, numer, denor, p, smooth = ctx.vars
        M = numer.view(-1, 1, 1) - (probs * labels).mul_(2)
        N = denor.view(-1, 1, 1) - probs.pow(p)
        mppi_1 = probs.pow(p - 1).mul_(p).mul_(M)
        grads = torch.where(labels == 1, probs.pow(p).mul_(2 * (1.0 - p)) - mppi_1 + N.mul_(2), -mppi_1)
        grads = grads.div_((probs.pow(p) + N).pow(2)).mul_(probs).mul_(1.0 - probs)
        grads = grads.mul_(grad_output.view(-1, 1, 1)).neg_()
        return grads, None, None, None


class SoftDiceLossV2(nn.Module):
    """
    soft-dice loss, useful in binary segmentation
    """

    def __init__(self, p=1, smooth=1, reduction='mean'):
        super(SoftDiceLossV2, self).__init__()
        self.p = p
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, logits, labels):
        """
        args: logits: tensor of shape (N, H, W)
        args: label: tensor of shape(N, H, W)
        """
        loss = SoftDiceLossV2Func.apply(logits, labels, self.p, self.smooth)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


class SoftDiceLossV3Func(torch.autograd.Function):
    """
    compute backward directly for better numeric stability
    """

    @staticmethod
    def forward(ctx, logits, labels, p, smooth):
        logits = logits.float()
        loss = soft_dice_cpp.soft_dice_forward(logits, labels, p, smooth)
        ctx.vars = logits, labels, p, smooth
        return loss

    @staticmethod
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

    def __init__(self, p=1, smooth=1.0, reduction='mean'):
        super(SoftDiceLossV3, self).__init__()
        self.p = p
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, logits, labels):
        """
        args: logits: tensor of shape (N, H, W)
        args: label: tensor of shape(N, H, W)
        """
        loss = SoftDiceLossV3Func.apply(logits, labels, self.p, self.smooth)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


class SwishV1(nn.Module):

    def __init__(self):
        super(SwishV1, self).__init__()

    def forward(self, feat):
        return feat * torch.sigmoid(feat)


class SwishFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, feat):
        sig = torch.sigmoid(feat)
        out = feat * torch.sigmoid(feat)
        grad = sig * (1 + feat * (1 - sig))
        ctx.grad = grad
        return out

    @staticmethod
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
    def forward(ctx, feat):
        ctx.feat = feat
        return swish_cpp.swish_forward(feat)

    @staticmethod
    def backward(ctx, grad_output):
        feat = ctx.feat
        return swish_cpp.swish_backward(grad_output, feat)


class SwishV3(nn.Module):

    def __init__(self):
        super(SwishV3, self).__init__()

    def forward(self, feat):
        return SwishFunctionV3.apply(feat)


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


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BatchSoftDiceLoss,
     lambda: ([], {}),
     lambda: ([torch.ones([4, 4, 4, 4], dtype=torch.int64), torch.ones([4, 4, 4], dtype=torch.int64)], {}),
     False),
    (FocalLossV1,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (FocalLossV2,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (GeneralizedSoftDiceLoss,
     lambda: ([], {}),
     lambda: ([torch.ones([4, 4, 4, 4], dtype=torch.int64), torch.ones([4, 4, 4], dtype=torch.int64)], {}),
     False),
    (LabelSmoothSoftmaxCEV1,
     lambda: ([], {}),
     lambda: ([torch.ones([4, 4], dtype=torch.int64), torch.ones([4], dtype=torch.int64)], {}),
     False),
    (MishV1,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MishV2,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
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

