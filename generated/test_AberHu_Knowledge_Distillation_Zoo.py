import sys
_module = sys.modules[__name__]
del sys
dataset = _module
kd_losses = _module
ab = _module
afd = _module
at = _module
bss = _module
cc = _module
crd = _module
dml = _module
fitnet = _module
fsp = _module
ft = _module
irg = _module
logits = _module
lwm = _module
nst = _module
ofd = _module
pkt = _module
rkd = _module
sobolev = _module
sp = _module
st = _module
vid = _module
network = _module
train_base = _module
train_bss = _module
train_crd = _module
train_dml = _module
train_ft = _module
train_kd = _module
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


import torch


import torch.nn as nn


import torch.nn.functional as F


import math


from torch.autograd import grad


import numpy as np


import time


import logging


import torch.backends.cudnn as cudnn


import torchvision.transforms as transforms


import torchvision.datasets as dst


from itertools import chain


class AB(nn.Module):
    """
	Knowledge Transfer via Distillation of Activation Boundaries Formed by Hidden Neurons
	https://arxiv.org/pdf/1811.03233.pdf
	"""

    def __init__(self, margin):
        super(AB, self).__init__()
        self.margin = margin

    def forward(self, fm_s, fm_t):
        loss = (fm_s + self.margin).pow(2) * ((fm_s > -self.margin) & (fm_t <= 0)).float() + (fm_s - self.margin).pow(2) * ((fm_s <= self.margin) & (fm_t > 0)).float()
        loss = loss.mean()
        return loss


class AFD(nn.Module):
    """
	Pay Attention to Features, Transfer Learn Faster CNNs
	https://openreview.net/pdf?id=ryxyCeHtPB
	"""

    def __init__(self, in_channels, att_f):
        super(AFD, self).__init__()
        mid_channels = int(in_channels * att_f)
        self.attention = nn.Sequential(*[nn.Conv2d(in_channels, mid_channels, 1, 1, 0, bias=True), nn.ReLU(inplace=True), nn.Conv2d(mid_channels, in_channels, 1, 1, 0, bias=True)])
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, fm_s, fm_t, eps=1e-06):
        fm_t_pooled = F.adaptive_avg_pool2d(fm_t, 1)
        rho = self.attention(fm_t_pooled)
        rho = torch.sigmoid(rho.squeeze())
        rho = rho / torch.sum(rho, dim=1, keepdim=True)
        fm_s_norm = torch.norm(fm_s, dim=(2, 3), keepdim=True)
        fm_s = torch.div(fm_s, fm_s_norm + eps)
        fm_t_norm = torch.norm(fm_t, dim=(2, 3), keepdim=True)
        fm_t = torch.div(fm_t, fm_t_norm + eps)
        loss = rho * torch.pow(fm_s - fm_t, 2).mean(dim=(2, 3))
        loss = loss.sum(1).mean(0)
        return loss


class AT(nn.Module):
    """
	Paying More Attention to Attention: Improving the Performance of Convolutional
	Neural Netkworks wia Attention Transfer
	https://arxiv.org/pdf/1612.03928.pdf
	"""

    def __init__(self, p):
        super(AT, self).__init__()
        self.p = p

    def forward(self, fm_s, fm_t):
        loss = F.mse_loss(self.attention_map(fm_s), self.attention_map(fm_t))
        return loss

    def attention_map(self, fm, eps=1e-06):
        am = torch.pow(torch.abs(fm), self.p)
        am = torch.sum(am, dim=1, keepdim=True)
        norm = torch.norm(am, dim=(2, 3), keepdim=True)
        am = torch.div(am, norm + eps)
        return am


class BSS(nn.Module):
    """
	Knowledge Distillation with Adversarial Samples Supporting Decision Boundary
	https://arxiv.org/pdf/1805.05532.pdf
	"""

    def __init__(self, T):
        super(BSS, self).__init__()
        self.T = T

    def forward(self, attacked_out_s, attacked_out_t):
        loss = F.kl_div(F.log_softmax(attacked_out_s / self.T, dim=1), F.softmax(attacked_out_t / self.T, dim=1), reduction='batchmean')
        return loss


class CC(nn.Module):
    """
	Correlation Congruence for Knowledge Distillation
	http://openaccess.thecvf.com/content_ICCV_2019/papers/
	Peng_Correlation_Congruence_for_Knowledge_Distillation_ICCV_2019_paper.pdf
	"""

    def __init__(self, gamma, P_order):
        super(CC, self).__init__()
        self.gamma = gamma
        self.P_order = P_order

    def forward(self, feat_s, feat_t):
        corr_mat_s = self.get_correlation_matrix(feat_s)
        corr_mat_t = self.get_correlation_matrix(feat_t)
        loss = F.mse_loss(corr_mat_s, corr_mat_t)
        return loss

    def get_correlation_matrix(self, feat):
        feat = F.normalize(feat, p=2, dim=-1)
        sim_mat = torch.matmul(feat, feat.t())
        corr_mat = torch.zeros_like(sim_mat)
        for p in range(self.P_order + 1):
            corr_mat += math.exp(-2 * self.gamma) * (2 * self.gamma) ** p / math.factorial(p) * torch.pow(sim_mat, p)
        return corr_mat


class ContrastLoss(nn.Module):
    """
	contrastive loss, corresponding to Eq.(18)
	"""

    def __init__(self, n_data, eps=1e-07):
        super(ContrastLoss, self).__init__()
        self.n_data = n_data
        self.eps = eps

    def forward(self, x):
        bs = x.size(0)
        N = x.size(1) - 1
        M = float(self.n_data)
        pos_pair = x.select(1, 0)
        log_pos = torch.div(pos_pair, pos_pair.add(N / M + self.eps)).log_()
        neg_pair = x.narrow(1, 1, N)
        log_neg = torch.div(neg_pair.clone().fill_(N / M), neg_pair.add(N / M + self.eps)).log_()
        loss = -(log_pos.sum() + log_neg.sum()) / bs
        return loss


class ContrastMemory(nn.Module):

    def __init__(self, feat_dim, n_data, nce_n, nce_t, nce_mom):
        super(ContrastMemory, self).__init__()
        self.N = nce_n
        self.T = nce_t
        self.momentum = nce_mom
        self.Z_t = None
        self.Z_s = None
        stdv = 1.0 / math.sqrt(feat_dim / 3.0)
        self.register_buffer('memory_t', torch.rand(n_data, feat_dim).mul_(2 * stdv).add_(-stdv))
        self.register_buffer('memory_s', torch.rand(n_data, feat_dim).mul_(2 * stdv).add_(-stdv))

    def forward(self, feat_s, feat_t, idx, sample_idx):
        bs = feat_s.size(0)
        feat_dim = self.memory_s.size(1)
        n_data = self.memory_s.size(0)
        weight_s = torch.index_select(self.memory_s, 0, sample_idx.view(-1)).detach()
        weight_s = weight_s.view(bs, self.N + 1, feat_dim)
        out_t = torch.bmm(weight_s, feat_t.view(bs, feat_dim, 1))
        out_t = torch.exp(torch.div(out_t, self.T)).squeeze().contiguous()
        weight_t = torch.index_select(self.memory_t, 0, sample_idx.view(-1)).detach()
        weight_t = weight_t.view(bs, self.N + 1, feat_dim)
        out_s = torch.bmm(weight_t, feat_s.view(bs, feat_dim, 1))
        out_s = torch.exp(torch.div(out_s, self.T)).squeeze().contiguous()
        if self.Z_t is None:
            self.Z_t = (out_t.mean() * n_data).detach().item()
        if self.Z_s is None:
            self.Z_s = (out_s.mean() * n_data).detach().item()
        out_t = torch.div(out_t, self.Z_t)
        out_s = torch.div(out_s, self.Z_s)
        with torch.no_grad():
            pos_mem_t = torch.index_select(self.memory_t, 0, idx.view(-1))
            pos_mem_t.mul_(self.momentum)
            pos_mem_t.add_(torch.mul(feat_t, 1 - self.momentum))
            pos_mem_t = F.normalize(pos_mem_t, p=2, dim=1)
            self.memory_t.index_copy_(0, idx, pos_mem_t)
            pos_mem_s = torch.index_select(self.memory_s, 0, idx.view(-1))
            pos_mem_s.mul_(self.momentum)
            pos_mem_s.add_(torch.mul(feat_s, 1 - self.momentum))
            pos_mem_s = F.normalize(pos_mem_s, p=2, dim=1)
            self.memory_s.index_copy_(0, idx, pos_mem_s)
        return out_s, out_t


class Embed(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(Embed, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = F.normalize(x, p=2, dim=1)
        return x


class CRD(nn.Module):
    """
	Contrastive Representation Distillation
	https://openreview.net/pdf?id=SkgpBJrtvS

	includes two symmetric parts:
	(a) using teacher as anchor, choose positive and negatives over the student side
	(b) using student as anchor, choose positive and negatives over the teacher side

	Args:
		s_dim: the dimension of student's feature
		t_dim: the dimension of teacher's feature
		feat_dim: the dimension of the projection space
		nce_n: number of negatives paired with each positive
		nce_t: the temperature
		nce_mom: the momentum for updating the memory buffer
		n_data: the number of samples in the training set, which is the M in Eq.(19)
	"""

    def __init__(self, s_dim, t_dim, feat_dim, nce_n, nce_t, nce_mom, n_data):
        super(CRD, self).__init__()
        self.embed_s = Embed(s_dim, feat_dim)
        self.embed_t = Embed(t_dim, feat_dim)
        self.contrast = ContrastMemory(feat_dim, n_data, nce_n, nce_t, nce_mom)
        self.criterion_s = ContrastLoss(n_data)
        self.criterion_t = ContrastLoss(n_data)

    def forward(self, feat_s, feat_t, idx, sample_idx):
        feat_s = self.embed_s(feat_s)
        feat_t = self.embed_t(feat_t)
        out_s, out_t = self.contrast(feat_s, feat_t, idx, sample_idx)
        loss_s = self.criterion_s(out_s)
        loss_t = self.criterion_t(out_t)
        loss = loss_s + loss_t
        return loss


class DML(nn.Module):
    """
	Deep Mutual Learning
	https://zpascal.net/cvpr2018/Zhang_Deep_Mutual_Learning_CVPR_2018_paper.pdf
	"""

    def __init__(self):
        super(DML, self).__init__()

    def forward(self, out1, out2):
        loss = F.kl_div(F.log_softmax(out1, dim=1), F.softmax(out2, dim=1), reduction='batchmean')
        return loss


class Hint(nn.Module):
    """
	FitNets: Hints for Thin Deep Nets
	https://arxiv.org/pdf/1412.6550.pdf
	"""

    def __init__(self):
        super(Hint, self).__init__()

    def forward(self, fm_s, fm_t):
        loss = F.mse_loss(fm_s, fm_t)
        return loss


class FSP(nn.Module):
    """
	A Gift from Knowledge Distillation: Fast Optimization, Network Minimization and Transfer Learning
	http://openaccess.thecvf.com/content_cvpr_2017/papers/Yim_A_Gift_From_CVPR_2017_paper.pdf
	"""

    def __init__(self):
        super(FSP, self).__init__()

    def forward(self, fm_s1, fm_s2, fm_t1, fm_t2):
        loss = F.mse_loss(self.fsp_matrix(fm_s1, fm_s2), self.fsp_matrix(fm_t1, fm_t2))
        return loss

    def fsp_matrix(self, fm1, fm2):
        if fm1.size(2) > fm2.size(2):
            fm1 = F.adaptive_avg_pool2d(fm1, (fm2.size(2), fm2.size(3)))
        fm1 = fm1.view(fm1.size(0), fm1.size(1), -1)
        fm2 = fm2.view(fm2.size(0), fm2.size(1), -1).transpose(1, 2)
        fsp = torch.bmm(fm1, fm2) / fm1.size(2)
        return fsp


class FT(nn.Module):
    """
	araphrasing Complex Network: Network Compression via Factor Transfer
	http://papers.nips.cc/paper/7541-paraphrasing-complex-network-network-compression-via-factor-transfer.pdf
	"""

    def __init__(self):
        super(FT, self).__init__()

    def forward(self, factor_s, factor_t):
        loss = F.l1_loss(self.normalize(factor_s), self.normalize(factor_t))
        return loss

    def normalize(self, factor):
        norm_factor = F.normalize(factor.view(factor.size(0), -1))
        return norm_factor


class IRG(nn.Module):
    """
	Knowledge Distillation via Instance Relationship Graph
	http://openaccess.thecvf.com/content_CVPR_2019/papers/
	Liu_Knowledge_Distillation_via_Instance_Relationship_Graph_CVPR_2019_paper.pdf

	The official code is written by Caffe
	https://github.com/yufanLIU/IRG
	"""

    def __init__(self, w_irg_vert, w_irg_edge, w_irg_tran):
        super(IRG, self).__init__()
        self.w_irg_vert = w_irg_vert
        self.w_irg_edge = w_irg_edge
        self.w_irg_tran = w_irg_tran

    def forward(self, irg_s, irg_t):
        fm_s1, fm_s2, feat_s, out_s = irg_s
        fm_t1, fm_t2, feat_t, out_t = irg_t
        loss_irg_vert = F.mse_loss(out_s, out_t)
        irg_edge_feat_s = self.euclidean_dist_feat(feat_s, squared=True)
        irg_edge_feat_t = self.euclidean_dist_feat(feat_t, squared=True)
        irg_edge_fm_s1 = self.euclidean_dist_fm(fm_s1, squared=True)
        irg_edge_fm_t1 = self.euclidean_dist_fm(fm_t1, squared=True)
        irg_edge_fm_s2 = self.euclidean_dist_fm(fm_s2, squared=True)
        irg_edge_fm_t2 = self.euclidean_dist_fm(fm_t2, squared=True)
        loss_irg_edge = (F.mse_loss(irg_edge_feat_s, irg_edge_feat_t) + F.mse_loss(irg_edge_fm_s1, irg_edge_fm_t1) + F.mse_loss(irg_edge_fm_s2, irg_edge_fm_t2)) / 3.0
        irg_tran_s = self.euclidean_dist_fms(fm_s1, fm_s2, squared=True)
        irg_tran_t = self.euclidean_dist_fms(fm_t1, fm_t2, squared=True)
        loss_irg_tran = F.mse_loss(irg_tran_s, irg_tran_t)
        loss = self.w_irg_vert * loss_irg_vert + self.w_irg_edge * loss_irg_edge + self.w_irg_tran * loss_irg_tran
        return loss

    def euclidean_dist_fms(self, fm1, fm2, squared=False, eps=1e-12):
        """
		Calculating the IRG Transformation, where fm1 precedes fm2 in the network.
		"""
        if fm1.size(2) > fm2.size(2):
            fm1 = F.adaptive_avg_pool2d(fm1, (fm2.size(2), fm2.size(3)))
        if fm1.size(1) < fm2.size(1):
            fm2 = (fm2[:, 0::2, :, :] + fm2[:, 1::2, :, :]) / 2.0
        fm1 = fm1.view(fm1.size(0), -1)
        fm2 = fm2.view(fm2.size(0), -1)
        fms_dist = torch.sum(torch.pow(fm1 - fm2, 2), dim=-1).clamp(min=eps)
        if not squared:
            fms_dist = fms_dist.sqrt()
        fms_dist = fms_dist / fms_dist.max()
        return fms_dist

    def euclidean_dist_fm(self, fm, squared=False, eps=1e-12):
        """
		Calculating the IRG edge of feature map. 
		"""
        fm = fm.view(fm.size(0), -1)
        fm_square = fm.pow(2).sum(dim=1)
        fm_prod = torch.mm(fm, fm.t())
        fm_dist = (fm_square.unsqueeze(0) + fm_square.unsqueeze(1) - 2 * fm_prod).clamp(min=eps)
        if not squared:
            fm_dist = fm_dist.sqrt()
        fm_dist = fm_dist.clone()
        fm_dist[range(len(fm)), range(len(fm))] = 0
        fm_dist = fm_dist / fm_dist.max()
        return fm_dist

    def euclidean_dist_feat(self, feat, squared=False, eps=1e-12):
        """
		Calculating the IRG edge of feat.
		"""
        feat_square = feat.pow(2).sum(dim=1)
        feat_prod = torch.mm(feat, feat.t())
        feat_dist = (feat_square.unsqueeze(0) + feat_square.unsqueeze(1) - 2 * feat_prod).clamp(min=eps)
        if not squared:
            feat_dist = feat_dist.sqrt()
        feat_dist = feat_dist.clone()
        feat_dist[range(len(feat)), range(len(feat))] = 0
        feat_dist = feat_dist / feat_dist.max()
        return feat_dist


class Logits(nn.Module):
    """
	Do Deep Nets Really Need to be Deep?
	http://papers.nips.cc/paper/5484-do-deep-nets-really-need-to-be-deep.pdf
	"""

    def __init__(self):
        super(Logits, self).__init__()

    def forward(self, out_s, out_t):
        loss = F.mse_loss(out_s, out_t)
        return loss


class LwM(nn.Module):
    """
	Learning without Memorizing
	https://arxiv.org/pdf/1811.08051.pdf
	"""

    def __init__(self):
        super(LwM, self).__init__()

    def forward(self, out_s, fm_s, out_t, fm_t, target):
        target_out_t = torch.gather(out_t, 1, target.view(-1, 1))
        grad_fm_t = grad(outputs=target_out_t, inputs=fm_t, grad_outputs=torch.ones_like(target_out_t), create_graph=True, retain_graph=True, only_inputs=True)[0]
        weights_t = F.adaptive_avg_pool2d(grad_fm_t, 1)
        cam_t = torch.sum(torch.mul(weights_t, grad_fm_t), dim=1, keepdim=True)
        cam_t = F.relu(cam_t)
        cam_t = cam_t.view(cam_t.size(0), -1)
        norm_cam_t = F.normalize(cam_t, p=2, dim=1)
        target_out_s = torch.gather(out_s, 1, target.view(-1, 1))
        grad_fm_s = grad(outputs=target_out_s, inputs=fm_s, grad_outputs=torch.ones_like(target_out_s), create_graph=True, retain_graph=True, only_inputs=True)[0]
        weights_s = F.adaptive_avg_pool2d(grad_fm_s, 1)
        cam_s = torch.sum(torch.mul(weights_s, grad_fm_s), dim=1, keepdim=True)
        cam_s = F.relu(cam_s)
        cam_s = cam_s.view(cam_s.size(0), -1)
        norm_cam_s = F.normalize(cam_s, p=2, dim=1)
        loss = F.l1_loss(norm_cam_s, norm_cam_t.detach())
        return loss


class NST(nn.Module):
    """
	Like What You Like: Knowledge Distill via Neuron Selectivity Transfer
	https://arxiv.org/pdf/1707.01219.pdf
	"""

    def __init__(self):
        super(NST, self).__init__()

    def forward(self, fm_s, fm_t):
        fm_s = fm_s.view(fm_s.size(0), fm_s.size(1), -1)
        fm_s = F.normalize(fm_s, dim=2)
        fm_t = fm_t.view(fm_t.size(0), fm_t.size(1), -1)
        fm_t = F.normalize(fm_t, dim=2)
        loss = self.poly_kernel(fm_t, fm_t).mean() + self.poly_kernel(fm_s, fm_s).mean() - 2 * self.poly_kernel(fm_s, fm_t).mean()
        return loss

    def poly_kernel(self, fm1, fm2):
        fm1 = fm1.unsqueeze(1)
        fm2 = fm2.unsqueeze(2)
        out = (fm1 * fm2).sum(-1).pow(2)
        return out


class OFD(nn.Module):
    """
	A Comprehensive Overhaul of Feature Distillation
	http://openaccess.thecvf.com/content_ICCV_2019/papers/
	Heo_A_Comprehensive_Overhaul_of_Feature_Distillation_ICCV_2019_paper.pdf
	"""

    def __init__(self, in_channels, out_channels):
        super(OFD, self).__init__()
        self.connector = nn.Sequential(*[nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(out_channels)])
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, fm_s, fm_t):
        margin = self.get_margin(fm_t)
        fm_t = torch.max(fm_t, margin)
        fm_s = self.connector(fm_s)
        mask = 1.0 - ((fm_s <= fm_t) & (fm_t <= 0.0)).float()
        loss = torch.mean((fm_s - fm_t) ** 2 * mask)
        return loss

    def get_margin(self, fm, eps=1e-06):
        mask = (fm < 0.0).float()
        masked_fm = fm * mask
        margin = masked_fm.sum(dim=(0, 2, 3), keepdim=True) / (mask.sum(dim=(0, 2, 3), keepdim=True) + eps)
        return margin


class PKTCosSim(nn.Module):
    """
	Learning Deep Representations with Probabilistic Knowledge Transfer
	http://openaccess.thecvf.com/content_ECCV_2018/papers/Nikolaos_Passalis_Learning_Deep_Representations_ECCV_2018_paper.pdf
	"""

    def __init__(self):
        super(PKTCosSim, self).__init__()

    def forward(self, feat_s, feat_t, eps=1e-06):
        feat_s_norm = torch.sqrt(torch.sum(feat_s ** 2, dim=1, keepdim=True))
        feat_s = feat_s / (feat_s_norm + eps)
        feat_s[feat_s != feat_s] = 0
        feat_t_norm = torch.sqrt(torch.sum(feat_t ** 2, dim=1, keepdim=True))
        feat_t = feat_t / (feat_t_norm + eps)
        feat_t[feat_t != feat_t] = 0
        feat_s_cos_sim = torch.mm(feat_s, feat_s.transpose(0, 1))
        feat_t_cos_sim = torch.mm(feat_t, feat_t.transpose(0, 1))
        feat_s_cos_sim = (feat_s_cos_sim + 1.0) / 2.0
        feat_t_cos_sim = (feat_t_cos_sim + 1.0) / 2.0
        feat_s_cond_prob = feat_s_cos_sim / torch.sum(feat_s_cos_sim, dim=1, keepdim=True)
        feat_t_cond_prob = feat_t_cos_sim / torch.sum(feat_t_cos_sim, dim=1, keepdim=True)
        loss = torch.mean(feat_t_cond_prob * torch.log((feat_t_cond_prob + eps) / (feat_s_cond_prob + eps)))
        return loss


class RKD(nn.Module):
    """
	Relational Knowledge Distillation
	https://arxiv.org/pdf/1904.05068.pdf
	"""

    def __init__(self, w_dist, w_angle):
        super(RKD, self).__init__()
        self.w_dist = w_dist
        self.w_angle = w_angle

    def forward(self, feat_s, feat_t):
        loss = self.w_dist * self.rkd_dist(feat_s, feat_t) + self.w_angle * self.rkd_angle(feat_s, feat_t)
        return loss

    def rkd_dist(self, feat_s, feat_t):
        feat_t_dist = self.pdist(feat_t, squared=False)
        mean_feat_t_dist = feat_t_dist[feat_t_dist > 0].mean()
        feat_t_dist = feat_t_dist / mean_feat_t_dist
        feat_s_dist = self.pdist(feat_s, squared=False)
        mean_feat_s_dist = feat_s_dist[feat_s_dist > 0].mean()
        feat_s_dist = feat_s_dist / mean_feat_s_dist
        loss = F.smooth_l1_loss(feat_s_dist, feat_t_dist)
        return loss

    def rkd_angle(self, feat_s, feat_t):
        feat_t_vd = feat_t.unsqueeze(0) - feat_t.unsqueeze(1)
        norm_feat_t_vd = F.normalize(feat_t_vd, p=2, dim=2)
        feat_t_angle = torch.bmm(norm_feat_t_vd, norm_feat_t_vd.transpose(1, 2)).view(-1)
        feat_s_vd = feat_s.unsqueeze(0) - feat_s.unsqueeze(1)
        norm_feat_s_vd = F.normalize(feat_s_vd, p=2, dim=2)
        feat_s_angle = torch.bmm(norm_feat_s_vd, norm_feat_s_vd.transpose(1, 2)).view(-1)
        loss = F.smooth_l1_loss(feat_s_angle, feat_t_angle)
        return loss

    def pdist(self, feat, squared=False, eps=1e-12):
        feat_square = feat.pow(2).sum(dim=1)
        feat_prod = torch.mm(feat, feat.t())
        feat_dist = (feat_square.unsqueeze(0) + feat_square.unsqueeze(1) - 2 * feat_prod).clamp(min=eps)
        if not squared:
            feat_dist = feat_dist.sqrt()
        feat_dist = feat_dist.clone()
        feat_dist[range(len(feat)), range(len(feat))] = 0
        return feat_dist


class Sobolev(nn.Module):
    """
	Sobolev Training for Neural Networks
	https://arxiv.org/pdf/1706.04859.pdf

	Knowledge Transfer with Jacobian Matching
	http://de.arxiv.org/pdf/1803.00443
	"""

    def __init__(self):
        super(Sobolev, self).__init__()

    def forward(self, out_s, out_t, img, target):
        target_out_s = torch.gather(out_s, 1, target.view(-1, 1))
        grad_s = grad(outputs=target_out_s, inputs=img, grad_outputs=torch.ones_like(target_out_s), create_graph=True, retain_graph=True, only_inputs=True)[0]
        norm_grad_s = F.normalize(grad_s.view(grad_s.size(0), -1), p=2, dim=1)
        target_out_t = torch.gather(out_t, 1, target.view(-1, 1))
        grad_t = grad(outputs=target_out_t, inputs=img, grad_outputs=torch.ones_like(target_out_t), create_graph=True, retain_graph=True, only_inputs=True)[0]
        norm_grad_t = F.normalize(grad_t.view(grad_t.size(0), -1), p=2, dim=1)
        loss = F.mse_loss(norm_grad_s, norm_grad_t.detach())
        return loss


class SP(nn.Module):
    """
	Similarity-Preserving Knowledge Distillation
	https://arxiv.org/pdf/1907.09682.pdf
	"""

    def __init__(self):
        super(SP, self).__init__()

    def forward(self, fm_s, fm_t):
        fm_s = fm_s.view(fm_s.size(0), -1)
        G_s = torch.mm(fm_s, fm_s.t())
        norm_G_s = F.normalize(G_s, p=2, dim=1)
        fm_t = fm_t.view(fm_t.size(0), -1)
        G_t = torch.mm(fm_t, fm_t.t())
        norm_G_t = F.normalize(G_t, p=2, dim=1)
        loss = F.mse_loss(norm_G_s, norm_G_t)
        return loss


class SoftTarget(nn.Module):
    """
	Distilling the Knowledge in a Neural Network
	https://arxiv.org/pdf/1503.02531.pdf
	"""

    def __init__(self, T):
        super(SoftTarget, self).__init__()
        self.T = T

    def forward(self, out_s, out_t):
        loss = F.kl_div(F.log_softmax(out_s / self.T, dim=1), F.softmax(out_t / self.T, dim=1), reduction='batchmean') * self.T * self.T
        return loss


def conv1x1(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)


class VID(nn.Module):
    """
	Variational Information Distillation for Knowledge Transfer
	https://zpascal.net/cvpr2019/Ahn_Variational_Information_Distillation_for_Knowledge_Transfer_CVPR_2019_paper.pdf
	"""

    def __init__(self, in_channels, mid_channels, out_channels, init_var, eps=1e-06):
        super(VID, self).__init__()
        self.eps = eps
        self.regressor = nn.Sequential(*[conv1x1(in_channels, mid_channels), nn.ReLU(), conv1x1(mid_channels, mid_channels), nn.ReLU(), conv1x1(mid_channels, out_channels)])
        self.alpha = nn.Parameter(np.log(np.exp(init_var - eps) - 1.0) * torch.ones(out_channels))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, fm_s, fm_t):
        pred_mean = self.regressor(fm_s)
        pred_var = torch.log(1.0 + torch.exp(self.alpha)) + self.eps
        pred_var = pred_var.view(1, -1, 1, 1)
        neg_log_prob = 0.5 * (torch.log(pred_var) + (pred_mean - fm_t) ** 2 / pred_var)
        loss = torch.mean(neg_log_prob)
        return loss


class resblock(nn.Module):

    def __init__(self, in_channels, out_channels, return_before_act):
        super(resblock, self).__init__()
        self.return_before_act = return_before_act
        self.downsample = in_channels != out_channels
        if self.downsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
            self.ds = nn.Sequential(*[nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False), nn.BatchNorm2d(out_channels)])
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.ds = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        pout = self.conv1(x)
        pout = self.bn1(pout)
        pout = self.relu(pout)
        pout = self.conv2(pout)
        pout = self.bn2(pout)
        if self.downsample:
            residual = self.ds(x)
        pout += residual
        out = self.relu(pout)
        if not self.return_before_act:
            return out
        else:
            return pout, out


class resnet20(nn.Module):

    def __init__(self, num_class):
        super(resnet20, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.res1 = self.make_layer(resblock, 3, 16, 16)
        self.res2 = self.make_layer(resblock, 3, 16, 32)
        self.res3 = self.make_layer(resblock, 3, 32, 64)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_class)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.num_class = num_class

    def make_layer(self, block, num, in_channels, out_channels):
        layers = [block(in_channels, out_channels, False)]
        for i in range(num - 2):
            layers.append(block(out_channels, out_channels, False))
        layers.append(block(out_channels, out_channels, True))
        return nn.Sequential(*layers)

    def forward(self, x):
        pstem = self.conv1(x)
        pstem = self.bn1(pstem)
        stem = self.relu(pstem)
        stem = pstem, stem
        rb1 = self.res1(stem[1])
        rb2 = self.res2(rb1[1])
        rb3 = self.res3(rb2[1])
        feat = self.avgpool(rb3[1])
        feat = feat.view(feat.size(0), -1)
        out = self.fc(feat)
        return stem, rb1, rb2, rb3, feat, out

    def get_channel_num(self):
        return [16, 16, 32, 64, 64, self.num_class]

    def get_chw_num(self):
        return [(16, 32, 32), (16, 32, 32), (32, 16, 16), (64, 8, 8), (64,), (self.num_class,)]


class resnet110(nn.Module):

    def __init__(self, num_class):
        super(resnet110, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.res1 = self.make_layer(resblock, 18, 16, 16)
        self.res2 = self.make_layer(resblock, 18, 16, 32)
        self.res3 = self.make_layer(resblock, 18, 32, 64)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_class)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.num_class = num_class

    def make_layer(self, block, num, in_channels, out_channels):
        layers = [block(in_channels, out_channels, False)]
        for i in range(num - 2):
            layers.append(block(out_channels, out_channels, False))
        layers.append(block(out_channels, out_channels, True))
        return nn.Sequential(*layers)

    def forward(self, x):
        pstem = self.conv1(x)
        pstem = self.bn1(pstem)
        stem = self.relu(pstem)
        stem = pstem, stem
        rb1 = self.res1(stem[1])
        rb2 = self.res2(rb1[1])
        rb3 = self.res3(rb2[1])
        feat = self.avgpool(rb3[1])
        feat = feat.view(feat.size(0), -1)
        out = self.fc(feat)
        return stem, rb1, rb2, rb3, feat, out

    def get_channel_num(self):
        return [16, 16, 32, 64, 64, self.num_class]

    def get_chw_num(self):
        return [(16, 32, 32), (16, 32, 32), (32, 16, 16), (64, 8, 8), (64,), (self.num_class,)]


class paraphraser(nn.Module):

    def __init__(self, in_channels_t, k, use_bn=True):
        super(paraphraser, self).__init__()
        factor_channels = int(in_channels_t * k)
        self.encoder = nn.Sequential(*[nn.Conv2d(in_channels_t, in_channels_t, 3, 1, 1, bias=bool(1 - use_bn)), nn.BatchNorm2d(in_channels_t) if use_bn else nn.Sequential(), nn.LeakyReLU(0.1, inplace=True), nn.Conv2d(in_channels_t, factor_channels, 3, 1, 1, bias=bool(1 - use_bn)), nn.BatchNorm2d(factor_channels) if use_bn else nn.Sequential(), nn.LeakyReLU(0.1, inplace=True), nn.Conv2d(factor_channels, factor_channels, 3, 1, 1, bias=bool(1 - use_bn)), nn.BatchNorm2d(factor_channels) if use_bn else nn.Sequential(), nn.LeakyReLU(0.1, inplace=True)])
        self.decoder = nn.Sequential(*[nn.ConvTranspose2d(factor_channels, factor_channels, 3, 1, 1, bias=bool(1 - use_bn)), nn.BatchNorm2d(factor_channels) if use_bn else nn.Sequential(), nn.LeakyReLU(0.1, inplace=True), nn.ConvTranspose2d(factor_channels, in_channels_t, 3, 1, 1, bias=bool(1 - use_bn)), nn.BatchNorm2d(in_channels_t) if use_bn else nn.Sequential(), nn.LeakyReLU(0.1, inplace=True), nn.ConvTranspose2d(in_channels_t, in_channels_t, 3, 1, 1, bias=bool(1 - use_bn)), nn.BatchNorm2d(in_channels_t) if use_bn else nn.Sequential(), nn.LeakyReLU(0.1, inplace=True)])
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return z, out


class translator(nn.Module):

    def __init__(self, in_channels_s, in_channels_t, k, use_bn=True):
        super(translator, self).__init__()
        factor_channels = int(in_channels_t * k)
        self.encoder = nn.Sequential(*[nn.Conv2d(in_channels_s, in_channels_s, 3, 1, 1, bias=bool(1 - use_bn)), nn.BatchNorm2d(in_channels_s) if use_bn else nn.Sequential(), nn.LeakyReLU(0.1, inplace=True), nn.Conv2d(in_channels_s, factor_channels, 3, 1, 1, bias=bool(1 - use_bn)), nn.BatchNorm2d(factor_channels) if use_bn else nn.Sequential(), nn.LeakyReLU(0.1, inplace=True), nn.Conv2d(factor_channels, factor_channels, 3, 1, 1, bias=bool(1 - use_bn)), nn.BatchNorm2d(factor_channels) if use_bn else nn.Sequential(), nn.LeakyReLU(0.1, inplace=True)])
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        z = self.encoder(x)
        return z


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AB,
     lambda: ([], {'margin': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (AFD,
     lambda: ([], {'in_channels': 4, 'att_f': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (AT,
     lambda: ([], {'p': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (BSS,
     lambda: ([], {'T': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (CC,
     lambda: ([], {'gamma': 4, 'P_order': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
    (ContrastLoss,
     lambda: ([], {'n_data': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DML,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Embed,
     lambda: ([], {'in_dim': 4, 'out_dim': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (FSP,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (FT,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Hint,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Logits,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (NST,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (OFD,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (PKTCosSim,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
    (RKD,
     lambda: ([], {'w_dist': 4, 'w_angle': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
    (SP,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (SoftTarget,
     lambda: ([], {'T': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (VID,
     lambda: ([], {'in_channels': 4, 'mid_channels': 4, 'out_channels': 4, 'init_var': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (paraphraser,
     lambda: ([], {'in_channels_t': 4, 'k': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (resblock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'return_before_act': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (translator,
     lambda: ([], {'in_channels_s': 4, 'in_channels_t': 4, 'k': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_AberHu_Knowledge_Distillation_Zoo(_paritybench_base):
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

