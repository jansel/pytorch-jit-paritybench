import sys
_module = sys.modules[__name__]
del sys
BilinearSampling = _module
DirectVOLayer = _module
ImagePyramid = _module
KITTIdataset = _module
LKVOLearner = _module
LKVOLearnerFinetune = _module
MatInverse = _module
SfMLearner = _module
networks = _module
options = _module
base_options = _module
test_options = _module
train_options = _module
testKITTI = _module
train_main_ddvo = _module
train_main_finetune = _module
train_main_posenet = _module
util = _module
get_data = _module
html = _module
image_pool = _module
png = _module
util = _module
visualizer = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import torch


from torch import FloatTensor


from torch.autograd import Variable


from torch.nn.functional import grid_sample


from torch.nn import ReplicationPad2d


import torch.nn as nn


from torch import optim


from torch.nn import AvgPool2d


import numpy as np


import scipy.io as sio


from torch.nn.functional import conv2d


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from torch.autograd import gradcheck


import itertools


from torch.nn import init


import functools


import torchvision


from collections import OrderedDict


import random


import inspect


import re


import collections


class LaplacianLayer(nn.Module):

    def __init__(self):
        super(LaplacianLayer, self).__init__()
        w_nom = torch.FloatTensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]).view(1, 1, 3, 3)
        w_den = torch.FloatTensor([[0, 1, 0], [1, 4, 1], [0, 1, 0]]).view(1, 1, 3, 3)
        self.register_buffer('w_nom', w_nom)
        self.register_buffer('w_den', w_den)

    def forward(self, input, do_normalize=True):
        assert input.dim() == 2 or input.dim() == 3 or input.dim() == 4
        input_size = input.size()
        if input.dim() == 4:
            x = input.view(input_size[0] * input_size[1], 1, input_size[2], input_size[3])
        elif input.dim() == 3:
            x = input.unsqueeze(1)
        else:
            x = input.unsqueeze(0).unsqueeze(0)
        x_nom = torch.nn.functional.conv2d(input=x, weight=Variable(self.w_nom), stride=1, padding=0)
        if do_normalize:
            x_den = torch.nn.functional.conv2d(input=x, weight=Variable(self.w_den), stride=1, padding=0)
            x = x_nom.abs() / x_den
        else:
            x = x_nom.abs()
        if input.dim() == 4:
            return x.view(input_size[0], input_size[1], input_size[2] - 2, input_size[3] - 2)
        elif input.dim() == 3:
            return x.squeeze(1)
        elif input.dim() == 2:
            return x.squeeze(0).squeeze(0)


class GradientLayer(nn.Module):

    def __init__(self):
        super(GradientLayer, self).__init__()
        wx = torch.FloatTensor([-0.5, 0, 0.5]).view(1, 1, 1, 3)
        wy = torch.FloatTensor([[-0.5], [0], [0.5]]).view(1, 1, 3, 1)
        self.register_buffer('wx', wx)
        self.register_buffer('wy', wy)
        self.padx_func = torch.nn.ReplicationPad2d((1, 1, 0, 0))
        self.pady_func = torch.nn.ReplicationPad2d((0, 0, 1, 1))

    def forward(self, img):
        img_ = img.unsqueeze(1)
        img_padx = self.padx_func(img_)
        img_dx = torch.nn.functional.conv2d(input=img_padx, weight=Variable(self.wx), stride=1, padding=0).squeeze(1)
        img_pady = self.pady_func(img_)
        img_dy = torch.nn.functional.conv2d(input=img_pady, weight=Variable(self.wy), stride=1, padding=0).squeeze(1)
        return img_dx, img_dy


class Twist2Mat(nn.Module):

    def __init__(self):
        super(Twist2Mat, self).__init__()
        self.register_buffer('o', torch.zeros(1, 1))
        self.register_buffer('E', torch.eye(3))

    def cprodmat_batch(self, a_batch):
        batch_size, _ = a_batch.size()
        o = Variable(self.o).expand(batch_size, 1)
        a0 = a_batch[:, 0:1]
        a1 = a_batch[:, 1:2]
        a2 = a_batch[:, 2:3]
        return torch.cat((o, -a2, a1, a2, o, -a0, -a1, a0, o), 1).view(batch_size, 3, 3)

    def forward(self, twist_batch):
        batch_size, _ = twist_batch.size()
        rot_angle = twist_batch.norm(p=2, dim=1).view(batch_size, 1).clamp(min=1e-05)
        rot_axis = twist_batch / rot_angle.expand(batch_size, 3)
        A = self.cprodmat_batch(rot_axis)
        return Variable(self.E).view(1, 3, 3).expand(batch_size, 3, 3) + A * rot_angle.sin().view(batch_size, 1, 1).expand(batch_size, 3, 3) + A.bmm(A) * (1 - rot_angle.cos()).view(batch_size, 1, 1).expand(batch_size, 3, 3)


IMG_CHAN = 3


class ImagePyramidLayer(nn.Module):

    def __init__(self, chan, pyramid_layer_num):
        super(ImagePyramidLayer, self).__init__()
        self.pyramid_layer_num = pyramid_layer_num
        F = torch.FloatTensor([[0.0751, 0.1238, 0.0751], [0.1238, 0.2042, 0.1238], [0.0751, 0.1238, 0.0751]]).view(1, 1, 3, 3)
        self.register_buffer('smooth_kernel', F)
        if chan > 1:
            f = F
            F = torch.zeros(chan, chan, 3, 3)
            for i in range(chan):
                F[(i), (i), :, :] = f
        self.register_buffer('smooth_kernel_K', F)
        self.avg_pool_func = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.reflection_pad_func = torch.nn.ReflectionPad2d(1)

    def downsample(self, input):
        output_dim = input.dim()
        output_size = input.size()
        if output_dim == 2:
            F = self.smooth_kernel
            input = input.unsqueeze(0).unsqueeze(0)
        elif output_dim == 3:
            F = self.smooth_kernel
            input = input.unsqueeze(1)
        else:
            F = self.smooth_kernel_K
        x = self.reflection_pad_func(input)
        x = conv2d(input=x, weight=Variable(F), stride=1, padding=0)
        padding = [0, int(np.mod(input.size(-1), 2)), 0, int(np.mod(input.size(-2), 2))]
        x = torch.nn.ReplicationPad2d(padding)(x)
        x = self.avg_pool_func(x)
        if output_dim == 2:
            x = x.squeeze(0).squeeze(0)
        elif output_dim == 3:
            x = x.squeeze(1)
        return x

    def forward(self, input, do_detach=True):
        pyramid = [input]
        for i in range(self.pyramid_layer_num - 1):
            img_d = self.downsample(pyramid[i])
            if isinstance(img_d, Variable) and do_detach:
                img_d = img_d.detach()
            pyramid.append(img_d)
            assert np.ceil(pyramid[i].size(-1) / 2) == img_d.size(-1)
        return pyramid

    def get_coords(self, imH, imW):
        x_pyramid = [np.arange(imW) + 0.5]
        y_pyramid = [np.arange(imH) + 0.5]
        for i in range(self.pyramid_layer_num - 1):
            offset = 2 ** i
            stride = 2 ** (i + 1)
            x_pyramid.append(np.arange(offset, offset + stride * np.ceil(x_pyramid[i].shape[0] / 2), stride))
            y_pyramid.append(np.arange(offset, offset + stride * np.ceil(y_pyramid[i].shape[0] / 2), stride))
        return x_pyramid, y_pyramid


def compute_SSIM(img0, mu0, sigma0, img1, mu1, sigma1):
    img0_img1_pad = img0 * img1
    sigma01 = AvgPool2d(kernel_size=3, stride=1, padding=0)(img0_img1_pad) - mu0 * mu1
    C1 = 0.001
    C2 = 0.009
    ssim_n = (2 * mu0 * mu1 + C1) * (2 * sigma01 + C2)
    ssim_d = (mu0 ** 2 + mu1 ** 2 + C1) * (sigma0 + sigma1 + C2)
    ssim = ssim_n / ssim_d
    return ((1 - ssim) * 0.5).clamp(0, 1)


def compute_img_stats(img):
    img_pad = img
    mu = AvgPool2d(kernel_size=3, stride=1, padding=0)(img_pad)
    sigma = AvgPool2d(kernel_size=3, stride=1, padding=0)(img_pad ** 2) - mu ** 2
    return mu, sigma


def compute_photometric_cost_norm(img_diff, mask):
    cost = img_diff.abs().sum(1) * mask
    num_in_view = mask.sum(1)
    cost_norm = cost.sum(1) / (num_in_view + 1e-10)
    return cost_norm * (1 / 127.5), (num_in_view / mask.size(1)).min()


def gradient(input, do_normalize=False):
    if input.dim() == 2:
        D_ry = input[1:, :]
        D_ly = input[:-1, :]
        D_rx = input[:, 1:]
        D_lx = input[:, :-1]
    elif input.dim() == 3:
        D_ry = input[:, 1:, :]
        D_ly = input[:, :-1, :]
        D_rx = input[:, :, 1:]
        D_lx = input[:, :, :-1]
    elif input.dim() == 4:
        D_ry = input[:, :, 1:, :]
        D_ly = input[:, :, :-1, :]
        D_rx = input[:, :, :, 1:]
        D_lx = input[:, :, :, :-1]
    Dx = D_rx - D_lx
    Dy = D_ry - D_ly
    if do_normalize:
        Dx = Dx / (D_rx + D_lx)
        Dy = Dy / (D_ry + D_ly)
    return Dx, Dy


def grid_bilinear_sampling(A, x, y):
    batch_size, k, h, w = A.size()
    x_norm = x / ((w - 1) / 2) - 1
    y_norm = y / ((h - 1) / 2) - 1
    grid = torch.cat((x_norm.view(batch_size, h, w, 1), y_norm.view(batch_size, h, w, 1)), 3)
    Q = grid_sample(A, grid, mode='bilinear')
    in_view_mask = Variable(((x_norm.data > -1 + 2 / w) & (x_norm.data < 1 - 2 / w) & (y_norm.data > -1 + 2 / h) & (y_norm.data < 1 - 2 / h)).type_as(A.data))
    return Q.view(batch_size, k, h * w), in_view_mask


class Inverse(torch.autograd.Function):

    def forward(self, input):
        h, w = input.size()
        assert h == w
        H = input.inverse()
        self.save_for_backward(H)
        return H

    def backward(self, grad_output):
        H, = self.saved_tensors
        h, w = H.size()
        assert h == w
        Hl = H.t().repeat(1, h).view(h * h, h, 1)
        Hr = H.repeat(h, 1).view(h * h, 1, h)
        r = Hl.bmm(Hr).view(h, h, h, h) * grad_output.contiguous().view(1, 1, h, h).expand(h, h, h, h)
        return -r.sum(-1).sum(-1)


def inv(input):
    return Inverse()(input)


def inv_rigid_transformation(rot_mat_batch, trans_batch):
    inv_rot_mat_batch = rot_mat_batch.transpose(1, 2)
    inv_trans_batch = -inv_rot_mat_batch.bmm(trans_batch.unsqueeze(-1)).squeeze(-1)
    return inv_rot_mat_batch, inv_trans_batch


def meshgrid(x, y):
    imW = x.size(0)
    imH = y.size(0)
    X = x.unsqueeze(0).repeat(imH, 1)
    Y = y.unsqueeze(1).repeat(1, imW)
    return X, Y


class DirectVO(nn.Module):

    def __init__(self, imH=128, imW=416, pyramid_layer_num=5, max_itr_num=20):
        super(DirectVO, self).__init__()
        self.max_itr_num = max_itr_num
        self.imH = imH
        self.imW = imW
        self.pyramid_layer_num = pyramid_layer_num
        self.twist2mat_batch_func = Twist2Mat()
        self.img_gradient_func = GradientLayer()
        self.pyramid_func = ImagePyramidLayer(chan=3, pyramid_layer_num=self.pyramid_layer_num)
        self.laplacian_func = LaplacianLayer()
        x_pyramid, y_pyramid = self.pyramid_func.get_coords(self.imH, self.imW)
        for i in range(self.pyramid_layer_num):
            self.register_buffer('x_' + str(i), torch.from_numpy(x_pyramid[i]).float())
            self.register_buffer('y_' + str(i), torch.from_numpy(y_pyramid[i]).float())
        self.register_buffer('o', torch.zeros(1, 1))
        self.register_buffer('E', torch.eye(3))

    def setCamera(self, cx, cy, fx, fy):
        self.camparams = dict(fx=fx, fy=fy, cx=cx, cy=cy)

    def init(self, ref_frame_pyramid, inv_depth_pyramid):
        assert self.pyramid_layer_num == len(inv_depth_pyramid)
        self.inv_depth_pyramid = inv_depth_pyramid
        self.ref_frame_pyramid = ref_frame_pyramid
        for i in range(self.pyramid_layer_num):
            assert self.ref_frame_pyramid[i].size(-1) == inv_depth_pyramid[i].size(-1)
            assert self.ref_frame_pyramid[i].size(-2) == inv_depth_pyramid[i].size(-2)
        self.init_lk_terms()

    def init_lk_terms(self):
        self.xy_pyramid = []
        self.ref_imgrad_x_pyramid = []
        self.ref_imgrad_y_pyramid = []
        self.invH_pyramid = []
        self.dIdp_pyramid = []
        self.invH_dIdp_pyramid = []
        for i in range(self.pyramid_layer_num):
            _, h, w = self.ref_frame_pyramid[i].size()
            x = (Variable(getattr(self, 'x_' + str(i))) - self.camparams['cx']) / self.camparams['fx']
            y = (Variable(getattr(self, 'y_' + str(i))) - self.camparams['cy']) / self.camparams['fy']
            X, Y = meshgrid(x, y)
            xy = torch.cat((X.view(1, X.numel()), Y.view(1, Y.numel())), 0)
            self.xy_pyramid.append(xy)
            imgrad_x, imgrad_y = self.img_gradient_func(self.ref_frame_pyramid[i])
            self.ref_imgrad_x_pyramid.append(imgrad_x * (self.camparams['fx'] / 2 ** i))
            self.ref_imgrad_y_pyramid.append(imgrad_y * (self.camparams['fy'] / 2 ** i))
            dIdp = self.compute_dIdp(self.ref_imgrad_x_pyramid[i], self.ref_imgrad_y_pyramid[i], self.inv_depth_pyramid[i], self.xy_pyramid[i])
            self.dIdp_pyramid.append(dIdp)
            invH = inv(dIdp.t().mm(dIdp))
            self.invH_pyramid.append(invH)
            self.invH_dIdp_pyramid.append(invH.mm(dIdp.t()))

    def init_xy_pyramid(self, ref_frames_pyramid):
        self.xy_pyramid = []
        self.ref_imgrad_x_pyramid = []
        self.ref_imgrad_y_pyramid = []
        for i in range(self.pyramid_layer_num):
            _, h, w = ref_frames_pyramid[i].size()
            x = (Variable(getattr(self, 'x_' + str(i))) - self.camparams['cx']) / self.camparams['fx']
            y = (Variable(getattr(self, 'y_' + str(i))) - self.camparams['cy']) / self.camparams['fy']
            X, Y = meshgrid(x, y)
            xy = torch.cat((X.view(1, X.numel()), Y.view(1, Y.numel())), 0)
            self.xy_pyramid.append(xy)

    def compute_dIdp(self, imgrad_x, imgrad_y, inv_depth, xy):
        k, h, w = imgrad_x.size()
        _, pt_num = xy.size()
        assert h * w == pt_num
        feat_dim = pt_num * k
        x = xy[(0), :].view(pt_num, 1)
        y = xy[(1), :].view(pt_num, 1)
        xty = x * y
        O = Variable(self.o).expand(pt_num, 1)
        inv_depth_ = inv_depth.view(pt_num, 1)
        dxdp = torch.cat((-xty, 1 + x ** 2, -y, inv_depth_, O, -inv_depth_.mul(x)), 1)
        dydp = torch.cat((-1 - y ** 2, xty, x, O, inv_depth_, -inv_depth_.mul(y)), 1)
        imgrad_x_ = imgrad_x.view(feat_dim, 1).expand(feat_dim, 6)
        imgrad_y_ = imgrad_y.view(feat_dim, 1).expand(feat_dim, 6)
        dIdp = imgrad_x_.mul(dxdp.repeat(k, 1)) + imgrad_y_.mul(dydp.repeat(k, 1))
        return dIdp

    def LKregress(self, invH_dIdp, mask, It):
        batch_size, pt_num = mask.size()
        _, k, _ = It.size()
        feat_dim = k * pt_num
        invH_dIdp_ = invH_dIdp.view(1, 6, feat_dim).expand(batch_size, 6, feat_dim)
        mask_ = mask.view(batch_size, 1, pt_num).expand(batch_size, k, pt_num)
        dp = invH_dIdp_.bmm((mask_ * It).view(batch_size, feat_dim, 1))
        return dp.view(batch_size, 6)

    def warp_batch(self, img_batch, level_idx, R_batch, t_batch):
        return self.warp_batch_func(img_batch, self.inv_depth_pyramid[level_idx], level_idx, R_batch, t_batch)

    def warp_batch_func(self, img_batch, inv_depth, level_idx, R_batch, t_batch):
        batch_size, k, h, w = img_batch.size()
        xy = self.xy_pyramid[level_idx]
        _, N = xy.size()
        xyz = R_batch[:, :, 0:2].bmm(xy.view(1, 2, N).expand(batch_size, 2, N)) + R_batch[:, :, 2:3].expand(batch_size, 3, N) + t_batch.view(batch_size, 3, 1).expand(batch_size, 3, N) * inv_depth.view(-1, 1, N).expand(batch_size, 3, N)
        z = xyz[:, 2:3, :].clamp(min=1e-10)
        xy_warp = xyz[:, 0:2, :] / z.expand(batch_size, 2, N)
        u_warp = (xy_warp[:, (0), :] * self.camparams['fx'] + self.camparams['cx'] - getattr(self, 'x_' + str(level_idx))[0]).view(batch_size, N) / 2 ** level_idx
        v_warp = (xy_warp[:, (1), :] * self.camparams['fy'] + self.camparams['cy'] - getattr(self, 'y_' + str(level_idx))[0]).view(batch_size, N) / 2 ** level_idx
        Q, in_view_mask = grid_bilinear_sampling(img_batch, u_warp, v_warp)
        return Q, in_view_mask * (z.view_as(in_view_mask) > 1e-10).float()

    def compute_phtometric_loss(self, ref_frames_pyramid, src_frames_pyramid, ref_inv_depth_pyramid, src_inv_depth_pyramid, rot_mat_batch, trans_batch, use_ssim=True, levels=None, ref_expl_mask_pyramid=None, src_expl_mask_pyramid=None):
        bundle_size = rot_mat_batch.size(0) + 1
        inv_rot_mat_batch, inv_trans_batch = inv_rigid_transformation(rot_mat_batch, trans_batch)
        src_pyramid = []
        ref_pyramid = []
        depth_pyramid = []
        if levels is None:
            levels = range(self.pyramid_layer_num)
        use_expl_mask = not (ref_expl_mask_pyramid is None or src_expl_mask_pyramid is None)
        if use_expl_mask:
            expl_mask_pyramid = []
            for level_idx in levels:
                ref_mask = ref_expl_mask_pyramid[level_idx].unsqueeze(0).repeat(bundle_size - 1, 1, 1)
                src_mask = src_expl_mask_pyramid[level_idx]
                expl_mask_pyramid.append(torch.cat((ref_mask, src_mask), 0))
        for level_idx in levels:
            ref_frame = ref_frames_pyramid[level_idx].unsqueeze(0).repeat(bundle_size - 1, 1, 1, 1)
            src_frame = src_frames_pyramid[level_idx]
            ref_depth = ref_inv_depth_pyramid[level_idx].unsqueeze(0).repeat(bundle_size - 1, 1, 1)
            src_depth = src_inv_depth_pyramid[level_idx]
            ref_pyramid.append(torch.cat((ref_frame, src_frame), 0) / 127.5)
            src_pyramid.append(torch.cat((src_frame, ref_frame), 0) / 127.5)
            depth_pyramid.append(torch.cat((ref_depth, src_depth), 0))
        rot_mat = torch.cat((rot_mat_batch, inv_rot_mat_batch), 0)
        trans = torch.cat((trans_batch, inv_trans_batch), 0)
        loss = 0
        frames_warp_pyramid = []
        ref_frame_warp_pyramid = []
        for level_idx in levels:
            _, h, w = depth_pyramid[level_idx].size()
            warp_img, in_view_mask = self.warp_batch_func(src_pyramid[level_idx], depth_pyramid[level_idx], level_idx, rot_mat, trans)
            warp_img = warp_img.view((bundle_size - 1) * 2, IMG_CHAN, h, w)
            if use_expl_mask:
                mask = in_view_mask.view(-1, h, w) * expl_mask_pyramid[level_idx]
            else:
                mask = in_view_mask
            mask_expand = mask.view((bundle_size - 1) * 2, 1, h, w).expand((bundle_size - 1) * 2, IMG_CHAN, h, w)
            rgb_loss = ((ref_pyramid[level_idx] - warp_img).abs() * mask_expand).mean()
            if use_ssim and level_idx < 1:
                warp_mu, warp_sigma = compute_img_stats(warp_img)
                ref_mu, ref_sigma = compute_img_stats(ref_pyramid[level_idx])
                ssim = compute_SSIM(ref_pyramid[level_idx], ref_mu, ref_sigma, warp_img, warp_mu, warp_sigma)
                ssim_loss = (ssim * mask_expand[:, :, 1:-1, 1:-1]).mean()
                loss += 0.85 * ssim_loss + 0.15 * rgb_loss
            else:
                loss += rgb_loss
        return loss

    def compute_smoothness_cost(self, inv_depth):
        x = self.laplacian_func(inv_depth)
        return x.mean()

    def compute_image_aware_laplacian_smoothness_cost(self, depth, img):
        img_lap = self.laplacian_func(img / 255, do_normalize=False)
        depth_lap = self.laplacian_func(depth, do_normalize=False)
        x = (-img_lap.mean(1)).exp() * depth_lap
        return x.mean()

    def compute_image_aware_2nd_smoothness_cost(self, depth, img):
        img_lap = self.laplacian_func(img / 255, do_normalize=False)
        depth_grad_x, depth_grad_y = gradient(depth, do_normalize=False)
        depth_grad_x2, depth_grad_xy = gradient(depth_grad_x, do_normalize=False)
        depth_grad_yx, depth_grad_y2 = gradient(depth_grad_y, do_normalize=False)
        return depth_grad_x2.abs().mean() + depth_grad_xy.abs().mean() + depth_grad_yx.abs().mean() + depth_grad_y2.abs().mean()

    def compute_image_aware_1st_smoothness_cost(self, depth, img):
        depth_grad_x, depth_grad_y = gradient(depth, do_normalize=False)
        img_grad_x, img_grad_y = gradient(img / 255, do_normalize=False)
        if img.dim() == 3:
            weight_x = torch.exp(-img_grad_x.abs().mean(0))
            weight_y = torch.exp(-img_grad_y.abs().mean(0))
            cost = ((depth_grad_x.abs() * weight_x)[:-1, :] + (depth_grad_y.abs() * weight_y)[:, :-1]).mean()
        else:
            weight_x = torch.exp(-img_grad_x.abs().mean(1))
            weight_y = torch.exp(-img_grad_y.abs().mean(1))
            cost = ((depth_grad_x.abs() * weight_x)[:, :-1, :] + (depth_grad_y.abs() * weight_y)[:, :, :-1]).mean()
        return cost

    def multi_scale_smoothness_cost(self, inv_depth_pyramid, levels=None):
        cost = 0
        if levels is None:
            levels = range(self.pyramid_layer_num)
        for level_idx in levels:
            inv_depth = inv_depth_pyramid[level_idx]
            if inv_depth.dim() == 4:
                inv_depth = inv_depth.squeeze(1)
            cost += self.compute_smoothness_cost(inv_depth) / 2 ** level_idx
        return cost

    def multi_scale_image_aware_smoothness_cost(self, inv_depth_pyramid, img_pyramid, levels=None, type='lap'):
        cost = 0
        if levels is None:
            levels = range(self.pyramid_layer_num)
        for level_idx in levels:
            inv_depth = inv_depth_pyramid[level_idx]
            if inv_depth.dim() == 4:
                inv_depth = inv_depth.squeeze(1)
            if type == 'lap':
                c = self.compute_image_aware_laplacian_smoothness_cost(inv_depth, img_pyramid[level_idx])
            elif type == '1st':
                c = self.compute_image_aware_1st_smoothness_cost(inv_depth, img_pyramid[level_idx])
            elif type == '2nd':
                c = self.compute_image_aware_2nd_smoothness_cost(inv_depth, img_pyramid[level_idx])
            else:
                None
            cost += c / 2 ** level_idx
        return cost

    def update(self, frames_pyramid, max_itr_num=10):
        frame_num, k, h, w = frames_pyramid[0].size()
        trans_batch = Variable(self.o).expand(frame_num, 3).contiguous()
        trans_batch_prev = Variable(self.o).expand(frame_num, 3).contiguous()
        rot_mat_batch = Variable(self.E).unsqueeze(0).expand(frame_num, 3, 3).contiguous()
        rot_mat_batch_prev = Variable(self.E).unsqueeze(0).expand(frame_num, 3, 3).contiguous()
        pixel_warp = []
        in_view_mask = []
        cur_time = timer()
        for level_idx in range(self.pyramid_layer_num - 1, -1, -1):
            max_photometric_cost = self.o.squeeze().expand(frame_num) + 10000
            for i in range(max_itr_num):
                pixel_warp, in_view_mask = self.warp_batch(frames_pyramid[level_idx], level_idx, rot_mat_batch, trans_batch)
                temporal_grad = pixel_warp - self.ref_frame_pyramid[level_idx].view(3, -1).unsqueeze(0).expand_as(pixel_warp)
                photometric_cost, min_perc_in_view = compute_photometric_cost_norm(temporal_grad.data, in_view_mask.data)
                if min_perc_in_view < 0.5:
                    break
                if (photometric_cost < max_photometric_cost).max() > 0:
                    trans_batch_prev = trans_batch
                    rot_mat_batch_prev = rot_mat_batch
                    dp = self.LKregress(invH_dIdp=self.invH_dIdp_pyramid[level_idx], mask=in_view_mask, It=temporal_grad)
                    d_rot_mat_batch = self.twist2mat_batch_func(-dp[:, 0:3])
                    trans_batch_new = d_rot_mat_batch.bmm(trans_batch.view(frame_num, 3, 1)).view(frame_num, 3) - dp[:, 3:6]
                    rot_mat_batch_new = d_rot_mat_batch.bmm(rot_mat_batch)
                    trans_list = []
                    rot_list = []
                    for k in range(frame_num):
                        if photometric_cost[k] < max_photometric_cost[k]:
                            rot_list.append(rot_mat_batch_new[k:k + 1, :, :])
                            trans_list.append(trans_batch_new[k:k + 1, :])
                            max_photometric_cost[k] = photometric_cost[k]
                        else:
                            rot_list.append(rot_mat_batch[k:k + 1, :, :])
                            trans_list.append(trans_batch[k:k + 1, :])
                    rot_mat_batch = torch.cat(rot_list, 0)
                    trans_batch = torch.cat(trans_list, 0)
                else:
                    break
            rot_mat_batch = rot_mat_batch_prev
            trans_batch = trans_batch_prev
        return rot_mat_batch, trans_batch, frames_pyramid

    def update_with_init_pose(self, frames_pyramid, rot_mat_batch, trans_batch, max_itr_num=10):
        frame_num, k, h, w = frames_pyramid[0].size()
        trans_batch_prev = trans_batch
        rot_mat_batch_prev = rot_mat_batch
        pixel_warp = []
        in_view_mask = []
        cur_time = timer()
        for level_idx in range(len(frames_pyramid) - 1, -1, -1):
            max_photometric_cost = self.o.squeeze().expand(frame_num) + 10000
            for i in range(max_itr_num):
                pixel_warp, in_view_mask = self.warp_batch(frames_pyramid[level_idx], level_idx, rot_mat_batch, trans_batch)
                temporal_grad = pixel_warp - self.ref_frame_pyramid[level_idx].view(3, -1).unsqueeze(0).expand_as(pixel_warp)
                photometric_cost, min_perc_in_view = compute_photometric_cost_norm(temporal_grad.data, in_view_mask.data)
                if min_perc_in_view < 0.5:
                    break
                if (photometric_cost < max_photometric_cost).max() > 0:
                    trans_batch_prev = trans_batch
                    rot_mat_batch_prev = rot_mat_batch
                    dp = self.LKregress(invH_dIdp=self.invH_dIdp_pyramid[level_idx], mask=in_view_mask, It=temporal_grad)
                    d_rot_mat_batch = self.twist2mat_batch_func(dp[:, 0:3]).transpose(1, 2)
                    trans_batch_new = d_rot_mat_batch.bmm(trans_batch.view(frame_num, 3, 1)).view(frame_num, 3) - dp[:, 3:6]
                    rot_mat_batch_new = d_rot_mat_batch.bmm(rot_mat_batch)
                    trans_list = []
                    rot_list = []
                    for k in range(frame_num):
                        if photometric_cost[k] < max_photometric_cost[k]:
                            rot_list.append(rot_mat_batch_new[k:k + 1, :, :])
                            trans_list.append(trans_batch_new[k:k + 1, :])
                            max_photometric_cost[k] = photometric_cost[k]
                        else:
                            rot_list.append(rot_mat_batch[k:k + 1, :, :])
                            trans_list.append(trans_batch[k:k + 1, :])
                    rot_mat_batch = torch.cat(rot_list, 0)
                    trans_batch = torch.cat(trans_list, 0)
                else:
                    break
            rot_mat_batch = rot_mat_batch_prev
            trans_batch = trans_batch_prev
        return rot_mat_batch, trans_batch

    def forward(self, ref_frame_pyramid, src_frame_pyramid, ref_inv_depth_pyramid, max_itr_num=10):
        self.init(ref_frame_pyramid=ref_frame_pyramid, inv_depth_pyramid=ref_inv_depth_pyramid)
        rot_mat_batch, trans_batch, src_frames_pyramid = self.update(src_frame_pyramid, max_itr_num=max_itr_num)
        return rot_mat_batch, trans_batch


class ImageSmoothLayer(nn.Module):

    def __init__(self, chan):
        super(ImageSmoothLayer, self).__init__()
        F = torch.FloatTensor([[0.0751, 0.1238, 0.0751], [0.1238, 0.2042, 0.1238], [0.0751, 0.1238, 0.0751]]).view(1, 1, 3, 3)
        self.register_buffer('smooth_kernel', F)
        if chan > 1:
            f = F
            F = torch.zeros(chan, chan, 3, 3)
            for i in range(chan):
                F[(i), (i), :, :] = f
        self.register_buffer('smooth_kernel_K', F)
        self.reflection_pad_func = torch.nn.ReflectionPad2d(1)

    def forward(self, input):
        output_dim = input.dim()
        output_size = input.size()
        if output_dim == 2:
            F = self.smooth_kernel
            input = input.unsqueeze(0).unsqueeze(0)
        elif output_dim == 3:
            F = self.smooth_kernel
            input = input.unsqueeze(1)
        else:
            F = self.smooth_kernel_K
        x = self.reflection_pad_func(input)
        x = conv2d(input=x, weight=Variable(F), stride=1, padding=0)
        if output_dim == 2:
            x = x.squeeze(0).squeeze(0)
        elif output_dim == 3:
            x = x.squeeze(1)
        return x


class FlipLR(nn.Module):

    def __init__(self, imW, dim_w):
        super(FlipLR, self).__init__()
        inv_indices = torch.arange(imW - 1, -1, -1).long()
        self.register_buffer('inv_indices', inv_indices)
        self.dim_w = dim_w

    def forward(self, input):
        return input.index_select(self.dim_w, Variable(self.inv_indices))


class PoseNet(nn.Module):

    def __init__(self, bundle_size):
        super(PoseNet, self).__init__()
        self.bundle_size = bundle_size
        model = [nn.Conv2d(bundle_size * 3, 16, kernel_size=7, stride=2, padding=3, bias=True), nn.ReLU(True), nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2, bias=True), nn.ReLU(True), nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=True), nn.ReLU(True), nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=True), nn.ReLU(True), nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=True), nn.ReLU(True), nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True), nn.ReLU(True), nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True), nn.ReLU(True), nn.Conv2d(256, 6 * (bundle_size - 1), kernel_size=3, stride=2, padding=1, bias=True)]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        assert self.bundle_size * 3 == input.size(1)
        p = self.model.forward(input)
        p = p.view(input.size(0), 6 * (self.bundle_size - 1), -1).mean(2)
        return p.view(input.size(0), self.bundle_size - 1, 6) * 0.01


class Conv(nn.Module):

    def __init__(self, input_nc, output_nc, kernel_size, stride, padding, activation_func=nn.ELU()):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels=input_nc, out_channels=output_nc, kernel_size=kernel_size, stride=stride, padding=0, bias=True)
        self.activation_fn = activation_func
        self.pad_fn = nn.ReplicationPad2d(padding)

    def forward(self, input):
        if self.activation_fn == None:
            return self.conv(self.pad_fn(input))
        else:
            return self.activation_fn(self.conv(self.pad_fn(input)))


class ConvBlock(nn.Module):

    def __init__(self, input_nc, output_nc, kernel_size):
        super(ConvBlock, self).__init__()
        p = int(np.floor((kernel_size - 1) / 2))
        self.activation_fn = nn.ELU()
        self.conv1 = Conv(input_nc, output_nc, kernel_size, 1, p, self.activation_fn)
        self.conv2 = Conv(output_nc, output_nc, kernel_size, 1, p, None)

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        padding = [0, int(np.mod(input.size(-1), 2)), 0, int(np.mod(input.size(-2), 2))]
        x_pad = torch.nn.ReplicationPad2d(padding)(x)
        return torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)(self.activation_fn(x_pad))


DISP_SCALING = 10


MIN_DISP = 0.01


class UpConv(nn.Module):

    def __init__(self, input_nc, output_nc, kernel_size):
        super(UpConv, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels=input_nc, out_channels=output_nc, kernel_size=2, bias=True, stride=2, padding=0)
        self.activation_fn = nn.ELU()

    def forward(self, input):
        return self.activation_fn(self.deconv(input))


class VggDepthEstimator(nn.Module):

    def __init__(self, input_size=None):
        super(VggDepthEstimator, self).__init__()
        self.conv_layers = nn.ModuleList([ConvBlock(3, 32, 7)])
        self.conv_layers.append(ConvBlock(32, 64, 5))
        self.conv_layers.append(ConvBlock(64, 128, 3))
        self.conv_layers.append(ConvBlock(128, 256, 3))
        self.conv_layers.append(ConvBlock(256, 512, 3))
        self.conv_layers.append(ConvBlock(512, 512, 3))
        self.conv_layers.append(ConvBlock(512, 512, 3))
        self.upconv_layers = nn.ModuleList([UpConv(512, 512, 3)])
        self.iconv_layers = nn.ModuleList([Conv(512 * 2, 512, 3, 1, 1)])
        self.upconv_layers.append(UpConv(512, 512, 3))
        self.iconv_layers.append(Conv(512 * 2, 512, 3, 1, 1))
        self.invdepth_layers = nn.ModuleList([Conv(512, 1, 3, 1, 1, nn.Sigmoid())])
        self.upconv_layers.append(UpConv(512, 256, 3))
        self.iconv_layers.append(Conv(256 * 2, 256, 3, 1, 1))
        self.invdepth_layers.append(Conv(256, 1, 3, 1, 1, nn.Sigmoid()))
        self.upconv_layers.append(UpConv(256, 128, 3))
        self.iconv_layers.append(Conv(128 * 2, 128, 3, 1, 1))
        self.invdepth_layers.append(Conv(128, 1, 3, 1, 1, nn.Sigmoid()))
        self.upconv_layers.append(UpConv(128, 64, 3))
        self.iconv_layers.append(Conv(64 * 2 + 1, 64, 3, 1, 1))
        self.invdepth_layers.append(Conv(64, 1, 3, 1, 1, nn.Sigmoid()))
        self.upconv_layers.append(UpConv(64, 32, 3))
        self.iconv_layers.append(Conv(32 * 2 + 1, 32, 3, 1, 1))
        self.invdepth_layers.append(Conv(32, 1, 3, 1, 1, nn.Sigmoid()))
        self.upconv_layers.append(UpConv(32, 16, 3))
        self.iconv_layers.append(Conv(16 + 1, 16, 3, 1, 1))
        self.invdepth_layers.append(Conv(16, 1, 3, 1, 1, nn.Sigmoid()))

    def init_weights(self):

        def weights_init(m):
            classname = m.__class__.__name__
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):
                m.bias.data = torch.zeros(m.bias.data.size())
        self.apply(weights_init)

    def forward(self, input):
        conv_feat = self.conv_layers[0].forward(input)
        self.conv_feats = [conv_feat]
        for i in range(1, len(self.conv_layers)):
            conv_feat = self.conv_layers[i].forward(self.conv_feats[i - 1])
            self.conv_feats.append(conv_feat)
        upconv_feats = []
        invdepth_pyramid = []
        for i in range(0, len(self.upconv_layers)):
            if i == 0:
                x = self.upconv_layers[i].forward(self.conv_feats[-1])
            else:
                x = self.upconv_layers[i].forward(upconv_feats[i - 1])
            if i < len(self.upconv_layers) - 1:
                if x.size(-1) != self.conv_feats[-2 - i].size(-1):
                    x = x[:, :, :, :-1]
                if x.size(-2) != self.conv_feats[-2 - i].size(-2):
                    x = x[:, :, :-1, :]
            if i == len(self.upconv_layers) - 1:
                x = torch.cat((x, nn.Upsample(scale_factor=2, mode='bilinear')(invdepth_pyramid[-1])), 1)
            elif i > 3:
                x = torch.cat((x, self.conv_feats[-(2 + i)], nn.Upsample(scale_factor=2, mode='bilinear')(invdepth_pyramid[-1])), 1)
            else:
                x = torch.cat((x, self.conv_feats[-(2 + i)]), 1)
            upconv_feats.append(self.iconv_layers[i].forward(x))
            if i > 0:
                invdepth_pyramid.append(self.invdepth_layers[i - 1].forward(upconv_feats[-1]))
        invdepth_pyramid = invdepth_pyramid[-1::-1]
        invdepth_pyramid = invdepth_pyramid[0:5]
        for i in range(len(invdepth_pyramid)):
            invdepth_pyramid[i] = invdepth_pyramid[i].squeeze(1) * DISP_SCALING + MIN_DISP
        return invdepth_pyramid


class LKVOKernel(nn.Module):
    """
     only support single training isinstance
    """

    def __init__(self, img_size=[128, 416], smooth_term='lap'):
        super(LKVOKernel, self).__init__()
        self.img_size = img_size
        self.fliplr_func = FlipLR(imW=img_size[1], dim_w=3)
        self.vo = DirectVO(imH=img_size[0], imW=img_size[1], pyramid_layer_num=4)
        self.pose_net = PoseNet(3)
        self.depth_net = VggDepthEstimator(img_size)
        self.pyramid_func = ImagePyramidLayer(chan=1, pyramid_layer_num=4)
        self.smooth_term = smooth_term

    def forward(self, frames, camparams, ref_frame_idx, lambda_S=0.5, do_data_augment=True, use_ssim=True, max_lk_iter_num=10, lk_level=1):
        assert frames.size(0) == 1 and frames.dim() == 5
        frames = frames.squeeze(0)
        camparams = camparams.squeeze(0).data
        if do_data_augment:
            if np.random.rand() > 0.5:
                frames = self.fliplr_func(frames)
                camparams[2] = self.img_size[1] - camparams[2]
        bundle_size = frames.size(0)
        src_frame_idx = tuple(range(0, ref_frame_idx)) + tuple(range(ref_frame_idx + 1, bundle_size))
        frames_pyramid = self.vo.pyramid_func(frames)
        ref_frame_pyramid = [frame[(ref_frame_idx), :, :, :] for frame in frames_pyramid]
        src_frames_pyramid = [frame[(src_frame_idx), :, :, :] for frame in frames_pyramid]
        self.vo.setCamera(fx=camparams[0], cx=camparams[2], fy=camparams[4], cy=camparams[5])
        inv_depth_pyramid = self.depth_net.forward((frames - 127) / 127)
        inv_depth_mean_ten = inv_depth_pyramid[0].mean() * 0.1
        inv_depth_norm_pyramid = [(depth / inv_depth_mean_ten) for depth in inv_depth_pyramid]
        inv_depth0_pyramid = self.pyramid_func(inv_depth_norm_pyramid[0], do_detach=False)
        ref_inv_depth_pyramid = [depth[(ref_frame_idx), :, :] for depth in inv_depth_norm_pyramid]
        ref_inv_depth0_pyramid = [depth[(ref_frame_idx), :, :] for depth in inv_depth0_pyramid]
        src_inv_depth_pyramid = [depth[(src_frame_idx), :, :] for depth in inv_depth_norm_pyramid]
        src_inv_depth0_pyramid = [depth[(src_frame_idx), :, :] for depth in inv_depth0_pyramid]
        self.vo.init(ref_frame_pyramid=ref_frame_pyramid, inv_depth_pyramid=ref_inv_depth0_pyramid)
        p = self.pose_net.forward((frames.view(1, -1, frames.size(2), frames.size(3)) - 127) / 127)
        rot_mat_batch = self.vo.twist2mat_batch_func(p[(0), :, 0:3]).contiguous()
        trans_batch = p[(0), :, 3:6].contiguous()
        rot_mat_batch, trans_batch = self.vo.update_with_init_pose(src_frames_pyramid[0:lk_level], max_itr_num=max_lk_iter_num, rot_mat_batch=rot_mat_batch, trans_batch=trans_batch)
        photometric_cost = self.vo.compute_phtometric_loss(self.vo.ref_frame_pyramid, src_frames_pyramid, ref_inv_depth_pyramid, src_inv_depth_pyramid, rot_mat_batch, trans_batch, levels=[0, 1, 2, 3], use_ssim=use_ssim)
        smoothness_cost = self.vo.multi_scale_image_aware_smoothness_cost(inv_depth0_pyramid, frames_pyramid, levels=[2, 3], type=self.smooth_term) + self.vo.multi_scale_image_aware_smoothness_cost(inv_depth_norm_pyramid, frames_pyramid, levels=[2, 3], type=self.smooth_term)
        cost = photometric_cost + lambda_S * smoothness_cost
        return cost, photometric_cost, smoothness_cost, self.vo.ref_frame_pyramid[0], ref_inv_depth0_pyramid[0] * inv_depth_mean_ten


class LKVOLearner(nn.Module):

    def __init__(self, img_size=[128, 416], ref_frame_idx=1, lambda_S=0.5, use_ssim=True, smooth_term='lap', gpu_ids=[0]):
        super(LKVOLearner, self).__init__()
        self.lkvo = nn.DataParallel(LKVOKernel(img_size, smooth_term=smooth_term), device_ids=gpu_ids)
        self.ref_frame_idx = ref_frame_idx
        self.lambda_S = lambda_S
        self.use_ssim = use_ssim

    def forward(self, frames, camparams, max_lk_iter_num=10, lk_level=1):
        cost, photometric_cost, smoothness_cost, ref_frame, ref_inv_depth = self.lkvo.forward(frames, camparams, self.ref_frame_idx, self.lambda_S, max_lk_iter_num=max_lk_iter_num, use_ssim=self.use_ssim, lk_level=lk_level)
        return cost.mean(), photometric_cost.mean(), smoothness_cost.mean(), ref_frame, ref_inv_depth

    def save_model(self, file_path):
        torch.save(self.cpu().lkvo.module.depth_net.state_dict(), file_path)
        self

    def load_model(self, depth_net_file_path, pose_net_file_path):
        self.lkvo.module.depth_net.load_state_dict(torch.load(depth_net_file_path))
        self.lkvo.module.pose_net.load_state_dict(torch.load(pose_net_file_path))

    def init_weights(self):
        self.lkvo.module.depth_net.init_weights()

    def get_parameters(self):
        return self.lkvo.module.depth_net.parameters()


class PoseExpNet(nn.Module):

    def __init__(self, bundle_size):
        super(PoseExpNet, self).__init__()
        self.bundle_size = bundle_size
        self.convlyr1 = nn.Sequential(*[nn.Conv2d(bundle_size * 3, 16, kernel_size=7, stride=2, padding=3, bias=True), nn.ReLU(True)])
        self.convlyr2 = nn.Sequential(*[nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2, bias=True), nn.ReLU(True)])
        self.convlyr3 = nn.Sequential(*[nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=True), nn.ReLU(True)])
        self.convlyr4 = nn.Sequential(*[nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=True), nn.ReLU(True)])
        self.convlyr5 = nn.Sequential(*[nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=True), nn.ReLU(True)])
        self.poselyr = nn.Sequential(*[nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True), nn.ReLU(True), nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True), nn.ReLU(True), nn.Conv2d(256, 6 * (bundle_size - 1), kernel_size=3, stride=2, padding=1, bias=True)])
        self.uplyr5 = nn.Sequential(*[nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, bias=True, stride=2, padding=0), nn.ReLU(True)])
        self.uplyr4 = nn.Sequential(*[nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, bias=True, stride=2, padding=0), nn.ReLU(True)])
        self.uplyr3 = nn.Sequential(*[nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, bias=True, stride=2, padding=0), nn.ReLU(True)])
        self.uplyr2 = nn.Sequential(*[nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, bias=True, stride=2, padding=0), nn.ReLU(True)])
        self.uplyr1 = nn.Sequential(*[nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, bias=True, stride=2, padding=0), nn.ReLU(True)])
        self.explyr4 = nn.Sequential(*[nn.Conv2d(128, bundle_size, kernel_size=3, stride=1, padding=1, bias=True), nn.Sigmoid()])
        self.explyr3 = nn.Sequential(*[nn.Conv2d(64, bundle_size, kernel_size=3, stride=1, padding=1, bias=True), nn.Sigmoid()])
        self.explyr2 = nn.Sequential(*[nn.Conv2d(32, bundle_size, kernel_size=3, stride=1, padding=1, bias=True), nn.Sigmoid()])
        self.explyr1 = nn.Sequential(*[nn.Conv2d(16, bundle_size, kernel_size=3, stride=1, padding=1, bias=True), nn.Sigmoid()])

    def forward(self, input):
        conv1 = self.convlyr1(input)
        conv2 = self.convlyr2(conv1)
        conv3 = self.convlyr3(conv2)
        conv4 = self.convlyr4(conv3)
        conv5 = self.convlyr5(conv4)
        p = self.poselyr.forward(conv5)
        p = p.view(input.size(0), 6 * (self.bundle_size - 1), -1).mean(2)
        p = p.view(input.size(0), self.bundle_size - 1, 6) * 0.01
        upcnv5 = self.uplyr5(conv5)
        upcnv4 = self.uplyr4(upcnv5)
        upcnv3 = self.uplyr3(upcnv4)
        upcnv2 = self.uplyr2(upcnv3)
        upcnv1 = self.uplyr1(upcnv2)
        mask4 = self.explyr4(upcnv4)
        mask3 = self.explyr3(upcnv3)
        mask2 = self.explyr2(upcnv2)
        mask1 = self.explyr1(upcnv1)
        return p, [mask1, mask2, mask3, mask4]


class SfMKernel(nn.Module):
    """
     only support single training isinstance
    """

    def __init__(self, img_size=[128, 416], smooth_term='lap', use_expl_mask=False):
        super(SfMKernel, self).__init__()
        self.img_size = img_size
        self.fliplr_func = FlipLR(imW=img_size[1], dim_w=3)
        self.vo = DirectVO(imH=img_size[0], imW=img_size[1], pyramid_layer_num=4)
        self.depth_net = VggDepthEstimator(img_size)
        if use_expl_mask:
            self.pose_net = PoseExpNet(3)
        else:
            self.pose_net = PoseNet(3)
        self.pyramid_func = ImagePyramidLayer(chan=1, pyramid_layer_num=4)
        self.smooth_term = smooth_term
        self.use_expl_mask = use_expl_mask

    def forward(self, frames, camparams, ref_frame_idx, lambda_S=0.5, lambda_E=0.01, do_data_augment=True, use_ssim=True):
        assert frames.size(0) == 1 and frames.dim() == 5
        frames = frames.squeeze(0)
        camparams = camparams.squeeze(0).data
        if do_data_augment:
            if np.random.rand() > 0.5:
                frames = self.fliplr_func(frames)
                camparams[2] = self.img_size[1] - camparams[2]
        bundle_size = frames.size(0)
        src_frame_idx = tuple(range(0, ref_frame_idx)) + tuple(range(ref_frame_idx + 1, bundle_size))
        frames_pyramid = self.vo.pyramid_func(frames)
        ref_frame_pyramid = [frame[(ref_frame_idx), :, :, :] for frame in frames_pyramid]
        src_frames_pyramid = [frame[(src_frame_idx), :, :, :] for frame in frames_pyramid]
        self.vo.setCamera(fx=camparams[0], cx=camparams[2], fy=camparams[4], cy=camparams[5])
        self.vo.init_xy_pyramid(ref_frame_pyramid)
        if self.use_expl_mask:
            p, expl_mask_pyramid = self.pose_net.forward((frames.view(1, -1, frames.size(2), frames.size(3)) - 127) / 127)
            expl_mask_reg_cost = 0
            for mask in expl_mask_pyramid:
                expl_mask_reg_cost += mask.mean()
            ref_expl_mask_pyramid = [mask.squeeze(0)[ref_frame_idx, ...] for mask in expl_mask_pyramid]
            src_expl_mask_pyramid = [mask.squeeze(0)[src_frame_idx, ...] for mask in expl_mask_pyramid]
            expl_mask = ref_expl_mask_pyramid[0]
        else:
            p = self.pose_net.forward((frames.view(1, -1, frames.size(2), frames.size(3)) - 127) / 127)
            ref_expl_mask_pyramid = None
            src_expl_mask_pyramid = None
            expl_mask_reg_cost = 0
            expl_mask = None
        rot_mat_batch = self.vo.twist2mat_batch_func(p[(0), :, 0:3])
        trans_batch = p[(0), :, 3:6]
        inv_depth_pyramid = self.depth_net.forward((frames - 127) / 127)
        inv_depth_mean_ten = inv_depth_pyramid[0].mean() * 0.1
        inv_depth_norm_pyramid = [(depth / inv_depth_mean_ten) for depth in inv_depth_pyramid]
        ref_inv_depth_pyramid = [depth[(ref_frame_idx), :, :] for depth in inv_depth_norm_pyramid]
        src_inv_depth_pyramid = [depth[(src_frame_idx), :, :] for depth in inv_depth_norm_pyramid]
        photometric_cost = self.vo.compute_phtometric_loss(ref_frame_pyramid, src_frames_pyramid, ref_inv_depth_pyramid, src_inv_depth_pyramid, rot_mat_batch, trans_batch, levels=[0, 1, 2, 3], use_ssim=use_ssim, ref_expl_mask_pyramid=ref_expl_mask_pyramid, src_expl_mask_pyramid=src_expl_mask_pyramid)
        inv_depth0_pyramid = self.pyramid_func(inv_depth_norm_pyramid[0], do_detach=False)
        smoothness_cost = self.vo.multi_scale_image_aware_smoothness_cost(inv_depth0_pyramid, frames_pyramid, levels=[2, 3], type=self.smooth_term) + self.vo.multi_scale_image_aware_smoothness_cost(inv_depth_norm_pyramid, frames_pyramid, levels=[2, 3], type=self.smooth_term)
        cost = photometric_cost + lambda_S * smoothness_cost - lambda_E * expl_mask_reg_cost
        return cost, photometric_cost, smoothness_cost, ref_frame_pyramid[0], ref_inv_depth_pyramid[0] * inv_depth_mean_ten, expl_mask


class SfMLearner(nn.Module):

    def __init__(self, img_size=[128, 416], ref_frame_idx=1, lambda_S=0.5, lambda_E=0.01, use_ssim=True, smooth_term='lap', use_expl_mask=False, gpu_ids=[0]):
        super(SfMLearner, self).__init__()
        self.sfmkernel = nn.DataParallel(SfMKernel(img_size, smooth_term=smooth_term, use_expl_mask=use_expl_mask), device_ids=gpu_ids)
        self.ref_frame_idx = ref_frame_idx
        self.lambda_S = lambda_S
        self.lambda_E = lambda_E
        self.use_ssim = use_ssim
        self.use_expl_mask = use_expl_mask

    def forward(self, frames, camparams, max_lk_iter_num=10):
        cost, photometric_cost, smoothness_cost, ref_frame, ref_inv_depth, ref_expl_mask = self.sfmkernel.forward(frames, camparams, self.ref_frame_idx, self.lambda_S, self.lambda_E, use_ssim=self.use_ssim)
        return cost.mean(), photometric_cost.mean(), smoothness_cost.mean(), ref_frame, ref_inv_depth, ref_expl_mask

    def save_model(self, file_path):
        torch.save(self.cpu().sfmkernel.module.depth_net.state_dict(), file_path + '_depth_net.pth')
        torch.save(self.sfmkernel.module.pose_net.state_dict(), file_path + '_pose_net.pth')
        self

    def load_model(self, file_path):
        self.sfmkernel.module.depth_net.load_state_dict(torch.load(file_path + '_depth_net.pth'))
        self.sfmkernel.module.pose_net.load_state_dict(torch.load(file_path + '_pose_net.pth'))

    def init_weights(self):
        self.sfmkernel.module.depth_net.init_weights()

    def get_parameters(self):
        return itertools.chain(self.sfmkernel.module.depth_net.parameters(), self.sfmkernel.module.pose_net.parameters())


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Conv,
     lambda: ([], {'input_nc': 4, 'output_nc': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ConvBlock,
     lambda: ([], {'input_nc': 4, 'output_nc': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (FlipLR,
     lambda: ([], {'imW': 4, 'dim_w': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     False),
    (GradientLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (ImagePyramidLayer,
     lambda: ([], {'chan': 4, 'pyramid_layer_num': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ImageSmoothLayer,
     lambda: ([], {'chan': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (LaplacianLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (PoseExpNet,
     lambda: ([], {'bundle_size': 4}),
     lambda: ([torch.rand([4, 12, 64, 64])], {}),
     True),
    (Twist2Mat,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3])], {}),
     False),
    (UpConv,
     lambda: ([], {'input_nc': 4, 'output_nc': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_MightyChaos_LKVOLearner(_paritybench_base):
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

