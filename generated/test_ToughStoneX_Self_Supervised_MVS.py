import sys
_module = sys.modules[__name__]
del sys
argParser = _module
cleaner = _module
dataset = _module
data_io = _module
data_path = _module
dtu = _module
utils = _module
fusion = _module
arange = _module
depthfusion = _module
homography = _module
modules = _module
unsup_loss = _module
unsup_seg_loss = _module
models = _module
augmentations = _module
modules = _module
network = _module
seg_dff = _module
test = _module
train = _module
utils = _module
config = _module
datasets = _module
dtu_yao = _module
dtu_yao_eval = _module
eval = _module
eval_dense = _module
losses = _module
homography = _module
modules = _module
unsup_loss = _module
unsup_seg_loss = _module
augmentations = _module
module = _module
mvsnet = _module
seg_dff = _module
train = _module
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


from torch.utils.data import Dataset


from torchvision import transforms


import torch


import torch.nn as nn


import torch.nn.functional as F


from torch.autograd import Variable


import time


from torchvision import models


import random


import logging


import torch.backends.cudnn as cudnn


from torch.utils.data import DataLoader


import copy


import torch.nn.parallel


import torch.optim as optim


import torchvision.utils as vutils


import math


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """

    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)
        self.mask_pool = nn.AvgPool2d(3, 1)
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y, mask):
        x = x.permute(0, 3, 1, 2)
        y = y.permute(0, 3, 1, 2)
        mask = mask.permute(0, 3, 1, 2)
        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)
        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y
        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)
        SSIM_mask = self.mask_pool(mask)
        output = SSIM_mask * torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)
        return output.permute(0, 2, 3, 1)


def gradient(pred):
    D_dy = pred[:, 1:, :, :] - pred[:, :-1, :, :]
    D_dx = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    return D_dx, D_dy


def compute_reconstr_loss(warped, ref, mask, simple=True):
    if simple:
        return F.smooth_l1_loss(warped * mask, ref * mask, reduction='mean')
    else:
        alpha = 0.5
        ref_dx, ref_dy = gradient(ref * mask)
        warped_dx, warped_dy = gradient(warped * mask)
        photo_loss = F.smooth_l1_loss(warped * mask, ref * mask, reduction='mean')
        grad_loss = F.smooth_l1_loss(warped_dx, ref_dx, reduction='mean') + F.smooth_l1_loss(warped_dy, ref_dy, reduction='mean')
        return (1 - alpha) * photo_loss + alpha * grad_loss


def gradient_x(img):
    return img[:, :, :-1, :] - img[:, :, 1:, :]


def gradient_y(img):
    return img[:, :-1, :, :] - img[:, 1:, :, :]


def depth_smoothness(depth, img, lambda_wt=1):
    """Computes image-aware depth smoothness loss."""
    depth_dx = gradient_x(depth)
    depth_dy = gradient_y(depth)
    image_dx = gradient_x(img)
    image_dy = gradient_y(img)
    weights_x = torch.exp(-(lambda_wt * torch.mean(torch.abs(image_dx), 3, keepdim=True)))
    weights_y = torch.exp(-(lambda_wt * torch.mean(torch.abs(image_dy), 3, keepdim=True)))
    smoothness_x = depth_dx * weights_x
    smoothness_y = depth_dy * weights_y
    return torch.mean(torch.abs(smoothness_x)) + torch.mean(torch.abs(smoothness_y))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _cam2pixel(cam_coords, proj_c2p):
    """Transform coordinates in the camera frame to the pixel frame."""
    pcoords = torch.matmul(proj_c2p, cam_coords)
    x = pcoords[:, 0:1, :]
    y = pcoords[:, 1:2, :]
    z = pcoords[:, 2:3, :]
    x_norm = x / (z + 1e-10)
    y_norm = y / (z + 1e-10)
    pixel_coords = torch.cat([x_norm, y_norm], dim=1)
    return pixel_coords


def _meshgrid_abs(height, width):
    """Meshgrid in the absolute coordinates."""
    x_t = torch.matmul(torch.ones([height, 1]), torch.linspace(-1.0, 1.0, width).unsqueeze(1).permute(1, 0))
    y_t = torch.matmul(torch.linspace(-1.0, 1.0, height).unsqueeze(1), torch.ones([1, width]))
    x_t = (x_t + 1.0) * 0.5 * (width - 1)
    y_t = (y_t + 1.0) * 0.5 * (height - 1)
    x_t_flat = x_t.reshape(1, -1)
    y_t_flat = y_t.reshape(1, -1)
    ones = torch.ones_like(x_t_flat)
    grid = torch.cat([x_t_flat, y_t_flat, ones], dim=0)
    return grid


def _pixel2cam(depth, pixel_coords, intrinsic_mat_inv):
    """Transform coordinates in the pixel frame to the camera frame."""
    cam_coords = torch.matmul(intrinsic_mat_inv.float(), pixel_coords.float()) * depth.float()
    return cam_coords


def _bilinear_sample(im, x, y, name='bilinear_sampler'):
    """Perform bilinear sampling on im given list of x, y coordinates.
    Implements the differentiable sampling mechanism with bilinear kernel
    in https://arxiv.org/abs/1506.02025.
    x,y are tensors specifying normalized coordinates [-1, 1] to be sampled on im.
    For example, (-1, -1) in (x, y) corresponds to pixel location (0, 0) in im,
    and (1, 1) in (x, y) corresponds to the bottom right pixel in im.
    Args:
        im: Batch of images with shape [B, h, w, channels].
        x: Tensor of normalized x coordinates in [-1, 1], with shape [B, h, w, 1].
        y: Tensor of normalized y coordinates in [-1, 1], with shape [B, h, w, 1].
        name: Name scope for ops.
    Returns:
        Sampled image with shape [B, h, w, channels].
        Principled mask with shape [B, h, w, 1], dtype:float32.  A value of 1.0
        in the mask indicates that the corresponding coordinate in the sampled
        image is valid.
      """
    x = x.reshape(-1)
    y = y.reshape(-1)
    batch_size, height, width, channels = im.shape
    x, y = x.float(), y.float()
    max_y = int(height - 1)
    max_x = int(width - 1)
    x = (x + 1.0) * (width - 1.0) / 2.0
    y = (y + 1.0) * (height - 1.0) / 2.0
    x0 = torch.floor(x).int()
    x1 = x0 + 1
    y0 = torch.floor(y).int()
    y1 = y0 + 1
    mask = (x0 >= 0) & (x1 <= max_x) & (y0 >= 0) & (y0 <= max_y)
    mask = mask.float()
    x0 = torch.clamp(x0, 0, max_x)
    x1 = torch.clamp(x1, 0, max_x)
    y0 = torch.clamp(y0, 0, max_y)
    y1 = torch.clamp(y1, 0, max_y)
    dim2 = width
    dim1 = width * height
    base = torch.arange(batch_size) * dim1
    base = base.reshape(-1, 1)
    base = base.repeat(1, height * width)
    base = base.reshape(-1)
    base = base.long()
    base_y0 = base + y0.long() * dim2
    base_y1 = base + y1.long() * dim2
    idx_a = base_y0 + x0.long()
    idx_b = base_y1 + x0.long()
    idx_c = base_y0 + x1.long()
    idx_d = base_y1 + x1.long()
    im_flat = im.reshape(-1, channels).float()
    pixel_a = im_flat[idx_a]
    pixel_b = im_flat[idx_b]
    pixel_c = im_flat[idx_c]
    pixel_d = im_flat[idx_d]
    wa = (x1.float() - x) * (y1.float() - y)
    wb = (x1.float() - x) * (1.0 - (y1.float() - y))
    wc = (1.0 - (x1.float() - x)) * (y1.float() - y)
    wd = (1.0 - (x1.float() - x)) * (1.0 - (y1.float() - y))
    wa, wb, wc, wd = wa.unsqueeze(1), wb.unsqueeze(1), wc.unsqueeze(1), wd.unsqueeze(1)
    output = wa * pixel_a + wb * pixel_b + wc * pixel_c + wd * pixel_d
    output = output.reshape(batch_size, height, width, channels)
    mask = mask.reshape(batch_size, height, width, 1)
    return output, mask


def _spatial_transformer(img, coords):
    """A wrapper over binlinear_sampler(), taking absolute coords as input."""
    img_height = img.shape[1]
    img_width = img.shape[2]
    px = coords[:, :, :, :1]
    py = coords[:, :, :, 1:]
    px = px / (img_width - 1) * 2.0 - 1.0
    py = py / (img_height - 1) * 2.0 - 1.0
    output_img, mask = _bilinear_sample(img, px, py)
    return output_img, mask


def inverse_warping(img, left_cam, right_cam, depth):
    R_left = left_cam[:, 0:1, 0:3, 0:3]
    R_right = right_cam[:, 0:1, 0:3, 0:3]
    t_left = left_cam[:, 0:1, 0:3, 3:4]
    t_right = right_cam[:, 0:1, 0:3, 3:4]
    K_left = left_cam[:, 1:2, 0:3, 0:3]
    K_right = right_cam[:, 1:2, 0:3, 0:3]
    K_left = K_left.squeeze(1)
    K_left_inv = torch.inverse(K_left)
    R_left_trans = R_left.squeeze(1).permute(0, 2, 1)
    R_right_trans = R_right.squeeze(1).permute(0, 2, 1)
    R_left = R_left.squeeze(1)
    t_left = t_left.squeeze(1)
    R_right = R_right.squeeze(1)
    t_right = t_right.squeeze(1)
    R_rel = torch.matmul(R_right, R_left_trans)
    t_rel = t_right - torch.matmul(R_rel, t_left)
    batch_size = R_left.shape[0]
    filler = torch.Tensor([0.0, 0.0, 0.0, 1.0]).reshape(1, 1, 4)
    filler = filler.repeat(batch_size, 1, 1)
    transform_mat = torch.cat([R_rel, t_rel], dim=2)
    transform_mat = torch.cat([transform_mat.float(), filler.float()], dim=1)
    batch_size, img_height, img_width, _ = img.shape
    depth = depth.reshape(batch_size, 1, img_height * img_width)
    grid = _meshgrid_abs(img_height, img_width)
    grid = grid.unsqueeze(0).repeat(batch_size, 1, 1)
    cam_coords = _pixel2cam(depth, grid, K_left_inv)
    ones = torch.ones([batch_size, 1, img_height * img_width], device=device)
    cam_coords_hom = torch.cat([cam_coords, ones], dim=1)
    hom_filler = torch.Tensor([0.0, 0.0, 0.0, 1.0]).reshape(1, 1, 4)
    hom_filler = hom_filler.repeat(batch_size, 1, 1)
    intrinsic_mat_hom = torch.cat([K_left.float(), torch.zeros([batch_size, 3, 1], device=device)], dim=2)
    intrinsic_mat_hom = torch.cat([intrinsic_mat_hom, hom_filler], dim=1)
    proj_target_cam_to_source_pixel = torch.matmul(intrinsic_mat_hom, transform_mat)
    source_pixel_coords = _cam2pixel(cam_coords_hom, proj_target_cam_to_source_pixel)
    source_pixel_coords = source_pixel_coords.reshape(batch_size, 2, img_height, img_width)
    source_pixel_coords = source_pixel_coords.permute(0, 2, 3, 1)
    warped_right, mask = _spatial_transformer(img, source_pixel_coords)
    return warped_right, mask


class UnSupLoss(nn.Module):

    def __init__(self):
        super(UnSupLoss, self).__init__()
        self.ssim = SSIM()

    def forward(self, imgs, cams, depth):
        imgs = torch.unbind(imgs, 1)
        cams = torch.unbind(cams, 1)
        assert len(imgs) == len(cams), 'Different number of images and projection matrices'
        img_height, img_width = imgs[0].shape[2], imgs[0].shape[3]
        num_views = len(imgs)
        ref_img = imgs[0]
        ref_img = F.interpolate(ref_img, scale_factor=0.25, mode='bilinear')
        ref_img = ref_img.permute(0, 2, 3, 1)
        ref_cam = cams[0]
        self.reconstr_loss = 0
        self.ssim_loss = 0
        self.smooth_loss = 0
        warped_img_list = []
        mask_list = []
        reprojection_losses = []
        for view in range(1, num_views):
            view_img = imgs[view]
            view_cam = cams[view]
            view_img = F.interpolate(view_img, scale_factor=0.25, mode='bilinear')
            view_img = view_img.permute(0, 2, 3, 1)
            warped_img, mask = inverse_warping(view_img, ref_cam, view_cam, depth)
            warped_img_list.append(warped_img)
            mask_list.append(mask)
            reconstr_loss = compute_reconstr_loss(warped_img, ref_img, mask, simple=False)
            valid_mask = 1 - mask
            reprojection_losses.append(reconstr_loss + 10000.0 * valid_mask)
            if view < 3:
                self.ssim_loss += torch.mean(self.ssim(ref_img, warped_img, mask))
        self.smooth_loss += depth_smoothness(depth.unsqueeze(dim=-1), ref_img, args.smooth_lambda)
        reprojection_volume = torch.stack(reprojection_losses).permute(1, 2, 3, 4, 0)
        top_vals, top_inds = torch.topk(torch.neg(reprojection_volume), k=3, sorted=False)
        top_vals = torch.neg(top_vals)
        top_mask = top_vals < 10000.0 * torch.ones_like(top_vals, device=device)
        top_mask = top_mask.float()
        top_vals = torch.mul(top_vals, top_mask)
        self.reconstr_loss = torch.mean(torch.sum(top_vals, dim=-1))
        self.unsup_loss = 12 * self.reconstr_loss + 6 * self.ssim_loss + 0.18 * self.smooth_loss
        return self.unsup_loss


def approximation_error(V, W, H, square_root=True):
    return torch.norm(V - torch.mm(W, H))


EPSILON = 1e-07


def multiplicative_update_step(V, W, H, update_h=None, VH=None, HH=None):
    if VH is None:
        assert HH is None
        Ht = torch.t(H)
        VH = torch.mm(V, Ht)
        HH = torch.mm(H, Ht)
    WHH = torch.mm(W, HH)
    WHH[WHH == 0] = EPSILON
    W *= VH / WHH
    if update_h:
        Wt = torch.t(W)
        WV = torch.mm(Wt, V)
        WWH = torch.mm(torch.mm(Wt, W), H)
        WWH[WWH == 0] = EPSILON
        H *= WV / WWH
        VH, HH = None, None
    return W, H, VH, HH


def NMF(V, k, W=None, H=None, random_seed=None, max_iter=200, tol=0.0001, cuda=True, verbose=False):
    if verbose:
        start_time = time.time()
    scale = torch.sqrt(V.mean() / k)
    if random_seed is not None:
        if cuda:
            current_random_seed = torch.initial_seed()
            torch.manual_seed(random_seed)
        else:
            current_random_seed = torch.initial_seed()
            torch.manual_seed(random_seed)
    if W is None:
        if cuda:
            W = torch.FloatTensor(V.size(0), k).normal_()
        else:
            W = torch.randn(V.size(0), k)
        W *= scale
    update_H = True
    if H is None:
        if cuda:
            H = torch.FloatTensor(k, V.size(1)).normal_()
        else:
            H = torch.randn(k, V.size(1))
        H *= scale
    else:
        update_H = False
    if random_seed is not None:
        if cuda:
            torch.manual_seed(current_random_seed)
        else:
            torch.manual_seed(current_random_seed)
    W = torch.abs(W)
    H = torch.abs(H)
    error_at_init = approximation_error(V, W, H, square_root=True)
    previous_error = error_at_init
    VH = None
    HH = None
    for n_iter in range(max_iter):
        W, H, VH, HH = multiplicative_update_step(V, W, H, update_h=update_H, VH=VH, HH=HH)
        if tol > 0 and n_iter % 10 == 0:
            error = approximation_error(V, W, H, square_root=True)
            if (previous_error - error) / error_at_init < tol:
                break
            previous_error = error
    if verbose:
        None
    return W, H


class SegDFF(nn.Module):

    def __init__(self, K, max_iter=50):
        super(SegDFF, self).__init__()
        self.K = K
        self.max_iter = max_iter
        self.net = models.vgg19(pretrained=True)
        del self.net.features._modules['36']

    def forward(self, imgs):
        batch_size = imgs.size(0)
        heatmaps = []
        for b in range(batch_size):
            imgs_b = imgs[b]
            with torch.no_grad():
                imgs_b = F.interpolate(imgs_b, size=(224, 224), mode='bilinear', align_corners=False)
                features = self.net.features(imgs_b)
                flat_features = features.permute(0, 2, 3, 1).contiguous().view(-1, features.size(1))
                W, _ = NMF(flat_features, self.K, random_seed=1, cuda=True, max_iter=self.max_iter, verbose=False)
                isnan = torch.sum(torch.isnan(W).float())
                while isnan > 0:
                    None
                    W, _ = NMF(flat_features, self.K, random_seed=random.randint(0, 255), cuda=True, max_iter=self.max_iter, verbose=False)
                    isnan = torch.sum(torch.isnan(W).float())
                heatmap = W.view(features.size(0), features.size(2), features.size(3), self.K)
                heatmaps.append(heatmap)
        heatmaps = torch.stack(heatmaps, dim=0)
        heatmaps.requires_grad = False
        return heatmaps


def compute_seg_loss(warped_seg, ref_seg, mask):
    mask = mask.repeat(1, 1, 1, warped_seg.size(3))
    warped_seg_filtered = warped_seg[mask > 0.5]
    ref_seg_filtered = ref_seg[mask > 0.5]
    warped_seg_filtered_flatten = warped_seg_filtered.contiguous().view(-1, warped_seg.size(3))
    ref_seg_filtered_flatten = ref_seg_filtered.contiguous().view(-1, ref_seg.size(3))
    ref_seg_filtered_flatten = torch.argmax(ref_seg_filtered_flatten, dim=1)
    loss = F.cross_entropy(warped_seg_filtered_flatten, ref_seg_filtered_flatten, size_average=True)
    return loss


class UnSupSegLoss(nn.Module):

    def __init__(self, args):
        super(UnSupSegLoss, self).__init__()
        self.seg_model = SegDFF(K=args.seg_clusters, max_iter=50)

    def forward(self, imgs, cams, depth):
        seg_maps = self.seg_model(imgs)
        seg_maps = seg_maps.permute(0, 1, 4, 2, 3)
        seg_maps = torch.unbind(seg_maps, 1)
        cams = torch.unbind(cams, 1)
        height, width = depth.size(1), depth.size(2)
        num_views = len(seg_maps)
        ref_seg = seg_maps[0]
        ref_seg = F.interpolate(ref_seg, size=(height, width), mode='bilinear')
        ref_seg = ref_seg.permute(0, 2, 3, 1)
        ref_cam = cams[0]
        warped_seg_list = []
        mask_list = []
        reprojection_losses = []
        view_segs = []
        for view in range(1, num_views):
            view_seg = seg_maps[view]
            view_seg = F.interpolate(view_seg, size=(height, width), mode='bilinear')
            view_seg = view_seg.permute(0, 2, 3, 1)
            view_cam = cams[view]
            view_segs.append(view_seg)
            warped_seg, mask = inverse_warping(view_seg, ref_cam, view_cam, depth)
            warped_seg_list.append(warped_seg)
            mask_list.append(mask)
            reprojection_losses.append(compute_seg_loss(warped_seg, ref_seg, mask))
        reproj_seg_loss = sum(reprojection_losses) * 1.0
        view_segs = torch.stack(view_segs, dim=1)
        return reproj_seg_loss, ref_seg, view_segs


class UnSupSegLossAcc(nn.Module):

    def __init__(self, args):
        super(UnSupSegLossAcc, self).__init__()

    def forward(self, seg_maps, cams, depth):
        seg_maps = torch.unbind(seg_maps, 1)
        cams = torch.unbind(cams, 1)
        height, width = depth.size(1), depth.size(2)
        num_views = len(seg_maps)
        ref_seg = seg_maps[0]
        ref_seg = F.interpolate(ref_seg, size=(height, width), mode='bilinear')
        ref_seg = ref_seg.permute(0, 2, 3, 1)
        ref_cam = cams[0]
        warped_seg_list = []
        mask_list = []
        reprojection_losses = []
        view_segs = []
        for view in range(1, num_views):
            view_seg = seg_maps[view]
            view_seg = F.interpolate(view_seg, size=(height, width), mode='bilinear')
            view_seg = view_seg.permute(0, 2, 3, 1)
            view_cam = cams[view]
            view_segs.append(view_seg)
            warped_seg, mask = inverse_warping(view_seg, ref_cam, view_cam, depth)
            warped_seg_list.append(warped_seg)
            mask_list.append(mask)
            reprojection_losses.append(compute_seg_loss(warped_seg, ref_seg, mask))
        reproj_seg_loss = sum(reprojection_losses) * 1.0
        view_segs = torch.stack(view_segs, dim=1)
        return reproj_seg_loss, ref_seg, view_segs


class ConvBnReLU(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class ConvBn(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBn, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class ConvBnReLU3D(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class ConvBn3D(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBn3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = ConvBnReLU(in_channels, out_channels, kernel_size=3, stride=stride, pad=1)
        self.conv2 = ConvBn(out_channels, out_channels, kernel_size=3, stride=1, pad=1)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        return out


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, bias=True), nn.LeakyReLU(0.1))


class FeaturePyramid(nn.Module):

    def __init__(self):
        super(FeaturePyramid, self).__init__()
        self.conv0aa = conv(3, 64, kernel_size=3, stride=1)
        self.conv0ba = conv(64, 64, kernel_size=3, stride=1)
        self.conv0bb = conv(64, 64, kernel_size=3, stride=1)
        self.conv0bc = conv(64, 32, kernel_size=3, stride=1)
        self.conv0bd = conv(32, 32, kernel_size=3, stride=1)
        self.conv0be = conv(32, 32, kernel_size=3, stride=1)
        self.conv0bf = conv(32, 16, kernel_size=3, stride=1)
        self.conv0bg = conv(16, 16, kernel_size=3, stride=1)
        self.conv0bh = conv(16, 16, kernel_size=3, stride=1)

    def forward(self, img, scales=5):
        fp = []
        f = self.conv0aa(img)
        f = self.conv0bh(self.conv0bg(self.conv0bf(self.conv0be(self.conv0bd(self.conv0bc(self.conv0bb(self.conv0ba(f))))))))
        fp.append(f)
        for scale in range(scales - 1):
            img = nn.functional.interpolate(img, scale_factor=0.5, mode='bilinear', align_corners=None).detach()
            f = self.conv0aa(img)
            f = self.conv0bh(self.conv0bg(self.conv0bf(self.conv0be(self.conv0bd(self.conv0bc(self.conv0bb(self.conv0ba(f))))))))
            fp.append(f)
        return fp


class CostRegNet(nn.Module):

    def __init__(self):
        super(CostRegNet, self).__init__()
        self.conv0 = ConvBnReLU3D(32, 8)
        self.conv1 = ConvBnReLU3D(8, 16, stride=2)
        self.conv2 = ConvBnReLU3D(16, 16)
        self.conv3 = ConvBnReLU3D(16, 32, stride=2)
        self.conv4 = ConvBnReLU3D(32, 32)
        self.conv5 = ConvBnReLU3D(32, 64, stride=2)
        self.conv6 = ConvBnReLU3D(64, 64)
        self.conv7 = nn.Sequential(nn.ConvTranspose3d(64, 32, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False), nn.BatchNorm3d(32), nn.ReLU(inplace=True))
        self.conv9 = nn.Sequential(nn.ConvTranspose3d(32, 16, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False), nn.BatchNorm3d(16), nn.ReLU(inplace=True))
        self.conv11 = nn.Sequential(nn.ConvTranspose3d(16, 8, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False), nn.BatchNorm3d(8), nn.ReLU(inplace=True))
        self.prob = nn.Conv3d(8, 1, 3, stride=1, padding=1)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        x = conv2 + self.conv9(x)
        x = conv0 + self.conv11(x)
        x = self.prob(x)
        return x


def calDepthHypo(netArgs, ref_depths, ref_intrinsics, src_intrinsics, ref_extrinsics, src_extrinsics, depth_min, depth_max, level):
    nhypothesis_init = 48
    d = 4
    pixel_interval = 1
    nBatch = ref_depths.shape[0]
    height = ref_depths.shape[1]
    width = ref_depths.shape[2]
    with torch.no_grad():
        ref_depths = ref_depths
        ref_intrinsics = ref_intrinsics.double()
        src_intrinsics = src_intrinsics.squeeze(1).double()
        ref_extrinsics = ref_extrinsics.double()
        src_extrinsics = src_extrinsics.squeeze(1).double()
        interval_maps = []
        depth_hypos = ref_depths.unsqueeze(1).repeat(1, d * 2, 1, 1).double()
        for batch in range(nBatch):
            xx, yy = torch.meshgrid([torch.arange(0, width), torch.arange(0, height)])
            xxx = xx.reshape([-1]).double()
            yyy = yy.reshape([-1]).double()
            X = torch.stack([xxx, yyy, torch.ones_like(xxx)], dim=0)
            D1 = torch.transpose(ref_depths[batch, :, :], 0, 1).reshape([-1]).double()
            D2 = D1 + 1
            X1 = X * D1
            X2 = X * D2
            ray1 = torch.matmul(torch.inverse(ref_intrinsics[batch]), X1)
            ray2 = torch.matmul(torch.inverse(ref_intrinsics[batch]), X2)
            X1 = torch.cat([ray1, torch.ones_like(xxx).unsqueeze(0).double()], dim=0)
            X1 = torch.matmul(torch.inverse(ref_extrinsics[batch]), X1)
            X2 = torch.cat([ray2, torch.ones_like(xxx).unsqueeze(0).double()], dim=0)
            X2 = torch.matmul(torch.inverse(ref_extrinsics[batch]), X2)
            X1 = torch.matmul(src_extrinsics[batch][0], X1)
            X2 = torch.matmul(src_extrinsics[batch][0], X2)
            X1 = X1[:3]
            X1 = torch.matmul(src_intrinsics[batch][0], X1)
            X1_d = X1[2].clone()
            X1 /= X1_d
            X2 = X2[:3]
            X2 = torch.matmul(src_intrinsics[batch][0], X2)
            X2_d = X2[2].clone()
            X2 /= X2_d
            k = (X2[1] - X1[1]) / (X2[0] - X1[0])
            b = X1[1] - k * X1[0]
            theta = torch.atan(k)
            X3 = X1 + torch.stack([torch.cos(theta) * pixel_interval, torch.sin(theta) * pixel_interval, torch.zeros_like(X1[2, :])], dim=0)
            A = torch.matmul(ref_intrinsics[batch], ref_extrinsics[batch][:3, :3])
            tmp = torch.matmul(src_intrinsics[batch][0], src_extrinsics[batch][0, :3, :3])
            A = torch.matmul(A, torch.inverse(tmp))
            tmp1 = X1_d * torch.matmul(A, X1)
            tmp2 = torch.matmul(A, X3)
            M1 = torch.cat([X.t().unsqueeze(2), tmp2.t().unsqueeze(2)], dim=2)[:, 1:, :]
            M2 = tmp1.t()[:, 1:]
            tmp1 = torch.inverse(M1)
            tmp2 = M2.unsqueeze(2)
            ans = torch.matmul(tmp1, tmp2)
            delta_d = ans[:, 0, 0]
            interval_maps = torch.abs(delta_d).mean().repeat(ref_depths.shape[2], ref_depths.shape[1]).t()
            for depth_level in range(-d, d):
                depth_hypos[batch, depth_level + d, :, :] += depth_level * interval_maps
        return depth_hypos.float()


def calSweepingDepthHypo(ref_in, src_in, ref_ex, src_ex, depth_min, depth_max, nhypothesis_init=48):
    batchSize = ref_in.shape[0]
    depth_range = depth_max[0] - depth_min[0]
    depth_interval_mean = depth_range / (nhypothesis_init - 1)
    assert nhypothesis_init % 2 == 0
    depth_hypos = torch.range(depth_min[0], depth_max[0], depth_interval_mean).unsqueeze(0)
    for b in range(1, batchSize):
        depth_hypos = torch.cat((depth_hypos, torch.range(depth_min[0], depth_max[0], depth_interval_mean).unsqueeze(0)), 0)
    return depth_hypos


def conditionIntrinsics(intrinsics, img_shape, fp_shapes):
    down_ratios = []
    for fp_shape in fp_shapes:
        down_ratios.append(img_shape[2] / fp_shape[2])
    intrinsics_out = []
    for down_ratio in down_ratios:
        intrinsics_tmp = intrinsics.clone()
        intrinsics_tmp[:, :2, :] = intrinsics_tmp[:, :2, :] / down_ratio
        intrinsics_out.append(intrinsics_tmp)
    return torch.stack(intrinsics_out).permute(1, 0, 2, 3)


def depth_regression(p, depth_values):
    depth_values = depth_values.view(*depth_values.shape, 1, 1)
    depth = torch.sum(p * depth_values, 1)
    return depth


def depth_regression_refine(prob_volume, depth_hypothesis):
    return torch.sum(prob_volume * depth_hypothesis, 1)


def homo_warping(src_fea, src_proj, ref_proj, depth_values):
    batch, channels = src_fea.shape[0], src_fea.shape[1]
    num_depth = depth_values.shape[1]
    height, width = src_fea.shape[2], src_fea.shape[3]
    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]
        trans = proj[:, :3, 3:4]
        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device), torch.arange(0, width, dtype=torch.float32, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x)))
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)
        rot_xyz = torch.matmul(rot, xyz)
        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_values.view(batch, 1, num_depth, 1)
        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)
        proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]
        proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
        proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)
        grid = proj_xy
    warped_src_fea = F.grid_sample(src_fea, grid.view(batch, num_depth * height, width, 2), mode='bilinear', padding_mode='zeros')
    warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)
    return warped_src_fea


def proj_cost(settings, ref_feature, src_feature, level, ref_in, src_in, ref_ex, src_ex, depth_hypos):
    batch, channels = ref_feature.shape[0], ref_feature.shape[1]
    num_depth = depth_hypos.shape[1]
    height, width = ref_feature.shape[2], ref_feature.shape[3]
    nSrc = len(src_feature)
    volume_sum = ref_feature.unsqueeze(2).repeat(1, 1, num_depth, 1, 1)
    volume_sq_sum = volume_sum.pow_(2)
    for src in range(settings.nsrc):
        with torch.no_grad():
            src_proj = torch.matmul(src_in[:, src, :, :], src_ex[:, src, 0:3, :])
            ref_proj = torch.matmul(ref_in, ref_ex[:, 0:3, :])
            last = torch.tensor([[[0, 0, 0, 1.0]]]).repeat(len(src_in), 1, 1)
            src_proj = torch.cat((src_proj, last), 1)
            ref_proj = torch.cat((ref_proj, last), 1)
            proj = torch.matmul(src_proj, torch.inverse(ref_proj))
            rot = proj[:, :3, :3]
            trans = proj[:, :3, 3:4]
            y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=ref_feature.device), torch.arange(0, width, dtype=torch.float32, device=ref_feature.device)])
            y, x = y.contiguous(), x.contiguous()
            y, x = y.view(height * width), x.view(height * width)
            xyz = torch.stack((x, y, torch.ones_like(x)))
            xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)
            rot_xyz = torch.matmul(rot, xyz)
            rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_hypos.view(batch, 1, num_depth, height * width)
            proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)
            proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]
            proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
            proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
            proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)
            grid = proj_xy
        warped_src_fea = F.grid_sample(src_feature[src][level], grid.view(batch, num_depth * height, width, 2), mode='bilinear', padding_mode='zeros')
        warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)
        volume_sum = volume_sum + warped_src_fea
        volume_sq_sum = volume_sq_sum + warped_src_fea.pow_(2)
    cost_volume = volume_sq_sum.div_(settings.nsrc + 1).sub_(volume_sum.div_(settings.nsrc + 1).pow_(2))
    if settings.mode == 'test':
        del volume_sum
        del volume_sq_sum
        torch.cuda.empty_cache()
    return cost_volume


class CVPMVSNet(nn.Module):

    def __init__(self, args):
        super(CVPMVSNet, self).__init__()
        self.featurePyramid = FeaturePyramid()
        self.cost_reg_refine = CostRegNet()
        self.args = args

    def forward(self, ref_img, src_imgs, ref_in, src_in, ref_ex, src_ex, depth_min, depth_max):
        depth_est_list = []
        output = {}
        ref_feature_pyramid = self.featurePyramid(ref_img, self.args.nscale)
        src_feature_pyramids = []
        for i in range(self.args.nsrc):
            src_feature_pyramids.append(self.featurePyramid(src_imgs[:, i, :, :, :], self.args.nscale))
        ref_in_multiscales = conditionIntrinsics(ref_in, ref_img.shape, [feature.shape for feature in ref_feature_pyramid])
        src_in_multiscales = []
        for i in range(self.args.nsrc):
            src_in_multiscales.append(conditionIntrinsics(src_in[:, i], ref_img.shape, [feature.shape for feature in src_feature_pyramids[i]]))
        src_in_multiscales = torch.stack(src_in_multiscales).permute(1, 0, 2, 3, 4)
        depth_hypos = calSweepingDepthHypo(ref_in_multiscales[:, -1], src_in_multiscales[:, 0, -1], ref_ex, src_ex, depth_min, depth_max)
        ref_volume = ref_feature_pyramid[-1].unsqueeze(2).repeat(1, 1, len(depth_hypos[0]), 1, 1)
        volume_sum = ref_volume
        volume_sq_sum = ref_volume.pow_(2)
        if self.args.mode == 'test' or self.args.mode == 'eval':
            del ref_volume
        for src_idx in range(self.args.nsrc):
            warped_volume = homo_warping(src_feature_pyramids[src_idx][-1], ref_in_multiscales[:, -1], src_in_multiscales[:, src_idx, -1, :, :], ref_ex, src_ex[:, src_idx], depth_hypos)
            if self.args.mode == 'train':
                volume_sum = volume_sum + warped_volume
                volume_sq_sum = volume_sq_sum + warped_volume ** 2
            elif self.args.mode == 'test' or self.args.mode == 'eval':
                volume_sum = volume_sum + warped_volume
                volume_sq_sum = volume_sq_sum + warped_volume ** 2
                del warped_volume
            else:
                None
        cost_volume = volume_sq_sum.div_(self.args.nsrc + 1).sub_(volume_sum.div_(self.args.nsrc + 1).pow_(2))
        if self.args.mode == 'test' or self.args.mode == 'eval':
            del volume_sum
            del volume_sq_sum
        cost_reg = self.cost_reg_refine(cost_volume)
        prob_volume = F.softmax(cost_reg, dim=1)
        depth = depth_regression(prob_volume, depth_values=depth_hypos)
        depth_est_list.append(depth)
        for level in range(self.args.nscale - 2, -1, -1):
            depth_up = nn.functional.interpolate(depth[None, :], size=None, scale_factor=2, mode='bilinear', align_corners=None)
            depth_up = depth_up.squeeze(0)
            depth_hypos = calDepthHypo(self.args, depth_up, ref_in_multiscales[:, level, :, :], src_in_multiscales[:, :, level, :, :], ref_ex, src_ex, depth_min, depth_max, level)
            cost_volume = proj_cost(self.args, ref_feature_pyramid[level], src_feature_pyramids, level, ref_in_multiscales[:, level, :, :], src_in_multiscales[:, :, level, :, :], ref_ex, src_ex[:, :], depth_hypos)
            cost_reg2 = self.cost_reg_refine(cost_volume)
            if self.args.mode == 'test' or self.args.mode == 'eval':
                del cost_volume
            prob_volume = F.softmax(cost_reg2, dim=1)
            if self.args.mode == 'test' or self.args.mode == 'eval':
                del cost_reg2
            depth = depth_regression_refine(prob_volume, depth_hypos)
            depth_est_list.append(depth)
        with torch.no_grad():
            num_depth = prob_volume.shape[1]
            prob_volume_sum4 = 4 * F.avg_pool3d(F.pad(prob_volume.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)), (4, 1, 1), stride=1, padding=0).squeeze(1)
            depth_index = depth_regression(prob_volume, depth_values=torch.arange(num_depth, device=prob_volume.device, dtype=torch.float)).long()
            prob_confidence = torch.gather(prob_volume_sum4, 1, depth_index.unsqueeze(1)).squeeze(1)
        if self.args.mode == 'test' or self.args.mode == 'eval':
            del prob_volume
        depth_est_list.reverse()
        output['depth_est_list'] = depth_est_list
        output['prob_confidence'] = prob_confidence
        return output


class ColorJitter(transforms.ColorJitter):

    def __call__(self, imgs):
        transform = self.get_params(self.brightness, self.contrast, self.saturation, self.hue)
        return [transform(im) for im in imgs]


class RandomGamma:

    def __init__(self, min_gamma=0.7, max_gamma=1.5, clip_image=False):
        self._min_gamma = min_gamma
        self._max_gamma = max_gamma
        self._clip_image = clip_image

    @staticmethod
    def get_params(min_gamma, max_gamma):
        return np.random.uniform(min_gamma, max_gamma)

    @staticmethod
    def adjust_gamma(image, gamma, clip_image):
        adjusted = torch.pow(image, gamma)
        if clip_image:
            adjusted.clamp_(0.0, 1.0)
        return adjusted

    def __call__(self, imgs):
        res = []
        for im in imgs:
            gamma = self.get_params(self._min_gamma, self._max_gamma)
            res.append(self.adjust_gamma(im, gamma, self._clip_image))
        return res


class ToPILImage(transforms.ToPILImage):

    def __call__(self, imgs):
        return [super(ToPILImage, self).__call__(im) for im in imgs]


class ToTensor(transforms.ToTensor):

    def __call__(self, imgs):
        return [super(ToTensor, self).__call__(im) for im in imgs]


def get_transform():
    transform = []
    transform.append(ToPILImage())
    transform.append(ColorJitter(brightness=0.5, contrast=0.5, saturation=0, hue=0))
    transform.append(ToTensor())
    transform.append(RandomGamma(min_gamma=0.7, max_gamma=2.0, clip_image=True))
    return transforms.Compose(transform)


def random_image_mask(img, filter_size):
    """

    :param img: [B x 3 x H x W]
    :param crop_size:
    :return:
    """
    fh, fw = filter_size
    _, _, h, w = img.size()
    if fh == h and fw == w:
        return img, None
    x = np.random.randint(0, w - fw)
    y = np.random.randint(0, h - fh)
    filter_mask = torch.ones_like(img)
    filter_mask[:, :, y:y + fh, x:x + fw] = 0.0
    img = img * filter_mask
    return img, filter_mask


class Augmentor(nn.Module):

    def __init__(self):
        super(Augmentor, self).__init__()
        self.transform = get_transform()

    def forward(self, imgs):
        imgs = [torch.stack(self.transform(im_b), dim=0) for im_b in imgs]
        imgs = torch.stack(imgs, dim=0)
        ref_img = imgs[:, 0]
        h, w = ref_img.size(2), ref_img.size(3)
        ref_img, filter_mask = random_image_mask(ref_img, filter_size=(h // 4, w // 4))
        imgs[:, 0] = ref_img
        return imgs, filter_mask


class Hourglass3d(nn.Module):

    def __init__(self, channels):
        super(Hourglass3d, self).__init__()
        self.conv1a = ConvBnReLU3D(channels, channels * 2, kernel_size=3, stride=2, pad=1)
        self.conv1b = ConvBnReLU3D(channels * 2, channels * 2, kernel_size=3, stride=1, pad=1)
        self.conv2a = ConvBnReLU3D(channels * 2, channels * 4, kernel_size=3, stride=2, pad=1)
        self.conv2b = ConvBnReLU3D(channels * 4, channels * 4, kernel_size=3, stride=1, pad=1)
        self.dconv2 = nn.Sequential(nn.ConvTranspose3d(channels * 4, channels * 2, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False), nn.BatchNorm3d(channels * 2))
        self.dconv1 = nn.Sequential(nn.ConvTranspose3d(channels * 2, channels, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False), nn.BatchNorm3d(channels))
        self.redir1 = ConvBn3D(channels, channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = ConvBn3D(channels * 2, channels * 2, kernel_size=1, stride=1, pad=0)

    def forward(self, x):
        conv1 = self.conv1b(self.conv1a(x))
        conv2 = self.conv2b(self.conv2a(conv1))
        dconv2 = F.relu(self.dconv2(conv2) + self.redir2(conv1), inplace=True)
        dconv1 = F.relu(self.dconv1(dconv2) + self.redir1(x), inplace=True)
        return dconv1


class FeatureNet(nn.Module):

    def __init__(self):
        super(FeatureNet, self).__init__()
        self.inplanes = 32
        self.conv0 = ConvBnReLU(3, 8, 3, 1, 1)
        self.conv1 = ConvBnReLU(8, 8, 3, 1, 1)
        self.conv2 = ConvBnReLU(8, 16, 5, 2, 2)
        self.conv3 = ConvBnReLU(16, 16, 3, 1, 1)
        self.conv4 = ConvBnReLU(16, 16, 3, 1, 1)
        self.conv5 = ConvBnReLU(16, 32, 5, 2, 2)
        self.conv6 = ConvBnReLU(32, 32, 3, 1, 1)
        self.feature = nn.Conv2d(32, 32, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(self.conv0(x))
        x = self.conv4(self.conv3(self.conv2(x)))
        x = self.feature(self.conv6(self.conv5(x)))
        return x


class RefineNet(nn.Module):

    def __init__(self):
        super(RefineNet, self).__init__()
        self.conv1 = ConvBnReLU(4, 32)
        self.conv2 = ConvBnReLU(32, 32)
        self.conv3 = ConvBnReLU(32, 32)
        self.res = ConvBnReLU(32, 1)

    def forward(self, img, depth_init):
        img = F.interpolate(img, scale_factor=0.25, mode='bilinear')
        depth_init = depth_init.unsqueeze(dim=1)
        concat = torch.cat((img, depth_init), dim=1)
        depth_residual = self.res(self.conv3(self.conv2(self.conv1(concat))))
        depth_refined = depth_init + depth_residual
        depth_refined = depth_refined.squeeze(dim=1)
        return depth_refined


class MVSNet(nn.Module):

    def __init__(self, refine=True):
        super(MVSNet, self).__init__()
        self.refine = refine
        self.feature = FeatureNet()
        self.cost_regularization = CostRegNet()
        if self.refine:
            self.refine_network = RefineNet()

    def forward(self, imgs, proj_matrices, depth_values):
        imgs = torch.unbind(imgs, 1)
        proj_matrices = torch.unbind(proj_matrices, 1)
        assert len(imgs) == len(proj_matrices), 'Different number of images and projection matrices'
        img_height, img_width = imgs[0].shape[2], imgs[0].shape[3]
        num_depth = depth_values.shape[1]
        num_views = len(imgs)
        features = [self.feature(img) for img in imgs]
        ref_feature, src_features = features[0], features[1:]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]
        ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, num_depth, 1, 1)
        volume_sum = ref_volume
        volume_sq_sum = ref_volume ** 2
        del ref_volume
        for src_fea, src_proj in zip(src_features, src_projs):
            warped_volume = homo_warping(src_fea, src_proj, ref_proj, depth_values)
            if self.training:
                volume_sum = volume_sum + warped_volume
                volume_sq_sum = volume_sq_sum + warped_volume ** 2
            else:
                volume_sum += warped_volume
                volume_sq_sum += warped_volume.pow_(2)
            del warped_volume
        volume_variance = volume_sq_sum.div_(num_views).sub_(volume_sum.div_(num_views).pow_(2))
        cost_reg = self.cost_regularization(volume_variance)
        cost_reg = cost_reg.squeeze(1)
        prob_volume = F.softmax(cost_reg, dim=1)
        depth = depth_regression(prob_volume, depth_values=depth_values)
        with torch.no_grad():
            prob_volume_sum4 = 4 * F.avg_pool3d(F.pad(prob_volume.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)), (4, 1, 1), stride=1, padding=0).squeeze(1)
            depth_index = depth_regression(prob_volume, depth_values=torch.arange(num_depth, device=prob_volume.device, dtype=torch.float)).long()
            photometric_confidence = torch.gather(prob_volume_sum4, 1, depth_index.unsqueeze(1)).squeeze(1)
        if not self.refine:
            return {'depth': depth, 'photometric_confidence': photometric_confidence}
        else:
            refined_depth = self.refine_network(imgs[0], depth)
            return {'depth': refined_depth, 'photometric_confidence': photometric_confidence}


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvBn,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvBn3D,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (ConvBnReLU,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvBnReLU3D,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (FeatureNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (FeaturePyramid,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (Hourglass3d,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (SSIM,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_ToughStoneX_Self_Supervised_MVS(_paritybench_base):
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

