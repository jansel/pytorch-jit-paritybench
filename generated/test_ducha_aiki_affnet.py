import sys
_module = sys.modules[__name__]
del sys
HandCraftedModules = _module
HardNet = _module
LAF = _module
Losses = _module
OnePassSIR = _module
ReprojectionStuff = _module
SparseImgRepresenter = _module
Utils = _module
architectures = _module
augmentation = _module
dataset = _module
Losses = _module
Utils = _module
hesaffBaum = _module
optimization_script = _module
pytorch_sift = _module
HardNet = _module
Losses = _module
NMS = _module
extract_geomOriTh = _module
extract_geom_and_desc_upisup = _module
extract_geom_and_desc_upisupTh = _module
hesaffBaum = _module
hesaffnet = _module
LAF = _module
Utils = _module
architectures = _module
detect_affine_shape = _module
Losses = _module
Utils = _module
gen_ds = _module
pytorch_sift = _module
train_AffNet_test_on_graffity = _module
train_OriNet_test_on_graffity = _module

from _paritybench_helpers import _mock_config
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch


import torch.nn as nn


import torch.nn.functional as F


from torch.autograd import Variable


import math


import numpy as np


import torch.backends.cudnn as cudnn


from copy import deepcopy


from scipy.spatial.distance import cdist


from numpy.linalg import inv


from scipy.linalg import schur


from scipy.linalg import sqrtm


from torch.autograd import Variable as V


import torch.nn.init


import torch.optim as optim


import torch.utils.data as data


import random


import copy


import torch.nn


from torch import nn


import scipy.io as sio


class ScalePyramid(nn.Module):

    def __init__(self, nLevels=3, init_sigma=1.6, border=5):
        super(ScalePyramid, self).__init__()
        self.nLevels = nLevels
        self.init_sigma = init_sigma
        self.sigmaStep = 2 ** (1.0 / float(self.nLevels))
        self.b = border
        self.minSize = 2 * self.b + 2 + 1
        return

    def forward(self, x):
        pixelDistance = 1.0
        curSigma = 0.5
        if self.init_sigma > curSigma:
            sigma = np.sqrt(self.init_sigma ** 2 - curSigma ** 2)
            curSigma = self.init_sigma
            curr = GaussianBlur(sigma=sigma)(x)
        else:
            curr = x
        sigmas = [[curSigma]]
        pixel_dists = [[1.0]]
        pyr = [[curr]]
        j = 0
        while True:
            curr = pyr[-1][0]
            for i in range(1, self.nLevels + 2):
                sigma = curSigma * np.sqrt(self.sigmaStep * self.sigmaStep -
                    1.0)
                curr = GaussianBlur(sigma=sigma)(curr)
                curSigma *= self.sigmaStep
                pyr[j].append(curr)
                sigmas[j].append(curSigma)
                pixel_dists[j].append(pixelDistance)
                if i == self.nLevels:
                    nextOctaveFirstLevel = F.avg_pool2d(curr, kernel_size=1,
                        stride=2, padding=0)
            pixelDistance = pixelDistance * 2.0
            curSigma = self.init_sigma
            if nextOctaveFirstLevel[(0), (0), :, :].size(0
                ) <= self.minSize or nextOctaveFirstLevel[(0), (0), :, :].size(
                1) <= self.minSize:
                break
            pyr.append([nextOctaveFirstLevel])
            sigmas.append([curSigma])
            pixel_dists.append([pixelDistance])
            j += 1
        return pyr, sigmas, pixel_dists


class HessianResp(nn.Module):

    def __init__(self):
        super(HessianResp, self).__init__()
        self.gx = nn.Conv2d(1, 1, kernel_size=(1, 3), bias=False)
        self.gx.weight.data = torch.from_numpy(np.array([[[[0.5, 0, -0.5]]]
            ], dtype=np.float32))
        self.gy = nn.Conv2d(1, 1, kernel_size=(3, 1), bias=False)
        self.gy.weight.data = torch.from_numpy(np.array([[[[0.5], [0], [-
            0.5]]]], dtype=np.float32))
        self.gxx = nn.Conv2d(1, 1, kernel_size=(1, 3), bias=False)
        self.gxx.weight.data = torch.from_numpy(np.array([[[[1.0, -2.0, 1.0
            ]]]], dtype=np.float32))
        self.gyy = nn.Conv2d(1, 1, kernel_size=(3, 1), bias=False)
        self.gyy.weight.data = torch.from_numpy(np.array([[[[1.0], [-2.0],
            [1.0]]]], dtype=np.float32))
        return

    def forward(self, x, scale):
        gxx = self.gxx(F.pad(x, (1, 1, 0, 0), 'replicate'))
        gyy = self.gyy(F.pad(x, (0, 0, 1, 1), 'replicate'))
        gxy = self.gy(F.pad(self.gx(F.pad(x, (1, 1, 0, 0), 'replicate')), (
            0, 0, 1, 1), 'replicate'))
        return torch.abs(gxx * gyy - gxy * gxy) * scale ** 4


def abc2A(a, b, c, normalize=False):
    A1_ell = torch.cat([a.view(-1, 1, 1), b.view(-1, 1, 1)], dim=2)
    A2_ell = torch.cat([b.view(-1, 1, 1), c.view(-1, 1, 1)], dim=2)
    return torch.cat([A1_ell, A2_ell], dim=1)


def CircularGaussKernel(kernlen=None, circ_zeros=False, sigma=None, norm=True):
    assert kernlen is not None or sigma is not None
    if kernlen is None:
        kernlen = int(2.0 * 3.0 * sigma + 1.0)
        if kernlen % 2 == 0:
            kernlen = kernlen + 1
        halfSize = kernlen / 2
    halfSize = kernlen / 2
    r2 = float(halfSize * halfSize)
    if sigma is None:
        sigma2 = 0.9 * r2
        sigma = np.sqrt(sigma2)
    else:
        sigma2 = 2.0 * sigma * sigma
    x = np.linspace(-halfSize, halfSize, kernlen)
    xv, yv = np.meshgrid(x, x, sparse=False, indexing='xy')
    distsq = xv ** 2 + yv ** 2
    kernel = np.exp(-(distsq / sigma2))
    if circ_zeros:
        kernel *= (distsq <= r2).astype(np.float32)
    if norm:
        kernel /= np.sum(kernel)
    return kernel


def rectifyAffineTransformationUpIsUp(A):
    det = torch.sqrt(torch.abs(A[:, (0), (0)] * A[:, (1), (1)] - A[:, (1),
        (0)] * A[:, (0), (1)] + 1e-10))
    b2a2 = torch.sqrt(A[:, (0), (1)] * A[:, (0), (1)] + A[:, (0), (0)] * A[
        :, (0), (0)])
    A1_ell = torch.cat([(b2a2 / det).contiguous().view(-1, 1, 1), 0 * det.
        view(-1, 1, 1)], dim=2)
    A2_ell = torch.cat([((A[:, (1), (1)] * A[:, (0), (1)] + A[:, (1), (0)] *
        A[:, (0), (0)]) / (b2a2 * det)).contiguous().view(-1, 1, 1), (det /
        b2a2).contiguous().view(-1, 1, 1)], dim=2)
    return torch.cat([A1_ell, A2_ell], dim=1)


class AffineShapeEstimator(nn.Module):

    def __init__(self, threshold=0.001, patch_size=19):
        super(AffineShapeEstimator, self).__init__()
        self.threshold = threshold
        self.PS = patch_size
        self.gx = nn.Conv2d(1, 1, kernel_size=(1, 3), bias=False)
        self.gx.weight.data = torch.from_numpy(np.array([[[[-1, 0, 1]]]],
            dtype=np.float32))
        self.gy = nn.Conv2d(1, 1, kernel_size=(3, 1), bias=False)
        self.gy.weight.data = torch.from_numpy(np.array([[[[-1], [0], [1]]]
            ], dtype=np.float32))
        self.gk = torch.from_numpy(CircularGaussKernel(kernlen=self.PS,
            sigma=self.PS / 2 / 3.0).astype(np.float32))
        self.gk = Variable(self.gk, requires_grad=False)
        return

    def invSqrt(self, a, b, c):
        eps = 1e-12
        mask = (b != 0).float()
        r1 = mask * (c - a) / (2.0 * b + eps)
        t1 = torch.sign(r1) / (torch.abs(r1) + torch.sqrt(1.0 + r1 * r1))
        r = 1.0 / torch.sqrt(1.0 + t1 * t1)
        t = t1 * r
        r = r * mask + 1.0 * (1.0 - mask)
        t = t * mask
        x = 1.0 / torch.sqrt(r * r * a - 2.0 * r * t * b + t * t * c)
        z = 1.0 / torch.sqrt(t * t * a + 2.0 * r * t * b + r * r * c)
        d = torch.sqrt(x * z)
        x = x / d
        z = z / d
        l1 = torch.max(x, z)
        l2 = torch.min(x, z)
        new_a = r * r * x + t * t * z
        new_b = -r * t * x + t * r * z
        new_c = t * t * x + r * r * z
        return new_a, new_b, new_c, l1, l2

    def forward(self, x):
        if x.is_cuda:
            self.gk = self.gk
        else:
            self.gk = self.gk.cpu()
        gx = self.gx(F.pad(x, (1, 1, 0, 0), 'replicate'))
        gy = self.gy(F.pad(x, (0, 0, 1, 1), 'replicate'))
        a1 = (gx * gx * self.gk.unsqueeze(0).unsqueeze(0).expand_as(gx)).view(x
            .size(0), -1).mean(dim=1)
        b1 = (gx * gy * self.gk.unsqueeze(0).unsqueeze(0).expand_as(gx)).view(x
            .size(0), -1).mean(dim=1)
        c1 = (gy * gy * self.gk.unsqueeze(0).unsqueeze(0).expand_as(gx)).view(x
            .size(0), -1).mean(dim=1)
        a, b, c, l1, l2 = self.invSqrt(a1, b1, c1)
        rat1 = l1 / l2
        mask = (torch.abs(rat1) <= 6.0).float().view(-1)
        return rectifyAffineTransformationUpIsUp(abc2A(a, b, c))


class OrientationDetector(nn.Module):

    def __init__(self, mrSize=3.0, patch_size=None):
        super(OrientationDetector, self).__init__()
        if patch_size is None:
            patch_size = 32
        self.PS = patch_size
        self.bin_weight_kernel_size, self.bin_weight_stride = (self.
            get_bin_weight_kernel_size_and_stride(self.PS, 1))
        self.mrSize = mrSize
        self.num_ang_bins = 36
        self.gx = nn.Conv2d(1, 1, kernel_size=(1, 3), bias=False)
        self.gx.weight.data = torch.from_numpy(np.array([[[[0.5, 0, -0.5]]]
            ], dtype=np.float32))
        self.gy = nn.Conv2d(1, 1, kernel_size=(3, 1), bias=False)
        self.gy.weight.data = torch.from_numpy(np.array([[[[0.5], [0], [-
            0.5]]]], dtype=np.float32))
        self.angular_smooth = nn.Conv1d(1, 1, kernel_size=3, padding=1,
            bias=False)
        self.angular_smooth.weight.data = torch.from_numpy(np.array([[[0.33,
            0.34, 0.33]]], dtype=np.float32))
        self.gk = 10.0 * torch.from_numpy(CircularGaussKernel(kernlen=self.
            PS).astype(np.float32))
        self.gk = Variable(self.gk, requires_grad=False)
        return

    def get_bin_weight_kernel_size_and_stride(self, patch_size,
        num_spatial_bins):
        bin_weight_stride = int(round(2.0 * np.floor(patch_size / 2) /
            float(num_spatial_bins + 1)))
        bin_weight_kernel_size = int(2 * bin_weight_stride - 1)
        return bin_weight_kernel_size, bin_weight_stride

    def get_rotation_matrix(self, angle_in_radians):
        angle_in_radians = angle_in_radians.view(-1, 1, 1)
        sin_a = torch.sin(angle_in_radians)
        cos_a = torch.cos(angle_in_radians)
        A1_x = torch.cat([cos_a, sin_a], dim=2)
        A2_x = torch.cat([-sin_a, cos_a], dim=2)
        transform = torch.cat([A1_x, A2_x], dim=1)
        return transform

    def forward(self, x, return_rot_matrix=False):
        gx = self.gx(F.pad(x, (1, 1, 0, 0), 'replicate'))
        gy = self.gy(F.pad(x, (0, 0, 1, 1), 'replicate'))
        mag = torch.sqrt(gx * gx + gy * gy + 1e-10)
        if x.is_cuda:
            self.gk = self.gk
        mag = mag * self.gk.unsqueeze(0).unsqueeze(0).expand_as(mag)
        ori = torch.atan2(gy, gx)
        o_big = float(self.num_ang_bins) * (ori + 1.0 * math.pi) / (2.0 *
            math.pi)
        bo0_big = torch.floor(o_big)
        wo1_big = o_big - bo0_big
        bo0_big = bo0_big % self.num_ang_bins
        bo1_big = (bo0_big + 1) % self.num_ang_bins
        wo0_big = (1.0 - wo1_big) * mag
        wo1_big = wo1_big * mag
        ang_bins = []
        for i in range(0, self.num_ang_bins):
            ang_bins.append(F.adaptive_avg_pool2d((bo0_big == i).float() *
                wo0_big, (1, 1)))
        ang_bins = torch.cat(ang_bins, 1).view(-1, 1, self.num_ang_bins)
        ang_bins = self.angular_smooth(ang_bins)
        values, indices = ang_bins.view(-1, self.num_ang_bins).max(1)
        angle = -(2.0 * float(np.pi) * indices.float() / float(self.
            num_ang_bins) - float(math.pi))
        if return_rot_matrix:
            return self.get_rotation_matrix(angle)
        return angle


class NMS2d(nn.Module):

    def __init__(self, kernel_size=3, threshold=0):
        super(NMS2d, self).__init__()
        self.MP = nn.MaxPool2d(kernel_size, stride=1, return_indices=False,
            padding=kernel_size / 2)
        self.eps = 1e-05
        self.th = threshold
        return

    def forward(self, x):
        if self.th > self.eps:
            return x * (x > self.th).float() * (x + self.eps - self.MP(x) > 0
                ).float()
        else:
            return (x - self.MP(x) + self.eps > 0).float() * x


class NMS3d(nn.Module):

    def __init__(self, kernel_size=3, threshold=0):
        super(NMS3d, self).__init__()
        self.MP = nn.MaxPool3d(kernel_size, stride=1, return_indices=False,
            padding=(0, kernel_size // 2, kernel_size // 2))
        self.eps = 1e-05
        self.th = threshold
        return

    def forward(self, x):
        if self.th > self.eps:
            return x * (x > self.th).float() * (x + self.eps - self.MP(x) > 0
                ).float()
        else:
            return (x - self.MP(x) + self.eps > 0).float() * x


def generate_2dgrid(h, w, centered=True):
    if centered:
        x = torch.linspace(-w / 2 + 1, w / 2, w)
        y = torch.linspace(-h / 2 + 1, h / 2, h)
    else:
        x = torch.linspace(0, w - 1, w)
        y = torch.linspace(0, h - 1, h)
    grid2d = torch.stack([y.repeat(w, 1).t().contiguous().view(-1), x.
        repeat(h)], 1)
    return grid2d


def sc_y_x2LAFs(sc_y_x):
    base_LAF = torch.eye(2).float().unsqueeze(0).expand(sc_y_x.size(0), 2, 2)
    if sc_y_x.is_cuda:
        base_LAF = base_LAF.cuda()
    base_A = Variable(base_LAF, requires_grad=False)
    A = sc_y_x[:, :1].unsqueeze(1).expand_as(base_A) * base_A
    LAFs = torch.cat([A, torch.cat([sc_y_x[:, 2:].unsqueeze(-1), sc_y_x[:, 
        1:2].unsqueeze(-1)], dim=1)], dim=2)
    return LAFs


def generate_3dgrid(d, h, w, centered=True):
    if type(d) is not list:
        if centered:
            z = torch.linspace(-d / 2 + 1, d / 2, d)
        else:
            z = torch.linspace(0, d - 1, d)
        dl = d
    else:
        z = torch.FloatTensor(d)
        dl = len(d)
    grid2d = generate_2dgrid(h, w, centered=centered)
    grid3d = torch.cat([z.repeat(w * h, 1).t().contiguous().view(-1, 1),
        grid2d.repeat(dl, 1)], dim=1)
    return grid3d


def zero_response_at_border(x, b):
    if b < x.size(3) and b < x.size(2):
        x[:, :, 0:b, :] = 0
        x[:, :, x.size(2) - b:, :] = 0
        x[:, :, :, 0:b] = 0
        x[:, :, :, x.size(3) - b:] = 0
    else:
        return x * 0
    return x


class NMS3dAndComposeA(nn.Module):

    def __init__(self, w=0, h=0, kernel_size=3, threshold=0, scales=None,
        border=3, mrSize=1.0):
        super(NMS3dAndComposeA, self).__init__()
        self.eps = 1e-07
        self.ks = 3
        self.th = threshold
        self.cube_idxs = []
        self.border = border
        self.mrSize = mrSize
        self.beta = 1.0
        self.grid_ones = Variable(torch.ones(3, 3, 3, 3), requires_grad=False)
        self.NMS3d = NMS3d(kernel_size, threshold)
        if w > 0 and h > 0:
            self.spatial_grid = generate_2dgrid(h, w, False).view(1, h, w, 2
                ).permute(3, 1, 2, 0)
            self.spatial_grid = Variable(self.spatial_grid)
        else:
            self.spatial_grid = None
        return

    def forward(self, low, cur, high, num_features=0, octaveMap=None,
        scales=None):
        assert low.size() == cur.size() == high.size()
        self.is_cuda = low.is_cuda
        resp3d = torch.cat([low, cur, high], dim=1)
        mrSize_border = int(self.mrSize)
        if octaveMap is not None:
            nmsed_resp = zero_response_at_border(self.NMS3d(resp3d.
                unsqueeze(1)).squeeze(1)[:, 1:2, :, :], mrSize_border) * (
                1.0 - octaveMap.float())
        else:
            nmsed_resp = zero_response_at_border(self.NMS3d(resp3d.
                unsqueeze(1)).squeeze(1)[:, 1:2, :, :], mrSize_border)
        num_of_nonzero_responces = (nmsed_resp > 0).float().sum().item()
        if num_of_nonzero_responces <= 1:
            return None, None, None
        if octaveMap is not None:
            octaveMap = (octaveMap.float() + nmsed_resp.float()).byte()
        nmsed_resp = nmsed_resp.view(-1)
        if num_features > 0 and num_features < num_of_nonzero_responces:
            nmsed_resp, idxs = torch.topk(nmsed_resp, k=num_features, dim=0)
        else:
            idxs = nmsed_resp.data.nonzero().squeeze()
            nmsed_resp = nmsed_resp[idxs]
        if type(scales) is not list:
            self.grid = generate_3dgrid(3, self.ks, self.ks)
        else:
            self.grid = generate_3dgrid(scales, self.ks, self.ks)
        self.grid = Variable(self.grid.t().contiguous().view(3, 3, 3, 3),
            requires_grad=False)
        if self.spatial_grid is None:
            self.spatial_grid = generate_2dgrid(low.size(2), low.size(3), False
                ).view(1, low.size(2), low.size(3), 2).permute(3, 1, 2, 0)
            self.spatial_grid = Variable(self.spatial_grid)
        if self.is_cuda:
            self.spatial_grid = self.spatial_grid
            self.grid_ones = self.grid_ones
            self.grid = self.grid
        sc_y_x = F.conv2d(resp3d, self.grid, padding=1) / (F.conv2d(resp3d,
            self.grid_ones, padding=1) + 1e-08)
        sc_y_x[(0), 1:, :, :] = sc_y_x[(0), 1:, :, :] + self.spatial_grid[:,
            :, :, (0)]
        sc_y_x = sc_y_x.view(3, -1).t()
        sc_y_x = sc_y_x[(idxs), :]
        min_size = float(min(cur.size(2), cur.size(3)))
        sc_y_x[:, (0)] = sc_y_x[:, (0)] / min_size
        sc_y_x[:, (1)] = sc_y_x[:, (1)] / float(cur.size(2))
        sc_y_x[:, (2)] = sc_y_x[:, (2)] / float(cur.size(3))
        return nmsed_resp, sc_y_x2LAFs(sc_y_x), octaveMap


def sc_y_x_and_A2LAFs(sc_y_x, A_flat):
    base_A = A_flat.view(-1, 2, 2)
    A = sc_y_x[:, :1].unsqueeze(1).expand_as(base_A) * base_A
    LAFs = torch.cat([A, torch.cat([sc_y_x[:, 2:].unsqueeze(-1), sc_y_x[:, 
        1:2].unsqueeze(-1)], dim=1)], dim=2)
    return LAFs


class NMS3dAndComposeAAff(nn.Module):

    def __init__(self, w=0, h=0, kernel_size=3, threshold=0, scales=None,
        border=3, mrSize=1.0):
        super(NMS3dAndComposeAAff, self).__init__()
        self.eps = 1e-07
        self.ks = 3
        self.th = threshold
        self.cube_idxs = []
        self.border = border
        self.mrSize = mrSize
        self.beta = 1.0
        self.grid_ones = Variable(torch.ones(3, 3, 3, 3), requires_grad=False)
        self.NMS3d = NMS3d(kernel_size, threshold)
        if w > 0 and h > 0:
            self.spatial_grid = generate_2dgrid(h, w, False).view(1, h, w, 2
                ).permute(3, 1, 2, 0)
            self.spatial_grid = Variable(self.spatial_grid)
        else:
            self.spatial_grid = None
        return

    def forward(self, low, cur, high, num_features=0, octaveMap=None,
        scales=None, aff_resp=None):
        assert low.size() == cur.size() == high.size()
        self.is_cuda = low.is_cuda
        resp3d = torch.cat([low, cur, high], dim=1)
        mrSize_border = int(self.mrSize)
        if octaveMap is not None:
            nmsed_resp = zero_response_at_border(self.NMS3d(resp3d.
                unsqueeze(1)).squeeze(1)[:, 1:2, :, :], mrSize_border) * (
                1.0 - octaveMap.float())
        else:
            nmsed_resp = zero_response_at_border(self.NMS3d(resp3d.
                unsqueeze(1)).squeeze(1)[:, 1:2, :, :], mrSize_border)
        num_of_nonzero_responces = (nmsed_resp > 0).float().sum().item()
        if num_of_nonzero_responces <= 1:
            return None, None, None
        if octaveMap is not None:
            octaveMap = (octaveMap.float() + nmsed_resp.float()).byte()
        nmsed_resp = nmsed_resp.view(-1)
        if num_features > 0 and num_features < num_of_nonzero_responces:
            nmsed_resp, idxs = torch.topk(nmsed_resp, k=num_features, dim=0)
        else:
            idxs = nmsed_resp.data.nonzero().squeeze()
            nmsed_resp = nmsed_resp[idxs]
        if type(scales) is not list:
            self.grid = generate_3dgrid(3, self.ks, self.ks)
        else:
            self.grid = generate_3dgrid(scales, self.ks, self.ks)
        self.grid = Variable(self.grid.t().contiguous().view(3, 3, 3, 3),
            requires_grad=False)
        if self.spatial_grid is None:
            self.spatial_grid = generate_2dgrid(low.size(2), low.size(3), False
                ).view(1, low.size(2), low.size(3), 2).permute(3, 1, 2, 0)
            self.spatial_grid = Variable(self.spatial_grid)
        if self.is_cuda:
            self.spatial_grid = self.spatial_grid
            self.grid_ones = self.grid_ones
            self.grid = self.grid
        sc_y_x = F.conv2d(resp3d, self.grid, padding=1) / (F.conv2d(resp3d,
            self.grid_ones, padding=1) + 1e-08)
        sc_y_x[(0), 1:, :, :] = sc_y_x[(0), 1:, :, :] + self.spatial_grid[:,
            :, :, (0)]
        sc_y_x = sc_y_x.view(3, -1).t()
        sc_y_x = sc_y_x[(idxs), :]
        if aff_resp is not None:
            A_matrices = aff_resp.view(4, -1).t()[(idxs), :]
        min_size = float(min(cur.size(2), cur.size(3)))
        sc_y_x[:, (0)] = sc_y_x[:, (0)] / min_size
        sc_y_x[:, (1)] = sc_y_x[:, (1)] / float(cur.size(2))
        sc_y_x[:, (2)] = sc_y_x[:, (2)] / float(cur.size(3))
        return nmsed_resp, sc_y_x_and_A2LAFs(sc_y_x, A_matrices), octaveMap


class L2Norm(nn.Module):

    def __init__(self):
        super(L2Norm, self).__init__()
        self.eps = 1e-08

    def forward(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim=1) + self.eps)
        x = x / norm.unsqueeze(-1).expand_as(x)
        return x


class L1Norm(nn.Module):

    def __init__(self):
        super(L1Norm, self).__init__()
        self.eps = 1e-10

    def forward(self, x):
        norm = torch.sum(torch.abs(x), dim=1) + self.eps
        x = x / norm.expand_as(x)
        return x


class HardTFeatNet(nn.Module):
    """TFeat model definition
    """

    def __init__(self, sm):
        super(HardTFeatNet, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(1, 32, kernel_size=7), nn.
            Tanh(), nn.MaxPool2d(kernel_size=2, stride=2), nn.Conv2d(32, 64,
            kernel_size=6), nn.Tanh())
        self.classifier = nn.Sequential(nn.Dropout(0.1), nn.Conv2d(64, 128,
            kernel_size=8), nn.Tanh())
        self.SIFT = sm

    def input_norm(self, x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-07
        return (x - mp.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)
            ) / sp.unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)

    def forward(self, input):
        x_features = self.features(self.input_norm(input))
        x = x_features.view(x_features.size(0), -1)
        x = self.classifier(x_features)
        return L2Norm()(x.view(x.size(0), -1))


class HardNet(nn.Module):
    """HardNet model definition
    """

    def __init__(self):
        super(HardNet, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(1, 32, kernel_size=3,
            padding=1, bias=False), nn.BatchNorm2d(32, affine=False), nn.
            ReLU(), nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False), nn.ReLU(), nn.Conv2d(32, 64,
            kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d
            (64, affine=False), nn.ReLU(), nn.Conv2d(64, 64, kernel_size=3,
            padding=1, bias=False), nn.BatchNorm2d(64, affine=False), nn.
            ReLU(), nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1,
            bias=False), nn.BatchNorm2d(128, affine=False), nn.ReLU(), nn.
            Conv2d(128, 128, kernel_size=3, padding=1, bias=False), nn.
            BatchNorm2d(128, affine=False), nn.ReLU(), nn.Dropout(0.1), nn.
            Conv2d(128, 128, kernel_size=8, bias=False), nn.BatchNorm2d(128,
            affine=False))

    def input_norm(self, x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-07
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).
            expand_as(x)) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1
            ).expand_as(x)

    def forward(self, input):
        x_features = self.features(self.input_norm(input))
        x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x)


def denormalizeLAFs(LAFs, w, h):
    w = float(w)
    h = float(h)
    num_lafs = LAFs.size(0)
    min_size = min(h, w)
    coef = torch.ones(1, 2, 3).float() * min_size
    coef[0, 0, 2] = w
    coef[0, 1, 2] = h
    if LAFs.is_cuda:
        coef = coef.cuda()
    return Variable(coef.expand(num_lafs, 2, 3)) * LAFs


def angles2A(angles):
    cos_a = torch.cos(angles).view(-1, 1, 1)
    sin_a = torch.sin(angles).view(-1, 1, 1)
    A1_ang = torch.cat([cos_a, sin_a], dim=2)
    A2_ang = torch.cat([-sin_a, cos_a], dim=2)
    return torch.cat([A1_ang, A2_ang], dim=1)


def LAFs_to_H_frames(aff_pts):
    H3_x = torch.Tensor([0, 0, 1]).unsqueeze(0).unsqueeze(0).repeat(aff_pts
        .size(0), 1, 1)
    if aff_pts.is_cuda:
        H3_x = H3_x.cuda()
    return torch.cat([aff_pts, H3_x], dim=1)


def checkTouchBoundary(LAFs):
    pts = torch.FloatTensor([[-1, -1, 1, 1], [-1, 1, -1, 1], [1, 1, 1, 1]]
        ).unsqueeze(0)
    if LAFs.is_cuda:
        pts = pts.cuda()
    out_pts = torch.bmm(LAFs_to_H_frames(LAFs), pts.expand(LAFs.size(0), 3, 4)
        )[:, :2, :]
    good_points = 1 - (((out_pts > 1.0) + (out_pts < 0.0)).sum(dim=1).sum(
        dim=1) > 0)
    return good_points


def normalizeLAFs(LAFs, w, h):
    w = float(w)
    h = float(h)
    num_lafs = LAFs.size(0)
    min_size = min(h, w)
    coef = torch.ones(1, 2, 3).float() / min_size
    coef[0, 0, 2] = 1.0 / w
    coef[0, 1, 2] = 1.0 / h
    if LAFs.is_cuda:
        coef = coef.cuda()
    return Variable(coef.expand(num_lafs, 2, 3)) * LAFs


def generate_patch_grid_from_normalized_LAFs(LAFs, w, h, PS):
    num_lafs = LAFs.size(0)
    min_size = min(h, w)
    coef = torch.ones(1, 2, 3) * min_size
    coef[0, 0, 2] = w
    coef[0, 1, 2] = h
    if LAFs.is_cuda:
        coef = coef.cuda()
    grid = torch.nn.functional.affine_grid(LAFs * Variable(coef.expand(
        num_lafs, 2, 3)), torch.Size((num_lafs, 1, PS, PS)))
    grid[:, :, :, (0)] = 2.0 * grid[:, :, :, (0)] / float(w) - 1.0
    grid[:, :, :, (1)] = 2.0 * grid[:, :, :, (1)] / float(h) - 1.0
    return grid


def extract_patches(img, LAFs, PS=32):
    w = img.size(3)
    h = img.size(2)
    ch = img.size(1)
    grid = generate_patch_grid_from_normalized_LAFs(LAFs, float(w), float(h
        ), PS)
    return torch.nn.functional.grid_sample(img.expand(grid.size(0), ch, h,
        w), grid)


def extract_patches_from_pyramid_with_inv_index(scale_pyramid, pyr_inv_idxs,
    LAFs, PS=19):
    patches = torch.zeros(LAFs.size(0), scale_pyramid[0][0].size(1), PS, PS)
    if LAFs.is_cuda:
        patches = patches.cuda()
    patches = Variable(patches)
    if pyr_inv_idxs is not None:
        for i in range(len(scale_pyramid)):
            for j in range(len(scale_pyramid[i])):
                cur_lvl_idxs = pyr_inv_idxs[i][j]
                if cur_lvl_idxs is None:
                    continue
                cur_lvl_idxs = cur_lvl_idxs.view(-1)
                patches[(cur_lvl_idxs), :, :, :] = extract_patches(
                    scale_pyramid[i][j], LAFs[(cur_lvl_idxs), :, :], PS)
    return patches


def get_inverted_pyr_index(scale_pyr, pyr_idxs, level_idxs):
    pyr_inv_idxs = []
    for i in range(len(scale_pyr)):
        pyr_inv_idxs.append([])
        cur_idxs = pyr_idxs == i
        for j in range(0, len(scale_pyr[i])):
            cur_lvl_idxs = torch.nonzero(((level_idxs == j) * cur_idxs).data)
            if len(cur_lvl_idxs.size()) == 0:
                pyr_inv_idxs[i].append(None)
            else:
                pyr_inv_idxs[i].append(cur_lvl_idxs.squeeze())
    return pyr_inv_idxs


def get_LAFs_scales(LAFs):
    return torch.sqrt(torch.abs(LAFs[:, (0), (0)] * LAFs[:, (1), (1)] - 
        LAFs[:, (0), (1)] * LAFs[:, (1), (0)]) + 1e-12)


def get_pyramid_and_level_index_for_LAFs(dLAFs, sigmas, pix_dists, PS):
    scales = get_LAFs_scales(dLAFs)
    needed_sigmas = scales / PS
    sigmas_full_list = []
    level_idxs_full = []
    oct_idxs_full = []
    for oct_idx in range(len(sigmas)):
        sigmas_full_list = sigmas_full_list + list(np.array(sigmas[oct_idx]
            ) * np.array(pix_dists[oct_idx]))
        oct_idxs_full = oct_idxs_full + [oct_idx] * len(sigmas[oct_idx])
        level_idxs_full = level_idxs_full + list(range(0, len(sigmas[oct_idx]))
            )
    oct_idxs_full = torch.LongTensor(oct_idxs_full)
    level_idxs_full = torch.LongTensor(level_idxs_full)
    closest_imgs = cdist(np.array(sigmas_full_list).reshape(-1, 1),
        needed_sigmas.data.cpu().numpy().reshape(-1, 1)).argmin(axis=0)
    closest_imgs = torch.from_numpy(closest_imgs)
    if dLAFs.is_cuda:
        closest_imgs = closest_imgs.cuda()
        oct_idxs_full = oct_idxs_full.cuda()
        level_idxs_full = level_idxs_full.cuda()
    return Variable(oct_idxs_full[closest_imgs]), Variable(level_idxs_full[
        closest_imgs])


class OnePassSIR(nn.Module):

    def __init__(self, border=16, num_features=500, patch_size=32, mrSize=
        3.0, nlevels=3, th=None, num_Baum_iters=0, init_sigma=1.6, RespNet=
        None, OriNet=None, AffNet=None):
        super(OnePassSIR, self).__init__()
        self.mrSize = mrSize
        self.PS = patch_size
        self.b = border
        self.num = num_features
        self.th = th
        if th is not None:
            self.num = -1
        else:
            self.th = 0
        self.nlevels = nlevels
        self.num_Baum_iters = num_Baum_iters
        self.init_sigma = init_sigma
        if RespNet is not None:
            self.RespNet = RespNet
        else:
            self.RespNet = HessianResp()
        if OriNet is not None:
            self.OriNet = OriNet
        else:
            self.OriNet = OrientationDetector(patch_size=19)
        if AffNet is not None:
            self.AffNet = AffNet
        else:
            self.AffNet = AffineShapeEstimator(patch_size=19)
        self.ScalePyrGen = ScalePyramid(nLevels=self.nlevels, init_sigma=
            self.init_sigma, border=self.b)
        return

    def multiScaleDetectorAff(self, x, num_features=0):
        t = time.time()
        self.scale_pyr, self.sigmas, self.pix_dists = self.ScalePyrGen(x)
        aff_matrices = []
        top_responces = []
        pyr_idxs = []
        level_idxs = []
        det_t = 0
        nmst = 0
        for oct_idx in range(len(self.sigmas)):
            octave = self.scale_pyr[oct_idx]
            sigmas_oct = self.sigmas[oct_idx]
            pix_dists_oct = self.pix_dists[oct_idx]
            low = None
            cur = None
            high = None
            octaveMap = (self.scale_pyr[oct_idx][0] * 0).byte()
            nms_f = NMS3dAndComposeAAff(w=octave[0].size(3), h=octave[0].
                size(2), border=self.b, mrSize=self.mrSize)
            oct_aff_map = self.AffNet(octave[0])
            for level_idx in range(1, len(octave) - 1):
                if cur is None:
                    low = torch.clamp(self.RespNet(octave[level_idx - 1],
                        sigmas_oct[level_idx - 1]) - self.th, min=0)
                else:
                    low = cur
                if high is None:
                    cur = torch.clamp(self.RespNet(octave[level_idx],
                        sigmas_oct[level_idx]) - self.th, min=0)
                else:
                    cur = high
                high = torch.clamp(self.RespNet(octave[level_idx + 1],
                    sigmas_oct[level_idx + 1]) - self.th, min=0)
                top_resp, aff_matrix, octaveMap_current = nms_f(low, cur,
                    high, num_features=num_features, octaveMap=octaveMap,
                    scales=sigmas_oct[level_idx - 1:level_idx + 2],
                    aff_resp=oct_aff_map)
                if top_resp is None:
                    continue
                octaveMap = octaveMap_current
                not_touch_boundary_idx = checkTouchBoundary(torch.cat([
                    aff_matrix[:, :2, :2] * 3.0, aff_matrix[:, :, 2:]], dim=2))
                aff_matrices.append(aff_matrix[not_touch_boundary_idx.byte()]
                    ), top_responces.append(top_resp[not_touch_boundary_idx
                    .byte()])
                pyr_id = Variable(oct_idx * torch.ones(aff_matrices[-1].
                    size(0)))
                lev_id = Variable((level_idx - 1) * torch.ones(aff_matrices
                    [-1].size(0)))
                if x.is_cuda:
                    pyr_id = pyr_id
                    lev_id = lev_id
                pyr_idxs.append(pyr_id)
                level_idxs.append(lev_id)
        all_responses = torch.cat(top_responces, dim=0)
        aff_m_scales = torch.cat(aff_matrices, dim=0)
        pyr_idxs_scales = torch.cat(pyr_idxs, dim=0)
        level_idxs_scale = torch.cat(level_idxs, dim=0)
        if num_features > 0 and num_features < all_responses.size(0):
            all_responses, idxs = torch.topk(all_responses, k=num_features)
            LAFs = torch.index_select(aff_m_scales, 0, idxs)
            final_pyr_idxs = pyr_idxs_scales[idxs]
            final_level_idxs = level_idxs_scale[idxs]
        else:
            return (all_responses, aff_m_scales, pyr_idxs_scales,
                level_idxs_scale)
        return all_responses, LAFs, final_pyr_idxs, final_level_idxs

    def getOrientation(self, LAFs, final_pyr_idxs, final_level_idxs):
        pyr_inv_idxs = get_inverted_pyr_index(self.scale_pyr,
            final_pyr_idxs, final_level_idxs)
        patches_small = extract_patches_from_pyramid_with_inv_index(self.
            scale_pyr, pyr_inv_idxs, LAFs, PS=self.OriNet.PS)
        max_iters = 1
        for i in range(max_iters):
            angles = self.OriNet(patches_small)
            if len(angles.size()) > 2:
                LAFs = torch.cat([torch.bmm(LAFs[:, :, :2], angles), LAFs[:,
                    :, 2:]], dim=2)
            else:
                LAFs = torch.cat([torch.bmm(LAFs[:, :, :2], angles2A(angles
                    ).view(-1, 2, 2)), LAFs[:, :, 2:]], dim=2)
            if i != max_iters - 1:
                patches_small = extract_patches_from_pyramid_with_inv_index(
                    self.scale_pyr, pyr_inv_idxs, LAFs, PS=self.OriNet.PS)
        return LAFs

    def extract_patches_from_pyr(self, dLAFs, PS=41):
        pyr_idxs, level_idxs = get_pyramid_and_level_index_for_LAFs(dLAFs,
            self.sigmas, self.pix_dists, PS)
        pyr_inv_idxs = get_inverted_pyr_index(self.scale_pyr, pyr_idxs,
            level_idxs)
        patches = extract_patches_from_pyramid_with_inv_index(self.
            scale_pyr, pyr_inv_idxs, normalizeLAFs(dLAFs, self.scale_pyr[0]
            [0].size(3), self.scale_pyr[0][0].size(2)), PS=PS)
        return patches

    def forward(self, x, do_ori=True):
        t = time.time()
        num_features_prefilter = self.num
        responses, LAFs, final_pyr_idxs, final_level_idxs = (self.
            multiScaleDetectorAff(x, num_features_prefilter))
        None
        t = time.time()
        LAFs[:, 0:2, 0:2] = self.mrSize * LAFs[:, :, 0:2]
        if do_ori:
            LAFs = self.getOrientation(LAFs, final_pyr_idxs, final_level_idxs)
        return denormalizeLAFs(LAFs, x.size(3), x.size(2)), responses


def batched_forward(model, data, batch_size, **kwargs):
    n_patches = len(data)
    if n_patches > batch_size:
        bs = batch_size
        n_batches = n_patches / bs + 1
        for batch_idx in range(n_batches):
            st = batch_idx * bs
            if batch_idx == n_batches - 1:
                if (batch_idx + 1) * bs > n_patches:
                    end = n_patches
                else:
                    end = (batch_idx + 1) * bs
            else:
                end = (batch_idx + 1) * bs
            if st >= end:
                continue
            if batch_idx == 0:
                first_batch_out = model(data[st:end], kwargs)
                out_size = torch.Size([n_patches] + list(first_batch_out.
                    size()[1:]))
                out = torch.zeros(out_size)
                if data.is_cuda:
                    out = out.cuda()
                out = Variable(out)
                out[st:end] = first_batch_out
            else:
                out[st:end, :, :] = model(data[st:end], kwargs)
        return out
    else:
        return model(data, kwargs)


def batch_eig2x2(A):
    trace = A[:, (0), (0)] + A[:, (1), (1)]
    delta1 = trace * trace - 4 * (A[:, (0), (0)] * A[:, (1), (1)] - A[:, (1
        ), (0)] * A[:, (0), (1)])
    mask = delta1 > 0
    delta = torch.sqrt(torch.abs(delta1))
    l1 = mask.float() * (trace + delta) / 2.0 + 1000.0 * (1.0 - mask.float())
    l2 = mask.float() * (trace - delta) / 2.0 + 0.0001 * (1.0 - mask.float())
    return l1, l2


class ScaleSpaceAffinePatchExtractor(nn.Module):

    def __init__(self, border=16, num_features=500, patch_size=32, mrSize=
        3.0, nlevels=3, num_Baum_iters=0, init_sigma=1.6, th=None, RespNet=
        None, OriNet=None, AffNet=None):
        super(ScaleSpaceAffinePatchExtractor, self).__init__()
        self.mrSize = mrSize
        self.PS = patch_size
        self.b = border
        self.num = num_features
        self.nlevels = nlevels
        self.num_Baum_iters = num_Baum_iters
        self.init_sigma = init_sigma
        self.th = th
        if th is not None:
            self.num = -1
        else:
            self.th = 0
        if RespNet is not None:
            self.RespNet = RespNet
        else:
            self.RespNet = HessianResp()
        if OriNet is not None:
            self.OriNet = OriNet
        else:
            self.OriNet = OrientationDetector(patch_size=19)
        if AffNet is not None:
            self.AffNet = AffNet
        else:
            self.AffNet = AffineShapeEstimator(patch_size=19)
        self.ScalePyrGen = ScalePyramid(nLevels=self.nlevels, init_sigma=
            self.init_sigma, border=self.b)
        return

    def multiScaleDetector(self, x, num_features=0):
        t = time.time()
        self.scale_pyr, self.sigmas, self.pix_dists = self.ScalePyrGen(x)
        aff_matrices = []
        top_responces = []
        pyr_idxs = []
        level_idxs = []
        det_t = 0
        nmst = 0
        for oct_idx in range(len(self.sigmas)):
            octave = self.scale_pyr[oct_idx]
            sigmas_oct = self.sigmas[oct_idx]
            pix_dists_oct = self.pix_dists[oct_idx]
            low = None
            cur = None
            high = None
            octaveMap = (self.scale_pyr[oct_idx][0] * 0).byte()
            nms_f = NMS3dAndComposeA(w=octave[0].size(3), h=octave[0].size(
                2), border=self.b, mrSize=self.mrSize)
            for level_idx in range(1, len(octave) - 1):
                if cur is None:
                    low = torch.clamp(self.RespNet(octave[level_idx - 1],
                        sigmas_oct[level_idx - 1]) - self.th, min=0)
                else:
                    low = cur
                if high is None:
                    cur = torch.clamp(self.RespNet(octave[level_idx],
                        sigmas_oct[level_idx]) - self.th, min=0)
                else:
                    cur = high
                high = torch.clamp(self.RespNet(octave[level_idx + 1],
                    sigmas_oct[level_idx + 1]) - self.th, min=0)
                top_resp, aff_matrix, octaveMap_current = nms_f(low, cur,
                    high, num_features=num_features, octaveMap=octaveMap,
                    scales=sigmas_oct[level_idx - 1:level_idx + 2])
                if top_resp is None:
                    continue
                octaveMap = octaveMap_current
                aff_matrices.append(aff_matrix), top_responces.append(top_resp)
                pyr_id = Variable(oct_idx * torch.ones(aff_matrix.size(0)))
                lev_id = Variable((level_idx - 1) * torch.ones(aff_matrix.
                    size(0)))
                if x.is_cuda:
                    pyr_id = pyr_id
                    lev_id = lev_id
                pyr_idxs.append(pyr_id)
                level_idxs.append(lev_id)
        all_responses = torch.cat(top_responces, dim=0)
        aff_m_scales = torch.cat(aff_matrices, dim=0)
        pyr_idxs_scales = torch.cat(pyr_idxs, dim=0)
        level_idxs_scale = torch.cat(level_idxs, dim=0)
        if num_features > 0 and num_features < all_responses.size(0):
            all_responses, idxs = torch.topk(all_responses, k=num_features)
            LAFs = torch.index_select(aff_m_scales, 0, idxs)
            final_pyr_idxs = pyr_idxs_scales[idxs]
            final_level_idxs = level_idxs_scale[idxs]
        else:
            return (all_responses, aff_m_scales, pyr_idxs_scales,
                level_idxs_scale)
        return all_responses, LAFs, final_pyr_idxs, final_level_idxs

    def getAffineShape(self, final_resp, LAFs, final_pyr_idxs,
        final_level_idxs, num_features=0):
        pe_time = 0
        affnet_time = 0
        pyr_inv_idxs = get_inverted_pyr_index(self.scale_pyr,
            final_pyr_idxs, final_level_idxs)
        t = time.time()
        patches_small = extract_patches_from_pyramid_with_inv_index(self.
            scale_pyr, pyr_inv_idxs, LAFs, PS=self.AffNet.PS)
        pe_time += time.time() - t
        t = time.time()
        base_A = torch.eye(2).unsqueeze(0).expand(final_pyr_idxs.size(0), 2, 2)
        if final_resp.is_cuda:
            base_A = base_A
        base_A = Variable(base_A)
        is_good = None
        n_patches = patches_small.size(0)
        for i in range(self.num_Baum_iters):
            t = time.time()
            A = batched_forward(self.AffNet, patches_small, 256)
            is_good_current = 1
            affnet_time += time.time() - t
            if is_good is None:
                is_good = is_good_current
            else:
                is_good = is_good * is_good_current
            base_A = torch.bmm(A, base_A)
            new_LAFs = torch.cat([torch.bmm(base_A, LAFs[:, :, 0:2]), LAFs[
                :, :, 2:]], dim=2)
            if i != self.num_Baum_iters - 1:
                pe_time += time.time() - t
                t = time.time()
                patches_small = extract_patches_from_pyramid_with_inv_index(
                    self.scale_pyr, pyr_inv_idxs, new_LAFs, PS=self.AffNet.PS)
                pe_time += time.time() - t
                l1, l2 = batch_eig2x2(A)
                ratio1 = torch.abs(l1 / (l2 + 1e-08))
                converged_mask = (ratio1 <= 1.2) * (ratio1 >= 0.8)
        l1, l2 = batch_eig2x2(base_A)
        ratio = torch.abs(l1 / (l2 + 1e-08))
        idxs_mask = (ratio < 6.0) * (ratio > 1.0 / 6.0) * checkTouchBoundary(
            new_LAFs)
        num_survived = idxs_mask.float().sum()
        if num_features > 0 and num_survived.data.item() > num_features:
            final_resp = final_resp * idxs_mask.float()
            final_resp, idxs = torch.topk(final_resp, k=num_features)
        else:
            idxs = Variable(torch.nonzero(idxs_mask.data).view(-1).long())
            final_resp = final_resp[idxs]
        final_pyr_idxs = final_pyr_idxs[idxs]
        final_level_idxs = final_level_idxs[idxs]
        base_A = torch.index_select(base_A, 0, idxs)
        LAFs = torch.index_select(LAFs, 0, idxs)
        new_LAFs = torch.cat([torch.bmm(base_A, LAFs[:, :, 0:2]), LAFs[:, :,
            2:]], dim=2)
        None
        None
        return final_resp, new_LAFs, final_pyr_idxs, final_level_idxs

    def getOrientation(self, LAFs, final_pyr_idxs, final_level_idxs):
        pyr_inv_idxs = get_inverted_pyr_index(self.scale_pyr,
            final_pyr_idxs, final_level_idxs)
        patches_small = extract_patches_from_pyramid_with_inv_index(self.
            scale_pyr, pyr_inv_idxs, LAFs, PS=self.OriNet.PS)
        max_iters = 1
        for i in range(max_iters):
            angles = self.OriNet(patches_small)
            if len(angles.size()) > 2:
                LAFs = torch.cat([torch.bmm(LAFs[:, :, :2], angles), LAFs[:,
                    :, 2:]], dim=2)
            else:
                LAFs = torch.cat([torch.bmm(LAFs[:, :, :2], angles2A(angles
                    ).view(-1, 2, 2)), LAFs[:, :, 2:]], dim=2)
            if i != max_iters:
                patches_small = extract_patches_from_pyramid_with_inv_index(
                    self.scale_pyr, pyr_inv_idxs, LAFs, PS=self.OriNet.PS)
        return LAFs

    def extract_patches_from_pyr(self, dLAFs, PS=41):
        pyr_idxs, level_idxs = get_pyramid_and_level_index_for_LAFs(dLAFs,
            self.sigmas, self.pix_dists, PS)
        pyr_inv_idxs = get_inverted_pyr_index(self.scale_pyr, pyr_idxs,
            level_idxs)
        patches = extract_patches_from_pyramid_with_inv_index(self.
            scale_pyr, pyr_inv_idxs, normalizeLAFs(dLAFs, self.scale_pyr[0]
            [0].size(3), self.scale_pyr[0][0].size(2)), PS=PS)
        return patches

    def forward(self, x, do_ori=False):
        t = time.time()
        num_features_prefilter = self.num
        if self.num_Baum_iters > 0:
            num_features_prefilter = int(1.5 * self.num)
        responses, LAFs, final_pyr_idxs, final_level_idxs = (self.
            multiScaleDetector(x, num_features_prefilter))
        None
        t = time.time()
        LAFs[:, 0:2, 0:2] = self.mrSize * LAFs[:, :, 0:2]
        if self.num_Baum_iters > 0:
            responses, LAFs, final_pyr_idxs, final_level_idxs = (self.
                getAffineShape(responses, LAFs, final_pyr_idxs,
                final_level_idxs, self.num))
        None
        t = time.time()
        if do_ori:
            LAFs = self.getOrientation(LAFs, final_pyr_idxs, final_level_idxs)
        return denormalizeLAFs(LAFs, x.size(3), x.size(2)), responses


class L2Norm(nn.Module):

    def __init__(self):
        super(L2Norm, self).__init__()
        self.eps = 1e-10

    def forward(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim=1) + self.eps)
        x = x / norm.unsqueeze(-1).expand_as(x)
        return x


class L1Norm(nn.Module):

    def __init__(self):
        super(L1Norm, self).__init__()
        self.eps = 1e-10

    def forward(self, x):
        norm = torch.sum(torch.abs(x), dim=1) + self.eps
        x = x / norm.expand_as(x)
        return x


class GaussianBlur(nn.Module):

    def __init__(self, sigma=1.6):
        super(GaussianBlur, self).__init__()
        weight = self.calculate_weights(sigma)
        self.register_buffer('buf', weight)
        return

    def calculate_weights(self, sigma):
        kernel = CircularGaussKernel(sigma=sigma, circ_zeros=False)
        h, w = kernel.shape
        halfSize = float(h) / 2.0
        self.pad = int(np.floor(halfSize))
        return torch.from_numpy(kernel.astype(np.float32)).view(1, 1, h, w)

    def forward(self, x):
        w = Variable(self.buf)
        if x.is_cuda:
            w = w
        return F.conv2d(F.pad(x, (self.pad, self.pad, self.pad, self.pad),
            'replicate'), w, padding=0)


class LocalNorm2d(nn.Module):

    def __init__(self, kernel_size=33):
        super(LocalNorm2d, self).__init__()
        self.ks = kernel_size
        self.pool = nn.AvgPool2d(kernel_size=self.ks, stride=1, padding=0)
        self.eps = 1e-10
        return

    def forward(self, x):
        pd = int(self.ks / 2)
        mean = self.pool(F.pad(x, (pd, pd, pd, pd), 'reflect'))
        return torch.clamp((x - mean) / (torch.sqrt(torch.abs(self.pool(F.
            pad(x * x, (pd, pd, pd, pd), 'reflect')) - mean * mean)) + self
            .eps), min=-6.0, max=6.0)


def get_rotation_matrix(angle_in_radians):
    angle_in_radians = angle_in_radians.view(-1, 1, 1)
    sin_a = torch.sin(angle_in_radians)
    cos_a = torch.cos(angle_in_radians)
    A1_x = torch.cat([cos_a, sin_a], dim=2)
    A2_x = torch.cat([-sin_a, cos_a], dim=2)
    transform = torch.cat([A1_x, A2_x], dim=1)
    return transform


class OriNetFast(nn.Module):

    def __init__(self, PS=16):
        super(OriNetFast, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3,
            padding=1, bias=False), nn.BatchNorm2d(16, affine=False), nn.
            ReLU(), nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1,
            bias=False), nn.BatchNorm2d(16, affine=False), nn.ReLU(), nn.
            Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False), nn.ReLU(), nn.Conv2d(32, 32,
            kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d
            (32, affine=False), nn.ReLU(), nn.Conv2d(32, 64, kernel_size=3,
            stride=2, padding=1, bias=False), nn.BatchNorm2d(64, affine=
            False), nn.ReLU(), nn.Conv2d(64, 64, kernel_size=3, stride=1,
            padding=1, bias=False), nn.BatchNorm2d(64, affine=False), nn.
            ReLU(), nn.Dropout(0.25), nn.Conv2d(64, 2, kernel_size=int(PS /
            4), stride=1, padding=1, bias=True), nn.Tanh(), nn.
            AdaptiveAvgPool2d(1))
        self.PS = PS
        self.features.apply(self.weights_init)
        self.halfPS = int(PS / 4)
        return

    def input_norm(self, x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-07
        return (x - mp.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)
            ) / sp.unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.orthogonal(m.weight.data, gain=0.9)
            try:
                nn.init.constant(m.bias.data, 0.01)
            except:
                pass
        return

    def forward(self, input, return_rot_matrix=True):
        xy = self.features(self.input_norm(input)).view(-1, 2)
        angle = torch.atan2(xy[:, (0)] + 1e-08, xy[:, (1)] + 1e-08)
        if return_rot_matrix:
            return get_rotation_matrix(angle)
        return angle


class GHH(nn.Module):

    def __init__(self, n_in, n_out, s=4, m=4):
        super(GHH, self).__init__()
        self.n_out = n_out
        self.s = s
        self.m = m
        self.conv = nn.Linear(n_in, n_out * s * m)
        d = torch.arange(0, s)
        self.deltas = -1.0 * (d % 2 != 0).float() + 1.0 * (d % 2 == 0).float()
        self.deltas = Variable(self.deltas)
        return

    def forward(self, x):
        x_feats = self.conv(x.view(x.size(0), -1)).view(x.size(0), self.
            n_out, self.s, self.m)
        max_feats = x_feats.max(dim=3)[0]
        if x.is_cuda:
            self.deltas = self.deltas
        else:
            self.deltas = self.deltas.cpu()
        out = (max_feats * self.deltas.view(1, 1, -1).expand_as(max_feats)
            ).sum(dim=2)
        return out


class YiNet(nn.Module):

    def __init__(self, PS=28):
        super(YiNet, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(1, 10, kernel_size=5,
            padding=0, bias=True), nn.ReLU(), nn.MaxPool2d(kernel_size=3,
            stride=2, padding=1), nn.Conv2d(10, 20, kernel_size=5, stride=1,
            padding=0, bias=True), nn.ReLU(), nn.MaxPool2d(kernel_size=4,
            stride=2, padding=2), nn.Conv2d(20, 50, kernel_size=3, stride=1,
            padding=0, bias=True), nn.ReLU(), nn.AdaptiveMaxPool2d(1), GHH(
            50, 100), GHH(100, 2))
        self.input_mean = 0.427117081207483
        self.input_std = 0.21888339179665006
        self.PS = PS
        return

    def import_weights(self, dir_name):
        self.features[0].weight.data = torch.from_numpy(np.load(os.path.
            join(dir_name, 'layer0_W.npy'))).float()
        self.features[0].bias.data = torch.from_numpy(np.load(os.path.join(
            dir_name, 'layer0_b.npy'))).float().view(-1)
        self.features[3].weight.data = torch.from_numpy(np.load(os.path.
            join(dir_name, 'layer1_W.npy'))).float()
        self.features[3].bias.data = torch.from_numpy(np.load(os.path.join(
            dir_name, 'layer1_b.npy'))).float().view(-1)
        self.features[6].weight.data = torch.from_numpy(np.load(os.path.
            join(dir_name, 'layer2_W.npy'))).float()
        self.features[6].bias.data = torch.from_numpy(np.load(os.path.join(
            dir_name, 'layer2_b.npy'))).float().view(-1)
        self.features[9].conv.weight.data = torch.from_numpy(np.load(os.
            path.join(dir_name, 'layer3_W.npy'))).float().view(50, 1600
            ).contiguous().t().contiguous()
        self.features[9].conv.bias.data = torch.from_numpy(np.load(os.path.
            join(dir_name, 'layer3_b.npy'))).float().view(1600)
        self.features[10].conv.weight.data = torch.from_numpy(np.load(os.
            path.join(dir_name, 'layer4_W.npy'))).float().view(100, 32
            ).contiguous().t().contiguous()
        self.features[10].conv.bias.data = torch.from_numpy(np.load(os.path
            .join(dir_name, 'layer4_b.npy'))).float().view(32)
        self.input_mean = float(np.load(os.path.join(dir_name,
            'input_mean.npy')))
        self.input_std = float(np.load(os.path.join(dir_name, 'input_std.npy'))
            )
        return

    def input_norm1(self, x):
        return (x - self.input_mean) / self.input_std

    def input_norm(self, x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-07
        return (x - mp.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)
            ) / sp.unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)

    def forward(self, input, return_rot_matrix=False):
        xy = self.features(self.input_norm(input))
        angle = torch.atan2(xy[:, (0)] + 1e-08, xy[:, (1)] + 1e-08)
        if return_rot_matrix:
            return get_rotation_matrix(-angle)
        return angle


class AffNetFast4(nn.Module):

    def __init__(self, PS=32):
        super(AffNetFast4, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3,
            padding=1, bias=False), nn.BatchNorm2d(16, affine=False), nn.
            ReLU(), nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1,
            bias=False), nn.BatchNorm2d(16, affine=False), nn.ReLU(), nn.
            Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False), nn.ReLU(), nn.Conv2d(32, 32,
            kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d
            (32, affine=False), nn.ReLU(), nn.Conv2d(32, 64, kernel_size=3,
            stride=2, padding=1, bias=False), nn.BatchNorm2d(64, affine=
            False), nn.ReLU(), nn.Conv2d(64, 64, kernel_size=3, stride=1,
            padding=1, bias=False), nn.ReLU(), nn.Dropout(0.25), nn.Conv2d(
            64, 4, kernel_size=8, stride=1, padding=0, bias=True), nn.
            AdaptiveAvgPool2d(1))
        self.PS = PS
        self.features.apply(self.weights_init)
        self.halfPS = int(PS / 2)
        return

    def input_norm(self, x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1).detach()
        sp = torch.std(flat, dim=1).detach() + 1e-07
        return (x - mp.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)
            ) / sp.unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.orthogonal(m.weight.data, gain=0.8)
            try:
                if m.weight.data.shape[-1] == 8:
                    nn.init.orthogonal(m.weight.data, gain=1.0)
                    None
                    m.bias.data = torch.FloatTensor([1, 0, 0, 1])
                else:
                    nn.init.constant(m.bias.data, 0.01)
            except:
                pass
        return

    def forward(self, input, return_A_matrix=False):
        xy = self.features(self.input_norm(input)).view(-1, 2, 2).contiguous()
        return rectifyAffineTransformationUpIsUp(xy).contiguous()


class AffNetFast(nn.Module):

    def __init__(self, PS=32):
        super(AffNetFast, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3,
            padding=1, bias=False), nn.BatchNorm2d(16, affine=False), nn.
            ReLU(), nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1,
            bias=False), nn.BatchNorm2d(16, affine=False), nn.ReLU(), nn.
            Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False), nn.ReLU(), nn.Conv2d(32, 32,
            kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d
            (32, affine=False), nn.ReLU(), nn.Conv2d(32, 64, kernel_size=3,
            stride=2, padding=1, bias=False), nn.BatchNorm2d(64, affine=
            False), nn.ReLU(), nn.Conv2d(64, 64, kernel_size=3, stride=1,
            padding=1, bias=False), nn.BatchNorm2d(64, affine=False), nn.
            ReLU(), nn.Dropout(0.25), nn.Conv2d(64, 3, kernel_size=8,
            stride=1, padding=0, bias=True), nn.Tanh(), nn.AdaptiveAvgPool2d(1)
            )
        self.PS = PS
        self.features.apply(self.weights_init)
        self.halfPS = int(PS / 2)
        return

    def input_norm(self, x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1).detach()
        sp = torch.std(flat, dim=1).detach() + 1e-07
        return (x - mp.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)
            ) / sp.unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.orthogonal(m.weight.data, gain=0.8)
            try:
                nn.init.constant(m.bias.data, 0.01)
            except:
                pass
        return

    def forward(self, input, return_A_matrix=False):
        xy = self.features(self.input_norm(input)).view(-1, 3)
        a1 = torch.cat([1.0 + xy[:, (0)].contiguous().view(-1, 1, 1), 0 *
            xy[:, (0)].contiguous().view(-1, 1, 1)], dim=2).contiguous()
        a2 = torch.cat([xy[:, (1)].contiguous().view(-1, 1, 1), 1.0 + xy[:,
            (2)].contiguous().view(-1, 1, 1)], dim=2).contiguous()
        return rectifyAffineTransformationUpIsUp(torch.cat([a1, a2], dim=1)
            .contiguous())


class AffNetFast52RotUp(nn.Module):

    def __init__(self, PS=32):
        super(AffNetFast52RotUp, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3,
            padding=1, bias=False), nn.BatchNorm2d(16, affine=False), nn.
            ReLU(), nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1,
            bias=False), nn.BatchNorm2d(16, affine=False), nn.ReLU(), nn.
            Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False), nn.ReLU(), nn.Conv2d(32, 32,
            kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d
            (32, affine=False), nn.ReLU(), nn.Conv2d(32, 64, kernel_size=3,
            stride=2, padding=1, bias=False), nn.BatchNorm2d(64, affine=
            False), nn.ReLU(), nn.Conv2d(64, 64, kernel_size=3, stride=1,
            padding=1, bias=False), nn.ReLU(), nn.Dropout(0.25), nn.Conv2d(
            64, 5, kernel_size=8, stride=1, padding=0, bias=True), nn.
            AdaptiveAvgPool2d(1))
        self.PS = PS
        self.features.apply(self.weights_init)
        self.halfPS = int(PS / 2)
        return

    def input_norm(self, x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1).detach()
        sp = torch.std(flat, dim=1).detach() + 1e-07
        return (x - mp.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)
            ) / sp.unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.orthogonal(m.weight.data, gain=0.8)
            try:
                if m.weight.data.shape[-1] == 8:
                    nn.init.orthogonal(m.weight.data, gain=1.0)
                    None
                    m.bias.data = torch.FloatTensor([1, 0, 1, 0, 1])
                else:
                    nn.init.constant(m.bias.data, 0.01)
            except:
                pass
        return

    def forward(self, input, return_A_matrix=False):
        x = self.features(self.input_norm(input)).view(-1, 5)
        angle = torch.atan2(x[:, (3)], x[:, (4)] + 1e-08)
        rot = get_rotation_matrix(angle)
        return torch.bmm(rot, rectifyAffineTransformationUpIsUp(torch.cat([
            torch.cat([x[:, 0:1].view(-1, 1, 1), x[:, 1:2].view(x.size(0), 
            1, 1).contiguous()], dim=2), x[:, 1:3].view(-1, 1, 2).
            contiguous()], dim=1)).contiguous())


class AffNetFast52Rot(nn.Module):

    def __init__(self, PS=32):
        super(AffNetFast52Rot, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3,
            padding=1, bias=False), nn.BatchNorm2d(16, affine=False), nn.
            ReLU(), nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1,
            bias=False), nn.BatchNorm2d(16, affine=False), nn.ReLU(), nn.
            Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False), nn.ReLU(), nn.Conv2d(32, 32,
            kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d
            (32, affine=False), nn.ReLU(), nn.Conv2d(32, 64, kernel_size=3,
            stride=2, padding=1, bias=False), nn.BatchNorm2d(64, affine=
            False), nn.ReLU(), nn.Conv2d(64, 64, kernel_size=3, stride=1,
            padding=1, bias=False), nn.ReLU(), nn.Dropout(0.25), nn.Conv2d(
            64, 5, kernel_size=8, stride=1, padding=0, bias=True), nn.
            AdaptiveAvgPool2d(1), nn.Tanh())
        self.PS = PS
        self.features.apply(self.weights_init)
        self.halfPS = int(PS / 2)
        return

    def input_norm(self, x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1).detach()
        sp = torch.std(flat, dim=1).detach() + 1e-07
        return (x - mp.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)
            ) / sp.unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.orthogonal(m.weight.data, gain=0.8)
            try:
                if m.weight.data.shape[-1] == 8:
                    nn.init.orthogonal(m.weight.data, gain=1.0)
                    None
                    m.bias.data = torch.FloatTensor([0.8, 0, 0.8, 0, 1])
                else:
                    nn.init.constant(m.bias.data, 0.01)
            except:
                pass
        return

    def forward(self, input, return_A_matrix=False):
        x = self.features(self.input_norm(input)).view(-1, 5)
        angle = torch.atan2(x[:, (3)], x[:, (4)] + 1e-08)
        rot = get_rotation_matrix(angle)
        return torch.bmm(rot, torch.cat([torch.cat([x[:, 0:1].view(-1, 1, 1
            ), x[:, 1:2].view(x.size(0), 1, 1).contiguous()], dim=2), x[:, 
            1:3].view(-1, 1, 2).contiguous()], dim=1))


class AffNetFast5Rot(nn.Module):

    def __init__(self, PS=32):
        super(AffNetFast5Rot, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3,
            padding=1, bias=False), nn.BatchNorm2d(16, affine=False), nn.
            ReLU(), nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1,
            bias=False), nn.BatchNorm2d(16, affine=False), nn.ReLU(), nn.
            Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False), nn.ReLU(), nn.Conv2d(32, 32,
            kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d
            (32, affine=False), nn.ReLU(), nn.Conv2d(32, 64, kernel_size=3,
            stride=2, padding=1, bias=False), nn.BatchNorm2d(64, affine=
            False), nn.ReLU(), nn.Conv2d(64, 64, kernel_size=3, stride=1,
            padding=1, bias=False), nn.ReLU(), nn.Dropout(0.25), nn.Conv2d(
            64, 5, kernel_size=8, stride=1, padding=0, bias=True), nn.
            AdaptiveAvgPool2d(1))
        self.PS = PS
        self.features.apply(self.weights_init)
        self.halfPS = int(PS / 2)
        return

    def input_norm(self, x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1).detach()
        sp = torch.std(flat, dim=1).detach() + 1e-07
        return (x - mp.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)
            ) / sp.unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.orthogonal(m.weight.data, gain=0.8)
            try:
                if m.weight.data.shape[-1] == 8:
                    nn.init.orthogonal(m.weight.data, gain=1.0)
                    None
                    m.bias.data = torch.FloatTensor([1, 0, 1, 0, 1])
                else:
                    nn.init.constant(m.bias.data, 0.01)
            except:
                pass
        return

    def forward(self, input, return_A_matrix=False):
        x = self.features(self.input_norm(input)).view(-1, 5)
        rot = get_rotation_matrix(torch.atan2(x[:, (3)], x[:, (4)] + 1e-08))
        if input.is_cuda:
            return torch.bmm(rot, torch.cat([torch.cat([x[:, 0:1].view(-1, 
                1, 1), torch.zeros(x.size(0), 1, 1)], dim=2), x[:, 1:3].
                view(-1, 1, 2).contiguous()], dim=1))
        else:
            return torch.bmm(rot, torch.cat([torch.cat([x[:, 0:1].view(-1, 
                1, 1), torch.zeros(x.size(0), 1, 1)], dim=2), x[:, 1:3].
                view(-1, 1, 2).contiguous()], dim=1))


class AffNetFast4Rot(nn.Module):

    def __init__(self, PS=32):
        super(AffNetFast4Rot, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3,
            padding=1, bias=False), nn.BatchNorm2d(16, affine=False), nn.
            ReLU(), nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1,
            bias=False), nn.BatchNorm2d(16, affine=False), nn.ReLU(), nn.
            Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False), nn.ReLU(), nn.Conv2d(32, 32,
            kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d
            (32, affine=False), nn.ReLU(), nn.Conv2d(32, 64, kernel_size=3,
            stride=2, padding=1, bias=False), nn.BatchNorm2d(64, affine=
            False), nn.ReLU(), nn.Conv2d(64, 64, kernel_size=3, stride=1,
            padding=1, bias=False), nn.ReLU(), nn.Dropout(0.25), nn.Conv2d(
            64, 4, kernel_size=8, stride=1, padding=0, bias=True), nn.
            AdaptiveAvgPool2d(1), nn.Tanh())
        self.PS = PS
        self.features.apply(self.weights_init)
        self.halfPS = int(PS / 2)
        return

    def input_norm(self, x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1).detach()
        sp = torch.std(flat, dim=1).detach() + 1e-07
        return (x - mp.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)
            ) / sp.unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.orthogonal(m.weight.data, gain=0.8)
            try:
                if m.weight.data.shape[-1] == 8:
                    nn.init.orthogonal(m.weight.data, gain=1.0)
                    None
                    m.bias.data = torch.FloatTensor([0.8, 0, 0, 0.8])
                else:
                    nn.init.constant(m.bias.data, 0.01)
            except:
                pass
        return

    def forward(self, input, return_A_matrix=False):
        return self.features(self.input_norm(input)).view(-1, 2, 2).contiguous(
            )


class AffNetFast4RotNosc(nn.Module):

    def __init__(self, PS=32):
        super(AffNetFast4RotNosc, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3,
            padding=1, bias=False), nn.BatchNorm2d(16, affine=False), nn.
            ReLU(), nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1,
            bias=False), nn.BatchNorm2d(16, affine=False), nn.ReLU(), nn.
            Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False), nn.ReLU(), nn.Conv2d(32, 32,
            kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d
            (32, affine=False), nn.ReLU(), nn.Conv2d(32, 64, kernel_size=3,
            stride=2, padding=1, bias=False), nn.BatchNorm2d(64, affine=
            False), nn.ReLU(), nn.Conv2d(64, 64, kernel_size=3, stride=1,
            padding=1, bias=False), nn.ReLU(), nn.Dropout(0.25), nn.Conv2d(
            64, 4, kernel_size=8, stride=1, padding=0, bias=True), nn.
            AdaptiveAvgPool2d(1))
        self.PS = PS
        self.features.apply(self.weights_init)
        self.halfPS = int(PS / 2)
        return

    def input_norm(self, x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1).detach()
        sp = torch.std(flat, dim=1).detach() + 1e-07
        return (x - mp.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)
            ) / sp.unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.orthogonal(m.weight.data, gain=0.8)
            try:
                if m.weight.data.shape[-1] == 8:
                    nn.init.orthogonal(m.weight.data, gain=1.0)
                    None
                    m.bias.data = torch.FloatTensor([1, 0, 0, 1])
                else:
                    nn.init.constant(m.bias.data, 0.01)
            except:
                pass
        return

    def forward(self, input, return_A_matrix=False):
        A = self.features(self.input_norm(input)).view(-1, 2, 2).contiguous()
        scale = torch.sqrt(torch.abs(A[:, (0), (0)] * A[:, (1), (1)] - A[:,
            (1), (0)] * A[:, (0), (1)] + 1e-10))
        return A / (scale.view(-1, 1, 1).repeat(1, 2, 2) + 1e-08)


class AffNetFastScale(nn.Module):

    def __init__(self, PS=32):
        super(AffNetFastScale, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3,
            padding=1, bias=False), nn.BatchNorm2d(16, affine=False), nn.
            ReLU(), nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1,
            bias=False), nn.BatchNorm2d(16, affine=False), nn.ReLU(), nn.
            Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False), nn.ReLU(), nn.Conv2d(32, 32,
            kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d
            (32, affine=False), nn.ReLU(), nn.Conv2d(32, 64, kernel_size=3,
            stride=2, padding=1, bias=False), nn.BatchNorm2d(64, affine=
            False), nn.ReLU(), nn.Conv2d(64, 64, kernel_size=3, stride=1,
            padding=1, bias=False), nn.BatchNorm2d(64, affine=False), nn.
            ReLU(), nn.Dropout(0.25), nn.Conv2d(64, 4, kernel_size=8,
            stride=1, padding=0, bias=True), nn.Tanh(), nn.AdaptiveAvgPool2d(1)
            )
        self.PS = PS
        self.features.apply(self.weights_init)
        self.halfPS = int(PS / 2)
        return

    def input_norm(self, x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1).detach()
        sp = torch.std(flat, dim=1).detach() + 1e-07
        return (x - mp.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)
            ) / sp.unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.orthogonal(m.weight.data, gain=0.8)
            try:
                nn.init.constant(m.bias.data, 0.01)
            except:
                pass
        return

    def forward(self, input, return_A_matrix=False):
        xy = self.features(self.input_norm(input)).view(-1, 4)
        a1 = torch.cat([1.0 + xy[:, (0)].contiguous().view(-1, 1, 1), 0 *
            xy[:, (0)].contiguous().view(-1, 1, 1)], dim=2).contiguous()
        a2 = torch.cat([xy[:, (1)].contiguous().view(-1, 1, 1), 1.0 + xy[:,
            (2)].contiguous().view(-1, 1, 1)], dim=2).contiguous()
        scale = torch.exp(xy[:, (3)].contiguous().view(-1, 1, 1).repeat(1, 
            2, 2))
        return scale * rectifyAffineTransformationUpIsUp(torch.cat([a1, a2],
            dim=1).contiguous())


class AffNetFast2Par(nn.Module):

    def __init__(self, PS=32):
        super(AffNetFast2Par, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3,
            padding=1, bias=False), nn.BatchNorm2d(16, affine=False), nn.
            ReLU(), nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1,
            bias=False), nn.BatchNorm2d(16, affine=False), nn.ReLU(), nn.
            Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False), nn.ReLU(), nn.Conv2d(32, 32,
            kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d
            (32, affine=False), nn.ReLU(), nn.Conv2d(32, 64, kernel_size=3,
            stride=2, padding=1, bias=False), nn.BatchNorm2d(64, affine=
            False), nn.ReLU(), nn.Conv2d(64, 64, kernel_size=3, stride=1,
            padding=1, bias=False), nn.ReLU(), nn.Dropout(0.25), nn.Conv2d(
            64, 3, kernel_size=8, stride=1, padding=0, bias=True), nn.
            AdaptiveAvgPool2d(1))
        self.PS = PS
        self.features.apply(self.weights_init)
        self.halfPS = int(PS / 2)
        return

    def input_norm(self, x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1).detach()
        sp = torch.std(flat, dim=1).detach() + 1e-07
        return (x - mp.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)
            ) / sp.unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.orthogonal(m.weight.data, gain=0.8)
            try:
                if m.weight.data.shape[-1] == 8:
                    nn.init.orthogonal(m.weight.data, gain=1.0)
                    None
                    m.bias.data = torch.FloatTensor([0, 0, 1])
                else:
                    nn.init.constant(m.bias.data, 0.01)
            except:
                pass
        return

    def forward(self, input, return_A_matrix=False):
        x = self.features(self.input_norm(input)).view(-1, 3)
        angle = torch.atan2(x[:, (1)], x[:, (2)] + 1e-08)
        rot = get_rotation_matrix(angle)
        tilt = torch.exp(1.8 * F.tanh(x[:, (0)]))
        tilt_matrix = torch.eye(2).unsqueeze(0).repeat(input.size(0), 1, 1)
        if x.is_cuda:
            tilt_matrix = tilt_matrix
        tilt_matrix[:, (0), (0)] = torch.sqrt(tilt)
        tilt_matrix[:, (1), (1)] = 1.0 / torch.sqrt(tilt)
        return rectifyAffineTransformationUpIsUp(torch.bmm(rot, tilt_matrix)
            ).contiguous()


def rectifyAffineTransformationUpIsUpFullyConv(A):
    det = torch.sqrt(torch.abs(A[:, 0:1, :, :] * A[:, 3:4, :, :] - A[:, 1:2,
        :, :] * A[:, 2:3, :, :] + 1e-10))
    b2a2 = torch.sqrt(A[:, 1:2, :, :] * A[:, 1:2, :, :] + A[:, 0:1, :, :] *
        A[:, 0:1, :, :])
    return torch.cat([(b2a2 / det).contiguous(), 0 * det.contiguous(), (A[:,
        3:4, :, :] * A[:, 1:2, :, :] + A[:, 2:3, :, :] * A[:, 0:1, :, :]) /
        (b2a2 * det), (det / b2a2).contiguous()], dim=1)


class AffNetFastFullConv(nn.Module):

    def __init__(self, PS=32, stride=2):
        super(AffNetFastFullConv, self).__init__()
        self.lrn = LocalNorm2d(33)
        self.features = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3,
            padding=1, bias=False), nn.BatchNorm2d(16, affine=False), nn.
            ReLU(), nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1,
            bias=False), nn.BatchNorm2d(16, affine=False), nn.ReLU(), nn.
            Conv2d(16, 32, kernel_size=3, stride=stride, padding=1, bias=
            False), nn.BatchNorm2d(32, affine=False), nn.ReLU(), nn.Conv2d(
            32, 32, kernel_size=3, stride=1, padding=1, bias=False), nn.
            BatchNorm2d(32, affine=False), nn.ReLU(), nn.Conv2d(32, 64,
            kernel_size=3, stride=stride, padding=1, bias=False), nn.
            BatchNorm2d(64, affine=False), nn.ReLU(), nn.Conv2d(64, 64,
            kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d
            (64, affine=False), nn.ReLU(), nn.Dropout(0.25), nn.Conv2d(64, 
            3, kernel_size=8, stride=1, padding=0, bias=True))
        self.stride = stride
        self.PS = PS
        self.features.apply(self.weights_init)
        self.halfPS = int(PS / 2)
        return

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.orthogonal(m.weight.data, gain=0.8)
            try:
                nn.init.constant(m.bias.data, 0.01)
            except:
                pass
        return

    def forward(self, input, return_A_matrix=False):
        norm_inp = self.lrn(input)
        ff = self.features(F.pad(norm_inp, (14, 14, 14, 14), 'reflect'))
        xy = F.tanh(F.upsample(ff, (input.size(2), input.size(3)), mode=
            'bilinear'))
        a0bc = torch.cat([1.0 + xy[:, 0:1, :, :].contiguous(), 0 * xy[:, 1:
            2, :, :].contiguous(), xy[:, 1:2, :, :].contiguous(), 1.0 + xy[
            :, 2:, :, :].contiguous()], dim=1).contiguous()
        return rectifyAffineTransformationUpIsUpFullyConv(a0bc).contiguous()


class AffNetFast52RotL(nn.Module):

    def __init__(self, PS=32):
        super(AffNetFast52RotL, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3,
            padding=1, bias=False), nn.BatchNorm2d(16, affine=False), nn.
            ReLU(), nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1,
            bias=False), nn.BatchNorm2d(16, affine=False), nn.ReLU(), nn.
            Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False), nn.ReLU(), nn.Conv2d(32, 32,
            kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d
            (32, affine=False), nn.ReLU(), nn.Conv2d(32, 64, kernel_size=3,
            stride=2, padding=1, bias=False), nn.BatchNorm2d(64, affine=
            False), nn.ReLU(), nn.Conv2d(64, 64, kernel_size=3, stride=1,
            padding=1, bias=False), nn.ReLU(), nn.Dropout(0.25), nn.Conv2d(
            64, 5, kernel_size=8, stride=1, padding=0, bias=True), nn.
            AdaptiveAvgPool2d(1))
        self.PS = PS
        self.features.apply(self.weights_init)
        self.halfPS = int(PS / 2)
        return

    def input_norm(self, x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1).detach()
        sp = torch.std(flat, dim=1).detach() + 1e-07
        return (x - mp.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)
            ) / sp.unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.orthogonal(m.weight.data, gain=0.8)
            try:
                if m.weight.data.shape[-1] == 8:
                    nn.init.orthogonal(m.weight.data, gain=1.0)
                    None
                    m.bias.data = torch.FloatTensor([0.8, 0, 0.8, 0, 1])
                else:
                    nn.init.constant(m.bias.data, 0.01)
            except:
                pass
        return

    def forward(self, input, return_A_matrix=False):
        x = self.features(self.input_norm(input)).view(-1, 5)
        angle = torch.atan2(x[:, (3)], x[:, (4)] + 1e-08)
        rot = get_rotation_matrix(angle)
        return torch.bmm(rot, torch.cat([torch.cat([x[:, 0:1].view(-1, 1, 1
            ), x[:, 1:2].view(x.size(0), 1, 1).contiguous()], dim=2), x[:, 
            1:3].view(-1, 1, 2).contiguous()], dim=1))


class AffNetFastBias(nn.Module):

    def __init__(self, PS=32):
        super(AffNetFastBias, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3,
            padding=1, bias=False), nn.BatchNorm2d(16, affine=False), nn.
            ReLU(), nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1,
            bias=False), nn.BatchNorm2d(16, affine=False), nn.ReLU(), nn.
            Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False), nn.ReLU(), nn.Conv2d(32, 32,
            kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d
            (32, affine=False), nn.ReLU(), nn.Conv2d(32, 64, kernel_size=3,
            stride=2, padding=1, bias=False), nn.BatchNorm2d(64, affine=
            False), nn.ReLU(), nn.Conv2d(64, 64, kernel_size=3, stride=1,
            padding=1, bias=False), nn.BatchNorm2d(64, affine=False), nn.
            ReLU(), nn.Dropout(0.25), nn.Conv2d(64, 3, kernel_size=8,
            stride=1, padding=0, bias=True), nn.Tanh(), nn.AdaptiveAvgPool2d(1)
            )
        self.PS = PS
        self.features.apply(self.weights_init)
        self.halfPS = int(PS / 2)
        return

    def input_norm(self, x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1).detach()
        sp = torch.std(flat, dim=1).detach() + 1e-07
        return (x - mp.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)
            ) / sp.unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.orthogonal(m.weight.data, gain=0.8)
            try:
                if m.weight.data.shape[-1] == 8:
                    nn.init.orthogonal(m.weight.data, gain=1.0)
                    None
                    m.bias.data = torch.FloatTensor([0.8, 0, 0.8])
                else:
                    nn.init.constant(m.bias.data, 0.01)
            except:
                pass
        return

    def forward(self, input, return_A_matrix=False):
        xy = self.features(self.input_norm(input)).view(-1, 3)
        a1 = torch.cat([xy[:, (0)].contiguous().view(-1, 1, 1), 0 * xy[:, (
            0)].contiguous().view(-1, 1, 1)], dim=2).contiguous()
        a2 = torch.cat([xy[:, (1)].contiguous().view(-1, 1, 1), xy[:, (2)].
            contiguous().view(-1, 1, 1)], dim=2).contiguous()
        return rectifyAffineTransformationUpIsUp(torch.cat([a1, a2], dim=1)
            .contiguous())


class L2Norm(nn.Module):

    def __init__(self):
        super(L2Norm, self).__init__()
        self.eps = 1e-10

    def forward(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim=1) + self.eps)
        x = x / norm.unsqueeze(-1).expand_as(x)
        return x


class L1Norm(nn.Module):

    def __init__(self):
        super(L1Norm, self).__init__()
        self.eps = 1e-10

    def forward(self, x):
        norm = torch.sum(torch.abs(x), dim=1) + self.eps
        x = x / norm.expand_as(x)
        return x


class GaussianBlur(nn.Module):

    def __init__(self, sigma=1.6):
        super(GaussianBlur, self).__init__()
        weight = self.calculate_weights(sigma)
        self.register_buffer('buf', weight)
        return

    def calculate_weights(self, sigma):
        kernel = CircularGaussKernel(sigma=sigma, circ_zeros=False)
        h, w = kernel.shape
        halfSize = float(h) / 2.0
        self.pad = int(np.floor(halfSize))
        return torch.from_numpy(kernel.astype(np.float32)).view(1, 1, h, w)

    def forward(self, x):
        w = Variable(self.buf)
        if x.is_cuda:
            w = w
        return F.conv2d(F.pad(x, (self.pad, self.pad, self.pad, self.pad),
            'replicate'), w, padding=0)


class L2Norm(nn.Module):

    def __init__(self):
        super(L2Norm, self).__init__()
        self.eps = 1e-10

    def forward(self, x):
        norm = torch.sqrt(torch.abs(torch.sum(x * x, dim=1)) + self.eps)
        x = x / norm.unsqueeze(1).expand_as(x)
        return x


def get_bin_weight_kernel_size_and_stride(patch_size, num_spatial_bins):
    bin_weight_stride = int(round(2.0 * math.floor(patch_size / 2) / float(
        num_spatial_bins + 1)))
    bin_weight_kernel_size = int(2 * bin_weight_stride - 1)
    return bin_weight_kernel_size, bin_weight_stride


def getPoolingKernel(kernel_size=25):
    step = 1.0 / float(np.floor(kernel_size / 2.0))
    x_coef = np.arange(step / 2.0, 1.0, step)
    xc2 = np.hstack([x_coef, [1], x_coef[::-1]])
    kernel = np.outer(xc2.T, xc2)
    kernel = np.maximum(0, kernel)
    return kernel


class SIFTNet(nn.Module):

    def CircularGaussKernel(self, kernlen=21):
        halfSize = kernlen / 2
        r2 = float(halfSize * halfSize)
        sigma2 = 0.9 * r2
        disq = 0
        kernel = np.zeros((kernlen, kernlen))
        for y in range(kernlen):
            for x in range(kernlen):
                disq = (y - halfSize) * (y - halfSize) + (x - halfSize) * (x -
                    halfSize)
                if disq < r2:
                    kernel[y, x] = math.exp(-disq / sigma2)
                else:
                    kernel[y, x] = 0.0
        return kernel

    def __init__(self, patch_size=65, num_ang_bins=8, num_spatial_bins=4,
        clipval=0.2):
        super(SIFTNet, self).__init__()
        gk = torch.from_numpy(self.CircularGaussKernel(kernlen=patch_size).
            astype(np.float32))
        self.bin_weight_kernel_size, self.bin_weight_stride = (
            get_bin_weight_kernel_size_and_stride(patch_size, num_spatial_bins)
            )
        self.gk = Variable(gk)
        self.num_ang_bins = num_ang_bins
        self.num_spatial_bins = num_spatial_bins
        self.clipval = clipval
        self.gx = nn.Sequential(nn.Conv2d(1, 1, kernel_size=(1, 3), bias=False)
            )
        for l in self.gx:
            if isinstance(l, nn.Conv2d):
                l.weight.data = torch.from_numpy(np.array([[[[-1, 0, 1]]]],
                    dtype=np.float32))
        self.gy = nn.Sequential(nn.Conv2d(1, 1, kernel_size=(3, 1), bias=False)
            )
        for l in self.gy:
            if isinstance(l, nn.Conv2d):
                l.weight.data = torch.from_numpy(np.array([[[[-1], [0], [1]
                    ]]], dtype=np.float32))
        self.pk = nn.Sequential(nn.Conv2d(1, 1, kernel_size=(self.
            bin_weight_kernel_size, self.bin_weight_kernel_size), stride=(
            self.bin_weight_stride, self.bin_weight_stride), bias=False))
        for l in self.pk:
            if isinstance(l, nn.Conv2d):
                nw = getPoolingKernel(kernel_size=self.bin_weight_kernel_size)
                new_weights = np.array(nw.reshape((1, 1, self.
                    bin_weight_kernel_size, self.bin_weight_kernel_size)))
                l.weight.data = torch.from_numpy(new_weights.astype(np.float32)
                    )

    def forward(self, x):
        gx = self.gx(F.pad(x, (1, 1, 0, 0), 'replicate'))
        gy = self.gy(F.pad(x, (0, 0, 1, 1), 'replicate'))
        mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-10)
        ori = torch.atan2(gy, gx + 1e-08)
        if x.is_cuda:
            self.gk = self.gk
        else:
            self.gk = self.gk.cpu()
        mag = mag * self.gk.expand_as(mag)
        o_big = (ori + 2.0 * math.pi) / (2.0 * math.pi) * float(self.
            num_ang_bins)
        bo0_big = torch.floor(o_big)
        wo1_big = o_big - bo0_big
        bo0_big = bo0_big % self.num_ang_bins
        bo1_big = (bo0_big + 1) % self.num_ang_bins
        wo0_big = (1.0 - wo1_big) * mag
        wo1_big = wo1_big * mag
        ang_bins = []
        for i in range(0, self.num_ang_bins):
            ang_bins.append(self.pk((bo0_big == i).float() * wo0_big + (
                bo1_big == i).float() * wo1_big))
        ang_bins = torch.cat(ang_bins, 1)
        ang_bins = ang_bins.view(ang_bins.size(0), -1)
        ang_bins = L2Norm()(ang_bins)
        ang_bins = torch.clamp(ang_bins, 0.0, float(self.clipval))
        ang_bins = L2Norm()(ang_bins)
        return ang_bins


class L2Norm(nn.Module):

    def __init__(self):
        super(L2Norm, self).__init__()
        self.eps = 1e-08

    def forward(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim=1) + self.eps)
        x = x / norm.unsqueeze(-1).expand_as(x)
        return x


class L1Norm(nn.Module):

    def __init__(self):
        super(L1Norm, self).__init__()
        self.eps = 1e-10

    def forward(self, x):
        norm = torch.sum(torch.abs(x), dim=1) + self.eps
        x = x / norm.expand_as(x)
        return x


class HardNetNarELU(nn.Module):
    """TFeat model definition
    """

    def __init__(self, sm):
        super(HardNetNarELU, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3,
            padding=1), nn.ELU(), nn.Conv2d(16, 16, kernel_size=3, padding=
            1), nn.ELU(), nn.Conv2d(16, 32, kernel_size=3, stride=2,
            padding=1), nn.ELU(), nn.Conv2d(32, 32, kernel_size=3, padding=
            1), nn.ELU(), nn.Conv2d(32, 64, kernel_size=3, stride=2,
            padding=1), nn.ELU(), nn.Conv2d(64, 64, kernel_size=3, padding=
            1), nn.ELU())
        self.classifier = nn.Sequential(nn.Dropout(0.1), nn.Conv2d(64, 128,
            kernel_size=8), nn.BatchNorm2d(128, affine=False))
        self.SIFT = sm
        return

    def input_norm(self, x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-07
        return (x - mp.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)
            ) / sp.unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)

    def forward(self, input):
        x_features = self.features(input)
        x = nn.AdaptiveAvgPool2d(1)(x_features).view(x_features.size(0), -1)
        return x


class HardNet(nn.Module):
    """HardNet model definition
    """

    def __init__(self):
        super(HardNet, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(1, 32, kernel_size=3,
            padding=1, bias=False), nn.BatchNorm2d(32, affine=False), nn.
            ReLU(), nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False), nn.ReLU(), nn.Conv2d(32, 64,
            kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d
            (64, affine=False), nn.ReLU(), nn.Conv2d(64, 64, kernel_size=3,
            padding=1, bias=False), nn.BatchNorm2d(64, affine=False), nn.
            ReLU(), nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1,
            bias=False), nn.BatchNorm2d(128, affine=False), nn.ReLU(), nn.
            Conv2d(128, 128, kernel_size=3, padding=1, bias=False), nn.
            BatchNorm2d(128, affine=False), nn.ReLU(), nn.Dropout(0.1), nn.
            Conv2d(128, 128, kernel_size=8, bias=False), nn.BatchNorm2d(128,
            affine=False))

    def input_norm(self, x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-07
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).
            expand_as(x)) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1
            ).expand_as(x)

    def forward(self, input):
        x_features = self.features(self.input_norm(input))
        x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x)


class NMS2d(nn.Module):

    def __init__(self, kernel_size=3, threshold=0):
        super(NMS2d, self).__init__()
        self.MP = nn.MaxPool2d(kernel_size, stride=1, return_indices=False,
            padding=kernel_size / 2)
        self.eps = 1e-05
        self.th = threshold
        return

    def forward(self, x):
        if self.th > self.eps:
            return x * (x > self.th).float() * (x + self.eps - self.MP(x) > 0
                ).float()
        else:
            return (x - self.MP(x) + self.eps > 0).float() * x


class NMS3d(nn.Module):

    def __init__(self, kernel_size=3, threshold=0):
        super(NMS3d, self).__init__()
        self.MP = nn.MaxPool3d(kernel_size, stride=1, return_indices=False,
            padding=(0, kernel_size / 2, kernel_size / 2))
        self.eps = 1e-05
        self.th = threshold
        return

    def forward(self, x):
        if self.th > self.eps:
            return x * (x > self.th).float() * (x + self.eps - self.MP(x) > 0
                ).float()
        else:
            return (x - self.MP(x) + self.eps > 0).float() * x


class NMS3dAndComposeA(nn.Module):

    def __init__(self, kernel_size=3, threshold=0, scales=None, border=3,
        mrSize=1.0):
        super(NMS3dAndComposeA, self).__init__()
        self.eps = 1e-07
        self.ks = 3
        if type(scales) is not list:
            self.grid = generate_3dgrid(3, self.ks, self.ks)
        else:
            self.grid = generate_3dgrid(scales, self.ks, self.ks)
        self.grid = Variable(self.grid.t().contiguous().view(3, 3, 3, 3),
            requires_grad=False)
        self.th = threshold
        self.cube_idxs = []
        self.border = border
        self.mrSize = mrSize
        self.beta = 1.0
        self.grid_ones = Variable(torch.ones(3, 3, 3, 3), requires_grad=False)
        self.NMS3d = NMS3d(kernel_size, threshold)
        return

    def forward(self, low, cur, high, octaveMap=None, num_features=0):
        assert low.size() == cur.size() == high.size()
        self.is_cuda = low.is_cuda
        resp3d = torch.cat([low, cur, high], dim=1)
        mrSize_border = int(self.mrSize)
        if octaveMap is not None:
            nmsed_resp = zero_response_at_border(self.NMS3d(resp3d.
                unsqueeze(1)).squeeze(1)[:, 1:2, :, :], mrSize_border) * (
                1.0 - octaveMap.float())
        else:
            nmsed_resp = zero_response_at_border(self.NMS3d(resp3d.
                unsqueeze(1)).squeeze(1)[:, 1:2, :, :], mrSize_border)
        num_of_nonzero_responces = (nmsed_resp > 0).sum().data[0]
        if num_of_nonzero_responces == 0:
            return None, None, None
        if octaveMap is not None:
            octaveMap = (octaveMap.float() + nmsed_resp.float()).byte()
        nmsed_resp = nmsed_resp.view(-1)
        if num_features > 0 and num_features < num_of_nonzero_responces:
            nmsed_resp, idxs = torch.topk(nmsed_resp, k=num_features)
        else:
            idxs = nmsed_resp.data.nonzero().squeeze()
            nmsed_resp = nmsed_resp[idxs]
        spatial_grid = Variable(generate_2dgrid(low.size(2), low.size(3), 
            False)).view(1, low.size(2), low.size(3), 2)
        spatial_grid = spatial_grid.permute(3, 1, 2, 0)
        if self.is_cuda:
            spatial_grid = spatial_grid
            self.grid = self.grid
            self.grid_ones = self.grid_ones
        sc_y_x = F.conv2d(resp3d, self.grid, padding=1) / (F.conv2d(resp3d,
            self.grid_ones, padding=1) + 1e-08)
        sc_y_x[(0), 1:, :, :] = sc_y_x[(0), 1:, :, :] + spatial_grid[:, :,
            :, (0)]
        sc_y_x = sc_y_x.view(3, -1).t()
        sc_y_x = sc_y_x[(idxs), :]
        min_size = float(min(cur.size(2), cur.size(3)))
        sc_y_x[:, (0)] = sc_y_x[:, (0)] / min_size
        sc_y_x[:, (1)] = sc_y_x[:, (1)] / float(cur.size(2))
        sc_y_x[:, (2)] = sc_y_x[:, (2)] / float(cur.size(3))
        return nmsed_resp, sc_y_x2LAFs(sc_y_x), octaveMap


class L2Norm(nn.Module):

    def __init__(self):
        super(L2Norm, self).__init__()
        self.eps = 1e-10

    def forward(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim=1) + self.eps)
        x = x / norm.unsqueeze(-1).expand_as(x)
        return x


class L1Norm(nn.Module):

    def __init__(self):
        super(L1Norm, self).__init__()
        self.eps = 1e-10

    def forward(self, x):
        norm = torch.sum(torch.abs(x), dim=1) + self.eps
        x = x / norm.expand_as(x)
        return x


class GaussianBlur(nn.Module):

    def __init__(self, sigma=1.6):
        super(GaussianBlur, self).__init__()
        weight = self.calculate_weights(sigma)
        self.register_buffer('buf', weight)
        return

    def calculate_weights(self, sigma):
        kernel = CircularGaussKernel(sigma=sigma, circ_zeros=False)
        h, w = kernel.shape
        halfSize = float(h) / 2.0
        self.pad = int(np.floor(halfSize))
        return torch.from_numpy(kernel.astype(np.float32)).view(1, 1, h, w)

    def forward(self, x):
        w = Variable(self.buf)
        if x.is_cuda:
            w = w
        return F.conv2d(F.pad(x, (self.pad, self.pad, self.pad, self.pad),
            'replicate'), w, padding=0)


class OriNetFast(nn.Module):

    def __init__(self, PS=16):
        super(OriNetFast, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3,
            padding=1, bias=False), nn.BatchNorm2d(16, affine=False), nn.
            ReLU(), nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1,
            bias=False), nn.BatchNorm2d(16, affine=False), nn.ReLU(), nn.
            Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False), nn.ReLU(), nn.Conv2d(32, 32,
            kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d
            (32, affine=False), nn.ReLU(), nn.Conv2d(32, 64, kernel_size=3,
            stride=2, padding=1, bias=False), nn.BatchNorm2d(64, affine=
            False), nn.ReLU(), nn.Conv2d(64, 64, kernel_size=3, stride=1,
            padding=1, bias=False), nn.BatchNorm2d(64, affine=False), nn.
            ReLU(), nn.Dropout(0.25), nn.Conv2d(64, 2, kernel_size=int(PS /
            4), stride=1, padding=1, bias=True), nn.Tanh(), nn.
            AdaptiveAvgPool2d(1))
        self.PS = PS
        self.features.apply(self.weights_init)
        self.halfPS = int(PS / 4)
        return

    def input_norm(self, x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-07
        return (x - mp.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)
            ) / sp.unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.orthogonal(m.weight.data, gain=0.9)
            try:
                nn.init.constant(m.bias.data, 0.01)
            except:
                pass
        return

    def forward(self, input, return_rot_matrix=True):
        xy = self.features(self.input_norm(input)).view(-1, 2)
        angle = torch.atan2(xy[:, (0)] + 1e-08, xy[:, (1)] + 1e-08)
        if return_rot_matrix:
            return get_rotation_matrix(angle)
        return angle


class GHH(nn.Module):

    def __init__(self, n_in, n_out, s=4, m=4):
        super(GHH, self).__init__()
        self.n_out = n_out
        self.s = s
        self.m = m
        self.conv = nn.Linear(n_in, n_out * s * m)
        d = torch.arange(0, s)
        self.deltas = -1.0 * (d % 2 != 0).float() + 1.0 * (d % 2 == 0).float()
        self.deltas = Variable(self.deltas)
        return

    def forward(self, x):
        x_feats = self.conv(x.view(x.size(0), -1)).view(x.size(0), self.
            n_out, self.s, self.m)
        max_feats = x_feats.max(dim=3)[0]
        if x.is_cuda:
            self.deltas = self.deltas
        else:
            self.deltas = self.deltas.cpu()
        out = (max_feats * self.deltas.view(1, 1, -1).expand_as(max_feats)
            ).sum(dim=2)
        return out


class YiNet(nn.Module):

    def __init__(self, PS=28):
        super(YiNet, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(1, 10, kernel_size=5,
            padding=0, bias=True), nn.ReLU(), nn.MaxPool2d(kernel_size=3,
            stride=2, padding=1), nn.Conv2d(10, 20, kernel_size=5, stride=1,
            padding=0, bias=True), nn.ReLU(), nn.MaxPool2d(kernel_size=4,
            stride=2, padding=2), nn.Conv2d(20, 50, kernel_size=3, stride=1,
            padding=0, bias=True), nn.ReLU(), nn.AdaptiveMaxPool2d(1), GHH(
            50, 100), GHH(100, 2))
        self.input_mean = 0.427117081207483
        self.input_std = 0.21888339179665006
        self.PS = PS
        return

    def import_weights(self, dir_name):
        self.features[0].weight.data = torch.from_numpy(np.load(os.path.
            join(dir_name, 'layer0_W.npy'))).float()
        self.features[0].bias.data = torch.from_numpy(np.load(os.path.join(
            dir_name, 'layer0_b.npy'))).float().view(-1)
        self.features[3].weight.data = torch.from_numpy(np.load(os.path.
            join(dir_name, 'layer1_W.npy'))).float()
        self.features[3].bias.data = torch.from_numpy(np.load(os.path.join(
            dir_name, 'layer1_b.npy'))).float().view(-1)
        self.features[6].weight.data = torch.from_numpy(np.load(os.path.
            join(dir_name, 'layer2_W.npy'))).float()
        self.features[6].bias.data = torch.from_numpy(np.load(os.path.join(
            dir_name, 'layer2_b.npy'))).float().view(-1)
        self.features[9].conv.weight.data = torch.from_numpy(np.load(os.
            path.join(dir_name, 'layer3_W.npy'))).float().view(50, 1600
            ).contiguous().t().contiguous()
        self.features[9].conv.bias.data = torch.from_numpy(np.load(os.path.
            join(dir_name, 'layer3_b.npy'))).float().view(1600)
        self.features[10].conv.weight.data = torch.from_numpy(np.load(os.
            path.join(dir_name, 'layer4_W.npy'))).float().view(100, 32
            ).contiguous().t().contiguous()
        self.features[10].conv.bias.data = torch.from_numpy(np.load(os.path
            .join(dir_name, 'layer4_b.npy'))).float().view(32)
        self.input_mean = float(np.load(os.path.join(dir_name,
            'input_mean.npy')))
        self.input_std = float(np.load(os.path.join(dir_name, 'input_std.npy'))
            )
        return

    def input_norm1(self, x):
        return (x - self.input_mean) / self.input_std

    def input_norm(self, x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-07
        return (x - mp.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)
            ) / sp.unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)

    def forward(self, input, return_rot_matrix=False):
        xy = self.features(self.input_norm(input))
        angle = torch.atan2(xy[:, (0)] + 1e-08, xy[:, (1)] + 1e-08)
        if return_rot_matrix:
            return get_rotation_matrix(-angle)
        return angle


class AffNetFast(nn.Module):

    def __init__(self, PS=32):
        super(AffNetFast, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3,
            padding=1, bias=False), nn.BatchNorm2d(16, affine=False), nn.
            ReLU(), nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1,
            bias=False), nn.BatchNorm2d(16, affine=False), nn.ReLU(), nn.
            Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False), nn.ReLU(), nn.Conv2d(32, 32,
            kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d
            (32, affine=False), nn.ReLU(), nn.Conv2d(32, 64, kernel_size=3,
            stride=2, padding=1, bias=False), nn.BatchNorm2d(64, affine=
            False), nn.ReLU(), nn.Conv2d(64, 64, kernel_size=3, stride=1,
            padding=1, bias=False), nn.BatchNorm2d(64, affine=False), nn.
            ReLU(), nn.Dropout(0.25), nn.Conv2d(64, 3, kernel_size=8,
            stride=1, padding=0, bias=True), nn.Tanh(), nn.AdaptiveAvgPool2d(1)
            )
        self.PS = PS
        self.features.apply(self.weights_init)
        self.halfPS = int(PS / 2)
        return

    def input_norm(self, x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-07
        return (x - mp.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)
            ) / sp.unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.orthogonal(m.weight.data, gain=0.8)
            try:
                nn.init.constant(m.bias.data, 0.01)
            except:
                pass
        return

    def forward(self, input, return_A_matrix=False):
        xy = self.features(self.input_norm(input)).view(-1, 3)
        a1 = torch.cat([1.0 + xy[:, (0)].contiguous().view(-1, 1, 1), 0 *
            xy[:, (0)].contiguous().view(-1, 1, 1)], dim=2).contiguous()
        a2 = torch.cat([xy[:, (1)].contiguous().view(-1, 1, 1), 1.0 + xy[:,
            (2)].contiguous().view(-1, 1, 1)], dim=2).contiguous()
        return rectifyAffineTransformationUpIsUp(torch.cat([a1, a2], dim=1)
            .contiguous())


class L2Norm(nn.Module):

    def __init__(self):
        super(L2Norm, self).__init__()
        self.eps = 1e-10

    def forward(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim=1) + self.eps)
        x = x / norm.unsqueeze(-1).expand_as(x)
        return x


class L1Norm(nn.Module):

    def __init__(self):
        super(L1Norm, self).__init__()
        self.eps = 1e-10

    def forward(self, x):
        norm = torch.sum(torch.abs(x), dim=1) + self.eps
        x = x / norm.expand_as(x)
        return x


class L2Norm(nn.Module):

    def __init__(self):
        super(L2Norm, self).__init__()
        self.eps = 1e-10

    def forward(self, x):
        norm = torch.sqrt(torch.abs(torch.sum(x * x, dim=1)) + self.eps)
        x = x / norm.unsqueeze(1).expand_as(x)
        return x


class SIFTNet(nn.Module):

    def CircularGaussKernel(self, kernlen=21):
        halfSize = kernlen / 2
        r2 = float(halfSize * halfSize)
        sigma2 = 0.9 * r2
        disq = 0
        kernel = np.zeros((kernlen, kernlen))
        for y in range(kernlen):
            for x in range(kernlen):
                disq = (y - halfSize) * (y - halfSize) + (x - halfSize) * (x -
                    halfSize)
                if disq < r2:
                    kernel[y, x] = math.exp(-disq / sigma2)
                else:
                    kernel[y, x] = 0.0
        return kernel

    def __init__(self, patch_size=65, num_ang_bins=8, num_spatial_bins=4,
        clipval=0.2):
        super(SIFTNet, self).__init__()
        gk = torch.from_numpy(self.CircularGaussKernel(kernlen=patch_size).
            astype(np.float32))
        self.bin_weight_kernel_size, self.bin_weight_stride = (
            get_bin_weight_kernel_size_and_stride(patch_size, num_spatial_bins)
            )
        self.gk = Variable(gk)
        self.num_ang_bins = num_ang_bins
        self.num_spatial_bins = num_spatial_bins
        self.clipval = clipval
        self.gx = nn.Sequential(nn.Conv2d(1, 1, kernel_size=(1, 3), bias=False)
            )
        for l in self.gx:
            if isinstance(l, nn.Conv2d):
                l.weight.data = torch.from_numpy(np.array([[[[-1, 0, 1]]]],
                    dtype=np.float32))
        self.gy = nn.Sequential(nn.Conv2d(1, 1, kernel_size=(3, 1), bias=False)
            )
        for l in self.gy:
            if isinstance(l, nn.Conv2d):
                l.weight.data = torch.from_numpy(np.array([[[[-1], [0], [1]
                    ]]], dtype=np.float32))
        self.pk = nn.Sequential(nn.Conv2d(1, 1, kernel_size=(self.
            bin_weight_kernel_size, self.bin_weight_kernel_size), stride=(
            self.bin_weight_stride, self.bin_weight_stride), bias=False))
        for l in self.pk:
            if isinstance(l, nn.Conv2d):
                nw = getPoolingKernel(kernel_size=self.bin_weight_kernel_size)
                new_weights = np.array(nw.reshape((1, 1, self.
                    bin_weight_kernel_size, self.bin_weight_kernel_size)))
                l.weight.data = torch.from_numpy(new_weights.astype(np.float32)
                    )

    def forward(self, x):
        gx = self.gx(F.pad(x, (1, 1, 0, 0), 'replicate'))
        gy = self.gy(F.pad(x, (0, 0, 1, 1), 'replicate'))
        mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-10)
        ori = torch.atan2(gy, gx + 1e-08)
        if x.is_cuda:
            self.gk = self.gk
        else:
            self.gk = self.gk.cpu()
        mag = mag * self.gk.expand_as(mag)
        o_big = (ori + 2.0 * math.pi) / (2.0 * math.pi) * float(self.
            num_ang_bins)
        bo0_big = torch.floor(o_big)
        wo1_big = o_big - bo0_big
        bo0_big = bo0_big % self.num_ang_bins
        bo1_big = (bo0_big + 1) % self.num_ang_bins
        wo0_big = (1.0 - wo1_big) * mag
        wo1_big = wo1_big * mag
        ang_bins = []
        for i in range(0, self.num_ang_bins):
            ang_bins.append(self.pk((bo0_big == i).float() * wo0_big + (
                bo1_big == i).float() * wo1_big))
        ang_bins = torch.cat(ang_bins, 1)
        ang_bins = ang_bins.view(ang_bins.size(0), -1)
        ang_bins = L2Norm()(ang_bins)
        ang_bins = torch.clamp(ang_bins, 0.0, float(self.clipval))
        ang_bins = L2Norm()(ang_bins)
        return ang_bins


class HardTFeatNet(nn.Module):
    """TFeat model definition
    """

    def __init__(self, sm):
        super(HardTFeatNet, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(1, 32, kernel_size=7), nn.
            Tanh(), nn.MaxPool2d(kernel_size=2, stride=2), nn.Conv2d(32, 64,
            kernel_size=6), nn.Tanh())
        self.classifier = nn.Sequential(nn.Dropout(0.1), nn.Conv2d(64, 128,
            kernel_size=8), nn.Tanh())
        self.SIFT = sm

    def input_norm(self, x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-07
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).
            expand_as(x)) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1
            ).expand_as(x)

    def forward(self, input):
        x_features = self.features(self.input_norm(input))
        x = x_features.view(x_features.size(0), -1)
        x = self.classifier(x_features)
        return x.view(x.size(0), -1)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_ducha_aiki_affnet(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(AffNetFast(*[], **{}), [torch.rand([4, 1, 64, 64])], {})

    @_fails_compile()
    def test_001(self):
        self._check(AffNetFast2Par(*[], **{}), [torch.rand([4, 1, 64, 64])], {})

    @_fails_compile()
    def test_002(self):
        self._check(AffNetFast4(*[], **{}), [torch.rand([4, 1, 64, 64])], {})

    @_fails_compile()
    def test_003(self):
        self._check(AffNetFast4Rot(*[], **{}), [torch.rand([4, 1, 64, 64])], {})

    @_fails_compile()
    def test_004(self):
        self._check(AffNetFast4RotNosc(*[], **{}), [torch.rand([4, 1, 64, 64])], {})

    @_fails_compile()
    def test_005(self):
        self._check(AffNetFast52Rot(*[], **{}), [torch.rand([4, 1, 64, 64])], {})

    @_fails_compile()
    def test_006(self):
        self._check(AffNetFast52RotL(*[], **{}), [torch.rand([4, 1, 64, 64])], {})

    @_fails_compile()
    def test_007(self):
        self._check(AffNetFast52RotUp(*[], **{}), [torch.rand([4, 1, 64, 64])], {})

    @_fails_compile()
    def test_008(self):
        self._check(AffNetFast5Rot(*[], **{}), [torch.rand([4, 1, 64, 64])], {})

    @_fails_compile()
    def test_009(self):
        self._check(AffNetFastBias(*[], **{}), [torch.rand([4, 1, 64, 64])], {})

    @_fails_compile()
    def test_010(self):
        self._check(AffNetFastScale(*[], **{}), [torch.rand([4, 1, 64, 64])], {})

    def test_011(self):
        self._check(GHH(*[], **{'n_in': 4, 'n_out': 4}), [torch.rand([4, 4])], {})

    @_fails_compile()
    def test_012(self):
        self._check(GaussianBlur(*[], **{}), [torch.rand([4, 1, 4, 4])], {})

    @_fails_compile()
    def test_013(self):
        self._check(HardNet(*[], **{}), [torch.rand([4, 1, 64, 64])], {})

    @_fails_compile()
    def test_014(self):
        self._check(HardNetNarELU(*[], **{'sm': 4}), [torch.rand([4, 1, 64, 64])], {})

    def test_015(self):
        self._check(HardTFeatNet(*[], **{'sm': 4}), [torch.rand([4, 1, 64, 64])], {})

    def test_016(self):
        self._check(HessianResp(*[], **{}), [torch.rand([4, 1, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_017(self):
        self._check(L1Norm(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_018(self):
        self._check(L2Norm(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_019(self):
        self._check(OriNetFast(*[], **{}), [torch.rand([4, 1, 64, 64])], {})

    @_fails_compile()
    def test_020(self):
        self._check(ScalePyramid(*[], **{}), [torch.rand([4, 1, 4, 4])], {})

    @_fails_compile()
    def test_021(self):
        self._check(YiNet(*[], **{}), [torch.rand([4, 1, 64, 64])], {})

