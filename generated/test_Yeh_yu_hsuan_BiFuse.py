import sys
_module = sys.modules[__name__]
del sys
CETransform = _module
Cube2Equirec = _module
CubePad = _module
Equirec2Cube = _module
EquirecRotate = _module
EquirecRotate2 = _module
ModelSaver = _module
SpherePad = _module
Transform = _module
Utils = _module
main = _module
FCRN = _module
models = _module
resnet = _module
utils = _module
vis3D = _module

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


import math


import numpy as np


import torch.nn.functional as F


from torch.autograd import Variable


import scipy.misc as sic


import matplotlib.pyplot as plt


import torch.utils.model_zoo as model_zoo


from torch.nn.parameter import Parameter


import time


from torch.utils import data


from torch.utils.data import DataLoader


from torchvision import transforms


import torchvision.models


import collections


class CETransform(nn.Module):

    def __init__(self):
        super(CETransform, self).__init__()
        equ_h = [512, 128, 64, 32, 16]
        cube_h = [256, 64, 32, 16, 8]
        self.c2e = dict()
        self.e2c = dict()
        for h in equ_h:
            a = Utils.Equirec2Cube(1, h, h * 2, h // 2, 90)
            self.e2c['(%d,%d)' % (h, h * 2)] = a
        for h in cube_h:
            a = Utils.Cube2Equirec(1, h, h * 2, h * 4)
            self.c2e['(%d)' % h] = a

    def E2C(self, x):
        [bs, c, h, w] = x.shape
        key = '(%d,%d)' % (h, w)
        assert key in self.e2c
        return self.e2c[key].ToCubeTensor(x)

    def C2E(self, x):
        [bs, c, h, w] = x.shape
        key = '(%d)' % h
        assert key in self.c2e and h == w
        return self.c2e[key].ToEquirecTensor(x)

    def forward(self, equi, cube):
        return self.e2c(equi), self.c2e(cube)


class CustomPad(nn.Module):

    def __init__(self, pad_func):
        super(CustomPad, self).__init__()
        self.pad_func = pad_func

    def forward(self, x):
        return self.pad_func(x)


CubePad = getattr(Utils.CubePad, 'CubePad')


class Equirec2Cube:

    def __init__(self, batch_size, equ_h, equ_w, out_dim, FOV, RADIUS=128, CUDA=True):
        batch_size = 1
        R_lst = []
        theta_lst = np.array([-90, 0, 90, 180], np.float) / 180 * np.pi
        phi_lst = np.array([90, -90], np.float) / 180 * np.pi
        self.equ_h = equ_h
        self.equ_w = equ_w
        self.CUDA = CUDA
        for theta in theta_lst:
            angle_axis = theta * np.array([0, 1, 0], np.float)
            R = cv2.Rodrigues(angle_axis)[0]
            R_lst.append(R)
        for phi in phi_lst:
            angle_axis = phi * np.array([1, 0, 0], np.float)
            R = cv2.Rodrigues(angle_axis)[0]
            R_lst.append(R)
        R_lst = [Variable(torch.FloatTensor(x)) for x in R_lst]
        self.out_dim = out_dim
        equ_cx = (equ_w - 1) / 2.0
        equ_cy = (equ_h - 1) / 2.0
        c_x = (out_dim - 1) / 2.0
        c_y = (out_dim - 1) / 2.0
        wangle = (180 - FOV) / 2.0
        w_len = 2 * RADIUS * np.sin(np.radians(FOV / 2.0)) / np.sin(np.radians(wangle))
        f = RADIUS / w_len * out_dim
        cx = c_x
        cy = c_y
        self.intrisic = {'f': float(f), 'cx': float(cx), 'cy': float(cy)}
        interval = w_len / (out_dim - 1)
        z_map = np.zeros([out_dim, out_dim], np.float32) + RADIUS
        x_map = np.tile((np.arange(out_dim) - c_x) * interval, [out_dim, 1])
        y_map = np.tile((np.arange(out_dim) - c_y) * interval, [out_dim, 1]).T
        D = np.sqrt(x_map ** 2 + y_map ** 2 + z_map ** 2)
        xyz = np.zeros([out_dim, out_dim, 3], np.float)
        xyz[:, :, 0] = RADIUS / D * x_map[:, :]
        xyz[:, :, 1] = RADIUS / D * y_map[:, :]
        xyz[:, :, 2] = RADIUS / D * z_map[:, :]
        if CUDA:
            xyz = Variable(torch.FloatTensor(xyz))
        else:
            xyz = Variable(torch.FloatTensor(xyz))
        reshape_xyz = xyz.view(out_dim * out_dim, 3).transpose(0, 1)
        self.batch_size = batch_size
        self.loc = []
        self.grid = []
        for i, R in enumerate(R_lst):
            result = torch.matmul(R, reshape_xyz).transpose(0, 1)
            tmp_xyz = result.contiguous().view(1, out_dim, out_dim, 3)
            self.grid.append(tmp_xyz)
            lon = torch.atan2(result[:, 0], result[:, 2]).view(1, out_dim, out_dim, 1) / np.pi
            lat = torch.asin(result[:, 1] / RADIUS).view(1, out_dim, out_dim, 1) / (np.pi / 2)
            self.loc.append(torch.cat([lon.repeat(batch_size, 1, 1, 1), lat.repeat(batch_size, 1, 1, 1)], dim=3))
        new_lst = [3, 5, 1, 0, 2, 4]
        self.R_lst = [R_lst[x] for x in new_lst]
        self.grid_lst = []
        for iii in new_lst:
            grid = self.grid[iii].clone()
            scale = self.intrisic['f'] / grid[:, :, :, 2:3]
            grid *= scale
            self.grid_lst.append(grid)

    def _ToCube(self, batch, mode):
        batch_size = batch.size()[0]
        new_lst = [3, 5, 1, 0, 2, 4]
        out = []
        for i in new_lst:
            coor = self.loc[i] if self.CUDA else self.loc[i]
            result = []
            for ii in range(batch_size):
                tmp = F.grid_sample(batch[ii:ii + 1], coor, mode=mode)
                result.append(tmp)
            result = torch.cat(result, dim=0)
            out.append(result)
        return out

    def GetGrid(self):
        new_lst = [3, 5, 1, 0, 2, 4]
        out = [self.grid[x] for x in new_lst]
        out = torch.cat(out, dim=0)
        return out

    def ToCubeNumpy(self, batch):
        out = self._ToCube(batch)
        result = [x.data.cpu().numpy() for x in out]
        return result

    def ToCubeTensor(self, batch, mode='bilinear'):
        assert mode in ['bilinear', 'nearest']
        batch_size = batch.size()[0]
        cube = self._ToCube(batch, mode=mode)
        out_batch = None
        for batch_idx in range(batch_size):
            for cube_idx in range(6):
                patch = torch.unsqueeze(cube[cube_idx][batch_idx, :, :, :], 0)
                if out_batch is None:
                    out_batch = patch
                else:
                    out_batch = torch.cat([out_batch, patch], dim=0)
        return out_batch


class SpherePad(nn.Module):

    def __init__(self, pad_size):
        super(SpherePad, self).__init__()
        self.pad_size = pad_size
        self.data = {}
        self.relation = {'back': ['top-up_yes_yes_no', 'down-down_yes_yes_no', 'right-right_no_no_no', 'left-left_no_no_no'], 'down': ['front-down_no_no_no', 'back-down_yes_yes_no', 'left-down_yes_no_yes', 'right-down_no_yes_yes'], 'front': ['top-down_no_no_no', 'down-up_no_no_no', 'left-right_no_no_no', 'right-left_no_no_no'], 'left': ['top-left_yes_no_yes', 'down-left_no_yes_yes', 'back-right_no_no_no', 'front-left_no_no_no'], 'right': ['top-right_no_yes_yes', 'down-right_yes_no_yes', 'front-right_no_no_no', 'back-left_no_no_no'], 'top': ['back-up_yes_yes_no', 'front-up_no_no_no', 'left-up_no_yes_yes', 'right-up_yes_no_yes']}

    def _GetLoc(self, R_lst, grid_lst, K):
        out = {}
        pad = self.pad_size
        f, cx, cy = K['f'], K['cx'], K['cy']
        K_mat = torch.FloatTensor(np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]]))
        grid_front = grid_lst[2]
        orders = ['back', 'down', 'front', 'left', 'right', 'top']
        for i, face in enumerate(orders):
            out[face] = {}
            for j, connect_side in enumerate(['up', 'down', 'left', 'right']):
                connected_face = self.relation[face][j].split('-')[0]
                idx = orders.index(connected_face)
                R_world_to_connected = R_lst[idx]
                R_world_to_itself = R_lst[i]
                R_itself_to_connected = torch.matmul(R_world_to_connected, R_world_to_itself.transpose(0, 1))
                new_grid = torch.matmul(grid_front, R_itself_to_connected.transpose(0, 1))
                proj = torch.matmul(new_grid, K_mat.transpose(0, 1))
                x = proj[:, :, :, 0:1] / proj[:, :, :, 2:3]
                y = proj[:, :, :, 1:2] / proj[:, :, :, 2:3]
                x = (x - cx) / cx
                y = (y - cy) / cy
                xy = torch.cat([x, y], dim=3)
                out[face][connect_side] = {}
                x = xy[:, :, :, 0:1]
                y = xy[:, :, :, 1:2]
                """
                mask1 = np.logical_and(x >= -1.01, x <= 1.01)
                mask2 = np.logical_and(y >= -1.01, y <= 1.01)
                mask = np.logical_and(mask1, mask2)
                """
                mask1 = (x >= -1.01) & (x <= 1.01)
                mask2 = (y >= -1.01) & (y <= 1.01)
                mask = mask1 & mask2
                xy = torch.clamp(xy, -1, 1)
                if connect_side == 'up':
                    out[face][connect_side]['mask'] = mask[:, :pad, :, :]
                    out[face][connect_side]['xy'] = xy[:, :pad, :, :]
                elif connect_side == 'down':
                    out[face][connect_side]['mask'] = mask[:, -pad:, :, :]
                    out[face][connect_side]['xy'] = xy[:, -pad:, :, :]
                elif connect_side == 'left':
                    out[face][connect_side]['mask'] = mask[:, :, :pad, :]
                    out[face][connect_side]['xy'] = xy[:, :, :pad, :]
                elif connect_side == 'right':
                    out[face][connect_side]['mask'] = mask[:, :, -pad:, :]
                    out[face][connect_side]['xy'] = xy[:, :, -pad:, :]
        return out

    def forward(self, inputs):
        [bs, c, h, w] = inputs.shape
        assert bs % 6 == 0 and h == w
        key = '(%d,%d,%d)' % (h, w, self.pad_size)
        if key not in self.data:
            theta = 2 * np.arctan((0.5 * h + self.pad_size) / (0.5 * h))
            e2c_ori = Equirec2Cube(1, 2 * h, 4 * h, h, 90)
            e2c = Equirec2Cube(1, 2 * h, 4 * h, h + 2 * self.pad_size, theta / np.pi * 180)
            R_lst = [x.transpose(0, 1) for x in e2c.R_lst]
            grid_lst = e2c.grid_lst
            K = e2c_ori.intrisic
            self.data[key] = self._GetLoc(R_lst, grid_lst, K)
        pad = self.pad_size
        orders = ['back', 'down', 'front', 'left', 'right', 'top']
        out = []
        for i, face in enumerate(orders):
            this_face = inputs[i::6]
            this_face = F.pad(this_face, (pad, pad, pad, pad))
            repeats = this_face.shape[0]
            for j, connect_side in enumerate(['up', 'down', 'left', 'right']):
                connected_face_name = self.relation[face][j].split('-')[0]
                connected_face = inputs[orders.index(connected_face_name)::6]
                mask = self.data[key][face][connect_side]['mask'].repeat(repeats, 1, 1, c).permute(0, 3, 1, 2)
                xy = self.data[key][face][connect_side]['xy'].repeat(repeats, 1, 1, 1)
                interpo = F.grid_sample(connected_face, xy, mode='bilinear')
                if connect_side == 'up':
                    this_face[:, :, :pad, :][mask] = interpo[mask]
                elif connect_side == 'down':
                    this_face[:, :, -pad:, :][mask] = interpo[mask]
                elif connect_side == 'left':
                    this_face[:, :, :, :pad][mask] = interpo[mask]
                elif connect_side == 'right':
                    this_face[:, :, :, -pad:][mask] = interpo[mask]
            out.append(this_face)
        out = torch.cat(out, dim=0)
        [bs, c, h, w] = out.shape
        out = out.view(-1, bs // 6, c, h, w).transpose(0, 1).contiguous().view(bs, c, h, w)
        return out


class Depth2Points(nn.Module):

    def __init__(self, xyz_grid, CUDA=True):
        super(Depth2Points, self).__init__()
        self.xyz_grid = xyz_grid
        self.order = ['back', 'down', 'front', 'left', 'right', 'up']
        self.CUDA = CUDA

    def forward(self, x):
        [bs, c, h, w] = x.size()
        if bs % 6 != 0 or c != 1:
            None
            exit()
        bs = bs // 6
        grid = self.xyz_grid
        grid = grid if self.CUDA else grid
        all_pts = []
        for i in range(bs):
            cubemap = x[i * 6:(i + 1) * 6, 0, :, :]
            for j, face in enumerate(self.order):
                if face == 'back' or face == 'front':
                    scale = cubemap[j, :, :] / torch.abs(grid[j, :, :, 2])
                elif face == 'down' or face == 'up':
                    scale = cubemap[j, :, :] / torch.abs(grid[j, :, :, 1])
                elif face == 'left' or face == 'right':
                    scale = cubemap[j, :, :] / torch.abs(grid[j, :, :, 0])
                else:
                    None
                    exit()
                pt_x = (scale * grid[j, :, :, 0]).view(1, h, w, 1)
                pt_y = (scale * grid[j, :, :, 1]).view(1, h, w, 1)
                pt_z = (scale * grid[j, :, :, 2]).view(1, h, w, 1)
                pt = torch.cat([pt_x, pt_y, pt_z], dim=3)
                all_pts.append(pt)
        point_cloud = torch.cat(all_pts, dim=0)
        return point_cloud


class EquirecDepth2Points(nn.Module):

    def __init__(self, xyz_grid, CUDA=True):
        super(EquirecDepth2Points, self).__init__()
        self.grid = xyz_grid
        self.CUDA = CUDA

    def forward(self, depth):
        norm = torch.norm(self.grid, p=2, dim=3).unsqueeze(3)
        pts = []
        grid = self.grid if self.CUDA else self.grid
        for i in range(depth.size()[0]):
            tmp = grid / norm * depth[i:i + 1, 0, :, :].unsqueeze(3)
            pts.append(tmp)
        result = torch.cat(pts, dim=0)
        return result


class Unpool(nn.Module):

    def __init__(self, num_channels, stride=2):
        super(Unpool, self).__init__()
        self.num_channels = num_channels
        self.stride = stride
        self.weights = torch.autograd.Variable(torch.zeros(num_channels, 1, stride, stride))
        self.weights[:, :, 0, 0] = 1

    def forward(self, x):
        return F.conv_transpose2d(x, self.weights, stride=self.stride, groups=self.num_channels)


class Decoder(nn.Module):
    names = ['deconv2', 'deconv3', 'upconv', 'upproj']

    def __init__(self):
        super(Decoder, self).__init__()
        self.layer1 = None
        self.layer2 = None
        self.layer3 = None
        self.layer4 = None

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class DeConv(Decoder):

    def __init__(self, in_channels, kernel_size):
        assert kernel_size >= 2, 'kernel_size out of range: {}'.format(kernel_size)
        super(DeConv, self).__init__()

        def convt(in_channels):
            stride = 2
            padding = (kernel_size - 1) // 2
            output_padding = kernel_size % 2
            assert -2 - 2 * padding + kernel_size + output_padding == 0, 'deconv parameters incorrect'
            module_name = 'deconv{}'.format(kernel_size)
            return nn.Sequential(collections.OrderedDict([(module_name, nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size, stride, padding, output_padding, bias=False)), ('batchnorm', nn.BatchNorm2d(in_channels // 2)), ('relu', nn.ReLU(inplace=True))]))
        self.layer1 = convt(in_channels)
        self.layer2 = convt(in_channels // 2)
        self.layer3 = convt(in_channels // 2 ** 2)
        self.layer4 = convt(in_channels // 2 ** 3)


class UpConv(Decoder):

    def upconv_module(self, in_channels):
        upconv = nn.Sequential(collections.OrderedDict([('unpool', Unpool(in_channels)), ('conv', nn.Conv2d(in_channels, in_channels // 2, kernel_size=5, stride=1, padding=2, bias=False)), ('batchnorm', nn.BatchNorm2d(in_channels // 2)), ('relu', nn.ReLU())]))
        return upconv

    def __init__(self, in_channels):
        super(UpConv, self).__init__()
        self.layer1 = self.upconv_module(in_channels)
        self.layer2 = self.upconv_module(in_channels // 2)
        self.layer3 = self.upconv_module(in_channels // 4)
        self.layer4 = self.upconv_module(in_channels // 8)


class UpProj(Decoder):


    class UpProjModule(nn.Module):

        def __init__(self, in_channels, out_channels=None, padding=None):
            super(UpProj.UpProjModule, self).__init__()
            if out_channels is None:
                out_channels = in_channels // 2
            self.pad_3 = padding(1)
            self.pad_5 = padding(2)
            self.unpool = Unpool(in_channels)
            self.upper_branch = nn.Sequential(collections.OrderedDict([('pad1', CustomPad(self.pad_5)), ('conv1', nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=0, bias=False)), ('batchnorm1', nn.BatchNorm2d(out_channels)), ('relu', nn.ReLU()), ('pad2', CustomPad(self.pad_3)), ('conv2', nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False)), ('batchnorm2', nn.BatchNorm2d(out_channels))]))
            self.bottom_branch = nn.Sequential(collections.OrderedDict([('pad', CustomPad(self.pad_5)), ('conv', nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=0, bias=False)), ('batchnorm', nn.BatchNorm2d(out_channels))]))
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.unpool(x)
            x1 = self.upper_branch(x)
            x2 = self.bottom_branch(x)
            x = x1 + x2
            x = self.relu(x)
            return x

    def __init__(self, in_channels, padding):
        super(UpProj, self).__init__()
        self.padding = getattr(Utils.CubePad, padding)
        self.layer1 = self.UpProjModule(in_channels, padding=self.padding)
        self.layer2 = self.UpProjModule(in_channels // 2, padding=self.padding)
        self.layer3 = self.UpProjModule(in_channels // 4, padding=self.padding)
        self.layer4 = self.UpProjModule(in_channels // 8, padding=self.padding)


class PreprocBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size_lst, stride=2):
        super(PreprocBlock, self).__init__()
        assert len(kernel_size_lst) == 4 and out_channels % 4 == 0
        self.lst = nn.ModuleList([])
        for h, w in kernel_size_lst:
            padding = h // 2, w // 2
            tmp = nn.Sequential(nn.Conv2d(in_channels, out_channels // 4, kernel_size=(h, w), stride=stride, padding=padding), nn.BatchNorm2d(out_channels // 4), nn.ReLU(inplace=True))
            self.lst.append(tmp)

    def forward(self, x):
        out = []
        for conv in self.lst:
            out.append(conv(x))
        out = torch.cat(out, dim=1)
        return out


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2.0 / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        m.weight.data.normal_(0, math.sqrt(2.0 / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


class Refine(nn.Module):

    def __init__(self):
        super(Refine, self).__init__()
        self.refine_1 = nn.Sequential(nn.Conv2d(5, 32, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.refine_2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.deconv_1 = nn.Sequential(nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=0, groups=1, bias=True, dilation=1), nn.BatchNorm2d(64), nn.LeakyReLU(inplace=True))
        self.deconv_2 = nn.Sequential(nn.ConvTranspose2d(192, 32, kernel_size=4, stride=2, padding=1, output_padding=0, groups=1, bias=True, dilation=1), nn.BatchNorm2d(32), nn.LeakyReLU(inplace=True))
        self.refine_3 = nn.Sequential(nn.Conv2d(96, 16, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1, bias=False))
        self.bilinear_1 = nn.UpsamplingBilinear2d(size=(256, 512))
        self.bilinear_2 = nn.UpsamplingBilinear2d(size=(512, 1024))

    def forward(self, inputs):
        x = inputs
        out_1 = self.refine_1(x)
        out_2 = self.refine_2(out_1)
        deconv_out1 = self.deconv_1(out_2)
        up_1 = self.bilinear_1(out_2)
        deconv_out2 = self.deconv_2(torch.cat((deconv_out1, up_1), dim=1))
        up_2 = self.bilinear_2(out_1)
        out_3 = self.refine_3(torch.cat((deconv_out2, up_2), dim=1))
        return out_3


def choose_decoder(decoder, in_channels, padding):
    if decoder[:6] == 'deconv':
        assert len(decoder) == 7
        kernel_size = int(decoder[6])
        return DeConv(in_channels, kernel_size)
    elif decoder == 'upproj':
        return UpProj(in_channels, padding=padding)
    elif decoder == 'upconv':
        return UpConv(in_channels)
    else:
        assert False, 'invalid option for decoder: {}'.format(decoder)


class MyModel(nn.Module):

    def __init__(self, layers, decoder, output_size=None, in_channels=3, pretrained=True):
        super(MyModel, self).__init__()
        bs = 1
        self.equi_model = fusion_ResNet(bs, layers, decoder, (512, 1024), 3, pretrained, padding='ZeroPad')
        self.cube_model = fusion_ResNet(bs * 6, layers, decoder, (256, 256), 3, pretrained, padding='SpherePad')
        self.refine_model = Refine()
        if layers <= 34:
            num_channels = 512
        elif layers >= 50:
            num_channels = 2048
        self.equi_decoder = choose_decoder(decoder, num_channels // 2, padding='ZeroPad')
        self.equi_conv3 = nn.Sequential(nn.Conv2d(num_channels // 32, 1, kernel_size=3, stride=1, padding=1, bias=False), nn.UpsamplingBilinear2d(size=(512, 1024)))
        self.cube_decoder = choose_decoder(decoder, num_channels // 2, padding='SpherePad')
        mypad = getattr(Utils.CubePad, 'SpherePad')
        self.cube_conv3 = nn.Sequential(mypad(1), nn.Conv2d(num_channels // 32, 1, kernel_size=3, stride=1, padding=0, bias=False), nn.UpsamplingBilinear2d(size=(256, 256)))
        self.equi_decoder.apply(weights_init)
        self.equi_conv3.apply(weights_init)
        self.cube_decoder.apply(weights_init)
        self.cube_conv3.apply(weights_init)
        self.ce = CETransform()
        if layers <= 34:
            ch_lst = [64, 64, 128, 256, 512, 256, 128, 64, 32]
        else:
            ch_lst = [64, 256, 512, 1024, 2048, 1024, 512, 256, 128]
        self.conv_e2c = nn.ModuleList([])
        self.conv_c2e = nn.ModuleList([])
        self.conv_mask = nn.ModuleList([])
        for i in range(9):
            conv_c2e = nn.Sequential(nn.Conv2d(ch_lst[i], ch_lst[i], kernel_size=3, padding=1), nn.ReLU(inplace=True))
            conv_e2c = nn.Sequential(nn.Conv2d(ch_lst[i], ch_lst[i], kernel_size=3, padding=1), nn.ReLU(inplace=True))
            conv_mask = nn.Sequential(nn.Conv2d(ch_lst[i] * 2, 1, kernel_size=1, padding=0), nn.Sigmoid())
            self.conv_e2c.append(conv_e2c)
            self.conv_c2e.append(conv_c2e)
            self.conv_mask.append(conv_mask)
        self.grid = Utils.Equirec2Cube(None, 512, 1024, 256, 90).GetGrid()
        self.d2p = Utils.Depth2Points(self.grid)

    def forward_FCRN_fusion(self, equi, fusion=False):
        cube = self.ce.E2C(equi)
        feat_equi = self.equi_model.pre_encoder2(equi)
        feat_cube = self.cube_model.pre_encoder(cube)
        for e in range(5):
            if fusion:
                aaa = self.conv_e2c[e](feat_equi)
                tmp_cube = self.ce.E2C(aaa)
                tmp_equi = self.conv_c2e[e](self.ce.C2E(feat_cube))
                mask_equi = self.conv_mask[e](torch.cat([aaa, tmp_equi], dim=1))
                mask_cube = 1 - mask_equi
                tmp_cube = tmp_cube.clone() * self.ce.E2C(mask_cube)
                tmp_equi = tmp_equi.clone() * mask_equi
            else:
                tmp_cube = 0
                tmp_equi = 0
            feat_cube = feat_cube + tmp_cube
            feat_equi = feat_equi + tmp_equi
            if e < 4:
                feat_cube = getattr(self.cube_model, 'layer%d' % (e + 1))(feat_cube)
                feat_equi = getattr(self.equi_model, 'layer%d' % (e + 1))(feat_equi)
            else:
                feat_cube = self.cube_model.conv2(feat_cube)
                feat_equi = self.equi_model.conv2(feat_equi)
                feat_cube = self.cube_model.bn2(feat_cube)
                feat_equi = self.equi_model.bn2(feat_equi)
        for d in range(4):
            if fusion:
                aaa = self.conv_e2c[d + 5](feat_equi)
                tmp_cube = self.ce.E2C(aaa)
                tmp_equi = self.conv_c2e[d + 5](self.ce.C2E(feat_cube))
                mask_equi = self.conv_mask[d + 5](torch.cat([aaa, tmp_equi], dim=1))
                mask_cube = 1 - mask_equi
                tmp_cube = tmp_cube.clone() * self.ce.E2C(mask_cube)
                tmp_equi = tmp_equi.clone() * mask_equi
                tmp_equi = tmp_equi.clone() * mask_equi
            else:
                tmp_cube = 0
                tmp_equi = 0
            feat_cube = feat_cube + tmp_cube
            feat_equi = feat_equi + tmp_equi
            feat_equi = getattr(self.equi_decoder, 'layer%d' % (d + 1))(feat_equi)
            feat_cube = getattr(self.cube_decoder, 'layer%d' % (d + 1))(feat_cube)
        equi_depth = self.equi_conv3(feat_equi)
        cube_depth = self.cube_conv3(feat_cube)
        cube_pts = self.d2p(cube_depth)
        c2e_depth = self.ce.C2E(torch.norm(cube_pts, p=2, dim=3).unsqueeze(1))
        feat_cat = torch.cat((equi, equi_depth, c2e_depth), dim=1)
        refine_final = self.refine_model(feat_cat)
        return equi_depth, cube_depth, refine_final

    def forward_FCRN_cube(self, equi):
        cube = self.ce.E2C(equi)
        feat_cube = self.cube_model.pre_encoder(cube)
        for e in range(5):
            if e < 4:
                feat_cube = getattr(self.cube_model, 'layer%d' % (e + 1))(feat_cube)
            else:
                feat_cube = self.cube_model.conv2(feat_cube)
                feat_cube = self.cube_model.bn2(feat_cube)
        for d in range(4):
            feat_cube = getattr(self.cube_decoder, 'layer%d' % (d + 1))(feat_cube)
        cube_depth = self.cube_conv3(feat_cube)
        return cube_depth

    def forward_FCRN_equi(self, equi):
        feat_equi = self.equi_model.pre_encoder2(equi)
        for e in range(5):
            if e < 4:
                feat_equi = getattr(self.equi_model, 'layer%d' % (e + 1))(feat_equi)
            else:
                feat_equi = self.equi_model.conv2(feat_equi)
                feat_equi = self.equi_model.bn2(feat_equi)
        for d in range(4):
            feat_equi = getattr(self.equi_decoder, 'layer%d' % (d + 1))(feat_equi)
        equi_depth = self.equi_conv3(feat_equi)
        return equi_depth

    def forward(self, x):
        return self.forward_FCRN_fusion(x, True)


def conv3x3(in_planes, out_planes, stride=1, auto_padding=True):
    """3x3 convolution with padding"""
    p = 1 if auto_padding else 0
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=p, bias=False)


def conv1x1(in_planes, out_planes, stride=1, auto_padding=True):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, padding='ZeroPad'):
        super(ResNet, self).__init__()
        self.padding = getattr(Utils.CubePad, padding)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.layer1 = self._make_layer(block, 64, layers[0], padding=self.padding)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, padding=self.padding)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, padding=self.padding)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, padding=self.padding)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, padding, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, padding=padding))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, padding=padding))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

