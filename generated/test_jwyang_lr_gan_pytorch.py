import sys
_module = sys.modules[__name__]
del sys
_ext = _module
stnm = _module
functions = _module
gridgen = _module
stnm = _module
modules = _module
gridgen = _module
stnm = _module
train = _module

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


from torch.autograd import Function


import numpy as np


from torch.nn.modules.module import Module


from torch.autograd import Variable


import random


import torch.nn as nn


import torch.nn.parallel


import torch.backends.cudnn as cudnn


import torch.optim as optim


import torch.utils.data


import torchvision.datasets as dset


import torchvision.transforms as transforms


import torchvision.utils as vutils


class AffineGridGenFunction(Function):

    def __init__(self, height, width, lr=1):
        super(AffineGridGenFunction, self).__init__()
        self.lr = lr
        self.height, self.width = height, width
        self.grid = torch.Tensor(self.height, self.width, 3)
        for i in range(self.height):
            self.grid.select(2, 0).select(0, i).fill_(-1 + float(i) / (self.height - 1) * 2)
        for j in range(self.width):
            self.grid.select(2, 1).select(1, j).fill_(-1 + float(j) / (self.width - 1) * 2)
        self.grid.select(2, 2).fill_(1)

    def forward(self, input1):
        self.input1 = input1
        output = torch.zeros(torch.Size([input1.size(0)]) + self.grid.size())
        self.batchgrid = torch.zeros(torch.Size([input1.size(0)]) + self.grid.size())
        for i in range(input1.size(0)):
            self.batchgrid[i] = self.grid
        if input1.is_cuda:
            self.batchgrid = self.batchgrid
            output = output
        batchgrid_temp = self.batchgrid.view(-1, self.height * self.width, 3)
        batchgrid_temp.contiguous()
        input_temp = torch.transpose(input1, 1, 2)
        input_temp.contiguous()
        output_temp = torch.bmm(batchgrid_temp, input_temp)
        output = output_temp.view(-1, self.height, self.width, 2)
        output.contiguous()
        return output

    def backward(self, grad_output):
        grad_input1 = torch.zeros(self.input1.size())
        if grad_output.is_cuda:
            self.batchgrid = self.batchgrid
            grad_input1 = grad_input1
        grad_output_temp = grad_output.contiguous()
        grad_output_view = grad_output_temp.view(-1, self.height * self.width, 2)
        grad_output_view.contiguous()
        grad_output_temp = torch.transpose(grad_output_view, 1, 2)
        grad_output_temp.contiguous()
        batchgrid_temp = self.batchgrid.view(-1, self.height * self.width, 3)
        batchgrid_temp.contiguous()
        grad_input1 = torch.baddbmm(grad_input1, grad_output_temp, batchgrid_temp)
        return grad_input1


class AffineGridGen(Module):

    def __init__(self, height, width, lr=1, aux_loss=False):
        super(AffineGridGen, self).__init__()
        self.height, self.width = height, width
        self.aux_loss = aux_loss
        self.f = AffineGridGenFunction(self.height, self.width, lr=lr)
        self.lr = lr

    def forward(self, input):
        if not self.aux_loss:
            return self.f(input)
        else:
            identity = torch.from_numpy(np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32))
            batch_identity = torch.zeros([input.size(0), 2, 3])
            for i in range(input.size(0)):
                batch_identity[i] = identity
            batch_identity = Variable(batch_identity)
            loss = torch.mul(input - batch_identity, input - batch_identity)
            loss = torch.sum(loss, 1)
            loss = torch.sum(loss, 2)
            return self.f(input), loss.view(-1, 1)


class CylinderGridGenFunction(Function):

    def __init__(self, height, width, lr=1):
        super(CylinderGridGenFunction, self).__init__()
        self.lr = lr
        self.height, self.width = height, width
        self.grid = np.zeros([self.height, self.width, 3], dtype=np.float32)
        self.grid[:, :, (0)] = np.expand_dims(np.repeat(np.expand_dims(np.arange(-1, 1, 2.0 / self.height), 0), repeats=self.width, axis=0).T, 0)
        self.grid[:, :, (1)] = np.expand_dims(np.repeat(np.expand_dims(np.arange(-1, 1, 2.0 / self.width), 0), repeats=self.height, axis=0), 0)
        self.grid[:, :, (2)] = np.ones([self.height, width])
        self.grid = torch.from_numpy(self.grid.astype(np.float32))

    def forward(self, input1):
        self.input1 = (1 + torch.cos(input1)) / 2
        output = torch.zeros(torch.Size([input1.size(0), self.height, self.width, 2]))
        if not self.input1.is_cuda:
            for i in range(self.input1.size(0)):
                x = self.input1[i][0]
                low = int(np.ceil(self.width * self.input1[i][0]))
                frac = self.width * self.input1[i][0] - low
                interp = frac * 2 * (1 - x) + (1 - frac) * 2 * -x
                output[(i), :, :, (1)] = torch.zeros(self.grid[:, :, (1)].size())
                if low <= self.width and low > 0:
                    output[(i), :, :low, (1)].fill_(2 * (1 - x))
                if low < self.width and low >= 0:
                    output[(i), :, low:, (1)].fill_(2 * -x)
                output[(i), :, :, (1)] = output[(i), :, :, (1)] + self.grid[:, :, (1)]
                output[(i), :, :, (0)] = self.grid[:, :, (0)]
        else:
            None
        return output

    def backward(self, grad_output):
        grad_input1 = torch.zeros(self.input1.size())
        if not grad_output.is_cuda:
            for i in range(self.input1.size(0)):
                grad_input1[i] = -torch.sum(torch.sum(grad_output[(i), :, :, (1)], 1)) * torch.sin(self.input1[i]) / 2
        else:
            None
        return grad_input1 * self.lr


class CylinderGridGen(Module):

    def __init__(self, height, width, lr=1, aux_loss=False):
        super(CylinderGridGen, self).__init__()
        self.height, self.width = height, width
        self.aux_loss = aux_loss
        self.f = CylinderGridGenFunction(self.height, self.width, lr=lr)
        self.lr = lr

    def forward(self, input):
        if not self.aux_loss:
            return self.f(input)
        else:
            return self.f(input), torch.mul(input, input).view(-1, 1)


class AffineGridGenV2(Module):

    def __init__(self, height, width, lr=1, aux_loss=False):
        super(AffineGridGenV2, self).__init__()
        self.height, self.width = height, width
        self.aux_loss = aux_loss
        self.lr = lr
        self.grid = np.zeros([self.height, self.width, 3], dtype=np.float32)
        self.grid[:, :, (0)] = np.expand_dims(np.repeat(np.expand_dims(np.arange(-1, 1, 2.0 / self.height), 0), repeats=self.width, axis=0).T, 0)
        self.grid[:, :, (1)] = np.expand_dims(np.repeat(np.expand_dims(np.arange(-1, 1, 2.0 / self.width), 0), repeats=self.height, axis=0), 0)
        self.grid[:, :, (2)] = np.ones([self.height, width])
        self.grid = torch.from_numpy(self.grid.astype(np.float32))

    def forward(self, input1):
        self.batchgrid = torch.zeros(torch.Size([input1.size(0)]) + self.grid.size())
        for i in range(input1.size(0)):
            self.batchgrid[i] = self.grid
        self.batchgrid = Variable(self.batchgrid)
        if input1.is_cuda:
            self.batchgrid = self.batchgrid
        output = torch.bmm(self.batchgrid.view(-1, self.height * self.width, 3), torch.transpose(input1, 1, 2)).view(-1, self.height, self.width, 2)
        return output


class CylinderGridGenV2(Module):

    def __init__(self, height, width, lr=1):
        super(CylinderGridGenV2, self).__init__()
        self.height, self.width = height, width
        self.lr = lr
        self.grid = np.zeros([self.height, self.width, 3], dtype=np.float32)
        self.grid[:, :, (0)] = np.expand_dims(np.repeat(np.expand_dims(np.arange(-1, 1, 2.0 / self.height), 0), repeats=self.width, axis=0).T, 0)
        self.grid[:, :, (1)] = np.expand_dims(np.repeat(np.expand_dims(np.arange(-1, 1, 2.0 / self.width), 0), repeats=self.height, axis=0), 0)
        self.grid[:, :, (2)] = np.ones([self.height, width])
        self.grid = torch.from_numpy(self.grid.astype(np.float32))

    def forward(self, input):
        self.batchgrid = torch.zeros(torch.Size([input.size(0)]) + self.grid.size())
        for i in range(input.size(0)):
            self.batchgrid[(i), :, :, :] = self.grid
        self.batchgrid = Variable(self.batchgrid)
        input_u = input.view(-1, 1, 1, 1).repeat(1, self.height, self.width, 1)
        output0 = self.batchgrid[:, :, :, 0:1]
        output1 = torch.atan(torch.tan(np.pi / 2.0 * (self.batchgrid[:, :, :, 1:2] + self.batchgrid[:, :, :, 2:] * input_u[:, :, :, :]))) / (np.pi / 2)
        output = torch.cat([output0, output1], 3)
        return output


class DenseAffineGridGen(Module):

    def __init__(self, height, width, lr=1, aux_loss=False):
        super(DenseAffineGridGen, self).__init__()
        self.height, self.width = height, width
        self.aux_loss = aux_loss
        self.lr = lr
        self.grid = np.zeros([self.height, self.width, 3], dtype=np.float32)
        self.grid[:, :, (0)] = np.expand_dims(np.repeat(np.expand_dims(np.arange(-1, 1, 2.0 / self.height), 0), repeats=self.width, axis=0).T, 0)
        self.grid[:, :, (1)] = np.expand_dims(np.repeat(np.expand_dims(np.arange(-1, 1, 2.0 / self.width), 0), repeats=self.height, axis=0), 0)
        self.grid[:, :, (2)] = np.ones([self.height, width])
        self.grid = torch.from_numpy(self.grid.astype(np.float32))

    def forward(self, input1):
        self.batchgrid = torch.zeros(torch.Size([input1.size(0)]) + self.grid.size())
        for i in range(input1.size(0)):
            self.batchgrid[i] = self.grid
        self.batchgrid = Variable(self.batchgrid)
        x = torch.mul(self.batchgrid, input1[:, :, :, 0:3])
        y = torch.mul(self.batchgrid, input1[:, :, :, 3:6])
        output = torch.cat([torch.sum(x, 3), torch.sum(y, 3)], 3)
        return output


class DenseAffine3DGridGen(Module):

    def __init__(self, height, width, lr=1, aux_loss=False):
        super(DenseAffine3DGridGen, self).__init__()
        self.height, self.width = height, width
        self.aux_loss = aux_loss
        self.lr = lr
        self.grid = np.zeros([self.height, self.width, 3], dtype=np.float32)
        self.grid[:, :, (0)] = np.expand_dims(np.repeat(np.expand_dims(np.arange(-1, 1, 2.0 / self.height), 0), repeats=self.width, axis=0).T, 0)
        self.grid[:, :, (1)] = np.expand_dims(np.repeat(np.expand_dims(np.arange(-1, 1, 2.0 / self.width), 0), repeats=self.height, axis=0), 0)
        self.grid[:, :, (2)] = np.ones([self.height, width])
        self.grid = torch.from_numpy(self.grid.astype(np.float32))
        self.theta = self.grid[:, :, (0)] * np.pi / 2 + np.pi / 2
        self.phi = self.grid[:, :, (1)] * np.pi
        self.x = torch.sin(self.theta) * torch.cos(self.phi)
        self.y = torch.sin(self.theta) * torch.sin(self.phi)
        self.z = torch.cos(self.theta)
        self.grid3d = torch.from_numpy(np.zeros([self.height, self.width, 4], dtype=np.float32))
        self.grid3d[:, :, (0)] = self.x
        self.grid3d[:, :, (1)] = self.y
        self.grid3d[:, :, (2)] = self.z
        self.grid3d[:, :, (3)] = self.grid[:, :, (2)]

    def forward(self, input1):
        self.batchgrid3d = torch.zeros(torch.Size([input1.size(0)]) + self.grid3d.size())
        for i in range(input1.size(0)):
            self.batchgrid3d[i] = self.grid3d
        self.batchgrid3d = Variable(self.batchgrid3d)
        x = torch.sum(torch.mul(self.batchgrid3d, input1[:, :, :, 0:4]), 3)
        y = torch.sum(torch.mul(self.batchgrid3d, input1[:, :, :, 4:8]), 3)
        z = torch.sum(torch.mul(self.batchgrid3d, input1[:, :, :, 8:]), 3)
        r = torch.sqrt(x ** 2 + y ** 2 + z ** 2) + 1e-05
        theta = torch.acos(z / r) / (np.pi / 2) - 1
        phi = torch.atan(y / (x + 1e-05)) + np.pi * x.lt(0).type(torch.FloatTensor) * (y.ge(0).type(torch.FloatTensor) - y.lt(0).type(torch.FloatTensor))
        phi = phi / np.pi
        output = torch.cat([theta, phi], 3)
        return output


class DenseAffine3DGridGen_rotate(Module):

    def __init__(self, height, width, lr=1, aux_loss=False):
        super(DenseAffine3DGridGen_rotate, self).__init__()
        self.height, self.width = height, width
        self.aux_loss = aux_loss
        self.lr = lr
        self.grid = np.zeros([self.height, self.width, 3], dtype=np.float32)
        self.grid[:, :, (0)] = np.expand_dims(np.repeat(np.expand_dims(np.arange(-1, 1, 2.0 / self.height), 0), repeats=self.width, axis=0).T, 0)
        self.grid[:, :, (1)] = np.expand_dims(np.repeat(np.expand_dims(np.arange(-1, 1, 2.0 / self.width), 0), repeats=self.height, axis=0), 0)
        self.grid[:, :, (2)] = np.ones([self.height, width])
        self.grid = torch.from_numpy(self.grid.astype(np.float32))
        self.theta = self.grid[:, :, (0)] * np.pi / 2 + np.pi / 2
        self.phi = self.grid[:, :, (1)] * np.pi
        self.x = torch.sin(self.theta) * torch.cos(self.phi)
        self.y = torch.sin(self.theta) * torch.sin(self.phi)
        self.z = torch.cos(self.theta)
        self.grid3d = torch.from_numpy(np.zeros([self.height, self.width, 4], dtype=np.float32))
        self.grid3d[:, :, (0)] = self.x
        self.grid3d[:, :, (1)] = self.y
        self.grid3d[:, :, (2)] = self.z
        self.grid3d[:, :, (3)] = self.grid[:, :, (2)]

    def forward(self, input1, input2):
        self.batchgrid3d = torch.zeros(torch.Size([input1.size(0)]) + self.grid3d.size())
        for i in range(input1.size(0)):
            self.batchgrid3d[i] = self.grid3d
        self.batchgrid3d = Variable(self.batchgrid3d)
        self.batchgrid = torch.zeros(torch.Size([input1.size(0)]) + self.grid.size())
        for i in range(input1.size(0)):
            self.batchgrid[i] = self.grid
        self.batchgrid = Variable(self.batchgrid)
        x = torch.sum(torch.mul(self.batchgrid3d, input1[:, :, :, 0:4]), 3)
        y = torch.sum(torch.mul(self.batchgrid3d, input1[:, :, :, 4:8]), 3)
        z = torch.sum(torch.mul(self.batchgrid3d, input1[:, :, :, 8:]), 3)
        r = torch.sqrt(x ** 2 + y ** 2 + z ** 2) + 1e-05
        theta = torch.acos(z / r) / (np.pi / 2) - 1
        phi = torch.atan(y / (x + 1e-05)) + np.pi * x.lt(0).type(torch.FloatTensor) * (y.ge(0).type(torch.FloatTensor) - y.lt(0).type(torch.FloatTensor))
        phi = phi / np.pi
        input_u = input2.view(-1, 1, 1, 1).repeat(1, self.height, self.width, 1)
        output = torch.cat([theta, phi], 3)
        output1 = torch.atan(torch.tan(np.pi / 2.0 * (output[:, :, :, 1:2] + self.batchgrid[:, :, :, 2:] * input_u[:, :, :, :]))) / (np.pi / 2)
        output2 = torch.cat([output[:, :, :, 0:1], output1], 3)
        return output2


class Depth3DGridGen(Module):

    def __init__(self, height, width, lr=1, aux_loss=False):
        super(Depth3DGridGen, self).__init__()
        self.height, self.width = height, width
        self.aux_loss = aux_loss
        self.lr = lr
        self.grid = np.zeros([self.height, self.width, 3], dtype=np.float32)
        self.grid[:, :, (0)] = np.expand_dims(np.repeat(np.expand_dims(np.arange(-1, 1, 2.0 / self.height), 0), repeats=self.width, axis=0).T, 0)
        self.grid[:, :, (1)] = np.expand_dims(np.repeat(np.expand_dims(np.arange(-1, 1, 2.0 / self.width), 0), repeats=self.height, axis=0), 0)
        self.grid[:, :, (2)] = np.ones([self.height, width])
        self.grid = torch.from_numpy(self.grid.astype(np.float32))
        self.theta = self.grid[:, :, (0)] * np.pi / 2 + np.pi / 2
        self.phi = self.grid[:, :, (1)] * np.pi
        self.x = torch.sin(self.theta) * torch.cos(self.phi)
        self.y = torch.sin(self.theta) * torch.sin(self.phi)
        self.z = torch.cos(self.theta)
        self.grid3d = torch.from_numpy(np.zeros([self.height, self.width, 4], dtype=np.float32))
        self.grid3d[:, :, (0)] = self.x
        self.grid3d[:, :, (1)] = self.y
        self.grid3d[:, :, (2)] = self.z
        self.grid3d[:, :, (3)] = self.grid[:, :, (2)]

    def forward(self, depth, trans0, trans1, rotate):
        self.batchgrid3d = torch.zeros(torch.Size([depth.size(0)]) + self.grid3d.size())
        for i in range(depth.size(0)):
            self.batchgrid3d[i] = self.grid3d
        self.batchgrid3d = Variable(self.batchgrid3d)
        self.batchgrid = torch.zeros(torch.Size([depth.size(0)]) + self.grid.size())
        for i in range(depth.size(0)):
            self.batchgrid[i] = self.grid
        self.batchgrid = Variable(self.batchgrid)
        x = self.batchgrid3d[:, :, :, 0:1] * depth + trans0.view(-1, 1, 1, 1).repeat(1, self.height, self.width, 1)
        y = self.batchgrid3d[:, :, :, 1:2] * depth + trans1.view(-1, 1, 1, 1).repeat(1, self.height, self.width, 1)
        z = self.batchgrid3d[:, :, :, 2:3] * depth
        r = torch.sqrt(x ** 2 + y ** 2 + z ** 2) + 1e-05
        theta = torch.acos(z / r) / (np.pi / 2) - 1
        phi = torch.atan(y / (x + 1e-05)) + np.pi * x.lt(0).type(torch.FloatTensor) * (y.ge(0).type(torch.FloatTensor) - y.lt(0).type(torch.FloatTensor))
        phi = phi / np.pi
        input_u = rotate.view(-1, 1, 1, 1).repeat(1, self.height, self.width, 1)
        output = torch.cat([theta, phi], 3)
        output1 = torch.atan(torch.tan(np.pi / 2.0 * (output[:, :, :, 1:2] + self.batchgrid[:, :, :, 2:] * input_u[:, :, :, :]))) / (np.pi / 2)
        output2 = torch.cat([output[:, :, :, 0:1], output1], 3)
        return output2


class Depth3DGridGen_with_mask(Module):

    def __init__(self, height, width, lr=1, aux_loss=False, ray_tracing=False):
        super(Depth3DGridGen_with_mask, self).__init__()
        self.height, self.width = height, width
        self.aux_loss = aux_loss
        self.lr = lr
        self.ray_tracing = ray_tracing
        self.grid = np.zeros([self.height, self.width, 3], dtype=np.float32)
        self.grid[:, :, (0)] = np.expand_dims(np.repeat(np.expand_dims(np.arange(-1, 1, 2.0 / self.height), 0), repeats=self.width, axis=0).T, 0)
        self.grid[:, :, (1)] = np.expand_dims(np.repeat(np.expand_dims(np.arange(-1, 1, 2.0 / self.width), 0), repeats=self.height, axis=0), 0)
        self.grid[:, :, (2)] = np.ones([self.height, width])
        self.grid = torch.from_numpy(self.grid.astype(np.float32))
        self.theta = self.grid[:, :, (0)] * np.pi / 2 + np.pi / 2
        self.phi = self.grid[:, :, (1)] * np.pi
        self.x = torch.sin(self.theta) * torch.cos(self.phi)
        self.y = torch.sin(self.theta) * torch.sin(self.phi)
        self.z = torch.cos(self.theta)
        self.grid3d = torch.from_numpy(np.zeros([self.height, self.width, 4], dtype=np.float32))
        self.grid3d[:, :, (0)] = self.x
        self.grid3d[:, :, (1)] = self.y
        self.grid3d[:, :, (2)] = self.z
        self.grid3d[:, :, (3)] = self.grid[:, :, (2)]

    def forward(self, depth, trans0, trans1, rotate):
        self.batchgrid3d = torch.zeros(torch.Size([depth.size(0)]) + self.grid3d.size())
        for i in range(depth.size(0)):
            self.batchgrid3d[i] = self.grid3d
        self.batchgrid3d = Variable(self.batchgrid3d)
        self.batchgrid = torch.zeros(torch.Size([depth.size(0)]) + self.grid.size())
        for i in range(depth.size(0)):
            self.batchgrid[i] = self.grid
        self.batchgrid = Variable(self.batchgrid)
        if depth.is_cuda:
            self.batchgrid = self.batchgrid
            self.batchgrid3d = self.batchgrid3d
        x_ = self.batchgrid3d[:, :, :, 0:1] * depth + trans0.view(-1, 1, 1, 1).repeat(1, self.height, self.width, 1)
        y_ = self.batchgrid3d[:, :, :, 1:2] * depth + trans1.view(-1, 1, 1, 1).repeat(1, self.height, self.width, 1)
        z = self.batchgrid3d[:, :, :, 2:3] * depth
        rotate_z = rotate.view(-1, 1, 1, 1).repeat(1, self.height, self.width, 1) * np.pi
        x = x_ * torch.cos(rotate_z) - y_ * torch.sin(rotate_z)
        y = x_ * torch.sin(rotate_z) + y_ * torch.cos(rotate_z)
        r = torch.sqrt(x ** 2 + y ** 2 + z ** 2) + 1e-05
        theta = torch.acos(z / r) / (np.pi / 2) - 1
        if depth.is_cuda:
            phi = torch.atan(y / (x + 1e-05)) + np.pi * x.lt(0).type(torch.FloatTensor) * (y.ge(0).type(torch.FloatTensor) - y.lt(0).type(torch.FloatTensor))
        else:
            phi = torch.atan(y / (x + 1e-05)) + np.pi * x.lt(0).type(torch.FloatTensor) * (y.ge(0).type(torch.FloatTensor) - y.lt(0).type(torch.FloatTensor))
        phi = phi / np.pi
        output = torch.cat([theta, phi], 3)
        return output


class STNMFunction(Function):

    def forward(self, canvas, fgimg, fggrid, fgmask):
        self.canvas = canvas
        self.fgimg = fgimg
        self.fggrid = fggrid
        self.fgmask = fgmask
        output = torch.zeros(canvas.size()[0], canvas.size()[1], canvas.size()[2], canvas.size()[3])
        if not canvas.is_cuda:
            None
        else:
            output = output
            stnm.BilinearSamplerBHWD_updateOutput_cuda(canvas, fgimg, fggrid, fgmask, output)
        return output

    def backward(self, grad_output):
        grad_canvas = torch.zeros(self.canvas.size())
        grad_fgimg = torch.zeros(self.fgimg.size())
        grad_fggrid = torch.zeros(self.fggrid.size())
        grad_fgmask = torch.zeros(self.fgmask.size())
        if not grad_output.is_cuda:
            None
        else:
            grad_output = grad_output.contiguous()
            grad_canvas = grad_canvas.contiguous()
            grad_fgimg = grad_fgimg.contiguous()
            grad_fggrid = grad_fggrid.contiguous()
            grad_fgmask = grad_fgmask.contiguous()
            stnm.BilinearSamplerBHWD_updateGradInput_cuda(self.canvas, self.fgimg, self.fggrid, self.fgmask, grad_canvas, grad_fgimg, grad_fggrid, grad_fgmask, grad_output)
        return grad_canvas, grad_fgimg, grad_fggrid, grad_fgmask


class STNM(Module):

    def __init__(self):
        super(STNM, self).__init__()
        self.f = STNMFunction()

    def forward(self, canvas, fgimg, fggrid, fgmask):
        return self.f(canvas, fgimg, fggrid, fgmask)


parser = argparse.ArgumentParser()


opt = parser.parse_args()


class _netG(nn.Module):

    def __init__(self, ngpu, nsize):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        self.nsize_out = 2
        self.lstmcell = nn.LSTMCell(nz, nz)
        self.Gbgc, self.depth_in_bg = self.buildNetGbg(nsize)
        self.Gbgi = nn.Sequential(nn.ConvTranspose2d(self.depth_in_bg, nc, 4, 2, 1, bias=True), nn.Tanh())
        self.Gfgc, self.depth_in = self.buildNetGfg(nsize)
        self.Gfgi = nn.Sequential(nn.ConvTranspose2d(self.depth_in, nc, 4, 2, 1, bias=False), nn.Tanh())
        self.Gfgm = nn.Sequential(nn.ConvTranspose2d(self.depth_in, 1, 4, 2, 1, bias=True), nn.Sigmoid())
        self.Gtransform = nn.Linear(nz, 6)
        self.Gtransform.weight.data.zero_()
        self.Gtransform.bias.data.zero_()
        self.Gtransform.bias.data[0] = opt.maxobjscale
        self.Gtransform.bias.data[4] = opt.maxobjscale
        self.Ggrid = AffineGridGen(nsize, nsize, aux_loss=False)
        self.Compositors = []
        for t in range(ntimestep - 1):
            self.Compositors.append(STNM())
        self.encoderconv = self.buildEncoderConv(self.depth_in, nsize // 2, self.nsize_out)
        self.encoderfc = self.buildEncoderFC(self.depth_in, self.nsize_out, nz)
        self.nlnet = nn.Sequential(nn.Linear(nz + nz, nz), nn.BatchNorm1d(nz), nn.Tanh())

    def buildNetGbg(self, nsize):
        net = nn.Sequential()
        size_map = 1
        name = str(size_map)
        net.add_module('convt' + name, nn.ConvTranspose2d(nz, ngf * 4, 4, 4, 0, bias=True))
        net.add_module('bn' + name, nn.BatchNorm2d(ngf * 4))
        net.add_module('relu' + name, nn.ReLU(True))
        size_map = 4
        depth_in = 4 * ngf
        depth_out = 2 * ngf
        while size_map < nsize / 2:
            name = str(size_map)
            net.add_module('convt' + name, nn.ConvTranspose2d(depth_in, depth_out, 4, 2, 1, bias=True))
            net.add_module('bn' + name, nn.BatchNorm2d(depth_out))
            net.add_module('relu' + name, nn.ReLU(True))
            depth_in = depth_out
            depth_out = max(depth_in // 2, 64)
            size_map = size_map * 2
        return net, depth_in

    def buildNetGfg(self, nsize):
        net = nn.Sequential()
        size_map = 1
        name = str(size_map)
        net.add_module('convt' + name, nn.ConvTranspose2d(nz, ngf * 8, 4, 4, 0, bias=False))
        net.add_module('bn' + name, nn.BatchNorm2d(ngf * 8))
        net.add_module('relu' + name, nn.ReLU(True))
        size_map = 4
        depth_in = 8 * ngf
        depth_out = 4 * ngf
        while size_map < nsize / 2:
            name = str(size_map)
            net.add_module('convt' + name, nn.ConvTranspose2d(depth_in, depth_out, 4, 2, 1, bias=False))
            net.add_module('bn' + name, nn.BatchNorm2d(depth_out))
            net.add_module('relu' + name, nn.ReLU(True))
            depth_in = depth_out
            depth_out = max(depth_in // 2, 64)
            size_map = size_map * 2
        return net, depth_in

    def buildEncoderConv(self, depth_in, nsize_in, nsize_out):
        net = nn.Sequential()
        nsize_i = nsize_in
        while nsize_i > nsize_out:
            name = str(nsize_i)
            net.add_module('avgpool' + name, nn.AvgPool2d(4, 2, 1))
            net.add_module('bn' + name, nn.BatchNorm2d(depth_in))
            net.add_module('lrelu' + name, nn.LeakyReLU(0.2, inplace=True))
            nsize_i = nsize_i // 2
        return net

    def buildEncoderFC(self, depth_in, nsize_in, out_dim):
        net = nn.Sequential(nn.Linear(depth_in * nsize_in * nsize_in, out_dim), nn.BatchNorm1d(out_dim), nn.Tanh())
        return net

    def clampT(self, Tin):
        x_s = Tin.select(1, 0)
        x_r = Tin.select(1, 1)
        x_t = Tin.select(1, 2)
        y_r = Tin.select(1, 3)
        y_s = Tin.select(1, 4)
        y_t = Tin.select(1, 5)
        x_s_clamp = torch.unsqueeze(x_s.clamp(opt.maxobjscale, 2 * opt.maxobjscale), 1)
        x_r_clmap = torch.unsqueeze(x_r.clamp(-rot, rot), 1)
        x_t_clmap = torch.unsqueeze(x_t.clamp(-1.0, 1.0), 1)
        y_r_clamp = torch.unsqueeze(y_r.clamp(-rot, rot), 1)
        y_s_clamp = torch.unsqueeze(y_s.clamp(opt.maxobjscale, 2 * opt.maxobjscale), 1)
        y_t_clamp = torch.unsqueeze(y_t.clamp(-1.0, 1.0), 1)
        Tout = torch.cat([x_s_clamp, x_r_clmap, x_t_clmap, y_r_clamp, y_s_clamp, y_t_clamp], 1)
        return Tout

    def forward(self, input):
        batchSize = input.size()[1]
        hx = Variable(torch.zeros(batchSize, nz))
        cx = Variable(torch.zeros(batchSize, nz))
        outputsT = []
        fgimgsT = []
        fgmaskT = []
        for i in range(ntimestep):
            hx, cx = self.lstmcell(input[i], (hx, cx))
            hx_view = hx.contiguous().view(batchSize, nz, 1, 1)
            if i == 0:
                input_view = input[i].view(batchSize, nz, 1, 1)
                bgc = self.Gbgc(input_view)
                canvas = self.Gbgi(bgc)
                outputsT.append(canvas)
            else:
                if ntimestep > 2 and i == 1:
                    encConv = self.encoderconv(bgc)
                    encConv_view = encConv.view(batchSize, self.depth_in * self.nsize_out * self.nsize_out)
                    encFC = self.encoderfc(encConv_view)
                    concat = torch.cat([hx, encFC], 1)
                    comb = self.nlnet(concat)
                    input4g = comb
                    input4g_view = input4g.contiguous().view(batchSize, nz, 1, 1)
                elif ntimestep > 2 and i > 1:
                    encConv = self.encoderconv(fgc)
                    encConv_view = encConv.view(batchSize, self.depth_in * self.nsize_out * self.nsize_out)
                    encFC = self.encoderfc(encConv_view)
                    concat = torch.cat([hx, encFC], 1)
                    comb = self.nlnet(concat)
                    input4g = comb
                    input4g_view = input4g.contiguous().view(batchSize, nz, 1, 1)
                else:
                    input4g = hx
                    input4g_view = hx_view
                fgc = self.Gfgc(input4g_view)
                fgi = self.Gfgi(fgc)
                fgm = self.Gfgm(fgc)
                fgt = self.Gtransform(input4g)
                fgt_clamp = self.clampT(fgt)
                fgt_view = fgt_clamp.contiguous().view(batchSize, 2, 3)
                fgg = self.Ggrid(fgt_view)
                canvas4c = canvas.permute(0, 2, 3, 1).contiguous()
                fgi4c = fgi.permute(0, 2, 3, 1).contiguous()
                fgm4c = fgm.permute(0, 2, 3, 1).contiguous()
                temp = self.Compositors[i - 1](canvas4c, fgi4c, fgg, fgm4c)
                canvas = temp.permute(0, 3, 1, 2).contiguous()
                outputsT.append(canvas)
                fgimgsT.append(fgi)
                fgmaskT.append(fgm)
        return outputsT[ntimestep - 1], outputsT, fgimgsT, fgmaskT


class _netD(nn.Module):

    def __init__(self, ngpu, nsize):
        super(_netD, self).__init__()
        self.ngpu = ngpu
        self.main = self.buildNet(nsize)

    def buildNet(self, nsize):
        net = nn.Sequential()
        depth_in = nc
        depth_out = ndf
        size_map = nsize
        while size_map > 4:
            name = str(size_map)
            net.add_module('conv' + name, nn.Conv2d(depth_in, depth_out, 4, 2, 1, bias=False))
            if size_map < nsize:
                net.add_module('bn' + name, nn.BatchNorm2d(depth_out))
            net.add_module('lrelu' + name, nn.LeakyReLU(0.2, inplace=True))
            depth_in = depth_out
            depth_out = 2 * depth_in
            size_map = size_map // 2
        name = str(size_map)
        net.add_module('conv' + name, nn.Conv2d(depth_in, 1, 4, 1, 0, bias=False))
        net.add_module('sigmoid' + name, nn.Sigmoid())
        return net

    def forward(self, input):
        if isinstance(input.data, torch.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output.view(-1, 1)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Depth3DGridGen,
     lambda: ([], {'height': 4, 'width': 4}),
     lambda: ([torch.rand([256, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (Depth3DGridGen_with_mask,
     lambda: ([], {'height': 4, 'width': 4}),
     lambda: ([torch.rand([256, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_jwyang_lr_gan_pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

