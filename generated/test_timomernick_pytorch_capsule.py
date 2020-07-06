import sys
_module = sys.modules[__name__]
del sys
master = _module
capsule_conv_layer = _module
capsule_layer = _module
capsule_network = _module
main = _module

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


import torch.nn as nn


import torch.optim as optim


from torch.autograd import Variable


from torchvision import datasets


from torchvision import transforms


import torch.nn.functional as F


import torchvision.utils as vutils


class CapsuleConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(CapsuleConvLayer, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=9, stride=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv0(x))


class ConvUnit(nn.Module):

    def __init__(self, in_channels):
        super(ConvUnit, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=9, stride=2, bias=True)

    def forward(self, x):
        return self.conv0(x)


class CapsuleLayer(nn.Module):

    def __init__(self, in_units, in_channels, num_units, unit_size, use_routing):
        super(CapsuleLayer, self).__init__()
        self.in_units = in_units
        self.in_channels = in_channels
        self.num_units = num_units
        self.use_routing = use_routing
        if self.use_routing:
            self.W = nn.Parameter(torch.randn(1, in_channels, num_units, unit_size, in_units))
        else:

            def create_conv_unit(unit_idx):
                unit = ConvUnit(in_channels=in_channels)
                self.add_module('unit_' + str(unit_idx), unit)
                return unit
            self.units = [create_conv_unit(i) for i in range(self.num_units)]

    @staticmethod
    def squash(s):
        mag_sq = torch.sum(s ** 2, dim=2, keepdim=True)
        mag = torch.sqrt(mag_sq)
        s = mag_sq / (1.0 + mag_sq) * (s / mag)
        return s

    def forward(self, x):
        if self.use_routing:
            return self.routing(x)
        else:
            return self.no_routing(x)

    def no_routing(self, x):
        u = [self.units[i](x) for i in range(self.num_units)]
        u = torch.stack(u, dim=1)
        u = u.view(x.size(0), self.num_units, -1)
        return CapsuleLayer.squash(u)

    def routing(self, x):
        batch_size = x.size(0)
        x = x.transpose(1, 2)
        x = torch.stack([x] * self.num_units, dim=2).unsqueeze(4)
        W = torch.cat([self.W] * batch_size, dim=0)
        u_hat = torch.matmul(W, x)
        b_ij = Variable(torch.zeros(1, self.in_channels, self.num_units, 1))
        num_iterations = 3
        for iteration in range(num_iterations):
            c_ij = F.softmax(b_ij)
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            v_j = CapsuleLayer.squash(s_j)
            v_j1 = torch.cat([v_j] * self.in_channels, dim=1)
            u_vj1 = torch.matmul(u_hat.transpose(3, 4), v_j1).squeeze(4).mean(dim=0, keepdim=True)
            b_ij = b_ij + u_vj1
        return v_j.squeeze(1)


class CapsuleNetwork(nn.Module):

    def __init__(self, image_width, image_height, image_channels, conv_inputs, conv_outputs, num_primary_units, primary_unit_size, num_output_units, output_unit_size):
        super(CapsuleNetwork, self).__init__()
        self.reconstructed_image_count = 0
        self.image_channels = image_channels
        self.image_width = image_width
        self.image_height = image_height
        self.conv1 = CapsuleConvLayer(in_channels=conv_inputs, out_channels=conv_outputs)
        self.primary = CapsuleLayer(in_units=0, in_channels=conv_outputs, num_units=num_primary_units, unit_size=primary_unit_size, use_routing=False)
        self.digits = CapsuleLayer(in_units=num_primary_units, in_channels=primary_unit_size, num_units=num_output_units, unit_size=output_unit_size, use_routing=True)
        reconstruction_size = image_width * image_height * image_channels
        self.reconstruct0 = nn.Linear(num_output_units * output_unit_size, int(reconstruction_size * 2 / 3))
        self.reconstruct1 = nn.Linear(int(reconstruction_size * 2 / 3), int(reconstruction_size * 3 / 2))
        self.reconstruct2 = nn.Linear(int(reconstruction_size * 3 / 2), reconstruction_size)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.digits(self.primary(self.conv1(x)))

    def loss(self, images, input, target, size_average=True):
        return self.margin_loss(input, target, size_average) + self.reconstruction_loss(images, input, size_average)

    def margin_loss(self, input, target, size_average=True):
        batch_size = input.size(0)
        v_mag = torch.sqrt((input ** 2).sum(dim=2, keepdim=True))
        zero = Variable(torch.zeros(1))
        m_plus = 0.9
        m_minus = 0.1
        max_l = torch.max(m_plus - v_mag, zero).view(batch_size, -1) ** 2
        max_r = torch.max(v_mag - m_minus, zero).view(batch_size, -1) ** 2
        loss_lambda = 0.5
        T_c = target
        L_c = T_c * max_l + loss_lambda * (1.0 - T_c) * max_r
        L_c = L_c.sum(dim=1)
        if size_average:
            L_c = L_c.mean()
        return L_c

    def reconstruction_loss(self, images, input, size_average=True):
        v_mag = torch.sqrt((input ** 2).sum(dim=2))
        _, v_max_index = v_mag.max(dim=1)
        v_max_index = v_max_index.data
        batch_size = input.size(0)
        all_masked = [None] * batch_size
        for batch_idx in range(batch_size):
            input_batch = input[batch_idx]
            batch_masked = Variable(torch.zeros(input_batch.size()))
            batch_masked[v_max_index[batch_idx]] = input_batch[v_max_index[batch_idx]]
            all_masked[batch_idx] = batch_masked
        masked = torch.stack(all_masked, dim=0)
        masked = masked.view(input.size(0), -1)
        output = self.relu(self.reconstruct0(masked))
        output = self.relu(self.reconstruct1(output))
        output = self.sigmoid(self.reconstruct2(output))
        output = output.view(-1, self.image_channels, self.image_height, self.image_width)
        if self.reconstructed_image_count % 10 == 0:
            if output.size(1) == 2:
                zeros = torch.zeros(output.size(0), 1, output.size(2), output.size(3))
                output_image = torch.cat([zeros, output.data.cpu()], dim=1)
            else:
                output_image = output.data.cpu()
            vutils.save_image(output_image, 'reconstruction.png')
        self.reconstructed_image_count += 1
        error = (output - images).view(output.size(0), -1)
        error = error ** 2
        error = torch.sum(error, dim=1) * 0.0005
        if size_average:
            error = error.mean()
        return error


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (CapsuleConvLayer,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     True),
    (CapsuleLayer,
     lambda: ([], {'in_units': 4, 'in_channels': 4, 'num_units': 4, 'unit_size': 4, 'use_routing': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (ConvUnit,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     True),
]

class Test_timomernick_pytorch_capsule(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

