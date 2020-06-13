import sys
_module = sys.modules[__name__]
del sys
davis = _module
demo_retarget = _module
demo_vi = _module
lib = _module
resample2d_package = _module
_ext = _module
resample2d = _module
build = _module
functions = _module
modules = _module
resample2d = _module
model = _module
ConvLSTM = _module
models = _module
correlation_package = _module
correlation = _module
correlation = _module
flow_modules = _module
gated_conv = _module
utils = _module
vinet = _module
utils = _module

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


from torch.nn.modules.module import Module


import torch


from torch import nn


import torch.nn.functional as f


import torch.nn as nn


import torch.nn.functional as F


from torch.autograd import Variable


import math


from torch.backends import cudnn


from random import *


import numpy as np


import random


from torch.utils.data.sampler import Sampler


from torch.utils.data import DataLoader


class Resample2dFunction(Function):

    def __init__(self, kernel_size=1):
        super(Resample2dFunction, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, input1, input2):
        self.save_for_backward(input1, input2)
        assert input1.is_contiguous() == True
        assert input2.is_contiguous() == True
        with torch.cuda.device_of(input1):
            _, d, _, _ = input1.size()
            b, _, h, w = input2.size()
            output = input1.new().resize_(b, d, h, w).zero_()
            resample2d.Resample2d_cuda_forward(input1, input2, output, self
                .kernel_size)
        return output

    def backward(self, gradOutput):
        input1, input2 = self.saved_tensors
        assert gradOutput.is_contiguous() == True
        with torch.cuda.device_of(input1):
            b, c, h, w = input1.size()
            gradInput1 = input1.new().resize_(b, c, h, w).zero_()
            b, c, h, w = input2.size()
            gradInput2 = input2.new().resize_(b, c, h, w).zero_()
            resample2d.Resample2d_cuda_backward(input1, input2, gradOutput,
                gradInput1, gradInput2, self.kernel_size)
        return gradInput1, gradInput2


class Resample2d(Module):

    def __init__(self, kernel_size=1):
        super(Resample2d, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, input1, input2):
        input1_c = input1.contiguous()
        result = Resample2dFunction(self.kernel_size)(input1_c, input2)
        return result


class ConvLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, kernel_size, type='2d'):
        super(ConvLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        pad = kernel_size // 2
        if type == '3d':
            self.Gates = nn.Conv3d(input_size + hidden_size, 4 *
                hidden_size, (1, kernel_size, kernel_size), padding=(0, pad,
                pad))
        else:
            self.Gates = nn.Conv2d(input_size + hidden_size, 4 *
                hidden_size, kernel_size, padding=pad)

    def forward(self, input_, prev_state=None):
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = torch.zeros(state_size).to(input_.device
                ), torch.zeros(state_size).to(input_.device)
        prev_hidden, prev_cell = prev_state
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)
        in_gate = f.sigmoid(in_gate)
        remember_gate = f.sigmoid(remember_gate)
        out_gate = f.sigmoid(out_gate)
        cell_gate = f.tanh(cell_gate)
        cell = remember_gate * prev_cell + in_gate * cell_gate
        hidden = out_gate * f.tanh(cell)
        return hidden, cell


class CorrelationFunction(Function):

    def __init__(self, pad_size=3, kernel_size=3, max_displacement=20,
        stride1=1, stride2=2, corr_multiply=1):
        super(CorrelationFunction, self).__init__()
        self.pad_size = pad_size
        self.kernel_size = kernel_size
        self.max_displacement = max_displacement
        self.stride1 = stride1
        self.stride2 = stride2
        self.corr_multiply = corr_multiply

    def forward(self, input1, input2):
        self.save_for_backward(input1, input2)
        assert input1.is_contiguous() == True
        assert input2.is_contiguous() == True
        with torch.cuda.device_of(input1):
            rbot1 = input1.new()
            rbot2 = input2.new()
            output = input1.new()
            correlation.Correlation_forward_cuda(input1, input2, rbot1,
                rbot2, output, self.pad_size, self.kernel_size, self.
                max_displacement, self.stride1, self.stride2, self.
                corr_multiply)
        return output

    def backward(self, grad_output):
        input1, input2 = self.saved_tensors
        assert grad_output.is_contiguous() == True
        with torch.cuda.device_of(input1):
            rbot1 = input1.new()
            rbot2 = input2.new()
            grad_input1 = input1.new()
            grad_input2 = input2.new()
            correlation.Correlation_backward_cuda(input1, input2, rbot1,
                rbot2, grad_output, grad_input1, grad_input2, self.pad_size,
                self.kernel_size, self.max_displacement, self.stride1, self
                .stride2, self.corr_multiply)
        return grad_input1, grad_input2


class Correlation(Module):

    def __init__(self, pad_size=0, kernel_size=0, max_displacement=0,
        stride1=1, stride2=2, corr_multiply=1):
        super(Correlation, self).__init__()
        self.pad_size = pad_size
        self.kernel_size = kernel_size
        self.max_displacement = max_displacement
        self.stride1 = stride1
        self.stride2 = stride2
        self.corr_multiply = corr_multiply

    def forward(self, input1, input2):
        result = CorrelationFunction(self.pad_size, self.kernel_size, self.
            max_displacement, self.stride1, self.stride2, self.corr_multiply)(
            input1, input2)
        return result


def conv_(batch_norm, in_planes, out_planes, kernel_size=3, stride=1,
    dilation=1):
    if batch_norm:
        return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=
            kernel_size, stride=stride, dilation=dilation, padding=(
            kernel_size - 1) * dilation // 2, bias=False), nn.BatchNorm2d(
            out_planes), nn.LeakyReLU(0.1, inplace=True))
    else:
        return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=
            kernel_size, stride=stride, dilation=dilation, padding=(
            kernel_size - 1) * dilation // 2, bias=True), nn.LeakyReLU(0.1,
            inplace=True))


class MaskEstimator_(nn.Module):

    def __init__(self, args, ch_in):
        super(MaskEstimator_, self).__init__()
        self.args = args
        self.convs = nn.Sequential(conv_(False, ch_in, ch_in // 2), conv_(
            False, ch_in // 2, ch_in // 2), nn.Conv2d(in_channels=ch_in // 
            2, out_channels=1, kernel_size=3, stride=1, padding=1, dilation
            =1, groups=1, bias=True), nn.Sigmoid())

    def forward(self, x):
        return self.convs(x)


def get_grid(x):
    torchHorizontal = torch.linspace(-1.0, 1.0, x.size(3)).view(1, 1, 1, x.
        size(3)).expand(x.size(0), 1, x.size(2), x.size(3))
    torchVertical = torch.linspace(-1.0, 1.0, x.size(2)).view(1, 1, x.size(
        2), 1).expand(x.size(0), 1, x.size(2), x.size(3))
    grid = torch.cat([torchHorizontal, torchVertical], 1)
    return grid


class WarpingLayer(nn.Module):

    def __init__(self):
        super(WarpingLayer, self).__init__()

    def forward(self, x, flow):
        flow_for_grip = torch.zeros_like(flow)
        flow_for_grip[:, (0), :, :] = flow[:, (0), :, :] / ((flow.size(3) -
            1.0) / 2.0)
        flow_for_grip[:, (1), :, :] = flow[:, (1), :, :] / ((flow.size(2) -
            1.0) / 2.0)
        grid = (get_grid(x) + flow_for_grip).permute(0, 2, 3, 1)
        x_warp = F.grid_sample(x, grid)
        return x_warp


def conv(batch_norm, in_planes, out_planes, kernel_size=3, stride=1, dilation=1
    ):
    if batch_norm:
        return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=
            kernel_size, stride=stride, dilation=dilation, padding=(
            kernel_size - 1) * dilation // 2, bias=False), nn.BatchNorm2d(
            out_planes), nn.LeakyReLU(0.1, inplace=True))
    else:
        return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=
            kernel_size, stride=stride, dilation=dilation, padding=(
            kernel_size - 1) * dilation // 2, bias=True), nn.LeakyReLU(0.1,
            inplace=True))


class ContextNetwork(nn.Module):

    def __init__(self, args, ch_in):
        super(ContextNetwork, self).__init__()
        self.args = args
        self.convs = nn.Sequential(conv(args.batch_norm, ch_in, 128, 3, 1, 
            1), conv(args.batch_norm, 128, 128, 3, 1, 2), conv(args.
            batch_norm, 128, 128, 3, 1, 4), conv(args.batch_norm, 128, 96, 
            3, 1, 8), conv(args.batch_norm, 96, 64, 3, 1, 16), conv(args.
            batch_norm, 64, 32, 3, 1, 1), conv(args.batch_norm, 32, 2, 3, 1, 1)
            )

    def forward(self, x):
        return self.convs(x)


class LongFlowEstimatorCorr(nn.Module):

    def __init__(self, args, ch_in):
        super(LongFlowEstimatorCorr, self).__init__()
        self.args = args
        self.convs = nn.Sequential(conv(args.batch_norm, ch_in, 128), conv(
            args.batch_norm, 128, 128), conv(args.batch_norm, 128, 96),
            conv(args.batch_norm, 96, 64), conv(args.batch_norm, 64, 32))
        self.conv1 = nn.Conv2d(in_channels=32, out_channels=2, kernel_size=
            3, stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.convs_fine = ContextNetwork(args, 32 + 2)

    def forward(self, x):
        x = self.convs(x)
        flo_coarse = self.conv1(x)
        flo_fine = self.convs_fine(torch.cat([x, flo_coarse], 1))
        flo = flo_coarse + flo_fine
        return flo


class LongFlowNetCorr(nn.Module):

    def __init__(self, args, in_ch):
        super(LongFlowNetCorr, self).__init__()
        self.args = args
        self.corr = Correlation(pad_size=args.search_range, kernel_size=1,
            max_displacement=args.search_range, stride1=1, stride2=1,
            corr_multiply=1)
        self.flow_estimator = LongFlowEstimatorCorr(args, in_ch + (args.
            search_range * 2 + 1) ** 2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                if m.bias is not None:
                    nn.init.uniform_(m.bias)
                nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.
                ConvTranspose3d):
                if m.bias is not None:
                    nn.init.uniform_(m.bias)
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x1, x2, upflow=None):
        corr = self.corr(x1.contiguous(), x2.contiguous())
        if upflow is not None:
            flow = self.flow_estimator(torch.cat([x1, corr, upflow], dim=1))
        else:
            flow = self.flow_estimator(torch.cat([x1, corr], dim=1))
        return flow


class GatedConvolution(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
        dilation=1, padding=0, bias=False, type='3d', status='train'):
        super(GatedConvolution, self).__init__()
        assert type in ['2d', '3d']
        assert status in ['train', 'test']
        self.status = status
        self.type = type
        if type == '3d':
            self.conv = nn.Conv3d(in_channels, out_channels * 2,
                kernel_size, stride=stride, dilation=dilation, padding=
                padding, bias=bias)
        elif type == '2d':
            self.conv = nn.Conv2d(in_channels, out_channels * 2,
                kernel_size, stride=stride, dilation=dilation, padding=
                padding, bias=bias)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        c = x.size(1)
        phi, gate = torch.split(x, c // 2, 1)
        if self.status == 'train':
            return torch.sigmoid(gate) * self.relu(phi)
        else:
            return torch.sigmoid(gate) * self.relu(phi), torch.sigmoid(gate)


class GatedUpConvolution(nn.Module):

    def __init__(self, size, in_channels, out_channels, kernel_size, stride,
        padding, bias, mode='trilinear', type='3d', status='train'):
        super(GatedUpConvolution, self).__init__()
        assert type in ['2d', '3d']
        assert status in ['train', 'test']
        self.status = status
        self.type = type
        self.leaky_relu = nn.LeakyReLU(0.2)
        if type == '3d':
            self.conv = nn.Sequential(nn.Upsample(size=size, mode=mode), nn
                .Conv3d(in_channels, out_channels * 2, kernel_size, stride=
                stride, padding=padding, bias=bias))
        elif type == '2d':
            self.conv = nn.Sequential(nn.Upsample(size=size, mode=mode), nn
                .Conv2d(in_channels, out_channels * 2, kernel_size, stride=
                stride, padding=padding, bias=bias))

    def forward(self, x):
        x = self.conv(x)
        c = x.size(1)
        phi, gate = torch.split(x, c // 2, 1)
        if self.status == 'train':
            return torch.sigmoid(gate) * self.leaky_relu(phi)
        else:
            return torch.sigmoid(gate) * self.leaky_relu(phi), torch.sigmoid(
                gate)


class VI_2D_Encoder_3(nn.Module):

    def __init__(self, opt):
        super(VI_2D_Encoder_3, self).__init__()
        self.opt = opt
        st = 2 if self.opt.double_size else 1
        self.ec0 = GatedConvolution(5, 32, kernel_size=(3, 3), stride=(st,
            st), padding=(1, 1), bias=False, type='2d')
        self.ec1 = GatedConvolution(32, 64, kernel_size=(3, 3), stride=(2, 
            2), padding=(1, 1), bias=False, type='2d')
        self.ec2 = GatedConvolution(64, 64, kernel_size=(3, 3), stride=(1, 
            1), padding=(1, 1), bias=False, type='2d')
        self.ec3_1 = GatedConvolution(64, 96, kernel_size=(3, 3), stride=(2,
            2), padding=(1, 1), bias=False, type='2d')
        self.ec3_2 = GatedConvolution(96, 96, kernel_size=(3, 3), stride=(1,
            1), padding=(1, 1), bias=False, type='2d')
        self.ec4_1 = GatedConvolution(96, 128, kernel_size=(3, 3), stride=(
            2, 2), padding=(1, 1), bias=False, type='2d')
        self.ec4 = GatedConvolution(128, 128, kernel_size=(3, 3), stride=(1,
            1), padding=(1, 1), bias=False, type='2d')
        self.ec5 = GatedConvolution(128, 128, kernel_size=(3, 3), stride=(1,
            1), padding=(1, 1), bias=False, type='2d')
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d
                ):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out_1 = self.ec0(x)
        out_2 = self.ec2(self.ec1(out_1))
        out_4 = self.ec3_2(self.ec3_1(out_2))
        out = self.ec5(self.ec4(self.ec4_1(out_4)))
        return out, out_4, out_2, out_1


class VI_2D_Decoder_3(nn.Module):

    def __init__(self, opt):
        super(VI_2D_Decoder_3, self).__init__()
        self.opt = opt
        dv = 2 if self.opt.double_size else 1
        self.dc0 = GatedConvolution(128, 128, kernel_size=(1, 3, 3), stride
            =(1, 1, 1), padding=(0, 1, 1), bias=False)
        self.dc1 = GatedConvolution(128, 128, kernel_size=(1, 3, 3), stride
            =(1, 1, 1), padding=(0, 1, 1), bias=False)
        self.dc1_1 = GatedUpConvolution((1, opt.crop_size // 4 // dv, opt.
            crop_size // 4 // dv), 128, 96, kernel_size=(1, 3, 3), stride=(
            1, 1, 1), padding=(0, 1, 1), bias=False)
        self.dc2_1 = GatedConvolution(96 + 96, 96, kernel_size=(1, 3, 3),
            stride=(1, 1, 1), padding=(0, 1, 1), bias=False)
        self.dc2_bt1 = GatedConvolution(96, 96, kernel_size=(1, 3, 3),
            stride=(1, 1, 1), dilation=(1, 2, 2), padding=(0, 2, 2), bias=False
            )
        self.dc2_bt2 = GatedConvolution(96, 96, kernel_size=(1, 3, 3),
            stride=(1, 1, 1), dilation=(1, 4, 4), padding=(0, 4, 4), bias=False
            )
        self.dc2_bt3 = GatedConvolution(96, 96, kernel_size=(1, 3, 3),
            stride=(1, 1, 1), dilation=(1, 8, 8), padding=(0, 8, 8), bias=False
            )
        self.dc2_2 = GatedUpConvolution((1, opt.crop_size // 2 // dv, opt.
            crop_size // 2 // dv), 96, 64, kernel_size=(1, 3, 3), stride=(1,
            1, 1), padding=(0, 1, 1), bias=False)
        self.dc3_1 = GatedConvolution(64 + 64, 64, kernel_size=(1, 3, 3),
            stride=(1, 1, 1), padding=(0, 1, 1), bias=False)
        self.dc3_2 = GatedConvolution(64, 64, kernel_size=(1, 3, 3), stride
            =(1, 1, 1), padding=(0, 1, 1), bias=False)
        self.dc4 = GatedUpConvolution((1, opt.crop_size // dv, opt.
            crop_size // dv), 64, 32, kernel_size=(1, 3, 3), stride=(1, 1, 
            1), padding=(0, 1, 1), bias=False)
        if self.opt.double_size:
            self.upsample = nn.Upsample(size=(1, opt.crop_size, opt.
                crop_size), mode='trilinear')
        self.dc5 = GatedConvolution(32, 16, kernel_size=(1, 3, 3), stride=(
            1, 1, 1), padding=(0, 1, 1), bias=False)
        self.dc6 = nn.Conv3d(16, 3, kernel_size=(1, 3, 3), stride=(1, 1, 1),
            padding=(0, 1, 1), bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d
                ):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, x2_64_warp=None, x2_128_warp=None):
        x1_64 = self.dc1_1(self.dc1(self.dc0(x)))
        if x2_64_warp is not None and x2_128_warp is not None:
            x1_64 = self.dc2_bt3(self.dc2_bt2(self.dc2_bt1(self.dc2_1(torch
                .cat([x1_64, x2_64_warp], 1)))))
            x1_128 = self.dc2_2(x1_64)
            if self.opt.double_size:
                d6 = self.dc6(self.dc5(self.upsample(self.dc4(self.dc3_2(
                    self.dc3_1(torch.cat([x1_128, x2_128_warp], 1)))))))
            else:
                d6 = self.dc6(self.dc5(self.dc4(self.dc3_2(self.dc3_1(torch
                    .cat([x1_128, x2_128_warp], 1))))))
        return d6, None


class VI_2D_BottleNeck(nn.Module):

    def __init__(self, opt, in_ch):
        super(VI_2D_BottleNeck, self).__init__()
        self.opt = opt
        self.bt0 = GatedConvolution(in_ch, 128, kernel_size=(3, 3), stride=
            (1, 1), dilation=(1, 1), padding=(1, 1), bias=False, type='2d')
        self.bt1 = GatedConvolution(128, 128, kernel_size=(3, 3), stride=(1,
            1), dilation=(2, 2), padding=(2, 2), bias=False, type='2d')
        self.bt2 = GatedConvolution(128, 128, kernel_size=(3, 3), stride=(1,
            1), dilation=(4, 4), padding=(4, 4), bias=False, type='2d')
        self.bt3 = GatedConvolution(128, 128, kernel_size=(3, 3), stride=(1,
            1), dilation=(8, 8), padding=(8, 8), bias=False, type='2d')
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d
                ):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        res = self.bt3(self.bt2(self.bt1(self.bt0(x))))
        return res


class VI_Aggregator(nn.Module):

    def __init__(self, opt, in_ch, T):
        super(VI_Aggregator, self).__init__()
        self.opt = opt
        self.stAgg = GatedConvolution(in_ch, in_ch, kernel_size=(T, 3, 3),
            stride=(1, 1, 1), padding=(0, 1, 1), bias=False, type='3d')
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d
                ):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        return self.stAgg(x)


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_mcahny_Deep_Video_Inpainting(_paritybench_base):
    pass
    def test_000(self):
        self._check(ContextNetwork(*[], **{'args': _mock_config(batch_norm=4), 'ch_in': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(GatedConvolution(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'stride': 1}), [torch.rand([4, 4, 64, 64, 64])], {})

    @_fails_compile()
    def test_002(self):
        self._check(GatedUpConvolution(*[], **{'size': 4, 'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4, 'bias': 4}), [torch.rand([4, 4, 4, 4, 4])], {})

    def test_003(self):
        self._check(LongFlowEstimatorCorr(*[], **{'args': _mock_config(batch_norm=4), 'ch_in': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_004(self):
        self._check(MaskEstimator_(*[], **{'args': _mock_config(), 'ch_in': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_005(self):
        self._check(VI_2D_BottleNeck(*[], **{'opt': 4, 'in_ch': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_006(self):
        self._check(VI_Aggregator(*[], **{'opt': 4, 'in_ch': 4, 'T': 4}), [torch.rand([4, 4, 64, 64, 64])], {})

    def test_007(self):
        self._check(WarpingLayer(*[], **{}), [torch.rand([4, 2, 4, 4]), torch.rand([4, 2, 4, 4])], {})

