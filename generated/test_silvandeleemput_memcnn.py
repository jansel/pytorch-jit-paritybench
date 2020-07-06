import sys
_module = sys.modules[__name__]
del sys
conf = _module
memcnn = _module
config = _module
tests = _module
test_config = _module
data = _module
cifar = _module
sampling = _module
test_cifar = _module
test_sampling = _module
minimal = _module
test_examples = _module
experiment = _module
factory = _module
manager = _module
test_factory = _module
test_manager = _module
models = _module
additive = _module
affine = _module
resnet = _module
revop = _module
test_couplings = _module
test_is_invertible_module = _module
test_memory_saving = _module
test_models = _module
test_multi = _module
test_resnet = _module
test_revop = _module
utils = _module
train = _module
trainers = _module
classification = _module
test_classification = _module
test_train = _module
log = _module
loss = _module
stats = _module
test_log = _module
test_loss = _module
test_stats = _module
setup = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


from torch.utils.data import DataLoader


import torchvision.transforms as transforms


import numpy as np


from torch.utils.data.sampler import Sampler


import torch.utils.data as data


import torch.nn as nn


import logging


import torch.nn


import warnings


import copy


from torch import set_grad_enabled


import math


import random


import time


from torchvision.datasets.cifar import CIFAR10


from torch.nn.modules.module import Module


class ExampleOperation(nn.Module):

    def __init__(self, channels):
        super(ExampleOperation, self).__init__()
        self.seq = nn.Sequential(nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(3, 3), padding=1), nn.BatchNorm2d(num_features=channels), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.seq(x)


class AdditiveBlockFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, xin, Fm, Gm, *weights):
        """Forward pass computes:
        {x1, x2} = x
        y1 = x1 + Fm(x2)
        y2 = x2 + Gm(y1)
        output = {y1, y2}

        Parameters
        ----------
        ctx : torch.autograd.Function
            The backward pass context object
        x : TorchTensor
            Input tensor. Must have channels (2nd dimension) that can be partitioned in two equal partitions
        Fm : nn.Module
            Module to use for computation, must retain dimensions such that Fm(X)=Y, X.shape == Y.shape
        Gm : nn.Module
            Module to use for computation, must retain dimensions such that Gm(X)=Y, X.shape == Y.shape
        *weights : TorchTensor
            weights for Fm and Gm in that order {Fm_w1, ... Fm_wn, Gm_w1, ... Gm_wn}

        Note
        ----
        All tensor/autograd variable input arguments and the output are
        TorchTensors for the scope of this function

        """
        assert xin.shape[1] % 2 == 0
        ctx.Fm = Fm
        ctx.Gm = Gm
        with torch.no_grad():
            x = xin.detach()
            x1, x2 = torch.chunk(x, 2, dim=1)
            x1, x2 = x1.contiguous(), x2.contiguous()
            fmr = Fm.forward(x2)
            y1 = x1 + fmr
            x1.set_()
            del x1
            gmr = Gm.forward(y1)
            y2 = x2 + gmr
            x2.set_()
            del x2
            output = torch.cat([y1, y2], dim=1)
        ctx.save_for_backward(xin, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        Fm, Gm = ctx.Fm, ctx.Gm
        xin, output = ctx.saved_tensors
        x = xin.detach()
        x1, x2 = torch.chunk(x, 2, dim=1)
        GWeights = [p for p in Gm.parameters()]
        assert grad_output.shape[1] % 2 == 0
        with set_grad_enabled(True):
            x1.requires_grad_()
            x2.requires_grad_()
            y1 = x1 + Fm.forward(x2)
            y2 = x2 + Gm.forward(y1)
            y = torch.cat([y1, y2], dim=1)
            dd = torch.autograd.grad(y, (x1, x2) + tuple(Gm.parameters()) + tuple(Fm.parameters()), grad_output)
            GWgrads = dd[2:2 + len(GWeights)]
            FWgrads = dd[2 + len(GWeights):]
            grad_input = torch.cat([dd[0], dd[1]], dim=1)
        return (grad_input, None, None) + FWgrads + GWgrads


class AdditiveBlockFunction2(torch.autograd.Function):

    @staticmethod
    def forward(ctx, xin, Fm, Gm, *weights):
        """Forward pass computes:
        {x1, x2} = x
        y1 = x1 + Fm(x2)
        y2 = x2 + Gm(y1)
        output = {y1, y2}

        Parameters
        ----------
        ctx : torch.autograd.Function
            The backward pass context object
        x : TorchTensor
            Input tensor. Must have channels (2nd dimension) that can be partitioned in two equal partitions
        Fm : nn.Module
            Module to use for computation, must retain dimensions such that Fm(X)=Y, X.shape == Y.shape
        Gm : nn.Module
            Module to use for computation, must retain dimensions such that Gm(X)=Y, X.shape == Y.shape
        *weights : TorchTensor
            weights for Fm and Gm in that order {Fm_w1, ... Fm_wn, Gm_w1, ... Gm_wn}

        Note
        ----
        All tensor/autograd variable input arguments and the output are
        TorchTensors for the scope of this fuction

        """
        assert xin.shape[1] % 2 == 0
        ctx.Fm = Fm
        ctx.Gm = Gm
        with torch.no_grad():
            x = xin.detach()
            x1, x2 = torch.chunk(x, 2, dim=1)
            x1, x2 = x1.contiguous(), x2.contiguous()
            fmr = Fm.forward(x2)
            y1 = x1 + fmr
            x1.set_()
            del x1
            gmr = Gm.forward(y1)
            y2 = x2 + gmr
            x2.set_()
            del x2
            output = torch.cat([y1, y2], dim=1).detach_()
        ctx.save_for_backward(x, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        Fm, Gm = ctx.Fm, ctx.Gm
        x, output = ctx.saved_tensors
        with torch.no_grad():
            y1, y2 = torch.chunk(output, 2, dim=1)
            y1, y2 = y1.contiguous(), y2.contiguous()
            assert grad_output.shape[1] % 2 == 0
            y1_grad, y2_grad = torch.chunk(grad_output, 2, dim=1)
            y1_grad, y2_grad = y1_grad.contiguous(), y2_grad.contiguous()
        with set_grad_enabled(True):
            z1_stop = y1.detach()
            z1_stop.requires_grad = True
            G_z1 = Gm.forward(z1_stop)
            x2 = y2 - G_z1
            x2_stop = x2.detach()
            x2_stop.requires_grad = True
            F_x2 = Fm.forward(x2_stop)
            x1 = y1 - F_x2
            x1_stop = x1.detach()
            x1_stop.requires_grad = True
            y1 = x1_stop + F_x2
            y2 = x2_stop + G_z1
            dd = torch.autograd.grad(y2, (z1_stop,) + tuple(Gm.parameters()), y2_grad, retain_graph=False)
            z1_grad = dd[0] + y1_grad
            GWgrads = dd[1:]
            dd = torch.autograd.grad(y1, (x1_stop, x2_stop) + tuple(Fm.parameters()), z1_grad, retain_graph=False)
            FWgrads = dd[2:]
            x2_grad = dd[1] + y2_grad
            x1_grad = dd[0]
            grad_input = torch.cat([x1_grad, x2_grad], dim=1)
        return (grad_input, None, None) + FWgrads + GWgrads


class AdditiveBlockInverseFunction(torch.autograd.Function):

    @staticmethod
    def forward(cty, y, Fm, Gm, *weights):
        """Forward pass computes:
        {y1, y2} = y
        x2 = y2 - Gm(y1)
        x1 = y1 - Fm(x2)
        output = {x1, x2}

        Parameters
        ----------
        cty : torch.autograd.Function
            The backward pass context object
        y : TorchTensor
            Input tensor. Must have channels (2nd dimension) that can be partitioned in two equal partitions
        Fm : nn.Module
            Module to use for computation, must retain dimensions such that Fm(X)=Y, X.shape == Y.shape
        Gm : nn.Module
            Module to use for computation, must retain dimensions such that Gm(X)=Y, X.shape == Y.shape
        *weights : TorchTensor
            weights for Fm and Gm in that order {Fm_w1, ... Fm_wn, Gm_w1, ... Gm_wn}

        Note
        ----
        All tensor/autograd variable input arguments and the output are
        TorchTensors for the scope of this fuction

        """
        assert y.shape[1] % 2 == 0
        cty.Fm = Fm
        cty.Gm = Gm
        with torch.no_grad():
            y1, y2 = torch.chunk(y, 2, dim=1)
            y1, y2 = y1.contiguous(), y2.contiguous()
            gmr = Gm.forward(y1)
            x2 = y2 - gmr
            y2.set_()
            del y2
            fmr = Fm.forward(x2)
            x1 = y1 - fmr
            y1.set_()
            del y1
            output = torch.cat([x1, x2], dim=1)
            x1.set_()
            x2.set_()
            del x1, x2
        cty.save_for_backward(y.data, output)
        return output

    @staticmethod
    def backward(cty, grad_output):
        Fm, Gm = cty.Fm, cty.Gm
        yin, output = cty.saved_tensors
        y = yin.detach()
        y1, y2 = torch.chunk(y, 2, dim=1)
        FWeights = [p for p in Fm.parameters()]
        assert grad_output.shape[1] % 2 == 0
        with set_grad_enabled(True):
            y2.requires_grad = True
            y1.requires_grad = True
            x2 = y2 - Gm.forward(y1)
            x1 = y1 - Fm.forward(x2)
            x = torch.cat([x1, x2], dim=1)
            dd = torch.autograd.grad(x, (y2, y1) + tuple(Fm.parameters()) + tuple(Gm.parameters()), grad_output)
            FWgrads = dd[2:2 + len(FWeights)]
            GWgrads = dd[2 + len(FWeights):]
            grad_input = torch.cat([dd[0], dd[1]], dim=1)
        return (grad_input, None, None) + FWgrads + GWgrads


class AdditiveBlockInverseFunction2(torch.autograd.Function):

    @staticmethod
    def forward(cty, y, Fm, Gm, *weights):
        """Forward pass computes:
        {y1, y2} = y
        x2 = y2 - Gm(y1)
        x1 = y1 - Fm(x2)
        output = {x1, x2}

        Parameters
        ----------
        cty : torch.autograd.Function
            The backward pass context object
        y : TorchTensor
            Input tensor. Must have channels (2nd dimension) that can be partitioned in two equal partitions
        Fm : nn.Module
            Module to use for computation, must retain dimensions such that Fm(X)=Y, X.shape == Y.shape
        Gm : nn.Module
            Module to use for computation, must retain dimensions such that Gm(X)=Y, X.shape == Y.shape
        *weights : TorchTensor
            weights for Fm and Gm in that order {Fm_w1, ... Fm_wn, Gm_w1, ... Gm_wn}

        Note
        ----
        All tensor/autograd variable input arguments and the output are
        TorchTensors for the scope of this fuction

        """
        assert y.shape[1] % 2 == 0
        cty.Fm = Fm
        cty.Gm = Gm
        with torch.no_grad():
            y1, y2 = torch.chunk(y, 2, dim=1)
            y1, y2 = y1.contiguous(), y2.contiguous()
            gmr = Gm.forward(y1)
            x2 = y2 - gmr
            y2.set_()
            del y2
            fmr = Fm.forward(x2)
            x1 = y1 - fmr
            y1.set_()
            del y1
            output = torch.cat([x1, x2], dim=1).detach_()
        cty.save_for_backward(y, output)
        return output

    @staticmethod
    def backward(cty, grad_output):
        Fm, Gm = cty.Fm, cty.Gm
        y, output = cty.saved_tensors
        with torch.no_grad():
            x1, x2 = torch.chunk(output, 2, dim=1)
            x1, x2 = x1.contiguous(), x2.contiguous()
            assert grad_output.shape[1] % 2 == 0
            x1_grad, x2_grad = torch.chunk(grad_output, 2, dim=1)
            x1_grad, x2_grad = x1_grad.contiguous(), x2_grad.contiguous()
        with set_grad_enabled(True):
            z1_stop = x2.detach()
            z1_stop.requires_grad = True
            F_z1 = Fm.forward(z1_stop)
            y1 = x1 + F_z1
            y1_stop = y1.detach()
            y1_stop.requires_grad = True
            G_y1 = Gm.forward(y1_stop)
            y2 = x2 + G_y1
            y2_stop = y2.detach()
            y2_stop.requires_grad = True
            z1 = y2_stop - G_y1
            x1 = y1_stop - F_z1
            x2 = z1
            dd = torch.autograd.grad(x1, (z1_stop,) + tuple(Fm.parameters()), x1_grad)
            z1_grad = dd[0] + x2_grad
            FWgrads = dd[1:]
            dd = torch.autograd.grad(x2, (y2_stop, y1_stop) + tuple(Gm.parameters()), z1_grad, retain_graph=False)
            GWgrads = dd[2:]
            y1_grad = dd[1] + x1_grad
            y2_grad = dd[0]
            grad_input = torch.cat([y1_grad, y2_grad], dim=1)
        return (grad_input, None, None) + FWgrads + GWgrads


class AdditiveCoupling(nn.Module):

    def __init__(self, Fm, Gm=None, implementation_fwd=-1, implementation_bwd=-1):
        """
        This computes the output :math:`y` on forward given input :math:`x` and arbitrary modules :math:`Fm` and :math:`Gm` according to:

        :math:`(x1, x2) = x`

        :math:`y1 = x1 + Fm(x2)`

        :math:`y2 = x2 + Gm(y1)`

        :math:`y = (y1, y2)`

        Parameters
        ----------
            Fm : :obj:`torch.nn.Module`
                A torch.nn.Module encapsulating an arbitrary function

            Gm : :obj:`torch.nn.Module`
                A torch.nn.Module encapsulating an arbitrary function
                (If not specified a deepcopy of Fm is used as a Module)

            implementation_fwd : :obj:`int`
                Switch between different Additive Operation implementations for forward pass. Default = -1

            implementation_bwd : :obj:`int`
                Switch between different Additive Operation implementations for inverse pass. Default = -1

        """
        super(AdditiveCoupling, self).__init__()
        if Gm is None:
            Gm = copy.deepcopy(Fm)
        self.Gm = Gm
        self.Fm = Fm
        self.implementation_fwd = implementation_fwd
        self.implementation_bwd = implementation_bwd
        if implementation_bwd != -1 or implementation_fwd != -1:
            warnings.warn('Other implementations than the default (-1) are now deprecated.', DeprecationWarning)

    def forward(self, x):
        args = [x, self.Fm, self.Gm] + [w for w in self.Fm.parameters()] + [w for w in self.Gm.parameters()]
        if self.implementation_fwd == 0:
            out = AdditiveBlockFunction.apply(*args)
        elif self.implementation_fwd == 1:
            out = AdditiveBlockFunction2.apply(*args)
        elif self.implementation_fwd == -1:
            x1, x2 = torch.chunk(x, 2, dim=1)
            x1, x2 = x1.contiguous(), x2.contiguous()
            fmd = self.Fm.forward(x2)
            y1 = x1 + fmd
            gmd = self.Gm.forward(y1)
            y2 = x2 + gmd
            out = torch.cat([y1, y2], dim=1)
        else:
            raise NotImplementedError('Selected implementation ({}) not implemented...'.format(self.implementation_fwd))
        return out

    def inverse(self, y):
        args = [y, self.Fm, self.Gm] + [w for w in self.Fm.parameters()] + [w for w in self.Gm.parameters()]
        if self.implementation_bwd == 0:
            x = AdditiveBlockInverseFunction.apply(*args)
        elif self.implementation_bwd == 1:
            x = AdditiveBlockInverseFunction2.apply(*args)
        elif self.implementation_bwd == -1:
            y1, y2 = torch.chunk(y, 2, dim=1)
            y1, y2 = y1.contiguous(), y2.contiguous()
            gmd = self.Gm.forward(y1)
            x2 = y2 - gmd
            fmd = self.Fm.forward(x2)
            x1 = y1 - fmd
            x = torch.cat([x1, x2], dim=1)
        else:
            raise NotImplementedError('Inverse for selected implementation ({}) not implemented...'.format(self.implementation_bwd))
        return x


class AdditiveBlock(AdditiveCoupling):

    def __init__(self, Fm, Gm=None, implementation_fwd=1, implementation_bwd=1):
        warnings.warn('This class has been deprecated. Use the AdditiveCoupling class instead.', DeprecationWarning)
        super(AdditiveBlock, self).__init__(Fm=Fm, Gm=Gm, implementation_fwd=implementation_fwd, implementation_bwd=implementation_bwd)


class AffineAdapterNaive(nn.Module):
    """ Naive Affine adapter

        Outputs exp(f(x)), f(x) given f(.) and x
    """

    def __init__(self, module):
        super(AffineAdapterNaive, self).__init__()
        self.f = module

    def forward(self, x):
        t = self.f(x)
        s = torch.exp(t)
        return s, t


class AffineAdapterSigmoid(nn.Module):
    """ Sigmoid based affine adapter

        Partitions the output h of f(x) = h into s and t by extracting every odd and even channel
        Outputs sigmoid(s), t
    """

    def __init__(self, module):
        super(AffineAdapterSigmoid, self).__init__()
        self.f = module

    def forward(self, x):
        h = self.f(x)
        assert h.shape[1] % 2 == 0
        scale = torch.sigmoid(h[:, 1::2, :] + 2.0)
        shift = h[:, 0::2, :]
        return scale, shift


class AffineBlockFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, xin, Fm, Gm, *weights):
        """Forward pass for the affine block computes:
        {x1, x2} = x
        {log_s1, t1} = Fm(x2)
        s1 = exp(log_s1)
        y1 = s1 * x1 + t1
        {log_s2, t2} = Gm(y1)
        s2 = exp(log_s2)
        y2 = s2 * x2 + t2
        output = {y1, y2}

        Parameters
        ----------
        ctx : torch.autograd.function.RevNetFunctionBackward
            The backward pass context object
        x : TorchTensor
            Input tensor. Must have channels (2nd dimension) that can be partitioned in two equal partitions
        Fm : nn.Module
            Module to use for computation, must retain dimensions such that Fm(X)=Y, X.shape == Y.shape
        Gm : nn.Module
            Module to use for computation, must retain dimensions such that Gm(X)=Y, X.shape == Y.shape
        *weights : TorchTensor
            weights for Fm and Gm in that order {Fm_w1, ... Fm_wn, Gm_w1, ... Gm_wn}

        Note
        ----
        All tensor/autograd variable input arguments and the output are
        TorchTensors for the scope of this function

        """
        assert xin.shape[1] % 2 == 0
        ctx.Fm = Fm
        ctx.Gm = Gm
        with torch.no_grad():
            x = xin.detach()
            x1, x2 = torch.chunk(x, 2, dim=1)
            x1, x2 = x1.contiguous(), x2.contiguous()
            x2var = x2
            fmr1, fmr2 = Fm.forward(x2var)
            y1 = x1 * fmr1 + fmr2
            x1.set_()
            del x1
            y1var = y1
            gmr1, gmr2 = Gm.forward(y1var)
            y2 = x2 * gmr1 + gmr2
            x2.set_()
            del x2
            output = torch.cat([y1, y2], dim=1).detach_()
        ctx.save_for_backward(xin, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        Fm, Gm = ctx.Fm, ctx.Gm
        xin, output = ctx.saved_tensors
        x = xin.detach()
        x1, x2 = torch.chunk(x.detach(), 2, dim=1)
        GWeights = [p for p in Gm.parameters()]
        assert grad_output.shape[1] % 2 == 0
        with set_grad_enabled(True):
            x1.requires_grad = True
            x2.requires_grad = True
            fmr1, fmr2 = Fm.forward(x2)
            y1 = x1 * fmr1 + fmr2
            gmr1, gmr2 = Gm.forward(y1)
            y2 = x2 * gmr1 + gmr2
            y = torch.cat([y1, y2], dim=1)
            dd = torch.autograd.grad(y, (x1, x2) + tuple(Gm.parameters()) + tuple(Fm.parameters()), grad_output)
            GWgrads = dd[2:2 + len(GWeights)]
            FWgrads = dd[2 + len(GWeights):]
            grad_input = torch.cat([dd[0], dd[1]], dim=1)
        return (grad_input, None, None) + FWgrads + GWgrads


class AffineBlockFunction2(torch.autograd.Function):

    @staticmethod
    def forward(ctx, xin, Fm, Gm, *weights):
        """Forward pass for the affine block computes:
        {x1, x2} = x
        {log_s1, t1} = Fm(x2)
        s1 = exp(log_s1)
        y1 = s1 * x1 + t1
        {log_s2, t2} = Gm(y1)
        s2 = exp(log_s2)
        y2 = s2 * x2 + t2
        output = {y1, y2}

        Parameters
        ----------
        ctx : torch.autograd.function.RevNetFunctionBackward
            The backward pass context object
        x : TorchTensor
            Input tensor. Must have channels (2nd dimension) that can be partitioned in two equal partitions
        Fm : nn.Module
            Module to use for computation, must retain dimensions such that Fm(X)=Y, X.shape == Y.shape
        Gm : nn.Module
            Module to use for computation, must retain dimensions such that Gm(X)=Y, X.shape == Y.shape
        *weights : TorchTensor
            weights for Fm and Gm in that order {Fm_w1, ... Fm_wn, Gm_w1, ... Gm_wn}

        Note
        ----
        All tensor/autograd variable input arguments and the output are
        TorchTensors for the scope of this fuction

        """
        assert xin.shape[1] % 2 == 0
        ctx.Fm = Fm
        ctx.Gm = Gm
        with torch.no_grad():
            x = xin.detach()
            x1, x2 = torch.chunk(x, 2, dim=1)
            x1, x2 = x1.contiguous(), x2.contiguous()
            x2var = x2
            fmr1, fmr2 = Fm.forward(x2var)
            y1 = x1 * fmr1 + fmr2
            x1.set_()
            del x1
            y1var = y1
            gmr1, gmr2 = Gm.forward(y1var)
            y2 = x2 * gmr1 + gmr2
            x2.set_()
            del x2
            output = torch.cat([y1, y2], dim=1).detach_()
        ctx.save_for_backward(xin, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        Fm, Gm = ctx.Fm, ctx.Gm
        x, output = ctx.saved_tensors
        with set_grad_enabled(False):
            y1, y2 = torch.chunk(output, 2, dim=1)
            y1, y2 = y1.contiguous(), y2.contiguous()
            assert grad_output.shape[1] % 2 == 0
            y1_grad, y2_grad = torch.chunk(grad_output, 2, dim=1)
            y1_grad, y2_grad = y1_grad.contiguous(), y2_grad.contiguous()
        with set_grad_enabled(True):
            z1_stop = y1
            z1_stop.requires_grad = True
            G_z11, G_z12 = Gm.forward(z1_stop)
            x2 = (y2 - G_z12) / G_z11
            x2_stop = x2.detach()
            x2_stop.requires_grad = True
            F_x21, F_x22 = Fm.forward(x2_stop)
            x1 = (y1 - F_x22) / F_x21
            x1_stop = x1.detach()
            x1_stop.requires_grad = True
            z1 = x1_stop * F_x21 + F_x22
            y2_ = x2_stop * G_z11 + G_z12
            y1_ = z1
            dd = torch.autograd.grad(y2_, (z1_stop,) + tuple(Gm.parameters()), y2_grad)
            z1_grad = dd[0] + y1_grad
            GWgrads = dd[1:]
            dd = torch.autograd.grad(y1_, (x1_stop, x2_stop) + tuple(Fm.parameters()), z1_grad, retain_graph=False)
            FWgrads = dd[2:]
            x2_grad = dd[1] + y2_grad
            x1_grad = dd[0]
            grad_input = torch.cat([x1_grad, x2_grad], dim=1)
            y1_.detach_()
            y2_.detach_()
            del y1_, y2_
        return (grad_input, None, None) + FWgrads + GWgrads


class AffineBlockInverseFunction(torch.autograd.Function):

    @staticmethod
    def forward(cty, yin, Fm, Gm, *weights):
        """Forward inverse pass for the affine block computes:
        {y1, y2} = y
        {log_s2, t2} = Gm(y1)
        s2 = exp(log_s2)
        x2 = (y2 - t2) / s2
        {log_s1, t1} = Fm(x2)
        s1 = exp(log_s1)
        x1 = (y1 - t1) / s1
        output = {x1, x2}

        Parameters
        ----------
        cty : torch.autograd.function.RevNetInverseFunctionBackward
            The backward pass context object
        y : TorchTensor
            Input tensor. Must have channels (2nd dimension) that can be partitioned in two equal partitions
        Fm : nn.Module
            Module to use for computation, must retain dimensions such that Fm(X)=Y, X.shape == Y.shape
        Gm : nn.Module
            Module to use for computation, must retain dimensions such that Gm(X)=Y, X.shape == Y.shape
        *weights : TorchTensor
            weights for Fm and Gm in that order {Fm_w1, ... Fm_wn, Gm_w1, ... Gm_wn}

        Note
        ----
        All tensor/autograd variable input arguments and the output are
        TorchTensors for the scope of this fuction

        """
        assert yin.shape[1] % 2 == 0
        cty.Fm = Fm
        cty.Gm = Gm
        with torch.no_grad():
            y = yin.detach()
            y1, y2 = torch.chunk(y, 2, dim=1)
            y1, y2 = y1.contiguous(), y2.contiguous()
            y1var = y1
            gmr1, gmr2 = Gm.forward(y1var)
            x2 = (y2 - gmr2) / gmr1
            y2.set_()
            del y2
            x2var = x2
            fmr1, fmr2 = Fm.forward(x2var)
            x1 = (y1 - fmr2) / fmr1
            y1.set_()
            del y1
            output = torch.cat([x1, x2], dim=1).detach_()
        cty.save_for_backward(yin, output)
        return output

    @staticmethod
    def backward(cty, grad_output):
        Fm, Gm = cty.Fm, cty.Gm
        yin, output = cty.saved_tensors
        y = yin.detach()
        y1, y2 = torch.chunk(y.detach(), 2, dim=1)
        FWeights = [p for p in Gm.parameters()]
        assert grad_output.shape[1] % 2 == 0
        with set_grad_enabled(True):
            y2.requires_grad = True
            y1.requires_grad = True
            gmr1, gmr2 = Gm.forward(y1)
            x2 = (y2 - gmr2) / gmr1
            fmr1, fmr2 = Fm.forward(x2)
            x1 = (y1 - fmr2) / fmr1
            x = torch.cat([x1, x2], dim=1)
            dd = torch.autograd.grad(x, (y2, y1) + tuple(Fm.parameters()) + tuple(Gm.parameters()), grad_output)
            FWgrads = dd[2:2 + len(FWeights)]
            GWgrads = dd[2 + len(FWeights):]
            grad_input = torch.cat([dd[0], dd[1]], dim=1)
        return (grad_input, None, None) + FWgrads + GWgrads


class AffineBlockInverseFunction2(torch.autograd.Function):

    @staticmethod
    def forward(cty, yin, Fm, Gm, *weights):
        """Forward pass for the affine block computes:

        Parameters
        ----------
        cty : torch.autograd.function.RevNetInverseFunctionBackward
            The backward pass context object
        y : TorchTensor
            Input tensor. Must have channels (2nd dimension) that can be partitioned in two equal partitions
        Fm : nn.Module
            Module to use for computation, must retain dimensions such that Fm(X)=Y, X.shape == Y.shape
        Gm : nn.Module
            Module to use for computation, must retain dimensions such that Gm(X)=Y, X.shape == Y.shape
        *weights : TorchTensor
            weights for Fm and Gm in that order {Fm_w1, ... Fm_wn, Gm_w1, ... Gm_wn}

        Note
        ----
        All tensor/autograd variable input arguments and the output are
        TorchTensors for the scope of this fuction

        """
        assert yin.shape[1] % 2 == 0
        cty.Fm = Fm
        cty.Gm = Gm
        with torch.no_grad():
            y = yin.detach()
            y1, y2 = torch.chunk(y, 2, dim=1)
            y1, y2 = y1.contiguous(), y2.contiguous()
            y1var = y1
            gmr1, gmr2 = Gm.forward(y1var)
            x2 = (y2 - gmr2) / gmr1
            y2.set_()
            del y2
            x2var = x2
            fmr1, fmr2 = Fm.forward(x2var)
            x1 = (y1 - fmr2) / fmr1
            y1.set_()
            del y1
            output = torch.cat([x1, x2], dim=1).detach_()
        cty.save_for_backward(yin, output)
        return output

    @staticmethod
    def backward(cty, grad_output):
        Fm, Gm = cty.Fm, cty.Gm
        y, output = cty.saved_tensors
        with set_grad_enabled(False):
            x1, x2 = torch.chunk(output, 2, dim=1)
            x1, x2 = x1.contiguous(), x2.contiguous()
            assert grad_output.shape[1] % 2 == 0
            x1_grad, x2_grad = torch.chunk(grad_output, 2, dim=1)
            x1_grad, x2_grad = x1_grad.contiguous(), x2_grad.contiguous()
        with set_grad_enabled(True):
            z1_stop = x2
            z1_stop.requires_grad = True
            F_z11, F_z12 = Fm.forward(z1_stop)
            y1 = x1 * F_z11 + F_z12
            y1_stop = y1.detach()
            y1_stop.requires_grad = True
            G_y11, G_y12 = Gm.forward(y1_stop)
            y2 = x2 * G_y11 + G_y12
            y2_stop = y2.detach()
            y2_stop.requires_grad = True
            z1 = (y2_stop - G_y12) / G_y11
            x1_ = (y1_stop - F_z12) / F_z11
            x2_ = z1
            dd = torch.autograd.grad(x1_, (z1_stop,) + tuple(Fm.parameters()), x1_grad)
            z1_grad = dd[0] + x2_grad
            FWgrads = dd[1:]
            dd = torch.autograd.grad(x2_, (y2_stop, y1_stop) + tuple(Gm.parameters()), z1_grad, retain_graph=False)
            GWgrads = dd[2:]
            y1_grad = dd[1] + x1_grad
            y2_grad = dd[0]
            grad_input = torch.cat([y1_grad, y2_grad], dim=1)
        return (grad_input, None, None) + FWgrads + GWgrads


class AffineCoupling(nn.Module):

    def __init__(self, Fm, Gm=None, adapter=None, implementation_fwd=-1, implementation_bwd=-1):
        """
        This computes the output :math:`y` on forward given input :math:`x` and arbitrary modules :math:`Fm` and :math:`Gm` according to:

        :math:`(x1, x2) = x`

        :math:`(log({s1}), t1) = Fm(x2)`

        :math:`s1 = exp(log({s1}))`

        :math:`y1 = s1 * x1 + t1`

        :math:`(log({s2}), t2) = Gm(y1)`

        :math:`s2 = exp(log({s2}))`

        :math:`y2 = s2 * x2 + t2`

        :math:`y = (y1, y2)`

        Parameters
        ----------
            Fm : :obj:`torch.nn.Module`
                A torch.nn.Module encapsulating an arbitrary function

            Gm : :obj:`torch.nn.Module`
                A torch.nn.Module encapsulating an arbitrary function
                (If not specified a deepcopy of Gm is used as a Module)

            adapter : :obj:`torch.nn.Module` class
                An optional wrapper class A for Fm and Gm which must output
                s, t = A(x) with shape(s) = shape(t) = shape(x)
                s, t are respectively the scale and shift tensors for the affine coupling.

            implementation_fwd : :obj:`int`
                Switch between different Affine Operation implementations for forward pass. Default = -1

            implementation_bwd : :obj:`int`
                Switch between different Affine Operation implementations for inverse pass. Default = -1

        """
        super(AffineCoupling, self).__init__()
        if Gm is None:
            Gm = copy.deepcopy(Fm)
        self.Gm = adapter(Gm) if adapter is not None else Gm
        self.Fm = adapter(Fm) if adapter is not None else Fm
        self.implementation_fwd = implementation_fwd
        self.implementation_bwd = implementation_bwd
        if implementation_bwd != -1 or implementation_fwd != -1:
            warnings.warn('Other implementations than the default (-1) are now deprecated.', DeprecationWarning)

    def forward(self, x):
        args = [x, self.Fm, self.Gm] + [w for w in self.Fm.parameters()] + [w for w in self.Gm.parameters()]
        if self.implementation_fwd == 0:
            out = AffineBlockFunction.apply(*args)
        elif self.implementation_fwd == 1:
            out = AffineBlockFunction2.apply(*args)
        elif self.implementation_fwd == -1:
            x1, x2 = torch.chunk(x, 2, dim=1)
            x1, x2 = x1.contiguous(), x2.contiguous()
            fmr1, fmr2 = self.Fm.forward(x2)
            y1 = x1 * fmr1 + fmr2
            gmr1, gmr2 = self.Gm.forward(y1)
            y2 = x2 * gmr1 + gmr2
            out = torch.cat([y1, y2], dim=1)
        else:
            raise NotImplementedError('Selected implementation ({}) not implemented...'.format(self.implementation_fwd))
        return out

    def inverse(self, y):
        args = [y, self.Fm, self.Gm] + [w for w in self.Fm.parameters()] + [w for w in self.Gm.parameters()]
        if self.implementation_bwd == 0:
            x = AffineBlockInverseFunction.apply(*args)
        elif self.implementation_bwd == 1:
            x = AffineBlockInverseFunction2.apply(*args)
        elif self.implementation_bwd == -1:
            y1, y2 = torch.chunk(y, 2, dim=1)
            y1, y2 = y1.contiguous(), y2.contiguous()
            gmr1, gmr2 = self.Gm.forward(y1)
            x2 = (y2 - gmr2) / gmr1
            fmr1, fmr2 = self.Fm.forward(x2)
            x1 = (y1 - fmr2) / fmr1
            x = torch.cat([x1, x2], dim=1)
        else:
            raise NotImplementedError('Inverse for selected implementation ({}) not implemented...'.format(self.implementation_bwd))
        return x


class AffineBlock(AffineCoupling):

    def __init__(self, Fm, Gm=None, implementation_fwd=1, implementation_bwd=1):
        warnings.warn('This class has been deprecated. Use the AffineCoupling class instead.', DeprecationWarning)
        super(AffineBlock, self).__init__(Fm=Fm, Gm=Gm, implementation_fwd=implementation_fwd, implementation_bwd=implementation_bwd)


def batch_norm(x):
    """match Tensorflow batch norm settings"""
    return nn.BatchNorm2d(x, momentum=0.99, eps=0.001)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlockSub(nn.Module):

    def __init__(self, inplanes, planes, stride=1, noactivation=False):
        super(BasicBlockSub, self).__init__()
        self.noactivation = noactivation
        if not self.noactivation:
            self.bn1 = batch_norm(inplanes)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = batch_norm(planes)
        self.conv2 = conv3x3(planes, planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if not self.noactivation:
            x = self.bn1(x)
            x = self.relu(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, noactivation=False):
        super(BasicBlock, self).__init__()
        self.basicblock_sub = BasicBlockSub(inplanes, planes, stride, noactivation)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.basicblock_sub(x)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return out


class BottleneckSub(nn.Module):

    def __init__(self, inplanes, planes, stride=1, noactivation=False):
        super(BottleneckSub, self).__init__()
        self.noactivation = noactivation
        if not self.noactivation:
            self.bn1 = batch_norm(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = batch_norm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = batch_norm(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if not self.noactivation:
            x = self.bn1(x)
            x = self.relu(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv3(x)
        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, noactivation=False):
        super(Bottleneck, self).__init__()
        self.bottleneck_sub = BottleneckSub(inplanes, planes, stride, noactivation)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.bottleneck_sub(x)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return out


class InvertibleCheckpointFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, fn, fn_inverse, keep_input, num_bwd_passes, num_inputs, *inputs_and_weights):
        ctx.fn = fn
        ctx.fn_inverse = fn_inverse
        ctx.keep_input = keep_input
        ctx.weights = inputs_and_weights[num_inputs:]
        ctx.num_bwd_passes = num_bwd_passes
        ctx.num_inputs = num_inputs
        inputs = inputs_and_weights[:num_inputs]
        ctx.input_requires_grad = [element.requires_grad for element in inputs]
        with torch.no_grad():
            x = [element.detach() for element in inputs]
            outputs = ctx.fn(*x)
        if not isinstance(outputs, tuple):
            outputs = outputs,
        detached_outputs = tuple([element.detach_() for element in outputs])
        if not ctx.keep_input:
            if not pytorch_version_one_and_above:
                for element in inputs:
                    element.data.set_()
            else:
                for element in inputs:
                    element.storage().resize_(0)
        ctx.inputs = [inputs] * num_bwd_passes
        ctx.outputs = [detached_outputs] * num_bwd_passes
        return detached_outputs

    @staticmethod
    def backward(ctx, *grad_outputs):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError('InvertibleCheckpointFunction is not compatible with .grad(), please use .backward() if possible')
        if len(ctx.outputs) == 0:
            raise RuntimeError('Trying to perform backward on the InvertibleCheckpointFunction for more than {} times! Try raising `num_bwd_passes` by one.'.format(ctx.num_bwd_passes))
        inputs = ctx.inputs.pop()
        outputs = ctx.outputs.pop()
        if not ctx.keep_input:
            with torch.no_grad():
                inputs_inverted = ctx.fn_inverse(*outputs)
                if not isinstance(inputs_inverted, tuple):
                    inputs_inverted = inputs_inverted,
                if pytorch_version_one_and_above:
                    for element_original, element_inverted in zip(inputs, inputs_inverted):
                        element_original.storage().resize_(int(np.prod(element_original.size())))
                        element_original.set_(element_inverted)
                else:
                    for element_original, element_inverted in zip(inputs, inputs_inverted):
                        element_original.set_(element_inverted)
        with torch.set_grad_enabled(True):
            detached_inputs = tuple([element.detach().requires_grad_() for element in inputs])
            temp_output = ctx.fn(*detached_inputs)
        if not isinstance(temp_output, tuple):
            temp_output = temp_output,
        gradients = torch.autograd.grad(outputs=temp_output, inputs=detached_inputs + ctx.weights, grad_outputs=grad_outputs)
        for element, element_grad in zip(inputs, gradients[:ctx.num_inputs]):
            element.grad = element_grad
        for element, element_grad in zip(outputs, grad_outputs):
            element.grad = element_grad
        return (None, None, None, None, None) + gradients


class InvertibleModuleWrapper(nn.Module):

    def __init__(self, fn, keep_input=False, keep_input_inverse=False, num_bwd_passes=1, disable=False):
        """
        The InvertibleModuleWrapper which enables memory savings during training by exploiting
        the invertible properties of the wrapped module.

        Parameters
        ----------
            fn : :obj:`torch.nn.Module`
                A torch.nn.Module which has a forward and an inverse function implemented with
                :math:`x == m.inverse(m.forward(x))`

            keep_input : :obj:`bool`, optional
                Set to retain the input information on forward, by default it can be discarded since it will be
                reconstructed upon the backward pass.

            keep_input_inverse : :obj:`bool`, optional
                Set to retain the input information on inverse, by default it can be discarded since it will be
                reconstructed upon the backward pass.

            num_bwd_passes :obj:`int`, optional
                Number of backward passes to retain a link with the output. After the last backward pass the output
                is discarded and memory is freed.
                Warning: if this value is raised higher than the number of required passes memory will not be freed
                correctly anymore and the training process can quickly run out of memory.
                Hence, The typical use case is to keep this at 1, until it raises an error for raising this value.

            disable : :obj:`bool`, optional
                This will disable the detached graph approach with the backward hook.
                Essentially this renders the function as `y = fn(x)` without any of the memory savings.
                Setting this to true will also ignore the keep_input and keep_input_inverse properties.

        Attributes
        ----------
            keep_input : :obj:`bool`, optional
                Set to retain the input information on forward, by default it can be discarded since it will be
                reconstructed upon the backward pass.

            keep_input_inverse : :obj:`bool`, optional
                Set to retain the input information on inverse, by default it can be discarded since it will be
                reconstructed upon the backward pass.

        Raises
        ------
        NotImplementedError
            If an unknown coupling or implementation is given.

        """
        super(InvertibleModuleWrapper, self).__init__()
        self.disable = disable
        self.keep_input = keep_input
        self.keep_input_inverse = keep_input_inverse
        self.num_bwd_passes = num_bwd_passes
        self._fn = fn

    def forward(self, *xin):
        """Forward operation :math:`R(x) = y`

        Parameters
        ----------
            *xin : :obj:`torch.Tensor` tuple
                Input torch tensor(s).

        Returns
        -------
            :obj:`torch.Tensor` tuple
                Output torch tensor(s) *y.

        """
        if not self.disable:
            y = InvertibleCheckpointFunction.apply(self._fn.forward, self._fn.inverse, self.keep_input, self.num_bwd_passes, len(xin), *(xin + tuple([p for p in self._fn.parameters() if p.requires_grad])))
        else:
            y = self._fn(*xin)
        if isinstance(y, tuple) and len(y) == 1:
            return y[0]
        return y

    def inverse(self, *yin):
        """Inverse operation :math:`R^{-1}(y) = x`

        Parameters
        ----------
            *yin : :obj:`torch.Tensor` tuple
                Input torch tensor(s).

        Returns
        -------
            :obj:`torch.Tensor` tuple
                Output torch tensor(s) *x.

        """
        if not self.disable:
            x = InvertibleCheckpointFunction.apply(self._fn.inverse, self._fn.forward, self.keep_input_inverse, self.num_bwd_passes, len(yin), *(yin + tuple([p for p in self._fn.parameters() if p.requires_grad])))
        else:
            x = self._fn.inverse(*yin)
        if isinstance(x, tuple) and len(x) == 1:
            return x[0]
        return x


def create_coupling(Fm, Gm=None, coupling='additive', implementation_fwd=-1, implementation_bwd=-1, adapter=None):
    if coupling == 'additive':
        fn = AdditiveCoupling(Fm, Gm, implementation_fwd=implementation_fwd, implementation_bwd=implementation_bwd)
    elif coupling == 'affine':
        fn = AffineCoupling(Fm, Gm, adapter=adapter, implementation_fwd=implementation_fwd, implementation_bwd=implementation_bwd)
    else:
        raise NotImplementedError('Unknown coupling method: %s' % coupling)
    return fn


class RevBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, noactivation=False):
        super(RevBasicBlock, self).__init__()
        if downsample is None and stride == 1:
            gm = BasicBlockSub(inplanes // 2, planes // 2, stride, noactivation)
            fm = BasicBlockSub(inplanes // 2, planes // 2, stride, noactivation)
            coupling = create_coupling(Fm=fm, Gm=gm, coupling='additive')
            self.revblock = InvertibleModuleWrapper(fn=coupling, keep_input=False)
        else:
            self.basicblock_sub = BasicBlockSub(inplanes, planes, stride, noactivation)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        if self.downsample is not None:
            out = self.basicblock_sub(x)
            residual = self.downsample(x)
            out += residual
        else:
            out = self.revblock(x)
        return out


class RevBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, noactivation=False):
        super(RevBottleneck, self).__init__()
        if downsample is None and stride == 1:
            gm = BottleneckSub(inplanes // 2, planes // 2, stride, noactivation)
            fm = BottleneckSub(inplanes // 2, planes // 2, stride, noactivation)
            coupling = create_coupling(Fm=fm, Gm=gm, coupling='additive')
            self.revblock = InvertibleModuleWrapper(fn=coupling, keep_input=False)
        else:
            self.bottleneck_sub = BottleneckSub(inplanes, planes, stride, noactivation)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        if self.downsample is not None:
            out = self.bottleneck_sub(x)
            residual = self.downsample(x)
            out += residual
        else:
            out = self.revblock(x)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, channels_per_layer=None, strides=None, init_max_pool=False, init_kernel_size=7, batch_norm_fix=True, implementation=0):
        if channels_per_layer is None:
            channels_per_layer = [(2 ** (i + 6)) for i in range(len(layers))]
            channels_per_layer = [channels_per_layer[0]] + channels_per_layer
        if strides is None:
            strides = [2] * len(channels_per_layer)
        self.batch_norm_fix = batch_norm_fix
        self.channels_per_layer = channels_per_layer
        self.strides = strides
        self.init_max_pool = init_max_pool
        self.implementation = implementation
        assert len(self.channels_per_layer) == len(layers) + 1
        self.inplanes = channels_per_layer[0]
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=init_kernel_size, stride=strides[0], padding=(init_kernel_size - 1) // 2, bias=False)
        self.bn1 = batch_norm(self.inplanes)
        self.relu = nn.ReLU(inplace=False)
        if self.init_max_pool:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, channels_per_layer[1], layers[0], stride=strides[1], noactivation=True)
        self.layer2 = self._make_layer(block, channels_per_layer[2], layers[1], stride=strides[2])
        self.layer3 = self._make_layer(block, channels_per_layer[3], layers[2], stride=strides[3])
        self.has_4_layers = len(layers) >= 4
        if self.has_4_layers:
            self.layer4 = self._make_layer(block, channels_per_layer[4], layers[3], stride=strides[4])
        self.bn_final = batch_norm(self.inplanes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels_per_layer[-1] * block.expansion, num_classes)
        self.configure()
        self.init_weights()

    def init_weights(self):
        """Initialization using He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.reset_parameters()

    def configure(self):
        """Initialization specific configuration settings"""
        for m in self.modules():
            if isinstance(m, InvertibleModuleWrapper):
                m.implementation = self.implementation
            elif isinstance(m, nn.BatchNorm2d):
                if self.batch_norm_fix:
                    m.momentum = 0.99
                    m.eps = 0.001
                else:
                    m.momentum = 0.1
                    m.eps = 1e-05

    def _make_layer(self, block, planes, blocks, stride=1, noactivation=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), batch_norm(planes * block.expansion))
        layers = [block(self.inplanes, planes, stride, downsample, noactivation)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.init_max_pool:
            x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.has_4_layers:
            x = self.layer4(x)
        x = self.bn_final(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ReversibleBlock(InvertibleModuleWrapper):

    def __init__(self, Fm, Gm=None, coupling='additive', keep_input=False, keep_input_inverse=False, implementation_fwd=-1, implementation_bwd=-1, adapter=None):
        """The ReversibleBlock

        Warning
        -------
        This class has been deprecated. Use the more flexible InvertibleModuleWrapper class.

        Note
        ----
        The `implementation_fwd` and `implementation_bwd` parameters can be set to one of the following implementations:

        * -1 Naive implementation without reconstruction on the backward pass.
        * 0  Memory efficient implementation, compute gradients directly.
        * 1  Memory efficient implementation, similar to approach in Gomez et al. 2017.


        Parameters
        ----------
            Fm : :obj:`torch.nn.Module`
                A torch.nn.Module encapsulating an arbitrary function

            Gm : :obj:`torch.nn.Module`, optional
                A torch.nn.Module encapsulating an arbitrary function
                (If not specified a deepcopy of Fm is used as a Module)

            coupling : :obj:`str`, optional
                Type of coupling ['additive', 'affine']. Default = 'additive'

            keep_input : :obj:`bool`, optional
                Set to retain the input information on forward, by default it can be discarded since it will be
                reconstructed upon the backward pass.

            keep_input_inverse : :obj:`bool`, optional
                Set to retain the input information on inverse, by default it can be discarded since it will be
                reconstructed upon the backward pass.

            implementation_fwd : :obj:`int`, optional
                Switch between different Operation implementations for forward training (Default = 1).
                If using the naive implementation (-1) then `keep_input` should be True.

            implementation_bwd : :obj:`int`, optional
                Switch between different Operation implementations for backward training (Default = 1).
                If using the naive implementation (-1) then `keep_input_inverse` should be True.

            adapter : :obj:`class`, optional
                Only relevant when using the 'affine' coupling.
                Should be a class of type :obj:`torch.nn.Module` that serves as an
                optional wrapper class A for Fm and Gm which must output
                s, t = A(x) with shape(s) = shape(t) = shape(x).
                s, t are respectively the scale and shift tensors for the affine coupling.

        Attributes
        ----------
            keep_input : :obj:`bool`, optional
                Set to retain the input information on forward, by default it can be discarded since it will be
                reconstructed upon the backward pass.

            keep_input_inverse : :obj:`bool`, optional
                Set to retain the input information on inverse, by default it can be discarded since it will be
                reconstructed upon the backward pass.

        Raises
        ------
        NotImplementedError
            If an unknown coupling or implementation is given.

        """
        warnings.warn('This class has been deprecated. Use the more flexible InvertibleModuleWrapper class', DeprecationWarning)
        fn = create_coupling(Fm=Fm, Gm=Gm, coupling=coupling, implementation_fwd=implementation_fwd, implementation_bwd=implementation_bwd, adapter=adapter)
        super(ReversibleBlock, self).__init__(fn, keep_input=keep_input, keep_input_inverse=keep_input_inverse)


class MultiplicationInverse(torch.nn.Module):

    def __init__(self, factor=2):
        super(MultiplicationInverse, self).__init__()
        self.factor = torch.nn.Parameter(torch.ones(1) * factor)

    def forward(self, x):
        return x * self.factor

    def inverse(self, y):
        return y / self.factor


class IdentityInverse(torch.nn.Module):

    def __init__(self, multiply_forward=False, multiply_inverse=False):
        super(IdentityInverse, self).__init__()
        self.factor = torch.nn.Parameter(torch.ones(1))
        self.multiply_forward = multiply_forward
        self.multiply_inverse = multiply_inverse

    def forward(self, x):
        if self.multiply_forward:
            return x * self.factor
        else:
            return x

    def inverse(self, y):
        if self.multiply_inverse:
            return y * self.factor
        else:
            return y


class MultiSharedOutputs(torch.nn.Module):

    def forward(self, x):
        y = x * x
        return y, y

    def inverse(self, y, y2):
        x = torch.max(torch.sqrt(y), torch.sqrt(y2))
        return x


class SubModule(torch.nn.Module):

    def __init__(self, in_filters=5, out_filters=5):
        super(SubModule, self).__init__()
        self.bn = torch.nn.BatchNorm2d(out_filters)
        self.conv = torch.nn.Conv2d(in_filters, out_filters, (3, 3), padding=1)

    def forward(self, x):
        return self.bn(self.conv(x))


class SubModuleStack(torch.nn.Module):

    def __init__(self, Gm, coupling='additive', depth=10, implementation_fwd=-1, implementation_bwd=-1, keep_input=False, adapter=None, num_bwd_passes=1):
        super(SubModuleStack, self).__init__()
        fn = create_coupling(Fm=Gm, Gm=Gm, coupling=coupling, implementation_fwd=implementation_fwd, implementation_bwd=implementation_bwd, adapter=adapter)
        self.stack = torch.nn.ModuleList([InvertibleModuleWrapper(fn=fn, keep_input=keep_input, keep_input_inverse=keep_input, num_bwd_passes=num_bwd_passes) for _ in range(depth)])

    def forward(self, x):
        for rev_module in self.stack:
            x = rev_module.forward(x)
        return x

    def inverse(self, y):
        for rev_module in reversed(self.stack):
            y = rev_module.inverse(y)
        return y


class SplitChannels(torch.nn.Module):

    def __init__(self, split_location):
        self.split_location = split_location
        super(SplitChannels, self).__init__()

    def forward(self, x):
        return x[:, :self.split_location, :].clone(), x[:, self.split_location:, :].clone()

    def inverse(self, x, y):
        return torch.cat([x, y], dim=1)


class ConcatenateChannels(torch.nn.Module):

    def __init__(self, split_location):
        self.split_location = split_location
        super(ConcatenateChannels, self).__init__()

    def forward(self, x, y):
        return torch.cat([x, y], dim=1)

    def inverse(self, x):
        return x[:, :self.split_location, :].clone(), x[:, self.split_location:, :].clone()


class SimpleTestingModel(torch.nn.Module):

    def __init__(self, klasses):
        super(SimpleTestingModel, self).__init__()
        self.conv = torch.nn.Conv2d(3, klasses, 1)
        self.avgpool = torch.nn.AvgPool2d(32)
        self.klasses = klasses

    def forward(self, x):
        return self.avgpool(self.conv(x)).reshape(x.shape[0], self.klasses)


class DummyModel(torch.nn.Module):

    def __init__(self, block):
        super(DummyModel, self).__init__()
        self.block = block
        self.conv = torch.nn.Conv2d(1, 1, 1)

    def forward(self, x):
        return self.conv(x)


def _assert_no_grad(variable):
    msg = "nn criterions don't compute the gradient w.r.t. targets - please mark these variables as not requiring gradients"
    assert not variable.requires_grad, msg


class CrossEntropyLossTF(Module):

    def __init__(self):
        super(CrossEntropyLossTF, self).__init__()

    def forward(self, Ypred, Y, W=None):
        _assert_no_grad(Y)
        lsm = nn.Softmax(dim=1)
        y_onehot = torch.zeros(Ypred.shape[0], Ypred.shape[1], dtype=torch.float32, device=Ypred.device)
        y_onehot.scatter_(1, Y.data.view(-1, 1), 1)
        if W is not None:
            y_onehot = y_onehot * W
        return torch.mean(-y_onehot * torch.log(lsm(Ypred))) * Ypred.shape[1]


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AffineAdapterNaive,
     lambda: ([], {'module': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (AffineAdapterSigmoid,
     lambda: ([], {'module': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BasicBlockSub,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BottleneckSub,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConcatenateChannels,
     lambda: ([], {'split_location': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (CrossEntropyLossTF,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.zeros([4], dtype=torch.int64)], {}),
     False),
    (DummyModel,
     lambda: ([], {'block': 4}),
     lambda: ([torch.rand([4, 1, 64, 64])], {}),
     True),
    (ExampleOperation,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (IdentityInverse,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MultiSharedOutputs,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MultiplicationInverse,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SplitChannels,
     lambda: ([], {'split_location': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SubModule,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 5, 64, 64])], {}),
     True),
]

class Test_silvandeleemput_memcnn(_paritybench_base):
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

