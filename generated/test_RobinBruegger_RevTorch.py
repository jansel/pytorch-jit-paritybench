import sys
_module = sys.modules[__name__]
del sys
revtorch = _module
revtorch = _module
setup = _module

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


import random


class ReversibleBlock(nn.Module):
    """
    Elementary building block for building (partially) reversible architectures

    Implementation of the Reversible block described in the RevNet paper
    (https://arxiv.org/abs/1707.04585). Must be used inside a :class:`revtorch.ReversibleSequence`
    for autograd support.

    Arguments:
        f_block (nn.Module): arbitrary subnetwork whos output shape is equal to its input shape
        g_block (nn.Module): arbitrary subnetwork whos output shape is equal to its input shape
        split_along_dim (integer): dimension along which the tensor is split into the two parts requried for the reversible block
        fix_random_seed (boolean): Use the same random seed for the forward and backward pass if set to true 
    """

    def __init__(self, f_block, g_block, split_along_dim=1, fix_random_seed=False):
        super(ReversibleBlock, self).__init__()
        self.f_block = f_block
        self.g_block = g_block
        self.split_along_dim = split_along_dim
        self.fix_random_seed = fix_random_seed
        self.random_seeds = {}

    def _init_seed(self, namespace):
        if self.fix_random_seed:
            self.random_seeds[namespace] = random.randint(0, sys.maxsize)
            self._set_seed(namespace)

    def _set_seed(self, namespace):
        if self.fix_random_seed:
            torch.manual_seed(self.random_seeds[namespace])

    def forward(self, x):
        """
        Performs the forward pass of the reversible block. Does not record any gradients.
        :param x: Input tensor. Must be splittable along dimension 1.
        :return: Output tensor of the same shape as the input tensor
        """
        x1, x2 = torch.chunk(x, 2, dim=self.split_along_dim)
        y1, y2 = None, None
        with torch.no_grad():
            self._init_seed('f')
            y1 = x1 + self.f_block(x2)
            self._init_seed('g')
            y2 = x2 + self.g_block(y1)
        return torch.cat([y1, y2], dim=self.split_along_dim)

    def backward_pass(self, y, dy, retain_graph):
        """
        Performs the backward pass of the reversible block.

        Calculates the derivatives of the block's parameters in f_block and g_block, as well as the inputs of the
        forward pass and its gradients.

        :param y: Outputs of the reversible block
        :param dy: Derivatives of the outputs
        :param retain_graph: Whether to retain the graph on intercepted backwards
        :return: A tuple of (block input, block input derivatives). The block inputs are the same shape as the block outptus.
        """
        y1, y2 = torch.chunk(y, 2, dim=self.split_along_dim)
        del y
        assert not y1.requires_grad, 'y1 must already be detached'
        assert not y2.requires_grad, 'y2 must already be detached'
        dy1, dy2 = torch.chunk(dy, 2, dim=self.split_along_dim)
        del dy
        assert not dy1.requires_grad, 'dy1 must not require grad'
        assert not dy2.requires_grad, 'dy2 must not require grad'
        y1.requires_grad = True
        y2.requires_grad = True
        with torch.enable_grad():
            self._set_seed('g')
            gy1 = self.g_block(y1)
            gy1.backward(dy2, retain_graph=retain_graph)
        with torch.no_grad():
            x2 = y2 - gy1
            del y2, gy1
            dx1 = dy1 + y1.grad
            del dy1
            y1.grad = None
        with torch.enable_grad():
            x2.requires_grad = True
            self._set_seed('f')
            fx2 = self.f_block(x2)
            fx2.backward(dx1, retain_graph=retain_graph)
        with torch.no_grad():
            x1 = y1 - fx2
            del y1, fx2
            dx2 = dy2 + x2.grad
            del dy2
            x2.grad = None
            x = torch.cat([x1, x2.detach()], dim=self.split_along_dim)
            dx = torch.cat([dx1, dx2], dim=self.split_along_dim)
        return x, dx


class _ReversibleModuleFunction(torch.autograd.function.Function):
    """
    Integrates the reversible sequence into the autograd framework
    """

    @staticmethod
    def forward(ctx, x, reversible_blocks, eagerly_discard_variables):
        """
        Performs the forward pass of a reversible sequence within the autograd framework
        :param ctx: autograd context
        :param x: input tensor
        :param reversible_blocks: nn.Modulelist of reversible blocks
        :return: output tensor
        """
        assert isinstance(reversible_blocks, nn.ModuleList)
        for block in reversible_blocks:
            assert isinstance(block, ReversibleBlock)
            x = block(x)
        ctx.y = x.detach()
        ctx.reversible_blocks = reversible_blocks
        ctx.eagerly_discard_variables = eagerly_discard_variables
        return x

    @staticmethod
    def backward(ctx, dy):
        """
        Performs the backward pass of a reversible sequence within the autograd framework
        :param ctx: autograd context
        :param dy: derivatives of the outputs
        :return: derivatives of the inputs
        """
        y = ctx.y
        if ctx.eagerly_discard_variables:
            del ctx.y
        for i in range(len(ctx.reversible_blocks) - 1, -1, -1):
            y, dy = ctx.reversible_blocks[i].backward_pass(y, dy, not ctx.eagerly_discard_variables)
        if ctx.eagerly_discard_variables:
            del ctx.reversible_blocks
        return dy, None, None


class ReversibleSequence(nn.Module):
    """
    Basic building element for (partially) reversible networks

    A reversible sequence is a sequence of arbitrarly many reversible blocks. The entire sequence is reversible.
    The activations are only saved at the end of the sequence. Backpropagation leverages the reversible nature of
    the reversible sequece to save memory.

    Arguments:
        reversible_blocks (nn.ModuleList): A ModuleList that exclusivly contains instances of ReversibleBlock
        which are to be used in the reversible sequence.
        eagerly_discard_variables (bool): If set to true backward() discards the variables requried for 
		calculating the gradient and therefore saves memory. Disable if you call backward() multiple times.
    """

    def __init__(self, reversible_blocks, eagerly_discard_variables=True):
        super(ReversibleSequence, self).__init__()
        assert isinstance(reversible_blocks, nn.ModuleList)
        for block in reversible_blocks:
            assert isinstance(block, ReversibleBlock)
        self.reversible_blocks = reversible_blocks
        self.eagerly_discard_variables = eagerly_discard_variables

    def forward(self, x):
        """
        Forward pass of a reversible sequence
        :param x: Input tensor
        :return: Output tensor
        """
        x = _ReversibleModuleFunction.apply(x, self.reversible_blocks, self.eagerly_discard_variables)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ReversibleBlock,
     lambda: ([], {'f_block': _mock_layer(), 'g_block': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_RobinBruegger_RevTorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

