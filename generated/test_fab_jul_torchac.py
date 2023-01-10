import sys
_module = sys.modules[__name__]
del sys
update_version = _module
mnist_autoencoder_example = _module
setup = _module
test = _module
torchac = _module
torchac = _module

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


import itertools


import warnings


import time


import torch


from torch import nn


import torch.nn.functional as F


import numpy as np


import torchvision


import matplotlib.pyplot as plt


import matplotlib


from torch.utils.cpp_extension import load


class STEQuantize(torch.autograd.Function):
    """Straight-Through Estimator for Quantization.

  Forward pass implements quantization by rounding to integers,
  backward pass is set to gradients of the identity function.
  """

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.round()

    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs


class Autoencoder(nn.Module):

    def __init__(self, bottleneck_size, L):
        if L % 2 != 1:
            raise ValueError(f'number of levels L={L}, must be odd number!')
        super(Autoencoder, self).__init__()
        self.L = L
        self.enc = nn.Sequential(nn.Conv2d(1, 32, 5, stride=2, padding=2), nn.InstanceNorm2d(32), nn.ReLU(), nn.Conv2d(32, 32, 5, stride=2, padding=2), nn.InstanceNorm2d(32), nn.ReLU(), nn.Conv2d(32, 32, 5, stride=2, padding=2), nn.InstanceNorm2d(32), nn.ReLU(), nn.Conv2d(32, 32, 5, stride=2, padding=2), nn.InstanceNorm2d(32), nn.ReLU(), nn.Conv2d(32, bottleneck_size, 1, stride=1, padding=0, bias=False))
        self.dec = nn.Sequential(nn.ConvTranspose2d(bottleneck_size, 32, 5, stride=2, padding=2, output_padding=1), nn.InstanceNorm2d(32), nn.ReLU(), nn.ConvTranspose2d(32, 32, 5, stride=2, padding=2, output_padding=1), nn.InstanceNorm2d(32), nn.ReLU(), nn.ConvTranspose2d(32, 32, 5, stride=2, padding=2, output_padding=1), nn.InstanceNorm2d(32), nn.ReLU(), nn.ConvTranspose2d(32, 32, 5, stride=2, padding=2, output_padding=1), nn.InstanceNorm2d(32), nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 1, stride=1, padding=0), nn.ReLU(), nn.Conv2d(32, 1, 1, stride=1))
        self.quantize = STEQuantize.apply
        self.bottleneck_shape = bottleneck_size, 2, 2

    def forward(self, image):
        latent = self.enc(image)
        jiggle = 0.2
        spread = self.L - 1 + jiggle
        latent = torch.sigmoid(latent) * spread - spread / 2
        latent_quantized = self.quantize(latent)
        reconstructions = self.dec(latent_quantized)
        sym = latent_quantized + self.L // 2
        return reconstructions, sym


_WRITE_BITS = False


def estimate_bitrate_from_pmf(pmf, sym):
    L = pmf.shape[-1]
    pmf = pmf.reshape(-1, L)
    sym = sym.reshape(-1, 1)
    assert pmf.shape[0] == sym.shape[0]
    relevant_probabilities = torch.gather(pmf, dim=1, index=sym)
    bitrate = torch.sum(-torch.log2(relevant_probabilities.clamp(min=0.001)))
    return bitrate


def pmf_to_cdf(pmf):
    cdf = pmf.cumsum(dim=-1)
    spatial_dimensions = pmf.shape[:-1] + (1,)
    zeros = torch.zeros(spatial_dimensions, dtype=pmf.dtype, device=pmf.device)
    cdf_with_0 = torch.cat([zeros, cdf], dim=-1)
    cdf_with_0 = cdf_with_0.clamp(max=1.0)
    return cdf_with_0


class ConditionalProbabilityModel(nn.Module):

    def __init__(self, L, bottleneck_shape):
        super(ConditionalProbabilityModel, self).__init__()
        self.L = L
        self.bottleneck_shape = bottleneck_shape
        self.bottleneck_size, _, _ = bottleneck_shape
        num_output_channels = self.bottleneck_size * L
        self.model = nn.Sequential(nn.Conv2d(1, self.bottleneck_size, 3, padding=1), nn.BatchNorm2d(self.bottleneck_size), nn.ReLU(), nn.Conv2d(self.bottleneck_size, self.bottleneck_size, 3, padding=1), nn.BatchNorm2d(self.bottleneck_size), nn.ReLU(), nn.Conv2d(self.bottleneck_size, num_output_channels, 1, padding=0))

    def forward(self, sym, labels):
        batch_size = sym.shape[0]
        _, H, W = self.bottleneck_shape
        bottleneck_shape_with_batch_dim = batch_size, 1, H, W
        static_input = torch.ones(bottleneck_shape_with_batch_dim, dtype=torch.float32, device=sym.device)
        dynamic_input = static_input * labels.reshape(-1, 1, 1, 1)
        dynamic_input = dynamic_input / 9 - 0.5
        output = self.model(dynamic_input)
        _, C, H, W = output.shape
        assert C == self.bottleneck_size * self.L
        output_reshaped = output.reshape(batch_size, self.bottleneck_size, self.L, H, W)
        output_probabilities = F.softmax(output_reshaped, dim=2)
        output_probabilities = output_probabilities.permute(0, 1, 3, 4, 2)
        estimated_bits = estimate_bitrate_from_pmf(output_probabilities, sym=sym)
        output_cdf = pmf_to_cdf(output_probabilities)
        sym = sym
        output_cdf = output_cdf.detach().cpu()
        sym = sym.detach().cpu()
        byte_stream = torchac.encode_float_cdf(output_cdf, sym, check_input_bounds=True)
        real_bits = len(byte_stream) * 8
        if _WRITE_BITS:
            with open('outfile.b', 'wb') as fout:
                fout.write(byte_stream)
            with open('outfile.b', 'rb') as fin:
                byte_stream = fin.read()
        assert torchac.decode_float_cdf(output_cdf, byte_stream).equal(sym)
        return estimated_bits, real_bits

