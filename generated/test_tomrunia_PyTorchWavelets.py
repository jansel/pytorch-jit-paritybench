import sys
_module = sys.modules[__name__]
del sys
examples = _module
benchmark = _module
plot = _module
plot_comparison = _module
simple_example = _module
setup = _module
wavelets_pytorch = _module
network = _module
transform = _module
wavelets = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, string, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
import numpy as np
from torch import Tensor
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import numpy as np


import torch


import torch.nn as nn


from abc import ABCMeta


from abc import abstractmethod


import scipy.signal


import scipy.optimize


from torch.autograd import Variable


class TorchFilterBank(nn.Module):

    def __init__(self, filters=None, cuda=True):
        """
        Temporal filter bank in PyTorch storing a collection of nn.Conv1d filters.
        When cuda=True, the convolutions are performed on the GPU. If initialized with
        filters=None, the set_filters() method has to be called before actual running
        the convolutions.

        :param filters: list, collection of variable sized 1D filters (default: [])
        :param cuda: boolean, whether to run on GPU or not (default: True)
        """
        super(TorchFilterBank, self).__init__()
        self._cuda = cuda
        self._filters = [] if not filters else self.set_filters(filters)

    def forward(self, x):
        """
        Takes a batch of signals and convoles each signal with all elements in the filter
        bank. After convoling the entire filter bank, the method returns a tensor of
        shape [N,N_scales,1/2,T] where the 1/2 number of channels depends on whether
        the filter bank is composed of real or complex filters. If the filters are
        complex the 2 channels represent [real, imag] parts.

        :param x: torch.Variable, batch of input signals of shape [N,1,T]
        :return: torch.Variable, batch of outputs of size [N,N_scales,1/2,T]
        """
        if not self._filters:
            raise ValueError('PyTorch filters not initialized. Please call set_filters() first.')
            return None
        results = [None] * len(self._filters)
        for ind, conv in enumerate(self._filters):
            results[ind] = conv(x)
        results = torch.stack(results)
        results = results.permute(1, 0, 2, 3)
        return results

    def set_filters(self, filters, padding_type='SAME'):
        """
        Given a list of temporal 1D filters of variable size, this method creates a
        list of nn.conv1d objects that collectively form the filter bank.

        :param filters: list, collection of filters each a np.ndarray
        :param padding_type: str, should be SAME or VALID
        :return:
        """
        assert isinstance(filters, list)
        assert padding_type in ['SAME', 'VALID']
        self._filters = [None] * len(filters)
        for ind, filt in enumerate(filters):
            assert filt.dtype in (np.float32, np.float64, np.complex64, np.complex128)
            if np.iscomplex(filt).any():
                chn_out = 2
                filt_weights = np.asarray([np.real(filt), np.imag(filt)], np.float32)
            else:
                chn_out = 1
                filt_weights = filt.astype(np.float32)[(None), :]
            filt_weights = np.expand_dims(filt_weights, 1)
            filt_size = filt_weights.shape[-1]
            padding = self._get_padding(padding_type, filt_size)
            conv = nn.Conv1d(1, chn_out, kernel_size=filt_size, padding=padding, bias=False)
            conv.weight.data = torch.from_numpy(filt_weights)
            conv.weight.requires_grad_(False)
            if self._cuda:
                conv
            self._filters[ind] = conv

    @staticmethod
    def _get_padding(padding_type, kernel_size):
        assert isinstance(kernel_size, int)
        assert padding_type in ['SAME', 'VALID']
        if padding_type == 'SAME':
            return (kernel_size - 1) // 2
        return 0

