import sys
_module = sys.modules[__name__]
del sys
benchmarks = _module
common = _module
scattering1d = _module
scattering2d = _module
scattering3d = _module
conf = _module
classif_keras = _module
plot_classif_torch = _module
plot_filters = _module
plot_real_signal = _module
plot_synthetic = _module
reconstruct_torch = _module
cifar_resnet_torch = _module
cifar_small_sample = _module
cifar_torch = _module
long_mnist_classify_torch = _module
mnist_keras = _module
plot_invert_scattering_torch = _module
plot_scattering_disk = _module
plot_sklearn = _module
regularized_inverse_scattering_MNIST_torch = _module
scattering3d_qm7_torch = _module
kymatio = _module
backend = _module
base_backend = _module
numpy_backend = _module
tensorflow_backend = _module
torch_backend = _module
torch_skcuda_backend = _module
caching = _module
datasets = _module
frontend = _module
base_frontend = _module
entry = _module
keras_frontend = _module
numpy_frontend = _module
sklearn_frontend = _module
tensorflow_frontend = _module
torch_frontend = _module
keras = _module
numpy = _module
torch_backend = _module
torch_skcuda_backend = _module
core = _module
scattering1d = _module
filter_bank = _module
base_frontend = _module
torch_frontend = _module
utils = _module
torch_backend = _module
torch_skcuda_backend = _module
torch_frontend = _module
numpy_backend = _module
torch_backend = _module
torch_skcuda_backend = _module
scattering3d = _module
filter_bank = _module
torch_frontend = _module
sklearn = _module
tensorflow = _module
version = _module
setup = _module
test_torch_backend = _module
test_filters_scattering1d = _module
test_numpy_scattering1d = _module
test_tensorflow_scattering1d = _module
test_torch_scattering1d = _module
test_utils_scattering1d = _module
test_frontend_scattering2d = _module
test_keras_scattering2d = _module
test_numpy_scattering2d = _module
test_sklearn_2d = _module
test_tensorflow_scattering2d = _module
test_torch_scattering2d = _module
test_numpy_scattering3d = _module
test_tensorflow_scattering3d = _module
test_torch_scattering3d = _module
test_utils_scattering3d = _module

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


from torch.nn import Linear


from torch.nn import NLLLoss


from torch.nn import LogSoftmax


from torch.nn import Sequential


from torch.optim import Adam


from scipy.io import wavfile


import numpy as np


from sklearn.metrics import confusion_matrix


from torch.autograd import backward


import torch.nn as nn


import torch.nn.functional as F


import torch.optim


from torchvision import datasets


from torchvision import transforms


from numpy.random import RandomState


import math


from torch import optim


from scipy.misc import face


import torch.optim as optim


from torch.autograd import Variable


from torch.utils.data import DataLoader


import time


from sklearn import linear_model


from sklearn import model_selection


from sklearn import preprocessing


from sklearn import pipeline


from scipy.spatial.distance import pdist


from torch.autograd import Function


import logging


import warnings


from collections import namedtuple


from string import Template


import numbers


from torch.nn import ReflectionPad2d


from scipy.fftpack import fftn


from scipy.fftpack import ifftn


from scipy.special import sph_harm


from scipy.special import factorial


from torch.autograd import gradcheck


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Scattering2dResNet(nn.Module):

    def __init__(self, in_channels, k=2, n=4, num_classes=10, standard=False):
        super(Scattering2dResNet, self).__init__()
        self.inplanes = 16 * k
        self.ichannels = 16 * k
        if standard:
            self.init_conv = nn.Sequential(nn.Conv2d(3, self.ichannels, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(self.ichannels), nn.ReLU(True))
            self.layer1 = self._make_layer(BasicBlock, 16 * k, n)
            self.standard = True
        else:
            self.K = in_channels
            self.init_conv = nn.Sequential(nn.BatchNorm2d(in_channels, eps=1e-05, affine=False), nn.Conv2d(in_channels, self.ichannels, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(self.ichannels), nn.ReLU(True))
            self.standard = False
        self.layer2 = self._make_layer(BasicBlock, 32 * k, n)
        self.layer3 = self._make_layer(BasicBlock, 64 * k, n)
        self.avgpool = nn.AdaptiveAvgPool2d(2)
        self.fc = nn.Linear(64 * k * 4, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        if not self.standard:
            x = x.view(x.size(0), self.K, 8, 8)
        x = self.init_conv(x)
        if self.standard:
            x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Identity(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


class Scattering2dCNN(nn.Module):
    """
        Simple CNN with 3x3 convs based on VGG
    """

    def __init__(self, in_channels, classifier_type='cnn'):
        super(Scattering2dCNN, self).__init__()
        self.in_channels = in_channels
        self.classifier_type = classifier_type
        self.build()

    def build(self):
        cfg = [256, 256, 256, 'M', 512, 512, 512, 1024, 1024]
        layers = []
        self.K = self.in_channels
        self.bn = nn.BatchNorm2d(self.K)
        if self.classifier_type == 'cnn':
            for v in cfg:
                if v == 'M':
                    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                else:
                    conv2d = nn.Conv2d(self.in_channels, v, kernel_size=3, padding=1)
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                    self.in_channels = v
            layers += [nn.AdaptiveAvgPool2d(2)]
            self.features = nn.Sequential(*layers)
            self.classifier = nn.Linear(1024 * 4, 10)
        elif self.classifier_type == 'mlp':
            self.classifier = nn.Sequential(nn.Linear(self.K * 8 * 8, 1024), nn.ReLU(), nn.Linear(1024, 1024), nn.ReLU(), nn.Linear(1024, 10))
            self.features = None
        elif self.classifier_type == 'linear':
            self.classifier = nn.Linear(self.K * 8 * 8, 10)
            self.features = None

    def forward(self, x):
        x = self.bn(x.view(-1, self.K, 8, 8))
        if self.features:
            x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class View(nn.Module):

    def __init__(self, *args):
        super(View, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(-1, *self.shape)


class Generator(nn.Module):

    def __init__(self, num_input_channels, num_hidden_channels, num_output_channels=1, filter_size=3):
        super(Generator, self).__init__()
        self.num_input_channels = num_input_channels
        self.num_hidden_channels = num_hidden_channels
        self.num_output_channels = num_output_channels
        self.filter_size = filter_size
        self.build()

    def build(self):
        padding = (self.filter_size - 1) // 2
        self.main = nn.Sequential(nn.ReflectionPad2d(padding), nn.Conv2d(self.num_input_channels, self.num_hidden_channels, self.filter_size, bias=False), nn.BatchNorm2d(self.num_hidden_channels, eps=0.001, momentum=0.9), nn.ReLU(inplace=True), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), nn.ReflectionPad2d(padding), nn.Conv2d(self.num_hidden_channels, self.num_hidden_channels, self.filter_size, bias=False), nn.BatchNorm2d(self.num_hidden_channels, eps=0.001, momentum=0.9), nn.ReLU(inplace=True), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), nn.ReflectionPad2d(padding), nn.Conv2d(self.num_hidden_channels, self.num_output_channels, self.filter_size, bias=False), nn.BatchNorm2d(self.num_output_channels, eps=0.001, momentum=0.9), nn.Tanh())

    def forward(self, input_tensor):
        return self.main(input_tensor)


def input_checks(x):
    if x is None:
        raise TypeError('The input should be not empty.')
    if not x.is_contiguous():
        raise RuntimeError('The input must be contiguous.')


class ScatteringTorch(nn.Module):

    def __init__(self):
        super(ScatteringTorch, self).__init__()
        self.frontend_name = 'torch'

    def register_filters(self):
        """ This function should be called after filters are generated,
        saving those arrays as module buffers. """
        raise NotImplementedError

    def scattering(self, x):
        """ This function should compute the scattering transform."""
        raise NotImplementedError

    def forward(self, x):
        """This method is an alias for `scattering`."""
        input_checks(x)
        return self.scattering(x)
    _doc_array = 'torch.Tensor'
    _doc_array_n = ''
    _doc_alias_name = 'forward'
    _doc_alias_call = '.forward'
    _doc_frontend_paragraph = """
        This class inherits from `torch.nn.Module`. As a result, it has all
        the same capabilities, including transferring the object to the GPU
        using the `cuda` or `to` methods. This object would then take GPU
        tensors as input and output the scattering coefficients of those
        tensors.
        """
    _doc_sample = 'torch.randn({shape})'
    _doc_has_shape = True
    _doc_has_out_type = True


class ScatteringBase:

    def __init__(self):
        super(ScatteringBase, self).__init__()

    def build(self):
        """ Defines elementary routines.

        This function should always call and create the filters via
        self.create_filters() defined below. For instance, via:
        self.filters = self.create_filters() """
        raise NotImplementedError

    def _instantiate_backend(self, import_string):
        """ This function should instantiate the backend to be used if not already
        specified"""
        if isinstance(self.backend, str):
            if self.backend.startswith(self.frontend_name):
                try:
                    self.backend = importlib.import_module(import_string + self.backend + '_backend', 'backend').backend
                except ImportError:
                    raise ImportError('Backend ' + self.backend + ' not found!')
            else:
                raise ImportError('The backend ' + self.backend + ' can not be called from the frontend ' + self.frontend_name + '.')
        elif not self.backend.name.startswith(self.frontend_name):
            raise ImportError('The backend ' + self.backend.name + ' is not supported by the frontend ' + self.frontend_name + '.')

    def create_filters(self):
        """ This function should run a filterbank function that
        will create the filters as numpy array, and then, it should
        save those arrays. """
        raise NotImplementedError


def compute_border_indices(J, i0, i1):
    """
    Computes border indices at all scales which correspond to the original
    signal boundaries after padding.

    At the finest resolution,
    original_signal = padded_signal[..., i0:i1].
    This function finds the integers i0, i1 for all temporal subsamplings
    by 2**J, being conservative on the indices.

    Parameters
    ----------
    J : int
        maximal subsampling by 2**J
    i0 : int
        start index of the original signal at the finest resolution
    i1 : int
        end index (excluded) of the original signal at the finest resolution

    Returns
    -------
    ind_start, ind_end: dictionaries with keys in [0, ..., J] such that the
        original signal is in padded_signal[ind_start[j]:ind_end[j]]
        after subsampling by 2**j
    """
    ind_start = {(0): i0}
    ind_end = {(0): i1}
    for j in range(1, J + 1):
        ind_start[j] = ind_start[j - 1] // 2 + ind_start[j - 1] % 2
        ind_end[j] = ind_end[j - 1] // 2 + ind_end[j - 1] % 2
    return ind_start, ind_end


def compute_sigma_psi(xi, Q, r=math.sqrt(0.5)):
    """
    Computes the frequential width sigma for a Morlet filter of frequency xi
    belonging to a family with Q wavelets.

    The frequential width is adapted so that the intersection of the
    frequency responses of the next filter occurs at a r-bandwidth specified
    by r, to ensure a correct coverage of the whole frequency axis.

    Parameters
    ----------
    xi : float
        frequency of the filter in [0, 1]
    Q : int
        number of filters per octave, Q is an integer >= 1
    r : float, optional
        Positive parameter defining the bandwidth to use.
        Should be < 1. We recommend keeping the default value.
        The larger r, the larger the filters in frequency domain.

    Returns
    -------
    sigma : float
        frequential width of the Morlet wavelet.

    Refs
    ----
    Convolutional operators in the time-frequency domain, V. Lostanlen,
    PhD Thesis, 2017
    https://tel.archives-ouvertes.fr/tel-01559667
    """
    factor = 1.0 / math.pow(2, 1.0 / Q)
    term1 = (1 - factor) / (1 + factor)
    term2 = 1.0 / math.sqrt(2 * math.log(1.0 / r))
    return xi * term1 * term2


def compute_xi_max(Q):
    """
    Computes the maximal xi to use for the Morlet family, depending on Q.

    Parameters
    ----------
    Q : int
        number of wavelets per octave (integer >= 1)

    Returns
    -------
    xi_max : float
        largest frequency of the wavelet frame.
    """
    xi_max = max(1.0 / (1.0 + math.pow(2.0, 3.0 / Q)), 0.35)
    return xi_max


def get_max_dyadic_subsampling(xi, sigma, alpha=5.0):
    """
    Computes the maximal dyadic subsampling which is possible for a Gabor
    filter of frequency xi and width sigma

    Finds the maximal integer j such that:
    omega_0 < 2^{-(j + 1)}
    where omega_0 is the boundary of the filter, defined as
    omega_0 = xi + alpha * sigma

    This ensures that the filter can be subsampled by a factor 2^j without
    aliasing.

    We use the same formula for Gabor and Morlet filters.

    Parameters
    ----------
    xi : float
        frequency of the filter in [0, 1]
    sigma : float
        frequential width of the filter
    alpha : float, optional
        parameter controlling the error done in the aliasing.
        The larger alpha, the smaller the error. Defaults to 5.

    Returns
    -------
    j : int
        integer such that 2^j is the maximal subsampling accepted by the
        Gabor filter without aliasing.
    """
    upper_bound = min(xi + alpha * sigma, 0.5)
    j = math.floor(-math.log2(upper_bound)) - 1
    j = int(j)
    return j


def move_one_dyadic_step(cv, Q, alpha=5.0):
    """
    Computes the parameters of the next wavelet on the low frequency side,
    based on the parameters of the current wavelet.

    This function is used in the loop defining all the filters, starting
    at the wavelet frequency and then going to the low frequencies by
    dyadic steps. This makes the loop in compute_params_filterbank much
    simpler to read.

    The steps are defined as:
    xi_{n+1} = 2^{-1/Q} xi_n
    sigma_{n+1} = 2^{-1/Q} sigma_n

    Parameters
    ----------
    cv : dictionary
        stands for current_value. Is a dictionary with keys:
        *'key': a tuple (j, n) where n is a counter and j is the maximal
            dyadic subsampling accepted by this wavelet.
        *'xi': central frequency of the wavelet
        *'sigma': width of the wavelet
    Q : int
        number of wavelets per octave. Controls the relationship between
        the frequency and width of the current wavelet and the next wavelet.
    alpha : float, optional
        tolerance parameter for the aliasing. The larger alpha,
        the more conservative the algorithm is. Defaults to 5.

    Returns
    -------
    new_cv : dictionary
        a dictionary with the same keys as the ones listed for cv,
        whose values are updated
    """
    factor = 1.0 / math.pow(2.0, 1.0 / Q)
    n = cv['key']
    new_cv = {'xi': cv['xi'] * factor, 'sigma': cv['sigma'] * factor}
    new_cv['j'] = get_max_dyadic_subsampling(new_cv['xi'], new_cv['sigma'], alpha=alpha)
    new_cv['key'] = n + 1
    return new_cv


def compute_params_filterbank(sigma_low, Q, r_psi=math.sqrt(0.5), alpha=5.0):
    """
    Computes the parameters of a Morlet wavelet filterbank.

    This family is defined by constant ratios between the frequencies and
    width of adjacent filters, up to a minimum frequency where the frequencies
    are translated.
    This ensures that the low-pass filter has the largest temporal support
    among all filters, while preserving the coverage of the whole frequency
    axis.

    The keys of the dictionaries are tuples of integers (j, n) where n is a
    counter (starting at 0 for the highest frequency filter) and j is the
    maximal dyadic subsampling accepted by this filter.

    Parameters
    ----------
    sigma_low : float
        frequential width of the low-pass filter. This acts as a
        lower-bound on the frequential widths of the band-pass filters,
        so as to ensure that the low-pass filter has the largest temporal
        support among all filters.
    Q : int
        number of wavelets per octave.
    r_psi : float, optional
        Should be >0 and <1. Controls the redundancy of the filters
        (the larger r_psi, the larger the overlap between adjacent wavelets).
        Defaults to sqrt(0.5).
    alpha : float, optional
        tolerance factor for the aliasing after subsampling.
        The larger alpha, the more conservative the value of maximal
        subsampling is. Defaults to 5.

    Returns
    -------
    xi : dictionary
        dictionary containing the central frequencies of the wavelets.
    sigma : dictionary
        dictionary containing the frequential widths of the wavelets.

    Refs
    ----
    Convolutional operators in the time-frequency domain, 2.1.3, V. Lostanlen,
    PhD Thesis, 2017
    https://tel.archives-ouvertes.fr/tel-01559667
    """
    xi_max = compute_xi_max(Q)
    sigma_max = compute_sigma_psi(xi_max, Q, r=r_psi)
    xi = []
    sigma = []
    j = []
    if sigma_max <= sigma_low:
        last_xi = sigma_max
    else:
        current = {'key': 0, 'j': 0, 'xi': xi_max, 'sigma': sigma_max}
        while current['sigma'] > sigma_low:
            xi.append(current['xi'])
            sigma.append(current['sigma'])
            j.append(current['j'])
            current = move_one_dyadic_step(current, Q, alpha=alpha)
        last_xi = xi[-1]
    num_intermediate = Q - 1
    for q in range(1, num_intermediate + 1):
        factor = (num_intermediate + 1.0 - q) / (num_intermediate + 1.0)
        new_xi = factor * last_xi
        new_sigma = sigma_low
        xi.append(new_xi)
        sigma.append(new_sigma)
        j.append(get_max_dyadic_subsampling(new_xi, new_sigma, alpha=alpha))
    return xi, sigma, j


def calibrate_scattering_filters(J, Q, r_psi=math.sqrt(0.5), sigma0=0.1, alpha=5.0):
    """
    Calibrates the parameters of the filters used at the 1st and 2nd orders
    of the scattering transform.

    These filterbanks share the same low-pass filterbank, but use a
    different Q: Q_1 = Q and Q_2 = 1.

    The dictionaries for the band-pass filters have keys which are 2-tuples
    of the type (j, n), where n is an integer >=0 counting the filters (for
    identification purposes) and j is an integer >= 0 denoting the maximal
    subsampling 2**j which can be performed on a signal convolved with this
    filter without aliasing.

    Parameters
    ----------
    J : int
        maximal scale of the scattering (controls the number of wavelets)
    Q : int
        number of wavelets per octave for the first order
    r_psi : float, optional
        Should be >0 and <1. Controls the redundancy of the filters
        (the larger r_psi, the larger the overlap between adjacent wavelets).
        Defaults to sqrt(0.5)
    sigma0 : float, optional
        frequential width of the low-pass filter at scale J=0
        (the subsequent widths are defined by sigma_J = sigma0 / 2^J).
        Defaults to 1e-1
    alpha : float, optional
        tolerance factor for the aliasing after subsampling.
        The larger alpha, the more conservative the value of maximal
        subsampling is. Defaults to 5.

    Returns
    -------
    sigma_low : float
        frequential width of the low-pass filter
    xi1 : dictionary
        dictionary containing the center frequencies of the first order
        filters. See above for a decsription of the keys.
    sigma1 : dictionary
        dictionary containing the frequential width of the first order
        filters. See above for a description of the keys.
    xi2 : dictionary
        dictionary containing the center frequencies of the second order
        filters. See above for a decsription of the keys.
    sigma2 : dictionary
        dictionary containing the frequential width of the second order
        filters. See above for a description of the keys.
    """
    if Q < 1:
        raise ValueError('Q should always be >= 1, got {}'.format(Q))
    sigma_low = sigma0 / math.pow(2, J)
    xi1, sigma1, j1 = compute_params_filterbank(sigma_low, Q, r_psi=r_psi, alpha=alpha)
    xi2, sigma2, j2 = compute_params_filterbank(sigma_low, 1, r_psi=r_psi, alpha=alpha)
    return sigma_low, xi1, sigma1, j1, xi2, sigma2, j2


def compute_meta_scattering(J, Q, max_order=2):
    """Get metadata on the transform.

    This information specifies the content of each scattering coefficient,
    which order, which frequencies, which filters were used, and so on.

    Parameters
    ----------
    J : int
        The maximum log-scale of the scattering transform.
        In other words, the maximum scale is given by `2**J`.
    Q : int >= 1
        The number of first-order wavelets per octave.
        Second-order wavelets are fixed to one wavelet per octave.
    max_order : int, optional
        The maximum order of scattering coefficients to compute.
        Must be either equal to `1` or `2`. Defaults to `2`.

    Returns
    -------
    meta : dictionary
        A dictionary with the following keys:

        - `'order`' : tensor
            A Tensor of length `C`, the total number of scattering
            coefficients, specifying the scattering order.
        - `'xi'` : tensor
            A Tensor of size `(C, max_order)`, specifying the center
            frequency of the filter used at each order (padded with NaNs).
        - `'sigma'` : tensor
            A Tensor of size `(C, max_order)`, specifying the frequency
            bandwidth of the filter used at each order (padded with NaNs).
        - `'j'` : tensor
            A Tensor of size `(C, max_order)`, specifying the dyadic scale
            of the filter used at each order (padded with NaNs).
        - `'n'` : tensor
            A Tensor of size `(C, max_order)`, specifying the indices of
            the filters used at each order (padded with NaNs).
        - `'key'` : list
            The tuples indexing the corresponding scattering coefficient
            in the non-vectorized output.
    """
    sigma_low, xi1s, sigma1s, j1s, xi2s, sigma2s, j2s = calibrate_scattering_filters(J, Q)
    meta = {}
    meta['order'] = [[], [], []]
    meta['xi'] = [[], [], []]
    meta['sigma'] = [[], [], []]
    meta['j'] = [[], [], []]
    meta['n'] = [[], [], []]
    meta['key'] = [[], [], []]
    meta['order'][0].append(0)
    meta['xi'][0].append(())
    meta['sigma'][0].append(())
    meta['j'][0].append(())
    meta['n'][0].append(())
    meta['key'][0].append(())
    for n1, (xi1, sigma1, j1) in enumerate(zip(xi1s, sigma1s, j1s)):
        meta['order'][1].append(1)
        meta['xi'][1].append((xi1,))
        meta['sigma'][1].append((sigma1,))
        meta['j'][1].append((j1,))
        meta['n'][1].append((n1,))
        meta['key'][1].append((n1,))
        if max_order < 2:
            continue
        for n2, (xi2, sigma2, j2) in enumerate(zip(xi2s, sigma2s, j2s)):
            if j2 > j1:
                meta['order'][2].append(2)
                meta['xi'][2].append((xi1, xi2))
                meta['sigma'][2].append((sigma1, sigma2))
                meta['j'][2].append((j1, j2))
                meta['n'][2].append((n1, n2))
                meta['key'][2].append((n1, n2))
    for field, value in meta.items():
        meta[field] = value[0] + value[1] + value[2]
    pad_fields = ['xi', 'sigma', 'j', 'n']
    pad_len = max_order
    for field in pad_fields:
        meta[field] = [(x + (math.nan,) * (pad_len - len(x))) for x in meta[field]]
    array_fields = ['order', 'xi', 'sigma', 'j', 'n']
    for field in array_fields:
        meta[field] = np.array(meta[field])
    return meta


def compute_temporal_support(h_f, criterion_amplitude=0.001):
    """
    Computes the (half) temporal support of a family of centered,
    symmetric filters h provided in the Fourier domain

    This function computes the support T which is the smallest integer
    such that for all signals x and all filters h,

    \\| x \\conv h - x \\conv h_{[-T, T]} \\|_{\\infty} \\leq \\epsilon
        \\| x \\|_{\\infty}  (1)

    where 0<\\epsilon<1 is an acceptable error, and h_{[-T, T]} denotes the
    filter h whose support is restricted in the interval [-T, T]

    The resulting value T used to pad the signals to avoid boundary effects
    and numerical errors.

    If the support is too small, no such T might exist.
    In this case, T is defined as the half of the support of h, and a
    UserWarning is raised.

    Parameters
    ----------
    h_f : array_like
        a numpy array of size batch x time, where each row contains the
        Fourier transform of a filter which is centered and whose absolute
        value is symmetric
    criterion_amplitude : float, optional
        value \\epsilon controlling the numerical
        error. The larger criterion_amplitude, the smaller the temporal
        support and the larger the numerical error. Defaults to 1e-3

    Returns
    -------
    t_max : int
        temporal support which ensures (1) for all rows of h_f

    """
    h = ifft(h_f, axis=1)
    half_support = h.shape[1] // 2
    l1_residual = np.fliplr(np.cumsum(np.fliplr(np.abs(h)[:, :half_support]), axis=1))
    if np.any(np.max(l1_residual, axis=0) <= criterion_amplitude):
        T = np.min(np.where(np.max(l1_residual, axis=0) <= criterion_amplitude)[0]) + 1
    else:
        T = half_support
        warnings.warn('Signal support is too small to avoid border effects')
    return T


def adaptive_choice_P(sigma, eps=1e-07):
    """
    Adaptive choice of the value of the number of periods in the frequency
    domain used to compute the Fourier transform of a Morlet wavelet.

    This function considers a Morlet wavelet defined as the sum
    of
    * a Gabor term hat psi(omega) = hat g_{sigma}(omega - xi)
    where 0 < xi < 1 is some frequency and g_{sigma} is
    the Gaussian window defined in Fourier by
    hat g_{sigma}(omega) = e^{-omega^2/(2 sigma^2)}
    * a low pass term \\hat \\phi which is proportional to \\hat g_{\\sigma}.

    If \\sigma is too large, then these formula will lead to discontinuities
    in the frequency interval [0, 1] (which is the interval used by numpy.fft).
    We therefore choose a larger integer P >= 1 such that at the boundaries
    of the Fourier transform of both filters on the interval [1-P, P], the
    magnitude of the entries is below the required machine precision.
    Mathematically, this means we would need P to satisfy the relations:

    |\\hat \\psi(P)| <= eps and |\\hat \\phi(1-P)| <= eps

    Since 0 <= xi <= 1, the latter implies the former. Hence the formula which
    is easily derived using the explicit formula for g_{\\sigma} in Fourier.

    Parameters
    ----------
    sigma: float
        Positive number controlling the bandwidth of the filters
    eps : float, optional
        Positive number containing required precision. Defaults to 1e-7

    Returns
    -------
    P : int
        integer controlling the number of periods used to ensure the
        periodicity of the final Morlet filter in the frequency interval
        [0, 1[. The value of P will lead to the use of the frequency
        interval [1-P, P[, so that there are 2*P - 1 periods.
    """
    val = math.sqrt(-2 * sigma ** 2 * math.log(eps))
    P = int(math.ceil(val + 1))
    return P


def get_normalizing_factor(h_f, normalize='l1'):
    """
    Computes the desired normalization factor for a filter defined in Fourier.

    Parameters
    ----------
    h_f : array_like
        numpy vector containing the Fourier transform of a filter
    normalized : string, optional
        desired normalization type, either 'l1' or 'l2'. Defaults to 'l1'.

    Returns
    -------
    norm_factor : float
        such that h_f * norm_factor is the adequately normalized vector.
    """
    h_real = ifft(h_f)
    if np.abs(h_real).sum() < 1e-07:
        raise ValueError('Zero division error is very likely to occur, ' + 'aborting computations now.')
    if normalize == 'l1':
        norm_factor = 1.0 / np.abs(h_real).sum()
    elif normalize == 'l2':
        norm_factor = 1.0 / np.sqrt((np.abs(h_real) ** 2).sum())
    else:
        raise ValueError("Supported normalizations only include 'l1' and 'l2'")
    return norm_factor


def periodize_filter_fourier(h_f, nperiods=1):
    """
    Computes a periodization of a filter provided in the Fourier domain.

    Parameters
    ----------
    h_f : array_like
        complex numpy array of shape (N*n_periods,)
    n_periods: int, optional
        Number of periods which should be used to periodize

    Returns
    -------
    v_f : array_like
        complex numpy array of size (N,), which is a periodization of
        h_f as described in the formula:
        v_f[k] = sum_{i=0}^{n_periods - 1} h_f[i * N + k]
    """
    N = h_f.shape[0] // nperiods
    v_f = h_f.reshape(nperiods, N).mean(axis=0)
    return v_f


def gauss_1d(N, sigma, normalize='l1', P_max=5, eps=1e-07):
    """
    Computes the Fourier transform of a low pass gaussian window.

    \\hat g_{\\sigma}(\\omega) = e^{-\\omega^2 / 2 \\sigma^2}

    Parameters
    ----------
    N : int
        size of the temporal support
    sigma : float
        bandwidth parameter
    normalize : string, optional
        normalization types for the filters. Defaults to 'l1'
        Supported normalizations are 'l1' and 'l2' (understood in time domain).
    P_max : int, optional
        integer controlling the maximal number of periods to use to ensure
        the periodicity of the Fourier transform. (At most 2*P_max - 1 periods
        are used, to ensure an equal distribution around 0.5). Defaults to 5
        Should be >= 1
    eps : float, optional
        required machine precision (to choose the adequate P)

    Returns
    -------
    g_f : array_like
        numpy array of size (N,) containing the Fourier transform of the
        filter (with the frequencies in the np.fft.fftfreq convention).
    """
    if type(P_max) != int:
        raise ValueError('P_max should be an int, got {}'.format(type(P_max)))
    if P_max < 1:
        raise ValueError('P_max should be non-negative, got {}'.format(P_max))
    P = min(adaptive_choice_P(sigma, eps=eps), P_max)
    assert P >= 1
    if P == 1:
        freqs_low = np.fft.fftfreq(N)
    elif P > 1:
        freqs_low = np.arange((1 - P) * N, P * N, dtype=float) / float(N)
    g_f = np.exp(-freqs_low ** 2 / (2 * sigma ** 2))
    g_f = periodize_filter_fourier(g_f, nperiods=2 * P - 1)
    g_f *= get_normalizing_factor(g_f, normalize=normalize)
    return g_f


def morlet_1d(N, xi, sigma, normalize='l1', P_max=5, eps=1e-07):
    """
    Computes the Fourier transform of a Morlet filter.

    A Morlet filter is the sum of a Gabor filter and a low-pass filter
    to ensure that the sum has exactly zero mean in the temporal domain.
    It is defined by the following formula in time:
    psi(t) = g_{sigma}(t) (e^{i xi t} - beta)
    where g_{sigma} is a Gaussian envelope, xi is a frequency and beta is
    the cancelling parameter.

    Parameters
    ----------
    N : int
        size of the temporal support
    xi : float
        central frequency (in [0, 1])
    sigma : float
        bandwidth parameter
    normalize : string, optional
        normalization types for the filters. Defaults to 'l1'.
        Supported normalizations are 'l1' and 'l2' (understood in time domain).
    P_max: int, optional
        integer controlling the maximal number of periods to use to ensure
        the periodicity of the Fourier transform. (At most 2*P_max - 1 periods
        are used, to ensure an equal distribution around 0.5). Defaults to 5
        Should be >= 1
    eps : float
        required machine precision (to choose the adequate P)

    Returns
    -------
    morlet_f : array_like
        numpy array of size (N,) containing the Fourier transform of the Morlet
        filter at the frequencies given by np.fft.fftfreq(N).
    """
    if type(P_max) != int:
        raise ValueError('P_max should be an int, got {}'.format(type(P_max)))
    if P_max < 1:
        raise ValueError('P_max should be non-negative, got {}'.format(P_max))
    P = min(adaptive_choice_P(sigma, eps=eps), P_max)
    assert P >= 1
    freqs = np.arange((1 - P) * N, P * N, dtype=float) / float(N)
    if P == 1:
        freqs_low = np.fft.fftfreq(N)
    elif P > 1:
        freqs_low = freqs
    gabor_f = np.exp(-(freqs - xi) ** 2 / (2 * sigma ** 2))
    low_pass_f = np.exp(-freqs_low ** 2 / (2 * sigma ** 2))
    gabor_f = periodize_filter_fourier(gabor_f, nperiods=2 * P - 1)
    low_pass_f = periodize_filter_fourier(low_pass_f, nperiods=2 * P - 1)
    kappa = gabor_f[0] / low_pass_f[0]
    morlet_f = gabor_f - kappa * low_pass_f
    morlet_f *= get_normalizing_factor(morlet_f, normalize=normalize)
    return morlet_f


def scattering_filter_factory(J_support, J_scattering, Q, r_psi=math.sqrt(0.5), criterion_amplitude=0.001, normalize='l1', max_subsampling=None, sigma0=0.1, alpha=5.0, P_max=5, eps=1e-07, **kwargs):
    """
    Builds in Fourier the Morlet filters used for the scattering transform.

    Each single filter is provided as a dictionary with the following keys:
    * 'xi': central frequency, defaults to 0 for low-pass filters.
    * 'sigma': frequential width
    * k where k is an integer bounded below by 0. The maximal value for k
        depends on the type of filter, it is dynamically chosen depending
        on max_subsampling and the characteristics of the filters.
        Each value for k is an array (or tensor) of size 2**(J_support - k)
        containing the Fourier transform of the filter after subsampling by
        2**k

    Parameters
    ----------
    J_support : int
        2**J_support is the desired support size of the filters
    J_scattering : int
        parameter for the scattering transform (2**J_scattering
        corresponds to the averaging support of the low-pass filter)
    Q : int
        number of wavelets per octave at the first order. For audio signals,
        a value Q >= 12 is recommended in order to separate partials.
    r_psi : float, optional
        Should be >0 and <1. Controls the redundancy of the filters
        (the larger r_psi, the larger the overlap between adjacent wavelets).
        Defaults to sqrt(0.5).
    criterion_amplitude : float, optional
        Represents the numerical error which is allowed to be lost after
        convolution and padding. Defaults to 1e-3.
    normalize : string, optional
        Normalization convention for the filters (in the
        temporal domain). Supported values include 'l1' and 'l2'; a ValueError
        is raised otherwise. Defaults to 'l1'.
    max_subsampling: int or None, optional
        maximal dyadic subsampling to compute, in order
        to save computation time if it is not required. Defaults to None, in
        which case this value is dynamically adjusted depending on the filters.
    sigma0 : float, optional
        parameter controlling the frequential width of the
        low-pass filter at J_scattering=0; at a an absolute J_scattering, it
        is equal to sigma0 / 2**J_scattering. Defaults to 1e-1
    alpha : float, optional
        tolerance factor for the aliasing after subsampling.
        The larger alpha, the more conservative the value of maximal
        subsampling is. Defaults to 5.
    P_max : int, optional
        maximal number of periods to use to make sure that the Fourier
        transform of the filters is periodic. P_max = 5 is more than enough for
        double precision. Defaults to 5. Should be >= 1
    eps : float, optional
        required machine precision for the periodization (single
        floating point is enough for deep learning applications).
        Defaults to 1e-7

    Returns
    -------
    phi_f : dictionary
        a dictionary containing the low-pass filter at all possible
        subsamplings. See above for a description of the dictionary structure.
        The possible subsamplings are controlled by the inputs they can
        receive, which correspond to the subsamplings performed on top of the
        1st and 2nd order transforms.
    psi1_f : dictionary
        a dictionary containing the band-pass filters of the 1st order,
        only for the base resolution as no subsampling is used in the
        scattering tree.
        Each value corresponds to a dictionary for a single filter, see above
        for an exact description.
        The keys of this dictionary are of the type (j, n) where n is an
        integer counting the filters and j the maximal dyadic subsampling
        which can be performed on top of the filter without aliasing.
    psi2_f : dictionary
        a dictionary containing the band-pass filters of the 2nd order
        at all possible subsamplings. The subsamplings are determined by the
        input they can receive, which depends on the scattering tree.
        Each value corresponds to a dictionary for a single filter, see above
        for an exact description.
        The keys of this dictionary are of th etype (j, n) where n is an
        integer counting the filters and j is the maximal dyadic subsampling
        which can be performed on top of this filter without aliasing.
    t_max_phi : int
        temporal size to use to pad the signal on the right and on the
        left by making at most criterion_amplitude error. Assumes that the
        temporal support of the low-pass filter is larger than all filters.

    Refs
    ----
    Convolutional operators in the time-frequency domain, V. Lostanlen,
    PhD Thesis, 2017
    https://tel.archives-ouvertes.fr/tel-01559667
    """
    sigma_low, xi1, sigma1, j1s, xi2, sigma2, j2s = calibrate_scattering_filters(J_scattering, Q, r_psi=r_psi, sigma0=sigma0, alpha=alpha)
    phi_f = {}
    psi1_f = []
    psi2_f = []
    for n2, j2 in enumerate(j2s):
        if max_subsampling is None:
            possible_subsamplings_after_order1 = [j1 for j1 in j1s if j2 > j1]
            if len(possible_subsamplings_after_order1) > 0:
                max_sub_psi2 = max(possible_subsamplings_after_order1)
            else:
                max_sub_psi2 = 0
        else:
            max_sub_psi2 = max_subsampling
        T = 2 ** J_support
        psi_f = {}
        psi_f[0] = morlet_1d(T, xi2[n2], sigma2[n2], normalize=normalize, P_max=P_max, eps=eps)
        for subsampling in range(1, max_sub_psi2 + 1):
            factor_subsampling = 2 ** subsampling
            psi_f[subsampling] = periodize_filter_fourier(psi_f[0], nperiods=factor_subsampling)
        psi2_f.append(psi_f)
    for n1, j1 in enumerate(j1s):
        T = 2 ** J_support
        psi1_f.append({(0): morlet_1d(T, xi1[n1], sigma1[n1], normalize=normalize, P_max=P_max, eps=eps)})
    if max_subsampling is None:
        max_subsampling_after_psi1 = max(j1s)
        max_subsampling_after_psi2 = max(j2s)
        max_sub_phi = max(max_subsampling_after_psi1, max_subsampling_after_psi2)
    else:
        max_sub_phi = max_subsampling
    phi_f[0] = gauss_1d(T, sigma_low, P_max=P_max, eps=eps)
    for subsampling in range(1, max_sub_phi + 1):
        factor_subsampling = 2 ** subsampling
        phi_f[subsampling] = periodize_filter_fourier(phi_f[0], nperiods=factor_subsampling)
    for n1, j1 in enumerate(j1s):
        psi1_f[n1]['xi'] = xi1[n1]
        psi1_f[n1]['sigma'] = sigma1[n1]
        psi1_f[n1]['j'] = j1
    for n2, j2 in enumerate(j2s):
        psi2_f[n2]['xi'] = xi2[n2]
        psi2_f[n2]['sigma'] = sigma2[n2]
        psi2_f[n2]['j'] = j2
    phi_f['xi'] = 0.0
    phi_f['sigma'] = sigma_low
    phi_f['j'] = 0
    t_max_phi = compute_temporal_support(phi_f[0].reshape(1, -1), criterion_amplitude=criterion_amplitude)
    return phi_f, psi1_f, psi2_f, t_max_phi


def compute_minimum_support_to_pad(T, J, Q, criterion_amplitude=0.001, normalize='l1', r_psi=math.sqrt(0.5), sigma0=0.1, alpha=5.0, P_max=5, eps=1e-07):
    """
    Computes the support to pad given the input size and the parameters of the
    scattering transform.

    Parameters
    ----------
    T : int
        temporal size of the input signal
    J : int
        scale of the scattering
    Q : int
        number of wavelets per octave
    normalize : string, optional
        normalization type for the wavelets.
        Only `'l2'` or `'l1'` normalizations are supported.
        Defaults to `'l1'`
    criterion_amplitude: float `>0` and `<1`, optional
        Represents the numerical error which is allowed to be lost after
        convolution and padding.
        The larger criterion_amplitude, the smaller the padding size is.
        Defaults to `1e-3`
    r_psi : float, optional
        Should be `>0` and `<1`. Controls the redundancy of the filters
        (the larger r_psi, the larger the overlap between adjacent
        wavelets).
        Defaults to `sqrt(0.5)`.
    sigma0 : float, optional
        parameter controlling the frequential width of the
        low-pass filter at J_scattering=0; at a an absolute J_scattering,
        it is equal to :math:`\\frac{\\sigma_0}{2^J}`.
        Defaults to `1e-1`.
    alpha : float, optional
        tolerance factor for the aliasing after subsampling.
        The larger the alpha, the more conservative the value of maximal
        subsampling is.
        Defaults to `5`.
    P_max : int, optional
        maximal number of periods to use to make sure that the Fourier
        transform of the filters is periodic.
        `P_max = 5` is more than enough for double precision.
        Defaults to `5`.
    eps : float, optional
        required machine precision for the periodization (single
        floating point is enough for deep learning applications).
        Defaults to `1e-7`.

    Returns
    -------
    min_to_pad: int
        minimal value to pad the signal on one size to avoid any
        boundary error.
    """
    J_tentative = int(np.ceil(np.log2(T)))
    _, _, _, t_max_phi = scattering_filter_factory(J_tentative, J, Q, normalize=normalize, to_torch=False, max_subsampling=0, criterion_amplitude=criterion_amplitude, r_psi=r_psi, sigma0=sigma0, alpha=alpha, P_max=P_max, eps=eps)
    min_to_pad = 3 * t_max_phi
    return min_to_pad


def compute_padding(J_pad, T):
    """
    Computes the padding to be added on the left and on the right
    of the signal.

    It should hold that 2**J_pad >= T

    Parameters
    ----------
    J_pad : int
        2**J_pad is the support of the padded signal
    T : int
        original signal support size

    Returns
    -------
    pad_left: amount to pad on the left ("beginning" of the support)
    pad_right: amount to pad on the right ("end" of the support)
    """
    T_pad = 2 ** J_pad
    if T_pad < T:
        raise ValueError('Padding support should be larger than the original' + 'signal size!')
    to_add = 2 ** J_pad - T
    pad_left = to_add // 2
    pad_right = to_add - pad_left
    if max(pad_left, pad_right) >= T:
        raise ValueError('Too large padding value, will lead to NaN errors')
    return pad_left, pad_right


def precompute_size_scattering(J, Q, max_order=2, detail=False):
    """Get size of the scattering transform

    The number of scattering coefficients depends on the filter
    configuration and so can be calculated using a few of the scattering
    transform parameters.

    Parameters
    ----------
    J : int
        The maximum log-scale of the scattering transform.
        In other words, the maximum scale is given by `2**J`.
    Q : int >= 1
        The number of first-order wavelets per octave.
        Second-order wavelets are fixed to one wavelet per octave.
    max_order : int, optional
        The maximum order of scattering coefficients to compute.
        Must be either equal to `1` or `2`. Defaults to `2`.
    detail : boolean, optional
        Specifies whether to provide a detailed size (number of coefficient
        per order) or an aggregate size (total number of coefficients).

    Returns
    -------
    size : int or tuple
        If `detail` is `False`, returns the number of coefficients as an
        integer. If `True`, returns a tuple of size `max_order` containing
        the number of coefficients in each order.
    """
    sigma_low, xi1, sigma1, j1, xi2, sigma2, j2 = calibrate_scattering_filters(J, Q)
    size_order0 = 1
    size_order1 = len(xi1)
    size_order2 = 0
    for n1 in range(len(xi1)):
        for n2 in range(len(xi2)):
            if j2[n2] > j1[n1]:
                size_order2 += 1
    if detail:
        if max_order == 2:
            return size_order0, size_order1, size_order2
        else:
            return size_order0, size_order1
    elif max_order == 2:
        return size_order0 + size_order1 + size_order2
    else:
        return size_order0 + size_order1


class ScatteringBase1D(ScatteringBase):

    def __init__(self, J, shape, Q=1, max_order=2, average=True, oversampling=0, vectorize=True, out_type='array', backend=None):
        super(ScatteringBase1D, self).__init__()
        self.J = J
        self.shape = shape
        self.Q = Q
        self.max_order = max_order
        self.average = average
        self.oversampling = oversampling
        self.vectorize = vectorize
        self.out_type = out_type
        self.backend = backend

    def build(self):
        """Set up padding and filters

        Certain internal data, such as the amount of padding and the wavelet
        filters to be used in the scattering transform, need to be computed
        from the parameters given during construction. This function is called
        automatically during object creation and no subsequent calls are
        therefore needed.
        """
        self.r_psi = math.sqrt(0.5)
        self.sigma0 = 0.1
        self.alpha = 5.0
        self.P_max = 5
        self.eps = 1e-07
        self.criterion_amplitude = 0.001
        self.normalize = 'l1'
        if isinstance(self.shape, numbers.Integral):
            self.T = self.shape
        elif isinstance(self.shape, tuple):
            self.T = self.shape[0]
            if len(self.shape) > 1:
                raise ValueError('If shape is specified as a tuple, it must have exactly one element')
        else:
            raise ValueError('shape must be an integer or a 1-tuple')
        min_to_pad = compute_minimum_support_to_pad(self.T, self.J, self.Q, r_psi=self.r_psi, sigma0=self.sigma0, alpha=self.alpha, P_max=self.P_max, eps=self.eps, criterion_amplitude=self.criterion_amplitude, normalize=self.normalize)
        J_max_support = int(np.floor(np.log2(3 * self.T - 2)))
        self.J_pad = min(int(np.ceil(np.log2(self.T + 2 * min_to_pad))), J_max_support)
        self.pad_left, self.pad_right = compute_padding(self.J_pad, self.T)
        self.ind_start, self.ind_end = compute_border_indices(self.J, self.pad_left, self.pad_left + self.T)

    def create_filters(self):
        self.phi_f, self.psi1_f, self.psi2_f, _ = scattering_filter_factory(self.J_pad, self.J, self.Q, normalize=self.normalize, criterion_amplitude=self.criterion_amplitude, r_psi=self.r_psi, sigma0=self.sigma0, alpha=self.alpha, P_max=self.P_max, eps=self.eps)

    def meta(self):
        """Get meta information on the transform

        Calls the static method `compute_meta_scattering()` with the
        parameters of the transform object.

        Returns
        ------
        meta : dictionary
            See the documentation for `compute_meta_scattering()`.
        """
        return compute_meta_scattering(self.J, self.Q, max_order=self.max_order)

    def output_size(self, detail=False):
        """Get size of the scattering transform

        Calls the static method `precompute_size_scattering()` with the
        parameters of the transform object.

        Parameters
        ----------
        detail : boolean, optional
            Specifies whether to provide a detailed size (number of coefficient
            per order) or an aggregate size (total number of coefficients).

        Returns
        ------
        size : int or tuple
            See the documentation for `precompute_size_scattering()`.
        """
        return precompute_size_scattering(self.J, self.Q, max_order=self.max_order, detail=detail)
    _doc_shape = 'T'
    _doc_instantiation_shape = {(True): 'S = Scattering1D(J, T, Q)', (False): 'S = Scattering1D(J, Q)'}
    _doc_param_shape = """shape : int
            The length of the input signals.
        """
    _doc_attrs_shape = """J_pad : int
            The logarithm of the padded length of the signals.
        pad_left : int
            The amount of padding to the left of the signal.
        pad_right : int
            The amount of padding to the right of the signal.
        phi_f : dictionary
            A dictionary containing the lowpass filter at all resolutions. See
            `filter_bank.scattering_filter_factory` for an exact description.
        psi1_f : dictionary
            A dictionary containing all the first-order wavelet filters, each
            represented as a dictionary containing that filter at all
            resolutions. See `filter_bank.scattering_filter_factory` for an
            exact description.
        psi2_f : dictionary
            A dictionary containing all the second-order wavelet filters, each
            represented as a dictionary containing that filter at all
            resolutions. See `filter_bank.scattering_filter_factory` for an
            exact description.
        """
    _doc_param_average = """average : boolean, optional
            Determines whether the output is averaged in time or not. The
            averaged output corresponds to the standard scattering transform,
            while the un-averaged output skips the last convolution by
            :math:`\\phi_J(t)`.  This parameter may be modified after object
            creation. Defaults to `True`.
        """
    _doc_attr_average = """average : boolean
            Controls whether the output should be averaged (the standard
            scattering transform) or not (resulting in wavelet modulus
            coefficients). Note that to obtain unaveraged output, the
            `vectorize` flag must be set to `False` or `out_type` must be set
            to `'list'`.
     """
    _doc_param_vectorize = """vectorize : boolean, optional
            Determines wheter to return a vectorized scattering transform
            (that is, a large array containing the output) or a dictionary
            (where each entry corresponds to a separate scattering
            coefficient). This parameter may be modified after object
            creation. Deprecated in favor of `out_type` (see below). Defaults
            to True.
        out_type : str, optional
            The format of the output of a scattering transform. If set to
            `'list'`, then the output is a list containing each individual
            scattering coefficient with meta information. Otherwise, if set to
            `'array'`, the output is a large array containing the
            concatenation of all scattering coefficients. Defaults to
            `'array'`.
        """
    _doc_attr_vectorize = """vectorize : boolean
            Controls whether the output should be vectorized into a single
            Tensor or collected into a dictionary. Deprecated in favor of
            `out_type`. For more details, see the documentation for
            `scattering`.
        out_type : str
            Specifices the output format of the transform, which is currently
            one of `'array'` or `'list`'. If `'array'`, the output is a large
            array containing the scattering coefficients. If `'list`', the
            output is a list of dictionaries, each containing a scattering
            coefficient along with meta information. For more information, see
            the documentation for `scattering`.
        """
    _doc_class = """The 1D scattering transform

        The scattering transform computes a cascade of wavelet transforms
        alternated with a complex modulus non-linearity. The scattering
        transform of a 1D signal :math:`x(t)` may be written as

            $S_J x = [S_J^{{(0)}} x, S_J^{{(1)}} x, S_J^{{(2)}} x]$

        where

            $S_J^{{(0)}} x(t) = x \\star \\phi_J(t)$,

            $S_J^{{(1)}} x(t, \\lambda) = |x \\star \\psi_\\lambda^{{(1)}}| \\star \\phi_J$, and

            $S_J^{{(2)}} x(t, \\lambda, \\mu) = |\\,| x \\star \\psi_\\lambda^{{(1)}}| \\star \\psi_\\mu^{{(2)}} | \\star \\phi_J$.

        In the above formulas, :math:`\\star` denotes convolution in time. The
        filters $\\psi_\\lambda^{{(1)}}(t)$ and $\\psi_\\mu^{{(2)}}(t)$ are analytic
        wavelets with center frequencies $\\lambda$ and $\\mu$, while
        $\\phi_J(t)$ is a real lowpass filter centered at the zero frequency.

        The `Scattering1D` class implements the 1D scattering transform for a
        given set of filters whose parameters are specified at initialization.
        While the wavelets are fixed, other parameters may be changed after
        the object is created, such as whether to compute all of
        :math:`S_J^{{(0)}} x`, $S_J^{{(1)}} x$, and $S_J^{{(2)}} x$ or just
        $S_J^{{(0)}} x$ and $S_J^{{(1)}} x$.
        {frontend_paragraph}
        Given an input `{array}` `x` of shape `(B, T)`, where `B` is the
        number of signals to transform (the batch size) and `T` is the length
        of the signal, we compute its scattering transform by passing it to
        the `scattering` method (or calling the alias `{alias_name}`). Note
        that `B` can be one, in which case it may be omitted, giving an input
        of shape `(T,)`.

        Example
        -------
        ::

            # Set the parameters of the scattering transform.
            J = 6
            T = 2 ** 13
            Q = 8

            # Generate a sample signal.
            x = {sample}

            # Define a Scattering1D object.
            {instantiation}

            # Calculate the scattering transform.
            Sx = S.scattering(x)

            # Equivalently, use the alias.
            Sx = S{alias_call}(x)

        Above, the length of the signal is :math:`T = 2^{{13}} = 8192`, while the
        maximum scale of the scattering transform is set to :math:`2^J = 2^6 =
        64`. The time-frequency resolution of the first-order wavelets
        :math:`\\psi_\\lambda^{{(1)}}(t)` is set to `Q = 8` wavelets per octave.
        The second-order wavelets :math:`\\psi_\\mu^{{(2)}}(t)` always have one
        wavelet per octave.

        Parameters
        ----------
        J : int
            The maximum log-scale of the scattering transform. In other words,
            the maximum scale is given by :math:`2^J`.
        {param_shape}Q : int >= 1
            The number of first-order wavelets per octave (second-order
            wavelets are fixed to one wavelet per octave). Defaults to `1`.
        max_order : int, optional
            The maximum order of scattering coefficients to compute. Must be
            either `1` or `2`. Defaults to `2`.
        {param_average}oversampling : integer >= 0, optional
            Controls the oversampling factor relative to the default as a
            power of two. Since the convolving by wavelets (or lowpass
            filters) and taking the modulus reduces the high-frequency content
            of the signal, we can subsample to save space and improve
            performance. However, this may reduce precision in the
            calculation. If this is not desirable, `oversampling` can be set
            to a large value to prevent too much subsampling. This parameter
            may be modified after object creation. Defaults to `0`.
        {param_vectorize}
        Attributes
        ----------
        J : int
            The maximum log-scale of the scattering transform. In other words,
            the maximum scale is given by `2 ** J`.
        {param_shape}Q : int
            The number of first-order wavelets per octave (second-order
            wavelets are fixed to one wavelet per octave).
        {attrs_shape}max_order : int
            The maximum scattering order of the transform.
        {attr_average}oversampling : int
            The number of powers of two to oversample the output compared to
            the default subsampling rate determined from the filters.
        {attr_vectorize}"""
    _doc_scattering = """Apply the scattering transform

       Given an input `{array}` of size `(B, T)`, where `B` is the batch
       size (it can be potentially an integer or a shape) and `T` is the length
       of the individual signals, this function computes its scattering
       transform. If the `vectorize` flag is set to `True` (or if it is not
       available in this frontend), the output is in the form of a `{array}`
       or size `(B, C, T1)`, where `T1` is the signal length after subsampling
       to the scale :math:`2^J` (with the appropriate oversampling factor to
       reduce aliasing), and `C` is the number of scattering coefficients. If
       `vectorize` is set `False`, however, the output is a dictionary
       containing `C` keys, each a tuple whose length corresponds to the
       scattering order and whose elements are the sequence of filter indices
       used.

       Note that the `vectorize` flag has been deprecated in favor of the
       `out_type` parameter. If this is set to `'array'` (the default), the
       `vectorize` flag is still respected, but if not, `out_type` takes
       precedence. The two current output types are `'array'` and `'list'`.
       The former gives the type of output described above. If set to
       `'list'`, however, the output is a list of dictionaries, each
       dictionary corresponding to a scattering coefficient and its associated
       meta information. The coefficient is stored under the `'coef'` key,
       while other keys contain additional information, such as `'j'` (the
       scale of the filter used) and `'n`' (the filter index).

       Furthermore, if the `average` flag is set to `False`, these outputs
       are not averaged, but are simply the wavelet modulus coefficients of
       the filters.

       Parameters
       ----------
       x : {array}
           An input `{array}` of size `(B, T)`.

       Returns
       -------
       S : tensor or dictionary
           If `out_type` is `'array'` and the `vectorize` flag is `True`, the
           output is a{n} `{array}` containing the scattering coefficients,
           while if `vectorize` is `False`, it is a dictionary indexed by
           tuples of filter indices. If `out_type` is `'list'`, the output is
           a list of dictionaries as described above.
    """

    @classmethod
    def _document(cls):
        instantiation = cls._doc_instantiation_shape[cls._doc_has_shape]
        param_shape = cls._doc_param_shape if cls._doc_has_shape else ''
        attrs_shape = cls._doc_attrs_shape if cls._doc_has_shape else ''
        param_average = cls._doc_param_average if cls._doc_has_out_type else ''
        attr_average = cls._doc_attr_average if cls._doc_has_out_type else ''
        param_vectorize = cls._doc_param_vectorize if cls._doc_has_out_type else ''
        attr_vectorize = cls._doc_attr_vectorize if cls._doc_has_out_type else ''
        cls.__doc__ = ScatteringBase1D._doc_class.format(array=cls._doc_array, frontend_paragraph=cls._doc_frontend_paragraph, alias_name=cls._doc_alias_name, alias_call=cls._doc_alias_call, instantiation=instantiation, param_shape=param_shape, attrs_shape=attrs_shape, param_average=param_average, attr_average=attr_average, param_vectorize=param_vectorize, attr_vectorize=attr_vectorize, sample=cls._doc_sample.format(shape=cls._doc_shape))
        cls.scattering.__doc__ = ScatteringBase1D._doc_scattering.format(array=cls._doc_array, n=cls._doc_array_n)


def scattering1d(x, pad, unpad, backend, J, psi1, psi2, phi, pad_left=0, pad_right=0, ind_start=None, ind_end=None, oversampling=0, max_order=2, average=True, size_scattering=(0, 0, 0), vectorize=False, out_type='array'):
    """
    Main function implementing the 1-D scattering transform.

    Parameters
    ----------
    x : Tensor
        a torch Tensor of size `(B, 1, T)` where `T` is the temporal size
    psi1 : dictionary
        a dictionary of filters (in the Fourier domain), with keys (`j`, `q`).
        `j` corresponds to the downsampling factor for
        :math:`x \\ast psi1[(j, q)]``, and `q` corresponds to a pitch class
        (chroma).
        * psi1[(j, n)] is itself a dictionary, with keys corresponding to the
        dilation factors: psi1[(j, n)][j2] corresponds to a support of size
        :math:`2^{J_\\text{max} - j_2}`, where :math:`J_\\text{max}` has been
        defined a priori (`J_max = size` of the padding support of the input)
        * psi1[(j, n)] only has real values;
        the tensors are complex so that broadcasting applies
    psi2 : dictionary
        a dictionary of filters, with keys (j2, n2). Same remarks as for psi1
    phi : dictionary
        a dictionary of filters of scale :math:`2^J` with keys (`j`)
        where :math:`2^j` is the downsampling factor.
        The array `phi[j]` is a real-valued filter.
    J : int
        scale of the scattering
    pad_left : int, optional
        how much to pad the signal on the left. Defaults to `0`
    pad_right : int, optional
        how much to pad the signal on the right. Defaults to `0`
    ind_start : dictionary of ints, optional
        indices to truncate the signal to recover only the
        parts which correspond to the actual signal after padding and
        downsampling. Defaults to None
    ind_end : dictionary of ints, optional
        See description of ind_start
    oversampling : int, optional
        how much to oversample the scattering (with respect to :math:`2^J`):
        the higher, the larger the resulting scattering
        tensor along time. Defaults to `0`
    order2 : boolean, optional
        Whether to compute the 2nd order or not. Defaults to `False`.
    average_U1 : boolean, optional
        whether to average the first order vector. Defaults to `True`
    size_scattering : tuple
        Contains the number of channels of the scattering, precomputed for
        speed-up. Defaults to `(0, 0, 0)`.
    vectorize : boolean, optional
        whether to return a dictionary or a tensor. Defaults to False.

    """
    subsample_fourier = backend.subsample_fourier
    modulus_complex = backend.modulus_complex
    real = backend.real
    fft = backend.fft
    cdgmm = backend.cdgmm
    concatenate = backend.concatenate
    batch_size = x.shape[0]
    kJ = max(J - oversampling, 0)
    temporal_size = ind_end[kJ] - ind_start[kJ]
    out_S_0, out_S_1, out_S_2 = [], [], []
    U_0 = pad(x, pad_left=pad_left, pad_right=pad_right)
    U_0_hat = fft(U_0, 'C2C')
    k0 = max(J - oversampling, 0)
    if average:
        S_0_c = cdgmm(U_0_hat, phi[0])
        S_0_hat = subsample_fourier(S_0_c, 2 ** k0)
        S_0_r = fft(S_0_hat, 'C2R', inverse=True)
        S_0 = unpad(S_0_r, ind_start[k0], ind_end[k0])
    else:
        S_0 = x
    out_S_0.append({'coef': S_0, 'j': (), 'n': ()})
    for n1 in range(len(psi1)):
        j1 = psi1[n1]['j']
        k1 = max(j1 - oversampling, 0)
        assert psi1[n1]['xi'] < 0.5 / 2 ** k1
        U_1_c = cdgmm(U_0_hat, psi1[n1][0])
        U_1_hat = subsample_fourier(U_1_c, 2 ** k1)
        U_1_c = fft(U_1_hat, 'C2C', inverse=True)
        U_1_m = modulus_complex(U_1_c)
        if average or max_order > 1:
            U_1_hat = fft(U_1_m, 'C2C')
        if average:
            k1_J = max(J - k1 - oversampling, 0)
            S_1_c = cdgmm(U_1_hat, phi[k1])
            S_1_hat = subsample_fourier(S_1_c, 2 ** k1_J)
            S_1_r = fft(S_1_hat, 'C2R', inverse=True)
            S_1 = unpad(S_1_r, ind_start[k1_J + k1], ind_end[k1_J + k1])
        else:
            U_1_r = real(U_1_m)
            S_1 = unpad(U_1_r, ind_start[k1], ind_end[k1])
        out_S_1.append({'coef': S_1, 'j': (j1,), 'n': (n1,)})
        if max_order == 2:
            for n2 in range(len(psi2)):
                j2 = psi2[n2]['j']
                if j2 > j1:
                    assert psi2[n2]['xi'] < psi1[n1]['xi']
                    k2 = max(j2 - k1 - oversampling, 0)
                    U_2_c = cdgmm(U_1_hat, psi2[n2][k1])
                    U_2_hat = subsample_fourier(U_2_c, 2 ** k2)
                    U_2_c = fft(U_2_hat, 'C2C', inverse=True)
                    U_2_m = modulus_complex(U_2_c)
                    if average:
                        U_2_hat = fft(U_2_m, 'C2C')
                        k2_J = max(J - k2 - k1 - oversampling, 0)
                        S_2_c = cdgmm(U_2_hat, phi[k1 + k2])
                        S_2_hat = subsample_fourier(S_2_c, 2 ** k2_J)
                        S_2_r = fft(S_2_hat, 'C2R', inverse=True)
                        S_2 = unpad(S_2_r, ind_start[k1 + k2 + k2_J], ind_end[k1 + k2 + k2_J])
                    else:
                        U_2_r = real(U_2_m)
                        S_2 = unpad(U_2_r, ind_start[k1 + k2], ind_end[k1 + k2])
                    out_S_2.append({'coef': S_2, 'j': (j1, j2), 'n': (n1, n2)})
    out_S = []
    out_S.extend(out_S_0)
    out_S.extend(out_S_1)
    out_S.extend(out_S_2)
    if out_type == 'array' and vectorize:
        out_S = concatenate([x['coef'] for x in out_S])
    elif out_type == 'array' and not vectorize:
        out_S = {x['n']: x['coef'] for x in out_S}
    elif out_type == 'list':
        for x in out_S:
            x.pop('n')
    return out_S


class ScatteringTorch1D(ScatteringTorch, ScatteringBase1D):

    def __init__(self, J, shape, Q=1, max_order=2, average=True, oversampling=0, vectorize=True, out_type='array', backend='torch'):
        ScatteringTorch.__init__(self)
        ScatteringBase1D.__init__(self, J, shape, Q, max_order, average, oversampling, vectorize, out_type, backend)
        ScatteringBase1D._instantiate_backend(self, 'kymatio.scattering1d.backend.')
        ScatteringBase1D.build(self)
        ScatteringBase1D.create_filters(self)
        self.register_filters()

    def register_filters(self):
        """ This function run the filterbank function that
        will create the filters as numpy array, and then, it
        saves those arrays as module's buffers."""
        n = 0
        for k in self.phi_f.keys():
            if type(k) != str:
                self.phi_f[k] = torch.from_numpy(self.phi_f[k]).float().view(-1, 1)
                self.register_buffer('tensor' + str(n), self.phi_f[k])
                n += 1
        for psi_f in self.psi1_f:
            for sub_k in psi_f.keys():
                if type(sub_k) != str:
                    psi_f[sub_k] = torch.from_numpy(psi_f[sub_k]).float().view(-1, 1)
                    self.register_buffer('tensor' + str(n), psi_f[sub_k])
                    n += 1
        for psi_f in self.psi2_f:
            for sub_k in psi_f.keys():
                if type(sub_k) != str:
                    psi_f[sub_k] = torch.from_numpy(psi_f[sub_k]).float().view(-1, 1)
                    self.register_buffer('tensor' + str(n), psi_f[sub_k])
                    n += 1

    def load_filters(self):
        """This function loads filters from the module's buffer """
        buffer_dict = dict(self.named_buffers())
        n = 0
        for k in self.phi_f.keys():
            if type(k) != str:
                self.phi_f[k] = buffer_dict['tensor' + str(n)]
                n += 1
        for psi_f in self.psi1_f:
            for sub_k in psi_f.keys():
                if type(sub_k) != str:
                    psi_f[sub_k] = buffer_dict['tensor' + str(n)]
                    n += 1
        for psi_f in self.psi2_f:
            for sub_k in psi_f.keys():
                if type(sub_k) != str:
                    psi_f[sub_k] = buffer_dict['tensor' + str(n)]
                    n += 1

    def scattering(self, x):
        if len(x.shape) < 1:
            raise ValueError('Input tensor x should have at least one axis, got {}'.format(len(x.shape)))
        if not self.out_type in ('array', 'list'):
            raise RuntimeError("The out_type must be one of 'array' or 'list'.")
        if not self.average and self.out_type == 'array' and self.vectorize:
            raise ValueError("Options average=False, out_type='array' and vectorize=True are mutually incompatible. Please set out_type to 'list' or vectorize to False.")
        if not self.vectorize:
            warnings.warn("The vectorize option is deprecated and will be removed in version 0.3. Please set out_type='list' for equivalent functionality.", DeprecationWarning)
        batch_shape = x.shape[:-1]
        signal_shape = x.shape[-1:]
        x = x.reshape((-1, 1) + signal_shape)
        self.load_filters()
        if self.vectorize:
            size_scattering = precompute_size_scattering(self.J, self.Q, max_order=self.max_order, detail=True)
        else:
            size_scattering = 0
        S = scattering1d(x, self.backend.pad, self.backend.unpad, self.backend, self.J, self.psi1_f, self.psi2_f, self.phi_f, max_order=self.max_order, average=self.average, pad_left=self.pad_left, pad_right=self.pad_right, ind_start=self.ind_start, ind_end=self.ind_end, oversampling=self.oversampling, vectorize=self.vectorize, size_scattering=size_scattering, out_type=self.out_type)
        if self.out_type == 'array' and self.vectorize:
            scattering_shape = S.shape[-2:]
            new_shape = batch_shape + scattering_shape
            S = S.reshape(new_shape)
        elif self.out_type == 'array' and not self.vectorize:
            for k, v in S.items():
                scattering_shape = v.shape[-2:]
                new_shape = batch_shape + scattering_shape
                S[k] = v.reshape(new_shape)
        elif self.out_type == 'list':
            for x in S:
                scattering_shape = x['coef'].shape[-1:]
                new_shape = batch_shape + scattering_shape
                x['coef'] = x['coef'].reshape(new_shape)
        return S


def fft2(x):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        return scipy.fftpack.fft2(x)


def gabor_2d(M, N, sigma, theta, xi, slant=1.0, offset=0):
    """
        Computes a 2D Gabor filter.
        A Gabor filter is defined by the following formula in space:
        psi(u) = g_{sigma}(u) e^(i xi^T u)
        where g_{sigma} is a Gaussian envelope and xi is a frequency.

        Parameters
        ----------
        M, N : int
            spatial sizes
        sigma : float
            bandwidth parameter
        xi : float
            central frequency (in [0, 1])
        theta : float
            angle in [0, pi]
        slant : float, optional
            parameter which guides the elipsoidal shape of the morlet
        offset : int, optional
            offset by which the signal starts

        Returns
        -------
        morlet_fft : ndarray
            numpy array of size (M, N)
    """
    gab = np.zeros((M, N), np.complex64)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], np.float32)
    R_inv = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]], np.float32)
    D = np.array([[1, 0], [0, slant * slant]])
    curv = np.dot(R, np.dot(D, R_inv)) / (2 * sigma * sigma)
    for ex in [-2, -1, 0, 1, 2]:
        for ey in [-2, -1, 0, 1, 2]:
            [xx, yy] = np.mgrid[offset + ex * M:offset + M + ex * M, offset + ey * N:offset + N + ey * N]
            arg = -(curv[0, 0] * np.multiply(xx, xx) + (curv[0, 1] + curv[1, 0]) * np.multiply(xx, yy) + curv[1, 1] * np.multiply(yy, yy)) + 1.0j * (xx * xi * np.cos(theta) + yy * xi * np.sin(theta))
            gab += np.exp(arg)
    norm_factor = 2 * 3.1415 * sigma * sigma / slant
    gab /= norm_factor
    return gab


def morlet_2d(M, N, sigma, theta, xi, slant=0.5, offset=0):
    """
        Computes a 2D Morlet filter.
        A Morlet filter is the sum of a Gabor filter and a low-pass filter
        to ensure that the sum has exactly zero mean in the temporal domain.
        It is defined by the following formula in space:
        psi(u) = g_{sigma}(u) (e^(i xi^T u) - beta)
        where g_{sigma} is a Gaussian envelope, xi is a frequency and beta is
        the cancelling parameter.

        Parameters
        ----------
        M, N : int
            spatial sizes
        sigma : float
            bandwidth parameter
        xi : float
            central frequency (in [0, 1])
        theta : float
            angle in [0, pi]
        slant : float, optional
            parameter which guides the elipsoidal shape of the morlet
        offset : int, optional
            offset by which the signal starts

        Returns
        -------
        morlet_fft : ndarray
            numpy array of size (M, N)
    """
    wv = gabor_2d(M, N, sigma, theta, xi, slant, offset)
    wv_modulus = gabor_2d(M, N, sigma, theta, 0, slant, offset)
    K = np.sum(wv) / np.sum(wv_modulus)
    mor = wv - K * wv_modulus
    return mor


def periodize_filter_fft(x, res):
    """
        Parameters
        ----------
        x : numpy array
            signal to periodize in Fourier
        res :
            resolution to which the signal is cropped.

        Returns
        -------
        crop : numpy array
            It returns a crop version of the filter, assuming that
             the convolutions will be done via compactly supported signals.
    """
    M = x.shape[0]
    N = x.shape[1]
    crop = np.zeros((M // 2 ** res, N // 2 ** res), x.dtype)
    mask = np.ones(x.shape, np.float32)
    len_x = int(M * (1 - 2 ** -res))
    start_x = int(M * 2 ** (-res - 1))
    len_y = int(N * (1 - 2 ** -res))
    start_y = int(N * 2 ** (-res - 1))
    mask[start_x:start_x + len_x, :] = 0
    mask[:, start_y:start_y + len_y] = 0
    x = np.multiply(x, mask)
    for k in range(int(M / 2 ** res)):
        for l in range(int(N / 2 ** res)):
            for i in range(int(2 ** res)):
                for j in range(int(2 ** res)):
                    crop[k, l] += x[k + i * int(M / 2 ** res), l + j * int(N / 2 ** res)]
    return crop


def filter_bank(M, N, J, L=8):
    """
        Builds in Fourier the Morlet filters used for the scattering transform.
        Each single filter is provided as a dictionary with the following keys:
        * 'j' : scale
        * 'theta' : angle used
        Parameters
        ----------
        M, N : int
            spatial support of the input
        J : int
            logscale of the scattering
        L : int, optional
            number of angles used for the wavelet transform
        Returns
        -------
        filters : list
            A two list of dictionary containing respectively the low-pass and
             wavelet filters.
        Notes
        -----
        The design of the filters is optimized for the value L = 8.
    """
    filters = {}
    filters['psi'] = []
    for j in range(J):
        for theta in range(L):
            psi = {}
            psi['j'] = j
            psi['theta'] = theta
            psi_signal = morlet_2d(M, N, 0.8 * 2 ** j, (int(L - L / 2 - 1) - theta) * np.pi / L, 3.0 / 4.0 * np.pi / 2 ** j, 4.0 / L)
            psi_signal_fourier = fft2(psi_signal)
            psi_signal_fourier = np.real(psi_signal_fourier)
            for res in range(min(j + 1, max(J - 1, 1))):
                psi_signal_fourier_res = periodize_filter_fft(psi_signal_fourier, res)
                psi[res] = psi_signal_fourier_res
            filters['psi'].append(psi)
    filters['phi'] = {}
    phi_signal = gabor_2d(M, N, 0.8 * 2 ** (J - 1), 0, 0)
    phi_signal_fourier = fft2(phi_signal)
    phi_signal_fourier = np.real(phi_signal_fourier)
    filters['phi']['j'] = J
    for res in range(J):
        phi_signal_fourier_res = periodize_filter_fft(phi_signal_fourier, res)
        filters['phi'][res] = phi_signal_fourier_res
    return filters


class ScatteringBase2D(ScatteringBase):

    def __init__(self, J, shape, L=8, max_order=2, pre_pad=False, backend=None, out_type='array'):
        super(ScatteringBase2D, self).__init__()
        self.pre_pad = pre_pad
        self.L = L
        self.backend = backend
        self.J = J
        self.shape = shape
        self.max_order = max_order
        self.out_type = out_type

    def build(self):
        self.M, self.N = self.shape
        if 2 ** self.J > self.M or 2 ** self.J > self.N:
            raise RuntimeError('The smallest dimension should be larger than 2^J.')
        self.M_padded, self.N_padded = compute_padding(self.M, self.N, self.J)
        self.pad = self.backend.Pad([(self.M_padded - self.M) // 2, (self.M_padded - self.M + 1) // 2, (self.N_padded - self.N) // 2, (self.N_padded - self.N + 1) // 2], [self.M, self.N], pre_pad=self.pre_pad)
        self.unpad = self.backend.unpad

    def create_filters(self):
        filters = filter_bank(self.M_padded, self.N_padded, self.J, self.L)
        self.phi, self.psi = filters['phi'], filters['psi']
    _doc_shape = 'M, N'
    _doc_instantiation_shape = {(True): 'S = Scattering2D(J, (M, N))', (False): 'S = Scattering2D(J)'}
    _doc_param_shape = """shape : tuple of ints
            Spatial support (M, N) of the input
        """
    _doc_attrs_shape = """Psi : dictionary
            Contains the wavelets filters at all resolutions. See
            `filter_bank.filter_bank` for an exact description.
        Phi : dictionary
            Contains the low-pass filters at all resolutions. See
            `filter_bank.filter_bank` for an exact description.
        M_padded, N_padded : int
             Spatial support of the padded input.
        """
    _doc_param_out_type = """out_type : str, optional
            The format of the output of a scattering transform. If set to
            `'list'`, then the output is a list containing each individual
            scattering path with meta information. Otherwise, if set to
            `'array'`, the output is a large array containing the
            concatenation of all scattering coefficients. Defaults to
            `'array'`.
        """
    _doc_attr_out_type = """out_type : str
            The format of the scattering output. See documentation for
            `out_type` parameter above and the documentation for `scattering`.
        """
    _doc_class = """The 2D scattering transform

        The scattering transform computes two wavelet transform
        followed by modulus non-linearity. It can be summarized as

            $S_J x = [S_J^{{(0)}} x, S_J^{{(1)}} x, S_J^{{(2)}} x]$

        where

            $S_J^{{(0)}} x = x \\star \\phi_J$,

            $S_J^{{(1)}} x = [|x \\star \\psi^{{(1)}}_\\lambda| \\star \\phi_J]_\\lambda$, and

            $S_J^{{(2)}} x = [||x \\star \\psi^{{(1)}}_\\lambda| \\star
            \\psi^{{(2)}}_\\mu| \\star \\phi_J]_{{\\lambda, \\mu}}$.

        where $\\star$ denotes the convolution (in space), $\\phi_J$ is a
        lowpass filter, $\\psi^{{(1)}}_\\lambda$ is a family of bandpass filters
        and $\\psi^{{(2)}}_\\mu$ is another family of bandpass filters. Only
        Morlet filters are used in this implementation. Convolutions are
        efficiently performed in the Fourier domain.
        {frontend_paragraph}
        Example
        -------
        ::

            # Set the parameters of the scattering transform.
            J = 3
            M, N = 32, 32

            # Generate a sample signal.
            x = {sample}

            # Define a Scattering2D object.
            {instantiation}

            # Calculate the scattering transform.
            Sx = S.scattering(x)

            # Equivalently, use the alias.
            Sx = S{alias_call}(x)

        Parameters
        ----------
        J : int
            Log-2 of the scattering scale.
        {param_shape}L : int, optional
            Number of angles used for the wavelet transform. Defaults to `8`.
        max_order : int, optional
            The maximum order of scattering coefficients to compute. Must be
            either `1` or `2`. Defaults to `2`.
        pre_pad : boolean, optional
            Controls the padding: if set to False, a symmetric padding is
            applied on the signal. If set to True, the software will assume
            the signal was padded externally. Defaults to `False`.
        backend : object, optional
            Controls the backend which is combined with the frontend.
        {param_out_type}
        Attributes
        ----------
        J : int
            Log-2 of the scattering scale.
        {param_shape}L : int, optional
            Number of angles used for the wavelet transform.
        max_order : int, optional
            The maximum order of scattering coefficients to compute.
            Must be either `1` or `2`.
        pre_pad : boolean
            Controls the padding: if set to False, a symmetric padding is
            applied on the signal. If set to True, the software will assume
            the signal was padded externally.
        {attrs_shape}{attr_out_type}
        Notes
        -----
        The design of the filters is optimized for the value `L = 8`.

        The `pre_pad` flag is particularly useful when cropping bigger images
        because this does not introduce border effects inherent to padding.
        """
    _doc_scattering = """Apply the scattering transform

       Parameters
       ----------
       input : {array}
           An input `{array}` of size `(B, M, N)`.

       Raises
       ------
       RuntimeError
           In the event that the input does not have at least two dimensions,
           or the tensor is not contiguous, or the tensor is not of the
           correct spatial size, padded or not.
       TypeError
           In the event that the input is not of type `{array}`.

       Returns
       -------
       S : {array}
           Scattering transform of the input. If `out_type` is set to
           `'array'` (or if it is not availabel for this frontend), this is
           a{n} `{array}` of shape `(B, C, M1, N1)` where `M1 = M // 2 ** J`
           and `N1 = N // 2 ** J`. The `C` is the number of scattering
           channels calculated. If `out_type` is `'list'`, the output is a
           list of dictionaries, with each dictionary corresponding to a
           scattering coefficient and its meta information. The actual
           coefficient is contained in the `'coef'` key, while other keys hold
           additional information, such as `'j'` (the scale of the filter
           used), and `'theta'` (the angle index of the filter used).
    """

    @classmethod
    def _document(cls):
        instantiation = cls._doc_instantiation_shape[cls._doc_has_shape]
        param_shape = cls._doc_param_shape if cls._doc_has_shape else ''
        attrs_shape = cls._doc_attrs_shape if cls._doc_has_shape else ''
        param_out_type = cls._doc_param_out_type if cls._doc_has_out_type else ''
        attr_out_type = cls._doc_attr_out_type if cls._doc_has_out_type else ''
        cls.__doc__ = ScatteringBase2D._doc_class.format(array=cls._doc_array, frontend_paragraph=cls._doc_frontend_paragraph, alias_name=cls._doc_alias_name, alias_call=cls._doc_alias_call, instantiation=instantiation, param_shape=param_shape, attrs_shape=attrs_shape, param_out_type=param_out_type, attr_out_type=attr_out_type, sample=cls._doc_sample.format(shape=cls._doc_shape))
        cls.scattering.__doc__ = ScatteringBase2D._doc_scattering.format(array=cls._doc_array, n=cls._doc_array_n)


def scattering2d(x, pad, unpad, backend, J, L, phi, psi, max_order, out_type='array'):
    subsample_fourier = backend.subsample_fourier
    modulus = backend.modulus
    fft = backend.fft
    cdgmm = backend.cdgmm
    concatenate = backend.concatenate
    out_S_0, out_S_1, out_S_2 = [], [], []
    U_r = pad(x)
    U_0_c = fft(U_r, 'C2C')
    U_1_c = cdgmm(U_0_c, phi[0])
    U_1_c = subsample_fourier(U_1_c, k=2 ** J)
    S_0 = fft(U_1_c, 'C2R', inverse=True)
    S_0 = unpad(S_0)
    out_S_0.append({'coef': S_0, 'j': (), 'theta': ()})
    for n1 in range(len(psi)):
        j1 = psi[n1]['j']
        theta1 = psi[n1]['theta']
        U_1_c = cdgmm(U_0_c, psi[n1][0])
        if j1 > 0:
            U_1_c = subsample_fourier(U_1_c, k=2 ** j1)
        U_1_c = fft(U_1_c, 'C2C', inverse=True)
        U_1_c = modulus(U_1_c)
        U_1_c = fft(U_1_c, 'C2C')
        S_1_c = cdgmm(U_1_c, phi[j1])
        S_1_c = subsample_fourier(S_1_c, k=2 ** (J - j1))
        S_1_r = fft(S_1_c, 'C2R', inverse=True)
        S_1_r = unpad(S_1_r)
        out_S_1.append({'coef': S_1_r, 'j': (j1,), 'theta': (theta1,)})
        if max_order < 2:
            continue
        for n2 in range(len(psi)):
            j2 = psi[n2]['j']
            theta2 = psi[n2]['theta']
            if j2 <= j1:
                continue
            U_2_c = cdgmm(U_1_c, psi[n2][j1])
            U_2_c = subsample_fourier(U_2_c, k=2 ** (j2 - j1))
            U_2_c = fft(U_2_c, 'C2C', inverse=True)
            U_2_c = modulus(U_2_c)
            U_2_c = fft(U_2_c, 'C2C')
            S_2_c = cdgmm(U_2_c, phi[j2])
            S_2_c = subsample_fourier(S_2_c, k=2 ** (J - j2))
            S_2_r = fft(S_2_c, 'C2R', inverse=True)
            S_2_r = unpad(S_2_r)
            out_S_2.append({'coef': S_2_r, 'j': (j1, j2), 'theta': (theta1, theta2)})
    out_S = []
    out_S.extend(out_S_0)
    out_S.extend(out_S_1)
    out_S.extend(out_S_2)
    if out_type == 'array':
        out_S = concatenate([x['coef'] for x in out_S])
    return out_S


class ScatteringTorch2D(ScatteringTorch, ScatteringBase2D):

    def __init__(self, J, shape, L=8, max_order=2, pre_pad=False, backend='torch', out_type='array'):
        ScatteringTorch.__init__(self)
        ScatteringBase2D.__init__(**locals())
        ScatteringBase2D._instantiate_backend(self, 'kymatio.scattering2d.backend.')
        ScatteringBase2D.build(self)
        ScatteringBase2D.create_filters(self)
        self.register_filters()

    def register_single_filter(self, v, n):
        current_filter = torch.from_numpy(v).unsqueeze(-1)
        self.register_buffer('tensor' + str(n), current_filter)
        return current_filter

    def register_filters(self):
        """ This function run the filterbank function that
            will create the filters as numpy array, and then, it
            saves those arrays as module's buffers."""
        n = 0
        for c, phi in self.phi.items():
            if not isinstance(c, int):
                continue
            self.phi[c] = self.register_single_filter(phi, n)
            n = n + 1
        for j in range(len(self.psi)):
            for k, v in self.psi[j].items():
                if not isinstance(k, int):
                    continue
                self.psi[j][k] = self.register_single_filter(v, n)
                n = n + 1

    def load_single_filter(self, n, buffer_dict):
        return buffer_dict['tensor' + str(n)]

    def load_filters(self):
        """ This function loads filters from the module's buffers """
        buffer_dict = dict(self.named_buffers())
        n = 0
        phis = self.phi
        for c, phi in phis.items():
            if not isinstance(c, int):
                continue
            phis[c] = self.load_single_filter(n, buffer_dict)
            n = n + 1
        psis = self.psi
        for j in range(len(psis)):
            for k, v in psis[j].items():
                if not isinstance(k, int):
                    continue
                psis[j][k] = self.load_single_filter(n, buffer_dict)
                n = n + 1
        return phis, psis

    def scattering(self, input):
        if not torch.is_tensor(input):
            raise TypeError('The input should be a PyTorch Tensor.')
        if len(input.shape) < 2:
            raise RuntimeError('Input tensor must have at least two dimensions.')
        if not input.is_contiguous():
            raise RuntimeError('Tensor must be contiguous.')
        if (input.shape[-1] != self.N or input.shape[-2] != self.M) and not self.pre_pad:
            raise RuntimeError('Tensor must be of spatial size (%i,%i).' % (self.M, self.N))
        if (input.shape[-1] != self.N_padded or input.shape[-2] != self.M_padded) and self.pre_pad:
            raise RuntimeError('Padded tensor must be of spatial size (%i,%i).' % (self.M_padded, self.N_padded))
        if not self.out_type in ('array', 'list'):
            raise RuntimeError("The out_type must be one of 'array' or 'list'.")
        phi, psi = self.load_filters()
        batch_shape = input.shape[:-2]
        signal_shape = input.shape[-2:]
        input = input.reshape((-1,) + signal_shape)
        S = scattering2d(input, self.pad, self.unpad, self.backend, self.J, self.L, phi, psi, self.max_order, self.out_type)
        if self.out_type == 'array':
            scattering_shape = S.shape[-3:]
            S = S.reshape(batch_shape + scattering_shape)
        else:
            scattering_shape = S[0]['coef'].shape[-2:]
            new_shape = batch_shape + scattering_shape
            for x in S:
                x['coef'] = x['coef'].reshape(new_shape)
        return S


def gaussian_3d(M, N, O, sigma, fourier=True):
    """
        Computes a 3D Gaussian filter.

        Parameters
        ----------
        M, N, O : int
            spatial sizes
        sigma : float
            gaussian width parameter
        fourier : boolean
            if true, the Gaussian if computed in Fourier space
	    if false, the Gaussian if computed in signal space

        Returns
        -------
        gaussian : ndarray
            numpy array of size (M, N, O) and type float32 ifftshifted such
            that the origin is at the point [0, 0, 0]
    """
    grid = np.fft.ifftshift(np.mgrid[-M // 2:-M // 2 + M, -N // 2:-N // 2 + N, -O // 2:-O // 2 + O].astype('float32'), axes=(1, 2, 3))
    _sigma = sigma
    if fourier:
        grid[0] *= 2 * np.pi / M
        grid[1] *= 2 * np.pi / N
        grid[2] *= 2 * np.pi / O
        _sigma = 1.0 / sigma
    gaussian = np.exp(-0.5 * (grid ** 2).sum(0) / _sigma ** 2)
    if not fourier:
        gaussian /= (2 * np.pi) ** 1.5 * _sigma ** 3
    return gaussian


def gaussian_filter_bank(M, N, O, J, sigma_0, fourier=True):
    """
        Computes a set of 3D Gaussian filters of scales j = [0, ..., J].

        Parameters
        ----------
        M, N, O : int
            spatial sizes
        J : int
            maximal scale of the wavelets
        sigma_0 : float
            width parameter of father Gaussian filter
        fourier : boolean
            if true, wavelets are computed in Fourier space
	    if false, wavelets are computed in signal space

        Returns
        -------
        gaussians : ndarray
            torch array array of size (J+1, M, N, O, 2) containing the (J+1)
            Gaussian filters.
    """
    gaussians = np.zeros((J + 1, M, N, O), dtype='complex64')
    for j in range(J + 1):
        sigma = sigma_0 * 2 ** j
        gaussians[j, ...] = gaussian_3d(M, N, O, sigma, fourier=fourier)
    return gaussians


def double_factorial(i):
    """Computes the double factorial of an integer."""
    return 1 if i < 1 else np.prod(np.arange(i, 0, -2))


def sqrt(x):
    """
        Compute the square root of an array
        This suppresses any warnings due to invalid input, unless the array is
        real and has negative values. This fixes the erroneous warnings
        introduced by an Intel SVM bug for large single-precision arrays. For
        more information, see:
            https://github.com/numpy/numpy/issues/11448
            https://github.com/ContinuumIO/anaconda-issues/issues/9129
        Parameters
        ----------
        x : numpy array
            An array for which we would like to compute the square root.
        Returns
        -------
        y : numpy array
            The square root of the array.
    """
    if np.isrealobj(x) and (x < 0).any():
        warnings.warn('Negative value encountered in sqrt', RuntimeWarning, stacklevel=1)
    old_settings = np.seterr(invalid='ignore')
    y = np.sqrt(x)
    np.seterr(**old_settings)
    return y


def get_3d_angles(cartesian_grid):
    """
        Given a cartesian grid, computes the spherical coord angles (theta, phi).
        Parameters
        ----------
        cartesian_grid: numpy array
            4D array of shape (3, M, N, O)
        Returns
        -------
        polar: numpy array
            polar angles, shape (M, N, O)
        azimutal: numpy array
            azimutal angles, shape (M, N, O)
    """
    z, y, x = cartesian_grid
    azimuthal = np.arctan2(y, x)
    rxy = sqrt(x ** 2 + y ** 2)
    polar = np.arctan2(z, rxy) + np.pi / 2
    return polar, azimuthal


def solid_harmonic_3d(M, N, O, sigma, l, fourier=True):
    """
        Computes a set of 3D Solid Harmonic Wavelets.
	A solid harmonic wavelet has two integer orders l >= 0 and -l <= m <= l
	In spherical coordinates (r, theta, phi), a solid harmonic wavelet is
	the product of a polynomial Gaussian r^l exp(-0.5 r^2 / sigma^2)
	with a spherical harmonic function Y_{l,m} (theta, phi).

        Parameters
        ----------
        M, N, O : int
            spatial sizes
        sigma : float
            width parameter of the solid harmonic wavelets
        l : int
            first integer order of the wavelets
        fourier : boolean
            if true, wavelets are computed in Fourier space
	    if false, wavelets are computed in signal space

        Returns
        -------
        solid_harm : ndarray, type complex64
            numpy array of size (2l+1, M, N, 0) and type complex64 containing
            the 2l+1 wavelets of order (l , m) with -l <= m <= l.
            It is ifftshifted such that the origin is at the point [., 0, 0, 0]
    """
    solid_harm = np.zeros((2 * l + 1, M, N, O), np.complex64)
    grid = np.fft.ifftshift(np.mgrid[-M // 2:-M // 2 + M, -N // 2:-N // 2 + N, -O // 2:-O // 2 + O].astype('float32'), axes=(1, 2, 3))
    _sigma = sigma
    if fourier:
        grid[0] *= 2 * np.pi / M
        grid[1] *= 2 * np.pi / N
        grid[2] *= 2 * np.pi / O
        _sigma = 1.0 / sigma
    r_square = (grid ** 2).sum(0)
    r_power_l = sqrt(r_square ** l)
    gaussian = np.exp(-0.5 * r_square / _sigma ** 2).astype('complex64')
    if l == 0:
        if fourier:
            return gaussian.reshape((1, M, N, O))
        return gaussian.reshape((1, M, N, O)) / ((2 * np.pi) ** 1.5 * _sigma ** 3)
    polynomial_gaussian = r_power_l * gaussian / _sigma ** l
    polar, azimuthal = get_3d_angles(grid)
    for i_m, m in enumerate(range(-l, l + 1)):
        solid_harm[i_m] = sph_harm(m, l, azimuthal, polar) * polynomial_gaussian
    if l % 2 == 0:
        norm_factor = 1.0 / (2 * np.pi * np.sqrt(l + 0.5) * double_factorial(l + 1))
    else:
        norm_factor = 1.0 / (2 ** (0.5 * (l + 3)) * np.sqrt(np.pi * (2 * l + 1)) * factorial((l + 1) / 2))
    if fourier:
        norm_factor *= (2 * np.pi) ** 1.5 * (-1.0j) ** l
    else:
        norm_factor /= _sigma ** 3
    solid_harm *= norm_factor
    return solid_harm


def solid_harmonic_filter_bank(M, N, O, J, L, sigma_0, fourier=True):
    """
        Computes a set of 3D Solid Harmonic Wavelets of scales j = [0, ..., J]
        and first orders l = [0, ..., L].

        Parameters
        ----------
        M, N, O : int
            spatial sizes
        J : int
            maximal scale of the wavelets
        L : int
            maximal first order of the wavelets
        sigma_0 : float
            width parameter of mother solid harmonic wavelet
        fourier : boolean
            if true, wavelets are computed in Fourier space
	    if false, wavelets are computed in signal space

        Returns
        -------
        filters : list of ndarray
            the element number l of the list is a torch array array of size
            (J+1, 2l+1, M, N, O, 2) containing the (J+1)x(2l+1) wavelets of order l.
    """
    filters = []
    for l in range(L + 1):
        filters_l = np.zeros((J + 1, 2 * l + 1, M, N, O), dtype='complex64')
        for j in range(J + 1):
            sigma = sigma_0 * 2 ** j
            filters_l[j, ...] = solid_harmonic_3d(M, N, O, sigma, l, fourier=fourier)
        filters.append(filters_l)
    return filters


class ScatteringBase3D(ScatteringBase):

    def __init__(self, J, shape, L=3, sigma_0=1, max_order=2, rotation_covariant=True, method='integral', points=None, integral_powers=(0.5, 1.0, 2.0), backend=None):
        super(ScatteringBase3D, self).__init__()
        self.J = J
        self.shape = shape
        self.L = L
        self.sigma_0 = sigma_0
        self.max_order = max_order
        self.rotation_covariant = rotation_covariant
        self.method = method
        self.points = points
        self.integral_powers = integral_powers
        self.backend = backend

    def build(self):
        self.M, self.N, self.O = self.shape

    def create_filters(self):
        self.filters = solid_harmonic_filter_bank(self.M, self.N, self.O, self.J, self.L, self.sigma_0)
        self.gaussian_filters = gaussian_filter_bank(self.M, self.N, self.O, self.J + 1, self.sigma_0)
    _doc_shape = 'M, N, O'
    _doc_class = """The 3D solid harmonic scattering transform

        This class implements solid harmonic scattering on a 3D input image.
        For details see https://arxiv.org/abs/1805.00571.
        {frontend_paragraph}

        Example
        -------
        ::

            # Set the parameters of the scattering transform.
            J = 3
            M, N, O = 32, 32, 32

            # Generate a sample signal.
            x = {sample}

            # Define a HarmonicScattering3D object.
            S = HarmonicScattering3D(J, (M, N, O))

            # Calculate the scattering transform.
            Sx = S.scattering(x)

            # Equivalently, use the alias.
            Sx = S{alias_call}(x)

        Parameters
        ----------
        J: int
            Number of scales.
        shape: tuple of ints
            Shape `(M, N, O)` of the input signal
        L: int, optional
            Number of `l` values. Defaults to `3`.
        sigma_0: float, optional
            Bandwidth of mother wavelet. Defaults to `1`.
        max_order: int, optional
            The maximum order of scattering coefficients to compute. Must be
            either `1` or `2`. Defaults to `2`.
        rotation_covariant: bool, optional
            If set to `True` the first-order moduli take the form:

            $\\sqrt{{\\sum_m (x \\star \\psi_{{j,l,m}})^2)}}$

            if set to `False` the first-order moduli take the form:

            $x \\star \\psi_{{j,l,m}}$

            The second order moduli change analogously. Defaults to `True`.
        method: string, optional
            Specifies the method for obtaining scattering coefficients.
            Currently, only `'integral'` is available. Defaults to `'integral'`.
        integral_powers: array-like
            List of exponents to the power of which moduli are raised before
            integration.
        """
    _doc_scattering = """Apply the scattering transform

       Parameters
       ----------
       input_array: {array}
           Input of size `(batch_size, M, N, O)`.

       Returns
       -------
       output: {array}
           If max_order is `1` it returns a{n} `{array}` with the first-order
           scattering coefficients. If max_order is `2` it returns a{n}
           `{array}` with the first- and second- order scattering
           coefficients, concatenated along the feature axis.
    """

    @classmethod
    def _document(cls):
        cls.__doc__ = ScatteringBase3D._doc_class.format(array=cls._doc_array, frontend_paragraph=cls._doc_frontend_paragraph, alias_name=cls._doc_alias_name, alias_call=cls._doc_alias_call, sample=cls._doc_sample.format(shape=cls._doc_shape))
        cls.scattering.__doc__ = ScatteringBase3D._doc_scattering.format(array=cls._doc_array, n=cls._doc_array_n)


def scattering3d(x, filters, rotation_covariant, L, J, max_order, backend, averaging):
    """
    The forward pass of 3D solid harmonic scattering
    Parameters
    ----------
    input_array: torch tensor
        input of size (batchsize, M, N, O)
    Returns
    -------
    output: tuple | torch tensor
        if max_order is 1 it returns a torch tensor with the
        first order scattering coefficients
        if max_order is 2 it returns a torch tensor with the
        first and second order scattering coefficients,
        concatenated along the feature axis
    """
    fft = backend.fft
    cdgmm3d = backend.cdgmm3d
    modulus = backend.modulus
    modulus_rotation = backend.modulus_rotation
    concatenate = backend.concatenate
    U_0_c = fft(x)
    s_order_1, s_order_2 = [], []
    for l in range(L + 1):
        s_order_1_l, s_order_2_l = [], []
        for j_1 in range(J + 1):
            U_1_m = None
            if rotation_covariant:
                for m in range(len(filters[l][j_1])):
                    U_1_c = cdgmm3d(U_0_c, filters[l][j_1][m])
                    U_1_c = fft(U_1_c, inverse=True)
                    U_1_m = modulus_rotation(U_1_c, U_1_m)
            else:
                U_1_c = cdgmm3d(U_0_c, filters[l][j_1][0])
                U_1_c = fft(U_1_c, inverse=True)
                U_1_m = modulus(U_1_c)
            S_1_l = averaging(U_1_m)
            s_order_1_l.append(S_1_l)
            if max_order > 1:
                U_1_c = fft(U_1_m)
                for j_2 in range(j_1 + 1, J + 1):
                    U_2_m = None
                    if rotation_covariant:
                        for m in range(len(filters[l][j_2])):
                            U_2_c = cdgmm3d(U_1_c, filters[l][j_2][m])
                            U_2_c = fft(U_2_c, inverse=True)
                            U_2_m = modulus_rotation(U_2_c, U_2_m)
                    else:
                        U_2_c = cdgmm3d(U_1_c, filters[l][j_2][0])
                        U_2_c = fft(U_2_c, inverse=True)
                        U_2_m = modulus(U_2_c)
                    S_2_l = averaging(U_2_m)
                    s_order_2_l.append(S_2_l)
        s_order_1.append(s_order_1_l)
        if max_order == 2:
            s_order_2.append(s_order_2_l)
    S = s_order_1
    if max_order == 2:
        S = [(x + y) for x, y in zip(S, s_order_2)]
    S = [x for y in zip(*S) for x in y]
    S = concatenate(S, L)
    return S


class HarmonicScatteringTorch3D(ScatteringTorch, ScatteringBase3D):

    def __init__(self, J, shape, L=3, sigma_0=1, max_order=2, rotation_covariant=True, method='integral', points=None, integral_powers=(0.5, 1.0, 2.0), backend='torch'):
        ScatteringTorch.__init__(self)
        ScatteringBase3D.__init__(self, J, shape, L, sigma_0, max_order, rotation_covariant, method, points, integral_powers, backend)
        self.build()

    def build(self):
        ScatteringBase3D._instantiate_backend(self, 'kymatio.scattering3d.backend.')
        ScatteringBase3D.build(self)
        ScatteringBase3D.create_filters(self)
        self.register_filters()

    def register_filters(self):
        for k in range(len(self.filters)):
            filt = torch.zeros(self.filters[k].shape + (2,))
            filt[..., 0] = torch.from_numpy(self.filters[k].real).reshape(self.filters[k].shape)
            filt[..., 1] = torch.from_numpy(self.filters[k].imag).reshape(self.filters[k].shape)
            self.filters[k] = filt
            self.register_buffer('tensor' + str(k), self.filters[k])
        g = torch.zeros(self.gaussian_filters.shape + (2,))
        g[..., 0] = torch.from_numpy(self.gaussian_filters.real)
        self.gaussian_filters = g
        self.register_buffer('tensor_gaussian_filter', self.gaussian_filters)

    def scattering(self, input_array):
        if not torch.is_tensor(input_array):
            raise TypeError('The input should be a torch.cuda.FloatTensor, a torch.FloatTensor or a torch.DoubleTensor.')
        if input_array.dim() < 3:
            raise RuntimeError('Input tensor must have at least three dimensions.')
        if input_array.shape[-1] != self.O or input_array.shape[-2] != self.N or input_array.shape[-3] != self.M:
            raise RuntimeError('Tensor must be of spatial size (%i, %i, %i).' % (self.M, self.N, self.O))
        input_array = input_array.contiguous()
        batch_shape = input_array.shape[:-3]
        signal_shape = input_array.shape[-3:]
        input_array = input_array.reshape((-1,) + signal_shape)
        x = input_array.new(input_array.shape + (2,)).fill_(0)
        x[..., 0] = input_array
        buffer_dict = dict(self.named_buffers())
        for k in range(len(self.filters)):
            self.filters[k] = buffer_dict['tensor' + str(k)]
        methods = ['integral']
        if not self.method in methods:
            raise ValueError('method must be in {}'.format(methods))
        if self.method == 'integral':
            self.averaging = lambda x: self.backend.compute_integrals(x, self.integral_powers)
        S = scattering3d(x, filters=self.filters, rotation_covariant=self.rotation_covariant, L=self.L, J=self.J, max_order=self.max_order, backend=self.backend, averaging=self.averaging)
        scattering_shape = S.shape[1:]
        S = S.reshape(batch_shape + scattering_shape)
        return S


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Generator,
     lambda: ([], {'num_input_channels': 4, 'num_hidden_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Identity,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Scattering2dCNN,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Scattering2dResNet,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 8, 8])], {}),
     False),
    (View,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_kymatio_kymatio(_paritybench_base):
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

