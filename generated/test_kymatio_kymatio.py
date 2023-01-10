import sys
_module = sys.modules[__name__]
del sys
numpy_benchmark = _module
scattering1d = _module
scattering2d = _module
scattering3d = _module
torch_benchmark = _module
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
jax_backend = _module
numpy_backend = _module
tensorflow_backend = _module
torch_backend = _module
torch_skcuda_backend = _module
caching = _module
datasets = _module
frontend = _module
base_frontend = _module
entry = _module
jax_frontend = _module
keras_frontend = _module
numpy_frontend = _module
sklearn_frontend = _module
tensorflow_frontend = _module
torch_frontend = _module
jax = _module
keras = _module
numpy = _module
torch_backend = _module
torch_skcuda_backend = _module
core = _module
timefrequency_scattering = _module
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
test_jnp_vs_np = _module
test_numpy_backend = _module
test_tensorflow_backend = _module
test_torch_backend = _module
test_correctness = _module
test_filters_scattering1d = _module
test_jax_scattering1d = _module
test_keras_scattering1d = _module
test_numpy_backend_1d = _module
test_numpy_scattering1d = _module
test_tensorflow_backend_1d = _module
test_tensorflow_scattering1d = _module
test_timefrequency_scattering = _module
test_torch_backend_1d = _module
test_torch_scattering1d = _module
test_utils_scattering1d = _module
test_frontend_scattering2d = _module
test_jax_scattering2d = _module
test_keras_scattering2d = _module
test_numpy_backend_2d = _module
test_numpy_scattering2d = _module
test_sklearn_2d = _module
test_tensorflow_backend_2d = _module
test_tensorflow_scattering2d = _module
test_torch_backend_2d = _module
test_torch_scattering2d = _module
test_jax_scattering3d = _module
test_numpy_scattering3d = _module
test_tensorflow_scattering3d = _module
test_torch_scattering3d = _module
test_utils_scattering3d = _module

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


from torch.nn import Linear


from torch.nn import NLLLoss


from torch.nn import LogSoftmax


from torch.nn import Sequential


from torch.optim import Adam


from scipy.io import wavfile


import numpy as np


from sklearn.metrics import confusion_matrix


import matplotlib.pyplot as plt


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


from collections import Counter


import numbers


from warnings import warn


from torch.nn import ReflectionPad2d


from scipy.special import sph_harm


from scipy.special import factorial


import tensorflow as tf


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


class ScatteringTorch(nn.Module):

    def __init__(self):
        super(ScatteringTorch, self).__init__()
        self.frontend_name = 'torch'

    def register_filters(self):
        """ This function should be called after filters are generated,
        saving those arrays as module buffers. """
        raise NotImplementedError

    def forward(self, x):
        """This method is an alias for `scattering`."""
        self.backend.input_checks(x)
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

    def _check_filterbanks(psi1s, psi2s):
        assert all(psi1['xi'] < 0.5 / 2 ** psi1['j'] for psi1 in psi1s)
        psi_generator = itertools.product(psi1s, psi2s)
        condition = lambda psi1_or_2: psi1_or_2[0]['j'] < psi1_or_2[1]['j']
        implication = lambda psi1_or_2: psi1_or_2[0]['xi'] > psi1_or_2[1]['xi']
        assert all(map(implication, filter(condition, psi_generator)))

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


    class _DryBackend:
        __getattr__ = lambda self, attr: lambda *args, **kwargs: None


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


def anden_generator(J, Q, sigma0, r_psi, **unused_kwargs):
    """
    Yields the center frequencies and bandwidths of a filterbank, in compliance
    with the ScatNet package. Center frequencies follow a geometric progression
    of common factor 2**(1/Q) above a certain "elbow frequency" xi_elbow and
    an arithmetic progression of common difference (1/Q) below xi_elbow.

    The corresponding bandwidth sigma is proportional to center frequencies
    for xi>=xi_elbow and are constant (sigma=sigma_min) for xi<xi_elbow.

    The formula for xi_elbow is quite complicated and involves four hyperparameters
    J, Q, r_psi, and sigma0:

    xi_elbow = compute_xi_max(Q) * (sigma0/2**J)/compute_sigma_psi(xi, Q, r_psi)

    where compute_xi_max and compute_sigma_psi are defined elsewhere in this module.

    Intuitively, the role of xi_elbow is to make the filterbank as "wavelet-like"
    as possible (common xi/sigma ratio) while guaranteeing a lower bound on sigma
    (hence an upper bound on time support) and full coverage of the Fourier
    domain between pi/2**J and pi.

    Parameters
    ----------
    J : int
        log-scale of the scattering transform, such that wavelets of both
        filterbanks have a maximal support that is proportional to 2**J.

    Q : int
        number of wavelets per octave in the geometric progression portion of
        the filterbank.

    r_psi : float in (0, 1)
        Should be >0 and <1. The higher the r_psi, the greater the sigmas.
        Adjacent wavelets peak at 1 and meet at r_psi.

    sigma0 : float
        Should be >0. The minimum bandwidth is sigma0/2**J.
    """
    xi = compute_xi_max(Q)
    sigma = compute_sigma_psi(xi, Q, r=r_psi)
    sigma_min = sigma0 / 2 ** J
    if sigma <= sigma_min:
        xi = sigma
    else:
        yield xi, sigma
        while sigma > sigma_min * math.pow(2, 1 / Q):
            xi /= math.pow(2, 1 / Q)
            sigma /= math.pow(2, 1 / Q)
            yield xi, sigma
    elbow_xi = xi
    for q in range(Q - 1):
        xi -= 1 / Q * elbow_xi
        yield xi, sigma_min


def compute_border_indices(log2_T, J, i0, i1):
    """
    Computes border indices at all scales which correspond to the original
    signal boundaries after padding.

    At the finest resolution,
    original_signal = padded_signal[..., i0:i1].
    This function finds the integers i0, i1 for all temporal subsamplings
    by 2**J, being conservative on the indices.

    Maximal subsampling is by `2**log2_T` if `T=None`, else by
    `2**max(log2_T, J)`. We compute indices up to latter to be sure.

    Parameters
    ----------
    log2_T : int
        Maximal subsampling by low-pass filtering is `2**log2_T`.
    J : int
        Maximal subsampling by band-pass filtering is `2**J`.
    i0 : int
        start index of the original signal at the finest resolution
    i1 : int
        end index (excluded) of the original signal at the finest resolution

    Returns
    -------
    ind_start, ind_end: dictionaries with keys in [0, ..., log2_T] such that the
        original signal is in padded_signal[ind_start[j]:ind_end[j]]
        after subsampling by 2**j
    """
    ind_start = {(0): i0}
    ind_end = {(0): i1}
    for j in range(1, max(log2_T, J) + 1):
        ind_start[j] = ind_start[j - 1] // 2 + ind_start[j - 1] % 2
        ind_end[j] = ind_end[j - 1] // 2 + ind_end[j - 1] % 2
    return ind_start, ind_end


def compute_padding(N, N_input):
    """
    Computes the padding to be added on the left and on the right
    of the signal.

    It should hold that N >= N_input

    Parameters
    ----------
    N : int
        support of the padded signal
    N_input : int
        support of the unpadded signal

    Returns
    -------
    pad_left: amount to pad on the left ("beginning" of the support)
    pad_right: amount to pad on the right ("end" of the support)
    """
    if N < N_input:
        raise ValueError('Padding support should be larger than the original' + 'signal size!')
    to_add = N - N_input
    pad_left = to_add // 2
    pad_right = to_add - pad_left
    if max(pad_left, pad_right) >= N_input:
        raise ValueError('Too large padding value, will lead to NaN errors')
    return pad_left, pad_right


def compute_temporal_support(h_f, criterion_amplitude=0.001):
    """
    Computes the (half) temporal support of a family of centered,
    symmetric filters h provided in the Fourier domain

    This function computes the support N which is the smallest integer
    such that for all signals x and all filters h,

    \\| x \\conv h - x \\conv h_{[-N, N]} \\|_{\\infty} \\leq \\epsilon
        \\| x \\|_{\\infty}  (1)

    where 0<\\epsilon<1 is an acceptable error, and h_{[-N, N]} denotes the
    filter h whose support is restricted in the interval [-N, N]

    The resulting value N used to pad the signals to avoid boundary effects
    and numerical errors.

    If the support is too small, no such N might exist.
    In this case, N is defined as the half of the support of h, and a
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
        N = np.min(np.where(np.max(l1_residual, axis=0) <= criterion_amplitude)[0]) + 1
    else:
        N = half_support
        warnings.warn('Signal support is too small to avoid border effects')
    return N


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


def morlet_1d(N, xi, sigma):
    """
    Computes the Fourier transform of a Morlet or Gauss filter.
    A Morlet filter is the sum of a Gabor filter and a low-pass filter
    to ensure that the sum has exactly zero mean in the temporal domain.
    It is defined by the following formula in time:
    psi(t) = g_{sigma}(t) (e^{i xi t} - kappa)
    where g_{sigma} is a Gaussian envelope, xi is a frequency and kappa is
    a corrective term which ensures that psi has a null average.
    If xi is None, the definition becomes: phi(t) = g_{sigma}(t)
    Parameters
    ----------
    N : int
        size of the temporal support
    xi : float or None
        center frequency in (0, 1]
    sigma : float
        bandwidth parameter
    Returns
    -------
    filter_f : array_like
        numpy array of size (N,) containing the Fourier transform of the Morlet
        filter at the frequencies given by np.fft.fftfreq(N).
    """
    P = min(adaptive_choice_P(sigma), 5)
    freqs = np.arange((1 - P) * N, P * N, dtype=float) / float(N)
    if P == 1:
        freqs_low = np.fft.fftfreq(N)
    elif P > 1:
        freqs_low = freqs
    low_pass_f = np.exp(-freqs_low ** 2 / (2 * sigma ** 2))
    low_pass_f = low_pass_f.reshape(2 * P - 1, -1).mean(axis=0)
    if xi:
        gabor_f = np.exp(-(freqs - xi) ** 2 / (2 * sigma ** 2))
        gabor_f = gabor_f.reshape(2 * P - 1, -1).mean(axis=0)
        kappa = gabor_f[0] / low_pass_f[0]
        filter_f = gabor_f - kappa * low_pass_f
    else:
        filter_f = low_pass_f
    filter_f /= np.abs(ifft(filter_f)).sum()
    return filter_f


def gauss_1d(N, sigma):
    return morlet_1d(N, xi=None, sigma=sigma)


def parse_T(T, J, N_input, T_alias='T'):
    """
    Parses T in Scattering1D base frontend.
    Parses T and F in TimeFrequencyScattering base frontend.

    Parameters
    ----------
    T : None, string, integer 0, or float >= 1
        user-provided T value
    J : int
        user-provided J value
    N_input : int
        input size
    T_alias : string
        Used for printing error messages.
        Typically 'T' (default) or 'F' (in TimeFrequencyScattering).

    Returns
    -------
    T_parsed : int
        (2**J) if T is None, zero, or 'global'; user-provided T otherwise
    average : string
        'global' if T is 'global'; False if T is zero; 'local' otherwise
    """
    if T is None:
        return 2 ** J, 'local'
    elif T == 'global':
        return 2 ** J, 'global'
    elif T > N_input:
        raise ValueError("The support {} of the low-pass filter cannot exceed input length (got {} > {}). For large averaging size, consider passing {}='global'.".format(T_alias, T, N_input, T_alias))
    elif T == 0:
        return 2 ** J, False
    elif T < 1:
        raise ValueError('{} must be ==0 or >=1 (got {})'.format(T_alias, T))
    else:
        return T, 'local'


def scattering1d(U_0, backend, filters, oversampling, average_local):
    """
    Main function implementing the 1-D scattering transform.

    Parameters
    ----------
    U_0 : Tensor
        an backend-compatible array of size `(B, 1, N)` where `B` is batch size
        and `N` is the padded signal length.
    backend : module
        Kymatio module which matches the type of U_0.
    psi1 : list
        a list of dictionaries, expressing wavelet band-pass filters in the
        Fourier domain for the first layer of the scattering transform.
        Each `psi1[n1]` is a dictionary with keys:
            * `j`: int, subsampling factor
            * `xi`: float, center frequency
            * `sigma`: float, bandwidth
            * `levels`: list, values taken by the wavelet in the Fourier domain
                        at different levels of detail.
        Each psi1[n]['levels'][level] is an array with size N/2**level.
    psi2 : dictionary
        Same as psi1, but for the second layer of the scattering transform.
    phi : dictionary
        a dictionary expressing the low-pass filter in the Fourier domain.
        Keys:
        * `j`: int, subsampling factor (also known as log_T)
        * `xi`: float, center frequency (=0 by convention)
        * `sigma`: float, bandwidth
        * 'levels': list, values taken by the lowpass in the Fourier domain
                    at different levels of detail.
    oversampling : int >=0
        Yields scattering coefficients with a temporal stride equal to
        max(1, 2**(log2_T-oversampling)). Hence, raising oversampling by
        one unit halves the stride, until reaching a stride of 1.
    average_local : boolean
        whether to locally average the result by means of a low-pass filter phi.
    """
    U_0_hat = backend.rfft(U_0)
    phi = filters[0]
    log2_T = phi['j']
    k0 = max(log2_T - oversampling, 0)
    if average_local:
        S_0_c = backend.cdgmm(U_0_hat, phi['levels'][0])
        S_0_hat = backend.subsample_fourier(S_0_c, 2 ** k0)
        S_0_r = backend.irfft(S_0_hat)
        yield {'coef': S_0_r, 'j': (), 'n': ()}
    else:
        yield {'coef': U_0, 'j': (), 'n': ()}
    psi1 = filters[1]
    for n1 in range(len(psi1)):
        j1 = psi1[n1]['j']
        sub1_adj = min(j1, log2_T) if average_local else j1
        k1 = max(sub1_adj - oversampling, 0)
        U_1_c = backend.cdgmm(U_0_hat, psi1[n1]['levels'][0])
        U_1_hat = backend.subsample_fourier(U_1_c, 2 ** k1)
        U_1_c = backend.ifft(U_1_hat)
        U_1_m = backend.modulus(U_1_c)
        if average_local or len(filters) > 2:
            U_1_hat = backend.rfft(U_1_m)
        if average_local:
            k1_J = max(log2_T - k1 - oversampling, 0)
            S_1_c = backend.cdgmm(U_1_hat, phi['levels'][k1])
            S_1_hat = backend.subsample_fourier(S_1_c, 2 ** k1_J)
            S_1_r = backend.irfft(S_1_hat)
            yield {'coef': S_1_r, 'j': (j1,), 'n': (n1,)}
        else:
            yield {'coef': U_1_m, 'j': (j1,), 'n': (n1,)}
        if len(filters) > 2:
            psi2 = filters[2]
            for n2 in range(len(psi2)):
                j2 = psi2[n2]['j']
                if j2 > j1:
                    sub2_adj = min(j2, log2_T) if average_local else j2
                    k2 = max(sub2_adj - k1 - oversampling, 0)
                    U_2_c = backend.cdgmm(U_1_hat, psi2[n2]['levels'][k1])
                    U_2_hat = backend.subsample_fourier(U_2_c, 2 ** k2)
                    U_2_c = backend.ifft(U_2_hat)
                    U_2_m = backend.modulus(U_2_c)
                    if average_local:
                        U_2_hat = backend.rfft(U_2_m)
                        k2_log2_T = max(log2_T - k2 - k1 - oversampling, 0)
                        S_2_c = backend.cdgmm(U_2_hat, phi['levels'][k1 + k2])
                        S_2_hat = backend.subsample_fourier(S_2_c, 2 ** k2_log2_T)
                        S_2_r = backend.irfft(S_2_hat)
                        yield {'coef': S_2_r, 'j': (j1, j2), 'n': (n1, n2)}
                    else:
                        yield {'coef': U_2_m, 'j': (j1, j2), 'n': (n1, n2)}


def get_max_dyadic_subsampling(xi, sigma, alpha, **unused_kwargs):
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
        The larger alpha, the smaller the error.

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


def scattering_filter_factory(N, J, Q, T, filterbank):
    """
    Builds in Fourier the Morlet filters used for the scattering transform.

    Each single filter is provided as a dictionary with the following keys:
    * 'xi': normalized center frequency, where 0.5 corresponds to Nyquist.
    * 'sigma': normalized bandwidth in the Fourier.
    * 'j': log2 of downsampling factor after filtering. j=0 means no downsampling,
        j=1 means downsampling by one half, etc.
    * 'levels': list of NumPy arrays containing the filter at various levels
        of downsampling. levels[0] is at full resolution, levels[1] at half
        resolution, etc.

    Parameters
    ----------
    N : int
        padded length of the input signal. Corresponds to self._N_padded for the
        scattering object.
    J : int
        log-scale of the scattering transform, such that wavelets of both
        filterbanks have a maximal support that is proportional to 2**J.
    Q : tuple
        number of wavelets per octave at the first and second order
        Q = (Q1, Q2). Q1 and Q2 are both int >= 1.
    T : int
        temporal support of low-pass filter, controlling amount of imposed
        time-shift invariance and maximum subsampling
    filterbank : tuple (callable filterbank_fn, dict filterbank_kwargs)
        filterbank_fn should take J and Q as positional arguments and
        **filterbank_kwargs as optional keyword arguments.
        Corresponds to the self.filterbank property of the scattering object.
        As of v0.3, only anden_generator is supported as filterbank_fn.

    Returns
    -------
    phi_f, psi1_f, psi2_f ... : dictionaries
        phi_f corresponds to the low-pass filter and psi1_f, psi2_f, to the
        wavelet filterbanks at layers 1 and 2 respectively.
        See above for a description of the dictionary structure.
    """
    filterbank_fn, filterbank_kwargs = filterbank
    max_j = 0
    previous_J = 0
    log2_T = math.floor(math.log2(T))
    psis_f = []
    for Q_layer in Q:
        psi_f = []
        for xi, sigma in filterbank_fn(J, Q_layer, **filterbank_kwargs):
            psi_levels = [morlet_1d(N, xi, sigma)]
            j = get_max_dyadic_subsampling(xi, sigma, **filterbank_kwargs)
            for level in range(1, min(previous_J, j, log2_T)):
                psi_level = psi_levels[0].reshape(2 ** level, -1).mean(axis=0)
                psi_levels.append(psi_level)
            psi_f.append({'levels': psi_levels, 'xi': xi, 'sigma': sigma, 'j': j})
            max_j = max(j, max_j)
        previous_J = max_j
        psis_f.append(psi_f)
    sigma_low = filterbank_kwargs['sigma0'] / T
    phi_levels = [gauss_1d(N, sigma_low)]
    for level in range(1, max(previous_J, 1 + log2_T)):
        phi_level = phi_levels[0].reshape(2 ** level, -1).mean(axis=0)
        phi_levels.append(phi_level)
    phi_f = {'levels': phi_levels, 'xi': 0, 'sigma': sigma_low, 'j': log2_T, 'N': N}
    return tuple([phi_f] + psis_f)


class ScatteringBase1D(ScatteringBase):

    def __init__(self, J, shape, Q=1, T=None, max_order=2, oversampling=0, out_type='array', backend=None):
        super(ScatteringBase1D, self).__init__()
        self.J = J
        self.shape = shape
        self.Q = Q
        self.T = T
        self.max_order = max_order
        self.oversampling = oversampling
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
        if np.any(np.array(self.Q) < 1):
            raise ValueError('Q must always be >= 1, got {}'.format(self.Q))
        if isinstance(self.Q, int):
            self.Q = self.Q, 1
        elif isinstance(self.Q, tuple):
            if len(self.Q) == 1:
                self.Q = self.Q + (1,)
            elif len(self.Q) < 1 or len(self.Q) > 2:
                raise NotImplementedError('Q must be an integer, 1-tuple or 2-tuple. Scattering transforms beyond order 2 are not implemented.')
        else:
            raise ValueError('Q must be an integer or a tuple')
        if isinstance(self.shape, numbers.Integral):
            self.shape = self.shape,
        elif isinstance(self.shape, tuple):
            if len(self.shape) > 1:
                raise ValueError('If shape is specified as a tuple, it must have exactly one element')
        else:
            raise ValueError('shape must be an integer or a 1-tuple')
        N_input = self.shape[0]
        self.T, self.average = parse_T(self.T, self.J, N_input)
        self.log2_T = math.floor(math.log2(self.T))
        phi_f = gauss_1d(N_input, self.sigma0 / self.T)
        min_to_pad = 3 * compute_temporal_support(phi_f.reshape(1, -1), criterion_amplitude=0.001)
        J_max_support = int(np.floor(np.log2(3 * N_input - 2)))
        J_pad = min(int(np.ceil(np.log2(N_input + 2 * min_to_pad))), J_max_support)
        self._N_padded = 2 ** J_pad
        self.pad_left, self.pad_right = compute_padding(self._N_padded, N_input)
        self.ind_start, self.ind_end = compute_border_indices(self.log2_T, self.J, self.pad_left, self.pad_left + N_input)

    def create_filters(self):
        self.phi_f, self.psi1_f, self.psi2_f = scattering_filter_factory(self._N_padded, self.J, self.Q, self.T, self.filterbank)
        ScatteringBase._check_filterbanks(self.psi1_f, self.psi2_f)

    def scattering(self, x):
        ScatteringBase1D._check_runtime_args(self)
        ScatteringBase1D._check_input(self, x)
        x_shape = self.backend.shape(x)
        batch_shape, signal_shape = x_shape[:-1], x_shape[-1:]
        x = self.backend.reshape_input(x, signal_shape)
        U_0 = self.backend.pad(x, pad_left=self.pad_left, pad_right=self.pad_right)
        filters = [self.phi_f, self.psi1_f, self.psi2_f][:1 + self.max_order]
        S_gen = scattering1d(U_0, self.backend, filters, self.oversampling, self.average == 'local')
        if self.out_type in ['array', 'list']:
            S = list()
        elif self.out_type == 'dict':
            S = dict()
        for path in S_gen:
            path['order'] = len(path['n'])
            if self.average == 'local':
                res = max(self.log2_T - self.oversampling, 0)
            elif path['order'] > 0:
                res = max(path['j'][-1] - self.oversampling, 0)
            else:
                res = 0
            if self.average == 'global':
                path['coef'] = self.backend.average_global(path['coef'])
            else:
                path['coef'] = self.backend.unpad(path['coef'], self.ind_start[res], self.ind_end[res])
            path['coef'] = self.backend.reshape_output(path['coef'], batch_shape, n_kept_dims=1)
            if self.out_type in ['array', 'list']:
                S.append(path)
            elif self.out_type == 'dict':
                S[path['n']] = path['coef']
        if self.out_type == 'dict':
            return S
        S.sort(key=lambda path: (path['order'], path['n']))
        if self.out_type == 'array':
            S = self.backend.concatenate([path['coef'] for path in S], dim=-2)
        return S

    def meta(self):
        """Get metadata on the transform.

        This information specifies the content of each scattering coefficient,
        which order, which frequencies, which filters were used, and so on.

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
        backend = self._DryBackend()
        filters = [self.phi_f, self.psi1_f, self.psi2_f][:1 + self.max_order]
        S_gen = scattering1d(None, backend, filters, self.oversampling, average_local=False)
        S = sorted(list(S_gen), key=lambda path: (len(path['n']), path['n']))
        meta = dict(order=np.array([len(path['n']) for path in S]))
        meta['key'] = [path['n'] for path in S]
        meta['n'] = np.stack([np.append(path['n'], (np.nan,) * (self.max_order - len(path['n']))) for path in S])
        filterbanks = (self.psi1_f, self.psi2_f)[:self.max_order]
        for key in ['xi', 'sigma', 'j']:
            meta[key] = meta['n'] * np.nan
            for order, filterbank in enumerate(filterbanks):
                for n, psi in enumerate(filterbank):
                    meta[key][meta['n'][:, order] == n, order] = psi[key]
        return meta

    def output_size(self, detail=False):
        """Number of scattering coefficients.

        Parameters
        ----------
        detail : boolean, optional
            Whether to aggregate the count (detail=False, default) across
            orders or to break it down by scattering depth (layers 0, 1, and 2).

        Returns
        ------
        size : int or tuple
            If `detail=False` (default), total number of scattering coefficients.
            Else, number of coefficients at zeroth, first, and second order.
        """
        if detail:
            return tuple(Counter(self.meta()['order']).values())
        return len(self.meta()['key'])

    def _check_runtime_args(self):
        if not self.out_type in ('array', 'dict', 'list'):
            raise ValueError("out_type must be one of 'array', 'dict', or 'list'. Got: {}".format(self.out_type))
        if not self.average and self.out_type == 'array':
            raise ValueError("Cannot convert to out_type='array' with T=0. Please set out_type to 'dict' or 'list'.")
        if self.oversampling < 0:
            raise ValueError('oversampling must be nonnegative. Got: {}'.format(self.oversampling))
        if not isinstance(self.oversampling, numbers.Integral):
            raise ValueError('oversampling must be integer. Got: {}'.format(self.oversampling))

    def _check_input(self, x):
        if len(x.shape) < 1:
            raise ValueError('Input tensor x should have at least one axis, got {}'.format(len(x.shape)))

    @property
    def J_pad(self):
        warn('The attribute J_pad is deprecated and will be removed in v0.4. Measure len(self.phi_f[0]) for the padded length (previously 2**J_pad) or access shape[0] for the unpadded length (previously N).', DeprecationWarning)
        return int(np.log2(self._N_padded))

    @property
    def N(self):
        warn('The attribute N is deprecated and will be removed in v0.4. Measure len(self.phi_f[0]) for the padded length (previously 2**J_pad) or access shape[0] for the unpadded length (previously N).', DeprecationWarning)
        return int(self.shape[0])

    @property
    def filterbank(self):
        filterbank_kwargs = {'alpha': self.alpha, 'r_psi': self.r_psi, 'sigma0': self.sigma0}
        return anden_generator, filterbank_kwargs
    _doc_shape = 'N'
    _doc_instantiation_shape = {(True): 'S = Scattering1D(J, N, Q)', (False): 'S = Scattering1D(J, Q)'}
    _doc_param_shape = """shape : int
            The length of the input signals.
        """
    _doc_attrs_shape = """pad_left : int
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
            creation. Defaults to `True`. Deprecated in v0.3 in favour of `T`
            and will  be removed in v0.4. Replace `average=False` by `T=0` and
            set `T>1` or leave `T=None` for `average=True` (default).
        """
    _doc_attr_average = """average : boolean
            Controls whether the output should be averaged (the standard
            scattering transform) or not (resulting in wavelet modulus
            coefficients). Note that to obtain unaveraged output, the
            `vectorize` flag must be set to `False` or `out_type` must be set
            to `'list'`. Deprecated in favor of `T`. For more details,
            see the documentation for `scattering`.
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
        Given an input `{array}` `x` of shape `(B, N)`, where `B` is the
        number of signals to transform (the batch size) and `N` is the length
        of the signal, we compute its scattering transform by passing it to
        the `scattering` method (or calling the alias `{alias_name}`). Note
        that `B` can be one, in which case it may be omitted, giving an input
        of shape `(N,)`.

        Example
        -------
        ::

            # Set the parameters of the scattering transform.
            J = 6
            N = 2 ** 13
            Q = 8

            # Generate a sample signal.
            x = {sample}

            # Define a Scattering1D object.
            {instantiation}

            # Calculate the scattering transform.
            Sx = S.scattering(x)

            # Equivalently, use the alias.
            Sx = S{alias_call}(x)

        Above, the length of the signal is :math:`N = 2^{{13}} = 8192`, while the
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
        {param_shape}Q : int or tuple
            By default, Q (int) is the number of wavelets per octave for the first
            order and that for the second order has one wavelet per octave. This
            default value can be modified by passing Q as a tuple with two values,
            i.e. Q = (Q1, Q2), where Q1 and Q2 are the number of wavelets per
            octave for the first and second order, respectively.
        T : int
            temporal support of low-pass filter, controlling amount of imposed
            time-shift invariance and maximum subsampling
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
        T : int
            temporal support of low-pass filter, controlling amount of imposed
            time-shift invariance and maximum subsampling
        {attrs_shape}max_order : int
            The maximum scattering order of the transform.
        {attr_average}oversampling : int
            The number of powers of two to oversample the output compared to
            the default subsampling rate determined from the filters.
        {attr_vectorize}"""
    _doc_scattering = """Apply the scattering transform

       Given an input `{array}` of size `(B, N)`, where `B` is the batch
       size (it can be potentially an integer or a shape) and `N` is the length
       of the individual signals, this function computes its scattering
       transform. If the `vectorize` flag is set to `True` (or if it is not
       available in this frontend), the output is in the form of a `{array}`
       or size `(B, C, N1)`, where `N1` is the signal length after subsampling
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
           An input `{array}` of size `(B, N)`.

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


class ScatteringTorch1D(ScatteringTorch, ScatteringBase1D):

    def __init__(self, J, shape, Q=1, T=None, max_order=2, oversampling=0, out_type='array', backend='torch'):
        ScatteringTorch.__init__(self)
        ScatteringBase1D.__init__(self, J, shape, Q, T, max_order, oversampling, out_type, backend)
        ScatteringBase1D._instantiate_backend(self, 'kymatio.scattering1d.backend.')
        ScatteringBase1D.build(self)
        ScatteringBase1D.create_filters(self)
        self.register_filters()

    def register_filters(self):
        """This function run the filterbank function that
        will create the filters as numpy array, and then, it
        saves those arrays as module's buffers."""
        n = 0
        for level in range(len(self.phi_f['levels'])):
            self.phi_f['levels'][level] = torch.from_numpy(self.phi_f['levels'][level]).float().view(-1, 1)
            self.register_buffer('tensor' + str(n), self.phi_f['levels'][level])
            n += 1
        for psi_f in self.psi1_f:
            for level in range(len(psi_f['levels'])):
                psi_f['levels'][level] = torch.from_numpy(psi_f['levels'][level]).float().view(-1, 1)
                self.register_buffer('tensor' + str(n), psi_f['levels'][level])
                n += 1
        for psi_f in self.psi2_f:
            for level in range(len(psi_f['levels'])):
                psi_f['levels'][level] = torch.from_numpy(psi_f['levels'][level]).float().view(-1, 1)
                self.register_buffer('tensor' + str(n), psi_f['levels'][level])
                n += 1
        return n

    def load_filters(self):
        """This function loads filters from the module's buffer"""
        buffer_dict = dict(self.named_buffers())
        n = 0
        for level in range(len(self.phi_f['levels'])):
            self.phi_f['levels'][level] = buffer_dict['tensor' + str(n)]
            n += 1
        for psi_f in self.psi1_f:
            for level in range(len(psi_f['levels'])):
                psi_f['levels'][level] = buffer_dict['tensor' + str(n)]
                n += 1
        for psi_f in self.psi2_f:
            for level in range(len(psi_f['levels'])):
                psi_f['levels'][level] = buffer_dict['tensor' + str(n)]
                n += 1
        return n

    def scattering(self, x):
        self.load_filters()
        return super().scattering(x)


def frequency_scattering(X, backend, filters_fr, oversampling_fr, average_local_fr, spinned):
    """
    Parameters
    ----------
    X : dictionary with keys 'coef' and 'n1_max'
        if spinned, X['coef']=Y_2 is complex-valued and indexed by
        (batch, n1, time[j2]), for some fixed n2 and variable n1 s.t. j1 < j2.
        else, X['coef']=S_1 is real-valued and indexed by
        (batch, n1, time[log2_T]) for variable n1 < len(psi1_f)
    backend : module
    filters_fr : [phi, psis] list where
        * phi is a dictionary describing the low-pass filter of width F, used
          to average S1 and S2 in frequency if and only if average_local_fr.
        * psis is a list of dictionaries, each describing a low-pass or band-pass
          filter indexed by n_fr. The first element, n_fr=0, corresponds
          to a low-pass filter of width 2**J_fr and satisfies xi=0, i.e, spin=0.
          Other elements, such that n_fr>0, correspond to "spinned" band-pass
          filter, where spin denotes the sign of the center frequency xi.
    oversampling_fr : int >=0
        Yields joint time-frequency scattering coefficients with a frequential
        stride max(1, 2**(log2_F-oversampling_fr)). Raising oversampling_fr
        by one halves the stride, until reaching a stride of 1.
    average_local_fr : boolean
        whether the result will be locally averaged with phi after this function
    spinned: boolean
        if True (complex input), yields Y_fr for all n_fr
        else (real input), yields Y_fr for only those n_fr s.t. spin>=0

    Yields
    ------
    Y_2_fr{n2,n_fr}(t, n1) = (Y_2*psi_{n_fr})(t[j2], n1[j_fr]),
        conv. over n1, broadcast over t, n1 zero-padded up to N_fr

    Definitions
    -----------
    U_0(t) = x(t)
    S_0(t) = (x * phi)(t)
    U_1{n1}(t) = |x * psi_{n1}|(t)
    S_1(n1, t) = (U_1 * phi)(t), conv. over t, broadcast over n1
    Y_1_fr{n_fr}(t, n1) = (S_1*psi_{n_fr})(t[log2_T], n1[n_fr]),
        conv. over n1, broadcast over t, n1 zero-padded up to N_fr
    Y_2{n2}(t, n1) = (U_1 * psi_{n2})(t[j2], n2),
        conv. over t, broadcast over n1
    Y_2_fr{n2,n_fr}(t, n1) = (Y_2*psi_{n_fr})(t[j2], n1[j_fr]),
        conv. over n1, broadcast over t, n1 zero-padded up to N_fr
    """
    phi, psis = filters_fr
    log2_F = phi['j']
    X_T = backend.swap_time_frequency(X['coef'])
    pad_right = phi['N'] - X['n1_max']
    X_pad = backend.pad_frequency(X_T, pad_right)
    if spinned:
        X_hat = backend.cfft(X_pad)
    else:
        X_hat = backend.rfft(X_pad)
        psis = filter(lambda psi: psi['xi'] >= 0, psis)
    for n_fr, psi in enumerate(psis):
        j_fr = psi['j']
        spin = np.sign(psi['xi'])
        sub_fr_adj = min(j_fr, log2_F) if average_local_fr else j_fr
        k_fr = max(sub_fr_adj - oversampling_fr, 0)
        Y_fr_hat = backend.cdgmm(X_hat, psi['levels'][0])
        Y_fr_sub = backend.subsample_fourier(Y_fr_hat, 2 ** k_fr)
        Y_fr = backend.ifft(Y_fr_sub)
        Y_fr = backend.swap_time_frequency(Y_fr)
        yield {**X, 'coef': Y_fr, 'n': X['n'][1:] + (n_fr,), 'j_fr': (j_fr,), 'n_fr': (n_fr,), 'n1_stride': 2 ** j_fr, 'spin': spin}


def time_scattering_widthfirst(U_0, backend, filters, oversampling, average_local):
    """
    Parameters
    ----------
    U_0 : array indexed by (batch, time)
    backend : module
    filters : [phi, psi1, psi2] list of dictionaries. same as scattering1d
    oversampling : int >=0
        Yields scattering coefficients with a temporal stride equal to
        max(1, 2**(log2_T-oversampling)). Hence, raising oversampling by
        one unit halves the stride, until reaching a stride of 1.
    average_local : boolean
        whether to locally average the result by means of a low-pass filter phi.

    Yields
    ------
    if average_local:
        * S_0 indexed by (batch, time[log2_T])
    else:
        * U_0 indexed by (batch, time)
    * S_1 indexed by (batch, n1, time[log2_T])
    for n2 < len(psi2):
        * Y_2{n2} indexed by (batch, n1, time[j1]) and n1 s.t. j1 < j2

    Definitions
    -----------
    U_0(t) = x(t)
    S_0(t) = (x * phi)(t)
    U_1{n1}(t) = |x * psi_{n1}|(t)
    S_1(n1, t) = (|x * psi_{n1}| * phi)(t), conv. over t, broadcast over n1
    Y_2{n2}(n1, t) = (U_1 * psi_{n2})(n1, t), conv. over t, broadcast over n1
    """
    U_0_hat = backend.rfft(U_0)
    phi = filters[0]
    log2_T = phi['j']
    if average_local:
        k0 = max(log2_T - oversampling, 0)
        S_0_c = backend.cdgmm(U_0_hat, phi['levels'][0])
        S_0_hat = backend.subsample_fourier(S_0_c, 2 ** k0)
        S_0_r = backend.irfft(S_0_hat)
        yield {'coef': S_0_r, 'j': (), 'n': ()}
    else:
        yield {'coef': U_0, 'j': (), 'n': ()}
    psi1 = filters[1]
    U_1_hats = []
    S_1_list = []
    for n1 in range(len(psi1)):
        j1 = psi1[n1]['j']
        sub1_adj = min(j1, log2_T) if average_local else j1
        k1 = max(sub1_adj - oversampling, 0)
        U_1_c = backend.cdgmm(U_0_hat, psi1[n1]['levels'][0])
        U_1_hat = backend.subsample_fourier(U_1_c, 2 ** k1)
        U_1_c = backend.ifft(U_1_hat)
        U_1_m = backend.modulus(U_1_c)
        U_1_hat = backend.rfft(U_1_m)
        U_1_hats.append({'coef': U_1_hat, 'j': (j1,), 'n': (n1,)})
        k1_J = max(log2_T - k1 - oversampling, 0)
        S_1_c = backend.cdgmm(U_1_hat, phi['levels'][k1])
        S_1_hat = backend.subsample_fourier(S_1_c, 2 ** k1_J)
        S_1_r = backend.irfft(S_1_hat)
        S_1_list.append(S_1_r)
    S_1 = backend.concatenate(S_1_list)
    yield {'coef': S_1, 'j': (-1,), 'n': (-1,), 'n1_max': len(S_1_list)}
    psi2 = filters[2]
    for n2 in range(len(psi2)):
        j2 = psi2[n2]['j']
        Y_2_list = []
        for U_1_hat in U_1_hats:
            j1 = U_1_hat['j'][0]
            sub1_adj = min(j1, log2_T) if average_local else j1
            k1 = max(sub1_adj - oversampling, 0)
            if j2 > j1:
                sub2_adj = min(j2, log2_T) if average_local else j2
                k2 = max(sub2_adj - k1 - oversampling, 0)
                U_2_c = backend.cdgmm(U_1_hat['coef'], psi2[n2]['levels'][k1])
                U_2_hat = backend.subsample_fourier(U_2_c, 2 ** k2)
                U_2_c = backend.ifft(U_2_hat)
                Y_2_list.append(U_2_c)
        if len(Y_2_list) > 0:
            Y_2 = backend.concatenate(Y_2_list)
            yield {'coef': Y_2, 'j': (-1, j2), 'n': (-1, n2), 'n1_max': len(Y_2_list)}


def joint_timefrequency_scattering(U_0, backend, filters, oversampling, average_local, filters_fr, oversampling_fr, average_local_fr):
    """
    Parameters
    ----------
    U_0 : array indexed by (batch, time)
    backend : module
    filters : [phi, psi1, psi2] list of dictionaries. same as scattering1d
    oversampling : int >=0
     Yields scattering coefficients with a temporal stride equal to
     max(1, 2**(log2_T-oversampling)). Hence, raising oversampling by
     one unit halves the stride, until reaching a stride of 1.

    average_local : boolean
     whether to locally average the result by means of a low-pass filter phi.
    filters_fr : [phi, psis] list where
     * phi is a dictionary describing the low-pass filter of width F, used
       to average S1 and S2 in frequency if and only if average_local_fr.
     * psis is a list of dictionaries, each describing a low-pass or band-pass
       filter indexed by n_fr. The first element, n_fr=0, corresponds
       to a low-pass filter of width 2**J_fr and satisfies xi=0, i.e, spin=0.
       Other elements, such that n_fr>0, correspond to "spinned" band-pass
       filter, where spin denotes the sign of the center frequency xi.
    oversampling_fr : int >=0
     Yields joint time-frequency scattering coefficients with a frequential
     stride of max(1, 2**(log2_F-oversampling_fr)). Raising oversampling_fr
     by one halves the stride, until reaching a stride of 1.
    average_local_fr : boolean
     whether the result will be locally averaged with phi after this function

    Yields
    ------
    # Zeroth order
    if average_local:
     * S_0 indexed by (batch, time[log2_T])
    else:
     * U_0 indexed by (batch, time)

    # First order
    for n_fr < len(filters_fr):
     * Y_1_fr indexed by (batch, n1[n_fr], time[log2_T]), complex-valued,
         where n1 has been zero-padded to size N_fr before convolution

    # Second order
    for n2 < len(psi2):
     for n_fr < len(filters_fr):
         * Y_2_fr indexed by (batch, n1[n_fr], time[n2]), complex-valued,
             where n1 has been zero-padded to size N_fr before convolution

    Definitions
    -----------
    U_0(t) = x(t)
    S_0(t) = (x * phi)(t)
    U_1{n1}(t) = |x * psi_{n1}|(t)
    S_1(n1, t) = (U_1 * phi)(t), conv. over t, broadcast over n1
    Y_1_fr{n_fr}(t, n1) = (S_1*psi_{n_fr})(t[log2_T], n1[n_fr]),
     conv. over n1, broadcast over t, n1 zero-padded up to N_fr
    Y_2{n2}(t, n1) = (U_1 * psi_{n2})(t[j2], n2),
     conv. over t, broadcast over n1
    Y_2_fr{n2,n_fr}(t, n1) = (Y_2*psi_{n_fr})(t[j2], n1[j_fr]),
     conv. over n1, broadcast over t, n1 zero-padded up to N_fr
    """
    time_gen = time_scattering_widthfirst(U_0, backend, filters, oversampling, average_local)
    yield next(time_gen)
    S_1 = next(time_gen)
    yield from frequency_scattering(S_1, backend, filters_fr, oversampling_fr, average_local_fr, spinned=False)
    for Y_2 in time_gen:
        yield from frequency_scattering(Y_2, backend, filters_fr, oversampling_fr, average_local_fr, spinned=True)


def frequency_averaging(U_2, backend, phi_fr_f, oversampling_fr, average_fr):
    """
    Parameters
    ----------
    U_2 : dictionary with keys 'coef' and 'j_fr', typically returned by
        frequency_scattering or time_averaging
    backend : module
    phi_fr_f : dictionary. Frequential low-pass filter in Fourier domain.
    oversampling_fr : int >=0
        Yields joint time-frequency scattering coefficients with a frequential
        stride max(1, 2**(log2_F-oversampling_fr)). Raising oversampling_fr
        by one halves the stride, until reaching a stride of 1.
    average_fr : string
        Either 'local', 'global', or False.

    Returns
    -------
    if average_fr == 'local':
        * S_2{n2,n_fr} indexed by (batch, n1[log2_F], time[j2])
    if average_fr == 'global':
        * S_2{n2,n_fr} indexed by (batch, 1, time[j2])
    if average_fr == False:
        * S_2{n2,n_fr} indexed by (batch, n1[j_fr], time[j2])

    Definitions
    -----------
    U_0(t) = x(t)
    S_0(t) = (x * phi)(t)
    U_1{n1}(t) = |x * psi_{n1}|(t)
    S_1(n1, t) = (U_1 * phi)(t), conv. over t, broadcast over n1
    Y_1_fr{n_fr}(t, n1) = (S_1*psi_{n_fr})(t[log2_T], n1[n_fr]),
        conv. over n1, broadcast over t, n1 zero-padded up to N_fr
    Y_2{n2}(t, n1) = (U_1 * psi_{n2})(t[j2], n2),
        conv. over t, broadcast over n1
    Y_2_fr{n2,n_fr}(t, n1) = (Y_2*psi_{n_fr})(t[j2], n1[j_fr]),
        conv. over n1, broadcast over t, n1 zero-padded up to N_fr
    U_2{n2,n_fr}(t, n1) = |Y_2_fr{n2,n_fr}|(t[j2], n1[j_fr])
    """
    if average_fr:
        log2_F = phi_fr_f['j']
        U_2_T = backend.swap_time_frequency(U_2['coef'])
        if average_fr == 'global':
            S_2_T = backend.average_global(U_2_T)
            n1_stride = U_2['n1_max']
        elif average_fr == 'local':
            k_in = min(U_2['j_fr'][-1], log2_F)
            k_J = max(log2_F - k_in - oversampling_fr, 0)
            U_hat = backend.rfft(U_2_T)
            S_c = backend.cdgmm(U_hat, phi_fr_f['levels'][k_in])
            S_hat = backend.subsample_fourier(S_c, 2 ** k_J)
            S_2_T = backend.irfft(S_hat)
            n1_stride = 2 ** max(log2_F - oversampling_fr, 0)
        S_2 = backend.swap_time_frequency(S_2_T)
        return {**U_2, 'coef': S_2, 'n1_stride': n1_stride}
    elif not average_fr:
        n1_stride = 2 ** max(U_2['j_fr'][-1] - oversampling_fr, 0)
        return {**U_2, 'n1_stride': n1_stride}


def time_averaging(U_2, backend, phi_f, oversampling):
    """
    Parameters
    ----------
    U_2 : dictionary with keys 'coef' and 'j', typically returned by
        frequency_scattering
    backend : module
    phi_f : dictionary. Temporal low-pass filter in Fourier domain,
        same as scattering1d
    oversampling : int >=0
        Yields scattering coefficients with a temporal stride equal to
        max(1, 2**(log2_T-oversampling)). Hence, raising oversampling by
        one unit halves the stride, until reaching a stride of 1.

    Returns
    -------
    S_2{n2,n_fr} indexed by (batch, n1[n_fr], time[log2_T])

    Definitions
    -----------
    U_0(t) = x(t)
    S_0(t) = (x * phi)(t)
    U_1{n1}(t) = |x * psi_{n1}|(t)
    S_1(n1, t) = (U_1 * phi)(t), conv. over t, broadcast over n1
    Y_1_fr{n_fr}(t, n1) = (S_1*psi_{n_fr})(t[log2_T], n1[n_fr]),
        conv. over n1, broadcast over t, n1 zero-padded up to N_fr
    Y_2{n2}(t, n1) = (U_1 * psi_{n2})(t[j2], n2),
        conv. over t, broadcast over n1
    Y_2_fr{n2,n_fr}(t, n1) = (Y_2*psi_{n_fr})(t[j2], n1[j_fr]),
        conv. over n1, broadcast over t, n1 zero-padded up to N_fr
    U_2{n2,n_fr}(t, n1) = |Y_2_fr{n2,n_fr}|(t, n1)
    """
    log2_T = phi_f['j']
    k_in = U_2['j'][-1]
    k_J = max(log2_T - k_in - oversampling, 0)
    U_hat = backend.rfft(U_2['coef'])
    S_c = backend.cdgmm(U_hat, phi_f['levels'][k_in])
    S_hat = backend.subsample_fourier(S_c, 2 ** k_J)
    S_2 = backend.irfft(S_hat)
    return {**U_2, 'coef': S_2}


def time_formatting(path, backend):
    if path['coef'] is None:
        coef_list = [None] * (1 + path['n1_max'] // path['n1_stride'])
    else:
        coef_list = backend.split_frequency_axis(path['coef'])
    for i, n1 in enumerate(range(0, path['n1_max'], path['n1_stride'])):
        split_path = {**path, 'coef': coef_list[i], 'order': len(path['n']) - 1}
        split_path['n'] = (n1,) + split_path['n']
        del split_path['n1_max']
        del split_path['n1_stride']
        yield split_path


def jtfs_average_and_format(U_gen, backend, phi_f, oversampling, average, phi_fr_f, oversampling_fr, average_fr, out_type, format):
    path = next(U_gen)
    log2_T = phi_f['j']
    if average == 'global':
        path['coef'] = backend.average_global(path['coef'])
    yield {**path, 'order': 0}
    for path in U_gen:
        path['coef'] = backend.modulus(path['coef'])
        if average == 'global':
            path['coef'] = backend.average_global(path['coef'])
        elif average == 'local' and len(path['n']) > 1:
            path = time_averaging(path, backend, phi_f, oversampling)
        if average_fr and not path['spin'] == 0:
            path = frequency_averaging(path, backend, phi_fr_f, oversampling_fr, average_fr)
        if not (out_type == 'array' and format == 'joint'):
            path['coef'] = backend.unpad_frequency(path['coef'], path['n1_max'], path['n1_stride'])
        if format == 'joint':
            yield {**path, 'order': len(path['n'])}
        elif format == 'time':
            yield from time_formatting(path, backend)


def spin(filterbank_fn, filterbank_kwargs):

    def spinned_fn(J, Q, **kwargs):
        yield from filterbank_fn(J, Q, **kwargs)
        for xi, sigma in filterbank_fn(J, Q, **kwargs):
            yield -xi, sigma
    return spinned_fn, filterbank_kwargs


class TimeFrequencyScatteringBase(ScatteringBase1D):

    def __init__(self, *, J, J_fr, shape, Q, T=None, oversampling=0, Q_fr=1, F=None, oversampling_fr=0, out_type='array', format='joint', backend=None):
        max_order = 2
        super(TimeFrequencyScatteringBase, self).__init__(J, shape, Q, T, max_order, oversampling, out_type, backend)
        self.J_fr = J_fr
        self.Q_fr = Q_fr
        self.F = F
        self.oversampling_fr = oversampling_fr
        self.format = format

    def build(self):
        super(TimeFrequencyScatteringBase, self).build()
        super(TimeFrequencyScatteringBase, self).create_filters()
        if np.any(np.array(self.Q_fr) < 1):
            raise ValueError('Q_fr must be >= 1, got {}'.format(self.Q_fr))
        if isinstance(self.Q_fr, int):
            self.Q_fr = self.Q_fr,
        elif isinstance(self.Q_fr, tuple):
            if len(self.Q_fr) != 1:
                raise NotImplementedError('Q_fr must be an integer or 1-tuple. Time-frequency scattering beyond order 2 is not implemented.')
        else:
            raise ValueError('Q_fr must be an integer or 1-tuple.')
        N_input_fr = len(self.psi1_f)
        self.F, self.average_fr = parse_T(self.F, self.J_fr, N_input_fr, T_alias='F')
        self.log2_F = math.floor(math.log2(self.F))
        min_to_pad_fr = 8 * min(self.F, 2 ** self.J_fr)
        K_fr = max(self.J_fr - self.oversampling_fr, 0)
        N_padded_fr_subsampled = (N_input_fr + min_to_pad_fr) // 2 ** K_fr
        self._N_padded_fr = N_padded_fr_subsampled * 2 ** K_fr

    def create_filters(self):
        phi0_fr_f, = scattering_filter_factory(self._N_padded_fr, self.J_fr, (), self.F, self.filterbank_fr)
        phi1_fr_f, psis_fr_f = scattering_filter_factory(self._N_padded_fr, self.J_fr, self.Q_fr, 2 ** self.J_fr, self.filterbank_fr)
        self.filters_fr = phi0_fr_f, [phi1_fr_f] + psis_fr_f
        assert all(abs(psi1['xi']) < 0.5 / 2 ** psi1['j'] for psi1 in psis_fr_f)

    def scattering(self, x):
        TimeFrequencyScatteringBase._check_runtime_args(self)
        TimeFrequencyScatteringBase._check_input(self, x)
        x_shape = self.backend.shape(x)
        batch_shape, signal_shape = x_shape[:-1], x_shape[-1:]
        x = self.backend.reshape_input(x, signal_shape)
        U_0 = self.backend.pad(x, pad_left=self.pad_left, pad_right=self.pad_right)
        filters = [self.phi_f, self.psi1_f, self.psi2_f]
        U_gen = joint_timefrequency_scattering(U_0, self.backend, filters, self.oversampling, self.average == 'local', self.filters_fr, self.oversampling_fr, self.average_fr == 'local')
        S_gen = jtfs_average_and_format(U_gen, self.backend, self.phi_f, self.oversampling, self.average, self.filters_fr[0], self.oversampling_fr, self.average_fr, self.out_type, self.format)
        path = next(S_gen)
        if not self.average == 'global':
            res = self.log2_T if self.average else 0
            path['coef'] = self.backend.unpad(path['coef'], self.ind_start[res], self.ind_end[res])
        path['coef'] = self.backend.reshape_output(path['coef'], batch_shape, n_kept_dims=1)
        S = [path]
        for path in S_gen:
            if not self.average == 'global':
                if not self.average and len(path['n']) > 1:
                    res = max(path['j'][-1] - self.oversampling, 0)
                else:
                    res = max(self.log2_T - self.oversampling, 0)
                path['coef'] = self.backend.unpad(path['coef'], self.ind_start[res], self.ind_end[res])
            path['coef'] = self.backend.reshape_output(path['coef'], batch_shape, n_kept_dims=1 + (self.format == 'joint'))
            S.append(path)
        if self.format == 'joint' and self.out_type == 'array':
            S = S[1:]
            return self.backend.concatenate([path['coef'] for path in S], dim=-3)
        elif self.format == 'time' and self.out_type == 'array':
            return self.backend.concatenate([path['coef'] for path in S], dim=-2)
        elif self.out_type == 'dict':
            return {path['n']: path['coef'] for path in S}
        elif self.out_type == 'list':
            return S

    def meta(self):
        filters = [self.phi_f, self.psi1_f, self.psi2_f]
        U_gen = joint_timefrequency_scattering(None, self._DryBackend(), filters, self.oversampling, self.average == 'local', self.filters_fr, self.oversampling_fr, self.average_fr == 'local')
        S_gen = jtfs_average_and_format(U_gen, self._DryBackend(), self.phi_f, self.oversampling, self.average, self.filters_fr[0], self.oversampling_fr, self.average_fr, self.out_type, self.format)
        S = sorted(list(S_gen), key=lambda path: (len(path['n']), path['n']))
        meta = dict(key=[path['n'] for path in S], n=[], n_fr=[], order=[])
        for path in S:
            if len(path['n']) == 0:
                if not (self.format == 'joint' and self.out_type == 'array'):
                    meta['n'].append([np.nan, np.nan])
                    meta['n_fr'].append(np.nan)
                    meta['order'].append(0)
            else:
                if len(path['n']) == 1:
                    n1_range = range(0, path['n1_max'], path['n1_stride'])
                    meta['n'].append([n1_range, np.nan])
                elif len(path['n']) == 2 and self.format == 'joint':
                    n1_range = range(0, path['n1_max'], path['n1_stride'])
                    meta['n'].append([n1_range, path['n'][0]])
                elif len(path['n']) == 2 and self.format == 'time':
                    meta['n'].append([path['n'][0], np.nan])
                elif len(path['n']) == 3 and self.format == 'time':
                    meta['n'].append(path['n'][:2])
                meta['n_fr'].append(path['n_fr'][0])
                meta['order'].append(len(path['n']) - (self.format == 'time'))
        meta['n'] = np.array(meta['n'], dtype=object)
        meta['n_fr'] = np.array(meta['n_fr'])
        meta['order'] = np.array(meta['order'])
        for key in ['xi', 'sigma', 'j']:
            meta[key] = np.zeros((meta['n_fr'].shape[0], 2)) * np.nan
            for order, filterbank in enumerate(filters[1:]):
                for n, psi in enumerate(filterbank):
                    meta[key][meta['n'][:, order] == n, order] = psi[key]
            meta[key + '_fr'] = meta['n_fr'] * np.nan
            for n_fr, psi_fr in enumerate(self.filters_fr[1]):
                meta[key + '_fr'][meta['n_fr'] == n_fr] = psi_fr[key]
        meta['spin'] = np.sign(meta['xi_fr'])
        return meta

    def _check_runtime_args(self):
        super(TimeFrequencyScatteringBase, self)._check_runtime_args()
        if self.format == 'joint':
            if not self.average_fr and self.out_type == 'array':
                raise ValueError("Cannot convert to format='joint' with out_type='array' and F=0. Either set format='time', out_type='dict', or out_type='list'.")
        if self.oversampling_fr < 0:
            raise ValueError('oversampling_fr must be nonnegative. Got: {}'.format(self.oversampling_fr))
        if not isinstance(self.oversampling_fr, numbers.Integral):
            raise ValueError('oversampling_fr must be integer. Got: {}'.format(self.oversampling_fr))
        if self.format not in ['time', 'joint']:
            raise ValueError("format must be 'time' or 'joint'. Got: {}".format(self.format))

    @property
    def filterbank_fr(self):
        filterbank_kwargs = {'alpha': self.alpha, 'r_psi': self.r_psi, 'sigma0': self.sigma0}
        return spin(anden_generator, filterbank_kwargs)


class TimeFrequencyScatteringTorch(ScatteringTorch1D, TimeFrequencyScatteringBase):

    def __init__(self, *, J, J_fr, shape, Q, T=None, oversampling=0, Q_fr=1, F=None, oversampling_fr=0, out_type='array', format='joint', backend='torch'):
        ScatteringTorch.__init__(self)
        TimeFrequencyScatteringBase.__init__(self, J=J, J_fr=J_fr, shape=shape, Q=Q, T=T, oversampling=oversampling, Q_fr=Q_fr, F=F, oversampling_fr=oversampling_fr, out_type=out_type, format=format, backend=backend)
        ScatteringBase1D._instantiate_backend(self, 'kymatio.scattering1d.backend.')
        TimeFrequencyScatteringBase.build(self)
        TimeFrequencyScatteringBase.create_filters(self)
        self.register_filters()

    def register_filters(self):
        n = super(TimeFrequencyScatteringTorch, self).register_filters()
        for level in range(len(self.filters_fr[0]['levels'])):
            self.filters_fr[0]['levels'][level] = torch.from_numpy(self.filters_fr[0]['levels'][level]).float().view(-1, 1)
            self.register_buffer('tensor' + str(n), self.filters_fr[0]['levels'][level])
            n += 1
        for psi_f in self.filters_fr[1]:
            for level in range(len(psi_f['levels'])):
                psi_f['levels'][level] = torch.from_numpy(psi_f['levels'][level]).float().view(-1, 1)
                self.register_buffer('tensor' + str(n), psi_f['levels'][level])
                n += 1

    def load_filters(self):
        buffer_dict = dict(self.named_buffers())
        n = super(TimeFrequencyScatteringTorch, self).load_filters()
        for level in range(len(self.filters_fr[0]['levels'])):
            self.filters_fr[0]['levels'][level] = buffer_dict['tensor' + str(n)]
            n += 1
        for psi_f in self.filters_fr[1]:
            for level in range(len(psi_f['levels'])):
                psi_f['levels'][level] = buffer_dict['tensor' + str(n)]
                n += 1


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
            psi = {'levels': [], 'j': j, 'theta': theta}
            psi_signal = morlet_2d(M, N, 0.8 * 2 ** j, (int(L - L / 2 - 1) - theta) * np.pi / L, 3.0 / 4.0 * np.pi / 2 ** j, 4.0 / L)
            psi_signal_fourier = np.real(fft2(psi_signal))
            psi_levels = []
            for res in range(min(j + 1, max(J - 1, 1))):
                psi_levels.append(periodize_filter_fft(psi_signal_fourier, res))
            psi['levels'] = psi_levels
            filters['psi'].append(psi)
    phi_signal = gabor_2d(M, N, 0.8 * 2 ** (J - 1), 0, 0)
    phi_signal_fourier = np.real(fft2(phi_signal))
    filters['phi'] = {'levels': [], 'j': J}
    for res in range(J):
        filters['phi']['levels'].append(periodize_filter_fft(phi_signal_fourier, res))
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
        M, N = self.shape
        if 2 ** self.J > M or 2 ** self.J > N:
            raise RuntimeError('The smallest dimension should be larger than 2^J.')
        self._M_padded, self._N_padded = compute_padding(M, N, self.J)
        if not self.pre_pad:
            self.pad = self.backend.Pad([(self._M_padded - M) // 2, (self._M_padded - M + 1) // 2, (self._N_padded - N) // 2, (self._N_padded - N + 1) // 2], [M, N])
        else:
            self.pad = lambda x: x
        self.unpad = self.backend.unpad

    def create_filters(self):
        filters = filter_bank(self._M_padded, self._N_padded, self.J, self.L)
        self.phi, self.psi = filters['phi'], filters['psi']

    def scattering(self, x):
        """ This function should call the functional scattering."""
        raise NotImplementedError

    @property
    def M(self):
        warn('The attribute M is deprecated and will be removed in v0.4. Replace by shape[0].', DeprecationWarning)
        return int(self.shape[0])

    @property
    def N(self):
        warn('The attribute N is deprecated and will be removed in v0.4. Replace by shape[1].', DeprecationWarning)
        return int(self.shape[1])
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
    rfft = backend.rfft
    ifft = backend.ifft
    irfft = backend.irfft
    cdgmm = backend.cdgmm
    concatenate = backend.concatenate
    out_S_0, out_S_1, out_S_2 = [], [], []
    U_r = pad(x)
    U_0_c = rfft(U_r)
    U_1_c = cdgmm(U_0_c, phi['levels'][0])
    U_1_c = subsample_fourier(U_1_c, k=2 ** J)
    S_0 = irfft(U_1_c)
    S_0 = unpad(S_0)
    out_S_0.append({'coef': S_0, 'j': (), 'n': (), 'theta': ()})
    for n1 in range(len(psi)):
        j1 = psi[n1]['j']
        theta1 = psi[n1]['theta']
        U_1_c = cdgmm(U_0_c, psi[n1]['levels'][0])
        if j1 > 0:
            U_1_c = subsample_fourier(U_1_c, k=2 ** j1)
        U_1_c = ifft(U_1_c)
        U_1_c = modulus(U_1_c)
        U_1_c = rfft(U_1_c)
        S_1_c = cdgmm(U_1_c, phi['levels'][j1])
        S_1_c = subsample_fourier(S_1_c, k=2 ** (J - j1))
        S_1_r = irfft(S_1_c)
        S_1_r = unpad(S_1_r)
        out_S_1.append({'coef': S_1_r, 'j': (j1,), 'n': (n1,), 'theta': (theta1,)})
        if max_order < 2:
            continue
        for n2 in range(len(psi)):
            j2 = psi[n2]['j']
            theta2 = psi[n2]['theta']
            if j2 <= j1:
                continue
            U_2_c = cdgmm(U_1_c, psi[n2]['levels'][j1])
            U_2_c = subsample_fourier(U_2_c, k=2 ** (j2 - j1))
            U_2_c = ifft(U_2_c)
            U_2_c = modulus(U_2_c)
            U_2_c = rfft(U_2_c)
            S_2_c = cdgmm(U_2_c, phi['levels'][j2])
            S_2_c = subsample_fourier(S_2_c, k=2 ** (J - j2))
            S_2_r = irfft(S_2_c)
            S_2_r = unpad(S_2_r)
            out_S_2.append({'coef': S_2_r, 'j': (j1, j2), 'n': (n1, n2), 'theta': (theta1, theta2)})
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
        if pre_pad:
            self.pad = lambda x: x.reshape(x.shape + (1,))
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
        for phi_level in self.phi['levels']:
            self.register_single_filter(phi_level, n)
            n = n + 1
        for psi in self.psi:
            for psi_level in psi['levels']:
                self.register_single_filter(psi_level, n)
                n = n + 1

    def load_single_filter(self, n, buffer_dict):
        return buffer_dict['tensor' + str(n)]

    def load_filters(self):
        """ This function loads filters from the module's buffers """
        buffer_dict = dict(self.named_buffers())
        n = 0
        phis = {k: v for k, v in self.phi.items() if k != 'levels'}
        phis['levels'] = []
        for phi_level in self.phi['levels']:
            phis['levels'].append(self.load_single_filter(n, buffer_dict))
            n = n + 1
        psis = [{} for _ in range(len(self.psi))]
        for j in range(len(self.psi)):
            psis[j] = {k: v for k, v in self.psi[j].items() if k != 'levels'}
            psis[j]['levels'] = []
            for psi_level in self.psi[j]['levels']:
                psis[j]['levels'].append(self.load_single_filter(n, buffer_dict))
                n = n + 1
        return phis, psis

    def scattering(self, input):
        if not torch.is_tensor(input):
            raise TypeError('The input should be a PyTorch Tensor.')
        if len(input.shape) < 2:
            raise RuntimeError('Input tensor must have at least two dimensions.')
        if not input.is_contiguous():
            raise RuntimeError('Tensor must be contiguous.')
        if (input.shape[-1] != self.shape[-1] or input.shape[-2] != self.shape[-2]) and not self.pre_pad:
            raise RuntimeError('Tensor must be of spatial size (%i,%i).' % (self.shape[0], self.shape[1]))
        if (input.shape[-1] != self._N_padded or input.shape[-2] != self._M_padded) and self.pre_pad:
            raise RuntimeError('Padded tensor must be of spatial size (%i,%i).' % (self._M_padded, self._N_padded))
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

    def scattering(self, x):
        """ This function should call the functional scattering."""
        raise NotImplementedError
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
    rfft = backend.rfft
    ifft = backend.ifft
    cdgmm3d = backend.cdgmm3d
    modulus = backend.modulus
    modulus_rotation = backend.modulus_rotation
    concatenate = backend.concatenate
    U_0_c = rfft(x)
    s_order_1, s_order_2 = [], []
    for l in range(L + 1):
        s_order_1_l, s_order_2_l = [], []
        for j_1 in range(J + 1):
            U_1_m = None
            if rotation_covariant:
                for m in range(len(filters[l][j_1])):
                    U_1_c = cdgmm3d(U_0_c, filters[l][j_1][m])
                    U_1_c = ifft(U_1_c)
                    U_1_m = modulus_rotation(U_1_c, U_1_m)
            else:
                U_1_c = cdgmm3d(U_0_c, filters[l][j_1][0])
                U_1_c = ifft(U_1_c)
                U_1_m = modulus(U_1_c)
            S_1_l = averaging(U_1_m)
            s_order_1_l.append(S_1_l)
            if max_order > 1:
                U_1_c = rfft(U_1_m)
                for j_2 in range(j_1 + 1, J + 1):
                    U_2_m = None
                    if rotation_covariant:
                        for m in range(len(filters[l][j_2])):
                            U_2_c = cdgmm3d(U_1_c, filters[l][j_2][m])
                            U_2_c = ifft(U_2_c)
                            U_2_m = modulus_rotation(U_2_c, U_2_m)
                    else:
                        U_2_c = cdgmm3d(U_1_c, filters[l][j_2][0])
                        U_2_c = ifft(U_2_c)
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
        input_array = input_array.reshape((-1,) + signal_shape + (1,))
        buffer_dict = dict(self.named_buffers())
        for k in range(len(self.filters)):
            self.filters[k] = buffer_dict['tensor' + str(k)]
        methods = ['integral']
        if not self.method in methods:
            raise ValueError('method must be in {}'.format(methods))
        if self.method == 'integral':
            self.averaging = lambda x: self.backend.compute_integrals(x, self.integral_powers)
        S = scattering3d(input_array, filters=self.filters, rotation_covariant=self.rotation_covariant, L=self.L, J=self.J, max_order=self.max_order, backend=self.backend, averaging=self.averaging)
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

