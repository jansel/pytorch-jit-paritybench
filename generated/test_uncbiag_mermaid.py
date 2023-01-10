import sys
_module = sys.modules[__name__]
del sys
apply_map = _module
atlas_reg = _module
invert_map = _module
mm_reg = _module
to_be_converted_to_tests = _module
benchmarkConv = _module
benchmarkSTN = _module
plotDebugResults = _module
testAdaptiveMultiGaussian = _module
testDiffSmooth = _module
testFFTAdaptiveGaussian = _module
testFFTGaussianSmoothing = _module
testGaussianFourierSmootherBackprop = _module
testInterpolation = _module
testOMTMask = _module
testRK = _module
testSTN = _module
testSTN1D = _module
testSTN2D = _module
testSTN3D = _module
testSTN_ND = _module
testSVF = _module
testSVFMap = _module
testSmoother = _module
test_with_excel = _module
tst_interp = _module
sphinx_rtd_theme = _module
conf = _module
mermaid = _module
config_parser = _module
custom_optimizers = _module
custom_pytorch_extensions = _module
custom_pytorch_extensions_module_version = _module
data_loader = _module
data_manager = _module
data_pool = _module
data_utils = _module
data_wrapper = _module
deep_loss = _module
deep_networks = _module
deep_smoothers = _module
example_generation = _module
fileio = _module
finite_differences = _module
finite_differences_multi_channel = _module
fixwarnings = _module
forward_models = _module
forward_models_wrap = _module
image_manipulations = _module
image_sampling = _module
libraries = _module
functions = _module
map_scale_utils = _module
stn_nd = _module
modules = _module
asym_conv = _module
stn_nd = _module
load_default_settings = _module
metrics = _module
model_evaluation = _module
model_factory = _module
module_parameters = _module
multiscale_optimizer = _module
noisy_convolution = _module
ode_int = _module
optimizer_data_loaders = _module
registration_networks = _module
regularizer_factory = _module
res_recorder = _module
rungekutta_integrators = _module
similarity_helper_omt = _module
similarity_measure_factory = _module
simple_interface = _module
smoother_factory = _module
spline_interpolation = _module
torchdiffeq = _module
_impl = _module
adams = _module
adjoint = _module
dopri5 = _module
fixed_adams = _module
fixed_grid = _module
interp = _module
misc = _module
odeint = _module
rk_common = _module
solvers = _module
tsit5 = _module
utils = _module
viewers = _module
visualize_registration_results = _module
mermaid_apps = _module
create_synthetic_regularization_test_cases = _module
extract_slices_from_3d_data_set = _module
normalize_image_intensities = _module
mermaid_demos = _module
example_2d_synth = _module
example_custom_registration = _module
example_minimal_registration_without_simple_interface = _module
example_simple_interface = _module
example_step_by_step_registration = _module
rdmm_synth_data_generation = _module
combine_shape = _module
context = _module
create_circle = _module
create_ellipse = _module
create_poly = _module
create_rect = _module
create_triangle = _module
demo_for_generation = _module
initial = _module
moving_shape = _module
shape = _module
utils_for_general = _module
utils_for_regularizer = _module
mermaid_experiments = _module
boxplot_across_stages = _module
check_klein_overlaps = _module
command_line_execution_tools = _module
compute_validation_results = _module
det_jac_as_tabular = _module
experiment_utils = _module
extra_validation_for_synthetic_test_cases = _module
extract_energies_from_logs = _module
generic_experiment_driver = _module
multi_stage_smoother_learning = _module
synth_parameter_sweep_plot = _module
visualize_multi_stage = _module
mermaid_settings = _module
json_viewer = _module
generate_label_overlapping_plot = _module
affine_and_histogram_eq = _module
setup = _module
test_finite_differences = _module
test_inverse_map_issue = _module
test_module_parameters = _module
test_registration_algorithms = _module
test_similarity_helper_omt = _module
test_stn_cpu = _module
test_stn_gpu = _module
version = _module

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


from torch.autograd import Variable


import matplotlib.pyplot as plt


from torch.nn.parameter import Parameter


from time import time


import numpy as np


import torch.nn.functional as F


import time


from torch.autograd import Function


from torch.autograd import gradcheck


from torch.autograd.gradcheck import *


from torch.autograd.gradcheck import _differentiable_outputs


from torch.autograd.gradcheck import _as_tuple


import torch.optim as optim


import torch.nn as nn


import matplotlib.image as img


import math


from functools import reduce


from torch.optim import Optimizer


from math import isinf


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from torchvision import transforms


from torchvision import utils


import random


from abc import ABCMeta


from abc import abstractmethod


import copy


from scipy import ndimage as nd


from torch.nn import Module


from torch.nn.modules import Module


from torch.nn.modules.module import Module


from functools import partial


from collections import defaultdict


from torch.nn.modules.utils import _single


from torch.nn.modules.utils import _pair


from torch.nn.modules.utils import _triple


import collections


from math import floor


from numpy import log


from numpy import shape as numpy_shape


import torch.nn


import numpy.testing as npt


import warnings


import abc


import torch.nn.init as init


import matplotlib as matplt


import scipy.ndimage as ndimage


import scipy.stats as sstats


import scipy.io as sio


from scipy import stats


import matplotlib


from matplotlib import rcParams


def AdaptVal(x):
    """ adapt float32/16, gpu/cpu, float 16 is not recommended to use for it is not stable"""
    if USE_CUDA:
        if not USE_FLOAT16:
            return x
        else:
            return x.half()
    else:
        return x


class SobelFilter(object):

    def __init__(self):
        dx = AdaptVal(torch.Tensor([[[-1.0, -3.0, -1.0], [-3.0, -6.0, -3.0], [-1.0, -3.0, -1.0]], [[0.0, 0.0, 0.0], [0.0, 0, 0.0], [0.0, 0.0, 0.0]], [[1.0, 3.0, 1.0], [3.0, 6.0, 3.0], [1.0, 3.0, 1.0]]])).view(1, 1, 3, 3, 3)
        dy = AdaptVal(torch.Tensor([[[1.0, 3.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -3.0, -1.0]], [[3.0, 6.0, 3.0], [0.0, 0, 0.0], [-3.0, -6.0, -3.0]], [[1.0, 3.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -3.0, -1.0]]])).view(1, 1, 3, 3, 3)
        dz = AdaptVal(torch.Tensor([[[-1.0, 0.0, 1.0], [-3.0, 0.0, 3.0], [-1.0, 0.0, 1.0]], [[-3.0, 0.0, 3.0], [-6.0, 0, 6.0], [-3.0, 0.0, 3.0]], [[-1.0, 0.0, 1.0], [-3.0, 0.0, 3.0], [-1.0, 0.0, 1.0]]])).view(1, 1, 3, 3, 3)
        self.spatial_filter = torch.cat((dx, dy, dz), 0)
        self.spatial_filter = self.spatial_filter.repeat(1, 1, 1, 1, 1)

    def __call__(self, disField):
        conv = F.conv3d
        jacobiField = conv(disField, self.spatial_filter)
        return torch.mean(jacobiField ** 2)


class ImageReconst(nn.Module):

    def __init__(self, I0, dim, sz, spacing):
        super(ImageReconst, self).__init__()
        self.dim = dim
        self.sz = sz
        self.spacing = spacing
        self.fourier_smoother = self.__gen_fourier_smoother()
        self.Target = self.__get_smoothed_target(I0)
        self.Source = self.__init_rand_source()
        self.sobel_filter = SobelFilter()
        self.smooth_factor = 0.1

    def __get_smoothed_target(self, I0):
        ITarget = AdaptVal(torch.from_numpy(I0.copy()))
        ITarget = self.fourier_smoother(ITarget).detach()
        return ITarget

    def __init_rand_source(self):
        ISource = nn.Parameter(torch.rand(self.Target.shape) * 2 - 1)
        return ISource

    def __gen_fourier_smoother(self):
        gaussianStd = 0.05
        mus = np.zeros(self.dim)
        stds = gaussianStd * np.ones(self.dim)
        centered_id = utils.centered_identity_map(self.sz, self.spacing)
        g = utils.compute_normalized_gaussian(centered_id, mus, stds)
        FFilter, _ = ce.create_complex_fourier_filter(g, self.sz)
        fourier_smoother = ce.FourierConvolution(FFilter)
        return fourier_smoother

    def get_reconst_img(self):
        return self.Source.detach()

    def get_target(self):
        return self.Target

    def forward(self):
        diff = self.fourier_smoother(self.Source) - self.Target
        smoothness = self.sobel_filter(self.Source)
        loss = torch.mean(torch.abs(diff)) + smoothness * self.smooth_factor
        return loss


def STNVal(x, ini):
    """
    the cuda version of stn is writing in float32
    so the input would first be converted into float32,
    the output would be converted to adaptive type
    """
    if USE_CUDA:
        if USE_FLOAT16:
            if ini == 1:
                return x.float()
            elif ini == -1:
                return x.half()
            else:
                raise ValueError('ini should be 1 or -1')
        else:
            return x
    else:
        return x


FFTVal = STNVal


def sel_fftn(dim):
    """
    sel the gpu and cpu version of the fft
    :param dim:
    :return: function pointer
    """
    if dim in [1, 2, 3]:
        f = torch.fft.rfftn
    else:
        None
    return f


def sel_ifftn(dim):
    """
    select the cpu and gpu version of the ifft
    :param dim:
    :return: function pointer
    """
    if dim in [1, 2, 3]:
        f = torch.fft.irfftn
    else:
        None
    return f


class FourierConvolution(nn.Module):
    """
    pyTorch function to compute convolutions in the Fourier domain: f = g*h
    """

    def __init__(self, complex_fourier_filter):
        """
        Constructor for the Fouier-based convolution
        
        :param complex_fourier_filter: Filter in the Fourier domain as created by *createComplexFourierFilter*
        """
        super(FourierConvolution, self).__init__()
        self.complex_fourier_filter = complex_fourier_filter
        self.dim = complex_fourier_filter.dim() - 1
        self.fftn = sel_fftn(self.dim)
        self.ifftn = sel_ifftn(self.dim)
        """The filter in the Fourier domain"""

    def forward(self, input):
        """
        Performs the Fourier-based filtering
        the 3d cpu fft is not implemented in fftn, to avoid fusing with batch and channel, here 3d is calcuated in loop
        1d 2d cpu works well because fft and fft2 is inbuilt, similarly , 1d 2d 3d gpu fft also is inbuilt

        in gpu implementation, the rfft is used for efficiency, which means the filter should be symmetric
        (input_real+input_img)(filter_real+filter_img) = (input_real*filter_real-input_img*filter_img) + (input_img*filter_real+input_real*filter_img)i
        filter_img =0, then get input_real*filter_real + (input_img*filter_real)i ac + bci

        :param input: Image
        :return: Filtered-image
        """
        input = FFTVal(input, ini=1)
        dims = tuple(range(2, len(input.shape)))
        f_input = self.fftn(input, dim=dims)
        f_filter_real = self.complex_fourier_filter[0]
        f_filter_real = f_filter_real.expand_as(f_input)
        f_conv = f_input * f_filter_real
        dim_input = len(input.shape)
        dim_input_batch = dim_input - self.dim
        conv_ouput_real = self.ifftn(f_conv, s=input.shape[dim_input_batch:], dim=dims)
        result = conv_ouput_real
        return FFTVal(result, ini=-1)


class InverseFourierConvolution(nn.Module):
    """
    pyTorch function to compute convolutions in the Fourier domain: f = g*h
    But uses the inverse of the smoothing filter
    """

    def __init__(self, complex_fourier_filter):
        """
        Constructor for the Fouier-based convolution (WARNING: EXPERIMENTAL)

        :param complex_fourier_filter: Filter in the Fourier domain as created by *createComplexFourierFilter*
        """
        super(InverseFourierConvolution, self).__init__()
        self.complex_fourier_filter = complex_fourier_filter
        self.dim = complex_fourier_filter.dim() - 1
        self.fftn = sel_fftn(self.dim)
        self.ifftn = sel_ifftn(self.dim)
        """Fourier filter"""
        self.alpha = 0.1
        """Regularizing weight"""

    def set_alpha(self, alpha):
        """
        Sets the regularizing weight
        
        :param alpha: regularizing weight
        """
        self.alpha = alpha

    def get_alpha(self):
        """
        Returns the regularizing weight
        
        :return: regularizing weight 
        """
        return self.alpha

    def forward(self, input):
        """
        Performs the Fourier-based filtering

        :param input: Image
        :return: Filtered-image
        """
        input = FFTVal(input, ini=1)
        dims = tuple(range(2, len(input.shape)))
        f_input = self.fftn(input, dim=dims)
        f_filter_real = self.complex_fourier_filter[0]
        f_filter_real = f_filter_real.expand_as(f_input)
        f_filter_real += self.alpha
        f_conv = f_input / f_filter_real
        dim_input = len(input.shape)
        dim_input_batch = dim_input - self.dim
        conv_ouput_real = self.ifftn(f_conv, s=input.shape[dim_input_batch:], dim=dims)
        result = conv_ouput_real
        return FFTVal(result, ini=-1)


class FourierGaussianConvolution(nn.Module):
    """
    pyTorch function to compute Gaussian convolutions in the Fourier domain: f = g*h.
    Also allows to differentiate through the Gaussian standard deviation.
    """

    def __init__(self, gaussian_fourier_filter_generator):
        """
        Constructor for the Fouier-based convolution
        :param sigma: standard deviation for the filter
        """
        super(FourierGaussianConvolution, self).__init__()
        self.gaussian_fourier_filter_generator = gaussian_fourier_filter_generator
        self.dim = self.gaussian_fourier_filter_generator.get_dimension()
        self.fftn = sel_fftn(self.dim)
        self.ifftn = sel_ifftn(self.dim)
        self.sigma_hook = None

    def register_zero_grad_hooker(self, param):
        """
        freeze the param that doesn't need gradient update
        :param param: param
        :return: hook function
        """
        hook = param.register_hook(lambda grad: grad * 0)
        return hook

    def freeze_sigma(self, sigma):
        """
        freeze the sigma in gaussian function
        :param sigma: sigma in gaussian
        :return:
        """
        if self.sigma_hook is None and sigma.requires_grad:
            self.sigma_hook = self.register_zero_grad_hooker(sigma)
        return sigma

    def _compute_convolution(self, input, complex_fourier_filter):
        input = FFTVal(input, ini=1)
        f_input = self.fftn(input, self.dim, onesided=True)
        f_filter_real = complex_fourier_filter[0]
        f_filter_real = f_filter_real.expand_as(f_input[..., 0])
        f_filter_real = torch.stack((f_filter_real, f_filter_real), -1)
        f_conv = f_input * f_filter_real
        dim_input = len(input.shape)
        dim_input_batch = dim_input - self.dim
        conv_ouput_real = self.ifftn(f_conv, self.dim, onesided=True, signal_sizes=input.shape[dim_input_batch:])
        result = conv_ouput_real
        return FFTVal(result, ini=-1)


class FourierSingleGaussianConvolution(FourierGaussianConvolution):
    """
    pyTorch function to compute Gaussian convolutions in the Fourier domain: f = g*h.
    Also allows to differentiate through the Gaussian standard deviation.
    """

    def __init__(self, gaussian_fourier_filter_generator, compute_std_gradient):
        """
        Constructor for the Fouier-based convolution
        :param sigma: standard deviation for the filter
        :param compute_std_gradient: if True computes the gradient with respect to the std, otherwise set to 0
        """
        super(FourierSingleGaussianConvolution, self).__init__(gaussian_fourier_filter_generator)
        self.gaussian_fourier_filter_generator = gaussian_fourier_filter_generator
        self.complex_fourier_filter = None
        self.complex_fourier_xsqr_filter = None
        self.input = None
        self.sigma = None
        self.compute_std_gradient = compute_std_gradient

    def forward(self, input, sigma):
        """
        Performs the Fourier-based filtering
        the 3d cpu fft is not implemented in fftn, to avoid fusing with batch and channel, here 3d is calcuated in loop
        1d 2d cpu works well because fft and fft2 is inbuilt, similarly , 1d 2d 3d gpu fft also is inbuilt

        in gpu implementation, the rfft is used for efficiency, which means the filter should be symmetric
        :param input: Image
        :return: Filtered-image
        """
        self.input = input
        self.sigma = sigma if self.compute_std_gradient else self.freeze_sigma(sigma)
        self.complex_fourier_filter = self.gaussian_fourier_filter_generator.get_gaussian_filters(self.sigma)[0]
        self.complex_fourier_xsqr_filter = self.gaussian_fourier_filter_generator.get_gaussian_xsqr_filters(self.sigma)[0]
        return self._compute_convolution(input, self.complex_fourier_filter)


class FourierMultiGaussianConvolution(FourierGaussianConvolution):
    """
    pyTorch function to compute multi Gaussian convolutions in the Fourier domain: f = g*h.
    Also allows to differentiate through the Gaussian standard deviation.
    """

    def __init__(self, gaussian_fourier_filter_generator, compute_std_gradients, compute_weight_gradients):
        """
        Constructor for the Fouier-based convolution

        :param gaussian_fourier_filter_generator: class instance that creates and caches the Gaussian filters
        :param compute_std_gradients: if set to True the gradients for std are computed, otherwise they are filled w/ zero
        :param compute_weight_gradients: if set to True the gradients for weights are computed, otherwise they are filled w/ zero
        """
        super(FourierMultiGaussianConvolution, self).__init__(gaussian_fourier_filter_generator)
        self.gaussian_fourier_filter_generator = gaussian_fourier_filter_generator
        self.complex_fourier_filters = None
        self.complex_fourier_xsqr_filters = None
        self.input = None
        self.weights = None
        self.sigmas = None
        self.nr_of_gaussians = None
        self.compute_std_gradients = compute_std_gradients
        self.compute_weight_gradients = compute_weight_gradients
        self.weight_hook = None

    def freeze_weight(self, weight):
        """
        freeze the weight for the muli-gaussian
        :param weight: weight  for the multi-gaussian
        :return:
        """
        if self.weight_hook is None and self.weights.requires_grad:
            self.weight_hook = self.register_zero_grad_hooker(weight)
        return weight

    def forward(self, input, sigmas, weights):
        """
        Performs the Fourier-based filtering
        the 3d cpu fft is not implemented in fftn, to avoid fusing with batch and channel, here 3d is calcuated in loop
        1d 2d cpu works well because fft and fft2 is inbuilt, similarly , 1d 2d 3d gpu fft also is inbuilt

        in gpu implementation, the rfft is used for efficiency, which means the filter should be symmetric
        :param input: Image
        :return: Filtered-image
        """
        self.input = input
        self.sigmas = sigmas if self.compute_std_gradients else self.freeze_sigma(sigmas)
        self.weights = weights if self.compute_weight_gradients else self.freeze_weight(weights)
        self.nr_of_gaussians = len(self.sigmas)
        nr_of_weights = len(self.weights)
        assert self.nr_of_gaussians == nr_of_weights
        self.complex_fourier_filters = self.gaussian_fourier_filter_generator.get_gaussian_filters(self.sigmas)
        self.complex_fourier_xsqr_filters = self.gaussian_fourier_filter_generator.get_gaussian_xsqr_filters(self.sigmas)
        ret = torch.zeros_like(input)
        for i in range(self.nr_of_gaussians):
            ret += self.weights[i] * self._compute_convolution(input, self.complex_fourier_filters[i])
        return ret


class FourierSetOfGaussianConvolutions(FourierGaussianConvolution):
    """
    pyTorch function to compute a set of Gaussian convolutions (as in the multi-Gaussian) in the Fourier domain: f = g*h.
    Also allows to differentiate through the standard deviations. THe output is not a smoothed field, but the
    set of all of them. This can then be fed into a subsequent neural network for further processing.
    """

    def __init__(self, gaussian_fourier_filter_generator, compute_std_gradients):
        """
        Constructor for the Fouier-based convolution

        :param gaussian_fourier_filter_generator: class instance that creates and caches the Gaussian filters
        :param compute_std_gradients: if set to True the gradients for the stds are computed, otherwise they are filled w/ zero
        """
        super(FourierSetOfGaussianConvolutions, self).__init__(gaussian_fourier_filter_generator)
        self.gaussian_fourier_filter_generator = gaussian_fourier_filter_generator
        self.complex_fourier_filters = None
        self.complex_fourier_xsqr_filters = None
        self.input = None
        self.sigmas = None
        self.nr_of_gaussians = None
        self.compute_std_gradients = compute_std_gradients

    def forward(self, input, sigmas):
        """
        Performs the Fourier-based filtering
        the 3d cpu fft is not implemented in fftn, to avoid fusing with batch and channel, here 3d is calculated in loop
        1d 2d cpu works well because fft and fft2 is inbuilt, similarly , 1d 2d 3d gpu fft also is inbuilt

        in gpu implementation, the rfft is used for efficiency, which means the filter should be symmetric
        :param input: Image
        :return: Filtered-image
        """
        self.input = input
        self.sigmas = sigmas if self.compute_std_gradients else self.freeze_sigma(sigmas)
        self.nr_of_gaussians = len(self.sigmas)
        self.complex_fourier_filters = self.gaussian_fourier_filter_generator.get_gaussian_filters(self.sigmas)
        if self.compute_std_gradients:
            self.complex_fourier_xsqr_filters = self.gaussian_fourier_filter_generator.get_gaussian_xsqr_filters(self.sigmas)
        sz = input.size()
        new_sz = [self.nr_of_gaussians] + list(sz)
        ret = AdaptVal(MyTensor(*new_sz))
        for i in range(self.nr_of_gaussians):
            ret[i, ...] = self._compute_convolution(input, self.complex_fourier_filters[i])
        return ret


def DimConv(dim):
    if dim == 1:
        return nn.Conv1d
    elif dim == 2:
        return nn.Conv2d
    elif dim == 3:
        return nn.Conv3d
    else:
        raise ValueError('Only supported for dimensions 1, 2, and 3')


def DimConvTranspose(dim):
    if dim == 1:
        return nn.ConvTranspose1d
    elif dim == 2:
        return nn.ConvTranspose2d
    elif dim == 3:
        return nn.ConvTranspose3d
    else:
        raise ValueError('Only supported for dimensions 1, 2, and 3')


def DimNoisyConv(dim):
    if dim == 1:
        return nc.NoisyConv1d
    elif dim == 2:
        return nc.NoisyConv2d
    elif dim == 3:
        return nc.NoisyConv3d
    else:
        raise ValueError('Only supported for dimensions 1, 2, and 3')


def DimNoisyConvTranspose(dim):
    if dim == 1:
        return nc.NoisyConvTranspose1d
    elif dim == 2:
        return nc.NoisyConvTranspose2d
    elif dim == 3:
        return nc.NoisyConvTranspose3d
    else:
        raise ValueError('Only supported for dimensions 1, 2, and 3')


def DimBatchNorm(dim):
    if dim == 1:
        return nn.BatchNorm1d
    elif dim == 2:
        return nn.BatchNorm2d
    elif dim == 3:
        return nn.BatchNorm3d
    else:
        raise ValueError('Only supported for dimensions 1, 2, and 3')


def DimInstanceNorm(dim):
    if dim == 1:
        return nn.InstanceNorm1d
    elif dim == 2:
        return nn.InstanceNorm2d
    elif dim == 3:
        return nn.InstanceNorm3d
    else:
        raise ValueError('Only supported for dimensions 1, 2, and 3')


def DimNormalization(dim, normalization_type, nr_channels, im_sz):
    normalization_types = ['batch', 'instance', 'layer', 'group']
    if normalization_type is not None:
        if normalization_type.lower() not in normalization_types:
            raise ValueError("normalization type either needs to be None or in ['layer'|'batch'|'instance']")
    else:
        return None
    if normalization_type.lower() == 'batch':
        return DimBatchNorm(dim)(nr_channels, eps=0.0001, momentum=0.75, affine=True)
    elif normalization_type.lower() == 'layer':
        int_im_sz = [int(elem) for elem in im_sz]
        layer_sz = [int(nr_channels)] + int_im_sz
        return nn.LayerNorm(layer_sz)
    elif normalization_type.lower() == 'instance':
        return DimInstanceNorm(dim)(nr_channels, eps=0.0001, momentum=0.75, affine=True)
    elif normalization_type.lower() == 'group':
        channels_per_group = nr_channels
        nr_groups = max(1, nr_channels // channels_per_group)
        return nn.GroupNorm(num_groups=nr_groups, num_channels=nr_channels)
    else:
        raise ValueError('Unknown normalization type: {}'.format(normalization_type))


class conv_norm_in_rel(nn.Module):

    def __init__(self, dim, in_channels, out_channels, kernel_size, im_sz, stride=1, active_unit='relu', same_padding=False, normalization_type='layer', reverse=False, group=1, dilation=1, use_noisy_convolution=False, noisy_convolution_std=0.25, noisy_convolution_optimize_over_std=False, use_noise_layer=False, noise_layer_std=0.25, start_reducing_from_iter=0):
        super(conv_norm_in_rel, self).__init__()
        self.use_noisy_convolution = use_noisy_convolution
        self.use_noise_layer = use_noise_layer
        if normalization_type is None:
            conv_bias = True
        else:
            conv_bias = False
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        if not reverse:
            if self.use_noisy_convolution:
                self.conv = DimNoisyConv(dim)(in_channels, out_channels, kernel_size, stride, padding=padding, groups=group, dilation=dilation, bias=conv_bias, scalar_sigmas=True, optimize_sigmas=noisy_convolution_optimize_over_std, std_init=noisy_convolution_std)
            else:
                self.conv = DimConv(dim)(in_channels, out_channels, kernel_size, stride, padding=padding, groups=group, dilation=dilation, bias=conv_bias)
        elif self.use_noisy_convolution:
            self.conv = DimNoisyConvTranspose(dim)(in_channels, out_channels, kernel_size, stride, padding=padding, groups=group, dilation=dilation, bias=conv_bias, scalar_sigmas=True, optimize_sigmas=noisy_convolution_optimize_over_std, std_init=noisy_convolution_std)
        else:
            self.conv = DimConvTranspose(dim)(in_channels, out_channels, kernel_size, stride, padding=padding, groups=group, dilation=dilation, bias=conv_bias)
        self.normalization = DimNormalization(dim, normalization_type, out_channels, im_sz) if normalization_type else None
        self.noisy_layer = nc.NoisyLayer(std_init=noise_layer_std, start_reducing_from_iter=start_reducing_from_iter) if self.use_noise_layer else None
        if active_unit == 'relu':
            self.active_unit = nn.ReLU(inplace=True)
        elif active_unit == 'elu':
            self.active_unit = nn.ELU(inplace=True)
        elif active_unit == 'leaky_relu':
            self.active_unit = nn.LeakyReLU(inplace=True)
        else:
            self.active_unit = None

    def forward(self, x, iter=0):
        if self.use_noisy_convolution:
            x = self.conv(x, iter=iter)
        else:
            x = self.conv(x)
        if self.normalization is not None:
            x = self.normalization(x)
        if self.use_noise_layer:
            x = self.noisy_layer(x, iter=iter)
        if self.active_unit is not None:
            x = self.active_unit(x)
        return x


class encoder_block_2d(nn.Module):

    def __init__(self, input_feature, output_feature, im_sz, use_dropout, normalization_type, dim):
        super(encoder_block_2d, self).__init__()
        self.dim = dim
        if normalization_type is None:
            conv_bias = True
        else:
            conv_bias = False
        self.conv_input = DimConv(self.dim)(in_channels=input_feature, out_channels=output_feature, kernel_size=3, stride=1, padding=1, dilation=1, bias=conv_bias)
        self.conv_inblock1 = DimConv(self.dim)(in_channels=output_feature, out_channels=output_feature, kernel_size=3, stride=1, padding=1, dilation=1, bias=conv_bias)
        self.conv_inblock2 = DimConv(self.dim)(in_channels=output_feature, out_channels=output_feature, kernel_size=3, stride=1, padding=1, dilation=1, bias=conv_bias)
        self.conv_pooling = DimConv(self.dim)(in_channels=output_feature, out_channels=output_feature, kernel_size=2, stride=2, padding=0, dilation=1, bias=conv_bias)
        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()
        self.prelu3 = nn.PReLU()
        self.prelu4 = nn.PReLU()
        if normalization_type:
            self.norm_1 = DimNormalization(self.dim, normalization_type=normalization_type, nr_channels=output_feature, im_sz=im_sz)
            self.norm_2 = DimNormalization(self.dim, normalization_type=normalization_type, nr_channels=output_feature, im_sz=im_sz)
            self.norm_3 = DimNormalization(self.dim, normalization_type=normalization_type, nr_channels=output_feature, im_sz=im_sz)
            self.norm_4 = DimNormalization(self.dim, normalization_type=normalization_type, nr_channels=output_feature, im_sz=im_sz)
        self.use_dropout = use_dropout
        self.normalization_type = normalization_type
        self.dropout = nn.Dropout(0.2)

    def apply_dropout(self, input):
        if self.use_dropout:
            return self.dropout(input)
        else:
            return input

    def forward_with_normalization(self, x):
        output = self.conv_input(x)
        output = self.apply_dropout(self.prelu1(self.norm_1(output)))
        output = self.apply_dropout(self.prelu2(self.norm_2(self.conv_inblock1(output))))
        output = self.apply_dropout(self.prelu3(self.norm_3(self.conv_inblock2(output))))
        return self.prelu4(self.norm_4(self.conv_pooling(output)))

    def forward_without_normalization(self, x):
        output = self.conv_input(x)
        output = self.apply_dropout(self.prelu1(output))
        output = self.apply_dropout(self.prelu2(self.conv_inblock1(output)))
        output = self.apply_dropout(self.prelu3(self.conv_inblock2(output)))
        return self.prelu4(self.conv_pooling(output))

    def forward(self, x):
        if self.normalization_type:
            return self.forward_with_normalization(x)
        else:
            return self.forward_without_normalization(x)


class decoder_block_2d(nn.Module):

    def __init__(self, input_feature, output_feature, im_sz, pooling_filter, use_dropout, normalization_type, dim, last_block=False):
        super(decoder_block_2d, self).__init__()
        self.dim = dim
        if normalization_type is None:
            conv_bias = True
        else:
            conv_bias = False
        self.conv_unpooling = DimConvTranspose(self.dim)(in_channels=input_feature, out_channels=input_feature, kernel_size=pooling_filter, stride=2, padding=0, output_padding=0, bias=conv_bias)
        self.conv_inblock1 = DimConv(self.dim)(in_channels=input_feature, out_channels=input_feature, kernel_size=3, stride=1, padding=1, dilation=1, bias=conv_bias)
        self.conv_inblock2 = DimConv(self.dim)(in_channels=input_feature, out_channels=input_feature, kernel_size=3, stride=1, padding=1, dilation=1, bias=conv_bias)
        if last_block:
            self.conv_output = DimConv(self.dim)(in_channels=input_feature, out_channels=output_feature, kernel_size=3, stride=1, padding=1, dilation=1)
        else:
            self.conv_output = DimConv(self.dim)(in_channels=input_feature, out_channels=output_feature, kernel_size=3, stride=1, padding=1, dilation=1, bias=conv_bias)
        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()
        self.prelu3 = nn.PReLU()
        self.prelu4 = nn.PReLU()
        if normalization_type:
            self.norm_1 = DimNormalization(self.dim, normalization_type=normalization_type, nr_channels=input_feature, im_sz=im_sz)
            self.norm_2 = DimNormalization(self.dim, normalization_type=normalization_type, nr_channels=input_feature, im_sz=im_sz)
            self.norm_3 = DimNormalization(self.dim, normalization_type=normalization_type, nr_channels=input_feature, im_sz=im_sz)
            if not last_block:
                self.norm_4 = DimNormalization(self.dim, normalization_type=normalization_type, nr_channels=input_feature, im_sz=im_sz)
        self.use_dropout = use_dropout
        self.normalization_type = normalization_type
        self.last_block = last_block
        self.dropout = nn.Dropout(0.2)
        self.output_feature = output_feature

    def apply_dropout(self, input):
        if self.use_dropout:
            return self.dropout(input)
        else:
            return input

    def forward_with_normalization(self, x):
        output = self.prelu1(self.norm_1(self.conv_unpooling(x)))
        output = self.apply_dropout(self.prelu2(self.norm_2(self.conv_inblock1(output))))
        output = self.apply_dropout(self.prelu3(self.norm_3(self.conv_inblock2(output))))
        if self.last_block:
            return self.conv_output(output)
        else:
            return self.apply_dropout(self.prelu4(self.norm_4(self.conv_output(output))))

    def forward_without_normalization(self, x):
        output = self.prelu1(self.conv_unpooling(x))
        output = self.apply_dropout(self.prelu2(self.conv_inblock1(output)))
        output = self.apply_dropout(self.prelu3(self.conv_inblock2(output)))
        if self.last_block:
            return self.conv_output(output)
        else:
            return self.apply_dropout(self.prelu4(self.conv_output(output)))

    def forward(self, x):
        if self.normalization_type:
            return self.forward_with_normalization(x)
        else:
            return self.forward_without_normalization(x)


class WeightRangeLoss(nn.Module):

    def __init__(self, dim, decay_factor, weight_type):
        super(WeightRangeLoss, self).__init__()
        self.dim = dim
        self.decay_factor = decay_factor
        self.is_w_K_w = weight_type == 'w_K_w'

    def forward(self, x, spacing, weights):
        weights = weights if not self.is_w_K_w else torch.sqrt(weights)
        view_sz = [1] + [len(weights)] + [1] * self.dim
        init_weights = weights.view(*view_sz)
        diff = x - init_weights
        volumeElement = spacing.prod()
        loss = utils.remove_infs_from_variable(diff ** 2).sum() * volumeElement
        return loss

    def cal_weights_for_weightrange(self, epoch):

        def sigmoid_decay(ep, static=5, k=5):
            static = static
            if ep < static:
                return float(1.0)
            else:
                ep = ep - static
                factor = k / (k + np.exp(ep / k))
            return float(factor)
        cur_weight = max(sigmoid_decay(epoch, static=10, k=self.decay_factor), 0.1)
        return cur_weight


class WeightInputRangeLoss(nn.Module):

    def __init__(self):
        super(WeightInputRangeLoss, self).__init__()

    def forward(self, x, spacing, use_weighted_linear_softmax=False, weights=None, min_weight=0.0, max_weight=1.0, dim=None):
        if spacing is not None:
            volumeElement = spacing.prod()
        else:
            volumeElement = 1.0
        if not use_weighted_linear_softmax:
            xd = x - torch.clamp(x, min_weight, max_weight)
            loss = utils.remove_infs_from_variable(xd ** 2).sum() * volumeElement
        else:
            if weights is None or dim is None:
                raise ValueError('Weights and dim need to be defined to use the weighted linear softmax')
            sz = x.size()
            input_offset = x.sum(dim=dim) / sz[dim]
            loss = MyTensor(1).zero_()
            None
            for c in range(sz[dim]):
                if dim == 0:
                    eff_input = weights[c] + x[c, ...] - input_offset
                elif dim == 1:
                    eff_input = weights[c] + x[:, c, ...] - input_offset
                elif dim == 2:
                    eff_input = weights[c] + x[:, :, c, ...] - input_offset
                elif dim == 3:
                    eff_input = weights[c] + x[:, :, :, c, ...] - input_offset
                elif dim == 4:
                    eff_input = weights[c] + x[:, :, :, :, c, ...] - input_offset
                else:
                    raise ValueError('Only dimensions {0,1,2,3,4} are supported')
                eff_input_d = eff_input - torch.clamp(eff_input, min_weight, max_weight)
                loss += utils.remove_infs_from_variable(eff_input_d ** 2).sum() * volumeElement
        return loss


class HLoss(nn.Module):

    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x, spacing):
        volumeElement = spacing.prod()
        b = x * torch.log(x)
        b = -1.0 * b.sum() * volumeElement
        return b


class GlobalHLoss(nn.Module):

    def __init__(self):
        super(GlobalHLoss, self).__init__()

    def forward(self, x, spacing):
        nr_of_labels = x.size()[1]
        P = MyTensor(nr_of_labels).zero_()
        sz = list(x.size())
        nr_of_elements = [sz[0]] + sz[2:]
        current_norm = float(np.array(nr_of_elements).prod().astype('float32'))
        for n in range(nr_of_labels):
            P[n] = x[:, n, ...].sum() / current_norm
        b = MyTensor(1).zero_()
        for n in range(nr_of_labels):
            b = b - P[n] * torch.log(P[n])
        return b


class OMTLoss(nn.Module):
    """
    OMT Loss function
    """

    def __init__(self, spacing, desired_power, use_log_transform, params, img_sz):
        super(OMTLoss, self).__init__()
        self.params = params
        self.spacing = spacing
        self.desired_power = desired_power
        self.use_log_transform = use_log_transform
        self.img_sz = img_sz
        self.use_boundary_mask = False
        self.mask = None
        if self.use_boundary_mask:
            None
            self.mask = utils.omt_boundary_weight_mask(img_sz, spacing, mask_range=3, mask_value=10, smoother_std=0.04)
        self.volume_element = self.spacing.prod()

    def compute_omt_penalty(self, weights, multi_gaussian_stds):
        if weights.size()[1] != len(multi_gaussian_stds):
            raise ValueError('Number of weights need to be the same as number of Gaussians. Format recently changed for weights to B x weights x X x Y')
        penalty = MyTensor(1).zero_()
        max_std = multi_gaussian_stds.max()
        min_std = multi_gaussian_stds.min()
        if self.desired_power == 2:
            for i, s in enumerate(multi_gaussian_stds):
                if self.use_log_transform:
                    penalty += (weights[:, i, ...].sum() if self.mask is None else weights[:, i] * self.mask[:, 0]) * torch.log(max_std / s) ** self.desired_power
                else:
                    penalty += (weights[:, i, ...].sum() if self.mask is None else weights[:, i] * self.mask[:, 0]) * (s - max_std) ** self.desired_power
            if self.use_log_transform:
                penalty /= torch.log(max_std / min_std) ** self.desired_power
            else:
                penalty /= (max_std - min_std) ** self.desired_power
        else:
            for i, s in enumerate(multi_gaussian_stds):
                if self.use_log_transform:
                    penalty += (weights[:, i, ...] if self.mask is None else weights[:, i] * self.mask[:, 0]).sum() * torch.abs(torch.log(max_std / s)) ** self.desired_power
                else:
                    penalty += (weights[:, i, ...] if self.mask is None else weights[:, i] * self.mask[:, 0]).sum() * torch.abs(s - max_std) ** self.desired_power
            if self.use_log_transform:
                penalty /= torch.abs(torch.log(max_std / min_std)) ** self.desired_power
            else:
                penalty /= torch.abs(max_std - min_std) ** self.desired_power
        penalty *= self.volume_element
        return penalty

    def cal_weights_for_omt(self, epoch):

        def sigmoid_decay(ep, static=5, k=5):
            static = static
            if ep < static:
                return float(1.0)
            else:
                ep = ep - static
                factor = k / (k + np.exp(ep / k))
            return float(factor)
        cur_weight = max(sigmoid_decay(epoch, static=30, k=10), 0.001)
        return cur_weight

    def forward(self, weights, gaussian_stds):
        return self.compute_omt_penalty(weights=weights, multi_gaussian_stds=gaussian_stds)


class TotalVariationLoss(nn.Module):
    """
    Loss function to penalize total variation
    """

    def __init__(self, dim, im_sz, spacing, use_omt_weighting=False, gaussian_stds=None, omt_power=1.0, omt_use_log_transformed_std=True, params=None):
        """

        :param params: ParameterDict() object to hold and keep track of general parameters
        """
        super(TotalVariationLoss, self).__init__()
        self.params = params
        """ParameterDict() parameters"""
        self.dim = dim
        self.im_sz = im_sz
        self.spacing = spacing
        self.use_omt_weighting = use_omt_weighting
        self.gaussian_stds = gaussian_stds
        self.omt_power = omt_power
        self.omt_use_log_transformed_std = omt_use_log_transformed_std
        self.smooth_image_for_edge_detection = self.params['smooth_image_for_edge_detection', True, 'Smooth image for edge detection']
        self.smooth_image_for_edge_detection_std = self.params['smooth_image_for_edge_detection_std', 0.01, 'Standard deviation for edge detection']
        self.tv_weights = None
        if self.use_omt_weighting:
            self.tv_weights = self._compute_tv_weights()
        if self.smooth_image_for_edge_detection:
            s_m_params = pars.ParameterDict()
            s_m_params['smoother']['type'] = 'gaussian'
            s_m_params['smoother']['gaussian_std'] = self.smooth_image_for_edge_detection_std
            self.image_smoother = sf.SmootherFactory(im_sz, spacing=spacing).create_smoother(s_m_params)
        else:
            self.image_smoother = None

    def _compute_tv_weights(self):
        multi_gaussian_stds = self.gaussian_stds.detach().cpu().numpy()
        max_std = max(multi_gaussian_stds)
        min_std = min(multi_gaussian_stds)
        tv_weights = MyTensor(len(multi_gaussian_stds))
        desired_power = self.omt_power
        use_log_transform = self.omt_use_log_transformed_std
        for i, s in enumerate(multi_gaussian_stds):
            if use_log_transform:
                tv_weights[i] = abs(np.log(max_std / s)) ** desired_power
            else:
                tv_weights[i] = abs(s - max_std) ** desired_power
        if use_log_transform:
            tv_weights /= abs(np.log(max_std / min_std)) ** desired_power
        else:
            tv_weights /= abs(max_std - min_std) ** desired_power
        return tv_weights

    def compute_local_weighted_tv_norm(self, I, weights, spacing, nr_of_gaussians, use_color_tv, pnorm=2):
        volumeElement = spacing.prod()
        individual_sum_of_total_variation_penalty = MyTensor(nr_of_gaussians).zero_()
        if self.smooth_image_for_edge_detection:
            I_edge = self.image_smoother.smooth(I)
        else:
            I_edge = I
        g_I = deep_smoothers.compute_localized_edge_penalty(I_edge[:, 0, ...], spacing, self.params)
        for g in range(nr_of_gaussians):
            c_local_norm_grad = deep_smoothers._compute_local_norm_of_gradient(weights[:, g, ...], spacing, pnorm)
            to_sum = g_I * c_local_norm_grad * volumeElement
            current_tv = to_sum.sum()
            individual_sum_of_total_variation_penalty[g] = current_tv
        if use_color_tv:
            if self.use_omt_weighting:
                total_variation_penalty = torch.norm(self.tv_weights * individual_sum_of_total_variation_penalty, p=2)
            else:
                total_variation_penalty = torch.norm(individual_sum_of_total_variation_penalty, p=2)
        elif self.use_omt_weighting:
            total_variation_penalty = (self.tv_weights * individual_sum_of_total_variation_penalty).sum()
        else:
            total_variation_penalty = individual_sum_of_total_variation_penalty.sum()
        return total_variation_penalty

    def forward(self, input_images, label_probabilities, use_color_tv=False):
        nr_of_gaussians = label_probabilities.size()[1]
        current_penalty = self.compute_local_weighted_tv_norm(input_images, label_probabilities, self.spacing, nr_of_gaussians, use_color_tv)
        return current_penalty


class ClusteringLoss(nn.Module):
    """
    Loss function for image clustering (this is here a relaxation of normalized cuts)
    """

    def __init__(self, dim, params):
        """

        :param params: ParameterDict() object to hold and keep track of general parameters
        """
        super(ClusteringLoss, self).__init__()
        self.params = params
        """ParameterDict() parameters"""
        self.dim = dim

    def _compute_cut_cost_for_label_k_1d(self, w_edge, p):
        raise ValueError('Not yet implemented')

    def _compute_cut_cost_for_label_k_3d(self, w_edge, p):
        raise ValueError('Not yet implemented')

    def _compute_cut_cost_for_label_k_2d(self, w_edge, p):
        batch_size = p.size()[0]
        cut_cost = AdaptVal(torch.zeros(batch_size))
        fdt = fd.FD_torch(spacing=np.array([1.0] * self.dim))
        p_xp = fdt.dXf(p)
        p_yp = fdt.dYf(p)
        for b in range(batch_size):
            nom = (p[b, ...] * (p[b, ...] + w_edge[b, ...] * (p_xp[b, ...] + p_yp[b, ...]))).sum()
            denom = (p[b, ...] * (1.0 + 2 * w_edge[b, ...])).sum()
            cut_cost[b] = nom / denom
        return cut_cost

    def _compute_cut_cost_for_label_k(self, w_edge, p):
        if self.dim == 1:
            return self._compute_cut_cost_for_label_k_1d(w_edge, p)
        elif self.dim == 2:
            return self._compute_cut_cost_for_label_k_2d(w_edge, p)
        elif self.dim == 3:
            return self._compute_cut_cost_for_label_k_3d(w_edge, p)
        else:
            raise ValueError('Only defined for dimensions {1,2,3}')

    def get_last_kernel_size(self):
        return 1

    def forward(self, input_images, spacing, label_probabilities):
        localized_edge_penalty = deep_smoothers.compute_localized_edge_penalty(input_images[:, 0, ...], spacing, self.params)
        batch_size = label_probabilities.size()[0]
        nr_of_clusters = label_probabilities.size()[1]
        current_penalties = AdaptVal(torch.ones(batch_size) * nr_of_clusters)
        for k in range(nr_of_clusters):
            current_penalties -= self._compute_cut_cost_for_label_k(w_edge=localized_edge_penalty, p=label_probabilities[:, k, ...])
        current_penalty = current_penalties.sum()
        return current_penalty


def weighted_softmax(input, dim=None, weights=None):
    """Applies a softmax function.

    Weighted_softmax is defined as:

    :math:`weighted_softmax(x) = \\frac{w_i exp(x_i)}{\\sum_j w_j exp(x_j)}`

    It is applied to all slices along dim, and will rescale them so that the elements
    lie in the range `(0, 1)` and sum to 1.

    See :class:`~torch.nn.WeightedSoftmax` for more details.

    Arguments:
        input (Variable): input
        dim (int): A dimension along which weighted_softmax will be computed.

    """
    if dim is None:
        raise ValueError('dimension needs to be defined!')
    sz = input.size()
    if weights is None:
        weights = [1.0] * sz[dim]
    nr_of_weights = len(weights)
    assert sz[dim] == nr_of_weights
    ret = torch.zeros_like(input)
    max_in, _ = torch.max(input, dim=dim)
    if dim == 0:
        norm = torch.zeros_like(input[0, ...])
        for c in range(sz[0]):
            norm += weights[c] * torch.exp(input[c, ...] - max_in)
        for c in range(sz[0]):
            ret[c, ...] = weights[c] * torch.exp(input[c, ...] - max_in) / norm
    elif dim == 1:
        norm = torch.zeros_like(input[:, 0, ...])
        for c in range(sz[1]):
            norm += weights[c] * torch.exp(input[:, c, ...] - max_in)
        for c in range(sz[1]):
            ret[:, c, ...] = weights[c] * torch.exp(input[:, c, ...] - max_in) / norm
    elif dim == 2:
        norm = torch.zeros_like(input[:, :, 0, ...])
        for c in range(sz[2]):
            norm += weights[c] * torch.exp(input[:, :, c, ...] - max_in)
        for c in range(sz[2]):
            ret[:, :, c, ...] = weights[c] * torch.exp(input[:, :, c, ...] - max_in) / norm
    elif dim == 3:
        norm = torch.zeros_like(input[:, :, :, 0, ...])
        for c in range(sz[3]):
            norm += weights[c] * torch.exp(input[:, :, :, c, ...] - max_in)
        for c in range(sz[3]):
            ret[:, :, :, c, ...] = weights[c] * torch.exp(input[:, :, :, c, ...] - max_in) / norm
    elif dim == 4:
        norm = torch.zeros_like(input[:, :, :, :, 0, ...])
        for c in range(sz[4]):
            norm += weights[c] * torch.exp(input[:, :, :, :, c, ...] - max_in)
        for c in range(sz[4]):
            ret[:, :, :, :, c, ...] = weights[c] * torch.exp(input[:, :, :, :, c, ...] - max_in) / norm
    else:
        raise ValueError('weighted_softmax is only supported for dimensions 0, 1, 2, 3, and 4.')
    return ret


class WeightedSoftmax(nn.Module):
    """Applies the WeightedSoftmax function to an n-dimensional input Tensor
    rescaling them so that the elements of the n-dimensional output Tensor
    lie in the range (0,1) and sum to 1

    WeightedSoftmax is defined as
    :math:`f_i(x) = \\frac{w_i\\exp(x_i)}{\\sum_j w_j\\exp(x_j)}`

    It is assumed that w_i>0 and that the weights sum up to one.
    The effect of this weighting is that for a zero input (x=0) the output for f_i(x) will be w_i.
    I.e., we can obtain a default output which is not 1/n.

    Shape:
        - Input: any shape
        - Output: same as input

    Returns:
        a Tensor of the same dimension and shape as the input with
        values in the range [0, 1]

    Arguments:
        dim (int): A dimension along which WeightedSoftmax will be computed (so every slice
            along dim will sum to 1).

    Examples::

        >>> m = nn.WeightedSoftmax()
        >>> input = autograd.torch.randn(2, 3))
        >>> print(input)
        >>> print(m(input))
    """

    def __init__(self, dim=None, weights=None):
        super(WeightedSoftmax, self).__init__()
        self.dim = dim
        self.weights = weights

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, 'dim'):
            self.dim = None
        if not hasattr(self, 'weights'):
            self.weights = None

    def forward(self, input):
        return weighted_softmax(input, self.dim, self.weights, _stacklevel=5)

    def __repr__(self):
        return self.__class__.__name__ + '()'


def stable_softmax(input, dim=None):
    """Applies a numerically stqable softmax function.

    stable_softmax is defined as:

    :math:`stable_softmax(x) = \\frac{exp(x_i)}{\\sum_j exp(x_j)}`

    It is applied to all slices along dim, and will rescale them so that the elements
    lie in the range `(0, 1)` and sum to 1.

    See :class:`~torch.nn.StableSoftmax` for more details.

    Arguments:
        input (Variable): input
        dim (int): A dimension along which stable_softmax will be computed.

    """
    if dim is None:
        raise ValueError('dimension needs to be defined!')
    sz = input.size()
    ret = torch.zeros_like(input)
    max_in, _ = torch.max(input, dim=dim)
    if dim == 0:
        norm = torch.zeros_like(input[0, ...])
        for c in range(sz[0]):
            norm += torch.exp(input[c, ...] - max_in)
        for c in range(sz[0]):
            ret[c, ...] = torch.exp(input[c, ...] - max_in) / norm
    elif dim == 1:
        norm = torch.zeros_like(input[:, 0, ...])
        for c in range(sz[1]):
            norm += torch.exp(input[:, c, ...] - max_in)
        for c in range(sz[1]):
            ret[:, c, ...] = torch.exp(input[:, c, ...] - max_in) / norm
    elif dim == 2:
        norm = torch.zeros_like(input[:, :, 0, ...])
        for c in range(sz[2]):
            norm += torch.exp(input[:, :, c, ...] - max_in)
        for c in range(sz[2]):
            ret[:, :, c, ...] = torch.exp(input[:, :, c, ...] - max_in) / norm
    elif dim == 3:
        norm = torch.zeros_like(input[:, :, :, 0, ...])
        for c in range(sz[3]):
            norm += torch.exp(input[:, :, :, c, ...] - max_in)
        for c in range(sz[3]):
            ret[:, :, :, c, ...] = torch.exp(input[:, :, :, c, ...] - max_in) / norm
    elif dim == 4:
        norm = torch.zeros_like(input[:, :, :, :, 0, ...])
        for c in range(sz[4]):
            norm += torch.exp(input[:, :, :, :, c, ...] - max_in)
        for c in range(sz[4]):
            ret[:, :, :, :, c, ...] = torch.exp(input[:, :, :, :, c, ...] - max_in) / norm
    else:
        raise ValueError('weighted_softmax is only supported for dimensions 0, 1, 2, 3, and 4.')
    return ret


class StableSoftmax(nn.Module):
    """Applies the StableSoftmax function to an n-dimensional input Tensor
    rescaling them so that the elements of the n-dimensional output Tensor
    lie in the range (0,1) and sum to 1

    StableSoftmax is defined as
    :math:`f_i(x) = \\frac{exp(x_i)}{\\sum_j exp(x_j)}`

    Shape:
        - Input: any shape
        - Output: same as input

    Returns:
        a Tensor of the same dimension and shape as the input with
        values in the range [0, 1]

    Arguments:
        dim (int): A dimension along which WeightedSoftmax will be computed (so every slice
            along dim will sum to 1).

    Examples::

        >>> m = nn.StableSoftmax()
        >>> input = autograd.torch.randn(2, 3))
        >>> print(input)
        >>> print(m(input))
    """

    def __init__(self, dim=None):
        super(StableSoftmax, self).__init__()
        self.dim = dim

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, 'dim'):
            self.dim = None

    def forward(self, input):
        return stable_softmax(input, self.dim, _stacklevel=5)

    def __repr__(self):
        return self.__class__.__name__ + '()'


def weighted_linear_softmax(input, dim=None, weights=None):
    """Applies a softmax function.

    Weighted_linear_softmax is defined as:

    :math:`weighted_linear_softmax(x) = \\frac{clamp(x_i+w_i,0,1)}{\\sum_j clamp(x_j+w_j,0,1)}`

    It is applied to all slices along dim, and will rescale them so that the elements
    lie in the range `(0, 1)` and sum to 1.

    See :class:`~torch.nn.WeightedLinearSoftmax` for more details.

    Arguments:
        input (Variable): input
        dim (int): A dimension along which weighted_linear_softmax will be computed.

    """
    if dim is None:
        raise ValueError('dimension needs to be defined!')
    sz = input.size()
    if weights is None:
        weights = [1.0 / sz[dim]] * sz[dim]
    nr_of_weights = len(weights)
    assert sz[dim] == nr_of_weights
    ret = torch.zeros_like(input)
    input_offset = input.sum(dim=dim) / sz[dim]
    if dim == 0:
        norm = torch.zeros_like(input[0, ...])
        for c in range(sz[0]):
            norm += torch.clamp(weights[c] + input[c, ...] - input_offset, min=0, max=1)
        for c in range(sz[0]):
            ret[c, ...] = torch.clamp(weights[c] + input[c, ...] - input_offset, min=0, max=1) / norm
    elif dim == 1:
        norm = torch.zeros_like(input[:, 0, ...])
        for c in range(sz[1]):
            norm += torch.clamp(weights[c] + input[:, c, ...] - input_offset, min=0, max=1)
        for c in range(sz[1]):
            ret[:, c, ...] = torch.clamp(weights[c] + input[:, c, ...] - input_offset, min=0, max=1) / norm
    elif dim == 2:
        norm = torch.zeros_like(input[:, :, 0, ...])
        for c in range(sz[2]):
            norm += torch.clamp(weights[c] + input[:, :, c, ...] - input_offset, min=0, max=1)
        for c in range(sz[2]):
            ret[:, :, c, ...] = torch.clamp(weights[c] + input[:, :, c, ...] - input_offset, min=0, max=1) / norm
    elif dim == 3:
        norm = torch.zeros_like(input[:, :, :, 0, ...])
        for c in range(sz[3]):
            norm += torch.clamp(weights[c] + input[:, :, :, c, ...] - input_offset, min=0, max=1)
        for c in range(sz[3]):
            ret[:, :, :, c, ...] = torch.clamp(weights[c] + input[:, :, :, c, ...] - input_offset, min=0, max=1) / norm
    elif dim == 4:
        norm = torch.zeros_like(input[:, :, :, :, 0, ...])
        for c in range(sz[4]):
            norm += torch.clamp(weights[c] + input[:, :, :, :, c, ...] - input_offset, min=0, max=1)
        for c in range(sz[4]):
            ret[:, :, :, :, c, ...] = torch.clamp(weights[c] + input[:, :, :, :, c, ...] - input_offset, min=0, max=1) / norm
    else:
        raise ValueError('weighted_softmax is only supported for dimensions 0, 1, 2, 3, and 4.')
    return ret


class WeightedLinearSoftmax(nn.Module):
    """Applies the a WeightedLinearSoftmax function to an n-dimensional input Tensor
    rescaling them so that the elements of the n-dimensional output Tensor
    lie in the range (0,1) and sum to 1

    WeightedSoftmax is defined as
    :math:`f_i(x) = \\frac{clamp(x_i+w_i,0,1)}{\\sum_j clamp(x_j+w_j,0,1)}`

    It is assumed that 0<=w_i<=1 and that the weights sum up to one.
    The effect of this weighting is that for a zero input (x=0) the output for f_i(x) will be w_i.

    Shape:
        - Input: any shape
        - Output: same as input

    Returns:
        a Tensor of the same dimension and shape as the input with
        values in the range [0, 1]

    Arguments:
        dim (int): A dimension along which WeightedSoftmax will be computed (so every slice
            along dim will sum to 1).

    Examples::

        >>> m = nn.WeightedLinearSoftmax()
        >>> input = torch.randn(2, 3)
        >>> print(input)
        >>> print(m(input))
    """

    def __init__(self, dim=None, weights=None):
        super(WeightedLinearSoftmax, self).__init__()
        self.dim = dim
        self.weights = weights

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, 'dim'):
            self.dim = None
        if not hasattr(self, 'weights'):
            self.weights = None

    def forward(self, input):
        return weighted_linear_softmax(input, self.dim, self.weights, _stacklevel=5)

    def __repr__(self):
        return self.__class__.__name__ + '()'


def weighted_linear_softnorm(input, dim=None, weights=None):
    """Applies a weighted linear softnorm function.

    Weighted_linear_softnorm is defined as:

    :math:`weighted_linear_softnorm(x) = \\frac{clamp(x_i+w_i,0,1)}{\\sqrt{\\sum_j clamp(x_j+w_j,0,1)**2}}`

    It is applied to all slices along dim, and will rescale them so that the elements
    lie in the range `(0, 1)` and sum to 1.

    See :class:`~torch.nn.WeightedLinearSoftnorm` for more details.

    Arguments:
        input (Variable): input
        dim (int): A dimension along which weighted_linear_softnorm will be computed.

    """
    if dim is None:
        raise ValueError('dimension needs to be defined!')
    sz = input.size()
    if weights is None:
        weights = [1.0 / sz[dim]] * sz[dim]
    nr_of_weights = len(weights)
    assert sz[dim] == nr_of_weights
    ret = torch.zeros_like(input)
    clamped_vals = torch.zeros_like(input)
    if dim == 0:
        input_offset = input.sum(dim=0) / sz[0]
        for c in range(sz[0]):
            clamped_vals[c, ...] = torch.clamp(weights[c] + input[c, ...] - input_offset, min=0, max=1)
        norm = torch.norm(clamped_vals, p=2, dim=dim)
        for c in range(sz[0]):
            ret[c, ...] = clamped_vals[c, ...] / norm
    elif dim == 1:
        input_offset = input.sum(dim=1) / sz[1]
        for c in range(sz[1]):
            clamped_vals[:, c, ...] = torch.clamp(weights[c] + input[:, c, ...] - input_offset, min=0, max=1)
        norm = torch.norm(clamped_vals, p=2, dim=dim)
        for c in range(sz[1]):
            ret[:, c, ...] = clamped_vals[:, c, ...] / norm
    elif dim == 2:
        input_offset = input.sum(dim=2) / sz[2]
        for c in range(sz[2]):
            clamped_vals[:, :, c, ...] = torch.clamp(weights[c] + input[:, :, c, ...] - input_offset, min=0, max=1)
        norm = torch.norm(clamped_vals, p=2, dim=dim)
        for c in range(sz[2]):
            ret[:, :, c, ...] = clamped_vals[:, :, c, ...] / norm
    elif dim == 3:
        input_offset = input.sum(dim=3) / sz[3]
        for c in range(sz[3]):
            clamped_vals[:, :, :, c, ...] = torch.clamp(weights[c] + input[:, :, :, c, ...] - input_offset, min=0, max=1)
        norm = torch.norm(clamped_vals, p=2, dim=dim)
        for c in range(sz[3]):
            ret[:, :, :, c, ...] = clamped_vals[:, :, :, c, ...] / norm
    elif dim == 4:
        input_offset = input.sum(dim=4) / sz[4]
        for c in range(sz[4]):
            clamped_vals[:, :, :, :, c, ...] = torch.clamp(weights[c] + input[:, :, :, :, c, ...] - input_offset, min=0, max=1)
        norm = torch.norm(clamped_vals, p=2, dim=dim)
        for c in range(sz[4]):
            ret[:, :, :, :, c, ...] = clamped_vals[:, :, :, :, c, ...] / norm
    else:
        raise ValueError('weighted_softmax is only supported for dimensions 0, 1, 2, 3, and 4.')
    return ret


class WeightedLinearSoftnorm(nn.Module):
    """Applies the a WeightedLinearSoftnorm function to an n-dimensional input Tensor
    rescaling them so that the elements of the n-dimensional output Tensor
    lie in the range (0,1) and the square sum to 1

    WeightedLinearSoftnorm is defined as
    :math:`f_i(x) = \\frac{clamp(x_i+w_i,0,1)}{\\sqrt{\\sum_j clamp(x_j+w_j,0,1)**2}}`

    It is assumed that 0<=w_i<=1 and that the weights sum up to one.
    The effect of this weighting is that for a zero input (x=0) the output for f_i(x) will be w_i.

    Shape:
        - Input: any shape
        - Output: same as input

    Returns:
        a Tensor of the same dimension and shape as the input with
        values in the range [0, 1]

    Arguments:
        dim (int): A dimension along which WeightedSoftmax will be computed (so every slice
            along dim will sum to 1).

    Examples::

        >>> m = nn.WeightedLinearSoftnorm()
        >>> input = torch.randn(2, 3)
        >>> print(input)
        >>> print(m(input))
    """

    def __init__(self, dim=None, weights=None):
        super(WeightedLinearSoftnorm, self).__init__()
        self.dim = dim
        self.weights = weights

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, 'dim'):
            self.dim = None
        if not hasattr(self, 'weights'):
            self.weights = None

    def forward(self, input):
        return weighted_linear_softnorm(input, self.dim, self.weights, _stacklevel=5)

    def __repr__(self):
        return self.__class__.__name__ + '()'


def linear_softnorm(input, dim=None):
    """Normalizes so that the squares of the resulting values sum up to one and are positive.

    linear_softnorm is defined as:

    :math:`linear_softnorm(x) = \\frac{clamp(x_i,0,1)}{\\sqrt{\\sum_j clamp(x_j,0,1)^2}}`

    It is applied to all slices along dim, and will rescale them so that the elements
    lie in the range `(0, 1)` and their squares sum to 1.

    See :class:`~torch.nn.LinearSoftnorm` for more details.

    Arguments:
        input (Variable): input
        dim (int): A dimension along which linear_softnorm will be computed.

    """
    if dim is None:
        raise ValueError('dimension needs to be defined!')
    sz = input.size()
    ret = torch.zeros_like(input)
    clamped_vals = torch.clamp(input, min=0, max=1)
    norm = torch.norm(clamped_vals, p=2, dim=dim)
    if dim == 0:
        for c in range(sz[0]):
            ret[c, ...] = clamped_vals[c, ...] / norm
    elif dim == 1:
        for c in range(sz[1]):
            ret[:, c, ...] = clamped_vals[:, c, ...] / norm
    elif dim == 2:
        for c in range(sz[2]):
            ret[:, :, c, ...] = clamped_vals[:, :, c, ...] / norm
    elif dim == 3:
        for c in range(sz[3]):
            ret[:, :, :, c, ...] = clamped_vals[:, :, :, c, ...] / norm
    elif dim == 4:
        for c in range(sz[4]):
            ret[:, :, :, :, c, ...] = clamped_vals[:, :, :, :, c, ...] / norm
    else:
        raise ValueError('linear_softnorm is only supported for dimensions 0, 1, 2, 3, and 4.')
    return ret


class LinearSoftnorm(nn.Module):
    """Applies the a LinearSoftnrom function to an n-dimensional input Tensor
    rescaling them so that the elements of the n-dimensional output Tensor
    lie in the range (0,1) and their square sums up to 1

    LinearSoftnorm is defined as
    :math:`f_i(x) = \\frac{clamp(x_i,0,1)}{\\sqrt{\\sum_j clamp(x_j,0,1)**2}}`

    Shape:
        - Input: any shape
        - Output: same as input

    Returns:
        a Tensor of the same dimension and shape as the input with
        values in the range [0, 1]

    Arguments:
        dim (int): A dimension along which WeightedSoftmax will be computed (so every slice
            along dim will sum to 1).

    Examples::

        >>> m = nn.LinearSoftnorm()
        >>> input = torch.randn(2, 3)
        >>> print(input)
        >>> print(m(input))
    """

    def __init__(self, dim=None):
        super(LinearSoftnorm, self).__init__()
        self.dim = dim

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, 'dim'):
            self.dim = None

    def forward(self, input):
        return linear_softnorm(input, self.dim, _stacklevel=5)

    def __repr__(self):
        return self.__class__.__name__ + '()'


def linear_softmax(input, dim=None):
    """Applies linear a softmax function.

    linear_softmax is defined as:

    :math:`linear_softmax(x) = \\frac{clamp(x_i,0,1)}{\\sum_j clamp(x_j,0,1)}`

    It is applied to all slices along dim, and will rescale them so that the elements
    lie in the range `(0, 1)` and sum to 1.

    See :class:`~torch.nn.LinearSoftmax` for more details.

    Arguments:
        input (Variable): input
        dim (int): A dimension along which linear_softmax will be computed.

    """
    if dim is None:
        raise ValueError('dimension needs to be defined!')
    sz = input.size()
    ret = torch.zeros_like(input)
    if dim == 0:
        norm = torch.zeros_like(input[0, ...])
        for c in range(sz[0]):
            norm += torch.clamp(input[c, ...], min=0, max=1)
        for c in range(sz[0]):
            ret[c, ...] = torch.clamp(input[c, ...], min=0, max=1) / norm
    elif dim == 1:
        norm = torch.zeros_like(input[:, 0, ...])
        for c in range(sz[1]):
            norm += torch.clamp(input[:, c, ...], min=0, max=1)
        for c in range(sz[1]):
            ret[:, c, ...] = torch.clamp(input[:, c, ...], min=0, max=1) / norm
    elif dim == 2:
        norm = torch.zeros_like(input[:, :, 0, ...])
        for c in range(sz[2]):
            norm += torch.clamp(input[:, :, c, ...], min=0, max=1)
        for c in range(sz[2]):
            ret[:, :, c, ...] = torch.clamp(input[:, :, c, ...], min=0, max=1) / norm
    elif dim == 3:
        norm = torch.zeros_like(input[:, :, :, 0, ...])
        for c in range(sz[3]):
            norm += torch.clamp(input[:, :, :, c, ...], min=0, max=1)
        for c in range(sz[3]):
            ret[:, :, :, c, ...] = torch.clamp(input[:, :, :, c, ...], min=0, max=1) / norm
    elif dim == 4:
        norm = torch.zeros_like(input[:, :, :, :, 0, ...])
        for c in range(sz[4]):
            norm += torch.clamp(input[:, :, :, :, c, ...], min=0, max=1)
        for c in range(sz[4]):
            ret[:, :, :, :, c, ...] = torch.clamp(input[:, :, :, :, c, ...], min=0, max=1) / norm
    else:
        raise ValueError('linear_softmax is only supported for dimensions 0, 1, 2, 3, and 4.')
    return ret


class LinearSoftmax(nn.Module):
    """Applies the a LinearSoftmax function to an n-dimensional input Tensor
    rescaling them so that the elements of the n-dimensional output Tensor
    lie in the range (0,1) and sum to 1

    WeightedSoftmax is defined as
    :math:`f_i(x) = \\frac{clamp(x_i,0,1)}{\\sum_j clamp(x_j,0,1)}`

    Shape:
        - Input: any shape
        - Output: same as input

    Returns:
        a Tensor of the same dimension and shape as the input with
        values in the range [0, 1]

    Arguments:
        dim (int): A dimension along which WeightedSoftmax will be computed (so every slice
            along dim will sum to 1).

    Examples::

        >>> m = nn.LinearSoftmax()
        >>> input = torch.randn(2, 3)
        >>> print(input)
        >>> print(m(input))
    """

    def __init__(self, dim=None):
        super(LinearSoftmax, self).__init__()
        self.dim = dim

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, 'dim'):
            self.dim = None

    def forward(self, input):
        return linear_softmax(input, self.dim, _stacklevel=5)

    def __repr__(self):
        return self.__class__.__name__ + '()'


def weighted_sqrt_softmax(input, dim=None, weights=None):
    """Applies a weighted square-root softmax function.

    Weighted_sqrt_softmax is defined as:

    :math:`weighted_sqrt_softmax(x) = \\frac{\\sqrt{w_i} exp(x_i)}{\\sqrt{\\sum_j w_j (exp(x_j))^2}}`

    It is applied to all slices along dim, and will rescale them so that the elements
    lie in the range `(0, 1)` and sum to 1.

    See :class:`~torch.nn.WeightedSoftmax` for more details.

    Arguments:
        input (Variable): input
        dim (int): A dimension along which weighted_softmax will be computed.

    """
    if dim is None:
        raise ValueError('dimension needs to be defined!')
    sz = input.size()
    if weights is None:
        weights = [1.0] * sz[dim]
    nr_of_weights = len(weights)
    assert sz[dim] == nr_of_weights
    ret = torch.zeros_like(input)
    max_in, _ = torch.max(input, dim=dim)
    if dim == 0:
        norm_sqr = torch.zeros_like(input[0, ...])
        for c in range(sz[0]):
            norm_sqr += weights[c] * torch.exp(input[c, ...] - max_in) ** 2
        norm = torch.sqrt(norm_sqr)
        for c in range(sz[0]):
            ret[c, ...] = torch.sqrt(weights[c]) * torch.exp(input[c, ...] - max_in) / norm
    elif dim == 1:
        norm_sqr = torch.zeros_like(input[:, 0, ...])
        for c in range(sz[1]):
            norm_sqr += weights[c] * torch.exp(input[:, c, ...] - max_in) ** 2
        norm = torch.sqrt(norm_sqr)
        for c in range(sz[1]):
            ret[:, c, ...] = torch.sqrt(weights[c]) * torch.exp(input[:, c, ...] - max_in) / norm
    elif dim == 2:
        norm_sqr = torch.zeros_like(input[:, :, 0, ...])
        for c in range(sz[2]):
            norm_sqr += weights[c] * torch.exp(input[:, :, c, ...] - max_in) ** 2
        norm = torch.sqrt(norm_sqr)
        for c in range(sz[2]):
            ret[:, :, c, ...] = torch.sqrt(weights[c]) * torch.exp(input[:, :, c, ...] - max_in) / norm
    elif dim == 3:
        norm_sqr = torch.zeros_like(input[:, :, :, 0, ...])
        for c in range(sz[3]):
            norm_sqr += weights[c] * torch.exp(input[:, :, :, c, ...] - max_in) ** 2
        norm = torch.sqrt(norm_sqr)
        for c in range(sz[3]):
            ret[:, :, :, c, ...] = torch.sqrt(weights[c]) * torch.exp(input[:, :, :, c, ...] - max_in) / norm
    elif dim == 4:
        norm_sqr = torch.zeros_like(input[:, :, :, :, 0, ...])
        for c in range(sz[4]):
            norm_sqr += weights[c] * torch.exp(input[:, :, :, :, c, ...] - max_in) ** 2
        norm = torch.sqrt(norm_sqr)
        for c in range(sz[4]):
            ret[:, :, :, :, c, ...] = torch.sqrt(weights[c]) * torch.exp(input[:, :, :, :, c, ...] - max_in) / norm
    else:
        raise ValueError('weighted_softmax is only supported for dimensions 0, 1, 2, 3, and 4.')
    return ret


class WeightedSqrtSoftmax(nn.Module):
    """Applies the WeightedSqrtSoftmax function to an n-dimensional input Tensor
    rescaling them so that the elements of the n-dimensional output Tensor
    lie in the range (0,1) and their squares sum to 1

    WeightedSoftmax is defined as
    :math:`f_i(x) = \\frac{\\sqrt{w_i}\\exp(x_i)}{\\sqrt{\\sum_j w_j\\exp(x_j)^2}}`

    It is assumed that w_i>=0 and that the weights sum up to one.
    The effect of this weighting is that for a zero input (x=0) the output for f_i(x) will be \\sqrt{w_i}.
    I.e., we can obtain a default output which is not 1/n and if we sqaure the outputs we are back
    to the original weights for zero (input). This is useful behavior to implement, for example, local
    kernel weightings while avoiding square roots of weights that may be close to zero (and hence potential
    numerical issues with the gradient). The assumption is, of course, here that the weights are fixed and are not being
    optimized over, otherwise there would still be numerical issues. TODO: check that this is indeed working as planned.

    Shape:
        - Input: any shape
        - Output: same as input

    Returns:
        a Tensor of the same dimension and shape as the input with
        positive values such that their squares sum up to one.

    Arguments:
        dim (int): A dimension along which WeightedSqrtSoftmax will be computed (so every slice
            along dim will sum to 1).

    Examples::

        >>> m = nn.WeightedSqrtSoftmax()
        >>> input = autograd.torch.randn(2, 3))
        >>> print(input)
        >>> print(m(input))
    """

    def __init__(self, dim=None, weights=None):
        super(WeightedSqrtSoftmax, self).__init__()
        self.dim = dim
        self.weights = weights

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, 'dim'):
            self.dim = None
        if not hasattr(self, 'weights'):
            self.weights = None

    def forward(self, input):
        return weighted_sqrt_softmax(input, self.dim, self.weights, _stacklevel=5)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ODEWrapFunc(nn.Module):
    """
    a wrap on tensor based torchdiffeq input
    """

    def __init__(self, nested_class, has_combined_input=False, pars=None, variables_from_optimizer=None, extra_var=None, dim_info=None):
        """

        :param nested_class: the model to be integrated
        :param has_combined_input: the model has combined input in x e.g. EPDiff* equation, otherwise, model has individual input e.g. advect* , has x,u two inputs
        :param pars: ParameterDict, settings passed to integrator
        :param variables_from_optimizer: allows passing variables (as a dict from the optimizer; e.g., the current iteration)
        :param extra_var: extra variable
        :param dim_info: the input x can be a tensor concatenated by several variables along channel, dim_info is a list indicates the dim of each variable,
        """
        super(ODEWrapFunc, self).__init__()
        self.nested_class = nested_class
        """the model to be integrated"""
        self.pars = pars
        """ParameterDict, settings passed to integrator"""
        self.variables_from_optimizer = variables_from_optimizer
        """allows passing variables (as a dict from the optimizer; e.g., the current iteration)"""
        self.extra_var = extra_var
        """extra variable"""
        self.has_combined_input = has_combined_input
        """the model has combined input in x e.g. EPDiff* equation, otherwise, model has individual input e.g. advect* , has x,u two inputs"""
        self.dim_info = dim_info
        """the input x can be a tensor concatenated by several variables along channel, dim_info is a list indicates the dim of each variable"""
        self.opt_param = None

    def set_dim_info(self, dim_info):
        self.dim_info = [0] + list(np.cumsum(dim_info))

    def set_opt_param(self, opt_param):
        self.opt_param = opt_param

    def set_debug_mode_on(self):
        self.nested_class.debug_mode_on = True

    def factor_input(self, y):
        x = [y[:, self.dim_info[ind]:self.dim_info[ind + 1], ...] for ind in range(len(self.dim_info) - 1)]
        if not self.has_combined_input:
            u = x[0]
            x = x[1:]
        else:
            u = None
        return u, x

    @staticmethod
    def factor_res(u, res):
        if u is not None:
            res = torch.cat((torch.zeros_like(u), *res), 1)
        else:
            res = torch.cat(res, 1)
        return res

    def forward(self, t, y):
        u, x = self.factor_input(y)
        res = self.nested_class.f(t, x, u, pars=self.pars, variables_from_optimizer=self.variables_from_optimizer)
        res = self.factor_res(u, res)
        return res


class ODEWrapFunc_tuple(nn.Module):
    """
    a warp on tuple based torchdiffeq input
    """

    def __init__(self, nested_class, has_combined_input=False, pars=None, variables_from_optimizer=None, extra_var=None, dim_info=None):
        """

        :param nested_class: the model to be integrated
        :param has_combined_input: the model has combined input in x e.g. EPDiff* equation, otherwise, model has individual input e.g. advect* , has x,u two inputs
        :param pars: ParameterDict, settings passed to integrator
        :param variables_from_optimizer: allows passing variables (as a dict from the optimizer; e.g., the current iteration)
        :param extra_var: extra variable
        :param dim_info: not use in tuple version
        """
        super(ODEWrapFunc_tuple, self).__init__()
        self.nested_class = nested_class
        """ the model to be integrated"""
        self.pars = pars
        """ParameterDict, settings passed to integrator"""
        self.variables_from_optimizer = variables_from_optimizer
        """ allows passing variables (as a dict from the optimizer; e.g., the current iteration)"""
        self.extra_var = extra_var
        """extra variable"""
        self.has_combined_input = has_combined_input
        """ the model has combined input in x e.g. EPDiff* equation, otherwise, model has individual input e.g. advect* , has x,u two inputs"""
        self.dim_info = dim_info
        """not use in tuple version"""
        self.opt_param = None

    def set_dim_info(self, dim_info):
        self.dim_info = [0] + list(np.cumsum(dim_info))

    def set_opt_param(self, opt_param):
        self.opt_param = opt_param

    def set_debug_mode_on(self):
        self.nested_class.debug_mode_on = True

    def factor_input(self, y):
        if not self.has_combined_input:
            u = y[0]
            x = list(y[1:])
        else:
            x = list(y)
            u = None
        return u, x

    @staticmethod
    def factor_res(u, res):
        if u is not None:
            zero_grad = torch.zeros_like(u)
            zero_grad.requires_grad = res[0].requires_grad
            return zero_grad, *res
        else:
            return tuple(res)

    def forward(self, t, y):
        u, x = self.factor_input(y)
        res = self.nested_class.f(t, x, u, pars=self.pars, variables_from_optimizer=self.variables_from_optimizer)
        res = self.factor_res(u, res)
        return res


class STNFunction_ND_BCXYZ(Module):
    """
   Spatial transform function for 1D, 2D, and 3D. In BCXYZ format (this IS the format used in the current toolbox).
   """

    def __init__(self, spacing, zero_boundary=False, using_bilinear=True, using_01_input=True):
        """
        Constructor

        :param ndim: (int) spatial transformation of the transform
        """
        super(STNFunction_ND_BCXYZ, self).__init__()
        self.spacing = spacing
        self.ndim = len(spacing)
        self.zero_boundary = 'zeros' if zero_boundary else 'border'
        self.mode = 'bilinear' if using_bilinear else 'nearest'
        self.using_01_input = using_01_input

    def forward_stn(self, input1, input2, ndim):
        if ndim == 1:
            phi_rs = input2.reshape(list(input2.size()) + [1])
            input1_rs = input1.reshape(list(input1.size()) + [1])
            phi_rs_size = list(phi_rs.size())
            phi_rs_size[1] = 2
            phi_rs_ordered = torch.zeros(phi_rs_size, dtype=phi_rs.dtype, device=phi_rs.device)
            phi_rs_ordered[:, 1, ...] = phi_rs[:, 0, ...]
            output_rs = torch.nn.functional.grid_sample(input1_rs, phi_rs_ordered.permute([0, 2, 3, 1]), mode=self.mode, padding_mode=self.zero_boundary, align_corners=True)
            output = output_rs[:, :, :, 0]
        if ndim == 2:
            input2_ordered = torch.zeros_like(input2)
            input2_ordered[:, 0, ...] = input2[:, 1, ...]
            input2_ordered[:, 1, ...] = input2[:, 0, ...]
            output = torch.nn.functional.grid_sample(input1, input2_ordered.permute([0, 2, 3, 1]), mode=self.mode, padding_mode=self.zero_boundary, align_corners=True)
        if ndim == 3:
            input2_ordered = torch.zeros_like(input2)
            input2_ordered[:, 0, ...] = input2[:, 2, ...]
            input2_ordered[:, 1, ...] = input2[:, 1, ...]
            input2_ordered[:, 2, ...] = input2[:, 0, ...]
            output = torch.nn.functional.grid_sample(input1, input2_ordered.permute([0, 2, 3, 4, 1]), mode=self.mode, padding_mode=self.zero_boundary, align_corners=True)
        return output

    def forward(self, input1, input2):
        """
        Perform the actual spatial transform

        :param input1: image in BCXYZ format
        :param input2: spatial transform in BdimXYZ format
        :return: spatially transformed image in BCXYZ format
        """
        assert len(self.spacing) + 2 == len(input2.size())
        if self.using_01_input:
            output = self.forward_stn(input1, map_scale_utils.scale_map(input2, self.spacing), self.ndim)
        else:
            output = self.forward_stn(input1, input2, self.ndim)
        return output


class AsymConv(Module):
    """
    the implementation of location-dependent convolution method
    the input should include a BxCxXxYxZ image, a 1x1xHxWxDxK3 kernel, where each location has a corresponding filter
    we would extend into guassian based convolution method
    the input should include a BxCxXxYxZ image, a 1x1xXxYxZ kernel, where each location refers to the deviation of a gaussian filter
    the implementaiton is based on img2col convolution
    """

    def __init__(self, kernel_size):
        super(AsymConv, self).__init__()
        self.k_sz = kernel_size

    def forward(self, X, W):
        assert self.k_sz % 2 == 1, 'the kernel size must be odd'
        hk_sz = self.k_sz // 2
        X = F.pad(X, (hk_sz, hk_sz, hk_sz, hk_sz, hk_sz, hk_sz), 'replicate')
        None
        X_col = X.unfold(2, self.k_sz, 1).unfold(3, self.k_sz, 1).unfold(4, self.k_sz, 1)
        None
        dim_B, dim_C, dim_X, dim_Y, dim_Z, K, _, _ = X_col.shape
        X_col = X_col.contiguous().view(dim_B, dim_C, dim_X, dim_Y, dim_Z, -1)
        None
        res = X_col * W
        res = res.sum(5)
        None
        return res


class STN_ND_BCXYZ(Module):
    """
    Spatial transform code for nD spatial transoforms. Uses the BCXYZ image format.
    """

    def __init__(self, spacing, zero_boundary=False, use_bilinear=True, use_01_input=True, use_compile_version=False):
        super(STN_ND_BCXYZ, self).__init__()
        self.spacing = spacing
        """spatial dimension"""
        if use_compile_version:
            if use_bilinear:
                self.f = STNFunction_ND_BCXYZ_Compile(self.spacing, zero_boundary)
            else:
                self.f = partial(get_nn_interpolation, spacing=self.spacing)
        else:
            self.f = STNFunction_ND_BCXYZ(self.spacing, zero_boundary=zero_boundary, using_bilinear=use_bilinear, using_01_input=use_01_input)
        """spatial transform function"""

    def forward(self, input1, input2):
        """
       Simply returns the transformed input

       :param input1: image in BCXYZ format 
       :param input2: map in BdimXYZ format
       :return: returns the transformed image
       """
        return self.f(input1, input2)


class NoisyLinear(nn.Module):
    """Applies a noisy linear transformation to the incoming data: :math:`y = (mu_w + sigma_w \\cdot epsilon_w)x + mu_b + sigma_b \\cdot epsilon_b`
    More details can be found in the paper `ZZ` _ .
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias. Default: True
        factorised: whether or not to use factorised noise. Default: True
        std_init: initialization constant for standard deviation component of weights. If None,             defaults to 0.017 for independent and 0.4 for factorised. Default: None

    Shape:
        - Input: (N, in_features)
        - Output:(N, out_features)

    Attributes:
        weight: the learnable weights of the module of shape (out_features x in_features)
        bias:   the learnable bias of the module of shape (out_features)

    Examples::
        >>> m = nn.NoisyLinear(20, 30)
        >>> input = autograd.Variable(torch.randn(128, 20))
        >>> output = m(input)
        >>> print(output.size())
    """

    def __init__(self, in_features, out_features, bias=True, factorised=True, std_init=None):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.factorised = factorised
        self.weight_mu = Parameter(MyTensor(out_features, in_features))
        self.weight_sigma = Parameter(MyTensor(out_features, in_features))
        if bias:
            self.bias_mu = Parameter(MyTensor(out_features))
            self.bias_sigma = Parameter(MyTensor(out_features))
        else:
            self.register_parameter('bias', None)
        if std_init is None:
            if self.factorised:
                self.std_init = 0.4
            else:
                self.std_init = 0.017
        else:
            self.std_init = std_init
        self.reset_parameters(bias)

    def reset_parameters(self, bias):
        if self.factorised:
            mu_range = 1.0 / math.sqrt(self.weight_mu.size(1))
            self.weight_mu.data.uniform_(-mu_range, mu_range)
            self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))
            if bias:
                self.bias_mu.data.uniform_(-mu_range, mu_range)
                self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))
        else:
            mu_range = math.sqrt(3.0 / self.weight_mu.size(1))
            self.weight_mu.data.uniform_(-mu_range, mu_range)
            self.weight_sigma.data.fill_(self.std_init)
            if bias:
                self.bias_mu.data.uniform_(-mu_range, mu_range)
                self.bias_sigma.data.fill_(self.std_init)

    def scale_noise(self, size):
        x = MyTensor(size).normal_()
        x = x.sign().mul(x.abs().sqrt())
        return x

    def forward(self, input):
        if self.factorised:
            epsilon_in = self.scale_noise(self.in_features)
            epsilon_out = self.scale_noise(self.out_features)
            weight_epsilon = epsilon_out.ger(epsilon_in)
            bias_epsilon = self.scale_noise(self.out_features)
        else:
            weight_epsilon = MyTensor(*(self.out_features, self.in_features)).normal_()
            bias_epsilon = MyTensor(self.out_features).normal_()
        return F.linear(input, self.weight_mu + self.weight_sigma.mul(weight_epsilon), self.bias_mu + self.bias_sigma.mul(bias_epsilon))

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class NoisyLayer(nn.Module):

    def __init__(self, std_init=None, start_reducing_from_iter=25):
        super(NoisyLayer, self).__init__()
        self.std_init = std_init
        if self.std_init is None:
            self.std_init = 0.25
        else:
            self.std_init = std_init
        self.start_reducing_from_iter = start_reducing_from_iter

    def forward(self, input, iter=0):
        noise_epsilon = MyTensor(input.size()).normal_()
        if self.training:
            effective_iter = max(0, iter - self.start_reducing_from_iter)
            output = input + 1.0 / (effective_iter + 1) * self.std_init * noise_epsilon
        else:
            output = input
        return output


class _NoisyConvNd(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, transposed, output_padding, groups, bias, scalar_sigmas=True, optimize_sigmas=False, std_init=None, start_reducing_from_iter=25):
        super(_NoisyConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.scalar_sigmas = scalar_sigmas
        self.optimize_sigmas = optimize_sigmas
        self.std_init = std_init
        self.start_reducing_from_iter = start_reducing_from_iter
        if self.std_init is None:
            self.std_init = 0.25
        else:
            self.std_init = std_init
        if transposed:
            self.weight = Parameter(MyTensor(in_channels, out_channels // groups, *kernel_size))
            if self.scalar_sigmas:
                if self.optimize_sigmas:
                    self.weight_sigma = Parameter(MyTensor(1))
                else:
                    self.weight_sigma = MyTensor(1)
            elif self.optimize_sigmas:
                self.weight_sigma = Parameter(MyTensor(in_channels, out_channels // groups))
            else:
                self.weight_sigma = MyTensor(in_channels, out_channels // groups)
        else:
            self.weight = Parameter(MyTensor(out_channels, in_channels // groups, *kernel_size))
            if self.scalar_sigmas:
                if self.optimize_sigmas:
                    self.weight_sigma = Parameter(MyTensor(1))
                else:
                    self.weight_sigma = MyTensor(1)
            elif self.optimize_sigmas:
                self.weight_sigma = Parameter(MyTensor(out_channels, in_channels // groups))
            else:
                self.weight_sigma = MyTensor(out_channels, in_channels // groups)
        if bias:
            self.bias = Parameter(MyTensor(out_channels))
            if self.scalar_sigmas:
                if self.optimize_sigmas:
                    self.bias_sigma = Parameter(MyTensor(1))
                else:
                    self.bias_sigma = MyTensor(1)
            elif self.optimize_sigmas:
                self.bias_sigma = Parameter(MyTensor(out_channels))
            else:
                self.bias_sigma = MyTensor(out_channels)
        else:
            self.register_parameter('bias', None)
            self.register_parameter('bias_sigma', None)
        self.reset_parameters(bias)

    def reset_parameters(self, bias):
        nn.init.kaiming_normal_(self.weight.data)
        self.weight_sigma.data.fill_(self.std_init)
        if bias:
            self.bias.data.fill_(0)
            self.bias_sigma.data.fill_(self.std_init)

    def extra_repr(self):
        s = '{in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}'
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)


class NoisyConv1d(_NoisyConvNd):
    """Applies a 1D noisy convolution over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size
    :math:`(N, C_{in}, L)` and output :math:`(N, C_{out}, L_{out})` can be
    precisely described as:

    .. math::

        \\begin{equation*}
        \\text{out}(N_i, C_{out_j}) = \\text{bias}(C_{out_j}) +
                                \\sum_{k = 0}^{C_{in} - 1} \\text{weight}(C_{out_j}, k) \\star \\text{input}(N_i, k)
        \\end{equation*},

    where :math:`\\star` is the valid `cross-correlation`_ operator,
    :math:`N` is a batch size, :math:`C` denotes a number of channels,
    :math:`L` is a length of signal sequence.

    * :attr:`stride` controls the stride for the cross-correlation, a single
      number or a one-element tuple.

    * :attr:`padding` controls the amount of implicit zero-paddings on both sides
      for :attr:`padding` number of points.

    * :attr:`dilation` controls the spacing between the kernel points; also
      known as the à trous algorithm. It is harder to describe, but this `link`_
      has a nice visualization of what :attr:`dilation` does.

    * :attr:`groups` controls the connections between inputs and outputs.
      :attr:`in_channels` and :attr:`out_channels` must both be divisible by
      :attr:`groups`. For example,

        * At groups=1, all inputs are convolved to all outputs.
        * At groups=2, the operation becomes equivalent to having two conv
          layers side by side, each seeing half the input channels,
          and producing half the output channels, and both subsequently
          concatenated.
        * At groups= :attr:`in_channels`, each input channel is convolved with
          its own set of filters (of size
          :math:`\\left\\lfloor \\frac{\\text{out_channels}}{\\text{in_channels}} \\right\\rfloor`).

    .. note::

         Depending of the size of your kernel, several (of the last)
         columns of the input might be lost, because it is a valid
         `cross-correlation`_, and not a full `cross-correlation`_.
         It is up to the user to add proper padding.

    .. note::

         The configuration when `groups == in_channels` and `out_channels == K * in_channels`
         where `K` is a positive integer is termed in literature as depthwise convolution.

         In other words, for an input of size :math:`(N, C_{in}, L_{in})`, if you want a
         depthwise convolution with a depthwise multiplier `K`,
         then you use the constructor arguments
         :math:`(\\text{in_channels}=C_{in}, \\text{out_channels}=C_{in} * K, ..., \\text{groups}=C_{in})`

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of
            the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel
            elements. Default: 1
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``

    Shape:
        - Input: :math:`(N, C_{in}, L_{in})`
        - Output: :math:`(N, C_{out}, L_{out})` where

          .. math::
              L_{out} = \\left\\lfloor\\frac{L_{in} + 2 \\times \\text{padding} - \\text{dilation}
                        \\times (\\text{kernel_size} - 1) - 1}{\\text{stride}} + 1\\right\\rfloor

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
            (out_channels, in_channels, kernel_size)
        bias (Tensor):   the learnable bias of the module of shape
            (out_channels)

    Examples::

        >>> m = nn.NoisyConv1d(16, 33, 3, stride=2)
        >>> input = torch.randn(20, 16, 50)
        >>> output = m(input)

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, scalar_sigmas=True, optimize_sigmas=False, std_init=None, start_reducing_from_iter=25):
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        super(NoisyConv1d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, False, _single(0), groups, bias, scalar_sigmas, optimize_sigmas, std_init, start_reducing_from_iter)

    def forward(self, input, iter=0):
        weight_epsilon = MyTensor(*(self.out_channels, self.in_channels, *self.kernel_size)).normal_()
        bias_epsilon = MyTensor(self.out_channels).normal_()
        if self.bias is not None:
            if self.training:
                effective_iter = max(0, iter - self.start_reducing_from_iter)
                new_bias_value = self.bias + 1.0 / (effective_iter + 1) * self.bias_sigma * bias_epsilon
            else:
                new_bias_value = self.bias
        else:
            new_bias_value = None
        if self.scalar_sigmas:
            if self.optimize_sigmas:
                if self.bias is not None:
                    None
                else:
                    None
            if self.training:
                effective_iter = max(0, iter - self.start_reducing_from_iter)
                new_weight_value = self.weight + 1.0 / (effective_iter + 1) * self.weight_sigma * weight_epsilon
            else:
                new_weight_value = self.weight
            return F.conv1d(input, new_weight_value, new_bias_value, self.stride, self.padding, self.dilation, self.groups)
        else:
            if self.training:
                delta_weight = weight_epsilon
                sz = self.weight_sigma.size()
                for i in range(sz[0]):
                    delta_weight[i, ...] *= self.weight_sigma[i]
                effective_iter = max(0, iter - self.start_reducing_from_iter)
                new_weight_value = self.weight + 1.0 / (effective_iter + 1) * delta_weight
            else:
                new_weight_value = self.weight
            return F.conv1d(input, new_weight_value, new_bias_value, self.stride, self.padding, self.dilation, self.groups)


class NoisyConv2d(_NoisyConvNd):
    """Applies a 2D noisy convolution over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size
    :math:`(N, C_{in}, H, W)` and output :math:`(N, C_{out}, H_{out}, W_{out})`
    can be precisely described as:

    .. math::

        \\begin{equation*}
        \\text{out}(N_i, C_{out_j}) = \\text{bias}(C_{out_j}) +
                                \\sum_{k = 0}^{C_{in} - 1} \\text{weight}(C_{out_j}, k) \\star \\text{input}(N_i, k)
        \\end{equation*},

    where :math:`\\star` is the valid 2D `cross-correlation`_ operator,
    :math:`N` is a batch size, :math:`C` denotes a number of channels,
    :math:`H` is a height of input planes in pixels, and :math:`W` is
    width in pixels.

    * :attr:`stride` controls the stride for the cross-correlation, a single
      number or a tuple.

    * :attr:`padding` controls the amount of implicit zero-paddings on both
      sides for :attr:`padding` number of points for each dimension.

    * :attr:`dilation` controls the spacing between the kernel points; also
      known as the à trous algorithm. It is harder to describe, but this `link`_
      has a nice visualization of what :attr:`dilation` does.

    * :attr:`groups` controls the connections between inputs and outputs.
      :attr:`in_channels` and :attr:`out_channels` must both be divisible by
      :attr:`groups`. For example,

        * At groups=1, all inputs are convolved to all outputs.
        * At groups=2, the operation becomes equivalent to having two conv
          layers side by side, each seeing half the input channels,
          and producing half the output channels, and both subsequently
          concatenated.
        * At groups= :attr:`in_channels`, each input channel is convolved with
          its own set of filters (of size
          :math:`\\left\\lfloor\\frac{\\text{out_channels}}{\\text{in_channels}}\\right\\rfloor`).

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

    .. note::

         Depending of the size of your kernel, several (of the last)
         columns of the input might be lost, because it is a valid `cross-correlation`_,
         and not a full `cross-correlation`_.
         It is up to the user to add proper padding.

    .. note::

         The configuration when `groups == in_channels` and `out_channels == K * in_channels`
         where `K` is a positive integer is termed in literature as depthwise convolution.

         In other words, for an input of size :math:`(N, C_{in}, H_{in}, W_{in})`, if you want a
         depthwise convolution with a depthwise multiplier `K`,
         then you use the constructor arguments
         :math:`(\\text{in_channels}=C_{in}, \\text{out_channels}=C_{in} * K, ..., \\text{groups}=C_{in})`

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` where

          .. math::
              H_{out} = \\left\\lfloor\\frac{H_{in}  + 2 \\times \\text{padding}[0] - \\text{dilation}[0]
                        \\times (\\text{kernel_size}[0] - 1) - 1}{\\text{stride}[0]} + 1\\right\\rfloor

              W_{out} = \\left\\lfloor\\frac{W_{in}  + 2 \\times \\text{padding}[1] - \\text{dilation}[1]
                        \\times (\\text{kernel_size}[1] - 1) - 1}{\\text{stride}[1]} + 1\\right\\rfloor

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
                         (out_channels, in_channels, kernel_size[0], kernel_size[1])
        bias (Tensor):   the learnable bias of the module of shape (out_channels)

    Examples::

        >>> # With square kernels and equal stride
        >>> m = nn.NoisyConv2d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.NoisyConv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> # non-square kernels and unequal stride and with padding and dilation
        >>> m = nn.NoisyConv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
        >>> input = torch.randn(20, 16, 50, 100)
        >>> output = m(input)

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, scalar_sigmas=True, optimize_sigmas=False, std_init=None, start_reducing_from_iter=25):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(NoisyConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, False, _pair(0), groups, bias, scalar_sigmas, optimize_sigmas, std_init, start_reducing_from_iter)

    def forward(self, input, iter=0):
        weight_epsilon = MyTensor(*(self.out_channels, self.in_channels, *self.kernel_size)).normal_()
        bias_epsilon = MyTensor(self.out_channels).normal_()
        if self.bias is not None:
            if self.training:
                effective_iter = max(0, iter - self.start_reducing_from_iter)
                new_bias_value = self.bias + 1.0 / (effective_iter + 1) * self.bias_sigma * bias_epsilon
            else:
                new_bias_value = self.bias
        else:
            new_bias_value = None
        if self.scalar_sigmas:
            if self.optimize_sigmas:
                if self.bias is not None:
                    None
                else:
                    None
            if self.training:
                effective_iter = max(0, iter - self.start_reducing_from_iter)
                new_weight_value = self.weight + 1.0 / (effective_iter + 1) * self.weight_sigma * weight_epsilon
            else:
                new_weight_value = self.weight
            return F.conv2d(input, new_weight_value, new_bias_value, self.stride, self.padding, self.dilation, self.groups)
        else:
            if self.training:
                delta_weight = weight_epsilon
                sz = self.weight_sigma.size()
                for i in range(sz[0]):
                    for j in range(sz[1]):
                        delta_weight[i, j, ...] *= self.weight_sigma[i, j]
                effective_iter = max(0, iter - self.start_reducing_from_iter)
                new_weight_value = self.weight + 1.0 / (effective_iter + 1) * delta_weight
            else:
                new_weight_value = self.weight
            return F.conv2d(input, new_weight_value, new_bias_value, self.stride, self.padding, self.dilation, self.groups)


class NoisyConv3d(_NoisyConvNd):
    """Applies a 3D noisy convolution over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C_{in}, D, H, W)`
    and output :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})` can be precisely described as:

    .. math::

        \\begin{equation*}
        \\text{out}(N_i, C_{out_j}) = \\text{bias}(C_{out_j}) +
                                \\sum_{k = 0}^{C_{in} - 1} \\text{weight}(C_{out_j}, k) \\star \\text{input}(N_i, k)
        \\end{equation*},

    where :math:`\\star` is the valid 3D `cross-correlation`_ operator

    * :attr:`stride` controls the stride for the cross-correlation.

    * :attr:`padding` controls the amount of implicit zero-paddings on both
      sides for :attr:`padding` number of points for each dimension.

    * :attr:`dilation` controls the spacing between the kernel points; also known as the à trous algorithm.
      It is harder to describe, but this `link`_ has a nice visualization of what :attr:`dilation` does.

    * :attr:`groups` controls the connections between inputs and outputs.
      :attr:`in_channels` and :attr:`out_channels` must both be divisible by
      :attr:`groups`. For example,

        * At groups=1, all inputs are convolved to all outputs.
        * At groups=2, the operation becomes equivalent to having two conv
          layers side by side, each seeing half the input channels,
          and producing half the output channels, and both subsequently
          concatenated.
        * At groups= :attr:`in_channels`, each input channel is convolved with
          its own set of filters (of size
          :math:`\\left\\lfloor\\frac{\\text{out_channels}}{\\text{in_channels}}\\right\\rfloor`).

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for the depth, height and width dimension
        - a ``tuple`` of three ints -- in which case, the first `int` is used for the depth dimension,
          the second `int` for the height dimension and the third `int` for the width dimension

    .. note::

         Depending of the size of your kernel, several (of the last)
         columns of the input might be lost, because it is a valid `cross-correlation`_,
         and not a full `cross-correlation`_.
         It is up to the user to add proper padding.

    .. note::

         The configuration when `groups == in_channels` and `out_channels == K * in_channels`
         where `K` is a positive integer is termed in literature as depthwise convolution.

         In other words, for an input of size :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`, if you want a
         depthwise convolution with a depthwise multiplier `K`,
         then you use the constructor arguments
         :math:`(\\text{in_channels}=C_{in}, \\text{out_channels}=C_{in} * K, ..., \\text{groups}=C_{in})`

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to all three sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``

    Shape:
        - Input: :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})` where

          .. math::
              D_{out} = \\left\\lfloor\\frac{D_{in} + 2 \\times \\text{padding}[0] - \\text{dilation}[0]
                    \\times (\\text{kernel_size}[0] - 1) - 1}{\\text{stride}[0]} + 1\\right\\rfloor

              H_{out} = \\left\\lfloor\\frac{H_{in} + 2 \\times \\text{padding}[1] - \\text{dilation}[1]
                    \\times (\\text{kernel_size}[1] - 1) - 1}{\\text{stride}[1]} + 1\\right\\rfloor

              W_{out} = \\left\\lfloor\\frac{W_{in} + 2 \\times \\text{padding}[2] - \\text{dilation}[2]
                    \\times (\\text{kernel_size}[2] - 1) - 1}{\\text{stride}[2]} + 1\\right\\rfloor

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
                         (out_channels, in_channels, kernel_size[0], kernel_size[1], kernel_size[2])
        bias (Tensor):   the learnable bias of the module of shape (out_channels)

    Examples::

        >>> # With square kernels and equal stride
        >>> m = nn.NoisyConv3d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.NoisyConv3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(4, 2, 0))
        >>> input = torch.randn(20, 16, 10, 50, 100)
        >>> output = m(input)

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, scalar_sigmas=True, optimize_sigmas=False, std_init=None, start_reducing_from_iter=25):
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)
        super(NoisyConv3d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, False, _triple(0), groups, bias, scalar_sigmas, optimize_sigmas, std_init, start_reducing_from_iter)

    def forward(self, input):
        weight_epsilon = MyTensor(*(self.out_channels, self.in_channels, *self.kernel_size)).normal_()
        bias_epsilon = MyTensor(self.out_channels).normal_()
        if self.bias is not None:
            if self.training:
                effective_iter = max(0, iter - self.start_reducing_from_iter)
                new_bias_value = self.bias + 1.0 / (effective_iter + 1) * self.bias_sigma * bias_epsilon
            else:
                new_bias_value = self.bias
        else:
            new_bias_value = None
        if self.scalar_sigmas:
            if self.optimize_sigmas:
                if self.bias is not None:
                    None
                else:
                    None
            if self.training:
                effective_iter = max(0, iter - self.start_reducing_from_iter)
                new_weight_value = self.weight + 1.0 / (effective_iter + 1) * self.weight_sigma * weight_epsilon
            else:
                new_weight_value = self.weight
            return F.conv3d(input, new_weight_value, new_bias_value, self.stride, self.padding, self.dilation, self.groups)
        else:
            if self.training:
                delta_weight = weight_epsilon
                sz = self.weight_sigma.size()
                for i in range(sz[0]):
                    for j in range(sz[1]):
                        for k in range(sz[2]):
                            delta_weight[i, j, k, ...] *= self.weight_sigma[i, j, k]
                effective_iter = max(0, iter - self.start_reducing_from_iter)
                new_weight_value = self.weight + 1.0 / (effective_iter + 1) * delta_weight
            else:
                new_weight_value = self.weight
            return F.conv3d(input, new_weight_value, new_bias_value, self.stride, self.padding, self.dilation, self.groups)


class _NoisyConvTransposeMixin(object):

    def forward(self, input, output_size=None, iter=0):
        output_padding = self._output_padding(input, output_size)
        func = self._backend.ConvNd(self.stride, self.padding, self.dilation, self.transposed, output_padding, self.groups)
        if self.bias is None:
            return func(input, self.weight)
        else:
            return func(input, self.weight, self.bias)

    def _output_padding(self, input, output_size):
        if output_size is None:
            return self.output_padding
        output_size = list(output_size)
        k = input.dim() - 2
        if len(output_size) == k + 2:
            output_size = output_size[-2:]
        if len(output_size) != k:
            raise ValueError('output_size must have {} or {} elements (got {})'.format(k, k + 2, len(output_size)))

        def dim_size(d):
            return (input.size(d + 2) - 1) * self.stride[d] - 2 * self.padding[d] + self.kernel_size[d]
        min_sizes = [dim_size(d) for d in range(k)]
        max_sizes = [(min_sizes[d] + self.stride[d] - 1) for d in range(k)]
        for size, min_size, max_size in zip(output_size, min_sizes, max_sizes):
            if size < min_size or size > max_size:
                raise ValueError('requested an output size of {}, but valid sizes range from {} to {} (for an input of {})'.format(output_size, min_sizes, max_sizes, input.size()[2:]))
        return tuple([(output_size[d] - min_sizes[d]) for d in range(k)])


class NoisyConvTranspose1d(_NoisyConvTransposeMixin, _NoisyConvNd):
    """Applies a 1D noisy transposed convolution operator over an input image
    composed of several input planes.

    This module can be seen as the gradient of Conv1d with respect to its input.
    It is also known as a fractionally-strided convolution or
    a deconvolution (although it is not an actual deconvolution operation).

    * :attr:`stride` controls the stride for the cross-correlation.

    * :attr:`padding` controls the amount of implicit zero-paddings on both
      sides for ``kernel_size - 1 - padding`` number of points. See note
      below for details.

    * :attr:`output_padding` controls the additional size added to one side
      of the output shape. See note below for details.

    * :attr:`dilation` controls the spacing between the kernel points; also known as the à trous algorithm.
      It is harder to describe, but this `link`_ has a nice visualization of what :attr:`dilation` does.

    * :attr:`groups` controls the connections between inputs and outputs.
      :attr:`in_channels` and :attr:`out_channels` must both be divisible by
      :attr:`groups`. For example,

        * At groups=1, all inputs are convolved to all outputs.
        * At groups=2, the operation becomes equivalent to having two conv
          layers side by side, each seeing half the input channels,
          and producing half the output channels, and both subsequently
          concatenated.
        * At groups= :attr:`in_channels`, each input channel is convolved with
          its own set of filters (of size
          :math:`\\left\\lfloor\\frac{\\text{out_channels}}{\\text{in_channels}}\\right\\rfloor`).

    .. note::

         Depending of the size of your kernel, several (of the last)
         columns of the input might be lost, because it is a valid `cross-correlation`_,
         and not a full `cross-correlation`_.
         It is up to the user to add proper padding.

    .. note::
        The :attr:`padding` argument effectively adds ``kernel_size - 1 - padding``
        amount of zero padding to both sizes of the input. This is set so that
        when a :class:`~torch.nn.Conv1d` and a :class:`~torch.nn.ConvTranspose1d`
        are initialized with same parameters, they are inverses of each other in
        regard to the input and output shapes. However, when ``stride > 1``,
        :class:`~torch.nn.Conv1d` maps multiple input shapes to the same output
        shape. :attr:`output_padding` is provided to resolve this ambiguity by
        effectively increasing the calculated output shape on one side. Note
        that :attr:`output_padding` is only used to find output shape, but does
        not actually add zero-padding to output.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): ``kernel_size - 1 - padding`` zero-padding
            will be added to both sides of the input. Default: 0
        output_padding (int or tuple, optional): Additional size added to one side
            of the output shape. Default: 0
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1

    Shape:
        - Input: :math:`(N, C_{in}, L_{in})`
        - Output: :math:`(N, C_{out}, L_{out})` where

          .. math::
              L_{out} = (L_{in} - 1) \\times \\text{stride} - 2 \\times \\text{padding}
                    + \\text{kernel_size} + \\text{output_padding}

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
                         (in_channels, out_channels, kernel_size[0], kernel_size[1])
        bias (Tensor):   the learnable bias of the module of shape (out_channels)
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, scalar_sigmas=True, optimize_sigmas=False, std_init=None, start_reducing_from_iter=25):
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        output_padding = _single(output_padding)
        super(NoisyConvTranspose1d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, True, output_padding, groups, bias, scalar_sigmas=scalar_sigmas, optimize_sigmas=optimize_sigmas, std_init=std_init, start_reducing_from_iter=start_reducing_from_iter)

    def forward(self, input, output_size=None, iter=0):
        output_padding = self._output_padding(input, output_size)
        weight_epsilon = MyTensor(*(self.out_channels, self.in_channels, *self.kernel_size)).normal_()
        bias_epsilon = MyTensor(self.out_channels).normal_()
        if self.bias is not None:
            if self.training:
                effective_iter = max(0, iter - self.start_reducing_from_iter)
                new_bias_value = self.bias + 1.0 / (effective_iter + 1) * self.bias_sigma * bias_epsilon
            else:
                new_bias_value = self.bias
        else:
            new_bias_value = None
        if self.scalar_sigmas:
            if self.optimize_sigmas:
                if self.bias is not None:
                    None
                else:
                    None
            if self.training:
                effective_iter = max(0, iter - self.start_reducing_from_iter)
                new_weight_value = self.weight + 1.0 / (1 + effective_iter) * self.weight_sigma * weight_epsilon
            else:
                new_weight_value = self.weight
            return F.conv_transpose1d(input, new_weight_value, new_bias_value, self.stride, self.padding, output_padding, self.groups, self.dilation)
        else:
            if self.training:
                delta_weight = weight_epsilon
                sz = self.weight_sigma.size()
                for i in range(sz[0]):
                    delta_weight[i, ...] *= self.weight_sigma[i]
                effective_iter = max(0, iter - self.start_reducing_from_iter)
                new_weight_value = self.weight + 1.0 / (1 + effective_iter) * self.weight_sigma * weight_epsilon
            else:
                new_weight_value = self.weight
            return F.conv_transpose1d(input, new_weight_value, new_bias_value, self.stride, self.padding, output_padding, self.groups, self.dilation)


class NoisyConvTranspose2d(_NoisyConvTransposeMixin, _NoisyConvNd):
    """Applies a 2D noisy transposed convolution operator over an input image
    composed of several input planes.

    This module can be seen as the gradient of Conv2d with respect to its input.
    It is also known as a fractionally-strided convolution or
    a deconvolution (although it is not an actual deconvolution operation).

    * :attr:`stride` controls the stride for the cross-correlation.

    * :attr:`padding` controls the amount of implicit zero-paddings on both
      sides for ``kernel_size - 1 - padding`` number of points. See note
      below for details.

    * :attr:`output_padding` controls the additional size added to one side
      of the output shape. See note below for details.

    * :attr:`dilation` controls the spacing between the kernel points; also known as the à trous algorithm.
      It is harder to describe, but this `link`_ has a nice visualization of what :attr:`dilation` does.

    * :attr:`groups` controls the connections between inputs and outputs.
      :attr:`in_channels` and :attr:`out_channels` must both be divisible by
      :attr:`groups`. For example,

        * At groups=1, all inputs are convolved to all outputs.
        * At groups=2, the operation becomes equivalent to having two conv
          layers side by side, each seeing half the input channels,
          and producing half the output channels, and both subsequently
          concatenated.
        * At groups= :attr:`in_channels`, each input channel is convolved with
          its own set of filters (of size
          :math:`\\left\\lfloor\\frac{\\text{out_channels}}{\\text{in_channels}}\\right\\rfloor`).

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`output_padding`
    can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimensions
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

    .. note::

         Depending of the size of your kernel, several (of the last)
         columns of the input might be lost, because it is a valid `cross-correlation`_,
         and not a full `cross-correlation`_.
         It is up to the user to add proper padding.

    .. note::
        The :attr:`padding` argument effectively adds ``kernel_size - 1 - padding``
        amount of zero padding to both sizes of the input. This is set so that
        when a :class:`~torch.nn.Conv2d` and a :class:`~torch.nn.ConvTranspose2d`
        are initialized with same parameters, they are inverses of each other in
        regard to the input and output shapes. However, when ``stride > 1``,
        :class:`~torch.nn.Conv2d` maps multiple input shapes to the same output
        shape. :attr:`output_padding` is provided to resolve this ambiguity by
        effectively increasing the calculated output shape on one side. Note
        that :attr:`output_padding` is only used to find output shape, but does
        not actually add zero-padding to output.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): ``kernel_size - 1 - padding`` zero-padding
            will be added to both sides of each dimension in the input. Default: 0
        output_padding (int or tuple, optional): Additional size added to one side
            of each dimension in the output shape. Default: 0
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` where

          .. math::
              H_{out} = (H_{in} - 1) \\times \\text{stride}[0] - 2 \\times \\text{padding}[0]
                    + \\text{kernel_size}[0] + \\text{output_padding}[0]

              W_{out} = (W_{in} - 1) \\times \\text{stride}[1] - 2 \\times \\text{padding}[1]
                    + \\text{kernel_size}[1] + \\text{output_padding}[1]

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
                         (in_channels, out_channels, kernel_size[0], kernel_size[1])
        bias (Tensor):   the learnable bias of the module of shape (out_channels)

    Examples::

        >>> # With square kernels and equal stride
        >>> m = nn.NoisyConvTranspose2d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.NoisyConvTranspose2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> input = torch.randn(20, 16, 50, 100)
        >>> output = m(input)
        >>> # exact output size can be also specified as an argument
        >>> input = torch.randn(1, 16, 12, 12)
        >>> downsample = nn.NoisyConv2d(16, 16, 3, stride=2, padding=1)
        >>> upsample = nn.NoisyConvTranspose2d(16, 16, 3, stride=2, padding=1)
        >>> h = downsample(input)
        >>> h.size()
        torch.Size([1, 16, 6, 6])
        >>> output = upsample(h, output_size=input.size())
        >>> output.size()
        torch.Size([1, 16, 12, 12])

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, scalar_sigmas=True, optimize_sigmas=False, std_init=None, start_reducing_from_iter=25):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        output_padding = _pair(output_padding)
        super(NoisyConvTranspose2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, True, output_padding, groups, bias, scalar_sigmas=scalar_sigmas, optimize_sigmas=optimize_sigmas, std_init=std_init, start_reducing_from_iter=start_reducing_from_iter)

    def forward(self, input, output_size=None, iter=0):
        output_padding = self._output_padding(input, output_size)
        weight_epsilon = MyTensor(*(self.out_channels, self.in_channels, *self.kernel_size)).normal_()
        bias_epsilon = MyTensor(self.out_channels).normal_()
        if self.bias is not None:
            if self.training:
                effective_iter = max(0, iter - self.start_reducing_from_iter)
                new_bias_value = self.bias + 1.0 / (1 + effective_iter) * self.bias_sigma * bias_epsilon
            else:
                new_bias_value = self.bias
        else:
            new_bias_value = None
        if self.scalar_sigmas:
            if self.optimize_sigmas:
                if self.bias is not None:
                    None
                else:
                    None
            if self.training:
                effective_iter = max(0, iter - self.start_reducing_from_iter)
                new_weight_value = self.weight + 1.0 / (1 + effective_iter) * self.weight_sigma * weight_epsilon
            else:
                new_weight_value = self.weight
            return F.conv_transpose2d(input, new_weight_value, new_bias_value, self.stride, self.padding, output_padding, self.groups, self.dilation)
        else:
            if self.training:
                delta_weight = weight_epsilon
                sz = self.weight_sigma.size()
                for i in range(sz[0]):
                    for j in range(sz[1]):
                        delta_weight[i, j, ...] *= self.weight_sigma[i, j]
                effective_iter = max(0, iter - self.start_reducing_from_iter)
                new_weight_value = self.weight + 1.0 / (1 + effective_iter) * self.weight_sigma * weight_epsilon
            else:
                new_weight_value = self.weight
            return F.conv_transpose2d(input, new_weight_value, new_bias_value, self.stride, self.padding, output_padding, self.groups, self.dilation)


class NoisyConvTranspose3d(_NoisyConvTransposeMixin, _NoisyConvNd):
    """Applies a 3D noisy transposed convolution operator over an input image composed of several input
    planes.
    The transposed convolution operator multiplies each input value element-wise by a learnable kernel,
    and sums over the outputs from all input feature planes.

    This module can be seen as the gradient of Conv3d with respect to its input.
    It is also known as a fractionally-strided convolution or
    a deconvolution (although it is not an actual deconvolution operation).

    * :attr:`stride` controls the stride for the cross-correlation.

    * :attr:`padding` controls the amount of implicit zero-paddings on both
      sides for ``kernel_size - 1 - padding`` number of points. See note
      below for details.

    * :attr:`output_padding` controls the additional size added to one side
      of the output shape. See note below for details.

    * :attr:`dilation` controls the spacing between the kernel points; also known as the à trous algorithm.
      It is harder to describe, but this `link`_ has a nice visualization of what :attr:`dilation` does.

    * :attr:`groups` controls the connections between inputs and outputs.
      :attr:`in_channels` and :attr:`out_channels` must both be divisible by
      :attr:`groups`. For example,

        * At groups=1, all inputs are convolved to all outputs.
        * At groups=2, the operation becomes equivalent to having two conv
          layers side by side, each seeing half the input channels,
          and producing half the output channels, and both subsequently
          concatenated.
        * At groups= :attr:`in_channels`, each input channel is convolved with
          its own set of filters (of size
          :math:`\\left\\lfloor\\frac{\\text{out_channels}}{\\text{in_channels}}\\right\\rfloor`).

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`output_padding`
    can either be:

        - a single ``int`` -- in which case the same value is used for the depth, height and width dimensions
        - a ``tuple`` of three ints -- in which case, the first `int` is used for the depth dimension,
          the second `int` for the height dimension and the third `int` for the width dimension

    .. note::

         Depending of the size of your kernel, several (of the last)
         columns of the input might be lost, because it is a valid `cross-correlation`_,
         and not a full `cross-correlation`_.
         It is up to the user to add proper padding.

    .. note::
        The :attr:`padding` argument effectively adds ``kernel_size - 1 - padding``
        amount of zero padding to both sizes of the input. This is set so that
        when a :class:`~torch.nn.Conv3d` and a :class:`~torch.nn.ConvTranspose3d`
        are initialized with same parameters, they are inverses of each other in
        regard to the input and output shapes. However, when ``stride > 1``,
        :class:`~torch.nn.Conv3d` maps multiple input shapes to the same output
        shape. :attr:`output_padding` is provided to resolve this ambiguity by
        effectively increasing the calculated output shape on one side. Note
        that :attr:`output_padding` is only used to find output shape, but does
        not actually add zero-padding to output.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): ``kernel_size - 1 - padding`` zero-padding
            will be added to both sides of each dimension in the input. Default: 0
        output_padding (int or tuple, optional): Additional size added to one side
            of each dimension in the output shape. Default: 0
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1

    Shape:
        - Input: :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})` where

          .. math::
              D_{out} = (D_{in} - 1) \\times \\text{stride}[0] - 2 \\times \\text{padding}[0]
                    + \\text{kernel_size}[0] + \\text{output_padding}[0]

              H_{out} = (H_{in} - 1) \\times \\text{stride}[1] - 2 \\times \\text{padding}[1]
                    + \\text{kernel_size}[1] + \\text{output_padding}[1]

              W_{out} = (W_{in} - 1) \\times \\text{stride}[2] - 2 \\times \\text{padding}[2]
                    + \\text{kernel_size}[2] + \\text{output_padding}[2]

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
                         (in_channels, out_channels, kernel_size[0], kernel_size[1], kernel_size[2])
        bias (Tensor):   the learnable bias of the module of shape (out_channels)

    Examples::

        >>> # With square kernels and equal stride
        >>> m = nn.NoisyConvTranspose3d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.NoisyConv3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(0, 4, 2))
        >>> input = torch.randn(20, 16, 10, 50, 100)
        >>> output = m(input)

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, scalar_sigmas=True, optimize_sigmas=False, std_init=None, start_reducing_from_iter=25):
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)
        output_padding = _triple(output_padding)
        super(NoisyConvTranspose3d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, True, output_padding, groups, bias, scalar_sigmas=scalar_sigmas, optimize_sigmas=optimize_sigmas, std_init=std_init, start_reducing_from_iter=start_reducing_from_iter)

    def forward(self, input, output_size=None):
        output_padding = self._output_padding(input, output_size)
        weight_epsilon = MyTensor(*(self.out_channels, self.in_channels, *self.kernel_size)).normal_()
        bias_epsilon = MyTensor(self.out_channels).normal_()
        if self.bias is not None:
            if self.training:
                effective_iter = max(0, iter - self.start_reducing_from_iter)
                new_bias_value = self.bias + 1.0 / (effective_iter + 1) * self.bias_sigma * bias_epsilon
            else:
                new_bias_value = self.bias
        else:
            new_bias_value = None
        if self.scalar_sigmas:
            if self.optimize_sigmas:
                if self.bias is not None:
                    None
                else:
                    None
            if self.training:
                effective_iter = max(0, iter - self.start_reducing_from_iter)
                new_weight_value = self.weight + 1.0 / (effective_iter + 1) * self.weight_sigma * weight_epsilon
            else:
                new_weight_value = self.weight
            return F.conv_transpose3d(input, new_weight_value, new_bias_value, self.stride, self.padding, output_padding, self.groups, self.dilation)
        else:
            if self.training:
                delta_weight = weight_epsilon
                sz = self.weight_sigma.size()
                for i in range(sz[0]):
                    for j in range(sz[1]):
                        for k in range(sz[2]):
                            delta_weight[i, j, k, ...] *= self.weight_sigma[i, j, k]
                effective_iter = max(0, iter - self.start_reducing_from_iter)
                new_weight_value = self.weight + 1.0 / (effective_iter + 1) * self.weight_sigma * weight_epsilon
            else:
                new_weight_value = self.weight
            return F.conv_transpose3d(input, new_weight_value, new_bias_value, self.stride, self.padding, output_padding, self.groups, self.dilation)


class ODEBlock(nn.Module):
    """
    A interface class for torchdiffeq, https://github.com/rtqichen/torchdiffeq
    we add some constrains in torchdiffeq package to avoid collapse or traps, so this local version is recommended
    the solvers supported by the torchdiffeq are listed as following
    SOLVERS = {
    'explicit_adams': AdamsBashforth,
    'fixed_adams': AdamsBashforthMoulton,
    'adams': VariableCoefficientAdamsBashforth,
    'tsit5': Tsit5Solver,
    'dopri5': Dopri5Solver,
    'euler': Euler,
    'midpoint': Midpoint,
    'rk4': RK4,
}

    """

    def __init__(self, param=None):
        super(ODEBlock, self).__init__()
        self.odefunc = None
        """the ode problem to be solved"""
        tFrom = param['tFrom', 0.0, 'time to solve a model from']
        """time to solve a model from"""
        tTo = param['tTo', 1.0, 'time to solve a model to']
        """time to solve a model to"""
        self.integration_time = torch.Tensor([tFrom, tTo]).float()
        """intergration time, list, typically set as [0,1]"""
        self.method = param['solver', 'rk4', 'ode solver']
        """ solver,rk4 as default, supported list: explicit_adams,fixed_adams,tsit5,dopri5,euler,midpoint, rk4 """
        self.adjoin_on = param['adjoin_on', True, 'use adjoint optimization']
        """ adjoint method, benefits from memory consistency, which can be refer to "Neural Ordinary Differential Equations" """
        self.rtol = param['rtol', 1e-05, 'relative error tolerance for dopri5']
        """ relative error tolerance for dopri5"""
        self.atol = param['atol', 1e-05, 'absolute error tolerance for dopri5']
        """ absolute error tolerance for dopri5"""
        self.n_step = param['number_of_time_steps', 20, 'Number of time-steps to per unit time-interval integrate the ode']
        """ Number of time-steps to per unit time-interval integrate the PDE, for fixed time-step solver, i.e. rk4"""
        self.dt = 1.0 / self.n_step
        """time step, we assume integration time is from 0,1 so the step is 1/n_step"""

    def solve(self, x):
        return self.forward(x)

    def set_func(self, func):
        self.odefunc = func

    def get_dt(self):
        return self.dt

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x) if type(x) is not tuple else self.integration_time.type_as(x[0])
        odesolver = torchdiffeq.odeint_adjoint if self.adjoin_on else torchdiffeq.odeint
        try:
            out = odesolver(self.odefunc, x, self.integration_time, rtol=self.rtol, atol=self.atol, method=self.method, options={'step_size': self.dt})
        except:
            None
            self.odefunc.set_debug_mode_on()
            out = odesolver(self.odefunc, x, self.integration_time, rtol=self.rtol, atol=self.atol, method=self.method, options={'step_size': self.dt})
        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


class ODEWrapBlock(nn.Module):
    """
    A warp on ODE method, providing interface for embedded rungekutta sovler and for torchdiffeq solver
    """

    def __init__(self, model, cparams=None, use_odeint=True, use_ode_tuple=False, tFrom=0.0, tTo=1.0):
        """

        :param model: the ode/pde model to be solved
        :param cparams: ParameterDict, the model settings
        :param use_odeint: if true, use torchdiffeq, else, use embedded rungekutta (rk4)
        :param use_ode_tuple: assume torchdiffeq is used, if use_ode_tuple, take the tuple as the solver input, else take a tensor as the solver input
        :param tFrom: start time point, typically 0
        :param tTo: end time point, typically 1
        """
        super(ODEWrapBlock, self).__init__()
        self.model = model
        """ the ode/pde model to be solved"""
        self.cparams = cparams
        """ParameterDict, the model settings"""
        self.use_odeint = use_odeint
        """if true, use torchdiffeq, else, use embedded rungekutta (rk4)"""
        self.use_ode_tuple = use_ode_tuple
        """assume torchdiffeq is used, if use_ode_tuple, take the tuple as the solver input, else take a tensor as the solver input """
        self.integrator = None
        """if use_odeint, then intergrator from torchdiffeq is used else use the embedded rk4 intergrator"""
        self.tFrom = tFrom
        """start time point, typically 0"""
        self.tTo = tTo
        """ end time point, typically 1"""

    def get_dt(self):
        self.n_step = self.cparams['number_of_time_steps', 20, 'Number of time-steps to per unit time-interval integrate the PDE']
        self.dt = 1.0 / self.n_step
        return self.dt

    def init_solver(self, pars_to_pass_i, variables_from_optimizer, has_combined_input=False):
        if self.use_odeint:
            self.integrator = ODEBlock(self.cparams)
            wraped_func = FMW.ODEWrapFunc_tuple if self.use_ode_tuple else FMW.ODEWrapFunc
            func = wraped_func(self.model, has_combined_input=has_combined_input, pars=pars_to_pass_i, variables_from_optimizer=variables_from_optimizer)
            self.integrator.set_func(func)
        else:
            self.integrator = RK.RK4(self.model.f, self.model.u, pars_to_pass_i, self.cparams)
            self.integrator.set_pars(pars_to_pass_i)

    def solve_odeint(self, input_list):
        if self.use_ode_tuple:
            return self.solve_odeint_tuple(input_list)
        else:
            return self.solve_odeint_tensor(input_list)

    def solve_odeint_tensor(self, input_list):
        input_list_dim = [item.shape[1] for item in input_list]
        self.integrator.odefunc.set_dim_info(input_list_dim)
        input_tensor = torch.cat(tuple(input_list), 1)
        output_tensor = self.integrator.solve(input_tensor)
        dim_info = [0] + list(np.cumsum(input_list_dim))
        output_list = [output_tensor[:, dim_info[ind]:dim_info[ind + 1], ...] for ind in range(len(dim_info) - 1)]
        return output_list

    def solve_odeint_tuple(self, input_list):
        output_tuple = self.integrator.solve(tuple(input_list))
        return list(output_tuple)

    def solve_embedded_ode(self, input_list, variables_from_optimizer):
        return self.integrator.solve(input_list, self.tFrom, self.tTo, variables_from_optimizer)

    def solve(self, input_list, variables_from_optimizer):
        if self.use_odeint:
            return self.solve_odeint(input_list)
        else:
            return self.solve_embedded_ode(input_list, variables_from_optimizer)


class PerformSplineInterpolationHelper(Function):
    """
    Performs spline interpolation, given weights, indices, and coefficients.
    This is simply a convenience class which avoids computing the gradient of the actual interpolation via automatic differentiation
    (as this would be very memory intensive).
    """

    def __init__(self, index):
        """
        Constructor

        :param index: index array for interpolation (as computed from _compute_interpolation_weights)
        """
        super(PerformSplineInterpolationHelper, self).__init__()
        self.index = index

    def forward(self, c, weight):
        """
        Performs the interpolation for given coefficients and weights (we do not compute the gradient wrt. the indices)

        :param c: interpolation coefficients
        :param weight: interpolation weights
        :return: interpolated signal
        """
        sz_weight = weight.size()
        self.batch_size = c.size()[0]
        self.nr_of_channels = c.size()[1]
        self.n = sz_weight[0] - 1
        self.dim = sz_weight[2]
        self.c = c
        self.weight = weight
        w = MyTensor(*([self.batch_size, self.nr_of_channels] + list(sz_weight[3:]))).zero_()
        if self.dim == 1:
            for b in range(0, self.batch_size):
                b_ind = MyLongTensor(*list(self.index.size()[3:])).fill_(b)
                for ch in range(0, self.nr_of_channels):
                    ch_ind = MyLongTensor(*list(self.index.size()[3:])).fill_(ch)
                    for k1 in range(0, self.n + 1):
                        w[b, ch, ...] += weight[k1, b, 0, ...] * c[b_ind, ch_ind, self.index[k1, b, 0, ...]]
        elif self.dim == 2:
            for b in range(0, self.batch_size):
                b_ind = MyLongTensor(*list(self.index.size()[3:])).fill_(b)
                for ch in range(0, self.nr_of_channels):
                    ch_ind = MyLongTensor(*list(self.index.size()[3:])).fill_(ch)
                    for k1 in range(0, self.n + 1):
                        for k2 in range(0, self.n + 1):
                            w[b, ch, ...] += weight[k1, b, 0, ...] * weight[k2, b, 1, ...] * c[b_ind, ch_ind, self.index[k1, b, 0, ...], self.index[k2, b, 1, ...]]
        elif self.dim == 3:
            for b in range(0, self.batch_size):
                b_ind = MyLongTensor(*list(self.index.size()[3:])).fill_(b)
                for ch in range(0, self.nr_of_channels):
                    ch_ind = MyLongTensor(*list(self.index.size()[3:])).fill_(ch)
                    for k1 in range(0, self.n + 1):
                        for k2 in range(0, self.n + 1):
                            for k3 in range(0, self.n + 1):
                                w[b, ch, ...] += weight[k1, b, 0, ...] * weight[k2, b, 1, ...] * weight[k3, b, 2, ...] * c[b_ind, ch_ind, self.index[k1, b, 0, ...], self.index[k2, b, 1, ...], self.index[k3, b, 2, ...]]
        else:
            raise ValueError('Dimension needs to be 1, 2, or 3.')
        return w

    def _get_linear_view(self, t):
        """
        Takes a tensor and converts it to a linear view (needed for fast accumulation via put_)

        :param t: tensor
        :return: linearized view
        """
        lt = t.view(t.nelement())
        return lt

    def _sub2ind(self, indices, target_sz):
        """
        Similar to matlab's sub2ind. Converts ijk indices to a linear index

        :param indices: ijk indices (as a list)
        :param target_sz: target size to which these indices belong
        :return: linearized indices
        """
        aug_t_sz = list(target_sz) + [1]
        dim = len(indices)
        l_indices = MyLongTensor(indices[0].nelement()).zero_()
        for d in range(dim):
            l_indices += self._get_linear_view(indices[d]) * int(np.prod(aug_t_sz[d + 1:]))
        return l_indices

    def _accumulate(self, vals, indices, target_sz):
        """
        Necessary to compute the adjoint to the indexing into the coefficient array. Here we add entries based on where
        they were mapped from (via indexing).

        :param vals: Values
        :param indices: indices
        :param target_sz: target size
        :return: Returns accumulated values
        """
        acc_res = MyTensor(target_sz).zero_()
        l_acc_res = self._get_linear_view(acc_res)
        l_vals = self._get_linear_view(vals)
        l_indices = self._sub2ind(indices, target_sz)
        l_acc_res.put_(l_indices, l_vals, accumulate=True)
        return acc_res

    def backward(self, grad_output):
        """
        Computes the gradient with respect to the coefficent array and the weights

        :param grad_output: grad output from previous "layer"
        :return: gradient
        """
        grad_c = MyTensor(self.c.size()).zero_()
        grad_weight = MyTensor(self.weight.size()).zero_()
        if self.dim == 1:
            for b in range(0, self.batch_size):
                for k1 in range(0, self.n + 1):
                    for ch in range(0, self.nr_of_channels):
                        grad_weight[k1, b, 0, ...] += grad_output[b, ch, ...] * self.c[b, ch, ...][self.index[k1, b, 0, ...]]
            for b in range(0, self.batch_size):
                for ch in range(0, self.nr_of_channels):
                    for k1 in range(0, self.n + 1):
                        grad_c[b, ch, ...] += self._accumulate(self.weight[k1, b, 0, ...] * grad_output[b, ch, ...], [self.index[k1, b, 0, ...]], self.c.size()[2:])
        elif self.dim == 2:
            for b in range(0, self.batch_size):
                for k1 in range(0, self.n + 1):
                    for k2 in range(0, self.n + 1):
                        for ch in range(0, self.nr_of_channels):
                            grad_weight[k1, b, 0, ...] += grad_output[b, ch, ...] * self.weight[k2, b, 1, ...] * self.c[b, ch, ...][self.index[k1, b, 0, ...], self.index[k2, b, 1, ...]]
                            grad_weight[k2, b, 1, ...] += grad_output[b, ch, ...] * self.weight[k1, b, 0, ...] * self.c[b, ch, ...][self.index[k1, b, 0, ...], self.index[k2, b, 1, ...]]
            for b in range(0, self.batch_size):
                for ch in range(0, self.nr_of_channels):
                    for k1 in range(0, self.n + 1):
                        for k2 in range(0, self.n + 1):
                            grad_c[b, ch, ...] += self._accumulate(self.weight[k1, b, 0, ...] * self.weight[k2, b, 1, ...] * grad_output[b, ch, ...], [self.index[k1, b, 0, ...], self.index[k2, b, 1, ...]], self.c.size()[2:])
        elif self.dim == 3:
            for b in range(0, self.batch_size):
                for k1 in range(0, self.n + 1):
                    for k2 in range(0, self.n + 1):
                        for k3 in range(0, self.n + 1):
                            for ch in range(0, self.nr_of_channels):
                                grad_weight[k1, b, 0, ...] += grad_output[b, ch, ...] * self.weight[k2, b, 1, ...] * self.weight[k3, b, 2, ...] * self.c[b, ch, ...][self.index[k1, b, 0, ...], self.index[k2, b, 1, ...], self.index[k3, b, 2, ...]]
                                grad_weight[k2, b, 1, ...] += grad_output[b, ch, ...] * self.weight[k1, b, 0, ...] * self.weight[k3, b, 2, ...] * self.c[b, ch, ...][self.index[k1, b, 0, ...], self.index[k2, b, 1, ...], self.index[k3, b, 2, ...]]
                                grad_weight[k3, b, 2, ...] += grad_output[b, ch, ...] * self.weight[k1, b, 0, ...] * self.weight[k2, b, 1, ...] * self.c[b, ch, ...][self.index[k1, b, 0, ...], self.index[k2, b, 1, ...], self.index[k3, b, 2, ...]]
            for b in range(0, self.batch_size):
                for ch in range(0, self.nr_of_channels):
                    for k1 in range(0, self.n + 1):
                        for k2 in range(0, self.n + 1):
                            for k3 in range(0, self.n + 1):
                                grad_c[b, ch, ...] += self._accumulate(self.weight[k1, b, 0, ...] * self.weight[k2, b, 1, ...] * self.weight[k3, b, 2, ...] * grad_output[b, ch, ...], [self.index[k1, b, 0, ...], self.index[k2, b, 1, ...], self.index[k3, b, 2, ...]], self.c.size()[2:])
        else:
            raise ValueError('Dimension needs to be 1, 2, or 3.')
        return grad_c, grad_weight


def perform_spline_interpolation_helper(c, weight, index):
    """
    Helper function to instantiate the spline interpolation helper (for a more efficent gradient computation w/o automatic differentiation)

    :param c: interpolation coefficients
    :param weight: interpolation weights
    :param index: interpolation indices
    :return: interpolated signal
    """
    return PerformSplineInterpolationHelper(index)(c, weight)


class SplineInterpolation_ND_BCXYZ(Module):
    """
    Spline transform code for nD (1D, 2D, and 3D) spatial spline transforms. Uses the BCXYZ image format.
    Spline orders 3 to 9 are supported. Only order 3 is currently well tested.

    The code is a generalization (and pyTorch-ification) of the 2D spline code by Philippe Thevenaz:
    http://bigwww.epfl.ch/thevenaz/interpolation/

    The main difference is that the code supports 1D, 2D, and 3D images in pyTorch format (i.e., the first
    two dimensions are the batch size and the number of channels. Furthermore, great care has been taken to
    avoid loops over pixels to obtain a reasonably high performance interpolation.

    """

    def __init__(self, spacing, spline_order):
        """
        Constructor for spline interpolation

        :param spacing: spacing of the map which will be used for interpolation (this is NOT the spacing of the image data from which to compute the interpolation coefficient)
        :param spline_order: desired order of the spline: [3,4,5,6,7,8,9]
        """
        super(SplineInterpolation_ND_BCXYZ, self).__init__()
        self.spacing = spacing
        """spatial spacing; IMPORTANT: needs to be the spacing of the map at which locations the interpolation should be performed 
        (NOT the spacing of the image from which the coefficient are computed)"""
        self.spline_order = spline_order
        """spline order"""
        self.n = spline_order
        self.Ns = None
        if self.n not in [2, 3, 4, 5, 6, 7, 8, 9]:
            raise ValueError('Unknown spline order')
        self.poles = dict()
        self.poles[2] = AdaptVal(torch.from_numpy(np.array([np.sqrt(8.0) - 3.0]).astype('float32')))
        self.poles[3] = AdaptVal(torch.from_numpy(np.array([np.sqrt(3.0) - 2.0]).astype('float32')))
        self.poles[4] = AdaptVal(torch.from_numpy(np.array([np.sqrt(664.0 - np.sqrt(438976.0)) + np.sqrt(304.0) - 19.0, np.sqrt(664.0 + np.sqrt(438976.0)) - np.sqrt(304.0) - 19.0]).astype('float32')))
        self.poles[5] = AdaptVal(torch.from_numpy(np.array([np.sqrt(135.0 / 2.0 - np.sqrt(17745.0 / 4.0)) + np.sqrt(105.0 / 4.0) - 13.0 / 2.0, np.sqrt(135.0 / 2.0 + np.sqrt(17745.0 / 4.0)) - np.sqrt(105.0 / 4.0) - 13.0 / 2.0]).astype('float32')))
        self.poles[6] = AdaptVal(torch.from_numpy(np.array([-0.48829458930304476, -0.08167927107623751, -0.0014141518083258177]).astype('float32')))
        self.poles[7] = AdaptVal(torch.from_numpy(np.array([-0.5352804307964382, -0.12255461519232669, -0.009148694809608277]).astype('float32')))
        self.poles[8] = AdaptVal(torch.from_numpy(np.array([-0.5746869092487654, -0.16303526929728093, -0.02363229469484485, -0.00015382131064169092]).astype('float32')))
        self.poles[9] = AdaptVal(torch.from_numpy(np.array([-0.6079973891686258, -0.20175052019315323, -0.04322260854048175, -0.002121306903180818]).astype('float32')))

    def _scale_map_to_ijk(self, phi, spacing, sz_image):
        """
        Scales the map to the [0,i-1]x[0,j-1]x[0,k-1] format from the standard mermaid format which assumes the spacing has been taken into account

        :param map: map in BxCxXxYxZ format
        :param spacing: spacing in XxYxZ format (of the map which hold the interpolation corrdinates)
        :param ijk-size of image that needs to be interpolated
        :return: returns the scaled map
        """
        sz = phi.size()
        scaling = (np.array(list(sz_image[2:])).astype('float32') - 1.0) / (np.array(list(sz[2:])).astype('float32') - 1.0)
        phi_scaled = torch.zeros_like(phi)
        ndim = len(spacing)
        for d in range(ndim):
            phi_scaled[:, d, ...] = phi[:, d, ...] * (scaling[d] / spacing[d])
        return phi_scaled

    def _slice_dim(self, val, idx, dim):
        """
        Conveninece function to allow slicing an array at a particular index of a dimension

        :param val: array
        :param idx: index
        :param dim: dimension along which to slice
        :return: returns the sliced array
        """
        if dim == 1:
            return val[:, :, idx, ...]
        elif dim == 2:
            return val[:, :, :, idx, ...]
        elif dim == 3:
            return val[:, :, :, :, idx, ...]
        else:
            raise ValueError('Dimension needs to be 1, 2, or 3')

    def _initial_causal_coefficient(self, c, z, tol, dim=1):
        """
        Computes the initial causal coefficient for the spline filter.

        :param c: coefficient array
        :param z: pole
        :param tol: tolerance
        :return: returns the intial causal coefficient
        """
        if self.Ns is None:
            raise ValueError('Unknown data length')
        if dim not in [1, 2, 3]:
            raise ValueError('Dimension needs to be 1, 2, or 3')
        horizon = self.Ns[dim - 1]
        if tol > 0:
            horizon = int(np.ceil(np.log(tol) / np.log(np.abs(z))))
        if horizon < self.Ns[dim - 1]:
            zn = z.clone()
            Sum = self._slice_dim(c, 0, dim=dim)
            for n in range(1, horizon):
                Sum += zn * self._slice_dim(c, n, dim=dim)
                zn *= z
            return Sum
        else:
            zn = z.clone()
            iz = 1.0 / z
            z2n = z ** (self.Ns[dim - 1] - 1.0)
            Sum = self._slice_dim(c, 0, dim=dim) + z2n * self._slice_dim(c, -1, dim=dim)
            z2n *= z2n * iz
            for n in range(1, self.Ns[dim - 1] - 1):
                Sum += (zn + z2n) * self._slice_dim(c, n, dim=dim)
                zn *= z
                z2n *= iz
            return Sum / (1.0 - zn * zn)

    def _initial_anti_causal_coefficient(self, c, z, dim=1):
        """
        Computes the intial anti causal coefficient for spline interpolation (i.e., for the filter that runs backward)

        :param c: coefficients
        :param z: pole
        :return: anti-causal coefficient
        """
        if self.Ns is None:
            raise ValueError('Unknown data length')
        return z / (z * z - 1.0) * (z * self._slice_dim(c, -2, dim=dim) + self._slice_dim(c, -1, dim=dim))

    def _convert_to_interpolation_cofficients_in_dim_1(self, c, z, tol):
        """
        Converts cofficients (or initialy the signal) into interpolation coefficients along dimension 1.

        :param c: coefficient array (on first use this should contain the signal itself)
        :param z: pole
        :param tol: tolerance
        :return: returns c itself with was modified in place
        """
        dim = 1
        nb_poles = len(z)
        lam = 1.0
        for k in range(0, nb_poles):
            lam *= (1.0 - z[k]) * (1.0 - 1.0 / z[k])
        c *= lam
        for k in range(0, nb_poles):
            c[:, :, 0, ...] = self._initial_causal_coefficient(c, z[k], tol, dim=dim)
            for n in range(1, self.Ns[dim - 1]):
                c[:, :, n, ...] = c[:, :, n, ...] + z[k] * c[:, :, n - 1, ...]
            c[:, :, -1, ...] = self._initial_anti_causal_coefficient(c, z[k], dim=dim)
            for n in range(self.Ns[dim - 1] - 2, -1, -1):
                c[:, :, n, ...] = z[k] * (c[:, :, n + 1, ...] - c[:, :, n, ...])
        return c

    def _convert_to_interpolation_cofficients_in_dim_2(self, c, z, tol):
        """
        Converts cofficients (or initialy the signal) into interpolation coefficients along dimension 2.

        :param c: coefficient array (on first use this should contain the signal itself)
        :param z: pole
        :param tol: tolerance
        :return: returns c itself with was modified in place
        """
        dim = 2
        nb_poles = len(z)
        lam = 1.0
        for k in range(0, nb_poles):
            lam *= (1.0 - z[k]) * (1.0 - 1.0 / z[k])
        c *= lam
        for k in range(0, nb_poles):
            c[:, :, :, 0, ...] = self._initial_causal_coefficient(c, z[k], tol, dim=dim)
            for n in range(1, self.Ns[dim - 1]):
                c[:, :, :, n, ...] = c[:, :, :, n, ...] + z[k] * c[:, :, :, n - 1, ...]
            c[:, :, :, -1, ...] = self._initial_anti_causal_coefficient(c, z[k], dim=dim)
            for n in range(self.Ns[dim - 1] - 2, -1, -1):
                c[:, :, :, n, ...] = z[k] * (c[:, :, :, n + 1, ...] - c[:, :, :, n, ...])
        return c

    def _convert_to_interpolation_cofficients_in_dim_3(self, c, z, tol):
        """
        Converts cofficients (or initialy the signal) into interpolation coefficients along dimension 3.

        :param c: coefficient array (on first use this should contain the signal itself)
        :param z: pole
        :param tol: tolerance
        :return: returns c itself with was modified in place
        """
        dim = 3
        nb_poles = len(z)
        lam = 1.0
        for k in range(0, nb_poles):
            lam *= (1.0 - z[k]) * (1.0 - 1.0 / z[k])
        c *= lam
        for k in range(0, nb_poles):
            c[:, :, :, :, 0, ...] = self._initial_causal_coefficient(c, z[k], tol, dim=dim)
            for n in range(1, self.Ns[dim - 1]):
                c[:, :, :, :, n, ...] = c[:, :, :, :, n, ...] + z[k] * c[:, :, :, :, n - 1, ...]
            c[:, :, :, :, -1, ...] = self._initial_anti_causal_coefficient(c, z[k], dim=dim)
            for n in range(self.Ns[dim - 1] - 2, -1, -1):
                c[:, :, :, :, n, ...] = z[k] * (c[:, :, :, :, n + 1, ...] - c[:, :, :, :, n, ...])
        return c

    def _convert_to_interpolation_cofficients_in_dim(self, c, z, tol, dim=1):
        """
        Converts cofficients (or initialy the signal) into interpolation coefficients along desired dimension.

        :param c: coefficient array (on first use this should contain the signal itself)
        :param z: pole
        :param tol: tolerance
        :param dim: dimension along which to filter the coefficients
        :return: returns c itself with was modified in place
        """
        if dim == 1:
            cr = self._convert_to_interpolation_cofficients_in_dim_1(c, z, tol)
        elif dim == 2:
            cr = self._convert_to_interpolation_cofficients_in_dim_2(c, z, tol)
        elif dim == 3:
            cr = self._convert_to_interpolation_cofficients_in_dim_3(c, z, tol)
        else:
            raise ValueError('not yet implemented')
        return cr

    def _convert_to_interpolation_coefficients(self, s, z, tol):
        """
        Converts the input signal, s, into a set of filter coefficients. Makes use of the separability of spline interpolation.

        :param s: input signal
        :param z: poles
        :param tol: tolerance
        :return: returns the computed coefficients c
        """
        sz = s.size()
        dim = len(sz) - 2
        if dim not in [1, 2, 3]:
            raise ValueError('Signal needs to be of dimensions 1, 2, or 3 and in format B x C x X x Y x Z')
        c = MyTensor(*list(s.size())).zero_()
        c[:] = s
        self.Ns = list(s.size()[2:])
        if np.any(np.array(self.Ns) <= 1):
            raise ValueError('Expected at least two values, but at least one of the dimensions has less')
        for d in range(dim):
            c = self._convert_to_interpolation_cofficients_in_dim(c, z, tol, dim=d + 1)
        return c

    def _get_interpolation_coefficients(self, s, tol=0):
        """
        Obtains the interpolation coefficients for a given signal s.

        :param s: signal
        :param tol: tolerance
        :return: interpolation coefficients c
        """
        return self._convert_to_interpolation_coefficients(s, self.poles[self.n], tol)

    def _compute_interpolation_weights(self, x):
        """
        Compute the interpolation weights at coordinates x

        :param x: coordinates in i,j,k format (will have to be converted to this format from map coordinates first)
        :return: returns a two-tuple of (index,weight) holding the interpolation indices and weights
        """
        sz = x.size()
        dim = sz[1]
        index = MyLongTensor(*([self.n + 1] + list(x.size())))
        weight = MyTensor(*([self.n + 1] + list(x.size()))).zero_()
        if self.n % 2 == 0:
            for d in range(dim):
                i = torch.floor(x[:, d, ...].data + 0.5) - self.n // 2
                for k in range(0, self.n + 1):
                    index[k, :, d, ...] = i + k
        else:
            for d in range(dim):
                i = torch.floor(x[:, d, ...].data) - self.n // 2
                for k in range(0, self.n + 1):
                    index[k, :, d, ...] = i + k
        if self.n == 2:
            w = x - index[1, ...].float()
            weight[1, ...] = 3.0 / 4.0 - w * w
            weight[2, ...] = 1.0 / 2.0 * (w - weight[1, ...] + 1.0)
            weight[0, ...] = 1.0 - weight[1, ...] - weight[2, ...]
        elif self.n == 3:
            w = x - index[1, ...].float()
            weight[3, ...] = 1.0 / 6.0 * w * w * w
            weight[0, ...] = 1.0 / 6.0 + 1.0 / 2.0 * w * (w - 1.0) - weight[3, ...]
            weight[2, ...] = w + weight[0, ...] - 2.0 * weight[3, ...]
            weight[1, ...] = 1.0 - weight[0, ...] - weight[2, ...] - weight[3, ...]
        elif self.n == 4:
            w = x - index[2].float()
            w2 = w * w
            t = 1.0 / 6.0 * w2
            weight[0] = 1.0 / 2.0 - w
            weight[0] *= weight[0]
            weight[0] *= 1.0 / 24.0 * weight[0]
            t0 = w * (t - 11.0 / 24.0)
            t1 = 19.0 / 96.0 + w2 * (1.0 / 4.0 - t)
            weight[1] = t1 + t0
            weight[3] = t1 - t0
            weight[4] = weight[0] + t0 + 1.0 / 2.0 * w
            weight[2] = 1.0 - weight[0] - weight[1] - weight[3] - weight[4]
        elif self.n == 5:
            w = x - index[2].float()
            w2 = w * w
            weight[5] = 1.0 / 120.0 * w * w2 * w2
            w2 -= w
            w4 = w2 * w2
            w -= 1.0 / 2.0
            t = w2 * (w2 - 3.0)
            weight[0] = 1.0 / 24.0 * (1.0 / 5.0 + w2 + w4) - weight[5]
            t0 = 1.0 / 24.0 * (w2 * (w2 - 5.0) + 46.0 / 5.0)
            t1 = -1.0 / 12.0 * w * (t + 4.0)
            weight[2] = t0 + t1
            weight[3] = t0 - t1
            t0 = 1.0 / 16.0 * (9.0 / 5.0 - t)
            t1 = 1.0 / 24.0 * w * (w4 - w2 - 5.0)
            weight[1] = t0 + t1
            weight[4] = t0 - t1
        elif self.n == 6:
            w = x - index[3].float()
            weight[0] = 1.0 / 2.0 - w
            weight[0] *= weight[0] * weight[0]
            weight[0] *= weight[0] / 720.0
            weight[1] = (361.0 / 192.0 - w * (59.0 / 8.0 + w * (-185.0 / 16.0 + w * (25.0 / 3.0 + w * (-5.0 / 2.0 + w) * (1.0 / 2.0 + w))))) / 120.0
            weight[2] = (10543.0 / 960.0 + w * (-289.0 / 16.0 + w * (79.0 / 16.0 + w * (43.0 / 6.0 + w * (-17.0 / 4.0 + w * (-1.0 + w)))))) / 48.0
            w2 = w * w
            weight[3] = (5887.0 / 320.0 - w2 * (231.0 / 16.0 - w2 * (21.0 / 4.0 - w2))) / 36.0
            weight[4] = (10543.0 / 960.0 + w * (289.0 / 16.0 + w * (79.0 / 16.0 + w * (-43.0 / 6.0 + w * (-17.0 / 4.0 + w * (1.0 + w)))))) / 48.0
            weight[6] = 1.0 / 2.0 + w
            weight[6] *= weight[6] * weight[6]
            weight[6] *= weight[6] / 720.0
            weight[5] = 1.0 - weight[0] - weight[1] - weight[2] - weight[3] - weight[4] - weight[6]
        elif self.n == 7:
            w = x - index[3].float()
            weight[0] = 1.0 - w
            weight[0] *= weight[0]
            weight[0] *= weight[0] * weight[0]
            weight[0] *= (1.0 - w) / 5040.0
            w2 = w * w
            weight[1] = (120.0 / 7.0 + w * (-56.0 + w * (72.0 + w * (-40.0 + w2 * (12.0 + w * (-6.0 + w)))))) / 720.0
            weight[2] = (397.0 / 7.0 - w * (245.0 / 3.0 + w * (-15.0 + w * (-95.0 / 3.0 + w * (15.0 + w * (5.0 + w * (-5.0 + w))))))) / 240.0
            weight[3] = (2416.0 / 35.0 + w2 * (-48.0 + w2 * (16.0 + w2 * (-4.0 + w)))) / 144.0
            weight[4] = (1191.0 / 35.0 - w * (-49.0 + w * (-9.0 + w * (19.0 + w * (-3.0 + w) * (-3.0 + w2))))) / 144.0
            weight[5] = (40.0 / 7.0 + w * (56.0 / 3.0 + w * (24.0 + w * (40.0 / 3.0 + w2 * (-4.0 + w * (-2.0 + w)))))) / 240.0
            weight[7] = w2
            weight[7] *= weight[7] * weight[7]
            weight[7] *= w / 5040.0
            weight[6] = 1.0 - weight[0] - weight[1] - weight[2] - weight[3] - weight[4] - weight[5] - weight[7]
        elif self.n == 8:
            w = x - index[4].float()
            weight[0] = 1.0 / 2.0 - w
            weight[0] *= weight[0]
            weight[0] *= weight[0]
            weight[0] *= weight[0] / 40320.0
            w2 = w * w
            weight[1] = (39.0 / 16.0 - w * (6.0 + w * (-9.0 / 2.0 + w2))) * (21.0 / 16.0 + w * (-15.0 / 4.0 + w * (9.0 / 2.0 + w * (-3.0 + w)))) / 5040.0
            weight[2] = (82903.0 / 1792.0 + w * (-4177.0 / 32.0 + w * (2275.0 / 16.0 + w * (-487.0 / 8.0 + w * (-85.0 / 8.0 + w * (41.0 / 2.0 + w * (-5.0 + w * (-2.0 + w)))))))) / 1440.0
            weight[3] = (310661.0 / 1792.0 - w * (14219.0 / 64.0 + w * (-199.0 / 8.0 + w * (-1327.0 / 16.0 + w * (245.0 / 8.0 + w * (53.0 / 4.0 + w * (-8.0 + w * (-1.0 + w)))))))) / 720.0
            weight[4] = (2337507.0 / 8960.0 + w2 * (-2601.0 / 16.0 + w2 * (387.0 / 8.0 + w2 * (-9.0 + w2)))) / 576.0
            weight[5] = (310661.0 / 1792.0 - w * (-14219.0 / 64.0 + w * (-199.0 / 8.0 + w * (1327.0 / 16.0 + w * (245.0 / 8.0 + w * (-53.0 / 4.0 + w * (-8.0 + w * (1.0 + w)))))))) / 720.0
            weight[7] = (39.0 / 16.0 - w * (-6.0 + w * (-9.0 / 2.0 + w2))) * (21.0 / 16.0 + w * (15.0 / 4.0 + w * (9.0 / 2.0 + w * (3.0 + w)))) / 5040.0
            weight[8] = 1.0 / 2.0 + w
            weight[8] *= weight[8]
            weight[8] *= weight[8]
            weight[8] *= weight[8] / 40320.0
            weight[6] = 1.0 - weight[0] - weight[1] - weight[2] - weight[3] - weight[4] - weight[5] - weight[7] - weight[8]
        elif self.n == 9:
            w = x - index[4].float()
            weight[0] = 1.0 - w
            weight[0] *= weight[0]
            weight[0] *= weight[0]
            weight[0] *= weight[0] * (1.0 - w) / 362880.0
            weight[1] = (502.0 / 9.0 + w * (-246.0 + w * (472.0 + w * (-504.0 + w * (308.0 + w * (-84.0 + w * (-56.0 / 3.0 + w * (24.0 + w * (-8.0 + w))))))))) / 40320.0
            weight[2] = (3652.0 / 9.0 - w * (2023.0 / 2.0 + w * (-952.0 + w * (938.0 / 3.0 + w * (112.0 + w * (-119.0 + w * (56.0 / 3.0 + w * (14.0 + w * (-7.0 + w))))))))) / 10080.0
            weight[3] = (44117.0 / 42.0 + w * (-2427.0 / 2.0 + w * (66.0 + w * (434.0 + w * (-129.0 + w * (-69.0 + w * (34.0 + w * (6.0 + w * (-6.0 + w))))))))) / 4320.0
            w2 = w * w
            weight[4] = (78095.0 / 63.0 - w2 * (700.0 + w2 * (-190.0 + w2 * (100.0 / 3.0 + w2 * (-5.0 + w))))) / 2880.0
            weight[5] = (44117.0 / 63.0 + w * (809.0 + w * (44.0 + w * (-868.0 / 3.0 + w * (-86.0 + w * (46.0 + w * (68.0 / 3.0 + w * (-4.0 + w * (-4.0 + w))))))))) / 2880.0
            weight[6] = (3652.0 / 21.0 - w * (-867.0 / 2.0 + w * (-408.0 + w * (-134.0 + w * (48.0 + w * (51.0 + w * (-4.0 + w) * (-1.0 + w) * (2.0 + w))))))) / 4320.0
            weight[7] = (251.0 / 18.0 + w * (123.0 / 2.0 + w * (118.0 + w * (126.0 + w * (77.0 + w * (21.0 + w * (-14.0 / 3.0 + w * (-6.0 + w * (-2.0 + w))))))))) / 10080.0
            weight[9] = w2 * w2
            weight[9] *= weight[9] * w / 362880.0
            weight[8] = 1.0 - weight[0] - weight[1] - weight[2] - weight[3] - weight[4] - weight[5] - weight[6] - weight[7] - weight[9]
        else:
            raise ValueError('Unsupported spline order')
        return index, weight

    def _interpolate(self, c, x):
        """
        Given the computed interpolation coefficients c and the map coordinates x (in ijk format) compute the interpolated values

        :param c: interpolation coefficients
        :param x: map coordinates
        :return: interpolated values
        """
        sz = c.size()
        dim = x.size()[1]
        if dim not in [1, 2, 3]:
            raise ValueError('Only dimensions 1, 2, and 3 are currently supported')
        index, weight = self._compute_interpolation_weights(x)
        for d in range(dim):
            width = sz[2 + d]
            width2 = 2 * width - 2
            lt_z = index[:, :, d, ...] < 0
            ge_z = index[:, :, d, ...] >= 0
            index[:, :, d, ...][lt_z] = -index[:, :, d, ...][lt_z] - width2 * (-index[:, :, d, ...][lt_z] / width2)
            index[:, :, d, ...][ge_z] = index[:, :, d, ...][ge_z] - width2 * (index[:, :, d, ...][ge_z] / width2)
            ge_w = index[:, :, d, ...] >= width
            index[:, :, d, ...][ge_w] = width2 - index[:, :, d, ...][ge_w]
        w = perform_spline_interpolation_helper(c, weight, index)
        return w

    def forward(self, im, phi):
        """
        Perform the actual spatial transform

        :param im: image in BCXYZ format
        :param phi: spatial transform in BdimXYZ format (assumes that phi makes use of the spacing defined when contructing the object)
        :return: spatially transformed image in BCXYZ format
        """
        c = self._get_interpolation_coefficients(im)
        interpolated_values = self._interpolate(c, self._scale_map_to_ijk(phi, self.spacing, im.size()))
        return interpolated_values


class ConvBnRel(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, active_unit='relu', same_padding=False, bn=False, reverse=False, bias=False):
        super(ConvBnRel, self).__init__()
        padding = int((kernel_size - 1) // 2) if same_padding else 0
        if not reverse:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias)
        else:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.0001, momentum=0, affine=True) if bn else None
        if active_unit == 'relu':
            self.active_unit = nn.ReLU(inplace=True)
        elif active_unit == 'elu':
            self.active_unit = nn.ELU(inplace=True)
        else:
            self.active_unit = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.active_unit is not None:
            x = self.active_unit(x)
        return x


class FcRel(nn.Module):

    def __init__(self, in_features, out_features, active_unit='relu'):
        super(FcRel, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        if active_unit == 'relu':
            self.active_unit = nn.ReLU(inplace=True)
        elif active_unit == 'elu':
            self.active_unit = nn.ELU(inplace=True)
        else:
            self.active_unit = None

    def forward(self, x):
        x = self.fc(x)
        if self.active_unit is not None:
            x = self.active_unit(x)
        return x


def organize_data(moving, target, sched='depth_concat'):
    if sched == 'depth_concat':
        input = torch.cat([moving, target], dim=1)
    elif sched == 'width_concat':
        input = torch.cat((moving, target), dim=3)
    elif sched == 'list_concat':
        input = torch.cat((moving.unsqueeze(0), target.unsqueeze(0)), dim=0)
    elif sched == 'difference':
        input = moving - target
    return input


class AdpSmoother(nn.Module):
    """
    a simple conv. implementation, generate displacement field
    """

    def __init__(self, inputs, dim, net_sched=None):
        super(AdpSmoother, self).__init__()
        self.dim = dim
        self.net_sched = 'm_only'
        self.s = inputs['s'].detach()
        self.t = inputs['t'].detach()
        self.mask = Parameter(torch.cat([torch.ones(inputs['s'].size())] * dim, 1), requires_grad=True)
        self.get_net_sched()

    def get_net_sched(self, debugging=True, using_bn=True, active_unit='relu', using_sigmoid=False, kernel_size=5):
        padding_size = (kernel_size - 1) // 2
        if self.net_sched == 'm_only':
            if debugging:
                self.net = nn.Conv2d(2, 2, kernel_size, 1, padding=padding_size, bias=False, groups=2)
            else:
                net = [ConvBnRel(self.dim, 20, 5, active_unit=active_unit, same_padding=True, bn=using_bn), ConvBnRel(20, self.dim, 5, active_unit=active_unit, same_padding=True, bn=using_bn)]
                if using_sigmoid:
                    net += [nn.Sigmoid()]
                self.net = nn.Sequential(*net)
        elif self.net_sched == 'm_f_s':
            if debugging:
                self.net = nn.Conv2d(self.dim + 1, self.dim, kernel_size, 1, padding=padding_size, bias=False)
            else:
                net = [ConvBnRel(self.dim + 1, 20, 5, active_unit=active_unit, same_padding=True, bn=using_bn), ConvBnRel(20, self.dim, 5, active_unit=active_unit, same_padding=True, bn=using_bn)]
                if using_sigmoid:
                    net += [nn.Sigmoid()]
                self.net = nn.Sequential(*net)
        elif self.net_sched == 'm_d_s':
            if debugging:
                self.net = nn.Conv2d(self.dim + 1, self.dim, kernel_size, 1, padding=padding_size, bias=False)
            else:
                net = [ConvBnRel(self.dim + 1, 20, 5, active_unit=active_unit, same_padding=True, bn=using_bn), ConvBnRel(20, self.dim, 5, active_unit=active_unit, same_padding=True, bn=using_bn)]
                if using_sigmoid:
                    net += [nn.Sigmoid()]
                self.net = nn.Sequential(*net)
        elif self.net_sched == 'm_f_s_t':
            if debugging:
                self.net = nn.Conv2d(self.dim + 2, self.dim, kernel_size, 1, padding=padding_size, bias=False)
            else:
                net = [ConvBnRel(self.dim + 2, 20, 5, active_unit=active_unit, same_padding=True, bn=using_bn), ConvBnRel(20, self.dim, 5, active_unit=active_unit, same_padding=True, bn=using_bn)]
                if using_sigmoid:
                    net += [nn.Sigmoid()]
                self.net = nn.Sequential(*net)
        elif self.net_sched == 'm_d_s_f_t':
            if debugging:
                self.net = nn.Conv2d(self.dim + 2, self.dim, kernel_size, 1, padding=padding_size, bias=False)
            else:
                net = [ConvBnRel(self.dim + 2, 20, 5, active_unit=active_unit, same_padding=True, bn=using_bn), ConvBnRel(20, self.dim, 5, active_unit=active_unit, same_padding=True, bn=using_bn)]
                if using_sigmoid:
                    net += [nn.Sigmoid()]
                self.net = nn.Sequential(*net)

    def prepare_data(self, m, new_s):
        input = None
        if self.net_sched == 'm_only':
            input = m
        elif self.net_sched == 'm_f_s':
            input = organize_data(m, self.s, sched='depth_concat')
        elif self.net_sched == 'm_d_s':
            input = organize_data(m, new_s, sched='depth_concat')
        elif self.net_sched == 'm_f_s_t':
            input = organize_data(m, self.s, sched='depth_concat')
            input = organize_data(input, self.t, sched='depth_concat')
        elif self.net_sched == 'm_f_s_t':
            input = organize_data(m, self.s, sched='depth_concat')
            input = organize_data(input, self.t, sched='depth_concat')
        elif self.net_sched == 'm_d_s_f_t':
            input = organize_data(m, new_s, sched='depth_concat')
            input = organize_data(input, self.t, sched='depth_concat')
        return input

    def forward(self, m, new_s=None):
        m = m * self.mask
        input = self.prepare_data(m, new_s)
        x = input
        x = self.net(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ConvBnRel,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FcRel,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (HLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_uncbiag_mermaid(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

