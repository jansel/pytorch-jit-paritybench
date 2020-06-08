import sys
_module = sys.modules[__name__]
del sys
core_qnn = _module
quaternion_layers = _module
quaternion_ops = _module
recurrent_models = _module
stacked_lstm = _module
cae = _module
convolutional_models = _module
copy_task = _module
recurrent_models = _module
plot_copy_task_curves = _module
psnr_ssim = _module
copy_task = _module
recurrent_models = _module
r2h = _module
r2h_models = _module
r2h_ae = _module
r2h_models = _module
setup = _module

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


import numpy as np


from numpy.random import RandomState


import torch


from torch.autograd import Variable


import torch.nn.functional as F


import torch.nn as nn


from torch.nn.parameter import Parameter


from torch.nn import Module


import math


from scipy.stats import chi


from torch.nn import Parameter


from torch.nn import functional as F


import torch.optim


from torch import autograd


import torch.optim as optim


from torch.optim.lr_scheduler import StepLR


def get_kernel_and_weight_shape(operation, in_channels, out_channels,
    kernel_size):
    if operation == 'convolution1d':
        if type(kernel_size) is not int:
            raise ValueError(
                """An invalid kernel_size was supplied for a 1d convolution. The kernel size
                must be integer in the case. Found kernel_size = """
                 + str(kernel_size))
        else:
            ks = kernel_size
            w_shape = (out_channels, in_channels) + tuple((ks,))
    else:
        if operation == 'convolution2d' and type(kernel_size) is int:
            ks = kernel_size, kernel_size
        elif operation == 'convolution3d' and type(kernel_size) is int:
            ks = kernel_size, kernel_size, kernel_size
        elif type(kernel_size) is not int:
            if operation == 'convolution2d' and len(kernel_size) != 2:
                raise ValueError(
                    """An invalid kernel_size was supplied for a 2d convolution. The kernel size
                    must be either an integer or a tuple of 2. Found kernel_size = """
                     + str(kernel_size))
            elif operation == 'convolution3d' and len(kernel_size) != 3:
                raise ValueError(
                    """An invalid kernel_size was supplied for a 3d convolution. The kernel size
                    must be either an integer or a tuple of 3. Found kernel_size = """
                     + str(kernel_size))
            else:
                ks = kernel_size
        w_shape = (out_channels, in_channels) + (*ks,)
    return ks, w_shape


def affect_init_conv(r_weight, i_weight, j_weight, k_weight, kernel_size,
    init_func, rng, init_criterion):
    if r_weight.size() != i_weight.size() or r_weight.size() != j_weight.size(
        ) or r_weight.size() != k_weight.size():
        raise ValueError(
            'The real and imaginary weights should have the same size . Found: r:'
             + str(r_weight.size()) + ' i:' + str(i_weight.size()) + ' j:' +
            str(j_weight.size()) + ' k:' + str(k_weight.size()))
    elif 2 >= r_weight.dim():
        raise Exception(
            'affect_conv_init accepts only tensors that have more than 2 dimensions. Found dimension = '
             + str(real_weight.dim()))
    r, i, j, k = init_func(r_weight.size(1), r_weight.size(0), rng=rng,
        kernel_size=kernel_size, criterion=init_criterion)
    r, i, j, k = torch.from_numpy(r), torch.from_numpy(i), torch.from_numpy(j
        ), torch.from_numpy(k)
    r_weight.data = r.type_as(r_weight.data)
    i_weight.data = i.type_as(i_weight.data)
    j_weight.data = j.type_as(j_weight.data)
    k_weight.data = k.type_as(k_weight.data)


def random_init(in_features, out_features, rng, kernel_size=None, criterion
    ='glorot'):
    if kernel_size is not None:
        receptive_field = np.prod(kernel_size)
        fan_in = in_features * receptive_field
        fan_out = out_features * receptive_field
    else:
        fan_in = in_features
        fan_out = out_features
    if criterion == 'glorot':
        s = 1.0 / np.sqrt(2 * (fan_in + fan_out))
    elif criterion == 'he':
        s = 1.0 / np.sqrt(2 * fan_in)
    else:
        raise ValueError('Invalid criterion: ' + criterion)
    if kernel_size is None:
        kernel_shape = in_features, out_features
    elif type(kernel_size) is int:
        kernel_shape = (out_features, in_features) + tuple((kernel_size,))
    else:
        kernel_shape = (out_features, in_features) + (*kernel_size,)
    number_of_weights = np.prod(kernel_shape)
    v_r = np.random.uniform(-1.0, 1.0, number_of_weights)
    v_i = np.random.uniform(-1.0, 1.0, number_of_weights)
    v_j = np.random.uniform(-1.0, 1.0, number_of_weights)
    v_k = np.random.uniform(-1.0, 1.0, number_of_weights)
    v_r = v_r.reshape(kernel_shape)
    v_i = v_i.reshape(kernel_shape)
    v_j = v_j.reshape(kernel_shape)
    v_k = v_k.reshape(kernel_shape)
    weight_r = v_r
    weight_i = v_i
    weight_j = v_j
    weight_k = v_k
    return weight_r, weight_i, weight_j, weight_k


def unitary_init(in_features, out_features, rng, kernel_size=None,
    criterion='he'):
    if kernel_size is not None:
        receptive_field = np.prod(kernel_size)
        fan_in = in_features * receptive_field
        fan_out = out_features * receptive_field
    else:
        fan_in = in_features
        fan_out = out_features
    if kernel_size is None:
        kernel_shape = in_features, out_features
    elif type(kernel_size) is int:
        kernel_shape = (out_features, in_features) + tuple((kernel_size,))
    else:
        kernel_shape = (out_features, in_features) + (*kernel_size,)
    number_of_weights = np.prod(kernel_shape)
    v_r = np.random.uniform(-1.0, 1.0, number_of_weights)
    v_i = np.random.uniform(-1.0, 1.0, number_of_weights)
    v_j = np.random.uniform(-1.0, 1.0, number_of_weights)
    v_k = np.random.uniform(-1.0, 1.0, number_of_weights)
    for i in range(0, number_of_weights):
        norm = np.sqrt(v_r[i] ** 2 + v_i[i] ** 2 + v_j[i] ** 2 + v_k[i] ** 2
            ) + 0.0001
        v_r[i] /= norm
        v_i[i] /= norm
        v_j[i] /= norm
        v_k[i] /= norm
    v_r = v_r.reshape(kernel_shape)
    v_i = v_i.reshape(kernel_shape)
    v_j = v_j.reshape(kernel_shape)
    v_k = v_k.reshape(kernel_shape)
    return v_r, v_i, v_j, v_k


def quaternion_init(in_features, out_features, rng, kernel_size=None,
    criterion='glorot'):
    if kernel_size is not None:
        receptive_field = np.prod(kernel_size)
        fan_in = in_features * receptive_field
        fan_out = out_features * receptive_field
    else:
        fan_in = in_features
        fan_out = out_features
    if criterion == 'glorot':
        s = 1.0 / np.sqrt(2 * (fan_in + fan_out))
    elif criterion == 'he':
        s = 1.0 / np.sqrt(2 * fan_in)
    else:
        raise ValueError('Invalid criterion: ' + criterion)
    rng = RandomState(np.random.randint(1, 1234))
    if kernel_size is None:
        kernel_shape = in_features, out_features
    elif type(kernel_size) is int:
        kernel_shape = (out_features, in_features) + tuple((kernel_size,))
    else:
        kernel_shape = (out_features, in_features) + (*kernel_size,)
    modulus = chi.rvs(4, loc=0, scale=s, size=kernel_shape)
    number_of_weights = np.prod(kernel_shape)
    v_i = np.random.uniform(-1.0, 1.0, number_of_weights)
    v_j = np.random.uniform(-1.0, 1.0, number_of_weights)
    v_k = np.random.uniform(-1.0, 1.0, number_of_weights)
    for i in range(0, number_of_weights):
        norm = np.sqrt(v_i[i] ** 2 + v_j[i] ** 2 + v_k[i] ** 2 + 0.0001)
        v_i[i] /= norm
        v_j[i] /= norm
        v_k[i] /= norm
    v_i = v_i.reshape(kernel_shape)
    v_j = v_j.reshape(kernel_shape)
    v_k = v_k.reshape(kernel_shape)
    phase = rng.uniform(low=-np.pi, high=np.pi, size=kernel_shape)
    weight_r = modulus * np.cos(phase)
    weight_i = modulus * v_i * np.sin(phase)
    weight_j = modulus * v_j * np.sin(phase)
    weight_k = modulus * v_k * np.sin(phase)
    return weight_r, weight_i, weight_j, weight_k


def quaternion_transpose_conv(input, r_weight, i_weight, j_weight, k_weight,
    bias, stride, padding, output_padding, groups, dilatation):
    """
    Applies a quaternion trasposed convolution to the incoming data:

    """
    cat_kernels_4_r = torch.cat([r_weight, -i_weight, -j_weight, -k_weight],
        dim=1)
    cat_kernels_4_i = torch.cat([i_weight, r_weight, -k_weight, j_weight],
        dim=1)
    cat_kernels_4_j = torch.cat([j_weight, k_weight, r_weight, -i_weight],
        dim=1)
    cat_kernels_4_k = torch.cat([k_weight, -j_weight, i_weight, r_weight],
        dim=1)
    cat_kernels_4_quaternion = torch.cat([cat_kernels_4_r, cat_kernels_4_i,
        cat_kernels_4_j, cat_kernels_4_k], dim=0)
    if input.dim() == 3:
        convfunc = F.conv_transpose1d
    elif input.dim() == 4:
        convfunc = F.conv_transpose2d
    elif input.dim() == 5:
        convfunc = F.conv_transpose3d
    else:
        raise Exception(
            'The convolutional input is either 3, 4 or 5 dimensions. input.dim = '
             + str(input.dim()))
    return convfunc(input, cat_kernels_4_quaternion, bias, stride, padding,
        output_padding, groups, dilatation)


class QuaternionTransposeConv(Module):
    """Applies a Quaternion Transposed Convolution (or Deconvolution) to the incoming data.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride,
        dilatation=1, padding=0, output_padding=0, groups=1, bias=True,
        init_criterion='he', weight_init='quaternion', seed=None, operation
        ='convolution2d', rotation=False, quaternion_format=False):
        super(QuaternionTransposeConv, self).__init__()
        self.in_channels = in_channels // 4
        self.out_channels = out_channels // 4
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.dilatation = dilatation
        self.init_criterion = init_criterion
        self.weight_init = weight_init
        self.seed = seed if seed is not None else np.random.randint(0, 1234)
        self.rng = RandomState(self.seed)
        self.operation = operation
        self.rotation = rotation
        self.quaternion_format = quaternion_format
        self.winit = {'quaternion': quaternion_init, 'unitary':
            unitary_init, 'random': random_init}[self.weight_init]
        self.kernel_size, self.w_shape = get_kernel_and_weight_shape(self.
            operation, self.out_channels, self.in_channels, kernel_size)
        self.r_weight = Parameter(torch.Tensor(*self.w_shape))
        self.i_weight = Parameter(torch.Tensor(*self.w_shape))
        self.j_weight = Parameter(torch.Tensor(*self.w_shape))
        self.k_weight = Parameter(torch.Tensor(*self.w_shape))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        affect_init_conv(self.r_weight, self.i_weight, self.j_weight, self.
            k_weight, self.kernel_size, self.winit, self.rng, self.
            init_criterion)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, input):
        if self.rotation:
            return quaternion_tranpose_conv_rotation(input, self.r_weight,
                self.i_weight, self.j_weight, self.k_weight, self.bias,
                self.stride, self.padding, self.output_padding, self.groups,
                self.dilatation, self.quaternion_format)
        else:
            return quaternion_transpose_conv(input, self.r_weight, self.
                i_weight, self.j_weight, self.k_weight, self.bias, self.
                stride, self.padding, self.output_padding, self.groups,
                self.dilatation)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'in_channels=' + str(self.
            in_channels) + ', out_channels=' + str(self.out_channels
            ) + ', bias=' + str(self.bias is not None
            ) + ', kernel_size=' + str(self.kernel_size) + ', stride=' + str(
            self.stride) + ', padding=' + str(self.padding
            ) + ', dilation=' + str(self.dilation) + ', init_criterion=' + str(
            self.init_criterion) + ', weight_init=' + str(self.weight_init
            ) + ', seed=' + str(self.seed) + ', operation=' + str(self.
            operation) + ')'


def quaternion_conv_rotation(input, zero_kernel, r_weight, i_weight,
    j_weight, k_weight, bias, stride, padding, groups, dilatation,
    quaternion_format, scale=None):
    """
    Applies a quaternion rotation and convolution transformation to the incoming data:

    The rotation W*x*W^t can be replaced by R*x following:
    https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation

    Works for unitary and non unitary weights.

    The initial size of the input must be a multiple of 3 if quaternion_format = False and
    4 if quaternion_format = True.
    """
    square_r = r_weight * r_weight
    square_i = i_weight * i_weight
    square_j = j_weight * j_weight
    square_k = k_weight * k_weight
    norm = torch.sqrt(square_r + square_i + square_j + square_k + 0.0001)
    r_n_weight = r_weight / norm
    i_n_weight = i_weight / norm
    j_n_weight = j_weight / norm
    k_n_weight = k_weight / norm
    norm_factor = 2.0
    square_i = norm_factor * (i_n_weight * i_n_weight)
    square_j = norm_factor * (j_n_weight * j_n_weight)
    square_k = norm_factor * (k_n_weight * k_n_weight)
    ri = norm_factor * r_n_weight * i_n_weight
    rj = norm_factor * r_n_weight * j_n_weight
    rk = norm_factor * r_n_weight * k_n_weight
    ij = norm_factor * i_n_weight * j_n_weight
    ik = norm_factor * i_n_weight * k_n_weight
    jk = norm_factor * j_n_weight * k_n_weight
    if quaternion_format:
        if scale is not None:
            rot_kernel_1 = torch.cat([zero_kernel, scale * (1.0 - (square_j +
                square_k)), scale * (ij - rk), scale * (ik + rj)], dim=1)
            rot_kernel_2 = torch.cat([zero_kernel, scale * (ij + rk), scale *
                (1.0 - (square_i + square_k)), scale * (jk - ri)], dim=1)
            rot_kernel_3 = torch.cat([zero_kernel, scale * (ik - rj), scale *
                (jk + ri), scale * (1.0 - (square_i + square_j))], dim=1)
        else:
            rot_kernel_1 = torch.cat([zero_kernel, 1.0 - (square_j +
                square_k), ij - rk, ik + rj], dim=1)
            rot_kernel_2 = torch.cat([zero_kernel, ij + rk, 1.0 - (square_i +
                square_k), jk - ri], dim=1)
            rot_kernel_3 = torch.cat([zero_kernel, ik - rj, jk + ri, 1.0 -
                (square_i + square_j)], dim=1)
        zero_kernel2 = torch.cat([zero_kernel, zero_kernel, zero_kernel,
            zero_kernel], dim=1)
        global_rot_kernel = torch.cat([zero_kernel2, rot_kernel_1,
            rot_kernel_2, rot_kernel_3], dim=0)
    else:
        if scale is not None:
            rot_kernel_1 = torch.cat([scale * (1.0 - (square_j + square_k)),
                scale * (ij - rk), scale * (ik + rj)], dim=0)
            rot_kernel_2 = torch.cat([scale * (ij + rk), scale * (1.0 - (
                square_i + square_k)), scale * (jk - ri)], dim=0)
            rot_kernel_3 = torch.cat([scale * (ik - rj), scale * (jk + ri),
                scale * (1.0 - (square_i + square_j))], dim=0)
        else:
            rot_kernel_1 = torch.cat([1.0 - (square_j + square_k), ij - rk,
                ik + rj], dim=0)
            rot_kernel_2 = torch.cat([ij + rk, 1.0 - (square_i + square_k),
                jk - ri], dim=0)
            rot_kernel_3 = torch.cat([ik - rj, jk + ri, 1.0 - (square_i +
                square_j)], dim=0)
        global_rot_kernel = torch.cat([rot_kernel_1, rot_kernel_2,
            rot_kernel_3], dim=0)
    if input.dim() == 3:
        convfunc = F.conv1d
    elif input.dim() == 4:
        convfunc = F.conv2d
    elif input.dim() == 5:
        convfunc = F.conv3d
    else:
        raise Exception(
            'The convolutional input is either 3, 4 or 5 dimensions. input.dim = '
             + str(input.dim()))
    return convfunc(input, global_rot_kernel, bias, stride, padding,
        dilatation, groups)


def quaternion_conv(input, r_weight, i_weight, j_weight, k_weight, bias,
    stride, padding, groups, dilatation):
    """
    Applies a quaternion convolution to the incoming data:
    """
    cat_kernels_4_r = torch.cat([r_weight, -i_weight, -j_weight, -k_weight],
        dim=1)
    cat_kernels_4_i = torch.cat([i_weight, r_weight, -k_weight, j_weight],
        dim=1)
    cat_kernels_4_j = torch.cat([j_weight, k_weight, r_weight, -i_weight],
        dim=1)
    cat_kernels_4_k = torch.cat([k_weight, -j_weight, i_weight, r_weight],
        dim=1)
    cat_kernels_4_quaternion = torch.cat([cat_kernels_4_r, cat_kernels_4_i,
        cat_kernels_4_j, cat_kernels_4_k], dim=0)
    if input.dim() == 3:
        convfunc = F.conv1d
    elif input.dim() == 4:
        convfunc = F.conv2d
    elif input.dim() == 5:
        convfunc = F.conv3d
    else:
        raise Exception(
            'The convolutional input is either 3, 4 or 5 dimensions. input.dim = '
             + str(input.dim()))
    return convfunc(input, cat_kernels_4_quaternion, bias, stride, padding,
        dilatation, groups)


class QuaternionConv(Module):
    """Applies a Quaternion Convolution to the incoming data.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride,
        dilatation=1, padding=0, groups=1, bias=True, init_criterion=
        'glorot', weight_init='quaternion', seed=None, operation=
        'convolution2d', rotation=False, quaternion_format=True, scale=False):
        super(QuaternionConv, self).__init__()
        self.in_channels = in_channels // 4
        self.out_channels = out_channels // 4
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.dilatation = dilatation
        self.init_criterion = init_criterion
        self.weight_init = weight_init
        self.seed = seed if seed is not None else np.random.randint(0, 1234)
        self.rng = RandomState(self.seed)
        self.operation = operation
        self.rotation = rotation
        self.quaternion_format = quaternion_format
        self.winit = {'quaternion': quaternion_init, 'unitary':
            unitary_init, 'random': random_init}[self.weight_init]
        self.scale = scale
        self.kernel_size, self.w_shape = get_kernel_and_weight_shape(self.
            operation, self.in_channels, self.out_channels, kernel_size)
        self.r_weight = Parameter(torch.Tensor(*self.w_shape))
        self.i_weight = Parameter(torch.Tensor(*self.w_shape))
        self.j_weight = Parameter(torch.Tensor(*self.w_shape))
        self.k_weight = Parameter(torch.Tensor(*self.w_shape))
        if self.scale:
            self.scale_param = Parameter(torch.Tensor(self.r_weight.shape))
        else:
            self.scale_param = None
        if self.rotation:
            self.zero_kernel = Parameter(torch.zeros(self.r_weight.shape),
                requires_grad=False)
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        affect_init_conv(self.r_weight, self.i_weight, self.j_weight, self.
            k_weight, self.kernel_size, self.winit, self.rng, self.
            init_criterion)
        if self.scale_param is not None:
            torch.nn.init.xavier_uniform_(self.scale_param.data)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, input):
        if self.rotation:
            return quaternion_conv_rotation(input, self.zero_kernel, self.
                r_weight, self.i_weight, self.j_weight, self.k_weight, self
                .bias, self.stride, self.padding, self.groups, self.
                dilatation, self.quaternion_format, self.scale_param)
        else:
            return quaternion_conv(input, self.r_weight, self.i_weight,
                self.j_weight, self.k_weight, self.bias, self.stride, self.
                padding, self.groups, self.dilatation)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'in_channels=' + str(self.
            in_channels) + ', out_channels=' + str(self.out_channels
            ) + ', bias=' + str(self.bias is not None
            ) + ', kernel_size=' + str(self.kernel_size) + ', stride=' + str(
            self.stride) + ', padding=' + str(self.padding
            ) + ', init_criterion=' + str(self.init_criterion
            ) + ', weight_init=' + str(self.weight_init) + ', seed=' + str(self
            .seed) + ', rotation=' + str(self.rotation) + ', q_format=' + str(
            self.quaternion_format) + ', operation=' + str(self.operation
            ) + ')'


def quaternion_linear(input, r_weight, i_weight, j_weight, k_weight, bias=True
    ):
    """
    Applies a quaternion linear transformation to the incoming data:

    It is important to notice that the forward phase of a QNN is defined
    as W * Inputs (with * equal to the Hamilton product). The constructed
    cat_kernels_4_quaternion is a modified version of the quaternion representation
    so when we do torch.mm(Input,W) it's equivalent to W * Inputs.

    """
    cat_kernels_4_r = torch.cat([r_weight, -i_weight, -j_weight, -k_weight],
        dim=0)
    cat_kernels_4_i = torch.cat([i_weight, r_weight, -k_weight, j_weight],
        dim=0)
    cat_kernels_4_j = torch.cat([j_weight, k_weight, r_weight, -i_weight],
        dim=0)
    cat_kernels_4_k = torch.cat([k_weight, -j_weight, i_weight, r_weight],
        dim=0)
    cat_kernels_4_quaternion = torch.cat([cat_kernels_4_r, cat_kernels_4_i,
        cat_kernels_4_j, cat_kernels_4_k], dim=1)
    if input.dim() == 2:
        if bias is not None:
            return torch.addmm(bias, input, cat_kernels_4_quaternion)
        else:
            return torch.mm(input, cat_kernels_4_quaternion)
    else:
        output = torch.matmul(input, cat_kernels_4_quaternion)
        if bias is not None:
            return output + bias
        else:
            return output


def quaternion_linear_rotation(input, zero_kernel, r_weight, i_weight,
    j_weight, k_weight, bias=None, quaternion_format=False, scale=None):
    """
    Applies a quaternion rotation transformation to the incoming data:

    The rotation W*x*W^t can be replaced by R*x following:
    https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation

    Works for unitary and non unitary weights.

    The initial size of the input must be a multiple of 3 if quaternion_format = False and
    4 if quaternion_format = True.
    """
    square_r = r_weight * r_weight
    square_i = i_weight * i_weight
    square_j = j_weight * j_weight
    square_k = k_weight * k_weight
    norm = torch.sqrt(square_r + square_i + square_j + square_k + 0.0001)
    r_n_weight = r_weight / norm
    i_n_weight = i_weight / norm
    j_n_weight = j_weight / norm
    k_n_weight = k_weight / norm
    norm_factor = 2.0
    square_i = norm_factor * (i_n_weight * i_n_weight)
    square_j = norm_factor * (j_n_weight * j_n_weight)
    square_k = norm_factor * (k_n_weight * k_n_weight)
    ri = norm_factor * r_n_weight * i_n_weight
    rj = norm_factor * r_n_weight * j_n_weight
    rk = norm_factor * r_n_weight * k_n_weight
    ij = norm_factor * i_n_weight * j_n_weight
    ik = norm_factor * i_n_weight * k_n_weight
    jk = norm_factor * j_n_weight * k_n_weight
    if quaternion_format:
        if scale is not None:
            rot_kernel_1 = torch.cat([zero_kernel, scale * (1.0 - (square_j +
                square_k)), scale * (ij - rk), scale * (ik + rj)], dim=0)
            rot_kernel_2 = torch.cat([zero_kernel, scale * (ij + rk), scale *
                (1.0 - (square_i + square_k)), scale * (jk - ri)], dim=0)
            rot_kernel_3 = torch.cat([zero_kernel, scale * (ik - rj), scale *
                (jk + ri), scale * (1.0 - (square_i + square_j))], dim=0)
        else:
            rot_kernel_1 = torch.cat([zero_kernel, 1.0 - (square_j +
                square_k), ij - rk, ik + rj], dim=0)
            rot_kernel_2 = torch.cat([zero_kernel, ij + rk, 1.0 - (square_i +
                square_k), jk - ri], dim=0)
            rot_kernel_3 = torch.cat([zero_kernel, ik - rj, jk + ri, 1.0 -
                (square_i + square_j)], dim=0)
        zero_kernel2 = torch.cat([zero_kernel, zero_kernel, zero_kernel,
            zero_kernel], dim=0)
        global_rot_kernel = torch.cat([zero_kernel2, rot_kernel_1,
            rot_kernel_2, rot_kernel_3], dim=1)
    else:
        if scale is not None:
            rot_kernel_1 = torch.cat([scale * (1.0 - (square_j + square_k)),
                scale * (ij - rk), scale * (ik + rj)], dim=0)
            rot_kernel_2 = torch.cat([scale * (ij + rk), scale * (1.0 - (
                square_i + square_k)), scale * (jk - ri)], dim=0)
            rot_kernel_3 = torch.cat([scale * (ik - rj), scale * (jk + ri),
                scale * (1.0 - (square_i + square_j))], dim=0)
        else:
            rot_kernel_1 = torch.cat([1.0 - (square_j + square_k), ij - rk,
                ik + rj], dim=0)
            rot_kernel_2 = torch.cat([ij + rk, 1.0 - (square_i + square_k),
                jk - ri], dim=0)
            rot_kernel_3 = torch.cat([ik - rj, jk + ri, 1.0 - (square_i +
                square_j)], dim=0)
        global_rot_kernel = torch.cat([rot_kernel_1, rot_kernel_2,
            rot_kernel_3], dim=1)
    if input.dim() == 2:
        if bias is not None:
            return torch.addmm(bias, input, global_rot_kernel)
        else:
            return torch.mm(input, global_rot_kernel)
    else:
        output = torch.matmul(input, global_rot_kernel)
        if bias is not None:
            return output + bias
        else:
            return output


def affect_init(r_weight, i_weight, j_weight, k_weight, init_func, rng,
    init_criterion):
    if r_weight.size() != i_weight.size() or r_weight.size() != j_weight.size(
        ) or r_weight.size() != k_weight.size():
        raise ValueError(
            'The real and imaginary weights should have the same size . Found: r:'
             + str(r_weight.size()) + ' i:' + str(i_weight.size()) + ' j:' +
            str(j_weight.size()) + ' k:' + str(k_weight.size()))
    elif r_weight.dim() != 2:
        raise Exception(
            'affect_init accepts only matrices. Found dimension = ' + str(
            r_weight.dim()))
    kernel_size = None
    r, i, j, k = init_func(r_weight.size(0), r_weight.size(1), rng,
        kernel_size, init_criterion)
    r, i, j, k = torch.from_numpy(r), torch.from_numpy(i), torch.from_numpy(j
        ), torch.from_numpy(k)
    r_weight.data = r.type_as(r_weight.data)
    i_weight.data = i.type_as(i_weight.data)
    j_weight.data = j.type_as(j_weight.data)
    k_weight.data = k.type_as(k_weight.data)


class QuaternionLinearAutograd(Module):
    """Applies a quaternion linear transformation to the incoming data. A custom
    Autograd function is call to drastically reduce the VRAM consumption. Nonetheless, computing
    time is also slower compared to QuaternionLinear().
    """

    def __init__(self, in_features, out_features, bias=True, init_criterion
        ='glorot', weight_init='quaternion', seed=None, rotation=False,
        quaternion_format=True, scale=False):
        super(QuaternionLinearAutograd, self).__init__()
        self.in_features = in_features // 4
        self.out_features = out_features // 4
        self.rotation = rotation
        self.quaternion_format = quaternion_format
        self.r_weight = Parameter(torch.Tensor(self.in_features, self.
            out_features))
        self.i_weight = Parameter(torch.Tensor(self.in_features, self.
            out_features))
        self.j_weight = Parameter(torch.Tensor(self.in_features, self.
            out_features))
        self.k_weight = Parameter(torch.Tensor(self.in_features, self.
            out_features))
        self.scale = scale
        if self.scale:
            self.scale_param = Parameter(torch.Tensor(self.in_features,
                self.out_features))
        else:
            self.scale_param = None
        if self.rotation:
            self.zero_kernel = Parameter(torch.zeros(self.r_weight.shape),
                requires_grad=False)
        if bias:
            self.bias = Parameter(torch.Tensor(self.out_features * 4))
        else:
            self.register_parameter('bias', None)
        self.init_criterion = init_criterion
        self.weight_init = weight_init
        self.seed = seed if seed is not None else np.random.randint(0, 1234)
        self.rng = RandomState(self.seed)
        self.reset_parameters()

    def reset_parameters(self):
        winit = {'quaternion': quaternion_init, 'unitary': unitary_init,
            'random': random_init}[self.weight_init]
        if self.scale_param is not None:
            torch.nn.init.xavier_uniform_(self.scale_param.data)
        if self.bias is not None:
            self.bias.data.fill_(0)
        affect_init(self.r_weight, self.i_weight, self.j_weight, self.
            k_weight, winit, self.rng, self.init_criterion)

    def forward(self, input):
        if self.rotation:
            return quaternion_linear_rotation(input, self.zero_kernel, self
                .r_weight, self.i_weight, self.j_weight, self.k_weight,
                self.bias, self.quaternion_format, self.scale_param)
        else:
            return quaternion_linear(input, self.r_weight, self.i_weight,
                self.j_weight, self.k_weight, self.bias)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'in_features=' + str(self.
            in_features) + ', out_features=' + str(self.out_features
            ) + ', bias=' + str(self.bias is not None
            ) + ', init_criterion=' + str(self.init_criterion
            ) + ', weight_init=' + str(self.weight_init) + ', rotation=' + str(
            self.rotation) + ', seed=' + str(self.seed) + ')'


def get_i(input):
    if input.dim() < 4:
        nb_hidden = input.size()[-1]
    else:
        nb_hidden = input.size()[1]
    if input.dim() == 2:
        return input.narrow(1, nb_hidden // 4, nb_hidden // 4)
    if input.dim() == 3:
        return input.narrow(2, nb_hidden // 4, nb_hidden // 4)
    if input.dim() >= 4:
        return input.narrow(1, nb_hidden // 4, nb_hidden // 4)


def check_input(input):
    if input.dim() not in {2, 3, 4, 5}:
        raise RuntimeError(
            'Quaternion linear accepts only input of dimension 2 or 3. Quaternion conv accepts up to 5 dim  input.dim = '
             + str(input.dim()))
    if input.dim() < 4:
        nb_hidden = input.size()[-1]
    else:
        nb_hidden = input.size()[1]
    if nb_hidden % 4 != 0:
        raise RuntimeError(
            'Quaternion Tensors must be divisible by 4. input.size()[1] = ' +
            str(nb_hidden))


def get_r(input):
    check_input(input)
    if input.dim() < 4:
        nb_hidden = input.size()[-1]
    else:
        nb_hidden = input.size()[1]
    if input.dim() == 2:
        return input.narrow(1, 0, nb_hidden // 4)
    if input.dim() == 3:
        return input.narrow(2, 0, nb_hidden // 4)
    if input.dim() >= 4:
        return input.narrow(1, 0, nb_hidden // 4)


def get_j(input):
    check_input(input)
    if input.dim() < 4:
        nb_hidden = input.size()[-1]
    else:
        nb_hidden = input.size()[1]
    if input.dim() == 2:
        return input.narrow(1, nb_hidden // 2, nb_hidden // 4)
    if input.dim() == 3:
        return input.narrow(2, nb_hidden // 2, nb_hidden // 4)
    if input.dim() >= 4:
        return input.narrow(1, nb_hidden // 2, nb_hidden // 4)


def get_k(input):
    check_input(input)
    if input.dim() < 4:
        nb_hidden = input.size()[-1]
    else:
        nb_hidden = input.size()[1]
    if input.dim() == 2:
        return input.narrow(1, nb_hidden - nb_hidden // 4, nb_hidden // 4)
    if input.dim() == 3:
        return input.narrow(2, nb_hidden - nb_hidden // 4, nb_hidden // 4)
    if input.dim() >= 4:
        return input.narrow(1, nb_hidden - nb_hidden // 4, nb_hidden // 4)


class QuaternionLinearFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, r_weight, i_weight, j_weight, k_weight, bias=None):
        ctx.save_for_backward(input, r_weight, i_weight, j_weight, k_weight,
            bias)
        check_input(input)
        cat_kernels_4_r = torch.cat([r_weight, -i_weight, -j_weight, -
            k_weight], dim=0)
        cat_kernels_4_i = torch.cat([i_weight, r_weight, -k_weight,
            j_weight], dim=0)
        cat_kernels_4_j = torch.cat([j_weight, k_weight, r_weight, -
            i_weight], dim=0)
        cat_kernels_4_k = torch.cat([k_weight, -j_weight, i_weight,
            r_weight], dim=0)
        cat_kernels_4_quaternion = torch.cat([cat_kernels_4_r,
            cat_kernels_4_i, cat_kernels_4_j, cat_kernels_4_k], dim=1)
        if input.dim() == 2:
            if bias is not None:
                return torch.addmm(bias, input, cat_kernels_4_quaternion)
            else:
                return torch.mm(input, cat_kernels_4_quaternion)
        else:
            output = torch.matmul(input, cat_kernels_4_quaternion)
            if bias is not None:
                return output + bias
            else:
                return output

    @staticmethod
    def backward(ctx, grad_output):
        input, r_weight, i_weight, j_weight, k_weight, bias = ctx.saved_tensors
        (grad_input) = (grad_weight_r) = (grad_weight_i) = (grad_weight_j) = (
            grad_weight_k) = (grad_bias) = None
        input_r = torch.cat([r_weight, -i_weight, -j_weight, -k_weight], dim=0)
        input_i = torch.cat([i_weight, r_weight, -k_weight, j_weight], dim=0)
        input_j = torch.cat([j_weight, k_weight, r_weight, -i_weight], dim=0)
        input_k = torch.cat([k_weight, -j_weight, i_weight, r_weight], dim=0)
        cat_kernels_4_quaternion_T = Variable(torch.cat([input_r, input_i,
            input_j, input_k], dim=1).permute(1, 0), requires_grad=False)
        r = get_r(input)
        i = get_i(input)
        j = get_j(input)
        k = get_k(input)
        input_r = torch.cat([r, -i, -j, -k], dim=0)
        input_i = torch.cat([i, r, -k, j], dim=0)
        input_j = torch.cat([j, k, r, -i], dim=0)
        input_k = torch.cat([k, -j, i, r], dim=0)
        input_mat = Variable(torch.cat([input_r, input_i, input_j, input_k],
            dim=1), requires_grad=False)
        r = get_r(grad_output)
        i = get_i(grad_output)
        j = get_j(grad_output)
        k = get_k(grad_output)
        input_r = torch.cat([r, i, j, k], dim=1)
        input_i = torch.cat([-i, r, k, -j], dim=1)
        input_j = torch.cat([-j, -k, r, i], dim=1)
        input_k = torch.cat([-k, j, -i, r], dim=1)
        grad_mat = torch.cat([input_r, input_i, input_j, input_k], dim=0)
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(cat_kernels_4_quaternion_T)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_mat.permute(1, 0).mm(input_mat).permute(1, 0)
            unit_size_x = r_weight.size(0)
            unit_size_y = r_weight.size(1)
            grad_weight_r = grad_weight.narrow(0, 0, unit_size_x).narrow(1,
                0, unit_size_y)
            grad_weight_i = grad_weight.narrow(0, 0, unit_size_x).narrow(1,
                unit_size_y, unit_size_y)
            grad_weight_j = grad_weight.narrow(0, 0, unit_size_x).narrow(1,
                unit_size_y * 2, unit_size_y)
            grad_weight_k = grad_weight.narrow(0, 0, unit_size_x).narrow(1,
                unit_size_y * 3, unit_size_y)
        if ctx.needs_input_grad[5]:
            grad_bias = grad_output.sum(0).squeeze(0)
        return (grad_input, grad_weight_r, grad_weight_i, grad_weight_j,
            grad_weight_k, grad_bias)


class QuaternionLinear(Module):
    """Applies a quaternion linear transformation to the incoming data.
    """

    def __init__(self, in_features, out_features, bias=True, init_criterion
        ='he', weight_init='quaternion', seed=None):
        super(QuaternionLinear, self).__init__()
        self.in_features = in_features // 4
        self.out_features = out_features // 4
        self.r_weight = Parameter(torch.Tensor(self.in_features, self.
            out_features))
        self.i_weight = Parameter(torch.Tensor(self.in_features, self.
            out_features))
        self.j_weight = Parameter(torch.Tensor(self.in_features, self.
            out_features))
        self.k_weight = Parameter(torch.Tensor(self.in_features, self.
            out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(self.out_features * 4))
        else:
            self.register_parameter('bias', None)
        self.init_criterion = init_criterion
        self.weight_init = weight_init
        self.seed = seed if seed is not None else np.random.randint(0, 1234)
        self.rng = RandomState(self.seed)
        self.reset_parameters()

    def reset_parameters(self):
        winit = {'quaternion': quaternion_init, 'unitary': unitary_init}[
            self.weight_init]
        if self.bias is not None:
            self.bias.data.fill_(0)
        affect_init(self.r_weight, self.i_weight, self.j_weight, self.
            k_weight, winit, self.rng, self.init_criterion)

    def forward(self, input):
        if input.dim() == 3:
            T, N, C = input.size()
            input = input.view(T * N, C)
            output = QuaternionLinearFunction.apply(input, self.r_weight,
                self.i_weight, self.j_weight, self.k_weight, self.bias)
            output = output.view(T, N, output.size(1))
        elif input.dim() == 2:
            output = QuaternionLinearFunction.apply(input, self.r_weight,
                self.i_weight, self.j_weight, self.k_weight, self.bias)
        else:
            raise NotImplementedError
        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'in_features=' + str(self.
            in_features) + ', out_features=' + str(self.out_features
            ) + ', bias=' + str(self.bias is not None
            ) + ', init_criterion=' + str(self.init_criterion
            ) + ', weight_init=' + str(self.weight_init) + ', seed=' + str(self
            .seed) + ')'


class StackedQLSTM(nn.Module):

    def __init__(self, feat_size, hidden_size, use_cuda, n_layers,
        batch_first=True):
        super(StackedQLSTM, self).__init__()
        self.batch_first = batch_first
        self.layers = nn.ModuleList([QLSTM(feat_size, hidden_size, use_cuda
            ) for _ in range(n_layers)])

    def forward(self, x):
        if self.batch_first:
            x = x.permute(1, 0, 2)
        for layer in self.layers:
            x = layer(x)
            x = x[:, :, :-1]
        if self.batch_first:
            x = x.permute(1, 0, 2)
        return x


class QLSTM(nn.Module):

    def __init__(self, feat_size, hidden_size, CUDA):
        super(QLSTM, self).__init__()
        self.act = nn.Tanh()
        self.act_gate = nn.Sigmoid()
        self.input_dim = feat_size
        self.hidden_dim = hidden_size
        self.CUDA = CUDA
        self.num_classes = feat_size + 1
        self.wfx = QuaternionLinearAutograd(self.input_dim, self.hidden_dim)
        self.ufh = QuaternionLinearAutograd(self.hidden_dim, self.
            hidden_dim, bias=False)
        self.wix = QuaternionLinearAutograd(self.input_dim, self.hidden_dim)
        self.uih = QuaternionLinearAutograd(self.hidden_dim, self.
            hidden_dim, bias=False)
        self.wox = QuaternionLinearAutograd(self.input_dim, self.hidden_dim)
        self.uoh = QuaternionLinearAutograd(self.hidden_dim, self.
            hidden_dim, bias=False)
        self.wcx = QuaternionLinearAutograd(self.input_dim, self.hidden_dim)
        self.uch = QuaternionLinearAutograd(self.hidden_dim, self.
            hidden_dim, bias=False)
        self.fco = nn.Linear(self.hidden_dim, self.num_classes)
        self.adam = torch.optim.Adam(self.parameters(), lr=0.005)

    def forward(self, x):
        h_init = Variable(torch.zeros(x.shape[1], self.hidden_dim))
        if self.CUDA:
            x = x
            h_init = h_init
        wfx_out = self.wfx(x)
        wix_out = self.wix(x)
        wox_out = self.wox(x)
        wcx_out = self.wcx(x)
        out = []
        c = h_init
        h = h_init
        for k in range(x.shape[0]):
            ft = self.act_gate(wfx_out[k] + self.ufh(h))
            it = self.act_gate(wix_out[k] + self.uih(h))
            ot = self.act_gate(wox_out[k] + self.uoh(h))
            at = wcx_out[k] + self.uch(h)
            c = it * self.act(at) + ft * c
            h = ot * self.act(c)
            output = self.fco(h)
            out.append(output.unsqueeze(0))
        return torch.cat(out, 0)


class QAE(nn.Module):

    def __init__(self):
        super(QAE, self).__init__()
        self.act = nn.ReLU()
        self.output_act = nn.Sigmoid()
        self.e1 = QuaternionLinear(65536, 4096)
        self.d1 = QuaternionLinear(4096, 65536)

    def forward(self, x):
        e1 = self.act(self.e1(x))
        d1 = self.d1(e1)
        out = self.output_act(d1)
        return out

    def name(self):
        return 'QAE'


class QCAE(nn.Module):

    def __init__(self):
        super(QCAE, self).__init__()
        self.act = nn.Hardtanh()
        self.output_act = nn.Hardtanh()
        self.e1 = QuaternionConv(4, 32, kernel_size=3, stride=2, padding=1)
        self.e2 = QuaternionConv(32, 40, kernel_size=3, stride=2, padding=1)
        self.d5 = QuaternionTransposeConv(40, 32, kernel_size=3, stride=2,
            padding=1, output_padding=1)
        self.d6 = QuaternionTransposeConv(32, 4, kernel_size=3, stride=2,
            padding=1, output_padding=1)

    def forward(self, x):
        e1 = self.act(self.e1(x))
        e2 = self.act(self.e2(e1))
        d5 = self.act(self.d5(e2))
        d6 = self.d6(d5)
        out = self.output_act(d6)
        return out

    def name(self):
        return 'QCAE'


class CAE(nn.Module):

    def __init__(self):
        super(CAE, self).__init__()
        self.act = nn.Hardtanh()
        self.output_act = nn.Hardtanh()
        self.e1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.e2 = nn.Conv2d(32, 40, kernel_size=3, stride=2, padding=1)
        self.d1 = nn.ConvTranspose2d(40, 32, kernel_size=3, stride=2,
            padding=1, output_padding=1)
        self.d2 = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2,
            padding=1, output_padding=1)
        torch.nn.init.xavier_uniform_(self.e1.weight)
        torch.nn.init.xavier_uniform_(self.e2.weight)
        torch.nn.init.xavier_uniform_(self.d1.weight)
        torch.nn.init.xavier_uniform_(self.d2.weight)
        self.e1.bias.data.fill_(0.0)
        self.e2.bias.data.fill_(0.0)
        self.d1.bias.data.fill_(0.0)
        self.d2.bias.data.fill_(0.0)

    def forward(self, x):
        e1 = self.act(self.e1(x))
        e2 = self.act(self.e2(e1))
        d1 = self.act(self.d1(e2))
        d2 = self.d2(d1)
        out = self.output_act(d2)
        return out

    def name(self):
        return 'CAE'


class QRNN(nn.Module):

    def __init__(self, feat_size, hidden_size, CUDA):
        super(QRNN, self).__init__()
        self.input_dim = feat_size
        self.hidden_dim = hidden_size
        self.num_classes = feat_size
        self.CUDA = CUDA
        self.wx = QuaternionLinearAutograd(self.input_dim, self.hidden_dim)
        self.uh = QuaternionLinearAutograd(self.hidden_dim, self.hidden_dim)
        self.fco = nn.Linear(curr_dim, self.num_classes)
        self.adam = torch.optim.Adam(self.parameters(), lr=0.005)

    def forward(self, x):
        h_init = Variable(torch.zeros(x.shape[1], self.hidden_dim))
        if self.CUDA:
            x = x
            h_init = h_init
        wx_out = self.wx(x)
        h = h_init
        out = []
        for k in range(x.shape[0]):
            at = wx_out[k] + self.uh(h)
            h = at
            output = nn.Tanh()(self.fco(h))
            out.append(output.unsqueeze(0))
        return torch.cat(out, 0)


class QLSTM(nn.Module):

    def __init__(self, feat_size, hidden_size, CUDA):
        super(QLSTM, self).__init__()
        self.act = nn.Tanh()
        self.act_gate = nn.Sigmoid()
        self.input_dim = feat_size
        self.hidden_dim = hidden_size
        self.CUDA = CUDA
        self.num_classes = feat_size + 1
        self.wfx = QuaternionLinearAutograd(self.input_dim, self.hidden_dim)
        self.ufh = QuaternionLinearAutograd(self.hidden_dim, self.
            hidden_dim, bias=False)
        self.wix = QuaternionLinearAutograd(self.input_dim, self.hidden_dim)
        self.uih = QuaternionLinearAutograd(self.hidden_dim, self.
            hidden_dim, bias=False)
        self.wox = QuaternionLinearAutograd(self.input_dim, self.hidden_dim)
        self.uoh = QuaternionLinearAutograd(self.hidden_dim, self.
            hidden_dim, bias=False)
        self.wcx = QuaternionLinearAutograd(self.input_dim, self.hidden_dim)
        self.uch = QuaternionLinearAutograd(self.hidden_dim, self.
            hidden_dim, bias=False)
        self.fco = nn.Linear(self.hidden_dim, self.num_classes)
        self.adam = torch.optim.Adam(self.parameters(), lr=0.005)

    def forward(self, x):
        h_init = Variable(torch.zeros(x.shape[1], self.hidden_dim))
        if self.CUDA:
            x = x
            h_init = h_init
        wfx_out = self.wfx(x)
        wix_out = self.wix(x)
        wox_out = self.wox(x)
        wcx_out = self.wcx(x)
        out = []
        c = h_init
        h = h_init
        for k in range(x.shape[0]):
            ft = self.act_gate(wfx_out[k] + self.ufh(h))
            it = self.act_gate(wix_out[k] + self.uih(h))
            ot = self.act_gate(wox_out[k] + self.uoh(h))
            at = wcx_out[k] + self.uch(h)
            c = it * self.act(at) + ft * c
            h = ot * self.act(c)
            output = self.fco(h)
            out.append(output.unsqueeze(0))
        return torch.cat(out, 0)


class RNN(nn.Module):

    def __init__(self, feat_size, hidden_size, CUDA):
        super(RNN, self).__init__()
        self.input_dim = feat_size
        self.hidden_dim = hidden_size
        self.num_classes = feat_size
        self.CUDA = CUDA
        self.wx = nn.Linear(self.input_dim, self.hidden_dim)
        self.uh = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fco = nn.Linear(self.hidden_dim, self.num_classes)
        self.adam = torch.optim.Adam(self.parameters(), lr=0.0002)

    def forward(self, x):
        h_init = Variable(torch.zeros(x.shape[1], self.hidden_dim))
        if self.CUDA:
            x = x
            h_init = h_init
        wx_out = self.wx(x)
        h = h_init
        out = []
        for k in range(x.shape[0]):
            at = wx_out[k] + self.uh(h)
            h = at
            output = nn.Tanh()(self.fco(h))
            out.append(output.unsqueeze(0))
        return torch.cat(out, 0)


class LSTM(nn.Module):

    def __init__(self, feat_size, hidden_size, CUDA):
        super(LSTM, self).__init__()
        self.act = nn.Tanh()
        self.act_gate = nn.Sigmoid()
        self.input_dim = feat_size
        self.hidden_dim = hidden_size
        self.CUDA = CUDA
        self.num_classes = feat_size + 1
        self.wfx = nn.Linear(self.input_dim, self.hidden_dim)
        self.ufh = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.wix = nn.Linear(self.input_dim, self.hidden_dim)
        self.uih = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.wox = nn.Linear(self.input_dim, self.hidden_dim)
        self.uoh = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.wcx = nn.Linear(self.input_dim, self.hidden_dim)
        self.uch = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.fco = nn.Linear(self.hidden_dim, self.num_classes)
        self.adam = torch.optim.Adam(self.parameters(), lr=0.005)

    def forward(self, x):
        h_init = Variable(torch.zeros(x.shape[1], self.hidden_dim))
        if self.CUDA:
            x = x
            h_init = h_init
        wfx_out = self.wfx(x)
        wix_out = self.wix(x)
        wox_out = self.wox(x)
        wcx_out = self.wcx(x)
        out = []
        c = h_init
        h = h_init
        for k in range(x.shape[0]):
            ft = self.act_gate(wfx_out[k] + self.ufh(h))
            it = self.act_gate(wix_out[k] + self.uih(h))
            ot = self.act_gate(wox_out[k] + self.uoh(h))
            at = wcx_out[k] + self.uch(h)
            c = it * self.act(at) + ft * c
            h = ot * self.act(c)
            output = self.fco(h)
            out.append(output.unsqueeze(0))
        return torch.cat(out, 0)


class QRNN(nn.Module):

    def __init__(self, feat_size, hidden_size, CUDA):
        super(QRNN, self).__init__()
        self.input_dim = feat_size
        self.hidden_dim = hidden_size
        self.num_classes = feat_size
        self.CUDA = CUDA
        self.wx = QuaternionLinearAutograd(self.input_dim, self.hidden_dim)
        self.uh = QuaternionLinearAutograd(self.hidden_dim, self.hidden_dim)
        self.fco = nn.Linear(curr_dim, self.num_classes)
        self.adam = torch.optim.Adam(self.parameters(), lr=0.005)

    def forward(self, x):
        h_init = Variable(torch.zeros(x.shape[1], self.hidden_dim))
        if self.CUDA:
            x = x
            h_init = h_init
        wx_out = self.wx(x)
        h = h_init
        out = []
        for k in range(x.shape[0]):
            at = wx_out[k] + self.uh(h)
            h = at
            output = nn.Tanh()(self.fco(h))
            out.append(output.unsqueeze(0))
        return torch.cat(out, 0)


class QLSTM(nn.Module):

    def __init__(self, feat_size, hidden_size, CUDA):
        super(QLSTM, self).__init__()
        self.act = nn.Tanh()
        self.act_gate = nn.Sigmoid()
        self.input_dim = feat_size
        self.hidden_dim = hidden_size
        self.CUDA = CUDA
        self.num_classes = feat_size + 1
        self.wfx = QuaternionLinearAutograd(self.input_dim, self.hidden_dim)
        self.ufh = QuaternionLinearAutograd(self.hidden_dim, self.
            hidden_dim, bias=False)
        self.wix = QuaternionLinearAutograd(self.input_dim, self.hidden_dim)
        self.uih = QuaternionLinearAutograd(self.hidden_dim, self.
            hidden_dim, bias=False)
        self.wox = QuaternionLinearAutograd(self.input_dim, self.hidden_dim)
        self.uoh = QuaternionLinearAutograd(self.hidden_dim, self.
            hidden_dim, bias=False)
        self.wcx = QuaternionLinearAutograd(self.input_dim, self.hidden_dim)
        self.uch = QuaternionLinearAutograd(self.hidden_dim, self.
            hidden_dim, bias=False)
        self.fco = nn.Linear(self.hidden_dim, self.num_classes)
        self.adam = torch.optim.Adam(self.parameters(), lr=0.005)

    def forward(self, x):
        h_init = Variable(torch.zeros(x.shape[1], self.hidden_dim))
        if self.CUDA:
            x = x
            h_init = h_init
        wfx_out = self.wfx(x)
        wix_out = self.wix(x)
        wox_out = self.wox(x)
        wcx_out = self.wcx(x)
        out = []
        c = h_init
        h = h_init
        for k in range(x.shape[0]):
            ft = self.act_gate(wfx_out[k] + self.ufh(h))
            it = self.act_gate(wix_out[k] + self.uih(h))
            ot = self.act_gate(wox_out[k] + self.uoh(h))
            at = wcx_out[k] + self.uch(h)
            c = it * self.act(at) + ft * c
            h = ot * self.act(c)
            output = self.fco(h)
            out.append(output.unsqueeze(0))
        return torch.cat(out, 0)


class RNN(nn.Module):

    def __init__(self, feat_size, hidden_size, CUDA):
        super(RNN, self).__init__()
        self.input_dim = feat_size
        self.hidden_dim = hidden_size
        self.num_classes = feat_size
        self.CUDA = CUDA
        self.wx = nn.Linear(self.input_dim, self.hidden_dim)
        self.uh = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fco = nn.Linear(self.hidden_dim, self.num_classes)
        self.adam = torch.optim.Adam(self.parameters(), lr=0.0002)

    def forward(self, x):
        h_init = Variable(torch.zeros(x.shape[1], self.hidden_dim))
        if self.CUDA:
            x = x
            h_init = h_init
        wx_out = self.wx(x)
        h = h_init
        out = []
        for k in range(x.shape[0]):
            at = wx_out[k] + self.uh(h)
            h = at
            output = nn.Tanh()(self.fco(h))
            out.append(output.unsqueeze(0))
        return torch.cat(out, 0)


class LSTM(nn.Module):

    def __init__(self, feat_size, hidden_size, CUDA):
        super(LSTM, self).__init__()
        self.act = nn.Tanh()
        self.act_gate = nn.Sigmoid()
        self.input_dim = feat_size
        self.hidden_dim = hidden_size
        self.CUDA = CUDA
        self.num_classes = feat_size + 1
        self.wfx = nn.Linear(self.input_dim, self.hidden_dim)
        self.ufh = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.wix = nn.Linear(self.input_dim, self.hidden_dim)
        self.uih = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.wox = nn.Linear(self.input_dim, self.hidden_dim)
        self.uoh = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.wcx = nn.Linear(self.input_dim, self.hidden_dim)
        self.uch = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.fco = nn.Linear(self.hidden_dim, self.num_classes)
        self.adam = torch.optim.Adam(self.parameters(), lr=0.005)

    def forward(self, x):
        h_init = Variable(torch.zeros(x.shape[1], self.hidden_dim))
        if self.CUDA:
            x = x
            h_init = h_init
        wfx_out = self.wfx(x)
        wix_out = self.wix(x)
        wox_out = self.wox(x)
        wcx_out = self.wcx(x)
        out = []
        c = h_init
        h = h_init
        for k in range(x.shape[0]):
            ft = self.act_gate(wfx_out[k] + self.ufh(h))
            it = self.act_gate(wix_out[k] + self.uih(h))
            ot = self.act_gate(wox_out[k] + self.uoh(h))
            at = wcx_out[k] + self.uch(h)
            c = it * self.act(at) + ft * c
            h = ot * self.act(c)
            output = self.fco(h)
            out.append(output.unsqueeze(0))
        return torch.cat(out, 0)


def q_normalize(input, channel=1):
    r = get_r(input)
    i = get_i(input)
    j = get_j(input)
    k = get_k(input)
    norm = torch.sqrt(r * r + i * i + j * j + k * k + 0.0001)
    r = r / norm
    i = i / norm
    j = j / norm
    k = k / norm
    return torch.cat([r, i, j, k], dim=channel)


class R2HQDNN(nn.Module):

    def __init__(self, proj_dim, proj_act, proj_norm, input_dim, num_classes):
        super(R2HQDNN, self).__init__()
        self.proj_dim = proj_dim
        self.proj_act = proj_act
        self.proj_norm = proj_norm
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.proj = nn.Linear(self.input_dim, self.proj_dim)
        if self.proj_act == 'tanh':
            self.p_activation = nn.Tanh()
        elif self.proj_act == 'hardtanh':
            self.p_activation = nn.Hardtanh()
        else:
            self.p_activation = nn.ReLU()
        self.fc1 = QuaternionLinearAutograd(self.proj_dim, 32)
        self.fc2 = QuaternionLinearAutograd(32, 32)
        self.fc3 = QuaternionLinearAutograd(32, 32)
        self.out = nn.Linear(32, self.num_classes)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.p_activation(self.proj(x))
        if self.proj_norm:
            x = q_normalize(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.out(x)


class R2H(nn.Module):

    def __init__(self, proj_dim, proj_act, proj_norm, input_dim):
        super(R2H, self).__init__()
        self.proj_dim = proj_dim
        self.proj_act = proj_act
        self.proj_norm = proj_norm
        self.input_dim = input_dim
        self.proj = nn.Linear(self.input_dim, self.proj_dim)
        if self.proj_act == 'tanh':
            self.p_activation = nn.Tanh()
        elif self.proj_act == 'hardtanh':
            self.p_activation = nn.Hardtanh()
        else:
            self.p_activation = nn.ReLU()
        self.out_act = nn.Tanh()
        self.d = QuaternionLinearAutograd(self.proj_dim, self.input_dim)

    def forward(self, x, trained=False):
        x = self.p_activation(self.proj(x))
        if self.proj_norm:
            x = q_normalize(x)
        if not trained:
            return self.out_act(self.d(x))
        else:
            return x


class QDNN(nn.Module):

    def __init__(self, input_dim, num_classes):
        super(QDNN, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.fc1 = QuaternionLinearAutograd(self.input_dim, 32)
        self.fc2 = QuaternionLinearAutograd(32, 32)
        self.fc3 = QuaternionLinearAutograd(32, 32)
        self.out = nn.Linear(32, self.num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.out(x)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_Orkis_Research_Pytorch_Quaternion_Neural_Networks(_paritybench_base):
    pass
    @_fails_compile()

    def test_000(self):
        self._check(QuaternionTransposeConv(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'stride': 1}), [torch.rand([4, 4, 4, 4])], {})
    @_fails_compile()

    def test_001(self):
        self._check(QuaternionConv(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'stride': 1}), [torch.rand([4, 4, 4, 4])], {})
    @_fails_compile()

    def test_002(self):
        self._check(QuaternionLinearAutograd(*[], **{'in_features': 4, 'out_features': 4}), [torch.rand([4, 4, 4, 4])], {})
    @_fails_compile()

    def test_003(self):
        self._check(QuaternionLinear(*[], **{'in_features': 4, 'out_features': 4}), [torch.rand([4, 4])], {})
    @_fails_compile()

    def test_004(self):
        self._check(QLSTM(*[], **{'feat_size': 4, 'hidden_size': 4, 'CUDA': 4}), [torch.rand([4, 4, 4, 4])], {})
    @_fails_compile()

    def test_005(self):
        self._check(QCAE(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_006(self):
        self._check(CAE(*[], **{}), [torch.rand([4, 3, 64, 64])], {})
    @_fails_compile()

    def test_007(self):
        self._check(RNN(*[], **{'feat_size': 4, 'hidden_size': 4, 'CUDA': 4}), [torch.rand([4, 4, 4, 4])], {})
    @_fails_compile()

    def test_008(self):
        self._check(LSTM(*[], **{'feat_size': 4, 'hidden_size': 4, 'CUDA': 4}), [torch.rand([4, 4, 4, 4])], {})
    @_fails_compile()

    def test_009(self):
        self._check(R2HQDNN(*[], **{'proj_dim': 4, 'proj_act': 4, 'proj_norm': 4, 'input_dim': 4, 'num_classes': 4}), [torch.rand([4, 4])], {})
    @_fails_compile()

    def test_010(self):
        self._check(R2H(*[], **{'proj_dim': 4, 'proj_act': 4, 'proj_norm': 4, 'input_dim': 4}), [torch.rand([4, 4, 4, 4])], {})
    @_fails_compile()

    def test_011(self):
        self._check(QDNN(*[], **{'input_dim': 4, 'num_classes': 4}), [torch.rand([4, 4, 4, 4])], {})
