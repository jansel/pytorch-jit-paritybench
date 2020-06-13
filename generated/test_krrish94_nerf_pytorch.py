import sys
_module = sys.modules[__name__]
del sys
cache_dataset = _module
eval_nerf = _module
lieutils = _module
nerf = _module
cfgnode = _module
load_blender = _module
load_llff = _module
metrics = _module
models = _module
nerf_helpers = _module
train_utils = _module
volume_rendering_utils = _module
tiny_nerf = _module
train_nerf = _module

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


import torch


from torch import cos


from torch import sin


import math


from typing import Optional


import numpy as np


from torch.utils.tensorboard import SummaryWriter


def get_small_and_large_angle_inds(theta: torch.Tensor, eps: float=0.001):
    """Returns the indices of small and non-small (large) angles, given
    a tensor of angles, and the threshold below (exclusive) which angles
    are considered 'small'.

    Args:
        theta (torch.Tensor): Angle (magnitude of axis-angle vector).
        eps (float): Threshold (exclusive) below which an angle is
            considered 'small'.
    """
    small_inds = torch.abs(theta) < eps
    large_inds = small_inds == 0
    return small_inds, large_inds


def sin_theta_by_theta(theta: torch.Tensor, eps: float=0.001):
    """Computes :math:`\\frac{sin \\theta}{\\theta}`. 

    Args:
        theta (torch.Tensor): Angle (magnitude of axis-angle vector).
        eps (float): Threshold (exclusive) below which an angle is
            considered 'small'.

    """
    result = torch.zeros_like(theta)
    small_inds, large_inds = get_small_and_large_angle_inds(theta, eps)
    theta_sq = theta[small_inds] ** 2
    result[small_inds] = 1 - theta_sq / 6 * (1 - theta_sq / 20 * (1 - 
        theta_sq / 42))
    result[large_inds] = torch.sin(theta[large_inds]) / theta[large_inds]
    return result


def grad_sin_theta_by_theta(theta: torch.Tensor, eps: float=0.001):
    """Computes :math:`\\frac{\\partial sin \\theta}{\\partial \\theta \\theta}`. 

    Args:
        theta (torch.Tensor): Angle (magnitude of axis-angle vector).
        eps (float): Threshold below which an angle is considered small.

    """
    result = torch.zeros_like(theta)
    s, l = get_small_and_large_angle_inds(theta, eps)
    theta_sq = theta ** 2
    result[s] = -theta[s] / 3 * (1 - theta_sq[s] / 10 * (1 - theta_sq[s] / 
        28 * (1 - theta_sq[s] / 54)))
    result[l] = cos(theta[l]) / theta[l] - sin(theta[l]) / theta_sq[l]
    return result


class SinThetaByTheta_Function(torch.autograd.Function):

    @staticmethod
    def forward(ctx, theta):
        ctx.save_for_backward(theta)
        return sin_theta_by_theta(theta)

    @staticmethod
    def backward(ctx, grad_output):
        theta, = ctx.saved_tensors
        grad_theta = None
        if ctx.needs_input_grad[0]:
            grad_theta = grad_output * grad_sin_theta_by_theta(theta).to(
                grad_output.device)
        return grad_theta


class SinThetaByTheta(torch.nn.Module):

    def __init__(self):
        super(SinThetaByTheta, self).__init__()

    def forward(self, x):
        return SinThetaByTheta_Function.apply(x)


def grad_one_minus_cos_theta_by_theta_sq(theta: torch.Tensor, eps: float=0.001
    ):
    """Computes :math:`\\frac{\\partial}{\\partial \\theta}\\frac{1 - cos \\theta}{\\theta^2}`.

    Args:
        theta (torch.Tensor): Angle (magnitude of axis-angle vector).
        eps (float): Threshold (exclusive) below which an angle is
            considered 'small'.

    """
    result = torch.zeros_like(theta)
    s, l = get_small_and_large_angle_inds(theta, eps)
    theta_sq = theta ** 2
    result[s] = -theta[s] / 12 * (1 - theta_sq[s] / 5 * (1 / 3 - theta_sq[s
        ] / 56 * (1 / 2 - theta_sq[s] / 135)))
    result[l] = sin(theta[l]) / theta_sq[l] - 2 * (1 - cos(theta[l])) / (
        theta_sq[l] * theta[l])
    return result


def one_minus_cos_theta_by_theta_sq(theta: torch.Tensor, eps: float=0.001):
    """Computes :math:`\\frac{1 - cos \\theta}{\\theta^2}`. 

    Args:
        theta (torch.Tensor): Angle (magnitude of axis-angle vector).
        eps (float): Threshold (exclusive) below which an angle is
            considered 'small'.

    """
    result = torch.zeros_like(theta)
    s, l = get_small_and_large_angle_inds(theta, eps)
    theta_sq = theta ** 2
    result[s] = 1 / 2 * (1 - theta_sq[s] / 12 * (1 - theta_sq[s] / 30 * (1 -
        theta_sq[s] / 56)))
    result[l] = (1 - cos(theta[l])) / theta_sq[l]
    return result


class ThetaBySinTheta_Function(torch.autograd.Function):

    @staticmethod
    def forward(ctx, theta):
        ctx.save_for_backward(theta)
        return one_minus_cos_theta_by_theta_sq(theta)

    @staticmethod
    def backward(ctx, grad_output):
        theta, = ctx.saved_tensors
        grad_theta = None
        if ctx.needs_input_grad[0]:
            grad_theta = grad_output * grad_one_minus_cos_theta_by_theta_sq(
                theta).to(grad_output.device)
        return grad_theta


class ThetaBySinTheta(torch.nn.Module):

    def __init__(self):
        super(ThetaBySinTheta, self).__init__()

    def forward(self, x):
        return ThetaBySinTheta_Function.apply(x)


class OneMinusCosThetaByThetaSq_Function(torch.autograd.Function):

    @staticmethod
    def forward(ctx, theta):
        ctx.save_for_backward(theta)
        return one_minus_cos_theta_by_theta_sq(theta)

    @staticmethod
    def backward(ctx, grad_output):
        theta, = ctx.saved_tensors
        grad_theta = None
        if ctx.needs_input_grad[0]:
            grad_theta = grad_output * grad_one_minus_cos_theta_by_theta_sq(
                theta).to(grad_output.device)
        return grad_theta


class OneMinusCosThetaByThetaSq(torch.nn.Module):

    def __init__(self):
        super(OneMinusCosThetaByThetaSq, self).__init__()

    def forward(self, x):
        return OneMinusCosThetaByThetaSq_Function.apply(x)


def grad_theta_minus_sin_theta_by_theta_cube(theta: torch.Tensor, eps:
    float=0.001):
    """Computes :math:`\\frac{\\partial}{\\partial \\theta}\\frac{\\theta - sin \\theta}{\\theta^3}`.

    Args:
        theta (torch.Tensor): Angle (magnitude of axis-angle vector).
        eps (float): Threshold (exclusive) below which an angle is
            considered 'small'.

    """
    result = torch.zeros_like(theta)
    s, l = get_small_and_large_angle_inds(theta, eps)
    theta_sq = theta[s] ** 2
    result[s] = -theta[s] / 60 * (1 - theta_sq / 21 * (1 - theta_sq / 24 *
        (1 / 2 - theta_sq / 165)))
    result[l] = (3 * sin(theta[l]) - theta[l] * (cos(theta[l]) + 2)) / theta[l
        ] ** 4
    return result


def theta_minus_sin_theta_by_theta_cube(theta: torch.Tensor, eps: float=0.001):
    """Computes :math:`\\frac{\\theta - sin \\theta}{\\theta^3}`. 

    Args:
        theta (torch.Tensor): Angle (magnitude of axis-angle vector).
        eps (float): Threshold (exclusive) below which an angle is
            considered 'small'.

    """
    result = torch.zeros_like(theta)
    s, l = get_small_and_large_angle_inds(theta, eps)
    theta_sq = theta[s] ** 2
    result[s] = 1 / 6 * (1 - theta_sq / 20 * (1 - theta_sq / 42 * (1 - 
        theta_sq / 72)))
    result[l] = (theta[l] - sin(theta[l])) / theta[l] ** 3
    return result


class ThetaMinusSinThetaByThetaCube_Function(torch.autograd.Function):

    @staticmethod
    def forward(ctx, theta):
        ctx.save_for_backward(theta)
        return theta_minus_sin_theta_by_theta_cube(theta)

    @staticmethod
    def backward(ctx, grad_output):
        theta, = ctx.saved_tensors
        grad_theta = None
        if ctx.needs_input_grad[0]:
            grad_theta = (grad_output *
                grad_theta_minus_sin_theta_by_theta_cube(theta).to(
                grad_output.device))
        return grad_theta


class ThetaMinusSinThetaByThetaCube(torch.nn.Module):

    def __init__(self):
        super(ThetaMinusSinThetaByThetaCube, self).__init__()

    def forward(self, x):
        return ThetaMinusSinThetaByThetaCube_Function.apply(x)


class VeryTinyNeRFModel(torch.nn.Module):
    """Define a "very tiny" NeRF model comprising three fully connected layers.
    """

    def __init__(self, filter_size=128, num_encoding_functions=6,
        use_viewdirs=True):
        super(VeryTinyNeRFModel, self).__init__()
        self.num_encoding_functions = num_encoding_functions
        self.xyz_encoding_dims = 3 + 3 * 2 * num_encoding_functions
        if use_viewdirs is True:
            self.viewdir_encoding_dims = 3 + 3 * 2 * num_encoding_functions
        else:
            self.viewdir_encoding_dims = 0
        self.layer1 = torch.nn.Linear(self.xyz_encoding_dims + self.
            viewdir_encoding_dims, filter_size)
        self.layer2 = torch.nn.Linear(filter_size, filter_size)
        self.layer3 = torch.nn.Linear(filter_size, 4)
        self.relu = torch.nn.functional.relu

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x


class MultiHeadNeRFModel(torch.nn.Module):
    """Define a "multi-head" NeRF model (radiance and RGB colors are predicted by
    separate heads).
    """

    def __init__(self, hidden_size=128, num_encoding_functions=6,
        use_viewdirs=True):
        super(MultiHeadNeRFModel, self).__init__()
        self.num_encoding_functions = num_encoding_functions
        self.xyz_encoding_dims = 3 + 3 * 2 * num_encoding_functions
        if use_viewdirs is True:
            self.viewdir_encoding_dims = 3 + 3 * 2 * num_encoding_functions
        else:
            self.viewdir_encoding_dims = 0
        self.layer1 = torch.nn.Linear(self.xyz_encoding_dims, hidden_size)
        self.layer2 = torch.nn.Linear(hidden_size, hidden_size)
        self.layer3_1 = torch.nn.Linear(hidden_size, 1)
        self.layer3_2 = torch.nn.Linear(hidden_size, hidden_size)
        self.layer4 = torch.nn.Linear(self.viewdir_encoding_dims +
            hidden_size, hidden_size)
        self.layer5 = torch.nn.Linear(hidden_size, hidden_size)
        self.layer6 = torch.nn.Linear(hidden_size, 3)
        self.relu = torch.nn.functional.relu

    def forward(self, x):
        x, view = x[(...), :self.xyz_encoding_dims], x[(...), self.
            xyz_encoding_dims:]
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        sigma = self.layer3_1(x)
        feat = self.relu(self.layer3_2(x))
        x = torch.cat((feat, view), dim=-1)
        x = self.relu(self.layer4(x))
        x = self.relu(self.layer5(x))
        x = self.layer6(x)
        return torch.cat((x, sigma), dim=-1)


class ReplicateNeRFModel(torch.nn.Module):
    """NeRF model that follows the figure (from the supp. material of NeRF) to
    every last detail. (ofc, with some flexibility)
    """

    def __init__(self, hidden_size=256, num_layers=4, num_encoding_fn_xyz=6,
        num_encoding_fn_dir=4, include_input_xyz=True, include_input_dir=True):
        super(ReplicateNeRFModel, self).__init__()
        self.dim_xyz = (3 if include_input_xyz else 0
            ) + 2 * 3 * num_encoding_fn_xyz
        self.dim_dir = (3 if include_input_dir else 0
            ) + 2 * 3 * num_encoding_fn_dir
        self.layer1 = torch.nn.Linear(self.dim_xyz, hidden_size)
        self.layer2 = torch.nn.Linear(hidden_size, hidden_size)
        self.layer3 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc_alpha = torch.nn.Linear(hidden_size, 1)
        self.layer4 = torch.nn.Linear(hidden_size + self.dim_dir, 
            hidden_size // 2)
        self.layer5 = torch.nn.Linear(hidden_size // 2, hidden_size // 2)
        self.fc_rgb = torch.nn.Linear(hidden_size // 2, 3)
        self.relu = torch.nn.functional.relu

    def forward(self, x):
        xyz, direction = x[(...), :self.dim_xyz], x[(...), self.dim_xyz:]
        x_ = self.relu(self.layer1(xyz))
        x_ = self.relu(self.layer2(x_))
        feat = self.layer3(x_)
        alpha = self.fc_alpha(x_)
        y_ = self.relu(self.layer4(torch.cat((feat, direction), dim=-1)))
        y_ = self.relu(self.layer5(y_))
        rgb = self.fc_rgb(y_)
        return torch.cat((rgb, alpha), dim=-1)


class PaperNeRFModel(torch.nn.Module):
    """Implements the NeRF model as described in Fig. 7 (appendix) of the
    arXiv submission (v0). """

    def __init__(self, num_layers=8, hidden_size=256, skip_connect_every=4,
        num_encoding_fn_xyz=6, num_encoding_fn_dir=4, include_input_xyz=
        True, include_input_dir=True, use_viewdirs=True):
        super(PaperNeRFModel, self).__init__()
        include_input_xyz = 3 if include_input_xyz else 0
        include_input_dir = 3 if include_input_dir else 0
        self.dim_xyz = include_input_xyz + 2 * 3 * num_encoding_fn_xyz
        self.dim_dir = include_input_dir + 2 * 3 * num_encoding_fn_dir
        self.layers_xyz = torch.nn.ModuleList()
        self.use_viewdirs = use_viewdirs
        self.layers_xyz.append(torch.nn.Linear(self.dim_xyz, 256))
        for i in range(1, 8):
            if i == 4:
                self.layers_xyz.append(torch.nn.Linear(self.dim_xyz + 256, 256)
                    )
            else:
                self.layers_xyz.append(torch.nn.Linear(256, 256))
        self.fc_feat = torch.nn.Linear(256, 256)
        self.fc_alpha = torch.nn.Linear(256, 1)
        self.layers_dir = torch.nn.ModuleList()
        self.layers_dir.append(torch.nn.Linear(256 + self.dim_dir, 128))
        for i in range(3):
            self.layers_dir.append(torch.nn.Linear(128, 128))
        self.fc_rgb = torch.nn.Linear(128, 3)
        self.relu = torch.nn.functional.relu

    def forward(self, x):
        xyz, dirs = x[(...), :self.dim_xyz], x[(...), self.dim_xyz:]
        for i in range(8):
            if i == 4:
                x = self.layers_xyz[i](torch.cat((xyz, x), -1))
            else:
                x = self.layers_xyz[i](x)
            x = self.relu(x)
        feat = self.fc_feat(x)
        alpha = self.fc_alpha(feat)
        if self.use_viewdirs:
            x = self.layers_dir[0](torch.cat((feat, dirs), -1))
        else:
            x = self.layers_dir[0](feat)
        x = self.relu(x)
        for i in range(1, 3):
            x = self.layers_dir[i](x)
            x = self.relu(x)
        rgb = self.fc_rgb(x)
        return torch.cat((rgb, alpha), dim=-1)


class FlexibleNeRFModel(torch.nn.Module):

    def __init__(self, num_layers=4, hidden_size=128, skip_connect_every=4,
        num_encoding_fn_xyz=6, num_encoding_fn_dir=4, include_input_xyz=
        True, include_input_dir=True, use_viewdirs=True):
        super(FlexibleNeRFModel, self).__init__()
        include_input_xyz = 3 if include_input_xyz else 0
        include_input_dir = 3 if include_input_dir else 0
        self.dim_xyz = include_input_xyz + 2 * 3 * num_encoding_fn_xyz
        self.dim_dir = include_input_dir + 2 * 3 * num_encoding_fn_dir
        self.skip_connect_every = skip_connect_every
        if not use_viewdirs:
            self.dim_dir = 0
        self.layer1 = torch.nn.Linear(self.dim_xyz, hidden_size)
        self.layers_xyz = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            if (i % self.skip_connect_every == 0 and i > 0 and i != 
                num_layers - 1):
                self.layers_xyz.append(torch.nn.Linear(self.dim_xyz +
                    hidden_size, hidden_size))
            else:
                self.layers_xyz.append(torch.nn.Linear(hidden_size,
                    hidden_size))
        self.use_viewdirs = use_viewdirs
        if self.use_viewdirs:
            self.layers_dir = torch.nn.ModuleList()
            self.layers_dir.append(torch.nn.Linear(self.dim_dir +
                hidden_size, hidden_size // 2))
            self.fc_alpha = torch.nn.Linear(hidden_size, 1)
            self.fc_rgb = torch.nn.Linear(hidden_size // 2, 3)
            self.fc_feat = torch.nn.Linear(hidden_size, hidden_size)
        else:
            self.fc_out = torch.nn.Linear(hidden_size, 4)
        self.relu = torch.nn.functional.relu

    def forward(self, x):
        if self.use_viewdirs:
            xyz, view = x[(...), :self.dim_xyz], x[(...), self.dim_xyz:]
        else:
            xyz = x[(...), :self.dim_xyz]
        x = self.layer1(xyz)
        for i in range(len(self.layers_xyz)):
            if i % self.skip_connect_every == 0 and i > 0 and i != len(self
                .linear_layers) - 1:
                x = torch.cat((x, xyz), dim=-1)
            x = self.relu(self.layers_xyz[i](x))
        if self.use_viewdirs:
            feat = self.relu(self.fc_feat(x))
            alpha = self.fc_alpha(x)
            x = torch.cat((feat, view), dim=-1)
            for l in self.layers_dir:
                x = self.relu(l(x))
            rgb = self.fc_rgb(x)
            return torch.cat((rgb, alpha), dim=-1)
        else:
            return self.fc_out(x)


class VeryTinyNerfModel(torch.nn.Module):
    """Define a "very tiny" NeRF model comprising three fully connected layers.
    """

    def __init__(self, filter_size=128, num_encoding_functions=6):
        super(VeryTinyNerfModel, self).__init__()
        self.layer1 = torch.nn.Linear(3 + 3 * 2 * num_encoding_functions,
            filter_size)
        self.layer2 = torch.nn.Linear(filter_size, filter_size)
        self.layer3 = torch.nn.Linear(filter_size, 4)
        self.relu = torch.nn.functional.relu

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_krrish94_nerf_pytorch(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(OneMinusCosThetaByThetaSq(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(SinThetaByTheta(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_002(self):
        self._check(ThetaBySinTheta(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_003(self):
        self._check(ThetaMinusSinThetaByThetaCube(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_004(self):
        self._check(VeryTinyNeRFModel(*[], **{}), [torch.rand([78, 78])], {})

    def test_005(self):
        self._check(VeryTinyNerfModel(*[], **{}), [torch.rand([39, 39])], {})

