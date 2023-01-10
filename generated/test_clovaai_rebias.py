import sys
_module = sys.modules[__name__]
del sys
criterions = _module
comparison_methods = _module
dist = _module
hsic = _module
sigma_utils = _module
datasets = _module
colour_mnist = _module
imagenet = _module
kinetics = _module
kinetics_tools = _module
decoder = _module
kinetics = _module
loader = _module
meters = _module
transform = _module
video_container = _module
download = _module
evaluator = _module
logger = _module
main_action = _module
main_biased_mnist = _module
main_imagenet = _module
make_clusters = _module
models = _module
ResNet3D = _module
action_models = _module
head_helper = _module
nonlocal_helper = _module
resnet_helper = _module
stem_helper = _module
weight_init_helper = _module
imagenet_models = _module
mnist_models = _module
rebias_models = _module
optims = _module
trainer = _module

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


import torch.nn.functional as F


import numpy as np


from torch.utils import data


from torchvision import transforms


from torchvision.datasets import MNIST


import torch.utils.data


import math


import random


from torch.utils.data.distributed import DistributedSampler


from torch.utils.data.sampler import RandomSampler


from collections import deque


import time


import torchvision


from torchvision.utils import save_image


from sklearn.cluster import MiniBatchKMeans


from torch.utils.model_zoo import load_url as load_state_dict_from_url


from torch.optim import Adam


from torch.optim.lr_scheduler import StepLR


from torch.optim.lr_scheduler import CosineAnnealingLR


import itertools


class GradMulConst(torch.autograd.Function):
    """ This layer is used to create an adversarial loss.
    """

    @staticmethod
    def forward(ctx, x, const):
        ctx.const = const
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.const, None


def grad_mul_const(x, const):
    return GradMulConst.apply(x, const)


class RUBi(nn.Module):
    """RUBi
    Cadene, Remi, et al. "RUBi: Reducing Unimodal Biases for Visual Question Answering.",
    Advances in Neural Information Processing Systems. 2019.
    """

    def __init__(self, question_loss_weight=1.0, **kwargs):
        super(RUBi, self).__init__()
        self.question_loss_weight = question_loss_weight
        self.fc = nn.Linear(kwargs.get('feat_dim', 128), kwargs.get('num_classes', 10))

    def forward(self, f_feat, g_feat, labels, f_pred, **kwargs):
        """Compute RUBi loss.

        Parameters
        ----------
        f_feat: NOT USED (for compatibility with other losses).
        g_feat: features from biased network (will be passed to `self.fc` for computing `g_pred`)
        labels: class labels
        f_pred: logit values from the target network
        """
        g_feat = g_feat.view(g_feat.shape[0], -1)
        g_feat = grad_mul_const(g_feat, 0.0)
        g_pred = self.fc(g_feat)
        logits_rubi = f_pred * torch.sigmoid(g_pred)
        fusion_loss = F.cross_entropy(logits_rubi, labels)
        question_loss = F.cross_entropy(g_pred, labels)
        loss = fusion_loss + self.question_loss_weight * question_loss
        return loss


class LearnedMixin(nn.Module):
    """LearnedMixin + H
    Clark, Christopher, Mark Yatskar, and Luke Zettlemoyer.
    "Don't Take the Easy Way Out: Ensemble Based Methods for Avoiding Known Dataset Biases.",
    EMNLP 2019.
    """

    def __init__(self, w=0.36, **kwargs):
        """
        :param w: Weight of the entropy penalty
        :param smooth: Add a learned sigmoid(a) factor to the bias to smooth it
        :param smooth_init: How to initialize `a`
        :param constant_smooth: Constant to add to the bias to smooth it
        """
        super(LearnedMixin, self).__init__()
        self.w = w
        self.fc = nn.Linear(kwargs.get('feat_dim', 128), 1)

    def forward(self, f_feat, g_feat, labels, f_pred, g_pred):
        f_feat = f_feat.view(f_feat.shape[0], -1)
        f_pred = f_pred.view(f_pred.shape[0], -1)
        g_pred = g_pred.view(g_pred.shape[0], -1)
        factor = self.fc.forward(f_feat)
        factor = F.softplus(factor)
        g_pred *= factor
        loss = F.cross_entropy(f_pred + g_pred, labels)
        bias_lp = F.log_softmax(g_pred, 1)
        entropy = -(torch.exp(bias_lp) * bias_lp).sum(1).mean()
        return loss + self.w * entropy


class MSELoss(nn.Module):
    """ A simple mean squared error (MSE) implementation.
    """

    def __init__(self, reduction='mean', **kwargs):
        super().__init__()
        self.reduction = reduction

    def forward(self, input, target, **kwargs):
        return F.mse_loss(input, target, reduction=self.reduction)


class L1Loss(nn.Module):
    """ A simple mean absolute error (MAE) implementation.
    """

    def __init__(self, reduction='mean', **kwargs):
        super().__init__()
        self.reduction = reduction

    def forward(self, input, target, **kwargs):
        return F.l1_loss(input, target, reduction=self.reduction)


class HSIC(nn.Module):
    """Base class for the finite sample estimator of Hilbert-Schmidt Independence Criterion (HSIC)
    ..math:: HSIC (X, Y) := || C_{x, y} ||^2_{HS}, where HSIC (X, Y) = 0 iif X and Y are independent.

    Empirically, we use the finite sample estimator of HSIC (with m observations) by,
    (1) biased estimator (HSIC_0)
        Gretton, Arthur, et al. "Measuring statistical dependence with Hilbert-Schmidt norms." 2005.
        :math: (m - 1)^2 tr KHLH.
        where K_{ij} = kernel_x (x_i, x_j), L_{ij} = kernel_y (y_i, y_j), H = 1 - m^{-1} 1 1 (Hence, K, L, H are m by m matrices).
    (2) unbiased estimator (HSIC_1)
        Song, Le, et al. "Feature selection via dependence maximization." 2012.
        :math: rac{1}{m (m - 3)} igg[ tr (	ilde K 	ilde L) + rac{1^	op 	ilde K 1 1^	op 	ilde L 1}{(m-1)(m-2)} - rac{2}{m-2} 1^	op 	ilde K 	ilde L 1 igg].
        where 	ilde K and 	ilde L are related to K and L by the diagonal entries of 	ilde K_{ij} and 	ilde L_{ij} are set to zero.

    Parameters
    ----------
    sigma_x : float
        the kernel size of the kernel function for X.
    sigma_y : float
        the kernel size of the kernel function for Y.
    algorithm: str ('unbiased' / 'biased')
        the algorithm for the finite sample estimator. 'unbiased' is used for our paper.
    reduction: not used (for compatibility with other losses).
    """

    def __init__(self, sigma_x, sigma_y=None, algorithm='unbiased', reduction=None):
        super(HSIC, self).__init__()
        if sigma_y is None:
            sigma_y = sigma_x
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        if algorithm == 'biased':
            self.estimator = self.biased_estimator
        elif algorithm == 'unbiased':
            self.estimator = self.unbiased_estimator
        else:
            raise ValueError('invalid estimator: {}'.format(algorithm))

    def _kernel_x(self, X):
        raise NotImplementedError

    def _kernel_y(self, Y):
        raise NotImplementedError

    def biased_estimator(self, input1, input2):
        """Biased estimator of Hilbert-Schmidt Independence Criterion
        Gretton, Arthur, et al. "Measuring statistical dependence with Hilbert-Schmidt norms." 2005.
        """
        K = self._kernel_x(input1)
        L = self._kernel_y(input2)
        KH = K - K.mean(0, keepdim=True)
        LH = L - L.mean(0, keepdim=True)
        N = len(input1)
        return torch.trace(KH @ LH / (N - 1) ** 2)

    def unbiased_estimator(self, input1, input2):
        """Unbiased estimator of Hilbert-Schmidt Independence Criterion
        Song, Le, et al. "Feature selection via dependence maximization." 2012.
        """
        kernel_XX = self._kernel_x(input1)
        kernel_YY = self._kernel_y(input2)
        tK = kernel_XX - torch.diag(kernel_XX)
        tL = kernel_YY - torch.diag(kernel_YY)
        N = len(input1)
        hsic = torch.trace(tK @ tL) + torch.sum(tK) * torch.sum(tL) / (N - 1) / (N - 2) - 2 * torch.sum(tK, 0).dot(torch.sum(tL, 0)) / (N - 2)
        return hsic / (N * (N - 3))

    def forward(self, input1, input2, **kwargs):
        return self.estimator(input1, input2)


class RbfHSIC(HSIC):
    """Radial Basis Function (RBF) kernel HSIC implementation.
    """

    def _kernel(self, X, sigma):
        X = X.view(len(X), -1)
        XX = X @ X.t()
        X_sqnorms = torch.diag(XX)
        X_L2 = -2 * XX + X_sqnorms.unsqueeze(1) + X_sqnorms.unsqueeze(0)
        gamma = 1 / (2 * sigma ** 2)
        kernel_XX = torch.exp(-gamma * X_L2)
        return kernel_XX

    def _kernel_x(self, X):
        return self._kernel(X, self.sigma_x)

    def _kernel_y(self, Y):
        return self._kernel(Y, self.sigma_y)


class MinusRbfHSIC(RbfHSIC):
    """``Minus'' RbfHSIC for the ``max'' optimization.
    """

    def forward(self, input1, input2, **kwargs):
        return -self.estimator(input1, input2)


DATA_CROP_SIZE = 224


DATA_NUM_FRAMES = 8


FC_INIT_STD = 0.01


NONLOCAL_GROUP = [[1], [1], [1], [1]]


NONLOCAL_INSTANTIATION = 'dot_product'


NONLOCAL_LOCATION = [[[]], [[]], [[]], [[]]]


NUM_BLOCK_TEMP_KERNEL = [[2], [2], [2], [2]]


RESNET_INPLACE_RELU = True


RESNET_STRIDE_1X1 = False


class ResNetBasicHead(nn.Module):
    """
    ResNe(X)t 3D head.
    This layer performs a fully-connected projection during training, when the
    input size is 1x1x1. It performs a convolutional projection during testing
    when the input size is larger than 1x1x1. If the inputs are from multiple
    different pathways, the inputs will be concatenated after pooling.
    """

    def __init__(self, dim_in, num_classes, pool_size, dropout_rate=0.0, feature_position='post', act_func='softmax', final_bottleneck_dim=None):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        ResNetBasicHead takes p pathways as input where p in [1, infty].

        Args:
            dim_in (list): the list of channel dimensions of the p inputs to the
                ResNetHead.
            num_classes (int): the channel dimensions of the p outputs to the
                ResNetHead.
            pool_size (list): the list of kernel sizes of p spatial temporal
                poolings, temporal pool kernel size, spatial pool kernel size,
                spatial pool kernel size in order.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
        """
        super(ResNetBasicHead, self).__init__()
        assert len({len(pool_size), len(dim_in)}) == 1, 'pathway dimensions are not consistent.'
        self.num_pathways = len(pool_size)
        for pathway in range(self.num_pathways):
            avg_pool = nn.AvgPool3d(pool_size[pathway], stride=1)
            self.add_module('pathway{}_avgpool'.format(pathway), avg_pool)
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        if final_bottleneck_dim:
            self.final_bottleneck_dim = final_bottleneck_dim
            self.final_bottleneck = nn.Conv3d(sum(dim_in), final_bottleneck_dim, kernel_size=1, stride=1, padding=0, bias=False)
            self.final_bottleneck_bn = nn.BatchNorm3d(final_bottleneck_dim, eps=1e-05, momentum=0.1)
            self.final_bottleneck_act = nn.ReLU(inplace=True)
            dim_in = final_bottleneck_dim
        else:
            self.final_bottleneck_dim = None
            dim_in = sum(dim_in)
        self.projection = nn.Linear(dim_in, num_classes, bias=True)
        self.feature_position = feature_position
        if act_func == 'softmax':
            self.act = nn.Softmax(dim=4)
        elif act_func == 'sigmoid':
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError('{} is not supported as an activationfunction.'.format(act_func))

    def forward(self, inputs):
        assert len(inputs) == self.num_pathways, 'Input tensor does not contain {} pathway'.format(self.num_pathways)
        pool_out = []
        if self.final_bottleneck_dim:
            for pathway in range(self.num_pathways):
                inputs[pathway] = self.final_bottleneck(inputs[pathway])
                inputs[pathway] = self.final_bottleneck_bn(inputs[pathway])
                inputs[pathway] = self.final_bottleneck_act(inputs[pathway])
        for pathway in range(self.num_pathways):
            m = getattr(self, 'pathway{}_avgpool'.format(pathway))
            pool_out.append(m(inputs[pathway]))
        h = torch.cat(pool_out, 1)
        x = h.permute((0, 2, 3, 4, 1))
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        if self.feature_position == 'final_bottleneck':
            h = x.mean([1, 2, 3])
            h = h.view(h.shape[0], -1)
        x = self.projection(x)
        if self.feature_position == 'logit':
            h = x
        if not self.training:
            x = self.act(x)
            x = x.mean([1, 2, 3])
        x = x.view(x.shape[0], -1)
        return x, h


class Nonlocal(nn.Module):
    """
    Builds Non-local Neural Networks as a generic family of building
    blocks for capturing long-range dependencies. Non-local Network
    computes the response at a position as a weighted sum of the
    features at all positions. This building block can be plugged into
    many computer vision architectures.
    More details in the paper: https://arxiv.org/pdf/1711.07971.pdf
    """

    def __init__(self, dim, dim_inner, pool_size=None, instantiation='softmax', norm_type='batchnorm', zero_init_final_conv=False, zero_init_final_norm=True, norm_eps=1e-05, norm_momentum=0.1):
        """
        Args:
            dim (int): number of dimension for the input.
            dim_inner (int): number of dimension inside of the Non-local block.
            pool_size (list): the kernel size of spatial temporal pooling,
                temporal pool kernel size, spatial pool kernel size, spatial
                pool kernel size in order. By default pool_size is None,
                then there would be no pooling used.
            instantiation (string): supports two different instantiation method:
                "dot_product": normalizing correlation matrix with L2.
                "softmax": normalizing correlation matrix with Softmax.
            norm_type (string): support BatchNorm and LayerNorm for
                normalization.
                "batchnorm": using BatchNorm for normalization.
                "layernorm": using LayerNorm for normalization.
                "none": not using any normalization.
            zero_init_final_conv (bool): If true, zero initializing the final
                convolution of the Non-local block.
            zero_init_final_norm (bool):
                If true, zero initializing the final batch norm of the Non-local
                block.
        """
        super(Nonlocal, self).__init__()
        self.dim = dim
        self.dim_inner = dim_inner
        self.pool_size = pool_size
        self.instantiation = instantiation
        self.norm_type = norm_type
        self.use_pool = False if pool_size is None else any(size > 1 for size in pool_size)
        self.norm_eps = norm_eps
        self.norm_momentum = norm_momentum
        self._construct_nonlocal(zero_init_final_conv, zero_init_final_norm)

    def _construct_nonlocal(self, zero_init_final_conv, zero_init_final_norm):
        self.conv_theta = nn.Conv3d(self.dim, self.dim_inner, kernel_size=1, stride=1, padding=0)
        self.conv_phi = nn.Conv3d(self.dim, self.dim_inner, kernel_size=1, stride=1, padding=0)
        self.conv_g = nn.Conv3d(self.dim, self.dim_inner, kernel_size=1, stride=1, padding=0)
        self.conv_out = nn.Conv3d(self.dim_inner, self.dim, kernel_size=1, stride=1, padding=0)
        self.conv_out.zero_init = zero_init_final_conv
        if self.norm_type == 'batchnorm':
            self.bn = nn.BatchNorm3d(self.dim, eps=self.norm_eps, momentum=self.norm_momentum)
            self.bn.transform_final_bn = zero_init_final_norm
        elif self.norm_type == 'layernorm':
            self.ln = nn.GroupNorm(1, self.dim, eps=self.norm_eps, affine=False)
        elif self.norm_type == 'none':
            pass
        else:
            raise NotImplementedError('Norm type {} is not supported'.format(self.norm_type))
        if self.use_pool:
            self.pool = nn.MaxPool3d(kernel_size=self.pool_size, stride=self.pool_size, padding=[0, 0, 0])

    def forward(self, x):
        x_identity = x
        N, C, T, H, W = x.size()
        theta = self.conv_theta(x)
        if self.use_pool:
            x = self.pool(x)
        phi = self.conv_phi(x)
        g = self.conv_g(x)
        theta = theta.view(N, self.dim_inner, -1)
        phi = phi.view(N, self.dim_inner, -1)
        g = g.view(N, self.dim_inner, -1)
        theta_phi = torch.einsum('nct,ncp->ntp', (theta, phi))
        if self.instantiation == 'softmax':
            theta_phi = theta_phi * self.dim_inner ** -0.5
            theta_phi = nn.functional.softmax(theta_phi, dim=2)
        elif self.instantiation == 'dot_product':
            spatial_temporal_dim = theta_phi.shape[2]
            theta_phi = theta_phi / spatial_temporal_dim
        else:
            raise NotImplementedError('Unknown norm type {}'.format(self.instantiation))
        theta_phi_g = torch.einsum('ntg,ncg->nct', (theta_phi, g))
        theta_phi_g = theta_phi_g.view(N, self.dim_inner, T, H, W)
        p = self.conv_out(theta_phi_g)
        if self.norm_type == 'batchnorm':
            p = self.bn(p)
        elif self.norm_type == 'layernorm':
            p = self.ln(p)
        return x_identity + p


class ResBlock(nn.Module):
    """
    Residual block.
    """

    def __init__(self, dim_in, dim_out, temp_kernel_size, stride, trans_func, dim_inner, num_groups=1, stride_1x1=False, inplace_relu=True, eps=1e-05, bn_mmt=0.1):
        """
        ResBlock class constructs redisual blocks. More details can be found in:
            Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
            "Deep residual learning for image recognition."
            https://arxiv.org/abs/1512.03385
        Args:
            dim_in (int): the channel dimensions of the input.
            dim_out (int): the channel dimension of the output.
            temp_kernel_size (int): the temporal kernel sizes of the middle
                convolution in the bottleneck.
            stride (int): the stride of the bottleneck.
            trans_func (string): transform function to be used to construct the
                bottleneck.
            dim_inner (int): the inner dimension of the block.
            num_groups (int): number of groups for the convolution. num_groups=1
                is for standard ResNet like networks, and num_groups>1 is for
                ResNeXt like networks.
            stride_1x1 (bool): if True, apply stride to 1x1 conv, otherwise
                apply stride to the 3x3 conv.
            inplace_relu (bool): calculate the relu on the original input
                without allocating new memory.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
        """
        super(ResBlock, self).__init__()
        self._inplace_relu = inplace_relu
        self._eps = eps
        self._bn_mmt = bn_mmt
        self._construct(dim_in, dim_out, temp_kernel_size, stride, trans_func, dim_inner, num_groups, stride_1x1, inplace_relu)

    def _construct(self, dim_in, dim_out, temp_kernel_size, stride, trans_func, dim_inner, num_groups, stride_1x1, inplace_relu):
        if dim_in != dim_out or stride != 1:
            self.branch1 = nn.Conv3d(dim_in, dim_out, kernel_size=1, stride=[1, stride, stride], padding=0, bias=False)
            self.branch1_bn = nn.BatchNorm3d(dim_out, eps=self._eps, momentum=self._bn_mmt)
        self.branch2 = trans_func(dim_in, dim_out, temp_kernel_size, stride, dim_inner, num_groups, stride_1x1=stride_1x1, inplace_relu=inplace_relu)
        self.relu = nn.ReLU(self._inplace_relu)

    def forward(self, x):
        if hasattr(self, 'branch1'):
            x = self.branch1_bn(self.branch1(x)) + self.branch2(x)
        else:
            x = x + self.branch2(x)
        x = self.relu(x)
        return x


class BasicTransform(nn.Module):
    """
    Basic transformation: Tx3x3, 1x3x3, where T is the size of temporal kernel.
    """

    def __init__(self, dim_in, dim_out, temp_kernel_size, stride, dim_inner=None, num_groups=1, stride_1x1=None, inplace_relu=True, eps=1e-05, bn_mmt=0.1):
        """
        Args:
            dim_in (int): the channel dimensions of the input.
            dim_out (int): the channel dimension of the output.
            temp_kernel_size (int): the temporal kernel sizes of the middle
                convolution in the bottleneck.
            stride (int): the stride of the bottleneck.
            dim_inner (None): the inner dimension would not be used in
                BasicTransform.
            num_groups (int): number of groups for the convolution. Number of
                group is always 1 for BasicTransform.
            stride_1x1 (None): stride_1x1 will not be used in BasicTransform.
            inplace_relu (bool): if True, calculate the relu on the original
                input without allocating new memory.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
        """
        super(BasicTransform, self).__init__()
        self.temp_kernel_size = temp_kernel_size
        self._inplace_relu = inplace_relu
        self._eps = eps
        self._bn_mmt = bn_mmt
        self._construct(dim_in, dim_out, stride)

    def _construct(self, dim_in, dim_out, stride):
        self.a = nn.Conv3d(dim_in, dim_out, kernel_size=[self.temp_kernel_size, 3, 3], stride=[1, stride, stride], padding=[int(self.temp_kernel_size // 2), 1, 1], bias=False)
        self.a_bn = nn.BatchNorm3d(dim_out, eps=self._eps, momentum=self._bn_mmt)
        self.a_relu = nn.ReLU(inplace=self._inplace_relu)
        self.b = nn.Conv3d(dim_out, dim_out, kernel_size=[1, 3, 3], stride=[1, 1, 1], padding=[0, 1, 1], bias=False)
        self.b_bn = nn.BatchNorm3d(dim_out, eps=self._eps, momentum=self._bn_mmt)
        self.b_bn.transform_final_bn = True

    def forward(self, x):
        x = self.a(x)
        x = self.a_bn(x)
        x = self.a_relu(x)
        x = self.b(x)
        x = self.b_bn(x)
        return x


class BottleneckTransform(nn.Module):
    """
    Bottleneck transformation: Tx1x1, 1x3x3, 1x1x1, where T is the size of
        temporal kernel.
    """

    def __init__(self, dim_in, dim_out, temp_kernel_size, stride, dim_inner, num_groups, stride_1x1=False, inplace_relu=True, eps=1e-05, bn_mmt=0.1):
        """
        Args:
            dim_in (int): the channel dimensions of the input.
            dim_out (int): the channel dimension of the output.
            temp_kernel_size (int): the temporal kernel sizes of the middle
                convolution in the bottleneck.
            stride (int): the stride of the bottleneck.
            dim_inner (int): the inner dimension of the block.
            num_groups (int): number of groups for the convolution. num_groups=1
                is for standard ResNet like networks, and num_groups>1 is for
                ResNeXt like networks.
            stride_1x1 (bool): if True, apply stride to 1x1 conv, otherwise
                apply stride to the 3x3 conv.
            inplace_relu (bool): if True, calculate the relu on the original
                input without allocating new memory.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
        """
        super(BottleneckTransform, self).__init__()
        self.temp_kernel_size = temp_kernel_size
        self._inplace_relu = inplace_relu
        self._eps = eps
        self._bn_mmt = bn_mmt
        self._stride_1x1 = stride_1x1
        self._construct(dim_in, dim_out, stride, dim_inner, num_groups)

    def _construct(self, dim_in, dim_out, stride, dim_inner, num_groups):
        str1x1, str3x3 = (stride, 1) if self._stride_1x1 else (1, stride)
        self.a = nn.Conv3d(dim_in, dim_inner, kernel_size=[self.temp_kernel_size, 1, 1], stride=[1, str1x1, str1x1], padding=[int(self.temp_kernel_size // 2), 0, 0], bias=False)
        self.a_bn = nn.BatchNorm3d(dim_inner, eps=self._eps, momentum=self._bn_mmt)
        self.a_relu = nn.ReLU(inplace=self._inplace_relu)
        self.b = nn.Conv3d(dim_inner, dim_inner, [1, 3, 3], stride=[1, str3x3, str3x3], padding=[0, 1, 1], groups=num_groups, bias=False)
        self.b_bn = nn.BatchNorm3d(dim_inner, eps=self._eps, momentum=self._bn_mmt)
        self.b_relu = nn.ReLU(inplace=self._inplace_relu)
        self.c = nn.Conv3d(dim_inner, dim_out, kernel_size=[1, 1, 1], stride=[1, 1, 1], padding=[0, 0, 0], bias=False)
        self.c_bn = nn.BatchNorm3d(dim_out, eps=self._eps, momentum=self._bn_mmt)
        self.c_bn.transform_final_bn = True

    def forward(self, x):
        x = self.a(x)
        x = self.a_bn(x)
        x = self.a_relu(x)
        x = self.b(x)
        x = self.b_bn(x)
        x = self.b_relu(x)
        x = self.c(x)
        x = self.c_bn(x)
        return x


def get_trans_func(name):
    """
    Retrieves the transformation module by name.
    """
    trans_funcs = {'bottleneck_transform': BottleneckTransform, 'basic_transform': BasicTransform}
    assert name in trans_funcs.keys(), "Transformation function '{}' not supported".format(name)
    return trans_funcs[name]


class ResStage(nn.Module):
    """
    Stage of 3D ResNet. It expects to have one or more tensors as input for
        single pathway (C2D, I3D, SlowOnly), and multi-pathway (SlowFast) cases.
        More details can be found here:

        Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
        "Slowfast networks for video recognition."
        https://arxiv.org/pdf/1812.03982.pdf
    """

    def __init__(self, dim_in, dim_out, stride, temp_kernel_sizes, num_blocks, dim_inner, num_groups, num_block_temp_kernel, nonlocal_inds, nonlocal_group, instantiation='softmax', trans_func_name='bottleneck_transform', stride_1x1=False, inplace_relu=True):
        """
        The `__init__` method of any subclass should also contain these arguments.
        ResStage builds p streams, where p can be greater or equal to one.
        Args:
            dim_in (list): list of p the channel dimensions of the input.
                Different channel dimensions control the input dimension of
                different pathways.
            dim_out (list): list of p the channel dimensions of the output.
                Different channel dimensions control the input dimension of
                different pathways.
            temp_kernel_sizes (list): list of the p temporal kernel sizes of the
                convolution in the bottleneck. Different temp_kernel_sizes
                control different pathway.
            stride (list): list of the p strides of the bottleneck. Different
                stride control different pathway.
            num_blocks (list): list of p numbers of blocks for each of the
                pathway.
            dim_inner (list): list of the p inner channel dimensions of the
                input. Different channel dimensions control the input dimension
                of different pathways.
            num_groups (list): list of number of p groups for the convolution.
                num_groups=1 is for standard ResNet like networks, and
                num_groups>1 is for ResNeXt like networks.
            num_block_temp_kernel (list): extent the temp_kernel_sizes to
                num_block_temp_kernel blocks, then fill temporal kernel size
                of 1 for the rest of the layers.
            nonlocal_inds (list): If the tuple is empty, no nonlocal layer will
                be added. If the tuple is not empty, add nonlocal layers after
                the index-th block.
            nonlocal_group (list): list of number of p nonlocal groups. Each
                number controls how to fold temporal dimension to batch
                dimension before applying nonlocal transformation.
                https://github.com/facebookresearch/video-nonlocal-net.
            instantiation (string): different instantiation for nonlocal layer.
                Supports two different instantiation method:
                    "dot_product": normalizing correlation matrix with L2.
                    "softmax": normalizing correlation matrix with Softmax.
            trans_func_name (string): name of the the transformation function apply
                on the network.
        """
        super(ResStage, self).__init__()
        assert all(num_block_temp_kernel[i] <= num_blocks[i] for i in range(len(temp_kernel_sizes)))
        self.num_blocks = num_blocks
        self.nonlocal_group = nonlocal_group
        self.temp_kernel_sizes = [((temp_kernel_sizes[i] * num_blocks[i])[:num_block_temp_kernel[i]] + [1] * (num_blocks[i] - num_block_temp_kernel[i])) for i in range(len(temp_kernel_sizes))]
        assert len({len(dim_in), len(dim_out), len(temp_kernel_sizes), len(stride), len(num_blocks), len(dim_inner), len(num_groups), len(num_block_temp_kernel), len(nonlocal_inds), len(nonlocal_group)}) == 1
        self.num_pathways = len(self.num_blocks)
        self._construct(dim_in, dim_out, stride, dim_inner, num_groups, trans_func_name, stride_1x1, inplace_relu, nonlocal_inds, instantiation)

    def _construct(self, dim_in, dim_out, stride, dim_inner, num_groups, trans_func_name, stride_1x1, inplace_relu, nonlocal_inds, instantiation):
        for pathway in range(self.num_pathways):
            for i in range(self.num_blocks[pathway]):
                trans_func = get_trans_func(trans_func_name)
                res_block = ResBlock(dim_in[pathway] if i == 0 else dim_out[pathway], dim_out[pathway], self.temp_kernel_sizes[pathway][i], stride[pathway] if i == 0 else 1, trans_func, dim_inner[pathway], num_groups[pathway], stride_1x1=stride_1x1, inplace_relu=inplace_relu)
                self.add_module('pathway{}_res{}'.format(pathway, i), res_block)
                if i in nonlocal_inds[pathway]:
                    nln = Nonlocal(dim_out[pathway], dim_out[pathway] // 2, [1, 2, 2], instantiation=instantiation)
                    self.add_module('pathway{}_nonlocal{}'.format(pathway, i), nln)

    def forward(self, inputs):
        output = []
        for pathway in range(self.num_pathways):
            x = inputs[pathway]
            for i in range(self.num_blocks[pathway]):
                m = getattr(self, 'pathway{}_res{}'.format(pathway, i))
                x = m(x)
                if hasattr(self, 'pathway{}_nonlocal{}'.format(pathway, i)):
                    nln = getattr(self, 'pathway{}_nonlocal{}'.format(pathway, i))
                    b, c, t, h, w = x.shape
                    if self.nonlocal_group[pathway] > 1:
                        x = x.permute(0, 2, 1, 3, 4)
                        x = x.reshape(b * self.nonlocal_group[pathway], t // self.nonlocal_group[pathway], c, h, w)
                        x = x.permute(0, 2, 1, 3, 4)
                    x = nln(x)
                    if self.nonlocal_group[pathway] > 1:
                        x = x.permute(0, 2, 1, 3, 4)
                        x = x.reshape(b, t, c, h, w)
                        x = x.permute(0, 2, 1, 3, 4)
            output.append(x)
        return output


class ResNetBasicStem(nn.Module):
    """
    ResNe(X)t 3D stem module.
    Performs spatiotemporal Convolution, BN, and Relu following by a
        spatiotemporal pooling.
    """

    def __init__(self, dim_in, dim_out, kernel, stride, padding, inplace_relu=True, eps=1e-05, bn_mmt=0.1):
        """
        The `__init__` method of any subclass should also contain these arguments.

        Args:
            dim_in (int): the channel dimension of the input. Normally 3 is used
                for rgb input, and 2 or 3 is used for optical flow input.
            dim_out (int): the output dimension of the convolution in the stem
                layer.
            kernel (list): the kernel size of the convolution in the stem layer.
                temporal kernel size, height kernel size, width kernel size in
                order.
            stride (list): the stride size of the convolution in the stem layer.
                temporal kernel stride, height kernel size, width kernel size in
                order.
            padding (int): the padding size of the convolution in the stem
                layer, temporal padding size, height padding size, width
                padding size in order.
            inplace_relu (bool): calculate the relu on the original input
                without allocating new memory.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
        """
        super(ResNetBasicStem, self).__init__()
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.inplace_relu = inplace_relu
        self.eps = eps
        self.bn_mmt = bn_mmt
        self._construct_stem(dim_in, dim_out)

    def _construct_stem(self, dim_in, dim_out):
        self.conv = nn.Conv3d(dim_in, dim_out, self.kernel, stride=self.stride, padding=self.padding, bias=False)
        self.bn = nn.BatchNorm3d(dim_out, eps=self.eps, momentum=self.bn_mmt)
        self.relu = nn.ReLU(self.inplace_relu)
        self.pool_layer = nn.MaxPool3d(kernel_size=[1, 3, 3], stride=[1, 2, 2], padding=[0, 1, 1])

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool_layer(x)
        return x


class VideoModelStem(nn.Module):
    """
    Video 3D stem module. Provides stem operations of Conv, BN, ReLU, MaxPool
    on input data tensor for one or multiple pathways.
    """

    def __init__(self, dim_in, dim_out, kernel, stride, padding, inplace_relu=True, eps=1e-05, bn_mmt=0.1):
        """
        The `__init__` method of any subclass should also contain these
        arguments. List size of 1 for single pathway models (C2D, I3D, SlowOnly
        and etc), list size of 2 for two pathway models (SlowFast).

        Args:
            dim_in (list): the list of channel dimensions of the inputs.
            dim_out (list): the output dimension of the convolution in the stem
                layer.
            kernel (list): the kernels' size of the convolutions in the stem
                layers. Temporal kernel size, height kernel size, width kernel
                size in order.
            stride (list): the stride sizes of the convolutions in the stem
                layer. Temporal kernel stride, height kernel size, width kernel
                size in order.
            padding (list): the paddings' sizes of the convolutions in the stem
                layer. Temporal padding size, height padding size, width padding
                size in order.
            inplace_relu (bool): calculate the relu on the original input
                without allocating new memory.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
        """
        super(VideoModelStem, self).__init__()
        assert len({len(dim_in), len(dim_out), len(kernel), len(stride), len(padding)}) == 1, 'Input pathway dimensions are not consistent.'
        self.num_pathways = len(dim_in)
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.inplace_relu = inplace_relu
        self.eps = eps
        self.bn_mmt = bn_mmt
        self._construct_stem(dim_in, dim_out)

    def _construct_stem(self, dim_in, dim_out):
        for pathway in range(len(dim_in)):
            stem = ResNetBasicStem(dim_in[pathway], dim_out[pathway], self.kernel[pathway], self.stride[pathway], self.padding[pathway], self.inplace_relu, self.eps, self.bn_mmt)
            self.add_module('pathway{}_stem'.format(pathway), stem)

    def forward(self, x):
        assert len(x) == self.num_pathways, 'Input tensor does not contain {} pathway'.format(self.num_pathways)
        for pathway in range(len(x)):
            m = getattr(self, 'pathway{}_stem'.format(pathway))
            x[pathway] = m(x[pathway])
        return x


ZERO_INIT_FINAL_BN = False


_MODEL_STAGE_DEPTH = {(18.1): (2, 2, 2, 2), (18): (2, 2, 2, 2), (34.1): (3, 4, 6, 3), (50): (3, 4, 6, 3), (101): (3, 4, 23, 3)}


_MODEL_TRANS_FUNC = {(18.1): 'basic_transform', (18): 'basic_transform', (34.1): 'basic_transform', (50): 'bottleneck_transform', (101): 'bottleneck_transform'}


_POOL1 = [[1, 1, 1]]


_TEMPORAL_KERNEL_BASIS = {'11111': [[[1]], [[1]], [[1]], [[1]], [[1]]], '33333': [[[3]], [[3]], [[3]], [[3]], [[3]]], '11133': [[[1]], [[1]], [[1]], [[3]], [[3]]]}


def init_weights(model, fc_init_std=0.01, zero_init_final_bn=True):
    """
    Performs ResNet style weight initialization.
    Args:
        fc_init_std (float): the expected standard deviation for fc layer.
        zero_init_final_bn (bool): if True, zero initialize the final bn for
            every bottleneck.
    """
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            """
            Follow the initialization method proposed in:
            {He, Kaiming, et al.
            "Delving deep into rectifiers: Surpassing human-level
            performance on imagenet classification."
            arXiv preprint arXiv:1502.01852 (2015)}
            """
            c2_msra_fill(m)
        elif isinstance(m, nn.BatchNorm3d):
            if hasattr(m, 'transform_final_bn') and m.transform_final_bn and zero_init_final_bn:
                batchnorm_weight = 0.0
            else:
                batchnorm_weight = 1.0
            m.weight.data.fill_(batchnorm_weight)
            m.bias.data.zero_()
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(mean=0.0, std=fc_init_std)
            m.bias.data.zero_()


width_multiplier = {(18.1): [1, 1, 2, 4, 8], (34.1): [1, 1, 2, 4, 8], (18): [1, 4, 8, 16, 32], (50): [1, 4, 8, 16, 32]}


class ResNet3DModel(nn.Module):
    """
    ResNet model builder. It builds a ResNet like network backbone without
    lateral connection (C2D, I3D, SlowOnly).

    Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
    "Slowfast networks for video recognition."
    https://arxiv.org/pdf/1812.03982.pdf

    Xiaolong Wang, Ross Girshick, Abhinav Gupta, and Kaiming He.
    "Non-local neural networks."
    https://arxiv.org/pdf/1711.07971.pdf
    """

    def __init__(self, model_arch='33333', resnet_depth=18, feature_position='post', width_per_group=32, dropout_rate=0.0, num_classes=400, final_bottleneck_dim=0):
        """
        The `__init__` method of any subclass should also contain these
            arguments.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(ResNet3DModel, self).__init__()
        self.num_pathways = 1
        self._construct_network(model_arch=model_arch, resnet_depth=resnet_depth, dropout_rate=dropout_rate, width_per_group=width_per_group, num_classes=num_classes, feature_position=feature_position, final_bottleneck_dim=final_bottleneck_dim)
        init_weights(self, FC_INIT_STD, ZERO_INIT_FINAL_BN)

    def _construct_network(self, model_arch='33333', resnet_depth=18, feature_position='post', num_groups=1, width_per_group=32, input_channel_num=None, dropout_rate=0.0, num_classes=400, final_bottleneck_dim=0):
        """
        Builds a single pathway ResNet model.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        if input_channel_num is None:
            input_channel_num = [3]
        pool_size = _POOL1
        assert len({len(pool_size), self.num_pathways}) == 1
        assert resnet_depth in _MODEL_STAGE_DEPTH.keys()
        d2, d3, d4, d5 = _MODEL_STAGE_DEPTH[resnet_depth]
        trans_func = _MODEL_TRANS_FUNC[resnet_depth]
        dim_inner = num_groups * width_per_group
        temp_kernel = _TEMPORAL_KERNEL_BASIS[str(model_arch)]
        self.s1 = VideoModelStem(dim_in=input_channel_num, dim_out=[width_per_group * width_multiplier[resnet_depth][0]], kernel=[temp_kernel[0][0] + [7, 7]], stride=[[1, 2, 2]], padding=[[temp_kernel[0][0][0] // 2, 3, 3]])
        self.s2 = ResStage(dim_in=[width_per_group * width_multiplier[resnet_depth][0]], dim_out=[width_per_group * width_multiplier[resnet_depth][1]], dim_inner=[dim_inner], temp_kernel_sizes=temp_kernel[1], stride=[1], num_blocks=[d2], num_groups=[num_groups], num_block_temp_kernel=NUM_BLOCK_TEMP_KERNEL[0], nonlocal_inds=NONLOCAL_LOCATION[0], nonlocal_group=NONLOCAL_GROUP[0], instantiation=NONLOCAL_INSTANTIATION, trans_func_name=trans_func, stride_1x1=RESNET_STRIDE_1X1, inplace_relu=RESNET_INPLACE_RELU)
        for pathway in range(self.num_pathways):
            pool = nn.MaxPool3d(kernel_size=pool_size[pathway], stride=pool_size[pathway], padding=[0, 0, 0])
            self.add_module('pathway{}_pool'.format(pathway), pool)
        self.s3 = ResStage(dim_in=[width_per_group * width_multiplier[resnet_depth][1]], dim_out=[width_per_group * width_multiplier[resnet_depth][2]], dim_inner=[dim_inner * 2], temp_kernel_sizes=temp_kernel[2], stride=[2], num_blocks=[d3], num_groups=[num_groups], num_block_temp_kernel=NUM_BLOCK_TEMP_KERNEL[1], nonlocal_inds=NONLOCAL_LOCATION[1], nonlocal_group=NONLOCAL_GROUP[1], instantiation=NONLOCAL_INSTANTIATION, trans_func_name=trans_func, stride_1x1=RESNET_STRIDE_1X1, inplace_relu=RESNET_INPLACE_RELU)
        self.s4 = ResStage(dim_in=[width_per_group * width_multiplier[resnet_depth][2]], dim_out=[width_per_group * width_multiplier[resnet_depth][3]], dim_inner=[dim_inner * 4], temp_kernel_sizes=temp_kernel[3], stride=[2], num_blocks=[d4], num_groups=[num_groups], num_block_temp_kernel=NUM_BLOCK_TEMP_KERNEL[2], nonlocal_inds=NONLOCAL_LOCATION[2], nonlocal_group=NONLOCAL_GROUP[2], instantiation=NONLOCAL_INSTANTIATION, trans_func_name=trans_func, stride_1x1=RESNET_STRIDE_1X1, inplace_relu=RESNET_INPLACE_RELU)
        self.s5 = ResStage(dim_in=[width_per_group * width_multiplier[resnet_depth][3]], dim_out=[width_per_group * width_multiplier[resnet_depth][4]], dim_inner=[dim_inner * 8], temp_kernel_sizes=temp_kernel[4], stride=[2], num_blocks=[d5], num_groups=[num_groups], num_block_temp_kernel=NUM_BLOCK_TEMP_KERNEL[3], nonlocal_inds=NONLOCAL_LOCATION[3], nonlocal_group=NONLOCAL_GROUP[3], instantiation=NONLOCAL_INSTANTIATION, trans_func_name=trans_func, stride_1x1=RESNET_STRIDE_1X1, inplace_relu=RESNET_INPLACE_RELU)
        self.head = ResNetBasicHead(dim_in=[width_per_group * width_multiplier[resnet_depth][4]], num_classes=num_classes, pool_size=[[DATA_NUM_FRAMES // pool_size[0][0], DATA_CROP_SIZE // 32 // pool_size[0][1], DATA_CROP_SIZE // 32 // pool_size[0][2]]], dropout_rate=dropout_rate, feature_position=feature_position, final_bottleneck_dim=final_bottleneck_dim)

    def forward(self, x, logits_only=False):
        x = [x]
        x = self.s1(x)
        x = self.s2(x)
        for pathway in range(self.num_pathways):
            pool = getattr(self, 'pathway{}_pool'.format(pathway))
            x[pathway] = pool(x[pathway])
        x = self.s3(x)
        x = self.s4(x)
        x = self.s5(x)
        x, h = self.head(x)
        if logits_only:
            return x
        else:
            return x, h


class BasicBlock_(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, kernel_size=1):
        super(BasicBlock_, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        if identity.size(-1) != out.size(-1):
            diff = identity.size(-1) - out.size(-1)
            identity = identity[:, :, :-diff, :-diff]
        out += identity
        out = self.relu(out)
        return out


class Bottleneck_(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, kernel_size=1):
        super(Bottleneck_, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=stride, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        if residual.size(-1) != out.size(-1):
            diff = residual.size(-1) - out.size(-1)
            residual = residual[:, :, :-diff, :-diff]
        out += residual
        out = self.relu(out)
        return out


class BagNetDeep(nn.Module):

    def __init__(self, block, layers, strides=[2, 2, 2, 1], kernel3=[0, 0, 0, 0], num_classes=1000, feature_pos='post', avg_pool=True):
        super(BagNetDeep, self).__init__()
        self.inplanes = 64
        self.feature_pos = feature_pos
        self.conv1 = nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=0.001)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], kernel3=kernel3[0], prefix='layer1')
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], kernel3=kernel3[1], prefix='layer2')
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], kernel3=kernel3[2], prefix='layer3')
        self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], kernel3=kernel3[3], prefix='layer4')
        self.avgpool = nn.AvgPool2d(1, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.avg_pool = avg_pool
        self.block = block
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, kernel3=0, prefix=''):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        kernel = 1 if kernel3 == 0 else 3
        layers.append(block(self.inplanes, planes, stride, downsample, kernel_size=kernel))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            kernel = 1 if kernel3 <= i else 3
            layers.append(block(self.inplanes, planes, kernel_size=kernel))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x_ = nn.AvgPool2d(x.size()[2], stride=1)(x)
        x = x_.view(x_.size(0), -1)
        x = self.fc(x)
        return x, x_


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError('Dilation > 1 not supported in BasicBlock')
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, feature_pos='post', zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None, rf=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.feature_pos = feature_pos
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError('replace_stride_with_dilation should be None or a 3-element tuple, got {}'.format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride), norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x_ = self.avgpool(x)
        x = torch.flatten(x_, 1)
        x = self.fc(x)
        return x, x_


class SimpleConvNet(nn.Module):

    def __init__(self, num_classes=None, kernel_size=7, feature_pos='post'):
        super(SimpleConvNet, self).__init__()
        padding = kernel_size // 2
        layers = [nn.Conv2d(3, 16, kernel_size=kernel_size, padding=padding), nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.Conv2d(16, 32, kernel_size=kernel_size, padding=padding), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.Conv2d(32, 64, kernel_size=kernel_size, padding=padding), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Conv2d(64, 128, kernel_size=kernel_size, padding=padding), nn.BatchNorm2d(128), nn.ReLU(inplace=True)]
        self.extracter = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 10)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if feature_pos not in ['pre', 'post', 'logits']:
            raise ValueError(feature_pos)
        self.feature_pos = feature_pos

    def forward(self, x, logits_only=False):
        pre_gap_feats = self.extracter(x)
        post_gap_feats = self.avgpool(pre_gap_feats)
        post_gap_feats = torch.flatten(post_gap_feats, 1)
        logits = self.fc(post_gap_feats)
        if logits_only:
            return logits
        elif self.feature_pos == 'pre':
            feats = pre_gap_feats
        elif self.feature_pos == 'post':
            feats = post_gap_feats
        else:
            feats = logits
        return logits, feats


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BasicBlock_,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BasicTransform,
     lambda: ([], {'dim_in': 4, 'dim_out': 4, 'temp_kernel_size': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (BottleneckTransform,
     lambda: ([], {'dim_in': 4, 'dim_out': 4, 'temp_kernel_size': 4, 'stride': 1, 'dim_inner': 4, 'num_groups': 1}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (L1Loss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (MSELoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (MinusRbfHSIC,
     lambda: ([], {'sigma_x': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (Nonlocal,
     lambda: ([], {'dim': 4, 'dim_inner': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     False),
    (RbfHSIC,
     lambda: ([], {'sigma_x': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (ResNetBasicStem,
     lambda: ([], {'dim_in': 4, 'dim_out': 4, 'kernel': 4, 'stride': 1, 'padding': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (SimpleConvNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
]

class Test_clovaai_rebias(_paritybench_base):
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

