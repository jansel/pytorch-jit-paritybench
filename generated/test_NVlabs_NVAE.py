import sys
_module = sys.modules[__name__]
del sys
datasets = _module
distributions = _module
evaluate = _module
fid = _module
fid_score = _module
inception = _module
lmdb_datasets = _module
model = _module
neural_ar_operations = _module
neural_operations = _module
convert_tfrecord_to_lmdb = _module
create_celeba64_lmdb = _module
create_ffhq_lmdb = _module
precompute_fid_statistics = _module
thirdparty = _module
adamax = _module
functions = _module
inplaced_sync_batchnorm = _module
lsun = _module
swish = _module
train = _module
utils = _module

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


import numpy as np


import torch


import torchvision.datasets as dset


import torchvision.transforms as transforms


from torch.utils.data import Dataset


from scipy.io import loadmat


import torch.nn as nn


import torch.nn.functional as F


import matplotlib.pyplot as plt


from time import time


from torch.multiprocessing import Process


from torch.cuda.amp import autocast


from torch.utils.data import DataLoader


import torchvision.transforms as TF


from scipy import linalg


from torch.nn.functional import adaptive_avg_pool2d


import torchvision


import torch.utils.data as data


import time


from torch.distributions.bernoulli import Bernoulli


from torch.autograd import Variable


from collections import OrderedDict


from torch.optim import Optimizer


from torch.autograd.function import Function


import torch.distributed as dist


from torch.nn.modules.batchnorm import _BatchNorm


from torch.cuda.amp import GradScaler


import logging


def _inception_v3(*args, **kwargs):
    """Wraps `torchvision.models.inception_v3`

    Skips default weight inititialization if supported by torchvision version.
    See https://github.com/mseitzer/pytorch-fid/issues/28.
    """
    try:
        version = tuple(map(int, torchvision.__version__.split('.')[:2]))
    except ValueError:
        version = 0,
    if version >= (0, 6):
        kwargs['init_weights'] = False
    return torchvision.models.inception_v3(*args, **kwargs)


class FIDInceptionA(torchvision.models.inception.InceptionA):
    """InceptionA block patched for FID computation"""

    def __init__(self, in_channels, pool_features):
        super(FIDInceptionA, self).__init__(in_channels, pool_features)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionC(torchvision.models.inception.InceptionC):
    """InceptionC block patched for FID computation"""

    def __init__(self, in_channels, channels_7x7):
        super(FIDInceptionC, self).__init__(in_channels, channels_7x7)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)
        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionE_1(torchvision.models.inception.InceptionE):
    """First InceptionE block patched for FID computation"""

    def __init__(self, in_channels):
        super(FIDInceptionE_1, self).__init__(in_channels)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [self.branch3x3_2a(branch3x3), self.branch3x3_2b(branch3x3)]
        branch3x3 = torch.cat(branch3x3, 1)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [self.branch3x3dbl_3a(branch3x3dbl), self.branch3x3dbl_3b(branch3x3dbl)]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionE_2(torchvision.models.inception.InceptionE):
    """Second InceptionE block patched for FID computation"""

    def __init__(self, in_channels):
        super(FIDInceptionE_2, self).__init__(in_channels)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [self.branch3x3_2a(branch3x3), self.branch3x3_2b(branch3x3)]
        branch3x3 = torch.cat(branch3x3, 1)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [self.branch3x3dbl_3a(branch3x3dbl), self.branch3x3dbl_3b(branch3x3dbl)]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


FID_WEIGHTS_URL = 'https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth'


def fid_inception_v3(model_dir=None):
    """Build pretrained Inception model for FID computation

    The Inception model for FID computation uses a different set of weights
    and has a slightly different structure than torchvision's Inception.

    This method first constructs torchvision's Inception and then patches the
    necessary parts that are different in the FID Inception model.
    """
    inception = _inception_v3(num_classes=1008, aux_logits=False, pretrained=False)
    inception.Mixed_5b = FIDInceptionA(192, pool_features=32)
    inception.Mixed_5c = FIDInceptionA(256, pool_features=64)
    inception.Mixed_5d = FIDInceptionA(288, pool_features=64)
    inception.Mixed_6b = FIDInceptionC(768, channels_7x7=128)
    inception.Mixed_6c = FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6d = FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6e = FIDInceptionC(768, channels_7x7=192)
    inception.Mixed_7b = FIDInceptionE_1(1280)
    inception.Mixed_7c = FIDInceptionE_2(2048)
    state_dict = load_state_dict_from_url(FID_WEIGHTS_URL, model_dir=model_dir, progress=True)
    inception.load_state_dict(state_dict)
    return inception


class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""
    DEFAULT_BLOCK_INDEX = 3
    BLOCK_INDEX_BY_DIM = {(64): 0, (192): 1, (768): 2, (2048): 3}

    def __init__(self, output_blocks=[DEFAULT_BLOCK_INDEX], resize_input=True, normalize_input=True, requires_grad=False, use_fid_inception=True, model_dir=None):
        """Build pretrained InceptionV3

        Parameters
        ----------
        output_blocks : list of int
            Indices of blocks to return features of. Possible values are:
                - 0: corresponds to output of first max pooling
                - 1: corresponds to output of second max pooling
                - 2: corresponds to output which is fed to aux classifier
                - 3: corresponds to output of final average pooling
        resize_input : bool
            If true, bilinearly resizes input to width and height 299 before
            feeding input to model. As the network without fully connected
            layers is fully convolutional, it should be able to handle inputs
            of arbitrary size, so resizing might not be strictly needed
        normalize_input : bool
            If true, scales the input from range (0, 1) to the range the
            pretrained Inception network expects, namely (-1, 1)
        requires_grad : bool
            If true, parameters of the model require gradients. Possibly useful
            for finetuning the network
        use_fid_inception : bool
            If true, uses the pretrained Inception model used in Tensorflow's
            FID implementation. If false, uses the pretrained Inception model
            available in torchvision. The FID Inception model has different
            weights and a slightly different structure from torchvision's
            Inception model. If you want to compute FID scores, you are
            strongly advised to set this parameter to true to get comparable
            results.
        model_dir: is used for storing pretrained checkpoints
        """
        super(InceptionV3, self).__init__()
        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)
        assert self.last_needed_block <= 3, 'Last possible output block index is 3'
        self.blocks = nn.ModuleList()
        if use_fid_inception:
            inception = fid_inception_v3(model_dir)
        else:
            inception = _inception_v3(pretrained=True)
        block0 = [inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3, inception.Conv2d_2b_3x3, nn.MaxPool2d(kernel_size=3, stride=2)]
        self.blocks.append(nn.Sequential(*block0))
        if self.last_needed_block >= 1:
            block1 = [inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3, nn.MaxPool2d(kernel_size=3, stride=2)]
            self.blocks.append(nn.Sequential(*block1))
        if self.last_needed_block >= 2:
            block2 = [inception.Mixed_5b, inception.Mixed_5c, inception.Mixed_5d, inception.Mixed_6a, inception.Mixed_6b, inception.Mixed_6c, inception.Mixed_6d, inception.Mixed_6e]
            self.blocks.append(nn.Sequential(*block2))
        if self.last_needed_block >= 3:
            block3 = [inception.Mixed_7a, inception.Mixed_7b, inception.Mixed_7c, nn.AdaptiveAvgPool2d(output_size=(1, 1))]
            self.blocks.append(nn.Sequential(*block3))
        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        """Get Inception feature maps

        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in
            range (0, 1)

        Returns
        -------
        List of torch.autograd.Variable, corresponding to the selected output
        block, sorted ascending by index
        """
        outp = []
        x = inp
        if self.resize_input:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        if self.normalize_input:
            x = 2 * x - 1
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)
            if idx == self.last_needed_block:
                break
        return outp


CHANNEL_MULT = 2


BN_EPS = 1e-05


def average_tensor(t, is_distributed):
    if is_distributed:
        size = float(dist.get_world_size())
        dist.all_reduce(t.data, op=dist.ReduceOp.SUM)
        t.data /= size


def norm(t, dim):
    return torch.sqrt(torch.sum(t * t, dim))


@torch.jit.script
def normalize_weight_jit(log_weight_norm, weight):
    n = torch.exp(log_weight_norm)
    wn = torch.sqrt(torch.sum(weight * weight, dim=[1, 2, 3]))
    weight = n * weight / (wn.view(-1, 1, 1, 1) + 1e-05)
    return weight


class Conv2D(nn.Conv2d):
    """Allows for weights as input."""

    def __init__(self, C_in, C_out, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, data_init=False, weight_norm=True):
        """
        Args:
            use_shared (bool): Use weights for this layer or not?
        """
        super(Conv2D, self).__init__(C_in, C_out, kernel_size, stride, padding, dilation, groups, bias)
        self.log_weight_norm = None
        if weight_norm:
            init = norm(self.weight, dim=[1, 2, 3]).view(-1, 1, 1, 1)
            self.log_weight_norm = nn.Parameter(torch.log(init + 0.01), requires_grad=True)
        self.data_init = data_init
        self.init_done = False
        self.weight_normalized = self.normalize_weight()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): of size (B, C_in, H, W).
            params (ConvParam): containing `weight` and `bias` (optional) of conv operation.
        """
        if self.data_init and not self.init_done:
            with torch.no_grad():
                weight = self.weight / (norm(self.weight, dim=[1, 2, 3]).view(-1, 1, 1, 1) + 1e-05)
                bias = None
                out = F.conv2d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)
                mn = torch.mean(out, dim=[0, 2, 3])
                st = 5 * torch.std(out, dim=[0, 2, 3])
                average_tensor(mn, is_distributed=True)
                average_tensor(st, is_distributed=True)
                if self.bias is not None:
                    self.bias.data = -mn / (st + 1e-05)
                self.log_weight_norm.data = -torch.log(st.view(-1, 1, 1, 1) + 1e-05)
                self.init_done = True
        self.weight_normalized = self.normalize_weight()
        bias = self.bias
        return F.conv2d(x, self.weight_normalized, bias, self.stride, self.padding, self.dilation, self.groups)

    def normalize_weight(self):
        """ applies weight normalization """
        if self.log_weight_norm is not None:
            weight = normalize_weight_jit(self.log_weight_norm, self.weight)
        else:
            weight = self.weight
        return weight


SYNC_BN = True


class SyncBatchNorm(Function):

    @staticmethod
    def forward(self, input, weight, bias, running_mean, running_var, eps, momentum, process_group, world_size):
        input = input.contiguous()
        count = torch.empty(1, dtype=running_mean.dtype, device=input.device).fill_(input.numel() // input.size(1))
        mean, invstd = torch.batch_norm_stats(input, eps)
        num_channels = input.shape[1]
        combined = torch.cat([mean, invstd, count], dim=0)
        combined_list = [torch.empty_like(combined) for k in range(world_size)]
        dist.all_gather(combined_list, combined, async_op=False)
        combined = torch.stack(combined_list, dim=0)
        mean_all, invstd_all, count_all = torch.split(combined, num_channels, dim=1)
        size = count_all.view(-1).long().sum()
        if size == 1:
            raise ValueError('Expected more than 1 value per channel when training, got input size {}'.format(size))
        mean, invstd = torch.batch_norm_gather_stats_with_counts(input, mean_all, invstd_all, running_mean, running_var, momentum, eps, count_all.view(-1))
        self.save_for_backward(input, weight, mean, invstd, bias, count_all)
        self.process_group = process_group
        out = torch.batch_norm_elemt(input, weight, bias, mean, invstd, eps)
        assert eps == 1e-05, 'I assumed below that eps is 1e-5'
        out = out * torch.sigmoid(out)
        return out

    @staticmethod
    def backward(self, grad_output):
        grad_output = grad_output.contiguous()
        saved_input, weight, mean, invstd, bias, count_tensor = self.saved_tensors
        eps = 1e-05
        out = torch.batch_norm_elemt(saved_input, weight, bias, mean, invstd, eps)
        sigmoid_out = torch.sigmoid(out)
        grad_output *= sigmoid_out * (1 + out * (1 - sigmoid_out))
        grad_input = grad_weight = grad_bias = None
        process_group = self.process_group
        sum_dy, sum_dy_xmu, grad_weight, grad_bias = torch.batch_norm_backward_reduce(grad_output, saved_input, mean, invstd, weight, self.needs_input_grad[0], self.needs_input_grad[1], self.needs_input_grad[2])
        if self.needs_input_grad[0]:
            num_channels = sum_dy.shape[0]
            combined = torch.cat([sum_dy, sum_dy_xmu], dim=0)
            torch.distributed.all_reduce(combined, torch.distributed.ReduceOp.SUM, process_group, async_op=False)
            sum_dy, sum_dy_xmu = torch.split(combined, num_channels)
            divisor = count_tensor.sum()
            mean_dy = sum_dy / divisor
            mean_dy_xmu = sum_dy_xmu / divisor
            grad_input = torch.batch_norm_backward_elemt(grad_output, saved_input, mean, invstd, weight, mean_dy, mean_dy_xmu)
        if weight is None or not self.needs_input_grad[1]:
            grad_weight = None
        if weight is None or not self.needs_input_grad[2]:
            grad_bias = None
        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None


def get_batchnorm(*args, **kwargs):
    if SYNC_BN:
        return SyncBatchNorm(*args, **kwargs)
    else:
        return nn.BatchNorm2d(*args, **kwargs)


class BNELUConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride=1, padding=0, dilation=1):
        super(BNELUConv, self).__init__()
        self.upsample = stride == -1
        stride = abs(stride)
        self.bn = get_batchnorm(C_in, eps=BN_EPS, momentum=0.05)
        self.conv_0 = Conv2D(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=True, dilation=dilation)

    def forward(self, x):
        x = self.bn(x)
        out = F.elu(x)
        if self.upsample:
            out = F.interpolate(out, scale_factor=2, mode='nearest')
        out = self.conv_0(out)
        return out


class SyncBatchNormSwish(_BatchNorm):
    """Applies Batch Normalization over a N-Dimensional input (a mini-batch of [N-2]D inputs
    with additional channel dimension) as described in the paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift`_ .

    .. math::

        y = \\frac{x - \\mathrm{E}[x]}{ \\sqrt{\\mathrm{Var}[x] + \\epsilon}} * \\gamma + \\beta

    The mean and standard-deviation are calculated per-dimension over all
    mini-batches of the same process groups. :math:`\\gamma` and :math:`\\beta`
    are learnable parameter vectors of size `C` (where `C` is the input size).
    By default, the elements of :math:`\\gamma` are sampled from
    :math:`\\mathcal{U}(0, 1)` and the elements of :math:`\\beta` are set to 0.

    Also by default, during training this layer keeps running estimates of its
    computed mean and variance, which are then used for normalization during
    evaluation. The running estimates are kept with a default :attr:`momentum`
    of 0.1.

    If :attr:`track_running_stats` is set to ``False``, this layer then does not
    keep running estimates, and batch statistics are instead used during
    evaluation time as well.

    .. note::
        This :attr:`momentum` argument is different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically, the
        update rule for running statistics here is
        :math:`\\hat{x}_\\text{new} = (1 - \\text{momentum}) \\times \\hat{x} + \\text{momemtum} \\times x_t`,
        where :math:`\\hat{x}` is the estimated statistic and :math:`x_t` is the
        new observed value.

    Because the Batch Normalization is done over the `C` dimension, computing statistics
    on `(N, +)` slices, it's common terminology to call this Volumetric Batch Normalization
    or Spatio-temporal Batch Normalization.

    Currently SyncBatchNorm only supports DistributedDataParallel with single GPU per process. Use
    torch.nn.SyncBatchNorm.convert_sync_batchnorm() to convert BatchNorm layer to SyncBatchNorm before wrapping
    Network with DDP.

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, +)`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics and always uses batch
            statistics in both training and eval modes. Default: ``True``
        process_group: synchronization of stats happen within each process group
            individually. Default behavior is synchronization across the whole
            world

    Shape:
        - Input: :math:`(N, C, +)`
        - Output: :math:`(N, C, +)` (same shape as input)

    Examples::

        >>> # With Learnable Parameters
        >>> m = nn.SyncBatchNorm(100)
        >>> # creating process group (optional)
        >>> # process_ids is a list of int identifying rank ids.
        >>> process_group = torch.distributed.new_group(process_ids)
        >>> # Without Learnable Parameters
        >>> m = nn.BatchNorm3d(100, affine=False, process_group=process_group)
        >>> input = torch.randn(20, 100, 35, 45, 10)
        >>> output = m(input)

        >>> # network is nn.BatchNorm layer
        >>> sync_bn_network = nn.SyncBatchNorm.convert_sync_batchnorm(network, process_group)
        >>> # only single gpu per process is currently supported
        >>> ddp_sync_bn_network = torch.nn.parallel.DistributedDataParallel(
        >>>                         sync_bn_network,
        >>>                         device_ids=[args.local_rank],
        >>>                         output_device=args.local_rank)

    .. _`Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift`:
        https://arxiv.org/abs/1502.03167
    """

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, process_group=None):
        super(SyncBatchNormSwish, self).__init__(num_features, eps, momentum, affine, track_running_stats)
        self.process_group = process_group
        self.ddp_gpu_size = None

    def _check_input_dim(self, input):
        if input.dim() < 2:
            raise ValueError('expected at least 2D input (got {}D input)'.format(input.dim()))

    def _specify_ddp_gpu_num(self, gpu_size):
        if gpu_size > 1:
            raise ValueError('SyncBatchNorm is only supported for DDP with single GPU per process')
        self.ddp_gpu_size = gpu_size

    def forward(self, input):
        if not input.is_cuda:
            raise ValueError('SyncBatchNorm expected input tensor to be on GPU')
        self._check_input_dim(input)
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum
        if self.training and self.track_running_stats:
            self.num_batches_tracked = self.num_batches_tracked + 1
            if self.momentum is None:
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:
                exponential_average_factor = self.momentum
        need_sync = self.training or not self.track_running_stats
        if need_sync:
            process_group = torch.distributed.group.WORLD
            if self.process_group:
                process_group = self.process_group
            world_size = torch.distributed.get_world_size(process_group)
            need_sync = world_size > 1
        if not need_sync:
            out = F.batch_norm(input, self.running_mean, self.running_var, self.weight, self.bias, self.training or not self.track_running_stats, exponential_average_factor, self.eps)
            return swish.apply(out)
        else:
            if not self.ddp_gpu_size and False:
                raise AttributeError('SyncBatchNorm is only supported within torch.nn.parallel.DistributedDataParallel')
            return sync_batch_norm.apply(input, self.weight, self.bias, self.running_mean, self.running_var, self.eps, exponential_average_factor, process_group, world_size)


class BNSwishConv(nn.Module):
    """ReLU + Conv2d + BN."""

    def __init__(self, C_in, C_out, kernel_size, stride=1, padding=0, dilation=1):
        super(BNSwishConv, self).__init__()
        self.upsample = stride == -1
        stride = abs(stride)
        self.bn_act = SyncBatchNormSwish(C_in, eps=BN_EPS, momentum=0.05)
        self.conv_0 = Conv2D(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=True, dilation=dilation)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): of size (B, C_in, H, W)
        """
        out = self.bn_act(x)
        if self.upsample:
            out = F.interpolate(out, scale_factor=2, mode='nearest')
        out = self.conv_0(out)
        return out


class ELUConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride=1, padding=0, dilation=1):
        super(ELUConv, self).__init__()
        self.upsample = stride == -1
        stride = abs(stride)
        self.conv_0 = Conv2D(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=True, dilation=dilation, data_init=True)

    def forward(self, x):
        out = F.elu(x)
        if self.upsample:
            out = F.interpolate(out, scale_factor=2, mode='nearest')
        out = self.conv_0(out)
        return out


class ConvBNSwish(nn.Module):

    def __init__(self, Cin, Cout, k=3, stride=1, groups=1, dilation=1):
        padding = dilation * (k - 1) // 2
        super(ConvBNSwish, self).__init__()
        self.conv = nn.Sequential(Conv2D(Cin, Cout, k, stride, padding, groups=groups, bias=False, dilation=dilation, weight_norm=False), SyncBatchNormSwish(Cout, eps=BN_EPS, momentum=0.05))

    def forward(self, x):
        return self.conv(x)


class InvertedResidual(nn.Module):

    def __init__(self, Cin, Cout, stride, ex, dil, k, g):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2, -1]
        hidden_dim = int(round(Cin * ex))
        self.use_res_connect = self.stride == 1 and Cin == Cout
        self.upsample = self.stride == -1
        self.stride = abs(self.stride)
        groups = hidden_dim if g == 0 else g
        layers0 = [nn.UpsamplingNearest2d(scale_factor=2)] if self.upsample else []
        layers = [get_batchnorm(Cin, eps=BN_EPS, momentum=0.05), ConvBNSwish(Cin, hidden_dim, k=1), ConvBNSwish(hidden_dim, hidden_dim, stride=self.stride, groups=groups, k=k, dilation=dil), Conv2D(hidden_dim, Cout, 1, 1, 0, bias=False, weight_norm=False), get_batchnorm(Cout, momentum=0.05)]
        layers0.extend(layers)
        self.conv = nn.Sequential(*layers0)

    def forward(self, x):
        return self.conv(x)


OPS = OrderedDict([('res_elu', lambda Cin, Cout, stride: ELUConv(Cin, Cout, 3, stride, 1)), ('res_bnelu', lambda Cin, Cout, stride: BNELUConv(Cin, Cout, 3, stride, 1)), ('res_bnswish', lambda Cin, Cout, stride: BNSwishConv(Cin, Cout, 3, stride, 1)), ('res_bnswish5', lambda Cin, Cout, stride: BNSwishConv(Cin, Cout, 3, stride, 2, 2)), ('mconv_e6k5g0', lambda Cin, Cout, stride: InvertedResidual(Cin, Cout, stride, ex=6, dil=1, k=5, g=0)), ('mconv_e3k5g0', lambda Cin, Cout, stride: InvertedResidual(Cin, Cout, stride, ex=3, dil=1, k=5, g=0)), ('mconv_e3k5g8', lambda Cin, Cout, stride: InvertedResidual(Cin, Cout, stride, ex=3, dil=1, k=5, g=8)), ('mconv_e6k11g0', lambda Cin, Cout, stride: InvertedResidual(Cin, Cout, stride, ex=6, dil=1, k=11, g=0))])


class SE(nn.Module):

    def __init__(self, Cin, Cout):
        super(SE, self).__init__()
        num_hidden = max(Cout // 16, 4)
        self.se = nn.Sequential(nn.Linear(Cin, num_hidden), nn.ReLU(inplace=True), nn.Linear(num_hidden, Cout), nn.Sigmoid())

    def forward(self, x):
        se = torch.mean(x, dim=[2, 3])
        se = se.view(se.size(0), -1)
        se = self.se(se)
        se = se.view(se.size(0), -1, 1, 1)
        return x * se


def act(t):
    return SwishFN.apply(t)


class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.conv_1 = Conv2D(C_in, C_out // 4, 1, stride=2, padding=0, bias=True)
        self.conv_2 = Conv2D(C_in, C_out // 4, 1, stride=2, padding=0, bias=True)
        self.conv_3 = Conv2D(C_in, C_out // 4, 1, stride=2, padding=0, bias=True)
        self.conv_4 = Conv2D(C_in, C_out - 3 * (C_out // 4), 1, stride=2, padding=0, bias=True)

    def forward(self, x):
        out = act(x)
        conv1 = self.conv_1(out)
        conv2 = self.conv_2(out[:, :, 1:, 1:])
        conv3 = self.conv_3(out[:, :, :, 1:])
        conv4 = self.conv_4(out[:, :, 1:, :])
        out = torch.cat([conv1, conv2, conv3, conv4], dim=1)
        return out


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class UpSample(nn.Module):

    def __init__(self):
        super(UpSample, self).__init__()
        pass

    def forward(self, x):
        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)


def get_skip_connection(C, stride, affine, channel_mult):
    if stride == 1:
        return Identity()
    elif stride == 2:
        return FactorizedReduce(C, int(channel_mult * C))
    elif stride == -1:
        return nn.Sequential(UpSample(), Conv2D(C, int(C / channel_mult), kernel_size=1))


def get_stride_for_cell_type(cell_type):
    if cell_type.startswith('normal') or cell_type.startswith('combiner'):
        stride = 1
    elif cell_type.startswith('down'):
        stride = 2
    elif cell_type.startswith('up'):
        stride = -1
    else:
        raise NotImplementedError(cell_type)
    return stride


class Cell(nn.Module):

    def __init__(self, Cin, Cout, cell_type, arch, use_se):
        super(Cell, self).__init__()
        self.cell_type = cell_type
        stride = get_stride_for_cell_type(self.cell_type)
        self.skip = get_skip_connection(Cin, stride, affine=False, channel_mult=CHANNEL_MULT)
        self.use_se = use_se
        self._num_nodes = len(arch)
        self._ops = nn.ModuleList()
        for i in range(self._num_nodes):
            stride = get_stride_for_cell_type(self.cell_type) if i == 0 else 1
            C = Cin if i == 0 else Cout
            primitive = arch[i]
            op = OPS[primitive](C, Cout, stride)
            self._ops.append(op)
        if self.use_se:
            self.se = SE(Cout, Cout)

    def forward(self, s):
        skip = self.skip(s)
        for i in range(self._num_nodes):
            s = self._ops[i](s)
        s = self.se(s) if self.use_se else s
        return skip + 0.1 * s


def channel_mask(c_in, g_in, c_out, zero_diag):
    assert c_in % c_out == 0 or c_out % c_in == 0, '%d - %d' % (c_in, c_out)
    assert g_in == 1 or g_in == c_in
    if g_in == 1:
        mask = np.ones([c_out, c_in], dtype=np.float32)
        if c_out >= c_in:
            ratio = c_out // c_in
            for i in range(c_in):
                mask[i * ratio:(i + 1) * ratio, i + 1:] = 0
                if zero_diag:
                    mask[i * ratio:(i + 1) * ratio, i:i + 1] = 0
        else:
            ratio = c_in // c_out
            for i in range(c_out):
                mask[i:i + 1, (i + 1) * ratio:] = 0
                if zero_diag:
                    mask[i:i + 1, i * ratio:(i + 1) * ratio] = 0
    elif g_in == c_in:
        mask = np.ones([c_out, c_in // g_in], dtype=np.float32)
        if zero_diag:
            mask = 0.0 * mask
    return mask


def create_conv_mask(kernel_size, c_in, g_in, c_out, zero_diag, mirror):
    m = (kernel_size - 1) // 2
    mask = np.ones([c_out, c_in // g_in, kernel_size, kernel_size], dtype=np.float32)
    mask[:, :, m:, :] = 0
    mask[:, :, m, :m] = 1
    mask[:, :, m, m] = channel_mask(c_in, g_in, c_out, zero_diag)
    if mirror:
        mask = np.copy(mask[:, :, ::-1, ::-1])
    return mask


class ARConv2d(nn.Conv2d):
    """Allows for weights as input."""

    def __init__(self, C_in, C_out, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, masked=False, zero_diag=False, mirror=False):
        """
        Args:
            use_shared (bool): Use weights for this layer or not?
        """
        super(ARConv2d, self).__init__(C_in, C_out, kernel_size, stride, padding, dilation, groups, bias)
        self.masked = masked
        if self.masked:
            assert kernel_size % 2 == 1, 'kernel size should be an odd value.'
            self.mask = torch.from_numpy(create_conv_mask(kernel_size, C_in, groups, C_out, zero_diag, mirror))
            init_mask = self.mask.cpu()
        else:
            self.mask = 1.0
            init_mask = 1.0
        init = torch.log(norm(self.weight * init_mask, dim=[1, 2, 3]).view(-1, 1, 1, 1) + 0.01)
        self.log_weight_norm = nn.Parameter(init, requires_grad=True)
        self.weight_normalized = None

    def normalize_weight(self):
        weight = self.weight
        if self.masked:
            assert self.mask.size() == weight.size()
            weight = weight * self.mask
        weight = normalize_weight_jit(self.log_weight_norm, weight)
        return weight

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): of size (B, C_in, H, W).
            params (ConvParam): containing `weight` and `bias` (optional) of conv operation.
        """
        self.weight_normalized = self.normalize_weight()
        bias = self.bias
        return F.conv2d(x, self.weight_normalized, bias, self.stride, self.padding, self.dilation, self.groups)


class ARInvertedResidual(nn.Module):

    def __init__(self, inz, inf, ex=6, dil=1, k=5, mirror=False):
        super(ARInvertedResidual, self).__init__()
        hidden_dim = int(round(inz * ex))
        padding = dil * (k - 1) // 2
        layers = []
        layers.extend([ARConv2d(inz, hidden_dim, kernel_size=3, padding=1, masked=True, mirror=mirror, zero_diag=True), nn.ELU(inplace=True)])
        layers.extend([ARConv2d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=k, padding=padding, dilation=dil, masked=True, mirror=mirror, zero_diag=False), nn.ELU(inplace=True)])
        self.convz = nn.Sequential(*layers)
        self.hidden_dim = hidden_dim

    def forward(self, z, ftr):
        z = self.convz(z)
        return z


class MixLogCDFParam(nn.Module):

    def __init__(self, num_z, num_mix, num_ftr, mirror):
        super(MixLogCDFParam, self).__init__()
        num_out = num_z * (3 * num_mix + 3)
        self.conv = ELUConv(num_ftr, num_out, kernel_size=1, padding=0, masked=True, zero_diag=False, weight_init_coeff=0.1, mirror=mirror)
        self.num_z = num_z
        self.num_mix = num_mix

    def forward(self, ftr):
        out = self.conv(ftr)
        b, c, h, w = out.size()
        out = out.view(b, self.num_z, c // self.num_z, h, w)
        m = self.num_mix
        logit_pi, mu, log_s, log_a, b, _ = torch.split(out, [m, m, m, 1, 1, 1], dim=2)
        return logit_pi, mu, log_s, log_a, b


def mix_log_cdf_flow(z1, logit_pi, mu, log_s, log_a, b):
    log_s = torch.clamp(log_s, min=-7)
    z = z1.unsqueeze(dim=2)
    log_pi = torch.log_softmax(logit_pi, dim=2)
    u = -(z - mu) * torch.exp(-log_s)
    softplus_u = F.softplus(u)
    log_mix_cdf = log_pi - softplus_u
    log_one_minus_mix_cdf = log_mix_cdf + u
    log_mix_cdf = torch.logsumexp(log_mix_cdf, dim=2)
    log_one_minus_mix_cdf = torch.logsumexp(log_one_minus_mix_cdf, dim=2)
    log_a = log_a.squeeze_(dim=2)
    b = b.squeeze_(dim=2)
    new_z = torch.exp(log_a) * (log_mix_cdf - log_one_minus_mix_cdf) + b
    log_mix_pdf = torch.logsumexp(log_pi + u - log_s - 2 * softplus_u, dim=2)
    log_det = log_a - log_mix_cdf - log_one_minus_mix_cdf + log_mix_pdf
    return new_z, log_det


class CellAR(nn.Module):

    def __init__(self, num_z, num_ftr, num_c, arch, mirror):
        super(CellAR, self).__init__()
        assert num_c % num_z == 0
        self.cell_type = 'ar_nn'
        ex = 6
        self.conv = ARInvertedResidual(num_z, num_ftr, ex=ex, mirror=mirror)
        self.use_mix_log_cdf = False
        if self.use_mix_log_cdf:
            self.param = MixLogCDFParam(num_z, num_mix=3, num_ftr=self.conv.hidden_dim, mirror=mirror)
        else:
            self.mu = ARELUConv(self.conv.hidden_dim, num_z, kernel_size=1, padding=0, masked=True, zero_diag=False, weight_init_coeff=0.1, mirror=mirror)

    def forward(self, z, ftr):
        s = self.conv(z, ftr)
        if self.use_mix_log_cdf:
            logit_pi, mu, log_s, log_a, b = self.param(s)
            new_z, log_det = mix_log_cdf_flow(z, logit_pi, mu, log_s, log_a, b)
        else:
            mu = self.mu(s)
            new_z = z - mu
            log_det = torch.zeros_like(new_z)
        return new_z, log_det


class PairedCellAR(nn.Module):

    def __init__(self, num_z, num_ftr, num_c, arch=None):
        super(PairedCellAR, self).__init__()
        self.cell1 = CellAR(num_z, num_ftr, num_c, arch, mirror=False)
        self.cell2 = CellAR(num_z, num_ftr, num_c, arch, mirror=True)

    def forward(self, z, ftr):
        new_z, log_det1 = self.cell1(z, ftr)
        new_z, log_det2 = self.cell2(new_z, ftr)
        log_det1 += log_det2
        return new_z, log_det1


class DecCombinerCell(nn.Module):

    def __init__(self, Cin1, Cin2, Cout, cell_type):
        super(DecCombinerCell, self).__init__()
        self.cell_type = cell_type
        self.conv = Conv2D(Cin1 + Cin2, Cout, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x1, x2):
        out = torch.cat([x1, x2], dim=1)
        out = self.conv(out)
        return out


def one_hot(indices, depth, dim):
    indices = indices.unsqueeze(dim)
    size = list(indices.size())
    size[dim] = depth
    y_onehot = torch.zeros(size)
    y_onehot.zero_()
    y_onehot.scatter_(dim, indices, 1)
    return y_onehot


class DiscMixLogistic:

    def __init__(self, param, num_mix=10, num_bits=8):
        B, C, H, W = param.size()
        self.num_mix = num_mix
        self.logit_probs = param[:, :num_mix, :, :]
        l = param[:, num_mix:, :, :].view(B, 3, 3 * num_mix, H, W)
        self.means = l[:, :, :num_mix, :, :]
        self.log_scales = torch.clamp(l[:, :, num_mix:2 * num_mix, :, :], min=-7.0)
        self.coeffs = torch.tanh(l[:, :, 2 * num_mix:3 * num_mix, :, :])
        self.max_val = 2.0 ** num_bits - 1

    def log_prob(self, samples):
        assert torch.max(samples) <= 1.0 and torch.min(samples) >= 0.0
        samples = 2 * samples - 1.0
        B, C, H, W = samples.size()
        assert C == 3, 'only RGB images are considered.'
        samples = samples.unsqueeze(4)
        samples = samples.expand(-1, -1, -1, -1, self.num_mix).permute(0, 1, 4, 2, 3)
        mean1 = self.means[:, 0, :, :, :]
        mean2 = self.means[:, 1, :, :, :] + self.coeffs[:, 0, :, :, :] * samples[:, 0, :, :, :]
        mean3 = self.means[:, 2, :, :, :] + self.coeffs[:, 1, :, :, :] * samples[:, 0, :, :, :] + self.coeffs[:, 2, :, :, :] * samples[:, 1, :, :, :]
        mean1 = mean1.unsqueeze(1)
        mean2 = mean2.unsqueeze(1)
        mean3 = mean3.unsqueeze(1)
        means = torch.cat([mean1, mean2, mean3], dim=1)
        centered = samples - means
        inv_stdv = torch.exp(-self.log_scales)
        plus_in = inv_stdv * (centered + 1.0 / self.max_val)
        cdf_plus = torch.sigmoid(plus_in)
        min_in = inv_stdv * (centered - 1.0 / self.max_val)
        cdf_min = torch.sigmoid(min_in)
        log_cdf_plus = plus_in - F.softplus(plus_in)
        log_one_minus_cdf_min = -F.softplus(min_in)
        cdf_delta = cdf_plus - cdf_min
        mid_in = inv_stdv * centered
        log_pdf_mid = mid_in - self.log_scales - 2.0 * F.softplus(mid_in)
        log_prob_mid_safe = torch.where(cdf_delta > 1e-05, torch.log(torch.clamp(cdf_delta, min=1e-10)), log_pdf_mid - np.log(self.max_val / 2))
        log_probs = torch.where(samples < -0.999, log_cdf_plus, torch.where(samples > 0.99, log_one_minus_cdf_min, log_prob_mid_safe))
        log_probs = torch.sum(log_probs, 1) + F.log_softmax(self.logit_probs, dim=1)
        return torch.logsumexp(log_probs, dim=1)

    def sample(self, t=1.0):
        gumbel = -torch.log(-torch.log(torch.Tensor(self.logit_probs.size()).uniform_(1e-05, 1.0 - 1e-05)))
        sel = one_hot(torch.argmax(self.logit_probs / t + gumbel, 1), self.num_mix, dim=1)
        sel = sel.unsqueeze(1)
        means = torch.sum(self.means * sel, dim=2)
        log_scales = torch.sum(self.log_scales * sel, dim=2)
        coeffs = torch.sum(self.coeffs * sel, dim=2)
        u = torch.Tensor(means.size()).uniform_(1e-05, 1.0 - 1e-05)
        x = means + torch.exp(log_scales) / t * (torch.log(u) - torch.log(1.0 - u))
        x0 = torch.clamp(x[:, 0, :, :], -1, 1.0)
        x1 = torch.clamp(x[:, 1, :, :] + coeffs[:, 0, :, :] * x0, -1, 1)
        x2 = torch.clamp(x[:, 2, :, :] + coeffs[:, 1, :, :] * x0 + coeffs[:, 2, :, :] * x1, -1, 1)
        x0 = x0.unsqueeze(1)
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)
        x = torch.cat([x0, x1, x2], 1)
        x = x / 2.0 + 0.5
        return x

    def mean(self):
        sel = torch.softmax(self.logit_probs, dim=1)
        sel = sel.unsqueeze(1)
        means = torch.sum(self.means * sel, dim=2)
        coeffs = torch.sum(self.coeffs * sel, dim=2)
        x = means
        x0 = torch.clamp(x[:, 0, :, :], -1, 1.0)
        x1 = torch.clamp(x[:, 1, :, :] + coeffs[:, 0, :, :] * x0, -1, 1)
        x2 = torch.clamp(x[:, 2, :, :] + coeffs[:, 1, :, :] * x0 + coeffs[:, 2, :, :] * x1, -1, 1)
        x0 = x0.unsqueeze(1)
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)
        x = torch.cat([x0, x1, x2], 1)
        x = x / 2.0 + 0.5
        return x


class EncCombinerCell(nn.Module):

    def __init__(self, Cin1, Cin2, Cout, cell_type):
        super(EncCombinerCell, self).__init__()
        self.cell_type = cell_type
        self.conv = Conv2D(Cin2, Cout, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x1, x2):
        x2 = self.conv(x2)
        out = x1 + x2
        return out


@torch.jit.script
def sample_normal_jit(mu, sigma):
    eps = mu.mul(0).normal_()
    z = eps.mul_(sigma).add_(mu)
    return z, eps


@torch.jit.script
def soft_clamp5(x: torch.Tensor):
    return x.div(5.0).tanh_().mul(5.0)


class Normal:

    def __init__(self, mu, log_sigma, temp=1.0):
        self.mu = soft_clamp5(mu)
        log_sigma = soft_clamp5(log_sigma)
        self.sigma = torch.exp(log_sigma) + 0.01
        if temp != 1.0:
            self.sigma *= temp

    def sample(self):
        return sample_normal_jit(self.mu, self.sigma)

    def sample_given_eps(self, eps):
        return eps * self.sigma + self.mu

    def log_p(self, samples):
        normalized_samples = (samples - self.mu) / self.sigma
        log_p = -0.5 * normalized_samples * normalized_samples - 0.5 * np.log(2 * np.pi) - torch.log(self.sigma)
        return log_p

    def kl(self, normal_dist):
        term1 = (self.mu - normal_dist.mu) / normal_dist.sigma
        term2 = self.sigma / normal_dist.sigma
        return 0.5 * (term1 * term1 + term2 * term2) - 0.5 - torch.log(term2)


class NormalDecoder:

    def __init__(self, param, num_bits=8):
        B, C, H, W = param.size()
        self.num_c = C // 2
        mu = param[:, :self.num_c, :, :]
        log_sigma = param[:, self.num_c:, :, :]
        self.dist = Normal(mu, log_sigma)

    def log_prob(self, samples):
        assert torch.max(samples) <= 1.0 and torch.min(samples) >= 0.0
        samples = 2 * samples - 1.0
        return self.dist.log_p(samples)

    def sample(self, t=1.0):
        x, _ = self.dist.sample()
        x = torch.clamp(x, -1, 1.0)
        x = x / 2.0 + 0.5
        return x


def get_input_size(dataset):
    if dataset in {'mnist', 'omniglot'}:
        return 32
    elif dataset == 'cifar10':
        return 32
    elif dataset.startswith('celeba') or dataset.startswith('imagenet') or dataset.startswith('lsun'):
        size = int(dataset.split('_')[-1])
        return size
    elif dataset == 'ffhq':
        return 256
    else:
        raise NotImplementedError


def groups_per_scale(num_scales, num_groups_per_scale, is_adaptive, divider=2, minimum_groups=1):
    g = []
    n = num_groups_per_scale
    for s in range(num_scales):
        assert n >= 1
        g.append(n)
        if is_adaptive:
            n = n // divider
            n = max(minimum_groups, n)
    return g


class AutoEncoder(nn.Module):

    def __init__(self, args, writer, arch_instance):
        super(AutoEncoder, self).__init__()
        self.writer = writer
        self.arch_instance = arch_instance
        self.dataset = args.dataset
        self.crop_output = self.dataset in {'mnist', 'omniglot', 'stacked_mnist'}
        self.use_se = args.use_se
        self.res_dist = args.res_dist
        self.num_bits = args.num_x_bits
        self.num_latent_scales = args.num_latent_scales
        self.num_groups_per_scale = args.num_groups_per_scale
        self.num_latent_per_group = args.num_latent_per_group
        self.groups_per_scale = groups_per_scale(self.num_latent_scales, self.num_groups_per_scale, args.ada_groups, minimum_groups=args.min_groups_per_scale)
        self.vanilla_vae = self.num_latent_scales == 1 and self.num_groups_per_scale == 1
        self.num_channels_enc = args.num_channels_enc
        self.num_channels_dec = args.num_channels_dec
        self.num_preprocess_blocks = args.num_preprocess_blocks
        self.num_preprocess_cells = args.num_preprocess_cells
        self.num_cell_per_cond_enc = args.num_cell_per_cond_enc
        self.num_postprocess_blocks = args.num_postprocess_blocks
        self.num_postprocess_cells = args.num_postprocess_cells
        self.num_cell_per_cond_dec = args.num_cell_per_cond_dec
        self.input_size = get_input_size(self.dataset)
        self.num_mix_output = args.num_mixture_dec
        c_scaling = CHANNEL_MULT ** (self.num_preprocess_blocks + self.num_latent_scales - 1)
        spatial_scaling = 2 ** (self.num_preprocess_blocks + self.num_latent_scales - 1)
        prior_ftr0_size = int(c_scaling * self.num_channels_dec), self.input_size // spatial_scaling, self.input_size // spatial_scaling
        self.prior_ftr0 = nn.Parameter(torch.rand(size=prior_ftr0_size), requires_grad=True)
        self.z0_size = [self.num_latent_per_group, self.input_size // spatial_scaling, self.input_size // spatial_scaling]
        self.stem = self.init_stem()
        self.pre_process, mult = self.init_pre_process(mult=1)
        if self.vanilla_vae:
            self.enc_tower = []
        else:
            self.enc_tower, mult = self.init_encoder_tower(mult)
        self.with_nf = args.num_nf > 0
        self.num_flows = args.num_nf
        self.enc0 = self.init_encoder0(mult)
        self.enc_sampler, self.dec_sampler, self.nf_cells, self.enc_kv, self.dec_kv, self.query = self.init_normal_sampler(mult)
        if self.vanilla_vae:
            self.dec_tower = []
            self.stem_decoder = Conv2D(self.num_latent_per_group, mult * self.num_channels_enc, (1, 1), bias=True)
        else:
            self.dec_tower, mult = self.init_decoder_tower(mult)
        self.post_process, mult = self.init_post_process(mult)
        self.image_conditional = self.init_image_conditional(mult)
        self.all_log_norm = []
        self.all_conv_layers = []
        self.all_bn_layers = []
        for n, layer in self.named_modules():
            if isinstance(layer, Conv2D) or isinstance(layer, ARConv2d):
                self.all_log_norm.append(layer.log_weight_norm)
                self.all_conv_layers.append(layer)
            if isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.SyncBatchNorm) or isinstance(layer, SyncBatchNormSwish):
                self.all_bn_layers.append(layer)
        None
        None
        self.sr_u = {}
        self.sr_v = {}
        self.num_power_iter = 4

    def init_stem(self):
        Cout = self.num_channels_enc
        Cin = 1 if self.dataset in {'mnist', 'omniglot'} else 3
        stem = Conv2D(Cin, Cout, 3, padding=1, bias=True)
        return stem

    def init_pre_process(self, mult):
        pre_process = nn.ModuleList()
        for b in range(self.num_preprocess_blocks):
            for c in range(self.num_preprocess_cells):
                if c == self.num_preprocess_cells - 1:
                    arch = self.arch_instance['down_pre']
                    num_ci = int(self.num_channels_enc * mult)
                    num_co = int(CHANNEL_MULT * num_ci)
                    cell = Cell(num_ci, num_co, cell_type='down_pre', arch=arch, use_se=self.use_se)
                    mult = CHANNEL_MULT * mult
                else:
                    arch = self.arch_instance['normal_pre']
                    num_c = self.num_channels_enc * mult
                    cell = Cell(num_c, num_c, cell_type='normal_pre', arch=arch, use_se=self.use_se)
                pre_process.append(cell)
        return pre_process, mult

    def init_encoder_tower(self, mult):
        enc_tower = nn.ModuleList()
        for s in range(self.num_latent_scales):
            for g in range(self.groups_per_scale[s]):
                for c in range(self.num_cell_per_cond_enc):
                    arch = self.arch_instance['normal_enc']
                    num_c = int(self.num_channels_enc * mult)
                    cell = Cell(num_c, num_c, cell_type='normal_enc', arch=arch, use_se=self.use_se)
                    enc_tower.append(cell)
                if not (s == self.num_latent_scales - 1 and g == self.groups_per_scale[s] - 1):
                    num_ce = int(self.num_channels_enc * mult)
                    num_cd = int(self.num_channels_dec * mult)
                    cell = EncCombinerCell(num_ce, num_cd, num_ce, cell_type='combiner_enc')
                    enc_tower.append(cell)
            if s < self.num_latent_scales - 1:
                arch = self.arch_instance['down_enc']
                num_ci = int(self.num_channels_enc * mult)
                num_co = int(CHANNEL_MULT * num_ci)
                cell = Cell(num_ci, num_co, cell_type='down_enc', arch=arch, use_se=self.use_se)
                enc_tower.append(cell)
                mult = CHANNEL_MULT * mult
        return enc_tower, mult

    def init_encoder0(self, mult):
        num_c = int(self.num_channels_enc * mult)
        cell = nn.Sequential(nn.ELU(), Conv2D(num_c, num_c, kernel_size=1, bias=True), nn.ELU())
        return cell

    def init_normal_sampler(self, mult):
        enc_sampler, dec_sampler, nf_cells = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        enc_kv, dec_kv, query = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for s in range(self.num_latent_scales):
            for g in range(self.groups_per_scale[self.num_latent_scales - s - 1]):
                num_c = int(self.num_channels_enc * mult)
                cell = Conv2D(num_c, 2 * self.num_latent_per_group, kernel_size=3, padding=1, bias=True)
                enc_sampler.append(cell)
                for n in range(self.num_flows):
                    arch = self.arch_instance['ar_nn']
                    num_c1 = int(self.num_channels_enc * mult)
                    num_c2 = 8 * self.num_latent_per_group
                    nf_cells.append(PairedCellAR(self.num_latent_per_group, num_c1, num_c2, arch))
                if not (s == 0 and g == 0):
                    num_c = int(self.num_channels_dec * mult)
                    cell = nn.Sequential(nn.ELU(), Conv2D(num_c, 2 * self.num_latent_per_group, kernel_size=1, padding=0, bias=True))
                    dec_sampler.append(cell)
            mult = mult / CHANNEL_MULT
        return enc_sampler, dec_sampler, nf_cells, enc_kv, dec_kv, query

    def init_decoder_tower(self, mult):
        dec_tower = nn.ModuleList()
        for s in range(self.num_latent_scales):
            for g in range(self.groups_per_scale[self.num_latent_scales - s - 1]):
                num_c = int(self.num_channels_dec * mult)
                if not (s == 0 and g == 0):
                    for c in range(self.num_cell_per_cond_dec):
                        arch = self.arch_instance['normal_dec']
                        cell = Cell(num_c, num_c, cell_type='normal_dec', arch=arch, use_se=self.use_se)
                        dec_tower.append(cell)
                cell = DecCombinerCell(num_c, self.num_latent_per_group, num_c, cell_type='combiner_dec')
                dec_tower.append(cell)
            if s < self.num_latent_scales - 1:
                arch = self.arch_instance['up_dec']
                num_ci = int(self.num_channels_dec * mult)
                num_co = int(num_ci / CHANNEL_MULT)
                cell = Cell(num_ci, num_co, cell_type='up_dec', arch=arch, use_se=self.use_se)
                dec_tower.append(cell)
                mult = mult / CHANNEL_MULT
        return dec_tower, mult

    def init_post_process(self, mult):
        post_process = nn.ModuleList()
        for b in range(self.num_postprocess_blocks):
            for c in range(self.num_postprocess_cells):
                if c == 0:
                    arch = self.arch_instance['up_post']
                    num_ci = int(self.num_channels_dec * mult)
                    num_co = int(num_ci / CHANNEL_MULT)
                    cell = Cell(num_ci, num_co, cell_type='up_post', arch=arch, use_se=self.use_se)
                    mult = mult / CHANNEL_MULT
                else:
                    arch = self.arch_instance['normal_post']
                    num_c = int(self.num_channels_dec * mult)
                    cell = Cell(num_c, num_c, cell_type='normal_post', arch=arch, use_se=self.use_se)
                post_process.append(cell)
        return post_process, mult

    def init_image_conditional(self, mult):
        C_in = int(self.num_channels_dec * mult)
        if self.dataset in {'mnist', 'omniglot'}:
            C_out = 1
        elif self.num_mix_output == 1:
            C_out = 2 * 3
        else:
            C_out = 10 * self.num_mix_output
        return nn.Sequential(nn.ELU(), Conv2D(C_in, C_out, 3, padding=1, bias=True))

    def forward(self, x):
        s = self.stem(2 * x - 1.0)
        for cell in self.pre_process:
            s = cell(s)
        combiner_cells_enc = []
        combiner_cells_s = []
        for cell in self.enc_tower:
            if cell.cell_type == 'combiner_enc':
                combiner_cells_enc.append(cell)
                combiner_cells_s.append(s)
            else:
                s = cell(s)
        combiner_cells_enc.reverse()
        combiner_cells_s.reverse()
        idx_dec = 0
        ftr = self.enc0(s)
        param0 = self.enc_sampler[idx_dec](ftr)
        mu_q, log_sig_q = torch.chunk(param0, 2, dim=1)
        dist = Normal(mu_q, log_sig_q)
        z, _ = dist.sample()
        log_q_conv = dist.log_p(z)
        nf_offset = 0
        for n in range(self.num_flows):
            z, log_det = self.nf_cells[n](z, ftr)
            log_q_conv -= log_det
        nf_offset += self.num_flows
        all_q = [dist]
        all_log_q = [log_q_conv]
        s = 0
        dist = Normal(mu=torch.zeros_like(z), log_sigma=torch.zeros_like(z))
        log_p_conv = dist.log_p(z)
        all_p = [dist]
        all_log_p = [log_p_conv]
        idx_dec = 0
        s = self.prior_ftr0.unsqueeze(0)
        batch_size = z.size(0)
        s = s.expand(batch_size, -1, -1, -1)
        for cell in self.dec_tower:
            if cell.cell_type == 'combiner_dec':
                if idx_dec > 0:
                    param = self.dec_sampler[idx_dec - 1](s)
                    mu_p, log_sig_p = torch.chunk(param, 2, dim=1)
                    ftr = combiner_cells_enc[idx_dec - 1](combiner_cells_s[idx_dec - 1], s)
                    param = self.enc_sampler[idx_dec](ftr)
                    mu_q, log_sig_q = torch.chunk(param, 2, dim=1)
                    dist = Normal(mu_p + mu_q, log_sig_p + log_sig_q) if self.res_dist else Normal(mu_q, log_sig_q)
                    z, _ = dist.sample()
                    log_q_conv = dist.log_p(z)
                    for n in range(self.num_flows):
                        z, log_det = self.nf_cells[nf_offset + n](z, ftr)
                        log_q_conv -= log_det
                    nf_offset += self.num_flows
                    all_log_q.append(log_q_conv)
                    all_q.append(dist)
                    dist = Normal(mu_p, log_sig_p)
                    log_p_conv = dist.log_p(z)
                    all_p.append(dist)
                    all_log_p.append(log_p_conv)
                s = cell(s, z)
                idx_dec += 1
            else:
                s = cell(s)
        if self.vanilla_vae:
            s = self.stem_decoder(z)
        for cell in self.post_process:
            s = cell(s)
        logits = self.image_conditional(s)
        kl_all = []
        kl_diag = []
        log_p, log_q = 0.0, 0.0
        for q, p, log_q_conv, log_p_conv in zip(all_q, all_p, all_log_q, all_log_p):
            if self.with_nf:
                kl_per_var = log_q_conv - log_p_conv
            else:
                kl_per_var = q.kl(p)
            kl_diag.append(torch.mean(torch.sum(kl_per_var, dim=[2, 3]), dim=0))
            kl_all.append(torch.sum(kl_per_var, dim=[1, 2, 3]))
            log_q += torch.sum(log_q_conv, dim=[1, 2, 3])
            log_p += torch.sum(log_p_conv, dim=[1, 2, 3])
        return logits, log_q, log_p, kl_all, kl_diag

    def sample(self, num_samples, t):
        scale_ind = 0
        z0_size = [num_samples] + self.z0_size
        dist = Normal(mu=torch.zeros(z0_size), log_sigma=torch.zeros(z0_size), temp=t)
        z, _ = dist.sample()
        idx_dec = 0
        s = self.prior_ftr0.unsqueeze(0)
        batch_size = z.size(0)
        s = s.expand(batch_size, -1, -1, -1)
        for cell in self.dec_tower:
            if cell.cell_type == 'combiner_dec':
                if idx_dec > 0:
                    param = self.dec_sampler[idx_dec - 1](s)
                    mu, log_sigma = torch.chunk(param, 2, dim=1)
                    dist = Normal(mu, log_sigma, t)
                    z, _ = dist.sample()
                s = cell(s, z)
                idx_dec += 1
            else:
                s = cell(s)
                if cell.cell_type == 'up_dec':
                    scale_ind += 1
        if self.vanilla_vae:
            s = self.stem_decoder(z)
        for cell in self.post_process:
            s = cell(s)
        logits = self.image_conditional(s)
        return logits

    def decoder_output(self, logits):
        if self.dataset in {'mnist', 'omniglot'}:
            return Bernoulli(logits=logits)
        elif self.dataset in {'stacked_mnist', 'cifar10', 'celeba_64', 'celeba_256', 'imagenet_32', 'imagenet_64', 'ffhq', 'lsun_bedroom_128', 'lsun_bedroom_256', 'lsun_church_64', 'lsun_church_128'}:
            if self.num_mix_output == 1:
                return NormalDecoder(logits, num_bits=self.num_bits)
            else:
                return DiscMixLogistic(logits, self.num_mix_output, num_bits=self.num_bits)
        else:
            raise NotImplementedError

    def spectral_norm_parallel(self):
        """ This method computes spectral normalization for all conv layers in parallel. This method should be called
         after calling the forward method of all the conv layers in each iteration. """
        weights = {}
        for l in self.all_conv_layers:
            weight = l.weight_normalized
            weight_mat = weight.view(weight.size(0), -1)
            if weight_mat.shape not in weights:
                weights[weight_mat.shape] = []
            weights[weight_mat.shape].append(weight_mat)
        loss = 0
        for i in weights:
            weights[i] = torch.stack(weights[i], dim=0)
            with torch.no_grad():
                num_iter = self.num_power_iter
                if i not in self.sr_u:
                    num_w, row, col = weights[i].shape
                    self.sr_u[i] = F.normalize(torch.ones(num_w, row).normal_(0, 1), dim=1, eps=0.001)
                    self.sr_v[i] = F.normalize(torch.ones(num_w, col).normal_(0, 1), dim=1, eps=0.001)
                    num_iter = 10 * self.num_power_iter
                for j in range(num_iter):
                    self.sr_v[i] = F.normalize(torch.matmul(self.sr_u[i].unsqueeze(1), weights[i]).squeeze(1), dim=1, eps=0.001)
                    self.sr_u[i] = F.normalize(torch.matmul(weights[i], self.sr_v[i].unsqueeze(2)).squeeze(2), dim=1, eps=0.001)
            sigma = torch.matmul(self.sr_u[i].unsqueeze(1), torch.matmul(weights[i], self.sr_v[i].unsqueeze(2)))
            loss += torch.sum(sigma)
        return loss

    def batchnorm_loss(self):
        loss = 0
        for l in self.all_bn_layers:
            if l.affine:
                loss += torch.max(torch.abs(l.weight))
        return loss


class Swish(torch.autograd.Function):

    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class DummyDDP(nn.Module):

    def __init__(self, model):
        super(DummyDDP, self).__init__()
        self.module = model

    def forward(self, *input, **kwargs):
        return self.module(*input, **kwargs)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ARConv2d,
     lambda: ([], {'C_in': 4, 'C_out': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ARInvertedResidual,
     lambda: ([], {'inz': 4, 'inf': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Conv2D,
     lambda: ([], {'C_in': 4, 'C_out': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (DecCombinerCell,
     lambda: ([], {'Cin1': 4, 'Cin2': 4, 'Cout': 4, 'cell_type': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (DummyDDP,
     lambda: ([], {'model': _mock_layer()}),
     lambda: ([], {'input': torch.rand([4, 4])}),
     False),
    (EncCombinerCell,
     lambda: ([], {'Cin1': 4, 'Cin2': 4, 'Cout': 4, 'cell_type': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (FIDInceptionA,
     lambda: ([], {'in_channels': 4, 'pool_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FIDInceptionC,
     lambda: ([], {'in_channels': 4, 'channels_7x7': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FIDInceptionE_1,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FIDInceptionE_2,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Identity,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SE,
     lambda: ([], {'Cin': 4, 'Cout': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (UpSample,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_NVlabs_NVAE(_paritybench_base):
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

