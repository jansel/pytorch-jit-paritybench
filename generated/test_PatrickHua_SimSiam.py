import sys
_module = sys.modules[__name__]
del sys
arguments = _module
augmentations = _module
byol_aug = _module
eval_aug = _module
gaussian_blur = _module
simclr_aug = _module
simsiam_aug = _module
swav_aug = _module
configs = _module
datasets = _module
random_dataset = _module
linear_eval = _module
main = _module
models = _module
backbones = _module
cifar_resnet_1 = _module
cifar_resnet_2 = _module
byol = _module
simclr = _module
simsiam = _module
optimizers = _module
larc = _module
lars = _module
lars_simclr = _module
lr_scheduler = _module
tools = _module
accuracy = _module
average_meter = _module
file_exist_fn = _module
knn_monitor = _module
logger = _module
plotter = _module

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


import numpy as np


import random


import re


import warnings


from torch import Tensor


from torchvision.transforms.functional import to_pil_image


from torchvision.transforms.functional import to_tensor


from torch.nn.functional import conv2d


from torch.nn.functional import pad as torch_pad


from typing import Any


from typing import List


from typing import Sequence


from typing import Optional


import numbers


from typing import Tuple


import torchvision.transforms as T


import torchvision


import torch.nn as nn


import torch.nn.functional as F


from torchvision.models import resnet50


from torchvision.models import resnet18


import copy


from torch import nn


from torchvision import transforms


from math import pi


from math import cos


from collections import OrderedDict


from torch.nn.parameter import Parameter


from torch.optim.optimizer import Optimizer


from torch.optim.optimizer import required


import matplotlib


import matplotlib.pyplot as plt


def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)
    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]
    if len(size) != 2:
        raise ValueError(error_msg)
    return size


def _cast_squeeze_in(img: Tensor, req_dtype: torch.dtype) ->Tuple[Tensor, bool, bool, torch.dtype]:
    need_squeeze = False
    if img.ndim < 4:
        img = img.unsqueeze(dim=0)
        need_squeeze = True
    out_dtype = img.dtype
    need_cast = False
    if out_dtype != req_dtype:
        need_cast = True
        img = img
    return img, need_cast, need_squeeze, out_dtype


def _cast_squeeze_out(img: Tensor, need_cast: bool, need_squeeze: bool, out_dtype: torch.dtype):
    if need_squeeze:
        img = img.squeeze(dim=0)
    if need_cast:
        img = torch.round(img)
    return img


def _get_gaussian_kernel1d(kernel_size: int, sigma: float) ->Tensor:
    ksize_half = (kernel_size - 1) * 0.5
    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
    pdf = torch.exp(-0.5 * (x / sigma).pow(2))
    kernel1d = pdf / pdf.sum()
    return kernel1d


def _get_gaussian_kernel2d(kernel_size: List[int], sigma: List[float], dtype: torch.dtype, device: torch.device) ->Tensor:
    kernel1d_x = _get_gaussian_kernel1d(kernel_size[0], sigma[0])
    kernel1d_y = _get_gaussian_kernel1d(kernel_size[1], sigma[1])
    kernel2d = torch.mm(kernel1d_y[:, None], kernel1d_x[None, :])
    return kernel2d


def _is_tensor_a_torch_image(x: Tensor) ->bool:
    return x.ndim >= 2


def _gaussian_blur(img: Tensor, kernel_size: List[int], sigma: List[float]) ->Tensor:
    """PRIVATE METHOD. Performs Gaussian blurring on the img by given kernel.

    .. warning::

        Module ``transforms.functional_tensor`` is private and should not be used in user application.
        Please, consider instead using methods from `transforms.functional` module.

    Args:
        img (Tensor): Image to be blurred
        kernel_size (sequence of int or int): Kernel size of the Gaussian kernel ``(kx, ky)``.
        sigma (sequence of float or float, optional): Standard deviation of the Gaussian kernel ``(sx, sy)``.

    Returns:
        Tensor: An image that is blurred using gaussian kernel of given parameters
    """
    if not (isinstance(img, torch.Tensor) or _is_tensor_a_torch_image(img)):
        raise TypeError('img should be Tensor Image. Got {}'.format(type(img)))
    dtype = img.dtype if torch.is_floating_point(img) else torch.float32
    kernel = _get_gaussian_kernel2d(kernel_size, sigma, dtype=dtype, device=img.device)
    kernel = kernel.expand(img.shape[-3], 1, kernel.shape[0], kernel.shape[1])
    img, need_cast, need_squeeze, out_dtype = _cast_squeeze_in(img, kernel.dtype)
    padding = [kernel_size[0] // 2, kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[1] // 2]
    img = torch_pad(img, padding, mode='reflect')
    img = conv2d(img, kernel, groups=img.shape[-3])
    img = _cast_squeeze_out(img, need_cast, need_squeeze, out_dtype)
    return img


@torch.jit.unused
def _is_pil_image(img: Any) ->bool:
    return isinstance(img, Image.Image)


def gaussian_blur(img: Tensor, kernel_size: List[int], sigma: Optional[List[float]]=None) ->Tensor:
    """Performs Gaussian blurring on the img by given kernel.
    The image can be a PIL Image or a Tensor, in which case it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

    Args:
        img (PIL Image or Tensor): Image to be blurred
        kernel_size (sequence of ints or int): Gaussian kernel size. Can be a sequence of integers
            like ``(kx, ky)`` or a single integer for square kernels.
            In torchscript mode kernel_size as single int is not supported, use a tuple or
            list of length 1: ``[ksize, ]``.
        sigma (sequence of floats or float, optional): Gaussian kernel standard deviation. Can be a
            sequence of floats like ``(sigma_x, sigma_y)`` or a single float to define the
            same sigma in both X/Y directions. If None, then it is computed using
            ``kernel_size`` as ``sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8``.
            Default, None. In torchscript mode sigma as single float is
            not supported, use a tuple or list of length 1: ``[sigma, ]``.

    Returns:
        PIL Image or Tensor: Gaussian Blurred version of the image.
    """
    if not isinstance(kernel_size, (int, list, tuple)):
        raise TypeError('kernel_size should be int or a sequence of integers. Got {}'.format(type(kernel_size)))
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size, kernel_size]
    if len(kernel_size) != 2:
        raise ValueError('If kernel_size is a sequence its length should be 2. Got {}'.format(len(kernel_size)))
    for ksize in kernel_size:
        if ksize % 2 == 0 or ksize < 0:
            raise ValueError('kernel_size should have odd and positive integers. Got {}'.format(kernel_size))
    if sigma is None:
        sigma = [(ksize * 0.15 + 0.35) for ksize in kernel_size]
    if sigma is not None and not isinstance(sigma, (int, float, list, tuple)):
        raise TypeError('sigma should be either float or sequence of floats. Got {}'.format(type(sigma)))
    if isinstance(sigma, (int, float)):
        sigma = [float(sigma), float(sigma)]
    if isinstance(sigma, (list, tuple)) and len(sigma) == 1:
        sigma = [sigma[0], sigma[0]]
    if len(sigma) != 2:
        raise ValueError('If sigma is a sequence, its length should be 2. Got {}'.format(len(sigma)))
    for s in sigma:
        if s <= 0.0:
            raise ValueError('sigma should have positive values. Got {}'.format(sigma))
    t_img = img
    if not isinstance(img, torch.Tensor):
        if not _is_pil_image(img):
            raise TypeError('img should be PIL Image or Tensor. Got {}'.format(type(img)))
        t_img = to_tensor(img)
    output = _gaussian_blur(t_img, kernel_size, sigma)
    if not isinstance(img, torch.Tensor):
        output = to_pil_image(output)
    return output


class GaussianBlur(torch.nn.Module):
    """Blurs image with randomly chosen Gaussian blur.
    The image can be a PIL Image or a Tensor, in which case it is expected
    to have [..., C, H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        kernel_size (int or sequence): Size of the Gaussian kernel.
        sigma (float or tuple of float (min, max)): Standard deviation to be used for
            creating kernel to perform blurring. If float, sigma is fixed. If it is tuple
            of float (min, max), sigma is chosen uniformly at random to lie in the
            given range.

    Returns:
        PIL Image or Tensor: Gaussian blurred version of the input image.

    """

    def __init__(self, kernel_size, sigma=(0.1, 2.0)):
        super().__init__()
        self.kernel_size = _setup_size(kernel_size, 'Kernel size should be a tuple/list of two integers')
        for ks in self.kernel_size:
            if ks <= 0 or ks % 2 == 0:
                raise ValueError('Kernel size value should be an odd and positive number.')
        if isinstance(sigma, numbers.Number):
            if sigma <= 0:
                raise ValueError('If sigma is a single number, it must be positive.')
            sigma = sigma, sigma
        elif isinstance(sigma, Sequence) and len(sigma) == 2:
            if not 0.0 < sigma[0] <= sigma[1]:
                raise ValueError('sigma values should be positive and of the form (min, max).')
        else:
            raise ValueError('sigma should be a single number or a list/tuple with length 2.')
        self.sigma = sigma

    @staticmethod
    def get_params(sigma_min: float, sigma_max: float) ->float:
        """Choose sigma for random gaussian blurring.

        Args:
            sigma_min (float): Minimum standard deviation that can be chosen for blurring kernel.
            sigma_max (float): Maximum standard deviation that can be chosen for blurring kernel.

        Returns:
            float: Standard deviation to be passed to calculate kernel for gaussian blurring.
        """
        return torch.empty(1).uniform_(sigma_min, sigma_max).item()

    def forward(self, img: Tensor) ->Tensor:
        """
        Args:
            img (PIL Image or Tensor): image to be blurred.

        Returns:
            PIL Image or Tensor: Gaussian blurred image
        """
        sigma = self.get_params(self.sigma[0], self.sigma[1])
        return gaussian_blur(img, self.kernel_size, [sigma, sigma])

    def __repr__(self):
        s = '(kernel_size={}, '.format(self.kernel_size)
        s += 'sigma={})'.format(self.sigma)
        return self.__class__.__name__ + s


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


HPS = dict(max_steps=int(1000.0 * 1281167 / 4096), mlp_hidden_size=4096, projection_size=256, base_target_ema=0.004, optimizer_config=dict(optimizer_name='lars', beta=0.9, trust_coef=0.001, weight_decay=1.5e-06, exclude_bias_from_adaption=True), learning_rate_schedule=dict(base_learning_rate=0.2, warmup_steps=int(10.0 * 1281167 / 4096), anneal_schedule='cosine'), batchnorm_kwargs=dict(decay_rate=0.9, eps=1e-05), seed=1337)


class MLP(nn.Module):

    def __init__(self, in_dim):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, HPS['mlp_hidden_size']), nn.BatchNorm1d(HPS['mlp_hidden_size'], eps=HPS['batchnorm_kwargs']['eps'], momentum=1 - HPS['batchnorm_kwargs']['decay_rate']), nn.ReLU(inplace=True))
        self.layer2 = nn.Linear(HPS['mlp_hidden_size'], HPS['projection_size'])

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


def D(p, z, version='simplified'):
    if version == 'original':
        z = z.detach()
        p = F.normalize(p, dim=1)
        z = F.normalize(z, dim=1)
        return -(p * z).sum(dim=1).mean()
    elif version == 'simplified':
        return -F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        raise Exception


class BYOL(nn.Module):

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.projector = MLP(backbone.output_dim)
        self.online_encoder = nn.Sequential(self.backbone, self.projector)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.online_predictor = MLP(HPS['projection_size'])
        raise NotImplementedError('Please put update_moving_average to training')

    def target_ema(self, k, K, base_ema=HPS['base_target_ema']):
        return 1 - base_ema * (cos(pi * k / K) + 1) / 2

    @torch.no_grad()
    def update_moving_average(self, global_step, max_steps):
        tau = self.target_ema(global_step, max_steps)
        for online, target in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target.data = tau * target.data + (1 - tau) * online.data

    def forward(self, x1, x2):
        f_o, h_o = self.online_encoder, self.online_predictor
        f_t = self.target_encoder
        z1_o = f_o(x1)
        z2_o = f_o(x2)
        p1_o = h_o(z1_o)
        p2_o = h_o(z2_o)
        with torch.no_grad():
            z1_t = f_t(x1)
            z2_t = f_t(x2)
        L = D(p1_o, z2_t) / 2 + D(p2_o, z1_t) / 2
        return {'loss': L}


class projection_MLP(nn.Module):

    def __init__(self, in_dim, hidden_dim=2048, out_dim=2048):
        super().__init__()
        """ page 3 baseline setting
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out- 
        put fc. Its output fc has no ReLU. The hidden fc is 2048-d. 
        This MLP has 3 layers.
        """
        self.layer1 = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True))
        self.layer2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True))
        self.layer3 = nn.Sequential(nn.Linear(hidden_dim, out_dim), nn.BatchNorm1d(hidden_dim))
        self.num_layers = 3

    def set_layers(self, num_layers):
        self.num_layers = num_layers

    def forward(self, x):
        if self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        else:
            raise Exception
        return x


def NT_XentLoss(z1, z2, temperature=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    N, Z = z1.shape
    device = z1.device
    representations = torch.cat([z1, z2], dim=0)
    similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=-1)
    l_pos = torch.diag(similarity_matrix, N)
    r_pos = torch.diag(similarity_matrix, -N)
    positives = torch.cat([l_pos, r_pos]).view(2 * N, 1)
    diag = torch.eye(2 * N, dtype=torch.bool, device=device)
    diag[N:, :N] = diag[:N, N:] = diag[:N, :N]
    negatives = similarity_matrix[~diag].view(2 * N, -1)
    logits = torch.cat([positives, negatives], dim=1)
    logits /= temperature
    labels = torch.zeros(2 * N, device=device, dtype=torch.int64)
    loss = F.cross_entropy(logits, labels, reduction='sum')
    return loss / (2 * N)


class SimCLR(nn.Module):

    def __init__(self, backbone=resnet50()):
        super().__init__()
        self.backbone = backbone
        self.projector = projection_MLP(backbone.output_dim)
        self.encoder = nn.Sequential(self.backbone, self.projector)

    def forward(self, x1, x2):
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        loss = NT_XentLoss(z1, z2)
        return {'loss': loss}


class prediction_MLP(nn.Module):

    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048):
        super().__init__()
        """ page 3 baseline setting
        Prediction MLP. The prediction MLP (h) has BN applied 
        to its hidden fc layers. Its output fc does not have BN
        (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers. 
        The dimension of h’s input and output (z and p) is d = 2048, 
        and h’s hidden layer’s dimension is 512, making h a 
        bottleneck structure (ablation in supplement). 
        """
        self.layer1 = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True))
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        """
        Adding BN to the output of the prediction MLP h does not work
        well (Table 3d). We find that this is not about collapsing. 
        The training is unstable and the loss oscillates.
        """

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class SimSiam(nn.Module):

    def __init__(self, backbone=resnet50()):
        super().__init__()
        self.backbone = backbone
        self.projector = projection_MLP(backbone.output_dim)
        self.encoder = nn.Sequential(self.backbone, self.projector)
        self.predictor = prediction_MLP()

    def forward(self, x1, x2):
        f, h = self.encoder, self.predictor
        z1, z2 = f(x1), f(x2)
        p1, p2 = h(z1), h(z2)
        L = D(p1, z2) / 2 + D(p2, z1) / 2
        return {'loss': L}


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'in_planes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Bottleneck,
     lambda: ([], {'in_planes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MLP,
     lambda: ([], {'in_dim': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (projection_MLP,
     lambda: ([], {'in_dim': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
]

class Test_PatrickHua_SimSiam(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

