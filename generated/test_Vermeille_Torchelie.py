import sys
_module = sys.modules[__name__]
del sys
conf = _module
cifar = _module
conditional = _module
gan = _module
imagenet = _module
pixelcnn = _module
setup = _module
test_datalearning = _module
test_datasets = _module
test_hyper = _module
test_loss = _module
test_models = _module
test_nn = _module
test_optims = _module
test_recipes = _module
test_sched = _module
test_tensorboard_callback = _module
test_tranforms = _module
test_utils = _module
torchelie = _module
callbacks = _module
avg = _module
callbacks = _module
inspector = _module
data_learning = _module
datasets = _module
concat = _module
debug = _module
ms1m = _module
pix2pix = _module
distributions = _module
hyper = _module
loss = _module
bitempered = _module
deepdreamloss = _module
face_rec = _module
focal = _module
functional = _module
hinge = _module
ls = _module
penalty = _module
standard = _module
neuralstyleloss = _module
perceptualloss = _module
lr_scheduler = _module
models = _module
alexnet = _module
attention = _module
autogan = _module
classifier = _module
convnext = _module
efficient = _module
hourglass = _module
mlpmixer = _module
patchgan = _module
perceptualnet = _module
pix2pix = _module
pix2pixhd = _module
pixcnn = _module
registry = _module
resnet = _module
snres_discr = _module
stylegan2 = _module
unet = _module
vgg = _module
nn = _module
adain = _module
batchnorm = _module
blocks = _module
condseq = _module
conv = _module
debug = _module
encdec = _module
functional = _module
transformer = _module
vq = _module
graph = _module
imagenetinputnorm = _module
interpolate = _module
layers = _module
maskedconv = _module
noise = _module
pixelnorm = _module
resblock = _module
reshape = _module
transformer = _module
utils = _module
vq = _module
withsavedactivations = _module
optim = _module
recipes = _module
algorithm = _module
classification = _module
cut = _module
deepdream = _module
feature_vis = _module
gan = _module
image_prior = _module
neural_style = _module
pix2pix = _module
recipebase = _module
stylegan2 = _module
trainandcall = _module
trainandtest = _module
unpaired = _module
serving_utils = _module
transforms = _module
augments = _module
differentiable = _module
randaugment = _module
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


import torch


import torch.nn as nn


import torch.nn.functional as F


from torchvision.datasets import CIFAR10


import torchvision.transforms as TF


from torchvision.datasets import MNIST


import copy


from torchvision.datasets import FashionMNIST


from torchvision.datasets import SVHN


from torch.utils.data import DataLoader


from torchvision.transforms import ToPILImage


from torch.optim import SGD


import matplotlib.pyplot as plt


from torchvision.transforms import PILToTensor


import time


import numpy as np


from typing import Union


from typing import List


from typing import Optional


from typing import Tuple


import random


from typing import Sequence


from torchvision.datasets import ImageFolder


from torch.utils.data import Dataset


from typing import Any


from typing import Generic


from typing import TypeVar


from typing import overload


from typing import Callable


from torchvision.datasets.utils import download_and_extract_archive


from torchvision.transforms.functional import to_tensor


import math


from torch.distributions import TransformedDistribution


from typing import Dict


from typing import cast


from torch.optim import Optimizer


from torch.optim.lr_scheduler import _LRScheduler


from collections import OrderedDict


import collections


from torch.autograd import Function


from torch import Tensor


import warnings


from torch.nn import Module


from torch.nn.parameter import Parameter


import functools


from collections import defaultdict


from typing import Iterable


import torchvision.models as tvmodels


from torch.cuda.amp import autocast


import torchvision.transforms.functional as TFF


from torchvision.transforms.functional import to_pil_image


import torchvision.transforms.functional as F


from torchvision.transforms import functional as F


from torchvision.transforms import InterpolationMode


from enum import Enum


import torch.distributed as dist


from functools import wraps


from inspect import isfunction


class LearnableImage(nn.Module):

    def init_img(self, init: torch.Tensor) ->None:
        raise NotImplementedError


class ColorTransform(nn.Module):

    def __call__(self, img: torch.Tensor) ->torch.Tensor:
        raise NotImplementedError

    def invert(self, t: torch.Tensor) ->torch.Tensor:
        raise NotImplementedError


class PixelImage(LearnableImage):
    """
    A learnable image parameterized by its pixel values

    Args:
        shape (tuple of int): a tuple like (channels, height, width)
        sd (float): pixels are initialized with a random gaussian value of mean
            0.5 and standard deviation `sd`
        init_img (tensor, optional): an image tensor to initialize the pixel
            values
    """
    shape: Tuple[int, ...]
    pixels: torch.Tensor

    def __init__(self, shape: Tuple[int, ...], sd: float=0.01, init_img: torch.Tensor=None) ->None:
        super(PixelImage, self).__init__()
        self.shape = shape
        n, ch, h, w = shape
        self.pixels = torch.nn.Parameter(sd * torch.randn(n, ch, h, w), requires_grad=True)
        if init_img is not None:
            self.init_img(init_img)

    def init_img(self, init_img):
        self.pixels.data.copy_(init_img - 0.5)

    def forward(self):
        """
        Return the image
        """
        return self.pixels


def _rfft2d_freqs(h, w):
    fy = torch.fft.fftfreq(h)[:, None]
    fx = torch.fft.fftfreq(w)[:w // 2 + (2 if w % 2 == 1 else 1)]
    return np.sqrt(fx * fx + fy * fy)


class SpectralImage(LearnableImage):
    """
    A learnable image parameterized by its Fourier representation.

    See https://distill.pub/2018/differentiable-parameterizations/

    Implementation ported from
    https://github.com/tensorflow/lucid/blob/master/lucid/optvis/param/spatial.py

    Args:
        shape (tuple of int): a tuple like (channels, height, width)
        sd (float): amplitudes are initialized with a random gaussian value of
            mean 0.5 and standard deviation `sd`
        init_img (tensor, optional): an image tensor to initialize the image
    """
    shape: Tuple[int, ...]
    decay_power: float
    spectrum_var: torch.Tensor
    spertum_scale: torch.Tensor

    def __init__(self, shape: Tuple[int, ...], sd: float=0.01, decay_power: int=1, init_img: torch.Tensor=None) ->None:
        super(SpectralImage, self).__init__()
        self.shape = shape
        n, ch, h, w = shape
        freqs = _rfft2d_freqs(h, w)
        fh, fw = freqs.shape
        self.decay_power = decay_power
        init_val = sd * torch.randn(n, ch, fh, fw, dtype=torch.complex64)
        spectrum_var = torch.nn.Parameter(init_val)
        self.spectrum_var = spectrum_var
        spectrum_scale = 1.0 / np.maximum(freqs, 1.0 / max(h, w)) ** self.decay_power
        self.register_buffer('spectrum_scale', spectrum_scale)
        if init_img is not None:
            self.init_img(init_img)

    def init_img(self, init_img: torch.Tensor) ->None:
        assert init_img.dim() == 4 and init_img.shape == self.shape
        fft = torch.fft.rfft2(init_img[0] * 4, s=(self.shape[2], self.shape[3]), norm='ortho')
        with torch.no_grad():
            self.spectrum_var.copy_(fft / self.spectrum_scale)

    def forward(self) ->torch.Tensor:
        """
        Return the image
        """
        n, ch, h, w = self.shape
        scaled_spectrum = self.spectrum_var * self.spectrum_scale
        img = torch.fft.irfft2(scaled_spectrum, norm='ortho', s=(h, w))
        return img / 4


class CorrelateColors(ColorTransform):
    """
    Takes an learnable image and applies the inverse color decorrelation from
    ImageNet (ie, it correlates the color like ImageNet to ease optimization)
    """
    color_correlation: torch.Tensor

    def __init__(self) ->None:
        super(CorrelateColors, self).__init__()
        color_correlation_svd_sqrt = torch.tensor([[0.26, 0.09, 0.02], [0.27, 0.0, -0.05], [0.27, -0.09, 0.03]])
        max_norm_svd_sqrt = float(np.max(np.linalg.norm(color_correlation_svd_sqrt, axis=0)))
        cc = color_correlation_svd_sqrt / max_norm_svd_sqrt
        self.register_buffer('color_correlation', cc)

    def __call__(self, img: torch.Tensor) ->torch.Tensor:
        """
        Correlate the color of the image `img` and return the result
        """
        t_flat = img.view(img.shape[0], 3, -1).transpose(2, 1)
        t_flat = torch.matmul(t_flat, self.color_correlation.t())
        t = t_flat.transpose(2, 1).view(img.shape)
        return t

    def invert(self, t: torch.Tensor) ->torch.Tensor:
        """
        Decorrelate the color of the image `t` and return the result
        """
        t_flat = t.view(t.shape[0], 3, -1).transpose(2, 1)
        t_flat = torch.matmul(t_flat, self.color_correlation.inverse().t()[None])
        t = t_flat.transpose(2, 1).reshape(*t.shape)
        return t


class RGB(ColorTransform):

    def __call__(self, x: torch.Tensor) ->torch.Tensor:
        return x

    def invert(self, x: torch.Tensor) ->torch.Tensor:
        return x


def ortho(w: torch.Tensor) ->torch.Tensor:
    """
    Returns the orthogonal loss for weight matrix `m`, from Big GAN.

    https://arxiv.org/abs/1809.11096

    :math:`R_{\\beta}(W)= ||W^T W  \\odot (1 - I)||_F^2`
    """
    cosine = torch.einsum('ij,ji->ij', w, w)
    no_diag = 1 - torch.eye(w.shape[0], device=w.device)
    return (cosine * no_diag).pow(2).sum(dim=1).mean()


class OrthoLoss(nn.Module):
    """
    Orthogonal loss

    See :func:`torchelie.loss.ortho` for details.
    """

    def forward(self, w):
        return ortho(w)


def total_variation(i: torch.Tensor) ->torch.Tensor:
    """
    Returns the total variation loss for batch of images `i`
    """
    v = F.l1_loss(i[:, :, 1:, :], i[:, :, :-1, :])
    h = F.l1_loss(i[:, :, :, 1:], i[:, :, :, :-1])
    return v + h


class TotalVariationLoss(nn.Module):
    """
    Total Variation loss

    See :func:`torchelie.loss.total_variation` for details.
    """

    def forward(self, x):
        return total_variation(x)


def continuous_cross_entropy(pred: torch.Tensor, soft_targets: torch.Tensor, weights: Optional[torch.Tensor]=None, reduction: str='mean') ->torch.Tensor:
    """
    Compute the cross entropy between the logits `pred` and a normalized
    distribution `soft_targets`. If `soft_targets` is a one-hot vector, this is
    equivalent to `nn.functional.cross_entropy` with a label
    """
    if weights is None:
        ce = torch.sum(-soft_targets * F.log_softmax(pred, 1), 1)
    else:
        ce = torch.sum(-weights * soft_targets * F.log_softmax(pred, 1), 1)
    if reduction == 'mean':
        return ce.mean()
    if reduction == 'sum':
        return ce.sum()
    if reduction == 'none':
        return ce
    assert False, f'{reduction} not a valid reduction method'


class ContinuousCEWithLogits(nn.Module):
    """
    Cross Entropy loss accepting continuous target values

    See :func:`torchelie.loss.continuous_cross_entropy` for details.
    """

    def forward(self, pred, soft_targets):
        return continuous_cross_entropy(pred, soft_targets)


def exp_t(x: torch.Tensor, t: float) ->torch.Tensor:
    if t == 1:
        return torch.exp(x)
    return torch.clamp(1 + (1 - t) * x, min=0) ** (1 / (1 - t))


def log_t(x: torch.Tensor, t: float) ->torch.Tensor:
    if t == 1:
        return torch.log(x)
    return (x ** (1 - t) - 1) / (1 - t)


def lambdas(a: torch.Tensor, t: float, n_iters: int=3) ->torch.Tensor:
    mu = torch.max(a, dim=1, keepdim=True).values
    a_tilde = a - mu
    for i in range(n_iters):
        za = exp_t(a_tilde, t).sum(1, keepdim=True)
        a_tilde = za ** (1 - t) * (a - mu)
    return -log_t(1 / za, t) + mu


def tempered_log_softmax(x: torch.Tensor, t: float, n_iters: int=3) ->torch.Tensor:
    """
    Tempered log softmax. Computes log softmax along dimension 1

    Args:
        x (tensor): activations
        t (float): temperature
        n_iters (int): number of iters to converge (default: 3

    Returns:
        result of tempered log softmax
    """
    return x - lambdas(x, t, n_iters=n_iters)


def tempered_nll_loss(x: torch.Tensor, y: torch.Tensor, t1: float, t2: float, weight: Optional[torch.Tensor]=None, reduction: str='mean') ->torch.Tensor:
    """
    Compute tempered nll loss

    Args:
        x (tensor): activations of log softmax
        y (tensor): labels
        t1 (float): temperature 1
        t2 (float): temperature 2
        weight (tensor): a tensor that associates a weight to each class
        reduction (str): how to reduce the batch of losses: 'none', 'sum', or
            'mean'
    Returns:
        the loss
    """
    x = exp_t(x, t2)
    y_hat = x[torch.arange(0, x.shape[0]).long(), y]
    out = -log_t(y_hat, t1) - (1 - torch.sum(x ** (2 - t1), dim=1)) / (2 - t1)
    if weight is not None:
        out = weight[y] * out
    if reduction == 'none':
        return out
    if reduction == 'mean' and weight is not None:
        return torch.sum(out / weight[y].sum())
    if reduction == 'mean' and weight is None:
        return torch.sum(out / out.shape[0])
    if reduction == 'sum':
        return out.sum()
    assert False, f'{reduction} not a valid reduction method'


def tempered_cross_entropy(x: torch.Tensor, y: torch.Tensor, t1: float, t2: float, n_iters: int=3, weight: Optional[torch.Tensor]=None, reduction: str='mean') ->torch.Tensor:
    """
    The bi-tempered loss from https://arxiv.org/abs/1906.03361

    Args:
        x (tensor): a tensor of batched logits like for cross_entropy
        y (tensor): a tensor of labels
        t1 (float): temperature 1
        t2 (float): temperature 2
        weight (tensor): a tensor that associates a weight to each class
        reduction (str): how to reduce the batch of losses: 'none', 'sum', or
            'mean'

    Returns:
        the loss
    """
    sm = tempered_log_softmax(x, t2, n_iters=n_iters)
    return tempered_nll_loss(sm, y, t1, t2, weight=weight, reduction=reduction)


class TemperedCrossEntropyLoss(torch.nn.Module):
    """
    The bi-tempered loss from https://arxiv.org/abs/1906.03361

    Args:
        t1 (float): temperature 1
        t2 (float): temperature 2
        weight (tensor): a tensor that associates a weight to each class
        reduction (str): how to reduce the batch of losses: 'none', 'sum', or
            'mean'
    """

    def __init__(self, t1, t2, weight=None, reduction='mean'):
        super(TemperedCrossEntropyLoss, self).__init__()
        self.t1 = t1
        self.t2 = t2
        self.weight = weight
        self.reduction = reduction

    def forward(self, x, y):
        """
        Forward pass

        Args:
            x (tensor): a tensor of batched logits like for
                cross_entropy
            y (tensor): a tensor of labels

        Returns:
            the loss
        """
        return tempered_cross_entropy(x, y, self.t1, self.t2, weight=self.weight, reduction=self.reduction)


class DeepDreamLoss(nn.Module):
    """
    The Deep Dream loss

    Args:
        model (nn.Module): a pretrained network on which to compute the
            activations
        dream_layer (str): the name of the layer on which the activations are
            to be maximized
        max_reduction (int): the maximum factor of reduction of the image, for
            multiscale generation
    """

    def __init__(self, model: nn.Module, dream_layer: str, max_reduction: int=3) ->None:
        super(DeepDreamLoss, self).__init__()
        self.dream_layer = dream_layer
        self.octaves = max_reduction
        model = model.eval()
        self.net = tnn.WithSavedActivations(model, names=[self.dream_layer])
        self.i = 0

    def get_acts_(self, img: torch.Tensor, detach: bool) ->torch.Tensor:
        octave = self.i % (self.octaves * 2) / 2 + 1
        this_sz_img = F.interpolate(img, scale_factor=1 / octave)
        _, activations = self.net(this_sz_img, detach=detach)
        return activations[self.dream_layer]

    def forward(self, input_img: torch.Tensor) ->torch.Tensor:
        """
        Compute the Deep Dream loss on `input_img`
        """
        dream = self.get_acts_(input_img, detach=False)
        self.i += 1
        dream_loss = -dream.pow(2).sum()
        return dream_loss


class L2Constraint(nn.Module):
    """
    From `Ranjan 2017`_ , L2-constrained Softmax Loss for Discriminative Face
    Verification.

    Args:
        dim (int): number of channels of the feature vector
        num_classes (int): number of identities
        fixed (bool): whether to use the fixed or dynamic version of AdaCos
            (default: False)

    :: _Ranjan 2017: https://arxiv.org/abs/1703.09507
    """

    def __init__(self, dim: int, num_classes: int, s: float=30.0):
        super(L2Constraint, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, dim))
        nn.init.orthogonal_(self.weight)
        self.num_classes = num_classes
        self.s = s

    def forward(self, input: torch.Tensor, label: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            input (tensor): feature vectors
            label (tensor): labels

        Returns:
            scaled cosine logits, cosine logits
        """
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        output = cosine * self.s
        return output, cosine

    def __repr__(self):
        return 'L2CrossEntropy(s={})'.format(self.s)


class AdaCos(nn.Module):
    """
    From AdaCos: [Adaptively Scaling Cosine Logits for Effectively Learning
    Deep Face Representations](https://arxiv.org/abs/1905.00292)

    Args:
        dim (int): number of channels of the feature vector
        num_classes (int): number of identities
        fixed (bool): whether to use the fixed or dynamic version of AdaCos
            (default: False)
        estimate_B (bool): is using dynamic AdaCos, B is estimated from the
            real angles of the cosine similarity. However I found that this
            method was not numerically stable and experimented with the
            approximation :code:`B = num_classes - 1` that was more satisfying.
    """
    s: torch.Tensor

    def __init__(self, dim: int, num_classes: int, fixed: bool=False, estimate_B: bool=False):
        super(AdaCos, self).__init__()
        self.fixed = fixed
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, dim))
        nn.init.xavier_normal_(self.weight)
        self.num_classes = num_classes
        self.register_buffer('s', torch.tensor(math.sqrt(2) * math.log(num_classes - 1)))
        self.register_buffer('B', torch.tensor(num_classes - 1.0))
        self.estimate_B = estimate_B

    def forward(self, input: torch.Tensor, label: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            input (tensor): feature vectors
            label (tensor): labels

        Returns:
            scaled cosine logits, cosine logits
        """
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        if self.fixed:
            return cosine * self.s, cosine
        if self.training:
            with torch.no_grad():
                correct_cos = torch.gather(cosine, 1, label.unsqueeze(1))
                theta_med = torch.acos(correct_cos).median()
                if self.estimate_B:
                    output = cosine * self.s
                    expout = output.exp()
                    correct_expout = torch.gather(expout, 1, label.unsqueeze(1))
                    correct_expout.squeeze_()
                    self.B = torch.mean(expout.sum(1) - correct_expout, dim=0)
                self.s = torch.log(self.B) / math.cos(min(math.pi / 4, theta_med.item()))
        return cosine * self.s, cosine

    def __repr__(self) ->str:
        return 'FixedAdaCos(fixed={})'.format(self.fixed)


def experimental(func):
    """
    Decorator that warns about a function being experimental
    """
    msg = f'{func.__qualname__}() is experimental, and may change or be deleted soon if not already broken'

    def deprecate_doc(doc):
        if doc is None:
            return f'**Experimental**\n\n.. warning::\n  {msg}\n\n.\n'
        else:
            return '**Experimental**: ' + dedent(func.__doc__) + f'.. warning::\n {msg}\n\n\n'
    if isfunction(func):
        func.__doc__ = deprecate_doc(func.__doc__)

        @wraps(func)
        def wrapped(*args, **kwargs):
            warnings.warn(msg, FutureWarning, stacklevel=2)
            return func(*args, **kwargs)
        return wrapped
    else:
        cls = func

        def __getstate__(self):
            return super(cls, self).__getstate__()

        def __setstate__(self, state):
            return super(cls, self).__setstate__(state)
        d = {'__doc__': deprecate_doc(cls.__doc__), '__init__': cls.__init__, '__module__': cls.__module__}
        return type(cls.__name__, (cls,), d)


class FocalLoss(nn.Module):
    """
    The focal loss

    https://arxiv.org/abs/1708.02002

    See :func:`torchelie.loss.focal_loss` for details.
    """

    def __init__(self, gamma: float=0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, input: torch.Tensor, target: torch.Tensor) ->torch.Tensor:
        return focal_loss(input, target, self.gamma)


class ImageNetInputNorm(nn.Module):
    """
    Normalize images channels as torchvision models expects, in a
    differentiable way
    """

    def __init__(self):
        super(ImageNetInputNorm, self).__init__()
        self.register_buffer('norm_mean', torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1))
        self.register_buffer('norm_std', torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1))

    def forward(self, input):
        return (input - self.norm_mean) / self.norm_std

    def inverse(self, input):
        return input * self.norm_std + self.norm_mean


def layer_by_name(net: nn.Module, name: str) ->Optional[nn.Module]:
    """
    Get a submodule at any depth of a net by its name

    Args:
        net (nn.Module): the base module containing other modules
        name (str): a name of a submodule of `net`, like `"layer3.0.conv1"`.

    Returns:
        The found layer or `None`
    """
    for layer in net.named_modules():
        if layer[0] == name:
            return layer[1]
    return None


class WithSavedActivations(nn.Module):
    """
    Hook :code:`model` in order to get intermediate activations. The
    activations to save can be either specified by module type or layer name.
    """

    def __init__(self, model, types=(nn.Conv2d, nn.Linear), names=None):
        super(WithSavedActivations, self).__init__()
        self.model = model
        self.activations = {}
        self.detach = True
        self.handles = []
        self.set_keep_layers(types, names)

    def set_keep_layers(self, types=(nn.Conv2d, nn.Linear), names=None):
        for h in self.handles:
            h.remove()
        if names is None:
            for name, layer in self.model.named_modules():
                if isinstance(layer, types):
                    h = layer.register_forward_hook(functools.partial(self._save, name))
                    self.handles.append(h)
        else:
            for name in names:
                layer = layer_by_name(self.model, name)
                h = layer.register_forward_hook(functools.partial(self._save, name))
                self.handles.append(h)

    def _save(self, name, module, input, output):
        if self.detach:
            self.activations[name] = output.detach().clone()
        else:
            self.activations[name] = output.clone()

    def forward(self, input, detach: bool):
        """
        Call :code:`self.model(input)`.

        Args:
            input: input to the model
            detach (bool): if True, intermediate activations will be
                :code:`.detach()`d.

        Returns
            model output, a name => activation dict with saved intermediate
            activations.
        """
        self.detach = detach
        self.activations = {}
        out = self.model(input)
        acts = self.activations
        self.activations = {}
        return out, acts


T_Module = TypeVar('T_Module', bound=nn.Module)


def kaiming_gain(m: T_Module, a: float=0, nonlinearity='leaky_relu', mode='fan_in') ->float:
    """
    Return the std needed to initialize a weight matrix with given parameters.
    """
    if mode == 'fan_inout':
        fan = (math.sqrt(nn.init._calculate_correct_fan(m.weight, 'fan_in')) + math.sqrt(nn.init._calculate_correct_fan(m.weight, 'fan_out'))) / 2
    else:
        fan = math.sqrt(nn.init._calculate_correct_fan(m.weight, mode))
    gain = nn.init.calculate_gain(nonlinearity, param=a)
    return gain / fan


def kaiming(m: T_Module, a: float=0, nonlinearity: str='leaky_relu', mode: str='fan_out', dynamic: bool=False) ->T_Module:
    """
    Initialize a module with kaiming normal init

    Args:
        m (nn.Module): the module to init
        a (float): the slope of the nonlinearity
        nonlinearity (str): type of the nonlinearity
        dynamic (bool): wether to scale the weights on the forward pass for
            equalized LR such as ProGAN (default: False)

    Returns:
        the initialized module
    """
    assert isinstance(m.weight, torch.Tensor)
    if nonlinearity in ['relu', 'leaky_relu']:
        if a == 0:
            nonlinearity = 'relu'
        else:
            nonlinearity = 'leaky_relu'
    if not dynamic:
        nn.init.kaiming_normal_(m.weight, a=a, nonlinearity=nonlinearity, mode=mode)
    else:
        nn.init.normal_(m.weight, 0, 1)
        weight_scale(m, scale=kaiming_gain(m, a=a, nonlinearity=nonlinearity, mode=mode))
    if hasattr(m, 'biais') and m.bias is not None:
        assert isinstance(m.bias, torch.Tensor)
        nn.init.constant_(m.bias, 0)
    return m


class Registry:

    def __init__(self):
        self.sources = ['https://s3.eu-west-3.amazonaws.com/torchelie.models']
        self.known_models = {}

    def from_source(self, src: str, model: str) ->dict:
        uri = f'{src}/{model}'
        if uri.lower().startswith('http'):
            return torch.hub.load_state_dict_from_url(uri, map_location='cpu', file_name=model.replace('/', '.'))
        else:
            return torch.load(uri, map_location='cpu')

    def fetch(self, model: str) ->dict:
        for source in reversed(self.sources):
            try:
                return self.from_source(source, model)
            except Exception as e:
                None
        raise Exception(f'No source contains pretrained model {model}')

    def register_decorator(self, f):

        def _f(*args, pretrained: Optional[str]=None, **kwargs):
            model = f(*args, **kwargs)
            if pretrained:
                ckpt = self.fetch(f'{pretrained}/{f.__name__}.pth')
                tu.load_state_dict_forgiving(model, ckpt)
            return model
        self.known_models[f.__name__] = _f
        return _f

    def get_model(self, name, *args, **kwargs):
        return self.known_models[name](*args, **kwargs)


registry = Registry()


register = registry.register_decorator


@register
def vgg19(num_classes: int) ->'VGG':
    return VGG([64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'], num_classes)


class PerceptualNet(WithSavedActivations):
    """
    Make a VGG16 with appropriately named layers that records intermediate
    activations.

    Args:
        layers (list of str): the names of the layers for which to save the
            activations.
        use_avg_pool (bool): Whether to replace max pooling with averange
            pooling (default: True)
        remove_unused_layers (bool): whether to remove layers past the last one
            used (default: True)
    """

    def __init__(self, layers: List[str], use_avg_pool: bool=True, remove_unused_layers: bool=True) ->None:
        layer_names = ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'maxpool1', 'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'maxpool2', 'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'maxpool3', 'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'maxpool4', 'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4']
        m = vgg19(1, pretrained='perceptual/imagenet').features
        flat_vgg = [layer for layer in m.modules() if isinstance(layer, (nn.Conv2d, nn.ReLU, nn.MaxPool2d))]
        m = nn.Sequential(OrderedDict([(n, mod) for n, mod in zip(layer_names, flat_vgg)]))
        for nm, mod in m.named_modules():
            if 'relu' in nm:
                setattr(m, nm, nn.ReLU(False))
            elif 'pool' in nm and use_avg_pool:
                setattr(m, nm, nn.AvgPool2d(2, 2))
        if remove_unused_layers:
            m = m[:max([layer_names.index(layer) for layer in layers]) + 1]
        super().__init__(m, names=layers)


def bgram(m: torch.Tensor) ->torch.Tensor:
    """
    Return the batched Gram matrix of `m`

    Args:
        m (tensor): a matrix of dim 3, first one is the batch

    Returns:
        The batch of Gram matrix
    """
    assert m.dim() == 4
    m = m.view(m.shape[0], m.shape[1], -1)
    g = torch.einsum('bik,bjk->bij', m, m) / m.shape[2]
    return g


class NeuralStyleLoss(nn.Module):
    """
    Style Transfer loss by Leon Gatys

    https://arxiv.org/abs/1508.06576

    set the style and content before performing a forward pass.
    """
    net: PerceptualNet

    def __init__(self) ->None:
        super(NeuralStyleLoss, self).__init__()
        self.style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
        self.content_layers = ['conv3_2']
        self.content = {}
        self.style_maps = {}
        self.net = PerceptualNet(self.style_layers + self.content_layers, remove_unused_layers=False)
        self.norm = ImageNetInputNorm()
        tu.freeze(self.net)

    def get_style_content_(self, img: torch.Tensor, detach: bool) ->Dict[str, Dict[str, torch.Tensor]]:
        activations: Dict[str, torch.Tensor]
        _, activations = self.net(self.norm(img), detach=detach)
        activations = {k: F.instance_norm(a.float()) for k, a in activations.items()}
        return activations

    def set_style(self, style_img: torch.Tensor, style_ratio: float, style_layers: Optional[List[str]]=None) ->None:
        """
        Set the style.

        Args:
            style_img (3xHxW tensor): an image tensor
            style_ratio (float): a multiplier for the style loss to make it
                greater or smaller than the content loss
            style_layer (list of str, optional): the layers on which to compute
                the style, or `None` to keep them unchanged
        """
        if style_layers is not None:
            self.style_layers = style_layers
            self.net.set_keep_layers(names=self.style_layers + self.content_layers)
        self.ratio = torch.tensor(style_ratio)
        with torch.no_grad():
            out = self.get_style_content_(style_img, detach=True)
        self.style_maps = {k: bgram(out[k]) for k in self.style_layers}

    def set_content(self, content_img: torch.Tensor, content_layers: Optional[List[str]]=None) ->None:
        """
        Set the content.

        Args:
            content_img (3xHxW tensor): an image tensor
            content_layer (str, optional): the layer on which to compute the
                content representation, or `None` to keep it unchanged
        """
        if content_layers is not None:
            self.content_layers = content_layers
            self.net.set_keep_layers(names=self.style_layers + self.content_layers)
        with torch.no_grad():
            out = self.get_style_content_(content_img, detach=True)
        self.content = {a: out[a] for a in self.content_layers}

    def forward(self, input_img: torch.Tensor) ->Tuple[torch.Tensor, Dict[str, float]]:
        """
        Actually compute the loss
        """
        out = self.get_style_content_(input_img, detach=False)
        c_ratio = 1.0 - self.ratio.squeeze()
        s_ratio = self.ratio.squeeze()
        style_loss = sum(F.l1_loss(self.style_maps[a], bgram(out[a])) for a in self.style_layers) / len(self.style_maps)
        content_loss = sum(F.mse_loss(self.content[a], out[a]) for a in self.content_layers) / len(self.content_layers)
        loss = c_ratio * content_loss + s_ratio * style_loss
        return loss, {'style': style_loss.item(), 'content': content_loss.item()}


class CondSeq(nn.Sequential):
    """
    An extension to torch's Sequential that allows conditioning either as a
    second forward argument or `condition()`
    """

    def condition(self, z: Any) ->None:
        """
        Conditions all the layers on z

        Args:
            z: conditioning
        """
        for m in self:
            if hasattr(m, 'condition') and m is not self:
                cast(Callable, m.condition)(z)

    def forward(self, x: Any, z: Optional[Any]=None) ->Any:
        """
        Forward pass

        Args:
            x: input
            z (optional): conditioning. condition() must be called first if
                left None
        """
        for nm, m in self.named_children():
            try:
                if hasattr(m, 'condition') and z is not None:
                    x = m(x, z)
                else:
                    x = m(x)
            except Exception as e:
                raise Exception(f'Exception during forward pass of {nm}') from e
        return x


def Conv2d(in_ch, out_ch, ks, stride=1, bias=True) ->nn.Conv2d:
    """
    A Conv2d with 'same' padding
    """
    return nn.Conv2d(in_ch, out_ch, ks, padding=(ks - 1) // 2, stride=stride, bias=bias)


class Interpolate2d(nn.Module):
    """
    A wrapper around :func:`pytorch.nn.functional.interpolate`
    """

    def __init__(self, mode: str, size: Optional[List[int]]=None, scale_factor: Optional[float]=None) ->None:
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x: torch.Tensor, size: Optional[List[int]]=None) ->torch.Tensor:
        rsf = True if self.scale_factor is not None else None
        align = False if self.mode != 'nearest' else None
        if not size:
            return F.interpolate(x, mode=self.mode, size=self.size, scale_factor=self.scale_factor, recompute_scale_factor=rsf, align_corners=align)
        else:
            return F.interpolate(x, mode=self.mode, size=size, recompute_scale_factor=rsf, align_corners=align)

    def extra_repr(self) ->str:
        return f'scale_factor={self.scale_factor} size={self.size}'


class InterpolateBilinear2d(Interpolate2d):
    """
    A wrapper around :func:`pytorch.nn.functional.interpolate` with bilinear
    mode.
    """

    def __init__(self, size: Optional[List[int]]=None, scale_factor: Optional[float]=None) ->None:
        super().__init__(size=size, scale_factor=scale_factor, mode='bilinear')


@torch.no_grad()
def insert_after(base: nn.Sequential, key: str, new: nn.Module, name: str) ->nn.Sequential:
    """
    Insert module :code:`new` with name :code:`name` after element :code:`key`
    in sequential :code:`base` and return the new sequence.
    """
    modules_list = list(base._modules.items())
    found = -1
    for i, (nm, m) in enumerate(modules_list):
        if nm == key:
            found = i
            break
    assert found != -1
    modules_list.insert(found + 1, (name, new))
    base._modules = OrderedDict(modules_list)
    return base


@torch.no_grad()
def insert_before(base: nn.Sequential, key: str, new: nn.Module, name: str) ->nn.Sequential:
    """
    Insert module :code:`new` with name :code:`name` before element :code:`key`
    in sequential :code:`base` and return the new sequence.
    """
    modules_list = list(base._modules.items())
    found = -1
    for i, (nm, m) in enumerate(modules_list):
        if nm == key:
            found = i
            break
    assert found != -1
    modules_list.insert(found, (name, new))
    base._modules = OrderedDict(modules_list)
    return base


class UBlock(nn.Module):
    downsample: nn.Module

    def __init__(self, in_channels: int, hidden_channels: int, inner: nn.Module) ->None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_channels = hidden_channels
        self.set_encoder_num_layers(2)
        self.downsample = CondSeq(nn.MaxPool2d(2))
        self.inner = inner
        assert isinstance(inner.in_channels, int)
        assert isinstance(inner.out_channels, int)
        self.upsample = nn.ConvTranspose2d(inner.out_channels, inner.out_channels // 2, kernel_size=2, stride=2)
        self.set_decoder_num_layers(2)

    def to_bilinear_sampling(self) ->'UBlock':
        self.downsample = BinomialFilter2d(2)
        assert isinstance(self.inner.in_channels, int)
        assert isinstance(self.inner.out_channels, int)
        self.upsample = ConvBlock(self.inner.out_channels, self.inner.out_channels // 2, 3)
        insert_before(self.upsample, 'conv', InterpolateBilinear2d(scale_factor=2), 'upsample')
        return self

    def condition(self, z: torch.Tensor) ->None:

        def condition_if(m):
            if hasattr(m, 'condition'):
                m.condition(z)
        condition_if(self.in_conv)
        condition_if(self.downsample)
        condition_if(self.inner)
        condition_if(self.upsample)
        condition_if(self.out_conv)

    def forward(self, x_orig: torch.Tensor, z: Optional[torch.Tensor]=None) ->torch.Tensor:
        if z is not None:
            self.condition(z)
        x_skip = self.in_conv(x_orig)
        x = self.upsample(self.inner(self.downsample(x_skip)))
        x_skip = torch.cat([x, x_skip], dim=1)
        return self.out_conv(x_skip)

    def set_encoder_num_layers(self, num_layers: int) ->'UBlock':
        layers = CondSeq()
        for i in range(num_layers):
            layers.add_module(f'conv_{i}', ConvBlock(self.in_channels if i == 0 else self.hidden_channels, self.hidden_channels, 3))
        self.in_conv = layers
        return self

    def set_decoder_num_layers(self, num_layers: int) ->'UBlock':
        assert isinstance(self.inner.in_channels, int)
        assert isinstance(self.inner.out_channels, int)
        inner_out_ch = getattr(self.upsample, 'out_channels', self.inner.out_channels)
        layers = CondSeq()
        for i in range(num_layers):
            in_ch = inner_out_ch + self.hidden_channels if i == 0 else self.hidden_channels
            out_ch = self.in_channels if i == num_layers - 1 else self.hidden_channels
            layers.add_module(f'conv_{i}', ConvBlock(in_ch, out_ch, 3))
        self.out_conv = layers
        return self

    def remove_upsampling_conv(self) ->'UBlock':
        self.upsample = InterpolateBilinear2d(scale_factor=2)
        self.set_decoder_num_layers(len(self.out_conv))
        return self

    def remove_batchnorm(self) ->'UBlock':
        for m in self.modules():
            if isinstance(m, ConvBlock):
                m.remove_batchnorm()
        return self

    def leaky(self) ->'UBlock':
        for m in self.modules():
            if isinstance(m, ConvBlock):
                m.leaky()
        return self

    def set_padding_mode(self, mode: str) ->'UBlock':
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.padding_mode = mode
        return self


def xavier(m: T_Module, a: float=0, nonlinearity: str='relu', mode: str='fan_in', dynamic: bool=False) ->T_Module:
    """
    Initialize a module with xavier normal init

    Args:
        m (nn.Module): the module to init
        dynamic (bool): wether to scale the weights on the forward pass for
            equalized LR such as ProGAN (default: False)

    Returns:
        the initialized module
    """
    assert isinstance(m.weight, torch.Tensor)
    if nonlinearity in ['relu', 'leaky_relu']:
        if a == 0:
            nonlinearity = 'relu'
        else:
            nonlinearity = 'leaky_relu'
    if not dynamic:
        nn.init.xavier_normal_(m.weight)
    else:
        nn.init.normal_(m.weight, 0, 1)
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight)
        gain = nn.init.calculate_gain(nonlinearity, param=a)
        weight_scale(m, scale=gain * math.sqrt(2.0 / (fan_in + fan_out)))
    if hasattr(m, 'biais') and m.bias is not None:
        assert isinstance(m.bias, torch.Tensor)
        nn.init.constant_(m.bias, 0)
    return m


class LayerScale(nn.Module):

    def __init__(self, num_features):
        super().__init__()
        self.scales = nn.Parameter(torch.empty(num_features))
        nn.init.normal_(self.scales, 0, 1e-05)

    def forward(self, w):
        s = self.scales.view(-1, *([1] * (w.ndim - 1)))
        return s * w


class ConvNeXtBlock(nn.Module):

    def __init__(self, ch):
        super().__init__()
        e = 4
        self.branch = nn.Sequential(nn.Conv2d(ch, ch, kernel_size=7, padding=3, groups=ch, bias=False), nn.GroupNorm(1, ch), tu.kaiming(tnn.Conv1x1(ch, ch * e)), nn.SiLU(True), tu.constant_init(tnn.Conv1x1(ch * e, ch), 0))
        nn.utils.parametrize.register_parametrization(self.branch[-1], 'weight', LayerScale(ch))

    def forward(self, x):
        return self.branch(x).add_(x)


class ConvNeXt(nn.Sequential):

    def __init__(self, num_classes, arch):
        super().__init__()
        self.add_module('input', tu.xavier(nn.Conv2d(3, arch[0], 4, 1, 0)))
        prev_ch = arch[0]
        ch = arch[0]
        self.add_module(f'norm0', nn.GroupNorm(1, ch))
        self.add_module(f'act0', nn.SiLU(True))
        for i in range(len(arch)):
            if isinstance(arch[i], int):
                ch = arch[i]
                self.add_module(f'layer{i}', ConvNeXtBlock(ch))
                prev_ch = ch
            else:
                assert arch[i] == 'D'
                self.add_module(f'act{i}', nn.SiLU(True))
                self.add_module(f'norm{i}', nn.GroupNorm(1, ch))
                self.add_module(f'layer{i}', tu.kaiming(nn.Conv2d(ch, arch[i + 1], 2, 2, 0)))
        self.add_module(f'norm{i}', nn.GroupNorm(1, ch))
        self.add_module('classifier', ClassificationHead(arch[-1], num_classes))


class MixerBlock(nn.Module):

    def __init__(self, seq_len, in_features, hidden_token_mix, hidden_channel_mix):
        super().__init__()
        self.norm1 = nn.LayerNorm((seq_len, in_features))
        self.tokens_mlp = SpatialMlpBlock(seq_len, hidden_token_mix)
        self.norm2 = nn.LayerNorm((seq_len, in_features))
        self.channels_mlp = ChannelMlpBlock(in_features, hidden_channel_mix)

    def forward(self, x):
        x = x + self.tokens_mlp(self.norm1(x))
        x = x + self.channels_mlp(self.norm2(x))
        return x


class MultiScaleDiscriminator(nn.Module):

    def __init__(self, base_model: nn.Module, n_scales=3):
        super().__init__()
        self.scales = nn.ModuleList()
        self.scales.append(base_model)
        for i in range(n_scales - 1):
            self.scales.append(copy.deepcopy(base_model))

    def forward(self, x: torch.Tensor, flatten=True) ->List[torch.Tensor]:
        N = x.shape[0]
        outs = []
        for i in range(len(self.scales)):
            scale = 2 ** i
            out = self.scales[i](nn.functional.interpolate(x, scale_factor=1 / scale, mode='area')).view(N, -1)
            outs.append(out.view(N, -1))
        if flatten:
            return torch.cat(outs, dim=1)
        else:
            return outs


class FactoredPredictor(nn.Module):
    heads: nn.ModuleList

    def __init__(self, hid_ch: int, out_ch: int, n_pred: int) ->None:
        super(FactoredPredictor, self).__init__()
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(hid_ch + i, hid_ch + i), nn.ReLU(inplace=True), nn.Linear(hid_ch + i, out_ch)) for i in range(n_pred)])

    def forward(self, x: torch.Tensor, y: torch.Tensor) ->torch.Tensor:
        x = x.transpose(1, -1)
        y = y.transpose(1, -1)
        out = torch.stack([self.heads[i](torch.cat([x, y[..., :i]], dim=-1)) for i in range(len(self.heads))], dim=2)
        return out.transpose(1, -1)

    def sample(self, x: torch.Tensor, temp: float) ->torch.Tensor:
        sampled = torch.empty(x.shape[0], 0, device=x.device)
        for i in range(len(self.heads)):
            logits = self.heads[i](torch.cat([x, self.normalize(sampled)], dim=1)) / temp
            samp = torch.distributions.Categorical(logits=logits, validate_args=True).sample((1,))
            samp = samp.t()
            sampled = torch.cat([sampled, self.cls_to_val(samp.float())], dim=1)
        return sampled

    def normalize(self, x: torch.Tensor) ->torch.Tensor:
        return x

    def cls_to_val(self, cls: torch.Tensor) ->torch.Tensor:
        return cls.float()


class PixelPredictor(FactoredPredictor):

    def __init__(self, hid_ch: int, n_ch: int=3):
        super(PixelPredictor, self).__init__(hid_ch, 256, n_ch)

    def normalize(self, x: torch.Tensor) ->torch.Tensor:
        return x * 2 - 1

    def cls_to_val(self, cls: torch.Tensor) ->torch.Tensor:
        return cls.float() / 255


class ResBlk(nn.Module):

    @experimental
    def __init__(self, in_ch: int, hid_ch: int, out_ch: int, ks: int, sz: Tuple[int, int]) ->None:
        super(ResBlk, self).__init__()
        self.go = tnn.CondSeq(nn.BatchNorm2d(in_ch), nn.ReLU(inplace=False), tnn.Conv1x1(in_ch, hid_ch), nn.BatchNorm2d(hid_ch), nn.ReLU(inplace=True), tnn.TopLeftConv2d(hid_ch, hid_ch, ks, center=True, bias=sz), nn.BatchNorm2d(hid_ch), nn.ReLU(inplace=True), tnn.Conv1x1(hid_ch, out_ch))

    def condition(self, z: torch.Tensor) ->None:
        self.go.condition(z)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        out = self.go(x)
        return x + out


class PixCNNBase(nn.Module):

    @experimental
    def __init__(self, in_ch: int, hid: int, out_ch: int, quant_lvls: int, sz: Tuple[int, int], n_layer: int=3) ->None:
        super(PixCNNBase, self).__init__()
        self.sz = sz
        self.lin = tnn.CondSeq(tnn.TopLeftConv2d(in_ch, hid, 5, center=False, bias=sz), nn.ReLU(inplace=True))
        sz2 = sz[0] // 2, sz[1] // 2
        sz4 = sz[0] // 4, sz[1] // 4
        self.l1 = nn.Sequential(*[ResBlk(hid, hid * 2, hid, 5, sz) for _ in range(n_layer)])
        self.l2 = nn.Sequential(*[ResBlk(hid, hid * 2, hid, 5, sz2) for _ in range(n_layer)])
        self.l3 = nn.Sequential(*[ResBlk(hid, hid * 2, hid, 5, sz4) for _ in range(n_layer)])
        self.l4 = nn.Sequential(*[ResBlk(hid, hid * 2, hid, 5, sz4) for _ in range(n_layer)])
        self.l4 = nn.Sequential(*[ResBlk(hid, hid * 2, hid, 5, sz4) for _ in range(n_layer)])
        self.l5 = nn.Sequential(*[ResBlk(hid * 2, hid * 4, hid * 2, 5, sz2) for _ in range(n_layer)])
        self.l6 = nn.Sequential(*[ResBlk(hid * 3, hid * 6, hid * 3, 5, sz) for _ in range(n_layer)])
        self.lout = PixelPredictor(hid * 3, out_ch)

    def _body(self, x: torch.Tensor) ->torch.Tensor:
        x = self.lin(x)
        x1 = self.l1(x)
        x2 = self.l2(x1[..., ::2, ::2])
        x3 = self.l3(x2[..., ::2, ::2])
        x4 = self.l4(x3)
        x5 = self.l5(torch.cat([F.interpolate(x4, scale_factor=2), x2], dim=1))
        x6 = self.l6(torch.cat([F.interpolate(x5, scale_factor=2), x1], dim=1))
        return F.relu(x6)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        x6 = self._body(x)
        return self.lout(x6, x)

    def sample_xy(self, x: torch.Tensor, coord_x: int, coord_y: int, temp: float) ->torch.Tensor:
        x6 = self._body(x)
        return self.lout.sample(x6[:, :, coord_y, coord_x], temp)


class PixelCNN(PixCNNBase):
    """
    A PixelCNN model with 6 blocks

    Args:
        hid (int): the number of hidden channels in the blocks
        sz ((int, int)): the size of the images to learn. Must be square
        channels (int): number of channels in the data. 3 for RGB images
    """

    @experimental
    def __init__(self, hid: int, sz: Tuple[int, int], channels: int=3, n_layer: int=3) ->None:
        super(PixelCNN, self).__init__(channels, hid, channels, 256, sz, n_layer=n_layer)
        self.channels = channels

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        """A forward pass for training"""
        return super().forward(x)

    def sample(self, temp: float, N: int) ->torch.Tensor:
        """
        Sample a batch of images

        Args:
            temp (float): the sampling temperature
            N (int): number of images to generate in the batch

        Returns:
            A batch of images
        """
        img = torch.zeros(N, self.channels, *self.sz, device=next(self.parameters()).device).uniform_(0, 1)
        return self.sample_(img, temp)

    def partial_sample(self, x: torch.Tensor, temp: float) ->torch.Tensor:
        x[:, :, x.shape[2] // 2, :] = 0
        return self.sample_(x, temp, start_coord=(x.shape[2] // 2, 0))

    def sample_cond(self, cond: torch.Tensor, temp: float) ->torch.Tensor:
        device = next(self.parameters()).device
        img = torch.empty(cond.shape[0], self.sz[0], *self.sz, device=device).uniform_(0, 1)
        cond_rsz = F.interpolate(cond, size=img.shape[2:], mode='nearest')
        img = torch.cat([img, cond_rsz], dim=1)
        return self.sample_(img, temp)[:, cond.shape[1]:]

    def sample_(self, img: torch.Tensor, temp: float=0, start_coord: Tuple[int, int]=(0, 0)) ->torch.Tensor:
        self.eval()
        with torch.no_grad():
            for row in range(start_coord[0], self.sz[0]):
                for c in range(start_coord[1] if row == start_coord[0] else 0, self.sz[0]):
                    x = self.sample_xy(img * 2 - 1, c, row, temp)
                    img[:, :, row, c] = x
        return img


class ResNet(nn.Module):

    def __init__(self, arch: List[str], num_classes: int) ->None:
        super().__init__()

        def parse(layer: str) ->List[int]:
            return [int(x) for x in layer.split(':')]
        self.arch = list(map(parse, arch))
        self.features = tnn.CondSeq()
        self.features.add_module('input', ResNetInput(3, self.arch[0][0]))
        self.features.input.set_stride(self.arch[0][1])
        self._change_block_type('basic')
        self.classifier = ClassificationHead(self.arch[-1][0], num_classes)

    def _make_block(self, block_type: str, in_ch: int, out_ch: int, stride: int) ->nn.Module:
        if block_type == 'basic':
            return tnn.ResBlock(in_ch, out_ch, stride)
        if block_type == 'bottleneck':
            return tnn.ResBlockBottleneck(in_ch, out_ch, stride)
        if block_type == 'preact_basic':
            return tnn.PreactResBlock(in_ch, out_ch, stride)
        if block_type == 'preact_bottleneck':
            return tnn.PreactResBlockBottleneck(in_ch, out_ch, stride)
        if block_type == 'resnext':
            return tnn.ResBlockBottleneck(in_ch, out_ch, stride).resnext()
        if block_type == 'preact_resnext':
            return tnn.PreactResBlockBottleneck(in_ch, out_ch, stride).resnext()
        if block_type == 'wide':
            return tnn.ResBlockBottleneck(in_ch, out_ch, stride).wide()
        if block_type == 'preact_wide':
            return tnn.PreactResBlockBottleneck(in_ch, out_ch, stride).wide()
        assert False

    def _change_block_type(self, ty: str) ->None:
        arch = self.arch[1:]
        feats = tnn.CondSeq()
        assert isinstance(self.features.input, (ResNetInput, ResNetInputImproved))
        feats.add_module('input', self.features.input)
        self.features.input.set_stride(self.arch[0][1])
        in_ch = self.arch[0][0]
        self.in_channels = in_ch
        for i, (ch, s) in enumerate(arch):
            feats.add_module(f'block_{i}', self._make_block(ty, in_ch, ch, stride=s))
            in_ch = ch
        self.out_channels = ch
        if 'preact' in ty:
            assert isinstance(feats.block_0, PREACT_BLOCKS)
            feats.block_0.no_preact()
            feats.add_module('final_bn', nn.BatchNorm2d(self.out_channels))
            feats.add_module('final_relu', nn.ReLU(True))
        self.features = feats

    def use_standard_input(self):
        inp = self.features.input
        self.features.input = ResNetInput(inp.in_channels, inp.out_channels)
        self.features.input.set_stride(self.arch[0][1])

    def to_bottleneck(self) ->'ResNet':
        self._change_block_type('bottleneck')
        return self

    def to_preact_bottleneck(self) ->'ResNet':
        self._change_block_type('preact_bottleneck')
        return self

    def to_preact(self) ->'ResNet':
        self._change_block_type('preact_basic')
        return self

    def to_resnext(self) ->'ResNet':
        self._change_block_type('resnext')
        return self

    def to_preact_resnext(self) ->'ResNet':
        self._change_block_type('preact_resnext')
        return self

    def to_wide(self) ->'ResNet':
        self._change_block_type('wide')
        return self

    def to_preact_wide(self) ->'ResNet':
        self._change_block_type('preact_wide')
        return self

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        return self.classifier(self.features(x))

    def add_se(self) ->'ResNet':
        for m in self.features:
            if isinstance(m, BLOCKS):
                m.use_se()
        return self

    def set_input_specs(self, input_size: int=224, in_channels: int=3) ->'ResNet':
        assert isinstance(self.features.input, (ResNetInputImproved, ResNetInput))
        self.features.input.set_input_specs(input_size=input_size, in_channels=in_channels)
        return self


class AdaIN2d(nn.Module):
    """
    Adaptive InstanceNormalization from `*Arbitrary Style Transfer in Real-time
    with Adaptive Instance Normalization* (Huang et al, 2017)
    <https://arxiv.org/abs/1703.06868>`_

    Args:
        channels (int): number of input channels
        cond_channels (int): number of conditioning channels from which bias
            and scale will be derived
    """
    weight: Optional[torch.Tensor]
    bias: Optional[torch.Tensor]

    def __init__(self, channels: int, cond_channels: int) ->None:
        super(AdaIN2d, self).__init__()
        self.make_weight = nn.Linear(cond_channels, channels)
        self.make_bias = nn.Linear(cond_channels, channels)
        self.weight = None
        self.bias = None

    def forward(self, x: torch.Tensor, z: Optional[torch.Tensor]=None) ->torch.Tensor:
        """
        Forward pass

        Args:
            x (4D tensor): input tensor
            z (2D tensor, optional): conditioning vector. If not present,
                :code:`condition(z)` must be called first

        Returns:
            x, renormalized
        """
        if z is not None:
            self.condition(z)
        m = x.mean(dim=(2, 3), keepdim=True)
        s = torch.sqrt(x.var(dim=(2, 3), keepdim=True) + 1e-08)
        z_w = self.weight
        z_b = self.bias
        assert z_w is not None and z_b is not None, 'AdaIN did not receive a conditioning vector yet'
        weight = z_w / (s + 1e-05)
        bias = -m * weight + z_b
        out = weight * x + bias
        return out

    def condition(self, z: torch.Tensor) ->None:
        """
        Conditions the layer before the forward pass if z will not be present
        when calling forward

        Args:
            z (2D tensor, optional): conditioning vector
        """
        self.weight = self.make_weight(z)[:, :, None, None] + 1
        self.bias = self.make_bias(z)[:, :, None, None]


class FiLM2d(nn.Module):
    """
    Feature-wise Linear Modulation from
    https://distill.pub/2018/feature-wise-transformations/
    The difference with AdaIN is that FiLM does not uses the input's mean and
    std in its calculations

    Args:
        channels (int): number of input channels
        cond_channels (int): number of conditioning channels from which bias
            and scale will be derived
    """
    weight: Optional[torch.Tensor]
    bias: Optional[torch.Tensor]

    def __init__(self, channels: int, cond_channels: int):
        super(FiLM2d, self).__init__()
        self.make_weight = nn.Linear(cond_channels, channels)
        self.make_bias = nn.Linear(cond_channels, channels)
        self.weight = None
        self.bias = None

    def forward(self, x, z: Optional[torch.Tensor]=None) ->torch.Tensor:
        """
        Forward pass

        Args:
            x (4D tensor): input tensor
            z (2D tensor, optional): conditioning vector. If not present,
                :code:`condition(z)` must be called first

        Returns:
            x, conditioned
        """
        if z is not None:
            self.condition(z)
        w = self.weight
        assert w is not None
        x = w * x
        b = self.bias
        if b is not None:
            x = x + b
        return x

    def condition(self, z: torch.Tensor) ->None:
        """
        Conditions the layer before the forward pass if z will not be present
        when calling forward

        Args:
            z (2D tensor, optional): conditioning vector
        """
        self.weight = self.make_weight(z)[:, :, None, None].mul_(0.1).add_(1)
        self.bias = self.make_bias(z)[:, :, None, None].mul_(0.01)


class BatchNorm2dBase_(nn.Module):

    @experimental
    def __init__(self, channels, momentum=0.8):
        super(BatchNorm2dBase_, self).__init__()
        self.register_buffer('running_mean', torch.zeros(1, channels, 1, 1))
        self.register_buffer('running_var', torch.ones(1, channels, 1, 1))
        self.register_buffer('step', torch.ones(1))
        self.momentum = momentum

    def update_moments(self, x):
        if self.training:
            m = x.mean(dim=(0, 2, 3), keepdim=True)
            v = torch.sqrt(x.var(dim=(0, 2, 3), unbiased=False, keepdim=True) + 1e-08)
            self.running_mean.copy_(self.momentum * self.running_mean + (1 - self.momentum) * m)
            self.running_var.copy_(self.momentum * self.running_var + (1 - self.momentum) * v)
            self.step += 1
        else:
            m = self.running_mean
            v = self.running_var
        return m, v


class MovingAverageBN2dBase_(nn.Module):

    @experimental
    def __init__(self, channels, momentum=0.8):
        super(MovingAverageBN2dBase_, self).__init__()
        self.register_buffer('running_mean', torch.zeros(1, channels, 1, 1))
        self.register_buffer('running_var', torch.zeros(1, channels, 1, 1))
        self.register_buffer('step', torch.ones(1))
        self.momentum = momentum

    def update_moments(self, x):
        if self.training:
            m = x.mean(dim=(0, 2, 3), keepdim=True)
            v = x.var(dim=(0, 2, 3), keepdim=True)
            m = self.momentum * self.running_mean + (1 - self.momentum) * m
            self.running_mean.copy_(m.detach())
            m = m / (1 - self.momentum ** self.step)
            v = self.momentum * self.running_var + (1 - self.momentum) * v
            self.running_var.copy_(v.detach())
            v = v / (1 - self.momentum ** self.step)
            self.step += 1
        else:
            m = self.running_mean
            v = self.running_var
        return m, torch.sqrt(v)


class AttenNorm2d(nn.BatchNorm2d):
    """
    From https://arxiv.org/abs/1908.01259
    """

    def __init__(self, num_features, num_weights, eps=1e-08, momentum=0.8, track_running_stats=True):
        super(AttenNorm2d, self).__init__(num_features, eps=eps, momentum=momentum, affine=False, track_running_stats=track_running_stats)
        self.gamma = nn.Parameter(torch.ones(num_weights, num_features))
        self.beta = nn.Parameter(torch.zeros(num_weights, num_features))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(num_features, num_weights)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        output = super(AttenNorm2d, self).forward(x)
        size = output.size()
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.fc(y)
        y = self.sigmoid(y)
        gamma = torch.mm(y, self.gamma)
        beta = torch.mm(y, self.beta)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1).expand(size)
        beta = beta.unsqueeze(-1).unsqueeze(-1).expand(size)
        return gamma * output + beta


class GhostBatchNorm2d(nn.Module):

    def __init__(self, num_features, ghost_batch_size, eps=1e-05, affine=True, momentum=0.8):
        """
        BatchNorm2d with virtual batch size for greater regularization effect
        and / or greater reproducibility.

        Args:
            num_features (int): number of input features
            ghost_batch_size (int): batch size to consider when computing
                statistics and normalizing at train time. Must be able to
                divide the actual batch size when training.
            eps (float): epsilon for numerical stability
            affine (bool): whether to add an affine transformation after
                statistics normalization or not
            momentum (float): exponential moving average coefficient for
                tracking batch statistics used at test time.
        """
        super().__init__()
        self.num_features = num_features
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.weight = None
            self.bias = None
        self.momentum = momentum
        self.ghost_batch_size = ghost_batch_size
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.zeros(1, dtype=torch.int32))
        self.eps = eps

    def forward(self, x):
        B, C, H, W = x.shape
        if self.training:
            assert B % self.ghost_batch_size == 0, f'Batch size {B} cannot be divided by ghost size {self.ghost_batch_size}'
            x = x.reshape(-1, self.ghost_batch_size, C, H, W)
            var, mean = torch.var_mean(x, [1, 3, 4], unbiased=False)
            with torch.no_grad():
                self.running_mean.mul_(self.momentum).add_(mean.mean(0), alpha=1 - self.momentum)
                self.running_var.mul_(self.momentum).add_(var.mean(0), alpha=1 - self.momentum)
            mean = mean[:, None, :, None, None]
            var = var[:, None, :, None, None]
        else:
            x = x.reshape(B, C, H, W)
            mean = self.running_mean[:, None, None]
            var = self.running_var[:, None, None]
        if self.weight is not None:
            weight = self.weight[:, None, None]
            bias = self.bias[:, None, None]
            weight = weight / var.sqrt().add(self.eps)
            bias = bias.addcmul(mean, weight, value=-1)
            x = bias.addcmul(x, weight)
        else:
            x = (x - mean) / torch.sqrt(var + self.eps)
        x = x.reshape(B, C, H, W)
        return x

    def extra_repr(self):
        return f'num_features={self.num_features}, ghost_batch_size={self.ghost_batch_size}'


def Conv1x1(in_ch: int, out_ch: int, stride: int=1, bias: bool=True) ->nn.Conv2d:
    """
    A 1x1 Conv2d
    """
    return Conv2d(in_ch, out_ch, 1, stride=stride, bias=bias)


class AutoGANGenBlock(nn.Module):
    """
    A block of the generator discovered by AutoGAN.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        skips_ch (list of int): a list with one element per incoming skip
            connection. Each element is the number of channels of the incoming
            skip connection.
        ks (int): kernel size of the convolutions
        mode (str): usampling mode, 'nearest' or 'bilinear'
    """

    def __init__(self, in_ch: int, out_ch: int, skips_ch: List[int], ks: int=3, mode: str='nearest') ->None:
        super(AutoGANGenBlock, self).__init__()
        assert mode in ['nearest', 'bilinear']
        self.mode = mode
        self.preact = CondSeq()
        self.conv1 = ConvBlock(in_ch, out_ch, ks)
        self.conv1.to_preact().remove_batchnorm()
        self.conv1.leaky()
        self.conv2 = ConvBlock(out_ch, out_ch, ks)
        self.conv2.to_preact().remove_batchnorm()
        self.conv2.leaky()
        self.shortcut = None
        if in_ch != out_ch:
            self.shortcut = kaiming(Conv1x1(in_ch, out_ch, 1))
            self.preact.add_module('relu', self.conv1.relu)
            del self.conv1.relu
        self.skip_convs = nn.ModuleList([kaiming(Conv1x1(ch, out_ch)) for ch in skips_ch])

    def forward(self, x: torch.Tensor, skips: List[torch.Tensor]=[]) ->Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            x (tensor): input tensor
            skips (list of tensor): a tensor per incoming skip connection

        Return:
            output tensor, intermediate value to use in the next block's skip
                connections
        """
        x = F.interpolate(x, scale_factor=2.0, mode=self.mode)
        x = self.preact(x)
        x_skip = x
        if self.shortcut is not None:
            x_skip = self.shortcut(x_skip)
        x_mid = self.conv1(x)
        x_w_skips = x_mid.clone()
        for conv, skip in zip(self.skip_convs, skips):
            x_w_skips.add_(conv(F.interpolate(skip, size=x_mid.shape[-2:], mode=self.mode)))
        x = self.conv2(x_w_skips)
        return x + x_skip, x_mid


class ModulatedConv(nn.Conv2d):

    def __init__(self, in_channels: int, noise_channels: int, *args, demodulate: bool=True, gain: float=1, **kwargs):
        super(ModulatedConv, self).__init__(in_channels, *args, **kwargs)
        with torch.no_grad():
            self.make_s = tu.xavier(nn.Linear(noise_channels, in_channels))
            self.make_s.bias.data.fill_(1)
        self.demodulate = demodulate
        self.gain = gain

    def to_equal_lr(self, leak: float=0.2) ->'ModulatedConv':
        self.weight.data.normal_(0, 1)
        tu.xavier(self.make_s, dynamic=True)
        self.make_s.bias.data.fill_(1)
        return self

    def condition(self, z: torch.Tensor) ->None:
        self.s = self.make_s(z)

    def forward(self, x: torch.Tensor, z: Optional[torch.Tensor]=None) ->torch.Tensor:
        if z is not None:
            self.condition(z)
        N, C, H, W = x.shape
        C_out, C_in = self.weight.shape[:2]
        w_prime = torch.einsum('oihw,bi->boihw', self.weight, self.s)
        if self.demodulate:
            w_prime_prime = torch.einsum('boihw,boihw->bo', w_prime, w_prime)
            w_prime_prime = w_prime_prime.add_(1e-08).rsqrt()
            w = w_prime * w_prime_prime[..., None, None, None]
        else:
            w = w_prime
        w = self.gain * w
        w = w.view(-1, *w.shape[2:])
        x = F.conv2d(x.view(1, -1, H, W), w, None, self.stride, self.padding, self.dilation, N)
        x = x.view(N, C_out, H, W)
        if self.bias is not None:
            return x.add_(self.bias.view(-1, 1, 1))
        else:
            return x


def tup(x):
    if isinstance(x, (tuple, list)):
        return list(x)
    return [x]


class ModuleGraph(nn.Sequential):
    """
    Allows description of networks as computation graphs. The graph is
    constructed by labelling inputs and outputs of each node. Each node will be
    ran in declaration order, fetching its input values from a pool of named
    values populated from previous node's output values and keyword arguments
    in forward.

    Simple example:

    >>> m = tnn.ModuleGraph(outputs='y')
    >>> m.add_operation(
            inputs=['x'],
            operation=nn.Linear(10, 20),
            name='linear',
            outputs=['y'])
    >>> m(x=torch.randn(1, 10))
    <a bunch of numbers>

    Multiple inputs example:

    If a layer takes more than 1 input, labels can be a tuple or a list of
    labels instead. The same applies if a module returns more than 1 output
    values.

    >>> m = tnn.ModuleGraph(outputs=['x1', 'y'])
    >>> m.add_operation(
            inputs=['x0'],
            operation=nn.Linear(10, 20)
            name='linear',
            outputs=['x1'])
    >>> m.add_operation(
            inputs=['x1', 'z'],
            operation=nn.AdaIN2d(20, 3)
            name='adain',
            outputs=['y'])
    >>> m(x0=torch.randn(1, 10), z=torch.randn(1, 3))['y']
    <a bunch of numbers>
    """

    def __init__(self, outputs: Union[str, List[str]]) ->None:
        super().__init__()
        self.ins: List[List[str]] = []
        self.outs: List[List[str]] = []
        self.outputs = outputs

    def add_operation(self, inputs: List[str], outputs: List[str], name: str, operation: nn.Module) ->'ModuleGraph':
        self.ins.append(inputs)
        self.outs.append(outputs)
        self.add_module(name, operation)
        return self

    def forward(self, **args):
        variables = dict(args)
        for i_names, f, o_names in zip(self.ins, self._modules.values(), self.outs):
            ins = [variables[k] for k in i_names]
            outs = tup(f(*ins))
            for o, k in zip(outs, o_names):
                variables[k] = o
        if isinstance(self.outputs, str):
            return variables[self.outputs]
        return {k: variables[k] for k in self.outputs}

    @experimental
    def to_dot(self) ->str:
        txt = ''
        for i_names, f_nm, o_names in zip(self.ins, self._modules.keys(), self.outs):
            for k in i_names:
                txt += f'{k} -> {f_nm};\n'
            for k in o_names:
                txt += f'{f_nm} -> {k};\n'
            txt += f'{f_nm} [shape=square];\n'
        return txt


class Noise:

    def __init__(self, std):
        self.std = std

    def __call__(self, x):
        return x + torch.randn_like(x) * self.std


class Debug(nn.Module):
    """
    An pass-through layer that prints some debug info during forward pass.
    It prints its name, the input's shape, mean of channels means, mean,
    mean of channels std, and std.

    Args:
        name (str): this layer's name
    """

    @experimental
    def __init__(self, name):
        super(Debug, self).__init__()
        self.name = name

    def forward(self, x):
        None
        None
        None
        if x.ndim == 2:
            None
        if x.ndim == 4:
            None
        None
        return x


class Dummy(nn.Module):
    """
    A pure pass-through layer
    """

    def forward(self, x):
        return x


class ConvDeconvBlock(nn.Module):

    def __init__(self, in_channels: int, hidden_channels: int, inner: nn.Module) ->None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_channels = hidden_channels
        self.with_skip = False
        self.pre = CondSeq()
        self.downsample = CondSeq(OrderedDict([('conv_0', ConvBlock(self.in_channels, self.hidden_channels, 4, stride=2))]))
        self.inner = inner
        assert isinstance(inner.out_channels, int)
        self.upsample = CondSeq(OrderedDict([('conv_0', ConvBlock(inner.out_channels, self.in_channels, 4, stride=2).to_transposed_conv())]))
        self.post = CondSeq()

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        x = self.pre(x)
        out = self.upsample(self.inner(self.downsample(x)))
        if self.with_skip:
            out = torch.cat([out, x], dim=1)
        out = self.post(out)
        return out

    def add_skip(self) ->'ConvDeconvBlock':
        self.with_skip = True
        if self.with_skip:
            self.out_channels = self.in_channels + self.in_channels
        return self

    def leaky(self) ->'ConvDeconvBlock':
        for block in [self.pre, self.downsample, self.upsample, self.post]:
            for m in block:
                if isinstance(m, ConvBlock):
                    m.leaky()
        return self


class AdaptiveConcatPool2d(nn.Module):
    """
    Pools with AdaptiveMaxPool2d AND AdaptiveAvgPool2d and concatenates both
    results.

    Args:
        target_size: the target output size (single integer or
            double-integer tuple)
    """

    def __init__(self, target_size):
        super(AdaptiveConcatPool2d, self).__init__()
        self.target_size = target_size

    def forward(self, x):
        return torch.cat([nn.functional.adaptive_avg_pool2d(x, self.target_size), nn.functional.adaptive_max_pool2d(x, self.target_size)], dim=1)


class SelfAttention2d(nn.Module):
    """
    Self Attention such as used in SAGAN or BigGAN.

    Args:
        ch (int): number of input / output channels
    """

    def __init__(self, ch: int, num_heads: int=1, out_ch: Optional[int]=None):
        super().__init__()
        self.num_heads = num_heads
        self.key = tu.xavier(nn.Conv1d(ch, ch, 1, bias=True))
        self.query = tu.xavier(nn.Conv1d(ch, ch, 1, bias=True))
        self.value = tu.xavier(nn.Conv1d(ch, ch, 1))
        out_ch = out_ch or ch
        self.out = tu.xavier(nn.Conv2d(ch, out_ch, 1))

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        """
        forward
        """
        N, C, H, W = x.shape
        K = self.num_heads
        x_flat = x.view(N, C, -1)
        k = self.key(x_flat).view(N, K, -1, H * W)
        q = self.query(x_flat).view(N, K, -1, H * W)
        v = self.value(x_flat).view(N, K, -1, H * W)

        def kqv(k, q, v, out_shape):
            affinity = torch.matmul(q.permute(0, 1, 3, 2), k).mul_(1 / math.sqrt(q.shape[2]))
            attention = F.softmax(affinity, dim=-1)
            return torch.matmul(v, attention.transpose(-1, -2)).view(*out_shape)
        if self.training:
            out = torch.utils.checkpoint.checkpoint(kqv, k, q, v, x.shape, preserve_rng_state=False)
        else:
            out = kqv(k, q, v, x.shape)
        return self.out(out)


class UnitGaussianPrior(nn.Module):
    """
    Force a representation to fit a unit gaussian prior. It projects with a
    nn.Linear the input vector to a mu and sigma that represent a gaussian
    distribution from which the output is sampled. The backward pass includes a
    kl divergence loss between N(mu, sigma) and N(0, 1).

    This can be used to implement VAEs or information bottlenecks

    In train mode, the output is sampled from N(mu, sigma) but at test time mu
    is returned.

    Args:
        in_channels (int): dimension of input channels
        num_latents (int): dimension of output latents
        strength (float): strength of the kl loss. When using this to implement
            a VAE, set strength to :code:`1/number of output dim of the model`
            or set it to 1 but make sure that the loss for each output
            dimension is summed, but averaged over the batch.
        kl_reduction (str): how the implicit kl loss is reduced over the batch
            samples. 'sum' means the kl term of each sample is summed, while
            'mean' divides the loss by the number of examples.
    """

    def __init__(self, in_channels, num_latents, strength=1, kl_reduction='mean'):
        super().__init__()
        self.project = tu.kaiming(nn.Linear(in_channels, 2 * num_latents))
        self.project.bias.data[num_latents:].fill_(1)
        self.strength = strength
        assert kl_reduction in ['mean', 'sum']
        self.reduction = kl_reduction

    def forward(self, x):
        """
        Args:
            x (Tensor): A 2D (N, in_channels) tensor

        Returns:
            A 2D (N, num_channels) tensor sampled from the implicit gaussian
                distribution.
        """
        x = self.project(x)
        mu, sigma = torch.chunk(x, 2, dim=1)
        if self.training:
            sigma = torch.exp(0.5 * sigma).add_(1e-05)
            strength = self.strength
            if self.reduction == 'mean':
                strength = strength / x.shape[0]
            return tch.nn.functional.unit_gaussian_prior(mu, sigma, strength)
        else:
            return mu


class InformationBottleneck(UnitGaussianPrior):
    pass


class MinibatchStddev(nn.Module):
    """Minibatch Stddev layer from Progressive GAN"""

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        stddev_map = torch.sqrt(x.var(dim=0) + 1e-08).mean()
        stddev = stddev_map.expand(x.shape[0], 1, *x.shape[2:])
        return torch.cat([x, stddev], dim=1)


class HardSigmoid(nn.Module):
    """
    Hard Sigmoid
    """

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        return x.add_(0.5).clamp_(min=0, max=1)


class HardSwish(nn.Module):
    """
    Hard Swish
    """

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        return x.add(0.5).clamp_(min=0, max=1).mul_(x)


def drop_path(x, drop_prob=0.0, training=False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x * random_tensor.div(keep_prob)
    return output


class DropPath(nn.Module):

    def __init__(self, p, **kwargs):
        super().__init__()
        self.p = p

    def forward(self, x):
        x = drop_path(x, self.p, self.training)
        return x

    def extra_repr(self):
        return 'p=%s' % repr(self.p)


class MaskedConv2d(nn.Conv2d):
    """
    A masked 2D convolution for PixelCNN

    Args:
        in_chan (int): number of input channels
        out_chan (int): number of output channels
        ks (int): kernel size
        center (bool): whereas central pixel is masked or not
        stride (int): stride, defaults to 1
        bias (2-tuple of ints): A spatial bias. Either the spatial dimensions
            of the input for a different bias at each location, or (1, 1) for
            the same bias everywhere (default)
    """

    def __init__(self, in_chan, out_chan, ks, center, stride=1, bias=(1, 1)):
        super(MaskedConv2d, self).__init__(in_chan, out_chan, (ks // 2 + 1, ks), padding=0, stride=stride, bias=False)
        self.register_buffer('mask', torch.ones(ks // 2 + 1, ks))
        self.mask[-1, ks // 2 + (1 if center else 0):] = 0
        self.spatial_bias = None
        if bias is not None:
            self.spatial_bias = nn.Parameter(torch.zeros(out_chan, *bias))
        nn.init.kaiming_uniform_(self.weight)

    def forward(self, x):
        self.weight_orig = self.weight
        del self.weight
        self.weight = self.weight_orig * self.mask
        ks = self.weight.shape[-1]
        x = F.pad(x, (ks // 2, ks // 2, ks // 2, 0))
        res = super(MaskedConv2d, self).forward(x)
        self.weight = self.weight_orig
        del self.weight_orig
        if self.spatial_bias is not None:
            return res + self.spatial_bias
        else:
            return res


class TopLeftConv2d(nn.Module):
    """
    A 2D convolution for PixelCNN made of a convolution above the current pixel
    and another on the left.

    Args:
        in_chan (int): number of input channels
        out_chan (int): number of output channels
        ks (int): kernel size
        center (bool): whereas central pixel is masked or not
        stride (int): stride, defaults to 1
        bias (2-tuple of ints): A spatial bias. Either the spatial dimensions
            of the input for a different bias at each location, or (1, 1) for
            the same bias everywhere (default)
    """

    @experimental
    def __init__(self, in_chan, out_chan, ks, center, stride=1, bias=(1, 1)):
        super(TopLeftConv2d, self).__init__()
        self.top = kaiming(nn.Conv2d(in_chan, out_chan, (ks // 2, ks), bias=False, stride=stride))
        self.left = kaiming(nn.Conv2d(in_chan, out_chan, (1, ks // 2 + (1 if center else 0)), stride=stride, bias=False))
        self.ks = ks
        self.center = center
        self.bias = nn.Parameter(torch.zeros(out_chan, *bias))

    def forward(self, x):
        top = self.top(F.pad(x[:, :, :-1, :], (self.ks // 2, self.ks // 2, self.ks // 2, 0)))
        if not self.center:
            left = self.left(F.pad(x[:, :, :, :-1], (self.ks // 2, 0, 0, 0)))
        else:
            left = self.left(F.pad(x, (self.ks // 2, 0, 0, 0)))
        return top + left + self.bias


class PixelNorm(torch.nn.Module):
    """
    PixelNorm from ProgressiveGAN
    """

    def forward(self, x):
        return x / (x.mean(dim=1, keepdim=True).sqrt() + 1e-08)


class SEBlock(nn.Module):
    """
    A Squeeze-And-Excite block

    Args:
        in_ch (int): input channels
        reduction (int): channels reduction factor for the hidden number of
            channels
    """

    def __init__(self, in_ch: int, reduction: int=16) ->None:
        super(SEBlock, self).__init__()
        reduc = in_ch // reduction
        self.proj = nn.Sequential(collections.OrderedDict([('pool', nn.AdaptiveAvgPool2d(1)), ('squeeze', kaiming(Conv1x1(in_ch, reduc))), ('relu', nn.ReLU(True)), ('excite', kaiming(Conv1x1(reduc, in_ch))), ('attn', nn.Sigmoid())]))

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        return x * self.proj(x)


def Conv3x3(in_ch: int, out_ch: int, stride: int=1, bias: bool=True) ->nn.Conv2d:
    """
    A 3x3 Conv2d with 'same' padding
    """
    return Conv2d(in_ch, out_ch, 3, stride=stride, bias=bias)


def _make_resnet_shortcut(in_channels: int, out_channels: int, stride: int) ->CondSeq:
    assert stride in [1, 2]
    shortcut = CondSeq()
    if stride != 1:
        shortcut.add_module('pool', nn.AvgPool2d(2, 2, ceil_mode=True))
    if in_channels != out_channels:
        shortcut.add_module('conv', kaiming(Conv1x1(in_channels, out_channels, bias=False)))
        shortcut.add_module('bn', nn.BatchNorm2d(out_channels))
    return shortcut


def constant_init(m: nn.Module, val: float) ->nn.Module:
    """
    Initialize a module with gaussian weights of standard deviation std

    Args:
        m (nn.Module): the module to init

    Returns:
        the initialized module
    """
    assert isinstance(m.weight, torch.Tensor)
    nn.init.constant_(m.weight, val)
    if hasattr(m, 'biais') and m.bias is not None:
        assert isinstance(m.bias, torch.Tensor)
        nn.init.constant_(m.bias, 0)
    return m


class ResBlockBottleneck(nn.Module):
    """
    A Residual Block. Skip connection will be added if the number of input and
    output channels don't match or stride > 1 is used.

    Args:
        in_ch (int): input channels
        out_ch (int): output channels
        stride (int): stride
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int=1) ->None:
        super(ResBlockBottleneck, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.pre = CondSeq()
        self.wide(divisor=4)
        self.shortcut = _make_resnet_shortcut(in_channels, out_channels, stride)
        self.post = CondSeq()
        self.post.relu = nn.ReLU(True)

    def condition(self, z: torch.Tensor) ->None:
        self.branch.condition(z)
        self.shortcut.condition(z)
        self.pre.condition(z)
        self.post.condition(z)

    def forward(self, x: torch.Tensor, z: Optional[torch.Tensor]=None) ->torch.Tensor:
        if z is not None:
            self.condition(z)
        x = self.pre(x)
        x = self.branch(x).add_(self.shortcut(x))
        return self.post(x)

    def remove_batchnorm(self) ->'ResBlockBottleneck':
        remove_batchnorm(self.branch)
        remove_batchnorm(self.shortcut)
        assert isinstance(self.branch.conv3, nn.Conv2d)
        constant_init(self.branch.conv3, 0)
        return self

    def use_se(self) ->'ResBlockBottleneck':
        self.branch.add_module('se', SEBlock(self.out_channels))
        return self

    def resnext(self) ->'ResBlockBottleneck':
        self.wide(divisor=2)
        c = self.branch.conv2
        assert isinstance(c, nn.Conv2d)
        self.branch.conv2 = kaiming(nn.Conv2d(c.in_channels, c.out_channels, kernel_size=3, padding=1, stride=self.stride, bias=c.bias is not None, groups=32))
        return self

    def wide(self, divisor: int=2) ->'ResBlockBottleneck':
        in_ch = self.in_channels
        out_ch = self.out_channels
        mid = out_ch // divisor
        stride = self.stride
        self.branch = CondSeq(collections.OrderedDict([('conv1', kaiming(Conv1x1(in_ch, mid, bias=False))), ('bn1', constant_init(nn.BatchNorm2d(mid), 1)), ('relu', nn.ReLU(True)), ('conv2', kaiming(Conv3x3(mid, mid, stride=stride, bias=False))), ('bn2', constant_init(nn.BatchNorm2d(mid), 1)), ('relu2', nn.ReLU(True)), ('conv3', kaiming(Conv1x1(mid, out_ch, bias=False))), ('bn3', constant_init(nn.BatchNorm2d(out_ch), 0))]))
        return self

    def upsample_instead(self) ->'ResBlockBottleneck':
        if self.stride == 1:
            return self
        assert isinstance(self.branch.conv2, nn.Conv2d)
        self.branch.conv2.stride = 1, 1
        insert_before(self.branch, 'conv2', InterpolateBilinear2d(scale_factor=self.stride), 'upsample')
        if hasattr(self.shortcut, 'pool'):
            self.shortcut.pool = InterpolateBilinear2d(scale_factor=self.stride)
        return self


class ResBlock(nn.Module):
    """
    A Residual Block. Skip connection will be added if the number of input and
    output channels don't match or stride > 1 is used.

    Args:
        in_channels (int): input channels
        out_channels (int): output channels
        stride (int): stride
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int=1) ->None:
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.pre = CondSeq()
        self.branch = CondSeq(collections.OrderedDict([('conv1', kaiming(Conv3x3(in_channels, out_channels, stride=stride, bias=False))), ('bn1', constant_init(nn.BatchNorm2d(out_channels), 1)), ('relu', nn.ReLU(True)), ('conv2', kaiming(Conv3x3(out_channels, out_channels, bias=False))), ('bn2', constant_init(nn.BatchNorm2d(out_channels), 0))]))
        self.shortcut = _make_resnet_shortcut(in_channels, out_channels, stride)
        self.post = CondSeq()
        self.post.relu = nn.ReLU(True)

    def condition(self, z: torch.Tensor) ->None:
        self.branch.condition(z)
        self.shortcut.condition(z)
        self.pre.condition(z)
        self.post.condition(z)

    def forward(self, x: torch.Tensor, z: Optional[torch.Tensor]=None) ->torch.Tensor:
        if z is not None:
            self.condition(z)
        x = self.pre(x)
        x = self.branch(x).add_(self.shortcut(x))
        return self.post(x)

    def remove_batchnorm(self) ->'ResBlock':
        remove_batchnorm(self.branch)
        remove_batchnorm(self.shortcut)
        assert isinstance(self.branch.conv2, nn.Conv2d)
        constant_init(self.branch.conv2, 0)
        return self

    def use_se(self) ->'ResBlock':
        self.branch.add_module('se', SEBlock(self.out_channels))
        return self


def make_preact_resnet_shortcut(in_ch: int, out_ch: int, stride: int) ->CondSeq:
    assert stride in [1, 2]
    sc: List[Tuple[str, nn.Module]] = []
    if stride != 1:
        sc.append(('pool', nn.AvgPool2d(2, 2, ceil_mode=True)))
    if in_ch != out_ch:
        sc.append(('conv', kaiming(Conv1x1(in_ch, out_ch))))
    return CondSeq(collections.OrderedDict(sc))


class PreactResBlock(nn.Module):
    """
    A Preactivated Residual Block. Skip connection will be added if the number
    of input and output channels don't match or stride > 1 is used.

    Args:
        in_ch (int): input channels
        out_ch (int): output channels
        stride (int): stride
    """
    branch: CondSeq
    shortcut: CondSeq

    def __init__(self, in_channels: int, out_channels: int, stride: int=1) ->None:
        super(PreactResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.pre = CondSeq()
        self.preact = CondSeq()
        self.branch = CondSeq(collections.OrderedDict([('bn1', constant_init(nn.BatchNorm2d(in_channels), 1)), ('relu', nn.ReLU(True)), ('conv1', kaiming(Conv3x3(in_channels, out_channels, stride=stride, bias=False))), ('bn2', constant_init(nn.BatchNorm2d(out_channels), 1)), ('relu2', nn.ReLU(True)), ('conv2', constant_init(Conv3x3(out_channels, out_channels), 0))]))
        self.shortcut = make_preact_resnet_shortcut(in_channels, out_channels, stride)
        if in_channels != out_channels:
            self.preact_skip()
        self.post = CondSeq()

    def remove_batchnorm(self) ->'PreactResBlock':
        remove_batchnorm(self.branch)
        remove_batchnorm(self.shortcut)
        remove_batchnorm(self.preact)
        return self

    def condition(self, z: torch.Tensor) ->None:
        self.pre.condition(z)
        self.preact.condition(z)
        self.branch.condition(z)
        self.shortcut.condition(z)
        self.post.condition(z)

    def preact_skip(self) ->'PreactResBlock':
        if hasattr(self.branch, 'bn1'):
            self.preact.add_module('bn1', cast(nn.Module, self.branch.bn1))
            del self.branch.bn1
        if hasattr(self.branch, 'relu'):
            self.preact.add_module('relu', cast(nn.Module, self.branch.relu))
            del self.branch.relu
        return self

    def no_preact(self) ->'PreactResBlock':
        mods = collections.OrderedDict([*self.preact.named_children(), *self.branch.named_children()])
        self.branch = CondSeq(mods)
        self.preact = CondSeq()
        return self

    def forward(self, x: torch.Tensor, z: Optional[torch.Tensor]=None) ->torch.Tensor:
        if z is not None:
            self.condition(z)
        x = self.pre(x)
        x = self.preact(x)
        x = self.shortcut(x) + self.branch(x)
        return self.post(x)

    def use_se(self) ->'PreactResBlock':
        self.branch.add_module('se', SEBlock(self.out_channels))
        return self


class PreactResBlockBottleneck(nn.Module):
    """
    A Preactivated Residual Block. Skip connection will be added if the number
    of input and output channels don't match or stride > 1 is used.

    Args:
        in_ch (int): input channels
        out_ch (int): output channels
        stride (int): stride
    """
    branch: CondSeq
    shortcut: CondSeq

    def __init__(self, in_channels: int, out_channels: int, stride: int=1) ->None:
        super(PreactResBlockBottleneck, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.pre = CondSeq()
        self.preact = CondSeq()
        self.branch = CondSeq()
        self.wide(divisor=4)
        self.shortcut = make_preact_resnet_shortcut(in_channels, out_channels, stride)
        if in_channels != out_channels:
            self.preact_skip()
        self.post = CondSeq()

    def remove_batchnorm(self) ->'PreactResBlockBottleneck':
        remove_batchnorm(self.branch)
        remove_batchnorm(self.shortcut)
        remove_batchnorm(self.preact)
        return self

    def condition(self, z: torch.Tensor) ->None:
        self.pre.condition(z)
        self.preact.condition(z)
        self.branch.condition(z)
        self.shortcut.condition(z)
        self.post.condition(z)

    def preact_skip(self) ->'PreactResBlockBottleneck':
        if hasattr(self.branch, 'bn1'):
            self.preact.add_module('bn1', cast(nn.Module, self.branch.bn1))
            del self.branch.bn1
        if hasattr(self.branch, 'relu'):
            self.preact.add_module('relu', cast(nn.Module, self.branch.relu))
            del self.branch.relu
        return self

    def forward(self, x: torch.Tensor, z: Optional[torch.Tensor]=None) ->torch.Tensor:
        if z is not None:
            self.condition(z)
        x = self.pre(x)
        x = self.preact(x)
        x = self.shortcut(x) + self.branch(x)
        return self.post(x)

    def use_se(self) ->'PreactResBlockBottleneck':
        self.branch.add_module('se', SEBlock(self.out_channels))
        return self

    def no_preact(self) ->'PreactResBlockBottleneck':
        mods = collections.OrderedDict([*self.preact.named_children(), *self.branch.named_children()])
        self.branch = CondSeq(mods)
        self.preact = CondSeq()
        return self

    def resnext(self, groups: int=32) ->'PreactResBlockBottleneck':
        self.wide(divisor=4)
        c = self.branch.conv2
        assert isinstance(c, nn.Conv2d)
        self.branch.conv2 = kaiming(nn.Conv2d(c.in_channels, c.out_channels, 3, groups=groups, stride=self.stride, padding=1, bias=False))
        return self

    def wide(self, divisor: int=2) ->'PreactResBlockBottleneck':
        in_ch = self.in_channels
        out_ch = self.out_channels
        stride = self.stride
        mid = out_ch // divisor
        self.branch = CondSeq(collections.OrderedDict([('bn1', constant_init(nn.BatchNorm2d(in_ch), 1)), ('relu', nn.ReLU(True)), ('conv1', kaiming(Conv1x1(in_ch, mid, bias=False))), ('bn2', constant_init(nn.BatchNorm2d(mid), 1)), ('relu2', nn.ReLU(True)), ('conv2', kaiming(Conv3x3(mid, mid, stride=stride, bias=False))), ('bn3', constant_init(nn.BatchNorm2d(mid), 1)), ('relu3', nn.ReLU(True)), ('conv3', constant_init(Conv1x1(mid, out_ch), 0))]))
        return self


class Lambda(nn.Module):
    """
    Applies a lambda function on forward()

    Args:
        lamb (fn): the lambda function
    """

    def __init__(self, lam):
        super(Lambda, self).__init__()
        self.lam = lam

    def forward(self, x):
        return self.lam(x)


class Reshape(nn.Module):
    """
    Reshape the input volume

    Args:
        *shape (ints): new shape, WITHOUT specifying batch size as first
        dimension, as it will remain unchanged.
    """

    def __init__(self, *shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.shape[0], *self.shape)


class LocalSelfAttentionHook(nn.Module):

    def forward(self, x, attn, pad):
        return x, attn, pad


def local_attention_2d(x: Tensor, conv_kqv: nn.Conv2d, posenc: Tensor, num_heads: int, patch_size: int) ->Tensor:
    B, inC, fullH, fullW = x.shape
    N = num_heads
    P = patch_size
    H, W = fullH // P, fullW // P
    x = x.view(B, inC, H, P, W, P).permute(0, 2, 4, 1, 3, 5).reshape(B * H * W, inC, P, P)
    k, q, v = torch.chunk(F.conv2d(x, conv_kqv.weight / math.sqrt(inC // N), conv_kqv.bias), 3, dim=1)
    hidC = k.shape[1] // N
    k = k.view(B, H * W, N, hidC, P * P)
    q = q.view(B, H * W, N, hidC, P * P)
    kq = torch.softmax(torch.matmul(q.transpose(-1, -2), k) + posenc, dim=-1)
    v = v.view(B, H * W, N, hidC, P * P)
    kqv = torch.matmul(v, kq.transpose(-1, -2)).view(B, H, W, N, hidC, P, P)
    kqv = kqv.permute(0, 3, 4, 1, 5, 2, 6).reshape(B, hidC * N, fullH, fullW)
    return kqv, kq


class VectorQuantization(Function):

    @staticmethod
    def compute_indices(inputs_orig, codebook):
        bi = []
        SZ = 10000
        for i in range(0, inputs_orig.size(0), SZ):
            inputs = inputs_orig[i:i + SZ]
            distances_matrix = torch.cdist(inputs, codebook)
            indic = torch.min(distances_matrix, dim=-1)[1].unsqueeze(1)
            bi.append(indic)
        return torch.cat(bi, dim=0)

    @staticmethod
    def flatten(x):
        code_dim = x.size(-1)
        return x.view(-1, code_dim)

    @staticmethod
    def restore_shapes(codes, indices, target_shape):
        idx_shape = list(target_shape)
        idx_shape[-1] = 1
        return codes.view(*target_shape), indices.view(*idx_shape)

    @staticmethod
    def forward(ctx, inputs, codebook, commitment=0.25, dim=1):
        inputs_flat = VectorQuantization.flatten(inputs)
        indices = VectorQuantization.compute_indices(inputs_flat, codebook)
        codes = codebook[indices.view(-1), :]
        codes, indices = VectorQuantization.restore_shapes(codes, indices, inputs.shape)
        ctx.save_for_backward(codes, inputs, torch.tensor([float(commitment)]), codebook, indices)
        ctx.mark_non_differentiable(indices)
        return codes, indices

    @staticmethod
    def backward(ctx, straight_through, unused_indices):
        codes, inputs, beta, codebook, indices = ctx.saved_tensors
        diff = 2 * (inputs - codes) / inputs.numel()
        commitment = beta.item() * diff
        code_disp = VectorQuantization.flatten(-diff)
        indices = VectorQuantization.flatten(indices)
        code_disp = torch.zeros_like(codebook).index_add_(0, indices.view(-1), code_disp)
        return straight_through + commitment, code_disp, None, None


quantize = VectorQuantization.apply


class VQ(nn.Module):
    """
    Quantization layer from *Neural Discrete Representation Learning*

    Args:
        latent_dim (int): number of features along which to quantize
        num_tokens (int): number of tokens in the codebook
        dim (int): dimension along which to quantize
        return_indices (bool): whether to return the indices of the quantized
            code points
    """
    embedding: nn.Embedding
    dim: int
    commitment: float
    initialized: torch.Tensor
    return_indices: bool
    init_mode: str

    def __init__(self, latent_dim: int, num_tokens: int, dim: int=1, commitment: float=0.25, init_mode: str='normal', return_indices: bool=True, max_age: int=1000):
        super(VQ, self).__init__()
        self.embedding = nn.Embedding(num_tokens, latent_dim)
        nn.init.normal_(self.embedding.weight, 0, 1.1)
        self.dim = dim
        self.commitment = commitment
        self.register_buffer('initialized', torch.Tensor([0]))
        self.return_indices = return_indices
        assert init_mode in ['normal', 'first']
        self.init_mode = init_mode
        self.register_buffer('age', torch.empty(num_tokens).fill_(max_age))
        self.max_age = max_age

    def update_usage(self, indices):
        with torch.no_grad():
            self.age += 1
            if torch.distributed.is_initialized():
                n_gpu = torch.distributed.get_world_size()
                all_indices = [torch.empty_like(indices) for _ in range(n_gpu)]
                torch.distributed.all_gather(all_indices, indices)
                indices = torch.cat(all_indices)
            used = torch.unique(indices)
            self.age[used] = 0

    def resample_dead(self, x):
        with torch.no_grad():
            dead = torch.nonzero(self.age > self.max_age, as_tuple=True)[0]
            if len(dead) == 0:
                return
            None
            x_flat = x.view(-1, x.shape[-1])
            emb_weight = self.embedding.weight.data
            emb_weight[dead[:len(x_flat)]] = x_flat[torch.randperm(len(x_flat))[:len(dead)]]
            self.age[dead[:len(x_flat)]] = 0
            if torch.distributed.is_initialized():
                torch.distributed.broadcast(emb_weight, 0)

    def forward(self, x: torch.Tensor) ->Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass

        Args:
            x (tensor): input tensor

        Returns:
            quantized tensor, or (quantized tensor, indices) if
            `self.return_indices`
        """
        dim = self.dim
        nb_codes = self.embedding.weight.shape[0]
        codebook = self.embedding.weight
        if self.init_mode == 'first' and self.initialized.item() == 0 and self.training:
            n_proto = self.embedding.weight.shape[0]
            ch_first = x.transpose(dim, -1).contiguous().view(-1, x.shape[dim])
            n_samples = ch_first.shape[0]
            idx = torch.randint(0, n_samples, (n_proto,))[:nb_codes]
            self.embedding.weight.data.copy_(ch_first[idx])
            self.initialized[:] = 1
        needs_transpose = dim != -1 or dim != x.dim() - 1
        if needs_transpose:
            x = x.transpose(-1, dim).contiguous()
        if self.training:
            self.resample_dead(x)
        codes, indices = quantize(x, codebook, self.commitment, self.dim)
        if self.training:
            self.update_usage(indices)
        if needs_transpose:
            codes = codes.transpose(-1, dim)
            indices = indices.transpose(-1, dim)
        if self.return_indices:
            return codes, indices
        else:
            return codes


class MultiVQ(nn.Module):
    """
    Multi codebooks quantization layer from *Neural Discrete Representation
    Learning*

    Args:
        latent_dim (int): number of features along which to quantize
        num_tokens (int): number of tokens in the codebook
        num_codebooks (int): number of parallel codebooks
        dim (int): dimension along which to quantize
            an angular distance
        return_indices (bool): whether to return the indices of the quantized
            code points
    """

    def __init__(self, latent_dim: int, num_tokens: int, num_codebooks: int, dim: int=1, commitment: float=0.25, init_mode: str='normal', return_indices: bool=True, max_age: int=1000):
        assert latent_dim % num_codebooks == 0, 'num_codebooks must divide evenly latent_dim'
        super(MultiVQ, self).__init__()
        self.dim = dim
        self.num_codebooks = num_codebooks
        self.return_indices = return_indices
        self.vqs = nn.ModuleList([VQ(latent_dim // num_codebooks, num_tokens, dim=dim, commitment=commitment, init_mode=init_mode, return_indices=return_indices, max_age=max_age) for _ in range(num_codebooks)])

    def forward(self, x: torch.Tensor) ->Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x_chunks = torch.chunk(x, self.num_codebooks, dim=self.dim)
        quantized = [vq(chunk) for chunk, vq in zip(x_chunks, self.vqs)]
        if self.return_indices:
            q = torch.cat([q[0] for q in quantized], dim=self.dim)
            return q, torch.cat([q[1] for q in quantized], dim=self.dim)
        else:
            return torch.cat(quantized, dim=self.dim)


class BinomialFilter2d(torch.nn.Module):

    def __init__(self, stride: int):
        super().__init__()
        self.stride = stride
        self.register_buffer('weight', torch.tensor([[[1.0, 2, 1], [2, 4, 2], [1, 2, 1]]]) / 16)

    def forward(self, x):
        x = torch.nn.functional.pad(x, (1, 1, 1, 1), mode='replicate')
        return torch.nn.functional.conv2d(x, self.weight.expand(x.shape[1], 1, -1, -1), groups=x.shape[1], stride=self.stride, padding=0)


class DeepDreamOptim(Optimizer):
    """Optimizer used by Deep Dream. It rescales the gradient by the average of
    the absolute values of the gradient.

    :math:`\\theta_i := \\theta_i - lr \\frac{g_i}{\\epsilon+\\frac{1}{M}\\sum_j^M |g_j|}`

    Args:
        params: parameters as expected by Pytorch's optimizers
        lr (float): the learning rate
        eps (float): epsilon value to avoid dividing by zero
    """

    def __init__(self, params, lr=0.001, eps=1e-08, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if not 0.0 <= eps:
            raise ValueError('Invalid epsilon value: {}'.format(eps))
        defaults = dict(lr=lr, eps=eps, weight_decay=weight_decay)
        super(DeepDreamOptim, self).__init__(params, defaults)

    def step(self, closure=None):
        """Update the weights

        Args:
            closure (optional fn): a function that computes gradients
        """
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.data.mul_(1 - group['lr'] * group['weight_decay'])
                step_size = group['lr']
                eps = group['eps']
                p.data.add_(p.grad.data, alpha=-step_size / (eps + p.grad.data.abs().mean()))
        return loss


class CallbacksRunner:

    def __init__(self):
        self.cbs = [[], [], []]
        self.reset()

    def reset(self):
        self.state = {'metrics': {}}

    def __call__(self, name, *args, **kwargs):
        for cb in self.callbacks():
            if hasattr(cb, name):
                getattr(cb, name)(self.state, *args, **kwargs)

    def callbacks(self):
        for cbs in self.cbs:
            for cb in cbs:
                yield cb

    def named_callbacks(self):
        for step, cbs in zip(['prologue', 'middle', 'epilogue'], self.cbs):
            counts = defaultdict(int)
            for cb in cbs:
                nm = cb.__class__.__name__
                cnt = counts[nm]
                counts[nm] += 1
                yield '_'.join([nm, step, str(cnt)]), cb

    def state_dict(self):
        serial_cb = {}
        for nm, cb in self.named_callbacks():
            if hasattr(cb, 'state_dict'):
                serial_cb[nm] = cb.state_dict()
        return {'state': {k: v for k, v in self.state.items() if k[0] != '_'}, 'callbacks': serial_cb}

    def load_state_dict(self, dicc):
        self.state = dicc['state']
        for nm, cb in self.named_callbacks():
            if hasattr(cb, 'load_state_dict') and nm in dicc['callbacks']:
                cb.load_state_dict(dicc['callbacks'][nm])

    def update_state(self, state_additions):
        self.state.update(state_additions)

    def add_prologue(self, cb):
        self.cbs[0].append(cb)

    def add_callback(self, cb):
        self.cbs[1].append(cb)

    def add_epilogue(self, cb):
        self.cbs[2].append(cb)

    def add_prologues(self, cbs):
        for cb in cbs:
            self.add_prologue(cb)

    def add_callbacks(self, cbs):
        for cb in cbs:
            self.add_callback(cb)

    def add_epilogues(self, cbs):
        for cb in cbs:
            self.add_epilogue(cb)

    def __repr__(self):
        return 'Prologue:\n{}\nCallbacks:\n{}\nEpilogue:\n{}'.format(tu.indent('\n'.join([repr(c) for c in self.cbs[0]])), tu.indent('\n'.join([repr(c) for c in self.cbs[1]])), tu.indent('\n'.join([repr(c) for c in self.cbs[2]])))


class RecipeBase:

    def __init__(self):
        self._modules = set()
        self._savable = set()
        self.device = 'cpu'

    def __repr__(self) ->str:
        return self.__class__.__name__ + ':\n' + tu.indent('Modules:\n' + tu.indent('\n'.join([(m + ':\n' + tu.indent(repr(getattr(self, m)))) for m in self._modules])) + '\n' + 'Savables:\n' + tu.indent('\n'.join([(m + ':\n' + tu.indent(repr(getattr(self, m)))) for m in self._savable])))

    def _check_init(self):
        if '_modules' not in self.__dict__:
            raise AttributeError('You forgot to call RecipeBase.__init__()')

    def register(self, name, value):
        """
        Register an object into the recipe as a member. Calling
        :code:`recipe.register('foo', bar)` registers bar, and makes it usable
        through :code:`recipe.foo`.

        Args:
            name (str): member's name
            value: the object to register
        """
        self._check_init()
        self._modules.discard(name)
        self._savable.discard(name)
        if isinstance(value, torch.nn.Module):
            self._modules.add(name)
        else:
            self._savable.add(name)
        self.__dict__[name] = value

    def state_dict(self):
        """
        Returns:
            A state dict
        """
        sd = OrderedDict()
        for nm in self._modules:
            mod = self.__dict__[nm]
            if isinstance(mod, (torch.nn.parallel.DistributedDataParallel, torch.nn.parallel.DataParallel)):
                sd[nm] = mod.module.state_dict()
            else:
                sd[nm] = mod.state_dict()
        for nm in self._savable:
            val = self.__dict__[nm]
            if hasattr(val, 'state_dict'):
                sd[nm] = val.state_dict()
            else:
                sd[nm] = val
        return sd

    def load_state_dict(self, state_dict):
        """
        Restore a recipe
        """
        for key, state in state_dict.items():
            val = self.__dict__[key]
            if hasattr(val, 'load_state_dict'):
                if isinstance(val, torch.nn.Module):
                    if isinstance(val, (torch.nn.parallel.DistributedDataParallel, torch.nn.parallel.DataParallel)):
                        tu.load_state_dict_forgiving(val.module, state)
                    else:
                        tu.load_state_dict_forgiving(val, state)
                else:
                    val.load_state_dict(state)
            else:
                self.__dict__[key] = val
        return self

    def modules(self):
        """
        Iterate over all nn.Modules registered in the recipe
        """
        for m in self._modules:
            yield self.__dict__[m]

    def to(self, device):
        """
        Move a recipe and all its movable registered objects to a device

        Args:
            device: a torch device
        """
        self._check_init()
        self.device = device
        for m in self.modules():
            m
        for nm in self._savable:
            m = self.__dict__[nm]
            if hasattr(m, 'to'):
                self.__dict__[nm] = m
        return self

    def cuda(self):
        """
        Move a recipe and all its movable registered objects to cuda
        """
        return self

    def cpu(self):
        """
        Move a recipe and all its movable registered objects to cpu
        """
        return self


class Recipe(RecipeBase):
    """
    Basic recipe that iterates mutiple epochs over a dataset. That loop is
    instrumented through several configurable callbacks. Callbacks can handle
    events before and after each batch, before and after each epoch. Each batch
    is treated with a user supplied function that manipulates it and returns a
    dict that returns a state usable by the callbacks.

    A recipe can be saved by calling its :code:`state_dict()` member. All its
    hyper parameters and state will be saved so that it can restart, but
    requires the exact same setting of callbacks.

    You can register multiple objects to a Recipe with :code:`register()`. If
    it has a :code:`state_dict()` member, its state will be saved into the
    recipe's when calling the recipe's :code:`state_dict()` function. If it
    has a member :code:`to()`, moving the recipe to another device will also
    move those objects.

    Args:
        call_fun (Callable): A function that takes a batch as an argument and
            returns a dict of value to feed the state
        loader (Iterable): any iterable (most likely a DataLoader)
    """

    def __init__(self, call_fun, loader):
        super(Recipe, self).__init__()
        self.call_fun = call_fun
        self.loader = loader
        self.callbacks = CallbacksRunner()
        self.register('callbacks', self.callbacks)
        self.callbacks.update_state({'_loader': loader})

    def run(self, epochs):
        """
        Run the recipe for :code:`epochs` epochs.

        Args:
            epochs (int): number of epochs

        Returns:
            The state
        """
        self
        for epoch in range(epochs):
            self.callbacks('on_epoch_start')
            for batch in self.loader:
                self.callbacks.update_state({'batch': batch})
                batch = tu.send_to_device(batch, self.device, non_blocking=True)
                self.callbacks.update_state({'_batch_gpu': batch})
                self.callbacks('on_batch_start')
                out = self.call_fun(batch)
                out = tu.send_to_device(out, 'cpu', non_blocking=False)
                self.callbacks.update_state(out)
                self.callbacks('on_batch_end')
            self.callbacks('on_epoch_end')
        return self.callbacks.state


class DeepDream(torch.nn.Module):
    """
    Deep Dream recipe

    First instantiate the recipe then call `recipe(n_iter, img)`

    Args:
        model (nn.Module): the trained model to use
        dream_layer (str): the layer to use on which activations will be
            maximized
    """

    def __init__(self, model, dream_layer):
        super(DeepDream, self).__init__()
        self.loss = DeepDreamLoss(model, dream_layer)
        self.norm = tnn.ImageNetInputNorm()

    def fit(self, ref, iters, lr=0.0003, device='cpu', visdom_env='deepdream'):
        """
        Args:
            lr (float, optional): the learning rate
            visdom_env (str or None): the name of the visdom env to use, or None
                to disable Visdom
        """
        ref_tensor = TF.ToTensor()(ref).unsqueeze(0)
        canvas = ParameterizedImg(1, 3, ref_tensor.shape[2], ref_tensor.shape[3], init_img=ref_tensor, space='spectral', colors='uncorr')

        def forward(_):
            img = canvas()
            rnd = random.randint(0, 10)
            loss = self.loss(self.norm(img[:, :, rnd:, rnd:]))
            loss.backward()
            return {'loss': loss, 'img': img}
        loop = Recipe(forward, range(iters))
        loop.register('model', self)
        loop.register('canvas', canvas)
        loop.callbacks.add_callbacks([tcb.Counter(), tcb.Log('loss', 'loss'), tcb.Log('img', 'img'), tcb.Optimizer(DeepDreamOptim(canvas.parameters(), lr=lr)), tcb.VisdomLogger(visdom_env=visdom_env, log_every=10), tcb.StdoutLogger(log_every=10)])
        loop
        loop.run(1)
        return canvas.render().cpu()


class FeatureVis(torch.nn.Module):
    """
    Feature viz

    First instantiate the recipe then call `recipe(n_iter, img)`

    Args:
        model (nn.Module): the trained model to use
        layer (str): the layer to use on which activations will be maximized
        input_size (int, or (int, int)): the size of the image the model
            accepts as input
        num_feature (int): the number of channels of the input image (e.g 1 for grey, 3 for RGB)
        lr (float, optional): the learning rate
        device (device): where to run the computation
        visdom_env (str or None): the name of the visdom env to use, or None
            to disable Visdom
    """

    def __init__(self, model, layer, input_size, *, num_feature=3, lr=0.001, device='cpu', visdom_env='feature_vis'):
        super().__init__()
        self.device = device
        self.model = tnn.WithSavedActivations(model, names=[layer])
        self.layer = layer
        if isinstance(input_size, (list, tuple)):
            self.input_size = input_size
        else:
            self.input_size = input_size, input_size
        self.num_feature = num_feature
        self.norm = tnn.ImageNetInputNorm() if num_feature == 3 else torch.nn.InstanceNorm2d(num_feature, momentum=0)
        self.lr = lr
        self.visdom_env = visdom_env

    def fit(self, n_iters, neuron):
        """
        Run the recipe

        Args:
            n_iters (int): number of iterations to run
            neuron (int): the feature map to maximize

        Returns:
            the optimized image
        """
        canvas = ParameterizedImg(1, self.num_feature, self.input_size[0] + 10, self.input_size[1] + 10, colors='corr' if self.num_feature != 3 else 'uncorr')

        def forward(_):
            cim = canvas()
            rnd = random.randint(0, cim.shape[2] // 10)
            im = cim[:, :, rnd:, rnd:]
            im = torch.nn.functional.interpolate(im, size=(self.input_size[0], self.input_size[1]), mode='bilinear')
            _, acts = self.model(self.norm(im), detach=False)
            fmap = acts[self.layer]
            loss = -fmap[0][neuron].sum()
            loss.backward()
            return {'loss': loss, 'img': cim}
        loop = Recipe(forward, range(n_iters))
        loop.register('canvas', canvas)
        loop.register('model', self)
        loop.callbacks.add_callbacks([tcb.Counter(), tcb.Log('loss', 'loss'), tcb.Log('img', 'img'), tcb.Optimizer(DeepDreamOptim(canvas.parameters(), lr=self.lr)), tcb.VisdomLogger(visdom_env=self.visdom_env, log_every=10), tcb.StdoutLogger(log_every=10)])
        loop
        loop.run(1)
        return canvas.render().cpu()


class NeuralStyle(torch.nn.Module):
    """
    Neural Style Recipe

    First instantiate the recipe then call `recipe(n_iter, img)`

    Args:
        device (device): where to run the computation
        visdom_env (str or None): the name of the visdom env to use, or None
            to disable Visdom
    """

    def __init__(self, device='cpu', visdom_env='style'):
        super(NeuralStyle, self).__init__()
        self.loss = NeuralStyleLoss()
        self.loss2 = NeuralStyleLoss()
        self.device = device
        self.visdom_env = visdom_env

    def fit(self, iters, content_img, style_img, style_ratio, *, second_scale_ratio=1, content_layers=None, init_with_content=False):
        """
        Run the recipe

        Args:
            n_iters (int): number of iterations to run
            content (PIL.Image): content image
            style (PIL.Image): style image
            ratio (float): weight of style loss
            content_layers (list of str): layers on which to reconstruct
                content
        """
        self.loss
        self.loss.set_style(to_tensor(style_img)[None], style_ratio)
        self.loss.set_content(to_tensor(content_img)[None], content_layers)
        self.loss2
        self.loss2.set_style(torch.nn.functional.interpolate(to_tensor(style_img)[None], scale_factor=0.5, mode='bilinear', align_corners=False, recompute_scale_factor=True), style_ratio)
        self.loss2.set_content(torch.nn.functional.interpolate(to_tensor(content_img)[None], scale_factor=0.5, mode='bilinear', align_corners=False, recompute_scale_factor=True), content_layers)
        canvas = ParameterizedImg(1, 3, content_img.height, content_img.width, init_img=to_tensor(content_img)[None] if init_with_content else None)
        canvas
        self.opt = tch.optim.Lookahead(torch.optim.Adam(canvas.parameters(), 0.05, weight_decay=0.0))

        def forward(_):
            img = canvas()
            loss, losses = self.loss(img)
            loss.backward()
            loss, losses = self.loss2(torch.nn.functional.interpolate(canvas(), scale_factor=0.5, mode='bilinear', align_corners=False, recompute_scale_factor=True))
            (second_scale_ratio * loss).backward()
            return {'loss': loss, 'content': losses['content'], 'style': losses['style'], 'img': img}
        loop = Recipe(forward, range(iters))
        loop.register('canvas', canvas)
        loop.register('model', self)
        loop.callbacks.add_callbacks([tcb.Counter(), tcb.WindowedMetricAvg('loss'), tcb.WindowedMetricAvg('content'), tcb.WindowedMetricAvg('style'), tcb.Log('img', 'img'), tcb.VisdomLogger(visdom_env=self.visdom_env, log_every=10), tcb.StdoutLogger(log_every=10), tcb.Optimizer(self.opt)])
        loop
        loop.run(1)
        return canvas.render().cpu()


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval=None, eigvec=None):
        self.alphastd = alphastd
        self.eigval = torch.Tensor([0.2175, 0.0188, 0.0045])
        self.eigvec = torch.Tensor([[-0.5675, 0.7192, 0.4009], [-0.5808, -0.0045, -0.814], [-0.5836, -0.6948, 0.4203]])
        if eigval is not None:
            self.eigval = eigval
        if eigvec is not None:
            self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img
        is_pil = False
        if isinstance(img, PILImage):
            img = F.to_tensor(img)
            is_pil = True
        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone().mul(alpha.view(1, 3).expand(3, 3)).mul(self.eigval.view(1, 3).expand(3, 3)).sum(1).squeeze()
        img = img.add(rgb.view(3, 1, 1).expand_as(img)).clamp_(0, 1)
        if is_pil:
            img = F.to_pil_image(img)
        return img


def indent(text: str, amount: int=4) ->str:
    """
    Indent :code:`text` by :code:`amount` spaces.

    Args:
        text (str): some text
        amount (int): an indentation amount

    Returns:
        indented text
    """
    return '\n'.join(' ' * amount + line for line in text.splitlines())


def freeze(net: T_Module) ->T_Module:
    """
    Freeze all parameters of `net`
    """
    for p in net.parameters():
        p.requires_grad_(False)
    return net


class FrozenModule(nn.Module):
    """
    Wrap a module to eval model, can't be turned back to training mode

    Args:
        m (nn.Module): a module
    """

    def __init__(self, m: nn.Module) ->None:
        super(FrozenModule, self).__init__()
        self.m = freeze(m).eval()

    def train(self, mode: bool=True) ->'FrozenModule':
        return self

    def __getattr__(self, name):
        None
        return getattr(super(FrozenModule, self).__getattr__('m'), name)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AdaptiveConcatPool2d,
     lambda: ([], {'target_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (AttenNorm2d,
     lambda: ([], {'num_features': 4, 'num_weights': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (BinomialFilter2d,
     lambda: ([], {'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CondSeq,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Debug,
     lambda: ([], {'name': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DropPath,
     lambda: ([], {'p': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Dummy,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FactoredPredictor,
     lambda: ([], {'hid_ch': 4, 'out_ch': 4, 'n_pred': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (GhostBatchNorm2d,
     lambda: ([], {'num_features': 4, 'ghost_batch_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (HardSigmoid,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (HardSwish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ImageNetInputNorm,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 4, 4])], {}),
     True),
    (Lambda,
     lambda: ([], {'lam': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LayerScale,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (LocalSelfAttentionHook,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (MaskedConv2d,
     lambda: ([], {'in_chan': 4, 'out_chan': 4, 'ks': 4, 'center': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MinibatchStddev,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MultiScaleDiscriminator,
     lambda: ([], {'base_model': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MultiVQ,
     lambda: ([], {'latent_dim': 4, 'num_tokens': 4, 'num_codebooks': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (OrthoLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (PixelImage,
     lambda: ([], {'shape': [4, 4, 4, 4]}),
     lambda: ([], {}),
     False),
    (PixelNorm,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PixelPredictor,
     lambda: ([], {'hid_ch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (PreactResBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (PreactResBlockBottleneck,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (RGB,
     lambda: ([], {}),
     lambda: ([], {'x': 4}),
     True),
    (ResBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ResBlockBottleneck,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Reshape,
     lambda: ([], {}),
     lambda: ([torch.rand([4])], {}),
     True),
    (SpectralImage,
     lambda: ([], {'shape': [4, 4, 4, 4]}),
     lambda: ([], {}),
     False),
    (TemperedCrossEntropyLoss,
     lambda: ([], {'t1': 4, 't2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.ones([4], dtype=torch.int64)], {}),
     False),
    (VQ,
     lambda: ([], {'latent_dim': 4, 'num_tokens': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_Vermeille_Torchelie(_paritybench_base):
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

    def test_013(self):
        self._check(*TESTCASES[13])

    def test_014(self):
        self._check(*TESTCASES[14])

    def test_015(self):
        self._check(*TESTCASES[15])

    def test_016(self):
        self._check(*TESTCASES[16])

    def test_017(self):
        self._check(*TESTCASES[17])

    def test_018(self):
        self._check(*TESTCASES[18])

    def test_019(self):
        self._check(*TESTCASES[19])

    def test_020(self):
        self._check(*TESTCASES[20])

    def test_021(self):
        self._check(*TESTCASES[21])

    def test_022(self):
        self._check(*TESTCASES[22])

    def test_023(self):
        self._check(*TESTCASES[23])

    def test_024(self):
        self._check(*TESTCASES[24])

    def test_025(self):
        self._check(*TESTCASES[25])

    def test_026(self):
        self._check(*TESTCASES[26])

    def test_027(self):
        self._check(*TESTCASES[27])

    def test_028(self):
        self._check(*TESTCASES[28])

    def test_029(self):
        self._check(*TESTCASES[29])

    def test_030(self):
        self._check(*TESTCASES[30])

    def test_031(self):
        self._check(*TESTCASES[31])

