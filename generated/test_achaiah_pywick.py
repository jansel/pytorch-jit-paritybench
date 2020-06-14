import sys
_module = sys.modules[__name__]
del sys
conf = _module
mnist_example = _module
mnist_loader_example = _module
pywick = _module
CSVLogger = _module
Callback = _module
CallbackContainer = _module
CyclicLRScheduler = _module
EarlyStopping = _module
ExperimentLogger = _module
History = _module
LRScheduler = _module
LambdaCallback = _module
ModelCheckpoint = _module
ReduceLROnPlateau = _module
SimpleModelCheckpoint = _module
TQDM = _module
callbacks = _module
conditions = _module
constraints = _module
custom_regularizers = _module
data_stats = _module
BaseDataset = _module
CSVDataset = _module
ClonedFolderDataset = _module
FolderDataset = _module
MultiFolderDataset = _module
PredictFolderDataset = _module
TensorDataset = _module
UsefulDataset = _module
datasets = _module
data_utils = _module
tnt = _module
batchdataset = _module
concatdataset = _module
dataset = _module
listdataset = _module
multipartitiondataset = _module
resampledataset = _module
shuffledataset = _module
splitdataset = _module
table = _module
transform = _module
transformdataset = _module
functions = _module
activations_autofn = _module
activations_jit = _module
affine = _module
batchrenorm = _module
cyclicLR = _module
group_norm = _module
mish = _module
swish = _module
gridsearch = _module
grid_test = _module
pipeline = _module
image_utils = _module
initializers = _module
losses = _module
lovasz_losses = _module
meters = _module
apmeter = _module
aucmeter = _module
averagemeter = _module
averagevaluemeter = _module
classerrormeter = _module
confusionmeter = _module
mapmeter = _module
meter = _module
movingaveragevaluemeter = _module
msemeter = _module
timemeter = _module
metrics = _module
misc = _module
models = _module
classification = _module
bninception = _module
dpn = _module
adaptive_avgmax_pool = _module
convert_from_mxnet = _module
dualpath = _module
model_factory = _module
fbresnet = _module
inception_resv2_wide = _module
inceptionresnetv2 = _module
inceptionv4 = _module
nasnet = _module
nasnet_mobile = _module
pnasnet = _module
polynet = _module
pyramid_resnet = _module
resnet_preact = _module
resnet_swish = _module
resnext = _module
resnext_features = _module
resnext101_32x4d_features = _module
resnext101_64x4d_features = _module
resnext50_32x4d_features = _module
senet = _module
testnets = _module
large_densenet = _module
opt_densenset = _module
pnn = _module
se_densenet_full = _module
se_efficient_densenet = _module
se_module = _module
wideresnet = _module
xception = _module
localization = _module
fpn = _module
retina_fpn = _module
model_locations = _module
model_utils = _module
segmentation = _module
bisenet = _module
carvana_unet = _module
config = _module
da_basenets = _module
basic = _module
densenet = _module
download = _module
fcn = _module
jpu = _module
model_store = _module
resnet = _module
resnetv1b = _module
segbase = _module
vgg = _module
danet = _module
deeplab_v2_res = _module
deeplab_v3 = _module
deeplab_v3_plus = _module
denseaspp = _module
drn = _module
drn_seg = _module
duc_hdc = _module
dunet = _module
enet = _module
fcn16s = _module
fcn32s = _module
fcn8s = _module
fcn_utils = _module
frrn = _module
fusionnet = _module
gcnnets = _module
gcn = _module
gcn_densenet = _module
gcn_nasnet = _module
gcn_psp = _module
gcn_resnext = _module
lex_extractors = _module
lexpsp = _module
mnas_linknets = _module
decoder = _module
inception4 = _module
inception_resnet = _module
linknet = _module
resnext = _module
resnext101_32x4d_features = _module
ocnet = _module
refinenet = _module
blocks = _module
refinenet = _module
resnet_gcn = _module
seg_net = _module
Unet_nested = _module
Unet_nested_layers = _module
autofocusNN = _module
dabnet = _module
deeplabv2 = _module
deeplabv3 = _module
deeplabv3_resnet = _module
deeplabv3_xception = _module
densenet_se_seg = _module
difnet = _module
dilated_resnet = _module
encnet = _module
esp_net = _module
ExFuseLayer = _module
UnetExFuse = _module
exfuse = _module
unet_layer = _module
fc_densenet = _module
flatten = _module
lg_kernel_exfuse = _module
deeplab_resnet = _module
large_kernel = _module
large_kernel_exfuse = _module
seg_resnet = _module
seg_resnext = _module
mixnet = _module
layers = _module
mdconv = _module
mixnet = _module
utils = _module
msc = _module
psanet = _module
psp_saeed = _module
resnet = _module
tiramisu_test = _module
tkcnet = _module
base = _module
files = _module
resnet = _module
tkcnet = _module
unet_plus_plus = _module
unet_se = _module
tiramisu = _module
u_net = _module
unet_dilated = _module
unet_res = _module
unet_stack = _module
modules = _module
_utils = _module
module_trainer = _module
stn = _module
optimizers = _module
adamw = _module
addsign = _module
eve = _module
lookahead = _module
nadam = _module
powersign = _module
radam = _module
ralamb = _module
rangerlars = _module
sgdw = _module
sign_internal_decay = _module
swa = _module
regularizers = _module
samplers = _module
transforms = _module
affine_transforms = _module
distortion_transforms = _module
image_transforms = _module
tensor_transforms = _module
setup = _module
multi_input_multi_target = _module
single_input_multi_target = _module
single_input_single_target = _module
simple_multi_input_multi_target = _module
simple_multi_input_no_target = _module
simple_multi_input_single_target = _module
single_input_multi_target = _module
single_input_no_target = _module
single_input_single_target = _module
test_meters = _module
test_affine_transforms = _module
test_image_transforms = _module
test_tensor_transforms = _module

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


import torch as th


import torch.nn as nn


import torch.nn.functional as F


from torch.utils.data import DataLoader


import math


import torch


import numpy as np


from torch import nn as nn


from torch.nn import functional as F


from torch.nn.modules import Module


from torch.nn.parameter import Parameter


from torch.nn.modules.batchnorm import _BatchNorm


import torch.nn.init


from torch.autograd import Function


from torch.autograd import Variable


from torch import Tensor


from typing import Iterable


from typing import Set


import torch.utils.model_zoo as model_zoo


from collections import OrderedDict


from torch.utils import model_zoo


from functools import reduce


from torch import nn


from torch.utils.checkpoint import checkpoint_sequential


import re


import torch.utils.checkpoint as cp


from enum import Enum


from math import ceil


import torch.nn.init as init


from math import floor


from torch.nn import init


import torch.functional as F


from typing import Optional


from typing import Sequence


from typing import Union


from torch.nn import Module


from torch.nn import Conv2d


from torch.nn import BatchNorm2d


from torch.nn import Linear


from torch.nn.functional import upsample


import warnings


import torch.optim as optim


import functools


import torch.backends.cudnn as cudnn


from collections import defaultdict


from torch.optim.optimizer import Optimizer


class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(1600, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 1600)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(1600, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 1600)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


def laplace():
    return np.array([[0.25, 0.5, 0.25], [0.5, -3.0, 0.5], [0.25, 0.5, 0.25]]
        ).astype(np.float32)[None, None, ...]


class Laplace(nn.Module):
    """
    Laplace filter for a stack of data.
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3, bias=False, padding=1)
        self.conv.weight.data.copy_(torch.from_numpy(laplace()))
        self.conv.weight.requires_grad = False

    def forward(self, x):
        return self.conv(x)


def laplace3d():
    l = np.zeros((3, 3, 3))
    l[1, 1, 1] = -6.0
    l[1, 1, 2] = 1.0
    l[1, 1, 0] = 1.0
    l[1, 0, 1] = 1.0
    l[1, 2, 1] = 1.0
    l[0, 1, 1] = 1.0
    l[2, 1, 1] = 1.0
    return l.astype(np.float32)[None, None, ...]


class Laplace3D(nn.Module):
    """
    Laplace filter for a stack of data.
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(1, 1, 3, bias=False, padding=1)
        self.conv.weight.data.copy_(torch.from_numpy(laplace3d()))
        self.conv.weight.requires_grad = False

    def forward(self, x):
        return self.conv(x)


class LaplaceL2(nn.Module):
    """
    Laplace regularizer for a 2D convolutional layer.
    """

    def __init__(self):
        super().__init__()
        self.laplace = Laplace()

    def forward(self, x):
        ic, oc, k1, k2 = x.size()
        return self.laplace(x.view(ic * oc, 1, k1, k2)).pow(2).mean() / 2


class LaplaceL23D(nn.Module):
    """
    Laplace regularizer for a 2D convolutional layer.
    """

    def __init__(self):
        super().__init__()
        self.laplace = Laplace3D()

    def forward(self, x):
        ic, oc, k1, k2, k3 = x.size()
        return self.laplace(x.view(ic * oc, 1, k1, k2, k3)).pow(2).mean() / 2


class LaplaceL1(nn.Module):
    """
    Laplace regularizer for a 2D convolutional layer.
    """

    def __init__(self):
        super().__init__()
        self.laplace = Laplace()

    def forward(self, x):
        ic, oc, k1, k2 = x.size()
        return self.laplace(x.view(ic * oc, 1, k1, k2)).abs().mean()


class SwishAutoFn(torch.autograd.Function):
    """Swish - Described in: https://arxiv.org/abs/1710.05941
    Memory efficient variant from:
     https://medium.com/the-artificial-impostor/more-memory-efficient-swish-activation-function-e07c22c12a76
    """

    @staticmethod
    def forward(ctx, x):
        result = x.mul(torch.sigmoid(x))
        ctx.save_for_backward(x)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        x_sigmoid = torch.sigmoid(x)
        return grad_output.mul(x_sigmoid * (1 + x * (1 - x_sigmoid)))


class SwishAuto(nn.Module):

    def __init__(self, inplace: bool=False):
        super(SwishAuto, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return SwishAutoFn.apply(x)


class MishAutoFn(torch.autograd.Function):
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function - https://arxiv.org/abs/1908.08681
    Experimental memory-efficient variant
    """

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        y = x.mul(torch.tanh(F.softplus(x)))
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        x_sigmoid = torch.sigmoid(x)
        x_tanh_sp = F.softplus(x).tanh()
        return grad_output.mul(x_tanh_sp + x * x_sigmoid * (1 - x_tanh_sp *
            x_tanh_sp))


class MishAuto(nn.Module):

    def __init__(self, inplace: bool=False):
        super(MishAuto, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return MishAutoFn.apply(x)


@torch.jit.script
def swish_jit_bwd(x, grad_output):
    x_sigmoid = torch.sigmoid(x)
    return grad_output * (x_sigmoid * (1 + x * (1 - x_sigmoid)))


@torch.jit.script
def swish_jit_fwd(x):
    return x.mul(torch.sigmoid(x))


class SwishJitAutoFn(torch.autograd.Function):
    """ torch.jit.script optimised Swish
    Inspired by conversation btw Jeremy Howard & Adam Pazske
    https://twitter.com/jeremyphoward/status/1188251041835315200
    """

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return swish_jit_fwd(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        return swish_jit_bwd(x, grad_output)


class SwishJit(nn.Module):

    def __init__(self, inplace: bool=False):
        super(SwishJit, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return SwishJitAutoFn.apply(x)


class BatchReNorm1d(Module):

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True,
        rmax=3.0, dmax=5.0):
        super(BatchReNorm1d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.rmax = rmax
        self.dmax = dmax
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('r', torch.ones(num_features))
        self.register_buffer('d', torch.zeros(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.r.fill_(1)
        self.d.zero_()
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.size(1) != self.running_mean.nelement():
            raise ValueError('got {}-feature tensor, expected {}'.format(
                input.size(1), self.num_features))

    def forward(self, input):
        self._check_input_dim(input)
        n = input.size()[0]
        if self.training:
            mean = torch.mean(input, dim=0)
            sum = torch.sum((input - mean.expand_as(input)) ** 2, dim=0)
            if sum == 0 and self.eps == 0:
                invstd = 0.0
            else:
                invstd = 1.0 / torch.sqrt(sum / n + self.eps)
            unbiased_var = sum / (n - 1)
            self.r = torch.clamp(torch.sqrt(unbiased_var).data / torch.sqrt
                (self.running_var), 1.0 / self.rmax, self.rmax)
            self.d = torch.clamp((mean.data - self.running_mean) / torch.
                sqrt(self.running_var), -self.dmax, self.dmax)
            r = self.r.expand_as(input)
            d = self.d.expand_as(input)
            input_normalized = (input - mean.expand_as(input)
                ) * invstd.expand_as(input)
            input_normalized = input_normalized * r + d
            self.running_mean += self.momentum * (mean.data - self.running_mean
                )
            self.running_var += self.momentum * (unbiased_var.data - self.
                running_var)
            if not self.affine:
                return input_normalized
            output = input_normalized * self.weight.expand_as(input)
            output += self.bias.unsqueeze(0).expand_as(input)
            return output
        else:
            mean = self.running_mean.expand_as(input)
            invstd = 1.0 / torch.sqrt(self.running_var.expand_as(input) +
                self.eps)
            input_normalized = (input - mean.expand_as(input)
                ) * invstd.expand_as(input)
            if not self.affine:
                return input_normalized
            output = input_normalized * self.weight.expand_as(input)
            output += self.bias.unsqueeze(0).expand_as(input)
            return output

    def __repr__(self):
        return (
            '{name}({num_features}, eps={eps}, momentum={momentum},affine={affine}, rmax={rmax}, dmax={dmax})'
            .format(name=self.__class__.__name__, **self.__dict__))


def group_norm(input, group, running_mean, running_var, weight=None, bias=
    None, use_input_stats=True, momentum=0.1, eps=1e-05):
    """Applies Group Normalization for channels in the same group in each data sample in a
    batch.

    See :class:`~torch.nn.GroupNorm2d`, for details.
    """
    if not use_input_stats and (running_mean is None or running_var is None):
        raise ValueError(
            'Expected running_mean and running_var to be not None when use_input_stats=False'
            )
    b, c = input.size(0), input.size(1)
    if weight is not None:
        weight = weight.repeat(b)
    if bias is not None:
        bias = bias.repeat(b)

    def _group_norm(input, group, running_mean=None, running_var=None,
        weight=None, bias=None, use_input_stats=None, momentum=None, eps=None):
        if running_mean is not None:
            running_mean_orig = running_mean
            running_mean = running_mean_orig.repeat(b)
        if running_var is not None:
            running_var_orig = running_var
            running_var = running_var_orig.repeat(b)
        input_reshaped = input.contiguous().view(1, int(b * c / group),
            group, *input.size()[2:])
        out = F.batch_norm(input_reshaped, running_mean, running_var,
            weight=weight, bias=bias, training=use_input_stats, momentum=
            momentum, eps=eps)
        if running_mean is not None:
            running_mean_orig.copy_(running_mean.view(b, int(c / group)).
                mean(0, keepdim=False))
        if running_var is not None:
            running_var_orig.copy_(running_var.view(b, int(c / group)).mean
                (0, keepdim=False))
        return out.view(b, c, *input.size()[2:])
    return _group_norm(input, group, running_mean=running_mean, running_var
        =running_var, weight=weight, bias=bias, use_input_stats=
        use_input_stats, momentum=momentum, eps=eps)


class _GroupNorm(_BatchNorm):

    def __init__(self, num_features, num_groups=1, eps=1e-05, momentum=0.1,
        affine=False, track_running_stats=False):
        self.num_groups = num_groups
        self.track_running_stats = track_running_stats
        super(_GroupNorm, self).__init__(int(num_features / num_groups),
            eps, momentum, affine)

    def _check_input_dim(self, input):
        return NotImplemented

    def forward(self, input):
        self._check_input_dim(input)
        return group_norm(input, self.num_groups, self.running_mean, self.
            running_var, self.weight, self.bias, self.training or not self.
            track_running_stats, self.momentum, self.eps)


def mish(x, inplace: bool=False):
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function - https://arxiv.org/abs/1908.08681
    """
    return x.mul(F.softplus(x).tanh())


class Mish(nn.Module):
    """
        Mish - "Mish: A Self Regularized Non-Monotonic Neural Activation Function"
        https://arxiv.org/abs/1908.08681v1
        implemented for PyTorch / FastAI by lessw2020
        github: https://github.com/lessw2020/mish
    """

    def __init__(self, inplace: bool=False):
        super(Mish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return mish(x, self.inplace)


class Swish(nn.Module):
    """
    Swish activation function, a special case of ARiA,
    for ARiA = f(x, 1, 0, 1, 1, b, 1)
    """

    def __init__(self, b=1.0):
        super(Swish, self).__init__()
        self.b = b

    def forward(self, x):
        sigmoid = F.sigmoid(x) ** self.b
        return x * sigmoid


class Aria(nn.Module):
    """
    Aria activation function described in `this paper <https://arxiv.org/abs/1805.08878/>`_.
    """

    def __init__(self, A=0, K=1.0, B=1.0, v=1.0, C=1.0, Q=1.0):
        super(Aria, self).__init__()
        self.A = A
        self.k = K
        self.B = B
        self.v = v
        self.C = C
        self.Q = Q

    def forward(self, x):
        aria = self.A + (self.k - self.A) / (self.C + self.Q * F.exp(-x) **
            self.B) ** (1 / self.v)
        return x * aria


class Aria2(nn.Module):
    """
    ARiA2 activation function, a special case of ARiA, for ARiA = f(x, 1, 0, 1, 1, b, 1/a)
    """

    def __init__(self, a=1.5, b=2.0):
        super(Aria2, self).__init__()
        self.a = a
        self.b = b

    def forward(self, x):
        aria2 = 1 + (F.exp(-x) ** self.b) ** -self.a
        return x * aria2


def hard_swish(x, inplace: bool=False):
    inner = F.relu6(x + 3.0).div_(6.0)
    return x.mul_(inner) if inplace else x.mul(inner)


class HardSwish(nn.Module):

    def __init__(self, inplace: bool=False):
        super(HardSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return hard_swish(x, self.inplace)


class StableBCELoss(nn.Module):

    def __init__(self, **kwargs):
        super(StableBCELoss, self).__init__()

    def forward(self, input, target):
        neg_abs = -input.abs()
        loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
        return loss.mean()


class WeightedSoftDiceLoss(torch.nn.Module):

    def __init__(self):
        super(WeightedSoftDiceLoss, self).__init__()

    def forward(self, logits, labels, weights):
        probs = torch.sigmoid(logits)
        num = labels.size(0)
        w = weights.view(num, -1)
        w2 = w * w
        m1 = probs.view(num, -1)
        m2 = labels.view(num, -1)
        intersection = m1 * m2
        score = 2.0 * ((w2 * intersection).sum(1) + 1) / ((w2 * m1).sum(1) +
            (w2 * m2).sum(1) + 1)
        score = 1 - score.sum() / num
        return score


class BCELoss2d(nn.Module):

    def __init__(self, weight=None, size_average=True, **kwargs):
        super(BCELoss2d, self).__init__()
        self.bce_loss = nn.BCELoss(weight, size_average)

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        return self.bce_loss(probs_flat, targets_flat)


class SoftDiceLoss(nn.Module):

    def __init__(self, smooth=1.0, **kwargs):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        num = targets.size(0)
        probs = torch.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = m1 * m2
        score = 2.0 * (intersection.sum(1) + self.smooth) / (m1.sum(1) + m2
            .sum(1) + self.smooth)
        score = 1 - score.sum() / num
        return score


class FocalLoss(nn.Module):
    """
    Weighs the contribution of each sample to the loss based in the classification error.
    If a sample is already classified correctly by the CNN, its contribution to the loss decreases.

    :eps: Focusing parameter. eps=0 is equivalent to BCE_loss
    """

    def __init__(self, l=0.5, eps=1e-06):
        super(FocalLoss, self).__init__()
        self.l = l
        self.eps = eps

    def forward(self, logits, targets):
        targets = targets.view(-1)
        probs = torch.sigmoid(logits).view(-1)
        losses = -(targets * torch.pow(1.0 - probs, self.l) * torch.log(
            probs + self.eps) + (1.0 - targets) * torch.pow(probs, self.l) *
            torch.log(1.0 - probs + self.eps))
        loss = torch.mean(losses)
        return loss


class ThresholdedL1Loss(nn.Module):

    def __init__(self, threshold=0.5, **kwargs):
        super(ThresholdedL1Loss, self).__init__()
        self.threshold = threshold

    def forward(self, logits, targets):
        targets = targets.view(-1)
        probs = torch.sigmoid(logits).view(-1)
        probs = (probs > 0.5).float()
        losses = torch.abs(targets - probs)
        loss = torch.mean(losses)
        return loss


class BCEDiceTL1Loss(nn.Module):

    def __init__(self, threshold=0.5):
        super(BCEDiceTL1Loss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(weight=None, size_average=None,
            reduce=None, reduction='mean', pos_weight=None)
        self.dice = SoftDiceLoss()
        self.tl1 = ThresholdedL1Loss(threshold=threshold)

    def forward(self, logits, targets):
        return self.bce(logits, targets) + self.dice(logits, targets
            ) + self.tl1(logits, targets)


class BCEDiceFocalLoss(nn.Module):
    """
        :param num_classes: number of classes
        :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                            focus on hard misclassified example
        :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
        :param weights: (list(), default = [1,1,1]) Optional weighing (0.0-1.0) of the losses in order of [bce, dice, focal]
    """

    def __init__(self, focal_param, weights=[1.0, 1.0, 1.0], **kwargs):
        super(BCEDiceFocalLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(weight=None, size_average=None,
            reduce=None, reduction='mean', pos_weight=None)
        self.dice = SoftDiceLoss()
        self.focal = FocalLoss(l=focal_param)
        self.weights = weights

    def forward(self, logits, targets):
        logits = logits.squeeze()
        return self.weights[0] * self.bce(logits, targets) + self.weights[1
            ] * self.dice(logits, targets) + self.weights[2] * self.focal(
            logits.unsqueeze(1), targets.unsqueeze(1))


class BCEDiceLoss(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super(BCEDiceLoss, self).__init__()
        self.bce = BCELoss2d()
        self.dice = SoftDiceLoss()

    def forward(self, logits, targets):
        return self.bce(logits, targets) + self.dice(logits, targets)


class WeightedBCELoss2d(nn.Module):

    def __init__(self, **kwargs):
        super(WeightedBCELoss2d, self).__init__()

    def forward(self, logits, labels, weights):
        w = weights.view(-1)
        z = logits.view(-1)
        t = labels.view(-1)
        loss = w * z.clamp(min=0) - w * z * t + w * torch.log(1 + torch.exp
            (-z.abs()))
        loss = loss.sum() / w.sum()
        return loss


class WeightedSoftDiceLoss(nn.Module):

    def __init__(self, **kwargs):
        super(WeightedSoftDiceLoss, self).__init__()

    def forward(self, logits, labels, weights):
        probs = torch.sigmoid(logits)
        num = labels.size(0)
        w = weights.view(num, -1)
        w2 = w * w
        m1 = probs.view(num, -1)
        m2 = labels.view(num, -1)
        intersection = m1 * m2
        smooth = 1.0
        score = 2.0 * ((w2 * intersection).sum(1) + smooth) / ((w2 * m1).
            sum(1) + (w2 * m2).sum(1) + smooth)
        score = 1 - score.sum() / num
        return score


class BCEDicePenalizeBorderLoss(nn.Module):

    def __init__(self, kernel_size=21, **kwargs):
        super(BCEDicePenalizeBorderLoss, self).__init__()
        self.bce = WeightedBCELoss2d()
        self.dice = WeightedSoftDiceLoss()
        self.kernel_size = kernel_size

    def to(self, device):
        super().to(device=device)
        self.bce.to(device=device)
        self.dice.to(device=device)

    def forward(self, logits, labels):
        a = F.avg_pool2d(labels, kernel_size=self.kernel_size, padding=self
            .kernel_size // 2, stride=1)
        ind = a.ge(0.01) * a.le(0.99)
        ind = ind.float()
        weights = torch.ones(a.size()).to(device=logits.device)
        w0 = weights.sum()
        weights = weights + ind * 2
        w1 = weights.sum()
        weights = weights / w1 * w0
        loss = self.bce(logits, labels, weights) + self.dice(logits, labels,
            weights)
        return loss


class FocalLoss2(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)

    Params:
        :param num_class:
        :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
        :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                        focus on hard misclassified example
        :param smooth: (float,double) smooth value when cross entropy
        :param balance_index: (int) balance class index, should be specific when alpha is float
        :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, num_class, alpha=None, gamma=2, balance_index=-1,
        smooth=None, size_average=True):
        super(FocalLoss2, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.size_average = size_average
        if self.alpha is None:
            self.alpha = torch.ones(self.num_class, 1)
        elif isinstance(self.alpha, (list, np.ndarray)):
            assert len(self.alpha) == self.num_class
            self.alpha = torch.FloatTensor(alpha).view(self.num_class, 1)
            self.alpha = self.alpha / self.alpha.sum()
        elif isinstance(self.alpha, float):
            alpha = torch.ones(self.num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[balance_index] = self.alpha
            self.alpha = alpha
        else:
            raise TypeError('Not support alpha type')
        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, logit, target):
        if logit.dim() > 2:
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = target.view(-1, 1)
        epsilon = 1e-10
        alpha = self.alpha.to(logit.device)
        idx = target.cpu().long()
        one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        one_hot_key = one_hot_key.to(logit.device)
        if self.smooth:
            one_hot_key = torch.clamp(one_hot_key, self.smooth / (self.
                num_class - 1), 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + epsilon
        logpt = pt.log()
        gamma = self.gamma
        alpha = alpha[idx]
        loss = -1 * alpha * torch.pow(1 - pt, gamma) * logpt
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class FocalLoss3(nn.Module):
    """
        This criterion is a implemenation of Focal Loss, which is proposed in Focal Loss for Dense Object Detection.
            Loss(x, class) = - lpha (1-softmax(x)[class])^gamma \\log(softmax(x)[class])
        The losses are averaged across observations for each minibatch.

        Params:
            :param alpha: (1D Tensor, Variable) - the scalar factor for this criterion
            :param gamma: (float, double) - gamma > 0
            :param size_average: (bool) - size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are instead summed for each minibatch.
    """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss3, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num + 1))
        elif isinstance(alpha, Variable):
            self.alpha = alpha
        else:
            self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        P = F.softmax(inputs)
        if len(inputs.size()) == 3:
            torch_out = torch.zeros(inputs.size())
        else:
            b, c, h, w = inputs.size()
            torch_out = torch.zeros([b, c + 1, h, w])
        if inputs.is_cuda:
            torch_out = torch_out
        class_mask = Variable(torch_out)
        class_mask.scatter_(1, targets.long(), 1.0)
        class_mask = class_mask[:, :-1, :, :]
        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha
        alpha = self.alpha[targets.data.view(-1)].view_as(targets)
        probs = (P * class_mask).sum(1)
        log_p = probs.log()
        batch_loss = -alpha * torch.pow(1 - probs, self.gamma) * log_p
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class BinaryFocalLoss(nn.Module):
    """
        Implementation of binary focal loss. For multi-class focal loss use one of the other implementations.

        gamma = 0 is equivalent to BinaryCrossEntropy Loss
    """

    def __init__(self, gamma=1.333, eps=1e-06, alpha=1.0, **kwargs):
        super().__init__()
        self.gamma = gamma
        self.eps = eps
        self.alpha = alpha

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets,
            reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()


class PoissonLoss(nn.Module):

    def __init__(self, bias=1e-12, **kwargs):
        super().__init__()
        self.bias = bias

    def forward(self, output, target):
        with torch.no_grad:
            return (output - target * torch.log(output + self.bias)).mean()


class PoissonLoss3d(nn.Module):

    def __init__(self, bias=1e-12, **kwargs):
        super().__init__()
        self.bias = bias

    def forward(self, output, target):
        with torch.no_grad:
            lag = target.size(1) - output.size(1)
            return (output - target[:, lag:, :] * torch.log(output + self.bias)
                ).mean()


class L1Loss3d(nn.Module):

    def __init__(self, bias=1e-12, **kwargs):
        super().__init__()
        self.bias = bias

    def forward(self, output, target):
        with torch.no_grad:
            lag = target.size(1) - output.size(1)
            return (output - target[:, lag:, :]).abs().mean()


class MSE3D(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        with torch.no_grad:
            lag = target.size(1) - output.size(1)
            return (output - target[:, lag:, :]).pow(2).mean()


class BCEWithLogitsViewLoss(nn.BCEWithLogitsLoss):
    """
    Silly wrapper of nn.BCEWithLogitsLoss because BCEWithLogitsLoss only takes a 1-D array
    """

    def __init__(self, weight=None, size_average=True, **kwargs):
        super().__init__(weight=weight, size_average=size_average)

    def forward(self, input, target):
        """
        :param input:
        :param target:
        :return:

        Simply passes along input.view(-1), target.view(-1)
        """
        return super().forward(input.view(-1), target.view(-1))


class mIoULoss(nn.Module):

    def __init__(self, weight=None, size_average=True, num_classes=2, **kwargs
        ):
        super(mIoULoss, self).__init__()
        self.classes = num_classes

    def forward(self, inputs, target_oneHot):
        N = inputs.size()[0]
        inputs = F.softmax(inputs, dim=1)
        inter = inputs * target_oneHot
        inter = inter.view(N, self.classes, -1).sum(2)
        union = inputs + target_oneHot - inputs * target_oneHot
        union = union.view(N, self.classes, -1).sum(2)
        loss = inter / union
        return -loss.mean()


class ComboBCEDiceLoss(nn.Module):
    """
        Combination BinaryCrossEntropy (BCE) and Dice Loss with an optional running mean and loss weighing.
    """

    def __init__(self, use_running_mean=False, bce_weight=1, dice_weight=1,
        eps=1e-06, gamma=0.9, combined_loss_only=True, **kwargs):
        """

        :param use_running_mean: - bool (default: False) Whether to accumulate a running mean and add it to the loss with (1-gamma)
        :param bce_weight: - float (default: 1.0) Weight multiplier for the BCE loss (relative to dice)
        :param dice_weight: - float (default: 1.0) Weight multiplier for the Dice loss (relative to BCE)
        :param eps: -
        :param gamma:
        :param combined_loss_only: - bool (default: True) whether to return a single combined loss or three separate losses
        """
        super().__init__()
        """
        Note: BCEWithLogitsLoss already performs a torch.sigmoid(pred)
        before applying BCE!
        """
        self.bce_logits_loss = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.eps = eps
        self.gamma = gamma
        self.combined_loss_only = combined_loss_only
        self.use_running_mean = use_running_mean
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        if self.use_running_mean == True:
            self.register_buffer('running_bce_loss', torch.zeros(1))
            self.register_buffer('running_dice_loss', torch.zeros(1))
            self.reset_parameters()

    def to(self, device):
        super().to(device=device)
        self.bce_logits_loss.to(device=device)

    def reset_parameters(self):
        self.running_bce_loss.zero_()
        self.running_dice_loss.zero_()

    def forward(self, outputs, targets):
        outputs = outputs.squeeze()
        assert len(outputs.shape) == len(targets.shape)
        assert outputs.size(-0) == targets.size(-0)
        assert outputs.size(-1) == targets.size(-1)
        assert outputs.size(-2) == targets.size(-2)
        bce_loss = self.bce_logits_loss(outputs, targets)
        dice_target = (targets == 1).float()
        dice_output = F.sigmoid(outputs)
        intersection = (dice_output * dice_target).sum()
        union = dice_output.sum() + dice_target.sum() + self.eps
        dice_loss = -torch.log(2 * intersection / union)
        if self.use_running_mean == False:
            bmw = self.bce_weight
            dmw = self.dice_weight
        else:
            self.running_bce_loss = (self.running_bce_loss * self.gamma + 
                bce_loss.data * (1 - self.gamma))
            self.running_dice_loss = (self.running_dice_loss * self.gamma +
                dice_loss.data * (1 - self.gamma))
            bm = float(self.running_bce_loss)
            dm = float(self.running_dice_loss)
            bmw = 1 - bm / (bm + dm)
            dmw = 1 - dm / (bm + dm)
        loss = bce_loss * bmw + dice_loss * dmw
        if self.combined_loss_only:
            return loss
        else:
            return loss, bce_loss, dice_loss


class ComboSemsegLossWeighted(nn.Module):

    def __init__(self, use_running_mean=False, bce_weight=1, dice_weight=1,
        eps=1e-06, gamma=0.9, use_weight_mask=False, combined_loss_only=
        False, **kwargs):
        super().__init__()
        self.use_weight_mask = use_weight_mask
        self.nll_loss = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.eps = eps
        self.gamma = gamma
        self.combined_loss_only = combined_loss_only
        self.use_running_mean = use_running_mean
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        if self.use_running_mean == True:
            self.register_buffer('running_bce_loss', torch.zeros(1))
            self.register_buffer('running_dice_loss', torch.zeros(1))
            self.reset_parameters()

    def to(self, device):
        super().to(device=device)
        self.nll_loss.to(device=device)

    def reset_parameters(self):
        self.running_bce_loss.zero_()
        self.running_dice_loss.zero_()

    def forward(self, outputs, targets, weights):
        assert len(outputs.shape) == len(targets.shape)
        assert outputs.size(0) == targets.size(0)
        assert outputs.size(2) == targets.size(2)
        assert outputs.size(3) == targets.size(3)
        assert outputs.size(0) == weights.size(0)
        assert outputs.size(2) == weights.size(1)
        assert outputs.size(3) == weights.size(2)
        if self.use_weight_mask:
            bce_loss = F.binary_cross_entropy_with_logits(input=outputs,
                target=targets, weight=weights)
        else:
            bce_loss = self.nll_loss(input=outputs, target=targets)
        dice_target = (targets == 1).float()
        dice_output = F.sigmoid(outputs)
        intersection = (dice_output * dice_target).sum()
        union = dice_output.sum() + dice_target.sum() + self.eps
        dice_loss = -torch.log(2 * intersection / union)
        if self.use_running_mean == False:
            bmw = self.bce_weight
            dmw = self.dice_weight
        else:
            self.running_bce_loss = (self.running_bce_loss * self.gamma + 
                bce_loss.data * (1 - self.gamma))
            self.running_dice_loss = (self.running_dice_loss * self.gamma +
                dice_loss.data * (1 - self.gamma))
            bm = float(self.running_bce_loss)
            dm = float(self.running_dice_loss)
            bmw = 1 - bm / (bm + dm)
            dmw = 1 - dm / (bm + dm)
        loss = bce_loss * bmw + dice_loss * dmw
        if self.combined_loss_only:
            return loss
        else:
            return loss, bce_loss, dice_loss


class OhemCrossEntropy2d(nn.Module):
    """
    Online Hard Example Loss with Cross Entropy (used for classification)

    OHEM description: http://www.erogol.com/online-hard-example-mining-pytorch/
    """

    def __init__(self, ignore_label=-1, thresh=0.7, min_kept=100000,
        use_weight=True, **kwargs):
        super(OhemCrossEntropy2d, self).__init__()
        self.ignore_label = ignore_label
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        if use_weight:
            None
            weight = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 
                1.0166, 0.9969, 0.9754, 1.0489, 0.8786, 1.0023, 0.9539, 
                0.9843, 1.1116, 0.9037, 1.0865, 1.0955, 1.0865, 1.1529, 1.0507]
                )
            self.criterion = torch.nn.CrossEntropyLoss(weight=weight,
                ignore_index=ignore_label)
        else:
            None
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=
                ignore_label)

    def to(self, device):
        super().to(device=device)
        self.criterion.to(device=device)

    def forward(self, predict, target, weight=None):
        """
        Args:
            predict:(n, c, h, w)
            target:(n, h, w)
            weight (Tensor, optional): a manual rescaling weight given to each class.
                                       If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), '{0} vs {1} '.format(predict
            .size(0), target.size(0))
        assert predict.size(2) == target.size(1), '{0} vs {1} '.format(predict
            .size(2), target.size(1))
        assert predict.size(3) == target.size(2), '{0} vs {1} '.format(predict
            .size(3), target.size(3))
        n, c, h, w = predict.size()
        input_label = target.data.cpu().numpy().ravel().astype(np.int32)
        x = np.rollaxis(predict.data.cpu().numpy(), 1).reshape((c, -1))
        input_prob = np.exp(x - x.max(axis=0).reshape((1, -1)))
        input_prob /= input_prob.sum(axis=0).reshape((1, -1))
        valid_flag = input_label != self.ignore_label
        valid_inds = np.where(valid_flag)[0]
        label = input_label[valid_flag]
        num_valid = valid_flag.sum()
        if self.min_kept >= num_valid:
            None
        elif num_valid > 0:
            prob = input_prob[:, (valid_flag)]
            pred = prob[label, np.arange(len(label), dtype=np.int32)]
            threshold = self.thresh
            if self.min_kept > 0:
                index = pred.argsort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if pred[threshold_index] > self.thresh:
                    threshold = pred[threshold_index]
            kept_flag = pred <= threshold
            valid_inds = valid_inds[kept_flag]
        label = input_label[valid_inds].copy()
        input_label.fill(self.ignore_label)
        input_label[valid_inds] = label
        valid_flag_new = input_label != self.ignore_label
        target = Variable(torch.from_numpy(input_label.reshape(target.size(
            ))).long())
        return self.criterion(predict, target)


class EncNetLoss(nn.CrossEntropyLoss):
    """
    2D Cross Entropy Loss with SE Loss

    Specifically used for EncNet.
    se_loss is the Semantic Encoding Loss from the paper `Context Encoding for Semantic Segmentation <https://arxiv.org/pdf/1803.08904v1>`_.
    It computes probabilities of contexts appearing together.

    Without SE_loss and Aux_loss this class simply forwards inputs to Torch's Cross Entropy Loss (nn.CrossEntropyLoss)
    """

    def __init__(self, se_loss=True, se_weight=0.2, nclass=19, aux=False,
        aux_weight=0.4, weight=None, ignore_index=-1, **kwargs):
        super(EncNetLoss, self).__init__(weight, None, ignore_index)
        self.se_loss = se_loss
        self.aux = aux
        self.nclass = nclass
        self.se_weight = se_weight
        self.aux_weight = aux_weight
        self.bceloss = nn.BCELoss(weight)

    def forward(self, *inputs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])
        if not self.se_loss and not self.aux:
            return super(EncNetLoss, self).forward(*inputs)
        elif not self.se_loss:
            pred1, pred2, target = tuple(inputs)
            loss1 = super(EncNetLoss, self).forward(pred1, target)
            loss2 = super(EncNetLoss, self).forward(pred2, target)
            return dict(loss=loss1 + self.aux_weight * loss2)
        elif not self.aux:
            pred, se_pred, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass
                ).type_as(pred)
            loss1 = super(EncNetLoss, self).forward(pred, target)
            loss2 = self.bceloss(torch.sigmoid(se_pred), se_target)
            return dict(loss=loss1 + self.se_weight * loss2)
        else:
            pred1, se_pred, pred2, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass
                ).type_as(pred1)
            loss1 = super(EncNetLoss, self).forward(pred1, target)
            loss2 = super(EncNetLoss, self).forward(pred2, target)
            loss3 = self.bceloss(torch.sigmoid(se_pred), se_target)
            return dict(loss=loss1 + self.aux_weight * loss2 + self.
                se_weight * loss3)

    @staticmethod
    def _get_batch_label_vector(target, nclass):
        batch = target.size(0)
        tvect = Variable(torch.zeros(batch, nclass))
        for i in range(batch):
            hist = torch.histc(target[i].cpu().data.float(), bins=nclass,
                min=0, max=nclass - 1)
            vect = hist > 0
            tvect[i] = vect
        return tvect


class FocalBinaryTverskyFunc(Function):
    """
        Focal Tversky Loss as defined in `this paper <https://arxiv.org/abs/1810.07842>`_

        `Authors' implementation <https://github.com/nabsabraham/focal-tversky-unet>`_ in Keras.

        Params:
            :param alpha: controls the penalty for false positives.
            :param beta: penalty for false negative.
            :param gamma : focal coefficient range[1,3]
            :param reduction: return mode

        Notes:
            alpha = beta = 0.5 => dice coeff
            alpha = beta = 1 => tanimoto coeff
            alpha + beta = 1 => F beta coeff
            add focal index -> loss=(1-T_index)**(1/gamma)
    """

    def __init__(ctx, alpha=0.5, beta=0.7, gamma=1.0, reduction='mean'):
        """
        :param alpha: controls the penalty for false positives.
        :param beta: penalty for false negative.
        :param gamma : focal coefficient range[1,3]
        :param reduction: return mode
        Notes:
        alpha = beta = 0.5 => dice coeff
        alpha = beta = 1 => tanimoto coeff
        alpha + beta = 1 => F beta coeff
        add focal index -> loss=(1-T_index)**(1/gamma)
        """
        ctx.alpha = alpha
        ctx.beta = beta
        ctx.epsilon = 1e-06
        ctx.reduction = reduction
        ctx.gamma = gamma
        sum = ctx.beta + ctx.alpha
        if sum != 1:
            ctx.beta = ctx.beta / sum
            ctx.alpha = ctx.alpha / sum

    def forward(ctx, input, target):
        batch_size = input.size(0)
        _, input_label = input.max(1)
        input_label = input_label.float()
        target_label = target.float()
        ctx.save_for_backward(input, target_label)
        input_label = input_label.view(batch_size, -1)
        target_label = target_label.view(batch_size, -1)
        ctx.P_G = torch.sum(input_label * target_label, 1)
        ctx.P_NG = torch.sum(input_label * (1 - target_label), 1)
        ctx.NP_G = torch.sum((1 - input_label) * target_label, 1)
        index = ctx.P_G / (ctx.P_G + ctx.alpha * ctx.P_NG + ctx.beta * ctx.
            NP_G + ctx.epsilon)
        loss = torch.pow(1 - index, 1 / ctx.gamma)
        if ctx.reduction == 'none':
            loss = loss
        elif ctx.reduction == 'sum':
            loss = torch.sum(loss)
        else:
            loss = torch.mean(loss)
        return loss

    def backward(ctx, grad_out):
        """
        :param ctx:
        :param grad_out:
        :return:
        d_loss/dT_loss=(1/gamma)*(T_loss)**(1/gamma-1)
        (dT_loss/d_P1)  = 2*P_G*[G*(P_G+alpha*P_NG+beta*NP_G)-(G+alpha*NG)]/[(P_G+alpha*P_NG+beta*NP_G)**2]
                        = 2*P_G
        (dT_loss/d_p0)=
        """
        inputs, target = ctx.saved_tensors
        inputs = inputs.float()
        target = target.float()
        batch_size = inputs.size(0)
        sum = (ctx.P_G + ctx.alpha * ctx.P_NG + ctx.beta * ctx.NP_G + ctx.
            epsilon)
        P_G = ctx.P_G.view(batch_size, 1, 1, 1, 1)
        if inputs.dim() == 5:
            sum = sum.view(batch_size, 1, 1, 1, 1)
        elif inputs.dim() == 4:
            sum = sum.view(batch_size, 1, 1, 1)
            P_G = ctx.P_G.view(batch_size, 1, 1, 1)
        sub = (ctx.alpha * (1 - target) + target) * P_G
        dL_dT = 1 / ctx.gamma * torch.pow(P_G / sum, 1 / ctx.gamma - 1)
        dT_dp0 = -2 * (target / sum - sub / sum / sum)
        dL_dp0 = dL_dT * dT_dp0
        dT_dp1 = ctx.beta * (1 - target) * P_G / sum / sum
        dL_dp1 = dL_dT * dT_dp1
        grad_input = torch.cat((dL_dp1, dL_dp0), dim=1)
        return grad_input, None


class MultiTverskyLoss(nn.Module):
    """
    Tversky Loss for segmentation adaptive with multi class segmentation

    Args
        :param alpha (Tensor, float, optional): controls the penalty for false positives.
        :param beta (Tensor, float, optional): controls the penalty for false negative.
        :param gamma (Tensor, float, optional): focal coefficient
        :param weights (Tensor, optional): a manual rescaling weight given to each class. If given, it has to be a Tensor of size `C`
    """

    def __init__(self, alpha=0.5, beta=0.5, gamma=1.0, reduction='mean',
        weights=None):
        """
        :param alpha (Tensor, float, optional): controls the penalty for false positives.
        :param beta (Tensor, float, optional): controls the penalty for false negative.
        :param gamma (Tensor, float, optional): focal coefficient
        :param weights (Tensor, optional): a manual rescaling weight given to each
            class. If given, it has to be a Tensor of size `C`
        """
        super(MultiTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.reduction = reduction
        self.weights = weights

    def forward(self, inputs, targets):
        num_class = inputs.size(1)
        weight_losses = 0.0
        if self.weights is not None:
            assert len(self.weights
                ) == num_class, 'number of classes should be equal to length of weights '
            weights = self.weights
        else:
            weights = [1.0 / num_class] * num_class
        input_slices = torch.split(inputs, [1] * num_class, dim=1)
        for idx in range(num_class):
            input_idx = input_slices[idx]
            input_idx = torch.cat((1 - input_idx, input_idx), dim=1)
            target_idx = (targets == idx) * 1
            loss_func = FocalBinaryTverskyFunc(self.alpha, self.beta, self.
                gamma, self.reduction)
            loss_idx = loss_func(input_idx, target_idx)
            weight_losses += loss_idx * weights[idx]
        return weight_losses


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


class LovaszSoftmax(nn.Module):

    def __init__(self, reduction='mean', **kwargs):
        super(LovaszSoftmax, self).__init__()
        self.reduction = reduction

    def prob_flatten(self, input, target):
        assert input.dim() in [4, 5]
        num_class = input.size(1)
        if input.dim() == 4:
            input = input.permute(0, 2, 3, 1).contiguous()
            input_flatten = input.view(-1, num_class)
        elif input.dim() == 5:
            input = input.permute(0, 2, 3, 4, 1).contiguous()
            input_flatten = input.view(-1, num_class)
        target_flatten = target.view(-1)
        return input_flatten, target_flatten

    def lovasz_softmax_flat(self, inputs, targets):
        num_classes = inputs.size(1)
        losses = []
        for c in range(num_classes):
            target_c = (targets == c).float()
            if num_classes == 1:
                input_c = inputs[:, (0)]
            else:
                input_c = inputs[:, (c)]
            loss_c = (torch.autograd.Variable(target_c) - input_c).abs()
            loss_c_sorted, loss_index = torch.sort(loss_c, 0, descending=True)
            target_c_sorted = target_c[loss_index]
            losses.append(torch.dot(loss_c_sorted, torch.autograd.Variable(
                lovasz_grad(target_c_sorted))))
        losses = torch.stack(losses)
        if self.reduction == 'none':
            loss = losses
        elif self.reduction == 'sum':
            loss = losses.sum()
        else:
            loss = losses.mean()
        return loss

    def forward(self, inputs, targets):
        inputs, targets = self.prob_flatten(inputs, targets)
        losses = self.lovasz_softmax_flat(inputs, targets)
        return losses


class ActiveContourLoss(nn.Module):
    """
        `Learning Active Contour Models for Medical Image Segmentation <http://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Learning_Active_Contour_Models_for_Medical_Image_Segmentation_CVPR_2019_paper.pdf>`_
        Note that is only works for B/W masks right now... which is kind of the point of this loss as contours in RGB should be cast to B/W
        before computing the loss.

        Params:
            :param len_w: (float, default=1.0) - The multiplier to use when adding boundary loss.
            :param reg_w: (float, default=1.0) - The multiplier to use when adding region loss.
            :param apply_log: (bool, default=True) - Whether to transform the log into log space (due to the
    """

    def __init__(self, len_w=1.0, reg_w=1.0, apply_log=True, **kwargs):
        super(ActiveContourLoss, self).__init__()
        self.len_w = len_w
        self.reg_w = reg_w
        self.epsilon = 1e-08
        self.apply_log = apply_log

    def forward(self, logits, target):
        image_size = logits.size(3)
        target = target.unsqueeze(1)
        probs = F.softmax(logits, dim=0)
        """
        length term:
            - Subtract adjacent pixels from each other in X and Y directions
            - Determine where they differ from the ground truth (targets)
            - Calculate MSE
        """
        x = probs[:, :, 1:, :] - probs[:, :, :-1, :]
        y = probs[:, :, :, 1:] - probs[:, :, :, :-1]
        target_x = target[:, :, 1:, :] - target[:, :, :-1, :]
        target_y = target[:, :, :, 1:] - target[:, :, :, :-1]
        delta_x = (target_x - x).abs()
        delta_y = (target_y - y).abs()
        length_loss = torch.sqrt(delta_x.sum() ** 2 + delta_y.sum() ** 2 +
            self.epsilon)
        """
        region term (should this be done in log space to avoid instabilities?)
            - compute the error produced by all pixels that are not equal to 0 outside of the ground truth mask
            - compute error produced by all pixels that are not equal to 1 inside the mask
        """
        error_in = probs[:, (0), :, :] * (target[:, (0), :, :] - 1) ** 2
        probs_diff = (probs[:, (0), :, :] - target[:, (0), :, :]).abs()
        error_out = probs_diff * target[:, (0), :, :]
        if self.apply_log:
            loss = torch.log(length_loss) + torch.log(error_in.sum() +
                error_out.sum())
        else:
            loss = self.reg_w * (error_in.sum() + error_out.sum())
        return torch.clamp(loss, min=0.0)


def softmax_helper(x):
    rpt = [(1) for _ in range(len(x.size()))]
    rpt[1] = x.size(1)
    x_max = x.max(1, keepdim=True)[0].repeat(*rpt)
    e_x = torch.exp(x - x_max)
    return e_x / e_x.sum(1, keepdim=True).repeat(*rpt)


class BDLoss(nn.Module):

    def __init__(self, **kwargs):
        """
        compute boudary loss
        only compute the loss of foreground
        ref: https://github.com/LIVIAETS/surface-loss/blob/108bd9892adca476e6cdf424124bc6268707498e/losses.py#L74
        """
        super(BDLoss, self).__init__()

    def forward(self, logits, target, bound):
        """
        Takes 2D or 3D logits.

        logits: (batch_size, class, x,y,(z))
        target: ground truth, shape: (batch_size, 1, x,y,(z))
        bound: precomputed distance map, shape (batch_size, class, x,y,(z))

        Torch Eigensum description: https://stackoverflow.com/questions/55894693/understanding-pytorch-einsum
        """
        compute_directive = 'bcxy,bcxy->bcxy'
        if len(logits) == 5:
            compute_directive = 'bcxyz,bcxyz->bcxyz'
        net_output = softmax_helper(logits)
        pc = net_output[:, 1:, (...)].type(torch.float32)
        dc = bound[:, 1:, (...)].type(torch.float32)
        multipled = torch.einsum(compute_directive, pc, dc)
        bd_loss = multipled.mean()
        return bd_loss


class TverskyLoss(nn.Module):
    """Computes the Tversky loss [1].
        Args:
            :param alpha: controls the penalty for false positives.
            :param beta: controls the penalty for false negatives.
            :param eps: added to the denominator for numerical stability.
        Returns:
            tversky_loss: the Tversky loss.
        Notes:
            alpha = beta = 0.5 => dice coeff
            alpha = beta = 1 => tanimoto coeff
            alpha + beta = 1 => F beta coeff
        References:
            [1]: https://arxiv.org/abs/1706.05721
    """

    def __init__(self, alpha, beta, eps=1e-07, **kwargs):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, logits, targets):
        """
        Args:
            :param logits: a tensor of shape [B, C, H, W]. Corresponds to the raw output or logits of the model.
            :param targets: a tensor of shape [B, H, W] or [B, 1, H, W].
            :return: loss
        """
        num_classes = logits.shape[1]
        if num_classes == 1:
            true_1_hot = torch.eye(num_classes + 1)[targets.squeeze(1).long()]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            true_1_hot_f = true_1_hot[:, 0:1, :, :]
            true_1_hot_s = true_1_hot[:, 1:2, :, :]
            true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
            pos_prob = torch.sigmoid(logits)
            neg_prob = 1 - pos_prob
            probas = torch.cat([pos_prob, neg_prob], dim=1)
        else:
            true_1_hot = torch.eye(num_classes)[targets.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            probas = F.softmax(logits, dim=1)
        true_1_hot = true_1_hot.type(logits.type())
        dims = (0,) + tuple(range(2, logits.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        fps = torch.sum(probas * (1 - true_1_hot), dims)
        fns = torch.sum((1 - probas) * true_1_hot, dims)
        num = intersection
        denom = intersection + self.alpha * fps + self.beta * fns
        tversky_loss = (num / (denom + self.eps)).mean()
        return 1 - tversky_loss


class AngularPenaltySMLoss(nn.Module):

    def __init__(self, in_features, out_features, loss_type='arcface', eps=
        1e-07, s=None, m=None):
        """
        Angular Penalty Softmax Loss

        Three 'loss_types' available: ['arcface', 'sphereface', 'cosface']
        These losses are described in the following papers:

        ArcFace: https://arxiv.org/abs/1801.07698
        SphereFace: https://arxiv.org/abs/1704.08063
        CosFace/Ad Margin: https://arxiv.org/abs/1801.05599

        - Example -
        criterion = AngularPenaltySMLoss(in_features, out_features, loss_type='arcface') # loss_type in ['arcface', 'sphereface', 'cosface']
        """
        super(AngularPenaltySMLoss, self).__init__()
        loss_type = loss_type.lower()
        assert loss_type in ['arcface', 'sphereface', 'cosface']
        if loss_type == 'arcface':
            self.s = 64.0 if not s else s
            self.m = 0.5 if not m else m
        if loss_type == 'sphereface':
            self.s = 64.0 if not s else s
            self.m = 1.35 if not m else m
        if loss_type == 'cosface':
            self.s = 30.0 if not s else s
            self.m = 0.4 if not m else m
        self.loss_type = loss_type
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.eps = eps

    def forward(self, x, labels):
        """
        input shape (N, in_features)
        """
        assert len(x) == len(labels)
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.out_features
        for W in self.fc.parameters():
            W = F.normalize(W, p=2, dim=1)
        x = F.normalize(x, p=2, dim=1)
        wf = self.fc(x)
        if self.loss_type == 'cosface':
            numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]
                ) - self.m)
        if self.loss_type == 'arcface':
            numerator = self.s * torch.cos(torch.acos(torch.clamp(torch.
                diagonal(wf.transpose(0, 1)[labels]), -1.0 + self.eps, 1 -
                self.eps)) + self.m)
        if self.loss_type == 'sphereface':
            numerator = self.s * torch.cos(self.m * torch.acos(torch.clamp(
                torch.diagonal(wf.transpose(0, 1)[labels]), -1.0 + self.eps,
                1 - self.eps)))
        excl = torch.cat([torch.cat((wf[(i), :y], wf[(i), y + 1:])).
            unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s *
            excl), dim=1)
        L = numerator - torch.log(denominator)
        return -torch.mean(L)


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = labels != ignore
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\\infty and +\\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        return logits.sum() * 0.0
    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * Variable(signs)
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    return loss


def isnan(x):
    return x != x


def mean(l, ignore_nan=True, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\\infty and +\\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = mean(lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(
            0), lab.unsqueeze(0), ignore)) for log, lab in zip(logits, labels))
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore)
            )
    return loss


class LovaszBinaryLoss(torch.nn.modules.Module):

    def __init__(self):
        super(LovaszBinaryLoss, self).__init__()

    def forward(self, input, target):
        return lovasz_hinge(input, target)


class StableBCELoss(torch.nn.modules.Module):

    def __init__(self):
        super(StableBCELoss, self).__init__()

    def forward(self, input, target):
        neg_abs = -input.abs()
        loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
        return loss.mean()


class BNInception(nn.Module):

    def __init__(self, num_classes=1000):
        super(BNInception, self).__init__()
        inplace = True
        self.conv1_7x7_s2 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2,
            2), padding=(3, 3))
        self.conv1_7x7_s2_bn = nn.BatchNorm2d(64, affine=True)
        self.conv1_relu_7x7 = nn.ReLU(inplace)
        self.pool1_3x3_s2 = nn.MaxPool2d((3, 3), stride=(2, 2), dilation=(1,
            1), ceil_mode=True)
        self.conv2_3x3_reduce = nn.Conv2d(64, 64, kernel_size=(1, 1),
            stride=(1, 1))
        self.conv2_3x3_reduce_bn = nn.BatchNorm2d(64, affine=True)
        self.conv2_relu_3x3_reduce = nn.ReLU(inplace)
        self.conv2_3x3 = nn.Conv2d(64, 192, kernel_size=(3, 3), stride=(1, 
            1), padding=(1, 1))
        self.conv2_3x3_bn = nn.BatchNorm2d(192, affine=True)
        self.conv2_relu_3x3 = nn.ReLU(inplace)
        self.pool2_3x3_s2 = nn.MaxPool2d((3, 3), stride=(2, 2), dilation=(1,
            1), ceil_mode=True)
        self.inception_3a_1x1 = nn.Conv2d(192, 64, kernel_size=(1, 1),
            stride=(1, 1))
        self.inception_3a_1x1_bn = nn.BatchNorm2d(64, affine=True)
        self.inception_3a_relu_1x1 = nn.ReLU(inplace)
        self.inception_3a_3x3_reduce = nn.Conv2d(192, 64, kernel_size=(1, 1
            ), stride=(1, 1))
        self.inception_3a_3x3_reduce_bn = nn.BatchNorm2d(64, affine=True)
        self.inception_3a_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_3a_3x3 = nn.Conv2d(64, 64, kernel_size=(3, 3),
            stride=(1, 1), padding=(1, 1))
        self.inception_3a_3x3_bn = nn.BatchNorm2d(64, affine=True)
        self.inception_3a_relu_3x3 = nn.ReLU(inplace)
        self.inception_3a_double_3x3_reduce = nn.Conv2d(192, 64,
            kernel_size=(1, 1), stride=(1, 1))
        self.inception_3a_double_3x3_reduce_bn = nn.BatchNorm2d(64, affine=True
            )
        self.inception_3a_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_3a_double_3x3_1 = nn.Conv2d(64, 96, kernel_size=(3, 
            3), stride=(1, 1), padding=(1, 1))
        self.inception_3a_double_3x3_1_bn = nn.BatchNorm2d(96, affine=True)
        self.inception_3a_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_3a_double_3x3_2 = nn.Conv2d(96, 96, kernel_size=(3, 
            3), stride=(1, 1), padding=(1, 1))
        self.inception_3a_double_3x3_2_bn = nn.BatchNorm2d(96, affine=True)
        self.inception_3a_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_3a_pool = nn.AvgPool2d(3, stride=1, padding=1,
            ceil_mode=True, count_include_pad=True)
        self.inception_3a_pool_proj = nn.Conv2d(192, 32, kernel_size=(1, 1),
            stride=(1, 1))
        self.inception_3a_pool_proj_bn = nn.BatchNorm2d(32, affine=True)
        self.inception_3a_relu_pool_proj = nn.ReLU(inplace)
        self.inception_3b_1x1 = nn.Conv2d(256, 64, kernel_size=(1, 1),
            stride=(1, 1))
        self.inception_3b_1x1_bn = nn.BatchNorm2d(64, affine=True)
        self.inception_3b_relu_1x1 = nn.ReLU(inplace)
        self.inception_3b_3x3_reduce = nn.Conv2d(256, 64, kernel_size=(1, 1
            ), stride=(1, 1))
        self.inception_3b_3x3_reduce_bn = nn.BatchNorm2d(64, affine=True)
        self.inception_3b_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_3b_3x3 = nn.Conv2d(64, 96, kernel_size=(3, 3),
            stride=(1, 1), padding=(1, 1))
        self.inception_3b_3x3_bn = nn.BatchNorm2d(96, affine=True)
        self.inception_3b_relu_3x3 = nn.ReLU(inplace)
        self.inception_3b_double_3x3_reduce = nn.Conv2d(256, 64,
            kernel_size=(1, 1), stride=(1, 1))
        self.inception_3b_double_3x3_reduce_bn = nn.BatchNorm2d(64, affine=True
            )
        self.inception_3b_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_3b_double_3x3_1 = nn.Conv2d(64, 96, kernel_size=(3, 
            3), stride=(1, 1), padding=(1, 1))
        self.inception_3b_double_3x3_1_bn = nn.BatchNorm2d(96, affine=True)
        self.inception_3b_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_3b_double_3x3_2 = nn.Conv2d(96, 96, kernel_size=(3, 
            3), stride=(1, 1), padding=(1, 1))
        self.inception_3b_double_3x3_2_bn = nn.BatchNorm2d(96, affine=True)
        self.inception_3b_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_3b_pool = nn.AvgPool2d(3, stride=1, padding=1,
            ceil_mode=True, count_include_pad=True)
        self.inception_3b_pool_proj = nn.Conv2d(256, 64, kernel_size=(1, 1),
            stride=(1, 1))
        self.inception_3b_pool_proj_bn = nn.BatchNorm2d(64, affine=True)
        self.inception_3b_relu_pool_proj = nn.ReLU(inplace)
        self.inception_3c_3x3_reduce = nn.Conv2d(320, 128, kernel_size=(1, 
            1), stride=(1, 1))
        self.inception_3c_3x3_reduce_bn = nn.BatchNorm2d(128, affine=True)
        self.inception_3c_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_3c_3x3 = nn.Conv2d(128, 160, kernel_size=(3, 3),
            stride=(2, 2), padding=(1, 1))
        self.inception_3c_3x3_bn = nn.BatchNorm2d(160, affine=True)
        self.inception_3c_relu_3x3 = nn.ReLU(inplace)
        self.inception_3c_double_3x3_reduce = nn.Conv2d(320, 64,
            kernel_size=(1, 1), stride=(1, 1))
        self.inception_3c_double_3x3_reduce_bn = nn.BatchNorm2d(64, affine=True
            )
        self.inception_3c_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_3c_double_3x3_1 = nn.Conv2d(64, 96, kernel_size=(3, 
            3), stride=(1, 1), padding=(1, 1))
        self.inception_3c_double_3x3_1_bn = nn.BatchNorm2d(96, affine=True)
        self.inception_3c_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_3c_double_3x3_2 = nn.Conv2d(96, 96, kernel_size=(3, 
            3), stride=(2, 2), padding=(1, 1))
        self.inception_3c_double_3x3_2_bn = nn.BatchNorm2d(96, affine=True)
        self.inception_3c_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_3c_pool = nn.MaxPool2d((3, 3), stride=(2, 2),
            dilation=(1, 1), ceil_mode=True)
        self.inception_4a_1x1 = nn.Conv2d(576, 224, kernel_size=(1, 1),
            stride=(1, 1))
        self.inception_4a_1x1_bn = nn.BatchNorm2d(224, affine=True)
        self.inception_4a_relu_1x1 = nn.ReLU(inplace)
        self.inception_4a_3x3_reduce = nn.Conv2d(576, 64, kernel_size=(1, 1
            ), stride=(1, 1))
        self.inception_4a_3x3_reduce_bn = nn.BatchNorm2d(64, affine=True)
        self.inception_4a_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_4a_3x3 = nn.Conv2d(64, 96, kernel_size=(3, 3),
            stride=(1, 1), padding=(1, 1))
        self.inception_4a_3x3_bn = nn.BatchNorm2d(96, affine=True)
        self.inception_4a_relu_3x3 = nn.ReLU(inplace)
        self.inception_4a_double_3x3_reduce = nn.Conv2d(576, 96,
            kernel_size=(1, 1), stride=(1, 1))
        self.inception_4a_double_3x3_reduce_bn = nn.BatchNorm2d(96, affine=True
            )
        self.inception_4a_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_4a_double_3x3_1 = nn.Conv2d(96, 128, kernel_size=(3,
            3), stride=(1, 1), padding=(1, 1))
        self.inception_4a_double_3x3_1_bn = nn.BatchNorm2d(128, affine=True)
        self.inception_4a_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_4a_double_3x3_2 = nn.Conv2d(128, 128, kernel_size=(3,
            3), stride=(1, 1), padding=(1, 1))
        self.inception_4a_double_3x3_2_bn = nn.BatchNorm2d(128, affine=True)
        self.inception_4a_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_4a_pool = nn.AvgPool2d(3, stride=1, padding=1,
            ceil_mode=True, count_include_pad=True)
        self.inception_4a_pool_proj = nn.Conv2d(576, 128, kernel_size=(1, 1
            ), stride=(1, 1))
        self.inception_4a_pool_proj_bn = nn.BatchNorm2d(128, affine=True)
        self.inception_4a_relu_pool_proj = nn.ReLU(inplace)
        self.inception_4b_1x1 = nn.Conv2d(576, 192, kernel_size=(1, 1),
            stride=(1, 1))
        self.inception_4b_1x1_bn = nn.BatchNorm2d(192, affine=True)
        self.inception_4b_relu_1x1 = nn.ReLU(inplace)
        self.inception_4b_3x3_reduce = nn.Conv2d(576, 96, kernel_size=(1, 1
            ), stride=(1, 1))
        self.inception_4b_3x3_reduce_bn = nn.BatchNorm2d(96, affine=True)
        self.inception_4b_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_4b_3x3 = nn.Conv2d(96, 128, kernel_size=(3, 3),
            stride=(1, 1), padding=(1, 1))
        self.inception_4b_3x3_bn = nn.BatchNorm2d(128, affine=True)
        self.inception_4b_relu_3x3 = nn.ReLU(inplace)
        self.inception_4b_double_3x3_reduce = nn.Conv2d(576, 96,
            kernel_size=(1, 1), stride=(1, 1))
        self.inception_4b_double_3x3_reduce_bn = nn.BatchNorm2d(96, affine=True
            )
        self.inception_4b_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_4b_double_3x3_1 = nn.Conv2d(96, 128, kernel_size=(3,
            3), stride=(1, 1), padding=(1, 1))
        self.inception_4b_double_3x3_1_bn = nn.BatchNorm2d(128, affine=True)
        self.inception_4b_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_4b_double_3x3_2 = nn.Conv2d(128, 128, kernel_size=(3,
            3), stride=(1, 1), padding=(1, 1))
        self.inception_4b_double_3x3_2_bn = nn.BatchNorm2d(128, affine=True)
        self.inception_4b_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_4b_pool = nn.AvgPool2d(3, stride=1, padding=1,
            ceil_mode=True, count_include_pad=True)
        self.inception_4b_pool_proj = nn.Conv2d(576, 128, kernel_size=(1, 1
            ), stride=(1, 1))
        self.inception_4b_pool_proj_bn = nn.BatchNorm2d(128, affine=True)
        self.inception_4b_relu_pool_proj = nn.ReLU(inplace)
        self.inception_4c_1x1 = nn.Conv2d(576, 160, kernel_size=(1, 1),
            stride=(1, 1))
        self.inception_4c_1x1_bn = nn.BatchNorm2d(160, affine=True)
        self.inception_4c_relu_1x1 = nn.ReLU(inplace)
        self.inception_4c_3x3_reduce = nn.Conv2d(576, 128, kernel_size=(1, 
            1), stride=(1, 1))
        self.inception_4c_3x3_reduce_bn = nn.BatchNorm2d(128, affine=True)
        self.inception_4c_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_4c_3x3 = nn.Conv2d(128, 160, kernel_size=(3, 3),
            stride=(1, 1), padding=(1, 1))
        self.inception_4c_3x3_bn = nn.BatchNorm2d(160, affine=True)
        self.inception_4c_relu_3x3 = nn.ReLU(inplace)
        self.inception_4c_double_3x3_reduce = nn.Conv2d(576, 128,
            kernel_size=(1, 1), stride=(1, 1))
        self.inception_4c_double_3x3_reduce_bn = nn.BatchNorm2d(128, affine
            =True)
        self.inception_4c_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_4c_double_3x3_1 = nn.Conv2d(128, 160, kernel_size=(3,
            3), stride=(1, 1), padding=(1, 1))
        self.inception_4c_double_3x3_1_bn = nn.BatchNorm2d(160, affine=True)
        self.inception_4c_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_4c_double_3x3_2 = nn.Conv2d(160, 160, kernel_size=(3,
            3), stride=(1, 1), padding=(1, 1))
        self.inception_4c_double_3x3_2_bn = nn.BatchNorm2d(160, affine=True)
        self.inception_4c_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_4c_pool = nn.AvgPool2d(3, stride=1, padding=1,
            ceil_mode=True, count_include_pad=True)
        self.inception_4c_pool_proj = nn.Conv2d(576, 128, kernel_size=(1, 1
            ), stride=(1, 1))
        self.inception_4c_pool_proj_bn = nn.BatchNorm2d(128, affine=True)
        self.inception_4c_relu_pool_proj = nn.ReLU(inplace)
        self.inception_4d_1x1 = nn.Conv2d(608, 96, kernel_size=(1, 1),
            stride=(1, 1))
        self.inception_4d_1x1_bn = nn.BatchNorm2d(96, affine=True)
        self.inception_4d_relu_1x1 = nn.ReLU(inplace)
        self.inception_4d_3x3_reduce = nn.Conv2d(608, 128, kernel_size=(1, 
            1), stride=(1, 1))
        self.inception_4d_3x3_reduce_bn = nn.BatchNorm2d(128, affine=True)
        self.inception_4d_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_4d_3x3 = nn.Conv2d(128, 192, kernel_size=(3, 3),
            stride=(1, 1), padding=(1, 1))
        self.inception_4d_3x3_bn = nn.BatchNorm2d(192, affine=True)
        self.inception_4d_relu_3x3 = nn.ReLU(inplace)
        self.inception_4d_double_3x3_reduce = nn.Conv2d(608, 160,
            kernel_size=(1, 1), stride=(1, 1))
        self.inception_4d_double_3x3_reduce_bn = nn.BatchNorm2d(160, affine
            =True)
        self.inception_4d_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_4d_double_3x3_1 = nn.Conv2d(160, 192, kernel_size=(3,
            3), stride=(1, 1), padding=(1, 1))
        self.inception_4d_double_3x3_1_bn = nn.BatchNorm2d(192, affine=True)
        self.inception_4d_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_4d_double_3x3_2 = nn.Conv2d(192, 192, kernel_size=(3,
            3), stride=(1, 1), padding=(1, 1))
        self.inception_4d_double_3x3_2_bn = nn.BatchNorm2d(192, affine=True)
        self.inception_4d_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_4d_pool = nn.AvgPool2d(3, stride=1, padding=1,
            ceil_mode=True, count_include_pad=True)
        self.inception_4d_pool_proj = nn.Conv2d(608, 128, kernel_size=(1, 1
            ), stride=(1, 1))
        self.inception_4d_pool_proj_bn = nn.BatchNorm2d(128, affine=True)
        self.inception_4d_relu_pool_proj = nn.ReLU(inplace)
        self.inception_4e_3x3_reduce = nn.Conv2d(608, 128, kernel_size=(1, 
            1), stride=(1, 1))
        self.inception_4e_3x3_reduce_bn = nn.BatchNorm2d(128, affine=True)
        self.inception_4e_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_4e_3x3 = nn.Conv2d(128, 192, kernel_size=(3, 3),
            stride=(2, 2), padding=(1, 1))
        self.inception_4e_3x3_bn = nn.BatchNorm2d(192, affine=True)
        self.inception_4e_relu_3x3 = nn.ReLU(inplace)
        self.inception_4e_double_3x3_reduce = nn.Conv2d(608, 192,
            kernel_size=(1, 1), stride=(1, 1))
        self.inception_4e_double_3x3_reduce_bn = nn.BatchNorm2d(192, affine
            =True)
        self.inception_4e_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_4e_double_3x3_1 = nn.Conv2d(192, 256, kernel_size=(3,
            3), stride=(1, 1), padding=(1, 1))
        self.inception_4e_double_3x3_1_bn = nn.BatchNorm2d(256, affine=True)
        self.inception_4e_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_4e_double_3x3_2 = nn.Conv2d(256, 256, kernel_size=(3,
            3), stride=(2, 2), padding=(1, 1))
        self.inception_4e_double_3x3_2_bn = nn.BatchNorm2d(256, affine=True)
        self.inception_4e_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_4e_pool = nn.MaxPool2d((3, 3), stride=(2, 2),
            dilation=(1, 1), ceil_mode=True)
        self.inception_5a_1x1 = nn.Conv2d(1056, 352, kernel_size=(1, 1),
            stride=(1, 1))
        self.inception_5a_1x1_bn = nn.BatchNorm2d(352, affine=True)
        self.inception_5a_relu_1x1 = nn.ReLU(inplace)
        self.inception_5a_3x3_reduce = nn.Conv2d(1056, 192, kernel_size=(1,
            1), stride=(1, 1))
        self.inception_5a_3x3_reduce_bn = nn.BatchNorm2d(192, affine=True)
        self.inception_5a_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_5a_3x3 = nn.Conv2d(192, 320, kernel_size=(3, 3),
            stride=(1, 1), padding=(1, 1))
        self.inception_5a_3x3_bn = nn.BatchNorm2d(320, affine=True)
        self.inception_5a_relu_3x3 = nn.ReLU(inplace)
        self.inception_5a_double_3x3_reduce = nn.Conv2d(1056, 160,
            kernel_size=(1, 1), stride=(1, 1))
        self.inception_5a_double_3x3_reduce_bn = nn.BatchNorm2d(160, affine
            =True)
        self.inception_5a_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_5a_double_3x3_1 = nn.Conv2d(160, 224, kernel_size=(3,
            3), stride=(1, 1), padding=(1, 1))
        self.inception_5a_double_3x3_1_bn = nn.BatchNorm2d(224, affine=True)
        self.inception_5a_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_5a_double_3x3_2 = nn.Conv2d(224, 224, kernel_size=(3,
            3), stride=(1, 1), padding=(1, 1))
        self.inception_5a_double_3x3_2_bn = nn.BatchNorm2d(224, affine=True)
        self.inception_5a_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_5a_pool = nn.AvgPool2d(3, stride=1, padding=1,
            ceil_mode=True, count_include_pad=True)
        self.inception_5a_pool_proj = nn.Conv2d(1056, 128, kernel_size=(1, 
            1), stride=(1, 1))
        self.inception_5a_pool_proj_bn = nn.BatchNorm2d(128, affine=True)
        self.inception_5a_relu_pool_proj = nn.ReLU(inplace)
        self.inception_5b_1x1 = nn.Conv2d(1024, 352, kernel_size=(1, 1),
            stride=(1, 1))
        self.inception_5b_1x1_bn = nn.BatchNorm2d(352, affine=True)
        self.inception_5b_relu_1x1 = nn.ReLU(inplace)
        self.inception_5b_3x3_reduce = nn.Conv2d(1024, 192, kernel_size=(1,
            1), stride=(1, 1))
        self.inception_5b_3x3_reduce_bn = nn.BatchNorm2d(192, affine=True)
        self.inception_5b_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_5b_3x3 = nn.Conv2d(192, 320, kernel_size=(3, 3),
            stride=(1, 1), padding=(1, 1))
        self.inception_5b_3x3_bn = nn.BatchNorm2d(320, affine=True)
        self.inception_5b_relu_3x3 = nn.ReLU(inplace)
        self.inception_5b_double_3x3_reduce = nn.Conv2d(1024, 192,
            kernel_size=(1, 1), stride=(1, 1))
        self.inception_5b_double_3x3_reduce_bn = nn.BatchNorm2d(192, affine
            =True)
        self.inception_5b_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_5b_double_3x3_1 = nn.Conv2d(192, 224, kernel_size=(3,
            3), stride=(1, 1), padding=(1, 1))
        self.inception_5b_double_3x3_1_bn = nn.BatchNorm2d(224, affine=True)
        self.inception_5b_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_5b_double_3x3_2 = nn.Conv2d(224, 224, kernel_size=(3,
            3), stride=(1, 1), padding=(1, 1))
        self.inception_5b_double_3x3_2_bn = nn.BatchNorm2d(224, affine=True)
        self.inception_5b_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_5b_pool = nn.MaxPool2d((3, 3), stride=(1, 1),
            padding=(1, 1), dilation=(1, 1), ceil_mode=True)
        self.inception_5b_pool_proj = nn.Conv2d(1024, 128, kernel_size=(1, 
            1), stride=(1, 1))
        self.inception_5b_pool_proj_bn = nn.BatchNorm2d(128, affine=True)
        self.inception_5b_relu_pool_proj = nn.ReLU(inplace)
        self.last_linear = nn.Linear(1024, num_classes)

    def features(self, input):
        conv1_7x7_s2_out = self.conv1_7x7_s2(input)
        conv1_7x7_s2_bn_out = self.conv1_7x7_s2_bn(conv1_7x7_s2_out)
        conv1_relu_7x7_out = self.conv1_relu_7x7(conv1_7x7_s2_bn_out)
        pool1_3x3_s2_out = self.pool1_3x3_s2(conv1_relu_7x7_out)
        conv2_3x3_reduce_out = self.conv2_3x3_reduce(pool1_3x3_s2_out)
        conv2_3x3_reduce_bn_out = self.conv2_3x3_reduce_bn(conv2_3x3_reduce_out
            )
        conv2_relu_3x3_reduce_out = self.conv2_relu_3x3_reduce(
            conv2_3x3_reduce_bn_out)
        conv2_3x3_out = self.conv2_3x3(conv2_relu_3x3_reduce_out)
        conv2_3x3_bn_out = self.conv2_3x3_bn(conv2_3x3_out)
        conv2_relu_3x3_out = self.conv2_relu_3x3(conv2_3x3_bn_out)
        pool2_3x3_s2_out = self.pool2_3x3_s2(conv2_relu_3x3_out)
        inception_3a_1x1_out = self.inception_3a_1x1(pool2_3x3_s2_out)
        inception_3a_1x1_bn_out = self.inception_3a_1x1_bn(inception_3a_1x1_out
            )
        inception_3a_relu_1x1_out = self.inception_3a_relu_1x1(
            inception_3a_1x1_bn_out)
        inception_3a_3x3_reduce_out = self.inception_3a_3x3_reduce(
            pool2_3x3_s2_out)
        inception_3a_3x3_reduce_bn_out = self.inception_3a_3x3_reduce_bn(
            inception_3a_3x3_reduce_out)
        inception_3a_relu_3x3_reduce_out = self.inception_3a_relu_3x3_reduce(
            inception_3a_3x3_reduce_bn_out)
        inception_3a_3x3_out = self.inception_3a_3x3(
            inception_3a_relu_3x3_reduce_out)
        inception_3a_3x3_bn_out = self.inception_3a_3x3_bn(inception_3a_3x3_out
            )
        inception_3a_relu_3x3_out = self.inception_3a_relu_3x3(
            inception_3a_3x3_bn_out)
        inception_3a_double_3x3_reduce_out = (self.
            inception_3a_double_3x3_reduce(pool2_3x3_s2_out))
        inception_3a_double_3x3_reduce_bn_out = (self.
            inception_3a_double_3x3_reduce_bn(
            inception_3a_double_3x3_reduce_out))
        inception_3a_relu_double_3x3_reduce_out = (self.
            inception_3a_relu_double_3x3_reduce(
            inception_3a_double_3x3_reduce_bn_out))
        inception_3a_double_3x3_1_out = self.inception_3a_double_3x3_1(
            inception_3a_relu_double_3x3_reduce_out)
        inception_3a_double_3x3_1_bn_out = self.inception_3a_double_3x3_1_bn(
            inception_3a_double_3x3_1_out)
        inception_3a_relu_double_3x3_1_out = (self.
            inception_3a_relu_double_3x3_1(inception_3a_double_3x3_1_bn_out))
        inception_3a_double_3x3_2_out = self.inception_3a_double_3x3_2(
            inception_3a_relu_double_3x3_1_out)
        inception_3a_double_3x3_2_bn_out = self.inception_3a_double_3x3_2_bn(
            inception_3a_double_3x3_2_out)
        inception_3a_relu_double_3x3_2_out = (self.
            inception_3a_relu_double_3x3_2(inception_3a_double_3x3_2_bn_out))
        inception_3a_pool_out = self.inception_3a_pool(pool2_3x3_s2_out)
        inception_3a_pool_proj_out = self.inception_3a_pool_proj(
            inception_3a_pool_out)
        inception_3a_pool_proj_bn_out = self.inception_3a_pool_proj_bn(
            inception_3a_pool_proj_out)
        inception_3a_relu_pool_proj_out = self.inception_3a_relu_pool_proj(
            inception_3a_pool_proj_bn_out)
        inception_3a_output_out = torch.cat([inception_3a_relu_1x1_out,
            inception_3a_relu_3x3_out, inception_3a_relu_double_3x3_2_out,
            inception_3a_relu_pool_proj_out], 1)
        inception_3b_1x1_out = self.inception_3b_1x1(inception_3a_output_out)
        inception_3b_1x1_bn_out = self.inception_3b_1x1_bn(inception_3b_1x1_out
            )
        inception_3b_relu_1x1_out = self.inception_3b_relu_1x1(
            inception_3b_1x1_bn_out)
        inception_3b_3x3_reduce_out = self.inception_3b_3x3_reduce(
            inception_3a_output_out)
        inception_3b_3x3_reduce_bn_out = self.inception_3b_3x3_reduce_bn(
            inception_3b_3x3_reduce_out)
        inception_3b_relu_3x3_reduce_out = self.inception_3b_relu_3x3_reduce(
            inception_3b_3x3_reduce_bn_out)
        inception_3b_3x3_out = self.inception_3b_3x3(
            inception_3b_relu_3x3_reduce_out)
        inception_3b_3x3_bn_out = self.inception_3b_3x3_bn(inception_3b_3x3_out
            )
        inception_3b_relu_3x3_out = self.inception_3b_relu_3x3(
            inception_3b_3x3_bn_out)
        inception_3b_double_3x3_reduce_out = (self.
            inception_3b_double_3x3_reduce(inception_3a_output_out))
        inception_3b_double_3x3_reduce_bn_out = (self.
            inception_3b_double_3x3_reduce_bn(
            inception_3b_double_3x3_reduce_out))
        inception_3b_relu_double_3x3_reduce_out = (self.
            inception_3b_relu_double_3x3_reduce(
            inception_3b_double_3x3_reduce_bn_out))
        inception_3b_double_3x3_1_out = self.inception_3b_double_3x3_1(
            inception_3b_relu_double_3x3_reduce_out)
        inception_3b_double_3x3_1_bn_out = self.inception_3b_double_3x3_1_bn(
            inception_3b_double_3x3_1_out)
        inception_3b_relu_double_3x3_1_out = (self.
            inception_3b_relu_double_3x3_1(inception_3b_double_3x3_1_bn_out))
        inception_3b_double_3x3_2_out = self.inception_3b_double_3x3_2(
            inception_3b_relu_double_3x3_1_out)
        inception_3b_double_3x3_2_bn_out = self.inception_3b_double_3x3_2_bn(
            inception_3b_double_3x3_2_out)
        inception_3b_relu_double_3x3_2_out = (self.
            inception_3b_relu_double_3x3_2(inception_3b_double_3x3_2_bn_out))
        inception_3b_pool_out = self.inception_3b_pool(inception_3a_output_out)
        inception_3b_pool_proj_out = self.inception_3b_pool_proj(
            inception_3b_pool_out)
        inception_3b_pool_proj_bn_out = self.inception_3b_pool_proj_bn(
            inception_3b_pool_proj_out)
        inception_3b_relu_pool_proj_out = self.inception_3b_relu_pool_proj(
            inception_3b_pool_proj_bn_out)
        inception_3b_output_out = torch.cat([inception_3b_relu_1x1_out,
            inception_3b_relu_3x3_out, inception_3b_relu_double_3x3_2_out,
            inception_3b_relu_pool_proj_out], 1)
        inception_3c_3x3_reduce_out = self.inception_3c_3x3_reduce(
            inception_3b_output_out)
        inception_3c_3x3_reduce_bn_out = self.inception_3c_3x3_reduce_bn(
            inception_3c_3x3_reduce_out)
        inception_3c_relu_3x3_reduce_out = self.inception_3c_relu_3x3_reduce(
            inception_3c_3x3_reduce_bn_out)
        inception_3c_3x3_out = self.inception_3c_3x3(
            inception_3c_relu_3x3_reduce_out)
        inception_3c_3x3_bn_out = self.inception_3c_3x3_bn(inception_3c_3x3_out
            )
        inception_3c_relu_3x3_out = self.inception_3c_relu_3x3(
            inception_3c_3x3_bn_out)
        inception_3c_double_3x3_reduce_out = (self.
            inception_3c_double_3x3_reduce(inception_3b_output_out))
        inception_3c_double_3x3_reduce_bn_out = (self.
            inception_3c_double_3x3_reduce_bn(
            inception_3c_double_3x3_reduce_out))
        inception_3c_relu_double_3x3_reduce_out = (self.
            inception_3c_relu_double_3x3_reduce(
            inception_3c_double_3x3_reduce_bn_out))
        inception_3c_double_3x3_1_out = self.inception_3c_double_3x3_1(
            inception_3c_relu_double_3x3_reduce_out)
        inception_3c_double_3x3_1_bn_out = self.inception_3c_double_3x3_1_bn(
            inception_3c_double_3x3_1_out)
        inception_3c_relu_double_3x3_1_out = (self.
            inception_3c_relu_double_3x3_1(inception_3c_double_3x3_1_bn_out))
        inception_3c_double_3x3_2_out = self.inception_3c_double_3x3_2(
            inception_3c_relu_double_3x3_1_out)
        inception_3c_double_3x3_2_bn_out = self.inception_3c_double_3x3_2_bn(
            inception_3c_double_3x3_2_out)
        inception_3c_relu_double_3x3_2_out = (self.
            inception_3c_relu_double_3x3_2(inception_3c_double_3x3_2_bn_out))
        inception_3c_pool_out = self.inception_3c_pool(inception_3b_output_out)
        inception_3c_output_out = torch.cat([inception_3c_relu_3x3_out,
            inception_3c_relu_double_3x3_2_out, inception_3c_pool_out], 1)
        inception_4a_1x1_out = self.inception_4a_1x1(inception_3c_output_out)
        inception_4a_1x1_bn_out = self.inception_4a_1x1_bn(inception_4a_1x1_out
            )
        inception_4a_relu_1x1_out = self.inception_4a_relu_1x1(
            inception_4a_1x1_bn_out)
        inception_4a_3x3_reduce_out = self.inception_4a_3x3_reduce(
            inception_3c_output_out)
        inception_4a_3x3_reduce_bn_out = self.inception_4a_3x3_reduce_bn(
            inception_4a_3x3_reduce_out)
        inception_4a_relu_3x3_reduce_out = self.inception_4a_relu_3x3_reduce(
            inception_4a_3x3_reduce_bn_out)
        inception_4a_3x3_out = self.inception_4a_3x3(
            inception_4a_relu_3x3_reduce_out)
        inception_4a_3x3_bn_out = self.inception_4a_3x3_bn(inception_4a_3x3_out
            )
        inception_4a_relu_3x3_out = self.inception_4a_relu_3x3(
            inception_4a_3x3_bn_out)
        inception_4a_double_3x3_reduce_out = (self.
            inception_4a_double_3x3_reduce(inception_3c_output_out))
        inception_4a_double_3x3_reduce_bn_out = (self.
            inception_4a_double_3x3_reduce_bn(
            inception_4a_double_3x3_reduce_out))
        inception_4a_relu_double_3x3_reduce_out = (self.
            inception_4a_relu_double_3x3_reduce(
            inception_4a_double_3x3_reduce_bn_out))
        inception_4a_double_3x3_1_out = self.inception_4a_double_3x3_1(
            inception_4a_relu_double_3x3_reduce_out)
        inception_4a_double_3x3_1_bn_out = self.inception_4a_double_3x3_1_bn(
            inception_4a_double_3x3_1_out)
        inception_4a_relu_double_3x3_1_out = (self.
            inception_4a_relu_double_3x3_1(inception_4a_double_3x3_1_bn_out))
        inception_4a_double_3x3_2_out = self.inception_4a_double_3x3_2(
            inception_4a_relu_double_3x3_1_out)
        inception_4a_double_3x3_2_bn_out = self.inception_4a_double_3x3_2_bn(
            inception_4a_double_3x3_2_out)
        inception_4a_relu_double_3x3_2_out = (self.
            inception_4a_relu_double_3x3_2(inception_4a_double_3x3_2_bn_out))
        inception_4a_pool_out = self.inception_4a_pool(inception_3c_output_out)
        inception_4a_pool_proj_out = self.inception_4a_pool_proj(
            inception_4a_pool_out)
        inception_4a_pool_proj_bn_out = self.inception_4a_pool_proj_bn(
            inception_4a_pool_proj_out)
        inception_4a_relu_pool_proj_out = self.inception_4a_relu_pool_proj(
            inception_4a_pool_proj_bn_out)
        inception_4a_output_out = torch.cat([inception_4a_relu_1x1_out,
            inception_4a_relu_3x3_out, inception_4a_relu_double_3x3_2_out,
            inception_4a_relu_pool_proj_out], 1)
        inception_4b_1x1_out = self.inception_4b_1x1(inception_4a_output_out)
        inception_4b_1x1_bn_out = self.inception_4b_1x1_bn(inception_4b_1x1_out
            )
        inception_4b_relu_1x1_out = self.inception_4b_relu_1x1(
            inception_4b_1x1_bn_out)
        inception_4b_3x3_reduce_out = self.inception_4b_3x3_reduce(
            inception_4a_output_out)
        inception_4b_3x3_reduce_bn_out = self.inception_4b_3x3_reduce_bn(
            inception_4b_3x3_reduce_out)
        inception_4b_relu_3x3_reduce_out = self.inception_4b_relu_3x3_reduce(
            inception_4b_3x3_reduce_bn_out)
        inception_4b_3x3_out = self.inception_4b_3x3(
            inception_4b_relu_3x3_reduce_out)
        inception_4b_3x3_bn_out = self.inception_4b_3x3_bn(inception_4b_3x3_out
            )
        inception_4b_relu_3x3_out = self.inception_4b_relu_3x3(
            inception_4b_3x3_bn_out)
        inception_4b_double_3x3_reduce_out = (self.
            inception_4b_double_3x3_reduce(inception_4a_output_out))
        inception_4b_double_3x3_reduce_bn_out = (self.
            inception_4b_double_3x3_reduce_bn(
            inception_4b_double_3x3_reduce_out))
        inception_4b_relu_double_3x3_reduce_out = (self.
            inception_4b_relu_double_3x3_reduce(
            inception_4b_double_3x3_reduce_bn_out))
        inception_4b_double_3x3_1_out = self.inception_4b_double_3x3_1(
            inception_4b_relu_double_3x3_reduce_out)
        inception_4b_double_3x3_1_bn_out = self.inception_4b_double_3x3_1_bn(
            inception_4b_double_3x3_1_out)
        inception_4b_relu_double_3x3_1_out = (self.
            inception_4b_relu_double_3x3_1(inception_4b_double_3x3_1_bn_out))
        inception_4b_double_3x3_2_out = self.inception_4b_double_3x3_2(
            inception_4b_relu_double_3x3_1_out)
        inception_4b_double_3x3_2_bn_out = self.inception_4b_double_3x3_2_bn(
            inception_4b_double_3x3_2_out)
        inception_4b_relu_double_3x3_2_out = (self.
            inception_4b_relu_double_3x3_2(inception_4b_double_3x3_2_bn_out))
        inception_4b_pool_out = self.inception_4b_pool(inception_4a_output_out)
        inception_4b_pool_proj_out = self.inception_4b_pool_proj(
            inception_4b_pool_out)
        inception_4b_pool_proj_bn_out = self.inception_4b_pool_proj_bn(
            inception_4b_pool_proj_out)
        inception_4b_relu_pool_proj_out = self.inception_4b_relu_pool_proj(
            inception_4b_pool_proj_bn_out)
        inception_4b_output_out = torch.cat([inception_4b_relu_1x1_out,
            inception_4b_relu_3x3_out, inception_4b_relu_double_3x3_2_out,
            inception_4b_relu_pool_proj_out], 1)
        inception_4c_1x1_out = self.inception_4c_1x1(inception_4b_output_out)
        inception_4c_1x1_bn_out = self.inception_4c_1x1_bn(inception_4c_1x1_out
            )
        inception_4c_relu_1x1_out = self.inception_4c_relu_1x1(
            inception_4c_1x1_bn_out)
        inception_4c_3x3_reduce_out = self.inception_4c_3x3_reduce(
            inception_4b_output_out)
        inception_4c_3x3_reduce_bn_out = self.inception_4c_3x3_reduce_bn(
            inception_4c_3x3_reduce_out)
        inception_4c_relu_3x3_reduce_out = self.inception_4c_relu_3x3_reduce(
            inception_4c_3x3_reduce_bn_out)
        inception_4c_3x3_out = self.inception_4c_3x3(
            inception_4c_relu_3x3_reduce_out)
        inception_4c_3x3_bn_out = self.inception_4c_3x3_bn(inception_4c_3x3_out
            )
        inception_4c_relu_3x3_out = self.inception_4c_relu_3x3(
            inception_4c_3x3_bn_out)
        inception_4c_double_3x3_reduce_out = (self.
            inception_4c_double_3x3_reduce(inception_4b_output_out))
        inception_4c_double_3x3_reduce_bn_out = (self.
            inception_4c_double_3x3_reduce_bn(
            inception_4c_double_3x3_reduce_out))
        inception_4c_relu_double_3x3_reduce_out = (self.
            inception_4c_relu_double_3x3_reduce(
            inception_4c_double_3x3_reduce_bn_out))
        inception_4c_double_3x3_1_out = self.inception_4c_double_3x3_1(
            inception_4c_relu_double_3x3_reduce_out)
        inception_4c_double_3x3_1_bn_out = self.inception_4c_double_3x3_1_bn(
            inception_4c_double_3x3_1_out)
        inception_4c_relu_double_3x3_1_out = (self.
            inception_4c_relu_double_3x3_1(inception_4c_double_3x3_1_bn_out))
        inception_4c_double_3x3_2_out = self.inception_4c_double_3x3_2(
            inception_4c_relu_double_3x3_1_out)
        inception_4c_double_3x3_2_bn_out = self.inception_4c_double_3x3_2_bn(
            inception_4c_double_3x3_2_out)
        inception_4c_relu_double_3x3_2_out = (self.
            inception_4c_relu_double_3x3_2(inception_4c_double_3x3_2_bn_out))
        inception_4c_pool_out = self.inception_4c_pool(inception_4b_output_out)
        inception_4c_pool_proj_out = self.inception_4c_pool_proj(
            inception_4c_pool_out)
        inception_4c_pool_proj_bn_out = self.inception_4c_pool_proj_bn(
            inception_4c_pool_proj_out)
        inception_4c_relu_pool_proj_out = self.inception_4c_relu_pool_proj(
            inception_4c_pool_proj_bn_out)
        inception_4c_output_out = torch.cat([inception_4c_relu_1x1_out,
            inception_4c_relu_3x3_out, inception_4c_relu_double_3x3_2_out,
            inception_4c_relu_pool_proj_out], 1)
        inception_4d_1x1_out = self.inception_4d_1x1(inception_4c_output_out)
        inception_4d_1x1_bn_out = self.inception_4d_1x1_bn(inception_4d_1x1_out
            )
        inception_4d_relu_1x1_out = self.inception_4d_relu_1x1(
            inception_4d_1x1_bn_out)
        inception_4d_3x3_reduce_out = self.inception_4d_3x3_reduce(
            inception_4c_output_out)
        inception_4d_3x3_reduce_bn_out = self.inception_4d_3x3_reduce_bn(
            inception_4d_3x3_reduce_out)
        inception_4d_relu_3x3_reduce_out = self.inception_4d_relu_3x3_reduce(
            inception_4d_3x3_reduce_bn_out)
        inception_4d_3x3_out = self.inception_4d_3x3(
            inception_4d_relu_3x3_reduce_out)
        inception_4d_3x3_bn_out = self.inception_4d_3x3_bn(inception_4d_3x3_out
            )
        inception_4d_relu_3x3_out = self.inception_4d_relu_3x3(
            inception_4d_3x3_bn_out)
        inception_4d_double_3x3_reduce_out = (self.
            inception_4d_double_3x3_reduce(inception_4c_output_out))
        inception_4d_double_3x3_reduce_bn_out = (self.
            inception_4d_double_3x3_reduce_bn(
            inception_4d_double_3x3_reduce_out))
        inception_4d_relu_double_3x3_reduce_out = (self.
            inception_4d_relu_double_3x3_reduce(
            inception_4d_double_3x3_reduce_bn_out))
        inception_4d_double_3x3_1_out = self.inception_4d_double_3x3_1(
            inception_4d_relu_double_3x3_reduce_out)
        inception_4d_double_3x3_1_bn_out = self.inception_4d_double_3x3_1_bn(
            inception_4d_double_3x3_1_out)
        inception_4d_relu_double_3x3_1_out = (self.
            inception_4d_relu_double_3x3_1(inception_4d_double_3x3_1_bn_out))
        inception_4d_double_3x3_2_out = self.inception_4d_double_3x3_2(
            inception_4d_relu_double_3x3_1_out)
        inception_4d_double_3x3_2_bn_out = self.inception_4d_double_3x3_2_bn(
            inception_4d_double_3x3_2_out)
        inception_4d_relu_double_3x3_2_out = (self.
            inception_4d_relu_double_3x3_2(inception_4d_double_3x3_2_bn_out))
        inception_4d_pool_out = self.inception_4d_pool(inception_4c_output_out)
        inception_4d_pool_proj_out = self.inception_4d_pool_proj(
            inception_4d_pool_out)
        inception_4d_pool_proj_bn_out = self.inception_4d_pool_proj_bn(
            inception_4d_pool_proj_out)
        inception_4d_relu_pool_proj_out = self.inception_4d_relu_pool_proj(
            inception_4d_pool_proj_bn_out)
        inception_4d_output_out = torch.cat([inception_4d_relu_1x1_out,
            inception_4d_relu_3x3_out, inception_4d_relu_double_3x3_2_out,
            inception_4d_relu_pool_proj_out], 1)
        inception_4e_3x3_reduce_out = self.inception_4e_3x3_reduce(
            inception_4d_output_out)
        inception_4e_3x3_reduce_bn_out = self.inception_4e_3x3_reduce_bn(
            inception_4e_3x3_reduce_out)
        inception_4e_relu_3x3_reduce_out = self.inception_4e_relu_3x3_reduce(
            inception_4e_3x3_reduce_bn_out)
        inception_4e_3x3_out = self.inception_4e_3x3(
            inception_4e_relu_3x3_reduce_out)
        inception_4e_3x3_bn_out = self.inception_4e_3x3_bn(inception_4e_3x3_out
            )
        inception_4e_relu_3x3_out = self.inception_4e_relu_3x3(
            inception_4e_3x3_bn_out)
        inception_4e_double_3x3_reduce_out = (self.
            inception_4e_double_3x3_reduce(inception_4d_output_out))
        inception_4e_double_3x3_reduce_bn_out = (self.
            inception_4e_double_3x3_reduce_bn(
            inception_4e_double_3x3_reduce_out))
        inception_4e_relu_double_3x3_reduce_out = (self.
            inception_4e_relu_double_3x3_reduce(
            inception_4e_double_3x3_reduce_bn_out))
        inception_4e_double_3x3_1_out = self.inception_4e_double_3x3_1(
            inception_4e_relu_double_3x3_reduce_out)
        inception_4e_double_3x3_1_bn_out = self.inception_4e_double_3x3_1_bn(
            inception_4e_double_3x3_1_out)
        inception_4e_relu_double_3x3_1_out = (self.
            inception_4e_relu_double_3x3_1(inception_4e_double_3x3_1_bn_out))
        inception_4e_double_3x3_2_out = self.inception_4e_double_3x3_2(
            inception_4e_relu_double_3x3_1_out)
        inception_4e_double_3x3_2_bn_out = self.inception_4e_double_3x3_2_bn(
            inception_4e_double_3x3_2_out)
        inception_4e_relu_double_3x3_2_out = (self.
            inception_4e_relu_double_3x3_2(inception_4e_double_3x3_2_bn_out))
        inception_4e_pool_out = self.inception_4e_pool(inception_4d_output_out)
        inception_4e_output_out = torch.cat([inception_4e_relu_3x3_out,
            inception_4e_relu_double_3x3_2_out, inception_4e_pool_out], 1)
        inception_5a_1x1_out = self.inception_5a_1x1(inception_4e_output_out)
        inception_5a_1x1_bn_out = self.inception_5a_1x1_bn(inception_5a_1x1_out
            )
        inception_5a_relu_1x1_out = self.inception_5a_relu_1x1(
            inception_5a_1x1_bn_out)
        inception_5a_3x3_reduce_out = self.inception_5a_3x3_reduce(
            inception_4e_output_out)
        inception_5a_3x3_reduce_bn_out = self.inception_5a_3x3_reduce_bn(
            inception_5a_3x3_reduce_out)
        inception_5a_relu_3x3_reduce_out = self.inception_5a_relu_3x3_reduce(
            inception_5a_3x3_reduce_bn_out)
        inception_5a_3x3_out = self.inception_5a_3x3(
            inception_5a_relu_3x3_reduce_out)
        inception_5a_3x3_bn_out = self.inception_5a_3x3_bn(inception_5a_3x3_out
            )
        inception_5a_relu_3x3_out = self.inception_5a_relu_3x3(
            inception_5a_3x3_bn_out)
        inception_5a_double_3x3_reduce_out = (self.
            inception_5a_double_3x3_reduce(inception_4e_output_out))
        inception_5a_double_3x3_reduce_bn_out = (self.
            inception_5a_double_3x3_reduce_bn(
            inception_5a_double_3x3_reduce_out))
        inception_5a_relu_double_3x3_reduce_out = (self.
            inception_5a_relu_double_3x3_reduce(
            inception_5a_double_3x3_reduce_bn_out))
        inception_5a_double_3x3_1_out = self.inception_5a_double_3x3_1(
            inception_5a_relu_double_3x3_reduce_out)
        inception_5a_double_3x3_1_bn_out = self.inception_5a_double_3x3_1_bn(
            inception_5a_double_3x3_1_out)
        inception_5a_relu_double_3x3_1_out = (self.
            inception_5a_relu_double_3x3_1(inception_5a_double_3x3_1_bn_out))
        inception_5a_double_3x3_2_out = self.inception_5a_double_3x3_2(
            inception_5a_relu_double_3x3_1_out)
        inception_5a_double_3x3_2_bn_out = self.inception_5a_double_3x3_2_bn(
            inception_5a_double_3x3_2_out)
        inception_5a_relu_double_3x3_2_out = (self.
            inception_5a_relu_double_3x3_2(inception_5a_double_3x3_2_bn_out))
        inception_5a_pool_out = self.inception_5a_pool(inception_4e_output_out)
        inception_5a_pool_proj_out = self.inception_5a_pool_proj(
            inception_5a_pool_out)
        inception_5a_pool_proj_bn_out = self.inception_5a_pool_proj_bn(
            inception_5a_pool_proj_out)
        inception_5a_relu_pool_proj_out = self.inception_5a_relu_pool_proj(
            inception_5a_pool_proj_bn_out)
        inception_5a_output_out = torch.cat([inception_5a_relu_1x1_out,
            inception_5a_relu_3x3_out, inception_5a_relu_double_3x3_2_out,
            inception_5a_relu_pool_proj_out], 1)
        inception_5b_1x1_out = self.inception_5b_1x1(inception_5a_output_out)
        inception_5b_1x1_bn_out = self.inception_5b_1x1_bn(inception_5b_1x1_out
            )
        inception_5b_relu_1x1_out = self.inception_5b_relu_1x1(
            inception_5b_1x1_bn_out)
        inception_5b_3x3_reduce_out = self.inception_5b_3x3_reduce(
            inception_5a_output_out)
        inception_5b_3x3_reduce_bn_out = self.inception_5b_3x3_reduce_bn(
            inception_5b_3x3_reduce_out)
        inception_5b_relu_3x3_reduce_out = self.inception_5b_relu_3x3_reduce(
            inception_5b_3x3_reduce_bn_out)
        inception_5b_3x3_out = self.inception_5b_3x3(
            inception_5b_relu_3x3_reduce_out)
        inception_5b_3x3_bn_out = self.inception_5b_3x3_bn(inception_5b_3x3_out
            )
        inception_5b_relu_3x3_out = self.inception_5b_relu_3x3(
            inception_5b_3x3_bn_out)
        inception_5b_double_3x3_reduce_out = (self.
            inception_5b_double_3x3_reduce(inception_5a_output_out))
        inception_5b_double_3x3_reduce_bn_out = (self.
            inception_5b_double_3x3_reduce_bn(
            inception_5b_double_3x3_reduce_out))
        inception_5b_relu_double_3x3_reduce_out = (self.
            inception_5b_relu_double_3x3_reduce(
            inception_5b_double_3x3_reduce_bn_out))
        inception_5b_double_3x3_1_out = self.inception_5b_double_3x3_1(
            inception_5b_relu_double_3x3_reduce_out)
        inception_5b_double_3x3_1_bn_out = self.inception_5b_double_3x3_1_bn(
            inception_5b_double_3x3_1_out)
        inception_5b_relu_double_3x3_1_out = (self.
            inception_5b_relu_double_3x3_1(inception_5b_double_3x3_1_bn_out))
        inception_5b_double_3x3_2_out = self.inception_5b_double_3x3_2(
            inception_5b_relu_double_3x3_1_out)
        inception_5b_double_3x3_2_bn_out = self.inception_5b_double_3x3_2_bn(
            inception_5b_double_3x3_2_out)
        inception_5b_relu_double_3x3_2_out = (self.
            inception_5b_relu_double_3x3_2(inception_5b_double_3x3_2_bn_out))
        inception_5b_pool_out = self.inception_5b_pool(inception_5a_output_out)
        inception_5b_pool_proj_out = self.inception_5b_pool_proj(
            inception_5b_pool_out)
        inception_5b_pool_proj_bn_out = self.inception_5b_pool_proj_bn(
            inception_5b_pool_proj_out)
        inception_5b_relu_pool_proj_out = self.inception_5b_relu_pool_proj(
            inception_5b_pool_proj_bn_out)
        inception_5b_output_out = torch.cat([inception_5b_relu_1x1_out,
            inception_5b_relu_3x3_out, inception_5b_relu_double_3x3_2_out,
            inception_5b_relu_pool_proj_out], 1)
        return inception_5b_output_out

    def logits(self, features):
        adaptiveAvgPoolWidth = features.shape[2]
        x = F.avg_pool2d(features, kernel_size=adaptiveAvgPoolWidth)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x


def pooling_factor(pool_type='avg'):
    return 2 if pool_type == 'avgmaxc' else 1


class AdaptiveAvgMaxPool2d(torch.nn.Module):
    """Selectable global pooling layer with dynamic input kernel size
    """

    def __init__(self, output_size=1, pool_type='avg'):
        super(AdaptiveAvgMaxPool2d, self).__init__()
        self.output_size = output_size
        self.pool_type = pool_type
        if pool_type == 'avgmaxc' or pool_type == 'avgmax':
            self.pool = nn.ModuleList([nn.AdaptiveAvgPool2d(output_size),
                nn.AdaptiveMaxPool2d(output_size)])
        elif pool_type == 'max':
            self.pool = nn.AdaptiveMaxPool2d(output_size)
        else:
            if pool_type != 'avg':
                None
            self.pool = nn.AdaptiveAvgPool2d(output_size)

    def forward(self, x):
        if self.pool_type == 'avgmaxc':
            x = torch.cat([p(x) for p in self.pool], dim=1)
        elif self.pool_type == 'avgmax':
            x = 0.5 * torch.sum(torch.stack([p(x) for p in self.pool]), 0
                ).squeeze(dim=0)
        else:
            x = self.pool(x)
        return x

    def factor(self):
        return pooling_factor(self.pool_type)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + 'output_size=' + str(self.
            output_size) + ', pool_type=' + self.pool_type + ')'


class CatBnAct(nn.Module):

    def __init__(self, in_chs, activation_fn=nn.ReLU(inplace=True)):
        super(CatBnAct, self).__init__()
        self.bn = nn.BatchNorm2d(in_chs, eps=0.001)
        self.act = activation_fn

    def forward(self, x):
        x = torch.cat(x, dim=1) if isinstance(x, tuple) else x
        return self.act(self.bn(x))


class BnActConv2d(nn.Module):

    def __init__(self, in_chs, out_chs, kernel_size, stride, padding=0,
        groups=1, activation_fn=nn.ReLU(inplace=True)):
        super(BnActConv2d, self).__init__()
        self.bn = nn.BatchNorm2d(in_chs, eps=0.001)
        self.act = activation_fn
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride, padding,
            groups=groups, bias=False)

    def forward(self, x):
        return self.conv(self.act(self.bn(x)))


class InputBlock(nn.Module):

    def __init__(self, num_init_features, kernel_size=7, padding=3,
        activation_fn=nn.ReLU(inplace=True)):
        super(InputBlock, self).__init__()
        self.conv = nn.Conv2d(3, num_init_features, kernel_size=kernel_size,
            stride=2, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(num_init_features, eps=0.001)
        self.act = activation_fn
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        return x


class DualPathBlock(nn.Module):

    def __init__(self, in_chs, num_1x1_a, num_3x3_b, num_1x1_c, inc, groups,
        block_type='normal', b=False):
        super(DualPathBlock, self).__init__()
        self.num_1x1_c = num_1x1_c
        self.inc = inc
        self.b = b
        if block_type == 'proj':
            self.key_stride = 1
            self.has_proj = True
        elif block_type == 'down':
            self.key_stride = 2
            self.has_proj = True
        else:
            assert block_type is 'normal'
            self.key_stride = 1
            self.has_proj = False
        if self.has_proj:
            if self.key_stride == 2:
                self.c1x1_w_s2 = BnActConv2d(in_chs=in_chs, out_chs=
                    num_1x1_c + 2 * inc, kernel_size=1, stride=2)
            else:
                self.c1x1_w_s1 = BnActConv2d(in_chs=in_chs, out_chs=
                    num_1x1_c + 2 * inc, kernel_size=1, stride=1)
        self.c1x1_a = BnActConv2d(in_chs=in_chs, out_chs=num_1x1_a,
            kernel_size=1, stride=1)
        self.c3x3_b = BnActConv2d(in_chs=num_1x1_a, out_chs=num_3x3_b,
            kernel_size=3, stride=self.key_stride, padding=1, groups=groups)
        if b:
            self.c1x1_c = CatBnAct(in_chs=num_3x3_b)
            self.c1x1_c1 = nn.Conv2d(num_3x3_b, num_1x1_c, kernel_size=1,
                bias=False)
            self.c1x1_c2 = nn.Conv2d(num_3x3_b, inc, kernel_size=1, bias=False)
        else:
            self.c1x1_c = BnActConv2d(in_chs=num_3x3_b, out_chs=num_1x1_c +
                inc, kernel_size=1, stride=1)

    def forward(self, x):
        x_in = torch.cat(x, dim=1) if isinstance(x, tuple) else x
        if self.has_proj:
            if self.key_stride == 2:
                x_s = self.c1x1_w_s2(x_in)
            else:
                x_s = self.c1x1_w_s1(x_in)
            x_s1 = x_s[:, :self.num_1x1_c, :, :]
            x_s2 = x_s[:, self.num_1x1_c:, :, :]
        else:
            x_s1 = x[0]
            x_s2 = x[1]
        x_in = self.c1x1_a(x_in)
        x_in = self.c3x3_b(x_in)
        if self.b:
            x_in = self.c1x1_c(x_in)
            out1 = self.c1x1_c1(x_in)
            out2 = self.c1x1_c2(x_in)
        else:
            x_in = self.c1x1_c(x_in)
            out1 = x_in[:, :self.num_1x1_c, :, :]
            out2 = x_in[:, self.num_1x1_c:, :, :]
        resid = x_s1 + out1
        dense = torch.cat([x_s2, out2], dim=1)
        return resid, dense


def adaptive_avgmax_pool2d(x, pool_type='avg', padding=0, count_include_pad
    =False):
    """Selectable global pooling function with dynamic input kernel size
    """
    if pool_type == 'avgmaxc':
        x = torch.cat([F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)),
            padding=padding, count_include_pad=count_include_pad), F.
            max_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=
            padding)], dim=1)
    elif pool_type == 'avgmax':
        x_avg = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding
            =padding, count_include_pad=count_include_pad)
        x_max = F.max_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding
            =padding)
        x = 0.5 * (x_avg + x_max)
    elif pool_type == 'max':
        x = F.max_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=padding
            )
    else:
        if pool_type != 'avg':
            print(
                'Invalid pool type %s specified. Defaulting to average pooling.'
                 % pool_type)
        x = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=
            padding, count_include_pad=count_include_pad)
    return x


class DPN(nn.Module):

    def __init__(self, small=False, num_init_features=64, k_r=96, groups=32,
        b=False, k_sec=(3, 4, 20, 3), inc_sec=(16, 32, 24, 128),
        num_classes=1000, test_time_pool=False):
        super(DPN, self).__init__()
        self.test_time_pool = test_time_pool
        self.b = b
        bw_factor = 1 if small else 4
        blocks = OrderedDict()
        if small:
            blocks['conv1_1'] = InputBlock(num_init_features, kernel_size=3,
                padding=1)
        else:
            blocks['conv1_1'] = InputBlock(num_init_features, kernel_size=7,
                padding=3)
        bw = 64 * bw_factor
        inc = inc_sec[0]
        r = k_r * bw // (64 * bw_factor)
        blocks['conv2_1'] = DualPathBlock(num_init_features, r, r, bw, inc,
            groups, 'proj', b)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[0] + 1):
            blocks['conv2_' + str(i)] = DualPathBlock(in_chs, r, r, bw, inc,
                groups, 'normal', b)
            in_chs += inc
        bw = 128 * bw_factor
        inc = inc_sec[1]
        r = k_r * bw // (64 * bw_factor)
        blocks['conv3_1'] = DualPathBlock(in_chs, r, r, bw, inc, groups,
            'down', b)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[1] + 1):
            blocks['conv3_' + str(i)] = DualPathBlock(in_chs, r, r, bw, inc,
                groups, 'normal', b)
            in_chs += inc
        bw = 256 * bw_factor
        inc = inc_sec[2]
        r = k_r * bw // (64 * bw_factor)
        blocks['conv4_1'] = DualPathBlock(in_chs, r, r, bw, inc, groups,
            'down', b)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[2] + 1):
            blocks['conv4_' + str(i)] = DualPathBlock(in_chs, r, r, bw, inc,
                groups, 'normal', b)
            in_chs += inc
        bw = 512 * bw_factor
        inc = inc_sec[3]
        r = k_r * bw // (64 * bw_factor)
        blocks['conv5_1'] = DualPathBlock(in_chs, r, r, bw, inc, groups,
            'down', b)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[3] + 1):
            blocks['conv5_' + str(i)] = DualPathBlock(in_chs, r, r, bw, inc,
                groups, 'normal', b)
            in_chs += inc
        blocks['conv5_bn_ac'] = CatBnAct(in_chs)
        self.features = nn.Sequential(blocks)
        self.last_linear = nn.Conv2d(in_chs, num_classes, kernel_size=1,
            bias=True)

    def forward(self, x):
        x = self.features(x)
        if not self.training and self.test_time_pool:
            x = F.avg_pool2d(x, kernel_size=7, stride=1)
            out = self.last_linear(x)
            out = adaptive_avgmax_pool2d(out, pool_type='avgmax')
        else:
            x = adaptive_avgmax_pool2d(x, pool_type='avg')
            out = self.last_linear(x)
        return out.view(out.size(0), -1)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=True)
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
        out += residual
        out = self.relu(out)
        return out


class FBResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        self.input_space = None
        self.input_size = 299, 299, 3
        self.mean = None
        self.std = None
        super(FBResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
            bias=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.last_linear = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def features(self, input):
        x = self.conv1(input)
        self.conv1_input = x.clone()
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def logits(self, features):
        adaptiveAvgPoolWidth = features.shape[2]
        x = F.avg_pool2d(features, kernel_size=adaptiveAvgPoolWidth)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x


class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=
            kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes, eps=0.001, momentum=0, affine=True
            )
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class StemBlock(nn.Module):
    """
    input 299*299*3
    output 35*35*384
    """

    def __init__(self):
        super(StemBlock, self).__init__()
        self.model_a = nn.Sequential(BasicConv2d(3, 32, kernel_size=3,
            stride=2), BasicConv2d(32, 32, kernel_size=3, stride=1, padding
            =1), BasicConv2d(32, 64, kernel_size=3, stride=1))
        self.branch_a0 = nn.MaxPool2d(3, stride=2)
        self.branch_a1 = BasicConv2d(64, 96, kernel_size=3, stride=2)
        self.branch_b0 = nn.Sequential(BasicConv2d(160, 64, kernel_size=1,
            stride=1), BasicConv2d(64, 96, kernel_size=3, stride=1))
        self.branch_b1 = nn.Sequential(BasicConv2d(160, 64, kernel_size=1,
            stride=1), BasicConv2d(64, 64, kernel_size=(1, 7), stride=1,
            padding=(0, 3)), BasicConv2d(64, 64, kernel_size=(7, 1), stride
            =1, padding=(3, 0)), BasicConv2d(64, 96, kernel_size=3, stride=1))
        self.branch_c0 = BasicConv2d(192, 192, kernel_size=3, stride=2)
        self.branch_c1 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x = self.model_a(x)
        x_0 = self.branch_a0(x)
        x_1 = self.branch_a1(x)
        x = torch.cat((x_0, x_1), 1)
        x_0 = self.branch_b0(x)
        x_1 = self.branch_b1(x)
        x = torch.cat((x_0, x_1), 1)
        x_0 = self.branch_c0(x)
        x_1 = self.branch_c1(x)
        x = torch.cat((x_0, x_1), 1)
        return x


class InceptionResA(nn.Module):
    """
    input 35*35*384
    output 35*35*384
    """

    def __init__(self, scale=1.0):
        super(InceptionResA, self).__init__()
        self.relu = nn.ReLU(inplace=False)
        self.scale = scale
        self.branch_0 = BasicConv2d(384, 32, kernel_size=1, stride=1)
        self.branch_1 = nn.Sequential(BasicConv2d(384, 32, kernel_size=1,
            stride=1), BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1))
        self.branch_2 = nn.Sequential(BasicConv2d(384, 32, kernel_size=1,
            stride=1), BasicConv2d(32, 48, kernel_size=3, stride=1, padding
            =1), BasicConv2d(48, 64, kernel_size=3, stride=1, padding=1))
        self.branch_all = BasicConv2d(128, 384, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.relu(x)
        x_0 = self.branch_0(x)
        x_1 = self.branch_1(x)
        x_2 = self.branch_2(x)
        x_new = torch.cat((x_0, x_1, x_2), 1)
        x_new = self.branch_all(x_new)
        x = x + x_new * self.scale
        return x


class ReductionA(nn.Module):
    """
    input 35*35*384
    output 17*17*1152
    """

    def __init__(self):
        super(ReductionA, self).__init__()
        self.branch_0 = nn.MaxPool2d(3, stride=2)
        self.branch_1 = BasicConv2d(384, 384, kernel_size=3, stride=2)
        self.branch_2 = nn.Sequential(BasicConv2d(384, 256, kernel_size=1,
            stride=1), BasicConv2d(256, 256, kernel_size=3, stride=1,
            padding=1), BasicConv2d(256, 384, kernel_size=3, stride=2))

    def forward(self, x):
        x_0 = self.branch_0(x)
        x_1 = self.branch_1(x)
        x_2 = self.branch_2(x)
        return torch.cat((x_0, x_1, x_2), 1)


class InceptionResB(nn.Module):
    """
    input 17*17*1152
    output 17*17*1152
    """

    def __init__(self, scale=1.0):
        super(InceptionResB, self).__init__()
        self.relu = nn.ReLU(inplace=False)
        self.scale = scale
        self.branch_0 = BasicConv2d(1152, 192, kernel_size=1, stride=1)
        self.branch_1 = nn.Sequential(BasicConv2d(1152, 128, kernel_size=1,
            stride=1), BasicConv2d(128, 160, kernel_size=(1, 7), stride=1,
            padding=(0, 3)), BasicConv2d(160, 192, kernel_size=(7, 1),
            stride=1, padding=(3, 0)))
        self.branch_all = BasicConv2d(384, 1152, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.relu(x)
        x_0 = self.branch_0(x)
        x_1 = self.branch_1(x)
        x_new = torch.cat((x_0, x_1), 1)
        x_new = self.branch_all(x_new)
        x = x + x_new * self.scale
        return x


class ReductionB(nn.Module):
    """
    input 17*17*1152
    ouput 8*8*2144
    """

    def __init__(self):
        super(ReductionB, self).__init__()
        self.branch_0 = nn.MaxPool2d(3, stride=2)
        self.branch_1 = nn.Sequential(BasicConv2d(1152, 256, kernel_size=1,
            stride=1), BasicConv2d(256, 384, kernel_size=3, stride=2))
        self.branch_2 = nn.Sequential(BasicConv2d(1152, 256, kernel_size=1,
            stride=1), BasicConv2d(256, 288, kernel_size=3, stride=2))
        self.branch_3 = nn.Sequential(BasicConv2d(1152, 256, kernel_size=1,
            stride=1), BasicConv2d(256, 288, kernel_size=3, stride=1,
            padding=1), BasicConv2d(288, 320, kernel_size=3, stride=2))

    def forward(self, x):
        x_0 = self.branch_0(x)
        x_1 = self.branch_1(x)
        x_2 = self.branch_2(x)
        x_3 = self.branch_3(x)
        return torch.cat((x_0, x_1, x_2, x_3), 1)


class InceptionResC(nn.Module):
    """
    input 8*8*2144
    output 8*8*2144
    """

    def __init__(self, scale=1.0):
        super(InceptionResC, self).__init__()
        self.scale = scale
        self.relu = nn.ReLU(inplace=False)
        self.branch_0 = BasicConv2d(2144, 192, kernel_size=1, stride=1)
        self.branch_1 = nn.Sequential(BasicConv2d(2144, 192, kernel_size=1,
            stride=1), BasicConv2d(192, 224, kernel_size=(1, 3), stride=1,
            padding=(0, 1)), BasicConv2d(224, 256, kernel_size=(3, 1),
            stride=1, padding=(1, 0)))
        self.branch_all = BasicConv2d(448, 2144, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.relu(x)
        x_0 = self.branch_0(x)
        x_1 = self.branch_1(x)
        x_new = torch.cat((x_0, x_1), 1)
        x_new = self.branch_all(x_new)
        x = x + x_new * self.scale
        return x


class InceptionResV2(nn.Module):

    def __init__(self, num_classes=1000):
        super(InceptionResV2, self).__init__()
        self.stem = StemBlock()
        self.inception_resA5 = nn.Sequential(InceptionResA(), InceptionResA
            (), InceptionResA(), InceptionResA(), InceptionResA())
        self.reductionA = ReductionA()
        self.inception_resB10 = nn.Sequential(InceptionResB(),
            InceptionResB(), InceptionResB(), InceptionResB(),
            InceptionResB(), InceptionResB(), InceptionResB(),
            InceptionResB(), InceptionResB(), InceptionResB())
        self.reductionB = ReductionB()
        self.inception_resC5 = nn.Sequential(InceptionResC(), InceptionResC
            (), InceptionResC(), InceptionResC(), InceptionResC())
        self.avg_pool = nn.AvgPool2d(8, count_include_pad=False)
        self.dropout = nn.Dropout2d(p=0.8)
        self.last_linear = nn.Linear(2144, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.inception_resA5(x)
        x = self.reductionA(x)
        x = self.inception_resB10(x)
        x = self.reductionB(x)
        x = self.inception_resC5(x)
        x = self.avg_pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x


class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=
            kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes, eps=0.001, momentum=0.1,
            affine=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Mixed_5b(nn.Module):

    def __init__(self):
        super(Mixed_5b, self).__init__()
        self.branch0 = BasicConv2d(192, 96, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(192, 48, kernel_size=1,
            stride=1), BasicConv2d(48, 64, kernel_size=5, stride=1, padding=2))
        self.branch2 = nn.Sequential(BasicConv2d(192, 64, kernel_size=1,
            stride=1), BasicConv2d(64, 96, kernel_size=3, stride=1, padding
            =1), BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1))
        self.branch3 = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False), BasicConv2d(192, 64, kernel_size=1,
            stride=1))

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Block35(nn.Module):

    def __init__(self, scale=1.0):
        super(Block35, self).__init__()
        self.scale = scale
        self.branch0 = BasicConv2d(320, 32, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(320, 32, kernel_size=1,
            stride=1), BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1))
        self.branch2 = nn.Sequential(BasicConv2d(320, 32, kernel_size=1,
            stride=1), BasicConv2d(32, 48, kernel_size=3, stride=1, padding
            =1), BasicConv2d(48, 64, kernel_size=3, stride=1, padding=1))
        self.conv2d = nn.Conv2d(128, 320, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Mixed_6a(nn.Module):

    def __init__(self):
        super(Mixed_6a, self).__init__()
        self.branch0 = BasicConv2d(320, 384, kernel_size=3, stride=2)
        self.branch1 = nn.Sequential(BasicConv2d(320, 256, kernel_size=1,
            stride=1), BasicConv2d(256, 256, kernel_size=3, stride=1,
            padding=1), BasicConv2d(256, 384, kernel_size=3, stride=2))
        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Block17(nn.Module):

    def __init__(self, scale=1.0):
        super(Block17, self).__init__()
        self.scale = scale
        self.branch0 = BasicConv2d(1088, 192, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(1088, 128, kernel_size=1,
            stride=1), BasicConv2d(128, 160, kernel_size=(1, 7), stride=1,
            padding=(0, 3)), BasicConv2d(160, 192, kernel_size=(7, 1),
            stride=1, padding=(3, 0)))
        self.conv2d = nn.Conv2d(384, 1088, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Mixed_7a(nn.Module):

    def __init__(self):
        super(Mixed_7a, self).__init__()
        self.branch0 = nn.Sequential(BasicConv2d(1088, 256, kernel_size=1,
            stride=1), BasicConv2d(256, 384, kernel_size=3, stride=2))
        self.branch1 = nn.Sequential(BasicConv2d(1088, 256, kernel_size=1,
            stride=1), BasicConv2d(256, 288, kernel_size=3, stride=2))
        self.branch2 = nn.Sequential(BasicConv2d(1088, 256, kernel_size=1,
            stride=1), BasicConv2d(256, 288, kernel_size=3, stride=1,
            padding=1), BasicConv2d(288, 320, kernel_size=3, stride=2))
        self.branch3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Block8(nn.Module):

    def __init__(self, scale=1.0, noReLU=False):
        super(Block8, self).__init__()
        self.scale = scale
        self.noReLU = noReLU
        self.branch0 = BasicConv2d(2080, 192, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(2080, 192, kernel_size=1,
            stride=1), BasicConv2d(192, 224, kernel_size=(1, 3), stride=1,
            padding=(0, 1)), BasicConv2d(224, 256, kernel_size=(3, 1),
            stride=1, padding=(1, 0)))
        self.conv2d = nn.Conv2d(448, 2080, kernel_size=1, stride=1)
        if not self.noReLU:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        if not self.noReLU:
            out = self.relu(out)
        return out


class InceptionResNetV2(nn.Module):

    def __init__(self, num_classes=1001):
        super(InceptionResNetV2, self).__init__()
        self.input_space = None
        self.input_size = 299, 299, 3
        self.mean = None
        self.std = None
        self.conv2d_1a = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1)
        self.conv2d_2b = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1
            )
        self.maxpool_3a = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1, stride=1)
        self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3, stride=1)
        self.maxpool_5a = nn.MaxPool2d(3, stride=2)
        self.mixed_5b = Mixed_5b()
        self.repeat = nn.Sequential(Block35(scale=0.17), Block35(scale=0.17
            ), Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17
            ), Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17
            ), Block35(scale=0.17), Block35(scale=0.17))
        self.mixed_6a = Mixed_6a()
        self.repeat_1 = nn.Sequential(Block17(scale=0.1), Block17(scale=0.1
            ), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1),
            Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1),
            Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1),
            Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1),
            Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1),
            Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1))
        self.mixed_7a = Mixed_7a()
        self.repeat_2 = nn.Sequential(Block8(scale=0.2), Block8(scale=0.2),
            Block8(scale=0.2), Block8(scale=0.2), Block8(scale=0.2), Block8
            (scale=0.2), Block8(scale=0.2), Block8(scale=0.2), Block8(scale
            =0.2))
        self.block8 = Block8(noReLU=True)
        self.conv2d_7b = BasicConv2d(2080, 1536, kernel_size=1, stride=1)
        self.avgpool_1a = nn.AvgPool2d(8, count_include_pad=False)
        self.last_linear = nn.Linear(1536, num_classes)

    def features(self, input):
        x = self.conv2d_1a(input)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.maxpool_5a(x)
        x = self.mixed_5b(x)
        x = self.repeat(x)
        x = self.mixed_6a(x)
        x = self.repeat_1(x)
        x = self.mixed_7a(x)
        x = self.repeat_2(x)
        x = self.block8(x)
        x = self.conv2d_7b(x)
        return x

    def logits(self, features):
        x = self.avgpool_1a(features)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x


class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=
            kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes, eps=0.001, momentum=0.1,
            affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Mixed_3a(nn.Module):

    def __init__(self):
        super(Mixed_3a, self).__init__()
        self.maxpool = nn.MaxPool2d(3, stride=2)
        self.conv = BasicConv2d(64, 96, kernel_size=3, stride=2)

    def forward(self, x):
        x0 = self.maxpool(x)
        x1 = self.conv(x)
        out = torch.cat((x0, x1), 1)
        return out


class Mixed_4a(nn.Module):

    def __init__(self):
        super(Mixed_4a, self).__init__()
        self.branch0 = nn.Sequential(BasicConv2d(160, 64, kernel_size=1,
            stride=1), BasicConv2d(64, 96, kernel_size=3, stride=1))
        self.branch1 = nn.Sequential(BasicConv2d(160, 64, kernel_size=1,
            stride=1), BasicConv2d(64, 64, kernel_size=(1, 7), stride=1,
            padding=(0, 3)), BasicConv2d(64, 64, kernel_size=(7, 1), stride
            =1, padding=(3, 0)), BasicConv2d(64, 96, kernel_size=(3, 3),
            stride=1))

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        return out


class Mixed_5a(nn.Module):

    def __init__(self):
        super(Mixed_5a, self).__init__()
        self.conv = BasicConv2d(192, 192, kernel_size=3, stride=2)
        self.maxpool = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.conv(x)
        x1 = self.maxpool(x)
        out = torch.cat((x0, x1), 1)
        return out


class Inception_A(nn.Module):

    def __init__(self):
        super(Inception_A, self).__init__()
        self.branch0 = BasicConv2d(384, 96, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(384, 64, kernel_size=1,
            stride=1), BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1))
        self.branch2 = nn.Sequential(BasicConv2d(384, 64, kernel_size=1,
            stride=1), BasicConv2d(64, 96, kernel_size=3, stride=1, padding
            =1), BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1))
        self.branch3 = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False), BasicConv2d(384, 96, kernel_size=1,
            stride=1))

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Reduction_A(nn.Module):

    def __init__(self):
        super(Reduction_A, self).__init__()
        self.branch0 = BasicConv2d(384, 384, kernel_size=3, stride=2)
        self.branch1 = nn.Sequential(BasicConv2d(384, 192, kernel_size=1,
            stride=1), BasicConv2d(192, 224, kernel_size=3, stride=1,
            padding=1), BasicConv2d(224, 256, kernel_size=3, stride=2))
        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Inception_B(nn.Module):

    def __init__(self):
        super(Inception_B, self).__init__()
        self.branch0 = BasicConv2d(1024, 384, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(1024, 192, kernel_size=1,
            stride=1), BasicConv2d(192, 224, kernel_size=(1, 7), stride=1,
            padding=(0, 3)), BasicConv2d(224, 256, kernel_size=(7, 1),
            stride=1, padding=(3, 0)))
        self.branch2 = nn.Sequential(BasicConv2d(1024, 192, kernel_size=1,
            stride=1), BasicConv2d(192, 192, kernel_size=(7, 1), stride=1,
            padding=(3, 0)), BasicConv2d(192, 224, kernel_size=(1, 7),
            stride=1, padding=(0, 3)), BasicConv2d(224, 224, kernel_size=(7,
            1), stride=1, padding=(3, 0)), BasicConv2d(224, 256,
            kernel_size=(1, 7), stride=1, padding=(0, 3)))
        self.branch3 = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False), BasicConv2d(1024, 128, kernel_size=1,
            stride=1))

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Reduction_B(nn.Module):

    def __init__(self):
        super(Reduction_B, self).__init__()
        self.branch0 = nn.Sequential(BasicConv2d(1024, 192, kernel_size=1,
            stride=1), BasicConv2d(192, 192, kernel_size=3, stride=2))
        self.branch1 = nn.Sequential(BasicConv2d(1024, 256, kernel_size=1,
            stride=1), BasicConv2d(256, 256, kernel_size=(1, 7), stride=1,
            padding=(0, 3)), BasicConv2d(256, 320, kernel_size=(7, 1),
            stride=1, padding=(3, 0)), BasicConv2d(320, 320, kernel_size=3,
            stride=2))
        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Inception_C(nn.Module):

    def __init__(self):
        super(Inception_C, self).__init__()
        self.branch0 = BasicConv2d(1536, 256, kernel_size=1, stride=1)
        self.branch1_0 = BasicConv2d(1536, 384, kernel_size=1, stride=1)
        self.branch1_1a = BasicConv2d(384, 256, kernel_size=(1, 3), stride=
            1, padding=(0, 1))
        self.branch1_1b = BasicConv2d(384, 256, kernel_size=(3, 1), stride=
            1, padding=(1, 0))
        self.branch2_0 = BasicConv2d(1536, 384, kernel_size=1, stride=1)
        self.branch2_1 = BasicConv2d(384, 448, kernel_size=(3, 1), stride=1,
            padding=(1, 0))
        self.branch2_2 = BasicConv2d(448, 512, kernel_size=(1, 3), stride=1,
            padding=(0, 1))
        self.branch2_3a = BasicConv2d(512, 256, kernel_size=(1, 3), stride=
            1, padding=(0, 1))
        self.branch2_3b = BasicConv2d(512, 256, kernel_size=(3, 1), stride=
            1, padding=(1, 0))
        self.branch3 = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False), BasicConv2d(1536, 256, kernel_size=1,
            stride=1))

    def forward(self, x):
        x0 = self.branch0(x)
        x1_0 = self.branch1_0(x)
        x1_1a = self.branch1_1a(x1_0)
        x1_1b = self.branch1_1b(x1_0)
        x1 = torch.cat((x1_1a, x1_1b), 1)
        x2_0 = self.branch2_0(x)
        x2_1 = self.branch2_1(x2_0)
        x2_2 = self.branch2_2(x2_1)
        x2_3a = self.branch2_3a(x2_2)
        x2_3b = self.branch2_3b(x2_2)
        x2 = torch.cat((x2_3a, x2_3b), 1)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class InceptionV4(nn.Module):

    def __init__(self, num_classes=1001):
        super(InceptionV4, self).__init__()
        self.input_space = None
        self.input_size = 299, 299, 3
        self.mean = None
        self.std = None
        self.features = nn.Sequential(BasicConv2d(3, 32, kernel_size=3,
            stride=2), BasicConv2d(32, 32, kernel_size=3, stride=1),
            BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1),
            Mixed_3a(), Mixed_4a(), Mixed_5a(), Inception_A(), Inception_A(
            ), Inception_A(), Inception_A(), Reduction_A(), Inception_B(),
            Inception_B(), Inception_B(), Inception_B(), Inception_B(),
            Inception_B(), Inception_B(), Reduction_B(), Inception_C(),
            Inception_C(), Inception_C())
        self.last_linear = nn.Linear(1536, num_classes)

    def logits(self, features):
        adaptiveAvgPoolWidth = features.shape[2]
        x = F.avg_pool2d(features, kernel_size=adaptiveAvgPoolWidth)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x


class MaxPoolPad(nn.Module):

    def __init__(self):
        super(MaxPoolPad, self).__init__()
        self.pad = nn.ZeroPad2d((1, 0, 1, 0))
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        x = self.pad(x)
        x = self.pool(x)
        x = x[:, :, 1:, 1:]
        return x


class AvgPoolPad(nn.Module):

    def __init__(self, stride=2, padding=1):
        super(AvgPoolPad, self).__init__()
        self.pad = nn.ZeroPad2d((1, 0, 1, 0))
        self.pool = nn.AvgPool2d(3, stride=stride, padding=padding,
            count_include_pad=False)

    def forward(self, x):
        x = self.pad(x)
        x = self.pool(x)
        x = x[:, :, 1:, 1:]
        return x


class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, dw_kernel, dw_stride,
        dw_padding, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise_conv2d = nn.Conv2d(in_channels, in_channels,
            dw_kernel, stride=dw_stride, padding=dw_padding, bias=bias,
            groups=in_channels)
        self.pointwise_conv2d = nn.Conv2d(in_channels, out_channels, 1,
            stride=1, bias=bias)

    def forward(self, x):
        x = self.depthwise_conv2d(x)
        x = self.pointwise_conv2d(x)
        return x


class BranchSeparables(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
        padding, bias=False):
        super(BranchSeparables, self).__init__()
        self.relu = nn.ReLU()
        self.separable_1 = SeparableConv2d(in_channels, in_channels,
            kernel_size, stride, padding, bias=bias)
        self.bn_sep_1 = nn.BatchNorm2d(in_channels, eps=0.001, momentum=0.1,
            affine=True)
        self.relu1 = nn.ReLU()
        self.separable_2 = SeparableConv2d(in_channels, out_channels,
            kernel_size, 1, padding, bias=bias)
        self.bn_sep_2 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=
            0.1, affine=True)

    def forward(self, x):
        x = self.relu(x)
        x = self.separable_1(x)
        x = self.bn_sep_1(x)
        x = self.relu1(x)
        x = self.separable_2(x)
        x = self.bn_sep_2(x)
        return x


class BranchSeparablesStem(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
        padding, bias=False):
        super(BranchSeparablesStem, self).__init__()
        self.relu = nn.ReLU()
        self.separable_1 = SeparableConv2d(in_channels, out_channels,
            kernel_size, stride, padding, bias=bias)
        self.bn_sep_1 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=
            0.1, affine=True)
        self.relu1 = nn.ReLU()
        self.separable_2 = SeparableConv2d(out_channels, out_channels,
            kernel_size, 1, padding, bias=bias)
        self.bn_sep_2 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=
            0.1, affine=True)

    def forward(self, x):
        x = self.relu(x)
        x = self.separable_1(x)
        x = self.bn_sep_1(x)
        x = self.relu1(x)
        x = self.separable_2(x)
        x = self.bn_sep_2(x)
        return x


class CellStem0(nn.Module):

    def __init__(self):
        super(CellStem0, self).__init__()
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(96, 42, 1, stride=1,
            bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(42, eps=0.001,
            momentum=0.1, affine=True))
        self.comb_iter_0_left = BranchSeparables(42, 42, 5, 2, 2)
        self.comb_iter_0_right = BranchSeparablesStem(96, 42, 7, 2, 3, bias
            =False)
        self.comb_iter_1_left = nn.MaxPool2d(3, stride=2, padding=1)
        self.comb_iter_1_right = BranchSeparablesStem(96, 42, 7, 2, 3, bias
            =False)
        self.comb_iter_2_left = nn.AvgPool2d(3, stride=2, padding=1,
            count_include_pad=False)
        self.comb_iter_2_right = BranchSeparablesStem(96, 42, 5, 2, 2, bias
            =False)
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False)
        self.comb_iter_4_left = BranchSeparables(42, 42, 3, 1, 1, bias=False)
        self.comb_iter_4_right = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        x1 = self.conv_1x1(x)
        x_comb_iter_0_left = self.comb_iter_0_left(x1)
        x_comb_iter_0_right = self.comb_iter_0_right(x)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
        x_comb_iter_1_left = self.comb_iter_1_left(x1)
        x_comb_iter_1_right = self.comb_iter_1_right(x)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
        x_comb_iter_2_left = self.comb_iter_2_left(x1)
        x_comb_iter_2_right = self.comb_iter_2_right(x)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right
        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1
        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x1)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right
        x_out = torch.cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3,
            x_comb_iter_4], 1)
        return x_out


class CellStem1(nn.Module):

    def __init__(self):
        super(CellStem1, self).__init__()
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(168, 84, 1, stride=1,
            bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(84, eps=0.001,
            momentum=0.1, affine=True))
        self.relu = nn.ReLU()
        self.path_1 = nn.Sequential()
        self.path_1.add_module('avgpool', nn.AvgPool2d(1, stride=2,
            count_include_pad=False))
        self.path_1.add_module('conv', nn.Conv2d(96, 42, 1, stride=1, bias=
            False))
        self.path_2 = nn.ModuleList()
        self.path_2.add_module('pad', nn.ZeroPad2d((0, 1, 0, 1)))
        self.path_2.add_module('avgpool', nn.AvgPool2d(1, stride=2,
            count_include_pad=False))
        self.path_2.add_module('conv', nn.Conv2d(96, 42, 1, stride=1, bias=
            False))
        self.final_path_bn = nn.BatchNorm2d(84, eps=0.001, momentum=0.1,
            affine=True)
        self.comb_iter_0_left = BranchSeparables(84, 84, 5, 2, 2, bias=False)
        self.comb_iter_0_right = BranchSeparables(84, 84, 7, 2, 3, bias=False)
        self.comb_iter_1_left = nn.MaxPool2d(3, stride=2, padding=1)
        self.comb_iter_1_right = BranchSeparables(84, 84, 7, 2, 3, bias=False)
        self.comb_iter_2_left = nn.AvgPool2d(3, stride=2, padding=1,
            count_include_pad=False)
        self.comb_iter_2_right = BranchSeparables(84, 84, 5, 2, 2, bias=False)
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False)
        self.comb_iter_4_left = BranchSeparables(84, 84, 3, 1, 1, bias=False)
        self.comb_iter_4_right = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x_conv0, x_stem_0):
        x_left = self.conv_1x1(x_stem_0)
        x_relu = self.relu(x_conv0)
        x_path1 = self.path_1(x_relu)
        x_path2 = self.path_2.pad(x_relu)
        x_path2 = x_path2[:, :, 1:, 1:]
        x_path2 = self.path_2.avgpool(x_path2)
        x_path2 = self.path_2.conv(x_path2)
        x_right = self.final_path_bn(torch.cat([x_path1, x_path2], 1))
        x_comb_iter_0_left = self.comb_iter_0_left(x_left)
        x_comb_iter_0_right = self.comb_iter_0_right(x_right)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
        x_comb_iter_1_left = self.comb_iter_1_left(x_left)
        x_comb_iter_1_right = self.comb_iter_1_right(x_right)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
        x_comb_iter_2_left = self.comb_iter_2_left(x_left)
        x_comb_iter_2_right = self.comb_iter_2_right(x_right)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right
        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1
        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x_left)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right
        x_out = torch.cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3,
            x_comb_iter_4], 1)
        return x_out


class FirstCell(nn.Module):

    def __init__(self, in_channels_left, out_channels_left,
        in_channels_right, out_channels_right):
        super(FirstCell, self).__init__()
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(in_channels_right,
            out_channels_right, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(out_channels_right,
            eps=0.001, momentum=0.1, affine=True))
        self.relu = nn.ReLU()
        self.path_1 = nn.Sequential()
        self.path_1.add_module('avgpool', nn.AvgPool2d(1, stride=2,
            count_include_pad=False))
        self.path_1.add_module('conv', nn.Conv2d(in_channels_left,
            out_channels_left, 1, stride=1, bias=False))
        self.path_2 = nn.ModuleList()
        self.path_2.add_module('pad', nn.ZeroPad2d((0, 1, 0, 1)))
        self.path_2.add_module('avgpool', nn.AvgPool2d(1, stride=2,
            count_include_pad=False))
        self.path_2.add_module('conv', nn.Conv2d(in_channels_left,
            out_channels_left, 1, stride=1, bias=False))
        self.final_path_bn = nn.BatchNorm2d(out_channels_left * 2, eps=
            0.001, momentum=0.1, affine=True)
        self.comb_iter_0_left = BranchSeparables(out_channels_right,
            out_channels_right, 5, 1, 2, bias=False)
        self.comb_iter_0_right = BranchSeparables(out_channels_right,
            out_channels_right, 3, 1, 1, bias=False)
        self.comb_iter_1_left = BranchSeparables(out_channels_right,
            out_channels_right, 5, 1, 2, bias=False)
        self.comb_iter_1_right = BranchSeparables(out_channels_right,
            out_channels_right, 3, 1, 1, bias=False)
        self.comb_iter_2_left = nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False)
        self.comb_iter_3_left = nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False)
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False)
        self.comb_iter_4_left = BranchSeparables(out_channels_right,
            out_channels_right, 3, 1, 1, bias=False)

    def forward(self, x, x_prev):
        x_relu = self.relu(x_prev)
        x_path1 = self.path_1(x_relu)
        x_path2 = self.path_2.pad(x_relu)
        x_path2 = x_path2[:, :, 1:, 1:]
        x_path2 = self.path_2.avgpool(x_path2)
        x_path2 = self.path_2.conv(x_path2)
        x_left = self.final_path_bn(torch.cat([x_path1, x_path2], 1))
        x_right = self.conv_1x1(x)
        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
        x_comb_iter_1_left = self.comb_iter_1_left(x_left)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2 = x_comb_iter_2_left + x_left
        x_comb_iter_3_left = self.comb_iter_3_left(x_left)
        x_comb_iter_3_right = self.comb_iter_3_right(x_left)
        x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right
        x_comb_iter_4_left = self.comb_iter_4_left(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_right
        x_out = torch.cat([x_left, x_comb_iter_0, x_comb_iter_1,
            x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class NormalCell(nn.Module):

    def __init__(self, in_channels_left, out_channels_left,
        in_channels_right, out_channels_right):
        super(NormalCell, self).__init__()
        self.conv_prev_1x1 = nn.Sequential()
        self.conv_prev_1x1.add_module('relu', nn.ReLU())
        self.conv_prev_1x1.add_module('conv', nn.Conv2d(in_channels_left,
            out_channels_left, 1, stride=1, bias=False))
        self.conv_prev_1x1.add_module('bn', nn.BatchNorm2d(
            out_channels_left, eps=0.001, momentum=0.1, affine=True))
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(in_channels_right,
            out_channels_right, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(out_channels_right,
            eps=0.001, momentum=0.1, affine=True))
        self.comb_iter_0_left = BranchSeparables(out_channels_right,
            out_channels_right, 5, 1, 2, bias=False)
        self.comb_iter_0_right = BranchSeparables(out_channels_left,
            out_channels_left, 3, 1, 1, bias=False)
        self.comb_iter_1_left = BranchSeparables(out_channels_left,
            out_channels_left, 5, 1, 2, bias=False)
        self.comb_iter_1_right = BranchSeparables(out_channels_left,
            out_channels_left, 3, 1, 1, bias=False)
        self.comb_iter_2_left = nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False)
        self.comb_iter_3_left = nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False)
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False)
        self.comb_iter_4_left = BranchSeparables(out_channels_right,
            out_channels_right, 3, 1, 1, bias=False)

    def forward(self, x, x_prev):
        x_left = self.conv_prev_1x1(x_prev)
        x_right = self.conv_1x1(x)
        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
        x_comb_iter_1_left = self.comb_iter_1_left(x_left)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2 = x_comb_iter_2_left + x_left
        x_comb_iter_3_left = self.comb_iter_3_left(x_left)
        x_comb_iter_3_right = self.comb_iter_3_right(x_left)
        x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right
        x_comb_iter_4_left = self.comb_iter_4_left(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_right
        x_out = torch.cat([x_left, x_comb_iter_0, x_comb_iter_1,
            x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class BranchSeparablesReduction(BranchSeparables):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
        padding, z_padding=1, bias=False):
        BranchSeparables.__init__(self, in_channels, out_channels,
            kernel_size, stride, padding, bias)
        self.padding = nn.ZeroPad2d((z_padding, 0, z_padding, 0))

    def forward(self, x):
        x = self.relu(x)
        x = self.padding(x)
        x = self.separable_1(x)
        x = x[:, :, 1:, 1:].contiguous()
        x = self.bn_sep_1(x)
        x = self.relu1(x)
        x = self.separable_2(x)
        x = self.bn_sep_2(x)
        return x


class ReductionCell0(nn.Module):

    def __init__(self, in_channels_left, out_channels_left,
        in_channels_right, out_channels_right):
        super(ReductionCell0, self).__init__()
        self.conv_prev_1x1 = nn.Sequential()
        self.conv_prev_1x1.add_module('relu', nn.ReLU())
        self.conv_prev_1x1.add_module('conv', nn.Conv2d(in_channels_left,
            out_channels_left, 1, stride=1, bias=False))
        self.conv_prev_1x1.add_module('bn', nn.BatchNorm2d(
            out_channels_left, eps=0.001, momentum=0.1, affine=True))
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(in_channels_right,
            out_channels_right, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(out_channels_right,
            eps=0.001, momentum=0.1, affine=True))
        self.comb_iter_0_left = BranchSeparablesReduction(out_channels_right,
            out_channels_right, 5, 2, 2, bias=False)
        self.comb_iter_0_right = BranchSeparablesReduction(out_channels_right,
            out_channels_right, 7, 2, 3, bias=False)
        self.comb_iter_1_left = MaxPoolPad()
        self.comb_iter_1_right = BranchSeparablesReduction(out_channels_right,
            out_channels_right, 7, 2, 3, bias=False)
        self.comb_iter_2_left = AvgPoolPad()
        self.comb_iter_2_right = BranchSeparablesReduction(out_channels_right,
            out_channels_right, 5, 2, 2, bias=False)
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False)
        self.comb_iter_4_left = BranchSeparablesReduction(out_channels_right,
            out_channels_right, 3, 1, 1, bias=False)
        self.comb_iter_4_right = MaxPoolPad()

    def forward(self, x, x_prev):
        x_left = self.conv_prev_1x1(x_prev)
        x_right = self.conv_1x1(x)
        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
        x_comb_iter_1_left = self.comb_iter_1_left(x_right)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2_right = self.comb_iter_2_right(x_left)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right
        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1
        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right
        x_out = torch.cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3,
            x_comb_iter_4], 1)
        return x_out


class ReductionCell1(nn.Module):

    def __init__(self, in_channels_left, out_channels_left,
        in_channels_right, out_channels_right):
        super(ReductionCell1, self).__init__()
        self.conv_prev_1x1 = nn.Sequential()
        self.conv_prev_1x1.add_module('relu', nn.ReLU())
        self.conv_prev_1x1.add_module('conv', nn.Conv2d(in_channels_left,
            out_channels_left, 1, stride=1, bias=False))
        self.conv_prev_1x1.add_module('bn', nn.BatchNorm2d(
            out_channels_left, eps=0.001, momentum=0.1, affine=True))
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(in_channels_right,
            out_channels_right, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(out_channels_right,
            eps=0.001, momentum=0.1, affine=True))
        self.comb_iter_0_left = BranchSeparables(out_channels_right,
            out_channels_right, 5, 2, 2, bias=False)
        self.comb_iter_0_right = BranchSeparables(out_channels_right,
            out_channels_right, 7, 2, 3, bias=False)
        self.comb_iter_1_left = nn.MaxPool2d(3, stride=2, padding=1)
        self.comb_iter_1_right = BranchSeparables(out_channels_right,
            out_channels_right, 7, 2, 3, bias=False)
        self.comb_iter_2_left = nn.AvgPool2d(3, stride=2, padding=1,
            count_include_pad=False)
        self.comb_iter_2_right = BranchSeparables(out_channels_right,
            out_channels_right, 5, 2, 2, bias=False)
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False)
        self.comb_iter_4_left = BranchSeparables(out_channels_right,
            out_channels_right, 3, 1, 1, bias=False)
        self.comb_iter_4_right = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x, x_prev):
        x_left = self.conv_prev_1x1(x_prev)
        x_right = self.conv_1x1(x)
        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
        x_comb_iter_1_left = self.comb_iter_1_left(x_right)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2_right = self.comb_iter_2_right(x_left)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right
        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1
        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right
        x_out = torch.cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3,
            x_comb_iter_4], 1)
        return x_out


class NASNetALarge(nn.Module):

    def __init__(self, num_classes=1001):
        super(NASNetALarge, self).__init__()
        self.num_classes = num_classes
        self.conv0 = nn.Sequential()
        self.conv0.add_module('conv', nn.Conv2d(in_channels=3, out_channels
            =96, kernel_size=3, padding=0, stride=2, bias=False))
        self.conv0.add_module('bn', nn.BatchNorm2d(96, eps=0.001, momentum=
            0.1, affine=True))
        self.cell_stem_0 = CellStem0()
        self.cell_stem_1 = CellStem1()
        self.cell_0 = FirstCell(in_channels_left=168, out_channels_left=84,
            in_channels_right=336, out_channels_right=168)
        self.cell_1 = NormalCell(in_channels_left=336, out_channels_left=
            168, in_channels_right=1008, out_channels_right=168)
        self.cell_2 = NormalCell(in_channels_left=1008, out_channels_left=
            168, in_channels_right=1008, out_channels_right=168)
        self.cell_3 = NormalCell(in_channels_left=1008, out_channels_left=
            168, in_channels_right=1008, out_channels_right=168)
        self.cell_4 = NormalCell(in_channels_left=1008, out_channels_left=
            168, in_channels_right=1008, out_channels_right=168)
        self.cell_5 = NormalCell(in_channels_left=1008, out_channels_left=
            168, in_channels_right=1008, out_channels_right=168)
        self.reduction_cell_0 = ReductionCell0(in_channels_left=1008,
            out_channels_left=336, in_channels_right=1008,
            out_channels_right=336)
        self.cell_6 = FirstCell(in_channels_left=1008, out_channels_left=
            168, in_channels_right=1344, out_channels_right=336)
        self.cell_7 = NormalCell(in_channels_left=1344, out_channels_left=
            336, in_channels_right=2016, out_channels_right=336)
        self.cell_8 = NormalCell(in_channels_left=2016, out_channels_left=
            336, in_channels_right=2016, out_channels_right=336)
        self.cell_9 = NormalCell(in_channels_left=2016, out_channels_left=
            336, in_channels_right=2016, out_channels_right=336)
        self.cell_10 = NormalCell(in_channels_left=2016, out_channels_left=
            336, in_channels_right=2016, out_channels_right=336)
        self.cell_11 = NormalCell(in_channels_left=2016, out_channels_left=
            336, in_channels_right=2016, out_channels_right=336)
        self.reduction_cell_1 = ReductionCell1(in_channels_left=2016,
            out_channels_left=672, in_channels_right=2016,
            out_channels_right=672)
        self.cell_12 = FirstCell(in_channels_left=2016, out_channels_left=
            336, in_channels_right=2688, out_channels_right=672)
        self.cell_13 = NormalCell(in_channels_left=2688, out_channels_left=
            672, in_channels_right=4032, out_channels_right=672)
        self.cell_14 = NormalCell(in_channels_left=4032, out_channels_left=
            672, in_channels_right=4032, out_channels_right=672)
        self.cell_15 = NormalCell(in_channels_left=4032, out_channels_left=
            672, in_channels_right=4032, out_channels_right=672)
        self.cell_16 = NormalCell(in_channels_left=4032, out_channels_left=
            672, in_channels_right=4032, out_channels_right=672)
        self.cell_17 = NormalCell(in_channels_left=4032, out_channels_left=
            672, in_channels_right=4032, out_channels_right=672)
        self.relu = nn.ReLU()
        self.avg_pool = nn.AvgPool2d(11, stride=1, padding=0)
        self.dropout = nn.Dropout()
        self.last_linear = nn.Linear(4032, self.num_classes)

    def features(self, input):
        x_conv0 = self.conv0(input)
        x_stem_0 = self.cell_stem_0(x_conv0)
        x_stem_1 = self.cell_stem_1(x_conv0, x_stem_0)
        x_cell_0 = self.cell_0(x_stem_1, x_stem_0)
        x_cell_1 = self.cell_1(x_cell_0, x_stem_1)
        x_cell_2 = self.cell_2(x_cell_1, x_cell_0)
        x_cell_3 = self.cell_3(x_cell_2, x_cell_1)
        x_cell_4 = self.cell_4(x_cell_3, x_cell_2)
        x_cell_5 = self.cell_5(x_cell_4, x_cell_3)
        x_reduction_cell_0 = self.reduction_cell_0(x_cell_5, x_cell_4)
        x_cell_6 = self.cell_6(x_reduction_cell_0, x_cell_4)
        x_cell_7 = self.cell_7(x_cell_6, x_reduction_cell_0)
        x_cell_8 = self.cell_8(x_cell_7, x_cell_6)
        x_cell_9 = self.cell_9(x_cell_8, x_cell_7)
        x_cell_10 = self.cell_10(x_cell_9, x_cell_8)
        x_cell_11 = self.cell_11(x_cell_10, x_cell_9)
        x_reduction_cell_1 = self.reduction_cell_1(x_cell_11, x_cell_10)
        x_cell_12 = self.cell_12(x_reduction_cell_1, x_cell_10)
        x_cell_13 = self.cell_13(x_cell_12, x_reduction_cell_1)
        x_cell_14 = self.cell_14(x_cell_13, x_cell_12)
        x_cell_15 = self.cell_15(x_cell_14, x_cell_13)
        x_cell_16 = self.cell_16(x_cell_15, x_cell_14)
        x_cell_17 = self.cell_17(x_cell_16, x_cell_15)
        return x_cell_17

    def logits(self, features):
        x = self.relu(features)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x


class MaxPoolPad(nn.Module):

    def __init__(self):
        super(MaxPoolPad, self).__init__()
        self.pad = nn.ZeroPad2d((1, 0, 1, 0))
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        x = self.pad(x)
        x = self.pool(x)
        x = x[:, :, 1:, 1:].contiguous()
        return x


class AvgPoolPad(nn.Module):

    def __init__(self, stride=2, padding=1):
        super(AvgPoolPad, self).__init__()
        self.pad = nn.ZeroPad2d((1, 0, 1, 0))
        self.pool = nn.AvgPool2d(3, stride=stride, padding=padding,
            count_include_pad=False)

    def forward(self, x):
        x = self.pad(x)
        x = self.pool(x)
        x = x[:, :, 1:, 1:].contiguous()
        return x


class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, dw_kernel, dw_stride,
        dw_padding, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise_conv2d = nn.Conv2d(in_channels, in_channels,
            dw_kernel, stride=dw_stride, padding=dw_padding, bias=bias,
            groups=in_channels)
        self.pointwise_conv2d = nn.Conv2d(in_channels, out_channels, 1,
            stride=1, bias=bias)

    def forward(self, x):
        x = self.depthwise_conv2d(x)
        x = self.pointwise_conv2d(x)
        return x


class BranchSeparables(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
        padding, name=None, bias=False):
        super(BranchSeparables, self).__init__()
        self.relu = nn.ReLU()
        self.separable_1 = SeparableConv2d(in_channels, in_channels,
            kernel_size, stride, padding, bias=bias)
        self.bn_sep_1 = nn.BatchNorm2d(in_channels, eps=0.001, momentum=0.1,
            affine=True)
        self.relu1 = nn.ReLU()
        self.separable_2 = SeparableConv2d(in_channels, out_channels,
            kernel_size, 1, padding, bias=bias)
        self.bn_sep_2 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=
            0.1, affine=True)
        self.name = name

    def forward(self, x):
        x = self.relu(x)
        if self.name == 'specific':
            x = nn.ZeroPad2d((1, 0, 1, 0))(x)
        x = self.separable_1(x)
        if self.name == 'specific':
            x = x[:, :, 1:, 1:].contiguous()
        x = self.bn_sep_1(x)
        x = self.relu1(x)
        x = self.separable_2(x)
        x = self.bn_sep_2(x)
        return x


class BranchSeparablesStem(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
        padding, bias=False):
        super(BranchSeparablesStem, self).__init__()
        self.relu = nn.ReLU()
        self.separable_1 = SeparableConv2d(in_channels, out_channels,
            kernel_size, stride, padding, bias=bias)
        self.bn_sep_1 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=
            0.1, affine=True)
        self.relu1 = nn.ReLU()
        self.separable_2 = SeparableConv2d(out_channels, out_channels,
            kernel_size, 1, padding, bias=bias)
        self.bn_sep_2 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=
            0.1, affine=True)

    def forward(self, x):
        x = self.relu(x)
        x = self.separable_1(x)
        x = self.bn_sep_1(x)
        x = self.relu1(x)
        x = self.separable_2(x)
        x = self.bn_sep_2(x)
        return x


class CellStem0(nn.Module):

    def __init__(self, stem_filters, num_filters=42):
        super(CellStem0, self).__init__()
        self.num_filters = num_filters
        self.stem_filters = stem_filters
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(self.stem_filters, self.
            num_filters, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(self.num_filters, eps
            =0.001, momentum=0.1, affine=True))
        self.comb_iter_0_left = BranchSeparables(self.num_filters, self.
            num_filters, 5, 2, 2)
        self.comb_iter_0_right = BranchSeparablesStem(self.stem_filters,
            self.num_filters, 7, 2, 3, bias=False)
        self.comb_iter_1_left = nn.MaxPool2d(3, stride=2, padding=1)
        self.comb_iter_1_right = BranchSeparablesStem(self.stem_filters,
            self.num_filters, 7, 2, 3, bias=False)
        self.comb_iter_2_left = nn.AvgPool2d(3, stride=2, padding=1,
            count_include_pad=False)
        self.comb_iter_2_right = BranchSeparablesStem(self.stem_filters,
            self.num_filters, 5, 2, 2, bias=False)
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False)
        self.comb_iter_4_left = BranchSeparables(self.num_filters, self.
            num_filters, 3, 1, 1, bias=False)
        self.comb_iter_4_right = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        x1 = self.conv_1x1(x)
        x_comb_iter_0_left = self.comb_iter_0_left(x1)
        x_comb_iter_0_right = self.comb_iter_0_right(x)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
        x_comb_iter_1_left = self.comb_iter_1_left(x1)
        x_comb_iter_1_right = self.comb_iter_1_right(x)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
        x_comb_iter_2_left = self.comb_iter_2_left(x1)
        x_comb_iter_2_right = self.comb_iter_2_right(x)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right
        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1
        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x1)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right
        x_out = torch.cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3,
            x_comb_iter_4], 1)
        return x_out


class CellStem1(nn.Module):

    def __init__(self, stem_filters, num_filters):
        super(CellStem1, self).__init__()
        self.num_filters = num_filters
        self.stem_filters = stem_filters
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(2 * self.num_filters,
            self.num_filters, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(self.num_filters, eps
            =0.001, momentum=0.1, affine=True))
        self.relu = nn.ReLU()
        self.path_1 = nn.Sequential()
        self.path_1.add_module('avgpool', nn.AvgPool2d(1, stride=2,
            count_include_pad=False))
        self.path_1.add_module('conv', nn.Conv2d(self.stem_filters, self.
            num_filters // 2, 1, stride=1, bias=False))
        self.path_2 = nn.ModuleList()
        self.path_2.add_module('pad', nn.ZeroPad2d((0, 1, 0, 1)))
        self.path_2.add_module('avgpool', nn.AvgPool2d(1, stride=2,
            count_include_pad=False))
        self.path_2.add_module('conv', nn.Conv2d(self.stem_filters, self.
            num_filters // 2, 1, stride=1, bias=False))
        self.final_path_bn = nn.BatchNorm2d(self.num_filters, eps=0.001,
            momentum=0.1, affine=True)
        self.comb_iter_0_left = BranchSeparables(self.num_filters, self.
            num_filters, 5, 2, 2, name='specific', bias=False)
        self.comb_iter_0_right = BranchSeparables(self.num_filters, self.
            num_filters, 7, 2, 3, name='specific', bias=False)
        self.comb_iter_1_left = MaxPoolPad()
        self.comb_iter_1_right = BranchSeparables(self.num_filters, self.
            num_filters, 7, 2, 3, name='specific', bias=False)
        self.comb_iter_2_left = AvgPoolPad()
        self.comb_iter_2_right = BranchSeparables(self.num_filters, self.
            num_filters, 5, 2, 2, name='specific', bias=False)
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False)
        self.comb_iter_4_left = BranchSeparables(self.num_filters, self.
            num_filters, 3, 1, 1, name='specific', bias=False)
        self.comb_iter_4_right = MaxPoolPad()

    def forward(self, x_conv0, x_stem_0):
        x_left = self.conv_1x1(x_stem_0)
        x_relu = self.relu(x_conv0)
        x_path1 = self.path_1(x_relu)
        x_path2 = self.path_2.pad(x_relu)
        x_path2 = x_path2[:, :, 1:, 1:]
        x_path2 = self.path_2.avgpool(x_path2)
        x_path2 = self.path_2.conv(x_path2)
        x_right = self.final_path_bn(torch.cat([x_path1, x_path2], 1))
        x_comb_iter_0_left = self.comb_iter_0_left(x_left)
        x_comb_iter_0_right = self.comb_iter_0_right(x_right)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
        x_comb_iter_1_left = self.comb_iter_1_left(x_left)
        x_comb_iter_1_right = self.comb_iter_1_right(x_right)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
        x_comb_iter_2_left = self.comb_iter_2_left(x_left)
        x_comb_iter_2_right = self.comb_iter_2_right(x_right)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right
        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1
        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x_left)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right
        x_out = torch.cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3,
            x_comb_iter_4], 1)
        return x_out


class FirstCell(nn.Module):

    def __init__(self, in_channels_left, out_channels_left,
        in_channels_right, out_channels_right):
        super(FirstCell, self).__init__()
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(in_channels_right,
            out_channels_right, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(out_channels_right,
            eps=0.001, momentum=0.1, affine=True))
        self.relu = nn.ReLU()
        self.path_1 = nn.Sequential()
        self.path_1.add_module('avgpool', nn.AvgPool2d(1, stride=2,
            count_include_pad=False))
        self.path_1.add_module('conv', nn.Conv2d(in_channels_left,
            out_channels_left, 1, stride=1, bias=False))
        self.path_2 = nn.ModuleList()
        self.path_2.add_module('pad', nn.ZeroPad2d((0, 1, 0, 1)))
        self.path_2.add_module('avgpool', nn.AvgPool2d(1, stride=2,
            count_include_pad=False))
        self.path_2.add_module('conv', nn.Conv2d(in_channels_left,
            out_channels_left, 1, stride=1, bias=False))
        self.final_path_bn = nn.BatchNorm2d(out_channels_left * 2, eps=
            0.001, momentum=0.1, affine=True)
        self.comb_iter_0_left = BranchSeparables(out_channels_right,
            out_channels_right, 5, 1, 2, bias=False)
        self.comb_iter_0_right = BranchSeparables(out_channels_right,
            out_channels_right, 3, 1, 1, bias=False)
        self.comb_iter_1_left = BranchSeparables(out_channels_right,
            out_channels_right, 5, 1, 2, bias=False)
        self.comb_iter_1_right = BranchSeparables(out_channels_right,
            out_channels_right, 3, 1, 1, bias=False)
        self.comb_iter_2_left = nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False)
        self.comb_iter_3_left = nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False)
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False)
        self.comb_iter_4_left = BranchSeparables(out_channels_right,
            out_channels_right, 3, 1, 1, bias=False)

    def forward(self, x, x_prev):
        x_relu = self.relu(x_prev)
        x_path1 = self.path_1(x_relu)
        x_path2 = self.path_2.pad(x_relu)
        x_path2 = x_path2[:, :, 1:, 1:]
        x_path2 = self.path_2.avgpool(x_path2)
        x_path2 = self.path_2.conv(x_path2)
        x_left = self.final_path_bn(torch.cat([x_path1, x_path2], 1))
        x_right = self.conv_1x1(x)
        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
        x_comb_iter_1_left = self.comb_iter_1_left(x_left)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2 = x_comb_iter_2_left + x_left
        x_comb_iter_3_left = self.comb_iter_3_left(x_left)
        x_comb_iter_3_right = self.comb_iter_3_right(x_left)
        x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right
        x_comb_iter_4_left = self.comb_iter_4_left(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_right
        x_out = torch.cat([x_left, x_comb_iter_0, x_comb_iter_1,
            x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class NormalCell(nn.Module):

    def __init__(self, in_channels_left, out_channels_left,
        in_channels_right, out_channels_right):
        super(NormalCell, self).__init__()
        self.conv_prev_1x1 = nn.Sequential()
        self.conv_prev_1x1.add_module('relu', nn.ReLU())
        self.conv_prev_1x1.add_module('conv', nn.Conv2d(in_channels_left,
            out_channels_left, 1, stride=1, bias=False))
        self.conv_prev_1x1.add_module('bn', nn.BatchNorm2d(
            out_channels_left, eps=0.001, momentum=0.1, affine=True))
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(in_channels_right,
            out_channels_right, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(out_channels_right,
            eps=0.001, momentum=0.1, affine=True))
        self.comb_iter_0_left = BranchSeparables(out_channels_right,
            out_channels_right, 5, 1, 2, bias=False)
        self.comb_iter_0_right = BranchSeparables(out_channels_left,
            out_channels_left, 3, 1, 1, bias=False)
        self.comb_iter_1_left = BranchSeparables(out_channels_left,
            out_channels_left, 5, 1, 2, bias=False)
        self.comb_iter_1_right = BranchSeparables(out_channels_left,
            out_channels_left, 3, 1, 1, bias=False)
        self.comb_iter_2_left = nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False)
        self.comb_iter_3_left = nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False)
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False)
        self.comb_iter_4_left = BranchSeparables(out_channels_right,
            out_channels_right, 3, 1, 1, bias=False)

    def forward(self, x, x_prev):
        x_left = self.conv_prev_1x1(x_prev)
        x_right = self.conv_1x1(x)
        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
        x_comb_iter_1_left = self.comb_iter_1_left(x_left)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2 = x_comb_iter_2_left + x_left
        x_comb_iter_3_left = self.comb_iter_3_left(x_left)
        x_comb_iter_3_right = self.comb_iter_3_right(x_left)
        x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right
        x_comb_iter_4_left = self.comb_iter_4_left(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_right
        x_out = torch.cat([x_left, x_comb_iter_0, x_comb_iter_1,
            x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class ReductionCell0(nn.Module):

    def __init__(self, in_channels_left, out_channels_left,
        in_channels_right, out_channels_right):
        super(ReductionCell0, self).__init__()
        self.conv_prev_1x1 = nn.Sequential()
        self.conv_prev_1x1.add_module('relu', nn.ReLU())
        self.conv_prev_1x1.add_module('conv', nn.Conv2d(in_channels_left,
            out_channels_left, 1, stride=1, bias=False))
        self.conv_prev_1x1.add_module('bn', nn.BatchNorm2d(
            out_channels_left, eps=0.001, momentum=0.1, affine=True))
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(in_channels_right,
            out_channels_right, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(out_channels_right,
            eps=0.001, momentum=0.1, affine=True))
        self.comb_iter_0_left = BranchSeparablesReduction(out_channels_right,
            out_channels_right, 5, 2, 2, bias=False)
        self.comb_iter_0_right = BranchSeparablesReduction(out_channels_right,
            out_channels_right, 7, 2, 3, bias=False)
        self.comb_iter_1_left = MaxPoolPad()
        self.comb_iter_1_right = BranchSeparablesReduction(out_channels_right,
            out_channels_right, 7, 2, 3, bias=False)
        self.comb_iter_2_left = AvgPoolPad()
        self.comb_iter_2_right = BranchSeparablesReduction(out_channels_right,
            out_channels_right, 5, 2, 2, bias=False)
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False)
        self.comb_iter_4_left = BranchSeparablesReduction(out_channels_right,
            out_channels_right, 3, 1, 1, bias=False)
        self.comb_iter_4_right = MaxPoolPad()

    def forward(self, x, x_prev):
        x_left = self.conv_prev_1x1(x_prev)
        x_right = self.conv_1x1(x)
        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
        x_comb_iter_1_left = self.comb_iter_1_left(x_right)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2_right = self.comb_iter_2_right(x_left)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right
        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1
        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right
        x_out = torch.cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3,
            x_comb_iter_4], 1)
        return x_out


class ReductionCell1(nn.Module):

    def __init__(self, in_channels_left, out_channels_left,
        in_channels_right, out_channels_right):
        super(ReductionCell1, self).__init__()
        self.conv_prev_1x1 = nn.Sequential()
        self.conv_prev_1x1.add_module('relu', nn.ReLU())
        self.conv_prev_1x1.add_module('conv', nn.Conv2d(in_channels_left,
            out_channels_left, 1, stride=1, bias=False))
        self.conv_prev_1x1.add_module('bn', nn.BatchNorm2d(
            out_channels_left, eps=0.001, momentum=0.1, affine=True))
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(in_channels_right,
            out_channels_right, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(out_channels_right,
            eps=0.001, momentum=0.1, affine=True))
        self.comb_iter_0_left = BranchSeparables(out_channels_right,
            out_channels_right, 5, 2, 2, name='specific', bias=False)
        self.comb_iter_0_right = BranchSeparables(out_channels_right,
            out_channels_right, 7, 2, 3, name='specific', bias=False)
        self.comb_iter_1_left = MaxPoolPad()
        self.comb_iter_1_right = BranchSeparables(out_channels_right,
            out_channels_right, 7, 2, 3, name='specific', bias=False)
        self.comb_iter_2_left = AvgPoolPad()
        self.comb_iter_2_right = BranchSeparables(out_channels_right,
            out_channels_right, 5, 2, 2, name='specific', bias=False)
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False)
        self.comb_iter_4_left = BranchSeparables(out_channels_right,
            out_channels_right, 3, 1, 1, name='specific', bias=False)
        self.comb_iter_4_right = MaxPoolPad()

    def forward(self, x, x_prev):
        x_left = self.conv_prev_1x1(x_prev)
        x_right = self.conv_1x1(x)
        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
        x_comb_iter_1_left = self.comb_iter_1_left(x_right)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2_right = self.comb_iter_2_right(x_left)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right
        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1
        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right
        x_out = torch.cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3,
            x_comb_iter_4], 1)
        return x_out


class NASNetAMobile(nn.Module):
    """NASNetAMobile (4 @ 1056) """

    def __init__(self, num_classes=1001, stem_filters=32,
        penultimate_filters=1056, filters_multiplier=2):
        super(NASNetAMobile, self).__init__()
        self.num_classes = num_classes
        self.stem_filters = stem_filters
        self.penultimate_filters = penultimate_filters
        self.filters_multiplier = filters_multiplier
        filters = self.penultimate_filters // 24
        self.conv0 = nn.Sequential()
        self.conv0.add_module('conv', nn.Conv2d(in_channels=3, out_channels
            =self.stem_filters, kernel_size=3, padding=0, stride=2, bias=False)
            )
        self.conv0.add_module('bn', nn.BatchNorm2d(self.stem_filters, eps=
            0.001, momentum=0.1, affine=True))
        self.cell_stem_0 = CellStem0(self.stem_filters, num_filters=filters //
            filters_multiplier ** 2)
        self.cell_stem_1 = CellStem1(self.stem_filters, num_filters=filters //
            filters_multiplier)
        self.cell_0 = FirstCell(in_channels_left=filters, out_channels_left
            =filters // 2, in_channels_right=2 * filters,
            out_channels_right=filters)
        self.cell_1 = NormalCell(in_channels_left=2 * filters,
            out_channels_left=filters, in_channels_right=6 * filters,
            out_channels_right=filters)
        self.cell_2 = NormalCell(in_channels_left=6 * filters,
            out_channels_left=filters, in_channels_right=6 * filters,
            out_channels_right=filters)
        self.cell_3 = NormalCell(in_channels_left=6 * filters,
            out_channels_left=filters, in_channels_right=6 * filters,
            out_channels_right=filters)
        self.reduction_cell_0 = ReductionCell0(in_channels_left=6 * filters,
            out_channels_left=2 * filters, in_channels_right=6 * filters,
            out_channels_right=2 * filters)
        self.cell_6 = FirstCell(in_channels_left=6 * filters,
            out_channels_left=filters, in_channels_right=8 * filters,
            out_channels_right=2 * filters)
        self.cell_7 = NormalCell(in_channels_left=8 * filters,
            out_channels_left=2 * filters, in_channels_right=12 * filters,
            out_channels_right=2 * filters)
        self.cell_8 = NormalCell(in_channels_left=12 * filters,
            out_channels_left=2 * filters, in_channels_right=12 * filters,
            out_channels_right=2 * filters)
        self.cell_9 = NormalCell(in_channels_left=12 * filters,
            out_channels_left=2 * filters, in_channels_right=12 * filters,
            out_channels_right=2 * filters)
        self.reduction_cell_1 = ReductionCell1(in_channels_left=12 *
            filters, out_channels_left=4 * filters, in_channels_right=12 *
            filters, out_channels_right=4 * filters)
        self.cell_12 = FirstCell(in_channels_left=12 * filters,
            out_channels_left=2 * filters, in_channels_right=16 * filters,
            out_channels_right=4 * filters)
        self.cell_13 = NormalCell(in_channels_left=16 * filters,
            out_channels_left=4 * filters, in_channels_right=24 * filters,
            out_channels_right=4 * filters)
        self.cell_14 = NormalCell(in_channels_left=24 * filters,
            out_channels_left=4 * filters, in_channels_right=24 * filters,
            out_channels_right=4 * filters)
        self.cell_15 = NormalCell(in_channels_left=24 * filters,
            out_channels_left=4 * filters, in_channels_right=24 * filters,
            out_channels_right=4 * filters)
        self.relu = nn.ReLU()
        self.avg_pool = nn.AvgPool2d(7, stride=1, padding=0)
        self.dropout = nn.Dropout()
        self.last_linear = nn.Linear(24 * filters, self.num_classes)

    def features(self, input):
        x_conv0 = self.conv0(input)
        x_stem_0 = self.cell_stem_0(x_conv0)
        x_stem_1 = self.cell_stem_1(x_conv0, x_stem_0)
        x_cell_0 = self.cell_0(x_stem_1, x_stem_0)
        x_cell_1 = self.cell_1(x_cell_0, x_stem_1)
        x_cell_2 = self.cell_2(x_cell_1, x_cell_0)
        x_cell_3 = self.cell_3(x_cell_2, x_cell_1)
        x_reduction_cell_0 = self.reduction_cell_0(x_cell_3, x_cell_2)
        x_cell_6 = self.cell_6(x_reduction_cell_0, x_cell_3)
        x_cell_7 = self.cell_7(x_cell_6, x_reduction_cell_0)
        x_cell_8 = self.cell_8(x_cell_7, x_cell_6)
        x_cell_9 = self.cell_9(x_cell_8, x_cell_7)
        x_reduction_cell_1 = self.reduction_cell_1(x_cell_9, x_cell_8)
        x_cell_12 = self.cell_12(x_reduction_cell_1, x_cell_9)
        x_cell_13 = self.cell_13(x_cell_12, x_reduction_cell_1)
        x_cell_14 = self.cell_14(x_cell_13, x_cell_12)
        x_cell_15 = self.cell_15(x_cell_14, x_cell_13)
        return x_cell_15

    def logits(self, features):
        x = self.relu(features)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x


class MaxPool(nn.Module):

    def __init__(self, kernel_size, stride=1, padding=1, zero_pad=False):
        super(MaxPool, self).__init__()
        self.zero_pad = nn.ZeroPad2d((1, 0, 1, 0)) if zero_pad else None
        self.pool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        if self.zero_pad:
            x = self.zero_pad(x)
        x = self.pool(x)
        if self.zero_pad:
            x = x[:, :, 1:, 1:]
        return x


class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, dw_kernel_size, dw_stride,
        dw_padding):
        super(SeparableConv2d, self).__init__()
        self.depthwise_conv2d = nn.Conv2d(in_channels, in_channels,
            kernel_size=dw_kernel_size, stride=dw_stride, padding=
            dw_padding, groups=in_channels, bias=False)
        self.pointwise_conv2d = nn.Conv2d(in_channels, out_channels,
            kernel_size=1, bias=False)

    def forward(self, x):
        x = self.depthwise_conv2d(x)
        x = self.pointwise_conv2d(x)
        return x


class BranchSeparables(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        stem_cell=False, zero_pad=False):
        super(BranchSeparables, self).__init__()
        padding = kernel_size // 2
        middle_channels = out_channels if stem_cell else in_channels
        self.zero_pad = nn.ZeroPad2d((1, 0, 1, 0)) if zero_pad else None
        self.relu_1 = nn.ReLU()
        self.separable_1 = SeparableConv2d(in_channels, middle_channels,
            kernel_size, dw_stride=stride, dw_padding=padding)
        self.bn_sep_1 = nn.BatchNorm2d(middle_channels, eps=0.001)
        self.relu_2 = nn.ReLU()
        self.separable_2 = SeparableConv2d(middle_channels, out_channels,
            kernel_size, dw_stride=1, dw_padding=padding)
        self.bn_sep_2 = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.relu_1(x)
        if self.zero_pad:
            x = self.zero_pad(x)
        x = self.separable_1(x)
        if self.zero_pad:
            x = x[:, :, 1:, 1:].contiguous()
        x = self.bn_sep_1(x)
        x = self.relu_2(x)
        x = self.separable_2(x)
        x = self.bn_sep_2(x)
        return x


class ReluConvBn(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(ReluConvBn, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=
            kernel_size, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.relu(x)
        x = self.conv(x)
        x = self.bn(x)
        return x


class FactorizedReduction(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(FactorizedReduction, self).__init__()
        self.relu = nn.ReLU()
        self.path_1 = nn.Sequential(OrderedDict([('avgpool', nn.AvgPool2d(1,
            stride=2, count_include_pad=False)), ('conv', nn.Conv2d(
            in_channels, out_channels // 2, kernel_size=1, bias=False))]))
        self.path_2 = nn.Sequential(OrderedDict([('pad', nn.ZeroPad2d((0, 1,
            0, 1))), ('avgpool', nn.AvgPool2d(1, stride=2,
            count_include_pad=False)), ('conv', nn.Conv2d(in_channels, 
            out_channels // 2, kernel_size=1, bias=False))]))
        self.final_path_bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.relu(x)
        x_path1 = self.path_1(x)
        x_path2 = self.path_2.pad(x)
        x_path2 = x_path2[:, :, 1:, 1:]
        x_path2 = self.path_2.avgpool(x_path2)
        x_path2 = self.path_2.conv(x_path2)
        out = self.final_path_bn(torch.cat([x_path1, x_path2], 1))
        return out


class CellBase(nn.Module):

    def cell_forward(self, x_left, x_right):
        x_comb_iter_0_left = self.comb_iter_0_left(x_left)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
        x_comb_iter_1_left = self.comb_iter_1_left(x_right)
        x_comb_iter_1_right = self.comb_iter_1_right(x_right)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2_right = self.comb_iter_2_right(x_right)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right
        x_comb_iter_3_left = self.comb_iter_3_left(x_comb_iter_2)
        x_comb_iter_3_right = self.comb_iter_3_right(x_right)
        x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right
        x_comb_iter_4_left = self.comb_iter_4_left(x_left)
        if self.comb_iter_4_right:
            x_comb_iter_4_right = self.comb_iter_4_right(x_right)
        else:
            x_comb_iter_4_right = x_right
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right
        x_out = torch.cat([x_comb_iter_0, x_comb_iter_1, x_comb_iter_2,
            x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class Cell(CellBase):

    def __init__(self, in_channels_left, out_channels_left,
        in_channels_right, out_channels_right, is_reduction=False, zero_pad
        =False, match_prev_layer_dimensions=False):
        super(Cell, self).__init__()
        stride = 2 if is_reduction else 1
        self.match_prev_layer_dimensions = match_prev_layer_dimensions
        if match_prev_layer_dimensions:
            self.conv_prev_1x1 = FactorizedReduction(in_channels_left,
                out_channels_left)
        else:
            self.conv_prev_1x1 = ReluConvBn(in_channels_left,
                out_channels_left, kernel_size=1)
        self.conv_1x1 = ReluConvBn(in_channels_right, out_channels_right,
            kernel_size=1)
        self.comb_iter_0_left = BranchSeparables(out_channels_left,
            out_channels_left, kernel_size=5, stride=stride, zero_pad=zero_pad)
        self.comb_iter_0_right = MaxPool(3, stride=stride, zero_pad=zero_pad)
        self.comb_iter_1_left = BranchSeparables(out_channels_right,
            out_channels_right, kernel_size=7, stride=stride, zero_pad=zero_pad
            )
        self.comb_iter_1_right = MaxPool(3, stride=stride, zero_pad=zero_pad)
        self.comb_iter_2_left = BranchSeparables(out_channels_right,
            out_channels_right, kernel_size=5, stride=stride, zero_pad=zero_pad
            )
        self.comb_iter_2_right = BranchSeparables(out_channels_right,
            out_channels_right, kernel_size=3, stride=stride, zero_pad=zero_pad
            )
        self.comb_iter_3_left = BranchSeparables(out_channels_right,
            out_channels_right, kernel_size=3)
        self.comb_iter_3_right = MaxPool(3, stride=stride, zero_pad=zero_pad)
        self.comb_iter_4_left = BranchSeparables(out_channels_left,
            out_channels_left, kernel_size=3, stride=stride, zero_pad=zero_pad)
        if is_reduction:
            self.comb_iter_4_right = ReluConvBn(out_channels_right,
                out_channels_right, kernel_size=1, stride=stride)
        else:
            self.comb_iter_4_right = None

    def forward(self, x_left, x_right):
        x_left = self.conv_prev_1x1(x_left)
        x_right = self.conv_1x1(x_right)
        x_out = self.cell_forward(x_left, x_right)
        return x_out


class PNASNet5Large(nn.Module):

    def __init__(self, num_classes=1001):
        super().__init__()
        self.num_classes = num_classes
        self.conv_0 = nn.Sequential(OrderedDict([('conv', nn.Conv2d(3, 96,
            kernel_size=3, stride=2, bias=False)), ('bn', nn.BatchNorm2d(96,
            eps=0.001))]))
        self.cell_stem_0 = CellStem0(in_channels_left=96, out_channels_left
            =54, in_channels_right=96, out_channels_right=54)
        self.cell_stem_1 = Cell(in_channels_left=96, out_channels_left=108,
            in_channels_right=270, out_channels_right=108,
            match_prev_layer_dimensions=True, is_reduction=True)
        self.cell_0 = Cell(in_channels_left=270, out_channels_left=216,
            in_channels_right=540, out_channels_right=216,
            match_prev_layer_dimensions=True)
        self.cell_1 = Cell(in_channels_left=540, out_channels_left=216,
            in_channels_right=1080, out_channels_right=216)
        self.cell_2 = Cell(in_channels_left=1080, out_channels_left=216,
            in_channels_right=1080, out_channels_right=216)
        self.cell_3 = Cell(in_channels_left=1080, out_channels_left=216,
            in_channels_right=1080, out_channels_right=216)
        self.cell_4 = Cell(in_channels_left=1080, out_channels_left=432,
            in_channels_right=1080, out_channels_right=432, is_reduction=
            True, zero_pad=True)
        self.cell_5 = Cell(in_channels_left=1080, out_channels_left=432,
            in_channels_right=2160, out_channels_right=432,
            match_prev_layer_dimensions=True)
        self.cell_6 = Cell(in_channels_left=2160, out_channels_left=432,
            in_channels_right=2160, out_channels_right=432)
        self.cell_7 = Cell(in_channels_left=2160, out_channels_left=432,
            in_channels_right=2160, out_channels_right=432)
        self.cell_8 = Cell(in_channels_left=2160, out_channels_left=864,
            in_channels_right=2160, out_channels_right=864, is_reduction=True)
        self.cell_9 = Cell(in_channels_left=2160, out_channels_left=864,
            in_channels_right=4320, out_channels_right=864,
            match_prev_layer_dimensions=True)
        self.cell_10 = Cell(in_channels_left=4320, out_channels_left=864,
            in_channels_right=4320, out_channels_right=864)
        self.cell_11 = Cell(in_channels_left=4320, out_channels_left=864,
            in_channels_right=4320, out_channels_right=864)
        self.relu = nn.ReLU()
        self.avg_pool = nn.AvgPool2d(11, stride=1, padding=0)
        self.dropout = nn.Dropout(0.5)
        self.last_linear = nn.Linear(4320, num_classes)

    def features(self, x):
        x_conv_0 = self.conv_0(x)
        x_stem_0 = self.cell_stem_0(x_conv_0)
        x_stem_1 = self.cell_stem_1(x_conv_0, x_stem_0)
        x_cell_0 = self.cell_0(x_stem_0, x_stem_1)
        x_cell_1 = self.cell_1(x_stem_1, x_cell_0)
        x_cell_2 = self.cell_2(x_cell_0, x_cell_1)
        x_cell_3 = self.cell_3(x_cell_1, x_cell_2)
        x_cell_4 = self.cell_4(x_cell_2, x_cell_3)
        x_cell_5 = self.cell_5(x_cell_3, x_cell_4)
        x_cell_6 = self.cell_6(x_cell_4, x_cell_5)
        x_cell_7 = self.cell_7(x_cell_5, x_cell_6)
        x_cell_8 = self.cell_8(x_cell_6, x_cell_7)
        x_cell_9 = self.cell_9(x_cell_7, x_cell_8)
        x_cell_10 = self.cell_10(x_cell_8, x_cell_9)
        x_cell_11 = self.cell_11(x_cell_9, x_cell_10)
        return x_cell_11

    def logits(self, features):
        x = self.relu(features)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x


class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1,
        padding=0, output_relu=True):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=
            kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU() if output_relu else None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu:
            x = self.relu(x)
        return x


class PolyConv2d(nn.Module):
    """A block that is used inside poly-N (poly-2, poly-3, and so on) modules.
    The Convolution layer is shared between all Inception blocks inside
    a poly-N module. BatchNorm layers are not shared between Inception blocks
    and therefore the number of BatchNorm layers is equal to the number of
    Inception blocks inside a poly-N module.
    """

    def __init__(self, in_planes, out_planes, kernel_size, num_blocks,
        stride=1, padding=0):
        super(PolyConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=
            kernel_size, stride=stride, padding=padding, bias=False)
        self.bn_blocks = nn.ModuleList([nn.BatchNorm2d(out_planes) for _ in
            range(num_blocks)])
        self.relu = nn.ReLU()

    def forward(self, x, block_index):
        x = self.conv(x)
        bn = self.bn_blocks[block_index]
        x = bn(x)
        x = self.relu(x)
        return x


class Stem(nn.Module):

    def __init__(self):
        super(Stem, self).__init__()
        self.conv1 = nn.Sequential(BasicConv2d(3, 32, kernel_size=3, stride
            =2), BasicConv2d(32, 32, kernel_size=3), BasicConv2d(32, 64,
            kernel_size=3, padding=1))
        self.conv1_pool_branch = nn.MaxPool2d(3, stride=2)
        self.conv1_branch = BasicConv2d(64, 96, kernel_size=3, stride=2)
        self.conv2_short = nn.Sequential(BasicConv2d(160, 64, kernel_size=1
            ), BasicConv2d(64, 96, kernel_size=3))
        self.conv2_long = nn.Sequential(BasicConv2d(160, 64, kernel_size=1),
            BasicConv2d(64, 64, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(64, 64, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(64, 96, kernel_size=3))
        self.conv2_pool_branch = nn.MaxPool2d(3, stride=2)
        self.conv2_branch = BasicConv2d(192, 192, kernel_size=3, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x0 = self.conv1_pool_branch(x)
        x1 = self.conv1_branch(x)
        x = torch.cat((x0, x1), 1)
        x0 = self.conv2_short(x)
        x1 = self.conv2_long(x)
        x = torch.cat((x0, x1), 1)
        x0 = self.conv2_pool_branch(x)
        x1 = self.conv2_branch(x)
        out = torch.cat((x0, x1), 1)
        return out


class BlockA(nn.Module):
    """Inception-ResNet-A block."""

    def __init__(self):
        super(BlockA, self).__init__()
        self.path0 = nn.Sequential(BasicConv2d(384, 32, kernel_size=1),
            BasicConv2d(32, 48, kernel_size=3, padding=1), BasicConv2d(48, 
            64, kernel_size=3, padding=1))
        self.path1 = nn.Sequential(BasicConv2d(384, 32, kernel_size=1),
            BasicConv2d(32, 32, kernel_size=3, padding=1))
        self.path2 = BasicConv2d(384, 32, kernel_size=1)
        self.conv2d = BasicConv2d(128, 384, kernel_size=1, output_relu=False)

    def forward(self, x):
        x0 = self.path0(x)
        x1 = self.path1(x)
        x2 = self.path2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        return out


class BlockB(nn.Module):
    """Inception-ResNet-B block."""

    def __init__(self):
        super(BlockB, self).__init__()
        self.path0 = nn.Sequential(BasicConv2d(1152, 128, kernel_size=1),
            BasicConv2d(128, 160, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(160, 192, kernel_size=(7, 1), padding=(3, 0)))
        self.path1 = BasicConv2d(1152, 192, kernel_size=1)
        self.conv2d = BasicConv2d(384, 1152, kernel_size=1, output_relu=False)

    def forward(self, x):
        x0 = self.path0(x)
        x1 = self.path1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        return out


class BlockC(nn.Module):
    """Inception-ResNet-C block."""

    def __init__(self):
        super(BlockC, self).__init__()
        self.path0 = nn.Sequential(BasicConv2d(2048, 192, kernel_size=1),
            BasicConv2d(192, 224, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(224, 256, kernel_size=(3, 1), padding=(1, 0)))
        self.path1 = BasicConv2d(2048, 192, kernel_size=1)
        self.conv2d = BasicConv2d(448, 2048, kernel_size=1, output_relu=False)

    def forward(self, x):
        x0 = self.path0(x)
        x1 = self.path1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        return out


class ReductionA(nn.Module):
    """A dimensionality reduction block that is placed after stage-a
    Inception-ResNet blocks.
    """

    def __init__(self):
        super(ReductionA, self).__init__()
        self.path0 = nn.Sequential(BasicConv2d(384, 256, kernel_size=1),
            BasicConv2d(256, 256, kernel_size=3, padding=1), BasicConv2d(
            256, 384, kernel_size=3, stride=2))
        self.path1 = BasicConv2d(384, 384, kernel_size=3, stride=2)
        self.path2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.path0(x)
        x1 = self.path1(x)
        x2 = self.path2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class ReductionB(nn.Module):
    """A dimensionality reduction block that is placed after stage-b
    Inception-ResNet blocks.
    """

    def __init__(self):
        super(ReductionB, self).__init__()
        self.path0 = nn.Sequential(BasicConv2d(1152, 256, kernel_size=1),
            BasicConv2d(256, 256, kernel_size=3, padding=1), BasicConv2d(
            256, 256, kernel_size=3, stride=2))
        self.path1 = nn.Sequential(BasicConv2d(1152, 256, kernel_size=1),
            BasicConv2d(256, 256, kernel_size=3, stride=2))
        self.path2 = nn.Sequential(BasicConv2d(1152, 256, kernel_size=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2))
        self.path3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.path0(x)
        x1 = self.path1(x)
        x2 = self.path2(x)
        x3 = self.path3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class InceptionResNetBPoly(nn.Module):
    """Base class for constructing poly-N Inception-ResNet-B modules.
    When `num_blocks` is equal to 1, a module will have only a first-order path
    and will be equal to a standard Inception-ResNet-B block.
    When `num_blocks` is equal to 2, a module will have first-order and
    second-order paths and will be called Inception-ResNet-B poly-2 module.
    Increasing value of the `num_blocks` parameter will produce a higher order
    Inception-ResNet-B poly-N modules.
    """

    def __init__(self, scale, num_blocks):
        super(InceptionResNetBPoly, self).__init__()
        assert num_blocks >= 1, 'num_blocks should be greater or equal to 1'
        self.scale = scale
        self.num_blocks = num_blocks
        self.path0_1x1 = PolyConv2d(1152, 128, kernel_size=1, num_blocks=
            self.num_blocks)
        self.path0_1x7 = PolyConv2d(128, 160, kernel_size=(1, 7),
            num_blocks=self.num_blocks, padding=(0, 3))
        self.path0_7x1 = PolyConv2d(160, 192, kernel_size=(7, 1),
            num_blocks=self.num_blocks, padding=(3, 0))
        self.path1 = PolyConv2d(1152, 192, kernel_size=1, num_blocks=self.
            num_blocks)
        self.conv2d_blocks = nn.ModuleList([BasicConv2d(384, 1152,
            kernel_size=1, output_relu=False) for _ in range(self.num_blocks)])
        self.relu = nn.ReLU()

    def forward_block(self, x, block_index):
        x0 = self.path0_1x1(x, block_index)
        x0 = self.path0_1x7(x0, block_index)
        x0 = self.path0_7x1(x0, block_index)
        x1 = self.path1(x, block_index)
        out = torch.cat((x0, x1), 1)
        conv2d_block = self.conv2d_blocks[block_index]
        out = conv2d_block(out)
        return out

    def forward(self, x):
        out = x
        for block_index in range(self.num_blocks):
            x = self.forward_block(x, block_index)
            out = out + x * self.scale
            x = self.relu(x)
        out = self.relu(out)
        return out


class InceptionResNetCPoly(nn.Module):
    """Base class for constructing poly-N Inception-ResNet-C modules.
    When `num_blocks` is equal to 1, a module will have only a first-order path
    and will be equal to a standard Inception-ResNet-C block.
    When `num_blocks` is equal to 2, a module will have first-order and
    second-order paths and will be called Inception-ResNet-C poly-2 module.
    Increasing value of the `num_blocks` parameter will produce a higher order
    Inception-ResNet-C poly-N modules.
    """

    def __init__(self, scale, num_blocks):
        super(InceptionResNetCPoly, self).__init__()
        assert num_blocks >= 1, 'num_blocks should be greater or equal to 1'
        self.scale = scale
        self.num_blocks = num_blocks
        self.path0_1x1 = PolyConv2d(2048, 192, kernel_size=1, num_blocks=
            self.num_blocks)
        self.path0_1x3 = PolyConv2d(192, 224, kernel_size=(1, 3),
            num_blocks=self.num_blocks, padding=(0, 1))
        self.path0_3x1 = PolyConv2d(224, 256, kernel_size=(3, 1),
            num_blocks=self.num_blocks, padding=(1, 0))
        self.path1 = PolyConv2d(2048, 192, kernel_size=1, num_blocks=self.
            num_blocks)
        self.conv2d_blocks = nn.ModuleList([BasicConv2d(448, 2048,
            kernel_size=1, output_relu=False) for _ in range(self.num_blocks)])
        self.relu = nn.ReLU()

    def forward_block(self, x, block_index):
        x0 = self.path0_1x1(x, block_index)
        x0 = self.path0_1x3(x0, block_index)
        x0 = self.path0_3x1(x0, block_index)
        x1 = self.path1(x, block_index)
        out = torch.cat((x0, x1), 1)
        conv2d_block = self.conv2d_blocks[block_index]
        out = conv2d_block(out)
        return out

    def forward(self, x):
        out = x
        for block_index in range(self.num_blocks):
            x = self.forward_block(x, block_index)
            out = out + x * self.scale
            x = self.relu(x)
        out = self.relu(out)
        return out


class MultiWay(nn.Module):
    """Base class for constructing N-way modules (2-way, 3-way, and so on)."""

    def __init__(self, scale, block_cls, num_blocks):
        super(MultiWay, self).__init__()
        assert num_blocks >= 1, 'num_blocks should be greater or equal to 1'
        self.scale = scale
        self.blocks = nn.ModuleList([block_cls() for _ in range(num_blocks)])
        self.relu = nn.ReLU()

    def forward(self, x):
        out = x
        for block in self.blocks:
            out = out + block(x) * self.scale
        out = self.relu(out)
        return out


class InceptionResNetA2Way(MultiWay):

    def __init__(self, scale):
        super(InceptionResNetA2Way, self).__init__(scale, block_cls=BlockA,
            num_blocks=2)


class InceptionResNetB2Way(MultiWay):

    def __init__(self, scale):
        super(InceptionResNetB2Way, self).__init__(scale, block_cls=BlockB,
            num_blocks=2)


class InceptionResNetBPoly3(InceptionResNetBPoly):

    def __init__(self, scale):
        super(InceptionResNetBPoly3, self).__init__(scale, num_blocks=3)


class InceptionResNetC2Way(MultiWay):

    def __init__(self, scale):
        super(InceptionResNetC2Way, self).__init__(scale, block_cls=BlockC,
            num_blocks=2)


class InceptionResNetCPoly3(InceptionResNetCPoly):

    def __init__(self, scale):
        super(InceptionResNetCPoly3, self).__init__(scale, num_blocks=3)


class PolyNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(PolyNet, self).__init__()
        self.stem = Stem()
        self.stage_a = nn.Sequential(InceptionResNetA2Way(scale=1),
            InceptionResNetA2Way(scale=0.992308), InceptionResNetA2Way(
            scale=0.984615), InceptionResNetA2Way(scale=0.976923),
            InceptionResNetA2Way(scale=0.969231), InceptionResNetA2Way(
            scale=0.961538), InceptionResNetA2Way(scale=0.953846),
            InceptionResNetA2Way(scale=0.946154), InceptionResNetA2Way(
            scale=0.938462), InceptionResNetA2Way(scale=0.930769))
        self.reduction_a = ReductionA()
        self.stage_b = nn.Sequential(InceptionResNetBPoly3(scale=0.923077),
            InceptionResNetB2Way(scale=0.915385), InceptionResNetBPoly3(
            scale=0.907692), InceptionResNetB2Way(scale=0.9),
            InceptionResNetBPoly3(scale=0.892308), InceptionResNetB2Way(
            scale=0.884615), InceptionResNetBPoly3(scale=0.876923),
            InceptionResNetB2Way(scale=0.869231), InceptionResNetBPoly3(
            scale=0.861538), InceptionResNetB2Way(scale=0.853846),
            InceptionResNetBPoly3(scale=0.846154), InceptionResNetB2Way(
            scale=0.838462), InceptionResNetBPoly3(scale=0.830769),
            InceptionResNetB2Way(scale=0.823077), InceptionResNetBPoly3(
            scale=0.815385), InceptionResNetB2Way(scale=0.807692),
            InceptionResNetBPoly3(scale=0.8), InceptionResNetB2Way(scale=
            0.792308), InceptionResNetBPoly3(scale=0.784615),
            InceptionResNetB2Way(scale=0.776923))
        self.reduction_b = ReductionB()
        self.stage_c = nn.Sequential(InceptionResNetCPoly3(scale=0.769231),
            InceptionResNetC2Way(scale=0.761538), InceptionResNetCPoly3(
            scale=0.753846), InceptionResNetC2Way(scale=0.746154),
            InceptionResNetCPoly3(scale=0.738462), InceptionResNetC2Way(
            scale=0.730769), InceptionResNetCPoly3(scale=0.723077),
            InceptionResNetC2Way(scale=0.715385), InceptionResNetCPoly3(
            scale=0.707692), InceptionResNetC2Way(scale=0.7))
        self.avg_pool = nn.AvgPool2d(9, stride=1)
        self.dropout = nn.Dropout(0.2)
        self.last_linear = nn.Linear(2048, num_classes)

    def features(self, x):
        x = self.stem(x)
        x = self.stage_a(x)
        x = self.reduction_a(x)
        x = self.stage_b(x)
        x = self.reduction_b(x)
        x = self.stage_c(x)
        return x

    def logits(self, x):
        x = self.avg_pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=
            stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
            padding=1, bias=False)
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


def make_linear_bn_relu(in_channels, out_channels):
    return [nn.Linear(in_channels, out_channels, bias=False), nn.
        BatchNorm1d(out_channels), nn.ReLU(inplace=True)]


def make_max_flat(out):
    flat = F.adaptive_max_pool2d(out, output_size=1)
    flat = flat.view(flat.size(0), -1)
    return flat


class PyResNet(nn.Module):

    def __init__(self, block, layers, in_shape=(3, 256, 256), num_classes=17):
        self.inplanes = 64
        super(PyResNet, self).__init__()
        in_channels, height, width = in_shape
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2,
            padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 64, layers[0])
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self.make_layer(block, 512, layers[3], stride=2)
        self.fc2 = nn.Sequential(*make_linear_bn_relu(128 * block.expansion,
            512), nn.Linear(512, num_classes))
        self.fc3 = nn.Sequential(*make_linear_bn_relu(256 * block.expansion,
            512), nn.Linear(512, num_classes))
        self.fc4 = nn.Sequential(*make_linear_bn_relu(512 * block.expansion,
            512), nn.Linear(512, num_classes))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.layer1(x)
        x = self.layer2(x)
        flat2 = make_max_flat(x)
        x = self.layer3(x)
        flat3 = make_max_flat(x)
        x = self.layer4(x)
        flat4 = make_max_flat(x)
        x = self.fc2(flat2) + self.fc3(flat3) + self.fc4(flat4)
        logit = x
        prob = torch.sigmoid(logit)
        return logit, prob


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride, remove_first_relu,
        add_last_bn, preact=False):
        super(BasicBlock, self).__init__()
        self._remove_first_relu = remove_first_relu
        self._add_last_bn = add_last_bn
        self._preact = preact
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False)
        if add_last_bn:
            self.bn3 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module('conv', nn.Conv2d(in_channels,
                out_channels, kernel_size=1, stride=stride, padding=0, bias
                =False))

    def forward(self, x):
        if self._preact:
            x = F.relu(self.bn1(x), inplace=True)
            y = self.conv1(x)
        else:
            y = self.bn1(x)
            if not self._remove_first_relu:
                y = F.relu(y, inplace=True)
            y = self.conv1(y)
        y = F.relu(self.bn2(y), inplace=True)
        y = self.conv2(y)
        if self._add_last_bn:
            y = self.bn3(y)
        y += self.shortcut(x)
        return y


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride, remove_first_relu,
        add_last_bn, preact=False):
        super(BottleneckBlock, self).__init__()
        self._remove_first_relu = remove_first_relu
        self._add_last_bn = add_last_bn
        self._preact = preact
        bottleneck_channels = out_channels // self.expansion
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels,
            kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels,
            kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(bottleneck_channels)
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels,
            kernel_size=1, stride=1, padding=0, bias=False)
        if add_last_bn:
            self.bn4 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module('conv', nn.Conv2d(in_channels,
                out_channels, kernel_size=1, stride=stride, padding=0, bias
                =False))

    def forward(self, x):
        if self._preact:
            x = F.relu(self.bn1(x), inplace=True)
            y = self.conv1(x)
        else:
            y = self.bn1(x)
            if not self._remove_first_relu:
                y = F.relu(y, inplace=True)
            y = self.conv1(y)
        y = F.relu(self.bn2(y), inplace=True)
        y = self.conv2(y)
        y = F.relu(self.bn3(y), inplace=True)
        y = self.conv3(y)
        if self._add_last_bn:
            y = self.bn4(y)
        y += self.shortcut(x)
        return y


def initialize_weights(method='kaiming', *models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.
                ConvTranspose2d) or isinstance(module, nn.Linear):
                if method == 'kaiming':
                    init.kaiming_normal_(module.weight.data, np.sqrt(2.0))
                elif method == 'xavier':
                    init.xavier_normal(module.weight.data, np.sqrt(2.0))
                elif method == 'orthogonal':
                    init.orthogonal(module.weight.data, np.sqrt(2.0))
                elif method == 'normal':
                    init.normal(module.weight.data, mean=0, std=0.02)
                if module.bias is not None:
                    init.constant(module.bias.data, 0)


class Network(nn.Module):

    def __init__(self, config):
        super(Network, self).__init__()
        input_shape = config['input_shape']
        n_classes = config['n_classes']
        base_channels = config['base_channels']
        self._remove_first_relu = config['remove_first_relu']
        self._add_last_bn = config['add_last_bn']
        block_type = config['block_type']
        depth = config['depth']
        preact_stage = config['preact_stage']
        assert block_type in ['basic', 'bottleneck']
        if block_type == 'basic':
            block = BasicBlock
            n_blocks_per_stage = (depth - 2) // 6
            assert n_blocks_per_stage * 6 + 2 == depth
        else:
            block = BottleneckBlock
            n_blocks_per_stage = (depth - 2) // 9
            assert n_blocks_per_stage * 9 + 2 == depth
        n_channels = [base_channels, base_channels * 2 * block.expansion, 
            base_channels * 4 * block.expansion]
        self.conv = nn.Conv2d(input_shape[1], n_channels[0], kernel_size=(3,
            3), stride=1, padding=1, bias=False)
        self.stage1 = self._make_stage(n_channels[0], n_channels[0],
            n_blocks_per_stage, block, stride=1, preact=preact_stage[0])
        self.stage2 = self._make_stage(n_channels[0], n_channels[1],
            n_blocks_per_stage, block, stride=2, preact=preact_stage[1])
        self.stage3 = self._make_stage(n_channels[1], n_channels[2],
            n_blocks_per_stage, block, stride=2, preact=preact_stage[2])
        self.bn = nn.BatchNorm2d(n_channels[2])
        with torch.no_grad():
            self.feature_size = self._forward_conv(torch.zeros(*input_shape)
                ).view(-1).shape[0]
        self.fc = nn.Linear(self.feature_size, n_classes)
        self.apply(initialize_weights)

    def _make_stage(self, in_channels, out_channels, n_blocks, block,
        stride, preact):
        stage = nn.Sequential()
        for index in range(n_blocks):
            block_name = 'block{}'.format(index + 1)
            if index == 0:
                stage.add_module(block_name, block(in_channels,
                    out_channels, stride=stride, remove_first_relu=self.
                    _remove_first_relu, add_last_bn=self._add_last_bn,
                    preact=preact))
            else:
                stage.add_module(block_name, block(out_channels,
                    out_channels, stride=1, remove_first_relu=self.
                    _remove_first_relu, add_last_bn=self._add_last_bn,
                    preact=False))
        return stage

    def _forward_conv(self, x):
        x = self.conv(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = F.relu(self.bn(x), inplace=True)
        x = F.adaptive_avg_pool2d(x, output_size=1)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Swish(nn.Module):

    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = True

    def forward(self, x):
        if self.inplace:
            x.mul_(torch.sigmoid(x))
            return x
        else:
            return x * torch.sigmoid(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.act = Swish(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.act(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.act = Swish(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.act(out)
        return out


class ResNet_swish(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet_swish, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.act = Swish(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.last_linear = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x


class LambdaBase(nn.Sequential):

    def __init__(self, *args):
        super(LambdaBase, self).__init__(*args)

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input


class LambdaBase(nn.Sequential):

    def __init__(self, *args):
        super(LambdaBase, self).__init__(*args)

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input


class LambdaBase(nn.Sequential):

    def __init__(self, *args):
        super(LambdaBase, self).__init__(*args)

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input


class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
            padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
            padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class Bottleneck(nn.Module):
    """
    Base class for bottlenecks that implements `forward()` method.
    """

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
        out = self.se_module(out) + residual
        out = self.relu(out)
        return out


class SENet(nn.Module):

    def __init__(self, block, layers, groups, reduction, dropout_p=0.2,
        inplanes=128, input_3x3=True, downsample_kernel_size=3,
        downsample_padding=1, num_classes=1000):
        """
        Parameters
        ----------
        block (nn.Module): Bottleneck class.
            - For SENet154: SEBottleneck
            - For SE-ResNet models: SEResNetBottleneck
            - For SE-ResNeXt models:  SEResNeXtBottleneck
        layers (list of ints): Number of residual blocks for 4 layers of the
            network (layer1...layer4).
        groups (int): Number of groups for the 3x3 convolution in each
            bottleneck block.
            - For SENet154: 64
            - For SE-ResNet models: 1
            - For SE-ResNeXt models:  32
        reduction (int): Reduction ratio for Squeeze-and-Excitation modules.
            - For all models: 16
        dropout_p (float or None): Drop probability for the Dropout layer.
            If `None` the Dropout layer is not used.
            - For SENet154: 0.2
            - For SE-ResNet models: None
            - For SE-ResNeXt models: None
        inplanes (int):  Number of input channels for layer1.
            - For SENet154: 128
            - For SE-ResNet models: 64
            - For SE-ResNeXt models: 64
        input_3x3 (bool): If `True`, use three 3x3 convolutions instead of
            a single 7x7 convolution in layer0.
            - For SENet154: True
            - For SE-ResNet models: False
            - For SE-ResNeXt models: False
        downsample_kernel_size (int): Kernel size for downsampling convolutions
            in layer2, layer3 and layer4.
            - For SENet154: 3
            - For SE-ResNet models: 1
            - For SE-ResNeXt models: 1
        downsample_padding (int): Padding for downsampling convolutions in
            layer2, layer3 and layer4.
            - For SENet154: 1
            - For SE-ResNet models: 0
            - For SE-ResNeXt models: 0
        num_classes (int): Number of outputs in `last_linear` layer.
            - For all models: 1000
        """
        super(SENet, self).__init__()
        self.inplanes = inplanes
        if input_3x3:
            layer0_modules = [('conv1', nn.Conv2d(3, 64, 3, stride=2,
                padding=1, bias=False)), ('bn1', nn.BatchNorm2d(64)), (
                'relu1', nn.ReLU(inplace=True)), ('conv2', nn.Conv2d(64, 64,
                3, stride=1, padding=1, bias=False)), ('bn2', nn.
                BatchNorm2d(64)), ('relu2', nn.ReLU(inplace=True)), (
                'conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1,
                bias=False)), ('bn3', nn.BatchNorm2d(inplanes)), ('relu3',
                nn.ReLU(inplace=True))]
        else:
            layer0_modules = [('conv1', nn.Conv2d(3, inplanes, kernel_size=
                7, stride=2, padding=3, bias=False)), ('bn1', nn.
                BatchNorm2d(inplanes)), ('relu1', nn.ReLU(inplace=True))]
        layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2, ceil_mode=
            True)))
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        self.layer1 = self._make_layer(block, planes=64, blocks=layers[0],
            groups=groups, reduction=reduction, downsample_kernel_size=1,
            downsample_padding=0)
        self.layer2 = self._make_layer(block, planes=128, blocks=layers[1],
            stride=2, groups=groups, reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding)
        self.layer3 = self._make_layer(block, planes=256, blocks=layers[2],
            stride=2, groups=groups, reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding)
        self.layer4 = self._make_layer(block, planes=512, blocks=layers[3],
            stride=2, groups=groups, reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        self.last_linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, groups, reduction, stride=
        1, downsample_kernel_size=1, downsample_padding=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=downsample_kernel_size, stride
                =stride, padding=downsample_padding, bias=False), nn.
                BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, groups, reduction,
            stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))
        return nn.Sequential(*layers)

    def features(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def logits(self, x):
        x = self.avg_pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x


class _DenseLayer(nn.Sequential):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
            growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate,
            growth_rate, kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate,
        drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate,
                growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):

    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features,
            num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):

    def __init__(self, num_classes=10, mode='A', num_init_features=64,
        growth_rate=32, bn_size=4, drop_rate=0, **kwargs):
        """
        input_args :
            growth_rate (int) - how many filters to add each layer (`k` in paper)
            block_config (list of 4 ints) - how many layers in each pooling block
            num_init_features (int) - the number of filters to learn in the first convolution layer
            bn_size (int) - multiplicative factor for number of bottle neck layers
              (i.e. bn_size * k features in the bottleneck layer)
            drop_rate (float) - dropout rate after each dense layer
            num_classes (int) - number of classification classes
        """
        super(DenseNet, self).__init__()
        self.features = nn.Sequential(OrderedDict([('conv0', nn.Conv2d(3,
            num_init_features, kernel_size=7, stride=2, padding=3, bias=
            False)), ('norm0', nn.BatchNorm2d(num_init_features)), ('relu0',
            nn.ReLU(inplace=True)), ('pool0', nn.MaxPool2d(kernel_size=3,
            stride=2, padding=1))]))
        block_dict = {'A': (6, 12, 24, 16), 'B': (6, 12, 32, 32), 'C': (6, 
            12, 48, 32), 'D': (6, 12, 64, 48)}
        block_config = block_dict[mode]
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=
                num_features, bn_size=bn_size, growth_rate=growth_rate,
                drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.classifier = nn.Linear(num_features, num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        out = self.classifier(out)
        return out


class _DenseLayer(nn.Sequential):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm_1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu_1', nn.ReLU(inplace=True)),
        self.add_module('conv_1', nn.Conv2d(num_input_features, bn_size *
            growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm_2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu_2', nn.ReLU(inplace=True)),
        self.add_module('conv_2', nn.Conv2d(bn_size * growth_rate,
            growth_rate, kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                training=self.training)
        return torch.cat([x, new_features], 1)


class _Transition(nn.Sequential):

    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features,
            num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class OptDenseNet(nn.Module):
    """Optimized Densenet-BC model"""

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
        num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000, **
        kwargs):
        super(OptDenseNet, self).__init__()
        self.features = nn.Sequential(OrderedDict([('conv0', nn.Conv2d(3,
            num_init_features, kernel_size=7, stride=2, padding=3, bias=
            False)), ('norm0', nn.BatchNorm2d(num_init_features)), ('relu0',
            nn.ReLU(inplace=True)), ('pool0', nn.MaxPool2d(kernel_size=3,
            stride=2, padding=1))]))
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            num_input_features = num_features
            for j in range(num_layers):
                layer = _DenseLayer(num_input_features + j * growth_rate,
                    growth_rate, bn_size, drop_rate)
                self.features.add_module('denseblock{}_layer{}'.format(i + 
                    1, j + 1), layer)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x, chunks=None):
        modules = [module for k, module in self._modules.items()][0]
        input_var = x.detach()
        input_var.requires_grad = True
        input_var = checkpoint_sequential(modules, chunks, input_var)
        input_var = F.relu(input_var, inplace=True)
        input_var = F.avg_pool2d(input_var, kernel_size=7, stride=1).view(
            input_var.size(0), -1)
        input_var = self.classifier(input_var)
        return input_var


def act_fn(act):
    if act == 'relu':
        act_ = nn.ReLU(inplace=False)
    elif act == 'lrelu':
        act_ = nn.LeakyReLU(inplace=True)
    elif act == 'prelu':
        act_ = nn.PReLU()
    elif act == 'rrelu':
        act_ = nn.RReLU(inplace=True)
    elif act == 'elu':
        act_ = nn.ELU(inplace=True)
    elif act == 'selu':
        act_ = nn.SELU(inplace=True)
    elif act == 'tanh':
        act_ = nn.Tanh()
    elif act == 'sigmoid':
        act_ = nn.Sigmoid()
    else:
        print('\n\nActivation function {} is not supported/understood\n\n'.
            format(act))
        act_ = None
    return act_


class PerturbLayerFirst(nn.Module):

    def __init__(self, in_channels=None, out_channels=None, nmasks=None,
        level=None, filter_size=None, debug=False, use_act=False, stride=1,
        act=None, unique_masks=False, mix_maps=None, train_masks=False,
        noise_type='uniform', input_size=None):
        super(PerturbLayerFirst, self).__init__()
        self.nmasks = nmasks
        self.unique_masks = unique_masks
        self.train_masks = train_masks
        self.level = level
        self.filter_size = filter_size
        self.use_act = use_act
        self.act = act_fn('sigmoid')
        self.debug = debug
        self.noise_type = noise_type
        self.in_channels = in_channels
        self.input_size = input_size
        self.mix_maps = mix_maps
        if filter_size == 1:
            padding = 0
            bias = True
        elif filter_size == 3 or filter_size == 5:
            padding = 1
            bias = False
        elif filter_size == 7:
            stride = 2
            padding = 3
            bias = False
        if self.filter_size > 0:
            self.noise = None
            self.layers = nn.Sequential(nn.Conv2d(in_channels, out_channels,
                kernel_size=filter_size, padding=padding, stride=stride,
                bias=bias), nn.BatchNorm2d(out_channels), self.act)
        else:
            noise_channels = in_channels if self.unique_masks else 1
            shape = 1, noise_channels, self.nmasks, input_size, input_size
            self.noise = nn.Parameter(torch.Tensor(*shape), requires_grad=
                self.train_masks)
            if noise_type == 'uniform':
                self.noise.data.uniform_(-1, 1)
            elif self.noise_type == 'normal':
                self.noise.data.normal_()
            else:
                None
            if nmasks != 1:
                if out_channels % in_channels != 0:
                    None
                groups = in_channels
            else:
                groups = 1
            self.layers = nn.Sequential(nn.BatchNorm2d(in_channels * self.
                nmasks), self.act, nn.Conv2d(in_channels * self.nmasks,
                out_channels, kernel_size=1, stride=1, groups=groups), nn.
                BatchNorm2d(out_channels), self.act)
            if self.mix_maps:
                self.mix_layers = nn.Sequential(nn.Conv2d(out_channels,
                    out_channels, kernel_size=1, stride=1, groups=1), nn.
                    BatchNorm2d(out_channels), self.act)

    def forward(self, x):
        if self.filter_size > 0:
            return self.layers(x)
        else:
            y = torch.add(x.unsqueeze(2), self.noise * self.level)
            y = y.view(-1, self.in_channels * self.nmasks, self.input_size,
                self.input_size)
            y = self.layers(y)
            if self.mix_maps:
                y = self.mix_layers(y)
            return y


class PerturbLayer(nn.Module):

    def __init__(self, in_channels=None, out_channels=None, nmasks=None,
        level=None, filter_size=None, debug=False, use_act=False, stride=1,
        act=None, unique_masks=False, mix_maps=None, train_masks=False,
        noise_type='uniform', input_size=None):
        super(PerturbLayer, self).__init__()
        self.nmasks = nmasks
        self.unique_masks = unique_masks
        self.train_masks = train_masks
        self.level = level
        self.filter_size = filter_size
        self.use_act = use_act
        self.act = act_fn(act)
        self.debug = debug
        self.noise_type = noise_type
        self.in_channels = in_channels
        self.input_size = input_size
        self.mix_maps = mix_maps
        if filter_size == 1:
            padding = 0
            bias = True
        elif filter_size == 3 or filter_size == 5:
            padding = 1
            bias = False
        elif filter_size == 7:
            stride = 2
            padding = 3
            bias = False
        if self.filter_size > 0:
            self.noise = None
            self.layers = nn.Sequential(nn.Conv2d(in_channels, out_channels,
                kernel_size=filter_size, padding=padding, stride=stride,
                bias=bias), nn.BatchNorm2d(out_channels), self.act)
        else:
            noise_channels = in_channels if self.unique_masks else 1
            shape = 1, noise_channels, self.nmasks, input_size, input_size
            self.noise = nn.Parameter(torch.Tensor(*shape), requires_grad=
                self.train_masks)
            if noise_type == 'uniform':
                self.noise.data.uniform_(-1, 1)
            elif self.noise_type == 'normal':
                self.noise.data.normal_()
            else:
                None
            if nmasks != 1:
                if out_channels % in_channels != 0:
                    None
                groups = in_channels
            else:
                groups = 1
            self.layers = nn.Sequential(nn.Conv2d(in_channels * self.nmasks,
                out_channels, kernel_size=1, stride=1, groups=groups), nn.
                BatchNorm2d(out_channels), self.act)
            if self.mix_maps:
                self.mix_layers = nn.Sequential(nn.Conv2d(out_channels,
                    out_channels, kernel_size=1, stride=1, groups=1), nn.
                    BatchNorm2d(out_channels), self.act)

    def forward(self, x):
        if self.filter_size > 0:
            return self.layers(x)
        else:
            y = torch.add(x.unsqueeze(2), self.noise * self.level)
            if self.use_act:
                y = self.act(y)
            y = y.view(-1, self.in_channels * self.nmasks, self.input_size,
                self.input_size)
            y = self.layers(y)
            if self.mix_maps:
                y = self.mix_layers(y)
            return y


class PerturbBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels=None, out_channels=None, stride=1,
        shortcut=None, nmasks=None, train_masks=False, level=None, use_act=
        False, filter_size=None, act=None, unique_masks=False, noise_type=
        None, input_size=None, pool_type=None, mix_maps=None):
        super(PerturbBasicBlock, self).__init__()
        self.shortcut = shortcut
        if pool_type == 'max':
            pool = nn.MaxPool2d
        elif pool_type == 'avg':
            pool = nn.AvgPool2d
        else:
            None
            return
        self.layers = nn.Sequential(PerturbLayer(in_channels=in_channels,
            out_channels=out_channels, nmasks=nmasks, input_size=input_size,
            level=level, filter_size=filter_size, use_act=use_act,
            train_masks=train_masks, act=act, unique_masks=unique_masks,
            noise_type=noise_type, mix_maps=mix_maps), pool(stride, stride),
            PerturbLayer(in_channels=out_channels, out_channels=
            out_channels, nmasks=nmasks, input_size=input_size // stride,
            level=level, filter_size=filter_size, use_act=use_act,
            train_masks=train_masks, act=act, unique_masks=unique_masks,
            noise_type=noise_type, mix_maps=mix_maps))

    def forward(self, x):
        residual = x
        y = self.layers(x)
        if self.shortcut:
            residual = self.shortcut(x)
        y += residual
        y = F.relu(y)
        return y


class PerturbResNet(nn.Module):

    def __init__(self, block, nblocks=None, avgpool=None, nfilters=None,
        nclasses=None, nmasks=None, input_size=32, level=None, filter_size=
        None, first_filter_size=None, use_act=False, train_masks=False,
        mix_maps=None, act=None, scale_noise=1, unique_masks=False, debug=
        False, noise_type=None, pool_type=None):
        super(PerturbResNet, self).__init__()
        self.nfilters = nfilters
        self.unique_masks = unique_masks
        self.noise_type = noise_type
        self.train_masks = train_masks
        self.pool_type = pool_type
        self.mix_maps = mix_maps
        self.act = act_fn(act)
        layers = [PerturbLayerFirst(in_channels=3, out_channels=3 *
            nfilters, nmasks=nfilters * 5, level=level * scale_noise * 20,
            debug=debug, filter_size=first_filter_size, use_act=use_act,
            train_masks=train_masks, input_size=input_size, act=act,
            unique_masks=self.unique_masks, noise_type=self.noise_type,
            mix_maps=mix_maps)]
        if first_filter_size == 7:
            layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.pre_layers = nn.Sequential(*layers, nn.Conv2d(self.nfilters * 
            3 * 1, self.nfilters, kernel_size=1, stride=1, bias=False), nn.
            BatchNorm2d(self.nfilters), self.act)
        self.layer1 = self._make_layer(block, 1 * nfilters, nblocks[0],
            stride=1, level=level, nmasks=nmasks, use_act=True, filter_size
            =filter_size, act=act, input_size=input_size)
        self.layer2 = self._make_layer(block, 2 * nfilters, nblocks[1],
            stride=2, level=level, nmasks=nmasks, use_act=True, filter_size
            =filter_size, act=act, input_size=input_size)
        self.layer3 = self._make_layer(block, 4 * nfilters, nblocks[2],
            stride=2, level=level, nmasks=nmasks, use_act=True, filter_size
            =filter_size, act=act, input_size=input_size // 2)
        self.layer4 = self._make_layer(block, 8 * nfilters, nblocks[3],
            stride=2, level=level, nmasks=nmasks, use_act=True, filter_size
            =filter_size, act=act, input_size=input_size // 4)
        self.avgpool = nn.AvgPool2d(avgpool, stride=1)
        self.linear = nn.Linear(8 * nfilters * block.expansion, nclasses)

    def _make_layer(self, block, out_channels, nblocks, stride=1, level=0.2,
        nmasks=None, use_act=False, filter_size=None, act=None, input_size=None
        ):
        shortcut = None
        if stride != 1 or self.nfilters != out_channels * block.expansion:
            shortcut = nn.Sequential(nn.Conv2d(self.nfilters, out_channels *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion))
        layers = []
        layers.append(block(self.nfilters, out_channels, stride, shortcut,
            level=level, nmasks=nmasks, use_act=use_act, filter_size=
            filter_size, act=act, unique_masks=self.unique_masks,
            noise_type=self.noise_type, train_masks=self.train_masks,
            input_size=input_size, pool_type=self.pool_type, mix_maps=self.
            mix_maps))
        self.nfilters = out_channels * block.expansion
        for i in range(1, nblocks):
            layers.append(block(self.nfilters, out_channels, level=level,
                nmasks=nmasks, use_act=use_act, train_masks=self.
                train_masks, filter_size=filter_size, act=act, unique_masks
                =self.unique_masks, noise_type=self.noise_type, input_size=
                input_size // stride, pool_type=self.pool_type, mix_maps=
                self.mix_maps))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre_layers(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class LeNet(nn.Module):

    def __init__(self, nfilters=None, nclasses=None, nmasks=None, level=
        None, filter_size=None, linear=128, input_size=28, debug=False,
        scale_noise=1, act='relu', use_act=False, first_filter_size=None,
        pool_type=None, dropout=None, unique_masks=False, train_masks=False,
        noise_type='uniform', mix_maps=None):
        super(LeNet, self).__init__()
        if filter_size == 5:
            n = 5
        else:
            n = 4
        if input_size == 32:
            first_channels = 3
        elif input_size == 28:
            first_channels = 1
        if pool_type == 'max':
            pool = nn.MaxPool2d
        elif pool_type == 'avg':
            pool = nn.AvgPool2d
        else:
            None
            return
        self.linear1 = nn.Linear(nfilters * n * n, linear)
        self.linear2 = nn.Linear(linear, nclasses)
        self.dropout = nn.Dropout(p=dropout)
        self.act = act_fn(act)
        self.batch_norm = nn.BatchNorm1d(linear)
        self.first_layers = nn.Sequential(PerturbLayer(in_channels=
            first_channels, out_channels=nfilters, nmasks=nmasks, level=
            level * scale_noise, filter_size=first_filter_size, use_act=
            use_act, act=act, unique_masks=unique_masks, train_masks=
            train_masks, noise_type=noise_type, input_size=input_size,
            mix_maps=mix_maps), pool(kernel_size=3, stride=2, padding=1),
            PerturbLayer(in_channels=nfilters, out_channels=nfilters,
            nmasks=nmasks, level=level, filter_size=filter_size, use_act=
            True, act=act, unique_masks=unique_masks, debug=debug,
            train_masks=train_masks, noise_type=noise_type, input_size=
            input_size // 2, mix_maps=mix_maps), pool(kernel_size=3, stride
            =2, padding=1), PerturbLayer(in_channels=nfilters, out_channels
            =nfilters, nmasks=nmasks, level=level, filter_size=filter_size,
            use_act=True, act=act, unique_masks=unique_masks, train_masks=
            train_masks, noise_type=noise_type, input_size=input_size // 4,
            mix_maps=mix_maps), pool(kernel_size=3, stride=2, padding=1))
        self.last_layers = nn.Sequential(self.dropout, self.linear1, self.
            batch_norm, self.act, self.dropout, self.linear2)

    def forward(self, x):
        x = self.first_layers(x)
        x = x.view(x.size(0), -1)
        x = self.last_layers(x)
        return x


class _DenseLayer(nn.Sequential):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('selayer', SELayer(channel=num_input_features)),
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
            growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate,
            growth_rate, kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate,
        drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate,
                growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):

    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('selayer', SELayer(channel=num_input_features))
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features,
            num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class SEDenseNet(nn.Module):
    """Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
        num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):
        super(SEDenseNet, self).__init__()
        self.features = nn.Sequential(OrderedDict([('conv0', nn.Conv2d(3,
            num_init_features, kernel_size=7, stride=2, padding=3, bias=
            False)), ('norm0', nn.BatchNorm2d(num_init_features)), ('relu0',
            nn.ReLU(inplace=True)), ('pool0', nn.MaxPool2d(kernel_size=3,
            stride=2, padding=1))]))
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=
                num_features, bn_size=bn_size, growth_rate=growth_rate,
                drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.classifier = nn.Linear(num_features, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size
            (0), -1)
        out = self.classifier(out)
        return out


def _bn_function_factory(norm, relu, conv):

    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output
    return bn_function


class _DenseLayer(nn.Module):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate,
        efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('selayer', SELayer(channel=num_input_features)),
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
            growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate,
            growth_rate, kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate
        self.efficient = efficient

    def forward(self, *prev_features):
        """原有的两次BN层需要消耗的两块显存空间，
        通过使用checkpoint，实现了只开辟一块空间用来存储中间特征
        """
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.efficient and any(prev_feature.requires_grad for
            prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                training=self.training)
        return new_features


class _DenseBlock(nn.Module):

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate,
        drop_rate, efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate,
                growth_rate=growth_rate, bn_size=bn_size, drop_rate=
                drop_rate, efficient=efficient)
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):

    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features,
            num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    """Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 3 or 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
            (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        small_inputs (bool) - set to True if images are 32x32. Otherwise assumes images are larger.
        efficient (bool) - set to True to use checkpointing. Much more memory efficient, but slower.
    """

    def __init__(self, growth_rate=12, block_config=(16, 16, 16),
        compression=0.5, num_init_features=24, bn_size=4, drop_rate=0,
        num_classes=4096, efficient=True):
        super(DenseNet, self).__init__()
        assert 0 < compression <= 1, 'compression of densenet should be between 0 and 1'
        self.features = nn.Sequential(OrderedDict([('conv0', nn.Conv2d(3,
            num_init_features, kernel_size=7, stride=2, padding=3, bias=
            False)), ('norm0', nn.BatchNorm2d(num_init_features)), ('relu0',
            nn.ReLU(inplace=True)), ('pool0', nn.MaxPool2d(kernel_size=3,
            stride=2, padding=1))]))
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=
                num_features, bn_size=bn_size, growth_rate=growth_rate,
                drop_rate=drop_rate, efficient=efficient)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                self.features.add_module('SELayer_%da' % (i + 1), SELayer(
                    channel=num_features))
                trans = _Transition(num_input_features=num_features,
                    num_output_features=int(num_features * compression))
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.classifier = nn.Linear(num_features, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7).view(features.size(0), -1)
        out = self.classifier(out)
        return out


class SELayer(nn.Module):

    def __init__(self, channel, reduction=16):
        assert channel > reduction, 'Make sure your input channel bigger than reduction which equals to {}'.format(
            reduction)
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True), nn.Linear(channel // reduction, channel),
            nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class WideResNet(nn.Module):

    def __init__(self, pooling, f, params):
        super(WideResNet, self).__init__()
        self.pooling = pooling
        self.f = f
        self.params = params

    def forward(self, x):
        x = self.f(x, self.params, self.pooling)
        return x


class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
        padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size,
            stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1,
            bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):

    def __init__(self, in_filters, out_filters, reps, strides=1,
        start_with_relu=True, grow_first=True):
        super(Block, self).__init__()
        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=
                strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None
        rep = []
        filters = in_filters
        if grow_first:
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1,
                padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters
        for i in range(reps - 1):
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(filters, filters, 3, stride=1,
                padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))
        if not grow_first:
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1,
                padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)
        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)
        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp
        x += skip
        return x


class Xception(nn.Module):

    def __init__(self, num_classes=1000):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.block1 = Block(64, 128, 2, 2, start_with_relu=False,
            grow_first=True)
        self.block2 = Block(128, 256, 2, 2, start_with_relu=True,
            grow_first=True)
        self.block3 = Block(256, 728, 2, 2, start_with_relu=True,
            grow_first=True)
        self.block4 = Block(728, 728, 3, 1, start_with_relu=True,
            grow_first=True)
        self.block5 = Block(728, 728, 3, 1, start_with_relu=True,
            grow_first=True)
        self.block6 = Block(728, 728, 3, 1, start_with_relu=True,
            grow_first=True)
        self.block7 = Block(728, 728, 3, 1, start_with_relu=True,
            grow_first=True)
        self.block8 = Block(728, 728, 3, 1, start_with_relu=True,
            grow_first=True)
        self.block9 = Block(728, 728, 3, 1, start_with_relu=True,
            grow_first=True)
        self.block10 = Block(728, 728, 3, 1, start_with_relu=True,
            grow_first=True)
        self.block11 = Block(728, 728, 3, 1, start_with_relu=True,
            grow_first=True)
        self.block12 = Block(728, 1024, 2, 2, start_with_relu=True,
            grow_first=False)
        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)
        self.fc = nn.Linear(2048, num_classes)

    def features(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        return x

    def logits(self, features):
        x = nn.ReLU(inplace=True)(features)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size
            =1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.
                expansion * planes, kernel_size=1, stride=stride, bias=
                False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class FPN(nn.Module):

    def __init__(self, block, num_blocks):
        super(FPN, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0
            )
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1,
            padding=0)
        self.latlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0
            )
        self.latlayer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0
            )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        """Upsample and add two feature maps.

        Args:
          x: (Tensor) top feature map to be upsampled.
          y: (Tensor) lateral feature map.

        Returns:
          (Tensor) added feature map.

        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.interpolate(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.

        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]

        So we choose bilinear upsample which supports arbitrary output sizes.
        """
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def forward(self, x):
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)
        return p2


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size
            =1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.
                expansion * planes, kernel_size=1, stride=stride, bias=
                False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class RetinaFPN(nn.Module):

    def __init__(self, block, num_blocks):
        super(RetinaFPN, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer2 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer3 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer4 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer5 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.conv6 = nn.Conv2d(2048, 256, kernel_size=3, stride=2, padding=1)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0
            )
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1,
            padding=0)
        self.latlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0
            )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        """Upsample and add two feature maps.

        Args:
          x: (Tensor) top feature map to be upsampled.
          y: (Tensor) lateral feature map.

        Returns:
          (Tensor) added feature map.

        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.interpolate(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.

        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]

        So we choose bilinear upsample which supports arbitrary output sizes.
        """
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def forward(self, x):
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        c5 = self.layer5(c4)
        p6 = self.conv6(c5)
        p7 = self.conv7(F.relu(p6))
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        return p3, p4, p5, p6, p7


class BiSeNet(nn.Module):

    def __init__(self, num_classes, pretrained=True, backbone='resnet18',
        aux=False, **kwargs):
        super(BiSeNet, self).__init__()
        self.aux = aux
        self.spatial_path = SpatialPath(3, 128, **kwargs)
        self.context_path = ContextPath(backbone=backbone, pretrained=
            pretrained, **kwargs)
        self.ffm = FeatureFusion(256, 256, 4, **kwargs)
        self.head = _BiSeHead(256, 64, num_classes, **kwargs)
        if aux:
            self.auxlayer1 = _BiSeHead(128, 256, num_classes, **kwargs)
            self.auxlayer2 = _BiSeHead(128, 256, num_classes, **kwargs)
        self.__setattr__('exclusive', ['spatial_path', 'context_path',
            'ffm', 'head', 'auxlayer1', 'auxlayer2'] if aux else [
            'spatial_path', 'context_path', 'ffm', 'head'])

    def forward(self, x):
        size = x.size()[2:]
        spatial_out = self.spatial_path(x)
        context_out = self.context_path(x)
        fusion_out = self.ffm(spatial_out, context_out[-1])
        outputs = []
        x = self.head(fusion_out)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        outputs.append(x)
        if self.aux:
            auxout1 = self.auxlayer1(context_out[0])
            auxout1 = F.interpolate(auxout1, size, mode='bilinear',
                align_corners=True)
            outputs.append(auxout1)
            auxout2 = self.auxlayer2(context_out[1])
            auxout2 = F.interpolate(auxout2, size, mode='bilinear',
                align_corners=True)
            outputs.append(auxout2)
            return tuple(outputs)
        else:
            return outputs[0]


class _BiSeHead(nn.Module):

    def __init__(self, in_channels, inter_channels, nclass, norm_layer=nn.
        BatchNorm2d, **kwargs):
        super(_BiSeHead, self).__init__()
        self.block = nn.Sequential(_ConvBNReLU(in_channels, inter_channels,
            3, 1, 1, norm_layer=norm_layer, **kwargs), nn.Dropout(0.1), nn.
            Conv2d(inter_channels, nclass, 1))

    def forward(self, x):
        x = self.block(x)
        return x


class _ConvBNReLU(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
        padding=1, dilation=1, groups=1, norm_layer=nn.BatchNorm2d, bias=
        False, **kwargs):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
            stride, padding, dilation, groups, bias)
        self.bn = norm_layer(out_channels)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class SpatialPath(nn.Module):
    """Spatial path"""

    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d,
        **kwargs):
        super(SpatialPath, self).__init__()
        inter_channels = 64
        self.conv7x7 = _ConvBNReLU(in_channels, inter_channels, 7, 2, 3,
            norm_layer=norm_layer, **kwargs)
        self.conv3x3_1 = _ConvBNReLU(inter_channels, inter_channels, 3, 2, 
            1, norm_layer=norm_layer, **kwargs)
        self.conv3x3_2 = _ConvBNReLU(inter_channels, inter_channels, 3, 2, 
            1, norm_layer=norm_layer, **kwargs)
        self.conv1x1 = _ConvBNReLU(inter_channels, out_channels, 1, 1, 0,
            norm_layer=norm_layer, **kwargs)

    def forward(self, x):
        x = self.conv7x7(x)
        x = self.conv3x3_1(x)
        x = self.conv3x3_2(x)
        x = self.conv1x1(x)
        return x


class _GlobalAvgPooling(nn.Module):

    def __init__(self, in_channels, out_channels, norm_layer, **kwargs):
        super(_GlobalAvgPooling, self).__init__()
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(
            in_channels, out_channels, 1, bias=False), norm_layer(
            out_channels), nn.ReLU(True))

    def forward(self, x):
        size = x.size()[2:]
        pool = self.gap(x)
        out = F.interpolate(pool, size, mode='bilinear', align_corners=True)
        return out


class AttentionRefinmentModule(nn.Module):

    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d,
        **kwargs):
        super(AttentionRefinmentModule, self).__init__()
        self.conv3x3 = _ConvBNReLU(in_channels, out_channels, 3, 1, 1,
            norm_layer=norm_layer, **kwargs)
        self.channel_attention = nn.Sequential(nn.AdaptiveAvgPool2d(1),
            _ConvBNReLU(out_channels, out_channels, 1, 1, 0, norm_layer=
            norm_layer, **kwargs), nn.Sigmoid())

    def forward(self, x):
        x = self.conv3x3(x)
        attention = self.channel_attention(x)
        x = x * attention
        return x


model_urls = {'resnet50':
    'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'}


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DilatedResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


class ContextPath(nn.Module):

    def __init__(self, pretrained=True, backbone='resnet18', norm_layer=nn.
        BatchNorm2d, **kwargs):
        super(ContextPath, self).__init__()
        if backbone == 'resnet18':
            pretrained = resnet18(pretrained=pretrained, **kwargs)
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))
        self.conv1 = pretrained.conv1
        self.bn1 = pretrained.bn1
        self.relu = pretrained.relu
        self.maxpool = pretrained.maxpool
        self.layer1 = pretrained.layer1
        self.layer2 = pretrained.layer2
        self.layer3 = pretrained.layer3
        self.layer4 = pretrained.layer4
        inter_channels = 128
        self.global_context = _GlobalAvgPooling(512, inter_channels,
            norm_layer, **kwargs)
        self.arms = nn.ModuleList([AttentionRefinmentModule(512,
            inter_channels, norm_layer, **kwargs), AttentionRefinmentModule
            (256, inter_channels, norm_layer, **kwargs)])
        self.refines = nn.ModuleList([_ConvBNReLU(inter_channels,
            inter_channels, 3, 1, 1, norm_layer=norm_layer, **kwargs),
            _ConvBNReLU(inter_channels, inter_channels, 3, 1, 1, norm_layer
            =norm_layer, **kwargs)])

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        context_blocks = []
        context_blocks.append(x)
        x = self.layer2(x)
        context_blocks.append(x)
        c3 = self.layer3(x)
        context_blocks.append(c3)
        c4 = self.layer4(c3)
        context_blocks.append(c4)
        context_blocks.reverse()
        global_context = self.global_context(c4)
        last_feature = global_context
        context_outputs = []
        for i, (feature, arm, refine) in enumerate(zip(context_blocks[:2],
            self.arms, self.refines)):
            feature = arm(feature)
            feature += last_feature
            last_feature = F.interpolate(feature, size=context_blocks[i + 1
                ].size()[2:], mode='bilinear', align_corners=True)
            last_feature = refine(last_feature)
            context_outputs.append(last_feature)
        return context_outputs


class FeatureFusion(nn.Module):

    def __init__(self, in_channels, out_channels, reduction=1, norm_layer=
        nn.BatchNorm2d, **kwargs):
        super(FeatureFusion, self).__init__()
        self.conv1x1 = _ConvBNReLU(in_channels, out_channels, 1, 1, 0,
            norm_layer=norm_layer, **kwargs)
        self.channel_attention = nn.Sequential(nn.AdaptiveAvgPool2d(1),
            _ConvBNReLU(out_channels, out_channels // reduction, 1, 1, 0,
            norm_layer=norm_layer, **kwargs), _ConvBNReLU(out_channels //
            reduction, out_channels, 1, 1, 0, norm_layer=norm_layer, **
            kwargs), nn.Sigmoid())

    def forward(self, x1, x2):
        fusion = torch.cat([x1, x2], dim=1)
        out = self.conv1x1(fusion)
        attention = self.channel_attention(out)
        out = out + out * attention
        return out


class Conv3BN(nn.Module):
    """A module which applies the following actions:
        - convolution with 3x3 kernel;
        - batch normalization (if enabled);
        - ELU.
    Attributes:
        in_ch: Number of input channels.
        out_ch: Number of output channels.
        bn: A boolean indicating if Batch Normalization is enabled or not.
    """

    def __init__(self, in_ch: int, out_ch: int, bn=True):
        super(Conv3BN, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch) if bn else None
        self.activation = nn.ELU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.activation(x)
        return x


BN_EPS = 0.0001


class ConvBnRelu2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1,
        dilation=1, stride=1, groups=1, is_bn=True, is_relu=True):
        super(ConvBnRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=
            kernel_size, padding=padding, stride=stride, dilation=dilation,
            groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=BN_EPS)
        self.relu = nn.ReLU(inplace=True)
        if is_bn is False:
            self.bn = None
        if is_relu is False:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

    def merge_bn(self):
        if self.bn == None:
            return
        assert self.conv.bias == None
        conv_weight = self.conv.weight.data
        bn_weight = self.bn.weight.data
        bn_bias = self.bn.bias.data
        bn_running_mean = self.bn.running_mean
        bn_running_var = self.bn.running_var
        bn_eps = self.bn.eps
        N, C, KH, KW = conv_weight.size()
        std = 1 / torch.sqrt(bn_running_var + bn_eps)
        std_bn_weight = (std * bn_weight).repeat(C * KH * KW, 1).t(
            ).contiguous().view(N, C, KH, KW)
        conv_weight_hat = std_bn_weight * conv_weight
        conv_bias_hat = bn_bias - bn_weight * std * bn_running_mean
        self.bn = None
        self.conv = nn.Conv2d(in_channels=self.conv.in_channels,
            out_channels=self.conv.out_channels, kernel_size=self.conv.
            kernel_size, padding=self.conv.padding, stride=self.conv.stride,
            dilation=self.conv.dilation, groups=self.conv.groups, bias=True)
        self.conv.weight.data = conv_weight_hat
        self.conv.bias.data = conv_bias_hat


class ConvResidual(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ConvResidual, self).__init__()
        self.block = nn.Sequential(ConvBnRelu2d(in_channels, out_channels,
            kernel_size=3, padding=1, stride=1), ConvBnRelu2d(out_channels,
            out_channels, kernel_size=3, padding=1, stride=1, is_relu=False))
        self.shortcut = None
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Conv2d(in_channels, out_channels,
                kernel_size=1, padding=0, stride=stride, bias=True)

    def forward(self, x):
        r = x if self.shortcut is None else self.shortcut(x)
        x = self.block(x)
        x = F.relu(x + r, inplace=True)
        return x


class StackEncoder(nn.Module):

    def __init__(self, x_channels, y_channels, kernel_size=3):
        super(StackEncoder, self).__init__()
        padding = (kernel_size - 1) // 2
        self.encode = nn.Sequential(ConvBnRelu2d(x_channels, y_channels,
            kernel_size=kernel_size, padding=padding, dilation=1, stride=1,
            groups=1), ConvBnRelu2d(y_channels, y_channels, kernel_size=
            kernel_size, padding=padding, dilation=1, stride=1, groups=1))

    def forward(self, x):
        y = self.encode(x)
        y_small = F.max_pool2d(y, kernel_size=2, stride=2)
        return y, y_small


class StackDecoder(nn.Module):

    def __init__(self, x_big_channels, x_channels, y_channels, kernel_size=3):
        super(StackDecoder, self).__init__()
        padding = (kernel_size - 1) // 2
        self.decode = nn.Sequential(ConvBnRelu2d(x_big_channels +
            x_channels, y_channels, kernel_size=kernel_size, padding=
            padding, dilation=1, stride=1, groups=1), ConvBnRelu2d(
            y_channels, y_channels, kernel_size=kernel_size, padding=
            padding, dilation=1, stride=1, groups=1), ConvBnRelu2d(
            y_channels, y_channels, kernel_size=kernel_size, padding=
            padding, dilation=1, stride=1, groups=1))

    def forward(self, x_big, x):
        N, C, H, W = x_big.size()
        y = F.interpolate(x, size=(H, W), mode='bilinear')
        y = torch.cat([y, x_big], 1)
        y = self.decode(y)
        return y


class ResStackEncoder(nn.Module):

    def __init__(self, x_channels, y_channels):
        super(ResStackEncoder, self).__init__()
        self.encode = ConvResidual(x_channels, y_channels)

    def forward(self, x):
        y = self.encode(x)
        y_small = F.max_pool2d(y, kernel_size=2, stride=2)
        return y, y_small


class ResStackDecoder(nn.Module):

    def __init__(self, x_big_channels, x_channels, y_channels, kernel_size=3):
        super(ResStackDecoder, self).__init__()
        padding = (kernel_size - 1) // 2
        self.decode = nn.Sequential(ConvBnRelu2d(x_big_channels +
            x_channels, y_channels, kernel_size=kernel_size, padding=
            padding, dilation=1, stride=1, groups=1), ConvResidual(
            y_channels, y_channels))

    def forward(self, x_big, x):
        N, C, H, W = x_big.size()
        y = F.interpolate(x, size=(H, W), mode='bilinear')
        y = torch.cat([y, x_big], 1)
        y = self.decode(y)
        return y


class UNet1024(nn.Module):

    def __init__(self, in_shape=(3, 1024, 1024), **kwargs):
        super(UNet1024, self).__init__()
        C, H, W = in_shape
        self.down1 = StackEncoder(C, 24, kernel_size=3)
        self.down2 = StackEncoder(24, 64, kernel_size=3)
        self.down3 = StackEncoder(64, 128, kernel_size=3)
        self.down4 = StackEncoder(128, 256, kernel_size=3)
        self.down5 = StackEncoder(256, 512, kernel_size=3)
        self.down6 = StackEncoder(512, 768, kernel_size=3)
        self.center = nn.Sequential(ConvBnRelu2d(768, 768, kernel_size=3,
            padding=1, stride=1))
        self.up6 = StackDecoder(768, 768, 512, kernel_size=3)
        self.up5 = StackDecoder(512, 512, 256, kernel_size=3)
        self.up4 = StackDecoder(256, 256, 128, kernel_size=3)
        self.up3 = StackDecoder(128, 128, 64, kernel_size=3)
        self.up2 = StackDecoder(64, 64, 24, kernel_size=3)
        self.up1 = StackDecoder(24, 24, 24, kernel_size=3)
        self.classify = nn.Conv2d(24, 1, kernel_size=1, padding=0, stride=1,
            bias=True)

    def forward(self, x):
        out = x
        down1, out = self.down1(out)
        down2, out = self.down2(out)
        down3, out = self.down3(out)
        down4, out = self.down4(out)
        down5, out = self.down5(out)
        down6, out = self.down6(out)
        pass
        out = self.center(out)
        out = self.up6(down6, out)
        out = self.up5(down5, out)
        out = self.up4(down4, out)
        out = self.up3(down3, out)
        out = self.up2(down2, out)
        out = self.up1(down1, out)
        out = self.classify(out)
        return out


class UNet512(nn.Module):

    def __init__(self, in_shape=(3, 512, 512), **kwargs):
        super(UNet512, self).__init__()
        C, H, W = in_shape
        self.down2 = StackEncoder(C, 64, kernel_size=3)
        self.down3 = StackEncoder(64, 128, kernel_size=3)
        self.down4 = StackEncoder(128, 256, kernel_size=3)
        self.down5 = StackEncoder(256, 512, kernel_size=3)
        self.down6 = StackEncoder(512, 1024, kernel_size=3)
        self.center = nn.Sequential(ConvBnRelu2d(1024, 1024, kernel_size=3,
            padding=1, stride=1))
        self.up6 = StackDecoder(1024, 1024, 512, kernel_size=3)
        self.up5 = StackDecoder(512, 512, 256, kernel_size=3)
        self.up4 = StackDecoder(256, 256, 128, kernel_size=3)
        self.up3 = StackDecoder(128, 128, 64, kernel_size=3)
        self.up2 = StackDecoder(64, 64, 32, kernel_size=3)
        self.classify = nn.Conv2d(32, 1, kernel_size=1, padding=0, stride=1,
            bias=True)

    def forward(self, x):
        out = x
        down2, out = self.down2(out)
        down3, out = self.down3(out)
        down4, out = self.down4(out)
        down5, out = self.down5(out)
        down6, out = self.down6(out)
        pass
        out = self.center(out)
        out = self.up6(down6, out)
        out = self.up5(down5, out)
        out = self.up4(down4, out)
        out = self.up3(down3, out)
        out = self.up2(down2, out)
        out = self.classify(out)
        return out


class UNet256(nn.Module):

    def __init__(self, in_shape=(3, 256, 256), **kwargs):
        super(UNet256, self).__init__()
        C, H, W = in_shape
        self.down2 = StackEncoder(C, 64, kernel_size=3)
        self.down3 = StackEncoder(64, 128, kernel_size=3)
        self.down4 = StackEncoder(128, 256, kernel_size=3)
        self.down5 = StackEncoder(256, 512, kernel_size=3)
        self.down6 = StackEncoder(512, 1024, kernel_size=3)
        self.center = nn.Sequential(ConvBnRelu2d(1024, 1024, kernel_size=3,
            padding=1, stride=1))
        self.up6 = StackDecoder(1024, 1024, 512, kernel_size=3)
        self.up5 = StackDecoder(512, 512, 256, kernel_size=3)
        self.up4 = StackDecoder(256, 256, 128, kernel_size=3)
        self.up3 = StackDecoder(128, 128, 64, kernel_size=3)
        self.up2 = StackDecoder(64, 64, 32, kernel_size=3)
        self.classify = nn.Conv2d(32, 1, kernel_size=1, padding=0, stride=1,
            bias=True)

    def forward(self, x):
        out = x
        down2, out = self.down2(out)
        down3, out = self.down3(out)
        down4, out = self.down4(out)
        down5, out = self.down5(out)
        down6, out = self.down6(out)
        pass
        out = self.center(out)
        out = self.up6(down6, out)
        out = self.up5(down5, out)
        out = self.up4(down4, out)
        out = self.up3(down3, out)
        out = self.up2(down2, out)
        out = self.classify(out)
        return out


class UNet128(nn.Module):

    def __init__(self, in_shape=(3, 128, 128), **kwargs):
        super(UNet128, self).__init__()
        C, H, W = in_shape
        self.down3 = StackEncoder(C, 128, kernel_size=3)
        self.down4 = StackEncoder(128, 256, kernel_size=3)
        self.down5 = StackEncoder(256, 512, kernel_size=3)
        self.down6 = StackEncoder(512, 1024, kernel_size=3)
        self.center = nn.Sequential(ConvBnRelu2d(1024, 1024, kernel_size=3,
            padding=1, stride=1))
        self.up6 = StackDecoder(1024, 1024, 512, kernel_size=3)
        self.up5 = StackDecoder(512, 512, 256, kernel_size=3)
        self.up4 = StackDecoder(256, 256, 128, kernel_size=3)
        self.up3 = StackDecoder(128, 128, 64, kernel_size=3)
        self.classify = nn.Conv2d(64, 1, kernel_size=1, padding=0, stride=1,
            bias=True)

    def forward(self, x):
        out = x
        down3, out = self.down3(out)
        down4, out = self.down4(out)
        down5, out = self.down5(out)
        down6, out = self.down6(out)
        pass
        out = self.center(out)
        out = self.up6(down6, out)
        out = self.up5(down5, out)
        out = self.up4(down4, out)
        out = self.up3(down3, out)
        out = self.classify(out)
        return out


class _ConvBNReLU(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, relu6=False, norm_layer=nn.
        BatchNorm2d, **kwargs):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
            stride, padding, dilation, groups, bias=False)
        self.bn = norm_layer(out_channels)
        self.relu = nn.ReLU6(True) if relu6 else nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class _ConvBNPReLU(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_ConvBNPReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
            stride, padding, dilation, groups, bias=False)
        self.bn = norm_layer(out_channels)
        self.prelu = nn.PReLU(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x


class _ConvBN(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_ConvBN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
            stride, padding, dilation, groups, bias=False)
        self.bn = norm_layer(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class _BNPReLU(nn.Module):

    def __init__(self, out_channels, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_BNPReLU, self).__init__()
        self.bn = norm_layer(out_channels)
        self.prelu = nn.PReLU(out_channels)

    def forward(self, x):
        x = self.bn(x)
        x = self.prelu(x)
        return x


class _PSPModule(nn.Module):

    def __init__(self, in_channels, sizes=(1, 2, 3, 6), **kwargs):
        super(_PSPModule, self).__init__()
        out_channels = int(in_channels / 4)
        self.avgpools = nn.ModuleList()
        self.convs = nn.ModuleList()
        for size in sizes:
            self.avgpool.append(nn.AdaptiveAvgPool2d(size))
            self.convs.append(_ConvBNReLU(in_channels, out_channels, 1, **
                kwargs))

    def forward(self, x):
        size = x.size()[2:]
        feats = [x]
        for avgpool, conv in enumerate(zip(self.avgpools, self.convs)):
            feats.append(F.interpolate(conv(avgpool(x)), size, mode=
                'bilinear', align_corners=True))
        return torch.cat(feats, dim=1)


class _DepthwiseConv(nn.Module):
    """conv_dw in MobileNet"""

    def __init__(self, in_channels, out_channels, stride, norm_layer=nn.
        BatchNorm2d, **kwargs):
        super(_DepthwiseConv, self).__init__()
        self.conv = nn.Sequential(_ConvBNReLU(in_channels, in_channels, 3,
            stride, 1, groups=in_channels, norm_layer=norm_layer),
            _ConvBNReLU(in_channels, out_channels, 1, norm_layer=norm_layer))

    def forward(self, x):
        return self.conv(x)


class InvertedResidual(nn.Module):

    def __init__(self, in_channels, out_channels, stride, expand_ratio,
        norm_layer=nn.BatchNorm2d, **kwargs):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]
        self.use_res_connect = stride == 1 and in_channels == out_channels
        layers = list()
        inter_channels = int(round(in_channels * expand_ratio))
        if expand_ratio != 1:
            layers.append(_ConvBNReLU(in_channels, inter_channels, 1, relu6
                =True, norm_layer=norm_layer))
        layers.extend([_ConvBNReLU(inter_channels, inter_channels, 3,
            stride, 1, groups=inter_channels, relu6=True, norm_layer=
            norm_layer), nn.Conv2d(inter_channels, out_channels, 1, bias=
            False), norm_layer(out_channels)])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class _DenseLayer(nn.Sequential):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate,
        dilation=1, norm_layer=nn.BatchNorm2d):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', norm_layer(num_input_features)),
        self.add_module('relu1', nn.ReLU(True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
            growth_rate, 1, 1, bias=False)),
        self.add_module('norm2', norm_layer(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate,
            growth_rate, 3, 1, dilation, dilation, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate,
        drop_rate, dilation=1, norm_layer=nn.BatchNorm2d):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate,
                growth_rate, bn_size, drop_rate, dilation, norm_layer)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):

    def __init__(self, num_input_features, num_output_features, norm_layer=
        nn.BatchNorm2d):
        super(_Transition, self).__init__()
        self.add_module('norm', norm_layer(num_input_features))
        self.add_module('relu', nn.ReLU(True))
        self.add_module('conv', nn.Conv2d(num_input_features,
            num_output_features, 1, 1, bias=False))
        self.add_module('pool', nn.AvgPool2d(2, 2))


class DenseNet(nn.Module):

    def __init__(self, growth_rate=12, block_config=(6, 12, 24, 16),
        num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000,
        norm_layer=nn.BatchNorm2d, **kwargs):
        super(DenseNet, self).__init__()
        self.features = nn.Sequential(OrderedDict([('conv0', nn.Conv2d(3,
            num_init_features, 7, 2, 3, bias=False)), ('norm0', norm_layer(
            num_init_features)), ('relu0', nn.ReLU(True)), ('pool0', nn.
            MaxPool2d(3, 2, 1))]))
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers, num_features, bn_size,
                growth_rate, drop_rate, norm_layer=norm_layer)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_features, num_features // 2,
                    norm_layer=norm_layer)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
        self.num_features = num_features
        self.features.add_module('norm5', norm_layer(num_features))
        self.classifier = nn.Linear(num_features, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        out = self.classifier(out)
        return out


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


_global_config['D'] = 4


def vgg16(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
    return model


class FCN32s(nn.Module):
    """There are some difference from original fcn"""

    def __init__(self, num_classes, backbone='vgg16', aux=False, pretrained
        =True, norm_layer=nn.BatchNorm2d, **kwargs):
        super(FCN32s, self).__init__()
        self.aux = aux
        if backbone == 'vgg16':
            self.pretrained = vgg16(pretrained=pretrained).features
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))
        self.head = _FCNHead(512, num_classes, norm_layer)
        if aux:
            self.auxlayer = _FCNHead(512, num_classes, norm_layer)
        self.__setattr__('exclusive', ['head', 'auxlayer'] if aux else ['head']
            )

    def forward(self, x):
        size = x.size()[2:]
        pool5 = self.pretrained(x)
        outputs = []
        out = self.head(pool5)
        out = F.interpolate(out, size, mode='bilinear', align_corners=True)
        outputs.append(out)
        if self.aux:
            auxout = self.auxlayer(pool5)
            auxout = F.interpolate(auxout, size, mode='bilinear',
                align_corners=True)
            outputs.append(auxout)
        return tuple(outputs)


class FCN16s(nn.Module):

    def __init__(self, num_classes, backbone='vgg16', aux=False, pretrained
        =True, norm_layer=nn.BatchNorm2d, **kwargs):
        super(FCN16s, self).__init__()
        self.aux = aux
        if backbone == 'vgg16':
            self.pretrained = vgg16(pretrained=pretrained).features
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))
        self.pool4 = nn.Sequential(*self.pretrained[:24])
        self.pool5 = nn.Sequential(*self.pretrained[24:])
        self.head = _FCNHead(512, num_classes, norm_layer)
        self.score_pool4 = nn.Conv2d(512, num_classes, 1)
        if aux:
            self.auxlayer = _FCNHead(512, num_classes, norm_layer)
        self.__setattr__('exclusive', ['head', 'score_pool4', 'auxlayer'] if
            aux else ['head', 'score_pool4'])

    def forward(self, x):
        pool4 = self.pool4(x)
        pool5 = self.pool5(pool4)
        outputs = []
        score_fr = self.head(pool5)
        score_pool4 = self.score_pool4(pool4)
        upscore2 = F.interpolate(score_fr, score_pool4.size()[2:], mode=
            'bilinear', align_corners=True)
        fuse_pool4 = upscore2 + score_pool4
        out = F.interpolate(fuse_pool4, x.size()[2:], mode='bilinear',
            align_corners=True)
        outputs.append(out)
        if self.aux:
            auxout = self.auxlayer(pool5)
            auxout = F.interpolate(auxout, x.size()[2:], mode='bilinear',
                align_corners=True)
            outputs.append(auxout)
        return tuple(outputs)


class FCN8s(nn.Module):

    def __init__(self, num_classes, backbone='vgg16', aux=False, pretrained
        =True, norm_layer=nn.BatchNorm2d, **kwargs):
        super(FCN8s, self).__init__()
        self.aux = aux
        if backbone == 'vgg16':
            self.pretrained = vgg16(pretrained=pretrained).features
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))
        self.pool3 = nn.Sequential(*self.pretrained[:17])
        self.pool4 = nn.Sequential(*self.pretrained[17:24])
        self.pool5 = nn.Sequential(*self.pretrained[24:])
        self.head = _FCNHead(512, num_classes, norm_layer)
        self.score_pool3 = nn.Conv2d(256, num_classes, 1)
        self.score_pool4 = nn.Conv2d(512, num_classes, 1)
        if aux:
            self.auxlayer = _FCNHead(512, num_classes, norm_layer)
        self.__setattr__('exclusive', ['head', 'score_pool3', 'score_pool4',
            'auxlayer'] if aux else ['head', 'score_pool3', 'score_pool4'])

    def forward(self, x):
        pool3 = self.pool3(x)
        pool4 = self.pool4(pool3)
        pool5 = self.pool5(pool4)
        outputs = []
        score_fr = self.head(pool5)
        score_pool4 = self.score_pool4(pool4)
        score_pool3 = self.score_pool3(pool3)
        upscore2 = F.interpolate(score_fr, score_pool4.size()[2:], mode=
            'bilinear', align_corners=True)
        fuse_pool4 = upscore2 + score_pool4
        upscore_pool4 = F.interpolate(fuse_pool4, score_pool3.size()[2:],
            mode='bilinear', align_corners=True)
        fuse_pool3 = upscore_pool4 + score_pool3
        out = F.interpolate(fuse_pool3, x.size()[2:], mode='bilinear',
            align_corners=True)
        outputs.append(out)
        if self.aux:
            auxout = self.auxlayer(pool5)
            auxout = F.interpolate(auxout, x.size()[2:], mode='bilinear',
                align_corners=True)
            outputs.append(auxout)
        return tuple(outputs)


class _FCNHead(nn.Module):

    def __init__(self, in_channels, channels, norm_layer=nn.BatchNorm2d, **
        kwargs):
        super(_FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3,
            padding=1, bias=False), norm_layer(inter_channels), nn.ReLU(
            inplace=True), nn.Dropout(0.1), nn.Conv2d(inter_channels,
            channels, 1))

    def forward(self, x):
        return self.block(x)


class SeparableConv2d(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1,
        dilation=1, bias=False, norm_layer=nn.BatchNorm2d):
        super(SeparableConv2d, self).__init__()
        self.conv = nn.Conv2d(inplanes, inplanes, kernel_size, stride,
            padding, dilation, groups=inplanes, bias=bias)
        self.bn = norm_layer(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, bias=bias)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class JPU(nn.Module):

    def __init__(self, in_channels, width=512, norm_layer=nn.BatchNorm2d,
        **kwargs):
        super(JPU, self).__init__()
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels[-1], width, 3,
            padding=1, bias=False), norm_layer(width), nn.ReLU(True))
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels[-2], width, 3,
            padding=1, bias=False), norm_layer(width), nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels[-3], width, 3,
            padding=1, bias=False), norm_layer(width), nn.ReLU(True))
        self.dilation1 = nn.Sequential(SeparableConv2d(3 * width, width, 3,
            padding=1, dilation=1, bias=False), norm_layer(width), nn.ReLU(
            True))
        self.dilation2 = nn.Sequential(SeparableConv2d(3 * width, width, 3,
            padding=2, dilation=2, bias=False), norm_layer(width), nn.ReLU(
            True))
        self.dilation3 = nn.Sequential(SeparableConv2d(3 * width, width, 3,
            padding=4, dilation=4, bias=False), norm_layer(width), nn.ReLU(
            True))
        self.dilation4 = nn.Sequential(SeparableConv2d(3 * width, width, 3,
            padding=8, dilation=8, bias=False), norm_layer(width), nn.ReLU(
            True))

    def forward(self, *inputs):
        feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2]), self.conv3
            (inputs[-3])]
        size = feats[-1].size()[2:]
        feats[-2] = F.interpolate(feats[-2], size, mode='bilinear',
            align_corners=True)
        feats[-3] = F.interpolate(feats[-3], size, mode='bilinear',
            align_corners=True)
        feat = torch.cat(feats, dim=1)
        feat = torch.cat([self.dilation1(feat), self.dilation2(feat), self.
            dilation3(feat), self.dilation4(feat)], dim=1)
        return inputs[0], inputs[1], inputs[2], feat


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
        norm_layer=nn.BatchNorm2d):
        super(BasicBlock, self).__init__()
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
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
        bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
        norm_layer=nn.BatchNorm2d):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
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

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=
        False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=
            norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
            norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
            norm_layer=norm_layer)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
            norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                    nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=nn.
        BatchNorm2d):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes *
                block.expansion, stride), norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
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
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class BasicBlockV1b(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=
        None, previous_dilation=1, norm_layer=nn.BatchNorm2d):
        super(BasicBlockV1b, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, dilation,
            dilation, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, previous_dilation,
            dilation=previous_dilation, bias=False)
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


class BottleneckV1b(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=
        None, previous_dilation=1, norm_layer=nn.BatchNorm2d):
        super(BottleneckV1b, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, dilation,
            dilation, bias=False)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(True)
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


class ResNetV1b(nn.Module):

    def __init__(self, block, layers, num_classes=1000, dilated=True,
        deep_stem=False, zero_init_residual=False, norm_layer=nn.
        BatchNorm2d, **kwargs):
        self.inplanes = 128 if deep_stem else 64
        super(ResNetV1b, self).__init__()
        if deep_stem:
            self.conv1 = nn.Sequential(nn.Conv2d(3, 64, 3, 2, 1, bias=False
                ), norm_layer(64), nn.ReLU(True), nn.Conv2d(64, 64, 3, 1, 1,
                bias=False), norm_layer(64), nn.ReLU(True), nn.Conv2d(64, 
                128, 3, 1, 1, bias=False))
        else:
            self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=
            norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
            norm_layer=norm_layer)
        if dilated:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                dilation=2, norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                dilation=4, norm_layer=norm_layer)
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                    nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BottleneckV1b):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlockV1b):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1,
        norm_layer=nn.BatchNorm2d):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, 1, stride, bias=False), norm_layer(planes *
                block.expansion))
        layers = []
        if dilation in (1, 2):
            layers.append(block(self.inplanes, planes, stride, dilation=1,
                downsample=downsample, previous_dilation=dilation))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, dilation=2,
                downsample=downsample, previous_dilation=dilation))
        else:
            raise RuntimeError('=> unknown dilation size: {}'.format(dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation,
                previous_dilation=dilation))
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
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def resnet101_v1s(pretrained=False, root='~/.torch/models', **kwargs):
    model = ResNetV1b(BottleneckV1b, [3, 4, 23, 3], deep_stem=True, **kwargs)
    if pretrained:
        from .model_store import get_resnet_file
        model.load_state_dict(torch.load(get_resnet_file('resnet101', root=
            root)), strict=False)
    return model


def resnet152_v1s(pretrained=False, root='~/.torch/models', **kwargs):
    model = ResNetV1b(BottleneckV1b, [3, 8, 36, 3], deep_stem=True, **kwargs)
    if pretrained:
        from .model_store import get_resnet_file
        model.load_state_dict(torch.load(get_resnet_file('resnet152', root=
            root)), strict=False)
    return model


def resnet50_v1s(pretrained=False, model_root='~/.torch/models', **kwargs):
    model = ResNetV1b(BottleneckV1b, [3, 4, 6, 3], deep_stem=True, **kwargs)
    if pretrained:
        from .model_store import get_resnet_file
        model.load_state_dict(torch.load(get_resnet_file('resnet50', root=
            model_root)), strict=False)
    return model


class SegBaseModel(nn.Module):
    """Base Model for Semantic Segmentation

    Parameters
    ----------
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    """

    def __init__(self, num_classes, pretrained=True, aux=False, backbone=
        'resnet101', **kwargs):
        super(SegBaseModel, self).__init__()
        self.aux = aux
        self.nclass = num_classes
        if backbone == 'resnet50':
            self.pretrained = resnet50_v1s(pretrained=pretrained, **kwargs)
        elif backbone == 'resnet101':
            self.pretrained = resnet101_v1s(pretrained=pretrained, **kwargs)
        elif backbone == 'resnet152':
            self.pretrained = resnet152_v1s(pretrained=pretrained, **kwargs)
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))

    def base_forward(self, x):
        """forwarding pre-trained network"""
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)
        c1 = self.pretrained.layer1(x)
        c2 = self.pretrained.layer2(c1)
        c3 = self.pretrained.layer3(c2)
        c4 = self.pretrained.layer4(c3)
        return c1, c2, c3, c4

    def evaluate(self, x):
        """evaluating network with inputs and targets"""
        return self.forward(x)[0]

    def demo(self, x):
        pred = self.forward(x)
        if self.aux:
            pred = pred[0]
        return pred


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(nn.Linear(512 * 7 * 7, 4096), nn.
            ReLU(True), nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(True),
            nn.Dropout(), nn.Linear(4096, num_classes))
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                    nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class _PositionAttentionModule(nn.Module):
    """ Position attention module"""

    def __init__(self, in_channels, **kwargs):
        super(_PositionAttentionModule, self).__init__()
        self.conv_b = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_c = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_d = nn.Conv2d(in_channels, in_channels, 1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_b = self.conv_b(x).view(batch_size, -1, height * width).permute(
            0, 2, 1)
        feat_c = self.conv_c(x).view(batch_size, -1, height * width)
        attention_s = self.softmax(torch.bmm(feat_b, feat_c))
        feat_d = self.conv_d(x).view(batch_size, -1, height * width)
        feat_e = torch.bmm(feat_d, attention_s.permute(0, 2, 1)).view(
            batch_size, -1, height, width)
        out = self.alpha * feat_e + x
        return out


class _ChannelAttentionModule(nn.Module):
    """Channel attention module"""

    def __init__(self, **kwargs):
        super(_ChannelAttentionModule, self).__init__()
        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_a = x.view(batch_size, -1, height * width)
        feat_a_transpose = x.view(batch_size, -1, height * width).permute(0,
            2, 1)
        attention = torch.bmm(feat_a, feat_a_transpose)
        attention_new = torch.max(attention, dim=-1, keepdim=True)[0
            ].expand_as(attention) - attention
        attention = self.softmax(attention_new)
        feat_e = torch.bmm(attention, feat_a).view(batch_size, -1, height,
            width)
        out = self.beta * feat_e + x
        return out


class _DAHead(nn.Module):

    def __init__(self, in_channels, nclass, aux=True, norm_layer=nn.
        BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_DAHead, self).__init__()
        self.aux = aux
        inter_channels = in_channels // 4
        self.conv_p1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels,
            3, padding=1, bias=False), norm_layer(inter_channels, **{} if 
            norm_kwargs is None else norm_kwargs), nn.ReLU(True))
        self.conv_c1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels,
            3, padding=1, bias=False), norm_layer(inter_channels, **{} if 
            norm_kwargs is None else norm_kwargs), nn.ReLU(True))
        self.pam = _PositionAttentionModule(inter_channels, **kwargs)
        self.cam = _ChannelAttentionModule(**kwargs)
        self.conv_p2 = nn.Sequential(nn.Conv2d(inter_channels,
            inter_channels, 3, padding=1, bias=False), norm_layer(
            inter_channels, **{} if norm_kwargs is None else norm_kwargs),
            nn.ReLU(True))
        self.conv_c2 = nn.Sequential(nn.Conv2d(inter_channels,
            inter_channels, 3, padding=1, bias=False), norm_layer(
            inter_channels, **{} if norm_kwargs is None else norm_kwargs),
            nn.ReLU(True))
        self.out = nn.Sequential(nn.Dropout(0.1), nn.Conv2d(inter_channels,
            nclass, 1))
        if aux:
            self.conv_p3 = nn.Sequential(nn.Dropout(0.1), nn.Conv2d(
                inter_channels, nclass, 1))
            self.conv_c3 = nn.Sequential(nn.Dropout(0.1), nn.Conv2d(
                inter_channels, nclass, 1))

    def forward(self, x):
        feat_p = self.conv_p1(x)
        feat_p = self.pam(feat_p)
        feat_p = self.conv_p2(feat_p)
        feat_c = self.conv_c1(x)
        feat_c = self.cam(feat_c)
        feat_c = self.conv_c2(feat_c)
        feat_fusion = feat_p + feat_c
        outputs = []
        fusion_out = self.out(feat_fusion)
        outputs.append(fusion_out)
        if self.aux:
            p_out = self.conv_p3(feat_p)
            c_out = self.conv_c3(feat_c)
            outputs.append(p_out)
            outputs.append(c_out)
        return tuple(outputs)


class Atrous_Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, rate=1, downsample=None):
        super(Atrous_Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            dilation=rate, padding=rate, bias=False)
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
        out += residual
        out = self.relu(out)
        return out


class Atrous_ResNet_features(nn.Module):

    def __init__(self, block, layers, pretrained=False):
        super(Atrous_ResNet_features, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, rate=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, rate=1)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, rate=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, rate=4)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        if pretrained:
            None
            resnet = models.resnet101(pretrained)
            self.conv1 = resnet.conv1
            self.bn1 = resnet.bn1
            self.layer1 = resnet.layer1
            self.layer2 = resnet.layer2

    def _make_layer(self, block, planes, blocks, stride=1, rate=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, rate, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride=1, rate=rate))
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
        return x


class Atrous_module(nn.Module):

    def __init__(self, inplanes, num_classes, rate):
        super(Atrous_module, self).__init__()
        planes = inplanes
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=3,
            stride=1, padding=rate, dilation=rate)
        self.fc1 = nn.Conv2d(planes, planes, kernel_size=1, stride=1)
        self.fc2 = nn.Conv2d(planes, num_classes, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class DeepLabv2_ASPP(nn.Module):
    """
    DeeplabV2 Resnet implementation with ASPP.
    """

    def __init__(self, num_classes, small=True, pretrained=False, **kwargs):
        super(DeepLabv2_ASPP, self).__init__()
        block = Atrous_Bottleneck
        self.resnet_features = Atrous_ResNet_features(block, [3, 4, 23, 3],
            pretrained)
        if small:
            rates = [2, 4, 8, 12]
        else:
            rates = [6, 12, 18, 24]
        self.aspp1 = Atrous_module(2048, num_classes, rate=rates[0])
        self.aspp2 = Atrous_module(2048, num_classes, rate=rates[1])
        self.aspp3 = Atrous_module(2048, num_classes, rate=rates[2])
        self.aspp4 = Atrous_module(2048, num_classes, rate=rates[3])

    def forward(self, x):
        x = self.resnet_features(x)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x = x1 + x2 + x3 + x4
        x = F.interpolate(x, scale_factor=8, mode='bilinear')
        return x


class DeepLabv2_FOV(nn.Module):
    """
        DeeplabV2 Resnet implementation with FOV.
        """

    def __init__(self, num_classes, pretrained=True, **kwargs):
        super(DeepLabv2_FOV, self).__init__()
        block = Atrous_Bottleneck
        self.resnet_features = Atrous_ResNet_features(block, [3, 4, 23, 3],
            pretrained)
        self.atrous = Atrous_module(2048, num_classes, rate=12)

    def forward(self, x):
        x = self.resnet_features(x)
        x = self.atrous(x)
        x = F.interpolate(x, scale_factor=8, mode='bilinear')
        return x


class Atrous_Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, rate=1, downsample=None):
        super(Atrous_Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            dilation=rate, padding=rate, bias=False)
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
        out += residual
        out = self.relu(out)
        return out


class Atrous_ResNet_features(nn.Module):

    def __init__(self, block, layers, pretrained=False):
        super(Atrous_ResNet_features, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, rate=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, rate=1)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, rate=1)
        self.layer4 = self._make_MG_unit(block, 512, stride=1, rate=2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        if pretrained:
            None
            resnet = models.resnet101(pretrained)
            self.conv1 = resnet.conv1
            self.bn1 = resnet.bn1
            self.layer1 = resnet.layer1
            self.layer2 = resnet.layer2

    def _make_layer(self, block, planes, blocks, stride=1, rate=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, rate, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks=[1, 2, 4], stride=1, rate=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, rate=blocks[0] *
            rate, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1, rate=
                blocks[i] * rate))
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
        return x


class Atrous_module(nn.Module):

    def __init__(self, inplanes, planes, rate):
        super(Atrous_module, self).__init__()
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=3,
            stride=1, padding=rate, dilation=rate)
        self.batch_norm = nn.BatchNorm2d(planes)

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.batch_norm(x)
        return x


class DeepLabv3(nn.Module):

    def __init__(self, num_classes, small=True, pretrained=True, **kwargs):
        super(DeepLabv3, self).__init__()
        block = Atrous_Bottleneck
        self.resnet_features = Atrous_ResNet_features(block, [3, 4, 23],
            pretrained)
        rates = [1, 6, 12, 18]
        self.aspp1 = Atrous_module(2048, 256, rate=rates[0])
        self.aspp2 = Atrous_module(2048, 256, rate=rates[1])
        self.aspp3 = Atrous_module(2048, 256, rate=rates[2])
        self.aspp4 = Atrous_module(2048, 256, rate=rates[3])
        self.image_pool = nn.Sequential(nn.AdaptiveMaxPool2d(1), nn.Conv2d(
            2048, 256, kernel_size=1))
        self.fc1 = nn.Sequential(nn.Conv2d(1280, 256, kernel_size=1), nn.
            BatchNorm2d(256))
        self.fc2 = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.resnet_features(x)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.image_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='nearest')
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = F.interpolate(x, scale_factor=(16, 16), mode='bilinear')
        return x


class Atrous_Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, rate=1, downsample=None):
        super(Atrous_Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            dilation=rate, padding=rate, bias=False)
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
        out += residual
        out = self.relu(out)
        return out


class Atrous_ResNet_features(nn.Module):

    def __init__(self, block, layers, pretrained=False):
        super(Atrous_ResNet_features, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, rate=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, rate=1)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, rate=1)
        self.layer4 = self._make_MG_unit(block, 512, stride=1, rate=2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        if pretrained:
            None
            resnet = models.resnet101(pretrained)
            self.conv1 = resnet.conv1
            self.bn1 = resnet.bn1
            self.layer1 = resnet.layer1
            self.layer2 = resnet.layer2

    def _make_layer(self, block, planes, blocks, stride=1, rate=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, rate, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks=[1, 2, 4], stride=1, rate=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, rate=blocks[0] *
            rate, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1, rate=
                blocks[i] * rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        conv2 = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x, conv2


class Atrous_module(nn.Module):

    def __init__(self, inplanes, planes, rate):
        super(Atrous_module, self).__init__()
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=3,
            stride=1, padding=rate, dilation=rate)
        self.batch_norm = nn.BatchNorm2d(planes)

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.batch_norm(x)
        return x


class DeepLabv3_plus(nn.Module):

    def __init__(self, num_classes, small=True, pretrained=True, **kwargs):
        super(DeepLabv3_plus, self).__init__()
        block = Atrous_Bottleneck
        self.resnet_features = Atrous_ResNet_features(block, [3, 4, 23],
            pretrained)
        rates = [1, 6, 12, 18]
        self.aspp1 = Atrous_module(2048, 256, rate=rates[0])
        self.aspp2 = Atrous_module(2048, 256, rate=rates[1])
        self.aspp3 = Atrous_module(2048, 256, rate=rates[2])
        self.aspp4 = Atrous_module(2048, 256, rate=rates[3])
        self.image_pool = nn.Sequential(nn.AdaptiveMaxPool2d(1), nn.Conv2d(
            2048, 256, kernel_size=1))
        self.fc1 = nn.Sequential(nn.Conv2d(1280, 256, kernel_size=1), nn.
            BatchNorm2d(256))
        self.reduce_conv2 = nn.Sequential(nn.Conv2d(256, 48, kernel_size=1),
            nn.BatchNorm2d(48))
        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3,
            stride=1, padding=1), nn.BatchNorm2d(256), nn.Conv2d(256, 256,
            kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.
            Conv2d(256, num_classes, kernel_size=1, stride=1))

    def forward(self, x):
        x, conv2 = self.resnet_features(x)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.image_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='nearest')
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.fc1(x)
        x = F.interpolate(x, scale_factor=(4, 4), mode='bilinear')
        low_lebel_features = self.reduce_conv2(conv2)
        x = torch.cat((x, low_lebel_features), dim=1)
        x = self.last_conv(x)
        x = F.interpolate(x, scale_factor=(4, 4), mode='bilinear')
        return x


class DilatedDenseNet(DenseNet):

    def __init__(self, growth_rate=12, block_config=(6, 12, 24, 16),
        num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000,
        dilate_scale=8, norm_layer=nn.BatchNorm2d, **kwargs):
        super(DilatedDenseNet, self).__init__(growth_rate, block_config,
            num_init_features, bn_size, drop_rate, num_classes, norm_layer)
        assert dilate_scale == 8 or dilate_scale == 16, 'dilate_scale can only set as 8 or 16'
        from functools import partial
        if dilate_scale == 8:
            self.features.denseblock3.apply(partial(self._conv_dilate,
                dilate=2))
            self.features.denseblock4.apply(partial(self._conv_dilate,
                dilate=4))
            del self.features.transition2.pool
            del self.features.transition3.pool
        elif dilate_scale == 16:
            self.features.denseblock4.apply(partial(self._conv_dilate,
                dilate=2))
            del self.features.transition3.pool

    def _conv_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.kernel_size == (3, 3):
                m.padding = dilate, dilate
                m.dilation = dilate, dilate


densenet_spec = {(121): (64, 32, [6, 12, 24, 16]), (161): (96, 48, [6, 12, 
    36, 24]), (169): (64, 32, [6, 12, 32, 32]), (201): (64, 32, [6, 12, 48,
    32])}


def get_dilated_densenet(num_layers, dilate_scale, pretrained=False, **kwargs):
    num_init_features, growth_rate, block_config = densenet_spec[num_layers]
    model = DilatedDenseNet(growth_rate, block_config, num_init_features,
        dilate_scale=dilate_scale)
    if pretrained:
        pattern = re.compile(
            '^(.*denselayer\\d+\\.(?:norm|relu|conv))\\.((?:[12])\\.(?:weight|bias|running_mean|running_var))$'
            )
        state_dict = model_zoo.load_url(model_urls['densenet%d' % num_layers])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    return model


def dilated_densenet121(dilate_scale, **kwargs):
    return get_dilated_densenet(121, dilate_scale, **kwargs)


def dilated_densenet161(dilate_scale, **kwargs):
    return get_dilated_densenet(161, dilate_scale, **kwargs)


def dilated_densenet169(dilate_scale, **kwargs):
    return get_dilated_densenet(169, dilate_scale, **kwargs)


def dilated_densenet201(dilate_scale, **kwargs):
    return get_dilated_densenet(201, dilate_scale, **kwargs)


class DenseASPP(nn.Module):

    def __init__(self, num_classes, pretrained=True, backbone='densenet161',
        aux=False, dilate_scale=8, **kwargs):
        super(DenseASPP, self).__init__()
        self.nclass = num_classes
        self.aux = aux
        self.dilate_scale = dilate_scale
        if backbone == 'densenet121':
            self.pretrained = dilated_densenet121(dilate_scale, pretrained=
                pretrained, **kwargs)
        elif backbone == 'densenet161':
            self.pretrained = dilated_densenet161(dilate_scale, pretrained=
                pretrained, **kwargs)
        elif backbone == 'densenet169':
            self.pretrained = dilated_densenet169(dilate_scale, pretrained=
                pretrained, **kwargs)
        elif backbone == 'densenet201':
            self.pretrained = dilated_densenet201(dilate_scale, pretrained=
                pretrained, **kwargs)
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))
        in_channels = self.pretrained.num_features
        self.head = _DenseASPPHead(in_channels, num_classes, **kwargs)
        if aux:
            self.auxlayer = _FCNHead(in_channels, num_classes, **kwargs)
        self.__setattr__('exclusive', ['head', 'auxlayer'] if aux else ['head']
            )

    def forward(self, x):
        size = x.size()[2:]
        features = self.pretrained.features(x)
        if self.dilate_scale > 8:
            features = F.interpolate(features, scale_factor=2, mode=
                'bilinear', align_corners=True)
        outputs = []
        x = self.head(features)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        outputs.append(x)
        if self.aux:
            auxout = self.auxlayer(features)
            auxout = F.interpolate(auxout, size, mode='bilinear',
                align_corners=True)
            outputs.append(auxout)
            return tuple(outputs)
        else:
            return outputs[0]


class _DenseASPPHead(nn.Module):

    def __init__(self, in_channels, nclass, norm_layer=nn.BatchNorm2d,
        norm_kwargs=None, **kwargs):
        super(_DenseASPPHead, self).__init__()
        self.dense_aspp_block = _DenseASPPBlock(in_channels, 256, 64,
            norm_layer, norm_kwargs)
        self.block = nn.Sequential(nn.Dropout(0.1), nn.Conv2d(in_channels +
            5 * 64, nclass, 1))

    def forward(self, x):
        x = self.dense_aspp_block(x)
        return self.block(x)


class _DenseASPPConv(nn.Sequential):

    def __init__(self, in_channels, inter_channels, out_channels,
        atrous_rate, drop_rate=0.1, norm_layer=nn.BatchNorm2d, norm_kwargs=None
        ):
        super(_DenseASPPConv, self).__init__()
        self.add_module('conv1', nn.Conv2d(in_channels, inter_channels, 1)),
        self.add_module('bn1', norm_layer(inter_channels, **{} if 
            norm_kwargs is None else norm_kwargs)),
        self.add_module('relu1', nn.ReLU(True)),
        self.add_module('conv2', nn.Conv2d(inter_channels, out_channels, 3,
            dilation=atrous_rate, padding=atrous_rate)),
        self.add_module('bn2', norm_layer(out_channels, **{} if norm_kwargs is
            None else norm_kwargs)),
        self.add_module('relu2', nn.ReLU(True)),
        self.drop_rate = drop_rate

    def forward(self, x):
        features = super(_DenseASPPConv, self).forward(x)
        if self.drop_rate > 0:
            features = F.dropout(features, p=self.drop_rate, training=self.
                training)
        return features


class _DenseASPPBlock(nn.Module):

    def __init__(self, in_channels, inter_channels1, inter_channels2,
        norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super(_DenseASPPBlock, self).__init__()
        self.aspp_3 = _DenseASPPConv(in_channels, inter_channels1,
            inter_channels2, 3, 0.1, norm_layer, norm_kwargs)
        self.aspp_6 = _DenseASPPConv(in_channels + inter_channels2 * 1,
            inter_channels1, inter_channels2, 6, 0.1, norm_layer, norm_kwargs)
        self.aspp_12 = _DenseASPPConv(in_channels + inter_channels2 * 2,
            inter_channels1, inter_channels2, 12, 0.1, norm_layer, norm_kwargs)
        self.aspp_18 = _DenseASPPConv(in_channels + inter_channels2 * 3,
            inter_channels1, inter_channels2, 18, 0.1, norm_layer, norm_kwargs)
        self.aspp_24 = _DenseASPPConv(in_channels + inter_channels2 * 4,
            inter_channels1, inter_channels2, 24, 0.1, norm_layer, norm_kwargs)

    def forward(self, x):
        aspp3 = self.aspp_3(x)
        x = torch.cat([aspp3, x], dim=1)
        aspp6 = self.aspp_6(x)
        x = torch.cat([aspp6, x], dim=1)
        aspp12 = self.aspp_12(x)
        x = torch.cat([aspp12, x], dim=1)
        aspp18 = self.aspp_18(x)
        x = torch.cat([aspp18, x], dim=1)
        aspp24 = self.aspp_24(x)
        x = torch.cat([aspp24, x], dim=1)
        return x


BatchNorm = nn.BatchNorm2d


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
        dilation=(1, 1), residual=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, padding=dilation[0],
            dilation=dilation[0])
        self.bn1 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, padding=dilation[1], dilation=
            dilation[1])
        self.bn2 = BatchNorm(planes)
        self.downsample = downsample
        self.stride = stride
        self.residual = residual

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        if self.residual:
            out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
        dilation=(1, 1), residual=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=dilation[1], bias=False, dilation=dilation[1])
        self.bn2 = BatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * 4)
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
        out += residual
        out = self.relu(out)
        return out


class DRN(nn.Module):

    def __init__(self, block, layers, num_classes=1000, channels=(16, 32, 
        64, 128, 256, 512, 512, 512), out_map=False, out_middle=False,
        pool_size=28, arch='D'):
        super(DRN, self).__init__()
        self.inplanes = channels[0]
        self.out_map = out_map
        self.out_dim = channels[-1]
        self.out_middle = out_middle
        self.arch = arch
        if arch == 'C':
            self.conv1 = nn.Conv2d(3, channels[0], kernel_size=7, stride=1,
                padding=3, bias=False)
            self.bn1 = BatchNorm(channels[0])
            self.relu = nn.ReLU(inplace=True)
            self.layer1 = self._make_layer(BasicBlock, channels[0], layers[
                0], stride=1)
            self.layer2 = self._make_layer(BasicBlock, channels[1], layers[
                1], stride=2)
        elif arch == 'D':
            self.layer0 = nn.Sequential(nn.Conv2d(3, channels[0],
                kernel_size=7, stride=1, padding=3, bias=False), BatchNorm(
                channels[0]), nn.ReLU(inplace=True))
            self.layer1 = self._make_conv_layers(channels[0], layers[0],
                stride=1)
            self.layer2 = self._make_conv_layers(channels[1], layers[1],
                stride=2)
        self.layer3 = self._make_layer(block, channels[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, channels[3], layers[3], stride=2)
        self.layer5 = self._make_layer(block, channels[4], layers[4],
            dilation=2, new_level=False)
        self.layer6 = None if layers[5] == 0 else self._make_layer(block,
            channels[5], layers[5], dilation=4, new_level=False)
        if arch == 'C':
            self.layer7 = None if layers[6] == 0 else self._make_layer(
                BasicBlock, channels[6], layers[6], dilation=2, new_level=
                False, residual=False)
            self.layer8 = None if layers[7] == 0 else self._make_layer(
                BasicBlock, channels[7], layers[7], dilation=1, new_level=
                False, residual=False)
        elif arch == 'D':
            self.layer7 = None if layers[6] == 0 else self._make_conv_layers(
                channels[6], layers[6], dilation=2)
            self.layer8 = None if layers[7] == 0 else self._make_conv_layers(
                channels[7], layers[7], dilation=1)
        if num_classes > 0:
            self.avgpool = nn.AvgPool2d(pool_size)
            self.fc = nn.Conv2d(self.out_dim, num_classes, kernel_size=1,
                stride=1, padding=0, bias=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, BatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1,
        new_level=True, residual=True):
        assert dilation == 1 or dilation % 2 == 0
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion))
        layers = list()
        layers.append(block(self.inplanes, planes, stride, downsample,
            dilation=(1, 1) if dilation == 1 else (dilation // 2 if
            new_level else dilation, dilation), residual=residual))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, residual=residual,
                dilation=(dilation, dilation)))
        return nn.Sequential(*layers)

    def _make_conv_layers(self, channels, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([nn.Conv2d(self.inplanes, channels, kernel_size=
                3, stride=stride if i == 0 else 1, padding=dilation, bias=
                False, dilation=dilation), BatchNorm(channels), nn.ReLU(
                inplace=True)])
            self.inplanes = channels
        return nn.Sequential(*modules)

    def forward(self, x):
        y = list()
        if self.arch == 'C':
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        elif self.arch == 'D':
            x = self.layer0(x)
        x = self.layer1(x)
        y.append(x)
        x = self.layer2(x)
        y.append(x)
        x = self.layer3(x)
        y.append(x)
        x = self.layer4(x)
        y.append(x)
        x = self.layer5(x)
        y.append(x)
        if self.layer6 is not None:
            x = self.layer6(x)
            y.append(x)
        if self.layer7 is not None:
            x = self.layer7(x)
            y.append(x)
        if self.layer8 is not None:
            x = self.layer8(x)
            y.append(x)
        if self.out_map:
            x = self.fc(x)
        else:
            x = self.avgpool(x)
            x = self.fc(x)
            x = x.view(x.size(0), -1)
        if self.out_middle:
            return x, y
        else:
            return x


def drn_c_42(pretrained=False, **kwargs):
    model = DRN(BasicBlock, [1, 1, 3, 4, 6, 3, 1, 1], arch='C', **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['drn-c-42']))
    return model


def drn_c_58(pretrained=False, **kwargs):
    model = DRN(Bottleneck, [1, 1, 3, 4, 6, 3, 1, 1], arch='C', **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['drn-c-58']))
    return model


def drn_d_105(pretrained=False, **kwargs):
    model = DRN(Bottleneck, [1, 1, 3, 4, 23, 3, 1, 1], arch='D', **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['drn-d-105']))
    return model


def drn_d_38(pretrained=False, **kwargs):
    model = DRN(BasicBlock, [1, 1, 3, 4, 6, 3, 1, 1], arch='D', **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['drn-d-38']))
    return model


def drn_d_54(pretrained=False, **kwargs):
    model = DRN(Bottleneck, [1, 1, 3, 4, 6, 3, 1, 1], arch='D', **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['drn-d-54']))
    return model


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = (1 - math.fabs(i / f - c)) * (1 - math.fabs(j /
                f - c))
    for c in range(1, w.size(0)):
        w[(c), (0), :, :] = w[(0), (0), :, :]


class DRNSeg(nn.Module):

    def __init__(self, num_classes, pretrained=True, model_name=None,
        use_torch_up=False, **kwargs):
        super(DRNSeg, self).__init__()
        if model_name == 'DRN_C_42':
            model = drn_c_42(pretrained=pretrained, num_classes=1000)
        elif model_name == 'DRN_C_58':
            model = drn_c_58(pretrained=pretrained, num_classes=1000)
        elif model_name == 'DRN_D_38':
            model = drn_d_38(pretrained=pretrained, num_classes=1000)
        elif model_name == 'DRN_D_54':
            model = drn_d_54(pretrained=pretrained, num_classes=1000)
        elif model_name == 'DRN_D_105':
            model = drn_d_105(pretrained=pretrained, num_classes=1000)
        else:
            raise Exception(
                'model_name must be supplied to DRNSeg constructor.')
        self.base = nn.Sequential(*list(model.children())[:-2])
        self.seg = nn.Conv2d(model.out_dim, num_classes, kernel_size=1,
            bias=True)
        self.softmax = nn.LogSoftmax()
        m = self.seg
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2.0 / n))
        m.bias.data.zero_()
        self.use_torch_up = use_torch_up
        up = nn.ConvTranspose2d(num_classes, num_classes, 16, stride=8,
            padding=4, output_padding=0, groups=num_classes, bias=False)
        fill_up_weights(up)
        up.weight.requires_grad = False
        self.up = up

    def forward(self, x):
        base = self.base(x)
        final = self.seg(base)
        if self.use_torch_up:
            return F.interpolate(final, x.size()[2:], mode='bilinear')
        else:
            return self.up(final)

    def optim_parameters(self, memo=None):
        for param in self.base.parameters():
            yield param
        for param in self.seg.parameters():
            yield param


class _DenseUpsamplingConvModule(nn.Module):

    def __init__(self, down_factor, in_dim, num_classes):
        super(_DenseUpsamplingConvModule, self).__init__()
        upsample_dim = down_factor ** 2 * num_classes
        self.conv = nn.Conv2d(in_dim, upsample_dim, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(upsample_dim)
        self.relu = nn.ReLU(inplace=True)
        self.pixel_shuffle = nn.PixelShuffle(down_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pixel_shuffle(x)
        return x


root = '/models/pytorch'


class ResNetDUCHDC(nn.Module):

    def __init__(self, num_classes, pretrained=True, **kwargs):
        super(ResNetDUCHDC, self).__init__()
        resnet = models.resnet152()
        if pretrained:
            resnet.load_state_dict(torch.load(res152_path))
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu,
            resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        for n, m in self.layer3.named_modules():
            if 'conv2' in n or 'downsample.0' in n:
                m.stride = 1, 1
        for n, m in self.layer4.named_modules():
            if 'conv2' in n or 'downsample.0' in n:
                m.stride = 1, 1
        layer3_group_config = [1, 2, 5, 9]
        for idx in range(len(self.layer3)):
            self.layer3[idx].conv2.dilation = layer3_group_config[idx % 4
                ], layer3_group_config[idx % 4]
            self.layer3[idx].conv2.padding = layer3_group_config[idx % 4
                ], layer3_group_config[idx % 4]
        layer4_group_config = [5, 9, 17]
        for idx in range(len(self.layer4)):
            self.layer4[idx].conv2.dilation = layer4_group_config[idx
                ], layer4_group_config[idx]
            self.layer4[idx].conv2.padding = layer4_group_config[idx
                ], layer4_group_config[idx]
        self.duc = _DenseUpsamplingConvModule(8, 2048, num_classes)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.duc(x)
        return x


class FeatureFused(nn.Module):
    """Module for fused features"""

    def __init__(self, inter_channels=48, norm_layer=nn.BatchNorm2d, **kwargs):
        super(FeatureFused, self).__init__()
        self.conv2 = nn.Sequential(nn.Conv2d(512, inter_channels, 1, bias=
            False), norm_layer(inter_channels), nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(1024, inter_channels, 1, bias=
            False), norm_layer(inter_channels), nn.ReLU(True))

    def forward(self, c2, c3, c4):
        size = c4.size()[2:]
        c2 = self.conv2(F.interpolate(c2, size, mode='bilinear',
            align_corners=True))
        c3 = self.conv3(F.interpolate(c3, size, mode='bilinear',
            align_corners=True))
        fused_feature = torch.cat([c4, c3, c2], dim=1)
        return fused_feature


class _DUHead(nn.Module):

    def __init__(self, in_channels, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_DUHead, self).__init__()
        self.fuse = FeatureFused(norm_layer=norm_layer, **kwargs)
        self.block = nn.Sequential(nn.Conv2d(in_channels, 256, 3, padding=1,
            bias=False), norm_layer(256), nn.ReLU(True), nn.Conv2d(256, 256,
            3, padding=1, bias=False), norm_layer(256), nn.ReLU(True))

    def forward(self, c2, c3, c4):
        fused_feature = self.fuse(c2, c3, c4)
        out = self.block(fused_feature)
        return out


class DUpsampling(nn.Module):
    """DUsampling module"""

    def __init__(self, in_channels, out_channels, scale_factor=2, **kwargs):
        super(DUpsampling, self).__init__()
        self.scale_factor = scale_factor
        self.conv_w = nn.Conv2d(in_channels, out_channels * scale_factor *
            scale_factor, 1, bias=False)

    def forward(self, x):
        x = self.conv_w(x)
        n, c, h, w = x.size()
        x = x.permute(0, 3, 2, 1).contiguous()
        x = x.view(n, w, h * self.scale_factor, c // self.scale_factor)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(n, h * self.scale_factor, w * self.scale_factor, c // (
            self.scale_factor * self.scale_factor))
        x = x.permute(0, 3, 1, 2)
        return x


class InitialBlock(nn.Module):
    """The initial block is composed of two branches:
    1. a main branch which performs a regular convolution with stride 2;
    2. an extension branch which performs max-pooling.

    Doing both operations in parallel and concatenating their results
    allows for efficient downsampling and expansion. The main branch
    outputs 13 feature maps while the extension branch outputs 3, for a
    total of 16 feature maps after concatenation.

    Keyword arguments:
    - in_channels (int): the number of input channels.
    - out_channels (int): the number output channels.
    - kernel_size (int, optional): the kernel size of the filters used in
    the convolution layer. Default: 3.
    - padding (int, optional): zero-padding added to both sides of the
    input. Default: 0.
    - bias (bool, optional): Adds a learnable bias to the output if
    ``True``. Default: False.
    - relu (bool, optional): When ``True`` ReLU is used as the activation
    function; otherwise, PReLU is used. Default: True.

    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=0,
        bias=False, relu=True):
        super().__init__()
        if relu:
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()
        self.main_branch = nn.Conv2d(in_channels, out_channels - 3,
            kernel_size=kernel_size, stride=2, padding=padding, bias=bias)
        self.ext_branch = nn.MaxPool2d(kernel_size, stride=2, padding=padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.out_prelu = activation

    def forward(self, x):
        main = self.main_branch(x)
        ext = self.ext_branch(x)
        out = torch.cat((main, ext), 1)
        out = self.batch_norm(out)
        return self.out_prelu(out)


class RegularBottleneck(nn.Module):
    """Regular bottlenecks are the main building block of ENet.
    Main branch:
    1. Shortcut connection.

    Extension branch:
    1. 1x1 convolution which decreases the number of channels by
    ``internal_ratio``, also called a projection;
    2. regular, dilated or asymmetric convolution;
    3. 1x1 convolution which increases the number of channels back to
    ``channels``, also called an expansion;
    4. dropout as a regularizer.

    Keyword arguments:
    - channels (int): the number of input and output channels.
    - internal_ratio (int, optional): a scale factor applied to
    ``channels`` used to compute the number of
    channels after the projection. eg. given ``channels`` equal to 128 and
    internal_ratio equal to 2 the number of channels after the projection
    is 64. Default: 4.
    - kernel_size (int, optional): the kernel size of the filters used in
    the convolution layer described above in item 2 of the extension
    branch. Default: 3.
    - padding (int, optional): zero-padding added to both sides of the
    input. Default: 0.
    - dilation (int, optional): spacing between kernel elements for the
    convolution described in item 2 of the extension branch. Default: 1.
    asymmetric (bool, optional): flags if the convolution described in
    item 2 of the extension branch is asymmetric or not. Default: False.
    - dropout_prob (float, optional): probability of an element to be
    zeroed. Default: 0 (no dropout).
    - bias (bool, optional): Adds a learnable bias to the output if
    ``True``. Default: False.
    - relu (bool, optional): When ``True`` ReLU is used as the activation
    function; otherwise, PReLU is used. Default: True.

    """

    def __init__(self, channels, internal_ratio=4, kernel_size=3, padding=0,
        dilation=1, asymmetric=False, dropout_prob=0, bias=False, relu=True):
        super().__init__()
        if internal_ratio <= 1 or internal_ratio > channels:
            raise RuntimeError(
                'Value out of range. Expected value in the interval [1, {0}], got internal_scale={1}.'
                .format(channels, internal_ratio))
        internal_channels = channels // internal_ratio
        if relu:
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()
        self.ext_conv1 = nn.Sequential(nn.Conv2d(channels,
            internal_channels, kernel_size=1, stride=1, bias=bias), nn.
            BatchNorm2d(internal_channels), activation)
        if asymmetric:
            self.ext_conv2 = nn.Sequential(nn.Conv2d(internal_channels,
                internal_channels, kernel_size=(kernel_size, 1), stride=1,
                padding=(padding, 0), dilation=dilation, bias=bias), nn.
                BatchNorm2d(internal_channels), activation, nn.Conv2d(
                internal_channels, internal_channels, kernel_size=(1,
                kernel_size), stride=1, padding=(0, padding), dilation=
                dilation, bias=bias), nn.BatchNorm2d(internal_channels),
                activation)
        else:
            self.ext_conv2 = nn.Sequential(nn.Conv2d(internal_channels,
                internal_channels, kernel_size=kernel_size, stride=1,
                padding=padding, dilation=dilation, bias=bias), nn.
                BatchNorm2d(internal_channels), activation)
        self.ext_conv3 = nn.Sequential(nn.Conv2d(internal_channels,
            channels, kernel_size=1, stride=1, bias=bias), nn.BatchNorm2d(
            channels), activation)
        self.ext_regul = nn.Dropout2d(p=dropout_prob)
        self.out_prelu = activation

    def forward(self, x):
        main = x
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)
        out = main + ext
        return self.out_prelu(out)


class DownsamplingBottleneck(nn.Module):
    """Downsampling bottlenecks further downsample the feature map size.

    Main branch:
    1. max pooling with stride 2; indices are saved to be used for
    unpooling later.

    Extension branch:
    1. 2x2 convolution with stride 2 that decreases the number of channels
    by ``internal_ratio``, also called a projection;
    2. regular convolution (by default, 3x3);
    3. 1x1 convolution which increases the number of channels to
    ``out_channels``, also called an expansion;
    4. dropout as a regularizer.

    Keyword arguments:
    - in_channels (int): the number of input channels.
    - out_channels (int): the number of output channels.
    - internal_ratio (int, optional): a scale factor applied to ``channels``
    used to compute the number of channels after the projection. eg. given
    ``channels`` equal to 128 and internal_ratio equal to 2 the number of
    channels after the projection is 64. Default: 4.
    - kernel_size (int, optional): the kernel size of the filters used in
    the convolution layer described above in item 2 of the extension branch.
    Default: 3.
    - padding (int, optional): zero-padding added to both sides of the
    input. Default: 0.
    - dilation (int, optional): spacing between kernel elements for the
    convolution described in item 2 of the extension branch. Default: 1.
    - asymmetric (bool, optional): flags if the convolution described in
    item 2 of the extension branch is asymmetric or not. Default: False.
    - return_indices (bool, optional):  if ``True``, will return the max
    indices along with the outputs. Useful when unpooling later.
    - dropout_prob (float, optional): probability of an element to be
    zeroed. Default: 0 (no dropout).
    - bias (bool, optional): Adds a learnable bias to the output if
    ``True``. Default: False.
    - relu (bool, optional): When ``True`` ReLU is used as the activation
    function; otherwise, PReLU is used. Default: True.

    """

    def __init__(self, in_channels, out_channels, internal_ratio=4,
        kernel_size=3, padding=0, return_indices=False, dropout_prob=0,
        bias=False, relu=True):
        super().__init__()
        self.return_indices = return_indices
        if internal_ratio <= 1 or internal_ratio > in_channels:
            raise RuntimeError(
                'Value out of range. Expected value in the interval [1, {0}], got internal_scale={1}. '
                .format(in_channels, internal_ratio))
        internal_channels = in_channels // internal_ratio
        if relu:
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()
        self.main_max1 = nn.MaxPool2d(kernel_size, stride=2, padding=
            padding, return_indices=return_indices)
        self.ext_conv1 = nn.Sequential(nn.Conv2d(in_channels,
            internal_channels, kernel_size=2, stride=2, bias=bias), nn.
            BatchNorm2d(internal_channels), activation)
        self.ext_conv2 = nn.Sequential(nn.Conv2d(internal_channels,
            internal_channels, kernel_size=kernel_size, stride=1, padding=
            padding, bias=bias), nn.BatchNorm2d(internal_channels), activation)
        self.ext_conv3 = nn.Sequential(nn.Conv2d(internal_channels,
            out_channels, kernel_size=1, stride=1, bias=bias), nn.
            BatchNorm2d(out_channels), activation)
        self.ext_regul = nn.Dropout2d(p=dropout_prob)
        self.out_prelu = activation

    def forward(self, x):
        if self.return_indices:
            main, max_indices = self.main_max1(x)
        else:
            main = self.main_max1(x)
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)
        n, ch_ext, h, w = ext.size()
        ch_main = main.size()[1]
        padding = torch.zeros(n, ch_ext - ch_main, h, w)
        if main.is_cuda:
            padding = padding
        main = torch.cat((main, padding), 1)
        out = main + ext
        return self.out_prelu(out), max_indices


class UpsamplingBottleneck(nn.Module):
    """The upsampling bottlenecks upsample the feature map resolution using max
    pooling indices stored from the corresponding downsampling bottleneck.

    Main branch:
    1. 1x1 convolution with stride 1 that decreases the number of channels by
    ``internal_ratio``, also called a projection;
    2. max unpool layer using the max pool indices from the corresponding
    downsampling max pool layer.

    Extension branch:
    1. 1x1 convolution with stride 1 that decreases the number of channels by
    ``internal_ratio``, also called a projection;
    2. transposed convolution (by default, 3x3);
    3. 1x1 convolution which increases the number of channels to
    ``out_channels``, also called an expansion;
    4. dropout as a regularizer.

    Keyword arguments:
    - in_channels (int): the number of input channels.
    - out_channels (int): the number of output channels.
    - internal_ratio (int, optional): a scale factor applied to ``in_channels``
     used to compute the number of channels after the projection. eg. given
     ``in_channels`` equal to 128 and ``internal_ratio`` equal to 2 the number
     of channels after the projection is 64. Default: 4.
    - kernel_size (int, optional): the kernel size of the filters used in the
    convolution layer described above in item 2 of the extension branch.
    Default: 3.
    - padding (int, optional): zero-padding added to both sides of the input.
    Default: 0.
    - dropout_prob (float, optional): probability of an element to be zeroed.
    Default: 0 (no dropout).
    - bias (bool, optional): Adds a learnable bias to the output if ``True``.
    Default: False.
    - relu (bool, optional): When ``True`` ReLU is used as the activation
    function; otherwise, PReLU is used. Default: True.

    """

    def __init__(self, in_channels, out_channels, internal_ratio=4,
        kernel_size=3, padding=0, dropout_prob=0, bias=False, relu=True):
        super().__init__()
        if internal_ratio <= 1 or internal_ratio > in_channels:
            raise RuntimeError(
                'Value out of range. Expected value in the interval [1, {0}], got internal_scale={1}. '
                .format(in_channels, internal_ratio))
        internal_channels = in_channels // internal_ratio
        if relu:
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()
        self.main_conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels,
            kernel_size=1, bias=bias), nn.BatchNorm2d(out_channels))
        self.main_unpool1 = nn.MaxUnpool2d(kernel_size=2)
        self.ext_conv1 = nn.Sequential(nn.Conv2d(in_channels,
            internal_channels, kernel_size=1, bias=bias), nn.BatchNorm2d(
            internal_channels), activation)
        self.ext_conv2 = nn.Sequential(nn.ConvTranspose2d(internal_channels,
            internal_channels, kernel_size=kernel_size, stride=2, padding=
            padding, output_padding=1, bias=bias), nn.BatchNorm2d(
            internal_channels), activation)
        self.ext_conv3 = nn.Sequential(nn.Conv2d(internal_channels,
            out_channels, kernel_size=1, bias=bias), nn.BatchNorm2d(
            out_channels), activation)
        self.ext_regul = nn.Dropout2d(p=dropout_prob)
        self.out_prelu = activation

    def forward(self, x, max_indices):
        main = self.main_conv1(x)
        main = self.main_unpool1(main, max_indices)
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)
        out = main + ext
        return self.out_prelu(out)


class ENet(nn.Module):
    """Generate the ENet model.

    :param num_classes: (int): the number of classes to segment.
    :param encoder_relu: (bool, optional): When ``True`` ReLU is used as the
        activation function in the encoder blocks/layers; otherwise, PReLU
        is used. Default: False.
    :param decoder_relu: (bool, optional): When ``True`` ReLU is used as the
        activation function in the decoder blocks/layers; otherwise, PReLU
        is used. Default: True.
    """

    def __init__(self, num_classes, encoder_relu=False, decoder_relu=True,
        **kwargs):
        super().__init__()
        self.initial_block = InitialBlock(3, 16, padding=1, relu=encoder_relu)
        self.downsample1_0 = DownsamplingBottleneck(16, 64, padding=1,
            return_indices=True, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_1 = RegularBottleneck(64, padding=1, dropout_prob=
            0.01, relu=encoder_relu)
        self.regular1_2 = RegularBottleneck(64, padding=1, dropout_prob=
            0.01, relu=encoder_relu)
        self.regular1_3 = RegularBottleneck(64, padding=1, dropout_prob=
            0.01, relu=encoder_relu)
        self.regular1_4 = RegularBottleneck(64, padding=1, dropout_prob=
            0.01, relu=encoder_relu)
        self.downsample2_0 = DownsamplingBottleneck(64, 128, padding=1,
            return_indices=True, dropout_prob=0.1, relu=encoder_relu)
        self.regular2_1 = RegularBottleneck(128, padding=1, dropout_prob=
            0.1, relu=encoder_relu)
        self.dilated2_2 = RegularBottleneck(128, dilation=2, padding=2,
            dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric2_3 = RegularBottleneck(128, kernel_size=5, padding=
            2, asymmetric=True, dropout_prob=0.1, relu=encoder_relu)
        self.dilated2_4 = RegularBottleneck(128, dilation=4, padding=4,
            dropout_prob=0.1, relu=encoder_relu)
        self.regular2_5 = RegularBottleneck(128, padding=1, dropout_prob=
            0.1, relu=encoder_relu)
        self.dilated2_6 = RegularBottleneck(128, dilation=8, padding=8,
            dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric2_7 = RegularBottleneck(128, kernel_size=5,
            asymmetric=True, padding=2, dropout_prob=0.1, relu=encoder_relu)
        self.dilated2_8 = RegularBottleneck(128, dilation=16, padding=16,
            dropout_prob=0.1, relu=encoder_relu)
        self.regular3_0 = RegularBottleneck(128, padding=1, dropout_prob=
            0.1, relu=encoder_relu)
        self.dilated3_1 = RegularBottleneck(128, dilation=2, padding=2,
            dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric3_2 = RegularBottleneck(128, kernel_size=5, padding=
            2, asymmetric=True, dropout_prob=0.1, relu=encoder_relu)
        self.dilated3_3 = RegularBottleneck(128, dilation=4, padding=4,
            dropout_prob=0.1, relu=encoder_relu)
        self.regular3_4 = RegularBottleneck(128, padding=1, dropout_prob=
            0.1, relu=encoder_relu)
        self.dilated3_5 = RegularBottleneck(128, dilation=8, padding=8,
            dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric3_6 = RegularBottleneck(128, kernel_size=5,
            asymmetric=True, padding=2, dropout_prob=0.1, relu=encoder_relu)
        self.dilated3_7 = RegularBottleneck(128, dilation=16, padding=16,
            dropout_prob=0.1, relu=encoder_relu)
        self.upsample4_0 = UpsamplingBottleneck(128, 64, padding=1,
            dropout_prob=0.1, relu=decoder_relu)
        self.regular4_1 = RegularBottleneck(64, padding=1, dropout_prob=0.1,
            relu=decoder_relu)
        self.regular4_2 = RegularBottleneck(64, padding=1, dropout_prob=0.1,
            relu=decoder_relu)
        self.upsample5_0 = UpsamplingBottleneck(64, 16, padding=1,
            dropout_prob=0.1, relu=decoder_relu)
        self.regular5_1 = RegularBottleneck(16, padding=1, dropout_prob=0.1,
            relu=decoder_relu)
        self.transposed_conv = nn.ConvTranspose2d(16, num_classes,
            kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)

    def forward(self, x):
        x = self.initial_block(x)
        x, max_indices1_0 = self.downsample1_0(x)
        x = self.regular1_1(x)
        x = self.regular1_2(x)
        x = self.regular1_3(x)
        x = self.regular1_4(x)
        x, max_indices2_0 = self.downsample2_0(x)
        x = self.regular2_1(x)
        x = self.dilated2_2(x)
        x = self.asymmetric2_3(x)
        x = self.dilated2_4(x)
        x = self.regular2_5(x)
        x = self.dilated2_6(x)
        x = self.asymmetric2_7(x)
        x = self.dilated2_8(x)
        x = self.regular3_0(x)
        x = self.dilated3_1(x)
        x = self.asymmetric3_2(x)
        x = self.dilated3_3(x)
        x = self.regular3_4(x)
        x = self.dilated3_5(x)
        x = self.asymmetric3_6(x)
        x = self.dilated3_7(x)
        x = self.upsample4_0(x, max_indices2_0)
        x = self.regular4_1(x)
        x = self.regular4_2(x)
        x = self.upsample5_0(x, max_indices1_0)
        x = self.regular5_1(x)
        x = self.transposed_conv(x)
        return x


def get_upsampling_weight(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) /
        factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
        dtype=np.float64)
    weight[(list(range(in_channels))), (list(range(out_channels))), :, :
        ] = filt
    return torch.from_numpy(weight).float()


class FCN32VGG(nn.Module):

    def __init__(self, num_classes, pretrained=True, **kwargs):
        super(FCN32VGG, self).__init__()
        vgg = models.vgg16()
        if pretrained:
            vgg.load_state_dict(torch.load(vgg16_caffe_path))
        features, classifier = list(vgg.features.children()), list(vgg.
            classifier.children())
        features[0].padding = 100, 100
        for f in features:
            if 'MaxPool' in f.__class__.__name__:
                f.ceil_mode = True
            elif 'ReLU' in f.__class__.__name__:
                f.inplace = True
        self.features5 = nn.Sequential(*features)
        fc6 = nn.Conv2d(512, 4096, kernel_size=7)
        fc6.weight.data.copy_(classifier[0].weight.data.view(4096, 512, 7, 7))
        fc6.bias.data.copy_(classifier[0].bias.data)
        fc7 = nn.Conv2d(4096, 4096, kernel_size=1)
        fc7.weight.data.copy_(classifier[3].weight.data.view(4096, 4096, 1, 1))
        fc7.bias.data.copy_(classifier[3].bias.data)
        score_fr = nn.Conv2d(4096, num_classes, kernel_size=1)
        score_fr.weight.data.zero_()
        score_fr.bias.data.zero_()
        self.score_fr = nn.Sequential(fc6, nn.ReLU(inplace=True), nn.
            Dropout(), fc7, nn.ReLU(inplace=True), nn.Dropout(), score_fr)
        self.upscore = nn.ConvTranspose2d(num_classes, num_classes,
            kernel_size=64, stride=32, bias=False)
        self.upscore.weight.data.copy_(get_upsampling_weight(num_classes,
            num_classes, 64))

    def forward(self, x):
        x_size = x.size()
        pool5 = self.features5(x)
        score_fr = self.score_fr(pool5)
        upscore = self.upscore(score_fr)
        return upscore[:, :, 19:19 + x_size[2], 19:19 + x_size[3]].contiguous()


class CrossEntropyLoss2d(nn.Module):

    def __init__(self, weight=None, size_average=True, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)


class FocalLoss2d(nn.Module):

    def __init__(self, gamma=2, weight=None, size_average=True,
        ignore_index=255):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss((1 - F.softmax(inputs)) ** self.gamma * F.
            log_softmax(inputs), targets)


class Conv2dDeformable(nn.Module):

    def __init__(self, regular_filter, cuda=True):
        super(Conv2dDeformable, self).__init__()
        assert isinstance(regular_filter, nn.Conv2d)
        self.regular_filter = regular_filter
        self.offset_filter = nn.Conv2d(regular_filter.in_channels, 2 *
            regular_filter.in_channels, kernel_size=3, padding=1, bias=False)
        self.offset_filter.weight.data.normal_(0, 0.0005)
        self.input_shape = None
        self.grid_w = None
        self.grid_h = None
        self.cuda = cuda

    def forward(self, x):
        x_shape = x.size()
        offset = self.offset_filter(x)
        offset_w, offset_h = torch.split(offset, self.regular_filter.
            in_channels, 1)
        offset_w = offset_w.contiguous().view(-1, int(x_shape[2]), int(
            x_shape[3]))
        offset_h = offset_h.contiguous().view(-1, int(x_shape[2]), int(
            x_shape[3]))
        if not self.input_shape or self.input_shape != x_shape:
            self.input_shape = x_shape
            grid_w, grid_h = np.meshgrid(np.linspace(-1, 1, x_shape[3]), np
                .linspace(-1, 1, x_shape[2]))
            grid_w = torch.Tensor(grid_w)
            grid_h = torch.Tensor(grid_h)
            if self.cuda:
                grid_w = grid_w
                grid_h = grid_h
            self.grid_w = nn.Parameter(grid_w)
            self.grid_h = nn.Parameter(grid_h)
        offset_w = offset_w + self.grid_w
        offset_h = offset_h + self.grid_h
        x = x.contiguous().view(-1, int(x_shape[2]), int(x_shape[3])
            ).unsqueeze(1)
        x = F.grid_sample(x, torch.stack((offset_h, offset_w), 3))
        x = x.contiguous().view(-1, int(x_shape[1]), int(x_shape[2]), int(
            x_shape[3]))
        x = self.regular_filter(x)
        return x


class conv2DBatchNorm(nn.Module):

    def __init__(self, in_channels, n_filters, k_size, stride, padding,
        bias=True, dilation=1, with_bn=True):
        super(conv2DBatchNorm, self).__init__()
        if dilation > 1:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters),
                kernel_size=k_size, padding=padding, stride=stride, bias=
                bias, dilation=dilation)
        else:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters),
                kernel_size=k_size, padding=padding, stride=stride, bias=
                bias, dilation=1)
        if with_bn:
            self.cb_unit = nn.Sequential(conv_mod, nn.BatchNorm2d(int(
                n_filters)))
        else:
            self.cb_unit = nn.Sequential(conv_mod)

    def forward(self, inputs):
        outputs = self.cb_unit(inputs)
        return outputs


class deconv2DBatchNorm(nn.Module):

    def __init__(self, in_channels, n_filters, k_size, stride, padding,
        bias=True):
        super(deconv2DBatchNorm, self).__init__()
        self.dcb_unit = nn.Sequential(nn.ConvTranspose2d(int(in_channels),
            int(n_filters), kernel_size=k_size, padding=padding, stride=
            stride, bias=bias), nn.BatchNorm2d(int(n_filters)))

    def forward(self, inputs):
        outputs = self.dcb_unit(inputs)
        return outputs


class conv2DBatchNormRelu(nn.Module):

    def __init__(self, in_channels, n_filters, k_size, stride, padding,
        bias=True, dilation=1, with_bn=True):
        super(conv2DBatchNormRelu, self).__init__()
        if dilation > 1:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters),
                kernel_size=k_size, padding=padding, stride=stride, bias=
                bias, dilation=dilation)
        else:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters),
                kernel_size=k_size, padding=padding, stride=stride, bias=
                bias, dilation=1)
        if with_bn:
            self.cbr_unit = nn.Sequential(conv_mod, nn.BatchNorm2d(int(
                n_filters)), nn.ReLU(inplace=True))
        else:
            self.cbr_unit = nn.Sequential(conv_mod, nn.ReLU(inplace=True))

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs


class deconv2DBatchNormRelu(nn.Module):

    def __init__(self, in_channels, n_filters, k_size, stride, padding,
        bias=True):
        super(deconv2DBatchNormRelu, self).__init__()
        self.dcbr_unit = nn.Sequential(nn.ConvTranspose2d(int(in_channels),
            int(n_filters), kernel_size=k_size, padding=padding, stride=
            stride, bias=bias), nn.BatchNorm2d(int(n_filters)), nn.ReLU(
            inplace=True))

    def forward(self, inputs):
        outputs = self.dcbr_unit(inputs)
        return outputs


class RU(nn.Module):
    """
    Residual Unit for FRRN
    """

    def __init__(self, channels, kernel_size=3, strides=1):
        super(RU, self).__init__()
        self.conv1 = conv2DBatchNormRelu(channels, channels, k_size=
            kernel_size, stride=strides, padding=1)
        self.conv2 = conv2DBatchNorm(channels, channels, k_size=kernel_size,
            stride=strides, padding=1)

    def forward(self, x):
        incoming = x
        x = self.conv1(x)
        x = self.conv2(x)
        return x + incoming


class FRRU(nn.Module):
    """
    Full Resolution Residual Unit for FRRN
    """

    def __init__(self, prev_channels, out_channels, scale):
        super(FRRU, self).__init__()
        self.scale = scale
        self.prev_channels = prev_channels
        self.out_channels = out_channels
        self.conv1 = conv2DBatchNormRelu(prev_channels + 32, out_channels,
            k_size=3, stride=1, padding=1)
        self.conv2 = conv2DBatchNormRelu(out_channels, out_channels, k_size
            =3, stride=1, padding=1)
        self.conv_res = nn.Conv2d(out_channels, 32, kernel_size=1, stride=1,
            padding=0)

    def forward(self, y, z):
        x = torch.cat([y, nn.MaxPool2d(self.scale, self.scale)(z)], dim=1)
        y_prime = self.conv1(x)
        y_prime = self.conv2(y_prime)
        x = self.conv_res(y_prime)
        upsample_size = torch.Size([(_s * self.scale) for _s in y_prime.
            shape[-2:]])
        x = F.upsample(x, size=upsample_size, mode='nearest')
        z_prime = z + x
        return y_prime, z_prime


frrn_specs_dic = {'A': {'encoder': [[3, 96, 2], [4, 192, 4], [2, 384, 8], [
    2, 384, 16]], 'decoder': [[2, 192, 8], [2, 192, 4], [2, 96, 2]]}, 'B':
    {'encoder': [[3, 96, 2], [4, 192, 4], [2, 384, 8], [2, 384, 16], [2, 
    384, 32]], 'decoder': [[2, 192, 16], [2, 192, 8], [2, 192, 4], [2, 96, 2]]}
    }


class frrn(nn.Module):
    """
    Full Resolution Residual Networks for Semantic Segmentation
    URL: https://arxiv.org/abs/1611.08323

    References:
    1) Original Author's code: https://github.com/TobyPDE/FRRN
    2) TF implementation by @kiwonjoon: https://github.com/hiwonjoon/tf-frrn
    """

    def __init__(self, num_classes=21, model_type=None, **kwargs):
        super(frrn, self).__init__()
        self.n_classes = num_classes
        self.model_type = model_type
        self.K = 64 * 512
        self.conv1 = conv2DBatchNormRelu(3, 48, 5, 1, 2)
        self.up_residual_units = []
        self.down_residual_units = []
        for i in range(3):
            self.up_residual_units.append(RU(channels=48, kernel_size=3,
                strides=1))
            self.down_residual_units.append(RU(channels=48, kernel_size=3,
                strides=1))
        self.up_residual_units = nn.ModuleList(self.up_residual_units)
        self.down_residual_units = nn.ModuleList(self.down_residual_units)
        self.split_conv = nn.Conv2d(48, 32, kernel_size=1, padding=0,
            stride=1, bias=True)
        self.encoder_frru_specs = frrn_specs_dic[self.model_type]['encoder']
        self.decoder_frru_specs = frrn_specs_dic[self.model_type]['decoder']
        prev_channels = 48
        self.encoding_frrus = {}
        for n_blocks, channels, scale in self.encoder_frru_specs:
            for block in range(n_blocks):
                key = '_'.join(map(str, ['encoding_frru', n_blocks,
                    channels, scale, block]))
                setattr(self, key, FRRU(prev_channels=prev_channels,
                    out_channels=channels, scale=scale))
            prev_channels = channels
        self.decoding_frrus = {}
        for n_blocks, channels, scale in self.decoder_frru_specs:
            for block in range(n_blocks):
                key = '_'.join(map(str, ['decoding_frru', n_blocks,
                    channels, scale, block]))
                setattr(self, key, FRRU(prev_channels=prev_channels,
                    out_channels=channels, scale=scale))
            prev_channels = channels
        self.merge_conv = nn.Conv2d(prev_channels + 32, 48, kernel_size=1,
            padding=0, stride=1, bias=True)
        self.classif_conv = nn.Conv2d(48, self.n_classes, kernel_size=1,
            padding=0, stride=1, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        for i in range(3):
            x = self.up_residual_units[i](x)
        y = x
        z = self.split_conv(x)
        prev_channels = 48
        for n_blocks, channels, scale in self.encoder_frru_specs:
            y_pooled = F.max_pool2d(y, stride=2, kernel_size=2, padding=0)
            for block in range(n_blocks):
                key = '_'.join(map(str, ['encoding_frru', n_blocks,
                    channels, scale, block]))
                y, z = getattr(self, key)(y_pooled, z)
            prev_channels = channels
        for n_blocks, channels, scale in self.decoder_frru_specs:
            upsample_size = torch.Size([(_s * 2) for _s in y.size()[-2:]])
            y_upsampled = F.upsample(y, size=upsample_size, mode='bilinear')
            for block in range(n_blocks):
                key = '_'.join(map(str, ['decoding_frru', n_blocks,
                    channels, scale, block]))
                y, z = getattr(self, key)(y_upsampled, z)
            prev_channels = channels
        x = torch.cat([F.upsample(y, scale_factor=2, mode='bilinear'), z],
            dim=1)
        x = self.merge_conv(x)
        for i in range(3):
            x = self.down_residual_units[i](x)
        x = self.classif_conv(x)
        return x


class ResBlock(nn.Module):

    def __init__(self, in_channels):
        super(ResBlock, self).__init__()
        self.layer = nn.Sequential(nn.Conv2d(in_channels, in_channels,
            kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1))

    def forward(self, x):
        conv = self.layer(x)
        return F.relu(x.expand_as(conv) + conv)


class ConvResConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvResConv, self).__init__()
        self.layer = nn.Sequential(nn.Conv2d(in_channels, out_channels,
            kernel_size=kernel_size, padding=padding), nn.ReLU(inplace=True
            ), ResBlock(out_channels), nn.Conv2d(out_channels, out_channels,
            kernel_size=3, padding=1))

    def forward(self, x):
        return self.layer(x)


class DeconvBN(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DeconvBN, self).__init__()
        self.layer = nn.Sequential(nn.ConvTranspose2d(in_channels,
            out_channels, kernel_size=2, stride=2))

    def forward(self, x):
        return self.layer(x)


class FusionNet(nn.Module):

    def __init__(self, num_classes, **kwargs):
        super(FusionNet, self).__init__()
        self.enc1 = ConvResConv(3, 64)
        self.enc2 = ConvResConv(64, 128)
        self.enc3 = ConvResConv(128, 256)
        self.enc4 = ConvResConv(256, 512)
        self.middle = ConvResConv(512, 1024)
        self.dec1 = ConvResConv(512, 512)
        self.dec2 = ConvResConv(256, 256)
        self.dec3 = ConvResConv(128, 128)
        self.dec4 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1))
        self.deconvbn1024_512 = DeconvBN(1024, 512)
        self.deconvbn512_256 = DeconvBN(512, 256)
        self.deconvbn256_128 = DeconvBN(256, 128)
        self.deconvbn128_64 = DeconvBN(128, 64)
        self.final = nn.Conv2d(64, num_classes, kernel_size=1, stride=1)
        self.activation = nn.Sigmoid()
        initialize_weights(self)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self._do_downsample(F.relu(enc1)))
        enc3 = self.enc3(self._do_downsample(F.relu(enc2)))
        enc4 = self.enc4(self._do_downsample(F.relu(enc3)))
        middle = self.deconvbn1024_512(self.middle(self._do_downsample(F.
            relu(enc4))))
        dec1 = self.deconvbn512_256(self.dec1(F.relu(middle + enc4)))
        dec2 = self.deconvbn256_128(self.dec2(F.relu(dec1 + enc3)))
        dec3 = self.deconvbn128_64(self.dec3(F.relu(dec2 + enc2)))
        dec4 = self.dec4(F.relu(dec3 + enc1))
        output = self.final(dec4)
        return self.activation(output)

    def _do_downsample(self, x, kernel_size=2, stride=2):
        return F.max_pool2d(x, kernel_size=kernel_size, stride=stride)


class _GlobalConvModule(nn.Module):

    def __init__(self, in_dim, out_dim, kernel_size):
        super(_GlobalConvModule, self).__init__()
        pad0 = floor((kernel_size[0] - 1) / 2)
        pad1 = floor((kernel_size[1] - 1) / 2)
        self.conv_l1 = nn.Conv2d(in_dim, out_dim, kernel_size=(kernel_size[
            0], 1), padding=(pad0, 0))
        self.conv_l2 = nn.Conv2d(out_dim, out_dim, kernel_size=(1,
            kernel_size[1]), padding=(0, pad1))
        self.conv_r1 = nn.Conv2d(in_dim, out_dim, kernel_size=(1,
            kernel_size[1]), padding=(0, pad1))
        self.conv_r2 = nn.Conv2d(out_dim, out_dim, kernel_size=(kernel_size
            [0], 1), padding=(pad0, 0))

    def forward(self, x):
        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)
        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)
        x = x_l + x_r
        return x


class _BoundaryRefineModule(nn.Module):

    def __init__(self, dim):
        super(_BoundaryRefineModule, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        out = x + residual
        return out


class GCN(nn.Module):

    def __init__(self, num_classes, pretrained=True, k=7):
        super(GCN, self).__init__()
        self.K = k
        resnet = models.resnet152(pretrained=pretrained)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.layer1 = nn.Sequential(resnet.maxpool, resnet.layer1)
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.gcm1 = _GlobalConvModule(2048, num_classes, (self.K, self.K))
        self.gcm2 = _GlobalConvModule(1024, num_classes, (self.K, self.K))
        self.gcm3 = _GlobalConvModule(512, num_classes, (self.K, self.K))
        self.gcm4 = _GlobalConvModule(256, num_classes, (self.K, self.K))
        self.brm1 = _BoundaryRefineModule(num_classes)
        self.brm2 = _BoundaryRefineModule(num_classes)
        self.brm3 = _BoundaryRefineModule(num_classes)
        self.brm4 = _BoundaryRefineModule(num_classes)
        self.brm5 = _BoundaryRefineModule(num_classes)
        self.brm6 = _BoundaryRefineModule(num_classes)
        self.brm7 = _BoundaryRefineModule(num_classes)
        self.brm8 = _BoundaryRefineModule(num_classes)
        self.brm9 = _BoundaryRefineModule(num_classes)
        initialize_weights(self.gcm1, self.gcm2, self.gcm3, self.gcm4, self
            .brm1, self.brm2, self.brm3, self.brm4, self.brm5, self.brm6,
            self.brm7, self.brm8, self.brm9)

    def forward(self, x):
        size = x.size()[2:]
        fm0 = self.layer0(x)
        fm1 = self.layer1(fm0)
        fm2 = self.layer2(fm1)
        fm3 = self.layer3(fm2)
        fm4 = self.layer4(fm3)
        gcfm1 = self.brm1(self.gcm1(fm4))
        gcfm2 = self.brm2(self.gcm2(fm3))
        gcfm3 = self.brm3(self.gcm3(fm2))
        gcfm4 = self.brm4(self.gcm4(fm1))
        fs1 = self.brm5(F.interpolate(gcfm1, size=fm3.size()[2:], mode=
            'bilinear') + gcfm2)
        fs2 = self.brm6(F.interpolate(fs1, size=fm2.size()[2:], mode=
            'bilinear') + gcfm3)
        fs3 = self.brm7(F.interpolate(fs2, size=fm1.size()[2:], mode=
            'bilinear') + gcfm4)
        fs4 = self.brm8(F.interpolate(fs3, size=fm0.size()[2:], mode=
            'bilinear'))
        out = self.brm9(F.interpolate(fs4, size=size, mode='bilinear'))
        return out


class _GlobalConvModule(nn.Module):

    def __init__(self, in_dim, out_dim, kernel_size):
        super(_GlobalConvModule, self).__init__()
        pad0 = floor((kernel_size[0] - 1) / 2)
        pad1 = floor((kernel_size[1] - 1) / 2)
        self.conv_l1 = nn.Conv2d(in_dim, out_dim, kernel_size=(kernel_size[
            0], 1), padding=(pad0, 0))
        self.conv_l2 = nn.Conv2d(out_dim, out_dim, kernel_size=(1,
            kernel_size[1]), padding=(0, pad1))
        self.conv_r1 = nn.Conv2d(in_dim, out_dim, kernel_size=(1,
            kernel_size[1]), padding=(0, pad1))
        self.conv_r2 = nn.Conv2d(out_dim, out_dim, kernel_size=(kernel_size
            [0], 1), padding=(pad0, 0))

    def forward(self, x):
        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)
        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)
        x = x_l + x_r
        return x


class _BoundaryRefineModule(nn.Module):

    def __init__(self, dim):
        super(_BoundaryRefineModule, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        out = x + residual
        return out


class GCN_Densenet(nn.Module):

    def __init__(self, num_classes, pretrained=True, k=7, **kwargs):
        super(GCN_Densenet, self).__init__()
        self.K = k
        densenet = models.densenet161(pretrained=pretrained)
        self.layer0 = nn.Sequential(densenet.features.conv0, densenet.
            features.norm0, densenet.features.relu0)
        self.layer1 = nn.Sequential(densenet.features.pool0, densenet.
            features.denseblock1)
        self.layer2 = nn.Sequential(densenet.features.transition1, densenet
            .features.denseblock2)
        self.layer3 = nn.Sequential(densenet.features.transition2, densenet
            .features.denseblock3)
        self.layer4 = nn.Sequential(densenet.features.transition3, densenet
            .features.denseblock4)
        self.gcm1 = _GlobalConvModule(2208, num_classes, (self.K, self.K))
        self.gcm2 = _GlobalConvModule(2112, num_classes, (self.K, self.K))
        self.gcm3 = _GlobalConvModule(768, num_classes, (self.K, self.K))
        self.gcm4 = _GlobalConvModule(384, num_classes, (self.K, self.K))
        self.brm1 = _BoundaryRefineModule(num_classes)
        self.brm2 = _BoundaryRefineModule(num_classes)
        self.brm3 = _BoundaryRefineModule(num_classes)
        self.brm4 = _BoundaryRefineModule(num_classes)
        self.brm5 = _BoundaryRefineModule(num_classes)
        self.brm6 = _BoundaryRefineModule(num_classes)
        self.brm7 = _BoundaryRefineModule(num_classes)
        self.brm8 = _BoundaryRefineModule(num_classes)
        self.brm9 = _BoundaryRefineModule(num_classes)
        initialize_weights(self.gcm1, self.gcm2, self.gcm3, self.gcm4, self
            .brm1, self.brm2, self.brm3, self.brm4, self.brm5, self.brm6,
            self.brm7, self.brm8, self.brm9)

    def forward(self, x):
        size = x.size()[2:]
        fm0 = self.layer0(x)
        fm1 = self.layer1(fm0)
        fm2 = self.layer2(fm1)
        fm3 = self.layer3(fm2)
        fm4 = self.layer4(fm3)
        gcfm1 = self.brm1(self.gcm1(fm4))
        gcfm2 = self.brm2(self.gcm2(fm3))
        gcfm3 = self.brm3(self.gcm3(fm2))
        gcfm4 = self.brm4(self.gcm4(fm1))
        fs1 = self.brm5(F.upsample(gcfm1, size=fm3.size()[2:], mode=
            'bilinear') + gcfm2)
        fs2 = self.brm6(F.upsample(fs1, size=fm2.size()[2:], mode=
            'bilinear') + gcfm3)
        fs3 = self.brm7(F.upsample(fs2, size=fm1.size()[2:], mode=
            'bilinear') + gcfm4)
        fs4 = self.brm8(F.upsample(fs3, size=fm0.size()[2:], mode='bilinear'))
        out = self.brm9(F.upsample(fs4, size=size, mode='bilinear'))
        return out


class _GlobalConvModule(nn.Module):

    def __init__(self, in_dim, out_dim, kernel_size):
        super(_GlobalConvModule, self).__init__()
        pad0 = floor((kernel_size[0] - 1) / 2)
        pad1 = floor((kernel_size[1] - 1) / 2)
        self.conv_l1 = nn.Conv2d(in_dim, out_dim, kernel_size=(kernel_size[
            0], 1), padding=(pad0, 0))
        self.conv_l2 = nn.Conv2d(out_dim, out_dim, kernel_size=(1,
            kernel_size[1]), padding=(0, pad1))
        self.conv_r1 = nn.Conv2d(in_dim, out_dim, kernel_size=(1,
            kernel_size[1]), padding=(0, pad1))
        self.conv_r2 = nn.Conv2d(out_dim, out_dim, kernel_size=(kernel_size
            [0], 1), padding=(pad0, 0))

    def forward(self, x):
        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)
        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)
        x = x_l + x_r
        return x


class _BoundaryRefineModule(nn.Module):

    def __init__(self, dim):
        super(_BoundaryRefineModule, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        out = x + residual
        return out


class _LearnedBilinearDeconvModule(nn.Module):

    def __init__(self, channels):
        super(_LearnedBilinearDeconvModule, self).__init__()
        self.deconv = nn.ConvTranspose2d(channels, channels, kernel_size=4,
            stride=2, padding=1)
        self.deconv.weight.data = self.make_bilinear_weights(4, channels)
        self.deconv.bias.data.zero_()

    def forward(self, x):
        out = self.deconv(x)
        return out

    def make_bilinear_weights(self, size, num_channels):
        factor = (size + 1) // 2
        if size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:size, :size]
        filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center
            ) / factor)
        filt = torch.from_numpy(filt)
        w = torch.zeros(num_channels, num_channels, size, size)
        for i in range(num_channels):
            w[i, i] = filt
        return w


class GCN_NASNet(nn.Module):

    def __init__(self, num_classes, pretrained=True, k=7):
        super(GCN_NASNet, self).__init__()
        self.K = k
        model = NASNetALarge(num_classes=1001)
        if pretrained:
            model.load_state_dict(torch.utils.model_zoo.load_url(
                'http://data.lip6.fr/cadene/pretrainedmodels/nasnetalarge-a1897284.pth'
                ))
        self.nasnet = model
        self.gcm1 = _GlobalConvModule(4032, num_classes, (self.K, self.K))
        self.gcm2 = _GlobalConvModule(2016, num_classes, (self.K, self.K))
        self.gcm3 = _GlobalConvModule(1008, num_classes, (self.K, self.K))
        self.gcm4 = _GlobalConvModule(num_classes, num_classes, (self.K,
            self.K))
        self.brm1 = _BoundaryRefineModule(num_classes)
        self.brm2 = _BoundaryRefineModule(num_classes)
        self.brm3 = _BoundaryRefineModule(num_classes)
        self.brm4 = _BoundaryRefineModule(num_classes)
        self.brm5 = _BoundaryRefineModule(num_classes)
        self.brm6 = _BoundaryRefineModule(num_classes)
        self.brm7 = _BoundaryRefineModule(num_classes)
        self.brm8 = _BoundaryRefineModule(num_classes)
        self.deconv = _LearnedBilinearDeconvModule(num_classes)
        initialize_weights(self.gcm1, self.gcm2, self.gcm3, self.gcm4, self
            .brm1, self.brm2, self.brm3, self.brm4, self.brm5, self.brm6,
            self.brm7, self.brm8)

    def forward(self, x):
        x_conv0 = self.nasnet.conv0(x)
        x_stem_0 = self.nasnet.cell_stem_0(x_conv0)
        x_stem_1 = self.nasnet.cell_stem_1(x_conv0, x_stem_0)
        x_cell_0 = self.nasnet.cell_0(x_stem_1, x_stem_0)
        x_cell_1 = self.nasnet.cell_1(x_cell_0, x_stem_1)
        x_cell_2 = self.nasnet.cell_2(x_cell_1, x_cell_0)
        x_cell_3 = self.nasnet.cell_3(x_cell_2, x_cell_1)
        x_cell_4 = self.nasnet.cell_4(x_cell_3, x_cell_2)
        x_cell_5 = self.nasnet.cell_5(x_cell_4, x_cell_3)
        x_reduction_cell_0 = self.nasnet.reduction_cell_0(x_cell_5, x_cell_4)
        x_cell_6 = self.nasnet.cell_6(x_reduction_cell_0, x_cell_4)
        x_cell_7 = self.nasnet.cell_7(x_cell_6, x_reduction_cell_0)
        x_cell_8 = self.nasnet.cell_8(x_cell_7, x_cell_6)
        x_cell_9 = self.nasnet.cell_9(x_cell_8, x_cell_7)
        x_cell_10 = self.nasnet.cell_10(x_cell_9, x_cell_8)
        x_cell_11 = self.nasnet.cell_11(x_cell_10, x_cell_9)
        x_reduction_cell_1 = self.nasnet.reduction_cell_1(x_cell_11, x_cell_10)
        x_cell_12 = self.nasnet.cell_12(x_reduction_cell_1, x_cell_10)
        x_cell_13 = self.nasnet.cell_13(x_cell_12, x_reduction_cell_1)
        x_cell_14 = self.nasnet.cell_14(x_cell_13, x_cell_12)
        x_cell_15 = self.nasnet.cell_15(x_cell_14, x_cell_13)
        x_cell_16 = self.nasnet.cell_16(x_cell_15, x_cell_14)
        x_cell_17 = self.nasnet.cell_17(x_cell_16, x_cell_15)
        gcfm1 = self.brm1(self.gcm1(x_cell_17))
        gcfm2 = self.brm2(self.gcm2(x_cell_11))
        gcfm3 = self.brm3(self.gcm3(x_cell_5))
        fs1 = self.brm4(self.deconv(gcfm1) + gcfm2)
        fs2 = self.brm5(self.deconv(fs1) + gcfm3)
        fs3 = self.brm6(self.deconv(fs2))
        fs4 = self.brm7(self.deconv(fs3))
        out = self.brm8(self.deconv(self.gcm4(fs4)))
        return out


class MaxPoolPad(nn.Module):

    def __init__(self):
        super(MaxPoolPad, self).__init__()
        self.pad = nn.ZeroPad2d((1, 0, 1, 0))
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        x = self.pad(x)
        x = self.pool(x)
        x = x[:, :, 1:, 1:]
        return x


class AvgPoolPad(nn.Module):

    def __init__(self, stride=2, padding=1):
        super(AvgPoolPad, self).__init__()
        self.pad = nn.ZeroPad2d((1, 0, 1, 0))
        self.pool = nn.AvgPool2d(3, stride=stride, padding=padding,
            count_include_pad=False)

    def forward(self, x):
        x = self.pad(x)
        x = self.pool(x)
        x = x[:, :, 1:, 1:]
        return x


class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, dw_kernel, dw_stride,
        dw_padding, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise_conv2d = nn.Conv2d(in_channels, in_channels,
            dw_kernel, stride=dw_stride, padding=dw_padding, bias=bias,
            groups=in_channels)
        self.pointwise_conv2d = nn.Conv2d(in_channels, out_channels, 1,
            stride=1, bias=bias)

    def forward(self, x):
        x = self.depthwise_conv2d(x)
        x = self.pointwise_conv2d(x)
        return x


class BranchSeparables(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
        padding, bias=False):
        super(BranchSeparables, self).__init__()
        self.relu = nn.ReLU()
        self.separable_1 = SeparableConv2d(in_channels, in_channels,
            kernel_size, stride, padding, bias=bias)
        self.bn_sep_1 = nn.BatchNorm2d(in_channels, eps=0.001, momentum=0.1,
            affine=True)
        self.relu1 = nn.ReLU()
        self.separable_2 = SeparableConv2d(in_channels, out_channels,
            kernel_size, 1, padding, bias=bias)
        self.bn_sep_2 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=
            0.1, affine=True)

    def forward(self, x):
        x = self.relu(x)
        x = self.separable_1(x)
        x = self.bn_sep_1(x)
        x = self.relu1(x)
        x = self.separable_2(x)
        x = self.bn_sep_2(x)
        return x


class BranchSeparablesStem(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
        padding, bias=False):
        super(BranchSeparablesStem, self).__init__()
        self.relu = nn.ReLU()
        self.separable_1 = SeparableConv2d(in_channels, out_channels,
            kernel_size, stride, padding, bias=bias)
        self.bn_sep_1 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=
            0.1, affine=True)
        self.relu1 = nn.ReLU()
        self.separable_2 = SeparableConv2d(out_channels, out_channels,
            kernel_size, 1, padding, bias=bias)
        self.bn_sep_2 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=
            0.1, affine=True)

    def forward(self, x):
        x = self.relu(x)
        x = self.separable_1(x)
        x = self.bn_sep_1(x)
        x = self.relu1(x)
        x = self.separable_2(x)
        x = self.bn_sep_2(x)
        return x


class CellStem0(nn.Module):

    def __init__(self, stem_filters, num_filters=42):
        super(CellStem0, self).__init__()
        self.num_filters = num_filters
        self.stem_filters = stem_filters
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(self.stem_filters, self.
            num_filters, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(self.num_filters, eps
            =0.001, momentum=0.1, affine=True))
        self.comb_iter_0_left = BranchSeparables(self.num_filters, self.
            num_filters, 5, 2, 2)
        self.comb_iter_0_right = BranchSeparablesStem(self.stem_filters,
            self.num_filters, 7, 2, 3, bias=False)
        self.comb_iter_1_left = nn.MaxPool2d(3, stride=2, padding=1)
        self.comb_iter_1_right = BranchSeparablesStem(self.stem_filters,
            self.num_filters, 7, 2, 3, bias=False)
        self.comb_iter_2_left = nn.AvgPool2d(3, stride=2, padding=1,
            count_include_pad=False)
        self.comb_iter_2_right = BranchSeparablesStem(self.stem_filters,
            self.num_filters, 5, 2, 2, bias=False)
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False)
        self.comb_iter_4_left = BranchSeparables(self.num_filters, self.
            num_filters, 3, 1, 1, bias=False)
        self.comb_iter_4_right = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        x1 = self.conv_1x1(x)
        x_comb_iter_0_left = self.comb_iter_0_left(x1)
        x_comb_iter_0_right = self.comb_iter_0_right(x)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
        x_comb_iter_1_left = self.comb_iter_1_left(x1)
        x_comb_iter_1_right = self.comb_iter_1_right(x)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
        x_comb_iter_2_left = self.comb_iter_2_left(x1)
        x_comb_iter_2_right = self.comb_iter_2_right(x)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right
        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1
        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x1)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right
        x_out = torch.cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3,
            x_comb_iter_4], 1)
        return x_out


class CellStem1(nn.Module):

    def __init__(self, stem_filters, num_filters):
        super(CellStem1, self).__init__()
        self.num_filters = num_filters
        self.stem_filters = stem_filters
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(2 * self.num_filters,
            self.num_filters, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(self.num_filters, eps
            =0.001, momentum=0.1, affine=True))
        self.relu = nn.ReLU()
        self.path_1 = nn.Sequential()
        self.path_1.add_module('avgpool', nn.AvgPool2d(1, stride=2,
            count_include_pad=False))
        self.path_1.add_module('conv', nn.Conv2d(self.stem_filters, self.
            num_filters // 2, 1, stride=1, bias=False))
        self.path_2 = nn.ModuleList()
        self.path_2.add_module('pad', nn.ZeroPad2d((0, 1, 0, 1)))
        self.path_2.add_module('avgpool', nn.AvgPool2d(1, stride=2,
            count_include_pad=False))
        self.path_2.add_module('conv', nn.Conv2d(self.stem_filters, self.
            num_filters // 2, 1, stride=1, bias=False))
        self.final_path_bn = nn.BatchNorm2d(self.num_filters, eps=0.001,
            momentum=0.1, affine=True)
        self.comb_iter_0_left = BranchSeparables(self.num_filters, self.
            num_filters, 5, 2, 2, bias=False)
        self.comb_iter_0_right = BranchSeparables(self.num_filters, self.
            num_filters, 7, 2, 3, bias=False)
        self.comb_iter_1_left = nn.MaxPool2d(3, stride=2, padding=1)
        self.comb_iter_1_right = BranchSeparables(self.num_filters, self.
            num_filters, 7, 2, 3, bias=False)
        self.comb_iter_2_left = nn.AvgPool2d(3, stride=2, padding=1,
            count_include_pad=False)
        self.comb_iter_2_right = BranchSeparables(self.num_filters, self.
            num_filters, 5, 2, 2, bias=False)
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False)
        self.comb_iter_4_left = BranchSeparables(self.num_filters, self.
            num_filters, 3, 1, 1, bias=False)
        self.comb_iter_4_right = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x_conv0, x_stem_0):
        x_left = self.conv_1x1(x_stem_0)
        x_relu = self.relu(x_conv0)
        x_path1 = self.path_1(x_relu)
        x_path2 = self.path_2.pad(x_relu)
        x_path2 = x_path2[:, :, 1:, 1:]
        x_path2 = self.path_2.avgpool(x_path2)
        x_path2 = self.path_2.conv(x_path2)
        x_right = self.final_path_bn(torch.cat([x_path1, x_path2], 1))
        x_comb_iter_0_left = self.comb_iter_0_left(x_left)
        x_comb_iter_0_right = self.comb_iter_0_right(x_right)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
        x_comb_iter_1_left = self.comb_iter_1_left(x_left)
        x_comb_iter_1_right = self.comb_iter_1_right(x_right)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
        x_comb_iter_2_left = self.comb_iter_2_left(x_left)
        x_comb_iter_2_right = self.comb_iter_2_right(x_right)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right
        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1
        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x_left)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right
        x_out = torch.cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3,
            x_comb_iter_4], 1)
        return x_out


class FirstCell(nn.Module):

    def __init__(self, in_channels_left, out_channels_left,
        in_channels_right, out_channels_right):
        super(FirstCell, self).__init__()
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(in_channels_right,
            out_channels_right, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(out_channels_right,
            eps=0.001, momentum=0.1, affine=True))
        self.relu = nn.ReLU()
        self.path_1 = nn.Sequential()
        self.path_1.add_module('avgpool', nn.AvgPool2d(1, stride=2,
            count_include_pad=False))
        self.path_1.add_module('conv', nn.Conv2d(in_channels_left,
            out_channels_left, 1, stride=1, bias=False))
        self.path_2 = nn.ModuleList()
        self.path_2.add_module('pad', nn.ZeroPad2d((0, 1, 0, 1)))
        self.path_2.add_module('avgpool', nn.AvgPool2d(1, stride=2,
            count_include_pad=False))
        self.path_2.add_module('conv', nn.Conv2d(in_channels_left,
            out_channels_left, 1, stride=1, bias=False))
        self.final_path_bn = nn.BatchNorm2d(out_channels_left * 2, eps=
            0.001, momentum=0.1, affine=True)
        self.comb_iter_0_left = BranchSeparables(out_channels_right,
            out_channels_right, 5, 1, 2, bias=False)
        self.comb_iter_0_right = BranchSeparables(out_channels_right,
            out_channels_right, 3, 1, 1, bias=False)
        self.comb_iter_1_left = BranchSeparables(out_channels_right,
            out_channels_right, 5, 1, 2, bias=False)
        self.comb_iter_1_right = BranchSeparables(out_channels_right,
            out_channels_right, 3, 1, 1, bias=False)
        self.comb_iter_2_left = nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False)
        self.comb_iter_3_left = nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False)
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False)
        self.comb_iter_4_left = BranchSeparables(out_channels_right,
            out_channels_right, 3, 1, 1, bias=False)

    def forward(self, x, x_prev):
        x_relu = self.relu(x_prev)
        x_path1 = self.path_1(x_relu)
        x_path2 = self.path_2.pad(x_relu)
        x_path2 = x_path2[:, :, 1:, 1:]
        x_path2 = self.path_2.avgpool(x_path2)
        x_path2 = self.path_2.conv(x_path2)
        x_left = self.final_path_bn(torch.cat([x_path1, x_path2], 1))
        x_right = self.conv_1x1(x)
        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
        x_comb_iter_1_left = self.comb_iter_1_left(x_left)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2 = x_comb_iter_2_left + x_left
        x_comb_iter_3_left = self.comb_iter_3_left(x_left)
        x_comb_iter_3_right = self.comb_iter_3_right(x_left)
        x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right
        x_comb_iter_4_left = self.comb_iter_4_left(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_right
        x_out = torch.cat([x_left, x_comb_iter_0, x_comb_iter_1,
            x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class NormalCell(nn.Module):

    def __init__(self, in_channels_left, out_channels_left,
        in_channels_right, out_channels_right):
        super(NormalCell, self).__init__()
        self.conv_prev_1x1 = nn.Sequential()
        self.conv_prev_1x1.add_module('relu', nn.ReLU())
        self.conv_prev_1x1.add_module('conv', nn.Conv2d(in_channels_left,
            out_channels_left, 1, stride=1, bias=False))
        self.conv_prev_1x1.add_module('bn', nn.BatchNorm2d(
            out_channels_left, eps=0.001, momentum=0.1, affine=True))
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(in_channels_right,
            out_channels_right, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(out_channels_right,
            eps=0.001, momentum=0.1, affine=True))
        self.comb_iter_0_left = BranchSeparables(out_channels_right,
            out_channels_right, 5, 1, 2, bias=False)
        self.comb_iter_0_right = BranchSeparables(out_channels_left,
            out_channels_left, 3, 1, 1, bias=False)
        self.comb_iter_1_left = BranchSeparables(out_channels_left,
            out_channels_left, 5, 1, 2, bias=False)
        self.comb_iter_1_right = BranchSeparables(out_channels_left,
            out_channels_left, 3, 1, 1, bias=False)
        self.comb_iter_2_left = nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False)
        self.comb_iter_3_left = nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False)
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False)
        self.comb_iter_4_left = BranchSeparables(out_channels_right,
            out_channels_right, 3, 1, 1, bias=False)

    def forward(self, x, x_prev):
        x_left = self.conv_prev_1x1(x_prev)
        x_right = self.conv_1x1(x)
        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
        x_comb_iter_1_left = self.comb_iter_1_left(x_left)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2 = x_comb_iter_2_left + x_left
        x_comb_iter_3_left = self.comb_iter_3_left(x_left)
        x_comb_iter_3_right = self.comb_iter_3_right(x_left)
        x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right
        x_comb_iter_4_left = self.comb_iter_4_left(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_right
        x_out = torch.cat([x_left, x_comb_iter_0, x_comb_iter_1,
            x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class ReductionCell0(nn.Module):

    def __init__(self, in_channels_left, out_channels_left,
        in_channels_right, out_channels_right):
        super(ReductionCell0, self).__init__()
        self.conv_prev_1x1 = nn.Sequential()
        self.conv_prev_1x1.add_module('relu', nn.ReLU())
        self.conv_prev_1x1.add_module('conv', nn.Conv2d(in_channels_left,
            out_channels_left, 1, stride=1, bias=False))
        self.conv_prev_1x1.add_module('bn', nn.BatchNorm2d(
            out_channels_left, eps=0.001, momentum=0.1, affine=True))
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(in_channels_right,
            out_channels_right, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(out_channels_right,
            eps=0.001, momentum=0.1, affine=True))
        self.comb_iter_0_left = BranchSeparablesReduction(out_channels_right,
            out_channels_right, 5, 2, 2, bias=False)
        self.comb_iter_0_right = BranchSeparablesReduction(out_channels_right,
            out_channels_right, 7, 2, 3, bias=False)
        self.comb_iter_1_left = MaxPoolPad()
        self.comb_iter_1_right = BranchSeparablesReduction(out_channels_right,
            out_channels_right, 7, 2, 3, bias=False)
        self.comb_iter_2_left = AvgPoolPad()
        self.comb_iter_2_right = BranchSeparablesReduction(out_channels_right,
            out_channels_right, 5, 2, 2, bias=False)
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False)
        self.comb_iter_4_left = BranchSeparablesReduction(out_channels_right,
            out_channels_right, 3, 1, 1, bias=False)
        self.comb_iter_4_right = MaxPoolPad()

    def forward(self, x, x_prev):
        x_left = self.conv_prev_1x1(x_prev)
        x_right = self.conv_1x1(x)
        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
        x_comb_iter_1_left = self.comb_iter_1_left(x_right)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2_right = self.comb_iter_2_right(x_left)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right
        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1
        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right
        x_out = torch.cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3,
            x_comb_iter_4], 1)
        return x_out


class ReductionCell1(nn.Module):

    def __init__(self, in_channels_left, out_channels_left,
        in_channels_right, out_channels_right):
        super(ReductionCell1, self).__init__()
        self.conv_prev_1x1 = nn.Sequential()
        self.conv_prev_1x1.add_module('relu', nn.ReLU())
        self.conv_prev_1x1.add_module('conv', nn.Conv2d(in_channels_left,
            out_channels_left, 1, stride=1, bias=False))
        self.conv_prev_1x1.add_module('bn', nn.BatchNorm2d(
            out_channels_left, eps=0.001, momentum=0.1, affine=True))
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(in_channels_right,
            out_channels_right, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(out_channels_right,
            eps=0.001, momentum=0.1, affine=True))
        self.comb_iter_0_left = BranchSeparables(out_channels_right,
            out_channels_right, 5, 2, 2, bias=False)
        self.comb_iter_0_right = BranchSeparables(out_channels_right,
            out_channels_right, 7, 2, 3, bias=False)
        self.comb_iter_1_left = nn.MaxPool2d(3, stride=2, padding=1)
        self.comb_iter_1_right = BranchSeparables(out_channels_right,
            out_channels_right, 7, 2, 3, bias=False)
        self.comb_iter_2_left = nn.AvgPool2d(3, stride=2, padding=1,
            count_include_pad=False)
        self.comb_iter_2_right = BranchSeparables(out_channels_right,
            out_channels_right, 5, 2, 2, bias=False)
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False)
        self.comb_iter_4_left = BranchSeparables(out_channels_right,
            out_channels_right, 3, 1, 1, bias=False)
        self.comb_iter_4_right = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x, x_prev):
        x_left = self.conv_prev_1x1(x_prev)
        x_right = self.conv_1x1(x)
        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
        x_comb_iter_1_left = self.comb_iter_1_left(x_right)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2_right = self.comb_iter_2_right(x_left)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right
        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1
        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right
        x_out = torch.cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3,
            x_comb_iter_4], 1)
        return x_out


class NASNetALarge(nn.Module):
    """NASNetALarge (6 @ 4032) """

    def __init__(self, num_classes=1001, stem_filters=96,
        penultimate_filters=4032, filters_multiplier=2):
        super(NASNetALarge, self).__init__()
        self.num_classes = num_classes
        self.stem_filters = stem_filters
        self.penultimate_filters = penultimate_filters
        self.filters_multiplier = filters_multiplier
        filters = self.penultimate_filters // 24
        self.conv0 = nn.Sequential()
        self.conv0.add_module('conv', nn.Conv2d(in_channels=3, out_channels
            =self.stem_filters, kernel_size=3, padding=0, stride=2, bias=False)
            )
        self.conv0.add_module('bn', nn.BatchNorm2d(self.stem_filters, eps=
            0.001, momentum=0.1, affine=True))
        self.cell_stem_0 = CellStem0(self.stem_filters, num_filters=filters //
            filters_multiplier ** 2)
        self.cell_stem_1 = CellStem1(self.stem_filters, num_filters=filters //
            filters_multiplier)
        self.cell_0 = FirstCell(in_channels_left=filters, out_channels_left
            =filters // 2, in_channels_right=2 * filters,
            out_channels_right=filters)
        self.cell_1 = NormalCell(in_channels_left=2 * filters,
            out_channels_left=filters, in_channels_right=6 * filters,
            out_channels_right=filters)
        self.cell_2 = NormalCell(in_channels_left=6 * filters,
            out_channels_left=filters, in_channels_right=6 * filters,
            out_channels_right=filters)
        self.cell_3 = NormalCell(in_channels_left=6 * filters,
            out_channels_left=filters, in_channels_right=6 * filters,
            out_channels_right=filters)
        self.cell_4 = NormalCell(in_channels_left=6 * filters,
            out_channels_left=filters, in_channels_right=6 * filters,
            out_channels_right=filters)
        self.cell_5 = NormalCell(in_channels_left=6 * filters,
            out_channels_left=filters, in_channels_right=6 * filters,
            out_channels_right=filters)
        self.reduction_cell_0 = ReductionCell0(in_channels_left=6 * filters,
            out_channels_left=2 * filters, in_channels_right=6 * filters,
            out_channels_right=2 * filters)
        self.cell_6 = FirstCell(in_channels_left=6 * filters,
            out_channels_left=filters, in_channels_right=8 * filters,
            out_channels_right=2 * filters)
        self.cell_7 = NormalCell(in_channels_left=8 * filters,
            out_channels_left=2 * filters, in_channels_right=12 * filters,
            out_channels_right=2 * filters)
        self.cell_8 = NormalCell(in_channels_left=12 * filters,
            out_channels_left=2 * filters, in_channels_right=12 * filters,
            out_channels_right=2 * filters)
        self.cell_9 = NormalCell(in_channels_left=12 * filters,
            out_channels_left=2 * filters, in_channels_right=12 * filters,
            out_channels_right=2 * filters)
        self.cell_10 = NormalCell(in_channels_left=12 * filters,
            out_channels_left=2 * filters, in_channels_right=12 * filters,
            out_channels_right=2 * filters)
        self.cell_11 = NormalCell(in_channels_left=12 * filters,
            out_channels_left=2 * filters, in_channels_right=12 * filters,
            out_channels_right=2 * filters)
        self.reduction_cell_1 = ReductionCell1(in_channels_left=12 *
            filters, out_channels_left=4 * filters, in_channels_right=12 *
            filters, out_channels_right=4 * filters)
        self.cell_12 = FirstCell(in_channels_left=12 * filters,
            out_channels_left=2 * filters, in_channels_right=16 * filters,
            out_channels_right=4 * filters)
        self.cell_13 = NormalCell(in_channels_left=16 * filters,
            out_channels_left=4 * filters, in_channels_right=24 * filters,
            out_channels_right=4 * filters)
        self.cell_14 = NormalCell(in_channels_left=24 * filters,
            out_channels_left=4 * filters, in_channels_right=24 * filters,
            out_channels_right=4 * filters)
        self.cell_15 = NormalCell(in_channels_left=24 * filters,
            out_channels_left=4 * filters, in_channels_right=24 * filters,
            out_channels_right=4 * filters)
        self.cell_16 = NormalCell(in_channels_left=24 * filters,
            out_channels_left=4 * filters, in_channels_right=24 * filters,
            out_channels_right=4 * filters)
        self.cell_17 = NormalCell(in_channels_left=24 * filters,
            out_channels_left=4 * filters, in_channels_right=24 * filters,
            out_channels_right=4 * filters)
        self.relu = nn.ReLU()
        self.avg_pool = nn.AvgPool2d(11, stride=1, padding=0)
        self.dropout = nn.Dropout()
        self.last_linear = nn.Linear(24 * filters, self.num_classes)

    def features(self, input):
        x_conv0 = self.conv0(input)
        x_stem_0 = self.cell_stem_0(x_conv0)
        x_stem_1 = self.cell_stem_1(x_conv0, x_stem_0)
        x_cell_0 = self.cell_0(x_stem_1, x_stem_0)
        x_cell_1 = self.cell_1(x_cell_0, x_stem_1)
        x_cell_2 = self.cell_2(x_cell_1, x_cell_0)
        x_cell_3 = self.cell_3(x_cell_2, x_cell_1)
        x_cell_4 = self.cell_4(x_cell_3, x_cell_2)
        x_cell_5 = self.cell_5(x_cell_4, x_cell_3)
        x_reduction_cell_0 = self.reduction_cell_0(x_cell_5, x_cell_4)
        x_cell_6 = self.cell_6(x_reduction_cell_0, x_cell_4)
        x_cell_7 = self.cell_7(x_cell_6, x_reduction_cell_0)
        x_cell_8 = self.cell_8(x_cell_7, x_cell_6)
        x_cell_9 = self.cell_9(x_cell_8, x_cell_7)
        x_cell_10 = self.cell_10(x_cell_9, x_cell_8)
        x_cell_11 = self.cell_11(x_cell_10, x_cell_9)
        x_reduction_cell_1 = self.reduction_cell_1(x_cell_11, x_cell_10)
        x_cell_12 = self.cell_12(x_reduction_cell_1, x_cell_10)
        x_cell_13 = self.cell_13(x_cell_12, x_reduction_cell_1)
        x_cell_14 = self.cell_14(x_cell_13, x_cell_12)
        x_cell_15 = self.cell_15(x_cell_14, x_cell_13)
        x_cell_16 = self.cell_16(x_cell_15, x_cell_14)
        x_cell_17 = self.cell_17(x_cell_16, x_cell_15)
        return x_cell_17

    def logits(self, features):
        x = self.relu(features)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x


class _GlobalConvModule(nn.Module):

    def __init__(self, in_dim, out_dim, kernel_size):
        super(_GlobalConvModule, self).__init__()
        pad0 = floor((kernel_size[0] - 1) / 2)
        pad1 = floor((kernel_size[1] - 1) / 2)
        self.conv_l1 = nn.Conv2d(in_dim, out_dim, kernel_size=(kernel_size[
            0], 1), padding=(pad0, 0))
        self.conv_l2 = nn.Conv2d(out_dim, out_dim, kernel_size=(1,
            kernel_size[1]), padding=(0, pad1))
        self.conv_r1 = nn.Conv2d(in_dim, out_dim, kernel_size=(1,
            kernel_size[1]), padding=(0, pad1))
        self.conv_r2 = nn.Conv2d(out_dim, out_dim, kernel_size=(kernel_size
            [0], 1), padding=(pad0, 0))

    def forward(self, x):
        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)
        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)
        x = x_l + x_r
        return x


class _BoundaryRefineModule(nn.Module):

    def __init__(self, dim):
        super(_BoundaryRefineModule, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        out = x + residual
        return out


class _PyramidSpatialPoolingModule(nn.Module):

    def __init__(self, in_channels, down_channels, out_size, levels=(1, 2, 
        3, 6)):
        super(_PyramidSpatialPoolingModule, self).__init__()
        self.out_channels = len(levels) * down_channels
        self.layers = nn.ModuleList()
        for level in levels:
            layer = nn.Sequential(nn.AdaptiveAvgPool2d(level), nn.Conv2d(
                in_channels, down_channels, kernel_size=1, padding=0, bias=
                False), nn.BatchNorm2d(down_channels), nn.ReLU(inplace=True
                ), nn.Upsample(size=out_size, mode='bilinear'))
            self.layers.append(layer)

    def forward(self, x):
        features = [layer(x) for layer in self.layers]
        out = torch.cat(features, 1)
        return out


class GCN_PSP(nn.Module):

    def __init__(self, num_classes, pretrained=True, k=7, input_size=512):
        super(GCN_PSP, self).__init__()
        self.K = k
        self.input_size = input_size
        resnet = models.resnet152(pretrained=pretrained)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.layer1 = nn.Sequential(resnet.maxpool, resnet.layer1)
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.gcm1 = _GlobalConvModule(2048, num_classes, (self.K, self.K))
        self.gcm2 = _GlobalConvModule(1024, num_classes, (self.K, self.K))
        self.gcm3 = _GlobalConvModule(512, num_classes, (self.K, self.K))
        self.gcm4 = _GlobalConvModule(256, num_classes, (self.K, self.K))
        self.brm1 = _BoundaryRefineModule(num_classes)
        self.brm2 = _BoundaryRefineModule(num_classes)
        self.brm3 = _BoundaryRefineModule(num_classes)
        self.brm4 = _BoundaryRefineModule(num_classes)
        self.brm5 = _BoundaryRefineModule(num_classes)
        self.brm6 = _BoundaryRefineModule(num_classes)
        self.brm7 = _BoundaryRefineModule(num_classes)
        self.brm8 = _BoundaryRefineModule(num_classes)
        self.brm9 = _BoundaryRefineModule(num_classes)
        self.psp = _PyramidSpatialPoolingModule(num_classes, 10, input_size,
            levels=(1, 2, 3, 6, 8))
        self.final = nn.Sequential(nn.Conv2d(num_classes + self.psp.
            out_channels, num_classes, kernel_size=3, padding=1, bias=False
            ), nn.BatchNorm2d(num_classes), nn.ReLU(inplace=True), nn.
            Conv2d(num_classes, num_classes, kernel_size=1, padding=0))
        initialize_weights(self.gcm1, self.gcm2, self.gcm3, self.gcm4, self
            .brm1, self.brm2, self.brm3, self.brm4, self.brm5, self.brm6,
            self.brm7, self.brm8, self.brm9, self.psp, self.final)

    def forward(self, x):
        fm0 = self.layer0(x)
        fm1 = self.layer1(fm0)
        fm2 = self.layer2(fm1)
        fm3 = self.layer3(fm2)
        fm4 = self.layer4(fm3)
        gcfm1 = self.brm1(self.gcm1(fm4))
        gcfm2 = self.brm2(self.gcm2(fm3))
        gcfm3 = self.brm3(self.gcm3(fm2))
        gcfm4 = self.brm4(self.gcm4(fm1))
        fs1 = self.brm5(F.interpolate(gcfm1, fm3.size()[2:], mode=
            'bilinear') + gcfm2)
        fs2 = self.brm6(F.interpolate(fs1, fm2.size()[2:], mode='bilinear') +
            gcfm3)
        fs3 = self.brm7(F.interpolate(fs2, fm1.size()[2:], mode='bilinear') +
            gcfm4)
        fs4 = self.brm8(F.interpolate(fs3, fm0.size()[2:], mode='bilinear'))
        fs5 = self.brm9(F.interpolate(fs4, self.input_size, mode='bilinear'))
        ppm = torch.cat([self.psp(fs5), fs5], 1)
        out = self.final(ppm)
        return out


class _GlobalConvModule(nn.Module):

    def __init__(self, in_dim, out_dim, kernel_size):
        super(_GlobalConvModule, self).__init__()
        pad0 = floor((kernel_size[0] - 1) / 2)
        pad1 = floor((kernel_size[1] - 1) / 2)
        self.conv_l1 = nn.Conv2d(in_dim, out_dim, kernel_size=(kernel_size[
            0], 1), padding=(pad0, 0))
        self.conv_l2 = nn.Conv2d(out_dim, out_dim, kernel_size=(1,
            kernel_size[1]), padding=(0, pad1))
        self.conv_r1 = nn.Conv2d(in_dim, out_dim, kernel_size=(1,
            kernel_size[1]), padding=(0, pad1))
        self.conv_r2 = nn.Conv2d(out_dim, out_dim, kernel_size=(kernel_size
            [0], 1), padding=(pad0, 0))

    def forward(self, x):
        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)
        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)
        x = x_l + x_r
        return x


class _BoundaryRefineModule(nn.Module):

    def __init__(self, dim):
        super(_BoundaryRefineModule, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        out = x + residual
        return out


class _DeconvModule(nn.Module):

    def __init__(self, channels):
        super(_DeconvModule, self).__init__()
        self.deconv = nn.ConvTranspose2d(channels, channels, kernel_size=4,
            stride=2, padding=1)
        self.deconv.weight.data = self.make_bilinear_weights(4, channels)
        self.deconv.bias.data.zero_()

    def forward(self, x):
        out = self.deconv(x)
        return out

    def make_bilinear_weights(self, size, num_channels):
        factor = (size + 1) // 2
        if size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:size, :size]
        filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center
            ) / factor)
        filt = torch.from_numpy(filt)
        w = torch.zeros(num_channels, num_channels, size, size)
        for i in range(num_channels):
            w[i, i] = filt
        return w


class _PyramidSpatialPoolingModule(nn.Module):

    def __init__(self, in_channels, down_channels, out_size, levels=(1, 2, 
        3, 6)):
        super(_PyramidSpatialPoolingModule, self).__init__()
        self.out_channels = len(levels) * down_channels
        self.layers = nn.ModuleList()
        for level in levels:
            layer = nn.Sequential(nn.AdaptiveAvgPool2d(level), nn.Conv2d(
                in_channels, down_channels, kernel_size=1, padding=0, bias=
                False), nn.BatchNorm2d(down_channels), nn.ReLU(inplace=True
                ), nn.Upsample(size=out_size, mode='bilinear'))
            self.layers.append(layer)

    def forward(self, x):
        features = [layer(x) for layer in self.layers]
        out = torch.cat(features, 1)
        return out


class LambdaBase(nn.Sequential):

    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input


pretrained_settings = {'resnext101_32x4d': {'imagenet': {'url':
    'http://data.lip6.fr/cadene/pretrainedmodels/resnext101_32x4d-29e315fa.pth'
    , 'input_space': 'RGB', 'input_size': [3, 224, 224], 'input_range': [0,
    1], 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225],
    'num_classes': 1000}}, 'resnext101_64x4d': {'imagenet': {'url':
    'http://data.lip6.fr/cadene/pretrainedmodels/resnext101_64x4d-e77a0586.pth'
    , 'input_space': 'RGB', 'input_size': [3, 224, 224], 'input_range': [0,
    1], 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225],
    'num_classes': 1000}}}


def resnext101_64x4d(pretrained='imagenet'):
    """Pretrained ResNeXt101_64x4d model"""
    model = ResNeXt101_64x4d(num_classes=1000)
    if pretrained:
        settings = pretrained_settings['resnext101_64x4d'][pretrained]
        model.load_state_dict(model_zoo.load_url(settings['url']))
        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']
    return model


class ResNeXt(nn.Module):

    def __init__(self, pretrained=True):
        super(ResNeXt, self).__init__()
        if pretrained:
            self.resnext = resnext101_64x4d()
        else:
            self.resnext = resnext101_64x4d(pretrained=None)
        self.layer0 = nn.Sequential(self.resnext.features[0], self.resnext.
            features[1], self.resnext.features[2], self.resnext.features[3])
        self.layer1 = self.resnext.features[4]
        self.layer2 = self.resnext.features[5]
        self.layer3 = self.resnext.features[6]
        self.layer4 = self.resnext.features[7]
        self.layer5 = nn.Sequential(nn.AvgPool2d((7, 7), (1, 1)), Lambda(lambda
            x: x.view(x.size(0), -1)), nn.Sequential(Lambda(lambda x: x.
            view(1, -1) if 1 == len(x.size()) else x), nn.Linear(2048, 1000)))

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x


class GCN_Resnext(nn.Module):

    def __init__(self, num_classes, pretrained=True, k=7, input_size=512,
        **kwargs):
        super(GCN_Resnext, self).__init__()
        self.num_classes = num_classes
        self.K = k
        num_imd_feats = 40
        self.resnext = ResNeXt(pretrained)
        self.gcm1 = _GlobalConvModule(2048, num_imd_feats, (self.K, self.K))
        self.gcm2 = _GlobalConvModule(1024, num_imd_feats, (self.K, self.K))
        self.gcm3 = _GlobalConvModule(512, num_imd_feats, (self.K, self.K))
        self.gcm4 = _GlobalConvModule(256, num_imd_feats, (self.K, self.K))
        self.brm1 = _BoundaryRefineModule(num_imd_feats)
        self.brm2 = _BoundaryRefineModule(num_imd_feats)
        self.brm3 = _BoundaryRefineModule(num_imd_feats)
        self.brm4 = _BoundaryRefineModule(num_imd_feats)
        self.brm5 = _BoundaryRefineModule(num_imd_feats)
        self.brm6 = _BoundaryRefineModule(num_imd_feats)
        self.brm7 = _BoundaryRefineModule(num_imd_feats)
        self.brm8 = _BoundaryRefineModule(num_imd_feats)
        self.brm9 = _BoundaryRefineModule(num_imd_feats)
        self.deconv = _DeconvModule(num_imd_feats)
        self.psp_module = _PyramidSpatialPoolingModule(num_imd_feats, 30,
            input_size, levels=(1, 2, 3, 6))
        self.final = nn.Sequential(nn.Conv2d(num_imd_feats + self.
            psp_module.out_channels, num_imd_feats, kernel_size=3, padding=
            1), nn.BatchNorm2d(num_imd_feats), nn.ReLU(inplace=True), nn.
            Conv2d(num_imd_feats, num_classes, kernel_size=1, padding=0))
        self.initialize_weights(self.gcm1, self.gcm2, self.gcm3, self.gcm4)
        self.initialize_weights(self.brm1, self.brm2, self.brm3, self.brm4,
            self.brm5, self.brm6, self.brm7, self.brm8, self.brm9)
        self.initialize_weights(self.psp_module, self.final)

    def forward(self, x):
        fm0 = self.resnext.layer0(x)
        fm1 = self.resnext.layer1(fm0)
        fm2 = self.resnext.layer2(fm1)
        fm3 = self.resnext.layer3(fm2)
        fm4 = self.resnext.layer4(fm3)
        gcfm1 = self.brm1(self.gcm1(fm4))
        gcfm2 = self.brm2(self.gcm2(fm3))
        gcfm3 = self.brm3(self.gcm3(fm2))
        gcfm4 = self.brm4(self.gcm4(fm1))
        fs1 = self.brm5(self.deconv(gcfm1) + gcfm2)
        fs2 = self.brm6(self.deconv(fs1) + gcfm3)
        fs3 = self.brm7(self.deconv(fs2) + gcfm4)
        fs4 = self.brm8(self.deconv(fs3))
        fs5 = self.brm9(self.deconv(fs4))
        p = torch.cat([self.psp_module(fs5), fs5], 1)
        out = self.final(p)
        return out

    def initialize_weights(self, *models):
        for model in models:
            for module in model.modules():
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.
                    Linear):
                    nn.init.kaiming_normal_(module.weight)
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.BatchNorm2d):
                    module.weight.data.fill_(1)
                    module.bias.data.zero_()


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1
        ):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride=stride, dilation=dilation
            )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride=1, dilation=dilation)
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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1
        ):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            dilation=dilation, padding=dilation, bias=False)
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
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers=(3, 4, 23, 3)):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
            dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
            dilation=4)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_3 = self.layer3(x)
        x = self.layer4(x_3)
        return x, x_3


class _DenseLayer(nn.Sequential):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
            growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate,
            growth_rate, kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate,
        drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate,
                growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):

    def __init__(self, num_input_features, num_output_features, downsample=True
        ):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features,
            num_output_features, kernel_size=1, stride=1, bias=False))
        if downsample:
            self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))
        else:
            self.add_module('pool', nn.AvgPool2d(kernel_size=1, stride=1))


def densenet121(pretrained=False, **kwargs):
    """Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Orig_DenseNet(num_init_features=64, growth_rate=32,
        block_config=(6, 12, 24, 16), **kwargs)
    if pretrained:
        pattern = re.compile(
            '^(.*denselayer\\d+\\.(?:norm|relu|conv))\\.((?:[12])\\.(?:weight|bias|running_mean|running_var))$'
            )
        state_dict = model_zoo.load_url(model_urls['densenet121'])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    return model


class DenseNet(nn.Module):

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
        num_init_features=64, bn_size=4, drop_rate=0, pretrained=True):
        super(DenseNet, self).__init__()
        self.start_features = nn.Sequential(OrderedDict([('conv0', nn.
            Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3,
            bias=False)), ('norm0', nn.BatchNorm2d(num_init_features)), (
            'relu0', nn.ReLU(inplace=True)), ('pool0', nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1))]))
        num_features = num_init_features
        init_weights = list(densenet121(pretrained=True).features.children())
        start = 0
        for i, c in enumerate(self.start_features.children()):
            if pretrained:
                c.load_state_dict(init_weights[i].state_dict())
            start += 1
        self.blocks = nn.ModuleList()
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=
                num_features, bn_size=bn_size, growth_rate=growth_rate,
                drop_rate=drop_rate)
            if pretrained:
                block.load_state_dict(init_weights[start].state_dict())
            start += 1
            self.blocks.append(block)
            setattr(self, 'denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                downsample = i < 1
                trans = _Transition(num_input_features=num_features,
                    num_output_features=num_features // 2, downsample=
                    downsample)
                if pretrained:
                    trans.load_state_dict(init_weights[start].state_dict())
                start += 1
                self.blocks.append(trans)
                setattr(self, 'transition%d' % (i + 1), trans)
                num_features = num_features // 2

    def forward(self, x):
        out = self.start_features(x)
        deep_features = None
        for i, block in enumerate(self.blocks):
            out = block(out)
            if i == 5:
                deep_features = out
        return out, deep_features


class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes, expand1x1_planes,
        expand3x3_planes, dilation=1):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
            kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
            kernel_size=3, padding=dilation, dilation=dilation)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))], 1)


def load_weights_sequential(target, source_state):
    new_dict = OrderedDict()
    for (k1, v1), (k2, v2) in zip(target.state_dict().items(), source_state
        .items()):
        new_dict[k1] = v2
    target.load_state_dict(new_dict)


class SqueezeNet(nn.Module):

    def __init__(self, pretrained=False):
        super(SqueezeNet, self).__init__()
        self.feat_1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=
            2, padding=1), nn.ReLU(inplace=True))
        self.feat_2 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2,
            padding=1), Fire(64, 16, 64, 64), Fire(128, 16, 64, 64))
        self.feat_3 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2,
            padding=1), Fire(128, 32, 128, 128, 2), Fire(256, 32, 128, 128, 2))
        self.feat_4 = nn.Sequential(Fire(256, 48, 192, 192, 4), Fire(384, 
            48, 192, 192, 4), Fire(384, 64, 256, 256, 4), Fire(512, 64, 256,
            256, 4))
        if pretrained:
            weights = squeezenet1_1(pretrained=True).features.state_dict()
            load_weights_sequential(self, weights)

    def forward(self, x):
        f1 = self.feat_1(x)
        f2 = self.feat_2(f1)
        f3 = self.feat_3(f2)
        f4 = self.feat_4(f3)
        return f4, f3


class PSPModule(nn.Module):

    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for
            size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1),
            out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode=
            'bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class PSPUpsample(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3,
            padding=1), nn.BatchNorm2d(out_channels), nn.PReLU())

    def forward(self, x):
        h, w = 2 * x.size(2), 2 * x.size(3)
        p = F.upsample(input=x, size=(h, w), mode='bilinear')
        return self.conv(p)


def densenet(pretrained=True):
    return DenseNet(pretrained=pretrained)


def resnet101(pretrained=False, root='./pretrain_models', **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        print('===>> pretrained==True, returns a model pre-trained on ImageNet'
            )
        from .model_store import get_model_file
        model.load_state_dict(torch.load(get_model_file('resnet101', root=
            root)), strict=False)
    return model


def resnet152(pretrained=False, root='~/.encoding/models', **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DilatedResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


def resnet34(pretrained=True):
    model = ResNet(BasicBlock, [3, 4, 6, 3])
    if pretrained:
        load_weights_sequential(model, model_zoo.load_url(model_urls[
            'resnet34']))
    return model


def resnet50(pretrained=False, root='./pretrain_models', **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        print('===>> pretrained==True, returns a model pre-trained on ImageNet'
            )
        from .model_store import get_model_file
        model.load_state_dict(torch.load(get_model_file('resnet50', root=
            root)), strict=False)
    return model


extractor_models = {'resnet18': resnet18, 'resnet34': resnet34, 'resnet50':
    resnet50, 'resnet101': resnet101, 'resnet152': resnet152, 'densenet121':
    densenet}


class PSPNet(nn.Module):

    def __init__(self, num_classes=18, pretrained=True, backend=
        'densenet121', sizes=(1, 2, 3, 6), psp_size=2048,
        deep_features_size=1024, **kwargs):
        super().__init__()
        self.feats = extractor_models[backend](pretrained=pretrained)
        self.psp = PSPModule(psp_size, 1024, sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)
        self.up_1 = PSPUpsample(1024, 256)
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(64, 64)
        self.drop_2 = nn.Dropout2d(p=0.15)
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        self.classifier = nn.Sequential(nn.Linear(deep_features_size, 256),
            nn.ReLU(), nn.Linear(256, num_classes))

    def forward(self, x):
        f, class_f = self.feats(x)
        p = self.psp(f)
        p = self.drop_1(p)
        p = self.up_1(p)
        p = self.drop_2(p)
        p = self.up_2(p)
        p = self.drop_2(p)
        p = self.up_3(p)
        p = self.drop_2(p)
        return self.final(p)


class ConvRelu(nn.Module):

    def __init__(self, in_, out):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class DecoderBlock(nn.Module):

    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(ConvRelu(in_channels, middle_channels),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=3,
            stride=2, padding=1, output_padding=1), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.block(x)


class DecoderBlockV2(nn.Module):

    def __init__(self, in_channels, middle_channels, out_channels,
        is_deconv=True):
        super(DecoderBlockV2, self).__init__()
        self.in_channels = in_channels
        if is_deconv:
            """
                Paramaters for Deconvolution were chosen to avoid artifacts, following
                link https://distill.pub/2016/deconv-checkerboard/
            """
            self.block = nn.Sequential(ConvRelu(in_channels,
                middle_channels), nn.ConvTranspose2d(middle_channels,
                out_channels, kernel_size=4, stride=2, padding=1), nn.ReLU(
                inplace=True))
        else:
            self.block = nn.Sequential(nn.Upsample(scale_factor=2, mode=
                'bilinear'), ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels))

    def forward(self, x):
        return self.block(x)


nonlinearity = nn.ReLU


class DecoderBlockLinkNet(nn.Module):

    def __init__(self, in_channels=512, n_filters=256, kernel_size=3,
        is_deconv=False):
        super().__init__()
        if kernel_size == 3:
            conv_padding = 1
        elif kernel_size == 1:
            conv_padding = 0
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, kernel_size,
            padding=1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity(inplace=True)
        if is_deconv == True:
            self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels //
                4, 3, stride=2, padding=1, output_padding=conv_padding)
        else:
            self.deconv2 = nn.Upsample(scale_factor=2)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity(inplace=True)
        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, kernel_size,
            padding=conv_padding)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class DecoderBlockLinkNetV2(nn.Module):

    def __init__(self, in_channels=512, n_filters=256, kernel_size=4,
        is_deconv=False, is_upsample=True):
        super().__init__()
        self.is_upsample = is_upsample
        if kernel_size == 3:
            conv_stride = 1
        elif kernel_size == 1:
            conv_stride = 1
        elif kernel_size == 4:
            conv_stride = 2
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 3, padding=1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity(inplace=True)
        if is_deconv == True:
            self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels //
                4, kernel_size, stride=conv_stride, padding=1)
        else:
            self.deconv2 = nn.Upsample(scale_factor=2)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity(inplace=True)
        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 3, padding=1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity(inplace=True)

    def forward(self, x):
        if self.is_upsample:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu1(x)
            x = self.deconv2(x)
            x = self.norm2(x)
            x = self.relu2(x)
            x = self.conv3(x)
            x = self.norm3(x)
            x = self.relu3(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu1(x)
            x = self.norm2(x)
            x = self.relu2(x)
            x = self.conv3(x)
            x = self.norm3(x)
            x = self.relu3(x)
        return x


class DecoderBlockLinkNetInceptionV2(nn.Module):

    def __init__(self, in_channels=512, out_channels=512, n_filters=256,
        last_padding=0, kernel_size=3, is_deconv=False):
        super().__init__()
        if kernel_size == 3:
            conv_stride = 1
        elif kernel_size == 1:
            conv_stride = 1
        elif kernel_size == 4:
            conv_stride = 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=2)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nonlinearity(inplace=True)
        if is_deconv == True:
            self.deconv2 = nn.ConvTranspose2d(out_channels, out_channels,
                kernel_size, stride=conv_stride, padding=1)
        else:
            self.deconv2 = nn.Upsample(scale_factor=2)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nonlinearity(inplace=True)
        self.conv3 = nn.Conv2d(out_channels, n_filters, 3, padding=1 +
            last_padding)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=
            kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes, eps=0.001, momentum=0.1,
            affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Mixed_3a(nn.Module):

    def __init__(self):
        super(Mixed_3a, self).__init__()
        self.maxpool = nn.MaxPool2d(3, stride=2)
        self.conv = BasicConv2d(64, 96, kernel_size=3, stride=2)

    def forward(self, x):
        x0 = self.maxpool(x)
        x1 = self.conv(x)
        out = torch.cat((x0, x1), 1)
        return out


class Mixed_4a(nn.Module):

    def __init__(self):
        super(Mixed_4a, self).__init__()
        self.branch0 = nn.Sequential(BasicConv2d(160, 64, kernel_size=1,
            stride=1), BasicConv2d(64, 96, kernel_size=3, stride=1))
        self.branch1 = nn.Sequential(BasicConv2d(160, 64, kernel_size=1,
            stride=1), BasicConv2d(64, 64, kernel_size=(1, 7), stride=1,
            padding=(0, 3)), BasicConv2d(64, 64, kernel_size=(7, 1), stride
            =1, padding=(3, 0)), BasicConv2d(64, 96, kernel_size=(3, 3),
            stride=1))

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        return out


class Mixed_5a(nn.Module):

    def __init__(self):
        super(Mixed_5a, self).__init__()
        self.conv = BasicConv2d(192, 192, kernel_size=3, stride=2)
        self.maxpool = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.conv(x)
        x1 = self.maxpool(x)
        out = torch.cat((x0, x1), 1)
        return out


class Inception_A(nn.Module):

    def __init__(self):
        super(Inception_A, self).__init__()
        self.branch0 = BasicConv2d(384, 96, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(384, 64, kernel_size=1,
            stride=1), BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1))
        self.branch2 = nn.Sequential(BasicConv2d(384, 64, kernel_size=1,
            stride=1), BasicConv2d(64, 96, kernel_size=3, stride=1, padding
            =1), BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1))
        self.branch3 = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False), BasicConv2d(384, 96, kernel_size=1,
            stride=1))

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Reduction_A(nn.Module):

    def __init__(self):
        super(Reduction_A, self).__init__()
        self.branch0 = BasicConv2d(384, 384, kernel_size=3, stride=2)
        self.branch1 = nn.Sequential(BasicConv2d(384, 192, kernel_size=1,
            stride=1), BasicConv2d(192, 224, kernel_size=3, stride=1,
            padding=1), BasicConv2d(224, 256, kernel_size=3, stride=2))
        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Inception_B(nn.Module):

    def __init__(self):
        super(Inception_B, self).__init__()
        self.branch0 = BasicConv2d(1024, 384, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(1024, 192, kernel_size=1,
            stride=1), BasicConv2d(192, 224, kernel_size=(1, 7), stride=1,
            padding=(0, 3)), BasicConv2d(224, 256, kernel_size=(7, 1),
            stride=1, padding=(3, 0)))
        self.branch2 = nn.Sequential(BasicConv2d(1024, 192, kernel_size=1,
            stride=1), BasicConv2d(192, 192, kernel_size=(7, 1), stride=1,
            padding=(3, 0)), BasicConv2d(192, 224, kernel_size=(1, 7),
            stride=1, padding=(0, 3)), BasicConv2d(224, 224, kernel_size=(7,
            1), stride=1, padding=(3, 0)), BasicConv2d(224, 256,
            kernel_size=(1, 7), stride=1, padding=(0, 3)))
        self.branch3 = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False), BasicConv2d(1024, 128, kernel_size=1,
            stride=1))

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Reduction_B(nn.Module):

    def __init__(self):
        super(Reduction_B, self).__init__()
        self.branch0 = nn.Sequential(BasicConv2d(1024, 192, kernel_size=1,
            stride=1), BasicConv2d(192, 192, kernel_size=3, stride=2))
        self.branch1 = nn.Sequential(BasicConv2d(1024, 256, kernel_size=1,
            stride=1), BasicConv2d(256, 256, kernel_size=(1, 7), stride=1,
            padding=(0, 3)), BasicConv2d(256, 320, kernel_size=(7, 1),
            stride=1, padding=(3, 0)), BasicConv2d(320, 320, kernel_size=3,
            stride=2))
        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Inception_C(nn.Module):

    def __init__(self):
        super(Inception_C, self).__init__()
        self.branch0 = BasicConv2d(1536, 256, kernel_size=1, stride=1)
        self.branch1_0 = BasicConv2d(1536, 384, kernel_size=1, stride=1)
        self.branch1_1a = BasicConv2d(384, 256, kernel_size=(1, 3), stride=
            1, padding=(0, 1))
        self.branch1_1b = BasicConv2d(384, 256, kernel_size=(3, 1), stride=
            1, padding=(1, 0))
        self.branch2_0 = BasicConv2d(1536, 384, kernel_size=1, stride=1)
        self.branch2_1 = BasicConv2d(384, 448, kernel_size=(3, 1), stride=1,
            padding=(1, 0))
        self.branch2_2 = BasicConv2d(448, 512, kernel_size=(1, 3), stride=1,
            padding=(0, 1))
        self.branch2_3a = BasicConv2d(512, 256, kernel_size=(1, 3), stride=
            1, padding=(0, 1))
        self.branch2_3b = BasicConv2d(512, 256, kernel_size=(3, 1), stride=
            1, padding=(1, 0))
        self.branch3 = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False), BasicConv2d(1536, 256, kernel_size=1,
            stride=1))

    def forward(self, x):
        x0 = self.branch0(x)
        x1_0 = self.branch1_0(x)
        x1_1a = self.branch1_1a(x1_0)
        x1_1b = self.branch1_1b(x1_0)
        x1 = torch.cat((x1_1a, x1_1b), 1)
        x2_0 = self.branch2_0(x)
        x2_1 = self.branch2_1(x2_0)
        x2_2 = self.branch2_2(x2_1)
        x2_3a = self.branch2_3a(x2_2)
        x2_3b = self.branch2_3b(x2_2)
        x2 = torch.cat((x2_3a, x2_3b), 1)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class InceptionV4(nn.Module):

    def __init__(self, num_classes=1001):
        super(InceptionV4, self).__init__()
        self.input_space = None
        self.input_size = 299, 299, 3
        self.mean = None
        self.std = None
        self.features = nn.Sequential(BasicConv2d(3, 32, kernel_size=3,
            stride=2), BasicConv2d(32, 32, kernel_size=3, stride=1),
            BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1),
            Mixed_3a(), Mixed_4a(), Mixed_5a(), Inception_A(), Inception_A(
            ), Inception_A(), Inception_A(), Reduction_A(), Inception_B(),
            Inception_B(), Inception_B(), Inception_B(), Inception_B(),
            Inception_B(), Inception_B(), Reduction_B(), Inception_C(),
            Inception_C(), Inception_C())
        self.avg_pool = nn.AvgPool2d(8, count_include_pad=False)
        self.last_linear = nn.Linear(1536, num_classes)

    def logits(self, features):
        x = self.avg_pool(features)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x


class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=
            kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes, eps=0.001, momentum=0.1,
            affine=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Mixed_5b(nn.Module):

    def __init__(self):
        super(Mixed_5b, self).__init__()
        self.branch0 = BasicConv2d(192, 96, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(192, 48, kernel_size=1,
            stride=1), BasicConv2d(48, 64, kernel_size=5, stride=1, padding=2))
        self.branch2 = nn.Sequential(BasicConv2d(192, 64, kernel_size=1,
            stride=1), BasicConv2d(64, 96, kernel_size=3, stride=1, padding
            =1), BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1))
        self.branch3 = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False), BasicConv2d(192, 64, kernel_size=1,
            stride=1))

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Block35(nn.Module):

    def __init__(self, scale=1.0):
        super(Block35, self).__init__()
        self.scale = scale
        self.branch0 = BasicConv2d(320, 32, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(320, 32, kernel_size=1,
            stride=1), BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1))
        self.branch2 = nn.Sequential(BasicConv2d(320, 32, kernel_size=1,
            stride=1), BasicConv2d(32, 48, kernel_size=3, stride=1, padding
            =1), BasicConv2d(48, 64, kernel_size=3, stride=1, padding=1))
        self.conv2d = nn.Conv2d(128, 320, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Mixed_6a(nn.Module):

    def __init__(self):
        super(Mixed_6a, self).__init__()
        self.branch0 = BasicConv2d(320, 384, kernel_size=3, stride=2)
        self.branch1 = nn.Sequential(BasicConv2d(320, 256, kernel_size=1,
            stride=1), BasicConv2d(256, 256, kernel_size=3, stride=1,
            padding=1), BasicConv2d(256, 384, kernel_size=3, stride=2))
        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Block17(nn.Module):

    def __init__(self, scale=1.0):
        super(Block17, self).__init__()
        self.scale = scale
        self.branch0 = BasicConv2d(1088, 192, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(1088, 128, kernel_size=1,
            stride=1), BasicConv2d(128, 160, kernel_size=(1, 7), stride=1,
            padding=(0, 3)), BasicConv2d(160, 192, kernel_size=(7, 1),
            stride=1, padding=(3, 0)))
        self.conv2d = nn.Conv2d(384, 1088, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Mixed_7a(nn.Module):

    def __init__(self):
        super(Mixed_7a, self).__init__()
        self.branch0 = nn.Sequential(BasicConv2d(1088, 256, kernel_size=1,
            stride=1), BasicConv2d(256, 384, kernel_size=3, stride=2))
        self.branch1 = nn.Sequential(BasicConv2d(1088, 256, kernel_size=1,
            stride=1), BasicConv2d(256, 288, kernel_size=3, stride=2))
        self.branch2 = nn.Sequential(BasicConv2d(1088, 256, kernel_size=1,
            stride=1), BasicConv2d(256, 288, kernel_size=3, stride=1,
            padding=1), BasicConv2d(288, 320, kernel_size=3, stride=2))
        self.branch3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Block8(nn.Module):

    def __init__(self, scale=1.0, noReLU=False):
        super(Block8, self).__init__()
        self.scale = scale
        self.noReLU = noReLU
        self.branch0 = BasicConv2d(2080, 192, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(2080, 192, kernel_size=1,
            stride=1), BasicConv2d(192, 224, kernel_size=(1, 3), stride=1,
            padding=(0, 1)), BasicConv2d(224, 256, kernel_size=(3, 1),
            stride=1, padding=(1, 0)))
        self.conv2d = nn.Conv2d(448, 2080, kernel_size=1, stride=1)
        if not self.noReLU:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        if not self.noReLU:
            out = self.relu(out)
        return out


class InceptionResNetV2(nn.Module):

    def __init__(self, num_classes=1001):
        super(InceptionResNetV2, self).__init__()
        self.input_space = None
        self.input_size = 299, 299, 3
        self.mean = None
        self.std = None
        self.conv2d_1a = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1)
        self.conv2d_2b = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1
            )
        self.maxpool_3a = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1, stride=1)
        self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3, stride=1)
        self.maxpool_5a = nn.MaxPool2d(3, stride=2)
        self.mixed_5b = Mixed_5b()
        self.repeat = nn.Sequential(Block35(scale=0.17), Block35(scale=0.17
            ), Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17
            ), Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17
            ), Block35(scale=0.17), Block35(scale=0.17))
        self.mixed_6a = Mixed_6a()
        self.repeat_1 = nn.Sequential(Block17(scale=0.1), Block17(scale=0.1
            ), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1),
            Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1),
            Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1),
            Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1),
            Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1),
            Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1))
        self.mixed_7a = Mixed_7a()
        self.repeat_2 = nn.Sequential(Block8(scale=0.2), Block8(scale=0.2),
            Block8(scale=0.2), Block8(scale=0.2), Block8(scale=0.2), Block8
            (scale=0.2), Block8(scale=0.2), Block8(scale=0.2), Block8(scale
            =0.2))
        self.block8 = Block8(noReLU=True)
        self.conv2d_7b = BasicConv2d(2080, 1536, kernel_size=1, stride=1)
        self.avgpool_1a = nn.AvgPool2d(8, count_include_pad=False)
        self.last_linear = nn.Linear(1536, num_classes)

    def features(self, input):
        x = self.conv2d_1a(input)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.maxpool_5a(x)
        x = self.mixed_5b(x)
        x = self.repeat(x)
        x = self.mixed_6a(x)
        x = self.repeat_1(x)
        x = self.mixed_7a(x)
        x = self.repeat_2(x)
        x = self.block8(x)
        x = self.conv2d_7b(x)
        return x

    def logits(self, features):
        x = self.avgpool_1a(features)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x


class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=
            kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes, eps=0.001, momentum=0.1,
            affine=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class LinkNet18(nn.Module):

    def __init__(self, num_classes, pretrained=True, num_channels=3,
        is_deconv=False, decoder_kernel_size=4, **kwargs):
        super().__init__()
        filters = [64, 128, 256, 512]
        resnet = models.resnet18(pretrained=pretrained)
        self.mean = 0.485, 0.456, 0.406
        self.std = 0.229, 0.224, 0.225
        if num_channels == 3:
            self.firstconv = resnet.conv1
        else:
            self.firstconv = nn.Conv2d(num_channels, 64, kernel_size=(7, 7),
                stride=(2, 2), padding=(3, 3))
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        self.decoder4 = DecoderBlock(in_channels=filters[3], n_filters=
            filters[2], kernel_size=decoder_kernel_size, is_deconv=is_deconv)
        self.decoder3 = DecoderBlock(in_channels=filters[2], n_filters=
            filters[1], kernel_size=decoder_kernel_size, is_deconv=is_deconv)
        self.decoder2 = DecoderBlock(in_channels=filters[1], n_filters=
            filters[0], kernel_size=decoder_kernel_size, is_deconv=is_deconv)
        self.decoder1 = DecoderBlock(in_channels=filters[0], n_filters=
            filters[0], kernel_size=decoder_kernel_size, is_deconv=is_deconv)
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nonlinearity(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nonlinearity(inplace=True)
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

    def freeze(self):
        self.require_encoder_grad(False)

    def unfreeze(self):
        self.require_encoder_grad(True)

    def require_encoder_grad(self, requires_grad):
        blocks = [self.firstconv, self.encoder1, self.encoder2, self.
            encoder3, self.encoder4]
        for block in blocks:
            for p in block.parameters():
                p.requires_grad = requires_grad

    def forward(self, x):
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        f1 = self.finaldeconv1(d1)
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        f5 = self.finalconv3(f4)
        return f5


class LinkNet34(nn.Module):

    def __init__(self, num_classes, pretrained=True, num_channels=3,
        is_deconv=False, decoder_kernel_size=4, **kwargs):
        super().__init__()
        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=pretrained)
        self.mean = 0.485, 0.456, 0.406
        self.std = 0.229, 0.224, 0.225
        if num_channels == 3:
            self.firstconv = resnet.conv1
        else:
            self.firstconv = nn.Conv2d(num_channels, 64, kernel_size=(7, 7),
                stride=(2, 2), padding=(3, 3))
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        self.decoder4 = DecoderBlock(in_channels=filters[3], n_filters=
            filters[2], kernel_size=decoder_kernel_size, is_deconv=is_deconv)
        self.decoder3 = DecoderBlock(in_channels=filters[2], n_filters=
            filters[1], kernel_size=decoder_kernel_size, is_deconv=is_deconv)
        self.decoder2 = DecoderBlock(in_channels=filters[1], n_filters=
            filters[0], kernel_size=decoder_kernel_size, is_deconv=is_deconv)
        self.decoder1 = DecoderBlock(in_channels=filters[0], n_filters=
            filters[0], kernel_size=decoder_kernel_size, is_deconv=is_deconv)
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nonlinearity(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nonlinearity(inplace=True)
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

    def freeze(self):
        self.require_encoder_grad(False)

    def unfreeze(self):
        self.require_encoder_grad(True)

    def require_encoder_grad(self, requires_grad):
        blocks = [self.firstconv, self.encoder1, self.encoder2, self.
            encoder3, self.encoder4]
        for block in blocks:
            for p in block.parameters():
                p.requires_grad = requires_grad

    def forward(self, x):
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        f1 = self.finaldeconv1(d1)
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        f5 = self.finalconv3(f4)
        return f5


class LinkNet50(nn.Module):

    def __init__(self, num_classes, pretrained=True, num_channels=3,
        is_deconv=False, decoder_kernel_size=4, **kwargs):
        super().__init__()
        filters = [256, 512, 1024, 2048]
        resnet = models.resnet50(pretrained=pretrained)
        self.mean = 0.485, 0.456, 0.406
        self.std = 0.229, 0.224, 0.225
        if num_channels == 3:
            self.firstconv = resnet.conv1
        else:
            self.firstconv = nn.Conv2d(num_channels, 64, kernel_size=(7, 7),
                stride=(2, 2), padding=(3, 3))
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        self.decoder4 = DecoderBlock(in_channels=filters[3], n_filters=
            filters[2], kernel_size=decoder_kernel_size, is_deconv=is_deconv)
        self.decoder3 = DecoderBlock(in_channels=filters[2], n_filters=
            filters[1], kernel_size=decoder_kernel_size, is_deconv=is_deconv)
        self.decoder2 = DecoderBlock(in_channels=filters[1], n_filters=
            filters[0], kernel_size=decoder_kernel_size, is_deconv=is_deconv)
        self.decoder1 = DecoderBlock(in_channels=filters[0], n_filters=
            filters[0], kernel_size=decoder_kernel_size, is_deconv=is_deconv)
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nonlinearity(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nonlinearity(inplace=True)
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

    def freeze(self):
        self.require_encoder_grad(False)

    def unfreeze(self):
        self.require_encoder_grad(True)

    def require_encoder_grad(self, requires_grad):
        blocks = [self.firstconv, self.encoder1, self.encoder2, self.
            encoder3, self.encoder4]
        for block in blocks:
            for p in block.parameters():
                p.requires_grad = requires_grad

    def forward(self, x):
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        f1 = self.finaldeconv1(d1)
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        f5 = self.finalconv3(f4)
        return f5


class LinkNet101(nn.Module):

    def __init__(self, num_classes, pretrained=True, num_channels=3,
        is_deconv=False, decoder_kernel_size=4, **kwargs):
        super().__init__()
        filters = [256, 512, 1024, 2048]
        resnet = models.resnet101(pretrained=pretrained)
        self.mean = 0.485, 0.456, 0.406
        self.std = 0.229, 0.224, 0.225
        if num_channels == 3:
            self.firstconv = resnet.conv1
        else:
            self.firstconv = nn.Conv2d(num_channels, 64, kernel_size=(7, 7),
                stride=(2, 2), padding=(3, 3))
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        self.decoder4 = DecoderBlock(in_channels=filters[3], n_filters=
            filters[2], kernel_size=decoder_kernel_size, is_deconv=is_deconv)
        self.decoder3 = DecoderBlock(in_channels=filters[2], n_filters=
            filters[1], kernel_size=decoder_kernel_size, is_deconv=is_deconv)
        self.decoder2 = DecoderBlock(in_channels=filters[1], n_filters=
            filters[0], kernel_size=decoder_kernel_size, is_deconv=is_deconv)
        self.decoder1 = DecoderBlock(in_channels=filters[0], n_filters=
            filters[0], kernel_size=decoder_kernel_size, is_deconv=is_deconv)
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nonlinearity(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nonlinearity(inplace=True)
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

    def freeze(self):
        self.require_encoder_grad(False)

    def unfreeze(self):
        self.require_encoder_grad(True)

    def require_encoder_grad(self, requires_grad):
        blocks = [self.firstconv, self.encoder1, self.encoder2, self.
            encoder3, self.encoder4]
        for block in blocks:
            for p in block.parameters():
                p.requires_grad = requires_grad

    def forward(self, x):
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        f1 = self.finaldeconv1(d1)
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        f5 = self.finalconv3(f4)
        return f5


def resnext101_32x4d(num_classes=1000, pretrained='imagenet'):
    model = ResNeXt101_32x4d(num_classes=num_classes)
    model_blob = ResNeXt101_32x4d_blob(num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['resnext101_32x4d'][pretrained]
        assert num_classes == settings['num_classes'
            ], 'num_classes should be {}, but is {}'.format(settings[
            'num_classes'], num_classes)
        model_blob.load_state_dict(model_zoo.load_url(settings['url']))
        model.stem = nn.Sequential(model_blob.features[0], model_blob.
            features[1], model_blob.features[2], model_blob.features[3])
        model.layer1 = nn.Sequential(model_blob.features[4])
        model.layer2 = nn.Sequential(model_blob.features[5])
        model.layer3 = nn.Sequential(model_blob.features[6])
        model.layer4 = nn.Sequential(model_blob.features[7])
        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']
    return model


class LinkNeXt(nn.Module):

    def __init__(self, num_classes, pretrained=True, num_channels=3,
        is_deconv=False, decoder_kernel_size=4, **kwargs):
        super().__init__()
        filters = [256, 512, 1024, 2048]
        resnet = resnext101_32x4d(num_classes=1000, pretrained='imagenet')
        self.mean = 0.485, 0.456, 0.406
        self.std = 0.229, 0.224, 0.225
        self.stem = resnet.stem
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        self.decoder4 = DecoderBlock(in_channels=filters[3], n_filters=
            filters[2], kernel_size=decoder_kernel_size, is_deconv=is_deconv)
        self.decoder3 = DecoderBlock(in_channels=filters[2], n_filters=
            filters[1], kernel_size=decoder_kernel_size, is_deconv=is_deconv)
        self.decoder2 = DecoderBlock(in_channels=filters[1], n_filters=
            filters[0], kernel_size=decoder_kernel_size, is_deconv=is_deconv)
        self.decoder1 = DecoderBlock(in_channels=filters[0], n_filters=
            filters[0], kernel_size=decoder_kernel_size, is_deconv=is_deconv)
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nonlinearity(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nonlinearity(inplace=True)
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

    def freeze(self):
        self.require_encoder_grad(False)

    def unfreeze(self):
        self.require_encoder_grad(True)

    def require_encoder_grad(self, requires_grad):
        blocks = [self.stem, self.encoder1, self.encoder2, self.encoder3,
            self.encoder4]
        for block in blocks:
            for p in block.parameters():
                p.requires_grad = requires_grad

    def forward(self, x):
        x = self.stem(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        f1 = self.finaldeconv1(d1)
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        f5 = self.finalconv3(f4)
        return f5


class LinkNet152(nn.Module):

    def __init__(self, num_classes, pretrained=True, num_channels=3,
        is_deconv=False, decoder_kernel_size=3, **kwargs):
        super().__init__()
        filters = [256, 512, 1024, 2048]
        resnet = models.resnet152(pretrained=pretrained)
        self.mean = 0.485, 0.456, 0.406
        self.std = 0.229, 0.224, 0.225
        if num_channels == 3:
            self.firstconv = resnet.conv1
        else:
            self.firstconv = nn.Conv2d(num_channels, 64, kernel_size=(7, 7),
                stride=(2, 2), padding=(3, 3))
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        self.decoder4 = DecoderBlock(in_channels=filters[3], n_filters=
            filters[2], kernel_size=decoder_kernel_size, is_deconv=is_deconv)
        self.decoder3 = DecoderBlock(in_channels=filters[2], n_filters=
            filters[1], kernel_size=decoder_kernel_size, is_deconv=is_deconv)
        self.decoder2 = DecoderBlock(in_channels=filters[1], n_filters=
            filters[0], kernel_size=decoder_kernel_size, is_deconv=is_deconv)
        self.decoder1 = DecoderBlock(in_channels=filters[0], n_filters=
            filters[0], kernel_size=decoder_kernel_size, is_deconv=is_deconv)
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nonlinearity(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nonlinearity(inplace=True)
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

    def freeze(self):
        self.require_encoder_grad(False)

    def unfreeze(self):
        self.require_encoder_grad(True)

    def require_encoder_grad(self, requires_grad):
        blocks = [self.firstconv, self.encoder1, self.encoder2, self.
            encoder3, self.encoder4]
        for block in blocks:
            for p in block.parameters():
                p.requires_grad = requires_grad

    def forward(self, x):
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        f1 = self.finaldeconv1(d1)
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        f5 = self.finalconv3(f4)
        return f5


def inceptionv4(pretrained='imagenet'):
    """Pretrained InceptionV4 model"""
    settings = pretrained_settings['inceptionv4'][pretrained]
    model = InceptionV4(num_classes=1001)
    model.load_state_dict(model_zoo.load_url(settings['url']))
    if pretrained == 'imagenet':
        new_last_linear = nn.Linear(1536, settings['num_classes'])
        new_last_linear.weight.data = model.last_linear.weight.data[1:]
        new_last_linear.bias.data = model.last_linear.bias.data[1:]
        model.last_linear = new_last_linear
    model.input_space = settings['input_space']
    model.input_size = settings['input_size']
    model.input_range = settings['input_range']
    model.mean = settings['mean']
    model.std = settings['std']
    return model


class LinkCeption(nn.Module):

    def __init__(self, num_classes, pretrained=True, num_channels=3,
        is_deconv=False, decoder_kernel_size=4, **kwargs):
        super().__init__()
        self.mean = 0.5, 0.5, 0.5
        self.std = 0.5, 0.5, 0.5
        filters = [64, 384, 384, 1024, 1536]
        inception = inceptionv4(pretrained='imagenet')
        if num_channels == 3:
            self.stem1 = nn.Sequential(inception.features[0], inception.
                features[1], inception.features[2])
        else:
            self.stem1 = nn.Sequential(BasicConv2d(num_channels, 32,
                kernel_size=3, stride=2), inception.features[1], inception.
                features[2])
        self.stem2 = nn.Sequential(inception.features[3], inception.
            features[4], inception.features[5])
        self.block1 = nn.Sequential(inception.features[6], inception.
            features[7], inception.features[8], inception.features[9])
        self.tr1 = inception.features[10]
        self.block2 = nn.Sequential(inception.features[11], inception.
            features[12], inception.features[13], inception.features[14],
            inception.features[15], inception.features[16], inception.
            features[17])
        self.tr2 = inception.features[18]
        self.block3 = nn.Sequential(inception.features[19], inception.
            features[20], inception.features[21])
        self.decoder4 = DecoderBlockInception(in_channels=filters[4],
            out_channels=filters[3], n_filters=filters[3], last_padding=0,
            kernel_size=decoder_kernel_size, is_deconv=is_deconv)
        self.decoder3 = DecoderBlockInception(in_channels=filters[3],
            out_channels=filters[2], n_filters=filters[2], last_padding=0,
            kernel_size=decoder_kernel_size, is_deconv=is_deconv)
        self.decoder2 = DecoderBlockInception(in_channels=filters[2],
            out_channels=filters[1], n_filters=filters[1], last_padding=0,
            kernel_size=decoder_kernel_size, is_deconv=is_deconv)
        self.decoder1 = DecoderBlockInception(in_channels=filters[1],
            out_channels=filters[0], n_filters=filters[0], last_padding=0,
            kernel_size=decoder_kernel_size, is_deconv=is_deconv)
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 1, stride=2)
        self.finalnorm1 = nn.BatchNorm2d(32)
        self.finalrelu1 = nonlinearity(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalnorm2 = nn.BatchNorm2d(32)
        self.finalrelu2 = nonlinearity(inplace=True)
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=0)

    def freeze(self):
        self.require_encoder_grad(False)

    def unfreeze(self):
        self.require_encoder_grad(True)

    def require_encoder_grad(self, requires_grad):
        blocks = [self.stem1, self.stem2, self.block1, self.tr1, self.
            block2, self.tr2, self.block3]
        for block in blocks:
            for p in block.parameters():
                p.requires_grad = requires_grad

    def forward(self, x):
        final_shape = x.shape[2:]
        x = self.stem1(x)
        e1 = self.stem2(x)
        e2 = self.block1(e1)
        e3 = self.tr1(e2)
        e3 = self.block2(e3)
        e4 = self.tr2(e3)
        e4 = self.block3(e4)
        d4 = self.decoder4(e4)[:, :, 0:e3.size(2), 0:e3.size(3)] + e3
        d3 = self.decoder3(d4)[:, :, 0:e2.size(2), 0:e2.size(3)] + e2
        d2 = self.decoder2(d3)[:, :, 0:self.decoder2(e1).size(2), 0:self.
            decoder2(e1).size(3)] + self.decoder2(e1)
        d1 = self.decoder1(d2)
        f1 = self.finaldeconv1(d1)
        f1 = self.finalnorm1(f1)
        f2 = self.finalrelu1(f1)
        f2 = self.finalnorm2(f2)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        f5 = self.finalconv3(f4)
        out = F.interpolate(f5, size=final_shape, mode='bilinear')
        return out


def inceptionresnetv2(num_classes=1001, pretrained='imagenet'):
    """InceptionResNetV2 model architecture from the
    `"InceptionV4, Inception-ResNet..." <https://arxiv.org/abs/1602.07261>`_ paper.
    """
    if pretrained:
        settings = pretrained_settings['inceptionresnetv2'][pretrained]
        assert num_classes == settings['num_classes'
            ], 'num_classes should be {}, but is {}'.format(settings[
            'num_classes'], num_classes)
        model = InceptionResNetV2(num_classes=1001)
        model.load_state_dict(model_zoo.load_url(settings['url']))
        if pretrained == 'imagenet':
            new_last_linear = nn.Linear(1536, 1000)
            new_last_linear.weight.data = model.last_linear.weight.data[1:]
            new_last_linear.bias.data = model.last_linear.bias.data[1:]
            model.last_linear = new_last_linear
        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']
    else:
        model = InceptionResNetV2(num_classes=num_classes)
    return model


class LinkInceptionResNet(nn.Module):

    def __init__(self, num_classes, pretrained=True, num_channels=3,
        is_deconv=False, decoder_kernel_size=3, **kwargs):
        super().__init__()
        self.mean = 0.485, 0.456, 0.406
        self.std = 0.229, 0.224, 0.225
        filters = [64, 192, 320, 1088, 2080]
        ir = inceptionresnetv2(pretrained='imagenet', num_classes=1000)
        if num_channels == 3:
            self.stem1 = nn.Sequential(ir.conv2d_1a, ir.conv2d_2a, ir.conv2d_2b
                )
        else:
            self.stem1 = nn.Sequential(BasicConv2d(num_channels, 32,
                kernel_size=3, stride=2), ir.conv2d_2a, ir.conv2d_2b)
        self.maxpool_3a = ir.maxpool_3a
        self.stem2 = nn.Sequential(ir.conv2d_3b, ir.conv2d_4a)
        self.maxpool_5a = ir.maxpool_5a
        self.mixed_5b = ir.mixed_5b
        self.mixed_6a = ir.mixed_6a
        self.mixed_7a = ir.mixed_7a
        self.skip1 = ir.repeat
        self.skip2 = ir.repeat_1
        self.skip3 = ir.repeat_2
        self.decoder3 = DecoderBlockInception(in_channels=filters[4],
            out_channels=filters[3], n_filters=filters[3], last_padding=0,
            kernel_size=decoder_kernel_size, is_deconv=is_deconv)
        self.decoder2 = DecoderBlockInception(in_channels=filters[3],
            out_channels=filters[2], n_filters=filters[2], last_padding=0,
            kernel_size=decoder_kernel_size, is_deconv=is_deconv)
        self.decoder1 = DecoderBlockInception(in_channels=filters[2],
            out_channels=filters[1], n_filters=filters[1], last_padding=0,
            kernel_size=decoder_kernel_size, is_deconv=is_deconv)
        self.decoder0 = DecoderBlockInception(in_channels=filters[1],
            out_channels=filters[0], n_filters=filters[0], last_padding=2,
            kernel_size=decoder_kernel_size, is_deconv=is_deconv)
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalnorm1 = nn.BatchNorm2d(32)
        self.finalrelu1 = nonlinearity(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalnorm2 = nn.BatchNorm2d(32)
        self.finalrelu2 = nonlinearity(inplace=True)
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

    def freeze(self):
        self.require_encoder_grad(False)

    def unfreeze(self):
        self.require_encoder_grad(True)

    def require_encoder_grad(self, requires_grad):
        blocks = [self.stem1, self.stem2, self.mixed_5b, self.mixed_6a,
            self.mixed_7a, self.skip1, self.skip2, self.skip3]
        for block in blocks:
            for p in block.parameters():
                p.requires_grad = requires_grad

    def forward(self, x):
        x = self.stem1(x)
        x1 = self.maxpool_3a(x)
        x1 = self.stem2(x1)
        x2 = self.maxpool_3a(x1)
        x2 = self.mixed_5b(x2)
        e1 = self.skip1(x2)
        e1_resume = self.mixed_6a(e1)
        e2 = self.skip2(e1_resume)
        e2_resume = self.mixed_7a(e2)
        e3 = self.skip3(e2_resume)
        d3 = self.decoder3(e3)[:, :, 0:e2.size(2), 0:e2.size(3)] + e2
        d2 = self.decoder2(d3)[:, :, 0:e1.size(2), 0:e1.size(3)] + e1
        d1 = self.decoder1(d2)[:, :, 0:x1.size(2), 0:x1.size(3)] + x1
        d0 = self.decoder0(d1)
        f1 = self.finaldeconv1(d0)
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        f5 = self.finalconv3(f4)
        return f5


class LinkDenseNet161(nn.Module):

    def __init__(self, num_classes, pretrained=True, num_channels=3,
        is_deconv=False, decoder_kernel_size=4, **kwargs):
        super().__init__()
        filters = [384, 768, 2112, 2208]
        densenet = models.densenet161(pretrained=pretrained)
        self.mean = 0.485, 0.456, 0.406
        self.std = 0.229, 0.224, 0.225
        if num_channels == 3:
            self.firstconv = densenet.features.conv0
        else:
            self.firstconv = nn.Conv2d(num_channels, 64, kernel_size=(7, 7),
                stride=(2, 2), padding=(3, 3))
        self.stem = nn.Sequential(self.firstconv, densenet.features.norm0,
            densenet.features.relu0, densenet.features.pool0)
        self.encoder1 = nn.Sequential(densenet.features.denseblock1)
        self.encoder2 = nn.Sequential(densenet.features.transition1,
            densenet.features.denseblock2)
        self.encoder3 = nn.Sequential(densenet.features.transition2,
            densenet.features.denseblock3)
        self.encoder4 = nn.Sequential(densenet.features.transition3,
            densenet.features.denseblock4)
        self.decoder4 = DecoderBlock(in_channels=filters[3], n_filters=
            filters[2], kernel_size=decoder_kernel_size, is_deconv=is_deconv)
        self.decoder3 = DecoderBlock(in_channels=filters[2], n_filters=
            filters[1], kernel_size=decoder_kernel_size, is_deconv=is_deconv)
        self.decoder2 = DecoderBlock(in_channels=filters[1], n_filters=
            filters[0], kernel_size=decoder_kernel_size, is_deconv=is_deconv)
        self.decoder1 = DecoderBlock(in_channels=filters[0], n_filters=
            filters[0], kernel_size=decoder_kernel_size, is_deconv=is_deconv)
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nonlinearity(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nonlinearity(inplace=True)
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

    def require_encoder_grad(self, requires_grad):
        blocks = [self.stem, self.encoder1, self.encoder2, self.encoder3,
            self.encoder4]
        for block in blocks:
            for p in block.parameters():
                p.requires_grad = requires_grad

    def freeze(self):
        self.require_encoder_grad(False)

    def unfreeze(self):
        self.require_encoder_grad(True)

    def forward(self, x):
        x = self.stem(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        f1 = self.finaldeconv1(d1)
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        f5 = self.finalconv3(f4)
        return f5


class LinkDenseNet121(nn.Module):

    def __init__(self, num_classes, pretrained=True, num_channels=3,
        is_deconv=False, decoder_kernel_size=4, **kwargs):
        super().__init__()
        filters = [256, 512, 1024, 1024]
        densenet = models.densenet121(pretrained=pretrained)
        self.mean = 0.485, 0.456, 0.406
        self.std = 0.229, 0.224, 0.225
        if num_channels == 3:
            self.firstconv = densenet.features.conv0
        else:
            self.firstconv = nn.Conv2d(num_channels, 64, kernel_size=(7, 7),
                stride=(2, 2), padding=(3, 3))
        self.stem = nn.Sequential(self.firstconv, densenet.features.norm0,
            densenet.features.relu0, densenet.features.pool0)
        self.encoder1 = nn.Sequential(densenet.features.denseblock1)
        self.encoder2 = nn.Sequential(densenet.features.transition1,
            densenet.features.denseblock2)
        self.encoder3 = nn.Sequential(densenet.features.transition2,
            densenet.features.denseblock3)
        self.encoder4 = nn.Sequential(densenet.features.transition3,
            densenet.features.denseblock4)
        self.decoder4 = DecoderBlock(in_channels=filters[3], n_filters=
            filters[2], kernel_size=decoder_kernel_size, is_deconv=is_deconv)
        self.decoder3 = DecoderBlock(in_channels=filters[2], n_filters=
            filters[1], kernel_size=decoder_kernel_size, is_deconv=is_deconv)
        self.decoder2 = DecoderBlock(in_channels=filters[1], n_filters=
            filters[0], kernel_size=decoder_kernel_size, is_deconv=is_deconv)
        self.decoder1 = DecoderBlock(in_channels=filters[0], n_filters=
            filters[0], kernel_size=decoder_kernel_size, is_deconv=is_deconv)
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nonlinearity(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nonlinearity(inplace=True)
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

    def require_encoder_grad(self, requires_grad):
        blocks = [self.stem, self.encoder1, self.encoder2, self.encoder3,
            self.encoder4]
        for block in blocks:
            for p in block.parameters():
                p.requires_grad = requires_grad

    def freeze(self):
        self.require_encoder_grad(False)

    def unfreeze(self):
        self.require_encoder_grad(True)

    def forward(self, x):
        x = self.stem(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        f1 = self.finaldeconv1(d1)
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        f5 = self.finalconv3(f4)
        return f5


class CoarseLinkNet50(nn.Module):

    def __init__(self, num_classes, pretrained=True, num_channels=3,
        is_deconv=False, decoder_kernel_size=4, **kwargs):
        super().__init__()
        filters = [256, 512, 1024, 2048]
        resnet = models.resnet50(pretrained=pretrained)
        self.mean = 0.485, 0.456, 0.406
        self.std = 0.229, 0.224, 0.225
        if num_channels == 3:
            self.firstconv = resnet.conv1
        else:
            self.firstconv = nn.Conv2d(num_channels, 64, kernel_size=(7, 7),
                stride=(2, 2), padding=(3, 3))
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        self.decoder4 = DecoderBlock(in_channels=filters[3], n_filters=
            filters[2], kernel_size=decoder_kernel_size, is_deconv=is_deconv)
        self.decoder3 = DecoderBlock(in_channels=filters[2], n_filters=
            filters[1], kernel_size=decoder_kernel_size, is_deconv=is_deconv)
        self.decoder2 = DecoderBlock(in_channels=filters[1], n_filters=
            filters[0], kernel_size=decoder_kernel_size, is_deconv=is_deconv)
        self.decoder1 = DecoderBlock(in_channels=filters[0], n_filters=
            filters[0], kernel_size=decoder_kernel_size, is_deconv=is_deconv)
        self.finalconv1 = nn.Conv2d(filters[0], 32, 2, padding=1)
        self.finalrelu1 = nonlinearity(inplace=True)
        self.finalconv2 = nn.Conv2d(32, num_classes, 2, padding=1)

    def freeze(self):
        self.require_encoder_grad(False)

    def unfreeze(self):
        self.require_encoder_grad(True)

    def require_encoder_grad(self, requires_grad):
        blocks = [self.firstconv, self.encoder1, self.encoder2, self.
            encoder3, self.encoder4]
        for block in blocks:
            for p in block.parameters():
                p.requires_grad = requires_grad

    def forward(self, x):
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        f1 = self.finalconv1(d1)
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)
        return f3


class ResNeXt101_32x4d_blob(nn.Module):

    def __init__(self, num_classes=1000):
        super(ResNeXt101_32x4d_blob, self).__init__()
        self.num_classes = num_classes
        resnext = resnext101_32x4d_features_blob()
        self.features = resnext.resnext101_32x4d_features
        self.avg_pool = nn.AvgPool2d((7, 7), (1, 1))
        self.last_linear = nn.Linear(2048, num_classes)

    def logits(self, input):
        x = self.avg_pool(input)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x


class ResNeXt101_32x4d(nn.Module):

    def __init__(self, num_classes=1000):
        super(ResNeXt101_32x4d, self).__init__()
        self.num_classes = num_classes
        resnext = resnext101_32x4d_features()
        self.stem = resnext.resnext101_32x4d_stem
        self.layer1 = resnext.resnext101_32x4d_layer1
        self.layer2 = resnext.resnext101_32x4d_layer2
        self.layer3 = resnext.resnext101_32x4d_layer3
        self.layer4 = resnext.resnext101_32x4d_layer4
        self.avg_pool = nn.AvgPool2d((7, 7), (1, 1))
        self.last_linear = nn.Linear(2048, num_classes)

    def logits(self, input):
        x = self.avg_pool(input)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.stem(input)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.logits(x)
        return x


class LambdaBase(nn.Sequential):

    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input


class resnext101_32x4d_features_blob(nn.Module):

    def __init__(self):
        super(resnext101_32x4d_features_blob, self).__init__()
        self.resnext101_32x4d_features = nn.Sequential(nn.Conv2d(3, 64, (7,
            7), (2, 2), (3, 3), 1, 1, bias=False), nn.BatchNorm2d(64), nn.
            ReLU(), nn.MaxPool2d((3, 3), (2, 2), (1, 1)), nn.Sequential(nn.
            Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(
            nn.Conv2d(64, 128, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(), nn.Conv2d(128, 128, (3, 3), (1,
            1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(128), nn.ReLU()),
            nn.Conv2d(128, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
            nn.BatchNorm2d(256)), nn.Sequential(nn.Conv2d(64, 256, (1, 1),
            (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(256))),
            LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(
            LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(
            256, 128, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.
            BatchNorm2d(128), nn.ReLU(), nn.Conv2d(128, 128, (3, 3), (1, 1),
            (1, 1), 1, 32, bias=False), nn.BatchNorm2d(128), nn.ReLU()), nn
            .Conv2d(128, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn
            .BatchNorm2d(256)), Lambda(lambda x: x)), LambdaReduce(lambda x,
            y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.
            Sequential(nn.Sequential(nn.Conv2d(256, 128, (1, 1), (1, 1), (0,
            0), 1, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(), nn.
            Conv2d(128, 128, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn
            .BatchNorm2d(128), nn.ReLU()), nn.Conv2d(128, 256, (1, 1), (1, 
            1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(256)), Lambda(lambda
            x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU())), nn.
            Sequential(nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(
            nn.Sequential(nn.Conv2d(256, 256, (1, 1), (1, 1), (0, 0), 1, 1,
            bias=False), nn.BatchNorm2d(256), nn.ReLU(), nn.Conv2d(256, 256,
            (3, 3), (2, 2), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(256),
            nn.ReLU()), nn.Conv2d(256, 512, (1, 1), (1, 1), (0, 0), 1, 1,
            bias=False), nn.BatchNorm2d(512)), nn.Sequential(nn.Conv2d(256,
            512, (1, 1), (2, 2), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(
            512))), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.
            Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(
            nn.Conv2d(512, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(), nn.Conv2d(256, 256, (3, 3), (1,
            1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(256), nn.ReLU()),
            nn.Conv2d(256, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
            nn.BatchNorm2d(512)), Lambda(lambda x: x)), LambdaReduce(lambda
            x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x,
            nn.Sequential(nn.Sequential(nn.Conv2d(512, 256, (1, 1), (1, 1),
            (0, 0), 1, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(), nn.
            Conv2d(256, 256, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn
            .BatchNorm2d(256), nn.ReLU()), nn.Conv2d(256, 512, (1, 1), (1, 
            1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512)), Lambda(lambda
            x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.
            Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(
            nn.Conv2d(512, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(), nn.Conv2d(256, 256, (3, 3), (1,
            1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(256), nn.ReLU()),
            nn.Conv2d(256, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
            nn.BatchNorm2d(512)), Lambda(lambda x: x)), LambdaReduce(lambda
            x, y: x + y), nn.ReLU())), nn.Sequential(nn.Sequential(
            LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(
            512, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.
            BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (2, 2),
            (1, 1), 1, 32, bias=False), nn.BatchNorm2d(512), nn.ReLU()), nn
            .Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
            nn.BatchNorm2d(1024)), nn.Sequential(nn.Conv2d(512, 1024, (1, 1
            ), (2, 2), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024))),
            LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(
            LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(
            1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.
            BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (1, 1),
            (1, 1), 1, 32, bias=False), nn.BatchNorm2d(512), nn.ReLU()), nn
            .Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
            nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda
            x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x,
            nn.Sequential(nn.Sequential(nn.Conv2d(1024, 512, (1, 1), (1, 1),
            (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(), nn.
            Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn
            .BatchNorm2d(512), nn.ReLU()), nn.Conv2d(512, 1024, (1, 1), (1,
            1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda
            x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.
            Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(
            nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (1,
            1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(512), nn.ReLU()),
            nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
            nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda
            x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x,
            nn.Sequential(nn.Sequential(nn.Conv2d(1024, 512, (1, 1), (1, 1),
            (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(), nn.
            Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn
            .BatchNorm2d(512), nn.ReLU()), nn.Conv2d(512, 1024, (1, 1), (1,
            1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda
            x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.
            Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(
            nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (1,
            1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(512), nn.ReLU()),
            nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
            nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda
            x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x,
            nn.Sequential(nn.Sequential(nn.Conv2d(1024, 512, (1, 1), (1, 1),
            (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(), nn.
            Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn
            .BatchNorm2d(512), nn.ReLU()), nn.Conv2d(512, 1024, (1, 1), (1,
            1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda
            x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.
            Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(
            nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (1,
            1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(512), nn.ReLU()),
            nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
            nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda
            x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x,
            nn.Sequential(nn.Sequential(nn.Conv2d(1024, 512, (1, 1), (1, 1),
            (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(), nn.
            Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn
            .BatchNorm2d(512), nn.ReLU()), nn.Conv2d(512, 1024, (1, 1), (1,
            1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda
            x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.
            Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(
            nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (1,
            1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(512), nn.ReLU()),
            nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
            nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda
            x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x,
            nn.Sequential(nn.Sequential(nn.Conv2d(1024, 512, (1, 1), (1, 1),
            (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(), nn.
            Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn
            .BatchNorm2d(512), nn.ReLU()), nn.Conv2d(512, 1024, (1, 1), (1,
            1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda
            x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.
            Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(
            nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (1,
            1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(512), nn.ReLU()),
            nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
            nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda
            x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x,
            nn.Sequential(nn.Sequential(nn.Conv2d(1024, 512, (1, 1), (1, 1),
            (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(), nn.
            Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn
            .BatchNorm2d(512), nn.ReLU()), nn.Conv2d(512, 1024, (1, 1), (1,
            1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda
            x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.
            Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(
            nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (1,
            1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(512), nn.ReLU()),
            nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
            nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda
            x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x,
            nn.Sequential(nn.Sequential(nn.Conv2d(1024, 512, (1, 1), (1, 1),
            (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(), nn.
            Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn
            .BatchNorm2d(512), nn.ReLU()), nn.Conv2d(512, 1024, (1, 1), (1,
            1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda
            x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.
            Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(
            nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (1,
            1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(512), nn.ReLU()),
            nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
            nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda
            x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x,
            nn.Sequential(nn.Sequential(nn.Conv2d(1024, 512, (1, 1), (1, 1),
            (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(), nn.
            Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn
            .BatchNorm2d(512), nn.ReLU()), nn.Conv2d(512, 1024, (1, 1), (1,
            1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda
            x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.
            Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(
            nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (1,
            1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(512), nn.ReLU()),
            nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
            nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda
            x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x,
            nn.Sequential(nn.Sequential(nn.Conv2d(1024, 512, (1, 1), (1, 1),
            (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(), nn.
            Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn
            .BatchNorm2d(512), nn.ReLU()), nn.Conv2d(512, 1024, (1, 1), (1,
            1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda
            x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.
            Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(
            nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (1,
            1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(512), nn.ReLU()),
            nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
            nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda
            x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x,
            nn.Sequential(nn.Sequential(nn.Conv2d(1024, 512, (1, 1), (1, 1),
            (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(), nn.
            Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn
            .BatchNorm2d(512), nn.ReLU()), nn.Conv2d(512, 1024, (1, 1), (1,
            1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda
            x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.
            Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(
            nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (1,
            1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(512), nn.ReLU()),
            nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
            nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda
            x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x,
            nn.Sequential(nn.Sequential(nn.Conv2d(1024, 512, (1, 1), (1, 1),
            (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(), nn.
            Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn
            .BatchNorm2d(512), nn.ReLU()), nn.Conv2d(512, 1024, (1, 1), (1,
            1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda
            x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU())), nn.
            Sequential(nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(
            nn.Sequential(nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 
            1, bias=False), nn.BatchNorm2d(1024), nn.ReLU(), nn.Conv2d(1024,
            1024, (3, 3), (2, 2), (1, 1), 1, 32, bias=False), nn.
            BatchNorm2d(1024), nn.ReLU()), nn.Conv2d(1024, 2048, (1, 1), (1,
            1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(2048)), nn.
            Sequential(nn.Conv2d(1024, 2048, (1, 1), (2, 2), (0, 0), 1, 1,
            bias=False), nn.BatchNorm2d(2048))), LambdaReduce(lambda x, y: 
            x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.
            Sequential(nn.Sequential(nn.Conv2d(2048, 1024, (1, 1), (1, 1),
            (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024), nn.ReLU(), nn.
            Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
            nn.BatchNorm2d(1024), nn.ReLU()), nn.Conv2d(1024, 2048, (1, 1),
            (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(2048)),
            Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU
            ()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.
            Sequential(nn.Conv2d(2048, 1024, (1, 1), (1, 1), (0, 0), 1, 1,
            bias=False), nn.BatchNorm2d(1024), nn.ReLU(), nn.Conv2d(1024, 
            1024, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn.
            BatchNorm2d(1024), nn.ReLU()), nn.Conv2d(1024, 2048, (1, 1), (1,
            1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(2048)), Lambda(lambda
            x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU())))

    def forward(self, input):
        x = self.resnext101_32x4d_features(input)
        return x


class resnext101_32x4d_features(nn.Module):

    def __init__(self):
        super(resnext101_32x4d_features, self).__init__()
        self.resnext101_32x4d_stem = nn.Sequential(nn.Conv2d(3, 64, (7, 7),
            (2, 2), (3, 3), 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(
            ), nn.MaxPool2d((3, 3), (2, 2), (1, 1)))
        self.resnext101_32x4d_layer1 = nn.Sequential(nn.Sequential(nn.
            Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(
            nn.Conv2d(64, 128, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(), nn.Conv2d(128, 128, (3, 3), (1,
            1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(128), nn.ReLU()),
            nn.Conv2d(128, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
            nn.BatchNorm2d(256)), nn.Sequential(nn.Conv2d(64, 256, (1, 1),
            (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(256))),
            LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(
            LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(
            256, 128, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.
            BatchNorm2d(128), nn.ReLU(), nn.Conv2d(128, 128, (3, 3), (1, 1),
            (1, 1), 1, 32, bias=False), nn.BatchNorm2d(128), nn.ReLU()), nn
            .Conv2d(128, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn
            .BatchNorm2d(256)), Lambda(lambda x: x)), LambdaReduce(lambda x,
            y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.
            Sequential(nn.Sequential(nn.Conv2d(256, 128, (1, 1), (1, 1), (0,
            0), 1, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(), nn.
            Conv2d(128, 128, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn
            .BatchNorm2d(128), nn.ReLU()), nn.Conv2d(128, 256, (1, 1), (1, 
            1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(256)), Lambda(lambda
            x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU())))
        self.resnext101_32x4d_layer2 = nn.Sequential(nn.Sequential(nn.
            Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(
            nn.Conv2d(256, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(), nn.Conv2d(256, 256, (3, 3), (2,
            2), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(256), nn.ReLU()),
            nn.Conv2d(256, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
            nn.BatchNorm2d(512)), nn.Sequential(nn.Conv2d(256, 512, (1, 1),
            (2, 2), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512))),
            LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(
            LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(
            512, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.
            BatchNorm2d(256), nn.ReLU(), nn.Conv2d(256, 256, (3, 3), (1, 1),
            (1, 1), 1, 32, bias=False), nn.BatchNorm2d(256), nn.ReLU()), nn
            .Conv2d(256, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn
            .BatchNorm2d(512)), Lambda(lambda x: x)), LambdaReduce(lambda x,
            y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.
            Sequential(nn.Sequential(nn.Conv2d(512, 256, (1, 1), (1, 1), (0,
            0), 1, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(), nn.
            Conv2d(256, 256, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn
            .BatchNorm2d(256), nn.ReLU()), nn.Conv2d(256, 512, (1, 1), (1, 
            1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512)), Lambda(lambda
            x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.
            Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(
            nn.Conv2d(512, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(), nn.Conv2d(256, 256, (3, 3), (1,
            1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(256), nn.ReLU()),
            nn.Conv2d(256, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
            nn.BatchNorm2d(512)), Lambda(lambda x: x)), LambdaReduce(lambda
            x, y: x + y), nn.ReLU())))
        self.resnext101_32x4d_layer3 = nn.Sequential(nn.Sequential(nn.
            Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(
            nn.Conv2d(512, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (2,
            2), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(512), nn.ReLU()),
            nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
            nn.BatchNorm2d(1024)), nn.Sequential(nn.Conv2d(512, 1024, (1, 1
            ), (2, 2), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024))),
            LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(
            LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(
            1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.
            BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (1, 1),
            (1, 1), 1, 32, bias=False), nn.BatchNorm2d(512), nn.ReLU()), nn
            .Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
            nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda
            x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x,
            nn.Sequential(nn.Sequential(nn.Conv2d(1024, 512, (1, 1), (1, 1),
            (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(), nn.
            Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn
            .BatchNorm2d(512), nn.ReLU()), nn.Conv2d(512, 1024, (1, 1), (1,
            1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda
            x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.
            Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(
            nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (1,
            1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(512), nn.ReLU()),
            nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
            nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda
            x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x,
            nn.Sequential(nn.Sequential(nn.Conv2d(1024, 512, (1, 1), (1, 1),
            (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(), nn.
            Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn
            .BatchNorm2d(512), nn.ReLU()), nn.Conv2d(512, 1024, (1, 1), (1,
            1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda
            x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.
            Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(
            nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (1,
            1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(512), nn.ReLU()),
            nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
            nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda
            x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x,
            nn.Sequential(nn.Sequential(nn.Conv2d(1024, 512, (1, 1), (1, 1),
            (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(), nn.
            Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn
            .BatchNorm2d(512), nn.ReLU()), nn.Conv2d(512, 1024, (1, 1), (1,
            1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda
            x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.
            Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(
            nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (1,
            1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(512), nn.ReLU()),
            nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
            nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda
            x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x,
            nn.Sequential(nn.Sequential(nn.Conv2d(1024, 512, (1, 1), (1, 1),
            (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(), nn.
            Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn
            .BatchNorm2d(512), nn.ReLU()), nn.Conv2d(512, 1024, (1, 1), (1,
            1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda
            x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.
            Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(
            nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (1,
            1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(512), nn.ReLU()),
            nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
            nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda
            x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x,
            nn.Sequential(nn.Sequential(nn.Conv2d(1024, 512, (1, 1), (1, 1),
            (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(), nn.
            Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn
            .BatchNorm2d(512), nn.ReLU()), nn.Conv2d(512, 1024, (1, 1), (1,
            1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda
            x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.
            Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(
            nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (1,
            1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(512), nn.ReLU()),
            nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
            nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda
            x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x,
            nn.Sequential(nn.Sequential(nn.Conv2d(1024, 512, (1, 1), (1, 1),
            (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(), nn.
            Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn
            .BatchNorm2d(512), nn.ReLU()), nn.Conv2d(512, 1024, (1, 1), (1,
            1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda
            x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.
            Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(
            nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (1,
            1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(512), nn.ReLU()),
            nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
            nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda
            x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x,
            nn.Sequential(nn.Sequential(nn.Conv2d(1024, 512, (1, 1), (1, 1),
            (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(), nn.
            Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn
            .BatchNorm2d(512), nn.ReLU()), nn.Conv2d(512, 1024, (1, 1), (1,
            1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda
            x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.
            Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(
            nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (1,
            1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(512), nn.ReLU()),
            nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
            nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda
            x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x,
            nn.Sequential(nn.Sequential(nn.Conv2d(1024, 512, (1, 1), (1, 1),
            (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(), nn.
            Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn
            .BatchNorm2d(512), nn.ReLU()), nn.Conv2d(512, 1024, (1, 1), (1,
            1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda
            x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.
            Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(
            nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (1,
            1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(512), nn.ReLU()),
            nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
            nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda
            x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x,
            nn.Sequential(nn.Sequential(nn.Conv2d(1024, 512, (1, 1), (1, 1),
            (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(), nn.
            Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn
            .BatchNorm2d(512), nn.ReLU()), nn.Conv2d(512, 1024, (1, 1), (1,
            1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda
            x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.
            Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(
            nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (1,
            1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(512), nn.ReLU()),
            nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
            nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda
            x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x,
            nn.Sequential(nn.Sequential(nn.Conv2d(1024, 512, (1, 1), (1, 1),
            (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(), nn.
            Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn
            .BatchNorm2d(512), nn.ReLU()), nn.Conv2d(512, 1024, (1, 1), (1,
            1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda
            x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.
            Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(
            nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (1,
            1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(512), nn.ReLU()),
            nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
            nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda
            x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x,
            nn.Sequential(nn.Sequential(nn.Conv2d(1024, 512, (1, 1), (1, 1),
            (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(), nn.
            Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn
            .BatchNorm2d(512), nn.ReLU()), nn.Conv2d(512, 1024, (1, 1), (1,
            1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda
            x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU())))
        self.resnext101_32x4d_layer4 = nn.Sequential(nn.Sequential(nn.
            Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(
            nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
            nn.BatchNorm2d(1024), nn.ReLU(), nn.Conv2d(1024, 1024, (3, 3),
            (2, 2), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(1024), nn.
            ReLU()), nn.Conv2d(1024, 2048, (1, 1), (1, 1), (0, 0), 1, 1,
            bias=False), nn.BatchNorm2d(2048)), nn.Sequential(nn.Conv2d(
            1024, 2048, (1, 1), (2, 2), (0, 0), 1, 1, bias=False), nn.
            BatchNorm2d(2048))), LambdaReduce(lambda x, y: x + y), nn.ReLU(
            )), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.
            Sequential(nn.Conv2d(2048, 1024, (1, 1), (1, 1), (0, 0), 1, 1,
            bias=False), nn.BatchNorm2d(1024), nn.ReLU(), nn.Conv2d(1024, 
            1024, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn.
            BatchNorm2d(1024), nn.ReLU()), nn.Conv2d(1024, 2048, (1, 1), (1,
            1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(2048)), Lambda(lambda
            x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.
            Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(
            nn.Conv2d(2048, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
            nn.BatchNorm2d(1024), nn.ReLU(), nn.Conv2d(1024, 1024, (3, 3),
            (1, 1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(1024), nn.
            ReLU()), nn.Conv2d(1024, 2048, (1, 1), (1, 1), (0, 0), 1, 1,
            bias=False), nn.BatchNorm2d(2048)), Lambda(lambda x: x)),
            LambdaReduce(lambda x, y: x + y), nn.ReLU())))

    def forward(self, input):
        x = self.resnext101_32x4d_stem(input)
        x = self.resnext101_32x4d_layer1(x)
        x = self.resnext101_32x4d_layer2(x)
        x = self.resnext101_32x4d_layer3(x)
        x = self.resnext101_32x4d_layer4(x)
        return x


class _OCHead(nn.Module):

    def __init__(self, nclass, oc_arch, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_OCHead, self).__init__()
        if oc_arch == 'base':
            self.context = nn.Sequential(nn.Conv2d(2048, 512, 3, 1, padding
                =1, bias=False), norm_layer(512), nn.ReLU(True),
                BaseOCModule(512, 512, 256, 256, scales=[1], norm_layer=
                norm_layer, **kwargs))
        elif oc_arch == 'pyramid':
            self.context = nn.Sequential(nn.Conv2d(2048, 512, 3, 1, padding
                =1, bias=False), norm_layer(512), nn.ReLU(True),
                PyramidOCModule(512, 512, 256, 512, scales=[1, 2, 3, 6],
                norm_layer=norm_layer, **kwargs))
        elif oc_arch == 'asp':
            self.context = ASPOCModule(2048, 512, 256, 512, norm_layer=
                norm_layer, **kwargs)
        else:
            raise ValueError('Unknown OC architecture!')
        self.out = nn.Conv2d(512, nclass, 1)

    def forward(self, x):
        x = self.context(x)
        return self.out(x)


class BaseAttentionBlock(nn.Module):
    """The basic implementation for self-attention block/non-local block."""

    def __init__(self, in_channels, out_channels, key_channels,
        value_channels, scale=1, norm_layer=nn.BatchNorm2d, **kwargs):
        super(BaseAttentionBlock, self).__init__()
        self.scale = scale
        self.key_channels = key_channels
        self.value_channels = value_channels
        if scale > 1:
            self.pool = nn.MaxPool2d(scale)
        self.f_value = nn.Conv2d(in_channels, value_channels, 1)
        self.f_key = nn.Sequential(nn.Conv2d(in_channels, key_channels, 1),
            norm_layer(key_channels), nn.ReLU(True))
        self.f_query = self.f_key
        self.W = nn.Conv2d(value_channels, out_channels, 1)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, x):
        batch_size, c, w, h = x.size()
        if self.scale > 1:
            x = self.pool(x)
        value = self.f_value(x).view(batch_size, self.value_channels, -1
            ).permute(0, 2, 1)
        query = self.f_query(x).view(batch_size, self.key_channels, -1
            ).permute(0, 2, 1)
        key = self.f_key(x).view(batch_size, self.key_channels, -1)
        sim_map = torch.bmm(query, key) * self.key_channels ** -0.5
        sim_map = F.softmax(sim_map, dim=-1)
        context = torch.bmm(sim_map, value).permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.value_channels, *x.size()[2:])
        context = self.W(context)
        if self.scale > 1:
            context = F.interpolate(context, size=(w, h), mode='bilinear',
                align_corners=True)
        return context


class BaseOCModule(nn.Module):
    """Base-OC"""

    def __init__(self, in_channels, out_channels, key_channels,
        value_channels, scales=[1], norm_layer=nn.BatchNorm2d, concat=True,
        **kwargs):
        super(BaseOCModule, self).__init__()
        self.stages = nn.ModuleList([BaseAttentionBlock(in_channels,
            out_channels, key_channels, value_channels, scale, norm_layer,
            **kwargs) for scale in scales])
        in_channels = in_channels * 2 if concat else in_channels
        self.project = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1
            ), norm_layer(out_channels), nn.ReLU(True), nn.Dropout2d(0.05))
        self.concat = concat

    def forward(self, x):
        priors = [stage(x) for stage in self.stages]
        context = priors[0]
        for i in range(1, len(priors)):
            context += priors[i]
        if self.concat:
            context = torch.cat([context, x], 1)
        out = self.project(context)
        return out


class PyramidAttentionBlock(nn.Module):
    """The basic implementation for pyramid self-attention block/non-local block"""

    def __init__(self, in_channels, out_channels, key_channels,
        value_channels, scale=1, norm_layer=nn.BatchNorm2d, **kwargs):
        super(PyramidAttentionBlock, self).__init__()
        self.scale = scale
        self.value_channels = value_channels
        self.key_channels = key_channels
        self.f_value = nn.Conv2d(in_channels, value_channels, 1)
        self.f_key = nn.Sequential(nn.Conv2d(in_channels, key_channels, 1),
            norm_layer(key_channels), nn.ReLU(True))
        self.f_query = self.f_key
        self.W = nn.Conv2d(value_channels, out_channels, 1)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, x):
        batch_size, c, w, h = x.size()
        local_x = list()
        local_y = list()
        step_w, step_h = w // self.scale, h // self.scale
        for i in range(self.scale):
            for j in range(self.scale):
                start_x, start_y = step_w * i, step_h * j
                end_x, end_y = min(start_x + step_w, w), min(start_y +
                    step_h, h)
                if i == self.scale - 1:
                    end_x = w
                if j == self.scale - 1:
                    end_y = h
                local_x += [start_x, end_x]
                local_y += [start_y, end_y]
        value = self.f_value(x)
        query = self.f_query(x)
        key = self.f_key(x)
        local_list = list()
        local_block_cnt = self.scale ** 2 * 2
        for i in range(0, local_block_cnt, 2):
            value_local = value[:, :, local_x[i]:local_x[i + 1], local_y[i]
                :local_y[i + 1]]
            query_local = query[:, :, local_x[i]:local_x[i + 1], local_y[i]
                :local_y[i + 1]]
            key_local = key[:, :, local_x[i]:local_x[i + 1], local_y[i]:
                local_y[i + 1]]
            w_local, h_local = value_local.size(2), value_local.size(3)
            value_local = value_local.contiguous().view(batch_size, self.
                value_channels, -1).permute(0, 2, 1)
            query_local = query_local.contiguous().view(batch_size, self.
                key_channels, -1).permute(0, 2, 1)
            key_local = key_local.contiguous().view(batch_size, self.
                key_channels, -1)
            sim_map = torch.bmm(query_local, key_local
                ) * self.key_channels ** -0.5
            sim_map = F.softmax(sim_map, dim=-1)
            context_local = torch.bmm(sim_map, value_local).permute(0, 2, 1
                ).contiguous()
            context_local = context_local.view(batch_size, self.
                value_channels, w_local, h_local)
            local_list.append(context_local)
        context_list = list()
        for i in range(0, self.scale):
            row_tmp = list()
            for j in range(self.scale):
                row_tmp.append(local_list[j + i * self.scale])
            context_list.append(torch.cat(row_tmp, 3))
        context = torch.cat(context_list, 2)
        context = self.W(context)
        return context


class PyramidOCModule(nn.Module):
    """Pyramid-OC"""

    def __init__(self, in_channels, out_channels, key_channels,
        value_channels, scales=[1], norm_layer=nn.BatchNorm2d, **kwargs):
        super(PyramidOCModule, self).__init__()
        self.stages = nn.ModuleList([PyramidAttentionBlock(in_channels,
            out_channels, key_channels, value_channels, scale, norm_layer,
            **kwargs) for scale in scales])
        self.up_dr = nn.Sequential(nn.Conv2d(in_channels, in_channels * len
            (scales), 1), norm_layer(in_channels * len(scales)), nn.ReLU(True))
        self.project = nn.Sequential(nn.Conv2d(in_channels * len(scales) * 
            2, out_channels, 1), norm_layer(out_channels), nn.ReLU(True),
            nn.Dropout2d(0.05))

    def forward(self, x):
        priors = [stage(x) for stage in self.stages]
        context = [self.up_dr(x)]
        for i in range(len(priors)):
            context += [priors[i]]
        context = torch.cat(context, 1)
        out = self.project(context)
        return out


class ASPOCModule(nn.Module):
    """ASP-OC"""

    def __init__(self, in_channels, out_channels, key_channels,
        value_channels, atrous_rates=(12, 24, 36), norm_layer=nn.
        BatchNorm2d, **kwargs):
        super(ASPOCModule, self).__init__()
        self.context = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3,
            padding=1), norm_layer(out_channels), nn.ReLU(True),
            BaseOCModule(out_channels, out_channels, key_channels,
            value_channels, [2], norm_layer, False, **kwargs))
        rate1, rate2, rate3 = tuple(atrous_rates)
        self.b1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3,
            padding=rate1, dilation=rate1, bias=False), norm_layer(
            out_channels), nn.ReLU(True))
        self.b2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3,
            padding=rate2, dilation=rate2, bias=False), norm_layer(
            out_channels), nn.ReLU(True))
        self.b3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3,
            padding=rate3, dilation=rate3, bias=False), norm_layer(
            out_channels), nn.ReLU(True))
        self.b4 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1,
            bias=False), norm_layer(out_channels), nn.ReLU(True))
        self.project = nn.Sequential(nn.Conv2d(out_channels * 5,
            out_channels, 1, bias=False), norm_layer(out_channels), nn.ReLU
            (True), nn.Dropout2d(0.1))

    def forward(self, x):
        feat1 = self.context(x)
        feat2 = self.b1(x)
        feat3 = self.b2(x)
        feat4 = self.b3(x)
        feat5 = self.b4(x)
        out = torch.cat((feat1, feat2, feat3, feat4, feat5), dim=1)
        out = self.project(out)
        return out


class ResidualConvUnit(nn.Module):

    def __init__(self, features):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1,
            padding=1, bias=True)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1,
            padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out + x


class MultiResolutionFusion(nn.Module):

    def __init__(self, out_feats, *shapes):
        super().__init__()
        _, max_size = max(shapes, key=lambda x: x[1])
        self.scale_factors = []
        for i, shape in enumerate(shapes):
            feat, size = shape
            if max_size % size != 0:
                raise ValueError('max_size not divisble by shape {}'.format(i))
            self.scale_factors.append(max_size // size)
            self.add_module('resolve{}'.format(i), nn.Conv2d(feat,
                out_feats, kernel_size=3, stride=1, padding=1, bias=False))

    def forward(self, *xs):
        output = self.resolve0(xs[0])
        if self.scale_factors[0] != 1:
            output = nn.functional.interpolate(output, scale_factor=self.
                scale_factors[0], mode='bilinear', align_corners=True)
        for i, x in enumerate(xs[1:], 1):
            output += self.__getattr__('resolve{}'.format(i))(x)
            if self.scale_factors[i] != 1:
                output = nn.functional.interpolate(output, scale_factor=
                    self.scale_factors[i], mode='bilinear', align_corners=True)
        return output


class ChainedResidualPool(nn.Module):

    def __init__(self, feats):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        for i in range(1, 4):
            self.add_module('block{}'.format(i), nn.Sequential(nn.MaxPool2d
                (kernel_size=5, stride=1, padding=2), nn.Conv2d(feats,
                feats, kernel_size=3, stride=1, padding=1, bias=False)))

    def forward(self, x):
        x = self.relu(x)
        path = x
        for i in range(1, 4):
            path = self.__getattr__('block{}'.format(i))(path)
            x = x + path
        return x


class ChainedResidualPoolImproved(nn.Module):

    def __init__(self, feats):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        for i in range(1, 5):
            self.add_module('block{}'.format(i), nn.Sequential(nn.Conv2d(
                feats, feats, kernel_size=3, stride=1, padding=1, bias=
                False), nn.MaxPool2d(kernel_size=5, stride=1, padding=2)))

    def forward(self, x):
        x = self.relu(x)
        path = x
        for i in range(1, 5):
            path = self.__getattr__('block{}'.format(i))(path)
            x += path
        return x


class BaseRefineNetBlock(nn.Module):

    def __init__(self, features, residual_conv_unit,
        multi_resolution_fusion, chained_residual_pool, *shapes):
        super().__init__()
        for i, shape in enumerate(shapes):
            feats = shape[0]
            self.add_module('rcu{}'.format(i), nn.Sequential(
                residual_conv_unit(feats), residual_conv_unit(feats)))
        if len(shapes) != 1:
            self.mrf = multi_resolution_fusion(features, *shapes)
        else:
            self.mrf = None
        self.crp = chained_residual_pool(features)
        self.output_conv = residual_conv_unit(features)

    def forward(self, *xs):
        rcu_xs = []
        for i, x in enumerate(xs):
            rcu_xs.append(self.__getattr__('rcu{}'.format(i))(x))
        if self.mrf is not None:
            out = self.mrf(*rcu_xs)
        else:
            out = rcu_xs[0]
        out = self.crp(out)
        return self.output_conv(out)


class RefineNetBlockImprovedPooling(nn.Module):

    def __init__(self, features, *shapes):
        super().__init__(features, ResidualConvUnit, MultiResolutionFusion,
            ChainedResidualPoolImproved, *shapes)


class RefineNetBlock(BaseRefineNetBlock):

    def __init__(self, features, *shapes):
        super().__init__(features, ResidualConvUnit, MultiResolutionFusion,
            ChainedResidualPool, *shapes)


class BaseRefineNet4Cascade(nn.Module):

    def __init__(self, input_shape, refinenet_block, num_classes=1,
        features=256, resnet_factory=models.resnet101, pretrained=True,
        freeze_resnet=False, **kwargs):
        """Multi-path 4-Cascaded RefineNet for image segmentation

        Args:
            input_shape ((int, int)): (channel, size) assumes input has
                equal height and width
            refinenet_block (block): RefineNet Block
            num_classes (int, optional): number of classes
            features (int, optional): number of features in refinenet
            resnet_factory (func, optional): A Resnet model from torchvision.
                Default: models.resnet101
            pretrained (bool, optional): Use pretrained version of resnet
                Default: True
            freeze_resnet (bool, optional): Freeze resnet model
                Default: True

        Raises:
            ValueError: size of input_shape not divisible by 32
        """
        super().__init__()
        input_channel, input_size = input_shape
        if input_size % 32 != 0:
            raise ValueError('{} not divisble by 32'.format(input_shape))
        resnet = resnet_factory(pretrained=pretrained)
        self.layer1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu,
            resnet.maxpool, resnet.layer1)
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        if freeze_resnet:
            layers = [self.layer1, self.layer2, self.layer3, self.layer4]
            for layer in layers:
                for param in layer.parameters():
                    param.requires_grad = False
        self.layer1_rn = nn.Conv2d(256, features, kernel_size=3, stride=1,
            padding=1, bias=False)
        self.layer2_rn = nn.Conv2d(512, features, kernel_size=3, stride=1,
            padding=1, bias=False)
        self.layer3_rn = nn.Conv2d(1024, features, kernel_size=3, stride=1,
            padding=1, bias=False)
        self.layer4_rn = nn.Conv2d(2048, 2 * features, kernel_size=3,
            stride=1, padding=1, bias=False)
        self.refinenet4 = RefineNetBlock(2 * features, (2 * features, 
            input_size // 32))
        self.refinenet3 = RefineNetBlock(features, (2 * features, 
            input_size // 32), (features, input_size // 16))
        self.refinenet2 = RefineNetBlock(features, (features, input_size //
            16), (features, input_size // 8))
        self.refinenet1 = RefineNetBlock(features, (features, input_size //
            8), (features, input_size // 4))
        self.output_conv = nn.Sequential(ResidualConvUnit(features),
            ResidualConvUnit(features), nn.Conv2d(features, num_classes,
            kernel_size=1, stride=1, padding=0, bias=True))

    def forward(self, x):
        size = x.size()[2:]
        layer_1 = self.layer1(x)
        layer_2 = self.layer2(layer_1)
        layer_3 = self.layer3(layer_2)
        layer_4 = self.layer4(layer_3)
        layer_1_rn = self.layer1_rn(layer_1)
        layer_2_rn = self.layer2_rn(layer_2)
        layer_3_rn = self.layer3_rn(layer_3)
        layer_4_rn = self.layer4_rn(layer_4)
        path_4 = self.refinenet4(layer_4_rn)
        path_3 = self.refinenet3(path_4, layer_3_rn)
        path_2 = self.refinenet2(path_3, layer_2_rn)
        path_1 = self.refinenet1(path_2, layer_1_rn)
        out_conv = self.output_conv(path_1)
        out = F.interpolate(out_conv, size, mode='bilinear', align_corners=True
            )
        return out


class GlobalConvolutionBlock(nn.Module):

    def __init__(self, in_channels, out_channels, k):
        super(GlobalConvolutionBlock, self).__init__()
        self.left = nn.Sequential(nn.Conv2d(in_channels, out_channels,
            kernel_size=(k[0], 1), padding=(k[0] // 2, 0)), nn.Conv2d(
            out_channels, out_channels, kernel_size=(1, k[1]), padding=(0, 
            k[1] // 2)))
        self.right = nn.Sequential(nn.Conv2d(in_channels, out_channels,
            kernel_size=(1, k[1]), padding=(0, k[1] // 2)), nn.Conv2d(
            out_channels, out_channels, kernel_size=(k[0], 1), padding=(k[0
            ] // 2, 0)))

    def forward(self, x):
        left = self.left(x)
        right = self.right(x)
        return left + right


class BoundaryRefine(nn.Module):

    def __init__(self, in_channels):
        super(BoundaryRefine, self).__init__()
        self.layer = nn.Sequential(nn.Conv2d(in_channels, in_channels,
            kernel_size=3, padding=1), nn.BatchNorm2d(in_channels), nn.ReLU
            (inplace=True), nn.Conv2d(in_channels, in_channels, kernel_size
            =3, padding=1), nn.BatchNorm2d(in_channels))

    def forward(self, x):
        convs = self.layer(x)
        return x.expand_as(convs) + convs


class ResnetGCN(nn.Module):

    def __init__(self, num_classes, pretrained=True, **kwargs):
        super(ResnetGCN, self).__init__()
        resent = models.resnet101(pretrained=pretrained)
        self.layer0 = nn.Sequential(resent.conv1, resent.bn1, resent.relu,
            resent.maxpool)
        self.layer1 = resent.layer1
        self.layer2 = resent.layer2
        self.layer3 = resent.layer3
        self.layer4 = resent.layer4
        ks = 7
        self.gcn256 = GlobalConvolutionBlock(256, num_classes, (59, 79))
        self.br256 = BoundaryRefine(num_classes)
        self.gcn512 = GlobalConvolutionBlock(512, num_classes, (29, 39))
        self.br512 = BoundaryRefine(num_classes)
        self.gcn1024 = GlobalConvolutionBlock(1024, num_classes, (13, 19))
        self.br1024 = BoundaryRefine(num_classes)
        self.gcn2048 = GlobalConvolutionBlock(2048, num_classes, (7, 9))
        self.br2048 = BoundaryRefine(num_classes)
        self.br1 = BoundaryRefine(num_classes)
        self.br2 = BoundaryRefine(num_classes)
        self.br3 = BoundaryRefine(num_classes)
        self.br4 = BoundaryRefine(num_classes)
        self.br5 = BoundaryRefine(num_classes)
        self.activation = nn.Sigmoid()
        self.deconv1 = nn.ConvTranspose2d(1, 1, 2, stride=2)
        self.deconv2 = nn.ConvTranspose2d(1, 1, 2, stride=2)
        initialize_weights(self.gcn256, self.gcn512, self.gcn1024, self.
            gcn2048, self.br5, self.br4, self.br3, self.br2, self.br1, self
            .br256, self.br512, self.br1024, self.br2048, self.deconv1,
            self.deconv2)

    def forward(self, x):
        x = self.layer0(x)
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        enc1 = self.br256(self.gcn256(layer1))
        enc2 = self.br512(self.gcn512(layer2))
        enc3 = self.br1024(self.gcn1024(layer3))
        enc4 = self.br2048(self.gcn2048(layer4))
        dec1 = self.br1(F.interpolate(enc4, size=enc3.size()[2:], mode=
            'bilinear') + enc3)
        dec2 = self.br2(F.interpolate(dec1, enc2.size()[2:], mode=
            'bilinear') + enc2)
        dec3 = self.br3(F.interpolate(dec2, enc1.size()[2:], mode=
            'bilinear') + enc1)
        dec4 = self.br4(self.deconv1(dec3))
        score_map = self.br5(self.deconv2(dec4))
        return self.activation(score_map)

    def _do_upsample(self, num_classes=1, kernel_size=2, stride=2):
        return nn.ConvTranspose2d(num_classes, num_classes, kernel_size=
            kernel_size, stride=stride)


class _DecoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels, num_conv_layers):
        super(_DecoderBlock, self).__init__()
        middle_channels = in_channels / 2
        layers = [nn.ConvTranspose2d(in_channels, in_channels, kernel_size=
            2, stride=2), nn.Conv2d(in_channels, middle_channels,
            kernel_size=3, padding=1), nn.BatchNorm2d(middle_channels), nn.
            ReLU(inplace=True)]
        layers += [nn.Conv2d(middle_channels, middle_channels, kernel_size=
            3, padding=1), nn.BatchNorm2d(middle_channels), nn.ReLU(inplace
            =True)] * (num_conv_layers - 2)
        layers += [nn.Conv2d(middle_channels, out_channels, kernel_size=3,
            padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
        self.decode = nn.Sequential(*layers)

    def forward(self, x):
        return self.decode(x)


def init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.weight, 1)


class UNet(nn.Module):

    def __init__(self, in_channels=3, num_classes=4, feature_scale=4,
        is_deconv=True, is_batchnorm=True):
        super(UNet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)
        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)
        self.cls = nn.Sequential(nn.Dropout(p=0.5), nn.Conv2d(256, 3, 1),
            nn.AdaptiveMaxPool2d(1), nn.Sigmoid())
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)
        self.final_1 = nn.Conv2d(filters[0], num_classes, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)
        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)
        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)
        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)
        center = self.center(maxpool4)
        cls_branch = self.cls(center).squeeze()
        up4 = self.up_concat4(center, conv4)
        up3 = self.up_concat3(up4, conv3)
        up2 = self.up_concat2(up3, conv2)
        up1 = self.up_concat1(up2, conv1)
        final_1 = self.final_1(up1)
        return final_1


class UNet_Nested(nn.Module):

    def __init__(self, in_channels=3, n_classes=4, feature_scale=4,
        is_deconv=True, is_batchnorm=True, is_ds=True):
        super(UNet_Nested, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.is_ds = is_ds
        self.feature_scale = feature_scale
        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]
        self.conv00 = unetConv2(self.in_channels, filters[0], self.is_batchnorm
            )
        self.maxpool0 = nn.MaxPool2d(kernel_size=2)
        self.conv10 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv20 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv30 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.conv40 = unetConv2(filters[3], filters[4], self.is_batchnorm)
        self.cls = nn.Sequential(nn.Dropout(p=0.5), nn.Conv2d(256, 3, 1),
            nn.AdaptiveMaxPool2d(1), nn.Sigmoid())
        self.up_concat01 = unetUp(filters[1], filters[0], self.is_deconv)
        self.up_concat11 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat21 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat31 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat02 = unetUp(filters[1], filters[0], self.is_deconv, 3)
        self.up_concat12 = unetUp(filters[2], filters[1], self.is_deconv, 3)
        self.up_concat22 = unetUp(filters[3], filters[2], self.is_deconv, 3)
        self.up_concat03 = unetUp(filters[1], filters[0], self.is_deconv, 4)
        self.up_concat13 = unetUp(filters[2], filters[1], self.is_deconv, 4)
        self.up_concat04 = unetUp(filters[1], filters[0], self.is_deconv, 5)
        self.final_1 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_2 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_3 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_4 = nn.Conv2d(filters[0], n_classes, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        X_00 = self.conv00(inputs)
        maxpool0 = self.maxpool0(X_00)
        X_10 = self.conv10(maxpool0)
        maxpool1 = self.maxpool1(X_10)
        X_20 = self.conv20(maxpool1)
        maxpool2 = self.maxpool2(X_20)
        X_30 = self.conv30(maxpool2)
        maxpool3 = self.maxpool3(X_30)
        X_40 = self.conv40(maxpool3)
        cls_branch = self.cls(X_40).squeeze()
        X_01 = self.up_concat01(X_10, X_00)
        X_11 = self.up_concat11(X_20, X_10)
        X_21 = self.up_concat21(X_30, X_20)
        X_31 = self.up_concat31(X_40, X_30)
        X_02 = self.up_concat02(X_11, X_00, X_01)
        X_12 = self.up_concat12(X_21, X_10, X_11)
        X_22 = self.up_concat22(X_31, X_20, X_21)
        X_03 = self.up_concat03(X_12, X_00, X_01, X_02)
        X_13 = self.up_concat13(X_22, X_10, X_11, X_12)
        X_04 = self.up_concat04(X_13, X_00, X_01, X_02, X_03)
        final_1 = self.final_1(X_01)
        final_2 = self.final_2(X_02)
        final_3 = self.final_3(X_03)
        final_4 = self.final_4(X_04)
        final = (final_1 + final_2 + final_3 + final_4) / 4
        if self.is_ds:
            return final
        else:
            return final_4


class UNet_Nested_dilated(nn.Module):

    def __init__(self, num_classes=4, in_channels=3, feature_scale=4,
        is_deconv=True, is_batchnorm=True, is_ds=True, **kwargs):
        super(UNet_Nested_dilated, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.is_ds = is_ds
        self.feature_scale = feature_scale
        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]
        self.conv00 = unetConv2(self.in_channels, filters[0], self.is_batchnorm
            )
        self.maxpool0 = nn.MaxPool2d(kernel_size=2)
        self.conv10 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv20 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv30 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.conv40 = unetConv2(filters[3], filters[4], self.is_batchnorm)
        self.dilated = unetConv2_dilation(filters[4], filters[4], self.
            is_batchnorm)
        self.cls = nn.Sequential(nn.Dropout(p=0.5), nn.Conv2d(256, 3, 1),
            nn.AdaptiveMaxPool2d(1), nn.Sigmoid())
        self.up_concat01 = unetUp(filters[1], filters[0], self.is_deconv)
        self.up_concat11 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat21 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat31 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat02 = unetUp(filters[1], filters[0], self.is_deconv, 3)
        self.up_concat12 = unetUp(filters[2], filters[1], self.is_deconv, 3)
        self.up_concat22 = unetUp(filters[3], filters[2], self.is_deconv, 3)
        self.up_concat03 = unetUp(filters[1], filters[0], self.is_deconv, 4)
        self.up_concat13 = unetUp(filters[2], filters[1], self.is_deconv, 4)
        self.up_concat04 = unetUp(filters[1], filters[0], self.is_deconv, 5)
        self.final_1 = nn.Conv2d(filters[0], num_classes, 1)
        self.final_2 = nn.Conv2d(filters[0], num_classes, 1)
        self.final_3 = nn.Conv2d(filters[0], num_classes, 1)
        self.final_4 = nn.Conv2d(filters[0], num_classes, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        X_00 = self.conv00(inputs)
        maxpool0 = self.maxpool0(X_00)
        X_10 = self.conv10(maxpool0)
        maxpool1 = self.maxpool1(X_10)
        X_20 = self.conv20(maxpool1)
        maxpool2 = self.maxpool2(X_20)
        X_30 = self.conv30(maxpool2)
        maxpool3 = self.maxpool3(X_30)
        X_40 = self.conv40(maxpool3)
        X_40_d = self.dilated(X_40)
        cls_branch = self.cls(X_40_d).squeeze()
        X_01 = self.up_concat01(X_10, X_00)
        X_11 = self.up_concat11(X_20, X_10)
        X_21 = self.up_concat21(X_30, X_20)
        X_31 = self.up_concat31(X_40_d, X_30)
        X_02 = self.up_concat02(X_11, X_00, X_01)
        X_12 = self.up_concat12(X_21, X_10, X_11)
        X_22 = self.up_concat22(X_31, X_20, X_21)
        X_03 = self.up_concat03(X_12, X_00, X_01, X_02)
        X_13 = self.up_concat13(X_22, X_10, X_11, X_12)
        X_04 = self.up_concat04(X_13, X_00, X_01, X_02, X_03)
        final_1 = self.final_1(X_01)
        final_2 = self.final_2(X_02)
        final_3 = self.final_3(X_03)
        final_4 = self.final_4(X_04)
        final = (final_1 + final_2 + final_3 + final_4) / 4
        if self.is_ds:
            return final
        else:
            return final_4


class unetConv2(nn.Module):

    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1,
        padding=1):
        super(unetConv2, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                    nn.BatchNorm2d(out_size), nn.ReLU(inplace=True))
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size
        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                    nn.ReLU(inplace=True))
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)
        return x


class unetConv2_res(nn.Module):

    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1,
        padding=1):
        super(unetConv2_res, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        self.conv0 = nn.Conv2d(in_size, out_size, 1)
        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                    nn.BatchNorm2d(out_size), nn.ReLU(inplace=True))
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size
        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                    nn.ReLU(inplace=True))
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        inputs_ori = self.conv0(inputs)
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)
        return x + inputs_ori


class unetUp(nn.Module):

    def __init__(self, in_size, out_size, is_deconv, n_concat=2):
        super(unetUp, self).__init__()
        self.conv = unetConv2(in_size + (n_concat - 2) * out_size, out_size,
            False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4,
                stride=2, padding=1)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        for m in self.children():
            if m.__class__.__name__.find('unetConv2') != -1:
                continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs0, *input):
        outputs0 = self.up(inputs0)
        for i in range(len(input)):
            outputs0 = torch.cat([outputs0, input[i]], 1)
        return self.conv(outputs0)


class UnetConv3(nn.Module):

    def __init__(self, in_size, out_size, is_batchnorm, kernel_size=(3, 3, 
        3), padding_size=(1, 1, 1), init_stride=(1, 1, 1)):
        super(UnetConv3, self).__init__()
        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size,
                kernel_size, init_stride, padding_size), nn.BatchNorm3d(
                out_size), nn.ReLU(inplace=True))
            self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size,
                kernel_size, 1, padding_size), nn.BatchNorm3d(out_size), nn
                .ReLU(inplace=True))
        else:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size,
                kernel_size, init_stride, padding_size), nn.ReLU(inplace=True))
            self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size,
                kernel_size, 1, padding_size), nn.ReLU(inplace=True))
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class UnetUp3(nn.Module):

    def __init__(self, in_size, out_size, is_deconv, is_batchnorm=True):
        super(UnetUp3, self).__init__()
        if is_deconv:
            self.conv = UnetConv3(in_size, out_size, is_batchnorm)
            self.up = nn.ConvTranspose3d(in_size, out_size, kernel_size=(4,
                4, 4), stride=(2, 2, 2), padding=(1, 1, 1))
        else:
            self.conv = UnetConv3(in_size + out_size, out_size, is_batchnorm)
            self.up = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear')
        for m in self.children():
            if m.__class__.__name__.find('UnetConv3') != -1:
                continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2, 0]
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))


class Upold(nn.Module):

    def __init__(self, in_size, out_size, is_deconv):
        super(Upold, self).__init__()
        self.conv = unetConv2(in_size, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4,
                stride=2, padding=1)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        for m in self.children():
            if m.__class__.__name__.find('unetConv2') != -1:
                continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2]
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))


class unetConv2_SELU(nn.Module):

    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1,
        padding=1):
        super(unetConv2_SELU, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                    nn.BatchNorm2d(out_size), nn.SELU(inplace=True))
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size
        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                    nn.SELU(inplace=True))
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)
        return x


class unetUp_SELU(nn.Module):

    def __init__(self, in_size, out_size, is_deconv):
        super(unetUp_SELU, self).__init__()
        self.conv = unetConv2_SELU(in_size, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4,
                stride=2, padding=1)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        for m in self.children():
            if m.__class__.__name__.find('unetConv2') != -1:
                continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2]
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))


class unetConv2_dilation(nn.Module):

    def __init__(self, in_size, out_size, is_batchnorm=True, n=4, ks=3,
        stride=1):
        super(unetConv2_dilation, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        s = stride
        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, 2 **
                    (i - 1), 2 ** (i - 1)), nn.BatchNorm2d(out_size), nn.
                    ReLU(inplace=True))
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size
        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s),
                    nn.ReLU(inplace=True))
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        output = inputs
        x_0 = inputs
        conv = getattr(self, 'conv1')
        x_1 = conv(x_0)
        conv = getattr(self, 'conv2')
        x_2 = conv(x_1)
        conv = getattr(self, 'conv3')
        x_3 = conv(x_2)
        conv = getattr(self, 'conv4')
        x_4 = conv(x_3)
        return x_0 + x_1 + x_2 + x_3 + x_4


class unetConv2_dilation2(nn.Module):

    def __init__(self, in_size, out_size, is_batchnorm=True, n=3, ks=3,
        stride=1):
        super(unetConv2_dilation2, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        s = stride
        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, 2 **
                    (i - 1), 2 ** (i - 1)), nn.BatchNorm2d(out_size), nn.
                    ReLU(inplace=True))
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size
        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s),
                    nn.ReLU(inplace=True))
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        output = inputs
        x_0 = inputs
        conv = getattr(self, 'conv1')
        x_1 = conv(x_0)
        conv = getattr(self, 'conv2')
        x_2 = conv(x_1)
        conv = getattr(self, 'conv3')
        x_3 = conv(x_2)
        return x_0 + x_1 + x_2 + x_3


class SELayer(nn.Module):

    def __init__(self, channel, reduction=2, is_bn=True, is_cse=True,
        is_sse=True):
        super(SELayer, self).__init__()
        self.is_cse = is_cse
        self.is_sse = is_sse
        self.is_bn = is_bn
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True), nn.Linear(channel // reduction, channel),
            nn.Sigmoid())
        self.sse1 = nn.Conv2d(channel, 1, 1)
        self.sse2 = nn.Sigmoid()
        self.bn = nn.BatchNorm2d(channel)

    def forward(self, x):
        b, c, _, _ = x.size()
        y_c = self.avg_pool(x).view(b, c)
        y_c = self.fc(y_c).view(b, c, 1, 1)
        if self.is_bn:
            out_c = self.bn(x * y_c)
        else:
            out_c = x * y_c
        y_s = self.sse2(self.sse1(x))
        if self.is_bn:
            out_s = self.bn(x * y_s)
        else:
            out_s = x * y_s
        if self.is_cse and not self.is_sse:
            return out_c
        elif self.is_sse and not self.is_cse:
            return out_s
        else:
            return out_c + out_s


class Downsample(nn.Module):

    def __init__(self, in_size, out_size, is_batchnorm, ks=4, stride=2,
        padding=1):
        super(Downsample, self).__init__()
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
            self.conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p
                ), nn.BatchNorm2d(out_size), nn.ReLU(inplace=True))
        else:
            self.conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p
                ), nn.ReLU(inplace=True))
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = self.conv(inputs)
        return x


class Atrous_module(nn.Module):

    def __init__(self, inplanes, planes, rate):
        super(Atrous_module, self).__init__()
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=3,
            stride=1, padding=rate, dilation=rate)
        self.batch_norm = nn.BatchNorm2d(planes)

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.batch_norm(x)
        return x


class BasicBlock(nn.Module):

    def __init__(self, inplanes1, outplanes1, outplanes2, kernel=3,
        downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(inplanes1, outplanes1, kernel_size=kernel,
            dilation=2)
        self.bn1 = nn.BatchNorm3d(outplanes1)
        self.conv2 = nn.Conv3d(outplanes1, outplanes2, kernel_size=kernel,
            dilation=2)
        self.bn2 = nn.BatchNorm3d(outplanes2)
        self.relu = nn.ReLU(inplace=True)
        if inplanes1 == outplanes2:
            self.downsample = downsample
        else:
            self.downsample = nn.Sequential(nn.Conv3d(inplanes1, outplanes2,
                kernel_size=1), nn.BatchNorm3d(outplanes2))
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        residual = x[:, :, 4:-4, 4:-4, 4:-4]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.downsample is not None:
            residual = self.downsample(residual)
        x += residual
        x = self.relu(x)
        return x


class Autofocus_single(nn.Module):

    def __init__(self, inplanes1, outplanes1, outplanes2, padding_list,
        dilation_list, num_branches, kernel=3):
        super(Autofocus_single, self).__init__()
        self.padding_list = padding_list
        self.dilation_list = dilation_list
        self.num_branches = num_branches
        self.conv1 = nn.Conv3d(inplanes1, outplanes1, kernel_size=kernel,
            dilation=2)
        self.bn1 = nn.BatchNorm3d(outplanes1)
        self.bn_list2 = nn.ModuleList()
        for i in range(len(self.padding_list)):
            self.bn_list2.append(nn.BatchNorm3d(outplanes2))
        self.conv2 = nn.Conv3d(outplanes1, outplanes2, kernel_size=kernel,
            dilation=self.dilation_list[0])
        self.convatt1 = nn.Conv3d(outplanes1, int(outplanes1 / 2),
            kernel_size=kernel)
        self.convatt2 = nn.Conv3d(int(outplanes1 / 2), self.num_branches,
            kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        if inplanes1 == outplanes2:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(nn.Conv3d(inplanes1, outplanes2,
                kernel_size=1), nn.BatchNorm3d(outplanes2))
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        residual = x[:, :, 4:-4, 4:-4, 4:-4]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        feature = x.detach()
        att = self.relu(self.convatt1(feature))
        att = self.convatt2(att)
        att = F.softmax(att, dim=1)
        att = att[:, :, 1:-1, 1:-1, 1:-1]
        x1 = self.conv2(x)
        shape = x1.size()
        x1 = self.bn_list2[0](x1) * att[:, 0:1, :, :, :].expand(shape)
        for i in range(1, self.num_branches):
            x2 = F.conv3d(x, self.conv2.weight, padding=self.padding_list[i
                ], dilation=self.dilation_list[i])
            x2 = self.bn_list2[i](x2)
            x1 += x2 * att[:, i:i + 1, :, :, :].expand(shape)
        if self.downsample is not None:
            residual = self.downsample(residual)
        x = x1 + residual
        x = self.relu(x)
        return x


class Autofocus(nn.Module):

    def __init__(self, inplanes1, outplanes1, outplanes2, padding_list,
        dilation_list, num_branches, kernel=3):
        super(Autofocus, self).__init__()
        self.padding_list = padding_list
        self.dilation_list = dilation_list
        self.num_branches = num_branches
        self.conv1 = nn.Conv3d(inplanes1, outplanes1, kernel_size=kernel,
            dilation=self.dilation_list[0])
        self.convatt11 = nn.Conv3d(inplanes1, int(inplanes1 / 2),
            kernel_size=kernel)
        self.convatt12 = nn.Conv3d(int(inplanes1 / 2), self.num_branches,
            kernel_size=1)
        self.bn_list1 = nn.ModuleList()
        for i in range(self.num_branches):
            self.bn_list1.append(nn.BatchNorm3d(outplanes1))
        self.conv2 = nn.Conv3d(outplanes1, outplanes2, kernel_size=kernel,
            dilation=self.dilation_list[0])
        self.convatt21 = nn.Conv3d(outplanes1, int(outplanes1 / 2),
            kernel_size=kernel)
        self.convatt22 = nn.Conv3d(int(outplanes1 / 2), self.num_branches,
            kernel_size=1)
        self.bn_list2 = nn.ModuleList()
        for i in range(self.num_branches):
            self.bn_list2.append(nn.BatchNorm3d(outplanes2))
        self.relu = nn.ReLU(inplace=True)
        if inplanes1 == outplanes2:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(nn.Conv3d(inplanes1, outplanes2,
                kernel_size=1), nn.BatchNorm3d(outplanes2))
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        residual = x[:, :, 4:-4, 4:-4, 4:-4]
        feature = x.detach()
        att = self.relu(self.convatt11(feature))
        att = self.convatt12(att)
        att = F.softmax(att, dim=1)
        att = att[:, :, 1:-1, 1:-1, 1:-1]
        x1 = self.conv1(x)
        shape = x1.size()
        x1 = self.bn_list1[0](x1) * att[:, 0:1, :, :, :].expand(shape)
        for i in range(1, self.num_branches):
            x2 = F.conv3d(x, self.conv1.weight, padding=self.padding_list[i
                ], dilation=self.dilation_list[i])
            x2 = self.bn_list1[i](x2)
            x1 += x2 * att[:, i:i + 1, :, :, :].expand(shape)
        x = self.relu(x1)
        feature2 = x.detach()
        att2 = self.relu(self.convatt21(feature2))
        att2 = self.convatt22(att2)
        att2 = F.softmax(att2, dim=1)
        att2 = att2[:, :, 1:-1, 1:-1, 1:-1]
        x21 = self.conv2(x)
        shape = x21.size()
        x21 = self.bn_list2[0](x21) * att2[:, 0:1, :, :, :].expand(shape)
        for i in range(1, self.num_branches):
            x22 = F.conv3d(x, self.conv2.weight, padding=self.padding_list[
                i], dilation=self.dilation_list[i])
            x22 = self.bn_list2[i](x22)
            x21 += x22 * att2[:, i:i + 1, :, :, :].expand(shape)
        if self.downsample is not None:
            residual = self.downsample(residual)
        x = x21 + residual
        x = self.relu(x)
        return x


class Basic(nn.Module):

    def __init__(self, channels, kernel_size):
        super(Basic, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv3d(self.channels[0], self.channels[1],
            kernel_size=self.kernel_size)
        self.bn1 = nn.BatchNorm3d(self.channels[1])
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(self.channels[1], self.channels[2],
            kernel_size=self.kernel_size)
        self.bn2 = nn.BatchNorm3d(self.channels[2])
        self.layer3 = BasicBlock(self.channels[2], self.channels[3], self.
            channels[4])
        self.layer4 = BasicBlock(self.channels[4], self.channels[5], self.
            channels[6])
        self.layer5 = BasicBlock(self.channels[6], self.channels[7], self.
            channels[8])
        self.fc = nn.Conv3d(self.channels[8], self.channels[9], kernel_size=1)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.fc(x)
        return x


class ASPP_c(nn.Module):

    def __init__(self, dilation_list, channels, kernel_size, num_branches):
        super(ASPP_c, self).__init__()
        channels.insert(-1, 30)
        self.channels = channels
        self.kernel_size = kernel_size
        self.padding_list = dilation_list
        self.dilation_list = dilation_list
        self.num_branches = num_branches
        self.conv1 = nn.Conv3d(self.channels[0], self.channels[1],
            kernel_size=self.kernel_size)
        self.bn1 = nn.BatchNorm3d(self.channels[1])
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(self.channels[1], self.channels[2],
            kernel_size=self.kernel_size)
        self.bn2 = nn.BatchNorm3d(self.channels[2])
        self.layer3 = BasicBlock(self.channels[2], self.channels[3], self.
            channels[4])
        self.layer4 = BasicBlock(self.channels[4], self.channels[5], self.
            channels[6])
        self.layer5 = BasicBlock(self.channels[6], self.channels[7], self.
            channels[8])
        self.aspp = nn.ModuleList()
        for i in range(self.num_branches):
            self.aspp.append(nn.Conv3d(self.channels[8], self.channels[9],
                kernel_size=self.kernel_size, padding=self.padding_list[i],
                dilation=self.dilation_list[i]))
        self.bn9 = nn.BatchNorm3d(4 * self.channels[9])
        self.fc9 = nn.Conv3d(4 * self.channels[9], self.channels[10],
            kernel_size=1)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        aspp_out = []
        for aspp_scale in self.aspp:
            aspp_out.append(aspp_scale(x))
        aspp_out = torch.cat(aspp_out, 1)
        out = self.bn9(aspp_out)
        out = self.relu(out)
        out = self.fc9(out)
        return out


class ASPP_s(nn.Module):

    def __init__(self, dilation_list, channels, kernel_size, num_branches):
        super(ASPP_s, self).__init__()
        channels.insert(-1, 120)
        self.channels = channels
        self.kernel_size = kernel_size
        self.padding_list = dilation_list
        self.dilation_list = dilation_list
        self.num_branches = num_branches
        self.conv1 = nn.Conv3d(self.channels[0], self.channels[1],
            kernel_size=self.kernel_size)
        self.bn1 = nn.BatchNorm3d(self.channels[1])
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(self.channels[1], self.channels[2],
            kernel_size=self.kernel_size)
        self.bn2 = nn.BatchNorm3d(self.channels[2])
        self.layer3 = BasicBlock(self.channels[2], self.channels[3], self.
            channels[4])
        self.layer4 = BasicBlock(self.channels[4], self.channels[5], self.
            channels[6])
        self.layer5 = BasicBlock(self.channels[6], self.channels[7], self.
            channels[8])
        self.conv_list = nn.ModuleList()
        self.bn_list = nn.ModuleList()
        self.last_list = nn.ModuleList()
        for i in range(self.num_branches):
            self.conv_list.append(nn.Conv3d(self.channels[8], self.channels
                [9], kernel_size=self.kernel_size, padding=self.
                padding_list[i], dilation=self.dilation_list[i]))
            self.bn_list.append(nn.BatchNorm3d(self.channels[9]))
            self.last_list.append(nn.Conv3d(self.channels[9], self.channels
                [10], kernel_size=1))
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        out = self.conv_list[0](x)
        out = self.relu(self.bn_list[0](out))
        out = self.last_list[0](out)
        for i in range(1, self.num_branches):
            out1 = self.conv_list[i](x)
            out1 = self.relu(self.bn_list[i](out1))
            out += self.last_list[i](out1)
        return out


class AFN(nn.Module):

    def __init__(self, blocks, padding_list, dilation_list, channels,
        kernel_size, num_branches):
        super(AFN, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.padding_list = padding_list
        self.dilation_list = dilation_list
        self.num_branches = num_branches
        self.blocks = blocks
        self.conv1 = nn.Conv3d(self.channels[0], self.channels[1],
            kernel_size=self.kernel_size)
        self.bn1 = nn.BatchNorm3d(self.channels[1])
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(self.channels[1], self.channels[2],
            kernel_size=self.kernel_size)
        self.bn2 = nn.BatchNorm3d(self.channels[2])
        self.layers = nn.ModuleList()
        for i in range(len(blocks)):
            block = blocks[i]
            index = int(2 * i + 2)
            if block == BasicBlock:
                self.layers.append(block(self.channels[index], self.
                    channels[index + 1], self.channels[index + 2]))
            else:
                self.layers.append(block(self.channels[index], self.
                    channels[index + 1], self.channels[index + 2], self.
                    padding_list, self.dilation_list, self.num_branches))
        self.fc = nn.Conv3d(self.channels[8], self.channels[9], kernel_size=1)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        for layer in self.layers:
            x = layer(x)
        x = self.fc(x)
        return x


class Conv(nn.Module):

    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1),
        groups=1, bn_acti=False, bias=False):
        super().__init__()
        self.bn_acti = bn_acti
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=bias)
        if self.bn_acti:
            self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv(input)
        if self.bn_acti:
            output = self.bn_prelu(output)
        return output


class BNPReLU(nn.Module):

    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=0.001)
        self.acti = nn.PReLU(nIn)

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)
        return output


class DABModule(nn.Module):

    def __init__(self, nIn, d=1, kSize=3, dkSize=3):
        super().__init__()
        self.bn_relu_1 = BNPReLU(nIn)
        self.conv3x3 = Conv(nIn, nIn // 2, kSize, 1, padding=1, bn_acti=True)
        self.dconv3x1 = Conv(nIn // 2, nIn // 2, (dkSize, 1), 1, padding=(1,
            0), groups=nIn // 2, bn_acti=True)
        self.dconv1x3 = Conv(nIn // 2, nIn // 2, (1, dkSize), 1, padding=(0,
            1), groups=nIn // 2, bn_acti=True)
        self.ddconv3x1 = Conv(nIn // 2, nIn // 2, (dkSize, 1), 1, padding=(
            1 * d, 0), dilation=(d, 1), groups=nIn // 2, bn_acti=True)
        self.ddconv1x3 = Conv(nIn // 2, nIn // 2, (1, dkSize), 1, padding=(
            0, 1 * d), dilation=(1, d), groups=nIn // 2, bn_acti=True)
        self.bn_relu_2 = BNPReLU(nIn // 2)
        self.conv1x1 = Conv(nIn // 2, nIn, 1, 1, padding=0, bn_acti=False)

    def forward(self, input):
        output = self.bn_relu_1(input)
        output = self.conv3x3(output)
        br1 = self.dconv3x1(output)
        br1 = self.dconv1x3(br1)
        br2 = self.ddconv3x1(output)
        br2 = self.ddconv1x3(br2)
        output = br1 + br2
        output = self.bn_relu_2(output)
        output = self.conv1x1(output)
        return output + input


class DownSamplingBlock(nn.Module):

    def __init__(self, nIn, nOut):
        super().__init__()
        self.nIn = nIn
        self.nOut = nOut
        if self.nIn < self.nOut:
            nConv = nOut - nIn
        else:
            nConv = nOut
        self.conv3x3 = Conv(nIn, nConv, kSize=3, stride=2, padding=1)
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv3x3(input)
        if self.nIn < self.nOut:
            max_pool = self.max_pool(input)
            output = torch.cat([output, max_pool], 1)
        output = self.bn_prelu(output)
        return output


class InputInjection(nn.Module):

    def __init__(self, ratio):
        super().__init__()
        self.pool = nn.ModuleList()
        for i in range(0, ratio):
            self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, input):
        for pool in self.pool:
            input = pool(input)
        return input


class DABNet(nn.Module):

    def __init__(self, num_classes=19, block_1=3, block_2=6, **kwargs):
        super().__init__()
        self.init_conv = nn.Sequential(Conv(3, 32, 3, 2, padding=1, bn_acti
            =True), Conv(32, 32, 3, 1, padding=1, bn_acti=True), Conv(32, 
            32, 3, 1, padding=1, bn_acti=True))
        self.down_1 = InputInjection(1)
        self.down_2 = InputInjection(2)
        self.down_3 = InputInjection(3)
        self.bn_prelu_1 = BNPReLU(32 + 3)
        self.downsample_1 = DownSamplingBlock(32 + 3, 64)
        self.DAB_Block_1 = nn.Sequential()
        for i in range(0, block_1):
            self.DAB_Block_1.add_module('DAB_Module_1_' + str(i), DABModule
                (64, d=2))
        self.bn_prelu_2 = BNPReLU(128 + 3)
        dilation_block_2 = [4, 4, 8, 8, 16, 16]
        self.downsample_2 = DownSamplingBlock(128 + 3, 128)
        self.DAB_Block_2 = nn.Sequential()
        for i in range(0, block_2):
            self.DAB_Block_2.add_module('DAB_Module_2_' + str(i), DABModule
                (128, d=dilation_block_2[i]))
        self.bn_prelu_3 = BNPReLU(256 + 3)
        self.classifier = nn.Sequential(Conv(259, num_classes, 1, 1, padding=0)
            )

    def forward(self, input):
        output0 = self.init_conv(input)
        down_1 = self.down_1(input)
        down_2 = self.down_2(input)
        down_3 = self.down_3(input)
        output0_cat = self.bn_prelu_1(torch.cat([output0, down_1], 1))
        output1_0 = self.downsample_1(output0_cat)
        output1 = self.DAB_Block_1(output1_0)
        output1_cat = self.bn_prelu_2(torch.cat([output1, output1_0, down_2
            ], 1))
        output2_0 = self.downsample_2(output1_cat)
        output2 = self.DAB_Block_2(output2_0)
        output2_cat = self.bn_prelu_3(torch.cat([output2, output2_0, down_3
            ], 1))
        out = self.classifier(output2_cat)
        out = F.interpolate(out, input.size()[2:], mode='bilinear',
            align_corners=False)
        return out


class _ASPPModule(nn.Module):
    """Atrous Spatial Pyramid Pooling"""

    def __init__(self, in_channels, out_channels, pyramids):
        super(_ASPPModule, self).__init__()
        self.stages = nn.Module()
        for i, (dilation, padding) in enumerate(zip(pyramids, pyramids)):
            self.stages.add_module('c{}'.format(i), nn.Conv2d(in_channels=
                in_channels, out_channels=out_channels, kernel_size=3,
                stride=1, padding=padding, dilation=dilation, bias=True))
        for m in self.stages.children():
            nn.init.normal_(m.weight, mean=0, std=0.01)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        h = 0
        for stage in self.stages.children():
            h += stage(x)
        return h


class DeepLabV2(nn.Sequential):

    def __init__(self, num_classes, n_blocks, pyramids, **kwargs):
        super(DeepLabV2, self).__init__()
        self.add_module('layer1', nn.Sequential(OrderedDict([('conv1',
            _ConvBatchNormReLU(3, 64, 7, 2, 3, 1)), ('pool', nn.MaxPool2d(3,
            2, 1, ceil_mode=True))])))
        self.add_module('layer2', _ResBlock(n_blocks[0], 64, 64, 256, 1, 1))
        self.add_module('layer3', _ResBlock(n_blocks[1], 256, 128, 512, 2, 1))
        self.add_module('layer4', _ResBlock(n_blocks[2], 512, 256, 1024, 1, 2))
        self.add_module('layer5', _ResBlock(n_blocks[3], 1024, 512, 2048, 1, 4)
            )
        self.add_module('aspp', _ASPPModule(2048, num_classes, pyramids))

    def forward(self, x):
        logits = super(DeepLabV2, self).forward(x)
        logits = F.interpolate(logits, size=x.shape[2:], mode='bilinear',
            align_corners=True)
        return logits

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


class _ASPPModule(nn.Module):
    """Atrous Spatial Pyramid Pooling with image pool"""

    def __init__(self, in_channels, out_channels, pyramids):
        super(_ASPPModule, self).__init__()
        self.stages = nn.Module()
        self.stages.add_module('c0', _ConvBatchNormReLU(in_channels,
            out_channels, 1, 1, 0, 1))
        for i, (dilation, padding) in enumerate(zip(pyramids, pyramids)):
            self.stages.add_module('c{}'.format(i + 1), _ConvBatchNormReLU(
                in_channels, out_channels, 3, 1, padding, dilation))
        self.imagepool = nn.Sequential(OrderedDict([('pool', nn.
            AdaptiveAvgPool2d(1)), ('conv', _ConvBatchNormReLU(in_channels,
            out_channels, 1, 1, 0, 1))]))

    def forward(self, x):
        h = self.imagepool(x)
        h = [F.interpolate(h, size=x.shape[2:], mode='bilinear')]
        for stage in self.stages.children():
            h += [stage(x)]
        h = torch.cat(h, dim=1)
        return h


class DeepLabV3(nn.Sequential):
    """DeepLab v3"""

    def __init__(self, n_classes, n_blocks, pyramids, grids, output_stride):
        super(DeepLabV3, self).__init__()
        if output_stride == 8:
            stride = [1, 2, 1, 1]
            dilation = [1, 1, 2, 2]
        elif output_stride == 16:
            stride = [1, 2, 2, 1]
            dilation = [1, 1, 1, 2]
        self.add_module('layer1', nn.Sequential(OrderedDict([('conv1',
            _ConvBatchNormReLU(3, 64, 7, 2, 3, 1)), ('pool', nn.MaxPool2d(3,
            2, 1, ceil_mode=True))])))
        self.add_module('layer2', _ResBlock(n_blocks[0], 64, 64, 256,
            stride[0], dilation[0]))
        self.add_module('layer3', _ResBlock(n_blocks[1], 256, 128, 512,
            stride[1], dilation[1]))
        self.add_module('layer4', _ResBlock(n_blocks[2], 512, 256, 1024,
            stride[2], dilation[2]))
        self.add_module('layer5', _ResBlock(n_blocks[3], 1024, 512, 2048,
            stride[3], dilation[3], mg=grids))
        self.add_module('aspp', _ASPPModule(2048, 256, pyramids))
        self.add_module('fc1', _ConvBatchNormReLU(256 * (len(pyramids) + 2),
            256, 1, 1, 0, 1))
        self.add_module('fc2', nn.Conv2d(256, n_classes, kernel_size=1))

    def forward(self, x):
        logits = super(DeepLabV3, self).forward(x)
        logits = F.interpolate(logits, size=x.shape[2:], mode='bilinear',
            align_corners=True)
        return logits

    def freeze_bn(self):
        for m in self.named_modules():
            if isinstance(m[1], nn.BatchNorm2d):
                m[1].eval()


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, rate=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            dilation=rate, padding=rate, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.rate = rate

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
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, nInputChannels, block, layers, os=16, pretrained=False):
        self.inplanes = 64
        super(ResNet, self).__init__()
        if os == 16:
            strides = [1, 2, 2, 1]
            rates = [1, 1, 1, 2]
            blocks = [1, 2, 4]
        elif os == 8:
            strides = [1, 2, 1, 1]
            rates = [1, 1, 2, 2]
            blocks = [1, 2, 1]
        else:
            raise NotImplementedError
        self.conv1 = nn.Conv2d(nInputChannels, 64, kernel_size=7, stride=2,
            padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides
            [0], rate=rates[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=
            strides[1], rate=rates[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=
            strides[2], rate=rates[2])
        self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=
            strides[3], rate=rates[3])
        self._init_weight()
        if pretrained:
            self._load_pretrained_model()

    def _make_layer(self, block, planes, blocks, stride=1, rate=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, rate, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks=[1, 2, 4], stride=1, rate=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, rate=blocks[0] *
            rate, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1, rate=
                blocks[i] * rate))
        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        low_level_feat = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x, low_level_feat

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self):
        pretrain_dict = model_zoo.load_url(
            'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)


class ASPP_module(nn.Module):

    def __init__(self, inplanes, planes, rate):
        super(ASPP_module, self).__init__()
        if rate == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = rate
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=
            kernel_size, stride=1, padding=padding, dilation=rate, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self._init_weight()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)
        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def ResNet101(nInputChannels=3, os=16, pretrained=False):
    model = ResNet(nInputChannels, Bottleneck, [3, 4, 23, 3], os,
        pretrained=pretrained)
    return model


class DeepLabv3_plus(nn.Module):

    def __init__(self, num_classes=21, pretrained=False, nInputChannels=3,
        os=16, _print=True, **kwargs):
        if _print:
            None
            None
            None
            None
        super(DeepLabv3_plus, self).__init__()
        self.resnet_features = ResNet101(nInputChannels, os, pretrained=
            pretrained)
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError
        self.aspp1 = ASPP_module(2048, 256, rate=rates[0])
        self.aspp2 = ASPP_module(2048, 256, rate=rates[1])
        self.aspp3 = ASPP_module(2048, 256, rate=rates[2])
        self.aspp4 = ASPP_module(2048, 256, rate=rates[3])
        self.relu = nn.ReLU()
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(2048, 256, 1, stride=1, bias=False), nn.BatchNorm2d(
            256), nn.ReLU())
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)
        self.last_linear = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3,
            stride=1, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(
            ), nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias
            =False), nn.BatchNorm2d(256), nn.ReLU(), nn.Conv2d(256,
            num_classes, kernel_size=1, stride=1))

    def forward(self, input):
        x, low_level_features = self.resnet_features(input)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.upsample(x5, size=x4.size()[2:], mode='bilinear',
            align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = F.upsample(x, size=(int(math.ceil(input.size()[-2] / 4)), int(
            math.ceil(input.size()[-1] / 4))), mode='bilinear',
            align_corners=True)
        low_level_features = self.conv2(low_level_features)
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)
        x = torch.cat((x, low_level_features), dim=1)
        x = self.last_linear(x)
        x = F.upsample(x, size=input.size()[2:], mode='bilinear',
            align_corners=True)
        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class SeparableConv2d(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=0,
        dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride,
            padding, dilation, groups=inplanes, bias=bias)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


def fixed_padding(inputs, kernel_size, rate):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs


class SeparableConv2d_same(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=
        1, bias=False):
        super(SeparableConv2d_same, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, 0,
            dilation, groups=inplanes, bias=bias)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = fixed_padding(x, self.conv1.kernel_size[0], rate=self.conv1.
            dilation[0])
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):

    def __init__(self, inplanes, planes, reps, stride=1, dilation=1,
        start_with_relu=True, grow_first=True, is_last=False):
        super(Block, self).__init__()
        if planes != inplanes or stride != 1:
            self.skip = nn.Conv2d(inplanes, planes, 1, stride=stride, bias=
                False)
            self.skipbn = nn.BatchNorm2d(planes)
        else:
            self.skip = None
        self.relu = nn.ReLU(inplace=True)
        rep = []
        filters = inplanes
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d_same(inplanes, planes, 3, stride=1,
                dilation=dilation))
            rep.append(nn.BatchNorm2d(planes))
            filters = planes
        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d_same(filters, filters, 3, stride=1,
                dilation=dilation))
            rep.append(nn.BatchNorm2d(filters))
        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d_same(inplanes, planes, 3, stride=1,
                dilation=dilation))
            rep.append(nn.BatchNorm2d(planes))
        if not start_with_relu:
            rep = rep[1:]
        if stride != 1:
            rep.append(SeparableConv2d_same(planes, planes, 3, stride=2))
        if is_last:
            rep.append(SeparableConv2d_same(planes, planes, 3, stride=1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)
        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp
        x += skip
        return x


class Xception(nn.Module):
    """
    Modified Alighed Xception
    """

    def __init__(self, inplanes=3, os=16, pretrained=False):
        super(Xception, self).__init__()
        if os == 16:
            entry_block3_stride = 2
            middle_block_rate = 1
            exit_block_rates = 1, 2
        elif os == 8:
            entry_block3_stride = 1
            middle_block_rate = 2
            exit_block_rates = 2, 4
        else:
            raise NotImplementedError
        self.conv1 = nn.Conv2d(inplanes, 32, 3, stride=2, padding=1, bias=False
            )
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.block1 = Block(64, 128, reps=2, stride=2, start_with_relu=False)
        self.block2 = Block(128, 256, reps=2, stride=2, start_with_relu=
            True, grow_first=True)
        self.block3 = Block(256, 728, reps=2, stride=entry_block3_stride,
            start_with_relu=True, grow_first=True)
        self.block4 = Block(728, 728, reps=3, stride=1, dilation=
            middle_block_rate, start_with_relu=True, grow_first=True)
        self.block5 = Block(728, 728, reps=3, stride=1, dilation=
            middle_block_rate, start_with_relu=True, grow_first=True)
        self.block6 = Block(728, 728, reps=3, stride=1, dilation=
            middle_block_rate, start_with_relu=True, grow_first=True)
        self.block7 = Block(728, 728, reps=3, stride=1, dilation=
            middle_block_rate, start_with_relu=True, grow_first=True)
        self.block8 = Block(728, 728, reps=3, stride=1, dilation=
            middle_block_rate, start_with_relu=True, grow_first=True)
        self.block9 = Block(728, 728, reps=3, stride=1, dilation=
            middle_block_rate, start_with_relu=True, grow_first=True)
        self.block10 = Block(728, 728, reps=3, stride=1, dilation=
            middle_block_rate, start_with_relu=True, grow_first=True)
        self.block11 = Block(728, 728, reps=3, stride=1, dilation=
            middle_block_rate, start_with_relu=True, grow_first=True)
        self.block12 = Block(728, 728, reps=3, stride=1, dilation=
            middle_block_rate, start_with_relu=True, grow_first=True)
        self.block13 = Block(728, 728, reps=3, stride=1, dilation=
            middle_block_rate, start_with_relu=True, grow_first=True)
        self.block14 = Block(728, 728, reps=3, stride=1, dilation=
            middle_block_rate, start_with_relu=True, grow_first=True)
        self.block15 = Block(728, 728, reps=3, stride=1, dilation=
            middle_block_rate, start_with_relu=True, grow_first=True)
        self.block16 = Block(728, 728, reps=3, stride=1, dilation=
            middle_block_rate, start_with_relu=True, grow_first=True)
        self.block17 = Block(728, 728, reps=3, stride=1, dilation=
            middle_block_rate, start_with_relu=True, grow_first=True)
        self.block18 = Block(728, 728, reps=3, stride=1, dilation=
            middle_block_rate, start_with_relu=True, grow_first=True)
        self.block19 = Block(728, 728, reps=3, stride=1, dilation=
            middle_block_rate, start_with_relu=True, grow_first=True)
        self.block20 = Block(728, 1024, reps=2, stride=1, dilation=
            exit_block_rates[0], start_with_relu=True, grow_first=False,
            is_last=True)
        self.conv3 = SeparableConv2d_same(1024, 1536, 3, stride=1, dilation
            =exit_block_rates[1])
        self.bn3 = nn.BatchNorm2d(1536)
        self.conv4 = SeparableConv2d_same(1536, 1536, 3, stride=1, dilation
            =exit_block_rates[1])
        self.bn4 = nn.BatchNorm2d(1536)
        self.conv5 = SeparableConv2d_same(1536, 2048, 3, stride=1, dilation
            =exit_block_rates[1])
        self.bn5 = nn.BatchNorm2d(2048)
        self.__init_weight()
        if pretrained:
            self.__load_xception_pretrained()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.block1(x)
        low_level_feat = x
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        x = self.block17(x)
        x = self.block18(x)
        x = self.block19(x)
        x = self.block20(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        return x, low_level_feat

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def __load_xception_pretrained(self):
        pretrain_dict = model_zoo.load_url(
            'http://data.lip6.fr/cadene/pretrainedmodels/xception-43020ad28.pth'
            )
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                if k.startswith('block12'):
                    model_dict[k.replace('block12', 'block20')] = v
                elif k.startswith('block11'):
                    model_dict[k.replace('block11', 'block12')] = v
                    model_dict[k.replace('block11', 'block13')] = v
                    model_dict[k.replace('block11', 'block14')] = v
                    model_dict[k.replace('block11', 'block15')] = v
                    model_dict[k.replace('block11', 'block16')] = v
                    model_dict[k.replace('block11', 'block17')] = v
                    model_dict[k.replace('block11', 'block18')] = v
                    model_dict[k.replace('block11', 'block19')] = v
                elif k.startswith('conv3'):
                    model_dict[k] = v
                elif k.startswith('bn3'):
                    model_dict[k] = v
                    model_dict[k.replace('bn3', 'bn4')] = v
                elif k.startswith('conv4'):
                    model_dict[k.replace('conv4', 'conv5')] = v
                elif k.startswith('bn4'):
                    model_dict[k.replace('bn4', 'bn5')] = v
                else:
                    model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)


class ASPP_module(nn.Module):

    def __init__(self, inplanes, planes, rate):
        super(ASPP_module, self).__init__()
        if rate == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = rate
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=
            kernel_size, stride=1, padding=padding, dilation=rate, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.__init_weight()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)
        return self.relu(x)

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class DeepLabv3_plus(nn.Module):

    def __init__(self, num_classes=21, pretrained=False, nInputChannels=3,
        os=16, _print=True, **kwargs):
        if _print:
            None
            None
            None
            None
        super(DeepLabv3_plus, self).__init__()
        self.xception_features = Xception(nInputChannels, os, pretrained)
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError
        self.aspp1 = ASPP_module(2048, 256, rate=rates[0])
        self.aspp2 = ASPP_module(2048, 256, rate=rates[1])
        self.aspp3 = ASPP_module(2048, 256, rate=rates[2])
        self.aspp4 = ASPP_module(2048, 256, rate=rates[3])
        self.relu = nn.ReLU()
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(2048, 256, 1, stride=1, bias=False), nn.BatchNorm2d(
            256), nn.ReLU())
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(128, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)
        self.last_linear = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3,
            stride=1, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(
            ), nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias
            =False), nn.BatchNorm2d(256), nn.ReLU(), nn.Conv2d(256,
            num_classes, kernel_size=1, stride=1))

    def forward(self, input):
        x, low_level_features = self.xception_features(input)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear',
            align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = F.interpolate(x, size=(int(math.ceil(input.size()[-2] / 4)),
            int(math.ceil(input.size()[-1] / 4))), mode='bilinear',
            align_corners=True)
        low_level_features = self.conv2(low_level_features)
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)
        x = torch.cat((x, low_level_features), dim=1)
        x = self.last_linear(x)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear',
            align_corners=True)
        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class DenseLayer(nn.Sequential):

    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_features=in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(in_channels=in_channels,
            out_channels=growth_rate, kernel_size=3, stride=1, padding=1,
            bias=True))
        self.add_module('drop', nn.Dropout2d(0.2))

    def forward(self, x):
        out = super(DenseLayer, self).forward(x)
        return out


class DenseBlock(nn.Module):

    def __init__(self, in_channels, growth_rate, n_layers, upsample=False):
        super(DenseBlock, self).__init__()
        self.upsample = upsample
        self.layers = nn.ModuleList([DenseLayer(in_channels + i *
            growth_rate, growth_rate) for i in range(n_layers)])
        self.SE_upsample1 = nn.Conv2d(growth_rate * n_layers, growth_rate *
            n_layers // 16, kernel_size=1)
        self.SE_upsample2 = nn.Conv2d(growth_rate * n_layers // 16, 
            growth_rate * n_layers, kernel_size=1)
        self.SE1 = nn.Conv2d(in_channels + growth_rate * n_layers, (
            in_channels + growth_rate * n_layers) // 16, kernel_size=1)
        self.SE2 = nn.Conv2d((in_channels + growth_rate * n_layers) // 16, 
            in_channels + growth_rate * n_layers, kernel_size=1)

    def forward(self, x):
        if self.upsample:
            new_features = []
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], 1)
                new_features.append(out)
            out = torch.cat(new_features, 1)
            fm_size = out.size()[2]
            scale_weight = F.avg_pool2d(out, fm_size)
            scale_weight = F.relu(self.SE_upsample1(scale_weight))
            scale_weight = F.sigmoid(self.SE_upsample2(scale_weight))
            out = out * scale_weight.expand_as(out)
            return out
        else:
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], 1)
            fm_size = x.size()[2]
            scale_weight = F.avg_pool2d(x, fm_size)
            scale_weight = F.relu(self.SE1(scale_weight))
            scale_weight = F.sigmoid(self.SE2(scale_weight))
            x = x * scale_weight.expand_as(x)
            return x


class TransitionDown(nn.Sequential):

    def __init__(self, in_channels):
        super(TransitionDown, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_features=in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(in_channels=in_channels,
            out_channels=in_channels, kernel_size=1, stride=1, padding=0,
            bias=True))
        self.add_module('drop', nn.Dropout2d(0.2))
        self.add_module('maxpool', nn.MaxPool2d(2))

    def forward(self, x):
        out = super(TransitionDown, self).forward(x)
        return out


def center_crop(layer, max_height, max_width):
    _, _, h, w = layer.size()
    xy1 = (w - max_width) // 2
    xy2 = (h - max_height) // 2
    return layer[:, :, xy2:xy2 + max_height, xy1:xy1 + max_width]


class TransitionUp(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(TransitionUp, self).__init__()
        self.convTrans = nn.ConvTranspose2d(in_channels=in_channels,
            out_channels=out_channels, kernel_size=3, stride=2, padding=0,
            bias=True)

    def forward(self, x, skip):
        out = self.convTrans(x)
        out = center_crop(out, skip.size(2), skip.size(3))
        out = torch.cat([out, skip], 1)
        return out


class Bottleneck(nn.Sequential):

    def __init__(self, in_channels, growth_rate, n_layers):
        super(Bottleneck, self).__init__()
        self.add_module('bottleneck', DenseBlock(in_channels, growth_rate,
            n_layers, upsample=True))

    def forward(self, x):
        out = super(Bottleneck, self).forward(x)
        return out


class FCDenseNet(nn.Module):

    def __init__(self, in_channels=3, down_blocks=(5, 5, 5, 5, 5),
        up_blocks=(5, 5, 5, 5, 5), bottleneck_layers=5, growth_rate=16,
        out_chans_first_conv=48, n_classes=12):
        super(FCDenseNet, self).__init__()
        self.down_blocks = down_blocks
        self.up_blocks = up_blocks
        cur_channels_count = 0
        skip_connection_channel_counts = []
        self.add_module('firstconv', nn.Conv2d(in_channels=in_channels,
            out_channels=out_chans_first_conv, kernel_size=3, stride=1,
            padding=1, bias=True))
        cur_channels_count = out_chans_first_conv
        self.denseBlocksDown = nn.ModuleList([])
        self.transDownBlocks = nn.ModuleList([])
        for i in range(len(down_blocks)):
            self.denseBlocksDown.append(DenseBlock(cur_channels_count,
                growth_rate, down_blocks[i]))
            cur_channels_count += growth_rate * down_blocks[i]
            skip_connection_channel_counts.insert(0, cur_channels_count)
            self.transDownBlocks.append(TransitionDown(cur_channels_count))
        self.add_module('bottleneck', Bottleneck(cur_channels_count,
            growth_rate, bottleneck_layers))
        prev_block_channels = growth_rate * bottleneck_layers
        cur_channels_count += prev_block_channels
        self.transUpBlocks = nn.ModuleList([])
        self.denseBlocksUp = nn.ModuleList([])
        for i in range(len(up_blocks) - 1):
            self.transUpBlocks.append(TransitionUp(prev_block_channels,
                prev_block_channels))
            cur_channels_count = (prev_block_channels +
                skip_connection_channel_counts[i])
            self.denseBlocksUp.append(DenseBlock(cur_channels_count,
                growth_rate, up_blocks[i], upsample=True))
            prev_block_channels = growth_rate * up_blocks[i]
            cur_channels_count += prev_block_channels
        self.transUpBlocks.append(TransitionUp(prev_block_channels,
            prev_block_channels))
        cur_channels_count = (prev_block_channels +
            skip_connection_channel_counts[-1])
        self.denseBlocksUp.append(DenseBlock(cur_channels_count,
            growth_rate, up_blocks[-1], upsample=False))
        cur_channels_count += growth_rate * up_blocks[-1]
        self.finalConv = nn.Conv2d(in_channels=cur_channels_count,
            out_channels=n_classes, kernel_size=1, stride=1, padding=0,
            bias=True)
        self.softmax = nn.LogSoftmax()

    def forward(self, x):
        out = self.firstconv(x)
        skip_connections = []
        for i in range(len(self.down_blocks)):
            out = self.denseBlocksDown[i](out)
            skip_connections.append(out)
            out = self.transDownBlocks[i](out)
        out = self.bottleneck(out)
        for i in range(len(self.up_blocks)):
            skip = skip_connections.pop()
            out = self.transUpBlocks[i](out, skip)
            out = self.denseBlocksUp[i](out)
        out = self.finalConv(out)
        out = self.softmax(out)
        return out


class Mask(nn.Module):

    def __init__(self, inplanes=21):
        super(Mask, self).__init__()
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        self.bn0 = nn.BatchNorm2d(inplanes)
        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(4)
        self.conv1 = nn.Conv2d(inplanes, 8, kernel_size=5, stride=1,
            padding=2, bias=False)
        self.conv2 = nn.Conv2d(8, 4, kernel_size=5, stride=1, padding=2,
            bias=False)
        self.conv3 = nn.Conv2d(4, 1, kernel_size=5, stride=1, padding=2,
            bias=False)

    def forward(self, x):
        x = self.relu(self.bn0(x))
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.sig(self.conv3(x))
        return x


class Diffuse(nn.Module):

    def __init__(self, inplanes, outplanes=64, clamp=False):
        super(Diffuse, self).__init__()
        self.alpha = Parameter(torch.Tensor(1))
        self.beta = Parameter(torch.Tensor(1))
        self.alpha.data.fill_(0)
        self.beta.data.fill_(0)
        self.clamp = clamp
        self.softmax = nn.Softmax(2)
        self.conv = nn.Conv2d(in_channels=inplanes, out_channels=outplanes,
            kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(outplanes)

    def forward(self, F, pred, seed):
        b, c, h, w = pred.size()
        F = self.bn(self.conv(F))
        F = nn.functional.adaptive_max_pool2d(F, (h, w))
        F = F.view(b, -1, h * w)
        W = torch.bmm(F.transpose(1, 2), F)
        P = self.softmax(W)
        if self.clamp:
            self.alpha.data = torch.clamp(self.alpha.data, 0, 1)
            self.beta.data = torch.clamp(self.beta.data, 0, 1)
        pred_vec = pred.view(b, c, -1)
        out_vec = torch.bmm(P, pred_vec.transpose(1, 2)).transpose(1, 2
            ).contiguous()
        out = 1 / (1 + torch.exp(self.beta)) * (1 / (1 + torch.exp(self.
            alpha)) * out_vec.view(b, c, h, w) + torch.exp(self.alpha) / (1 +
            torch.exp(self.alpha)) * seed) + torch.exp(self.beta) / (1 +
            torch.exp(self.beta)) * pred
        return out, P


affine_par = True


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None
        ):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        self.relu = nn.ReLU(inplace=True)
        padding = dilation
        self.conv2 = conv3x3(planes, planes, stride=1, padding=padding,
            dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None
        ):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=
            stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
            padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine=affine_par)
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
        out += residual
        out = self.relu(out)
        return out


class Classifier_Module(nn.Module):

    def __init__(self, dilation_series, padding_series, num_classes, inplane):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(nn.Conv2d(inplane, num_classes,
                kernel_size=3, stride=1, padding=padding, dilation=dilation,
                bias=True))
        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes, isseed=True):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1,
            ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
            dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
            dilation=4)
        if isseed:
            if block.__name__ == 'Bottleneck':
                self.layer5 = self._make_pred_layer(Classifier_Module, [6, 
                    12, 18, 24], [6, 12, 18, 24], num_classes, 2048)
            else:
                self.layer5 = self._make_pred_layer(Classifier_Module, [6, 
                    12, 18, 24], [6, 12, 18, 24], num_classes, 512)
        self.isseed = isseed
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if (stride != 1 or self.inplanes != planes * block.expansion or 
            dilation == 2 or dilation == 4):
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par))
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=
            dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def _make_pred_layer(self, block, dilation_series, padding_series,
        num_classes, inplane):
        return block(dilation_series, padding_series, num_classes, inplane)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        if self.isseed:
            out = self.layer5(x4)
        else:
            out = x1, x2, x3, x4
        return out


class DifNet(nn.Module):

    def __init__(self, num_classes, layers, **kwargs):
        super(DifNet, self).__init__()
        if layers <= 34:
            self.diffuse0 = Diffuse(3)
            self.diffuse1 = Diffuse(64)
            self.diffuse2 = Diffuse(128)
            self.diffuse3 = Diffuse(256)
            self.diffuse4 = Diffuse(512)
        else:
            self.diffuse0 = Diffuse(3)
            self.diffuse1 = Diffuse(64 * 4)
            self.diffuse2 = Diffuse(128 * 4)
            self.diffuse3 = Diffuse(256 * 4)
            self.diffuse4 = Diffuse(512 * 4)
        if layers == 18:
            self.model_sed = ResNet(BasicBlock, [2, 2, 2, 2], num_classes)
            self.model_dif = ResNet(BasicBlock, [2, 2, 2, 2], num_classes,
                isseed=False)
        elif layers == 34:
            self.model_sed = ResNet(BasicBlock, [3, 4, 6, 3], num_classes)
            self.model_dif = ResNet(BasicBlock, [3, 4, 6, 3], num_classes,
                isseed=False)
        elif layers == 50:
            self.model_sed = ResNet(Bottleneck, [3, 4, 6, 3], num_classes)
            self.model_dif = ResNet(Bottleneck, [3, 4, 6, 3], num_classes,
                isseed=False)
        elif layers == 101:
            self.model_sed = ResNet(Bottleneck, [3, 4, 23, 3], num_classes)
            self.model_dif = ResNet(Bottleneck, [3, 4, 23, 3], num_classes,
                isseed=False)
        elif layers == 152:
            self.model_sed = ResNet(Bottleneck, [3, 8, 36, 3], num_classes)
            self.model_dif = ResNet(Bottleneck, [3, 8, 36, 3], num_classes,
                isseed=False)
        elif layers == 1850:
            self.model_sed = ResNet(BasicBlock, [3, 2, 2, 2], num_classes)
            self.model_dif = ResNet(Bottleneck, [3, 4, 6, 3], num_classes,
                isseed=False)
        else:
            None
            exit()
        self.mask_layer = Mask(inplanes=num_classes)

    def get_alpha(self):
        return torch.stack((self.diffuse0.alpha.data, self.diffuse1.alpha.
            data, self.diffuse2.alpha.data, self.diffuse3.alpha.data, self.
            diffuse4.alpha.data)).t()

    def get_beta(self):
        return torch.stack((self.diffuse0.beta.data, self.diffuse1.beta.
            data, self.diffuse2.beta.data, self.diffuse3.beta.data, self.
            diffuse4.beta.data)).t()

    def forward(self, x):
        sed = self.model_sed(x)
        sed_out = sed.clone()
        mask = self.mask_layer(sed)
        sed = sed * mask
        dif = self.model_dif(x)
        pred0, P0 = self.diffuse0(x, sed, sed)
        pred1, P1 = self.diffuse1(dif[0], pred0, sed)
        pred2, P2 = self.diffuse2(dif[1], pred1, sed)
        pred3, P3 = self.diffuse3(dif[2], pred2, sed)
        pred4, P4 = self.diffuse4(dif[3], pred3, sed)
        return F.interpolate(pred4, size=x.shape[2:], mode='bilinear',
            align_corners=True)


class BasicBlock(nn.Module):
    """ResNet BasicBlock
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=
        None, previous_dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=
            stride, padding=dilation, dilation=dilation, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
            padding=previous_dilation, dilation=previous_dilation, bias=False)
        self.bn2 = norm_layer(planes)
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


class Bottleneck(nn.Module):
    """ResNet Bottleneck
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=
        None, previous_dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=dilation, dilation=dilation, bias=False)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def _sum_each(self, x, y):
        assert len(x) == len(y)
        z = []
        for i in range(len(x)):
            z.append(x[i] + y[i])
        return z

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
        out += residual
        out = self.relu(out)
        return out


class DilatedResNet(nn.Module):
    """Dilated Pre-trained ResNet Model, which preduces the stride of 8 featuremaps at conv5.

    Parameters
    ----------
    block : Block
        Class for the residual block. Options are BasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    classes : int, default 1000
        Number of classification classes.
    dilated : bool, default False
        Applying dilation strategy to pretrained ResNet yielding a stride-8 model,
        typically used in Semantic Segmentation.
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).

    Reference:

        - He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

        - Yu, Fisher, and Vladlen Koltun. "Multi-scale context aggregation by dilated convolutions."
    """

    def __init__(self, block, layers, num_classes=1000, dilated=True,
        norm_layer=nn.BatchNorm2d):
        self.inplanes = 64
        super(DilatedResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=
            norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
            norm_layer=norm_layer)
        if dilated:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                dilation=2, norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                dilation=4, norm_layer=norm_layer)
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                norm_layer=norm_layer)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, norm_layer):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1,
        norm_layer=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion))
        layers = []
        if dilation == 1 or dilation == 2:
            layers.append(block(self.inplanes, planes, stride, dilation=1,
                downsample=downsample, previous_dilation=dilation,
                norm_layer=norm_layer))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, dilation=2,
                downsample=downsample, previous_dilation=dilation,
                norm_layer=norm_layer))
        else:
            raise RuntimeError('=> unknown dilation size: {}'.format(dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation,
                previous_dilation=dilation, norm_layer=norm_layer))
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
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class _EncHead(nn.Module):

    def __init__(self, in_channels, nclass, se_loss=True, lateral=True,
        norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_EncHead, self).__init__()
        self.lateral = lateral
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels, 512, 3, padding=1,
            bias=False), norm_layer(512, **{} if norm_kwargs is None else
            norm_kwargs), nn.ReLU(True))
        if lateral:
            self.connect = nn.ModuleList([nn.Sequential(nn.Conv2d(512, 512,
                1, bias=False), norm_layer(512, **{} if norm_kwargs is None
                 else norm_kwargs), nn.ReLU(True)), nn.Sequential(nn.Conv2d
                (1024, 512, 1, bias=False), norm_layer(512, **{} if 
                norm_kwargs is None else norm_kwargs), nn.ReLU(True))])
            self.fusion = nn.Sequential(nn.Conv2d(3 * 512, 512, 3, padding=
                1, bias=False), norm_layer(512, **{} if norm_kwargs is None
                 else norm_kwargs), nn.ReLU(True))
        self.encmodule = EncModule(512, nclass, ncodes=32, se_loss=se_loss,
            norm_layer=norm_layer, norm_kwargs=norm_kwargs, **kwargs)
        self.conv6 = nn.Sequential(nn.Dropout(0.1, False), nn.Conv2d(512,
            nclass, 1))

    def forward(self, *inputs):
        feat = self.conv5(inputs[-1])
        if self.lateral:
            c2 = self.connect[0](inputs[1])
            c3 = self.connect[1](inputs[2])
            feat = self.fusion(torch.cat([feat, c2, c3], 1))
        outs = list(self.encmodule(feat))
        outs[0] = self.conv6(outs[0])
        return tuple(outs)


class EncModule(nn.Module):

    def __init__(self, in_channels, nclass, ncodes=32, se_loss=True,
        norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(EncModule, self).__init__()
        self.se_loss = se_loss
        self.encoding = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1,
            bias=False), norm_layer(in_channels, **{} if norm_kwargs is
            None else norm_kwargs), nn.ReLU(True), Encoding(D=in_channels,
            K=ncodes), nn.BatchNorm1d(ncodes), nn.ReLU(True), Mean(dim=1))
        self.fc = nn.Sequential(nn.Linear(in_channels, in_channels), nn.
            Sigmoid())
        if self.se_loss:
            self.selayer = nn.Linear(in_channels, nclass)

    def forward(self, x):
        en = self.encoding(x)
        b, c, _, _ = x.size()
        gamma = self.fc(en)
        y = gamma.view(b, c, 1, 1)
        outputs = [F.relu_(x + x * y)]
        if self.se_loss:
            outputs.append(self.selayer(en))
        return tuple(outputs)


class Encoding(nn.Module):

    def __init__(self, D, K):
        super(Encoding, self).__init__()
        self.D, self.K = D, K
        self.codewords = nn.Parameter(torch.Tensor(K, D), requires_grad=True)
        self.scale = nn.Parameter(torch.Tensor(K), requires_grad=True)
        self.reset_params()

    def reset_params(self):
        std1 = 1.0 / (self.K * self.D) ** (1 / 2)
        self.codewords.data.uniform_(-std1, std1)
        self.scale.data.uniform_(-1, 0)

    def forward(self, X):
        assert X.size(1) == self.D
        B, D = X.size(0), self.D
        if X.dim() == 3:
            X = X.transpose(1, 2).contiguous()
        elif X.dim() == 4:
            X = X.view(B, D, -1).transpose(1, 2).contiguous()
        else:
            raise RuntimeError('Encoding Layer unknown input dims!')
        A = F.softmax(self.scale_l2(X, self.codewords, self.scale), dim=2)
        E = self.aggregate(A, X, self.codewords)
        return E

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'N x' + str(self.D
            ) + '=>' + str(self.K) + 'x' + str(self.D) + ')'

    @staticmethod
    def scale_l2(X, C, S):
        S = S.view(1, 1, C.size(0), 1)
        X = X.unsqueeze(2).expand(X.size(0), X.size(1), C.size(0), C.size(1))
        C = C.unsqueeze(0).unsqueeze(0)
        SL = S * (X - C)
        SL = SL.pow(2).sum(3)
        return SL

    @staticmethod
    def aggregate(A, X, C):
        A = A.unsqueeze(3)
        X = X.unsqueeze(2).expand(X.size(0), X.size(1), C.size(0), C.size(1))
        C = C.unsqueeze(0).unsqueeze(0)
        E = A * (X - C)
        E = E.sum(1)
        return E


class Mean(nn.Module):

    def __init__(self, dim, keep_dim=False):
        super(Mean, self).__init__()
        self.dim = dim
        self.keep_dim = keep_dim

    def forward(self, input):
        return input.mean(self.dim, self.keep_dim)


class CBR(nn.Module):
    """
    This class defines the convolution layer with batch normalization and PReLU activation
    """

    def __init__(self, nIn, nOut, kSize, stride=1):
        """

        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride,
            padding=(padding, padding), bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=0.001)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        """
        :param input: input feature map
        :return: transformed feature map
        """
        output = self.conv(input)
        output = self.bn(output)
        output = self.act(output)
        return output


class BR(nn.Module):
    """
        This class groups the batch normalization and PReLU activation
    """

    def __init__(self, nOut):
        """
        :param nOut: output feature maps
        """
        super().__init__()
        self.bn = nn.BatchNorm2d(nOut, eps=0.001)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        """
        :param input: input feature map
        :return: normalized and thresholded feature map
        """
        output = self.bn(input)
        output = self.act(output)
        return output


class CB(nn.Module):
    """
       This class groups the convolution and batch normalization
    """

    def __init__(self, nIn, nOut, kSize, stride=1):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optinal stide for down-sampling
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride,
            padding=(padding, padding), bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=0.001)

    def forward(self, input):
        """

        :param input: input feature map
        :return: transformed feature map
        """
        output = self.conv(input)
        output = self.bn(output)
        return output


class C(nn.Module):
    """
    This class is for a convolutional layer.
    """

    def __init__(self, nIn, nOut, kSize, stride=1):
        """

        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride,
            padding=(padding, padding), bias=False)

    def forward(self, input):
        """
        :param input: input feature map
        :return: transformed feature map
        """
        output = self.conv(input)
        return output


class CDilated(nn.Module):
    """
    This class defines the dilated convolution.
    """

    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        """
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride,
            padding=(padding, padding), bias=False, dilation=d)

    def forward(self, input):
        """
        :param input: input feature map
        :return: transformed feature map
        """
        output = self.conv(input)
        return output


class DownSamplerB(nn.Module):

    def __init__(self, nIn, nOut):
        super().__init__()
        n = int(nOut / 5)
        n1 = nOut - 4 * n
        self.c1 = C(nIn, n, 3, 2)
        self.d1 = CDilated(n, n1, 3, 1, 1)
        self.d2 = CDilated(n, n, 3, 1, 2)
        self.d4 = CDilated(n, n, 3, 1, 4)
        self.d8 = CDilated(n, n, 3, 1, 8)
        self.d16 = CDilated(n, n, 3, 1, 16)
        self.bn = nn.BatchNorm2d(nOut, eps=0.001)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        output1 = self.c1(input)
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        d16 = self.d16(output1)
        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16
        combine = torch.cat([d1, add1, add2, add3, add4], 1)
        output = self.bn(combine)
        output = self.act(output)
        return output


class DilatedParllelResidualBlockB(nn.Module):
    """
    This class defines the ESP block, which is based on the following principle
        Reduce ---> Split ---> Transform --> Merge
    """

    def __init__(self, nIn, nOut, add=True):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param add: if true, add a residual connection through identity operation. You can use projection too as
                in ResNet paper, but we avoid to use it if the dimensions are not the same because we do not want to
                increase the module complexity
        """
        super().__init__()
        n = int(nOut / 5)
        n1 = nOut - 4 * n
        self.c1 = C(nIn, n, 1, 1)
        self.d1 = CDilated(n, n1, 3, 1, 1)
        self.d2 = CDilated(n, n, 3, 1, 2)
        self.d4 = CDilated(n, n, 3, 1, 4)
        self.d8 = CDilated(n, n, 3, 1, 8)
        self.d16 = CDilated(n, n, 3, 1, 16)
        self.bn = BR(nOut)
        self.add = add

    def forward(self, input):
        """
        :param input: input feature map
        :return: transformed feature map
        """
        output1 = self.c1(input)
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        d16 = self.d16(output1)
        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16
        combine = torch.cat([d1, add1, add2, add3, add4], 1)
        if self.add:
            combine = input + combine
        output = self.bn(combine)
        return output


class InputProjectionA(nn.Module):
    """
    This class projects the input image to the same spatial dimensions as the feature map.
    For example, if the input image is 512 x512 x3 and spatial dimensions of feature map size are 56x56xF, then
    this class will generate an output of 56x56x3
    """

    def __init__(self, samplingTimes):
        """
        :param samplingTimes: The rate at which you want to down-sample the image
        """
        super().__init__()
        self.pool = nn.ModuleList()
        for i in range(0, samplingTimes):
            self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, input):
        """
        :param input: Input RGB Image
        :return: down-sampled image (pyramid-based approach)
        """
        for pool in self.pool:
            input = pool(input)
        return input


class ESPNet_Encoder(nn.Module):
    """
    This class defines the ESPNet-C network in the paper
    """

    def __init__(self, classes=20, p=5, q=3):
        """
        :param classes: number of classes in the dataset. Default is 20 for the cityscapes
        :param p: depth multiplier
        :param q: depth multiplier
        """
        super().__init__()
        self.level1 = CBR(3, 16, 3, 2)
        self.sample1 = InputProjectionA(1)
        self.sample2 = InputProjectionA(2)
        self.b1 = BR(16 + 3)
        self.level2_0 = DownSamplerB(16 + 3, 64)
        self.level2 = nn.ModuleList()
        for i in range(0, p):
            self.level2.append(DilatedParllelResidualBlockB(64, 64))
        self.b2 = BR(128 + 3)
        self.level3_0 = DownSamplerB(128 + 3, 128)
        self.level3 = nn.ModuleList()
        for i in range(0, q):
            self.level3.append(DilatedParllelResidualBlockB(128, 128))
        self.b3 = BR(256)
        self.classifier = C(256, classes, 1, 1)

    def forward(self, input):
        """
        :param input: Receives the input RGB image
        :return: the transformed feature map with spatial dimensions 1/8th of the input image
        """
        output0 = self.level1(input)
        inp1 = self.sample1(input)
        inp2 = self.sample2(input)
        output0_cat = self.b1(torch.cat([output0, inp1], 1))
        output1_0 = self.level2_0(output0_cat)
        for i, layer in enumerate(self.level2):
            if i == 0:
                output1 = layer(output1_0)
            else:
                output1 = layer(output1)
        output1_cat = self.b2(torch.cat([output1, output1_0, inp2], 1))
        output2_0 = self.level3_0(output1_cat)
        for i, layer in enumerate(self.level3):
            if i == 0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)
        output2_cat = self.b3(torch.cat([output2_0, output2], 1))
        classifier = self.classifier(output2_cat)
        return classifier


class ESPNet(nn.Module):
    """
    This class defines the ESPNet network
    """

    def __init__(self, num_classes=20, p=2, q=3, encoderFile=None):
        """
        :param num_classes: number of classes in the dataset. Default is 20 for the cityscapes
        :param p: depth multiplier
        :param q: depth multiplier
        :param encoderFile: pretrained encoder weights. Recall that we first trained the ESPNet-C and then attached the
                            RUM-based light weight decoder. See paper for more details.
        """
        super().__init__()
        self.encoder = ESPNet_Encoder(num_classes, p, q)
        if encoderFile != None:
            self.encoder.load_state_dict(torch.load(encoderFile))
            None
        self.modules = []
        for i, m in enumerate(self.encoder.children()):
            self.modules.append(m)
        self.level3_C = C(128 + 3, num_classes, 1, 1)
        self.br = nn.BatchNorm2d(num_classes, eps=0.001)
        self.conv = CBR(19 + num_classes, num_classes, 3, 1)
        self.up_l3 = nn.Sequential(nn.ConvTranspose2d(num_classes,
            num_classes, 2, stride=2, padding=0, output_padding=0, bias=False))
        self.combine_l2_l3 = nn.Sequential(BR(2 * num_classes),
            DilatedParllelResidualBlockB(2 * num_classes, num_classes, add=
            False))
        self.up_l2 = nn.Sequential(nn.ConvTranspose2d(num_classes,
            num_classes, 2, stride=2, padding=0, output_padding=0, bias=
            False), BR(num_classes))
        self.classifier = nn.ConvTranspose2d(num_classes, num_classes, 2,
            stride=2, padding=0, output_padding=0, bias=False)

    def forward(self, input):
        """
        :param input: RGB image
        :return: transformed feature map
        """
        output0 = self.modules[0](input)
        inp1 = self.modules[1](input)
        inp2 = self.modules[2](input)
        output0_cat = self.modules[3](torch.cat([output0, inp1], 1))
        output1_0 = self.modules[4](output0_cat)
        for i, layer in enumerate(self.modules[5]):
            if i == 0:
                output1 = layer(output1_0)
            else:
                output1 = layer(output1)
        output1_cat = self.modules[6](torch.cat([output1, output1_0, inp2], 1))
        output2_0 = self.modules[7](output1_cat)
        for i, layer in enumerate(self.modules[8]):
            if i == 0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)
        output2_cat = self.modules[9](torch.cat([output2_0, output2], 1))
        output2_c = self.up_l3(self.br(self.modules[10](output2_cat)))
        output1_C = self.level3_C(output1_cat)
        comb_l2_l3 = self.up_l2(self.combine_l2_l3(torch.cat([output1_C,
            output2_c], 1)))
        concat_features = self.conv(torch.cat([comb_l2_l3, output0_cat], 1))
        classifier = self.classifier(concat_features)
        return classifier


class CNA(nn.Module):

    def __init__(self, in_c, out_c, kernel_size=3, stride=1, padding=1,
        norm=nn.InstanceNorm2d, act=nn.ReLU):
        super(CNA, self).__init__()
        self.layer = nn.Sequential(nn.Conv2d(in_c, out_c, kernel_size,
            stride, padding), norm(out_c), act(True))

    def forward(self, x):
        return self.layer(x)


class UpCNA(nn.Module):

    def __init__(self, in_c, out_c, kernel_size=4, stride=2, padding=1,
        norm=nn.InstanceNorm2d, act=nn.ReLU):
        super(UpCNA, self).__init__()
        self.layer = nn.Sequential(nn.ConvTranspose2d(in_c, out_c,
            kernel_size, stride, padding), norm(out_c), act(True))

    def forward(self, x):
        return self.layer(x)


class SEB_dw(nn.Module):

    def __init__(self, low_feature, high_feature, norm=nn.InstanceNorm2d,
        up_scale=2):
        super(SEB_dw, self).__init__()
        self.conv = CNA(high_feature, low_feature, norm=norm)
        self.up = nn.UpsamplingBilinear2d(scale_factor=up_scale)

    def forward(self, low_feature, high_feature):
        high_feature = self.conv(high_feature)
        high_feature = self.up(high_feature)
        return low_feature * high_feature


class SEB(nn.Module):

    def __init__(self, low_feature, high_features, norm=nn.InstanceNorm2d,
        up_scale=2):
        super(SEB, self).__init__()
        self.sebs = []
        for c in range(len(high_features) - 1, 0, -1):
            self.sebs.append(nn.Sequential(CNA(high_features[c],
                high_features[c - 1], norm=norm), nn.UpsamplingBilinear2d(
                scale_factor=up_scale)))

    def forward(self, low_feature, *high_features):
        high_features = reversed(high_features)
        low_feature = self.seb[0](high_features[0]) * high_features[1]
        for c in range(1, len(high_features)):
            high_feature = self.sebs[c](high_features[c])
            low_feature *= high_feature
        return low_feature


class GCN(nn.Module):

    def __init__(self, in_c, out_c, ks=7, norm=nn.InstanceNorm2d):
        super(GCN, self).__init__()
        self.conv_l1 = CNA(in_c, out_c, kernel_size=(ks, 1), padding=(ks //
            2, 0), norm=norm)
        self.conv_l2 = CNA(out_c, out_c, kernel_size=(1, ks), padding=(0, 
            ks // 2), norm=norm)
        self.conv_r1 = CNA(in_c, out_c, kernel_size=(1, ks), padding=(0, ks //
            2), norm=norm)
        self.conv_r2 = CNA(out_c, out_c, kernel_size=(ks, 1), padding=(ks //
            2, 0), norm=norm)

    def forward(self, x):
        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)
        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)
        return x_l + x_r


class ECRE(nn.Module):

    def __init__(self, in_c, up_scale=2, norm=nn.InstanceNorm2d):
        super(ECRE, self).__init__()
        self.ecre = nn.Sequential(CNA(in_c, in_c * up_scale * up_scale,
            norm=norm), nn.PixelShuffle(up_scale))

    def forward(self, input_):
        return self.ecre(input_)


class DAP(nn.Module):

    def __init__(self, in_c, k=3, norm=nn.InstanceNorm2d):
        super(DAP, self).__init__()
        self.k2 = k * k
        self.conv = CNA(in_c, in_c * k * k, norm=norm)
        self.padd = nn.ZeroPad2d(k // 2)

    def forward(self, input_):
        batch, input_c, max_i, max_j = input_.shape
        x = self.conv(input_)
        x = self.padd(x)
        a = [(0, max_i, 0, max_j), (0, max_i, 1, max_j + 1), (0, max_i, 2, 
            max_j + 2), (1, max_i + 1, 0, max_j), (1, max_i + 1, 1, max_j +
            1), (1, max_i + 1, 2, max_j + 2), (2, max_i + 2, 0, max_j), (2,
            max_i + 2, 1, max_j + 1), (2, max_i + 2, 2, max_j + 2)]
        R = torch.zeros([batch, input_c, self.k2, max_i * max_j])
        for dap_c in range(input_c):
            for c, (s_i, e_i, s_j, e_j) in enumerate(a):
                R[:, (dap_c), (c)] = x[:, (c), s_i:e_i, s_j:e_j].contiguous(
                    ).view(batch, -1)
        R = torch.mean(R, 2).reshape(batch, input_c, max_i, max_j)
        return R


class ExFuseLevel(nn.Module):

    def __init__(self, in_c, out_c=21, norm=nn.InstanceNorm2d):
        super(ExFuseLevel, self).__init__()
        self.seb = SEB(in_c * 2, in_c, norm=norm)
        self.gcn = GCN(in_c, out_c, norm=norm)
        self.upconv = nn.Sequential(nn.ConvTranspose2d(out_c, out_c, 3, 2, 
            1, output_padding=1), norm(out_c), nn.ReLU(True))

    def forward(self, low_level, high_level, prev_feature):
        level = self.seb(low_level, high_level)
        level = self.gcn(level)
        return self.upconv(level + prev_feature)


class UnetExFuseLevel(nn.Module):

    def __init__(self, in_c, out_c=21, norm=nn.InstanceNorm2d):
        super(UnetExFuseLevel, self).__init__()
        self.seb = SEB(in_c * 2, in_c, norm=norm)
        self.gcn = GCN(in_c, in_c, norm=norm)
        self.upconv = nn.Sequential(nn.ConvTranspose2d(in_c, out_c, 3, 2, 1,
            output_padding=1), norm(out_c), nn.ReLU(True))

    def forward(self, low_level, high_level, prev_feature):
        level = self.seb(low_level, high_level)
        level = self.gcn(level)
        return self.upconv(level + prev_feature)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


class UnetGCN(nn.Module):

    def __init__(self, feature_scale=4, n_classes=1, is_deconv=True, norm=
        nn.InstanceNorm2d, is_pool=True):
        super(UnetGCN, self).__init__()
        filters = [64, 128, 256, 512, 1024]
        filters = [(x // feature_scale) for x in filters]
        self.conv1 = UnetConv2D(1, filters[0], norm)
        self.gcn1 = GCN(filters[0], filters[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=2) if is_pool else ConvBNReLU(
            filters[0], filters[0], norm, stride=2)
        self.conv2 = UnetConv2D(filters[0], filters[1], norm)
        self.gcn2 = GCN(filters[1], filters[1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=2) if is_pool else ConvBNReLU(
            filters[1], filters[1], norm, stride=2)
        self.conv3 = UnetConv2D(filters[1], filters[2], norm)
        self.gcn3 = GCN(filters[2], filters[2])
        self.maxpool3 = nn.MaxPool2d(kernel_size=2) if is_pool else ConvBNReLU(
            filters[2], filters[2], norm, stride=2)
        self.conv4 = UnetConv2D(filters[2], filters[3], norm)
        self.gcn4 = GCN(filters[3], filters[3])
        self.maxpool4 = nn.MaxPool2d(kernel_size=2) if is_pool else ConvBNReLU(
            filters[3], filters[3], norm, stride=2)
        self.center = UnetConv2D(filters[3], filters[4], norm)
        self.up_concat4 = UnetUpConv2D(filters[4], filters[3], norm, is_deconv)
        self.up_concat3 = UnetUpConv2D(filters[3], filters[2], norm, is_deconv)
        self.up_concat2 = UnetUpConv2D(filters[2], filters[1], norm, is_deconv)
        self.up_concat1 = UnetUpConv2D(filters[1], filters[0], norm, is_deconv)
        self.final = nn.Conv2d(filters[0], n_classes, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.apply(weights_init_kaiming)
            elif isinstance(m, nn.BatchNorm2d):
                m.apply(weights_init_kaiming)

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        conv1 = self.gcn1(conv1)
        maxpool1 = self.maxpool1(conv1)
        conv2 = self.conv2(maxpool1)
        conv2 = self.gcn2(conv2)
        maxpool2 = self.maxpool2(conv2)
        conv3 = self.conv3(maxpool2)
        conv3 = self.gcn3(conv3)
        maxpool3 = self.maxpool3(conv3)
        conv4 = self.conv4(maxpool3)
        conv4 = self.gcn4(conv4)
        maxpool4 = self.maxpool4(conv4)
        center = self.center(maxpool4)
        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)
        final = self.final(up1)
        return final


class UnetGCNSEB(nn.Module):

    def __init__(self, feature_scale=4, n_classes=1, is_deconv=True, norm=
        nn.InstanceNorm2d, is_pool=True):
        super(UnetGCNSEB, self).__init__()
        filters = [64, 128, 256, 512, 1024]
        filters = [(x // feature_scale) for x in filters]
        self.conv1 = UnetConv2D(1, filters[0], norm)
        self.gcn1 = GCN(filters[0], filters[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=2) if is_pool else ConvBNReLU(
            filters[0], filters[0], norm, stride=2)
        self.conv2 = UnetConv2D(filters[0], filters[1], norm)
        self.gcn2 = GCN(filters[1], filters[1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=2) if is_pool else ConvBNReLU(
            filters[1], filters[1], norm, stride=2)
        self.conv3 = UnetConv2D(filters[1], filters[2], norm)
        self.gcn3 = GCN(filters[2], filters[2])
        self.maxpool3 = nn.MaxPool2d(kernel_size=2) if is_pool else ConvBNReLU(
            filters[2], filters[2], norm, stride=2)
        self.conv4 = UnetConv2D(filters[2], filters[3], norm)
        self.gcn4 = GCN(filters[3], filters[3])
        self.maxpool4 = nn.MaxPool2d(kernel_size=2) if is_pool else ConvBNReLU(
            filters[3], filters[3], norm, stride=2)
        self.center = UnetConv2D(filters[3], filters[4], norm)
        self.up_concat4 = SEB(filters[4], filters[3])
        self.up_concat3 = SEB(filters[3], filters[2])
        self.up_concat2 = SEB(filters[2], filters[1])
        self.up_concat1 = SEB(filters[1], filters[0])
        self.final = nn.Conv2d(filters[0], 1, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.apply(weights_init_kaiming)
            elif isinstance(m, nn.BatchNorm2d):
                m.apply(weights_init_kaiming)

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        conv1 = self.gcn1(conv1)
        maxpool1 = self.maxpool1(conv1)
        conv2 = self.conv2(maxpool1)
        conv2 = self.gcn2(conv2)
        maxpool2 = self.maxpool2(conv2)
        conv3 = self.conv3(maxpool2)
        conv3 = self.gcn3(conv3)
        maxpool3 = self.maxpool3(conv3)
        conv4 = self.conv4(maxpool3)
        conv4 = self.gcn4(conv4)
        maxpool4 = self.maxpool4(conv4)
        center = self.center(maxpool4)
        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)
        final = self.final(up1)
        return final


class UnetUpECRE(nn.Module):

    def __init__(self, in_size, out_size, norm, is_deconv=False):
        super(UnetUpECRE, self).__init__()
        self.conv = UnetConv2D(in_size + out_size, out_size, norm)
        self.up = ECRE(in_size)
        for m in self.children():
            if m.__class__.__name__.find('UnetConv2D') != -1:
                continue
            m.apply(weights_init_kaiming)

    def forward(self, input1, input2):
        output2 = self.up(input2)
        offset = output2.size()[2] - input1.size()[2]
        padding = [offset // 2] * 4
        output1 = F.pad(input1, padding)
        output = torch.cat([output1, output2], 1)
        return self.conv(output), output2


class UnetGCNECRE(nn.Module):

    def __init__(self, feature_scale=4, n_classes=1, is_deconv=True, norm=
        nn.InstanceNorm2d, is_pool=True):
        super(UnetGCNECRE, self).__init__()
        filters = [64, 128, 256, 512, 1024]
        filters = [(x // feature_scale) for x in filters]
        self.conv1 = UnetConv2D(1, filters[0], norm)
        self.gcn1 = GCN(filters[0], filters[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=2) if is_pool else ConvBNReLU(
            filters[0], filters[0], norm, stride=2)
        self.conv2 = UnetConv2D(filters[0], filters[1], norm)
        self.gcn2 = GCN(filters[1], filters[1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=2) if is_pool else ConvBNReLU(
            filters[1], filters[1], norm, stride=2)
        self.conv3 = UnetConv2D(filters[1], filters[2], norm)
        self.gcn3 = GCN(filters[2], filters[2])
        self.maxpool3 = nn.MaxPool2d(kernel_size=2) if is_pool else ConvBNReLU(
            filters[2], filters[2], norm, stride=2)
        self.conv4 = UnetConv2D(filters[2], filters[3], norm)
        self.gcn4 = GCN(filters[3], filters[3])
        self.maxpool4 = nn.MaxPool2d(kernel_size=2) if is_pool else ConvBNReLU(
            filters[3], filters[3], norm, stride=2)
        self.center = UnetConv2D(filters[3], filters[4], norm)
        self.up_concat4 = UnetUpECRE(filters[4], filters[3], norm)
        self.up_concat3 = UnetUpECRE(filters[3], filters[2], norm)
        self.up_concat2 = UnetUpECRE(filters[2], filters[1], norm)
        self.up_concat1 = UnetUpECRE(filters[1], filters[0], norm)
        self.final = nn.Conv2d(filters[0], n_classes, 1)
        self.ecre4 = ConvBNReLU(filters[4], 1, norm, stride=2)
        self.ecre3 = ConvBNReLU(filters[3], 1, norm, stride=2)
        self.ecre2 = ConvBNReLU(filters[2], 1, norm, stride=2)
        self.ecre1 = ConvBNReLU(filters[1], 1, norm, stride=2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.apply(weights_init_kaiming)
            elif isinstance(m, nn.BatchNorm2d):
                m.apply(weights_init_kaiming)

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        conv1 = self.gcn1(conv1)
        maxpool1 = self.maxpool1(conv1)
        conv2 = self.conv2(maxpool1)
        conv2 = self.gcn2(conv2)
        maxpool2 = self.maxpool2(conv2)
        conv3 = self.conv3(maxpool2)
        conv3 = self.gcn3(conv3)
        maxpool3 = self.maxpool3(conv3)
        conv4 = self.conv4(maxpool3)
        conv4 = self.gcn4(conv4)
        maxpool4 = self.maxpool4(conv4)
        center = self.center(maxpool4)
        up4, ecre4 = self.up_concat4(conv4, center)
        up3, ecre3 = self.up_concat3(conv3, up4)
        up2, ecre2 = self.up_concat2(conv2, up3)
        up1, ecre1 = self.up_concat1(conv1, up2)
        final = self.final(up1)
        return final


class UnetExFuse(nn.Module):

    def __init__(self, num_classes=1, pretrained=False, feature_scale=4,
        is_deconv=True, norm=nn.InstanceNorm2d, is_pool=True, **kwargs):
        super(UnetExFuse, self).__init__()
        filters = [64, 128, 256, 512, 1024]
        filters = [(x // feature_scale) for x in filters]
        self.conv1 = UnetConv2D(1, filters[0], norm)
        self.gcn1 = GCN(filters[0], filters[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=2) if is_pool else ConvBNReLU(
            filters[0], filters[0], norm, stride=2)
        self.conv2 = UnetConv2D(filters[0], filters[1], norm)
        self.gcn2 = GCN(filters[1], filters[1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=2) if is_pool else ConvBNReLU(
            filters[1], filters[1], norm, stride=2)
        self.conv3 = UnetConv2D(filters[1], filters[2], norm)
        self.gcn3 = GCN(filters[2], filters[2])
        self.maxpool3 = nn.MaxPool2d(kernel_size=2) if is_pool else ConvBNReLU(
            filters[2], filters[2], norm, stride=2)
        self.conv4 = UnetConv2D(filters[2], filters[3], norm)
        self.gcn4 = GCN(filters[3], filters[3])
        self.maxpool4 = nn.MaxPool2d(kernel_size=2) if is_pool else ConvBNReLU(
            filters[3], filters[3], norm, stride=2)
        self.center = UnetConv2D(filters[3], filters[4], norm)
        self.up_concat4 = UnetUpConv2D(filters[4], filters[3], norm, is_deconv)
        self.level4 = UnetExFuseLevel(filters[3], filters[2])
        self.level3 = UnetExFuseLevel(filters[2], filters[1])
        self.level2 = UnetExFuseLevel(filters[1], filters[0])
        self.final = nn.Sequential(DAP(filters[0]), nn.Conv2d(filters[0], 1, 1)
            )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.apply(weights_init_kaiming)
            elif isinstance(m, nn.BatchNorm2d):
                m.apply(weights_init_kaiming)

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        conv1 = self.gcn1(conv1)
        maxpool1 = self.maxpool1(conv1)
        conv2 = self.conv2(maxpool1)
        conv2 = self.gcn2(conv2)
        maxpool2 = self.maxpool2(conv2)
        conv3 = self.conv3(maxpool2)
        conv3 = self.gcn3(conv3)
        maxpool3 = self.maxpool3(conv3)
        conv4 = self.conv4(maxpool3)
        conv4 = self.gcn4(conv4)
        maxpool4 = self.maxpool4(conv4)
        center = self.center(maxpool4)
        up4 = self.up_concat4(conv4, center)
        l4 = self.level4(conv4, center, up4)
        l3 = self.level3(conv3, conv4, l4)
        l2 = self.level2(conv2, conv3, l3)
        final = self.final(l2)
        return final


class ConvBNReLU(nn.Module):

    def __init__(self, in_size, out_size, norm, kernel_size=3, stride=1,
        padding=1, act=nn.ReLU):
        super(ConvBNReLU, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size,
            stride, padding), norm(out_size), act(inplace=True))
        for m in self.children():
            m.apply(weights_init_kaiming)

    def forward(self, inputs):
        return self.conv1(inputs)


class UnetConv2D(nn.Module):

    def __init__(self, in_size, out_size, norm, kernel_size=3, stride=1,
        padding=1, act=nn.ReLU):
        super(UnetConv2D, self).__init__()
        self.conv1 = ConvBNReLU(in_size, out_size, norm, kernel_size,
            stride, padding, act)
        self.conv2 = ConvBNReLU(out_size, out_size, norm, kernel_size, 1,
            padding, act)

    def forward(self, inputs):
        x = self.conv1(inputs)
        return self.conv2(x)


class UnetUpConv2D(nn.Module):

    def __init__(self, in_size, out_size, norm, is_deconv=True, act=nn.ReLU):
        super(UnetUpConv2D, self).__init__()
        self.conv = UnetConv2D(in_size, out_size, norm, act=act)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4,
                stride=2, padding=1)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        for m in self.children():
            if m.__class__.__name__.find('UnetConv2D') != -1:
                continue
            m.apply(weights_init_kaiming)

    def forward(self, input1, input2):
        output2 = self.up(input2)
        offset = output2.size()[2] - input1.size()[2]
        padding = [offset // 2] * 4
        output1 = F.pad(input1, padding)
        output = torch.cat([output1, output2], 1)
        return self.conv(output)


class FCDenseNet(Module):
    """
    The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation
    https://arxiv.org/abs/1611.09326

    In this paper, we extend DenseNets to deal with the problem of semantic segmentation. We achieve state-of-the-art
    results on urban scene benchmark datasets such as CamVid and Gatech, without any further post-processing module nor
    pretraining. Moreover, due to smart construction of the model, our approach has much less parameters than currently
    published best entries for these datasets.
    """

    def __init__(self, in_channels: int=3, out_channels: int=1000,
        initial_num_features: int=48, dropout: float=0.2,
        down_dense_growth_rates: Union[int, Sequence[int]]=16,
        down_dense_bottleneck_ratios: Union[Optional[int], Sequence[
        Optional[int]]]=None, down_dense_num_layers: Union[int, Sequence[
        int]]=(4, 5, 7, 10, 12), down_transition_compression_factors: Union
        [float, Sequence[float]]=1.0, middle_dense_growth_rate: int=16,
        middle_dense_bottleneck: Optional[int]=None,
        middle_dense_num_layers: int=15, up_dense_growth_rates: Union[int,
        Sequence[int]]=16, up_dense_bottleneck_ratios: Union[Optional[int],
        Sequence[Optional[int]]]=None, up_dense_num_layers: Union[int,
        Sequence[int]]=(12, 10, 7, 5, 4)):
        super(FCDenseNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if type(down_dense_growth_rates) == int:
            down_dense_growth_rates = (down_dense_growth_rates,) * 5
        if down_dense_bottleneck_ratios is None or type(
            down_dense_bottleneck_ratios) == int:
            down_dense_bottleneck_ratios = (down_dense_bottleneck_ratios,) * 5
        if type(down_dense_num_layers) == int:
            down_dense_num_layers = (down_dense_num_layers,) * 5
        if type(down_transition_compression_factors) == float:
            down_transition_compression_factors = (
                down_transition_compression_factors,) * 5
        if type(up_dense_growth_rates) == int:
            up_dense_growth_rates = (up_dense_growth_rates,) * 5
        if up_dense_bottleneck_ratios is None or type(
            up_dense_bottleneck_ratios) == int:
            up_dense_bottleneck_ratios = (up_dense_bottleneck_ratios,) * 5
        if type(up_dense_num_layers) == int:
            up_dense_num_layers = (up_dense_num_layers,) * 5
        self.features = Conv2d(in_channels, initial_num_features,
            kernel_size=3, padding=1, bias=False)
        current_channels = self.features.out_channels
        down_dense_params = [{'concat_input': True, 'growth_rate': gr,
            'num_layers': nl, 'dense_layer_params': {'dropout': dropout,
            'bottleneck_ratio': br}} for gr, nl, br in zip(
            down_dense_growth_rates, down_dense_num_layers,
            down_dense_bottleneck_ratios)]
        down_transition_params = [{'dropout': dropout, 'compression': c} for
            c in down_transition_compression_factors]
        skip_connections_channels = []
        self.down_dense = Module()
        self.down_trans = Module()
        down_pairs_params = zip(down_dense_params, down_transition_params)
        for i, (dense_params, transition_params) in enumerate(down_pairs_params
            ):
            block = DenseBlock(current_channels, **dense_params)
            current_channels = block.out_channels
            self.down_dense.add_module(f'block_{i}', block)
            skip_connections_channels.append(block.out_channels)
            transition = TransitionDown(current_channels, **transition_params)
            current_channels = transition.out_channels
            self.down_trans.add_module(f'trans_{i}', transition)
        self.middle = DenseBlock(current_channels, middle_dense_growth_rate,
            middle_dense_num_layers, concat_input=True, dense_layer_params=
            {'dropout': dropout, 'bottleneck_ratio': middle_dense_bottleneck})
        current_channels = self.middle.out_channels
        up_transition_params = [{'skip_channels': sc} for sc in reversed(
            skip_connections_channels)]
        up_dense_params = [{'concat_input': False, 'growth_rate': gr,
            'num_layers': nl, 'dense_layer_params': {'dropout': dropout,
            'bottleneck_ratio': br}} for gr, nl, br in zip(
            up_dense_growth_rates, up_dense_num_layers,
            up_dense_bottleneck_ratios)]
        self.up_dense = Module()
        self.up_trans = Module()
        up_pairs_params = zip(up_transition_params, up_dense_params)
        for i, (transition_params_up, dense_params_up) in enumerate(
            up_pairs_params):
            transition = TransitionUp(current_channels, **transition_params_up)
            current_channels = transition.out_channels
            self.up_trans.add_module(f'trans_{i}', transition)
            block = DenseBlock(current_channels, **dense_params_up)
            current_channels = block.out_channels
            self.up_dense.add_module(f'block_{i}', block)
        self.final = Conv2d(current_channels, out_channels, kernel_size=1,
            bias=False)
        for module in self.modules():
            if isinstance(module, Conv2d):
                init.kaiming_normal_(module.weight)
            elif isinstance(module, BatchNorm2d):
                module.reset_parameters()
            elif isinstance(module, Linear):
                init.xavier_uniform(module.weight)
                init.constant(module.bias, 0)

    def forward(self, x):
        res = self.features(x)
        skip_tensors = []
        for dense, trans in zip(self.down_dense.children(), self.down_trans
            .children()):
            res = dense(res)
            skip_tensors.append(res)
            res = trans(res)
        res = self.middle(res)
        for skip, trans, dense in zip(reversed(skip_tensors), self.up_trans
            .children(), self.up_dense.children()):
            res = trans(res, skip)
            res = dense(res)
        res = self.final(res)
        return res

    def predict(self, x):
        logits = self(x)
        return F.softmax(logits)


class Flatten(Module):

    def forward(self, x):
        return x.view(x.size(0), -1)


class Resnet(nn.Module):

    def __init__(self, orig_resnet, **kwargs):
        super(Resnet, self).__init__()
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def forward(self, x, return_feature_maps=False):
        conv_out = []
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        conv_out.append(x)
        x = self.layer2(x)
        conv_out.append(x)
        x = self.layer3(x)
        conv_out.append(x)
        x = self.layer4(x)
        conv_out.append(x)
        if return_feature_maps:
            return conv_out
        return x


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(conv3x3(in_planes, out_planes, stride), nn.
        BatchNorm2d(out_planes), nn.ReLU(inplace=True))


class C1BilinearDeepSup(nn.Module):

    def __init__(self, num_class=150, fc_dim=2048, use_softmax=False):
        super(C1BilinearDeepSup, self).__init__()
        self.use_softmax = use_softmax
        self.cbr = conv3x3_bn_relu(fc_dim, fc_dim // 4, 1)
        self.cbr_deepsup = conv3x3_bn_relu(fc_dim // 2, fc_dim // 4, 1)
        self.conv_last = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)
        self.conv_last_deepsup = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]
        x = self.cbr(conv5)
        x = self.conv_last(x)
        if self.use_softmax:
            x = nn.functional.upsample(x, size=segSize, mode='bilinear')
            x = nn.functional.softmax(x, dim=1)
            return x
        conv4 = conv_out[-2]
        _ = self.cbr_deepsup(conv4)
        _ = self.conv_last_deepsup(_)
        x = nn.functional.log_softmax(x, dim=1)
        _ = nn.functional.log_softmax(_, dim=1)
        return x, _


class C1Bilinear(nn.Module):

    def __init__(self, num_class=150, fc_dim=2048, use_softmax=False):
        super(C1Bilinear, self).__init__()
        self.use_softmax = use_softmax
        self.cbr = conv3x3_bn_relu(fc_dim, fc_dim // 4, 1)
        self.conv_last = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]
        x = self.cbr(conv5)
        x = self.conv_last(x)
        if self.use_softmax:
            x = nn.functional.upsample(x, size=segSize, mode='bilinear')
            x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.log_softmax(x, dim=1)
        return x


class PPMBilinear(nn.Module):

    def __init__(self, num_class=150, fc_dim=4096, use_softmax=False,
        pool_scales=(1, 2, 3, 6)):
        super(PPMBilinear, self).__init__()
        self.use_softmax = use_softmax
        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(nn.AdaptiveAvgPool2d(scale), nn.
                Conv2d(fc_dim, 512, kernel_size=1, bias=False), nn.
                BatchNorm2d(512), nn.ReLU(inplace=True)))
        self.ppm = nn.ModuleList(self.ppm)
        self.conv_last = nn.Sequential(nn.Conv2d(fc_dim + len(pool_scales) *
            512, 512, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d
            (512), nn.ReLU(inplace=True), nn.Dropout2d(0.1), nn.Conv2d(512,
            num_class, kernel_size=1))

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]
        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.upsample(pool_scale(conv5), (
                input_size[2], input_size[3]), mode='bilinear'))
        ppm_out = torch.cat(ppm_out, 1)
        x = self.conv_last(ppm_out)
        if self.use_softmax:
            x = nn.functional.upsample(x, size=segSize, mode='bilinear')
            x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.log_softmax(x, dim=1)
        return x


class PPMBilinearDeepsup(nn.Module):

    def __init__(self, num_class=150, fc_dim=4096, use_softmax=False,
        pool_scales=(1, 2, 3, 6)):
        super(PPMBilinearDeepsup, self).__init__()
        self.use_softmax = use_softmax
        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(nn.AdaptiveAvgPool2d(scale), nn.
                Conv2d(fc_dim, 512, kernel_size=1, bias=False), nn.
                BatchNorm2d(512), nn.ReLU(inplace=True)))
        self.ppm = nn.ModuleList(self.ppm)
        self.cbr_deepsup = conv3x3_bn_relu(fc_dim // 2, fc_dim // 4, 1)
        self.conv_last = nn.Sequential(nn.Conv2d(fc_dim + len(pool_scales) *
            512, 512, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d
            (512), nn.ReLU(inplace=True), nn.Dropout2d(0.1), nn.Conv2d(512,
            num_class, kernel_size=1))
        self.conv_last_deepsup = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)
        self.dropout_deepsup = nn.Dropout2d(0.1)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]
        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.upsample(pool_scale(conv5), (
                input_size[2], input_size[3]), mode='bilinear'))
        ppm_out = torch.cat(ppm_out, 1)
        x = self.conv_last(ppm_out)
        if self.use_softmax:
            x = nn.functional.upsample(x, size=segSize, mode='bilinear')
            x = nn.functional.softmax(x, dim=1)
            return x
        conv4 = conv_out[-2]
        _ = self.cbr_deepsup(conv4)
        _ = self.dropout_deepsup(_)
        _ = self.conv_last_deepsup(_)
        x = nn.functional.log_softmax(x, dim=1)
        _ = nn.functional.log_softmax(_, dim=1)
        return x, _


class UPerNet(nn.Module):

    def __init__(self, num_class=150, fc_dim=4096, use_softmax=False,
        pool_scales=(1, 2, 3, 6), fpn_inplanes=(256, 512, 1024, 2048),
        fpn_dim=256):
        super(UPerNet, self).__init__()
        self.use_softmax = use_softmax
        self.ppm_pooling = []
        self.ppm_conv = []
        for scale in pool_scales:
            self.ppm_pooling.append(nn.AdaptiveAvgPool2d(scale))
            self.ppm_conv.append(nn.Sequential(nn.Conv2d(fc_dim, 512,
                kernel_size=1, bias=False), nn.BatchNorm2d(512), nn.ReLU(
                inplace=True)))
        self.ppm_pooling = nn.ModuleList(self.ppm_pooling)
        self.ppm_conv = nn.ModuleList(self.ppm_conv)
        self.ppm_last_conv = conv3x3_bn_relu(fc_dim + len(pool_scales) * 
            512, fpn_dim, 1)
        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:-1]:
            self.fpn_in.append(nn.Sequential(nn.Conv2d(fpn_inplane, fpn_dim,
                kernel_size=1, bias=False), nn.BatchNorm2d(fpn_dim), nn.
                ReLU(inplace=True)))
        self.fpn_in = nn.ModuleList(self.fpn_in)
        self.fpn_out = []
        for i in range(len(fpn_inplanes) - 1):
            self.fpn_out.append(nn.Sequential(conv3x3_bn_relu(fpn_dim,
                fpn_dim, 1)))
        self.fpn_out = nn.ModuleList(self.fpn_out)
        self.conv_last = nn.Sequential(conv3x3_bn_relu(len(fpn_inplanes) *
            fpn_dim, fpn_dim, 1), nn.Conv2d(fpn_dim, num_class, kernel_size=1))

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]
        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale, pool_conv in zip(self.ppm_pooling, self.ppm_conv):
            ppm_out.append(pool_conv(nn.functional.upsample(pool_scale(
                conv5), (input_size[2], input_size[3]), mode='bilinear')))
        ppm_out = torch.cat(ppm_out, 1)
        f = self.ppm_last_conv(ppm_out)
        fpn_feature_list = [f]
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x)
            f = nn.functional.upsample(f, size=conv_x.size()[2:], mode=
                'bilinear')
            f = conv_x + f
            fpn_feature_list.append(self.fpn_out[i](f))
        fpn_feature_list.reverse()
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]
        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(nn.functional.upsample(fpn_feature_list[i],
                output_size, mode='bilinear'))
        fusion_out = torch.cat(fusion_list, 1)
        x = self.conv_last(fusion_out)
        if self.use_softmax:
            x = nn.functional.upsample(x, size=segSize, mode='bilinear')
            x = nn.functional.softmax(x, dim=1)
            return x
        x = nn.functional.log_softmax(x, dim=1)
        return x


class _ConvBatchNormReluBlock(nn.Sequential):

    def __init__(self, inplanes, outplanes, kernel_size, stride, padding,
        dilation, relu=True):
        super(_ConvBatchNormReluBlock, self).__init__()
        self.add_module('cov', nn.Conv2d(in_channels=inplanes, out_channels
            =outplanes, kernel_size=kernel_size, stride=stride, padding=
            padding, dilation=dilation, bias=False))
        self.add_module('bn', nn.BatchNorm2d(num_features=outplanes,
            momentum=0.999, affine=True))
        if relu:
            self.add_module('relu', nn.ReLU())

    def forward(self, x):
        return super(_ConvBatchNormReluBlock, self).forward(x)


class _ResidualBlockMulGrid(nn.Sequential):
    """
		Residual Block with multi-grid , note: best model-> (1, 2, 1)
	"""

    def __init__(self, layers, inplanes, midplanes, outplanes, stride,
        dilation, mulgrid=[1, 2, 1]):
        super(_ResidualBlockMulGrid, self).__init__()
        self.add_module('block1', _Bottleneck(inplanes, midplanes,
            outplanes, stride, dilation * mulgrid[0], True))
        self.add_module('block2', _Bottleneck(outplanes, midplanes,
            outplanes, stride, dilation * mulgrid[1], False))
        self.add_module('block3', _Bottleneck(outplanes, midplanes,
            outplanes, stride, dilation * mulgrid[2], False))

    def forward(self, x):
        return super(_ResidualBlockMulGrid, self).forward(x)


class _Bottleneck(nn.Sequential):

    def __init__(self, inplanes, midplanes, outplanes, stride, dilation,
        downsample):
        super(_Bottleneck, self).__init__()
        self.reduce = _ConvBatchNormReluBlock(inplanes, midplanes, 1,
            stride, 0, 1)
        self.conv3x3 = _ConvBatchNormReluBlock(midplanes, midplanes, 3, 1,
            dilation, dilation)
        self.increase = _ConvBatchNormReluBlock(midplanes, outplanes, 1, 1,
            0, 1, relu=False)
        self.downsample = downsample
        if self.downsample:
            self.proj = _ConvBatchNormReluBlock(inplanes, outplanes, 1,
                stride, 0, 1, relu=False)

    def forward(self, x):
        h = self.reduce(x)
        h = self.conv3x3(h)
        h = self.increase(h)
        if self.downsample:
            h += self.proj(x)
        else:
            h += x
        return F.relu(h)


class _BoundaryRefineModule(nn.Module):

    def __init__(self, dim):
        super(_BoundaryRefineModule, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        out = x + residual
        return out


class _GlobalConvModule(nn.Module):

    def __init__(self, in_dim, out_dim, kernel_size):
        super(_GlobalConvModule, self).__init__()
        pad0 = int((kernel_size[0] - 1) / 2)
        pad1 = int((kernel_size[1] - 1) / 2)
        self.conv_l1 = nn.Conv2d(in_dim, out_dim, kernel_size=(kernel_size[
            0], 1), padding=(pad0, 0))
        self.conv_l2 = nn.Conv2d(out_dim, out_dim, kernel_size=(1,
            kernel_size[1]), padding=(0, pad1))
        self.conv_r1 = nn.Conv2d(in_dim, out_dim, kernel_size=(1,
            kernel_size[1]), padding=(0, pad1))
        self.conv_r2 = nn.Conv2d(out_dim, out_dim, kernel_size=(kernel_size
            [0], 1), padding=(pad0, 0))

    def forward(self, x):
        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)
        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)
        x = x_l + x_r
        return x


class ModelBuilder:

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(0.0001)
        elif classname.find('Linear') != -1:
            m.weight.data.normal_(0.0, 0.0001)

    def build_encoder(self, arch='resnet50_dilated8', fc_dim=512, weights=
        '', **kwargs):
        pretrained = True if len(weights) == 0 else False
        if arch == 'resnet34':
            raise NotImplementedError
            orig_resnet = resnet.__dict__['resnet34'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet34_dilated8':
            raise NotImplementedError
            orig_resnet = resnet.__dict__['resnet34'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)
        elif arch == 'resnet34_dilated16':
            raise NotImplementedError
            orig_resnet = resnet.__dict__['resnet34'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=16)
        elif arch == 'resnet50':
            orig_resnet = resnet.resnet50(**kwargs)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet50_dilated8':
            orig_resnet = resnet.resnet50(**kwargs)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)
        elif arch == 'resnet50_dilated16':
            orig_resnet = resnet.resnet50(**kwargs)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=16)
        elif arch == 'resnet101':
            orig_resnet = resnet.resnet101(**kwargs)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet101_dilated8':
            orig_resnet = resnet.resnet101(**kwargs)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)
        elif arch == 'resnet101_dilated16':
            orig_resnet = resnet.resnet101(**kwargs)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=16)
        elif arch == 'resnext101':
            orig_resnext = resnext.resnext101(**kwargs)
            net_encoder = Resnet(orig_resnext)
        else:
            raise Exception('Architecture undefined!')
        if len(weights) > 0:
            print('Loading weights for net_encoder')
            net_encoder.load_state_dict(torch.load(weights, map_location=lambda
                storage, loc: storage), strict=False)
        return net_encoder

    def build_decoder(self, arch='ppm_bilinear_deepsup', fc_dim=512,
        num_classes=150, weights='', use_softmax=False):
        if arch == 'c1_bilinear_deepsup':
            net_decoder = C1BilinearDeepSup(num_class=num_classes, fc_dim=
                fc_dim, use_softmax=use_softmax)
        elif arch == 'c1_bilinear':
            net_decoder = C1Bilinear(num_class=num_classes, fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'ppm_bilinear':
            net_decoder = PPMBilinear(num_class=num_classes, fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'ppm_bilinear_deepsup':
            net_decoder = PPMBilinearDeepsup(num_class=num_classes, fc_dim=
                fc_dim, use_softmax=use_softmax)
        elif arch == 'upernet_lite':
            net_decoder = UPerNet(num_class=num_classes, fc_dim=fc_dim,
                use_softmax=use_softmax, fpn_dim=256)
        elif arch == 'upernet':
            net_decoder = UPerNet(num_class=num_classes, fc_dim=fc_dim,
                use_softmax=use_softmax, fpn_dim=512)
        else:
            raise Exception('Architecture undefined!')
        net_decoder.apply(self.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_decoder')
            net_decoder.load_state_dict(torch.load(weights, map_location=lambda
                storage, loc: storage), strict=False)
        return net_decoder


class GCN(nn.Module):

    def __init__(self, num_classes, kernel_size=7):
        super(GCN, self).__init__()
        self.resnet_features = ModelBuilder().build_encoder('resnet101')
        self.layer0 = nn.Sequential(self.resnet_features.conv1, self.
            resnet_features.bn1, self.resnet_features.relu1, self.
            resnet_features.conv3, self.resnet_features.bn3, self.
            resnet_features.relu3)
        self.layer1 = nn.Sequential(self.resnet_features.maxpool, self.
            resnet_features.layer1)
        self.layer2 = self.resnet_features.layer2
        self.layer3 = self.resnet_features.layer3
        self.layer4 = self.resnet_features.layer4
        self.gcm1 = _GlobalConvModule(2048, num_classes, (kernel_size,
            kernel_size))
        self.gcm2 = _GlobalConvModule(1024, num_classes, (kernel_size,
            kernel_size))
        self.gcm3 = _GlobalConvModule(512, num_classes, (kernel_size,
            kernel_size))
        self.gcm4 = _GlobalConvModule(256, num_classes, (kernel_size,
            kernel_size))
        self.brm1 = _BoundaryRefineModule(num_classes)
        self.brm2 = _BoundaryRefineModule(num_classes)
        self.brm3 = _BoundaryRefineModule(num_classes)
        self.brm4 = _BoundaryRefineModule(num_classes)
        self.brm5 = _BoundaryRefineModule(num_classes)
        self.brm6 = _BoundaryRefineModule(num_classes)
        self.brm7 = _BoundaryRefineModule(num_classes)
        self.brm8 = _BoundaryRefineModule(num_classes)
        self.brm9 = _BoundaryRefineModule(num_classes)
        self.deconv1 = nn.ConvTranspose2d(num_classes, num_classes,
            kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv2 = nn.ConvTranspose2d(num_classes, num_classes,
            kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv3 = nn.ConvTranspose2d(num_classes, num_classes,
            kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv4 = nn.ConvTranspose2d(num_classes, num_classes,
            kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv5 = nn.ConvTranspose2d(num_classes, num_classes,
            kernel_size=4, stride=2, padding=1, bias=False)

    def forward(self, x):
        f0 = self.layer0(x)
        f1 = self.layer1(f0)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        gcfm1 = self.brm1(self.gcm1(f4))
        gcfm2 = self.brm2(self.gcm2(f3))
        gcfm3 = self.brm3(self.gcm3(f2))
        gcfm4 = self.brm4(self.gcm4(f1))
        fs1 = self.brm5(self.deconv1(gcfm1) + gcfm2)
        fs2 = self.brm6(self.deconv2(fs1) + gcfm3)
        fs3 = self.brm7(self.deconv3(fs2) + gcfm4)
        fs4 = self.brm8(self.deconv4(fs3))
        out = self.brm9(self.deconv5(fs4))
        return out

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


class SEB(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(SEB, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3,
            stride=1, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        x1, x2 = x
        return x1 * self.upsample(self.conv(x2))


class GCNFuse(nn.Module):
    """
    :param kernel_size: (int) Must be an ODD number!!
    """

    def __init__(self, num_classes=1, backbone='resnet101', kernel_size=7,
        dap_k=3, **kwargs):
        super(GCNFuse, self).__init__()
        self.num_classes = num_classes
        self.resnet_features = ModelBuilder().build_encoder(arch=backbone,
            **kwargs)
        self.layer0 = nn.Sequential(self.resnet_features.conv1, self.
            resnet_features.bn1, self.resnet_features.relu1, self.
            resnet_features.conv3, self.resnet_features.bn3, self.
            resnet_features.relu3)
        self.layer1 = nn.Sequential(self.resnet_features.maxpool, self.
            resnet_features.layer1)
        self.layer2 = self.resnet_features.layer2
        self.layer3 = self.resnet_features.layer3
        self.layer4 = self.resnet_features.layer4
        self.gcm1 = _GlobalConvModule(2048, num_classes * 4, (kernel_size,
            kernel_size))
        self.gcm2 = _GlobalConvModule(1024, num_classes, (kernel_size,
            kernel_size))
        self.gcm3 = _GlobalConvModule(512, num_classes * dap_k ** 2, (
            kernel_size, kernel_size))
        self.gcm4 = _GlobalConvModule(256, num_classes * dap_k ** 2, (
            kernel_size, kernel_size))
        self.deconv1 = nn.ConvTranspose2d(num_classes, num_classes * dap_k **
            2, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv2 = nn.ConvTranspose2d(num_classes, num_classes * dap_k **
            2, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv3 = nn.ConvTranspose2d(num_classes * dap_k ** 2, 
            num_classes * dap_k ** 2, kernel_size=4, stride=2, padding=1,
            bias=False)
        self.deconv4 = nn.ConvTranspose2d(num_classes * dap_k ** 2, 
            num_classes * dap_k ** 2, kernel_size=4, stride=2, padding=1,
            bias=False)
        self.deconv5 = nn.ConvTranspose2d(num_classes * dap_k ** 2, 
            num_classes * dap_k ** 2, kernel_size=4, stride=2, padding=1,
            bias=False)
        self.ecre = nn.PixelShuffle(2)
        self.seb1 = SEB(2048, 1024)
        self.seb2 = SEB(3072, 512)
        self.seb3 = SEB(3584, 256)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.DAP = nn.Sequential(nn.PixelShuffle(dap_k), nn.AvgPool2d((
            dap_k, dap_k)))

    def forward(self, x):
        f0 = self.layer0(x)
        f1 = self.layer1(f0)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        x = self.gcm1(f4)
        out1 = self.ecre(x)
        seb1 = self.seb1([f3, f4])
        gcn1 = self.gcm2(seb1)
        seb2 = self.seb2([f2, torch.cat([f3, self.upsample2(f4)], dim=1)])
        gcn2 = self.gcm3(seb2)
        seb3 = self.seb3([f1, torch.cat([f2, self.upsample2(f3), self.
            upsample4(f4)], dim=1)])
        gcn3 = self.gcm4(seb3)
        y = self.deconv2(gcn1 + out1)
        y = self.deconv3(gcn2 + y)
        y = self.deconv4(gcn3 + y)
        y = self.deconv5(y)
        y = self.DAP(y)
        return y

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


class BasicBlock(nn.Module):
    expansion = 1

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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
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
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, **kwargs):
        self.inplanes = 128
        super(ResNet, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class GroupBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, groups=1, downsample=None):
        super(GroupBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 2)
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
        out += residual
        out = self.relu(out)
        return out


class ResNeXt(nn.Module):

    def __init__(self, block, layers, groups=32, num_classes=1000, **kwargs):
        self.inplanes = 128
        super(ResNeXt, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 128, layers[0], groups=groups)
        self.layer2 = self._make_layer(block, 256, layers[1], stride=2,
            groups=groups)
        self.layer3 = self._make_layer(block, 512, layers[2], stride=2,
            groups=groups)
        self.layer4 = self._make_layer(block, 1024, layers[3], stride=2,
            groups=groups)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(1024 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1
                    ] * m.out_channels // m.groups
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, groups=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, groups, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=groups))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Swish(nn.Module):

    def forward(self, x):
        return x * torch.sigmoid(x)


class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.shape[0], -1)


class SEModule(nn.Module):

    def __init__(self, ch, squeeze_ch):
        super().__init__()
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(ch,
            squeeze_ch, 1, 1, 0, bias=True), Swish(), nn.Conv2d(squeeze_ch,
            ch, 1, 1, 0, bias=True))

    def forward(self, x):
        return x * torch.sigmoid(self.se(x))


def _split_channels(total_filters, num_groups):
    """
    https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mixnet/custom_layers.py#L33
    """
    split = [(total_filters // num_groups) for _ in range(num_groups)]
    split[0] += total_filters - sum(split)
    return split


class MDConv(nn.Module):

    def __init__(self, in_channels, kernel_sizes, stride, dilatied=False,
        bias=False):
        super().__init__()
        if not isinstance(kernel_sizes, list):
            kernel_sizes = [kernel_sizes]
        self.in_channels = _split_channels(in_channels, len(kernel_sizes))
        self.convs = nn.ModuleList()
        for ch, k in zip(self.in_channels, kernel_sizes):
            dilation = 1
            if stride[0] == 1 and dilatied:
                dilation, stride = (k - 1) // 2, 3
                None
            pad = (stride[0] - 1 + dilation * (k - 1)) // 2
            conv = nn.Conv2d(ch, ch, k, stride, pad, dilation, groups=ch,
                bias=bias)
            self.convs.append(conv)

    def forward(self, x):
        xs = torch.split(x, self.in_channels, 1)
        return torch.cat([conv(x) for conv, x in zip(self.convs, xs)], 1)


class MixBlock(nn.Module):

    def __init__(self, dw_ksize, expand_ksize, project_ksize, in_channels,
        out_channels, expand_ratio, id_skip, strides, se_ratio, swish, dilated
        ):
        super().__init__()
        self.id_skip = id_skip and all(s == 1 for s in strides
            ) and in_channels == out_channels
        act_fn = lambda : Swish() if swish else nn.ReLU(True)
        layers = []
        expaned_ch = in_channels * expand_ratio
        if expand_ratio != 1:
            expand = nn.Sequential(nn.Conv2d(in_channels, expaned_ch,
                expand_ksize, bias=False), nn.BatchNorm2d(expaned_ch), act_fn()
                )
            layers.append(expand)
        depthwise = nn.Sequential(MDConv(expaned_ch, dw_ksize, strides,
            bias=False), nn.BatchNorm2d(expaned_ch), act_fn())
        layers.append(depthwise)
        if se_ratio > 0:
            se = SEModule(expaned_ch, int(expaned_ch * se_ratio))
            layers.append(se)
        project = nn.Sequential(nn.Conv2d(expaned_ch, out_channels,
            project_ksize, bias=False), nn.BatchNorm2d(out_channels))
        layers.append(project)
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        out = self.layers(x)
        if self.id_skip:
            out = out + x
        return out


class MixModule(nn.Module):

    def __init__(self, dw_ksize, expand_ksize, project_ksize, num_repeat,
        in_channels, out_channels, expand_ratio, id_skip, strides, se_ratio,
        swish, dilated):
        super().__init__()
        layers = [MixBlock(dw_ksize, expand_ksize, project_ksize,
            in_channels, out_channels, expand_ratio, id_skip, strides,
            se_ratio, swish, dilated)]
        for _ in range(num_repeat - 1):
            layers.append(MixBlock(dw_ksize, expand_ksize, project_ksize,
                in_channels, out_channels, expand_ratio, id_skip, [1, 1],
                se_ratio, swish, dilated))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class MixNet(nn.Module):

    def __init__(self, stem, blocks_args, head, dropout_rate, num_classes=
        1000, **kwargs):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, stem, 3, 2, 1, bias=False),
            nn.BatchNorm2d(stem), nn.ReLU(True))
        self.blocks = nn.Sequential(*[MixModule(*args) for args in blocks_args]
            )
        self.classifier = nn.Sequential(nn.Conv2d(blocks_args[-1].
            out_channels, head, 1, bias=False), nn.BatchNorm2d(head), nn.
            ReLU(True), nn.AdaptiveAvgPool2d(1), Flatten(), nn.Dropout(
            dropout_rate), nn.Linear(head, num_classes))
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                    nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in',
                    nonlinearity='linear')

    def forward(self, x):
        stem = self.stem(x)
        feature = self.blocks(stem)
        out = self.classifier(feature)
        return out


class MSC(nn.Module):
    """Multi-scale inputs"""

    def __init__(self, scale, pyramids=[0.5, 0.75]):
        super(MSC, self).__init__()
        self.scale = scale
        self.pyramids = pyramids

    def forward(self, x):
        logits = self.scale(x)
        interp = lambda l: F.interpolate(l, size=logits.shape[2:], mode=
            'bilinear', align_corners=True)
        logits_pyramid = []
        for p in self.pyramids:
            size = [int(s * p) for s in x.shape[2:]]
            h = F.interpolate(x, size=size, mode='bilinear', align_corners=True
                )
            logits_pyramid.append(self.scale(h))
        logits_all = [logits] + [interp(l) for l in logits_pyramid]
        logits_max = torch.max(torch.stack(logits_all), dim=0)[0]
        if self.training:
            return [logits] + logits_pyramid + [logits_max]
        else:
            return logits_max


class _PSAHead(nn.Module):

    def __init__(self, nclass, input_size, norm_layer=nn.BatchNorm2d, **kwargs
        ):
        super(_PSAHead, self).__init__()
        psa_out_channels = (input_size // 8) ** 2
        self.psa = _PointwiseSpatialAttention(2048, psa_out_channels,
            norm_layer)
        self.conv_post = _ConvBNReLU(1024, 2048, 1, norm_layer=norm_layer)
        self.project = nn.Sequential(_ConvBNReLU(4096, 512, 3, padding=1,
            norm_layer=norm_layer), nn.Dropout2d(0.1, False), nn.Conv2d(512,
            nclass, 1))

    def forward(self, x):
        global_feature = self.psa(x)
        out = self.conv_post(global_feature)
        out = torch.cat([x, out], dim=1)
        out = self.project(out)
        return out


class _PointwiseSpatialAttention(nn.Module):

    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d,
        **kwargs):
        super(_PointwiseSpatialAttention, self).__init__()
        reduced_channels = 512
        self.collect_attention = _AttentionGeneration(in_channels,
            reduced_channels, out_channels, norm_layer)
        self.distribute_attention = _AttentionGeneration(in_channels,
            reduced_channels, out_channels, norm_layer)

    def forward(self, x):
        collect_fm = self.collect_attention(x)
        distribute_fm = self.distribute_attention(x)
        psa_fm = torch.cat([collect_fm, distribute_fm], dim=1)
        return psa_fm


class _AttentionGeneration(nn.Module):

    def __init__(self, in_channels, reduced_channels, out_channels,
        norm_layer, **kwargs):
        super(_AttentionGeneration, self).__init__()
        self.conv_reduce = _ConvBNReLU(in_channels, reduced_channels, 1,
            norm_layer=norm_layer)
        self.attention = nn.Sequential(_ConvBNReLU(reduced_channels,
            reduced_channels, 1, norm_layer=norm_layer), nn.Conv2d(
            reduced_channels, out_channels, 1, bias=False))
        self.reduced_channels = reduced_channels

    def forward(self, x):
        reduce_x = self.conv_reduce(x)
        attention = self.attention(reduce_x)
        n, c, h, w = attention.size()
        attention = attention.view(n, c, -1)
        reduce_x = reduce_x.view(n, self.reduced_channels, -1)
        fm = torch.bmm(reduce_x, torch.softmax(attention, dim=1))
        fm = fm.view(n, self.reduced_channels, h, w)
        return fm


class PyramidPoolingModule(nn.Module):

    def __init__(self, in_size, in_channels, out_channels, setting):
        super(PyramidPoolingModule, self).__init__()
        self.features = []
        for s in setting:
            pool_size = int(math.ceil(float(in_size[0]) / s)), int(math.
                ceil(float(in_size[1]) / s))
            self.features.append(nn.Sequential(nn.AvgPool2d(kernel_size=
                pool_size, stride=pool_size, ceil_mode=True), nn.Conv2d(
                in_channels, out_channels, kernel_size=1, bias=False), nn.
                BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.
                UpsamplingBilinear2d(size=in_size)))
        self.features = nn.ModuleList(modules=self.features)

    def forward(self, x):
        out = []
        out.append(x)
        for m in self.features:
            out.append(m(x))
        out = torch.cat(out, 1)
        return out


class PSPNet(nn.Module):

    def __init__(self, num_classes, pretrained=True, **kwargs):
        super(PSPNet, self).__init__()
        feats = list(models.resnet101(pretrained=pretrained).modules())
        resent = models.resnet101(pretrained=pretrained)
        self.layer0 = nn.Sequential(resent.conv1, resent.bn1, resent.relu,
            resent.maxpool)
        self.layer1 = resent.layer1
        self.layer2 = resent.layer2
        self.layer3 = resent.layer3
        self.layer4 = resent.layer4
        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation = 2, 2
                m.padding = 2, 2
                m.stride = 1, 1
            if 'downsample.0' in n:
                m.stride = 1, 1
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation = 4, 4
                m.padding = 4, 4
                m.stride = 1, 1
            if 'downsample.0' in n:
                m.stride = 1, 1
        self.ppm = PyramidPoolingModule(in_size=(30, 40), in_channels=2048,
            out_channels=512, setting=(1, 2, 3, 6))
        self.final = nn.Sequential(nn.Conv2d(4096, 512, kernel_size=1,
            stride=1, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=
            True), nn.Conv2d(512, num_classes, kernel_size=1))
        self.activation = nn.Sigmoid()
        initialize_weights(self.ppm, self.final)

    def forward(self, x):
        input_size = x.size()
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.ppm(x)
        x = self.final(x)
        upsample = F.interpolate(x, input_size[2:], mode='bilinear')
        return upsample


class _ConvBatchNormReLU(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
        padding, dilation, relu=True):
        super(_ConvBatchNormReLU, self).__init__()
        self.add_module('conv', nn.Conv2d(in_channels=in_channels,
            out_channels=out_channels, kernel_size=kernel_size, stride=
            stride, padding=padding, dilation=dilation, bias=False))
        self.add_module('bn', nn.BatchNorm2d(num_features=out_channels, eps
            =1e-05, momentum=0.999, affine=True))
        if relu:
            self.add_module('relu', nn.ReLU())

    def forward(self, x):
        return super(_ConvBatchNormReLU, self).forward(x)


class _Bottleneck(nn.Sequential):
    """Bottleneck Unit"""

    def __init__(self, in_channels, mid_channels, out_channels, stride,
        dilation, downsample):
        super(_Bottleneck, self).__init__()
        self.reduce = _ConvBatchNormReLU(in_channels, mid_channels, 1,
            stride, 0, 1)
        self.conv3x3 = _ConvBatchNormReLU(mid_channels, mid_channels, 3, 1,
            dilation, dilation)
        self.increase = _ConvBatchNormReLU(mid_channels, out_channels, 1, 1,
            0, 1, relu=False)
        self.downsample = downsample
        if self.downsample:
            self.proj = _ConvBatchNormReLU(in_channels, out_channels, 1,
                stride, 0, 1, relu=False)

    def forward(self, x):
        h = self.reduce(x)
        h = self.conv3x3(h)
        h = self.increase(h)
        if self.downsample:
            h += self.proj(x)
        else:
            h += x
        return F.relu(h)


class _ResBlock(nn.Sequential):
    """Residual Block"""

    def __init__(self, n_layers, in_channels, mid_channels, out_channels,
        stride, dilation, mg=None):
        super(_ResBlock, self).__init__()
        if mg is None:
            mg = [(1) for _ in range(n_layers)]
        else:
            assert n_layers == len(mg
                ), '{} values expected, but got: mg={}'.format(n_layers, mg)
        self.add_module('block1', _Bottleneck(in_channels, mid_channels,
            out_channels, stride, dilation * mg[0], True))
        for i, g in zip(range(2, n_layers + 1), mg[1:]):
            self.add_module('block' + str(i), _Bottleneck(out_channels,
                mid_channels, out_channels, 1, dilation * g, False))

    def __call__(self, x):
        return super(_ResBlock, self).forward(x)


class DenseLayer(nn.Sequential):

    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.add_module('norm', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(True))
        self.add_module('conv', nn.Conv2d(in_channels, growth_rate,
            kernel_size=3, stride=1, padding=1, bias=True))
        self.add_module('drop', nn.Dropout2d(0.2))

    def forward(self, x):
        return super().forward(x)


class DenseBlock(nn.Module):

    def __init__(self, in_channels, growth_rate, n_layers, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.layers = nn.ModuleList([DenseLayer(in_channels + i *
            growth_rate, growth_rate) for i in range(n_layers)])

    def forward(self, x):
        if self.upsample:
            new_features = []
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], 1)
                new_features.append(out)
            return torch.cat(new_features, 1)
        else:
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], 1)
            return x


class TransitionDown(nn.Sequential):

    def __init__(self, in_channels):
        super().__init__()
        self.add_module('norm', nn.BatchNorm2d(num_features=in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(in_channels, in_channels,
            kernel_size=1, stride=1, padding=0, bias=True))
        self.add_module('drop', nn.Dropout2d(0.2))
        self.add_module('maxpool', nn.MaxPool2d(2))

    def forward(self, x):
        return super().forward(x)


class TransitionUp(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convTrans = nn.ConvTranspose2d(in_channels=in_channels,
            out_channels=out_channels, kernel_size=3, stride=2, padding=0,
            bias=True)

    def forward(self, x, skip):
        out = self.convTrans(x)
        out = center_crop(out, skip.size(2), skip.size(3))
        out = torch.cat([out, skip], 1)
        return out


class Bottleneck(nn.Sequential):

    def __init__(self, in_channels, growth_rate, n_layers):
        super().__init__()
        self.add_module('bottleneck', DenseBlock(in_channels, growth_rate,
            n_layers, upsample=True))

    def forward(self, x):
        return super().forward(x)


class FCDenseNet(nn.Module):

    def __init__(self, num_classes=12, in_channels=3, down_blocks=(5, 5, 5,
        5, 5), up_blocks=(5, 5, 5, 5, 5), bottleneck_layers=5, growth_rate=
        16, out_chans_first_conv=48, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.down_blocks = down_blocks
        self.up_blocks = up_blocks
        cur_channels_count = 0
        skip_connection_channel_counts = []
        self.add_module('firstconv', nn.Conv2d(in_channels=in_channels,
            out_channels=out_chans_first_conv, kernel_size=3, stride=1,
            padding=1, bias=True))
        cur_channels_count = out_chans_first_conv
        self.denseBlocksDown = nn.ModuleList([])
        self.transDownBlocks = nn.ModuleList([])
        for i in range(len(down_blocks)):
            self.denseBlocksDown.append(DenseBlock(cur_channels_count,
                growth_rate, down_blocks[i]))
            cur_channels_count += growth_rate * down_blocks[i]
            skip_connection_channel_counts.insert(0, cur_channels_count)
            self.transDownBlocks.append(TransitionDown(cur_channels_count))
        self.add_module('bottleneck', Bottleneck(cur_channels_count,
            growth_rate, bottleneck_layers))
        prev_block_channels = growth_rate * bottleneck_layers
        cur_channels_count += prev_block_channels
        self.transUpBlocks = nn.ModuleList([])
        self.denseBlocksUp = nn.ModuleList([])
        for i in range(len(up_blocks) - 1):
            self.transUpBlocks.append(TransitionUp(prev_block_channels,
                prev_block_channels))
            cur_channels_count = (prev_block_channels +
                skip_connection_channel_counts[i])
            self.denseBlocksUp.append(DenseBlock(cur_channels_count,
                growth_rate, up_blocks[i], upsample=True))
            prev_block_channels = growth_rate * up_blocks[i]
            cur_channels_count += prev_block_channels
        self.transUpBlocks.append(TransitionUp(prev_block_channels,
            prev_block_channels))
        cur_channels_count = (prev_block_channels +
            skip_connection_channel_counts[-1])
        self.denseBlocksUp.append(DenseBlock(cur_channels_count,
            growth_rate, up_blocks[-1], upsample=False))
        cur_channels_count += growth_rate * up_blocks[-1]
        self.finalConv = nn.Conv2d(in_channels=cur_channels_count,
            out_channels=num_classes, kernel_size=1, stride=1, padding=0,
            bias=True)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out = self.firstconv(x)
        skip_connections = []
        for i in range(len(self.down_blocks)):
            out = self.denseBlocksDown[i](out)
            skip_connections.append(out)
            out = self.transDownBlocks[i](out)
        out = self.bottleneck(out)
        for i in range(len(self.up_blocks)):
            skip = skip_connections.pop()
            out = self.transUpBlocks[i](out, skip)
            out = self.denseBlocksUp[i](out)
        out = self.finalConv(out)
        return out


up_kwargs = {'mode': 'bilinear', 'align_corners': True}


class BaseNet(nn.Module):

    def __init__(self, nclass, backbone, aux, se_loss, dilated=True,
        norm_layer=None, base_size=576, crop_size=608, mean=[0.485, 0.456, 
        0.406], std=[0.229, 0.224, 0.225], root='./pretrain_models',
        multi_grid=False, multi_dilation=None, **kwargs):
        super(BaseNet, self).__init__()
        self.nclass = nclass
        self.aux = aux
        self.se_loss = se_loss
        self.mean = mean
        self.std = std
        self.base_size = base_size
        self.crop_size = crop_size
        if backbone == 'resnet50':
            self.pretrained = resnet50(pretrained=True, dilated=dilated,
                norm_layer=norm_layer, root=root, multi_grid=multi_grid,
                multi_dilation=multi_dilation)
        elif backbone == 'resnet101':
            self.pretrained = resnet101(pretrained=True, dilated=dilated,
                norm_layer=norm_layer, root=root, multi_grid=multi_grid,
                multi_dilation=multi_dilation)
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))
        self._up_kwargs = up_kwargs

    def base_forward(self, x):
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)
        c1 = self.pretrained.layer1(x)
        c2 = self.pretrained.layer2(c1)
        c3 = self.pretrained.layer3(c2)
        c4 = self.pretrained.layer4(c3)
        return c1, c2, c3, c4


class BasicBlock(nn.Module):
    """ResNet BasicBlock
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=
        None, previous_dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=
            stride, padding=dilation, dilation=dilation, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
            padding=previous_dilation, dilation=previous_dilation, bias=False)
        self.bn2 = norm_layer(planes)
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


class Bottleneck(nn.Module):
    """ResNet Bottleneck
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=
        None, previous_dilation=1, norm_layer=None, r2_factor=None):
        super(Bottleneck, self).__init__()
        self.r2_factor = r2_factor
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes)
        if self.r2_factor:
            self.avg_pool = nn.AvgPool2d(r2_factor, stride=1, padding=1)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=dilation, dilation=dilation, bias=False)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def _sum_each(self, x, y):
        assert len(x) == len(y)
        z = []
        for i in range(len(x)):
            z.append(x[i] + y[i])
        return z

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if self.r2_factor:
            out = self.avg_pool(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """Dilated Pre-trained ResNet Model, which preduces the stride of 8 featuremaps at conv5.

    Parameters
    ----------
    block : Block
        Class for the residual block. Options are BasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    classes : int, default 1000
        Number of classification classes.
    dilated : bool, default False
        Applying dilation strategy to pretrained ResNet yielding a stride-8 model,
        typically used in Semantic Segmentation.
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).

    Reference:

        - He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

        - Yu, Fisher, and Vladlen Koltun. "Multi-scale context aggregation by dilated convolutions."
    """

    def __init__(self, block, layers, num_classes=1000, dilated=True,
        norm_layer=nn.BatchNorm2d, multi_grid=False, multi_dilation=None):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=
            norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
            norm_layer=norm_layer)
        if dilated:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                dilation=2, norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                dilation=4, norm_layer=norm_layer, r2_factor=3)
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                norm_layer=norm_layer)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, norm_layer):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1,
        norm_layer=None, r2_factor=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion))
        layers = []
        if dilation == 1 or dilation == 2:
            layers.append(block(self.inplanes, planes, stride, dilation=1,
                downsample=downsample, previous_dilation=dilation,
                norm_layer=norm_layer))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, dilation=4,
                downsample=downsample, previous_dilation=dilation,
                norm_layer=norm_layer, r2_factor=r2_factor))
        else:
            raise RuntimeError('=> unknown dilation size: {}'.format(dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            if r2_factor:
                layers.append(block(self.inplanes, planes, dilation=
                    dilation, previous_dilation=dilation, norm_layer=
                    norm_layer, r2_factor=3))
            else:
                layers.append(block(self.inplanes, planes, dilation=
                    dilation, previous_dilation=dilation, norm_layer=
                    norm_layer))
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
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class TFAHead(nn.Module):
    """
       input:
        x: B x C x H x W  (C = 2048)
       output: B x nClass x H x W
    """

    def __init__(self, in_channels, out_channels, norm_layer, r1, r2):
        super(TFAHead, self).__init__()
        inter_channels = in_channels // 4
        self.TFA_level_1 = self._make_level(2048, inter_channels, r1[0], r2
            [0], norm_layer)
        self.TFA_level_list = nn.ModuleList()
        for i in range(1, len(r1)):
            self.TFA_level_list.append(self._make_level(inter_channels,
                inter_channels, r1[i], r2[i], norm_layer))
        self.conv51 = nn.Sequential(nn.Conv2d(in_channels + inter_channels *
            len(r1), inter_channels, 3, padding=1, bias=False), norm_layer(
            inter_channels), nn.ReLU())
        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(
            inter_channels, out_channels, 1))

    def _make_level(self, inChannel, outChannel, r1, r2, norm_layer):
        avg_agg = nn.AvgPool2d(r2, stride=1, padding=r2 // 2)
        conv = nn.Sequential(nn.Conv2d(inChannel, outChannel, kernel_size=3,
            stride=1, padding=r1, dilation=r1), norm_layer(outChannel), nn.
            ReLU())
        return nn.Sequential(avg_agg, conv)

    def forward(self, x):
        TFA_out_list = []
        TFA_out_list.append(x)
        level_1_out = self.TFA_level_1(x)
        TFA_out_list.append(level_1_out)
        for i, layer in enumerate(self.TFA_level_list):
            if i == 0:
                output1 = layer(level_1_out)
                TFA_out_list.append(output1)
            else:
                output1 = layer(output1)
                TFA_out_list.append(output1)
        TFA_out = torch.cat(TFA_out_list, 1)
        out = self.conv51(TFA_out)
        out = self.conv6(out)
        return out


class ConvSamePad2d(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size:
        int, bias: bool=True):
        super().__init__()
        left_top_pad = right_bottom_pad = kernel_size // 2
        if kernel_size % 2 == 0:
            right_bottom_pad -= 1
        self.layer = nn.Sequential(nn.ReflectionPad2d((left_top_pad,
            right_bottom_pad, left_top_pad, right_bottom_pad)), nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size
            =kernel_size, bias=bias))

    def forward(self, inputs):
        return self.layer(inputs)


class StandardUnit(nn.Module):

    def __init__(self, in_channels, out_channels, drop_rate=0.5):
        super().__init__()
        self.layer = nn.Sequential(ConvSamePad2d(in_channels=in_channels,
            out_channels=out_channels, kernel_size=3), nn.Dropout2d(p=
            drop_rate), ConvSamePad2d(in_channels=out_channels,
            out_channels=out_channels, kernel_size=3), nn.Dropout2d(p=
            drop_rate))

    def forward(self, inputs):
        return self.layer(inputs)


class Final1x1ConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(ConvSamePad2d(in_channels=in_channels,
            out_channels=out_channels, kernel_size=1), nn.Sigmoid())

    def forward(self, inputs):
        return self.layer(inputs)


class NestNet(nn.Module):

    def __init__(self, num_classes, in_channels=3, deep_supervision=True,
        **kwargs):
        super().__init__()
        self.deep_supervision = deep_supervision
        filters = [32, 64, 128, 256, 512]
        self.x_00 = StandardUnit(in_channels=in_channels, out_channels=
            filters[0])
        self.pool0 = nn.MaxPool2d(kernel_size=2)
        self.x_01 = StandardUnit(in_channels=filters[0] * 2, out_channels=
            filters[0])
        self.x_02 = StandardUnit(in_channels=filters[0] * 3, out_channels=
            filters[0])
        self.x_03 = StandardUnit(in_channels=filters[0] * 4, out_channels=
            filters[0])
        self.x_04 = StandardUnit(in_channels=filters[0] * 5, out_channels=
            filters[0])
        self.up_10_to_01 = nn.ConvTranspose2d(in_channels=filters[1],
            out_channels=filters[0], kernel_size=2, stride=2)
        self.up_11_to_02 = nn.ConvTranspose2d(in_channels=filters[1],
            out_channels=filters[0], kernel_size=2, stride=2)
        self.up_12_to_03 = nn.ConvTranspose2d(in_channels=filters[1],
            out_channels=filters[0], kernel_size=2, stride=2)
        self.up_13_to_04 = nn.ConvTranspose2d(in_channels=filters[1],
            out_channels=filters[0], kernel_size=2, stride=2)
        self.x_10 = StandardUnit(in_channels=filters[0], out_channels=
            filters[1])
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.x_11 = StandardUnit(in_channels=filters[1] * 2, out_channels=
            filters[1])
        self.x_12 = StandardUnit(in_channels=filters[1] * 3, out_channels=
            filters[1])
        self.x_13 = StandardUnit(in_channels=filters[1] * 4, out_channels=
            filters[1])
        self.up_20_to_11 = nn.ConvTranspose2d(in_channels=filters[2],
            out_channels=filters[1], kernel_size=2, stride=2)
        self.up_21_to_12 = nn.ConvTranspose2d(in_channels=filters[2],
            out_channels=filters[1], kernel_size=2, stride=2)
        self.up_22_to_13 = nn.ConvTranspose2d(in_channels=filters[2],
            out_channels=filters[1], kernel_size=2, stride=2)
        self.x_20 = StandardUnit(in_channels=filters[1], out_channels=
            filters[2])
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.x_21 = StandardUnit(in_channels=filters[2] * 2, out_channels=
            filters[2])
        self.x_22 = StandardUnit(in_channels=filters[2] * 3, out_channels=
            filters[2])
        self.up_30_to_21 = nn.ConvTranspose2d(in_channels=filters[3],
            out_channels=filters[2], kernel_size=2, stride=2)
        self.up_31_to_22 = nn.ConvTranspose2d(in_channels=filters[3],
            out_channels=filters[2], kernel_size=2, stride=2)
        self.x_30 = StandardUnit(in_channels=filters[2], out_channels=
            filters[3])
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.x_31 = StandardUnit(in_channels=filters[3] * 2, out_channels=
            filters[3])
        self.up_40_to_31 = nn.ConvTranspose2d(in_channels=filters[4],
            out_channels=filters[3], kernel_size=2, stride=2)
        self.x_40 = StandardUnit(in_channels=filters[3], out_channels=
            filters[4])
        self.final_1x1_x01 = Final1x1ConvLayer(in_channels=filters[0],
            out_channels=num_classes)
        self.final_1x1_x02 = Final1x1ConvLayer(in_channels=filters[0],
            out_channels=num_classes)
        self.final_1x1_x03 = Final1x1ConvLayer(in_channels=filters[0],
            out_channels=num_classes)
        self.final_1x1_x04 = Final1x1ConvLayer(in_channels=filters[0],
            out_channels=num_classes)

    def forward(self, inputs, L=4):
        if not 1 <= L <= 4:
            raise ValueError(
                'the model pruning factor `L` should be 1 <= L <= 4')
        x_00_output = self.x_00(inputs)
        x_10_output = self.x_10(self.pool0(x_00_output))
        x_10_up_sample = self.up_10_to_01(x_10_output)
        x_01_output = self.x_01(torch.cat([x_00_output, x_10_up_sample], 1))
        nestnet_output_1 = self.final_1x1_x01(x_01_output)
        if L == 1:
            return nestnet_output_1
        x_20_output = self.x_20(self.pool1(x_10_output))
        x_20_up_sample = self.up_20_to_11(x_20_output)
        x_11_output = self.x_11(torch.cat([x_10_output, x_20_up_sample], 1))
        x_11_up_sample = self.up_11_to_02(x_11_output)
        x_02_output = self.x_02(torch.cat([x_00_output, x_01_output,
            x_11_up_sample], 1))
        nestnet_output_2 = self.final_1x1_x01(x_02_output)
        if L == 2:
            if self.deep_supervision:
                return (nestnet_output_1 + nestnet_output_2) / 2
            else:
                return nestnet_output_2
        x_30_output = self.x_30(self.pool2(x_20_output))
        x_30_up_sample = self.up_30_to_21(x_30_output)
        x_21_output = self.x_21(torch.cat([x_20_output, x_30_up_sample], 1))
        x_21_up_sample = self.up_21_to_12(x_21_output)
        x_12_output = self.x_12(torch.cat([x_10_output, x_11_output,
            x_21_up_sample], 1))
        x_12_up_sample = self.up_12_to_03(x_12_output)
        x_03_output = self.x_03(torch.cat([x_00_output, x_01_output,
            x_02_output, x_12_up_sample], 1))
        nestnet_output_3 = self.final_1x1_x01(x_03_output)
        if L == 3:
            if self.deep_supervision:
                return (nestnet_output_1 + nestnet_output_2 + nestnet_output_3
                    ) / 3
            else:
                return nestnet_output_3
        x_40_output = self.x_40(self.pool3(x_30_output))
        x_40_up_sample = self.up_40_to_31(x_40_output)
        x_31_output = self.x_31(torch.cat([x_30_output, x_40_up_sample], 1))
        x_31_up_sample = self.up_31_to_22(x_31_output)
        x_22_output = self.x_22(torch.cat([x_20_output, x_21_output,
            x_31_up_sample], 1))
        x_22_up_sample = self.up_22_to_13(x_22_output)
        x_13_output = self.x_13(torch.cat([x_10_output, x_11_output,
            x_12_output, x_22_up_sample], 1))
        x_13_up_sample = self.up_13_to_04(x_13_output)
        x_04_output = self.x_04(torch.cat([x_00_output, x_01_output,
            x_02_output, x_03_output, x_13_up_sample], 1))
        nestnet_output_4 = self.final_1x1_x01(x_04_output)
        if L == 4:
            if self.deep_supervision:
                return (nestnet_output_1 + nestnet_output_2 +
                    nestnet_output_3 + nestnet_output_4) / 4
            else:
                return nestnet_output_4


class UNetDec(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UNetDec, self).__init__()
        self.up = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3,
            padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(
            inplace=True), nn.Conv2d(out_channels, out_channels, 3, padding
            =1, bias=False), nn.ReLU(inplace=True), nn.BatchNorm2d(
            out_channels), nn.ConvTranspose2d(out_channels, out_channels, 2,
            stride=2), nn.ReLU(inplace=True))
        self.SE1 = nn.Conv2d(in_channels, in_channels // 16, kernel_size=1)
        self.SE2 = nn.Conv2d(in_channels // 16, in_channels, kernel_size=1)

    def forward(self, x):
        fm_size = x.size()[2]
        scale_weight = F.avg_pool2d(x, fm_size)
        scale_weight = F.relu(self.SE1(scale_weight))
        scale_weight = F.sigmoid(self.SE2(scale_weight))
        x = x * scale_weight.expand_as(x)
        out = self.up(x)
        return out


class Dilated_UNetEnc(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=False):
        super(Dilated_UNetEnc, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, 3, padding=1,
            dilation=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(
            inplace=True), nn.Conv2d(out_channels, out_channels, 3, padding
            =1, dilation=1, bias=False), nn.BatchNorm2d(out_channels), nn.
            ReLU(inplace=True), nn.MaxPool2d(2, stride=2, ceil_mode=True)]
        if dropout:
            layers += [nn.Dropout(0.5)]
        self.down = nn.Sequential(*layers)

    def forward(self, x):
        out = self.down(x)
        return out


class Dilated_Bottleneck_block(nn.Module):

    def __init__(self, in_channels, out_channels, dilation_rate, dropout=False
        ):
        super(Dilated_Bottleneck_block, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, 3, padding=
            dilation_rate, dilation=dilation_rate, bias=False), nn.
            BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
        if dropout:
            layers += [nn.Dropout(0.5)]
        self.center = nn.Sequential(*layers)

    def forward(self, x):
        out = self.center(x)
        return out


class Dilated_UNet(nn.Module):

    def __init__(self, inChannel, num_classes, init_features, network_depth,
        bottleneck_layers):
        super(Dilated_UNet, self).__init__()
        self.network_depth = network_depth
        self.bottleneck_layers = bottleneck_layers
        skip_connection_channel_counts = []
        self.add_module('firstconv', nn.Conv2d(in_channels=inChannel,
            out_channels=init_features, kernel_size=3, stride=1, padding=1,
            bias=True))
        self.encodingBlocks = nn.ModuleList([])
        features = init_features
        for i in range(self.network_depth):
            self.encodingBlocks.append(Dilated_UNetEnc(features, 2 * features))
            skip_connection_channel_counts.insert(0, 2 * features)
            features *= 2
        final_encoding_channels = skip_connection_channel_counts[0]
        self.bottleNecks = nn.ModuleList([])
        for i in range(self.bottleneck_layers):
            dilation_factor = 1
            self.bottleNecks.append(Dilated_Bottleneck_block(
                final_encoding_channels, final_encoding_channels,
                dilation_rate=dilation_factor))
        self.decodingBlocks = nn.ModuleList([])
        for i in range(self.network_depth):
            if i == 0:
                prev_deconv_channels = final_encoding_channels
            self.decodingBlocks.append(UNetDec(prev_deconv_channels +
                skip_connection_channel_counts[i],
                skip_connection_channel_counts[i]))
            prev_deconv_channels = skip_connection_channel_counts[i]
        self.final = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        out = self.firstconv(x)
        skip_connections = []
        for i in range(self.network_depth):
            out = self.encodingBlocks[i](out)
            skip_connections.append(out)
        for i in range(self.bottleneck_layers):
            out = self.bottleNecks[i](out)
        for i in range(self.network_depth):
            skip = skip_connections.pop()
            out = self.decodingBlocks[i](torch.cat([out, skip], 1))
        out = self.final(out)
        return out


class DenseLayer(nn.Sequential):

    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.add_module('norm', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(True))
        self.add_module('conv', nn.Conv2d(in_channels, growth_rate,
            kernel_size=3, stride=1, padding=1, bias=True))
        self.add_module('drop', nn.Dropout2d(0.2))

    def forward(self, x):
        return super().forward(x)


class DenseBlock(nn.Module):

    def __init__(self, in_channels, growth_rate, n_layers, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.layers = nn.ModuleList([DenseLayer(in_channels + i *
            growth_rate, growth_rate) for i in range(n_layers)])

    def forward(self, x):
        if self.upsample:
            new_features = []
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], 1)
                new_features.append(out)
            return torch.cat(new_features, 1)
        else:
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], 1)
            return x


class TransitionDown(nn.Sequential):

    def __init__(self, in_channels):
        super().__init__()
        self.add_module('norm', nn.BatchNorm2d(num_features=in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(in_channels, in_channels,
            kernel_size=1, stride=1, padding=0, bias=True))
        self.add_module('drop', nn.Dropout2d(0.2))
        self.add_module('maxpool', nn.MaxPool2d(2))

    def forward(self, x):
        return super().forward(x)


class TransitionUp(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convTrans = nn.ConvTranspose2d(in_channels=in_channels,
            out_channels=out_channels, kernel_size=3, stride=2, padding=0,
            bias=True)

    def forward(self, x, skip):
        out = self.convTrans(x)
        out = center_crop(out, skip.size(2), skip.size(3))
        out = torch.cat([out, skip], 1)
        return out


class Bottleneck(nn.Sequential):

    def __init__(self, in_channels, growth_rate, n_layers):
        super().__init__()
        self.add_module('bottleneck', DenseBlock(in_channels, growth_rate,
            n_layers, upsample=True))

    def forward(self, x):
        return super().forward(x)


class FCDenseNet(nn.Module):

    def __init__(self, in_channels=3, down_blocks=(5, 5, 5, 5, 5),
        up_blocks=(5, 5, 5, 5, 5), bottleneck_layers=5, growth_rate=16,
        out_chans_first_conv=48, num_classes=12, **kwargs):
        super().__init__()
        self.down_blocks = down_blocks
        self.up_blocks = up_blocks
        cur_channels_count = 0
        skip_connection_channel_counts = []
        self.add_module('firstconv', nn.Conv2d(in_channels=in_channels,
            out_channels=out_chans_first_conv, kernel_size=3, stride=1,
            padding=1, bias=True))
        cur_channels_count = out_chans_first_conv
        self.denseBlocksDown = nn.ModuleList([])
        self.transDownBlocks = nn.ModuleList([])
        for i in range(len(down_blocks)):
            self.denseBlocksDown.append(DenseBlock(cur_channels_count,
                growth_rate, down_blocks[i]))
            cur_channels_count += growth_rate * down_blocks[i]
            skip_connection_channel_counts.insert(0, cur_channels_count)
            self.transDownBlocks.append(TransitionDown(cur_channels_count))
        self.add_module('bottleneck', Bottleneck(cur_channels_count,
            growth_rate, bottleneck_layers))
        prev_block_channels = growth_rate * bottleneck_layers
        cur_channels_count += prev_block_channels
        self.transUpBlocks = nn.ModuleList([])
        self.denseBlocksUp = nn.ModuleList([])
        for i in range(len(up_blocks) - 1):
            self.transUpBlocks.append(TransitionUp(prev_block_channels,
                prev_block_channels))
            cur_channels_count = (prev_block_channels +
                skip_connection_channel_counts[i])
            self.denseBlocksUp.append(DenseBlock(cur_channels_count,
                growth_rate, up_blocks[i], upsample=True))
            prev_block_channels = growth_rate * up_blocks[i]
            cur_channels_count += prev_block_channels
        self.transUpBlocks.append(TransitionUp(prev_block_channels,
            prev_block_channels))
        cur_channels_count = (prev_block_channels +
            skip_connection_channel_counts[-1])
        self.denseBlocksUp.append(DenseBlock(cur_channels_count,
            growth_rate, up_blocks[-1], upsample=False))
        cur_channels_count += growth_rate * up_blocks[-1]
        self.finalConv = nn.Conv2d(in_channels=cur_channels_count,
            out_channels=num_classes, kernel_size=1, stride=1, padding=0,
            bias=True)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out = self.firstconv(x)
        skip_connections = []
        for i in range(len(self.down_blocks)):
            out = self.denseBlocksDown[i](out)
            skip_connections.append(out)
            out = self.transDownBlocks[i](out)
        out = self.bottleneck(out)
        for i in range(len(self.up_blocks)):
            skip = skip_connections.pop()
            out = self.transUpBlocks[i](out, skip)
            out = self.denseBlocksUp[i](out)
        out = self.finalConv(out)
        out = self.softmax(out)
        return out


class _EncoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=False):
        super(_EncoderBlock, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3), nn.
            BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.Conv2d(
            out_channels, out_channels, kernel_size=3), nn.BatchNorm2d(
            out_channels), nn.ReLU(inplace=True)]
        if dropout:
            layers.append(nn.Dropout())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class _DecoderBlock(nn.Module):

    def __init__(self, in_channels, middle_channels, out_channels):
        super(_DecoderBlock, self).__init__()
        self.decode = nn.Sequential(nn.Conv2d(in_channels, middle_channels,
            kernel_size=3), nn.BatchNorm2d(middle_channels), nn.ReLU(
            inplace=True), nn.Conv2d(middle_channels, middle_channels,
            kernel_size=3), nn.BatchNorm2d(middle_channels), nn.ReLU(
            inplace=True), nn.ConvTranspose2d(middle_channels, out_channels,
            kernel_size=2, stride=2))

    def forward(self, x):
        return self.decode(x)


class UNet(nn.Module):
    """
    Basic Unet
    """

    def __init__(self, num_classes, **kwargs):
        super(UNet, self).__init__()
        self.enc1 = _EncoderBlock(3, 64)
        self.enc2 = _EncoderBlock(64, 128)
        self.enc3 = _EncoderBlock(128, 256)
        self.enc4 = _EncoderBlock(256, 512, dropout=True)
        self.center = _DecoderBlock(512, 1024, 512)
        self.dec4 = _DecoderBlock(1024, 512, 256)
        self.dec3 = _DecoderBlock(512, 256, 128)
        self.dec2 = _DecoderBlock(256, 128, 64)
        self.dec1 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3), nn.
            BatchNorm2d(64), nn.ReLU(inplace=True), nn.Conv2d(64, 64,
            kernel_size=3), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        initialize_weights(self)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        center = self.center(enc4)
        dec4 = self.dec4(torch.cat([center, F.interpolate(enc4, center.size
            ()[2:], mode='bilinear')], 1))
        dec3 = self.dec3(torch.cat([dec4, F.interpolate(enc3, dec4.size()[2
            :], mode='bilinear')], 1))
        dec2 = self.dec2(torch.cat([dec3, F.interpolate(enc2, dec3.size()[2
            :], mode='bilinear')], 1))
        dec1 = self.dec1(torch.cat([dec2, F.interpolate(enc1, dec2.size()[2
            :], mode='bilinear')], 1))
        final = self.final(dec1)
        return F.interpolate(final, x.size()[2:], mode='bilinear')


class Conv_transition(nn.Module):
    """
    resnet block contains inception
    """

    def __init__(self, kernel_size, in_channels, out_channels):
        super(Conv_transition, self).__init__()
        if not kernel_size:
            kernel_size = [1, 3, 5]
        paddings = [int(a / 2) for a in kernel_size]
        self.Conv1 = nn.Conv2d(in_channels, out_channels, kernel_size[0],
            stride=1, padding=paddings[0])
        self.Conv2 = nn.Conv2d(in_channels, out_channels, kernel_size[1],
            stride=1, padding=paddings[1])
        self.Conv3 = nn.Conv2d(in_channels, out_channels, kernel_size[2],
            stride=1, padding=paddings[2])
        self.Conv_f = nn.Conv2d(3 * out_channels, out_channels, 3, stride=1,
            padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.PReLU()

    def forward(self, x):
        x1 = self.act(self.Conv1(x))
        x2 = self.act(self.Conv2(x))
        x3 = self.act(self.Conv3(x))
        x = torch.cat([x1, x2, x3], dim=1)
        return self.act(self.bn(self.Conv_f(x)))


class Dense_layer(nn.Module):
    """
    an two-layer
    """

    def __init__(self, in_channels, growth_rate):
        super(Dense_layer, self).__init__()
        self.Conv0 = nn.Conv2d(in_channels, in_channels + growth_rate, 3,
            stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels + growth_rate)
        self.Conv1 = nn.Conv2d(in_channels + growth_rate, growth_rate,
            kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels + growth_rate)
        self.Conv2 = nn.Conv2d(in_channels + growth_rate, in_channels,
            kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(in_channels)
        self.act = nn.PReLU()

    def forward(self, x):
        x1 = self.act(self.bn1(self.Conv0(x)))
        x1 = self.act(self.bn2(torch.cat([self.Conv1(x1), x], dim=1)))
        return self.act(self.bn3(self.Conv2(x1)))


class Fire_Down(nn.Module):

    def __init__(self, kernel_size, in_channels, inner_channels, out_channels):
        super(Fire_Down, self).__init__()
        dilations = [1, 3, 5]
        self.Conv1 = nn.Conv2d(in_channels, inner_channels, kernel_size=
            kernel_size, stride=1, padding=dilations[0], dilation=dilations[0])
        self.Conv4 = nn.Conv2d(in_channels, inner_channels, kernel_size=
            kernel_size, stride=1, padding=dilations[1], dilation=dilations[1])
        self.Conv8 = nn.Conv2d(in_channels, inner_channels, kernel_size=
            kernel_size, stride=1, padding=dilations[2], dilation=dilations[2])
        self.Conv_f3 = nn.Conv2d(3 * inner_channels, out_channels,
            kernel_size=kernel_size, stride=2, padding=1)
        self.Conv_f1 = nn.Conv2d(out_channels, out_channels, kernel_size=1,
            stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = nn.PReLU()

    def forward(self, x):
        x1 = self.act(self.Conv1(x))
        x2 = self.act(self.Conv4(x))
        x3 = self.act(self.Conv8(x))
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.act(self.Conv_f3(x))
        return self.act(self.bn1(self.Conv_f1(x)))


class Fire_Up(nn.Module):

    def __init__(self, kernel_size, in_channels, inner_channels,
        out_channels, out_padding=(1, 1)):
        super(Fire_Up, self).__init__()
        padds = int(kernel_size / 2)
        self.Conv1 = nn.Conv2d(in_channels, inner_channels, kernel_size=3,
            stride=1, padding=1)
        if not out_padding:
            out_padding = 1, 1
        self.ConvT4 = nn.ConvTranspose2d(inner_channels, out_channels,
            kernel_size=kernel_size, stride=2, padding=padds,
            output_padding=out_padding)
        self.Conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1,
            stride=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = nn.PReLU()

    def forward(self, x):
        x = self.act(self.Conv1(x))
        x = self.act(self.ConvT4(x))
        x = self.act(self.bn1(self.Conv2(x)))
        return x


class UNetDilated(nn.Module):
    """
    Unet utilizing dilation
    """

    def __init__(self, num_classes, **kwargs):
        super(UNetDilated, self).__init__()
        self.Conv0 = self._transition(3, 8)
        self.down1 = self._down_block(8, 16, 16)
        self.down2 = self._down_block(16, 16, 32)
        self.down3 = self._down_block(32, 32, 64)
        self.down4 = self._down_block(64, 64, 96)
        self.down5 = self._down_block(96, 96, 128)
        self.tran0 = self._transition(128, 256)
        self.db0 = self._dense_block(256, 32)
        self.up1 = self._up_block(256, 96, 96)
        self.db1 = self._dense_block(96, 32)
        self.conv1 = nn.Conv2d(96 * 2, 96, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(96)
        self.up2 = self._up_block(96, 64, 64)
        self.db2 = self._dense_block(64, 24)
        self.conv2 = nn.Conv2d(64 * 2, 64, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.up3 = self._up_block(64, 32, 32)
        self.db3 = self._dense_block(32, 10)
        self.conv3 = nn.Conv2d(32 * 2, 32, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.up4 = self._up_block(32, 16, 16)
        self.db4 = self._dense_block(16, 8)
        self.conv4 = nn.Conv2d(16 * 2, 16, 3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(16)
        self.up5 = self._up_block(16, 16, 16)
        self.db5 = self._dense_block(16, 4)
        self.conv5 = nn.Conv2d(16, num_classes, 3, stride=1, padding=1)
        self.clss = nn.LogSoftmax()
        self.act = nn.PReLU()

    def forward(self, x):
        x1 = self.Conv0(x)
        down1 = self.down1(x1)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        down5 = self.down5(down4)
        down5 = self.tran0(down5)
        down5 = self.db0(down5)
        up1 = self.act(self.bn1(self.conv1(torch.cat([self.db1(self.up1(
            down5)), down4], dim=1))))
        del down5, down4
        up2 = self.act(self.bn2(self.conv2(torch.cat([self.db2(self.up2(up1
            )), down3], dim=1))))
        del down3
        up3 = self.act(self.bn3(self.conv3(torch.cat([self.db3(self.up3(up2
            )), down2], dim=1))))
        del down2
        up4 = self.act(self.bn4(self.conv4(torch.cat([self.db4(self.up4(up3
            )), down1], dim=1))))
        del down1
        up5 = self.up5(up4)
        return self.conv5(up5)

    def _transition(self, in_channels, out_channels):
        layers = []
        layers.append(Conv_transition([1, 3, 5], in_channels, out_channels))
        return nn.Sequential(*layers)

    def _down_block(self, in_channels, inner_channels, out_channels):
        layers = []
        layers.append(Fire_Down(3, in_channels, inner_channels, out_channels))
        return nn.Sequential(*layers)

    def _up_block(self, in_channels, inner_channels, out_channels,
        output_padding=(1, 1)):
        layers = []
        layers.append(Fire_Up(3, in_channels, inner_channels, out_channels,
            output_padding))
        return nn.Sequential(*layers)

    def _dense_block(self, in_channels, growth_rate):
        layers = []
        layers.append(Dense_layer(in_channels, growth_rate))
        return nn.Sequential(*layers)


class UnetEncoder(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UnetEncoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layer = nn.Sequential(nn.Conv2d(self.in_channels, self.
            out_channels, kernel_size=3, padding=1), nn.BatchNorm2d(self.
            out_channels), nn.ReLU(), nn.Conv2d(self.out_channels, self.
            out_channels, kernel_size=3, padding=1), nn.BatchNorm2d(self.
            out_channels), nn.ReLU())

    def forward(self, x):
        return self.layer(x)


class UnetDecoder(nn.Module):

    def __init__(self, in_channels, featrures, out_channels):
        super(UnetDecoder, self).__init__()
        self.in_channels = in_channels
        self.features = featrures
        self.out_channels = out_channels
        self.layer = nn.Sequential(nn.Conv2d(self.in_channels, self.
            features, kernel_size=3, padding=1), nn.BatchNorm2d(self.
            features), nn.ReLU(), nn.Conv2d(self.features, self.features,
            kernel_size=3, padding=1), nn.BatchNorm2d(self.features), nn.
            ReLU(), nn.ConvTranspose2d(self.features, self.out_channels,
            kernel_size=2, stride=2), nn.BatchNorm2d(self.out_channels), nn
            .ReLU())

    def forward(self, x):
        return self.layer(x)


class UNet(nn.Module):

    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.num_classes = num_classes
        self.down1 = UnetEncoder(3, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.down2 = UnetEncoder(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.down3 = UnetEncoder(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.down4 = UnetEncoder(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.center = nn.Sequential(nn.Conv2d(512, 1024, kernel_size=3,
            padding=1), nn.BatchNorm2d(1024), nn.ReLU(), nn.Conv2d(1024, 
            1024, kernel_size=3, padding=1), nn.BatchNorm2d(1024), nn.ReLU(
            ), nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2), nn.
            BatchNorm2d(512), nn.ReLU())
        self.up1 = UnetDecoder(1024, 512, 256)
        self.up2 = UnetDecoder(512, 256, 128)
        self.up3 = UnetDecoder(256, 128, 64)
        self.up4 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.
            Conv2d(64, 64, 3, padding=1))
        self.output = nn.Conv2d(64, self.num_classes, kernel_size=1, stride=1)
        self.final = nn.Sigmoid()
        initialize_weights(self)

    def forward(self, x):
        en1 = self.down1(x)
        po1 = self.pool1(en1)
        en2 = self.down2(po1)
        po2 = self.pool2(en2)
        en3 = self.down3(po2)
        po3 = self.pool3(en3)
        en4 = self.down4(po3)
        po4 = self.pool4(en4)
        c1 = self.center(po4)
        dec1 = self.up1(torch.cat([c1, F.interpolate(en4, c1.size()[2:],
            mode='bilinear')], 1))
        dec2 = self.up2(torch.cat([dec1, F.interpolate(en3, dec1.size()[2:],
            mode='bilinear')], 1))
        dec3 = self.up3(torch.cat([dec2, F.interpolate(en2, dec2.size()[2:],
            mode='bilinear')], 1))
        dec4 = self.up4(torch.cat([dec3, F.interpolate(en1, dec3.size()[2:],
            mode='bilinear')], 1))
        out = self.output(dec4)
        return self.final(out)


class Conv2dX2_Res(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(Conv2dX2_Res, self).__init__()
        self.layer = nn.Sequential(nn.Conv2d(in_channels, out_channels,
            kernel_size=kernel_size, padding=padding), nn.BatchNorm2d(
            out_channels), nn.ReLU(inplace=True), nn.Conv2d(out_channels,
            out_channels, kernel_size=kernel_size, padding=padding))

    def forward(self, x):
        conv = self.layer(x)
        return F.relu(x.expand_as(conv) + conv)


class PassConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
        padding=0):
        super(PassConv, self).__init__()
        self.layer = nn.Sequential(nn.Conv2d(in_channels, out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding), nn.
            ReLU(inplace=True))

    def forward(self, x):
        return self.layer(x)


class DeconvX2_Res(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super(DeconvX2_Res, self).__init__()
        self.convx2_res = Conv2dX2_Res(in_channels, in_channels,
            kernel_size=3, padding=1)
        self.upsample = nn.Sequential(nn.ConvTranspose2d(in_channels,
            out_channels, kernel_size=kernel_size, stride=stride), nn.ReLU(
            inplace=True))

    def forward(self, x):
        convx2_res = self.convx2_res(x)
        return self.upsample(convx2_res)


class UNetRes(nn.Module):

    def __init__(self, num_classes, **kwargs):
        super(UNetRes, self).__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(
            inplace=True), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace
            =True))
        self.pool1 = nn.Conv2d(64, 128, kernel_size=2, stride=2)
        self.enc2 = Conv2dX2_Res(128, 128, 3, padding=1)
        self.pool2 = nn.Conv2d(128, 256, kernel_size=2, stride=2)
        self.enc3 = Conv2dX2_Res(256, 256, 3, padding=1)
        self.pool3 = nn.Conv2d(256, 512, kernel_size=2, stride=2)
        self.enc4 = Conv2dX2_Res(512, 512, 3, padding=1)
        self.pool4 = nn.Conv2d(512, 1024, kernel_size=2, stride=2)
        self.middle = nn.Sequential(Conv2dX2_Res(1024, 1024, 3, padding=1),
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2), nn.ReLU
            (inplace=True))
        self.pass_enc4 = PassConv(512, 512)
        self.pass_enc3 = PassConv(256, 256)
        self.pass_enc2 = PassConv(128, 128)
        self.pass_enc1 = PassConv(64, 64)
        self.dec1 = DeconvX2_Res(512, 256, 2, stride=2)
        self.dec2 = DeconvX2_Res(256, 128, 2, stride=2)
        self.dec3 = DeconvX2_Res(128, 64, 2, stride=2)
        self.dec4 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.
            Conv2d(64, num_classes, kernel_size=1, stride=1))
        self.activation = nn.Sigmoid()
        initialize_weights(self)

    def forward(self, x):
        en1 = self.enc1(x)
        en2 = self.enc2(self.pool1(en1))
        en3 = self.enc3(self.pool2(en2))
        en4 = self.enc4(self.pool3(en3))
        middle = self.middle(self.pool4(en4))
        dec1 = self.dec1(en4 + middle)
        dec2 = self.dec2(en3 + dec1)
        dec3 = self.dec3(en2 + dec2)
        dec4 = self.dec4(en1 + dec3)
        return self.activation(dec4)


class ConvBNReluStack(nn.Module):

    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=1,
        **kwargs):
        super(ConvBNReluStack, self).__init__()
        in_dim = int(in_dim)
        out_dim = int(out_dim)
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size, stride=stride,
            padding=padding)
        self.bn = nn.BatchNorm2d(out_dim)
        self.activation = nn.PReLU()

    def forward(self, inputs_):
        x = self.conv(inputs_)
        x = self.bn(x)
        x = self.activation(x)
        return x


class UNetDownStack(nn.Module):

    def __init__(self, input_dim, filters, pool=True):
        super(UNetDownStack, self).__init__()
        self.stack1 = ConvBNReluStack(input_dim, filters, 1, stride=1,
            padding=0)
        self.stack3 = ConvBNReluStack(input_dim, filters, 3, stride=1,
            padding=1)
        self.stack5 = ConvBNReluStack(input_dim, filters, 5, stride=1,
            padding=2)
        self.stack_pool = nn.AvgPool2d(3, stride=1, padding=1)
        self.reducer = ConvBNReluStack(filters * 3 + input_dim, filters,
            kernel_size=1, stride=1, padding=0)
        self.pool = nn.MaxPool2d(2, stride=2) if pool else None

    def forward(self, inputs_):
        x1 = self.stack1(inputs_)
        x3 = self.stack3(inputs_)
        x5 = self.stack5(inputs_)
        x_pool = self.stack_pool(inputs_)
        x = torch.cat([x1, x3, x5, x_pool], dim=1)
        x = self.reducer(x)
        if self.pool:
            return x, self.pool(x)
        return x


class UNetUpStack(nn.Module):

    def __init__(self, input_dim, filters, kernel_size=3):
        super(UNetUpStack, self).__init__()
        self.scale_factor = 2
        self.stack1 = ConvBNReluStack(input_dim, filters, 1, stride=1,
            padding=0)
        self.stack3 = ConvBNReluStack(input_dim, filters, 3, stride=1,
            padding=1)
        self.stack5 = ConvBNReluStack(input_dim, filters, 5, stride=1,
            padding=2)
        self.stack_pool = nn.AvgPool2d(3, stride=1, padding=1)
        self.reducer = ConvBNReluStack(filters * 3 + input_dim, filters,
            kernel_size=1, stride=1, padding=0)

    def forward(self, inputs_, down):
        x = F.interpolate(inputs_, scale_factor=self.scale_factor)
        x = torch.cat([x, down], dim=1)
        x1 = self.stack1(x)
        x3 = self.stack3(x)
        x5 = self.stack5(x)
        x_pool = self.stack_pool(x)
        x = torch.cat([x1, x3, x5, x_pool], dim=1)
        x = self.reducer(x)
        return x


class UNet_stack(nn.Module):

    def get_n_stacks(self, input_size, **kwargs):
        n_stacks = 0
        width, height = input_size, input_size
        while width % 2 == 0 and height % 2 == 0:
            n_stacks += 1
            width = width // 2
            height = height // 2
        return n_stacks

    def __init__(self, input_size=512, filters=12, kernel_size=3,
        max_stacks=6, **kwargs):
        super(UNet_stack, self).__init__()
        self.n_stacks = min(self.get_n_stacks((input_size, input_size)),
            max_stacks)
        self.down1 = UNetDownStack(3, filters)
        prev_filters = filters
        for i in range(2, self.n_stacks + 1):
            n = i
            layer = UNetDownStack(prev_filters, prev_filters * 2)
            layer_name = 'down' + str(n)
            setattr(self, layer_name, layer)
            prev_filters *= 2
        self.center = UNetDownStack(prev_filters, prev_filters * 2, pool=False)
        prev_filters = prev_filters * 3
        for i in range(self.n_stacks):
            n = self.n_stacks - i
            layer = UNetUpStack(prev_filters, prev_filters // 3, kernel_size)
            layer_name = 'up' + str(n)
            setattr(self, layer_name, layer)
            prev_filters = prev_filters // 2
        self.classify = nn.Conv2d(prev_filters * 2 // 3, 1, kernel_size,
            stride=1, padding=1)

    def forward(self, inputs_):
        down1, down1_pool = self.down1(inputs_)
        downs = [down1]
        prev_down_pool = down1_pool
        for i in range(2, self.n_stacks + 1):
            layer_name = 'down' + str(i)
            layer = getattr(self, layer_name)
            down, prev_down_pool = layer(prev_down_pool)
            downs.append(down)
        center = self.center(prev_down_pool)
        prev = center
        for i in range(self.n_stacks):
            n = self.n_stacks - i
            matching_down = downs.pop()
            layer_name = 'up' + str(n)
            layer = getattr(self, layer_name)
            prev = layer(prev, matching_down)
        x = self.classify(prev)
        return x


class UNet960(nn.Module):

    def __init__(self, filters=12, kernel_size=3, **kwargs):
        super(UNet960, self).__init__()
        self.down1 = UNetDownStack(3, filters)
        self.down2 = UNetDownStack(filters, filters * 2)
        self.down3 = UNetDownStack(filters * 2, filters * 4)
        self.down4 = UNetDownStack(filters * 4, filters * 8)
        self.down5 = UNetDownStack(filters * 8, filters * 16)
        self.down6 = UNetDownStack(filters * 16, filters * 32)
        self.center = UNetDownStack(filters * 32, filters * 64, pool=False)
        self.up6 = UNetUpStack(filters * 96, filters * 32, kernel_size)
        self.up5 = UNetUpStack(filters * 48, filters * 16, kernel_size)
        self.up4 = UNetUpStack(filters * 24, filters * 8, kernel_size)
        self.up3 = UNetUpStack(filters * 12, filters * 4, kernel_size)
        self.up2 = UNetUpStack(filters * 6, filters * 2, kernel_size)
        self.up1 = UNetUpStack(filters * 3, filters, kernel_size)
        self.classify = nn.Conv2d(filters, 1, kernel_size, stride=1, padding=1)

    def forward(self, inputs_):
        down1, down1_pool = self.down1(inputs_)
        down2, down2_pool = self.down2(down1_pool)
        down3, down3_pool = self.down3(down2_pool)
        down4, down4_pool = self.down4(down3_pool)
        down5, down5_pool = self.down5(down4_pool)
        down6, down6_pool = self.down6(down5_pool)
        center = self.center(down6_pool)
        up6 = self.up6(center, down6)
        up5 = self.up5(up6, down5)
        up4 = self.up4(up5, down4)
        up3 = self.up3(up4, down3)
        up2 = self.up2(up3, down2)
        up1 = self.up1(up2, down1)
        x = self.classify(up1)
        return x


def F_bilinear_interp2d(input, coords):
    """
    bilinear interpolation of 2d torch Tensor
    """
    x = torch.clamp(coords[:, :, (0)], 0, input.size(1) - 2)
    x0 = x.floor()
    x1 = x0 + 1
    y = torch.clamp(coords[:, :, (1)], 0, input.size(2) - 2)
    y0 = y.floor()
    y1 = y0 + 1
    stride = torch.LongTensor(input.stride())
    x0_ix = x0.mul(stride[1]).long()
    x1_ix = x1.mul(stride[1]).long()
    y0_ix = y0.mul(stride[2]).long()
    y1_ix = y1.mul(stride[2]).long()
    input_flat = input.view(input.size(0), -1).contiguous()
    vals_00 = input_flat.gather(1, x0_ix.add(y0_ix).detach())
    vals_10 = input_flat.gather(1, x1_ix.add(y0_ix).detach())
    vals_01 = input_flat.gather(1, x0_ix.add(y1_ix).detach())
    vals_11 = input_flat.gather(1, x1_ix.add(y1_ix).detach())
    xd = x - x0
    yd = y - y0
    xm = 1 - xd
    ym = 1 - yd
    x_mapped = vals_00.mul(xm).mul(ym) + vals_10.mul(xd).mul(ym) + vals_01.mul(
        xm).mul(yd) + vals_11.mul(xd).mul(yd)
    return x_mapped.view_as(input)


def th_iterproduct(*args):
    return th.from_numpy(np.indices(args).reshape((len(args), -1)).T)


def F_affine2d(x, matrix, center=True):
    """
    2D Affine image transform on torch Tensor
    """
    if matrix.dim() == 2:
        matrix = matrix.view(-1, 2, 3)
    A_batch = matrix[:, :, :2]
    if A_batch.size(0) != x.size(0):
        A_batch = A_batch.repeat(x.size(0), 1, 1)
    b_batch = matrix[:, :, (2)].unsqueeze(1)
    _coords = th_iterproduct(x.size(1), x.size(2))
    with torch.no_grad:
        coords = _coords.unsqueeze(0).repeat(x.size(0), 1, 1).float()
    if center:
        coords[:, :, (0)] = coords[:, :, (0)] - (x.size(1) / 2.0 + 0.5)
        coords[:, :, (1)] = coords[:, :, (1)] - (x.size(2) / 2.0 + 0.5)
    new_coords = coords.bmm(A_batch.transpose(1, 2)) + b_batch.expand_as(coords
        )
    if center:
        new_coords[:, :, (0)] = new_coords[:, :, (0)] + (x.size(1) / 2.0 + 0.5)
        new_coords[:, :, (1)] = new_coords[:, :, (1)] + (x.size(2) / 2.0 + 0.5)
    x_transformed = F_bilinear_interp2d(x, new_coords)
    return x_transformed


class STN2d(nn.Module):

    def __init__(self, local_net):
        super(STN2d, self).__init__()
        self.local_net = local_net

    def forward(self, x):
        params = self.local_net(x)
        x_transformed = F_affine2d(x[0], params.view(2, 3))
        return x_transformed


def th_flatten(x):
    """Flatten tensor"""
    return x.contiguous().view(-1)


def F_trilinear_interp3d(input, coords):
    """
    trilinear interpolation of 3D image
    """
    x = torch.clamp(coords[:, (0)], 0, input.size(1) - 2)
    x0 = x.floor()
    x1 = x0 + 1
    y = torch.clamp(coords[:, (1)], 0, input.size(2) - 2)
    y0 = y.floor()
    y1 = y0 + 1
    z = torch.clamp(coords[:, (2)], 0, input.size(3) - 2)
    z0 = z.floor()
    z1 = z0 + 1
    stride = torch.LongTensor(input.stride())[1:]
    x0_ix = x0.mul(stride[0]).long()
    x1_ix = x1.mul(stride[0]).long()
    y0_ix = y0.mul(stride[1]).long()
    y1_ix = y1.mul(stride[1]).long()
    z0_ix = z0.mul(stride[2]).long()
    z1_ix = z1.mul(stride[2]).long()
    input_flat = th_flatten(input)
    vals_000 = input_flat[x0_ix.add(y0_ix).add(z0_ix).detach()]
    vals_100 = input_flat[x1_ix.add(y0_ix).add(z0_ix).detach()]
    vals_010 = input_flat[x0_ix.add(y1_ix).add(z0_ix).detach()]
    vals_001 = input_flat[x0_ix.add(y0_ix).add(z1_ix).detach()]
    vals_101 = input_flat[x1_ix.add(y0_ix).add(z1_ix).detach()]
    vals_011 = input_flat[x0_ix.add(y1_ix).add(z1_ix).detach()]
    vals_110 = input_flat[x1_ix.add(y1_ix).add(z0_ix).detach()]
    vals_111 = input_flat[x1_ix.add(y1_ix).add(z1_ix).detach()]
    xd = x - x0
    yd = y - y0
    zd = z - z0
    xm = 1 - xd
    ym = 1 - yd
    zm = 1 - zd
    x_mapped = vals_000.mul(xm).mul(ym).mul(zm) + vals_100.mul(xd).mul(ym).mul(
        zm) + vals_010.mul(xm).mul(yd).mul(zm) + vals_001.mul(xm).mul(ym).mul(
        zd) + vals_101.mul(xd).mul(ym).mul(zd) + vals_011.mul(xm).mul(yd).mul(
        zd) + vals_110.mul(xd).mul(yd).mul(zm) + vals_111.mul(xd).mul(yd).mul(
        zd)
    return x_mapped.view_as(input)


def F_affine3d(x, matrix, center=True):
    A = matrix[:3, :3]
    b = matrix[:3, (3)]
    with torch.no_grad:
        coords = th_iterproduct(x.size(1), x.size(2), x.size(3)).float()
    if center:
        coords[:, (0)] = coords[:, (0)] - (x.size(1) / 2.0 + 0.5)
        coords[:, (1)] = coords[:, (1)] - (x.size(2) / 2.0 + 0.5)
        coords[:, (2)] = coords[:, (2)] - (x.size(3) / 2.0 + 0.5)
    new_coords = F.linear(coords, A, b)
    if center:
        new_coords[:, (0)] = new_coords[:, (0)] + (x.size(1) / 2.0 + 0.5)
        new_coords[:, (1)] = new_coords[:, (1)] + (x.size(2) / 2.0 + 0.5)
        new_coords[:, (2)] = new_coords[:, (2)] + (x.size(3) / 2.0 + 0.5)
    x_transformed = F_trilinear_interp3d(x, new_coords)
    return x_transformed


class STN3d(nn.Module):

    def __init__(self, local_net):
        self.local_net = local_net

    def forward(self, x):
        params = self.local_net(x)
        x_transformed = F_affine3d(x, params.view(3, 4))
        return x_transformed


class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(1600, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x, y, z):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 1600)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x), F.log_softmax(x), F.log_softmax(x)


class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(1600, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 1600)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x), F.log_softmax(x)


class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(1600, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 1600)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)


class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(1600, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x, y, z):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 1600)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x), F.log_softmax(x), F.log_softmax(x)


class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(1600, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x, y, z):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 1600)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return th.abs(10 - x)


class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(1600, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x, y, z):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 1600)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(1600, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 1600)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x), F.log_softmax(x)


class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(1600, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 1600)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return th.abs(10 - x)


class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(1600, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 1600)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)

