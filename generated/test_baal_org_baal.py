import sys
_module = sys.modules[__name__]
del sys
baal = _module
active = _module
active_loop = _module
dataset = _module
base = _module
nlp_datasets = _module
numpy = _module
pytorch_dataset = _module
file_dataset = _module
heuristics = _module
heuristics = _module
heuristics_gpu = _module
stochastics = _module
bayesian = _module
common = _module
consistent_dropout = _module
dropout = _module
weight_drop = _module
calibration = _module
calibration = _module
ensemble = _module
metrics = _module
mixin = _module
modelwrapper = _module
transformers_trainer_wrapper = _module
utils = _module
array_utils = _module
cuda_utils = _module
equality = _module
iterutils = _module
log_configuration = _module
metrics = _module
plot_utils = _module
pytorch_lightning = _module
ssl_iterator = _module
ssl_module = _module
transforms = _module
experiments = _module
mlp_mcdropout = _module
mlp_regression_mcdropout = _module
nlp_bert_mcdropout = _module
active_image_classification = _module
lightning_flash_example = _module
segmentation = _module
unet_mcdropout_pascal = _module
utils = _module
pimodel_cifar10 = _module
pimodel_mcdropout_cifar10 = _module
vgg_mcdropout_cifar10 = _module
tests = _module
active_loop_test = _module
dataset_test = _module
nlp_dataset_test = _module
test_numpy_dataset = _module
file_dataset_test = _module
heuristic_test = _module
heuristics_gpu_test = _module
stochastic_heuristic_test = _module
common_test = _module
consistent_dropout_test = _module
dropconnect_test = _module
dropout_test = _module
calibration_test = _module
conftest = _module
documentation_test = _module
ensemble_test = _module
integration_test = _module
test_mixin = _module
modelwrapper_test = _module
test_utils = _module
transformers_trainer_wrapper_test = _module
itertools_test = _module
metrics_test = _module
plotutils_test = _module
ssl_iterator_test = _module
ssl_module_test = _module
test_array_utils = _module
test_cuda_utils = _module
test_equality = _module
test_pytorch_lightning = _module
test_transforms = _module

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


import types


import warnings


from typing import Callable


import numpy as np


import torch.utils.data as torchdata


from typing import Union


from typing import List


from typing import Optional


from typing import Any


from typing import TYPE_CHECKING


from typing import Protocol


from sklearn.utils import check_random_state


from torch.utils import data as torchdata


import torch


from itertools import zip_longest


from typing import Tuple


from copy import deepcopy


from typing import Dict


from typing import Sequence


from typing import Mapping


from torch import Tensor


import random


from torch.utils.data import Dataset


from collections.abc import Sequence


from functools import wraps as _wraps


import scipy.stats


from scipy.special import xlogy


import torch.nn.functional as F


import copy


from torch import nn


from torch.nn import functional as F


from torch.nn.modules.dropout import _DropoutNd


from typing import cast


from torch.optim import Adam


from typing import OrderedDict


from collections import defaultdict


from torch.optim import Optimizer


from torch.utils.data import DataLoader


from torch.utils.data.dataloader import default_collate


from scipy.special import softmax


from scipy.special import expit


from collections.abc import Mapping


from functools import singledispatch


import math


from sklearn.metrics import confusion_matrix


from sklearn.metrics import auc


from itertools import cycle


import torch.cuda


from torch import optim


from torchvision import transforms


from torchvision.datasets import MNIST


import pandas as pd


from sklearn.preprocessing import StandardScaler


from sklearn.model_selection import train_test_split


import torch.backends


from torch.nn import CrossEntropyLoss


from torchvision.datasets import CIFAR10


from torchvision.models import vgg16


from torchvision.transforms import transforms


from torchvision import datasets


from functools import partial


from torch.hub import load_state_dict_from_url


from torchvision.models import vgg11


import torch.multiprocessing


from torchvision.transforms import Lambda


import time


from torchvision.transforms import Resize


from torchvision.transforms import RandomRotation


from torchvision.transforms import Compose


from torchvision.transforms import ToTensor


from torchvision.transforms import ToPILImage


from torchvision.models import vgg


from torch.utils.data import ConcatDataset


from collections import namedtuple


from collections import OrderedDict


class BayesianModule(torch.nn.Module):
    patching_function: Callable[..., torch.nn.Module]
    unpatch_function: Callable[..., torch.nn.Module]

    def __init__(self, module, *args, **kwargs):
        super().__init__()
        self.parent_module = self.__class__.patching_function(module, *args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.parent_module(*args, **kwargs)

    def unpatch(self) ->torch.nn.Module:
        return self.__class__.unpatch_function(self.parent_module)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unpatch()


class ConsistentDropout(_DropoutNd):
    """
    ConsistentDropout is useful when doing research.
    It guarantees that while the masks are the same between batches
    during inference. The masks are different inside the batch.

    This is slower than using regular Dropout, but it is useful
    when you want to use the same set of weights for each sample used in inference.

    From BatchBALD (Kirsch et al, 2019), this is necessary to use BatchBALD and remove noise
    from the prediction.

    Args:
        p (float): probability of an element to be zeroed. Default: 0.5

    Notes:
        For optimal results, you should use a batch size of one
        during inference time.
        Furthermore, to guarantee that each sample uses the same
        set of weights,
        you must use `replicate_in_memory=True` in ModelWrapper,
        which is the default.
    """

    def __init__(self, p=0.5):
        super().__init__(p=p, inplace=False)
        self.reset()

    def forward(self, x):
        if self.training:
            return F.dropout(x, self.p, training=True, inplace=False)
        else:
            if self._mask is None or self._mask.shape != x.shape:
                self._mask = self._make_mask(x)
            return torch.mul(x, self._mask)

    def _make_mask(self, x):
        return F.dropout(torch.ones_like(x, device=x.device), self.p, training=True)

    def reset(self):
        self._mask = None

    def eval(self):
        self.reset()
        return super().eval()


class ConsistentDropout2d(_DropoutNd):
    """
    ConsistentDropout is useful when doing research.
    It guarantees that while the mask are the same between batches,
    they are different inside the batch.

    This is slower than using regular Dropout, but it is useful
    when you want to use the same set of weights for each unlabelled sample.

    Args:
        p (float): probability of an element to be zeroed. Default: 0.5

    Notes:
        For optimal results, you should use a batch size of one
        during inference time.
        Furthermore, to guarantee that each sample uses the same
        set of weights,
        you must use `replicate_in_memory=True` in ModelWrapper,
        which is the default.
    """

    def __init__(self, p=0.5):
        super().__init__(p=p, inplace=False)
        self.reset()

    def forward(self, x):
        if self.training:
            return F.dropout2d(x, self.p, training=True, inplace=False)
        else:
            if self._mask is None or self._mask.shape != x.shape:
                self._mask = self._make_mask(x)
            return torch.mul(x, self._mask)

    def _make_mask(self, x):
        return F.dropout2d(torch.ones_like(x, device=x.device), self.p, training=True)

    def reset(self):
        self._mask = None

    def eval(self):
        self.reset()
        return super().eval()


class WeightDropMixin:
    _kwargs: Dict

    def unpatch(self):
        new_module = self.__class__.__bases__[0](**self._kwargs)
        new_module.load_state_dict(self.state_dict())
        return new_module


class WeightDropConv2d(torch.nn.Conv2d, WeightDropMixin):
    """
    Reimplemmentation of WeightDrop for Conv2D. Thanks to PytorchNLP for the initial implementation
    of class WeightDropLinear. Their `License
    <https://github.com/PetrochukM/PyTorch-NLP/blob/master/LICENSE>`__.
    Wrapper around :class: 'torch.nn.Conv' that adds '' weight_dropout '' named argument.

    Args:
        weight_dropout (float): The probability a weight will be dropped.
    """

    def __init__(self, weight_dropout=0.0, **kwargs):
        wanted = ['in_channels', 'out_channels', 'kernel_size', 'dilation', 'padding']
        self._kwargs = {k: v for k, v in kwargs.items() if k in wanted}
        super().__init__(**self._kwargs)
        self._weight_dropout = weight_dropout
        self._torch_version = parse_version(torch.__version__)

    def forward(self, input):
        kwargs = {'input': input, 'weight': torch.nn.functional.dropout(self.weight, p=self._weight_dropout, training=True)}
        if self._torch_version >= parse_version('1.8.0'):
            kwargs['bias'] = self.bias
        return self._conv_forward(**kwargs)


class WeightDropLinear(torch.nn.Linear, WeightDropMixin):
    """
    Thanks to PytorchNLP for the initial implementation
    # code from https://pytorchnlp.readthedocs.io/en/latest/_modules/torchnlp/nn/weight_drop.html
    of class WeightDropLinear. Their `License
    <https://github.com/PetrochukM/PyTorch-NLP/blob/master/LICENSE>`__.
    Wrapper around :class:`torch.nn.Linear` that adds ``weight_dropout`` named argument.

    Args:
        weight_dropout (float): The probability a weight will be dropped.
    """

    def __init__(self, weight_dropout=0.0, **kwargs):
        wanted = ['in_features', 'out_features']
        self._kwargs = {k: v for k, v in kwargs.items() if k in wanted}
        super().__init__(**self._kwargs)
        self._weight_dropout = weight_dropout

    def forward(self, input):
        w = torch.nn.functional.dropout(self.weight, p=self._weight_dropout, training=True)
        return torch.nn.functional.linear(input, w, self.bias)


def get_weight_drop_module(name: str, weight_dropout, **kwargs):
    return {'Conv2d': WeightDropConv2d, 'Linear': WeightDropLinear}[name](weight_dropout, **kwargs)


def _dropconnect_mapping_fn(module: torch.nn.Module, layers, weight_dropout) ->Optional[nn.Module]:
    new_module: Optional[nn.Module] = None
    for layer in layers:
        if isinstance(module, getattr(torch.nn, layer)):
            new_module = get_weight_drop_module(layer, weight_dropout, **module.__dict__)
            break
    if isinstance(module, nn.Dropout):
        module._baal_p: float = module.p
        module.p = 0.0
    return new_module


def replace_layers_in_module(module: nn.Module, mapping_fn: Callable, *args, **kwargs) ->bool:
    """
    Recursively iterate over the children of a module and replace them according to `mapping_fn`.

    Returns:
        True if a layer has been changed.
    """
    changed = False
    for name, child in module.named_children():
        new_module = mapping_fn(child, *args, **kwargs)
        if new_module is not None:
            changed = True
            module.add_module(name, new_module)
        changed |= replace_layers_in_module(child, mapping_fn, *args, **kwargs)
    return changed


def _patching_wrapper(module: nn.Module, inplace: bool, patching_fn: Callable[..., Optional[nn.Module]], *args, **kwargs) ->nn.Module:
    if not inplace:
        module = copy.deepcopy(module)
    changed = replace_layers_in_module(module, patching_fn, *args, **kwargs)
    if not changed:
        warnings.warn('No layer was modified by patch_module!', UserWarning)
    return module


def patch_module(module: torch.nn.Module, layers: Sequence, weight_dropout: float=0.0, inplace: bool=True) ->torch.nn.Module:
    """
    Replace given layers with weight_drop module of that layer.

    Args:
        module : torch.nn.Module
            The module in which you would like to replace dropout layers.
        layers : list[str]
            Name of layers to be replaced from ['Conv', 'Linear', 'LSTM', 'GRU'].
        weight_dropout (float): The probability a weight will be dropped.
        inplace : bool, optional
            Whether to modify the module in place or return a copy of the module.

    Returns:
        torch.nn.Module:
            The modified module, which is either the same object as you passed in
            (if inplace = True) or a copy of that object.
    """
    return _patching_wrapper(module, inplace=inplace, patching_fn=_dropconnect_mapping_fn, layers=layers, weight_dropout=weight_dropout)


def _droconnect_unmapping_fn(module: torch.nn.Module) ->Optional[nn.Module]:
    new_module: Optional[nn.Module] = None
    if isinstance(module, WeightDropMixin):
        new_module = module.unpatch()
    if isinstance(module, nn.Dropout):
        module.p = module._baal_p
    return new_module


def unpatch_module(module: torch.nn.Module, inplace: bool=True) ->torch.nn.Module:
    """
    Unpatch Dropconnect module to recover initial module.

    Args:
        module (torch.nn.Module):
            The module in which you would like to replace dropout layers.
        inplace (bool, optional):
            Whether to modify the module in place or return a copy of the module.

    Returns:
        torch.nn.Module
            The modified module, which is either the same object as you passed in
            (if inplace = True) or a copy of that object.
    """
    return _patching_wrapper(module, inplace=inplace, patching_fn=_droconnect_unmapping_fn)


class MCConsistentDropoutModule(BayesianModule):
    patching_function = patch_module
    unpatch_function = unpatch_module


class Dropout(_DropoutNd):
    """Randomly zeroes some of the elements of the input
    tensor with probability :attr:`p` using samples from a Bernoulli
    distribution. Each channel will be zeroed out independently on every forward
    call.
    This has proven to be an effective technique for regularization and
    preventing the co-adaptation of neurons as described in the paper
    `Improving neural networks by preventing co-adaptation of feature
    detectors`_ .
    Furthermore, the outputs are scaled by a factor of :math:`\\frac{1}{1-p}` during
    training.
    Shape:
        - Input: :math:`(*)`. Input can be of any shape.
        - Output: :math:`(*)`. Output is of the same shape as input.

    Args:
        p (float, optional):
            Probability of an element to be zeroed. Default: 0.5
        inplace (bool, optional):
            If set to ``True``, will do this operation in-place. Default: ``False``

    Examples::
        >>> m = nn.Dropout(p=0.2)
        >>> input = torch.randn(20, 16)
        >>> output = m(input)
    .. _Improving neural networks by preventing co-adaptation of feature
        detectors: https://arxiv.org/abs/1207.0580
    """

    def forward(self, input):
        return F.dropout(input, self.p, True, self.inplace)


class Dropout2d(_DropoutNd):
    """Randomly zero out entire channels (a channel is a 2D feature map,
    e.g., the :math:`j`-th channel of the :math:`i`-th sample in the
    batched input is a 2D tensor :math:`\\text{input}[i, j]`) of the input tensor).
    Each channel will be zeroed out independently on every forward call.
    with probability :attr:`p` using samples from a Bernoulli distribution.

    Usually the input comes from :class:`nn.Conv2d` modules.

    As described in the paper
    `Efficient Object Localization Using Convolutional Networks`_ ,
    if adjacent pixels within feature maps are strongly correlated
    (as is normally the case in early convolution layers) then i.i.d. dropout
    will not regularize the activations and will otherwise just result
    in an effective learning rate decrease.

    In this case, :func:`nn.Dropout2d` will help promote independence between
    feature maps and should be used instead.
    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)

    Args:
        p (float, optional):
            Probability of an element to be zero-ed.
        inplace (bool, optional):
            If set to ``True``, will do this operation in-place.

    Examples::

        >>> m = nn.Dropout2d(p=0.2)
        >>> input = torch.randn(20, 16, 32, 32)
        >>> output = m(input)

    .. _Efficient Object Localization Using Convolutional Networks:
       http://arxiv.org/abs/1411.4280

    """

    def forward(self, input):
        return F.dropout2d(input, self.p, True, self.inplace)


class MCDropoutModule(BayesianModule):
    """Create a module that with all dropout layers patched.

    Args:
        module (torch.nn.Module):
            A fully specified neural network.
    """
    patching_function = patch_module
    unpatch_function = unpatch_module


class MCDropoutConnectModule(BayesianModule):
    """Create a module that with all dropout layers patched.
    With MCDropoutConnectModule, it could be decided which type of modules to be
    replaced.

    Args:
        module (torch.nn.Module):
            A fully specified neural network.
        layers (list[str]):
            Name of layers to be replaced from ['Conv', 'Linear', 'LSTM', 'GRU'].
        weight_dropout (float): The probability a weight will be dropped.
    """
    patching_function = patch_module
    unpatch_function = unpatch_module


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        dropout = nn.Dropout2d(0.5)
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        activation = Activation(activation)
        super().__init__(dropout, conv2d, upsampling, activation)


class FocalLoss(nn.Module):
    """
    References:
        Author: clcarwin
        Site https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
    """

    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)
            input = input.contiguous().view(-1, input.size(2))
        target = target.view(-1, 1)
        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.data.exp()
        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            select = (target != 0).type(torch.LongTensor)
            at = self.alpha.gather(0, select.data.view(-1))
            logpt = logpt * at
        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class GaussianNoise(nn.Module):
    """Add random gaussian noise to images."""

    def __init__(self, std=0.05):
        super(GaussianNoise, self).__init__()
        self.std = std

    def forward(self, x):
        return x + torch.randn(x.size()).type_as(x) * self.std


class RandomTranslation(nn.Module):
    """Randomly translate images."""

    def __init__(self, augment_translation=10):
        super(RandomTranslation, self).__init__()
        self.augment_translation = augment_translation

    def forward(self, x):
        """
            Randomly translate images.
        Args:
            x (Tensor) : (N, C, H, W) image tensor

        Returns:
            (N, C, H, W) translated image tensor
        """
        batch_size = len(x)
        t_min = -self.augment_translation / x.shape[-1]
        t_max = (self.augment_translation + 1) / x.shape[-1]
        matrix = torch.eye(3)[None].repeat((batch_size, 1, 1))
        tx = (t_min - t_max) * torch.rand(batch_size) + t_max
        ty = (t_min - t_max) * torch.rand(batch_size) + t_max
        matrix[:, 0, 2] = tx
        matrix[:, 1, 2] = ty
        matrix = matrix[:, 0:2, :]
        grid = nn.functional.affine_grid(matrix, x.shape).type_as(x)
        x = nn.functional.grid_sample(x, grid)
        return x


class Flatten(nn.Module):

    def forward(self, x):
        return x.view([x.shape[0], -1])


class SimpleModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.linear = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = x.view(x.shape[0], -1)
        x = self.dropout(x)
        x = self.linear(x)
        x = self.sigmoid(x)
        return x


class DummyModel(nn.Module):

    def __init__(self):
        super(DummyModel, self).__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.linear = nn.Linear(8, 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = x.view(x.shape[0], -1)
        x = self.dropout(x)
        x = self.linear(x)
        return x


N_CLASS = 3


class AModel(nn.Module):


    class Flatten(nn.Module):

        def forward(self, input):
            return input.view(input.size(0), -1)

    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(nn.Conv2d(3, 5, 5), nn.AdaptiveAvgPool2d(5), self.Flatten(), nn.Linear(125, N_CLASS))

    def forward(self, x):
        return self.seq(x)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AModel,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (ConsistentDropout,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConsistentDropout2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Dropout,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Dropout2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Flatten,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FocalLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.ones([4], dtype=torch.int64)], {}),
     True),
    (GaussianNoise,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MCConsistentDropoutModule,
     lambda: ([], {'module': _mock_layer(), 'layers': 1}),
     lambda: ([], {'input': torch.rand([4, 4])}),
     False),
    (MCDropoutConnectModule,
     lambda: ([], {'module': _mock_layer(), 'layers': 1}),
     lambda: ([], {'input': torch.rand([4, 4])}),
     False),
    (MCDropoutModule,
     lambda: ([], {'module': _mock_layer(), 'layers': 1}),
     lambda: ([], {'input': torch.rand([4, 4])}),
     False),
    (RandomTranslation,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (WeightDropLinear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_baal_org_baal(_paritybench_base):
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

