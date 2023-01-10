import sys
_module = sys.modules[__name__]
del sys
conf = _module
examples = _module
bernoulli_grad_var = _module
bernoulli_toy = _module
data_loader = _module
fixed_mnist = _module
introduction = _module
vae = _module
bernoulli_vae = _module
discrete_vae = _module
normal_vae = _module
train = _module
vae = _module
vae_reference = _module
setup = _module
storch = _module
exceptions = _module
excluded_init = _module
inference = _module
method = _module
arm = _module
baseline = _module
method = _module
multi_sample_reinforce = _module
rao_blackwell = _module
relax = _module
unordered = _module
nn = _module
losses = _module
sampling = _module
expect = _module
importance_sampling = _module
mapo = _module
method = _module
seq = _module
swor = _module
unordered_set = _module
storch = _module
tensor = _module
typing = _module
unique = _module
util = _module
wrappers = _module
test = _module
collect_env = _module
is_unbiased = _module
pyro_enum = _module
test = _module
test2 = _module
test_allowed = _module
test_broadcast_all = _module
test_swor = _module
test_tensor = _module
testancestral = _module
testcat = _module
testmanycosts = _module
testrelax = _module

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


from torch.distributions import Bernoulli


from torchvision import transforms


from torchvision import datasets


import torch.utils.data as data


import numpy as np


from torch.distributions import Normal


import torch.nn as nn


from torch.distributions import Distribution


from torch.distributions.utils import clamp_probs


from torch.distributions import OneHotCategorical


from typing import List


from typing import Type


import torch.utils.data


from torch import optim


from torchvision.utils import save_image


from torch.utils.tensorboard import SummaryWriter


from torch import nn


from torch.nn import functional as F


import torch as _torch


from typing import Optional


from typing import Union


from typing import Tuple


from torch.distributions import TransformedDistribution


from torch.distributions import Uniform


from torch.distributions import SigmoidTransform


from torch.distributions import AffineTransform


from abc import ABC


from abc import abstractmethod


from torch.distributions import Categorical


from typing import Dict


from typing import Callable


import warnings


from functools import reduce


from torch.nn import Parameter


import torch.nn.functional as F


import itertools


from typing import Iterable


from torch.distributions import Gumbel


from queue import Queue


from collections import deque


from typing import Any


from typing import Iterator


from typing import Deque


from itertools import product


from torch import Size


from torch.distributions import RelaxedOneHotCategorical


from torch.distributions import RelaxedBernoulli


from copy import copy


from collections.abc import Iterable


from collections.abc import Mapping


from functools import wraps


import re


from collections import namedtuple


from torch.distributions import Poisson


import torch.distributions as td


def is_iterable(a: Any):
    return isinstance(a, Iterable) and not storch.is_tensor(a) and not isinstance(a, str) and not isinstance(a, torch.Storage)


def _handle_deterministic(fn, fn_args, fn_kwargs, reduce_plates: Optional[Union[str, List[str]]]=None, flatten_plates: bool=False, **wrapper_kwargs):
    if storch.wrappers._context_stochastic:
        raise NotImplementedError('It is currently not allowed to open a deterministic context in a stochastic context')
    new_fn_args, new_fn_kwargs, parents, plates = _prepare_args(fn_args, fn_kwargs, flatten_plates=flatten_plates, **wrapper_kwargs)
    if not parents:
        return fn(*fn_args, **fn_kwargs)
    args = new_fn_args
    kwargs = new_fn_kwargs
    storch.wrappers._context_deterministic += 1
    try:
        outputs = fn(*args, **kwargs)
    finally:
        storch.wrappers._context_deterministic -= 1
    if storch.wrappers._ignore_wrap:
        return outputs
    if reduce_plates:
        if isinstance(reduce_plates, str):
            reduce_plates = [reduce_plates]
        plates = [p for p in plates if p.name not in reduce_plates]
    outputs = _prepare_outputs_det(outputs, parents, plates, fn.__name__, 1, unflatten_plates=flatten_plates)[0]
    return outputs


def _deterministic(fn, reduce_plates: Optional[Union[str, List[str]]]=None, **wrapper_kwargs):

    @wraps(fn)
    def wrapper(*args, **kwargs):
        nonlocal reduce_plates
        return _handle_deterministic(fn, args, kwargs, reduce_plates, **wrapper_kwargs)
    return wrapper


def deterministic(fn: Optional[Callable]=None, **kwargs):
    """
    Wraps the input function around a deterministic storch wrapper.
    This wrapper unwraps :class:`~storch.Tensor` objects to :class:`~torch.Tensor` objects, aligning the tensors
    according to the plates, then runs `fn` on the unwrapped Tensors.

    Args:
        fn: Optional function to wrap. If None, this returns another wrapper that accepts a function that will be instantiated
        by the given kwargs.
        unwrap: Set to False to prevent unwrapping :class:`~storch.Tensor` objects.
        fn_args: List of non-keyword arguments to the wrapped function
        fn_kwargs: Dictionary of keyword arguments to the wrapped function
        unwrap: Whether to unwrap the arguments to their torch.Tensor counterpart (default: True)
        align_tensors: Whether to automatically align the input arguments (default: True)
        l_broadcast: Whether to automatically left-broadcast (default: True)
        expand_plates: Instead of adding singleton dimensions on non-existent plates, this will
        add the plate size itself (default: False) flatten_plates sets this to True automatically.
        flatten_plates: Flattens the plate dimensions into a single batch dimension if set to true.
        This can be useful for functions that are written to only work for tensors with a single batch dimension.
        Note that outputs are unflattened automatically. (default: False)
        dim: Replaces the dim input in fn_kwargs by the plate dimension corresponding to the given string (optional)
        dims: Replaces the dims input in fn_kwargs by the plate dimensions corresponding to the given strings (optional)
        self_wrapper: storch.Tensor that wraps a
    Returns:
        Callable: The wrapped function `fn`.
    """
    if fn:
        return _deterministic(fn, **kwargs)
    return lambda _f: _deterministic(_f, **kwargs)


class DiscreteVAE(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 2 * 10)
        self.fc4 = nn.Linear(2 * 10, 256)
        self.fc5 = nn.Linear(256, 512)
        self.fc6 = nn.Linear(512, 784)

    def encode(self, x):
        h1 = self.fc1(x).relu()
        h2 = self.fc2(h1).relu()
        return self.fc3(h2)

    def decode(self, z):
        h3 = self.fc4(z).relu()
        h4 = self.fc5(h3).relu()
        return self.fc6(h4).sigmoid()


_size = Union[Size, List[int], Tuple[int, ...]]


Dims = Union[int, _size]


class Baseline(torch.nn.Module):

    def __init__(self, in_dim: Dims):
        super().__init__()
        self.reshape = False
        if not isinstance(in_dim, int):
            self.reshape = True
            in_dim = reduce(mul, in_dim)
        self.in_dim = in_dim
        self.fc1 = torch.nn.Linear(in_dim, 50)
        self.fc2 = torch.nn.Linear(50, 1)

    def forward(self, x: storch.Tensor):
        if self.reshape or self.in_dim <= 1:
            x = x.reshape(x.shape[:x.plate_dims] + (-1,))
        return self.fc2(F.relu(self.fc1(x))).squeeze(-1)


is_tensor = lambda a: isinstance(a, torch.Tensor) or isinstance(a, Tensor)


def get_distr_parameters(d: Distribution, filter_requires_grad=True) ->Dict[str, torch.Tensor]:
    params = {}
    while d:
        for k in d.arg_constraints:
            try:
                p = getattr(d, k)
                if is_tensor(p) and (not filter_requires_grad or p.requires_grad):
                    params[k] = p
            except AttributeError:
                if _debug:
                    None
                pass
        if hasattr(d, 'base_dist'):
            d = d.base_dist
        else:
            d = None
    return params


def has_differentiable_path(output: Tensor, input: Tensor):
    for c in input.walk_children(only_differentiable=True):
        if c is output:
            return True
    return False


def rsample_gumbel_softmax(distr: Distribution, n: int, temperature: torch.Tensor, straight_through: bool=False) ->torch.Tensor:
    if isinstance(distr, (Categorical, OneHotCategorical)):
        if straight_through:
            gumbel_distr = RelaxedOneHotCategoricalStraightThrough(temperature, probs=distr.probs)
        else:
            gumbel_distr = RelaxedOneHotCategorical(temperature, probs=distr.probs)
    elif isinstance(distr, Bernoulli):
        if straight_through:
            gumbel_distr = RelaxedBernoulliStraightThrough(temperature, probs=distr.probs)
        else:
            gumbel_distr = RelaxedBernoulli(temperature, probs=distr.probs)
    else:
        raise ValueError('Using Gumbel Softmax with non-discrete distribution')
    return gumbel_distr.rsample((n,))


def magic_box(l: Tensor):
    """
    Implements the MagicBox operator from
    DiCE: The Infinitely Differentiable Monte-Carlo Estimator https://arxiv.org/abs/1802.05098
    It returns 1 in the forward pass, but returns magic_box(l) \\cdot r in the backwards pass.
    This allows for any-order gradient estimation.
    """
    return torch.exp(l - l.detach())


def rsample_gumbel(distr: Distribution, n: int) ->torch.Tensor:
    gumbel_distr = Gumbel(distr.logits, 1)
    return gumbel_distr.rsample((n,))


@storch.deterministic
def conditional_gumbel_rsample(hard_sample: torch.Tensor, probs: torch.Tensor, bernoulli: bool, temperature) ->torch.Tensor:
    """
    Conditionally re-samples from the distribution given the hard sample.
    This samples z \\sim p(z|b), where b is the hard sample and p(z) is a gumbel distribution.
    """
    shape = hard_sample.shape
    probs = clamp_probs(probs.expand_as(hard_sample))
    v = clamp_probs(torch.rand(shape, dtype=probs.dtype, device=probs.device))
    if bernoulli:
        pos_probs = probs[hard_sample == 1]
        v_prime = torch.zeros_like(hard_sample)
        v_prime[hard_sample == 1] = v[hard_sample == 1] * pos_probs + (1 - pos_probs)
        v_prime[hard_sample == 0] = v[hard_sample == 0] * (1 - probs[hard_sample == 0])
        log_sample = (probs.log() + probs.log1p() + v_prime.log() + v_prime.log1p()) / temperature
        return log_sample.sigmoid()
    b = hard_sample.max(-1).indices
    log_v = v.log()
    log_v_b = torch.gather(log_v, -1, b.unsqueeze(-1))
    cond_gumbels = -(-torch.div(log_v, probs) - log_v_b).log()
    index_sample = hard_sample.bool()
    cond_gumbels[index_sample] = -(-log_v[index_sample]).log()
    scores = cond_gumbels / temperature
    return (scores - scores.logsumexp(dim=-1, keepdim=True)).exp()


def discretize(tensor: torch.Tensor, distr: Distribution) ->torch.Tensor:
    if isinstance(distr, Bernoulli):
        return tensor.round()
    argmax = tensor.max(-1)[1]
    hard_sample = torch.zeros_like(tensor)
    if argmax.dim() < hard_sample.dim():
        argmax = argmax.unsqueeze(-1)
    return hard_sample.scatter_(-1, argmax, 1)


def expand_with_ignore_as(tensor, expand_as, ignore_dim: Union[str, int]) ->torch.Tensor:
    """
    Expands the tensor like expand_as, but ignores a single dimension.
    Ie, if tensor is of size a x b,  expand_as of size d x a x c and dim=-1, then the return will be of size d x a x b.
    It also automatically expands all plate dimensions correctly.
    :param ignore_dim: Can be a string referring to the plate dimension
    """

    def _expand_with_ignore(tensor, expand_as, dim: int):
        new_dims = expand_as.ndim - tensor.ndim
        return tensor[(...,) + (None,) * new_dims].expand(expand_as.shape[:dim] + (-1,) + (expand_as.shape[dim + 1:] if dim != -1 else ()))
    if isinstance(ignore_dim, str):
        return storch.deterministic(_expand_with_ignore, expand_plates=True, dim=ignore_dim)(tensor, expand_as)
    return storch.deterministic(_expand_with_ignore, expand_plates=True)(tensor, expand_as, ignore_dim)


AnyTensor = Union[storch.Tensor, torch.Tensor]


def left_expand_as(tensor, expand_as):
    diff = expand_as.ndim - tensor.ndim
    return tensor[(None,) * diff].expand(expand_as.shape[:diff] + (-1,) * tensor.ndim)


@storch.deterministic(l_broadcast=False)
def right_expand_as(tensor, expand_as):
    diff = expand_as.ndim - tensor.ndim
    return tensor[(...,) + (None,) * diff].expand((-1,) * tensor.ndim + expand_as.shape[tensor.ndim:])


def log1mexp(a: torch.Tensor) ->torch.Tensor:
    """See appendix A of http://jmlr.org/papers/v21/19-985.html.
    Numerically stable implementation of log(1-exp(a))"""
    c = -0.693
    a1 = -a.abs()
    eps = 1e-06
    return torch.where(a1 > c, torch.log(-a1.expm1() + eps), torch.log1p(-a1.exp() + eps))


@storch.deterministic
def cond_gumbel_sample(all_joint_log_probs, perturbed_log_probs) ->torch.Tensor:
    gumbel_d = Gumbel(loc=all_joint_log_probs, scale=1.0)
    G_yv = gumbel_d.rsample()
    Z = G_yv.max(dim=-1)[0]
    T = perturbed_log_probs
    vi = T - G_yv + log1mexp(G_yv - Z.unsqueeze(-1))
    return T - vi.relu() - torch.nn.Softplus()(-vi.abs())


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Baseline,
     lambda: ([], {'in_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_HEmile_storchastic(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

