import sys
_module = sys.modules[__name__]
del sys
cplxmodule = _module
cplx = _module
nn = _module
init = _module
masked = _module
base = _module
complex = _module
real = _module
modules = _module
activation = _module
base = _module
batchnorm = _module
casting = _module
container = _module
conv = _module
extra = _module
linear = _module
pooling = _module
relevance = _module
base = _module
ard = _module
base = _module
vd = _module
extensions = _module
complex = _module
ell_zero = _module
lasso = _module
ard = _module
base = _module
vd = _module
utils = _module
sparsity = _module
spectrum = _module
views = _module
setup = _module
test_batchnorm = _module
test_cplx = _module
test_cplxparameter = _module
test_draw_welch = _module
test_init = _module
test_masked = _module
test_mnist = _module
test_modules = _module
test_onnx = _module
test_relevance = _module
test_spectrum = _module
test_utils = _module

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


from copy import deepcopy


import torch


import torch.nn.functional as F


from math import sqrt


import math


import numpy as np


from torch.nn import init


from functools import wraps


from torch.nn import Linear


from torch.nn import Conv1d


from torch.nn import Conv2d


from torch.nn import Conv3d


from torch.nn import Bilinear


import torch.nn


from functools import lru_cache


from collections import OrderedDict


from torch.nn.modules.utils import _single


from torch.nn.modules.utils import _pair


from torch.nn.modules.utils import _triple


import scipy


import scipy.special


from numpy import euler_gamma


import warnings


from itertools import starmap


import copy


import matplotlib.pyplot as plt


from scipy.signal import welch


from matplotlib.ticker import EngFormatter


from torchvision import datasets


from torchvision import transforms


import torch.sparse


from numpy.testing import assert_allclose


class BaseMasked(torch.nn.Module):
    """The base class for linear layers that should have fixed sparsity
    pattern.

    Attributes
    ----------
    is_sparse : bool, read-only
        Indicates if the instance has a valid usable mask.

    mask : torch.Tensor
        The current mask used for the weights. Always guaranteed to have
        the same shape, dtype (float, double) and be on the same device
        as `.weight`.

    Details
    -------
    As of pytorch 1.1.0 there is no mechanism to preallocate runtime buffers
    before loading state_dict. Thus we use a custom `__init__` and 'piggy-back'
    on the documented, but private method `Module._load_from_state_dict` to
    conditionally allocate or free mask buffers via `.mask_` method. This API
    places a restriction on the order of bases classes when subclassing. In
    order for `super().__init__` in subclasses to do the necessary mask setting
    up and initialize the `torch.nn.Module` itself, the `BaseMasked` should be
    placed as far to the right in base class list, but before `torch.nn.Module`.

    The masks could be set either manually through setting `.mask`, or loaded in
    bulk with `deploy_masks`, or via `model.load_state_dict(..., strict=False)`.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer('mask', None)

    @property
    def is_sparse(self):
        """Check if the layer is in sparse mode."""
        return isinstance(self.mask, torch.Tensor)

    def mask_(self, mask):
        """Update or reset the mask to a new one, broadcasting it if necessary.

        Arguments
        ---------
        mask : torch.Tensor or None
            The mask to be used. Device migration, dtype conversion and
            broadcasting are done automatically to conform to `.weight`.

        Details
        -------
        Effectively switches on / off masking of the weights.
        """
        if mask is not None and not isinstance(mask, torch.Tensor):
            raise TypeError(f'`mask` must be either a Tensor or `None`. Got {type(mask).__name__}.')
        if mask is not None:
            mask = mask.detach()
            mask = mask.expand(self.weight.shape).contiguous()
            self.register_buffer('mask', mask)
        elif self.is_sparse and mask is None:
            del self.mask
            self.register_buffer('mask', None)
        elif not self.is_sparse and mask is None:
            pass
        return self

    def __setattr__(self, name, value):
        """Special routing syntax like `.require_grad = ...`."""
        if name != 'mask':
            return super().__setattr__(name, value)
        self.mask_(value)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """Surgically load the state with the runtime masks from a dict."""
        mask = prefix + 'mask'
        super()._load_from_state_dict({k: v for k, v in state_dict.items() if k != mask}, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        mask_in_missing = mask in missing_keys
        if mask in state_dict:
            if mask_in_missing:
                missing_keys.remove(mask)
            self.mask_(state_dict[mask])
        elif strict:
            if not mask_in_missing:
                missing_keys.append(mask)
        elif mask_in_missing:
            missing_keys.remove(mask)


class MaskedWeightMixin:
    """A mixin for accessing read-only masked weight,"""

    @property
    def weight_masked(self):
        """Return a sparsified weight of the parent *Linear."""
        if not self.is_sparse:
            msg = f'`{type(self).__name__}` has no sparsity mask. Please, either set a mask attribute, or call `deploy_masks()`.'
            raise RuntimeError(msg)
        return self.weight * self.mask


class SparsityStats(object):
    __sparsity_ignore__ = ()

    def sparsity(self, **kwargs):
        raise NotImplementedError('Derived classes must implement a method to estimate sparsity.')


class _BaseCplxMixin(MaskedWeightMixin, BaseMasked, SparsityStats):
    __sparsity_ignore__ = 'mask',

    def sparsity(self, *, hard=True, **kwargs):
        weight = self.weight
        if self.is_sparse:
            mask = torch.gt(self.mask, 0) if hard else self.mask
            n_dropped = float(weight.real.numel())
            n_dropped -= float(mask.sum().item())
        else:
            n_dropped = 0.0
        return [(id(weight.real), n_dropped), (id(weight.imag), n_dropped)]


class Cplx(object):
    """A type partially implementing complex valued tensors in torch.

    Details
    -------
    Creates a complex tensor object from the real and imaginary torch tensors,
    or pythonic floats and complex numbers. This is a container-wrapper which
    does not copy the supplied torch tensors on creation.
    """
    __slots__ = '__real', '__imag'

    def __new__(cls, real, imag=None):
        if isinstance(real, cls):
            return real
        if isinstance(real, complex):
            real, imag = torch.tensor(real.real), torch.tensor(real.imag)
        elif isinstance(real, float):
            if imag is None:
                imag = 0.0
            elif not isinstance(imag, float):
                raise TypeError('Imaginary part must be float.')
            real, imag = torch.tensor(real), torch.tensor(imag)
        elif not isinstance(real, torch.Tensor):
            raise TypeError('Real part must be torch.Tensor.')
        if imag is None:
            imag = torch.zeros_like(real)
        elif not isinstance(imag, torch.Tensor):
            raise TypeError('Imaginary part must be torch.Tensor.')
        if real.shape != imag.shape:
            raise ValueError('Real and imaginary parts have mistmatching shape.')
        self = super().__new__(cls)
        self.__real, self.__imag = real, imag
        return self

    def __copy__(self):
        """Shallow: a new instance with references to the real-imag data."""
        return type(self)(self.__real, self.__imag)

    def __deepcopy__(self, memo):
        """Deep: a new instance with copies of the real-imag data."""
        real = deepcopy(self.__real, memo)
        imag = deepcopy(self.__imag, memo)
        return type(self)(real, imag)

    @property
    def real(self):
        """Real part of the complex tensor."""
        return self.__real

    @property
    def imag(self):
        """Imaginary part of the complex tensor."""
        return self.__imag

    def __getitem__(self, key):
        """Index the complex tensor."""
        return type(self)(self.__real[key], self.__imag[key])

    def __setitem__(self, key, value):
        """Alter the complex tensor at index inplace."""
        if not isinstance(value, (Cplx, complex)):
            self.__real[key], self.__imag[key] = value, value
        else:
            self.__real[key], self.__imag[key] = value.real, value.imag

    def __iter__(self):
        """Iterate over the zero-th dimension of the complex tensor."""
        return map(type(self), self.__real, self.__imag)

    def __reversed__(self):
        """Reverse the complex tensor along the zero-th dimension."""
        return type(self)(reversed(self.__real), reversed(self.__imag))

    def clone(self):
        """Clone a complex tensor."""
        return type(self)(self.__real.clone(), self.__imag.clone())

    @property
    def conj(self):
        """The complex conjugate of the complex tensor."""
        return type(self)(self.__real, -self.__imag)

    def conjugate(self):
        """The complex conjugate of the complex tensor."""
        return self.conj

    def __pos__(self):
        """Return the complex tensor as is."""
        return self

    def __neg__(self):
        """Flip the sign of the complex tensor."""
        return type(self)(-self.__real, -self.__imag)

    def __add__(u, v):
        """Sum of complex tensors."""
        if not isinstance(v, (Cplx, complex)):
            return type(u)(u.__real + v, u.__imag)
        return type(u)(u.__real + v.real, u.__imag + v.imag)
    __radd__ = __add__
    __iadd__ = __add__

    def __sub__(u, v):
        """Difference of complex tensors."""
        if not isinstance(v, (Cplx, complex)):
            return type(u)(u.__real - v, u.__imag)
        return type(u)(u.__real - v.real, u.__imag - v.imag)

    def __rsub__(u, v):
        """Difference of complex tensors."""
        return -u + v
    __isub__ = __sub__

    def __mul__(u, v):
        """Elementwise product of complex tensors."""
        if not isinstance(v, (Cplx, complex)):
            return type(u)(u.__real * v, u.__imag * v)
        return type(u)(u.__real * v.real - u.__imag * v.imag, u.__imag * v.real + u.__real * v.imag)
    __rmul__ = __mul__
    __imul__ = __mul__

    def __truediv__(u, v):
        """Elementwise division of complex tensors."""
        if not isinstance(v, (Cplx, complex)):
            return type(u)(u.__real / v, u.__imag / v)
        denom = v.real * v.real + v.imag * v.imag
        return u * (v.conjugate() / denom)

    def __rtruediv__(u, v):
        """Elementwise division of something by a complex tensor."""
        denom = u.__real * u.__real + u.__imag * u.__imag
        return u.conjugate() / denom * v
    __itruediv__ = __truediv__

    def __matmul__(u, v):
        """Complex matrix-matrix product of complex tensors."""
        if not isinstance(v, Cplx):
            return type(u)(torch.matmul(u.__real, v), torch.matmul(u.__imag, v))
        re = torch.matmul(u.__real, v.__real) - torch.matmul(u.__imag, v.__imag)
        im = torch.matmul(u.__imag, v.__real) + torch.matmul(u.__real, v.__imag)
        return type(u)(re, im)

    def __rmatmul__(u, v):
        """Matrix multiplication by a complex tensor from the right."""
        return type(u)(torch.matmul(v, u.__real), torch.matmul(v, u.__imag))
    __imatmul__ = __matmul__

    def __abs__(self):
        """Compute the complex modulus:
        $$
            \\mathbb{C}^{\\ldots \\times d}
                \\to \\mathbb{R}_+^{\\ldots \\times d}
            \\colon u + i v \\mapsto \\lvert u + i v \\rvert
            \\,. $$
        """
        input = torch.stack([self.__real, self.__imag], dim=0)
        return torch.norm(input, p=2, dim=0, keepdim=False)

    @property
    def angle(self):
        """Compute the complex argument:
        $$
            \\mathbb{C}^{\\ldots \\times d}
                \\to \\mathbb{R}^{\\ldots \\times d}
            \\colon \\underbrace{u + i v}_{r e^{i\\phi}} \\mapsto \\phi
                    = \\arctan \\tfrac{v}{u}
            \\,. $$
        """
        return torch.atan2(self.__imag, self.__real)

    def apply(self, f, *a, **k):
        """Applies the function to real and imaginary parts."""
        return type(self)(f(self.__real, *a, **k), f(self.__imag, *a, **k))

    @property
    def shape(self):
        """Returns the shape of the complex tensor."""
        return self.__real.shape

    def __len__(self):
        """The size of the zero-th dimension of the complex tensor."""
        return self.shape[0]

    def t(self):
        """The transpose of a 2d compelx tensor."""
        return type(self)(self.__real.t(), self.__imag.t())

    def h(self):
        """The Hermitian transpose of a 2d compelx tensor."""
        return self.conj.t()

    def flatten(self, start_dim=0, end_dim=-1):
        return type(self)(self.__real.flatten(start_dim, end_dim), self.__imag.flatten(start_dim, end_dim))

    def view(self, *shape):
        """Return a view of the complex tensor."""
        shape = shape[0] if shape and isinstance(shape[0], tuple) else shape
        return type(self)(self.__real.view(*shape), self.__imag.view(*shape))

    def view_as(self, other):
        """Return a view of the complex tensor of shape other."""
        shape = other.shape
        return self.view(*shape)

    def reshape(self, *shape):
        """Reshape the complex tensor."""
        shape = shape[0] if shape and isinstance(shape[0], tuple) else shape
        return type(self)(self.__real.reshape(*shape), self.__imag.reshape(*shape))

    def size(self, *dim):
        """Returns the size of the complex tensor."""
        return self.__real.size(*dim)

    def squeeze(self, dim=None):
        """Returns the complex tensor with all the dimensions of input of
        size one removed.
        """
        if dim is None:
            return type(self)(self.__real.squeeze(), self.__imag.squeeze())
        else:
            return type(self)(self.__real.squeeze(dim=dim), self.__imag.squeeze(dim=dim))

    def unsqueeze(self, dim=None):
        """Returns a new complex tensor with a dimension of size one inserted
        at the specified position.
        """
        if dim is None:
            return type(self)(self.__real.unsqueeze(), self.__imag.unsqueeze())
        else:
            return type(self)(self.__real.unsqueeze(dim=dim), self.__imag.unsqueeze(dim=dim))

    def item(self):
        """The scalar value of zero-dim complex tensor."""
        return float(self.__real) + 1.0j * float(self.__imag)

    @classmethod
    def from_numpy(cls, numpy):
        """Create a complex tensor from numpy array."""
        re = torch.from_numpy(numpy.real)
        im = torch.from_numpy(numpy.imag)
        return cls(re, im)

    def numpy(self):
        """Export a complex tensor as complex numpy array."""
        return self.__real.numpy() + 1.0j * self.__imag.numpy()

    def __repr__(self):
        return f'{self.__class__.__name__}(\n  real={self.__real},\n  imag={self.__imag}\n)'

    def detach(self):
        """Return a copy of the complex tensor detached from autograd graph."""
        return type(self)(self.__real.detach(), self.__imag.detach())

    def requires_grad_(self, requires_grad=True):
        """Toggle the gradient of real and imaginary parts."""
        return type(self)(self.__real.requires_grad_(requires_grad), self.__imag.requires_grad_(requires_grad))

    @property
    def grad(self):
        """Collect the accumulated gradinet of the complex tensor."""
        re, im = self.__real.grad, self.__imag.grad
        return None if re is None or im is None else type(self)(re, im)

    def cuda(self, device=None, non_blocking=False):
        """Move the complex tensor to a CUDA device."""
        re = self.__real
        im = self.__imag
        return type(self)(re, im)

    def cpu(self):
        """Move the complex tensor to CPU."""
        return type(self)(self.__real.cpu(), self.__imag.cpu())

    def to(self, *args, **kwargs):
        """Move / typecast the complex tensor."""
        return type(self)(self.__real, self.__imag)

    @property
    def device(self):
        """The hosting device of the complex tensor."""
        return self.__real.device

    @property
    def dtype(self):
        """The base dtype of the complex tensor."""
        return self.__real.dtype

    def dim(self):
        """The number of dimensions in the complex tensor."""
        return len(self.shape)

    def permute(self, *dims):
        """Shuffle the dimensions of the complex tensor."""
        return type(self)(self.__real.permute(*dims), self.__imag.permute(*dims))

    def transpose(self, dim0, dim1):
        """Transpose the specified dimensions of the complex tensor."""
        return type(self)(self.__real.transpose(dim0, dim1), self.__imag.transpose(dim0, dim1))

    def is_complex(self):
        """Test if the tensor indeed represents a complex number."""
        return True

    @classmethod
    def empty(cls, *sizes, dtype=None, device=None, requires_grad=False):
        """Create an empty complex tensor."""
        re = torch.empty(*sizes, dtype=dtype, device=device, requires_grad=requires_grad)
        return cls(re, torch.empty_like(re, requires_grad=requires_grad))

    @classmethod
    def zeros(cls, *sizes, dtype=None, device=None, requires_grad=False):
        """Create an empty complex tensor."""
        re = torch.zeros(*sizes, dtype=dtype, device=device, requires_grad=requires_grad)
        return cls(re, torch.zeros_like(re, requires_grad=requires_grad))

    @classmethod
    def ones(cls, *sizes, dtype=None, device=None, requires_grad=False):
        """Create an empty complex tensor."""
        re = torch.ones(*sizes, dtype=dtype, device=device, requires_grad=requires_grad)
        return cls(re, torch.zeros_like(re, requires_grad=requires_grad))


class CplxParameter(torch.nn.ParameterDict):
    """Torch-friendly container for complex-valued parameter."""

    def __init__(self, cplx):
        if not isinstance(cplx, Cplx):
            raise TypeError(f'`{type(self).__name__}` accepts only Cplx tensors.')
        super().__init__({'real': torch.nn.Parameter(cplx.real), 'imag': torch.nn.Parameter(cplx.imag)})
        torch.nn.Module.__setattr__(self, '_cplx', cplx)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        missing, unexpected = [], []
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing, unexpected, error_msgs)
        if len(missing) == 2:
            real, dot, _ = prefix.rpartition('.')
            if real not in state_dict:
                missing = [real]
            else:
                par, missing, unexpected = state_dict[real], [], []
                if real in unexpected:
                    unexpected.remove(real)
                self._load_from_state_dict({f'{prefix}real': par, f'{prefix}imag': torch.zeros_like(par)}, prefix, local_metadata, strict, [], [], error_msgs)
        elif len(missing) == 1:
            error_msgs.append(f'Complex parameter requires both `.real` and `.imag` parts. Missing `{missing[0]}`.')
        if strict and unexpected:
            error_msgs.append(f'Complex parameter disallows redundant key(s) in state_dict: {unexpected}.')
        unexpected_keys.extend(unexpected)
        missing_keys.extend(missing)

    def extra_repr(self):
        return repr(tuple(self._cplx.shape))[1:-1]

    @property
    def data(self):
        return self._cplx


class CplxParameterAccessor:
    """Cosmetic complex parameter accessor.

    Details
    -------
    This works both for the default `forward()` inherited from Linear,
    and for what the user expects to see when they request weight from
    the layer (masked zero values).

    Warning
    -------
    This hacky property works only because torch.nn.Module implements
    its own special attribute access mechanism via `__getattr__`. This
    is why `SparseWeightMixin` in .masked couldn't work with 'weight'
    as a read-only @property.
    """

    def __getattr__(self, name):
        attr = super().__getattr__(name)
        if not isinstance(attr, CplxParameter):
            return attr
        return Cplx(attr.real, attr.imag)


class BaseRealToCplx(torch.nn.Module):
    pass


def _promote_callable_to_split(fn):
    """Create a runtime class promoting a real function to split activation."""


    class template(CplxToCplx):

        @wraps(fn)
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.args, self.kwargs = args, kwargs

        def extra_repr(self):
            pieces = [f'{v!r}' for v in self.args]
            pieces.extend(f'{k}={v!r}' for k, v in self.kwargs.items())
            return ', '.join(pieces)

        def forward(self, input):
            return input.apply(fn, *self.args, **self.kwargs)
    template.__name__ = f'CplxSplitFunc{fn.__name__.title()}'
    template.__qualname__ = f'<runtime type `{template.__name__}`>'
    template.__doc__ = f'Split activation based on `{fn.__name__}`'
    template.forward.__doc__ = f'Call `{fn.__name__}` on real and imaginary components independently.'
    return template


def _promote_module_to_split(Module):
    """Make a runtime class promoting a Module subclass to split activation."""


    class template(Module, CplxToCplx):

        def forward(self, input):
            """Apply to real and imaginary parts independently."""
            return input.apply(super().forward)
    template.__name__ = f'CplxSplitLayer{Module.__name__}'
    template.__qualname__ = f'<runtime type `{template.__name__}`>'
    template.__doc__ = f'Split activation based on `{Module.__name__}`'
    return template


class _CplxToCplxMeta(type):
    """Meta class for promoting real activations to split complex ones."""

    @lru_cache(maxsize=None)
    def __getitem__(self, Base):
        if not isinstance(Base, type) and callable(Base):
            return _promote_callable_to_split(Base)
        elif isinstance(Base, type) and issubclass(Base, torch.nn.Module):
            if issubclass(Base, (CplxToCplx, BaseRealToCplx)):
                return Base
            if Base is torch.nn.Module:
                return CplxToCplx
            return _promote_module_to_split(Base)
        raise TypeError(f'Expecting either a torch.nn.Module subclass, or a callable for promotion. Got `{type(Base)}`.')


class CplxToCplx(CplxParameterAccessor, torch.nn.Module, metaclass=_CplxToCplxMeta):
    pass


class CplxLinear(CplxToCplx):
    """Complex linear transform:
    $$
        F
        \\colon \\mathbb{C}^{\\ldots \\times d_0}
                \\to \\mathbb{C}^{\\ldots \\times d_1}
        \\colon u + i v \\mapsto W_\\mathrm{re} (u + i v) + i W_\\mathrm{im} (u + i v)
                = (W_\\mathrm{re} u - W_\\mathrm{im} v)
                    + i (W_\\mathrm{im} u + W_\\mathrm{re} v)
        \\,. $$
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = CplxParameter(cplx.Cplx.empty(out_features, in_features))
        if bias:
            self.bias = CplxParameter(cplx.Cplx.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        weight, bias = self.weight, self.bias
        init.cplx_kaiming_uniform_(weight, a=math.sqrt(5))
        if bias is not None:
            fan_in, _ = init.get_fans(weight)
            bound = 1 / math.sqrt(fan_in)
            init.cplx_uniform_independent_(bias, -bound, bound)

    def forward(self, input):
        return cplx.linear(input, self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(self.in_features, self.out_features, self.bias is not None)


class CplxLinearMasked(CplxLinear, _BaseCplxMixin):

    def forward(self, input):
        return cplx.linear(input, self.weight_masked, self.bias)


class CplxBilinear(CplxToCplx):
    """Complex bilinear transform:
    $$
        F
        \\colon \\mathbb{C}^{\\ldots \\times d_0}
                    \\times  \\mathbb{C}^{\\ldots \\times d_1}
                \\to \\mathbb{C}^{\\ldots \\times d_2}
        \\colon (u, v) \\mapsto (u^\\top A_j v)_{j=1}^{d_2}
        \\,. $$
    """

    def __init__(self, in1_features, in2_features, out_features, bias=True, conjugate=True):
        super().__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.weight = CplxParameter(cplx.Cplx.empty(out_features, in1_features, in2_features))
        if bias:
            self.bias = CplxParameter(cplx.Cplx.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.conjugate = conjugate
        self.reset_parameters()

    def reset_parameters(self):
        init.cplx_kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init.get_fans(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.cplx_uniform_independent_(self.bias, -bound, bound)

    def forward(self, input1, input2):
        return cplx.bilinear(input1, input2, self.weight, self.bias, self.conjugate)

    def extra_repr(self):
        fmt = 'in1_features={}, in2_features={}, out_features={}, bias={}, conjugate={}'
        return fmt.format(self.in1_features, self.in2_features, self.out_features, self.bias is not None, self.conjugate)


class CplxBilinearMasked(CplxBilinear, _BaseCplxMixin):

    def forward(self, input1, input2):
        return cplx.bilinear(input1, input2, self.weight_masked, self.bias)


class CplxConvNd(CplxToCplx):
    """An almost verbatim copy of `_ConvNd` from torch/nn/modules/conv.py"""

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, transposed, output_padding, groups, bias, padding_mode):
        super().__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size, self.stride = kernel_size, stride
        self.padding, self.dilation = padding, dilation
        self.transposed, self.output_padding = transposed, output_padding
        self.groups, self.padding_mode = groups, padding_mode
        if transposed:
            self.weight = CplxParameter(cplx.Cplx.empty(in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = CplxParameter(cplx.Cplx.empty(out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = CplxParameter(cplx.Cplx.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.cplx_kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init.get_fans(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.cplx_uniform_independent_(self.bias, -bound, bound)

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
        if self.padding_mode != 'zeros':
            s += ", padding_mode='{padding_mode}'"
        return s.format(**self.__dict__)


class CplxConv1d(CplxConvNd):
    """Complex 1D convolution:
    $$
        F
        \\colon \\mathbb{C}^{B \\times c_{in} \\times L}
                \\to \\mathbb{C}^{B \\times c_{out} \\times L'}
        \\colon u + i v \\mapsto (W_\\mathrm{re} \\star u - W_\\mathrm{im} \\star v)
                                + i (W_\\mathrm{im} \\star u + W_\\mathrm{re} \\star v)
        \\,. $$

    See torch.nn.Conv1d for reference on the input dimensions.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super().__init__(in_channels, out_channels, _single(kernel_size), _single(stride), _single(padding), _single(dilation), False, _single(0), groups, bias, padding_mode)

    def forward(self, input):
        return cplx.conv1d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups, self.padding_mode)


class CplxConv1dMasked(CplxConv1d, _BaseCplxMixin):

    def forward(self, input):
        return cplx.conv1d(input, self.weight_masked, self.bias, self.stride, self.padding, self.dilation, self.groups, self.padding_mode)


class CplxConv2d(CplxConvNd):
    """Complex 2D convolution:
    $$
        F
        \\colon \\mathbb{C}^{B \\times c_{in} \\times L}
                \\to \\mathbb{C}^{B \\times c_{out} \\times L'}
        \\colon u + i v \\mapsto (W_\\mathrm{re} \\star u - W_\\mathrm{im} \\star v)
                                + i (W_\\mathrm{im} \\star u + W_\\mathrm{re} \\star v)
        \\,. $$

    See torch.nn.Conv2d for reference on the input dimensions.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super().__init__(in_channels, out_channels, _pair(kernel_size), _pair(stride), _pair(padding), _pair(dilation), False, _pair(0), groups, bias, padding_mode)

    def forward(self, input):
        return cplx.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups, self.padding_mode)


class CplxConv2dMasked(CplxConv2d, _BaseCplxMixin):

    def forward(self, input):
        return cplx.conv2d(input, self.weight_masked, self.bias, self.stride, self.padding, self.dilation, self.groups, self.padding_mode)


class CplxConv3d(CplxConvNd):
    """Complex 3D convolution:
    $$
        F
        \\colon \\mathbb{C}^{B \\times c_{in} \\times L}
                \\to \\mathbb{C}^{B \\times c_{out} \\times L'}
        \\colon u + i v \\mapsto (W_\\mathrm{re} \\star u - W_\\mathrm{im} \\star v)
                                + i (W_\\mathrm{im} \\star u + W_\\mathrm{re} \\star v)
        \\,. $$

    See torch.nn.Conv2d for reference on the input dimensions.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super().__init__(in_channels, out_channels, _triple(kernel_size), _triple(stride), _triple(padding), _triple(dilation), False, _triple(0), groups, bias, padding_mode)

    def forward(self, input):
        return cplx.conv3d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups, self.padding_mode)


class CplxConv3dMasked(CplxConv3d, _BaseCplxMixin):

    def forward(self, input):
        return cplx.conv3d(input, self.weight_masked, self.bias, self.stride, self.padding, self.dilation, self.groups, self.padding_mode)


class _BaseRealMixin(MaskedWeightMixin, BaseMasked, SparsityStats):
    __sparsity_ignore__ = 'mask',

    def sparsity(self, *, hard=True, **kwargs):
        if self.is_sparse:
            mask = torch.gt(self.mask, 0) if hard else self.mask
            n_dropped = float(self.weight.numel())
            n_dropped -= float(mask.sum().item())
        else:
            n_dropped = 0.0
        return [(id(self.weight), n_dropped)]


class LinearMasked(Linear, _BaseRealMixin):

    def forward(self, input):
        return F.linear(input, self.weight_masked, self.bias)


class Conv1dMasked(Conv1d, _BaseRealMixin):

    def forward(self, input):
        return F.conv1d(input, self.weight_masked, self.bias, self.stride, self.padding, self.dilation, self.groups)


class Conv2dMasked(Conv2d, _BaseRealMixin):

    def forward(self, input):
        return F.conv2d(input, self.weight_masked, self.bias, self.stride, self.padding, self.dilation, self.groups)


class Conv3dMasked(Conv3d, _BaseRealMixin):

    def forward(self, input):
        return F.conv3d(input, self.weight_masked, self.bias, self.stride, self.padding, self.dilation, self.groups)


class BilinearMasked(Bilinear, _BaseRealMixin):

    def forward(self, input1, input2):
        return F.bilinear(input1, input2, self.weight_masked, self.bias)


class BaseCplxToReal(torch.nn.Module):
    pass


def whiten2x2(tensor, training=True, running_mean=None, running_cov=None, momentum=0.1, nugget=1e-05):
    """Solve R M R = I for R and a given 2x2 matrix M = [[a, b], [c, d]].

    Arguments
    ---------
    tensor : torch.tensor
        The input data expected to be at least 3d, with shape [2, B, F, ...],
        where `B` is the batch dimension, `F` -- the channels/features,
        `...` -- the spatial dimensions (if present). The leading dimension
        `2` represents real and imaginary components (stacked).

    training : bool, default=True
        Determines whether to update running feature statistics, if they are
        provided, or use them instead of batch computed statistics. If `False`
        then `running_mean` and `running_cov` MUST be provided.

    running_mean : torch.tensor, or None
        The tensor with running mean statistics having shape [2, F]. Ignored
        if explicitly `None`.

    running_cov : torch.tensor, or None
        The tensor with running real-imaginary covariance statistics having
        shape [2, 2, F]. Ignored if explicitly `None`.

    momentum : float, default=0.1
        The weight in the exponential moving average used to keep track of the
        running feature statistics.

    nugget : float, default=1e-05
        The ridge coefficient to stabilise the estimate of the real-imaginary
        covariance.

    Details
    -------
    Using (tril) L L^T = V seems to 'favour' the first dimension (re), so
    Trabelsi et al. (2018) used explicit 2x2 root of M: such R that M = RR.

    For M = [[a, b], [c, d]] we have the following facts:
        (1) inv M = \\frac1{ad - bc} [[d, -b], [-c, a]]
        (2) \\sqrt{M} = \\frac1{t} [[a + s, b], [c, d + s]]
            for s = \\sqrt{ad - bc}, t = \\sqrt{a + d + 2 s}
            det \\sqrt{M} = t^{-2} (ad + s(d + a) + s^2 - bc) = s

    Therefore `inv \\sqrt{M} = [[p, q], [r, s]]`, where
        [[p, q], [r, s]] = \\frac1{t s} [[d + s, -b], [-c, a + s]]
    """
    assert tensor.dim() >= 3
    tail = 1, tensor.shape[2], *([1] * (tensor.dim() - 3))
    axes = 1, *range(3, tensor.dim())
    if training or running_mean is None:
        mean = tensor.mean(dim=axes)
        if running_mean is not None:
            running_mean += momentum * (mean.data - running_mean)
    else:
        mean = running_mean
    tensor = tensor - mean.reshape(2, *tail)
    if training or running_cov is None:
        var = (tensor * tensor).mean(dim=axes) + nugget
        cov_uu, cov_vv = var[0], var[1]
        cov_vu = cov_uv = (tensor[0] * tensor[1]).mean([(a - 1) for a in axes])
        if running_cov is not None:
            cov = torch.stack([cov_uu.data, cov_uv.data, cov_vu.data, cov_vv.data], dim=0).reshape(2, 2, -1)
            running_cov += momentum * (cov - running_cov)
    else:
        cov_uu, cov_uv, cov_vu, cov_vv = running_cov.reshape(4, -1)
    sqrdet = torch.sqrt(cov_uu * cov_vv - cov_uv * cov_vu)
    denom = sqrdet * torch.sqrt(cov_uu + 2 * sqrdet + cov_vv)
    p, q = (cov_vv + sqrdet) / denom, -cov_uv / denom
    r, s = -cov_vu / denom, (cov_uu + sqrdet) / denom
    out = torch.stack([tensor[0] * p.reshape(tail) + tensor[1] * r.reshape(tail), tensor[0] * q.reshape(tail) + tensor[1] * s.reshape(tail)], dim=0)
    return out


def cplx_batch_norm(input, running_mean, running_var, weight=None, bias=None, training=True, momentum=0.1, eps=1e-05):
    """Applies complex-valued Batch Normalization as described in
    (Trabelsi et al., 2018) for each channel across a batch of data.

    Arguments
    ---------
    input : complex-valued tensor
        The input complex-valued data is expected to be at least 2d, with
        shape [B, F, ...], where `B` is the batch dimension, `F` -- the
        channels/features, `...` -- the spatial dimensions (if present).

    running_mean : torch.tensor, or None
        The tensor with running mean statistics having shape [2, F]. Ignored
        if explicitly `None`.

    running_var : torch.tensor, or None
        The tensor with running real-imaginary covariance statistics having
        shape [2, 2, F]. Ignored if explicitly `None`.

    weight : torch.tensor, default=None
        The 2x2 weight matrix of the affine transformation of real and
        imaginary parts post normalization. Has shape [2, 2, F] . Ignored
        together with `bias` if explicitly `None`.

    bias : torch.tensor, or None
        The offest (bias) of the affine transformation of real and imaginary
        parts post normalization. Has shape [2, F] . Ignored together with
        `weight` if explicitly `None`.

    training : bool, default=True
        Determines whether to update running feature statistics, if they are
        provided, or use them instead of batch computed statistics. If `False`
        then `running_mean` and `running_var` MUST be provided.

    momentum : float, default=0.1
        The weight in the exponential moving average used to keep track of the
        running feature statistics.

    eps : float, default=1e-05
        The ridge coefficient to stabilise the estimate of the real-imaginary
        covariance.

    Details
    -------
    Has non standard interface for running stats and weight and bias of the
    affine transformation for purposes of improved memory locality (noticeable
    speedup both on host and device computations).
    """
    assert running_mean is None and running_var is None or running_mean is not None and running_var is not None
    assert weight is None and bias is None or weight is not None and bias is not None
    x = torch.stack([input.real, input.imag], dim=0)
    z = whiten2x2(x, training=training, running_mean=running_mean, running_cov=running_var, momentum=momentum, nugget=eps)
    if weight is not None and bias is not None:
        shape = 1, x.shape[2], *([1] * (x.dim() - 3))
        weight = weight.reshape(2, 2, *shape)
        z = torch.stack([z[0] * weight[0, 0] + z[1] * weight[0, 1], z[0] * weight[1, 0] + z[1] * weight[1, 1]], dim=0) + bias.reshape(2, *shape)
    return cplx.Cplx(z[0], z[1])


class _CplxBatchNorm(CplxToCplx):
    """The base clas for Complex-valeud batch normalization layer.

    Taken from `torch.nn.modules.batchnorm` verbatim.
    """

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = torch.nn.Parameter(torch.empty(2, 2, num_features))
            self.bias = torch.nn.Parameter(torch.empty(2, num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.empty(2, num_features))
            self.register_buffer('running_var', torch.empty(2, 2, num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_running_stats()
        self.reset_parameters()

    def reset_running_stats(self):
        if not self.track_running_stats:
            return
        self.num_batches_tracked.zero_()
        self.running_mean.zero_()
        self.running_var.copy_(torch.eye(2, 2).unsqueeze(-1))

    def reset_parameters(self):
        if not self.affine:
            return
        self.weight.data.copy_(torch.eye(2, 2).unsqueeze(-1))
        init.zeros_(self.bias)

    def _check_input_dim(self, input):
        raise NotImplementedError

    def forward(self, input):
        self._check_input_dim(input)
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum
        return cplx_batch_norm(input, self.running_mean, self.running_var, self.weight, self.bias, self.training or not self.track_running_stats, exponential_average_factor, self.eps)

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, track_running_stats={track_running_stats}'.format(**vars(self))


class CplxBatchNorm1d(_CplxBatchNorm):
    """Complex-valued batch normalization for 2D or 3D data.
    See torch.nn.BatchNorm1d for details.
    """

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'.format(input.dim()))


class CplxBatchNorm2d(_CplxBatchNorm):
    """Complex-valued batch normalization for 4D data.
    See torch.nn.BatchNorm2d for details.
    """

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))


class CplxBatchNorm3d(_CplxBatchNorm):
    """Complex-valued batch normalization for 5D data.
    See torch.nn.BatchNorm3d for details.
    """

    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'.format(input.dim()))


class InterleavedRealToCplx(BaseRealToCplx):
    """Reinterpret the last dimension as interleaved real and imaginary
    components of a complex tensor. The input tensor must have even number
    in the last dimension, and the output has all dimensions preserved but
    the last, which is halved and not squeezed.
    $$
        F
        \\colon \\mathbb{R}^{\\ldots \\times [d \\times 2]}
                \\to \\mathbb{C}^{\\ldots \\times d}
        \\colon x \\mapsto \\bigr(
            x_{2k} + i x_{2k+1}
        \\bigl)_{k=0}^{d-1}
        \\,. $$

    Inverts `CplxToInterleavedReal`.
    """

    def __init__(self, copy=False, dim=-1):
        super().__init__()
        self.copy, self.dim = copy, dim

    def forward(self, input):
        return cplx.from_interleaved_real(input, self.copy, self.dim)


class ConcatenatedRealToCplx(BaseRealToCplx):
    """Interpret the last dimension as a concatenation of real part and then
    imaginary component. Preserves all dimensions except for the last, which
    is halved and not squeezed.
    $$
        F
        \\colon \\mathbb{R}^{\\ldots \\times [2 \\times d]}
                \\to \\mathbb{C}^{\\ldots \\times d}
        \\colon x \\mapsto \\bigr(
            x_{k} + i x_{d + k}
        \\bigl)_{k=0}^{d-1}
        \\,. $$

    Inverts `CplxToConcatenatedReal`.
    """

    def __init__(self, copy=False, dim=-1):
        super().__init__()
        self.copy, self.dim = copy, dim

    def forward(self, input):
        return cplx.from_concatenated_real(input, self.copy, self.dim)


class CplxToInterleavedReal(BaseCplxToReal):
    """Represent a Cplx tensor in the interleaved format along the last
    dimension: in consecutive pairs of real and imaginary parts
    $$
        F
        \\colon \\mathbb{C}^{\\ldots \\times d}
                \\to \\mathbb{R}^{\\ldots \\times [d \\times 2]}
        \\colon u + i v \\mapsto \\bigl(u_\\omega, v_\\omega\\bigr)_{\\omega}
        \\,. $$

    Inverts `InterleavedRealToCplx`.
    """

    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return cplx.to_interleaved_real(input, True, self.dim)


class CplxToConcatenatedReal(BaseCplxToReal):
    """Represent a Cplx tensor in concatenated format along the last
    dimension: the whole real component followed by the whole imaginary part
    $$
        F
        \\colon \\mathbb{C}^{\\ldots \\times d}
                \\to \\mathbb{R}^{\\ldots \\times [2 \\times d]}
        \\colon u + i v \\mapsto \\bigl(u_\\omega, v_\\omega \\bigr)_{\\omega}
        \\,. $$

    Inverts `ConcatenatedRealToCplx`.
    """

    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return cplx.to_concatenated_real(input, None, self.dim)


class AsTypeCplx(BaseRealToCplx):
    """Interpret the tensor as a Cplx tensor having zero imaginary part
    (embeds $\\mathbb{R} \\hookrightarrow \\mathbb{C}$):
    $$
        F
        \\colon \\mathbb{R}^{\\ldots \\times d}
                \\to \\mathbb{C}^{\\ldots \\times d}
        \\colon x \\mapsto x + 0 i
        \\,. $$

    Inverts `nn.linear.CplxReal`.
    """

    def forward(self, input):
        return cplx.Cplx(input)


class TensorToCplx(BaseRealToCplx):
    """Interpret a tensor with the last dimension of size exactly 2, which
    represents the real and imaginary components of a complex tensor. All
    dimensions preserved but the last, which is dropped.
    $$
        F
        \\colon \\mathbb{R}^{\\ldots \\times 2}
                \\to \\mathbb{C}^{\\ldots}
        \\colon x \\mapsto x_{\\ldots 0} + i x_{\\ldots 1}
        \\,. $$

    Inverts `CplxToTensor`.
    """

    def forward(self, input):
        """input must be a , and may have
        arbitrary number of.
        """
        assert input.shape[-1] == 2
        return cplx.Cplx(input[..., 0], input[..., 1])


class CplxToTensor(BaseCplxToReal):
    """Represent a Cplx tensor in torch's complex tensor format with a new
    last dimension of size exactly 2, representing the real and imaginary
    components of complex numbers.
    $$
        F
        \\colon \\mathbb{C}^{\\ldots}
                \\to \\mathbb{R}^{\\ldots \\times 2}
        \\colon u + i v \\mapsto \\bigl(u_\\omega, v_\\omega \\bigr)_{\\omega}
        \\,. $$

    Inverts `TensorToCplx`.
    """

    def forward(self, input):
        return cplx.to_interleaved_real(input, False, -1)


def is_from_cplx(module):
    if isinstance(module, (CplxToCplx, BaseCplxToReal)):
        return True
    if isinstance(module, torch.nn.Sequential):
        return is_from_cplx(module[0])
    if isinstance(module, type):
        return issubclass(module, (CplxToCplx, BaseCplxToReal))
    return False


def is_to_cplx(module):
    if isinstance(module, (CplxToCplx, BaseRealToCplx)):
        return True
    if isinstance(module, torch.nn.Sequential):
        return is_to_cplx(module[-1])
    if isinstance(module, type):
        return issubclass(module, (CplxToCplx, BaseRealToCplx))
    return False


def is_cplx_to_cplx(module):
    return is_from_cplx(module) and is_to_cplx(module)


class CplxSequential(torch.nn.Sequential, CplxToCplx):
    """Sequence of complex-to-complex modules:
    $$
        z_l = F_l(z_{l-1})
        \\,, $$
    for $l=1..L$ and the complex input $z_0$.
    """

    def __init__(self, *args):
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            modules = args[0].items()
        else:
            modules = enumerate(args)
        bad_modules = [str(n) for n, m in modules if not is_cplx_to_cplx(m)]
        if bad_modules:
            raise TypeError(f'Only complex-to-complex modules can be used in {self.__class__.__name__}. The following modules failed: {bad_modules}.')
        super().__init__(*args)


class CplxConvTransposeNd(CplxConvNd):
    """An almost verbatim copy of `_ConvTransposeNd` from torch/nn/modules/conv.py"""

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, transposed, output_padding, groups, bias, padding_mode):
        if padding_mode not in ('zeros', 'circular'):
            raise ValueError('Only "zeros" or "circular" padding mode are supported by `{}`'.format(self.__class__.__name__))
        super().__init__(in_channels, out_channels, _triple(kernel_size), _triple(stride), _triple(padding), _triple(dilation), transposed, output_padding, groups, bias, padding_mode)
    _output_padding = torch.nn.modules.conv._ConvTransposeNd._output_padding


class CplxConvTranspose1d(CplxConvTransposeNd):
    """Complex 1D transposed convolution.

    See torch.nn.ConvTranspose1d for reference on the input dimensions.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, output_padding=0, groups=1, bias=None, padding_mode='zeros'):
        super().__init__(in_channels, out_channels, _single(kernel_size), _single(stride), _single(padding), _single(dilation), True, _single(output_padding), groups, bias, padding_mode)

    def forward(self, input, output_size=None):
        if self.padding_mode not in ('zeros', 'circular'):
            raise ValueError('Only `zeros` or `circular` padding mode are supported by `CplxConvTranspose1d`')
        output_padding = self._output_padding(input, output_size, self.stride, self.padding, self.kernel_size)
        return cplx.conv_transpose1d(input, self.weight, self.bias, self.stride, self.padding, output_padding, self.groups, self.dilation, self.padding_mode)


class CplxConvTranspose2d(CplxConvTransposeNd):
    """Complex 2D transposed convolution.

    See torch.nn.ConvTranspose2d for reference on the input dimensions.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, output_padding=0, groups=1, bias=None, padding_mode='zeros'):
        super().__init__(in_channels, out_channels, _pair(kernel_size), _pair(stride), _pair(padding), _pair(dilation), True, _pair(output_padding), groups, bias, padding_mode)

    def forward(self, input, output_size=None):
        if self.padding_mode not in ('zeros', 'circular'):
            raise ValueError('Only `zeros` or `circular` padding mode are supported by `CplxConvTranspose2d`')
        output_padding = self._output_padding(input, output_size, self.stride, self.padding, self.kernel_size)
        return cplx.conv_transpose2d(input, self.weight, self.bias, self.stride, self.padding, output_padding, self.groups, self.dilation, self.padding_mode)


class CplxConvTranspose3d(CplxConvTransposeNd):
    """Complex 3D transposed convolution.

    See torch.nn.ConvTranspose3d for reference on the input dimensions.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, output_padding=0, groups=1, bias=None, padding_mode='zeros'):
        super().__init__(in_channels, out_channels, _triple(kernel_size), _triple(stride), _triple(padding), _triple(dilation), True, _triple(output_padding), groups, bias, padding_mode)

    def forward(self, input, output_size=None):
        if self.padding_mode not in ('zeros', 'circular'):
            raise ValueError('Only `zeros` or `circular` padding mode are supported by `CplxConvTranspose3d`')
        output_padding = self._output_padding(input, output_size, self.stride, self.padding, self.kernel_size)
        return cplx.conv_transpose3d(input, self.weight, self.bias, self.stride, self.padding, output_padding, self.groups, self.dilation, self.padding_mode)


class CplxDropout(torch.nn.Dropout2d, CplxToCplx):
    """Complex 1d dropout layer: simultaneous dropout on both real and
    imaginary parts.

    See torch.nn.Dropout1d for reference on the input dimensions and arguments.
    """

    def forward(self, input):
        *head, n_last = input.shape
        tensor = torch.stack([input.real, input.imag], dim=-1)
        output = super().forward(tensor.reshape(-1, 1, 2))
        output = output.reshape(*head, -1)
        return cplx.from_interleaved_real(output, False, -1)


class CplxIdentity(torch.nn.Identity, CplxToCplx):
    pass


class CplxReal(BaseCplxToReal):

    def forward(self, input):
        return input.real


class CplxImag(BaseCplxToReal):

    def forward(self, input):
        return input.imag


class CplxPhaseShift(CplxToCplx):
    """A learnable complex phase shift
    $$
        F
        \\colon \\mathbb{C}^{\\ldots \\times C \\times d}
                \\to \\mathbb{C}^{\\ldots \\times C \\times d}
        \\colon z \\mapsto z_{\\ldots kj} e^{i \\theta_{kj}}
        \\,, $$
    where $\\theta_k$ is the phase shift of the $k$-the channel in radians.
    Torch's broadcasting rules apply and the passed dimensions conform with
    the upstream input. For example, `CplxPhaseShift(C, 1)` shifts each $d$-dim
    complex vector by the phase of its channel, and `CplxPhaseShift(d)` shifts
    each complex feature in all channels by the same phase. Finally calling
    with CplxPhaseShift(1) shifts the inputs by the single common phase.
    """

    def __init__(self, *dim):
        super().__init__()
        self.phi = torch.nn.Parameter(torch.randn(*dim) * 0.02)

    def forward(self, input):
        return cplx.phaseshift(input, self.phi)


class CplxMaxPoolNd(CplxToCplx):
    """An almost verbatim copy of `_MaxPoolNd` from torch/nn/modules/pooling.py"""

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False) ->None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def extra_repr(self) ->str:
        return 'kernel_size={kernel_size}, stride={stride}, padding={padding}, dilation={dilation}, ceil_mode={ceil_mode}'.format(**self.__dict__)


class CplxMaxPool1d(CplxMaxPoolNd):
    """Applies a 1D max pooling over a complex input signal composed of
    several input planes.

    See Also
    --------
    See docs for the parameters of `torch.nn.MaxPool1d`.
    """

    def forward(self, input):
        return cplx.max_pool1d(input, self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode)


class CplxMaxPool2d(CplxMaxPoolNd):
    """Applies a 2D max pooling over a complex input signal composed of
    several input planes.

    See Also
    --------
    See docs for the parameters of `torch.nn.MaxPool2d`.
    """

    def forward(self, input):
        return cplx.max_pool2d(input, self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode)


class CplxMaxPool3d(CplxMaxPoolNd):
    """Applies a 3D max pooling over a complex input signal composed of
    several input planes.

    See Also
    --------
    See docs for the parameters of `torch.nn.MaxPool3d`.
    """

    def forward(self, input):
        return cplx.max_pool3d(input, self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode)


class BaseARD(torch.nn.Module):
    """\\alpha-based variational dropout.

    Attributes
    ----------
    penalty : computed torch.Tensor, read-only
        The Kullback-Leibler divergence between the mean field approximate
        variational posterior of the weights and the scale-free log-uniform
        prior:
        $$
            KL(\\mathcal{N}(w\\mid \\theta, \\alpha \\theta^2) \\|
                    \\tfrac1{\\lvert w \\rvert})
                = \\mathbb{E}_{\\xi \\sim \\mathcal{N}(1, \\alpha)}
                    \\log{\\lvert \\xi \\rvert}
                - \\tfrac12 \\log \\alpha + C
            \\,. $$

    log_alpha : computed torch.Tensor, read-only
        Log-variance of the multiplicative scaling noise. Computed as a log
        of the ratio of the variance of the weight to the squared absolute
        value of the weight. The higher the log-alpha the less relevant the
        parameter is.

    Get the dropout mask based on the confidence level $\\tau \\in (0, 1)$:
        $$
            \\Pr(\\lvert w_i \\rvert > 0)
                \\leq \\Pr(z_i \\neq 0)
                = 1 - \\sigma\\bigl(
                    \\log\\alpha + \\beta \\log \\tfrac{-\\gamma}{\\zeta}
                \\bigr)
                \\leq \\tau
            \\,. $$
        For $\\tau=0.25$ and $\\beta=0.66$ we have `threshold=2.96`.

    """

    @property
    def penalty(self):
        """Get the penalty induced by the variational approximation.

        Returns
        -------
        mask : torch.Tensor, differentiable, read-only
            The penalty term to be computed, collected, and added to the
            negative log-likelihood. In variational dropout and automatic
            relevance determination methods this is the kl-divergence term
            of the ELBO, which depends on the variational approximation,
            and not the input data (like in VAE). The requires the use of
            forward hooks with specific trait modules that compute the
            differentiable penalty on forward pass. Which is currently out
            of the scope of this package.

        Details
        -------
        Making penalty into a property emphasizes its read-only-ness, however
        the same could've been achieved with a method.
        """
        raise NotImplementedError('Derived classes must compute their own penalty.')

    def relevance(self, **kwargs):
        """Get the dropout mask based on the provided parameters.

        Returns
        -------
        mask : torch.Tensor
            A nonnegative tensor of the same shape as the `.weight` parameter
            with explicit zeros indicating a dropped out parameter, a value to
            be set to and fixed at zero. A nonzero value indicates a relevant
            parameter, which is to be kept. A binary mask is `hard`, whereas a
            `soft` mask may use arbitrary positive values (not necessarily in
            [0, 1]) to represent retained parameters. Soft masks might occur in
            sparsification methods, where the sparsity mask is learnt and
            likely co-adapts to the weights, e.g. in \\ell_0 probabilistic
            regularization. Soft masks would require cleaning up to eliminate
            the result of such co-adaptation (see `nn.masked.binarize_masks`).
        """
        raise NotImplementedError('Derived classes must implement a float mask of relevant coefficients.')


class GaussianMixin:
    """Trait class with log-alpha property for variational dropout.

    Attributes
    ----------
    log_alpha : computed torch.Tensor, read-only
        Log-variance of the multiplicative scaling noise. Computed as a log
        of the ratio of the variance of the weight to the squared absolute
        value of the weight. The higher the log-alpha the less relevant the
        parameter is.
    """

    def reset_variational_parameters(self):
        self.log_sigma2.data.uniform_(-10, -10)

    @property
    def log_alpha(self):
        """Get $\\log \\alpha$ from $(\\theta, \\sigma^2)$ parameterization."""
        return self.log_sigma2 - 2 * torch.log(abs(self.weight) + 1e-12)


class CplxLinearGaussian(GaussianMixin, CplxLinear):
    """Complex-valued linear layer with variational dropout."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias=bias)
        self.log_sigma2 = torch.nn.Parameter(torch.Tensor(*self.weight.shape))
        self.reset_variational_parameters()

    def forward(self, input):
        mu = super().forward(input)
        if not self.training:
            return mu
        s2 = F.linear(input.real * input.real + input.imag * input.imag, torch.exp(self.log_sigma2), None)
        return mu + cplx.randn_like(s2) * torch.sqrt(torch.clamp(s2, 1e-08))


class CplxBilinearGaussian(GaussianMixin, CplxBilinear):
    """Complex-valued bilinear layer with variational dropout."""

    def __init__(self, in1_features, in2_features, out_features, bias=True, conjugate=True):
        super().__init__(in1_features, in2_features, out_features, bias=bias, conjugate=conjugate)
        self.log_sigma2 = torch.nn.Parameter(torch.Tensor(*self.weight.shape))
        self.reset_variational_parameters()

    def forward(self, input1, input2):
        mu = super().forward(input1, input2)
        if not self.training:
            return mu
        s2 = F.bilinear(input1.real * input1.real + input1.imag * input1.imag, input2.real * input2.real + input2.imag * input2.imag, torch.exp(self.log_sigma2), None)
        return mu + cplx.randn_like(s2) * torch.sqrt(torch.clamp(s2, 1e-08))


class CplxConvNdGaussianMixin(GaussianMixin):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
        if self.padding_mode != 'zeros':
            raise ValueError(f'Only `zeros` padding mode is supported. Got `{self.padding_mode}`.')
        self.log_sigma2 = torch.nn.Parameter(torch.Tensor(*self.weight.shape))
        self.reset_variational_parameters()

    def _forward_impl(self, input, conv):
        mu = super().forward(input)
        if not self.training:
            return mu
        s2 = conv(input.real * input.real + input.imag * input.imag, torch.exp(self.log_sigma2), None, self.stride, self.padding, self.dilation, self.groups)
        return mu + cplx.randn_like(s2) * torch.sqrt(torch.clamp(s2, 1e-08))


class CplxConv1dGaussian(CplxConvNdGaussianMixin, CplxConv1d):
    """1D complex-valued convolution layer with variational dropout."""

    def forward(self, input):
        return self._forward_impl(input, F.conv1d)


class CplxConv2dGaussian(CplxConvNdGaussianMixin, CplxConv2d):
    """2D complex-valued convolution layer with variational dropout."""

    def forward(self, input):
        return self._forward_impl(input, F.conv2d)


class CplxConv3dGaussian(CplxConvNdGaussianMixin, CplxConv3d):
    """3D complex-valued convolution layer with variational dropout."""

    def forward(self, input):
        return self._forward_impl(input, F.conv3d)


class ExpiFunction(torch.autograd.Function):
    """Pythonic differentiable port of scipy's Exponential Integral Ei.
    $$
        Ei
            \\colon \\mathbb{R} \\to \\mathbb{R} \\cup \\{\\pm \\infty\\}
            \\colon x \\mapsto \\int_{-\\infty}^x \\tfrac{e^t}{t} dt
        \\,. $$

    Notes
    -----
    This may potentially introduce a memory transfer and compute bottleneck
    during the forward pass due to CPU-GPU device switch. Backward pass does
    not suffer from this issue and is computed on-device.
    """

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        x_cpu = x.data.cpu().numpy()
        output = scipy.special.expi(x_cpu, dtype=x_cpu.dtype)
        return torch.from_numpy(output)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[-1]
        return grad_output * torch.exp(x) / x


torch_expi = ExpiFunction.apply


class CplxVDMixin:
    """Trait class with kl-divergence penalty of the cplx variational dropout.

    Details
    -------
    This module assumes the standard loss-minimization framework. Hence
    instead of -ve KL divergence for ELBO and log-likelihood maximization,
    this property computes and returns the divergence as is, which implies
    minimization of minus log-likelihood (and, thus, minus ELBO).

    Attributes
    ----------
    penalty : computed torch.Tensor, read-only
        The Kullback-Leibler divergence between the mean field approximate
        complex variational posterior of the weights and the scale-free
        log-uniform complex prior:
        $$
            KL(\\mathcal{CN}(w\\mid \\theta, \\alpha \\theta \\bar{\\theta}, 0) \\|
                    \\tfrac1{\\lvert w \\rvert^2})
                = 2 \\mathbb{E}_{\\xi \\sim \\mathcal{CN}(1, \\alpha, 0)}
                    \\log{\\lvert \\xi \\rvert}
                  + C - \\log \\alpha
                = C - \\log \\alpha - Ei( - \\tfrac1{\\alpha})
            \\,, $$
        where $Ei(x) = \\int_{-\\infty}^x e^t t^{-1} dt$ is the exponential
        integral. Unlike real-valued variational dropout, this KL divergence
        does not need an approximation, since it can be computed exactly via
        a special function. $Ei(x)$ behaves well on the -ve values, and near
        $0-$. The constant $C$ is fixed to Euler's gamma, so that the divergence
        is +ve.
    """

    @property
    def penalty(self):
        """Exact complex KL divergence."""
        n_log_alpha = -self.log_alpha
        return euler_gamma + n_log_alpha - torch_expi(-torch.exp(n_log_alpha))


class RelevanceMixin(SparsityStats):
    __sparsity_ignore__ = 'log_sigma2',

    def relevance(self, *, threshold, **kwargs):
        """Get the relevance mask based on the threshold."""
        with torch.no_grad():
            return torch.le(self.log_alpha, threshold)

    def sparsity(self, *, threshold, **kwargs):
        relevance = self.relevance(threshold=threshold)
        n_relevant = float(relevance.sum().item())
        return [(id(self.weight), self.weight.numel() - n_relevant)]


class CplxLinearVD(CplxVDMixin, RelevanceMixin, CplxLinearGaussian, BaseARD):
    """Complex-valued linear layer with variational dropout."""
    pass


class CplxBilinearVD(CplxVDMixin, RelevanceMixin, CplxBilinearGaussian, BaseARD):
    """Complex-valued bilinear layer with variational dropout."""
    pass


class CplxConv1dVD(CplxVDMixin, RelevanceMixin, CplxConv1dGaussian, BaseARD):
    """1D complex-valued convolution layer with variational dropout."""
    pass


class CplxConv2dVD(CplxVDMixin, RelevanceMixin, CplxConv2dGaussian, BaseARD):
    """2D complex-valued convolution layer with variational dropout."""
    pass


class CplxConv3dVD(CplxVDMixin, RelevanceMixin, CplxConv3dGaussian, BaseARD):
    """3D complex-valued convolution layer with variational dropout."""
    pass


class CplxVDScaleFreeMixin:

    @property
    def penalty(self):
        """The Kullback-Leibler divergence between the mean field approximate
        complex variational posterior of the weights and the scale-free
        log-uniform complex prior:
        $$
            KL(\\mathcal{CN}(w\\mid \\theta, \\alpha \\theta \\bar{\\theta}, 0) \\|
                    \\tfrac1{\\lvert w \\rvert})
                = \\mathbb{E}_{\\xi \\sim \\mathcal{CN}(1, \\alpha, 0)}
                    \\log{\\lvert \\xi \\rvert}
                  + \\log \\lvert \\theta \\rvert
                  + C - \\log \\alpha \\lvert \\theta \\rvert^2
                = - \\log \\lvert \\theta \\rvert - \\log \\alpha
                  + C - \\tfrac12 Ei( - \\tfrac1{\\alpha})
            \\,, $$
        where $Ei(x) = \\int_{-\\infty}^x e^t t^{-1} dt$ is the exponential
        integral. Unlike real-valued variational dropout, this KL divergence
        does not need an approximation, since it can be computed exactly via
        a special function. $Ei(x)$ behaves well on the -ve values, and near
        $0-$. The constant $C$ is fixed to half of Euler's gamma, so that the
        divergence is +ve.
        """
        log_abs_w = torch.log(abs(self.weight) + 1e-12)
        n_log_alpha = 2 * log_abs_w - self.log_sigma2
        ei = torch_expi(-torch.exp(n_log_alpha))
        return log_abs_w - self.log_sigma2 - 0.5 * ei


class CplxLinearVDScaleFree(CplxVDScaleFreeMixin, CplxLinearVD):
    """Complex-valued linear layer with scale-free prior."""
    pass


class CplxBilinearVDScaleFree(CplxVDScaleFreeMixin, CplxBilinearVD):
    """Complex-valued bilinear layer with scale-free prior."""
    pass


class CplxConv1dVDScaleFree(CplxVDScaleFreeMixin, CplxConv1dVD):
    """1D complex-valued convolution layer with scale-free prior."""
    pass


class CplxConv2dVDScaleFree(CplxVDScaleFreeMixin, CplxConv2dVD):
    """2D complex-valued convolution layer with scale-free prior."""
    pass


class CplxConv3dVDScaleFree(CplxVDScaleFreeMixin, CplxConv3dVD):
    """3D complex-valued convolution layer with scale-free prior."""
    pass


class CplxVDApproxMixin:

    @property
    def penalty(self):
        """Softplus-sigmoid approximation of the complex KL divergence.
        $$
            \\alpha \\mapsto
                \\log (1 + e^{-\\log \\alpha}) - C
                - k_1 \\sigma(k_2 + k_3 \\log \\alpha)
            \\,, $$
        with $C$ chosen as $- k_1$. Note that $x \\mapsto \\log(1 + e^x)$
        is known as `softplus` and in fact needs different compute paths
        depending on the sign of $x$, much like the stable method for the
        `log-sum-exp`:
        $$
            x \\mapsto
                \\log(1+e^{-\\lvert x\\rvert}) + \\max{\\{x, 0\\}}
            \\,. $$

        See the accompanying notebook for the MC estimation of the k1-k3
        constants: `k1, k2, k3 = 0.57810091, 1.45926293, 1.36525956`
        """
        n_log_alpha = -self.log_alpha
        sigmoid = torch.sigmoid(1.36526 * n_log_alpha - 1.45926)
        return F.softplus(n_log_alpha) + 0.5781 * sigmoid


class CplxLinearVDApprox(CplxVDApproxMixin, CplxLinearVD):
    """Complex-valued linear layer with approximate
    var-dropout penalty.
    """
    pass


class CplxBilinearVDApprox(CplxVDApproxMixin, CplxBilinearVD):
    """Complex-valued bilinear layer with approximate
    var-dropout penalty.
    """
    pass


class CplxConv1dVDApprox(CplxVDApproxMixin, CplxConv1dVD):
    """1D complex-valued convolution layer with approximate
    var-dropout penalty.
    """
    pass


class CplxConv2dVDApprox(CplxVDApproxMixin, CplxConv2dVD):
    """2D complex-valued convolution layer with approximate
    var-dropout penalty.
    """
    pass


class CplxConv3dVDApprox(CplxVDApproxMixin, CplxConv3dVD):
    """3D complex-valued convolution layer with approximate
    var-dropout penalty.
    """
    pass


class BogusExpiFunction(ExpiFunction):
    """The Dummy Expi function, that computes bogus values on the forward pass,
    but correct values on the backwards pass, provided there is no downstream
    dependence on its forward-pass output.
    """

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.zeros_like(x)


bogus_expi = BogusExpiFunction.apply


class CplxVDBogusMixin:

    @property
    def penalty(self):
        """KL-div with bogus forward output, but correct gradient."""
        log_alpha = self.log_alpha
        return -log_alpha - bogus_expi(-torch.exp(-log_alpha))


class CplxLinearVDBogus(CplxVDBogusMixin, CplxLinearVD):
    """Complex-valued linear layer with correct var dropout penalty
    gradient, but bogus penalty values.
    """
    pass


class CplxBilinearVDBogus(CplxVDBogusMixin, CplxBilinearVD):
    """Complex-valued bilinear layer with correct var dropout penalty
    gradient, but bogus penalty values.
    """
    pass


class CplxConv1dVDBogus(CplxVDBogusMixin, CplxConv1dVD):
    """1D complex-valued convolution layer with correct var dropout penalty
    gradient, but bogus penalty values.
    """
    pass


class CplxConv2dVDBogus(CplxVDBogusMixin, CplxConv2dVD):
    """2D complex-valued convolution layer with correct var dropout penalty
    gradient, but bogus penalty values.
    """
    pass


class CplxConv3dVDBogus(CplxVDBogusMixin, CplxConv3dVD):
    """3D complex-valued convolution layer with correct var dropout penalty
    gradient, but bogus penalty values.
    """
    pass


class LinearL0(torch.nn.Linear, BaseARD, SparsityStats):
    """L0 regularized linear layer according to [1]_.

    Details
    -------
    This implementation use -ve log-alpha parametrization in order to keep the
    layer's parameters interpretation aligned with the interpretation in
    variational dropout layer of Kingma et al (2015) and Molchanov et al (2017)
    (see also `cplxmodule.relevance.LinearARD`).

    This implementation follows the ICLR2018 closely, specifically it uses the
    equations 10-13, but ignores the caveat just before section 3. Instead, it
    uses the same sample of the gate $z$ for the whole minitbatch, as mentioned
    just before section 4.1, which could lead to much "larger variance in the
    gradients" w.r.t weights (Kingma et al. 2015).

    References
    ----------
    .. [1] Louizos, C., Welling M., Kingma, D. P. (2018). Learning Sparse
           Neural Networks through L0 Regularization. ICLR 2018
           https://arxiv.org/abs/1712.01312.pdf

    .. [2] Gale, T., Elsen, E., Hooker, S. (2019). The State of Sparsity in
           Deep Neural Networks. Arxiv preprint arXiv:1902.09574
           https://arxiv.org/abs/1902.09574.pdf

    .. [3] Maddison, C. J., Mnih, A., Teh, Y. W. (2017). The Concrete
           Distribution: a Continuous Relaxation of discrete Random Variables.
           ICLR 2017
           https://arxiv.org/pdf/1611.00712.pdf
    """
    __sparsity_ignore__ = 'log_alpha',
    beta, gamma, zeta = 0.66, -0.1, 1.1

    def __init__(self, in_features, out_features, bias=True, group=None):
        super().__init__(in_features, out_features, bias=bias)
        if group == 'input':
            shape = 1, in_features
        elif group == 'output':
            shape = out_features, 1
        else:
            shape = out_features, in_features
        self.log_alpha = torch.nn.Parameter(torch.Tensor(*shape))
        self.reset_variational_parameters()

    def reset_variational_parameters(self):
        self.log_alpha.data.uniform_(-2.197, -2.197)

    @property
    def penalty(self):
        """Penalty the probability of a nonzero gate $z$:
        $$
            \\Pr(\\lvert w_i \\rvert > 0)
                \\leq \\Pr(z_i \\neq 0)
                = 1 - \\sigma\\bigl(
                    \\log\\alpha + \\beta \\log \\tfrac{-\\gamma}{\\zeta}
                \\bigr)
            \\,, $$
        where $\\sigma(x) = (1+e^{-x})^{-1}$, which also satisfies the
        realtion $1 - \\sigma(x) = \\sigma(-x)$.
        """
        shift = -self.beta * math.log(-self.gamma / self.zeta)
        return torch.sigmoid(shift - self.log_alpha)

    def forward(self, input):
        n, m = self.log_alpha.shape
        if self.training:
            if n == 1 or m == 1:
                u = torch.rand(*input.shape[:-1], n, m, dtype=input.dtype, device=input.device)
            else:
                u = torch.rand_like(self.log_alpha)
            mask = self.gate(torch.log(u) - torch.log(1 - u))
        else:
            mask = self.gate(None)
        if n == 1:
            output = F.linear(input * mask.squeeze(-2), self.weight)
        elif m == 1:
            output = F.linear(input, self.weight) * mask.squeeze(-1)
        else:
            output = F.linear(input, self.weight * mask)
        if self.bias is not None:
            output += self.bias
        return output

    def gate(self, logit=None):
        """Implements the binary concrete hard-sigmoid transformation:
        $$
            F
            \\colon \\mathbb{R} \\to \\mathbb{R}
            \\colon x \\to g \\bigl(
                    \\ell_{\\zeta, \\gamma}(
                        \\sigma_{\\beta^{-1}}(x - \\log \\alpha)
                    )\\bigr)
            \\,, $$
        where $g(x) = \\min\\{1, \\max\\{0, x\\}\\}$ is the hard-sigmoid, $\\gamma <
        0 < \\zeta$ are the stretch parameters, $\\beta$ is the temperature and
        $\\sigma(z) = (1+e^{-z})^{-1}$.

        On train
        https://github.com/AMLab-Amsterdam/L0_regularization/blob/master/l0_layers.py#L64

        On eval
        https://github.com/AMLab-Amsterdam/L0_regularization/blob/master/l0_layers.py#L103
        """
        if logit is not None:
            s = torch.sigmoid((logit - self.log_alpha) / self.beta)
        else:
            s = torch.sigmoid(-self.log_alpha)
        return torch.clamp((self.zeta - self.gamma) * s + self.gamma, 0, 1)

    def relevance(self, *, hard, **kwargs):
        """Get the dropout mask based on the confidence level $\\tau \\in (0, 1)$:
        $$
            \\Pr(\\lvert w_i \\rvert > 0)
                \\leq \\Pr(z_i \\neq 0)
                = 1 - \\sigma\\bigl(
                    \\log\\alpha + \\beta \\log \\tfrac{-\\gamma}{\\zeta}
                \\bigr)
                \\leq \\tau
            \\,. $$
        For $\\tau=0.25$ and $\\beta=0.66$ we have `threshold=2.96`.
        """
        with torch.no_grad():
            velue = torch.gt(self.gate(None), 0) if hard else self.gate(None)
            return velue.expand_as(self.weight)

    def sparsity(self, *, hard, **kwargs):
        n_relevant = float(self.relevance(hard=hard).sum().item())
        return [(id(self.weight), self.weight.numel() - n_relevant)]


class LinearLASSO(torch.nn.Linear, BaseARD, SparsityStats):

    @property
    def penalty(self):
        return abs(self.weight)

    def relevance(self, *, threshold, **kwargs):
        with torch.no_grad():
            return torch.ge(torch.log(abs(self.weight) + 1e-20), threshold)

    def sparsity(self, *, threshold, **kwargs):
        n_relevant = float(self.relevance(threshold=threshold).sum().item())
        return [(id(self.weight), self.weight.numel() - n_relevant)]


class LinearGaussian(GaussianMixin, torch.nn.Linear):
    """Linear layer with variational dropout.

    Details
    -------
    See `torch.nn.Linear` for reference on the dimensions and parameters.
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias=bias)
        self.log_sigma2 = torch.nn.Parameter(torch.Tensor(*self.weight.shape))
        self.reset_variational_parameters()

    def forward(self, input):
        mu = super().forward(input)
        if not self.training:
            return mu
        s2 = F.linear(input * input, torch.exp(self.log_sigma2), None)
        return mu + torch.randn_like(s2) * torch.sqrt(torch.clamp(s2, 1e-08))


class BilinearGaussian(GaussianMixin, torch.nn.Bilinear):
    """Bilinear layer with variational dropout.

    Details
    -------
    See `torch.nn.Bilinear` for reference on the dimensions and parameters.
    """

    def __init__(self, in1_features, in2_features, out_features, bias=True):
        super().__init__(in1_features, in2_features, out_features, bias=bias)
        self.log_sigma2 = torch.nn.Parameter(torch.Tensor(*self.weight.shape))
        self.reset_variational_parameters()

    def forward(self, input1, input2):
        """Forward pass of the SGVB method for a bilinear layer.

        Straightforward generalization of the local reparameterization trick.
        """
        mu = super().forward(input1, input2)
        if not self.training:
            return mu
        s2 = F.bilinear(input1 * input1, input2 * input2, torch.exp(self.log_sigma2), None)
        return mu + torch.randn_like(s2) * torch.sqrt(torch.clamp(s2, 1e-08))


class ConvNdGaussianMixin(GaussianMixin):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
        if self.padding_mode != 'zeros':
            raise ValueError(f'Only `zeros` padding mode is supported. Got `{self.padding_mode}`.')
        self.log_sigma2 = torch.nn.Parameter(torch.Tensor(*self.weight.shape))
        self.reset_variational_parameters()

    def _forward_impl(self, input, conv):
        """Forward pass of the SGVB method for a Nd convolutional layer.

        Details
        -------
        A convolution can be represented as matrix-vector product of the doubly
        block-circulant embedding (Toeplitz) of the kernel and the unravelled
        input. As such, it is an implicit linear layer with block structured
        weight matrix, but unlike it, the local reparameterization trick has
        a little caveat. If the kernel itself is assumed to have the specified
        variational distribution, then the outputs will be spatially correlated
        due to the same weight block being reused at each location:
        $$
            cov(y_{f\\beta}, y_{k\\omega})
                = \\delta_{f=k} \\sum_{c \\alpha}
                    \\sigma^2_{fc \\alpha}
                    x_{c i_\\beta(\\alpha)}
                    x_{c i_\\omega(\\alpha)}
            \\,, $$
        where $i_\\beta(\\alpha)$ is the location in $x$ for the output location
        $\\beta$ and kernel offset $\\alpha$ (depends on stride and dilation).
        In contrast, if instead the Toeplitz embedding blocks are assumed iid
        draws from the variational distribution, then covariance becomes
        $$
            cov(y_{f\\beta}, y_{k\\omega})
                = \\delta_{f\\beta = k\\omega} \\sum_{c \\alpha}
                    \\sigma^2_{fc \\alpha}
                    \\lvert x_{c i_\\omega(\\alpha)} \\rvert^2
            \\,. $$
        Molchanov et al. (2017) implicitly assume that kernels is are iid draws
        from the variational distribution for different spatial locations. This
        effectively zeroes the spatial cross-correlation in the output, reduces
        the variance of the gradient in SGVB method.
        """
        mu = super().forward(input)
        if not self.training:
            return mu
        s2 = conv(input * input, torch.exp(self.log_sigma2), None, self.stride, self.padding, self.dilation, self.groups)
        return mu + torch.randn_like(s2) * torch.sqrt(torch.clamp(s2, 1e-08))


class Conv1dGaussian(ConvNdGaussianMixin, torch.nn.Conv1d):
    """1D convolution layer with variational dropout.

    Details
    -------
    See `torch.nn.Conv1d` for reference on the dimensions and parameters. See
    `cplxmodule.nn.relevance.ConvNdGaussianMixin` for details about the
    implementation of the reparameterization trick.
    """

    def forward(self, input):
        return self._forward_impl(input, F.conv1d)


class Conv2dGaussian(ConvNdGaussianMixin, torch.nn.Conv2d):
    """2D convolution layer with variational dropout.

    Details
    -------
    See `torch.nn.Conv2d` for reference on the dimensions and parameters. See
    `cplxmodule.nn.relevance.ConvNdGaussianMixin` for details about the
    implementation of the reparameterization trick.
    """

    def forward(self, input):
        return self._forward_impl(input, F.conv2d)


class Conv3dGaussian(ConvNdGaussianMixin, torch.nn.Conv3d):
    """3D convolution layer with variational dropout.

    Details
    -------
    See `torch.nn.Conv3d` for reference on the dimensions and parameters. See
    `cplxmodule.nn.relevance.ConvNdGaussianMixin` for details about the
    implementation of the reparameterization trick.
    """

    def forward(self, input):
        return self._forward_impl(input, F.conv3d)


class RealVDMixin:
    """Trait class with kl-divergence penalty of the variational dropout.

    Details
    -------
    This uses the ideas and formulae of Kingma et al. and Molchanov et al.
    This module assumes the standard loss-minimization framework. Hence
    instead of -ve KL divergence for ELBO and log-likelihood maximization,
    this property computes and returns the divergence as is, which implies
    minimization of minus log-likelihood (and, thus, minus ELBO).

    Attributes
    ----------
    penalty : computed torch.Tensor, read-only
        The Kullback-Leibler divergence between the mean field approximate
        variational posterior of the weights and the scale-free log-uniform
        prior:
        $$
            KL(\\mathcal{N}(w\\mid \\theta, \\alpha \\theta^2) \\|
                    \\tfrac1{\\lvert w \\rvert})
                = \\mathbb{E}_{\\xi \\sim \\mathcal{N}(1, \\alpha)}
                    \\log{\\lvert \\xi \\rvert}
                - \\tfrac12 \\log \\alpha + C
            \\,. $$
    """

    @property
    def penalty(self):
        """Sofplus-sigmoid approximation of the Kl divergence from
        arxiv:1701.05369:
        $$
            \\alpha \\mapsto
                \\tfrac12 \\log (1 + e^{-\\log \\alpha}) - C
                - k_1 \\sigma(k_2 + k_3 \\log \\alpha)
            \\,, $$
        with $C$ chosen to be $- k_1$. Note that $x \\mapsto \\log(1 + e^x)$
        is known as `softplus` and in fact needs different compute paths
        depending on the sign of $x$, much like the stable method for the
        `log-sum-exp`:
        $$
            x \\mapsto
                \\log(1 + e^{-\\lvert x\\rvert}) + \\max{\\{x, 0\\}}
            \\,. $$
        See the paper eq. (14) (mind the overall negative sign) or the
        accompanying notebook for the MC estimation of the constants:
        `k1, k2, k3 = 0.63576, 1.87320, 1.48695`
        """
        n_log_alpha = -self.log_alpha
        sigmoid = torch.sigmoid(1.48695 * n_log_alpha - 1.8732)
        return F.softplus(n_log_alpha) / 2 + 0.63576 * sigmoid


class LinearVD(RealVDMixin, RelevanceMixin, LinearGaussian, BaseARD):
    """Linear layer with variational dropout.

    Details
    -------
    See `torch.nn.Linear` for reference on the dimensions and parameters.
    """
    pass


class BilinearVD(RealVDMixin, RelevanceMixin, BilinearGaussian, BaseARD):
    """Bilinear layer with variational dropout.

    Details
    -------
    See `torch.nn.Bilinear` for reference on the dimensions and parameters.
    """
    pass


class Conv1dVD(RealVDMixin, RelevanceMixin, Conv1dGaussian, BaseARD):
    """1D convolution layer with variational dropout.

    Details
    -------
    See `torch.nn.Conv1d` for reference on the dimensions and parameters. See
    `cplxmodule.nn.relevance.ConvNdGaussianMixin` for details about the
    implementation of the reparameterization trick.
    """
    pass


class Conv2dVD(RealVDMixin, RelevanceMixin, Conv2dGaussian, BaseARD):
    """2D convolution layer with variational dropout.

    Details
    -------
    See `torch.nn.Conv2d` for reference on the dimensions and parameters. See
    `cplxmodule.nn.relevance.ConvNdGaussianMixin` for details about the
    implementation of the reparameterization trick.
    """
    pass


class Conv3dVD(RealVDMixin, RelevanceMixin, Conv3dGaussian, BaseARD):
    """3D convolution layer with variational dropout.

    Details
    -------
    See `torch.nn.Conv3d` for reference on the dimensions and parameters. See
    `cplxmodule.nn.relevance.ConvNdGaussianMixin` for details about the
    implementation of the reparameterization trick.
    """
    pass


class Model(torch.nn.Module):
    """A convolutional net."""

    def __init__(self, conv2d=Conv2d, linear=Linear):
        super().__init__()
        self.conv1 = conv2d(1, 20, 5, 1)
        self.conv2 = conv2d(20, 50, 5, 1)
        self.fc1 = linear(4 * 4 * 50, 500)
        self.fc2 = linear(500, 10)

    def forward(self, x):
        x = F.avg_pool2d(F.relu(self.conv1(x)), 2, 2)
        x = F.avg_pool2d(F.relu(self.conv2(x)), 2, 2)
        x = F.relu(self.fc1(x.reshape(-1, 4 * 4 * 50)))
        return F.log_softmax(self.fc2(x), dim=1)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BilinearGaussian,
     lambda: ([], {'in1_features': 4, 'in2_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (BilinearVD,
     lambda: ([], {'in1_features': 4, 'in2_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (Conv1dGaussian,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (Conv1dVD,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (Conv2dGaussian,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Conv2dVD,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Conv3dGaussian,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Conv3dVD,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (CplxIdentity,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CplxReal,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CplxSequential,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LinearGaussian,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (LinearL0,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (LinearLASSO,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LinearVD,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_ivannz_cplxmodule(_paritybench_base):
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

