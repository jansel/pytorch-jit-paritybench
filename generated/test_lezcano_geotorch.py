import sys
_module = sys.modules[__name__]
del sys
conf = _module
examples = _module
copying_problem = _module
eigenvalue = _module
sequential_mnist = _module
geotorch = _module
almostorthogonal = _module
constraints = _module
exceptions = _module
fixedrank = _module
glp = _module
grassmannian = _module
lowrank = _module
parametrize = _module
product = _module
psd = _module
pssd = _module
pssdfixedrank = _module
pssdlowrank = _module
reals = _module
skew = _module
sl = _module
so = _module
sphere = _module
stiefel = _module
symmetric = _module
utils = _module
setup = _module
test = _module
test_almostorthogonal = _module
test_glp = _module
test_integration = _module
test_lowrank = _module
test_orthogonal = _module
test_positive_semidefinite = _module
test_product = _module
test_skew = _module
test_sl = _module
test_sphere = _module
test_symmetric = _module

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


from torch import nn


import torch.nn.functional as F


import torch.nn as nn


import math


from torchvision import datasets


from torchvision import transforms


import re


import itertools


import types


class modrelu(nn.Module):

    def __init__(self, features):
        super(modrelu, self).__init__()
        self.features = features
        self.b = nn.Parameter(torch.Tensor(self.features))
        self.reset_parameters()

    def reset_parameters(self):
        self.b.data.uniform_(-0.01, 0.01)

    def forward(self, inputs):
        norm = torch.abs(inputs)
        biased_norm = norm + self.b
        magnitude = nn.functional.relu(biased_norm)
        phase = torch.sign(inputs)
        return phase * magnitude


class ExpRNNCell(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(ExpRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.recurrent_kernel = nn.Linear(hidden_size, hidden_size, bias=False)
        self.input_kernel = nn.Linear(input_size, hidden_size)
        self.nonlinearity = modrelu(hidden_size)
        if args.constraints == 'orthogonal':
            geotorch.orthogonal(self.recurrent_kernel, 'weight')
        elif args.constraints == 'lowrank':
            geotorch.low_rank(self.recurrent_kernel, 'weight', hidden_size)
        elif args.constraints == 'almostorthogonal':
            geotorch.almost_orthogonal(self.recurrent_kernel, 'weight', args.r, args.f)
        else:
            raise ValueError('Unexpected constraints. Got {}'.format(args.constraints))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.input_kernel.weight.data, nonlinearity='relu')

        def init_(x):
            x.uniform_(0.0, math.pi / 2.0)
            c = torch.cos(x.data)
            x.data = -torch.sqrt((1.0 - c) / (1.0 + c))
        K = self.recurrent_kernel
        K.weight = torus_init_(K.weight, init_=init_)

    def default_hidden(self, input_):
        return input_.new_zeros(input_.size(0), self.hidden_size, requires_grad=False)

    def forward(self, input_, hidden):
        input_ = self.input_kernel(input_)
        hidden = self.recurrent_kernel(hidden)
        out = input_ + hidden
        return self.nonlinearity(out)


n_classes = 10


class Model(nn.Module):

    def __init__(self, hidden_size, permute):
        super(Model, self).__init__()
        self.permute = permute
        if self.permute:
            self.register_buffer('permutation', torch.randperm(784))
        self.rnn = ExpRNNCell(1, hidden_size)
        self.lin = nn.Linear(hidden_size, n_classes)
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, inputs):
        if self.permute:
            inputs = inputs[:, self.permutation]
        out_rnn = self.rnn.default_hidden(inputs[:, 0, ...])
        with geotorch.parametrize.cached():
            for input in torch.unbind(inputs, dim=1):
                out_rnn = self.rnn(input.unsqueeze(dim=1), out_rnn)
        return self.lin(out_rnn)

    def loss(self, logits, y):
        return self.loss_func(logits, y)

    def correct(self, logits, y):
        return torch.eq(torch.argmax(logits, dim=1), y).float().sum()


class ProductManifold(nn.ModuleList):

    def __init__(self, manifolds):
        """
        Product manifold :math:`M_1 \\times \\dots \\times M_k`. It can be indexed like a
        regular Python list.

        .. note::

            This is an abstract manifold. It may be used by composing it on the
            left and the right by an appropriate linear immersion / submersion.
            See for example the implementation in :class:`~geotorch.LowRank`

        Args:
            manifolds (iterable): An iterable of manifolds
        """
        super().__init__(manifolds)

    def forward(self, Xs):
        return tuple(mani(X) for mani, X in zip(self, Xs))

    def right_inverse(self, Xs, check_in_manifold=True):
        return tuple(mani.right_inverse(X, check_in_manifold) for mani, X in zip(self, Xs))


class InManifoldError(ValueError):

    def __init__(self, X, M):
        super().__init__('Tensor not contained in {}. Got\n{}'.format(M, X))


def _extra_repr(**kwargs):
    if 'n' in kwargs:
        ret = 'n={}'.format(kwargs['n'])
    elif 'dim' in kwargs:
        ret = 'dim={}'.format(kwargs['dim'])
    else:
        ret = ''
    if 'k' in kwargs:
        ret += ', k={}'.format(kwargs['k'])
    if 'rank' in kwargs:
        ret += ', rank={}'.format(kwargs['rank'])
    if 'radius' in kwargs:
        ret += ', radius={}'.format(kwargs['radius'])
    if 'lam' in kwargs:
        ret += ', lambda={}'.format(kwargs['lam'])
    if 'f' in kwargs:
        ret += ', f={}'.format(kwargs['f'].__name__)
    if 'tensorial_size' in kwargs:
        ts = kwargs['tensorial_size']
        if len(ts) != 0:
            ret += ', tensorial_size={}'.format(tuple(ts))
    if 'triv' in kwargs:
        ret += ', triv={}'.format(kwargs['triv'].__name__)
    if 'no_inv' in kwargs:
        if kwargs['no_inv']:
            ret += ', no inverse'
    if 'transposed' in kwargs:
        if kwargs['transposed']:
            ret += ', transposed'
    return ret


class Rn(nn.Module):

    def __init__(self, size):
        """
        Vector space of unconstrained vectors.

        Args:
            size (torch.size): Size of the tensor to be parametrized
        """
        super().__init__()
        self.n = size[-1]
        self.tensorial_size = size[:-1]

    def forward(self, X):
        return X

    def right_inverse(self, X, check_in_manifold=True):
        if check_in_manifold and not self.in_manifold(X):
            raise InManifoldError(X, self)
        return X

    def in_manifold(self, X):
        return X.size() == self.tensorial_size + (self.n,)

    def extra_repr(self):
        return _extra_repr(n=self.n, tensorial_size=self.tensorial_size)


class NonSquareError(ValueError):

    def __init__(self, name, size):
        super().__init__('The {} parametrization can just be applied to square matrices. Got a tensor of size {}'.format(name, size))


class VectorError(ValueError):

    def __init__(self, name, size):
        super().__init__('Cannot instantiate {} on a tensor of less than 2 dimensions. Got a tensor of size {}'.format(name, size))


class Skew(nn.Module):

    def __init__(self, lower=True):
        """
        Vector space of skew-symmetric matrices, parametrized in terms of
        the upper or lower triangular part of a matrix.

        Args:
            size (torch.size): Size of the tensor to be parametrized
            lower (bool): Optional. Uses the lower triangular part of the matrix
                to parametrize the matrix. Default: ``True``
        """
        super().__init__()
        self.lower = lower

    @staticmethod
    def frame(X, lower):
        if lower:
            X = X.tril(-1)
        else:
            X = X.triu(1)
        return X - X.transpose(-2, -1)

    def forward(self, X):
        if len(X.size()) < 2:
            raise VectorError(type(self).__name__, X.size())
        if X.size(-2) != X.size(-1):
            raise NonSquareError(type(self).__name__, X.size())
        return self.frame(X, self.lower)

    @staticmethod
    def in_manifold(X):
        return X.dim() >= 2 and X.size(-2) == X.size(-1) and torch.allclose(X, -X.transpose(-2, -1))


def _has_orthonormal_columns(X, eps):
    k = X.size(-1)
    Id = torch.eye(k, dtype=X.dtype, device=X.device)
    return torch.allclose(X.transpose(-2, -1) @ X, Id, atol=eps)


def cayley_map(X):
    n = X.size(-1)
    Id = torch.eye(n, dtype=X.dtype, device=X.device)
    return torch.linalg.solve(Id.add(X, alpha=-0.5), Id.add(X, alpha=0.5))


def uniform_init_(tensor):
    """Fills in the input ``tensor`` in place with an orthogonal matrix.
    If square, the matrix will have positive determinant.
    The tensor will be distributed according to the Haar measure.
    The input tensor must have at least 2 dimensions.
    For tensors with more than 2 dimensions the first dimensions are treated as
    batch dimensions.

    Args:
        tensor (torch.Tensor): a 2-dimensional tensor or a batch of them
    """
    if tensor.ndim < 2:
        raise ValueError('Only tensors with 2 or more dimensions are supported. Got a tensor of shape {}'.format(tuple(tensor.size())))
    n, k = tensor.size()[-2:]
    transpose = n < k
    with torch.no_grad():
        x = torch.empty_like(tensor).normal_(0, 1)
        if transpose:
            x.transpose_(-2, -1)
        q, r = torch.linalg.qr(x)
        d = r.diagonal(dim1=-2, dim2=-1).sign()
        q *= d.unsqueeze(-2)
        if transpose:
            q.transpose_(-2, -1)
        if n == k:
            mask = (torch.det(q) >= 0.0).float()
            mask[mask == 0.0] = -1.0
            mask = mask.unsqueeze(-1)
            q[..., 0] *= mask
        tensor.copy_(q)
        return tensor


def _in_sphere(x, r, eps):
    norm = x.norm(dim=-1)
    rs = torch.full_like(norm, r)
    return (torch.norm(norm - rs, p=float('inf')) < eps).all()


def project(x):
    return x / x.norm(dim=-1, keepdim=True)


def uniform_init_sphere_(x, r=1.0):
    """Samples a point uniformly on the sphere into the tensor ``x``.
    If ``x`` has :math:`d > 1` dimensions, the first :math:`d-1` dimensions
    are treated as batch dimensions.
    """
    with torch.no_grad():
        x.normal_()
        x.data = r * project(x)
    return x


class SphereEmbedded(nn.Module):

    def __init__(self, size, radius=1.0):
        """
        Sphere as the orthogonal projection from
        :math:`\\mathbb{R}^n` to :math:`\\mathbb{S}^{n-1}`, that is,
        :math:`x \\mapsto \\frac{x}{\\lVert x \\rVert}`.

        Args:
            size (torch.size): Size of the tensor to be parametrized
            radius (float): Optional.
                Radius of the sphere. A positive number. Default: ``1.``
        """
        super().__init__()
        self.n = size[-1]
        self.tensorial_size = size[:-1]
        self.radius = SphereEmbedded.parse_radius(radius)

    @staticmethod
    def parse_radius(radius):
        if radius <= 0.0:
            raise ValueError('The radius has to be a positive real number. Got {}'.format(radius))
        return radius

    def forward(self, x):
        return self.radius * project(x)

    def right_inverse(self, x, check_in_manifold=True):
        if check_in_manifold and not self.in_manifold(x):
            raise InManifoldError(x, self)
        return x / self.radius

    def in_manifold(self, x, eps=1e-05):
        """
        Checks that a vector is on the sphere.

        For tensors with more than 2 dimensions the first dimensions are
        treated as batch dimensions.

        Args:
            X (torch.Tensor): The vector to be checked.
            eps (float): Optional. Threshold at which the norm is considered
                    to be equal to ``1``. Default: ``1e-5``
        """
        return _in_sphere(x, self.radius, eps)

    def sample(self):
        """
        Returns a uniformly sampled vector on the sphere.
        """
        x = torch.empty(*(self.tensorial_size + (self.n,)))
        return uniform_init_sphere_(x, r=self.radius)

    def extra_repr(self):
        return _extra_repr(n=self.n, radius=self.radius, tensorial_size=self.tensorial_size)


class sinc_class(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        ret = torch.sin(x) / x
        ret[x.abs() < 1e-45] = 1.0
        return ret

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        ret = torch.cos(x) / x - torch.sin(x) / (x * x)
        ret[x.abs() < 1e-10] = 0.0
        return ret * grad_output


sinc = sinc_class.apply


class Sphere(nn.Module):

    def __init__(self, size, radius=1.0):
        """
        Sphere as a map from the tangent space onto the sphere using the
        exponential map.

        Args:
            size (torch.size): Size of the tensor to be parametrized
            radius (float): Optional.
                Radius of the sphere. A positive number. Default: ``1.``
        """
        super().__init__()
        self.n = size[-1]
        self.tensorial_size = size[:-1]
        self.radius = Sphere.parse_radius(radius)
        self.register_buffer('base', uniform_init_sphere_(torch.empty(*size)))

    @staticmethod
    def parse_radius(radius):
        if radius <= 0.0:
            raise ValueError('The radius has to be a positive real number. Got {}'.format(radius))
        return radius

    def frame(self, x, v):
        projection = (v.unsqueeze(-2) @ x.unsqueeze(-1)).squeeze(-1)
        v = v - projection * x
        return v

    def forward(self, v):
        x = self.base
        v = self.frame(x, v)
        vnorm = v.norm(dim=-1, keepdim=True)
        return self.radius * (torch.cos(vnorm) * x + sinc(vnorm) * v)

    def right_inverse(self, x, check_in_manifold=True):
        if check_in_manifold and not self.in_manifold(x):
            raise InManifoldError(x, self)
        with torch.no_grad():
            x = x / self.radius
            self.base.copy_(x)
        return torch.zeros_like(x)

    def in_manifold(self, x, eps=1e-05):
        """
        Checks that a vector is on the sphere.

        For tensors with more than 2 dimensions the first dimensions are
        treated as batch dimensions.

        Args:
            X (torch.Tensor): The vector to be checked.
            eps (float): Optional. Threshold at which the norm is considered
                    to be equal to ``1``. Default: ``1e-5``
        """
        return _in_sphere(x, self.radius, eps)

    def sample(self):
        """
        Returns a uniformly sampled vector on the sphere.
        """
        device = self.base.device
        dtype = self.base.dtype
        x = torch.empty(*(self.tensorial_size + (self.n,)), device=device, dtype=dtype)
        return uniform_init_sphere_(x, r=self.radius)

    def extra_repr(self):
        return _extra_repr(n=self.n, radius=self.radius, tensorial_size=self.tensorial_size)


def transpose(fun):

    def new_fun(self, X, *args, **kwargs):
        if self.transposed:
            X = X.transpose(-2, -1)
        X = fun(self, X, *args, **kwargs)
        if self.transposed:
            X = X.transpose(-2, -1)
        return X
    return new_fun


class Symmetric(nn.Module):

    def __init__(self, lower=True):
        """
        Vector space of symmetric matrices, parametrized in terms of the upper or lower
        triangular part of a matrix.

        Args:
            size (torch.size): Size of the tensor to be parametrized
            lower (bool): Optional. Uses the lower triangular part of the matrix to
                parametrize the matrix. Default: ``True``
        """
        super().__init__()
        self.lower = lower

    @staticmethod
    def frame(X, lower):
        if lower:
            return X.tril(0) + X.tril(-1).transpose(-2, -1)
        else:
            return X.triu(0) + X.triu(1).transpose(-2, -1)

    def forward(self, X):
        if len(X.size()) < 2:
            raise VectorError(type(self).__name__, X.size())
        if X.size(-2) != X.size(-1):
            raise NonSquareError(type(self).__name__, X.size())
        return self.frame(X, self.lower)

    @staticmethod
    def in_manifold(X, eps=1e-06):
        return X.dim() >= 2 and X.size(-2) == X.size(-1) and torch.allclose(X, X.transpose(-2, -1), atol=eps)


class InverseError(ValueError):

    def __init__(self, M):
        super().__init__('Cannot initialize the parametrization {} as no inverse for the function {} was specified in the constructor'.format(M, M.f.__name__))


class RankError(ValueError):

    def __init__(self, n, k, rank):
        super().__init__('The rank has to be 1 <= rank <= min({}, {}). Found {}'.format(n, k, rank))


class SymF(ProductManifold):

    def __init__(self, size, rank, f, triv='expm'):
        """
        Space of the symmetric matrices of rank at most k with eigenvalues
        in the image of a given function

        Args:
            size (torch.size): Size of the tensor to be parametrized
            rank (int): Rank of the matrices.
                It has to be less or equal to
                :math:`\\min(\\texttt{size}[-1], \\texttt{size}[-2])`
            f (callable or pair of callables): Either:

                - A callable

                - A pair of callables such that the second is a (right)
                  inverse of the first
            triv (str or callable): Optional.
                A map that maps skew-symmetric matrices onto the orthogonal matrices
                surjectively. This is used to optimize the :math:`Q` in the eigenvalue
                decomposition. It can be one of ``["expm", "cayley"]`` or a custom
                callable. Default: ``"expm"``
        """
        n, tensorial_size = SymF.parse_size(size)
        if rank > n or rank < 1:
            raise RankError(n, n, rank)
        super().__init__(SymF.manifolds(n, rank, tensorial_size, triv))
        self.n = n
        self.tensorial_size = tensorial_size
        self.rank = rank
        f, inv = SymF.parse_f(f)
        self.f = f
        self.inv = inv

    @classmethod
    def parse_size(cls, size):
        if len(size) < 2:
            raise VectorError(cls.__name__, size)
        n, k = size[-2:]
        tensorial_size = size[:-2]
        if n != k:
            raise NonSquareError(cls.__name__, size)
        return n, tensorial_size

    @staticmethod
    def parse_f(f):
        if callable(f):
            return f, None
        elif isinstance(f, tuple) and callable(f[0]) and callable(f[1]):
            return f
        else:
            raise ValueError('Argument f is not callable nor a pair of callables. Found {}'.format(f))

    @staticmethod
    def manifolds(n, rank, tensorial_size, triv):
        size_q = tensorial_size + (n, rank)
        size_l = tensorial_size + (rank,)
        return Stiefel(size_q, triv=triv), Rn(size_l)

    def frame(self, X):
        L = X.diagonal(dim1=-2, dim2=-1)[..., :self.rank]
        X = X[..., :self.rank]
        return X, L

    def submersion(self, Q, L):
        L = self.f(L)
        return Q * L.unsqueeze(-2) @ Q.transpose(-2, -1)

    def forward(self, X):
        X = self.frame(X)
        Q, L = super().forward(X)
        return self.submersion(Q, L)

    def frame_inv(self, X1, X2):
        size = self.tensorial_size + (self.n, self.n)
        ret = torch.zeros(*size, dtype=X1.dtype, device=X1.device)
        with torch.no_grad():
            ret[..., :self.rank] += X1
            ret[..., :self.rank, :self.rank] += torch.diag_embed(X2)
        return ret

    def submersion_inv(self, X, check_in_manifold=True):
        with torch.no_grad():
            L, Q = torch.linalg.eigh(X)
        if check_in_manifold and not self.in_manifold_eigen(L):
            raise InManifoldError(X, self)
        if self.inv is None:
            raise InverseError(self)
        with torch.no_grad():
            Q = Q[..., -self.rank:]
            L = L[..., -self.rank:]
            L = self.inv(L)
        return L, Q

    def right_inverse(self, X, check_in_manifold=True):
        L, Q = self.submersion_inv(X, check_in_manifold)
        X1, X2 = super().right_inverse([Q, L], check_in_manifold=False)
        return self.frame_inv(X1, X2)

    def in_manifold_eigen(self, L, eps=1e-06):
        """
        Checks that an ascending ordered vector of eigenvalues is in the manifold.

        Args:
            L (torch.Tensor): Vector of eigenvalues of shape `(*, rank)`
            eps (float): Optional. Threshold at which the eigenvalues are
                considered to be zero
                Default: ``1e-6``
        """
        if L.size()[:-1] != self.tensorial_size:
            return False
        if L.size(-1) > self.rank:
            D = L[..., :-self.rank]
            infty_norm_err = D.abs().max(dim=-1).values
            if (infty_norm_err > 5.0 * eps).any():
                return False
        return (L[..., -self.rank:] >= -eps).all().item()

    def in_manifold(self, X, eps=1e-06):
        """
        Checks that a matrix is in the manifold.

        Args:
            X (torch.Tensor): The matrix or batch of matrices of shape ``(*, n, n)`` to check.
            eps (float): Optional. Threshold at which the singular values are
                    considered to be zero. Default: ``1e-6``
        """
        size = self.tensorial_size + (self.n, self.n)
        if X.size() != size or not Symmetric.in_manifold(X, eps):
            return False
        L = torch.linalg.eigvalsh(X)
        return self.in_manifold_eigen(L, eps)

    def sample(self, init_=torch.nn.init.xavier_normal_, factorized=False):
        """
        Returns a randomly sampled matrix on the manifold as

        .. math::

            WW^\\intercal \\qquad W_{i,j} \\sim \\texttt{init_}

        By default ``init\\_`` is a (xavier) normal distribution, so that the
        returned matrix follows a Wishart distribution.

        The output of this method can be used to initialize a parametrized tensor
        that has been parametrized with this or any other manifold as::

            >>> layer = nn.Linear(20, 20)
            >>> M = PSSD(layer.weight.size())
            >>> geotorch.register_parametrization(layer, "weight", M)
            >>> layer.weight = M.sample()

        Args:
            init\\_ (callable): Optional.
                    A function that takes a tensor and fills it in place according
                    to some distribution. See
                    `torch.init <https://pytorch.org/docs/stable/nn.init.html>`_.
                    Default: ``torch.nn.init.xavier_normal_``
        """
        with torch.no_grad():
            device = self[0].base.device
            dtype = self[0].base.dtype
            X = torch.empty(*(self.tensorial_size + (self.n, self.n)), device=device, dtype=dtype)
            init_(X)
            X = X @ X.transpose(-2, -1)
            L, Q = torch.linalg.eigh(X)
            L = L[..., -self.rank:]
            Q = Q[..., -self.rank:]
            if factorized:
                return L, Q
            else:
                return Q * L.unsqueeze(-2) @ Q.transpose(-2, -1)

    def extra_repr(self):
        return _extra_repr(n=self.n, rank=self.rank, tensorial_size=self.tensorial_size, f=self.f, no_inv=self.inv is None)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ProductManifold,
     lambda: ([], {'manifolds': [_mock_layer()]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Rn,
     lambda: ([], {'size': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Skew,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Sphere,
     lambda: ([], {'size': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SphereEmbedded,
     lambda: ([], {'size': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Symmetric,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (modrelu,
     lambda: ([], {'features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_lezcano_geotorch(_paritybench_base):
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

