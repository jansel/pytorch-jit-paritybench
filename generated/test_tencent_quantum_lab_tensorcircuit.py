import sys
_module = sys.modules[__name__]
del sys
darkify = _module
setup = _module
benchmark = _module
qml_benchmark = _module
qml_jd = _module
qml_pennylane = _module
qml_tc_jax = _module
qml_tc_tf = _module
qml_tfquantum = _module
utils = _module
vqe_pennylane = _module
vqe_qibo = _module
vqe_qiskit = _module
vqe_tc = _module
vqe_tfquantum = _module
toctree_filter = _module
cnconf = _module
conf = _module
generate_rst = _module
adiabatic_vqnhe = _module
_barplot = _module
batched_parameters_structures = _module
bp_benchmark = _module
bp_validation = _module
chaotic_behavior = _module
checkpoint_memsave = _module
clifford_optimization = _module
ghz_dqas = _module
gradient_benchmark = _module
hamiltonian_building = _module
hchainhamiltonian = _module
hybrid_gpu_pipeline = _module
incremental_twoqubit = _module
jacobian_cal = _module
jsonio = _module
lightcone_simplify = _module
mcnoise_boost = _module
mcnoise_check = _module
mpsvsexact = _module
noise_calibration = _module
noisy_qml = _module
noisy_sampling_jit = _module
optperformance_comparison = _module
parameter_shift = _module
qaoa_dqas = _module
qaoa_parallel_opt = _module
qaoa_shot_noise = _module
qem_dqas = _module
quantumng = _module
quditcircuit = _module
sample_benchmark = _module
sample_value_gradient = _module
simple_qaoa = _module
time_evolution = _module
universal_lr = _module
variational_dynamics = _module
variational_dynamics_generalized = _module
vmap_randomness = _module
vqe2d = _module
vqe_extra = _module
vqe_extra_mpo = _module
vqe_extra_mpo_spopt = _module
vqe_shot_noise = _module
vqeh2o_benchmark = _module
vqetfim_benchmark = _module
vqnhe_h6 = _module
setup = _module
tensorcircuit = _module
abstractcircuit = _module
applications = _module
dqas = _module
graphdata = _module
layers = _module
vags = _module
van = _module
vqes = _module
asciiart = _module
backends = _module
abstract_backend = _module
backend_factory = _module
jax_backend = _module
jax_ops = _module
numpy_backend = _module
pytorch_backend = _module
pytorch_ops = _module
tensorflow_backend = _module
tf_ops = _module
basecircuit = _module
channels = _module
circuit = _module
cons = _module
densitymatrix = _module
experimental = _module
gates = _module
interfaces = _module
numpy = _module
scipy = _module
tensorflow = _module
tensortrans = _module
keras = _module
mps_base = _module
mpscircuit = _module
noisemodel = _module
quantum = _module
results = _module
counts = _module
readout_mitigation = _module
simplify = _module
templates = _module
blocks = _module
chems = _module
dataset = _module
graphs = _module
measurements = _module
torchnn = _module
translation = _module
vis = _module
tests = _module
conftest = _module
test_backends = _module
test_calibrating = _module
test_channels = _module
test_circuit = _module
test_dmcircuit = _module
test_gates = _module
test_interfaces = _module
test_keras = _module
test_miscs = _module
test_mpscircuit = _module
test_noisemodel = _module
test_qaoa = _module
test_quantum = _module
test_quantum_attr = _module
test_results = _module
test_simplify = _module
test_templates = _module
test_torchnn = _module
test_van = _module

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


import tensorflow as tf


import torch


import uuid


import time


from functools import reduce


from functools import partial


from typing import Any


from typing import Callable


from typing import List


from typing import Optional


from typing import Sequence


from typing import Tuple


from typing import Union


import logging


from functools import wraps


from scipy import optimize


def is_sequence(x: Any) ->bool:
    if isinstance(x, list) or isinstance(x, tuple):
        return True
    return False


Array = Any


def return_partial(f: Callable[..., Any], return_argnums: Union[int, Sequence[int]]=0) ->Callable[..., Any]:
    """
    Return a callable function for output ith parts of the original output along the first axis.
    Original output supports List and Tensor.

    :Example:

    >>> from tensorcircuit.utils import return_partial
    >>> testin = np.array([[1,2],[3,4],[5,6],[7,8]])
    >>> # Method 1:
    >>> return_partial(lambda x: x, [1, 3])(testin)
    (array([3, 4]), array([7, 8]))
    >>> # Method 2:
    >>> from functools import partial
    >>> @partial(return_partial, return_argnums=(0,2))
    ... def f(inp):
    ...     return inp
    ...
    >>> f(testin)
    (array([1, 2]), array([5, 6]))

    :param f: The function to be applied this method
    :type f: Callable[..., Any]
    :param return_partial: The ith parts of original output along the first axis (axis=0 or dim=0)
    :type return_partial: Union[int, Sequence[int]]
    :return: The modified callable function
    :rtype: Callable[..., Any]
    """
    if isinstance(return_argnums, int):
        return_argnums = [return_argnums]
        one_input = True
    else:
        one_input = False

    @wraps(f)
    def wrapper(*args: Any, **kws: Any) ->Any:
        r = f(*args, **kws)
        nr = [r[ind] for ind in return_argnums]
        if one_input:
            return nr[0]
        return tuple(nr)
    return wrapper


class ExtendedBackend:
    """
    Add tensorcircuit specific backend methods, especially with their docstrings.
    """

    def copy(self: Any, a: Tensor) ->Tensor:
        """
        Return the copy of ``a``, matrix exponential.

        :param a: tensor in matrix form
        :type a: Tensor
        :return: matrix exponential of matrix ``a``
        :rtype: Tensor
        """
        raise NotImplementedError("Backend '{}' has not implemented `copy`.".format(self.name))

    def expm(self: Any, a: Tensor) ->Tensor:
        """
        Return the expm of tensor ''a''.

        :param a: tensor in matrix form
        :type a: Tensor
        :return: matrix exponential of matrix ``a``
        :rtype: Tensor
        """
        raise NotImplementedError("Backend '{}' has not implemented `expm`.".format(self.name))

    def sqrtmh(self: Any, a: Tensor) ->Tensor:
        """
        Return the sqrtm of a Hermitian matrix ``a``.

        :param a: tensor in matrix form
        :type a: Tensor
        :return: sqrtm of ``a``
        :rtype: Tensor
        """
        e, v = self.eigh(a)
        e = self.sqrt(e)
        return v @ self.diagflat(e) @ self.adjoint(v)

    def eigvalsh(self: Any, a: Tensor) ->Tensor:
        """
        Get the eigenvalues of matrix ``a``.

        :param a: tensor in matrix form
        :type a: Tensor
        :return: eigenvalues of ``a``
        :rtype: Tensor
        """
        raise NotImplementedError("Backend '{}' has not implemented `eigvalsh`.".format(self.name))

    def sin(self: Any, a: Tensor) ->Tensor:
        """
        Return the  elementwise sine of a tensor ``a``.

        :param a: tensor in matrix form
        :type a: Tensor
        :return: sine of ``a``
        :rtype: Tensor
        """
        raise NotImplementedError("Backend '{}' has not implemented `sin`.".format(self.name))

    def cos(self: Any, a: Tensor) ->Tensor:
        """
        Return the cosine of a tensor ``a``.

        :param a: tensor in matrix form
        :type a: Tensor
        :return: cosine of ``a``
        :rtype: Tensor
        """
        raise NotImplementedError("Backend '{}' has not implemented `cos`.".format(self.name))

    def acos(self: Any, a: Tensor) ->Tensor:
        """
        Return the acos of a tensor ``a``.

        :param a: tensor in matrix form
        :type a: Tensor
        :return: acos of ``a``
        :rtype: Tensor
        """
        raise NotImplementedError("Backend '{}' has not implemented `acos`.".format(self.name))

    def acosh(self: Any, a: Tensor) ->Tensor:
        """
        Return the acosh of a tensor ``a``.

        :param a: tensor in matrix form
        :type a: Tensor
        :return: acosh of ``a``
        :rtype: Tensor
        """
        raise NotImplementedError("Backend '{}' has not implemented `acosh`.".format(self.name))

    def asin(self: Any, a: Tensor) ->Tensor:
        """
        Return the acos of a tensor ``a``.

        :param a: tensor in matrix form
        :type a: Tensor
        :return: asin of ``a``
        :rtype: Tensor
        """
        raise NotImplementedError("Backend '{}' has not implemented `asin`.".format(self.name))

    def asinh(self: Any, a: Tensor) ->Tensor:
        """
        Return the asinh of a tensor ``a``.

        :param a: tensor in matrix form
        :type a: Tensor
        :return: asinh of ``a``
        :rtype: Tensor
        """
        raise NotImplementedError("Backend '{}' has not implemented `asinh`.".format(self.name))

    def atan(self: Any, a: Tensor) ->Tensor:
        """
        Return the atan of a tensor ``a``.

        :param a: tensor in matrix form
        :type a: Tensor
        :return: atan of ``a``
        :rtype: Tensor
        """
        raise NotImplementedError("Backend '{}' has not implemented `atan`.".format(self.name))

    def atan2(self: Any, y: Tensor, x: Tensor) ->Tensor:
        """
        Return the atan of a tensor ``y``/``x``.

        :param a: tensor in matrix form
        :type a: Tensor
        :return: atan2 of ``a``
        :rtype: Tensor
        """
        raise NotImplementedError("Backend '{}' has not implemented `atan2`.".format(self.name))

    def atanh(self: Any, a: Tensor) ->Tensor:
        """
        Return the atanh of a tensor ``a``.

        :param a: tensor in matrix form
        :type a: Tensor
        :return: atanh of ``a``
        :rtype: Tensor
        """
        raise NotImplementedError("Backend '{}' has not implemented `atanh`.".format(self.name))

    def cosh(self: Any, a: Tensor) ->Tensor:
        """
        Return the cosh of a tensor ``a``.

        :param a: tensor in matrix form
        :type a: Tensor
        :return: cosh of ``a``
        :rtype: Tensor
        """
        raise NotImplementedError("Backend '{}' has not implemented `cosh`.".format(self.name))

    def tan(self: Any, a: Tensor) ->Tensor:
        """
        Return the tan of a tensor ``a``.

        :param a: tensor in matrix form
        :type a: Tensor
        :return: tan of ``a``
        :rtype: Tensor
        """
        raise NotImplementedError("Backend '{}' has not implemented `tan`.".format(self.name))

    def tanh(self: Any, a: Tensor) ->Tensor:
        """
        Return the tanh of a tensor ``a``.

        :param a: tensor in matrix form
        :type a: Tensor
        :return: tanh of ``a``
        :rtype: Tensor
        """
        raise NotImplementedError("Backend '{}' has not implemented `tanh`.".format(self.name))

    def sinh(self: Any, a: Tensor) ->Tensor:
        """
        Return the sinh of a tensor ``a``.

        :param a: tensor in matrix form
        :type a: Tensor
        :return: sinh of ``a``
        :rtype: Tensor
        """
        raise NotImplementedError("Backend '{}' has not implemented `sinh`.".format(self.name))

    def abs(self: Any, a: Tensor) ->Tensor:
        """
        Return the elementwise abs value of a matrix ``a``.

        :param a: tensor in matrix form
        :type a: Tensor
        :return: abs of ``a``
        :rtype: Tensor
        """
        raise NotImplementedError("Backend '{}' has not implemented `abs`.".format(self.name))

    def kron(self: Any, a: Tensor, b: Tensor) ->Tensor:
        """
        Return the kronecker product of two matrices ``a`` and ``b``.

        :param a: tensor in matrix form
        :type a: Tensor
        :param b: tensor in matrix form
        :type b: Tensor
        :return: kronecker product of ``a`` and ``b``
        :rtype: Tensor
        """
        raise NotImplementedError("Backend '{}' has not implemented `kron`.".format(self.name))

    def size(self: Any, a: Tensor) ->Tensor:
        """
        Return the total number of elements in ``a`` in tensor form.

        :param a: tensor
        :type a: Tensor
        :return: the total number of elements in ``a``
        :rtype: Tensor
        """
        raise NotImplementedError("Backend '{}' has not implemented `size`.".format(self.name))

    def sizen(self: Any, a: Tensor) ->int:
        """
        Return the total number of elements in tensor ``a``, but in integer form.

        :param a: tensor
        :type a: Tensor
        :return: the total number of elements in tensor ``a``
        :rtype: int
        """
        return reduce(mul, list(a.shape) + [1])

    def numpy(self: Any, a: Tensor) ->Tensor:
        """
        Return the numpy array of a tensor ``a``, but may not work in a jitted function.

        :param a: tensor in matrix form
        :type a: Tensor
        :return: numpy array of ``a``
        :rtype: Tensor
        """
        raise NotImplementedError("Backend '{}' has not implemented `numpy`.".format(self.name))

    def real(self: Any, a: Tensor) ->Tensor:
        """
        Return the elementwise real value of a tensor ``a``.

        :param a: tensor
        :type a: Tensor
        :return: real value of ``a``
        :rtype: Tensor
        """
        raise NotImplementedError("Backend '{}' has not implemented `real`.".format(self.name))

    def imag(self: Any, a: Tensor) ->Tensor:
        """
        Return the elementwise imaginary value of a tensor ``a``.

        :param a: tensor
        :type a: Tensor
        :return: imaginary value of ``a``
        :rtype: Tensor
        """
        raise NotImplementedError("Backend '{}' has not implemented `imag`.".format(self.name))

    def adjoint(self: Any, a: Tensor) ->Tensor:
        """
        Return the conjugate and transpose of a tensor ``a``

        :param a: Input tensor
        :type a: Tensor
        :return: adjoint tensor of ``a``
        :rtype: Tensor
        """
        return self.conj(self.transpose(a))

    def i(self: Any, dtype: str) ->Tensor:
        """
        Return 1.j in as a tensor compatible with the backend.

        :param dtype: "complex64" or "complex128"
        :type dtype: str
        :return: 1.j tensor
        :rtype: Tensor
        """
        raise NotImplementedError("Backend '{}' has not implemented `i`.".format(self.name))

    def reshape2(self: Any, a: Tensor) ->Tensor:
        """
        Reshape a tensor to the [2, 2, ...] shape.

        :param a: Input tensor
        :type a: Tensor
        :return: the reshaped tensor
        :rtype: Tensor
        """
        nleg = int(np.log2(self.sizen(a)))
        a = self.reshape(a, [(2) for _ in range(nleg)])
        return a

    def reshapem(self: Any, a: Tensor) ->Tensor:
        """
        Reshape a tensor to the [l, l] shape.

        :param a: Input tensor
        :type a: Tensor
        :return: the reshaped tensor
        :rtype: Tensor
        """
        l = int(np.sqrt(self.sizen(a)))
        a = self.reshape(a, [l, l])
        return a

    def dtype(self: Any, a: Tensor) ->str:
        """
        Obtain dtype string for tensor ``a``

        :param a: The tensor
        :type a: Tensor
        :return: dtype str, such as "complex64"
        :rtype: str
        """
        raise NotImplementedError("Backend '{}' has not implemented `dtype`.".format(self.name))

    def stack(self: Any, a: Sequence[Tensor], axis: int=0) ->Tensor:
        """
        Concatenates a sequence of tensors ``a`` along a new dimension ``axis``.

        :param a: List of tensors in the same shape
        :type a: Sequence[Tensor]
        :param axis: the stack axis, defaults to 0
        :type axis: int, optional
        :return: concatenated tensor
        :rtype: Tensor
        """
        raise NotImplementedError("Backend '{}' has not implemented `stack`.".format(self.name))

    def concat(self: Any, a: Sequence[Tensor], axis: int=0) ->Tensor:
        """
        Join a sequence of arrays along an existing axis.

        :param a: [description]
        :type a: Sequence[Tensor]
        :param axis: [description], defaults to 0
        :type axis: int, optional
        """
        raise NotImplementedError("Backend '{}' has not implemented `concat`.".format(self.name))

    def tile(self: Any, a: Tensor, rep: Tensor) ->Tensor:
        """
        Constructs a tensor by tiling a given tensor.

        :param a: [description]
        :type a: Tensor
        :param rep: 1d tensor with length the same as the rank of ``a``
        :type rep: Tensor
        :return: [description]
        :rtype: Tensor
        """
        raise NotImplementedError("Backend '{}' has not implemented `tile`.".format(self.name))

    def mean(self: Any, a: Tensor, axis: Optional[Sequence[int]]=None, keepdims: bool=False) ->Tensor:
        """
        Compute the arithmetic mean for ``a`` along the specified ``axis``.

        :param a: tensor to take average
        :type a: Tensor
        :param axis: the axis to take mean, defaults to None indicating sum over flatten array
        :type axis: Optional[Sequence[int]], optional
        :param keepdims: _description_, defaults to False
        :type keepdims: bool, optional
        :return: _description_
        :rtype: Tensor
        """
        raise NotImplementedError("Backend '{}' has not implemented `mean`.".format(self.name))

    def std(self: Any, a: Tensor, axis: Optional[Sequence[int]]=None, keepdims: bool=False) ->Tensor:
        """
        Compute the standard deviation along the specified axis.

        :param a: _description_
        :type a: Tensor
        :param axis: Axis or axes along which the standard deviation is computed,
            defaults to None, implying all axis
        :type axis: Optional[Sequence[int]], optional
        :param keepdims: If this is set to True,
            the axes which are reduced are left in the result as dimensions with size one,
            defaults to False
        :type keepdims: bool, optional
        :return: _description_
        :rtype: Tensor
        """
        raise NotImplementedError("Backend '{}' has not implemented `std`.".format(self.name))

    def min(self: Any, a: Tensor, axis: Optional[int]=None) ->Tensor:
        """
        Return the minimum of an array or minimum along an axis.

        :param a: [description]
        :type a: Tensor
        :param axis: [description], defaults to None
        :type axis: Optional[int], optional
        :return: [description]
        :rtype: Tensor
        """
        raise NotImplementedError("Backend '{}' has not implemented `min`.".format(self.name))

    def max(self: Any, a: Tensor, axis: Optional[int]=None) ->Tensor:
        """
        Return the maximum of an array or maximum along an axis.

        :param a: [description]
        :type a: Tensor
        :param axis: [description], defaults to None
        :type axis: Optional[int], optional
        :return: [description]
        :rtype: Tensor
        """
        raise NotImplementedError("Backend '{}' has not implemented `max`.".format(self.name))

    def argmax(self: Any, a: Tensor, axis: int=0) ->Tensor:
        """
        Return the index of maximum of an array an axis.

        :param a: [description]
        :type a: Tensor
        :param axis: [description], defaults to 0, different behavior from numpy defaults!
        :type axis: int
        :return: [description]
        :rtype: Tensor
        """
        raise NotImplementedError("Backend '{}' has not implemented `argmax`.".format(self.name))

    def argmin(self: Any, a: Tensor, axis: int=0) ->Tensor:
        """
        Return the index of minimum of an array an axis.

        :param a: [description]
        :type a: Tensor
        :param axis: [description], defaults to 0, different behavior from numpy defaults!
        :type axis: int
        :return: [description]
        :rtype: Tensor
        """
        raise NotImplementedError("Backend '{}' has not implemented `argmin`.".format(self.name))

    def unique_with_counts(self: Any, a: Tensor, **kws: Any) ->Tuple[Tensor, Tensor]:
        """
        Find the unique elements and their corresponding counts of the given tensor ``a``.

        :param a: [description]
        :type a: Tensor
        :return: Unique elements, corresponding counts
        :rtype: Tuple[Tensor, Tensor]
        """
        raise NotImplementedError("Backend '{}' has not implemented `unique_with_counts`.".format(self.name))

    def sigmoid(self: Any, a: Tensor) ->Tensor:
        """
        Compute sigmoid of input ``a``

        :param a: [description]
        :type a: Tensor
        :return: [description]
        :rtype: Tensor
        """
        raise NotImplementedError("Backend '{}' has not implemented `sigmoid`.".format(self.name))

    def relu(self: Any, a: Tensor) ->Tensor:
        """
        Rectified linear unit activation function.
        Computes the element-wise function:

        .. math ::

            \\mathrm{relu}(x)=\\max(x,0)


        :param a: Input tensor
        :type a: Tensor
        :return: Tensor after relu
        :rtype: Tensor
        """
        raise NotImplementedError("Backend '{}' has not implemented `relu`.".format(self.name))

    def softmax(self: Any, a: Sequence[Tensor], axis: Optional[int]=None) ->Tensor:
        """
        Softmax function.
        Computes the function which rescales elements to the range [0,1] such that the elements along axis sum to 1.

        .. math ::

            \\mathrm{softmax}(x) = \\frac{\\exp(x_i)}{\\sum_j \\exp(x_j)}


        :param a: Tensor
        :type a: Sequence[Tensor]
        :param axis: A dimension along which Softmax will be computed , defaults to None for all axis sum.
        :type axis: int, optional
        :return: concatenated tensor
        :rtype: Tensor
        """
        raise NotImplementedError("Backend '{}' has not implemented `softmax`.".format(self.name))

    def onehot(self: Any, a: Tensor, num: int) ->Tensor:
        """
        One-hot encodes the given ``a``.
        Each index in the input ``a`` is encoded as a vector of zeros of length ``num``
        with the element at index set to one:

        :param a: input tensor
        :type a: Tensor
        :param num: number of features in onehot dimension
        :type num: int
        :return: onehot tensor with the last extra dimension
        :rtype: Tensor
        """
        raise NotImplementedError("Backend '{}' has not implemented `onehot`.".format(self.name))

    def one_hot(self: Any, a: Tensor, num: int) ->Tensor:
        """
        See doc for :py:meth:`onehot`
        """
        return self.onehot(a, num)

    def cumsum(self: Any, a: Tensor, axis: Optional[int]=None) ->Tensor:
        """
        Return the cumulative sum of the elements along a given axis.

        :param a: [description]
        :type a: Tensor
        :param axis: The default behavior is the same as numpy, different from tf/torch
            as cumsum of the flatten 1D array, defaults to None
        :type axis: Optional[int], optional
        :return: [description]
        :rtype: Tensor
        """
        raise NotImplementedError("Backend '{}' has not implemented `cumsum`.".format(self.name))

    def is_tensor(self: Any, a: Tensor) ->bool:
        """
        Return a boolean on whether ``a`` is a tensor in backend package.

        :param a: a tensor to be determined
        :type a: Tensor
        :return: whether ``a`` is a tensor
        :rtype: bool
        """
        raise NotImplementedError("Backend '{}' has not implemented `is_tensor`.".format(self.name))

    def cast(self: Any, a: Tensor, dtype: str) ->Tensor:
        """
        Cast the tensor dtype of a ``a``.

        :param a: tensor
        :type a: Tensor
        :param dtype: "float32", "float64", "complex64", "complex128"
        :type dtype: str
        :return: ``a`` of new dtype
        :rtype: Tensor
        """
        raise NotImplementedError("Backend '{}' has not implemented `cast`.".format(self.name))

    def mod(self: Any, x: Tensor, y: Tensor) ->Tensor:
        """
        Compute y-mod of x (negative number behavior is not guaranteed to be consistent)

        :param x: input values
        :type x: Tensor
        :param y: mod ``y``
        :type y: Tensor
        :return: results
        :rtype: Tensor
        """
        raise NotImplementedError("Backend '{}' has not implemented `mod`.".format(self.name))

    def reverse(self: Any, a: Tensor) ->Tensor:
        """
        return ``a[::-1]``, only 1D tensor is guaranteed for consistent behavior

        :param a: 1D tensor
        :type a: Tensor
        :return: 1D tensor in reverse order
        :rtype: Tensor
        """
        return a[::-1]

    def right_shift(self: Any, x: Tensor, y: Tensor) ->Tensor:
        """
        Shift the bits of an integer x to the right y bits.

        :param x: input values
        :type x: Tensor
        :param y: Number of bits shift to ``x``
        :type y: Tensor
        :return: result with the same shape as ``x``
        :rtype: Tensor
        """
        raise NotImplementedError("Backend '{}' has not implemented `right_shift`.".format(self.name))

    def left_shift(self: Any, x: Tensor, y: Tensor) ->Tensor:
        """
        Shift the bits of an integer x to the left y bits.

        :param x: input values
        :type x: Tensor
        :param y: Number of bits shift to ``x``
        :type y: Tensor
        :return: result with the same shape as ``x``
        :rtype: Tensor
        """
        raise NotImplementedError("Backend '{}' has not implemented `left_shift`.".format(self.name))

    def arange(self: Any, start: int, stop: Optional[int]=None, step: int=1) ->Tensor:
        """
        Values are generated within the half-open interval [start, stop)

        :param start: start index
        :type start: int
        :param stop: end index, defaults to None
        :type stop: Optional[int], optional
        :param step: steps, defaults to 1
        :type step: Optional[int], optional
        :return: _description_
        :rtype: Tensor
        """
        raise NotImplementedError("Backend '{}' has not implemented `arange`.".format(self.name))

    def solve(self: Any, A: Tensor, b: Tensor, **kws: Any) ->Tensor:
        """
        Solve the linear system Ax=b and return the solution x.

        :param A: The multiplied matrix.
        :type A: Tensor
        :param b: The resulted matrix.
        :type b: Tensor
        :return: The solution of the linear system.
        :rtype: Tensor
        """
        raise NotImplementedError("Backend '{}' has not implemented `solve`.".format(self.name))

    def searchsorted(self: Any, a: Tensor, v: Tensor, side: str='left') ->Tensor:
        """
        Find indices where elements should be inserted to maintain order.

        :param a: input array sorted in ascending order
        :type a: Tensor
        :param v: value to inserted
        :type v: Tensor
        :param side:  If ‘left’, the index of the first suitable location found is given.
            If ‘right’, return the last such index.
            If there is no suitable index, return either 0 or N (where N is the length of a),
            defaults to "left"
        :type side: str, optional
        :return: Array of insertion points with the same shape as v, or an integer if v is a scalar.
        :rtype: Tensor
        """
        raise NotImplementedError("Backend '{}' has not implemented `searchsorted`.".format(self.name))

    def tree_map(self: Any, f: Callable[..., Any], *pytrees: Any) ->Any:
        """
        Return the new tree map with multiple arg function ``f`` through pytrees.

        :param f: The function
        :type f: Callable[..., Any]
        :param pytrees: inputs as any python structure
        :type pytrees: Any
        :raises NotImplementedError: raise when neither tensorflow or jax is installed.
        :return: The new tree map with the same structure but different values.
        :rtype: Any
        """
        try:
            import tensorflow as tf
        except ImportError:
            raise NotImplementedError('No installed ML backend for `tree_map`')
        return tf.nest.map_structure(f, *pytrees)

    def tree_flatten(self: Any, pytree: Any) ->Tuple[Any, Any]:
        """
        Flatten python structure to 1D list

        :param pytree: python structure to be flattened
        :type pytree: Any
        :return: The 1D list of flattened structure and treedef
            which can be used for later unflatten
        :rtype: Tuple[Any, Any]
        """
        try:
            import tensorflow as tf
        except ImportError:
            raise NotImplementedError('No installed ML backend for `tree_flatten`')
        leaves = tf.nest.flatten(pytree)
        treedef = pytree
        return leaves, treedef

    def tree_unflatten(self: Any, treedef: Any, leaves: Any) ->Any:
        """
        Pack 1D list to pytree defined via ``treedef``

        :param treedef: Def of pytree structure, the second return from ``tree_flatten``
        :type treedef: Any
        :param leaves: the 1D list of flattened data structure
        :type leaves: Any
        :return: Packed pytree
        :rtype: Any
        """
        try:
            import tensorflow as tf
        except ImportError:
            raise NotImplementedError('No installed ML backend for `tree_unflatten`')
        return tf.nest.pack_sequence_as(treedef, leaves)

    def to_dlpack(self: Any, a: Tensor) ->Any:
        """
        Transform the tensor ``a`` as a dlpack capsule

        :param a: _description_
        :type a: Tensor
        :return: _description_
        :rtype: Any
        """
        raise NotImplementedError("Backend '{}' has not implemented `to_dlpack`.".format(self.name))

    def from_dlpack(self: Any, a: Any) ->Tensor:
        """
        Transform a dlpack capsule to a tensor

        :param a: the dlpack capsule
        :type a: Any
        :return: _description_
        :rtype: Tensor
        """
        raise NotImplementedError("Backend '{}' has not implemented `from_dlpack`.".format(self.name))

    def set_random_state(self: Any, seed: Optional[int]=None, get_only: bool=False) ->Any:
        """
        Set the random state attached to the backend.

        :param seed: the random seed, defaults to be None
        :type seed: Optional[int], optional
        :param get_only: If set to be true, only get the random state in return
            instead of setting the state on the backend
        :type get_only: bool, defaults to be False
        """
        raise NotImplementedError("Backend '{}' has not implemented `set_random_state`.".format(self.name))

    def get_random_state(self: Any, seed: Optional[int]=None) ->Any:
        """
        Get the backend specific random state object.

        :param seed: [description], defaults to be None
        :type seed: Optional[int], optional
        :return:the backend specific random state object
        :rtype: Any
        """
        return self.set_random_state(seed, True)

    def random_split(self: Any, key: Any) ->Tuple[Any, Any]:
        """
        A jax like split API, but it doesn't split the key generator for other backends.
        It is just for a consistent interface of random code;
        make sure you know what the function actually does.
        This function is mainly a utility to write backend agnostic code instead of doing magic things.

        :param key: [description]
        :type key: Any
        :return: [description]
        :rtype: Tuple[Any, Any]
        """
        return key, key

    def implicit_randn(self: Any, shape: Union[int, Sequence[int]]=1, mean: float=0, stddev: float=1, dtype: str='32') ->Tensor:
        """
        Call the random normal function with the random state management behind the scene.

        :param shape: [description], defaults to 1
        :type shape: Union[int, Sequence[int]], optional
        :param mean: [description], defaults to 0
        :type mean: float, optional
        :param stddev: [description], defaults to 1
        :type stddev: float, optional
        :param dtype: [description], defaults to "32"
        :type dtype: str, optional
        :return: [description]
        :rtype: Tensor
        """
        g = getattr(self, 'g', None)
        if g is None:
            self.set_random_state()
            g = getattr(self, 'g', None)
        r = self.stateful_randn(g, shape, mean, stddev, dtype)
        return r

    def stateful_randn(self: Any, g: Any, shape: Union[int, Sequence[int]]=1, mean: float=0, stddev: float=1, dtype: str='32') ->Tensor:
        """
        [summary]

        :param self: [description]
        :type self: Any
        :param g: stateful register for each package
        :type g: Any
        :param shape: shape of output sampling tensor
        :type shape: Union[int, Sequence[int]]
        :param mean: [description], defaults to 0
        :type mean: float, optional
        :param stddev: [description], defaults to 1
        :type stddev: float, optional
        :param dtype: only real data type is supported, "32" or "64", defaults to "32"
        :type dtype: str, optional
        :return: [description]
        :rtype: Tensor
        """
        raise NotImplementedError("Backend '{}' has not implemented `stateful_randn`.".format(self.name))

    def implicit_randu(self: Any, shape: Union[int, Sequence[int]]=1, low: float=0, high: float=1, dtype: str='32') ->Tensor:
        """
        Call the random normal function with the random state management behind the scene.

        :param shape: [description], defaults to 1
        :type shape: Union[int, Sequence[int]], optional
        :param mean: [description], defaults to 0
        :type mean: float, optional
        :param stddev: [description], defaults to 1
        :type stddev: float, optional
        :param dtype: [description], defaults to "32"
        :type dtype: str, optional
        :return: [description]
        :rtype: Tensor
        """
        g = getattr(self, 'g', None)
        if g is None:
            self.set_random_state()
            g = getattr(self, 'g', None)
        r = self.stateful_randu(g, shape, low, high, dtype)
        return r

    def stateful_randu(self: Any, g: Any, shape: Union[int, Sequence[int]]=1, low: float=0, high: float=1, dtype: str='32') ->Tensor:
        """
        Uniform random sampler from ``low`` to ``high``.

        :param g: stateful register for each package
        :type g: Any
        :param shape: shape of output sampling tensor, defaults to 1
        :type shape: Union[int, Sequence[int]], optional
        :param low: [description], defaults to 0
        :type low: float, optional
        :param high: [description], defaults to 1
        :type high: float, optional
        :param dtype: only real data type is supported, "32" or "64", defaults to "32"
        :type dtype: str, optional
        :return: [description]
        :rtype: Tensor
        """
        raise NotImplementedError("Backend '{}' has not implemented `stateful_randu`.".format(self.name))

    def implicit_randc(self: Any, a: Union[int, Sequence[int], Tensor], shape: Union[int, Sequence[int]], p: Optional[Union[Sequence[float], Tensor]]=None) ->Tensor:
        """
        [summary]

        :param g: [description]
        :type g: Any
        :param a: The possible options
        :type a: Union[int, Sequence[int], Tensor]
        :param shape: Sampling output shape
        :type shape: Union[int, Sequence[int]]
        :param p: probability for each option in a, defaults to None, as equal probability distribution
        :type p: Optional[Union[Sequence[float], Tensor]], optional
        :return: [description]
        :rtype: Tensor
        """
        g = getattr(self, 'g', None)
        if g is None:
            self.set_random_state()
            g = getattr(self, 'g', None)
        r = self.stateful_randc(g, a, shape, p)
        return r

    def stateful_randc(self: Any, g: Any, a: Union[int, Sequence[int], Tensor], shape: Union[int, Sequence[int]], p: Optional[Union[Sequence[float], Tensor]]=None) ->Tensor:
        """
        [summary]

        :param g: [description]
        :type g: Any
        :param a: The possible options
        :type a: Union[int, Sequence[int], Tensor]
        :param shape: Sampling output shape
        :type shape: Union[int, Sequence[int]]
        :param p: probability for each option in a, defaults to None, as equal probability distribution
        :type p: Optional[Union[Sequence[float], Tensor]], optional
        :return: [description]
        :rtype: Tensor
        """
        raise NotImplementedError("Backend '{}' has not implemented `stateful_randc`.".format(self.name))

    def probability_sample(self: Any, shots: int, p: Tensor, status: Optional[Tensor]=None, g: Any=None) ->Tensor:
        """
        Drawn ``shots`` samples from probability distribution p, given the external randomness
        determined by uniform distributed ``status`` tensor or backend random generator ``g``.
        This method is similar with ``stateful_randc``, but it supports ``status`` beyond ``g``,
        which is convenient when jit or vmap

        :param shots: Number of samples to draw with replacement
        :type shots: int
        :param p: prbability vector
        :type p: Tensor
        :param status: external randomness as a tensor with each element drawn uniformly from [0, 1],
            defaults to None
        :type status: Optional[Tensor], optional
        :param g: backend random genrator, defaults to None
        :type g: Any, optional
        :return: The drawn sample as an int tensor
        :rtype: Tensor
        """
        if status is not None:
            status = self.convert_to_tensor(status)
        elif g is not None:
            status = self.stateful_randu(g, shape=[shots])
        else:
            status = self.implicit_randu(shape=[shots])
        p = p / self.sum(p)
        p_cuml = self.cumsum(p)
        r = p_cuml[-1] * (1 - self.cast(status, p.dtype))
        ind = self.searchsorted(p_cuml, r)
        a = self.arange(self.shape_tuple(p)[0])
        res = self.gather1d(a, ind)
        return res

    def gather1d(self: Any, operand: Tensor, indices: Tensor) ->Tensor:
        """
        Return ``operand[indices]``, both ``operand`` and ``indices`` are rank-1 tensor.

        :param operand: rank-1 tensor
        :type operand: Tensor
        :param indices: rank-1 tensor with int dtype
        :type indices: Tensor
        :return: ``operand[indices]``
        :rtype: Tensor
        """
        return operand[indices]

    def scatter(self: Any, operand: Tensor, indices: Tensor, updates: Tensor) ->Tensor:
        """
        Roughly equivalent to operand[indices] = updates, indices only support shape with rank 2 for now.

        :param operand: [description]
        :type operand: Tensor
        :param indices: [description]
        :type indices: Tensor
        :param updates: [description]
        :type updates: Tensor
        :return: [description]
        :rtype: Tensor
        """
        raise NotImplementedError("Backend '{}' has not implemented `scatter`.".format(self.name))

    def coo_sparse_matrix(self: Any, indices: Tensor, values: Tensor, shape: Tensor) ->Tensor:
        """
        Generate the coo format sparse matrix from indices and values,
        which is the only sparse format supported in different ML backends.

        :param indices: shape [n, 2] for n non zero values in the returned matrix
        :type indices: Tensor
        :param values: shape [n]
        :type values: Tensor
        :param shape: Tuple[int, ...]
        :type shape: Tensor
        :return: [description]
        :rtype: Tensor
        """
        raise NotImplementedError("Backend '{}' has not implemented `coo`.".format(self.name))

    def coo_sparse_matrix_from_numpy(self: Any, a: Tensor) ->Tensor:
        """
        Generate the coo format sparse matrix from scipy coo sparse matrix.

        :param a: Scipy coo format sparse matrix
        :type a: Tensor
        :return: SparseTensor in backend format
        :rtype: Tensor
        """
        return self.coo_sparse_matrix(indices=np.array([a.row, a.col]).T, values=a.data, shape=a.shape)

    def sparse_dense_matmul(self: Any, sp_a: Tensor, b: Tensor) ->Tensor:
        """
        A sparse matrix multiplies a dense matrix.

        :param sp_a: a sparse matrix
        :type sp_a: Tensor
        :param b: a dense matrix
        :type b: Tensor
        :return: dense matrix
        :rtype: Tensor
        """
        raise NotImplementedError("Backend '{}' has not implemented `sparse_dense_matmul`.".format(self.name))

    def to_dense(self: Any, sp_a: Tensor) ->Tensor:
        """
        Convert a sparse matrix to dense tensor.

        :param sp_a: a sparse matrix
        :type sp_a: Tensor
        :return: the resulted dense matrix
        :rtype: Tensor
        """
        raise NotImplementedError("Backend '{}' has not implemented `to_dense`.".format(self.name))

    def is_sparse(self: Any, a: Tensor) ->bool:
        """
        Determine whether the type of input ``a`` is  ``sparse``.

        :param a: input matrix ``a``
        :type a: Tensor
        :return: a bool indicating whether the matrix ``a`` is sparse
        :rtype: bool
        """
        raise NotImplementedError("Backend '{}' has not implemented `is_sparse`.".format(self.name))

    def device(self: Any, a: Tensor) ->str:
        """
        get the universal device str for the tensor, in the format of tf

        :param a: the tensor
        :type a: Tensor
        :return: device str where the tensor lives on
        :rtype: str
        """
        raise NotImplementedError("Backend '{}' has not implemented `device`.".format(self.name))

    def device_move(self: Any, a: Tensor, dev: Any) ->Tensor:
        """
        move tensor ``a`` to device ``dev``

        :param a: the tensor
        :type a: Tensor
        :param dev: device str or device obj in corresponding backend
        :type dev: Any
        :return: the tensor on new device
        :rtype: Tensor
        """
        raise NotImplementedError("Backend '{}' has not implemented `device_move`.".format(self.name))

    def _dev2str(self: Any, dev: Any) ->str:
        """
        device object to universal dev str

        :param dev: device object
        :type dev: Any
        :return: dev str
        :rtype: str
        """
        raise NotImplementedError("Backend '{}' has not implemented `_dev2str`.".format(self.name))

    def _str2dev(self: Any, str_: str) ->Any:
        """
        device object to universal dev str

        :param str_: dev str
        :type str_: str
        :return: device object
        :rtype: Any
        """
        raise NotImplementedError("Backend '{}' has not implemented `_str2dev`.".format(self.name))

    def cond(self: Any, pred: bool, true_fun: Callable[[], Tensor], false_fun: Callable[[], Tensor]) ->Tensor:
        """
        The native cond for XLA compiling, wrapper for ``tf.cond`` and limited functionality of ``jax.lax.cond``.

        :param pred: [description]
        :type pred: bool
        :param true_fun: [description]
        :type true_fun: Callable[[], Tensor]
        :param false_fun: [description]
        :type false_fun: Callable[[], Tensor]
        :return: [description]
        :rtype: Tensor
        """
        raise NotImplementedError("Backend '{}' has not implemented `cond`.".format(self.name))

    def switch(self: Any, index: Tensor, branches: Sequence[Callable[[], Tensor]]) ->Tensor:
        """
        ``branches[index]()``

        :param index: [description]
        :type index: Tensor
        :param branches: [description]
        :type branches: Sequence[Callable[[], Tensor]]
        :return: [description]
        :rtype: Tensor
        """
        raise NotImplementedError("Backend '{}' has not implemented `switch`.".format(self.name))

    def stop_gradient(self: Any, a: Tensor) ->Tensor:
        """
        Stop backpropagation from ``a``.

        :param a: [description]
        :type a: Tensor
        :return: [description]
        :rtype: Tensor
        """
        raise NotImplementedError("Backend '{}' has not implemented `stop_gradient`.".format(self.name))

    def grad(self: Any, f: Callable[..., Any], argnums: Union[int, Sequence[int]]=0, has_aux: bool=False) ->Callable[..., Any]:
        """
        Return the function which is the grad function of input ``f``.

        :Example:

        >>> f = lambda x,y: x**2+2*y
        >>> g = tc.backend.grad(f)
        >>> g(tc.num_to_tensor(1),tc.num_to_tensor(2))
        2
        >>> g = tc.backend.grad(f, argnums=(0,1))
        >>> g(tc.num_to_tensor(1),tc.num_to_tensor(2))
        [2, 2]

        :param f: the function to be differentiated
        :type f: Callable[..., Any]
        :param argnums: the position of args in ``f`` that are to be differentiated, defaults to be 0
        :type argnums: Union[int, Sequence[int]], optional
        :return: the grad function of ``f`` with the same set of arguments as ``f``
        :rtype: Callable[..., Any]
        """
        raise NotImplementedError("Backend '{}' has not implemented `grad`.".format(self.name))

    def value_and_grad(self: Any, f: Callable[..., Any], argnums: Union[int, Sequence[int]]=0, hax_aux: bool=False) ->Callable[..., Tuple[Any, Any]]:
        """
        Return the function which returns the value and grad of ``f``.

        :Example:

        >>> f = lambda x,y: x**2+2*y
        >>> g = tc.backend.value_and_grad(f)
        >>> g(tc.num_to_tensor(1),tc.num_to_tensor(2))
        5, 2
        >>> g = tc.backend.value_and_grad(f, argnums=(0,1))
        >>> g(tc.num_to_tensor(1),tc.num_to_tensor(2))
        5, [2, 2]

        :param f: the function to be differentiated
        :type f: Callable[..., Any]
        :param argnums: the position of args in ``f`` that are to be differentiated, defaults to be 0
        :type argnums: Union[int, Sequence[int]], optional
        :return: the value and grad function of ``f`` with the same set of arguments as ``f``
        :rtype: Callable[..., Tuple[Any, Any]]
        """
        raise NotImplementedError("Backend '{}' has not implemented `value_and_grad`.".format(self.name))

    def jvp(self: Any, f: Callable[..., Any], inputs: Union[Tensor, Sequence[Tensor]], v: Union[Tensor, Sequence[Tensor]]) ->Tuple[Union[Tensor, Sequence[Tensor]], Union[Tensor, Sequence[Tensor]]]:
        """
        Function that computes a (forward-mode) Jacobian-vector product of ``f``.
        Strictly speaking, this function is value_and_jvp.

        :param f: The function to compute jvp
        :type f: Callable[..., Any]
        :param inputs: input for ``f``
        :type inputs: Union[Tensor, Sequence[Tensor]]
        :param v: tangents
        :type v: Union[Tensor, Sequence[Tensor]]
        :return: (``f(*inputs)``, jvp_tensor), where jvp_tensor is the same shape as the output of ``f``
        :rtype: Tuple[Union[Tensor, Sequence[Tensor]], Union[Tensor, Sequence[Tensor]]]
        """
        raise NotImplementedError("Backend '{}' has not implemented `jvp`.".format(self.name))

    def vjp(self: Any, f: Callable[..., Any], inputs: Union[Tensor, Sequence[Tensor]], v: Union[Tensor, Sequence[Tensor]]) ->Tuple[Union[Tensor, Sequence[Tensor]], Union[Tensor, Sequence[Tensor]]]:
        """
        Function that computes the dot product between a vector v and the Jacobian
        of the given function at the point given by the inputs. (reverse mode AD relevant)
        Strictly speaking, this function is value_and_vjp.

        :param f: the function to carry out vjp calculation
        :type f: Callable[..., Any]
        :param inputs: input for ``f``
        :type inputs: Union[Tensor, Sequence[Tensor]]
        :param v: value vector or gradient from downstream in reverse mode AD
            the same shape as return of function ``f``
        :type v: Union[Tensor, Sequence[Tensor]]
        :return: (``f(*inputs)``, vjp_tensor), where vjp_tensor is the same shape as inputs
        :rtype: Tuple[Union[Tensor, Sequence[Tensor]], Union[Tensor, Sequence[Tensor]]]
        """
        raise NotImplementedError("Backend '{}' has not implemented `vjp`.".format(self.name))

    def jacfwd(self: Any, f: Callable[..., Any], argnums: Union[int, Sequence[int]]=0) ->Tensor:
        """
        Compute the Jacobian of ``f`` using the forward mode AD.

        :param f: the function whose Jacobian is required
        :type f: Callable[..., Any]
        :param argnums: the position of the arg as Jacobian input, defaults to 0
        :type argnums: Union[int, Sequence[int]], optional
        :return: outer tuple for input args, inner tuple for outputs
        :rtype: Tensor
        """
        jvp1 = return_partial(self.jvp, return_argnums=1)
        if isinstance(argnums, int):
            argnums = argnums,

        def _transform(t: Tensor, input_shape: List[int]) ->Tensor:
            output_shape = list(self.shape_tuple(t))[1:]
            t = self.reshape(t, [t.shape[0], -1])
            t = self.transpose(t)
            t = self.reshape(t, output_shape + input_shape)
            return t

        def wrapper(*args: Any, **kws: Any) ->Any:
            pf = partial(f, **kws)
            jjs = []
            for argnum in argnums:
                jj = self.vmap(jvp1, vectorized_argnums=2)(pf, args, tuple([(self.reshape(self.eye(self.sizen(arg), dtype=arg.dtype), [-1] + list(self.shape_tuple(arg))) if i == argnum else self.reshape(self.zeros([self.sizen(arg), self.sizen(arg)], dtype=arg.dtype), [-1] + list(self.shape_tuple(arg)))) for i, arg in enumerate(args)]))
                jj = self.tree_map(partial(_transform, input_shape=list(self.shape_tuple(args[argnum]))), jj)
                jjs.append(jj)
            if len(jjs) == 1:
                return jjs[0]
            return tuple(jjs)
        return wrapper

    def jacrev(self: Any, f: Callable[..., Any], argnums: Union[int, Sequence[int]]=0) ->Tensor:
        """
        Compute the Jacobian of ``f`` using reverse mode AD.

        :param f: The function whose Jacobian is required
        :type f: Callable[..., Any]
        :param argnums: the position of the arg as Jacobian input, defaults to 0
        :type argnums: Union[int, Sequence[int]], optional
        :return: outer tuple for output, inner tuple for input args
        :rtype: Tensor
        """
        vjp1 = return_partial(self.vjp, return_argnums=1)
        if isinstance(argnums, int):
            argnums = argnums,

        def wrapper(*args: Any, **kws: Any) ->Any:
            pf = partial(f, **kws)
            values = f(*args, **kws)
            collect = tuple
            if isinstance(values, list):
                values = tuple(values)
                collect = list
            elif not isinstance(values, tuple):
                values = tuple([values])

                def _first(x: Sequence[Any]) ->Any:
                    return x[0]
                collect = _first
            jjs = []
            for k in range(len(values)):
                jj = self.vmap(vjp1, vectorized_argnums=2)(pf, args, collect([(self.reshape(self.eye(self.sizen(v), dtype=v.dtype), [-1] + list(self.shape_tuple(v))) if i == k else self.reshape(self.zeros([self.sizen(v), self.sizen(v)], dtype=v.dtype), [-1] + list(self.shape_tuple(v)))) for i, v in enumerate(values)]))
                jj = self.tree_map(lambda _: self.reshape(_, list(self.shape_tuple(values[k])) + list(self.shape_tuple(_))[1:]), jj)
                if len(jj) == 1:
                    jj = jj[0]
                jjs.append(jj)
            if len(jjs) == 1:
                return jjs[0]
            return tuple(jjs)
        return wrapper
    jacbwd = jacrev

    def hessian(self: Any, f: Callable[..., Any], argnums: Union[int, Sequence[int]]=0) ->Tensor:
        return self.jacfwd(self.jacrev(f, argnums=argnums), argnums=argnums)

    def jit(self: Any, f: Callable[..., Any], static_argnums: Optional[Union[int, Sequence[int]]]=None, jit_compile: Optional[bool]=None) ->Callable[..., Any]:
        """
        Return the jitted version of function ``f``.

        :param f: function to be jitted
        :type f: Callable[..., Any]
        :param static_argnums: index of args that doesn't regarded as tensor,
            only work for jax backend
        :type static_argnums: Optional[Union[int, Sequence[int]]], defaults to None
        :param jit_compile: whether open XLA compilation, only works for tensorflow backend,
            defaults False since several ops has no XLA correspondence
        :type jit_compile: bool
        :return: jitted version of ``f``
        :rtype: Callable[..., Any]
        """
        raise NotImplementedError("Backend '{}' has not implemented `jit`.".format(self.name))

    def vmap(self: Any, f: Callable[..., Any], vectorized_argnums: Union[int, Sequence[int]]=0) ->Any:
        """
        Return the vectorized map or batched version of ``f`` on the first extra axis.
        The general interface supports ``f`` with multiple arguments and broadcast in the fist dimension.

        :param f: function to be broadcasted.
        :type f: Callable[..., Any]
        :param vectorized_argnums: the args to be vectorized,
            these arguments should share the same batch shape in the fist dimension
        :type vectorized_argnums: Union[int, Sequence[int]], defaults to 0
        :return: vmap version of ``f``
        :rtype: Any
        """
        raise NotImplementedError("Backend '{}' has not implemented `vmap`.".format(self.name))

    def vectorized_value_and_grad(self: Any, f: Callable[..., Any], argnums: Union[int, Sequence[int]]=0, vectorized_argnums: Union[int, Sequence[int]]=0, has_aux: bool=False) ->Callable[..., Tuple[Any, Any]]:
        """
        Return the VVAG function of ``f``. The inputs for ``f`` is (args[0], args[1], args[2], ...),
        and the output of ``f`` is a scalar. Suppose VVAG(f) is a function with inputs in the form
        (vargs[0], args[1], args[2], ...), where vagrs[0] has one extra dimension than args[0] in the first axis
        and consistent with args[0] in shape for remaining dimensions, i.e. shape(vargs[0]) = [batch] + shape(args[0]).
        (We only cover cases where ``vectorized_argnums`` defaults to 0 here for demonstration).
        VVAG(f) returns a tuple as a value tensor with shape [batch, 1] and a gradient tuple with shape:
        ([batch]+shape(args[argnum]) for argnum in argnums). The gradient for argnums=k is defined as

        .. math::

            g^k = \\frac{\\partial \\sum_{i\\in batch} f(vargs[0][i], args[1], ...)}{\\partial args[k]}

        Therefore, if argnums=0, the gradient is reduced to

        .. math::

            g^0_i = \\frac{\\partial f(vargs[0][i])}{\\partial vargs[0][i]}

        , which is specifically suitable for batched VQE optimization, where args[0] is the circuit parameters.

        And if argnums=1, the gradient is like

        .. math::
            g^1_i = \\frac{\\partial \\sum_j f(vargs[0][j], args[1])}{\\partial args[1][i]}

        , which is suitable for quantum machine learning scenarios, where ``f`` is the loss function,
        args[0] corresponds to the input data and args[1] corresponds to the weights in the QML model.

        :param f: [description]
        :type f: Callable[..., Any]
        :param argnums: [description], defaults to 0
        :type argnums: Union[int, Sequence[int]], optional
        :param vectorized_argnums: the args to be vectorized, these arguments should share the same batch shape
            in the fist dimension
        :type vectorized_argnums: Union[int, Sequence[int]], defaults to 0
        :return: [description]
        :rtype: Callable[..., Tuple[Any, Any]]
        """
        raise NotImplementedError("Backend '{}' has not implemented `vectorized_value_and_grad`.".format(self.name))

    def __repr__(self: Any) ->str:
        return self.name + '_backend'
    __str__ = __repr__


PRNGKeyArray = Any


dtypestr = 'complex64'


logger = logging.getLogger(__name__)


npdtype = np.complex64


pytree = Any


class optax_optimizer:

    def __init__(self, optimizer: Any) ->None:
        self.optimizer = optimizer
        self.state = None

    def update(self, grads: pytree, params: pytree) ->pytree:
        if self.state is None:
            self.state = self.optimizer.init(params)
        updates, self.state = self.optimizer.update(grads, self.state)
        params = optax.apply_updates(params, updates)
        return params


class torch_optimizer:

    def __init__(self, optimizer: Any) ->None:
        self.optimizer = optimizer
        self.is_init = False

    def update(self, grads: pytree, params: pytree) ->pytree:
        params, treedef = PyTorchBackend.tree_flatten(None, params)
        grads, _ = PyTorchBackend.tree_flatten(None, grads)
        if self.is_init is False:
            self.optimizer = self.optimizer(params)
            self.is_init = True
        with torchlib.no_grad():
            for g, p in zip(grads, params):
                p.grad = g
        self.optimizer.step()
        self.optimizer.zero_grad()
        params = PyTorchBackend.tree_unflatten(None, treedef, params)
        return params


RGenerator = Any


def _random_choice_tf(g: RGenerator, a: Union[int, Sequence[int], Tensor], shape: Union[int, Sequence[int]], p: Optional[Union[Sequence[float], Tensor]]=None) ->Tensor:
    if isinstance(a, int):
        assert a > 0
        a = tf.range(a)
    if not (isinstance(a, tf.Tensor) or isinstance(a, tf.Variable)):
        a = tf.constant(a)
    assert len(a.shape) == 1
    if isinstance(shape, int):
        shape = shape,
    if p is None:
        dtype = tf.float32
        p = tf.ones_like(a)
        p = tf.cast(p, dtype=dtype)
        p /= tf.reduce_sum(p)
    else:
        if not (isinstance(p, tf.Tensor) or isinstance(p, tf.Variable)):
            p = tf.constant(p)
        dtype = p.dtype
    shape1 = reduce(mul, shape)
    p_cuml = tf.cumsum(p)
    r = p_cuml[-1] * (1 - g.uniform([shape1], dtype=dtype))
    ind = tf.searchsorted(p_cuml, r)
    res = tf.gather(a, ind)
    return tf.reshape(res, shape)


class keras_optimizer:

    def __init__(self, optimizer: Any) ->None:
        self.optimizer = optimizer
        self.is_variable = True

    def _c2v(self, v: Tensor) ->Tensor:
        if not isinstance(v, tf.Variable):
            v = tf.Variable(v)
            self.is_variable = False
        return v

    def _apply_gradients(self, grads: Tensor, params: Tensor) ->None:
        self.optimizer.apply_gradients([(grads, params)])

    def update(self, grads: pytree, params: pytree) ->pytree:
        params = TensorFlowBackend.tree_map(None, self._c2v, params)
        TensorFlowBackend.tree_map(None, self._apply_gradients, grads, params)
        if not self.is_variable:
            return TensorFlowBackend.tree_map(None, tf.convert_to_tensor, params)
        return params


bk = Any


module2backend = {'tensorflow': 'tensorflow', 'numpy': 'numpy', 'jaxlib': 'jax', 'torch': 'pytorch', 'jax': 'jax'}


def which_backend(a: Tensor, return_backend: bool=True) ->Any:
    """
    Given a tensor ``a``, return the corresponding backend

    :param a: the tensor
    :type a: Tensor
    :param return_backend: if true, return backend object, if false, return backend str,
        defaults to True
    :type return_backend: bool, optional
    :return: the backend object or backend str
    :rtype: Any
    """
    module = type(a).__module__.split('.')[0]
    bkstr = module2backend[module]
    if not return_backend:
        return bkstr
    return get_backend(bkstr)


def tensor_to_numpy(t: Tensor) ->Array:
    if isinstance(t, int) or isinstance(t, float):
        return t
    return which_backend(t).numpy(t)


def general_args_to_numpy(args: Any) ->Any:
    """
    Given a pytree, get the corresponding numpy array pytree

    :param args: pytree
    :type args: Any
    :return: the same format pytree with all tensor replaced by numpy array
    :rtype: Any
    """
    return backend.tree_map(tensor_to_numpy, args)


def numpy_to_tensor(t: Array, backend: Any) ->Tensor:
    if isinstance(t, int) or isinstance(t, float):
        return t
    return backend.convert_to_tensor(t)


def numpy_args_to_backend(args: Any, dtype: Any=None, target_backend: Any=None) ->Any:
    """
    Given a pytree of numpy arrays, get the corresponding tensor pytree

    :param args: pytree of numpy arrays
    :type args: Any
    :param dtype: str of str of the same pytree shape as args, defaults to None
    :type dtype: Any, optional
    :param target_backend: str or backend object, defaults to None,
        indicating the current default backend
    :type target_backend: Any, optional
    :return: the same format pytree with all numpy array replaced by the tensors
        in the target backend
    :rtype: Any
    """
    if target_backend is None:
        target_backend = backend
    elif isinstance(target_backend, str):
        target_backend = get_backend(target_backend)
    if dtype is None:
        return backend.tree_map(partial(numpy_to_tensor, backend=target_backend), args)
    else:
        if isinstance(dtype, str):
            leaves, treedef = backend.tree_flatten(args)
            dtype = [dtype for _ in range(len(leaves))]
            dtype = backend.tree_unflatten(treedef, dtype)
        t = backend.tree_map(partial(numpy_to_tensor, backend=target_backend), args)
        t = backend.tree_map(target_backend.cast, t, dtype)
        return t


def tensor_to_dlpack(t: Tensor) ->Any:
    return which_backend(t).to_dlpack(t)


def general_args_to_backend(args: Any, dtype: Any=None, target_backend: Any=None, enable_dlpack: bool=True) ->Any:
    if not enable_dlpack:
        args = general_args_to_numpy(args)
        args = numpy_args_to_backend(args, dtype, target_backend)
        return args
    caps = backend.tree_map(tensor_to_dlpack, args)
    if target_backend is None:
        target_backend = backend
    elif isinstance(target_backend, str):
        target_backend = get_backend(target_backend)
    if dtype is None:
        return backend.tree_map(target_backend.from_dlpack, caps)
    if isinstance(dtype, str):
        leaves, treedef = backend.tree_flatten(args)
        dtype = [dtype for _ in range(len(leaves))]
        dtype = backend.tree_unflatten(treedef, dtype)
    t = backend.tree_map(target_backend.from_dlpack, caps)
    t = backend.tree_map(target_backend.cast, t, dtype)
    return t


def torch_interface(fun: Callable[..., Any], jit: bool=False, enable_dlpack: bool=False) ->Callable[..., Any]:
    """
    Wrap a quantum function on different ML backend with a pytorch interface.

    :Example:

    .. code-block:: python

        import torch

        tc.set_backend("tensorflow")


        def f(params):
            c = tc.Circuit(1)
            c.rx(0, theta=params[0])
            c.ry(0, theta=params[1])
            return c.expectation([tc.gates.z(), [0]])


        f_torch = tc.interfaces.torch_interface(f, jit=True)

        a = torch.ones([2], requires_grad=True)
        b = f_torch(a)
        c = b ** 2
        c.backward()

        print(a.grad)

    :param fun: The quantum function with tensor in and tensor out
    :type fun: Callable[..., Any]
    :param jit: whether to jit ``fun``, defaults to False
    :type jit: bool, optional
    :param enable_dlpack: whether transform tensor backend via dlpack, defaults to False
    :type enable_dlpack: bool, optional
    :return: The same quantum function but now with torch tensor in and torch tensor out
        while AD is also supported
    :rtype: Callable[..., Any]
    """
    import torch

    def vjp_fun(x: Tensor, v: Tensor) ->Tuple[Tensor, Tensor]:
        return backend.vjp(fun, x, v)
    if jit is True:
        fun = backend.jit(fun)
        vjp_fun = backend.jit(vjp_fun)


    class Fun(torch.autograd.Function):

        @staticmethod
        def forward(ctx: Any, *x: Any) ->Any:
            ctx.xdtype = backend.tree_map(lambda s: s.dtype, x)
            if len(ctx.xdtype) == 1:
                ctx.xdtype = ctx.xdtype[0]
            ctx.device = backend.tree_flatten(x)[0][0].device
            x = general_args_to_backend(x, enable_dlpack=enable_dlpack)
            y = fun(*x)
            ctx.ydtype = backend.tree_map(lambda s: s.dtype, y)
            if len(x) == 1:
                x = x[0]
            ctx.x = x
            y = general_args_to_backend(y, target_backend='pytorch', enable_dlpack=enable_dlpack)
            y = backend.tree_map(lambda s: s, y)
            return y

        @staticmethod
        def backward(ctx: Any, *grad_y: Any) ->Any:
            if len(grad_y) == 1:
                grad_y = grad_y[0]
            grad_y = backend.tree_map(lambda s: s.contiguous(), grad_y)
            grad_y = general_args_to_backend(grad_y, dtype=ctx.ydtype, enable_dlpack=enable_dlpack)
            _, g = vjp_fun(ctx.x, grad_y)
            r = general_args_to_backend(g, dtype=ctx.xdtype, target_backend='pytorch', enable_dlpack=enable_dlpack)
            r = backend.tree_map(lambda s: s, r)
            if not is_sequence(r):
                return r,
            return r
    return Fun.apply


class QuantumNet(torch.nn.Module):

    def __init__(self, f: Callable[..., Any], weights_shape: Sequence[Tuple[int, ...]], initializer: Union[Any, Sequence[Any]]=None, use_vmap: bool=True, vectorized_argnums: Union[int, Sequence[int]]=0, use_interface: bool=True, use_jit: bool=True, enable_dlpack: bool=False):
        """
        PyTorch nn Module wrapper on quantum function ``f``.

        :Example:

        .. code-block:: python

            K = tc.set_backend("tensorflow")

            n = 6
            nlayers = 2
            batch = 2

            def qpred(x, weights):
                c = tc.Circuit(n)
                for i in range(n):
                    c.rx(i, theta=x[i])
                for j in range(nlayers):
                    for i in range(n - 1):
                        c.cnot(i, i + 1)
                    for i in range(n):
                        c.rx(i, theta=weights[2 * j, i])
                        c.ry(i, theta=weights[2 * j + 1, i])
                ypred = K.stack([c.expectation_ps(x=[i]) for i in range(n)])
                ypred = K.real(ypred)
                return ypred

            ql = tc.torchnn.QuantumNet(qpred, weights_shape=[2*nlayers, n])

            ql(torch.ones([batch, n]))


        :param f: Quantum function with tensor in (input and weights) and tensor out.
        :type f: Callable[..., Any]
        :param weights_shape: list of shape tuple for different weights as the non-first parameters for ``f``
        :type weights_shape: Sequence[Tuple[int, ...]]
        :param initializer: function that gives the shape tuple returns torch tensor, defaults to None
        :type initializer: Union[Any, Sequence[Any]], optional
        :param use_vmap: whether apply vmap (batch input) on ``f``, defaults to True
        :type use_vmap: bool, optional
        :param vectorized_argnums: which position of input should be batched, need to be customized when
            multiple inputs for the torch model, defaults to be 0.
        :type vectorized_argnums: Union[int, Sequence[int]]
        :param use_interface: whether transform ``f`` with torch interface, defaults to True
        :type use_interface: bool, optional
        :param use_jit: whether jit ``f``, defaults to True
        :type use_jit: bool, optional
        :param enable_dlpack: whether enbale dlpack in interfaces, defaults to False
        :type enable_dlpack: bool, optional
        """
        super().__init__()
        if use_vmap:
            f = backend.vmap(f, vectorized_argnums=vectorized_argnums)
        if use_interface:
            f = torch_interface(f, jit=use_jit, enable_dlpack=enable_dlpack)
        self.f = f
        self.q_weights = torch.nn.ParameterList()
        if isinstance(weights_shape[0], int):
            weights_shape = [weights_shape]
        if not is_sequence(initializer):
            initializer = [initializer]
        for ws, initf in zip(weights_shape, initializer):
            if initf is None:
                initf = torch.randn
            self.q_weights.append(torch.nn.Parameter(initf(ws)))

    def forward(self, *inputs: Tensor) ->Tensor:
        ypred = self.f(*inputs, *self.q_weights)
        return ypred

