import sys
_module = sys.modules[__name__]
del sys
conf = _module
backward_modes = _module
bundle_adjustment = _module
homography_estimation = _module
motion_planning_2d = _module
pose_graph_benchmark = _module
pose_graph_cube = _module
pose_graph_synthetic = _module
se2_inverse = _module
simple_example = _module
state_estimation_2d = _module
tactile_pose_estimation = _module
noxfile = _module
setup = _module
tests = _module
core = _module
common = _module
test_cost_function = _module
test_cost_weight = _module
test_manifold = _module
test_objective = _module
test_robust_cost = _module
test_theseus_function = _module
test_variable = _module
test_vectorizer = _module
collision = _module
test_collision_factor = _module
test_eff_obj_contact = _module
test_signed_distance_field = _module
utils = _module
test_inverse_kinematics = _module
test_urdf_model = _module
test_between = _module
test_moving_frame_between = _module
test_reprojection = _module
test_variable_difference = _module
test_double_integrator = _module
test_quasi_static_pushing_planar = _module
extlib = _module
test_baspacho = _module
test_baspacho_simple = _module
test_cusolver_lu_solver = _module
test_mat_mult = _module
geometry = _module
common = _module
functional = _module
common = _module
test_so3 = _module
point_types_mypy_check = _module
test_point_types = _module
test_se2 = _module
test_se3 = _module
test_so2 = _module
test_so3 = _module
test_vector = _module
autograd = _module
common = _module
test_baspacho_sparse_backward = _module
test_lu_cuda_sparse_backward = _module
test_sparse_backward = _module
test_baspacho_sparse_solver = _module
test_cholmod_sparse_solver = _module
test_dense_solver = _module
test_lu_cuda_sparse_solver = _module
linearization_test_utils = _module
nonlinear = _module
common = _module
test_backwards = _module
test_dogleg = _module
test_gauss_newton = _module
test_levenberg_marquardt = _module
test_state_history = _module
test_trust_region = _module
test_dense_linearization = _module
test_manifold_gaussian = _module
test_sparse_linearization = _module
test_variable_ordering = _module
test_dlm_perturbation = _module
test_theseus_layer = _module
test_utils = _module
theseus = _module
constants = _module
cost_function = _module
cost_weight = _module
objective = _module
robust_cost_function = _module
robust_loss = _module
theseus_function = _module
variable = _module
vectorizer = _module
embodied = _module
collision = _module
eff_obj_contact = _module
signed_distance_field = _module
kinematics = _module
kinematics_model = _module
measurements = _module
between = _module
moving_frame_between = _module
reprojection = _module
misc = _module
local_cost_fn = _module
motionmodel = _module
double_integrator = _module
quasi_static_pushing_planar = _module
constants = _module
lie_group = _module
so3 = _module
utils = _module
lie_group = _module
lie_group_check = _module
manifold = _module
point_types = _module
se2 = _module
se3 = _module
so2 = _module
so3 = _module
utils = _module
vector = _module
optimizer = _module
baspacho_sparse_autograd = _module
cholmod_sparse_autograd = _module
lu_cuda_sparse_autograd = _module
dense_linearization = _module
linear = _module
baspacho_sparse_solver = _module
cholmod_sparse_solver = _module
dense_solver = _module
linear_optimizer = _module
linear_solver = _module
lu_cuda_sparse_solver = _module
utils = _module
linear_system = _module
linearization = _module
manifold_gaussian = _module
dogleg = _module
gauss_newton = _module
levenberg_marquardt = _module
nonlinear_least_squares = _module
nonlinear_optimizer = _module
trust_region = _module
optimizer = _module
sparse_linearization = _module
variable_ordering = _module
theseus_layer = _module
third_party = _module
easyaug = _module
utils = _module
examples = _module
data = _module
util = _module
motion_planning = _module
misc = _module
models = _module
motion_planner = _module
pose_graph = _module
dataset = _module
misc = _module
models = _module
pose_estimator = _module
trainer = _module
sparse_matrix_utils = _module
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


import time


from collections import defaultdict


import numpy as np


import torch


import logging


import random


from typing import Dict


from typing import List


from typing import Type


import warnings


from typing import Any


from typing import Optional


from typing import Tuple


from typing import cast


import torch.nn as nn


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


import matplotlib.pyplot as plt


import torch.nn.functional as F


import torch.utils.data


from scipy.io import savemat


from typing import Union


import copy


from enum import Enum


from scipy.sparse import csr_matrix


from scipy.sparse import tril


from torch.autograd import grad


from torch.autograd import gradcheck


import itertools


import math


import scipy.sparse


import abc


from itertools import chain


from typing import Callable


from typing import Sequence


import torch.autograd.functional as autogradF


from collections import OrderedDict


from typing import Iterable


from itertools import count


from typing import Set


from scipy import ndimage


import torch.linalg


from scipy.sparse import csc_matrix


from typing import NoReturn


from torch.autograd.function import once_differentiable


from typing import NamedTuple


import matplotlib as mpl


import matplotlib.patches as mpatches


import collections


import torch.optim as optim


from scipy.sparse import lil_matrix


class SimpleCNN(nn.Module):

    def __init__(self, D=32):
        super(SimpleCNN, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(3, D, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(D, 3, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(D)

    def forward(self, img):
        x = self.relu(self.bn1(self.conv1(img)))
        return self.conv2(x)


class SimpleNN(nn.Module):

    def __init__(self, in_size, out_size, hid_size=30, use_offset=False):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(in_size, hid_size), nn.ReLU(), nn.Linear(hid_size, hid_size), nn.ReLU(), nn.Linear(hid_size, out_size))

    def forward(self, state_):
        return self.fc(state_)


class BackwardMode(Enum):
    UNROLL = 0
    IMPLICIT = 1
    TRUNCATED = 2
    DLM = 3

    @staticmethod
    def resolve(key: Union[str, 'BackwardMode']) ->'BackwardMode':
        if isinstance(key, BackwardMode):
            return key
        if not isinstance(key, str):
            raise ValueError('Backward mode must be th.BackwardMode or string.')
        try:
            backward_mode = BackwardMode[key.upper()]
        except KeyError:
            raise ValueError(f'Unrecognized backward mode f{key}.Valid choices are unroll, implicit, truncated, dlm.')
        return backward_mode


DeviceType = Optional[Union[str, torch.device]]


class Variable:
    """A variable in a differentiable optimization problem."""
    _ids = count(0)

    def __init__(self, tensor: torch.Tensor, name: Optional[str]=None):
        self._id = next(Variable._ids)
        self._num_updates = 0
        if name:
            self.name = name
        else:
            self.name = f'{self.__class__.__name__}__{self._id}'
        self.tensor = tensor

    def copy(self, new_name: Optional[str]=None) ->'Variable':
        if not new_name:
            new_name = f'{self.name}_copy'
        return Variable(self.tensor.clone(), name=new_name)

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        the_copy = self.copy()
        memo[id(self)] = the_copy
        return the_copy

    def update(self, data: Union[torch.Tensor, 'Variable'], batch_ignore_mask: Optional[torch.Tensor]=None):
        if isinstance(data, Variable):
            tensor = data.tensor
        else:
            tensor = data
        if len(tensor.shape) != len(self.tensor.shape) or tensor.shape[1:] != self.tensor.shape[1:]:
            raise ValueError(f'Tried to update tensor {self.name} with data incompatible with original tensor shape. Given {tensor.shape[1:]}. Expected: {self.tensor.shape[1:]}')
        if tensor.dtype != self.dtype:
            raise ValueError(f'Tried to update used tensor of dtype {tensor.dtype} but Variable {self.name} has dtype {self.dtype}.')
        if batch_ignore_mask is not None and batch_ignore_mask.any():
            mask_shape = (-1,) + (1,) * (tensor.ndim - 1)
            self.tensor = torch.where(batch_ignore_mask.view(mask_shape), self.tensor, tensor)
        else:
            self.tensor = tensor
        self._num_updates += 1

    def __repr__(self) ->str:
        return f'{self.__class__.__name__}(tensor={self.tensor}, name={self.name})'

    def __str__(self) ->str:
        return repr(self)

    def to(self, *args, **kwargs):
        self.tensor = self.tensor

    @property
    def shape(self) ->torch.Size:
        return self.tensor.shape

    @property
    def device(self) ->torch.device:
        return self.tensor.device

    @property
    def dtype(self) ->torch.dtype:
        return self.tensor.dtype

    @property
    def ndim(self) ->int:
        return self.tensor.ndim

    def __getitem__(self, item):
        return self.tensor[item]

    def __setitem__(self, item, value):
        self.tensor[item] = value


def _CHECK_DTYPE_SUPPORTED(dtype):
    if dtype not in [torch.float32, torch.float64]:
        raise ValueError(f'Unsupported data type {dtype}. Theseus only supports 32- and 64-bit tensors.')


class Manifold(Variable, abc.ABC):

    def __init__(self, *args: Any, tensor: Optional[torch.Tensor]=None, name: Optional[str]=None, dtype: Optional[torch.dtype]=None, strict: bool=False):
        if tensor is None and dtype is None:
            dtype = torch.get_default_dtype()
        if tensor is not None:
            checks_enabled, silent_unchecks = _LieGroupCheckContext.get_context()
            if checks_enabled:
                tensor = self._check_tensor(tensor, strict)
            elif not silent_unchecks:
                warnings.warn(f'Manifold consistency checks are disabled for {self.__class__.__name__}.', RuntimeWarning)
            if dtype is not None and tensor.dtype != dtype:
                warnings.warn(f'tensor.dtype {tensor.dtype} does not match given dtype {dtype}, tensor.dtype will take precendence.')
            dtype = tensor.dtype
        _CHECK_DTYPE_SUPPORTED(dtype)
        super().__init__(self.__class__._init_tensor(*args), name=name)
        if tensor is not None:
            self.update(tensor)

    @staticmethod
    @abc.abstractmethod
    def _init_tensor(*args: Any) ->torch.Tensor:
        pass

    @abc.abstractmethod
    def dof(self) ->int:
        pass

    def numel(self) ->int:
        return self.tensor[0].numel()

    @abc.abstractmethod
    def _local_impl(self, variable2: 'Manifold', jacobians: Optional[List[torch.Tensor]]=None) ->torch.Tensor:
        pass

    @abc.abstractmethod
    def _retract_impl(self, delta: torch.Tensor) ->'Manifold':
        pass

    @abc.abstractmethod
    def _project_impl(self, euclidean_grad: torch.Tensor, is_sparse: bool=False) ->torch.Tensor:
        pass

    def project(self, euclidean_grad: torch.Tensor, is_sparse: bool=False) ->torch.Tensor:
        return self._project_impl(euclidean_grad, is_sparse)

    @staticmethod
    @abc.abstractmethod
    def normalize(tensor: torch.Tensor) ->torch.Tensor:
        pass

    @staticmethod
    @abc.abstractmethod
    def _check_tensor_impl(tensor: torch.Tensor) ->bool:
        pass

    @classmethod
    def _check_tensor(cls, tensor: torch.Tensor, strict: bool=True) ->torch.Tensor:
        check = cls._check_tensor_impl(tensor)
        if not check:
            if strict:
                raise ValueError(f'The input tensor is not valid for {cls.__name__}.')
            else:
                tensor = cls.normalize(tensor)
                warnings.warn(f'The input tensor is not valid for {cls.__name__} and has been normalized.')
        return tensor

    def local(self, variable2: 'Manifold', jacobians: Optional[List[torch.Tensor]]=None) ->torch.Tensor:
        local_out = self._local_impl(variable2, jacobians)
        return local_out

    def retract(self, delta: torch.Tensor) ->'Manifold':
        return self._retract_impl(delta)

    @abc.abstractmethod
    def _copy_impl(self, new_name: Optional[str]=None) ->'Manifold':
        pass

    def copy(self, new_name: Optional[str]=None) ->'Manifold':
        if not new_name:
            new_name = f'{self.name}_copy'
        var_copy = self._copy_impl(new_name=new_name)
        return var_copy

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        the_copy = self.copy()
        memo[id(self)] = the_copy
        return the_copy

    def to(self, *args, **kwargs):
        _, dtype, *_ = torch._C._nn._parse_to(*args, **kwargs)
        if dtype is not None:
            _CHECK_DTYPE_SUPPORTED(dtype)
        super()


_CostFunctionSchema = Tuple[str, ...]


class _VectorizationMode(Enum):
    ERROR = 0
    WEIGHTED_ERROR = 1
    FULL = 2


__FROM_THESEUS_LAYER_TOKEN__ = '__FROM_THESEUS_LAYER_TOKEN__'


class TheseusLayerDLMForward(torch.autograd.Function):
    """
    Functionally the same as the forward method in a TheseusLayer
    but computes the direct loss minimization in the backward pass.
    """
    _DLM_EPSILON_STR = 'dlm_epsilon'
    _GRAD_SUFFIX = '_grad'

    @staticmethod
    def forward(ctx, objective, optimizer, optimizer_kwargs, bwd_objective, bwd_optimizer, epsilon, n, *inputs):
        input_keys = inputs[:n]
        input_vals = inputs[n:2 * n]
        differentiable_tensors = inputs[2 * n:]
        ctx.n = n
        ctx.k = len(differentiable_tensors)
        inputs = dict(zip(input_keys, input_vals))
        ctx.input_keys = input_keys
        optim_tensors, info = _forward(objective, optimizer, optimizer_kwargs, inputs)
        if ctx.k > 0:
            ctx.bwd_objective = bwd_objective
            ctx.bwd_optimizer = bwd_optimizer
            ctx.epsilon = epsilon
            with torch.enable_grad():
                grad_sol = torch.autograd.grad(objective.error_squared_norm().sum(), differentiable_tensors, allow_unused=True)
            ctx.save_for_backward(*input_vals, *grad_sol, *differentiable_tensors, *optim_tensors)
        return *optim_tensors, info

    @staticmethod
    @once_differentiable
    def backward(ctx, *grad_outputs):
        n, k = ctx.n, ctx.k
        saved_tensors = ctx.saved_tensors
        input_vals = saved_tensors[:n]
        grad_sol = saved_tensors[n:n + k]
        differentiable_tensors = saved_tensors[n + k:n + k + k]
        optim_tensors = saved_tensors[n + k + k:]
        grad_outputs = grad_outputs[:-1]
        bwd_objective: Objective = ctx.bwd_objective
        bwd_optimizer: Optimizer = ctx.bwd_optimizer
        epsilon = ctx.epsilon
        input_keys = ctx.input_keys
        bwd_data = dict(zip(input_keys, input_vals))
        for k, v in zip(bwd_objective.optim_vars.keys(), optim_tensors):
            bwd_data[k] = v.detach()
        grad_data = {TheseusLayerDLMForward._DLM_EPSILON_STR: torch.tensor(epsilon).reshape(1, 1)}
        for i, name in enumerate(bwd_objective.optim_vars.keys()):
            grad_data[name + TheseusLayerDLMForward._GRAD_SUFFIX] = grad_outputs[i]
        bwd_data.update(grad_data)
        bwd_objective.update(bwd_data)
        with torch.no_grad():
            bwd_optimizer.linear_solver.linearization.linearize()
            delta = bwd_optimizer.linear_solver.solve()
            bwd_optimizer.objective.retract_vars_sequence(delta, bwd_optimizer.linear_solver.linearization.ordering)
        with torch.enable_grad():
            grad_perturbed = torch.autograd.grad(bwd_objective.error_squared_norm().sum(), differentiable_tensors, allow_unused=True)
        nones = [None] * (ctx.n * 2)
        grads = [((gs - gp) / epsilon if gs is not None else None) for gs, gp in zip(grad_sol, grad_perturbed)]
        return None, None, None, None, None, None, None, *nones, *grads


class SparseStructure(abc.ABC):

    def __init__(self, col_ind: np.ndarray, row_ptr: np.ndarray, num_rows: int, num_cols: int, dtype: np.dtype=np.float_):
        self.col_ind = col_ind
        self.row_ptr = row_ptr
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.dtype = dtype

    def csr_straight(self, val: torch.Tensor) ->csr_matrix:
        return csr_matrix((val, self.col_ind, self.row_ptr), (self.num_rows, self.num_cols), dtype=self.dtype)

    def csc_transpose(self, val: torch.Tensor) ->csc_matrix:
        return csc_matrix((val, self.col_ind, self.row_ptr), (self.num_cols, self.num_rows), dtype=self.dtype)

    def mock_csc_transpose(self) ->csc_matrix:
        return csc_matrix((np.ones(len(self.col_ind), dtype=self.dtype), self.col_ind, self.row_ptr), (self.num_cols, self.num_rows), dtype=self.dtype)


_LUCudaSolveFunctionBwdReturnType = Tuple[torch.Tensor, torch.Tensor, None, None, None, None, None, None]


def _sparse_mat_vec_bwd_backend(ctx: Any, grad_output: torch.Tensor, is_tmat: bool) ->Tuple[torch.Tensor, torch.Tensor]:
    A_val, A_row_ptr, A_col_ind, v = ctx.saved_tensors
    num_rows = len(A_row_ptr) - 1
    A_grad = torch.zeros_like(A_val)
    v_grad = torch.zeros_like(v)
    for row in range(num_rows):
        start = A_row_ptr[row]
        end = A_row_ptr[row + 1]
        columns = A_col_ind[start:end].long()
        if is_tmat:
            A_grad[:, start:end] = v[:, row].view(-1, 1) * grad_output[:, columns]
            v_grad[:, row] = (grad_output[:, columns] * A_val[:, start:end]).sum(dim=1)
        else:
            A_grad[:, start:end] = v[:, columns] * grad_output[:, row].view(-1, 1)
            v_grad[:, columns] += grad_output[:, row].view(-1, 1) * A_val[:, start:end]
    return A_grad, v_grad


def _sparse_mat_vec_fwd_backend(ctx: Any, num_cols: int, A_row_ptr: torch.Tensor, A_col_ind: torch.Tensor, A_val: torch.Tensor, v: torch.Tensor, op: Callable[[int, int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]) ->torch.Tensor:
    assert A_row_ptr.ndim == 1
    assert A_col_ind.ndim == 1
    assert A_val.ndim == 2
    assert v.ndim == 2
    ctx.save_for_backward(A_val, A_row_ptr, A_col_ind, v)
    ctx.num_cols = num_cols
    return op(A_val.shape[0], num_cols, A_row_ptr, A_col_ind, A_val, v)


def _tmat_vec_cpu(batch_size: int, num_cols: int, A_row_ptr: torch.Tensor, A_col_ind: torch.Tensor, A_val: torch.Tensor, v: torch.Tensor) ->torch.Tensor:
    assert batch_size == A_val.shape[0]
    num_rows = len(A_row_ptr) - 1
    retv_data = np.array([(csc_matrix((A_val[i].numpy(), A_col_ind, A_row_ptr), (num_cols, num_rows)) * v[i]) for i in range(batch_size)], dtype=np.float64)
    return torch.tensor(retv_data, dtype=torch.float64)


class _SparseMtvPAutograd(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, num_cols: int, A_row_ptr: torch.Tensor, A_col_ind: torch.Tensor, A_val: torch.Tensor, v: torch.Tensor) ->torch.Tensor:
        return _sparse_mat_vec_fwd_backend(ctx, num_cols, A_row_ptr, A_col_ind, A_val, v, tmat_vec)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx: Any, grad_output: torch.Tensor) ->Tuple[None, None, None, torch.Tensor, torch.Tensor]:
        A_grad, v_grad = _sparse_mat_vec_bwd_backend(ctx, grad_output, True)
        return None, None, None, A_grad, v_grad


sparse_mtv = _SparseMtvPAutograd.apply


def _mat_vec_cpu(batch_size: int, num_cols: int, A_row_ptr: torch.Tensor, A_col_ind: torch.Tensor, A_val: torch.Tensor, v: torch.Tensor) ->torch.Tensor:
    assert batch_size == A_val.shape[0]
    num_rows = len(A_row_ptr) - 1
    retv_data = np.array([(csr_matrix((A_val[i].numpy(), A_col_ind, A_row_ptr), (num_rows, num_cols)) * v[i]) for i in range(batch_size)], dtype=np.float64)
    return torch.tensor(retv_data, dtype=torch.float64)


class _SparseMvPAutograd(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, num_cols: int, A_row_ptr: torch.Tensor, A_col_ind: torch.Tensor, A_val: torch.Tensor, v: torch.Tensor) ->torch.Tensor:
        return _sparse_mat_vec_fwd_backend(ctx, num_cols, A_row_ptr, A_col_ind, A_val, v, mat_vec)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx: Any, grad_output: torch.Tensor) ->Tuple[None, None, None, torch.Tensor, torch.Tensor]:
        A_grad, v_grad = _sparse_mat_vec_bwd_backend(ctx, grad_output, False)
        return None, None, None, A_grad, v_grad


sparse_mv = _SparseMvPAutograd.apply


def convert_to_alpha_beta_damping_tensors(damping: Union[float, torch.Tensor], damping_eps: float, ellipsoidal_damping: bool, batch_size: int, device: DeviceType, dtype: torch.dtype) ->Tuple[torch.Tensor, torch.Tensor]:
    damping = torch.as_tensor(damping)
    if damping.ndim > 1:
        raise ValueError('Damping must be a float or a 1-D tensor.')
    if damping.ndim == 0 or damping.shape[0] == 1 and batch_size != 1:
        damping = damping.repeat(batch_size)
    return (damping, damping_eps * torch.ones_like(damping)) if ellipsoidal_damping else (torch.zeros_like(damping), damping)


class NonlinearOptimizerStatus(Enum):
    START = 0
    CONVERGED = 1
    MAX_ITERATIONS = 2
    FAIL = -1


def as_variable(value: Union[float, Sequence[float], torch.Tensor, Variable], device: DeviceType=None, dtype: Optional[torch.dtype]=None, name: Optional[str]=None) ->Variable:
    if isinstance(value, Variable):
        return value
    if isinstance(device, str):
        device = torch.device(device)
    tensor = torch.as_tensor(value, dtype=dtype, device=device)
    if isinstance(value, float):
        tensor = tensor.view(1, 1)
    return Variable(tensor, name=name)


class LieGroup(Manifold):

    def __init__(self, *args: Any, tensor: Optional[torch.Tensor]=None, name: Optional[str]=None, dtype: torch.dtype=torch.float, strict: bool=False):
        super().__init__(*args, tensor=tensor, name=name, dtype=dtype, strict=strict)

    @staticmethod
    def _check_jacobians_list(jacobians: List[torch.Tensor]):
        if len(jacobians) != 0:
            raise ValueError('jacobians list to be populated must be empty.')

    @staticmethod
    @abc.abstractmethod
    def _init_tensor(*args: Any) ->torch.Tensor:
        pass

    @abc.abstractmethod
    def dof(self) ->int:
        pass

    @staticmethod
    @abc.abstractmethod
    def rand(*size: int, generator: Optional[torch.Generator]=None, dtype: Optional[torch.dtype]=None, device: DeviceType=None, requires_grad: bool=False) ->'LieGroup':
        pass

    @staticmethod
    @abc.abstractmethod
    def randn(*size: int, generator: Optional[torch.Generator]=None, dtype: Optional[torch.dtype]=None, device: DeviceType=None, requires_grad: bool=False) ->'LieGroup':
        pass

    def __str__(self) ->str:
        return repr(self)

    @staticmethod
    @abc.abstractmethod
    def exp_map(tangent_vector: torch.Tensor, jacobians: Optional[List[torch.Tensor]]=None) ->'LieGroup':
        pass

    @abc.abstractmethod
    def _log_map_impl(self, jacobians: Optional[List[torch.Tensor]]=None) ->torch.Tensor:
        pass

    @abc.abstractmethod
    def to_matrix(self) ->torch.Tensor:
        pass

    def log_map(self, jacobians: Optional[List[torch.Tensor]]=None) ->torch.Tensor:
        return self._log_map_impl(jacobians)

    @abc.abstractmethod
    def _adjoint_impl(self) ->torch.Tensor:
        pass

    def adjoint(self) ->torch.Tensor:
        return self._adjoint_impl()

    def _project_check(self, euclidean_grad: torch.Tensor, is_sparse: bool=False):
        if euclidean_grad.dtype != self.dtype:
            raise ValueError('Euclidean gradients must be of the same type as the Lie group.')
        if euclidean_grad.device != self.device:
            raise ValueError('Euclidean gradients must be on the same device as the Lie group.')
        if euclidean_grad.shape[-self.ndim + is_sparse:] != self.shape[is_sparse:]:
            raise ValueError('Euclidean gradients must have consistent shapes with the Lie group.')

    def between(self, variable2: 'LieGroup', jacobians: Optional[List[torch.Tensor]]=None) ->'LieGroup':
        v1_inverse = self._inverse_impl()
        between = v1_inverse._compose_impl(variable2)
        if jacobians is not None:
            LieGroup._check_jacobians_list(jacobians)
            Jinv = LieGroup._inverse_jacobian(self)
            Jcmp0, Jcmp1 = v1_inverse._compose_jacobian(variable2)
            Jbetween0 = torch.matmul(Jcmp0, Jinv)
            jacobians.extend([Jbetween0, Jcmp1])
        return between

    @abc.abstractmethod
    def _compose_impl(self, variable2: 'LieGroup') ->'LieGroup':
        pass

    def compose(self, variable2: 'LieGroup', jacobians: Optional[List[torch.Tensor]]=None) ->'LieGroup':
        composition = self._compose_impl(variable2)
        if jacobians is not None:
            LieGroup._check_jacobians_list(jacobians)
            jacobians.extend(self._compose_jacobian(variable2))
        return composition

    @abc.abstractmethod
    def _inverse_impl(self) ->'LieGroup':
        pass

    def inverse(self, jacobian: Optional[List[torch.Tensor]]=None) ->'LieGroup':
        the_inverse = self._inverse_impl()
        if jacobian is not None:
            LieGroup._check_jacobians_list(jacobian)
            jacobian.append(self._inverse_jacobian(self))
        return the_inverse

    def _compose_jacobian(self, group2: 'LieGroup') ->Tuple[torch.Tensor, torch.Tensor]:
        if not type(self) is type(group2):
            raise ValueError('Lie groups for compose must be of the same type.')
        g2_inverse = group2._inverse_impl()
        jac1 = g2_inverse.adjoint()
        jac2 = torch.eye(group2.dof(), dtype=self.dtype).repeat(group2.shape[0], 1, 1)
        return jac1, jac2

    @staticmethod
    def _inverse_jacobian(group: 'LieGroup') ->torch.Tensor:
        return -group.adjoint()

    def _local_impl(self, variable2: Manifold, jacobians: List[torch.Tensor]=None) ->torch.Tensor:
        variable2 = cast(LieGroup, variable2)
        diff = self.between(variable2)
        if jacobians is not None:
            LieGroup._check_jacobians_list(jacobians)
            dlog: List[torch.Tensor] = []
            ret = diff.log_map(dlog)
            jacobians.append(-diff.inverse().adjoint() @ dlog[0])
            jacobians.append(dlog[0])
        else:
            ret = diff.log_map()
        return ret

    def _retract_impl(self, delta: torch.Tensor) ->'LieGroup':
        return self.compose(self.exp_map(delta))

    def copy(self, new_name: Optional[str]=None) ->'LieGroup':
        return cast(LieGroup, super().copy(new_name=new_name))


class _ScalarModel(nn.Module):

    def __init__(self, hidden_size: int):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(1, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1))

    def forward(self):
        dummy = torch.ones(1, 1)
        return self.layers(dummy)


class _OrderOfMagnitudeModel(nn.Module):

    def __init__(self, hidden_size: int, max_order: int):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(1, hidden_size), nn.ReLU(), nn.Linear(hidden_size, max_order), nn.ReLU())
        self.register_buffer('magnitudes', (10 ** torch.arange(max_order)).unsqueeze(0))

    def forward(self):
        dummy = torch.ones(1, 1)
        mag_weights = self.layers(dummy).softmax(dim=1)
        return (mag_weights * self.magnitudes).sum(dim=1, keepdim=True)


class ScalarCollisionWeightModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = _OrderOfMagnitudeModel(10, 5)

    def forward(self, _: Dict[str, torch.Tensor]):
        return {'collision_w': self.model()}


class ScalarCollisionWeightAndCostEpstModel(nn.Module):

    def __init__(self, robot_radius: float):
        super().__init__()
        self.collision_weight_model = _OrderOfMagnitudeModel(200, 5)
        self.safety_dist_model = _ScalarModel(100)
        self.robot_radius = robot_radius

    def forward(self, _: Dict[str, torch.Tensor]):
        collision_w = self.collision_weight_model()
        safety_dist = self.safety_dist_model().sigmoid()
        return {'collision_w': collision_w, 'cost_eps': safety_dist + self.robot_radius}


class TactileMeasModel(nn.Module):

    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, output_size)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, k: torch.Tensor):
        x = torch.cat([x1, x2], dim=1)
        k1_ = k.unsqueeze(1)
        x1_ = x.unsqueeze(-1)
        x = torch.mul(x1_, k1_)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ScalarCollisionWeightAndCostEpstModel,
     lambda: ([], {'robot_radius': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ScalarCollisionWeightModel,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SimpleCNN,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (SimpleNN,
     lambda: ([], {'in_size': 4, 'out_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (_OrderOfMagnitudeModel,
     lambda: ([], {'hidden_size': 4, 'max_order': 4}),
     lambda: ([], {}),
     True),
    (_ScalarModel,
     lambda: ([], {'hidden_size': 4}),
     lambda: ([], {}),
     True),
]

class Test_facebookresearch_theseus(_paritybench_base):
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

