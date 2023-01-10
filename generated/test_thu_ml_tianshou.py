import sys
_module = sys.modules[__name__]
del sys
conf = _module
examples = _module
atari = _module
atari_c51 = _module
atari_dqn = _module
atari_fqf = _module
atari_iqn = _module
atari_network = _module
atari_ppo = _module
atari_qrdqn = _module
atari_rainbow = _module
atari_sac = _module
atari_wrapper = _module
acrobot_dualdqn = _module
bipedal_bdq = _module
bipedal_hardcore_sac = _module
lunarlander_dqn = _module
mcc_sac = _module
irl_gail = _module
analysis = _module
fetch_her_ddpg = _module
gen_json = _module
mujoco_a2c = _module
mujoco_ddpg = _module
mujoco_env = _module
mujoco_npg = _module
mujoco_ppo = _module
mujoco_redq = _module
mujoco_reinforce = _module
mujoco_sac = _module
mujoco_td3 = _module
mujoco_trpo = _module
plotter = _module
tools = _module
offline = _module
atari_bcq = _module
atari_cql = _module
atari_crr = _module
atari_il = _module
convert_rl_unplugged_atari = _module
d4rl_bcq = _module
d4rl_cql = _module
d4rl_il = _module
d4rl_td3_bc = _module
utils = _module
env = _module
spectator = _module
replay = _module
vizdoom_c51 = _module
vizdoom_ppo = _module
setup = _module
test_nni = _module
test = _module
base = _module
test_batch = _module
test_buffer = _module
test_collector = _module
test_env = _module
test_env_finite = _module
test_returns = _module
test_utils = _module
continuous = _module
test_ddpg = _module
test_npg = _module
test_ppo = _module
test_redq = _module
test_sac_with_il = _module
test_td3 = _module
test_trpo = _module
discrete = _module
test_a2c_with_il = _module
test_bdq = _module
test_c51 = _module
test_dqn = _module
test_drqn = _module
test_fqf = _module
test_iqn = _module
test_pg = _module
test_ppo = _module
test_qrdqn = _module
test_rainbow = _module
test_sac = _module
modelbased = _module
test_dqn_icm = _module
test_ppo_icm = _module
test_psrl = _module
gather_cartpole_data = _module
gather_pendulum_data = _module
test_bcq = _module
test_cql = _module
test_discrete_bcq = _module
test_discrete_cql = _module
test_discrete_crr = _module
test_gail = _module
test_td3_bc = _module
pistonball = _module
pistonball_continuous = _module
test_pistonball = _module
test_pistonball_continuous = _module
test_tic_tac_toe = _module
tic_tac_toe = _module
throughput = _module
test_batch_profile = _module
test_buffer_profile = _module
test_collector_profile = _module
tianshou = _module
data = _module
batch = _module
buffer = _module
cached = _module
her = _module
manager = _module
prio = _module
vecbuf = _module
collector = _module
converter = _module
segtree = _module
gym_wrappers = _module
pettingzoo_env = _module
venv_wrappers = _module
venvs = _module
worker = _module
dummy = _module
ray = _module
subproc = _module
exploration = _module
random = _module
policy = _module
base = _module
imitation = _module
base = _module
bcq = _module
cql = _module
discrete_bcq = _module
discrete_cql = _module
discrete_crr = _module
gail = _module
td3_bc = _module
icm = _module
psrl = _module
modelfree = _module
a2c = _module
bdq = _module
c51 = _module
ddpg = _module
discrete_sac = _module
dqn = _module
fqf = _module
iqn = _module
npg = _module
pg = _module
ppo = _module
qrdqn = _module
rainbow = _module
redq = _module
sac = _module
td3 = _module
trpo = _module
multiagent = _module
mapolicy = _module
trainer = _module
offpolicy = _module
onpolicy = _module
logger = _module
tensorboard = _module
wandb = _module
lr_scheduler = _module
net = _module
common = _module
continuous = _module
discrete = _module
progress_bar = _module
statistics = _module
warning = _module

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


from torch.utils.tensorboard import SummaryWriter


from typing import Any


from typing import Callable


from typing import Dict


from typing import Optional


from typing import Sequence


from typing import Tuple


from typing import Type


from typing import Union


from torch import nn


from torch.optim.lr_scheduler import LambdaLR


from torch.distributions import Independent


from torch.distributions import Normal


import random


import time


from typing import List


import torch.nn.functional as F


import copy


from itertools import starmap


from collections import Counter


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


from torch.utils.data import DistributedSampler


import warnings


import torch.nn as nn


from copy import deepcopy


from functools import partial


from collections.abc import Collection


from numbers import Number


from typing import Iterable


from typing import Iterator


from typing import no_type_check


from abc import ABC


from abc import abstractmethod


from torch.nn.utils import clip_grad_norm_


import math


from torch.distributions import Categorical


from torch.distributions import kl_divergence


class DQN(nn.Module):
    """Reference: Human-level control through deep reinforcement learning.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(self, c: int, h: int, w: int, device: Union[str, int, torch.device]='cpu') ->None:
        super().__init__()
        self.device = device
        self.c = c
        self.h = h
        self.w = w
        self.net = nn.Sequential(nn.Conv2d(c, 32, kernel_size=8, stride=4), nn.ReLU(inplace=True), nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(inplace=True), nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(inplace=True), nn.Flatten())
        with torch.no_grad():
            self.output_dim = np.prod(self.net(torch.zeros(1, c, h, w)).shape[1:])

    def forward(self, x: Union[np.ndarray, torch.Tensor], state: Optional[Any]=None, info: Dict[str, Any]={}) ->Tuple[torch.Tensor, Any]:
        """Mapping: x -> Q(x, \\*)."""
        x = torch.as_tensor(x, device=self.device, dtype=torch.float32)
        return self.net(x.reshape(-1, self.c, self.w, self.h)), state


class C51(DQN):
    """Reference: A distributional perspective on reinforcement learning.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(self, c: int, h: int, w: int, action_shape: Sequence[int], num_atoms: int=51, device: Union[str, int, torch.device]='cpu') ->None:
        self.action_num = np.prod(action_shape)
        super().__init__(c, h, w, [self.action_num * num_atoms], device)
        self.num_atoms = num_atoms

    def forward(self, obs: Union[np.ndarray, torch.Tensor], state: Optional[Any]=None, info: Dict[str, Any]={}) ->Tuple[torch.Tensor, Any]:
        """Mapping: x -> Z(x, \\*)."""
        obs, state = super().forward(obs)
        obs = obs.view(-1, self.num_atoms).softmax(dim=-1)
        obs = obs.view(-1, self.action_num, self.num_atoms)
        return obs, state


class NoisyLinear(nn.Module):
    """Implementation of Noisy Networks. arXiv:1706.10295.

    :param int in_features: the number of input features.
    :param int out_features: the number of output features.
    :param float noisy_std: initial standard deviation of noisy linear layers.

    .. note::

        Adapted from https://github.com/ku2482/fqf-iqn-qrdqn.pytorch/blob/master
        /fqf_iqn_qrdqn/network.py .
    """

    def __init__(self, in_features: int, out_features: int, noisy_std: float=0.5) ->None:
        super().__init__()
        self.mu_W = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.sigma_W = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.mu_bias = nn.Parameter(torch.FloatTensor(out_features))
        self.sigma_bias = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('eps_p', torch.FloatTensor(in_features))
        self.register_buffer('eps_q', torch.FloatTensor(out_features))
        self.in_features = in_features
        self.out_features = out_features
        self.sigma = noisy_std
        self.reset()
        self.sample()

    def reset(self) ->None:
        bound = 1 / np.sqrt(self.in_features)
        self.mu_W.data.uniform_(-bound, bound)
        self.mu_bias.data.uniform_(-bound, bound)
        self.sigma_W.data.fill_(self.sigma / np.sqrt(self.in_features))
        self.sigma_bias.data.fill_(self.sigma / np.sqrt(self.in_features))

    def f(self, x: torch.Tensor) ->torch.Tensor:
        x = torch.randn(x.size(0), device=x.device)
        return x.sign().mul_(x.abs().sqrt_())

    def sample(self) ->None:
        self.eps_p.copy_(self.f(self.eps_p))
        self.eps_q.copy_(self.f(self.eps_q))

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        if self.training:
            weight = self.mu_W + self.sigma_W * self.eps_q.ger(self.eps_p)
            bias = self.mu_bias + self.sigma_bias * self.eps_q.clone()
        else:
            weight = self.mu_W
            bias = self.mu_bias
        return F.linear(x, weight, bias)


class Rainbow(DQN):
    """Reference: Rainbow: Combining Improvements in Deep Reinforcement Learning.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(self, c: int, h: int, w: int, action_shape: Sequence[int], num_atoms: int=51, noisy_std: float=0.5, device: Union[str, int, torch.device]='cpu', is_dueling: bool=True, is_noisy: bool=True) ->None:
        super().__init__(c, h, w, action_shape, device, features_only=True)
        self.action_num = np.prod(action_shape)
        self.num_atoms = num_atoms

        def linear(x, y):
            if is_noisy:
                return NoisyLinear(x, y, noisy_std)
            else:
                return nn.Linear(x, y)
        self.Q = nn.Sequential(linear(self.output_dim, 512), nn.ReLU(inplace=True), linear(512, self.action_num * self.num_atoms))
        self._is_dueling = is_dueling
        if self._is_dueling:
            self.V = nn.Sequential(linear(self.output_dim, 512), nn.ReLU(inplace=True), linear(512, self.num_atoms))
        self.output_dim = self.action_num * self.num_atoms

    def forward(self, obs: Union[np.ndarray, torch.Tensor], state: Optional[Any]=None, info: Dict[str, Any]={}) ->Tuple[torch.Tensor, Any]:
        """Mapping: x -> Z(x, \\*)."""
        obs, state = super().forward(obs)
        q = self.Q(obs)
        q = q.view(-1, self.action_num, self.num_atoms)
        if self._is_dueling:
            v = self.V(obs)
            v = v.view(-1, 1, self.num_atoms)
            logits = q - q.mean(dim=1, keepdim=True) + v
        else:
            logits = q
        probs = logits.softmax(dim=2)
        return probs, state


class QRDQN(DQN):
    """Reference: Distributional Reinforcement Learning with Quantile     Regression.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(self, c: int, h: int, w: int, action_shape: Sequence[int], num_quantiles: int=200, device: Union[str, int, torch.device]='cpu') ->None:
        self.action_num = np.prod(action_shape)
        super().__init__(c, h, w, [self.action_num * num_quantiles], device)
        self.num_quantiles = num_quantiles

    def forward(self, obs: Union[np.ndarray, torch.Tensor], state: Optional[Any]=None, info: Dict[str, Any]={}) ->Tuple[torch.Tensor, Any]:
        """Mapping: x -> Z(x, \\*)."""
        obs, state = super().forward(obs)
        obs = obs.view(-1, self.action_num, self.num_quantiles)
        return obs, state


ModuleType = Type[nn.Module]


def miniblock(input_size: int, output_size: int=0, norm_layer: Optional[ModuleType]=None, activation: Optional[ModuleType]=None, linear_layer: Type[nn.Linear]=nn.Linear) ->List[nn.Module]:
    """Construct a miniblock with given input/output-size, norm layer and     activation."""
    layers: List[nn.Module] = [linear_layer(input_size, output_size)]
    if norm_layer is not None:
        layers += [norm_layer(output_size)]
    if activation is not None:
        layers += [activation()]
    return layers


class MLP(nn.Module):
    """Simple MLP backbone.

    Create a MLP of size input_dim * hidden_sizes[0] * hidden_sizes[1] * ...
    * hidden_sizes[-1] * output_dim

    :param int input_dim: dimension of the input vector.
    :param int output_dim: dimension of the output vector. If set to 0, there
        is no final linear layer.
    :param hidden_sizes: shape of MLP passed in as a list, not including
        input_dim and output_dim.
    :param norm_layer: use which normalization before activation, e.g.,
        ``nn.LayerNorm`` and ``nn.BatchNorm1d``. Default to no normalization.
        You can also pass a list of normalization modules with the same length
        of hidden_sizes, to use different normalization module in different
        layers. Default to no normalization.
    :param activation: which activation to use after each layer, can be both
        the same activation for all layers if passed in nn.Module, or different
        activation for different Modules if passed in a list. Default to
        nn.ReLU.
    :param device: which device to create this model on. Default to None.
    :param linear_layer: use this module as linear layer. Default to nn.Linear.
    :param bool flatten_input: whether to flatten input data. Default to True.
    """

    def __init__(self, input_dim: int, output_dim: int=0, hidden_sizes: Sequence[int]=(), norm_layer: Optional[Union[ModuleType, Sequence[ModuleType]]]=None, activation: Optional[Union[ModuleType, Sequence[ModuleType]]]=nn.ReLU, device: Optional[Union[str, int, torch.device]]=None, linear_layer: Type[nn.Linear]=nn.Linear, flatten_input: bool=True) ->None:
        super().__init__()
        self.device = device
        if norm_layer:
            if isinstance(norm_layer, list):
                assert len(norm_layer) == len(hidden_sizes)
                norm_layer_list = norm_layer
            else:
                norm_layer_list = [norm_layer for _ in range(len(hidden_sizes))]
        else:
            norm_layer_list = [None] * len(hidden_sizes)
        if activation:
            if isinstance(activation, list):
                assert len(activation) == len(hidden_sizes)
                activation_list = activation
            else:
                activation_list = [activation for _ in range(len(hidden_sizes))]
        else:
            activation_list = [None] * len(hidden_sizes)
        hidden_sizes = [input_dim] + list(hidden_sizes)
        model = []
        for in_dim, out_dim, norm, activ in zip(hidden_sizes[:-1], hidden_sizes[1:], norm_layer_list, activation_list):
            model += miniblock(in_dim, out_dim, norm, activ, linear_layer)
        if output_dim > 0:
            model += [linear_layer(hidden_sizes[-1], output_dim)]
        self.output_dim = output_dim or hidden_sizes[-1]
        self.model = nn.Sequential(*model)
        self.flatten_input = flatten_input

    @no_type_check
    def forward(self, obs: Union[np.ndarray, torch.Tensor]) ->torch.Tensor:
        if self.device is not None:
            obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        if self.flatten_input:
            obs = obs.flatten(1)
        return self.model(obs)


class Net(nn.Module):
    """Wrapper of MLP to support more specific DRL usage.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.

    :param state_shape: int or a sequence of int of the shape of state.
    :param action_shape: int or a sequence of int of the shape of action.
    :param hidden_sizes: shape of MLP passed in as a list.
    :param norm_layer: use which normalization before activation, e.g.,
        ``nn.LayerNorm`` and ``nn.BatchNorm1d``. Default to no normalization.
        You can also pass a list of normalization modules with the same length
        of hidden_sizes, to use different normalization module in different
        layers. Default to no normalization.
    :param activation: which activation to use after each layer, can be both
        the same activation for all layers if passed in nn.Module, or different
        activation for different Modules if passed in a list. Default to
        nn.ReLU.
    :param device: specify the device when the network actually runs. Default
        to "cpu".
    :param bool softmax: whether to apply a softmax layer over the last layer's
        output.
    :param bool concat: whether the input shape is concatenated by state_shape
        and action_shape. If it is True, ``action_shape`` is not the output
        shape, but affects the input shape only.
    :param int num_atoms: in order to expand to the net of distributional RL.
        Default to 1 (not use).
    :param bool dueling_param: whether to use dueling network to calculate Q
        values (for Dueling DQN). If you want to use dueling option, you should
        pass a tuple of two dict (first for Q and second for V) stating
        self-defined arguments as stated in
        class:`~tianshou.utils.net.common.MLP`. Default to None.
    :param linear_layer: use this module as linear layer. Default to nn.Linear.

    .. seealso::

        Please refer to :class:`~tianshou.utils.net.common.MLP` for more
        detailed explanation on the usage of activation, norm_layer, etc.

        You can also refer to :class:`~tianshou.utils.net.continuous.Actor`,
        :class:`~tianshou.utils.net.continuous.Critic`, etc, to see how it's
        suggested be used.
    """

    def __init__(self, state_shape: Union[int, Sequence[int]], action_shape: Union[int, Sequence[int]]=0, hidden_sizes: Sequence[int]=(), norm_layer: Optional[ModuleType]=None, activation: Optional[ModuleType]=nn.ReLU, device: Union[str, int, torch.device]='cpu', softmax: bool=False, concat: bool=False, num_atoms: int=1, dueling_param: Optional[Tuple[Dict[str, Any], Dict[str, Any]]]=None, linear_layer: Type[nn.Linear]=nn.Linear) ->None:
        super().__init__()
        self.device = device
        self.softmax = softmax
        self.num_atoms = num_atoms
        input_dim = int(np.prod(state_shape))
        action_dim = int(np.prod(action_shape)) * num_atoms
        if concat:
            input_dim += action_dim
        self.use_dueling = dueling_param is not None
        output_dim = action_dim if not self.use_dueling and not concat else 0
        self.model = MLP(input_dim, output_dim, hidden_sizes, norm_layer, activation, device, linear_layer)
        self.output_dim = self.model.output_dim
        if self.use_dueling:
            q_kwargs, v_kwargs = dueling_param
            q_output_dim, v_output_dim = 0, 0
            if not concat:
                q_output_dim, v_output_dim = action_dim, num_atoms
            q_kwargs: Dict[str, Any] = {**q_kwargs, 'input_dim': self.output_dim, 'output_dim': q_output_dim, 'device': self.device}
            v_kwargs: Dict[str, Any] = {**v_kwargs, 'input_dim': self.output_dim, 'output_dim': v_output_dim, 'device': self.device}
            self.Q, self.V = MLP(**q_kwargs), MLP(**v_kwargs)
            self.output_dim = self.Q.output_dim

    def forward(self, obs: Union[np.ndarray, torch.Tensor], state: Any=None, info: Dict[str, Any]={}) ->Tuple[torch.Tensor, Any]:
        """Mapping: obs -> flatten (inside MLP)-> logits."""
        logits = self.model(obs)
        bsz = logits.shape[0]
        if self.use_dueling:
            q, v = self.Q(logits), self.V(logits)
            if self.num_atoms > 1:
                q = q.view(bsz, -1, self.num_atoms)
                v = v.view(bsz, -1, self.num_atoms)
            logits = q - q.mean(dim=1, keepdim=True) + v
        elif self.num_atoms > 1:
            logits = logits.view(bsz, -1, self.num_atoms)
        if self.softmax:
            logits = torch.softmax(logits, dim=-1)
        return logits, state


IndexType = Union[slice, int, np.ndarray, List[int]]


def _assert_type_keys(keys: Iterable[str]) ->None:
    assert all(isinstance(key, str) for key in keys), f'keys should all be string, but got {keys}'


def _is_scalar(value: Any) ->bool:
    if isinstance(value, torch.Tensor):
        return value.numel() == 1 and not value.shape
    else:
        return np.isscalar(value)


def _create_value(inst: Any, size: int, stack: bool=True) ->Union['Batch', np.ndarray, torch.Tensor]:
    """Create empty place-holders accroding to inst's shape.

    :param bool stack: whether to stack or to concatenate. E.g. if inst has shape of
        (3, 5), size = 10, stack=True returns an np.ndarry with shape of (10, 3, 5),
        otherwise (10, 5)
    """
    has_shape = isinstance(inst, (np.ndarray, torch.Tensor))
    is_scalar = _is_scalar(inst)
    if not stack and is_scalar:
        raise TypeError(f'cannot concatenate with {inst} which is scalar')
    if has_shape:
        shape = (size, *inst.shape) if stack else (size, *inst.shape[1:])
    if isinstance(inst, np.ndarray):
        target_type = inst.dtype.type if issubclass(inst.dtype.type, (np.bool_, np.number)) else object
        return np.full(shape, fill_value=None if target_type == object else 0, dtype=target_type)
    elif isinstance(inst, torch.Tensor):
        return torch.full(shape, fill_value=0, device=inst.device, dtype=inst.dtype)
    elif isinstance(inst, (dict, Batch)):
        zero_batch = Batch()
        for key, val in inst.items():
            zero_batch.__dict__[key] = _create_value(val, size, stack=stack)
        return zero_batch
    elif is_scalar:
        return _create_value(np.asarray(inst), size, stack=stack)
    else:
        return np.array([None for _ in range(size)], object)


def _is_batch_set(obj: Any) ->bool:
    if isinstance(obj, np.ndarray):
        return obj.dtype == object and all(isinstance(element, (dict, Batch)) for element in obj)
    elif isinstance(obj, (list, tuple)):
        if len(obj) > 0 and all(isinstance(element, (dict, Batch)) for element in obj):
            return True
    return False


def _is_number(value: Any) ->bool:
    return isinstance(value, (Number, np.number, np.bool_))


def _to_array_with_correct_type(obj: Any) ->np.ndarray:
    if isinstance(obj, np.ndarray) and issubclass(obj.dtype.type, (np.bool_, np.number)):
        return obj
    obj_array = np.asanyarray(obj)
    if not issubclass(obj_array.dtype.type, (np.bool_, np.number)):
        obj_array = obj_array.astype(object)
    if obj_array.dtype == object:
        if not obj_array.shape:
            obj_array = obj_array.item(0)
        elif all(isinstance(arr, np.ndarray) for arr in obj_array.reshape(-1)):
            return obj_array
        elif any(isinstance(arr, torch.Tensor) for arr in obj_array.reshape(-1)):
            raise ValueError('Numpy arrays of tensors are not supported yet.')
    return obj_array


def _parse_value(obj: Any) ->Optional[Union['Batch', np.ndarray, torch.Tensor]]:
    if isinstance(obj, Batch):
        return obj
    elif isinstance(obj, np.ndarray) and issubclass(obj.dtype.type, (np.bool_, np.number)) or isinstance(obj, torch.Tensor) or obj is None:
        return obj
    elif _is_number(obj):
        return np.asanyarray(obj)
    elif isinstance(obj, dict):
        return Batch(obj)
    else:
        if not isinstance(obj, np.ndarray) and isinstance(obj, Collection) and len(obj) > 0 and all(isinstance(element, torch.Tensor) for element in obj):
            try:
                return torch.stack(obj)
            except RuntimeError as exception:
                raise TypeError('Batch does not support non-stackable iterable of torch.Tensor as unique value yet.') from exception
        if _is_batch_set(obj):
            obj = Batch(obj)
        else:
            try:
                obj = _to_array_with_correct_type(obj)
            except ValueError as exception:
                raise TypeError('Batch does not support heterogeneous list/tuple of tensors as unique value yet.') from exception
        return obj


class Batch:
    """The internal data structure in Tianshou.

    Batch is a kind of supercharged array (of temporal data) stored individually in a
    (recursive) dictionary of object that can be either numpy array, torch tensor, or
    batch themselves. It is designed to make it extremely easily to access, manipulate
    and set partial view of the heterogeneous data conveniently.

    For a detailed description, please refer to :ref:`batch_concept`.
    """

    def __init__(self, batch_dict: Optional[Union[dict, 'Batch', Sequence[Union[dict, 'Batch']], np.ndarray]]=None, copy: bool=False, **kwargs: Any) ->None:
        if copy:
            batch_dict = deepcopy(batch_dict)
        if batch_dict is not None:
            if isinstance(batch_dict, (dict, Batch)):
                _assert_type_keys(batch_dict.keys())
                for batch_key, obj in batch_dict.items():
                    self.__dict__[batch_key] = _parse_value(obj)
            elif _is_batch_set(batch_dict):
                self.stack_(batch_dict)
        if len(kwargs) > 0:
            self.__init__(kwargs, copy=copy)

    def __setattr__(self, key: str, value: Any) ->None:
        """Set self.key = value."""
        self.__dict__[key] = _parse_value(value)

    def __getattr__(self, key: str) ->Any:
        """Return self.key. The "Any" return type is needed for mypy."""
        return getattr(self.__dict__, key)

    def __contains__(self, key: str) ->bool:
        """Return key in self."""
        return key in self.__dict__

    def __getstate__(self) ->Dict[str, Any]:
        """Pickling interface.

        Only the actual data are serialized for both efficiency and simplicity.
        """
        state = {}
        for batch_key, obj in self.items():
            if isinstance(obj, Batch):
                obj = obj.__getstate__()
            state[batch_key] = obj
        return state

    def __setstate__(self, state: Dict[str, Any]) ->None:
        """Unpickling interface.

        At this point, self is an empty Batch instance that has not been
        initialized, so it can safely be initialized by the pickle state.
        """
        self.__init__(**state)

    def __getitem__(self, index: Union[str, IndexType]) ->Any:
        """Return self[index]."""
        if isinstance(index, str):
            return self.__dict__[index]
        batch_items = self.items()
        if len(batch_items) > 0:
            new_batch = Batch()
            for batch_key, obj in batch_items:
                if isinstance(obj, Batch) and obj.is_empty():
                    new_batch.__dict__[batch_key] = Batch()
                else:
                    new_batch.__dict__[batch_key] = obj[index]
            return new_batch
        else:
            raise IndexError('Cannot access item from empty Batch object.')

    def __setitem__(self, index: Union[str, IndexType], value: Any) ->None:
        """Assign value to self[index]."""
        value = _parse_value(value)
        if isinstance(index, str):
            self.__dict__[index] = value
            return
        if not isinstance(value, Batch):
            raise ValueError('Batch does not supported tensor assignment. Use a compatible Batch or dict instead.')
        if not set(value.keys()).issubset(self.__dict__.keys()):
            raise ValueError('Creating keys is not supported by item assignment.')
        for key, val in self.items():
            try:
                self.__dict__[key][index] = value[key]
            except KeyError:
                if isinstance(val, Batch):
                    self.__dict__[key][index] = Batch()
                elif isinstance(val, torch.Tensor) or isinstance(val, np.ndarray) and issubclass(val.dtype.type, (np.bool_, np.number)):
                    self.__dict__[key][index] = 0
                else:
                    self.__dict__[key][index] = None

    def __iadd__(self, other: Union['Batch', Number, np.number]) ->'Batch':
        """Algebraic addition with another Batch instance in-place."""
        if isinstance(other, Batch):
            for (batch_key, obj), value in zip(self.__dict__.items(), other.__dict__.values()):
                if isinstance(obj, Batch) and obj.is_empty():
                    continue
                else:
                    self.__dict__[batch_key] += value
            return self
        elif _is_number(other):
            for batch_key, obj in self.items():
                if isinstance(obj, Batch) and obj.is_empty():
                    continue
                else:
                    self.__dict__[batch_key] += other
            return self
        else:
            raise TypeError('Only addition of Batch or number is supported.')

    def __add__(self, other: Union['Batch', Number, np.number]) ->'Batch':
        """Algebraic addition with another Batch instance out-of-place."""
        return deepcopy(self).__iadd__(other)

    def __imul__(self, value: Union[Number, np.number]) ->'Batch':
        """Algebraic multiplication with a scalar value in-place."""
        assert _is_number(value), 'Only multiplication by a number is supported.'
        for batch_key, obj in self.__dict__.items():
            if isinstance(obj, Batch) and obj.is_empty():
                continue
            self.__dict__[batch_key] *= value
        return self

    def __mul__(self, value: Union[Number, np.number]) ->'Batch':
        """Algebraic multiplication with a scalar value out-of-place."""
        return deepcopy(self).__imul__(value)

    def __itruediv__(self, value: Union[Number, np.number]) ->'Batch':
        """Algebraic division with a scalar value in-place."""
        assert _is_number(value), 'Only division by a number is supported.'
        for batch_key, obj in self.__dict__.items():
            if isinstance(obj, Batch) and obj.is_empty():
                continue
            self.__dict__[batch_key] /= value
        return self

    def __truediv__(self, value: Union[Number, np.number]) ->'Batch':
        """Algebraic division with a scalar value out-of-place."""
        return deepcopy(self).__itruediv__(value)

    def __repr__(self) ->str:
        """Return str(self)."""
        self_str = self.__class__.__name__ + '(\n'
        flag = False
        for batch_key, obj in self.__dict__.items():
            rpl = '\n' + ' ' * (6 + len(batch_key))
            obj_name = pprint.pformat(obj).replace('\n', rpl)
            self_str += f'    {batch_key}: {obj_name},\n'
            flag = True
        if flag:
            self_str += ')'
        else:
            self_str = self.__class__.__name__ + '()'
        return self_str

    def to_numpy(self) ->None:
        """Change all torch.Tensor to numpy.ndarray in-place."""
        for batch_key, obj in self.items():
            if isinstance(obj, torch.Tensor):
                self.__dict__[batch_key] = obj.detach().cpu().numpy()
            elif isinstance(obj, Batch):
                obj.to_numpy()

    def to_torch(self, dtype: Optional[torch.dtype]=None, device: Union[str, int, torch.device]='cpu') ->None:
        """Change all numpy.ndarray to torch.Tensor in-place."""
        if not isinstance(device, torch.device):
            device = torch.device(device)
        for batch_key, obj in self.items():
            if isinstance(obj, torch.Tensor):
                if dtype is not None and obj.dtype != dtype or obj.device.type != device.type or device.index != obj.device.index:
                    if dtype is not None:
                        obj = obj.type(dtype)
                    self.__dict__[batch_key] = obj
            elif isinstance(obj, Batch):
                obj.to_torch(dtype, device)
            else:
                if not isinstance(obj, np.ndarray):
                    obj = np.asanyarray(obj)
                obj = torch.from_numpy(obj)
                if dtype is not None:
                    obj = obj.type(dtype)
                self.__dict__[batch_key] = obj

    def __cat(self, batches: Sequence[Union[dict, 'Batch']], lens: List[int]) ->None:
        """Private method for Batch.cat_.

        ::

            >>> a = Batch(a=np.random.randn(3, 4))
            >>> x = Batch(a=a, b=np.random.randn(4, 4))
            >>> y = Batch(a=Batch(a=Batch()), b=np.random.randn(4, 4))

        If we want to concatenate x and y, we want to pad y.a.a with zeros.
        Without ``lens`` as a hint, when we concatenate x.a and y.a, we would
        not be able to know how to pad y.a. So ``Batch.cat_`` should compute
        the ``lens`` to give ``Batch.__cat`` a hint.
        ::

            >>> ans = Batch.cat([x, y])
            >>> # this is equivalent to the following line
            >>> ans = Batch(); ans.__cat([x, y], lens=[3, 4])
            >>> # this lens is equal to [len(a), len(b)]
        """
        sum_lens = [0]
        for len_ in lens:
            sum_lens.append(sum_lens[-1] + len_)
        keys_map = [set(batch_key for batch_key, obj in batch.items() if not (isinstance(obj, Batch) and obj.is_empty())) for batch in batches]
        keys_shared = set.intersection(*keys_map)
        values_shared = [[batch[key] for batch in batches] for key in keys_shared]
        for key, shared_value in zip(keys_shared, values_shared):
            if all(isinstance(element, (dict, Batch)) for element in shared_value):
                batch_holder = Batch()
                batch_holder.__cat(shared_value, lens=lens)
                self.__dict__[key] = batch_holder
            elif all(isinstance(element, torch.Tensor) for element in shared_value):
                self.__dict__[key] = torch.cat(shared_value)
            else:
                shared_value = np.concatenate(shared_value)
                self.__dict__[key] = _to_array_with_correct_type(shared_value)
        keys_total = set.union(*[set(batch.keys()) for batch in batches])
        keys_reserve_or_partial = set.difference(keys_total, keys_shared)
        keys_reserve = set.difference(keys_total, set.union(*keys_map))
        keys_partial = keys_reserve_or_partial.difference(keys_reserve)
        for key in keys_reserve:
            self.__dict__[key] = Batch()
        for key in keys_partial:
            for i, batch in enumerate(batches):
                if key not in batch.__dict__:
                    continue
                value = batch.get(key)
                if isinstance(value, Batch) and value.is_empty():
                    continue
                try:
                    self.__dict__[key][sum_lens[i]:sum_lens[i + 1]] = value
                except KeyError:
                    self.__dict__[key] = _create_value(value, sum_lens[-1], stack=False)
                    self.__dict__[key][sum_lens[i]:sum_lens[i + 1]] = value

    def cat_(self, batches: Union['Batch', Sequence[Union[dict, 'Batch']]]) ->None:
        """Concatenate a list of (or one) Batch objects into current batch."""
        if isinstance(batches, Batch):
            batches = [batches]
        batch_list = []
        for batch in batches:
            if isinstance(batch, dict):
                if len(batch) > 0:
                    batch_list.append(Batch(batch))
            elif isinstance(batch, Batch):
                if not batch.is_empty():
                    batch_list.append(batch)
            else:
                raise ValueError(f'Cannot concatenate {type(batch)} in Batch.cat_')
        if len(batch_list) == 0:
            return
        batches = batch_list
        try:
            lens = [(0 if batch.is_empty(recurse=True) else len(batch)) for batch in batches]
        except TypeError as exception:
            raise ValueError(f'Batch.cat_ meets an exception. Maybe because there is any scalar in {batches} but Batch.cat_ does not support the concatenation of scalar.') from exception
        if not self.is_empty():
            batches = [self] + list(batches)
            lens = [0 if self.is_empty(recurse=True) else len(self)] + lens
        self.__cat(batches, lens)

    @staticmethod
    def cat(batches: Sequence[Union[dict, 'Batch']]) ->'Batch':
        """Concatenate a list of Batch object into a single new batch.

        For keys that are not shared across all batches, batches that do not
        have these keys will be padded by zeros with appropriate shapes. E.g.
        ::

            >>> a = Batch(a=np.zeros([3, 4]), common=Batch(c=np.zeros([3, 5])))
            >>> b = Batch(b=np.zeros([4, 3]), common=Batch(c=np.zeros([4, 5])))
            >>> c = Batch.cat([a, b])
            >>> c.a.shape
            (7, 4)
            >>> c.b.shape
            (7, 3)
            >>> c.common.c.shape
            (7, 5)
        """
        batch = Batch()
        batch.cat_(batches)
        return batch

    def stack_(self, batches: Sequence[Union[dict, 'Batch']], axis: int=0) ->None:
        """Stack a list of Batch object into current batch."""
        batch_list = []
        for batch in batches:
            if isinstance(batch, dict):
                if len(batch) > 0:
                    batch_list.append(Batch(batch))
            elif isinstance(batch, Batch):
                if not batch.is_empty():
                    batch_list.append(batch)
            else:
                raise ValueError(f'Cannot concatenate {type(batch)} in Batch.stack_')
        if len(batch_list) == 0:
            return
        batches = batch_list
        if not self.is_empty():
            batches = [self] + batches
        keys_map = [set(batch_key for batch_key, obj in batch.items() if not (isinstance(obj, Batch) and obj.is_empty())) for batch in batches]
        keys_shared = set.intersection(*keys_map)
        values_shared = [[batch[key] for batch in batches] for key in keys_shared]
        for shared_key, value in zip(keys_shared, values_shared):
            if all(isinstance(element, torch.Tensor) for element in value):
                self.__dict__[shared_key] = torch.stack(value, axis)
            elif all(isinstance(element, (Batch, dict)) for element in value):
                self.__dict__[shared_key] = Batch.stack(value, axis)
            else:
                try:
                    self.__dict__[shared_key] = _to_array_with_correct_type(np.stack(value, axis))
                except ValueError:
                    warnings.warn('You are using tensors with different shape, fallback to dtype=object by default.')
                    self.__dict__[shared_key] = np.array(value, dtype=object)
        keys_total = set.union(*[set(batch.keys()) for batch in batches])
        keys_reserve = set.difference(keys_total, set.union(*keys_map))
        keys_reserve_or_partial = set.difference(keys_total, keys_shared)
        keys_partial = keys_reserve_or_partial.difference(keys_reserve)
        if keys_partial and axis != 0:
            raise ValueError(f'Stack of Batch with non-shared keys {keys_partial} is only supported with axis=0, but got axis={axis}!')
        for key in keys_reserve:
            self.__dict__[key] = Batch()
        for key in keys_partial:
            for i, batch in enumerate(batches):
                if key not in batch.__dict__:
                    continue
                value = batch.get(key)
                if isinstance(value, Batch) and value.is_empty():
                    continue
                try:
                    self.__dict__[key][i] = value
                except KeyError:
                    self.__dict__[key] = _create_value(value, len(batches))
                    self.__dict__[key][i] = value

    @staticmethod
    def stack(batches: Sequence[Union[dict, 'Batch']], axis: int=0) ->'Batch':
        """Stack a list of Batch object into a single new batch.

        For keys that are not shared across all batches, batches that do not
        have these keys will be padded by zeros. E.g.
        ::

            >>> a = Batch(a=np.zeros([4, 4]), common=Batch(c=np.zeros([4, 5])))
            >>> b = Batch(b=np.zeros([4, 6]), common=Batch(c=np.zeros([4, 5])))
            >>> c = Batch.stack([a, b])
            >>> c.a.shape
            (2, 4, 4)
            >>> c.b.shape
            (2, 4, 6)
            >>> c.common.c.shape
            (2, 4, 5)

        .. note::

            If there are keys that are not shared across all batches, ``stack``
            with ``axis != 0`` is undefined, and will cause an exception.
        """
        batch = Batch()
        batch.stack_(batches, axis)
        return batch

    def empty_(self, index: Optional[Union[slice, IndexType]]=None) ->'Batch':
        """Return an empty Batch object with 0 or None filled.

        If "index" is specified, it will only reset the specific indexed-data.
        ::

            >>> data.empty_()
            >>> print(data)
            Batch(
                a: array([[0., 0.],
                          [0., 0.]]),
                b: array([None, None], dtype=object),
            )
            >>> b={'c': [2., 'st'], 'd': [1., 0.]}
            >>> data = Batch(a=[False,  True], b=b)
            >>> data[0] = Batch.empty(data[1])
            >>> data
            Batch(
                a: array([False,  True]),
                b: Batch(
                       c: array([None, 'st']),
                       d: array([0., 0.]),
                   ),
            )
        """
        for batch_key, obj in self.items():
            if isinstance(obj, torch.Tensor):
                self.__dict__[batch_key][index] = 0
            elif obj is None:
                continue
            elif isinstance(obj, np.ndarray):
                if obj.dtype == object:
                    self.__dict__[batch_key][index] = None
                else:
                    self.__dict__[batch_key][index] = 0
            elif isinstance(obj, Batch):
                self.__dict__[batch_key].empty_(index=index)
            else:
                warnings.warn('You are calling Batch.empty on a NumPy scalar, which may cause undefined behaviors.')
                if _is_number(obj):
                    self.__dict__[batch_key] = obj.__class__(0)
                else:
                    self.__dict__[batch_key] = None
        return self

    @staticmethod
    def empty(batch: 'Batch', index: Optional[IndexType]=None) ->'Batch':
        """Return an empty Batch object with 0 or None filled.

        The shape is the same as the given Batch.
        """
        return deepcopy(batch).empty_(index)

    def update(self, batch: Optional[Union[dict, 'Batch']]=None, **kwargs: Any) ->None:
        """Update this batch from another dict/Batch."""
        if batch is None:
            self.update(kwargs)
            return
        for batch_key, obj in batch.items():
            self.__dict__[batch_key] = _parse_value(obj)
        if kwargs:
            self.update(kwargs)

    def __len__(self) ->int:
        """Return len(self)."""
        lens = []
        for obj in self.__dict__.values():
            if isinstance(obj, Batch) and obj.is_empty(recurse=True):
                continue
            elif hasattr(obj, '__len__') and (isinstance(obj, Batch) or obj.ndim > 0):
                lens.append(len(obj))
            else:
                raise TypeError(f'Object {obj} in {self} has no len()')
        if len(lens) == 0:
            raise TypeError(f'Object {self} has no len()')
        return min(lens)

    def is_empty(self, recurse: bool=False) ->bool:
        """Test if a Batch is empty.

        If ``recurse=True``, it further tests the values of the object; else
        it only tests the existence of any key.

        ``b.is_empty(recurse=True)`` is mainly used to distinguish
        ``Batch(a=Batch(a=Batch()))`` and ``Batch(a=1)``. They both raise
        exceptions when applied to ``len()``, but the former can be used in
        ``cat``, while the latter is a scalar and cannot be used in ``cat``.

        Another usage is in ``__len__``, where we have to skip checking the
        length of recursively empty Batch.
        ::

            >>> Batch().is_empty()
            True
            >>> Batch(a=Batch(), b=Batch(c=Batch())).is_empty()
            False
            >>> Batch(a=Batch(), b=Batch(c=Batch())).is_empty(recurse=True)
            True
            >>> Batch(d=1).is_empty()
            False
            >>> Batch(a=np.float64(1.0)).is_empty()
            False
        """
        if len(self.__dict__) == 0:
            return True
        if not recurse:
            return False
        return all(False if not isinstance(obj, Batch) else obj.is_empty(recurse=True) for obj in self.values())

    @property
    def shape(self) ->List[int]:
        """Return self.shape."""
        if self.is_empty():
            return []
        else:
            data_shape = []
            for obj in self.__dict__.values():
                try:
                    data_shape.append(list(obj.shape))
                except AttributeError:
                    data_shape.append([])
            return list(map(min, zip(*data_shape))) if len(data_shape) > 1 else data_shape[0]

    def split(self, size: int, shuffle: bool=True, merge_last: bool=False) ->Iterator['Batch']:
        """Split whole data into multiple small batches.

        :param int size: divide the data batch with the given size, but one
            batch if the length of the batch is smaller than "size".
        :param bool shuffle: randomly shuffle the entire data batch if it is
            True, otherwise remain in the same. Default to True.
        :param bool merge_last: merge the last batch into the previous one.
            Default to False.
        """
        length = len(self)
        assert 1 <= size
        if shuffle:
            indices = np.random.permutation(length)
        else:
            indices = np.arange(length)
        merge_last = merge_last and length % size > 0
        for idx in range(0, length, size):
            if merge_last and idx + size + size >= length:
                yield self[indices[idx:]]
                break
            yield self[indices[idx:idx + size]]


class MultipleLRSchedulers:
    """A wrapper for multiple learning rate schedulers.

    Every time :meth:`~tianshou.utils.MultipleLRSchedulers.step` is called,
    it calls the step() method of each of the schedulers that it contains.
    Example usage:
    ::

        scheduler1 = ConstantLR(opt1, factor=0.1, total_iters=2)
        scheduler2 = ExponentialLR(opt2, gamma=0.9)
        scheduler = MultipleLRSchedulers(scheduler1, scheduler2)
        policy = PPOPolicy(..., lr_scheduler=scheduler)
    """

    def __init__(self, *args: torch.optim.lr_scheduler.LambdaLR):
        self.schedulers = args

    def step(self) ->None:
        """Take a step in each of the learning rate schedulers."""
        for scheduler in self.schedulers:
            scheduler.step()

    def state_dict(self) ->List[Dict]:
        """Get state_dict for each of the learning rate schedulers.

        :return: A list of state_dict of learning rate schedulers.
        """
        return [s.state_dict() for s in self.schedulers]

    def load_state_dict(self, state_dict: List[Dict]) ->None:
        """Load states from state_dict.

        :param List[Dict] state_dict: A list of learning rate scheduler
            state_dict, in the same order as the schedulers.
        """
        for s, sd in zip(self.schedulers, state_dict):
            s.__dict__.update(sd)


def _alloc_by_keys_diff(meta: 'Batch', batch: 'Batch', size: int, stack: bool=True) ->None:
    for key in batch.keys():
        if key in meta.keys():
            if isinstance(meta[key], Batch) and isinstance(batch[key], Batch):
                _alloc_by_keys_diff(meta[key], batch[key], size, stack)
            elif isinstance(meta[key], Batch) and meta[key].is_empty():
                meta[key] = _create_value(batch[key], size, stack)
        else:
            meta[key] = _create_value(batch[key], size, stack)


Hdf5ConvertibleValues = Union[int, float, Batch, np.ndarray, torch.Tensor, object, 'Hdf5ConvertibleType']


Hdf5ConvertibleType = Dict[str, Hdf5ConvertibleValues]


@no_type_check
def to_numpy(x: Any) ->Union[Batch, np.ndarray]:
    """Return an object without torch.Tensor."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        return x
    elif isinstance(x, (np.number, np.bool_, Number)):
        return np.asanyarray(x)
    elif x is None:
        return np.array(None, dtype=object)
    elif isinstance(x, (dict, Batch)):
        x = Batch(x) if isinstance(x, dict) else deepcopy(x)
        x.to_numpy()
        return x
    elif isinstance(x, (list, tuple)):
        return to_numpy(_parse_value(x))
    else:
        return np.asanyarray(x)


@no_type_check
def to_torch(x: Any, dtype: Optional[torch.dtype]=None, device: Union[str, int, torch.device]='cpu') ->Union[Batch, torch.Tensor]:
    """Return an object without np.ndarray."""
    if isinstance(x, np.ndarray) and issubclass(x.dtype.type, (np.bool_, np.number)):
        x = torch.from_numpy(x)
        if dtype is not None:
            x = x.type(dtype)
        return x
    elif isinstance(x, torch.Tensor):
        if dtype is not None:
            x = x.type(dtype)
        return x
    elif isinstance(x, (np.number, np.bool_, Number)):
        return to_torch(np.asanyarray(x), dtype, device)
    elif isinstance(x, (dict, Batch)):
        x = Batch(x, copy=True) if isinstance(x, dict) else deepcopy(x)
        x.to_torch(dtype, device)
        return x
    elif isinstance(x, (list, tuple)):
        return to_torch(_parse_value(x), dtype, device)
    else:
        raise TypeError(f'object {x} cannot be converted to torch.')


@no_type_check
def to_torch_as(x: Any, y: torch.Tensor) ->Union[Batch, torch.Tensor]:
    """Return an object without np.ndarray.

    Same as ``to_torch(x, dtype=y.dtype, device=y.device)``.
    """
    assert isinstance(y, torch.Tensor)
    return to_torch(x, dtype=y.dtype, device=y.device)


class VAE(nn.Module):
    """Implementation of VAE. It models the distribution of action. Given a     state, it can generate actions similar to those in batch. It is used     in BCQ algorithm.

    :param torch.nn.Module encoder: the encoder in VAE. Its input_dim must be
        state_dim + action_dim, and output_dim must be hidden_dim.
    :param torch.nn.Module decoder: the decoder in VAE. Its input_dim must be
        state_dim + latent_dim, and output_dim must be action_dim.
    :param int hidden_dim: the size of the last linear-layer in encoder.
    :param int latent_dim: the size of latent layer.
    :param float max_action: the maximum value of each dimension of action.
    :param Union[str, torch.device] device: which device to create this model on.
        Default to "cpu".

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.

    .. seealso::

        You can refer to `examples/offline/offline_bcq.py` to see how to use it.
    """

    def __init__(self, encoder: nn.Module, decoder: nn.Module, hidden_dim: int, latent_dim: int, max_action: float, device: Union[str, torch.device]='cpu'):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.log_std = nn.Linear(hidden_dim, latent_dim)
        self.decoder = decoder
        self.max_action = max_action
        self.latent_dim = latent_dim
        self.device = device

    def forward(self, state: torch.Tensor, action: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent_z = self.encoder(torch.cat([state, action], -1))
        mean = self.mean(latent_z)
        log_std = self.log_std(latent_z).clamp(-4, 15)
        std = torch.exp(log_std)
        latent_z = mean + std * torch.randn_like(std)
        reconstruction = self.decode(state, latent_z)
        return reconstruction, mean, std

    def decode(self, state: torch.Tensor, latent_z: Union[torch.Tensor, None]=None) ->torch.Tensor:
        if latent_z is None:
            latent_z = torch.randn(state.shape[:-1] + (self.latent_dim,)).clamp(-0.5, 0.5)
        return self.max_action * torch.tanh(self.decoder(torch.cat([state, latent_z], -1)))


class IntrinsicCuriosityModule(nn.Module):
    """Implementation of Intrinsic Curiosity Module. arXiv:1705.05363.

    :param torch.nn.Module feature_net: a self-defined feature_net which output a
        flattened hidden state.
    :param int feature_dim: input dimension of the feature net.
    :param int action_dim: dimension of the action space.
    :param hidden_sizes: hidden layer sizes for forward and inverse models.
    :param device: device for the module.
    """

    def __init__(self, feature_net: nn.Module, feature_dim: int, action_dim: int, hidden_sizes: Sequence[int]=(), device: Union[str, torch.device]='cpu') ->None:
        super().__init__()
        self.feature_net = feature_net
        self.forward_model = MLP(feature_dim + action_dim, output_dim=feature_dim, hidden_sizes=hidden_sizes, device=device)
        self.inverse_model = MLP(feature_dim * 2, output_dim=action_dim, hidden_sizes=hidden_sizes, device=device)
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.device = device

    def forward(self, s1: Union[np.ndarray, torch.Tensor], act: Union[np.ndarray, torch.Tensor], s2: Union[np.ndarray, torch.Tensor], **kwargs: Any) ->Tuple[torch.Tensor, torch.Tensor]:
        """Mapping: s1, act, s2 -> mse_loss, act_hat."""
        s1 = to_torch(s1, dtype=torch.float32, device=self.device)
        s2 = to_torch(s2, dtype=torch.float32, device=self.device)
        phi1, phi2 = self.feature_net(s1), self.feature_net(s2)
        act = to_torch(act, dtype=torch.long, device=self.device)
        phi2_hat = self.forward_model(torch.cat([phi1, F.one_hot(act, num_classes=self.action_dim)], dim=1))
        mse_loss = 0.5 * F.mse_loss(phi2_hat, phi2, reduction='none').sum(1)
        act_hat = self.inverse_model(torch.cat([phi1, phi2], dim=1))
        return mse_loss, act_hat


class PSRLModel(object):
    """Implementation of Posterior Sampling Reinforcement Learning Model.

    :param np.ndarray trans_count_prior: dirichlet prior (alphas), with shape
        (n_state, n_action, n_state).
    :param np.ndarray rew_mean_prior: means of the normal priors of rewards,
        with shape (n_state, n_action).
    :param np.ndarray rew_std_prior: standard deviations of the normal priors
        of rewards, with shape (n_state, n_action).
    :param float discount_factor: in [0, 1].
    :param float epsilon: for precision control in value iteration.
    :param lr_scheduler: a learning rate scheduler that adjusts the learning rate in
        optimizer in each policy.update(). Default to None (no lr_scheduler).
    """

    def __init__(self, trans_count_prior: np.ndarray, rew_mean_prior: np.ndarray, rew_std_prior: np.ndarray, discount_factor: float, epsilon: float) ->None:
        self.trans_count = trans_count_prior
        self.n_state, self.n_action = rew_mean_prior.shape
        self.rew_mean = rew_mean_prior
        self.rew_std = rew_std_prior
        self.rew_square_sum = np.zeros_like(rew_mean_prior)
        self.rew_std_prior = rew_std_prior
        self.discount_factor = discount_factor
        self.rew_count = np.full(rew_mean_prior.shape, epsilon)
        self.eps = epsilon
        self.policy: np.ndarray
        self.value = np.zeros(self.n_state)
        self.updated = False
        self.__eps = np.finfo(np.float32).eps.item()

    def observe(self, trans_count: np.ndarray, rew_sum: np.ndarray, rew_square_sum: np.ndarray, rew_count: np.ndarray) ->None:
        """Add data into memory pool.

        For rewards, we have a normal prior at first. After we observed a
        reward for a given state-action pair, we use the mean value of our
        observations instead of the prior mean as the posterior mean. The
        standard deviations are in inverse proportion to the number of the
        corresponding observations.

        :param np.ndarray trans_count: the number of observations, with shape
            (n_state, n_action, n_state).
        :param np.ndarray rew_sum: total rewards, with shape
            (n_state, n_action).
        :param np.ndarray rew_square_sum: total rewards' squares, with shape
            (n_state, n_action).
        :param np.ndarray rew_count: the number of rewards, with shape
            (n_state, n_action).
        """
        self.updated = False
        self.trans_count += trans_count
        sum_count = self.rew_count + rew_count
        self.rew_mean = (self.rew_mean * self.rew_count + rew_sum) / sum_count
        self.rew_square_sum += rew_square_sum
        raw_std2 = self.rew_square_sum / sum_count - self.rew_mean ** 2
        self.rew_std = np.sqrt(1 / (sum_count / (raw_std2 + self.__eps) + 1 / self.rew_std_prior ** 2))
        self.rew_count = sum_count

    def sample_trans_prob(self) ->np.ndarray:
        sample_prob = torch.distributions.Dirichlet(torch.from_numpy(self.trans_count)).sample().numpy()
        return sample_prob

    def sample_reward(self) ->np.ndarray:
        return np.random.normal(self.rew_mean, self.rew_std)

    def solve_policy(self) ->None:
        self.updated = True
        self.policy, self.value = self.value_iteration(self.sample_trans_prob(), self.sample_reward(), self.discount_factor, self.eps, self.value)

    @staticmethod
    def value_iteration(trans_prob: np.ndarray, rew: np.ndarray, discount_factor: float, eps: float, value: np.ndarray) ->Tuple[np.ndarray, np.ndarray]:
        """Value iteration solver for MDPs.

        :param np.ndarray trans_prob: transition probabilities, with shape
            (n_state, n_action, n_state).
        :param np.ndarray rew: rewards, with shape (n_state, n_action).
        :param float eps: for precision control.
        :param float discount_factor: in [0, 1].
        :param np.ndarray value: the initialize value of value array, with
            shape (n_state, ).

        :return: the optimal policy with shape (n_state, ).
        """
        Q = rew + discount_factor * trans_prob.dot(value)
        new_value = Q.max(axis=1)
        while not np.allclose(new_value, value, eps):
            value = new_value
            Q = rew + discount_factor * trans_prob.dot(value)
            new_value = Q.max(axis=1)
        Q += eps * np.random.randn(*Q.shape)
        return Q.argmax(axis=1), new_value

    def __call__(self, obs: np.ndarray, state: Any=None, info: Dict[str, Any]={}) ->np.ndarray:
        if not self.updated:
            self.solve_policy()
        return self.policy[obs]


class BaseNoise(ABC, object):
    """The action noise base class."""

    def __init__(self) ->None:
        super().__init__()

    def reset(self) ->None:
        """Reset to the initial state."""
        pass

    @abstractmethod
    def __call__(self, size: Sequence[int]) ->np.ndarray:
        """Generate new noise."""
        raise NotImplementedError


class GaussianNoise(BaseNoise):
    """The vanilla Gaussian process, for exploration in DDPG by default."""

    def __init__(self, mu: float=0.0, sigma: float=1.0) ->None:
        super().__init__()
        self._mu = mu
        assert 0 <= sigma, 'Noise std should not be negative.'
        self._sigma = sigma

    def __call__(self, size: Sequence[int]) ->np.ndarray:
        return np.random.normal(self._mu, self._sigma, size)


class RunningMeanStd(object):
    """Calculates the running mean and std of a data stream.

    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

    :param mean: the initial mean estimation for data array. Default to 0.
    :param std: the initial standard error estimation for data array. Default to 1.
    :param float clip_max: the maximum absolute value for data array. Default to
        10.0.
    :param float epsilon: To avoid division by zero.
    """

    def __init__(self, mean: Union[float, np.ndarray]=0.0, std: Union[float, np.ndarray]=1.0, clip_max: Optional[float]=10.0, epsilon: float=np.finfo(np.float32).eps.item()) ->None:
        self.mean, self.var = mean, std
        self.clip_max = clip_max
        self.count = 0
        self.eps = epsilon

    def norm(self, data_array: Union[float, np.ndarray]) ->Union[float, np.ndarray]:
        data_array = (data_array - self.mean) / np.sqrt(self.var + self.eps)
        if self.clip_max:
            data_array = np.clip(data_array, -self.clip_max, self.clip_max)
        return data_array

    def update(self, data_array: np.ndarray) ->None:
        """Add a batch of item into RMS with the same shape, modify mean/var/count."""
        batch_mean, batch_var = np.mean(data_array, axis=0), np.var(data_array, axis=0)
        batch_count = len(data_array)
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta ** 2 * self.count * batch_count / total_count
        new_var = m_2 / total_count
        self.mean, self.var = new_mean, new_var
        self.count = total_count


class Recurrent(nn.Module):
    """Simple Recurrent network based on LSTM.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(self, layer_num: int, state_shape: Union[int, Sequence[int]], action_shape: Union[int, Sequence[int]], device: Union[str, int, torch.device]='cpu', hidden_layer_size: int=128) ->None:
        super().__init__()
        self.device = device
        self.nn = nn.LSTM(input_size=hidden_layer_size, hidden_size=hidden_layer_size, num_layers=layer_num, batch_first=True)
        self.fc1 = nn.Linear(int(np.prod(state_shape)), hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, int(np.prod(action_shape)))

    def forward(self, obs: Union[np.ndarray, torch.Tensor], state: Optional[Dict[str, torch.Tensor]]=None, info: Dict[str, Any]={}) ->Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Mapping: obs -> flatten -> logits.

        In the evaluation mode, `obs` should be with shape ``[bsz, dim]``; in the
        training mode, `obs` should be with shape ``[bsz, len, dim]``. See the code
        and comment for more detail.
        """
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        if len(obs.shape) == 2:
            obs = obs.unsqueeze(-2)
        obs = self.fc1(obs)
        self.nn.flatten_parameters()
        if state is None:
            obs, (hidden, cell) = self.nn(obs)
        else:
            obs, (hidden, cell) = self.nn(obs, (state['hidden'].transpose(0, 1).contiguous(), state['cell'].transpose(0, 1).contiguous()))
        obs = self.fc2(obs[:, -1])
        return obs, {'hidden': hidden.transpose(0, 1).detach(), 'cell': cell.transpose(0, 1).detach()}


class ActorCritic(nn.Module):
    """An actor-critic network for parsing parameters.

    Using ``actor_critic.parameters()`` instead of set.union or list+list to avoid
    issue #449.

    :param nn.Module actor: the actor network.
    :param nn.Module critic: the critic network.
    """

    def __init__(self, actor: nn.Module, critic: nn.Module) ->None:
        super().__init__()
        self.actor = actor
        self.critic = critic


class DataParallelNet(nn.Module):
    """DataParallel wrapper for training agent with multi-GPU.

    This class does only the conversion of input data type, from numpy array to torch's
    Tensor. If the input is a nested dictionary, the user should create a similar class
    to do the same thing.

    :param nn.Module net: the network to be distributed in different GPUs.
    """

    def __init__(self, net: nn.Module) ->None:
        super().__init__()
        self.net = nn.DataParallel(net)

    def forward(self, obs: Union[np.ndarray, torch.Tensor], *args: Any, **kwargs: Any) ->Tuple[Any, Any]:
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, dtype=torch.float32)
        return self.net(*args, obs=obs, **kwargs)


class EnsembleLinear(nn.Module):
    """Linear Layer of Ensemble network.

    :param int ensemble_size: Number of subnets in the ensemble.
    :param int inp_feature: dimension of the input vector.
    :param int out_feature: dimension of the output vector.
    :param bool bias: whether to include an additive bias, default to be True.
    """

    def __init__(self, ensemble_size: int, in_feature: int, out_feature: int, bias: bool=True) ->None:
        super().__init__()
        k = np.sqrt(1.0 / in_feature)
        weight_data = torch.rand((ensemble_size, in_feature, out_feature)) * 2 * k - k
        self.weight = nn.Parameter(weight_data, requires_grad=True)
        self.bias: Union[nn.Parameter, None]
        if bias:
            bias_data = torch.rand((ensemble_size, 1, out_feature)) * 2 * k - k
            self.bias = nn.Parameter(bias_data, requires_grad=True)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        x = torch.matmul(x, self.weight)
        if self.bias is not None:
            x = x + self.bias
        return x


class BranchingNet(nn.Module):
    """Branching dual Q network.

    Network for the BranchingDQNPolicy, it uses a common network module, a value module
    and action "branches" one for each dimension.It allows for a linear scaling
    of Q-value the output w.r.t. the number of dimensions in the action space.
    For more info please refer to: arXiv:1711.08946.
    :param state_shape: int or a sequence of int of the shape of state.
    :param action_shape: int or a sequence of int of the shape of action.
    :param action_peer_branch: int or a sequence of int of the number of actions in
    each dimension.
    :param common_hidden_sizes: shape of the common MLP network passed in as a list.
    :param value_hidden_sizes: shape of the value MLP network passed in as a list.
    :param action_hidden_sizes: shape of the action MLP network passed in as a list.
    :param norm_layer: use which normalization before activation, e.g.,
    ``nn.LayerNorm`` and ``nn.BatchNorm1d``. Default to no normalization.
    You can also pass a list of normalization modules with the same length
    of hidden_sizes, to use different normalization module in different
    layers. Default to no normalization.
    :param activation: which activation to use after each layer, can be both
    the same activation for all layers if passed in nn.Module, or different
    activation for different Modules if passed in a list. Default to
    nn.ReLU.
    :param device: specify the device when the network actually runs. Default
    to "cpu".
    :param bool softmax: whether to apply a softmax layer over the last layer's
    output.
    """

    def __init__(self, state_shape: Union[int, Sequence[int]], num_branches: int=0, action_per_branch: int=2, common_hidden_sizes: List[int]=[], value_hidden_sizes: List[int]=[], action_hidden_sizes: List[int]=[], norm_layer: Optional[ModuleType]=None, activation: Optional[ModuleType]=nn.ReLU, device: Union[str, int, torch.device]='cpu') ->None:
        super().__init__()
        self.device = device
        self.num_branches = num_branches
        self.action_per_branch = action_per_branch
        common_input_dim = int(np.prod(state_shape))
        common_output_dim = 0
        self.common = MLP(common_input_dim, common_output_dim, common_hidden_sizes, norm_layer, activation, device)
        value_input_dim = common_hidden_sizes[-1]
        value_output_dim = 1
        self.value = MLP(value_input_dim, value_output_dim, value_hidden_sizes, norm_layer, activation, device)
        action_input_dim = common_hidden_sizes[-1]
        action_output_dim = action_per_branch
        self.branches = nn.ModuleList([MLP(action_input_dim, action_output_dim, action_hidden_sizes, norm_layer, activation, device) for _ in range(self.num_branches)])

    def forward(self, obs: Union[np.ndarray, torch.Tensor], state: Any=None, info: Dict[str, Any]={}) ->Tuple[torch.Tensor, Any]:
        """Mapping: obs -> model -> logits."""
        common_out = self.common(obs)
        value_out = self.value(common_out)
        value_out = torch.unsqueeze(value_out, 1)
        action_out = []
        for b in self.branches:
            action_out.append(b(common_out))
        action_scores = torch.stack(action_out, 1)
        action_scores = action_scores - torch.mean(action_scores, 2, keepdim=True)
        logits = value_out + action_scores
        return logits, state


class Actor(nn.Module):
    """Simple actor network.

    Will create an actor operated in discrete action space with structure of
    preprocess_net ---> action_shape.

    :param preprocess_net: a self-defined preprocess_net which output a
        flattened hidden state.
    :param action_shape: a sequence of int for the shape of action.
    :param hidden_sizes: a sequence of int for constructing the MLP after
        preprocess_net. Default to empty sequence (where the MLP now contains
        only a single linear layer).
    :param bool softmax_output: whether to apply a softmax layer over the last
        layer's output.
    :param int preprocess_net_output_dim: the output dimension of
        preprocess_net.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.

    .. seealso::

        Please refer to :class:`~tianshou.utils.net.common.Net` as an instance
        of how preprocess_net is suggested to be defined.
    """

    def __init__(self, preprocess_net: nn.Module, action_shape: Sequence[int], hidden_sizes: Sequence[int]=(), softmax_output: bool=True, preprocess_net_output_dim: Optional[int]=None, device: Union[str, int, torch.device]='cpu') ->None:
        super().__init__()
        self.device = device
        self.preprocess = preprocess_net
        self.output_dim = int(np.prod(action_shape))
        input_dim = getattr(preprocess_net, 'output_dim', preprocess_net_output_dim)
        self.last = MLP(input_dim, self.output_dim, hidden_sizes, device=self.device)
        self.softmax_output = softmax_output

    def forward(self, obs: Union[np.ndarray, torch.Tensor], state: Any=None, info: Dict[str, Any]={}) ->Tuple[torch.Tensor, Any]:
        """Mapping: s -> Q(s, \\*)."""
        logits, hidden = self.preprocess(obs, state)
        logits = self.last(logits)
        if self.softmax_output:
            logits = F.softmax(logits, dim=-1)
        return logits, hidden


class Critic(nn.Module):
    """Simple critic network. Will create an actor operated in discrete     action space with structure of preprocess_net ---> 1(q value).

    :param preprocess_net: a self-defined preprocess_net which output a
        flattened hidden state.
    :param hidden_sizes: a sequence of int for constructing the MLP after
        preprocess_net. Default to empty sequence (where the MLP now contains
        only a single linear layer).
    :param int last_size: the output dimension of Critic network. Default to 1.
    :param int preprocess_net_output_dim: the output dimension of
        preprocess_net.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.

    .. seealso::

        Please refer to :class:`~tianshou.utils.net.common.Net` as an instance
        of how preprocess_net is suggested to be defined.
    """

    def __init__(self, preprocess_net: nn.Module, hidden_sizes: Sequence[int]=(), last_size: int=1, preprocess_net_output_dim: Optional[int]=None, device: Union[str, int, torch.device]='cpu') ->None:
        super().__init__()
        self.device = device
        self.preprocess = preprocess_net
        self.output_dim = last_size
        input_dim = getattr(preprocess_net, 'output_dim', preprocess_net_output_dim)
        self.last = MLP(input_dim, last_size, hidden_sizes, device=self.device)

    def forward(self, obs: Union[np.ndarray, torch.Tensor], **kwargs: Any) ->torch.Tensor:
        """Mapping: s -> V(s)."""
        logits, _ = self.preprocess(obs, state=kwargs.get('state', None))
        return self.last(logits)


SIGMA_MAX = 2


SIGMA_MIN = -20


class ActorProb(nn.Module):
    """Simple actor network (output with a Gauss distribution).

    :param preprocess_net: a self-defined preprocess_net which output a
        flattened hidden state.
    :param action_shape: a sequence of int for the shape of action.
    :param hidden_sizes: a sequence of int for constructing the MLP after
        preprocess_net. Default to empty sequence (where the MLP now contains
        only a single linear layer).
    :param float max_action: the scale for the final action logits. Default to
        1.
    :param bool unbounded: whether to apply tanh activation on final logits.
        Default to False.
    :param bool conditioned_sigma: True when sigma is calculated from the
        input, False when sigma is an independent parameter. Default to False.
    :param int preprocess_net_output_dim: the output dimension of
        preprocess_net.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.

    .. seealso::

        Please refer to :class:`~tianshou.utils.net.common.Net` as an instance
        of how preprocess_net is suggested to be defined.
    """

    def __init__(self, preprocess_net: nn.Module, action_shape: Sequence[int], hidden_sizes: Sequence[int]=(), max_action: float=1.0, device: Union[str, int, torch.device]='cpu', unbounded: bool=False, conditioned_sigma: bool=False, preprocess_net_output_dim: Optional[int]=None) ->None:
        super().__init__()
        self.preprocess = preprocess_net
        self.device = device
        self.output_dim = int(np.prod(action_shape))
        input_dim = getattr(preprocess_net, 'output_dim', preprocess_net_output_dim)
        self.mu = MLP(input_dim, self.output_dim, hidden_sizes, device=self.device)
        self._c_sigma = conditioned_sigma
        if conditioned_sigma:
            self.sigma = MLP(input_dim, self.output_dim, hidden_sizes, device=self.device)
        else:
            self.sigma_param = nn.Parameter(torch.zeros(self.output_dim, 1))
        self._max = max_action
        self._unbounded = unbounded

    def forward(self, obs: Union[np.ndarray, torch.Tensor], state: Any=None, info: Dict[str, Any]={}) ->Tuple[Tuple[torch.Tensor, torch.Tensor], Any]:
        """Mapping: obs -> logits -> (mu, sigma)."""
        logits, hidden = self.preprocess(obs, state)
        mu = self.mu(logits)
        if not self._unbounded:
            mu = self._max * torch.tanh(mu)
        if self._c_sigma:
            sigma = torch.clamp(self.sigma(logits), min=SIGMA_MIN, max=SIGMA_MAX).exp()
        else:
            shape = [1] * len(mu.shape)
            shape[1] = -1
            sigma = (self.sigma_param.view(shape) + torch.zeros_like(mu)).exp()
        return (mu, sigma), state


class RecurrentActorProb(nn.Module):
    """Recurrent version of ActorProb.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(self, layer_num: int, state_shape: Sequence[int], action_shape: Sequence[int], hidden_layer_size: int=128, max_action: float=1.0, device: Union[str, int, torch.device]='cpu', unbounded: bool=False, conditioned_sigma: bool=False) ->None:
        super().__init__()
        self.device = device
        self.nn = nn.LSTM(input_size=int(np.prod(state_shape)), hidden_size=hidden_layer_size, num_layers=layer_num, batch_first=True)
        output_dim = int(np.prod(action_shape))
        self.mu = nn.Linear(hidden_layer_size, output_dim)
        self._c_sigma = conditioned_sigma
        if conditioned_sigma:
            self.sigma = nn.Linear(hidden_layer_size, output_dim)
        else:
            self.sigma_param = nn.Parameter(torch.zeros(output_dim, 1))
        self._max = max_action
        self._unbounded = unbounded

    def forward(self, obs: Union[np.ndarray, torch.Tensor], state: Optional[Dict[str, torch.Tensor]]=None, info: Dict[str, Any]={}) ->Tuple[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
        """Almost the same as :class:`~tianshou.utils.net.common.Recurrent`."""
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        if len(obs.shape) == 2:
            obs = obs.unsqueeze(-2)
        self.nn.flatten_parameters()
        if state is None:
            obs, (hidden, cell) = self.nn(obs)
        else:
            obs, (hidden, cell) = self.nn(obs, (state['hidden'].transpose(0, 1).contiguous(), state['cell'].transpose(0, 1).contiguous()))
        logits = obs[:, -1]
        mu = self.mu(logits)
        if not self._unbounded:
            mu = self._max * torch.tanh(mu)
        if self._c_sigma:
            sigma = torch.clamp(self.sigma(logits), min=SIGMA_MIN, max=SIGMA_MAX).exp()
        else:
            shape = [1] * len(mu.shape)
            shape[1] = -1
            sigma = (self.sigma_param.view(shape) + torch.zeros_like(mu)).exp()
        return (mu, sigma), {'hidden': hidden.transpose(0, 1).detach(), 'cell': cell.transpose(0, 1).detach()}


class RecurrentCritic(nn.Module):
    """Recurrent version of Critic.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(self, layer_num: int, state_shape: Sequence[int], action_shape: Sequence[int]=[0], device: Union[str, int, torch.device]='cpu', hidden_layer_size: int=128) ->None:
        super().__init__()
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.device = device
        self.nn = nn.LSTM(input_size=int(np.prod(state_shape)), hidden_size=hidden_layer_size, num_layers=layer_num, batch_first=True)
        self.fc2 = nn.Linear(hidden_layer_size + int(np.prod(action_shape)), 1)

    def forward(self, obs: Union[np.ndarray, torch.Tensor], act: Optional[Union[np.ndarray, torch.Tensor]]=None, info: Dict[str, Any]={}) ->torch.Tensor:
        """Almost the same as :class:`~tianshou.utils.net.common.Recurrent`."""
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        assert len(obs.shape) == 3
        self.nn.flatten_parameters()
        obs, (hidden, cell) = self.nn(obs)
        obs = obs[:, -1]
        if act is not None:
            act = torch.as_tensor(act, device=self.device, dtype=torch.float32)
            obs = torch.cat([obs, act], dim=1)
        obs = self.fc2(obs)
        return obs


class Perturbation(nn.Module):
    """Implementation of perturbation network in BCQ algorithm. Given a state and     action, it can generate perturbed action.

    :param torch.nn.Module preprocess_net: a self-defined preprocess_net which output a
        flattened hidden state.
    :param float max_action: the maximum value of each dimension of action.
    :param Union[str, int, torch.device] device: which device to create this model on.
        Default to cpu.
    :param float phi: max perturbation parameter for BCQ. Default to 0.05.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.

    .. seealso::

        You can refer to `examples/offline/offline_bcq.py` to see how to use it.
    """

    def __init__(self, preprocess_net: nn.Module, max_action: float, device: Union[str, int, torch.device]='cpu', phi: float=0.05):
        super(Perturbation, self).__init__()
        self.preprocess_net = preprocess_net
        self.device = device
        self.max_action = max_action
        self.phi = phi

    def forward(self, state: torch.Tensor, action: torch.Tensor) ->torch.Tensor:
        logits = self.preprocess_net(torch.cat([state, action], -1))[0]
        noise = self.phi * self.max_action * torch.tanh(logits)
        return (noise + action).clamp(-self.max_action, self.max_action)


class CosineEmbeddingNetwork(nn.Module):
    """Cosine embedding network for IQN. Convert a scalar in [0, 1] to a list     of n-dim vectors.

    :param num_cosines: the number of cosines used for the embedding.
    :param embedding_dim: the dimension of the embedding/output.

    .. note::

        From https://github.com/ku2482/fqf-iqn-qrdqn.pytorch/blob/master
        /fqf_iqn_qrdqn/network.py .
    """

    def __init__(self, num_cosines: int, embedding_dim: int) ->None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(num_cosines, embedding_dim), nn.ReLU())
        self.num_cosines = num_cosines
        self.embedding_dim = embedding_dim

    def forward(self, taus: torch.Tensor) ->torch.Tensor:
        batch_size = taus.shape[0]
        N = taus.shape[1]
        i_pi = np.pi * torch.arange(start=1, end=self.num_cosines + 1, dtype=taus.dtype, device=taus.device).view(1, 1, self.num_cosines)
        cosines = torch.cos(taus.view(batch_size, N, 1) * i_pi).view(batch_size * N, self.num_cosines)
        tau_embeddings = self.net(cosines).view(batch_size, N, self.embedding_dim)
        return tau_embeddings


class ImplicitQuantileNetwork(Critic):
    """Implicit Quantile Network.

    :param preprocess_net: a self-defined preprocess_net which output a
        flattened hidden state.
    :param int action_shape: a sequence of int for the shape of action.
    :param hidden_sizes: a sequence of int for constructing the MLP after
        preprocess_net. Default to empty sequence (where the MLP now contains
        only a single linear layer).
    :param int num_cosines: the number of cosines to use for cosine embedding.
        Default to 64.
    :param int preprocess_net_output_dim: the output dimension of
        preprocess_net.

    .. note::

        Although this class inherits Critic, it is actually a quantile Q-Network
        with output shape (batch_size, action_dim, sample_size).

        The second item of the first return value is tau vector.
    """

    def __init__(self, preprocess_net: nn.Module, action_shape: Sequence[int], hidden_sizes: Sequence[int]=(), num_cosines: int=64, preprocess_net_output_dim: Optional[int]=None, device: Union[str, int, torch.device]='cpu') ->None:
        last_size = int(np.prod(action_shape))
        super().__init__(preprocess_net, hidden_sizes, last_size, preprocess_net_output_dim, device)
        self.input_dim = getattr(preprocess_net, 'output_dim', preprocess_net_output_dim)
        self.embed_model = CosineEmbeddingNetwork(num_cosines, self.input_dim)

    def forward(self, obs: Union[np.ndarray, torch.Tensor], sample_size: int, **kwargs: Any) ->Tuple[Any, torch.Tensor]:
        """Mapping: s -> Q(s, \\*)."""
        logits, hidden = self.preprocess(obs, state=kwargs.get('state', None))
        batch_size = logits.size(0)
        taus = torch.rand(batch_size, sample_size, dtype=logits.dtype, device=logits.device)
        embedding = (logits.unsqueeze(1) * self.embed_model(taus)).view(batch_size * sample_size, -1)
        out = self.last(embedding).view(batch_size, sample_size, -1).transpose(1, 2)
        return (out, taus), hidden


class FractionProposalNetwork(nn.Module):
    """Fraction proposal network for FQF.

    :param num_fractions: the number of factions to propose.
    :param embedding_dim: the dimension of the embedding/input.

    .. note::

        Adapted from https://github.com/ku2482/fqf-iqn-qrdqn.pytorch/blob/master
        /fqf_iqn_qrdqn/network.py .
    """

    def __init__(self, num_fractions: int, embedding_dim: int) ->None:
        super().__init__()
        self.net = nn.Linear(embedding_dim, num_fractions)
        torch.nn.init.xavier_uniform_(self.net.weight, gain=0.01)
        torch.nn.init.constant_(self.net.bias, 0)
        self.num_fractions = num_fractions
        self.embedding_dim = embedding_dim

    def forward(self, obs_embeddings: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = torch.distributions.Categorical(logits=self.net(obs_embeddings))
        taus_1_N = torch.cumsum(dist.probs, dim=1)
        taus = F.pad(taus_1_N, (1, 0))
        tau_hats = (taus[:, :-1] + taus[:, 1:]).detach() / 2.0
        entropies = dist.entropy()
        return taus, tau_hats, entropies


class FullQuantileFunction(ImplicitQuantileNetwork):
    """Full(y parameterized) Quantile Function.

    :param preprocess_net: a self-defined preprocess_net which output a
        flattened hidden state.
    :param int action_shape: a sequence of int for the shape of action.
    :param hidden_sizes: a sequence of int for constructing the MLP after
        preprocess_net. Default to empty sequence (where the MLP now contains
        only a single linear layer).
    :param int num_cosines: the number of cosines to use for cosine embedding.
        Default to 64.
    :param int preprocess_net_output_dim: the output dimension of
        preprocess_net.

    .. note::

        The first return value is a tuple of (quantiles, fractions, quantiles_tau),
        where fractions is a Batch(taus, tau_hats, entropies).
    """

    def __init__(self, preprocess_net: nn.Module, action_shape: Sequence[int], hidden_sizes: Sequence[int]=(), num_cosines: int=64, preprocess_net_output_dim: Optional[int]=None, device: Union[str, int, torch.device]='cpu') ->None:
        super().__init__(preprocess_net, action_shape, hidden_sizes, num_cosines, preprocess_net_output_dim, device)

    def _compute_quantiles(self, obs: torch.Tensor, taus: torch.Tensor) ->torch.Tensor:
        batch_size, sample_size = taus.shape
        embedding = (obs.unsqueeze(1) * self.embed_model(taus)).view(batch_size * sample_size, -1)
        quantiles = self.last(embedding).view(batch_size, sample_size, -1).transpose(1, 2)
        return quantiles

    def forward(self, obs: Union[np.ndarray, torch.Tensor], propose_model: FractionProposalNetwork, fractions: Optional[Batch]=None, **kwargs: Any) ->Tuple[Any, torch.Tensor]:
        """Mapping: s -> Q(s, \\*)."""
        logits, hidden = self.preprocess(obs, state=kwargs.get('state', None))
        if fractions is None:
            taus, tau_hats, entropies = propose_model(logits.detach())
            fractions = Batch(taus=taus, tau_hats=tau_hats, entropies=entropies)
        else:
            taus, tau_hats = fractions.taus, fractions.tau_hats
        quantiles = self._compute_quantiles(logits, tau_hats)
        quantiles_tau = None
        if self.training:
            with torch.no_grad():
                quantiles_tau = self._compute_quantiles(logits, taus[:, 1:-1])
        return (quantiles, fractions, quantiles_tau), hidden


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (CosineEmbeddingNetwork,
     lambda: ([], {'num_cosines': 4, 'embedding_dim': 4}),
     lambda: ([torch.rand([4, 4, 1])], {}),
     True),
    (EnsembleLinear,
     lambda: ([], {'ensemble_size': 4, 'in_feature': 4, 'out_feature': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FractionProposalNetwork,
     lambda: ([], {'num_fractions': 4, 'embedding_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MLP,
     lambda: ([], {'input_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Net,
     lambda: ([], {'state_shape': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (NoisyLinear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Recurrent,
     lambda: ([], {'layer_num': 1, 'state_shape': 4, 'action_shape': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (RecurrentActorProb,
     lambda: ([], {'layer_num': 1, 'state_shape': 4, 'action_shape': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (RecurrentCritic,
     lambda: ([], {'layer_num': 1, 'state_shape': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
]

class Test_thu_ml_tianshou(_paritybench_base):
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

