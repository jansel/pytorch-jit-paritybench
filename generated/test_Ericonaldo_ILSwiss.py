import sys
_module = sys.modules[__name__]
del sys
rlkit = _module
core = _module
base_algorithm = _module
eval_util = _module
logger = _module
serializable = _module
tabulate = _module
train_util = _module
trainer = _module
vistools = _module
data_management = _module
aug_replay_buffer = _module
data_augmentation = _module
env_replay_buffer = _module
episodic_replay_buffer = _module
mil_color_print = _module
mil_utils = _module
normalizer = _module
path_builder = _module
relabel_horizon_replay_buffer = _module
relabel_replay_buffer = _module
replay_buffer = _module
simple_replay_buffer = _module
envs = _module
envpool = _module
envs_dict = _module
goal_env_utils = _module
ant = _module
hopper = _module
humanoid = _module
walker2d = _module
tasks_dict = _module
terminals = _module
vecenvs = _module
worker = _module
base = _module
dummy = _module
subproc = _module
utils = _module
wrappers = _module
exploration_strategies = _module
epsilon_greedy = _module
gaussian_strategy = _module
ou_strategy = _module
launchers = _module
config = _module
launcher_util = _module
policies = _module
argmax = _module
simple = _module
samplers = _module
normal_sampler = _module
vec_sampler = _module
adv_irl = _module
adv_irl = _module
adv_irl_visual = _module
disc_models = _module
cnn_disc_models = _module
rnn_disc_models = _module
simple_disc_models = _module
bc = _module
bc = _module
dagger = _module
dagger = _module
ddpg = _module
ddpg = _module
discrete_sac = _module
discrete_sac = _module
dqn = _module
double_dqn = _module
dqn = _module
gcsl = _module
rl = _module
her = _module
her = _module
sac = _module
td3 = _module
mbpo = _module
bnn_trainer = _module
fake_env = _module
mbpo = _module
ppo = _module
ppo = _module
sac = _module
sac_ae = _module
sac_alpha = _module
td3 = _module
torch_base_algorithm = _module
torch_rl_algorithm = _module
distributions = _module
encoders = _module
modules = _module
networks = _module
policies = _module
core = _module
normalizer = _module
pytorch_util = _module
transform_layer = _module
run_experiment = _module
adv_irl_exp_script = _module
adv_irl_exp_visual_script = _module
bc_exp_script = _module
dagger_exp_script = _module
discrete_sac_exp_script = _module
evaluate_policy = _module
gcsl_exp_script = _module
gen_expert_demos = _module
her_sac_exp_script = _module
her_td3_exp_script = _module
mbpo_exp_script = _module
ppo_exp_script = _module
ppo_norm_exp_script = _module
render_algorithm = _module
sac_alpha_exp_script = _module
sac_alpha_visual_exp_script = _module
sac_exp_script = _module
td3_exp_script = _module
video = _module
normalize_exp_demos = _module
test_obs_norm = _module

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


import abc


import time


from collections import OrderedDict


import numpy as np


from enum import Enum


from torch.utils.tensorboard import SummaryWriter


import torch


import torch.nn as nn


import random


import inspect


from collections import namedtuple


from copy import deepcopy


import torch.optim as optim


from torch import nn


from torch import autograd


import torch.nn.functional as F


from torch import nn as nn


from torch.distributions import Categorical


from itertools import count


from typing import Dict


from typing import List


from typing import Tuple


from typing import Union


from typing import Type


from typing import Callable


from numpy import log as np_log


from numpy import pi


from torch.nn import functional as F


from torch.nn import BatchNorm1d


from numpy.random import choice


import math


import numbers


from torch import tanh


from random import randint


OUT_DIM = {(2): 39, (4): 35, (6): 31}


OUT_DIM_64 = {(2): 29, (4): 25, (6): 21}


class CNNDisc(nn.Module):

    def __init__(self, input_shape, input_dim=0, num_filters=32, num_layer_blocks=2, hid_dim=100, hid_act='relu', clamp_magnitude=10.0):
        super().__init__()
        if hid_act == 'relu':
            hid_act_class = nn.ReLU
        elif hid_act == 'tanh':
            hid_act_class = nn.Tanh
        else:
            raise NotImplementedError()
        self.clamp_magnitude = clamp_magnitude
        super().__init__()
        assert len(input_shape) == 3
        self.input_shape = input_shape
        self.input_dim = input_dim
        self.num_layers = num_layer_blocks
        self.convs_list = nn.ModuleList([nn.Conv2d(input_shape[0], num_filters, 3, stride=2)])
        self.convs_list.append(hid_act_class())
        for i in range(self.num_layers - 1):
            self.convs_list.append(hid_act_class())
            self.convs_list.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))
        self.conv_model = nn.Sequential(*self.convs_list)
        out_dim = OUT_DIM_64[self.num_layers] if input_shape[-1] == 64 else OUT_DIM[self.num_layers]
        self.mod_list = nn.ModuleList([nn.Linear(num_filters * out_dim * out_dim + self.input_dim, hid_dim)])
        self.mod_list.append(nn.LayerNorm(hid_dim))
        self.mod_list.append(hid_act_class())
        for i in range(self.num_layers - 1):
            self.mod_list.append(nn.Linear(hid_dim, hid_dim))
            self.mod_list.append(nn.LayerNorm(hid_dim))
            self.mod_list.append(hid_act_class())
        self.mod_list.append(nn.Linear(hid_dim, 1))
        self.linear_model = nn.Sequential(*self.mod_list)

    def forward(self, obs, vec=None):
        output = self.conv_model(obs)
        output = output.view(output.size(0), -1)
        if self.input_dim != 0:
            assert vec is not None, 'act should not be none!'
            output = torch.cat([output, vec], axis=-1)
        output = self.model(output)
        output = torch.clamp(output, min=-1.0 * self.clamp_magnitude, max=self.clamp_magnitude)
        return output


class ResNetCNNDisc(nn.Module):

    def __init__(self, input_dim, num_layer_blocks=2, hid_dim=100, hid_act='relu', use_bn=True, clamp_magnitude=10.0):
        super().__init__()
        raise NotImplementedError


class RNNDisc(nn.Module):

    def __init__(self, input_dim, hid_dim=100, hid_act='relu', rnn_act='gru', num_layers=2, use_bn=True, drop_out=-1, bidirectional=True, clamp_magnitude=10.0):
        super().__init__()
        self.hid_dim = hid_dim
        self.drop_out = 0
        if drop_out > 0:
            self.drop_out = drop_out
        self.rnn_act = rnn_act
        if hid_act == 'relu':
            hid_act_class = nn.ReLU
        elif hid_act == 'tanh':
            hid_act_class = nn.Tanh
        else:
            raise NotImplementedError()
        self.clamp_magnitude = clamp_magnitude
        self.mod_list = nn.ModuleList([nn.Linear(input_dim, hid_dim)])
        if use_bn:
            self.mod_list.append(nn.BatchNorm1d(hid_dim))
        self.mod_list.append(hid_act_class())
        self.before_linear_model = nn.Sequential(*self.mod_list)
        self.rnn = None
        if self.rnn_act == 'gru':
            self.rnn = nn.GRU(input_size=hid_dim, hidden_size=hid_dim, num_layers=num_layers, batch_first=True, bias=True, dropout=self.drop_out, bidirectional=bidirectional)
        elif self.rnn_act == 'lstm':
            self.rnn = nn.LSTM(input_size=hid_dim, hidden_size=hid_dim, num_layers=num_layers, batch_first=True, bias=True, dropout=self.drop_out, bidirectional=bidirectional)
        else:
            raise NotImplementedError
        if bidirectional:
            hid_dim *= 2
        self.after_linear_model = nn.Sequential(nn.Linear(hid_dim, 1))

    def forward(self, batch):
        output = self.before_linear_model(batch)
        output = output.permute(1, 0, 2)
        output, final_state = self.rnn(output)
        output = self.after_linear_model(output)
        output = torch.clamp(output, min=-1.0 * self.clamp_magnitude, max=self.clamp_magnitude)
        output = output.permute(1, 0, 2)
        return output


class MLPDisc(nn.Module):

    def __init__(self, input_dim, num_layer_blocks=2, hid_dim=100, hid_act='relu', use_bn=True, clamp_magnitude=10.0):
        super().__init__()
        if hid_act == 'relu':
            hid_act_class = nn.ReLU
        elif hid_act == 'tanh':
            hid_act_class = nn.Tanh
        else:
            raise NotImplementedError()
        self.clamp_magnitude = clamp_magnitude
        self.mod_list = nn.ModuleList([nn.Linear(input_dim, hid_dim)])
        if use_bn:
            self.mod_list.append(nn.BatchNorm1d(hid_dim))
        self.mod_list.append(hid_act_class())
        for i in range(num_layer_blocks - 1):
            self.mod_list.append(nn.Linear(hid_dim, hid_dim))
            if use_bn:
                self.mod_list.append(nn.BatchNorm1d(hid_dim))
            self.mod_list.append(hid_act_class())
        self.mod_list.append(nn.Linear(hid_dim, 1))
        self.model = nn.Sequential(*self.mod_list)

    def forward(self, batch):
        output = self.model(batch)
        output = torch.clamp(output, min=-1.0 * self.clamp_magnitude, max=self.clamp_magnitude)
        return output


class ResNetAIRLDisc(nn.Module):

    def __init__(self, input_dim, num_layer_blocks=2, hid_dim=100, hid_act='relu', use_bn=True, clamp_magnitude=10.0):
        super().__init__()
        if hid_act == 'relu':
            hid_act_class = nn.ReLU
        elif hid_act == 'tanh':
            hid_act_class = nn.Tanh
        else:
            raise NotImplementedError()
        self.clamp_magnitude = clamp_magnitude
        self.first_fc = nn.Linear(input_dim, hid_dim)
        self.blocks_list = nn.ModuleList()
        for i in range(num_layer_blocks - 1):
            block = nn.ModuleList()
            block.append(nn.Linear(hid_dim, hid_dim))
            if use_bn:
                block.append(nn.BatchNorm1d(hid_dim))
            block.append(hid_act_class())
            self.blocks_list.append(nn.Sequential(*block))
        self.last_fc = nn.Linear(hid_dim, 1)

    def forward(self, batch):
        x = self.first_fc(batch)
        for block in self.blocks_list:
            x = x + block(x)
        output = self.last_fc(x)
        output = torch.clamp(output, min=-1.0 * self.clamp_magnitude, max=self.clamp_magnitude)
        return output


class HuberLoss(nn.Module):

    def __init__(self, delta=1):
        super().__init__()
        self.huber_loss_delta1 = nn.SmoothL1Loss()
        self.delta = delta

    def forward(self, x, x_hat):
        loss = self.huber_loss_delta1(x / self.delta, x_hat / self.delta)
        return loss * self.delta * self.delta


class LayerNorm(nn.Module):
    """
    Simple 1D LayerNorm.
    """

    def __init__(self, features, center=True, scale=False, eps=1e-06):
        super().__init__()
        self.center = center
        self.scale = scale
        self.eps = eps
        if self.scale:
            self.scale_param = nn.Parameter(torch.ones(features))
        else:
            self.scale_param = None
        if self.center:
            self.center_param = nn.Parameter(torch.zeros(features))
        else:
            self.center_param = None

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        output = (x - mean) / (std + self.eps)
        if self.scale:
            output = output * self.scale_param
        if self.center:
            output = output + self.center_param
        return output


class Serializable(object):

    def __init__(self, *args, **kwargs):
        self.__args = args
        self.__kwargs = kwargs

    def quick_init(self, locals_):
        if getattr(self, '_serializable_initialized', False):
            return
        if sys.version_info >= (3, 0):
            spec = inspect.getfullargspec(self.__init__)
            if spec.varkw:
                kwargs = locals_[spec.varkw].copy()
            else:
                kwargs = dict()
            if spec.kwonlyargs:
                for key in spec.kwonlyargs:
                    kwargs[key] = locals_[key]
        else:
            spec = inspect.getargspec(self.__init__)
            if spec.keywords:
                kwargs = locals_[spec.keywords]
            else:
                kwargs = dict()
        if spec.varargs:
            varargs = locals_[spec.varargs]
        else:
            varargs = tuple()
        in_order_args = [locals_[arg] for arg in spec.args][1:]
        self.__args = tuple(in_order_args) + varargs
        self.__kwargs = kwargs
        setattr(self, '_serializable_initialized', True)

    def __getstate__(self):
        return {'__args': self.__args, '__kwargs': self.__kwargs}

    def __setstate__(self, d):
        if sys.version_info >= (3, 0):
            spec = inspect.getfullargspec(self.__init__)
        else:
            spec = inspect.getargspec(self.__init__)
        in_order_args = spec.args[1:]
        out = type(self)(**dict(zip(in_order_args, d['__args']), **d['__kwargs']))
        self.__dict__.update(out.__dict__)

    @classmethod
    def clone(cls, obj, **kwargs):
        assert isinstance(obj, Serializable)
        d = obj.__getstate__()
        d['__kwargs'] = dict(d['__kwargs'], **kwargs)
        out = type(obj).__new__(type(obj))
        out.__setstate__(d)
        return out


def np_ify(tensor_or_other):
    if isinstance(tensor_or_other, torch.Tensor):
        return ptu.get_numpy(tensor_or_other)
    else:
        return tensor_or_other


def torch_ify(np_array_or_other):
    if isinstance(np_array_or_other, np.ndarray):
        return ptu.from_numpy(np_array_or_other)
    else:
        return np_array_or_other


class PyTorchModule(nn.Module, Serializable, metaclass=abc.ABCMeta):

    def get_param_values(self):
        return self.state_dict()

    def set_param_values(self, param_values):
        self.load_state_dict(param_values)

    def get_param_values_np(self):
        state_dict = self.state_dict()
        np_dict = OrderedDict()
        for key, tensor in state_dict.items():
            np_dict[key] = ptu.get_numpy(tensor)
        return np_dict

    def set_param_values_np(self, param_values):
        torch_dict = OrderedDict()
        for key, tensor in param_values.items():
            torch_dict[key] = ptu.from_numpy(tensor)
        self.load_state_dict(torch_dict)

    def copy(self):
        copy = Serializable.clone(self)
        ptu.copy_model_params_from_to(self, copy)
        return copy

    def save_init_params(self, locals):
        """
        Should call this FIRST THING in the __init__ method if you ever want
        to serialize or clone this network.

        Usage:
        ```
        def __init__(self, ...):
            self.init_serialization(locals())
            ...
        ```
        :param locals:
        :return:
        """
        Serializable.quick_init(self, locals)

    def __getstate__(self):
        d = Serializable.__getstate__(self)
        d['params'] = self.get_param_values()
        return d

    def __setstate__(self, d):
        Serializable.__setstate__(self, d)
        self.set_param_values(d['params'])

    def regularizable_parameters(self):
        """
        Return generator of regularizable parameters. Right now, all non-flat
        vectors are assumed to be regularizabled, presumably because only
        biases are flat.

        :return:
        """
        for param in self.parameters():
            if len(param.size()) > 1:
                yield param

    def eval_np(self, *args, **kwargs):
        """
        Eval this module with a numpy interface

        Same as a call to __call__ except all Variable input/outputs are
        replaced with numpy equivalents.

        Assumes the output is either a single object or a tuple of objects.
        """
        torch_args = tuple(torch_ify(x) for x in args)
        torch_kwargs = {k: torch_ify(v) for k, v in kwargs.items()}
        outputs = self.__call__(*torch_args, **torch_kwargs)
        if isinstance(outputs, tuple):
            return tuple(np_ify(x) for x in outputs)
        else:
            return np_ify(outputs)


def hsv2rgb(hsv):
    _device = hsv.device
    hsv = torch.clamp(hsv, 0, 1)
    hue = hsv[:, 0, :, :] * 360.0
    saturation = hsv[:, 1, :, :]
    value = hsv[:, 2, :, :]
    c = value * saturation
    x = -c * (torch.abs(hue / 60.0 % 2 - 1) - 1)
    m = (value - c).unsqueeze(dim=1)
    rgb_prime = torch.zeros_like(hsv)
    inds = (hue < 60) * (hue >= 0)
    rgb_prime[:, 0, :, :][inds] = c[inds]
    rgb_prime[:, 1, :, :][inds] = x[inds]
    inds = (hue < 120) * (hue >= 60)
    rgb_prime[:, 0, :, :][inds] = x[inds]
    rgb_prime[:, 1, :, :][inds] = c[inds]
    inds = (hue < 180) * (hue >= 120)
    rgb_prime[:, 1, :, :][inds] = c[inds]
    rgb_prime[:, 2, :, :][inds] = x[inds]
    inds = (hue < 240) * (hue >= 180)
    rgb_prime[:, 1, :, :][inds] = x[inds]
    rgb_prime[:, 2, :, :][inds] = c[inds]
    inds = (hue < 300) * (hue >= 240)
    rgb_prime[:, 2, :, :][inds] = c[inds]
    rgb_prime[:, 0, :, :][inds] = x[inds]
    inds = (hue < 360) * (hue >= 300)
    rgb_prime[:, 2, :, :][inds] = x[inds]
    rgb_prime[:, 0, :, :][inds] = c[inds]
    rgb = rgb_prime + torch.cat((m, m, m), dim=1)
    rgb = rgb
    return torch.clamp(rgb, 0, 1)


def rgb2hsv(rgb, eps=1e-08):
    _device = rgb.device
    r, g, b = rgb[:, 0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]
    Cmax = rgb.max(1)[0]
    Cmin = rgb.min(1)[0]
    delta = Cmax - Cmin
    hue = torch.zeros((rgb.shape[0], rgb.shape[2], rgb.shape[3]))
    hue[Cmax == r] = ((g - b) / (delta + eps) % 6)[Cmax == r]
    hue[Cmax == g] = ((b - r) / (delta + eps) + 2)[Cmax == g]
    hue[Cmax == b] = ((r - g) / (delta + eps) + 4)[Cmax == b]
    hue[Cmax == 0] = 0.0
    hue = hue / 6.0
    hue = hue.unsqueeze(dim=1)
    saturation = delta / (Cmax + eps)
    saturation[Cmax == 0.0] = 0.0
    saturation = saturation
    saturation = saturation.unsqueeze(dim=1)
    value = Cmax
    value = value
    value = value.unsqueeze(dim=1)
    return torch.cat((hue, saturation, value), dim=1)


class ColorJitterLayer(nn.Module):

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, p=0, batch_size=128, stack_size=3):
        super(ColorJitterLayer, self).__init__()
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5), clip_first_on_zero=False)
        self.prob = p
        self.batch_size = batch_size
        self.stack_size = stack_size

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError('If {} is a single number, it must be non negative.'.format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError('{} values should be between {}'.format(name, bound))
        else:
            raise TypeError('{} should be a single number or a list/tuple with lenght 2.'.format(name))
        if value[0] == value[1] == center:
            value = None
        return value

    def adjust_contrast(self, x):
        """
        Args:
            x: torch tensor img (rgb type)
        Factor: torch tensor with same length as x
                0 gives gray solid image, 1 gives original image,
        Returns:
            torch tensor image: Brightness adjusted
        """
        _device = x.device
        factor = torch.empty(self.batch_size, device=_device).uniform_(*self.contrast)
        factor = factor.reshape(-1, 1).repeat(1, self.stack_size).reshape(-1)
        means = torch.mean(x, dim=(2, 3), keepdim=True)
        return torch.clamp((x - means) * factor.view(len(x), 1, 1, 1) + means, 0, 1)

    def adjust_hue(self, x):
        _device = x.device
        factor = torch.empty(self.batch_size, device=_device).uniform_(*self.hue)
        factor = factor.reshape(-1, 1).repeat(1, self.stack_size).reshape(-1)
        h = x[:, 0, :, :]
        h += factor.view(len(x), 1, 1) * 255.0 / 360.0
        h = h % 1
        x[:, 0, :, :] = h
        return x

    def adjust_brightness(self, x):
        """
        Args:
            x: torch tensor img (hsv type)
        Factor:
            torch tensor with same length as x
            0 gives black image, 1 gives original image,
            2 gives the brightness factor of 2.
        Returns:
            torch tensor image: Brightness adjusted
        """
        _device = x.device
        factor = torch.empty(self.batch_size, device=_device).uniform_(*self.brightness)
        factor = factor.reshape(-1, 1).repeat(1, self.stack_size).reshape(-1)
        x[:, 2, :, :] = torch.clamp(x[:, 2, :, :] * factor.view(len(x), 1, 1), 0, 1)
        return torch.clamp(x, 0, 1)

    def adjust_saturate(self, x):
        """
        Args:
            x: torch tensor img (hsv type)
        Factor:
            torch tensor with same length as x
            0 gives black image and white, 1 gives original image,
            2 gives the brightness factor of 2.
        Returns:
            torch tensor image: Brightness adjusted
        """
        _device = x.device
        factor = torch.empty(self.batch_size, device=_device).uniform_(*self.saturation)
        factor = factor.reshape(-1, 1).repeat(1, self.stack_size).reshape(-1)
        x[:, 1, :, :] = torch.clamp(x[:, 1, :, :] * factor.view(len(x), 1, 1), 0, 1)
        return torch.clamp(x, 0, 1)

    def transform(self, inputs):
        hsv_transform_list = [rgb2hsv, self.adjust_brightness, self.adjust_hue, self.adjust_saturate, hsv2rgb]
        rgb_transform_list = [self.adjust_contrast]
        if random.uniform(0, 1) >= 0.5:
            transform_list = rgb_transform_list + hsv_transform_list
        else:
            transform_list = hsv_transform_list + rgb_transform_list
        for t in transform_list:
            inputs = t(inputs)
        return inputs

    def forward(self, inputs):
        _device = inputs.device
        random_inds = np.random.choice([True, False], len(inputs), p=[self.prob, 1 - self.prob])
        inds = torch.tensor(random_inds)
        if random_inds.sum() > 0:
            inputs[inds] = self.transform(inputs[inds])
        return inputs


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ColorJitterLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (HuberLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (LayerNorm,
     lambda: ([], {'features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MLPDisc,
     lambda: ([], {'input_dim': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (ResNetAIRLDisc,
     lambda: ([], {'input_dim': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
]

class Test_Ericonaldo_ILSwiss(_paritybench_base):
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

