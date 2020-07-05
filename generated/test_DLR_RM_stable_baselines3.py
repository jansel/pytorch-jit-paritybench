import sys
_module = sys.modules[__name__]
del sys
conf = _module
setup = _module
stable_baselines3 = _module
a2c = _module
a2c = _module
common = _module
atari_wrappers = _module
base_class = _module
bit_flipping_env = _module
buffers = _module
callbacks = _module
cmd_util = _module
distributions = _module
env_checker = _module
evaluation = _module
identity_env = _module
logger = _module
monitor = _module
noise = _module
policies = _module
preprocessing = _module
results_plotter = _module
running_mean_std = _module
save_util = _module
type_aliases = _module
utils = _module
vec_env = _module
base_vec_env = _module
dummy_vec_env = _module
subproc_vec_env = _module
util = _module
vec_check_nan = _module
vec_frame_stack = _module
vec_normalize = _module
vec_transpose = _module
vec_video_recorder = _module
ppo = _module
policies = _module
ppo = _module
sac = _module
policies = _module
sac = _module
td3 = _module
policies = _module
td3 = _module
tests = _module
test_callbacks = _module
test_cnn = _module
test_custom_policy = _module
test_deterministic = _module
test_distributions = _module
test_envs = _module
test_identity = _module
test_logger = _module
test_monitor = _module
test_predict = _module
test_run = _module
test_save_load = _module
test_sde = _module
test_spaces = _module
test_utils = _module
test_vec_check_nan = _module
test_vec_envs = _module
test_vec_normalize = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, string, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
import numpy as np
from torch import Tensor
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch as th


import torch.nn.functional as F


from typing import Type


from typing import Union


from typing import Callable


from typing import Optional


from typing import Dict


from typing import Any


from typing import Tuple


from typing import List


import torch.nn as nn


from torch.distributions import Normal


from torch.distributions import Categorical


from torch.distributions import Bernoulli


from itertools import zip_longest


import numpy as np


from functools import partial


import time


def get_device(device: Union[th.device, str]='auto') ->th.device:
    """
    Retrieve PyTorch device.
    It checks that the requested device is available first.
    For now, it supports only cpu and cuda.
    By default, it tries to use the gpu.

    :param device: (Union[str, th.device]) One for 'auto', 'cuda', 'cpu'
    :return: (th.device)
    """
    if device == 'auto':
        device = 'cuda'
    device = th.device(device)
    if device == th.device('cuda') and not th.cuda.is_available():
        return th.device('cpu')
    return device


class MlpExtractor(nn.Module):
    """
    Constructs an MLP that receives observations as an input and outputs a latent representation for the policy and
    a value network. The ``net_arch`` parameter allows to specify the amount and size of the hidden layers and how many
    of them are shared between the policy network and the value network. It is assumed to be a list with the following
    structure:

    1. An arbitrary length (zero allowed) number of integers each specifying the number of units in a shared layer.
       If the number of ints is zero, there will be no shared layers.
    2. An optional dict, to specify the following non-shared layers for the value network and the policy network.
       It is formatted like ``dict(vf=[<value layer sizes>], pi=[<policy layer sizes>])``.
       If it is missing any of the keys (pi or vf), no non-shared layers (empty list) is assumed.

    For example to construct a network with one shared layer of size 55 followed by two non-shared layers for the value
    network of size 255 and a single non-shared layer of size 128 for the policy network, the following layers_spec
    would be used: ``[55, dict(vf=[255, 255], pi=[128])]``. A simple shared network topology with two layers of size 128
    would be specified as [128, 128].

    Adapted from Stable Baselines.

    :param feature_dim: (int) Dimension of the feature vector (can be the output of a CNN)
    :param net_arch: ([int or dict]) The specification of the policy and value networks.
        See above for details on its formatting.
    :param activation_fn: (Type[nn.Module]) The activation function to use for the networks.
    :param device: (th.device)
    """

    def __init__(self, feature_dim: int, net_arch: List[Union[int, Dict[str, List[int]]]], activation_fn: Type[nn.Module], device: Union[th.device, str]='auto'):
        super(MlpExtractor, self).__init__()
        device = get_device(device)
        shared_net, policy_net, value_net = [], [], []
        policy_only_layers = []
        value_only_layers = []
        last_layer_dim_shared = feature_dim
        for idx, layer in enumerate(net_arch):
            if isinstance(layer, int):
                layer_size = layer
                shared_net.append(nn.Linear(last_layer_dim_shared, layer_size))
                shared_net.append(activation_fn())
                last_layer_dim_shared = layer_size
            else:
                assert isinstance(layer, dict), 'Error: the net_arch list can only contain ints and dicts'
                if 'pi' in layer:
                    assert isinstance(layer['pi'], list), "Error: net_arch[-1]['pi'] must contain a list of integers."
                    policy_only_layers = layer['pi']
                if 'vf' in layer:
                    assert isinstance(layer['vf'], list), "Error: net_arch[-1]['vf'] must contain a list of integers."
                    value_only_layers = layer['vf']
                break
        last_layer_dim_pi = last_layer_dim_shared
        last_layer_dim_vf = last_layer_dim_shared
        for idx, (pi_layer_size, vf_layer_size) in enumerate(zip_longest(policy_only_layers, value_only_layers)):
            if pi_layer_size is not None:
                assert isinstance(pi_layer_size, int), "Error: net_arch[-1]['pi'] must only contain integers."
                policy_net.append(nn.Linear(last_layer_dim_pi, pi_layer_size))
                policy_net.append(activation_fn())
                last_layer_dim_pi = pi_layer_size
            if vf_layer_size is not None:
                assert isinstance(vf_layer_size, int), "Error: net_arch[-1]['vf'] must only contain integers."
                value_net.append(nn.Linear(last_layer_dim_vf, vf_layer_size))
                value_net.append(activation_fn())
                last_layer_dim_vf = vf_layer_size
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        self.shared_net = nn.Sequential(*shared_net)
        self.policy_net = nn.Sequential(*policy_net)
        self.value_net = nn.Sequential(*value_net)

    def forward(self, features: th.Tensor) ->Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        shared_latent = self.shared_net(features)
        return self.policy_net(shared_latent), self.value_net(shared_latent)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (MlpExtractor,
     lambda: ([], {'feature_dim': 4, 'net_arch': [4, 4], 'activation_fn': _mock_layer}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_DLR_RM_stable_baselines3(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

