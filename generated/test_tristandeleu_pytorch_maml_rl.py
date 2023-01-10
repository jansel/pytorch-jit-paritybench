import sys
_module = sys.modules[__name__]
del sys
maml_rl = _module
baseline = _module
envs = _module
bandit = _module
mdp = _module
mujoco = _module
ant = _module
half_cheetah = _module
navigation = _module
utils = _module
normalized_env = _module
sync_vector_env = _module
wrappers = _module
episode = _module
metalearners = _module
base = _module
maml_trpo = _module
policies = _module
categorical_mlp = _module
normal_mlp = _module
policy = _module
samplers = _module
multi_task_sampler = _module
sampler = _module
tests = _module
test_multi_task_sampler = _module
test_episode = _module
test_torch_utils = _module
helpers = _module
optimization = _module
reinforcement_learning = _module
torch_utils = _module
test = _module
train = _module

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


import torch.nn as nn


import torch.nn.functional as F


from collections import OrderedDict


import numpy as np


from torch.nn.utils.convert_parameters import parameters_to_vector


from torch.distributions.kl import kl_divergence


from torch.distributions import Categorical


import math


from torch.distributions import Independent


from torch.distributions import Normal


import torch.multiprocessing as mp


import time


from copy import deepcopy


from functools import reduce


from torch.nn.utils.convert_parameters import _check_param_device


class LinearFeatureBaseline(nn.Module):
    """Linear baseline based on handcrafted features, as described in [1] 
    (Supplementary Material 2).

    [1] Yan Duan, Xi Chen, Rein Houthooft, John Schulman, Pieter Abbeel, 
        "Benchmarking Deep Reinforcement Learning for Continuous Control", 2016 
        (https://arxiv.org/abs/1604.06778)
    """

    def __init__(self, input_size, reg_coeff=1e-05):
        super(LinearFeatureBaseline, self).__init__()
        self.input_size = input_size
        self._reg_coeff = reg_coeff
        self.weight = nn.Parameter(torch.Tensor(self.feature_size), requires_grad=False)
        self.weight.data.zero_()
        self._eye = torch.eye(self.feature_size, dtype=torch.float32, device=self.weight.device)

    @property
    def feature_size(self):
        return 2 * self.input_size + 4

    def _feature(self, episodes):
        ones = episodes.mask.unsqueeze(2)
        observations = episodes.observations
        time_step = torch.arange(len(episodes)).view(-1, 1, 1) * ones / 100.0
        return torch.cat([observations, observations ** 2, time_step, time_step ** 2, time_step ** 3, ones], dim=2)

    def fit(self, episodes):
        featmat = self._feature(episodes).view(-1, self.feature_size)
        returns = episodes.returns.view(-1, 1)
        flat_mask = episodes.mask.flatten()
        flat_mask_nnz = torch.nonzero(flat_mask)
        featmat = featmat[flat_mask_nnz].view(-1, self.feature_size)
        returns = returns[flat_mask_nnz].view(-1, 1)
        reg_coeff = self._reg_coeff
        XT_y = torch.matmul(featmat.t(), returns)
        XT_X = torch.matmul(featmat.t(), featmat)
        for _ in range(5):
            try:
                coeffs, _ = torch.lstsq(XT_y, XT_X + reg_coeff * self._eye)
                if torch.isnan(coeffs).any() or torch.isinf(coeffs).any():
                    raise RuntimeError
                break
            except RuntimeError:
                reg_coeff *= 10
        else:
            raise RuntimeError('Unable to solve the normal equations in `LinearFeatureBaseline`. The matrix X^T*X (with X the design matrix) is not full-rank, regardless of the regularization (maximum regularization: {0}).'.format(reg_coeff))
        self.weight.copy_(coeffs.flatten())

    def forward(self, episodes):
        features = self._feature(episodes)
        values = torch.mv(features.view(-1, self.feature_size), self.weight)
        return values.view(features.shape[:2])


class Policy(nn.Module):

    def __init__(self, input_size, output_size):
        super(Policy, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.named_meta_parameters = self.named_parameters
        self.meta_parameters = self.parameters

    def update_params(self, loss, params=None, step_size=0.5, first_order=False):
        """Apply one step of gradient descent on the loss function `loss`, with 
        step-size `step_size`, and returns the updated parameters of the neural 
        network.
        """
        if params is None:
            params = OrderedDict(self.named_meta_parameters())
        grads = torch.autograd.grad(loss, params.values(), create_graph=not first_order)
        updated_params = OrderedDict()
        for (name, param), grad in zip(params.items(), grads):
            updated_params[name] = param - step_size * grad
        return updated_params

