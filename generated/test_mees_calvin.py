import sys
_module = sys.modules[__name__]
del sys
calvin_agent = _module
datasets = _module
base_dataset = _module
disk_dataset = _module
play_data_module = _module
random = _module
shm_dataset = _module
utils = _module
episode_utils = _module
shared_memory_utils = _module
evaluation = _module
evaluate_policy = _module
evaluate_policy_singlestep = _module
multistep_sequences = _module
utils = _module
inference = _module
rollouts_interactive = _module
rollouts_training = _module
test_consecutive = _module
test_policy_interactive = _module
test_single_goal = _module
models = _module
decoders = _module
action_decoder = _module
logistic_policy_network = _module
encoders = _module
goal_encoders = _module
language_network = _module
perceptual_encoders = _module
concat_encoders = _module
proprio_encoder = _module
tactile_encoder = _module
vision_network = _module
vision_network_gripper = _module
plan_encoders = _module
plan_proposal_net = _module
plan_recognition_net = _module
play_lmp = _module
rollout = _module
rollout = _module
rollout_long_horizon = _module
rollout_video = _module
training = _module
automatic_lang_annotator_mp = _module
compute_proprioception_statistics = _module
create_splits = _module
data_visualization = _module
dataset_task_statistics = _module
kl_callbacks = _module
language_annotator = _module
relabel_with_new_lang_model = _module
transforms = _module
utils = _module
visualizations = _module
visualize_annotations = _module
visualization = _module
tsne_plot = _module
setup = _module
visualize_dataset = _module
setup_local = _module
slurm_training = _module

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


import logging


from typing import Dict


from typing import Tuple


from typing import Union


import numpy as np


import torch


from torch.utils.data import Dataset


from typing import List


from torch.utils.data import DataLoader


import torchvision


import re


from collections import Counter


from collections import defaultdict


import time


from numpy import pi


import typing


import matplotlib.pyplot as plt


from torch import nn


from typing import Optional


import torch.nn as nn


import torch.nn.functional as F


import torchvision.models as models


from torch.nn.parameter import Parameter


from torch.distributions import Independent


from torch.distributions import Normal


from torch import Tensor


import torch.distributions as D


from functools import partial


from functools import reduce


from typing import Any


import torch.distributed as dist


from itertools import chain


from typing import Set


from torchvision.transforms.functional import resize


from torch.nn import Linear


from matplotlib.animation import ArtistAnimation


import matplotlib


class ActionDecoder(nn.Module):

    def act(self, latent_plan: torch.Tensor, perceptual_emb: torch.Tensor, latent_goal: torch.Tensor) ->torch.Tensor:
        raise NotImplementedError

    def loss(self, latent_plan: torch.Tensor, perceptual_emb: torch.Tensor, latent_goal: torch.Tensor, actions: torch.Tensor) ->torch.Tensor:
        raise NotImplementedError

    def loss_and_act(self, latent_plan: torch.Tensor, perceptual_emb: torch.Tensor, latent_goal: torch.Tensor, actions: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def clear_hidden_state(self) ->None:
        raise NotImplementedError

    def _sample(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, latent_plan: torch.Tensor, perceptual_emb: torch.Tensor, latent_goal: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError


def log_sum_exp(x):
    """numerically stable log_sum_exp implementation that prevents overflow"""
    axis = len(x.size()) - 1
    m, _ = torch.max(x, dim=axis)
    m2, _ = torch.max(x, dim=axis, keepdim=True)
    return m + torch.log(torch.sum(torch.exp(x - m2), dim=axis))


logger = logging.getLogger(__name__)


class VisualGoalEncoder(nn.Module):

    def __init__(self, hidden_size: int, latent_goal_features: int, in_features: int, l2_normalize_goal_embeddings: bool, activation_function: str):
        super().__init__()
        self.l2_normalize_output = l2_normalize_goal_embeddings
        self.act_fn = getattr(nn, activation_function)()
        self.mlp = nn.Sequential(nn.Linear(in_features=in_features, out_features=hidden_size), self.act_fn, nn.Linear(in_features=hidden_size, out_features=hidden_size), self.act_fn, nn.Linear(in_features=hidden_size, out_features=latent_goal_features))

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        x = self.mlp(x)
        if self.l2_normalize_output:
            x = F.normalize(x, p=2, dim=1)
        return x


class LanguageGoalEncoder(nn.Module):

    def __init__(self, language_features: int, hidden_size: int, latent_goal_features: int, word_dropout_p: float, l2_normalize_goal_embeddings: bool, activation_function: str):
        super().__init__()
        self.l2_normalize_output = l2_normalize_goal_embeddings
        self.act_fn = getattr(nn, activation_function)()
        self.mlp = nn.Sequential(nn.Dropout(word_dropout_p), nn.Linear(in_features=language_features, out_features=hidden_size), self.act_fn, nn.Linear(in_features=hidden_size, out_features=hidden_size), self.act_fn, nn.Linear(in_features=hidden_size, out_features=latent_goal_features))

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        x = self.mlp(x)
        if self.l2_normalize_output:
            x = F.normalize(x, p=2, dim=1)
        return x


class SBert(nn.Module):

    def __init__(self, nlp_model):
        super().__init__()
        if nlp_model == 'mpnet':
            weights = 'paraphrase-mpnet-base-v2'
        elif nlp_model == 'multi':
            weights = 'paraphrase-multilingual-mpnet-base-v2'
        else:
            weights = 'paraphrase-MiniLM-L6-v2'
        self.model = SentenceTransformer(weights)

    def forward(self, x: List) ->torch.Tensor:
        emb = self.model.encode(x, convert_to_tensor=True)
        return torch.unsqueeze(emb, 1)


class IdentityEncoder(nn.Module):

    def __init__(self, proprioception_dims):
        super(IdentityEncoder, self).__init__()
        self.n_state_obs = int(np.sum(np.diff([list(x) for x in [list(y) for y in proprioception_dims.keep_indices]])))
        self.identity = nn.Identity()

    @property
    def out_features(self):
        return self.n_state_obs

    def forward(self, x):
        return self.identity(x)


class TactileEncoder(nn.Module):

    def __init__(self, visual_features: int, freeze_tactile_backbone: bool=True):
        super(TactileEncoder, self).__init__()
        net = models.resnet18(pretrained=True)
        modules = list(net.children())[:-1]
        self.net = nn.Sequential(*modules)
        if freeze_tactile_backbone:
            for param in self.net.parameters():
                param.requires_grad = False
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, visual_features)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        x_l = self.net(x[:, :3, :, :]).squeeze()
        x_r = self.net(x[:, 3:, :, :]).squeeze()
        x = torch.cat((x_l, x_r), dim=-1)
        output = F.relu(self.fc1(x))
        output = self.fc2(output)
        return output


class VisionNetwork(nn.Module):

    def __init__(self, conv_encoder: str, activation_function: str, dropout_vis_fc: float, l2_normalize_output: bool, visual_features: int, num_c: int):
        super(VisionNetwork, self).__init__()
        self.l2_normalize_output = l2_normalize_output
        self.act_fn = getattr(nn, activation_function)()
        self.conv_model = eval(conv_encoder)
        self.conv_model = self.conv_model(self.act_fn, num_c)
        self.fc1 = nn.Sequential(nn.Linear(in_features=128, out_features=512), self.act_fn, nn.Dropout(dropout_vis_fc))
        self.fc2 = nn.Linear(in_features=512, out_features=visual_features)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        x = self.conv_model(x)
        x = self.fc1(x)
        x = self.fc2(x)
        if self.l2_normalize_output:
            x = F.normalize(x, p=2, dim=1)
        return x


class SpatialSoftmax(nn.Module):

    def __init__(self, num_rows: int, num_cols: int, temperature: Optional[float]=None):
        """
        Computes the spatial softmax of a convolutional feature map.
        Read more here:
        "Learning visual feature spaces for robotic manipulation with
        deep spatial autoencoders." Finn et al., http://arxiv.org/abs/1509.06113.
        :param num_rows:  size related to original image width
        :param num_cols:  size related to original image height
        :param temperature: Softmax temperature (optional). If None, a learnable temperature is created.
        """
        super(SpatialSoftmax, self).__init__()
        self.num_rows = num_rows
        self.num_cols = num_cols
        grid_x, grid_y = torch.meshgrid(torch.linspace(-1.0, 1.0, num_cols), torch.linspace(-1.0, 1.0, num_rows), indexing='ij')
        x_map = grid_x.reshape(-1)
        y_map = grid_y.reshape(-1)
        self.register_buffer('x_map', x_map)
        self.register_buffer('y_map', y_map)
        if temperature:
            self.register_buffer('temperature', torch.ones(1) * temperature)
        else:
            self.temperature = Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        n, c, h, w = x.shape
        x = x.view(-1, h * w)
        softmax_attention = F.softmax(x / self.temperature, dim=1)
        expected_x = torch.sum(self.x_map * softmax_attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.y_map * softmax_attention, dim=1, keepdim=True)
        expected_xy = torch.cat((expected_x, expected_y), 1)
        self.coords = expected_xy.view(-1, c * 2)
        return self.coords


class PlanProposalNetwork(nn.Module):

    def __init__(self, perceptual_features: int, latent_goal_features: int, plan_features: int, activation_function: str, min_std: float):
        super(PlanProposalNetwork, self).__init__()
        self.perceptual_features = perceptual_features
        self.latent_goal_features = latent_goal_features
        self.plan_features = plan_features
        self.min_std = min_std
        self.in_features = self.perceptual_features + self.latent_goal_features
        self.act_fn = getattr(nn, activation_function)()
        self.fc_model = nn.Sequential(nn.Linear(in_features=self.in_features, out_features=2048), self.act_fn, nn.Linear(in_features=2048, out_features=2048), self.act_fn, nn.Linear(in_features=2048, out_features=2048), self.act_fn, nn.Linear(in_features=2048, out_features=2048), self.act_fn)
        self.mean_fc = nn.Linear(in_features=2048, out_features=self.plan_features)
        self.variance_fc = nn.Linear(in_features=2048, out_features=self.plan_features)

    def forward(self, initial_percep_emb: torch.Tensor, latent_goal: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([initial_percep_emb, latent_goal], dim=-1)
        x = self.fc_model(x)
        mean = self.mean_fc(x)
        var = self.variance_fc(x)
        std = F.softplus(var) + self.min_std
        return mean, std

    def __call__(self, *args, **kwargs):
        mean, std = super().__call__(*args, **kwargs)
        pp_dist = Independent(Normal(mean, std), 1)
        return pp_dist


class PlanRecognitionNetwork(nn.Module):

    def __init__(self, in_features: int, plan_features: int, action_space: int, birnn_dropout_p: float, min_std: float):
        super(PlanRecognitionNetwork, self).__init__()
        self.plan_features = plan_features
        self.action_space = action_space
        self.min_std = min_std
        self.in_features = in_features
        self.birnn_model = nn.RNN(input_size=self.in_features, hidden_size=2048, nonlinearity='relu', num_layers=2, bidirectional=True, batch_first=True, dropout=birnn_dropout_p)
        self.mean_fc = nn.Linear(in_features=4096, out_features=self.plan_features)
        self.variance_fc = nn.Linear(in_features=4096, out_features=self.plan_features)

    def forward(self, perceptual_emb: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor]:
        x, hn = self.birnn_model(perceptual_emb)
        x = x[:, -1]
        mean = self.mean_fc(x)
        var = self.variance_fc(x)
        std = F.softplus(var) + self.min_std
        return mean, std

    def __call__(self, *args, **kwargs):
        mean, std = super().__call__(*args, **kwargs)
        pr_dist = Independent(Normal(mean, std), 1)
        return pr_dist


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (PlanRecognitionNetwork,
     lambda: ([], {'in_features': 4, 'plan_features': 4, 'action_space': 4, 'birnn_dropout_p': 0.5, 'min_std': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (SpatialSoftmax,
     lambda: ([], {'num_rows': 4, 'num_cols': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_mees_calvin(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

