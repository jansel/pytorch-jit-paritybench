import sys
_module = sys.modules[__name__]
del sys
sem_exp = _module
semantic_prediction = _module
visualization = _module
algo = _module
ppo = _module
arguments = _module
constants = _module
envs = _module
habitat = _module
objectgoal_env = _module
vector_env = _module
depth_utils = _module
fmm_planner = _module
map_builder = _module
pose = _module
rotation_utils = _module
main = _module
model = _module
test = _module
distributions = _module
model = _module
optimization = _module
storage = _module

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


import torch


import numpy as np


import torch.nn as nn


import torch.optim as optim


from queue import Queue


from typing import Any


from typing import Callable


from typing import Dict


from typing import List


from typing import Optional


from typing import Sequence


from typing import Set


from typing import Tuple


from typing import Union


import itertools


from collections import deque


from collections import defaultdict


import logging


from torch.nn import functional as F


from torch import nn


import inspect


import re


from torch import optim


from collections import namedtuple


from torch.utils.data.sampler import BatchSampler


from torch.utils.data.sampler import SubsetRandomSampler


FixedCategorical = torch.distributions.Categorical


class Categorical(nn.Module):

    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)

    def forward(self, x):
        x = self.linear(x)
        return FixedCategorical(logits=x)


class AddBias(nn.Module):

    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)
        return x + bias


FixedNormal = torch.distributions.Normal


class DiagGaussian(nn.Module):

    def __init__(self, num_inputs, num_outputs):
        super(DiagGaussian, self).__init__()
        self.fc_mean = nn.Linear(num_inputs, num_outputs)
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = self.fc_mean(x)
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros
        action_logstd = self.logstd(zeros)
        return FixedNormal(action_mean, action_logstd.exp())


class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)


class NNBase(nn.Module):

    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()
        self._hidden_size = hidden_size
        self._recurrent = recurrent
        if recurrent:
            self.gru = nn.GRUCell(recurrent_input_size, hidden_size)
            nn.init.orthogonal_(self.gru.weight_ih.data)
            nn.init.orthogonal_(self.gru.weight_hh.data)
            self.gru.bias_ih.data.fill_(0)
            self.gru.bias_hh.data.fill_(0)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def rec_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x = hxs = self.gru(x, hxs * masks[:, None])
        else:
            N = hxs.size(0)
            T = int(x.size(0) / N)
            x = x.view(T, N, x.size(1))
            masks = masks.view(T, N, 1)
            outputs = []
            for i in range(T):
                hx = hxs = self.gru(x[i], hxs * masks[i])
                outputs.append(hx)
            x = torch.stack(outputs, dim=0)
            x = x.view(T * N, -1)
        return x, hxs


class Goal_Oriented_Semantic_Policy(NNBase):

    def __init__(self, input_shape, recurrent=False, hidden_size=512, num_sem_categories=16):
        super(Goal_Oriented_Semantic_Policy, self).__init__(recurrent, hidden_size, hidden_size)
        out_size = int(input_shape[1] / 16.0) * int(input_shape[2] / 16.0)
        self.main = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(num_sem_categories + 8, 32, 3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(32, 64, 3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(64, 128, 3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(128, 64, 3, stride=1, padding=1), nn.ReLU(), nn.Conv2d(64, 32, 3, stride=1, padding=1), nn.ReLU(), Flatten())
        self.linear1 = nn.Linear(out_size * 32 + 8 * 2, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 256)
        self.critic_linear = nn.Linear(256, 1)
        self.orientation_emb = nn.Embedding(72, 8)
        self.goal_emb = nn.Embedding(num_sem_categories, 8)
        self.train()

    def forward(self, inputs, rnn_hxs, masks, extras):
        x = self.main(inputs)
        orientation_emb = self.orientation_emb(extras[:, 0])
        goal_emb = self.goal_emb(extras[:, 1])
        x = torch.cat((x, orientation_emb, goal_emb), 1)
        x = nn.ReLU()(self.linear1(x))
        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)
        x = nn.ReLU()(self.linear2(x))
        return self.critic_linear(x).squeeze(-1), x, rnn_hxs


class RL_Policy(nn.Module):

    def __init__(self, obs_shape, action_space, model_type=0, base_kwargs=None):
        super(RL_Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if model_type == 1:
            self.network = Goal_Oriented_Semantic_Policy(obs_shape, **base_kwargs)
        else:
            raise NotImplementedError
        if action_space.__class__.__name__ == 'Discrete':
            num_outputs = action_space.n
            self.dist = Categorical(self.network.output_size, num_outputs)
        elif action_space.__class__.__name__ == 'Box':
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.network.output_size, num_outputs)
        else:
            raise NotImplementedError
        self.model_type = model_type

    @property
    def is_recurrent(self):
        return self.network.is_recurrent

    @property
    def rec_state_size(self):
        """Size of rnn_hx."""
        return self.network.rec_state_size

    def forward(self, inputs, rnn_hxs, masks, extras):
        if extras is None:
            return self.network(inputs, rnn_hxs, masks)
        else:
            return self.network(inputs, rnn_hxs, masks, extras)

    def act(self, inputs, rnn_hxs, masks, extras=None, deterministic=False):
        value, actor_features, rnn_hxs = self(inputs, rnn_hxs, masks, extras)
        dist = self.dist(actor_features)
        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()
        action_log_probs = dist.log_probs(action)
        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks, extras=None):
        value, _, _ = self(inputs, rnn_hxs, masks, extras)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action, extras=None):
        value, actor_features, rnn_hxs = self(inputs, rnn_hxs, masks, extras)
        dist = self.dist(actor_features)
        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()
        return value, action_log_probs, dist_entropy, rnn_hxs


class ChannelPool(nn.MaxPool1d):

    def forward(self, x):
        n, c, w, h = x.size()
        x = x.view(n, c, w * h).permute(0, 2, 1)
        x = x.contiguous()
        pooled = F.max_pool1d(x, c, 1)
        _, _, c = pooled.size()
        pooled = pooled.permute(0, 2, 1)
        return pooled.view(n, c, w, h)


def get_grid(pose, grid_size, device):
    """
    Input:
        `pose` FloatTensor(bs, 3)
        `grid_size` 4-tuple (bs, _, grid_h, grid_w)
        `device` torch.device (cpu or gpu)
    Output:
        `rot_grid` FloatTensor(bs, grid_h, grid_w, 2)
        `trans_grid` FloatTensor(bs, grid_h, grid_w, 2)

    """
    pose = pose.float()
    x = pose[:, 0]
    y = pose[:, 1]
    t = pose[:, 2]
    bs = x.size(0)
    t = t * np.pi / 180.0
    cos_t = t.cos()
    sin_t = t.sin()
    theta11 = torch.stack([cos_t, -sin_t, torch.zeros(cos_t.shape).float()], 1)
    theta12 = torch.stack([sin_t, cos_t, torch.zeros(cos_t.shape).float()], 1)
    theta1 = torch.stack([theta11, theta12], 1)
    theta21 = torch.stack([torch.ones(x.shape), -torch.zeros(x.shape), x], 1)
    theta22 = torch.stack([torch.zeros(x.shape), torch.ones(x.shape), y], 1)
    theta2 = torch.stack([theta21, theta22], 1)
    rot_grid = F.affine_grid(theta1, torch.Size(grid_size))
    trans_grid = F.affine_grid(theta2, torch.Size(grid_size))
    return rot_grid, trans_grid


class Semantic_Mapping(nn.Module):
    """
    Semantic_Mapping
    """

    def __init__(self, args):
        super(Semantic_Mapping, self).__init__()
        self.device = args.device
        self.screen_h = args.frame_height
        self.screen_w = args.frame_width
        self.resolution = args.map_resolution
        self.z_resolution = args.map_resolution
        self.map_size_cm = args.map_size_cm // args.global_downscaling
        self.n_channels = 3
        self.vision_range = args.vision_range
        self.dropout = 0.5
        self.fov = args.hfov
        self.du_scale = args.du_scale
        self.cat_pred_threshold = args.cat_pred_threshold
        self.exp_pred_threshold = args.exp_pred_threshold
        self.map_pred_threshold = args.map_pred_threshold
        self.num_sem_categories = args.num_sem_categories
        self.max_height = int(360 / self.z_resolution)
        self.min_height = int(-40 / self.z_resolution)
        self.agent_height = args.camera_height * 100.0
        self.shift_loc = [self.vision_range * self.resolution // 2, 0, np.pi / 2.0]
        self.camera_matrix = du.get_camera_matrix(self.screen_w, self.screen_h, self.fov)
        self.pool = ChannelPool(1)
        vr = self.vision_range
        self.init_grid = torch.zeros(args.num_processes, 1 + self.num_sem_categories, vr, vr, self.max_height - self.min_height).float()
        self.feat = torch.ones(args.num_processes, 1 + self.num_sem_categories, self.screen_h // self.du_scale * self.screen_w // self.du_scale).float()

    def forward(self, obs, pose_obs, maps_last, poses_last):
        bs, c, h, w = obs.size()
        depth = obs[:, 3, :, :]
        point_cloud_t = du.get_point_cloud_from_z_t(depth, self.camera_matrix, self.device, scale=self.du_scale)
        agent_view_t = du.transform_camera_view_t(point_cloud_t, self.agent_height, 0, self.device)
        agent_view_centered_t = du.transform_pose_t(agent_view_t, self.shift_loc, self.device)
        max_h = self.max_height
        min_h = self.min_height
        xy_resolution = self.resolution
        z_resolution = self.z_resolution
        vision_range = self.vision_range
        XYZ_cm_std = agent_view_centered_t.float()
        XYZ_cm_std[..., :2] = XYZ_cm_std[..., :2] / xy_resolution
        XYZ_cm_std[..., :2] = (XYZ_cm_std[..., :2] - vision_range // 2.0) / vision_range * 2.0
        XYZ_cm_std[..., 2] = XYZ_cm_std[..., 2] / z_resolution
        XYZ_cm_std[..., 2] = (XYZ_cm_std[..., 2] - (max_h + min_h) // 2.0) / (max_h - min_h) * 2.0
        self.feat[:, 1:, :] = nn.AvgPool2d(self.du_scale)(obs[:, 4:, :, :]).view(bs, c - 4, h // self.du_scale * w // self.du_scale)
        XYZ_cm_std = XYZ_cm_std.permute(0, 3, 1, 2)
        XYZ_cm_std = XYZ_cm_std.view(XYZ_cm_std.shape[0], XYZ_cm_std.shape[1], XYZ_cm_std.shape[2] * XYZ_cm_std.shape[3])
        voxels = du.splat_feat_nd(self.init_grid * 0.0, self.feat, XYZ_cm_std).transpose(2, 3)
        min_z = int(25 / z_resolution - min_h)
        max_z = int((self.agent_height + 1) / z_resolution - min_h)
        agent_height_proj = voxels[..., min_z:max_z].sum(4)
        all_height_proj = voxels.sum(4)
        fp_map_pred = agent_height_proj[:, 0:1, :, :]
        fp_exp_pred = all_height_proj[:, 0:1, :, :]
        fp_map_pred = fp_map_pred / self.map_pred_threshold
        fp_exp_pred = fp_exp_pred / self.exp_pred_threshold
        fp_map_pred = torch.clamp(fp_map_pred, min=0.0, max=1.0)
        fp_exp_pred = torch.clamp(fp_exp_pred, min=0.0, max=1.0)
        pose_pred = poses_last
        agent_view = torch.zeros(bs, c, self.map_size_cm // self.resolution, self.map_size_cm // self.resolution)
        x1 = self.map_size_cm // (self.resolution * 2) - self.vision_range // 2
        x2 = x1 + self.vision_range
        y1 = self.map_size_cm // (self.resolution * 2)
        y2 = y1 + self.vision_range
        agent_view[:, 0:1, y1:y2, x1:x2] = fp_map_pred
        agent_view[:, 1:2, y1:y2, x1:x2] = fp_exp_pred
        agent_view[:, 4:, y1:y2, x1:x2] = torch.clamp(agent_height_proj[:, 1:, :, :] / self.cat_pred_threshold, min=0.0, max=1.0)
        corrected_pose = pose_obs

        def get_new_pose_batch(pose, rel_pose_change):
            pose[:, 1] += rel_pose_change[:, 0] * torch.sin(pose[:, 2] / 57.29577951308232) + rel_pose_change[:, 1] * torch.cos(pose[:, 2] / 57.29577951308232)
            pose[:, 0] += rel_pose_change[:, 0] * torch.cos(pose[:, 2] / 57.29577951308232) - rel_pose_change[:, 1] * torch.sin(pose[:, 2] / 57.29577951308232)
            pose[:, 2] += rel_pose_change[:, 2] * 57.29577951308232
            pose[:, 2] = torch.fmod(pose[:, 2] - 180.0, 360.0) + 180.0
            pose[:, 2] = torch.fmod(pose[:, 2] + 180.0, 360.0) - 180.0
            return pose
        current_poses = get_new_pose_batch(poses_last, corrected_pose)
        st_pose = current_poses.clone().detach()
        st_pose[:, :2] = -(st_pose[:, :2] * 100.0 / self.resolution - self.map_size_cm // (self.resolution * 2)) / (self.map_size_cm // (self.resolution * 2))
        st_pose[:, 2] = 90.0 - st_pose[:, 2]
        rot_mat, trans_mat = get_grid(st_pose, agent_view.size(), self.device)
        rotated = F.grid_sample(agent_view, rot_mat, align_corners=True)
        translated = F.grid_sample(rotated, trans_mat, align_corners=True)
        maps2 = torch.cat((maps_last.unsqueeze(1), translated.unsqueeze(1)), 1)
        map_pred, _ = torch.max(maps2, 1)
        return fp_map_pred, map_pred, pose_pred, current_poses


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Categorical,
     lambda: ([], {'num_inputs': 4, 'num_outputs': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ChannelPool,
     lambda: ([], {'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DiagGaussian,
     lambda: ([], {'num_inputs': 4, 'num_outputs': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Flatten,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_devendrachaplot_Object_Goal_Navigation(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

