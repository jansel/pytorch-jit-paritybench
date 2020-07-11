import sys
_module = sys.modules[__name__]
del sys
algo = _module
ppo = _module
arguments = _module
env = _module
habitat = _module
exploration_env = _module
utils = _module
noisy_actions = _module
pose = _module
supervision = _module
visualizations = _module
depth_utils = _module
fmm_planner = _module
map_builder = _module
rotation_utils = _module
main = _module
model = _module
convert_datasets = _module
convert_val_mini = _module
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


import torch


import torch.nn as nn


import torch.nn.functional as F


import torch.optim as optim


import math


import numpy as np


import matplotlib


from torch.nn import functional as F


from torchvision import transforms


import matplotlib.pyplot as plt


import time


from collections import deque


import logging


import torchvision.models as models


from torch import nn


import inspect


import re


from torch import optim


from collections import namedtuple


from torch.utils.data.sampler import BatchSampler


from torch.utils.data.sampler import SubsetRandomSampler


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
    x = pose[:, (0)]
    y = pose[:, (1)]
    t = pose[:, (2)]
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


class Neural_SLAM_Module(nn.Module):
    """
    """

    def __init__(self, args):
        super(Neural_SLAM_Module, self).__init__()
        self.device = args.device
        self.screen_h = args.frame_height
        self.screen_w = args.frame_width
        self.resolution = args.map_resolution
        self.map_size_cm = args.map_size_cm // args.global_downscaling
        self.n_channels = 3
        self.vision_range = args.vision_range
        self.dropout = 0.5
        self.use_pe = args.use_pose_estimation
        resnet = models.resnet18(pretrained=args.pretrained_resnet)
        self.resnet_l5 = nn.Sequential(*list(resnet.children())[0:8])
        self.conv = nn.Sequential(*filter(bool, [nn.Conv2d(512, 64, (1, 1), stride=(1, 1)), nn.ReLU()]))
        input_test = torch.randn(1, self.n_channels, self.screen_h, self.screen_w)
        conv_output = self.conv(self.resnet_l5(input_test))
        self.pool = ChannelPool(1)
        self.conv_output_size = conv_output.view(-1).size(0)
        self.proj1 = nn.Linear(self.conv_output_size, 1024)
        self.proj2 = nn.Linear(1024, 4096)
        if self.dropout > 0:
            self.dropout1 = nn.Dropout(self.dropout)
            self.dropout2 = nn.Dropout(self.dropout)
        self.deconv = nn.Sequential(*filter(bool, [nn.ConvTranspose2d(64, 32, (4, 4), stride=(2, 2), padding=(1, 1)), nn.ReLU(), nn.ConvTranspose2d(32, 16, (4, 4), stride=(2, 2), padding=(1, 1)), nn.ReLU(), nn.ConvTranspose2d(16, 2, (4, 4), stride=(2, 2), padding=(1, 1))]))
        self.pose_conv = nn.Sequential(*filter(bool, [nn.Conv2d(4, 64, (4, 4), stride=(2, 2)), nn.ReLU(), nn.Conv2d(64, 32, (4, 4), stride=(2, 2)), nn.ReLU(), nn.Conv2d(32, 16, (3, 3), stride=(1, 1)), nn.ReLU()]))
        pose_conv_output = self.pose_conv(torch.randn(1, 4, self.vision_range, self.vision_range))
        self.pose_conv_output_size = pose_conv_output.view(-1).size(0)
        self.pose_proj1 = nn.Linear(self.pose_conv_output_size, 1024)
        self.pose_proj2_x = nn.Linear(1024, 128)
        self.pose_proj2_y = nn.Linear(1024, 128)
        self.pose_proj2_o = nn.Linear(1024, 128)
        self.pose_proj3_x = nn.Linear(128, 1)
        self.pose_proj3_y = nn.Linear(128, 1)
        self.pose_proj3_o = nn.Linear(128, 1)
        if self.dropout > 0:
            self.pose_dropout1 = nn.Dropout(self.dropout)
        self.st_poses_eval = torch.zeros(args.num_processes, 3)
        self.st_poses_train = torch.zeros(args.slam_batch_size, 3)
        grid_size = self.vision_range * 2
        self.grid_map_eval = torch.zeros(args.num_processes, 2, grid_size, grid_size).float()
        self.grid_map_train = torch.zeros(args.slam_batch_size, 2, grid_size, grid_size).float()
        self.agent_view = torch.zeros(args.num_processes, 2, self.map_size_cm // self.resolution, self.map_size_cm // self.resolution).float()

    def forward(self, obs_last, obs, poses, maps, explored, current_poses, build_maps=True):
        bs, c, h, w = obs.size()
        resnet_output = self.resnet_l5(obs[:, :3, :, :])
        conv_output = self.conv(resnet_output)
        proj1 = nn.ReLU()(self.proj1(conv_output.view(-1, self.conv_output_size)))
        if self.dropout > 0:
            proj1 = self.dropout1(proj1)
        proj3 = nn.ReLU()(self.proj2(proj1))
        deconv_input = proj3.view(bs, 64, 8, 8)
        deconv_output = self.deconv(deconv_input)
        pred = torch.sigmoid(deconv_output)
        proj_pred = pred[:, :1, :, :]
        fp_exp_pred = pred[:, 1:, :, :]
        with torch.no_grad():
            bs, c, h, w = obs_last.size()
            resnet_output = self.resnet_l5(obs_last[:, :3, :, :])
            conv_output = self.conv(resnet_output)
            proj1 = nn.ReLU()(self.proj1(conv_output.view(-1, self.conv_output_size)))
            if self.dropout > 0:
                proj1 = self.dropout1(proj1)
            proj3 = nn.ReLU()(self.proj2(proj1))
            deconv_input = proj3.view(bs, 64, 8, 8)
            deconv_output = self.deconv(deconv_input)
            pred_last = torch.sigmoid(deconv_output)
            vr = self.vision_range
            grid_size = vr * 2
            if build_maps:
                st_poses = self.st_poses_eval.detach_()
                grid_map = self.grid_map_eval.detach_()
            else:
                st_poses = self.st_poses_train.detach_()
                grid_map = self.grid_map_train.detach_()
            st_poses.fill_(0.0)
            st_poses[:, (0)] = poses[:, (1)] * 200.0 / self.resolution / grid_size
            st_poses[:, (1)] = poses[:, (0)] * 200.0 / self.resolution / grid_size
            st_poses[:, (2)] = poses[:, (2)] * 57.29577951308232
            rot_mat, trans_mat = get_grid(st_poses, (bs, 2, grid_size, grid_size), self.device)
            grid_map.fill_(0.0)
            grid_map[:, :, vr:, int(vr / 2):int(vr / 2 + vr)] = pred_last
            translated = F.grid_sample(grid_map, trans_mat)
            rotated = F.grid_sample(translated, rot_mat)
            rotated = rotated[:, :, vr:, int(vr / 2):int(vr / 2 + vr)]
            pred_last_st = rotated
        pose_est_input = torch.cat((pred.detach(), pred_last_st.detach()), dim=1)
        pose_conv_output = self.pose_conv(pose_est_input)
        pose_conv_output = pose_conv_output.view(-1, self.pose_conv_output_size)
        proj1 = nn.ReLU()(self.pose_proj1(pose_conv_output))
        if self.dropout > 0:
            proj1 = self.pose_dropout1(proj1)
        proj2_x = nn.ReLU()(self.pose_proj2_x(proj1))
        pred_dx = self.pose_proj3_x(proj2_x)
        proj2_y = nn.ReLU()(self.pose_proj2_y(proj1))
        pred_dy = self.pose_proj3_y(proj2_y)
        proj2_o = nn.ReLU()(self.pose_proj2_o(proj1))
        pred_do = self.pose_proj3_o(proj2_o)
        pose_pred = torch.cat((pred_dx, pred_dy, pred_do), dim=1)
        if self.use_pe == 0:
            pose_pred = pose_pred * self.use_pe
        if build_maps:
            with torch.no_grad():
                agent_view = self.agent_view.detach_()
                agent_view.fill_(0.0)
                x1 = self.map_size_cm // (self.resolution * 2) - self.vision_range // 2
                x2 = x1 + self.vision_range
                y1 = self.map_size_cm // (self.resolution * 2)
                y2 = y1 + self.vision_range
                agent_view[:, :, y1:y2, x1:x2] = pred
                corrected_pose = poses + pose_pred

                def get_new_pose_batch(pose, rel_pose_change):
                    pose[:, (1)] += rel_pose_change[:, (0)] * torch.sin(pose[:, (2)] / 57.29577951308232) + rel_pose_change[:, (1)] * torch.cos(pose[:, (2)] / 57.29577951308232)
                    pose[:, (0)] += rel_pose_change[:, (0)] * torch.cos(pose[:, (2)] / 57.29577951308232) - rel_pose_change[:, (1)] * torch.sin(pose[:, (2)] / 57.29577951308232)
                    pose[:, (2)] += rel_pose_change[:, (2)] * 57.29577951308232
                    pose[:, (2)] = torch.fmod(pose[:, (2)] - 180.0, 360.0) + 180.0
                    pose[:, (2)] = torch.fmod(pose[:, (2)] + 180.0, 360.0) - 180.0
                    return pose
                current_poses = get_new_pose_batch(current_poses, corrected_pose)
                st_pose = current_poses.clone().detach()
                st_pose[:, :2] = -(st_pose[:, :2] * 100.0 / self.resolution - self.map_size_cm // (self.resolution * 2)) / (self.map_size_cm // (self.resolution * 2))
                st_pose[:, (2)] = 90.0 - st_pose[:, (2)]
                rot_mat, trans_mat = get_grid(st_pose, agent_view.size(), self.device)
                rotated = F.grid_sample(agent_view, rot_mat)
                translated = F.grid_sample(rotated, trans_mat)
                maps2 = torch.cat((maps.unsqueeze(1), translated[:, :1, :, :]), 1)
                explored2 = torch.cat((explored.unsqueeze(1), translated[:, 1:, :, :]), 1)
                map_pred = self.pool(maps2).squeeze(1)
                exp_pred = self.pool(explored2).squeeze(1)
        else:
            map_pred = None
            exp_pred = None
            current_poses = None
        return proj_pred, fp_exp_pred, map_pred, exp_pred, pose_pred, current_poses


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
            x = hxs = self.gru(x, hxs * masks[:, (None)])
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


class Global_Policy(NNBase):

    def __init__(self, input_shape, recurrent=False, hidden_size=512, downscaling=1):
        super(Global_Policy, self).__init__(recurrent, hidden_size, hidden_size)
        out_size = int(input_shape[1] / 16.0 * input_shape[2] / 16.0)
        self.main = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(8, 32, 3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(32, 64, 3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(64, 128, 3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(128, 64, 3, stride=1, padding=1), nn.ReLU(), nn.Conv2d(64, 32, 3, stride=1, padding=1), nn.ReLU(), Flatten())
        self.linear1 = nn.Linear(out_size * 32 + 8, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 256)
        self.critic_linear = nn.Linear(256, 1)
        self.orientation_emb = nn.Embedding(72, 8)
        self.train()

    def forward(self, inputs, rnn_hxs, masks, extras):
        x = self.main(inputs)
        orientation_emb = self.orientation_emb(extras).squeeze(1)
        x = torch.cat((x, orientation_emb), 1)
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
        if model_type == 0:
            self.network = Global_Policy(obs_shape, **base_kwargs)
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

class Test_devendrachaplot_Neural_SLAM(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

