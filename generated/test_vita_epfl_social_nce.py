import sys
_module = sys.modules[__name__]
del sys
crowd_nav = _module
imitate = _module
policy = _module
cadrl = _module
lstm_rl = _module
multi_human_rl = _module
policy_factory = _module
sail = _module
sarl = _module
contrastive = _module
model = _module
sampling = _module
test = _module
utils = _module
compare = _module
configure = _module
convert = _module
dataset = _module
demonstrate = _module
explorer = _module
frames = _module
memory = _module
plot = _module
pretrain = _module
trainer = _module
transform = _module
visualize = _module
crowd_sim = _module
envs = _module
linear = _module
orca = _module
action = _module
agent = _module
human = _module
info = _module
robot = _module
state = _module
setup = _module

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


import torch


import torch.nn as nn


import torch.optim as optim


import numpy as np


import itertools


from torch.nn.functional import softmax


import math


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from torch.utils.data.sampler import SubsetRandomSampler


import copy


import time


from torch.autograd import Variable


from torch.utils.data.sampler import SequentialSampler


from sklearn.manifold import TSNE


from matplotlib import cm


import matplotlib.pyplot as plt


import matplotlib


def mlp(input_dim, mlp_dims, last_relu=False):
    layers = []
    mlp_dims = [input_dim] + mlp_dims
    for i in range(len(mlp_dims) - 1):
        layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
        if i != len(mlp_dims) - 2 or last_relu:
            layers.append(nn.ReLU())
    net = nn.Sequential(*layers)
    return net


class ValueNetwork(nn.Module):

    def __init__(self, input_dim, self_state_dim, mlp1_dims, mlp2_dims, mlp3_dims, attention_dims, with_global_state, cell_size, cell_num):
        super().__init__()
        self.self_state_dim = self_state_dim
        self.global_state_dim = mlp1_dims[-1]
        self.embedding = mlp(input_dim, mlp1_dims, last_relu=True)
        self.pairwise = mlp(mlp1_dims[-1], mlp2_dims)
        self.with_global_state = with_global_state
        if with_global_state:
            self.attention = mlp(mlp1_dims[-1] * 2, attention_dims)
        else:
            self.attention = mlp(mlp1_dims[-1], attention_dims)
        self.cell_size = cell_size
        self.cell_num = cell_num
        mlp3_input_dim = mlp2_dims[-1] + self.self_state_dim
        self.vnet = mlp(mlp3_input_dim, mlp3_dims)
        self.attention_weights = None

    def forward(self, state):
        """
        First transform the world coordinates to self-centric coordinates and then do forward computation

        :param state: tensor of shape (batch_size, # of humans, length of a rotated state)
        :return:
        """
        size = state.shape
        self_state = state[:, 0, :self.self_state_dim]
        mlp1_output = self.embedding(state.view((-1, size[2])))
        mlp2_output = self.pairwise(mlp1_output)
        if self.with_global_state:
            global_state = torch.mean(mlp1_output.view(size[0], size[1], -1), 1, keepdim=True)
            global_state = global_state.expand((size[0], size[1], self.global_state_dim)).contiguous().view(-1, self.global_state_dim)
            attention_input = torch.cat([mlp1_output, global_state], dim=1)
        else:
            attention_input = mlp1_output
        scores = self.attention(attention_input).view(size[0], size[1], 1).squeeze(dim=2)
        weights = softmax(scores, dim=1).unsqueeze(2)
        self.attention_weights = weights[0, :, 0].data.cpu().numpy()
        features = mlp2_output.view(size[0], size[1], -1)
        weighted_feature = torch.sum(torch.mul(weights, features), dim=1)
        joint_state = torch.cat([self_state, weighted_feature], dim=1)
        value = self.vnet(joint_state)
        return value


class ValueNetwork1(nn.Module):

    def __init__(self, input_dim, self_state_dim, mlp_dims, lstm_hidden_dim):
        super().__init__()
        self.self_state_dim = self_state_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.mlp = mlp(self_state_dim + lstm_hidden_dim, mlp_dims)
        self.lstm = nn.LSTM(input_dim, lstm_hidden_dim, batch_first=True)

    def forward(self, state):
        """
        First transform the world coordinates to self-centric coordinates and then do forward computation

        :param state: tensor of shape (batch_size, # of humans, length of a joint state)
        :return:
        """
        size = state.shape
        self_state = state[:, 0, :self.self_state_dim]
        h0 = torch.zeros(1, size[0], self.lstm_hidden_dim)
        c0 = torch.zeros(1, size[0], self.lstm_hidden_dim)
        output, (hn, cn) = self.lstm(state, (h0, c0))
        hn = hn.squeeze(0)
        joint_state = torch.cat([self_state, hn], dim=1)
        value = self.mlp(joint_state)
        return value


class ValueNetwork2(nn.Module):

    def __init__(self, input_dim, self_state_dim, mlp1_dims, mlp_dims, lstm_hidden_dim):
        super().__init__()
        self.self_state_dim = self_state_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.mlp1 = mlp(input_dim, mlp1_dims)
        self.mlp = mlp(self_state_dim + lstm_hidden_dim, mlp_dims)
        self.lstm = nn.LSTM(mlp1_dims[-1], lstm_hidden_dim, batch_first=True)

    def forward(self, state):
        """
        First transform the world coordinates to self-centric coordinates and then do forward computation

        :param state: tensor of shape (batch_size, # of humans, length of a joint state)
        :return:
        """
        size = state.shape
        self_state = state[:, 0, :self.self_state_dim]
        state = torch.reshape(state, (-1, size[2]))
        mlp1_output = self.mlp1(state)
        mlp1_output = torch.reshape(mlp1_output, (size[0], size[1], -1))
        h0 = torch.zeros(1, size[0], self.lstm_hidden_dim)
        c0 = torch.zeros(1, size[0], self.lstm_hidden_dim)
        output, (hn, cn) = self.lstm(mlp1_output, (h0, c0))
        hn = hn.squeeze(0)
        joint_state = torch.cat([self_state, hn], dim=1)
        value = self.mlp(joint_state)
        return value


class MultiAgentTransform:

    def __init__(self, num_human, state_dim=4):
        self.num_human = num_human
        self.mask = torch.ones(num_human, num_human, state_dim).bool()
        for k in range(num_human):
            self.mask[k, k] = False

    def transform_frame(self, frame):
        bs = frame.shape[0]
        compare = frame.unsqueeze(1) - frame.unsqueeze(2)
        relative = torch.masked_select(compare, self.mask.repeat(bs, 1, 1, 1)).reshape(bs, self.num_human, -1)
        state = torch.cat([frame, relative], axis=2)
        return state


class ExtendedNetwork(nn.Module):
    """ Policy network for imitation learning """

    def __init__(self, num_human, embedding_dim=64, hidden_dim=64, local_dim=32):
        super().__init__()
        self.num_human = num_human
        self.transform = MultiAgentTransform(num_human)
        self.robot_encoder = nn.Sequential(nn.Linear(4, local_dim), nn.ReLU(inplace=True), nn.Linear(local_dim, local_dim), nn.ReLU(inplace=True))
        self.human_encoder = nn.Sequential(nn.Linear(4 * self.num_human, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True))
        self.human_head = nn.Sequential(nn.Linear(hidden_dim, local_dim), nn.ReLU(inplace=True))
        self.joint_embedding = nn.Sequential(nn.Linear(local_dim * 2, embedding_dim), nn.ReLU(inplace=True))
        self.pairwise = nn.Sequential(nn.Linear(embedding_dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim))
        self.attention = nn.Sequential(nn.Linear(embedding_dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))
        self.task_encoder = nn.Sequential(nn.Linear(4, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True))
        self.joint_encoder = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(inplace=True))
        self.planner = nn.Linear(hidden_dim, 2)

    def forward(self, robot_state, crowd_obsv):
        if len(robot_state.shape) < 2:
            robot_state = robot_state.unsqueeze(0)
            crowd_obsv = crowd_obsv.unsqueeze(0)
        emb_robot = self.robot_encoder(robot_state[:, :4])
        human_state = self.transform.transform_frame(crowd_obsv)
        feat_human = self.human_encoder(human_state)
        emb_human = self.human_head(feat_human)
        emb_concat = torch.cat([emb_robot.unsqueeze(1).repeat(1, self.num_human, 1), emb_human], axis=2)
        emb_pairwise = self.joint_embedding(emb_concat)
        feat_pairwise = self.pairwise(emb_pairwise)
        logit_pairwise = self.attention(emb_pairwise)
        score_pairwise = nn.functional.softmax(logit_pairwise, dim=1)
        feat_crowd = torch.sum(feat_pairwise * score_pairwise, dim=1)
        reparam_robot_state = torch.cat([robot_state[:, -2:] - robot_state[:, :2], robot_state[:, 2:4]], axis=1)
        feat_task = self.task_encoder(reparam_robot_state)
        feat_joint = self.joint_encoder(torch.cat([feat_task, feat_crowd], axis=1))
        action = self.planner(feat_joint)
        return action, feat_joint


class ProjHead(nn.Module):
    """
    Nonlinear projection head that maps the extracted motion features to the embedding space
    """

    def __init__(self, feat_dim, hidden_dim, head_dim):
        super(ProjHead, self).__init__()
        self.head = nn.Sequential(nn.Linear(feat_dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, head_dim))

    def forward(self, feat):
        return self.head(feat)


class EventEncoder(nn.Module):
    """
    Event encoder that maps an sampled event (location & time) to the embedding space
    """

    def __init__(self, hidden_dim, head_dim):
        super(EventEncoder, self).__init__()
        self.temporal = nn.Sequential(nn.Linear(1, hidden_dim), nn.ReLU(inplace=True))
        self.spatial = nn.Sequential(nn.Linear(2, hidden_dim), nn.ReLU(inplace=True))
        self.encoder = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, head_dim))

    def forward(self, state, time):
        emb_state = self.spatial(state)
        emb_time = self.temporal(time)
        return self.encoder(torch.cat([emb_time, emb_state], axis=-1))


class SpatialEncoder(nn.Module):
    """
    Spatial encoder that maps an sampled location to the embedding space
    """

    def __init__(self, hidden_dim, head_dim):
        super(SpatialEncoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(2, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, head_dim))

    def forward(self, state):
        return self.encoder(state)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ExtendedNetwork,
     lambda: ([], {'num_human': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (ProjHead,
     lambda: ([], {'feat_dim': 4, 'hidden_dim': 4, 'head_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_vita_epfl_social_nce(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

