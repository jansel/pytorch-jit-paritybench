import sys
_module = sys.modules[__name__]
del sys
core = _module
data = _module
base_collector = _module
benchmark = _module
benchmark_utils = _module
benchmark_dataset_saver = _module
bev_vae_dataset = _module
carla_benchmark_collector = _module
cict_dataset = _module
cilrs_dataset = _module
lbc_dataset = _module
envs = _module
base_drive_env = _module
drive_env_wrapper = _module
md_macro_env = _module
md_traj_env = _module
scenario_carla_env = _module
simple_carla_env = _module
eval = _module
base_evaluator = _module
carla_benchmark_evaluator = _module
serial_evaluator = _module
single_carla_evaluator = _module
models = _module
bev_speed_model = _module
cilrs_model = _module
common_model = _module
lbc_model = _module
model_wrappers = _module
mpc_controller = _module
pid_controller = _module
vae_model = _module
vehicle_controller = _module
policy = _module
auto_policy = _module
base_carla_policy = _module
cilrs_policy = _module
lbc_policy = _module
control_decoder = _module
traj_ppo = _module
traj_vac = _module
simulators = _module
base_simulator = _module
carla_data_provider = _module
carla_scenario_simulator = _module
carla_simulator = _module
fake_simulator = _module
srunner = _module
scenarioconfigs = _module
openscenario_configuration = _module
route_scenario_configuration = _module
scenario_configuration = _module
scenariomanager = _module
actorcontrols = _module
actor_control = _module
basic_control = _module
external_control = _module
npc_vehicle_control = _module
pedestrian_control = _module
simple_vehicle_control = _module
vehicle_longitudinal_control = _module
result_writer = _module
scenario_manager = _module
scenarioatomics = _module
atomic_behaviors = _module
atomic_criteria = _module
atomic_trigger_conditions = _module
timer = _module
traffic_events = _module
watchdog = _module
weather_sim = _module
scenarios = _module
basic_scenario = _module
change_lane = _module
control_loss = _module
control_loss_new = _module
cut_in = _module
cut_in_new = _module
follow_leading_vehicle = _module
follow_leading_vehicle_new = _module
junction_crossing_route = _module
maneuver_opposite_direction = _module
no_signal_junction_crossing = _module
object_crash_intersection = _module
object_crash_vehicle = _module
opposite_direction = _module
other_leading_vehicle = _module
route_scenario = _module
signalized_junction_left_turn = _module
signalized_junction_right_turn = _module
signalized_junction_straight = _module
tools = _module
openscenario_parser = _module
py_trees_port = _module
route_manipulation = _module
route_parser = _module
scenario_helper = _module
scenario_parser = _module
utils = _module
data_utils = _module
augmenter = _module
bev_utils = _module
coil_sampler = _module
data_writter = _module
splitter = _module
env_utils = _module
stuck_detector = _module
learner_utils = _module
log_saver_utils = _module
loss_utils = _module
optim_utils = _module
model_utils = _module
common = _module
resnet = _module
others = _module
checkpoint_helper = _module
ding_utils = _module
general_helper = _module
image_helper = _module
tcp_helper = _module
visualizer = _module
planner = _module
basic_planner = _module
behavior_planner = _module
lbc_planner = _module
planner_utils = _module
simulator_utils = _module
carla_agents = _module
navigation = _module
agent = _module
basic_agent = _module
behavior_agent = _module
controller = _module
global_route_planner = _module
global_route_planner_dao = _module
local_planner = _module
local_planner_behavior = _module
roaming_agent = _module
types_behavior = _module
misc = _module
carla_utils = _module
map_utils = _module
agent_manager_utils = _module
discrete_policy = _module
engine_utils = _module
evaluator_utils = _module
idm_policy_utils = _module
macro_policy = _module
map_manager_utils = _module
navigation_utils = _module
traffic_manager_utils = _module
traj_policy = _module
vehicle_utils = _module
sensor_utils = _module
auto_eval = _module
auto_run = _module
auto_run_case = _module
basic_tools = _module
coordinate_transformation = _module
parameters = _module
cict_datasaver = _module
cict_eval = _module
cict_eval_GAN = _module
cict_eval_traj = _module
cict_model = _module
cict_policy = _module
cict_test = _module
cict_train_GAN = _module
cict_train_traj = _module
collect_data = _module
collect_pm = _module
post = _module
cilrs_collect_data = _module
cilrs_env_wrapper = _module
cilrs_eval = _module
cilrs_test = _module
cilrs_train = _module
test_render = _module
train_drex_model = _module
train_ppo = _module
train_ppo_drex = _module
carla_env = _module
dataset = _module
eval_policy = _module
models = _module
train_rl = _module
train_sl = _module
latent_dqn_eval = _module
latent_dqn_test = _module
latent_dqn_train = _module
latent_rl_env = _module
model = _module
train_vae = _module
lbc_bev_eval = _module
lbc_bev_test = _module
lbc_birdview_train = _module
lbc_collect_data = _module
lbc_env_wrapper = _module
lbc_image_eval = _module
lbc_image_test = _module
lbc_img_train_phase0 = _module
lbc_img_train_phase1 = _module
basic_env_train = _module
macro_env_demo = _module
macro_env_dqn_eval = _module
macro_env_dqn_train = _module
macro_env_ppo_eval = _module
macro_env_ppo_train = _module
macro_env_test = _module
macro_env_train = _module
model = _module
tdv_ppo_train = _module
tdv_sac_train = _module
ddpg_config = _module
dqn_config = _module
ppo_config = _module
sac_config = _module
td3_config = _module
env_wrapper = _module
model = _module
simple_rl_eval = _module
simple_rl_test = _module
simple_rl_train = _module
conf = _module
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


import numpy as np


from typing import Any


from typing import Dict


import torch


from torchvision import transforms


from torch.utils.data import Dataset


import collections


import math


import copy


import random


import torchvision.transforms as transforms


from scipy.special import comb


import pandas as pd


from collections import deque


from itertools import product


from typing import List


from typing import Callable


from typing import Optional


from collections import defaultdict


from typing import Tuple


import torch.nn as nn


from typing import Union


from torchvision import models


from torch import nn


from torch.nn import functional as F


from collections import namedtuple


import torch.nn.functional as F


from torch.distributions import Normal


from torch.distributions import Independent


from torch.utils.data.sampler import Sampler


from collections import OrderedDict


import torchvision


import torch.utils.model_zoo as model_zoo


from functools import partial


from torch.utils.data import DataLoader


from torchvision.utils import save_image


import time


from torch.autograd import grad


from torchvision.transforms.transforms import ToTensor


import scipy.misc


import torch.optim as optim


from torch.utils.data import WeightedRandomSampler


from torchvision.transforms.transforms import Grayscale


from torch.optim import Adam


from collections.abc import Iterable


from copy import deepcopy


from torch.distributions.categorical import Categorical


from enum import Enum


import re


import torchvision.models as models


import logging


from torch import optim


import torchvision.utils as vutils


class BEVSpeedConvEncoder(nn.Module):
    """
    Convolutional encoder of Bird-eye View image and speed input. It takes a BeV image and a speed scalar as input.
    The BeV image is encoded by a convolutional encoder, to get a embedding feature which is half size of the
    embedding length. Then the speed value is repeated for half embedding length time, and concated to the above
    feature to get a final feature.

    :Arguments:
        - obs_shape (Tuple): BeV image shape.
        - hidden_dim_list (List): Conv encoder hidden layer dimension list.
        - embedding_size (int): Embedding feature dimensions.
        - kernel_size (List, optional): Conv kernel size for each layer. Defaults to [8, 4, 3].
        - stride (List, optional): Conv stride for each layer. Defaults to [4, 2, 1].
    """

    def __init__(self, obs_shape: Tuple, hidden_dim_list: List, embedding_size: int, kernel_size: List=[8, 4, 3], stride: List=[4, 2, 1]) ->None:
        super().__init__()
        assert len(kernel_size) == len(stride), (kernel_size, stride)
        self._obs_shape = obs_shape
        self._embedding_size = embedding_size
        self._relu = nn.ReLU()
        layers = []
        input_dim = obs_shape[0]
        for i in range(len(hidden_dim_list)):
            layers.append(nn.Conv2d(input_dim, hidden_dim_list[i], kernel_size[i], stride[i]))
            layers.append(self._relu)
            input_dim = hidden_dim_list[i]
        layers.append(nn.Flatten())
        self._model = nn.Sequential(*layers)
        flatten_size = self._get_flatten_size()
        self._mid = nn.Linear(flatten_size, self._embedding_size // 2)

    def _get_flatten_size(self) ->int:
        test_data = torch.randn(1, *self._obs_shape)
        with torch.no_grad():
            output = self._model(test_data)
        return output.shape[1]

    def forward(self, data: Dict) ->torch.Tensor:
        """
        Forward computation of encoder

        :Arguments:
            - data (Dict): Input data, must contain 'birdview' and 'speed'

        :Returns:
            torch.Tensor: Embedding feature.
        """
        image = data['birdview'].permute(0, 3, 1, 2)
        speed = data['speed']
        x = self._model(image)
        x = self._mid(x)
        speed_embedding_size = self._embedding_size - self._embedding_size // 2
        speed_vec = torch.unsqueeze(speed, 1).repeat(1, speed_embedding_size)
        h = torch.cat((x, speed_vec), dim=1)
        return h


class CILRSModel(nn.Module):

    def __init__(self, backbone='resnet18', pretrained=True, normalize=True, num_branch=6, speed_dim=1, embedding_dim=512, hidden_size=256, input_speed=True, predict_speed=True):
        super().__init__()
        self._normalize = normalize
        assert backbone in ['resnet18', 'resnet34', 'resnet50'], backbone
        backbone_cls = {'resnet18': models.resnet18, 'resnet34': models.resnet34, 'resnet50': models.resnet50}[backbone]
        self._backbone = backbone_cls(pretrained=pretrained)
        self._backbone.fc = nn.Sequential()
        self._num_branch = num_branch
        self._input_speed = input_speed
        self.predict_speed = predict_speed
        if input_speed:
            self._speed_in = nn.Sequential(nn.Linear(speed_dim, hidden_size), nn.ReLU(True), nn.Linear(hidden_size, embedding_dim))
            embedding_dim *= 2
        if predict_speed:
            self._speed_out = nn.Sequential(nn.Linear(embedding_dim, hidden_size), nn.ReLU(True), nn.Linear(hidden_size, hidden_size), nn.ReLU(True), nn.Linear(hidden_size, speed_dim))
        fc_branch_list = []
        for i in range(num_branch):
            fc_branch_list.append(nn.Sequential(nn.Linear(embedding_dim, hidden_size), nn.ReLU(True), nn.Linear(hidden_size, hidden_size), nn.ReLU(True), nn.Linear(hidden_size, 3), nn.Sigmoid()))
        self._branches = nn.ModuleList(fc_branch_list)

    def _normalize_imagenet(self, x):
        """
        Normalize input images according to ImageNet standards.
        :Arguments:
            x (tensor): input images
        """
        x = x.clone()
        x[:, 0] = (x[:, 0] - 0.485) / 0.229
        x[:, 1] = (x[:, 1] - 0.456) / 0.224
        x[:, 2] = (x[:, 2] - 0.406) / 0.225
        return x

    def encode(self, input_images):
        embedding = 0
        for x in input_images:
            if self._normalize:
                x = self._normalize_imagenet(x)
            embedding += self._backbone(x)
        return embedding

    def forward(self, embedding, speed, command):
        if len(command.shape) == 1:
            command = command.unsqueeze(1)
        if self._input_speed:
            if len(speed.shape) == 1:
                speed = speed.unsqueeze(1)
            embedding = torch.cat([embedding, self._speed_in(speed)], 1)
        control_pred = 0.0
        for i, branch in enumerate(self._branches):
            control_pred += branch(embedding) * (i == command - 1)
        if self.predict_speed:
            speed_pred = self._speed_out(embedding)
            return control_pred, speed_pred
        return control_pred


class VanillaVAE(nn.Module):
    """
    Vanilla Variational Auto Encoder model.

    :Interfaces: encode, decode, reparameterize, forward, loss_function, sample, generate

    :Arguments:
        - in_channels (int): the channel number of input
        - latent_dim (int): the latent dimension of the middle representation
        - hidden_dims (List): the hidden dimensions of each layer in the MLP architecture in encoder and decoder
        - kld_weight(float): the weight of KLD loss
    """

    def __init__(self, in_channels: int, latent_dim: int, hidden_dims: List=None, kld_weight: float=0.1) ->None:
        super(VanillaVAE, self).__init__()
        self.latent_dim = latent_dim
        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        self.hidden_dims = hidden_dims
        self.kld_weight = kld_weight
        for h_dim in hidden_dims:
            modules.append(nn.Sequential(nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(h_dim), nn.LeakyReLU()))
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 36, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 36, latent_dim)
        modules = []
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 36)
        hidden_dims.reverse()
        for i in range(len(hidden_dims) - 1):
            modules.append(nn.Sequential(nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(hidden_dims[i + 1]), nn.LeakyReLU()))
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(hidden_dims[-1]), nn.LeakyReLU(), nn.Conv2d(hidden_dims[-1], out_channels=7, kernel_size=3, padding=1), nn.Sigmoid())

    def encode(self, input: torch.Tensor) ->List[torch.Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.

        :Arguments:
            - input (Tensor): Input tensor to encode [N x C x H x W]
        :Returns:
            Tensor: List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return mu, log_var

    def decode(self, z: torch.Tensor) ->torch.Tensor:
        """
        Maps the given latent codes
        onto the image space.

        :Arguments:
            - z (Tensor): [B x D]
        :Returns:
            Tensor: Output decode tensor [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 6, 6)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) ->torch.Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).

        :Arguments:
            - mu (Tensor): Mean of the latent Gaussian [B x D]
            - logvar (Tensor): Standard deviation of the latent Gaussian [B x D]
        :Returns:
            Tensor: [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: torch.Tensor, **kwargs) ->List[torch.Tensor]:
        """
        [summary]

        :Arguments:
            - input (torch.Tensor): Input tensor

        :Returns:
            List[torch.Tensor]: Input and output tensor
        """
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def loss_function(self, *args, **kwargs) ->Dict:
        """
        Computes the VAE loss function.

        :math:`KL(N(\\mu, \\sigma), N(0, 1)) = \\log \\frac{1}{\\sigma} + \\frac{\\sigma^2 + \\mu^2}{2} - \\frac{1}{2}`

        :Returns:
            Dict: Dictionary containing loss information
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        kld_weight = self.kld_weight
        recons_loss = 0
        """
        weight = [8.7924e-01, 7.4700e-02, 1.0993e-02, 6.1075e-04, 2.6168e-03, 2.8066e-02, 3.7737e-03]
        vd = 1
        for i in range(7):
            cur = F.l1_loss(recons[:, i, ...], input[:, i, ...])
            recons_loss += 1 / weight[i] * cur * vd
            ret[str(i)] = cur
            if i==0 and cur > 0.05:
                vd = 0
        """
        recons_loss = F.mse_loss(recons, input)
        if recons_loss < 0.05:
            recons_loss = F.l1_loss(recons, input)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'reconstruction_Loss': recons_loss, 'KLD': -kld_loss}

    def sample(self, num_samples: int, current_device: int, **kwargs) ->torch.Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.

        :Arguments:
            - num_samples(Int): Number of samples.
            - current_device(Int): Device to run the model.
        :Returns:
            Tensor: Sampled decode tensor.
        """
        z = torch.randn(num_samples, self.latent_dim)
        z = z
        samples = self.decode(z)
        return samples

    def generate(self, x: torch.Tensor, **kwargs) ->torch.Tensor:
        """
        Given an input image x, returns the reconstructed image

        :Arguments:
            - x(Tensor): [B x C x H x W]
        :Returns:
            Tensor: [B x C x H x W]
        """
        return self.forward(x)[0]


class WpDecoder(nn.Module):

    def __init__(self, control_num=2, seq_len=30, use_relative_pos=True, dt=0.03, traj_control_mode='jerk'):
        super(WpDecoder, self).__init__()
        self.control_num = control_num
        self.seq_len = seq_len
        self.use_relative_pos = use_relative_pos
        self.dt = dt
        self.traj_control_mode = traj_control_mode

    def plant_model_acc(self, prev_state_batch, pedal_batch, steering_batch, dt=0.03):
        prev_state = prev_state_batch
        x_t = prev_state[:, 0]
        y_t = prev_state[:, 1]
        psi_t = prev_state[:, 2]
        v_t = prev_state[:, 3]
        steering_batch = steering_batch * 0.5
        steering_batch = torch.clamp(steering_batch, -0.5, 0.5)
        pedal_batch = pedal_batch * 5
        pedal_batch = torch.clamp(pedal_batch, -5, 5)
        beta = steering_batch
        a_t = pedal_batch
        v_t_1 = v_t + a_t * dt
        v_t_1 = torch.clamp(v_t_1, 0, 10)
        psi_dot = v_t * torch.tan(beta) / 2.5
        psi_t_1 = psi_dot * dt + psi_t
        x_dot = v_t_1 * torch.cos(psi_t_1)
        y_dot = v_t_1 * torch.sin(psi_t_1)
        x_t_1 = x_dot * dt + x_t
        y_t_1 = y_dot * dt + y_t
        current_state = torch.stack([x_t_1, y_t_1, psi_t_1, v_t_1], dim=1)
        return current_state

    def plant_model_jerk(self, prev_state_batch, jerk_batch, steering_rate_batch, dt=0.03):
        prev_state = prev_state_batch
        x_t = prev_state[:, 0]
        y_t = prev_state[:, 1]
        psi_t = prev_state[:, 2]
        v_t = prev_state[:, 3]
        pedal_t = prev_state[:, 4]
        steering_t = prev_state[:, 5]
        jerk_batch = jerk_batch * 4
        steering_rate_batch = steering_rate_batch * 0.5
        jerk_batch = torch.clamp(jerk_batch, -4, 4)
        steering_rate_batch = torch.clamp(steering_rate_batch, -0.5, 0.5)
        pedal_batch = pedal_t + jerk_batch * dt
        pedal_batch = torch.clamp(pedal_batch, -5, 5)
        steering_batch = steering_t + steering_rate_batch * dt
        steering_batch = torch.clamp(steering_batch, -0.5, 0.5)
        beta = steering_batch
        a_t = pedal_batch
        v_t_1 = v_t + a_t * dt
        v_t_1 = torch.clamp(v_t_1, 0, 10)
        psi_dot = v_t * torch.tan(beta) / 2.5
        psi_dot = torch.clamp(psi_dot, -3.14 / 2, 3.14 / 2)
        psi_t_1 = psi_dot * dt + psi_t
        x_dot = v_t_1 * torch.cos(psi_t_1)
        y_dot = v_t_1 * torch.sin(psi_t_1)
        x_t_1 = x_dot * dt + x_t
        y_t_1 = y_dot * dt + y_t
        current_state = torch.stack([x_t_1, y_t_1, psi_t_1, v_t_1, pedal_batch, steering_batch], dim=1)
        return current_state

    def decode(self, z, init_state):
        generated_traj = []
        prev_state = init_state
        assert z.shape[1] == self.seq_len * 2
        for i in range(self.seq_len):
            control_1 = z[:, 2 * i]
            control_2 = z[:, 2 * i + 1]
            if self.traj_control_mode == 'jerk':
                curr_state = self.plant_model_jerk(prev_state, control_1, control_2, self.dt)
            elif self.traj_control_mode == 'acc':
                curr_state = self.plant_model_acc(prev_state, control_1, control_2, self.dt)
            generated_traj.append(curr_state)
            prev_state = curr_state
        generated_traj = torch.stack(generated_traj, dim=1)
        return generated_traj

    def forward(self, z, init_state):
        return self.decode(z, init_state)


class CCDecoder(nn.Module):

    def __init__(self, control_num=2, seq_len=30, use_relative_pos=True, dt=0.03, traj_control_mode='jerk'):
        super(CCDecoder, self).__init__()
        self.control_num = control_num
        self.seq_len = seq_len
        self.use_relative_pos = use_relative_pos
        self.dt = dt
        self.traj_control_mode = traj_control_mode

    def plant_model_acc(self, prev_state_batch, pedal_batch, steering_batch, dt=0.03):
        prev_state = prev_state_batch
        x_t = prev_state[:, 0]
        y_t = prev_state[:, 1]
        psi_t = prev_state[:, 2]
        v_t = prev_state[:, 3]
        steering_batch = steering_batch * 0.5
        steering_batch = torch.clamp(steering_batch, -0.5, 0.5)
        pedal_batch = pedal_batch * 5
        pedal_batch = torch.clamp(pedal_batch, -5, 5)
        beta = steering_batch
        a_t = pedal_batch
        v_t_1 = v_t + a_t * dt
        v_t_1 = torch.clamp(v_t_1, 0, 10)
        psi_dot = v_t * torch.tan(beta) / 2.5
        psi_t_1 = psi_dot * dt + psi_t
        x_dot = v_t_1 * torch.cos(psi_t_1)
        y_dot = v_t_1 * torch.sin(psi_t_1)
        x_t_1 = x_dot * dt + x_t
        y_t_1 = y_dot * dt + y_t
        current_state = torch.stack([x_t_1, y_t_1, psi_t_1, v_t_1], dim=1)
        return current_state

    def plant_model_jerk(self, prev_state_batch, jerk_batch, steering_rate_batch, dt=0.03):
        prev_state = prev_state_batch
        x_t = prev_state[:, 0]
        y_t = prev_state[:, 1]
        psi_t = prev_state[:, 2]
        v_t = prev_state[:, 3]
        pedal_t = prev_state[:, 4]
        steering_t = prev_state[:, 5]
        jerk_batch = jerk_batch * 4
        steering_rate_batch = steering_rate_batch * 0.5
        jerk_batch = torch.clamp(jerk_batch, -4, 4)
        steering_rate_batch = torch.clamp(steering_rate_batch, -0.5, 0.5)
        pedal_batch = pedal_t + jerk_batch * dt
        pedal_batch = torch.clamp(pedal_batch, -5, 5)
        steering_batch = steering_t + steering_rate_batch * dt
        steering_batch = torch.clamp(steering_batch, -0.5, 0.5)
        beta = steering_batch
        a_t = pedal_batch
        v_t_1 = v_t + a_t * dt
        v_t_1 = torch.clamp(v_t_1, 0, 10)
        psi_dot = v_t * torch.tan(beta) / 2.5
        psi_dot = torch.clamp(psi_dot, -3.14 / 2, 3.14 / 2)
        psi_t_1 = psi_dot * dt + psi_t
        x_dot = v_t_1 * torch.cos(psi_t_1)
        y_dot = v_t_1 * torch.sin(psi_t_1)
        x_t_1 = x_dot * dt + x_t
        y_t_1 = y_dot * dt + y_t
        current_state = torch.stack([x_t_1, y_t_1, psi_t_1, v_t_1, pedal_batch, steering_batch], dim=1)
        return current_state

    def decode(self, z, init_state):
        generated_traj = []
        prev_state = init_state
        assert z.shape[1] == 2
        for i in range(self.seq_len):
            control_1 = z[:, 0]
            control_2 = z[:, 1]
            if self.traj_control_mode == 'jerk':
                curr_state = self.plant_model_jerk(prev_state, control_1, control_2, self.dt)
            elif self.traj_control_mode == 'acc':
                curr_state = self.plant_model_acc(prev_state, control_1, control_2, self.dt)
            generated_traj.append(curr_state)
            prev_state = curr_state
        generated_traj = torch.stack(generated_traj, dim=1)
        return generated_traj

    def forward(self, z, init_state):
        return self.decode(z, init_state)


class LocationLoss(torch.nn.Module):

    def __init__(self, crop_size=192, **kwargs):
        super().__init__()
        self._crop_size = crop_size

    def forward(self, pred_locations, teac_locations):
        pred_locations = pred_locations / (0.5 * self._crop_size) - 1
        return torch.mean(torch.abs(pred_locations - teac_locations), dim=(1, 2, 3))


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, input_channel=7, num_classes=1000, zero_init_residual=False, bias_first=True):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=bias_first)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


model_funcs = {'resnet18': (BasicBlock, [2, 2, 2, 2], -1), 'resnet34': (BasicBlock, [3, 4, 6, 3], 512), 'resnet50': (Bottleneck, [3, 4, 6, 3], -1), 'resnet101': (Bottleneck, [3, 4, 23, 3], -1), 'resnet152': (Bottleneck, [3, 8, 36, 3], -1)}


model_urls = {'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth', 'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth', 'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth', 'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth', 'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'}


def get_resnet(model_name='resnet18', pretrained=False, **kwargs):
    block, layers, c_out = model_funcs[model_name]
    model = ResNet(block, layers, **kwargs)
    if pretrained and kwargs.get('input_channel', 3) == 3:
        url = model_urls[model_name]
        None
        model.load_state_dict(model_zoo.load_url(url))
    return model, c_out


class ResnetBase(nn.Module):

    def __init__(self, backbone, input_channel=3, bias_first=True, pretrained=False):
        super().__init__()
        conv, c = get_resnet(backbone, input_channel=input_channel, bias_first=bias_first, pretrained=pretrained)
        self.conv = conv
        self.c = c
        self.backbone = backbone
        self.input_channel = input_channel
        self.bias_first = bias_first


class NormalizeV2(nn.Module):

    def __init__(self, mean, std):
        super().__init__()
        self.mean = torch.FloatTensor(mean).reshape(1, 3, 1, 1)
        self.std = torch.FloatTensor(std).reshape(1, 3, 1, 1)

    def forward(self, x):
        return (x - self.mean) / self.std


class SpatialSoftmax(nn.Module):

    def __init__(self, height, width, channel, temperature=None, data_format='NCHW'):
        super().__init__()
        self.data_format = data_format
        self.height = height
        self.width = width
        self.channel = channel
        self.temperature = 1.0
        pos_x, pos_y = np.meshgrid(np.linspace(-1.0, 1.0, self.height), np.linspace(-1.0, 1.0, self.width))
        pos_x = torch.from_numpy(pos_x.reshape(self.height * self.width)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self.height * self.width)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)

    def forward(self, feature):
        if self.data_format == 'NHWC':
            feature = feature.transpose(1, 3).tranpose(2, 3).view(-1, self.height * self.width)
        else:
            feature = feature.view(-1, self.height * self.width)
        weight = F.softmax(feature / self.temperature, dim=-1)
        expected_x = torch.sum(torch.autograd.Variable(self.pos_x) * weight, dim=1, keepdim=True)
        expected_y = torch.sum(torch.autograd.Variable(self.pos_y) * weight, dim=1, keepdim=True)
        expected_xy = torch.cat([expected_x, expected_y], 1)
        feature_keypoints = expected_xy.view(-1, self.channel, 2)
        return feature_keypoints


class Join(nn.Module):

    def __init__(self, params=None, module_name='Default'):
        super(Join, self).__init__()
        if params is None:
            raise ValueError('Creating a NULL fully connected block')
        if 'mode' not in params:
            raise ValueError(' Missing the mode parameter ')
        if 'after_process' not in params:
            raise ValueError(' Missing the after_process parameter ')
        """" ------------------ IMAGE MODULE ---------------- """
        self.after_process = params['after_process']
        self.mode = params['mode']

    def forward(self, x, m):
        if self.mode == 'cat':
            j = torch.cat((x, m), 1)
        else:
            raise ValueError('Mode to join networks not found')
        return self.after_process(j)


class Conv(nn.Module):

    def __init__(self, params=None, module_name='Default'):
        super(Conv, self).__init__()
        if params is None:
            raise ValueError('Creating a NULL fully connected block')
        if 'channels' not in params:
            raise ValueError(' Missing the channel sizes parameter ')
        if 'kernels' not in params:
            raise ValueError(' Missing the kernel sizes parameter ')
        if 'strides' not in params:
            raise ValueError(' Missing the strides parameter ')
        if 'dropouts' not in params:
            raise ValueError(' Missing the dropouts parameter ')
        if 'end_layer' not in params:
            raise ValueError(' Missing the end module parameter ')
        if len(params['dropouts']) != len(params['channels']) - 1:
            raise ValueError('Dropouts should be from the len of channel_sizes minus 1')
        """" ------------------ IMAGE MODULE ---------------- """
        self.layers = []
        for i in range(0, len(params['channels']) - 1):
            conv = nn.Conv2d(in_channels=params['channels'][i], out_channels=params['channels'][i + 1], kernel_size=params['kernels'][i], stride=params['strides'][i])
            dropout = nn.Dropout2d(p=params['dropouts'][i])
            relu = nn.ReLU(inplace=True)
            bn = nn.BatchNorm2d(params['channels'][i + 1])
            layer = nn.Sequential(*[conv, bn, dropout, relu])
            self.layers.append(layer)
        self.layers = nn.Sequential(*self.layers)
        self.module_name = module_name

    def forward(self, x):
        """ Each conv is: conv + batch normalization + dropout + relu """
        x = self.layers(x)
        x = x.view(-1, self.num_flat_features(x))
        return x, self.layers

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def get_conv_output(self, shape):
        """
           By inputing the shape of the input, simulate what is the ouputsize.
        """
        bs = 1
        input = torch.autograd.Variable(torch.rand(bs, *shape))
        output_feat, _ = self.forward(input)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size


class FC(nn.Module):

    def __init__(self, params=None, module_name='Default'):
        super(FC, self).__init__()
        """" ---------------------- FC ----------------------- """
        if params is None:
            raise ValueError('Creating a NULL fully connected block')
        if 'neurons' not in params:
            raise ValueError(' Missing the kernel sizes parameter ')
        if 'dropouts' not in params:
            raise ValueError(' Missing the dropouts parameter ')
        if 'end_layer' not in params:
            raise ValueError(' Missing the end module parameter ')
        if len(params['dropouts']) != len(params['neurons']) - 1:
            raise ValueError('Dropouts should be from the len of kernels minus 1')
        self.layers = []
        for i in range(0, len(params['neurons']) - 1):
            fc = nn.Linear(params['neurons'][i], params['neurons'][i + 1])
            dropout = nn.Dropout2d(p=params['dropouts'][i])
            relu = nn.ReLU(inplace=True)
            if i == len(params['neurons']) - 2 and params['end_layer']:
                self.layers.append(nn.Sequential(*[fc, dropout]))
            else:
                self.layers.append(nn.Sequential(*[fc, dropout, relu]))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        if type(x) is tuple:
            return self.layers(x[0]), x[1]
        else:
            return self.layers(x)


class Branching(nn.Module):

    def __init__(self, branched_modules=None):
        """

        Args:
            branch_config: A tuple containing number of branches and the output size.
        """
        super(Branching, self).__init__()
        """ ---------------------- BRANCHING MODULE --------------------- """
        if branched_modules is None:
            raise ValueError('No model provided after branching')
        self.branched_modules = nn.ModuleList(branched_modules)

    def forward(self, x):
        branches_outputs = []
        for branch in self.branched_modules:
            branches_outputs.append(branch(x))
        return branches_outputs


class ResNetv2(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNetv2, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(2)
        if block.__name__ == 'Bottleneck':
            self.fc = nn.Linear(6144, num_classes)
        else:
            self.fc = nn.Linear(1536, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x0 = self.maxpool(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x = self.avgpool(x4)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x, [x0, x1, x2, x3, x4]

    def get_layers_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x0 = self.relu(x)
        x = self.maxpool(x0)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.avgpool(x4)
        x = x5.view(x.size(0), -1)
        x = self.fc(x)
        all_layers = [x0, x1, x2, x3, x4, x5, x]
        return all_layers


class UNetDown(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, norm=True, dropout=0.0):
        super(UNetDown, self).__init__()
        norm_layer = nn.InstanceNorm2d if norm else nn.Identity
        self.model = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False), norm_layer(out_channels), nn.LeakyReLU(0.2), nn.Dropout(dropout))

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, norm=True, dropout=0.0):
        super(UNetUp, self).__init__()
        norm_layer = nn.InstanceNorm2d if norm else nn.Identity
        self.model = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False), norm_layer(out_channels), nn.ReLU(inplace=True), nn.Dropout(dropout))

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat([x, skip_input], dim=1)
        return x


class GeneratorUNet(nn.Module):

    def __init__(self, params):
        super(GeneratorUNet, self).__init__()
        self.params = params
        self.down_layers = nn.ModuleList()
        for i in range(len(self.params['down_channels']) - 1):
            self.down_layers.append(UNetDown(self.params['down_channels'][i], self.params['down_channels'][i + 1], kernel_size=self.params['kernel_size'], stride=self.params['stride'], padding=self.params['padding'], norm=self.params['down_norm'][i], dropout=self.params['down_dropout'][i]))
        self.up_layers = nn.ModuleList()
        for i in range(len(self.params['up_channels']) - 1):
            self.up_layers.append(UNetUp(self.params['up_channels'][i] + self.params['down_channels'][-i - 1], self.params['up_channels'][i + 1], kernel_size=self.params['kernel_size'], stride=self.params['stride'], padding=self.params['padding'], norm=self.params['up_norm'][i], dropout=self.params['up_dropout'][i]))
        self.final_layers = nn.Sequential(nn.Upsample(scale_factor=2), nn.ZeroPad2d((1, 0, 1, 0)), nn.Conv2d(2 * self.params['up_channels'][-1], self.params['final_channels'] * self.params['num_branches'], 4, padding=1), nn.Tanh())

    def forward(self, x, branch):
        d = []
        temp = x
        for down_layer in self.down_layers:
            temp = down_layer(temp)
            d.append(temp)
        for i, up_layer in enumerate(self.up_layers):
            temp = up_layer(temp, d[-i - 2])
        output = self.final_layers(temp)
        B, C, H, W = output.shape
        output = output.view(B, self.params['num_branches'], -1, H, W)
        batch_idx = torch.arange(0, B)
        return output[batch_idx.long(), branch.squeeze(1).long()]


class Discriminator(nn.Module):

    def __init__(self, params):
        super(Discriminator, self).__init__()
        self.params = params
        layers = nn.ModuleList()
        for i in range(len(self.params['channels']) - 1):
            layers.append(UNetDown(self.params['channels'][i], self.params['channels'][i + 1], kernel_size=self.params['kernel_size'], stride=self.params['stride'], padding=self.params['padding'], norm=self.params['norm'][i], dropout=self.params['dropout'][i]))
        self.model = nn.Sequential(*layers, nn.ZeroPad2d((1, 0, 1, 0)), nn.Conv2d(self.params['channels'][-1], 1, 4, padding=1, bias=False))

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        return self.model(x)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        try:
            nn.init.constant_(m.bias, 0.01)
        except:
            pass
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.constant_(m.bias, 0.01)


class CNN(nn.Module):

    def __init__(self, input_dim=1, out_dim=256):
        super(CNN, self).__init__()
        self.out_dim = out_dim
        self.conv1 = nn.Conv2d(input_dim, 64, 5, stride=3, padding=2)
        self.conv2 = nn.Conv2d(64, 128, 5, stride=4, padding=2)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, self.out_dim, 3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(self.out_dim)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.apply(weights_init)

    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv4(x)
        x = self.bn4(x)
        x = x.view(-1, self.out_dim)
        return x


class MLP_COS(nn.Module):

    def __init__(self, input_dim=25, out_dim=2):
        super(MLP_COS, self).__init__()
        self.rate = 1.0
        self.linear1 = nn.Linear(input_dim, 512)
        self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, 512)
        self.linear4 = nn.Linear(512, 256)
        self.linear5 = nn.Linear(256, out_dim)
        self.apply(weights_init)

    def forward(self, x, t, v0):
        B, C = x.shape
        T = t.shape[1]
        x = x.unsqueeze(1).expand(B, T, C)
        v0 = v0.unsqueeze(1).expand(B, T, 1)
        t = t.unsqueeze(-1)
        x = torch.cat([x, t, v0], dim=-1)
        x = self.linear1(x)
        x = torch.tanh(x)
        x = self.linear2(x)
        x = torch.tanh(x)
        x = self.linear3(x)
        x = torch.tanh(x)
        x = self.linear4(x)
        x = torch.cos(self.rate * x)
        x = self.linear5(x)
        return x


class ModelGRU(nn.Module):

    def __init__(self, params):
        super(ModelGRU, self).__init__()
        self.cnn_feature_dim = params['hidden_dim']
        self.rnn_hidden_dim = params['hidden_dim']
        self.cnn = CNN(input_dim=params['input_dim'], out_dim=self.cnn_feature_dim)
        self.gru = nn.GRU(input_size=self.cnn_feature_dim, hidden_size=self.rnn_hidden_dim, num_layers=3, batch_first=True, dropout=0.2)
        self.mlp = MLP_COS(input_dim=self.rnn_hidden_dim + 2, out_dim=params['out_dim'])

    def forward(self, x, t, v0):
        batch_size, timesteps, C, H, W = x.size()
        x = x.view(batch_size * timesteps, C, H, W)
        x = self.cnn(x)
        x = x.view(batch_size, timesteps, -1)
        x, h_n = self.gru(x)
        x = F.leaky_relu(x[:, -1, :])
        x = self.mlp(x, t, v0)
        return x


class TrexModel(nn.Module):

    def __init__(self, obs_shape):
        super(TrexModel, self).__init__()
        if isinstance(obs_shape, int) or len(obs_shape) == 1:
            self.encoder = nn.Sequential(FCEncoder(obs_shape, [512, 64]), nn.Linear(64, 1))
        elif len(obs_shape) == 3:
            self.encoder = ConvEncoder(obs_shape)
        else:
            raise KeyError('not support obs_shape for pre-defined encoder: {}, please customize your own Trex model'.format(obs_shape))

    def cum_return(self, traj: torch.Tensor, mode: str='sum') ->Tuple[torch.Tensor, torch.Tensor]:
        r = self.encoder(traj)
        if mode == 'sum':
            sum_rewards = torch.sum(r)
            sum_abs_rewards = torch.sum(torch.abs(r))
            return sum_rewards, sum_abs_rewards
        elif mode == 'batch':
            return r, torch.abs(r)
        else:
            raise KeyError('not support mode: {}, please choose mode=sum or mode=batch'.format(mode))

    def forward(self, traj_i: torch.Tensor, traj_j: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor]:
        """compute cumulative returns for each trajectory and return logits"""
        cum_r_i, abs_r_i = self.cum_return(traj_i)
        cum_r_j, abs_r_j = self.cum_return(traj_j)
        return torch.cat((cum_r_i.unsqueeze(0), cum_r_j.unsqueeze(0)), 0), abs_r_i + abs_r_j


def create_resnet_basic_block(width_output_feature_map, height_output_feature_map, nb_channel_in, nb_channel_out):
    basic_block = nn.Sequential(nn.Upsample(size=(width_output_feature_map, height_output_feature_map), mode='nearest'), nn.Conv2d(nb_channel_in, nb_channel_out, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), nn.BatchNorm2d(nb_channel_out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), nn.ReLU(inplace=True), nn.Conv2d(nb_channel_out, nb_channel_out, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), nn.BatchNorm2d(nb_channel_out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
    return basic_block


class ImplicitSupervisedModel(nn.Module):

    def __init__(self, nb_images_input, nb_images_output, hidden_size, nb_class_segmentation, nb_class_dist_to_tl, crop_sky=False):
        super().__init__()
        if crop_sky:
            self.size_state_RL = 6144
        else:
            self.size_state_RL = 8192
        resnet18 = models.resnet18(pretrained=False)
        assert resnet18.layer2[0].downsample[0].kernel_size == (1, 1)
        assert resnet18.layer3[0].downsample[0].kernel_size == (1, 1)
        assert resnet18.layer4[0].downsample[0].kernel_size == (1, 1)
        assert resnet18.layer2[0].downsample[0].stride == (2, 2)
        assert resnet18.layer3[0].downsample[0].stride == (2, 2)
        assert resnet18.layer4[0].downsample[0].stride == (2, 2)
        resnet18.layer2[0].downsample[0].kernel_size = 2, 2
        resnet18.layer3[0].downsample[0].kernel_size = 2, 2
        resnet18.layer4[0].downsample[0].kernel_size = 2, 2
        assert resnet18.layer2[0].downsample[0].kernel_size == (2, 2)
        assert resnet18.layer3[0].downsample[0].kernel_size == (2, 2)
        assert resnet18.layer4[0].downsample[0].kernel_size == (2, 2)
        assert resnet18.layer2[0].downsample[0].stride == (2, 2)
        assert resnet18.layer3[0].downsample[0].stride == (2, 2)
        assert resnet18.layer4[0].downsample[0].stride == (2, 2)
        new_conv1 = nn.Conv2d(nb_images_input * 3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        resnet18.conv1 = new_conv1
        self.encoder = torch.nn.Sequential(*list(resnet18.children())[:-2])
        self.last_conv_downsample = nn.Sequential(nn.Conv2d(512, 512, kernel_size=(2, 2), stride=(2, 2), bias=False), nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.fc1_traffic_light_inters = nn.Linear(self.size_state_RL, hidden_size)
        self.fc2_tl_inters_none = nn.Linear(hidden_size, 4)
        self.fc2_traffic_light_state = nn.Linear(hidden_size, 2)
        self.fc2_distance_to_tl = nn.Linear(hidden_size, nb_class_dist_to_tl)
        self.fc1_delta_y_yaw_camera = nn.Linear(self.size_state_RL, int(hidden_size / 4))
        self.fc2_delta_y_yaw_camera = nn.Linear(int(hidden_size / 4), 2 * nb_images_output)
        self.up_sampled_block_0 = create_resnet_basic_block(8, 8, 512, 512)
        self.up_sampled_block_1 = create_resnet_basic_block(16, 16, 512, 256)
        self.up_sampled_block_2 = create_resnet_basic_block(32, 32, 256, 128)
        self.up_sampled_block_3 = create_resnet_basic_block(64, 64, 128, 64)
        self.up_sampled_block_4 = create_resnet_basic_block(128, 128, 64, 32)
        self.last_conv_segmentation = nn.Conv2d(32, nb_class_segmentation * nb_images_output, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.last_bn = nn.BatchNorm2d(nb_class_segmentation * nb_images_output, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, batch_image):
        encoding = self.encoder(batch_image)
        encoding = self.last_conv_downsample(encoding)
        upsample0 = self.up_sampled_block_0(encoding)
        upsample1 = self.up_sampled_block_1(upsample0)
        upsample2 = self.up_sampled_block_2(upsample1)
        upsample3 = self.up_sampled_block_3(upsample2)
        upsample4 = self.up_sampled_block_4(upsample3)
        out_seg = self.last_bn(self.last_conv_segmentation(upsample4))
        classif_state_net = encoding.view(-1, self.size_state_RL)
        traffic_light_state_net = self.fc1_traffic_light_inters(classif_state_net)
        traffic_light_state_net = nn.functional.relu(traffic_light_state_net)
        classif_output = self.fc2_tl_inters_none(traffic_light_state_net)
        state_output = self.fc2_traffic_light_state(traffic_light_state_net)
        dist_to_tl_output = self.fc2_distance_to_tl(traffic_light_state_net)
        delta_position_yaw_state = self.fc1_delta_y_yaw_camera(classif_state_net)
        delta_position_yaw_state = nn.functional.relu(delta_position_yaw_state)
        delta_position_yaw_output = self.fc2_delta_y_yaw_camera(delta_position_yaw_state)
        return out_seg, classif_output, state_output, dist_to_tl_output, delta_position_yaw_output


class NoisyLinear(nn.Module):

    def __init__(self, in_features, out_features, std_init=0.1):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.FloatTensor(size).normal_()
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon = epsilon_out.ger(epsilon_in)
        self.bias_epsilon = epsilon_out

    def forward(self, input):
        if self.training:
            return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)


Orders = Enum('Order', 'Follow_Lane Straight Right Left ChangelaneLeft ChangelaneRight')


class ImplicitDQN(nn.Module):

    def __init__(self, action_space, history_length=4, quantile_embedding_dim=64, crop_sky=False, num_quantile_samples=32):
        super().__init__()
        self.action_space = action_space
        self.history_length = history_length
        self.magic_number_repeat_scaler_in_fc = 10
        self.magic_number_SCALE_steering_in_fc = 10
        self.quantile_embedding_dim = quantile_embedding_dim
        self.num_quantile_samples = num_quantile_samples
        if crop_sky:
            size_RL_state = 6144
        else:
            size_RL_state = 8192
        self.iqn_fc = nn.Linear(self.quantile_embedding_dim, size_RL_state)
        hidden_size = 1024
        self.fcnoisy_h_a = NoisyLinear(size_RL_state, hidden_size)
        hidden_size2 = 512
        self.fcnoisy0_z_a_lane_follow = NoisyLinear(hidden_size + 2 * self.magic_number_repeat_scaler_in_fc * self.history_length + 4 * self.magic_number_repeat_scaler_in_fc, hidden_size2)
        self.fcnoisy0_z_a_straight = NoisyLinear(hidden_size + 2 * self.magic_number_repeat_scaler_in_fc * self.history_length + 4 * self.magic_number_repeat_scaler_in_fc, hidden_size2)
        self.fcnoisy0_z_a_right = NoisyLinear(hidden_size + 2 * self.magic_number_repeat_scaler_in_fc * self.history_length + 4 * self.magic_number_repeat_scaler_in_fc, hidden_size2)
        self.fcnoisy0_z_a_left = NoisyLinear(hidden_size + 2 * self.magic_number_repeat_scaler_in_fc * self.history_length + 4 * self.magic_number_repeat_scaler_in_fc, hidden_size2)
        self.fcnoisy0_z_a_lane_right = NoisyLinear(hidden_size + 2 * self.magic_number_repeat_scaler_in_fc * self.history_length + 4 * self.magic_number_repeat_scaler_in_fc, hidden_size2)
        self.fcnoisy0_z_a_lane_left = NoisyLinear(hidden_size + 2 * self.magic_number_repeat_scaler_in_fc * self.history_length + 4 * self.magic_number_repeat_scaler_in_fc, hidden_size2)
        self.fcnoisy1_z_a_lane_follow = NoisyLinear(hidden_size2, action_space)
        self.fcnoisy1_z_a_lane_follow = NoisyLinear(hidden_size2, action_space)
        self.fcnoisy1_z_a_straight = NoisyLinear(hidden_size2, action_space)
        self.fcnoisy1_z_a_right = NoisyLinear(hidden_size2, action_space)
        self.fcnoisy1_z_a_left = NoisyLinear(hidden_size2, action_space)
        self.fcnoisy1_z_a_lane_right = NoisyLinear(hidden_size2, action_space)
        self.fcnoisy1_z_a_lane_left = NoisyLinear(hidden_size2, action_space)

    def forward(self, observations):
        if 'obs' in observations:
            observations = observations['obs']
        speeds = observations['speed']
        steerings = observations['steer']
        images = observations['image']
        targets = observations['targets']
        orders = observations['order']
        num_quantiles = self.num_quantile_samples
        batch_size = images.shape[0]
        quantiles = torch.FloatTensor(num_quantiles * batch_size, 1).uniform_(0, 1)
        quantile_net = quantiles.repeat([1, self.quantile_embedding_dim])
        quantile_net = torch.cos(torch.arange(1, self.quantile_embedding_dim + 1, 1, device=torch.device('cuda'), dtype=torch.float32) * math.pi * quantile_net)
        quantile_net = self.iqn_fc(quantile_net)
        quantile_net = F.relu(quantile_net)
        rl_state_net = images.repeat(num_quantiles, 1)
        rl_state_net = rl_state_net * quantile_net
        mask_lane_follow = orders == Orders.Follow_Lane.value
        mask_straight = orders == Orders.Straight.value
        mask_right = orders == Orders.Right.value
        mask_left = orders == Orders.Left.value
        mask_lane_right = orders == Orders.ChangelaneRight.value
        mask_lane_left = orders == Orders.ChangelaneLeft.value
        if batch_size != 1:
            mask_lane_follow = mask_lane_follow
            mask_straight = mask_straight
            mask_right = mask_right
            mask_left = mask_left
            mask_lane_right = mask_lane_right
            mask_lane_left = mask_lane_left
            mask_lane_follow = mask_lane_follow.float()[:, None].repeat(num_quantiles, 1)
            mask_straight = mask_straight.float()[:, None].repeat(num_quantiles, 1)
            mask_right = mask_right.float()[:, None].repeat(num_quantiles, 1)
            mask_left = mask_left.float()[:, None].repeat(num_quantiles, 1)
            mask_lane_right = mask_lane_right.float()[:, None].repeat(num_quantiles, 1)
            mask_lane_left = mask_lane_left.float()[:, None].repeat(num_quantiles, 1)
        else:
            mask_lane_follow = bool(mask_lane_follow)
            mask_straight = bool(mask_straight)
            mask_right = bool(mask_right)
            mask_left = bool(mask_left)
            mask_lane_right = bool(mask_lane_right)
            mask_lane_left = bool(mask_lane_left)
        just_before_order_heads_a = F.relu(self.fcnoisy_h_a(rl_state_net))
        steerings = steerings * self.magic_number_SCALE_steering_in_fc
        speeds = speeds.repeat(num_quantiles, self.magic_number_repeat_scaler_in_fc)
        steerings = steerings.repeat(num_quantiles, self.magic_number_repeat_scaler_in_fc)
        targets = targets.repeat(num_quantiles, self.magic_number_repeat_scaler_in_fc)
        just_before_order_heads_a_plus_speed_steering = torch.cat((just_before_order_heads_a, speeds, steerings, targets), 1)
        a_lane_follow = self.fcnoisy0_z_a_lane_follow(just_before_order_heads_a_plus_speed_steering)
        a_lane_follow = self.fcnoisy1_z_a_lane_follow(F.relu(a_lane_follow))
        a_straight = self.fcnoisy0_z_a_straight(just_before_order_heads_a_plus_speed_steering)
        a_straight = self.fcnoisy1_z_a_straight(F.relu(a_straight))
        a_right = self.fcnoisy0_z_a_right(just_before_order_heads_a_plus_speed_steering)
        a_right = self.fcnoisy1_z_a_right(F.relu(a_right))
        a_left = self.fcnoisy0_z_a_left(just_before_order_heads_a_plus_speed_steering)
        a_left = self.fcnoisy1_z_a_left(F.relu(a_left))
        a_lane_right = self.fcnoisy0_z_a_lane_right(just_before_order_heads_a_plus_speed_steering)
        a_lane_right = self.fcnoisy1_z_a_lane_right(F.relu(a_lane_right))
        a_lane_left = self.fcnoisy0_z_a_lane_left(just_before_order_heads_a_plus_speed_steering)
        a_lane_left = self.fcnoisy1_z_a_lane_left(F.relu(a_lane_left))
        a = a_lane_follow * mask_lane_follow + a_straight * mask_straight + a_right * mask_right + a_left * mask_left + a_lane_right * mask_lane_right + a_lane_left * mask_lane_left
        a = a.view(num_quantiles, batch_size, -1)
        a = a.mean(0)
        return {'logit': a}

    def reset_noise(self):
        for name, module in self.named_children():
            if 'fcnoisy' in name:
                module.reset_noise()


class Loss(torch.nn.Module):
    """
    Loss function for implicit affordances
    """

    def __init__(self, weights=[1.0, 1.0, 10.0, 1.0, 1.0]):
        super(Loss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.l2_loss = nn.MSELoss()
        self.weights = weights

    def forward(self, preds, gts):
        pred_seg, pred_is_junction, pred_tl_state, pred_tl_dis, pred_delta_yaw = preds
        gt_seg, gt_is_junction, gt_tl_state, gt_tl_dis, gt_delta_yaw = gts
        loss_seg = self.l2_loss(pred_seg, gt_seg) * self.weights[0]
        loss_is_junction = self.ce_loss(pred_is_junction, gt_is_junction) * self.weights[1]
        loss_tl_state = self.ce_loss(pred_tl_state, gt_tl_state) * self.weights[2]
        loss_tl_dis = self.ce_loss(pred_tl_dis, gt_tl_dis) * self.weights[3]
        loss_delta_yaw = self.l2_loss(pred_delta_yaw, gt_delta_yaw) * self.weights[4]
        loss = loss_seg + loss_is_junction + loss_tl_state + loss_tl_dis + loss_delta_yaw
        return loss, (loss.item(), loss_seg.item(), loss_is_junction.item(), loss_tl_state.item(), loss_tl_dis.item(), loss_delta_yaw.item())


class LatentDQNRLModel(nn.Module):

    def __init__(self, obs_shape: List=[192, 192, 7], action_shape: int=100, latent_dim: int=128, dueling: bool=True, head_hidden_size: Optional[int]=None, head_layer_num: int=1, activation: Optional[nn.Module]=nn.ReLU(), norm_type: Optional[str]=None, vae_path: Optional[str]=None) ->None:
        super().__init__()
        in_channels = obs_shape[-1]
        self._vae_model = VanillaVAE(in_channels=in_channels, latent_dim=latent_dim)
        if vae_path is not None:
            state_dict = torch.load(vae_path)
            self._vae_model.load_state_dict(state_dict)
        if head_hidden_size is None:
            head_hidden_size = latent_dim + 12
        if dueling:
            head_cls = DuelingHead
        else:
            head_cls = DiscreteHead
        multi_head = not isinstance(action_shape, int)
        if multi_head:
            self.head = MultiHead(head_cls, head_hidden_size, action_shape, layer_num=head_layer_num, activation=activation, norm_type=norm_type)
        else:
            self.head = head_cls(head_hidden_size, action_shape, head_layer_num, activation=activation, norm_type=norm_type)

    def forward(self, data: Dict) ->Dict:
        bev = data['birdview'].permute(0, 3, 1, 2)
        ego_info = data['ego_info']
        with torch.no_grad():
            mu, log_var = self._vae_model.encode(bev)
            feat = self._vae_model.reparameterize(mu, log_var)
        x = torch.cat([feat, ego_info], dim=1)
        x = self.head(x)
        return x


class DQNRLModel(nn.Module):

    def __init__(self, obs_shape: Tuple=[5, 32, 32], action_shape: Union[int, Tuple]=21, encoder_hidden_size_list: Tuple=[64, 128, 256], dueling: bool=True, head_hidden_size: Optional[int]=512, head_layer_num: int=1, activation: Optional[nn.Module]=nn.ReLU(), norm_type: Optional[str]=None) ->None:
        super().__init__()
        self._encoder = BEVSpeedConvEncoder(obs_shape, encoder_hidden_size_list, head_hidden_size, [3, 3, 3], [2, 2, 2])
        if dueling:
            head_cls = DuelingHead
        else:
            head_cls = DiscreteHead
        multi_head = not isinstance(action_shape, int)
        if multi_head:
            self._head = MultiHead(head_cls, head_hidden_size, action_shape, layer_num=head_layer_num, activation=activation, norm_type=norm_type)
        else:
            self._head = head_cls(head_hidden_size, action_shape, head_layer_num, activation=activation, norm_type=norm_type)

    def forward(self, obs):
        x = self._encoder(obs)
        y = self._head(x)
        return y


class DDPGRLModel(nn.Module):

    def __init__(self, obs_shape: Tuple=[5, 32, 32], action_shape: Union[int, tuple]=2, share_encoder: bool=False, encoder_hidden_size_list: List=[64, 128, 256], encoder_embedding_size: int=512, twin_critic: bool=False, actor_head_hidden_size: int=512, actor_head_layer_num: int=1, critic_head_hidden_size: int=512, critic_head_layer_num: int=1, activation: Optional[nn.Module]=nn.ReLU(), norm_type: Optional[str]=None) ->None:
        super().__init__()
        self._obs_shape = obs_shape
        self._act_shape = action_shape
        self.twin_critic = twin_critic
        self.share_encoder = share_encoder
        if self.share_encoder:
            self.actor_encoder = self.critic_encoder = BEVSpeedConvEncoder(self._obs_shape, encoder_hidden_size_list, encoder_embedding_size, [3, 3, 3], [2, 2, 2])
        else:
            self.actor_encoder = BEVSpeedConvEncoder(self._obs_shape, encoder_hidden_size_list, encoder_embedding_size, [3, 3, 3], [2, 2, 2])
            self.critic_encoder = BEVSpeedConvEncoder(self._obs_shape, encoder_hidden_size_list, encoder_embedding_size, [3, 3, 3], [2, 2, 2])
        self.actor_head = nn.Sequential(nn.Linear(encoder_embedding_size, actor_head_hidden_size), activation, RegressionHead(actor_head_hidden_size, action_shape, actor_head_layer_num, final_tanh=True, activation=activation, norm_type=norm_type))
        self.twin_critic = twin_critic
        if self.twin_critic:
            if not self.share_encoder:
                self._twin_encoder = BEVSpeedConvEncoder(self._obs_shape, encoder_hidden_size_list, encoder_embedding_size, [3, 3, 3], [2, 2, 2])
            else:
                self._twin_encoder = self.actor_encoder
            self.critic_head = [nn.Sequential(nn.Linear(encoder_embedding_size + self._act_shape, critic_head_hidden_size), activation, RegressionHead(critic_head_hidden_size, 1, critic_head_layer_num, final_tanh=False, activation=activation, norm_type=norm_type)) for _ in range(2)]
        else:
            self.critic_head = nn.Sequential(nn.Linear(encoder_embedding_size + self._act_shape, critic_head_hidden_size), activation, RegressionHead(critic_head_hidden_size, 1, critic_head_layer_num, final_tanh=False, activation=activation, norm_type=norm_type))
        self.actor = nn.ModuleList([self.actor_encoder, self.actor_head])
        if self.twin_critic:
            self.critic = nn.ModuleList([self.critic_encoder, *self.critic_head, self._twin_encoder])
        else:
            self.critic = nn.ModuleList([self.critic_encoder, self.critic_head])

    def forward(self, inputs, mode=None, **kwargs):
        assert mode in ['compute_actor_critic', 'compute_actor', 'compute_critic']
        f = getattr(self, mode)
        return f(inputs, **kwargs)

    def compute_critic(self, inputs: Dict) ->Dict:
        x0 = self.critic_encoder(inputs['obs'])
        x0 = torch.cat([x0, inputs['action']], dim=1)
        if self.twin_critic:
            x1 = self._twin_encoder(inputs['obs'])
            x1 = torch.cat([x1, inputs['action']], dim=1)
            x = [m(xi)['pred'] for m, xi in [(self.critic_head[0], x0), (self.critic_head[1], x1)]]
        else:
            x = self.critic_head(x0)['pred']
        return {'q_value': x}

    def compute_actor(self, inputs: Dict) ->Dict:
        x = self.actor_encoder(inputs)
        action = self.actor_head(x)['pred']
        return {'action': action}


class TD3RLModel(DDPGRLModel):

    def __init__(self, obs_shape: Tuple=[5, 32, 32], action_shape: Union[int, tuple]=2, share_encoder: bool=False, encoder_hidden_size_list: List=[64, 128, 256], encoder_embedding_size: int=512, twin_critic: bool=True, actor_head_hidden_size: int=512, actor_head_layer_num: int=1, critic_head_hidden_size: int=512, critic_head_layer_num: int=1, activation: Optional[nn.Module]=nn.ReLU(), norm_type: Optional[str]=None) ->None:
        super().__init__(obs_shape, action_shape, share_encoder, encoder_hidden_size_list, encoder_embedding_size, twin_critic, actor_head_hidden_size, actor_head_layer_num, critic_head_hidden_size, critic_head_layer_num, activation, norm_type)
        assert twin_critic


class SACRLModel(nn.Module):

    def __init__(self, obs_shape: Tuple=[5, 32, 32], action_shape: Union[int, tuple]=2, share_encoder: bool=False, encoder_hidden_size_list: List=[64, 128, 256], encoder_embedding_size: int=512, twin_critic: bool=False, actor_head_hidden_size: int=512, actor_head_layer_num: int=1, critic_head_hidden_size: int=512, critic_head_layer_num: int=1, activation: Optional[nn.Module]=nn.ReLU(), norm_type: Optional[str]=None, **kwargs) ->None:
        super().__init__()
        self._act = nn.ReLU()
        self._obs_shape = obs_shape
        self._act_shape = action_shape
        self.twin_critic = twin_critic
        self.share_encoder = share_encoder
        if self.share_encoder:
            self.actor_encoder = self.critic_encoder = BEVSpeedConvEncoder(self._obs_shape, encoder_hidden_size_list, encoder_embedding_size, [3, 3, 3], [2, 2, 2])
        else:
            self.actor_encoder = BEVSpeedConvEncoder(self._obs_shape, encoder_hidden_size_list, encoder_embedding_size, [3, 3, 3], [2, 2, 2])
            self.critic_encoder = BEVSpeedConvEncoder(self._obs_shape, encoder_hidden_size_list, encoder_embedding_size, [3, 3, 3], [2, 2, 2])
        self.actor = nn.Sequential(nn.Linear(encoder_embedding_size, actor_head_hidden_size), activation, ReparameterizationHead(actor_head_hidden_size, action_shape, actor_head_layer_num, sigma_type='conditioned', activation=activation, norm_type=norm_type))
        self.twin_critic = twin_critic
        if self.twin_critic:
            if self.share_encoder:
                self._twin_encoder = self.actor_encoder
            else:
                self._twin_encoder = BEVSpeedConvEncoder(self._obs_shape, encoder_hidden_size_list, encoder_embedding_size, [3, 3, 3], [2, 2, 2])
            self.critic = nn.ModuleList()
            for _ in range(2):
                self.critic.append(nn.Sequential(nn.Linear(encoder_embedding_size + self._act_shape, critic_head_hidden_size), activation, RegressionHead(critic_head_hidden_size, 1, critic_head_layer_num, final_tanh=False, activation=activation, norm_type=norm_type)))
        else:
            self.critic = nn.Sequential(nn.Linear(encoder_embedding_size + self._act_shape, critic_head_hidden_size), activation, RegressionHead(critic_head_hidden_size, 1, critic_head_layer_num, final_tanh=False, activation=activation, norm_type=norm_type))

    def forward(self, inputs, mode=None, **kwargs):
        self.mode = ['compute_actor', 'compute_critic']
        assert mode in self.mode, 'not support forward mode: {}/{}'.format(mode, self.mode)
        return getattr(self, mode)(inputs)

    def compute_critic(self, inputs: Dict[str, torch.Tensor]) ->Dict[str, torch.Tensor]:
        x0 = self.critic_encoder(inputs['obs'])
        x0 = torch.cat([x0, inputs['action']], dim=1)
        if self.twin_critic:
            x1 = self._twin_encoder(inputs['obs'])
            x1 = torch.cat([x1, inputs['action']], dim=1)
            x = [m(xi)['pred'] for m, xi in [(self.critic[0], x0), (self.critic[1], x1)]]
        else:
            x = self.critic(x0)['pred']
        return {'q_value': x}

    def compute_actor(self, inputs) ->Dict[str, torch.Tensor]:
        x = self.actor_encoder(inputs)
        x = self.actor(x)
        return {'logit': [x['mu'], x['sigma']]}


class PPORLModel(nn.Module):

    def __init__(self, obs_shape: Tuple=[5, 32, 32], action_shape: Union[int, Tuple]=2, action_space: str='continuous', share_encoder: bool=True, encoder_embedding_size: int=512, encoder_hidden_size_list: List=[64, 128, 256], actor_head_hidden_size: int=512, actor_head_layer_num: int=1, critic_head_hidden_size: int=512, critic_head_layer_num: int=1, activation: Optional[nn.Module]=nn.ReLU(), norm_type: Optional[str]=None, sigma_type: Optional[str]='independent', bound_type: Optional[str]=None) ->None:
        super().__init__()
        self._obs_shape = obs_shape
        self._act_shape = action_shape
        self.share_encoder = share_encoder
        if self.share_encoder:
            self.encoder = BEVSpeedConvEncoder(self._obs_shape, encoder_hidden_size_list, encoder_embedding_size, [3, 3, 3], [2, 2, 2])
        else:
            self.actor_encoder = BEVSpeedConvEncoder(self._obs_shape, encoder_hidden_size_list, encoder_embedding_size, [3, 3, 3], [2, 2, 2])
            self.critic_encoder = BEVSpeedConvEncoder(self._obs_shape, encoder_hidden_size_list, encoder_embedding_size, [3, 3, 3], [2, 2, 2])
        self.critic_head = RegressionHead(critic_head_hidden_size, 1, critic_head_layer_num, activation=activation, norm_type=norm_type)
        self.action_space = action_space
        assert self.action_space in ['discrete', 'continuous', 'hybrid'], self.action_space
        if self.action_space == 'continuous':
            self.multi_head = False
            self.actor_head = ReparameterizationHead(actor_head_hidden_size, action_shape, actor_head_layer_num, sigma_type=sigma_type, activation=activation, norm_type=norm_type, bound_type=bound_type)
        elif self.action_space == 'discrete':
            multi_head = not isinstance(action_shape, int)
            self.multi_head = multi_head
            if multi_head:
                self.actor_head = MultiHead(DiscreteHead, actor_head_hidden_size, action_shape, layer_num=actor_head_layer_num, activation=activation, norm_type=norm_type)
            else:
                self.actor_head = DiscreteHead(actor_head_hidden_size, action_shape, actor_head_layer_num, activation=activation, norm_type=norm_type)
        if self.share_encoder:
            self.actor = nn.ModuleList([self.encoder, self.actor_head])
            self.critic = nn.ModuleList([self.encoder, self.critic_head])
        else:
            self.actor = nn.ModuleList([self.actor_encoder, self.actor_head])
            self.critic = nn.ModuleList([self.critic_encoder, self.critic_head])

    def forward(self, inputs, mode=None, **kwargs):
        assert mode in ['compute_actor_critic', 'compute_actor', 'compute_critic']
        f = getattr(self, mode)
        return f(inputs, **kwargs)

    def compute_actor_critic(self, inputs) ->Dict[str, torch.Tensor]:
        if self.share_encoder:
            actor_embedding = critic_embedding = self.encoder(inputs)
        else:
            actor_embedding = self.actor_encoder(inputs)
            critic_embedding = self.critic_encoder(inputs)
        value = self.critic_head(critic_embedding)['pred']
        if self.action_space == 'continuous':
            logit = self.actor_head(actor_embedding)
            return {'logit': logit, 'value': value}
        elif self.action_space == 'discrete':
            logit = self.actor_head(actor_embedding)['logit']
        return {'logit': logit, 'value': value}

    def compute_actor(self, inputs: Dict) ->Dict:
        if self.share_encoder:
            x = self.encoder(inputs)
        else:
            x = self.actor_encoder(inputs)
        if self.action_space == 'discrete':
            return self.actor_head(x)
        elif self.action_space == 'continuous':
            x = self.actor_head(x)
            return {'logit': x}
        elif self.action_space == 'hybrid':
            action_type = self.actor_head[0](x)
            action_args = self.actor_head[1](x)
            return {'logit': {'action_type': action_type['logit'], 'action_args': action_args}}

    def compute_critic(self, inputs: Dict) ->Dict:
        if self.share_encoder:
            x = self.encoder(inputs)
        else:
            x = self.critic_encoder(inputs)
        x = self.critic_head(x)
        return {'value': x['pred']}


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CNN,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 128, 128])], {}),
     True),
    (ImplicitSupervisedModel,
     lambda: ([], {'nb_images_input': 4, 'nb_images_output': 4, 'hidden_size': 4, 'nb_class_segmentation': 4, 'nb_class_dist_to_tl': 4}),
     lambda: ([torch.rand([4, 12, 128, 128])], {}),
     True),
    (LocationLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Loss,
     lambda: ([], {}),
     lambda: ([(torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])), (torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]))], {}),
     False),
    (NoisyLinear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SpatialSoftmax,
     lambda: ([], {'height': 4, 'width': 4, 'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (UNetDown,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_opendilab_DI_drive(_paritybench_base):
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

