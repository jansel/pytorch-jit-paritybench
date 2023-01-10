import sys
_module = sys.modules[__name__]
del sys
cilrs_agent = _module
cilrs_wrapper = _module
cilrs_model = _module
branching = _module
fc = _module
join = _module
resnet = _module
trainer = _module
augmenter = _module
branched_loss = _module
dataset = _module
lbc_roaming_agent = _module
controller = _module
local_planner = _module
distributions = _module
ppo = _module
ppo_buffer = _module
ppo_policy = _module
torch_layers = _module
torch_util = _module
rl_birdview_agent = _module
rl_birdview_wrapper = _module
wandb_callback = _module
benchmark = _module
carla_gym = _module
carla_multi_agent_env = _module
control = _module
route = _module
speed = _module
velocity = _module
chauffeurnet = _module
rgb = _module
gnss = _module
waypoint_plan = _module
ego = _module
pedestrian = _module
stop_sign = _module
traffic_light_new = _module
vehicle = _module
obs_manager = _module
obs_manager_handler = _module
blocked = _module
collision = _module
encounter_light = _module
outside_route_lane = _module
route_deviation = _module
run_red_light = _module
run_stop_sign = _module
global_route_planner = _module
map_utils = _module
route_manipulation = _module
task_vehicle = _module
ego_vehicle_handler = _module
valeo_action = _module
leaderboard = _module
leaderboard_dagger = _module
valeo = _module
valeo_no_det_px = _module
basic_agent = _module
constant_speed_agent = _module
misc = _module
scenario_actor_handler = _module
zombie_vehicle = _module
zombie_vehicle_handler = _module
zombie_walker = _module
zombie_walker_handler = _module
envs = _module
corl2017_env = _module
endless_env = _module
leaderboard_env = _module
nocrash_env = _module
birdview_map = _module
config_utils = _module
dynamic_weather = _module
expert_noiser = _module
gps_utils = _module
hazard_actor = _module
traffic_light = _module
transforms = _module
data_collect = _module
train_il = _module
train_rl = _module
saving_utils = _module
server_utils = _module

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


from torchvision import transforms as T


import torch as th


import logging


import torch.nn as nn


import copy


from collections import OrderedDict


import torch


import torch.utils.model_zoo as model_zoo


import torchvision.models as models


import torch.optim as optim


import time


from torch.nn import functional as F


from torch.distributions import Beta


from torch.distributions import Normal


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from typing import Optional


from typing import Tuple


from collections import deque


from typing import Generator


from typing import NamedTuple


from typing import Dict


from typing import List


import queue


from typing import Union


from typing import Any


from functools import partial


from torch import nn


import math


import torch.nn.functional as F


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

    def load_network(self, checkpoint):
        """
        Load a network for a given model definition .

        Args:
            checkpoint: The checkpoint that the user wants to add .



        """
        coil_logger.add_message('Loading', {'Model': {'Loaded checkpoint: ' + str(checkpoint)}})


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
            if i < len(params['neurons']) - 2:
                self.layers.append(nn.Sequential(*[fc, dropout, relu]))
            elif params['end_layer'] is None:
                self.layers.append(nn.Sequential(*[fc, dropout]))
            else:
                self.layers.append(nn.Sequential(*[fc, dropout, params['end_layer']()]))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        if type(x) is tuple:
            return self.layers(x[0]), x[1]
        else:
            return self.layers(x)


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

    def load_network(self, checkpoint):
        """
        Load a network for a given model definition .

        Args:
            checkpoint: The checkpoint that the user wants to add .



        """
        coil_logger.add_message('Loading', {'Model': {'Loaded checkpoint: ' + str(checkpoint)}})


log = logging.getLogger(__name__)


class CoILICRA(nn.Module):

    def __init__(self, im_shape, input_states, acc_as_action, value_as_supervision, action_distribution, dim_features_supervision, rl_ckpt=None, freeze_value_head=False, freeze_action_head=False, resnet_pretrain=True, perception_output_neurons=512, measurements_neurons=[128, 128], measurements_dropouts=[0.0, 0.0], join_neurons=[512], join_dropouts=[0.0], speed_branch_neurons=[256, 256], speed_branch_dropouts=[0.0, 0.5], value_branch_neurons=[256, 256], value_branch_dropouts=[0.0, 0.5], number_of_branches=6, branches_neurons=[256, 256], branches_dropouts=[0.0, 0.5], squash_outputs=True, perception_net='resnet34'):
        super(CoILICRA, self).__init__()
        self._init_kwargs = copy.deepcopy(locals())
        del self._init_kwargs['self']
        del self._init_kwargs['__class__']
        if th.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.number_of_branches = number_of_branches
        rl_state_dict = None
        if rl_ckpt is not None:
            try:
                rl_state_dict = th.load(rl_ckpt, map_location='cpu')['policy_state_dict']
                log.info(f'Load rl_ckpt: {rl_ckpt}')
            except:
                log.info(f'Unable to load rl_ckpt: {rl_ckpt}')
        self.dim_features_supervision = dim_features_supervision
        if dim_features_supervision > 0:
            join_neurons[-1] = dim_features_supervision
        self._video_input = 'video' in perception_net
        self.perception = resnet.get_model(perception_net, im_shape, num_classes=perception_output_neurons, pretrained=resnet_pretrain)
        input_states_len = 0
        if 'speed' in input_states:
            input_states_len += 1
        if 'vec' in input_states:
            input_states_len += 2
        if 'cmd' in input_states:
            input_states_len += 6
        self.measurements = FC(params={'neurons': [input_states_len] + measurements_neurons, 'dropouts': measurements_dropouts, 'end_layer': nn.ReLU})
        self.join = Join(params={'after_process': FC(params={'neurons': [measurements_neurons[-1] + perception_output_neurons] + join_neurons, 'dropouts': join_dropouts, 'end_layer': nn.ReLU}), 'mode': 'cat'})
        if squash_outputs:
            end_layer_speed = nn.Sigmoid
            end_layer_action = nn.Tanh
        else:
            end_layer_speed = None
            end_layer_action = None
        self.speed_branch = FC(params={'neurons': [perception_output_neurons] + speed_branch_neurons + [1], 'dropouts': speed_branch_dropouts + [0.0], 'end_layer': end_layer_speed})
        self.value_as_supervision = value_as_supervision
        if value_as_supervision:
            self.value_branch = FC(params={'neurons': [join_neurons[-1]] + value_branch_neurons + [1], 'dropouts': value_branch_dropouts + [0.0], 'end_layer': None})
            if rl_state_dict is not None:
                self._load_state_dict(self.value_branch, rl_state_dict, 'value_head')
                log.info(f'Load rl_ckpt state dict for value_head.')
            if freeze_value_head:
                for param in self.value_branch.parameters():
                    param.requires_grad = False
                log.info('Freeze value head weights.')
        self.action_distribution = action_distribution
        assert action_distribution in ['beta', 'beta_shared', None]
        if action_distribution == 'beta':
            dim_out = 2
            mu_branch_vector = []
            sigma_branch_vector = []
            for i in range(number_of_branches):
                mu_branch_vector.append(FC(params={'neurons': [join_neurons[-1]] + branches_neurons + [dim_out], 'dropouts': branches_dropouts + [0.0], 'end_layer': nn.Softplus}))
                sigma_branch_vector.append(FC(params={'neurons': [join_neurons[-1]] + branches_neurons + [dim_out], 'dropouts': branches_dropouts + [0.0], 'end_layer': nn.Softplus}))
            self.mu_branches = Branching(mu_branch_vector)
            self.sigma_branches = Branching(sigma_branch_vector)
        elif action_distribution == 'beta_shared':
            dim_out = 2
            mu_branch_vector = []
            sigma_branch_vector = []
            for i in range(number_of_branches):
                policy_head = FC(params={'neurons': [join_neurons[-1]] + branches_neurons, 'dropouts': branches_dropouts, 'end_layer': nn.ReLU})
                dist_mu = nn.Sequential(nn.Linear(branches_neurons[-1], dim_out), nn.Softplus())
                dist_sigma = nn.Sequential(nn.Linear(branches_neurons[-1], dim_out), nn.Softplus())
                if rl_state_dict is not None:
                    self._load_state_dict(policy_head, rl_state_dict, 'policy_head')
                    self._load_state_dict(dist_mu, rl_state_dict, 'dist_mu')
                    self._load_state_dict(dist_sigma, rl_state_dict, 'dist_sigma')
                    log.info(f'Load rl_ckpt state dict for policy_head, dist_mu, dist_sigma.')
                mu_branch_vector.append(nn.Sequential(policy_head, dist_mu))
                sigma_branch_vector.append(nn.Sequential(policy_head, dist_sigma))
            self.mu_branches = Branching(mu_branch_vector)
            self.sigma_branches = Branching(sigma_branch_vector)
            if freeze_action_head:
                for param in self.mu_branches.parameters():
                    param.requires_grad = False
                for param in self.sigma_branches.parameters():
                    param.requires_grad = False
                log.info('Freeze action head weights.')
        else:
            if acc_as_action:
                dim_out = 2
            else:
                dim_out = 3
            branch_fc_vector = []
            for i in range(number_of_branches):
                branch_fc_vector.append(FC(params={'neurons': [join_neurons[-1]] + branches_neurons + [dim_out], 'dropouts': branches_dropouts + [0.0], 'end_layer': end_layer_action}))
            self.branches = Branching(branch_fc_vector)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.1)

    def forward(self, im, state):
        """
        im: (b, c, t, h, w) np.uint8
        state: (n,)
        """
        if not self._video_input:
            b, c, t, h, w = im.shape
            im = im.view(b, c * t, h, w)
        """ ###### APPLY THE PERCEPTION MODULE """
        x = self.perception(im)
        """ ###### APPLY THE MEASUREMENT MODULE """
        m = self.measurements(state)
        """ Join measurements and perception"""
        j = self.join(x, m)
        outputs = {'pred_speed': self.speed_branch(x)}
        if self.value_as_supervision:
            outputs['pred_value'] = self.value_branch(j)
        if self.action_distribution is None:
            outputs['action_branches'] = self.branches(j)
        else:
            outputs['mu_branches'] = self.mu_branches(j)
            outputs['sigma_branches'] = self.sigma_branches(j)
        if self.dim_features_supervision > 0:
            outputs['pred_features'] = j
        return outputs

    def forward_branch(self, command, im, state):
        with th.no_grad():
            im_tensor = im.unsqueeze(0)
            state_tensor = state.unsqueeze(0)
            command_tensor = command
            command_tensor.clamp_(0, self.number_of_branches - 1)
            outputs = self.forward(im_tensor, state_tensor)
            if self.action_distribution == 'beta' or self.action_distribution == 'beta_shared':
                action_branches = self._get_action_beta(outputs['mu_branches'], outputs['sigma_branches'])
                action = self.extract_branch(action_branches, command_tensor)
            else:
                action = self.extract_branch(outputs['action_branches'], command_tensor)
        return action[0].cpu().numpy(), outputs['pred_speed'].item()

    @staticmethod
    def extract_branch(action_branches, branch_number):
        """
        action_branches: list, len=num_branches, (batch_size, action_dim)
        """
        output_vec = th.stack(action_branches)
        if len(branch_number) > 1:
            branch_number = th.squeeze(branch_number.type(th.LongTensor))
        else:
            branch_number = branch_number.type(th.LongTensor)
        branch_number = th.stack([branch_number, th.LongTensor(range(0, len(branch_number)))])
        return output_vec[branch_number[0], branch_number[1], :]

    @property
    def init_kwargs(self):
        return self._init_kwargs

    @classmethod
    def load(cls, path):
        if th.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        saved_variables = th.load(path, map_location=device)
        saved_variables['policy_init_kwargs']['resnet_pretrain'] = False
        saved_variables['policy_init_kwargs']['rl_ckpt'] = None
        model = cls(**saved_variables['policy_init_kwargs'])
        log.info(f'load state dict : {path}')
        model.load_state_dict(saved_variables['policy_state_dict'])
        model
        return model

    @staticmethod
    def _get_action_beta(alpha_branches, beta_branches):
        action_branches = []
        for alpha, beta in zip(alpha_branches, beta_branches):
            x = th.zeros_like(alpha)
            x[:, 1] += 0.5
            mask1 = (alpha > 1) & (beta > 1)
            x[mask1] = (alpha[mask1] - 1) / (alpha[mask1] + beta[mask1] - 2)
            mask2 = (alpha <= 1) & (beta > 1)
            x[mask2] = 0.0
            mask3 = (alpha > 1) & (beta <= 1)
            x[mask3] = 1.0
            mask4 = (alpha <= 1) & (beta <= 1)
            x[mask4] = alpha[mask4] / (alpha[mask4] + beta[mask4])
            x = x * 2 - 1
            action_branches.append(x)
        return action_branches

    @staticmethod
    def _load_state_dict(il_net, rl_state_dict, key_word):
        rl_keys = [k for k in rl_state_dict.keys() if key_word in k]
        il_keys = il_net.state_dict().keys()
        assert len(rl_keys) == len(il_net.state_dict().keys()), f'mismatch number of layers loading {key_word}'
        new_state_dict = OrderedDict()
        for k_il, k_rl in zip(il_keys, rl_keys):
            new_state_dict[k_il] = rl_state_dict[k_rl]
        il_net.load_state_dict(new_state_dict)


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
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, input_shape, num_classes=1000):
        im_channels, im_h, im_w = input_shape
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(im_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(2, stride=0)
        with th.no_grad():
            x = th.zeros(1, im_channels, im_h, im_w)
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
            n_flatten = x.shape[1]
        self.fc = nn.Linear(n_flatten, num_classes)
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
        return x

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


def load_entry_point(name):
    mod_name, attr_name = name.split(':')
    mod = import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn


class XtMaCNN(nn.Module):
    """
    Inspired by https://github.com/xtma/pytorch_car_caring
    """

    def __init__(self, observation_space, features_dim=256, states_neurons=[256]):
        super().__init__()
        self.features_dim = features_dim
        n_input_channels = observation_space['birdview'].shape[0]
        self.cnn = nn.Sequential(nn.Conv2d(n_input_channels, 8, kernel_size=5, stride=2), nn.ReLU(), nn.Conv2d(8, 16, kernel_size=5, stride=2), nn.ReLU(), nn.Conv2d(16, 32, kernel_size=5, stride=2), nn.ReLU(), nn.Conv2d(32, 64, kernel_size=3, stride=2), nn.ReLU(), nn.Conv2d(64, 128, kernel_size=3, stride=2), nn.ReLU(), nn.Conv2d(128, 256, kernel_size=3, stride=1), nn.ReLU(), nn.Flatten())
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space['birdview'].sample()[None]).float()).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten + states_neurons[-1], 512), nn.ReLU(), nn.Linear(512, features_dim), nn.ReLU())
        states_neurons = [observation_space['state'].shape[0]] + states_neurons
        self.state_linear = []
        for i in range(len(states_neurons) - 1):
            self.state_linear.append(nn.Linear(states_neurons[i], states_neurons[i + 1]))
            self.state_linear.append(nn.ReLU())
        self.state_linear = nn.Sequential(*self.state_linear)
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, birdview, state):
        x = self.cnn(birdview)
        latent_state = self.state_linear(state)
        x = th.cat((x, latent_state), dim=1)
        x = self.linear(x)
        return x


class ImpalaCNN(nn.Module):

    def __init__(self, observation_space, chans=(16, 32, 32, 64, 64), states_neurons=[256], features_dim=256, nblock=2, batch_norm=False, final_relu=True):
        super().__init__()
        self.features_dim = features_dim
        self.final_relu = final_relu
        curshape = observation_space['birdview'].shape
        s = 1 / np.sqrt(len(chans))
        self.stacks = nn.ModuleList()
        for outchan in chans:
            stack = tu.CnnDownStack(curshape[0], nblock=nblock, outchan=outchan, scale=s, batch_norm=batch_norm)
            self.stacks.append(stack)
            curshape = stack.output_shape(curshape)
        n_image_latent = tu.intprod(curshape)
        self.dense = tu.NormedLinear(n_image_latent + states_neurons[-1], features_dim, scale=1.4)
        states_neurons = [observation_space['state'].shape[0]] + states_neurons
        self.state_linear = []
        for i in range(len(states_neurons) - 1):
            self.state_linear.append(tu.NormedLinear(states_neurons[i], states_neurons[i + 1]))
            self.state_linear.append(nn.ReLU())
        self.state_linear = nn.Sequential(*self.state_linear)

    def forward(self, birdview, state):
        for layer in self.stacks:
            birdview = layer(birdview)
        x = th.flatten(birdview, 1)
        x = th.relu(x)
        latent_state = self.state_linear(state)
        x = th.cat((x, latent_state), dim=1)
        x = self.dense(x)
        if self.final_relu:
            x = th.relu(x)
        return x


def NormedConv2d(*args, scale=1, **kwargs):
    """
    nn.Conv2d but with normalized fan-in init
    """
    out = nn.Conv2d(*args, **kwargs)
    out.weight.data *= scale / out.weight.norm(dim=(1, 2, 3), p=2, keepdim=True)
    if kwargs.get('bias', True):
        out.bias.data *= 0
    return out


class CnnBasicBlock(nn.Module):
    """
    Residual basic block (without batchnorm), as in ImpalaCNN
    Preserves channel number and shape
    """

    def __init__(self, inchan, scale=1, batch_norm=False):
        super().__init__()
        self.inchan = inchan
        self.batch_norm = batch_norm
        s = math.sqrt(scale)
        self.conv0 = NormedConv2d(self.inchan, self.inchan, 3, padding=1, scale=s)
        self.conv1 = NormedConv2d(self.inchan, self.inchan, 3, padding=1, scale=s)
        if self.batch_norm:
            self.bn0 = nn.BatchNorm2d(self.inchan)
            self.bn1 = nn.BatchNorm2d(self.inchan)

    def residual(self, x):
        if getattr(self, 'batch_norm', False):
            x = self.bn0(x)
        x = F.relu(x, inplace=False)
        x = self.conv0(x)
        if getattr(self, 'batch_norm', False):
            x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.conv1(x)
        return x

    def forward(self, x):
        return x + self.residual(x)


class CnnDownStack(nn.Module):
    """
    Downsampling stack from Impala CNN
    """

    def __init__(self, inchan, nblock, outchan, scale=1, pool=True, **kwargs):
        super().__init__()
        self.inchan = inchan
        self.outchan = outchan
        self.pool = pool
        self.firstconv = NormedConv2d(inchan, outchan, 3, padding=1)
        s = scale / math.sqrt(nblock)
        self.blocks = nn.ModuleList([CnnBasicBlock(outchan, scale=s, **kwargs) for _ in range(nblock)])

    def forward(self, x):
        x = self.firstconv(x)
        if getattr(self, 'pool', True):
            x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        for block in self.blocks:
            x = block(x)
        return x

    def output_shape(self, inshape):
        c, h, w = inshape
        assert c == self.inchan
        if getattr(self, 'pool', True):
            return self.outchan, (h + 1) // 2, (w + 1) // 2
        else:
            return self.outchan, h, w


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CnnBasicBlock,
     lambda: ([], {'inchan': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (CnnDownStack,
     lambda: ([], {'inchan': 4, 'nblock': 4, 'outchan': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_zhejz_carla_roach(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

