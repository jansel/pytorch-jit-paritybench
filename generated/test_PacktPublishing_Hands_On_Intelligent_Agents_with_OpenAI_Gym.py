import sys
_module = sys.modules[__name__]
del sys
pytorch_test = _module
test_box2d = _module
get_observation_action_space = _module
list_gym_envs = _module
rl_gym_boilerplate_code = _module
run_gym_env = _module
Q_learner_MountainCar = _module
rl_gym_boilerplate_code_v2 = _module
deep_Q_learner = _module
environment = _module
atari = _module
utils = _module
function_approximator = _module
cnn = _module
perceptron = _module
decay_schedule = _module
experience_memory = _module
params_manager = _module
weights_initializer = _module
carla_gym = _module
envs = _module
carla = _module
agent = _module
forward_agent = _module
carla_server_pb2 = _module
client = _module
driving_benchmark = _module
experiment = _module
experiment_suites = _module
basic_experiment_suite = _module
corl_2017 = _module
experiment_suite = _module
metrics = _module
recording = _module
results_printer = _module
image_converter = _module
planner = _module
astar = _module
city_track = _module
converter = _module
graph = _module
grid = _module
map = _module
sensor = _module
settings = _module
tcp = _module
transform = _module
util = _module
carla_env = _module
custom_environments = _module
custom_env_template = _module
a2c_agent = _module
async_a2c_agent = _module
batched_a2c_agent = _module
deep = _module
shallow = _module
run_roboschool_env = _module
setup_test = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import re, math, string, numpy, torch, torchtext, torchaudio, logging, itertools, numbers, inspect, functools, copy, scipy, types, time, torchvision, enum, random, typing, warnings, abc, collections, uuid
import numpy as np
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch


import random


import numpy as np


from collections import namedtuple


from torch.distributions.multivariate_normal import MultivariateNormal


from torch.distributions.categorical import Categorical


import torch.multiprocessing as mp


import torch.nn.functional as F


import time


class CNN(torch.nn.Module):

    def __init__(self, input_shape, output_shape, device=torch.device('cpu')):
        """
        A Convolution Neural Network (CNN) class to approximate functions with visual/image inputs

        :param input_shape:  Shape/dimension of the input image. Assumed to be resized to C x 84 x 84
        :param output_shape: Shape/dimension of the output.
        :param device: The device (cpu or cuda) that the CNN should use to store the inputs for the forward pass
        """
        super(CNN, self).__init__()
        self.device = device
        self.layer1 = torch.nn.Sequential(torch.nn.Conv2d(input_shape[0], 
            32, kernel_size=8, stride=4, padding=0), torch.nn.ReLU())
        self.layer2 = torch.nn.Sequential(torch.nn.Conv2d(32, 64,
            kernel_size=3, stride=2, padding=0), torch.nn.ReLU())
        self.layer3 = torch.nn.Sequential(torch.nn.Conv2d(64, 64,
            kernel_size=3, stride=1, padding=0), torch.nn.ReLU())
        self.out = torch.nn.Linear(64 * 7 * 7, output_shape)

    def forward(self, x):
        x = torch.from_numpy(x).float()
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.shape[0], -1)
        x = self.out(x)
        return x


class SLP(torch.nn.Module):
    """
    A Single Layer Perceptron (SLP) class to approximate functions
    """

    def __init__(self, input_shape, output_shape, device=torch.device('cpu')):
        """
        :param input_shape: Shape/dimension of the input
        :param output_shape: Shape/dimension of the output
        :param device: The device (cpu or cuda) that the SLP should use to store the inputs for the forward pass
        """
        super(SLP, self).__init__()
        self.device = device
        self.input_shape = input_shape[0]
        self.hidden_shape = 40
        self.linear1 = torch.nn.Linear(self.input_shape, self.hidden_shape)
        self.out = torch.nn.Linear(self.hidden_shape, output_shape)

    def forward(self, x):
        x = torch.from_numpy(x).float()
        x = torch.nn.functional.relu(self.linear1(x))
        x = self.out(x)
        return x


class Actor(torch.nn.Module):

    def __init__(self, input_shape, actor_shape, device=torch.device('cpu')):
        """
        Deep convolutional Neural Network to represent Actor in an Actor-Critic algorithm
        The Policy is parametrized using a Gaussian distribution with mean mu and variance sigma
        The Actor's policy parameters (mu, sigma) are output by the deep CNN implemented
        in this class.
        :param input_shape: Shape of each of the observations
        :param actor_shape: Shape of the actor's output. Typically the shape of the actions
        :param device: The torch.device (cpu or cuda) where the inputs and the parameters are to be stored and operated
        """
        super(Actor, self).__init__()
        self.device = device
        self.layer1 = torch.nn.Sequential(torch.nn.Conv2d(input_shape[2], 
            32, 8, stride=4, padding=0), torch.nn.ReLU())
        self.layer2 = torch.nn.Sequential(torch.nn.Conv2d(32, 64, 3, stride
            =2, padding=0), torch.nn.ReLU())
        self.layer3 = torch.nn.Sequential(torch.nn.Conv2d(64, 64, 3, stride
            =1, padding=0), torch.nn.ReLU())
        self.layer4 = torch.nn.Sequential(torch.nn.Linear(64 * 7 * 7, 512),
            torch.nn.ReLU())
        self.actor_mu = torch.nn.Linear(512, actor_shape)
        self.actor_sigma = torch.nn.Linear(512, actor_shape)

    def forward(self, x):
        """
        Forward pass through the Actor network. Takes batch_size x observations as input and produces mu and sigma
        as the outputs
        :param x: The observations
        :return: Mean (mu) and Sigma (sigma) for a Gaussian policy
        """
        x.requires_grad_()
        x = x
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.shape[0], -1)
        x = self.layer4(x)
        actor_mu = self.actor_mu(x)
        actor_sigma = self.actor_sigma(x)
        return actor_mu, actor_sigma


class DiscreteActor(torch.nn.Module):

    def __init__(self, input_shape, actor_shape, device=torch.device('cpu')):
        """
        Deep convolutional Neural Network to represent Actor in an Actor-Critic algorithm
        The Policy is parametrized using a categorical/discrete distribution with logits
        The Actor's policy parameters (logits) are output by the deep CNN implemented
        in this class.
        :param input_shape: Shape of each of the observations
        :param actor_shape: Shape of the actor's output. Typically the shape of the actions
        :param device: The torch.device (cpu or cuda) where the inputs and the parameters are to be stored and operated
        """
        super(DiscreteActor, self).__init__()
        self.device = device
        self.layer1 = torch.nn.Sequential(torch.nn.Conv2d(input_shape[2], 
            32, 8, stride=4, padding=0), torch.nn.ReLU())
        self.layer2 = torch.nn.Sequential(torch.nn.Conv2d(32, 64, 3, stride
            =2, padding=0), torch.nn.ReLU())
        self.layer3 = torch.nn.Sequential(torch.nn.Conv2d(64, 64, 3, stride
            =1, padding=0), torch.nn.ReLU())
        self.layer4 = torch.nn.Sequential(torch.nn.Linear(64 * 7 * 7, 512),
            torch.nn.ReLU())
        self.logits = torch.nn.Linear(512, actor_shape)

    def forward(self, x):
        """
        Forward pass through the Actor network. Takes batch_size x observations as input and produces mu and sigma
        as the outputs
        :param x: The observations
        :return: Mean (mu) and Sigma (sigma) for a Gaussian policy
        """
        x.requires_grad_()
        x = x
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.shape[0], -1)
        x = self.layer4(x)
        logits = self.logits(x)
        return logits


class Critic(torch.nn.Module):

    def __init__(self, input_shape, critic_shape=1, device=torch.device('cpu')
        ):
        """
        Deep convolutional Neural Network to represent the Critic in an Actor-Critic algorithm
        :param input_shape: Shape of each of the observations
        :param critic_shape: Shape of the Critic's output. Typically 1
        :param device: The torch.device (cpu or cuda) where the inputs and the parameters are to be stored and operated
        """
        super(Critic, self).__init__()
        self.device = device
        self.layer1 = torch.nn.Sequential(torch.nn.Conv2d(input_shape[2], 
            32, 8, stride=4, padding=0), torch.nn.ReLU())
        self.layer2 = torch.nn.Sequential(torch.nn.Conv2d(32, 64, 3, stride
            =2, padding=0), torch.nn.ReLU())
        self.layer3 = torch.nn.Sequential(torch.nn.Conv2d(64, 64, 3, stride
            =1, padding=0), torch.nn.ReLU())
        self.layer4 = torch.nn.Sequential(torch.nn.Linear(64 * 7 * 7, 512),
            torch.nn.ReLU())
        self.critic = torch.nn.Linear(512, critic_shape)

    def forward(self, x):
        """
        Forward pass through the Critic network. Takes batch_size x observations as input and produces the value
        estimate as the output
        :param x: The observations
        :return: Mean (mu) and Sigma (sigma) for a Gaussian policy
        """
        x.requires_grad_()
        x = x
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.shape[0], -1)
        x = self.layer4(x)
        critic = self.critic(x)
        return critic


class ActorCritic(torch.nn.Module):

    def __init__(self, input_shape, actor_shape, critic_shape, device=torch
        .device('cpu')):
        """
        Deep convolutional Neural Network to represent both policy  (Actor) and a value function (Critic).
        The Policy is parametrized using a Gaussian distribution with mean mu and variance sigma
        The Actor's policy parameters (mu, sigma) and the Critic's Value (value) are output by the deep CNN implemented
        in this class.
        :param input_shape: Shape of each of the observations
        :param actor_shape: Shape of the actor's output. Typically the shape of the actions
        :param critic_shape: Shape of the Critic's output. Typically 1
        :param device: The torch.device (cpu or cuda) where the inputs and the parameters are to be stored and operated
        """
        super(ActorCritic, self).__init__()
        self.device = device
        self.layer1 = torch.nn.Sequential(torch.nn.Conv2d(input_shape[2], 
            32, 8, stride=4, padding=0), torch.nn.ReLU())
        self.layer2 = torch.nn.Sequential(torch.nn.Conv2d(32, 64, 3, stride
            =2, padding=0), torch.nn.ReLU())
        self.layer3 = torch.nn.Sequential(torch.nn.Conv2d(64, 64, 3, stride
            =1, padding=0), torch.nn.ReLU())
        self.layer4 = torch.nn.Sequential(torch.nn.Linear(64 * 7 * 7, 512),
            torch.nn.ReLU())
        self.actor_mu = torch.nn.Linear(512, actor_shape)
        self.actor_sigma = torch.nn.Linear(512, actor_shape)
        self.critic = torch.nn.Linear(512, critic_shape)

    def forward(self, x):
        """
        Forward pass through the Actor-Critic network. Takes batch_size x observations as input and produces
        mu, sigma and the value estimate
        as the outputs
        :param x: The observations
        :return: Mean (actor_mu), Sigma (actor_sigma) for a Gaussian policy and the Critic's value estimate (critic)
        """
        x.requires_grad_()
        x = x
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.shape[0], -1)
        x = self.layer4(x)
        actor_mu = self.actor_mu(x)
        actor_sigma = self.actor_sigma(x)
        critic = self.critic(x)
        return actor_mu, actor_sigma, critic


class Actor(torch.nn.Module):

    def __init__(self, input_shape, output_shape, device=torch.device('cpu')):
        """
        A feed forward neural network that produces two continuous values mean (mu) and sigma, each of output_shape
        . Used to represent the Actor in an Actor-Critic algorithm
        :param input_shape: Shape of the inputs. This is typically the shape of each of the observations for the Actor
        :param output_shape: Shape of the outputs. This is the shape of the actions that the Actor should produce
        :param device: The torch.device (cpu or cuda) where the inputs and the parameters are to be stored and operated
        """
        super(Actor, self).__init__()
        self.device = device
        self.layer1 = torch.nn.Sequential(torch.nn.Linear(input_shape[0], 
            64), torch.nn.ReLU())
        self.layer2 = torch.nn.Sequential(torch.nn.Linear(64, 32), torch.nn
            .ReLU())
        self.actor_mu = torch.nn.Linear(32, output_shape)
        self.actor_sigma = torch.nn.Linear(32, output_shape)

    def forward(self, x):
        """
        Forward pass through the Actor network. Takes batch_size x observations as input and produces mu and sigma
        as the outputs
        :param x: The observations
        :return: Mean (mu) and Sigma (sigma) for a Gaussian policy
        """
        x = x
        x = self.layer1(x)
        x = self.layer2(x)
        mu = self.actor_mu(x)
        sigma = self.actor_sigma(x)
        return mu, sigma


class DiscreteActor(torch.nn.Module):

    def __init__(self, input_shape, output_shape, device=torch.device('cpu')):
        """
        A feed forward neural network that produces a logit for each action in the action space.
        Used to represent the Actor in an Actor-Critic algorithm
        :param input_shape: Shape of the inputs. This is typically the shape of each of the observations for the Actor
        :param output_shape: Shape of the outputs. This is the shape of the actions that the Actor should produce
        :param device: The torch.device (cpu or cuda) where the inputs and the parameters are to be stored and operated
        """
        super(DiscreteActor, self).__init__()
        self.device = device
        self.layer1 = torch.nn.Sequential(torch.nn.Linear(input_shape[0], 
            64), torch.nn.ReLU())
        self.layer2 = torch.nn.Sequential(torch.nn.Linear(64, 32), torch.nn
            .ReLU())
        self.logits = torch.nn.Linear(32, output_shape)

    def forward(self, x):
        """
        Forward pass through the Actor network. Takes batch_size x observations as input and produces mu and sigma
        as the outputs
        :param x: The observations
        :return: Mean (mu) and Sigma (sigma) for a Gaussian policy
        """
        x = x
        x = self.layer1(x)
        x = self.layer2(x)
        logits = self.logits(x)
        return logits


class Critic(torch.nn.Module):

    def __init__(self, input_shape, output_shape=1, device=torch.device('cpu')
        ):
        """
        A feed forward neural network that produces a continuous value. Used to represent the Critic
        in an Actor-Critic algorithm that estimates the value of the current observation/state
        :param input_shape: Shape of the inputs. This is typically the shape of the observations for the Actor
        :param output_shape: Shape of the output. This is most often 1 as the Critic is expected to produce a single
        value given given an observation/state
        :param device: The torch.device (cpu or cuda) where the inputs and the parameters are to be stored and operated
        """
        super(Critic, self).__init__()
        self.device = device
        self.layer1 = torch.nn.Sequential(torch.nn.Linear(input_shape[0], 
            64), torch.nn.ReLU())
        self.layer2 = torch.nn.Sequential(torch.nn.Linear(64, 32), torch.nn
            .ReLU())
        self.critic = torch.nn.Linear(32, output_shape)

    def forward(self, x):
        """
        Forward pass through the Critic network. Takes batch_size x observations as input and produces the value
        estimate as the output
        :param x: The observations
        :return: Mean (mu) and Sigma (sigma) for a Gaussian policy
        """
        x = x
        x = self.layer1(x)
        x = self.layer2(x)
        critic = self.critic(x)
        return critic


class ActorCritic(torch.nn.Module):

    def __init__(self, input_shape, actor_shape, critic_shape, device=torch
        .device('cpu')):
        """
        A feed forward neural network used to represent both an Actor and the Critic in an Actor-Critic algorithm.
        :param input_shape: Shape of the inputs. This is typically the shape of the observations
        :param actor_shape: Shape of the actor outputs. This is the shape of the actions that the Actor should produce
        :param critic_shape: Shape of the critic output. This is most often 1 as the Critic is expected to produce a
        single value given given an observation/state
        :param device: The torch.device (cpu or cuda) where the inputs and the parameters are to be stored and operated
        """
        super(ActorCritic, self).__init__()
        self.device = device
        self.layer1 = torch.nn.Sequential(torch.nn.Linear(input_shape[0], 
            32), torch.nn.ReLU())
        self.layer2 = torch.nn.Sequential(torch.nn.Linear(32, 16), torch.nn
            .ReLU())
        self.actor_mu = torch.nn.Linear(16, actor_shape)
        self.actor_sigma = torch.nn.Linear(16, actor_shape)
        self.critic = torch.nn.Linear(16, critic_shape)

    def forward(self, x):
        """
        Forward pass through the Actor-Critic network. Takes batch_size x observations as input and produces
        mu, sigma and the value estimate
        as the outputs
        :param x: The observations
        :return: Mean (actor_mu), Sigma (actor_sigma) for a Gaussian policy and the Critic's value estimate (critic)
        """
        x.requires_grad_()
        x = x
        x = self.layer1(x)
        x = self.layer2(x)
        actor_mu = self.actor_mu(x)
        actor_sigma = self.actor_sigma(x)
        critic = self.critic(x)
        return actor_mu, actor_sigma, critic


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_PacktPublishing_Hands_On_Intelligent_Agents_with_OpenAI_Gym(_paritybench_base):
    pass
    def test_000(self):
        self._check(Actor(*[], **{'input_shape': [4, 4], 'output_shape': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(ActorCritic(*[], **{'input_shape': [4, 4], 'actor_shape': 4, 'critic_shape': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(Critic(*[], **{'input_shape': [4, 4]}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(DiscreteActor(*[], **{'input_shape': [4, 4], 'output_shape': 4}), [torch.rand([4, 4, 4, 4])], {})

