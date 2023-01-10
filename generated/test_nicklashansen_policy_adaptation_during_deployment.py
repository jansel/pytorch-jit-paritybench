import sys
_module = sys.modules[__name__]
del sys
agent = _module
encoder = _module
arguments = _module
dm_control = _module
_render = _module
base = _module
base_test = _module
constants = _module
executor = _module
render_executor = _module
render_executor_test = _module
glfw_renderer = _module
glfw_renderer_test = _module
pyopengl = _module
egl_ext = _module
egl_renderer = _module
osmesa_renderer = _module
osmesa_renderer_test = _module
autowrap = _module
binding_generator = _module
c_declarations = _module
codegen_util = _module
header_parsing = _module
composer = _module
arena = _module
define = _module
entity = _module
entity_test = _module
environment = _module
environment_hooks_test = _module
environment_test = _module
hooks_test_utils = _module
initializer = _module
observation = _module
fake_physics = _module
obs_buffer = _module
obs_buffer_test = _module
observable = _module
mjcf = _module
mjcf_test = _module
updater = _module
updater_test = _module
robot = _module
task = _module
variation = _module
colors = _module
deterministic = _module
distributions = _module
distributions_test = _module
noises = _module
noises_test = _module
rotations = _module
variation_test = _module
variation_values = _module
entities = _module
props = _module
position_detector = _module
position_detector_test = _module
primitive = _module
primitive_test = _module
locomotion = _module
arenas = _module
assets = _module
bowl = _module
bowl_test = _module
corridors = _module
corridors_test = _module
covering = _module
covering_test = _module
floors = _module
floors_test = _module
labmaze_textures = _module
mazes = _module
mazes_test = _module
examples = _module
basic_cmu_2019 = _module
basic_rodent_2020 = _module
examples_test = _module
explore = _module
target_sphere = _module
target_sphere_test = _module
soccer = _module
boxhead = _module
boxhead_test = _module
initializers = _module
loader_test = _module
observables = _module
pitch = _module
pitch_test = _module
soccer_ball = _module
soccer_ball_test = _module
task_test = _module
team = _module
tasks = _module
escape = _module
escape_test = _module
go_to_target = _module
go_to_target_test = _module
random_goal_maze = _module
random_goal_maze_test = _module
reach = _module
reach_test = _module
walkers = _module
cmu_humanoid = _module
cmu_humanoid_test = _module
legacy_base = _module
rodent = _module
rodent_test = _module
scaled_actuators = _module
scaled_actuators_test = _module
attribute = _module
attribute_test = _module
code_for_debugging_test = _module
copier = _module
copier_test = _module
debugging = _module
debugging_test = _module
element = _module
element_test = _module
export_with_assets = _module
export_with_assets_test = _module
namescope = _module
parser = _module
physics = _module
physics_test = _module
schema = _module
traversal_utils = _module
xml_validation_test = _module
mujoco = _module
engine = _module
engine_test = _module
index = _module
index_test = _module
math = _module
math_test = _module
render_test = _module
testing = _module
decorators = _module
decorators_test = _module
generate_frames = _module
image_utils = _module
image_utils_test = _module
thread_safety_test = _module
wrapper = _module
core = _module
core_test = _module
mjbindings = _module
enums = _module
functions = _module
sizes = _module
types = _module
wrappers = _module
mjbindings_test = _module
util = _module
util_test = _module
rl = _module
control = _module
control_test = _module
suite = _module
acrobot = _module
ball_in_cup = _module
cartpole = _module
cheetah = _module
common = _module
settings = _module
mocap_demo = _module
finger = _module
fish = _module
hopper = _module
humanoid = _module
humanoid_CMU = _module
lqr = _module
lqr_solver = _module
manipulator = _module
pendulum = _module
point_mass = _module
quadruped = _module
reacher = _module
stacker = _module
swimmer = _module
domains_test = _module
lqr_test = _module
utils = _module
parse_amc = _module
parse_amc_test = _module
randomizers = _module
randomizers_test = _module
walker = _module
action_noise = _module
action_noise_test = _module
action_scale = _module
action_scale_test = _module
mujoco_profiling = _module
mujoco_profiling_test = _module
pixels = _module
pixels_test = _module
containers = _module
containers_test = _module
inverse_kinematics = _module
inverse_kinematics_test = _module
io = _module
rewards = _module
rewards_test = _module
xml_tools = _module
xml_tools_test = _module
viewer = _module
application = _module
application_test = _module
gui = _module
fullscreen_quad = _module
glfw_gui = _module
glfw_gui_test = _module
renderer = _module
renderer_test = _module
runtime = _module
runtime_test = _module
user_input = _module
user_input_test = _module
viewer_test = _module
views = _module
views_test = _module
setup = _module
dmc2gym = _module
wrappers = _module
eval = _module
logger = _module
summary = _module
train = _module
utils = _module
video = _module

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


import torch


import torch.nn as nn


import torch.nn.functional as F


from numpy.random import randint


import torchvision.transforms.functional as TF


from collections import deque


from copy import deepcopy


from torch.utils.tensorboard import SummaryWriter


from collections import defaultdict


import torchvision


import time


import random


def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability."""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


OUT_DIM = {(2): 39, (4): 35, (6): 31, (8): 27, (10): 23, (11): 21, (12): 19}


class CenterCrop(nn.Module):
    """Center-crop if observation is not already cropped"""

    def __init__(self, size):
        super().__init__()
        assert size == 84
        self.size = size

    def forward(self, x):
        assert x.ndim == 4, 'input must be a 4D tensor'
        if x.size(2) == self.size and x.size(3) == self.size:
            return x
        elif x.size(-1) == 100:
            return x[:, :, 8:-8, 8:-8]
        else:
            return ValueError('unexepcted input size')


class NormalizeImg(nn.Module):
    """Normalize observation"""

    def forward(self, x):
        return x / 255.0


def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


class PixelEncoder(nn.Module):
    """Convolutional encoder of pixel observations"""

    def __init__(self, obs_shape, feature_dim, num_layers=4, num_filters=32, num_shared_layers=4):
        super().__init__()
        assert len(obs_shape) == 3
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.num_shared_layers = num_shared_layers
        self.preprocess = nn.Sequential(CenterCrop(size=84), NormalizeImg())
        self.convs = nn.ModuleList([nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)])
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))
        out_dim = OUT_DIM[num_layers]
        self.fc = nn.Linear(num_filters * out_dim * out_dim, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

    def forward_conv(self, obs, detach=False):
        obs = self.preprocess(obs)
        conv = torch.relu(self.convs[0](obs))
        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            if i == self.num_shared_layers - 1 and detach:
                conv = conv.detach()
        h = conv.view(conv.size(0), -1)
        return h

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs, detach)
        h_fc = self.fc(h)
        h_norm = self.ln(h_fc)
        out = torch.tanh(h_norm)
        return out

    def copy_conv_weights_from(self, source, n=None):
        """Tie n first convolutional layers"""
        if n is None:
            n = self.num_layers
        for i in range(n):
            tie_weights(src=source.convs[i], trg=self.convs[i])


def make_encoder(obs_shape, feature_dim, num_layers, num_filters, num_shared_layers):
    assert num_layers in OUT_DIM.keys(), 'invalid number of layers'
    if num_shared_layers == -1 or num_shared_layers == None:
        num_shared_layers = num_layers
    assert num_shared_layers <= num_layers and num_shared_layers > 0, f'invalid number of shared layers, received {num_shared_layers} layers'
    return PixelEncoder(obs_shape, feature_dim, num_layers, num_filters, num_shared_layers)


def squash(mu, pi, log_pi):
    """Apply squashing function.
    See appendix C from https://arxiv.org/pdf/1812.05905.pdf.
    """
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-06).sum(-1, keepdim=True)
    return mu, pi, log_pi


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class Actor(nn.Module):
    """MLP actor network."""

    def __init__(self, obs_shape, action_shape, hidden_dim, encoder_feature_dim, log_std_min, log_std_max, num_layers, num_filters, num_shared_layers):
        super().__init__()
        self.encoder = make_encoder(obs_shape, encoder_feature_dim, num_layers, num_filters, num_shared_layers)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.trunk = nn.Sequential(nn.Linear(self.encoder.feature_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2 * action_shape[0]))
        self.apply(weight_init)

    def forward(self, obs, compute_pi=True, compute_log_pi=True, detach_encoder=False):
        obs = self.encoder(obs, detach=detach_encoder)
        mu, log_std = self.trunk(obs).chunk(2, dim=-1)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)
        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None
            entropy = None
        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None
        mu, pi, log_pi = squash(mu, pi, log_pi)
        return mu, pi, log_pi, log_std


class QFunction(nn.Module):
    """MLP for q-function."""

    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)
        obs_action = torch.cat([obs, action], dim=1)
        return self.trunk(obs_action)


class RotFunction(nn.Module):
    """MLP for rotation prediction."""

    def __init__(self, obs_dim, hidden_dim):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 4))

    def forward(self, h):
        return self.trunk(h)


class InvFunction(nn.Module):
    """MLP for inverse dynamics model."""

    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(2 * obs_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, action_dim))

    def forward(self, h, h_next):
        joint_h = torch.cat([h, h_next], dim=1)
        return self.trunk(joint_h)


class CURL(nn.Module):
    """Implements CURL, a contrastive learning method"""

    def __init__(self, obs_shape, z_dim, batch_size, critic, critic_target, output_type='continuous'):
        super(CURL, self).__init__()
        self.batch_size = batch_size
        self.encoder = critic.encoder
        self.encoder_target = critic_target.encoder
        self.W = nn.Parameter(torch.rand(z_dim, z_dim))
        self.output_type = output_type

    def encode(self, x, detach=False, ema=False):
        if ema:
            with torch.no_grad():
                z_out = self.encoder_target(x)
        else:
            z_out = self.encoder(x)
        if detach:
            z_out = z_out.detach()
        return z_out

    def compute_logits(self, z_a, z_pos):
        Wz = torch.matmul(self.W, z_pos.T)
        logits = torch.matmul(z_a, Wz)
        logits = logits - torch.max(logits, 1)[0][:, None]
        return logits


class Critic(nn.Module):
    """Critic network, employes two q-functions."""

    def __init__(self, obs_shape, action_shape, hidden_dim, encoder_feature_dim, num_layers, num_filters, num_shared_layers):
        super().__init__()
        self.encoder = make_encoder(obs_shape, encoder_feature_dim, num_layers, num_filters, num_shared_layers)
        self.Q1 = QFunction(self.encoder.feature_dim, action_shape[0], hidden_dim)
        self.Q2 = QFunction(self.encoder.feature_dim, action_shape[0], hidden_dim)
        self.apply(weight_init)

    def forward(self, obs, action, detach_encoder=False):
        obs = self.encoder(obs, detach=detach_encoder)
        q1 = self.Q1(obs, action)
        q2 = self.Q2(obs, action)
        return q1, q2


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (CenterCrop,
     lambda: ([], {'size': 84}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (InvFunction,
     lambda: ([], {'obs_dim': 4, 'action_dim': 4, 'hidden_dim': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     True),
    (NormalizeImg,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (QFunction,
     lambda: ([], {'obs_dim': 4, 'action_dim': 4, 'hidden_dim': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     True),
    (RotFunction,
     lambda: ([], {'obs_dim': 4, 'hidden_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_nicklashansen_policy_adaptation_during_deployment(_paritybench_base):
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

