import sys
_module = sys.modules[__name__]
del sys
breakout_a2c = _module
breakout_a2c_evaluate = _module
qbert_ppo = _module
half_cheetah_ddpg = _module
setup = _module
vel = _module
api = _module
callback = _module
data = _module
augmentation = _module
dataflow = _module
image_ops = _module
info = _module
learner = _module
metrics = _module
averaging_metric = _module
base_metric = _module
summing_metric = _module
value_metric = _module
model = _module
model_factory = _module
optimizer = _module
schedule = _module
scheduler = _module
source = _module
storage = _module
train_phase = _module
augmentations = _module
center_crop = _module
normalize = _module
random_crop = _module
random_horizontal_flip = _module
random_lighting = _module
random_rotate = _module
random_scale = _module
scale_min_size = _module
to_array = _module
to_tensor = _module
tta = _module
train_tta = _module
callbacks = _module
time_tracker = _module
commands = _module
augvis_command = _module
lr_find_command = _module
phase_train_command = _module
rnn = _module
generate_text = _module
summary_command = _module
train_command = _module
vis_store_command = _module
exceptions = _module
internals = _module
context = _module
generic_factory = _module
model_config = _module
parser = _module
provider = _module
tests = _module
fixture_a = _module
fixture_b = _module
test_parser = _module
test_provider = _module
launcher = _module
math = _module
functions = _module
processes = _module
accuracy = _module
loss_metric = _module
models = _module
imagenet = _module
resnet34 = _module
multilayer_rnn_sequence_classification = _module
multilayer_rnn_sequence_model = _module
vision = _module
cifar10_cnn_01 = _module
cifar_resnet_v1 = _module
cifar_resnet_v2 = _module
cifar_resnext = _module
mnist_cnn_01 = _module
modules = _module
input = _module
embedding = _module
identity = _module
image_to_tensor = _module
normalize_observations = _module
one_hot_encoding = _module
layers = _module
resnet_v1 = _module
resnet_v2 = _module
resnext = _module
rnn_cell = _module
rnn_layer = _module
notebook = _module
loader = _module
openai = _module
baselines = _module
bench = _module
benchmarks = _module
monitor = _module
common = _module
atari_wrappers = _module
retro_wrappers = _module
running_mean_std = _module
tile_images = _module
vec_env = _module
dummy_vec_env = _module
shmem_vec_env = _module
subproc_vec_env = _module
util = _module
vec_frame_stack = _module
vec_normalize = _module
logger = _module
optimizers = _module
adadelta = _module
adam = _module
rmsprop = _module
rmsprop_tf = _module
sgd = _module
phase = _module
cycle = _module
freeze = _module
generic = _module
unfreeze = _module
rl = _module
algo = _module
distributional_dqn = _module
dqn = _module
policy_gradient = _module
a2c = _module
acer = _module
ddpg = _module
ppo = _module
trpo = _module
algo_base = _module
env_base = _module
env_roller = _module
evaluator = _module
reinforcer_base = _module
replay_buffer = _module
rollout = _module
buffers = _module
backend = _module
circular_buffer_backend = _module
circular_vec_buffer_backend = _module
prioritized_buffer_backend = _module
prioritized_vec_buffer_backend = _module
segment_tree = _module
circular_replay_buffer = _module
prioritized_circular_replay_buffer = _module
test_circular_buffer_backend = _module
test_circular_vec_env_buffer_backend = _module
test_prioritized_circular_buffer_backend = _module
test_prioritized_vec_buffer_backend = _module
enjoy = _module
evaluate_env_command = _module
record_movie_command = _module
rl_train_command = _module
discount_bootstrap = _module
env = _module
classic_atari = _module
classic_control = _module
mujoco = _module
wrappers = _module
clip_episode_length = _module
env_normalize = _module
step_env_roller = _module
trajectory_replay_env_roller = _module
transition_replay_env_roller = _module
backbone = _module
double_nature_cnn = _module
double_noisy_nature_cnn = _module
lstm = _module
mlp = _module
nature_cnn = _module
nature_cnn_rnn = _module
nature_cnn_small = _module
noisy_nature_cnn = _module
deterministic_policy_model = _module
q_distributional_model = _module
q_dueling_model = _module
q_model = _module
q_noisy_model = _module
q_rainbow_model = _module
q_stochastic_policy_model = _module
stochastic_policy_model = _module
stochastic_policy_model_separate = _module
stochastic_policy_rnn_model = _module
action_head = _module
deterministic_action_head = _module
deterministic_critic_head = _module
noise = _module
eps_greedy = _module
ou_noise = _module
noisy_linear = _module
q_distributional_head = _module
q_distributional_noisy_dueling_head = _module
q_dueling_head = _module
q_head = _module
q_noisy_head = _module
test = _module
test_action_head = _module
value_head = _module
reinforcers = _module
buffered_mixed_policy_iteration_reinforcer = _module
buffered_off_policy_iteration_reinforcer = _module
on_policy_iteration_reinforcer = _module
test_integration = _module
vecenv = _module
dummy = _module
shared_mem = _module
subproc = _module
ladder = _module
linear_batch_scaler = _module
multi_step = _module
reduce_lr_on_plateau = _module
schedules = _module
constant = _module
linear = _module
linear_and_constant = _module
sources = _module
img_dir_source = _module
nlp = _module
imdb = _module
text_url = _module
cifar10 = _module
mnist = _module
mongodb = _module
classic = _module
strategy = _module
checkpoint_strategy = _module
classic_checkpoint_strategy = _module
streaming = _module
stdout = _module
visdom = _module
better = _module
intepolate = _module
module_util = _module
network = _module
random = _module
situational = _module
summary = _module
tensor_accumulator = _module
tensor_util = _module

from _paritybench_helpers import _mock_config
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch


import typing


import torch.nn as nn


import numpy as np


import torch.nn.functional as F


import torch.distributions as dist


import torch.nn.init as init


import torch.nn.utils


import torch.autograd


import torch.autograd as autograd


import math


import numpy.testing as nt


import torch.distributions as d


import collections


import itertools as it


from torch.autograd import Variable


from collections import OrderedDict


class AdaptiveConcatPool2d(nn.Module):
    """ Concat pooling - combined average pool and max pool """

    def __init__(self, sz=None):
        super().__init__()
        sz = sz or (1, 1)
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


class Lambda(nn.Module):
    """ Simple torch lambda layer """

    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


class Flatten(nn.Module):
    """ Flatten input vector """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


def one_hot_encoding(input_tensor, num_labels):
    """ One-hot encode labels from input """
    xview = input_tensor.view(-1, 1).to(torch.long)
    onehot = torch.zeros(xview.size(0), num_labels, device=input_tensor.
        device, dtype=torch.float)
    onehot.scatter_(1, xview, 1)
    return onehot.view(list(input_tensor.shape) + [-1])


class OneHotEncode(nn.Module):
    """ One-hot encoding layer """

    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x):
        return one_hot_encoding(x, self.num_classes)


def conv3x3(in_channels, out_channels, stride=1):
    """
    3x3 convolution with padding.
    Original code has had bias turned off, because Batch Norm would remove the bias either way
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=
        stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    """
    Single residual block consisting of two convolutional layers and a nonlinearity between them
    """

    def __init__(self, in_channels, out_channels, stride=1, divisor=None):
        super().__init__()
        self.stride = stride
        self.divisor = divisor
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels,
                out_channels, kernel_size=1, stride=stride, bias=False), nn
                .BatchNorm2d(out_channels))
        else:
            self.shortcut = None
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.shortcut is not None:
            residual = self.shortcut(x)
        else:
            residual = x
        out += residual
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    """
    A 'bottleneck' residual block consisting of three convolutional layers, where the first one is a downsampler,
    then we have a 3x3 followed by an upsampler.
    """

    def __init__(self, in_channels, out_channels, stride=1, divisor=4):
        super().__init__()
        self.stride = stride
        self.divisor = divisor
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels,
                out_channels, kernel_size=1, stride=stride, bias=False), nn
                .BatchNorm2d(out_channels))
        else:
            self.shortcut = None
        self.bottleneck_channels = out_channels // divisor
        self.conv1 = nn.Conv2d(in_channels, self.bottleneck_channels,
            kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.bottleneck_channels)
        self.conv2 = conv3x3(self.bottleneck_channels, self.
            bottleneck_channels, stride)
        self.bn2 = nn.BatchNorm2d(self.bottleneck_channels)
        self.conv3 = nn.Conv2d(self.bottleneck_channels, out_channels,
            kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.shortcut is not None:
            residual = self.shortcut(x)
        else:
            residual = x
        out += residual
        out = F.relu(out)
        return out


class BasicBlock(nn.Module):
    """
    Single residual block consisting of two convolutional layers and a nonlinearity between them
    """

    def __init__(self, in_channels, out_channels, stride=1, divisor=None):
        super().__init__()
        self.stride = stride
        self.divisor = divisor
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels,
                kernel_size=1, stride=stride, bias=False)
        else:
            self.shortcut = None
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)

    def forward(self, x):
        out = self.bn1(x)
        out = F.relu(out)
        if self.shortcut:
            residual = self.shortcut(out)
        else:
            residual = x
        out = self.conv1(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv2(out)
        out += residual
        return out


class Bottleneck(nn.Module):
    """
    A 'bottleneck' residual block consisting of three convolutional layers, where the first one is a downsampler,
    then we have a 3x3 followed by an upsampler.
    """

    def __init__(self, in_channels, out_channels, stride=1, divisor=4):
        super().__init__()
        self.stride = stride
        self.divisor = divisor
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels,
                out_channels, kernel_size=1, stride=stride, bias=False))
        else:
            self.shortcut = None
        self.bottleneck_channels = out_channels // divisor
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, self.bottleneck_channels,
            kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.bottleneck_channels)
        self.conv2 = nn.Conv2d(self.bottleneck_channels, self.
            bottleneck_channels, kernel_size=3, stride=stride, padding=1,
            bias=False)
        self.bn3 = nn.BatchNorm2d(self.bottleneck_channels)
        self.conv3 = nn.Conv2d(self.bottleneck_channels, out_channels,
            kernel_size=1, bias=False)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        if self.shortcut:
            residual = self.shortcut(out)
        else:
            residual = x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += residual
        return out


class ResNeXtBottleneck(nn.Module):
    """
    RexNeXt bottleneck type C (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)
    """

    def __init__(self, in_channels, out_channels, cardinality, divisor,
        stride=1):
        super(ResNeXtBottleneck, self).__init__()
        self.cardinality = cardinality
        self.stride = stride
        self.divisor = divisor
        D = out_channels // divisor
        C = cardinality
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels,
                out_channels, kernel_size=1, stride=stride, bias=False), nn
                .BatchNorm2d(out_channels))
        else:
            self.shortcut = None
        self.conv_reduce = nn.Conv2d(in_channels, D * C, kernel_size=1,
            stride=1, padding=0, bias=False)
        self.bn_reduce = nn.BatchNorm2d(D * C)
        self.conv_conv = nn.Conv2d(D * C, D * C, kernel_size=3, stride=
            stride, padding=1, groups=cardinality, bias=False)
        self.bn = nn.BatchNorm2d(D * C)
        self.conv_expand = nn.Conv2d(D * C, out_channels, kernel_size=1,
            stride=1, padding=0, bias=False)
        self.bn_expand = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        bottleneck = self.conv_reduce(x)
        bottleneck = F.relu(self.bn_reduce(bottleneck), inplace=True)
        bottleneck = self.conv_conv(bottleneck)
        bottleneck = F.relu(self.bn(bottleneck), inplace=True)
        bottleneck = self.conv_expand(bottleneck)
        bottleneck = self.bn_expand(bottleneck)
        if self.shortcut is not None:
            residual = self.shortcut(x)
        else:
            residual = x
        return F.relu(residual + bottleneck, inplace=True)


class DiagGaussianActionHead(nn.Module):
    """
    Action head where actions are normally distibuted uncorrelated variables with specific means and variances.

    Means are calculated directly from the network while standard deviation are a parameter of this module
    """
    LOG2PI = np.log(2.0 * np.pi)

    def __init__(self, input_dim, num_dimensions):
        super().__init__()
        self.input_dim = input_dim
        self.num_dimensions = num_dimensions
        self.linear_layer = nn.Linear(input_dim, num_dimensions)
        self.log_std = nn.Parameter(torch.zeros(1, num_dimensions))

    def forward(self, input_data):
        means = self.linear_layer(input_data)
        log_std_tile = self.log_std.repeat(means.size(0), 1)
        return torch.stack([means, log_std_tile], dim=-1)

    def sample(self, params, argmax_sampling=False):
        """ Sample from a probability space of all actions """
        means = params[:, :, (0)]
        log_std = params[:, :, (1)]
        if argmax_sampling:
            return means
        else:
            return torch.randn_like(means) * torch.exp(log_std) + means

    def logprob(self, action_sample, pd_params):
        """ Log-likelihood """
        means = pd_params[:, :, (0)]
        log_std = pd_params[:, :, (1)]
        std = torch.exp(log_std)
        z_score = (action_sample - means) / std
        return -(0.5 * (z_score ** 2 + self.LOG2PI).sum(dim=-1) + log_std.
            sum(dim=-1))

    def reset_weights(self):
        init.orthogonal_(self.linear_layer.weight, gain=0.01)
        init.constant_(self.linear_layer.bias, 0.0)

    def entropy(self, params):
        """
        Categorical distribution entropy calculation - sum probs * log(probs).
        In case of diagonal gaussian distribution - 1/2 log(2 pi e sigma^2)
        """
        log_std = params[:, :, (1)]
        return (log_std + 0.5 * (self.LOG2PI + 1)).sum(dim=-1)

    def kl_divergence(self, params_q, params_p):
        """
        Categorical distribution KL divergence calculation
        KL(Q || P) = sum Q_i log (Q_i / P_i)

        Formula is:
        log(sigma_p) - log(sigma_q) + (sigma_q^2 + (mu_q - mu_p)^2))/(2 * sigma_p^2)
        """
        means_q = params_q[:, :, (0)]
        log_std_q = params_q[:, :, (1)]
        means_p = params_p[:, :, (0)]
        log_std_p = params_p[:, :, (1)]
        std_q = torch.exp(log_std_q)
        std_p = torch.exp(log_std_p)
        kl_div = log_std_p - log_std_q + (std_q ** 2 + (means_q - means_p) ** 2
            ) / (2.0 * std_p ** 2) - 0.5
        return kl_div.sum(dim=-1)


class CategoricalActionHead(nn.Module):
    """ Action head with categorical actions """

    def __init__(self, input_dim, num_actions):
        super().__init__()
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.linear_layer = nn.Linear(input_dim, num_actions)

    def forward(self, input_data):
        return F.log_softmax(self.linear_layer(input_data), dim=1)

    def logprob(self, actions, action_logits):
        """ Logarithm of probability of given sample """
        neg_log_prob = F.nll_loss(action_logits, actions, reduction='none')
        return -neg_log_prob

    def sample(self, logits, argmax_sampling=False):
        """ Sample from a probability space of all actions """
        if argmax_sampling:
            return torch.argmax(logits, dim=-1)
        else:
            u = torch.rand_like(logits)
            return torch.argmax(logits - torch.log(-torch.log(u)), dim=-1)

    def reset_weights(self):
        init.orthogonal_(self.linear_layer.weight, gain=0.01)
        init.constant_(self.linear_layer.bias, 0.0)

    def entropy(self, logits):
        """ Categorical distribution entropy calculation - sum probs * log(probs) """
        probs = torch.exp(logits)
        entropy = -torch.sum(probs * logits, dim=-1)
        return entropy

    def kl_divergence(self, logits_q, logits_p):
        """
        Categorical distribution KL divergence calculation
        KL(Q || P) = sum Q_i log (Q_i / P_i)
        When talking about logits this is:
        sum exp(Q_i) * (Q_i - P_i)
        """
        return (torch.exp(logits_q) * (logits_q - logits_p)).sum(1, keepdim
            =True)


class ActionHead(nn.Module):
    """
    Network head for action determination. Returns probability distribution parametrization
    """

    def __init__(self, input_dim, action_space):
        super().__init__()
        self.action_space = action_space
        if isinstance(action_space, spaces.Box):
            assert len(action_space.shape) == 1
            self.head = DiagGaussianActionHead(input_dim, action_space.shape[0]
                )
        elif isinstance(action_space, spaces.Discrete):
            self.head = CategoricalActionHead(input_dim, action_space.n)
        else:
            raise NotImplementedError

    def forward(self, input_data):
        return self.head(input_data)

    def sample(self, policy_params, **kwargs):
        """ Sample from a probability space of all actions """
        return self.head.sample(policy_params, **kwargs)

    def reset_weights(self):
        """ Initialize weights to sane defaults """
        self.head.reset_weights()

    def entropy(self, policy_params):
        """ Entropy calculation - sum probs * log(probs) """
        return self.head.entropy(policy_params)

    def kl_divergence(self, params_q, params_p):
        """ Kullback–Leibler divergence between two sets of parameters """
        return self.head.kl_divergence(params_q, params_p)

    def logprob(self, action_sample, policy_params):
        """ - log probabilty of selected actions """
        return self.head.logprob(action_sample, policy_params)


class DeterministicActionHead(nn.Module):
    """
    Network head for action determination. Returns deterministic action depending on the inputs
    """

    def __init__(self, input_dim, action_space):
        super().__init__()
        self.action_space = action_space
        assert isinstance(action_space, spaces.Box)
        assert len(action_space.shape) == 1
        assert (np.abs(action_space.low) == action_space.high).all()
        self.register_buffer('max_action', torch.from_numpy(action_space.high))
        self.linear_layer = nn.Linear(input_dim, action_space.shape[0])

    def forward(self, input_data):
        return torch.tanh(self.linear_layer(input_data)) * self.max_action

    def sample(self, params, **_):
        """ Sample from a probability space of all actions """
        return {'actions': self(params)}

    def reset_weights(self):
        """ Initialize weights to sane defaults """
        init.orthogonal_(self.linear_layer.weight, gain=0.01)
        init.constant_(self.linear_layer.bias, 0.0)


class DeterministicCriticHead(nn.Module):
    """
    Network head for action-dependent critic.
    Returns deterministic action-value for given combination of action and state.
    """

    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, input_data):
        return self.linear(input_data)[:, (0)]

    def reset_weights(self):
        """ Initialize weights to sane defaults """
        init.uniform_(self.linear.weight, -0.003, 0.003)
        init.zeros_(self.linear.bias)


class Schedule:
    """ A schedule class encoding some kind of interpolation of a single value """

    def value(self, progress_indicator):
        """ Value at given progress step """
        raise NotImplementedError


class ConstantSchedule(Schedule):
    """ Interpolate variable linearly between start value and final value """

    def __init__(self, value):
        self._value = value

    def value(self, progress_indicator):
        """ Interpolate linearly between start and end """
        return self._value


class EpsGreedy(nn.Module):
    """ Epsilon-greedy action selection """

    def __init__(self, epsilon: typing.Union[Schedule, float], environment):
        super().__init__()
        if isinstance(epsilon, Schedule):
            self.epsilon_schedule = epsilon
        else:
            self.epsilon_schedule = ConstantSchedule(epsilon)
        self.action_space = environment.action_space

    def forward(self, actions, batch_info=None):
        if batch_info is None:
            epsilon = self.epsilon_schedule.value(1.0)
        else:
            epsilon = self.epsilon_schedule.value(batch_info['progress'])
        random_samples = torch.randint_like(actions, self.action_space.n)
        selector = torch.rand_like(random_samples, dtype=torch.float32)
        noisy_actions = torch.where(selector > epsilon, actions, random_samples
            )
        return noisy_actions

    def reset_training_state(self, dones, batch_info):
        """ A hook for a model to react when during training episode is finished """
        pass


class OrnsteinUhlenbeckNoiseProcess:
    """
    Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
    """

    def __init__(self, mu, sigma, theta=0.15, dt=0.01, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.x_prev = None
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev
            ) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size
            =self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu
            )

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.
            mu, self.sigma)


class OuNoise(nn.Module):
    """ Ornstein–Uhlenbeck noise process for action noise """

    def __init__(self, std_dev, environment):
        super().__init__()
        self.std_dev = std_dev
        self.action_space = environment.action_space
        self.processes = []
        self.register_buffer('low_tensor', torch.from_numpy(self.
            action_space.low).unsqueeze(0))
        self.register_buffer('high_tensor', torch.from_numpy(self.
            action_space.high).unsqueeze(0))

    def reset_training_state(self, dones, batch_info):
        """ A hook for a model to react when during training episode is finished """
        for idx, done in enumerate(dones):
            if done > 0.5:
                self.processes[idx].reset()

    def forward(self, actions, batch_info):
        """ Return model step after applying noise """
        while len(self.processes) < actions.shape[0]:
            len_action_space = self.action_space.shape[-1]
            self.processes.append(OrnsteinUhlenbeckNoiseProcess(np.zeros(
                len_action_space), float(self.std_dev) * np.ones(
                len_action_space)))
        noise = torch.from_numpy(np.stack([x() for x in self.processes])
            ).float()
        return torch.min(torch.max(actions + noise, self.low_tensor), self.
            high_tensor)


def scaled_noise(size, device):
    x = torch.randn(size, device=device)
    return x.sign().mul_(x.abs().sqrt_())


def factorized_gaussian_noise(in_features, out_features, device):
    """
    Factorised (cheaper) gaussian noise from "Noisy Networks for Exploration"
    by Meire Fortunato, Mohammad Gheshlaghi Azar, Bilal Piot and others
    """
    in_noise = scaled_noise(in_features, device=device)
    out_noise = scaled_noise(out_features, device=device)
    return out_noise.ger(in_noise), out_noise


def gaussian_noise(in_features, out_features, device):
    """ Normal gaussian N(0, 1) noise """
    return torch.randn((in_features, out_features), device=device
        ), torch.randn(out_features, device=device)


class NoisyLinear(nn.Module):
    """ NoisyNets noisy linear layer """

    def __init__(self, in_features, out_features, initial_std_dev: float=
        0.4, factorized_noise: bool=True):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.initial_std_dev = initial_std_dev
        self.factorized_noise = factorized_noise
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features)
            )
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

    def reset_weights(self):
        init.orthogonal_(self.weight_mu, gain=math.sqrt(2))
        init.constant_(self.bias_mu, 0.0)
        self.weight_sigma.data.fill_(self.initial_std_dev / math.sqrt(self.
            in_features))
        self.bias_sigma.data.fill_(self.initial_std_dev / math.sqrt(self.
            out_features))

    def forward(self, input_data):
        if self.training:
            if self.factorized_noise:
                weight_epsilon, bias_epsilon = factorized_gaussian_noise(self
                    .in_features, self.out_features, device=input_data.device)
            else:
                weight_epsilon, bias_epsilon = gaussian_noise(self.
                    in_features, self.out_features, device=input_data.device)
            return F.linear(input_data, self.weight_mu + self.weight_sigma *
                weight_epsilon, self.bias_mu + self.bias_sigma * bias_epsilon)
        else:
            return F.linear(input_data, self.weight_mu, self.bias_mu)

    def extra_repr(self):
        """Set the extra representation of the module

        To print customized extra information, you should reimplement
        this method in your own modules. Both single-line and multi-line
        strings are acceptable.
        """
        return (
            f'{self.in_features}, {self.out_features}, initial_std_dev={self.initial_std_dev}, factorized_noise={{self.factorized_noise}} '
            )


class QDistributionalHead(nn.Module):
    """ Network head calculating Q-function value for each (discrete) action. """

    def __init__(self, input_dim, action_space, vmin: float, vmax: float,
        atoms: int=1):
        super().__init__()
        assert isinstance(action_space, spaces.Discrete)
        assert vmax > vmin
        self.atoms = atoms
        self.vmin = vmin
        self.vmax = vmax
        self.action_space = action_space
        self.action_size = action_space.n
        self.atom_delta = (self.vmax - self.vmin) / (self.atoms - 1)
        self.linear_layer = nn.Linear(input_dim, self.action_size * self.atoms)
        self.register_buffer('support_atoms', torch.linspace(self.vmin,
            self.vmax, self.atoms))

    def histogram_info(self) ->dict:
        """ Return extra information about histogram """
        return {'support_atoms': self.support_atoms, 'atom_delta': self.
            atom_delta, 'vmin': self.vmin, 'vmax': self.vmax, 'num_atoms':
            self.atoms}

    def reset_weights(self):
        init.orthogonal_(self.linear_layer.weight, gain=1.0)
        init.constant_(self.linear_layer.bias, 0.0)

    def forward(self, input_data):
        histogram_logits = self.linear_layer(input_data).view(input_data.
            size(0), self.action_size, self.atoms)
        histogram_log = F.log_softmax(histogram_logits, dim=2)
        return histogram_log

    def sample(self, histogram_logits):
        """ Sample from a greedy strategy with given q-value histogram """
        histogram_probs = histogram_logits.exp()
        atoms = self.support_atoms.view(1, 1, self.atoms)
        return (histogram_probs * atoms).sum(dim=-1).argmax(dim=1)


class QDistributionalNoisyDuelingHead(nn.Module):
    """ Network head calculating Q-function value for each (discrete) action. """

    def __init__(self, input_dim, action_space, vmin: float, vmax: float,
        atoms: int=1, initial_std_dev: float=0.4, factorized_noise: bool=True):
        super().__init__()
        assert isinstance(action_space, spaces.Discrete)
        assert vmax > vmin
        self.atoms = atoms
        self.vmin = vmin
        self.vmax = vmax
        self.action_size = action_space.n
        self.action_space = action_space
        self.atom_delta = (self.vmax - self.vmin) / (self.atoms - 1)
        self.linear_layer_advantage = NoisyLinear(input_dim, self.
            action_size * self.atoms, initial_std_dev=initial_std_dev,
            factorized_noise=factorized_noise)
        self.linear_layer_value = NoisyLinear(input_dim, self.atoms,
            initial_std_dev=initial_std_dev, factorized_noise=factorized_noise)
        self.register_buffer('support_atoms', torch.linspace(self.vmin,
            self.vmax, self.atoms))

    def histogram_info(self) ->dict:
        """ Return extra information about histogram """
        return {'support_atoms': self.support_atoms, 'atom_delta': self.
            atom_delta, 'vmin': self.vmin, 'vmax': self.vmax, 'num_atoms':
            self.atoms}

    def reset_weights(self):
        self.linear_layer_advantage.reset_weights()
        self.linear_layer_value.reset_weights()

    def forward(self, advantage_features, value_features):
        adv = self.linear_layer_advantage(advantage_features).view(-1, self
            .action_size, self.atoms)
        val = self.linear_layer_value(value_features).view(-1, 1, self.atoms)
        histogram_output = val + adv - adv.mean(dim=1, keepdim=True)
        return F.log_softmax(histogram_output, dim=2)

    def sample(self, histogram_logits):
        """ Sample from a greedy strategy with given q-value histogram """
        histogram_probs = histogram_logits.exp()
        atoms = self.support_atoms.view(1, 1, self.atoms)
        return (histogram_probs * atoms).sum(dim=-1).argmax(dim=1)


class QDuelingHead(nn.Module):
    """ Network head calculating Q-function value for each (discrete) action using two separate inputs. """

    def __init__(self, input_dim, action_space):
        super().__init__()
        assert isinstance(action_space, spaces.Discrete)
        self.linear_layer_advantage = nn.Linear(input_dim, action_space.n)
        self.linear_layer_value = nn.Linear(input_dim, 1)
        self.action_space = action_space

    def reset_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.orthogonal_(m.weight, gain=1.0)
                init.constant_(m.bias, 0.0)

    def forward(self, advantage_data, value_data):
        adv = self.linear_layer_advantage(advantage_data)
        value = self.linear_layer_value(value_data)
        return adv - adv.mean(dim=1, keepdim=True) + value

    def sample(self, q_values):
        """ Sample from greedy strategy with given q-values """
        return q_values.argmax(dim=1)


class QHead(nn.Module):
    """ Network head calculating Q-function value for each (discrete) action. """

    def __init__(self, input_dim, action_space):
        super().__init__()
        assert isinstance(action_space, spaces.Discrete)
        self.action_space = action_space
        self.linear_layer = nn.Linear(input_dim, action_space.n)

    def reset_weights(self):
        init.orthogonal_(self.linear_layer.weight, gain=1.0)
        init.constant_(self.linear_layer.bias, 0.0)

    def forward(self, input_data):
        return self.linear_layer(input_data)

    def sample(self, q_values):
        """ Sample from epsilon-greedy strategy with given q-values """
        return q_values.argmax(dim=1)


class QNoisyHead(nn.Module):
    """ Network head calculating Q-function value for each (discrete) action. """

    def __init__(self, input_dim, action_space, initial_std_dev=0.4,
        factorized_noise=True):
        super().__init__()
        assert isinstance(action_space, spaces.Discrete)
        self.action_space = action_space
        self.linear_layer = NoisyLinear(input_dim, action_space.n,
            initial_std_dev=initial_std_dev, factorized_noise=factorized_noise)

    def reset_weights(self):
        self.linear_layer.reset_weights()

    def forward(self, input_data):
        return self.linear_layer(input_data)

    def sample(self, q_values):
        """ Sample from epsilon-greedy strategy with given q-values """
        return q_values.argmax(dim=1)


class ValueHead(nn.Module):
    """ Network head for value determination """

    def __init__(self, input_dim):
        super().__init__()
        self.linear_layer = nn.Linear(input_dim, 1)

    def reset_weights(self):
        init.orthogonal_(self.linear_layer.weight, gain=1.0)
        init.constant_(self.linear_layer.bias, 0.0)

    def forward(self, input_data):
        return self.linear_layer(input_data)[:, (0)]


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_MillionIntegrals_vel(_paritybench_base):
    pass
    def test_000(self):
        self._check(AdaptiveConcatPool2d(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(BasicBlock(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_002(self):
        self._check(Bottleneck(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(CategoricalActionHead(*[], **{'input_dim': 4, 'num_actions': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_004(self):
        self._check(DeterministicCriticHead(*[], **{'input_dim': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_005(self):
        self._check(DiagGaussianActionHead(*[], **{'input_dim': 4, 'num_dimensions': 4}), [torch.rand([4, 4])], {})

    def test_006(self):
        self._check(Flatten(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_007(self):
        self._check(Lambda(*[], **{'f': ReLU()}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_008(self):
        self._check(NoisyLinear(*[], **{'in_features': 4, 'out_features': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_009(self):
        self._check(OneHotEncode(*[], **{'num_classes': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_010(self):
        self._check(ResNeXtBottleneck(*[], **{'in_channels': 4, 'out_channels': 4, 'cardinality': 4, 'divisor': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_011(self):
        self._check(ValueHead(*[], **{'input_dim': 4}), [torch.rand([4, 4, 4, 4])], {})

