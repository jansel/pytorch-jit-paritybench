import sys
_module = sys.modules[__name__]
del sys
docs = _module
conf = _module
advanced_experiment = _module
approximator = _module
ddpg = _module
dqn = _module
generic_regressor = _module
logger = _module
room_env = _module
serialization = _module
simple_experiment = _module
examples = _module
acrobot_a2c = _module
acrobot_dqn = _module
atari_dqn = _module
car_on_hill_fqi = _module
cartpole_lspi = _module
double_chain_q_learning = _module
double_chain = _module
get_env_info = _module
grid_world_td = _module
habitat = _module
habitat_nav_dqn = _module
habitat_rearrange_sac = _module
igibson_dqn = _module
lqr_bbo = _module
lqr_pg = _module
minigrid_dqn = _module
mountain_car_sarsa = _module
pendulum_a2c = _module
pendulum_ac = _module
pendulum_ddpg = _module
pendulum_dpg = _module
pendulum_sac = _module
pendulum_trust_region = _module
plotting_and_normalization = _module
puddle_world_sarsa = _module
segway_test_bbo = _module
ship_steering_bbo = _module
simple_chain_qlearning = _module
taxi_mellow = _module
walker_stand_ddpg = _module
walker_stand_ddpg_shared_net = _module
mushroom_rl = _module
algorithms = _module
actor_critic = _module
classic_actor_critic = _module
copdac_q = _module
stochastic_ac = _module
deep_actor_critic = _module
a2c = _module
deep_actor_critic = _module
ppo = _module
sac = _module
td3 = _module
trpo = _module
policy_search = _module
black_box_optimization = _module
constrained_reps = _module
more = _module
pgpe = _module
reps = _module
rwr = _module
policy_gradient = _module
enac = _module
gpomdp = _module
reinforce = _module
value = _module
batch_td = _module
boosted_fqi = _module
double_fqi = _module
fqi = _module
lspi = _module
abstract_dqn = _module
averaged_dqn = _module
categorical_dqn = _module
double_dqn = _module
dueling_dqn = _module
maxmin_dqn = _module
noisy_dqn = _module
quantile_dqn = _module
rainbow = _module
td = _module
double_q_learning = _module
expected_sarsa = _module
maxmin_q_learning = _module
q_lambda = _module
q_learning = _module
r_learning = _module
rq_learning = _module
sarsa = _module
sarsa_lambda = _module
sarsa_lambda_continuous = _module
speedy_q_learning = _module
true_online_sarsa_lambda = _module
weighted_q_learning = _module
approximators = _module
_implementations = _module
action_regressor = _module
ensemble = _module
q_regressor = _module
parametric = _module
cmac = _module
linear = _module
torch_approximator = _module
regressor = _module
core = _module
agent = _module
environment = _module
console_logger = _module
data_logger = _module
serialization = _module
distributions = _module
distribution = _module
gaussian = _module
environments = _module
atari = _module
car_on_hill = _module
cart_pole = _module
dm_control_env = _module
finite_mdp = _module
generators = _module
grid_world = _module
simple_chain = _module
taxi = _module
gym_env = _module
habitat_env = _module
igibson_env = _module
inverted_pendulum = _module
lqr = _module
minigrid_env = _module
mujoco = _module
mujoco_envs = _module
air_hockey = _module
base = _module
defend = _module
double = _module
hit = _module
prepare = _module
repel = _module
single = _module
ball_in_a_cup = _module
data = _module
meshes = _module
puddle_world = _module
pybullet = _module
pybullet_envs = _module
segway = _module
ship_steering = _module
features = _module
basis_features = _module
features_implementation = _module
functional_features = _module
tiles_features = _module
torch_features = _module
basis = _module
fourier = _module
gaussian_rbf = _module
polynomial = _module
tensors = _module
constant_tensor = _module
gaussian_tensor = _module
random_fourier_tensor = _module
tiles = _module
voronoi = _module
policy = _module
deterministic_policy = _module
gaussian_policy = _module
noise_policy = _module
td_policy = _module
torch_policy = _module
solvers = _module
dynamic_programming = _module
utils = _module
angles = _module
callbacks = _module
callback = _module
collect_dataset = _module
collect_max_q = _module
collect_parameters = _module
collect_q = _module
plot_dataset = _module
dataset = _module
eligibility_trace = _module
folder = _module
frames = _module
minibatches = _module
observation_helper = _module
viewer = _module
numerical_gradient = _module
optimizers = _module
parameters = _module
plot = _module
plots = _module
common_plots = _module
databuffer = _module
plot_item_buffer = _module
window = _module
preprocessors = _module
contacts = _module
index_map = _module
joints_helper = _module
observation = _module
replay_memory = _module
running_stats = _module
spaces = _module
table = _module
value_functions = _module
variance_parameters = _module
setup = _module
utils = _module
test_a2c = _module
test_black_box = _module
test_ddpg = _module
test_dpg = _module
test_dqn = _module
test_fqi = _module
test_lspi = _module
test_policy_gradient = _module
test_sac = _module
test_stochastic_ac = _module
test_td = _module
test_trust_region = _module
test_cmac_approximator = _module
test_linear_approximator = _module
test_torch_approximator = _module
test_core = _module
test_logger = _module
test_distribution_interface = _module
test_gaussian_distribution = _module
test_air_hockey = _module
test_ball_in_a_cup = _module
test_air_hockey_bullet = _module
test_all_envs = _module
test_mujoco = _module
test_features = _module
test_deterministic_policy = _module
test_gaussian_policy = _module
test_noise_policy = _module
test_policy_interface = _module
test_td_policy = _module
test_torch_policy = _module
test_car_on_hill = _module
test_dynamic_programming = _module
test_lqr = _module
test_imports = _module
test_callbacks = _module
test_dataset = _module
test_folder = _module
test_preprocessors = _module

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


import torch.optim as optim


import torch.nn.functional as F


from copy import deepcopy


from itertools import chain


from torch.nn.parameter import Parameter


from sklearn.exceptions import NotFittedError


import warnings


from sklearn.ensemble import ExtraTreesRegressor


import itertools


from torch import optim


from torch import nn


class CriticNetwork(nn.Module):

    def __init__(self, input_shape, output_shape, **kwargs):
        super().__init__()
        n_input = input_shape[-1]
        n_output = output_shape[0]
        self._h = nn.Linear(n_input, n_output)
        nn.init.xavier_uniform_(self._h.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, state, action):
        state_action = torch.cat((state.float(), action.float()), dim=1)
        q = F.relu(self._h(state_action))
        return torch.squeeze(q)


class ActorNetwork(nn.Module):

    def __init__(self, input_shape, output_shape, **kwargs):
        super(ActorNetwork, self).__init__()
        n_input = input_shape[-1]
        n_output = output_shape[0]
        self._h = nn.Linear(n_input, n_output)
        nn.init.xavier_uniform_(self._h.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, state):
        return F.relu(self._h(torch.squeeze(state, 1).float()))


class Network(nn.Module):

    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(Network, self).__init__()
        n_input = input_shape[-1]
        n_output = output_shape[0]
        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)
        nn.init.xavier_uniform_(self._h1.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.xavier_uniform_(self._h2.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.xavier_uniform_(self._h3.weight, gain=nn.init.calculate_gain('linear'))

    def forward(self, state, **kwargs):
        features1 = torch.tanh(self._h1(torch.squeeze(state, -1).float()))
        features2 = torch.tanh(self._h2(features1))
        a = self._h3(features2)
        return a


class FeatureNetwork(nn.Module):

    def __init__(self, input_shape, output_shape, **kwargs):
        super().__init__()

    def forward(self, state, action=None):
        return torch.squeeze(state, 1).float()


class StateEmbedding(nn.Module):

    def __init__(self, input_shape):
        super().__init__()
        self._obs_shape = input_shape
        n_input = input_shape[0]
        self._h1 = nn.Conv2d(n_input, 32, kernel_size=8, stride=3)
        self._h2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self._h3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        dummy_obs = torch.zeros(1, *input_shape)
        self._output_shape = np.prod(self._h3(self._h2(self._h1(dummy_obs))).shape),
        nn.init.xavier_uniform_(self._h1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, state):
        h = state.view(-1, *self._obs_shape).float() / 255.0
        h = F.relu(self._h1(h))
        h = F.relu(self._h2(h))
        h = F.relu(self._h3(h))
        return h


eps = torch.finfo(torch.float32).eps


class CategoricalNetwork(nn.Module):

    def __init__(self, input_shape, output_shape, features_network, n_atoms, v_min, v_max, n_features, use_cuda, **kwargs):
        super().__init__()
        self._n_output = output_shape[0]
        self._phi = features_network(input_shape, (n_features,), n_features=n_features, **kwargs)
        self._n_atoms = n_atoms
        self._v_min = v_min
        self._v_max = v_max
        delta = (self._v_max - self._v_min) / (self._n_atoms - 1)
        self._a_values = torch.arange(self._v_min, self._v_max + eps, delta)
        if use_cuda:
            self._a_values = self._a_values
        self._p = nn.ModuleList([nn.Linear(n_features, n_atoms) for _ in range(self._n_output)])
        for i in range(self._n_output):
            nn.init.xavier_uniform_(self._p[i].weight, gain=nn.init.calculate_gain('linear'))

    def forward(self, state, action=None, get_distribution=False):
        features = self._phi(state)
        a_p = [F.softmax(self._p[i](features), -1) for i in range(self._n_output)]
        a_p = torch.stack(a_p, dim=1)
        if not get_distribution:
            q = torch.empty(a_p.shape[:-1])
            for i in range(a_p.shape[0]):
                q[i] = a_p[i] @ self._a_values
            if action is not None:
                return torch.squeeze(q.gather(1, action))
            else:
                return q
        elif action is not None:
            action = torch.unsqueeze(action.long(), 2).repeat(1, 1, self._n_atoms)
            return torch.squeeze(a_p.gather(1, action))
        else:
            return a_p


class DuelingNetwork(nn.Module):

    def __init__(self, input_shape, output_shape, features_network, n_features, avg_advantage, **kwargs):
        super().__init__()
        self._avg_advantage = avg_advantage
        self._n_output = output_shape[0]
        self._phi = features_network(input_shape, (n_features,), n_features=n_features, **kwargs)
        self._A = nn.Linear(n_features, self._n_output)
        self._V = nn.Linear(n_features, 1)
        nn.init.xavier_uniform_(self._A.weight, gain=nn.init.calculate_gain('linear'))
        nn.init.xavier_uniform_(self._V.weight, gain=nn.init.calculate_gain('linear'))

    def forward(self, state, action=None):
        features = self._phi(state)
        advantage = self._A(features)
        value = self._V(features)
        q = value + advantage
        if self._avg_advantage:
            q -= advantage.mean(1).reshape(-1, 1)
        else:
            q -= advantage.max(1).reshape(-1, 1)
        if action is None:
            return q
        else:
            q_acted = torch.squeeze(q.gather(1, action.long()))
            return q_acted


class NoisyNetwork(nn.Module):


    class NoisyLinear(nn.Module):
        __constants__ = ['in_features', 'out_features']

        def __init__(self, in_features, out_features, use_cuda, sigma_coeff=0.5, bias=True):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.mu_weight = Parameter(torch.Tensor(out_features, in_features))
            self.sigma_weight = Parameter(torch.Tensor(out_features, in_features))
            if bias:
                self.mu_bias = Parameter(torch.Tensor(out_features))
                self.sigma_bias = Parameter(torch.Tensor(out_features))
            else:
                self.register_parameter('bias', None)
            self._use_cuda = use_cuda
            self._sigma_coeff = sigma_coeff
            self.reset_parameters()

        def reset_parameters(self):
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.mu_weight)
            fan_in **= 0.5
            bound_weight = 1 / fan_in
            bound_sigma = self._sigma_coeff / fan_in
            nn.init.uniform_(self.mu_weight, -bound_weight, bound_weight)
            nn.init.constant_(self.sigma_weight, bound_sigma)
            if hasattr(self, 'mu_bias'):
                nn.init.uniform_(self.mu_bias, -bound_weight, bound_weight)
                nn.init.constant_(self.sigma_bias, bound_sigma)

        def forward(self, input):
            eps_output = torch.rand(self.mu_weight.shape[0], 1)
            eps_input = torch.rand(1, self.mu_weight.shape[1])
            if self._use_cuda:
                eps_output = eps_output
                eps_input = eps_input
            eps_dot = torch.matmul(self._noise(eps_output), self._noise(eps_input))
            weight = self.mu_weight + self.sigma_weight * eps_dot
            if hasattr(self, 'mu_bias'):
                self.bias = self.mu_bias + self.sigma_bias * self._noise(eps_output[:, 0])
            return F.linear(input, weight, self.bias)

        @staticmethod
        def _noise(x):
            return torch.sign(x) * torch.sqrt(torch.abs(x))

        def extra_repr(self):
            return 'in_features={}, out_features={}, mu_bias={}, sigma_bias={}'.format(self.in_features, self.out_features, self.mu_bias, self.sigma_bias is not None)

    def __init__(self, input_shape, output_shape, features_network, n_features, use_cuda, **kwargs):
        super().__init__()
        self._n_output = output_shape[0]
        self._phi = features_network(input_shape, (n_features,), n_features=n_features, **kwargs)
        self._Q = self.NoisyLinear(n_features, self._n_output, use_cuda)

    def forward(self, state, action=None):
        features = self._phi(state)
        q = self._Q(features)
        if action is None:
            return q
        else:
            q_acted = torch.squeeze(q.gather(1, action.long()))
            return q_acted


class QuantileNetwork(nn.Module):

    def __init__(self, input_shape, output_shape, features_network, n_quantiles, n_features, **kwargs):
        super().__init__()
        self._n_output = output_shape[0]
        self._phi = features_network(input_shape, (n_features,), n_features=n_features, **kwargs)
        self._n_quantiles = n_quantiles
        self._quant = nn.ModuleList([nn.Linear(n_features, n_quantiles) for _ in range(self._n_output)])
        for i in range(self._n_output):
            nn.init.xavier_uniform_(self._quant[i].weight, gain=nn.init.calculate_gain('linear'))

    def forward(self, state, action=None, get_quantiles=False):
        features = self._phi(state)
        a_quant = [self._quant[i](features) for i in range(self._n_output)]
        a_quant = torch.stack(a_quant, dim=1)
        if not get_quantiles:
            quant = a_quant.mean(-1)
            if action is not None:
                return torch.squeeze(quant.gather(1, action))
            else:
                return quant
        elif action is not None:
            action = torch.unsqueeze(action.long(), 2).repeat(1, 1, self._n_quantiles)
            return torch.squeeze(a_quant.gather(1, action))
        else:
            return a_quant


class RainbowNetwork(nn.Module):

    def __init__(self, input_shape, output_shape, features_network, n_atoms, v_min, v_max, n_features, use_cuda, sigma_coeff, **kwargs):
        super().__init__()
        self._n_output = output_shape[0]
        self._phi = features_network(input_shape, (n_features,), n_features=n_features, **kwargs)
        self._n_atoms = n_atoms
        self._v_min = v_min
        self._v_max = v_max
        delta = (self._v_max - self._v_min) / (self._n_atoms - 1)
        self._a_values = torch.arange(self._v_min, self._v_max + eps, delta)
        if use_cuda:
            self._a_values = self._a_values
        self._pv = NoisyNetwork.NoisyLinear(n_features, n_atoms, use_cuda, sigma_coeff)
        self._pa = nn.ModuleList([NoisyNetwork.NoisyLinear(n_features, n_atoms, use_cuda, sigma_coeff) for _ in range(self._n_output)])

    def forward(self, state, action=None, get_distribution=False):
        features = self._phi(state)
        a_pv = self._pv(features)
        a_pa = [self._pa[i](features) for i in range(self._n_output)]
        a_pa = torch.stack(a_pa, dim=1)
        a_pv = a_pv.unsqueeze(1).repeat(1, self._n_output, 1)
        mean_a_pa = a_pa.mean(1, keepdim=True).repeat(1, self._n_output, 1)
        softmax = F.softmax(a_pv + a_pa - mean_a_pa, dim=-1)
        if not get_distribution:
            q = torch.empty(softmax.shape[:-1])
            for i in range(softmax.shape[0]):
                q[i] = softmax[i] @ self._a_values
            if action is not None:
                return torch.squeeze(q.gather(1, action))
            else:
                return q
        elif action is not None:
            action = torch.unsqueeze(action.long(), 2).repeat(1, 1, self._n_atoms)
            return torch.squeeze(softmax.gather(1, action))
        else:
            return softmax


class ConstantTensor(nn.Module):
    """
    Pytorch module to implement a constant function (always one).

    """

    def forward(self, x):
        return torch.ones(x.shape[0], 1)

    @property
    def size(self):
        return 1


def to_float_tensor(x, use_cuda=False):
    """
    Function used to convert a numpy array to a float torch tensor.

    Args:
        x (np.ndarray): numpy array to be converted as torch tensor;
        use_cuda (bool): whether to build a cuda tensors or not.

    Returns:
        A float tensor build from the values contained in the input array.

    """
    x = torch.tensor(x, dtype=torch.float)
    return x if use_cuda else x


def to_int_tensor(x, use_cuda=False):
    """
    Function used to convert a numpy array to a float torch tensor.

    Args:
        x (np.ndarray): numpy array to be converted as torch tensor;
        use_cuda (bool): whether to build a cuda tensors or not.

    Returns:
        A float tensor build from the values contained in the input array.

    """
    x = torch.tensor(x, dtype=torch.int)
    return x if use_cuda else x


def uniform_grid(n_centers, low, high):
    """
    This function is used to create the parameters of uniformly spaced radial
    basis functions with 25% of overlap. It creates a uniformly spaced grid of
    ``n_centers[i]`` points in each ``ranges[i]``. Also returns a vector
    containing the appropriate scales of the radial basis functions.

    Args:
         n_centers (list): number of centers of each dimension;
         low (np.ndarray): lowest value for each dimension;
         high (np.ndarray): highest value for each dimension.

    Returns:
        The uniformly spaced grid and the scale vector.

    """
    n_features = len(low)
    b = np.zeros(n_features)
    c = list()
    tot_points = 1
    for i, n in enumerate(n_centers):
        start = low[i]
        end = high[i]
        b[i] = (end - start) ** 2 / n ** 3
        m = abs(start - end) / n
        if n == 1:
            c_i = (start + end) / 2.0
            c.append(np.array([c_i]))
        else:
            c_i = np.linspace(start - m * 0.1, end + m * 0.1, n)
            c.append(c_i)
        tot_points *= n
    n_rows = 1
    n_cols = 0
    grid = np.zeros((tot_points, n_features))
    for discrete_values in c:
        i1 = 0
        dim = len(discrete_values)
        for i in range(dim):
            for r in range(n_rows):
                idx_r = r + i * n_rows
                for c in range(n_cols):
                    grid[idx_r, c] = grid[r, c]
                grid[idx_r, n_cols] = discrete_values[i1]
            i1 += 1
        n_cols += 1
        n_rows *= len(discrete_values)
    return grid, b


class GaussianRBFTensor(nn.Module):
    """
    Pytorch module to implement a gaussian radial basis function.

    """

    def __init__(self, mu, scale, dim, use_cuda):
        """
        Constructor.

        Args:
            mu (np.ndarray): centers of the gaussian RBFs;
            scale (np.ndarray): scales for the RBFs;
            dim (np.ndarray): list of dimension to be considered for the computation of the features;
            use_cuda (bool): whether to use cuda for the computation or not.

        """
        self._mu = to_float_tensor(mu, use_cuda)
        self._scale = to_float_tensor(scale, use_cuda)
        if dim is not None:
            self._dim = to_int_tensor(dim, use_cuda)
        else:
            self._dim = None
        self._use_cuda = use_cuda

    def forward(self, x):
        if self._use_cuda:
            x = x
        if self._dim is not None:
            x = torch.index_select(x, 1, self._dim)
        x = x.unsqueeze(1).repeat(1, self._mu.shape[0], 1)
        delta = x - self._mu.repeat(x.shape[0], 1, 1)
        return torch.exp(-torch.sum(delta ** 2 / self._scale, -1)).squeeze(-1)

    @staticmethod
    def generate(n_centers, low, high, dimensions=None, use_cuda=False):
        """
        Factory method that generates the list of dictionaries to build the
        tensors representing a set of uniformly spaced Gaussian radial basis
        functions with a 25% overlap.

        Args:
            n_centers (list): list of the number of radial basis functions to be
                              used for each dimension;
            low (np.ndarray): lowest value for each dimension;
            high (np.ndarray): highest value for each dimension;
            dimensions (list, None): list of the dimensions of the input to be
                considered by the feature. The number of dimensions must match
                the number of elements in ``n_centers`` and ``low``;
            use_cuda (bool): whether to use cuda for the computation or not.

        Returns:
            The list of dictionaries as described above.

        """
        n_features = len(low)
        assert len(n_centers) == n_features
        assert len(low) == len(high)
        assert dimensions is None or n_features == len(dimensions)
        mu, scale = uniform_grid(n_centers, low, high)
        tensor_list = [GaussianRBFTensor(mu, scale, dimensions, use_cuda)]
        return tensor_list

    @property
    def size(self):
        return self._mu.shape[0]


class RandomFourierBasis(nn.Module):
    """
    Class implementing Random Fourier basis functions. The value of the feature
    is computed using the formula:

    .. math::
        \\sin{\\dfrac{PX}{\\nu}+\\varphi}

    where X is the input, m is the vector of the minumum input values (for each
    dimensions) , \\Delta is the vector of maximum

    This features have been presented in:

    "Towards generalization and simplicity in continuous control". Rajeswaran A. et Al..
    2017.

    """

    def __init__(self, P, phi, nu, use_cuda):
        """
        Constructor.

        Args:
            P (np.ndarray): weights matrix, every weight should be drawn from a normal distribution;
            phi (np.ndarray): bias vector, every weight should be drawn from a uniform distribution in the interval
                [-\\pi, \\pi);
             values of the input variables, i.e. delta = high - low;
            nu (float):  bandwidth parameter, it should be chosen approximately as the average pairwise distances
                between different observation vectors;
            use_cuda (bool): whether to use cuda for the computation or not.

        """
        self._P = to_float_tensor(P, use_cuda)
        self._phi = to_float_tensor(phi, use_cuda)
        self._nu = nu
        self._use_cuda = use_cuda

    def forward(self, x):
        if self._use_cuda:
            x = x
        return torch.sin(x @ self._P / self._nu + self._phi)

    def __str__(self):
        return str(self._P) + ' ' + str(self._phi)

    @staticmethod
    def generate(nu, n_output, input_size, use_cuda=False, use_bias=True):
        """
        Factory method to build random fourier basis. Includes a constant tensor into the output.

        Args:
            nu (float):  bandwidth parameter, it should be chosen approximately as the average pairwise distances
                between different observation vectors.
            n_output (int): number of basis to use;
            input_size (int): size of the input;
            use_cuda (bool): whether to use cuda for the computation or not.

        Returns:
            The list of the generated fourier basis functions.

        """
        if use_bias:
            n_output -= 1
        P = np.random.randn(input_size, n_output)
        phi = np.random.uniform(-np.pi, np.pi, n_output)
        tensor_list = [RandomFourierBasis(P, phi, nu, use_cuda)]
        if use_bias:
            tensor_list.append(ConstantTensor())
        return tensor_list

    @property
    def size(self):
        return self._phi.shape[0]


class ExampleNet(nn.Module):

    def __init__(self, input_shape, output_shape, **kwargs):
        super(ExampleNet, self).__init__()
        self._q = nn.Linear(input_shape[0], output_shape[0])
        nn.init.xavier_uniform_(self._q.weight, gain=nn.init.calculate_gain('linear'))

    def forward(self, x, a=None):
        x = x.float()
        q = self._q(x)
        if a is None:
            return q
        else:
            action = a.long()
            q_acted = torch.squeeze(q.gather(1, action))
            return q_acted


class LinearNetwork(nn.Module):

    def __init__(self, input_shape, output_shape, **kwargs):
        super().__init__()
        n_input = input_shape[-1]
        n_output = output_shape[0]
        self._h1 = nn.Linear(n_input, n_output)
        nn.init.xavier_uniform_(self._h1.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, state, action=None):
        q = F.relu(self._h1(torch.squeeze(state, 1).float()))
        if action is None:
            return q
        else:
            action = action.long()
            q_acted = torch.squeeze(q.gather(1, action))
            return q_acted


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ActorNetwork,
     lambda: ([], {'input_shape': [4, 4], 'output_shape': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConstantTensor,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CriticNetwork,
     lambda: ([], {'input_shape': [4, 4], 'output_shape': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (ExampleNet,
     lambda: ([], {'input_shape': [4, 4], 'output_shape': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FeatureNetwork,
     lambda: ([], {'input_shape': 4, 'output_shape': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LinearNetwork,
     lambda: ([], {'input_shape': [4, 4], 'output_shape': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Network,
     lambda: ([], {'input_shape': [4, 4], 'output_shape': [4, 4], 'n_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_MushroomRL_mushroom_rl(_paritybench_base):
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

