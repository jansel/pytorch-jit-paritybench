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
simple_experiment = _module
examples = _module
acrobot_dqn = _module
atari_dqn = _module
car_on_hill_fqi = _module
cartpole_lspi = _module
double_chain_q_learning = _module
double_chain = _module
grid_world_td = _module
humanoid_sac = _module
lqr_bbo = _module
lqr_pg = _module
mountain_car_sarsa = _module
pendulum_a2c = _module
pendulum_ac = _module
pendulum_ddpg = _module
pendulum_dpg = _module
pendulum_sac = _module
pendulum_trust_region = _module
plotting_and_normalization = _module
segway_test_bbo = _module
ship_steering_bbo = _module
simple_chain_qlearning = _module
taxi_mellow = _module
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
agent = _module
policy_search = _module
black_box_optimization = _module
pgpe = _module
reps = _module
rwr = _module
policy_gradient = _module
enac = _module
gpomdp = _module
reinforce = _module
value = _module
batch_td = _module
fqi = _module
lspi = _module
categorical_dqn = _module
td = _module
double_q_learning = _module
expected_sarsa = _module
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
linear = _module
torch_approximator = _module
regressor = _module
core = _module
distributions = _module
distribution = _module
gaussian = _module
environments = _module
atari = _module
car_on_hill = _module
cart_pole = _module
dm_control_env = _module
environment = _module
finite_mdp = _module
generators = _module
grid_world = _module
simple_chain = _module
taxi = _module
gym_env = _module
inverted_pendulum = _module
lqr = _module
mujoco = _module
mujoco_envs = _module
ball_in_a_cup = _module
humanoid_gait = _module
_external_simulation = _module
human_muscle = _module
mtc_model = _module
muscle_simulation = _module
reward_goals = _module
reward = _module
trajectory = _module
velocity_profile = _module
utils = _module
puddle_world = _module
segway = _module
ship_steering = _module
features = _module
basis_features = _module
features_implementation = _module
functional_features = _module
pytorch_features = _module
tiles_features = _module
basis = _module
fourier = _module
gaussian_rbf = _module
polynomial = _module
tensors = _module
gaussian_tensor = _module
tiles = _module
policy = _module
deterministic_policy = _module
gaussian_policy = _module
noise_policy = _module
td_policy = _module
torch_policy = _module
solvers = _module
dynamic_programming = _module
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
minibatches = _module
numerical_gradient = _module
parameters = _module
plots = _module
common_plots = _module
databuffer = _module
plot_item_buffer = _module
window = _module
preprocessors = _module
replay_memory = _module
running_stats = _module
spaces = _module
table = _module
value_functions = _module
variance_parameters = _module
viewer = _module
setup = _module
utils = _module
test_a2c = _module
test_batch_td = _module
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
test_linear_approximator = _module
test_torch_approximator = _module
test_distribution_interface = _module
test_gaussian_distribution = _module
test_ball_in_a_cup = _module
test_humanoid_gait = _module
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
test_imports = _module
test_callbacks = _module
test_dataset = _module
test_folder = _module
test_preprocessors = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


class CategoricalNetwork(nn.Module):

    def __init__(self, input_shape, output_shape, features_network, n_atoms, v_min, v_max, n_features, use_cuda, **kwargs):
        super().__init__()
        self._n_output = output_shape[0]
        self._phi = features_network(input_shape, (n_features,), n_features=n_features, **kwargs)
        self._n_atoms = n_atoms
        self._v_min = v_min
        self._v_max = v_max
        delta = (self._v_max - self._v_min) / (self._n_atoms - 1)
        self._a_values = torch.arange(self._v_min, self._v_max + delta, delta)
        if use_cuda:
            self._a_values = self._a_values
        self._p = nn.ModuleList([nn.Linear(n_features, n_atoms) for _ in range(self._n_output)])
        for i in range(self._n_output):
            nn.init.xavier_uniform_(self._p[i].weight, gain=nn.init.calculate_gain('linear'))

    def forward(self, state, action=None, get_distribution=False):
        features = self._phi(state, action)
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


class PyTorchGaussianRBF(nn.Module):
    """
    Pytorch module to implement a gaussian radial basis function.

    """

    def __init__(self, mu, scale, dim):
        self._mu = torch.from_numpy(mu)
        self._scale = torch.from_numpy(scale)
        if dim is not None:
            self._dim = torch.from_numpy(dim)
        else:
            self._dim = None

    def forward(self, x):
        if self._dim is not None:
            x = torch.index_select(x, 1, self._dim)
        delta = x - self._mu
        return torch.exp(-torch.sum(delta ** 2 / self._scale, 1))

    @staticmethod
    def generate(n_centers, low, high, dimensions=None):
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
                the number of elements in ``n_centers`` and ``low``.

        Returns:
            The list of dictionaries as described above.

        """
        n_features = len(low)
        assert len(n_centers) == n_features
        assert len(low) == len(high)
        assert dimensions is None or n_features == len(dimensions)
        grid, scale = uniform_grid(n_centers, low, high)
        tensor_list = list()
        for i in range(len(grid)):
            mu = grid[(i), :]
            tensor_list.append(PyTorchGaussianRBF(mu, scale, dimensions))
        return tensor_list


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


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ActorNetwork,
     lambda: ([], {'input_shape': [4, 4], 'output_shape': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CriticNetwork,
     lambda: ([], {'input_shape': [4, 4], 'output_shape': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (ExampleNet,
     lambda: ([], {'input_shape': [4, 4], 'output_shape': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (FeatureNetwork,
     lambda: ([], {'input_shape': 4, 'output_shape': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
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

