import sys
_module = sys.modules[__name__]
del sys
deep_rl = _module
A2C_agent = _module
BaseAgent = _module
CategoricalDQN_agent = _module
DDPG_agent = _module
DQN_agent = _module
NStepDQN_agent = _module
OptionCritic_agent = _module
PPO_agent = _module
QuantileRegressionDQN_agent = _module
TD3_agent = _module
agent = _module
component = _module
envs = _module
random_process = _module
replay = _module
network = _module
network_bodies = _module
network_heads = _module
network_utils = _module
utils = _module
config = _module
logger = _module
misc = _module
normalizer = _module
plot = _module
schedule = _module
torch_utils = _module
examples = _module
setup = _module
template_jobs = _module
template_plot = _module

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


import torch.nn as nn


import torch.nn.functional as F


import numpy as np


def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer


class NatureConvBody(nn.Module):

    def __init__(self, in_channels=4):
        super(NatureConvBody, self).__init__()
        self.feature_dim = 512
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=8,
            stride=4))
        self.conv2 = layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.conv3 = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))
        self.fc4 = layer_init(nn.Linear(7 * 7 * 64, self.feature_dim))

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = y.view(y.size(0), -1)
        y = F.relu(self.fc4(y))
        return y


class DDPGConvBody(nn.Module):

    def __init__(self, in_channels=4):
        super(DDPGConvBody, self).__init__()
        self.feature_dim = 39 * 39 * 32
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=3,
            stride=2))
        self.conv2 = layer_init(nn.Conv2d(32, 32, kernel_size=3))

    def forward(self, x):
        y = F.elu(self.conv1(x))
        y = F.elu(self.conv2(y))
        y = y.view(y.size(0), -1)
        return y


class FCBody(nn.Module):

    def __init__(self, state_dim, hidden_units=(64, 64), gate=F.relu):
        super(FCBody, self).__init__()
        dims = (state_dim,) + hidden_units
        self.layers = nn.ModuleList([layer_init(nn.Linear(dim_in, dim_out)) for
            dim_in, dim_out in zip(dims[:-1], dims[1:])])
        self.gate = gate
        self.feature_dim = dims[-1]

    def forward(self, x):
        for layer in self.layers:
            x = self.gate(layer(x))
        return x


class TwoLayerFCBodyWithAction(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_units=(64, 64), gate=F
        .relu):
        super(TwoLayerFCBodyWithAction, self).__init__()
        hidden_size1, hidden_size2 = hidden_units
        self.fc1 = layer_init(nn.Linear(state_dim, hidden_size1))
        self.fc2 = layer_init(nn.Linear(hidden_size1 + action_dim,
            hidden_size2))
        self.gate = gate
        self.feature_dim = hidden_size2

    def forward(self, x, action):
        x = self.gate(self.fc1(x))
        phi = self.gate(self.fc2(torch.cat([x, action], dim=1)))
        return phi


class OneLayerFCBodyWithAction(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_units, gate=F.relu):
        super(OneLayerFCBodyWithAction, self).__init__()
        self.fc_s = layer_init(nn.Linear(state_dim, hidden_units))
        self.fc_a = layer_init(nn.Linear(action_dim, hidden_units))
        self.gate = gate
        self.feature_dim = hidden_units * 2

    def forward(self, x, action):
        phi = self.gate(torch.cat([self.fc_s(x), self.fc_a(action)], dim=1))
        return phi


class DummyBody(nn.Module):

    def __init__(self, state_dim):
        super(DummyBody, self).__init__()
        self.feature_dim = state_dim

    def forward(self, x):
        return x


class BaseNet:

    def __init__(self):
        pass


class BaseNormalizer:

    def __init__(self, read_only=False):
        self.read_only = read_only

    def set_read_only(self):
        self.read_only = True

    def unset_read_only(self):
        self.read_only = False

    def state_dict(self):
        return None

    def load_state_dict(self, _):
        return


class RescaleNormalizer(BaseNormalizer):

    def __init__(self, coef=1.0):
        BaseNormalizer.__init__(self)
        self.coef = coef

    def __call__(self, x):
        if not isinstance(x, torch.Tensor):
            x = np.asarray(x)
        return self.coef * x


class Config:
    DEVICE = torch.device('cpu')

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.task_fn = None
        self.optimizer_fn = None
        self.actor_optimizer_fn = None
        self.critic_optimizer_fn = None
        self.network_fn = None
        self.actor_network_fn = None
        self.critic_network_fn = None
        self.replay_fn = None
        self.random_process_fn = None
        self.discount = None
        self.target_network_update_freq = None
        self.exploration_steps = None
        self.log_level = 0
        self.history_length = None
        self.double_q = False
        self.tag = 'vanilla'
        self.num_workers = 1
        self.gradient_clip = None
        self.entropy_weight = 0
        self.use_gae = False
        self.gae_tau = 1.0
        self.target_network_mix = 0.001
        self.state_normalizer = RescaleNormalizer()
        self.reward_normalizer = RescaleNormalizer()
        self.min_memory_size = None
        self.max_steps = 0
        self.rollout_length = None
        self.value_loss_weight = 1.0
        self.iteration_log_interval = 30
        self.categorical_v_min = None
        self.categorical_v_max = None
        self.categorical_n_atoms = 51
        self.num_quantiles = None
        self.optimization_epochs = 4
        self.mini_batch_size = 64
        self.termination_regularizer = 0
        self.sgd_update_frequency = None
        self.random_action_prob = None
        self.__eval_env = None
        self.log_interval = int(1000.0)
        self.save_interval = 0
        self.eval_interval = 0
        self.eval_episodes = 10
        self.async_actor = True
        self.tasks = False

    @property
    def eval_env(self):
        return self.__eval_env

    @eval_env.setter
    def eval_env(self, env):
        self.__eval_env = env
        self.state_dim = env.state_dim
        self.action_dim = env.action_dim
        self.task_name = env.name

    def add_argument(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)

    def merge(self, config_dict=None):
        if config_dict is None:
            args = self.parser.parse_args()
            config_dict = args.__dict__
        for key in config_dict.keys():
            setattr(self, key, config_dict[key])


def tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    x = np.asarray(x, dtype=np.float)
    x = torch.tensor(x, device=Config.DEVICE, dtype=torch.float32)
    return x


class VanillaNet(nn.Module, BaseNet):

    def __init__(self, output_dim, body):
        super(VanillaNet, self).__init__()
        self.fc_head = layer_init(nn.Linear(body.feature_dim, output_dim))
        self.body = body
        self

    def forward(self, x):
        phi = self.body(tensor(x))
        y = self.fc_head(phi)
        return y


class DuelingNet(nn.Module, BaseNet):

    def __init__(self, action_dim, body):
        super(DuelingNet, self).__init__()
        self.fc_value = layer_init(nn.Linear(body.feature_dim, 1))
        self.fc_advantage = layer_init(nn.Linear(body.feature_dim, action_dim))
        self.body = body
        self

    def forward(self, x, to_numpy=False):
        phi = self.body(tensor(x))
        value = self.fc_value(phi)
        advantange = self.fc_advantage(phi)
        q = value.expand_as(advantange) + (advantange - advantange.mean(1,
            keepdim=True).expand_as(advantange))
        return q


class CategoricalNet(nn.Module, BaseNet):

    def __init__(self, action_dim, num_atoms, body):
        super(CategoricalNet, self).__init__()
        self.fc_categorical = layer_init(nn.Linear(body.feature_dim, 
            action_dim * num_atoms))
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.body = body
        self

    def forward(self, x):
        phi = self.body(tensor(x))
        pre_prob = self.fc_categorical(phi).view((-1, self.action_dim, self
            .num_atoms))
        prob = F.softmax(pre_prob, dim=-1)
        log_prob = F.log_softmax(pre_prob, dim=-1)
        return prob, log_prob


class QuantileNet(nn.Module, BaseNet):

    def __init__(self, action_dim, num_quantiles, body):
        super(QuantileNet, self).__init__()
        self.fc_quantiles = layer_init(nn.Linear(body.feature_dim, 
            action_dim * num_quantiles))
        self.action_dim = action_dim
        self.num_quantiles = num_quantiles
        self.body = body
        self

    def forward(self, x):
        phi = self.body(tensor(x))
        quantiles = self.fc_quantiles(phi)
        quantiles = quantiles.view((-1, self.action_dim, self.num_quantiles))
        return quantiles


class OptionCriticNet(nn.Module, BaseNet):

    def __init__(self, body, action_dim, num_options):
        super(OptionCriticNet, self).__init__()
        self.fc_q = layer_init(nn.Linear(body.feature_dim, num_options))
        self.fc_pi = layer_init(nn.Linear(body.feature_dim, num_options *
            action_dim))
        self.fc_beta = layer_init(nn.Linear(body.feature_dim, num_options))
        self.num_options = num_options
        self.action_dim = action_dim
        self.body = body
        self

    def forward(self, x):
        phi = self.body(tensor(x))
        q = self.fc_q(phi)
        beta = F.sigmoid(self.fc_beta(phi))
        pi = self.fc_pi(phi)
        pi = pi.view(-1, self.num_options, self.action_dim)
        log_pi = F.log_softmax(pi, dim=-1)
        pi = F.softmax(pi, dim=-1)
        return {'q': q, 'beta': beta, 'log_pi': log_pi, 'pi': pi}


class DeterministicActorCriticNet(nn.Module, BaseNet):

    def __init__(self, state_dim, action_dim, actor_opt_fn, critic_opt_fn,
        phi_body=None, actor_body=None, critic_body=None):
        super(DeterministicActorCriticNet, self).__init__()
        if phi_body is None:
            phi_body = DummyBody(state_dim)
        if actor_body is None:
            actor_body = DummyBody(phi_body.feature_dim)
        if critic_body is None:
            critic_body = DummyBody(phi_body.feature_dim)
        self.phi_body = phi_body
        self.actor_body = actor_body
        self.critic_body = critic_body
        self.fc_action = layer_init(nn.Linear(actor_body.feature_dim,
            action_dim), 0.001)
        self.fc_critic = layer_init(nn.Linear(critic_body.feature_dim, 1), 
            0.001)
        self.actor_params = list(self.actor_body.parameters()) + list(self.
            fc_action.parameters())
        self.critic_params = list(self.critic_body.parameters()) + list(self
            .fc_critic.parameters())
        self.phi_params = list(self.phi_body.parameters())
        self.actor_opt = actor_opt_fn(self.actor_params + self.phi_params)
        self.critic_opt = critic_opt_fn(self.critic_params + self.phi_params)
        self

    def forward(self, obs):
        phi = self.feature(obs)
        action = self.actor(phi)
        return action

    def feature(self, obs):
        obs = tensor(obs)
        return self.phi_body(obs)

    def actor(self, phi):
        return torch.tanh(self.fc_action(self.actor_body(phi)))

    def critic(self, phi, a):
        return self.fc_critic(self.critic_body(phi, a))


class GaussianActorCriticNet(nn.Module, BaseNet):

    def __init__(self, state_dim, action_dim, phi_body=None, actor_body=
        None, critic_body=None):
        super(GaussianActorCriticNet, self).__init__()
        if phi_body is None:
            phi_body = DummyBody(state_dim)
        if actor_body is None:
            actor_body = DummyBody(phi_body.feature_dim)
        if critic_body is None:
            critic_body = DummyBody(phi_body.feature_dim)
        self.phi_body = phi_body
        self.actor_body = actor_body
        self.critic_body = critic_body
        self.fc_action = layer_init(nn.Linear(actor_body.feature_dim,
            action_dim), 0.001)
        self.fc_critic = layer_init(nn.Linear(critic_body.feature_dim, 1), 
            0.001)
        self.std = nn.Parameter(torch.zeros(action_dim))
        self.phi_params = list(self.phi_body.parameters())
        self.actor_params = list(self.actor_body.parameters()) + list(self.
            fc_action.parameters()) + self.phi_params
        self.actor_params.append(self.std)
        self.critic_params = list(self.critic_body.parameters()) + list(self
            .fc_critic.parameters()) + self.phi_params
        self

    def forward(self, obs, action=None):
        obs = tensor(obs)
        phi = self.phi_body(obs)
        phi_a = self.actor_body(phi)
        phi_v = self.critic_body(phi)
        mean = torch.tanh(self.fc_action(phi_a))
        v = self.fc_critic(phi_v)
        dist = torch.distributions.Normal(mean, F.softplus(self.std))
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1).unsqueeze(-1)
        entropy = dist.entropy().sum(-1).unsqueeze(-1)
        return {'a': action, 'log_pi_a': log_prob, 'ent': entropy, 'mean':
            mean, 'v': v}


class CategoricalActorCriticNet(nn.Module, BaseNet):

    def __init__(self, state_dim, action_dim, phi_body=None, actor_body=
        None, critic_body=None):
        super(CategoricalActorCriticNet, self).__init__()
        if phi_body is None:
            phi_body = DummyBody(state_dim)
        if actor_body is None:
            actor_body = DummyBody(phi_body.feature_dim)
        if critic_body is None:
            critic_body = DummyBody(phi_body.feature_dim)
        self.phi_body = phi_body
        self.actor_body = actor_body
        self.critic_body = critic_body
        self.fc_action = layer_init(nn.Linear(actor_body.feature_dim,
            action_dim), 0.001)
        self.fc_critic = layer_init(nn.Linear(critic_body.feature_dim, 1), 
            0.001)
        self.actor_params = list(self.actor_body.parameters()) + list(self.
            fc_action.parameters())
        self.critic_params = list(self.critic_body.parameters()) + list(self
            .fc_critic.parameters())
        self.phi_params = list(self.phi_body.parameters())
        self

    def forward(self, obs, action=None):
        obs = tensor(obs)
        phi = self.phi_body(obs)
        phi_a = self.actor_body(phi)
        phi_v = self.critic_body(phi)
        logits = self.fc_action(phi_a)
        v = self.fc_critic(phi_v)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).unsqueeze(-1)
        entropy = dist.entropy().unsqueeze(-1)
        return {'a': action, 'log_pi_a': log_prob, 'ent': entropy, 'v': v}


class TD3Net(nn.Module, BaseNet):

    def __init__(self, action_dim, actor_body_fn, critic_body_fn,
        actor_opt_fn, critic_opt_fn):
        super(TD3Net, self).__init__()
        self.actor_body = actor_body_fn()
        self.critic_body_1 = critic_body_fn()
        self.critic_body_2 = critic_body_fn()
        self.fc_action = layer_init(nn.Linear(self.actor_body.feature_dim,
            action_dim), 0.001)
        self.fc_critic_1 = layer_init(nn.Linear(self.critic_body_1.
            feature_dim, 1), 0.001)
        self.fc_critic_2 = layer_init(nn.Linear(self.critic_body_2.
            feature_dim, 1), 0.001)
        self.actor_params = list(self.actor_body.parameters()) + list(self.
            fc_action.parameters())
        self.critic_params = list(self.critic_body_1.parameters()) + list(self
            .fc_critic_1.parameters()) + list(self.critic_body_2.parameters()
            ) + list(self.fc_critic_2.parameters())
        self.actor_opt = actor_opt_fn(self.actor_params)
        self.critic_opt = critic_opt_fn(self.critic_params)
        self

    def forward(self, obs):
        obs = tensor(obs)
        return torch.tanh(self.fc_action(self.actor_body(obs)))

    def q(self, obs, a):
        obs = tensor(obs)
        a = tensor(a)
        x = torch.cat([obs, a], dim=1)
        q_1 = self.fc_critic_1(self.critic_body_1(x))
        q_2 = self.fc_critic_2(self.critic_body_2(x))
        return q_1, q_2


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_ShangtongZhang_DeepRL(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(CategoricalActorCriticNet(*[], **{'state_dim': 4, 'action_dim': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(DDPGConvBody(*[], **{}), [torch.rand([4, 4, 64, 64])], {})

    def test_002(self):
        self._check(DummyBody(*[], **{'state_dim': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(FCBody(*[], **{'state_dim': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_004(self):
        self._check(GaussianActorCriticNet(*[], **{'state_dim': 4, 'action_dim': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_005(self):
        self._check(OneLayerFCBodyWithAction(*[], **{'state_dim': 4, 'action_dim': 4, 'hidden_units': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_006(self):
        self._check(TwoLayerFCBodyWithAction(*[], **{'state_dim': 4, 'action_dim': 4}), [torch.rand([4, 4]), torch.rand([4, 4])], {})

