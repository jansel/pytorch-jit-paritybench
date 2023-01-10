import sys
_module = sys.modules[__name__]
del sys
config = _module
memory = _module
model = _module
train = _module
config = _module
model = _module
train = _module
config = _module
model = _module
train = _module
config = _module
model = _module
train = _module
config = _module
model = _module
train = _module
config = _module
model = _module
train = _module
config = _module
model = _module
train = _module
config = _module
memory = _module
model = _module
train = _module
config = _module
memory = _module
model = _module
train = _module
config = _module
memory = _module
model = _module
train = _module
config = _module
memory = _module
model = _module
train = _module
config = _module
memory = _module
model = _module
train = _module
config = _module
model = _module
train = _module
config = _module
model = _module
train = _module
config = _module
model = _module
shared_adam = _module
train = _module
worker = _module
config = _module
model = _module
shared_adam = _module
train = _module
worker = _module
config = _module
model = _module
shared_adam = _module
train = _module
worker = _module
config = _module
memory = _module
model = _module
train = _module
worker = _module
config = _module
model = _module
train = _module
config = _module
model = _module
train = _module
config = _module
model = _module
train = _module
config = _module
model = _module
train = _module
config = _module
memory = _module
model = _module
train = _module
config = _module
model = _module
train = _module
config = _module
model = _module
train = _module
config = _module
memory = _module
model = _module
train = _module
config = _module
memory = _module
model = _module
train = _module

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


import numpy as np


import random


import torch.optim as optim


from collections import namedtuple


import warnings


from collections import deque


import torch.multiprocessing as mp


import math


sigma_zero = 0.5


class NoisyLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_zero = sigma_zero
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
        self.weight_sigma.data.fill_(self.sigma_zero / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_zero / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, input):
        return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)


V_max = 5


V_min = -5


batch_size = 32


gamma = 0.99


n_step = 1


num_support = 8


class QNet(nn.Module):

    def __init__(self, num_inputs, num_outputs):
        super(QNet, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.dz = float(V_max - V_min) / (num_support - 1)
        self.z = torch.Tensor([(V_min + i * self.dz) for i in range(num_support)])
        self.fc = nn.Linear(num_inputs, 128)
        self.fc_adv = NoisyLinear(128, num_outputs * num_support)
        self.fc_val = nn.Linear(128, num_support)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)

    def forward(self, input):
        x = F.relu(self.fc(input))
        adv = self.fc_adv(x)
        val = self.fc_val(x)
        val = val.view(-1, 1, num_support)
        adv = adv.view(-1, self.num_outputs, num_support)
        z = val + (adv - adv.mean(1, keepdim=True))
        z = z.view(-1, self.num_outputs, num_support)
        p = nn.Softmax(dim=2)(z)
        return p

    def get_Qvalue(self, input):
        p = self.forward(input)
        p = p.squeeze(0)
        z_space = self.z.repeat(self.num_outputs, 1)
        Q = torch.sum(p * z_space, dim=1)
        return Q

    def reset_noise(self):
        self.fc_adv.reset_noise()

    def get_action(self, input):
        Q = self.get_Qvalue(input)
        action = torch.argmax(Q)
        return action.item()

    @classmethod
    def get_m(cls, _rewards, _masks, _prob_next_states_action):
        rewards = _rewards.numpy()
        masks = _masks.numpy()
        prob_next_states_action = _prob_next_states_action.detach().numpy()
        m_prob = np.zeros([batch_size, num_support], dtype=np.float32)
        dz = float(V_max - V_min) / (num_support - 1)
        batch_id = range(batch_size)
        for j in range(num_support):
            Tz = np.clip(rewards + masks * gamma ** n_step * (V_min + j * dz), V_min, V_max)
            bj = (Tz - V_min) / dz
            lj = np.floor(bj).astype(np.int64)
            uj = np.ceil(bj).astype(np.int64)
            blj = bj - lj
            buj = uj - bj
            m_prob[batch_id, lj[batch_id]] += (1 - masks + masks * prob_next_states_action[batch_id, j]) * buj[batch_id]
            m_prob[batch_id, uj[batch_id]] += (1 - masks + masks * prob_next_states_action[batch_id, j]) * blj[batch_id]
        return m_prob

    @classmethod
    def get_loss(cls, online_net, target_net, states, next_states, actions, rewards, masks):
        states = torch.stack(states)
        next_states = torch.stack(next_states)
        actions = torch.Tensor(actions).int()
        rewards = torch.Tensor(rewards)
        masks = torch.Tensor(masks)
        z_space = online_net.z.repeat(batch_size, online_net.num_outputs, 1)
        prob_next_states_online = online_net(next_states)
        prob_next_states_target = target_net(next_states)
        Q_next_state = torch.sum(prob_next_states_online * z_space, 2)
        next_actions = torch.argmax(Q_next_state, 1)
        prob_next_states_action = torch.stack([prob_next_states_target[i, action, :] for i, action in enumerate(next_actions)])
        m_prob = cls.get_m(rewards, masks, prob_next_states_action)
        m_prob = torch.tensor(m_prob)
        m_prob = (m_prob / torch.sum(m_prob, dim=1, keepdim=True)).detach()
        expand_dim_action = torch.unsqueeze(actions, -1)
        p = torch.sum(online_net(states) * expand_dim_action.float(), dim=1)
        loss = -torch.sum(m_prob * torch.log(p + 1e-20), 1)
        return loss

    @classmethod
    def train_model(cls, online_net, target_net, optimizer, batch, weights):
        loss = cls.get_loss(online_net, target_net, batch.state, batch.next_state, batch.action, batch.reward, batch.mask)
        loss = (loss * weights.detach()).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        online_net.reset_noise()
        return loss


ciritic_coefficient = 0.5


entropy_coefficient = 0.01


lambda_gae = 0.96


class GAE(nn.Module):

    def __init__(self, num_inputs, num_outputs):
        super(GAE, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.fc = nn.Linear(num_inputs, 128)
        self.fc_actor = nn.Linear(128, num_outputs)
        self.fc_critic = nn.Linear(128, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)

    def forward(self, input):
        x = F.relu(self.fc(input))
        policy = F.softmax(self.fc_actor(x))
        value = self.fc_critic(x)
        return policy, value

    @classmethod
    def get_gae(self, values, rewards, masks):
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        running_return = 0
        previous_value = 0
        running_advantage = 0
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + gamma * running_return * masks[t]
            running_tderror = rewards[t] + gamma * previous_value * masks[t] - values.data[t]
            running_advantage = running_tderror + gamma * lambda_gae * running_advantage * masks[t]
            returns[t] = running_return
            previous_value = values.data[t]
            advantages[t] = running_advantage
        return returns, advantages

    @classmethod
    def train_model(cls, net, transitions, optimizer):
        states, actions, rewards, masks = transitions.state, transitions.action, transitions.reward, transitions.mask
        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.Tensor(rewards)
        masks = torch.Tensor(masks)
        policies, values = net(states)
        policies = policies.view(-1, net.num_outputs)
        values = values.view(-1)
        returns, advantages = net.get_gae(values.view(-1).detach(), rewards, masks)
        log_policies = (torch.log(policies) * actions.detach()).sum(dim=1)
        actor_loss = -(log_policies * advantages).sum()
        critic_loss = (returns.detach() - values).pow(2).sum()
        entropy = (torch.log(policies) * policies).sum(1).sum()
        loss = actor_loss + ciritic_coefficient * critic_loss - entropy_coefficient * entropy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

    def get_action(self, input):
        policy, _ = self.forward(input)
        policy = policy[0].data.numpy()
        action = np.random.choice(self.num_outputs, 1, p=policy)[0]
        return action


def flat_grad(grads):
    grad_flatten = []
    for grad in grads:
        grad_flatten.append(grad.view(-1))
    grad_flatten = torch.cat(grad_flatten)
    return grad_flatten


def flat_hessian(hessians):
    hessians_flatten = []
    for hessian in hessians:
        hessians_flatten.append(hessian.contiguous().view(-1))
    hessians_flatten = torch.cat(hessians_flatten).data
    return hessians_flatten


def kl_divergence(policy, old_policy):
    kl = old_policy * torch.log(old_policy / policy)
    kl = kl.sum(1, keepdim=True)
    return kl


def fisher_vector_product(net, states, p, cg_damp=0.1):
    policy = net(states)
    old_policy = net(states).detach()
    kl = kl_divergence(policy, old_policy)
    kl = kl.mean()
    kl_grad = torch.autograd.grad(kl, net.parameters(), create_graph=True)
    kl_grad = flat_grad(kl_grad)
    kl_grad_p = (kl_grad * p.detach()).sum()
    kl_hessian_p = torch.autograd.grad(kl_grad_p, net.parameters())
    kl_hessian_p = flat_hessian(kl_hessian_p)
    return kl_hessian_p + cg_damp * p.detach()


def conjugate_gradient(net, states, loss_grad, n_step=10, residual_tol=1e-10):
    x = torch.zeros(loss_grad.size())
    r = loss_grad.clone()
    p = loss_grad.clone()
    r_dot_r = torch.dot(r, r)
    for i in range(n_step):
        A_dot_p = fisher_vector_product(net, states, p)
        alpha = r_dot_r / torch.dot(p, A_dot_p)
        x += alpha * p
        r -= alpha * A_dot_p
        new_r_dot_r = torch.dot(r, r)
        betta = new_r_dot_r / r_dot_r
        p = r + betta * p
        r_dot_r = new_r_dot_r
        if r_dot_r < residual_tol:
            break
    return x


def flat_params(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))
    params_flatten = torch.cat(params)
    return params_flatten


lr = 0.0001


def update_model(model, new_params):
    index = 0
    for params in model.parameters():
        params_length = len(params.view(-1))
        new_param = new_params[index:index + params_length]
        new_param = new_param.view(params.size())
        params.data.copy_(new_param)
        index += params_length


class TNPG(nn.Module):

    def __init__(self, num_inputs, num_outputs):
        super(TNPG, self).__init__()
        self.t = 0
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.fc_1 = nn.Linear(num_inputs, 128)
        self.fc_2 = nn.Linear(128, num_outputs)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)

    def forward(self, input):
        x = torch.tanh(self.fc_1(input))
        policy = F.softmax(self.fc_2(x))
        return policy

    @classmethod
    def train_model(cls, net, transitions):
        states, actions, rewards, masks = transitions.state, transitions.action, transitions.reward, transitions.mask
        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.Tensor(rewards)
        masks = torch.Tensor(masks)
        returns = torch.zeros_like(rewards)
        running_return = 0
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + gamma * running_return * masks[t]
            returns[t] = running_return
        policies = net(states)
        policies = policies.view(-1, net.num_outputs)
        policy_actions = (policies * actions.detach()).sum(dim=1)
        loss = (policy_actions * returns).mean()
        loss_grad = torch.autograd.grad(loss, net.parameters())
        loss_grad = flat_grad(loss_grad)
        step_dir = conjugate_gradient(net, states, loss_grad.data)
        params = flat_params(net)
        new_params = params + lr * step_dir
        update_model(net, new_params)
        return -loss

    def get_action(self, input):
        policy = self.forward(input)
        policy = policy[0].data.numpy()
        action = np.random.choice(self.num_outputs, 1, p=policy)[0]
        return action


max_kl = 0.01


class TRPO(nn.Module):

    def __init__(self, num_inputs, num_outputs):
        super(TRPO, self).__init__()
        self.t = 0
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.fc_1 = nn.Linear(num_inputs, 128)
        self.fc_2 = nn.Linear(128, num_outputs)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)

    def forward(self, input):
        x = torch.relu(self.fc_1(input))
        policy = F.softmax(self.fc_2(x))
        return policy

    @classmethod
    def train_model(cls, net, transitions):
        states, actions, rewards, masks = transitions.state, transitions.action, transitions.reward, transitions.mask
        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.Tensor(rewards)
        masks = torch.Tensor(masks)
        returns = torch.zeros_like(rewards)
        running_return = 0
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + gamma * running_return * masks[t]
            returns[t] = running_return
        policy = net(states)
        policy = policy.view(-1, net.num_outputs)
        policy_action = (policy * actions.detach()).sum(dim=1)
        old_policy = net(states).detach()
        old_policy = old_policy.view(-1, net.num_outputs)
        old_policy_action = (old_policy * actions.detach()).sum(dim=1)
        surrogate_loss = (policy_action / old_policy_action * returns).mean()
        surrogate_loss_grad = torch.autograd.grad(surrogate_loss, net.parameters())
        surrogate_loss_grad = flat_grad(surrogate_loss_grad)
        step_dir = conjugate_gradient(net, states, surrogate_loss_grad.data)
        params = flat_params(net)
        shs = (step_dir * fisher_vector_product(net, states, step_dir)).sum(0, keepdim=True)
        step_size = torch.sqrt(2 * max_kl / shs)[0]
        full_step = step_size * step_dir
        fraction = 1.0
        for _ in range(10):
            new_params = params + fraction * full_step
            update_model(net, new_params)
            policy = net(states)
            policy = policy.view(-1, net.num_outputs)
            policy_action = (policy * actions.detach()).sum(dim=1)
            surrogate_loss = (policy_action / old_policy_action * returns).mean()
            kl = kl_divergence(policy, old_policy)
            kl = kl.mean()
            if kl < max_kl:
                break
            fraction = fraction * 0.5
        return -surrogate_loss

    def get_action(self, input):
        policy = self.forward(input)
        policy = policy[0].data.numpy()
        action = np.random.choice(self.num_outputs, 1, p=policy)[0]
        return action


class BatchMaker:

    def __init__(self, states, actions, returns, advantages, old_policies):
        self.states = states
        self.actions = actions
        self.returns = returns
        self.advantages = advantages
        self.old_policies = old_policies

    def sample(self):
        sample_indexes = random.sample(range(len(self.states)), batch_size)
        states_sample = self.states[sample_indexes]
        actions_sample = self.actions[sample_indexes]
        retruns_sample = self.returns[sample_indexes]
        advantages_sample = self.advantages[sample_indexes]
        old_policies_sample = self.old_policies[sample_indexes]
        return states_sample, actions_sample, retruns_sample, advantages_sample, old_policies_sample


epoch_k = 10


epsilon_clip = 0.2


class PPO(nn.Module):

    def __init__(self, num_inputs, num_outputs):
        super(PPO, self).__init__()
        self.t = 0
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.fc = nn.Linear(num_inputs, 128)
        self.fc_actor = nn.Linear(128, num_outputs)
        self.fc_critic = nn.Linear(128, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)

    def forward(self, input):
        x = torch.relu(self.fc(input))
        policy = F.softmax(self.fc_actor(x), dim=-1)
        value = self.fc_critic(x)
        return policy, value

    @classmethod
    def get_gae(self, values, rewards, masks):
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        running_return = 0
        previous_value = 0
        running_advantage = 0
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + gamma * running_return * masks[t]
            running_tderror = rewards[t] + gamma * previous_value * masks[t] - values.data[t]
            running_advantage = running_tderror + gamma * lambda_gae * running_advantage * masks[t]
            returns[t] = running_return
            previous_value = values.data[t]
            advantages[t] = running_advantage
        return returns, advantages

    @classmethod
    def train_model(cls, net, transitions, optimizer):
        states, actions, rewards, masks = transitions.state, transitions.action, transitions.reward, transitions.mask
        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.Tensor(rewards)
        masks = torch.Tensor(masks)
        old_policies, old_values = net(states)
        old_policies = old_policies.view(-1, net.num_outputs).detach()
        returns, advantages = net.get_gae(old_values.view(-1).detach(), rewards, masks)
        batch_maker = BatchMaker(states, actions, returns, advantages, old_policies)
        for _ in range(epoch_k):
            for _ in range(len(states) // batch_size):
                states_sample, actions_sample, returns_sample, advantages_sample, old_policies_sample = batch_maker.sample()
                policies, values = net(states_sample)
                values = values.view(-1)
                policies = policies.view(-1, net.num_outputs)
                ratios = (policies / old_policies_sample * actions_sample.detach()).sum(dim=1)
                clipped_ratios = torch.clamp(ratios, min=1.0 - epsilon_clip, max=1.0 + epsilon_clip)
                actor_loss = -torch.min(ratios * advantages_sample, clipped_ratios * advantages_sample).sum()
                critic_loss = (returns_sample.detach() - values).pow(2).sum()
                policy_entropy = (torch.log(policies) * policies).sum(1, keepdim=True).mean()
                loss = actor_loss + ciritic_coefficient * critic_loss - entropy_coefficient * policy_entropy
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return loss

    def get_action(self, input):
        policy, _ = self.forward(input)
        policy = policy[0].data.numpy()
        action = np.random.choice(self.num_outputs, 1, p=policy)[0]
        return action


burn_in_length = 4


sequence_length = 32


class DRQN(nn.Module):

    def __init__(self, num_inputs, num_outputs):
        super(DRQN, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.lstm = nn.LSTM(input_size=num_inputs, hidden_size=16, batch_first=True)
        self.fc1 = nn.Linear(16, 128)
        self.fc2 = nn.Linear(128, num_outputs)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)

    def forward(self, x, hidden=None):
        out, hidden = self.lstm(x, hidden)
        out = F.relu(self.fc1(out))
        qvalue = self.fc2(out)
        return qvalue, hidden

    @classmethod
    def train_model(cls, online_net, target_net, optimizer, batch):

        def slice_burn_in(item):
            return item[:, burn_in_length:, :]
        states = torch.stack(batch.state).view(batch_size, sequence_length, online_net.num_inputs)
        next_states = torch.stack(batch.next_state).view(batch_size, sequence_length, online_net.num_inputs)
        actions = torch.stack(batch.action).view(batch_size, sequence_length, -1).long()
        rewards = torch.stack(batch.reward).view(batch_size, sequence_length, -1)
        masks = torch.stack(batch.mask).view(batch_size, sequence_length, -1)
        rnn_state = torch.stack(batch.rnn_state).view(batch_size, sequence_length, 2, -1)
        [h0, c0] = rnn_state[:, 0, :, :].transpose(0, 1)
        h0 = h0.unsqueeze(0).detach()
        c0 = c0.unsqueeze(0).detach()
        [h1, c1] = rnn_state[:, 1, :, :].transpose(0, 1)
        h1 = h1.unsqueeze(0).detach()
        c1 = c1.unsqueeze(0).detach()
        pred, _ = online_net(states, (h0, c0))
        next_pred, _ = target_net(next_states, (h1, c1))
        pred = slice_burn_in(pred)
        next_pred = slice_burn_in(next_pred)
        actions = slice_burn_in(actions)
        rewards = slice_burn_in(rewards)
        masks = slice_burn_in(masks)
        pred = pred.gather(2, actions)
        target = rewards + masks * gamma * next_pred.max(2, keepdim=True)[0]
        loss = F.mse_loss(pred, target.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

    def get_action(self, state, hidden):
        state = state.unsqueeze(0).unsqueeze(0)
        qvalue, hidden = self.forward(state, hidden)
        _, action = torch.max(qvalue, 2)
        return action.numpy()[0][0], hidden


class R2D2(nn.Module):

    def __init__(self, num_inputs, num_outputs):
        super(R2D2, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.lstm = nn.LSTM(input_size=num_inputs, hidden_size=16, batch_first=True)
        self.fc = nn.Linear(16, 128)
        self.fc_adv = nn.Linear(128, num_outputs)
        self.fc_val = nn.Linear(128, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)

    def forward(self, x, hidden=None):
        batch_size = x.size()[0]
        sequence_length = x.size()[1]
        out, hidden = self.lstm(x, hidden)
        out = F.relu(self.fc(out))
        adv = self.fc_adv(out)
        adv = adv.view(batch_size, sequence_length, self.num_outputs)
        val = self.fc_val(out)
        val = val.view(batch_size, sequence_length, 1)
        qvalue = val + (adv - adv.mean(dim=2, keepdim=True))
        return qvalue, hidden

    @classmethod
    def get_td_error(cls, online_net, target_net, batch, lengths):

        def slice_burn_in(item):
            return item[:, burn_in_length:, :]
        batch_size = torch.stack(batch.state).size()[0]
        states = torch.stack(batch.state).view(batch_size, sequence_length, online_net.num_inputs)
        next_states = torch.stack(batch.next_state).view(batch_size, sequence_length, online_net.num_inputs)
        actions = torch.stack(batch.action).view(batch_size, sequence_length, -1).long()
        rewards = torch.stack(batch.reward).view(batch_size, sequence_length, -1)
        masks = torch.stack(batch.mask).view(batch_size, sequence_length, -1)
        steps = torch.stack(batch.step).view(batch_size, sequence_length, -1)
        rnn_state = torch.stack(batch.rnn_state).view(batch_size, sequence_length, 2, -1)
        [h0, c0] = rnn_state[:, 0, :, :].transpose(0, 1)
        h0 = h0.unsqueeze(0).detach()
        c0 = c0.unsqueeze(0).detach()
        [h1, c1] = rnn_state[:, 1, :, :].transpose(0, 1)
        h1 = h1.unsqueeze(0).detach()
        c1 = c1.unsqueeze(0).detach()
        pred, _ = online_net(states, (h0, c0))
        next_pred, _ = target_net(next_states, (h1, c1))
        next_pred_online, _ = online_net(next_states, (h1, c1))
        pred = slice_burn_in(pred)
        next_pred = slice_burn_in(next_pred)
        actions = slice_burn_in(actions)
        rewards = slice_burn_in(rewards)
        masks = slice_burn_in(masks)
        steps = slice_burn_in(steps)
        next_pred_online = slice_burn_in(next_pred_online)
        pred = pred.gather(2, actions)
        _, next_pred_online_action = next_pred_online.max(2)
        target = rewards + masks * pow(gamma, steps) * next_pred.gather(2, next_pred_online_action.unsqueeze(2))
        td_error = pred - target.detach()
        for idx, length in enumerate(lengths):
            td_error[idx][length - burn_in_length:][:] = 0
        return td_error

    @classmethod
    def train_model(cls, online_net, target_net, optimizer, batch, lengths):
        td_error = cls.get_td_error(online_net, target_net, batch, lengths)
        loss = pow(td_error, 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss, td_error

    def get_action(self, state, hidden):
        state = state.unsqueeze(0).unsqueeze(0)
        qvalue, hidden = self.forward(state, hidden)
        _, action = torch.max(qvalue, 2)
        return action.numpy()[0][0], hidden


class QRDQN(nn.Module):

    def __init__(self, num_inputs, num_outputs):
        super(QRDQN, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_support = num_support
        self.fc1 = nn.Linear(num_inputs, 128)
        self.fc2 = nn.Linear(128, num_outputs * num_support)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.fc2(x)
        theta = x.view(-1, self.num_outputs, self.num_support)
        return theta

    def get_action(self, state):
        theta = self.forward(state)
        Q = theta.mean(dim=2, keepdim=True)
        action = torch.argmax(Q)
        return action.item()

    @classmethod
    def train_model(cls, online_net, target_net, optimizer, batch):
        states = torch.stack(batch.state)
        next_states = torch.stack(batch.next_state)
        actions = torch.Tensor(batch.action).long()
        rewards = torch.Tensor(batch.reward)
        masks = torch.Tensor(batch.mask)
        theta = online_net(states)
        action = actions.unsqueeze(1).unsqueeze(1).expand(-1, 1, num_support)
        theta_a = theta.gather(1, action).squeeze(1)
        next_theta = target_net(next_states)
        next_action = next_theta.mean(dim=2).max(1)[1]
        next_action = next_action.unsqueeze(1).unsqueeze(1).expand(batch_size, 1, num_support)
        next_theta_a = next_theta.gather(1, next_action).squeeze(1)
        T_theta = rewards.unsqueeze(1) + gamma * next_theta_a * masks.unsqueeze(1)
        T_theta_tile = T_theta.view(-1, num_support, 1).expand(-1, num_support, num_support)
        theta_a_tile = theta_a.view(-1, 1, num_support).expand(-1, num_support, num_support)
        error_loss = T_theta_tile - theta_a_tile
        huber_loss = F.smooth_l1_loss(theta_a_tile, T_theta_tile.detach(), reduction='none')
        tau = torch.arange(0.5 * (1 / num_support), 1, 1 / num_support).view(1, num_support)
        loss = (tau - (error_loss < 0).float()).abs() * huber_loss
        loss = loss.mean(dim=2).sum(dim=1).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss


num_quantile_sample = 32


num_tau_prime_sample = 8


num_tau_sample = 16


quantile_embedding_dim = 64


class IQN(nn.Module):

    def __init__(self, num_inputs, num_outputs):
        super(IQN, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.fc1 = nn.Linear(num_inputs, 128)
        self.fc2 = nn.Linear(128, num_outputs)
        self.phi = nn.Linear(quantile_embedding_dim, 128)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)

    def forward(self, state, tau, num_quantiles):
        input_size = state.size()[0]
        tau = tau.expand(input_size * num_quantiles, quantile_embedding_dim)
        pi_mtx = torch.Tensor(np.pi * np.arange(0, quantile_embedding_dim)).expand(input_size * num_quantiles, quantile_embedding_dim)
        cos_tau = torch.cos(tau * pi_mtx)
        phi = self.phi(cos_tau)
        phi = F.relu(phi)
        state_tile = state.expand(input_size, num_quantiles, self.num_inputs)
        state_tile = state_tile.flatten().view(-1, self.num_inputs)
        x = F.relu(self.fc1(state_tile))
        x = self.fc2(x * phi)
        z = x.view(-1, num_quantiles, self.num_outputs)
        z = z.transpose(1, 2)
        return z

    def get_action(self, state):
        tau = torch.Tensor(np.random.rand(num_quantile_sample, 1) * 0.5)
        z = self.forward(state, tau, num_quantile_sample)
        q = z.mean(dim=2, keepdim=True)
        action = torch.argmax(q)
        return action.item()

    @classmethod
    def train_model(cls, online_net, target_net, optimizer, batch):
        states = torch.stack(batch.state)
        next_states = torch.stack(batch.next_state)
        actions = torch.Tensor(batch.action).long()
        rewards = torch.Tensor(batch.reward)
        masks = torch.Tensor(batch.mask)
        tau = torch.Tensor(np.random.rand(batch_size * num_tau_sample, 1))
        z = online_net(states, tau, num_tau_sample)
        action = actions.unsqueeze(1).unsqueeze(1).expand(-1, 1, num_tau_sample)
        z_a = z.gather(1, action).squeeze(1)
        tau_prime = torch.Tensor(np.random.rand(batch_size * num_tau_prime_sample, 1))
        next_z = target_net(next_states, tau_prime, num_tau_prime_sample)
        next_action = next_z.mean(dim=2).max(1)[1]
        next_action = next_action.unsqueeze(1).unsqueeze(1).expand(batch_size, 1, num_tau_prime_sample)
        next_z_a = next_z.gather(1, next_action).squeeze(1)
        T_z = rewards.unsqueeze(1) + gamma * next_z_a * masks.unsqueeze(1)
        T_z_tile = T_z.view(-1, num_tau_prime_sample, 1).expand(-1, num_tau_prime_sample, num_tau_sample)
        z_a_tile = z_a.view(-1, 1, num_tau_sample).expand(-1, num_tau_prime_sample, num_tau_sample)
        error_loss = T_z_tile - z_a_tile
        huber_loss = F.smooth_l1_loss(z_a_tile, T_z_tile.detach(), reduction='none')
        tau = torch.arange(0, 1, 1 / num_tau_sample).view(1, num_tau_sample)
        loss = (tau - (error_loss < 0).float()).abs() * huber_loss
        loss = loss.mean(dim=2).sum(dim=1).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss


class Model(nn.Module):

    def __init__(self, num_inputs, num_outputs):
        super(Model, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.fc = nn.Linear(num_inputs, 128)
        self.fc_adv = nn.Linear(128, num_outputs)
        self.fc_val = nn.Linear(128, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)

    def forward(self, x):
        x = F.relu(self.fc(x))
        adv = self.fc_adv(x)
        adv = adv.view(-1, self.num_outputs)
        val = self.fc_val(x)
        val = val.view(-1, 1)
        qvalue = val + (adv - adv.mean(dim=1, keepdim=True))
        return qvalue


class GlobalModel(Model):

    def __init__(self, num_inputs, num_outputs):
        super(GlobalModel, self).__init__(num_inputs, num_outputs)


class LocalModel(Model):

    def __init__(self, num_inputs, num_outputs):
        super(LocalModel, self).__init__(num_inputs, num_outputs)

    def pull_from_global_model(self, global_model):
        self.load_state_dict(global_model.state_dict())

    def get_action(self, input):
        qvalue = self.forward(input)
        _, action = torch.max(qvalue, 1)
        return action.numpy()[0]


class DoubleDQNet(nn.Module):

    def __init__(self, num_inputs, num_outputs):
        super(DoubleDQNet, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.fc1 = nn.Linear(num_inputs, 128)
        self.fc2 = nn.Linear(128, num_outputs)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        qvalue = self.fc2(x)
        return qvalue

    @classmethod
    def train_model(cls, online_net, target_net, optimizer, batch):
        states = torch.stack(batch.state)
        next_states = torch.stack(batch.next_state)
        actions = torch.Tensor(batch.action).float()
        rewards = torch.Tensor(batch.reward)
        masks = torch.Tensor(batch.mask)
        pred = online_net(states).squeeze(1)
        _, action_from_online_net = online_net(next_states).squeeze(1).max(1)
        next_pred = target_net(next_states).squeeze(1)
        pred = torch.sum(pred.mul(actions), dim=1)
        target = rewards + masks * gamma * next_pred.gather(1, action_from_online_net.unsqueeze(1)).squeeze(1)
        loss = F.mse_loss(pred, target.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

    def get_action(self, input):
        qvalue = self.forward(input)
        _, action = torch.max(qvalue, 1)
        return action.numpy()[0]


class DuelDQNet(nn.Module):

    def __init__(self, num_inputs, num_outputs):
        super(DuelDQNet, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.fc = nn.Linear(num_inputs, 128)
        self.fc_adv = nn.Linear(128, num_outputs)
        self.fc_val = nn.Linear(128, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)

    def forward(self, x):
        x = F.relu(self.fc(x))
        adv = self.fc_adv(x)
        adv = adv.view(-1, self.num_outputs)
        val = self.fc_val(x)
        val = val.view(-1, 1)
        qvalue = val + (adv - adv.mean(dim=1, keepdim=True))
        return qvalue

    @classmethod
    def train_model(cls, online_net, target_net, optimizer, batch):
        states = torch.stack(batch.state)
        next_states = torch.stack(batch.next_state)
        actions = torch.Tensor(batch.action).float()
        rewards = torch.Tensor(batch.reward)
        masks = torch.Tensor(batch.mask)
        pred = online_net(states).squeeze(1)
        next_pred = target_net(next_states).squeeze(1)
        pred = torch.sum(pred.mul(actions), dim=1)
        target = rewards + masks * gamma * next_pred.max(1)[0]
        loss = F.mse_loss(pred, target.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

    def get_action(self, input):
        qvalue = self.forward(input)
        _, action = torch.max(qvalue, 1)
        return action.numpy()[0]


class Distributional_C51(nn.Module):

    def __init__(self, num_inputs, num_outputs):
        super(Distributional_C51, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.dz = float(V_max - V_min) / (num_support - 1)
        self.z = torch.Tensor([(V_min + i * self.dz) for i in range(num_support)])
        self.fc1 = nn.Linear(num_inputs, 128)
        self.fc2 = nn.Linear(128, num_outputs * num_support)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = self.fc2(x)
        z = x.view(-1, self.num_outputs, num_support)
        p = nn.Softmax(dim=2)(z)
        return p

    def get_action(self, input):
        p = self.forward(input)
        p = p.squeeze(0)
        z_space = self.z.repeat(self.num_outputs, 1)
        Q = torch.sum(p * z_space, dim=1)
        action = torch.argmax(Q)
        return action.item()

    @classmethod
    def get_m(cls, _rewards, _masks, _prob_next_states_action):
        rewards = _rewards.numpy()
        masks = _masks.numpy()
        prob_next_states_action = _prob_next_states_action.detach().numpy()
        m_prob = np.zeros([batch_size, num_support], dtype=np.float32)
        dz = float(V_max - V_min) / (num_support - 1)
        batch_id = range(batch_size)
        for j in range(num_support):
            Tz = np.clip(rewards + masks * gamma * (V_min + j * dz), V_min, V_max)
            bj = (Tz - V_min) / dz
            lj = np.floor(bj).astype(np.int64)
            uj = np.ceil(bj).astype(np.int64)
            blj = bj - lj
            buj = uj - bj
            m_prob[batch_id, lj[batch_id]] += (1 - masks + masks * prob_next_states_action[batch_id, j]) * buj[batch_id]
            m_prob[batch_id, uj[batch_id]] += (1 - masks + masks * prob_next_states_action[batch_id, j]) * blj[batch_id]
        return m_prob

    @classmethod
    def train_model(cls, online_net, target_net, optimizer, batch):
        states = torch.stack(batch.state)
        next_states = torch.stack(batch.next_state)
        actions = torch.Tensor(batch.action).int()
        rewards = torch.Tensor(batch.reward)
        masks = torch.Tensor(batch.mask)
        z_space = online_net.z.repeat(batch_size, online_net.num_outputs, 1)
        prob_next_states = target_net(next_states)
        Q_next_state = torch.sum(prob_next_states * z_space, 2)
        next_actions = torch.argmax(Q_next_state, 1)
        prob_next_states_action = torch.stack([prob_next_states[i, action, :] for i, action in enumerate(next_actions)])
        m_prob = cls.get_m(rewards, masks, prob_next_states_action)
        m_prob = torch.tensor(m_prob)
        m_prob = (m_prob / torch.sum(m_prob, dim=1, keepdim=True)).detach()
        expand_dim_action = torch.unsqueeze(actions, -1)
        p = torch.sum(online_net(states) * expand_dim_action.float(), dim=1)
        loss = -torch.sum(m_prob * torch.log(p + 1e-20), 1)
        loss = loss.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (DRQN,
     lambda: ([], {'num_inputs': 4, 'num_outputs': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (Distributional_C51,
     lambda: ([], {'num_inputs': 4, 'num_outputs': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (DoubleDQNet,
     lambda: ([], {'num_inputs': 4, 'num_outputs': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DuelDQNet,
     lambda: ([], {'num_inputs': 4, 'num_outputs': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GAE,
     lambda: ([], {'num_inputs': 4, 'num_outputs': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GlobalModel,
     lambda: ([], {'num_inputs': 4, 'num_outputs': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LocalModel,
     lambda: ([], {'num_inputs': 4, 'num_outputs': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Model,
     lambda: ([], {'num_inputs': 4, 'num_outputs': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (NoisyLinear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PPO,
     lambda: ([], {'num_inputs': 4, 'num_outputs': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (QNet,
     lambda: ([], {'num_inputs': 4, 'num_outputs': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (QRDQN,
     lambda: ([], {'num_inputs': 4, 'num_outputs': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (R2D2,
     lambda: ([], {'num_inputs': 4, 'num_outputs': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (TNPG,
     lambda: ([], {'num_inputs': 4, 'num_outputs': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TRPO,
     lambda: ([], {'num_inputs': 4, 'num_outputs': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_g6ling_Reinforcement_Learning_Pytorch_Cartpole(_paritybench_base):
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

    def test_008(self):
        self._check(*TESTCASES[8])

    def test_009(self):
        self._check(*TESTCASES[9])

    def test_010(self):
        self._check(*TESTCASES[10])

    def test_011(self):
        self._check(*TESTCASES[11])

    def test_012(self):
        self._check(*TESTCASES[12])

    def test_013(self):
        self._check(*TESTCASES[13])

    def test_014(self):
        self._check(*TESTCASES[14])

