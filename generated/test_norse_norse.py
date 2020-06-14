import sys
_module = sys.modules[__name__]
del sys
conf = _module
norse = _module
benchmark = _module
bindsnet_lif = _module
main = _module
norse_lif = _module
pysnn_lif = _module
speech_commands = _module
plot_mnist = _module
task = _module
cartpole = _module
cifar10 = _module
correlation_experiment = _module
mnist = _module
torch = _module
functional = _module
coba_lif = _module
correlation_sensor = _module
encode = _module
heaviside = _module
leaky_integrator = _module
lif = _module
lif_correlation = _module
lif_mc = _module
lif_mc_refrac = _module
lif_refrac = _module
logical = _module
lsnn = _module
spiking_vector_quantization = _module
stdp_sensor = _module
superspike = _module
test_coba_lif = _module
test_encode = _module
test_heaviside = _module
test_leaky_integrator = _module
test_lif = _module
test_lif_mc = _module
test_lif_mc_refrac = _module
test_lif_refrac = _module
test_logical = _module
test_lsnn = _module
test_tsodyks_makram = _module
threshold = _module
tsodyks_makram = _module
models = _module
conv = _module
module = _module
coba_lif = _module
encode = _module
if_current_encoder = _module
leaky_integrator = _module
lif = _module
lif_correlation = _module
lif_mc = _module
lif_mc_refrac = _module
lif_refrac = _module
lsnn = _module
test = _module
test_encode = _module
setup = _module

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


import time


import torch


import numpy as np


import random


import uuid


from collections import namedtuple


from typing import NamedTuple


from typing import Tuple


import torch.jit


from typing import Union


from typing import Callable


class LIFParameters(NamedTuple):
    """Parametrization of a LIF neuron

    Parameters:
        tau_syn_inv (torch.Tensor): inverse synaptic time
                                    constant (:math:`1/\\tau_\\text{syn}`) in 1/ms
        tau_mem_inv (torch.Tensor): inverse membrane time
                                    constant (:math:`1/\\tau_\\text{mem}`) in 1/ms
        v_leak (torch.Tensor): leak potential in mV
        v_th (torch.Tensor): threshold potential in mV
        v_reset (torch.Tensor): reset potential in mV
        method (str): method to determine the spike threshold
                      (relevant for surrogate gradients)
        alpha (float): hyper parameter to use in surrogate gradient computation
    """
    tau_syn_inv: torch.Tensor = torch.as_tensor(1.0 / 0.005)
    tau_mem_inv: torch.Tensor = torch.as_tensor(1.0 / 0.01)
    v_leak: torch.Tensor = torch.as_tensor(0.0)
    v_th: torch.Tensor = torch.as_tensor(1.0)
    v_reset: torch.Tensor = torch.as_tensor(0.0)
    method: str = 'super'
    alpha: float = 0.0


class Net(torch.nn.Module):

    def __init__(self, device='cpu', num_channels=1, feature_size=32, model
        ='super', dtype=torch.float):
        super(Net, self).__init__()
        self.features = int(((feature_size - 4) / 2 - 4) / 2)
        self.conv1 = torch.nn.Conv2d(num_channels, 32, 5, 1)
        self.conv2 = torch.nn.Conv2d(32, 64, 5, 1)
        self.fc1 = torch.nn.Linear(self.features * self.features * 64, 1024)
        self.lif0 = LIFFeedForwardCell((32, feature_size - 4, feature_size -
            4), p=LIFParameters(method=model, alpha=100.0))
        self.lif1 = LIFFeedForwardCell((64, int((feature_size - 4) / 2) - 4,
            int((feature_size - 4) / 2) - 4), p=LIFParameters(method=model,
            alpha=100.0))
        self.lif2 = LIFFeedForwardCell((1024,), p=LIFParameters(method=
            model, alpha=100.0))
        self.out = LICell(1024, 10)
        self.device = device
        self.dtype = dtype

    def forward(self, x):
        seq_length = x.shape[0]
        seq_batch_size = x.shape[1]
        s0 = self.lif0.initial_state(seq_batch_size, device=self.device,
            dtype=self.dtype)
        s1 = self.lif1.initial_state(seq_batch_size, device=self.device,
            dtype=self.dtype)
        s2 = self.lif2.initial_state(seq_batch_size, device=self.device,
            dtype=self.dtype)
        so = self.out.initial_state(seq_batch_size, device=self.device,
            dtype=self.dtype)
        voltages = torch.zeros(seq_length, seq_batch_size, 10, device=self.
            device, dtype=self.dtype)
        for ts in range(seq_length):
            z = self.conv1(x[(ts), :])
            z, s0 = self.lif0(z, s0)
            z = torch.nn.functional.max_pool2d(z, 2, 2)
            z = 10 * self.conv2(z)
            z, s1 = self.lif1(z, s1)
            z = torch.nn.functional.max_pool2d(z, 2, 2)
            z = z.view(-1, self.features ** 2 * 64)
            z = self.fc1(z)
            z, s2 = self.lif2(z, s2)
            v, so = self.out(torch.nn.functional.relu(z), so)
            voltages[(ts), :, :] = v
        return voltages


class ANNPolicy(torch.nn.Module):

    def __init__(self):
        super(ANNPolicy, self).__init__()
        self.state_space = 4
        self.action_space = 2
        self.l1 = torch.nn.Linear(self.state_space, 128, bias=False)
        self.l2 = torch.nn.Linear(128, self.action_space, bias=False)
        self.dropout = torch.nn.Dropout(p=0.6)
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.l1(x)
        x = self.dropout(x)
        x = torch.nn.functional.relu(x)
        x = self.l2(x)
        x = torch.nn.functional.softmax(x)
        return x


class Policy(torch.nn.Module):

    def __init__(self, device='cpu'):
        super(Policy, self).__init__()
        self.state_dim = 4
        self.input_features = 16
        self.hidden_features = 128
        self.output_features = 2
        self.device = device
        self.constant_current_encoder = ConstantCurrentLIFEncoder(40)
        self.lif = LIFCell(2 * self.state_dim, self.hidden_features, p=
            LIFParameters(method='super', alpha=100.0))
        self.dropout = torch.nn.Dropout(p=0.5)
        self.readout = LICell(self.hidden_features, self.output_features)
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        scale = 50
        x = x.to(self.device)
        x_pos = self.constant_current_encoder(torch.nn.functional.relu(
            scale * x))
        x_neg = self.constant_current_encoder(torch.nn.functional.relu(-
            scale * x))
        x = torch.cat([x_pos, x_neg], dim=2)
        seq_length, batch_size, _ = x.shape
        s1 = self.lif.initial_state(batch_size, device=self.device)
        so = self.readout.initial_state(batch_size, device=self.device)
        voltages = torch.zeros(seq_length, batch_size, self.output_features,
            device=self.device)
        for ts in range(seq_length):
            z1, s1 = self.lif(x[(ts), :, :], s1)
            z1 = self.dropout(z1)
            vo, so = self.readout(z1, so)
            voltages[(ts), :, :] = vo
        m, _ = torch.max(voltages, 0)
        p_y = torch.nn.functional.softmax(m, dim=1)
        return p_y


class LSNNParameters(NamedTuple):
    """Parameters of an LSNN neuron

    Parameters:
        tau_syn_inv (torch.Tensor): inverse synaptic time
                                    constant (:math:`1/\\tau_\\text{syn}`)
        tau_mem_inv (torch.Tensor): inverse membrane time
                                    constant (:math:`1/\\tau_\\text{mem}`)
        tau_adapt_inv (torch.Tensor): inverse adaptation time
                                      constant (:math:`1/\\tau_b`)
        v_leak (torch.Tensor): leak potential
        v_th (torch.Tensor): threshold potential
        v_reset (torch.Tensor): reset potential
        beta (torch.Tensor): adaptation constant
    """
    tau_syn_inv: torch.Tensor = torch.as_tensor(1.0 / 0.005)
    tau_mem_inv: torch.Tensor = torch.as_tensor(1.0 / 0.01)
    tau_adapt_inv: torch.Tensor = torch.as_tensor(1.0 / 700)
    v_leak: torch.Tensor = torch.as_tensor(0.0)
    v_th: torch.Tensor = torch.as_tensor(1.0)
    v_reset: torch.Tensor = torch.as_tensor(0.0)
    beta: torch.Tensor = torch.as_tensor(1.8)
    method: str = 'super'
    alpha: float = 100.0


class LSNNPolicy(torch.nn.Module):

    def __init__(self, device='cpu', model='super'):
        super(LSNNPolicy, self).__init__()
        self.state_dim = 4
        self.input_features = 16
        self.hidden_features = 128
        self.output_features = 2
        self.device = device
        self.constant_current_encoder = ConstantCurrentLIFEncoder(40)
        self.lif_layer = LSNNCell(2 * self.state_dim, self.hidden_features,
            p=LSNNParameters(model, alpha=100.0))
        self.dropout = torch.nn.Dropout(p=0.5)
        self.readout = LICell(self.hidden_features, self.output_features)
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        scale = 50
        _, x_pos = self.constant_current_encoder(torch.nn.functional.relu(
            scale * x))
        _, x_neg = self.constant_current_encoder(torch.nn.functional.relu(-
            scale * x))
        x = torch.cat([x_pos, x_neg], dim=2)
        seq_length, batch_size, _ = x.shape
        s1 = self.lif_layer.initial_state(batch_size, device=self.device)
        so = self.readout.initial_state(batch_size, device=self.device)
        voltages = torch.zeros(seq_length, batch_size, self.output_features,
            device=self.device)
        for ts in range(seq_length):
            z1, s1 = self.lif_layer(x[(ts), :, :], s1)
            z1 = self.dropout(z1)
            vo, so = self.readout(z1, so)
            voltages[(ts), :, :] = vo
        m, _ = torch.max(voltages, 0)
        p_y = torch.nn.functional.softmax(m, dim=1)
        return p_y


class LIFConvNet(torch.nn.Module):

    def __init__(self, input_features, seq_length, model='super', device=
        'cpu', only_first_spike=False):
        super(LIFConvNet, self).__init__()
        self.constant_current_encoder = ConstantCurrentLIFEncoder(seq_length
            =seq_length)
        self.only_first_spike = only_first_spike
        self.input_features = input_features
        self.rsnn = ConvNet4(device=device, method=model)
        self.device = device
        self.seq_length = seq_length

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.constant_current_encoder(x.view(-1, self.input_features) *
            FLAGS.input_scale)
        if self.only_first_spike:
            zeros = torch.zeros_like(x.cpu()).detach().numpy()
            idxs = x.cpu().nonzero().detach().numpy()
            spike_counter = np.zeros((FLAGS.batch_size, 28 * 28))
            for t, batch, nrn in idxs:
                if spike_counter[batch, nrn] == 0:
                    zeros[t, batch, nrn] = 1
                    spike_counter[batch, nrn] += 1
            x = torch.from_numpy(zeros).to(self.device)
        x = x.reshape(self.seq_length, batch_size, 1, 28, 28)
        voltages = self.rsnn(x)
        m, _ = torch.max(voltages, 0)
        log_p_y = torch.nn.functional.log_softmax(m, dim=1)
        return log_p_y


class ConvNet(torch.nn.Module):

    def __init__(self, device, num_channels=1, feature_size=28, method=
        'super', dtype=torch.float):
        super(ConvNet, self).__init__()
        self.features = int(((feature_size - 4) / 2 - 4) / 2)
        self.conv1 = torch.nn.Conv2d(num_channels, 20, 5, 1)
        self.conv2 = torch.nn.Conv2d(20, 50, 5, 1)
        self.fc1 = torch.nn.Linear(self.features * self.features * 50, 500)
        self.out = LICell(500, 10)
        self.device = device
        self.lif0 = LIFFeedForwardCell((20, feature_size - 4, feature_size -
            4), p=LIFParameters(method=method, alpha=100.0))
        self.lif1 = LIFFeedForwardCell((50, int((feature_size - 4) / 2) - 4,
            int((feature_size - 4) / 2) - 4), p=LIFParameters(method=method,
            alpha=100.0))
        self.lif2 = LIFFeedForwardCell((500,), p=LIFParameters(method=
            method, alpha=100.0))
        self.dtype = dtype

    def forward(self, x):
        seq_length = x.shape[0]
        batch_size = x.shape[1]
        s0 = self.lif0.initial_state(batch_size, self.device, self.dtype)
        s1 = self.lif1.initial_state(batch_size, self.device, self.dtype)
        s2 = self.lif2.initial_state(batch_size, self.device, self.dtype)
        so = self.out.initial_state(batch_size, device=self.device, dtype=
            self.dtype)
        voltages = torch.zeros(seq_length, batch_size, 10, device=self.
            device, dtype=self.dtype)
        for ts in range(seq_length):
            z = self.conv1(x[(ts), :])
            z, s0 = self.lif0(z, s0)
            z = torch.nn.functional.max_pool2d(z, 2, 2)
            z = 10 * self.conv2(z)
            z, s1 = self.lif1(z, s1)
            z = torch.nn.functional.max_pool2d(z, 2, 2)
            z = z.view(-1, self.features ** 2 * 50)
            z = self.fc1(z)
            z, s2 = self.lif2(z, s2)
            v, so = self.out(torch.nn.functional.relu(z), so)
            voltages[(ts), :, :] = v
        return voltages


class ConvNet4(torch.nn.Module):

    def __init__(self, device, num_channels=1, feature_size=28, method=
        'super', dtype=torch.float):
        super(ConvNet4, self).__init__()
        self.features = int(((feature_size - 4) / 2 - 4) / 2)
        self.conv1 = torch.nn.Conv2d(num_channels, 32, 5, 1)
        self.conv2 = torch.nn.Conv2d(32, 64, 5, 1)
        self.fc1 = torch.nn.Linear(self.features * self.features * 64, 1024)
        self.lif0 = LIFFeedForwardCell((32, feature_size - 4, feature_size -
            4), p=LIFParameters(method=method, alpha=100.0))
        self.lif1 = LIFFeedForwardCell((64, int((feature_size - 4) / 2) - 4,
            int((feature_size - 4) / 2) - 4), p=LIFParameters(method=method,
            alpha=100.0))
        self.lif2 = LIFFeedForwardCell((1024,), p=LIFParameters(method=
            method, alpha=100.0))
        self.out = LICell(1024, 10)
        self.device = device
        self.dtype = dtype

    def forward(self, x):
        seq_length = x.shape[0]
        batch_size = x.shape[1]
        s0 = self.lif0.initial_state(batch_size, device=self.device, dtype=
            self.dtype)
        s1 = self.lif1.initial_state(batch_size, device=self.device, dtype=
            self.dtype)
        s2 = self.lif2.initial_state(batch_size, device=self.device, dtype=
            self.dtype)
        so = self.out.initial_state(batch_size, device=self.device, dtype=
            self.dtype)
        voltages = torch.zeros(seq_length, batch_size, 10, device=self.
            device, dtype=self.dtype)
        for ts in range(seq_length):
            z = self.conv1(x[(ts), :])
            z, s0 = self.lif0(z, s0)
            z = torch.nn.functional.max_pool2d(z, 2, 2)
            z = 10 * self.conv2(z)
            z, s1 = self.lif1(z, s1)
            z = torch.nn.functional.max_pool2d(z, 2, 2)
            z = z.view(-1, self.features ** 2 * 64)
            z = self.fc1(z)
            z, s2 = self.lif2(z, s2)
            v, so = self.out(torch.nn.functional.relu(z), so)
            voltages[(ts), :, :] = v
        return voltages


class CobaLIFParameters(NamedTuple):
    """Parameters of conductance based LIF neuron.

    Parameters:
        tau_syn_exc_inv (torch.Tensor): inverse excitatory synaptic input
                                        time constant
        tau_syn_inh_inv (torch.Tensor): inverse inhibitory synaptic input
                                        time constant
        c_m_inv (torch.Tensor): inverse membrane capacitance
        g_l (torch.Tensor): leak conductance
        e_rev_I (torch.Tensor): inhibitory reversal potential
        e_rev_E (torch.Tensor): excitatory reversal potential
        v_rest (torch.Tensor): rest membrane potential
        v_reset (torch.Tensor): reset membrane potential
        v_thresh (torch.Tensor): threshold membrane potential
        method (str): method to determine the spike threshold
                      (relevant for surrogate gradients)
        alpha (float): hyper parameter to use in surrogate gradient computation
    """
    tau_syn_exc_inv: torch.Tensor = torch.as_tensor(1.0 / 5)
    tau_syn_inh_inv: torch.Tensor = torch.as_tensor(1.0 / 5)
    c_m_inv: torch.Tensor = torch.as_tensor(1 / 0.2)
    g_l: torch.Tensor = torch.as_tensor(1 / 20 * 1 / 0.2)
    e_rev_I: torch.Tensor = torch.as_tensor(-100)
    e_rev_E: torch.Tensor = torch.as_tensor(60)
    v_rest: torch.Tensor = torch.as_tensor(-20)
    v_reset: torch.Tensor = torch.as_tensor(-70)
    v_thresh: torch.Tensor = torch.as_tensor(-10)
    method: str = 'super'
    alpha: float = 100.0


class CobaLIFState(NamedTuple):
    """State of a conductance based LIF neuron.

    Parameters:
        z (torch.Tensor): recurrent spikes
        v (torch.Tensor): membrane potential
        g_e (torch.Tensor): excitatory input conductance
        g_i (torch.Tensor): inhibitory input conductance
    """
    z: torch.Tensor
    v: torch.Tensor
    g_e: torch.Tensor
    g_i: torch.Tensor


def heaviside(data):
    """
    A `heaviside step function <https://en.wikipedia.org/wiki/Heaviside_step_function>`_
    that truncates numbers <= 0 to 0 and everything else to 1.

    .. math::
        H[n]=\\begin{cases} 0, & n <= 0, \\ 1, & n \\g 0, \\end{cases}
    """
    return torch.where(data <= torch.zeros_like(data), torch.zeros_like(
        data), torch.ones_like(data))


class HeaviCirc(torch.autograd.Function):
    """Approximation of the heaviside step function as

    .. math::
        h(x,\\alpha) = \\frac{1}{2} + \\frac{1}{2} \\
        \\frac{x}{(x^2 + \\alpha^2)^{1/2}}
    """

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        return heaviside(x)

    @staticmethod
    def backward(ctx, dy):
        x, alpha = ctx.saved_tensors
        return dy * (-(x.pow(2) / (2 * (alpha ** 2 + x.pow(2)).pow(1.5))) +
            1 / (2 * (alpha ** 2 + x.pow(2)).sqrt())
            ) * 2 * alpha, torch.zeros_like(x)


heavi_circ_fn = HeaviCirc.apply


class HeaviTanh(torch.autograd.Function):
    """Approximation of the heaviside step function as

    .. math::
        h(x,k) = \\frac{1}{2} + \\frac{1}{2} \\text{tanh}(k x)
    """

    @staticmethod
    def forward(ctx, x, k):
        ctx.save_for_backward(x, k)
        return heaviside(x)

    @staticmethod
    def backward(ctx, dy):
        x, k = ctx.saved_tensors
        dtanh = 1 - (x * k).tanh().pow(2)
        return dy * dtanh, None


heavi_tanh_fn = HeaviTanh.apply


class HeaviTent(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        return heaviside(x)

    @staticmethod
    def backward(ctx, dy):
        x, alpha = ctx.saved_tensors
        return torch.relu(1 - torch.abs(x)) * alpha * dy, None


heavi_tent_fn = HeaviTent.apply


class Logistic(torch.autograd.Function):
    """Probalistic approximation of the heaviside step function as

    .. math::
        z \\sim p(\\frac{1}{2} + \\frac{1}{2} \\text{tanh}(k x))
    """

    @staticmethod
    def forward(ctx, x, k):
        ctx.save_for_backward(x, k)
        p = 0.5 + 0.5 * torch.tanh(k * x)
        return torch.distributions.bernoulli.Bernoulli(probs=p).sample()

    @staticmethod
    def backward(ctx, dy):
        x, k = ctx.saved_tensors
        dtanh = 1 - (x * k).tanh().pow(2)
        return dy * dtanh, None


logistic_fn = Logistic.apply


class SuperSpike(torch.autograd.Function):
    """SuperSpike surrogate gradient as described in Section 3.3.2 of

        F. Zenke, S. Ganguli, "SuperSpike: Supervised Learning in
                               Multilayer Spiking Neural Networks",
        Neural Computation 30, 1514–1541 (2018),
        doi:10.1162/neco_a_01086
    """

    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, alpha: float) ->torch.Tensor:
        ctx.save_for_backward(input_tensor)
        ctx.alpha = alpha
        return heaviside(input_tensor)

    @staticmethod
    def backward(ctx, grad_output):
        inp, = ctx.saved_tensors
        alpha = ctx.alpha
        grad_input = grad_output.clone()
        grad = grad_input / (alpha * torch.abs(inp) + 1.0).pow(2)
        return grad, None


def super_fn(x: torch.Tensor, alpha: float=100.0) ->torch.Tensor:
    return SuperSpike.apply(x, alpha)


def threshold(x: torch.Tensor, method: str, alpha: float) ->torch.Tensor:
    if method == 'heaviside':
        return heaviside(x)
    elif method == 'super':
        return super_fn(x, alpha)
    elif method == 'tanh':
        return heavi_tanh_fn(x, alpha)
    elif method == 'tent':
        return heavi_tent_fn(x, alpha)
    elif method == 'circ':
        return heavi_circ_fn(x, alpha)
    elif method == 'logistic':
        return logistic_fn(x, alpha)
    else:
        raise ValueError(
            f'Attempted to apply threshold function {method}, but no such ' +
            'function exist. We currently support heaviside, super, ' +
            'tanh, tent, circ, and logistic.')


def coba_lif_step(input_tensor: torch.Tensor, state: CobaLIFState,
    input_weights: torch.Tensor, recurrent_weights: torch.Tensor, p:
    CobaLIFParameters=CobaLIFParameters(), dt: float=0.001) ->Tuple[torch.
    Tensor, CobaLIFState]:
    """Euler integration step for a conductance based LIF neuron.

    Parameters:
        input_tensor (torch.Tensor): the input spikes at the current time step
        s (CobaLIFState): current state of the neuron
        input_weights (torch.Tensor): input weights
            (sign determines  contribution to inhibitory / excitatory input)
        recurrent_weights (torch.Tensor): recurrent weights
            (sign determines contribution to inhibitory / excitatory input)
        p (CobaLIFParameters): parameters of the neuron
        dt (float): Integration time step
    """
    dg_e = -dt * p.tau_syn_exc_inv * state.g_e
    g_e = state.g_e + dg_e
    dg_i = -dt * p.tau_syn_inh_inv * state.g_i
    g_i = state.g_i + dg_i
    g_e = g_e + torch.nn.functional.linear(input_tensor, torch.nn.
        functional.relu(input_weights))
    g_i = g_i + torch.nn.functional.linear(input_tensor, torch.nn.
        functional.relu(-input_weights))
    g_e = g_e + torch.nn.functional.linear(state.z, torch.nn.functional.
        relu(recurrent_weights))
    g_i = g_i + torch.nn.functional.linear(state.z, torch.nn.functional.
        relu(-recurrent_weights))
    dv = dt * p.c_m_inv * (p.g_l * (p.v_rest - state.v) + g_e * (p.e_rev_E -
        state.v) + g_i * (p.e_rev_I - state.v))
    v = state.v + dv
    z_new = threshold(v - p.v_thresh, p.method, p.alpha)
    v = (1 - z_new) * v + z_new * p.v_reset
    return z_new, CobaLIFState(z_new, v, g_e, g_i)


class CobaLIFCell(torch.nn.Module):
    """Module that computes a single euler-integration step of a conductance based
    LIF neuron-model. More specifically it implements one integration step of
    the following ODE

    .. math::
        \\begin{align*}
            \\dot{v} &= 1/c_{\\text{mem}} (g_l (v_{\\text{leak}} - v)               + g_e (E_{\\text{rev_e}} - v) + g_i (E_{\\text{rev_i}} - v)) \\\\
            \\dot{g_e} &= -1/\\tau_{\\text{syn}} g_e \\\\
            \\dot{g_i} &= -1/\\tau_{\\text{syn}} g_i
        \\end{align*}

    together with the jump condition

    .. math::
        z = \\Theta(v - v_{\\text{th}})

    and transition equations

    .. math::
        \\begin{align*}
            v &= (1-z) v + z v_{\\text{reset}} \\\\
            g_e &= g_e + \\text{relu}(w_{\\text{input}}) z_{\\text{in}} \\\\
            g_e &= g_e + \\text{relu}(w_{\\text{rec}}) z_{\\text{rec}} \\\\
            g_i &= g_i + \\text{relu}(-w_{\\text{input}}) z_{\\text{in}} \\\\
            g_i &= g_i + \\text{relu}(-w_{\\text{rec}}) z_{\\text{rec}} \\\\
        \\end{align*}

    where :math:`z_{\\text{rec}}` and :math:`z_{\\text{in}}` are the recurrent
    and input spikes respectively.

    Parameters:
        input_size (int): Size of the input.
        hidden_size (int): Size of the hidden state.
        p (LIFParameters): Parameters of the LIF neuron model.
        dt (float): Time step to use.

    Examples:

        >>> batch_size = 16
        >>> lif = CobaLIFCell(10, 20)
        >>> input = torch.randn(batch_size, 10)
        >>> s0 = lif.initial_state(batch_size)
        >>> output, s0 = lif(input, s0)
    """

    def __init__(self, input_size, hidden_size, p: CobaLIFParameters=
        CobaLIFParameters(), dt: float=0.001):
        super(CobaLIFCell, self).__init__()
        self.input_weights = torch.nn.Parameter(torch.randn(hidden_size,
            input_size) / np.sqrt(input_size))
        self.recurrent_weights = torch.nn.Parameter(torch.randn(hidden_size,
            hidden_size) / np.sqrt(hidden_size))
        self.p = p
        self.dt = dt

    def initial_state(self, batch_size: int, device: torch.device, dtype=
        torch.float) ->CobaLIFState:
        return CobaLIFState(z=torch.zeros(batch_size, self.hidden_size,
            device=device, dtype=dtype), v=torch.zeros(batch_size, self.
            hidden_size, device=device, dtype=dtype), g_e=torch.zeros(
            batch_size, self.hidden_size, device=device, dtype=dtype), g_i=
            torch.zeros(batch_size, self.hidden_size, device=device, dtype=
            dtype))

    def forward(self, input_tensor: torch.Tensor, state: CobaLIFState) ->Tuple[
        torch.Tensor, CobaLIFState]:
        return coba_lif_step(input_tensor, state, self.input_weights, self.
            recurrent_weights, p=self.p, dt=self.dt)


class ConstantCurrentLIFEncoder(torch.nn.Module):

    def __init__(self, seq_length: int, p: lif.LIFParameters=lif.
        LIFParameters(), dt: float=0.001):
        """
        Encodes input currents as fixed (constant) voltage currents, and simulates the spikes that 
        occur during a number of timesteps/iterations (seq_length).

        Example:
            >>> data = torch.tensor([2, 4, 8, 16])
            >>> seq_length = 2 # Simulate two iterations
            >>> constant_current_lif_encode(data, seq_length)
            (tensor([[0.2000, 0.4000, 0.8000, 0.0000],   # State in terms of membrane voltage
                    [0.3800, 0.7600, 0.0000, 0.0000]]), 
            tensor([[0., 0., 0., 1.],                   # Spikes for each iteration
                    [0., 0., 1., 1.]]))

        Parameters:
            seq_length (int): The number of iterations to simulate
            p (LIFParameters): Initial neuronp. Defaults to zero.
            dt (float): Time delta between simulation steps
        """
        super(ConstantCurrentLIFEncoder, self).__init__()
        self.seq_length = seq_length
        self.p = p
        self.dt = dt

    def forward(self, input_currents):
        return encode.constant_current_lif_encode(input_currents,
            seq_length=self.seq_length, p=self.p, dt=self.dt)


class PoissonEncoder(torch.nn.Module):

    def __init__(self, seq_length: int, f_max: float=100, dt: float=0.001):
        """
        Encodes a tensor of input values, which are assumed to be in the
        range [0,1] (if not signed, [-1,1] if signed) 
        into a tensor of one dimension higher of binary values,
        which represent input spikes.

        Parameters:
            input_values (torch.Tensor): Input data tensor with values assumed to be in the interval [0,1].
            sequence_length (int): Number of time steps in the resulting spike train.
            f_max (float): Maximal frequency (in Hertz) which will be emitted.
            dt (float): Integration time step (should coincide with the integration time step used in the model)
        """
        super(PoissonEncoder, self).__init__()
        self.seq_length = seq_length
        self.f_max = f_max
        self.dt = dt

    def forward(self, x):
        return encode.poisson_encode(x, self.seq_length, f_max=self.f_max,
            dt=self.dt)


class SignedPoissonEncoder(torch.nn.Module):

    def __init__(self, seq_length: int, f_max: float=100, dt: float=0.001):
        """
        Encodes a tensor of input values, which are assumed to be in the
        range [-1,1] (if not signed, [-1,1] if signed) 
        into a tensor of one dimension higher of binary values,
        which represent input spikes.

        Parameters:
            sequence_length (int): Number of time steps in the resulting spike train.
            f_max (float): Maximal frequency (in Hertz) which will be emitted.
            dt (float): Integration time step (should coincide with the integration time step used in the model)
        """
        super(SignedPoissonEncoder, self).__init__()
        self.seq_length = seq_length
        self.f_max = f_max
        self.dt = dt

    def forward(self, x):
        return encode.signed_poisson_encode(x, self.seq_length, f_max=self.
            f_max, dt=self.dt)


class SpikeLatencyLIFEncoder(torch.nn.Module):
    """Encodes an input value by the time the first spike occurs.
    Similar to the ConstantCurrentLIFEncoder, but the LIF can be
    thought to have an infinite refractory period.

    Parameters:
        sequence_length (int): Number of time steps in the resulting spike train.
        p (LIFParameters): Parameters of the LIF neuron model.
        dt (float): Integration time step (should coincide with the integration time step used in the model)
    """

    def __init__(self, seq_length, p=lif.LIFParameters(), dt=0.001):
        super(SpikeLatencyLIFEncoder, self).__init__()
        self.seq_length = seq_length
        self.p = p
        self.dt = dt

    def forward(self, input_current):
        return encode.spike_latency_lif_encode(input_current, self.
            seq_length, self.p, self.dt)


class SpikeLatencyEncoder(torch.nn.Module):
    """
    For all neurons, remove all but the first spike. This encoding basically measures the time it takes for a 
    neuron to spike *first*. Assuming that the inputs are constant, this makes sense in that strong inputs spikes
    fast.

    See `R. Van Rullen & S. J. Thorpe (2001): Rate Coding Versus Temporal Order Coding: What the Retinal Ganglion Cells Tell the Visual Cortex <https://doi.org/10.1162/08997660152002852>`_.

    Spikes are identified by their unique position in the input array. 

    Example:
        >>> data = torch.tensor([[0, 1, 1], [1, 1, 1]])
        >>> encoder = torch.nn.Sequential(
                        ConstantCurrentLIFEncoder()
                        SpikeLatencyEncoder()
                        )
        >>> encoder(data)
        tensor([[0, 1, 1],
                [1, 0, 0]])
    """

    def forward(self, input_spikes):
        return encode.spike_latency_encode(input_spikes)


def lif_current_encoder(input_current: torch.Tensor, voltage: torch.Tensor,
    p: LIFParameters=LIFParameters(), dt: float=0.001) ->Tuple[torch.Tensor,
    torch.Tensor]:
    """Computes a single euler-integration step of a leaky integrator. More
    specifically it implements one integration step of the following ODE

    .. math::
        \\begin{align*}
            \\dot{v} &= 1/\\tau_{\\text{mem}} (v_{\\text{leak}} - v + i) \\\\
            \\dot{i} &= -1/\\tau_{\\text{syn}} i
        \\end{align*}

    Parameters:
        input (torch.Tensor): the input current at the current time step
        voltage (torch.Tensor): current state of the LIF neuron
        p (LIFParameters): parameters of a leaky integrate and fire neuron
        dt (float): Integration timestep to use
    """
    dv = dt * p.tau_mem_inv * (p.v_leak - voltage + input_current)
    voltage = voltage + dv
    z = threshold(voltage - p.v_th, p.method, p.alpha)
    voltage = voltage - z * (voltage - p.v_reset)
    return z, voltage


def constant_current_lif_encode(input_current: torch.Tensor, seq_length:
    int, p: LIFParameters=LIFParameters(), dt: float=0.001) ->torch.Tensor:
    """
    Encodes input currents as fixed (constant) voltage currents, and simulates the spikes that 
    occur during a number of timesteps/iterations (seq_length).

    Example:
        >>> data = torch.tensor([2, 4, 8, 16])
        >>> seq_length = 2 # Simulate two iterations
        >>> constant_current_lif_encode(data, seq_length)
         # State in terms of membrane voltage
        (tensor([[0.2000, 0.4000, 0.8000, 0.0000],   
                 [0.3800, 0.7600, 0.0000, 0.0000]]), 
         # Spikes for each iteration
         tensor([[0., 0., 0., 1.],                   
                 [0., 0., 1., 1.]]))

    Parameters:
        input_current (torch.Tensor): The input tensor, representing LIF current
        seq_length (int): The number of iterations to simulate
        p (LIFParameters): Initial neuronp. Defaults to zero.
        dt (float): Time delta between simulation steps

    Returns:
        A tensor with an extra dimension of size `seq_length` containing spikes (1) or no spikes (0).
    """
    v = torch.zeros(*input_current.shape, device=input_current.device)
    z = torch.zeros(*input_current.shape, device=input_current.device)
    spikes = torch.zeros(seq_length, *input_current.shape, device=
        input_current.device)
    for ts in range(seq_length):
        z, v = lif_current_encoder(input_current=input_current, voltage=v,
            p=p, dt=dt)
        spikes[ts] = z
    return spikes


class IFConstantCurrentEncoder(torch.nn.Module):

    def __init__(self, seq_length, tau_mem_inv=1.0 / 0.01, v_th=1.0,
        v_reset=0.0, dt: float=0.001, device='cpu'):
        super(IFConstantCurrentEncoder, self).__init__()
        self.seq_length = seq_length
        self.tau_mem_inv = tau_mem_inv
        self.v_th = v_th
        self.v_reset = v_reset
        self.device = device
        self.dt = dt

    def forward(self, x):
        lif_parameters = LIFParameters(tau_mem_inv=self.tau_mem_inv, v_th=
            self.v_th, v_reset=self.v_reset)
        return constant_current_lif_encode(x, self.v, p=lif_parameters, dt=
            self.dt)


class LIParameters(NamedTuple):
    """Parameters of a leaky integrator

    Parameters:
        tau_syn_inv (torch.Tensor): inverse synaptic time constant
        tau_mem_inv (torch.Tensor): inverse membrane time constant
        v_leak (torch.Tensor): leak potential
        v_reset (torch.Tensor): reset potential
    """
    tau_syn_inv: torch.Tensor = torch.as_tensor(1.0 / 0.005)
    tau_mem_inv: torch.Tensor = torch.as_tensor(1.0 / 0.01)
    v_leak: torch.Tensor = torch.as_tensor(0.0)
    v_reset: torch.Tensor = torch.as_tensor(0.0)


class LIState(NamedTuple):
    """State of a leaky-integrator

    Parameters:
        v (torch.Tensor): membrane voltage
        i (torch.Tensor): input current
    """
    v: torch.Tensor
    i: torch.Tensor


def li_step(input_tensor: torch.Tensor, state: LIState, input_weights:
    torch.Tensor, p: LIParameters=LIParameters(), dt: float=0.001) ->Tuple[
    torch.Tensor, LIState]:
    """Single euler integration step of a leaky-integrator.
    More specifically it implements a discretized version of the ODE

    .. math::

        \\begin{align*}
            \\dot{v} &= 1/\\tau_{\\text{mem}} (v_{\\text{leak}} - v + i) \\\\
            \\dot{i} &= -1/\\tau_{\\text{syn}} i
        \\end{align*}


    and transition equations

    .. math::
        i = i + w i_{\\text{in}}

    Parameters:
        input_tensor (torch.Tensor); Input spikes
        s (LIState): state of the leaky integrator
        input_weights (torch.Tensor): weights for incoming spikes
        p (LIParameters): parameters of the leaky integrator
        dt (float): integration timestep to use
    """
    dv = dt * p.tau_mem_inv * (p.v_leak - state.v + state.i)
    v_new = state.v + dv
    di = -dt * p.tau_syn_inv * state.i
    i_decayed = state.i + di
    i_new = i_decayed + torch.nn.functional.linear(input_tensor, input_weights)
    return v_new, LIState(v_new, i_new)


class LICell(torch.nn.Module):
    """Cell for a leaky-integrator.
    More specifically it implements a discretized version of the ODE

    .. math::

        \\begin{align*}
            \\dot{v} &= 1/\\tau_{\\text{mem}} (v_{\\text{leak}} - v + i) \\\\
            \\dot{i} &= -1/\\tau_{\\text{syn}} i
        \\end{align*}


    and transition equations

    .. math::
        i = i + w i_{\\text{in}}

    Parameters:
        input_features (int); Input feature dimension
        output_features (int): Output feature dimension
        p (LIParameters): parameters of the leaky integrator
        dt (float): integration timestep to use
    """

    def __init__(self, input_features: int, output_features: int, p:
        LIParameters=LIParameters(), dt: float=0.001):
        super(LICell, self).__init__()
        self.input_weights = torch.nn.Parameter(torch.randn(output_features,
            input_features) / np.sqrt(input_features))
        self.p = p
        self.dt = dt
        self.output_features = output_features

    def initial_state(self, batch_size, device, dtype=torch.float) ->LIState:
        return LIState(v=torch.zeros((batch_size, self.output_features),
            device=device, dtype=dtype), i=torch.zeros((batch_size, self.
            output_features), device=device, dtype=dtype))

    def forward(self, input_tensor: torch.Tensor, state: LIState) ->Tuple[
        torch.Tensor, LIState]:
        return li_step(input_tensor, state, self.input_weights, p=self.p,
            dt=self.dt)


def li_feed_forward_step(input_tensor: torch.Tensor, state: LIState, p:
    LIParameters=LIParameters(), dt: float=0.001) ->Tuple[torch.Tensor, LIState
    ]:
    dv = dt * p.tau_mem_inv * (p.v_leak - state.v + state.i)
    v_new = state.v + dv
    di = -dt * p.tau_syn_inv * state.i
    i_decayed = state.i + di
    i_new = i_decayed + input_tensor
    return v_new, LIState(v_new, i_new)


class LIFeedForwardCell(torch.nn.Module):
    """Cell for a leaky-integrator.
    More specifically it implements a discretized version of the ODE

    .. math::

        \\begin{align*}
            \\dot{v} &= 1/\\tau_{\\text{mem}} (v_{\\text{leak}} - v + i) \\\\
            \\dot{i} &= -1/\\tau_{\\text{syn}} i
        \\end{align*}


    and transition equations

    .. math::
        i = i + w i_{\\text{in}}

    Parameters:
        shape: Shape of the preprocessed input spikes
        p (LIParameters): parameters of the leaky integrator
        dt (float): integration timestep to use
    """

    def __init__(self, shape, p: LIParameters=LIParameters(), dt: float=0.001):
        super(LIFeedForwardCell, self).__init__()
        self.p = p
        self.dt = dt
        self.shape = shape

    def initial_state(self, batch_size, device, dtype=torch.float):
        return LIState(v=torch.zeros(batch_size, *self.shape, device=device,
            dtype=dtype), i=torch.zeros(batch_size, *self.shape, device=
            device, dtype=dtype))

    def forward(self, input_tensor: torch.Tensor, s: LIState) ->Tuple[torch
        .Tensor, LIState]:
        return li_feed_forward_step(input_tensor, s, p=self.p, dt=self.dt)


class LIFState(NamedTuple):
    """State of a LIF neuron

    Parameters:
        z (torch.Tensor): recurrent spikes
        v (torch.Tensor): membrane potential
        i (torch.Tensor): synaptic input current
    """
    z: torch.Tensor
    v: torch.Tensor
    i: torch.Tensor


def lif_step(input_tensor: torch.Tensor, state: LIFState, input_weights:
    torch.Tensor, recurrent_weights: torch.Tensor, p: LIFParameters=
    LIFParameters(), dt: float=0.001) ->Tuple[torch.Tensor, LIFState]:
    """Computes a single euler-integration step of a LIF neuron-model. More
    specifically it implements one integration step of the following ODE

    .. math::
        \\begin{align*}
            \\dot{v} &= 1/\\tau_{\\text{mem}} (v_{\\text{leak}} - v + i) \\\\
            \\dot{i} &= -1/\\tau_{\\text{syn}} i
        \\end{align*}

    together with the jump condition

    .. math::
        z = \\Theta(v - v_{\\text{th}})

    and transition equations

    .. math::
        \\begin{align*}
            v &= (1-z) v + z v_{\\text{reset}} \\\\
            i &= i + w_{\\text{input}} z_{\\text{in}} \\\\
            i &= i + w_{\\text{rec}} z_{\\text{rec}}
        \\end{align*}

    where :math:`z_{\\text{rec}}` and :math:`z_{\\text{in}}` are the recurrent
    and input spikes respectively.

    Parameters:
        input_tensor (torch.Tensor): the input spikes at the current time step
        s (LIFState): current state of the LIF neuron
        input_weights (torch.Tensor): synaptic weights for incoming spikes
        recurrent_weights (torch.Tensor): synaptic weights for recurrent spikes
        p (LIFParameters): parameters of a leaky integrate and fire neuron
        dt (float): Integration timestep to use
    """
    dv = dt * p.tau_mem_inv * (p.v_leak - state.v + state.i)
    v_decayed = state.v + dv
    di = -dt * p.tau_syn_inv * state.i
    i_decayed = state.i + di
    z_new = threshold(v_decayed - p.v_th, p.method, p.alpha)
    v_new = (1 - z_new) * v_decayed + z_new * p.v_reset
    i_new = i_decayed + torch.nn.functional.linear(input_tensor, input_weights
        ) + torch.nn.functional.linear(state.z, recurrent_weights)
    return z_new, LIFState(z_new, v_new, i_new)


class LIFCell(torch.nn.Module):
    """Module that computes a single euler-integration step of a LIF
    neuron-model. More specifically it implements one integration step
    of the following ODE

    .. math::
        \\begin{align*}
            \\dot{v} &= 1/\\tau_{\\text{mem}} (v_{\\text{leak}} - v + i) \\\\
            \\dot{i} &= -1/\\tau_{\\text{syn}} i
        \\end{align*}

    together with the jump condition

    .. math::
        z = \\Theta(v - v_{\\text{th}})

    and transition equations

    .. math::
        \\begin{align*}
            v &= (1-z) v + z v_{\\text{reset}} \\\\
            i &= i + w_{\\text{input}} z_{\\text{in}} \\\\
            i &= i + w_{\\text{rec}} z_{\\text{rec}}
        \\end{align*}

    where :math:`z_{\\text{rec}}` and :math:`z_{\\text{in}}` are the
    recurrent and input spikes respectively.

    Parameters:
        input_size (int): Size of the input.
        hidden_size (int): Size of the hidden state.
        p (LIFParameters): Parameters of the LIF neuron model.
        dt (float): Time step to use.

    Examples:

        >>> batch_size = 16
        >>> lif = LIFCell(10, 20)
        >>> input = torch.randn(batch_size, 10)
        >>> s0 = lif.initial_state(batch_size)
        >>> output, s0 = lif(input, s0)
    """

    def __init__(self, input_size, hidden_size, p: LIFParameters=
        LIFParameters(), dt: float=0.001):
        super(LIFCell, self).__init__()
        self.input_weights = torch.nn.Parameter(torch.randn(hidden_size,
            input_size) * np.sqrt(2 / hidden_size))
        self.recurrent_weights = torch.nn.Parameter(torch.randn(hidden_size,
            hidden_size) * np.sqrt(2 / hidden_size))
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.p = p
        self.dt = dt

    def extra_repr(self):
        s = f'{self.input_size}, {self.hidden_size}, p={self.p}, dt={self.dt}'
        return s

    def initial_state(self, batch_size, device, dtype=torch.float) ->LIFState:
        return LIFState(z=torch.zeros(batch_size, self.hidden_size, device=
            device, dtype=dtype), v=torch.zeros(batch_size, self.
            hidden_size, device=device, dtype=dtype), i=torch.zeros(
            batch_size, self.hidden_size, device=device, dtype=dtype))

    def forward(self, input_tensor: torch.Tensor, state: LIFState) ->Tuple[
        torch.Tensor, LIFState]:
        return lif_step(input_tensor, state, self.input_weights, self.
            recurrent_weights, p=self.p, dt=self.dt)


class LIFLayer(torch.nn.Module):

    def __init__(self, *cell_args, **kw_args):
        super(LIFLayer, self).__init__()
        self.cell = LIFCell(*cell_args, **kw_args)

    def forward(self, input_tensor: torch.Tensor, state: LIFState) ->Tuple[
        torch.Tensor, LIFState]:
        inputs = input_tensor.unbind(0)
        outputs = []
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs += [out]
        return torch.stack(outputs), state


class LIFFeedForwardState(NamedTuple):
    """State of a feed forward LIF neuron

    Parameters:
        v (torch.Tensor): membrane potential
        i (torch.Tensor): synaptic input current
    """
    v: torch.Tensor
    i: torch.Tensor


def lif_feed_forward_step(input_tensor: torch.Tensor, state:
    LIFFeedForwardState=LIFFeedForwardState(0, 0), p: LIFParameters=
    LIFParameters(), dt: float=0.001) ->Tuple[torch.Tensor, LIFFeedForwardState
    ]:
    """Computes a single euler-integration step for a lif neuron-model.
    It takes as input the input current as generated by an arbitrary torch
    module or function. More specifically it implements one integration
    step of the following ODE

    .. math::
        \\begin{align*}
            \\dot{v} &= 1/\\tau_{\\text{mem}} (v_{\\text{leak}} - v + i) \\\\
            \\dot{i} &= -1/\\tau_{\\text{syn}} i
        \\end{align*}

    together with the jump condition

    .. math::
        z = \\Theta(v - v_{\\text{th}})

    and transition equations

    .. math::
        \\begin{align*}
            v &= (1-z) v + z v_{\\text{reset}} \\\\
            i &= i + i_{\\text{in}}
        \\end{align*}

    where :math:`i_{\\text{in}}` is meant to be the result of applying an
    arbitrary pytorch module (such as a convolution) to input spikes.

    Parameters:
        input_tensor (torch.Tensor): the input spikes at the current time step
        state (LIFFeedForwardState): current state of the LIF neuron
        p (LIFParameters): parameters of a leaky integrate and fire neuron
        dt (float): Integration timestep to use
    """
    dv = dt * p.tau_mem_inv * (p.v_leak - state.v + state.i)
    v_decayed = state.v + dv
    di = -dt * p.tau_syn_inv * state.i
    i_decayed = state.i + di
    z_new = threshold(v_decayed - p.v_th, p.method, p.alpha)
    v_new = (1 - z_new) * v_decayed + z_new * p.v_reset
    i_new = i_decayed + input_tensor
    return z_new, LIFFeedForwardState(v_new, i_new)


class LIFFeedForwardCell(torch.nn.Module):
    """Module that computes a single euler-integration step of a LIF neuron.
    It takes as input the input current as generated by an arbitrary torch
    module or function. More specifically it implements one integration step
    of the following ODE

    .. math::
        \\begin{align*}
            \\dot{v} &= 1/\\tau_{\\text{mem}} (v_{\\text{leak}} - v + i) \\\\
            \\dot{i} &= -1/\\tau_{\\text{syn}} i
        \\end{align*}

    together with the jump condition

    .. math::
        z = \\Theta(v - v_{\\text{th}})

    and transition equations

    .. math::
        i = i + i_{\\text{in}}

    where :math:`i_{\\text{in}}` is meant to be the result of applying
    an arbitrary pytorch module (such as a convolution) to input spikes.

    Parameters:
        shape: Shape of the feedforward state.
        p (LIFParameters): Parameters of the LIF neuron model.
        dt (float): Time step to use.

    Examples:

        >>> batch_size = 16
        >>> lif = LIFFeedForwardCell((20, 30))
        >>> data = torch.randn(batch_size, 20, 30)
        >>> s0 = lif.initial_state(batch_size, "cpu")
        >>> output, s0 = lif(data, s0)
    """

    def __init__(self, shape, p: LIFParameters=LIFParameters(), dt: float=0.001
        ):
        super(LIFFeedForwardCell, self).__init__()
        self.shape = shape
        self.p = p
        self.dt = dt

    def extra_repr(self):
        s = f'{self.shape}, p={self.p}, dt={self.dt}'
        return s

    def initial_state(self, batch_size, device, dtype=None
        ) ->LIFFeedForwardState:
        return LIFFeedForwardState(v=torch.zeros(batch_size, *self.shape,
            device=device, dtype=dtype), i=torch.zeros(batch_size, *self.
            shape, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor, state: LIFFeedForwardState) ->Tuple[
        torch.Tensor, LIFFeedForwardState]:
        return lif_feed_forward_step(x, state, p=self.p, dt=self.dt)


class CorrelationSensorState(NamedTuple):
    post_pre: torch.Tensor
    correlation_trace: torch.Tensor
    anti_correlation_trace: torch.Tensor


class CorrelationSensorParameters(NamedTuple):
    eta_p: torch.Tensor = torch.as_tensor(1.0)
    eta_m: torch.Tensor = torch.as_tensor(1.0)
    tau_ac_inv: torch.Tensor = torch.as_tensor(1.0 / 0.1)
    tau_c_inv: torch.Tensor = torch.as_tensor(1.0 / 0.1)


class LIFCorrelationParameters(NamedTuple):
    lif_parameters: LIFParameters = LIFParameters()
    input_correlation_parameters: CorrelationSensorParameters = (
        CorrelationSensorParameters())
    recurrent_correlation_parameters: CorrelationSensorParameters = (
        CorrelationSensorParameters())


class LIFCorrelationState(NamedTuple):
    lif_state: LIFState
    input_correlation_state: CorrelationSensorState
    recurrent_correlation_state: CorrelationSensorState


@torch.jit.script
def post_mask(weights, z):
    """Computes the mask produced by post-synaptic spikes on
    the synapse array.
    """
    return torch.zeros_like(weights) + z


@torch.jit.script
def post_pre_update(post_pre, post_spike_mask, pre_spike_mask):
    """Computes which synapses in the synapse array should be updated.
    """
    return heaviside(post_pre + post_spike_mask - pre_spike_mask)


@torch.jit.script
def pre_mask(weights, z):
    """Computes the mask produced by the pre-synaptic spikes on
    the synapse array."""
    return torch.transpose(torch.transpose(torch.zeros_like(weights), 1, 2) +
        z, 1, 2)


def correlation_sensor_step(z_pre: torch.Tensor, z_post: torch.Tensor,
    state: CorrelationSensorState, p: CorrelationSensorParameters=
    CorrelationSensorParameters(), dt: float=0.001) ->CorrelationSensorState:
    """Euler integration step of an idealized version of the correlation sensor
    as it is present on the BrainScaleS 2 chips.
    """
    dcorrelation_trace = dt * p.tau_c_inv * -state.correlation_trace
    correlation_trace_decayed = state.correlation_trace + (1 - state.post_pre
        ) * dcorrelation_trace
    danti_correlation_trace = dt * p.tau_ac_inv * -state.anti_correlation_trace
    anti_correlation_trace_decayed = (state.anti_correlation_trace + state.
        post_pre * danti_correlation_trace)
    pre_spike_mask = pre_mask(state.post_pre, z_pre)
    post_spike_mask = post_mask(state.post_pre, z_post)
    post_pre_new = post_pre_update(state.post_pre, post_spike_mask,
        pre_spike_mask)
    correlation_trace_new = (correlation_trace_decayed + p.eta_p *
        pre_spike_mask)
    anti_correlation_trace_new = (anti_correlation_trace_decayed + p.eta_m *
        post_spike_mask)
    return CorrelationSensorState(post_pre=post_pre_new, correlation_trace=
        correlation_trace_new, anti_correlation_trace=
        anti_correlation_trace_new)


def lif_correlation_step(input_tensor: torch.Tensor, state:
    LIFCorrelationState, input_weights: torch.Tensor, recurrent_weights:
    torch.Tensor, p: LIFCorrelationParameters=LIFCorrelationParameters(),
    dt: float=0.001) ->Tuple[torch.Tensor, LIFCorrelationState]:
    z_new, s_new = lif_step(input_tensor, state.lif_state, input_weights,
        recurrent_weights, p.lif_parameters, dt)
    input_correlation_state_new = correlation_sensor_step(z_pre=
        input_tensor, z_post=z_new, state=state.input_correlation_state, p=
        p.input_correlation_parameters, dt=dt)
    recurrent_correlation_state_new = correlation_sensor_step(z_pre=state.
        lif_state.z, z_post=z_new, state=state.recurrent_correlation_state,
        p=p.recurrent_correlation_parameters, dt=dt)
    return z_new, LIFCorrelationState(lif_state=s_new,
        input_correlation_state=input_correlation_state_new,
        recurrent_correlation_state=recurrent_correlation_state_new)


class LIFCorrelation(torch.nn.Module):

    def __init__(self, input_size, hidden_size, p: LIFCorrelationParameters
        =LIFCorrelationParameters(), dt: float=0.001):
        super(LIFCorrelation, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.p = p
        self.dt = dt

    def initial_state(self, batch_size, device, dtype=torch.float
        ) ->LIFCorrelationState:
        hidden_features = self.hidden_size
        input_features = self.input_size
        return LIFCorrelationState(lif_state=LIFState(z=torch.zeros(
            batch_size, hidden_features, device=device, dtype=dtype), v=
            torch.zeros(batch_size, hidden_features, device=device, dtype=
            dtype), i=torch.zeros(batch_size, hidden_features, device=
            device, dtype=dtype)), input_correlation_state=
            CorrelationSensorState(post_pre=torch.zeros((batch_size,
            input_features, hidden_features), device=device, dtype=dtype),
            correlation_trace=torch.zeros((batch_size, input_features,
            hidden_features), device=device, dtype=dtype).float(),
            anti_correlation_trace=torch.zeros((batch_size, input_features,
            hidden_features), device=device, dtype=dtype).float()),
            recurrent_correlation_state=CorrelationSensorState(
            correlation_trace=torch.zeros((batch_size, hidden_features,
            hidden_features), device=device, dtype=dtype),
            anti_correlation_trace=torch.zeros((batch_size, hidden_features,
            hidden_features), device=device, dtype=dtype), post_pre=torch.
            zeros((batch_size, hidden_features, hidden_features), device=
            device, dtype=dtype)))

    def forward(self, input_tensor: torch.Tensor, state:
        LIFCorrelationState, input_weights: torch.Tensor, recurrent_weights:
        torch.Tensor) ->Tuple[torch.Tensor, LIFCorrelationState]:
        return lif_correlation_step(input_tensor, state, input_weights,
            recurrent_weights, self.p, self.dt)


def lif_mc_step(input_tensor: torch.Tensor, state: LIFState, input_weights:
    torch.Tensor, recurrent_weights: torch.Tensor, g_coupling: torch.Tensor,
    p: LIFParameters=LIFParameters(), dt: float=0.001) ->Tuple[torch.Tensor,
    LIFState]:
    """Computes a single euler-integration step of a LIF multi-compartment
    neuron-model.

    Parameters:
        input_tensor (torch.Tensor): the input spikes at the current time step
        s (LIFState): current state of the neuron
        input_weights (torch.Tensor): synaptic weights for incoming spikes
        recurrent_weights (torch.Tensor): synaptic weights for recurrent spikes
        g_coupling (torch.Tensor): conductances between the neuron compartments
        p (LIFParameters): neuron parameters
        dt (float): Integration timestep to use
    """
    v_new = state.v + dt * torch.nn.functional.linear(state.v, g_coupling)
    return lif_step(input_tensor, LIFState(state.z, v_new, state.i),
        input_weights, recurrent_weights, p, dt)


class LIFMCCell(torch.nn.Module):
    """Computes a single euler-integration step of a LIF multi-compartment
    neuron-model.

    .. math::
        \\begin{align*}
            \\dot{v} &= 1/\\tau_{\\text{mem}} (v_{\\text{leak}}             - g_{\\text{coupling}} v + i) \\\\
            \\dot{i} &= -1/\\tau_{\\text{syn}} i
        \\end{align*}

    together with the jump condition

    .. math::
        z = \\Theta(v - v_{\\text{th}})

    and transition equations

    .. math::
        \\begin{align*}
            v &= (1-z) v + z v_{\\text{reset}} \\\\
            i &= i + w_{\\text{input}} z_{\\text{in}} \\\\
            i &= i + w_{\\text{rec}} z_{\\text{rec}}
        \\end{align*}

    where :math:`z_{\\text{rec}}` and :math:`z_{\\text{in}}` are the
    recurrent and input spikes respectively.


    Parameters:
        input (torch.Tensor): the input spikes at the current time step
        s (LIFState): current state of the neuron
        input_weights (torch.Tensor): synaptic weights for incoming spikes
        recurrent_weights (torch.Tensor): synaptic weights for recurrent spikes
        g_coupling (torch.Tensor): conductances between the neuron compartments
        p (LIFParameters): neuron parameters
        dt (float): Integration timestep to use
    """

    def __init__(self, input_size: int, hidden_size: int, p: LIFParameters=
        LIFParameters(), dt: float=0.001):
        self.input_weights = torch.nn.Parameter(torch.randn(hidden_size,
            input_size) / np.sqrt(input_size))
        self.recurrent_weights = torch.nn.Parameter(torch.randn(hidden_size,
            hidden_size) / np.sqrt(hidden_size))
        self.g_coupling = torch.nn.Parameter(torch.randn(hidden_size,
            hidden_size) / np.sqrt(hidden_size))
        self.p = p
        self.dt = dt

    def initial_state(self, batch_size: int, device: torch.device, dtype=
        torch.float) ->LIFState:
        return LIFState(z=torch.zeros(batch_size, self.hidden_size, device=
            device, dtype=dtype), v=torch.zeros(batch_size, self.
            hidden_size, device=device, dtype=dtype), i=torch.zeros(
            batch_size, self.hidden_size, device=device, dtype=dtype))

    def forward(self, input_tensor: torch.Tensor, state: LIFState) ->Tuple[
        torch.Tensor, LIFState]:
        return lif_mc_step(input_tensor, state, self.input_weights, self.
            recurrent_weights, self.g_coupling, p=self.p, dt=self.dt)


class LIFRefracParameters(NamedTuple):
    """Parameters of a LIF neuron with absolute refractory period.

    Parameters:
        lif (LIFParameters): parameters of the LIF neuron integration
        rho (torch.Tensor): refractory state (count towards zero)
    """
    lif: LIFParameters = LIFParameters()
    rho_reset: torch.Tensor = torch.as_tensor(5.0)


class LIFRefracState(NamedTuple):
    """State of a LIF neuron with absolute refractory period.

    Parameters:
        lif (LIFState): state of the LIF neuron integration
        rho (torch.Tensor): refractory state (count towards zero)
    """
    lif: LIFState
    rho: torch.Tensor


def lif_mc_refrac_step(input_tensor: torch.Tensor, state: LIFRefracState,
    input_weights: torch.Tensor, recurrent_weights: torch.Tensor,
    g_coupling: torch.Tensor, p: LIFRefracParameters=LIFRefracParameters(),
    dt: float=0.001) ->Tuple[torch.Tensor, LIFRefracState]:
    refrac_mask = threshold(state.rho, p.lif.method, p.lif.alpha)
    dv = (1 - refrac_mask) * dt * p.lif.tau_mem_inv * (p.lif.v_leak - state
        .lif.v + state.lif.i) + torch.nn.functional.linear(state.lif.v,
        g_coupling)
    v_decayed = state.lif.v + dv
    di = -dt * p.lif.tau_syn_inv * state.lif.i
    i_decayed = state.lif.i + di
    z_new = threshold(v_decayed - p.lif.v_th, p.lif.method, p.lif.alpha)
    v_new = (1 - z_new) * v_decayed + z_new * p.lif.v_reset
    i_new = i_decayed + torch.nn.functional.linear(input_tensor, input_weights
        ) + torch.nn.functional.linear(state.lif.z, recurrent_weights)
    rho_new = (1 - z_new) * torch.nn.functional.relu(state.rho - refrac_mask
        ) + z_new * p.rho_reset
    return z_new, LIFRefracState(LIFState(z_new, v_new, i_new), rho_new)


class LIFMCRefracCell(torch.nn.Module):

    def __init__(self, input_size: int, hidden_size: int, p:
        LIFRefracParameters=LIFRefracParameters(), dt: float=0.001):
        self.input_weights = torch.nn.Parameter(torch.randn(hidden_size,
            input_size) / np.sqrt(input_size))
        self.recurrent_weights = torch.nn.Parameter(torch.randn(hidden_size,
            hidden_size) / np.sqrt(hidden_size))
        self.g_coupling = torch.nn.Parameter(torch.randn(hidden_size,
            hidden_size) / np.sqrt(hidden_size))
        self.p = p
        self.dt = dt

    def initial_state(self, batch_size: int, device: torch.device, dtype:
        torch.float=torch.float) ->LIFRefracState:
        return LIFRefracState(lif=LIFState(z=torch.zeros(batch_size, self.
            hidden_size, device=device, dtype=dtype), v=torch.zeros(
            batch_size, self.hidden_size, device=device, dtype=dtype), i=
            torch.zeros(batch_size, self.hidden_size, device=device, dtype=
            dtype)), rho=torch.zeros(batch_size, self.hidden_size, device=
            device, dtype=dtype))

    def forward(self, input_tensor: torch.Tensor, state: LIFRefracState
        ) ->Tuple[torch.Tensor, LIFRefracState]:
        return lif_mc_refrac_step(input_tensor, state, self.input_weights,
            self.recurrent_weights, self.g_coupling, p=self.p, dt=self.dt)


def compute_refractory_update(state: LIFRefracState, z_new: torch.Tensor,
    v_new: torch.Tensor, p: LIFRefracParameters=LIFRefracParameters()) ->Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the refractory update.

    Parameters:
        state (LIFRefracState): Initial state of the refractory neuron.
        z_new (torch.Tensor): New spikes that were generated.
        v_new (torch.Tensor): New voltage after the lif update step.
        p (torch.Tensor): Refractoryp.
    """
    refrac_mask = threshold(state.rho, p.lif.method, p.lif.alpha)
    v_new = (1 - refrac_mask) * v_new + refrac_mask * state.lif.v
    z_new = (1 - refrac_mask) * z_new
    rho_new = (1 - z_new) * torch.nn.functional.relu(state.rho - refrac_mask
        ) + z_new * p.rho_reset
    return v_new, z_new, rho_new


def lif_refrac_step(input_tensor: torch.Tensor, state: LIFRefracState,
    input_weights: torch.Tensor, recurrent_weights: torch.Tensor, p:
    LIFRefracParameters=LIFRefracParameters(), dt: float=0.001) ->Tuple[
    torch.Tensor, LIFRefracState]:
    """Computes a single euler-integration step of a recurrently connected
     LIF neuron-model with a refractory period.

    Parameters:
        input_tensor (torch.Tensor): the input spikes at the current time step
        s (LIFRefracState): state at the current time step
        input_weights (torch.Tensor): synaptic weights for incoming spikes
        recurrent_weights (torch.Tensor): synaptic weights for recurrent spikes
        p (LIFRefracParameters): parameters of the lif neuron
        dt (float): Integration timestep to use
    """
    z_new, s_new = lif_step(input_tensor, state.lif, input_weights,
        recurrent_weights, p.lif, dt)
    v_new, z_new, rho_new = compute_refractory_update(state, z_new, s_new.v, p)
    return z_new, LIFRefracState(LIFState(z_new, v_new, s_new.i), rho_new)


class LIFRefracCell(torch.nn.Module):
    """Module that computes a single euler-integration step of a LIF
    neuron-model with absolute refractory period. More specifically it
    implements one integration step of the following ODE.

    .. math::
        \\begin{align*}
            \\dot{v} &= 1/\\tau_{\\text{mem}} (1-\\Theta(\\rho))             (v_{\\text{leak}} - v + i) \\\\
            \\dot{i} &= -1/\\tau_{\\text{syn}} i \\\\
            \\dot{\\rho} &= -1/\\tau_{\\text{refrac}} \\Theta(\\rho)
        \\end{align*}

    together with the jump condition

    .. math::
        \\begin{align*}
            z &= \\Theta(v - v_{\\text{th}}) \\\\
            z_r &= \\Theta(-\\rho)
        \\end{align*}

    and transition equations

    .. math::
        \\begin{align*}
            v &= (1-z) v + z v_{\\text{reset}} \\\\
            i &= i + w_{\\text{input}} z_{\\text{in}} \\\\
            i &= i + w_{\\text{rec}} z_{\\text{rec}} \\\\
            \\rho &= \\rho + z_r \\rho_{\\text{reset}}
        \\end{align*}

    where :math:`z_{\\text{rec}}` and :math:`z_{\\text{in}}` are the
    recurrent and input spikes respectively.

    Parameters:
        input (torch.Tensor): the input spikes at the current time step
        s (LIFRefracState): state at the current time step
        input_weights (torch.Tensor): synaptic weights for incoming spikes
        recurrent_weights (torch.Tensor): synaptic weights for recurrent spikes
        p (LIFRefracParameters): parameters of the lif neuron
        dt (float): Integration timestep to use

    Examples:

        >>> batch_size = 16
        >>> lif = LIFRefracCell(10, 20)
        >>> input = torch.randn(batch_size, 10)
        >>> s0 = lif.initial_state(batch_size)
        >>> output, s0 = lif(input, s0)
    """

    def __init__(self, input_size, hidden_size, p: LIFRefracParameters=
        LIFRefracParameters(), dt: float=0.001):
        super(LIFRefracCell, self).__init__()
        self.input_weights = torch.nn.Parameter(torch.randn(hidden_size,
            input_size) / np.sqrt(input_size))
        self.recurrent_weights = torch.nn.Parameter(torch.randn(hidden_size,
            hidden_size) / np.sqrt(hidden_size))
        self.hidden_size = hidden_size
        self.p = p
        self.dt = dt

    def initial_state(self, batch_size, device, dtype=torch.float
        ) ->LIFRefracState:
        return LIFRefracState(lif=LIFState(z=torch.zeros(batch_size, self.
            hidden_size, device=device, dtype=dtype), v=torch.zeros(
            batch_size, self.hidden_size, device=device, dtype=dtype), i=
            torch.zeros(batch_size, self.hidden_size, device=device, dtype=
            dtype)), rho=torch.zeros(batch_size, self.hidden_size, device=
            device, dtype=dtype))

    def forward(self, input_tensor: torch.Tensor, state: LIFRefracState
        ) ->Tuple[torch.Tensor, LIFRefracState]:
        return lif_refrac_step(input_tensor, state, self.input_weights,
            self.recurrent_weights, p=self.p, dt=self.dt)


class LIFRefracFeedForwardState(NamedTuple):
    """State of a feed forward LIF neuron with absolute refractory period.

    Parameters:
        lif (LIFFeedForwardState): state of the feed forward LIF
                                   neuron integration
        rho (torch.Tensor): refractory state (count towards zero)
    """
    lif: LIFFeedForwardState
    rho: torch.Tensor


def lif_refrac_feed_forward_step(input_tensor: torch.Tensor, state:
    LIFRefracFeedForwardState, p: LIFRefracParameters=LIFRefracParameters(),
    dt: float=0.001) ->Tuple[torch.Tensor, LIFRefracFeedForwardState]:
    """Computes a single euler-integration step of a feed forward
     LIF neuron-model with a refractory period.

    Parameters:
        input_tensor (torch.Tensor): the input spikes at the current time step
        s (LIFRefracFeedForwardState): state at the current time step
        p (LIFRefracParameters): parameters of the lif neuron
        dt (float): Integration timestep to use
    """
    z_new, s_new = lif_feed_forward_step(input_tensor, state.lif, p.lif, dt)
    v_new, z_new, rho_new = compute_refractory_update(state, z_new, s_new.v, p)
    return z_new, LIFRefracFeedForwardState(LIFFeedForwardState(v_new,
        s_new.i), rho_new)


class LIFRefracFeedForwardCell(torch.nn.Module):
    """Module that computes a single euler-integration step of a
    LIF neuron-model with absolute refractory period. More specifically
    it implements one integration step of the following ODE.

    .. math::
        \\begin{align*}
            \\dot{v} &= 1/\\tau_{\\text{mem}} (1-\\Theta(\\rho))             (v_{\\text{leak}} - v + i) \\\\
            \\dot{i} &= -1/\\tau_{\\text{syn}} i \\\\
            \\dot{\\rho} &= -1/\\tau_{\\text{refrac}} \\Theta(\\rho)
        \\end{align*}

    together with the jump condition

    .. math::
        \\begin{align*}
            z &= \\Theta(v - v_{\\text{th}}) \\\\
            z_r &= \\Theta(-\\rho)
        \\end{align*}

    and transition equations

    .. math::
        \\begin{align*}
            v &= (1-z) v + z v_{\\text{reset}} \\\\
            \\rho &= \\rho + z_r \\rho_{\\text{reset}}
        \\end{align*}

    Parameters:
        shape: Shape of the processed spike input
        p (LIFRefracParameters): parameters of the lif neuron
        dt (float): Integration timestep to use

    Examples:
        >>> batch_size = 16
        >>> lif = LIFRefracFeedForwardCell((20, 30))
        >>> input = torch.randn(batch_size, 20, 30)
        >>> s0 = lif.initial_state(batch_size)
        >>> output, s0 = lif(input, s0)
    """

    def __init__(self, shape, p: LIFRefracParameters=LIFRefracParameters(),
        dt: float=0.001):
        super(LIFRefracFeedForwardCell, self).__init__()
        self.shape = shape
        self.p = p
        self.dt = dt

    def initial_state(self, batch_size, device, dtype) ->LIFFeedForwardState:
        return LIFRefracFeedForwardState(LIFFeedForwardState(v=torch.zeros(
            batch_size, *self.shape, device=device, dtype=dtype), i=torch.
            zeros(batch_size, *self.shape, device=device, dtype=dtype)),
            rho=torch.zeros(batch_size, *self.shape, device=device, dtype=
            dtype))

    def forward(self, input_tensor: torch.Tensor, state:
        LIFRefracFeedForwardState) ->Tuple[torch.Tensor,
        LIFRefracFeedForwardState]:
        return lif_refrac_feed_forward_step(input_tensor, state, p=self.p,
            dt=self.dt)


class LSNNState(NamedTuple):
    """State of an LSNN neuron

    Parameters:
        z (torch.Tensor): recurrent spikes
        v (torch.Tensor): membrane potential
        i (torch.Tensor): synaptic input current
        b (torch.Tensor): threshold adaptation
    """
    z: torch.Tensor
    v: torch.Tensor
    i: torch.Tensor
    b: torch.Tensor


def lsnn_step(input_tensor: torch.Tensor, state: LSNNState, input_weights:
    torch.Tensor, recurrent_weights: torch.Tensor, p: LSNNParameters=
    LSNNParameters(), dt: float=0.001) ->Tuple[torch.Tensor, LSNNState]:
    """Euler integration step for LIF Neuron with threshold adaptation
    More specifically it implements one integration step of the following ODE

    .. math::
        \\begin{align*}
            \\dot{v} &= 1/\\tau_{\\text{mem}} (v_{\\text{leak}} - v + i) \\\\
            \\dot{i} &= -1/\\tau_{\\text{syn}} i \\\\
            \\dot{b} &= -1/\\tau_{b} b
        \\end{align*}

    together with the jump condition

    .. math::
        z = \\Theta(v - v_{\\text{th}} + b)

    and transition equations

    .. math::
        \\begin{align*}
            v &= (1-z) v + z v_{\\text{reset}} \\\\
            i &= i + w_{\\text{input}} z_{\\text{in}} \\\\
            i &= i + w_{\\text{rec}} z_{\\text{rec}} \\\\
            b &= b + \\beta z
        \\end{align*}

    where :math:`z_{\\text{rec}}` and :math:`z_{\\text{in}}` are the recurrent
    and input spikes respectively.

    Parameters:
        input_tensor (torch.Tensor): the input spikes at the current time step
        s (LSNNState): current state of the lsnn unit
        input_weights (torch.Tensor): synaptic weights for input spikes
        recurrent_weights (torch.Tensor): synaptic weights for recurrent spikes
        p (LSNNParameters): parameters of the lsnn unit
        dt (float): Integration timestep to use
    """
    dv = dt * p.tau_mem_inv * (p.v_leak - state.v + state.i)
    v_decayed = state.v + dv
    di = -dt * p.tau_syn_inv * state.i
    i_decayed = state.i + di
    db = dt * p.tau_adapt_inv * (p.v_th - state.b)
    b_decayed = state.b + db
    z_new = threshold(v_decayed - b_decayed, p.method, p.alpha)
    v_new = (1 - z_new) * v_decayed + z_new * p.v_reset
    i_new = i_decayed + torch.nn.functional.linear(input_tensor, input_weights
        ) + torch.nn.functional.linear(state.z, recurrent_weights)
    b_new = b_decayed + z_new * p.tau_adapt_inv * p.beta
    return z_new, LSNNState(z_new, v_new, i_new, b_new)


class LSNNCell(torch.nn.Module):
    """Module that computes a single euler-integration step of a LSNN
    neuron-model. More specifically it implements one integration step of
    the following ODE

    .. math::
        \\\\begin{align*}
            \\dot{v} &= 1/\\\\tau_{\\\\text{mem}} (v_{\\\\text{leak}} - v + i) \\\\\\\\
            \\dot{i} &= -1/\\\\tau_{\\\\text{syn}} i \\\\\\\\
            \\dot{b} &= -1/\\\\tau_{b} b
        \\end{align*}

    together with the jump condition

    .. math::
        z = \\Theta(v - v_{\\\\text{th}} + b)

    and transition equations

    .. math::
        \\\\begin{align*}
            v &= (1-z) v + z v_{\\\\text{reset}} \\\\\\\\
            i &= i + w_{\\\\text{input}} z_{\\\\text{in}} \\\\\\\\
            i &= i + w_{\\\\text{rec}} z_{\\\\text{rec}} \\\\\\\\
            b &= b + \\\\beta z
        \\end{align*}

    where :math:`z_{\\\\text{rec}}` and :math:`z_{\\\\text{in}}` are the
    recurrent and input spikes respectively.

    Parameters:
        input (torch.Tensor): the input spikes at the current time step
        s (LSNNState): current state of the lsnn unit
        input_weights (torch.Tensor): synaptic weights for input spikes
        recurrent_weights (torch.Tensor): synaptic weights for recurrent spikes
        p (LSNNParameters): parameters of the lsnn unit
        dt (float): Integration timestep to use
    """

    def __init__(self, input_features, output_features, p: LSNNParameters=
        LSNNParameters(), dt: float=0.001):
        super(LSNNCell, self).__init__()
        self.input_weights = torch.nn.Parameter(torch.randn(output_features,
            input_features) / np.sqrt(input_features))
        self.recurrent_weights = torch.nn.Parameter(torch.randn(
            output_features, output_features))
        self.input_features = input_features
        self.output_features = output_features
        self.p = p
        self.dt = dt

    def initial_state(self, batch_size, device, dtype=torch.float) ->LSNNState:
        """return the initial state of an LSNN neuron"""
        return LSNNState(z=torch.zeros(batch_size, self.output_features,
            device=device, dtype=dtype), v=torch.zeros(batch_size, self.
            output_features, device=device, dtype=dtype), i=torch.zeros(
            batch_size, self.output_features, device=device, dtype=dtype),
            b=torch.zeros(batch_size, self.output_features, device=device,
            dtype=dtype))

    def forward(self, input_tensor: torch.Tensor, state: LSNNState) ->Tuple[
        torch.Tensor, LSNNState]:
        return lsnn_step(input_tensor, state, self.input_weights, self.
            recurrent_weights, p=self.p, dt=self.dt)


class LSNNLayer(torch.nn.Module):
    """A Long short-term memory neuron module adapted from
        https://arxiv.org/abs/1803.09574

    Usage:
      >>> from norse.torch.module import LSNNLayer, LSNNCell
      >>> layer = LSNNLayer(LSNNCell, 2, 10)    // LSNNCell of shape 2 -> 10
      >>> state = layer.initial_state(5, "cpu") // 5 batch size running on CPU
      >>> data  = torch.zeros(2, 5, 2)          // Data of shape [5, 2, 10]
      >>> output, new_state = layer.forward(data, state)

    Parameters:
      cell (torch.nn.Module): the underling neuron module, uninitialized
      *cell_args: variable length input arguments for the underlying cell
                  constructor
    """

    def __init__(self, cell, *cell_args):
        super(LSNNLayer, self).__init__()
        self.cell = cell(*cell_args)

    def initial_state(self, batch_size, device, dtype=torch.float) ->LSNNState:
        """Return the initial state of the LSNN layer, as given by the
        internal LSNNCell"""
        return self.cell.initial_state(batch_size, device, dtype)

    def forward(self, input_tensor: torch.Tensor, state: LSNNState) ->Tuple[
        torch.Tensor, LSNNState]:
        inputs = input_tensor.unbind(0)
        outputs = []
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs += [out]
        return torch.stack(outputs), state


class LSNNFeedForwardState(NamedTuple):
    """Integration state kept for a lsnn module

    Parameters:
        v (torch.Tensor): membrane potential
        i (torch.Tensor): synaptic input current
        b (torch.Tensor): threshold adaptation
    """
    v: torch.Tensor
    i: torch.Tensor
    b: torch.Tensor


def lsnn_feed_forward_step(input_tensor: torch.Tensor, state:
    LSNNFeedForwardState, p: LSNNParameters=LSNNParameters(), dt: float=0.001
    ) ->Tuple[torch.Tensor, LSNNFeedForwardState]:
    """Euler integration step for LIF Neuron with threshold adaptation.
    More specifically it implements one integration step of the following ODE

    .. math::
        \\\\begin{align*}
            \\dot{v} &= 1/\\tau_{\\text{mem}} (v_{\\text{leak}} - v + i) \\\\
            \\dot{i} &= -1/\\tau_{\\text{syn}} i \\\\
            \\dot{b} &= -1/\\tau_{b} b
        \\end{align*}

    together with the jump condition

    .. math::
        z = \\Theta(v - v_{\\text{th}} + b)

    and transition equations

    .. math::
        \\begin{align*}
            v &= (1-z) v + z v_{\\text{reset}} \\\\
            i &= i + \\text{input} \\\\
            b &= b + \\beta z
        \\end{align*}

    Parameters:
        input_tensor (torch.Tensor): the input spikes at the current time step
        s (LSNNFeedForwardState): current state of the lsnn unit
        p (LSNNParameters): parameters of the lsnn unit
        dt (float): Integration timestep to use
    """
    dv = dt * p.tau_mem_inv * (p.v_leak - state.v + state.i)
    v_decayed = state.v + dv
    di = -dt * p.tau_syn_inv * state.i
    i_decayed = state.i + di
    db = dt * p.tau_adapt_inv * (p.v_th - state.b)
    b_decayed = state.b + db
    z_new = threshold(v_decayed - b_decayed, p.method, p.alpha)
    v_new = (1 - z_new) * v_decayed + z_new * p.v_reset
    b_new = (1 - z_new) * b_decayed + z_new * state.b
    i_new = i_decayed + input_tensor
    return z_new, LSNNFeedForwardState(v=v_new, i=i_new, b=b_new)


class LSNNFeedForwardCell(torch.nn.Module):
    """Euler integration cell for LIF Neuron with threshold adaptation.
    More specifically it implements one integration step of the following ODE

    .. math::
        \\begin{align*}
            \\dot{v} &= 1/\\tau_{\\text{mem}} (v_{\\text{leak}} - v + i) \\\\
            \\dot{i} &= -1/\\tau_{\\\\text{syn}} i \\\\
            \\dot{b} &= -1/\\tau_{b} b
        \\end{align*}

    together with the jump condition

    .. math::
        z = \\Theta(v - v_{\\text{th}} + b)

    and transition equations

    .. math::
        \\begin{align*}
            v &= (1-z) v + z v_{\\text{reset}} \\\\
            i &= i + \\text{input} \\\\
            b &= b + \\beta z
        \\end{align*}

    Parameters:
        input (torch.Tensor): the input spikes at the current time step
        s (LSNNFeedForwardState): current state of the lsnn unit
        p (LSNNParameters): parameters of the lsnn unit
        dt (float): Integration timestep to use
    """

    def __init__(self, shape, p: LSNNParameters=LSNNParameters(), dt: float
        =0.001):
        super(LSNNFeedForwardCell, self).__init__()
        self.shape = shape
        self.p = p
        self.dt = dt

    def initial_state(self, batch_size, device, dtype=torch.float
        ) ->LSNNFeedForwardState:
        """return the initial state of an LSNN neuron"""
        return LSNNFeedForwardState(v=torch.zeros(batch_size, *self.shape,
            device=device, dtype=dtype), i=torch.zeros(batch_size, *self.
            shape, device=device, dtype=dtype), b=torch.zeros(batch_size, *
            self.shape, device=device, dtype=dtype))

    def forward(self, input_tensor: torch.Tensor, state: LSNNFeedForwardState
        ) ->Tuple[torch.Tensor, LSNNFeedForwardState]:
        return lsnn_feed_forward_step(input_tensor, state, p=self.p, dt=self.dt
            )


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_norse_norse(_paritybench_base):
    pass
    def test_000(self):
        self._check(ANNPolicy(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(ConstantCurrentLIFEncoder(*[], **{'seq_length': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_002(self):
        self._check(Policy(*[], **{}), [torch.rand([4, 4])], {})

