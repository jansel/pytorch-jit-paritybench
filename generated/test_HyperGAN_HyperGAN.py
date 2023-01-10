import sys
_module = sys.modules[__name__]
del sys
generate = _module
chargan = _module
classification = _module
colorizer = _module
common = _module
alignment = _module
autoencode = _module
static = _module
sequential_mnist = _module
hypergan = _module
backend = _module
cpu_backend = _module
hogwild_backend = _module
multi_gpu_backend = _module
roundrobin_backend = _module
single_gpu_backend = _module
tpu_backend = _module
cli = _module
configurable_component = _module
configuration = _module
configurations = _module
generate_readme = _module
generate_samples = _module
replace_hyperparms = _module
run_all = _module
discriminators = _module
base_discriminator = _module
configurable_discriminator = _module
dcgan_discriminator = _module
distributions = _module
base_distribution = _module
fitness_distribution = _module
normal_distribution = _module
optimize_distribution = _module
truncated_normal_distribution = _module
uniform_distribution = _module
gan = _module
gan_component = _module
gans = _module
aligned_gan = _module
aligned_interpolate_gan = _module
base_gan = _module
ali_gan_combined = _module
ali_style_gan = _module
ali_vib_gan = _module
alialpha_gan = _module
aligned_ali_gan = _module
aligned_ali_gan3 = _module
aligned_ali_gan6 = _module
aligned_ali_gan7 = _module
aligned_ali_gan8 = _module
aligned_ali_gan_test = _module
aligned_ali_one_gan = _module
alpha_gan = _module
autoencoder_gan = _module
conditional_gan = _module
distribution_filtering_gan = _module
multi_generator_gan = _module
standard_gan = _module
generators = _module
base_generator = _module
configurable_generator = _module
dcgan_generator = _module
inputs = _module
crop_resize_transform = _module
fitness_image_loader = _module
image_loader = _module
unsupervised_image_folder = _module
layer = _module
layer_shape = _module
layers = _module
add = _module
cat = _module
channel_attention = _module
efficient_attention = _module
ez_norm = _module
layer = _module
mul = _module
multi_head_attention = _module
noise = _module
operation = _module
pixel_shuffle = _module
residual = _module
resizable_stack = _module
segment_softmax = _module
upsample = _module
losses = _module
ali_loss = _module
base_loss = _module
f_divergence_loss = _module
least_squares_loss = _module
logistic_loss = _module
qp_loss = _module
ragan_loss = _module
realness_loss = _module
softmax_loss = _module
standard_loss = _module
wasserstein_loss = _module
adaptive_instance_norm = _module
attention = _module
concat_noise = _module
const = _module
learned_noise = _module
modulated_conv2d = _module
multi_head_attention = _module
no_op = _module
pixel_norm = _module
reshape = _module
residual = _module
scaled_conv2d = _module
variational = _module
adamirror = _module
amsgrad = _module
competitive2_optimizer = _module
competitive_optimizer = _module
consensus_optimizer = _module
depth_optimizer = _module
elastic_weight_consolidation_optimizer = _module
ema_optimizer = _module
gan_optimizer = _module
gradient_magnitude_optimizer = _module
jr_optimizer = _module
local_nash_optimizer = _module
negative_momentum_optimizer = _module
orthonormal_optimizer = _module
potential_optimizer = _module
predictive_method_optimizer = _module
sga_optimizer = _module
social_optimizer = _module
sos_optimizer = _module
tpu_negative_momentum_optimizer = _module
parser = _module
process_manager = _module
pygame_viewer = _module
samplers = _module
aligned_sampler = _module
base_sampler = _module
batch_sampler = _module
batch_walk_sampler = _module
factorization_batch_walk_sampler = _module
grid_sampler = _module
input_sampler = _module
debug_sampler = _module
gang_sampler = _module
progressive_sampler = _module
random_walk_sampler = _module
sorted_sampler = _module
style_walk_sampler = _module
static_batch_sampler = _module
y_sampler = _module
search = _module
aligned_random_search = _module
alphagan_random_search = _module
default_configurations = _module
random_search = _module
threaded_tk_viewer = _module
tk_viewer = _module
train_hook_collection = _module
adversarial_norm_train_hook = _module
base_train_hook = _module
conjecture_train_hook = _module
differential_augmentation_train_hook = _module
rolling_memory_2_train_hook = _module
rolling_memory_train_hook = _module
extragradient_train_hook = _module
gradient_penalty_train_hook = _module
initialize_as_autoencoder = _module
match_support_train_hook = _module
adversarial_robust_train_hook = _module
competitive_train_hook = _module
force_equilibrium_train_hook = _module
gradient_locally_stable_train_hook = _module
imle_train_hook = _module
k_lipschitz_train_hook = _module
learning_rate_dropout_train_hook = _module
max_gp_train_hook = _module
minibatch_train_hook = _module
progress_compress_kbgan_train_hook = _module
progress_compress_train_hook = _module
self_supervised_train_hook = _module
weight_constraint_train_hook = _module
weight_penalty_train_hook = _module
zero_penalty_train_hook = _module
negative_momentum_train_hook = _module
online_ewc_train_hook = _module
stabilizing_training_train_hook = _module
trainable_gan = _module
trainers = _module
accumulate_gradient_trainer = _module
alternating_trainer = _module
balanced_trainer = _module
base_trainer = _module
competitive_alternating_trainer = _module
competitive_trainer = _module
consensus_trainer = _module
curriculum_trainer = _module
depth_trainer = _module
evolution_trainer = _module
gang_trainer = _module
incremental_trainer = _module
kbeam_trainer = _module
line_search_trainer = _module
multi_marginal_trainer = _module
multi_step_trainer = _module
multi_trainer_trainer = _module
proportional_control_trainer = _module
qualified_step_trainer = _module
simultaneous_trainer = _module
viewer = _module
setup = _module
conftest = _module
test_uniform_distribution = _module
test_base_gan = _module
test_standard_gan = _module
test_image_loader = _module
test_boundary_equilibrium_loss = _module
test_category_loss = _module
test_cramer_loss = _module
test_lamb_gan_loss = _module
test_least_squares_loss = _module
test_softmax_loss = _module
test_standard_gan_loss = _module
test_supervised_loss = _module
test_wasserstein_loss = _module
mocks = _module
test_websocket_server = _module
test_configuration = _module
test_gan = _module
test_gan_component = _module
test_parser = _module

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


import math


import torch


import uuid


import copy


import re


import string


import time


import torch.utils.data as data


from torch import optim


from torch.autograd import Variable


from torchvision import datasets


from torchvision import transforms


import random


import torch.nn as nn


from torch.autograd import grad as torch_grad


import torch.nn.functional as F


import torchvision


import torch.multiprocessing as mp


import inspect


from functools import reduce


from torch.distributions import uniform


import itertools


import types


from collections.abc import Sequence


from collections.abc import Iterable


import warnings


from torchvision.transforms import functional as F


from torch import nn


from torch.nn import functional as f


from torch.autograd import Function


from torch.utils.cpp_extension import load


from torch.optim import Optimizer


from torch.optim import Adam


from torch.nn.parameter import Parameter


from tensorflow.python.framework import ops


from tensorflow.python.ops import control_flow_ops


from tensorflow.python.ops import math_ops


from tensorflow.python.ops import state_ops


from tensorflow.python.training import optimizer


from copy import deepcopy


class ValidationException(Exception):
    """
    GAN components validate their configurations before creation.  
    
    `ValidationException` occcurs if they fail.
    """
    pass


class GANComponent(nn.Module):
    """
    GANComponents are pluggable pieces within a GAN.

    GAN objects are also GANComponents.
    """

    def __init__(self, gan, config):
        """
        Initializes a gan component based on a `gan` and a `config` dictionary.

        Different components require different config variables.  

        A `ValidationException` is raised if the GAN component configuration fails to validate.
        """
        super(GANComponent, self).__init__()
        self.gan = gan
        self.config = hc.Config(config)
        errors = self.validate()
        if errors != []:
            raise ValidationException(self.__class__.__name__ + ': ' + '\n'.join(errors))
        self.create()

    def create(self, *args):
        raise ValidationException('GANComponent.create() called directly.  Please override.')

    def required(self):
        """
        Return a list of required config strings and a `ValidationException` will be thrown if any are missing.

        Example: 
        ```python
            class MyComponent(GANComponent):
                def required(self):
                    "learn rate is required"
                    ["learn_rate"]
        ```
        """
        return []

    def validate(self):
        """
        Validates a GANComponent.  Return an array of error messages. Empty array `[]` means success.
        """
        errors = []
        required = self.required()
        for argument in required:
            if self.config.__getattr__(argument) == None:
                errors.append('`' + argument + '` required')
        if self.gan is None:
            errors.append('GANComponent constructed without GAN')
        return errors

    def add_metric(self, name, value):
        """adds metric to monitor during training
            name:string
            value:Tensor
        """
        return self.gan.add_metric(name, value)

    def metrics(self):
        """returns a metric : tensor hash"""
        return self.gan.metrics()

    def layer_regularizer(self, net):
        symbol = self.config.layer_regularizer
        op = self.lookup_function(symbol)
        if op and isinstance(op, types.FunctionType):
            net = op(self, net)
        return net

    def lookup_function(self, name):
        namespaced_method = name.split(':')[1]
        method = namespaced_method.split('.')[-1]
        namespace = '.'.join(namespaced_method.split('.')[0:-1])
        return getattr(importlib.import_module(namespace), method)

    def lookup_class(self, name):
        return self.lookup_function(name)

    def set_trainable(self, flag):
        for p in self.parameters():
            p.requires_grad = flag

    def latent_parameters(self):
        return []


class LayerShape:

    def __init__(self, *dims):
        self.dims = dims
        if len(dims) == 1:
            self.channels = dims[0]
        elif len(dims) == 2:
            self.channels = dims[0]
            self.height = dims[1]
        elif len(dims) == 3:
            self.channels = dims[0]
            self.height = dims[1]
            self.width = dims[2]
        elif len(dims) == 4:
            self.frames = dims[0]
            self.channels = dims[1]
            self.height = dims[2]
            self.width = dims[3]

    def squeeze_dims(self):
        filter(lambda x: x == 1, self.dims)

    def size(self):
        if len(self.dims) == 1:
            return self.channels
        if len(self.dims) == 2:
            return self.channels * self.height
        if len(self.dims) == 3:
            return self.channels * self.height * self.width
        return self.channels * self.height * self.width * self.frames

    def __repr__(self):
        return 'LayerShape(' + ', '.join([str(x) for x in self.dims]) + ')'

    def __str__(self):
        return self.__repr__()


class MultiHeadAttention(nn.Module):

    def __init__(self, input_size, output_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.heads = heads
        self.features = input_size // heads
        self.features_sqrt = math.sqrt(self.features)
        self.f = nn.Linear(input_size, input_size)
        self.g = nn.Linear(input_size, input_size)
        self.h = nn.Linear(input_size, input_size)
        self.o = nn.Linear(input_size, output_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, input_size = x.shape
        f = self.f(x).view(batch_size, 1, self.heads, self.features).permute(0, 2, 3, 1)
        g = self.g(x).view(batch_size, 1, self.heads, self.features).permute(0, 2, 1, 3)
        fg = torch.matmul(g, f) / self.features_sqrt
        attention_map = self.softmax(fg)
        h = self.h(x).view(batch_size, 1, self.heads, self.features).permute(0, 2, 1, 3)
        fgh = torch.matmul(attention_map, h)
        output = fgh.permute(0, 2, 1, 3).view(x.shape)
        return self.o(output)


class LearnedNoise(nn.Module):

    def __init__(self, batch_size, c, h, w, mul=0.1):
        super().__init__()
        self.batch_size = batch_size
        self.h = h
        self.w = w
        self.c = c
        self.weight = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.noise = torch.Tensor(self.batch_size, 1, self.h, self.w)

    def forward(self, input):
        self.noise.normal_()
        return input + self.noise * self.weight


class Residual(nn.Module):

    def __init__(self, channels, filter=3, stride=1, padding=1, activation=nn.ReLU):
        super(Residual, self).__init__()
        self.conv = nn.Conv2d(channels, channels, filter, stride, padding)
        self.activation = activation()

    def forward(self, x):
        return x + self.activation(self.conv(x))


class BaseLoss(GANComponent):

    def __init__(self, gan, config):
        super(BaseLoss, self).__init__(gan, config)
        self.relu = torch.nn.ReLU()

    def create(self, *args):
        pass

    def required(self):
        return ''.split()

    def forward(self, d_real, d_fake):
        d_loss, g_loss = [c.mean() for c in self._forward(d_real, d_fake)]
        return d_loss, g_loss


TINY = 1e-12


class FDivergenceLoss(BaseLoss):

    def __init__(self, gan, config):
        super(FDivergenceLoss, self).__init__(gan, config)
        self.tanh = torch.nn.Tanh()

    def _forward(self, d_real, d_fake):
        gan = self.gan
        config = self.config
        gfx = None
        gfg = None
        pi = config.pi or 2
        g_loss_type = config.g_loss_type or config.type or 'gan'
        d_loss_type = config.type or 'gan'
        alpha = config.alpha or 0.5
        if d_loss_type == 'kl':
            bounded_x = torch.clamp(d_real, max=np.exp(9.0))
            bounded_g = torch.clamp(d_fake, max=10.0)
            gfx = bounded_x
            gfg = bounded_g
        elif d_loss_type == 'js':
            gfx = np.log(2) - (1 + (-d_real).exp()).log()
            gfg = np.log(2) - (1 + (-d_fake).exp()).log()
        elif d_loss_type == 'js_weighted':
            gfx = -pi * np.log(pi) - (1 + (-d_real).exp()).log()
            gfg = -pi * np.log(pi) - (1 + (-d_fake).exp()).log()
        elif d_loss_type == 'gan':
            gfx = -(1 + (-d_real).exp()).log()
            gfg = -(1 + (-d_fake).exp()).log()
        elif d_loss_type == 'reverse_kl':
            gfx = -d_real.log()
            gfg = -d_fake.log()
        elif d_loss_type == 'pearson' or d_loss_type == 'jeffrey' or d_loss_type == 'alpha2':
            gfx = d_real
            gfg = d_fake
        elif d_loss_type == 'squared_hellinger':
            gfx = 1 - (-d_real).exp()
            gfg = 1 - (-d_fake).exp()
        elif d_loss_type == 'neyman':
            gfx = 1 - (-d_real).exp()
            gfx = torch.clamp(gfx, max=1.9)
            gfg = 1 - (-d_fake).exp()
        elif d_loss_type == 'total_variation':
            gfx = 0.5 * self.tanh(d_real)
            gfg = 0.5 * self.tanh(d_fake)
        elif d_loss_type == 'alpha1':
            gfx = 1.0 / (1 - alpha) - (1 + (-d_real).exp()).log()
            gfg = 1.0 / (1 - alpha) - (1 + (-d_fake).exp()).log()
        else:
            raise ('Unknown type ' + d_loss_type)
        conjugate = None
        if d_loss_type == 'kl':
            conjugate = (gfg - 1).exp()
        elif d_loss_type == 'js':
            bounded = torch.clamp(gfg, max=np.log(2.0) - TINY)
            conjugate = -(2 - bounded.exp()).log()
        elif d_loss_type == 'js_weighted':
            c = -pi * np.log(pi) - TINY
            bounded = gfg
            conjugate = (1 - pi) * ((1 - pi) / ((1 - pi) * (bounded / pi).exp())).log()
        elif d_loss_type == 'gan':
            conjugate = -(1 - gfg.exp()).log()
        elif d_loss_type == 'reverse_kl':
            conjugate = -1 - (-gfg).log()
        elif d_loss_type == 'pearson':
            conjugate = 0.25 * gfg ** 2 + gfg
        elif d_loss_type == 'neyman':
            conjugate = 2 - 2 * torch.sqrt(self.relu(1 - gfg) + 0.01)
        elif d_loss_type == 'squared_hellinger':
            conjugate = gfg / (1.0 - gfg)
        elif d_loss_type == 'jeffrey':
            raise 'jeffrey conjugate not implemented'
        elif d_loss_type == 'alpha2' or d_loss_type == 'alpha1':
            bounded = gfg
            bounded = 1.0 / alpha * (bounded * (alpha - 1) + 1)
            conjugate = bounded ** (alpha / (alpha - 1.0)) - 1.0 / alpha
        elif d_loss_type == 'total_variation':
            conjugate = gfg
        else:
            raise ('Unknown type ' + d_loss_type)
        gf_threshold = None
        if d_loss_type == 'kl':
            gf_threshold = 1
        elif d_loss_type == 'js':
            gf_threshold = 0
        elif d_loss_type == 'gan':
            gf_threshold = -np.log(2)
        elif d_loss_type == 'reverse_kl':
            gf_threshold = -1
        elif d_loss_type == 'pearson':
            gf_threshold = 0
        elif d_loss_type == 'squared_hellinger':
            gf_threshold = 0
        self.gf_threshold = gf_threshold
        d_loss = -gfx + conjugate
        g_loss = -conjugate
        if g_loss_type == 'gan':
            g_loss = -conjugate
        elif g_loss_type == 'total_variation':
            g_loss = -conjugate
        elif g_loss_type == 'js':
            g_loss = -d_fake.exp()
        elif g_loss_type == 'js_weighted':
            p = pi
            u = d_fake
            exp_bounded = p / u
            exp_bounded = torch.clamp(exp_bounded, max=4.0)
            inner = (-4.0 * u * exp_bounded.exp() + np.exp(2.0) * u ** 2 - 2.0 * np.exp(2.0) * u + np.exp(2.0)) / u ** 2
            inner = self.relu(inner)
            u = torch.clamp(u, max=0.1)
            sqrt = torch.sqrt(inner + 0.01) / (2 * np.exp(1))
            g_loss = (1.0 - u) / (2.0 * u)
        elif g_loss_type == 'pearson':
            g_loss = -(d_fake - 2.0) / 2.0
        elif g_loss_type == 'neyman':
            g_loss = 1.0 / torch.sqrt(1 - d_fake)
        elif g_loss_type == 'squared_hellinger':
            g_loss = -1.0 / ((d_fake - 1) ** 2 + 0.01)
        elif g_loss_type == 'reverse_kl':
            g_loss = -d_fake
        elif g_loss_type == 'kl':
            g_loss = -gfg * gfg.exp()
        elif g_loss_type == 'alpha1':
            a = alpha
            bounded = d_fake
            g_loss = 1.0 / (a * (a - 1)) * ((a * bounded).exp() - 1 - a * (bounded.exp() - 1))
        elif g_loss_type == 'alpha2':
            a = alpha
            bounded = torch.clamp(d_fake, max=4.0)
            g_loss = -(1.0 / (a * (a - 1))) * ((a * bounded).exp() - 1 - a * (bounded.exp() - 1))
        else:
            raise ('Unknown g_loss_type ' + g_loss_type)
        if self.config.regularizer:
            g_loss += self.g_regularizer(gfg, gfx)
        return [d_loss, g_loss]

    def g_regularizer(self, gfg, gfx):
        regularizer = None
        config = self.config
        pi = config.pi or 2
        alpha = config.alpha or 0.5
        ddfc = 0
        if config.regularizer == 'kl':
            bounded = torch.clamp(gfg, max=4.0)
            ddfc = (bounded - 1).exp()
        elif config.regularizer == 'js':
            ddfc = -(2 * gfg.exp()) / ((2 - gfg.exp()) ** 2 + 0.01)
        elif config.regularizer == 'js_weighted':
            ddfc = -((pi - 1) * (gfg / pi).exp()) / (pi * (pi * (gfg / pi).exp() - 1) ** 2)
        elif config.regularizer == 'gan':
            ddfc = 2 * gfg.exp() / ((1 - gfg.exp()) ** 2 + 0.01)
        elif config.regularizer == 'reverse_kl':
            ddfc = 1.0 / gfg ** 2
        elif config.regularizer == 'pearson':
            ddfc = 0.5
        elif config.regularizer == 'jeffrey':
            raise 'jeffrey regularizer not implemented'
        elif config.regularizer == 'squared_hellinger':
            ddfc = 2 / ((gfg - 1) ** 3 + 0.01)
        elif config.regularizer == 'neyman':
            ddfc = 1.0 / (2 * (1 - gfg ** 3 / 2.0))
        elif config.regularizer == 'total_variation':
            ddfc = 0
        elif config.regularizer == 'alpha1' or config.regularizer == 'alpha2':
            ddfc = -((alpha - 1) * gfg + 1) ** (1 / (alpha - 1) - 1)
        regularizer = ddfc * torch.norm(gfg, 2, 0) * (config.regularizer_lambda or 1)
        self.add_metric('fgan_regularizer', regularizer.mean())
        return regularizer

    def d_regularizers(self):
        return []


class LogisticLoss(BaseLoss):
    """ported from stylegan"""

    def __init__(self, gan, config):
        super(LogisticLoss, self).__init__(gan, config)
        self.softplus = torch.nn.Softplus(self.config.beta or 1, self.config.threshold or 20)

    def _forward(self, d_real, d_fake):
        d_loss = self.softplus(-d_real) + self.softplus(d_fake)
        g_loss = self.softplus(-d_fake)
        return [d_loss, g_loss]


class RaganLoss(BaseLoss):
    """https://arxiv.org/abs/1807.00734"""

    def __init__(self, gan, config):
        super(RaganLoss, self).__init__(gan, config)
        self.sigmoid = torch.nn.Sigmoid()

    def required(self):
        return ''.split()

    def _forward(self, d_real, d_fake):
        config = self.config
        gan = self.gan
        loss_type = self.config.type or 'standard'
        if config.rgan:
            cr = d_real
            cf = d_real
        else:
            cr = torch.mean(d_real, 0)
            cf = torch.mean(d_fake, 0)
        if loss_type == 'least_squares':
            a, b, c = config.labels or [-1, 1, 1]
            d_loss = 0.5 * (d_real - cf - b) ** 2 + 0.5 * (d_fake - cr - a) ** 2
            g_loss = 0.5 * (d_fake - cr - c) ** 2 + 0.5 * (d_real - cf - a) ** 2
        elif loss_type == 'hinge':
            d_loss = torch.clamp(1 - (d_real - cf), min=0) + torch.clamp(1 + (d_fake - cr), min=0)
            g_loss = torch.clamp(1 - (d_fake - cr), min=0) + torch.clamp(1 + (d_real - cf), min=0)
        elif loss_type == 'wasserstein':
            d_loss = -(d_real - cf) + (d_fake - cr)
            g_loss = -(d_fake - cr)
        elif loss_type == 'standard':
            criterion = torch.nn.BCEWithLogitsLoss()
            g_loss = criterion(d_fake - cr, torch.ones_like(d_fake))
            d_loss = criterion(d_real - cf, torch.ones_like(d_real)) + criterion(d_fake - cr, torch.zeros_like(d_fake))
        return [d_loss, g_loss]


class CategoricalLoss(nn.Module):

    def __init__(self, atoms=51, v_max=1.0, v_min=-1.0):
        super(CategoricalLoss, self).__init__()
        self.atoms = atoms
        self.v_max = v_max
        self.v_min = v_min
        self.supports = torch.linspace(v_min, v_max, atoms).view(1, 1, atoms)
        self.delta = (v_max - v_min) / (atoms - 1)

    def forward(self, anchor, feature, skewness=0.0):
        batch_size = feature.shape[0]
        skew = torch.zeros((batch_size, self.atoms)).fill_(skewness)
        Tz = skew + self.supports.view(1, -1) * torch.ones((batch_size, 1)).to(torch.float).view(-1, 1)
        Tz = Tz.clamp(self.v_min, self.v_max)
        b = (Tz - self.v_min) / self.delta
        l = b.floor()
        u = b.ceil()
        l[(u > 0) * (l == u)] -= 1
        u[(l < self.atoms - 1) * (l == u)] += 1
        offset = torch.linspace(0, (batch_size - 1) * self.atoms, batch_size).to(torch.int64).unsqueeze(dim=1).expand(batch_size, self.atoms)
        skewed_anchor = torch.zeros(batch_size, self.atoms)
        skewed_anchor.view(-1).index_add_(0, (l + offset).view(-1), (anchor * (u.float() - b)).view(-1))
        skewed_anchor.view(-1).index_add_(0, (u + offset).view(-1), (anchor * (b - l.float())).view(-1))
        loss = -(skewed_anchor * (feature + 1e-16).log()).sum(-1).mean()
        return loss


class RealnessLoss(BaseLoss):
    """https://arxiv.org/pdf/2002.05512v1.pdf"""

    def __init__(self, gan, config):
        super(RealnessLoss, self).__init__(gan, config)

    def required(self):
        return 'skew'.split()

    def _forward(self, d_real, d_fake):
        num_outcomes = d_real.shape[1]
        if not hasattr(self, 'anchor0'):
            gauss = np.random.normal(0, 0.3, 1000)
            count, bins = np.histogram(gauss, num_outcomes)
            self.anchor0 = count / num_outcomes
            unif = np.random.uniform(-1, 1, 1000)
            count, bins = np.histogram(unif, num_outcomes)
            self.anchor1 = count / num_outcomes
            self.anchor_real = torch.zeros((self.gan.batch_size(), num_outcomes), dtype=torch.float) + torch.tensor(self.anchor1, dtype=torch.float)
            self.anchor_fake = torch.zeros((self.gan.batch_size(), num_outcomes), dtype=torch.float) + torch.tensor(self.anchor0, dtype=torch.float)
            self.Triplet_Loss = CategoricalLoss(num_outcomes)
        feat_real = d_real.log_softmax(1).exp()
        feat_fake = d_fake.log_softmax(1).exp()
        d_loss = self.Triplet_Loss(self.anchor_real, feat_real, skewness=self.config.skew[1]) + self.Triplet_Loss(self.anchor_fake, feat_fake, skewness=self.config.skew[0])
        g_loss = -self.Triplet_Loss(self.anchor_fake, feat_fake, skewness=self.config.skew[0])
        g_loss += self.Triplet_Loss(self.anchor_real, feat_fake, skewness=self.config.skew[0])
        return [d_loss, g_loss]


class StandardLoss(BaseLoss):

    def __init__(self, gan, config):
        super(StandardLoss, self).__init__(gan, config)
        self.relu = torch.nn.ReLU()
        self.two = torch.Tensor([2.0])
        self.eps = torch.Tensor([1e-12])

    def _forward(self, d_real, d_fake):
        criterion = torch.nn.BCEWithLogitsLoss()
        g_loss = criterion(d_fake, torch.ones_like(d_fake))
        d_loss = criterion(d_real, torch.ones_like(d_real)) + criterion(d_fake, torch.zeros_like(d_fake))
        return [d_loss, g_loss]


class WassersteinLoss(BaseLoss):

    def _forward(self, d_real, d_fake):
        config = self.config
        if config.kl:
            d_fake_norm = torch.mean(d_fake.exp()) + 1e-08
            d_fake_ratio = (d_fake.exp() + 1e-08) / d_fake_norm
            d_fake = d_fake * d_fake_ratio
        d_loss = -d_real + d_fake
        g_loss = -d_fake
        return [d_loss, g_loss]


class EqualLinear(nn.Module):

    def __init__(self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))
        else:
            self.bias = None
        self.activation = activation
        self.scale = 1 / math.sqrt(in_dim) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)
        else:
            out = F.linear(input, self.weight * self.scale, bias=self.bias * self.lr_mul)
        return out


class AdaptiveInstanceNorm(nn.Module):

    def __init__(self, style_size, content_size, equal_linear=False):
        super(AdaptiveInstanceNorm, self).__init__()
        if equal_linear:
            self.gamma = EqualLinear(style_size, content_size)
            self.beta = EqualLinear(style_size, content_size)
        else:
            self.gamma = nn.Linear(style_size, content_size)
            self.beta = nn.Linear(style_size, content_size)

    def calc_mean_std(self, feat, eps=1e-05):
        size = feat.size()
        assert len(size) == 4
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std

    def calc_mean_std1d(self, feat, eps=1e-05):
        size = feat.size()
        assert len(size) == 3
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1)
        return feat_mean, feat_std

    def forward(self, content, style, epsilon=1e-05):
        style = style.view(content.shape[0], -1)
        gamma = self.gamma(style)
        beta = self.beta(style)
        if len(content.shape) == 4:
            c_mean, c_var = self.calc_mean_std(content, epsilon)
        elif len(content.shape) == 3:
            c_mean, c_var = self.calc_mean_std1d(content, epsilon)
        c_std = (c_var + epsilon).sqrt()
        return (1 + gamma.view(c_std.shape)) * ((content - c_mean) / c_std) + beta.view(c_std.shape)


class Attention(nn.Module):
    """ Self attention Layer from https://github.com/heykeetae/Self-Attention-GAN/blob/master/sagan_models.py """

    def __init__(self, in_dim):
        super(Attention, self).__init__()
        self.chanel_in = in_dim
        self.f = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.g = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.h = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.v = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.tensor(0.0))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        m_batchsize, C, width, height = x.size()
        f = self.f(x).view(m_batchsize, C, width * height).permute(0, 2, 1)
        g = self.g(x).view(m_batchsize, C, width * height)
        fg = torch.bmm(f, g)
        attention_map = self.softmax(fg)
        h = self.h(x).view(m_batchsize, C, width * height)
        fgh = torch.bmm(h, attention_map)
        return self.v(fgh.view(x.shape))


class ConcatNoise(nn.Module):

    def __init__(self):
        super(ConcatNoise, self).__init__()
        self.z = uniform.Uniform(torch.Tensor([-1.0]), torch.Tensor([1.0]))

    def forward(self, x):
        noise = self.z.sample(x.shape)
        cat = torch.cat([x, noise.view(*x.shape)], 1)
        return cat


class Const(nn.Module):

    def __init__(self, c, h, w, mul=0.1):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, c, h, w) * mul)

    def forward(self, _input):
        return self.weight


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)
    if k.ndim == 1:
        k = k[None, :] * k[:, None]
    k /= k.sum()
    return k


def upfirdn2d_op():
    global _upfirdn2d_op
    if _upfirdn2d_op is None:
        _upfirdn2d_op = load('upfirdn2d', sources=[os.path.join(module_path, 'upfirdn2d.cpp'), os.path.join(module_path, 'upfirdn2d_kernel.cu')])
    return _upfirdn2d_op


class UpFirDn2dBackward(Function):

    @staticmethod
    def forward(ctx, grad_output, kernel, grad_kernel, up, down, pad, g_pad, in_size, out_size):
        up_x, up_y = up
        down_x, down_y = down
        g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1 = g_pad
        grad_output = grad_output.reshape(-1, out_size[0], out_size[1], 1)
        grad_input = upfirdn2d_op().upfirdn2d(grad_output, grad_kernel, down_x, down_y, up_x, up_y, g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1)
        grad_input = grad_input.view(in_size[0], in_size[1], in_size[2], in_size[3])
        ctx.save_for_backward(kernel)
        pad_x0, pad_x1, pad_y0, pad_y1 = pad
        ctx.up_x = up_x
        ctx.up_y = up_y
        ctx.down_x = down_x
        ctx.down_y = down_y
        ctx.pad_x0 = pad_x0
        ctx.pad_x1 = pad_x1
        ctx.pad_y0 = pad_y0
        ctx.pad_y1 = pad_y1
        ctx.in_size = in_size
        ctx.out_size = out_size
        return grad_input

    @staticmethod
    def backward(ctx, gradgrad_input):
        kernel, = ctx.saved_tensors
        gradgrad_input = gradgrad_input.reshape(-1, ctx.in_size[2], ctx.in_size[3], 1)
        gradgrad_out = upfirdn2d_op().upfirdn2d(gradgrad_input, kernel, ctx.up_x, ctx.up_y, ctx.down_x, ctx.down_y, ctx.pad_x0, ctx.pad_x1, ctx.pad_y0, ctx.pad_y1)
        gradgrad_out = gradgrad_out.view(ctx.in_size[0], ctx.in_size[1], ctx.out_size[0], ctx.out_size[1])
        return gradgrad_out, None, None, None, None, None, None, None, None


class UpFirDn2d(Function):

    @staticmethod
    def forward(ctx, input, kernel, up, down, pad):
        up_x, up_y = up
        down_x, down_y = down
        pad_x0, pad_x1, pad_y0, pad_y1 = pad
        kernel_h, kernel_w = kernel.shape
        batch, channel, in_h, in_w = input.shape
        ctx.in_size = input.shape
        input = input.reshape(-1, in_h, in_w, 1)
        ctx.save_for_backward(kernel, torch.flip(kernel, [0, 1]))
        out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
        out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1
        ctx.out_size = out_h, out_w
        ctx.up = up_x, up_y
        ctx.down = down_x, down_y
        ctx.pad = pad_x0, pad_x1, pad_y0, pad_y1
        g_pad_x0 = kernel_w - pad_x0 - 1
        g_pad_y0 = kernel_h - pad_y0 - 1
        g_pad_x1 = in_w * up_x - out_w * down_x + pad_x0 - up_x + 1
        g_pad_y1 = in_h * up_y - out_h * down_y + pad_y0 - up_y + 1
        ctx.g_pad = g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1
        out = upfirdn2d_op().upfirdn2d(input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1)
        out = out.view(-1, channel, out_h, out_w)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        kernel, grad_kernel = ctx.saved_tensors
        grad_input = UpFirDn2dBackward.apply(grad_output, kernel, grad_kernel, ctx.up, ctx.down, ctx.pad, ctx.g_pad, ctx.in_size, ctx.out_size)
        return grad_input, None, None, None, None


def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    out = UpFirDn2d.apply(input, kernel, (up, up), (down, down), (pad[0], pad[1], pad[0], pad[1]))
    return out


class Blur(nn.Module):

    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()
        kernel = make_kernel(kernel)
        if upsample_factor > 1:
            kernel = kernel * upsample_factor ** 2
        self.register_buffer('kernel', kernel)
        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)
        return out


class ModulatedConv2d(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, style_dim, demodulate=True, upsample=True, downsample=False, lr_mul=1.0, blur_kernel=[1, 3, 3, 1]):
        super(ModulatedConv2d, self).__init__()
        self.eps = 1e-08
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample
        if upsample:
            factor = 2
            p = len(blur_kernel) - factor - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1
            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)
        if downsample:
            factor = 2
            p = len(blur_kernel) - factor + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2
            self.blur = Blur(blur_kernel, pad=(pad0, pad1))
        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2
        self.mod_weight = nn.Parameter(torch.randn(1, out_channel, in_channel, kernel_size, kernel_size))
        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1, lr_mul=lr_mul)
        self.demodulate = demodulate

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape
        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.mod_weight * style
        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-08)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)
        weight = weight.view(batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size)
        if self.upsample:
            input = input.reshape(1, batch * in_channel, height, width)
            weight = weight.view(batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size)
            weight = weight.transpose(1, 2).reshape(batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size)
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)
        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
        else:
            input = input.reshape(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
        return out


class NoOp(nn.Module):

    def __init__(self):
        super(NoOp, self).__init__()

    def forward(self, x):
        return x


class PixelNorm(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-08)


class Reshape(nn.Module):

    def __init__(self, *dims):
        self.dims = dims
        super(Reshape, self).__init__()

    def forward(self, x):
        return x.view(x.size()[0], *self.dims)


class ScaledConv2d(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, style_dim, padding=0, demodulate=True, upsample=True, downsample=False, lr_mul=1.0, blur_kernel=[1, 3, 3, 1]):
        super(ScaledConv2d, self).__init__()
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = padding
        self.weight = nn.Parameter(torch.randn(out_channel, in_channel, kernel_size, kernel_size))

    def forward(self, input):
        batch, in_channel, height, width = input.shape
        weight = self.scale * self.weight
        out = F.conv2d(input, weight, padding=self.padding)
        return out


class Variational(nn.Module):

    def __init__(self, channels, filter=1, stride=1, padding=0, activation=nn.LeakyReLU):
        super(Variational, self).__init__()
        self.mu_logit = nn.Conv2d(channels, channels, filter, stride, padding, padding_mode='reflect')
        self.sigma_logit = nn.Conv2d(channels, channels, filter, stride, padding, padding_mode='reflect')

    def forward(self, x):
        sigma = self.sigma_logit(x)
        mu = self.mu_logit(x)
        z = mu + torch.exp(0.5 * sigma) * torch.randn_like(sigma)
        self.sigma = sigma.view(x.shape[0], -1)
        self.mu = mu.view(x.shape[0], -1)
        return z


class MockDiscriminator(GANComponent):

    def create(self):
        self.sample = torch.zeros([2, 1], dtype=torch.float32)
        return self.sample


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Attention,
     lambda: ([], {'in_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConcatNoise,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Const,
     lambda: ([], {'c': 4, 'h': 4, 'w': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LearnedNoise,
     lambda: ([], {'batch_size': 4, 'c': 4, 'h': 4, 'w': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MultiHeadAttention,
     lambda: ([], {'input_size': 4, 'output_size': 4, 'heads': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (NoOp,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PixelNorm,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Reshape,
     lambda: ([], {}),
     lambda: ([torch.rand([4])], {}),
     True),
    (Residual,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Variational,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_HyperGAN_HyperGAN(_paritybench_base):
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

