import sys
_module = sys.modules[__name__]
del sys
compare = _module
compare_for_tutorial = _module
compare_hyperParams = _module
compare_hyperParams_task_free = _module
compare_replay = _module
compare_task_free = _module
data = _module
available = _module
datastream = _module
labelstream = _module
load = _module
manipulate = _module
eval = _module
callbacks = _module
evaluate = _module
main = _module
main_pretrain = _module
main_task_free = _module
models = _module
cl = _module
continual_learner = _module
fromp_optimizer = _module
memory_buffer = _module
memory_buffer_stream = _module
classifier = _module
classifier_stream = _module
cond_vae = _module
conv = _module
layers = _module
nets = _module
define_models = _module
fc = _module
excitability_modules = _module
layers = _module
nets = _module
feature_extractor = _module
generative_classifier = _module
separate_classifiers = _module
utils = _module
loss_functions = _module
modules = _module
ncl = _module
vae = _module
params = _module
options = _module
param_stamp = _module
param_values = _module
train = _module
train_standard = _module
train_stream = _module
train_task_based = _module
utils = _module
visual = _module
visual_plt = _module
visual_visdom = _module

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


import itertools


import torch


from torch.utils.data import DataLoader


import random


import copy


import numpy as np


from torchvision import transforms


from torch.utils.data import ConcatDataset


from torch.utils.data import Dataset


import time


from torch import optim


import abc


from torch import nn


from torch.distributions import Categorical


from torch.nn import functional as F


import math


from torch.optim.optimizer import Optimizer


from torch.nn.utils import parameters_to_vector


from torch.nn.utils import vector_to_parameters


import torch.nn as nn


import torch.nn.functional as F


from torch.nn.parameter import Parameter


from torch.utils.data.dataloader import DataLoader


from torch.utils.data import TensorDataset


def additive_nearest_kf(B, C):
    """Here it is assumed that all these matrices are symmetric, which is NOT CHECKED explicitly"""
    BR, BL = B['A'], B['G']
    CR, CL = C['A'], C['G']
    trBL, trBR, trCL, trCR = torch.trace(BL), torch.trace(BR), torch.trace(CL), torch.trace(CR)
    if min(trBL, trBR) <= 0:
        None
        return CR, CL
    elif min(trCL, trCR) <= 0:
        None
        return BR, BL
    pi = torch.sqrt(torch.trace(BL) * torch.trace(CR)) / torch.sqrt(torch.trace(CL) * torch.trace(BR))
    return BR + CR / pi, BL + CL * pi


class UnNormalize(object):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """Denormalize image, either single image (C,H,W) or image batch (N,C,H,W)"""
        batch = len(tensor.size()) == 4
        for t, m, s in zip(tensor.permute(1, 0, 2, 3) if batch else tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


AVAILABLE_TRANSFORMS = {'MNIST': [transforms.ToTensor()], 'MNIST32': [transforms.Pad(2), transforms.ToTensor()], 'CIFAR10': [transforms.ToTensor()], 'CIFAR100': [transforms.ToTensor()], 'CIFAR10_norm': [transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.2435, 0.2616])], 'CIFAR100_norm': [transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2761])], 'CIFAR10_denorm': UnNormalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.2435, 0.2616]), 'CIFAR100_denorm': UnNormalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2761]), 'augment_from_tensor': [transforms.ToPILImage(), transforms.RandomCrop(32, padding=4, padding_mode='symmetric'), transforms.RandomHorizontalFlip(), transforms.ToTensor()], 'augment': [transforms.RandomCrop(32, padding=4, padding_mode='symmetric'), transforms.RandomHorizontalFlip()]}


def get_data_loader(dataset, batch_size, cuda=False, drop_last=False, augment=False):
    """Return <DataLoader>-object for the provided <DataSet>-object [dataset]."""
    if augment:
        dataset_ = copy.deepcopy(dataset)
        dataset_.transform = transforms.Compose([dataset.transform, *AVAILABLE_TRANSFORMS['augment']])
    else:
        dataset_ = dataset
    return DataLoader(dataset_, batch_size=batch_size, shuffle=True, drop_last=drop_last, **{'num_workers': 0, 'pin_memory': True} if cuda else {})


class ContinualLearner(nn.Module, metaclass=abc.ABCMeta):
    """Abstract module to add continual learning capabilities to a classifier (e.g., param regularization, replay)."""

    def __init__(self):
        super().__init__()
        self.param_list = [self.named_parameters]
        self.optimizer = None
        self.optim_type = 'adam'
        self.optim_list = []
        self.scenario = 'task'
        self.classes_per_context = 2
        self.singlehead = False
        self.neg_samples = 'all'
        self.replay_mode = 'none'
        self.replay_targets = 'hard'
        self.KD_temp = 2.0
        self.use_replay = 'normal'
        self.eps_agem = 0.0
        self.lwf_weighting = False
        self.mask_dict = None
        self.excit_buffer_list = []
        self.weight_penalty = False
        self.reg_strength = 0
        self.precondition = False
        self.alpha = 1e-10
        self.importance_weighting = 'fisher'
        self.fisher_kfac = False
        self.fisher_n = None
        self.fisher_labels = 'all'
        self.fisher_batch = 1
        self.context_count = 0
        self.data_size = None
        self.epsilon = 0.1
        self.offline = False
        self.gamma = 1.0

    def _device(self):
        return next(self.parameters()).device

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda

    def apply_XdGmask(self, context):
        """Apply context-specific mask, by setting activity of pre-selected subset of nodes to zero.

        [context]   <int>, starting from 1"""
        assert self.mask_dict is not None
        torchType = next(self.parameters()).detach()
        for i, excit_buffer in enumerate(self.excit_buffer_list):
            gating_mask = np.repeat(1.0, len(excit_buffer))
            gating_mask[self.mask_dict[context][i]] = 0.0
            excit_buffer.set_(torchType.new(gating_mask))

    def reset_XdGmask(self):
        """Remove context-specific mask, by setting all "excit-buffers" to 1."""
        torchType = next(self.parameters()).detach()
        for excit_buffer in self.excit_buffer_list:
            gating_mask = np.repeat(1.0, len(excit_buffer))
            excit_buffer.set_(torchType.new(gating_mask))

    def register_starting_param_values(self):
        """Register the starting parameter values into the model as a buffer."""
        for gen_params in self.param_list:
            for n, p in gen_params():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    self.register_buffer('{}_SI_prev_context'.format(n), p.detach().clone())

    def prepare_importance_estimates_dicts(self):
        """Prepare <dicts> to store running importance estimates and param-values before update."""
        W = {}
        p_old = {}
        for gen_params in self.param_list:
            for n, p in gen_params():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    W[n] = p.data.clone().zero_()
                    p_old[n] = p.data.clone()
        return W, p_old

    def update_importance_estimates(self, W, p_old):
        """Update the running parameter importance estimates in W."""
        for gen_params in self.param_list:
            for n, p in gen_params():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    if p.grad is not None:
                        W[n].add_(-p.grad * (p.detach() - p_old[n]))
                    p_old[n] = p.detach().clone()

    def update_omega(self, W, epsilon):
        """After completing training on a context, update the per-parameter regularization strength.

        [W]         <dict> estimated parameter-specific contribution to changes in total loss of completed context
        [epsilon]   <float> dampening parameter (to bound [omega] when [p_change] goes to 0)"""
        for gen_params in self.param_list:
            for n, p in gen_params():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    p_prev = getattr(self, '{}_SI_prev_context'.format(n))
                    p_current = p.detach().clone()
                    p_change = p_current - p_prev
                    omega_add = W[n] / (p_change ** 2 + epsilon)
                    try:
                        omega = getattr(self, '{}_SI_omega'.format(n))
                    except AttributeError:
                        omega = p.detach().clone().zero_()
                    omega_new = omega + omega_add
                    self.register_buffer('{}_SI_prev_context'.format(n), p_current)
                    self.register_buffer('{}_SI_omega'.format(n), omega_new)

    def surrogate_loss(self):
        """Calculate SI's surrogate loss."""
        try:
            losses = []
            for gen_params in self.param_list:
                for n, p in gen_params():
                    if p.requires_grad:
                        n = n.replace('.', '__')
                        prev_values = getattr(self, '{}_SI_prev_context'.format(n))
                        omega = getattr(self, '{}_SI_omega'.format(n))
                        losses.append((omega * (p - prev_values) ** 2).sum())
            return sum(losses)
        except AttributeError:
            return torch.tensor(0.0, device=self._device())

    def initialize_fisher(self):
        """Initialize diagonal fisher matrix with the prior precision (as in NCL)."""
        for gen_params in self.param_list:
            for n, p in gen_params():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    self.register_buffer('{}_EWC_prev_context'.format(n), p.detach().clone() * 0)
                    self.register_buffer('{}_EWC_estimated_fisher'.format(n), torch.ones(p.shape) / self.data_size)

    def estimate_fisher(self, dataset, allowed_classes=None):
        """After completing training on a context, estimate diagonal of Fisher Information matrix.

        [dataset]:          <DataSet> to be used to estimate FI-matrix
        [allowed_classes]:  <list> with class-indeces of 'allowed' or 'active' classes"""
        est_fisher_info = {}
        for gen_params in self.param_list:
            for n, p in gen_params():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    est_fisher_info[n] = p.detach().clone().zero_()
        mode = self.training
        self.eval()
        data_loader = get_data_loader(dataset, batch_size=1 if self.fisher_batch == 1 else self.fisher_batch, cuda=self._is_on_cuda())
        for index, (x, y) in enumerate(data_loader):
            if self.fisher_n is not None:
                if index >= self.fisher_n:
                    break
            x = x
            output = self(x) if allowed_classes is None else self(x)[:, allowed_classes]
            if self.fisher_labels == 'all':
                with torch.no_grad():
                    label_weights = F.softmax(output, dim=1)
                for label_index in range(output.shape[1]):
                    label = torch.LongTensor([label_index])
                    negloglikelihood = F.cross_entropy(output, label)
                    self.zero_grad()
                    negloglikelihood.backward(retain_graph=True if label_index + 1 < output.shape[1] else False)
                    for gen_params in self.param_list:
                        for n, p in gen_params():
                            if p.requires_grad:
                                n = n.replace('.', '__')
                                if p.grad is not None:
                                    est_fisher_info[n] += label_weights[0][label_index] * p.grad.detach() ** 2
            else:
                if self.fisher_labels == 'true':
                    label = torch.LongTensor([y]) if type(y) == int else y
                    if allowed_classes is not None:
                        label = [int(np.where(i == allowed_classes)[0][0]) for i in label.numpy()]
                        label = torch.LongTensor(label)
                    label = label
                elif self.fisher_labels == 'pred':
                    label = output.max(1)[1]
                elif self.fisher_labels == 'sample':
                    with torch.no_grad():
                        label_weights = F.softmax(output, dim=1)
                    weights_array = np.array(label_weights[0].cpu())
                    label = np.random.choice(len(weights_array), 1, p=weights_array / weights_array.sum())
                    label = torch.LongTensor(label)
                negloglikelihood = F.nll_loss(F.log_softmax(output, dim=1), label)
                self.zero_grad()
                negloglikelihood.backward()
                for gen_params in self.param_list:
                    for n, p in gen_params():
                        if p.requires_grad:
                            n = n.replace('.', '__')
                            if p.grad is not None:
                                est_fisher_info[n] += p.grad.detach() ** 2
        est_fisher_info = {n: (p / index) for n, p in est_fisher_info.items()}
        for gen_params in self.param_list:
            for n, p in gen_params():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    self.register_buffer('{}_EWC_prev_context{}'.format(n, self.context_count + 1 if self.offline else ''), p.detach().clone())
                    if not self.offline and hasattr(self, '{}_EWC_estimated_fisher'.format(n)):
                        existing_values = getattr(self, '{}_EWC_estimated_fisher'.format(n))
                        est_fisher_info[n] += self.gamma * existing_values
                    self.register_buffer('{}_EWC_estimated_fisher{}'.format(n, self.context_count + 1 if self.offline else ''), est_fisher_info[n])
        self.context_count += 1
        self.train(mode=mode)

    def ewc_loss(self):
        """Calculate EWC-loss."""
        try:
            losses = []
            num_penalty_terms = self.context_count if self.offline and self.context_count > 0 else 1
            for context in range(1, num_penalty_terms + 1):
                for gen_params in self.param_list:
                    for n, p in gen_params():
                        if p.requires_grad:
                            n = n.replace('.', '__')
                            mean = getattr(self, '{}_EWC_prev_context{}'.format(n, context if self.offline else ''))
                            fisher = getattr(self, '{}_EWC_estimated_fisher{}'.format(n, context if self.offline else ''))
                            fisher = fisher if self.offline else self.gamma * fisher
                            losses.append((fisher * (p - mean) ** 2).sum())
            return 1.0 / 2 * sum(losses)
        except AttributeError:
            return torch.tensor(0.0, device=self._device())

    def initialize_kfac_fisher(self):
        """Initialize Kronecker-factored Fisher matrix with the prior precision (as in NCL)."""
        fcE = self.fcE
        classifier = self.classifier

        def initialize_for_fcLayer(layer):
            if not isinstance(layer, fc.layers.fc_layer):
                raise NotImplemented
            linear = layer.linear
            g_dim, a_dim = linear.weight.shape
            abar_dim = a_dim + 1 if linear.bias is not None else a_dim
            A = torch.eye(abar_dim) / np.sqrt(self.data_size)
            G = torch.eye(g_dim) / np.sqrt(self.data_size)
            return {'A': A, 'G': G, 'weight': linear.weight.data * 0, 'bias': None if linear.bias is None else linear.bias.data * 0}

        def initialize():
            est_fisher_info = {}
            for i in range(1, fcE.layers + 1):
                label = f'fcLayer{i}'
                layer = getattr(fcE, label)
                est_fisher_info[label] = initialize_for_fcLayer(layer)
            est_fisher_info['classifier'] = initialize_for_fcLayer(classifier)
            return est_fisher_info
        self.KFAC_FISHER_INFO = initialize()

    def estimate_kfac_fisher(self, dataset, allowed_classes=None):
        """After completing training on a context, estimate KFAC Fisher Information matrix.

        [dataset]:          <DataSet> to be used to estimate FI-matrix
        [allowed_classes]:  <list> with class-indeces of 'allowed' or 'active' classes
        """
        None
        fcE = self.fcE
        classifier = self.classifier

        def initialize_for_fcLayer(layer):
            if not isinstance(layer, fc.layers.fc_layer):
                raise NotImplemented
            linear = layer.linear
            g_dim, a_dim = linear.weight.shape
            abar_dim = a_dim + 1 if linear.bias is not None else a_dim
            A = torch.zeros(abar_dim, abar_dim)
            G = torch.zeros(g_dim, g_dim)
            if linear.bias is None:
                bias = None
            else:
                bias = linear.bias.data.clone()
            return {'A': A, 'G': G, 'weight': linear.weight.data.clone(), 'bias': bias}

        def initialize():
            est_fisher_info = {}
            for i in range(1, fcE.layers + 1):
                label = f'fcLayer{i}'
                layer = getattr(fcE, label)
                est_fisher_info[label] = initialize_for_fcLayer(layer)
            est_fisher_info['classifier'] = initialize_for_fcLayer(classifier)
            return est_fisher_info

        def update_fisher_info_layer(est_fisher_info, intermediate, label, layer, n_samples):
            if not isinstance(layer, fc.layers.fc_layer):
                raise NotImplemented
            if not hasattr(layer, 'phantom'):
                raise Exception(f'Layer {label} does not have phantom parameters')
            g = layer.phantom.grad.detach()
            G = g[..., None] @ g[..., None, :]
            _a = intermediate[label].detach()
            assert _a.shape[0] == 1
            a = _a[0]
            if classifier.bias is None:
                abar = a
            else:
                o = torch.ones(*a.shape[0:-1], 1)
                abar = torch.cat((a, o), -1)
            A = abar[..., None] @ abar[..., None, :]
            Ao = est_fisher_info[label]['A']
            Go = est_fisher_info[label]['G']
            est_fisher_info[label]['A'] = Ao + A / n_samples
            est_fisher_info[label]['G'] = Go + G / n_samples

        def update_fisher_info(est_fisher_info, intermediate, n_samples):
            for i in range(1, fcE.layers + 1):
                label = f'fcLayer{i}'
                layer = getattr(fcE, label)
                update_fisher_info_layer(est_fisher_info, intermediate, label, layer, n_samples)
            update_fisher_info_layer(est_fisher_info, intermediate, 'classifier', self.classifier, n_samples)
        est_fisher_info = initialize()
        mode = self.training
        self.eval()
        data_loader = get_data_loader(dataset, batch_size=1, cuda=self._is_on_cuda())
        n_samples = len(data_loader) if self.fisher_n is None else self.fisher_n
        for i, (x, _) in enumerate(data_loader):
            if i > n_samples:
                break
            x = x
            _output, intermediate = self(x, return_intermediate=True)
            output = _output if allowed_classes is None else _output[:, allowed_classes]
            dist = Categorical(logits=F.log_softmax(output, dim=1))
            label = dist.sample().detach()
            negloglikelihood = F.nll_loss(F.log_softmax(output, dim=1), label)
            self.zero_grad()
            negloglikelihood.backward()
            update_fisher_info(est_fisher_info, intermediate, n_samples)
        for label in est_fisher_info:
            An = est_fisher_info[label]['A']
            Gn = est_fisher_info[label]['G']
            Ao = self.gamma * self.KFAC_FISHER_INFO[label]['A']
            Go = self.KFAC_FISHER_INFO[label]['G']
            As, Gs = additive_nearest_kf({'A': Ao, 'G': Go}, {'A': An, 'G': Gn})
            self.KFAC_FISHER_INFO[label]['A'] = As
            self.KFAC_FISHER_INFO[label]['G'] = Gs
            for param_name in ['weight', 'bias']:
                p = est_fisher_info[label][param_name]
                self.KFAC_FISHER_INFO[label][param_name] = p
        self.train(mode=mode)

    def ewc_kfac_loss(self):
        fcE = self.fcE

        def loss_for_layer(label, layer):
            if not isinstance(layer, fc.layers.fc_layer):
                raise NotImplemented
            info = self.KFAC_FISHER_INFO[label]
            A = info['A'].detach()
            G = info['G'].detach()
            bias0 = info['bias']
            weight0 = info['weight']
            bias = layer.linear.bias
            weight = layer.linear.weight
            if bias0 is not None and bias is not None:
                p = torch.cat([weight, bias[..., None]], -1)
                p0 = torch.cat([weight0, bias0[..., None]], -1)
            else:
                p = weight
                p0 = weight0
            assert p.shape[-1] == A.shape[1]
            assert p0.shape[-1] == A.shape[1]
            dp = p - p0
            return torch.sum(dp * (G @ dp @ A))
        classifier = self.classifier
        if self.context_count > 0:
            l = loss_for_layer('classifier', classifier)
            for i in range(1, fcE.layers + 1):
                label = f'fcLayer{i}'
                nl = loss_for_layer(label, getattr(fcE, label))
                l += nl
            return 0.5 * l
        else:
            return torch.tensor(0.0, device=self._device())

    def estimate_owm_fisher(self, dataset, **kwargs):
        """After completing training on a context, estimate OWM Fisher Information matrix based on [dataset]."""
        fcE = self.fcE
        classifier = self.classifier

        def initialize_for_fcLayer(layer):
            if not isinstance(layer, fc.layers.fc_layer):
                raise NotImplemented
            linear = layer.linear
            g_dim, a_dim = linear.weight.shape
            abar_dim = a_dim + 1 if linear.bias is not None else a_dim
            A = torch.zeros(abar_dim, abar_dim)
            return {'A': A, 'weight': linear.weight.data.clone(), 'bias': None if linear.bias is None else linear.bias.data.clone()}

        def initialize():
            est_fisher_info = {}
            for i in range(1, fcE.layers + 1):
                label = f'fcLayer{i}'
                layer = getattr(fcE, label)
                est_fisher_info[label] = initialize_for_fcLayer(layer)
            est_fisher_info['classifier'] = initialize_for_fcLayer(classifier)
            return est_fisher_info

        def update_fisher_info_layer(est_fisher_info, intermediate, label, n_samples):
            _a = intermediate[label].detach()
            assert _a.shape[0] == 1
            a = _a[0]
            if classifier.bias is None:
                abar = a
            else:
                o = torch.ones(*a.shape[0:-1], 1)
                abar = torch.cat((a, o), -1)
            A = abar[..., None] @ abar[..., None, :]
            Ao = est_fisher_info[label]['A']
            est_fisher_info[label]['A'] = Ao + A / n_samples

        def update_fisher_info(est_fisher_info, intermediate, n_samples):
            for i in range(1, fcE.layers + 1):
                label = f'fcLayer{i}'
                update_fisher_info_layer(est_fisher_info, intermediate, label, n_samples)
            update_fisher_info_layer(est_fisher_info, intermediate, 'classifier', n_samples)
        est_fisher_info = initialize()
        mode = self.training
        self.eval()
        data_loader = get_data_loader(dataset, batch_size=1, cuda=self._is_on_cuda())
        n_samples = len(data_loader) if self.fisher_n is None else self.fisher_n
        for i, (x, _) in enumerate(data_loader):
            if i > n_samples:
                break
            x = x
            output, intermediate = self(x, return_intermediate=True)
            self.zero_grad()
            update_fisher_info(est_fisher_info, intermediate, n_samples)
        if self.context_count == 0:
            self.KFAC_FISHER_INFO = {}
        for label in est_fisher_info:
            An = est_fisher_info[label]['A']
            if self.context_count == 0:
                self.KFAC_FISHER_INFO[label] = {}
                As = An
            else:
                Ao = self.gamma * self.KFAC_FISHER_INFO[label]['A']
                frac = 1 / (self.context_count + 1)
                As = (1 - frac) * Ao + frac * An
            self.KFAC_FISHER_INFO[label]['A'] = As
            for param_name in ['weight', 'bias']:
                p = est_fisher_info[label][param_name]
                self.KFAC_FISHER_INFO[label][param_name] = p
        self.context_count += 1
        self.train(mode=mode)


def reservoir_sampling(samples_so_far, budget):
    """Reservoir sampling algorithm to decide whether an new sample should be stored in the buffer or not."""
    if samples_so_far < budget:
        return samples_so_far
    rand = np.random.randint(0, samples_so_far + 1)
    if rand < budget:
        return rand
    else:
        return -1


class MemoryBuffer(nn.Module, metaclass=abc.ABCMeta):
    """Abstract module for classifier for maintaining a memory buffer using (global-)class-based reservoir sampling."""

    def __init__(self):
        super().__init__()
        self.use_memory_buffer = False
        self.budget = 100
        self.samples_so_far = 0
        self.contexts_so_far = []
        self.prototypes = False
        self.compute_means = True
        self.norm_exemplars = True

    def _device(self):
        return next(self.parameters()).device

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda

    @abc.abstractmethod
    def feature_extractor(self, images, **kwargs):
        pass

    def initialize_buffer(self, config, return_c=False):
        """Initalize the memory buffer with tensors of correct shape filled with zeros."""
        self.buffer_x = torch.zeros(self.budget, config['channels'], config['size'], config['size'], dtype=torch.float32, device=self._device())
        self.buffer_y = torch.zeros(self.budget, dtype=torch.int64, device=self._device())
        if return_c:
            self.buffer_c = torch.zeros(self.budget, dtype=torch.int64, device=self._device())
        pass

    def add_new_samples(self, x, y, c):
        """Process the data, and based on reservoir sampling algorithm potentially add to the buffer."""
        self.compute_means = True
        for index in range(x.shape[0]):
            reservoir_index = reservoir_sampling(self.samples_so_far, self.budget)
            self.samples_so_far += 1
            if reservoir_index >= 0:
                self.buffer_x[reservoir_index] = x[index]
                self.buffer_y[reservoir_index] = y[index]
                if hasattr(self, 'buffer_c'):
                    self.buffer_c[reservoir_index] = c[index]

    def sample_from_buffer(self, size):
        """Randomly sample [size] samples from the memory buffer."""
        samples_in_buffer = min(self.samples_so_far, self.budget)
        if size > samples_in_buffer:
            size = samples_in_buffer
        selected_indeces = np.random.choice(samples_in_buffer, size=size, replace=False)
        x = self.buffer_x[selected_indeces]
        y = self.buffer_y[selected_indeces]
        c = self.buffer_c[selected_indeces] if hasattr(self, 'buffer_c') else None
        return x, y, c

    def keep_track_of_contexts_so_far(self, c):
        self.contexts_so_far += [item.item() for item in c]

    def sample_contexts(self, size):
        if len(self.contexts_so_far) == 0:
            raise AssertionError('No contexts have been observed yet.')
        else:
            return torch.tensor(np.random.choice(self.contexts_so_far, size, replace=True))

    def classify_with_prototypes(self, x, context=None):
        """Classify images by nearest-prototype / nearest-mean-of-exemplars rule (after transform to feature space)

        INPUT:      x = <tensor> of size (bsz,ich,isz,isz) with input image batch

        OUTPUT:     scores = <tensor> of size (bsz,n_classes)
        """
        mode = self.training
        self.eval()
        batch_size = x.size(0)
        if self.compute_means:
            self.possible_classes = []
            memory_set_means = []
            for y in range(self.classes):
                if y in self.buffer_y:
                    self.possible_classes.append(y)
                    x_this_y = self.buffer_x[self.buffer_y == y]
                    c_this_y = self.buffer_c[self.buffer_y == y] if hasattr(self, 'buffer_c') else None
                    with torch.no_grad():
                        features = self.feature_extractor(x_this_y, context=c_this_y)
                    if self.norm_exemplars:
                        features = F.normalize(features, p=2, dim=1)
                    mu_y = features.mean(dim=0, keepdim=True)
                    if self.norm_exemplars:
                        mu_y = F.normalize(mu_y, p=2, dim=1)
                    memory_set_means.append(mu_y.squeeze())
                else:
                    memory_set_means.append(None)
            self.memory_set_means = memory_set_means
            self.compute_means = False
        memory_set_means = [self.memory_set_means[i] for i in self.possible_classes]
        means = torch.stack(memory_set_means)
        means = torch.stack([means] * batch_size)
        means = means.transpose(1, 2)
        with torch.no_grad():
            feature = self.feature_extractor(x, context=context)
        if self.norm_exemplars:
            feature = F.normalize(feature, p=2, dim=1)
        feature = feature.unsqueeze(2)
        feature = feature.expand_as(means)
        scores = -(feature - means).pow(2).sum(dim=1).squeeze()
        all_scores = torch.ones(batch_size, self.classes, device=self._device()) * -np.inf
        all_scores[:, self.possible_classes] = scores
        self.train(mode=mode)
        return scores


class ConvLayers(nn.Module):
    """Convolutional feature extractor model for (natural) images. Possible to return (pre)activations of each layer.
    Also possible to supply a [skip_first]- or [skip_last]-argument to the forward-function to only pass certain layers.

    Input:  [batch_size] x [image_channels] x [image_size] x [image_size] tensor
    Output: [batch_size] x [out_channels] x [out_size] x [out_size] tensor
                - with [out_channels] = [start_channels] x 2**[reducing_layers] x [block.expansion]
                       [out_size] = [image_size] / 2**[reducing_layers]"""

    def __init__(self, conv_type='standard', block_type='basic', num_blocks=2, image_channels=3, depth=5, start_channels=16, reducing_layers=None, batch_norm=True, nl='relu', output='normal', global_pooling=False, gated=False):
        """Initialize stacked convolutional layers (either "standard" or "res-net" ones--1st layer is always standard).

        [conv_type]         <str> type of conv-layers to be used: [standard|resnet]
        [block_type]        <str> block-type to be used: [basic|bottleneck] (only relevant if [type]=resNet)
        [num_blocks]        <int> or <list> (with len=[depth]-1) of # blocks in each layer
        [image_channels]    <int> # channels of input image to encode
        [depth]             <int> # layers
        [start_channels]    <int> # channels in 1st layer, doubled in every "rl" (=reducing layer)
        [reducing_layers]   <int> # layers in which image-size is halved & # channels doubled (default=[depth]-1)
                                      ("rl"'s are the last conv-layers; in 1st layer # channels cannot double)
        [batch_norm]        <bool> whether to use batch-norm after each convolution-operation
        [nl]                <str> non-linearity to be used: [relu|leakyrelu]
        [output]            <str>  if - "normal", final layer is same as all others
                                      - "none", final layer has no batchnorm or non-linearity
        [global_pooling]    <bool> whether to include global average pooling layer at very end
        [gated]             <bool> whether conv-layers should be gated (not implemented for ResNet-layers)"""
        conv_type = 'standard' if depth < 2 else conv_type
        if conv_type == 'resNet':
            num_blocks = [num_blocks] * (depth - 1) if type(num_blocks) == int else num_blocks
            assert len(num_blocks) == depth - 1
            block = conv_layers.Bottleneck if block_type == 'bottleneck' else conv_layers.BasicBlock
        type_label = 'C' if conv_type == 'standard' else 'R{}'.format('b' if block_type == 'bottleneck' else '')
        channel_label = '{}-{}x{}'.format(image_channels, depth, start_channels)
        block_label = ''
        if conv_type == 'resNet' and depth > 1:
            block_label += '-'
            for block_num in num_blocks:
                block_label += 'b{}'.format(block_num)
        nd_label = '{bn}{nl}{gp}{gate}{out}'.format(bn='b' if batch_norm else '', nl='l' if nl == 'leakyrelu' else '', gp='p' if global_pooling else '', gate='g' if gated else '', out='n' if output == 'none' else '')
        nd_label = '' if nd_label == '' else '-{}'.format(nd_label)
        super().__init__()
        self.depth = depth
        self.rl = depth - 1 if reducing_layers is None else reducing_layers if depth + 1 > reducing_layers else depth
        rl_label = '' if self.rl == self.depth - 1 else '-rl{}'.format(self.rl)
        self.label = '{}{}{}{}{}'.format(type_label, channel_label, block_label, rl_label, nd_label)
        self.block_expansion = block.expansion if conv_type == 'resNet' else 1
        double_factor = self.rl if self.rl < depth else depth - 1
        self.out_channels = start_channels * 2 ** double_factor * self.block_expansion if depth > 0 else image_channels
        self.start_channels = start_channels
        self.global_pooling = global_pooling
        output_channels = start_channels
        for layer_id in range(1, depth + 1):
            reducing = True if layer_id > depth - self.rl else False
            input_channels = image_channels if layer_id == 1 else output_channels * self.block_expansion
            output_channels = output_channels * 2 if reducing and not layer_id == 1 else output_channels
            if conv_type == 'standard' or layer_id == 1:
                conv_layer = conv_layers.conv_layer(input_channels, output_channels, stride=2 if reducing else 1, drop=0, nl='no' if output == 'none' and layer_id == depth else nl, batch_norm=False if output == 'none' and layer_id == depth else batch_norm, gated=False if output == 'none' and layer_id == depth else gated)
            else:
                conv_layer = conv_layers.res_layer(input_channels, output_channels, block=block, num_blocks=num_blocks[layer_id - 2], stride=2 if reducing else 1, drop=0, batch_norm=batch_norm, nl=nl, no_fnl=True if output == 'none' and layer_id == depth else False)
            setattr(self, 'convLayer{}'.format(layer_id), conv_layer)
        self.pooling = nn.AdaptiveAvgPool2d((1, 1)) if global_pooling else modules.Identity()

    def forward(self, x, skip_first=0, skip_last=0, return_lists=False):
        if return_lists:
            hidden_act_list = []
            pre_act_list = []
        for layer_id in range(skip_first + 1, self.depth + 1 - skip_last):
            x, pre_act = getattr(self, 'convLayer{}'.format(layer_id))(x, return_pa=True)
            if return_lists:
                pre_act_list.append(pre_act)
                if layer_id < self.depth - skip_last:
                    hidden_act_list.append(x)
        x = self.pooling(x)
        return (x, hidden_act_list, pre_act_list) if return_lists else x

    def out_size(self, image_size, ignore_gp=False):
        """Given [image_size] of input, return the size of the "final" image that is outputted."""
        out_size = int(np.ceil(image_size / 2 ** self.rl)) if self.depth > 0 else image_size
        return 1 if self.global_pooling and not ignore_gp else out_size

    def out_units(self, image_size, ignore_gp=False):
        """Given [image_size] of input, return the total number of units in the output."""
        return self.out_channels * self.out_size(image_size, ignore_gp=ignore_gp) ** 2

    def layer_info(self, image_size):
        """Return list with shape of all hidden layers."""
        layer_list = []
        reduce_number = 0
        double_number = 0
        for layer_id in range(1, self.depth):
            reducing = True if layer_id > self.depth - self.rl else False
            if reducing:
                reduce_number += 1
            if reducing and layer_id > 1:
                double_number += 1
            pooling = True if self.global_pooling and layer_id == self.depth - 1 else False
            expansion = 1 if layer_id == 1 else self.block_expansion
            layer_list.append([self.start_channels * 2 ** double_number * expansion, 1 if pooling else int(np.ceil(image_size / 2 ** reduce_number)), 1 if pooling else int(np.ceil(image_size / 2 ** reduce_number))])
        return layer_list

    def list_init_layers(self):
        """Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers)."""
        list = []
        for layer_id in range(1, self.depth + 1):
            list += getattr(self, 'convLayer{}'.format(layer_id)).list_init_layers()
        return list

    @property
    def name(self):
        return self.label


class Identity(nn.Module):
    """A nn-module to simply pass on the input data."""

    def forward(self, x):
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '()'
        return tmpstr


class fc_layer(nn.Module):
    """Fully connected layer, with possibility of returning "pre-activations".

    Input:  [batch_size] x ... x [in_size] tensor
    Output: [batch_size] x ... x [out_size] tensor"""

    def __init__(self, in_size, out_size, nl=nn.ReLU(), drop=0.0, bias=True, batch_norm=False, excitability=False, excit_buffer=False, gated=False, phantom=False):
        super().__init__()
        self.bias = False if batch_norm else bias
        if drop > 0:
            self.dropout = nn.Dropout(drop)
        self.linear = em.LinearExcitability(in_size, out_size, bias=False if batch_norm else bias, excitability=excitability, excit_buffer=excit_buffer)
        if batch_norm:
            self.bn = nn.BatchNorm1d(out_size)
        if gated:
            self.gate = nn.Linear(in_size, out_size)
            self.sigmoid = nn.Sigmoid()
        if phantom:
            self.phantom = nn.Parameter(torch.zeros(out_size), requires_grad=True)
        if isinstance(nl, nn.Module):
            self.nl = nl
        elif not nl == 'none':
            self.nl = nn.ReLU() if nl == 'relu' else nn.LeakyReLU() if nl == 'leakyrelu' else modules.Identity()

    def forward(self, x, return_pa=False, **kwargs):
        input = self.dropout(x) if hasattr(self, 'dropout') else x
        pre_activ = self.bn(self.linear(input)) if hasattr(self, 'bn') else self.linear(input)
        gate = self.sigmoid(self.gate(x)) if hasattr(self, 'gate') else None
        gated_pre_activ = gate * pre_activ if hasattr(self, 'gate') else pre_activ
        if hasattr(self, 'phantom'):
            gated_pre_activ = gated_pre_activ + self.phantom
        output = self.nl(gated_pre_activ) if hasattr(self, 'nl') else gated_pre_activ
        return (output, gated_pre_activ) if return_pa else output

    def list_init_layers(self):
        """Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers)."""
        return [self.linear, self.gate] if hasattr(self, 'gate') else [self.linear]


class MLP(nn.Module):
    """Module for a multi-layer perceptron (MLP).

    Input:  [batch_size] x ... x [size_per_layer[0]] tensor
    Output: (tuple of) [batch_size] x ... x [size_per_layer[-1]] tensor"""

    def __init__(self, input_size=1000, output_size=10, layers=2, hid_size=1000, hid_smooth=None, size_per_layer=None, drop=0, batch_norm=False, nl='relu', bias=True, excitability=False, excit_buffer=False, gated=False, phantom=False, output='normal'):
        """sizes: 0th=[input], 1st=[hid_size], ..., 1st-to-last=[hid_smooth], last=[output].
        [input_size]       # of inputs
        [output_size]      # of units in final layer
        [layers]           # of layers
        [hid_size]         # of units in each hidden layer
        [hid_smooth]       if None, all hidden layers have [hid_size] units, else # of units linearly in-/decreases s.t.
                             final hidden layer has [hid_smooth] units (if only 1 hidden layer, it has [hid_size] units)
        [size_per_layer]   None or <list> with for each layer number of units (1st element = number of inputs)
                                --> overwrites [input_size], [output_size], [layers], [hid_size] and [hid_smooth]
        [drop]             % of each layer's inputs that is randomly set to zero during training
        [batch_norm]       <bool>; if True, batch-normalization is applied to each layer
        [nl]               <str>; type of non-linearity to be used (options: "relu", "leakyrelu", "none")
        [gated]            <bool>; if True, each linear layer has an additional learnable gate
                                    (whereby the gate is controlled by the same input as that goes through the gate)
        [phantom]          <bool>; if True, add phantom parameters to pre-activations, used for computing KFAC Fisher
        [output]           <str>; if - "normal", final layer is same as all others
                                     - "none", final layer has no non-linearity
                                     - "sigmoid", final layer has sigmoid non-linearity"""
        super().__init__()
        self.output = output
        if size_per_layer is None:
            hidden_sizes = []
            if layers > 1:
                if hid_smooth is not None:
                    hidden_sizes = [int(x) for x in np.linspace(hid_size, hid_smooth, num=layers - 1)]
                else:
                    hidden_sizes = [int(x) for x in np.repeat(hid_size, layers - 1)]
            size_per_layer = [input_size] + hidden_sizes + [output_size]
        self.layers = len(size_per_layer) - 1
        nd_label = '{drop}{bias}{exc}{bn}{nl}{gate}'.format(drop='' if drop == 0 else 'd{}'.format(drop), bias='' if bias else 'n', exc='e' if excitability else '', bn='b' if batch_norm else '', nl='l' if nl == 'leakyrelu' else 'n' if nl == 'none' else '', gate='g' if gated else '')
        nd_label = '{}{}'.format('' if nd_label == '' else '-{}'.format(nd_label), '' if output == 'normal' else '-{}'.format(output))
        size_statement = ''
        for i in size_per_layer:
            size_statement += '{}{}'.format('-' if size_statement == '' else 'x', i)
        self.label = 'F{}{}'.format(size_statement, nd_label) if self.layers > 0 else ''
        for lay_id in range(1, self.layers + 1):
            in_size = size_per_layer[lay_id - 1]
            out_size = size_per_layer[lay_id]
            layer = fc_layer(in_size, out_size, bias=bias, excitability=excitability, excit_buffer=excit_buffer, batch_norm=False if lay_id == self.layers and not output == 'normal' else batch_norm, gated=gated, nl=('none' if output == 'none' else nn.Sigmoid()) if lay_id == self.layers and not output == 'normal' else nl, drop=drop if lay_id > 1 else 0.0, phantom=phantom)
            setattr(self, 'fcLayer{}'.format(lay_id), layer)
        if self.layers < 1:
            self.noLayers = Identity()

    def forward(self, x, return_intermediate=False):
        if return_intermediate:
            intermediate = {}
        for lay_id in range(1, self.layers + 1):
            if return_intermediate:
                intermediate[f'fcLayer{lay_id}'] = x
            x = getattr(self, 'fcLayer{}'.format(lay_id))(x)
        return (x, intermediate) if return_intermediate else x

    @property
    def name(self):
        return self.label

    def list_init_layers(self):
        """Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers)."""
        list = []
        for layer_id in range(1, self.layers + 1):
            list += getattr(self, 'fcLayer{}'.format(layer_id)).list_init_layers()
        return list


class fc_layer_fixed_gates(nn.Module):
    """Fully connected layer, with possibility of returning "pre-activations". Has fixed gates (of specified dimension).

    Input:  [batch_size] x ... x [in_size] tensor         &        [batch_size] x ... x [gate_size] tensor
    Output: [batch_size] x ... x [out_size] tensor"""

    def __init__(self, in_size, out_size, nl=nn.ReLU(), drop=0.0, bias=True, excitability=False, excit_buffer=False, batch_norm=False, gate_size=0, gating_prop=0.8, device='cpu'):
        super().__init__()
        if drop > 0:
            self.dropout = nn.Dropout(drop)
        self.linear = em.LinearExcitability(in_size, out_size, bias=False if batch_norm else bias, excitability=excitability, excit_buffer=excit_buffer)
        if batch_norm:
            self.bn = nn.BatchNorm1d(out_size)
        if gate_size > 0:
            self.gate_mask = torch.tensor(np.random.choice([0.0, 1.0], size=(gate_size, out_size), p=[gating_prop, 1.0 - gating_prop]), dtype=torch.float, device=device)
        if isinstance(nl, nn.Module):
            self.nl = nl
        elif not nl == 'none':
            self.nl = nn.ReLU() if nl == 'relu' else nn.LeakyReLU() if nl == 'leakyrelu' else modules.Identity()

    def forward(self, x, gate_input=None, return_pa=False):
        input = self.dropout(x) if hasattr(self, 'dropout') else x
        pre_activ = self.bn(self.linear(input)) if hasattr(self, 'bn') else self.linear(input)
        gate = torch.matmul(gate_input, self.gate_mask) if hasattr(self, 'gate_mask') else None
        gated_pre_activ = gate * pre_activ if hasattr(self, 'gate_mask') else pre_activ
        output = self.nl(gated_pre_activ) if hasattr(self, 'nl') else gated_pre_activ
        return (output, gated_pre_activ) if return_pa else output

    def list_init_layers(self):
        """Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers)."""
        return [self.linear, self.gate] if hasattr(self, 'gate') else [self.linear]


class MLP_gates(nn.Module):
    """Module for a multi-layer perceptron (MLP). Possible to return (pre)activations of each layer.
    Also possible to supply a [skip_first]- or [skip_last]-argument to the forward-function to only pass certain layers.
    With gates controlled by [gate_input] (of size [gate_size]) with a randomly selected masked (prop=[gating_prop]).

    Input:  [batch_size] x ... x [size_per_layer[0]] tensor         &        [batch_size] x [gate_size]
    Output: (tuple of) [batch_size] x ... x [size_per_layer[-1]] tensor"""

    def __init__(self, input_size=1000, output_size=10, layers=2, hid_size=1000, hid_smooth=None, size_per_layer=None, drop=0, batch_norm=False, nl='relu', bias=True, excitability=False, excit_buffer=False, gate_size=0, gating_prop=0.0, final_gate=False, output='normal', device='cpu'):
        """sizes: 0th=[input], 1st=[hid_size], ..., 1st-to-last=[hid_smooth], last=[output].
        [input_size]       # of inputs
        [output_size]      # of units in final layer
        [layers]           # of layers
        [hid_size]         # of units in each hidden layer
        [hid_smooth]       if None, all hidden layers have [hid_size] units, else # of units linearly in-/decreases s.t.
                             final hidden layer has [hid_smooth] units (if only 1 hidden layer, it has [hid_size] units)
        [size_per_layer]   None or <list> with for each layer number of units (1st element = number of inputs)
                                --> overwrites [input_size], [output_size], [layers], [hid_size] and [hid_smooth]
        [drop]             % of each layer's inputs that is randomly set to zero during training
        [batch_norm]       <bool>; if True, batch-normalization is applied to each layer
        [nl]               <str>; type of non-linearity to be used (options: "relu", "leakyrelu", "none")
        [gate_size]        <int>; if>0, each linear layer has gate controlled by separate inputs of size [gate_size]
        [gating_prop]      <float>; probability for each unit to be gated
        [final_gate]       <bool>; whether final layer is allowed to have a gate
        [output]           <str>; if - "normal", final layer is same as all others
                                     - "none", final layer has no non-linearity
                                     - "sigmoid", final layer has sigmoid non-linearity"""
        super().__init__()
        self.output = output
        if size_per_layer is None:
            hidden_sizes = []
            if layers > 1:
                if hid_smooth is not None:
                    hidden_sizes = [int(x) for x in np.linspace(hid_size, hid_smooth, num=layers - 1)]
                else:
                    hidden_sizes = [int(x) for x in np.repeat(hid_size, layers - 1)]
            size_per_layer = [input_size] + hidden_sizes + [output_size] if layers > 0 else [input_size]
        self.layers = len(size_per_layer) - 1
        nd_label = '{drop}{bias}{exc}{bn}{nl}{gate}'.format(drop='' if drop == 0 else 'd{}'.format(drop), bias='' if bias else 'n', exc='e' if excitability else '', bn='b' if batch_norm else '', nl='l' if nl == 'leakyrelu' else 'n' if nl == 'none' else '', gate='g{}m{}'.format(gate_size, gating_prop) if gate_size > 0 and gating_prop > 0.0 else '')
        nd_label = '{}{}'.format('' if nd_label == '' else '-{}'.format(nd_label), '' if output == 'normal' else '-{}'.format(output))
        size_statement = ''
        for i in size_per_layer:
            size_statement += '{}{}'.format('-' if size_statement == '' else 'x', i)
        self.label = 'F{}{}'.format(size_statement, nd_label) if self.layers > 0 else ''
        for lay_id in range(1, self.layers + 1):
            in_size = size_per_layer[lay_id - 1]
            out_size = size_per_layer[lay_id]
            if not gate_size > 0.0 or not gating_prop > 0.0 or lay_id == self.layers and not final_gate:
                layer = fc_layer(in_size, out_size, bias=bias, excitability=excitability, excit_buffer=excit_buffer, batch_norm=False if lay_id == self.layers and not output == 'normal' else batch_norm, nl=('none' if output == 'none' else nn.Sigmoid()) if lay_id == self.layers and not output == 'normal' else nl, drop=drop if lay_id > 1 else 0.0)
            else:
                layer = fc_layer_fixed_gates(in_size, out_size, bias=bias, excitability=excitability, excit_buffer=excit_buffer, batch_norm=False if lay_id == self.layers and not output == 'normal' else batch_norm, gate_size=gate_size, gating_prop=gating_prop, device=device, nl=('none' if output == 'none' else nn.Sigmoid()) if lay_id == self.layers and not output == 'normal' else nl, drop=drop if lay_id > 1 else 0.0)
            setattr(self, 'fcLayer{}'.format(lay_id), layer)
        if self.layers < 1:
            self.noLayers = Identity()

    def forward(self, x, gate_input=None, skip_first=0, skip_last=0, return_lists=False):
        if return_lists:
            hidden_act_list = []
            pre_act_list = []
        for lay_id in range(skip_first + 1, self.layers + 1 - skip_last):
            x, pre_act = getattr(self, 'fcLayer{}'.format(lay_id))(x, gate_input=gate_input, return_pa=True)
            if return_lists:
                pre_act_list.append(pre_act)
                if lay_id < self.layers - skip_last:
                    hidden_act_list.append(x)
        return (x, hidden_act_list, pre_act_list) if return_lists else x

    @property
    def name(self):
        return self.label

    def list_init_layers(self):
        """Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers)."""
        list = []
        for layer_id in range(1, self.layers + 1):
            list += getattr(self, 'fcLayer{}'.format(layer_id)).list_init_layers()
        return list


class fc_multihead_layer(nn.Module):
    """Fully connected layer with a separate head for each context.

    Input:  [batch_size] x ... x [in_size] tensor         &        [batch_size] x ... x [n_contexts] tensor
    Output: [batch_size] x ... x [out_size] tensor"""

    def __init__(self, in_size, classes, n_contexts, nl=nn.ReLU(), drop=0.0, bias=True, excitability=False, excit_buffer=False, batch_norm=False, device='cpu'):
        super().__init__()
        if drop > 0:
            self.dropout = nn.Dropout(drop)
        self.linear = em.LinearExcitability(in_size, classes, bias=False if batch_norm else bias, excitability=excitability, excit_buffer=excit_buffer)
        if batch_norm:
            self.bn = nn.BatchNorm1d(classes)
        if n_contexts > 0:
            self.gate_mask = torch.zeros(size=(n_contexts, classes), dtype=torch.float, device=device)
            classes_per_context = int(classes / n_contexts)
            for context_id in range(n_contexts):
                self.gate_mask[context_id, context_id * classes_per_context:(context_id + 1) * classes_per_context] = 1.0
        if isinstance(nl, nn.Module):
            self.nl = nl
        elif not nl == 'none':
            self.nl = nn.ReLU() if nl == 'relu' else nn.LeakyReLU() if nl == 'leakyrelu' else modules.Identity()

    def forward(self, x, gate_input=None, return_pa=False):
        input = self.dropout(x) if hasattr(self, 'dropout') else x
        pre_activ = self.bn(self.linear(input)) if hasattr(self, 'bn') else self.linear(input)
        gate = torch.matmul(gate_input, self.gate_mask) if hasattr(self, 'gate_mask') else None
        gated_pre_activ = gate * pre_activ if hasattr(self, 'gate_mask') else pre_activ
        output = self.nl(gated_pre_activ) if hasattr(self, 'nl') else gated_pre_activ
        return (output, gated_pre_activ) if return_pa else output

    def list_init_layers(self):
        """Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers)."""
        return [self.linear, self.gate] if hasattr(self, 'gate') else [self.linear]


class Classifier(ContinualLearner, MemoryBuffer):
    """Model for classifying images, "enriched" as ContinualLearner- and MemoryBuffer-object."""

    def __init__(self, image_size, image_channels, classes, conv_type='standard', depth=0, start_channels=64, reducing_layers=3, conv_bn=True, conv_nl='relu', num_blocks=2, global_pooling=False, no_fnl=True, conv_gated=False, fc_layers=3, fc_units=1000, fc_drop=0, fc_bn=True, fc_nl='relu', fc_gated=False, bias=True, excitability=False, excit_buffer=False, phantom=False, xdg_prob=0.0, n_contexts=5, multihead=False, device='cpu'):
        super().__init__()
        self.classes = classes
        self.label = 'Classifier'
        self.stream_classifier = True
        self.depth = depth
        self.fc_layers = fc_layers
        self.fc_drop = fc_drop
        self.phantom = phantom
        self.xdg_prob = xdg_prob
        self.n_contexts = n_contexts
        self.multihead = multihead
        self.update_every = 1
        self.binaryCE = False
        self.binaryCE_distill = False
        if fc_layers < 1:
            raise ValueError('The classifier needs to have at least 1 fully-connected layer.')
        self.convE = ConvLayers(conv_type=conv_type, block_type='basic', num_blocks=num_blocks, image_channels=image_channels, depth=depth, start_channels=start_channels, reducing_layers=reducing_layers, batch_norm=conv_bn, nl=conv_nl, global_pooling=global_pooling, gated=conv_gated, output='none' if no_fnl else 'normal')
        self.flatten = modules.Flatten()
        self.conv_out_units = self.convE.out_units(image_size)
        self.conv_out_size = self.convE.out_size(image_size)
        self.conv_out_channels = self.convE.out_channels
        if self.xdg_prob > 0.0:
            self.fcE = MLP_gates(input_size=self.conv_out_units, output_size=fc_units, layers=fc_layers - 1, hid_size=fc_units, drop=fc_drop, batch_norm=fc_bn, nl=fc_nl, bias=bias, excitability=excitability, excit_buffer=excit_buffer, gate_size=n_contexts, gating_prop=xdg_prob, final_gate=True, device=device)
        else:
            self.fcE = MLP(input_size=self.conv_out_units, output_size=fc_units, layers=fc_layers - 1, hid_size=fc_units, drop=fc_drop, batch_norm=fc_bn, nl=fc_nl, bias=bias, excitability=excitability, excit_buffer=excit_buffer, gated=fc_gated, phantom=phantom)
        mlp_output_size = fc_units if fc_layers > 1 else self.conv_out_units
        if self.multihead:
            self.classifier = fc_multihead_layer(mlp_output_size, classes, n_contexts, excit_buffer=True, nl='none', drop=fc_drop, device=device)
        else:
            self.classifier = fc_layer(mlp_output_size, classes, excit_buffer=True, nl='none', drop=fc_drop, phantom=phantom)
        self.convE.frozen = False
        self.fcE.frozen = False

    def list_init_layers(self):
        """Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers)."""
        list = []
        list += self.convE.list_init_layers()
        list += self.fcE.list_init_layers()
        list += self.classifier.list_init_layers()
        return list

    @property
    def name(self):
        if self.depth > 0 and self.fc_layers > 1:
            return '{}_{}_c{}'.format(self.convE.name, self.fcE.name, self.classes)
        elif self.depth > 0:
            return '{}_{}c{}'.format(self.convE.name, 'drop{}-'.format(self.fc_drop) if self.fc_drop > 0 else '', self.classes)
        elif self.fc_layers > 1:
            return '{}_c{}'.format(self.fcE.name, self.classes)
        else:
            return 'i{}_{}c{}'.format(self.conv_out_units, 'drop{}-'.format(self.fc_drop) if self.fc_drop > 0 else '', self.classes)

    def forward(self, x, context=None):
        if (self.xdg_prob > 0.0 or self.multihead) and context is not None and (type(context) == np.ndarray or context.dim() < 2):
            context_one_hot = lf.to_one_hot(context, classes=self.n_contexts, device=self._device())
        hidden = self.convE(x)
        flatten_x = self.flatten(hidden)
        final_features = self.fcE(flatten_x, context_one_hot) if self.xdg_prob > 0.0 else self.fcE(flatten_x)
        out = self.classifier(final_features, context_one_hot) if self.multihead else self.classifier(final_features)
        return out

    def feature_extractor(self, images, context=None):
        if (self.xdg_prob > 0.0 or self.multihead) and context is not None and (type(context) == np.ndarray or context.dim() < 2):
            context_one_hot = lf.to_one_hot(context, classes=self.n_contexts, device=self._device())
        hidden = self.convE(images)
        flatten_x = self.flatten(hidden)
        final_features = self.fcE(flatten_x, context_one_hot) if self.xdg_prob > 0.0 else self.fcE(flatten_x)
        return final_features

    def classify(self, x, context=None, no_prototypes=False):
        """For input [x] (image/"intermediate" features), return predicted "scores"/"logits" for [allowed_classes]."""
        if self.prototypes and not no_prototypes:
            return self.classify_with_prototypes(x, context=context)
        else:
            return self.forward(x, context=context)

    def train_a_batch(self, x, y, c=None, x_=None, y_=None, c_=None, scores_=None, rnt=0.5, **kwargs):
        """Train model for one batch ([x],[y]), possibly supplemented with replayed data ([x_],[y_/scores_]).

        [x]               <tensor> batch of inputs (could be None, in which case only 'replayed' data is used)
        [y]               <1D-tensor> batch of corresponding labels
        [c]               <1D-tensor> or <np.ndarray>; for each batch-element in [x] its context-ID  --OR--
                          <2D-tensor>; for each batch-element in [x] a probability for every context-ID
        [x_]              None or (<list> of) <tensor> batch of replayed inputs
        [y_]              None or (<list> of) <tensor> batch of corresponding "replayed" labels
        [c_]
        [scores_]         None or (<list> of) <tensor> 2Dtensor:[batch]x[classes] predicted "scores"/"logits" for [x_]
        [rnt]             <number> in [0,1], relative importance of new context
        """
        self.train()
        if self.convE.frozen:
            self.convE.eval()
        if self.fcE.frozen:
            self.fcE.eval()
        self.optimizer.zero_grad()
        if x_ is not None:
            y_hat = self(x_, c_)
            predL_r, distilL_r = None, None
            if y_ is not None and y_ is not None:
                if self.binaryCE:
                    binary_targets_ = lf.to_one_hot(y_.cpu(), y_hat.size(1))
                    predL_r = F.binary_cross_entropy_with_logits(input=y_hat, target=binary_targets_, reduction='none').sum(dim=1).mean()
                else:
                    predL_r = F.cross_entropy(y_hat, y_, reduction='mean')
            if scores_ is not None and scores_ is not None:
                kd_fn = lf.loss_fn_kd_binary if self.binaryCE else lf.loss_fn_kd
                distilL_r = kd_fn(scores=y_hat, target_scores=scores_, T=self.KD_temp)
            if self.replay_targets == 'hard':
                loss_replay = predL_r
            elif self.replay_targets == 'soft':
                loss_replay = distilL_r
        loss_replay = None if x_ is None else loss_replay
        if self.use_replay in ('inequality', 'both') and x_ is not None:
            if self.use_replay == 'both':
                loss_replay = (1 - rnt) * loss_replay
            loss_replay.backward()
            grad_rep = []
            for p in self.parameters():
                if p.requires_grad:
                    grad_rep.append(p.grad.data.view(-1))
            grad_rep = torch.cat(grad_rep)
            if self.use_replay == 'inequality':
                self.optimizer.zero_grad()
        if x is not None:
            y_hat = self(x, c)
            if self.binaryCE:
                binary_targets = lf.to_one_hot(y.cpu(), y_hat.size(1))
                predL = None if y is None else F.binary_cross_entropy_with_logits(input=y_hat, target=binary_targets, reduction='none').sum(dim=1).mean()
            else:
                predL = None if y is None else F.cross_entropy(input=y_hat, target=y, reduction='mean')
            loss_cur = predL
            accuracy = None if y is None else (y == y_hat.max(1)[1]).sum().item() / x.size(0)
        else:
            accuracy = predL = None
        if x_ is None or self.use_replay == 'inequality':
            loss_total = loss_cur
        elif self.use_replay == 'both':
            loss_total = rnt * loss_cur
        else:
            loss_total = loss_replay if x is None else rnt * loss_cur + (1 - rnt) * loss_replay
        weight_penalty_loss = None
        if self.weight_penalty:
            if self.importance_weighting == 'si':
                weight_penalty_loss = self.surrogate_loss()
            elif self.importance_weighting == 'fisher':
                if self.fisher_kfac:
                    weight_penalty_loss = self.ewc_kfac_loss()
                else:
                    weight_penalty_loss = self.ewc_loss()
            loss_total += self.reg_strength * weight_penalty_loss
        loss_total.backward()
        if self.use_replay in ('inequality', 'both') and x_ is not None:
            grad_cur = []
            for p in self.parameters():
                if p.requires_grad:
                    grad_cur.append(p.grad.view(-1))
            grad_cur = torch.cat(grad_cur)
            angle = (grad_cur * grad_rep).sum()
            if angle < 0:
                length_rep = (grad_rep * grad_rep).sum()
                grad_proj = grad_cur - angle / (length_rep + self.eps_agem) * grad_rep
                index = 0
                for p in self.parameters():
                    if p.requires_grad:
                        n_param = p.numel()
                        p.grad.copy_(grad_proj[index:index + n_param].view_as(p))
                        index += n_param
        self.optimizer.step()
        return {'loss_total': loss_total.item(), 'loss_current': loss_cur.item() if x is not None else 0, 'loss_replay': loss_replay.item() if loss_replay is not None and x is not None else 0, 'pred': predL.item() if predL is not None else 0, 'pred_r': predL_r.item() if x_ is not None and predL_r is not None else 0, 'distil_r': distilL_r.item() if scores_ is not None and distilL_r is not None else 0, 'param_reg': weight_penalty_loss.item() if weight_penalty_loss is not None else 0, 'accuracy': accuracy if accuracy is not None else 0.0}


class DeconvLayers(nn.Module):
    """"Deconvolutional" feature decoder model for (natural) images. Possible to return (pre)activations of each layer.
    Also possible to supply a [skip_first]- or [skip_last]-argument to the forward-function to only pass certain layers.

    Input:  [batch_size] x [in_channels] x [in_size] x [in_size] tensor
    Output: (tuple of) [batch_size] x [image_channels] x [final_size] x [final_size] tensor
                - with [final_size] = [in_size] x 2**[reducing_layers]
                       [in_channels] = [final_channels] x 2**min([reducing_layers], [depth]-1)"""

    def __init__(self, image_channels=3, final_channels=16, depth=5, reducing_layers=None, batch_norm=True, nl='relu', gated=False, output='normal', smaller_kernel=False, deconv_type='standard'):
        """[image_channels] # channels of image to decode
        [final_channels]    # channels in layer before output, was halved in every "rl" (=reducing layer) when moving
                                through model; corresponds to [start_channels] in "ConvLayers"-module
        [depth]             # layers (seen from the image, # channels is halved in each layer going to output image)
        [reducing_layers]   # of layers in which image-size is doubled & number of channels halved (default=[depth]-1)
                               ("rl"'s are the first conv-layers encountered--i.e., last conv-layers as seen from image)
                               (note that in the last layer # channels cannot be halved)
        [batch_norm]        <bool> whether to use batch-norm after each convolution-operation
        [nl]                <str> what non-linearity to use -- choices: [relu, leakyrelu, sigmoid, none]
        [gated]             <bool> whether deconv-layers should be gated
        [output]            <str>; if - "normal", final layer is same as all others
                                      - "none", final layer has no non-linearity
                                      - "sigmoid", final layer has sigmoid non-linearity
        [smaller_kernel]    <bool> if True, use kernel-size of 2 (instead of 4) & without padding in reducing-layers"""
        super().__init__()
        self.depth = depth if depth > 0 else 0
        self.rl = self.depth - 1 if reducing_layers is None else min(self.depth, reducing_layers)
        type_label = 'Deconv' if deconv_type == 'standard' else 'DeResNet'
        nd_label = '{bn}{nl}{gate}{out}'.format(bn='-bn' if batch_norm else '', nl='-lr' if nl == 'leakyrelu' else '', gate='-gated' if gated else '', out='' if output == 'normal' else '-{}'.format(output))
        self.label = '{}-ic{}-{}x{}-rl{}{}{}'.format(type_label, image_channels, self.depth, final_channels, self.rl, 's' if smaller_kernel else '', nd_label)
        if self.depth > 0:
            self.in_channels = final_channels * 2 ** min(self.rl, self.depth - 1)
            self.final_channels = final_channels
        self.image_channels = image_channels
        if self.depth > 0:
            output_channels = self.in_channels
            for layer_id in range(1, self.depth + 1):
                reducing = True if layer_id < self.rl + 1 else False
                input_channels = output_channels
                output_channels = int(output_channels / 2) if reducing else output_channels
                if deconv_type == 'standard':
                    new_layer = conv_layers.deconv_layer(input_channels, output_channels if layer_id < self.depth else image_channels, stride=2 if reducing else 1, batch_norm=batch_norm if layer_id < self.depth else False, nl=nl if layer_id < self.depth or output == 'normal' else 'none' if output == 'none' else nn.Sigmoid(), gated=gated, smaller_kernel=smaller_kernel)
                else:
                    new_layer = conv_layers.deconv_res_layer(input_channels, output_channels if layer_id < self.depth else image_channels, stride=2 if reducing else 1, batch_norm=batch_norm if layer_id < self.depth else False, nl=nl, smaller_kernel=smaller_kernel, output='normal' if layer_id < self.depth else output)
                setattr(self, 'deconvLayer{}'.format(layer_id), new_layer)

    def forward(self, x, skip_first=0, skip_last=0, return_lists=False):
        if return_lists:
            hidden_act_list = []
            pre_act_list = []
        if self.depth > 0:
            for layer_id in range(skip_first + 1, self.depth + 1 - skip_last):
                x, pre_act = getattr(self, 'deconvLayer{}'.format(layer_id))(x, return_pa=True)
                if return_lists:
                    pre_act_list.append(pre_act)
                    if layer_id < self.depth - skip_last:
                        hidden_act_list.append(x)
        return (x, hidden_act_list, pre_act_list) if return_lists else x

    def image_size(self, in_units):
        """Given the number of units fed in, return the size of the target image."""
        if self.depth > 0:
            input_image_size = np.sqrt(in_units / self.in_channels)
            return input_image_size * 2 ** self.rl
        else:
            return np.sqrt(in_units / self.image_channels)

    def list_init_layers(self):
        """Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers)."""
        list = []
        for layer_id in range(1, self.depth + 1):
            list += getattr(self, 'deconvLayer{}'.format(layer_id)).list_init_layers()
        return list

    @property
    def name(self):
        return self.label


class fc_layer_split(nn.Module):
    """Fully connected layer outputting [mean] and [logvar] for each unit.

    Input:  [batch_size] x ... x [in_size] tensor
    Output: tuple with two [batch_size] x ... x [out_size] tensors"""

    def __init__(self, in_size, out_size, nl_mean=nn.Sigmoid(), nl_logvar=nn.Hardtanh(min_val=-4.5, max_val=0.0), drop=0.0, bias=True, excitability=False, excit_buffer=False, batch_norm=False, gated=False):
        super().__init__()
        self.mean = fc_layer(in_size, out_size, drop=drop, bias=bias, excitability=excitability, excit_buffer=excit_buffer, batch_norm=batch_norm, gated=gated, nl=nl_mean)
        self.logvar = fc_layer(in_size, out_size, drop=drop, bias=False, excitability=excitability, excit_buffer=excit_buffer, batch_norm=batch_norm, gated=gated, nl=nl_logvar)

    def forward(self, x):
        return self.mean(x), self.logvar(x)

    def list_init_layers(self):
        """Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers)."""
        list = []
        list += self.mean.list_init_layers()
        list += self.logvar.list_init_layers()
        return list


class CondVAE(ContinualLearner):
    """Class for conditional variational auto-encoder (cond-VAE) model."""

    def __init__(self, image_size, image_channels, classes, conv_type='standard', depth=0, start_channels=64, reducing_layers=3, conv_bn=True, conv_nl='relu', num_blocks=2, global_pooling=False, no_fnl=True, conv_gated=False, fc_layers=3, fc_units=1000, fc_drop=0, fc_bn=False, fc_nl='relu', fc_gated=False, excit_buffer=False, prior='standard', z_dim=20, per_class=False, n_modes=1, recon_loss='BCE', network_output='sigmoid', deconv_type='standard', dg_gates=False, dg_type='context', dg_prop=0.0, contexts=5, scenario='task', device='cuda', classifier=True, **kwargs):
        """Class for variational auto-encoder (VAE) models."""
        super().__init__()
        self.label = 'CondVAE'
        self.image_size = image_size
        self.image_channels = image_channels
        self.classes = classes
        self.fc_layers = fc_layers
        self.z_dim = z_dim
        self.fc_units = fc_units
        self.fc_drop = fc_drop
        self.depth = depth
        self.recon_loss = recon_loss
        self.network_output = network_output
        self.dg_type = dg_type
        self.dg_prop = dg_prop
        self.dg_gates = dg_gates if dg_prop is not None and dg_prop > 0.0 else False
        self.gate_size = (contexts if dg_type == 'context' else classes) if self.dg_gates else 0
        self.scenario = scenario
        self.optimizer = None
        self.optim_list = []
        self.prior = prior
        self.per_class = per_class
        self.n_modes = n_modes * classes if self.per_class else n_modes
        self.modes_per_class = n_modes if self.per_class else None
        self.lamda_rcl = 1.0
        self.lamda_vl = 1.0
        self.lamda_pl = 1.0 if classifier else 0.0
        self.average = True
        if fc_layers < 1:
            raise ValueError('VAE cannot have 0 fully-connected layers!')
        self.convE = ConvLayers(conv_type=conv_type, block_type='basic', num_blocks=num_blocks, image_channels=image_channels, depth=self.depth, start_channels=start_channels, reducing_layers=reducing_layers, batch_norm=conv_bn, nl=conv_nl, output='none' if no_fnl else 'normal', global_pooling=global_pooling, gated=conv_gated)
        self.flatten = modules.Flatten()
        self.conv_out_units = self.convE.out_units(image_size)
        self.conv_out_size = self.convE.out_size(image_size)
        self.conv_out_channels = self.convE.out_channels
        self.fcE = MLP(input_size=self.conv_out_units, output_size=fc_units, layers=fc_layers - 1, hid_size=fc_units, drop=fc_drop, batch_norm=fc_bn, nl=fc_nl, gated=fc_gated, excit_buffer=excit_buffer)
        mlp_output_size = fc_units if fc_layers > 1 else self.conv_out_units
        self.toZ = fc_layer_split(mlp_output_size, z_dim, nl_mean='none', nl_logvar='none')
        if classifier:
            self.classifier = fc_layer(mlp_output_size, classes, excit_buffer=True, nl='none')
        out_nl = True if fc_layers > 1 else True if self.depth > 0 and not no_fnl else False
        real_h_dim_down = fc_units if fc_layers > 1 else self.convE.out_units(image_size, ignore_gp=True)
        if self.dg_gates:
            self.fromZ = fc_layer_fixed_gates(z_dim, real_h_dim_down, batch_norm=out_nl and fc_bn, nl=fc_nl if out_nl else 'none', gate_size=self.gate_size, gating_prop=dg_prop, device=device)
        else:
            self.fromZ = fc_layer(z_dim, real_h_dim_down, batch_norm=out_nl and fc_bn, nl=fc_nl if out_nl else 'none')
        if self.dg_gates:
            self.fcD = MLP_gates(input_size=fc_units, output_size=self.convE.out_units(image_size, ignore_gp=True), layers=fc_layers - 1, hid_size=fc_units, drop=fc_drop, batch_norm=fc_bn, nl=fc_nl, gate_size=self.gate_size, gating_prop=dg_prop, device=device, output=self.network_output if self.depth == 0 else 'normal')
        else:
            self.fcD = MLP(input_size=fc_units, output_size=self.convE.out_units(image_size, ignore_gp=True), layers=fc_layers - 1, hid_size=fc_units, drop=fc_drop, batch_norm=fc_bn, nl=fc_nl, gated=fc_gated, output=self.network_output if self.depth == 0 else 'normal')
        self.to_image = modules.Reshape(image_channels=self.convE.out_channels if self.depth > 0 else image_channels)
        self.convD = DeconvLayers(image_channels=image_channels, final_channels=start_channels, depth=self.depth, reducing_layers=reducing_layers, batch_norm=conv_bn, nl=conv_nl, gated=conv_gated, output=self.network_output, deconv_type=deconv_type)
        if self.prior == 'GMM':
            self.z_class_means = nn.Parameter(torch.Tensor(self.n_modes, self.z_dim))
            self.z_class_logvars = nn.Parameter(torch.Tensor(self.n_modes, self.z_dim))
            self.z_class_means.data.normal_()
            self.z_class_logvars.data.normal_()
        self.convE.frozen = False
        self.fcE.frozen = False

    def get_name(self):
        convE_label = '{}--'.format(self.convE.name) if self.depth > 0 else ''
        fcE_label = '{}--'.format(self.fcE.name) if self.fc_layers > 1 else '{}{}-'.format('h' if self.depth > 0 else 'i', self.conv_out_units)
        z_label = 'z{}{}'.format(self.z_dim, '' if self.prior == 'standard' else '-{}{}{}'.format(self.prior, self.n_modes, 'pc' if self.per_class else ''))
        class_label = '-c{}'.format(self.classes) if hasattr(self, 'classifier') else ''
        decoder_label = '_{}{}'.format('tg' if self.dg_type == 'context' else 'cg', self.dg_prop) if self.dg_gates else ''
        return '{}={}{}{}{}{}'.format(self.label, convE_label, fcE_label, z_label, class_label, decoder_label)

    @property
    def name(self):
        return self.get_name()

    def list_init_layers(self):
        """Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers)."""
        list = []
        list += self.convE.list_init_layers()
        list += self.fcE.list_init_layers()
        if hasattr(self, 'classifier'):
            list += self.classifier.list_init_layers()
        list += self.toZ.list_init_layers()
        list += self.fromZ.list_init_layers()
        list += self.fcD.list_init_layers()
        list += self.convD.list_init_layers()
        return list

    def layer_info(self):
        """Return list with shape of all hidden layers."""
        layer_list = self.convE.layer_info(image_size=self.image_size)
        if self.fc_layers > 0 and self.depth > 0:
            layer_list.append([self.conv_out_channels, self.conv_out_size, self.conv_out_size])
        if self.fc_layers > 1:
            for layer_id in range(1, self.fc_layers):
                layer_list.append([self.fc_layer_sizes[layer_id]])
        return layer_list

    def encode(self, x):
        """Pass input through feed-forward connections, to get [z_mean], [z_logvar] and [hE]."""
        hidden_x = self.convE(x)
        image_features = self.flatten(hidden_x)
        hE = self.fcE(image_features)
        z_mean, z_logvar = self.toZ(hE)
        return z_mean, z_logvar, hE, hidden_x

    def classify(self, x, allowed_classes=None, **kwargs):
        """For input [x] (image/"intermediate" features), return predicted "scores"/"logits" for [allowed_classes]."""
        if hasattr(self, 'classifier'):
            image_features = self.flatten(self.convE(x))
            hE = self.fcE(image_features)
            scores = self.classifier(hE)
            return scores if allowed_classes is None else scores[:, allowed_classes]
        else:
            return None

    def reparameterize(self, mu, logvar):
        """Perform "reparametrization trick" to make these stochastic variables differentiable."""
        std = logvar.mul(0.5).exp_()
        eps = std.new(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def decode(self, z, gate_input=None):
        """Decode latent variable activations.

        INPUT:  - [z]           <2D-tensor>; latent variables to be decoded
                - [gate_input]  <1D-tensor> or <np.ndarray>; for each batch-element in [x] its class-/context-ID  --OR--
                                <2D-tensor>; for each batch-element in [x] a probability for every class-/context-ID

        OUTPUT: - [image_recon] <4D-tensor>"""
        if self.dg_gates and gate_input is not None and (type(gate_input) == np.ndarray or gate_input.dim() < 2):
            gate_input = lf.to_one_hot(gate_input, classes=self.gate_size, device=self._device())
        hD = self.fromZ(z, gate_input=gate_input) if self.dg_gates else self.fromZ(z)
        image_features = self.fcD(hD, gate_input=gate_input) if self.dg_gates else self.fcD(hD)
        image_recon = self.convD(self.to_image(image_features))
        return image_recon

    def forward(self, x, gate_input=None, full=False, reparameterize=True, **kwargs):
        """Forward function to propagate [x] through the encoder, reparametrization and decoder.

        Input: - [x]          <4D-tensor> of shape [batch_size]x[channels]x[image_size]x[image_size]
               - [gate_input] <1D-tensor> or <np.ndarray>; for each batch-element in [x] its class-ID (eg, [y]) ---OR---
                              <2D-tensor>; for each batch-element in [x] a probability for each class-ID (eg, [y_hat])

        If [full] is True, output should be a <tuple> consisting of:
        - [x_recon]     <4D-tensor> reconstructed image (features) in same shape as [x] (or 2 of those: mean & logvar)
        - [y_hat]       <2D-tensor> with predicted logits for each class
        - [mu]          <2D-tensor> with either [z] or the estimated mean of [z]
        - [logvar]      None or <2D-tensor> estimated log(SD^2) of [z]
        - [z]           <2D-tensor> reparameterized [z] used for reconstruction
        If [full] is False, output is the reconstructed image (i.e., [x_recon]).
        """
        mu, logvar, hE, hidden_x = self.encode(x)
        z = self.reparameterize(mu, logvar) if reparameterize else mu
        gate_input = gate_input if self.dg_gates else None
        x_recon = self.decode(z, gate_input=gate_input)
        y_hat = self.classifier(hE) if hasattr(self, 'classifier') else None
        return (x_recon, y_hat, mu, logvar, z) if full else x_recon

    def feature_extractor(self, images):
        """Extract "final features" (i.e., after both conv- and fc-layers of forward pass) from provided images."""
        return self.fcE(self.flatten(self.convE(images)))

    def sample(self, size, allowed_classes=None, class_probs=None, sample_mode=None, allowed_domains=None, only_x=True, **kwargs):
        """Generate [size] samples from the model. Outputs are tensors (not "requiring grad"), on same device as <self>.

        INPUT:  - [allowed_classes]     <list> of [class_ids] from which to sample
                - [class_probs]         <list> with for each class the probability it is sampled from it
                - [sample_mode]         <int> to sample from specific mode of [z]-distr'n, overwrites [allowed_classes]
                - [allowed_domains]     <list> of [context_ids] which are allowed to be used for 'context-gates' (if used)
                                          NOTE: currently only relevant if [scenario]=="domain"

        OUTPUT: - [X]         <4D-tensor> generated images / image-features
                - [y_used]    <ndarray> labels of classes intended to be sampled  (using <class_ids>)
                - [context_used] <ndarray> labels of domains/contexts used for context-gates in decoder"""
        self.eval()
        if self.prior == 'GMM':
            if sample_mode is None:
                if allowed_classes is None and class_probs is None or not self.per_class:
                    sampled_modes = np.random.randint(0, self.n_modes, size)
                    y_used = np.array([int(mode / self.modes_per_class) for mode in sampled_modes]) if self.per_class else None
                else:
                    if allowed_classes is None:
                        allowed_classes = [i for i in range(len(class_probs))]
                    allowed_modes = []
                    unweighted_probs = []
                    for index, class_id in enumerate(allowed_classes):
                        allowed_modes += list(range(class_id * self.modes_per_class, (class_id + 1) * self.modes_per_class))
                        if class_probs is not None:
                            for i in range(self.modes_per_class):
                                unweighted_probs.append(class_probs[index].item())
                    mode_probs = None if class_probs is None else [(p / sum(unweighted_probs)) for p in unweighted_probs]
                    sampled_modes = np.random.choice(allowed_modes, size, p=mode_probs, replace=True)
                    y_used = np.array([int(mode / self.modes_per_class) for mode in sampled_modes])
            else:
                sampled_modes = np.repeat(sample_mode, size)
                y_used = np.repeat(int(sample_mode / self.modes_per_class), size) if self.per_class else None
        else:
            y_used = None
        if self.prior == 'GMM':
            prior_means = self.z_class_means
            prior_logvars = self.z_class_logvars
            z_means = prior_means[sampled_modes, :]
            z_logvars = prior_logvars[sampled_modes, :]
            with torch.no_grad():
                z = self.reparameterize(z_means, z_logvars)
        else:
            z = torch.randn(size, self.z_dim)
        if y_used is None and self.dg_gates:
            if allowed_classes is None and class_probs is None:
                y_used = np.random.randint(0, self.classes, size)
            else:
                if allowed_classes is None:
                    allowed_classes = [i for i in range(len(class_probs))]
                y_used = np.random.choice(allowed_classes, size, p=class_probs, replace=True)
        context_used = None
        if self.dg_gates and self.dg_type == 'context':
            if self.scenario == 'domain':
                context_used = np.random.randint(0, self.gate_size, size) if allowed_domains is None else np.random.choice(allowed_domains, size, replace=True)
            else:
                classes_per_context = int(self.classes / self.gate_size)
                context_used = np.array([int(class_id / classes_per_context) for class_id in y_used])
        with torch.no_grad():
            X = self.decode(z, gate_input=(context_used if self.dg_type == 'context' else y_used) if self.dg_gates else None)
        return X if only_x else (X, y_used, context_used)

    def calculate_recon_loss(self, x, x_recon, average=False):
        """Calculate reconstruction loss for each element in the batch.

        INPUT:  - [x]           <tensor> with original input (1st dimension (ie, dim=0) is "batch-dimension")
                - [x_recon]     (tuple of 2x) <tensor> with reconstructed input in same shape as [x]
                - [average]     <bool>, if True, loss is average over all pixels; otherwise it is summed

        OUTPUT: - [reconL]      <1D-tensor> of length [batch_size]"""
        batch_size = x.size(0)
        if self.recon_loss == 'MSE':
            reconL = -lf.log_Normal_standard(x=x, mean=x_recon, average=average, dim=-1)
        elif self.recon_loss == 'BCE':
            reconL = F.binary_cross_entropy(input=x_recon.view(batch_size, -1), target=x.view(batch_size, -1), reduction='none')
            reconL = torch.mean(reconL, dim=1) if average else torch.sum(reconL, dim=1)
        else:
            raise NotImplementedError('Wrong choice for type of reconstruction-loss!')
        return reconL

    def calculate_log_p_z(self, z, y=None, y_prob=None, allowed_classes=None):
        """Calculate log-likelihood of sampled [z] under the prior distirbution.

        INPUT:  - [z]        <2D-tensor> with sampled latent variables (1st dimension (ie, dim=0) is "batch-dimension")

        OPTIONS THAT ARE RELEVANT ONLY IF self.per_class IS TRUE:
            - [y]               None or <1D-tensor> with target-classes (as integers)
            - [y_prob]          None or <2D-tensor> with probabilities for each class (in [allowed_classes])
            - [allowed_classes] None or <list> with class-IDs to use for selecting prior-mode(s)

        OUTPUT: - [log_p_z]   <1D-tensor> of length [batch_size]"""
        if self.prior == 'standard':
            log_p_z = lf.log_Normal_standard(z, average=False, dim=1)
        if self.prior == 'GMM':
            allowed_modes = list(range(self.n_modes))
            if y is None and allowed_classes is not None and self.per_class:
                allowed_modes = []
                for class_id in allowed_classes:
                    allowed_modes += list(range(class_id * self.modes_per_class, (class_id + 1) * self.modes_per_class))
            prior_means = self.z_class_means[allowed_modes, :]
            prior_logvars = self.z_class_logvars[allowed_modes, :]
            z_expand = z.unsqueeze(1)
            means = prior_means.unsqueeze(0)
            logvars = prior_logvars.unsqueeze(0)
            n_modes = self.modes_per_class if (y is not None or y_prob is not None) and self.per_class else len(allowed_modes)
            a = lf.log_Normal_diag(z_expand, mean=means, log_var=logvars, average=False, dim=2) - math.log(n_modes)
            if y is not None and self.per_class:
                modes_list = list()
                for i in range(len(y)):
                    target = y[i].item()
                    modes_list.append(list(range(target * self.modes_per_class, (target + 1) * self.modes_per_class)))
                modes_tensor = torch.LongTensor(modes_list)
                a = a.gather(dim=1, index=modes_tensor)
            a_max, _ = torch.max(a, dim=1)
            a_exp = torch.exp(a - a_max.unsqueeze(1))
            if y is None and y_prob is not None and self.per_class:
                batch_size = y_prob.size(0)
                y_prob = y_prob.view(-1, 1).repeat(1, self.modes_per_class).view(batch_size, -1)
                a_logsum = torch.log(torch.clamp(torch.sum(y_prob * a_exp, dim=1), min=1e-40))
            else:
                a_logsum = torch.log(torch.clamp(torch.sum(a_exp, dim=1), min=1e-40))
            log_p_z = a_logsum + a_max
        return log_p_z

    def calculate_variat_loss(self, z, mu, logvar, y=None, y_prob=None, allowed_classes=None):
        """Calculate reconstruction loss for each element in the batch.

        INPUT:  - [z]        <2D-tensor> with sampled latent variables (1st dimension (ie, dim=0) is "batch-dimension")
                - [mu]       <2D-tensor> by encoder predicted mean for [z]
                - [logvar]   <2D-tensor> by encoder predicted logvar for [z]

        OPTIONS THAT ARE RELEVANT ONLY IF self.per_class IS TRUE:
            - [y]               None or <1D-tensor> with target-classes (as integers)
            - [y_prob]          None or <2D-tensor> with probabilities for each class (in [allowed_classes])
            - [allowed_classes] None or <list> with class-IDs to use for selecting prior-mode(s)

        OUTPUT: - [variatL]   <1D-tensor> of length [batch_size]"""
        if self.prior == 'standard':
            variatL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        elif self.prior == 'GMM':
            log_p_z = self.calculate_log_p_z(z, y=y, y_prob=y_prob, allowed_classes=allowed_classes)
            log_q_z_x = lf.log_Normal_diag(z, mean=mu, log_var=logvar, average=False, dim=1)
            variatL = -(log_p_z - log_q_z_x)
        return variatL

    def loss_function(self, x, y, x_recon, y_hat, scores, mu, z, logvar=None, allowed_classes=None, batch_weights=None):
        """Calculate and return various losses that could be used for training and/or evaluating the model.

        INPUT:  - [x]           <4D-tensor> original image
                - [y]           <1D-tensor> with target-classes (as integers, corresponding to [allowed_classes])
                - [x_recon]     (tuple of 2x) <4D-tensor> reconstructed image in same shape as [x]
                - [y_hat]       <2D-tensor> with predicted "logits" for each class (corresponding to [allowed_classes])
                - [scores]         <2D-tensor> with target "logits" for each class (corresponding to [allowed_classes])
                                     (if len(scores)<len(y_hat), 0 probs are added during distillation step at the end)
                - [mu]             <2D-tensor> with either [z] or the estimated mean of [z]
                - [z]              <2D-tensor> with reparameterized [z]
                - [logvar]         None or <2D-tensor> with estimated log(SD^2) of [z]
                - [batch_weights]  <1D-tensor> with a weight for each batch-element (if None, normal average over batch)
                - [allowed_classes]None or <list> with class-IDs to use for selecting prior-mode(s)

        OUTPUT: - [reconL]       reconstruction loss indicating how well [x] and [x_recon] match
                - [variatL]      variational (KL-divergence) loss "indicating how close distribion [z] is to prior"
                - [predL]        prediction loss indicating how well targets [y] are predicted
                - [distilL]      knowledge distillation (KD) loss indicating how well the predicted "logits" ([y_hat])
                                     match the target "logits" ([scores])"""
        batch_size = x.size(0)
        reconL = self.calculate_recon_loss(x=x.view(batch_size, -1), average=True, x_recon=x_recon.view(batch_size, -1))
        reconL = lf.weighted_average(reconL, weights=batch_weights, dim=0)
        if logvar is not None:
            actual_y = torch.tensor([allowed_classes[i.item()] for i in y]) if allowed_classes is not None and y is not None else y
            if y is None and scores is not None:
                y_prob = F.softmax(scores / self.KD_temp, dim=1)
                if allowed_classes is not None and len(allowed_classes) > y_prob.size(1):
                    n_batch = y_prob.size(0)
                    zeros_to_add = torch.zeros(n_batch, len(allowed_classes) - y_prob.size(1))
                    zeros_to_add = zeros_to_add
                    y_prob = torch.cat([y_prob, zeros_to_add], dim=1)
            else:
                y_prob = None
            variatL = self.calculate_variat_loss(z=z, mu=mu, logvar=logvar, y=actual_y, y_prob=y_prob, allowed_classes=allowed_classes)
            variatL = lf.weighted_average(variatL, weights=batch_weights, dim=0)
            variatL /= self.image_channels * self.image_size ** 2
        else:
            variatL = torch.tensor(0.0, device=self._device())
        if y is not None and y_hat is not None:
            predL = F.cross_entropy(input=y_hat, target=y, reduction='none')
            predL = lf.weighted_average(predL, weights=batch_weights, dim=0)
        else:
            predL = torch.tensor(0.0, device=self._device())
        if scores is not None and y_hat is not None:
            n_classes_to_consider = y_hat.size(1)
            distilL = lf.loss_fn_kd(scores=y_hat[:, :n_classes_to_consider], target_scores=scores, T=self.KD_temp, weights=batch_weights)
        else:
            distilL = torch.tensor(0.0, device=self._device())
        return reconL, variatL, predL, distilL

    def train_a_batch(self, x, y=None, x_=None, y_=None, scores_=None, contexts_=None, rnt=0.5, active_classes=None, context=1, **kwargs):
        """Train model for one batch ([x],[y]), possibly supplemented with replayed data ([x_],[y_]).

        [x]                 <tensor> batch of inputs (could be None, in which case only 'replayed' data is used)
        [y]                 None or <tensor> batch of corresponding labels
        [x_]                None or (<list> of) <tensor> batch of replayed inputs
        [y_]                None or (<list> of) <1Dtensor>:[batch] of corresponding "replayed" labels
        [scores_]           None or (<list> of) <2Dtensor>:[batch]x[classes] target "scores"/"logits" for [x_]
        [contexts_]         None or (<list> of) <1Dtensor>/<ndarray>:[batch] of context-IDs of replayed samples (as <int>)
        [rnt]               <number> in [0,1], relative importance of new context
        [active_classes]    None or (<list> of) <list> with "active" classes
        [context]           <int>, for setting context-specific mask"""
        self.train()
        if self.convE.frozen:
            self.convE.eval()
        if self.fcE.frozen:
            self.fcE.eval()
        self.optimizer.zero_grad()
        accuracy = 0.0
        if x is not None:
            context_tensor = None
            if self.dg_gates and self.dg_type == 'context':
                context_tensor = torch.tensor(np.repeat(context - 1, x.size(0)))
            recon_batch, y_hat, mu, logvar, z = self(x, gate_input=(context_tensor if self.dg_type == 'context' else y) if self.dg_gates else None, full=True, reparameterize=True)
            if active_classes is not None:
                class_entries = active_classes[-1] if type(active_classes[0]) == list else active_classes
                if y_hat is not None:
                    y_hat = y_hat[:, class_entries]
            reconL, variatL, predL, _ = self.loss_function(x=x, y=y, x_recon=recon_batch, y_hat=y_hat, scores=None, mu=mu, z=z, logvar=logvar, allowed_classes=class_entries if active_classes is not None else None)
            loss_cur = self.lamda_rcl * reconL + self.lamda_vl * variatL + self.lamda_pl * predL
            if y is not None and y_hat is not None:
                _, predicted = y_hat.max(1)
                accuracy = (y == predicted).sum().item() / x.size(0)
        if x_ is not None:
            PerContext = type(y_) == list if y_ is not None else type(scores_) == list
            if not PerContext:
                y_ = [y_]
                scores_ = [scores_]
                active_classes = [active_classes] if active_classes is not None else None
            n_replays = len(y_) if y_ is not None else len(scores_)
            loss_replay = [torch.tensor(0.0, device=self._device())] * n_replays
            reconL_r = [torch.tensor(0.0, device=self._device())] * n_replays
            variatL_r = [torch.tensor(0.0, device=self._device())] * n_replays
            predL_r = [torch.tensor(0.0, device=self._device())] * n_replays
            distilL_r = [torch.tensor(0.0, device=self._device())] * n_replays
            if not type(x_) == list and not (self.dg_gates and PerContext):
                y_predicted = None
                if self.dg_gates and self.dg_type == 'class':
                    if y_[0] is not None:
                        y_predicted = y_[0]
                    else:
                        y_predicted = F.softmax(scores_[0] / self.KD_temp, dim=1)
                        if y_predicted.size(1) < self.classes:
                            n_batch = y_predicted.size(0)
                            zeros_to_add = torch.zeros(n_batch, self.classes - y_predicted.size(1))
                            zeros_to_add = zeros_to_add
                            y_predicted = torch.cat([y_predicted, zeros_to_add], dim=1)
                x_temp_ = x_
                gate_input = (contexts_ if self.dg_type == 'context' else y_predicted) if self.dg_gates else None
                recon_batch, y_hat_all, mu, logvar, z = self(x_temp_, gate_input=gate_input, full=True)
            for replay_id in range(n_replays):
                if type(x_) == list or PerContext and self.dg_gates:
                    y_predicted = None
                    if self.dg_gates and self.dg_type == 'class':
                        if y_ is not None and y_[replay_id] is not None:
                            y_predicted = y_[replay_id]
                            y_predicted = y_predicted + replay_id * len(active_classes[0])
                        else:
                            y_predicted = F.softmax(scores_[replay_id] / self.KD_temp, dim=1)
                            if y_predicted.size(1) < self.classes:
                                n_batch = y_predicted.size(0)
                                zeros_to_add_before = torch.zeros(n_batch, replay_id * y_predicted.size(1))
                                zeros_to_add_before = zeros_to_add_before
                                zeros_to_add_after = torch.zeros(n_batch, self.classes - (replay_id + 1) * y_predicted.size(1))
                                zeros_to_add_after = zeros_to_add_after
                                y_predicted = torch.cat([zeros_to_add_before, y_predicted, zeros_to_add_after], dim=1)
                    x_temp_ = x_[replay_id] if type(x_) == list else x_
                    gate_input = (contexts_[replay_id] if self.dg_type == 'context' else y_predicted) if self.dg_gates else None
                    recon_batch, y_hat_all, mu, logvar, z = self(x_temp_, full=True, gate_input=gate_input)
                y_hat = y_hat_all if active_classes is None or y_hat_all is None else y_hat_all[:, active_classes[replay_id]]
                reconL_r[replay_id], variatL_r[replay_id], predL_r[replay_id], distilL_r[replay_id] = self.loss_function(x=x_temp_, y=y_[replay_id] if y_ is not None else None, x_recon=recon_batch, y_hat=y_hat, scores=scores_[replay_id] if scores_ is not None else None, mu=mu, z=z, logvar=logvar, allowed_classes=active_classes[replay_id] if active_classes is not None else None)
                loss_replay[replay_id] = self.lamda_rcl * reconL_r[replay_id] + self.lamda_vl * variatL_r[replay_id]
                if self.replay_targets == 'hard':
                    loss_replay[replay_id] += self.lamda_pl * predL_r[replay_id]
                elif self.replay_targets == 'soft':
                    loss_replay[replay_id] += self.lamda_pl * distilL_r[replay_id]
        loss_replay = None if x_ is None else sum(loss_replay) / n_replays
        loss_total = loss_replay if x is None else loss_cur if x_ is None else rnt * loss_cur + (1 - rnt) * loss_replay
        weight_penalty_loss = None
        if self.weight_penalty:
            if self.importance_weighting == 'si':
                weight_penalty_loss = self.surrogate_loss()
            elif self.importance_weighting == 'fisher':
                if self.fisher_kfac:
                    weight_penalty_loss = self.ewc_kfac_loss()
                else:
                    weight_penalty_loss = self.ewc_loss()
            loss_total += self.reg_strength * weight_penalty_loss
        loss_total.backward()
        self.optimizer.step()
        return {'loss_total': loss_total.item(), 'accuracy': accuracy, 'recon': reconL.item() if x is not None else 0, 'variat': variatL.item() if x is not None else 0, 'pred': predL.item() if x is not None else 0, 'recon_r': sum(reconL_r).item() / n_replays if x_ is not None else 0, 'variat_r': sum(variatL_r).item() / n_replays if x_ is not None else 0, 'pred_r': sum(predL_r).item() / n_replays if x_ is not None and predL_r[0] is not None else 0, 'distil_r': sum(distilL_r).item() / n_replays if x_ is not None and distilL_r[0] is not None else 0, 'param_reg': weight_penalty_loss.item() if weight_penalty_loss is not None else 0}


class BasicBlock(nn.Module):
    """Standard building block for ResNets."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, batch_norm=True, nl='relu', no_fnl=False):
        super(BasicBlock, self).__init__()
        self.block_layer1 = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False if batch_norm else True), nn.BatchNorm2d(planes) if batch_norm else modules.Identity(), nn.ReLU() if nl == 'relu' else nn.LeakyReLU())
        self.block_layer2 = nn.Sequential(nn.Conv2d(planes, self.expansion * planes, kernel_size=3, stride=1, padding=1, bias=False if batch_norm else True), nn.BatchNorm2d(self.expansion * planes) if batch_norm else modules.Identity())
        self.shortcut = modules.Identity()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False if batch_norm else True), nn.BatchNorm2d(self.expansion * planes) if batch_norm else modules.Identity())
        self.nl = (nn.ReLU() if nl == 'relu' else nn.LeakyReLU()) if not no_fnl else modules.Identity()

    def forward(self, x):
        out = self.block_layer2(self.block_layer1(x))
        out += self.shortcut(x)
        return self.nl(out)

    def list_init_layers(self):
        """Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers)."""
        list = [self.block_layer1[0], self.block_layer2[0]]
        if not type(self.shortcut) == modules.Identity:
            list.append(self.shortcut[0])
        return list


class Bottleneck(nn.Module):
    """Building block (with "bottleneck") for ResNets."""
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, batch_norm=True, nl='relu', no_fnl=False):
        super(Bottleneck, self).__init__()
        self.block_layer1 = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=1, bias=False if batch_norm else True), nn.BatchNorm2d(planes) if batch_norm else modules.Identity(), nn.ReLU() if nl == 'relu' else nn.LeakyReLU())
        self.block_layer2 = nn.Sequential(nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False if batch_norm else True), nn.BatchNorm2d(planes) if batch_norm else modules.Identity(), nn.ReLU() if nl == 'relu' else nn.LeakyReLU())
        self.block_layer3 = nn.Sequential(nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False if batch_norm else True), nn.BatchNorm2d(self.expansion * planes) if batch_norm else modules.Identity())
        self.shortcut = modules.Identity()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False if batch_norm else True), nn.BatchNorm2d(self.expansion * planes) if batch_norm else True)
        self.nl = (nn.ReLU() if nl == 'relu' else nn.LeakyReLU()) if not no_fnl else modules.Identity()

    def forward(self, x):
        out = self.block_layer3(self.block_layer2(self.block_layer1(x)))
        out += self.shortcut(x)
        return self.nl(out)

    def list_init_layers(self):
        """Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers)."""
        list = [self.block_layer1[0], self.block_layer2[0], self.block_layer3[0]]
        if not type(self.shortcut) == modules.Identity:
            list.append(self.shortcut[0])
        return list


class conv_layer(nn.Module):
    """Standard convolutional layer. Possible to return pre-activations."""

    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1, drop=0, batch_norm=False, nl=nn.ReLU(), bias=True, gated=False):
        super().__init__()
        if drop > 0:
            self.dropout = nn.Dropout2d(drop)
        self.conv = nn.Conv2d(in_planes, out_planes, stride=stride, kernel_size=kernel_size, padding=padding, bias=bias)
        if batch_norm:
            self.bn = nn.BatchNorm2d(out_planes)
        if gated:
            self.gate = nn.Conv2d(in_planes, out_planes, stride=stride, kernel_size=kernel_size, padding=padding, bias=False)
            self.sigmoid = nn.Sigmoid()
        if isinstance(nl, nn.Module):
            self.nl = nl
        elif not nl == 'none':
            self.nl = nn.ReLU() if nl == 'relu' else nn.LeakyReLU() if nl == 'leakyrelu' else modules.Identity()

    def forward(self, x, return_pa=False):
        input = self.dropout(x) if hasattr(self, 'dropout') else x
        pre_activ = self.bn(self.conv(input)) if hasattr(self, 'bn') else self.conv(input)
        gate = self.sigmoid(self.gate(x)) if hasattr(self, 'gate') else None
        gated_pre_activ = gate * pre_activ if hasattr(self, 'gate') else pre_activ
        output = self.nl(gated_pre_activ) if hasattr(self, 'nl') else gated_pre_activ
        return (output, gated_pre_activ) if return_pa else output

    def list_init_layers(self):
        """Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers)."""
        return [self.conv]


class res_layer(nn.Module):
    """Convolutional res-net layer. Possible to return pre-activations."""

    def __init__(self, in_planes, out_planes, block=BasicBlock, num_blocks=2, stride=1, drop=0, batch_norm=True, nl='relu', no_fnl=False):
        super().__init__()
        self.num_blocks = num_blocks
        self.in_planes = in_planes
        self.out_planes = out_planes * block.expansion
        self.dropout = nn.Dropout2d(drop)
        for block_id in range(num_blocks):
            new_block = block(in_planes, out_planes, stride=stride if block_id == 0 else 1, batch_norm=batch_norm, nl=nl, no_fnl=True if block_id == num_blocks - 1 else False)
            setattr(self, 'block{}'.format(block_id + 1), new_block)
            in_planes = out_planes * block.expansion
        self.nl = (nn.ReLU() if nl == 'relu' else nn.LeakyReLU()) if not no_fnl else modules.Identity()

    def forward(self, x, return_pa=False):
        x = self.dropout(x)
        for block_id in range(self.num_blocks):
            x = getattr(self, 'block{}'.format(block_id + 1))(x)
        output = self.nl(x)
        return (output, x) if return_pa else output

    def list_init_layers(self):
        """Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers)."""
        list = []
        for block_id in range(self.num_blocks):
            list += getattr(self, 'block{}'.format(block_id + 1)).list_init_layers()
        return list


class DeconvBlock(nn.Module):
    """Building block for deconv-layer with multiple blocks."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, batch_norm=True, nl='relu', no_fnl=False, smaller_kernel=False):
        super(DeconvBlock, self).__init__()
        self.block_layer1 = nn.Sequential(nn.ConvTranspose2d(in_planes, planes, stride=stride, bias=False if batch_norm else True, kernel_size=(2 if smaller_kernel else 4) if stride == 2 else 3, padding=0 if stride == 2 and smaller_kernel else 1), nn.BatchNorm2d(planes) if batch_norm else modules.Identity(), nn.ReLU() if nl == 'relu' else nn.LeakyReLU())
        self.block_layer2 = nn.Sequential(nn.ConvTranspose2d(planes, self.expansion * planes, kernel_size=3, stride=1, padding=1, bias=False if batch_norm else True), nn.BatchNorm2d(self.expansion * planes) if batch_norm else modules.Identity())
        self.shortcut = modules.Identity()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.ConvTranspose2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, output_padding=0 if stride == 1 else 1, bias=False if batch_norm else True), nn.BatchNorm2d(self.expansion * planes) if batch_norm else modules.Identity())
        self.nl = (nn.ReLU() if nl == 'relu' else nn.LeakyReLU()) if not no_fnl else modules.Identity()

    def forward(self, x):
        out = self.block_layer2(self.block_layer1(x))
        out += self.shortcut(x)
        return self.nl(out)

    def list_init_layers(self):
        """Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers)."""
        list = [self.block_layer1[0], self.block_layer2[0]]
        if not type(self.shortcut) == modules.Identity:
            list.append(self.shortcut[0])
        return list


class deconv_layer(nn.Module):
    """Standard "deconvolutional" layer. Possible to return pre-activations."""

    def __init__(self, input_channels, output_channels, stride=1, drop=0, batch_norm=True, nl='relu', bias=True, gated=False, smaller_kernel=False):
        super().__init__()
        if drop > 0:
            self.dropout = nn.Dropout2d(drop)
        self.deconv = nn.ConvTranspose2d(input_channels, output_channels, bias=bias, stride=stride, kernel_size=(2 if smaller_kernel else 4) if stride == 2 else 3, padding=0 if stride == 2 and smaller_kernel else 1)
        if batch_norm:
            self.bn = nn.BatchNorm2d(output_channels)
        if gated:
            self.gate = nn.ConvTranspose2d(input_channels, output_channels, bias=False, stride=stride, kernel_size=(2 if smaller_kernel else 4) if stride == 2 else 3, padding=0 if stride == 2 and smaller_kernel else 1)
            self.sigmoid = nn.Sigmoid()
        if isinstance(nl, nn.Module):
            self.nl = nl
        elif nl in ('sigmoid', 'hardtanh'):
            self.nl = nn.Sigmoid() if nl == 'sigmoid' else nn.Hardtanh(min_val=-4.5, max_val=0)
        elif not nl == 'none':
            self.nl = nn.ReLU() if nl == 'relu' else nn.LeakyReLU() if nl == 'leakyrelu' else modules.Identity()

    def forward(self, x, return_pa=False):
        input = self.dropout(x) if hasattr(self, 'dropout') else x
        pre_activ = self.bn(self.deconv(input)) if hasattr(self, 'bn') else self.deconv(input)
        gate = self.sigmoid(self.gate(x)) if hasattr(self, 'gate') else None
        gated_pre_activ = gate * pre_activ if hasattr(self, 'gate') else pre_activ
        output = self.nl(gated_pre_activ) if hasattr(self, 'nl') else gated_pre_activ
        return (output, gated_pre_activ) if return_pa else output

    def list_init_layers(self):
        """Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers)."""
        return [self.deconv]


class deconv_layer_split(nn.Module):
    """"Deconvolutional" layer outputing [mean] and [logvar] for each unit."""

    def __init__(self, input_channels, output_channels, nl_mean='sigmoid', nl_logvar='hardtanh', stride=1, drop=0, batch_norm=True, bias=True, gated=False, smaller_kernel=False):
        super().__init__()
        self.mean = deconv_layer(input_channels, output_channels, nl=nl_mean, smaller_kernel=smaller_kernel, stride=stride, drop=drop, batch_norm=batch_norm, bias=bias, gated=gated)
        self.logvar = deconv_layer(input_channels, output_channels, nl=nl_logvar, smaller_kernel=smaller_kernel, stride=stride, drop=drop, batch_norm=batch_norm, bias=False, gated=gated)

    def forward(self, x, return_pa=False):
        mean, pre_activ = self.mean(x, return_pa=True)
        logvar = self.logvar(x)
        return ((mean, logvar), pre_activ) if return_pa else (mean, logvar)

    def list_init_layers(self):
        """Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers)."""
        list = []
        list += self.mean.list_init_layers()
        list += self.logvar.list_init_layers()
        return list


class deconv_res_layer(nn.Module):
    """Deconvolutional res-net layer. Possible to return pre-activations."""

    def __init__(self, in_planes, out_planes, block=DeconvBlock, num_blocks=2, stride=1, drop=0, batch_norm=True, nl='relu', smaller_kernel=False, output='normal'):
        super().__init__()
        self.num_blocks = num_blocks
        self.in_planes = in_planes
        self.out_planes = out_planes * block.expansion
        self.dropout = nn.Dropout2d(drop)
        for block_id in range(num_blocks):
            new_block = block(in_planes, out_planes, stride=stride if block_id == 0 else 1, batch_norm=batch_norm, nl=nl, no_fnl=True if block_id == num_blocks - 1 else False, smaller_kernel=smaller_kernel)
            setattr(self, 'block{}'.format(block_id + 1), new_block)
            in_planes = out_planes * block.expansion
        if output == 'sigmoid':
            self.nl = nn.Sigmoid()
        elif output == 'normal':
            self.nl = nn.ReLU() if nl == 'relu' else nn.LeakyReLU()
        elif output == 'none':
            self.nl = modules.Identity()
        else:
            raise NotImplementedError("Ouptut '{}' not implemented for deconvolutional ResNet layer.".format(output))

    def forward(self, x, return_pa=False):
        x = self.dropout(x)
        for block_id in range(self.num_blocks):
            x = getattr(self, 'block{}'.format(block_id + 1))(x)
        output = self.nl(x)
        return (output, x) if return_pa else output

    def list_init_layers(self):
        """Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers)."""
        list = []
        for block_id in range(self.num_blocks):
            list += getattr(self, 'block{}'.format(block_id + 1)).list_init_layers()
        return list


def linearExcitability(input, weight, excitability=None, bias=None):
    """Applies a linear transformation to the incoming data: :math:`y = c(xA^T) + b`.

    Shape:
        - input:        :math:`(N, *, in_features)`
        - weight:       :math:`(out_features, in_features)`
        - excitability: :math:`(out_features)`
        - bias:         :math:`(out_features)`
        - output:       :math:`(N, *, out_features)`
    (NOTE: `*` means any number of additional dimensions)"""
    if excitability is not None:
        output = input.matmul(weight.t()) * excitability
    else:
        output = input.matmul(weight.t())
    if bias is not None:
        output += bias
    return output


class LinearExcitability(nn.Module):
    """Module for a linear transformation with multiplicative excitability-parameter (i.e., learnable) and/or -buffer.

    Args:
        in_features:    size of each input sample
        out_features:   size of each output sample
        bias:           if 'False', layer will not learn an additive bias-parameter (DEFAULT=True)
        excitability:   if 'True', layer will learn a multiplicative excitability-parameter (DEFAULT=False)
        excit_buffer:   if 'True', layer will have excitability-buffer whose value can be set (DEFAULT=False)

    Shape:
        - input:    :math:`(N, *, in_features)` where `*` means any number of additional dimensions
        - output:   :math:`(N, *, out_features)` where all but the last dimension are the same shape as the input.

    Attributes:
        weight:         the learnable weights of the module of shape (out_features x in_features)
        excitability:   the learnable multiplication terms (out_features)
        bias:           the learnable bias of the module of shape (out_features)
        excit_buffer:   fixed multiplication variable (out_features)"""

    def __init__(self, in_features, out_features, bias=True, excitability=False, excit_buffer=False):
        super(LinearExcitability, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if excitability:
            self.excitability = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('excitability', None)
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        if excit_buffer:
            buffer = torch.Tensor(out_features).uniform_(1, 1)
            self.register_buffer('excit_buffer', buffer)
        else:
            self.register_buffer('excit_buffer', None)
        self.reset_parameters()

    def reset_parameters(self):
        """Modifies the parameters "in-place" to initialize / reset them at appropriate values."""
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.excitability is not None:
            self.excitability.data.uniform_(1, 1)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        """Running this model's forward step requires/returns:
            -[input]:   [batch_size]x[...]x[in_features]
            -[output]:  [batch_size]x[...]x[hidden_features]"""
        if self.excit_buffer is None:
            excitability = self.excitability
        elif self.excitability is None:
            excitability = self.excit_buffer
        else:
            excitability = self.excitability * self.excit_buffer
        return linearExcitability(input, self.weight, excitability, self.bias)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'in_features=' + str(self.in_features) + ', out_features=' + str(self.out_features) + ')'


class FeatureExtractor(torch.nn.Module):
    """Model for encoding (i.e., feature extraction) and images."""

    def __init__(self, image_size, image_channels, conv_type='standard', depth=0, start_channels=64, reducing_layers=3, conv_bn=True, conv_nl='relu', num_blocks=2, global_pooling=False, no_fnl=True, conv_gated=False):
        super().__init__()
        self.label = 'FeatureExtractor'
        self.depth = depth
        self.optim_type = None
        self.optimizer = None
        self.optim_list = []
        self.convE = ConvLayers(conv_type=conv_type, block_type='basic', num_blocks=num_blocks, image_channels=image_channels, depth=depth, start_channels=start_channels, reducing_layers=reducing_layers, batch_norm=conv_bn, nl=conv_nl, global_pooling=global_pooling, gated=conv_gated, output='none' if no_fnl else 'normal')
        self.conv_out_units = self.convE.out_units(image_size)
        self.conv_out_size = self.convE.out_size(image_size)
        self.conv_out_channels = self.convE.out_channels

    @property
    def name(self):
        return self.convE.name

    def _device(self):
        return next(self.parameters()).device

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda

    def list_init_layers(self):
        """Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers)."""
        list = self.convE.list_init_layers()
        return list

    def forward(self, x):
        return self.convE(x)

    def train_discriminatively(self, train_loader, iters, classes, lr=0.001, optimizer='adam'):
        """Train the feature extractor for [iters] iterations on data from [train_loader].

        [model]             model to optimize
        [train_loader]      <dataloader> for training [model] on
        [iters]             <int> (max) number of iterations (i.e., batches) to train for
        [classes]           <int> number of possible clasess (softmax layer with that many units will be added to model)
        """
        self.flatten = modules.Flatten()
        self.classifier = fc_layer(self.conv_out_units, classes, excit_buffer=True, nl='none')
        optim_list = [{'params': filter(lambda p: p.requires_grad, self.parameters()), 'lr': lr}]
        self.optimizer = optim.SGD(optim_list) if optimizer == 'sgd' else optim.Adam(optim_list, betas=(0.9, 0.999))
        self.train()
        bar = tqdm.tqdm(total=iters)
        iteration = epoch = 0
        while iteration < iters:
            epoch += 1
            for batch_idx, (data, y) in enumerate(train_loader):
                iteration += 1
                self.optimizer.zero_grad()
                data, y = data, y
                features = self(data)
                y_hat = self.classifier(self.flatten(features))
                loss = F.cross_entropy(input=y_hat, target=y, reduction='mean')
                accuracy = None if y is None else (y == y_hat.max(1)[1]).sum().item() / data.size(0)
                loss.backward()
                self.optimizer.step()
                bar.set_description(' <FEAUTRE EXTRACTOR> | training loss: {loss:.3} | training accuracy: {prec:.3} |'.format(loss=loss.cpu().item(), prec=accuracy))
                bar.update(1)
                if iteration == iters:
                    bar.close()
                    break


class VAE(ContinualLearner):
    """Class for variational auto-encoder (VAE) model."""

    def __init__(self, image_size, image_channels, conv_type='standard', depth=0, start_channels=64, reducing_layers=3, conv_bn=True, conv_nl='relu', num_blocks=2, global_pooling=False, no_fnl=True, conv_gated=False, fc_layers=3, fc_units=1000, fc_drop=0, fc_bn=False, fc_nl='relu', fc_gated=False, excit_buffer=False, prior='standard', z_dim=20, n_modes=1, recon_loss='BCE', network_output='sigmoid', deconv_type='standard', **kwargs):
        """Class for variational auto-encoder (VAE) models."""
        super().__init__()
        self.label = 'VAE'
        self.image_size = image_size
        self.image_channels = image_channels
        self.fc_layers = fc_layers
        self.z_dim = z_dim
        self.fc_units = fc_units
        self.fc_drop = fc_drop
        self.depth = depth
        self.recon_loss = recon_loss
        self.network_output = network_output
        self.optimizer = None
        self.optim_list = []
        self.prior = prior
        self.n_modes = n_modes
        self.lamda_rcl = 1.0
        self.lamda_vl = 1.0
        self.average = True
        if fc_layers < 1:
            raise ValueError('VAE cannot have 0 fully-connected layers!')
        self.convE = ConvLayers(conv_type=conv_type, block_type='basic', num_blocks=num_blocks, image_channels=image_channels, depth=self.depth, start_channels=start_channels, reducing_layers=reducing_layers, batch_norm=conv_bn, nl=conv_nl, output='none' if no_fnl else 'normal', global_pooling=global_pooling, gated=conv_gated)
        self.flatten = modules.Flatten()
        self.conv_out_units = self.convE.out_units(image_size)
        self.conv_out_size = self.convE.out_size(image_size)
        self.conv_out_channels = self.convE.out_channels
        self.fcE = MLP(input_size=self.conv_out_units, output_size=fc_units, layers=fc_layers - 1, hid_size=fc_units, drop=fc_drop, batch_norm=fc_bn, nl=fc_nl, gated=fc_gated, excit_buffer=excit_buffer)
        mlp_output_size = fc_units if fc_layers > 1 else self.conv_out_units
        self.toZ = fc_layer_split(mlp_output_size, z_dim, nl_mean='none', nl_logvar='none')
        out_nl = True if fc_layers > 1 else True if self.depth > 0 and not no_fnl else False
        real_h_dim_down = fc_units if fc_layers > 1 else self.convE.out_units(image_size, ignore_gp=True)
        self.fromZ = fc_layer(z_dim, real_h_dim_down, batch_norm=out_nl and fc_bn, nl=fc_nl if out_nl else 'none')
        self.fcD = MLP(input_size=fc_units, output_size=self.convE.out_units(image_size, ignore_gp=True), layers=fc_layers - 1, hid_size=fc_units, drop=fc_drop, batch_norm=fc_bn, nl=fc_nl, gated=fc_gated, output=self.network_output if self.depth == 0 else 'normal')
        self.to_image = modules.Reshape(image_channels=self.convE.out_channels if self.depth > 0 else image_channels)
        self.convD = DeconvLayers(image_channels=image_channels, final_channels=start_channels, depth=self.depth, reducing_layers=reducing_layers, batch_norm=conv_bn, nl=conv_nl, gated=conv_gated, output=self.network_output, deconv_type=deconv_type)
        if self.prior == 'GMM':
            self.z_class_means = nn.Parameter(torch.Tensor(self.n_modes, self.z_dim))
            self.z_class_logvars = nn.Parameter(torch.Tensor(self.n_modes, self.z_dim))
            self.z_class_means.data.normal_()
            self.z_class_logvars.data.normal_()
        self.convE.frozen = False
        self.fcE.frozen = False

    def get_name(self):
        convE_label = '{}--'.format(self.convE.name) if self.depth > 0 else ''
        fcE_label = '{}--'.format(self.fcE.name) if self.fc_layers > 1 else '{}{}-'.format('h' if self.depth > 0 else 'i', self.conv_out_units)
        z_label = 'z{}{}'.format(self.z_dim, '' if self.prior == 'standard' else '-{}{}'.format(self.prior, self.n_modes))
        decoder_label = '--{}'.format(self.network_output)
        return '{}={}{}{}{}'.format(self.label, convE_label, fcE_label, z_label, decoder_label)

    @property
    def name(self):
        return self.get_name()

    def list_init_layers(self):
        """Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers)."""
        list = []
        list += self.convE.list_init_layers()
        list += self.fcE.list_init_layers()
        list += self.toZ.list_init_layers()
        list += self.fromZ.list_init_layers()
        list += self.fcD.list_init_layers()
        list += self.convD.list_init_layers()
        return list

    def layer_info(self):
        """Return list with shape of all hidden layers."""
        layer_list = self.convE.layer_info(image_size=self.image_size)
        if self.fc_layers > 0 and self.depth > 0:
            layer_list.append([self.conv_out_channels, self.conv_out_size, self.conv_out_size])
        if self.fc_layers > 1:
            for layer_id in range(1, self.fc_layers):
                layer_list.append([self.fc_layer_sizes[layer_id]])
        return layer_list

    def encode(self, x):
        """Pass input through feed-forward connections, to get [z_mean], [z_logvar] and [hE]."""
        hidden_x = self.convE(x)
        image_features = self.flatten(hidden_x)
        hE = self.fcE(image_features)
        z_mean, z_logvar = self.toZ(hE)
        return z_mean, z_logvar, hE, hidden_x

    def reparameterize(self, mu, logvar):
        """Perform "reparametrization trick" to make these stochastic variables differentiable."""
        std = logvar.mul(0.5).exp_()
        eps = std.new(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def decode(self, z):
        """Decode latent variable activations [z] (=<2D-tensor>) into [image_recon] (=<4D-tensor>)."""
        hD = self.fromZ(z)
        image_features = self.fcD(hD)
        image_recon = self.convD(self.to_image(image_features))
        return image_recon

    def forward(self, x, full=False, reparameterize=True, **kwargs):
        """Forward function to propagate [x] through the encoder, reparametrization and decoder.

        Input: - [x]          <4D-tensor> of shape [batch_size]x[channels]x[image_size]x[image_size]

        If [full] is True, output should be a <tuple> consisting of:
        - [x_recon]     <4D-tensor> reconstructed image (features) in same shape as [x] (or 2 of those: mean & logvar)
        - [mu]          <2D-tensor> with either [z] or the estimated mean of [z]
        - [logvar]      None or <2D-tensor> estimated log(SD^2) of [z]
        - [z]           <2D-tensor> reparameterized [z] used for reconstruction
        If [full] is False, output is the reconstructed image (i.e., [x_recon]).
        """
        mu, logvar, hE, hidden_x = self.encode(x)
        z = self.reparameterize(mu, logvar) if reparameterize else mu
        x_recon = self.decode(z)
        return (x_recon, mu, logvar, z) if full else x_recon

    def feature_extractor(self, images):
        """Extract "final features" (i.e., after both conv- and fc-layers of forward pass) from provided images."""
        return self.fcE(self.flatten(self.convE(images)))

    def sample(self, size, sample_mode=None, **kwargs):
        """Generate [size] samples from the model. Outputs are tensors (not "requiring grad"), on same device as <self>.

        INPUT:  - [sample_mode]   <int> to sample from specific mode of [z]-distribution

        OUTPUT: - [X]             <4D-tensor> generated images / image-features"""
        self.eval()
        if self.prior == 'GMM':
            if sample_mode is None:
                sampled_modes = np.random.randint(0, self.n_modes, size)
            else:
                sampled_modes = np.repeat(sample_mode, size)
        if self.prior == 'GMM':
            prior_means = self.z_class_means
            prior_logvars = self.z_class_logvars
            z_means = prior_means[sampled_modes, :]
            z_logvars = prior_logvars[sampled_modes, :]
            with torch.no_grad():
                z = self.reparameterize(z_means, z_logvars)
        else:
            z = torch.randn(size, self.z_dim)
        with torch.no_grad():
            X = self.decode(z)
        return X

    def calculate_recon_loss(self, x, x_recon, average=False):
        """Calculate reconstruction loss for each element in the batch.

        INPUT:  - [x]           <tensor> with original input (1st dimension (ie, dim=0) is "batch-dimension")
                - [x_recon]     (tuple of 2x) <tensor> with reconstructed input in same shape as [x]
                - [average]     <bool>, if True, loss is average over all pixels; otherwise it is summed

        OUTPUT: - [reconL]      <1D-tensor> of length [batch_size]"""
        batch_size = x.size(0)
        if self.recon_loss == 'MSE':
            reconL = -lf.log_Normal_standard(x=x, mean=x_recon, average=average, dim=-1)
        elif self.recon_loss == 'BCE':
            reconL = F.binary_cross_entropy(input=x_recon.view(batch_size, -1), target=x.view(batch_size, -1), reduction='none')
            reconL = torch.mean(reconL, dim=1) if average else torch.sum(reconL, dim=1)
        else:
            raise NotImplementedError('Wrong choice for type of reconstruction-loss!')
        return reconL

    def calculate_log_p_z(self, z):
        """Calculate log-likelihood of sampled [z] under the prior distribution.

        INPUT:  - [z]        <2D-tensor> with sampled latent variables (1st dimension (ie, dim=0) is "batch-dimension")

        OUTPUT: - [log_p_z]   <1D-tensor> of length [batch_size]"""
        if self.prior == 'standard':
            log_p_z = lf.log_Normal_standard(z, average=False, dim=1)
        if self.prior == 'GMM':
            allowed_modes = list(range(self.n_modes))
            prior_means = self.z_class_means[allowed_modes, :]
            prior_logvars = self.z_class_logvars[allowed_modes, :]
            z_expand = z.unsqueeze(1)
            means = prior_means.unsqueeze(0)
            logvars = prior_logvars.unsqueeze(0)
            n_modes = len(allowed_modes)
            a = lf.log_Normal_diag(z_expand, mean=means, log_var=logvars, average=False, dim=2) - math.log(n_modes)
            a_max, _ = torch.max(a, dim=1)
            a_exp = torch.exp(a - a_max.unsqueeze(1))
            a_logsum = torch.log(torch.clamp(torch.sum(a_exp, dim=1), min=1e-40))
            log_p_z = a_logsum + a_max
        return log_p_z

    def calculate_variat_loss(self, z, mu, logvar):
        """Calculate reconstruction loss for each element in the batch.

        INPUT:  - [z]        <2D-tensor> with sampled latent variables (1st dimension (ie, dim=0) is "batch-dimension")
                - [mu]       <2D-tensor> by encoder predicted mean for [z]
                - [logvar]   <2D-tensor> by encoder predicted logvar for [z]

        OUTPUT: - [variatL]   <1D-tensor> of length [batch_size]"""
        if self.prior == 'standard':
            variatL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        elif self.prior == 'GMM':
            log_p_z = self.calculate_log_p_z(z)
            log_q_z_x = lf.log_Normal_diag(z, mean=mu, log_var=logvar, average=False, dim=1)
            variatL = -(log_p_z - log_q_z_x)
        return variatL

    def loss_function(self, x, x_recon, mu, z, logvar=None, batch_weights=None):
        """Calculate and return various losses that could be used for training and/or evaluating the model.

        INPUT:  - [x]           <4D-tensor> original image
                - [x_recon]     (tuple of 2x) <4D-tensor> reconstructed image in same shape as [x]
                - [mu]             <2D-tensor> with either [z] or the estimated mean of [z]
                - [z]              <2D-tensor> with reparameterized [z]
                - [logvar]         None or <2D-tensor> with estimated log(SD^2) of [z]
                - [batch_weights]  <1D-tensor> with a weight for each batch-element (if None, normal average over batch)

        OUTPUT: - [reconL]       reconstruction loss indicating how well [x] and [x_recon] match
                - [variatL]      variational (KL-divergence) loss "indicating how close distribion [z] is to prior"
        """
        batch_size = x.size(0)
        x_recon = (x_recon[0].view(batch_size, -1), x_recon[1].view(batch_size, -1)) if self.network_output == 'split' else x_recon.view(batch_size, -1)
        reconL = self.calculate_recon_loss(x=x.view(batch_size, -1), average=True, x_recon=x_recon)
        reconL = lf.weighted_average(reconL, weights=batch_weights, dim=0)
        if logvar is not None:
            variatL = self.calculate_variat_loss(z=z, mu=mu, logvar=logvar)
            variatL = lf.weighted_average(variatL, weights=batch_weights, dim=0)
            variatL /= self.image_channels * self.image_size ** 2
        else:
            variatL = torch.tensor(0.0, device=self._device())
        return reconL, variatL

    def get_latent_lls(self, x):
        """Encode [x] as [z!x] and return log-likelihood.

        Input:  - [x]              <4D-tensor> of shape [batch]x[channels]x[image_size]x[image_size]

        Output: - [log_likelihood] <1D-tensor> of shape [batch]
        """
        z_mu, z_logvar, _, _ = self.encode(x)
        log_p_z = self.calculate_log_p_z(z_mu)
        return log_p_z

    def estimate_lls(self, x, S='mean', importance=True):
        """Estimate log-likelihood for [x] using [S] importance samples (or Monte Carlo samples, if [importance]=False).

        Input:  - [x]              <4D-tensor> of shape [batch]x[channels]x[image_size]x[image_size]
                - [S]              <int> (= # importance samples) or 'mean' (= use [z_mu] as single importance sample)
                - [importance]     <bool> if True do importance sampling, otherwise do Monte Carlo sampling

        Output: - [log_likelihood] <1D-tensor> of shape [batch]
        """
        if importance:
            z_mu, z_logvar, _, _ = self.encode(x)
        if S == 'mean':
            if importance:
                log_p_z = self.calculate_log_p_z(z_mu)
                z_mu_dummy = torch.zeros_like(z_mu)
                log_q_z_x = lf.log_Normal_diag(z_mu_dummy, mean=z_mu_dummy, log_var=z_logvar, average=False, dim=1)
            elif self.prior == 'GMM':
                sampled_modes = np.random.randint(0, self.n_modes, x.size(0))
                z_mu = self.z_class_means[sampled_modes, :]
            else:
                z_mu = torch.zeros(x.size(0), self.z_dim)
            x_recon = self.decode(z_mu)
            log_p_x_z = lf.log_Normal_standard(x=x, mean=x_recon, average=False, dim=-1)
            log_likelihood = log_p_x_z + log_p_z - log_q_z_x if importance else log_p_x_z
        else:
            all_lls = torch.zeros([S, x.size(0)], dtype=torch.float32, device=self._device())
            for s_id in range(S):
                if importance:
                    z = self.reparameterize(z_mu, z_logvar)
                    log_p_z = self.calculate_log_p_z(z)
                    log_q_z_x = lf.log_Normal_diag(z, mean=z_mu, log_var=z_logvar, average=False, dim=1)
                elif self.prior == 'GMM':
                    sampled_modes = np.random.randint(0, self.n_modes, x.size(0))
                    z_means = self.z_class_means[sampled_modes, :]
                    z_logvars = self.z_class_logvars[sampled_modes, :]
                    z = self.reparameterize(z_means, z_logvars)
                else:
                    z = torch.randn(x.size(0), self.z_dim)
                x_recon = self.decode(z)
                log_p_x_z = lf.log_Normal_standard(x=x, mean=x_recon, average=False, dim=-1)
                all_lls[s_id] = log_p_x_z + log_p_z - log_q_z_x if importance else log_p_x_z
            log_likelihood = all_lls.logsumexp(dim=0) - np.log(S)
        return log_likelihood

    def train_a_batch(self, x, x_=None, rnt=0.5, **kwargs):
        """Train model for one batch ([x]), possibly supplemented with replayed data ([x_]).

        [x]                 <tensor> batch of inputs (could be None, in which case only 'replayed' data is used)
        [x_]                None or (<list> of) <tensor> batch of replayed inputs
        [rnt]               <number> in [0,1], relative importance of new context
        """
        self.train()
        if self.convE.frozen:
            self.convE.eval()
        if self.fcE.frozen:
            self.fcE.eval()
        self.optimizer.zero_grad()
        if x is not None:
            recon_batch, mu, logvar, z = self(x, full=True, reparameterize=True)
            reconL, variatL = self.loss_function(x=x, x_recon=recon_batch, mu=mu, z=z, logvar=logvar)
            loss_cur = self.lamda_rcl * reconL + self.lamda_vl * variatL
        if x_ is not None:
            recon_batch, mu, logvar, z = self(x_, full=True, reparameterize=True)
            reconL_r, variatL_r = self.loss_function(x=x_, x_recon=recon_batch, mu=mu, z=z, logvar=logvar)
            loss_replay = self.lamda_rcl * reconL_r + self.lamda_vl * variatL_r
        loss_total = loss_replay if x is None else loss_cur if x_ is None else rnt * loss_cur + (1 - rnt) * loss_replay
        loss_total.backward()
        self.optimizer.step()
        return {'loss_total': loss_total.item(), 'recon': reconL.item() if x is not None else 0, 'variat': variatL.item() if x is not None else 0, 'recon_r': reconL_r.item() if x_ is not None else 0, 'variat_r': variatL_r.item() if x_ is not None else 0}


class GenerativeClassifier(nn.Module):
    """Class for generative classifier with separate VAE for each class to be learned."""

    def __init__(self, image_size, image_channels, classes, conv_type='standard', depth=0, start_channels=64, reducing_layers=3, conv_bn=True, conv_nl='relu', num_blocks=2, global_pooling=False, no_fnl=True, conv_gated=False, fc_layers=3, fc_units=1000, fc_drop=0, fc_bn=False, fc_nl='relu', excit_buffer=False, fc_gated=False, z_dim=20, prior='standard', n_modes=1, recon_loss='BCE', network_output='sigmoid', deconv_type='standard'):
        super().__init__()
        self.classes = classes
        self.label = 'GenClassifier'
        self.S = 'mean'
        self.importance = True
        self.from_latent = False
        for class_id in range(classes):
            new_vae = VAE(image_size, image_channels, conv_type=conv_type, depth=depth, start_channels=start_channels, reducing_layers=reducing_layers, conv_bn=conv_bn, conv_nl=conv_nl, num_blocks=num_blocks, global_pooling=global_pooling, no_fnl=no_fnl, conv_gated=conv_gated, fc_layers=fc_layers, fc_units=fc_units, fc_drop=fc_drop, fc_bn=fc_bn, fc_nl=fc_nl, excit_buffer=excit_buffer, fc_gated=fc_gated, z_dim=z_dim, prior=prior, n_modes=n_modes, recon_loss=recon_loss, network_output=network_output, deconv_type=deconv_type)
            setattr(self, 'vae{}'.format(class_id), new_vae)

    def get_name(self):
        return 'x{}-{}'.format(self.classes, self.vae0.get_name())

    @property
    def name(self):
        return self.get_name()

    def _device(self):
        return next(self.parameters()).device

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda

    def sample(self, size, only_x=True, class_id=None, **kwargs):
        """Generate [size] samples from the model. Outputs are tensors (not "requiring grad"), on same device."""
        for sample_id in range(size):
            selected_class_id = np.random.randint(0, self.classes, 1)[0] if class_id is None else class_id
            model_to_sample_from = getattr(self, 'vae{}'.format(selected_class_id))
            new_sample = model_to_sample_from.sample(1)
            X = torch.cat([X, new_sample], dim=0) if sample_id > 0 else new_sample
            if not only_x:
                y = torch.cat([y, torch.LongTensor([selected_class_id])]) if sample_id > 0 else torch.LongTensor([selected_class_id])
        return X if only_x else (X, y)

    def classify(self, x, allowed_classes=None, **kwargs):
        """Given an input [x], get the scores based on [self.S] importance samples (if self.S=='mean', use [z_mu]).

        Input:  - [x]        <4D-tensor> of shape [batch]x[channels]x[image_size]x[image_size]

        Output: - [scores]   <2D-tensor> of shape [batch]x[allowed_classes]
        """
        if allowed_classes is None:
            allowed_classes = list(range(self.classes))
        scores = torch.zeros([x.size(0), len(allowed_classes)], dtype=torch.float32, device=self._device())
        for class_id in allowed_classes:
            if self.from_latent:
                scores[:, class_id] = getattr(self, 'vae{}'.format(class_id)).get_latent_lls(x)
            else:
                scores[:, class_id] = getattr(self, 'vae{}'.format(class_id)).estimate_lls(x, S=self.S, importance=self.importance)
        return scores


class SeparateClassifiers(nn.Module):
    """Model for classifying images with a separate network for each context."""

    def __init__(self, image_size, image_channels, classes_per_context, contexts, conv_type='standard', depth=0, start_channels=64, reducing_layers=3, conv_bn=True, conv_nl='relu', num_blocks=2, global_pooling=False, no_fnl=True, conv_gated=False, fc_layers=3, fc_units=1000, fc_drop=0, fc_bn=True, fc_nl='relu', fc_gated=False, bias=True, excitability=False, excit_buffer=False):
        super().__init__()
        self.classes_per_context = classes_per_context
        self.contexts = contexts
        self.label = 'SeparateClassifiers'
        self.depth = depth
        self.fc_layers = fc_layers
        self.fc_drop = fc_drop
        if fc_layers < 1:
            raise ValueError('The classifier needs to have at least 1 fully-connected layer.')
        for context_id in range(self.contexts):
            new_network = Classifier(image_size, image_channels, classes_per_context, conv_type=conv_type, depth=depth, start_channels=start_channels, reducing_layers=reducing_layers, conv_bn=conv_bn, conv_nl=conv_nl, num_blocks=num_blocks, global_pooling=global_pooling, no_fnl=no_fnl, conv_gated=conv_gated, fc_layers=fc_layers, fc_units=fc_units, fc_drop=fc_drop, fc_bn=fc_bn, fc_nl=fc_nl, fc_gated=fc_gated, bias=bias, excitability=excitability, excit_buffer=excit_buffer)
            setattr(self, 'context{}'.format(context_id + 1), new_network)

    def _device(self):
        return next(self.parameters()).device

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda

    def list_init_layers(self):
        """Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers)."""
        list = []
        for context_id in range(self.contexts):
            list += getattr(self, 'context{}'.format(context_id + 1)).list_init_layers()
        return list

    @property
    def name(self):
        return 'SepNets-{}'.format(self.context1.name)

    def train_a_batch(self, x, y, c=None, context=None, **kwargs):
        """Train model for one batch ([x],[y]) from the indicated context.

        [x]               <tensor> batch of inputs (could be None, in which case only 'replayed' data is used)
        [y]               <tensor> batch of corresponding labels
        [c]               <1D-tensor> or <np.ndarray>; for each batch-element in [x] its context-ID
        [context]         <int> the context, can be used if all elements in [x] are from same context
        """
        if context is not None:
            loss_dict = getattr(self, 'context{}'.format(context)).train_a_batch(x, y)
        else:
            for context_id in range(self.contexts):
                if context_id in c:
                    x_to_use = x[c == context_id]
                    y_to_use = y[c == context_id]
                    loss_dict = getattr(self, 'context{}'.format(context_id + 1)).train_a_batch(x_to_use, y_to_use)
        return loss_dict


class Shape(nn.Module):
    """A nn-module to shape a tensor of shape [shape]."""

    def __init__(self, shape):
        super().__init__()
        self.shape = shape
        self.dim = len(shape)

    def forward(self, x):
        return x.view(*self.shape)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '(shape = {})'.format(self.shape)
        return tmpstr


class Reshape(nn.Module):
    """A nn-module to reshape a tensor(-tuple) to a 4-dim "image"-tensor(-tuple) with [image_channels] channels."""

    def __init__(self, image_channels):
        super().__init__()
        self.image_channels = image_channels

    def forward(self, x):
        if type(x) == tuple:
            batch_size = x[0].size(0)
            image_size = int(np.sqrt(x[0].nelement() / (batch_size * self.image_channels)))
            return (x_item.view(batch_size, self.image_channels, image_size, image_size) for x_item in x)
        else:
            batch_size = x.size(0)
            image_size = int(np.sqrt(x.nelement() / (batch_size * self.image_channels)))
            return x.view(batch_size, self.image_channels, image_size, image_size)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '(channels = {})'.format(self.image_channels)
        return tmpstr


class Flatten(nn.Module):
    """A nn-module to flatten a multi-dimensional tensor to 2-dim tensor."""

    def forward(self, x):
        batch_size = x.size(0)
        return x.view(batch_size, -1)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '()'
        return tmpstr


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'in_planes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Bottleneck,
     lambda: ([], {'in_planes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DeconvBlock,
     lambda: ([], {'in_planes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FeatureExtractor,
     lambda: ([], {'image_size': 4, 'image_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Flatten,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Identity,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LinearExcitability,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Reshape,
     lambda: ([], {'image_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Shape,
     lambda: ([], {'shape': [4, 4]}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (conv_layer,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (deconv_layer,
     lambda: ([], {'input_channels': 4, 'output_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (deconv_layer_split,
     lambda: ([], {'input_channels': 4, 'output_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (deconv_res_layer,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (res_layer,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_GMvandeVen_continual_learning(_paritybench_base):
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

