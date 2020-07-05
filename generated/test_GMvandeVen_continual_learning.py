import sys
_module = sys.modules[__name__]
del sys
_compare = _module
_compare_replay = _module
_compare_taskID = _module
_compare_time = _module
callbacks = _module
continual_learner = _module
data = _module
encoder = _module
evaluate = _module
excitability_modules = _module
exemplars = _module
linear_nets = _module
main = _module
param_stamp = _module
param_values = _module
replayer = _module
train = _module
utils = _module
vae_models = _module
visual_plt = _module
visual_visdom = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, string, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
import numpy as np
from torch import Tensor
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import abc


import numpy as np


import torch


from torch import nn


from torch.nn import functional as F


import math


from torch.nn.parameter import Parameter


import copy


from torch import optim


from torch.utils.data import ConcatDataset


import time


from torch.utils.data import DataLoader


from torch.utils.data.dataloader import default_collate


from torchvision import transforms


class ContinualLearner(nn.Module, metaclass=abc.ABCMeta):
    """Abstract module to add continual learning capabilities to a classifier.

    Adds methods for "context-dependent gating" (XdG), "elastic weight consolidation" (EWC) and
    "synaptic intelligence" (SI) to its subclasses."""

    def __init__(self):
        super().__init__()
        self.mask_dict = None
        self.excit_buffer_list = []
        self.si_c = 0
        self.epsilon = 0.1
        self.ewc_lambda = 0
        self.gamma = 1.0
        self.online = True
        self.fisher_n = None
        self.emp_FI = False
        self.EWC_task_count = 0

    def _device(self):
        return next(self.parameters()).device

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda

    @abc.abstractmethod
    def forward(self, x):
        pass

    def apply_XdGmask(self, task):
        """Apply task-specific mask, by setting activity of pre-selected subset of nodes to zero.

        [task]   <int>, starting from 1"""
        assert self.mask_dict is not None
        torchType = next(self.parameters()).detach()
        for i, excit_buffer in enumerate(self.excit_buffer_list):
            gating_mask = np.repeat(1.0, len(excit_buffer))
            gating_mask[self.mask_dict[task][i]] = 0.0
            excit_buffer.set_(torchType.new(gating_mask))

    def reset_XdGmask(self):
        """Remove task-specific mask, by setting all "excit-buffers" to 1."""
        torchType = next(self.parameters()).detach()
        for excit_buffer in self.excit_buffer_list:
            gating_mask = np.repeat(1.0, len(excit_buffer))
            excit_buffer.set_(torchType.new(gating_mask))

    def estimate_fisher(self, dataset, allowed_classes=None, collate_fn=None):
        """After completing training on a task, estimate diagonal of Fisher Information matrix.

        [dataset]:          <DataSet> to be used to estimate FI-matrix
        [allowed_classes]:  <list> with class-indeces of 'allowed' or 'active' classes"""
        est_fisher_info = {}
        for n, p in self.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                est_fisher_info[n] = p.detach().clone().zero_()
        mode = self.training
        self.eval()
        data_loader = utils.get_data_loader(dataset, batch_size=1, cuda=self._is_on_cuda(), collate_fn=collate_fn)
        for index, (x, y) in enumerate(data_loader):
            if self.fisher_n is not None:
                if index >= self.fisher_n:
                    break
            x = x
            output = self(x) if allowed_classes is None else self(x)[:, (allowed_classes)]
            if self.emp_FI:
                label = torch.LongTensor([y]) if type(y) == int else y
                if allowed_classes is not None:
                    label = [int(np.where(i == allowed_classes)[0][0]) for i in label.numpy()]
                    label = torch.LongTensor(label)
                label = label
            else:
                label = output.max(1)[1]
            negloglikelihood = F.nll_loss(F.log_softmax(output, dim=1), label)
            self.zero_grad()
            negloglikelihood.backward()
            for n, p in self.named_parameters():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    if p.grad is not None:
                        est_fisher_info[n] += p.grad.detach() ** 2
        est_fisher_info = {n: (p / index) for n, p in est_fisher_info.items()}
        for n, p in self.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                self.register_buffer('{}_EWC_prev_task{}'.format(n, '' if self.online else self.EWC_task_count + 1), p.detach().clone())
                if self.online and self.EWC_task_count == 1:
                    existing_values = getattr(self, '{}_EWC_estimated_fisher'.format(n))
                    est_fisher_info[n] += self.gamma * existing_values
                self.register_buffer('{}_EWC_estimated_fisher{}'.format(n, '' if self.online else self.EWC_task_count + 1), est_fisher_info[n])
        self.EWC_task_count = 1 if self.online else self.EWC_task_count + 1
        self.train(mode=mode)

    def ewc_loss(self):
        """Calculate EWC-loss."""
        if self.EWC_task_count > 0:
            losses = []
            for task in range(1, self.EWC_task_count + 1):
                for n, p in self.named_parameters():
                    if p.requires_grad:
                        n = n.replace('.', '__')
                        mean = getattr(self, '{}_EWC_prev_task{}'.format(n, '' if self.online else task))
                        fisher = getattr(self, '{}_EWC_estimated_fisher{}'.format(n, '' if self.online else task))
                        fisher = self.gamma * fisher if self.online else fisher
                        losses.append((fisher * (p - mean) ** 2).sum())
            return 1.0 / 2 * sum(losses)
        else:
            return torch.tensor(0.0, device=self._device())

    def update_omega(self, W, epsilon):
        """After completing training on a task, update the per-parameter regularization strength.

        [W]         <dict> estimated parameter-specific contribution to changes in total loss of completed task
        [epsilon]   <float> dampening parameter (to bound [omega] when [p_change] goes to 0)"""
        for n, p in self.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                p_prev = getattr(self, '{}_SI_prev_task'.format(n))
                p_current = p.detach().clone()
                p_change = p_current - p_prev
                omega_add = W[n] / (p_change ** 2 + epsilon)
                try:
                    omega = getattr(self, '{}_SI_omega'.format(n))
                except AttributeError:
                    omega = p.detach().clone().zero_()
                omega_new = omega + omega_add
                self.register_buffer('{}_SI_prev_task'.format(n), p_current)
                self.register_buffer('{}_SI_omega'.format(n), omega_new)

    def surrogate_loss(self):
        """Calculate SI's surrogate loss."""
        try:
            losses = []
            for n, p in self.named_parameters():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    prev_values = getattr(self, '{}_SI_prev_task'.format(n))
                    omega = getattr(self, '{}_SI_omega'.format(n))
                    losses.append((omega * (p - prev_values) ** 2).sum())
            return sum(losses)
        except AttributeError:
            return torch.tensor(0.0, device=self._device())


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


class ExemplarHandler(nn.Module, metaclass=abc.ABCMeta):
    """Abstract  module for a classifier that can store and use exemplars.

    Adds a exemplar-methods to subclasses, and requires them to provide a 'feature-extractor' method."""

    def __init__(self):
        super().__init__()
        self.exemplar_sets = []
        self.exemplar_means = []
        self.compute_means = True
        self.memory_budget = 2000
        self.norm_exemplars = True
        self.herding = True

    def _device(self):
        return next(self.parameters()).device

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda

    @abc.abstractmethod
    def feature_extractor(self, images):
        pass

    def reduce_exemplar_sets(self, m):
        for y, P_y in enumerate(self.exemplar_sets):
            self.exemplar_sets[y] = P_y[:m]

    def construct_exemplar_set(self, dataset, n):
        """Construct set of [n] exemplars from [dataset] using 'herding'.

        Note that [dataset] should be from specific class; selected sets are added to [self.exemplar_sets] in order."""
        mode = self.training
        self.eval()
        n_max = len(dataset)
        exemplar_set = []
        if self.herding:
            first_entry = True
            dataloader = utils.get_data_loader(dataset, 128, cuda=self._is_on_cuda())
            for image_batch, _ in dataloader:
                image_batch = image_batch
                with torch.no_grad():
                    feature_batch = self.feature_extractor(image_batch)
                if first_entry:
                    features = feature_batch
                    first_entry = False
                else:
                    features = torch.cat([features, feature_batch], dim=0)
            if self.norm_exemplars:
                features = F.normalize(features, p=2, dim=1)
            class_mean = torch.mean(features, dim=0, keepdim=True)
            if self.norm_exemplars:
                class_mean = F.normalize(class_mean, p=2, dim=1)
            exemplar_features = torch.zeros_like(features[:min(n, n_max)])
            list_of_selected = []
            for k in range(min(n, n_max)):
                if k > 0:
                    exemplar_sum = torch.sum(exemplar_features[:k], dim=0).unsqueeze(0)
                    features_means = (features + exemplar_sum) / (k + 1)
                    features_dists = features_means - class_mean
                else:
                    features_dists = features - class_mean
                index_selected = np.argmin(torch.norm(features_dists, p=2, dim=1))
                if index_selected in list_of_selected:
                    raise ValueError('Exemplars should not be repeated!!!!')
                list_of_selected.append(index_selected)
                exemplar_set.append(dataset[index_selected][0].numpy())
                exemplar_features[k] = copy.deepcopy(features[index_selected])
                features[index_selected] = features[index_selected] + 10000
        else:
            indeces_selected = np.random.choice(n_max, size=min(n, n_max), replace=False)
            for k in indeces_selected:
                exemplar_set.append(dataset[k][0].numpy())
        self.exemplar_sets.append(np.array(exemplar_set))
        self.train(mode=mode)

    def classify_with_exemplars(self, x, allowed_classes=None):
        """Classify images by nearest-means-of-exemplars (after transform to feature representation)

        INPUT:      x = <tensor> of size (bsz,ich,isz,isz) with input image batch
                    allowed_classes = None or <list> containing all "active classes" between which should be chosen

        OUTPUT:     preds = <tensor> of size (bsz,)"""
        mode = self.training
        self.eval()
        batch_size = x.size(0)
        if self.compute_means:
            exemplar_means = []
            for P_y in self.exemplar_sets:
                exemplars = []
                for ex in P_y:
                    exemplars.append(torch.from_numpy(ex))
                exemplars = torch.stack(exemplars)
                with torch.no_grad():
                    features = self.feature_extractor(exemplars)
                if self.norm_exemplars:
                    features = F.normalize(features, p=2, dim=1)
                mu_y = features.mean(dim=0, keepdim=True)
                if self.norm_exemplars:
                    mu_y = F.normalize(mu_y, p=2, dim=1)
                exemplar_means.append(mu_y.squeeze())
            self.exemplar_means = exemplar_means
            self.compute_means = False
        exemplar_means = self.exemplar_means if allowed_classes is None else [self.exemplar_means[i] for i in allowed_classes]
        means = torch.stack(exemplar_means)
        means = torch.stack([means] * batch_size)
        means = means.transpose(1, 2)
        with torch.no_grad():
            feature = self.feature_extractor(x)
        if self.norm_exemplars:
            feature = F.normalize(feature, p=2, dim=1)
        feature = feature.unsqueeze(2)
        feature = feature.expand_as(means)
        dists = (feature - means).pow(2).sum(dim=1).squeeze()
        _, preds = dists.min(1)
        self.train(mode=mode)
        return preds


class fc_layer(nn.Module):
    """Fully connected layer, with possibility of returning "pre-activations".

    Input:  [batch_size] x ... x [in_size] tensor
    Output: [batch_size] x ... x [out_size] tensor"""

    def __init__(self, in_size, out_size, nl=nn.ReLU(), drop=0.0, bias=True, excitability=False, excit_buffer=False, batch_norm=False, gated=False):
        super().__init__()
        if drop > 0:
            self.dropout = nn.Dropout(drop)
        self.linear = em.LinearExcitability(in_size, out_size, bias=False if batch_norm else bias, excitability=excitability, excit_buffer=excit_buffer)
        if batch_norm:
            self.bn = nn.BatchNorm1d(out_size)
        if gated:
            self.gate = nn.Linear(in_size, out_size)
            self.sigmoid = nn.Sigmoid()
        if isinstance(nl, nn.Module):
            self.nl = nl
        elif not nl == 'none':
            self.nl = nn.ReLU() if nl == 'relu' else nn.LeakyReLU() if nl == 'leakyrelu' else utils.Identity()

    def forward(self, x, return_pa=False):
        input = self.dropout(x) if hasattr(self, 'dropout') else x
        pre_activ = self.bn(self.linear(input)) if hasattr(self, 'bn') else self.linear(input)
        gate = self.sigmoid(self.gate(x)) if hasattr(self, 'gate') else None
        gated_pre_activ = gate * pre_activ if hasattr(self, 'gate') else pre_activ
        output = self.nl(gated_pre_activ) if hasattr(self, 'nl') else gated_pre_activ
        return (output, gated_pre_activ) if return_pa else output

    def list_init_layers(self):
        """Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers)."""
        return [self.linear, self.gate] if hasattr(self, 'gate') else [self.linear]


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


class MLP(nn.Module):
    """Module for a multi-layer perceptron (MLP).

    Input:  [batch_size] x ... x [size_per_layer[0]] tensor
    Output: (tuple of) [batch_size] x ... x [size_per_layer[-1]] tensor"""

    def __init__(self, input_size=1000, output_size=10, layers=2, hid_size=1000, hid_smooth=None, size_per_layer=None, drop=0, batch_norm=True, nl='relu', bias=True, excitability=False, excit_buffer=False, gated=False, output='normal'):
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
        [output]           <str>; if - "normal", final layer is same as all others
                                     - "BCE", final layer has sigmoid non-linearity"""
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
        nd_label = '{drop}{bias}{exc}{bn}{nl}{gate}{out}'.format(drop='' if drop == 0 else '-drop{}'.format(drop), bias='' if bias else '-noBias', exc='-exc' if excitability else '', bn='-bn' if batch_norm else '', nl='-lr' if nl == 'leakyrelu' else '', gate='-gated' if gated else '', out='' if output == 'normal' else '-{}'.format(output))
        self.label = 'MLP({}{})'.format(size_per_layer, nd_label) if self.layers > 0 else ''
        for lay_id in range(1, self.layers + 1):
            in_size = size_per_layer[lay_id - 1]
            out_size = size_per_layer[lay_id]
            if lay_id == self.layers and output in ('logistic', 'gaussian'):
                layer = fc_layer_split(in_size, out_size, bias=bias, excitability=excitability, excit_buffer=excit_buffer, drop=drop, batch_norm=False, gated=gated, nl_mean=nn.Sigmoid() if output == 'logistic' else utils.Identity(), nl_logvar=nn.Hardtanh(min_val=-4.5, max_val=0.0) if output == 'logistic' else utils.Identity())
            else:
                layer = fc_layer(in_size, out_size, bias=bias, excitability=excitability, excit_buffer=excit_buffer, drop=drop, batch_norm=False if lay_id == self.layers and not output == 'normal' else batch_norm, gated=gated, nl=nn.Sigmoid() if lay_id == self.layers and not output == 'normal' else nl)
            setattr(self, 'fcLayer{}'.format(lay_id), layer)
        if self.layers < 1:
            self.noLayers = utils.Identity()

    def forward(self, x):
        for lay_id in range(1, self.layers + 1):
            x = getattr(self, 'fcLayer{}'.format(lay_id))(x)
        return x

    @property
    def name(self):
        return self.label

    def list_init_layers(self):
        """Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers)."""
        list = []
        for layer_id in range(1, self.layers + 1):
            list += getattr(self, 'fcLayer{}'.format(layer_id)).list_init_layers()
        return list


class Replayer(nn.Module, metaclass=abc.ABCMeta):
    """Abstract  module for a classifier/generator that can be trained with replay.

    Initiates ability to reset state of optimizer between tasks."""

    def __init__(self):
        super().__init__()
        self.optimizer = None
        self.optim_type = 'adam'
        self.optim_list = []
        self.replay_targets = 'hard'
        self.KD_temp = 2.0

    def _device(self):
        return next(self.parameters()).device

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda

    @abc.abstractmethod
    def forward(self, x):
        pass


class Identity(nn.Module):
    """A nn-module to simply pass on the input data."""

    def forward(self, x):
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '()'
        return tmpstr


class Reshape(nn.Module):
    """A nn-module to reshape a tensor to a 4-dim "image"-tensor with [image_channels] channels."""

    def __init__(self, image_channels):
        super().__init__()
        self.image_channels = image_channels

    def forward(self, x):
        batch_size = x.size(0)
        image_size = int(np.sqrt(x.nelement() / (batch_size * self.image_channels)))
        return x.view(batch_size, self.image_channels, image_size, image_size)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '(channels = {})'.format(self.image_channels)
        return tmpstr


class ToImage(nn.Module):
    """Reshape input units to image with pixel-values between 0 and 1.

    Input:  [batch_size] x [in_units] tensor
    Output: [batch_size] x [image_channels] x [image_size] x [image_size] tensor"""

    def __init__(self, image_channels=1):
        super().__init__()
        self.reshape = Reshape(image_channels=image_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.reshape(x)
        x = self.sigmoid(x)
        return x

    def image_size(self, in_units):
        """Given the number of units fed in, return the size of the target image."""
        image_size = np.sqrt(in_units / self.image_channels)
        return image_size


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
    (ToImage,
     lambda: ([], {}),
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

