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
model = _module
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
metrics = _module
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

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import torch


import torch.optim as optim


import numpy as np


import torch.optim


import torch.utils.data as data


import collections.abc as abc


import typing


import torch.nn as nn


from torch.optim import Optimizer


import torch.nn.functional as F


import torch.distributions as dist


import torchvision.models.resnet as m


import torch.nn.init as init


import numbers


import collections


import torch.nn.utils


import torch.autograd


import torch.autograd as autograd


import time


import itertools as it


import math


import numpy.testing as nt


import torch.distributions as d


import torch.optim.lr_scheduler as scheduler


import re


import random


from torch.autograd import Variable


from collections import OrderedDict


class TrainingHistory:
    """
    Simple aggregator for the training history.

    An output of training storing scalar metrics in a pandas dataframe.
    """

    def __init__(self):
        self.data = []

    def add(self, epoch_result):
        """ Add a datapoint to the history """
        self.data.append(epoch_result)

    def frame(self):
        """ Return history dataframe """
        return pd.DataFrame(self.data).set_index('epoch_idx')


class TrainingInfo(abc.MutableMapping):
    """
    Information that need to persist through the whole training process

    Data dict is any extra information processes may want to store
    """

    def __init__(self, start_epoch_idx=0, run_name: typing.Optional[str]=None, metrics=None, callbacks=None):
        self.data_dict = {}
        self.start_epoch_idx = start_epoch_idx
        self.metrics = metrics if metrics is not None else []
        self.callbacks = callbacks if callbacks is not None else []
        self.run_name = run_name
        self.history = TrainingHistory()
        self.optimizer_initial_state = None

    def restore(self, hidden_state):
        """ Restore any state from checkpoint - currently not implemented but possible to do so in the future """
        for callback in self.callbacks:
            callback.load_state_dict(self, hidden_state)
        if 'optimizer' in hidden_state:
            self.optimizer_initial_state = hidden_state['optimizer']

    def initialize(self):
        """
        Runs for the first time a training process is started from scratch. Is guaranteed to be run only once
        for the training process. Will be run before all other callbacks.
        """
        for callback in self.callbacks:
            callback.on_initialization(self)

    def on_train_begin(self):
        """
        Beginning of a training process - is run every time a training process is started, even if it's restarted from
        a checkpoint.
        """
        for callback in self.callbacks:
            callback.on_train_begin(self)

    def on_train_end(self):
        """
        Finalize training process. Runs each time at the end of a training process.
        """
        for callback in self.callbacks:
            callback.on_train_end(self)

    def __getitem__(self, item):
        return self.data_dict[item]

    def __setitem__(self, key, value):
        self.data_dict[key] = value

    def __delitem__(self, key):
        del self.data_dict[key]

    def __iter__(self):
        return iter(self.data_dict)

    def __len__(self):
        return len(self.data_dict)

    def __contains__(self, item):
        return item in self.data_dict


class BaseMetric:
    """ Base class for all the metrics """

    def __init__(self, name):
        self.name = name

    def calculate(self, batch_info):
        """ Calculate value of a metric based on supplied data """
        raise NotImplementedError

    def reset(self):
        """ Reset value of a metric """
        raise NotImplementedError

    def value(self):
        """ Return current value for the metric """
        raise NotImplementedError

    def write_state_dict(self, training_info: TrainingInfo, hidden_state_dict: dict) ->None:
        """ Potentially store some metric state to the checkpoint """
        pass

    def load_state_dict(self, training_info: TrainingInfo, hidden_state_dict: dict) ->None:
        """ Potentially load some metric state from the checkpoint """
        pass


class AveragingMetric(BaseMetric):
    """ Base class for metrics that simply calculate the average over the epoch """

    def __init__(self, name):
        super().__init__(name)
        self.storage = []

    def calculate(self, batch_info):
        """ Calculate value of a metric """
        value = self._value_function(batch_info)
        self.storage.append(value)

    def _value_function(self, batch_info):
        raise NotImplementedError

    def reset(self):
        """ Reset value of a metric """
        self.storage = []

    def value(self):
        """ Return current value for the metric """
        return float(np.mean(self.storage))


class Loss(AveragingMetric):
    """ Just a loss function """

    def __init__(self):
        super().__init__('loss')

    def _value_function(self, batch_info):
        """ Just forward a value of the loss"""
        return batch_info['loss'].item()


class Model(nn.Module):
    """ Class representing full neural network model """

    def metrics(self) ->list:
        """ Set of metrics for this model """
        return [Loss()]

    def train(self, mode=True):
        """
        Sets the module in training mode.

        This has any effect only on certain modules. See documentations of
        particular modules for details of their behaviors in training/evaluation
        mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
        etc.

        Returns:
            Module: self
        """
        super().train(mode)
        if mode:
            mu.apply_leaf(self, mu.set_train_mode)
        return self

    def summary(self, input_size=None, hashsummary=False):
        """ Print a model summary """
        if input_size is None:
            None
            None
            number = sum(p.numel() for p in self.model.parameters())
            None
            None
        else:
            summary(self, input_size)
        if hashsummary:
            for idx, hashvalue in enumerate(self.hashsummary()):
                None

    def hashsummary(self):
        """ Print a model summary - checksums of each layer parameters """
        children = list(self.children())
        result = []
        for child in children:
            result.extend(hashlib.sha256(x.detach().cpu().numpy().tobytes()).hexdigest() for x in child.parameters())
        return result

    def get_layer_groups(self):
        """ Return layers grouped """
        return [self]

    def reset_weights(self):
        """ Call proper initializers for the weights """
        pass

    @property
    def is_recurrent(self) ->bool:
        """ If the network is recurrent and needs to be fed state as well as the observations """
        return False


class RnnModel(Model):
    """ Class representing recurrent model """

    @property
    def is_recurrent(self) ->bool:
        """ If the network is recurrent and needs to be fed previous state """
        return True

    @property
    def state_dim(self) ->int:
        """ Dimension of model state """
        raise NotImplementedError

    def zero_state(self, batch_size):
        """ Initial state of the network """
        return torch.zeros(batch_size, self.state_dim)


class BackboneModel(Model):
    """ Model that serves as a backbone network to connect your heads to """


class RnnLinearBackboneModel(BackboneModel):
    """
    Model that serves as a backbone network to connect your heads to -
    one that spits out a single-dimension output and is a recurrent neural network
    """

    @property
    def is_recurrent(self) ->bool:
        """ If the network is recurrent and needs to be fed previous state """
        return True

    @property
    def output_dim(self) ->int:
        """ Final dimension of model output """
        raise NotImplementedError

    @property
    def state_dim(self) ->int:
        """ Dimension of model state """
        raise NotImplementedError

    def zero_state(self, batch_size):
        """ Initial state of the network """
        return torch.zeros(batch_size, self.state_dim, dtype=torch.float32)


class LinearBackboneModel(BackboneModel):
    """
    Model that serves as a backbone network to connect your heads to - one that spits out a single-dimension output
    """

    @property
    def output_dim(self) ->int:
        """ Final dimension of model output """
        raise NotImplementedError


class SupervisedModel(Model):
    """ Model for a supervised learning problem """

    def loss(self, x_data, y_true):
        """ Forward propagate network and return a value of loss function """
        y_pred = self(x_data)
        return y_pred, self.loss_value(x_data, y_true, y_pred)

    def loss_value(self, x_data, y_true, y_pred):
        """ Calculate a value of loss function """
        raise NotImplementedError


class RnnSupervisedModel(RnnModel):
    """ Model for a supervised learning problem """

    def loss(self, x_data, y_true):
        """ Forward propagate network and return a value of loss function """
        y_pred = self(x_data)
        return y_pred, self.loss_value(x_data, y_true, y_pred)

    def loss_value(self, x_data, y_true, y_pred):
        """ Calculate a value of loss function """
        raise NotImplementedError


NET_OUTPUT = 1024


class Resnet34(SupervisedModel):
    """ Resnet34 network model """

    def __init__(self, fc_layers=None, dropout=None, pretrained=True):
        super().__init__()
        self.fc_layers = fc_layers
        self.dropout = dropout
        self.pretrained = pretrained
        self.head_layers = 8
        self.group_cut_layers = 6, 10
        backbone = m.resnet34(pretrained=pretrained)
        if fc_layers:
            valid_children = list(backbone.children())[:-2]
            valid_children.extend([l.AdaptiveConcatPool2d(), l.Flatten()])
            layer_inputs = [NET_OUTPUT] + fc_layers[:-1]
            dropout = dropout or [None] * len(fc_layers)
            for idx, (layer_input, layet_output, layer_dropout) in enumerate(zip(layer_inputs, fc_layers, dropout)):
                valid_children.append(nn.BatchNorm1d(layer_input))
                if layer_dropout:
                    valid_children.append(nn.Dropout(layer_dropout))
                valid_children.append(nn.Linear(layer_input, layet_output))
                if idx == len(fc_layers) - 1:
                    valid_children.append(nn.LogSoftmax(dim=1))
                else:
                    valid_children.append(nn.ReLU())
            final_model = nn.Sequential(*valid_children)
        else:
            final_model = backbone
        self.model = final_model

    def freeze(self, number=None):
        """ Freeze given number of layers in the model """
        if number is None:
            number = self.head_layers
        for idx, child in enumerate(self.model.children()):
            if idx < number:
                mu.freeze_layer(child)

    def unfreeze(self):
        """ Unfreeze model layers """
        for idx, child in enumerate(self.model.children()):
            mu.unfreeze_layer(child)

    def get_layer_groups(self):
        """ Return layers grouped """
        g1 = list(self.model[:self.group_cut_layers[0]])
        g2 = list(self.model[self.group_cut_layers[0]:self.group_cut_layers[1]])
        g3 = list(self.model[self.group_cut_layers[1]:])
        return [g1, g2, g3]

    def forward(self, x):
        """ Calculate model value """
        return self.model(x)

    def loss_value(self, x_data, y_true, y_pred):
        """ Calculate value of the loss function """
        return F.nll_loss(y_pred, y_true)

    def metrics(self):
        """ Set of metrics for this model """
        return [Loss(), Accuracy()]


class AveragingSupervisedMetric(BaseMetric):
    """ Base class for metrics that simply calculate the average over the epoch """

    def __init__(self, name):
        super().__init__(name)
        self.storage = []

    def calculate(self, batch_info):
        """ Calculate value of a metric """
        value = self._value_function(batch_info['data'], batch_info['target'], batch_info['output'])
        self.storage.append(value)

    def _value_function(self, x_input, y_true, y_pred):
        raise NotImplementedError

    def reset(self):
        """ Reset value of a metric """
        self.storage = []

    def value(self):
        """ Return current value for the metric """
        return np.mean(self.storage)


class Accuracy(AveragingSupervisedMetric):
    """ Classification accuracy """

    def __init__(self):
        super().__init__('accuracy')

    def _value_function(self, x_input, y_true, y_pred):
        """ Return classification accuracy of input """
        if len(y_true.shape) == 1:
            return y_pred.argmax(1).eq(y_true).double().mean().item()
        else:
            raise NotImplementedError


class RnnLayer(RnnLinearBackboneModel):
    """ Generalization of RNN layer (Simple RNN, LSTM or GRU) """

    def __init__(self, input_size, hidden_size, rnn_type, bias=True, bidirectional=False, nonlinearity='tanh'):
        super().__init__()
        assert rnn_type in {'rnn', 'lstm', 'gru'}, 'RNN type {} is not supported'.format(rnn_type)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        if self.rnn_type == 'rnn':
            self.rnn_cell = nn.RNN(input_size=input_size, hidden_size=hidden_size, bias=bias, nonlinearity=nonlinearity, bidirectional=bidirectional, batch_first=True)
        elif self.rnn_type == 'lstm':
            self.rnn_cell = nn.LSTM(input_size=input_size, hidden_size=hidden_size, bias=bias, bidirectional=bidirectional, batch_first=True)
        elif self.rnn_type == 'gru':
            self.rnn_cell = nn.GRU(input_size=input_size, hidden_size=hidden_size, bias=bias, bidirectional=bidirectional, batch_first=True)

    def reset_weights(self):
        init.xavier_normal_(self.rnn_cell.weight_hh)
        init.xavier_normal_(self.rnn_cell.weight_ih)
        init.zeros_(self.rnn_cell.bias_ih)
        init.zeros_(self.rnn_cell.bias_hh)

    @property
    def output_dim(self) ->int:
        """ Final dimension of model output """
        if self.bidirectional:
            return 2.0 * self.hidden_size
        else:
            return self.hidden_size

    @property
    def state_dim(self) ->int:
        """ Dimension of model state """
        if self.rnn_type == 'lstm':
            return 2 * self.hidden_size
        else:
            return self.hidden_size

    def forward(self, input_data, state=None):
        if state is None:
            if self.bidirectional:
                state = self.zero_state(input_data.size(0)).unsqueeze(0).repeat(2, 1, 1)
            else:
                state = self.zero_state(input_data.size(0)).unsqueeze(0)
        if self.rnn_type == 'lstm':
            hidden_state, cell_state = torch.split(state, self.hidden_size, 2)
            hidden_state = hidden_state.contiguous()
            cell_state = cell_state.contiguous()
            output, (hidden_state, cell_state) = self.rnn_cell(input_data, (hidden_state, cell_state))
            new_state = torch.cat([hidden_state, cell_state], dim=2)
            return output, new_state
        else:
            return self.rnn_cell(input_data, state)


class MultilayerRnnSequenceClassification(SupervisedModel):
    """ Multilayer GRU network for sequence modeling (n:1) """

    def __init__(self, input_block: LinearBackboneModel, rnn_type: str, output_dim: int, rnn_layers: typing.List[int], rnn_dropout: float=0.0, bidirectional: bool=False, linear_layers: typing.List[int]=None, linear_dropout: float=0.0):
        super().__init__()
        self.output_dim = output_dim
        self.rnn_layers_sizes = rnn_layers
        self.rnn_dropout = rnn_dropout
        self.linear_layers_sizes = linear_layers
        self.linear_dropout = linear_dropout
        self.bidirectional = bidirectional
        self.input_block = input_block
        current_dim = self.input_block.output_dim
        self.rnn_layers = []
        self.rnn_dropout_layers = []
        bidirectional_multiplier = 1
        for idx, current_layer in enumerate(rnn_layers, 1):
            rnn = RnnLayer(input_size=current_dim * bidirectional_multiplier, hidden_size=current_layer, rnn_type=rnn_type, bidirectional=bidirectional)
            self.add_module('{}{:02}'.format(rnn_type, idx), rnn)
            self.rnn_layers.append(rnn)
            if self.rnn_dropout > 0.0:
                dropout_layer = nn.Dropout(p=self.rnn_dropout)
                self.add_module('rnn_dropout{:02}'.format(idx), dropout_layer)
                self.rnn_dropout_layers.append(dropout_layer)
            current_dim = current_layer
            if self.bidirectional:
                bidirectional_multiplier = 2
            else:
                bidirectional_multiplier = 1
        self.linear_layers = []
        self.linear_dropout_layers = []
        for idx, current_layer in enumerate(linear_layers, 1):
            linear_layer = nn.Linear(current_dim * bidirectional_multiplier, current_layer)
            self.add_module('linear{:02}'.format(idx), linear_layer)
            self.linear_layers.append(linear_layer)
            if self.linear_dropout > 0.0:
                dropout_layer = nn.Dropout(p=self.linear_dropout)
                self.add_module('linear_dropout{:02}'.format(idx), dropout_layer)
                self.linear_dropout_layers.append(dropout_layer)
            bidirectional_multiplier = 1
            current_dim = current_layer
        if self.bidirectional:
            self.output_layer = nn.Linear(bidirectional_multiplier * current_dim, output_dim)
        else:
            self.output_layer = nn.Linear(current_dim, output_dim)
        self.output_activation = nn.LogSoftmax(dim=1)

    def reset_weights(self):
        self.input_block.reset_weights()
        for layer in self.linear_layers:
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            nn.init.zeros_(layer.bias)
        nn.init.kaiming_normal_(self.output_layer.weight, nonlinearity='relu')
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, sequence):
        """ Forward propagate batch of sequences through the network, without accounting for the state """
        data = self.input_block(sequence)
        for idx in range(len(self.rnn_layers)):
            data, _ = self.rnn_layers[idx](data)
            if self.rnn_dropout_layers:
                data = self.rnn_dropout_layers[idx](data)
        if self.bidirectional:
            last_hidden_size = self.rnn_layers_sizes[-1]
            data = torch.cat([data[:, (-1), :last_hidden_size], data[:, (0), last_hidden_size:]], dim=1)
        else:
            data = data[:, (-1)]
        for idx in range(len(self.linear_layers_sizes)):
            data = F.relu(self.linear_layers[idx](data))
            if self.linear_dropout_layers:
                data = self.linear_dropout_layers[idx](data)
        data = self.output_layer(data)
        return self.output_activation(data)

    def get_layer_groups(self):
        return [self.input_block, self.rnn_layers, self.linear_layers, self.output_layer]

    @property
    def state_dim(self) ->int:
        """ Dimension of model state """
        return sum(x.state_dim for x in self.gru_layers)

    def loss_value(self, x_data, y_true, y_pred):
        """ Calculate a value of loss function """
        return F.nll_loss(y_pred, y_true)

    def metrics(self) ->list:
        """ Set of metrics for this model """
        return [Loss(), Accuracy()]


class MultilayerRnnSequenceModel(RnnSupervisedModel):
    """ Multilayer GRU network for sequence modeling (n:n) """

    def __init__(self, input_block: LinearBackboneModel, rnn_type: str, hidden_layers: typing.List[int], output_dim: int, dropout: float=0.0):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.input_block = input_block
        current_dim = self.input_block.output_dim
        self.recurrent_layers = []
        self.dropout_layers = []
        for idx, current_layer in enumerate(hidden_layers, 1):
            rnn = RnnLayer(input_size=current_dim, hidden_size=current_layer, rnn_type=rnn_type)
            self.add_module('{}{:02}'.format(rnn_type, idx), rnn)
            self.recurrent_layers.append(rnn)
            if dropout > 0.0:
                dropout_layer = nn.Dropout(p=dropout)
                self.add_module('rnn_dropout{:02}'.format(idx), dropout_layer)
                self.dropout_layers.append(dropout_layer)
            current_dim = current_layer
        self.output_layer = nn.Linear(current_dim, output_dim)
        self.output_activation = nn.LogSoftmax(dim=2)

    def reset_weights(self):
        self.input_block.reset_weights()

    def forward(self, sequence):
        """ Forward propagate batch of sequences through the network, without accounting for the state """
        data = self.input_block(sequence)
        for idx in range(len(self.recurrent_layers)):
            data, _ = self.recurrent_layers[idx](data)
            if self.dropout_layers:
                data = self.dropout_layers[idx](data)
        data = self.output_layer(data)
        return self.output_activation(data)

    def forward_state(self, sequence, state=None):
        """ Forward propagate a sequence through the network accounting for the state """
        if state is None:
            state = self.zero_state(sequence.size(0))
        data = self.input_block(sequence)
        state_outputs = []
        for idx in range(len(self.recurrent_layers)):
            layer_length = self.recurrent_layers[idx].state_dim
            current_state = state[:, :, :layer_length]
            state = state[:, :, layer_length:]
            data, new_h = self.recurrent_layers[idx](data, current_state)
            if self.dropout_layers:
                data = self.dropout_layers[idx](data)
            state_outputs.append(new_h)
        output_data = self.output_activation(self.output_layer(data))
        concatenated_hidden_output = torch.cat(state_outputs, dim=2)
        return output_data, concatenated_hidden_output

    @property
    def state_dim(self) ->int:
        """ Dimension of model state """
        return sum(x.state_dim for x in self.recurrent_layers)

    def zero_state(self, batch_size):
        """ Initial state of the network """
        return torch.zeros(1, batch_size, self.state_dim)

    def loss_value(self, x_data, y_true, y_pred):
        """ Calculate a value of loss function """
        y_pred = y_pred.view(-1, y_pred.size(2))
        y_true = y_true.view(-1)
        return F.nll_loss(y_pred, y_true)


class Net(SupervisedModel):
    """
    A simple MNIST classification model.

    Conv 3x3 - 32
    Conv 3x3 - 64
    MaxPool 2x2
    Dropout 0.25
    Flatten
    Dense - 128
    Dense - output (softmax)
    """

    @staticmethod
    def _weight_initializer(tensor):
        init.xavier_uniform_(tensor.weight, gain=init.calculate_gain('relu'))
        init.constant_(tensor.bias, 0.0)

    def __init__(self, img_rows, img_cols, img_channels, num_classes):
        super(Net, self).__init__()
        self.flattened_size = (img_rows - 4) // 2 * (img_cols - 4) // 2 * 64
        self.conv1 = nn.Conv2d(in_channels=img_channels, out_channels=32, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))
        self.dropout1 = nn.Dropout2d(p=0.25)
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def reset_weights(self):
        self._weight_initializer(self.conv1)
        self._weight_initializer(self.conv2)
        self._weight_initializer(self.fc1)
        self._weight_initializer(self.fc2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=(2, 2))
        x = self.dropout1(x)
        x = x.view(-1, self.flattened_size)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def loss_value(self, x_data, y_true, y_pred):
        """ Calculate a value of loss function """
        return F.nll_loss(y_pred, y_true)

    def metrics(self):
        """ Set of metrics for this model """
        return [Loss(), Accuracy()]


class ResNetV1(SupervisedModel):
    """ A ResNet V1 model as defined in the literature """

    def __init__(self, block, layers, inplanes, divisor=4, img_channels=3, num_classes=1000):
        super().__init__()
        self.num_classess = num_classes
        self.inplanes = inplanes
        self.divisor = divisor
        self.pre_conv = nn.Conv2d(img_channels, inplanes, kernel_size=(3, 3), padding=1, bias=False)
        self.pre_bn = nn.BatchNorm2d(inplanes)
        self.layer1 = self._make_layer(block, inplanes, inplanes, layers[0], stride=1)
        self.layer2 = self._make_layer(block, inplanes, inplanes * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, inplanes * 2, inplanes * 4, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(inplanes * 4, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def _make_layer(self, block, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(in_channels, out_channels, stride, divisor=self.divisor))
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels, divisor=self.divisor))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre_conv(x)
        x = self.pre_bn(x)
        x = F.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x

    def loss_value(self, x_data, y_true, y_pred):
        """ Calculate value of the loss function """
        return F.nll_loss(y_pred, y_true)

    def metrics(self):
        """ Set of metrics for this model """
        return [Loss(), Accuracy()]


class ResNetV2(SupervisedModel):
    """ A ResNet V2 (pre-activation resnet) model as defined in the literature """

    def __init__(self, block, layers, inplanes, divisor=4, img_channels=3, num_classes=1000):
        super().__init__()
        self.num_classess = num_classes
        self.inplanes = inplanes
        self.divisor = divisor
        self.pre_conv = nn.Conv2d(img_channels, inplanes, kernel_size=(3, 3), padding=1, bias=False)
        self.layer1 = self._make_layer(block, inplanes, inplanes, layers[0], stride=1)
        self.layer2 = self._make_layer(block, inplanes, inplanes * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, inplanes * 2, inplanes * 4, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(inplanes * 4, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def _make_layer(self, block, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(in_channels, out_channels, stride, divisor=self.divisor))
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels, divisor=self.divisor))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre_conv(x)
        x = F.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x

    def loss_value(self, x_data, y_true, y_pred):
        """ Calculate value of the loss function """
        return F.nll_loss(y_pred, y_true)

    def metrics(self):
        """ Set of metrics for this model """
        return [Loss(), Accuracy()]

    def summary(self):
        """ Print model summary """
        None


class ResNeXt(SupervisedModel):
    """ A ResNext model as defined in the literature """

    def __init__(self, block, layers, inplanes, image_features, cardinality=4, divisor=4, img_channels=3, num_classes=1000):
        super().__init__()
        self.num_classess = num_classes
        self.inplanes = inplanes
        self.divisor = divisor
        self.cardinality = cardinality
        self.pre_conv = nn.Conv2d(img_channels, image_features, kernel_size=(3, 3), padding=1, bias=False)
        self.pre_bn = nn.BatchNorm2d(image_features)
        self.layer1 = self._make_layer(block, image_features, inplanes, layers[0], stride=1)
        self.layer2 = self._make_layer(block, inplanes, inplanes * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, inplanes * 2, inplanes * 4, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(inplanes * 4, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def _make_layer(self, block, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(in_channels, out_channels, self.cardinality, self.divisor, stride=stride))
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels, self.cardinality, self.divisor, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre_conv(x)
        x = self.pre_bn(x)
        x = F.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x

    def loss_value(self, x_data, y_true, y_pred):
        """ Calculate value of the loss function """
        return F.nll_loss(y_pred, y_true)

    def metrics(self):
        """ Set of metrics for this model """
        return [Loss(), Accuracy()]

    def summary(self):
        """ Print model summary """
        None


class Source:
    """ Source of data for supervised learning algorithms """

    def __init__(self):
        pass

    def train_loader(self):
        """ PyTorch loader of training data """
        raise NotImplementedError

    def val_loader(self):
        """ PyTorch loader of validation data """
        raise NotImplementedError

    def train_dataset(self):
        """ Return the training dataset """
        raise NotImplementedError

    def val_dataset(self):
        """ Return the validation dataset """
        raise NotImplementedError

    def train_iterations_per_epoch(self):
        """ Return number of iterations per epoch """
        raise NotImplementedError

    def val_iterations_per_epoch(self):
        """ Return number of iterations per epoch - validation """
        raise NotImplementedError


class TextData(Source):
    """ An NLP torchtext data source """

    def __init__(self, train_source, val_source, train_iterator, val_iterator, data_field, target_field):
        super().__init__()
        self.train_source = train_source
        self.val_source = val_source
        self.train_iterator = train_iterator
        self.val_iterator = val_iterator
        self.data_field = data_field
        self.target_field = target_field

    def train_loader(self):
        """ PyTorch loader of training data """
        return self.train_iterator

    def val_loader(self):
        """ PyTorch loader of validation data """
        return self.val_iterator

    def train_dataset(self):
        """ Return the training dataset """
        return self.train_source

    def val_dataset(self):
        """ Return the validation dataset """
        return self.val_source

    def train_iterations_per_epoch(self):
        """ Return number of iterations per epoch """
        return len(self.train_iterator)

    def val_iterations_per_epoch(self):
        """ Return number of iterations per epoch - validation """
        return len(self.val_iterator)


class EmbeddingInput(LinearBackboneModel):
    """ Learnable Embedding input layer """

    def __init__(self, alphabet_size: int, output_dim: int, pretrained: bool=False, frozen: bool=False, source: TextData=None):
        super().__init__()
        self._output_dim = output_dim
        self._alphabet_size = alphabet_size
        self._pretrained = pretrained
        self._frozen = frozen
        self._source = source
        self.layer = nn.Embedding(self._alphabet_size, self._output_dim)

    def reset_weights(self):
        if self._pretrained:
            self.layer.weight.data.copy_(self._source.data_field.vocab.vectors)
        if self._frozen:
            self.layer.weight.requires_grad = False

    @property
    def output_dim(self) ->int:
        """ Final dimension of model output """
        return self._output_dim

    def forward(self, input_data):
        return self.layer(input_data)


class ImageToTensor(BackboneModel):
    """
    Convert simple image to tensor.

    Flip channels to a [C, W, H] order and potentially convert 8-bit color values to floats
    """

    def __init__(self):
        super().__init__()

    def reset_weights(self):
        pass

    def forward(self, image):
        result = image.permute(0, 3, 1, 2).contiguous()
        if result.dtype == torch.uint8:
            result = result.type(torch.float) / 255.0
        else:
            result = result.type(torch.float)
        return result


class NormalizeObservations(BackboneModel):
    """ Normalize a vector of observations """

    def __init__(self, input_shape, epsilon=1e-06):
        super().__init__()
        self.input_shape = input_shape
        self.epsilon = epsilon
        self.register_buffer('running_mean', torch.zeros(input_shape, dtype=torch.float))
        self.register_buffer('running_var', torch.ones(input_shape, dtype=torch.float))
        self.register_buffer('count', torch.tensor(epsilon, dtype=torch.float))

    def reset_weights(self):
        self.running_mean.zero_()
        self.running_var.fill_(1.0)
        self.count.fill_(self.epsilon)

    def forward(self, input_vector):
        input_vector = input_vector
        if self.training:
            batch_mean = input_vector.mean(dim=0)
            batch_var = input_vector.var(dim=0, unbiased=False)
            batch_count = input_vector.size(0)
            delta = batch_mean - self.running_mean
            tot_count = self.count + batch_count
            self.running_mean.add_(delta * batch_count / tot_count)
            m_a = self.running_var * self.count
            m_b = batch_var * batch_count
            M2 = m_a + m_b + delta ** 2 * self.count * batch_count / (self.count + batch_count)
            new_var = M2 / (self.count + batch_count)
            self.count.add_(batch_count)
            self.running_var.copy_(new_var)
        return (input_vector - self.running_mean.unsqueeze(0)) / torch.sqrt(self.running_var.unsqueeze(0))


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
    xview = input_tensor.view(-1, 1)
    onehot = torch.zeros(xview.size(0), num_labels, device=input_tensor.device, dtype=torch.float)
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
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    """
    Single residual block consisting of two convolutional layers and a nonlinearity between them
    """

    def __init__(self, in_channels, out_channels, stride=1, divisor=None):
        super().__init__()
        self.stride = stride
        self.divisor = divisor
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
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
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False))
        else:
            self.shortcut = None
        self.bottleneck_channels = out_channels // divisor
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, self.bottleneck_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.bottleneck_channels)
        self.conv2 = nn.Conv2d(self.bottleneck_channels, self.bottleneck_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.bottleneck_channels)
        self.conv3 = nn.Conv2d(self.bottleneck_channels, out_channels, kernel_size=1, bias=False)

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

    def __init__(self, in_channels, out_channels, cardinality, divisor, stride=1):
        super(ResNeXtBottleneck, self).__init__()
        self.cardinality = cardinality
        self.stride = stride
        self.divisor = divisor
        D = out_channels // divisor
        C = cardinality
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None
        self.conv_reduce = nn.Conv2d(in_channels, D * C, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_reduce = nn.BatchNorm2d(D * C)
        self.conv_conv = nn.Conv2d(D * C, D * C, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn = nn.BatchNorm2d(D * C)
        self.conv_expand = nn.Conv2d(D * C, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
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


class RnnCell(RnnLinearBackboneModel):
    """ Generalization of RNN cell (Simple RNN, LSTM or GRU) """

    def __init__(self, input_size, hidden_size, rnn_type, bias=True, nonlinearity='tanh'):
        super().__init__()
        assert rnn_type in {'rnn', 'lstm', 'gru'}, 'Rnn type {} is not supported'.format(rnn_type)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        if self.rnn_type == 'rnn':
            self.rnn_cell = nn.RNNCell(input_size=input_size, hidden_size=hidden_size, bias=bias, nonlinearity=nonlinearity)
        elif self.rnn_type == 'lstm':
            self.rnn_cell = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size, bias=bias)
        elif self.rnn_type == 'gru':
            self.rnn_cell = nn.GRUCell(input_size=input_size, hidden_size=hidden_size, bias=bias)

    def reset_weights(self):
        init.xavier_normal_(self.rnn_cell.weight_hh)
        init.xavier_normal_(self.rnn_cell.weight_ih)
        init.zeros_(self.rnn_cell.bias_ih)
        init.zeros_(self.rnn_cell.bias_hh)

    @property
    def output_dim(self) ->int:
        """ Final dimension of model output """
        return self.hidden_size

    @property
    def state_dim(self) ->int:
        """ Dimension of model state """
        if self.rnn_type == 'lstm':
            return 2 * self.hidden_size
        else:
            return self.hidden_size

    def forward(self, input_data, state):
        if self.rnn_type == 'lstm':
            hidden_state, cell_state = torch.split(state, self.hidden_size, 1)
            hidden_state, cell_state = self.rnn_cell(input_data, (hidden_state, cell_state))
            new_state = torch.cat([hidden_state, cell_state], dim=1)
            return hidden_state, new_state
        else:
            new_hidden_state = self.rnn_cell(input_data, state)
            return new_hidden_state, new_hidden_state


class EvaluatorMeta(type):
    """ Metaclass for Evaluator - gathers all provider methods in a class attribute """

    def __new__(mcs, name, bases, attributes):
        providers = {}
        for name, attr in attributes.items():
            if callable(attr):
                proper_name = getattr(attr, '_vel_evaluator_provides', None)
                if proper_name is not None:
                    providers[proper_name] = attr
        attributes['_providers'] = providers
        return super().__new__(mcs, name, bases, attributes)


class Evaluator(metaclass=EvaluatorMeta):
    """
    Different models may have different outputs and approach evaluating environment differently.

    Evaluator is an object that abstracts over that, providing unified interface between algorithms
    which just need certain outputs from models and models that may provide them in different ways.

    I'll try to maintain here a dictionary of possible common values that can be requested from the evaluator.
    Rollouts should communicate using the same names

    - rollout:estimated_returns
        - Bootstrapped return (sum of discounted future rewards) estimated using returns and value estimates
    - rollout:values
        - Value estimates from the model that was used to generate the rollout
    - rollout:estimated_advantages
        - Advantage of a rollout (state, action) pair by the model that was used to generate the rollout
    - rollout:actions
        - Actions performed in a rollout
    - rollout:logprobs
        - Logarithm of probability for **all** actions of a policy used to perform rollout
        (defined only for finite action spaces)
    - rollout:action:logprobs
        - Logarithm of probability only for selected actions
    - rollout:dones
        - Whether given observation is last in a trajectory
    - rollout:dones
        - Raw rewards received from the environment in this learning process
    - rollout:final_values
        - Value estimates for observation after final observation in the rollout
    - rollout:observations
        - Observations of the rollout
    - rollout:observations_next
        - Next observations in the rollout
    - rollout:weights
        - Error weights of rollout samples
    - rollout:q
        - Action-values for each action in current space
        (defined only for finite action spaces)

    - model:logprobs
        - Logarithm of probability of **all** actions in an environment as in current model policy
        (defined only for finite action spaces)
    - model:q
        - Action-value for **all** actions
        (defined only for finite action spaces)
    - model:q_dist
        - Action-value histogram for **all** actions
        (defined only for finite action spaces)
    - model:q_dist_next
        - Action-value histogram for **all** actions from the 'next' state in the rollout
        (defined only for finite action spaces)
    - model:q_next
        - Action-value for **all** actions from the 'next' state in the rollout
        (defined only for finite action spaces)
    - model:entropy
        - Policy entropy for selected states
    - model:action:q
        - Action-value for actions selected in the rollout
    - model:model_action:q
        - Action-value for actions that model would perform (Deterministic policy only)
    - model:actions
        - Actions that model would perform (Deterministic policy only)
    - model:action:logprobs
        - Logarithm of probability for performed actions
    - model:policy_params
        - Parametrizations of policy for each state
    - model:values
        - Value estimates for each state, estimated by the current model
    - model:values_next
        - Value estimates for 'next' state of each transition
    """

    @staticmethod
    def provides(name):
        """ Function decorator - value provided by the evaluator """

        def decorator(func):
            func._vel_evaluator_provides = name
            return func
        return decorator

    def __init__(self, rollout):
        self._storage = {}
        self.rollout = rollout

    def is_provided(self, name):
        """ Capability check if evaluator provides given value """
        if name in self._storage:
            return True
        elif name in self._providers:
            return True
        elif name.startswith('rollout:'):
            rollout_name = name[8:]
        else:
            return False

    def get(self, name):
        """
        Return a value from this evaluator.

        Because tensor calculated is cached, it may lead to suble bugs if the same value is used multiple times
        with and without no_grad() context.

        It is advised in such cases to not use no_grad and stick to .detach()
        """
        if name in self._storage:
            return self._storage[name]
        elif name in self._providers:
            value = self._storage[name] = self._providers[name](self)
            return value
        elif name.startswith('rollout:'):
            rollout_name = name[8:]
            value = self._storage[name] = self.rollout.batch_tensor(rollout_name)
            return value
        else:
            raise RuntimeError(f'Key {name} is not provided by this evaluator')

    def provide(self, name, value):
        """ Provide given value under specified name """
        self._storage[name] = value


class Rollout:
    """ Base class for environment rollout data """

    def to_transitions(self) ->'Transitions':
        """ Convert given rollout to Transitions """
        raise NotImplementedError

    def episode_information(self):
        """ List of information about finished episodes """
        raise NotImplementedError

    def frames(self) ->int:
        """ Number of frames in rollout """
        raise NotImplementedError

    def batch_tensor(self, name):
        """ A buffer of a given value in a 'flat' (minibatch-indexed) format """
        raise NotImplementedError

    def has_tensor(self, name):
        """ Return true if rollout contains tensor with given name """
        raise NotImplementedError

    def to_device(self, device):
        """ Move a rollout to a selected device """
        raise NotImplementedError


class RlModel(Model):
    """ Reinforcement learning model """

    def step(self, observations) ->dict:
        """
        Evaluate environment on given observations, return actions and potentially some extra information
        in a dictionary.
        """
        raise NotImplementedError

    def evaluate(self, rollout: Rollout) ->Evaluator:
        """ Evaluate model on a rollout """
        raise NotImplementedError


class RlRnnModel(Model):
    """ Reinforcement learning recurrent model """

    @property
    def is_recurrent(self) ->bool:
        """ If the network is recurrent and needs to be fed previous state """
        return True

    def step(self, observations, state) ->dict:
        """
        Evaluate environment on given observations, return actions and potentially some extra information
        in a dictionary.
        """
        raise NotImplementedError

    @property
    def state_dim(self) ->int:
        """ Dimension of model state """
        raise NotImplementedError

    def zero_state(self, batch_size):
        """ Initial state of the network """
        return torch.zeros(batch_size, self.state_dim)

    def evaluate(self, rollout: Rollout) ->Evaluator:
        """ Evaluate model on a rollout """
        raise NotImplementedError


class DoubleNatureCnn(LinearBackboneModel):
    """
    Neural network as defined in the paper 'Human-level control through deep reinforcement learning'
    but with two separate heads.
    """

    def __init__(self, input_width, input_height, input_channels, output_dim=512):
        super().__init__()
        self._output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=(8, 8), stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1)
        self.final_width = net_util.convolutional_layer_series(input_width, [(8, 0, 4), (4, 0, 2), (3, 0, 1)])
        self.final_height = net_util.convolutional_layer_series(input_height, [(8, 0, 4), (4, 0, 2), (3, 0, 1)])
        self.linear_layer_one = nn.Linear(self.final_width * self.final_height * 64, self.output_dim)
        self.linear_layer_two = nn.Linear(self.final_width * self.final_height * 64, self.output_dim)

    @property
    def output_dim(self) ->int:
        """ Final dimension of model output """
        return self._output_dim

    def reset_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight, gain=np.sqrt(2))
                init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                init.orthogonal_(m.weight, gain=np.sqrt(2))
                init.constant_(m.bias, 0.0)

    def forward(self, image):
        result = image
        result = F.relu(self.conv1(result))
        result = F.relu(self.conv2(result))
        result = F.relu(self.conv3(result))
        flattened = result.view(result.size(0), -1)
        output_one = F.relu(self.linear_layer_one(flattened))
        output_two = F.relu(self.linear_layer_two(flattened))
        return output_one, output_two


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
    return torch.randn((in_features, out_features), device=device), torch.randn(out_features, device=device)


class NoisyLinear(nn.Module):
    """ NoisyNets noisy linear layer """

    def __init__(self, in_features, out_features, initial_std_dev: float=0.4, factorized_noise: bool=True):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.initial_std_dev = initial_std_dev
        self.factorized_noise = factorized_noise
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

    def reset_weights(self):
        init.orthogonal_(self.weight_mu, gain=math.sqrt(2))
        init.constant_(self.bias_mu, 0.0)
        self.weight_sigma.data.fill_(self.initial_std_dev / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.initial_std_dev / math.sqrt(self.out_features))

    def forward(self, input_data):
        if self.training:
            if self.factorized_noise:
                weight_epsilon, bias_epsilon = factorized_gaussian_noise(self.in_features, self.out_features, device=input_data.device)
            else:
                weight_epsilon, bias_epsilon = gaussian_noise(self.in_features, self.out_features, device=input_data.device)
            return F.linear(input_data, self.weight_mu + self.weight_sigma * weight_epsilon, self.bias_mu + self.bias_sigma * bias_epsilon)
        else:
            return F.linear(input_data, self.weight_mu, self.bias_mu)

    def extra_repr(self):
        """Set the extra representation of the module

        To print customized extra information, you should reimplement
        this method in your own modules. Both single-line and multi-line
        strings are acceptable.
        """
        return f'{self.in_features}, {self.out_features}, initial_std_dev={self.initial_std_dev}, factorized_noise={{self.factorized_noise}} '


class DoubleNoisyNatureCnn(LinearBackboneModel):
    """
    Neural network as defined in the paper 'Human-level control through deep reinforcement learning'
    but with two separate heads and "noisy" linear layer.
    """

    def __init__(self, input_width, input_height, input_channels, output_dim=512, initial_std_dev=0.4, factorized_noise=True):
        super().__init__()
        self._output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=(8, 8), stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1)
        self.final_width = net_util.convolutional_layer_series(input_width, [(8, 0, 4), (4, 0, 2), (3, 0, 1)])
        self.final_height = net_util.convolutional_layer_series(input_height, [(8, 0, 4), (4, 0, 2), (3, 0, 1)])
        self.linear_layer_one = NoisyLinear(self.final_width * self.final_height * 64, self.output_dim, initial_std_dev=initial_std_dev, factorized_noise=factorized_noise)
        self.linear_layer_two = NoisyLinear(self.final_width * self.final_height * 64, self.output_dim, initial_std_dev=initial_std_dev, factorized_noise=factorized_noise)

    @property
    def output_dim(self) ->int:
        """ Final dimension of model output """
        return self._output_dim

    def reset_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight, gain=np.sqrt(2))
                init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                init.orthogonal_(m.weight, gain=np.sqrt(2))
                init.constant_(m.bias, 0.0)
            elif isinstance(m, NoisyLinear):
                m.reset_weights()

    def forward(self, image):
        result = image
        result = F.relu(self.conv1(result))
        result = F.relu(self.conv2(result))
        result = F.relu(self.conv3(result))
        flattened = result.view(result.size(0), -1)
        output_one = F.relu(self.linear_layer_one(flattened))
        output_two = F.relu(self.linear_layer_two(flattened))
        return output_one, output_two


class MLP(LinearBackboneModel):
    """ Simple Multi-Layer-Perceptron network """

    def __init__(self, input_length: int, hidden_layers: typing.List[int], activation: str='tanh', normalization: typing.Optional[str]=None):
        super().__init__()
        self.input_length = input_length
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.normalization = normalization
        layer_objects = []
        layer_sizes = zip([input_length] + hidden_layers, hidden_layers)
        for input_size, output_size in layer_sizes:
            layer_objects.append(nn.Linear(input_size, output_size))
            if self.normalization:
                layer_objects.append(net_util.normalization(normalization)(output_size))
            layer_objects.append(net_util.activation(activation)())
        self.model = nn.Sequential(*layer_objects)
        self.hidden_units = hidden_layers[-1] if hidden_layers else input_length

    @property
    def output_dim(self) ->int:
        """ Final dimension of model output """
        return self.hidden_units

    def reset_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.orthogonal_(m.weight, gain=np.sqrt(2))
                init.constant_(m.bias, 0.0)

    def forward(self, input_data):
        input_data = input_data.float()
        return self.model(input_data)


class NatureCnn(LinearBackboneModel):
    """ Neural network as defined in the paper 'Human-level control through deep reinforcement learning' """

    def __init__(self, input_width, input_height, input_channels, output_dim=512):
        super().__init__()
        self._output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=(8, 8), stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1)
        layer_series = [(8, 0, 4), (4, 0, 2), (3, 0, 1)]
        self.final_width = net_util.convolutional_layer_series(input_width, layer_series)
        self.final_height = net_util.convolutional_layer_series(input_height, layer_series)
        self.linear_layer = nn.Linear(self.final_width * self.final_height * 64, self.output_dim)

    @property
    def output_dim(self) ->int:
        """ Final dimension of model output """
        return self._output_dim

    def reset_weights(self):
        """ Call proper initializers for the weights """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight, gain=np.sqrt(2))
                init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                init.orthogonal_(m.weight, gain=np.sqrt(2))
                init.constant_(m.bias, 0.0)

    def forward(self, image):
        result = image
        result = F.relu(self.conv1(result))
        result = F.relu(self.conv2(result))
        result = F.relu(self.conv3(result))
        flattened = result.view(result.size(0), -1)
        return F.relu(self.linear_layer(flattened))


class NatureCnnSmall(LinearBackboneModel):
    """
    Neural network as defined in the paper 'Human-level control through deep reinforcement learning'
    Smaller version.
    """

    def __init__(self, input_width, input_height, input_channels, output_dim=128):
        super().__init__()
        self._output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=8, kernel_size=(8, 8), stride=4)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(4, 4), stride=2)
        self.final_width = net_util.convolutional_layer_series(input_width, [(8, 0, 4), (4, 0, 2)])
        self.final_height = net_util.convolutional_layer_series(input_height, [(8, 0, 4), (4, 0, 2)])
        self.linear_layer = nn.Linear(self.final_width * self.final_height * 16, self.output_dim)

    @property
    def output_dim(self) ->int:
        """ Final dimension of model output """
        return self._output_dim

    def reset_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight, gain=np.sqrt(2))
                init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                init.orthogonal_(m.weight, gain=np.sqrt(2))
                init.constant_(m.bias, 0.0)

    def forward(self, image):
        result = image
        result = F.relu(self.conv1(result))
        result = F.relu(self.conv2(result))
        flattened = result.view(result.size(0), -1)
        return F.relu(self.linear_layer(flattened))


class NoisyNatureCnn(LinearBackboneModel):
    """
    Neural network as defined in the paper 'Human-level control through deep reinforcement learning'
    implemented via "Noisy Networks for Exploration"
    """

    def __init__(self, input_width, input_height, input_channels, output_dim=512, initial_std_dev=0.4, factorized_noise=True):
        super().__init__()
        self._output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=(8, 8), stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1)
        layer_series = [(8, 0, 4), (4, 0, 2), (3, 0, 1)]
        self.final_width = net_util.convolutional_layer_series(input_width, layer_series)
        self.final_height = net_util.convolutional_layer_series(input_height, layer_series)
        self.linear_layer = NoisyLinear(self.final_width * self.final_height * 64, self.output_dim, initial_std_dev=initial_std_dev, factorized_noise=factorized_noise)

    @property
    def output_dim(self) ->int:
        """ Final dimension of model output """
        return self._output_dim

    def reset_weights(self):
        """ Call proper initializers for the weights """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight, gain=np.sqrt(2))
                init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                init.orthogonal_(m.weight, gain=np.sqrt(2))
                init.constant_(m.bias, 0.0)
            elif isinstance(m, NoisyLinear):
                m.reset_weights()

    def forward(self, image):
        result = image
        result = F.relu(self.conv1(result))
        result = F.relu(self.conv2(result))
        result = F.relu(self.conv3(result))
        flattened = result.view(result.size(0), -1)
        return F.relu(self.linear_layer(flattened))


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


class DeterministicPolicyEvaluator(Evaluator):
    """ Evaluator for DeterministicPolicyModel """

    def __init__(self, model: 'DeterministicPolicyModel', rollout: Rollout):
        super().__init__(rollout)
        self.model = model

    @Evaluator.provides('model:values_next')
    def model_estimated_values_next(self):
        """ Estimate state-value of the transition next state """
        observations = self.get('rollout:observations_next')
        action, value = self.model(observations)
        return value

    @Evaluator.provides('model:actions')
    def model_actions(self):
        """ Estimate state-value of the transition next state """
        observations = self.get('rollout:observations')
        model_action = self.model.action(observations)
        return model_action

    @Evaluator.provides('model:model_action:q')
    def model_model_action_q(self):
        observations = self.get('rollout:observations')
        model_actions = self.get('model:actions')
        return self.model.value(observations, model_actions)

    @Evaluator.provides('model:action:q')
    def model_action_q(self):
        observations = self.get('rollout:observations')
        rollout_actions = self.get('rollout:actions')
        return self.model.value(observations, rollout_actions)


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
        return (torch.exp(logits_q) * (logits_q - logits_p)).sum(1, keepdim=True)


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
        return -(0.5 * (z_score ** 2 + self.LOG2PI).sum(dim=-1) + log_std.sum(dim=-1))

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
        kl_div = log_std_p - log_std_q + (std_q ** 2 + (means_q - means_p) ** 2) / (2.0 * std_p ** 2) - 0.5
        return kl_div.sum(dim=-1)


class ActionHead(nn.Module):
    """
    Network head for action determination. Returns probability distribution parametrization
    """

    def __init__(self, input_dim, action_space):
        super().__init__()
        self.action_space = action_space
        if isinstance(action_space, spaces.Box):
            assert len(action_space.shape) == 1
            self.head = DiagGaussianActionHead(input_dim, action_space.shape[0])
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


class QStochasticPolicyEvaluator(Evaluator):
    """ Evaluator for QPolicyGradientModel """

    def __init__(self, model: 'QStochasticPolicyModel', rollout: Rollout):
        super().__init__(rollout)
        self.model = model
        observations = self.get('rollout:observations')
        logprobs, q = model(observations)
        self.provide('model:logprobs', logprobs)
        self.provide('model:q', q)

    @Evaluator.provides('model:action:logprobs')
    def model_action_logprobs(self):
        actions = self.get('rollout_actions')
        logprobs = self.get('model:logprobs')
        return self.model.action_head.logprob(actions, logprobs)


class Transitions(Rollout):
    """
    Rollout of random transitions, that don't necessarily have to be in any order

    transition_tensors - tensors that have a row (multidimensional) per each transition. E.g. state, reward, done
    """

    def __init__(self, size, environment_information, transition_tensors, extra_data=None):
        self.size = size
        self.environment_information = environment_information
        self.transition_tensors = transition_tensors
        self.extra_data = extra_data if extra_data is not None else {}

    def to_transitions(self) ->'Transitions':
        """ Convert given rollout to Transitions """
        return self

    def episode_information(self):
        """ List of information about finished episodes """
        return [info.get('episode') for info in self.environment_information if 'episode' in info]

    def frames(self):
        """ Number of frames in this rollout """
        return self.size

    def shuffled_batches(self, batch_size):
        """ Generate randomized batches of data """
        if batch_size >= self.size:
            yield self
        else:
            batch_splits = math_util.divide_ceiling(self.size, batch_size)
            indices = list(range(self.size))
            np.random.shuffle(indices)
            for sub_indices in np.array_split(indices, batch_splits):
                yield Transitions(size=len(sub_indices), environment_information=None, transition_tensors={k: v[sub_indices] for k, v in self.transition_tensors.items()})

    def has_tensor(self, name):
        """ Return true if rollout contains tensor with given name """
        return name in self.transition_tensors

    def batch_tensor(self, name):
        """ A buffer of a given value in a 'flat' (minibatch-indexed) format """
        return self.transition_tensors[name]

    def to_device(self, device, non_blocking=True):
        """ Move a rollout to a selected device """
        return Transitions(size=self.size, environment_information=self.environment_information, transition_tensors={k: v for k, v in self.transition_tensors.items()}, extra_data=self.extra_data)


class Trajectories(Rollout):
    """
    Rollout of trajectories - a number of consecutive transitions

    transition_tensors - tensors that have a row (multidimensional) per each transition. E.g. state, reward, done
    rollout_tensors - tensors that have a row (multidimensional) per whole rollout. E.g. final_value, initial rnn state
    """

    def __init__(self, num_steps, num_envs, environment_information, transition_tensors, rollout_tensors, extra_data=None):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.environment_information = environment_information
        self.transition_tensors = transition_tensors
        self.rollout_tensors = rollout_tensors
        self.extra_data = extra_data if extra_data is not None else {}

    def to_transitions(self) ->'Transitions':
        """ Convert given rollout to Transitions """
        return Transitions(size=self.num_steps * self.num_envs, environment_information=[ei for l in self.environment_information for ei in l] if self.environment_information is not None else None, transition_tensors={name: tensor_util.merge_first_two_dims(t) for name, t in self.transition_tensors.items()}, extra_data=self.extra_data)

    def shuffled_batches(self, batch_size):
        """ Generate randomized batches of data - only sample whole trajectories """
        if batch_size >= self.num_envs * self.num_steps:
            yield self
        else:
            rollouts_in_batch = batch_size // self.num_steps
            batch_splits = math_util.divide_ceiling(self.num_envs, rollouts_in_batch)
            indices = list(range(self.num_envs))
            np.random.shuffle(indices)
            for sub_indices in np.array_split(indices, batch_splits):
                yield Trajectories(num_steps=self.num_steps, num_envs=len(sub_indices), environment_information=None, transition_tensors={k: x[:, (sub_indices)] for k, x in self.transition_tensors.items()}, rollout_tensors={k: x[sub_indices] for k, x in self.rollout_tensors.items()})

    def batch_tensor(self, name):
        """ A buffer of a given value in a 'flat' (minibatch-indexed) format """
        if name in self.transition_tensors:
            return tensor_util.merge_first_two_dims(self.transition_tensors[name])
        else:
            return self.rollout_tensors[name]

    def has_tensor(self, name):
        """ Return true if rollout contains tensor with given name """
        return name in self.transition_tensors or name in self.rollout_tensors

    def flatten_tensor(self, tensor):
        """ Merge first two dims of a tensor """
        return tensor_util.merge_first_two_dims(tensor)

    def episode_information(self):
        """ List of information about finished episodes """
        return [info.get('episode') for infolist in self.environment_information for info in infolist if 'episode' in info]

    def frames(self):
        """ Number of frames in rollout """
        return self.num_steps * self.num_envs

    def to_device(self, device, non_blocking=True):
        """ Move a rollout to a selected device """
        return Trajectories(num_steps=self.num_steps, num_envs=self.num_envs, environment_information=self.environment_information, transition_tensors={k: v for k, v in self.transition_tensors.items()}, rollout_tensors={k: v for k, v in self.rollout_tensors.items()}, extra_data=self.extra_data)


class StochasticPolicyRnnEvaluator(Evaluator):
    """ Evaluate recurrent model from initial state """

    def __init__(self, model: 'StochasticPolicyRnnModel', rollout: Rollout):
        assert isinstance(rollout, Trajectories), 'For an RNN model, we must evaluate trajectories'
        super().__init__(rollout)
        self.model = model
        observation_trajectories = rollout.transition_tensors['observations']
        hidden_state = rollout.rollout_tensors['initial_hidden_state']
        action_accumulator = []
        value_accumulator = []
        for i in range(observation_trajectories.size(0)):
            action_output, value_output, hidden_state = model(observation_trajectories[i], hidden_state)
            action_accumulator.append(action_output)
            value_accumulator.append(value_output)
        policy_params = torch.cat(action_accumulator, dim=0)
        estimated_values = torch.cat(value_accumulator, dim=0)
        self.provide('model:policy_params', policy_params)
        self.provide('model:values', estimated_values)

    @Evaluator.provides('model:action:logprobs')
    def model_action_logprobs(self):
        actions = self.get('rollout:actions')
        policy_params = self.get('model:policy_params')
        return self.model.action_head.logprob(actions, policy_params)

    @Evaluator.provides('model:entropy')
    def model_entropy(self):
        policy_params = self.get('model:policy_params')
        return self.model.entropy(policy_params)


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
        noisy_actions = torch.where(selector > epsilon, actions, random_samples)
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
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


class OuNoise(nn.Module):
    """ Ornstein–Uhlenbeck noise process for action noise """

    def __init__(self, std_dev, environment):
        super().__init__()
        self.std_dev = std_dev
        self.action_space = environment.action_space
        self.processes = []
        self.register_buffer('low_tensor', torch.from_numpy(self.action_space.low).unsqueeze(0))
        self.register_buffer('high_tensor', torch.from_numpy(self.action_space.high).unsqueeze(0))

    def reset_training_state(self, dones, batch_info):
        """ A hook for a model to react when during training episode is finished """
        for idx, done in enumerate(dones):
            if done > 0.5:
                self.processes[idx].reset()

    def forward(self, actions, batch_info):
        """ Return model step after applying noise """
        while len(self.processes) < actions.shape[0]:
            len_action_space = self.action_space.shape[-1]
            self.processes.append(OrnsteinUhlenbeckNoiseProcess(np.zeros(len_action_space), float(self.std_dev) * np.ones(len_action_space)))
        noise = torch.from_numpy(np.stack([x() for x in self.processes])).float()
        return torch.min(torch.max(actions + noise, self.low_tensor), self.high_tensor)


class QDistributionalHead(nn.Module):
    """ Network head calculating Q-function value for each (discrete) action. """

    def __init__(self, input_dim, action_space, vmin: float, vmax: float, atoms: int=1):
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
        self.register_buffer('support_atoms', torch.linspace(self.vmin, self.vmax, self.atoms))

    def histogram_info(self) ->dict:
        """ Return extra information about histogram """
        return {'support_atoms': self.support_atoms, 'atom_delta': self.atom_delta, 'vmin': self.vmin, 'vmax': self.vmax, 'num_atoms': self.atoms}

    def reset_weights(self):
        init.orthogonal_(self.linear_layer.weight, gain=1.0)
        init.constant_(self.linear_layer.bias, 0.0)

    def forward(self, input_data):
        histogram_logits = self.linear_layer(input_data).view(input_data.size(0), self.action_size, self.atoms)
        histogram_log = F.log_softmax(histogram_logits, dim=2)
        return histogram_log

    def sample(self, histogram_logits):
        """ Sample from a greedy strategy with given q-value histogram """
        histogram_probs = histogram_logits.exp()
        atoms = self.support_atoms.view(1, 1, self.atoms)
        return (histogram_probs * atoms).sum(dim=-1).argmax(dim=1)


class QDistributionalNoisyDuelingHead(nn.Module):
    """ Network head calculating Q-function value for each (discrete) action. """

    def __init__(self, input_dim, action_space, vmin: float, vmax: float, atoms: int=1, initial_std_dev: float=0.4, factorized_noise: bool=True):
        super().__init__()
        assert isinstance(action_space, spaces.Discrete)
        assert vmax > vmin
        self.atoms = atoms
        self.vmin = vmin
        self.vmax = vmax
        self.action_size = action_space.n
        self.action_space = action_space
        self.atom_delta = (self.vmax - self.vmin) / (self.atoms - 1)
        self.linear_layer_advantage = NoisyLinear(input_dim, self.action_size * self.atoms, initial_std_dev=initial_std_dev, factorized_noise=factorized_noise)
        self.linear_layer_value = NoisyLinear(input_dim, self.atoms, initial_std_dev=initial_std_dev, factorized_noise=factorized_noise)
        self.register_buffer('support_atoms', torch.linspace(self.vmin, self.vmax, self.atoms))

    def histogram_info(self) ->dict:
        """ Return extra information about histogram """
        return {'support_atoms': self.support_atoms, 'atom_delta': self.atom_delta, 'vmin': self.vmin, 'vmax': self.vmax, 'num_atoms': self.atoms}

    def reset_weights(self):
        self.linear_layer_advantage.reset_weights()
        self.linear_layer_value.reset_weights()

    def forward(self, advantage_features, value_features):
        adv = self.linear_layer_advantage(advantage_features).view(-1, self.action_size, self.atoms)
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


class QNoisyHead(nn.Module):
    """ Network head calculating Q-function value for each (discrete) action. """

    def __init__(self, input_dim, action_space, initial_std_dev=0.4, factorized_noise=True):
        super().__init__()
        assert isinstance(action_space, spaces.Discrete)
        self.action_space = action_space
        self.linear_layer = NoisyLinear(input_dim, action_space.n, initial_std_dev=initial_std_dev, factorized_noise=factorized_noise)

    def reset_weights(self):
        self.linear_layer.reset_weights()

    def forward(self, input_data):
        return self.linear_layer(input_data)

    def sample(self, q_values):
        """ Sample from epsilon-greedy strategy with given q-values """
        return q_values.argmax(dim=1)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AdaptiveConcatPool2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BasicBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Bottleneck,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (CategoricalActionHead,
     lambda: ([], {'input_dim': 4, 'num_actions': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DeterministicCriticHead,
     lambda: ([], {'input_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DiagGaussianActionHead,
     lambda: ([], {'input_dim': 4, 'num_dimensions': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (EmbeddingInput,
     lambda: ([], {'alphabet_size': 4, 'output_dim': 4}),
     lambda: ([torch.zeros([4], dtype=torch.int64)], {}),
     True),
    (Flatten,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ImageToTensor,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Lambda,
     lambda: ([], {'f': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (NoisyLinear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (NormalizeObservations,
     lambda: ([], {'input_shape': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (OneHotEncode,
     lambda: ([], {'num_classes': 4}),
     lambda: ([torch.zeros([4], dtype=torch.int64)], {}),
     False),
    (ResNeXtBottleneck,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'cardinality': 4, 'divisor': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Resnet34,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (RnnCell,
     lambda: ([], {'input_size': 4, 'hidden_size': 4, 'rnn_type': 'rnn'}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (RnnLayer,
     lambda: ([], {'input_size': 4, 'hidden_size': 4, 'rnn_type': 'rnn'}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (ValueHead,
     lambda: ([], {'input_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_MillionIntegrals_vel(_paritybench_base):
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

    def test_015(self):
        self._check(*TESTCASES[15])

    def test_016(self):
        self._check(*TESTCASES[16])

    def test_017(self):
        self._check(*TESTCASES[17])

