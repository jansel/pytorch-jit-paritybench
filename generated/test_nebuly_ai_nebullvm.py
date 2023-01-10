import sys
_module = sys.modules[__name__]
del sys
forward_forward = _module
api = _module
functions = _module
app = _module
operations = _module
build_models = _module
data = _module
fetch_operations = _module
trainers = _module
root_op = _module
utils = _module
labels = _module
modules = _module
utils = _module
setup = _module
speedster = _module
functions = _module
tests = _module
test_huggingface = _module
test_onnx = _module
test_pytorch = _module
test_tensorflow = _module
root_op = _module
conf = _module
nebullvm = _module
functions = _module
apps = _module
base = _module
config = _module
installers = _module
auto_installer = _module
installers = _module
conversions = _module
converters = _module
huggingface = _module
pytorch = _module
tensorflow = _module
local = _module
inference_learners = _module
base = _module
blade_disc = _module
builders = _module
deepsparse = _module
neural_compressor = _module
onnx = _module
openvino = _module
pytorch = _module
tensor_rt = _module
tvm = _module
measures = _module
utils = _module
optimizations = _module
base = _module
compilers = _module
deepsparse = _module
intel_neural_compressor = _module
onnxruntime = _module
pytorch = _module
quantizations = _module
intel_neural_compressor = _module
onnx = _module
pytorch = _module
tensor_rt = _module
tensorflow = _module
tvm = _module
compressors = _module
intel = _module
scripts = _module
neural_magic_training = _module
sparseml = _module
optimizers = _module
test_deepsparse = _module
test_intel_neural_compressor = _module
test_onnxruntime = _module
test_openvino = _module
test_tensor_rt = _module
test_torchscript = _module
test_tvm = _module
utils = _module
optional_modules = _module
onnxruntime = _module
onnxsim = _module
torch_tensorrt = _module
utils = _module
tools = _module
base = _module
benchmark = _module
data = _module
feedback_collector = _module
huggingface = _module
logger = _module
onnx = _module
pytorch = _module
test_data = _module
tf = _module
transformations = _module
utils = _module
venv = _module

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


from abc import ABC


from abc import abstractmethod


import torch


from typing import Any


import torch.utils.data


from torchvision import datasets


from torchvision import transforms


from torch.utils.data import DataLoader


from typing import List


from collections import Generator


import logging


from typing import Union


from typing import Iterable


from typing import Sequence


from typing import Callable


from typing import Dict


from typing import Optional


import numpy as np


from torchvision import models


import torchvision.models as models


import tensorflow as tf


import abc


from typing import Generator


from typing import Tuple


from typing import Type


import time


import copy


import uuid


import re


from torch.nn import CrossEntropyLoss


from torch.nn import MSELoss


from torch.optim import SGD


import tensorflow.keras as keras


from tensorflow.keras import Model


from tensorflow.keras import layers


from enum import Enum


from collections import OrderedDict


from types import ModuleType


class BaseFFLayer(torch.nn.Module, ABC):

    @abstractmethod
    def ff_train(self, input_tensor: torch.Tensor, signs: torch.Tensor, theta: float):
        raise NotImplementedError

    @abstractmethod
    def positive_eval(self, input_tensor: torch.Tensor, theta: float):
        raise NotImplementedError

    @property
    def requires_training(self):
        return True


def alternative_loss_fn(y, theta, sign):
    logits = y.pow(2).mean(dim=1) - theta
    with torch.no_grad():
        accumulated_logits = logits.mean().item()
    logits = -logits * sign
    prob = torch.nan_to_num(torch.exp(logits))
    loss = torch.log(1 + prob)
    loss = loss.mean()
    return loss, accumulated_logits


def loss_fn(y, theta, sign):
    logits = torch.square(y).mean(dim=1) - theta
    loss = -logits * sign
    with torch.no_grad():
        accumulated_logits = logits.mean().item()
    loss = loss.mean()
    return loss, accumulated_logits


def probabilistic_loss_fn(y, theta, sign):
    logits = torch.square(y).mean(dim=1) - theta
    prob = torch.sigmoid(logits)
    loss = -torch.log(prob + 1e-06) * sign
    with torch.no_grad():
        accumulated_logits = logits.mean().item()
    loss = loss.mean()
    return loss, accumulated_logits


class FFLayer(BaseFFLayer):
    """Layer wrapper for efficient forward-forward layers."""

    def __init__(self, layer, optimizer_name: str, optimizer_kwargs: dict, loss_fn_name: str='loss_fn'):
        super().__init__()
        self.layer = layer
        self.optimizer = getattr(torch.optim, optimizer_name)(layer.parameters(), **optimizer_kwargs)
        if loss_fn_name == 'loss_fn':
            self.loss_fn = loss_fn
        elif loss_fn_name == 'alternative_loss_fn':
            self.loss_fn = alternative_loss_fn
        elif loss_fn_name == 'probabilistic_loss_fn':
            self.loss_fn = probabilistic_loss_fn

    def forward(self, x):
        return self.layer(x)

    def ff_train(self, input_tensor: torch.Tensor, signs: torch.Tensor, theta: float):
        """Train the layer with the given target."""
        y = self(input_tensor.detach())
        y_pos = y[torch.where(signs == 1)]
        y_neg = y[torch.where(signs == -1)]
        loss_pos, cumulated_logits_pos = self.loss_fn(y_pos, theta, sign=1)
        loss_neg, cumulated_logits_neg = self.loss_fn(y_neg, theta, sign=-1)
        self.optimizer.zero_grad()
        loss = loss_pos + loss_neg
        loss.backward()
        self.optimizer.step()
        separation = [cumulated_logits_pos, cumulated_logits_neg]
        y = torch.zeros(input_tensor.shape[0], *y_pos.shape[1:], device=input_tensor.device)
        y[torch.where(signs == 1)] = y_pos
        y[torch.where(signs == -1)] = y_neg
        return y.detach(), separation

    @torch.no_grad()
    def positive_eval(self, input_tensor: torch.Tensor, theta: float):
        """Evaluate the layer with the given input and theta."""
        y = self(input_tensor)
        return y, torch.square(y).mean(dim=1) - theta


class FFNormalization(BaseFFLayer):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        l2_norm = torch.norm(x.reshape(x.shape[0], -1), p=2, dim=1, keepdim=True) + 1e-08
        return x / l2_norm

    def ff_train(self, input_tensor: torch.Tensor, signs: torch.Tensor, theta: float):
        with torch.no_grad():
            output = self()
        return output, None

    @torch.no_grad()
    def positive_eval(self, input_tensor: torch.Tensor, theta: float):
        with torch.no_grad():
            output = self(input_tensor)
        return output, torch.zeros(input_tensor.shape[0], device=input_tensor.device)

    @property
    def requires_training(self):
        return False


class LinearReLU(torch.nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(self.linear(x))


class ProgressiveTrainingDataset(torch.utils.data.Dataset):
    """Dataset for progressive training."""

    def __init__(self, dataset_generator: Generator):
        with torch.no_grad():
            self.internal_dataset = [batch for data, sign in dataset_generator for batch in zip(data, sign)]

    def __getitem__(self, index):
        return self.internal_dataset[index]

    def __len__(self):
        return len(self.internal_dataset)


class FCNetFFProgressive(BaseFFLayer):
    """FCNet trained using forward-forward algorithm. The network is trained
    in a progressive manner, i.e. the first layer is trained, then the
    second layer, and so on.
    """

    def __init__(self, layer_sizes: list, optimizer_name: str, optimizer_kwargs: dict, epochs: int, loss_fn_name: str='loss_fn'):
        super().__init__()
        self.epochs = epochs
        self.layers = torch.nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(FFNormalization())
            self.layers.append(FFLayer(LinearReLU(layer_sizes[i], layer_sizes[i + 1]), optimizer_name, optimizer_kwargs, loss_fn_name))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def progressive_train(self, dl: torch.utils.data.DataLoader, theta: float):
        """Train the network in a progressive manner."""
        None
        for i, layer in enumerate(self.layers):
            if layer.requires_training:
                for epoch in range(self.epochs):
                    accumulated_separation = None
                    for j, (data, signs) in enumerate(dl):
                        data = data
                        signs = signs
                        _, separation = layer.ff_train(data, signs, theta)
                        if accumulated_separation is None:
                            accumulated_separation = separation
                        else:
                            accumulated_separation[0] += separation[0]
                            accumulated_separation[1] += separation[1]
                        if j % 100 == 0:
                            None
                    None
                    accumulated_separation[0] /= len(dl.dataset)
                    accumulated_separation[1] /= len(dl.dataset)
                    separation_ratio = (accumulated_separation[0] - accumulated_separation[1]) / abs(max(accumulated_separation))
                    None
                    None
                None
            dataset = ProgressiveTrainingDataset((layer(x), sign) for x, sign in dl)
            batch_size = dl.batch_size
            dl = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        None

    def ff_train(self, input_tensor: torch.Tensor, signs: torch.Tensor, theta: float):
        """Train the network with the given target."""
        accumulated_separation = None
        for layer in self.layers:
            input_tensor, separation = layer.ff_train(input_tensor, signs, theta)
            if accumulated_separation is None:
                accumulated_separation = separation
            else:
                accumulated_separation[0] += separation[0]
                accumulated_separation[1] += separation[1]
        return input_tensor, accumulated_separation

    @torch.no_grad()
    def positive_eval(self, input_tensor: torch.Tensor, theta: float):
        """Evaluate the network with the given input and theta."""
        accumulated_goodness = torch.zeros(input_tensor.shape[0], device=input_tensor.device)
        for i, layer in enumerate(self.layers):
            input_tensor, goodness = layer.positive_eval(input_tensor, theta)
            if i > 1:
                accumulated_goodness += goodness
        return input_tensor, accumulated_goodness

    @property
    def device(self):
        return next(self.parameters()).device


class NormLinearReLU(torch.nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        self.norm = FFNormalization()
        self.linear_relu = LinearReLU(in_features, out_features)

    def forward(self, x):
        return self.linear_relu(self.norm(x))


class RecurrentFFLayer(BaseFFLayer):

    def __init__(self, hidden_size: int, optimizer_name: str, optimizer_kwargs: dict, loss_fn_name: str):
        super().__init__()
        self.layer = NormLinearReLU(2 * hidden_size, hidden_size)
        self.optimizer = getattr(torch.optim, optimizer_name)(self.layer.parameters(), **optimizer_kwargs)
        self.loss_fn = eval(loss_fn_name)

    def forward(self, x_prev, x_same, x_next):
        x = torch.cat((x_prev, x_next), dim=1)
        new_x = self.layer(x)
        new_x = 0.3 * x_same + 0.7 * new_x
        return new_x

    def ff_train(self, x_prev: torch.Tensor, x_same: torch.Tensor, x_next: torch.Tensor, signs: torch.Tensor, theta: float):
        new_x = self(x_prev.detach(), x_same.detach(), x_next.detach())
        y_pos = new_x[signs == 1]
        y_neg = new_x[signs == -1]
        loss_pos, goodness_pos = self.loss_fn(y_pos, theta, 1)
        loss_neg, goodness_neg = self.loss_fn(y_neg, theta, -1)
        loss = loss_pos + loss_neg
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return new_x, [goodness_pos, goodness_neg]

    @torch.no_grad()
    def positive_eval(self, x_prev: torch.Tensor, x_same: torch.Tensor, x_next: torch.Tensor, theta: float):
        new_x = self(x_prev, x_same, x_next)
        goodness = new_x.pow(2).mean(dim=1) - theta
        return new_x, goodness


class RecurrentProjectionFFLayer(BaseFFLayer):

    def __init__(self, input_size: int, output_size: int, optimizer_name: str, optimizer_kwargs: dict, loss_fn_name: str):
        super().__init__()
        self.layer = NormLinearReLU(input_size, output_size)
        self.optimizer = getattr(torch.optim, optimizer_name)(self.layer.parameters(), **optimizer_kwargs)
        self.loss_fn = eval(loss_fn_name)

    def forward(self, x: torch.Tensor):
        return self.layer(x)

    def ff_train(self, x: torch.Tensor, signs: torch.Tensor, theta: float):
        new_x = self(x.detach())
        y_pos = new_x[signs == 1]
        y_neg = new_x[signs == -1]
        loss_pos, goodness_pos = self.loss_fn(y_pos, theta, 1)
        loss_neg, goodness_neg = self.loss_fn(y_neg, theta, -1)
        loss = loss_pos + loss_neg
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return new_x, [goodness_pos, goodness_neg]

    @torch.no_grad()
    def positive_eval(self, x: torch.Tensor, theta: float):
        new_x = self(x)
        goodness = new_x.pow(2).mean(dim=1) - theta
        return new_x, goodness


class RecurrentProjectedSoftmaxFFLayer(BaseFFLayer):

    def __init__(self, input_size: int, output_size: int, optimizer_name: str, optimizer_kwargs: dict, loss_fn_name: str):
        super().__init__()
        self.loss_fn = eval(loss_fn_name)
        self.norm = FFNormalization()
        self.linear = torch.nn.Linear(input_size, output_size)
        self.softmax = torch.nn.Softmax(dim=1)
        self.optimizer = getattr(torch.optim, optimizer_name)(self.linear.parameters(), **optimizer_kwargs)

    def forward(self, x: torch.Tensor):
        x = self.norm(x)
        x = self.linear(x)
        x = self.softmax(x)
        return x

    def ff_train(self, x: torch.Tensor, signs: torch.Tensor, theta: float):
        new_x = self(x.detach())
        y_pos = new_x[signs == 1]
        y_neg = new_x[signs == -1]
        loss_pos, goodness_pos = self.loss_fn(y_pos, theta, 1)
        loss_neg, goodness_neg = self.loss_fn(y_neg, theta, -1)
        loss = loss_pos + loss_neg
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return new_x, [goodness_pos, goodness_neg]

    @torch.no_grad()
    def positive_eval(self, x: torch.Tensor, theta: float):
        new_x = self(x)
        goodness = new_x.pow(2).mean(dim=1) - theta
        return new_x, goodness


class RecurrentFCNetFF(BaseFFLayer):
    """Recurrent FCNet trained using forward-forward algorithm."""

    def __init__(self, layer_sizes: list, optimizer_name: str, optimizer_kwargs: dict, loss_fn_name: str='loss_fn'):
        super().__init__()
        self.time_steps = 8
        self.test_time_steps = 8
        self.storable_time_steps = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.states = []
        self.layers = torch.nn.ModuleList()
        self.projector = RecurrentProjectionFFLayer(layer_sizes[0], layer_sizes[1], optimizer_name, optimizer_kwargs, loss_fn_name)
        for i in range(1, len(layer_sizes) - 1):
            self.layers.append(RecurrentFFLayer(layer_sizes[i], optimizer_name, optimizer_kwargs, loss_fn_name))
        self.proj_y = RecurrentProjectionFFLayer(layer_sizes[-1], layer_sizes[-2], optimizer_name, optimizer_kwargs, loss_fn_name)
        self.softmax = RecurrentProjectedSoftmaxFFLayer(layer_sizes[-2], layer_sizes[-1], optimizer_name, optimizer_kwargs, loss_fn_name)
        self.num_labels = layer_sizes[-1]

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def bottom_up(self, x: torch.Tensor, y: torch.Tensor):
        states = []
        x_proj = self.projector(x)
        for layer in self.layers:
            states.append(x_proj)
            x_proj = layer(x_proj, torch.zeros_like(x_proj, device=self.device), torch.zeros_like(x_proj, device=self.device))
        states.append(x_proj)
        states.append(y)
        y_arg = torch.argmax(y, dim=1)
        x_proj_ = x_proj.clone()
        x_proj_[torch.arange(x_proj.shape[0]), y_arg] = -1000000.0
        neg_prob = self.softmax(x_proj_)
        cumulative_neg_prob = torch.cumsum(neg_prob, dim=1)
        neg_samples = torch.argmax(1.0 * (cumulative_neg_prob > torch.rand(x.shape[0], 1)), dim=1)
        neg_samples = torch.functional.F.one_hot(neg_samples, num_classes=self.num_labels)
        return states, neg_samples

    def forward(self, x: torch.Tensor, prev_states: List[torch.Tensor]):
        x_proj = self.projector(x)
        new_states = []
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                next_state = prev_states[i + 2]
            else:
                next_state = self.proj_y(prev_states[i + 2].float())
            new_states.append(x_proj)
            x_proj = layer(prev_states[i], prev_states[i + 1], next_state)
        new_states.append(x_proj)
        y = self.softmax(x_proj)
        new_states.append(y)
        return new_states

    def ff_train(self, input_tensor: torch.Tensor, labels: torch.Tensor, theta: float):
        """Train the network with the given target."""
        with torch.no_grad():
            states, neg_samples = self.bottom_up(input_tensor, labels)
            neg_states, _ = self.bottom_up(input_tensor, neg_samples)
            states = [torch.cat([s, ns], dim=0) for s, ns in zip(states, neg_states)]
            signs = torch.cat([torch.ones(input_tensor.shape[0], device=self.device), -torch.ones(input_tensor.shape[0], device=self.device)], dim=0)
            input_tensor = torch.cat([input_tensor, input_tensor], dim=0)
        x_proj, accumulated_goodness = self.projector.ff_train(input_tensor, signs, theta)
        for _ in range(self.time_steps):
            new_states = []
            x = x_proj
            for j, layer in enumerate(self.layers):
                if j < len(self.layers) - 1:
                    next_state = states[j + 2]
                else:
                    next_state = self.proj_y(states[j + 2].float())
                new_states.append(x)
                x, goodnesses = layer.ff_train(states[j], states[j + 1], next_state, signs, theta)
                accumulated_goodness[0] += goodnesses[0]
                accumulated_goodness[1] += goodnesses[1]
            new_states.append(x)
            with torch.no_grad():
                x_ = states[-2][torch.where(signs == -1)]
                real_y = states[-1][torch.where(signs == 1)]
                x_[torch.arange(x_.shape[0]), torch.argmax(real_y, dim=1)] = -1000000.0
                y = self.softmax(x_)
                cumulative_y = torch.cumsum(y, dim=1)
                neg_samples = torch.argmax(1.0 * (cumulative_y > torch.rand(x_.shape[0], 1)), dim=1)
                neg_samples = torch.functional.F.one_hot(neg_samples, num_classes=self.num_labels)
                next_labels = states[-1].clone()
                next_labels[torch.where(signs == -1)] = neg_samples
                new_states.append(next_labels)
            states = new_states
        accumulated_goodness[0] /= self.time_steps * len(self.layers) + 1
        accumulated_goodness[1] /= self.time_steps * len(self.layers) + 1
        with torch.no_grad():
            states = [t[:input_tensor.shape[0] // 2] for t in states]
        return states, accumulated_goodness

    @torch.no_grad()
    def positive_eval(self, input_tensor: torch.Tensor, theta: float):
        """Evaluate the network with the given input and theta."""
        labels = torch.arange(0, self.num_labels, device=self.device)
        labels = torch.functional.F.one_hot(labels, num_classes=self.num_labels)
        original_bs = input_tensor.shape[0]
        input_tensor = input_tensor.unsqueeze(1).repeat(1, self.num_labels, 1).reshape(-1, input_tensor.shape[-1])
        labels = labels.unsqueeze(0).repeat(original_bs, 1, 1).reshape(-1, labels.shape[-1])
        states, _ = self.bottom_up(input_tensor, labels)
        x_proj, goodness = self.projector.positive_eval(input_tensor, theta)
        accumulated_goodness = goodness
        for time_step in range(self.test_time_steps):
            new_states = []
            x = x_proj
            for j, layer in enumerate(self.layers):
                if j < len(self.layers) - 1:
                    next_state = states[j + 2]
                else:
                    next_state = self.proj_y(states[j + 2].float())
                new_states.append(x)
                x, goodnesses = layer.positive_eval(states[j], states[j + 1], next_state, theta)
                if time_step in self.storable_time_steps:
                    accumulated_goodness += goodnesses
            new_states.append(x)
            if time_step in self.storable_time_steps:
                _, goodness = self.softmax.positive_eval(x, theta)
                accumulated_goodness += goodness
            new_states.append(states[-1])
            states = new_states
        accumulated_goodness = accumulated_goodness.reshape(original_bs, self.num_labels)
        prediction = torch.argmax(accumulated_goodness, dim=1)
        return prediction, accumulated_goodness


class LMFFLinearSoftmax(BaseFFLayer):

    def __init__(self, input_size: int, output_size: int, optimizer_name: str, optimizer_kwargs: dict):
        super().__init__()
        self.loss_fn = torch.nn.NLLLoss()
        self.norm = FFNormalization()
        self.linear = torch.nn.Linear(input_size, output_size)
        self.softmax = torch.nn.Softmax(dim=1)
        self.optimizer = getattr(torch.optim, optimizer_name)(self.parameters(), **optimizer_kwargs)

    def forward(self, x: torch.Tensor):
        x = self.norm(x)
        x = self.linear(x)
        x = self.softmax(x)
        return x

    def ff_train(self, input_tensor: torch.Tensor, labels: torch.Tensor, signs: torch.Tensor):
        x = input_tensor[torch.where(signs == 1)]
        y = labels[torch.where(signs == 1)]
        x = self(x)
        loss = self.loss_fn(x, torch.argmax(y, dim=1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        with torch.no_grad():
            x_neg = input_tensor[torch.where(signs == -1)]
            new_y_neg = self(x_neg)
            new_x = torch.zeros(len(input_tensor), *x.shape[1:], device=input_tensor.device)
            new_x[torch.where(signs == 1)] = x
            new_x[torch.where(signs == -1)] = new_y_neg
        return new_x, loss.item()

    @torch.no_grad()
    def positive_eval(self, x: torch.Tensor):
        pred = self(x)
        return pred


class LMFFNet(BaseFFLayer):

    def __init__(self, token_num: int, hidden_size: int, n_layers: int, seq_len: int, predicted_tokens: int, epochs: int, optimizer_name: str, optimizer_kwargs: dict, loss_fn_name: str='loss_fn'):
        super().__init__()
        self.token_num = token_num
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.predicted_tokens = predicted_tokens
        self.token2emb = RecurrentProjectionFFLayer(token_num * seq_len, hidden_size, optimizer_name, optimizer_kwargs, loss_fn_name)
        self.layers = torch.nn.ModuleList([FFLayer(NormLinearReLU(hidden_size, hidden_size), optimizer_name, optimizer_kwargs, loss_fn_name) for _ in range(n_layers)])
        self.emb2token = LMFFLinearSoftmax(n_layers * hidden_size, token_num, optimizer_name, optimizer_kwargs)
        self.epochs = epochs

    def forward(self, input_tensor: torch.Tensor):
        x = self.token2emb(input_tensor)
        xs = []
        for layer in self.layers:
            x = layer(x)
            xs.append(x)
        x = torch.cat(xs, dim=1)
        x = self.emb2token(x)
        return x

    def ff_train(self, input_tensor: torch.Tensor, prev_pred: torch.Tensor, labels: torch.Tensor, theta: float):
        signs = torch.cat([torch.ones(input_tensor.shape[0], device=input_tensor.device), -torch.ones(input_tensor.shape[0], device=input_tensor.device)])
        input_tensor = torch.cat([input_tensor, prev_pred], dim=0)
        labels = torch.cat([labels, labels], dim=0)
        for idx in range(self.epochs):
            x, goodness = self.token2emb.ff_train(input_tensor, signs, theta)
            if idx % 20 == 0:
                None
        accumulated_goodness = goodness
        xs = []
        for layer in self.layers:
            for epoch in range(self.epochs):
                x_new, goodness = layer.ff_train(x, signs, theta)
                if epoch % 20 == 0:
                    None
            x = x_new
            xs.append(x)
            accumulated_goodness[0] += goodness[0]
            accumulated_goodness[1] += goodness[1]
        x = torch.cat(xs, dim=1)
        for epoch in range(self.epochs):
            x_new, loss = self.emb2token.ff_train(x, labels, signs)
            if epoch % 20 == 0 or epoch < 20:
                None
        x = x_new
        next_input = input_tensor[signs == 1].roll(-self.token_num, dims=1)
        next_input[:, -self.token_num:] = torch.functional.F.one_hot(torch.argmax(x[signs == 1], dim=1), num_classes=self.token_num)
        return next_input, accumulated_goodness

    def LM_ff_train(self, input_tensor: torch.Tensor, theta: float):
        with torch.no_grad():
            input_tensor = input_tensor.reshape(-1, self.token_num * self.seq_len)
            labels = input_tensor[:, -self.token_num:].roll(-1, dims=0)
            temp = torch.argmax(labels, dim=1)
            None
            pred = self(input_tensor)
            new_char = torch.functional.F.one_hot(torch.argmax(pred, dim=1), num_classes=self.token_num)
            prev_pred = input_tensor.clone().roll(1)
            prev_pred[:, -self.token_num:] = new_char
        _, accumulated_goodness = self.ff_train(input_tensor, prev_pred, labels, theta)
        return accumulated_goodness

    @torch.no_grad()
    def positive_eval(self, input_tensor: torch.Tensor, theta: float):
        cumulated_goodness = torch.zeros(input_tensor.shape[0], device=input_tensor.device)
        prediction = torch.zeros(input_tensor.shape[0], self.predicted_tokens, self.token_num, device=input_tensor.device)
        for idx in range(self.predicted_tokens):
            x, goodness = self.token2emb.positive_eval(input_tensor, theta)
            cumulated_goodness += goodness
            xs = []
            for layer in self.layers:
                x, goodness = layer.positive_eval(x, theta)
                xs.append(x)
                cumulated_goodness += goodness
            x = torch.cat(xs, dim=1)
            x = self.emb2token.positive_eval(x)
            prediction[:, idx] = x
            input_tensor = input_tensor.roll(-self.token_num, dims=1)
            input_tensor[:, -self.token_num:] = torch.functional.F.one_hot(torch.argmax(x, dim=1), num_classes=self.token_num)
        cumulated_goodness /= self.predicted_tokens
        return prediction, cumulated_goodness


class _QuantWrapper(Module):

    def __init__(self, model: Module):
        super(_QuantWrapper, self).__init__()
        qconfig = model.qconfig if hasattr(model, 'qconfig') else None
        self.quant = QuantStub(qconfig)
        self.model = model
        self.dequant = DeQuantStub()

    def forward(self, *inputs: torch.Tensor):
        inputs = (self.quant(x) for x in inputs)
        outputs = self.model(*inputs)
        return tuple(self.dequant(x) for x in outputs)


class TestModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3)
        self.relu2 = torch.nn.ReLU()
        self.fcn = torch.nn.Linear(32, 2)

    def forward(self, input_tensor_0, input_tensor_1):
        x0 = self.relu2(self.conv2(self.relu1(self.conv1(input_tensor_0))))
        x1 = self.relu2(self.conv2(self.relu1(self.conv1(input_tensor_1))))
        x = x0 + x1
        x = self.fcn(x.mean(dim=(-2, -1)).view(-1, 32))
        return x


def flatten_outputs(outputs: Union[torch.Tensor, Iterable]) ->List[torch.Tensor]:
    new_outputs = []
    for output in outputs:
        if isinstance(output, torch.Tensor):
            new_outputs.append(output)
        else:
            flatten_list = flatten_outputs(output)
            new_outputs.extend(flatten_list)
    return new_outputs


class TransformerWrapper(Module):
    """Class for wrappering the Transformers and give them an API compatible
    with nebullvm. The class takes and input of the forward method positional
    arguments and transform them in the input dictionaries needed by
    transformers classes. At the end it also flattens their output.
    """

    def __init__(self, core_model: Module, encoded_input: Dict[str, torch.Tensor]):
        super().__init__()
        self.core_model = core_model
        self.inputs_types = OrderedDict()
        for key, value in encoded_input.items():
            self.inputs_types[key] = value.dtype

    def forward(self, *args: torch.Tensor):
        inputs = {key: value for key, value in zip(self.inputs_types.keys(), args)}
        outputs = self.core_model(**inputs)
        outputs = outputs.values() if isinstance(outputs, dict) else outputs
        return tuple(flatten_outputs(outputs))


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (FFNormalization,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LinearReLU,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (NormLinearReLU,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TestModel,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64]), torch.rand([4, 3, 64, 64])], {}),
     True),
]

class Test_nebuly_ai_nebullvm(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

