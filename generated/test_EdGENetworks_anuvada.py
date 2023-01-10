import sys
_module = sys.modules[__name__]
del sys
anuvada = _module
datasets = _module
data_loader = _module
models = _module
classification_attention_rnn = _module
classification_cnn = _module
fit_module_cnn = _module
fit_module_rnn = _module
utils = _module

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


from torch.autograd import Variable


from torch.nn.utils.rnn import pack_padded_sequence


from torch.nn.utils.rnn import pad_packed_sequence


import numpy as np


import torch.nn.functional as F


import pandas as pd


from collections import OrderedDict


from functools import partial


from torch.nn import CrossEntropyLoss


from torch.nn import Module


from torch.optim import SGD


DEFAULT_LOSS = CrossEntropyLoss()


DEFAULT_OPTIMIZER = partial(SGD, lr=0.01, momentum=0.9)


class ProgressBar(object):
    """Cheers @ajratner"""

    def __init__(self, N, length=40):
        self.N = max(1, N)
        self.nf = float(self.N)
        self.length = length
        self.ticks = set([round(i / 100.0 * N) for i in range(101)])
        self.ticks.add(N - 1)
        self.bar(0)

    def bar(self, i, message=''):
        """Assumes i ranges through [0, N-1]"""
        if i in self.ticks:
            b = int(np.ceil((i + 1) / self.nf * self.length))
            sys.stdout.write('\r[{0}{1}] {2}%\t{3}'.format('=' * b, ' ' * (self.length - b), int(100 * ((i + 1) / self.nf)), message))
            sys.stdout.flush()

    def close(self, message=''):
        self.bar(self.N - 1)
        sys.stdout.write('{0}\n\n'.format(message))
        sys.stdout.flush()


def add_metrics_to_log(log, metrics, y_true, y_pred, prefix=''):
    for metric in metrics:
        q = metric(y_true, y_pred)
        log[prefix + metric.__name__] = q
    return log


def log_to_message(log, precision=4):
    fmt = '{0}: {1:.' + str(precision) + 'f}'
    return '    '.join(fmt.format(k, v) for k, v in list(log.items()))


def make_batches(size, batch_size):
    """github.com/fchollet/keras/blob/master/keras/engine/training.py"""
    num_batches = int(np.ceil(size / float(batch_size)))
    return [(i * batch_size, min(size, (i + 1) * batch_size)) for i in range(0, num_batches)]


class FitModuleCNN(Module):

    def fit(self, x=None, y=None, batch_size=64, epochs=1, verbose=1, validation_split=0.0, validation_data=None, shuffle=True, initial_epoch=0, seed=None, loss=DEFAULT_LOSS, optimizer=DEFAULT_OPTIMIZER, run_on='cpu', metrics=None, multilabel=False):
        """Trains the model similar to Keras' .fit(...) method

        # Arguments
            x: training data Tensor.
            y: target data Tensor.
            batch_size: integer. Number of samples per gradient update.
            epochs: integer, the number of times to iterate
                over the training data arrays.
            verbose: 0, 1. Verbosity mode.
                0 = silent, 1 = verbose.
            validation_split: float between 0 and 1:
                fraction of the training data to be used as validation data.
                The model will set apart this fraction of the training data,
                will not train on it, and will evaluate
                the loss and any model metrics
                on this data at the end of each epoch.
            validation_data: (x_val, y_val) tuple on which to evaluate
                the loss and any model metrics
                at the end of each epoch. The model will not
                be trained on this data.
            shuffle: boolean, whether to shuffle the training data
                before each epoch.
            initial_epoch: epoch at which to start training
                (useful for resuming a previous training run)
            seed: random seed.
            optimizer: training optimizer
            loss: training loss
            metrics: list of functions with signatures `metric(y_true, y_pred)`
                where y_true and y_pred are both Tensors
            mask_for_rnn: True if gradient masking needs to passed

        # Returns
            list of OrderedDicts with training metrics
        """
        if run_on == 'cpu':
            self.dtype = torch.FloatTensor
            self.embedding_tensor = torch.LongTensor
        if run_on == 'gpu':
            self.dtype = torch.FloatTensor
            self.embedding_tensor = torch.LongTensor
            self
        if seed and seed >= 0:
            np.random.seed(seed)
            torch.manual_seed(seed)
        if validation_data:
            val_x, val_y = validation_data
        elif validation_split and 0.0 < validation_split < 1.0:
            split = int(x.size()[0] * (1.0 - validation_split))
            x, val_x = x[:split], x[split:]
            y, val_y = y[:split], y[split:]
        else:
            val_x, val_y = None, None
        opt = optimizer(self.parameters())
        logs = []
        self.train()
        n = x.size()[0]
        train_idxs = np.arange(n, dtype=np.int64)
        for t in range(initial_epoch, epochs):
            if verbose:
                None
            if shuffle:
                np.random.shuffle(train_idxs)
            batches = make_batches(n, batch_size)
            batches.pop()
            if verbose:
                pb = ProgressBar(len(batches))
            log = OrderedDict()
            epoch_loss = 0.0
            for batch_i, (batch_start, batch_end) in enumerate(batches):
                batch_idxs = train_idxs[batch_start:batch_end]
                batch_idxs = torch.from_numpy(batch_idxs).long()
                x_batch = Variable(x[batch_idxs]).type(self.embedding_tensor)
                if multilabel:
                    y_batch = Variable(y[batch_idxs]).type(self.dtype)
                else:
                    y_batch = Variable(y[batch_idxs]).type(self.embedding_tensor)
                self.batch_size = batch_size
                y_batch_pred = self(x_batch)
                opt.zero_grad()
                batch_loss = loss(y_batch_pred, y_batch)
                batch_loss.backward()
                opt.step()
                epoch_loss += batch_loss.data[0]
                log['loss'] = float(epoch_loss) / (batch_i + 1)
                if verbose:
                    pb.bar(batch_i, log_to_message(log))
            if metrics:
                y_train_pred = self.predict(x, batch_size)
                add_metrics_to_log(log, metrics, y, y_train_pred)
            if val_x is not None and val_y is not None:
                y_val_pred = self.predict(val_x)
                if multilabel:
                    val_loss = loss(Variable(y_val_pred).type(self.dtype), Variable(val_y).type(self.dtype))
                else:
                    val_loss = loss(Variable(y_val_pred).type(self.dtype), Variable(val_y).type(self.embedding_tensor))
                log['val_loss'] = val_loss.data[0]
                if metrics:
                    add_metrics_to_log(log, metrics, val_y, y_val_pred, 'val_')
            logs.append(log)
            if verbose:
                pb.close(log_to_message(log))
        return logs

    def predict(self, x):
        """Generates output predictions for the input samples.

        Computation is done in batches.

        # Arguments
            x: input data Tensor.
            batch_size: integer.

        # Returns
            prediction Tensor.
        """
        n = x.size()[0]
        train_idxs = np.arange(n, dtype=np.int64)
        batch_size = self.batch_size
        batches = make_batches(n, batch_size)
        batches.pop()
        self.eval()
        for batch_i, (batch_start, batch_end) in enumerate(batches):
            batch_idxs = train_idxs[batch_start:batch_end]
            batch_idxs = torch.from_numpy(batch_idxs).long()
            x_batch = x[batch_idxs]
            x_batch = Variable(x_batch).type(self.embedding_tensor)
            y_batch_pred = self(x_batch).data
            if batch_i == 0:
                y_pred = torch.zeros((n,) + y_batch_pred.size()[1:]).type(self.dtype)
            batch_idxs = batch_idxs.type(self.embedding_tensor)
            y_pred[batch_idxs] = y_batch_pred
        return y_pred


class FitModuleRNN(Module):

    def fit(self, x=None, y=None, masks_for_rnn=None, batch_size=64, epochs=1, verbose=1, validation_split=0.0, validation_data=None, shuffle=True, initial_epoch=0, seed=None, loss=DEFAULT_LOSS, optimizer=DEFAULT_OPTIMIZER, run_on='cpu', metrics=None, multilabel=False):
        """Trains the model similar to Keras' .fit(...) method

        # Arguments
            x: training data Tensor.
            y: target data Tensor.
            batch_size: integer. Number of samples per gradient update.
            epochs: integer, the number of times to iterate
                over the training data arrays.
            verbose: 0, 1. Verbosity mode.
                0 = silent, 1 = verbose.
            validation_split: float between 0 and 1:
                fraction of the training data to be used as validation data.
                The model will set apart this fraction of the training data,
                will not train on it, and will evaluate
                the loss and any model metrics
                on this data at the end of each epoch.
            validation_data: (x_val, y_val) tuple on which to evaluate
                the loss and any model metrics
                at the end of each epoch. The model will not
                be trained on this data.
            shuffle: boolean, whether to shuffle the training data
                before each epoch.
            initial_epoch: epoch at which to start training
                (useful for resuming a previous training run)
            seed: random seed.
            optimizer: training optimizer
            loss: training loss
            metrics: list of functions with signatures `metric(y_true, y_pred)`
                where y_true and y_pred are both Tensors
            mask_for_rnn: True if gradient masking needs to passed

        # Returns
            list of OrderedDicts with training metrics
        """
        if run_on == 'cpu':
            self.dtype = torch.FloatTensor
            self.embedding_tensor = torch.LongTensor
        if run_on == 'gpu':
            self.dtype = torch.FloatTensor
            self.embedding_tensor = torch.LongTensor
            self
        if seed and seed >= 0:
            np.random.seed(seed)
            torch.manual_seed(seed)
        if validation_data:
            val_x, val_y = validation_data
        elif validation_split and 0.0 < validation_split < 1.0:
            split = int(x.size()[0] * (1.0 - validation_split))
            x, val_x = x[:split], x[split:]
            y, val_y = y[:split], y[split:]
            masks_for_rnn, masks_for_rnn_val = masks_for_rnn[:split], masks_for_rnn[split:]
        else:
            val_x, val_y = None, None
        opt = optimizer(self.parameters())
        logs = []
        self.train()
        n = x.size()[0]
        train_idxs = np.arange(n, dtype=np.int64)
        for t in range(initial_epoch, epochs):
            if verbose:
                None
            if shuffle:
                np.random.shuffle(train_idxs)
            batches = make_batches(n, batch_size)
            batches.pop()
            if verbose:
                pb = ProgressBar(len(batches))
            log = OrderedDict()
            epoch_loss = 0.0
            for batch_i, (batch_start, batch_end) in enumerate(batches):
                batch_idxs = train_idxs[batch_start:batch_end]
                batch_idxs = torch.from_numpy(batch_idxs).long()
                x_batch = Variable(x[batch_idxs]).type(self.embedding_tensor)
                if multilabel:
                    y_batch = Variable(y[batch_idxs]).type(self.dtype)
                else:
                    y_batch = Variable(y[batch_idxs]).type(self.embedding_tensor)
                mask = masks_for_rnn[batch_start:batch_end]
                self.batch_size = batch_size
                init_hidden = self.init_hidden()
                y_batch_pred = self(x_batch, mask, init_hidden)
                opt.zero_grad()
                batch_loss = loss(y_batch_pred, y_batch)
                batch_loss.backward()
                opt.step()
                epoch_loss += batch_loss.data[0]
                log['loss'] = float(epoch_loss) / (batch_i + 1)
                if verbose:
                    pb.bar(batch_i, log_to_message(log))
            if metrics:
                y_train_pred = self.predict(x, batch_size)
                add_metrics_to_log(log, metrics, y, y_train_pred)
            if val_x is not None and val_y is not None and masks_for_rnn_val is not None:
                y_val_pred = self.predict(val_x, masks_for_rnn_val)
                if multilabel:
                    val_loss = loss(Variable(y_val_pred).type(self.dtype), Variable(val_y).type(self.dtype))
                else:
                    val_loss = loss(Variable(y_val_pred).type(self.dtype), Variable(val_y).type(self.embedding_tensor))
                log['val_loss'] = val_loss.data[0]
                if metrics:
                    add_metrics_to_log(log, metrics, val_y, y_val_pred, 'val_')
            logs.append(log)
            if verbose:
                pb.close(log_to_message(log))
        return logs

    def predict(self, x, masks_for_rnn):
        """Generates output predictions for the input samples.

        Computation is done in batches.

        # Arguments
            x: input data Tensor.
            batch_size: integer.

        # Returns
            prediction Tensor.
        """
        n = x.size()[0]
        train_idxs = np.arange(n, dtype=np.int64)
        batch_size = self.batch_size
        batches = make_batches(n, batch_size)
        batches.pop()
        self.eval()
        init_hidden = self.init_hidden()
        for batch_i, (batch_start, batch_end) in enumerate(batches):
            batch_idxs = train_idxs[batch_start:batch_end]
            mask = masks_for_rnn[batch_start:batch_end]
            batch_idxs = torch.from_numpy(batch_idxs).long()
            x_batch = x[batch_idxs]
            x_batch = Variable(x_batch).type(self.embedding_tensor)
            y_batch_pred = self(x_batch, mask, init_hidden).data
            if batch_i == 0:
                y_pred = torch.zeros((n,) + y_batch_pred.size()[1:]).type(self.dtype)
            batch_idxs = batch_idxs.type(self.embedding_tensor)
            y_pred[batch_idxs] = y_batch_pred
        return y_pred

