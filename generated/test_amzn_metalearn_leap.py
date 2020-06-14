import sys
_module = sys.modules[__name__]
del sys
leap = _module
leap = _module
updaters = _module
utils = _module
setup = _module
maml = _module
maml = _module
optim = _module
utils = _module
data = _module
getdata = _module
main = _module
model = _module
monitor = _module
resize = _module
run_multi = _module
wrapper = _module

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


from collections import OrderedDict


import torch


import torch.nn as nn


import numpy as np


import random


from abc import abstractmethod


from torch import nn


from torch import optim


def build_iterator(tensors, inner_bsz, outer_bsz, inner_steps, outer_steps,
    cuda=False, device=0):
    """Construct a task iterator from input and output tensor"""
    inner_size = inner_bsz * inner_steps
    outer_size = outer_bsz * outer_steps
    tsz = tensors[0].size(0)
    if tsz != inner_size + outer_size:
        raise ValueError('tensor size mismatch: expected {}, got {}'.format
            (inner_size + outer_size, tsz))

    def iterator(start, stop, size):
        for i in range(start, stop, size):
            out = tuple(t[i:i + size] for t in tensors)
            if cuda:
                out = tuple(t.cuda(device) for t in out)
            yield out
    return iterator(0, inner_size, inner_bsz), iterator(inner_size, tsz,
        outer_bsz)


def compute_auc(x):
    """Compute AUC (composite trapezoidal rule)"""
    T = len(x)
    v = 0
    for i in range(1, T):
        v += ((x[i] - x[i - 1]) / 2 + x[i - 1]) / T
    return v


class AggRes:
    """Results aggregation container
    Aggregates results over a mini-batch of tasks
    """

    def __init__(self, results):
        self.train_res, self.val_res = zip(*results)
        self.aggregate_train()
        self.aggregate_val()

    def aggregate_train(self):
        """Aggregate train results"""
        (self.train_meta_loss, self.train_loss, self.train_acc, self.
            train_losses, self.train_accs) = self.aggregate(self.train_res)

    def aggregate_val(self):
        """Aggregate val results"""
        (self.val_meta_loss, self.val_loss, self.val_acc, self.val_losses,
            self.val_accs) = self.aggregate(self.val_res)

    @staticmethod
    def aggregate(results):
        """Aggregate losses and accs across Res instances"""
        agg_losses = np.stack([res.losses for res in results], axis=1)
        agg_ncorrects = np.stack([res.ncorrects for res in results], axis=1)
        agg_nsamples = np.stack([res.nsamples for res in results], axis=1)
        mean_loss = agg_losses.mean()
        mean_losses = agg_losses.mean(axis=1)
        mean_meta_loss = compute_auc(mean_losses)
        mean_acc = agg_ncorrects.sum() / agg_nsamples.sum()
        mean_accs = agg_ncorrects.sum(axis=1) / agg_nsamples.sum(axis=1)
        return mean_meta_loss, mean_loss, mean_acc, mean_losses, mean_accs


def n_correct(p, y):
    """Number correct predictions"""
    _, p = p.max(1)
    correct = (p == y).sum().item()
    return correct


class Res:
    """Results container
    Attributes:
        losses (list): list of losses over batch iterator
        accs (list): list of accs over batch iterator
        meta_loss (float): auc over losses
        loss (float): mean loss over losses. Call ``aggregate`` to compute.
        acc (float): mean acc over accs. Call ``aggregate`` to compute.
    """

    def __init__(self):
        self.losses = []
        self.accs = []
        self.ncorrects = []
        self.nsamples = []
        self.meta_loss = 0
        self.loss = 0
        self.acc = 0

    def log(self, loss, pred, target):
        """Log loss and accuracies"""
        nsamples = target.size(0)
        ncorr = n_correct(pred.data, target.data)
        accuracy = ncorr / target.size(0)
        self.losses.append(loss)
        self.ncorrects.append(ncorr)
        self.nsamples.append(nsamples)
        self.accs.append(accuracy)

    def aggregate(self):
        """Compute aggregate statistics"""
        self.accs = np.array(self.accs)
        self.losses = np.array(self.losses)
        self.nsamples = np.array(self.nsamples)
        self.ncorrects = np.array(self.ncorrects)
        self.loss = self.losses.mean()
        self.meta_loss = compute_auc(self.losses)
        self.acc = self.ncorrects.sum() / self.nsamples.sum()


def maml_inner_step(input, output, model, optimizer, criterion, create_graph):
    """Create a computation graph through the gradient operation

    Arguments:
        input (torch.Tensor): input tensor.
        output (torch.Tensor): target tensor.
        model (torch.nn.Module): task learner.
        optimizer (maml.optim): optimizer for inner loop.
        criterion (func): loss criterion.
        create_graph (bool): create graph through gradient step.
    """
    new_parameters = None
    prediction = model(input)
    loss = criterion(prediction, output)
    loss.backward(create_graph=create_graph, retain_graph=create_graph)
    if create_graph:
        _, new_parameters = optimizer.step(retain_graph=create_graph)
    else:
        optimizer.step(retain_graph=create_graph)
    return loss, prediction, new_parameters


def build_dict(names, parameters):
    """Populate an ordered dictionary of parameters"""
    state_dict = OrderedDict({n: p for n, p in zip(names, parameters)})
    return state_dict


def _load_from_par_dict(module, par_dict, prefix):
    """Replace the module's _parameter dict with par_dict"""
    _new_parameters = OrderedDict()
    for name, param in module._parameters.items():
        key = prefix + name
        if key in par_dict:
            input_param = par_dict[key]
        else:
            input_param = param
        if input_param.shape != param.shape:
            raise ValueError(
                'size mismatch for {}: copying a param of {} from checkpoint, where the shape is {} in current model.'
                .format(key, param.shape, input_param.shape))
        _new_parameters[name] = input_param
    module._parameters = _new_parameters


def load_state_dict(module, state_dict):
    """Replaces parameters and buffers from :attr:`state_dict` into
    the given module and its descendants. In contrast to the module's
    method, this function will *not* do in-place copy of underlying data on
    *parameters*, but instead replace the ``_parameter`` dict in each
    module and its descendants. This allows us to backpropr through previous
    gradient steps using the standard top-level API.

    .. note::
        You must store the original state dict (with keep_vars=True) separately
        and, when ready to update them, use :funct:`load_state_dict` to return
        as the module's parameters.

    Arguments:
        module (torch.nn.Module): a module instance whose state to update.
        state_dict (dict): a dict containing parameters and
            persistent buffers.
    """
    par_names = [n for n, _ in module.named_parameters()]
    par_dict = OrderedDict({k: v for k, v in state_dict.items() if k in
        par_names})
    no_par_dict = OrderedDict({k: v for k, v in state_dict.items() if k not in
        par_names})
    excess = [k for k in state_dict.keys() if k not in list(no_par_dict.
        keys()) + list(par_dict.keys())]
    if excess:
        raise ValueError(
            "State variables %r not in the module's state dict %r" % (
            excess, par_names))
    metadata = getattr(state_dict, '_metadata', None)
    if metadata is not None:
        par_dict._metadata = metadata
        no_par_dict._metadata = metadata
    module.load_state_dict(no_par_dict, strict=False)

    def load(module, prefix=''):
        _load_from_par_dict(module, par_dict, prefix)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')
    load(module)


def maml_task(data_inner, data_outer, model, optimizer, criterion, create_graph
    ):
    """Adapt model parameters to task and use adapted params to predict new samples

    Arguments:
        data_inner (iterable): list of input-output for task adaptation.
        data_outer (iterable): list of input-output for task validation.
        model (torch.nn.Module): task learner.
        optimizer (maml.optim): optimizer for inner loop.
        criterion (func): loss criterion.
        create_graph (bool): create graph through gradient step.
    """
    original_parameters = model.state_dict(keep_vars=True)
    device = next(model.parameters()).device
    train_res = Res()
    for i, (input, output) in enumerate(data_inner):
        input = input.to(device, non_blocking=True)
        output = output.to(device, non_blocking=True)
        loss, prediction, new_params = maml_inner_step(input, output, model,
            optimizer, criterion, create_graph)
        train_res.log(loss.item(), prediction, output)
        if create_graph:
            load_state_dict(model, build_dict([n for n, _ in model.
                named_parameters()], new_params))
        for p in original_parameters.values():
            p.grad = None
    val_res = Res()
    predictions = []
    for i, (input, output) in enumerate(data_outer):
        input = input.to(device, non_blocking=True)
        output = output.to(device, non_blocking=True)
        prediction = model(input)
        predictions.append(prediction)
        batch_loss = criterion(prediction, output)
        loss += batch_loss
        val_res.log(batch_loss.item(), prediction, output)
    loss = loss / (i + 1)
    load_state_dict(model, original_parameters)
    return loss, predictions, (train_res, val_res)


def maml_outer_step(task_iterator, model, optimizer_cls, criterion,
    return_predictions=True, return_results=True, create_graph=True, **
    optimizer_kwargs):
    """MAML objective.

    Run MAML on a batch of tasks.


    Arguments:
        task_iterator (iterator): data sampler for K tasks. Of the format
            [task1, task2, task3] where each task is of the format
            task1 = (data_iterator_inner, data_iterator_outer) and each
            data_iterator_ = [(input_batch1, target_batch1), ...]

            ::note::
                the inner data_iterator defines the number of gradient

        model (Module): task learner.
        optimizer_cls (maml.optim.SGD, maml.optim.Adam): inner optimizer class.
            Must allow backpropagation through gradient step.
        criterion (func): loss criterion.
        return_predictions (bool): whether to return.
        return_results (bool): return accumulated meta-data.
        create_graph (bool): create computational graph through gradient step.
        optimizer_kwargs (kwargs): kwargs to optimizer.
    """
    loss = 0
    predictions, results = [], []
    for i, task in enumerate(task_iterator):
        inner_iterator, outer_iterator = task
        task_optimizer = optimizer_cls(model.parameters(), **optimizer_kwargs)
        task_loss, task_predictions, task_res = maml_task(inner_iterator,
            outer_iterator, model, task_optimizer, criterion, create_graph)
        loss += task_loss
        predictions.append(task_predictions)
        results.append(task_res)
    loss = loss / (i + 1)
    results = AggRes(results)
    out = [loss]
    if return_predictions:
        out.append(predictions)
    if return_results:
        out.append(results)
    return out


class MAML(nn.Module):
    """MAML

    Class Instance for the MAML objective

    Arguments:
        model (torch.nn.Module): task learner.
        optimizer_cls (maml.optim): task optimizer. Note: must allow backpropagation through gradient steps.
        criterion (func): loss criterion.
        tensor (bool): whether meta mini-batches come as a tensor or as a list of dataloaders.
        inner_bsz (int): if tensor=True, batch size in inner loop.
        outer_bsz (int): if tensor=True, batch size in outer loop.
        inner_steps (int): if tensor=True, number of steps in inner loop.
        outer_steps (int): if tensor=True, number of steps in outer loop.

    Example:
        >>> loss = maml.forward(task_iterator)
        >>> loss.backward()
        >>> meta_optimizer.step()
        >>> meta_optimizer.zero_grad()
    """

    def __init__(self, model, optimizer_cls, criterion, tensor, inner_bsz=
        None, outer_bsz=None, inner_steps=None, outer_steps=None, **
        optimizer_kwargs):
        super(MAML, self).__init__()
        self.model = model
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs
        self.criterion = criterion
        self.tensor = tensor
        self.inner_bsz = inner_bsz
        self.outer_bsz = outer_bsz
        self.inner_steps = inner_steps
        self.outer_steps = outer_steps
        if tensor:
            assert inner_bsz is not None, 'set inner_bsz with tensor=True'
            assert outer_bsz is not None, 'set outer_bsz with tensor=True'
            assert inner_steps is not None, 'set inner_steps with tensor=True'
            assert outer_steps is not None, 'set outer_steps with tensor=True'

    def forward(self, inputs, return_predictions=True, return_results=True,
        create_graph=True):
        task_iterator = inputs if not self.tensor else [build_iterator(i,
            self.inner_bsz, self.outer_bsz, self.inner_steps, self.
            outer_steps) for i in inputs]
        return maml_outer_step(task_iterator=task_iterator, model=self.
            model, optimizer_cls=self.optimizer_cls, criterion=self.
            criterion, return_predictions=return_predictions,
            return_results=return_results, create_graph=create_graph, **
            self.optimizer_kwargs)


class UnSqueeze(nn.Module):
    """Create channel dim if necessary."""

    def __init__(self):
        super(UnSqueeze, self).__init__()

    def forward(self, input):
        """Creates channel dimension on a 3-d tensor. Null-op if input is a 4-d tensor.

        Arguments:
            input (torch.Tensor): tensor to unsqueeze.
        """
        if input.dim() == 4:
            return input
        return input.unsqueeze(1)


class Squeeze(nn.Module):
    """Undo excess dimensions"""

    def __init__(self):
        super(Squeeze, self).__init__()

    def forward(self, input):
        """Squeeze singular dimensions of an input tensor.

        Arguments:
            input (torch.Tensor): tensor to unsqueeze.
        """
        if input.size(0) != 0:
            return input.squeeze()
        input = input.squeeze()
        return input.view(1, *input.size())


class OmniConv(nn.Module):
    """ConvNet classifier.

    Arguments:
        nclasses (int): number of classes to predict in each alphabet
        nlayers (int): number of convolutional layers (default=4).
        kernel_size (int): kernel size in each convolution (default=3).
        num_filters (int): number of output filters in each convolution (default=64)
        imsize (tuple): tuple of image height and width dimension.
        padding (bool, int, tuple): padding argument to convolution layers (default=True).
        max_pool(bool): use max pooling in each convolution layer (default=True)
        batch_norm (bool): use batch normalization in each convolution layer (default=True).
        multi_head (bool): multi-headed training (default=False).
    """

    def __init__(self, nclasses, nlayers=4, kernel_size=3, num_filters=64,
        imsize=(28, 28), padding=True, max_pool=True, batch_norm=True,
        multi_head=False):
        super(OmniConv, self).__init__()
        self.nlayers = nlayers
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.imsize = imsize
        self.max_pool = max_pool
        self.batch_norm = batch_norm
        self.multi_head = multi_head

        def conv_block(nin):
            block = [nn.Conv2d(nin, num_filters, kernel_size, padding=padding)]
            if max_pool:
                block.append(nn.MaxPool2d(2))
            if batch_norm:
                block.append(nn.BatchNorm2d(num_filters))
            block.append(nn.ReLU())
            return block
        layers = [UnSqueeze()]
        for i in range(nlayers):
            layers.extend(conv_block(1 if i == 0 else num_filters))
        if not max_pool:
            fsz = imsize[0] - 2 * nlayers if padding else imsize[0]
            layers.append(nn.AvgPool2d(fsz))
        layers.append(Squeeze())
        if not self.multi_head:
            layers.append(nn.Linear(num_filters, nclasses))
            self.model = nn.Sequential(*layers)
        else:
            self.conv = nn.Sequential(*layers)
            self.heads = nn.ModuleList([nn.Linear(num_filters, nclasses) for
                _ in range(50)])

    def forward(self, input, idx=None):
        if not self.multi_head:
            return self.model(input)
        input = self.conv(input)
        return self.heads[idx](input)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_amzn_metalearn_leap(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(Squeeze(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(UnSqueeze(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

