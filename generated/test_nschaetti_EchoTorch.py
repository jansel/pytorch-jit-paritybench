import sys
_module = sys.modules[__name__]
del sys
conf = _module
echotorch = _module
DatasetComposer = _module
HenonAttractor = _module
LambdaDataset = _module
LogisticMapDataset = _module
LorenzAttractor = _module
MackeyGlass2DDataset = _module
MackeyGlassDataset = _module
MemTestDataset = _module
NARMADataset = _module
PeriodicSignalDataset = _module
RosslerAttractor = _module
SinusoidalTimeseries = _module
SwitchAttractorDataset = _module
datasets = _module
HNilsNet = _module
NilsNet = _module
TNilsNet = _module
models = _module
BDESN = _module
BDESNCell = _module
BDESNPCA = _module
Conceptor = _module
ConceptorNet = _module
ConceptorNetCell = _module
ConceptorPool = _module
EESN = _module
ESN = _module
ESN2d = _module
ESNCell = _module
GatedESN = _module
HESN = _module
ICACell = _module
Identity = _module
IncSFACell = _module
LiESN = _module
LiESNCell = _module
OnlinePCACell = _module
PCACell = _module
RRCell = _module
SFACell = _module
StackedESN = _module
UMESN = _module
nn = _module
transforms = _module
images = _module
Character = _module
Character2Gram = _module
Character3Gram = _module
Compose = _module
Embedding = _module
FunctionWord = _module
GensimModel = _module
GloveVector = _module
PartOfSpeech = _module
Tag = _module
Token = _module
Transformer = _module
text = _module
utils = _module
error_measures = _module
utility_functions = _module
visualisation = _module
convert_images = _module
conceptors_4_patterns_generation = _module
logistic_map = _module
narma10_esn_feedbacks = _module
memtest = _module
NilsNet_example = _module
pca_tests = _module
switch_attractor_esn = _module
mackey_glass_esn = _module
narma10_esn = _module
narma10_esn_sgd = _module
narma10_gated_esn = _module
narma10_stacked_esn = _module
object_recognition = _module
sfa_logmap = _module
setup = _module
test = _module
test_narma10_prediction = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


from torch.utils.data.dataset import Dataset


import numpy as np


from random import shuffle


import collections


import math


import torchvision


import torch.nn as nn


import torch.sparse


from torch.autograd import Variable


import torch.nn.functional as F


from sklearn.decomposition import IncrementalPCA


import math as m


from scipy.interpolate import interp1d


import numpy.linalg as lin


from sklearn.decomposition import PCA


from torch.utils.data.dataloader import DataLoader


from torchvision import datasets


from torchvision import transforms


import torch.optim as optim


class HNilsNet(nn.Module):
    """
    A Hierarchical NilsNet
    """

    def __init__(self):
        """
        Constructor
        """
        pass

    def forward(self):
        """
        Forward
        :return:
        """
        pass


class NilsNet(nn.Module):
    """
    A NilsNet
    """

    def __init__(self, reservoir_dim, sfa_dim, ica_dim, pretrained=False, feature_selector='resnet18'):
        """
        Constructor
        """
        super(NilsNet, self).__init__()
        if feature_selector == 'resnet18':
            self.feature_selector = torchvision.models.resnet18(pretrained=True)
        elif feature_selector == 'resnet34':
            self.feature_selector = torchvision.models.resnet34(pretrained=True)
        elif feature_selector == 'resnet50':
            self.feature_selector = torchvision.models.resnet50(pretrained=True)
        elif feature_selector == 'alexnet':
            self.feature_selector = torchvision.models.alexnet(pretrained=True)
        self.reservoir_input_dim = self.feature_selector.fc.in_features
        self.feature_selector.fc = ecnn.Identity()

    def forward(self, x):
        """
        Forward
        :return:
        """
        return self.feature_selector(x)


class ESNCell(nn.Module):
    """
    Echo State Network layer
    """

    def __init__(self, input_dim, output_dim, spectral_radius=0.9, bias_scaling=0, input_scaling=1.0, w=None, w_in=None, w_bias=None, w_fdb=None, sparsity=None, input_set=[1.0, -1.0], w_sparsity=None, nonlin_func=torch.tanh, feedbacks=False, feedbacks_dim=None, wfdb_sparsity=None, normalize_feedbacks=False, seed=None, w_distrib='uniform', win_distrib='uniform', wbias_distrib='uniform', win_normal=(0.0, 1.0), w_normal=(0.0, 1.0), wbias_normal=(0.0, 1.0), dtype=torch.float32):
        """
        Constructor
        :param input_dim: Inputs dimension.
        :param output_dim: Reservoir size
        :param spectral_radius: Reservoir's spectral radius
        :param bias_scaling: Scaling of the bias, a constant input to each neuron (default: 0, no bias)
        :param input_scaling: Scaling of the input weight matrix, default 1.
        :param w: Internation weights matrix
        :param w_in: Input-reservoir weights matrix
        :param w_bias: Bias weights matrix
        :param sparsity:
        :param input_set:
        :param w_sparsity:
        :param nonlin_func: Reservoir's activation function (tanh, sig, relu)
        """
        super(ESNCell, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.spectral_radius = spectral_radius
        self.bias_scaling = bias_scaling
        self.input_scaling = input_scaling
        self.sparsity = sparsity
        self.input_set = input_set
        self.w_sparsity = w_sparsity
        self.nonlin_func = nonlin_func
        self.feedbacks = feedbacks
        self.feedbacks_dim = feedbacks_dim
        self.wfdb_sparsity = wfdb_sparsity
        self.normalize_feedbacks = normalize_feedbacks
        self.w_distrib = w_distrib
        self.win_distrib = win_distrib
        self.wbias_distrib = wbias_distrib
        self.win_normal = win_normal
        self.w_normal = w_normal
        self.wbias_normal = wbias_normal
        self.dtype = dtype
        self.register_buffer('hidden', self.init_hidden())
        self.register_buffer('w_in', self._generate_win(w_in, seed=seed))
        self.register_buffer('w', self._generate_w(w, seed=seed))
        self.register_buffer('w_bias', self._generate_wbias(w_bias, seed=seed))
        if feedbacks:
            self.register_buffer('w_fdb', self._generate_wfdb(w_fdb, seed=seed))

    def forward(self, u, y=None, w_out=None, reset_state=True):
        """
        Forward
        :param u: Input signal
        :param y: Target output signal for teacher forcing
        :param w_out: Output weights for teacher forcing
        :return: Resulting hidden states
        """
        time_length = int(u.size()[1])
        n_batches = int(u.size()[0])
        outputs = Variable(torch.zeros(n_batches, time_length, self.output_dim, dtype=self.dtype))
        outputs = outputs if self.hidden.is_cuda else outputs
        for b in range(n_batches):
            if reset_state:
                self.reset_hidden()
            for t in range(time_length):
                ut = u[b, t]
                u_win = self.w_in.mv(ut)
                x_w = self.w.mv(self.hidden)
                if self.feedbacks and self.training and y is not None:
                    yt = y[b, t]
                    y_wfdb = self.w_fdb.mv(yt)
                    x = u_win + x_w + y_wfdb + self.w_bias
                elif self.feedbacks and not self.training and w_out is not None:
                    bias_hidden = torch.cat((Variable(torch.ones(1)), self.hidden), dim=0)
                    yt = w_out.t().mv(bias_hidden)
                    if self.normalize_feedbacks:
                        yt -= torch.min(yt)
                        yt /= torch.max(yt) - torch.min(yt)
                        yt /= torch.sum(yt)
                    y_wfdb = self.w_fdb.mv(yt)
                    x = u_win + x_w + y_wfdb + self.w_bias
                else:
                    x = u_win + x_w + self.w_bias
                x = self.nonlin_func(x)
                self.hidden.data = x.view(self.output_dim).data
                outputs[b, t] = self.hidden
        return outputs

    def init_hidden(self):
        """
        Init hidden layer
        :return: Initiated hidden layer
        """
        return Variable(torch.zeros(self.output_dim, dtype=self.dtype), requires_grad=False)

    def reset_hidden(self):
        """
        Reset hidden layer
        :return:
        """
        self.hidden.fill_(0.0)

    def set_hidden(self, x):
        """
        Set hidden layer
        :param x:
        :return:
        """
        self.hidden.data = x.data

    def get_spectral_radius(self):
        """
        Get W's spectral radius
        :return: W's spectral radius
        """
        return echotorch.utils.spectral_radius(self.w)

    def _generate_w(self, w, seed=None):
        """
        Generate W matrix
        :return:
        """
        if w is None:
            w = self.generate_w(output_dim=self.output_dim, w_distrib=self.w_distrib, w_sparsity=self.w_sparsity, mean=self.w_normal[0], std=self.w_normal[1], seed=seed, dtype=self.dtype)
        elif callable(w):
            w = w(self.output_dim)
        w *= self.spectral_radius / echotorch.utils.spectral_radius(w)
        return Variable(w, requires_grad=False)

    def _generate_win(self, w_in, seed=None):
        """
        Generate Win matrix
        :return:
        """
        if seed is not None:
            np.random.seed(seed)
            torch.random.manual_seed(seed)
        if w_in is None:
            if self.win_distrib == 'uniform':
                w_in = self.generate_uniform_matrix(size=(self.output_dim, self.input_dim), sparsity=self.sparsity, input_set=self.input_set)
                if self.dtype == torch.float32:
                    w_in = torch.from_numpy(w_in.astype(np.float32))
                else:
                    w_in = torch.from_numpy(w_in.astype(np.float64))
            else:
                w_in = self.generate_gaussian_matrix(size=(self.output_dim, self.input_dim), sparsity=self.sparsity, mean=self.win_normal[0], std=self.win_normal[1], dtype=self.dtype)
            w_in *= self.input_scaling
        elif callable(w_in):
            w_in = w_in(self.output_dim, self.input_dim)
        return Variable(w_in, requires_grad=False)

    def _generate_wbias(self, w_bias, seed=None):
        """
        Generate Wbias matrix
        :return:
        """
        if seed is not None:
            torch.manual_seed(seed)
        if w_bias is None:
            if self.w_distrib == 'uniform':
                w_bias = self.generate_uniform_matrix(size=(1, self.output_dim), sparsity=1.0, input_set=[-1.0, 1.0])
                if self.dtype == torch.float32:
                    w_bias = torch.from_numpy(w_bias.astype(np.float32))
                else:
                    w_bias = torch.from_numpy(w_bias.astype(np.float64))
            else:
                w_bias = self.generate_gaussian_matrix(size=(1, self.output_dim), sparsity=1.0, mean=self.wbias_normal[0], std=self.wbias_normal[1], dtype=self.dtype)
            w_bias *= self.bias_scaling
        elif callable(w_bias):
            w_bias = w_bias(self.output_dim)
        return Variable(w_bias, requires_grad=False)

    def _generate_wfdb(self, w_fdb, seed=None):
        """
        Generate Wfdb matrix
        :return:
        """
        if seed is not None:
            torch.manual_seed(seed)
        if w_fdb is None:
            if self.wfdb_sparsity is None:
                w_fdb = self.input_scaling * (np.random.randint(0, 2, (self.output_dim, self.feedbacks_dim)) * 2.0 - 1.0)
                w_fdb = torch.from_numpy(w_fdb.astype(np.float32))
            else:
                w_fdb = self.input_scaling * np.random.choice(np.append([0], self.input_set), (self.output_dim, self.feedbacks_dim), p=np.append([1.0 - self.wfdb_sparsity], [self.wfdb_sparsity / len(self.input_set)] * len(self.input_set)))
                if self.dtype == torch.float32:
                    w_fdb = torch.from_numpy(w_fdb.astype(np.float32))
                else:
                    w_fdb = torch.from_numpy(w_fdb.astype(np.float64))
        elif callable(w_fdb):
            w_fdb = w_fdb(self.output_dim, self.feedbacks_dim)
        return Variable(w_fdb, requires_grad=False)

    @staticmethod
    def generate_uniform_matrix(size, sparsity, input_set):
        """
        Generate uniform Win matrix
        :param w_in:
        :param seed:
        :return:
        """
        if sparsity is None:
            w = np.random.randint(0, 2, size) * 2.0 - 1.0
        else:
            w = np.random.choice(np.append([0], input_set), size, p=np.append([1.0 - sparsity], [sparsity / len(input_set)] * len(input_set)))
        return w

    @staticmethod
    def generate_gaussian_matrix(size, sparsity, mean=0.0, std=1.0, dtype=torch.float32):
        """
        Generate gaussian Win matrix
        :return:
        """
        if sparsity is None:
            w = torch.zeros(size, dtype=dtype)
            w = w.normal_(mean=mean, std=std)
        else:
            w = torch.zeros(size, dtype=dtype)
            w = w.normal_(mean=mean, std=std)
            mask = torch.zeros(size, dtype=dtype)
            mask.bernoulli_(p=sparsity)
            w *= mask
        return w

    @staticmethod
    def generate_w(output_dim, w_distrib='uniform', w_sparsity=None, mean=0.0, std=1.0, seed=None, dtype=torch.float32):
        """
        Generate W matrix
        :param output_dim:
        :param w_sparsity:
        :return:
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        if w_distrib == 'uniform':
            w = ESNCell.generate_uniform_matrix(size=(output_dim, output_dim), sparsity=w_sparsity, input_set=[-1.0, 1.0])
            w = torch.from_numpy(w.astype(np.float32))
        else:
            w = ESNCell.generate_gaussian_matrix(size=(output_dim, output_dim), sparsity=w_sparsity, mean=mean, std=std, dtype=dtype)
        return w

    @staticmethod
    def to_sparse(m):
        """
        To sparse matrix
        :param m:
        :return:
        """
        rows = torch.LongTensor()
        columns = torch.LongTensor()
        values = torch.FloatTensor()
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                if m[i, j] != 0.0:
                    rows = torch.cat((rows, torch.LongTensor([i])), dim=0)
                    columns = torch.cat((columns, torch.LongTensor([j])), dim=0)
                    values = torch.cat((values, torch.FloatTensor([m[i, j]])), dim=0)
        indices = torch.cat((rows.unsqueeze(0), columns.unsqueeze(0)), dim=0)
        return torch.sparse.FloatTensor(indices, values)


class LiESNCell(ESNCell):
    """
    Leaky-Integrated Echo State Network layer
    """

    def __init__(self, leaky_rate=1.0, train_leaky_rate=False, *args, **kwargs):
        """
        Constructor
        :param leaky_rate: Reservoir's leaky rate (default 1.0, normal ESN)
        :param train_leaky_rate: Train leaky rate as parameter? (default: False)
        """
        super(LiESNCell, self).__init__(*args, **kwargs)
        if self.dtype == torch.float32:
            tensor_type = torch.FloatTensor
        else:
            tensor_type = torch.DoubleTensor
        if train_leaky_rate:
            self.leaky_rate = nn.Parameter(tensor_type(1).fill_(leaky_rate), requires_grad=True)
        else:
            self.register_buffer('leaky_rate', Variable(tensor_type(1).fill_(leaky_rate), requires_grad=False))

    def forward(self, u, y=None, w_out=None, reset_state=True):
        """
        Forward
        :param u: Input signal.
        :return: Resulting hidden states.
        """
        time_length = int(u.size()[1])
        n_batches = int(u.size()[0])
        outputs = Variable(torch.zeros(n_batches, time_length, self.output_dim, dtype=self.dtype))
        outputs = outputs if self.hidden.is_cuda else outputs
        for b in range(n_batches):
            if reset_state:
                self.reset_hidden()
            for t in range(time_length):
                ut = u[b, t]
                u_win = self.w_in.mv(ut)
                x_w = self.w.mv(self.hidden)
                if self.feedbacks and self.training and y is not None:
                    yt = y[b, t]
                    y_wfdb = self.w_fdb.mv(yt)
                    x = u_win + x_w + y_wfdb + self.w_bias
                elif self.feedbacks and not self.training and w_out is not None:
                    bias_hidden = torch.cat((Variable(torch.ones(1)), self.hidden), dim=0)
                    yt = w_out.t().mv(bias_hidden)
                    if self.normalize_feedbacks:
                        yt -= torch.min(yt)
                        yt /= torch.max(yt) - torch.min(yt)
                        yt /= torch.sum(yt)
                    y_wfdb = self.w_fdb.mv(yt)
                    x = u_win + x_w + y_wfdb + self.w_bias
                else:
                    x = u_win + x_w + self.w_bias
                x = self.nonlin_func(x)
                self.hidden.data = (self.hidden.mul(1.0 - self.leaky_rate) + x.view(self.output_dim).mul(self.leaky_rate)).data
                outputs[b, t] = self.hidden
        return outputs


class BDESNCell(nn.Module):
    """
    Bi-directional Echo State Network module
    """

    def __init__(self, input_dim, hidden_dim, spectral_radius=0.9, bias_scaling=0, input_scaling=1.0, w=None, w_in=None, w_bias=None, sparsity=None, input_set=[1.0, -1.0], w_sparsity=None, nonlin_func=torch.tanh, leaky_rate=1.0, create_cell=True):
        """
        Constructor
        :param input_dim: Inputs dimension.
        :param hidden_dim: Hidden layer dimension
        :param spectral_radius: Reservoir's spectral radius
        :param bias_scaling: Scaling of the bias, a constant input to each neuron (default: 0, no bias)
        :param input_scaling: Scaling of the input weight matrix, default 1.
        :param w: Internal weights matrix
        :param w_in: Input-reservoir weights matrix
        :param w_bias: Bias weights matrix
        :param sparsity:
        :param input_set:
        :param w_sparsity:
        :param nonlin_func: Reservoir's activation function (tanh, sig, relu)
        """
        super(BDESNCell, self).__init__()
        if create_cell:
            self.esn_cell = LiESNCell(leaky_rate, False, input_dim, hidden_dim, spectral_radius, bias_scaling, input_scaling, w, w_in, w_bias, None, sparsity, input_set, w_sparsity, nonlin_func)

    @property
    def w(self):
        """
        Hidden weight matrix
        :return:
        """
        return self.esn_cell.w

    @property
    def w_in(self):
        """
        Input matrix
        :return:
        """
        return self.esn_cell.w_in

    def reset(self):
        """
        Reset learning
        :return:
        """
        self.output.reset()
        self.train(True)

    def get_w_out(self):
        """
        Output matrix
        :return:
        """
        return self.output.w_out

    def set_w(self, w):
        """
        Set W
        :param w:
        :return:
        """
        self.esn_cell.w = w

    def forward(self, u, y=None):
        """
        Forward
        :param u: Input signal.
        :param y: Target outputs
        :return: Output or hidden states
        """
        forward_hidden_states = self.esn_cell(u)
        backward_hidden_states = self.esn_cell(Variable(torch.from_numpy(np.flip(u.data.numpy(), 1).copy())))
        backward_hidden_states = Variable(torch.from_numpy(np.flip(backward_hidden_states.data.numpy(), 1).copy()))
        return torch.cat((forward_hidden_states, backward_hidden_states), dim=2)

    def finalize(self):
        """
        Finalize training with LU factorization
        """
        self.output.finalize()
        self.train(False)

    def reset_hidden(self):
        """
        Reset hidden layer
        :return:
        """
        self.esn_cell.reset_hidden()

    def get_spectral_radius(self):
        """
        Get W's spectral radius
        :return: W's spectral radius
        """
        return self.esn_cell.get_spectral_raduis()


class RRCell(nn.Module):
    """
    Ridge Regression cell
    """

    def __init__(self, input_dim, output_dim, ridge_param=0.0, feedbacks=False, with_bias=True, learning_algo='inv', softmax_output=False, averaged=False, dtype=torch.float32):
        """
        Constructor
        :param input_dim: Inputs dimension.
        :param output_dim: Reservoir size
        """
        super(RRCell, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.ridge_param = ridge_param
        self.feedbacks = feedbacks
        self.with_bias = with_bias
        self.learning_algo = learning_algo
        self.softmax_output = softmax_output
        self.softmax = torch.nn.Softmax(dim=2)
        self.averaged = averaged
        self.n_samples = 0
        self.dtype = dtype
        if self.with_bias:
            self.x_size = input_dim + 1
        else:
            self.x_size = input_dim
        self.register_buffer('xTx', Variable(torch.zeros(self.x_size, self.x_size, dtype=dtype), requires_grad=False))
        self.register_buffer('xTy', Variable(torch.zeros(self.x_size, output_dim, dtype=dtype), requires_grad=False))
        self.register_buffer('w_out', Variable(torch.zeros(1, input_dim, dtype=dtype), requires_grad=False))

    def reset(self):
        """
        Reset learning
        :return:
        """
        """self.xTx.data = torch.zeros(self.x_size, self.x_size)
        self.xTy.data = torch.zeros(self.x_size, self.output_dim)
        self.w_out.data = torch.zeros(1, self.input_dim)"""
        self.xTx.data.fill_(0.0)
        self.xTy.data.fill_(0.0)
        self.w_out.data.fill_(0.0)
        self.train(True)

    def get_w_out(self):
        """
        Output matrix
        :return:
        """
        return self.w_out

    def forward(self, x, y=None):
        """
        Forward
        :param x: Input signal.
        :param y: Target outputs
        :return: Output or hidden states
        """
        batch_size = x.size()[0]
        time_length = x.size()[1]
        if self.with_bias:
            x = self._add_constant(x)
        if self.training:
            for b in range(batch_size):
                if not self.averaged:
                    self.xTx.data.add_(x[b].t().mm(x[b]).data)
                    self.xTy.data.add_(x[b].t().mm(y[b]).data)
                else:
                    self.xTx.data.add_((x[b].t().mm(x[b]) / time_length).data)
                    self.xTy.data.add_((x[b].t().mm(y[b]) / time_length).data)
                    self.n_samples += 1.0
            if self.with_bias:
                return x[:, :, 1:]
            else:
                return x
        elif not self.training:
            outputs = Variable(torch.zeros(batch_size, time_length, self.output_dim, dtype=self.dtype), requires_grad=False)
            outputs = outputs if self.w_out.is_cuda else outputs
            for b in range(batch_size):
                outputs[b] = torch.mm(x[b], self.w_out)
            if self.softmax_output:
                return self.softmax(outputs)
            else:
                return outputs

    def finalize(self, train=False):
        """
        Finalize training with LU factorization or Pseudo-inverse
        """
        if self.learning_algo == 'inv':
            if not self.averaged:
                ridge_xTx = self.xTx + self.ridge_param * torch.eye(self.input_dim + self.with_bias, dtype=self.dtype)
                inv_xTx = ridge_xTx.inverse()
                self.w_out.data = torch.mm(inv_xTx, self.xTy).data
            else:
                self.xTx = self.xTx / self.n_samples
                self.xTy = self.xTy / self.n_samples
                ridge_xTx = self.xTx + self.ridge_param * torch.eye(self.input_dim + self.with_bias, dtype=self.dtype)
                inv_xTx = ridge_xTx.inverse()
                self.w_out.data = torch.mm(inv_xTx, self.xTy).data
        else:
            self.w_out.data = torch.gesv(self.xTy, self.xTx + torch.eye(self.esn_cell.output_dim).mul(self.ridge_param)).data
        self.train(train)

    def _add_constant(self, x):
        """
        Add constant
        :param x:
        :return:
        """
        if x.is_cuda:
            bias = Variable(torch.ones((x.size()[0], x.size()[1], 1), dtype=self.dtype), requires_grad=False)
        else:
            bias = Variable(torch.ones((x.size()[0], x.size()[1], 1), dtype=self.dtype), requires_grad=False)
        return torch.cat((bias, x), dim=2)


class BDESN(nn.Module):
    """
    Bi-directional Echo State Network module
    """

    def __init__(self, input_dim, hidden_dim, output_dim, leaky_rate=1.0, spectral_radius=0.9, bias_scaling=0, input_scaling=1.0, w=None, w_in=None, w_bias=None, sparsity=None, input_set=[1.0, -1.0], w_sparsity=None, nonlin_func=torch.tanh, learning_algo='inv', ridge_param=0.0, create_cell=True):
        """
        Constructor
        :param input_dim: Inputs dimension.
        :param hidden_dim: Hidden layer dimension
        :param output_dim: Reservoir size
        :param spectral_radius: Reservoir's spectral radius
        :param bias_scaling: Scaling of the bias, a constant input to each neuron (default: 0, no bias)
        :param input_scaling: Scaling of the input weight matrix, default 1.
        :param w: Internal weights matrix
        :param w_in: Input-reservoir weights matrix
        :param w_bias: Bias weights matrix
        :param sparsity:
        :param input_set:
        :param w_sparsity:
        :param nonlin_func: Reservoir's activation function (tanh, sig, relu)
        :param learning_algo: Which learning algorithm to use (inv, LU, grad)
        """
        super(BDESN, self).__init__()
        self.output_dim = output_dim
        if create_cell:
            self.esn_cell = BDESNCell(input_dim=input_dim, hidden_dim=hidden_dim, spectral_radius=spectral_radius, bias_scaling=bias_scaling, input_scaling=input_scaling, w=w, w_in=w_in, w_bias=w_bias, sparsity=sparsity, input_set=input_set, w_sparsity=w_sparsity, nonlin_func=nonlin_func, leaky_rate=leaky_rate, create_cell=create_cell)
        self.output = RRCell(input_dim=hidden_dim * 2, output_dim=output_dim, ridge_param=ridge_param, learning_algo=learning_algo)

    @property
    def hidden(self):
        """
        Hidden layer
        :return:
        """
        return self.esn_cell.hidden

    @property
    def w(self):
        """
        Hidden weight matrix
        :return:
        """
        return self.esn_cell.w

    @property
    def w_in(self):
        """
        Input matrix
        :return:
        """
        return self.esn_cell.w_in

    def reset(self):
        """
        Reset learning
        :return:
        """
        self.output.reset()
        self.train(True)

    def get_w_out(self):
        """
        Output matrix
        :return:
        """
        return self.output.w_out

    def set_w(self, w):
        """
        Set W
        :param w:
        :return:
        """
        self.esn_cell.w = w

    def forward(self, u, y=None):
        """
        Forward
        :param u: Input signal.
        :return: Output or hidden states
        """
        hidden_states = self.esn_cell(u)
        return self.output(hidden_states, y)

    def finalize(self):
        """
        Finalize training with LU factorization
        """
        self.output.finalize()
        self.train(False)

    def reset_hidden(self):
        """
        Reset hidden layer
        :return:
        """
        self.esn_cell.reset_hidden()

    def get_spectral_radius(self):
        """
        Get W's spectral radius
        :return: W's spectral radius
        """
        return self.esn_cell.get_spectral_raduis()


class BDESNPCA(nn.Module):
    """
    Bi-directional Echo State Network module with PCA reduction
    """

    def __init__(self, input_dim, hidden_dim, output_dim, pca_dim, linear_dim, leaky_rate=1.0, spectral_radius=0.9, bias_scaling=0, input_scaling=1.0, w=None, w_in=None, w_bias=None, sparsity=None, input_set=[1.0, -1.0], w_sparsity=None, nonlin_func=torch.tanh, learning_algo='inv', ridge_param=0.0, create_cell=True, pca_batch_size=10):
        """
        Constructor
        :param input_dim: Inputs dimension.
        :param hidden_dim: Hidden layer dimension
        :param output_dim: Reservoir size
        :param spectral_radius: Reservoir's spectral radius
        :param bias_scaling: Scaling of the bias, a constant input to each neuron (default: 0, no bias)
        :param input_scaling: Scaling of the input weight matrix, default 1.
        :param w: Internal weights matrix
        :param w_in: Input-reservoir weights matrix
        :param w_bias: Bias weights matrix
        :param sparsity:
        :param input_set:
        :param w_sparsity:
        :param nonlin_func: Reservoir's activation function (tanh, sig, relu)
        :param learning_algo: Which learning algorithm to use (inv, LU, grad)
        """
        super(BDESNPCA, self).__init__()
        self.output_dim = output_dim
        self.pca_dim = pca_dim
        if create_cell:
            self.esn_cell = BDESNCell(input_dim=input_dim, hidden_dim=hidden_dim, spectral_radius=spectral_radius, bias_scaling=bias_scaling, input_scaling=input_scaling, w=w, w_in=w_in, w_bias=w_bias, sparsity=sparsity, input_set=input_set, w_sparsity=w_sparsity, nonlin_func=nonlin_func, leaky_rate=leaky_rate, create_cell=create_cell)
        self.ipca = IncrementalPCA(n_components=pca_dim, batch_size=pca_batch_size)
        self.linear1 = nn.Linear(pca_dim, linear_dim)
        self.linear2 = nn.Linear(linear_dim, output_dim)

    @property
    def hidden(self):
        """
        Hidden layer
        :return:
        """
        return self.esn_cell.hidden

    @property
    def w(self):
        """
        Hidden weight matrix
        :return:
        """
        return self.esn_cell.w

    @property
    def w_in(self):
        """
        Input matrix
        :return:
        """
        return self.esn_cell.w_in

    def reset(self):
        """
        Reset learning
        :return:
        """
        self.output.reset()
        self.train(True)

    def get_w_out(self):
        """
        Output matrix
        :return:
        """
        return self.output.w_out

    def set_w(self, w):
        """
        Set W
        :param w:
        :return:
        """
        self.esn_cell.w = w

    def forward(self, u, y=None):
        """
        Forward
        :param u: Input signal.
        :return: Output or hidden states
        """
        hidden_states = self.esn_cell(u)
        pca_states = torch.zeros(1, hidden_states.size(1), self.pca_dim)
        pca_states[0] = torch.from_numpy(self.ipca.fit_transform(hidden_states.data[0].numpy()).copy())
        pca_states = Variable(pca_states)
        return F.relu(self.linear2(F.relu(self.linear1(pca_states))))

    def finalize(self):
        """
        Finalize training with LU factorization
        """
        self.output.finalize()
        self.train(False)

    def reset_hidden(self):
        """
        Reset hidden layer
        :return:
        """
        self.esn_cell.reset_hidden()

    def get_spectral_radius(self):
        """
        Get W's spectral radius
        :return: W's spectral radius
        """
        return self.esn_cell.get_spectral_raduis()


def generalized_squared_cosine(Sa, Ua, Sb, Ub):
    """
    Generalized square cosine
    :param Sa:
    :param Ua:
    :param Sb:
    :param Ub:
    :return:
    """
    Sa = torch.diag(Sa)
    Sb = torch.diag(Sb)
    Va = torch.sqrt(Sa).mm(Ua.t())
    Vb = Ub.mm(torch.sqrt(Sb))
    Vab = Va.mm(Vb)
    num = torch.pow(torch.norm(Vab), 2)
    den = torch.norm(torch.diag(Sa), p=2) * torch.norm(torch.diag(Sb), p=2)
    return num / den


class Conceptor(RRCell):
    """
    Conceptor
    """

    def __init__(self, conceptor_dim, aperture=0.0, learning_algo='inv', name='', conceptor_matrix=None, dtype=torch.float32):
        """
        Constructor
        :param input_dim: Inputs dimension.
        :param output_dim: Reservoir size
        """
        super(Conceptor, self).__init__(conceptor_dim, conceptor_dim, ridge_param=aperture, feedbacks=False, with_bias=False, learning_algo=learning_algo, softmax_output=False, dtype=dtype)
        self.conceptor_dim = conceptor_dim
        self.aperture = aperture
        self.name = name
        self.n_samples = 0.0
        self.attenuation = 0.0
        self.register_buffer('R', Variable(torch.zeros(self.x_size, self.x_size, dtype=self.dtype), requires_grad=False))
        self.register_buffer('C', Variable(torch.zeros(1, conceptor_dim, dtype=self.dtype), requires_grad=False))
        if conceptor_matrix is not None:
            self.C = conceptor_matrix
            self.train(False)

    @property
    def quota(self):
        """
        Compute quota
        :return:
        """
        conceptor_matrix = self.get_C()
        return float(torch.sum(conceptor_matrix.mm(torch.eye(self.conceptor_dim, dtype=self.dtype))) / self.conceptor_dim)

    def plot(self, colorstring, linewidth=3, resolution=200, dim='2d'):
        """
        Plot 2D ellipse
        :return:
        """
        if dim == '2d':
            Conceptor.plot_ellipse_2D(self.get_C(), colorstring, linewidth, resolution)
        else:
            pass

    def clone(self):
        """
        Clone
        :return:
        """
        return Conceptor(conceptor_dim=self.conceptor_dim, aperture=self.aperture, name=self.name, conceptor_matrix=self.C, dtype=self.dtype)

    def show(self):
        """
        Show the conceptor matrix
        :return:
        """
        plt.imshow(self.C, cmap='Greys')
        plt.show()

    def set_aperture(self, new_a):
        """
        Change aperture
        :param new_a:
        :return:
        """
        self.C = Conceptor.phi_function(self.C, new_a / self.aperture)
        self.aperture = new_a

    def multiply_aperture(self, factor):
        """
        Multiply aperture
        :param factor:
        :return:
        """
        self.C = Conceptor.phi_function(self.C, factor)
        self.aperture *= factor

    def plot_delta_measure(self, start, end, steps=50):
        """
        Plot delta measure
        :param start:
        :param end:
        :return:
        """
        gamma_values = torch.logspace(start=start, end=end, steps=steps)
        gamma_log_values = torch.log10(gamma_values)
        C_norms = torch.zeros(steps)
        delta_scores = torch.zeros(steps)
        for i, gamma in enumerate(gamma_values):
            delta_scores[i], C_norms[i] = self.delta_measure(float(gamma), epsilon=0.1)
        plt.plot(gamma_log_values.numpy(), delta_scores.numpy())
        plt.plot(gamma_log_values.numpy(), C_norms.numpy())
        plt.show()

    def delta_measure(self, gamma, epsilon=0.01):
        """
        Compute Delta measure
        :param gamma:
        :param epsilon:
        :return:
        """
        A = Conceptor.phi_function(self.C, gamma - epsilon)
        B = Conceptor.phi_function(self.C, gamma + epsilon)
        A_norm = math.pow(torch.norm(A, p=2), 2)
        B_norm = math.pow(torch.norm(B, p=2), 2)
        d_C_norm = B_norm - A_norm
        d_log_gamma = np.log(gamma + epsilon) - np.log(gamma - epsilon)
        """if d_C_norm / d_log_gamma > 50.0:
            print(A)
            print(B)
            print(torch.norm(A, p=2))
            print(torch.norm(B, p=2))
            print(d_C_norm)
            print(gamma)
            print(d_C_norm / d_log_gamma)
            exit()
        # end if"""
        return d_C_norm / d_log_gamma, d_C_norm

    def get_C(self):
        """
        Output matrix
        :return:
        """
        return self.C

    def forward(self, x, y=None):
        """
        Forward
        :param x: Input signal.
        :param y: Target outputs
        :return: Output or hidden states
        """
        batch_size = x.size(0)
        time_length = x.size(1)
        if self.training:
            for b in range(batch_size):
                Rj = x[b].t().mm(x[b]) / time_length
                self.R.data.add_(Rj.data)
                self.n_samples += 1.0
            return x
        elif not self.training:
            outputs = Variable(torch.zeros(batch_size, time_length, self.output_dim, dtype=self.dtype), requires_grad=False)
            outputs = outputs if self.C.is_cuda else outputs
            for b in range(batch_size):
                outputs[b] = torch.mm(x[b], self.C)
            self.attenuation = torch.mean(torch.pow(torch.abs(x - outputs), 2)) / torch.mean(torch.pow(torch.abs(x), 2))
            return outputs

    def finalize(self):
        """
        Finalize training with LU factorization or Pseudo-inverse
        """
        self.R = self.R / self.n_samples
        U, S, V = torch.svd(self.R)
        Snew = torch.mm(torch.diag(S), torch.inverse(torch.diag(S) + math.pow(self.aperture, -2) * torch.eye(self.input_dim, dtype=self.dtype)))
        self.C.data = torch.mm(torch.mm(U, Snew), U.t()).data
        self.train(False)

    def set_conceptor(self, c):
        """
        Set conceptor
        :param c:
        :return:
        """
        self.w_out.data = c

    def singular_values(self):
        """
        Singular values
        :return:
        """
        Ua, Sa, Va = torch.svd(self.get_C())
        return Ua, torch.diag(Sa), Va

    def get_quota(self):
        """
        Sum of singular values
        :return:
        """
        return float(torch.sum(self.singular_values()))

    @staticmethod
    def plot_ellipse_2D(A, colorstring, linewidth=3, resolution=200):
        """
        Plot 2D ellipse
        :return:
        """
        plt.plot([-1, 1], [0, 0], '--', color='black', linewidth=1)
        plt.plot([0, 0], [-1, 1], '--', color='black', linewidth=1)
        plt.plot(np.cos(2.0 * math.pi * np.arange(200) / 200.0), np.sin(2.0 * math.pi * np.arange(200) / 200), '-', color='black', linewidth=1)
        circ_points = torch.from_numpy(np.array([np.cos(2.0 * math.pi * np.arange(0, resolution) / resolution), np.sin(2.0 * math.pi * np.arange(0, resolution) / resolution)]))
        E1 = torch.mm(A, circ_points)
        U, S, Ut = torch.svd(A)
        plt.plot(S[0].item() * np.array([0.0, U[0, 0]]), S[0].item() * np.array([0.0, U[1, 0]]), linewidth=linewidth, color=colorstring)
        plt.plot(S[1].item() * np.array([0.0, U[0, 1]]), S[1].item() * np.array([0.0, U[1, 1]]), linewidth=linewidth, color=colorstring)
        plt.plot(E1[(0), :].numpy(), E1[(1), :].numpy(), linewidth=linewidth, color=colorstring)

    @staticmethod
    def phi_function(C, gamma):
        """
        Multiply aperture matrix
        :param c:
        :param gamma:
        :return:
        """
        c = C.clone()
        conceptor_dim = c.shape[0]
        dtype = c.dtype
        return c.mm(torch.inverse(c + m.pow(gamma, -2) * (torch.eye(conceptor_dim, dtype=dtype) - c)))

    @staticmethod
    def morphing(conceptor_list, mu):
        """
        Morphing pattern
        :param conceptor_list:
        :return:
        """
        for i, c in enumerate(conceptor_list):
            if i == 0:
                M = c.mul(mu[i])
            else:
                M += c.mul(mu[i])
        return M

    @staticmethod
    def similarity(C1, C2):
        """
        Similarity between two conceptors
        :param C1:
        :param C2:
        :return:
        """
        Ua, Sa, _ = torch.svd(C1.get_C())
        Ub, Sb, _ = torch.svd(C2.get_C())
        return generalized_squared_cosine(Sa, Ua, Sb, Ub)

    def sim(self, cb, measure='gsc'):
        """
        Similarity with another conceptor
        :param cb:
        :return:
        """
        Ua, Sa, _ = torch.svd(self.C)
        Ub, Sb, _ = torch.svd(cb.get_C())
        if measure == 'gsc':
            return generalized_squared_cosine(Sa, Ua, Sb, Ub)

    def E_plus(self, x):
        """
        Positive evidence
        :param x: states (x)
        :return:
        """
        return x.mm(self.w_out).mm(x.t())

    def E_neg(self, x, conceptor_list):
        """
        Evidence against
        :param x:
        :param conceptor_list:
        :return:
        """
        for i, c in enumerate(conceptor_list):
            if i == 0:
                new_c = c
            else:
                new_c = new_c.logical_or(c)
        N = new_c.logical_not()
        return x.t().mm(N.w_out).mm(x)

    def E(self, x, conceptor_list):
        """
        Evidence
        :param x:
        :param conceptor_list:
        :return:
        """
        return self.E_plus(x) + self.E_neg(x, conceptor_list)

    def logical_or(self, c):
        """
        Logical OR
        :param c:
        :return:
        """
        C = self.C
        B = c.get_C()
        I = torch.eye(self.conceptor_dim, dtype=self.dtype)
        conceptor_matrix = torch.inverse(I + torch.inverse(C.mm(torch.inverse(I - C)) + B.mm(torch.inverse(I - B))))
        new_c = Conceptor(conceptor_dim=self.conceptor_dim, conceptor_matrix=conceptor_matrix, name='({} OR {})'.format(self.name, c.name), aperture=math.sqrt(math.pow(self.aperture, 2) + math.pow(c.aperture, 2)), dtype=self.dtype)
        return new_c

    def __or__(self, other):
        """
        OR
        :param other:
        :return:
        """
        return self.logical_or(other)

    def logical_not(self):
        """
        Logical NOT
        :param c:
        :return:
        """
        C = self.C
        conceptor_matrix = torch.eye(self.conceptor_dim, dtype=self.dtype) - C
        new_c = Conceptor(conceptor_dim=self.conceptor_dim, conceptor_matrix=conceptor_matrix, name='NOT {}'.format(self.name), aperture=1.0 / self.aperture, dtype=self.dtype)
        return new_c

    def __invert__(self):
        """
        NOT
        :return:
        """
        return self.logical_not()

    def logical_and(self, C2):
        """
        Logical AND
        :param c:
        :return:
        """
        A = self.get_C()
        B = C2.get_C()
        dim = A.shape[0]
        tol = 1e-14
        UC, SC, UtC = torch.svd(A)
        UB, SB, UtB = torch.svd(B)
        dSC = SC
        dSB = SB
        numRankC = int(torch.sum(1.0 * (dSC > tol)))
        numRankB = int(torch.sum(1.0 * (dSB > tol)))
        if numRankC < dim and numRankB < dim:
            UC0 = UC[:, numRankC:]
            UB0 = UB[:, numRankB:]
            W, Sigma, Wt = torch.svd(torch.mm(UC0, UC0.t()) + torch.mm(UB0, UB0.t()))
            numRankSigma = int(torch.sum(1.0 * (Sigma > tol)))
            Wgk = W[:, numRankSigma:]
            CandB = np.dot(np.dot(Wgk, torch.inverse(np.dot(np.dot(Wgk.T, torch.pinverse(A, tol) + torch.pinverse(B, tol) - np.eye(dim)), Wgk))), Wgk.T)
        else:
            CandB = torch.pinverse(A, tol) + torch.pinverse(B, tol) - torch.eye(dim, dtype=self.dtype)
        new_c = Conceptor(conceptor_dim=self.conceptor_dim, conceptor_matrix=CandB, name='({} AND {})'.format(self.name, C2.name), aperture=math.pow(math.pow(self.aperture, -2) + math.pow(C2.aperture, -2), -0.5), dtype=self.dtype)
        return new_c

    def __and__(self, other):
        """
        AND
        :param other:
        :return:
        """
        return self.logical_and(other)

    def mul(self, other):
        """
        Multiply
        :param other:
        :return:
        """
        if type(other) is Conceptor:
            new_c = self.get_C() * other.get_C()
        else:
            new_c = self.get_C() * other
        return Conceptor(self.conceptor_dim, self.aperture, conceptor_matrix=new_c, dtype=self.dtype)

    def __mul__(self, other):
        """
        Multiply
        :param other:
        :return:
        """
        if type(other) is Conceptor:
            new_c = self.get_C() * other.get_C()
        else:
            new_c = self.get_C() * other
        return Conceptor(self.conceptor_dim, self.aperture, conceptor_matrix=new_c, dtype=self.dtype)

    def __rmul__(self, other):
        """
        Multiply
        :param other:
        :return:
        """
        if type(other) is Conceptor:
            new_c = self.get_C() * other.get_C()
        else:
            new_c = self.get_C() * other
        return Conceptor(self.conceptor_dim, self.aperture, conceptor_matrix=new_c, dtype=self.dtype)

    def __imul__(self, other):
        """
        *=
        :param other:
        :return:
        """
        if type(other) is Conceptor:
            new_c = self.get_C() * other.get_C()
        else:
            new_c = self.get_C() * other
        return Conceptor(self.conceptor_dim, self.aperture, conceptor_matrix=new_c, dtype=self.dtype)

    def __add__(self, other):
        """
        Add
        :param other:
        :return:
        """
        if type(other) is Conceptor:
            new_c = self.get_C() + other.get_C()
        else:
            new_c = self.get_C() + other
        return Conceptor(self.conceptor_dim, self.aperture, conceptor_matrix=new_c, dtype=self.dtype)

    def __radd__(self, other):
        """
        Add
        :param other:
        :return:
        """
        if type(other) is Conceptor:
            new_c = self.get_C() + other.get_C()
        else:
            new_c = self.get_C() + other
        return Conceptor(self.conceptor_dim, self.aperture, conceptor_matrix=new_c, dtype=self.dtype)

    def __iadd__(self, other):
        """
        +=
        :param other:
        :return:
        """
        if type(other) is Conceptor:
            new_c = self.get_C() + other.get_C()
        else:
            new_c = self.get_C() + other
        return Conceptor(self.conceptor_dim, self.aperture, conceptor_matrix=new_c, dtype=self.dtype)

    def __ge__(self, other):
        """
        Greater or equal
        :param other:
        :return:
        """
        eig_v = torch.eig(other.get_C() - self.w_out, eigenvectors=False)
        return float(torch.max(eig_v)) >= 0.0

    def __gt__(self, other):
        """
        Greater
        :param other:
        :return:
        """
        eig_v = torch.eig(other.get_C() - self.w_out, eigenvectors=False)
        return float(torch.max(eig_v)) > 0.0

    def __lt__(self, other):
        """
        Less than
        :param other:
        :return:
        """
        return not self >= other

    def __le__(self, other):
        """
        Less or equal
        :param other:
        :return:
        """
        return not self > other


class ConceptorPool(object):
    """
    ConceptorPool
    """

    def __init__(self, conceptor_dim, conceptors=list(), esn=None, dtype=torch.float32):
        """
        Constructor
        :param conceptors:
        """
        self.conceptor_dim = conceptor_dim
        self.conceptors = conceptors
        self.name_to_conceptor = {}
        self.esn = esn
        self.dtype = dtype

    @property
    def A_SV(self):
        """
        Singular values of A
        :return:
        """
        return ConceptorPool.compute_A_SV(self.conceptors)

    @property
    def A(self):
        return ConceptorPool.compute_A(self.conceptors)

    @property
    def quota(self):
        """
        Quota
        :return:
        """
        return ConceptorPool.compute_quota(self.conceptors)

    def similarity_matrix(self):
        """
        Get similarity matrix
        :return:
        """
        return ConceptorPool.compute_similarity_matrix(self.conceptors)

    def finalize_conceptor(self, i):
        """
        Finalize conceptor
        :param i:
        :return:
        """
        self.conceptors[i].finalize()

    def finalize(self):
        """
        Finalize all conceptors
        :return:
        """
        for c in self.conceptors:
            c.finalize()

    def E_plus(self, p):
        """
        Positive evidence
        :param x: states (x)
        :return:
        """
        return ConceptorPool.compute_E_plus(self.conceptors, self.esn, p)

    def E_other(self, p):
        """
        Evidence for other
        :param p:
        :return:
        """
        batch_size = p.shape[0]
        time_length = p.shape[1]
        evidences = torch.zeros(batch_size, time_length)
        x = self.esn(u=p, return_states=True)
        A = self.A
        N = A.logical_not()
        for b in range(batch_size):
            for t in range(time_length):
                evidences[b, t] = torch.mm(x[b, t].view(1, -1), N.get_C()).mm(x[b, t].view(-1, 1))
        return torch.mean(evidences, dim=1)

    def E_neg(self, p):
        """
        Negative evidence
        :param p:
        :return:
        """
        batch_size = p.shape[0]
        n_conceptors = len(self.conceptors)
        time_length = p.shape[1]
        evidences = torch.zeros(batch_size, time_length, n_conceptors)
        x = self.esn(u=p, return_states=True)
        for b in range(batch_size):
            for i, c in enumerate(self.conceptors):
                other_c = list(self.conceptors)
                other_c.remove(c)
                A = ConceptorPool.compute_A(other_c)
                N = A.logical_not()
                for t in range(time_length):
                    evidences[b, t, i] = torch.mm(x[b, t].view(1, -1), N.get_C()).mm(x[b, t].view(-1, 1))
        return torch.mean(evidences, dim=1)

    def E(self, p):
        """
        Evidence for each conceptor
        :return:
        """
        return (self.E_plus(p) + self.E_neg(p)) / 2.0

    def add(self, aperture, name):
        """
        New conceptor
        :param aperture: Aperture
        :param name: Conceptor's name
        :return: New conceptor
        """
        new_conceptor = Conceptor(self.conceptor_dim, aperture=aperture, name=name, dtype=self.dtype)
        self.conceptors.append(new_conceptor)
        self.name_to_conceptor[name] = new_conceptor
        return new_conceptor

    def add_or(self, i, j):
        """
        Add an OR between conceptors
        :param i:
        :param j:
        :return:
        """
        self.append(self.conceptors[i].logical_or(self.conceptors[j]))

    def add_and(self, i, j):
        """
        Add an AND between conceptors
        :param i:
        :param j:
        :return:
        """
        self.append(self.conceptors[i].logical_and(self.conceptors[j]))

    def add_not(self, i):
        """
        Add an OR between conceptors
        :param i:
        :return:
        """
        self.append(self.conceptors[i].logical_not())

    def add_A(self):
        """
        Add an OR between conceptors
        :param i:
        :return:
        """
        A = ConceptorPool.compute_A(self.conceptors)
        self.append(A)

    def add_Not_A(self):
        """
        Add an OR between conceptors
        :param i:
        :return:
        """
        A = ConceptorPool.compute_A(self.conceptors)
        N = A.logical_not()
        self.append(N)

    def append(self, c):
        """
        Append a conceptor
        :param c:
        :return:
        """
        self.conceptors.append(c)
        self.name_to_conceptor[c.name] = c

    def morphing(self, mu):
        """
        Morphing pattern
        :param conceptor_list:
        :return:
        """
        for i, c in enumerate(self.conceptors):
            if i == 0:
                M = c.mul(mu[i])
            else:
                M += c.mul(mu[i])
        return M

    def __getitem__(self, item):
        """
        Get item
        :param item:
        :return:
        """
        if type(item) is int:
            return self.conceptors[item]
        elif type(item) is str:
            return self.name_to_conceptor[item]

    def __setitem__(self, key, value):
        """
        Set item
        :param key:
        :param value:
        :return:
        """
        self.conceptors[key] = value

    def __len__(self):
        """
        Length
        :return:
        """
        return len(self.conceptors)

    @staticmethod
    def compute_similarity_matrix(conceptors):
        """
        Get similarity matrix
        :return:
        """
        sim_matrix = torch.zeros(len(conceptors), len(conceptors))
        for i, ca in enumerate(conceptors):
            for j, cb in enumerate(conceptors):
                sim_matrix[i, j] = ca.sim(cb)
        return sim_matrix

    @staticmethod
    def compute_E_plus(conceptors, esn, p):
        """
        Positive evidence
        :param x: states (x)
        :return:
        """
        batch_size = p.shape[0]
        n_conceptors = len(conceptors)
        time_length = p.shape[1]
        evidences = torch.zeros(batch_size, time_length, n_conceptors)
        x = esn(u=p, return_states=True)
        for b in range(batch_size):
            for t in range(time_length):
                for i, c in enumerate(conceptors):
                    evidences[b, t, i] = torch.mm(x[b, t].view(1, -1), c.get_C()).mm(x[b, t].view(-1, 1))
        return torch.mean(evidences, dim=1)

    @staticmethod
    def compute_A_SV(conceptors):
        """
        Get singular values of A
        :param conceptors:
        :return:
        """
        A = ConceptorPool.compute_A(conceptors)
        _, S, _ = torch.svd(A.get_C())
        return S

    @staticmethod
    def compute_A(conceptors):
        """
        Compute A (OR of all conceptors)
        :param conceptors:
        :return:
        """
        for i, c in enumerate(conceptors):
            if i == 0:
                A = c
            else:
                A = c.logical_or(A)
        A.name = 'A'
        return A

    @staticmethod
    def compute_quota(conceptors):
        """
        Compute quota
        :param conceptors:
        :return:
        """
        S = ConceptorPool.compute_A_SV(conceptors)
        return float(torch.mean(S))


class ConceptorNetCell(LiESNCell):
    """
    Special reservoir layer for Conceptors
    """

    def __init__(self, *args, **kwargs):
        """
        Constructor
        :param leaky_rate: Reservoir's leaky rate (default 1.0, normal ESN)
        :param train_leaky_rate: Train leaky rate as parameter? (default: False)
        """
        super(ConceptorNetCell, self).__init__(*args, **kwargs)

    def forward(self, u=None, y=None, w_out=None, reset_state=True, input_recreation=None, conceptor=None, length=None, mu=None, x0=None):
        """
        Forward execution
        :param u:
        :param y:
        :param w_out:
        :param reset_state:
        :param generative_mode:
        :return:
        """
        if u is not None:
            time_length = int(u.size()[1])
        else:
            time_length = length
        if u is not None:
            n_batches = int(u.size()[0])
        else:
            n_batches = 1
        outputs = Variable(torch.zeros(n_batches, time_length, self.output_dim, dtype=self.dtype))
        outputs = outputs if self.hidden.is_cuda else outputs
        for b in range(n_batches):
            if x0 is not None:
                self.set_hidden(x0)
            elif reset_state:
                self.reset_hidden()
            for t in range(time_length):
                if self.training:
                    ut = u[b, t]
                    u_win = self.w_in.mv(ut)
                    x_w = self.w.mv(self.hidden)
                    x = u_win + x_w + self.w_bias
                    x = self.nonlin_func(x)
                    self.hidden.data = x.view(-1).data
                    outputs[b, t] = self.hidden
                else:
                    if u is not None:
                        ut = u[b, t]
                        u_win = self.w_in.mv(ut)
                        x_w = self.w.mv(self.hidden)
                        x = u_win + x_w + self.w_bias
                    else:
                        x_w = input_recreation(self.hidden.view(1, 1, -1))
                        x = x_w + self.w_bias
                    x = self.nonlin_func(x)
                    if type(conceptor) is Conceptor:
                        xc = conceptor(x.view(1, 1, -1)).view(-1)
                    elif type(conceptor) is ConceptorPool:
                        M = conceptor.morphing(mu[b, t])
                        xc = M(x.view(1, 1, -1)).view(-1)
                    else:
                        xc = x.view(-1)
                    self.hidden.data = xc.data
                    outputs[b, t] = self.hidden
        return outputs


class ConceptorNet(nn.Module):
    """
    ESN-based ConceptorNet
    """

    def __init__(self, input_dim, hidden_dim, output_dim=None, spectral_radius=0.9, bias_scaling=0, input_scaling=1.0, w=None, w_in=None, w_bias=None, sparsity=None, input_set=[1.0, -1.0], w_sparsity=None, leaky_rate=1.0, nonlin_func=torch.tanh, learning_algo='inv', ridge_param=0.0, with_bias=True, seed=None, washout=1, w_distrib='uniform', win_distrib='uniform', wbias_distrib='uniform', win_normal=(0.0, 1.0), w_normal=(0.0, 1.0), wbias_normal=(0.0, 1.0), w_ridge_param=0.0, dtype=torch.float32):
        """
        Constructor
        :param input_dim: Inputs dimension.
        :param hidden_dim: Hidden layer dimension
        :param spectral_radius: Reservoir's spectral radius
        :param bias_scaling: Scaling of the bias, a constant input to each neuron (default: 0, no bias)
        :param input_scaling: Scaling of the input weight matrix, default 1.
        :param w: Internation weights matrix
        :param w_in: Input-reservoir weights matrix
        :param w_bias: Bias weights matrix
        :param w_fdb: Feedback weights matrix
        :param sparsity:
        :param input_set:
        :param w_sparsity:
        :param nonlin_func: Reservoir's activation function (tanh, sig, relu)
        :param learning_algo: Which learning algorithm to use (inv, LU, grad)
        """
        super(ConceptorNet, self).__init__()
        self.with_bias = with_bias
        self.washout = washout
        self.hidden_dim = hidden_dim
        self.esn_cell = ConceptorNetCell(leaky_rate, False, input_dim, hidden_dim, spectral_radius=spectral_radius, bias_scaling=bias_scaling, input_scaling=input_scaling, w=w, w_in=w_in, w_bias=w_bias, sparsity=sparsity, input_set=input_set, w_sparsity=w_sparsity, nonlin_func=nonlin_func, feedbacks=False, feedbacks_dim=input_dim, wfdb_sparsity=None, normalize_feedbacks=False, seed=seed, w_distrib=w_distrib, win_distrib=win_distrib, wbias_distrib=wbias_distrib, win_normal=win_normal, w_normal=w_normal, wbias_normal=wbias_normal, dtype=dtype)
        self.input_recreation = RRCell(hidden_dim, hidden_dim, w_ridge_param, None, with_bias=False, learning_algo=learning_algo, softmax_output=False, averaged=True, dtype=dtype)
        if output_dim is not None:
            self.output = RRCell(hidden_dim, output_dim, ridge_param, None, with_bias=False, learning_algo=learning_algo, softmax_output=False, averaged=True, dtype=dtype)
        else:
            self.output = None

    @property
    def hidden(self):
        """
        Hidden layer
        :return:
        """
        return self.esn_cell.hidden

    @property
    def w(self):
        """
        Hidden weight matrix
        :return:
        """
        return self.esn_cell.w

    @property
    def w_in(self):
        """
        Input matrix
        :return:
        """
        return self.esn_cell.w_in

    @property
    def input_recreation_matrix(self):
        """
        Input recreation matrix
        :return:
        """
        return self.input_recreation.get_w_out()

    def arctanh(self, x):
        """
        Inverse tanh
        :param x:
        :return:
        """
        return 0.5 * torch.log((1 + x) / (1 - x))

    def set_train(self):
        """
        Mode
        :return:
        """
        self.train(True)
        self.input_recreation.train(True)
        if self.output is not None:
            self.output.train(True)

    def reset(self):
        """
        Reset learning
        :return:
        """
        self.output.reset()
        self.train(True)

    def get_w_out(self):
        """
        Output matrix
        :return:
        """
        return self.output.w_out

    def set_w(self, w):
        """
        Set W
        :param w:
        :return:
        """
        self.esn_cell.w = w

    def forward(self, u=None, y=None, c=None, reset_state=True, length=None, mu=None, return_states=False, x0=None):
        """
        Forward
        :param u: Input signal.
        :param y: Target outputs
        :return: Output or hidden states
        """
        if self.training:
            hidden_states = self.esn_cell(u, reset_state=reset_state, x0=x0)
            batch_size = hidden_states.shape[0]
            x = hidden_states[:, self.washout:]
            time_length = x.shape[1]
            x_tilda = hidden_states[:, self.washout - 1:-1]
            bias = self.esn_cell.w_bias[0].expand(batch_size, time_length, self.hidden_dim)
            self.input_recreation(x_tilda, self.arctanh(x) - bias)
            if self.output is not None and y is not None:
                self.output(x, y[:, self.washout:])
            return c(x)
        elif c is None:
            hidden_states = self.esn_cell(u, reset_state=reset_state, x0=x0)
            if self.output is not None and not return_states:
                return self.output(hidden_states)
            else:
                return hidden_states
        else:
            hidden_states = self.esn_cell(u=u, reset_state=reset_state, input_recreation=self.input_recreation, conceptor=c, length=length, mu=mu, x0=x0)
            if self.output is not None:
                return self.output(hidden_states)
            else:
                return self.input_recreation(hidden_states)

    def finalize(self, train=False):
        """
        Finalize training with LU factorization
        """
        self.input_recreation.finalize()
        if self.output is not None:
            self.output.finalize()
        self.train(train)

    def reset_hidden(self):
        """
        Reset hidden layer
        :return:
        """
        self.esn_cell.reset_hidden()

    def get_spectral_radius(self):
        """
        Get W's spectral radius
        :return: W's spectral radius
        """
        return self.esn_cell.get_spectral_raduis()


class ESN(nn.Module):
    """
    Echo State Network module
    """

    def __init__(self, input_dim, hidden_dim, output_dim, spectral_radius=0.9, bias_scaling=0, input_scaling=1.0, w=None, w_in=None, w_bias=None, w_fdb=None, sparsity=None, input_set=[1.0, -1.0], w_sparsity=None, nonlin_func=torch.tanh, learning_algo='inv', ridge_param=0.0, create_cell=True, feedbacks=False, with_bias=True, wfdb_sparsity=None, normalize_feedbacks=False, softmax_output=False, seed=None, washout=0, w_distrib='uniform', win_distrib='uniform', wbias_distrib='uniform', win_normal=(0.0, 1.0), w_normal=(0.0, 1.0), wbias_normal=(0.0, 1.0), dtype=torch.float32):
        """
        Constructor
        :param input_dim: Inputs dimension.
        :param hidden_dim: Hidden layer dimension
        :param output_dim: Reservoir size
        :param spectral_radius: Reservoir's spectral radius
        :param bias_scaling: Scaling of the bias, a constant input to each neuron (default: 0, no bias)
        :param input_scaling: Scaling of the input weight matrix, default 1.
        :param w: Internation weights matrix
        :param w_in: Input-reservoir weights matrix
        :param w_bias: Bias weights matrix
        :param w_fdb: Feedback weights matrix
        :param sparsity:
        :param input_set:
        :param w_sparsity:
        :param nonlin_func: Reservoir's activation function (tanh, sig, relu)
        :param learning_algo: Which learning algorithm to use (inv, LU, grad)
        """
        super(ESN, self).__init__()
        self.output_dim = output_dim
        self.feedbacks = feedbacks
        self.with_bias = with_bias
        self.normalize_feedbacks = normalize_feedbacks
        self.washout = washout
        self.dtype = dtype
        if create_cell:
            self.esn_cell = ESNCell(input_dim, hidden_dim, spectral_radius, bias_scaling, input_scaling, w, w_in, w_bias, w_fdb, sparsity, input_set, w_sparsity, nonlin_func, feedbacks, output_dim, wfdb_sparsity, normalize_feedbacks, seed, w_distrib, win_distrib, wbias_distrib, win_normal, w_normal, wbias_normal, dtype)
        self.output = RRCell(hidden_dim, output_dim, ridge_param, feedbacks, with_bias, learning_algo, softmax_output, dtype)

    @property
    def hidden(self):
        """
        Hidden layer
        :return:
        """
        return self.esn_cell.hidden

    @property
    def w(self):
        """
        Hidden weight matrix
        :return:
        """
        return self.esn_cell.w

    @property
    def w_in(self):
        """
        Input matrix
        :return:
        """
        return self.esn_cell.w_in

    def reset(self):
        """
        Reset learning
        :return:
        """
        self.output.reset()
        self.train(True)

    def get_w_out(self):
        """
        Output matrix
        :return:
        """
        return self.output.w_out

    def set_w(self, w):
        """
        Set W
        :param w:
        :return:
        """
        self.esn_cell.w = w

    def forward(self, u, y=None, reset_state=True):
        """
        Forward
        :param u: Input signal.
        :param y: Target outputs
        :return: Output or hidden states
        """
        if self.feedbacks and self.training:
            hidden_states = self.esn_cell(u, y, reset_state=reset_state)
        elif self.feedbacks and not self.training:
            hidden_states = self.esn_cell(u, w_out=self.output.w_out, reset_state=reset_state)
        else:
            hidden_states = self.esn_cell(u, reset_state=reset_state)
        if y is not None:
            return self.output(hidden_states[:, self.washout:], y[:, self.washout:])
        else:
            return self.output(hidden_states[:, self.washout:], y)

    def finalize(self):
        """
        Finalize training with LU factorization
        """
        self.output.finalize()
        self.train(False)

    def reset_hidden(self):
        """
        Reset hidden layer
        :return:
        """
        self.esn_cell.reset_hidden()

    def get_spectral_radius(self):
        """
        Get W's spectral radius
        :return: W's spectral radius
        """
        return self.esn_cell.get_spectral_raduis()


class PCACell(nn.Module):
    """
    Filter the input data through the most significatives principal components
    """

    def __init__(self, input_dim, output_dim, svd=False, reduce=False, var_rel=1e-12, var_abs=1e-15, var_part=None):
        """
        Constructor
        :param input_dim:
        :param output_dim:
        :param svd: If True use Singular Value Decomposition instead of the standard eigenvalue problem solver. Use it when PCANode complains about singular covariance matrices.
        :param reduce: Keep only those principal components which have a variance larger than 'var_abs'
        :param val_rel: Variance relative to first principal component threshold. Default is 1E-12.
        :param var_abs: Absolute variance threshold. Default is 1E-15.
        :param var_part: Variance relative to total variance threshold. Default is None.
        """
        super(PCACell, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.svd = svd
        self.var_abs = var_abs
        self.var_rel = var_rel
        self.var_part = var_part
        self.reduce = reduce
        self.register_buffer('xTx', Variable(torch.zeros(input_dim, input_dim), requires_grad=False))
        self.register_buffer('xTx_avg', Variable(torch.zeros(input_dim), requires_grad=False))
        self.d = None
        self.v = None
        self.total_variance = None
        self.tlen = 0
        self.avg = None
        self.explained_variance = None

    def reset(self):
        """
        Reset learning
        :return:
        """
        self._init_internals()
        self.train(True)

    def forward(self, x, y=None):
        """
        Forward
        :param x: Input signal.
        :param y: Target outputs
        :return: Output or hidden states
        """
        n_batches = int(x.size()[0])
        time_length = x.size()[1]
        outputs = Variable(torch.zeros(n_batches, time_length, self.output_dim))
        outputs = outputs if x.is_cuda else outputs
        for b in range(n_batches):
            s = x[b]
            if self.training:
                self._update_cov_matrix(s)
            else:
                outputs[b] = self._execute_pca(s)
        return outputs

    def finalize(self):
        """
        Finalize training with LU factorization or Pseudo-inverse
        """
        xTx, avg, tlen = self._fix(self.xTx, self.xTx_avg, self.tlen)
        self.avg = avg.unsqueeze(0)
        if self.tlen < self.input_dim:
            raise Exception(u'The number of observations ({}) is larger than  the number of input variables ({})'.format(self.tlen, self.input_dim))
        total_var = torch.diag(xTx).sum()
        d, v = torch.symeig(xTx, eigenvectors=True)
        if float(d.min()) < 0:
            pass
        indexes = range(d.size(0) - 1, -1, -1)
        d = torch.take(d, Variable(torch.LongTensor(indexes)))
        v = v[:, (indexes)]
        self.explained_variance = torch.sum(d) / total_var
        self.d = d[:self.output_dim]
        self.v = v[:, :self.output_dim]
        self.total_variance = total_var
        self.train(False)

    def get_explained_variance(self):
        """
        The explained variance is the fraction of the original variance that can be explained by the
        principal components.
        :return:
        """
        return self.explained_variance

    def get_proj_matrix(self, tranposed=True):
        """
        Get the projection matrix
        :param tranposed:
        :return:
        """
        self.train(False)
        if tranposed:
            return self.v
        return self.v.t()

    def get_rec_matrix(self, tranposed=1):
        """
        Returns the reconstruction matrix
        :param tranposed:
        :return:
        """
        self.train(False)
        if tranposed:
            return self.v.t()
        return self.v

    def _execute_pca(self, x, n=None):
        """
        Project the input on the first 'n' principal components
        :param x:
        :param n:
        :return:
        """
        if n is not None:
            return (x - self.avg).mm(self.v[:, :n])
        return (x - self.avg).mm(self.v)

    def _inverse(self, y, n=None):
        """
        Project data from the output to the input space using the first 'n' components.
        :param y:
        :param n:
        :return:
        """
        if n is None:
            n = y.shape[1]
        if n > self.output_dim:
            raise Exception(u'y has dimension {} but should but at most {}'.format(n, self.output_dim))
        v = self.get_rec_matrix()
        if n is not None:
            return y.mm(v[:n, :]) + self.avg
        else:
            return y.mm(v) + self.avg

    def _adjust_output_dim(self):
        """
        If the output dimensions is small than the input dimension
        :return:
        """
        if self.desired_variance is None and self.ouput_dim is None:
            self.output_dim = self.input_dim
            return None
        if self.output_dim is not None and self.output_dim >= 1:
            return self.input_dim - self.output_dim + 1, self.input_dim
        else:
            return None

    def _fix(self, mtx, avg, tlen, center=True):
        """
        Returns a triple containing the covariance matrix, the average and
        the number of observations.
        :param mtx:
        :param center:
        :return:
        """
        mtx /= tlen - 1
        if center:
            avg_mtx = torch.ger(avg, avg)
            avg_mtx /= tlen * (tlen - 1)
            mtx -= avg_mtx
        avg /= tlen
        return mtx, avg, tlen

    def _update_cov_matrix(self, x):
        """
        Update covariance matrix
        :param x:
        :return:
        """
        if self.xTx is None:
            self._init_internals()
        self.xTx.data.add_(x.t().mm(x).data)
        self.xTx_avg.add_(torch.sum(x, dim=0))
        self.tlen += x.size(0)

    def _init_cov_matrix(self):
        """
        Initialize covariance matrix
        :return:
        """
        self.xTx.data = torch.zeros(self.input_dim, self.input_dim)
        self.xTx_avg.data = torch.zeros(self.input_dim)

    def _init_internals(self):
        """
        Initialize internals
        :param x:
        :return:
        """
        self._init_cov_matrix()

    def _add_constant(self, x):
        """
        Add constant
        :param x:
        :return:
        """
        bias = Variable(torch.ones((x.size()[0], x.size()[1], 1)), requires_grad=False)
        return torch.cat((bias, x), dim=2)


class GatedESN(nn.Module):
    """
    Gated Echo State Network
    """

    def __init__(self, input_dim, reservoir_dim, pca_dim, hidden_dim, leaky_rate=1.0, spectral_radius=0.9, bias_scaling=0, input_scaling=1.0, w=None, w_in=None, w_bias=None, sparsity=None, input_set=[1.0, -1.0], w_sparsity=None, nonlin_func=torch.tanh, create_cell=True):
        """
        Constructor
        :param input_dim: Inputs dimension.
        :param hidden_dim: Hidden layer dimension
        :param reservoir_dim: Reservoir size
        :param spectral_radius: Reservoir's spectral radius
        :param bias_scaling: Scaling of the bias, a constant input to each neuron (default: 0, no bias)
        :param input_scaling: Scaling of the input weight matrix, default 1.
        :param w: Internal weights matrix
        :param w_in: Input-reservoir weights matrix
        :param w_bias: Bias weights matrix
        :param sparsity:
        :param input_set:
        :param w_sparsity:
        :param nonlin_func: Reservoir's activation function (tanh, sig, relu)
        :param learning_algo: Which learning algorithm to use (inv, LU, grad)
        """
        super(GatedESN, self).__init__()
        self.reservoir_dim = reservoir_dim
        self.pca_dim = pca_dim
        self.hidden_dim = hidden_dim
        self.finalized = False
        if create_cell:
            self.esn_cell = LiESNCell(input_dim=input_dim, output_dim=reservoir_dim, spectral_radius=spectral_radius, bias_scaling=bias_scaling, input_scaling=input_scaling, w=w, w_in=w_in, w_bias=w_bias, sparsity=sparsity, input_set=input_set, w_sparsity=w_sparsity, nonlin_func=nonlin_func, leaky_rate=leaky_rate)
        if self.pca_dim > 0:
            self.pca_cell = PCACell(input_dim=reservoir_dim, output_dim=pca_dim)
        self.register_parameter('wzp', nn.Parameter(self.init_wzp()))
        self.register_parameter('wzh', nn.Parameter(self.init_wzh()))
        self.register_parameter('bz', nn.Parameter(self.init_bz()))

    @property
    def hidden(self):
        """
        Hidden layer
        :return:
        """
        return self.esn_cell.hidden

    @property
    def w(self):
        """
        Hidden weight matrix
        :return:
        """
        return self.esn_cell.w

    @property
    def w_in(self):
        """
        Input matrix
        :return:
        """
        return self.esn_cell.w_in

    def init_hidden(self):
        """
        Init hidden layer
        :return: Initiated hidden layer
        """
        return Variable(torch.zeros(self.hidden_dim), requires_grad=False)

    def init_update(self):
        """
        Init hidden layer
        :return: Initiated hidden layer
        """
        return self.init_hidden()

    def init_wzp(self):
        """
        Init update-reduced matrix
        :return: Initiated update-reduced matrix
        """
        return torch.rand(self.pca_dim, self.hidden_dim)

    def init_wzh(self):
        """
        Init update-hidden matrix
        :return: Initiated update-hidden matrix
        """
        return torch.rand(self.pca_dim, self.hidden_dim)

    def init_bz(self):
        """
        Init update bias
        :return:
        """
        return torch.rand(self.hidden_dim)

    def reset(self):
        """
        Reset learning
        :return:
        """
        self.pca_cell.reset()
        self.reset_reservoir()
        self.train(True)

    def forward(self, u, y=None):
        """
        Forward
        :param u: Input signal.
        :return: Output or hidden states
        """
        time_length = int(u.size()[1])
        n_batches = int(u.size()[0])
        reservoir_states = self.esn_cell(u)
        reservoir_states.required_grad = False
        if self.pca_dim > 0:
            pca_states = self.pca_cell(reservoir_states)
            pca_states.required_grad = False
            if self.finalized:
                return
            hidden_states = Variable(torch.zeros(n_batches, time_length, self.hidden_dim))
            hidden_states = hidden_states if pca_states.is_cuda else hidden_states
        else:
            hidden_states = Variable(torch.zeros(n_batches, time_length, self.hidden_dim))
            hidden_states = hidden_states if reservoir_states.is_cuda else hidden_states
        for b in range(n_batches):
            hidden = self.init_hidden()
            if u.is_cuda:
                hidden = hidden
            for t in range(time_length):
                if self.pca_dim > 0:
                    pt = pca_states[b, t]
                else:
                    pt = reservoir_states[b, t]
                zt = F.sigmoid(self.wzp.mv(pt) + self.wzh.mv(hidden) + self.bz)
                ht = (1.0 - zt) * hidden + zt * pt
                hidden = ht.view(self.hidden_dim)
                hidden_states[b, t] = hidden
        return hidden_states

    def finalize(self):
        """
        Finalize training with LU factorization
        """
        self.pca_cell.finalize()
        self.finalized = True

    def reset_reservoir(self):
        """
        Reset hidden layer
        :return:
        """
        self.esn_cell.reset_hidden()

    def reset_hidden(self):
        """
        Reset hidden layer
        :return:
        """
        self.hidden.fill_(0.0)


class ICACell(nn.Module):
    """
    Principal Component Analysis layer. It can be used to handle different batch-mode algorithm for ICA.
    """

    def __init__(self, input_dim, output_dim):
        """
        Constructor
        :param input_dim: Inputs dimension.
        :param output_dim: Reservoir size
        """
        super(ICACell, self).__init__()
        pass

    def reset(self):
        """
        Reset learning
        :return:
        """
        self.train(True)

    def forward(self, x, y=None):
        """
        Forward
        :param x: Input signal.
        :param y: Target outputs
        :return: Output or hidden states
        """
        batch_size = x.size()[0]
        time_length = x.size()[1]
        if self.with_bias:
            x = self._add_constant(x)

    def finalize(self):
        """
        Finalize training with LU factorization or Pseudo-inverse
        """
        pass

    def _add_constant(self, x):
        """
        Add constant
        :param x:
        :return:
        """
        bias = Variable(torch.ones((x.size()[0], x.size()[1], 1)), requires_grad=False)
        return torch.cat((bias, x), dim=2)


class Identity(nn.Module):
    """
    Identity layer
    """

    def forward(self, x):
        """
        Forward
        :return:
        """
        return x


class LiESN(ESN):
    """
    Leaky-Integrated Echo State Network module
    """

    def __init__(self, input_dim, hidden_dim, output_dim, spectral_radius=0.9, bias_scaling=0, input_scaling=1.0, w=None, w_in=None, w_bias=None, sparsity=None, input_set=[1.0, -1.0], w_sparsity=None, nonlin_func=torch.tanh, learning_algo='inv', ridge_param=0.0, leaky_rate=1.0, train_leaky_rate=False, feedbacks=False, wfdb_sparsity=None, normalize_feedbacks=False, softmax_output=False, seed=None, washout=0, w_distrib='uniform', win_distrib='uniform', wbias_distrib='uniform', win_normal=(0.0, 1.0), w_normal=(0.0, 1.0), wbias_normal=(0.0, 1.0), dtype=torch.float32):
        """
        Constructor
        :param input_dim:
        :param hidden_dim:
        :param output_dim:
        :param spectral_radius:
        :param bias_scaling:
        :param input_scaling:
        :param w:
        :param w_in:
        :param w_bias:
        :param sparsity:
        :param input_set:
        :param w_sparsity:
        :param nonlin_func:
        :param learning_algo:
        :param ridge_param:
        :param leaky_rate:
        :param train_leaky_rate:
        :param feedbacks:
        """
        super(LiESN, self).__init__(input_dim, hidden_dim, output_dim, spectral_radius=spectral_radius, bias_scaling=bias_scaling, input_scaling=input_scaling, w=w, w_in=w_in, w_bias=w_bias, sparsity=sparsity, input_set=input_set, w_sparsity=w_sparsity, nonlin_func=nonlin_func, learning_algo=learning_algo, ridge_param=ridge_param, create_cell=False, feedbacks=feedbacks, wfdb_sparsity=wfdb_sparsity, normalize_feedbacks=normalize_feedbacks, softmax_output=softmax_output, seed=seed, washout=washout, w_distrib=w_distrib, win_distrib=win_distrib, wbias_distrib=wbias_distrib, win_normal=win_normal, w_normal=w_normal, wbias_normal=wbias_normal, dtype=torch.float32)
        self.esn_cell = LiESNCell(leaky_rate, train_leaky_rate, input_dim, hidden_dim, spectral_radius=spectral_radius, bias_scaling=bias_scaling, input_scaling=input_scaling, w=w, w_in=w_in, w_bias=w_bias, sparsity=sparsity, input_set=input_set, w_sparsity=w_sparsity, nonlin_func=nonlin_func, feedbacks=feedbacks, feedbacks_dim=output_dim, wfdb_sparsity=wfdb_sparsity, normalize_feedbacks=normalize_feedbacks, seed=seed, w_distrib=w_distrib, win_distrib=win_distrib, wbias_distrib=wbias_distrib, win_normal=win_normal, w_normal=w_normal, wbias_normal=wbias_normal, dtype=torch.float32)


class OnlinePCACell(nn.Module):
    """
    Online PCA cell
    We extract the principal components from the input data incrementally.
    Weng J., Zhang Y. and Hwang W.,
    Candid covariance-free incremental principal component analysis,
    IEEE Trans. Pattern Analysis and Machine Intelligence,
    vol. 25, 1034--1040, 2003.
    """

    def __init__(self, input_dim, output_dim, amn_params=(20, 200, 2000, 3), init_eigen_vectors=None, var_rel=1, numx_rng=None):
        """
        Constructor
        :param input_dim:
        :param output_dim:
        :param amn_params:
        :param init_eigen_vectors:
        :param var_rel:
        :param numx_rng:
        """
        super(OnlinePCACell, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.amn_params = amn_params
        self._init_v = init_eigen_vectors
        self.var_rel = var_rel
        self._train_iteration = 0
        self._training_type = None
        self._v = None
        self.v = None
        self.d = None
        self._var_tot = 1.0
        self._reduced_dims = self.output_dim

    @property
    def init_eigen_vectors(self):
        """
        Initial eigen vectors
        :return:
        """
        return self._init_v

    @init_eigen_vectors.setter
    def init_eigen_vectors(self, init_eigen_vectors=None):
        """
        Set initial eigen vectors
        :param init_eigen_vectors:
        :return:
        """
        self._init_v = init_eigen_vectors
        if self._input_dim is None:
            self._input_dim = self._init_v.shape[0]
        else:
            assert self.input_dim == self._init_v.shape[0], Exception(u'Dimension mismatch. init_eigen_vectors shape[0] must be {}, given {}'.format(self.input_dim, self._init_v.shape[0]))
        if self._output_dim is None:
            self._output_dim = self._init_v.shape[1]
        else:
            assert (self.output_dim == self._init_v.shape[1], Exception(u'Dimension mismatch, init_eigen_vectors shape[1] must be {}, given {}'.format(self.output_dim, self._init_v.shape[1])))
        if self.v is None:
            self._v = self._init_v.copy()
            self.d = torch.norm(self._v, p=2, dim=0)
            self.v = self._v / self.d

    def get_var_tot(self):
        """
        Get variance explained by PCA
        :return:
        """
        return self._var_tot

    def get_reduced_dimensionality(self):
        """
        Return reducible dimensionality based on the set thresholds.
        :return:
        """
        return self._reduced_dims

    def get_projmatrix(self, transposed=1):
        """
        Get projection matrix
        :param transposed:
        :return:
        """
        if transposed:
            return self.v
        return self.v.t()

    def get_recmatrix(self, transposed=1):
        """
        Get reconstruction matrix
        :param transposed:
        :return:
        """
        if transposed:
            return self.v.t()
        return self.v

    def reset(self):
        """
        Reset learning
        :return:
        """
        self.train(True)

    def forward(self, x, y=None):
        """
        Forward
        :param x: Input signal.
        :param y: Target outputs
        :return: Output or hidden states
        """
        self._update_pca(x)
        return self._execute(x)

    def _execute(self, x, n=None):
        """
        Project the input on the first 'n' components
        :param x:
        :param n:
        :return:
        """
        if n is not None:
            return x.mm(self.v[:, :n])
        return x.mm(self.v)

    def _update_pca(self, x):
        """
        Update the principal components
        :param x:
        :return:
        """
        [w1, w2] = self._amnesic(self.get_current_train_iteration() + 1)
        red_j = self.output_dim
        red_j_flag = False
        explained_var = 0.0
        r = x
        for j in range(self.output_dim):
            v = self._v[:, j:j + 1]
            d = self.d[j]
            v = w1 * v + w2 * r.mv(v) / d * r.t()
            d = torch.norm(v)
            vn = v / d
            r = r - r.mv(vn) * vn.t()
            explained_var += d
            if not red_j_flag:
                ratio = explained_var / self._var_tot
                if ratio > self.var_rel:
                    red_j = j
                    red_j_flag = True
            self._v[:, j:j + 1] = v
            self.v[:, j:j + 1] = vn
            self.d[j] = d
        self._var_tot = explained_var
        self._reduced_dims = red_j

    def _check_params(self, *args):
        """
        Initialize parameters
        :param args:
        :return:
        """
        if self._init_v is None:
            if self.output_dim is not None:
                self.init_eigen_vectors = 0.1 * torch.randn(self.input_dim, self.output_dim)
            else:
                self.init_eigen_vectors = 0.1 * torch.randn(self.input_dim, self.input_dim)

    def _amnesic(self, n):
        """
        Return amnesic weights
        :param n:
        :return:
        """
        _i = float(n + 1)
        n1, n2, m, c = self.amn_params
        if _i < n1:
            l = 0
        elif _i >= n1 and _i < n2:
            l = c * (_i - n1) / (n2 - n1)
        else:
            l = c + (_i - n2) / m
        _world = float(_i - 1 - l) / _i
        _wnew = float(1 + l) / _i
        return [_world, _wnew]

    def _add_constant(self, x):
        """
        Add constant
        :param x:
        :return:
        """
        bias = Variable(torch.ones((x.size()[0], x.size()[1], 1)), requires_grad=False)
        return torch.cat((bias, x), dim=2)


class SFACell(nn.Module):
    """
    Extract the slowly varying components from input data.
    """
    _type_keys = ['f', 'd', 'F', 'D']
    _type_conv = {('f', 'd'): 'd', ('f', 'F'): 'F', ('f', 'D'): 'D', ('d', 'F'): 'D', ('d', 'D'): 'D', ('F', 'd'): 'D', ('F', 'D'): 'D'}

    def __init__(self, input_dim, output_dim, include_last_sample=True, rank_deficit_method='none', use_bias=True):
        """
        Constructor
        :param input_dim: Input dimension
        :param output_dim: Number of slow feature
        :param include_last_sample: If set to False, the training method discards the last sample in every chunk during training when calculating the matrix.
        :param rank_deficit_method: 'none', 'reg', 'pca', 'svd', 'auto'.
        """
        super(SFACell, self).__init__()
        self.include_last_sample = include_last_sample
        self.use_bias = use_bias
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.xTx = torch.zeros(input_dim, input_dim)
        self.xTx_avg = torch.zeros(input_dim)
        self.dxTdx = torch.zeros(input_dim, input_dim)
        self.dxTdx_avg = torch.zeros(input_dim)
        self.set_rank_deficit_method(rank_deficit_method)
        self.rank_threshold = 1e-12
        self.rank_deficit = 0
        self.d = None
        self.sf = None
        self.avg = None
        self.bias = None
        self.tlen = None

    def time_derivative(self, x):
        """
        Compute the approximation of time derivative
        :param x:
        :return:
        """
        return x[1:, :] - x[:-1, :]

    def reset(self):
        """
        Reset learning
        :return:
        """
        self.train(True)

    def forward(self, x):
        """
        Forward
        :param x: Input signal.
        :return: Output or hidden states
        """
        for b in np.arange(0, x.size(0)):
            if self.training:
                last_sample_index = None if self.include_last_sample else -1
                xs = x[(b), :last_sample_index, :]
                xd = self.time_derivative(x[b])
                self.xTx.data.add(xs.t().mm(xs))
                self.dxTdx.data.add(xd.t().mm(xd))
                self.xTx_avg += torch.sum(xs, axis=1)
                self.dxTdx_avg += torch.sum(xd, axis=1)
                self.tlen += x.size(0)
            else:
                x[b].mv(self.sf) - self.bias
        return x

    def finalize(self):
        """
        Finalize training with LU factorization or Pseudo-inverse
        """
        self.xTx, self.xTx_avg, self.tlen = self._fix(self.xtX, self.xTx_avg, self.tlen, center=True)
        self.dxTdx, self.dxTdx_avg, self.tlen = self._fix(self.dxTdx, self.dxTdx_avg, self.tlen, center=False)
        rng = 1, self.output_dim
        self.d, self.sf = self._symeig(self.dxTdx, self.xTx, rng)
        d = self.d
        if torch.min(d) < 0:
            raise Exception(u'Got negative values in {}'.format(d))
        del self.xTx
        del self.dxTdx
        self.bias = self.xTx_avg * self.sf

    def _symeig(self, A, B, range, eigenvectors=True):
        """
        Solve standard and generalized eigenvalue problem for symmetric (hermitian) definite positive matrices.
        :param A: An N x N matrix
        :param B: An N x N matrix
        :param range: (lo, hi), the indexes of smallest and largest eigenvalues to be returned.
        :param eigenvectors: Return eigenvalues and eigenvector or only engeivalues
        :return: w, the eigenvalues and Z the eigenvectors
        """
        A = A.numpy()
        B = B.numpy()
        dtype = np.dtype()
        wB, ZB = np.linalg.eigh(B)
        self._assert_eigenvalues_real(wB)
        if wB.real.min() < 0:
            raise Exception(u'Got negative eigenvalues: {}'.format(wB))
        ZB = old_div(ZB.real, np.sqrt(wB.real))
        A = np.matmul(np.matmul(ZB.T, A), ZB)
        w, ZA = np.linalg.eigh(A)
        Z = np.matmul(ZB, ZA)
        self._assert_eigenvalues_real(w, dtype)
        w = w.real
        Z = Z.real
        idx = w.argsort()
        w = w.take(idx)
        Z = Z.take(idx, axis=1)
        n = A.shape[0]
        lo, hi = range
        if lo < 1:
            lo = 1
        if lo > n:
            lo = n
        if hi > n:
            hi = n
        if lo > hi:
            lo, hi = hi, lo
        Z = Z[:, lo - 1:hi]
        w = w[lo - 1:hi]
        w = self.refcast(w, dtype)
        Z = self.refcast(Z, dtype)
        if eigenvectors:
            return torch.FloatTensor(w), torch.FloatTensor(Z)
        else:
            return torch.FloatTensor(w)

    def refcast(self, array, dtype):
        """
        Cast the array to dtype only if necessary, otherwise return a reference.
        """
        dtype = np.dtype(dtype)
        if array.dtype == dtype:
            return array
        return array.astype(dtype)

    def _assert_eigenvalues_real(self, w, dtype):
        """
        Check eigenvalues
        :param w:
        :param dtype:
        :return:
        """
        tol = np.finfo(dtype.type).eps * 100
        if abs(w.imag).max() > tol:
            err = 'Some eigenvalues have significant imaginary part: %s ' % str(w)
            raise Exception(err)

    def _greatest_common_dtype(self, alist):
        """
        Apply conversion rules to find the common conversion type
        dtype 'd' is default for 'i' or unknown types
        (known types: 'f','d','F','D').
        """
        dtype = 'f'
        for array in alist:
            if array is None:
                continue
            tc = array.dtype.char
            if tc not in self._type_keys:
                tc = 'd'
            transition = dtype, tc
            if transition in self._type_conv:
                dtype = self._type_conv[transition]
        return dtype

    def _fix(self, mtx, avg, tlen, center=True):
        """
        Returns a triple containing the covariance matrix, the average and
        the number of observations.
        :param mtx:
        :param center:
        :return:
        """
        if self.use_bias:
            mtx /= tlen
        else:
            mtx /= tlen - 1
        if center:
            avg_mtx = np.outer(avg, avg)
            if self.use_bias:
                avg_mtx /= tlen * tlen
            else:
                avg_mtx /= tlen * (tlen - 1)
            mtx -= avg_mtx
        avg /= tlen
        return mtx, avg, tlen


class StackedESN(nn.Module):
    """
    Stacked Echo State Network module
    """

    def __init__(self, input_dim, hidden_dim, output_dim, leaky_rate=1.0, spectral_radius=0.9, bias_scaling=0, input_scaling=1.0, w=None, w_in=None, w_bias=None, sparsity=None, input_set=(1.0, -1.0), w_sparsity=None, nonlin_func=torch.tanh, learning_algo='inv', ridge_param=0.0, with_bias=True):
        """
        Constructor

        Arguments:
            :param input_dim: Inputs dimension.
            :param hidden_dim: Hidden layer dimension
            :param output_dim: Reservoir size
            :param spectral_radius: Reservoir's spectral radius
            :param bias_scaling: Scaling of the bias, a constant input to each neuron (default: 0, no bias)
            :param input_scaling: Scaling of the input weight matrix, default 1.
            :param w: Internation weights matrix
            :param w_in: Input-reservoir weights matrix
            :param w_bias: Bias weights matrix
            :param w_fdb: Feedback weights matrix
            :param sparsity:
            :param input_set:
            :param w_sparsity:
            :param nonlin_func: Reservoir's activation function (tanh, sig, relu)
            :param learning_algo: Which learning algorithm to use (inv, LU, grad)
        """
        super(StackedESN, self).__init__()
        self.n_layers = len(hidden_dim)
        self.esn_layers = list()
        self.n_features = 0
        for n in range(self.n_layers):
            layer_input_dim = input_dim if n == 0 else hidden_dim[n - 1]
            self.n_features += hidden_dim[n]
            layer_leaky_rate = leaky_rate[n] if type(leaky_rate) is list or type(leaky_rate) is np.ndarray else leaky_rate
            layer_spectral_radius = spectral_radius[n] if type(spectral_radius) is list or type(spectral_radius) is np.ndarray else spectral_radius
            layer_bias_scaling = bias_scaling[n] if type(bias_scaling) is list or type(bias_scaling) is np.ndarray else bias_scaling
            layer_input_scaling = input_scaling[n] if type(input_scaling) is list or type(input_scaling) is np.ndarray else input_scaling
            if type(w) is torch.Tensor and w.dim() == 3:
                layer_w = w[n]
            elif type(w) is torch.Tensor:
                layer_w = w
            else:
                layer_w = None
            if type(w_in) is torch.Tensor and w_in.dim() == 3:
                layer_w_in = w_in[n]
            elif type(w_in) is torch.Tensor:
                layer_w_in = w_in
            else:
                layer_w_in = None
            if type(w_bias) is torch.Tensor and w_bias.dim() == 2:
                layer_w_bias = w_bias[n]
            elif type(w_bias) is torch.Tensor:
                layer_w_bias = w_bias
            else:
                layer_w_bias = None
            layer_sparsity = sparsity[n] if type(sparsity) is list or type(sparsity) is np.ndarray else sparsity
            layer_input_set = input_set[n] if type(input_set) is list or type(input_set) is np.ndarray else input_set
            layer_w_sparsity = w_sparsity[n] if type(w_sparsity) is list or type(w_sparsity) is np.ndarray else w_sparsity
            layer_nonlin_func = nonlin_func[n] if type(nonlin_func) is list or type(nonlin_func) is np.ndarray else nonlin_func
            self.esn_layers.append(LiESNCell(layer_leaky_rate, False, layer_input_dim, hidden_dim[n], layer_spectral_radius, layer_bias_scaling, layer_input_scaling, layer_w, layer_w_in, layer_w_bias, None, layer_sparsity, layer_input_set, layer_w_sparsity, layer_nonlin_func))
        self.output = RRCell(self.n_features, output_dim, ridge_param, False, with_bias, learning_algo)

    @property
    def hidden(self):
        """
        Hidden layer
        :return:
        """
        hidden_states = list()
        for esn_cell in self.esn_layers:
            hidden_states.append(esn_cell.hidden)
        return hidden_states

    @property
    def w(self):
        """
        Hidden weight matrix
        :return:
        """
        w_mtx = list()
        for esn_cell in self.esn_layers:
            w_mtx.append(esn_cell.w)
        return w_mtx

    @property
    def w_in(self):
        """
        Input matrix
        :return:
        """
        win_mtx = list()
        for esn_cell in self.esn_layers:
            win_mtx.append(esn_cell.w_in)
        return win_mtx

    def reset(self):
        """
        Reset learning
        :return:
        """
        self.output.reset()
        self.train(True)

    def get_w_out(self):
        """
        Output matrix
        :return:
        """
        return self.output.w_out

    def forward(self, u, y=None):
        """
        Forward
        :param u: Input signal.
        :param y: Target outputs
        :return: Output or hidden states
        """
        hidden_states = Variable(torch.zeros(u.size(0), u.size(1), self.n_features))
        pos = 0
        for index, esn_cell in enumerate(self.esn_layers):
            layer_dim = esn_cell.output_dim
            if index == 0:
                last_hidden_states = esn_cell(u)
            else:
                last_hidden_states = esn_cell(last_hidden_states)
            hidden_states[:, :, pos:pos + layer_dim] = last_hidden_states
            pos += layer_dim
        return self.output(hidden_states, y)

    def finalize(self):
        """
        Finalize training with LU factorization
        """
        self.output.finalize()
        self.train(False)

    def reset_hidden(self):
        """
        Reset hidden layer
        :return:
        """
        self.esn_cell.reset_hidden()

    def get_spectral_radius(self):
        """
        Get W's spectral radius
        :return: W's spectral radius
        """
        return self.esn_cell.get_spectral_raduis()

    @staticmethod
    def generate_ws(n_layers, reservoir_size, w_sparsity):
        """
        Generate W matrices for a stacked ESN
        :param n_layers:
        :param reservoir_size:
        :param w_sparsity:
        :return:
        """
        ws = torch.FloatTensor(n_layers, reservoir_size, reservoir_size)
        for i in range(n_layers):
            ws[i] = ESNCell.generate_w(reservoir_size, w_sparsity)
        return ws


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Conceptor,
     lambda: ([], {'conceptor_dim': 4}),
     lambda: ([torch.rand([4, 4, 1])], {}),
     False),
    (Identity,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_nschaetti_EchoTorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

