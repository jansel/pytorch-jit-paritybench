import sys
_module = sys.modules[__name__]
del sys
master = _module
configs = _module
control_layer = _module
control_instructions = _module
control_layer = _module
controllers = _module
base = _module
feedforward = _module
recurrent = _module
conf = _module
formalisms = _module
buta_example = _module
cfg = _module
depth_generate = _module
generate_tests = _module
tree_automata = _module
trees = _module
cfg = _module
generate_test_files = _module
reverse_buffered = _module
test_generalization = _module
testing_mode = _module
trace_console = _module
models = _module
base = _module
buffered = _module
legacy = _module
buffered = _module
embedding = _module
lstm = _module
model = _module
vanilla = _module
vanilla = _module
run = _module
run_tests = _module
setup = _module
stacknn_utils = _module
data_readers = _module
errors = _module
loggers = _module
overrides = _module
testcase = _module
validation = _module
vector_ops = _module
structs = _module
base = _module
buffers = _module
queue = _module
stack = _module
null = _module
regularization = _module
simple = _module
simple_example = _module
tests = _module
tasks = _module
base = _module
cfg = _module
counting = _module
evaluation = _module
language_modeling = _module
natural = _module
reverse = _module
visualization = _module
visualizers = _module

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


from torch.nn import CrossEntropyLoss


from numpy.testing import assert_approx_equal


import torch


from abc import ABCMeta


from abc import abstractmethod


import numpy as np


import torch.nn as nn


from torch.autograd import Variable


from math import ceil


import torch.optim as optim


import torch.nn.functional as F


from sklearn.utils import shuffle


import random


import torch.autograd as autograd


import matplotlib.pyplot as plt


from torch.nn.functional import relu


from abc import abstractproperty


from copy import copy


from copy import deepcopy


import warnings


class ControlInstructions:
    """Stores the instruction values produced by a control layer.

    TODO: Perhaps stack.forward should be able to take this object as an argument.
    """

    def __init__(self, push_vectors, push_strengths, pop_strengths, read_strengths, pop_distributions=None, read_distributions=None):
        self.push_vectors = push_vectors
        self.push_strengths = push_strengths
        self.pop_strengths = pop_strengths
        self.read_strengths = read_strengths
        self.pop_distributions = pop_distributions
        self.read_distributions = read_distributions

    def make_tuple(self):
        return self.push_vectors, self.pop_strengths, self.push_strengths, self.read_strengths

    def __len__(self):
        return len(self.push_vectors)


class ControlLayer(torch.nn.Module):
    """Layer to convert a vector to stack instructions."""

    def __init__(self, input_size, stack_size, vision, device=None):
        """Construct a ControlLayer object.

        Args:
            input_size: The length of the vectors inputted to the ControlLayer.
            stack_size: The size of the vectors on the stack.
            vision: The maximum depth for reading and popping from the stack.
        """
        super().__init__()
        self._vector_map = torch.nn.Linear(input_size, stack_size)
        self._push_map = torch.nn.Linear(input_size, 1)
        self._pop_map = torch.nn.Linear(input_size, vision)
        self._read_map = torch.nn.Linear(input_size, vision)
        self._device = device

    def forward(self, input_vector):
        push_vector = torch.tanh(self._vector_map(input_vector))
        push_strength = torch.sigmoid(self._push_map(input_vector))
        pop_distribution = torch.softmax(self._pop_map(input_vector), 1)
        pop_strength = self._get_expectation(pop_distribution)
        read_distribution = torch.softmax(self._read_map(input_vector), 1)
        read_strength = self._get_expectation(read_distribution)
        return ControlInstructions(push_vector, push_strength.squeeze(1), pop_strength.squeeze(1), read_strength.squeeze(1), pop_distribution, read_distribution)

    def _get_expectation(self, distribution):
        """Take the expected value of a pop/read distribution."""
        values = torch.arange(distribution.size(1), device=self._device)
        values = values.unsqueeze(1)
        return torch.mm(distribution, values.float())


class Controller(nn.Module):
    """
    Abstract class for neural network Modules to be used in Models.
    Inherit from this class in order to create a custom architecture for
    a Model, or to create a controller compatible with a custom neural
    data structure.
    """
    __metaclass__ = ABCMeta

    def __init__(self, input_size, read_size, output_size):
        """
        Constructor for the Controller object.

        :type input_size: int
        :param input_size: The size of input vectors to this Controller

        :type read_size: int
        :param read_size: The size of vectors placed on the neural data
            structure

        :type output_size: int
        :param output_size: The size of vectors output from this Controller
        """
        super(Controller, self).__init__()
        self._input_size = input_size
        self._read_size = read_size
        self._output_size = output_size

    @abstractmethod
    def forward(self, x, r):
        """
        This Controller should take an input and the previous item read
        from the neural data structure and produce an output and a set
        of instructions for operating the neural data structure.

        :type x: Variable
        :param x: The input to this Controller

        :type r: Variable
        :param r: The previous item read from the neural data structure

        :rtype: tuple
        :return: The first item of the tuple should contain the output
            of the controller. The second item should be a tuple containing
            instructions for the neural data structure. For example, the
            return value corresponding to the instructions
                - output y
                - pop a strength u from the data structure
                - push v with strength d to the data structure
            is (y, (v, u, d))
        """
        raise NotImplementedError('Missing implementation for forward')

    @staticmethod
    def init_normal(tensor):
        """
        Populates a Variable with values drawn from a normal
        distribution with mean 0 and standard deviation 1/sqrt(n), where
        n is the length of the Variable.

        :type tensor: Variable
        :param tensor: The Variable to populate with values

        :return: None
        """
        n = tensor.data.shape[0]
        tensor.data.normal_(0, 1.0 / np.sqrt(n))

    def init_controller(self, batch_size):
        """
        Initializes various components of the controller.

        :type batch_size: int
        :param batch_size: The number of trials in each mini-batch where
            this Model is used

        :return: None
        """
        pass


class SimpleStructController(Controller):
    """
    Abstract class for Controllers to be used with SimpleStructs (see
    structs.simple.SimpleStruct). This class primarily contains
    reporting tools that record the SimpleStruct instructions at each
    time step.
    """

    def __init__(self, input_size, read_size, output_size, n_args=2):
        """
        Constructor for the SimpleStructController object. In addition to
        calling the base class constructor, this constructor initializes
        private properties used for reporting. Logged data are stored in
        self._log, a Numpy array whose columns contain the instructions
        computed by the SimpleStructController to the SimpleStruct at each
        time step.

        :type input_size: int
        :param input_size: The size of input vectors to this Controller

        :type read_size: int
        :param read_size: The size of vectors placed on the neural data
            structure

        :type output_size: int
        :param output_size: The size of vectors output from this Controller

        :type n_args: int
        :param n_args: The number of struct instructions, apart from the
            value to push onto the struct, that will be computed by the
            controller. By default, this value is 2: the push strength and
            the pop strength
        """
        super(SimpleStructController, self).__init__(input_size, read_size, output_size)
        self._n_args = n_args
        self._logging = False
        self.log_data = None
        self._log_data_size = 0
        self._curr_log_entry = 0
    """ Reporting """

    def init_log(self, log_data_size):
        """
        Initializes self._log_data to an empty array of a specified
        size.

        :type log_data_size: int
        :param log_data_size: The number of columns of self._log_data
            (i.e., the number of time steps for which data are logged)

        :return: None
        """
        self.log_data = np.zeros([self._n_args + self._read_size + self._input_size + self._output_size, log_data_size])
        self._log_data_size = log_data_size
        self._curr_log_entry = 0
        return

    def start_log(self, log_data_size=None):
        """
        Sets self._log to True, so that data will be logged the next
        time self.forward is called.

        :type log_data_size: int
        :param log_data_size: If a value is supplied for this argument,
            then self.init_log will be called.

        :return: None
        """
        self._logging = True
        if log_data_size is not None:
            self.init_log(log_data_size)
        return

    def stop_log(self):
        """
        Sets self._log to False, so that data will no longer be logged
        the next time self.forward is called.

        :return: None
        """
        self._logging = False
        return

    def _log(self, x, y, v, *instructions):
        """
        Records the action of the Controller at a particular time step to
        self._log_data.

        :type x: Variable
        :param x: The input to the Controller

        :type y: Variable
        :praam y: The output of the Controller

        :type v: Variable
        :param v: The value that will be pushed to the data structure

        :type instructions: list
        :param instructions: Other data structure instructions

        :return: None
        """
        t = self._curr_log_entry
        if not self._logging:
            return
        elif t >= self._log_data_size:
            return
        x_start = 0
        x_end = self._input_size
        y_start = self._input_size
        y_end = self._input_size + self._output_size
        i_start = self._input_size + self._output_size
        v_start = self._input_size + self._output_size + self._n_args
        self.log_data[x_start:x_end, t] = x.data.numpy()
        self.log_data[y_start:y_end, t] = y.data.numpy()
        self.log_data[v_start:, t] = v.data.numpy()
        for j in xrange(self._n_args):
            instruction = instructions[j].data.numpy()
            self.log_data[i_start + j, self._curr_log_entry] = instruction
        self._curr_log_entry += 1
        return


def unused_init_param(param_name, arg_value, obj):
    """
    Displays a warning message saying that a constructor for an object
    has received an argument value for an unused parameter.

    :type param_name: str
    :param param_name: The name of the unused parameter

    :param arg_value: The argument value passed to the constructor

    :param obj: The object being instantiated

    :return: None
    """
    if arg_value is not None:
        class_name = type(obj).__name__
        msg = 'Parameter {} is set to {}, '.format(param_name, arg_value)
        msg += 'but it is not used in {}.'.format(class_name)
        warnings.warn(msg, RuntimeWarning)


class DeepSimpleStructController(SimpleStructController):
    """
    A fully connected multilayer network producing instructions compatible
    with SimpleStructs (see structs.simple.SimpleStruct).
    """

    def __init__(self, input_size, read_size, output_size, n_args=2, discourage_pop=True, n_hidden_layers=2, non_linearity=nn.ReLU, **kwargs):
        """
        Constructor for the DeepSimpleStructController object.

        :type input_size: int
        :param input_size: The size of input vectors to this Controller

        :type read_size: int
        :param read_size: The size of vectors placed on the neural data
            structure

        :type output_size: int
        :param output_size: The size of vectors output from this Controller

        :type n_args: int
        :param n_args: The number of struct instructions, apart from the
            value to push onto the struct, that will be computed by the
            controller. By default, this value is 2: the push strength and
            the pop strength

        :type discourage_pop: bool
        :param discourage_pop: If True, then weights will be initialized
            to discourage popping

        :type n_hidden_layers: int
        :param n_hidden_layers: How many feedforward layers

        :type non_linearity: Module
        :param non_linearity: Non-linearity to apply to hidden layers
        """
        super(DeepSimpleStructController, self).__init__(input_size, read_size, output_size, n_args=n_args)
        for param_name, arg_value in kwargs.iteritems():
            unused_init_param(param_name, arg_value, self)
        nn_input_size = self._input_size + self._read_size
        nn_output_size = self._n_args + self._read_size + self._output_size
        nn_hidden_size = int(ceil((nn_input_size + nn_output_size) / 2.0))
        nn_sizes_list = [nn_input_size] + [nn_hidden_size] * n_hidden_layers
        self._network = nn.Sequential()
        for i in range(n_hidden_layers):
            self._network.add_module('lin' + str(i), nn.Linear(nn_sizes_list[i], nn_sizes_list[i + 1]))
            self._network.add_module('relu' + str(i), non_linearity())
        self._network.add_module('out', nn.Linear(nn_sizes_list[-1], nn_output_size))
        self.discourage_pop = discourage_pop
        self._network.apply(self.init_weights)

    def init_weights(self, module):
        """
        Initializes a linear layer with values drawn from a normal
        distribution

        :type module: Module
        :param module: The module (layer) to initialize

        :return: None
        """
        if type(module) == nn.Linear:
            DeepSimpleStructController.init_normal(module.weight)
            module.bias.data.fill_(0)
            if self.discourage_pop:
                module.bias.data[0] = -1.0
                if self._n_args >= 4:
                    module.bias.data[2] = 1.0
                    module.bias.data[3] = 1.0

    def forward(self, x, r):
        """
        Computes an output and data structure instructions using a
        multi-layer nn.

        :type x: Variable
        :param x: The input to this Controller

        :type r: Variable
        :param r: The previous item read from the neural data structure

        :rtype: tuple
        :return: A tuple of the form (y, (v, u, d)), interpreted as
            follows:
                - output y
                - pop a strength u from the data structure
                - push v with strength d to the data structure
        """
        nn_output = self._network(torch.cat([x, r], 1))
        output = nn_output[:, self._n_args + self._read_size:].contiguous()
        read_params = torch.sigmoid(nn_output[:, :self._n_args + self._read_size])
        v = read_params[:, self._n_args:].contiguous()
        instructions = tuple(read_params[:, j].contiguous() for j in xrange(self._n_args))
        self._log(x, torch.sigmoid(output), v, *instructions)
        return output, (v,) + instructions


class LinearSimpleStructController(SimpleStructController):
    """
    A single linear layer producing instructions compatible with
    SimpleStructs (see structs.simple.SimpleStruct).
    """

    def __init__(self, input_size, read_size, output_size, n_args=2, custom_initialization=True, discourage_pop=True, **kwargs):
        """
        Constructor for the LinearSimpleStruct object.

        :type input_size: int
        :param input_size: The size of input vectors to this Controller

        :type read_size: int
        :param read_size: The size of vectors placed on the neural data
            structure

        :type output_size: int
        :param output_size: The size of vectors output from this Controller

        :type n_args: int
        :param n_args: The number of struct instructions, apart from the
            value to push onto the struct, that will be computed by the
            controller. By default, this value is 2: the push strength and
            the pop strength

        :type discourage_pop: bool
        :param discourage_pop: If True, then weights will be initialized
            to discourage popping
        """
        super(LinearSimpleStructController, self).__init__(input_size, read_size, output_size, n_args=n_args)
        for param_name, arg_value in kwargs.iteritems():
            unused_init_param(param_name, arg_value, self)
        nn_input_size = self._input_size + self._read_size
        nn_output_size = self._n_args + self._read_size + self._output_size
        self._linear = nn.Linear(nn_input_size, nn_output_size)
        if custom_initialization:
            LinearSimpleStructController.init_normal(self._linear.weight)
            self._linear.bias.data.fill_(0)
        if discourage_pop:
            self._linear.bias.data[0] = -1.0
            if n_args >= 4:
                self._linear.bias.data[2] = 1.0
                self._linear.bias.data[3] = 1.0

    def forward(self, x, r):
        """
        Computes an output and data structure instructions using a
        single linear layer.

        :type x: Variable
        :param x: The input to this Controller

        :type r: Variable
        :param r: The previous item read from the neural data structure

        :rtype: tuple
        :return: A tuple of the form (y, (v, u, d)), interpreted as
            follows:
                - output y
                - pop a strength u from the data structure
                - push v with strength d to the data structure
        """
        nn_output = self._linear(torch.cat([x, r], 1))
        output = nn_output[:, self._n_args + self._read_size:].contiguous()
        read_params = torch.sigmoid(nn_output[:, :self._n_args + self._read_size])
        v = read_params[:, self._n_args:].contiguous()
        instructions = tuple(read_params[:, j].contiguous() for j in xrange(self._n_args))
        self._log(x, torch.sigmoid(output), v, *instructions)
        return output, (v,) + instructions


class RNNSimpleStructController(SimpleStructController):
    """
    An RNN producing instructions compatible with SimpleStructs (see
    structs.Simple.SimpleStruct).
    """

    def __init__(self, input_size, read_size, output_size, custom_initialization=True, discourage_pop=True, hidden_size=10, n_args=2, **kwargs):
        """
        Constructor for the RNNSimpleStructController object.

        :type input_size: int
        :param input_size: The size of input vectors to this Controller

        :type read_size: int
        :param read_size: The size of vectors placed on the neural data
            structure

        :type output_size: int
        :param output_size: The size of vectors output from this Controller

        :type discourage_pop: bool
        :param discourage_pop: If True, then weights will be initialized
            to discourage popping

        :type hidden_size: int
        :param hidden_size: The size of the hidden state vector

        :type n_args: int
        :param n_args: The number of struct instructions, apart from the
            value to push onto the struct, that will be computed by the
            controller. By default, this value is 2: the push strength and
            the pop strength
        """
        super(RNNSimpleStructController, self).__init__(input_size, read_size, output_size, n_args=n_args)
        for param_name, arg_value in kwargs.iteritems():
            unused_init_param(param_name, arg_value, self)
        self._hidden = None
        nn_input_size = self._input_size + self._read_size
        nn_output_size = self._n_args + self._read_size + self._output_size
        self._rnn = nn.RNNCell(nn_input_size, hidden_size)
        self._linear = nn.Linear(hidden_size, nn_output_size)
        if custom_initialization:
            RNNSimpleStructController.init_normal(self._rnn.weight_hh)
            RNNSimpleStructController.init_normal(self._rnn.weight_ih)
            self._rnn.bias_hh.data.fill_(0)
            self._rnn.bias_ih.data.fill_(0)
            RNNSimpleStructController.init_normal(self._linear.weight)
            self._linear.bias.data.fill_(0)
        if discourage_pop:
            self._linear.bias.data[0] = -1.0
            if n_args >= 4:
                self._linear.bias.data[2] = 1.0
                self._linear.bias.data[3] = 1.0

    def _init_hidden(self, batch_size):
        """
        Initializes the hidden state of the LSTM cell to zeros.

        :type batch_size: int
        :param batch_size: The number of trials in each mini-batch where
            this Model is used

        :return: None
        """
        rnn_hidden_shape = batch_size, self._rnn.hidden_size
        self._hidden = Variable(torch.zeros(rnn_hidden_shape))

    def init_controller(self, batch_size):
        self._init_hidden(batch_size)

    def forward(self, x, r):
        """
        Computes an output and data structure instructions using a
        single linear layer.

        :type x: Variable
        :param x: The input to this Controller

        :type r: Variable
        :param r: The previous item read from the neural data structure

        :rtype: tuple
        :return: A tuple of the form (y, (v, u, d)), interpreted as
            follows:
                - output y
                - pop a strength u from the data structure
                - push v with strength d to the data structure
        """
        self._hidden = self._rnn(torch.cat([x, r], 1), self._hidden)
        nn_output = self._linear(self._hidden)
        output = nn_output[:, self._n_args + self._read_size:].contiguous()
        read_params = torch.sigmoid(nn_output[:, :self._n_args + self._read_size])
        v = read_params[:, self._n_args:].contiguous()
        instructions = tuple(read_params[:, j].contiguous() for j in xrange(self._n_args))
        self._log(x, torch.sigmoid(output), v, *instructions)
        return output, (v,) + instructions


class LSTMSimpleStructController(SimpleStructController):
    """
    An LSTM producing instructions compatible with SimpleStructs (see
    structs.Simple.SimpleStruct).

    https://pytorch.org/docs/stable/nn.html#lstmcell
    """

    def __init__(self, input_size, read_size, output_size, custom_initialization=True, discourage_pop=True, hidden_size=10, n_args=2, **kwargs):
        """
        Constructor for the LSTMSimpleStructController object.

        :type input_size: int
        :param input_size: The size of input vectors to this Controller

        :type read_size: int
        :param read_size: The size of vectors placed on the neural data
            structure

        :type output_size: int
        :param output_size: The size of vectors output from this Controller

        :type discourage_pop: bool
        :param discourage_pop: If True, then weights will be initialized
            to discourage popping

        :type hidden_size: int
        :param hidden_size: The size of state vectors

        :type n_args: int
        :param n_args: The number of struct instructions, apart from the
            value to push onto the struct, that will be computed by the
            controller. By default, this value is 2: the push strength and
            the pop strength
        """
        super(LSTMSimpleStructController, self).__init__(input_size, read_size, output_size, n_args=n_args)
        for param_name, arg_value in kwargs.iteritems():
            unused_init_param(param_name, arg_value, self)
        self._hidden = None
        self._cell_state = None
        nn_input_size = self._input_size + self._read_size
        nn_output_size = self._n_args + self._read_size + self._output_size
        self._lstm = nn.LSTMCell(nn_input_size, hidden_size)
        self._linear = nn.Linear(hidden_size, nn_output_size)
        if custom_initialization:
            LSTMSimpleStructController.init_normal(self._lstm.weight_hh)
            LSTMSimpleStructController.init_normal(self._lstm.weight_ih)
            self._lstm.bias_hh.data.fill_(0)
            self._lstm.bias_ih.data.fill_(0)
            LSTMSimpleStructController.init_normal(self._linear.weight)
            self._linear.bias.data.fill_(0)
        if discourage_pop:
            self._linear.bias.data[0] = -1.0
            if n_args >= 4:
                self._linear.bias.data[2] = 1.0
                self._linear.bias.data[3] = 1.0

    def _init_hidden(self, batch_size):
        """
        Initializes the hidden state of the LSTM cell to zeros.

        :type batch_size: int
        :param batch_size: The number of trials in each mini-batch where
            this Model is used

        :return: None
        """
        lstm_hidden_shape = batch_size, self._lstm.hidden_size
        self._hidden = Variable(torch.zeros(lstm_hidden_shape))
        self._cell_state = Variable(torch.zeros(lstm_hidden_shape))

    def init_controller(self, batch_size):
        self._init_hidden(batch_size)

    def forward(self, x, r):
        """
        Computes an output and data structure instructions using a
        single linear layer.

        :type x: Variable
        :param x: The input to this Controller

        :type r: Variable
        :param r: The previous item read from the neural data structure

        :rtype: tuple
        :return: A tuple of the form (y, (v, u, d)), interpreted as
            follows:
                - output y
                - pop a strength u from the data structure
                - push v with strength d to the data structure
        """
        self._hidden, self._cell_state = self._lstm(torch.cat([x, r], 1), (self._hidden, self._cell_state))
        nn_output = self._linear(self._hidden)
        output = nn_output[:, self._n_args + self._read_size:].contiguous()
        read_params = torch.sigmoid(nn_output[:, :self._n_args + self._read_size])
        v = read_params[:, self._n_args:].contiguous()
        instructions = tuple(read_params[:, j].contiguous() for j in xrange(self._n_args))
        self._log(x, torch.sigmoid(output), v, *instructions)
        return output, (v,) + instructions


class GRUSimpleStructController(SimpleStructController):
    """
    An GRU producing instructions compatible with SimpleStructs (see
    structs.Simple.SimpleStruct).
    """

    def __init__(self, input_size, read_size, output_size, custom_initialization=True, discourage_pop=True, hidden_size=10, n_args=2, **kwargs):
        """
        Constructor for the GRUSimpleStructController object.

        :type input_size: int
        :param input_size: The size of input vectors to this Controller

        :type read_size: int
        :param read_size: The size of vectors placed on the neural data
            structure

        :type output_size: int
        :param output_size: The size of vectors output from this Controller

        :type discourage_pop: bool
        :param discourage_pop: If True, then weights will be initialized
            to discourage popping

        :type hidden_size: int
        :param hidden_size: The size of the hidden state vector

        :type n_args: int
        :param n_args: The number of struct instructions, apart from the
            value to push onto the struct, that will be computed by the
            controller. By default, this value is 2: the push strength and
            the pop strength
        """
        super(GRUSimpleStructController, self).__init__(input_size, read_size, output_size, n_args=n_args)
        for param_name, arg_value in kwargs.iteritems():
            unused_init_param(param_name, arg_value, self)
        self._hidden = None
        nn_input_size = self._input_size + self._read_size
        nn_output_size = self._n_args + self._read_size + self._output_size
        self._GRU = nn.GRUCell(nn_input_size, hidden_size)
        self._linear = nn.Linear(hidden_size, nn_output_size)
        if custom_initialization:
            GRUSimpleStructController.init_normal(self._GRU.weight_hh)
            GRUSimpleStructController.init_normal(self._GRU.weight_ih)
            self._GRU.bias_hh.data.fill_(0)
            self._GRU.bias_ih.data.fill_(0)
            GRUSimpleStructController.init_normal(self._linear.weight)
            self._linear.bias.data.fill_(0)
        if discourage_pop:
            self._linear.bias.data[0] = -1.0
            if n_args >= 4:
                self._linear.bias.data[2] = 1.0
                self._linear.bias.data[3] = 1.0

    def _init_hidden(self, batch_size):
        """
        Initializes the hidden state of the GRU cell to zeros.

        :type batch_size: int
        :param batch_size: The number of trials in each mini-batch where
            this Model is used

        :return: None
        """
        GRU_hidden_shape = batch_size, self._GRU.hidden_size
        self._hidden = Variable(torch.zeros(GRU_hidden_shape))

    def init_controller(self, batch_size):
        self._init_hidden(batch_size)

    def forward(self, x, r):
        """
        Computes an output and data structure instructions using a
        single linear layer.

        :type x: Variable
        :param x: The input to this Controller

        :type r: Variable
        :param r: The previous item read from the neural data structure

        :rtype: tuple
        :return: A tuple of the form (y, (v, u, d)), interpreted as
            follows:
                - output y
                - pop a strength u from the data structure
                - push v with strength d to the data structure
        """
        self._hidden = self._GRU(torch.cat([x, r], 1), self._hidden)
        nn_output = self._linear(self._hidden)
        output = nn_output[:, self._n_args + self._read_size:].contiguous()
        read_params = torch.sigmoid(nn_output[:, :self._n_args + self._read_size])
        v = read_params[:, self._n_args:].contiguous()
        instructions = tuple(read_params[:, j].contiguous() for j in xrange(self._n_args))
        self._log(x, torch.sigmoid(output), v, *instructions)
        return output, (v,) + instructions


class Operation(object):
    push = 0
    pop = 1


class Struct(nn.Module):
    """
    Abstract class for implementing neural data structures, such as
    stacks, queues, and dequeues. Data structures inheriting from this
    class are intended to be used in neural networks that are trained in
    mini-batches. Thus, the actions of the structures are performed for
    each trial in a mini-batch simultaneously.

    To create a custom data structure, you must create a class
    inheriting from this one that implements the pop, push, and read
    operations. Please see the documentation for self.pop. self.push,
    and self.read for more details.
    """
    __metaclass__ = ABCMeta

    def __init__(self, batch_size, embedding_size):
        """
        Constructor for the Struct object. The data of the Struct are
        stored in two parts. self.contents is a matrix containing a list
        of vectors. Each of these vectors is assigned a number known as
        its "strength," which is stored in self.strengths. For an item
        in self.contents to be assigned a strength of 1 means that it is
        fully in the data structure, and for it to have a strength of 0
        means it is deleted (but perhaps was previously in the
        structure).

        :type batch_size: int
        :param batch_size: The number of trials in each mini-batch (see
            class introduction)

        :type embedding_size: int
        :param embedding_size: The size of the vectors stored in this
            Struct
        """
        super(Struct, self).__init__()
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self._zeros = Variable(torch.zeros(batch_size))
        self.contents = Variable(torch.FloatTensor(0))
        self.strengths = Variable(torch.FloatTensor(0))
        return

    def forward(self, v, u, d, r=None):
        """
        Performs the following three operations:
            - Pop something from the data structure
            - Push something onto the data structure
            - Read an element of the data structure.

        :type v: torch.FloatTensor
        :param v: The value that will be pushed to the data structure

        :type u: float
        :param u: The total strength of values that will be popped from
            the data structure

        :type d: float
        :param d: The strength with which v will be pushed to the data
            structure

        :rtype: torch.FloatTensor
        :return: The value read from the data structure
        """
        self.pop(u)
        self.push(v, d)
        if r is not None:
            read_strength = r
        elif self._read_strength is not None:
            read_strength = self._read_strength
        else:
            read_strength = 1
        return self.read(read_strength)

    @abstractmethod
    def pop(self, strength):
        """
        Removes something from the data structure. This function needs
        to modify self.strengths, since deleted material remains in
        self.contents, but is assigned a strength of 0.

        :type strength: float
        :param strength: The quantity of items to pop, measured in terms
            of total strength

        :return: None
        """
        raise NotImplementedError('Missing implementation for pop')

    @abstractmethod
    def push(self, value, strength):
        """
        Adds something to the data structure. This function needs to
        modify both self.contents and self.strengths.

        :type value: torch.FloatTensor
        :param value: The value to be added to the data structure

        :type strength: float
        :param strength: The strength with which value will be added to
            the data structure

        :return: None
        """
        raise NotImplementedError('Missing implementation for push')

    @abstractmethod
    def read(self, strength):
        """
        Reads a value from the data structure. This function should not
        modify anything, but should return the value read.

        :type strength: float
        :param strength: The quantity of items to read, measured in
            terms of total strength.

        :rtype: torch.FloatTensor
        :return: The item(s) read from the data structure. These should
            be combined into a single tensor
        """
        raise NotImplementedError('Missing implementation for read')

    @property
    def read_strength(self):
        return 1.0


def tensor_to_string(tensor):
    """
    Formats a torch.FloatTensor as a string.

    :type tensor: torch.FloatTensor
    :param tensor: A tensor

    :rtype str
    :return: A string describing tensor
    """
    return '\t'.join('{:.4f} '.format(x) for x in tensor)


def to_string(obj):
    """
    Formats a PyTorch object as a string.

    :param obj: A PyTorch object (tensor or Variable)

    :rtype: str
    :return: A string description of obj
    """
    if isinstance(obj, torch.FloatTensor):
        return tensor_to_string(obj)
    elif isinstance(obj, Variable):
        return tensor_to_string(obj.data)
    else:
        return str(obj)


class SimpleStruct(Struct):
    """
    Abstract class that subsumes the stack and the queue. This class is
    intended for implementing data structures that have the following
    behavior:
        - self._values is a list of vectors represented by a matrix
        - popping consists of removing items from the structure in a
            cascading fashion
        - pushing consists of inserting an item at some position in the
            list of vectors
        - reading consists of taking the average of a cascade of items,
            weighted by their strengths.

    To use this class, the user must override self._pop_indices,
    self._push_index, and self_read_indices. Doing so specifies the
    direction of the popping and reading cascades, as well as the
    position in which pushed items are inserted. See Stack and Queue
    below for examples.
    """
    __metaclass__ = ABCMeta

    def __init__(self, batch_size, embedding_size, k=None, device=None):
        """
        Constructor for the SimpleStruct object.

        :type batch_size: int
        :param batch_size: The number of trials in each mini-batch

        :type embedding_size: int
        :param embedding_size: The size of the vectors stored in this
            SimpleStruct
        """
        super(SimpleStruct, self).__init__(batch_size, embedding_size)
        operations = [Operation.push, Operation.pop]
        self._reg_trackers = [None for _ in operations]
        self._read_strength = k
        self._values = []
        self._strengths = []
        self._device = device

    def init_contents(self, xs):
        """
        Initialize the SimpleStruct's contents to a specified collection
        of values. Each value will have a strength of 1.

        :type xs: Variable
        :param xs: An array of values that will be placed on the
            SimpleStruct. The dimensions should be [t, batch size,
            read size], where t is the number of values that will be
            placed on the SimpleStruct

        :return: None
        """
        length = xs.size(0)
        self._values = torch.unbind(xs)
        self._strengths = [Variable(torch.ones(self.batch_size, device=self._device)) for _ in length]

    def __len__(self):
        return len(self._values)
    """ Struct Operations """

    @abstractmethod
    def _pop_indices(self):
        """
        Specifies the direction of the popping cascade. See self.pop for
        details on the popping operation of the SimpleStruct. This
        function should either be a generator or return an iterator.

        :rtype: Iterator
        :return: An iterator looping over indices of self._values in
            the order of the popping cascade
        """
        raise NotImplementedError('Missing implementation for _pop_indices')

    @abstractmethod
    def _push_index(self):
        """
        Specifies the location where a pushed item is inserted. See
        self.push for details on the pushing operation of the
        SimpleStruct.

        :rtype: int
        :return: The index of an item in self._values after it has been
            pushed to the SimpleStruct
        """
        raise NotImplementedError('Missing implementation for _push_index')

    @abstractmethod
    def _read_indices(self):
        """
        Specifies the direction of the reading cascade. See self.read
        for details on the reading operation of the SimpleStruct. This
        function should either be a generator or return an iterator.

        :rtype: Iterator
        :return: An iterator looping over indices of self._values in
            the order of the reading cascade
        """
        raise NotImplementedError('Missing implementation for _read_indices')

    @property
    def read_strength(self):
        return self._read_strength

    def pop(self, strength):
        """
        Popping is done by decreasing the strength of items in the
        SimpleStruct until they reach a strength of 0. The pop operation
        begins with an amount of strength specified by the strength
        parameter, and this amount is "consumed" such that the total
        amount of strength subtracted is equal to the initial amount of
        strength. When an item reaches a strength of 0, but the amoount
        of remaining strength is greater than 0, the remaining strength
        is used to decrease the strength of the next item. The order in
        which the items are popped is determined by self._pop_indices.

        :type strength: Variable
        :param strength: The total amount of items to pop, measured by
            strength

        :return: None
        """
        self._track_reg(strength, Operation.pop)
        for i in self._pop_indices():
            local_strength = relu(self._strengths[i] - strength)
            strength = relu(strength - self._strengths[i])
            self._strengths[i] = local_strength
            if all(strength == 0):
                break

    def push(self, value, strength):
        """
        The push operation inserts a vector and a strength somewhere in
        self._values and self._strengths. The location of the new item
        is determined by self._push_index, which gives the index of the
        new item in self._values and self._strengths after the push
        operation is complete.

        :type value: Variable
        :param value: [batch_size x embedding_size] tensor to be pushed to
        the SimpleStruct

        :type strength: Variable
        :param strength: [batch_size] tensor of strengths with which value
        will be pushed

        :return: None
        """
        self._track_reg(strength, Operation.push)
        push_index = self._push_index()
        self._values.insert(push_index, value)
        self._strengths.insert(push_index, strength)

    def read(self, strength):
        """
        The read operation looks at the first few items on the stack, in
        the order determined by self._read_indices, such that the total
        strength of these items is equal to the value of the strength
        parameter. If necessary, the strength of the last vector is
        reduced so that the total strength of the items read is exactly
        equal to the strength parameter. The output of the read
        operation is computed by taking the sum of all the vectors
        looked at, weighted by their strengths.

        :type strength: float
        :param strength: The total amount of vectors to look at,
            measured by their strengths

        :rtype: Variable
        :return: The output of the read operation, described above
        """
        summary = Variable(torch.zeros([self.batch_size, self.embedding_size], device=self._device))
        strength_used = Variable(torch.zeros(self.batch_size, device=self._device))
        for i in self._read_indices():
            strength_weight = torch.min(self._strengths[i], relu(strength - strength_used))
            strength_weight = strength_weight.view(self.batch_size, 1)
            strength_weight = strength_weight.repeat(1, self.embedding_size)
            summary += strength_weight * self._values[i]
            strength_used = strength_used + self._strengths[i]
            if all(strength_used == strength):
                break
        return summary

    def set_reg_tracker(self, reg_tracker, operation):
        """
        Regularize an operation on this struct.

        :type reg_tracker: regularization.InterfaceRegTracker
        :param reg_tracker: Tracker that should be used to regularize.

        :type operation: Operation
        :param operation: Enum specifying which operation should be
        regularized.

        """
        self._reg_trackers[operation] = reg_tracker

    def _track_reg(self, strength, operation):
        """
        Private method to track regularization on interface calls.

        :type strength: Variable
        :param strength: Strength vector given to pop/push call.

        :type operation: Operation
        :param operation: Operation type specified by enum.

        """
        reg_tracker = self._reg_trackers[operation]
        if reg_tracker is not None:
            reg_tracker.regularize(strength)
    """ Reporting """

    def print_summary(self, batch):
        """
        Prints self._values and self._strengths to the console for a
        particular batch.

        :type batch: int
        :param batch: The number of the batch to print information for

        :return: None
        """
        if batch < 0 or batch >= self.batch_size:
            raise IndexError('There is no batch {}.'.format(batch))
        None
        None
        for t in reversed(range(len(self))):
            v_str = to_string(self._values[t][batch, :])
            s = self._strengths[t][batch].data.item()
            None

    def log(self):
        """
        Prints self._values and self._strengths to the console for all
        batches.

        :return: None
        """
        for b in range(self.batch_size):
            None
            self.print_summary(b)


def bottom_to_top(num_steps):
    return range(num_steps)


def top(num_steps):
    return num_steps


class Queue(SimpleStruct):
    """
    A neural queue (first in, first out). Items are popped and read from
    top-to-bottom, and items are pushed to the bottom.
    """

    def _pop_indices(self):
        return bottom_to_top(len(self))

    def _push_index(self):
        return top(len(self))

    def _read_indices(self):
        return bottom_to_top(len(self))


class InputBuffer(Queue):
    """
    A read-only neural queue.
    """

    def forward(self, u):
        """
        Skip the push step.

        :type u: float
        :param u: The total strength of values that will be popped from
            the data structure

        :rtype: torch.FloatTensor
        :return: The value read from the data structure
        """
        self.pop(u)
        return self.read(1.0)


_MAX_COUNT = 100000


def binary_reg_fn(strengths):
    """ Function that is low around 0 and 1. """
    term = 3.25 * strengths - 1.625
    return 1 / (1 + torch.pow(term, 12))


class InterfaceRegTracker(object):
    """
    Compute arbitrary regularization function on struct interface.
    """

    def __init__(self, reg_weight, reg_fn=binary_reg_fn):
        """
        Constructor for StructInterfaceLoss.

        :type reg_weight: float
        :param reg_weight: Linear weight for regularization loss.

        :type reg_fn: function
        :param reg_fn: Regularization function to apply over 1D tensor

        """
        self._reg_weight = reg_weight
        self._reg_fn = reg_fn
        self._loss = Variable(torch.zeros([1]))
        self._count = 0

    @property
    def reg_weight(self):
        return self._reg_weight

    @property
    def loss(self):
        return self._reg_weight * self._loss / self._count

    def regularize(self, strengths):
        assert self._count < _MAX_COUNT, 'Max regularization count exceeded. Are you calling reg_tracker.reset()?'
        losses = self._reg_fn(strengths)
        self._loss += torch.sum(losses)
        self._count += len(losses)

    def reset(self):
        self._loss = Variable(torch.zeros([1]))
        self._count = 0


class OutputBuffer(Queue):
    """
    A write-only neural queue.
    """

    def forward(self, v, d):
        """
        Only perform the push step.

        :type v: torch.FloatTensor
        :param v: The value that will be pushed to the data structure

        :type d: float
        :param d: The strength with which v will be pushed to the data
            structure

        :return: None
        """
        self.push(v, d)


def top_to_bottom(num_steps):
    return reversed(range(num_steps))


class Stack(SimpleStruct):
    """
    A neural stack (last in, first out). Items are popped and read from
    the top of the stack to the bottom, and items are pushed to the top.
    """

    def _pop_indices(self):
        return top_to_bottom(len(self))

    def _push_index(self):
        return top(len(self))

    def _read_indices(self):
        return top_to_bottom(len(self))


class NullStruct(Struct):
    """Neural datastructure that always reads a zero vector.

    This is useful for establishing baseline performance without a
    neural datastructure.
    """

    def pop(self, strength):
        pass

    def push(self, value, strength):
        pass

    def read(self, strength):
        return Variable(torch.zeros([self.batch_size, self.embedding_size]))


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ControlLayer,
     lambda: ([], {'input_size': 4, 'stack_size': 4, 'vision': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (InputBuffer,
     lambda: ([], {'batch_size': 4, 'embedding_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (OutputBuffer,
     lambda: ([], {'batch_size': 4, 'embedding_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_viking_sudo_rm_StackNN(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

