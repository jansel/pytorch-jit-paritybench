import sys
_module = sys.modules[__name__]
del sys
conf = _module
examples = _module
server = _module
tutorials = _module
advanced = _module
federated_sms_spam_prediction = _module
handcrafted_GRU = _module
preprocess = _module
monitor_network_traffic = _module
split_neural_network = _module
websockets_mnist = _module
run_websocket_client = _module
start_websocket_servers = _module
grid = _module
federated_learning = _module
mnist = _module
spam_prediction = _module
handcrafted_GRU = _module
websocket = _module
deploy_workers = _module
pen_testing = _module
steal_data_over_sockets = _module
run_websocket_server = _module
setup = _module
syft = _module
codes = _module
common = _module
util = _module
dependency_check = _module
exceptions = _module
execution = _module
action = _module
communication = _module
computation = _module
placeholder = _module
placeholder_id = _module
plan = _module
protocol = _module
role = _module
role_assignments = _module
state = _module
tracing = _module
translation = _module
abstract = _module
default = _module
torchscript = _module
type_wrapper = _module
federated = _module
fl_client = _module
fl_job = _module
floptimizer = _module
frameworks = _module
keras = _module
hook = _module
layers = _module
constructor = _module
model = _module
sequential = _module
tensorflow = _module
torch = _module
dp = _module
pate = _module
fl = _module
dataloader = _module
dataset = _module
utils = _module
functions = _module
he = _module
ciphertext = _module
context = _module
decryptor = _module
encryption_params = _module
encryptor = _module
evaluator = _module
integer_encoder = _module
key_generator = _module
modulus = _module
plaintext = _module
public_key = _module
secret_key = _module
base_converter = _module
global_variable = _module
numth = _module
operations = _module
rlwe = _module
rns_base = _module
rns_tool = _module
paillier = _module
hook = _module
hook_args = _module
linalg = _module
lr = _module
mpc = _module
beaver = _module
fss = _module
primitives = _module
securenn = _module
spdz = _module
nn = _module
conv = _module
functional = _module
pool = _module
rnn = _module
tensors = _module
decorators = _module
logging = _module
interpreters = _module
additive_shared = _module
autograd = _module
build_gradients = _module
gradients = _module
gradients_core = _module
native = _module
numpy = _module
paillier = _module
polynomial = _module
precision = _module
private = _module
torch_attributes = _module
generic = _module
hookable = _module
message_handler = _module
object = _module
sendable = _module
tensor = _module
attributes = _module
pointers = _module
string = _module
overload = _module
remote = _module
types = _module
id_provider = _module
metrics = _module
object_storage = _module
callable_pointer = _module
multi_pointer = _module
object_pointer = _module
object_wrapper = _module
pointer_dataset = _module
pointer_plan = _module
pointer_tensor = _module
string_pointer = _module
abstract_grid = _module
authentication = _module
account = _module
credential = _module
gcloud = _module
test = _module
terraform_notebook = _module
terraform_script = _module
grid_client = _module
network = _module
nodes_manager = _module
peer_events = _module
private_grid = _module
public_grid = _module
webrtc_connection = _module
messaging = _module
message = _module
sandbox = _module
serde = _module
compression = _module
msgpack = _module
native_serde = _module
proto = _module
torch_serde = _module
protobuf = _module
torch_serde = _module
syft_serializable = _module
version = _module
workers = _module
base = _module
message_handler = _module
node_client = _module
tfe = _module
virtual = _module
websocket_client = _module
websocket_server = _module
test_util = _module
conftest = _module
efficiency = _module
assertions = _module
test_activations_time = _module
test_linalg_time = _module
test_communication = _module
test_package_wrapper = _module
test_placeholder = _module
test_plan = _module
test_protocol = _module
test_role = _module
test_role_assignments = _module
test_state = _module
test_translation = _module
test_callable_pointer = _module
test_dataset_pointer = _module
test_multi_pointer = _module
test_pointer_plan = _module
test_pointer_tensor = _module
test_autograd = _module
test_functions = _module
test_gc = _module
test_hookable = _module
test_id_provider = _module
test_logging = _module
test_object_storage = _module
test_private = _module
test_string = _module
test_sequential = _module
test_message = _module
test_notebooks = _module
test_msgpack_serde = _module
test_msgpack_serde_full = _module
test_protobuf_serde = _module
test_protobuf_serde_full = _module
serde_helpers = _module
test_dependency_check = _module
test_exceptions = _module
test_grid = _module
test_local_worker = _module
test_sandbox = _module
test_udacity = _module
differential_privacy = _module
test_pate = _module
test_dataloader = _module
test_dataset = _module
test_utils = _module
test_hook = _module
test_hook_args = _module
test_lr = _module
test_operations = _module
test_crypto_store = _module
test_fss = _module
test_multiparty_nn = _module
test_securenn = _module
test_functional = _module
test_nn = _module
test_additive_shared = _module
test_fv = _module
test_native = _module
test_numpy = _module
test_paillier = _module
test_parameter = _module
test_polynomial = _module
test_precision = _module
test_tensor = _module
test_federated_learning = _module
test_hook = _module
test_base = _module
test_virtual = _module
test_websocket_worker = _module
test_worker = _module

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


import numpy as np


from torch import nn


import torch.nn.functional as F


import torch


import torch.nn as nn


import torch.nn.functional as f


import torch.optim as optim


import logging


from typing import List


from typing import Tuple


from typing import Union


import copy


import inspect


import warnings


from typing import Dict


from typing import Any


from functools import wraps


from math import inf


import math


import torch as th


from torch.nn import Module


from torch.nn import init


import re


from types import ModuleType


from collections import OrderedDict


from itertools import starmap


import numpy


from torch import Tensor


from functools import partial


from torch.nn import Parameter


class GRUCell(nn.Module):

    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.fc_ir = nn.Linear(input_size, hidden_size, bias=bias)
        self.fc_hr = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.fc_iz = nn.Linear(input_size, hidden_size, bias=bias)
        self.fc_hz = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.fc_in = nn.Linear(input_size, hidden_size, bias=bias)
        self.fc_hn = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.init_parameters()

    def init_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, h):
        x = x.view(-1, x.shape[1])
        i_r = self.fc_ir(x)
        h_r = self.fc_hr(h)
        i_z = self.fc_iz(x)
        h_z = self.fc_hz(h)
        i_n = self.fc_in(x)
        h_n = self.fc_hn(h)
        resetgate = F.sigmoid(i_r + h_r)
        inputgate = F.sigmoid(i_z + h_z)
        newgate = F.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (h - newgate)
        return hy


class GRU(nn.Module):

    def __init__(self, vocab_size, output_size=1, embedding_dim=50,
        hidden_dim=10, bias=True, dropout=0.2):
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_size = output_size
        self.dropout = nn.Dropout(p=dropout)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru_cell = GRUCell(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, h):
        batch_size = x.shape[0]
        if h.shape[0] != batch_size:
            h = h[:batch_size, :].contiguous()
        x = self.embedding(x)
        for t in range(x.shape[1]):
            h = self.gru_cell(x[:, (t), :], h)
        out = h.contiguous().view(-1, self.hidden_dim)
        out = self.dropout(out)
        sig_out = self.sigmoid(self.fc(out))
        return sig_out, h


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = f.relu(self.conv1(x))
        x = f.max_pool2d(x, 2, 2)
        x = f.relu(self.conv2(x))
        x = f.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = f.relu(self.fc1(x))
        x = self.fc2(x)
        return f.log_softmax(x, dim=1)


class GRUCell(nn.Module):

    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.fc_ir = nn.Linear(input_size, hidden_size, bias=bias)
        self.fc_hr = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.fc_iz = nn.Linear(input_size, hidden_size, bias=bias)
        self.fc_hz = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.fc_in = nn.Linear(input_size, hidden_size, bias=bias)
        self.fc_hn = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.init_parameters()

    def init_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, h):
        x = x.view(-1, x.shape[1])
        i_r = self.fc_ir(x)
        h_r = self.fc_hr(h)
        i_z = self.fc_iz(x)
        h_z = self.fc_hz(h)
        i_n = self.fc_in(x)
        h_n = self.fc_hn(h)
        resetgate = F.sigmoid(i_r + h_r)
        inputgate = F.sigmoid(i_z + h_z)
        newgate = F.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (h - newgate)
        return hy


class GRU(nn.Module):

    def __init__(self, vocab_size, output_size=1, embedding_dim=50,
        hidden_dim=10, bias=True, dropout=0.2):
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_size = output_size
        self.dropout = nn.Dropout(p=dropout)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru_cell = GRUCell(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, h):
        batch_size = x.shape[0]
        if h.shape[0] != batch_size:
            h = h[:batch_size, :].contiguous()
        x = self.embedding(x)
        for t in range(x.shape[1]):
            h = self.gru_cell(x[:, (t), :], h)
        out = h.contiguous().view(-1, self.hidden_dim)
        out = self.dropout(out)
        sig_out = self.sigmoid(self.fc(out))
        return sig_out, h


def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1
    ):
    """
    Overloads torch.nn.functional.conv2d to be able to use MPC on convolutional networks.
    The idea is to build new tensors from input and weight to compute a
    matrix multiplication equivalent to the convolution.
    Args:
        input: input image
        weight: convolution kernels
        bias: optional additive bias
        stride: stride of the convolution kernels
        padding:  implicit paddings on both sides of the input.
        dilation: spacing between kernel elements
        groups: split input into groups, in_channels should be divisible by the number of groups
    Returns:
        the result of the convolution (FixedPrecision Tensor)
    """
    assert len(input.shape) == 4
    assert len(weight.shape) == 4
    stride = torch.nn.modules.utils._pair(stride)
    padding = torch.nn.modules.utils._pair(padding)
    dilation = torch.nn.modules.utils._pair(dilation)
    batch_size, nb_channels_in, nb_rows_in, nb_cols_in = input.shape
    nb_channels_out, nb_channels_kernel, nb_rows_kernel, nb_cols_kernel = (
        weight.shape)
    if bias is not None:
        assert len(bias) == nb_channels_out
    assert nb_channels_in == nb_channels_kernel * groups
    assert nb_channels_in % groups == 0
    assert nb_channels_out % groups == 0
    nb_rows_out = int((nb_rows_in + 2 * padding[0] - dilation[0] * (
        nb_rows_kernel - 1) - 1) / stride[0] + 1)
    nb_cols_out = int((nb_cols_in + 2 * padding[1] - dilation[1] * (
        nb_cols_kernel - 1) - 1) / stride[1] + 1)
    if padding != (0, 0):
        padding_mode = 'constant'
        input = torch.nn.functional.pad(input, (padding[1], padding[1],
            padding[0], padding[0]), padding_mode)
        nb_rows_in += 2 * padding[0]
        nb_cols_in += 2 * padding[1]
    pattern_ind = []
    for ch in range(nb_channels_in):
        for r in range(nb_rows_kernel):
            for c in range(nb_cols_kernel):
                pixel = r * nb_cols_in * dilation[0] + c * dilation[1]
                pattern_ind.append(pixel + ch * nb_rows_in * nb_cols_in)
    im_flat = input.view(batch_size, -1)
    im_reshaped = []
    for cur_row_out in range(nb_rows_out):
        for cur_col_out in range(nb_cols_out):
            offset = cur_row_out * stride[0
                ] * nb_cols_in + cur_col_out * stride[1]
            tmp = [(ind + offset) for ind in pattern_ind]
            im_reshaped.append(im_flat[:, (tmp)])
    im_reshaped = torch.stack(im_reshaped).permute(1, 0, 2)
    weight_reshaped = weight.view(nb_channels_out // groups, -1).t()
    if groups > 1:
        res = []
        chunks_im = torch.chunk(im_reshaped, groups, dim=2)
        chunks_weights = torch.chunk(weight_reshaped, groups, dim=0)
        for g in range(groups):
            tmp = chunks_im[g].matmul(chunks_weights[g])
            res.append(tmp)
        res = torch.cat(res, dim=2)
    else:
        res = im_reshaped.matmul(weight_reshaped)
    if bias is not None:
        if bias.is_wrapper and res.is_wrapper:
            res += bias
        elif bias.is_wrapper:
            res += bias.child
        else:
            res += bias
    res = res.permute(0, 2, 1).view(batch_size, nb_channels_out,
        nb_rows_out, nb_cols_out).contiguous()
    return res


class Conv2d(nn.Module):
    """
    This class tries to be an exact python port of the torch.nn.Conv2d
    module. Because PySyft cannot hook into layers which are implemented in C++,
    our special functionalities (such as encrypted computation) do not work with
    torch.nn.Conv2d and so we must have python ports available for all layer types
    which we seek to use.

    Note: This module is tested to ensure that it outputs the exact output
    values that the main module outputs in the same order that the main module does.

    This module has not yet been tested with GPUs but should work out of the box.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros'):
        """For information on the constructor arguments, please see PyTorch's
        documentation in torch.nn.Conv2d"""
        super().__init__()
        temp_init = th.nn.Conv2d(in_channels=in_channels, out_channels=
            out_channels, kernel_size=kernel_size, stride=stride, padding=
            padding, dilation=dilation, groups=groups, bias=bias,
            padding_mode=padding_mode)
        self.weight = th.Tensor(temp_init.weight).fix_prec()
        if bias:
            self.bias = th.Tensor(temp_init.bias).fix_prec()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = temp_init.stride
        self.padding = temp_init.padding
        self.dilation = temp_init.dilation
        self.groups = groups
        self.padding_mode = padding_mode

    def forward(self, input):
        assert input.shape[1] == self.in_channels
        return conv2d(input, self.weight, self.bias, self.stride, self.
            padding, self.dilation, self.groups)


class AvgPool2d(Module):
    """
    This class is the beginning of an exact python port of the torch.nn.AvgPool2d
    module. Because PySyft cannot hook into layers which are implemented in C++,
    our special functionalities (such as encrypted computation) do not work with
    torch.nn.AvgPool2d and so we must have python ports available for all layer types
    which we seek to use.

    Note that this module has been tested to ensure that it outputs the exact output
    values that the main module outputs in the same order that the main module does.

    However, there is often some rounding error of unknown origin, usually less than
    1e-6 in magnitude.

    This module has not yet been tested with GPUs but should work out of the box.
    """

    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False,
        count_include_pad=True, divisor_override=None):
        """For information on the constructor arguments, please see PyTorch's
        documentation in torch.nn.AvgPool2d"""
        super().__init__()
        assert padding == 0
        assert ceil_mode is False
        assert count_include_pad is True
        assert divisor_override is None
        if stride is None:
            stride = kernel_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override
        self._one_over_kernel_size = 1 / (self.kernel_size * self.kernel_size)

    def forward(self, data):
        batch_size, out_channels, rows, cols = data.shape
        kernel_results = []
        for i in range(0, rows - self.kernel_size + 1, self.stride):
            for j in range(0, cols - self.kernel_size + 1, self.stride):
                kernel_out = data[:, :, i:i + self.kernel_size, j:j + self.
                    kernel_size].sum((2, 3)) * self._one_over_kernel_size
                kernel_results.append(kernel_out.unsqueeze(2))
        pred = th.cat(kernel_results, axis=2).view(batch_size, out_channels,
            int(rows / self.stride), int(cols / self.stride))
        return pred


class TopLevelTraceModel(torch.nn.Module):

    def __init__(self):
        super(TopLevelTraceModel, self).__init__()
        self.w1 = torch.nn.Parameter(torch.randn(3, 1), requires_grad=True)
        self.b1 = torch.nn.Parameter(torch.randn(1), requires_grad=True)

    def forward(self, x):
        x = x @ self.w1 + self.b1
        return x


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_OpenMined_PySyft(_paritybench_base):
    pass
    def test_000(self):
        self._check(GRUCell(*[], **{'input_size': 4, 'hidden_size': 4}), [torch.rand([4, 4, 64, 4]), torch.rand([4, 4, 1024, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(AvgPool2d(*[], **{'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(TopLevelTraceModel(*[], **{}), [torch.rand([3, 3])], {})

