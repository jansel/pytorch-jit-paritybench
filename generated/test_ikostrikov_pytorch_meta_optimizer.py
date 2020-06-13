import sys
_module = sys.modules[__name__]
del sys
data = _module
layer_norm = _module
layer_norm_lstm = _module
main = _module
meta_optimizer = _module
model = _module
utils = _module

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


import torch


import torch.nn as nn


import torch.nn.functional as F


from torch.autograd import Variable


import torch.optim as optim


from functools import reduce


import math


class LayerNorm1D(nn.Module):

    def __init__(self, num_outputs, eps=1e-05, affine=True):
        super(LayerNorm1D, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, num_outputs))
        self.bias = nn.Parameter(torch.zeros(1, num_outputs))

    def forward(self, inputs):
        input_mean = inputs.mean(1, keepdim=True).expand_as(inputs)
        input_std = inputs.std(1, keepdim=True).expand_as(inputs)
        x = (inputs - input_mean) / (input_std + self.eps)
        return x * self.weight.expand_as(x) + self.bias.expand_as(x)


class LayerNormLSTMCell(nn.Module):

    def __init__(self, num_inputs, num_hidden, forget_gate_bias=-1):
        super(LayerNormLSTMCell, self).__init__()
        self.forget_gate_bias = forget_gate_bias
        self.num_hidden = num_hidden
        self.fc_i2h = nn.Linear(num_inputs, 4 * num_hidden)
        self.fc_h2h = nn.Linear(num_hidden, 4 * num_hidden)
        self.ln_i2h = LayerNorm1D(4 * num_hidden)
        self.ln_h2h = LayerNorm1D(4 * num_hidden)
        self.ln_h2o = LayerNorm1D(num_hidden)

    def forward(self, inputs, state):
        hx, cx = state
        i2h = self.fc_i2h(inputs)
        h2h = self.fc_h2h(hx)
        x = self.ln_i2h(i2h) + self.ln_h2h(h2h)
        gates = x.split(self.num_hidden, 1)
        in_gate = F.sigmoid(gates[0])
        forget_gate = F.sigmoid(gates[1] + self.forget_gate_bias)
        out_gate = F.sigmoid(gates[2])
        in_transform = F.tanh(gates[3])
        cx = forget_gate * cx + in_gate * in_transform
        hx = out_gate * F.tanh(self.ln_h2o(cx))
        return hx, cx


def preprocess_gradients(x):
    p = 10
    eps = 1e-06
    indicator = (x.abs() > math.exp(-p)).float()
    x1 = (x.abs() + eps).log() / p * indicator - (1 - indicator)
    x2 = x.sign() * indicator + math.exp(p) * x * (1 - indicator)
    return torch.cat((x1, x2), 1)


class MetaOptimizer(nn.Module):

    def __init__(self, model, num_layers, hidden_size):
        super(MetaOptimizer, self).__init__()
        self.meta_model = model
        self.hidden_size = hidden_size
        self.linear1 = nn.Linear(3, hidden_size)
        self.ln1 = LayerNorm1D(hidden_size)
        self.lstms = []
        for i in range(num_layers):
            self.lstms.append(LayerNormLSTMCell(hidden_size, hidden_size))
        self.linear2 = nn.Linear(hidden_size, 1)
        self.linear2.weight.data.mul_(0.1)
        self.linear2.bias.data.fill_(0.0)

    def cuda(self):
        super(MetaOptimizer, self)
        for i in range(len(self.lstms)):
            self.lstms[i]

    def reset_lstm(self, keep_states=False, model=None, use_cuda=False):
        self.meta_model.reset()
        self.meta_model.copy_params_from(model)
        if keep_states:
            for i in range(len(self.lstms)):
                self.hx[i] = Variable(self.hx[i].data)
                self.cx[i] = Variable(self.cx[i].data)
        else:
            self.hx = []
            self.cx = []
            for i in range(len(self.lstms)):
                self.hx.append(Variable(torch.zeros(1, self.hidden_size)))
                self.cx.append(Variable(torch.zeros(1, self.hidden_size)))
                if use_cuda:
                    self.hx[i], self.cx[i] = self.hx[i], self.cx[i]

    def forward(self, x):
        x = F.tanh(self.ln1(self.linear1(x)))
        for i in range(len(self.lstms)):
            if x.size(0) != self.hx[i].size(0):
                self.hx[i] = self.hx[i].expand(x.size(0), self.hx[i].size(1))
                self.cx[i] = self.cx[i].expand(x.size(0), self.cx[i].size(1))
            self.hx[i], self.cx[i] = self.lstms[i](x, (self.hx[i], self.cx[i]))
            x = self.hx[i]
        x = self.linear2(x)
        return x.squeeze()

    def meta_update(self, model_with_grads, loss):
        grads = []
        for module in model_with_grads.children():
            grads.append(module._parameters['weight'].grad.data.view(-1))
            grads.append(module._parameters['bias'].grad.data.view(-1))
        flat_params = self.meta_model.get_flat_params()
        flat_grads = preprocess_gradients(torch.cat(grads))
        inputs = Variable(torch.cat((flat_grads, flat_params.data), 1))
        flat_params = flat_params + self(inputs)
        self.meta_model.set_flat_params(flat_params)
        self.meta_model.copy_params_to(model_with_grads)
        return self.meta_model.model


class FastMetaOptimizer(nn.Module):

    def __init__(self, model, num_layers, hidden_size):
        super(FastMetaOptimizer, self).__init__()
        self.meta_model = model
        self.linear1 = nn.Linear(6, 2)
        self.linear1.bias.data[0] = 1

    def forward(self, x):
        x = F.sigmoid(self.linear1(x))
        return x.split(1, 1)

    def reset_lstm(self, keep_states=False, model=None, use_cuda=False):
        self.meta_model.reset()
        self.meta_model.copy_params_from(model)
        if keep_states:
            self.f = Variable(self.f.data)
            self.i = Variable(self.i.data)
        else:
            self.f = Variable(torch.zeros(1, 1))
            self.i = Variable(torch.zeros(1, 1))
            if use_cuda:
                self.f = self.f
                self.i = self.i

    def meta_update(self, model_with_grads, loss):
        grads = []
        for module in model_with_grads.children():
            grads.append(module._parameters['weight'].grad.data.view(-1).
                unsqueeze(-1))
            grads.append(module._parameters['bias'].grad.data.view(-1).
                unsqueeze(-1))
        flat_params = self.meta_model.get_flat_params().unsqueeze(-1)
        flat_grads = torch.cat(grads)
        self.i = self.i.expand(flat_params.size(0), 1)
        self.f = self.f.expand(flat_params.size(0), 1)
        loss = loss.expand_as(flat_grads)
        inputs = Variable(torch.cat((preprocess_gradients(flat_grads),
            flat_params.data, loss), 1))
        inputs = torch.cat((inputs, self.f, self.i), 1)
        self.f, self.i = self(inputs)
        flat_params = self.f * flat_params - self.i * Variable(flat_grads)
        flat_params = flat_params.view(-1)
        self.meta_model.set_flat_params(flat_params)
        self.meta_model.copy_params_to(model_with_grads)
        return self.meta_model.model


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(28 * 28, 32)
        self.linear2 = nn.Linear(32, 10)

    def forward(self, inputs):
        x = inputs.view(-1, 28 * 28)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return F.log_softmax(x)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_ikostrikov_pytorch_meta_optimizer(_paritybench_base):
    pass
    def test_000(self):
        self._check(FastMetaOptimizer(*[], **{'model': 4, 'num_layers': 1, 'hidden_size': 4}), [torch.rand([6, 6])], {})

    def test_001(self):
        self._check(LayerNorm1D(*[], **{'num_outputs': 4}), [torch.rand([4, 4, 4, 4])], {})

