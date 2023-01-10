import sys
_module = sys.modules[__name__]
del sys
IndRNN_onlyrecurrent = _module
Indrnn_densenet = _module
Indrnn_plainnet = _module
Indrnn_residualnet_preact = _module
Indrnn_densenet_returnstates = _module
Indrnn_plainnet_returnstates = _module
Indrnn_residualnet_preact_returnstates = _module
cuda_IndRNN_onlyrecurrent = _module
data = _module
dynamiceval = _module
language_utils = _module
opts = _module
train_language = _module
utils = _module
IndRNN_onlyrecurrent = _module
Indrnn_action_train = _module
Indrnn_densenet = _module
Indrnn_densenet_FA = _module
Indrnn_plainnet = _module
Indrnn_residualnet_preact = _module
cuda_IndRNN_onlyrecurrent = _module
data_reader = _module
utils = _module
cuda_IndRNN_onlyrecurrent = _module
Data_gen = _module
IndRNN_onlyrecurrent = _module
Indrnn_densenet = _module
Indrnn_mnist_train = _module
Indrnn_plainnet = _module
Indrnn_residualnet_preact = _module
cuda_IndRNN_onlyrecurrent = _module
utils = _module
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


from torch.nn import Parameter


import torch.nn as nn


import torch.nn.functional as F


from torch.autograd import Variable


import math


from torch.nn.utils.rnn import pad_packed_sequence as unpack


from torch.nn.utils.rnn import pack_padded_sequence as pack


import torch.nn.init as weight_init


import numpy as np


from collections import OrderedDict


import time


from torch.autograd import Function


from collections import namedtuple


from collections import Counter


import copy


class IndRNNCell_onlyrecurrent(nn.Module):
    """An IndRNN cell with ReLU non-linearity. This is only the recurrent part where the input is already processed with w_{ih} * x + b_{ih}.

    .. math::
        input=w_{ih} * x + b_{ih}
        h' = \\relu(input +  w_{hh} (*) h)
    With (*) being element-wise vector multiplication.

    Args:
        hidden_size: The number of features in the hidden state h

    Inputs: input, hidden
        - **input** (batch, input_size): tensor containing input features
        - **hidden** (batch, hidden_size): tensor containing the initial hidden
          state for each element in the batch.

    Outputs: h'
        - **h'** (batch, hidden_size): tensor containing the next hidden state
          for each element in the batch
    """

    def __init__(self, hidden_size, hidden_max_abs=None, recurrent_init=None):
        super(IndRNNCell_onlyrecurrent, self).__init__()
        self.hidden_size = hidden_size
        self.recurrent_init = recurrent_init
        self.weight_hh = Parameter(torch.Tensor(hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        for name, weight in self.named_parameters():
            if 'weight_hh' in name:
                if self.recurrent_init is None:
                    nn.init.uniform(weight, a=0, b=1)
                else:
                    self.recurrent_init(weight)

    def forward(self, input, hx):
        return F.relu(input + hx * self.weight_hh.unsqueeze(0).expand(hx.size(0), len(self.weight_hh)))


IndRNN_CODE = """
extern "C" {

    __forceinline__ __device__ float reluf(float x)
    {
        return (x > 0.f) ? x : 0.f;
    }

    __forceinline__ __device__ float calc_grad_activation(float x)
    {
        return (x > 0.f) ? 1.f : 0.f;
    }

    __global__ void indrnn_fwd( const float * __restrict__ x,
                            const float * __restrict__ weight_hh, const float * __restrict__ h0,
                            const int len, const int batch, const int hidden_size, 
                            float * __restrict__ h)
    {
        int ncols = batch*hidden_size;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (col >= ncols) return;       
        const float weight_hh_cur = *(weight_hh + (col%hidden_size));
        float cur = *(h0 + col);
        const float *xp = x+col;
        float *hp = h+col;

        for (int row = 0; row < len; ++row)
        {
            cur=reluf(cur*weight_hh_cur+(*xp));
            *hp=cur;
            xp += ncols;
            hp += ncols;            
        }
    }

    __global__ void indrnn_bwd(const float * __restrict__ x,
                             const float * __restrict__ weight_hh, const float * __restrict__ h0,
                             const float * __restrict__ h,
                            const float * __restrict__ grad_h, 
                            const int len, const int batch, const int hidden_size, 
                            float * __restrict__ grad_x,
                            float * __restrict__ grad_weight_hh, float * __restrict__ grad_h0)
    {    
        int ncols = batch*hidden_size;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (col >= ncols) return;        
        const float weight_hh_cur = *(weight_hh + (col%hidden_size));
        float gweight_hh_cur = 0;
        float cur = 0;  // *(grad_last + col);        //0; strange gradient behavior. grad_last and grad_h, one of them is zero.     
        
        const float *xp = x+col + (len-1)*ncols;
        const float *hp = h+col + (len-1)*ncols;      
        float *gxp = grad_x + col + (len-1)*ncols;
        const float *ghp = grad_h + col + (len-1)*ncols;
        

        for (int row = len-1; row >= 0; --row)
        {        
            const float prev_h_val = (row>0) ? (*(hp-ncols)) : (*(h0+col));
            //float h_val_beforeact = prev_h_val*weight_hh_cur+(*xp);
            float gh_beforeact = ((*ghp) + cur)*calc_grad_activation(prev_h_val*weight_hh_cur+(*xp));
            cur = gh_beforeact*weight_hh_cur;
            gweight_hh_cur += gh_beforeact*prev_h_val;
            *gxp = gh_beforeact;

            xp -= ncols;
            hp -= ncols;
            gxp -= ncols;
            ghp -= ncols;        
        }

        atomicAdd(grad_weight_hh + (col%hidden_size), gweight_hh_cur);
        *(grad_h0 +col) = cur;
    }
}
"""


class IndRNN_onlyrecurrent(nn.Module):

    def __init__(self, hidden_size, gradclipvalue=0, hidden_max_abs=None, recurrent_init=None):
        super(IndRNN_onlyrecurrent, self).__init__()
        self.hidden_size = hidden_size
        self.recurrent_init = recurrent_init
        self.weight_hh = Parameter(torch.Tensor(hidden_size))
        self.gradclipvalue = gradclipvalue
        self.reset_parameters()

    def reset_parameters(self):
        for name, weight in self.named_parameters():
            if 'weight_hh' in name:
                if self.recurrent_init is None:
                    nn.init.uniform(weight, a=0, b=1)
                else:
                    self.recurrent_init(weight)

    def forward(self, input, h0=None):
        assert input.dim() == 2 or input.dim() == 3
        if h0 is None:
            h0 = input.data.new(input.size(-2), input.size(-1)).zero_()
        elif h0.size(-1) != input.size(-1) or h0.size(-2) != input.size(-2):
            raise RuntimeError('The initial hidden size must be equal to input_size. Expected {}, got {}'.format(h0.size(), input.size()))
        IndRNN_Compute = IndRNN_Compute_GPU(self.gradclipvalue)
        return IndRNN_Compute(input, self.weight_hh, h0)


class Batch_norm_overtime(nn.Module):

    def __init__(self, hidden_size, seq_len):
        super(Batch_norm_overtime, self).__init__()
        self.bn = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        x = x.permute(1, 2, 0)
        x = self.bn(x.clone())
        x = x.permute(2, 0, 1)
        return x


BN = Batch_norm_overtime


class IndRNNwithBN(nn.Sequential):

    def __init__(self, hidden_size, seq_len, bn_location='bn_before'):
        super(IndRNNwithBN, self).__init__()
        if bn_location == 'bn_before':
            self.add_module('norm1', BN(hidden_size, args.seq_len))
        self.add_module('indrnn1', IndRNN(hidden_size))
        if bn_location == 'bn_after':
            self.add_module('norm1', BN(hidden_size, args.seq_len))
        if bn_location != 'bn_before' and bn_location != 'bn_after':
            None
            assert 2 == 3


class Dropout_overtime_module(nn.Module):

    def __init__(self, p=0.5):
        super(Dropout_overtime_module, self).__init__()
        if p < 0 or p > 1:
            raise ValueError('dropout probability has to be between 0 and 1, but got {}'.format(p))
        self.p = p

    def forward(self, input):
        output = input.clone()
        if self.training and self.p > 0:
            noise = output.data.new(output.size(-2), output.size(-1))
            noise.bernoulli_(1 - self.p).div_(1 - self.p)
            noise = noise.unsqueeze(0).expand_as(output)
            output.mul_(noise)
        return output


class Linear_overtime_module(nn.Module):

    def __init__(self, input_size, hidden_size, bias=True):
        super(Linear_overtime_module, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size, bias=bias)
        self.input_size = input_size
        self.hidden_size = hidden_size

    def forward(self, x):
        y = x.contiguous().view(-1, self.input_size)
        y = self.fc(y)
        y = y.view(x.size()[0], x.size()[1], self.hidden_size)
        return y


Linear_overtime = Linear_overtime_module


class _DenseLayer(nn.Sequential):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, drop_rate_2):
        super(_DenseLayer, self).__init__()
        self.add_module('fc1', Linear_overtime(num_input_features, bn_size * growth_rate))
        self.add_module('IndRNNwithBN1', IndRNNwithBN(bn_size * growth_rate, args.seq_len, args.bn_location))
        if drop_rate_2 > 0:
            self.add_module('drop2', Dropout_overtime_module(drop_rate_2))
        self.add_module('fc2', Linear_overtime(bn_size * growth_rate, growth_rate))
        self.add_module('IndRNNwithBN2', IndRNNwithBN(growth_rate, args.seq_len, args.bn_location))
        if drop_rate > 0:
            self.add_module('drop1', Dropout_overtime_module(drop_rate))

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        return torch.cat([x, new_features], 2)


class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, drop_rate_2):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate, drop_rate_2)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):

    def __init__(self, num_input_features, num_output_features, drop_rate, last_layer=False):
        super(_Transition, self).__init__()
        self.add_module('fc', Linear_overtime(num_input_features, num_output_features))
        self.add_module('IndRNNwithBN', IndRNNwithBN(num_output_features, args.seq_len, args.bn_location))
        if drop_rate > 0:
            self.add_module('drop', Dropout_overtime_module(drop_rate))


class Dropout_overtime(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, p=0.5, training=False):
        output = input.clone()
        noise = input.data.new(input.size(-2), input.size(-1))
        if training:
            noise.bernoulli_(1 - p).div_(1 - p)
            noise = noise.unsqueeze(0).expand_as(input)
            output.mul_(noise)
        ctx.save_for_backward(noise)
        ctx.training = training
        return output

    @staticmethod
    def backward(ctx, grad_output):
        noise, = ctx.saved_tensors
        if ctx.training:
            return grad_output.mul(noise), None, None
        else:
            return grad_output, None, None


dropout_overtime = Dropout_overtime.apply


class stackedIndRNN_encoder(nn.Module):

    def __init__(self, input_size, outputclass):
        super(stackedIndRNN_encoder, self).__init__()
        hidden_size = args.hidden_size
        self.DIs = nn.ModuleList()
        denseinput = Linear_overtime(input_size, hidden_size)
        self.DIs.append(denseinput)
        for x in range(args.num_layers - 1):
            denseinput = Linear_overtime(hidden_size, hidden_size)
            self.DIs.append(denseinput)
        self.RNNs = nn.ModuleList()
        for x in range(args.num_layers):
            rnn = IndRNNwithBN(hidden_size=hidden_size, seq_len=args.seq_len, bn_location=args.bn_location)
            self.RNNs.append(rnn)
        self.classifier = nn.Linear(hidden_size, outputclass, bias=True)
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight_hh' in name:
                param.data.uniform_(0, U_bound)
            if args.u_lastlayer_ini and 'RNNs.' + str(args.num_layers - 1) + '.weight_hh' in name:
                param.data.uniform_(U_lowbound, U_bound)
            if 'fc' in name and 'weight' in name:
                nn.init.kaiming_uniform_(param, a=8, mode='fan_in')
            if 'classifier' in name and 'weight' in name:
                nn.init.kaiming_normal_(param.data)
            if ('norm' in name or 'Norm' in name) and 'weight' in name:
                param.data.fill_(1)
            if 'bias' in name:
                param.data.fill_(0.0)

    def forward(self, input):
        rnnoutputs = {}
        rnnoutputs['outlayer-1'] = input
        for x in range(len(self.RNNs)):
            rnnoutputs['dilayer%d' % x] = self.DIs[x](rnnoutputs['outlayer%d' % (x - 1)])
            rnnoutputs['outlayer%d' % x] = self.RNNs[x](rnnoutputs['dilayer%d' % x])
            if args.dropout > 0:
                rnnoutputs['outlayer%d' % x] = dropout_overtime(rnnoutputs['outlayer%d' % x], args.dropout, self.training)
        temp = rnnoutputs['outlayer%d' % (len(self.RNNs) - 1)][-1]
        output = self.classifier(temp)
        return output


class _residualBlock_ori(nn.Sequential):

    def __init__(self, hidden_size, drop_rate):
        super(_residualBlock_ori, self).__init__()
        self.add_module('fc1', Linear_overtime(hidden_size, hidden_size))
        self.add_module('IndRNNwithBN1', IndRNNwithBN(hidden_size, args.seq_len, args.bn_location))
        if drop_rate > 0:
            self.add_module('drop1', Dropout_overtime_module(drop_rate))
        self.add_module('fc2', Linear_overtime(hidden_size, hidden_size))
        self.add_module('IndRNNwithBN2', IndRNNwithBN(hidden_size, args.seq_len, args.bn_location))
        if drop_rate > 0:
            self.add_module('drop2', Dropout_overtime_module(drop_rate))

    def forward(self, x):
        new_features = super(_residualBlock_ori, self).forward(x)
        new_features = x + new_features
        return new_features


class _residualBlock_preact(nn.Sequential):

    def __init__(self, hidden_size, drop_rate):
        super(_residualBlock_preact, self).__init__()
        self.add_module('IndRNNwithBN1', IndRNNwithBN(hidden_size, args.seq_len, args.bn_location))
        if drop_rate > 0:
            self.add_module('drop1', Dropout_overtime_module(drop_rate))
        self.add_module('fc1', Linear_overtime(hidden_size, hidden_size))
        self.add_module('IndRNNwithBN2', IndRNNwithBN(hidden_size, args.seq_len, args.bn_location))
        if drop_rate > 0:
            self.add_module('drop2', Dropout_overtime_module(drop_rate))
        self.add_module('fc2', Linear_overtime(hidden_size, hidden_size))

    def forward(self, x):
        new_features = super(_residualBlock_preact, self).forward(x)
        new_features = x + new_features
        return new_features


_residualBlock = _residualBlock_preact


class Batch_norm_step_module(nn.Module):

    def __init__(self, hidden_size, seq_len):
        super(Batch_norm_step_module, self).__init__()
        self.hidden_size = hidden_size
        self.max_time_step = seq_len + 50
        self.bns = nn.ModuleList()
        for x in range(self.max_time_step):
            bn = nn.BatchNorm1d(hidden_size)
            self.bns.append(bn)
        for x in range(1, self.max_time_step):
            self.bns[x].weight = self.bns[0].weight
            self.bns[x].bias = self.bns[0].bias

    def forward(self, x):
        output = []
        for t, input_t in enumerate(x.split(1)):
            input_t = input_t.squeeze(dim=0)
            input_tbn = self.bns[t](input_t)
            output.append(input_tbn)
        output = torch.stack(output)
        return output


class FA_timediff(nn.Module):

    def __init__(self):
        super(FA_timediff, self).__init__()

    def forward(self, x):
        new_x_b = x.clone()
        new_x_f = x.clone()
        new_x_f[1:] = x[1:] - x[0:-1]
        new_x_b[0:-1] = x[0:-1] - x[1:]
        y = torch.cat([x, new_x_f, new_x_b], dim=2)
        return y


class FA_timediff_f(nn.Module):

    def __init__(self):
        super(FA_timediff_f, self).__init__()

    def forward(self, x):
        new_x_f = x.clone()
        new_x_f[1:] = x[1:] - x[0:-1]
        y = torch.cat([x, new_x_f], dim=2)
        return y


class Batch_norm_step(nn.Module):

    def __init__(self, hidden_size, seq_len):
        super(Batch_norm_step, self).__init__()
        self.max_time_step = seq_len * 3 + 1
        self.bns = nn.ModuleList()
        for x in range(self.max_time_step):
            bn = nn.BatchNorm1d(hidden_size)
            self.bns.append(bn)
        for x in range(1, self.max_time_step):
            self.bns[x].weight = self.bns[0].weight
            self.bns[x].bias = self.bns[0].bias

    def forward(self, x, BN_start):
        output = []
        for t, x_t in enumerate(x.split(1)):
            x_t = x_t.squeeze(dim=0)
            t_step = t + BN_start
            if t + BN_start >= self.max_time_step:
                t_step = self.max_time_step - 1
            y = self.bns[t_step](x_t)
            output.append(y)
        output = torch.stack(output)
        return output


class Batch_norm_step_maxseq(nn.Module):

    def __init__(self, hidden_size, seq_len):
        super(Batch_norm_step_maxseq, self).__init__()
        self.max_time_step = seq_len * 4 + 1
        self.max_start = seq_len * 2
        self.bns = nn.ModuleList()
        for x in range(self.max_time_step):
            bn = nn.BatchNorm1d(hidden_size)
            self.bns.append(bn)
        for x in range(1, self.max_time_step):
            self.bns[x].weight = self.bns[0].weight
            self.bns[x].bias = self.bns[0].bias

    def forward(self, x, BN_start0):
        output = []
        if BN_start0 > self.max_start:
            BN_start = self.max_start
        else:
            BN_start = BN_start0
        for t, x_t in enumerate(x.split(1)):
            x_t = x_t.squeeze(dim=0)
            t_step = t + BN_start
            y = self.bns[t_step](x_t)
            output.append(y)
        output = torch.stack(output)
        return output


class MyBatchNorm_stepCompute(nn.Module):
    __constants__ = ['track_running_stats', 'momentum', 'eps', 'weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked']

    def __init__(self, hidden_size, seq_len, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super(MyBatchNorm_stepCompute, self).__init__()
        time_d = seq_len * 4 + 1
        self.max_time_step = seq_len * 2
        channel_d = hidden_size
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.axes = 1,
        if self.affine:
            self.weight = Parameter(torch.Tensor(channel_d))
            self.bias = Parameter(torch.Tensor(channel_d))
            self.register_parameter('weight', self.weight)
            self.register_parameter('bias', self.bias)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(time_d, channel_d))
            self.register_buffer('running_var', torch.ones(time_d, channel_d))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight.data.fill_(1.0)
            self.bias.data.fill_(0.0)

    def _check_input_dim(self, input):
        if input.dim() != 3:
            raise ValueError('expected 3D input (got {}D input)'.format(input.dim()))

    def forward(self, input, BN_start0):
        self._check_input_dim(input)
        BN_start = BN_start0
        if BN_start0 > self.max_time_step:
            BN_start = self.max_time_step
        exponential_average_factor = 0.0
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum
        input_t, _, _ = input.size()
        if self.training:
            mean = input.mean(1)
            var = input.var(1, unbiased=False)
            n = input.size()[1]
            with torch.no_grad():
                self.running_mean[BN_start:input_t + BN_start] = exponential_average_factor * mean + (1 - exponential_average_factor) * self.running_mean[BN_start:input_t + BN_start]
                self.running_var[BN_start:input_t + BN_start] = exponential_average_factor * var * n / (n - 1) + (1 - exponential_average_factor) * self.running_var[BN_start:input_t + BN_start]
        else:
            mean = self.running_mean[BN_start:input_t + BN_start]
            var = self.running_var[BN_start:input_t + BN_start]
        input1 = (input - mean[:, None, :]) / torch.sqrt(var[:, None, :] + self.eps)
        if self.affine:
            input1 = input1 * self.weight[None, None, :] + self.bias[None, None, :]
        return input1


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BN,
     lambda: ([], {'hidden_size': 4, 'seq_len': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (Batch_norm_overtime,
     lambda: ([], {'hidden_size': 4, 'seq_len': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (Batch_norm_step,
     lambda: ([], {'hidden_size': 4, 'seq_len': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), 0], {}),
     False),
    (Batch_norm_step_maxseq,
     lambda: ([], {'hidden_size': 4, 'seq_len': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), 0], {}),
     False),
    (Batch_norm_step_module,
     lambda: ([], {'hidden_size': 4, 'seq_len': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Dropout_overtime_module,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (FA_timediff,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FA_timediff_f,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (IndRNNCell_onlyrecurrent,
     lambda: ([], {'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Linear_overtime,
     lambda: ([], {'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (Linear_overtime_module,
     lambda: ([], {'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (MyBatchNorm_stepCompute,
     lambda: ([], {'hidden_size': 4, 'seq_len': 4}),
     lambda: ([torch.rand([4, 4, 4]), 0], {}),
     False),
]

class Test_Sunnydreamrain_IndRNN_pytorch(_paritybench_base):
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

