import sys
_module = sys.modules[__name__]
del sys
dnc = _module
dnc = _module
faiss_index = _module
flann_index = _module
memory = _module
sam = _module
sdnc = _module
sparse_memory = _module
sparse_temporal_memory = _module
util = _module
setup = _module
tasks = _module
adding_task = _module
argmax_task = _module
copy_task = _module
test_gru = _module
test_indexes = _module
test_lstm = _module
test_rnn = _module
test_sam_gru = _module
test_sam_lstm = _module
test_sam_rnn = _module
test_sdnc_gru = _module
test_sdnc_lstm = _module
test_sdnc_rnn = _module
test_utils = _module

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


import torch.nn as nn


import torch as T


from torch.autograd import Variable as var


import numpy as np


from torch.nn.utils.rnn import pad_packed_sequence as pad


from torch.nn.utils.rnn import pack_padded_sequence as pack


from torch.nn.utils.rnn import PackedSequence


from torch.nn.init import orthogonal_


from torch.nn.init import xavier_uniform_


import torch.nn.functional as F


import math


import time


import torch


from torch.autograd import Variable


import re


import string


import warnings


import torch.optim as optim


from torch.nn.utils import clip_grad_norm_


import functools


def cuda(x, grad=False, gpu_id=-1):
    x = x.float() if T.is_tensor(x) else x
    if gpu_id == -1:
        t = T.FloatTensor(x)
        t.requires_grad = grad
        return t
    else:
        t = T.FloatTensor(x.pin_memory()).cuda(gpu_id)
        t.requires_grad = grad
        return t


class DNC(nn.Module):

    def __init__(self, input_size, hidden_size, rnn_type='lstm', num_layers
        =1, num_hidden_layers=2, bias=True, batch_first=True, dropout=0,
        bidirectional=False, nr_cells=5, read_heads=2, cell_size=10,
        nonlinearity='tanh', gpu_id=-1, independent_linears=False,
        share_memory=True, debug=False, clip=20):
        super(DNC, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.num_hidden_layers = num_hidden_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.nr_cells = nr_cells
        self.read_heads = read_heads
        self.cell_size = cell_size
        self.nonlinearity = nonlinearity
        self.gpu_id = gpu_id
        self.independent_linears = independent_linears
        self.share_memory = share_memory
        self.debug = debug
        self.clip = clip
        self.w = self.cell_size
        self.r = self.read_heads
        self.read_vectors_size = self.r * self.w
        self.output_size = self.hidden_size
        self.nn_input_size = self.input_size + self.read_vectors_size
        self.nn_output_size = self.output_size + self.read_vectors_size
        self.rnns = []
        self.memories = []
        for layer in range(self.num_layers):
            if self.rnn_type.lower() == 'rnn':
                self.rnns.append(nn.RNN(self.nn_input_size if layer == 0 else
                    self.nn_output_size, self.output_size, bias=self.bias,
                    nonlinearity=self.nonlinearity, batch_first=True,
                    dropout=self.dropout, num_layers=self.num_hidden_layers))
            elif self.rnn_type.lower() == 'gru':
                self.rnns.append(nn.GRU(self.nn_input_size if layer == 0 else
                    self.nn_output_size, self.output_size, bias=self.bias,
                    batch_first=True, dropout=self.dropout, num_layers=self
                    .num_hidden_layers))
            if self.rnn_type.lower() == 'lstm':
                self.rnns.append(nn.LSTM(self.nn_input_size if layer == 0 else
                    self.nn_output_size, self.output_size, bias=self.bias,
                    batch_first=True, dropout=self.dropout, num_layers=self
                    .num_hidden_layers))
            setattr(self, self.rnn_type.lower() + '_layer_' + str(layer),
                self.rnns[layer])
            if not self.share_memory:
                self.memories.append(Memory(input_size=self.output_size,
                    mem_size=self.nr_cells, cell_size=self.w, read_heads=
                    self.r, gpu_id=self.gpu_id, independent_linears=self.
                    independent_linears))
                setattr(self, 'rnn_layer_memory_' + str(layer), self.
                    memories[layer])
        if self.share_memory:
            self.memories.append(Memory(input_size=self.output_size,
                mem_size=self.nr_cells, cell_size=self.w, read_heads=self.r,
                gpu_id=self.gpu_id, independent_linears=self.
                independent_linears))
            setattr(self, 'rnn_layer_memory_shared', self.memories[0])
        self.output = nn.Linear(self.nn_output_size, self.input_size)
        orthogonal_(self.output.weight)
        if self.gpu_id != -1:
            [x for x in self.rnns]
            [x for x in self.memories]
            self.output

    def _init_hidden(self, hx, batch_size, reset_experience):
        if hx is None:
            hx = None, None, None
        chx, mhx, last_read = hx
        if chx is None:
            h = cuda(T.zeros(self.num_hidden_layers, batch_size, self.
                output_size), gpu_id=self.gpu_id)
            xavier_uniform_(h)
            chx = [((h, h) if self.rnn_type.lower() == 'lstm' else h) for x in
                range(self.num_layers)]
        if last_read is None:
            last_read = cuda(T.zeros(batch_size, self.w * self.r), gpu_id=
                self.gpu_id)
        if mhx is None:
            if self.share_memory:
                mhx = self.memories[0].reset(batch_size, erase=reset_experience
                    )
            else:
                mhx = [m.reset(batch_size, erase=reset_experience) for m in
                    self.memories]
        elif self.share_memory:
            mhx = self.memories[0].reset(batch_size, mhx, erase=
                reset_experience)
        else:
            mhx = [m.reset(batch_size, h, erase=reset_experience) for m, h in
                zip(self.memories, mhx)]
        return chx, mhx, last_read

    def _debug(self, mhx, debug_obj):
        if not debug_obj:
            debug_obj = {'memory': [], 'link_matrix': [], 'precedence': [],
                'read_weights': [], 'write_weights': [], 'usage_vector': []}
        debug_obj['memory'].append(mhx['memory'][0].data.cpu().numpy())
        debug_obj['link_matrix'].append(mhx['link_matrix'][0][0].data.cpu()
            .numpy())
        debug_obj['precedence'].append(mhx['precedence'][0].data.cpu().numpy())
        debug_obj['read_weights'].append(mhx['read_weights'][0].data.cpu().
            numpy())
        debug_obj['write_weights'].append(mhx['write_weights'][0].data.cpu(
            ).numpy())
        debug_obj['usage_vector'].append(mhx['usage_vector'][0].unsqueeze(0
            ).data.cpu().numpy())
        return debug_obj

    def _layer_forward(self, input, layer, hx=(None, None),
        pass_through_memory=True):
        chx, mhx = hx
        input, chx = self.rnns[layer](input.unsqueeze(1), chx)
        input = input.squeeze(1)
        if self.clip != 0:
            output = T.clamp(input, -self.clip, self.clip)
        else:
            output = input
        ξ = output
        if pass_through_memory:
            if self.share_memory:
                read_vecs, mhx = self.memories[0](ξ, mhx)
            else:
                read_vecs, mhx = self.memories[layer](ξ, mhx)
            read_vectors = read_vecs.view(-1, self.w * self.r)
        else:
            read_vectors = None
        return output, (chx, mhx, read_vectors)

    def forward(self, input, hx=(None, None, None), reset_experience=False,
        pass_through_memory=True):
        is_packed = type(input) is PackedSequence
        if is_packed:
            input, lengths = pad(input)
            max_length = lengths[0]
        else:
            max_length = input.size(1) if self.batch_first else input.size(0)
            lengths = [input.size(1)] * max_length if self.batch_first else [
                input.size(0)] * max_length
        batch_size = input.size(0) if self.batch_first else input.size(1)
        if not self.batch_first:
            input = input.transpose(0, 1)
        controller_hidden, mem_hidden, last_read = self._init_hidden(hx,
            batch_size, reset_experience)
        inputs = [T.cat([input[:, (x), :], last_read], 1) for x in range(
            max_length)]
        if self.debug:
            viz = None
        outs = [None] * max_length
        read_vectors = None
        for time in range(max_length):
            for layer in range(self.num_layers):
                chx = controller_hidden[layer]
                m = mem_hidden if self.share_memory else mem_hidden[layer]
                outs[time], (chx, m, read_vectors) = self._layer_forward(inputs
                    [time], layer, (chx, m), pass_through_memory)
                if self.debug:
                    viz = self._debug(m, viz)
                if self.share_memory:
                    mem_hidden = m
                else:
                    mem_hidden[layer] = m
                controller_hidden[layer] = chx
                if read_vectors is not None:
                    outs[time] = T.cat([outs[time], read_vectors], 1)
                else:
                    outs[time] = T.cat([outs[time], last_read], 1)
                inputs[time] = outs[time]
        if self.debug:
            viz = {k: np.array(v) for k, v in viz.items()}
            viz = {k: v.reshape(v.shape[0], v.shape[1] * v.shape[2]) for k,
                v in viz.items()}
        inputs = [self.output(i) for i in inputs]
        outputs = T.stack(inputs, 1 if self.batch_first else 0)
        if is_packed:
            outputs = pack(output, lengths)
        if self.debug:
            return outputs, (controller_hidden, mem_hidden, read_vectors), viz
        else:
            return outputs, (controller_hidden, mem_hidden, read_vectors)

    def __repr__(self):
        s = '\n----------------------------------------\n'
        s += '{name}({input_size}, {hidden_size}'
        if self.rnn_type != 'lstm':
            s += ', rnn_type={rnn_type}'
        if self.num_layers != 1:
            s += ', num_layers={num_layers}'
        if self.num_hidden_layers != 2:
            s += ', num_hidden_layers={num_hidden_layers}'
        if self.bias != True:
            s += ', bias={bias}'
        if self.batch_first != True:
            s += ', batch_first={batch_first}'
        if self.dropout != 0:
            s += ', dropout={dropout}'
        if self.bidirectional != False:
            s += ', bidirectional={bidirectional}'
        if self.nr_cells != 5:
            s += ', nr_cells={nr_cells}'
        if self.read_heads != 2:
            s += ', read_heads={read_heads}'
        if self.cell_size != 10:
            s += ', cell_size={cell_size}'
        if self.nonlinearity != 'tanh':
            s += ', nonlinearity={nonlinearity}'
        if self.gpu_id != -1:
            s += ', gpu_id={gpu_id}'
        if self.independent_linears != False:
            s += ', independent_linears={independent_linears}'
        if self.share_memory != True:
            s += ', share_memory={share_memory}'
        if self.debug != False:
            s += ', debug={debug}'
        if self.clip != 20:
            s += ', clip={clip}'
        s += ')\n' + super(DNC, self).__repr__(
            ) + '\n----------------------------------------\n'
        return s.format(name=self.__class__.__name__, **self.__dict__)


δ = 1e-06


def θ(a, b, dimA=2, dimB=2, normBy=2):
    """Batchwise Cosine distance

  Cosine distance

  Arguments:
      a {Tensor} -- A 3D Tensor (b * m * w)
      b {Tensor} -- A 3D Tensor (b * r * w)

  Keyword Arguments:
      dimA {number} -- exponent value of the norm for `a` (default: {2})
      dimB {number} -- exponent value of the norm for `b` (default: {1})

  Returns:
      Tensor -- Batchwise cosine distance (b * r * m)
  """
    a_norm = T.norm(a, normBy, dimA, keepdim=True).expand_as(a) + δ
    b_norm = T.norm(b, normBy, dimB, keepdim=True).expand_as(b) + δ
    x = T.bmm(a, b.transpose(1, 2)).transpose(1, 2) / (T.bmm(a_norm, b_norm
        .transpose(1, 2)).transpose(1, 2) + δ)
    return x


def σ(input, axis=1):
    """Softmax on an axis

  Softmax on an axis

  Arguments:
      input {Tensor} -- input Tensor

  Keyword Arguments:
      axis {number} -- axis on which to take softmax on (default: {1})

  Returns:
      Tensor -- Softmax output Tensor
  """
    input_size = input.size()
    trans_input = input.transpose(axis, len(input_size) - 1)
    trans_size = trans_input.size()
    input_2d = trans_input.contiguous().view(-1, trans_size[-1])
    soft_max_2d = F.softmax(input_2d, -1)
    soft_max_nd = soft_max_2d.view(*trans_size)
    return soft_max_nd.transpose(axis, len(input_size) - 1)


class Memory(nn.Module):

    def __init__(self, input_size, mem_size=512, cell_size=32, read_heads=4,
        gpu_id=-1, independent_linears=True):
        super(Memory, self).__init__()
        self.mem_size = mem_size
        self.cell_size = cell_size
        self.read_heads = read_heads
        self.gpu_id = gpu_id
        self.input_size = input_size
        self.independent_linears = independent_linears
        m = self.mem_size
        w = self.cell_size
        r = self.read_heads
        if self.independent_linears:
            self.read_keys_transform = nn.Linear(self.input_size, w * r)
            self.read_strengths_transform = nn.Linear(self.input_size, r)
            self.write_key_transform = nn.Linear(self.input_size, w)
            self.write_strength_transform = nn.Linear(self.input_size, 1)
            self.erase_vector_transform = nn.Linear(self.input_size, w)
            self.write_vector_transform = nn.Linear(self.input_size, w)
            self.free_gates_transform = nn.Linear(self.input_size, r)
            self.allocation_gate_transform = nn.Linear(self.input_size, 1)
            self.write_gate_transform = nn.Linear(self.input_size, 1)
            self.read_modes_transform = nn.Linear(self.input_size, 3 * r)
        else:
            self.interface_size = w * r + 3 * w + 5 * r + 3
            self.interface_weights = nn.Linear(self.input_size, self.
                interface_size)
        self.I = cuda(1 - T.eye(m).unsqueeze(0), gpu_id=self.gpu_id)

    def reset(self, batch_size=1, hidden=None, erase=True):
        m = self.mem_size
        w = self.cell_size
        r = self.read_heads
        b = batch_size
        if hidden is None:
            return {'memory': cuda(T.zeros(b, m, w).fill_(0), gpu_id=self.
                gpu_id), 'link_matrix': cuda(T.zeros(b, 1, m, m), gpu_id=
                self.gpu_id), 'precedence': cuda(T.zeros(b, 1, m), gpu_id=
                self.gpu_id), 'read_weights': cuda(T.zeros(b, r, m).fill_(0
                ), gpu_id=self.gpu_id), 'write_weights': cuda(T.zeros(b, 1,
                m).fill_(0), gpu_id=self.gpu_id), 'usage_vector': cuda(T.
                zeros(b, m), gpu_id=self.gpu_id)}
        else:
            hidden['memory'] = hidden['memory'].clone()
            hidden['link_matrix'] = hidden['link_matrix'].clone()
            hidden['precedence'] = hidden['precedence'].clone()
            hidden['read_weights'] = hidden['read_weights'].clone()
            hidden['write_weights'] = hidden['write_weights'].clone()
            hidden['usage_vector'] = hidden['usage_vector'].clone()
            if erase:
                hidden['memory'].data.fill_(0)
                hidden['link_matrix'].data.zero_()
                hidden['precedence'].data.zero_()
                hidden['read_weights'].data.fill_(0)
                hidden['write_weights'].data.fill_(0)
                hidden['usage_vector'].data.zero_()
        return hidden

    def get_usage_vector(self, usage, free_gates, read_weights, write_weights):
        usage = usage + (1 - usage) * (1 - T.prod(1 - write_weights, 1))
        ψ = T.prod(1 - free_gates.unsqueeze(2) * read_weights, 1)
        return usage * ψ

    def allocate(self, usage, write_gate):
        usage = δ + (1 - δ) * usage
        batch_size = usage.size(0)
        sorted_usage, φ = T.topk(usage, self.mem_size, dim=1, largest=False)
        v = var(sorted_usage.data.new(batch_size, 1).fill_(1))
        cat_sorted_usage = T.cat((v, sorted_usage), 1)
        prod_sorted_usage = T.cumprod(cat_sorted_usage, 1)[:, :-1]
        sorted_allocation_weights = (1 - sorted_usage
            ) * prod_sorted_usage.squeeze()
        _, φ_rev = T.topk(φ, k=self.mem_size, dim=1, largest=False)
        allocation_weights = sorted_allocation_weights.gather(1, φ_rev.long())
        return allocation_weights.unsqueeze(1), usage

    def write_weighting(self, memory, write_content_weights,
        allocation_weights, write_gate, allocation_gate):
        ag = allocation_gate.unsqueeze(-1)
        wg = write_gate.unsqueeze(-1)
        return wg * (ag * allocation_weights + (1 - ag) * write_content_weights
            )

    def get_link_matrix(self, link_matrix, write_weights, precedence):
        precedence = precedence.unsqueeze(2)
        write_weights_i = write_weights.unsqueeze(3)
        write_weights_j = write_weights.unsqueeze(2)
        prev_scale = 1 - write_weights_i - write_weights_j
        new_link_matrix = write_weights_i * precedence
        link_matrix = prev_scale * link_matrix + new_link_matrix
        return self.I.expand_as(link_matrix) * link_matrix

    def update_precedence(self, precedence, write_weights):
        return (1 - T.sum(write_weights, 2, keepdim=True)
            ) * precedence + write_weights

    def write(self, write_key, write_vector, erase_vector, free_gates,
        read_strengths, write_strength, write_gate, allocation_gate, hidden):
        hidden['usage_vector'] = self.get_usage_vector(hidden[
            'usage_vector'], free_gates, hidden['read_weights'], hidden[
            'write_weights'])
        write_content_weights = self.content_weightings(hidden['memory'],
            write_key, write_strength)
        alloc, _ = self.allocate(hidden['usage_vector'], allocation_gate *
            write_gate)
        hidden['write_weights'] = self.write_weighting(hidden['memory'],
            write_content_weights, alloc, write_gate, allocation_gate)
        weighted_resets = hidden['write_weights'].unsqueeze(3
            ) * erase_vector.unsqueeze(2)
        reset_gate = T.prod(1 - weighted_resets, 1)
        hidden['memory'] = hidden['memory'] * reset_gate
        hidden['memory'] = hidden['memory'] + T.bmm(hidden['write_weights']
            .transpose(1, 2), write_vector)
        hidden['link_matrix'] = self.get_link_matrix(hidden['link_matrix'],
            hidden['write_weights'], hidden['precedence'])
        hidden['precedence'] = self.update_precedence(hidden['precedence'],
            hidden['write_weights'])
        return hidden

    def content_weightings(self, memory, keys, strengths):
        d = θ(memory, keys)
        return σ(d * strengths.unsqueeze(2), 2)

    def directional_weightings(self, link_matrix, read_weights):
        rw = read_weights.unsqueeze(1)
        f = T.matmul(link_matrix, rw.transpose(2, 3)).transpose(2, 3)
        b = T.matmul(rw, link_matrix)
        return f.transpose(1, 2), b.transpose(1, 2)

    def read_weightings(self, memory, content_weights, link_matrix,
        read_modes, read_weights):
        forward_weight, backward_weight = self.directional_weightings(
            link_matrix, read_weights)
        content_mode = read_modes[:, :, (2)].contiguous().unsqueeze(2
            ) * content_weights
        backward_mode = T.sum(read_modes[:, :, 0:1].contiguous().unsqueeze(
            3) * backward_weight, 2)
        forward_mode = T.sum(read_modes[:, :, 1:2].contiguous().unsqueeze(3
            ) * forward_weight, 2)
        return backward_mode + content_mode + forward_mode

    def read_vectors(self, memory, read_weights):
        return T.bmm(read_weights, memory)

    def read(self, read_keys, read_strengths, read_modes, hidden):
        content_weights = self.content_weightings(hidden['memory'],
            read_keys, read_strengths)
        hidden['read_weights'] = self.read_weightings(hidden['memory'],
            content_weights, hidden['link_matrix'], read_modes, hidden[
            'read_weights'])
        read_vectors = self.read_vectors(hidden['memory'], hidden[
            'read_weights'])
        return read_vectors, hidden

    def forward(self, ξ, hidden):
        m = self.mem_size
        w = self.cell_size
        r = self.read_heads
        b = ξ.size()[0]
        if self.independent_linears:
            read_keys = T.tanh(self.read_keys_transform(ξ).view(b, r, w))
            read_strengths = F.softplus(self.read_strengths_transform(ξ).
                view(b, r))
            write_key = T.tanh(self.write_key_transform(ξ).view(b, 1, w))
            write_strength = F.softplus(self.write_strength_transform(ξ).
                view(b, 1))
            erase_vector = T.sigmoid(self.erase_vector_transform(ξ).view(b,
                1, w))
            write_vector = T.tanh(self.write_vector_transform(ξ).view(b, 1, w))
            free_gates = T.sigmoid(self.free_gates_transform(ξ).view(b, r))
            allocation_gate = T.sigmoid(self.allocation_gate_transform(ξ).
                view(b, 1))
            write_gate = T.sigmoid(self.write_gate_transform(ξ).view(b, 1))
            read_modes = σ(self.read_modes_transform(ξ).view(b, r, 3), -1)
        else:
            ξ = self.interface_weights(ξ)
            read_keys = T.tanh(ξ[:, :r * w].contiguous().view(b, r, w))
            read_strengths = F.softplus(ξ[:, r * w:r * w + r].contiguous().
                view(b, r))
            write_key = T.tanh(ξ[:, r * w + r:r * w + r + w].contiguous().
                view(b, 1, w))
            write_strength = F.softplus(ξ[:, (r * w + r + w)].contiguous().
                view(b, 1))
            erase_vector = T.sigmoid(ξ[:, r * w + r + w + 1:r * w + r + 2 *
                w + 1].contiguous().view(b, 1, w))
            write_vector = T.tanh(ξ[:, r * w + r + 2 * w + 1:r * w + r + 3 *
                w + 1].contiguous().view(b, 1, w))
            free_gates = T.sigmoid(ξ[:, r * w + r + 3 * w + 1:r * w + 2 * r +
                3 * w + 1].contiguous().view(b, r))
            allocation_gate = T.sigmoid(ξ[:, (r * w + 2 * r + 3 * w + 1)].
                contiguous().unsqueeze(1).view(b, 1))
            write_gate = T.sigmoid(ξ[:, (r * w + 2 * r + 3 * w + 2)].
                contiguous()).unsqueeze(1).view(b, 1)
            read_modes = σ(ξ[:, r * w + 2 * r + 3 * w + 3:r * w + 5 * r + 3 *
                w + 3].contiguous().view(b, r, 3), -1)
        hidden = self.write(write_key, write_vector, erase_vector,
            free_gates, read_strengths, write_strength, write_gate,
            allocation_gate, hidden)
        return self.read(read_keys, read_strengths, read_modes, hidden)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_ixaxaar_pytorch_dnc(_paritybench_base):
    pass
