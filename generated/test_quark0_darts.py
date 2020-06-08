import sys
_module = sys.modules[__name__]
del sys
architect = _module
genotypes = _module
model = _module
model_search = _module
operations = _module
test = _module
test_imagenet = _module
train = _module
train_imagenet = _module
train_search = _module
utils = _module
visualize = _module
architect = _module
data = _module
model = _module
model_search = _module
test = _module
train = _module
train_search = _module
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


import numpy as np


import torch.nn as nn


from torch.autograd import Variable


import torch.nn.functional as F


import logging


import torch.utils


import torch.backends.cudnn as cudnn


import random


import math


from collections import namedtuple


def drop_path(x, drop_prob):
    if drop_prob > 0.0:
        keep_prob = 1.0 - drop_prob
        mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).
            bernoulli_(keep_prob))
        x.div_(keep_prob)
        x.mul_(mask)
    return x


OPS = {'none': lambda C, stride, affine: Zero(stride), 'avg_pool_3x3': lambda
    C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1,
    count_include_pad=False), 'max_pool_3x3': lambda C, stride, affine: nn.
    MaxPool2d(3, stride=stride, padding=1), 'skip_connect': lambda C,
    stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C,
    affine=affine), 'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C,
    3, stride, 1, affine=affine), 'sep_conv_5x5': lambda C, stride, affine:
    SepConv(C, C, 5, stride, 2, affine=affine), 'sep_conv_7x7': lambda C,
    stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
    'dil_conv_3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2,
    affine=affine), 'dil_conv_5x5': lambda C, stride, affine: DilConv(C, C,
    5, stride, 4, 2, affine=affine), 'conv_7x1_1x7': lambda C, stride,
    affine: nn.Sequential(nn.ReLU(inplace=False), nn.Conv2d(C, C, (1, 7),
    stride=(1, stride), padding=(0, 3), bias=False), nn.Conv2d(C, C, (7, 1),
    stride=(stride, 1), padding=(3, 0), bias=False), nn.BatchNorm2d(C,
    affine=affine))}


class Cell(nn.Module):

    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction,
        reduction_prev):
        super(Cell, self).__init__()
        None
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)
        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat, reduction)

    def _compile(self, C, op_names, indices, concat, reduction):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)
        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, True)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)
            if self.training and drop_prob > 0.0:
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_prob)
            s = h1 + h2
            states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)


class AuxiliaryHeadCIFAR(nn.Module):

    def __init__(self, C, num_classes):
        """assuming input size 8x8"""
        super(AuxiliaryHeadCIFAR, self).__init__()
        self.features = nn.Sequential(nn.ReLU(inplace=True), nn.AvgPool2d(5,
            stride=3, padding=0, count_include_pad=False), nn.Conv2d(C, 128,
            1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.
            Conv2d(128, 768, 2, bias=False), nn.BatchNorm2d(768), nn.ReLU(
            inplace=True))
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class AuxiliaryHeadImageNet(nn.Module):

    def __init__(self, C, num_classes):
        """assuming input size 14x14"""
        super(AuxiliaryHeadImageNet, self).__init__()
        self.features = nn.Sequential(nn.ReLU(inplace=True), nn.AvgPool2d(5,
            stride=2, padding=0, count_include_pad=False), nn.Conv2d(C, 128,
            1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.
            Conv2d(128, 768, 2, bias=False), nn.ReLU(inplace=True))
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class NetworkCIFAR(nn.Module):

    def __init__(self, C, num_classes, layers, auxiliary, genotype):
        super(NetworkCIFAR, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary
        stem_multiplier = 3
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(nn.Conv2d(3, C_curr, 3, padding=1, bias=
            False), nn.BatchNorm2d(C_curr))
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction,
                reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev
        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary,
                num_classes)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input):
        logits_aux = None
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, logits_aux


class NetworkImageNet(nn.Module):

    def __init__(self, C, num_classes, layers, auxiliary, genotype):
        super(NetworkImageNet, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary
        self.stem0 = nn.Sequential(nn.Conv2d(3, C // 2, kernel_size=3,
            stride=2, padding=1, bias=False), nn.BatchNorm2d(C // 2), nn.
            ReLU(inplace=True), nn.Conv2d(C // 2, C, 3, stride=2, padding=1,
            bias=False), nn.BatchNorm2d(C))
        self.stem1 = nn.Sequential(nn.ReLU(inplace=True), nn.Conv2d(C, C, 3,
            stride=2, padding=1, bias=False), nn.BatchNorm2d(C))
        C_prev_prev, C_prev, C_curr = C, C, C
        self.cells = nn.ModuleList()
        reduction_prev = True
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction,
                reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev
        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadImageNet(C_to_auxiliary,
                num_classes)
        self.global_pooling = nn.AvgPool2d(7)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input):
        logits_aux = None
        s0 = self.stem0(input)
        s1 = self.stem1(s0)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, logits_aux


PRIMITIVES = ['none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect',
    'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5']


class MixedOp(nn.Module):

    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))


class Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction,
        reduction_prev):
        super(Cell, self).__init__()
        self.reduction = reduction
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False
                )
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier
        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride)
                self._ops.append(op)

    def forward(self, s0, s1, weights):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(self._ops[offset + j](h, weights[offset + j]) for j, h in
                enumerate(states))
            offset += len(states)
            states.append(s)
        return torch.cat(states[-self._multiplier:], dim=1)


Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')


class Network(nn.Module):

    def __init__(self, C, num_classes, layers, criterion, steps=4,
        multiplier=4, stem_multiplier=3):
        super(Network, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(nn.Conv2d(3, C_curr, 3, padding=1, bias=
            False), nn.BatchNorm2d(C_curr))
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr,
                reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        self._initialize_alphas()

    def new(self):
        model_new = Network(self._C, self._num_classes, self._layers, self.
            _criterion)
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, input):
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = F.softmax(self.alphas_reduce, dim=-1)
            else:
                weights = F.softmax(self.alphas_normal, dim=-1)
            s0, s1 = s1, cell(s0, s1, weights)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

    def _loss(self, input, target):
        logits = self(input)
        return self._criterion(logits, target)

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(PRIMITIVES)
        self.alphas_normal = Variable(0.001 * torch.randn(k, num_ops),
            requires_grad=True)
        self.alphas_reduce = Variable(0.001 * torch.randn(k, num_ops),
            requires_grad=True)
        self._arch_parameters = [self.alphas_normal, self.alphas_reduce]

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):

        def _parse(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for
                    k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2
                    ]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVES.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene
        gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu
            ().numpy())
        gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu
            ().numpy())
        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype(normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat)
        return genotype


class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(nn.ReLU(inplace=False), nn.Conv2d(C_in,
            C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine))

    def forward(self, x):
        return self.op(x)


class DilConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation,
        affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(nn.ReLU(inplace=False), nn.Conv2d(C_in,
            C_in, kernel_size=kernel_size, stride=stride, padding=padding,
            dilation=dilation, groups=C_in, bias=False), nn.Conv2d(C_in,
            C_out, kernel_size=1, padding=0, bias=False), nn.BatchNorm2d(
            C_out, affine=affine))

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(nn.ReLU(inplace=False), nn.Conv2d(C_in,
            C_in, kernel_size=kernel_size, stride=stride, padding=padding,
            groups=C_in, bias=False), nn.Conv2d(C_in, C_in, kernel_size=1,
            padding=0, bias=False), nn.BatchNorm2d(C_in, affine=affine), nn
            .ReLU(inplace=False), nn.Conv2d(C_in, C_in, kernel_size=
            kernel_size, stride=1, padding=padding, groups=C_in, bias=False
            ), nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine))

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.0)
        return x[:, :, ::self.stride, ::self.stride].mul(0.0)


class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0,
            bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0,
            bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze
            (1), 1)
        targets = (1 - self.epsilon
            ) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


STEPS = 8


INITRANGE = 0.04


def mask2d(B, D, keep_prob, cuda=True):
    m = torch.floor(torch.rand(B, D) + keep_prob) / keep_prob
    m = Variable(m, requires_grad=False)
    if cuda:
        m = m.cuda()
    return m


class DARTSCell(nn.Module):

    def __init__(self, ninp, nhid, dropouth, dropoutx, genotype):
        super(DARTSCell, self).__init__()
        self.nhid = nhid
        self.dropouth = dropouth
        self.dropoutx = dropoutx
        self.genotype = genotype
        steps = len(self.genotype.recurrent
            ) if self.genotype is not None else STEPS
        self._W0 = nn.Parameter(torch.Tensor(ninp + nhid, 2 * nhid).
            uniform_(-INITRANGE, INITRANGE))
        self._Ws = nn.ParameterList([nn.Parameter(torch.Tensor(nhid, 2 *
            nhid).uniform_(-INITRANGE, INITRANGE)) for i in range(steps)])

    def forward(self, inputs, hidden):
        T, B = inputs.size(0), inputs.size(1)
        if self.training:
            x_mask = mask2d(B, inputs.size(2), keep_prob=1.0 - self.dropoutx)
            h_mask = mask2d(B, hidden.size(2), keep_prob=1.0 - self.dropouth)
        else:
            x_mask = h_mask = None
        hidden = hidden[0]
        hiddens = []
        for t in range(T):
            hidden = self.cell(inputs[t], hidden, x_mask, h_mask)
            hiddens.append(hidden)
        hiddens = torch.stack(hiddens)
        return hiddens, hiddens[-1].unsqueeze(0)

    def _compute_init_state(self, x, h_prev, x_mask, h_mask):
        if self.training:
            xh_prev = torch.cat([x * x_mask, h_prev * h_mask], dim=-1)
        else:
            xh_prev = torch.cat([x, h_prev], dim=-1)
        c0, h0 = torch.split(xh_prev.mm(self._W0), self.nhid, dim=-1)
        c0 = c0.sigmoid()
        h0 = h0.tanh()
        s0 = h_prev + c0 * (h0 - h_prev)
        return s0

    def _get_activation(self, name):
        if name == 'tanh':
            f = F.tanh
        elif name == 'relu':
            f = F.relu
        elif name == 'sigmoid':
            f = F.sigmoid
        elif name == 'identity':
            f = lambda x: x
        else:
            raise NotImplementedError
        return f

    def cell(self, x, h_prev, x_mask, h_mask):
        s0 = self._compute_init_state(x, h_prev, x_mask, h_mask)
        states = [s0]
        for i, (name, pred) in enumerate(self.genotype.recurrent):
            s_prev = states[pred]
            if self.training:
                ch = (s_prev * h_mask).mm(self._Ws[i])
            else:
                ch = s_prev.mm(self._Ws[i])
            c, h = torch.split(ch, self.nhid, dim=-1)
            c = c.sigmoid()
            fn = self._get_activation(name)
            h = fn(h)
            s = s_prev + c * (h - s_prev)
            states += [s]
        output = torch.mean(torch.stack([states[i] for i in self.genotype.
            concat], -1), -1)
        return output


def embedded_dropout(embed, words, dropout=0.1, scale=None):
    if dropout:
        mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)
            ).bernoulli_(1 - dropout).expand_as(embed.weight) / (1 - dropout)
        mask = Variable(mask)
        masked_embed_weight = mask * embed.weight
    else:
        masked_embed_weight = embed.weight
    if scale:
        masked_embed_weight = scale.expand_as(masked_embed_weight
            ) * masked_embed_weight
    padding_idx = embed.padding_idx
    if padding_idx is None:
        padding_idx = -1
    X = embed._backend.Embedding.apply(words, masked_embed_weight,
        padding_idx, embed.max_norm, embed.norm_type, embed.
        scale_grad_by_freq, embed.sparse)
    return X


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp, nhid, nhidlast, dropout=0.5, dropouth=
        0.5, dropoutx=0.5, dropouti=0.5, dropoute=0.1, cell_cls=DARTSCell,
        genotype=None):
        super(RNNModel, self).__init__()
        self.lockdrop = LockedDropout()
        self.encoder = nn.Embedding(ntoken, ninp)
        assert ninp == nhid == nhidlast
        if cell_cls == DARTSCell:
            assert genotype is not None
            self.rnns = [cell_cls(ninp, nhid, dropouth, dropoutx, genotype)]
        else:
            assert genotype is None
            self.rnns = [cell_cls(ninp, nhid, dropouth, dropoutx)]
        self.rnns = torch.nn.ModuleList(self.rnns)
        self.decoder = nn.Linear(ninp, ntoken)
        self.decoder.weight = self.encoder.weight
        self.init_weights()
        self.ninp = ninp
        self.nhid = nhid
        self.nhidlast = nhidlast
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropoute = dropoute
        self.ntoken = ntoken
        self.cell_cls = cell_cls

    def init_weights(self):
        self.encoder.weight.data.uniform_(-INITRANGE, INITRANGE)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-INITRANGE, INITRANGE)

    def forward(self, input, hidden, return_h=False):
        batch_size = input.size(1)
        emb = embedded_dropout(self.encoder, input, dropout=self.dropoute if
            self.training else 0)
        emb = self.lockdrop(emb, self.dropouti)
        raw_output = emb
        new_hidden = []
        raw_outputs = []
        outputs = []
        for l, rnn in enumerate(self.rnns):
            current_input = raw_output
            raw_output, new_h = rnn(raw_output, hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
        hidden = new_hidden
        output = self.lockdrop(raw_output, self.dropout)
        outputs.append(output)
        logit = self.decoder(output.view(-1, self.ninp))
        log_prob = nn.functional.log_softmax(logit, dim=-1)
        model_output = log_prob
        model_output = model_output.view(-1, batch_size, self.ntoken)
        if return_h:
            return model_output, hidden, raw_outputs, outputs
        return model_output, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return [Variable(weight.new(1, bsz, self.nhid).zero_())]


class LockedDropout(nn.Module):

    def __init__(self):
        super(LockedDropout, self).__init__()

    def forward(self, x, dropout=0.5):
        if not self.training or not dropout:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m.div_(1 - dropout), requires_grad=False)
        mask = mask.expand_as(x)
        return mask * x


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_quark0_darts(_paritybench_base):
    pass
    @_fails_compile()

    def test_000(self):
        self._check(MixedOp(*[], **{'C': 4, 'stride': 1}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})
    @_fails_compile()

    def test_001(self):
        self._check(Network(*[], **{'C': 4, 'num_classes': 4, 'layers': 1, 'criterion': 4}), [torch.rand([4, 3, 64, 64])], {})

    def test_002(self):
        self._check(ReLUConvBN(*[], **{'C_in': 4, 'C_out': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(DilConv(*[], **{'C_in': 4, 'C_out': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4, 'dilation': 1}), [torch.rand([4, 4, 4, 4])], {})

    def test_004(self):
        self._check(SepConv(*[], **{'C_in': 4, 'C_out': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_005(self):
        self._check(Identity(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_006(self):
        self._check(Zero(*[], **{'stride': 1}), [torch.rand([4, 4, 4, 4])], {})

    def test_007(self):
        self._check(FactorizedReduce(*[], **{'C_in': 4, 'C_out': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_008(self):
        self._check(CrossEntropyLabelSmooth(*[], **{'num_classes': 4, 'epsilon': 4}), [torch.rand([4, 4]), torch.zeros([4], dtype=torch.int64)], {})
    @_fails_compile()

    def test_009(self):
        self._check(LockedDropout(*[], **{}), [torch.rand([4, 4, 4, 4])], {})
