import sys
_module = sys.modules[__name__]
del sys
data = _module
prep_enwik8 = _module
embed_regularize = _module
finetune = _module
generate = _module
locked_dropout = _module
main = _module
model = _module
pointer = _module
splitcross = _module
utils = _module
weight_drop = _module

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


import torch


import time


import math


import torch.nn as nn


from torch.autograd import Variable


from collections import defaultdict


from torch.nn import Parameter


from functools import wraps


class LockedDropout(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, dropout=0.5):
        if not self.training or not dropout:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m, requires_grad=False) / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x


def embedded_dropout(embed, words, dropout=0.1, scale=None):
    if dropout:
        mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)
            ).bernoulli_(1 - dropout).expand_as(embed.weight) / (1 - dropout)
        masked_embed_weight = mask * embed.weight
    else:
        masked_embed_weight = embed.weight
    if scale:
        masked_embed_weight = scale.expand_as(masked_embed_weight
            ) * masked_embed_weight
    padding_idx = embed.padding_idx
    if padding_idx is None:
        padding_idx = -1
    X = torch.nn.functional.embedding(words, masked_embed_weight,
        padding_idx, embed.max_norm, embed.norm_type, embed.
        scale_grad_by_freq, embed.sparse)
    return X


class SplitCrossEntropyLoss(nn.Module):
    """SplitCrossEntropyLoss calculates an approximate softmax"""

    def __init__(self, hidden_size, splits, verbose=False):
        super(SplitCrossEntropyLoss, self).__init__()
        self.hidden_size = hidden_size
        self.splits = [0] + splits + [100 * 1000000]
        self.nsplits = len(self.splits) - 1
        self.stats = defaultdict(list)
        self.verbose = verbose
        if self.nsplits > 1:
            self.tail_vectors = nn.Parameter(torch.zeros(self.nsplits - 1,
                hidden_size))
            self.tail_bias = nn.Parameter(torch.zeros(self.nsplits - 1))

    def logprob(self, weight, bias, hiddens, splits=None,
        softmaxed_head_res=None, verbose=False):
        if softmaxed_head_res is None:
            start, end = self.splits[0], self.splits[1]
            head_weight = None if end - start == 0 else weight[start:end]
            head_bias = None if end - start == 0 else bias[start:end]
            if self.nsplits > 1:
                head_weight = (self.tail_vectors if head_weight is None else
                    torch.cat([head_weight, self.tail_vectors]))
                head_bias = self.tail_bias if head_bias is None else torch.cat(
                    [head_bias, self.tail_bias])
            head_res = torch.nn.functional.linear(hiddens, head_weight,
                bias=head_bias)
            softmaxed_head_res = torch.nn.functional.log_softmax(head_res,
                dim=-1)
        if splits is None:
            splits = list(range(self.nsplits))
        results = []
        running_offset = 0
        for idx in splits:
            if idx == 0:
                results.append(softmaxed_head_res[:, :-(self.nsplits - 1)])
            else:
                start, end = self.splits[idx], self.splits[idx + 1]
                tail_weight = weight[start:end]
                tail_bias = bias[start:end]
                tail_res = torch.nn.functional.linear(hiddens, tail_weight,
                    bias=tail_bias)
                head_entropy = softmaxed_head_res[:, (-idx)].contiguous()
                tail_entropy = torch.nn.functional.log_softmax(tail_res, dim=-1
                    )
                results.append(head_entropy.view(-1, 1) + tail_entropy)
        if len(results) > 1:
            return torch.cat(results, dim=1)
        return results[0]

    def split_on_targets(self, hiddens, targets):
        split_targets = []
        split_hiddens = []
        mask = None
        for idx in range(1, self.nsplits):
            partial_mask = targets >= self.splits[idx]
            mask = mask + partial_mask if mask is not None else partial_mask
        for idx in range(self.nsplits):
            if self.nsplits == 1:
                split_targets, split_hiddens = [targets], [hiddens]
                continue
            if sum(len(t) for t in split_targets) == len(targets):
                split_targets.append([])
                split_hiddens.append([])
                continue
            tmp_mask = mask == idx
            split_targets.append(torch.masked_select(targets, tmp_mask))
            split_hiddens.append(hiddens.masked_select(tmp_mask.unsqueeze(1
                ).expand_as(hiddens)).view(-1, hiddens.size(1)))
        return split_targets, split_hiddens

    def forward(self, weight, bias, hiddens, targets, verbose=False):
        if self.verbose or verbose:
            for idx in sorted(self.stats):
                None
            None
        total_loss = None
        if len(hiddens.size()) > 2:
            hiddens = hiddens.view(-1, hiddens.size(2))
        split_targets, split_hiddens = self.split_on_targets(hiddens, targets)
        start, end = self.splits[0], self.splits[1]
        head_weight = None if end - start == 0 else weight[start:end]
        head_bias = None if end - start == 0 else bias[start:end]
        if self.nsplits > 1:
            head_weight = (self.tail_vectors if head_weight is None else
                torch.cat([head_weight, self.tail_vectors]))
            head_bias = self.tail_bias if head_bias is None else torch.cat([
                head_bias, self.tail_bias])
        combo = torch.cat([split_hiddens[i] for i in range(self.nsplits) if
            len(split_hiddens[i])])
        all_head_res = torch.nn.functional.linear(combo, head_weight, bias=
            head_bias)
        softmaxed_all_head_res = torch.nn.functional.log_softmax(all_head_res,
            dim=-1)
        if self.verbose or verbose:
            self.stats[0].append(combo.size()[0] * head_weight.size()[0])
        running_offset = 0
        for idx in range(self.nsplits):
            if len(split_targets[idx]) == 0:
                continue
            if idx == 0:
                softmaxed_head_res = softmaxed_all_head_res[running_offset:
                    running_offset + len(split_hiddens[idx])]
                entropy = -torch.gather(softmaxed_head_res, dim=1, index=
                    split_targets[idx].view(-1, 1))
            else:
                softmaxed_head_res = softmaxed_all_head_res[running_offset:
                    running_offset + len(split_hiddens[idx])]
                if self.verbose or verbose:
                    start, end = self.splits[idx], self.splits[idx + 1]
                    tail_weight = weight[start:end]
                    self.stats[idx].append(split_hiddens[idx].size()[0] *
                        tail_weight.size()[0])
                tail_res = self.logprob(weight, bias, split_hiddens[idx],
                    splits=[idx], softmaxed_head_res=softmaxed_head_res)
                head_entropy = softmaxed_head_res[:, (-idx)]
                indices = (split_targets[idx] - self.splits[idx]).view(-1, 1)
                tail_entropy = torch.gather(torch.nn.functional.log_softmax
                    (tail_res, dim=-1), dim=1, index=indices).squeeze()
                entropy = -(head_entropy + tail_entropy)
            running_offset += len(split_hiddens[idx])
            total_loss = entropy.float().sum(
                ) if total_loss is None else total_loss + entropy.float().sum()
        return (total_loss / len(targets)).type_as(weight)


class WeightDrop(torch.nn.Module):

    def __init__(self, module, weights, dropout=0, variational=False):
        super(WeightDrop, self).__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        self.variational = variational
        self._setup()

    def widget_demagnetizer_y2k_edition(*args, **kwargs):
        return

    def _setup(self):
        if issubclass(type(self.module), torch.nn.RNNBase):
            self.module.flatten_parameters = (self.
                widget_demagnetizer_y2k_edition)
        for name_w in self.weights:
            None
            w = getattr(self.module, name_w)
            del self.module._parameters[name_w]
            self.module.register_parameter(name_w + '_raw', Parameter(w.data))

    def _setweights(self):
        for name_w in self.weights:
            raw_w = getattr(self.module, name_w + '_raw')
            w = None
            if self.variational:
                mask = torch.autograd.Variable(torch.ones(raw_w.size(0), 1))
                if raw_w.is_cuda:
                    mask = mask
                mask = torch.nn.functional.dropout(mask, p=self.dropout,
                    training=True)
                w = mask.expand_as(raw_w) * raw_w
            else:
                w = torch.nn.functional.dropout(raw_w, p=self.dropout,
                    training=self.training)
            setattr(self.module, name_w, w)

    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_salesforce_awd_lstm_lm(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(LockedDropout(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

