import sys
_module = sys.modules[__name__]
del sys
master = _module
callback = _module
adversarial = _module
lr_finder = _module
lr_scheduler = _module
modelcheckpoint = _module
optimizater = _module
adabound = _module
adafactor = _module
adamw = _module
lamb = _module
lars = _module
lookahead = _module
nadam = _module
novograd = _module
planradam = _module
radam = _module
ralamb = _module
ralars = _module
sgdw = _module
progressbar = _module
trainingmonitor = _module
cluener = _module
losses = _module
dice_loss = _module
focal_loss = _module
label_smoothing = _module
metrics = _module
ner_metrics = _module
models = _module
albert_for_ner = _module
bert_for_ner = _module
layers = _module
crf = _module
linears = _module
transformers = _module
__main__ = _module
configuration_albert = _module
configuration_auto = _module
configuration_bert = _module
configuration_ctrl = _module
configuration_distilbert = _module
configuration_gpt2 = _module
configuration_openai = _module
configuration_roberta = _module
configuration_transfo_xl = _module
configuration_utils = _module
configuration_xlm = _module
configuration_xlnet = _module
file_utils = _module
modeling_albert = _module
modeling_albert_bright = _module
modeling_auto = _module
modeling_bert = _module
modeling_ctrl = _module
modeling_distilbert = _module
modeling_gpt2 = _module
modeling_openai = _module
modeling_roberta = _module
modeling_transfo_xl = _module
modeling_transfo_xl_utilities = _module
modeling_utils = _module
modeling_xlm = _module
modeling_xlnet = _module
tokenization_albert = _module
tokenization_auto = _module
tokenization_bert = _module
tokenization_ctrl = _module
tokenization_distilbert = _module
tokenization_gpt2 = _module
tokenization_openai = _module
tokenization_roberta = _module
tokenization_transfo_xl = _module
tokenization_utils = _module
tokenization_xlm = _module
tokenization_xlnet = _module
bert = _module
base = _module
processors = _module
ner_seq = _module
ner_span = _module
utils_ner = _module
run_ner_crf = _module
run_ner_softmax = _module
run_ner_span = _module
tools = _module
common = _module
convert_albert_tf_checkpoint_to_pytorch = _module
download_clue_data = _module
finetuning_argparse = _module
plot = _module

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


import copy


import torch


from torch.optim.lr_scheduler import _LRScheduler


from torch import nn


import torch.nn as nn


import torch.nn.functional as F


from torch.nn import CrossEntropyLoss


from typing import List


from typing import Optional


import logging


import math


from torch.nn import MSELoss


import collections


import numpy as np


from torch.nn.parameter import Parameter


import itertools


from collections import defaultdict


from torch.nn import functional as F


import time


from torch.utils.data import DataLoader


from torch.utils.data import RandomSampler


from torch.utils.data import SequentialSampler


from torch.utils.data import TensorDataset


from torch.utils.data.distributed import DistributedSampler


import random


from collections import OrderedDict


class DiceLoss(nn.Module):
    """DiceLoss implemented from 'Dice Loss for Data-imbalanced NLP Tasks'
    Useful in dealing with unbalanced data
    """

    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        prob = torch.softmax(input, dim=1)
        prob = torch.gather(prob, dim=1, index=target.unsqueeze(1))
        dsc_i = 1 - (1 - prob) * prob / ((1 - prob) * prob + 1)
        dice_loss = dsc_i.mean()
        return dice_loss


class FocalLoss(nn.Module):
    """Multi-class Focal loss implementation"""

    def __init__(self, gamma=2, weight=None, ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1 - pt) ** self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.weight, ignore_index=self.
            ignore_index)
        return loss


class LabelSmoothingCrossEntropy(nn.Module):

    def __init__(self, eps=0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction == 'sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction == 'mean':
                loss = loss.mean()
        return loss * self.eps / c + (1 - self.eps) * F.nll_loss(log_preds,
            target, reduction=self.reduction, ignore_index=self.ignore_index)


class CRF(nn.Module):
    """Conditional random field.
    This module implements a conditional random field [LMP01]_. The forward computation
    of this class computes the log likelihood of the given sequence of tags and
    emission score tensor. This class also has `~CRF.decode` method which finds
    the best tag sequence given an emission score tensor using `Viterbi algorithm`_.
    Args:
        num_tags: Number of tags.
        batch_first: Whether the first dimension corresponds to the size of a minibatch.
    Attributes:
        start_transitions (`~torch.nn.Parameter`): Start transition score tensor of size
            ``(num_tags,)``.
        end_transitions (`~torch.nn.Parameter`): End transition score tensor of size
            ``(num_tags,)``.
        transitions (`~torch.nn.Parameter`): Transition score tensor of size
            ``(num_tags, num_tags)``.
    .. [LMP01] Lafferty, J., McCallum, A., Pereira, F. (2001).
       "Conditional random fields: Probabilistic models for segmenting and
       labeling sequence data". *Proc. 18th International Conf. on Machine
       Learning*. Morgan Kaufmann. pp. 282–289.
    .. _Viterbi algorithm: https://en.wikipedia.org/wiki/Viterbi_algorithm
    """

    def __init__(self, num_tags: int, batch_first: bool=False) ->None:
        if num_tags <= 0:
            raise ValueError(f'invalid number of tags: {num_tags}')
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.reset_parameters()

    def reset_parameters(self) ->None:
        """Initialize the transition parameters.
        The parameters will be initialized randomly from a uniform distribution
        between -0.1 and 0.1.
        """
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def __repr__(self) ->str:
        return f'{self.__class__.__name__}(num_tags={self.num_tags})'

    def forward(self, emissions: torch.Tensor, tags: torch.LongTensor, mask:
        Optional[torch.ByteTensor]=None, reduction: str='mean') ->torch.Tensor:
        """Compute the conditional log likelihood of a sequence of tags given emission scores.
        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            tags (`~torch.LongTensor`): Sequence of tags tensor of size
                ``(seq_length, batch_size)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.
            reduction: Specifies  the reduction to apply to the output:
                ``none|sum|mean|token_mean``. ``none``: no reduction will be applied.
                ``sum``: the output will be summed over batches. ``mean``: the output will be
                averaged over batches. ``token_mean``: the output will be averaged over tokens.
        Returns:
            `~torch.Tensor`: The log likelihood. This will have size ``(batch_size,)`` if
            reduction is ``none``, ``()`` otherwise.
        """
        if reduction not in ('none', 'sum', 'mean', 'token_mean'):
            raise ValueError(f'invalid reduction: {reduction}')
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8, device=tags.device)
        if mask.dtype != torch.uint8:
            mask = mask.byte()
        self._validate(emissions, tags=tags, mask=mask)
        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)
        numerator = self._compute_score(emissions, tags, mask)
        denominator = self._compute_normalizer(emissions, mask)
        llh = numerator - denominator
        if reduction == 'none':
            return llh
        if reduction == 'sum':
            return llh.sum()
        if reduction == 'mean':
            return llh.mean()
        return llh.sum() / mask.float().sum()

    def decode(self, emissions: torch.Tensor, mask: Optional[torch.
        ByteTensor]=None, nbest: Optional[int]=None, pad_tag: Optional[int]
        =None) ->List[List[List[int]]]:
        """Find the most likely tag sequence using Viterbi algorithm.
        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.
            nbest (`int`): Number of most probable paths for each sequence
            pad_tag (`int`): Tag at padded positions. Often input varies in length and
                the length will be padded to the maximum length in the batch. Tags at
                the padded positions will be assigned with a padding tag, i.e. `pad_tag`
        Returns:
            A PyTorch tensor of the best tag sequence for each batch of shape
            (nbest, batch_size, seq_length)
        """
        if nbest is None:
            nbest = 1
        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.uint8,
                device=emissions.device)
        if mask.dtype != torch.uint8:
            mask = mask.byte()
        self._validate(emissions, mask=mask)
        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)
        if nbest == 1:
            return self._viterbi_decode(emissions, mask, pad_tag).unsqueeze(0)
        return self._viterbi_decode_nbest(emissions, mask, nbest, pad_tag)

    def _validate(self, emissions: torch.Tensor, tags: Optional[torch.
        LongTensor]=None, mask: Optional[torch.ByteTensor]=None) ->None:
        if emissions.dim() != 3:
            raise ValueError(
                f'emissions must have dimension of 3, got {emissions.dim()}')
        if emissions.size(2) != self.num_tags:
            raise ValueError(
                f'expected last dimension of emissions is {self.num_tags}, got {emissions.size(2)}'
                )
        if tags is not None:
            if emissions.shape[:2] != tags.shape:
                raise ValueError(
                    f'the first two dimensions of emissions and tags must match, got {tuple(emissions.shape[:2])} and {tuple(tags.shape)}'
                    )
        if mask is not None:
            if emissions.shape[:2] != mask.shape:
                raise ValueError(
                    f'the first two dimensions of emissions and mask must match, got {tuple(emissions.shape[:2])} and {tuple(mask.shape)}'
                    )
            no_empty_seq = not self.batch_first and mask[0].all()
            no_empty_seq_bf = self.batch_first and mask[:, (0)].all()
            if not no_empty_seq and not no_empty_seq_bf:
                raise ValueError('mask of the first timestep must all be on')

    def _compute_score(self, emissions: torch.Tensor, tags: torch.
        LongTensor, mask: torch.ByteTensor) ->torch.Tensor:
        seq_length, batch_size = tags.shape
        mask = mask.float()
        score = self.start_transitions[tags[0]]
        score += emissions[0, torch.arange(batch_size), tags[0]]
        for i in range(1, seq_length):
            score += self.transitions[tags[i - 1], tags[i]] * mask[i]
            score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i]
        seq_ends = mask.long().sum(dim=0) - 1
        last_tags = tags[seq_ends, torch.arange(batch_size)]
        score += self.end_transitions[last_tags]
        return score

    def _compute_normalizer(self, emissions: torch.Tensor, mask: torch.
        ByteTensor) ->torch.Tensor:
        seq_length = emissions.size(0)
        score = self.start_transitions + emissions[0]
        for i in range(1, seq_length):
            broadcast_score = score.unsqueeze(2)
            broadcast_emissions = emissions[i].unsqueeze(1)
            next_score = (broadcast_score + self.transitions +
                broadcast_emissions)
            next_score = torch.logsumexp(next_score, dim=1)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)
        score += self.end_transitions
        return torch.logsumexp(score, dim=1)

    def _viterbi_decode(self, emissions: torch.FloatTensor, mask: torch.
        ByteTensor, pad_tag: Optional[int]=None) ->List[List[int]]:
        if pad_tag is None:
            pad_tag = 0
        device = emissions.device
        seq_length, batch_size = mask.shape
        score = self.start_transitions + emissions[0]
        history_idx = torch.zeros((seq_length, batch_size, self.num_tags),
            dtype=torch.long, device=device)
        oor_idx = torch.zeros((batch_size, self.num_tags), dtype=torch.long,
            device=device)
        oor_tag = torch.full((seq_length, batch_size), pad_tag, dtype=torch
            .long, device=device)
        for i in range(1, seq_length):
            broadcast_score = score.unsqueeze(2)
            broadcast_emission = emissions[i].unsqueeze(1)
            next_score = (broadcast_score + self.transitions +
                broadcast_emission)
            next_score, indices = next_score.max(dim=1)
            score = torch.where(mask[i].unsqueeze(-1), next_score, score)
            indices = torch.where(mask[i].unsqueeze(-1), indices, oor_idx)
            history_idx[i - 1] = indices
        end_score = score + self.end_transitions
        _, end_tag = end_score.max(dim=1)
        seq_ends = mask.long().sum(dim=0) - 1
        history_idx = history_idx.transpose(1, 0).contiguous()
        history_idx.scatter_(1, seq_ends.view(-1, 1, 1).expand(-1, 1, self.
            num_tags), end_tag.view(-1, 1, 1).expand(-1, 1, self.num_tags))
        history_idx = history_idx.transpose(1, 0).contiguous()
        best_tags_arr = torch.zeros((seq_length, batch_size), dtype=torch.
            long, device=device)
        best_tags = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        for idx in range(seq_length - 1, -1, -1):
            best_tags = torch.gather(history_idx[idx], 1, best_tags)
            best_tags_arr[idx] = best_tags.data.view(batch_size)
        return torch.where(mask, best_tags_arr, oor_tag).transpose(0, 1)

    def _viterbi_decode_nbest(self, emissions: torch.FloatTensor, mask:
        torch.ByteTensor, nbest: int, pad_tag: Optional[int]=None) ->List[
        List[List[int]]]:
        if pad_tag is None:
            pad_tag = 0
        device = emissions.device
        seq_length, batch_size = mask.shape
        score = self.start_transitions + emissions[0]
        history_idx = torch.zeros((seq_length, batch_size, self.num_tags,
            nbest), dtype=torch.long, device=device)
        oor_idx = torch.zeros((batch_size, self.num_tags, nbest), dtype=
            torch.long, device=device)
        oor_tag = torch.full((seq_length, batch_size, nbest), pad_tag,
            dtype=torch.long, device=device)
        for i in range(1, seq_length):
            if i == 1:
                broadcast_score = score.unsqueeze(-1)
                broadcast_emission = emissions[i].unsqueeze(1)
                next_score = (broadcast_score + self.transitions +
                    broadcast_emission)
            else:
                broadcast_score = score.unsqueeze(-1)
                broadcast_emission = emissions[i].unsqueeze(1).unsqueeze(2)
                next_score = broadcast_score + self.transitions.unsqueeze(1
                    ) + broadcast_emission
            next_score, indices = next_score.view(batch_size, -1, self.num_tags
                ).topk(nbest, dim=1)
            if i == 1:
                score = score.unsqueeze(-1).expand(-1, -1, nbest)
                indices = indices * nbest
            next_score = next_score.transpose(2, 1)
            indices = indices.transpose(2, 1)
            score = torch.where(mask[i].unsqueeze(-1).unsqueeze(-1),
                next_score, score)
            indices = torch.where(mask[i].unsqueeze(-1).unsqueeze(-1),
                indices, oor_idx)
            history_idx[i - 1] = indices
        end_score = score + self.end_transitions.unsqueeze(-1)
        _, end_tag = end_score.view(batch_size, -1).topk(nbest, dim=1)
        seq_ends = mask.long().sum(dim=0) - 1
        history_idx = history_idx.transpose(1, 0).contiguous()
        history_idx.scatter_(1, seq_ends.view(-1, 1, 1, 1).expand(-1, 1,
            self.num_tags, nbest), end_tag.view(-1, 1, 1, nbest).expand(-1,
            1, self.num_tags, nbest))
        history_idx = history_idx.transpose(1, 0).contiguous()
        best_tags_arr = torch.zeros((seq_length, batch_size, nbest), dtype=
            torch.long, device=device)
        best_tags = torch.arange(nbest, dtype=torch.long, device=device).view(
            1, -1).expand(batch_size, -1)
        for idx in range(seq_length - 1, -1, -1):
            best_tags = torch.gather(history_idx[idx].view(batch_size, -1),
                1, best_tags)
            best_tags_arr[idx] = best_tags.data.view(batch_size, -1) // nbest
        return torch.where(mask.unsqueeze(-1), best_tags_arr, oor_tag).permute(
            2, 1, 0)


class FeedForwardNetwork(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0):
        super(FeedForwardNetwork, self).__init__()
        self.dropout_rate = dropout_rate
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x_proj = F.dropout(F.relu(self.linear1(x)), p=self.dropout_rate,
            training=self.training)
        x_proj = self.linear2(x_proj)
        return x_proj


class PoolerStartLogits(nn.Module):

    def __init__(self, hidden_size, num_classes):
        super(PoolerStartLogits, self).__init__()
        self.dense = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden_states, p_mask=None):
        x = self.dense(hidden_states)
        return x


class PoolerEndLogits(nn.Module):

    def __init__(self, hidden_size, num_classes):
        super(PoolerEndLogits, self).__init__()
        self.dense_0 = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dense_1 = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden_states, start_positions=None, p_mask=None):
        x = self.dense_0(torch.cat([hidden_states, start_positions], dim=-1))
        x = self.activation(x)
        x = self.LayerNorm(x)
        x = self.dense_1(x)
        return x


AlbertLayerNorm = torch.nn.LayerNorm


class AlbertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super(AlbertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.
            embedding_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.
            max_position_embeddings, config.embedding_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size,
            config.embedding_size)
        self.LayerNorm = AlbertLayerNorm(config.embedding_size, eps=config.
            layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long,
                device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = (words_embeddings + position_embeddings +
            token_type_embeddings)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class AlbertSelfAttention(nn.Module):

    def __init__(self, config):
        super(AlbertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                'The hidden size (%d) is not a multiple of the number of attention heads (%d)'
                 % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.
            num_attention_heads)
        self.all_head_size = (self.num_attention_heads * self.
            attention_head_size)
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.
            attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1,
            -2))
        attention_scores = attention_scores / math.sqrt(self.
            attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.
            all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        outputs = (context_layer, attention_probs
            ) if self.output_attentions else (context_layer,)
        return outputs


class AlbertSelfOutput(nn.Module):

    def __init__(self, config):
        super(AlbertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


def prune_linear_layer(layer, index, dim=0):
    """ Prune a linear layer (a model parameters) to keep only entries in index.
        Return the pruned layer as a new layer with requires_grad=True.
        Used to remove heads.
    """
    index = index.to(layer.weight.device)
    W = layer.weight.index_select(dim, index).clone().detach()
    if layer.bias is not None:
        if dim == 1:
            b = layer.bias.clone().detach()
        else:
            b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None
        ).to(layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b.contiguous())
        new_layer.bias.requires_grad = True
    return new_layer


class AlbertAttention(nn.Module):

    def __init__(self, config):
        super(AlbertAttention, self).__init__()
        self.self = AlbertSelfAttention(config)
        self.output = AlbertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads, self.self.
            attention_head_size)
        heads = set(heads) - self.pruned_heads
        for head in heads:
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)
        self.self.num_attention_heads = self.self.num_attention_heads - len(
            heads)
        self.self.all_head_size = (self.self.attention_head_size * self.
            self.num_attention_heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, input_tensor, attention_mask=None, head_mask=None):
        self_outputs = self.self(input_tensor, attention_mask, head_mask)
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = attention_output, self_outputs
        return outputs


class AlbertOutput(nn.Module):

    def __init__(self, config):
        super(AlbertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


def gelu(x):
    """ Implementation of the gelu activation function.
        XLNet is using OpenAI GPT's gelu (not exactly the same as BERT)
        Also see https://arxiv.org/abs/1606.08415
    """
    cdf = 0.5 * (1.0 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 *
        torch.pow(x, 3))))
    return x * cdf


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {'gelu': gelu, 'relu': torch.nn.functional.relu, 'swish': swish}


class AlbertIntermediate(nn.Module):

    def __init__(self, config):
        super(AlbertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = AlbertOutput(config)
        if isinstance(config.hidden_act, str) or sys.version_info[0
            ] == 2 and isinstance(config.hidden_act, unicode):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        intermediate_output = self.dense(hidden_states)
        intermediate_output = self.intermediate_act_fn(intermediate_output)
        output = self.output(intermediate_output)
        return output


class AlbertFFN(nn.Module):

    def __init__(self, config):
        super(AlbertFFN, self).__init__()
        self.intermediate = AlbertIntermediate(config)

    def forward(self, attention_output):
        output = self.intermediate(attention_output)
        return output


class AlbertLayer(nn.Module):

    def __init__(self, config):
        super(AlbertLayer, self).__init__()
        self.attention = AlbertAttention(config)
        self.ffn = AlbertFFN(config)
        self.LayerNorm = AlbertLayerNorm(config.hidden_size, eps=config.
            layer_norm_eps)
        self.LayerNorm_1 = AlbertLayerNorm(config.hidden_size, eps=config.
            layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        attention_outputs = self.attention(hidden_states, attention_mask,
            head_mask)
        attention_output = self.LayerNorm(attention_outputs[0] + hidden_states)
        ffn_output = self.ffn(attention_output)
        ffn_output = self.LayerNorm_1(ffn_output + attention_output)
        outputs = (ffn_output,) + attention_outputs[1:]
        return outputs


class AlbertGroup(nn.Module):

    def __init__(self, config):
        super(AlbertGroup, self).__init__()
        self.inner_group_num = config.inner_group_num
        self.inner_group = nn.ModuleList([AlbertLayer(config) for _ in
            range(config.inner_group_num)])

    def forward(self, hidden_states, attention_mask, head_mask):
        layer_attentions = ()
        layer_hidden_states = ()
        for inner_group_idx in range(self.inner_group_num):
            layer_module = self.inner_group[inner_group_idx]
            layer_outputs = layer_module(hidden_states, attention_mask,
                head_mask)
            hidden_states = layer_outputs[0]
            layer_attentions = layer_attentions + (layer_outputs[1],)
            layer_hidden_states = layer_hidden_states + (hidden_states,)
        return layer_hidden_states, layer_attentions


class AlbertTransformer(nn.Module):

    def __init__(self, config):
        super(AlbertTransformer, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.num_hidden_layers = config.num_hidden_layers
        self.num_hidden_groups = config.num_hidden_groups
        self.group = nn.ModuleList([AlbertGroup(config) for _ in range(
            config.num_hidden_groups)])

    def forward(self, hidden_states, attention_mask, head_mask):
        all_hidden_states = ()
        all_attentions = ()
        for layer_idx in range(self.num_hidden_layers):
            if self.output_hidden_states and layer_idx == 0:
                all_hidden_states = all_hidden_states + (hidden_states,)
            group_idx = int(layer_idx / self.num_hidden_layers * self.
                num_hidden_groups)
            layer_module = self.group[group_idx]
            layer_outputs = layer_module(hidden_states, attention_mask,
                head_mask[layer_idx])
            hidden_states = layer_outputs[0][-1]
            if self.output_attentions:
                all_attentions = all_attentions + layer_outputs[1]
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + layer_outputs[0]
        outputs = hidden_states,
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs


class AlbertEncoder(nn.Module):

    def __init__(self, config):
        super(AlbertEncoder, self).__init__()
        self.hidden_size = config.hidden_size
        self.embedding_size = config.embedding_size
        self.embedding_hidden_mapping_in = nn.Linear(self.embedding_size,
            self.hidden_size)
        self.transformer = AlbertTransformer(config)

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        if self.embedding_size != self.hidden_size:
            prev_output = self.embedding_hidden_mapping_in(hidden_states)
        else:
            prev_output = hidden_states
        outputs = self.transformer(prev_output, attention_mask, head_mask)
        return outputs


class AlbertPooler(nn.Module):

    def __init__(self, config):
        super(AlbertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, (0)]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class AlbertPredictionHeadTransform(nn.Module):

    def __init__(self, config):
        super(AlbertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.embedding_size)
        if isinstance(config.hidden_act, str) or sys.version_info[0
            ] == 2 and isinstance(config.hidden_act, unicode):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = AlbertLayerNorm(config.embedding_size, eps=config.
            layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class AlbertLMPredictionHead(nn.Module):

    def __init__(self, config):
        super(AlbertLMPredictionHead, self).__init__()
        self.transform = AlbertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.embedding_size, config.vocab_size,
            bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class AlbertOnlyMLMHead(nn.Module):

    def __init__(self, config):
        super(AlbertOnlyMLMHead, self).__init__()
        self.predictions = AlbertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class AlbertOnlyNSPHead(nn.Module):

    def __init__(self, config):
        super(AlbertOnlyNSPHead, self).__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class AlbertPreTrainingHeads(nn.Module):

    def __init__(self, config):
        super(AlbertPreTrainingHeads, self).__init__()
        self.predictions = AlbertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class AlbertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super(AlbertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.
            embedding_size, padding_idx=0)
        self.word_embeddings_2 = nn.Linear(config.embedding_size, config.
            hidden_size, bias=False)
        self.position_embeddings = nn.Embedding(config.
            max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size,
            config.hidden_size)
        self.LayerNorm = AlbertLayerNorm(config.hidden_size, eps=config.
            layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long,
                device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        words_embeddings = self.word_embeddings(input_ids)
        words_embeddings = self.word_embeddings_2(words_embeddings)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = (words_embeddings + position_embeddings +
            token_type_embeddings)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class AlbertSelfOutput(nn.Module):

    def __init__(self, config):
        super(AlbertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = AlbertLayerNorm(config.hidden_size, eps=config.
            layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class AlbertAttention(nn.Module):

    def __init__(self, config):
        super(AlbertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = AlbertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads, self.self.
            attention_head_size)
        heads = set(heads) - self.pruned_heads
        for head in heads:
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)
        self.self.num_attention_heads = self.self.num_attention_heads - len(
            heads)
        self.self.all_head_size = (self.self.attention_head_size * self.
            self.num_attention_heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, input_tensor, attention_mask=None, head_mask=None):
        self_outputs = self.self(input_tensor, attention_mask, head_mask)
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class AlbertOutput(nn.Module):

    def __init__(self, config):
        super(AlbertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = AlbertLayerNorm(config.hidden_size, eps=config.
            layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):

    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = AlbertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = AlbertOutput(config)

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        attention_outputs = self.attention(hidden_states, attention_mask,
            head_mask)
        attention_output = attention_outputs[0]
        attention_output_pre = attention_output
        intermediate_output = self.intermediate(attention_output_pre)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + attention_outputs[1:]
        return outputs


class AlbertEncoder(nn.Module):

    def __init__(self, config):
        super(AlbertEncoder, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.num_hidden_layers = config.num_hidden_layers
        self.layer_shared = BertLayer(config)

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        all_hidden_states = ()
        all_attentions = ()
        for i in range(self.num_hidden_layers):
            layer_module = self.layer_shared
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = layer_module(hidden_states, attention_mask,
                head_mask[i])
            hidden_states = layer_outputs[0]
            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        outputs = hidden_states,
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs


class AlbertLMPredictionHead(nn.Module):

    def __init__(self, config):
        super(AlbertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.project_layer = nn.Linear(config.hidden_size, config.
            embedding_size, bias=False)
        self.decoder = nn.Linear(config.embedding_size, config.vocab_size,
            bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.project_layer(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class AlbertOnlyMLMHead(nn.Module):

    def __init__(self, config):
        super(AlbertOnlyMLMHead, self).__init__()
        self.predictions = AlbertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class AlbertOnlyNSPHead(nn.Module):

    def __init__(self, config):
        super(AlbertOnlyNSPHead, self).__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class AlbertPreTrainingHeads(nn.Module):

    def __init__(self, config):
        super(AlbertPreTrainingHeads, self).__init__()
        self.predictions = AlbertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


BertLayerNorm = torch.nn.LayerNorm


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.
            hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.
            max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size,
            config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.
            layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long,
                device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = (words_embeddings + position_embeddings +
            token_type_embeddings)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):

    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                'The hidden size (%d) is not a multiple of the number of attention heads (%d)'
                 % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.
            num_attention_heads)
        self.all_head_size = (self.num_attention_heads * self.
            attention_head_size)
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.
            attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1,
            -2))
        attention_scores = attention_scores / math.sqrt(self.
            attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.
            all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        outputs = (context_layer, attention_probs
            ) if self.output_attentions else (context_layer,)
        return outputs


class BertSelfOutput(nn.Module):

    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.
            layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):

    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads, self.self.
            attention_head_size)
        heads = set(heads) - self.pruned_heads
        for head in heads:
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)
        self.self.num_attention_heads = self.self.num_attention_heads - len(
            heads)
        self.self.all_head_size = (self.self.attention_head_size * self.
            self.num_attention_heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, input_tensor, attention_mask=None, head_mask=None):
        self_outputs = self.self(input_tensor, attention_mask, head_mask)
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class BertIntermediate(nn.Module):

    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str) or sys.version_info[0
            ] == 2 and isinstance(config.hidden_act, unicode):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):

    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.
            layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):

    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        attention_outputs = self.attention(hidden_states, attention_mask,
            head_mask)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + attention_outputs[1:]
        return outputs


class BertEncoder(nn.Module):

    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config
            .num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = layer_module(hidden_states, attention_mask,
                head_mask[i])
            hidden_states = layer_outputs[0]
            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        outputs = hidden_states,
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs


class BertPooler(nn.Module):

    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, (0)]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):

    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str) or sys.version_info[0
            ] == 2 and isinstance(config.hidden_act, unicode):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.
            layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):

    def __init__(self, config):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size,
            bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertOnlyMLMHead(nn.Module):

    def __init__(self, config):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):

    def __init__(self, config):
        super(BertOnlyNSPHead, self).__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(nn.Module):

    def __init__(self, config):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


def scaled_dot_product_attention(q, k, v, mask, attention_mask=None,
    head_mask=None):
    matmul_qk = torch.matmul(q, k.permute(0, 1, 3, 2))
    dk = k.shape[-1]
    scaled_attention_logits = matmul_qk / np.sqrt(dk)
    if mask is not None:
        scaled_attention_logits += mask * -10000.0
    if attention_mask is not None:
        scaled_attention_logits = scaled_attention_logits + attention_mask
    attention_weights = torch.softmax(scaled_attention_logits, dim=-1)
    if head_mask is not None:
        attention_weights = attention_weights * head_mask
    output = torch.matmul(attention_weights, v)
    return output, attention_weights


class MultiHeadAttention(torch.nn.Module):

    def __init__(self, d_model_size, num_heads, output_attentions=False):
        super(MultiHeadAttention, self).__init__()
        self.output_attentions = output_attentions
        self.num_heads = num_heads
        self.d_model_size = d_model_size
        self.depth = int(d_model_size / self.num_heads)
        self.Wq = torch.nn.Linear(d_model_size, d_model_size)
        self.Wk = torch.nn.Linear(d_model_size, d_model_size)
        self.Wv = torch.nn.Linear(d_model_size, d_model_size)
        self.dense = torch.nn.Linear(d_model_size, d_model_size)

    def split_into_heads(self, x, batch_size):
        x = x.reshape(batch_size, -1, self.num_heads, self.depth)
        return x.permute([0, 2, 1, 3])

    def forward(self, v, k, q, mask, layer_past=None, attention_mask=None,
        head_mask=None):
        batch_size = q.shape[0]
        q = self.Wq(q)
        k = self.Wk(k)
        v = self.Wv(v)
        q = self.split_into_heads(q, batch_size)
        k = self.split_into_heads(k, batch_size)
        v = self.split_into_heads(v, batch_size)
        if layer_past is not None:
            past_key, past_value = layer_past[0], layer_past[1]
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)
        present = torch.stack((k, v))
        output = scaled_dot_product_attention(q, k, v, mask, attention_mask,
            head_mask)
        scaled_attention = output[0].permute([0, 2, 1, 3])
        attn = output[1]
        original_size_attention = scaled_attention.reshape(batch_size, -1,
            self.d_model_size)
        output = self.dense(original_size_attention)
        outputs = output, present
        if self.output_attentions:
            outputs = outputs + (attn,)
        return outputs


def point_wise_feed_forward_network(d_model_size, dff):
    return torch.nn.Sequential(torch.nn.Linear(d_model_size, dff), torch.nn
        .ReLU(), torch.nn.Linear(dff, d_model_size))


class EncoderLayer(torch.nn.Module):

    def __init__(self, d_model_size, num_heads, dff, rate=0.1,
        output_attentions=False):
        super(EncoderLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(d_model_size,
            num_heads, output_attentions)
        self.ffn = point_wise_feed_forward_network(d_model_size, dff)
        self.layernorm1 = torch.nn.LayerNorm(d_model_size, eps=1e-06)
        self.layernorm2 = torch.nn.LayerNorm(d_model_size, eps=1e-06)
        self.dropout1 = torch.nn.Dropout(rate)
        self.dropout2 = torch.nn.Dropout(rate)

    def forward(self, x, mask, layer_past=None, attention_mask=None,
        head_mask=None):
        normed = self.layernorm1(x)
        attn_outputs = self.multi_head_attention(normed, normed, normed,
            mask, layer_past=layer_past, attention_mask=attention_mask,
            head_mask=head_mask)
        attn_output = attn_outputs[0]
        attn_output = self.dropout1(attn_output)
        out1 = x + attn_output
        out2 = self.layernorm2(out1)
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout2(ffn_output)
        out2 = out1 + ffn_output
        outputs = (out2,) + attn_outputs[1:]
        return outputs


def create_sinusoidal_embeddings(n_pos, dim, out):
    position_enc = np.array([[(pos / np.power(10000, 2 * (j // 2) / dim)) for
        j in range(dim)] for pos in range(n_pos)])
    out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
    out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
    out.detach_()
    out.requires_grad = False


class Embeddings(nn.Module):

    def __init__(self, config):
        super(Embeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.dim,
            padding_idx=0)
        self.position_embeddings = nn.Embedding(config.
            max_position_embeddings, config.dim)
        if config.sinusoidal_pos_embds:
            create_sinusoidal_embeddings(n_pos=config.
                max_position_embeddings, dim=config.dim, out=self.
                position_embeddings.weight)
        self.LayerNorm = nn.LayerNorm(config.dim, eps=1e-12)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids):
        """
        Parameters
        ----------
        input_ids: torch.tensor(bs, max_seq_length)
            The token ids to embed.

        Outputs
        -------
        embeddings: torch.tensor(bs, max_seq_length, dim)
            The embedded tokens (plus position embeddings, no token_type embeddings)
        """
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=
            input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = word_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class MultiHeadSelfAttention(nn.Module):

    def __init__(self, config):
        super(MultiHeadSelfAttention, self).__init__()
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.dropout = nn.Dropout(p=config.attention_dropout)
        self.output_attentions = config.output_attentions
        assert self.dim % self.n_heads == 0
        self.q_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.k_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.v_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.out_lin = nn.Linear(in_features=config.dim, out_features=
            config.dim)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        attention_head_size = self.dim // self.n_heads
        if len(heads) == 0:
            return
        mask = torch.ones(self.n_heads, attention_head_size)
        heads = set(heads) - self.pruned_heads
        for head in heads:
            head -= sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        self.q_lin = prune_linear_layer(self.q_lin, index)
        self.k_lin = prune_linear_layer(self.k_lin, index)
        self.v_lin = prune_linear_layer(self.v_lin, index)
        self.out_lin = prune_linear_layer(self.out_lin, index, dim=1)
        self.n_heads = self.n_heads - len(heads)
        self.dim = attention_head_size * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, query, key, value, mask, head_mask=None):
        """
        Parameters
        ----------
        query: torch.tensor(bs, seq_length, dim)
        key: torch.tensor(bs, seq_length, dim)
        value: torch.tensor(bs, seq_length, dim)
        mask: torch.tensor(bs, seq_length)

        Outputs
        -------
        weights: torch.tensor(bs, n_heads, seq_length, seq_length)
            Attention weights
        context: torch.tensor(bs, seq_length, dim)
            Contextualized layer. Optional: only if `output_attentions=True`
        """
        bs, q_length, dim = query.size()
        k_length = key.size(1)
        dim_per_head = self.dim // self.n_heads
        mask_reshp = bs, 1, 1, k_length

        def shape(x):
            """ separate heads """
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x):
            """ group heads """
            return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads *
                dim_per_head)
        q = shape(self.q_lin(query))
        k = shape(self.k_lin(key))
        v = shape(self.v_lin(value))
        q = q / math.sqrt(dim_per_head)
        scores = torch.matmul(q, k.transpose(2, 3))
        mask = (mask == 0).view(mask_reshp).expand_as(scores)
        scores.masked_fill_(mask, -float('inf'))
        weights = nn.Softmax(dim=-1)(scores)
        weights = self.dropout(weights)
        if head_mask is not None:
            weights = weights * head_mask
        context = torch.matmul(weights, v)
        context = unshape(context)
        context = self.out_lin(context)
        if self.output_attentions:
            return context, weights
        else:
            return context,


class FFN(nn.Module):

    def __init__(self, config):
        super(FFN, self).__init__()
        self.dropout = nn.Dropout(p=config.dropout)
        self.lin1 = nn.Linear(in_features=config.dim, out_features=config.
            hidden_dim)
        self.lin2 = nn.Linear(in_features=config.hidden_dim, out_features=
            config.dim)
        assert config.activation in ['relu', 'gelu'
            ], "activation ({}) must be in ['relu', 'gelu']".format(config.
            activation)
        self.activation = gelu if config.activation == 'gelu' else nn.ReLU()

    def forward(self, input):
        x = self.lin1(input)
        x = self.activation(x)
        x = self.lin2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):

    def __init__(self, config):
        super(TransformerBlock, self).__init__()
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.hidden_dim = config.hidden_dim
        self.dropout = nn.Dropout(p=config.dropout)
        self.activation = config.activation
        self.output_attentions = config.output_attentions
        assert config.dim % config.n_heads == 0
        self.attention = MultiHeadSelfAttention(config)
        self.sa_layer_norm = nn.LayerNorm(normalized_shape=config.dim, eps=
            1e-12)
        self.ffn = FFN(config)
        self.output_layer_norm = nn.LayerNorm(normalized_shape=config.dim,
            eps=1e-12)

    def forward(self, x, attn_mask=None, head_mask=None):
        """
        Parameters
        ----------
        x: torch.tensor(bs, seq_length, dim)
        attn_mask: torch.tensor(bs, seq_length)

        Outputs
        -------
        sa_weights: torch.tensor(bs, n_heads, seq_length, seq_length)
            The attention weights
        ffn_output: torch.tensor(bs, seq_length, dim)
            The output of the transformer block contextualization.
        """
        sa_output = self.attention(query=x, key=x, value=x, mask=attn_mask,
            head_mask=head_mask)
        if self.output_attentions:
            sa_output, sa_weights = sa_output
        else:
            assert type(sa_output) == tuple
            sa_output = sa_output[0]
        sa_output = self.sa_layer_norm(sa_output + x)
        ffn_output = self.ffn(sa_output)
        ffn_output = self.output_layer_norm(ffn_output + sa_output)
        output = ffn_output,
        if self.output_attentions:
            output = (sa_weights,) + output
        return output


class Transformer(nn.Module):

    def __init__(self, config):
        super(Transformer, self).__init__()
        self.n_layers = config.n_layers
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        layer = TransformerBlock(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(
            config.n_layers)])

    def forward(self, x, attn_mask=None, head_mask=None):
        """
        Parameters
        ----------
        x: torch.tensor(bs, seq_length, dim)
            Input sequence embedded.
        attn_mask: torch.tensor(bs, seq_length)
            Attention mask on the sequence.

        Outputs
        -------
        hidden_state: torch.tensor(bs, seq_length, dim)
            Sequence of hiddens states in the last (top) layer
        all_hidden_states: Tuple[torch.tensor(bs, seq_length, dim)]
            Tuple of length n_layers with the hidden states from each layer.
            Optional: only if output_hidden_states=True
        all_attentions: Tuple[torch.tensor(bs, n_heads, seq_length, seq_length)]
            Tuple of length n_layers with the attention weights from each layer
            Optional: only if output_attentions=True
        """
        all_hidden_states = ()
        all_attentions = ()
        hidden_state = x
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state,)
            layer_outputs = layer_module(x=hidden_state, attn_mask=
                attn_mask, head_mask=head_mask[i])
            hidden_state = layer_outputs[-1]
            if self.output_attentions:
                assert len(layer_outputs) == 2
                attentions = layer_outputs[0]
                all_attentions = all_attentions + (attentions,)
            else:
                assert len(layer_outputs) == 1
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_state,)
        outputs = hidden_state,
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs


def prune_conv1d_layer(layer, index, dim=1):
    """ Prune a Conv1D layer (a model parameters) to keep only entries in index.
        A Conv1D work as a Linear layer (see e.g. BERT) but the weights are transposed.
        Return the pruned layer as a new layer with requires_grad=True.
        Used to remove heads.
    """
    index = index.to(layer.weight.device)
    W = layer.weight.index_select(dim, index).clone().detach()
    if dim == 0:
        b = layer.bias.clone().detach()
    else:
        b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = Conv1D(new_size[1], new_size[0]).to(layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    new_layer.bias.requires_grad = False
    new_layer.bias.copy_(b.contiguous())
    new_layer.bias.requires_grad = True
    return new_layer


class Attention(nn.Module):

    def __init__(self, nx, n_ctx, config, scale=False):
        super(Attention, self).__init__()
        self.output_attentions = config.output_attentions
        n_state = nx
        assert n_state % config.n_head == 0
        self.register_buffer('bias', torch.tril(torch.ones(n_ctx, n_ctx)).
            view(1, 1, n_ctx, n_ctx))
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale
        self.c_attn = Conv1D(n_state * 3, nx)
        self.c_proj = Conv1D(n_state, nx)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.n_head, self.split_size // self.n_head)
        heads = set(heads) - self.pruned_heads
        for head in heads:
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        index_attn = torch.cat([index, index + self.split_size, index + 2 *
            self.split_size])
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)
        self.split_size = self.split_size // self.n_head * (self.n_head -
            len(heads))
        self.n_head = self.n_head - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def _attn(self, q, k, v, attention_mask=None, head_mask=None):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        nd, ns = w.size(-2), w.size(-1)
        b = self.bias[:, :, ns - nd:ns, :ns]
        w = w * b - 10000.0 * (1 - b)
        if attention_mask is not None:
            w = w + attention_mask
        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)
        if head_mask is not None:
            w = w * head_mask
        outputs = [torch.matmul(w, v)]
        if self.output_attentions:
            outputs.append(w)
        return outputs

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)

    def forward(self, x, layer_past=None, attention_mask=None, head_mask=None):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        if layer_past is not None:
            past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[
                1]
            key = torch.cat((past_key, key), dim=-1)
            value = torch.cat((past_value, value), dim=-2)
        present = torch.stack((key.transpose(-2, -1), value))
        attn_outputs = self._attn(query, key, value, attention_mask, head_mask)
        a = attn_outputs[0]
        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)
        outputs = [a, present] + attn_outputs[1:]
        return outputs


class MLP(nn.Module):

    def __init__(self, n_state, config):
        super(MLP, self).__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = gelu
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)


class Block(nn.Module):

    def __init__(self, n_ctx, config, scale=False):
        super(Block, self).__init__()
        nx = config.n_embd
        self.ln_1 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.attn = Attention(nx, n_ctx, config, scale)
        self.ln_2 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)

    def forward(self, x, layer_past=None, attention_mask=None, head_mask=None):
        output_attn = self.attn(self.ln_1(x), layer_past=layer_past,
            attention_mask=attention_mask, head_mask=head_mask)
        a = output_attn[0]
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        outputs = [x] + output_attn[1:]
        return outputs


class Attention(nn.Module):

    def __init__(self, nx, n_ctx, config, scale=False):
        super(Attention, self).__init__()
        n_state = nx
        assert n_state % config.n_head == 0
        self.register_buffer('bias', torch.tril(torch.ones(n_ctx, n_ctx)).
            view(1, 1, n_ctx, n_ctx))
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale
        self.output_attentions = config.output_attentions
        self.c_attn = Conv1D(n_state * 3, nx)
        self.c_proj = Conv1D(n_state, nx)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.n_head, self.split_size // self.n_head)
        heads = set(heads) - self.pruned_heads
        for head in heads:
            head -= sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        index_attn = torch.cat([index, index + self.split_size, index + 2 *
            self.split_size])
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)
        self.split_size = self.split_size // self.n_head * (self.n_head -
            len(heads))
        self.n_head = self.n_head - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def _attn(self, q, k, v, attention_mask=None, head_mask=None):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        b = self.bias[:, :, :w.size(-2), :w.size(-1)]
        w = w * b + -10000.0 * (1 - b)
        if attention_mask is not None:
            w = w + attention_mask
        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)
        if head_mask is not None:
            w = w * head_mask
        outputs = [torch.matmul(w, v)]
        if self.output_attentions:
            outputs.append(w)
        return outputs

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)

    def forward(self, x, attention_mask=None, head_mask=None):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        attn_outputs = self._attn(query, key, value, attention_mask, head_mask)
        a = attn_outputs[0]
        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)
        outputs = [a] + attn_outputs[1:]
        return outputs


ACT_FNS = {'relu': nn.ReLU, 'swish': swish, 'gelu': gelu}


class MLP(nn.Module):

    def __init__(self, n_state, config):
        super(MLP, self).__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = ACT_FNS[config.afn]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)


class Block(nn.Module):

    def __init__(self, n_ctx, config, scale=False):
        super(Block, self).__init__()
        nx = config.n_embd
        self.attn = Attention(nx, n_ctx, config, scale)
        self.ln_1 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)
        self.ln_2 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)

    def forward(self, x, attention_mask=None, head_mask=None):
        attn_outputs = self.attn(x, attention_mask=attention_mask,
            head_mask=head_mask)
        a = attn_outputs[0]
        n = self.ln_1(x + a)
        m = self.mlp(n)
        h = self.ln_2(n + m)
        outputs = [h] + attn_outputs[1:]
        return outputs


class RobertaLMHead(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, config):
        super(RobertaLMHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = BertLayerNorm(config.hidden_size, eps=config.
            layer_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size,
            bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)
        x = self.decoder(x) + self.bias
        return x


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super(RobertaClassificationHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, (0), :]
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class PositionalEmbedding(nn.Module):

    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()
        self.demb = demb
        inv_freq = 1 / 10000 ** (torch.arange(0.0, demb, 2.0) / demb)
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        if bsz is not None:
            return pos_emb[:, (None), :].expand(-1, bsz, -1)
        else:
            return pos_emb[:, (None), :]


class PositionwiseFF(nn.Module):

    def __init__(self, d_model, d_inner, dropout, pre_lnorm=False,
        layer_norm_epsilon=1e-05):
        super(PositionwiseFF, self).__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout
        self.CoreNet = nn.Sequential(nn.Linear(d_model, d_inner), nn.ReLU(
            inplace=True), nn.Dropout(dropout), nn.Linear(d_inner, d_model),
            nn.Dropout(dropout))
        self.layer_norm = nn.LayerNorm(d_model, eps=layer_norm_epsilon)
        self.pre_lnorm = pre_lnorm

    def forward(self, inp):
        if self.pre_lnorm:
            core_out = self.CoreNet(self.layer_norm(inp))
            output = core_out + inp
        else:
            core_out = self.CoreNet(inp)
            output = self.layer_norm(inp + core_out)
        return output


class RelPartialLearnableMultiHeadAttn(nn.Module):

    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0, tgt_len
        =None, ext_len=None, mem_len=None, pre_lnorm=False, r_r_bias=None,
        r_w_bias=None, output_attentions=False, layer_norm_epsilon=1e-05):
        super(RelPartialLearnableMultiHeadAttn, self).__init__()
        self.output_attentions = output_attentions
        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout
        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)
        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model, eps=layer_norm_epsilon)
        self.scale = 1 / d_head ** 0.5
        self.pre_lnorm = pre_lnorm
        if r_r_bias is None or r_w_bias is None:
            self.r_r_bias = nn.Parameter(torch.FloatTensor(self.n_head,
                self.d_head))
            self.r_w_bias = nn.Parameter(torch.FloatTensor(self.n_head,
                self.d_head))
        else:
            self.r_r_bias = r_r_bias
            self.r_w_bias = r_w_bias
        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head,
            bias=False)

    def _rel_shift(self, x):
        zero_pad_shape = (x.size(0), 1) + x.size()[2:]
        zero_pad = torch.zeros(zero_pad_shape, device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)
        x_padded_shape = (x.size(1) + 1, x.size(0)) + x.size()[2:]
        x_padded = x_padded.view(*x_padded_shape)
        x = x_padded[1:].view_as(x)
        return x

    def forward(self, w, r, attn_mask=None, mems=None, head_mask=None):
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)
        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            r_head_k = self.r_net(r)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            r_head_k = self.r_net(r)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
        klen = w_head_k.size(0)
        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)
        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)
        rw_head_q = w_head_q + self.r_w_bias
        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))
        rr_head_q = w_head_q + self.r_r_bias
        BD = torch.einsum('ibnd,jnd->ijbn', (rr_head_q, r_head_k))
        BD = self._rel_shift(BD)
        attn_score = AC + BD
        attn_score.mul_(self.scale)
        if attn_mask is not None and torch.sum(attn_mask).item():
            attn_mask = attn_mask == 1
            if attn_mask.dim() == 2:
                if next(self.parameters()).dtype == torch.float16:
                    attn_score = attn_score.float().masked_fill(attn_mask[(
                        None), :, :, (None)], -65000).type_as(attn_score)
                else:
                    attn_score = attn_score.float().masked_fill(attn_mask[(
                        None), :, :, (None)], -1e+30).type_as(attn_score)
            elif attn_mask.dim() == 3:
                if next(self.parameters()).dtype == torch.float16:
                    attn_score = attn_score.float().masked_fill(attn_mask[:,
                        :, :, (None)], -65000).type_as(attn_score)
                else:
                    attn_score = attn_score.float().masked_fill(attn_mask[:,
                        :, :, (None)], -1e+30).type_as(attn_score)
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)
        if head_mask is not None:
            attn_prob = attn_prob * head_mask
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))
        attn_vec = attn_vec.contiguous().view(attn_vec.size(0), attn_vec.
            size(1), self.n_head * self.d_head)
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)
        if self.pre_lnorm:
            outputs = [w + attn_out]
        else:
            outputs = [self.layer_norm(w + attn_out)]
        if self.output_attentions:
            outputs.append(attn_prob)
        return outputs


class RelPartialLearnableDecoderLayer(nn.Module):

    def __init__(self, n_head, d_model, d_head, d_inner, dropout,
        layer_norm_epsilon=1e-05, **kwargs):
        super(RelPartialLearnableDecoderLayer, self).__init__()
        self.dec_attn = RelPartialLearnableMultiHeadAttn(n_head, d_model,
            d_head, dropout, layer_norm_epsilon=layer_norm_epsilon, **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout, pre_lnorm=
            kwargs.get('pre_lnorm'), layer_norm_epsilon=layer_norm_epsilon)

    def forward(self, dec_inp, r, dec_attn_mask=None, mems=None, head_mask=None
        ):
        attn_outputs = self.dec_attn(dec_inp, r, attn_mask=dec_attn_mask,
            mems=mems, head_mask=head_mask)
        ff_output = self.pos_ff(attn_outputs[0])
        outputs = [ff_output] + attn_outputs[1:]
        return outputs


class AdaptiveEmbedding(nn.Module):

    def __init__(self, n_token, d_embed, d_proj, cutoffs, div_val=1,
        sample_softmax=False):
        super(AdaptiveEmbedding, self).__init__()
        self.n_token = n_token
        self.d_embed = d_embed
        self.cutoffs = cutoffs + [n_token]
        self.div_val = div_val
        self.d_proj = d_proj
        self.emb_scale = d_proj ** 0.5
        self.cutoff_ends = [0] + self.cutoffs
        self.emb_layers = nn.ModuleList()
        self.emb_projs = nn.ParameterList()
        if div_val == 1:
            self.emb_layers.append(nn.Embedding(n_token, d_embed, sparse=
                sample_softmax > 0))
            if d_proj != d_embed:
                self.emb_projs.append(nn.Parameter(torch.FloatTensor(d_proj,
                    d_embed)))
        else:
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                d_emb_i = d_embed // div_val ** i
                self.emb_layers.append(nn.Embedding(r_idx - l_idx, d_emb_i))
                self.emb_projs.append(nn.Parameter(torch.FloatTensor(d_proj,
                    d_emb_i)))

    def forward(self, inp):
        if self.div_val == 1:
            embed = self.emb_layers[0](inp)
            if self.d_proj != self.d_embed:
                embed = F.linear(embed, self.emb_projs[0])
        else:
            param = next(self.parameters())
            inp_flat = inp.view(-1)
            emb_flat = torch.zeros([inp_flat.size(0), self.d_proj], dtype=
                param.dtype, device=param.device)
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                mask_i = (inp_flat >= l_idx) & (inp_flat < r_idx)
                indices_i = mask_i.nonzero().squeeze()
                if indices_i.numel() == 0:
                    continue
                inp_i = inp_flat.index_select(0, indices_i) - l_idx
                emb_i = self.emb_layers[i](inp_i)
                emb_i = F.linear(emb_i, self.emb_projs[i])
                emb_flat.index_copy_(0, indices_i, emb_i)
            embed_shape = inp.size() + (self.d_proj,)
            embed = emb_flat.view(embed_shape)
        embed.mul_(self.emb_scale)
        return embed


class ProjectedAdaptiveLogSoftmax(nn.Module):

    def __init__(self, n_token, d_embed, d_proj, cutoffs, div_val=1,
        keep_order=False):
        super(ProjectedAdaptiveLogSoftmax, self).__init__()
        self.n_token = n_token
        self.d_embed = d_embed
        self.d_proj = d_proj
        self.cutoffs = cutoffs + [n_token]
        self.cutoff_ends = [0] + self.cutoffs
        self.div_val = div_val
        self.shortlist_size = self.cutoffs[0]
        self.n_clusters = len(self.cutoffs) - 1
        self.head_size = self.shortlist_size + self.n_clusters
        if self.n_clusters > 0:
            self.cluster_weight = nn.Parameter(torch.zeros(self.n_clusters,
                self.d_embed))
            self.cluster_bias = nn.Parameter(torch.zeros(self.n_clusters))
        self.out_layers = nn.ModuleList()
        self.out_projs = nn.ParameterList()
        if div_val == 1:
            for i in range(len(self.cutoffs)):
                if d_proj != d_embed:
                    self.out_projs.append(nn.Parameter(torch.FloatTensor(
                        d_proj, d_embed)))
                else:
                    self.out_projs.append(None)
            self.out_layers.append(nn.Linear(d_embed, n_token))
        else:
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                d_emb_i = d_embed // div_val ** i
                self.out_projs.append(nn.Parameter(torch.FloatTensor(d_proj,
                    d_emb_i)))
                self.out_layers.append(nn.Linear(d_emb_i, r_idx - l_idx))
        self.keep_order = keep_order

    def _compute_logit(self, hidden, weight, bias, proj):
        if proj is None:
            logit = F.linear(hidden, weight, bias=bias)
        else:
            proj_hid = F.linear(hidden, proj.t().contiguous())
            logit = F.linear(proj_hid, weight, bias=bias)
        return logit

    def forward(self, hidden, labels=None, keep_order=False):
        """
            Params:
                hidden :: [len*bsz x d_proj]
                labels :: [len*bsz]
            Return:
                if labels is None:
                    out :: [len*bsz] Negative log likelihood
                else:
                    out :: [len*bsz x n_tokens] log probabilities of tokens over the vocabulary
            We could replace this implementation by the native PyTorch one
            if their's had an option to set bias on all clusters in the native one.
            here: https://github.com/pytorch/pytorch/blob/dbe6a7a9ff1a364a8706bf5df58a1ca96d2fd9da/torch/nn/modules/adaptive.py#L138
        """
        if labels is not None:
            labels = labels.view(-1)
            if hidden.size(0) != labels.size(0):
                raise RuntimeError(
                    'Input and labels should have the same size in the batch dimension.'
                    )
        if self.n_clusters == 0:
            logit = self._compute_logit(hidden, self.out_layers[0].weight,
                self.out_layers[0].bias, self.out_projs[0])
            if labels is not None:
                out = -F.log_softmax(logit, dim=-1).gather(1, labels.
                    unsqueeze(1)).squeeze(1)
            else:
                out = F.log_softmax(logit, dim=-1)
        else:
            weights, biases = [], []
            for i in range(len(self.cutoffs)):
                if self.div_val == 1:
                    l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                    weight_i = self.out_layers[0].weight[l_idx:r_idx]
                    bias_i = self.out_layers[0].bias[l_idx:r_idx]
                else:
                    weight_i = self.out_layers[i].weight
                    bias_i = self.out_layers[i].bias
                if i == 0:
                    weight_i = torch.cat([weight_i, self.cluster_weight], dim=0
                        )
                    bias_i = torch.cat([bias_i, self.cluster_bias], dim=0)
                weights.append(weight_i)
                biases.append(bias_i)
            head_weight, head_bias, head_proj = weights[0], biases[0
                ], self.out_projs[0]
            head_logit = self._compute_logit(hidden, head_weight, head_bias,
                head_proj)
            head_logprob = F.log_softmax(head_logit, dim=1)
            if labels is None:
                out = hidden.new_empty((head_logit.size(0), self.n_token))
            else:
                out = torch.zeros_like(labels, dtype=hidden.dtype, device=
                    hidden.device)
            offset = 0
            cutoff_values = [0] + self.cutoffs
            for i in range(len(cutoff_values) - 1):
                l_idx, r_idx = cutoff_values[i], cutoff_values[i + 1]
                if labels is not None:
                    mask_i = (labels >= l_idx) & (labels < r_idx)
                    indices_i = mask_i.nonzero().squeeze()
                    if indices_i.numel() == 0:
                        continue
                    target_i = labels.index_select(0, indices_i) - l_idx
                    head_logprob_i = head_logprob.index_select(0, indices_i)
                    hidden_i = hidden.index_select(0, indices_i)
                else:
                    hidden_i = hidden
                if i == 0:
                    if labels is not None:
                        logprob_i = head_logprob_i.gather(1, target_i[:, (
                            None)]).squeeze(1)
                    else:
                        out[:, :self.cutoffs[0]] = head_logprob[:, :self.
                            cutoffs[0]]
                else:
                    weight_i, bias_i, proj_i = weights[i], biases[i
                        ], self.out_projs[i]
                    tail_logit_i = self._compute_logit(hidden_i, weight_i,
                        bias_i, proj_i)
                    tail_logprob_i = F.log_softmax(tail_logit_i, dim=1)
                    cluster_prob_idx = self.cutoffs[0] + i - 1
                    if labels is not None:
                        logprob_i = head_logprob_i[:, (cluster_prob_idx)
                            ] + tail_logprob_i.gather(1, target_i[:, (None)]
                            ).squeeze(1)
                    else:
                        logprob_i = head_logprob[:, (cluster_prob_idx), (None)
                            ] + tail_logprob_i
                        out[:, l_idx:r_idx] = logprob_i
                if labels is not None:
                    if hasattr(self, 'keep_order'
                        ) and self.keep_order or keep_order:
                        out.index_copy_(0, indices_i, -logprob_i)
                    else:
                        out[offset:offset + logprob_i.size(0)].copy_(-logprob_i
                            )
                    offset += logprob_i.size(0)
        return out

    def log_prob(self, hidden):
        """ Computes log probabilities for all :math:`n\\_classes`
        From: https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/adaptive.py
        Args:
            hidden (Tensor): a minibatch of examples
        Returns:
            log-probabilities of for each class :math:`c`
            in range :math:`0 <= c <= n\\_classes`, where :math:`n\\_classes` is a
            parameter passed to ``AdaptiveLogSoftmaxWithLoss`` constructor.
        Shape:
            - Input: :math:`(N, in\\_features)`
            - Output: :math:`(N, n\\_classes)`
        """
        if self.n_clusters == 0:
            logit = self._compute_logit(hidden, self.out_layers[0].weight,
                self.out_layers[0].bias, self.out_projs[0])
            return F.log_softmax(logit, dim=-1)
        else:
            weights, biases = [], []
            for i in range(len(self.cutoffs)):
                if self.div_val == 1:
                    l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                    weight_i = self.out_layers[0].weight[l_idx:r_idx]
                    bias_i = self.out_layers[0].bias[l_idx:r_idx]
                else:
                    weight_i = self.out_layers[i].weight
                    bias_i = self.out_layers[i].bias
                if i == 0:
                    weight_i = torch.cat([weight_i, self.cluster_weight], dim=0
                        )
                    bias_i = torch.cat([bias_i, self.cluster_bias], dim=0)
                weights.append(weight_i)
                biases.append(bias_i)
            head_weight, head_bias, head_proj = weights[0], biases[0
                ], self.out_projs[0]
            head_logit = self._compute_logit(hidden, head_weight, head_bias,
                head_proj)
            out = hidden.new_empty((head_logit.size(0), self.n_token))
            head_logprob = F.log_softmax(head_logit, dim=1)
            cutoff_values = [0] + self.cutoffs
            for i in range(len(cutoff_values) - 1):
                start_idx, stop_idx = cutoff_values[i], cutoff_values[i + 1]
                if i == 0:
                    out[:, :self.cutoffs[0]] = head_logprob[:, :self.cutoffs[0]
                        ]
                else:
                    weight_i, bias_i, proj_i = weights[i], biases[i
                        ], self.out_projs[i]
                    tail_logit_i = self._compute_logit(hidden, weight_i,
                        bias_i, proj_i)
                    tail_logprob_i = F.log_softmax(tail_logit_i, dim=1)
                    logprob_i = head_logprob[:, (-i)] + tail_logprob_i
                    out[:, (start_idx), (stop_idx)] = logprob_i
            return out


CONFIG_NAME = 'config.json'


class Conv1D(nn.Module):

    def __init__(self, nf, nx):
        """ Conv1D layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2)
            Basically works like a Linear layer but the weights are transposed
        """
        super(Conv1D, self).__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x


class PoolerStartLogits(nn.Module):
    """ Compute SQuAD start_logits from sequence hidden states. """

    def __init__(self, config):
        super(PoolerStartLogits, self).__init__()
        self.dense = nn.Linear(config.hidden_size, 1)

    def forward(self, hidden_states, p_mask=None):
        """ Args:
            **p_mask**: (`optional`) ``torch.FloatTensor`` of shape `(batch_size, seq_len)`
                invalid position mask such as query and special symbols (PAD, SEP, CLS)
                1.0 means token should be masked.
        """
        x = self.dense(hidden_states).squeeze(-1)
        if p_mask is not None:
            if next(self.parameters()).dtype == torch.float16:
                x = x * (1 - p_mask) - 65500 * p_mask
            else:
                x = x * (1 - p_mask) - 1e+30 * p_mask
        return x


class PoolerEndLogits(nn.Module):
    """ Compute SQuAD end_logits from sequence hidden states and start token hidden state.
    """

    def __init__(self, config):
        super(PoolerEndLogits, self).__init__()
        self.dense_0 = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.activation = nn.Tanh()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.
            layer_norm_eps)
        self.dense_1 = nn.Linear(config.hidden_size, 1)

    def forward(self, hidden_states, start_states=None, start_positions=
        None, p_mask=None):
        """ Args:
            One of ``start_states``, ``start_positions`` should be not None.
            If both are set, ``start_positions`` overrides ``start_states``.

            **start_states**: ``torch.LongTensor`` of shape identical to hidden_states
                hidden states of the first tokens for the labeled span.
            **start_positions**: ``torch.LongTensor`` of shape ``(batch_size,)``
                position of the first token for the labeled span:
            **p_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, seq_len)``
                Mask of invalid position such as query and special symbols (PAD, SEP, CLS)
                1.0 means token should be masked.
        """
        assert start_states is not None or start_positions is not None, 'One of start_states, start_positions should be not None'
        if start_positions is not None:
            slen, hsz = hidden_states.shape[-2:]
            start_positions = start_positions[:, (None), (None)].expand(-1,
                -1, hsz)
            start_states = hidden_states.gather(-2, start_positions)
            start_states = start_states.expand(-1, slen, -1)
        x = self.dense_0(torch.cat([hidden_states, start_states], dim=-1))
        x = self.activation(x)
        x = self.LayerNorm(x)
        x = self.dense_1(x).squeeze(-1)
        if p_mask is not None:
            if next(self.parameters()).dtype == torch.float16:
                x = x * (1 - p_mask) - 65500 * p_mask
            else:
                x = x * (1 - p_mask) - 1e+30 * p_mask
        return x


class PoolerAnswerClass(nn.Module):
    """ Compute SQuAD 2.0 answer class from classification and start tokens hidden states. """

    def __init__(self, config):
        super(PoolerAnswerClass, self).__init__()
        self.dense_0 = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.activation = nn.Tanh()
        self.dense_1 = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, hidden_states, start_states=None, start_positions=
        None, cls_index=None):
        """
        Args:
            One of ``start_states``, ``start_positions`` should be not None.
            If both are set, ``start_positions`` overrides ``start_states``.

            **start_states**: ``torch.LongTensor`` of shape identical to ``hidden_states``.
                hidden states of the first tokens for the labeled span.
            **start_positions**: ``torch.LongTensor`` of shape ``(batch_size,)``
                position of the first token for the labeled span.
            **cls_index**: torch.LongTensor of shape ``(batch_size,)``
                position of the CLS token. If None, take the last token.

            note(Original repo):
                no dependency on end_feature so that we can obtain one single `cls_logits`
                for each sample
        """
        hsz = hidden_states.shape[-1]
        assert start_states is not None or start_positions is not None, 'One of start_states, start_positions should be not None'
        if start_positions is not None:
            start_positions = start_positions[:, (None), (None)].expand(-1,
                -1, hsz)
            start_states = hidden_states.gather(-2, start_positions).squeeze(-2
                )
        if cls_index is not None:
            cls_index = cls_index[:, (None), (None)].expand(-1, -1, hsz)
            cls_token_state = hidden_states.gather(-2, cls_index).squeeze(-2)
        else:
            cls_token_state = hidden_states[:, (-1), :]
        x = self.dense_0(torch.cat([start_states, cls_token_state], dim=-1))
        x = self.activation(x)
        x = self.dense_1(x).squeeze(-1)
        return x


class SQuADHead(nn.Module):
    """ A SQuAD head inspired by XLNet.

    Parameters:
        config (:class:`~transformers.XLNetConfig`): Model configuration class with all the parameters of the model.

    Inputs:
        **hidden_states**: ``torch.FloatTensor`` of shape ``(batch_size, seq_len, hidden_size)``
            hidden states of sequence tokens
        **start_positions**: ``torch.LongTensor`` of shape ``(batch_size,)``
            position of the first token for the labeled span.
        **end_positions**: ``torch.LongTensor`` of shape ``(batch_size,)``
            position of the last token for the labeled span.
        **cls_index**: torch.LongTensor of shape ``(batch_size,)``
            position of the CLS token. If None, take the last token.
        **is_impossible**: ``torch.LongTensor`` of shape ``(batch_size,)``
            Whether the question has a possible answer in the paragraph or not.
        **p_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, seq_len)``
            Mask of invalid position such as query and special symbols (PAD, SEP, CLS)
            1.0 means token should be masked.

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned if both ``start_positions`` and ``end_positions`` are provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification loss as the sum of start token, end token (and is_impossible if provided) classification losses.
        **start_top_log_probs**: (`optional`, returned if ``start_positions`` or ``end_positions`` is not provided)
            ``torch.FloatTensor`` of shape ``(batch_size, config.start_n_top)``
            Log probabilities for the top config.start_n_top start token possibilities (beam-search).
        **start_top_index**: (`optional`, returned if ``start_positions`` or ``end_positions`` is not provided)
            ``torch.LongTensor`` of shape ``(batch_size, config.start_n_top)``
            Indices for the top config.start_n_top start token possibilities (beam-search).
        **end_top_log_probs**: (`optional`, returned if ``start_positions`` or ``end_positions`` is not provided)
            ``torch.FloatTensor`` of shape ``(batch_size, config.start_n_top * config.end_n_top)``
            Log probabilities for the top ``config.start_n_top * config.end_n_top`` end token possibilities (beam-search).
        **end_top_index**: (`optional`, returned if ``start_positions`` or ``end_positions`` is not provided)
            ``torch.LongTensor`` of shape ``(batch_size, config.start_n_top * config.end_n_top)``
            Indices for the top ``config.start_n_top * config.end_n_top`` end token possibilities (beam-search).
        **cls_logits**: (`optional`, returned if ``start_positions`` or ``end_positions`` is not provided)
            ``torch.FloatTensor`` of shape ``(batch_size,)``
            Log probabilities for the ``is_impossible`` label of the answers.
    """

    def __init__(self, config):
        super(SQuADHead, self).__init__()
        self.start_n_top = config.start_n_top
        self.end_n_top = config.end_n_top
        self.start_logits = PoolerStartLogits(config)
        self.end_logits = PoolerEndLogits(config)
        self.answer_class = PoolerAnswerClass(config)

    def forward(self, hidden_states, start_positions=None, end_positions=
        None, cls_index=None, is_impossible=None, p_mask=None):
        outputs = ()
        start_logits = self.start_logits(hidden_states, p_mask=p_mask)
        if start_positions is not None and end_positions is not None:
            for x in (start_positions, end_positions, cls_index, is_impossible
                ):
                if x is not None and x.dim() > 1:
                    x.squeeze_(-1)
            end_logits = self.end_logits(hidden_states, start_positions=
                start_positions, p_mask=p_mask)
            loss_fct = CrossEntropyLoss()
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            if cls_index is not None and is_impossible is not None:
                cls_logits = self.answer_class(hidden_states,
                    start_positions=start_positions, cls_index=cls_index)
                loss_fct_cls = nn.BCEWithLogitsLoss()
                cls_loss = loss_fct_cls(cls_logits, is_impossible)
                total_loss += cls_loss * 0.5
            outputs = (total_loss,) + outputs
        else:
            bsz, slen, hsz = hidden_states.size()
            start_log_probs = F.softmax(start_logits, dim=-1)
            start_top_log_probs, start_top_index = torch.topk(start_log_probs,
                self.start_n_top, dim=-1)
            start_top_index_exp = start_top_index.unsqueeze(-1).expand(-1, 
                -1, hsz)
            start_states = torch.gather(hidden_states, -2, start_top_index_exp)
            start_states = start_states.unsqueeze(1).expand(-1, slen, -1, -1)
            hidden_states_expanded = hidden_states.unsqueeze(2).expand_as(
                start_states)
            p_mask = p_mask.unsqueeze(-1) if p_mask is not None else None
            end_logits = self.end_logits(hidden_states_expanded,
                start_states=start_states, p_mask=p_mask)
            end_log_probs = F.softmax(end_logits, dim=1)
            end_top_log_probs, end_top_index = torch.topk(end_log_probs,
                self.end_n_top, dim=1)
            end_top_log_probs = end_top_log_probs.view(-1, self.start_n_top *
                self.end_n_top)
            end_top_index = end_top_index.view(-1, self.start_n_top * self.
                end_n_top)
            start_states = torch.einsum('blh,bl->bh', hidden_states,
                start_log_probs)
            cls_logits = self.answer_class(hidden_states, start_states=
                start_states, cls_index=cls_index)
            outputs = (start_top_log_probs, start_top_index,
                end_top_log_probs, end_top_index, cls_logits) + outputs
        return outputs


class SequenceSummary(nn.Module):
    """ Compute a single vector summary of a sequence hidden states according to various possibilities:
        Args of the config class:
            summary_type:
                - 'last' => [default] take the last token hidden state (like XLNet)
                - 'first' => take the first token hidden state (like Bert)
                - 'mean' => take the mean of all tokens hidden states
                - 'cls_index' => supply a Tensor of classification token position (GPT/GPT-2)
                - 'attn' => Not implemented now, use multi-head attention
            summary_use_proj: Add a projection after the vector extraction
            summary_proj_to_labels: If True, the projection outputs to config.num_labels classes (otherwise to hidden_size). Default: False.
            summary_activation: 'tanh' => add a tanh activation to the output, Other => no activation. Default
            summary_first_dropout: Add a dropout before the projection and activation
            summary_last_dropout: Add a dropout after the projection and activation
    """

    def __init__(self, config):
        super(SequenceSummary, self).__init__()
        self.summary_type = config.summary_type if hasattr(config,
            'summary_use_proj') else 'last'
        if self.summary_type == 'attn':
            raise NotImplementedError
        self.summary = Identity()
        if hasattr(config, 'summary_use_proj') and config.summary_use_proj:
            if hasattr(config, 'summary_proj_to_labels'
                ) and config.summary_proj_to_labels and config.num_labels > 0:
                num_classes = config.num_labels
            else:
                num_classes = config.hidden_size
            self.summary = nn.Linear(config.hidden_size, num_classes)
        self.activation = Identity()
        if hasattr(config, 'summary_activation'
            ) and config.summary_activation == 'tanh':
            self.activation = nn.Tanh()
        self.first_dropout = Identity()
        if hasattr(config, 'summary_first_dropout'
            ) and config.summary_first_dropout > 0:
            self.first_dropout = nn.Dropout(config.summary_first_dropout)
        self.last_dropout = Identity()
        if hasattr(config, 'summary_last_dropout'
            ) and config.summary_last_dropout > 0:
            self.last_dropout = nn.Dropout(config.summary_last_dropout)

    def forward(self, hidden_states, cls_index=None):
        """ hidden_states: float Tensor in shape [bsz, ..., seq_len, hidden_size], the hidden-states of the last layer.
            cls_index: [optional] position of the classification token if summary_type == 'cls_index',
                shape (bsz,) or more generally (bsz, ...) where ... are optional leading dimensions of hidden_states.
                if summary_type == 'cls_index' and cls_index is None:
                    we take the last token of the sequence as classification token
        """
        if self.summary_type == 'last':
            output = hidden_states[:, (-1)]
        elif self.summary_type == 'first':
            output = hidden_states[:, (0)]
        elif self.summary_type == 'mean':
            output = hidden_states.mean(dim=1)
        elif self.summary_type == 'cls_index':
            if cls_index is None:
                cls_index = torch.full_like(hidden_states[(...), :1, :], 
                    hidden_states.shape[-2] - 1, dtype=torch.long)
            else:
                cls_index = cls_index.unsqueeze(-1).unsqueeze(-1)
                cls_index = cls_index.expand((-1,) * (cls_index.dim() - 1) +
                    (hidden_states.size(-1),))
            output = hidden_states.gather(-2, cls_index).squeeze(-2)
        elif self.summary_type == 'attn':
            raise NotImplementedError
        output = self.first_dropout(output)
        output = self.summary(output)
        output = self.activation(output)
        output = self.last_dropout(output)
        return output


class MultiHeadAttention(nn.Module):
    NEW_ID = itertools.count()

    def __init__(self, n_heads, dim, config):
        super(MultiHeadAttention, self).__init__()
        self.layer_id = next(MultiHeadAttention.NEW_ID)
        self.output_attentions = config.output_attentions
        self.dim = dim
        self.n_heads = n_heads
        self.dropout = config.attention_dropout
        assert self.dim % self.n_heads == 0
        self.q_lin = nn.Linear(dim, dim)
        self.k_lin = nn.Linear(dim, dim)
        self.v_lin = nn.Linear(dim, dim)
        self.out_lin = nn.Linear(dim, dim)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        attention_head_size = self.dim // self.n_heads
        if len(heads) == 0:
            return
        mask = torch.ones(self.n_heads, attention_head_size)
        heads = set(heads) - self.pruned_heads
        for head in heads:
            head -= sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        self.q_lin = prune_linear_layer(self.q_lin, index)
        self.k_lin = prune_linear_layer(self.k_lin, index)
        self.v_lin = prune_linear_layer(self.v_lin, index)
        self.out_lin = prune_linear_layer(self.out_lin, index, dim=1)
        self.n_heads = self.n_heads - len(heads)
        self.dim = attention_head_size * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, input, mask, kv=None, cache=None, head_mask=None):
        """
        Self-attention (if kv is None) or attention over source sentence (provided by kv).
        """
        bs, qlen, dim = input.size()
        if kv is None:
            klen = qlen if cache is None else cache['slen'] + qlen
        else:
            klen = kv.size(1)
        n_heads = self.n_heads
        dim_per_head = self.dim // n_heads
        mask_reshape = (bs, 1, qlen, klen) if mask.dim() == 3 else (bs, 1, 
            1, klen)

        def shape(x):
            """  projection """
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads *
                dim_per_head)
        q = shape(self.q_lin(input))
        if kv is None:
            k = shape(self.k_lin(input))
            v = shape(self.v_lin(input))
        elif cache is None or self.layer_id not in cache:
            k = v = kv
            k = shape(self.k_lin(k))
            v = shape(self.v_lin(v))
        if cache is not None:
            if self.layer_id in cache:
                if kv is None:
                    k_, v_ = cache[self.layer_id]
                    k = torch.cat([k_, k], dim=2)
                    v = torch.cat([v_, v], dim=2)
                else:
                    k, v = cache[self.layer_id]
            cache[self.layer_id] = k, v
        q = q / math.sqrt(dim_per_head)
        scores = torch.matmul(q, k.transpose(2, 3))
        mask = (mask == 0).view(mask_reshape).expand_as(scores)
        scores.masked_fill_(mask, -float('inf'))
        weights = F.softmax(scores.float(), dim=-1).type_as(scores)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        if head_mask is not None:
            weights = weights * head_mask
        context = torch.matmul(weights, v)
        context = unshape(context)
        outputs = self.out_lin(context),
        if self.output_attentions:
            outputs = outputs + (weights,)
        return outputs


class TransformerFFN(nn.Module):

    def __init__(self, in_dim, dim_hidden, out_dim, config):
        super(TransformerFFN, self).__init__()
        self.dropout = config.dropout
        self.lin1 = nn.Linear(in_dim, dim_hidden)
        self.lin2 = nn.Linear(dim_hidden, out_dim)
        self.act = gelu if config.gelu_activation else F.relu

    def forward(self, input):
        x = self.lin1(input)
        x = self.act(x)
        x = self.lin2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class XLMPredLayer(nn.Module):
    """
    Prediction layer (cross_entropy or adaptive_softmax).
    """

    def __init__(self, config):
        super(XLMPredLayer, self).__init__()
        self.asm = config.asm
        self.n_words = config.n_words
        self.pad_index = config.pad_index
        dim = config.emb_dim
        if config.asm is False:
            self.proj = nn.Linear(dim, config.n_words, bias=True)
        else:
            self.proj = nn.AdaptiveLogSoftmaxWithLoss(in_features=dim,
                n_classes=config.n_words, cutoffs=config.asm_cutoffs,
                div_value=config.asm_div_value, head_bias=True)

    def forward(self, x, y=None):
        """ Compute the loss, and optionally the scores.
        """
        outputs = ()
        if self.asm is False:
            scores = self.proj(x)
            outputs = (scores,) + outputs
            if y is not None:
                loss = F.cross_entropy(scores.view(-1, self.n_words), y.
                    view(-1), reduction='elementwise_mean')
                outputs = (loss,) + outputs
        else:
            scores = self.proj.log_prob(x)
            outputs = (scores,) + outputs
            if y is not None:
                _, loss = self.proj(x, y)
                outputs = (loss,) + outputs
        return outputs


class XLNetRelativeAttention(nn.Module):

    def __init__(self, config):
        super(XLNetRelativeAttention, self).__init__()
        self.output_attentions = config.output_attentions
        if config.d_model % config.n_head != 0:
            raise ValueError(
                'The hidden size (%d) is not a multiple of the number of attention heads (%d)'
                 % (config.d_model, config.n_head))
        self.n_head = config.n_head
        self.d_head = config.d_head
        self.d_model = config.d_model
        self.scale = 1 / config.d_head ** 0.5
        self.q = nn.Parameter(torch.FloatTensor(config.d_model, self.n_head,
            self.d_head))
        self.k = nn.Parameter(torch.FloatTensor(config.d_model, self.n_head,
            self.d_head))
        self.v = nn.Parameter(torch.FloatTensor(config.d_model, self.n_head,
            self.d_head))
        self.o = nn.Parameter(torch.FloatTensor(config.d_model, self.n_head,
            self.d_head))
        self.r = nn.Parameter(torch.FloatTensor(config.d_model, self.n_head,
            self.d_head))
        self.r_r_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.
            d_head))
        self.r_s_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.
            d_head))
        self.r_w_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.
            d_head))
        self.seg_embed = nn.Parameter(torch.FloatTensor(2, self.n_head,
            self.d_head))
        self.layer_norm = XLNetLayerNorm(config.d_model, eps=config.
            layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

    def prune_heads(self, heads):
        raise NotImplementedError

    @staticmethod
    def rel_shift(x, klen=-1):
        """perform relative shift to form the relative attention score."""
        x_size = x.shape
        x = x.reshape(x_size[1], x_size[0], x_size[2], x_size[3])
        x = x[1:, (...)]
        x = x.reshape(x_size[0], x_size[1] - 1, x_size[2], x_size[3])
        x = torch.index_select(x, 1, torch.arange(klen, device=x.device,
            dtype=torch.long))
        return x

    @staticmethod
    def rel_shift_bnij(x, klen=-1):
        x_size = x.shape
        x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
        x = x[:, :, 1:, :]
        x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
        x = torch.index_select(x, 3, torch.arange(klen, device=x.device,
            dtype=torch.long))
        return x

    def rel_attn_core(self, q_head, k_head_h, v_head_h, k_head_r, seg_mat=
        None, attn_mask=None, head_mask=None):
        """Core relative positional attention operations."""
        ac = torch.einsum('ibnd,jbnd->bnij', q_head + self.r_w_bias, k_head_h)
        bd = torch.einsum('ibnd,jbnd->bnij', q_head + self.r_r_bias, k_head_r)
        bd = self.rel_shift_bnij(bd, klen=ac.shape[3])
        if seg_mat is None:
            ef = 0
        else:
            ef = torch.einsum('ibnd,snd->ibns', q_head + self.r_s_bias,
                self.seg_embed)
            ef = torch.einsum('ijbs,ibns->bnij', seg_mat, ef)
        attn_score = (ac + bd + ef) * self.scale
        if attn_mask is not None:
            if attn_mask.dtype == torch.float16:
                attn_score = attn_score - 65500 * torch.einsum('ijbn->bnij',
                    attn_mask)
            else:
                attn_score = attn_score - 1e+30 * torch.einsum('ijbn->bnij',
                    attn_mask)
        attn_prob = F.softmax(attn_score, dim=3)
        attn_prob = self.dropout(attn_prob)
        if head_mask is not None:
            attn_prob = attn_prob * torch.einsum('ijbn->bnij', head_mask)
        attn_vec = torch.einsum('bnij,jbnd->ibnd', attn_prob, v_head_h)
        if self.output_attentions:
            return attn_vec, torch.einsum('bnij->ijbn', attn_prob)
        return attn_vec

    def post_attention(self, h, attn_vec, residual=True):
        """Post-attention processing."""
        attn_out = torch.einsum('ibnd,hnd->ibh', attn_vec, self.o)
        attn_out = self.dropout(attn_out)
        if residual:
            attn_out = attn_out + h
        output = self.layer_norm(attn_out)
        return output

    def forward(self, h, g, attn_mask_h, attn_mask_g, r, seg_mat, mems=None,
        target_mapping=None, head_mask=None):
        if g is not None:
            if mems is not None and mems.dim() > 1:
                cat = torch.cat([mems, h], dim=0)
            else:
                cat = h
            k_head_h = torch.einsum('ibh,hnd->ibnd', cat, self.k)
            v_head_h = torch.einsum('ibh,hnd->ibnd', cat, self.v)
            k_head_r = torch.einsum('ibh,hnd->ibnd', r, self.r)
            q_head_h = torch.einsum('ibh,hnd->ibnd', h, self.q)
            attn_vec_h = self.rel_attn_core(q_head_h, k_head_h, v_head_h,
                k_head_r, seg_mat=seg_mat, attn_mask=attn_mask_h, head_mask
                =head_mask)
            if self.output_attentions:
                attn_vec_h, attn_prob_h = attn_vec_h
            output_h = self.post_attention(h, attn_vec_h)
            q_head_g = torch.einsum('ibh,hnd->ibnd', g, self.q)
            if target_mapping is not None:
                q_head_g = torch.einsum('mbnd,mlb->lbnd', q_head_g,
                    target_mapping)
                attn_vec_g = self.rel_attn_core(q_head_g, k_head_h,
                    v_head_h, k_head_r, seg_mat=seg_mat, attn_mask=
                    attn_mask_g, head_mask=head_mask)
                if self.output_attentions:
                    attn_vec_g, attn_prob_g = attn_vec_g
                attn_vec_g = torch.einsum('lbnd,mlb->mbnd', attn_vec_g,
                    target_mapping)
            else:
                attn_vec_g = self.rel_attn_core(q_head_g, k_head_h,
                    v_head_h, k_head_r, seg_mat=seg_mat, attn_mask=
                    attn_mask_g, head_mask=head_mask)
                if self.output_attentions:
                    attn_vec_g, attn_prob_g = attn_vec_g
            output_g = self.post_attention(g, attn_vec_g)
            if self.output_attentions:
                attn_prob = attn_prob_h, attn_prob_g
        else:
            if mems is not None and mems.dim() > 1:
                cat = torch.cat([mems, h], dim=0)
            else:
                cat = h
            q_head_h = torch.einsum('ibh,hnd->ibnd', h, self.q)
            k_head_h = torch.einsum('ibh,hnd->ibnd', cat, self.k)
            v_head_h = torch.einsum('ibh,hnd->ibnd', cat, self.v)
            k_head_r = torch.einsum('ibh,hnd->ibnd', r, self.r)
            attn_vec = self.rel_attn_core(q_head_h, k_head_h, v_head_h,
                k_head_r, seg_mat=seg_mat, attn_mask=attn_mask_h, head_mask
                =head_mask)
            if self.output_attentions:
                attn_vec, attn_prob = attn_vec
            output_h = self.post_attention(h, attn_vec)
            output_g = None
        outputs = output_h, output_g
        if self.output_attentions:
            outputs = outputs + (attn_prob,)
        return outputs


class XLNetFeedForward(nn.Module):

    def __init__(self, config):
        super(XLNetFeedForward, self).__init__()
        self.layer_norm = XLNetLayerNorm(config.d_model, eps=config.
            layer_norm_eps)
        self.layer_1 = nn.Linear(config.d_model, config.d_inner)
        self.layer_2 = nn.Linear(config.d_inner, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        if isinstance(config.ff_activation, str) or sys.version_info[0
            ] == 2 and isinstance(config.ff_activation, unicode):
            self.activation_function = ACT2FN[config.ff_activation]
        else:
            self.activation_function = config.ff_activation

    def forward(self, inp):
        output = inp
        output = self.layer_1(output)
        output = self.activation_function(output)
        output = self.dropout(output)
        output = self.layer_2(output)
        output = self.dropout(output)
        output = self.layer_norm(output + inp)
        return output


class XLNetLayer(nn.Module):

    def __init__(self, config):
        super(XLNetLayer, self).__init__()
        self.rel_attn = XLNetRelativeAttention(config)
        self.ff = XLNetFeedForward(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, output_h, output_g, attn_mask_h, attn_mask_g, r,
        seg_mat, mems=None, target_mapping=None, head_mask=None):
        outputs = self.rel_attn(output_h, output_g, attn_mask_h,
            attn_mask_g, r, seg_mat, mems=mems, target_mapping=
            target_mapping, head_mask=head_mask)
        output_h, output_g = outputs[:2]
        if output_g is not None:
            output_g = self.ff(output_g)
        output_h = self.ff(output_h)
        outputs = (output_h, output_g) + outputs[2:]
        return outputs


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_lonePatient_BERT_NER_Pytorch(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(AlbertEmbeddings(*[], **{'config': _mock_config(vocab_size=4, embedding_size=4, hidden_size=4, max_position_embeddings=4, type_vocab_size=4, layer_norm_eps=1, hidden_dropout_prob=0.5)}), [torch.zeros([4, 4], dtype=torch.int64)], {})

    def test_001(self):
        self._check(AlbertOnlyNSPHead(*[], **{'config': _mock_config(hidden_size=4)}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_002(self):
        self._check(AlbertOutput(*[], **{'config': _mock_config(intermediate_size=4, hidden_size=4, layer_norm_eps=1, hidden_dropout_prob=0.5)}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(AlbertPooler(*[], **{'config': _mock_config(hidden_size=4)}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_004(self):
        self._check(AlbertSelfOutput(*[], **{'config': _mock_config(hidden_size=4, layer_norm_eps=1, hidden_dropout_prob=0.5)}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_005(self):
        self._check(Attention(*[], **{'nx': 4, 'n_ctx': 4, 'config': _mock_config(n_head=4, output_attentions=4, attn_pdrop=0.5, resid_pdrop=0.5)}), [torch.rand([4, 4, 4])], {})

    @_fails_compile()
    def test_006(self):
        self._check(BertEmbeddings(*[], **{'config': _mock_config(vocab_size=4, hidden_size=4, max_position_embeddings=4, type_vocab_size=4, layer_norm_eps=1, hidden_dropout_prob=0.5)}), [torch.zeros([4, 4], dtype=torch.int64)], {})

    def test_007(self):
        self._check(BertIntermediate(*[], **{'config': _mock_config(hidden_size=4, intermediate_size=4, hidden_act=ReLU())}), [torch.rand([4, 4, 4, 4])], {})

    def test_008(self):
        self._check(BertOnlyNSPHead(*[], **{'config': _mock_config(hidden_size=4)}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_009(self):
        self._check(BertOutput(*[], **{'config': _mock_config(intermediate_size=4, hidden_size=4, layer_norm_eps=1, hidden_dropout_prob=0.5)}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_010(self):
        self._check(BertPooler(*[], **{'config': _mock_config(hidden_size=4)}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_011(self):
        self._check(BertSelfOutput(*[], **{'config': _mock_config(hidden_size=4, layer_norm_eps=1, hidden_dropout_prob=0.5)}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_012(self):
        self._check(Conv1D(*[], **{'nf': 4, 'nx': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_013(self):
        self._check(Embeddings(*[], **{'config': _mock_config(vocab_size=4, dim=4, max_position_embeddings=4, sinusoidal_pos_embds=4, dropout=0.5)}), [torch.zeros([4, 4], dtype=torch.int64)], {})

    @_fails_compile()
    def test_014(self):
        self._check(FeedForwardNetwork(*[], **{'input_size': 4, 'hidden_size': 4, 'output_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_015(self):
        self._check(MultiHeadAttention(*[], **{'n_heads': 4, 'dim': 4, 'config': _mock_config(output_attentions=4, attention_dropout=0.5)}), [torch.rand([4, 4, 4]), torch.rand([4, 4])], {})

    @_fails_compile()
    def test_016(self):
        self._check(PoolerStartLogits(*[], **{'config': _mock_config(hidden_size=4)}), [torch.rand([4, 4, 4, 4])], {})

    def test_017(self):
        self._check(PositionwiseFF(*[], **{'d_model': 4, 'd_inner': 4, 'dropout': 0.5}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_018(self):
        self._check(RobertaClassificationHead(*[], **{'config': _mock_config(hidden_size=4, hidden_dropout_prob=0.5, num_labels=4)}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_019(self):
        self._check(RobertaLMHead(*[], **{'config': _mock_config(hidden_size=4, layer_norm_eps=1, vocab_size=4)}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_020(self):
        self._check(SQuADHead(*[], **{'config': _mock_config(start_n_top=4, end_n_top=4, hidden_size=4, layer_norm_eps=1)}), [torch.rand([4, 4, 4])], {})

    def test_021(self):
        self._check(TransformerFFN(*[], **{'in_dim': 4, 'dim_hidden': 4, 'out_dim': 4, 'config': _mock_config(dropout=0.5, gelu_activation=4)}), [torch.rand([4, 4, 4, 4])], {})

