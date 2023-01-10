import sys
_module = sys.modules[__name__]
del sys
competition_predict = _module
convert_test_data = _module
main = _module
convert_raw_data = _module
processor = _module
attack_train_utils = _module
dataset_utils = _module
evaluator = _module
functions_utils = _module
model_utils = _module
options = _module
trainer = _module

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


from collections import defaultdict


import time


import logging


from torch.utils.data import DataLoader


from sklearn.model_selection import KFold


import torch.nn as nn


from torch.utils.data import Dataset


import numpy as np


import copy


import random


import math


from itertools import repeat


from torch.cuda.amp import autocast as ac


from torch.utils.data import RandomSampler


class LabelSmoothingCrossEntropy(nn.Module):

    def __init__(self, eps=0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, output, target):
        c = output.size()[-1]
        log_pred = torch.log_softmax(output, dim=-1)
        if self.reduction == 'sum':
            loss = -log_pred.sum()
        else:
            loss = -log_pred.sum(dim=-1)
            if self.reduction == 'mean':
                loss = loss.mean()
        return loss * self.eps / c + (1 - self.eps) * torch.nn.functional.nll_loss(log_pred, target, reduction=self.reduction, ignore_index=self.ignore_index)


class FocalLoss(nn.Module):
    """Multi-class Focal loss implementation"""

    def __init__(self, gamma=2, weight=None, reduction='mean', ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        log_pt = torch.log_softmax(input, dim=1)
        pt = torch.exp(log_pt)
        log_pt = (1 - pt) ** self.gamma * log_pt
        loss = torch.nn.functional.nll_loss(log_pt, target, self.weight, reduction=self.reduction, ignore_index=self.ignore_index)
        return loss


class SpatialDropout(nn.Module):
    """
    对字级别的向量进行丢弃
    """

    def __init__(self, drop_prob):
        super(SpatialDropout, self).__init__()
        self.drop_prob = drop_prob

    @staticmethod
    def _make_noise(input):
        return input.new().resize_(input.size(0), *repeat(1, input.dim() - 2), input.size(2))

    def forward(self, inputs):
        output = inputs.clone()
        if not self.training or self.drop_prob == 0:
            return inputs
        else:
            noise = self._make_noise(inputs)
            if self.drop_prob == 1:
                noise.fill_(0)
            else:
                noise.bernoulli_(1 - self.drop_prob).div_(1 - self.drop_prob)
            noise = noise.expand_as(inputs)
            output.mul_(noise)
        return output


class ConditionalLayerNorm(nn.Module):

    def __init__(self, normalized_shape, cond_shape, eps=1e-12):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.Tensor(normalized_shape))
        self.bias = nn.Parameter(torch.Tensor(normalized_shape))
        self.weight_dense = nn.Linear(cond_shape, normalized_shape, bias=False)
        self.bias_dense = nn.Linear(cond_shape, normalized_shape, bias=False)
        self.reset_weight_and_bias()

    def reset_weight_and_bias(self):
        """
        此处初始化的作用是在训练开始阶段不让 conditional layer norm 起作用
        """
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
        nn.init.zeros_(self.weight_dense.weight)
        nn.init.zeros_(self.bias_dense.weight)

    def forward(self, inputs, cond=None):
        assert cond is not None, 'Conditional tensor need to input when use conditional layer norm'
        cond = torch.unsqueeze(cond, 1)
        weight = self.weight_dense(cond) + self.weight
        bias = self.bias_dense(cond) + self.bias
        mean = torch.mean(inputs, dim=-1, keepdim=True)
        outputs = inputs - mean
        variance = torch.mean(outputs ** 2, dim=-1, keepdim=True)
        std = torch.sqrt(variance + self.eps)
        outputs = outputs / std
        outputs = outputs * weight + bias
        return outputs


class BaseModel(nn.Module):

    def __init__(self, bert_dir, dropout_prob):
        super(BaseModel, self).__init__()
        config_path = os.path.join(bert_dir, 'config.json')
        assert os.path.exists(bert_dir) and os.path.exists(config_path), 'pretrained bert file does not exist'
        self.bert_module = BertModel.from_pretrained(bert_dir, output_hidden_states=True, hidden_dropout_prob=dropout_prob)
        self.bert_config = self.bert_module.config

    @staticmethod
    def _init_weights(blocks, **kwargs):
        """
        参数初始化，将 Linear / Embedding / LayerNorm 与 Bert 进行一样的初始化
        """
        for block in blocks:
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Embedding):
                    nn.init.normal_(module.weight, mean=0, std=kwargs.pop('initializer_range', 0.02))
                elif isinstance(module, nn.LayerNorm):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)


class CRFModel(BaseModel):

    def __init__(self, bert_dir, num_tags, dropout_prob=0.1, **kwargs):
        super(CRFModel, self).__init__(bert_dir=bert_dir, dropout_prob=dropout_prob)
        out_dims = self.bert_config.hidden_size
        mid_linear_dims = kwargs.pop('mid_linear_dims', 128)
        self.mid_linear = nn.Sequential(nn.Linear(out_dims, mid_linear_dims), nn.ReLU(), nn.Dropout(dropout_prob))
        out_dims = mid_linear_dims
        self.classifier = nn.Linear(out_dims, num_tags)
        self.loss_weight = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.loss_weight.data.fill_(-0.2)
        self.crf_module = CRF(num_tags=num_tags, batch_first=True)
        init_blocks = [self.mid_linear, self.classifier]
        self._init_weights(init_blocks, initializer_range=self.bert_config.initializer_range)

    def forward(self, token_ids, attention_masks, token_type_ids, labels=None, pseudo=None):
        bert_outputs = self.bert_module(input_ids=token_ids, attention_mask=attention_masks, token_type_ids=token_type_ids)
        seq_out = bert_outputs[0]
        seq_out = self.mid_linear(seq_out)
        emissions = self.classifier(seq_out)
        if labels is not None:
            if pseudo is not None:
                tokens_loss = -1.0 * self.crf_module(emissions=emissions, tags=labels.long(), mask=attention_masks.byte(), reduction='none')
                pseudo_nums = pseudo.sum().item()
                total_nums = token_ids.shape[0]
                rate = torch.sigmoid(self.loss_weight)
                if pseudo_nums == 0:
                    loss_0 = tokens_loss.mean()
                    loss_1 = (rate * pseudo * tokens_loss).sum()
                else:
                    if total_nums == pseudo_nums:
                        loss_0 = 0
                    else:
                        loss_0 = ((1 - rate) * (1 - pseudo) * tokens_loss).sum() / (total_nums - pseudo_nums)
                    loss_1 = (rate * pseudo * tokens_loss).sum() / pseudo_nums
                tokens_loss = loss_0 + loss_1
            else:
                tokens_loss = -1.0 * self.crf_module(emissions=emissions, tags=labels.long(), mask=attention_masks.byte(), reduction='mean')
            out = tokens_loss,
        else:
            tokens_out = self.crf_module.decode(emissions=emissions, mask=attention_masks.byte())
            out = tokens_out, emissions
        return out


class SpanModel(BaseModel):

    def __init__(self, bert_dir, num_tags, dropout_prob=0.1, loss_type='ce', **kwargs):
        """
        tag the subject and object corresponding to the predicate
        :param loss_type: train loss type in ['ce', 'ls_ce', 'focal']
        """
        super(SpanModel, self).__init__(bert_dir, dropout_prob=dropout_prob)
        out_dims = self.bert_config.hidden_size
        mid_linear_dims = kwargs.pop('mid_linear_dims', 128)
        self.num_tags = num_tags
        self.mid_linear = nn.Sequential(nn.Linear(out_dims, mid_linear_dims), nn.ReLU(), nn.Dropout(dropout_prob))
        out_dims = mid_linear_dims
        self.start_fc = nn.Linear(out_dims, num_tags)
        self.end_fc = nn.Linear(out_dims, num_tags)
        reduction = 'none'
        if loss_type == 'ce':
            self.criterion = nn.CrossEntropyLoss(reduction=reduction)
        elif loss_type == 'ls_ce':
            self.criterion = LabelSmoothingCrossEntropy(reduction=reduction)
        else:
            self.criterion = FocalLoss(reduction=reduction)
        self.loss_weight = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.loss_weight.data.fill_(-0.2)
        init_blocks = [self.mid_linear, self.start_fc, self.end_fc]
        self._init_weights(init_blocks)

    def forward(self, token_ids, attention_masks, token_type_ids, start_ids=None, end_ids=None, pseudo=None):
        bert_outputs = self.bert_module(input_ids=token_ids, attention_mask=attention_masks, token_type_ids=token_type_ids)
        seq_out = bert_outputs[0]
        seq_out = self.mid_linear(seq_out)
        start_logits = self.start_fc(seq_out)
        end_logits = self.end_fc(seq_out)
        out = start_logits, end_logits
        if start_ids is not None and end_ids is not None and self.training:
            start_logits = start_logits.view(-1, self.num_tags)
            end_logits = end_logits.view(-1, self.num_tags)
            active_loss = attention_masks.view(-1) == 1
            active_start_logits = start_logits[active_loss]
            active_end_logits = end_logits[active_loss]
            active_start_labels = start_ids.view(-1)[active_loss]
            active_end_labels = end_ids.view(-1)[active_loss]
            if pseudo is not None:
                start_loss = self.criterion(start_logits, start_ids.view(-1)).view(-1, 512).mean(dim=-1)
                end_loss = self.criterion(end_logits, end_ids.view(-1)).view(-1, 512).mean(dim=-1)
                pseudo_nums = pseudo.sum().item()
                total_nums = token_ids.shape[0]
                rate = torch.sigmoid(self.loss_weight)
                if pseudo_nums == 0:
                    start_loss = start_loss.mean()
                    end_loss = end_loss.mean()
                elif total_nums == pseudo_nums:
                    start_loss = (rate * pseudo * start_loss).sum() / pseudo_nums
                    end_loss = (rate * pseudo * end_loss).sum() / pseudo_nums
                else:
                    start_loss = (rate * pseudo * start_loss).sum() / pseudo_nums + ((1 - rate) * (1 - pseudo) * start_loss).sum() / (total_nums - pseudo_nums)
                    end_loss = (rate * pseudo * end_loss).sum() / pseudo_nums + ((1 - rate) * (1 - pseudo) * end_loss).sum() / (total_nums - pseudo_nums)
            else:
                start_loss = self.criterion(active_start_logits, active_start_labels)
                end_loss = self.criterion(active_end_logits, active_end_labels)
            loss = start_loss + end_loss
            out = (loss,) + out
        return out


class MRCModel(BaseModel):

    def __init__(self, bert_dir, dropout_prob=0.1, use_type_embed=False, loss_type='ce', **kwargs):
        """
        tag the subject and object corresponding to the predicate
        :param use_type_embed: type embedding for the sentence
        :param loss_type: train loss type in ['ce', 'ls_ce', 'focal']
        """
        super(MRCModel, self).__init__(bert_dir, dropout_prob=dropout_prob)
        self.use_type_embed = use_type_embed
        self.use_smooth = loss_type
        out_dims = self.bert_config.hidden_size
        if self.use_type_embed:
            embed_dims = kwargs.pop('predicate_embed_dims', self.bert_config.hidden_size)
            self.type_embedding = nn.Embedding(13, embed_dims)
            self.conditional_layer_norm = ConditionalLayerNorm(out_dims, embed_dims, eps=self.bert_config.layer_norm_eps)
        mid_linear_dims = kwargs.pop('mid_linear_dims', 128)
        self.mid_linear = nn.Sequential(nn.Linear(out_dims, mid_linear_dims), nn.ReLU(), nn.Dropout(dropout_prob))
        out_dims = mid_linear_dims
        self.start_fc = nn.Linear(out_dims, 2)
        self.end_fc = nn.Linear(out_dims, 2)
        reduction = 'none'
        if loss_type == 'ce':
            self.criterion = nn.CrossEntropyLoss(reduction=reduction)
        elif loss_type == 'ls_ce':
            self.criterion = LabelSmoothingCrossEntropy(reduction=reduction)
        else:
            self.criterion = FocalLoss(reduction=reduction)
        self.loss_weight = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.loss_weight.data.fill_(-0.2)
        init_blocks = [self.mid_linear, self.start_fc, self.end_fc]
        if self.use_type_embed:
            init_blocks.append(self.type_embedding)
        self._init_weights(init_blocks)

    def forward(self, token_ids, attention_masks, token_type_ids, ent_type=None, start_ids=None, end_ids=None, pseudo=None):
        bert_outputs = self.bert_module(input_ids=token_ids, attention_mask=attention_masks, token_type_ids=token_type_ids)
        seq_out = bert_outputs[0]
        if self.use_type_embed:
            assert ent_type is not None, 'Using predicate embedding, predicate should be implemented'
            predicate_feature = self.type_embedding(ent_type)
            seq_out = self.conditional_layer_norm(seq_out, predicate_feature)
        seq_out = self.mid_linear(seq_out)
        start_logits = self.start_fc(seq_out)
        end_logits = self.end_fc(seq_out)
        out = start_logits, end_logits
        if start_ids is not None and end_ids is not None:
            start_logits = start_logits.view(-1, 2)
            end_logits = end_logits.view(-1, 2)
            active_loss = token_type_ids.view(-1) == 1
            active_start_logits = start_logits[active_loss]
            active_end_logits = end_logits[active_loss]
            active_start_labels = start_ids.view(-1)[active_loss]
            active_end_labels = end_ids.view(-1)[active_loss]
            if pseudo is not None:
                start_loss = self.criterion(start_logits, start_ids.view(-1)).view(-1, 512).mean(dim=-1)
                end_loss = self.criterion(end_logits, end_ids.view(-1)).view(-1, 512).mean(dim=-1)
                pseudo_nums = pseudo.sum().item()
                total_nums = token_ids.shape[0]
                rate = torch.sigmoid(self.loss_weight)
                if pseudo_nums == 0:
                    start_loss = start_loss.mean()
                    end_loss = end_loss.mean()
                elif total_nums == pseudo_nums:
                    start_loss = (rate * pseudo * start_loss).sum() / pseudo_nums
                    end_loss = (rate * pseudo * end_loss).sum() / pseudo_nums
                else:
                    start_loss = (rate * pseudo * start_loss).sum() / pseudo_nums + ((1 - rate) * (1 - pseudo) * start_loss).sum() / (total_nums - pseudo_nums)
                    end_loss = (rate * pseudo * end_loss).sum() / pseudo_nums + ((1 - rate) * (1 - pseudo) * end_loss).sum() / (total_nums - pseudo_nums)
            else:
                start_loss = self.criterion(active_start_logits, active_start_labels)
                end_loss = self.criterion(active_end_logits, active_end_labels)
            loss = start_loss + end_loss
            out = (loss,) + out
        return out


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (SpatialDropout,
     lambda: ([], {'drop_prob': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_z814081807_DeepNER(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

