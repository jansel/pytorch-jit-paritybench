import sys
_module = sys.modules[__name__]
del sys
bert_bilstm_crf_ner = _module
bert_base_model = _module
bert_ner_model = _module
config = _module
dataset = _module
main = _module
preprocess = _module
utils = _module
commonUtils = _module
cutSentences = _module
decodeUtils = _module
metricsUtils = _module
trainUtils = _module
bert_re = _module
bert_config = _module
data_into_train_test = _module
relation_bar_chart = _module
data_loader = _module
dataset = _module
main = _module
models = _module
utils = _module
process = _module
get_result = _module
re_process = _module

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


import torch.nn as nn


import torch


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


import logging


import numpy as np


from torch.utils.data import RandomSampler


import random


import time


from sklearn.metrics import accuracy_score


from sklearn.metrics import f1_score


from sklearn.metrics import classification_report


from torch.nn.utils.rnn import pad_sequence


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


class BertNerModel(BaseModel):

    def __init__(self, args, **kwargs):
        super(BertNerModel, self).__init__(bert_dir=args.bert_dir, dropout_prob=args.dropout_prob)
        self.args = args
        self.num_layers = args.num_layers
        self.lstm_hidden = args.lstm_hidden
        gpu_ids = args.gpu_ids.split(',')
        device = torch.device('cpu' if gpu_ids[0] == '-1' else 'cuda:' + gpu_ids[0])
        self.device = device
        out_dims = self.bert_config.hidden_size
        if args.use_lstm == 'True':
            self.lstm = nn.LSTM(out_dims, args.lstm_hidden, args.num_layers, bidirectional=True, batch_first=True, dropout=args.dropout)
            self.linear = nn.Linear(args.lstm_hidden * 2, args.num_tags)
            self.criterion = nn.CrossEntropyLoss()
            init_blocks = [self.linear]
            self._init_weights(init_blocks, initializer_range=self.bert_config.initializer_range)
        else:
            mid_linear_dims = kwargs.pop('mid_linear_dims', 256)
            self.mid_linear = nn.Sequential(nn.Linear(out_dims, mid_linear_dims), nn.ReLU(), nn.Dropout(args.dropout))
            out_dims = mid_linear_dims
            self.classifier = nn.Linear(out_dims, args.num_tags)
            self.criterion = nn.CrossEntropyLoss()
            init_blocks = [self.mid_linear, self.classifier]
            self._init_weights(init_blocks, initializer_range=self.bert_config.initializer_range)
        if args.use_crf == 'True':
            self.crf = CRF(args.num_tags, batch_first=True)

    def init_hidden(self, batch_size):
        h0 = torch.randn(2 * self.num_layers, batch_size, self.lstm_hidden, requires_grad=True)
        c0 = torch.randn(2 * self.num_layers, batch_size, self.lstm_hidden, requires_grad=True)
        return h0, c0

    def forward(self, token_ids, attention_masks, token_type_ids, labels):
        bert_outputs = self.bert_module(input_ids=token_ids, attention_mask=attention_masks, token_type_ids=token_type_ids)
        seq_out = bert_outputs[0]
        batch_size = seq_out.size(0)
        if self.args.use_lstm == 'True':
            hidden = self.init_hidden(batch_size)
            seq_out, (hn, _) = self.lstm(seq_out, hidden)
            seq_out = seq_out.contiguous().view(-1, self.lstm_hidden * 2)
            seq_out = self.linear(seq_out)
            seq_out = seq_out.contiguous().view(batch_size, self.args.max_seq_len, -1)
        else:
            seq_out = self.mid_linear(seq_out)
            seq_out = self.classifier(seq_out)
        if self.args.use_crf == 'True':
            logits = self.crf.decode(seq_out, mask=attention_masks)
            if labels is None:
                return logits
            loss = -self.crf(seq_out, labels, mask=attention_masks, reduction='mean')
            outputs = (loss,) + (logits,)
            return outputs
        else:
            logits = seq_out
            if labels is None:
                return logits
            active_loss = attention_masks.view(-1) == 1
            active_logits = logits.view(-1, logits.size()[2])[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = self.criterion(active_logits, active_labels)
            outputs = (loss,) + (logits,)
            return outputs


class BertForRelationExtraction(nn.Module):

    def __init__(self, args):
        super(BertForRelationExtraction, self).__init__()
        self.bert = BertModel.from_pretrained(args.bert_dir)
        self.bert_config = self.bert.config
        out_dims = self.bert_config.hidden_size
        self.dropout = nn.Dropout(args.dropout_prob)
        self.linear = nn.Linear(out_dims * 4, args.num_tags)

    def forward(self, token_ids, attention_masks, token_type_ids, ids):
        bert_outputs = self.bert(input_ids=token_ids, attention_mask=attention_masks, token_type_ids=token_type_ids)
        seq_out = bert_outputs[0]
        batch_size = seq_out.size(0)
        seq_ent = torch.cat([torch.index_select(seq_out[i, :, :], 0, ids[i, :].long()).unsqueeze(0) for i in range(batch_size)], 0)
        seq_ent = self.dropout(seq_ent)
        seq_ent = seq_ent.view(batch_size, -1)
        seq_ent = self.linear(seq_ent)
        return seq_ent

