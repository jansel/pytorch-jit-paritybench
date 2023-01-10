import sys
_module = sys.modules[__name__]
del sys
config = _module
data_helper = _module
model = _module
run_cosent = _module
data_helper = _module
model = _module
run_sentence_bert_transformers_reg_loss = _module
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


from torch.utils.data import Dataset


from torch import nn


import re


import random


import numpy as np


from torch.utils.data import DataLoader


import pandas as pd


import time


from sklearn.model_selection import StratifiedKFold


from torch.utils.data import TensorDataset


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.config = BertConfig.from_pretrained('./mengzi_pretrain/config.json')
        self.bert = BertModel.from_pretrained('./mengzi_pretrain/pytorch_model.bin', config=self.config)

    def forward(self, input_ids, attention_mask, encoder_type='fist-last-avg'):
        """
        :param input_ids:
        :param attention_mask:
        :param encoder_type: encoder_type:  "first-last-avg", "last-avg", "cls", "pooler(cls + dense)"
        :return:
        """
        output = self.bert(input_ids, attention_mask, output_hidden_states=True)
        if encoder_type == 'fist-last-avg':
            first = output.hidden_states[1]
            last = output.hidden_states[-1]
            seq_length = first.size(1)
            first_avg = torch.avg_pool1d(first.transpose(1, 2), kernel_size=seq_length).squeeze(-1)
            last_avg = torch.avg_pool1d(last.transpose(1, 2), kernel_size=seq_length).squeeze(-1)
            final_encoding = torch.avg_pool1d(torch.cat([first_avg.unsqueeze(1), last_avg.unsqueeze(1)], dim=1).transpose(1, 2), kernel_size=2).squeeze(-1)
            return final_encoding
        if encoder_type == 'last-avg':
            sequence_output = output.last_hidden_state
            seq_length = sequence_output.size(1)
            final_encoding = torch.avg_pool1d(sequence_output.transpose(1, 2), kernel_size=seq_length).squeeze(-1)
            return final_encoding
        if encoder_type == 'cls':
            sequence_output = output.last_hidden_state
            cls = sequence_output[:, 0]
            return cls
        if encoder_type == 'pooler':
            pooler_output = output.pooler_output
            return pooler_output


class SentenceBert(nn.Module):

    def __init__(self):
        super(SentenceBert, self).__init__()
        self.config = BertConfig.from_pretrained('../mengzi_pretrain/config.json')
        self.model = BertModel.from_pretrained('../mengzi_pretrain/pytorch_model.bin', config=self.config)

    def get_embedding_vec(self, output, mask):
        token_embedding = output[0]
        input_mask_expanded = mask.unsqueeze(-1).expand(token_embedding.size()).float()
        sum_embedding = torch.sum(token_embedding * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-09)
        return sum_embedding / sum_mask

    def forward(self, s1_input_ids, s2_input_ids):
        s1_mask = torch.ne(s1_input_ids, 0)
        s2_mask = torch.ne(s2_input_ids, 0)
        s1_output = self.model(input_ids=s1_input_ids, attention_mask=s1_mask)
        s2_output = self.model(input_ids=s2_input_ids, attention_mask=s2_mask)
        s1_vec = s1_output[1]
        s2_vec = s2_output[1]
        return s1_vec, s2_vec

