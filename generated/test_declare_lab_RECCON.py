import sys
_module = sys.modules[__name__]
del sys
eval_classification = _module
eval_qa = _module
evaluate_squad = _module
simpletransformers = _module
classification = _module
classification_model = _module
classification_utils = _module
multi_label_classification_model = _module
multi_modal_classification_model = _module
transformer_models = _module
albert_model = _module
bert_model = _module
camembert_model = _module
distilbert_model = _module
electra_model = _module
flaubert_model = _module
layoutlm_model = _module
mmbt_model = _module
roberta_model = _module
xlm_model = _module
xlm_roberta_model = _module
xlnet_model = _module
config = _module
global_args = _module
model_args = _module
utils = _module
conv_ai = _module
conv_ai_model = _module
conv_ai_utils = _module
custom_models = _module
models = _module
experimental = _module
classification_model = _module
multi_label_classification_model = _module
albert_model = _module
bert_model = _module
camembert_model = _module
distilbert_model = _module
roberta_model = _module
xlm_model = _module
xlnet_model = _module
language_generation = _module
language_generation_model = _module
language_generation_utils = _module
language_modeling = _module
language_modeling_model = _module
language_modeling_utils = _module
language_representation = _module
representation_model = _module
gpt2_model = _module
model = _module
ner = _module
ner_model = _module
ner_utils = _module
question_answering = _module
question_answering_model = _module
question_answering_utils = _module
seq2seq = _module
seq2seq_model = _module
seq2seq_utils = _module
streamlit = _module
classification_view = _module
ner_view = _module
qa_view = _module
simple_view = _module
streamlit_utils = _module
t5 = _module
run_simple_transformers_streamlit_app = _module
t5_model = _module
t5_utils = _module
train_classification = _module
train_qa = _module

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


import logging


import math


import random


import warnings


import numpy as np


import pandas as pd


import torch


from scipy.stats import mode


from scipy.stats import pearsonr


from sklearn.metrics import confusion_matrix


from sklearn.metrics import label_ranking_average_precision_score


from sklearn.metrics import matthews_corrcoef


from sklearn.metrics import mean_squared_error


from torch.utils.data import DataLoader


from torch.utils.data import RandomSampler


from torch.utils.data import SequentialSampler


from torch.utils.data import TensorDataset


from torch.utils.data.distributed import DistributedSampler


from collections import Counter


import torch.nn as nn


from scipy.stats import spearmanr


from sklearn.metrics import f1_score


from torch.utils.data import Dataset


from torch.nn import CrossEntropyLoss


from torch.nn import MSELoss


from collections import defaultdict


from itertools import chain


import torch.nn.functional as F


from torch import nn


from torch.nn import BCEWithLogitsLoss


from torch.nn.utils.rnn import pad_sequence


from typing import Dict


from typing import List


from typing import Tuple


from functools import partial


from torch.functional import split


import collections


import re


import string


from torch.cuda import is_available


POOLING_BREAKDOWN = {(1): (1, 1), (2): (2, 1), (3): (3, 1), (4): (2, 2), (5): (5, 1), (6): (3, 2), (7): (7, 1), (8): (4, 2), (9): (3, 3)}


class ImageEncoder(nn.Module):

    def __init__(self, args):
        super().__init__()
        model = torchvision.models.resnet152(pretrained=True)
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)
        self.pool = nn.AdaptiveAvgPool2d(POOLING_BREAKDOWN[args.num_image_embeds])

    def forward(self, x):
        out = self.pool(self.model(x))
        out = torch.flatten(out, start_dim=2)
        out = out.transpose(1, 2).contiguous()
        return out


class MMBTForClassification(nn.Module):
    """
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    Examples::
        # For example purposes. Not runnable.
        transformer = BertModel.from_pretrained('bert-base-uncased')
        encoder = ImageEncoder(args)
        model = MMBTForClassification(config, transformer, encoder)
        outputs = model(input_modal, input_ids, labels=labels)
        loss, logits = outputs[:2]
    """

    def __init__(self, config, transformer, encoder):
        super().__init__()
        self.num_labels = config.num_labels
        self.mmbt = MMBTModel(config, transformer, encoder)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_modal, input_ids=None, modal_start_tokens=None, modal_end_tokens=None, attention_mask=None, token_type_ids=None, modal_token_type_ids=None, position_ids=None, modal_position_ids=None, head_mask=None, inputs_embeds=None, labels=None):
        outputs = self.mmbt(input_modal=input_modal, input_ids=input_ids, modal_start_tokens=modal_start_tokens, modal_end_tokens=modal_end_tokens, attention_mask=attention_mask, token_type_ids=token_type_ids, modal_token_type_ids=modal_token_type_ids, position_ids=position_ids, modal_position_ids=modal_position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        outputs = (logits,) + outputs[2:]
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs


class ElectraPooler(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ElectraPooler,
     lambda: ([], {'config': _mock_config(hidden_size=4)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ImageEncoder,
     lambda: ([], {'args': _mock_config(num_image_embeds=4)}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
]

class Test_declare_lab_RECCON(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

