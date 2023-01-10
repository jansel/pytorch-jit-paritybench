import sys
_module = sys.modules[__name__]
del sys
medtype_serving = _module
client = _module
_py2_var = _module
_py3_var = _module
setup = _module
example = _module
server = _module
server = _module
benchmark = _module
cli = _module
entity_linkers = _module
helper = _module
http = _module
medtype = _module
dataloader = _module
models = _module
zmq_decor = _module
setup = _module
dataloader = _module
dump_bert_model = _module
dump_linkers_output = _module
eval_models = _module
helper = _module
medtype = _module
models = _module
neleval = _module
analyze = _module
annotation = _module
brat = _module
configs = _module
coref_metrics = _module
document = _module
evaluate = _module
import_ = _module
interact = _module
munkres = _module
prepare = _module
significance = _module
summary = _module
tac = _module
test = _module
tests = _module
test_coref_metrics = _module
util = _module
utils = _module
weak = _module
merge_evaluations = _module
raw_to_processed_data = _module

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


import random


import time


import numpy as np


from itertools import chain


from collections import OrderedDict


from collections import defaultdict as ddict


import torch


import torch.nn as nn


from torch.utils.data import Dataset


import itertools


import logging


import logging.config


import warnings


from collections import Counter


from torch.utils.data import DataLoader


from sklearn.metrics import average_precision_score


from torch.nn import functional as F


class BertPlain(nn.Module):
    """
	MedType Bert-based Architecture. 
	----------
	params:        	Hyperparameters of the model
	num_tokens:   	Number of tokens in BERT model
	num_labels:	Total number of classes
	
	Returns
	-------
	The MedType model instance
		
	"""

    def __init__(self, params, num_tokens, num_labels):
        super().__init__()
        self.p = params
        self.bert = BertModel.from_pretrained(self.p.bert_model)
        self.bert.resize_token_embeddings(num_tokens)
        self.dropout = nn.Dropout(self.p.drop)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, mention_pos_idx, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        tok_embed = outputs[0]
        bsz, mtok, dim = tok_embed.shape
        tok_embed_flat = tok_embed.reshape(-1, dim)
        men_idx = torch.arange(bsz) * mtok + mention_pos_idx
        men_embed = tok_embed_flat[men_idx]
        pooled_output = self.dropout(men_embed)
        logits = self.classifier(pooled_output)
        loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        return loss, logits


class BertCombined(nn.Module):
    """
	MedType Bert-based Architecture which combines benefits of both WikiMed and PubMedDS dataset. 
	----------
	params:        	Hyperparameters of the model
	num_tokens:   	Number of tokens in BERT model
	num_labels:	Total number of classes
	
	Returns
	-------
	The MedType model instance
		
	"""

    def __init__(self, params, num_tokens, num_labels):
        super().__init__()
        self.p = params
        self.bert_wiki = BertModel.from_pretrained('models/bert_dumps/{}'.format(self.p.wiki_model))
        self.bert_pubmed = BertModel.from_pretrained('models/bert_dumps/{}'.format(self.p.pubmed_model))
        self.dropout = nn.Dropout(self.p.drop)
        if self.p.comb_opn == 'concat':
            class_in = self.bert_wiki.config.hidden_size * 2
        else:
            class_in = self.bert_wiki.config.hidden_size
        self.classifier = nn.Linear(class_in, num_labels)

    def forward(self, input_ids, attention_mask, mention_pos_idx, labels=None):
        out_wiki = self.bert_wiki(input_ids=input_ids, attention_mask=attention_mask)
        out_pubmed = self.bert_pubmed(input_ids=input_ids, attention_mask=attention_mask)
        tok_embed = out_wiki[0]
        bsz, mtok, dim = tok_embed.shape
        tok_embed_flat = tok_embed.reshape(-1, dim)
        men_idx = torch.arange(bsz) * mtok + mention_pos_idx
        wiki_embed = tok_embed_flat[men_idx]
        tok_embed = out_pubmed[0]
        bsz, mtok, dim = tok_embed.shape
        tok_embed_flat = tok_embed.reshape(-1, dim)
        men_idx = torch.arange(bsz) * mtok + mention_pos_idx
        pubmed_embed = tok_embed_flat[men_idx]
        if self.p.comb_opn == 'concat':
            pooled_output = torch.cat([wiki_embed, pubmed_embed], dim=1)
        elif self.p.comb_opn == 'add':
            pooled_output = wiki_embed + pubmed_embed
        else:
            raise NotImplementedError
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        return loss, logits

