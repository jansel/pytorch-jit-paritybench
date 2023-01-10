import sys
_module = sys.modules[__name__]
del sys
api = _module
api_utils = _module
config = _module
process = _module
sample = _module
data = _module
ann_searcher = _module
process_pone = _module
transform_test = _module
expand_negative_samples = _module
process_data = _module
langconv = _module
utils = _module
zh_wiki = _module
squeeze_test = _module
conceptnet = _module
eda = _module
fluency_perturbation = _module
keywords = _module
negative_sampler = _module
pmi = _module
read_dataset_eda = _module
word_graph = _module
transform_retrieval = _module
zh50w_process = _module
dataloader = _module
dataset_init = _module
eval = _module
header = _module
main = _module
metrics = _module
bleu = _module
bleu_scorer = _module
ir_metric = _module
metric = _module
models = _module
base = _module
bert_mc = _module
bert_na = _module
bert_nli = _module
bert_retrieval = _module
bert_retrieval_multi = _module
biencoder = _module
decoupling_gpt2gan = _module
dialogpt = _module
gpt2 = _module
gpt2gan = _module
gpt2gan_v2 = _module
gpt2gan_v2_bak = _module
gpt2retrieval = _module
gpt2v2 = _module
gpt2v2rl = _module
hash = _module
header = _module
kwgpt2 = _module
lccc_gpt2 = _module
model_utils = _module
pf_gpt2 = _module
pone = _module
retrieval = _module
seq2seq = _module
seq2seq_trs = _module
test = _module
uni = _module
when2talk = _module
multiview = _module
bert_multiview = _module
bertmc = _module
coherence = _module
diversity = _module
fluency = _module
header = _module
lccc_lm = _module
logic = _module
mmi = _module
multiview = _module
nli = _module
topic = _module
self_play = _module
collate_fn = _module
embedding = _module
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


import random


import numpy as np


import torch


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from torch.nn.utils import clip_grad_norm_


from torch.nn import DataParallel


from torch.optim import lr_scheduler


from torch.utils.tensorboard import SummaryWriter


import torch.optim as optim


import torch.nn as nn


import torch.nn.functional as F


from torchtext import vocab


from collections import Counter


import re


import math


from itertools import chain


import time


import logging


from copy import deepcopy


from torch.nn.utils.rnn import pad_sequence


from torch.nn.parallel.data_parallel import DataParallel


from torch.nn.parallel.parallel_apply import parallel_apply


from torch.nn.parallel._functions import Scatter


from queue import *


from torch.nn.parallel import DistributedDataParallel


import torch.nn.init as init


from collections import OrderedDict


from torch.distributions import MultivariateNormal


from scipy.stats import pearsonr


from scipy.stats import spearmanr


from sklearn.metrics import label_ranking_average_precision_score


from torch.optim.lr_scheduler import _LRScheduler


from math import *


from sklearn.feature_extraction.text import CountVectorizer


def generate_attention_mask_mc(inpt_ids):
    """
    inpt_ids: [B, N, S]
    """
    bsz = inpt_ids.size(0)
    attn_mask = torch.zeros_like(inpt_ids)
    for i in range(bsz):
        not_masked_token_idx = inpt_ids[i].nonzero().transpose(0, 1).tolist()
        attn_mask[i][not_masked_token_idx] = 1
    return attn_mask


class BERTMC(nn.Module):

    def __init__(self, model='bert-base-chinese'):
        super(BERTMC, self).__init__()
        self.model = BertForMultipleChoice.from_pretrained(model)

    def forward(self, inpt):
        attn_mask = generate_attention_mask_mc(inpt)
        output = self.model(input_ids=inpt, attention_mask=attn_mask)
        logits = output[0]
        return logits


def generate_attention_mask(inpt_ids):
    """
    generate the corresponding attention mask according to the `input_ids`, which will 
    be fed into the model (BERT or GPT2)
    :inpt_ids: [batch, seq]
    
    return :attn_mask: [batch, seq]; 1 for not masked and 0 for masked tokens
    """
    attn_mask = torch.zeros_like(inpt_ids)
    not_masked_token_idx = inpt_ids.nonzero().transpose(0, 1).tolist()
    attn_mask[not_masked_token_idx] = 1
    return attn_mask


class BERTMCFusion(nn.Module):

    def __init__(self, model='bert-base-chinese', dropout=0.1, num_layers=1, nhead=1):
        super(BERTMCFusion, self).__init__()
        self.bert = BertModel.from_pretrained(model)
        encoder_layer = nn.TransformerEncoderLayer(768, nhead=nhead, dim_feedforward=768, dropout=dropout)
        self.fusion = nn.TransformerEncoder(encoder_layer, num_layers)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(768 * 2, 1)

    def forward(self, inpt):
        num_choices = inpt.shape[1]
        inpt = inpt.view(-1, inpt.size(-1))
        attn_mask = generate_attention_mask(inpt)
        output = self.bert(input_ids=inpt, attention_mask=attn_mask)
        logits = output[1]
        logits = torch.stack(logits.split(num_choices))
        logits_ = self.fusion(logits.transpose(0, 1)).transpose(0, 1)
        opt = torch.cat([logits, logits_], dim=2)
        logits = self.classifier(self.dropout(opt)).squeeze(-1)
        return logits


class BERTNA(nn.Module):

    def __init__(self, config_path, max_size=10):
        super(BERTNA, self).__init__()
        self.model = BertModel.from_pretrained(config_path)
        hidden_size = self.model.config.get_config_dict('bert-base-chinese')[0]['hidden_size']
        vocab_size = self.model.config.get_config_dict('bert-base-chinese')[0]['vocab_size']
        self.head = nn.Linear(hidden_size, vocab_size)
        self.max_size = max_size

    def forward(self, inpt_ids):
        """inpt_ids: [B, S]; [MASK] tokens represent the predicted tokens"""
        outputs = self.model(input_ids=inpt_ids)[0]
        outputs = self.head(outputs)
        return outputs

    @torch.no_grad()
    def predict(self, inpt_ids):
        outputs = self.model(input_ids=inpt_ids)[0]
        outputs = F.softmax(self.head(outputs), dim=-1)
        outputs = outputs[:, 1:self.max_size + 1, :]
        batch_size, vocab_size = outputs.size(0), outputs.size(-1)
        outputs = outputs.reshape(-1, vocab_size)
        outputs = torch.multinomial(outputs, 1).squeeze(-1)
        outputs = outputs.view(batch_size, self.max_size)
        return outputs


class BERTNLI(nn.Module):

    def __init__(self):
        super(BERTNLI, self).__init__()
        self.model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=3)

    def forward(self, inpt):
        """
        inpt: [batch, seq]
        """
        attn_mask = generate_attention_mask(inpt)
        output = self.model(input_ids=inpt, attention_mask=attn_mask)
        logits = output[0]
        return logits


class BERTRetrieval(nn.Module):

    def __init__(self, model='bert-base-chinese'):
        super(BERTRetrieval, self).__init__()
        self.model = BertForSequenceClassification.from_pretrained(model, num_labels=2)

    def forward(self, inpt, token_type_ids, attn_mask):
        output = self.model(input_ids=inpt, attention_mask=attn_mask, token_type_ids=token_type_ids)
        logits = output[0]
        return logits


class TopicPrediction(nn.Module):
    'bert model as the backbone for semantic embedding;\n    follow this work: 2020-COLING Towards Topic-Guided Conversational Recommender System.\n    P_{topic}(t)=softmax(e_t^T\\cdot \rm{MLP([r^{(1)}; r^{(2)}])}), where r^{(1)} represents the concatenation of the dialog history and the target topic word ([SEP] separating); r^{(2)} represents the concatenation of the historical topic sequence'

    def __init__(self, vocab_size, dropout=0.3, model='bert-base-chinese'):
        super(TopicPrediction, self).__init__()
        self.bert = BertModel.from_pretrained(model)
        self.predictor = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(768, vocab_size))

    def forward(self, inpt, attn_mask):
        inpt_embd = self.bert(input_ids=inpt, attention_mask=attn_mask)[0][:, 0, :]
        rest = self.predictor(inpt_embd)
        return rest


class BERTMULTIVIEW(nn.Module):
    """
    Multi-view for automatic evaluation, retrieval-based dialog system and generation rerank:
    1. Fluency
    2. Coherence
    3. Diversity
    4. Naturalness
    5. Relatedness
    """

    def __init__(self):
        super(BERTMULTIVIEW, self).__init__()
        self.model = BertModel.from_pretrained('bert-base-chinese')
        self.fluency_m = nn.Linear(768, 2)
        self.coherence_m = nn.Linear(768, 2)
        self.diversity_m = nn.Linear(768, 2)
        self.naturalness_m = nn.Linear(768, 2)
        self.relatedness_m = nn.Linear(768, 2)
        self.head = nn.Linear(2 * 5, 2)

    def forward(self, inpt, aspect='coherence'):
        if aspect != 'overall':
            with torch.no_grad():
                attn_mask = generate_attention_mask(inpt)
                output = self.model(input_ids=inpt, attention_mask=attn_mask)[0]
                output = torch.mean(output, dim=1)
        else:
            attn_mask = generate_attention_mask(inpt)
            output = self.model(input_ids=inpt, attention_mask=attn_mask)[0]
            output = torch.mean(output, dim=1)
        if aspect == 'coherence':
            coherence_rest = self.coherence_m(output)
            return coherence_rest
        elif aspect == 'fluency':
            fluency_rest = self.fluency_m(output)
            return fluency_rest
        elif aspect == 'diversity':
            diversity_rest = self.diversity_m(output)
            return diversity_rest
        elif aspect == 'naturalness':
            naturalness_rest = self.naturalness_m(output)
            return naturalness_rest
        elif aspect == 'relatedness':
            relatedness_rest = self.relatedness_m(output)
            return relatedness_rest
        elif aspect == 'overall':
            fluency_m = self.fluency_m(output)
            coherence_m = self.coherence_m(output)
            diversity_m = self.diversity_m(output)
            naturalness_m = self.naturalness_m(output)
            relatedness_m = self.relatedness_m(output)
            output = torch.cat([fluency_m, coherence_m, diversity_m, naturalness_m, relatedness_m], dim=1)
            output = self.head(torch.relu(output))
            return output
        else:
            raise Exception(f'[!] target aspect {aspect} is unknown')


class BertEmbedding(nn.Module):
    """squeeze strategy: 1. first; 2. first-m; 3. average"""

    def __init__(self, m=0, lang='zh'):
        super(BertEmbedding, self).__init__()
        model_name = 'bert-base-chinese' if lang == 'zh' else 'bert-base-uncased'
        self.model = BertModel.from_pretrained(model_name)
        self.m = m

    def forward(self, ids, attn_mask, strategy='first'):
        """convert ids to embedding tensor; Return: [B, 768]"""
        embd = self.model(ids, attention_mask=attn_mask)[0]
        if strategy == 'first':
            rest = embd[:, 0, :]
        elif strategy == 'first-m':
            rest = embd[:, :self.m, :]
        elif strategy == 'average':
            rest = embd.mean(dim=1)
        else:
            raise Exception(f'[!] Unknow squeeze strategy {self.squeeze_strategy}')
        return rest


def to_cuda(x, model=False):
    if torch.cuda.is_available():
        if model:
            x
            return None
        else:
            return x
    else:
        return x


class PolyCompEncoder(nn.Module):

    def __init__(self, nhead, dim_feedforward, num_encoder_layers, dropout=0.1, m=16, lang='zh'):
        super(PolyCompEncoder, self).__init__()
        self.ctx_encoder = BertEmbedding(m=m, lang=lang)
        self.can_encoder = BertEmbedding(lang=lang)
        encoder_layer = nn.TransformerEncoderLayer(768, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        encoder_norm = nn.LayerNorm(768)
        self.trs_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.proj1 = nn.Linear(768 * 2, 768)
        self.gate = nn.Linear(768 * 3, 768)
        self.dropout = nn.Dropout(p=dropout)
        self.layernorm = nn.LayerNorm(768)

    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_rep = self.ctx_encoder(cid, cid_mask, strategy='first-m')
        rid_rep = self.can_encoder(rid, rid_mask, strategy='first')
        return cid_rep, rid_rep

    @torch.no_grad()
    def predict(self, cid, rid, rid_mask):
        batch_size = rid.shape[0]
        cid_rep, rid_rep = self._encode(cid.unsqueeze(0), rid, None, rid_mask)
        cid_rep = cid_rep.squeeze(0)
        weights = F.softmax(torch.matmul(cid_rep, rid_rep.t()).transpose(0, 1), dim=-1)
        cid_rep = torch.matmul(weights, cid_rep)
        cross_rep = torch.cat([cid_rep, rid_rep], dim=-1)
        cross_rep = self.dropout(torch.tanh(self.trs_encoder(torch.tanh(self.proj1(cross_rep).unsqueeze(1)))).squeeze(1))
        gate = torch.sigmoid(self.gate(torch.cat([rid_rep, cid_rep, cross_rep], dim=-1)))
        cross_rep = self.layernorm(gate * rid_rep + (1 - gate) * cross_rep)
        dot_product = (cid_rep * cross_rep).sum(-1)
        return dot_product

    def forward(self, cid, rid, cid_mask, rid_mask):
        batch_size = cid.shape[0]
        assert batch_size > 1, f'[!] batch size must bigger than 1, cause other elements in the batch will be seen as the negative samples'
        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
        weights = F.softmax(torch.matmul(cid_rep, rid_rep.t()).permute(0, 2, 1), dim=-1)
        cid_rep = torch.bmm(weights, cid_rep)
        cross_rep = torch.cat([cid_rep, rid_rep.unsqueeze(0).expand(batch_size, -1, -1)], dim=-1).permute(1, 0, 2)
        cross_rep = self.dropout(torch.tanh(self.trs_encoder(torch.tanh(self.proj1(cross_rep)))).permute(1, 0, 2))
        gate = torch.sigmoid(self.gate(torch.cat([rid_rep.unsqueeze(0).expand(batch_size, -1, -1), cid_rep, cross_rep], dim=-1)))
        cross_rep = self.layernorm(gate * rid_rep.unsqueeze(0).expand(batch_size, -1, -1) + (1 - gate) * cross_rep)
        dot_product = (cid_rep * cross_rep).sum(-1)
        mask = to_cuda(torch.eye(batch_size)).half()
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size))).sum().item()
        acc = acc_num / batch_size
        loss = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss.sum(dim=1)).mean()
        return loss, acc


class PolyCompEncoder_gate(nn.Module):

    def __init__(self, nhead, dim_feedforward, num_encoder_layers, dropout=0.1, m=16, lang='zh'):
        super(PolyCompEncoder_gate, self).__init__()
        self.ctx_encoder = BertEmbedding(m=m, lang=lang)
        self.can_encoder = BertEmbedding(lang=lang)
        encoder_layer = nn.TransformerEncoderLayer(768, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        encoder_norm = nn.LayerNorm(768)
        self.trs_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.proj1 = nn.Linear(768 * 2, 768)
        self.gate = nn.Linear(768 * 3, 768)
        self.dropout = nn.Dropout(p=dropout)
        self.layernorm = nn.LayerNorm(768)

    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_rep = self.ctx_encoder(cid, cid_mask, strategy='first-m')
        rid_rep = self.can_encoder(rid, rid_mask, strategy='first')
        return cid_rep, rid_rep

    @torch.no_grad()
    def predict(self, cid, rid, rid_mask):
        batch_size = rid.shape[0]
        cid_rep, rid_rep = self._encode(cid.unsqueeze(0), rid, None, rid_mask)
        cid_rep = cid_rep.squeeze(0)
        weights = F.softmax(torch.matmul(cid_rep, rid_rep.t()).transpose(0, 1), dim=-1)
        cid_rep = torch.matmul(weights, cid_rep)
        cross_rep = torch.cat([cid_rep, rid_rep], dim=-1)
        cross_rep = self.dropout(torch.tanh(self.trs_encoder(torch.tanh(self.proj1(cross_rep).unsqueeze(1)))).squeeze(1))
        cross_rep = self.layernorm(rid_rep + cross_rep)
        dot_product = (cid_rep * cross_rep).sum(-1)
        return dot_product

    def forward(self, cid, rid, cid_mask, rid_mask):
        batch_size = cid.shape[0]
        assert batch_size > 1, f'[!] batch size must bigger than 1, cause other elements in the batch will be seen as the negative samples'
        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
        weights = F.softmax(torch.matmul(cid_rep, rid_rep.t()).permute(0, 2, 1), dim=-1)
        cid_rep = torch.bmm(weights, cid_rep)
        cross_rep = torch.cat([cid_rep, rid_rep.unsqueeze(0).expand(batch_size, -1, -1)], dim=-1).permute(1, 0, 2)
        cross_rep = self.dropout(torch.tanh(self.trs_encoder(torch.tanh(self.proj1(cross_rep)))).permute(1, 0, 2))
        cross_rep = self.layernorm(rid_rep.unsqueeze(0).expand(batch_size, -1, -1) + cross_rep)
        dot_product = (cid_rep * cross_rep).sum(-1)
        mask = to_cuda(torch.eye(batch_size)).half()
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size))).sum().item()
        acc = acc_num / batch_size
        loss = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss.sum(dim=1)).mean()
        return loss, acc


class PolyCompEncoder_comp(nn.Module):

    def __init__(self, nhead, dim_feedforward, num_encoder_layers, dropout=0.1, m=16, lang='zh'):
        super(PolyCompEncoder_comp, self).__init__()
        self.ctx_encoder = BertEmbedding(m=m, lang=lang)
        self.can_encoder = BertEmbedding(lang=lang)
        encoder_layer = nn.TransformerEncoderLayer(768, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        encoder_norm = nn.LayerNorm(768)
        self.trs_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.proj1 = nn.Linear(768 * 2, 768)
        self.gate = nn.Linear(768 * 3, 768)
        self.dropout = nn.Dropout(p=dropout)
        self.layernorm = nn.LayerNorm(768)

    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_rep = self.ctx_encoder(cid, cid_mask, strategy='first-m')
        rid_rep = self.can_encoder(rid, rid_mask, strategy='first')
        return cid_rep, rid_rep

    @torch.no_grad()
    def predict(self, cid, rid, rid_mask):
        batch_size = rid.shape[0]
        cid_rep, rid_rep = self._encode(cid.unsqueeze(0), rid, None, rid_mask)
        cid_rep = cid_rep.squeeze(0)
        weights = F.softmax(torch.matmul(cid_rep, rid_rep.t()).transpose(0, 1), dim=-1)
        cid_rep = torch.matmul(weights, cid_rep)
        cross_rep = torch.cat([cid_rep, rid_rep], dim=-1)
        cross_rep = self.dropout(torch.tanh(self.proj1(cross_rep).unsqueeze(1)).squeeze(1))
        gate = torch.sigmoid(self.gate(torch.cat([rid_rep, cid_rep, cross_rep], dim=-1)))
        cross_rep = self.layernorm(gate * rid_rep + (1 - gate) * cross_rep)
        dot_product = (cid_rep * cross_rep).sum(-1)
        return dot_product

    def forward(self, cid, rid, cid_mask, rid_mask):
        batch_size = cid.shape[0]
        assert batch_size > 1, f'[!] batch size must bigger than 1, cause other elements in the batch will be seen as the negative samples'
        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
        weights = F.softmax(torch.matmul(cid_rep, rid_rep.t()).permute(0, 2, 1), dim=-1)
        cid_rep = torch.bmm(weights, cid_rep)
        cross_rep = torch.cat([cid_rep, rid_rep.unsqueeze(0).expand(batch_size, -1, -1)], dim=-1).permute(1, 0, 2)
        cross_rep = self.dropout(torch.tanh(self.proj1(cross_rep)).permute(1, 0, 2))
        gate = torch.sigmoid(self.gate(torch.cat([rid_rep.unsqueeze(0).expand(batch_size, -1, -1), cid_rep, cross_rep], dim=-1)))
        cross_rep = self.layernorm(gate * rid_rep.unsqueeze(0).expand(batch_size, -1, -1) + (1 - gate) * cross_rep)
        dot_product = (cid_rep * cross_rep).sum(-1)
        mask = to_cuda(torch.eye(batch_size)).half()
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size))).sum().item()
        acc = acc_num / batch_size
        loss = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss.sum(dim=1)).mean()
        return loss, acc


class PolyCompEncoder_car(nn.Module):

    def __init__(self, nhead, dim_feedforward, num_encoder_layers, dropout=0.1, m=16, lang='zh'):
        super(PolyCompEncoder_car, self).__init__()
        self.ctx_encoder = BertEmbedding(m=m, lang=lang)
        self.can_encoder = BertEmbedding(lang=lang)
        encoder_layer = nn.TransformerEncoderLayer(768, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        encoder_norm = nn.LayerNorm(768)
        self.trs_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.proj1 = nn.Linear(768 * 2, 768)
        self.gate = nn.Linear(768 * 3, 768)
        self.dropout = nn.Dropout(p=dropout)
        self.layernorm = nn.LayerNorm(768)

    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_rep = self.ctx_encoder(cid, cid_mask, strategy='first-m')
        rid_rep = self.can_encoder(rid, rid_mask, strategy='first')
        return cid_rep, rid_rep

    @torch.no_grad()
    def predict(self, cid, rid, rid_mask):
        batch_size = rid.shape[0]
        cid_rep, rid_rep = self._encode(cid.unsqueeze(0), rid, None, rid_mask)
        cid_rep = cid_rep.squeeze(0)
        weights = F.softmax(torch.matmul(cid_rep, rid_rep.t()).transpose(0, 1), dim=-1)
        cid_rep = torch.matmul(weights, cid_rep)
        cross_rep = torch.cat([rid_rep, rid_rep], dim=-1)
        cross_rep = self.dropout(torch.tanh(self.trs_encoder(torch.tanh(self.proj1(cross_rep).unsqueeze(1)))).squeeze(1))
        gate = torch.sigmoid(self.gate(torch.cat([rid_rep, cid_rep, cross_rep], dim=-1)))
        cross_rep = self.layernorm(gate * rid_rep + (1 - gate) * cross_rep)
        dot_product = (cid_rep * cross_rep).sum(-1)
        return dot_product

    def forward(self, cid, rid, cid_mask, rid_mask):
        batch_size = cid.shape[0]
        assert batch_size > 1, f'[!] batch size must bigger than 1, cause other elements in the batch will be seen as the negative samples'
        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
        weights = F.softmax(torch.matmul(cid_rep, rid_rep.t()).permute(0, 2, 1), dim=-1)
        cid_rep = torch.bmm(weights, cid_rep)
        cross_rep = torch.cat([rid_rep.unsqueeze(0).expand(batch_size, -1, -1), rid_rep.unsqueeze(0).expand(batch_size, -1, -1)], dim=-1).permute(1, 0, 2)
        cross_rep = self.dropout(torch.tanh(self.trs_encoder(torch.tanh(self.proj1(cross_rep)))).permute(1, 0, 2))
        gate = torch.sigmoid(self.gate(torch.cat([rid_rep.unsqueeze(0).expand(batch_size, -1, -1), cid_rep, cross_rep], dim=-1)))
        cross_rep = self.layernorm(gate * rid_rep.unsqueeze(0).expand(batch_size, -1, -1) + (1 - gate) * cross_rep)
        dot_product = (cid_rep * cross_rep).sum(-1)
        mask = to_cuda(torch.eye(batch_size)).half()
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size))).sum().item()
        acc = acc_num / batch_size
        loss = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss.sum(dim=1)).mean()
        return loss, acc


class PolyEncoder(nn.Module):

    def __init__(self, m=16, lang='zh'):
        super(PolyEncoder, self).__init__()
        self.ctx_encoder = BertEmbedding(m=m, lang=lang)
        self.can_encoder = BertEmbedding(lang=lang)

    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_rep = self.ctx_encoder(cid, cid_mask, strategy='first-m')
        rid_rep = self.can_encoder(rid, rid_mask, strategy='first')
        return cid_rep, rid_rep

    @torch.no_grad()
    def predict(self, cid, rid, rid_mask):
        batch_size = rid.shape[0]
        cid_rep, rid_rep = self._encode(cid.unsqueeze(0), rid, None, rid_mask)
        cid_rep = cid_rep.squeeze(0)
        weights = F.softmax(torch.matmul(cid_rep, rid_rep.t()).transpose(0, 1), dim=-1)
        cid_rep = torch.matmul(weights, cid_rep)
        dot_product = (cid_rep * rid_rep).sum(-1)
        return dot_product

    def forward(self, cid, rid, cid_mask, rid_mask):
        batch_size = cid.shape[0]
        assert batch_size > 1, f'[!] batch size must bigger than 1, cause other elements in the batch will be seen as the negative samples'
        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
        weights = F.softmax(torch.matmul(cid_rep, rid_rep.t()).permute(0, 2, 1), dim=-1)
        cid_rep = torch.bmm(weights, cid_rep)
        dot_product = (cid_rep * rid_rep.unsqueeze(0).expand(batch_size, -1, -1)).sum(-1)
        mask = to_cuda(torch.eye(batch_size)).half()
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size))).sum().item()
        acc = acc_num / batch_size
        loss = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss.sum(dim=1)).mean()
        return loss, acc


class RUBERTBiEncoder(nn.Module):
    """Re-used Bert bi-encoder model"""

    def __init__(self, max_turn_length=10, m=16, lang='zh'):
        super(RUBERTBiEncoder, self).__init__()
        self.ctx_encoder = BertEmbedding(m=m, lang=lang)
        self.can_encoder = BertEmbedding(lang=lang)
        self.max_chunk = 64

    def _encode(self, ids, ids_mask, ctx=True):
        if ctx:
            id_rep = []
            for idx in range(0, len(ids), self.max_chunk):
                subids = ids[idx:idx + self.max_chunk]
                submask = ids_mask[idx:idx + self.max_chunk]
                sub_id_rep = self.ctx_encoder(subids, submask, strategy='first-m')
                id_rep.append(sub_id_rep)
            id_rep = torch.cat(id_rep)
        else:
            id_rep = self.can_encoder(ids, ids_mask)
        return id_rep

    def squeeze(self, cid):
        """squeeze the context embeddings; cid: [B*T, M, E]"""
        rest = []
        for sample in cid:
            sample = sample.reshape(-1, sample.shape[-1])
            weight = self.turn_weight[:len(sample)].unsqueeze(1).expand(-1, 768)
            rest.append(torch.sum(weight * sample, dim=0))
        rest = torch.stack(rest)
        return rest

    @torch.no_grad()
    def talk_predict(self, cid, rid, rid_mask):
        """self.history_embd is used"""
        cid_rep = self._encode(cid, ctx=True)
        self.history_embd.append(cid_rep)
        cid_rep = torch.stack(self.history_embd)
        cid_rep = self.squeeze([cid_rep]).squeeze(0)
        rid_rep = self._encode(rid, rid_mask, ctx=False)
        dot_product = torch.matmul(cid_rep, rid_rep.t())
        return dot_product

    @torch.no_grad()
    def predict(self, cid, rid, cid_mask, rid_mask):
        """return the dot product of this turn and the rid_rep for the agent"""
        batch_size = rid.shape[0]
        cid_rep = self._encode(cid, cid_mask, ctx=True)
        cid_rep = self.squeeze([cid_rep]).squeeze(0)
        rid_rep = self._encode(rid, rid_mask, ctx=False)
        dot_product = torch.matmul(cid_rep, rid_rep.t())
        return dot_product

    def forward(self, cid, turn_length, rid, cid_mask, rid_mask):
        """cid: [B*T, S]; rid: [B, S]; cid_mask: [B*T, S]; rid_mask: [B, S]"""
        batch_size = rid.shape[0]
        cid_rep = self._encode(cid, cid_mask, ctx=True)
        cid_rep = self.squeeze(torch.split(cid_rep, turn_length))
        rid_rep = self._encode(rid, rid_mask, ctx=False)
        dot_product = torch.matmul(cid_rep, rid_rep.t())
        mask = to_cuda(torch.eye(batch_size)).half()
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size))).sum().item()
        acc = acc_num / batch_size
        loss = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss.sum(dim=1)).mean()
        return loss, acc


class BERTBiEncoder(nn.Module):
    """During training, the other elements in the batch are seen as the negative samples, which will lead to the fast training speed. More details can be found in paper: https://arxiv.org/pdf/1905.01969v2.pdf
    reference: https://github.com/chijames/Poly-Encoder/blob/master/encoder.py
    """

    def __init__(self, lang='zh'):
        super(BERTBiEncoder, self).__init__()
        self.ctx_encoder = BertEmbedding(lang=lang)
        self.can_encoder = BertEmbedding(lang=lang)

    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_rep = self.ctx_encoder(cid, cid_mask)
        rid_rep = self.can_encoder(rid, rid_mask)
        return cid_rep, rid_rep

    @torch.no_grad()
    def predict(self, cid, rid, rid_mask):
        batch_size = rid.shape[0]
        cid_rep, rid_rep = self._encode(cid.unsqueeze(0), rid, None, rid_mask)
        cid_rep = cid_rep.squeeze(0)
        dot_product = torch.matmul(cid_rep, rid_rep.t())
        return dot_product

    def forward(self, cid, rid, cid_mask, rid_mask):
        batch_size = cid.shape[0]
        assert batch_size > 1, f'[!] batch size must bigger than 1, cause other elements in the batch will be seen as the negative samples'
        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
        dot_product = torch.matmul(cid_rep, rid_rep.t())
        mask = to_cuda(torch.eye(batch_size)).half()
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size))).sum().item()
        acc = acc_num / batch_size
        loss = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss.sum(dim=1)).mean()
        return loss, acc


class BERTBiCompEncoder(nn.Module):
    """During training, the other elements in the batch are seen as the negative samples, which will lead to the fast training speed. More details can be found in paper: https://arxiv.org/pdf/1905.01969v2.pdf
    reference: https://github.com/chijames/Poly-Encoder/blob/master/encoder.py
    
    Set the different learning ratio
    """

    def __init__(self, nhead, dim_feedforward, num_encoder_layers, dropout=0.1, lang='zh'):
        super(BERTBiCompEncoder, self).__init__()
        self.ctx_encoder = BertEmbedding(lang=lang)
        self.can_encoder = BertEmbedding(lang=lang)
        encoder_layer = nn.TransformerEncoderLayer(768, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        encoder_norm = nn.LayerNorm(768)
        self.trs_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.proj1 = nn.Linear(768 * 2, 768)
        self.gate = nn.Linear(768 * 3, 768)
        self.dropout = nn.Dropout(p=dropout)
        self.layernorm = nn.LayerNorm(768)

    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_rep = self.ctx_encoder(cid, cid_mask)
        rid_rep = self.can_encoder(rid, rid_mask)
        return cid_rep, rid_rep

    @torch.no_grad()
    def predict(self, cid, rid, rid_mask):
        batch_size = rid.shape[0]
        cid_rep, rid_rep = self._encode(cid.unsqueeze(0), rid, None, rid_mask)
        cid_rep = cid_rep.squeeze(0)
        cross_rep = torch.cat([cid_rep.unsqueeze(0).expand(batch_size, -1), rid_rep], dim=1)
        cross_rep = self.dropout(torch.tanh(self.trs_encoder(torch.tanh(self.proj1(cross_rep).unsqueeze(1)))).squeeze(1))
        gate = torch.sigmoid(self.gate(torch.cat([rid_rep, cid_rep.unsqueeze(0).expand(batch_size, -1), cross_rep], dim=-1)))
        cross_rep = self.layernorm(gate * rid_rep + (1 - gate) * cross_rep)
        dot_product = torch.matmul(cid_rep, cross_rep.t())
        return dot_product

    def forward(self, cid, rid, cid_mask, rid_mask):
        batch_size = cid.shape[0]
        assert batch_size > 1, f'[!] batch size must bigger than 1, cause other elements in the batch will be seen as the negative samples'
        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
        cross_rep = []
        for cid_rep_ in cid_rep:
            cid_rep_ = cid_rep_.unsqueeze(0).expand(batch_size, -1)
            cross_rep.append(torch.cat([cid_rep_, rid_rep], dim=-1))
        cross_rep = torch.stack(cross_rep).permute(1, 0, 2)
        cross_rep = self.dropout(torch.tanh(self.trs_encoder(torch.tanh(self.proj1(cross_rep)))).permute(1, 0, 2))
        gate = torch.sigmoid(self.gate(torch.cat([rid_rep.unsqueeze(0).expand(batch_size, -1, -1), cid_rep.unsqueeze(1).expand(-1, batch_size, -1), cross_rep], dim=-1)))
        cross_rep = self.layernorm(gate * rid_rep.unsqueeze(0).expand(batch_size, -1, -1) + (1 - gate) * cross_rep)
        cid_rep = cid_rep.unsqueeze(1)
        dot_product = torch.bmm(cid_rep, cross_rep.permute(0, 2, 1)).squeeze(1)
        mask = to_cuda(torch.eye(batch_size)).half()
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size))).sum().item()
        acc = acc_num / batch_size
        loss = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss.sum(dim=1)).mean()
        return loss, acc


class BERTBiCompEncoder_car(nn.Module):
    """bi-encoder+TCM-{context-aware}
    """

    def __init__(self, nhead, dim_feedforward, num_encoder_layers, dropout=0.1, lang='zh'):
        super(BERTBiCompEncoder_car, self).__init__()
        self.ctx_encoder = BertEmbedding(lang=lang)
        self.can_encoder = BertEmbedding(lang=lang)
        encoder_layer = nn.TransformerEncoderLayer(768, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        encoder_norm = nn.LayerNorm(768)
        self.trs_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.proj1 = nn.Linear(768 * 2, 768)
        self.gate = nn.Linear(768 * 3, 768)
        self.dropout = nn.Dropout(p=dropout)
        self.layernorm = nn.LayerNorm(768)

    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_rep = self.ctx_encoder(cid, cid_mask)
        rid_rep = self.can_encoder(rid, rid_mask)
        return cid_rep, rid_rep

    @torch.no_grad()
    def predict(self, cid, rid, rid_mask):
        batch_size = rid.shape[0]
        cid_rep, rid_rep = self._encode(cid.unsqueeze(0), rid, None, rid_mask)
        cid_rep = cid_rep.squeeze(0)
        cross_rep = torch.cat([rid_rep, rid_rep], dim=1)
        cross_rep = self.dropout(torch.tanh(self.trs_encoder(torch.tanh(self.proj1(cross_rep).unsqueeze(1)))).squeeze(1))
        gate = torch.sigmoid(self.gate(torch.cat([rid_rep, cid_rep.unsqueeze(0).expand(batch_size, -1), cross_rep], dim=-1)))
        cross_rep = self.layernorm(gate * rid_rep + (1 - gate) * cross_rep)
        dot_product = torch.matmul(cid_rep, cross_rep.t())
        return dot_product

    def forward(self, cid, rid, cid_mask, rid_mask):
        batch_size = cid.shape[0]
        assert batch_size > 1, f'[!] batch size must bigger than 1, cause other elements in the batch will be seen as the negative samples'
        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
        cross_rep = []
        for cid_rep_ in cid_rep:
            cross_rep.append(torch.cat([rid_rep, rid_rep], dim=-1))
        cross_rep = torch.stack(cross_rep).permute(1, 0, 2)
        cross_rep = self.dropout(torch.tanh(self.trs_encoder(torch.tanh(self.proj1(cross_rep)))).permute(1, 0, 2))
        gate = torch.sigmoid(self.gate(torch.cat([rid_rep.unsqueeze(0).expand(batch_size, -1, -1), cid_rep.unsqueeze(1).expand(-1, batch_size, -1), cross_rep], dim=-1)))
        cross_rep = self.layernorm(gate * rid_rep.unsqueeze(0).expand(batch_size, -1, -1) + (1 - gate) * cross_rep)
        cid_rep = cid_rep.unsqueeze(1)
        dot_product = torch.bmm(cid_rep, cross_rep.permute(0, 2, 1)).squeeze(1)
        mask = to_cuda(torch.eye(batch_size)).half()
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size))).sum().item()
        acc = acc_num / batch_size
        loss = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss.sum(dim=1)).mean()
        return loss, acc


class BERTBiCompEncoder_comp(nn.Module):
    """During training, the other elements in the batch are seen as the negative samples, which will lead to the fast training speed. More details can be found in paper: https://arxiv.org/pdf/1905.01969v2.pdf
    reference: https://github.com/chijames/Poly-Encoder/blob/master/encoder.py
    
    Set the different learning ratio
    """

    def __init__(self, nhead, dim_feedforward, num_encoder_layers, dropout=0.1, lang='zh'):
        super(BERTBiCompEncoder_comp, self).__init__()
        self.ctx_encoder = BertEmbedding(lang=lang)
        self.can_encoder = BertEmbedding(lang=lang)
        encoder_layer = nn.TransformerEncoderLayer(768, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        encoder_norm = nn.LayerNorm(768)
        self.trs_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.proj1 = nn.Linear(768 * 2, 768)
        self.gate = nn.Linear(768 * 3, 768)
        self.dropout = nn.Dropout(p=dropout)
        self.layernorm = nn.LayerNorm(768)

    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_rep = self.ctx_encoder(cid, cid_mask)
        rid_rep = self.can_encoder(rid, rid_mask)
        return cid_rep, rid_rep

    @torch.no_grad()
    def predict(self, cid, rid, rid_mask):
        batch_size = rid.shape[0]
        cid_rep, rid_rep = self._encode(cid.unsqueeze(0), rid, None, rid_mask)
        cid_rep = cid_rep.squeeze(0)
        cross_rep = torch.cat([cid_rep.unsqueeze(0).expand(batch_size, -1), rid_rep], dim=1)
        cross_rep = self.dropout(torch.tanh(self.proj1(cross_rep).unsqueeze(1)).squeeze(1))
        gate = torch.sigmoid(self.gate(torch.cat([rid_rep, cid_rep.unsqueeze(0).expand(batch_size, -1), cross_rep], dim=-1)))
        cross_rep = self.layernorm(gate * rid_rep + (1 - gate) * cross_rep)
        dot_product = torch.matmul(cid_rep, cross_rep.t())
        return dot_product

    def forward(self, cid, rid, cid_mask, rid_mask):
        batch_size = cid.shape[0]
        assert batch_size > 1, f'[!] batch size must bigger than 1, cause other elements in the batch will be seen as the negative samples'
        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
        cross_rep = []
        for cid_rep_ in cid_rep:
            cid_rep_ = cid_rep_.unsqueeze(0).expand(batch_size, -1)
            cross_rep.append(torch.cat([cid_rep_, rid_rep], dim=-1))
        cross_rep = torch.stack(cross_rep).permute(1, 0, 2)
        cross_rep = self.dropout(torch.tanh(self.proj1(cross_rep)).permute(1, 0, 2))
        gate = torch.sigmoid(self.gate(torch.cat([rid_rep.unsqueeze(0).expand(batch_size, -1, -1), cid_rep.unsqueeze(1).expand(-1, batch_size, -1), cross_rep], dim=-1)))
        cross_rep = self.layernorm(gate * rid_rep.unsqueeze(0).expand(batch_size, -1, -1) + (1 - gate) * cross_rep)
        cid_rep = cid_rep.unsqueeze(1)
        dot_product = torch.bmm(cid_rep, cross_rep.permute(0, 2, 1)).squeeze(1)
        mask = to_cuda(torch.eye(batch_size)).half()
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size))).sum().item()
        acc = acc_num / batch_size
        loss = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss.sum(dim=1)).mean()
        return loss, acc


class BERTBiCompEncoder_gate(nn.Module):
    """During training, the other elements in the batch are seen as the negative samples, which will lead to the fast training speed. More details can be found in paper: https://arxiv.org/pdf/1905.01969v2.pdf
    reference: https://github.com/chijames/Poly-Encoder/blob/master/encoder.py
    
    Set the different learning ratio
    """

    def __init__(self, nhead, dim_feedforward, num_encoder_layers, dropout=0.1, lang='zh'):
        super(BERTBiCompEncoder_gate, self).__init__()
        self.ctx_encoder = BertEmbedding(lang=lang)
        self.can_encoder = BertEmbedding(lang=lang)
        encoder_layer = nn.TransformerEncoderLayer(768, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        encoder_norm = nn.LayerNorm(768)
        self.trs_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.proj1 = nn.Linear(768 * 2, 768)
        self.gate = nn.Linear(768 * 3, 768)
        self.dropout = nn.Dropout(p=dropout)
        self.layernorm = nn.LayerNorm(768)

    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_rep = self.ctx_encoder(cid, cid_mask)
        rid_rep = self.can_encoder(rid, rid_mask)
        return cid_rep, rid_rep

    @torch.no_grad()
    def predict(self, cid, rid, rid_mask):
        batch_size = rid.shape[0]
        cid_rep, rid_rep = self._encode(cid.unsqueeze(0), rid, None, rid_mask)
        cid_rep = cid_rep.squeeze(0)
        cross_rep = torch.cat([cid_rep.unsqueeze(0).expand(batch_size, -1), rid_rep], dim=1)
        cross_rep = self.dropout(torch.tanh(self.trs_encoder(torch.tanh(self.proj1(cross_rep).unsqueeze(1)))).squeeze(1))
        cross_rep = rid_rep + cross_rep
        dot_product = torch.matmul(cid_rep, cross_rep.t())
        return dot_product

    def forward(self, cid, rid, cid_mask, rid_mask):
        batch_size = cid.shape[0]
        assert batch_size > 1, f'[!] batch size must bigger than 1, cause other elements in the batch will be seen as the negative samples'
        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
        cross_rep = []
        for cid_rep_ in cid_rep:
            cid_rep_ = cid_rep_.unsqueeze(0).expand(batch_size, -1)
            cross_rep.append(torch.cat([cid_rep_, rid_rep], dim=-1))
        cross_rep = torch.stack(cross_rep).permute(1, 0, 2)
        cross_rep = self.dropout(torch.tanh(self.trs_encoder(torch.tanh(self.proj1(cross_rep)))).permute(1, 0, 2))
        cross_rep = rid_rep.unsqueeze(0).expand(batch_size, -1, -1) + cross_rep
        cid_rep = cid_rep.unsqueeze(1)
        dot_product = torch.bmm(cid_rep, cross_rep.permute(0, 2, 1)).squeeze(1)
        mask = to_cuda(torch.eye(batch_size)).half()
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size))).sum().item()
        acc = acc_num / batch_size
        loss = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss.sum(dim=1)).mean()
        return loss, acc


def top_k_top_p_filtering_batch(logits, top_k=0, top_p=0.0, filter_value=-np.inf, min_token_to_keep=1):
    """
    :logits: [batch, vocab]
    :return logits: [batch, vocab]
    refer to https://zhuanlan.zhihu.com/p/115076102
    """
    if top_k > 0:
        top_k = min(max(top_k, min_token_to_keep), logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_token_to_keep > 1:
            sorted_indices_to_remove[..., :min_token_to_keep] = 0
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


class GPT2RL(nn.Module):
    """
    past mechanism can speed up and decrease the capacity of the cuda memory
    In order to run the batch version, the GPT2RL contains the generator and discriminator
    Rollout with top_k_top_p mechanism are used
    
    In the debug mode, the attribute `memory` is not NoneType
    """

    def __init__(self, min_token, rollout_samples, generative_path, topk, topp, config_path='data/config/model_config_dialogue_small.json', vocab_path='data/vocab/vocab_small', memory=None):
        super(GPT2RL, self).__init__()
        self.vocab = BertTokenizer(vocab_file=vocab_path)
        vocab_size = len(self.vocab)
        self.debug = True if memory else False
        self.memory = memory
        self.model_config = GPT2Config.from_json_file(config_path)
        self.generator = GPT2LMHeadModel(config=self.model_config)
        self.generator.resize_token_embeddings(vocab_size)
        self.discriminator = BERTRetrieval()
        self.n_ctx = self.generator.config.to_dict().get('n_ctx')
        self.sample_topk, self.rl_topk = topk
        self.sample_topp, self.rl_topp = topp
        self.unk_id = self.vocab.convert_tokens_to_ids('[UNK]')
        self.sep_id = self.vocab.convert_tokens_to_ids('[SEP]')
        self.cls_id = self.vocab.convert_tokens_to_ids('[CLS]')
        self.pad_id = self.vocab.convert_tokens_to_ids('[PAD]')
        self.rollout_samples = rollout_samples
        self.min_token_to_keep = min_token
        self.load_model(self.generator, generative_path)
        None

    def load_model(self, model, path):
        state_dict = torch.load(path)
        try:
            model.load_state_dict(state_dict)
        except:
            current_module = True if 'model' in [i[0] for i in model.state_dict().items()][0] else False
            saved_module = True if 'model' in [i[0] for i in state_dict.items()][0] else False
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if saved_module and not current_module:
                    name = k[6:]
                    new_state_dict[name] = v
                elif not saved_module and current_module:
                    name = f'model.{k}'
                    new_state_dict[name] = v
                else:
                    pass
            model.load_state_dict(new_state_dict)
        None

    @torch.no_grad()
    def generative_predict(self, inpt_ids, max_len):
        """
        Generate the fake data for the discriminator
        inpt_ids: [batch, seq]
        return: [max_len, batch]
        """
        generated = []
        prev, past = inpt_ids.clone().detach(), None
        sep_batch_index = [0] * len(inpt_ids)
        sep_batch_flag = [0] * len(inpt_ids)
        for i in range(max_len):
            outputs = self.generator(input_ids=prev, past=past)
            outputs, past = outputs[:2]
            next_token_logits = outputs[:, -1, :]
            next_token_logits[:, self.unk_id] = -np.inf
            filtered_logits = top_k_top_p_filtering_batch(next_token_logits, top_k=self.sample_topk, top_p=self.sample_topp, min_token_to_keep=self.min_token_to_keep)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            for idx, token_ in enumerate(next_token.squeeze(1)):
                if sep_batch_flag[idx] == 0 and token_ == self.sep_id:
                    sep_batch_index[idx] = i + 1
                    sep_batch_flag[idx] = 1
            prev = next_token
            generated.append(next_token.squeeze(1).tolist())
        for i in range(len(sep_batch_flag)):
            if sep_batch_flag[i] == 0:
                sep_batch_index[i] = max_len
        return self.process_generated(generated, sep_batch_index, max_len)

    def process_generated(self, generated, index, max_len):
        """
        Process the generated samples for the training the discriminator
        1. filter tokens after [SEP]
        2. pad and return
        3. the last token of the sequence maybe the [PAD], [SEP] and other tokens

        :generated: [max_len, batch]
        :index: [batch]
        """
        rest = []
        for idx in range(len(index)):
            r = [item[idx] for item in generated]
            r = r[:index[idx]]
            if len(r) == 0 or len(r) < max_len and r[-1] != self.sep_id:
                r.append(self.sep_id)
            rest.append(torch.LongTensor(r))
        rest = pad_sequence(rest, batch_first=True, padding_value=self.pad_id)
        rest_len, batch_size = len(rest[0]), len(rest)
        if rest_len < max_len:
            add_pad = torch.LongTensor([[self.pad_id] * (max_len - rest_len)] * batch_size)
            rest = torch.cat((rest, add_pad), dim=1)
        if torch.cuda.is_available():
            rest = rest
        return rest

    @torch.no_grad()
    def rollout_batch(self, past, current_token, max_len):
        """
        Batch predict mode
        return :generated: [batch, max_len]
        Rollout speed up with `past` mechanism

        Also need to use `sep_batch_index`
        """
        batch_size = len(current_token)
        sep_batch_index = [([0] * batch_size) for _ in range(self.rollout_samples)]
        rollout_rest = []
        for rollout_idx in range(self.rollout_samples):
            response = []
            sep_batch_flag = [0] * batch_size
            past_ = tuple([i.clone().detach() for i in past])
            current_token_ = current_token.clone().detach()
            for lid in range(max_len):
                outputs = self.generator(input_ids=current_token_, past=past_)
                outputs, past_ = outputs[:2]
                next_token_logits = outputs[:, -1, :]
                next_token_logits[:, self.unk_id] = -np.inf
                next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1)
                current_token_ = next_token
                for idx, token_ in enumerate(next_token.squeeze(1)):
                    if sep_batch_flag[idx] == 0 and token_ == self.sep_id:
                        sep_batch_index[rollout_idx][idx] = lid + 1
                        sep_batch_flag[idx] = 1
                response.append(next_token.squeeze(1))
            for idx, v_ in enumerate(sep_batch_flag):
                if v_ == 0:
                    sep_batch_index[rollout_idx][idx] = max_len
            response = torch.stack(response).transpose(0, 1)
            for iidx, idx in enumerate(sep_batch_index[rollout_idx]):
                response[iidx][idx:] = self.pad_id
            rollout_rest.append(response)
        return rollout_rest

    @torch.no_grad()
    def obtain_rewards(self, rollout_samples):
        """
        Use discriminator to generate the rewards (scores) for the rollout samples
        :rollout_samples: self.rollout_samples*[batch, max_len]
        return :rewards: [batch] (average strategy are used)

        In order to use the DataParallel, cannot use the `.cuda` method

        Rewards range: [0, 100]
        """
        rewards = torch.zeros(len(rollout_samples[0]))
        for samples in rollout_samples:
            output = F.softmax(self.discriminator(samples), dim=-1)[:, 1].cpu()
            rewards += output
        return 100 * rewards / len(rollout_samples)

    def forward_mle(self, cid):
        """
        Train Generator with MLE
        """
        self.generator.train()
        attn_mask = generate_attention_mask(cid)
        logits = self.generator(cid, attention_mask=attn_mask)
        return logits[0]

    def forward_dis(self, inpt):
        """
        Train Discrimintator
        """
        self.discriminator.train()
        output = self.discriminator(inpt)
        return output

    def forward_rl(self, inpt_ids, max_len):
        """
        Train Generator with RL
        inpt_ids: [batch, seq]
        return actions and corresponding probs:
            torch tensor object: [batch, max_len], [batch, max_len]; [batch]
        """
        self.generator.train()
        generated, probs = [], []
        rewards = []
        inpt_ids_ = inpt_ids.clone().detach()
        sep_batch_index = torch.LongTensor([0] * len(inpt_ids))
        sep_batch_flag = [0] * len(inpt_ids)
        prev, past, history = inpt_ids_, None, inpt_ids_
        for i in range(1, max_len + 1):
            outputs = self.generator(input_ids=prev, past=past)
            outputs, past = outputs[:2]
            next_token_logits = outputs[:, -1, :]
            next_token_logits[:, self.unk_id] = -np.inf
            filtered_logits = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(filtered_logits, num_samples=1)
            next_token = next_token.squeeze(1)
            for idx, token_ in enumerate(next_token):
                if sep_batch_flag[idx] == 0 and token_ == self.sep_id:
                    sep_batch_index[idx] = i
                    sep_batch_flag[idx] = 1
            prob = filtered_logits[range(len(next_token)), next_token]
            generated.append(next_token)
            probs.append(prob)
            prev = next_token.unsqueeze(1)
            if i < max_len:
                rest = self.rollout_batch(past, next_token.unsqueeze(1), max_len - i)
                generated_ = torch.stack(generated).transpose(0, 1)
                rest = [torch.cat((inpt_ids, generated_, rollout_item), dim=1) for rollout_item in rest]
            else:
                for idx in range(len(sep_batch_index)):
                    if sep_batch_flag[idx] == 0:
                        sep_batch_index[idx] = max_len
                generated_ = torch.stack(generated).transpose(0, 1)
                for idx, index in enumerate(sep_batch_index):
                    generated_[idx][index:] = self.pad_id
                rest = [torch.cat((inpt_ids, generated_), dim=1)]
            rewards.append(self.obtain_rewards(rest))
        rewards = torch.stack(rewards).transpose(0, 1)
        probs = torch.stack(probs).transpose(0, 1)
        sep_batch_index = sep_batch_index
        return_data = {'probs': probs, 'rewards': rewards, 'sep_batch_index': sep_batch_index}
        return return_data

    def save_memory(self, path):
        """
        If debug mode is True, save the memory in the ckpt folder
        """
        with open(path, 'wb') as f:
            pickle.dump(self.memory, f)
        None

    def forward(self, cid, rid=None, max_size=None, mode='gen_rl'):
        """
        Compatible for DataParallel
        """
        if mode == 'gen_rl':
            assert max_size, f'[!] max_size must not be NoneType for gen_rl mode'
            data = self.forward_rl(cid, max_size)
            return data
        elif mode == 'gen_mle':
            if rid is None:
                raise Exception('[!] rid must not be NoneType for gen_mle mode')
            cid = torch.cat((cid, rid[:, 1:]), dim=1)
            shift_logits = self.forward_mle(cid)
            return shift_logits
        elif mode == 'dis':
            if rid is None:
                raise Exception('[!] rid must not be NoneType for gen_mle mode')
            cid = torch.cat((cid, rid), dim=1)
            output = self.forward_dis(cid)
            return output
        elif mode == 'gen_predict':
            f_rid = self.generative_predict(cid, max_size)
            return f_rid
        else:
            raise Exception(f'[!] Except to get [gen_rl; gen_mle; dis] mode, but got {mode}')


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, threshold=-float('Inf'), filter_value=-np.inf):
    assert logits.dim() == 1
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value
    return logits


class GPT2(nn.Module):

    def __init__(self, vocab_size, unk_id, sep_id, cls_id, topk, topp, repetition_penalty, config_path='data/config/model_config_dialogue_small.json'):
        super(GPT2, self).__init__()
        self.model_config = GPT2Config.from_json_file(config_path)
        self.model = GPT2LMHeadModel(config=self.model_config)
        self.model.resize_token_embeddings(vocab_size)
        self.n_ctx = self.model.config.to_dict().get('n_ctx')
        self.topk, self.topp = topk, topp
        self.unk_id = unk_id
        self.sep_id = sep_id
        self.cls_id = cls_id
        self.repetition_penalty = repetition_penalty

    def forward(self, inpt_ids):
        outputs = self.model(input_ids=inpt_ids)
        output = outputs[0]
        return output

    @torch.no_grad()
    def predict(self, inpt_ids, max_len):
        """batch_size is 1; inpt_ids: [seq]; return a list of ids (generated)"""
        generated = [self.cls_id]
        for _ in range(max_len):
            outputs = self.model(input_ids=inpt_ids)
            next_token_logits = outputs[0][-1, :]
            next_token_logits[self.unk_id] = -np.inf
            if generated:
                next_token_logits[list(set(generated))] /= self.repetition_penalty
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=self.topk, top_p=self.topp)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated.append(next_token.item())
            if next_token == self.sep_id:
                break
            inpt_ids = torch.cat((inpt_ids, next_token), dim=0)
            inpt_ids = inpt_ids[-self.n_ctx:]
        return generated

    @torch.no_grad()
    def predict_batch(self, inpt_ids, attn_mask, position_ids, max_len):
        """inpt_ids: [batch, seq]; return: samples"""
        batch_size = inpt_ids.shape[0]
        generated = [[self.cls_id] * batch_size]
        prev, past = inpt_ids, None
        stop_flag = np.zeros(batch_size)
        for _ in range(max_len):
            outputs = self.model(input_ids=prev, attention_mask=attn_mask, past=past, position_ids=position_ids)
            output, past = outputs[:2]
            next_token_logits = output[:, -1, :]
            next_token_logits[:, self.unk_id] = -np.inf
            for x in range(batch_size):
                y = [item[x] for item in generated]
                next_token_logits[x, y] /= self.repetition_penalty
            filtered_logits = top_k_top_p_filtering_batch(next_token_logits, top_k=self.topk, top_p=self.topp)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            for idx, i in enumerate(next_token.squeeze(1)):
                if i == self.sep_id:
                    stop_flag[idx] = 1
            generated.append([token.item() for token in next_token.squeeze(1)])
            prev = next_token
            if sum(stop_flag) == batch_size:
                break
            attn_mask = torch.cat([attn_mask, torch.tensor([1] * batch_size).unsqueeze(1)], dim=1)
            if past:
                position_ids = attn_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attn_mask == 0, 0)
                position_ids = position_ids[:, -1].unsqueeze(-1)
        ng, batch_size = [], len(generated[0])
        for i in range(batch_size):
            ng.append([g[i] for g in generated])
        return ng


class GPT2RL_V2(nn.Module):
    """
    past mechanism can speed up and decrease the capacity of the cuda memory
    In order to run the batch version, the GPT2RL contains the generator and discriminator
    Rollout with top_k_top_p mechanism are used
    Double discriminator
    """

    def __init__(self, min_token, rollout_samples, generative_path, discriminator_path, topk, topp, config_path='data/config/model_config_dialogue_small.json', vocab_path='data/vocab/vocab_small'):
        super(GPT2RL_V2, self).__init__()
        self.vocab = BertTokenizer(vocab_file=vocab_path)
        vocab_size = len(self.vocab)
        self.model_config = GPT2Config.from_json_file(config_path)
        self.generator = GPT2LMHeadModel(config=self.model_config)
        self.generator.resize_token_embeddings(vocab_size)
        self.discriminator = BERTRetrieval()
        self.discriminator_target = BERTRetrieval()
        self.n_ctx = self.generator.config.to_dict().get('n_ctx')
        self.topk, self.topp = topk, topp
        self.unk_id = self.vocab.convert_tokens_to_ids('[UNK]')
        self.sep_id = self.vocab.convert_tokens_to_ids('[SEP]')
        self.cls_id = self.vocab.convert_tokens_to_ids('[CLS]')
        self.pad_id = self.vocab.convert_tokens_to_ids('[PAD]')
        self.rollout_samples = rollout_samples
        self.min_token_to_keep = min_token
        self.load_model(self.generator, generative_path)
        self.load_model(self.discriminator, discriminator_path)
        self.load_model(self.discriminator_target, discriminator_path)
        None

    def reset_target_discriminator(self):
        self.discriminator_target.load_state_dict(self.discriminator.state_dict())
        None

    def load_model(self, model, path):
        state_dict = torch.load(path)
        try:
            model.load_state_dict(state_dict)
        except:
            current_module = True if 'model' in [i[0] for i in model.state_dict().items()][0] else False
            saved_module = True if 'model' in [i[0] for i in state_dict.items()][0] else False
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if saved_module and not current_module:
                    name = k[6:]
                    new_state_dict[name] = v
                elif not saved_module and current_module:
                    name = f'model.{k}'
                    new_state_dict[name] = v
                else:
                    pass
            model.load_state_dict(new_state_dict)
        None

    @torch.no_grad()
    def generative_predict(self, inpt_ids, max_len):
        """
        Generate the fake data for the discriminator
        inpt_ids: [batch, seq]
        return: [max_len, batch]
        """
        generated = []
        prev, past = inpt_ids.clone().detach(), None
        sep_batch_index = [0] * len(inpt_ids)
        sep_batch_flag = [0] * len(inpt_ids)
        for i in range(max_len):
            outputs = self.generator(input_ids=prev, past=past)
            outputs, past = outputs[:2]
            next_token_logits = outputs[:, -1, :]
            next_token_logits[:, self.unk_id] = -np.inf
            filtered_logits = top_k_top_p_filtering_batch(next_token_logits, top_k=self.topk, top_p=self.topp, min_token_to_keep=self.min_token_to_keep)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            for idx, token_ in enumerate(next_token.squeeze(1)):
                if sep_batch_flag[idx] == 0 and token_ == self.sep_id:
                    sep_batch_index[idx] = i + 1
                    sep_batch_flag[idx] = 1
            prev = next_token
            generated.append(next_token.squeeze(1).tolist())
        for i in range(len(sep_batch_flag)):
            if sep_batch_flag[i] == 0:
                sep_batch_index[i] = max_len
        return self.process_generated(generated, sep_batch_index, max_len)

    def process_generated(self, generated, index, max_len):
        """
        1. filter tokens after [SEP]
        2. pad and return
        3. make sure that the last token must be the [SEP]

        :generated: [max_len, batch]
        :index: [batch]
        """
        rest = []
        for idx in range(len(index)):
            r = [item[idx] for item in generated]
            r = r[:index[idx]]
            if len(r) == 0 or len(r) < max_len and r[-1] != self.sep_id:
                r.append(self.sep_id)
            if len(r) == max_len and r[-1] != self.sep_id:
                r[-1] = self.sep_id
            rest.append(torch.LongTensor(r))
        rest = pad_sequence(rest, batch_first=True, padding_value=self.pad_id)
        rest_len, batch_size = len(rest[0]), len(rest)
        if rest_len < max_len:
            add_pad = torch.LongTensor([[self.pad_id] * (max_len - rest_len)] * batch_size)
            rest = torch.cat((rest, add_pad), dim=1)
        if torch.cuda.is_available():
            rest = rest
        return rest

    @torch.no_grad()
    def rollout_batch(self, past, current_token, max_len):
        """
        Batch predict mode
        return :generated: [batch, max_len]
        Rollout speed up with `past` mechanism

        Also need to use `sep_batch_index`
        """
        batch_size = len(current_token)
        sep_batch_index = [([0] * batch_size) for _ in range(self.rollout_samples)]
        rollout_rest = []
        for rollout_idx in range(self.rollout_samples):
            response = []
            sep_batch_flag = [0] * batch_size
            past_ = tuple([i.clone().detach() for i in past])
            current_token_ = current_token.clone().detach()
            for lid in range(max_len):
                outputs = self.generator(input_ids=current_token_, past=past_)
                outputs, past_ = outputs[:2]
                next_token_logits = outputs[:, -1, :]
                next_token_logits[:, self.unk_id] = -np.inf
                filtered_logits = top_k_top_p_filtering_batch(next_token_logits, top_k=self.topk, top_p=self.topp, min_token_to_keep=self.min_token_to_keep)
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                current_token_ = next_token
                for idx, token_ in enumerate(next_token.squeeze(1)):
                    if sep_batch_flag[idx] == 0 and token_ == self.sep_id:
                        sep_batch_index[rollout_idx][idx] = lid + 1
                        sep_batch_flag[idx] = 1
                response.append(next_token.squeeze(1))
            for idx, v_ in enumerate(sep_batch_flag):
                if v_ == 0:
                    sep_batch_index[rollout_idx][idx] = max_len
            response = torch.stack(response).transpose(0, 1)
            for iidx, idx in enumerate(sep_batch_index[rollout_idx]):
                response[iidx][idx:] = self.pad_id
            rollout_rest.append(response)
        return rollout_rest

    @torch.no_grad()
    def obtain_rewards(self, rollout_samples):
        """
        Use discriminator to generate the rewards (scores) for the rollout samples
        :rollout_samples: self.rollout_samples*[batch, max_len]
        return :rewards: [batch] (average strategy are used)

        In order to use the DataParallel, cannot use the `.cuda` method

        Rewards range: [0, 100]
        """
        self.discriminator.eval()
        rewards = torch.zeros(len(rollout_samples[0]))
        for samples in rollout_samples:
            output = F.softmax(self.discriminator(samples), dim=-1)[:, 1].cpu()
            rewards += output
        return 100 * rewards / len(rollout_samples)

    def forward_mle(self, cid):
        """
        Train Generator with MLE
        """
        self.generator.train()
        attn_mask = generate_attention_mask(cid)
        logits = self.generator(cid, attention_mask=attn_mask)
        return logits[0]

    def forward_dis(self, inpt):
        """
        Train Discrimintator
        """
        self.discriminator.train()
        output = self.discriminator(inpt)
        return output

    @torch.no_grad()
    def forward_target_dis(self, inpt):
        """
        run the target discriminator network for generating pesudo labels
        """
        output = self.discriminator_target(inpt)
        return output

    def forward_rl(self, inpt_ids, max_len):
        """
        Train Generator with RL
        inpt_ids: [batch, seq]
        return actions and corresponding probs:
            torch tensor object: [batch, max_len], [batch, max_len]; [batch]
        """
        self.generator.train()
        generated, probs = [], []
        rewards = []
        inpt_ids_ = inpt_ids.clone().detach()
        sep_batch_index = torch.LongTensor([0] * len(inpt_ids))
        sep_batch_flag = [0] * len(inpt_ids)
        prev, past, history = inpt_ids_, None, inpt_ids_
        for i in range(1, max_len + 1):
            outputs = self.generator(input_ids=prev, past=past)
            outputs, past = outputs[:2]
            next_token_logits = outputs[:, -1, :]
            next_token_logits[:, self.unk_id] = -np.inf
            filtered_logits = top_k_top_p_filtering_batch(next_token_logits, top_k=self.topk, top_p=self.topp, min_token_to_keep=self.min_token_to_keep)
            filtered_logits = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(filtered_logits, num_samples=1)
            next_token = next_token.squeeze(1)
            for idx, token_ in enumerate(next_token):
                if sep_batch_flag[idx] == 0 and token_ == self.sep_id:
                    sep_batch_index[idx] = i
                    sep_batch_flag[idx] = 1
            prob = filtered_logits[range(len(next_token)), next_token]
            generated.append(next_token)
            probs.append(prob)
            prev = next_token.unsqueeze(1)
            if i < max_len:
                rest = self.rollout_batch(past, next_token.unsqueeze(1), max_len - i)
                generated_ = torch.stack(generated).transpose(0, 1)
                rest = [torch.cat((inpt_ids, generated_, rollout_item), dim=1) for rollout_item in rest]
            else:
                for idx in range(len(sep_batch_index)):
                    if sep_batch_flag[idx] == 0:
                        sep_batch_index[idx] = max_len
                generated_ = torch.stack(generated).transpose(0, 1)
                for idx, index in enumerate(sep_batch_index):
                    generated_[idx][index:] = self.pad_id
                rest = [torch.cat((inpt_ids, generated_), dim=1)]
            rewards.append(self.obtain_rewards(rest))
        rewards = torch.stack(rewards).transpose(0, 1)
        probs = torch.stack(probs).transpose(0, 1)
        sep_batch_index = sep_batch_index
        return_data = {'probs': probs, 'rewards': rewards, 'sep_batch_index': sep_batch_index}
        return return_data

    def forward(self, cid, rid=None, max_size=None, mode='gen_rl'):
        """
        Compatible for DataParallel
        """
        if mode == 'gen_rl':
            assert max_size, f'[!] max_size must not be NoneType for gen_rl mode'
            data = self.forward_rl(cid, max_size)
            return data
        elif mode == 'gen_mle':
            if rid is None:
                raise Exception('[!] rid must not be NoneType for gen_mle mode')
            cid = torch.cat((cid, rid[:, 1:]), dim=1)
            shift_logits = self.forward_mle(cid)
            return shift_logits
        elif mode == 'dis':
            if rid:
                cid = torch.cat((cid, rid), dim=1)
            output = self.forward_dis(cid)
            return output
        elif mode == 'target_dis':
            if rid is None:
                raise Exception('[!] rid must not be NoneType for gen_mle mode')
            cid = torch.cat((cid, rid), dim=1)
            output = self.forward_target_dis(cid)
            return output
        elif mode == 'gen_predict':
            f_rid = self.generative_predict(cid, max_size)
            return f_rid
        else:
            raise Exception(f'[!] Except to get [gen_rl; gen_mle; dis] mode, but got {mode}')


class GPT2Retrieval(nn.Module):

    def __init__(self, vocab_size, unk_id, sep_id, topk, topp, config_path='data/config/model_config_dialogue_small.json'):
        super(GPT2Retrieval, self).__init__()
        self.model_config = GPT2Config.from_json_file(config_path)
        self.model = GPT2Model(config=self.model_config)
        self.model.resize_token_embeddings(vocab_size)
        self.n_ctx = self.model.config.to_dict().get('n_ctx')
        self.topk, self.topp = topk, topp
        self.unk_id = unk_id
        self.sep_id = sep_id
        self.lm_head = nn.Linear(768 + 300, vocab_size)

    def forward(self, inpt_ids, candidates):
        """
        inpt_ids: [batch, seq]
        candidates: [batch, 300], default k:=2
        """
        inpt_seq_size = inpt_ids.shape[1]
        attn_mask = generate_attention_mask(inpt_ids)
        outputs = self.model(input_ids=inpt_ids, attention_mask=attn_mask)
        output = outputs[0]
        rest = candidates.float()
        """
        with torch.no_grad():
            rest = []
            for candidate in candidates:
                attn_mask = generate_attention_mask(candidate)    # [batch, seq]
                candidate_output = self.model(
                        input_ids=inpt_ids,
                        attention_mask=attn_mask)[0]    # [batch, seq, hidden]
                # avoid the [PAD] token, use masked mean operation
                candidate_output = candidate_output * attn_mask.unsqueeze(-1).float()
                sum_candidate = torch.sum(candidate_output, dim=1)    # [batch, hidden]
                sum_mask = torch.sum(attn_mask, dim=1)    # [batch]
                candidate_output = sum_candidate / sum_mask.unsqueeze(-1).float()    # [batch, hidden]
                rest.append(candidate_output)
        rest = torch.stack(rest).mean(dim=0)    # [k, batch, hidden] -> [batch, hidden]
        """
        rest = rest.view(rest.shape[0], 1, rest.shape[1]).expand(rest.shape[0], inpt_seq_size, rest.shape[1])
        output = torch.cat([output, rest], dim=2)
        output = self.lm_head(output)
        return output

    def predict(self, inpt_ids, candidates, max_len):
        """
        batch_size is 1
        inpt_ids: [seq]
        candidates: k*[seq]

        return a list of ids (generated)
        no pad, do not need attention_mask
        """
        with torch.no_grad():
            """
            rest = []
            for candidate in candidates:
                attn_mask = generate_attention_mask(candidate)
                candidate_output = self.model(
                        input_ids=inpt_ids,
                        attention_mask=attn_mask)[0]    # [seq, hidden]
                # avoid the [PAD] token, use masked mean operation
                candidate_output = candidate_output * attn_mask.unsqueeze(-1).float()
                sum_candidate = torch.sum(candidate_output, dim=1)    # [batch, hidden]
                sum_mask = torch.sum(attn_mask, dim=1)    # [batch]
                candidate_output = sum_candidate / sum_mask.unsqueeze(-1).float()    # [batch, hidden]
                rest.append(candidate_output)
            rest = torch.stack(rest).mean(dim=0)    # [hidden]
            """
            rest = candidates.float()
            generated, past = [], None
            for _ in range(max_len):
                outputs = self.model(input_ids=inpt_ids, past=past)
                outputs, past = outputs[:2]
                next_token_logits = outputs[-1, :]
                next_token_logits = torch.cat((next_token_logits, rest))
                next_token_logits = self.lm_head(next_token_logits)
                next_token_logits[self.unk_id] = -np.inf
                filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=self.topk, top_p=self.topp)
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                if next_token == self.sep_id:
                    break
                generated.append(next_token.item())
                inpt_ids = next_token
            return generated


class ActorCritic(nn.Module):

    def __init__(self, policy_size, embedding_size, action_std=0.5):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(nn.Linear(embedding_size * 2, embedding_size), nn.Tanh(), nn.Linear(embedding_size, policy_size), nn.Tanh())
        self.critic = nn.Sequential(nn.Linear(embedding_size * 2, embedding_size), nn.Tanh(), nn.Linear(embedding_size, 1))
        self.action_var = torch.full((policy_size,), action_std * action_std)
        if torch.cuda.is_available():
            self.action_var = self.action_var

    def forward(self, embedding):
        """only called by gpt2v2, compatible with gpt2v2 model; only the actor is used and critic is ignored;
        :embedding: [B, E*2]"""
        return self.actor(embedding)

    def act(self, state, memory=None):
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var)
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        if memory:
            memory.states.append(state)
            memory.actions.append(action)
            memory.logprobs.append(action_logprob)
        return action.detach()

    def evaluate(self, state, action):
        action_mean = self.actor(state)
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var)
        dist = MultivariateNormal(action_mean, cov_mat)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)
        return action_logprobs, torch.squeeze(state_value), dist_entropy


class GPT2V2(nn.Module):

    def __init__(self, vocab_size, unk_id, sep_id, cls_id, topk, topp, repetition_penalty, config_path='data/config/model_config_dialogue_small.json', embedding_size=300, policy_size=32):
        super(GPT2V2, self).__init__()
        self.model_config = GPT2Config.from_json_file(config_path)
        self.model = GPT2Model(config=self.model_config)
        self.model.resize_token_embeddings(vocab_size)
        self.n_ctx = self.model.config.to_dict().get('n_ctx')
        self.n_embd = self.model.config.to_dict().get('n_embd')
        self.topk, self.topp = topk, topp
        self.unk_id = unk_id
        self.sep_id = sep_id
        self.cls_id = cls_id
        self.repetition_penalty = repetition_penalty
        self.agent = ActorCritic(policy_size, embedding_size, action_std=0.1)
        self.proj = nn.Linear(self.n_embd + policy_size, vocab_size)

    def forward(self, inpt_ids, context_embd, response_embd):
        outputs = self.model(input_ids=inpt_ids)
        output = outputs[0]
        policy_embd = self.agent(torch.cat([context_embd, response_embd], dim=-1)).unsqueeze(1).expand(-1, output.shape[1], -1)
        output = torch.cat([output, policy_embd], dim=-1)
        output = self.proj(output)
        return output

    @torch.no_grad()
    def predict_batch(self, inpt_ids, attn_mask, position_ids, context_embd, response_embd, max_len):
        """past parameter and position_ids parameters should be careful
        https://github.com/huggingface/transformers/issues/3021#issuecomment-681792104"""
        batch_size = inpt_ids.shape[0]
        generated = [[self.cls_id] * batch_size]
        prev, past = inpt_ids, None
        stop_flag = np.zeros(batch_size)
        policy_embd = self.agent(torch.cat([context_embd, response_embd], dim=-1))
        for _ in range(max_len):
            outputs = self.model(input_ids=prev, attention_mask=attn_mask, position_ids=position_ids, past=past)
            output, past = outputs[:2]
            output = output[:, -1, :]
            output = torch.cat([output, policy_embd], dim=-1)
            next_token_logits = self.proj(output)
            next_token_logits[:, self.unk_id] = -np.inf
            for x in range(batch_size):
                y = [item[x] for item in generated]
                next_token_logits[x, y] /= self.repetition_penalty
            filtered_logits = top_k_top_p_filtering_batch(next_token_logits, top_k=self.topk, top_p=self.topp)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            for idx, i in enumerate(next_token.squeeze(1)):
                if i == self.sep_id:
                    stop_flag[idx] = 1
            generated.append([token.item() for token in next_token.squeeze(1)])
            prev = next_token
            if sum(stop_flag) == batch_size:
                break
            attn_mask = torch.cat([attn_mask, torch.tensor([1] * batch_size).unsqueeze(1)], dim=1)
            if past:
                position_ids = attn_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attn_mask == 0, 0)
                position_ids = position_ids[:, -1].unsqueeze(-1)
        ng, batch_size = [], len(generated[0])
        for i in range(batch_size):
            ng.append([g[i] for g in generated])
        return ng

    @torch.no_grad()
    def predict(self, inpt_ids, context_embd, response_embd, max_len):
        """inpt_ids: [seq]; return a list of ids (generated)"""
        generated = [self.cls_id]
        policy_embd = self.agent(torch.cat([context_embd, response_embd], dim=-1))
        for _ in range(max_len):
            outputs = self.model(input_ids=inpt_ids)
            output = outputs[0][-1, :]
            output = torch.cat([output, policy_embd], dim=-1)
            next_token_logits = self.proj(output)
            next_token_logits[self.unk_id] = -np.inf
            if generated:
                next_token_logits[list(set(generated))] /= self.repetition_penalty
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=self.topk, top_p=self.topp)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated.append(next_token.item())
            if next_token.item() == self.sep_id:
                break
            inpt_ids = torch.cat((inpt_ids, next_token), dim=0)
            inpt_ids = inpt_ids[-self.n_ctx:]
        return generated


class GPT2V2RL(nn.Module):

    def __init__(self, vocab_size, unk_id, sep_id, cls_id, pad_id, topk, topp, repetition_penalty, config_path='data/config/model_config_dialogue_small.json', embedding_size=300, policy_size=32, action_std=0.5, k_epochs=10, eps_clip=0.2):
        super(GPT2V2RL, self).__init__()
        self.model_config = GPT2Config.from_json_file(config_path)
        self.model = GPT2Model(config=self.model_config)
        self.model.resize_token_embeddings(vocab_size)
        self.n_ctx = self.model.config.to_dict().get('n_ctx')
        self.n_embd = self.model.config.to_dict().get('n_embd')
        self.topk, self.topp = topk, topp
        self.unk_id = unk_id
        self.sep_id = sep_id
        self.cls_id = cls_id
        self.pad_id = pad_id
        self.action_std, self.eps_clip = action_std, eps_clip
        self.k_epoch, self.embedding_size, self.policy_size = k_epochs, embedding_size, policy_size
        self.repetition_penalty = repetition_penalty
        self.agent = ActorCritic(policy_size, embedding_size, action_std=action_std)
        self.proj = nn.Linear(self.n_embd + policy_size, vocab_size)

    def update(self, memory, criterion, optimizer):
        """update the policy network; return the loss"""
        rewards = torch.tensor(memory.rewards, dtype=torch.float)
        rewards = to_cuda((rewards - rewards.mean()) / (rewards.std() + 1e-05))
        old_states = to_cuda(torch.stack(memory.states)).detach()
        old_actions = to_cuda(torch.stack(memory.actions)).detach()
        old_logprobs = to_cuda(torch.stack(memory.logprobs)).detach()
        losses = []
        for _ in range(self.k_epoch):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            ratios = torch.exp(logprobs - old_logprobs)
            advatanges = rewards - state_values.detach()
            surr1 = ratios * advatanges
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advatanges
            loss = -torch.min(surr1, surr2) - 0.01 * dist_entropy
            loss += 0.5 * criterion(state_values, rewards)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()
        return np.mean(losses)

    @torch.no_grad()
    def predict_batch(self, inpt_ids, attn_mask, position_ids, context_embd, response_embd, max_len):
        """past parameter and position_ids parameters should be careful
        https://github.com/huggingface/transformers/issues/3021#issuecomment-681792104"""
        batch_size = inpt_ids.shape[0]
        generated = [[self.cls_id] * batch_size]
        prev, past = inpt_ids, None
        stop_flag = np.zeros(batch_size)
        policy_embd = self.agent.act(torch.cat([context_embd, response_embd], dim=-1))
        for _ in range(max_len):
            outputs = self.model(input_ids=prev, attention_mask=attn_mask, position_ids=position_ids, past=past)
            output, past = outputs[:2]
            output = output[:, -1, :]
            output = torch.cat([output, policy_embd], dim=-1)
            next_token_logits = self.proj(output)
            next_token_logits[:, self.unk_id] = -np.inf
            for x in range(batch_size):
                y = [item[x] for item in generated]
                next_token_logits[x, y] /= self.repetition_penalty
            filtered_logits = top_k_top_p_filtering_batch(next_token_logits, top_k=self.topk, top_p=self.topp)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            for idx, i in enumerate(next_token.squeeze(1)):
                if i.item() == self.sep_id:
                    stop_flag[idx] = 1
            generated.append([token.item() for token in next_token.squeeze(1)])
            prev = next_token
            if sum(stop_flag) == batch_size:
                break
            attn_mask = torch.cat([attn_mask, torch.tensor([1] * batch_size).unsqueeze(1)], dim=1)
            if past:
                position_ids = attn_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attn_mask == 0, 0)
                position_ids = position_ids[:, -1].unsqueeze(-1)
        ng, batch_size = [], len(generated[0])
        for i in range(batch_size):
            p, flag = [], False
            for g in generated:
                if flag:
                    p.append(self.pad_id)
                    continue
                if g[i] == self.sep_id:
                    flag = True
                p.append(g[i])
            ng.append(p)
        return ng


class HashBERTBiEncoderModel(nn.Module):
    """Joint learn the hsahing code and the contextual embedding:
    1. hashing module is the regularization for the bi-encoder module,
    2. bi-encoding module can bring better performance for the hashing code."""

    def __init__(self, hidden_size, hash_code_size, dropout=0.3, lang='lang'):
        super(HashBERTBiEncoderModel, self).__init__()
        self.hash_code_size = hash_code_size
        self.encoder = BERTBiEncoder(lang=lang)
        self.ctx_hash_encoder = nn.Sequential(nn.Linear(768, hidden_size), nn.LeakyReLU(), nn.Dropout(p=dropout), nn.Linear(hidden_size, hash_code_size))
        self.ctx_hash_decoder = nn.Sequential(nn.Linear(hash_code_size, hidden_size), nn.LeakyReLU(), nn.Dropout(p=dropout), nn.Linear(hidden_size, 768))
        self.can_hash_encoder = nn.Sequential(nn.Linear(768, hidden_size), nn.LeakyReLU(), nn.Dropout(p=dropout), nn.Linear(hidden_size, hash_code_size))
        self.can_hash_decoder = nn.Sequential(nn.Linear(hash_code_size, hidden_size), nn.LeakyReLU(), nn.Dropout(p=dropout), nn.Linear(hidden_size, 768))

    def load_encoder(self, path):
        state_dict = torch.load(path)
        self.encoder.load_state_dict(state_dict)
        None

    @torch.no_grad()
    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_rep = self.encoder.ctx_encoder(cid, cid_mask)
        rid_rep = self.encoder.can_encoder(rid, rid_mask)
        return cid_rep, rid_rep

    @torch.no_grad()
    def predict(self, cid, rid, rid_mask):
        batch_size = rid.shape[0]
        cid_rep, rid_rep = self._encode(cid.unsqueeze(0), rid, None, rid_mask)
        cid_rep = cid.squeeze(0)
        ctx_hash_code = torch.sign(self.ctx_hash_encoder(cid_rep))
        can_hash_code = torch.sign(self.can_hash_encoder(rid_rep))
        matrix = torch.matmul(cid_hash_code, can_hash_code.t())
        distance = 0.5 * (self.hash_code_size - matrix)
        return distance

    def forward(self, cid, rid, cid_mask, rid_mask):
        """do we need the dot production loss? In my opinion, the hash loss is the replaction of the dot production loss. But need the experiment results to show it."""
        batch_size = cid.shape[0]
        assert batch_size > 1, f'[!] batch size must bigger than 1, cause other elements in the batch will be seen as the negative samples'
        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
        ctx_hash_code = self.ctx_hash_encoder(cid_rep)
        can_hash_code = self.can_hash_encoder(rid_rep)
        cid_rep_recons = self.ctx_hash_decoder(ctx_hash_code)
        rid_rep_recons = self.can_hash_decoder(can_hash_code)
        preserved_loss = torch.norm(cid_rep_recons - cid_rep, p=2, dim=1).mean() + torch.norm(rid_rep_recons - rid_rep, p=2, dim=1).mean()
        ctx_hash_code_h, can_hash_code_h = torch.sign(ctx_hash_code), torch.sign(can_hash_code)
        quantization_loss = torch.norm(ctx_hash_code - ctx_hash_code_h, p=2, dim=1).mean() + torch.norm(can_hash_code - can_hash_code_h, p=2, dim=1).mean()
        matrix = torch.matmul(ctx_hash_code, can_hash_code.t())
        mask = to_cuda(torch.eye(batch_size)).half()
        one_matrix = torch.ones_like(mask)
        zero_matrix = torch.zeros_like(mask)
        mask = torch.where(mask == 0, one_matrix, zero_matrix)
        hamming_distance = 0.5 * (self.hash_code_size - matrix)
        hash_loss = torch.norm(matrix - self.hash_code_size * mask, p=2).mean()
        acc_num = (torch.softmax(hamming_distance, dim=-1).min(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size))).sum().item()
        acc = acc_num / batch_size
        loss = preserved_loss + quantization_loss + hash_loss
        return loss, acc, (preserved_loss, quantization_loss, hash_loss)


class KWGPT2(nn.Module):

    def __init__(self, vocab_size, unk_id, sep_id, stp_id, topk, topp, repetition_penalty, config_path='data/config/model_config_dialogue_small.json'):
        super(KWGPT2, self).__init__()
        self.model_config = GPT2Config.from_json_file(config_path)
        self.model = GPT2LMHeadModel(config=self.model_config)
        self.model.resize_token_embeddings(vocab_size)
        self.n_ctx = self.model.config.to_dict().get('n_ctx')
        self.topk, self.topp = topk, topp
        self.unk_id = unk_id
        self.sep_id = sep_id
        self.stp_id = stp_id
        self.repetition_penalty = repetition_penalty

    def forward(self, inpt_ids):
        attn_mask = generate_attention_mask(inpt_ids)
        outputs = self.model(input_ids=inpt_ids, attention_mask=attn_mask)
        output = outputs[0]
        return output

    def predict(self, inpt_ids, max_len):
        """
        batch_size is 1
        inpt_ids: [seq]
        return a list of ids (generated)
        no pad, do not need attention_mask
        """
        with torch.no_grad():
            generated = []
            stp_counter = 0
            for _ in range(max_len):
                outputs = self.model(input_ids=inpt_ids)
                next_token_logits = outputs[0][-1, :]
                next_token_logits[self.unk_id] = -np.inf
                if generated:
                    next_token_logits[list(set(generated))] /= self.repetition_penalty
                filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=self.topk, top_p=self.topp)
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                if next_token == self.stp_id:
                    stp_counter += 1
                    if stp_counter == 2:
                        break
                generated.append(next_token.item())
                inpt_ids = torch.cat((inpt_ids, next_token), dim=0)
                inpt_ids = inpt_ids[-self.n_ctx:]
            return generated

    @torch.no_grad()
    def predict_batch(self, inpt_ids, max_len):
        """
        inpt_ids: [batch, seq]
        """
        generated = []
        prev, past = inpt_ids, None
        batch_size = inpt_ids.shape[0]
        for _ in range(max_len):
            outputs = self.model(input_ids=prev, past=past)
            output, past = outputs[:2]
            next_token_logits = output[:, -1, :]
            next_token_logits[:, self.unk_id] = -np.inf
            for x in range(batch_size):
                y = [itme[x] for item in generated]
                next_token_logits[x, y] /= self.repetition_penalty
            filtered_logits = top_k_top_p_filtering_batch(next_token_logits, top_k=self.topk, top_p=self.topp)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated.append([token.item() for token in next_token.squeeze(1)])
            prev = next_token
        ng, batch_size = [], len(generated[0])
        for i in range(batch_size):
            ng.append([g[i] for g in generated])
        return ng


class LCCC(nn.Module):

    def __init__(self, pretrained_path, topk, topp):
        super(LCCC, self).__init__()
        self.model = OpenAIGPTLMHeadModel.from_pretrained(pretrained_path)
        self.vocab = BertTokenizer.from_pretrained(pretrained_path, do_lower_case=True)
        self.topk, self.topp = topk, topp
        self.SPECIAL_TOKENS = ['[CLS]', '[SEP]', '[speaker1]', '[speaker2]']

    def forward(self, inpt_ids, token_type_ids):
        output = self.model(inpt_ids, token_type_ids=token_type_ids)[0]
        return output

    def build_input_from_segments(self, history, response, with_eos=True):
        """borrow from the thu-coai/CDial-GPT"""
        bos, eos, speaker1, speaker2 = self.vocab.convert_tokens_to_ids(self.SPECIAL_TOKENS)
        sequence = [[bos]] + history + [response + ([eos] if with_eos else [])]
        sequence = [sequence[0]] + [([speaker2 if i % 2 else speaker1] + s) for i, s in enumerate(sequence[1:])]
        instance = {}
        instance['input_ids'] = list(chain(*sequence))
        instance['token_type_ids'] = [bos] + [(speaker2 if i % 2 else speaker1) for i, s in enumerate(sequence[1:]) for _ in s]
        return instance

    def build_input_from_segments_batch(self, history, response, with_eos=True):
        instances = [self.build_input_from_segments([h], r, with_eos=with_eos) for h, r in zip(history, response)]
        return instances

    @torch.no_grad()
    def predict(self, inpt_ids, max_len, temperature=0.7, min_length=1):
        """batch_size is 1
        inpt_ids: without [CLS] and [SEP] token
        """
        current_output = []
        special_tokens_ids = self.vocab.convert_tokens_to_ids(self.SPECIAL_TOKENS)
        for i in range(max_len):
            instance = self.build_input_from_segments(inpt_ids, current_output, with_eos=False)
            input_ids = to_cuda(torch.LongTensor(instance['input_ids'])).unsqueeze(0)
            token_type_ids = to_cuda(torch.LongTensor(instance['token_type_ids'])).unsqueeze(0)
            logits, *_ = self.model(input_ids, token_type_ids=token_type_ids)
            logits = logits[0, -1, :] / temperature
            logits = top_k_top_p_filtering(logits, top_k=self.topk, top_p=self.topp)
            probs = F.softmax(logits, dim=-1)
            prev = torch.multinomial(probs, 1)
            if i < min_length and prev.item() in special_tokens_ids:
                while prev.item() in special_tokens_ids:
                    prev = torch.multinomial(probs, num_samples=1)
            if prev.item() in special_tokens_ids:
                break
            current_output.append(prev.item())
        return current_output

    @torch.no_grad()
    def predict_batch(self, inpt_ids, max_len, temperature=0.7):
        """batch size is not 1; but the length must be the same.
        .predict_batch can speed up the testing and provide the api for generation rerank
        inpt_ids: list of input_ids (length is Batch size)
        
        return: 
        current_output: [B, S] TYPE IS LIST
        """
        current_output = [[] for _ in range(len(inpt_ids))]
        stop_flag = [0] * len(inpt_ids)
        special_tokens_ids = self.vocab.convert_tokens_to_ids(self.SPECIAL_TOKENS)
        for i in range(max_len):
            instances = self.build_input_from_segments_batch(inpt_ids, current_output, with_eos=False)
            input_ids = to_cuda(torch.LongTensor([instance['input_ids'] for instance in instances]))
            token_type_ids = to_cuda(torch.LongTensor([instance['token_type_ids'] for instance in instances]))
            logits, *_ = self.model(input_ids, token_type_ids=token_type_ids)
            logits = logits[:, -1, :] / temperature
            logits = top_k_top_p_filtering_batch(logits, top_k=self.topk, top_p=self.topp)
            probs = F.softmax(logits, dim=-1)
            prev = torch.multinomial(probs, num_samples=1).squeeze(1).tolist()
            for idx, item in enumerate(prev):
                if item in special_tokens_ids:
                    stop_flag[idx] = 1
                current_output[idx].append(item)
            if sum(stop_flag) == len(stop_flag):
                break
        return current_output


class LCCCIR(nn.Module):

    def __init__(self, pretrained_path, topk, topp):
        super(LCCCIR, self).__init__()
        self.model = OpenAIGPTLMHeadModel.from_pretrained(pretrained_path)
        self.bs_head = nn.Linear(self.model.config.n_embd, 2)
        self.vocab = BertTokenizer.from_pretrained(pretrained_path, do_lower_case=True)
        self.topk, self.topp = topk, topp
        self.SPECIAL_TOKENS = ['[CLS]', '[SEP]', '[speaker1]', '[speaker2]']

    def build_input_from_segments(self, history, response, with_eos=True):
        """borrow from the thu-coai/CDial-GPT"""
        bos, eos, speaker1, speaker2 = self.vocab.convert_tokens_to_ids(self.SPECIAL_TOKENS)
        sequence = [[bos]] + history + [response + ([eos] if with_eos else [])]
        sequence = [sequence[0]] + [([speaker2 if i % 2 else speaker1] + s) for i, s in enumerate(sequence[1:])]
        instance = {}
        instance['input_ids'] = list(chain(*sequence))
        instance['token_type_ids'] = [bos] + [(speaker2 if i % 2 else speaker1) for i, s in enumerate(sequence[1:]) for _ in s]
        return instance

    def forward(self, inpt_ids, token_type_ids):
        """fine tuning process
        inpt_ids: [B, S]
        token_type_ids: [B, S]
        
        Maybe multi-task:
        (1) Binary classification
        (2) LM
        """
        transformer_outputs = self.model.transformer(inpt_ids, attention_mask=None, token_type_ids=token_type_ids, position_ids=None, head_mask=None, inputs_embeds=None)
        hidden_states = transformer_outputs[0]
        bs_state = torch.mean(hidden_states, dim=1)
        bs_rest = self.bs_head(bs_state)
        return bs_rest

    @torch.no_grad()
    def predict(self, inpt_ids, max_len, temperature=0.7, min_length=1):
        """batch_size is 1
        inpt_ids: without [CLS] and [SEP] token
        """
        current_output = []
        special_tokens_ids = self.vocab.convert_tokens_to_ids(self.SPECIAL_TOKENS)
        for i in range(max_len):
            instance = self.build_input_from_segments(inpt_ids, current_output, with_eos=False)
            input_ids = to_cuda(torch.LongTensor(instance['input_ids'])).unsqueeze(0)
            token_type_ids = to_cuda(torch.LongTensor(instance['token_type_ids'])).unsqueeze(0)
            logits, *_ = self.model(input_ids, token_type_ids=token_type_ids)
            logits = logits[0, -1, :] / temperature
            logits = top_k_top_p_filtering(logits, top_k=self.topk, top_p=self.topp)
            probs = F.softmax(logits, dim=-1)
            prev = torch.multinomial(probs, 1)
            if i < min_length and prev.item() in special_tokens_ids:
                while prev.item() in special_tokens_ids:
                    prev = torch.multinomial(probs, num_samples=1)
            if prev.item() in special_tokens_ids:
                break
            current_output.append(prev.item())
        return current_output


class CEWithLabelSmoothing(nn.Module):

    def __init__(self, vocab_size, label_smoothing=0.1, ignore_index=-1, reduction='word_mean'):
        """Cross Entropy Loss with Label Smoothing
        
        Arguments:
            vocab_size {int} -- # of vocabulary in the target language
        
        Keyword Arguments:
            label_smoothing {float} -- label smoothing factor (default: {.1})
            ignore_index {int} -- index need to ignore when calculate the loss (default: {-1})
            reduction {str} -- value in {"word_mean", "sum"}, "word mean": compute word level average loss, "sum":total loss (default: {"word_mean"}) 
        """
        super(CEWithLabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.ignore_index = ignore_index
        self.confidence = 1.0 - label_smoothing
        self.label_smoothing = label_smoothing
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.vocab_size = vocab_size
        self._true_dist = None
        self._reduction = reduction

    def forward(self, logits, target):
        assert logits.size(1) == self.vocab_size, 'size mismatch! %d!=%d' % (logits.size(1), self.vocab_size)
        true_dist = logits.clone()
        true_dist.fill_(self.label_smoothing / (self.vocab_size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.ignore_index] = 0.0
        mask = (target != self.ignore_index).float().unsqueeze(1)
        true_dist = true_dist * mask
        loss = self.criterion(self.log_softmax(logits), true_dist)
        n_words = torch.sum(mask)
        self._true_dist = true_dist
        self._kl = loss
        self._n_words = n_words
        if self._reduction == 'word_mean':
            return loss / n_words
        elif self._reduction == 'sum':
            return loss
        else:
            raise ValueError


class IRHead(nn.Module):

    def __init__(self, hidden_size, dropout=0.5):
        super(IRHead, self).__init__()
        self.M = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.hidden_layer = nn.Linear(hidden_size * 2 + 1, hidden_size)
        self.opt_layer = nn.Linear(hidden_size, 2)
        self.hidden_drop = nn.Dropout(p=dropout)

    def forward(self, src_embed, tgt_embed):
        """
        src_embed: [batch, hidden]
        tgt_embed: [batch, hidden]

        return the score: [batch, 2]
        """
        src_hidden = src_embed.unsqueeze(1)
        tgt_hidden = tgt_embed.unsqueeze(2)
        score = torch.bmm(torch.matmul(src_hidden, self.M), tgt_hidden).squeeze(2)
        src_hidden = src_hidden.squeeze(1)
        tgt_hidden = tgt_hidden.squeeze(2)
        inpt = torch.cat([src_hidden, score, tgt_hidden], 1)
        inpt = self.hidden_drop(torch.tanh(self.hidden_layer(inpt)))
        score = self.opt_layer(inpt)
        return score


class PositionEmbedding(nn.Module):
    """
    Position embedding for self-attention
    refer: https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    d_model: word embedding size or output size of the self-attention blocks
    max_len: the max length of the input squeezec
    """

    def __init__(self, d_model, dropout=0.5, max_len=100):
        super(PositionEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class PFGPT2(nn.Module):

    def __init__(self, vocab_size, unk_id, sep_id, topk, topp, config_path='data/config/model_config_dialogue_small.json'):
        super(PFGPT2, self).__init__()
        self.model_config = GPT2Config.from_json_file(config_path)
        self.model = GPT2LMHeadModel(config=self.model_config)
        self.model.resize_token_embeddings(vocab_size)
        self.n_ctx = self.model.config.to_dict().get('n_ctx')
        self.topk, self.topp = topk, topp
        self.unk_id = unk_id
        self.sep_id = sep_id

    def forward(self, inpt_ids):
        attn_mask = generate_attention_mask(inpt_ids)
        outputs = self.model(input_ids=inpt_ids, attention_mask=attn_mask)
        output = outputs[0]
        return output

    def predict(self, inpt_ids, max_len):
        """
        batch_size is 1
        inpt_ids: [seq]
        return a list of ids (generated)
        no pad, do not need attention_mask
        """
        with torch.no_grad():
            generated = []
            for _ in range(max_len):
                outputs = self.model(input_ids=inpt_ids)
                next_token_logits = outputs[0][-1, :]
                next_token_logits[self.unk_id] = -np.inf
                filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=self.topk, top_p=self.topp)
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                if next_token == self.sep_id:
                    break
                generated.append(next_token.item())
                inpt_ids = torch.cat((inpt_ids, next_token), dim=0)
                inpt_ids = inpt_ids[-self.n_ctx:]
            return generated

    @torch.no_grad()
    def predict_batch(self, inpt_ids, max_len):
        """
        inpt_ids: [batch, seq]
        """
        generated = []
        prev, past = inpt_ids, None
        for _ in range(max_len):
            outputs = self.model(input_ids=prev, past=past)
            output, past = outputs[:2]
            next_token_logits = output[:, -1, :]
            next_token_logits[:, self.unk_id] = -np.inf
            filtered_logits = top_k_top_p_filtering_batch(next_token_logits, top_k=self.topk, top_p=self.topp)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated.append([token.item() for token in next_token.squeeze(1)])
            prev = next_token
        ng, batch_size = [], len(generated[0])
        for i in range(batch_size):
            ng.append([g[i] for g in generated])
        return ng


class PONE(nn.Module):

    def __init__(self, lang='zh'):
        super(PONE, self).__init__()
        model_name = 'bert-base-chinese' if lang == 'zh' else 'bert-base-uncased'
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=1)

    def forward_(self, inpt):
        """
        inpt: [batch, seq]
        """
        attn_mask = generate_attention_mask(inpt)
        output = self.model(input_ids=inpt, attention_mask=attn_mask)
        logits = output[0]
        return logits

    def forward(self, inpt):
        logits = self.forward_(inpt)
        logits = torch.sigmoid(logits.squeeze(1))
        return logits


class PONE_cls(nn.Module):

    def __init__(self, lang='zh'):
        super(PONE_cls, self).__init__()
        model_name = 'bert-base-chinese' if lang == 'zh' else 'bert-base-uncased'
        self.model = BertModel.from_pretrained(model_name)
        self.head = nn.Linear(768, 1)

    @torch.no_grad()
    def forward_(self, inpt):
        """
        inpt: [batch, seq]
        """
        attn_mask = generate_attention_mask(inpt)
        output = self.model(input_ids=inpt, attention_mask=attn_mask)[0]
        output = torch.mean(output, dim=1)
        return output

    def forward(self, inpt):
        logits = self.forward_(inpt)
        logits = self.head(logits).squeeze(1)
        logits = torch.sigmoid(logits)
        return logits


class PONE_bi(nn.Module):

    def __init__(self, lang='zh', dropout=0.5):
        super(PONE_bi, self).__init__()
        model_name = 'bert-base-chinese' if lang == 'zh' else 'bert-base-uncased'
        self.model = BertModel.from_pretrained(model_name)
        self.head = IRHead(768, dropout=dropout)

    @torch.no_grad()
    def forward_(self, inpt):
        attn_mask = generate_attention_mask(inpt)
        output = self.model(input_ids=inpt, attention_mask=attn_mask)[0]
        output = torch.mean(output, dim=1)
        return output

    def forward(self, src, tgt):
        src = self.forward_(src)
        tgt = self.forward_(tgt)
        score = self.head(src, tgt)
        return score


class RetrievalModel(nn.Module):

    def __init__(self, hidden_size, dropout=0.5):
        super(RetrievalModel, self).__init__()
        self.model = IRHead(hidden_size, dropout=dropout)

    def forward(self, src, tgt):
        score = self.model(src, tgt)
        return score


class Attention(nn.Module):

    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.randn(hidden_size))
        stdv = 1.0 / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, context):
        """
        hidden: [batch, hidden_size]
        context: [seq, batch, hidden_size]

        return the context vector for decoding: [batch, hidden]
        """
        timestep = context.shape[0]
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1)
        context = context.transpose(0, 1)
        attn_energies = self.score(h, context)
        score = F.softmax(attn_energies, dim=1).unsqueeze(1)
        context = torch.bmm(score, context).squeeze(1)
        return context

    def score(self, hidden, context):
        """
        hidden: [batch, seq, hidden]
        context: [batch, seq, hidden]
        """
        energy = torch.tanh(self.attn(torch.cat([hidden, context], 2)))
        energy = energy.transpose(1, 2)
        v = self.v.repeat(context.shape[0], 1).unsqueeze(1)
        energy = torch.bmm(v, energy)
        return energy.squeeze(1)


class GRUEncoder(nn.Module):

    def __init__(self, embed_size, hidden_size, n_layers=1, dropout=0.5, bidirectional=True):
        super(GRUEncoder, self).__init__()
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(embed_size, hidden_size, num_layers=n_layers, dropout=0 if n_layers == 1 else dropout, bidirectional=bidirectional)
        self.times = n_layers * 2 if bidirectional else n_layers
        self.hidden_project = nn.Linear(self.times * hidden_size, hidden_size)
        self.init_weight()

    def init_weight(self):
        init.xavier_normal_(self.rnn.weight_hh_l0)
        init.xavier_normal_(self.rnn.weight_ih_l0)
        self.rnn.bias_ih_l0.data.fill_(0.0)
        self.rnn.bias_hh_l0.data.fill_(0.0)

    def forward(self, src, src_l):
        embed = nn.utils.rnn.pack_padded_sequence(src, src_l, enforce_sorted=False)
        output, hidden = self.rnn(embed)
        output, _ = nn.utils.rnn.pad_packed_sequence(output)
        if self.bidirectional:
            output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]
        hidden = hidden.permute(1, 2, 0)
        hidden = hidden.reshape(hidden.shape[0], -1)
        hidden = torch.tanh(self.hidden_project(hidden))
        return output, hidden


class GRUDecoder(nn.Module):

    def __init__(self, output_size, embed_size, hidden_size, n_layers=2, dropout=0.5):
        super(GRUDecoder, self).__init__()
        self.attention = Attention(hidden_size)
        self.rnn = nn.GRU(hidden_size + embed_size, hidden_size, num_layers=n_layers, dropout=0 if n_layers == 1 else dropout)
        self.opt_layer = nn.Linear(hidden_size * 2, output_size)
        self.init_weight()

    def init_weight(self):
        init.xavier_normal_(self.rnn.weight_hh_l0)
        init.xavier_normal_(self.rnn.weight_ih_l0)
        self.rnn.bias_ih_l0.data.fill_(0.0)
        self.rnn.bias_hh_l0.data.fill_(0.0)

    def forward(self, src, last_hidden, context):
        embed = src.unsqueeze(0)
        key = last_hidden.sum(axis=0)
        context_v = self.attention(key, context)
        context_v = context_v.unsqueeze(0)
        inpt = torch.cat([embed, context_v], 2)
        output, hidden = self.rnn(inpt, last_hidden)
        output = output.squeeze(0)
        output = torch.cat([output, context_v.squeeze(0)], 1)
        output = self.opt_layer(output)
        return output, hidden


class Seq2Seq(nn.Module):

    def __init__(self, vocab_size, embed_size, hidden_size, dropout=0.5, bidirectional=True, n_layers=1, cls=0, sep=0, unk=0):
        super(Seq2Seq, self).__init__()
        self.encoder = GRUEncoder(embed_size, hidden_size, n_layers=n_layers, dropout=dropout, bidirectional=bidirectional)
        self.decoder = GRUDecoder(vocab_size, embed_size, hidden_size, n_layers=n_layers, dropout=dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.vocab_size = vocab_size
        self.n_layer = n_layers
        self.cls = cls
        self.sep = sep
        self.unk = unk

    def forward(self, src, tgt, src_l):
        """src/tgt [seq, batch]; src_l: [batch]"""
        batch_size, max_len = src.shape[1], tgt.shape[0]
        final_opt = torch.zeros(max_len - 1, batch_size, self.vocab_size)
        src = self.embedding(src)
        context, hidden = self.encoder(src, src_l)
        hidden = hidden.repeat(self.n_layer, 1, 1)
        inpt = tgt[0]
        for t in range(1, max_len):
            inpt = self.embedding(inpt)
            output, hidden = self.decoder(inpt, hidden, context)
            final_opt[t - 1] = output
            inpt = tgt[t]
        return final_opt

    @torch.no_grad()
    def predict(self, src, src_l, max_len):
        batch_size = src.shape[1]
        final_opt = torch.zeros(max_len, batch_size, dtype=torch.long)
        src = self.embedding(src)
        context, hidden = self.encoder(src, src_l)
        hidden = hidden.repeat(self.n_layer, 1, 1)
        inpt = torch.zeros(batch_size, dtype=torch.long).fill_(self.cls)
        final_opt[0] = inpt
        stop_flag = [0] * batch_size
        for t in range(1, max_len):
            inpt = self.embedding(inpt)
            inpt, hidden = self.decoder(inpt, hidden, context)
            inpt[:, self.unk] = -np.inf
            next_token = torch.multinomial(F.softmax(inpt, dim=-1), num_samples=1).squeeze(1)
            final_opt[t] = next_token
            inpt = next_token
            for idx, item in enumerate(next_token):
                if stop_flag[idx] == 0 and item == self.sep:
                    stop_flag[idx] = 1
            if sum(stop_flag) == batch_size:
                break
        return final_opt


class Transformer(nn.Module):

    def __init__(self, n_vocab, d_model=512, n_head=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, max_len=512, share_word_embedding=False, pad=0):
        """
        Transformer Arguments:
            n_vocab {int} -- # of vocabulary
        
        Keyword Arguments:
            d_model {int} -- dimension of hidden state (default: {512})
            n_head {int} -- # of heads used in multi-head attention (default: {8})
            num_encoder_layers {int} -- # of transformer encoder layers (default: {6})
            num_decoder_layers {int} -- # of transformer decoder blocks (default: {6})
            dim_feedforward {int} -- dimension of hidden layer of position wise feed forward layer(default: {2048})
            dropout {float} -- dropout rate (default: {0.1})
            max_len {int} -- max input length (default: {512})
            share_word_embedding {bool} -- share word embedding between encoder and decoder
        """
        super(Transformer, self).__init__()
        self.n_vocab = n_vocab
        self.enc_word_embed = nn.Embedding(n_vocab, d_model, padding_idx=pad)
        self.pos_embed = PositionEmbedding(d_model, dropout=dropout, max_len=512)
        if share_word_embedding:
            self.dec_word_embed = self.enc_word_embed
        else:
            self.dec_word_embed = nn.Embedding(n_vocab, d_model, padding_idx=pad)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead=n_head, dim_feedforward=dim_feedforward, dropout=dropout)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead=n_head, dim_feedforward=dim_feedforward, dropout=dropout)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
        self.proj = nn.Linear(d_model, self.n_vocab)

    def forward(self, src, trg, src_mask=None, trg_mask=None, memory_mask=None, src_key_padding_mask=None, trg_key_padding_mask=None, memory_key_padding_mask=None):
        """forward computation for Transformer
        
        Arguments:
            src {torch.LongTensor} -- input mini-batch in shape (L_S, B)
            trg {torch.LongTensor} -- target mini-batch in shape (L_T, B)
        
        Keyword Arguments:
            src_turn {torch.LongTensor} -- turn ids in range (1, T) (default: {None})
        
        Returns:
            torch.Tensor -- logits in shape (L_T, V)
        """
        src_embed = self.pos_embed(self.enc_word_embed(src))
        trg_embed = self.pos_embed(self.dec_word_embed(trg))
        memory = self.encoder(src_embed, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(trg_embed, memory, tgt_mask=trg_mask, memory_mask=memory_mask, tgt_key_padding_mask=trg_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        logits = self.proj(output)
        return logits

    @torch.no_grad()
    def predict(self, src, src_mask=None, trg_mask=None, memory_mask=None, src_key_padding_mask=None, trg_key_padding_mask=None, memory_key_padding_mask=None, max_size=0, cls=0, sep=0, topk=0, topp=0.0):
        """return the trg_generated; shape of the returned tensor is [S, B]"""
        batch_size = src.size(1)
        stop_flag = [False] * batch_size
        src_embed = self.pos_embed(self.enc_word_embed(src))
        memory = self.encoder(src_embed, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        trg = torch.LongTensor([cls] * batch_size).unsqueeze(0)
        if torch.cuda.is_available():
            trg = trg
        for idx in range(1, max_size + 1):
            trg_embed = self.pos_embed(self.dec_word_embed(trg))
            trg_mask = nn.Transformer.generate_square_subsequent_mask(idx, idx)
            if torch.cuda.is_available():
                trg_mask = trg_mask
            output = self.decoder(trg_embed, memory, tgt_mask=trg_mask, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=memory_key_padding_mask)
            logits = self.proj(output[-1, :, :])
            logits = top_k_top_p_filtering_batch(logits, top_k=topk, top_p=topp)
            next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1).transpose(0, 1)
            trg = torch.cat([trg, next_token], dim=0)
            for idx, token_i in enumerate(next_token.squeeze(0)):
                if token_i == sep:
                    stop_flag[idx] = True
            if sum(stop_flag) == len(stop_flag):
                break
        return trg


class UNI(nn.Module):

    def __init__(self, config_path, vocab_path, topk, topp):
        super(UNI, self).__init__()
        self.model_config = GPT2Config.from_json_file(config_path)
        self.model = GPT2Model(config=self.model_config)
        self.vocab = BertTokenizer(vocab_file=vocab_path, do_lower_case=True)
        self.lm_head = nn.Linear(self.model_config.n_embd, self.model_config.vocab_size)
        self.topk, self.topp = topk, topp
        self.SPECIAL_TOKENS = ['[CLS]', '[SEP]', '[speaker1]', '[speaker2]']

    def forward(self, inpt_ids, token_type_ids):
        attn_mask = generate_attention_mask(inpt_ids)
        outputs = self.model(input_ids=inpt_ids, attention_mask=attn_mask, token_type_ids=token_type_ids)
        outputs = outputs[0]
        lm_outputs = self.lm_head(outputs)
        return lm_outputs

    def build_input_from_segments(self, history, response, with_eos=True):
        """borrow from the thu-coai/CDial-GPT; for test mode"""
        bos, eos, speaker1, speaker2 = self.vocab.convert_tokens_to_ids(self.SPECIAL_TOKENS)
        sequence = [[bos]] + history + [response + ([eos] if with_eos else [])]
        sequence = [sequence[0]] + [([speaker2 if i % 2 else speaker1] + s) for i, s in enumerate(sequence[1:])]
        instance = {}
        instance['input_ids'] = list(chain(*sequence))
        instance['token_type_ids'] = [bos] + [(speaker2 if i % 2 else speaker1) for i, s in enumerate(sequence[1:]) for _ in s]
        return instance

    @torch.no_grad()
    def gen(self, inpt_ids, token_type_ids, max_len=50, temperature=0.7):
        """batch size is 1"""
        current_output = []
        special_tokens_ids = self.vocab.convert_tokens_to_ids(self.SPECIAL_TOKENS)
        for i in range(max_len):
            instance = self.build_input_from_segments(inpt_ids, generated, with_eos=False)
            input_ids = torch.LongTensor(instance['input_ids']).unsqueeze(0)
            token_type_ids = to_cuda(torch.LongTensor(instance['token_type_ids'])).unsqueeze(0)
            attn_mask = generate_attention_mask(inpt_ids)
            outputs = self.model(input_ids=inpt_ids, attention_mask=attn_mask, token_type_ids=token_type_ids)
            outputs = outputs[0]
            logits = self.lm_head(outputs)[0, -1, :] / temperature
            logits = top_k_top_p_filtering(logits, top_k=self.topk, top_p=self.topp)
            probs = F.softmax(logits, dim=-1)
            prev = torch.multinomial(probs, 1)
            if i < min_length and prev.item() in special_tokens_ids:
                while prev.item() in special_tokens_ids:
                    prev = torch.multinomial(probs, num_samples=1)
            if prev.item() in special_tokens_ids:
                break
            current_output.append(prev.item())
        return current_output


class When2Talk(nn.Module):

    def __init__(self, vocab_size, unk_id, stp_id, topk, topp, repetition_penalty, config_path='data/config/model_config_dialogue_small.json'):
        super(When2Talk, self).__init__()
        self.model_config = GPT2Config.from_json_file(config_path)
        self.model = GPT2LMHeadModel(config=self.model_config)
        self.model.resize_token_embeddings(vocab_size)
        self.n_ctx = self.model.config.to_dict().get('n_ctx')
        self.topk, self.topp = topk, topp
        self.unk_id = unk_id
        self.stp_id = stp_id
        self.repetition_penalty = repetition_penalty

    def forward(self, inpt_ids):
        attn_mask = generate_attention_mask(inpt_ids)
        outputs = self.model(input_ids=inpt_ids, attention_mask=attn_mask)
        output = outputs[0]
        return output

    def predict(self, inpt_ids, max_len):
        """
        batch_size is 1; inpt_ids: [seq]
        the user token is [USER2]
        """
        with torch.no_grad():
            generated = []
            for _ in range(max_len):
                outputs = self.model(input_ids=inpt_ids)[0]
                next_token_logits = outputs[-1, :]
                if generated:
                    next_token_logits[list(set(generated))] /= self.repetition_penalty
                next_token_logits[self.unk_id] = -np.inf
                filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=self.topk, top_p=self.topp)
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                generated.append(next_token.item())
                if next_token.item() == self.stp_id:
                    break
                inpt_ids = torch.cat((inpt_ids, next_token), dim=0)
                inpt_ids = inpt_ids[-self.n_ctx:]
            return generated


class ESChat:
    """basic elasticsearch searcher"""

    def __init__(self, index_name, kb=True):
        self.es = Elasticsearch(http_auth=('elastic', 'elastic123'))
        self.index = index_name
        self.es.indices.put_settings(index=self.index, body={'index': {'max_result_window': 500000}})

    def must_search(self, query, samples=10, topic=None):
        """
        query is the string, which contains the utterances of the conversation context.
        1. topic is a list contains the topic words
        2. query utterance msg
        
        context: query is Q-Q matching
        response: query is Q-A matching, which seems better
        """
        query = query.replace('[SEP]', '')
        subitem_must = [{'match': {'utterance': {'query': i, 'boost': 1}}} for i in topic]
        subitem_should = [{'match': {'utterance': {'query': query, 'boost': 1}}}]
        dsl = {'query': {'bool': {'must': subitem_must, 'should': subitem_should}}, 'collapse': {'field': 'keyword'}}
        begin_samples, rest = samples, []
        hits = self.es.search(index=self.index, body=dsl, size=begin_samples)['hits']['hits']
        for h in hits:
            item = {'score': h['_score'], 'utterance': h['_source']['utterance']}
            if item['utterance'] in query or 'http' in item['utterance']:
                continue
            else:
                rest.append(item)
        return rest

    def search(self, query, samples=10, topic=None):
        """
        query is the string, which contains the utterances of the conversation context.
        1. topic is a list contains the topic words
        2. query utterance msg
        
        context: query is Q-Q matching
        response: query is Q-A matching, which seems better
        """
        query = query.replace('[SEP]', '')
        if not topic:
            dsl = {'query': {'match': {'utterance': query}}, 'collapse': {'field': 'keyword'}}
        else:
            subitem = [{'match': {'utterance': {'query': i, 'boost': 7}}} for i in topic]
            subitem.append({'match': {'utterance': {'query': query, 'boost': 1}}})
            dsl = {'query': {'bool': {'should': subitem}}, 'collapse': {'field': 'keyword'}}
        begin_samples, rest = samples, []
        while len(rest) == 0:
            hits = self.es.search(index=self.index, body=dsl, size=begin_samples)['hits']['hits']
            for h in hits:
                item = {'score': h['_score'], 'utterance': h['_source']['utterance']}
                if item['utterance'] in query or 'http' in item['utterance']:
                    continue
                else:
                    rest.append(item)
            begin_samples += 1
        return rest

    def multi_search(self, querys, samples=10):
        querys = [i[-150:] for i in querys]
        search_arr = []
        for query in querys:
            search_arr.append({'index': self.index})
            search_arr.append({'query': {'match': {'utterance': query}}, 'size': samples})
        request = ''
        for each in search_arr:
            request += f'{json.dumps(each)} \n'
        rest = self.es.msearch(body=request)
        return rest

    def talk(self, msgs, topic=None):
        rest = self.search(msgs, samples=1, topic=topic)[0]['utterance']
        return rest


class RetrievalBaseAgent:

    def __init__(self, searcher=True, kb=True):
        if searcher:
            self.searcher = ESChat('retrieval_database', kb=kb)
        self.history = []

    def show_parameters(self, args):
        None
        None
        None
        None
        for key, value in args.items():
            None
        None

    def save_model(self, path):
        try:
            state_dict = self.model.module.state_dict()
        except:
            state_dict = self.model.state_dict()
        torch.save(state_dict, path)
        None

    def load_model(self, path):
        """
        add the `module.` before the state_dict keys if the error are raised,
        which means that the DataParallel(self.model) are used to load the model
        """
        state_dict = torch.load(path)
        try:
            self.model.load_state_dict(state_dict)
        except:
            current_module = True if 'module' in [i[0] for i in self.model.state_dict().items()][0] else False
            saved_module = True if 'module' in [i[0] for i in state_dict.items()][0] else False
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if saved_module and not current_module:
                    name = k[7:]
                    new_state_dict[name] = v
                elif not saved_module and current_module:
                    name = f'module.{k}'
                    new_state_dict[name] = v
            self.model.load_state_dict(new_state_dict)
        None

    def train_model(self, train_iter, mode='train'):
        raise NotImplementedError

    def test_model(self, test_iter, path):
        raise NotImplementedError

    def process_utterances_biencoder(self, topic, msgs, max_len=0):

        def _length_limit(ids):
            if len(ids) > max_len:
                ids = [ids[0]] + ids[-(max_len - 1):]
            return ids
        utterances = self.searcher.search(msgs, samples=self.args['talk_samples'], topic=topic)
        utterances = [i['utterance'] for i in utterances]
        utterances = list(set(utterances) - set(self.history))
        inpt_ids = self.vocab.batch_encode_plus([msgs] + utterances)['input_ids']
        context_inpt_ids, response_inpt_ids = inpt_ids[0], inpt_ids[1:]
        context_inpt_ids = torch.LongTensor(_length_limit(context_inpt_ids))
        response_inpt_ids = [torch.LongTensor(_length_limit(i)) for i in response_inpt_ids]
        response_inpt_ids = pad_sequence(response_inpt_ids, batch_first=True, padding_value=self.args['pad'])
        attn_mask_index = response_inpt_ids.nonzero().tolist()
        attn_mask_index_x, attn_mask_index_y = [i[0] for i in attn_mask_index], [i[1] for i in attn_mask_index]
        attn_mask = torch.zeros_like(response_inpt_ids)
        attn_mask[attn_mask_index_x, attn_mask_index_y] = 1
        if torch.cuda.is_available():
            context_inpt_ids, response_inpt_ids, attn_mask = context_inpt_ids, response_inpt_ids, attn_mask
        return utterances, context_inpt_ids, response_inpt_ids, attn_mask

    def process_utterances(self, topic, msgs, max_len=0, context=True):
        """Process the utterances searched by Elasticsearch; input_ids/token_type_ids/attn_mask"""
        if not context:
            msgs = ''
        utterances_ = self.searcher.search(msgs, samples=self.args['talk_samples'], topic=topic)
        utterances_ = [i['utterance'] for i in utterances_]
        utterances_ = list(set(utterances_) - set(self.history))
        inpt_ids = self.vocab.batch_encode_plus([msgs] + utterances_)['input_ids']
        context_inpt_ids, responses_inpt_ids = inpt_ids[0], inpt_ids[1:]
        context_token_type_ids = [0] * len(context_inpt_ids)
        responses_token_type_ids = [([1] * len(i)) for i in responses_inpt_ids]
        collection = []
        for r1, r2 in zip(responses_inpt_ids, responses_token_type_ids):
            p1, p2 = context_inpt_ids + r1[1:], context_token_type_ids + r2[1:]
            if len(p1) > max_len:
                cut_size = len(p1) - max_len + 1
                p1, p2 = [p1[0]] + p1[cut_size:], [p2[0]] + p2[cut_size:]
            collection.append((p1, p2))
        inpt_ids = [torch.LongTensor(i[0]) for i in collection]
        token_type_ids = [torch.LongTensor(i[1]) for i in collection]
        inpt_ids = pad_sequence(inpt_ids, batch_first=True, padding_value=self.args['pad'])
        token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=self.args['pad'])
        attn_mask_index = inpt_ids.nonzero().tolist()
        attn_mask_index_x, attn_mask_index_y = [i[0] for i in attn_mask_index], [i[1] for i in attn_mask_index]
        attn_mask = torch.zeros_like(inpt_ids)
        attn_mask[attn_mask_index_x, attn_mask_index_y] = 1
        if torch.cuda.is_available():
            inpt_ids, token_type_ids, attn_mask = inpt_ids, token_type_ids, attn_mask
        return utterances_, inpt_ids, token_type_ids, attn_mask

    def talk(self, msgs, topic=None):
        """
        topic: topic of the conversation
        msgs: a string of the conversation context
        """
        raise NotImplementedError

    def get_res(self, data):
        """
        data = {
            "kg_path": group_id,
            "topic": topic,
            "node": robot_id
            "msgs": [
                {
                    'fromUser': robot_id,
                    'msg': msg,
                    'timestamp': timestamp
                },
                ...
            ]
        }
        """
        msgs = [i['msg'] for i in data['msgs']]
        msgs = ' [SEP] '.join(msgs)
        topic = data['topic'] if 'topic' in data else None
        res = self.talk(msgs, topic=topic)
        self.history.append(res)
        return res


class BERTMCF(RetrievalBaseAgent):

    def __init__(self):
        super(BERTMCF, self).__init__(searcher=False)
        self.vocab = BertTokenizer(vocab_file='data/vocab/vocab_small')
        self.model = BERTMCFusion()
        self.pad = 0
        if torch.cuda.is_available():
            self.model

    @torch.no_grad()
    def scores(self, msgs, groundtruths, resps):
        """
        msgs: {context}[SEP]{response}, a batch of the pair of context and response
        """
        msgs = [[f'{m} [SEP] {g}', f'{m} [SEP] {r}'] for m, g, r in zip(msgs, groundtruths, resps)]
        ids = []
        for i, j in msgs:
            ids.append(torch.LongTensor(self.vocab.encode(i)[-512:]))
            ids.append(torch.LongTensor(self.vocab.encode(j)[-512:]))
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        ids = torch.stack(ids.split(2))
        if torch.cuda.is_available():
            ids = ids
        output = self.model(ids)
        output = F.softmax(output, dim=-1)[:, 1]
        output = output.cpu().tolist()
        return output


class BERT_MULTIVIEW(RetrievalBaseAgent):

    def __init__(self):
        super(BERT_MULTIVIEW, self).__init__(searcher=False)
        self.vocab = BertTokenizer.from_pretrained('bert-base-chinese')
        self.model = BERTMULTIVIEW()
        self.pad = 0
        if torch.cuda.is_available():
            self.model

    @torch.no_grad()
    def scores(self, msgs, resps, details=False):
        """
        msgs: {context}[SEP]{response}, a batch of the pair of context and response
        default aggregation strategy is average, min or max are also needed
        """
        msgs = [f'{m} [SEP] {r}' for m, r in zip(msgs, resps)]
        ids = [torch.LongTensor(self.vocab.encode(i)[-512:]) for i in msgs]
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        if torch.cuda.is_available():
            ids = ids
        outputs = self.model(ids, aspect='null')
        outputs = [F.softmax(output, dim=-1)[:, 1] for output in outputs]
        if details:
            outputs = [output.cpu().tolist() for output in outputs]
            return outputs
        else:
            output = torch.stack(outputs).min(dim=0)[0]
            output = output.cpu().tolist()
            return output


class COHERENCE(RetrievalBaseAgent):
    """
    do not train it, just inference for scoring
    """

    def __init__(self):
        super(COHERENCE, self).__init__(searcher=False)
        self.vocab = BertTokenizer(vocab_file='data/vocab/vocab_small')
        self.model = BERTRetrieval()
        self.pad = 0
        if torch.cuda.is_available():
            self.model

    def reload_model(self, state_dict):
        self.model.load_state_dict(state_dict)
        None

    @torch.no_grad()
    def scores(self, msgs, resps):
        """
        msgs: {context}[SEP]{response}, a batch of the pair of context and response
        """
        msgs = [f'{m} [SEP] {r}' for m, r in zip(msgs, resps)]
        ids = [torch.LongTensor(self.vocab.encode(i)[-300:]) for i in msgs]
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        if torch.cuda.is_available():
            ids = ids
        output = self.model(ids)
        output = F.softmax(output, dim=-1)[:, 1]
        output = output.cpu().tolist()
        return output

    @torch.no_grad()
    def scores_(self, cid, rid):
        ipdb.set_trace()
        cid = torch.cat((cid, rid), dim=1)
        output = self.model(cid)
        output = F.softmax(output, dim=-1)[:, 1]
        output = output.cpu().tolist()
        return output


def cal_distinct(corpus):
    bigram_finder = BigramCollocationFinder.from_words(corpus)
    dist = FreqDist(corpus)
    try:
        bi_diversity = len(bigram_finder.ngram_fd) / bigram_finder.N
        uni_diversity = len(dist) / len(corpus)
        return (bi_diversity + uni_diversity) / 2
    except:
        return 0.0


class Distinct:
    """
    Micro-Distinct: instance level
    Macro-Distinct: corpus level (obtained dialog history)
    """

    def __init__(self):
        pass

    def filter(self, msg):
        msg = msg.replace('[SEP]', '')
        msg = list(jieba.cut(msg))
        return msg

    def make_corpus(self, batch):
        data = []
        for i in batch:
            data.extend(i)
        return data

    def _micro(self, r):
        return cal_distinct(r)

    def _macro(self, h):
        h = self.make_corpus(h)
        return cal_distinct(h)

    def scores(self, responses, history):
        """
        :response: a batch of response string
        :history: a batch of history string
        """
        r = [self.filter(i) for i in responses]
        micro_s = [self._micro(r_) for r_ in r]
        if history:
            h_ = [self.filter(i) for i in history]
            h = []
            for r_ in r:
                h.append(h_ + r_)
            macro_s = [self._macro(h_) for h_ in h]
            s = [((mi + ma) / 2) for mi, ma in zip(micro_s, macro_s)]
        else:
            s = micro_s
        return s


class BaseAgent:

    def __init__(self):
        self.history = []

    def show_parameters(self, args):
        None
        None
        None
        None
        for key, value in args.items():
            None
        None

    def save_model(self, path):
        """
        Only save the model (without the module. for DatatParallel)
        """
        try:
            state_dict = self.model.module.state_dict()
        except:
            state_dict = self.model.state_dict()
        torch.save(state_dict, path)
        None

    def load_model(self, path):
        """
        add the `module.` before the state_dict keys if the error are raised,
        which means that the DataParallel are used to load the model
        """
        state_dict = torch.load(path)
        try:
            self.model.load_state_dict(state_dict)
        except:
            current_module = True if 'module' in [i[0] for i in self.model.state_dict().items()][0] else False
            saved_module = True if 'module' in [i[0] for i in state_dict.items()][0] else False
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if saved_module and not current_module:
                    name = k[7:]
                    new_state_dict[name] = v
                elif not saved_module and current_module:
                    name = f'module.{k}'
                    new_state_dict[name] = v
                else:
                    pass
            self.model.load_state_dict(new_state_dict)
        None

    def train_model(self, train_iter, mode='train'):
        raise NotImplementedError

    def test_model(self, test_iter, path):
        raise NotImplementedError

    def talk(self, topic, msgs):
        """
        topic: topic of the conversation
        msgs: a string of the conversation context
        """
        raise NotImplementedError

    def get_res(self, data):
        """
        SMP-MCC 2020
        data = {
            "group_id": group_id,
            "topic": topic,
            "robot_id": robot_id
            "msgs": [
                {
                    'from_id': robot_id,
                    'msg': msg,
                    'timestamp': timestamp
                },
                ...
            ]
        }
        """
        msgs = [i['msg'] for i in data['msgs']]
        msgs = '[SEP]'.join(msgs)
        res = self.talk(msgs)
        self.history.append(res)
        return res


class LCCCLM(BaseAgent):

    def __init__(self, path, topk, topp, t=0.7):
        super(LCCCLM, self).__init__()
        self.model = OpenAIGPTLMHeadModel.from_pretrained(path)
        self.vocab = BertTokenizer.from_pretrained(path, do_lower_case=True)
        self.topk, self.topp = topk, topp
        self.SPECIAL_TOKENS = ['[CLS]', '[SEP]', '[speaker1]', '[speaker2]']
        self.temperature = t
        if torch.cuda.is_available():
            self.model

    def tokenize_(self, obj):
        """borrow from thu-coai/CDial-GPT"""
        return self.vocab.convert_tokens_to_ids(self.vocab.tokenize(obj))

    def build_input_from_segments(self, history, response, with_eos=True):
        """borrow from the thu-coai/CDial-GPT"""
        bos, eos, speaker1, speaker2 = self.vocab.convert_tokens_to_ids(self.SPECIAL_TOKENS)
        sequence = [[bos]] + history + [response + ([eos] if with_eos else [])]
        sequence = [sequence[0]] + [([speaker2 if i % 2 else speaker1] + s) for i, s in enumerate(sequence[1:])]
        instance = {}
        instance['input_ids'] = list(chain(*sequence))
        instance['token_type_ids'] = [bos] + [(speaker2 if i % 2 else speaker1) for i, s in enumerate(sequence[1:]) for _ in s]
        return instance

    def scores(self, msgs, resps, temperature=0.7):
        s = [self.score(m, r, temperature=temperature) for m, r in list(zip(msgs, resps))]
        return s

    @torch.no_grad()
    def score(self, msg, res, temperature=0.7, alpha=0.2):
        self.model.eval()
        inpt_ids = [self.tokenize_(msg)]
        opt_res = self.tokenize_(res)
        special_tokens_ids = self.vocab.convert_tokens_to_ids(self.SPECIAL_TOKENS)
        probs, current_output = [], []
        for res_idx in opt_res:
            instance = self.build_input_from_segments(inpt_ids, current_output, with_eos=False)
            input_ids = torch.LongTensor(instance['input_ids']).unsqueeze(0)
            token_type_ids = torch.LongTensor(instance['token_type_ids']).unsqueeze(0)
            logits, *_ = self.model(input_ids, token_type_ids=token_type_ids)
            logits = logits[0, -1, :] / temperature
            probs.append(F.softmax(logits, dim=-1)[res_idx].item())
            current_output.append(res_idx)
            if len(instance['input_ids']) >= 512:
                break
        rest = sum(np.log(probs))
        length_norm = (5 + len(opt_res)) ** alpha / (5 + 1) ** alpha
        rest /= length_norm
        return rest


class LOGIC(RetrievalBaseAgent):

    def __init__(self):
        super(LOGIC, self).__init__(searcher=False)
        self.vocab = BertTokenizer(vocab_file='data/vocab/vocab_small')
        self.model = BERTRetrieval()
        self.pad = 0
        if torch.cuda.is_available():
            self.model

    @torch.no_grad()
    def scores(self, msgs, resps):
        """
        msgs: {context} [SEP] {response}, a batch version[batch can be 1]
        ids: [batch, seq]/[1, seq](batch is 1)
        """
        msgs = [f'{m} [SEP] {r}' for m, r in zip(msgs, resps)]
        ids = [torch.LongTensor(self.vocab.encode(i)[-300:]) for i in msgs]
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        if torch.cuda.is_available():
            ids = ids
        output = self.model(ids)
        output = F.softmax(output, dim=-1)[:, 1]
        output = output.cpu().tolist()
        return output

    @torch.no_grad()
    def scores_(self, cid, rid):
        cid = torch.cat((cid, rid), dim=1)
        output = self.model(cid)
        output = F.softmax(output, dim=-1)[:, 1]
        output = output.cpu().tolist()
        return output


class Length:
    """
    Penalty for the short messages for the given conversations
    score = 1 - rac{1}{len(response)}
    """

    def __init__(self):
        self.weight_scores = {(0): 0, (1): 0, (2): 0.1, (3): 0.1, (4): 0.2, (5): 0.2, (6): 0.4, (7): 0.4, (8): 0.5, (9): 0.5, (10): 0.6, (11): 0.6, (12): 0.7, (13): 0.7, (14): 0.8, (15): 0.8, (16): 0.9, (17): 0.9}
        self.filter_tokens = [' ', ',', '。', '!', '?', '！', '？', ';', '.', '，', '#', '$', '*', '(', ')', '（', '）', '[', ']', '-', '+', '=', '\t', '\n', '{', '}']

    def _filter_l(self, s):
        s_ = []
        for i in list(s):
            if i not in self.filter_tokens:
                s_.append(i)
        return len(s)

    def _scores(self, l):
        if l > max(self.weight_scores.keys()):
            return 1.0
        else:
            return self.weight_scores[l]

    def scores(self, responses):
        response_length = [self._filter_l(i) for i in responses]
        scores = [self._scores(i) for i in response_length]
        return scores


class MMI(BaseAgent):
    """
    DialoGPT MMI Model for rerank the generated responses;
    In this work, we use it as a part of the multiview evaluation module

    The code in `GPT2-chitchat` use the loss to represent the MMI scores,
    but it is not good.
    For example, the generated responses is much longer than the original candidates,
    but actually it will obtain the very big loss (small score).

    So, the paper of DialoGPT mentioned that the MMI scores are obtained by 
    P(Source|Hypothesis), so in this code, we use the language model probability to do it.
    """

    def __init__(self):
        super(MMI, self).__init__()
        self.model_path = 'ckpt/train_generative/gpt2_mmi/best.pt'
        self.vocab = BertTokenizer(vocab_file='data/vocab/vocab_small')
        self.unk = self.vocab.convert_tokens_to_ids('[UNK]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.model = GPT2(len(self.vocab), self.unk, self.sep, len(self.vocab), 1.0, 1.0, config_path='data/config/model_config_dialogue.json')
        if torch.cuda.is_available():
            self.model

    def scores(self, sources, targets):
        s_ = []
        for s, t in zip(sources, targets):
            probs = self.score(s, t)
            probs = np.mean(probs)
            s_.append(probs)
        return s_

    def score(self, source, target):
        c_ids = self.vocab.encode(source)[1:]
        r_ids = self.vocab.encode(target)
        c_ids_l = len(c_ids)
        ids = r_ids + c_ids
        if len(ids) >= 300:
            return [0.2]
        ids = torch.LongTensor(ids)
        if torch.cuda.is_available():
            ids = ids
        output = self.model.model(input_ids=ids)[0]
        output = F.softmax(output, dim=-1)
        index_x = list(range(len(ids)))[-(c_ids_l + 1):-1]
        index_y = c_ids
        assert len(index_x) == len(index_y), f'[!] x and y must have the same length'
        probs = output[index_x, index_y].tolist()
        return probs


def load_corpus(path):
    cutter = thulac.thulac(seg_only=True)
    with open(path) as f:
        data = f.read().split('\n\n')
        data = [i.replace('\n', '') for i in data if i.strip()]
    None
    data_ = []
    for u in tqdm(data):
        u = [i[0] for i in cutter.cut(u)]
        u = ' '.join(u)
        data_.append(u)
    return data_


def load_stopwords():
    with open('stopwords.txt') as f:
        data = f.read().split('\n')
        data = [i for i in data if i.strip()]
    return data


def obtain_word_idf(corpus, sw=None):
    """
    data = [
        '我 来到 北京 清华大学',
        '他 来到 了 网易 杭研 大厦',
        '小明 硕士 毕业 于 中国 科学院',
        '我 爱 北京 天安门'
    ]
    """
    vectorizer = CountVectorizer(min_df=4, max_df=50000, stop_words=sw)
    data = vectorizer.fit_transform(corpus).toarray()
    idf_count = np.sum(data, axis=0) + 1
    whole_tokens = sum(idf_count)
    tf_count = idf_count / whole_tokens
    return vectorizer.get_feature_names(), idf_count, tf_count


class NIDF_TF:
    """
    Inverse Document Frequency for measuring the diversity: range from 0 to 1, 1 means very diverse and 0 means very non-diverse

    Refer to the paper (AAAI 2020):
    Learning from Easy to Complex: Adaptive Multi-curricula Learning for Neural Dialogue Generation
    """

    def __init__(self):
        self.args = {'corpus_path': 'data/zh50w/train_.txt', 'rest_path': 'ckpt/NIDF_TF/data.pkl', 'stopwords_path': 'data/stopwords.txt', 'stopwords': True, 'factor_tf': 0.5, 'factor_idf': 0.5}
        self.cutter = thulac.thulac(seg_only=True)
        if os.path.exists(self.args['rest_path']):
            self._load()
        else:
            if self.args['stopwords']:
                self.stopwords = load_stopwords(self.args['stopwords_path'])
            else:
                self.stopwords = None
            self._train()

    def _train(self):
        data = load_corpus(self.args['corpus_path'])
        self.whole_doc = len(data)
        self.words, self.idf_count, self.tf_count = obtain_word_idf(data)
        self.idf_count = np.log(self.whole_doc / self.idf_count)
        self.idf_max, self.idf_min = max(self.idf_count), min(self.idf_count)
        None
        with open(self.args['rest_path'], 'wb') as f:
            pickle.dump((self.words, self.whole_doc, self.idf_count, self.tf_count), f)

    def _load(self):
        with open(self.args['rest_path'], 'rb') as f:
            self.words, self.whole_doc, self.idf_count, self.tf_count = pickle.load(f)
        self.idf_max, self.idf_min = max(self.idf_count), min(self.idf_count)
        None

    def scores(self, responses, topk=3, tf=False):
        responses_ = []
        for i in responses:
            i = [j[0] for j in self.cutter.cut(i)]
            i = [j for j in i if j in self.words]
            responses_.append(i)
        scores = []
        delta = self.idf_max - self.idf_min
        for response in responses_:
            if tf:
                p_tf = []
                for w in response:
                    index = self.words.index(w)
                    ntf = self.tf_count[index]
                    p_tf.append(ntf)
                if len(p_tf) == 0:
                    p_tf = 0
                else:
                    p_tf = np.mean(p_tf)
            response = list(set(response))
            p_idf = []
            for w in response:
                index = self.words.index(w)
                nidf = (self.idf_count[index] - self.idf_min) / delta
                p_idf.append(nidf)
            if len(p_idf) == 0:
                p_idf = 0
            else:
                p_idf = np.mean(sorted(p_idf, reverse=True)[:topk])
            if tf:
                scores.append(self.args['factor_tf'] * p_tf + self.args['factor_idf'] * p_idf)
            else:
                scores.append(p_idf)
        return scores


class NLI(RetrievalBaseAgent):

    def __init__(self):
        super(NLI, self).__init__(searcher=False)
        self.vocab = BertTokenizer(vocab_file='data/vocab/vocab_small')
        self.model = BERTNLI()
        self.pad = 0
        if torch.cuda.is_available():
            self.model

    @torch.no_grad()
    def scores(self, msgs, resps):
        """
        msgs: {context} [SEP] {response}, a batch version[batch can be 1]
        ids: [batch, seq]/[1, seq](batch is 1)
        """
        msgs = [f'{m} [SEP] {r}' for m, r in zip(msgs, resps)]
        ids = [torch.LongTensor(self.vocab.encode(i)[-300:]) for i in msgs]
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        if torch.cuda.is_available():
            ids = ids
        output = self.model(ids)
        output = F.softmax(output, dim=-1)
        output = output[:, 1] + output[:, 2]
        output = output.cpu().tolist()
        return output

    @torch.no_grad()
    def scores_(self, cid, rid):
        cid = torch.cat((cid, rid), dim=1)
        output = self.model(cid)
        output = F.softmax(output, dim=-1)
        output = output[:, 1] + output[:, 2]
        output = output.cpu().tolist()
        return output


class RepetitionPenalty:
    """
    ACL 2020 Conversational Graph Grounded Policy Learning for Open-Domain Conversation Generation
    When the generated responses have 60% (threshold that can be set) terms which are the same as the terms in context or the generated responses.
    """

    def __init__(self, inner_count=3, context_count=3):
        self.ic = inner_count
        self.cc = context_count

    def _repetition_context(self, contexts, responses):
        s = []
        for c, r in zip(contexts, responses):
            c_terms = list(jieba.cut(c))
            r_terms = list(jieba.cut(r))
            if len(r_terms) == 1:
                s.append(0)
                continue
            counter = 0
            for r_item in r_terms:
                if r_item in c_terms:
                    counter += 1
            try:
                ratio = counter / len(r_terms)
            except:
                ratio = 1
            s.append(1 - ratio)
        return s

    def _repetition_inner(self, responses):
        """
        avoid the cases like: '我喜欢吃面条，喜欢吃面条，吃面条'
        y = 1 - x (x is the ratio of the repetition tokens, bigger x lower score)
        """
        s = []
        for response in responses:
            terms = list(jieba.cut(response))
            terms = Counter(terms)
            values = list(terms.values())
            if len(values) == 1:
                s.append(0)
                continue
            counter = 0
            for v in values:
                if v >= self.ic:
                    counter += v
            try:
                ratio = counter / sum(values)
            except:
                ratio = 1
            s.append(1 - ratio)
        return s

    def scores(self, contexts, responses):
        s1 = self._repetition_context(contexts, responses)
        s2 = self._repetition_inner(responses)
        s = [min(s1_, s2_) for s1_, s2_ in zip(s1, s2)]
        return s


class SAFETY_FLUENCY(BaseAgent):
    """
    Use the pre-trained Language Model to calculate the fluency scores.
    Use the pre-trained dialog model to calculate the safety scores.
    
    LCCC-GPT Model
    """

    def __init__(self, path):
        super(SAFETY_FLUENCY, self).__init__()
        self.vocab = BertTokenizer.from_pretrained(path)
        self.model = LCCC(path, 0, 0.9)
        self.SPECIAL_TOKENS = ['[CLS]', '[SEP]', '[speaker1]', '[speaker2]']
        if torch.cuda.is_available():
            self.model

    def tokenize_(self, obj):
        """borrow from thu-coai/CDial-GPT"""
        return self.model.vocab.convert_tokens_to_ids(self.model.vocab.tokenize(obj))

    def build_input_from_segments(self, history, response, with_eos=True):
        """borrow from the thu-coai/CDial-GPT"""
        bos, eos, speaker1, speaker2 = self.vocab.convert_tokens_to_ids(self.SPECIAL_TOKENS)
        sequence = [[bos]] + history + [response + ([eos] if with_eos else [])]
        sequence = [sequence[0]] + [([speaker2 if i % 2 else speaker1] + s) for i, s in enumerate(sequence[1:])]
        instance = {}
        instance['input_ids'] = list(chain(*sequence))
        instance['token_type_ids'] = [bos] + [(speaker2 if i % 2 else speaker1) for i, s in enumerate(sequence[1:]) for _ in s]
        return instance

    def scores(self, msgs, resps):
        fluency_scores, safety_scores = [], []
        for m, r in zip(msgs, resps):
            probs = self.score(m, r)
            probs = np.mean(probs)
            fluency_scores.append(probs)
        return fluency_scores

    @torch.no_grad()
    def score(self, msg, res):
        self.model.eval()
        msgs = [self.tokenize_(msg), self.tokenize_(res)]
        c_ids = self.vocab.encode(msg)
        r_ids = self.vocab.encode(res)[1:]
        l_r_ids = len(r_ids)
        c_ids = c_ids + r_ids
        if len(c_ids) >= 300:
            return [0.2]
        c_ids = torch.LongTensor(c_ids)
        if torch.cuda.is_available():
            c_ids = c_ids
        output = self.model.model(input_ids=c_ids)[0]
        output = F.softmax(output, dim=-1)
        index_x = list(range(len(c_ids)))[-(l_r_ids + 1):-1]
        index_y = r_ids
        probs = output[index_x, index_y].tolist()
        return probs


class MultiView(nn.Module):
    """
    Multi-view metric for Open-domain dialog systems
    1. Coherence
    2. Fluency
    3. Safety
    4. NLI
    5. topic

    Multi-view metric model have following applications in this repo:
    1. rerank the responses generated by GPT2
    2. better evaluator for the RL-based GPT2 fine-tuning
    3. rerank the final responses (GPT2, retrieval, MRC, KBQA)

    MultiView metric/model only predict and do not train it
    """

    def __init__(self, nli=False, coherence=False, length=False, logic=False, topic=False, fluency=False, nidf_tf=False, repetition_penalty=False, distinct=False, bertmcf=False, mmi=False, lccc=None, coherence_path=None, nli_path=None, logic_path=None, topic_path=None, mmi_path=None, bertmcf_path=None, fluency_path=None, bertmultiview=None, bertmultiview_path=None, lccc_path=None):
        super(MultiView, self).__init__()
        self.mode = {'bertmultiview': bertmultiview, 'coherence': coherence, 'logic': logic, 'topic': topic, 'fluency': fluency, 'nli': nli, 'distinct': distinct, 'length': length, 'nidf_tf': nidf_tf, 'mmi': mmi, 'repetition_penalty': repetition_penalty, 'bertmcf': bertmcf, 'lccc': lccc_path}
        self.mode_weight = {'bertmultiview': 1, 'coherence': 1, 'topic': 1.2, 'fluency': 0.5, 'length': 0.4, 'nidf_tf': 0.6, 'mmi': 0.5, 'distinct': 0.6, 'repetition_penalty': 0.2, 'bertmcf': 1, 'lccc': 1}
        self.topic_map = {'电影': 'movie', '美食': 'food', '数码产品': 'electric', '音乐': 'music', '体育': 'sport'}
        self.model = {}
        if topic and not topic_path or coherence and not coherence_path or fluency and not fluency_path or logic and not logic_path or bertmultiview and not bertmultiview_path or bertmcf and not bertmcf_path or lccc and not lccc_path:
            raise Exception(f'[!] essential path is not found')
        for k, v in self.mode.items():
            if not v:
                continue
            elif k == 'topic':
                self.model['topic'] = ff.load_model(topic_path)
            elif k == 'coherence':
                self.model['coherence'] = COHERENCE()
                self.model['coherence'].load_model(coherence_path)
            elif k == 'length':
                self.model['length'] = Length()
            elif k == 'nidf_tf':
                self.model['nidf_tf'] = NIDF_TF()
            elif k == 'repetition_penalty':
                self.model['repetition_penalty'] = RepetitionPenalty()
            elif k == 'logic':
                self.model['logic'] = LOGIC()
                self.model['logic'].load_model(logic_path)
            elif k == 'nli':
                self.model['nli'] = NLI()
                self.model['nli'].load_model(nli_path)
            elif k == 'mmi':
                self.model['mmi'] = MMI()
                self.model['mmi'].load_model(mmi_path)
            elif k == 'distinct':
                self.model['distinct'] = Distinct()
            elif k == 'fluency':
                self.model['fluency'] = SAFETY_FLUENCY()
                self.model['fluency'].load_model(fluency_path)
            elif k == 'bertmultiview':
                self.model['bertmultiview'] = BERT_MULTIVIEW()
                self.model['bertmultiview'].load_model(bertmultiview_path)
            elif k == 'bertmcf':
                self.model['bertmcf'] = BERTMCF()
                self.model['bertmcf'].load_model(bertmcf_path)
            elif k == 'lccc':
                self.model['lccc'] = LCCCLM(lccc_path, 0, 0.9)
        None
        for k, v in self.mode.items():
            if v:
                None

    def topic_scores(self, msg, topic):
        """
        msg is a string
        """
        topic = self.topic_map[topic]
        msg = ' '.join(jieba.cut(msg))
        try:
            label, value = self.model['topic'].predict(msg)
        except:
            return False
        label = label[0].replace('__label__', '')
        value = value[0]
        if topic == label:
            return True
        elif value <= 0.4:
            return True
        else:
            return False

    @torch.no_grad()
    def forward(self, context, response, groundtruth=None, topic=None, history=None, bertmultiview_details=False):
        """
        context: the string of the conversation context
        response: the string of the responses
        topic: a list of the topic of the conversation context
        history: a list of the utterances that are talked by the agent

        run one time, process one batch

        return the scores of the sub-models and the final average score
        average_score, (sub_model_score1, sub_model_score2, ...)
        :average_scores: [batch]
        :sub_model_score[i]: [batch]
        """
        scores = {k: [] for k, v in self.mode.items() if v}
        for k in scores.keys():
            if k == 'topic':
                response_ = [' '.join(jieba.cut(i)) for i in response]
                label, value = self.model[k].predict(response_)
                label = [i[0].replace('__label__', '') for i in label]
                value = [i[0] for i in value]
                rest = []
                for l, t, v in zip(label, topic, value):
                    if l == t:
                        rest.append(v)
                    else:
                        rest.append(1 - v)
                scores[k] = rest
            elif k in ['length', 'nidf_tf']:
                scores[k] = self.model[k].scores(response)
            elif k in ['distinct']:
                scores[k] = self.model[k].scores(response, history)
            elif k in ['bertmultiview']:
                scores[k] = self.model[k].scores(context, response, details=bertmultiview_details)
            elif k in ['bertmcf']:
                assert groundtruth is not None, 'bertmcf must use the groundtruth'
                scores[k] = self.model[k].scores(context, groundtruth, response)
            elif k in ['lccc']:
                scores[k] = self.model[k].scores(context, response, temperature=0.7)
            else:
                scores[k] = self.model[k].scores(context, response)
        average_scores = []
        batch_size = len(context)
        if bertmultiview_details:
            return None, scores
        else:
            for idx in range(batch_size):
                average_scores.append(np.sum([(v[idx] * self.mode_weight[key]) for key, v in scores.items()]))
            return average_scores, scores


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Attention,
     lambda: ([], {'hidden_size': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4, 4])], {}),
     True),
    (IRHead,
     lambda: ([], {'hidden_size': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     True),
    (MultiView,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (PositionEmbedding,
     lambda: ([], {'d_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (RetrievalModel,
     lambda: ([], {'hidden_size': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     True),
]

class Test_gmftbyGMFTBY_OpenDialog(_paritybench_base):
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

