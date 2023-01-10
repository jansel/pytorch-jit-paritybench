import sys
_module = sys.modules[__name__]
del sys
bert4torch = _module
activations = _module
layers = _module
losses = _module
models = _module
optimizers = _module
snippets = _module
tokenizers = _module
conf = _module
basic_extract_features = _module
basic_gibbs_sampling_via_mlm = _module
basic_language_model_CDial_GPT = _module
basic_language_model_GAU_alpha = _module
basic_language_model_bart = _module
basic_language_model_bert = _module
basic_language_model_cpm_lm = _module
basic_language_model_deberta_v2 = _module
basic_language_model_ernie = _module
basic_language_model_gpt2_ml = _module
basic_language_model_guwenbert = _module
basic_language_model_macbert = _module
basic_language_model_nezha_gen_gpt = _module
basic_language_model_nezha_gpt_dialog = _module
basic_language_model_roformer = _module
basic_language_model_simbert = _module
basic_language_model_t5_pegasus = _module
basic_language_model_transformer_xl = _module
basic_language_model_uer_t5 = _module
basic_language_model_wobert = _module
basic_language_model_xlnet = _module
basic_make_uncased_model_cased = _module
basic_test_parallel_apply = _module
basic_test_tokenizer = _module
convert_GAU_alpha = _module
convert_bart__cloudwalk = _module
convert_bart_fudanNLP = _module
convert_deberta_v2 = _module
convert_nezha_gpt_dialog = _module
convert_roberta_chess = _module
convert_simbert = _module
convert_t5_pegasus = _module
convert_transformer_xl = _module
task_conditional_language_model = _module
task_event_extraction_gplinker = _module
task_iflytek_bert_of_theseus = _module
task_language_model = _module
task_language_model_chinese_chess = _module
task_nl2sql_baseline = _module
pretrain_gpt_lm = _module
pretrain_roberta_mlm = _module
pretrain_roberta_mlm_data_gen = _module
simbert_v2_stage1 = _module
simbert_v2_stage2 = _module
simbert_v2_supervised = _module
task_relation_extraction_CasRel = _module
task_relation_extraction_CasRel_VAT = _module
task_relation_extraction_PGRC = _module
task_relation_extraction_SPN4RE = _module
task_relation_extraction_gplinker = _module
task_relation_extraction_tplinker = _module
task_relation_extraction_tplinker_plus = _module
task_sentiment_sohu = _module
training = _module
training_bert = _module
convert = _module
inference = _module
training = _module
task_sentence_similarity_lcqmc = _module
task_sentiment_classification = _module
task_sentiment_classification_GAU_alpha = _module
task_sentiment_classification_PET = _module
task_sentiment_classification_P_tuning = _module
task_sentiment_classification_albert = _module
task_sentiment_classification_deberta_v2 = _module
task_sentiment_classification_electra = _module
task_sentiment_classification_hierarchical_position = _module
task_sentiment_classification_nezha = _module
task_sentiment_classification_roformer = _module
task_sentiment_classification_roformer_v2 = _module
task_sentiment_classification_wobert = _module
task_sentiment_classification_xlnet = _module
config = _module
task_sentence_embedding_FinanceFAQ_step1_1 = _module
task_sentence_embedding_FinanceFAQ_step2_1 = _module
utils = _module
task_sentence_embedding_DimensionalityReduction = _module
task_sentence_embedding_model_distillation = _module
task_sentence_embedding_sup_CoSENT = _module
task_sentence_embedding_sup_ContrastiveLoss = _module
task_sentence_embedding_sup_CosineMSELoss = _module
task_sentence_embedding_sup_InfoNCE = _module
task_sentence_embedding_sup_concat_CrossEntropyLoss = _module
task_sentence_embedding_unsup_CT = _module
task_sentence_embedding_unsup_DiffCSE = _module
task_sentence_embedding_unsup_ESimCSE = _module
task_sentence_embedding_unsup_PromptBert = _module
task_sentence_embedding_unsup_SimCSE = _module
task_sentence_embedding_unsup_TSDAE = _module
task_sentence_embedding_unsup_bert_whitening = _module
task_kgclue_seq2seq = _module
task_question_answer_generation_by_seq2seq = _module
task_reading_comprehension_by_mlm = _module
task_reading_comprehension_by_seq2seq = _module
task_seq2seq_ape210k_math_word_problem = _module
task_seq2seq_autotitle = _module
task_seq2seq_autotitle_csl_bart = _module
task_seq2seq_autotitle_csl_mt5 = _module
task_seq2seq_autotitle_csl_t5_pegasus = _module
task_seq2seq_autotitle_csl_uer_t5 = _module
task_seq2seq_autotitle_csl_unilm = _module
task_seq2seq_autotitle_guwenbert = _module
task_seq2seq_simbert = _module
task_sequence_labeling_ner_CNN_Nested_NER = _module
task_sequence_labeling_ner_W2NER = _module
task_sequence_labeling_ner_cascade_crf = _module
task_sequence_labeling_ner_crf = _module
task_sequence_labeling_ner_crf_add_posseg = _module
task_sequence_labeling_ner_crf_freeze = _module
task_sequence_labeling_ner_efficient_global_pointer = _module
task_sequence_labeling_ner_global_pointer = _module
task_sequence_labeling_ner_lear = _module
task_sequence_labeling_ner_mrc = _module
task_sequence_labeling_ner_span = _module
task_sequence_labeling_ner_tplinker_plus = _module
convert = _module
finetune_step1_dataprocess = _module
finetune_step2_doccano = _module
finetune_step3_train = _module
model = _module
uie_predictor = _module
utils = _module
basic_simple_web_serving_simbert = _module
server = _module
create_documents = _module
create_index = _module
index_documents = _module
model = _module
client = _module
constants = _module
configs = _module
loggers = _module
trace_log = _module
view = _module
task_bert_cls_onnx = _module
client_request = _module
client_tritonclient = _module
task_amp = _module
task_data_parallel = _module
task_distributed_data_parallel = _module
task_sentiment_TemporalEnsembling = _module
task_sentiment_UDA = _module
task_sentiment_adversarial_training = _module
task_sentiment_exponential_moving_average = _module
task_sentiment_exponential_moving_average_warmup = _module
task_sentiment_mixup = _module
task_sentiment_virtual_adversarial_training = _module
tutorials_custom_fit_progress = _module
tutorials_load_transformers_model = _module
tutorials_small_tips = _module
setup = _module

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


import math


import torch


from torch import nn


import torch.nn as nn


import torch.nn.functional as F


import numpy as np


from typing import List


from typing import Optional


from typing import Union


import random


import warnings


import copy


import re


from torch.utils.checkpoint import checkpoint as grad_checkpoint


from torch.optim.lr_scheduler import LambdaLR


from torch.nn.utils.rnn import pad_sequence


import inspect


import time


import tensorflow as tf


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


import torch.optim as optim


from itertools import groupby


from torch import optim


from torch.distributions.bernoulli import Bernoulli


from collections import Counter


import collections


from scipy.optimize import linear_sum_assignment


from sklearn.metrics import f1_score


from sklearn.metrics import classification_report


from sklearn.metrics import accuracy_score


from torch import device


import pandas as pd


from sklearn.model_selection import StratifiedKFold


from torch.optim import Adam


from torch.utils.data import TensorDataset


from sklearn.metrics.pairwise import paired_cosine_distances


from sklearn.metrics import roc_auc_score


from torch import Tensor


from sklearn.decomposition import PCA


from scipy.stats import spearmanr


from scipy.stats import pearsonr


from sklearn.metrics.pairwise import paired_euclidean_distances


from sklearn.metrics.pairwise import paired_manhattan_distances


import scipy.stats


from collections import defaultdict


from sklearn.metrics import precision_recall_fscore_support


from torch.nn.utils.rnn import pack_padded_sequence


from torch.nn.utils.rnn import pad_packed_sequence


from collections import deque


import logging


from random import sample


import functools


from functools import partial


import torch.onnx


from torch.utils.data.distributed import DistributedSampler


from itertools import cycle


class LayerNorm(nn.Module):

    def __init__(self, hidden_size, eps=1e-12, conditional_size=False, weight=True, bias=True, norm_mode='normal', **kwargs):
        """layernorm 层，这里自行实现，目的是为了兼容 conditianal layernorm，使得可以做条件文本生成、条件分类等任务
           条件layernorm来自于苏剑林的想法，详情：https://spaces.ac.cn/archives/7124
        """
        super(LayerNorm, self).__init__()
        if weight:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.norm_mode = norm_mode
        self.eps = eps
        self.conditional_size = conditional_size
        if conditional_size:
            self.dense1 = nn.Linear(conditional_size, hidden_size, bias=False)
            self.dense1.weight.data.uniform_(0, 0)
            self.dense2 = nn.Linear(conditional_size, hidden_size, bias=False)
            self.dense2.weight.data.uniform_(0, 0)

    def forward(self, x):
        inputs = x[0]
        if self.norm_mode == 'rmsnorm':
            variance = inputs.pow(2).mean(-1, keepdim=True)
            o = inputs * torch.rsqrt(variance + self.eps)
        else:
            u = inputs.mean(-1, keepdim=True)
            s = (inputs - u).pow(2).mean(-1, keepdim=True)
            o = (inputs - u) / torch.sqrt(s + self.eps)
        if not hasattr(self, 'weight'):
            self.weight = 1
        if not hasattr(self, 'bias'):
            self.bias = 0
        if self.conditional_size:
            cond = x[1]
            for _ in range(len(inputs.shape) - len(cond.shape)):
                cond = cond.unsqueeze(dim=1)
            return (self.weight + self.dense1(cond)) * o + (self.bias + self.dense2(cond))
        else:
            return self.weight * o + self.bias


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """ sinusoid编码
        
        :param n_position: int, 位置长度
        :param d_hid: int, 位置编码长度
        :param padding_idx: padding的token_ids
        :return: [seq_len, d_hid]
    """
    position = torch.arange(0, n_position, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_hid, 2).float() * (-math.log(10000.0) / d_hid))
    embeddings_table = torch.zeros(n_position, d_hid)
    embeddings_table[:, 0::2] = torch.sin(position * div_term)
    embeddings_table[:, 1::2] = torch.cos(position * div_term)
    return embeddings_table
    position_ids = torch.arange(0, n_position).unsqueeze(1)
    position_ids = position_ids.expand(-1, d_hid)
    indices = torch.arange(0, d_hid)
    position_ids = position_ids * torch.pow(10000, -2 * torch.true_divide(torch.floor_divide(indices, 2), d_hid))
    position_ids[:, ::2] = torch.sin(position_ids[:, ::2])
    position_ids[:, 1::2] = torch.cos(position_ids[:, 1::2])
    return position_ids


class RelativePositionsEncoding(nn.Module):
    """nezha用的google相对位置编码
    来自论文：https://arxiv.org/abs/1803.02155
    """

    def __init__(self, qlen, klen, embedding_size, max_relative_position=127):
        super(RelativePositionsEncoding, self).__init__()
        vocab_size = max_relative_position * 2 + 1
        distance_mat = torch.arange(klen)[None, :] - torch.arange(qlen)[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -max_relative_position, max_relative_position)
        final_mat = distance_mat_clipped + max_relative_position
        embeddings_table = get_sinusoid_encoding_table(vocab_size, embedding_size)
        position_embeddings = nn.Embedding.from_pretrained(embeddings_table, freeze=True)(final_mat)
        self.register_buffer('position_embeddings', position_embeddings)

    def forward(self, qlen, klen):
        return self.position_embeddings[:qlen, :klen, :]


class RelativePositionsEncodingDebertaV2(nn.Module):
    """deberta用的相对位置编码
    来自论文：https://arxiv.org/abs/2006.03654
    """

    def __init__(self, qlen, klen, position_buckets, max_position):
        super(RelativePositionsEncodingDebertaV2, self).__init__()
        q_ids = torch.arange(0, qlen)
        k_ids = torch.arange(0, klen)
        rel_pos_ids = q_ids[:, None] - k_ids[None, :]
        if position_buckets > 0 and max_position > 0:
            rel_pos_ids = self.make_log_bucket_position(rel_pos_ids, position_buckets, max_position)
        rel_pos_ids = rel_pos_ids
        rel_pos_ids = rel_pos_ids[:qlen, :]
        rel_pos_ids = rel_pos_ids.unsqueeze(0)
        self.register_buffer('relative_position', rel_pos_ids)

    @staticmethod
    def make_log_bucket_position(relative_pos, bucket_size, max_position):
        sign = torch.sign(relative_pos)
        mid = bucket_size // 2
        abs_pos = torch.where((relative_pos < mid) & (relative_pos > -mid), torch.tensor(mid - 1).type_as(relative_pos), torch.abs(relative_pos))
        log_pos = torch.ceil(torch.log(abs_pos / mid) / torch.log(torch.tensor((max_position - 1) / mid)) * (mid - 1)) + mid
        bucket_pos = torch.where(abs_pos <= mid, relative_pos.type_as(log_pos), log_pos * sign)
        return bucket_pos

    def forward(self, qlen, klen):
        return self.relative_position[:, :qlen, :klen]


class RelativePositionsEncodingT5(nn.Module):
    """Google T5的相对位置编码
    来自论文：https://arxiv.org/abs/1910.10683
    """

    def __init__(self, qlen, klen, relative_attention_num_buckets, is_decoder=False):
        super(RelativePositionsEncodingT5, self).__init__()
        context_position = torch.arange(qlen, dtype=torch.long)[:, None]
        memory_position = torch.arange(klen, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position
        relative_position = self._relative_position_bucket(relative_position, bidirectional=not is_decoder, num_buckets=relative_attention_num_buckets)
        self.register_buffer('relative_position', relative_position)

    def forward(self, qlen, klen):
        return self.relative_position[:qlen, :klen]

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """直接来源于transformer
        """
        ret = 0
        n = -relative_position
        if bidirectional:
            num_buckets //= 2
            ret += (n < 0) * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))
        max_exact = num_buckets // 2
        is_small = n < max_exact
        val_if_large = max_exact + torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
        ret += torch.where(is_small, n, val_if_large)
        return ret


class RoPEPositionEncoding(nn.Module):
    """旋转式位置编码: https://kexue.fm/archives/8265
    """

    def __init__(self, max_position, embedding_size):
        super(RoPEPositionEncoding, self).__init__()
        position_embeddings = get_sinusoid_encoding_table(max_position, embedding_size)
        cos_position = position_embeddings[:, 1::2].repeat_interleave(2, dim=-1)
        sin_position = position_embeddings[:, ::2].repeat_interleave(2, dim=-1)
        self.register_buffer('cos_position', cos_position)
        self.register_buffer('sin_position', sin_position)

    def forward(self, qw, seq_dim=-2):
        seq_len = qw.shape[seq_dim]
        qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], dim=-1).reshape_as(qw)
        return qw * self.cos_position[:seq_len] + qw2 * self.sin_position[:seq_len]


class MultiHeadAttentionLayer(nn.Module):

    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, dropout_rate=0.1, attention_scale=True, return_attention_scores=False, bias=True, **kwargs):
        super(MultiHeadAttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        if kwargs.get('attention_head_size'):
            self.attention_head_size = kwargs.get('attention_head_size')
        else:
            self.attention_head_size = int(hidden_size / num_attention_heads)
        self.inner_dim = self.num_attention_heads * self.attention_head_size
        self.attention_scale = attention_scale
        self.return_attention_scores = return_attention_scores
        self.bias = bias
        self.q = nn.Linear(hidden_size, self.inner_dim, bias=bias)
        self.k = nn.Linear(hidden_size, self.inner_dim, bias=bias)
        self.v = nn.Linear(hidden_size, self.inner_dim, bias=bias)
        self.o = nn.Linear(self.inner_dim, hidden_size, bias=bias)
        self.dropout = nn.Dropout(attention_probs_dropout_prob)
        self.a_bias, self.p_bias = kwargs.get('a_bias'), kwargs.get('p_bias')
        if self.p_bias == 'typical_relative':
            self.relative_positions_encoding = RelativePositionsEncoding(qlen=kwargs.get('max_position'), klen=kwargs.get('max_position'), embedding_size=self.attention_head_size, max_relative_position=kwargs.get('max_relative_position'))
        elif self.p_bias == 'rotary':
            self.relative_positions_encoding = RoPEPositionEncoding(max_position=kwargs.get('max_position'), embedding_size=self.attention_head_size)
        elif self.p_bias == 't5_relative':
            self.relative_positions = RelativePositionsEncodingT5(qlen=kwargs.get('max_position'), klen=kwargs.get('max_position'), relative_attention_num_buckets=kwargs.get('relative_attention_num_buckets'), is_decoder=kwargs.get('is_decoder'))
            self.relative_positions_encoding = nn.Embedding(kwargs.get('relative_attention_num_buckets'), self.num_attention_heads)
        elif self.p_bias == 'deberta_v2':
            self.share_att_key = kwargs.get('share_att_key', False)
            self.position_buckets = kwargs.get('position_buckets', -1)
            self.max_relative_positions = kwargs.get('max_relative_positions', -1)
            if self.max_relative_positions < 1:
                self.max_relative_positions = kwargs.get('max_position_embeddings')
            self.pos_ebd_size = self.max_relative_positions
            if self.position_buckets > 0:
                self.pos_ebd_size = self.position_buckets
            self.pos_att_type = kwargs.get('pos_att_type', [])
            self.relative_positions = RelativePositionsEncodingDebertaV2(qlen=kwargs.get('max_position'), klen=kwargs.get('max_position'), position_buckets=kwargs.get('position_buckets'), max_position=kwargs.get('max_position'))
            self.relative_positions_encoding = nn.Embedding(kwargs.get('max_position'), self.hidden_size)
            self.norm_rel_ebd = [x.strip() for x in kwargs.get('norm_rel_ebd', 'none').lower().split('|')]
            if 'layer_norm' in self.norm_rel_ebd:
                self.layernorm = nn.LayerNorm(self.hidden_size, kwargs.get('layer_norm_eps', 1e-12), elementwise_affine=True)
            self.pos_dropout = nn.Dropout(dropout_rate)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, query_states=None):
        if query_states is None:
            query_states = hidden_states
        mixed_query_layer = self.q(query_states)
        if encoder_hidden_states is not None:
            mixed_key_layer = self.k(encoder_hidden_states)
            mixed_value_layer = self.v(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.k(hidden_states)
            mixed_value_layer = self.v(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        if self.p_bias == 'rotary':
            query_layer = self.relative_positions_encoding(query_layer)
            key_layer = self.relative_positions_encoding(key_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        if self.p_bias == 'typical_relative' and hasattr(self, 'relative_positions_encoding'):
            relations_keys = self.relative_positions_encoding(attention_scores.shape[-1], attention_scores.shape[-1])
            key_position_scores_r_t = torch.einsum('bnih,ijh->bnij', query_layer, relations_keys)
            attention_scores = attention_scores + key_position_scores_r_t
        elif self.p_bias == 't5_relative' and hasattr(self, 'relative_positions_encoding'):
            relations_keys = self.relative_positions(attention_scores.shape[-1], attention_scores.shape[-1])
            key_position_scores_r_t = self.relative_positions_encoding(relations_keys).permute([2, 0, 1]).unsqueeze(0)
            attention_scores = attention_scores + key_position_scores_r_t
        if self.p_bias == 'deberta_v2':
            scale_factor = 1
            if 'c2p' in self.pos_att_type:
                scale_factor += 1
            if 'p2c' in self.pos_att_type:
                scale_factor += 1
            scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
            attention_scores = attention_scores / scale
            rel_embeddings = self.pos_dropout(self.layernorm(self.relative_positions_encoding.weight))
            relations_keys = self.relative_positions(attention_scores.shape[-1], attention_scores.shape[-1])
            rel_att = self.disentangled_attention_bias(query_layer, key_layer, relations_keys, rel_embeddings, scale_factor)
            attention_scores = attention_scores + rel_att
        elif self.attention_scale:
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_mask = (1.0 - attention_mask) * -10000.0
            attention_scores = attention_scores + attention_mask
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        if self.p_bias == 'typical_relative' and hasattr(self, 'relative_positions_encoding'):
            relations_values = self.relative_positions_encoding(attention_scores.shape[-1], attention_scores.shape[-1])
            value_position_scores_r_t = torch.einsum('bnij,ijh->bnih', attention_probs, relations_values)
            context_layer = context_layer + value_position_scores_r_t
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.inner_dim,)
        context_layer = context_layer.view(*new_context_layer_shape)
        if self.return_attention_scores:
            return self.o(context_layer), attention_scores
        else:
            return self.o(context_layer)

    def disentangled_attention_bias(self, query_layer, key_layer, relative_pos, rel_embeddings, scale_factor):
        """deberta_v2使用，和原版区别是query_layer是4维
        """
        btz, n_head, q_len, d_head = query_layer.size()
        k_len = key_layer.size(-2)
        if relative_pos is None:
            relative_pos = self.relative_positions(q_len, k_len)
        if relative_pos.dim() == 2:
            relative_pos = relative_pos.unsqueeze(0).unsqueeze(0)
        elif relative_pos.dim() == 3:
            relative_pos = relative_pos.unsqueeze(1)
        elif relative_pos.dim() != 4:
            raise ValueError(f'Relative position ids must be of dim 2 or 3 or 4. {relative_pos.dim()}')
        att_span = self.pos_ebd_size
        relative_pos = relative_pos.long()
        rel_embeddings = rel_embeddings[0:att_span * 2, :].unsqueeze(0)
        if self.share_att_key:
            pos_query_layer = self.transpose_for_scores(self.q(rel_embeddings)).repeat(btz, 1, 1, 1)
            pos_key_layer = self.transpose_for_scores(self.k(rel_embeddings)).repeat(btz, 1, 1, 1)
        else:
            pass
        score = 0
        if 'c2p' in self.pos_att_type:
            scale = torch.sqrt(torch.tensor(d_head, dtype=torch.float) * scale_factor)
            c2p_att = torch.matmul(query_layer, pos_key_layer.transpose(-1, -2))
            c2p_pos = torch.clamp(relative_pos + att_span, 0, att_span * 2 - 1)
            c2p_att = torch.gather(c2p_att, dim=-1, index=c2p_pos.expand([btz, n_head, q_len, k_len]))
            score += c2p_att / scale
        if 'p2c' in self.pos_att_type:
            scale = torch.sqrt(torch.tensor(d_head, dtype=torch.float) * scale_factor)
            if k_len != q_len:
                r_pos = self.relative_positions(k_len, k_len)
                r_pos = r_pos.unsqueeze(0)
            else:
                r_pos = relative_pos
            p2c_pos = torch.clamp(-r_pos + att_span, 0, att_span * 2 - 1)
            p2c_att = torch.matmul(key_layer, pos_query_layer.transpose(-1, -2))
            p2c_att = torch.gather(p2c_att, dim=-1, index=p2c_pos.squeeze(0).expand([btz, n_head, k_len, k_len])).transpose(-1, -2)
            score += p2c_att / scale
        return score


def _gelu_new(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


def gelu_fast(x):
    return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))


def linear_act(x):
    return x


def quick_gelu(x):
    return x * torch.sigmoid(1.702 * x)


def get_activation(activation_string):
    """根据activation_string返回对应的激活函数

    :param activation_string: str, 传入的激活函数名
    :return: Any
    """
    if activation_string in ACT2FN:
        return ACT2FN[activation_string]
    else:
        raise KeyError(f'function {activation_string} not found in ACT2FN mapping {list(ACT2FN.keys())}')


class PositionWiseFeedForward(nn.Module):

    def __init__(self, hidden_size, intermediate_size, dropout_rate=0.5, hidden_act='gelu', is_dropout=False, bias=True, **kwargs):
        super(PositionWiseFeedForward, self).__init__()
        self.is_dropout = is_dropout
        self.intermediate_act_fn = get_activation(hidden_act)
        self.intermediateDense = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.outputDense = nn.Linear(intermediate_size, hidden_size, bias=bias)
        if self.is_dropout:
            self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        if self.is_dropout:
            x = self.dropout(self.intermediate_act_fn(self.intermediateDense(x)))
        else:
            x = self.intermediate_act_fn(self.intermediateDense(x))
        x = self.outputDense(x)
        return x


class GatedAttentionUnit(nn.Module):
    """门控注意力单元，
    链接：https://arxiv.org/abs/2202.10447
    介绍：https://kexue.fm/archives/8934
    说明：没有加入加性相对位置编码
    参考pytorch项目：https://github.com/lucidrains/FLASH-pytorch
    """

    def __init__(self, hidden_size, attention_key_size, intermediate_size, attention_probs_dropout_prob, hidden_act, is_dropout=False, attention_scale=True, bias=True, normalization='softmax_plus', **kwargs):
        super().__init__()
        self.intermediate_size = intermediate_size
        self.attention_head_size = attention_key_size
        self.attention_scale = attention_scale
        self.is_dropout = is_dropout
        self.normalization = normalization
        self.hidden_fn = get_activation(hidden_act)
        self.dropout = nn.Dropout(attention_probs_dropout_prob)
        self.i_dense = nn.Linear(hidden_size, self.intermediate_size * 2 + attention_key_size, bias=bias)
        self.offsetscale = self.OffsetScale(attention_key_size, heads=2, bias=bias)
        self.o_dense = nn.Linear(self.intermediate_size, hidden_size, bias=bias)
        self.a_bias, self.p_bias = kwargs.get('a_bias'), kwargs.get('p_bias')
        if self.p_bias == 'rotary':
            self.relative_positions_encoding = RoPEPositionEncoding(max_position=kwargs.get('max_position'), embedding_size=self.attention_head_size)

    def forward(self, hidden_states, attention_mask):
        hidden_states = self.hidden_fn(self.i_dense(hidden_states))
        u, v, qk = hidden_states.split([self.intermediate_size, self.intermediate_size, self.attention_head_size], dim=-1)
        q, k = self.offsetscale(qk)
        if self.p_bias == 'rotary':
            q = self.relative_positions_encoding(q)
            k = self.relative_positions_encoding(k)
        attention_scores = torch.einsum('b i d, b j d -> b i j', q, k)
        if self.attention_scale:
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_mask = (1.0 - attention_mask) * -1000000000000.0
            attention_scores = attention_scores + attention_mask.squeeze(1)
        attention_scores = self.attention_normalize(attention_scores, -1, self.normalization)
        if self.is_dropout:
            attention_scores = self.dropout(attention_scores)
        out = self.o_dense(u * torch.einsum('b i j, b j d -> b i d', attention_scores, v))
        return out

    def attention_normalize(self, a, dim=-1, method='softmax'):
        """不同的注意力归一化方案
        softmax：常规/标准的指数归一化；
        squared_relu：来自 https://arxiv.org/abs/2202.10447 ；
        softmax_plus：来自 https://kexue.fm/archives/8823 。
        """
        if method == 'softmax':
            return F.softmax(a, dim=dim)
        else:
            mask = (a > -100000000000.0).float()
            l = torch.maximum(torch.sum(mask, dim=dim, keepdims=True), torch.tensor(1))
            if method == 'squared_relu':
                return F.relu(a) ** 2 / l
            elif method == 'softmax_plus':
                return F.softmax(a * torch.log(l) / torch.log(torch.tensor(512.0)), dim=dim)
        return a


    class OffsetScale(nn.Module):
        """仿射变换
        """

        def __init__(self, head_size, heads=1, bias=True):
            super().__init__()
            self.gamma = nn.Parameter(torch.ones(heads, head_size))
            self.bias = bias
            if bias:
                self.beta = nn.Parameter(torch.zeros(heads, head_size))
            nn.init.normal_(self.gamma, std=0.02)

        def forward(self, x):
            out = torch.einsum('... d, h d -> ... h d', x, self.gamma)
            if self.bias:
                out = out + self.beta
            return out.unbind(dim=-2)


class SinusoidalPositionEncoding(nn.Module):
    """定义Sin-Cos位置Embedding
    """

    def __init__(self, max_position, embedding_size):
        super(SinusoidalPositionEncoding, self).__init__()
        self.position_embeddings = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(max_position, embedding_size), freeze=True)

    def forward(self, position_ids):
        return self.position_embeddings(position_ids)


class BertEmbeddings(nn.Module):
    """embeddings层
       构造word, position and token_type embeddings.
    """

    def __init__(self, vocab_size, embedding_size, hidden_size, max_position, segment_vocab_size, shared_segment_embeddings, dropout_rate, conditional_size=False, token_pad_ids=0, **kwargs):
        super(BertEmbeddings, self).__init__()
        self.shared_segment_embeddings = shared_segment_embeddings
        self.word_embeddings = nn.Embedding(vocab_size, embedding_size, padding_idx=token_pad_ids)
        if kwargs.get('p_bias') == 'sinusoid':
            self.position_embeddings = SinusoidalPositionEncoding(max_position, embedding_size)
        elif kwargs.get('p_bias') in {'rotary', 'typical_relative', 't5_relative', 'other_relative', 'deberta_v2'}:
            pass
        elif max_position > 0:
            self.position_embeddings = nn.Embedding(max_position, embedding_size)
        if segment_vocab_size > 0 and not shared_segment_embeddings and kwargs.get('use_segment_embedding', True):
            self.segment_embeddings = nn.Embedding(segment_vocab_size, embedding_size)
        self.emb_scale = kwargs.get('emb_scale', 1)
        self.layerNorm = LayerNorm(embedding_size, eps=1e-12, conditional_size=conditional_size, **kwargs)
        self.dropout = nn.Dropout(dropout_rate)
        if embedding_size != hidden_size:
            self.embedding_hidden_mapping_in = nn.Linear(embedding_size, hidden_size)

    def forward(self, token_ids, segment_ids=None, position_ids=None, conditional_emb=None, additional_embs=None, attention_mask=None):
        if not token_ids.requires_grad and token_ids.dtype in {torch.long, torch.int}:
            words_embeddings = self.word_embeddings(token_ids)
        else:
            words_embeddings = token_ids
        if hasattr(self, 'segment_embeddings'):
            segment_ids = torch.zeros_like(token_ids) if segment_ids is None else segment_ids
            segment_embeddings = self.segment_embeddings(segment_ids)
            embeddings = words_embeddings + segment_embeddings
        elif self.shared_segment_embeddings:
            segment_ids = torch.zeros_like(token_ids) if segment_ids is None else segment_ids
            segment_embeddings = self.word_embeddings(segment_ids)
            embeddings = words_embeddings + segment_embeddings
        else:
            embeddings = words_embeddings
        if additional_embs is not None:
            for emb in additional_embs:
                embeddings += emb
        if hasattr(self, 'position_embeddings'):
            seq_length = token_ids.size(1)
            if position_ids is None:
                position_ids = torch.arange(seq_length, dtype=torch.long, device=token_ids.device)
                position_ids = position_ids.unsqueeze(0).repeat(token_ids.shape[0], 1)
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        if self.emb_scale != 1:
            embeddings = embeddings * self.emb_scale
        if hasattr(self, 'layerNorm'):
            embeddings = self.layerNorm((embeddings, conditional_emb))
        if attention_mask is not None:
            embeddings *= attention_mask[:, 0, 0, :, None]
        embeddings = self.dropout(embeddings)
        if hasattr(self, 'embedding_hidden_mapping_in'):
            embeddings = self.embedding_hidden_mapping_in(embeddings)
        return embeddings


class BertLayer(nn.Module):
    """Transformer层:
       顺序为: Attention --> Add --> LayerNorm --> Feed Forward --> Add --> LayerNorm

       注意:
        1. 以上都不计dropout层，并不代表没有dropout，每一层的dropout使用略有不同，注意区分
        2. 原始的Transformer的encoder中的Feed Forward层一共有两层linear，
        3. config.intermediate_size的大小不仅是第一层linear的输出尺寸，也是第二层linear的输入尺寸
    """

    def __init__(self, hidden_size, num_attention_heads, dropout_rate, attention_probs_dropout_prob, intermediate_size, hidden_act, is_dropout=False, conditional_size=False, **kwargs):
        super(BertLayer, self).__init__()
        self.multiHeadAttention = MultiHeadAttentionLayer(hidden_size, num_attention_heads, attention_probs_dropout_prob, dropout_rate, **kwargs)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.layerNorm1 = LayerNorm(hidden_size, eps=1e-12, conditional_size=conditional_size, **kwargs)
        self.feedForward = PositionWiseFeedForward(hidden_size, intermediate_size, dropout_rate, hidden_act, is_dropout=is_dropout, **kwargs)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.layerNorm2 = LayerNorm(hidden_size, eps=1e-12, conditional_size=conditional_size, **kwargs)
        self.is_decoder = kwargs.get('is_decoder')
        if self.is_decoder:
            self.crossAttention = MultiHeadAttentionLayer(hidden_size, num_attention_heads, attention_probs_dropout_prob, dropout_rate, **kwargs)
            self.dropout3 = nn.Dropout(dropout_rate)
            self.layerNorm3 = LayerNorm(hidden_size, eps=1e-12, conditional_size=conditional_size, **kwargs)

    def forward(self, hidden_states, attention_mask, conditional_emb=None, encoder_hidden_states=None, encoder_attention_mask=None):
        self_attn_output = self.multiHeadAttention(hidden_states, attention_mask)
        hidden_states = hidden_states + self.dropout1(self_attn_output)
        hidden_states = self.layerNorm1((hidden_states, conditional_emb))
        if self.is_decoder and encoder_hidden_states is not None:
            cross_attn_output = self.crossAttention(hidden_states, None, encoder_hidden_states, encoder_attention_mask)
            hidden_states = hidden_states + self.dropout3(cross_attn_output)
            hidden_states = self.layerNorm3((hidden_states, conditional_emb))
        self_attn_output2 = self.feedForward(hidden_states)
        hidden_states = hidden_states + self.dropout2(self_attn_output2)
        hidden_states = self.layerNorm2((hidden_states, conditional_emb))
        return hidden_states


class T5Layer(BertLayer):
    """T5的Encoder的主体是基于Self-Attention的模块
    顺序：LN --> Att --> Add --> LN --> FFN --> Add
    """

    def __init__(self, *args, version='t5.1.0', **kwargs):
        super().__init__(*args, **kwargs)
        if version.endswith('t5.1.1'):
            kwargs['dropout_rate'] = args[2]
            kwargs['hidden_act'] = args[5]
            self.feedForward = self.T5PositionWiseFeedForward(hidden_size=args[0], intermediate_size=args[4], **kwargs)
        if self.is_decoder and hasattr(self.crossAttention, 'relative_positions_encoding'):
            del self.crossAttention.relative_positions_encoding
            del self.crossAttention.relative_positions

    def forward(self, hidden_states, attention_mask, conditional_emb=None, encoder_hidden_states=None, encoder_attention_mask=None):
        x = self.layerNorm1((hidden_states, conditional_emb))
        self_attn_output = self.multiHeadAttention(x, attention_mask)
        hidden_states = hidden_states + self.dropout1(self_attn_output)
        if self.is_decoder and encoder_hidden_states is not None:
            x = self.layerNorm3((hidden_states, conditional_emb))
            cross_attn_output = self.crossAttention(x, None, encoder_hidden_states, encoder_attention_mask)
            hidden_states = hidden_states + self.dropout3(cross_attn_output)
        x = self.layerNorm2((hidden_states, conditional_emb))
        ffn_output = self.feedForward(x)
        hidden_states = hidden_states + self.dropout2(ffn_output)
        return hidden_states


    class T5PositionWiseFeedForward(PositionWiseFeedForward):
        """参考transformer包: https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py
        """

        def __init__(self, hidden_size, intermediate_size, **kwargs):
            super().__init__(hidden_size, intermediate_size, **kwargs)
            self.intermediateDense = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.intermediateDense1 = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.outputDense = nn.Linear(intermediate_size, hidden_size, bias=False)

        def forward(self, x):
            x_gelu = self.intermediate_act_fn(self.intermediateDense(x))
            x_linear = self.intermediateDense1(x)
            x = x_gelu * x_linear
            if self.is_dropout:
                x = self.dropout(x)
            x = self.outputDense(x)
            return x


class XlnetLayer(BertLayer):
    """Transformer_XL层
    顺序为: Attention --> Add --> LayerNorm --> Feed Forward --> Add --> LayerNorm
    """

    def __init__(self, hidden_size, num_attention_heads, dropout_rate, attention_probs_dropout_prob, intermediate_size, hidden_act, **kwargs):
        super().__init__(hidden_size, num_attention_heads, dropout_rate, attention_probs_dropout_prob, intermediate_size, hidden_act, **kwargs)
        self.pre_lnorm = kwargs.get('pre_lnorm')
        self.multiHeadAttention = self.RelPartialLearnableMultiHeadAttn(hidden_size, num_attention_heads, attention_probs_dropout_prob, bias=False, **kwargs)

    def forward(self, hidden_states, segment_ids, pos_emb, attention_mask, mems_i, conditional_emb=None):
        hidden_states_cat = torch.cat([mems_i, hidden_states], 1) if mems_i is not None else hidden_states
        if self.pre_lnorm:
            hidden_states_cat = self.layerNorm1((hidden_states_cat, conditional_emb))
        self_attn_output = self.multiHeadAttention(hidden_states, hidden_states_cat, pos_emb, attention_mask, segment_ids)
        hidden_states = hidden_states + self.dropout1(self_attn_output)
        if not self.pre_lnorm:
            hidden_states = self.layerNorm1((hidden_states, conditional_emb))
        x = self.layerNorm2((hidden_states, conditional_emb)) if self.pre_lnorm else hidden_states
        self_attn_output2 = self.feedForward(x)
        hidden_states = hidden_states + self.dropout2(self_attn_output2)
        if not self.pre_lnorm:
            hidden_states = self.layerNorm2((hidden_states, conditional_emb))
        return hidden_states


    class RelPartialLearnableMultiHeadAttn(MultiHeadAttentionLayer):
        """Transformer_XL式相对位置编码, 这里修改成了MultiHeadAttentionLayer的batch_first代码格式
        """

        def __init__(self, *args, r_w_bias=None, r_r_bias=None, r_s_bias=None, **kwargs):
            super().__init__(*args, **kwargs)
            segment_vocab_size = kwargs.get('segment_vocab_size')
            if r_r_bias is None or r_w_bias is None:
                self.r_r_bias = nn.Parameter(torch.FloatTensor(self.num_attention_heads, self.attention_head_size))
                self.r_w_bias = nn.Parameter(torch.FloatTensor(self.num_attention_heads, self.attention_head_size))
                if segment_vocab_size > 0:
                    self.r_s_bias = nn.Parameter(torch.FloatTensor(self.num_attention_heads, self.attention_head_size))
            else:
                self.r_r_bias = r_r_bias
                self.r_w_bias = r_w_bias
                self.r_s_bias = r_s_bias
            if segment_vocab_size > 0:
                self.seg_embed = nn.Parameter(torch.FloatTensor(segment_vocab_size, self.num_attention_heads, self.attention_head_size))
            self.r = nn.Linear(self.hidden_size, self.hidden_size, bias=self.bias)
            self.rel_shift_opt = kwargs.get('rel_shift_opt')

        @staticmethod
        def rel_shift(x, zero_triu=False):
            """transformer_xl使用, 向左shift让右上角都是0, 对角线是同一个值, x: [btz, n_head, q_len, k_len]
            """
            q_len, k_len = x.size(2), x.size(-1)
            zero_pad = torch.zeros((*x.size()[:2], q_len, 1), device=x.device, dtype=x.dtype)
            x_padded = torch.cat([zero_pad, x], dim=-1)
            x_padded = x_padded.view(*x.size()[:2], k_len + 1, q_len)
            x = x_padded[:, :, 1:, :].view_as(x)
            if zero_triu:
                ones = torch.ones((q_len, k_len), device=x.device)
                x = x * torch.tril(ones, k_len - q_len)[None, None, :, :]
            return x

        @staticmethod
        def rel_shift_bnij(x, klen=-1):
            """ xlnet使用
            """
            x_size = x.shape
            x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
            x = x[:, :, 1:, :]
            x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
            x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
            return x

        def forward(self, w, cat, r, attention_mask=None, seg_mat=None):
            qlen, rlen, bsz = w.size(1), r.size(0), w.size(0)
            mixed_query_layer = self.q(cat)[:, -qlen:, :]
            mixed_key_layer = self.k(cat)
            mixed_value_layer = self.v(cat)
            w_head_q = self.transpose_for_scores(mixed_query_layer)
            w_head_k = self.transpose_for_scores(mixed_key_layer)
            w_head_v = self.transpose_for_scores(mixed_value_layer)
            r_head_k = self.r(r)
            r_head_k = r_head_k.view(rlen, self.num_attention_heads, self.attention_head_size)
            rw_head_q = w_head_q + self.r_w_bias.unsqueeze(1)
            AC = torch.einsum('bnid,bnjd->bnij', (rw_head_q, w_head_k))
            rr_head_q = w_head_q + self.r_r_bias.unsqueeze(1)
            BD = torch.einsum('bnid,jnd->bnij', (rr_head_q, r_head_k))
            BD = self.rel_shift_bnij(BD, klen=AC.shape[3]) if self.rel_shift_opt == 'xlnet' else self.rel_shift(BD)
            if hasattr(self, 'seg_embed') and self.r_r_bias is not None:
                seg_mat = F.one_hot(seg_mat, 2).float()
                EF = torch.einsum('bnid,snd->ibns', w_head_q + self.r_s_bias.unsqueeze(1), self.seg_embed)
                EF = torch.einsum('bijs,ibns->bnij', seg_mat, EF)
            else:
                EF = 0
            attention_scores = AC + BD + EF
            if self.attention_scale:
                attention_scores = attention_scores / math.sqrt(self.attention_head_size)
            if attention_mask is not None and attention_mask.any().item():
                attention_mask = 1.0 - attention_mask
                attention_scores = attention_scores.float().masked_fill(attention_mask.bool(), -1e+30).type_as(attention_mask)
            attention_probs = F.softmax(attention_scores, dim=-1)
            attention_probs = self.dropout(attention_probs)
            context_layer = torch.matmul(attention_probs, w_head_v)
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
            context_layer = context_layer.view(*new_context_layer_shape)
            if self.return_attention_scores:
                return self.o(context_layer), attention_scores
            else:
                return self.o(context_layer)


class AdaptiveEmbedding(nn.Module):
    """Transformer_XL的自适应embedding, 实现不同区间使用不同的维度
    可以实现如高频词用比如1024或512维，低频词用256或64维, 再用Linear层project到相同的维数
    """

    def __init__(self, vocab_size, embedding_size, hidden_size, cutoffs, div_val=1, sample_softmax=False, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.cutoffs = cutoffs + [vocab_size]
        self.div_val = div_val
        self.hidden_size = hidden_size
        self.emb_scale = hidden_size ** 0.5
        self.cutoff_ends = [0] + self.cutoffs
        self.emb_layers = nn.ModuleList()
        self.emb_projs = nn.ParameterList()
        if div_val == 1:
            self.emb_layers.append(nn.Embedding(vocab_size, embedding_size, sparse=sample_softmax > 0))
            if hidden_size != embedding_size:
                self.emb_projs.append(nn.Parameter(torch.FloatTensor(hidden_size, embedding_size)))
        else:
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                d_emb_i = embedding_size // div_val ** i
                self.emb_layers.append(nn.Embedding(r_idx - l_idx, d_emb_i))
                self.emb_projs.append(nn.Parameter(torch.FloatTensor(hidden_size, d_emb_i)))

    def forward(self, token_ids):
        if self.div_val == 1:
            embed = self.emb_layers[0](token_ids)
            if self.hidden_size != self.embedding_size:
                embed = nn.functional.linear(embed, self.emb_projs[0])
        else:
            param = next(self.parameters())
            inp_flat = token_ids.view(-1)
            emb_flat = torch.zeros([inp_flat.size(0), self.hidden_size], dtype=param.dtype, device=param.device)
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                mask_i = (inp_flat >= l_idx) & (inp_flat < r_idx)
                indices_i = mask_i.nonzero().squeeze()
                if indices_i.numel() == 0:
                    continue
                inp_i = inp_flat.index_select(0, indices_i) - l_idx
                emb_i = self.emb_layers[i](inp_i)
                emb_i = nn.functional.linear(emb_i, self.emb_projs[i])
                emb_flat.index_copy_(0, indices_i, emb_i)
            embed_shape = token_ids.size() + (self.hidden_size,)
            embed = emb_flat.view(embed_shape)
        embed.mul_(self.emb_scale)
        return embed


class Identity(nn.Module):

    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, *args):
        return args[0]


class XlnetPositionsEncoding(nn.Module):
    """Xlnet, transformer_xl使用的相对位置编码
       和SinusoidalPositionEncoding区别是一个是间隔排列, 一个是前后排列
    """

    def __init__(self, embedding_size):
        super().__init__()
        self.demb = embedding_size
        inv_freq = 1 / 10000 ** (torch.arange(0.0, embedding_size, 2.0) / embedding_size)
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        return pos_emb


class CRF(nn.Module):
    """Conditional random field: https://github.com/lonePatient/BERT-NER-Pytorch/blob/master/models/layers/crf.py
    """

    def __init__(self, num_tags: int, init_transitions: Optional[List[np.ndarray]]=None, freeze=False) ->None:
        if num_tags <= 0:
            raise ValueError(f'invalid number of tags: {num_tags}')
        super().__init__()
        self.num_tags = num_tags
        if init_transitions is None and not freeze:
            self.start_transitions = nn.Parameter(torch.empty(num_tags))
            self.end_transitions = nn.Parameter(torch.empty(num_tags))
            self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
            nn.init.uniform_(self.start_transitions, -0.1, 0.1)
            nn.init.uniform_(self.end_transitions, -0.1, 0.1)
            nn.init.uniform_(self.transitions, -0.1, 0.1)
        elif init_transitions is not None:
            transitions = torch.tensor(init_transitions[0], dtype=torch.float)
            start_transitions = torch.tensor(init_transitions[1], dtype=torch.float)
            end_transitions = torch.tensor(init_transitions[2], dtype=torch.float)
            if not freeze:
                self.transitions = nn.Parameter(transitions)
                self.start_transitions = nn.Parameter(start_transitions)
                self.end_transitions = nn.Parameter(end_transitions)
            else:
                self.register_buffer('transitions', transitions)
                self.register_buffer('start_transitions', start_transitions)
                self.register_buffer('end_transitions', end_transitions)

    def __repr__(self) ->str:
        return f'{self.__class__.__name__}(num_tags={self.num_tags})'

    def forward(self, emissions: torch.Tensor, mask: torch.ByteTensor, tags: torch.LongTensor, reduction: str='mean') ->torch.Tensor:
        """Compute the conditional log likelihood of a sequence of tags given emission scores.
            emissions: [btz, seq_len, num_tags]
            mask: [btz, seq_len]
            tags: [btz, seq_len]
        """
        if reduction not in ('none', 'sum', 'mean', 'token_mean'):
            raise ValueError(f'invalid reduction: {reduction}')
        if mask.dtype != torch.uint8:
            mask = mask.byte()
        self._validate(emissions, tags=tags, mask=mask)
        numerator = self._compute_score(emissions, tags, mask)
        denominator = self._compute_normalizer(emissions, mask)
        llh = denominator - numerator
        if reduction == 'none':
            return llh
        if reduction == 'sum':
            return llh.sum()
        if reduction == 'mean':
            return llh.mean()
        return llh.sum() / mask.float().sum()

    def decode(self, emissions: torch.Tensor, mask: Optional[torch.ByteTensor]=None, nbest: Optional[int]=None, pad_tag: Optional[int]=None) ->List[List[List[int]]]:
        """Find the most likely tag sequence using Viterbi algorithm.
        """
        if nbest is None:
            nbest = 1
        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.uint8, device=emissions.device)
        if mask.dtype != torch.uint8:
            mask = mask.byte()
        self._validate(emissions, mask=mask)
        best_path = self._viterbi_decode_nbest(emissions, mask, nbest, pad_tag)
        return best_path[0] if nbest == 1 else best_path

    def _validate(self, emissions: torch.Tensor, tags: Optional[torch.LongTensor]=None, mask: Optional[torch.ByteTensor]=None) ->None:
        if emissions.dim() != 3:
            raise ValueError(f'emissions must have dimension of 3, got {emissions.dim()}')
        if emissions.size(2) != self.num_tags:
            raise ValueError(f'expected last dimension of emissions is {self.num_tags}, got {emissions.size(2)}')
        if tags is not None:
            if emissions.shape[:2] != tags.shape:
                raise ValueError(f'the first two dimensions of emissions and tags must match, got {tuple(emissions.shape[:2])} and {tuple(tags.shape)}')
        if mask is not None:
            if emissions.shape[:2] != mask.shape:
                raise ValueError(f'the first two dimensions of emissions and mask must match, got {tuple(emissions.shape[:2])} and {tuple(mask.shape)}')
            no_empty_seq_bf = mask[:, 0].all()
            if not no_empty_seq_bf:
                raise ValueError('mask of the first timestep must all be on')

    def _compute_score(self, emissions: torch.Tensor, tags: torch.LongTensor, mask: torch.ByteTensor) ->torch.Tensor:
        batch_size, seq_length = tags.shape
        mask = mask.float()
        score = self.start_transitions[tags[:, 0]]
        score += emissions[torch.arange(batch_size), 0, tags[:, 0]]
        for i in range(1, seq_length):
            score += self.transitions[tags[:, i - 1], tags[:, i]] * mask[:, i]
            score += emissions[torch.arange(batch_size), i, tags[:, i]] * mask[:, i]
        seq_ends = mask.long().sum(dim=1) - 1
        last_tags = tags[torch.arange(batch_size), seq_ends]
        score += self.end_transitions[last_tags]
        return score

    def _compute_normalizer(self, emissions: torch.Tensor, mask: torch.ByteTensor) ->torch.Tensor:
        seq_length = emissions.size(1)
        score = self.start_transitions + emissions[:, 0]
        for i in range(1, seq_length):
            broadcast_score = score.unsqueeze(2)
            broadcast_emissions = emissions[:, i].unsqueeze(1)
            next_score = broadcast_score + self.transitions + broadcast_emissions
            next_score = torch.logsumexp(next_score, dim=1)
            score = torch.where(mask[:, i].unsqueeze(1).bool(), next_score, score)
        score += self.end_transitions
        return torch.logsumexp(score, dim=1)

    def _viterbi_decode_nbest(self, emissions: torch.FloatTensor, mask: torch.ByteTensor, nbest: int, pad_tag: Optional[int]=None) ->List[List[List[int]]]:
        if pad_tag is None:
            pad_tag = 0
        device = emissions.device
        batch_size, seq_length = mask.shape
        score = self.start_transitions + emissions[:, 0]
        history_idx = torch.zeros((batch_size, seq_length, self.num_tags, nbest), dtype=torch.long, device=device)
        oor_idx = torch.zeros((batch_size, self.num_tags, nbest), dtype=torch.long, device=device)
        oor_tag = torch.full((batch_size, seq_length, nbest), pad_tag, dtype=torch.long, device=device)
        for i in range(1, seq_length):
            if i == 1:
                broadcast_score = score.unsqueeze(-1)
                broadcast_emission = emissions[:, i].unsqueeze(1)
                next_score = broadcast_score + self.transitions + broadcast_emission
            else:
                broadcast_score = score.unsqueeze(-1)
                broadcast_emission = emissions[:, i].unsqueeze(1).unsqueeze(2)
                next_score = broadcast_score + self.transitions.unsqueeze(1) + broadcast_emission
            next_score, indices = next_score.view(batch_size, -1, self.num_tags).topk(nbest, dim=1)
            if i == 1:
                score = score.unsqueeze(-1).expand(-1, -1, nbest)
                indices = indices * nbest
            next_score = next_score.transpose(2, 1)
            indices = indices.transpose(2, 1)
            score = torch.where(mask[:, i].unsqueeze(-1).unsqueeze(-1).bool(), next_score, score)
            indices = torch.where(mask[:, i].unsqueeze(-1).unsqueeze(-1).bool(), indices, oor_idx)
            history_idx[:, i - 1] = indices
        end_score = score + self.end_transitions.unsqueeze(-1)
        _, end_tag = end_score.view(batch_size, -1).topk(nbest, dim=1)
        seq_ends = mask.long().sum(dim=1) - 1
        history_idx.scatter_(1, seq_ends.view(-1, 1, 1, 1).expand(-1, 1, self.num_tags, nbest), end_tag.view(-1, 1, 1, nbest).expand(-1, 1, self.num_tags, nbest))
        best_tags_arr = torch.zeros((batch_size, seq_length, nbest), dtype=torch.long, device=device)
        best_tags = torch.arange(nbest, dtype=torch.long, device=device).view(1, -1).expand(batch_size, -1)
        for idx in range(seq_length - 1, -1, -1):
            best_tags = torch.gather(history_idx[:, idx].view(batch_size, -1), 1, best_tags)
            best_tags_arr[:, idx] = torch_div(best_tags.data.view(batch_size, -1), nbest, rounding_mode='floor')
        return torch.where(mask.unsqueeze(-1).bool(), best_tags_arr, oor_tag).permute(2, 0, 1)


class GlobalPointer(nn.Module):
    """全局指针模块
    将序列的每个(start, end)作为整体来进行判断
    参考：https://kexue.fm/archives/8373
    """

    def __init__(self, hidden_size, heads, head_size, RoPE=True, max_len=512, use_bias=True, tril_mask=True):
        super().__init__()
        self.heads = heads
        self.head_size = head_size
        self.RoPE = RoPE
        self.tril_mask = tril_mask
        self.dense = nn.Linear(hidden_size, heads * head_size * 2, bias=use_bias)
        if self.RoPE:
            self.position_embedding = RoPEPositionEncoding(max_len, head_size)

    def forward(self, inputs, mask=None):
        """ 
        :param inputs: shape=[..., hdsz]
        :param mask: shape=[btz, seq_len], padding部分为0
        """
        sequence_output = self.dense(inputs)
        sequence_output = torch.stack(torch.chunk(sequence_output, self.heads, dim=-1), dim=-2)
        qw, kw = sequence_output[..., :self.head_size], sequence_output[..., self.head_size:]
        if self.RoPE:
            qw = self.position_embedding(qw)
            kw = self.position_embedding(kw)
        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)
        if mask is not None:
            attention_mask1 = 1 - mask.unsqueeze(1).unsqueeze(3)
            attention_mask2 = 1 - mask.unsqueeze(1).unsqueeze(2)
            logits = logits.masked_fill(attention_mask1.bool(), value=-float('inf'))
            logits = logits.masked_fill(attention_mask2.bool(), value=-float('inf'))
        if self.tril_mask:
            logits = logits - torch.tril(torch.ones_like(logits), -1) * 1000000000000.0
        return logits / self.head_size ** 0.5


class EfficientGlobalPointer(nn.Module):
    """更加参数高效的GlobalPointer
    参考：https://kexue.fm/archives/8877
    这里实现和GlobalPointer相似，而未采用原版的奇偶位来取qw和kw，个人理解两种方式是无区别的
    """

    def __init__(self, hidden_size, heads, head_size, RoPE=True, max_len=512, use_bias=True, tril_mask=True):
        super().__init__()
        self.heads = heads
        self.head_size = head_size
        self.RoPE = RoPE
        self.tril_mask = tril_mask
        self.p_dense = nn.Linear(hidden_size, head_size * 2, bias=use_bias)
        self.q_dense = nn.Linear(head_size * 2, heads * 2, bias=use_bias)
        if self.RoPE:
            self.position_embedding = RoPEPositionEncoding(max_len, head_size)

    def forward(self, inputs, mask=None):
        """ 
        :param inputs: shape=[..., hdsz]
        :param mask: shape=[btz, seq_len], padding部分为0
        """
        sequence_output = self.p_dense(inputs)
        qw, kw = sequence_output[..., :self.head_size], sequence_output[..., self.head_size:]
        if self.RoPE:
            qw = self.position_embedding(qw)
            kw = self.position_embedding(kw)
        logits = torch.einsum('bmd,bnd->bmn', qw, kw) / self.head_size ** 0.5
        bias_input = self.q_dense(sequence_output)
        bias = torch.stack(torch.chunk(bias_input, self.heads, dim=-1), dim=-2).transpose(1, 2) / 2
        logits = logits.unsqueeze(1) + bias[..., :1] + bias[..., 1:].transpose(2, 3)
        if mask is not None:
            attention_mask1 = 1 - mask.unsqueeze(1).unsqueeze(3)
            attention_mask2 = 1 - mask.unsqueeze(1).unsqueeze(2)
            logits = logits.masked_fill(attention_mask1.bool(), value=-float('inf'))
            logits = logits.masked_fill(attention_mask2.bool(), value=-float('inf'))
        if self.tril_mask:
            logits = logits - torch.tril(torch.ones_like(logits), -1) * 1000000000000.0
        return logits


class TplinkerHandshakingKernel(nn.Module):
    """Tplinker的HandshakingKernel实现
    """

    def __init__(self, hidden_size, shaking_type, inner_enc_type=''):
        super().__init__()
        self.shaking_type = shaking_type
        if shaking_type == 'cat':
            self.combine_fc = nn.Linear(hidden_size * 2, hidden_size)
        elif shaking_type == 'cat_plus':
            self.combine_fc = nn.Linear(hidden_size * 3, hidden_size)
        elif shaking_type == 'cln':
            self.tp_cln = LayerNorm(hidden_size, conditional_size=hidden_size)
        elif shaking_type == 'cln_plus':
            self.tp_cln = LayerNorm(hidden_size, conditional_size=hidden_size)
            self.inner_context_cln = LayerNorm(hidden_size, conditional_size=hidden_size)
        self.inner_enc_type = inner_enc_type
        if inner_enc_type == 'mix_pooling':
            self.lamtha = nn.Parameter(torch.rand(hidden_size))
        elif inner_enc_type == 'lstm':
            self.inner_context_lstm = nn.LSTM(hidden_size, hidden_size, num_layers=1, bidirectional=False, batch_first=True)

    def enc_inner_hiddens(self, seq_hiddens, inner_enc_type='lstm'):

        def pool(seqence, pooling_type):
            if pooling_type == 'mean_pooling':
                pooling = torch.mean(seqence, dim=-2)
            elif pooling_type == 'max_pooling':
                pooling, _ = torch.max(seqence, dim=-2)
            elif pooling_type == 'mix_pooling':
                pooling = self.lamtha * torch.mean(seqence, dim=-2) + (1 - self.lamtha) * torch.max(seqence, dim=-2)[0]
            return pooling
        if 'pooling' in inner_enc_type:
            inner_context = torch.stack([pool(seq_hiddens[:, :i + 1, :], inner_enc_type) for i in range(seq_hiddens.size()[1])], dim=1)
        elif inner_enc_type == 'lstm':
            inner_context, _ = self.inner_context_lstm(seq_hiddens)
        return inner_context

    def forward(self, seq_hiddens):
        """
        :param seq_hiddens: (batch_size, seq_len, hidden_size)
        :return: shaking_hiddenss: (batch_size, (1 + seq_len) * seq_len / 2, hidden_size) (32, 5+4+3+2+1, 5)
        """
        seq_len = seq_hiddens.size()[-2]
        shaking_hiddens_list = []
        for ind in range(seq_len):
            hidden_each_step = seq_hiddens[:, ind, :]
            visible_hiddens = seq_hiddens[:, ind:, :]
            repeat_hiddens = hidden_each_step[:, None, :].repeat(1, seq_len - ind, 1)
            if self.shaking_type == 'cat':
                shaking_hiddens = torch.cat([repeat_hiddens, visible_hiddens], dim=-1)
                shaking_hiddens = torch.tanh(self.combine_fc(shaking_hiddens))
            elif self.shaking_type == 'cat_plus':
                inner_context = self.enc_inner_hiddens(visible_hiddens, self.inner_enc_type)
                shaking_hiddens = torch.cat([repeat_hiddens, visible_hiddens, inner_context], dim=-1)
                shaking_hiddens = torch.tanh(self.combine_fc(shaking_hiddens))
            elif self.shaking_type == 'cln':
                shaking_hiddens = self.tp_cln([visible_hiddens, repeat_hiddens])
            elif self.shaking_type == 'cln_plus':
                inner_context = self.enc_inner_hiddens(visible_hiddens, self.inner_enc_type)
                shaking_hiddens = self.tp_cln([visible_hiddens, repeat_hiddens])
                shaking_hiddens = self.inner_context_cln([shaking_hiddens, inner_context])
            shaking_hiddens_list.append(shaking_hiddens)
        long_shaking_hiddens = torch.cat(shaking_hiddens_list, dim=1)
        return long_shaking_hiddens


class MixUp(nn.Module):
    """mixup方法实现
    
    :param method: str, 可选'embed', ’encoder‘分别表示在embedding和encoder层面做mixup, None表示mix后续处理, ’hidden‘表示对隐含层做mixup
    :param alpha: float
    :param layer_mix: None or int, 需要mix的隐含层index
    """

    def __init__(self, method='encoder', alpha=1.0, layer_mix=None):
        super().__init__()
        assert method in {'embed', 'encoder', 'hidden', None}
        self.method = method
        self.alpha = alpha
        self.perm_index = None
        self.lam = 0
        self.layer_mix = layer_mix

    def get_perm(self, inputs):
        if isinstance(inputs, torch.Tensor):
            return inputs[self.perm_index]
        elif isinstance(inputs, (list, tuple)):
            return [(inp[self.perm_index] if isinstance(inp, torch.Tensor) else inp) for inp in inputs]

    def mix_up(self, output, output1):
        if isinstance(output, torch.Tensor):
            return self.lam * output + (1.0 - self.lam) * output1
        elif isinstance(output, (list, tuple)):
            output_final = []
            for i in range(len(output)):
                if output[i] is None:
                    output_final.append(output[i])
                elif not output[i].requires_grad and output[i].dtype in {torch.long, torch.int}:
                    output_final.append(torch.max(output[i], output1[i]))
                else:
                    output_final.append(self.lam * output[i] + (1.0 - self.lam) * output1[i])
            return output_final
        else:
            raise ValueError('Illegal model output')

    def encode(self, model, inputs):
        batch_size = inputs[0].shape[0]
        device = inputs[0].device
        self.lam = np.random.beta(self.alpha, self.alpha)
        self.perm_index = torch.randperm(batch_size)
        if self.method is None:
            output = model(inputs)
            output1 = self.get_perm(output)
            return [output, output1]
        elif self.method == 'encoder':
            output = model(inputs)
            output1 = self.get_perm(output)
            output_final = self.mix_up(output, output1)
        elif self.method == 'embed':
            output = model.apply_embeddings(inputs)
            output1 = self.get_perm(output)
            output_final = self.mix_up(output, output1)
            output_final = model.apply_main_layers(output_final)
            output_final = model.apply_final_layers(output_final)
        elif self.method == 'hidden':
            if self.layer_mix is None:
                try:
                    layer_mix = random.randint(0, len(model.encoderLayer))
                except:
                    warnings.warn('LayerMix random failded')
                    layer_mix = 0
            else:
                layer_mix = self.layer_mix

            def apply_on_layer_end(l_i, output):
                if l_i == layer_mix:
                    output1 = self.get_perm(output)
                    return self.mix_up(output, output1)
                else:
                    return output
            model.apply_on_layer_end = apply_on_layer_end
            output_final = model(inputs)
        return output_final

    def forward(self, criterion, y_pred, y_true):
        """计算loss
        """
        y_true1 = y_true[self.perm_index]
        return self.lam * criterion(y_pred, y_true) + (1 - self.lam) * criterion(y_pred, y_true1)


class ConvLayer(nn.Module):
    """deberta_v2中使用
    """

    def __init__(self, hidden_size, dropout_rate=0.1, layer_norm_eps=1e-12, conv_kernel_size=3, conv_groups=1, conv_act='tanh', **kwargs):
        super().__init__()
        kernel_size = conv_kernel_size
        groups = conv_groups
        self.conv_act = conv_act
        self.conv = nn.Conv1d(hidden_size, hidden_size, kernel_size, padding=(kernel_size - 1) // 2, groups=groups)
        self.LayerNorm = nn.LayerNorm(hidden_size, layer_norm_eps)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, hidden_states, residual_states, input_mask):
        out = self.conv(hidden_states.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        rmask = (1 - input_mask).bool()
        out.masked_fill_(rmask.unsqueeze(-1).expand(out.size()), 0)
        out = get_activation(self.conv_act)(self.dropout(out))
        layer_norm_input = residual_states + out
        output = self.LayerNorm(layer_norm_input)
        if input_mask is None:
            output_states = output
        else:
            if input_mask.dim() != layer_norm_input.dim():
                if input_mask.dim() == 4:
                    input_mask = input_mask.squeeze(1).squeeze(1)
                input_mask = input_mask.unsqueeze(2)
            input_mask = input_mask
            output_states = output * input_mask
        return output_states


class MultiSampleDropout(nn.Module):
    """multisample dropout (wut): https://arxiv.org/abs/1905.09788
    """

    def __init__(self, hidden_size, num_labels, K=5, p=0.5):
        super().__init__()
        self.K = K
        self.dropout = nn.Dropout(p)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input):
        logits = torch.stack([self.classifier(self.dropout(input)) for _ in range(self.K)], dim=0)
        logits = torch.mean(logits, dim=0)
        return logits


class FocalLoss(nn.Module):
    """Multi-class Focal loss implementation"""

    def __init__(self, gamma=2, weight=None, ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, input, target):
        """
        :param input: torch.Tensor, shape=[N, C]
        :param target: torch.Tensor, shape=[N, ]
        """
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1 - pt) ** self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.weight, ignore_index=self.ignore_index)
        return loss


class LabelSmoothingCrossEntropy(nn.Module):

    def __init__(self, eps=0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, output, target):
        """
        :param output: torch.Tensor, shape=[N, C]
        :param target: torch.Tensor, shape=[N, ]
        """
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction == 'sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction == 'mean':
                loss = loss.mean()
        return loss * self.eps / c + (1 - self.eps) * F.nll_loss(log_preds, target, reduction=self.reduction, ignore_index=self.ignore_index)


class MultilabelCategoricalCrossentropy(nn.Module):
    """多标签分类的交叉熵；
    说明：y_true和y_pred的shape一致，y_true的元素非0即1， 1表示对应的类为目标类，0表示对应的类为非目标类。
    警告：请保证y_pred的值域是全体实数，换言之一般情况下y_pred不用加激活函数，尤其是不能加sigmoid或者softmax！预测阶段则输出y_pred大于0的类。如有疑问，请仔细阅读并理解本文。
    参考：https://kexue.fm/archives/7359
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, y_pred, y_true):
        """
        :param y_true: torch.Tensor, [..., num_classes]
        :param y_pred: torch.Tensor: [..., num_classes]
        """
        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_pos = y_pred - (1 - y_true) * 1000000000000.0
        y_pred_neg = y_pred - y_true * 1000000000000.0
        y_pred_pos = torch.cat([y_pred_pos, torch.zeros_like(y_pred_pos[..., :1])], dim=-1)
        y_pred_neg = torch.cat([y_pred_neg, torch.zeros_like(y_pred_neg[..., :1])], dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        return (pos_loss + neg_loss).mean()


class SparseMultilabelCategoricalCrossentropy(nn.Module):
    """稀疏版多标签分类的交叉熵；
       请保证y_pred的值域是全体实数，换言之一般情况下y_pred不用加激活函数，尤其是不能加sigmoid或者softmax，预测阶段则输出y_pred大于0的类；
       详情请看：https://kexue.fm/archives/7359 。
    """

    def __init__(self, mask_zero=False, epsilon=1e-07, **kwargs):
        super().__init__(**kwargs)
        self.mask_zero = mask_zero
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        """
        :param y_true: shape=[..., num_positive]
        :param y_pred: shape=[..., num_classes]
        """
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred = torch.cat([y_pred, zeros], dim=-1)
        if self.mask_zero:
            infs = zeros + float('inf')
            y_pred = torch.cat([infs, y_pred[..., 1:]], dim=-1)
        y_pos_2 = torch.gather(y_pred, dim=-1, index=y_true)
        y_pos_1 = torch.cat([y_pos_2, zeros], dim=-1)
        if self.mask_zero:
            y_pred = torch.cat([-infs, y_pred[..., 1:]], dim=-1)
            y_pos_2 = torch.gather(y_pred, dim=-1, index=y_true)
        pos_loss = torch.logsumexp(-y_pos_1, dim=-1)
        all_loss = torch.logsumexp(y_pred, dim=-1)
        aux_loss = torch.logsumexp(y_pos_2, dim=-1) - all_loss
        aux_loss = torch.clamp(1 - torch.exp(aux_loss), self.epsilon, 1)
        neg_loss = all_loss + torch.log(aux_loss)
        return pos_loss + neg_loss


class ContrastiveLoss(nn.Module):
    """对比损失：减小正例之间的距离，增大正例和反例之间的距离
    公式：labels * distance_matrix.pow(2) + (1-labels)*F.relu(margin-distance_matrix).pow(2)
    https://www.sbert.net/docs/package_reference/losses.html

    :param margin: float, 距离参数，distance>margin时候不参加梯度回传，默认为0.5
    :param size_average: bool, 是否对loss在样本维度上求均值，默认为True
    :param online: bool, 是否使用OnlineContrastiveLoss, 即仅计算困难样本的loss, 默认为False
    """

    def __init__(self, margin=0.5, size_average=True, online=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.size_average = size_average
        self.online = online

    def forward(self, distances, labels, pos_id=1, neg_id=0):
        """
        :param distances: torch.Tensor, 样本对之间的预测距离, shape=[btz]
        :param labels: torch.Tensor, 样本对之间的真实距离, shape=[btz]
        :param pos_id: int, 正样本的label
        :param neg_id: int, 负样本的label
        """
        if not self.online:
            losses = 0.5 * (labels.float() * distances.pow(2) + (1 - labels).float() * F.relu(self.margin - distances).pow(2))
            return losses.mean() if self.size_average else losses.sum()
        else:
            negs = distances[labels == neg_id]
            poss = distances[labels == pos_id]
            negative_pairs = negs[negs < (poss.max() if len(poss) > 1 else negs.mean())]
            positive_pairs = poss[poss > (negs.min() if len(negs) > 1 else poss.mean())]
            positive_loss = positive_pairs.pow(2).sum()
            negative_loss = F.relu(self.margin - negative_pairs).pow(2).sum()
            return positive_loss + negative_loss


class RDropLoss(nn.Module):
    """R-Drop的Loss实现，官方项目：https://github.com/dropreg/R-Drop

    :param alpha: float, 控制rdrop的loss的比例
    :param rank: str, 指示y_pred的排列方式, 支持['adjacent', 'updown']
    """

    def __init__(self, alpha=4, rank='adjacent'):
        super().__init__()
        self.alpha = alpha
        assert rank in {'adjacent', 'updown'}, "rank kwarg only support 'adjacent' and 'updown' "
        self.rank = rank
        self.loss_sup = nn.CrossEntropyLoss()
        self.loss_rdrop = nn.KLDivLoss(reduction='none')

    def forward(self, *args):
        """支持两种方式: 一种是y_pred, y_true, 另一种是y_pred1, y_pred2, y_true

        :param y_pred: torch.Tensor, 第一种方式的样本预测值, shape=[btz*2, num_labels]
        :param y_true: torch.Tensor, 样本真实值, 第一种方式shape=[btz*2,], 第二种方式shape=[btz,]
        :param y_pred1: torch.Tensor, 第二种方式的样本预测值, shape=[btz, num_labels]
        :param y_pred2: torch.Tensor, 第二种方式的样本预测值, shape=[btz, num_labels]
        """
        assert len(args) in {2, 3}, 'RDropLoss only support 2 or 3 input args'
        if len(args) == 2:
            y_pred, y_true = args
            loss_sup = self.loss_sup(y_pred, y_true)
            if self.rank == 'adjacent':
                y_pred1 = y_pred[1::2]
                y_pred2 = y_pred[::2]
            elif self.rank == 'updown':
                half_btz = y_true.shape[0] // 2
                y_pred1 = y_pred[:half_btz]
                y_pred2 = y_pred[half_btz:]
        else:
            y_pred1, y_pred2, y_true = args
            loss_sup = self.loss_sup(y_pred1, y_true)
        loss_rdrop1 = self.loss_rdrop(F.log_softmax(y_pred1, dim=-1), F.softmax(y_pred2, dim=-1))
        loss_rdrop2 = self.loss_rdrop(F.log_softmax(y_pred2, dim=-1), F.softmax(y_pred1, dim=-1))
        return loss_sup + torch.mean(loss_rdrop1 + loss_rdrop2) / 4 * self.alpha


class UDALoss(nn.Module):
    """UDALoss，使用时候需要继承一下，因为forward需要使用到global_step和total_steps
    https://arxiv.org/abs/1904.12848

    :param tsa_schedule: str, tsa策略，可选['linear_schedule', 'exp_schedule', 'log_schedule']
    :param start_p: float, tsa生效概率下限, 默认为0
    :param end_p: float, tsa生效概率上限, 默认为1
    :param return_all_loss: bool, 是否返回所有的loss，默认为True
    :return: loss
    """

    def __init__(self, tsa_schedule=None, start_p=0, end_p=1, return_all_loss=True):
        super().__init__()
        self.loss_sup = nn.CrossEntropyLoss()
        self.loss_unsup = nn.KLDivLoss(reduction='batchmean')
        self.tsa_schedule = tsa_schedule
        self.start = start_p
        self.end = end_p
        if self.tsa_schedule:
            assert self.tsa_schedule in {'linear_schedule', 'exp_schedule', 'log_schedule'}, 'tsa_schedule config illegal'
        self.return_all_loss = return_all_loss

    def forward(self, y_pred, y_true_sup, global_step, total_steps):
        """ y_pred由[pred_sup, true_unsup, pred_unsup]三部分组成
        
        :param y_pred: torch.Tensor, 样本预测值, shape=[btz_sup+btz_unsup*2, num_labels]
        :param y_true_sup: torch.Tensor, 样本真实值, shape=[btz_sup,]
        :param global_step: int, 当前训练步数
        :param total_steps: int, 训练总步数
        """
        sup_size = y_true_sup.size(0)
        unsup_size = (y_pred.size(0) - sup_size) // 2
        y_pred_sup = y_pred[:sup_size]
        if self.tsa_schedule is None:
            loss_sup = self.loss_sup(y_pred_sup, y_true_sup)
        else:
            threshold = self.get_tsa_threshold(self.tsa_schedule, global_step, total_steps, self.start, self.end)
            true_prob = torch.gather(F.softmax(y_pred_sup, dim=-1), dim=1, index=y_true_sup[:, None])
            sel_rows = true_prob.lt(threshold).sum(dim=-1).gt(0)
            loss_sup = self.loss_sup(y_pred_sup[sel_rows], y_true_sup[sel_rows]) if sel_rows.sum() > 0 else 0
        y_true_unsup = y_pred[sup_size:sup_size + unsup_size]
        y_true_unsup = F.softmax(y_true_unsup.detach(), dim=-1)
        y_pred_unsup = F.log_softmax(y_pred[sup_size + unsup_size:], dim=-1)
        loss_unsup = self.loss_unsup(y_pred_unsup, y_true_unsup)
        if self.return_all_loss:
            return loss_sup + loss_unsup, loss_sup, loss_unsup
        else:
            return loss_sup + loss_unsup

    @staticmethod
    def get_tsa_threshold(schedule, global_step, num_train_steps, start, end):
        training_progress = global_step / num_train_steps
        if schedule == 'linear_schedule':
            threshold = training_progress
        elif schedule == 'exp_schedule':
            scale = 5
            threshold = math.exp((training_progress - 1) * scale)
        elif schedule == 'log_schedule':
            scale = 5
            threshold = 1 - math.exp(-training_progress * scale)
        return threshold * (end - start) + start


class TemporalEnsemblingLoss(nn.Module):
    """TemporalEnsembling的实现，思路是在监督loss的基础上，增加一个mse的一致性损失loss

       - 官方项目：https://github.com/s-laine/tempens
       - pytorch第三方实现：https://github.com/ferretj/temporal-ensembling
       - 使用的时候，train_dataloader的shffle必须未False
    """

    def __init__(self, epochs, max_val=10.0, ramp_up_mult=-5.0, alpha=0.5, max_batch_num=100, hist_device='cpu'):
        super().__init__()
        self.loss_sup = nn.CrossEntropyLoss()
        self.max_epochs = epochs
        self.max_val = max_val
        self.ramp_up_mult = ramp_up_mult
        self.alpha = alpha
        self.max_batch_num = max_batch_num
        self.hist_unsup = []
        self.hist_sup = []
        self.hist_device = hist_device
        self.hist_input_y = []
        assert (self.alpha >= 0) & (self.alpha < 1)

    def forward(self, y_pred_sup, y_pred_unsup, y_true_sup, epoch, bti):
        """
        :param y_pred_sup: torch.Tensor, 监督学习样本预测值, shape=[btz, num_labels]
        :param y_pred_unsup: torch.Tensor, 无监督学习样本预测值, shape=[btz, num_labels]
        :param y_true_sup: int, 监督学习样本真实值, shape=[btz,]
        :param epoch: int, 当前epoch
        :param bti: int, 当前batch的序号
        """
        self.same_batch_check(y_pred_sup, y_pred_unsup, y_true_sup, bti)
        if self.max_batch_num is None or bti < self.max_batch_num:
            self.init_hist(bti, y_pred_sup, y_pred_unsup)
            sup_ratio = float(len(y_pred_sup)) / (len(y_pred_sup) + len(y_pred_unsup))
            w = self.weight_schedule(epoch, sup_ratio)
            sup_loss, unsup_loss = self.temporal_loss(y_pred_sup, y_pred_unsup, y_true_sup, bti)
            self.hist_unsup[bti] = self.update(self.hist_unsup[bti], y_pred_unsup.detach(), epoch)
            self.hist_sup[bti] = self.update(self.hist_sup[bti], y_pred_sup.detach(), epoch)
            return sup_loss + w * unsup_loss, sup_loss, w * unsup_loss
        else:
            return self.loss_sup(y_pred_sup, y_true_sup)

    def same_batch_check(self, y_pred_sup, y_pred_unsup, y_true_sup, bti):
        """检测数据的前几个batch必须是一致的, 这里写死是10个
        """
        if bti >= 10:
            return
        if bti >= len(self.hist_input_y):
            self.hist_input_y.append(y_true_sup)
        else:
            err_msg = 'TemporalEnsemblingLoss requests the same sort dataloader, you may need to set train_dataloader shuffle=False'
            assert self.hist_input_y[bti].equal(y_true_sup), err_msg

    def update(self, hist, y_pred, epoch):
        """更新历史logit，利用alpha门控来控制比例
        """
        Z = self.alpha * hist + (1.0 - self.alpha) * y_pred
        output = Z * (1.0 / (1.0 - self.alpha ** (epoch + 1)))
        return output

    def weight_schedule(self, epoch, sup_ratio):
        max_val = self.max_val * sup_ratio
        if epoch == 0:
            return 0.0
        elif epoch >= self.max_epochs:
            return max_val
        return max_val * np.exp(self.ramp_up_mult * (1.0 - float(epoch) / self.max_epochs) ** 2)

    def temporal_loss(self, y_pred_sup, y_pred_unsup, y_true_sup, bti):

        def mse_loss(out1, out2):
            quad_diff = torch.sum((F.softmax(out1, dim=1) - F.softmax(out2, dim=1)) ** 2)
            return quad_diff / out1.data.nelement()
        sup_loss = self.loss_sup(y_pred_sup, y_true_sup)
        unsup_loss = mse_loss(y_pred_unsup, self.hist_unsup[bti])
        unsup_loss += mse_loss(y_pred_sup, self.hist_sup[bti])
        return sup_loss, unsup_loss

    def init_hist(self, bti, y_pred_sup, y_pred_unsup):
        if bti >= len(self.hist_sup):
            self.hist_sup.append(torch.zeros_like(y_pred_sup))
            self.hist_unsup.append(torch.zeros_like(y_pred_unsup))


class BERT_BASE(nn.Module):
    """模型基类
    """

    def __init__(self, vocab_size, hidden_size, num_hidden_layers, num_attention_heads, intermediate_size, hidden_act, dropout_rate=None, attention_probs_dropout_prob=None, embedding_size=None, attention_head_size=None, attention_key_size=None, initializer_range=0.02, sequence_length=None, keep_tokens=None, compound_tokens=None, residual_attention_scores=False, ignore_invalid_weights=False, keep_hidden_layers=None, hierarchical_position=None, gradient_checkpoint=False, **kwargs):
        super(BERT_BASE, self).__init__()
        if keep_tokens is not None:
            vocab_size = len(keep_tokens)
        if compound_tokens is not None:
            vocab_size += len(compound_tokens)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = attention_head_size or self.hidden_size // self.num_attention_heads
        self.attention_key_size = attention_key_size or self.attention_head_size
        self.intermediate_size = intermediate_size
        self.dropout_rate = dropout_rate or 0
        self.attention_probs_dropout_prob = attention_probs_dropout_prob or 0
        self.hidden_act = hidden_act
        self.embedding_size = embedding_size or hidden_size
        self.initializer_range = initializer_range
        self.sequence_length = sequence_length
        self.keep_tokens = keep_tokens
        self.compound_tokens = compound_tokens
        self.attention_bias = None
        self.position_bias = None
        self.attention_scores = None
        self.residual_attention_scores = residual_attention_scores
        self.ignore_invalid_weights = ignore_invalid_weights
        self.keep_hidden_layers = set(range(num_hidden_layers)) if keep_hidden_layers is None else set(keep_hidden_layers)
        self.hierarchical_position = hierarchical_position
        self.gradient_checkpoint = gradient_checkpoint

    def build(self, attention_caches=None, layer_norm_cond=None, layer_norm_cond_hidden_size=None, layer_norm_cond_hidden_act=None, additional_input_layers=None, **kwargs):
        """模型构建函数

        :param attention_caches: 为Attention的K,V的缓存序列字典，格式为{Attention层名: [K缓存, V缓存]}；
        :param layer_norm_*: 实现Conditional Layer Normalization时使用，用来实现以“固定长度向量”为条件的条件Bert，暂未使用
        """
        self.attention_caches = attention_caches or {}
        self.output_all_encoded_layers = kwargs.get('output_all_encoded_layers', False)

    def forward(self, inputs):
        """定义模型的训练流程
        
        :param inputs: List[torch.Tensor], 默认顺序是[token_ids, segment_ids(若有), position_ids(若有), custom_attention_mask(若有), conditional_input(若有)]
        :return: List[torch.Tensor] or torch.Tensor, 模型输出，默认顺序为[last_hidden_state/all_encoded_layers, pooled_output(若有), mlm_scores(若有), nsp_scores(若有)]
        """
        outputs = self.apply_embeddings(inputs)
        outputs = self.apply_main_layers(outputs)
        outputs = self.apply_final_layers(outputs)
        return outputs

    @torch.no_grad()
    def predict(self, inputs):
        """定义模型的预测流程
        
        :param inputs: List[torch.Tensor], 默认顺序是[token_ids, segment_ids(若有), position_ids(若有), custom_attention_mask(若有), conditional_input(若有)]
        :return: List[torch.Tensor] or torch.Tensor, 模型输出，默认顺序为[last_hidden_state/all_encoded_layers, pooled_output(若有), mlm_scores(若有), nsp_scores(若有)]
        """
        self.eval()
        return self.forward(inputs)

    def init_model_weights(self, module):
        """ 初始化权重
        """
        if isinstance(module, (nn.Linear, nn.Embedding)) and module.weight.requires_grad:
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, LayerNorm):
            if hasattr(module, 'bias') and module.bias.requires_grad:
                module.bias.data.zero_()
            if hasattr(module, 'weight') and module.weight.requires_grad:
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None and module.bias.requires_grad:
            module.bias.data.zero_()

    def variable_mapping(self):
        """构建pytorch层与checkpoint的变量名之间的映射表
        """
        return {}

    def load_variable(self):
        raise NotImplementedError

    def load_embeddings(self, embeddings):
        """根据keep_tokens和compound_tokens对embedding进行修改
        """
        if self.keep_tokens is not None:
            embeddings = embeddings[self.keep_tokens]
        if self.compound_tokens is not None:
            ext_embeddings = []
            for item in self.compound_tokens:
                try:
                    ext_embeddings.append(torch.mean(embeddings[item], 0) * torch.ones_like(embeddings[item]))
                except IndexError:
                    ext_embeddings.append(torch.mean(embeddings, 0, keepdim=True))
                    warnings.warn(f'Initialize ext_embeddings from compound_tokens not in embedding index')
            embeddings = torch.cat([embeddings] + ext_embeddings, 0)
        return embeddings

    def load_pos_embeddings(self, embeddings):
        """根据hierarchical_position对pos_embedding进行修改
        """
        if self.hierarchical_position is not None:
            alpha = 0.4 if self.hierarchical_position is True else self.hierarchical_position
            embeddings = embeddings - alpha * embeddings[:1]
            embeddings = embeddings / (1 - alpha)
            position_index = torch.arange(self.max_position)[:, None]
            embeddings_x = take_along_dim(embeddings, torch_div(position_index, embeddings.size(0), rounding_mode='trunc'), dim=0)
            embeddings_y = take_along_dim(embeddings, position_index % embeddings.size(0), dim=0)
            embeddings = alpha * embeddings_x + (1 - alpha) * embeddings_y
        return embeddings

    def load_weights_from_pytorch_checkpoint(self, checkpoint, mapping=None):
        """根据mapping从checkpoint加载权重
        """
        file_state_dict = torch.load(checkpoint, map_location='cpu')
        mapping = mapping or self.variable_mapping()
        parameters_set = set([i[0] for i in self.named_parameters()])
        for layer_name in parameters_set:
            if layer_name in file_state_dict and layer_name not in mapping:
                mapping.update({layer_name: layer_name})
        state_dict_new = {}
        for new_key, old_key in mapping.items():
            if new_key not in self.state_dict():
                continue
            elif old_key in file_state_dict:
                state_dict_new[new_key] = self.load_variable(file_state_dict, old_key)
            elif old_key not in file_state_dict and not self.ignore_invalid_weights:
                None
            if new_key in parameters_set:
                parameters_set.remove(new_key)
        if not self.ignore_invalid_weights:
            for key in parameters_set:
                None
        del file_state_dict
        self.load_state_dict(state_dict_new, strict=False)

    def apply_embeddings(self, inputs):
        raise NotImplementedError

    def apply_main_layers(self, inputs):
        raise NotImplementedError

    def apply_final_layers(self, inputs):
        raise NotImplementedError

    def apply_on_layer_begin(self, l_i, inputs):
        """新增对layer block输入进行操作的函数
        """
        return inputs

    def apply_on_layer_end(self, l_i, inputs):
        """新增对layer block输出进行操作的函数
        """
        return inputs

    def compute_attention_bias(self, inputs=None):
        """定义每一层的Attention Bias
        """
        return self.attention_bias

    def compute_position_bias(self, inputs=None):
        """定义每一层的Position Bias（一般相对位置编码用）
        """
        return self.position_bias

    def set_outputs(self, outputs):
        """设置output和oututs属性
        """
        if not isinstance(outputs, list):
            outputs = [outputs]
        outputs = outputs[:]
        self.outputs = outputs
        if len(outputs) > 1:
            self.output = outputs
        else:
            self.output = outputs[0]


def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    """生成padding_ids, 从padding_idx+1开始。忽略填充符号
    """
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx


def get_kw(cls, kwargs):
    """保留排除cls的入参后的kwargs

    :param cls: 类
    :param kwargs: dict, 所有参数
    """
    kwargs_new = {}
    for k in kwargs:
        if k not in set(inspect.getargspec(cls)[0]):
            kwargs_new[k] = kwargs[k]
    return kwargs_new


class BERT(BERT_BASE):
    """构建BERT模型
    """

    def __init__(self, max_position, segment_vocab_size=2, with_pool=False, with_nsp=False, with_mlm=False, custom_position_ids=False, custom_attention_mask=False, shared_segment_embeddings=False, layer_norm_cond=None, layer_add_embs=None, is_dropout=False, token_pad_ids=0, **kwargs):
        super(BERT, self).__init__(**kwargs)
        self.max_position = max_position
        self.segment_vocab_size = segment_vocab_size
        self.with_pool = with_pool
        self.with_nsp = with_nsp
        self.with_mlm = with_mlm
        self.custom_position_ids = custom_position_ids
        self.custom_attention_mask = custom_attention_mask
        self.shared_segment_embeddings = shared_segment_embeddings
        self.is_dropout = is_dropout
        self.token_pad_ids = token_pad_ids
        if self.with_nsp and not self.with_pool:
            self.with_pool = True
        self.layer_norm_conds = layer_norm_cond
        self.layer_add_embs = layer_add_embs
        self.conditional_size = layer_norm_cond.weight.size(1) if layer_norm_cond is not None else None
        self.embeddings = BertEmbeddings(self.vocab_size, self.embedding_size, self.hidden_size, self.max_position, self.segment_vocab_size, self.shared_segment_embeddings, self.dropout_rate, self.conditional_size, **get_kw(BertEmbeddings, kwargs))
        kwargs['max_position'] = self.max_position
        layer = BertLayer(self.hidden_size, self.num_attention_heads, self.dropout_rate, self.attention_probs_dropout_prob, self.intermediate_size, self.hidden_act, is_dropout=self.is_dropout, conditional_size=self.conditional_size, **get_kw(BertLayer, kwargs))
        self.encoderLayer = nn.ModuleList([(copy.deepcopy(layer) if layer_id in self.keep_hidden_layers else Identity()) for layer_id in range(self.num_hidden_layers)])
        if self.with_pool:
            self.pooler = nn.Linear(self.hidden_size, self.hidden_size)
            self.pooler_activation = nn.Tanh() if self.with_pool is True else get_activation(self.with_pool)
            if self.with_nsp:
                self.nsp = nn.Linear(self.hidden_size, 2)
        else:
            self.pooler = None
            self.pooler_activation = None
        if self.with_mlm:
            self.mlmDense = nn.Linear(self.hidden_size, self.embedding_size)
            self.transform_act_fn = get_activation(self.hidden_act)
            self.mlmLayerNorm = LayerNorm(self.embedding_size, eps=1e-12, conditional_size=self.conditional_size)
            self.mlmDecoder = nn.Linear(self.embedding_size, self.vocab_size, bias=False)
            if kwargs.get('tie_emb_prj_weight') is True:
                self.mlmDecoder.weight = self.embeddings.word_embeddings.weight
            self.mlmBias = nn.Parameter(torch.zeros(self.vocab_size))
            self.mlmDecoder.bias = self.mlmBias

    def apply_embeddings(self, inputs):
        """BERT的embedding是token、position、segment三者embedding之和

        :param inputs: List[torch.Tensor], 默认顺序是[token_ids, segment_ids(若有), position_ids(若有), custom_attention_mask(若有), conditional_input(若有)]
        :return: List[torch.Tensor], [hidden_states, attention_mask, conditional_emb, ...]
        """
        assert isinstance(inputs, (tuple, list)), f'Inputs only support list,tuple format but passed {type(inputs)}'
        token_ids = inputs[0]
        index_ = 1
        if self.segment_vocab_size > 0:
            segment_ids = inputs[index_]
            index_ += 1
        else:
            segment_ids = None
        if self.custom_position_ids is True:
            position_ids = inputs[index_]
            index_ += 1
        elif self.custom_position_ids == 'start_at_padding':
            position_ids = create_position_ids_from_input_ids(token_ids, self.token_pad_ids)
        else:
            position_ids = None
        if self.custom_attention_mask:
            attention_mask = inputs[index_].long().unsqueeze(1).unsqueeze(2)
            index_ += 1
        elif not token_ids.requires_grad and token_ids.dtype in {torch.long, torch.int}:
            attention_mask = (token_ids != self.token_pad_ids).long().unsqueeze(1).unsqueeze(2)
            if self.token_pad_ids < 0:
                token_ids = token_ids * attention_mask[:, 0, 0, :]
        else:
            attention_mask = self.attention_mask_cache
        self.attention_mask_cache = attention_mask
        self.compute_attention_bias([token_ids, segment_ids])
        if self.attention_bias is not None:
            attention_mask = attention_mask * self.attention_bias
        try:
            attention_mask = attention_mask
        except StopIteration:
            attention_mask = attention_mask
        if self.layer_norm_conds is None:
            conditional_emb = None
        else:
            conditional_emb = self.layer_norm_conds(inputs[index_])
            index_ += 1
        if isinstance(self.layer_add_embs, nn.Module):
            additional_embs = [self.layer_add_embs(inputs[index_])]
            index_ += 1
        elif isinstance(self.layer_add_embs, (tuple, list)):
            additional_embs = []
            for layer in self.layer_add_embs:
                assert isinstance(layer, nn.Module), 'Layer_add_embs element should be nn.Module'
                additional_embs.append(layer(inputs[index_]))
                index_ += 1
        else:
            additional_embs = None
        hidden_states = self.embeddings(token_ids, segment_ids, position_ids, conditional_emb, additional_embs)
        return [hidden_states, attention_mask, conditional_emb] + inputs[index_:]

    def apply_main_layers(self, inputs):
        """BERT的主体是基于Self-Attention的模块；
        顺序:Att --> Add --> LN --> FFN --> Add --> LN
        
        :param inputs: List[torch.Tensor], 默认顺序为[hidden_states, attention_mask, conditional_emb]
        :return: List[torch.Tensor], 默认顺序为[encoded_layers, conditional_emb]
        """
        hidden_states, attention_mask, conditional_emb = inputs[:3]
        if len(inputs[3:]) >= 2:
            encoder_hidden_state, encoder_attention_mask = inputs[3], inputs[4]
        else:
            encoder_hidden_state, encoder_attention_mask = None, None
        encoded_layers = [hidden_states]
        layer_inputs = [hidden_states, attention_mask, conditional_emb, encoder_hidden_state, encoder_attention_mask]
        for l_i, layer_module in enumerate(self.encoderLayer):
            layer_inputs = self.apply_on_layer_begin(l_i, layer_inputs)
            hidden_states = grad_checkpoint(layer_module, *layer_inputs) if self.gradient_checkpoint else layer_module(*layer_inputs)
            layer_inputs[0] = hidden_states
            layer_inputs = self.apply_on_layer_end(l_i, layer_inputs)
            if self.output_all_encoded_layers:
                encoded_layers.append(hidden_states)
        if not self.output_all_encoded_layers:
            encoded_layers.append(hidden_states)
        return [encoded_layers, conditional_emb]

    def apply_final_layers(self, inputs):
        """根据剩余参数决定输出

        :param inputs: List[torch.Tensor], 默认顺序为[encoded_layers, conditional_emb]
        :return: List[torch.Tensor] or torch.Tensor, 模型输出，默认顺序为[last_hidden_state/all_encoded_layers, pooled_output(若有), mlm_scores(若有), nsp_scores(若有)]
        """
        encoded_layers, conditional_emb = inputs
        sequence_output = encoded_layers[-1]
        if not self.output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        if self.with_pool:
            pooled_output = self.pooler_activation(self.pooler(sequence_output[:, 0]))
        else:
            pooled_output = None
        if self.with_pool and self.with_nsp:
            nsp_scores = self.nsp(pooled_output)
        else:
            nsp_scores = None
        if self.with_mlm:
            mlm_hidden_state = self.mlmDense(sequence_output)
            mlm_hidden_state = self.transform_act_fn(mlm_hidden_state)
            mlm_hidden_state = self.mlmLayerNorm((mlm_hidden_state, conditional_emb))
            mlm_scores = self.mlmDecoder(mlm_hidden_state)
            mlm_activation = get_activation('linear' if self.with_mlm is True else self.with_mlm)
            mlm_scores = mlm_activation(mlm_scores)
        else:
            mlm_scores = None
        outputs = [value for value in [encoded_layers, pooled_output, mlm_scores, nsp_scores] if value is not None]
        return outputs if len(outputs) > 1 else outputs[0]

    def load_variable(self, state_dict, name, prefix='bert'):
        """加载单个变量的函数
        """
        variable = state_dict[name]
        if name in {f'{prefix}.embeddings.word_embeddings.weight', 'cls.predictions.bias', 'cls.predictions.decoder.weight', 'cls.predictions.decoder.bias'}:
            return self.load_embeddings(variable)
        elif name == f'{prefix}.embeddings.position_embeddings.weight':
            return self.load_pos_embeddings(variable)
        elif name == 'cls.seq_relationship.weight':
            return variable.T
        else:
            return variable

    def variable_mapping(self, prefix='bert'):
        mapping = {'embeddings.word_embeddings.weight': f'{prefix}.embeddings.word_embeddings.weight', 'embeddings.position_embeddings.weight': f'{prefix}.embeddings.position_embeddings.weight', 'embeddings.segment_embeddings.weight': f'{prefix}.embeddings.token_type_embeddings.weight', 'embeddings.layerNorm.weight': f'{prefix}.embeddings.LayerNorm.weight', 'embeddings.layerNorm.bias': f'{prefix}.embeddings.LayerNorm.bias', 'pooler.weight': f'{prefix}.pooler.dense.weight', 'pooler.bias': f'{prefix}.pooler.dense.bias', 'nsp.weight': 'cls.seq_relationship.weight', 'nsp.bias': 'cls.seq_relationship.bias', 'mlmDense.weight': 'cls.predictions.transform.dense.weight', 'mlmDense.bias': 'cls.predictions.transform.dense.bias', 'mlmLayerNorm.weight': 'cls.predictions.transform.LayerNorm.weight', 'mlmLayerNorm.bias': 'cls.predictions.transform.LayerNorm.bias', 'mlmBias': 'cls.predictions.bias', 'mlmDecoder.weight': 'cls.predictions.decoder.weight', 'mlmDecoder.bias': 'cls.predictions.decoder.bias'}
        for i in range(self.num_hidden_layers):
            prefix_i = f'{prefix}.encoder.layer.%d.' % i
            mapping.update({f'encoderLayer.{i}.multiHeadAttention.q.weight': prefix_i + 'attention.self.query.weight', f'encoderLayer.{i}.multiHeadAttention.q.bias': prefix_i + 'attention.self.query.bias', f'encoderLayer.{i}.multiHeadAttention.k.weight': prefix_i + 'attention.self.key.weight', f'encoderLayer.{i}.multiHeadAttention.k.bias': prefix_i + 'attention.self.key.bias', f'encoderLayer.{i}.multiHeadAttention.v.weight': prefix_i + 'attention.self.value.weight', f'encoderLayer.{i}.multiHeadAttention.v.bias': prefix_i + 'attention.self.value.bias', f'encoderLayer.{i}.multiHeadAttention.o.weight': prefix_i + 'attention.output.dense.weight', f'encoderLayer.{i}.multiHeadAttention.o.bias': prefix_i + 'attention.output.dense.bias', f'encoderLayer.{i}.layerNorm1.weight': prefix_i + 'attention.output.LayerNorm.weight', f'encoderLayer.{i}.layerNorm1.bias': prefix_i + 'attention.output.LayerNorm.bias', f'encoderLayer.{i}.feedForward.intermediateDense.weight': prefix_i + 'intermediate.dense.weight', f'encoderLayer.{i}.feedForward.intermediateDense.bias': prefix_i + 'intermediate.dense.bias', f'encoderLayer.{i}.feedForward.outputDense.weight': prefix_i + 'output.dense.weight', f'encoderLayer.{i}.feedForward.outputDense.bias': prefix_i + 'output.dense.bias', f'encoderLayer.{i}.layerNorm2.weight': prefix_i + 'output.LayerNorm.weight', f'encoderLayer.{i}.layerNorm2.bias': prefix_i + 'output.LayerNorm.bias'})
        if self.embedding_size != self.hidden_size:
            mapping.update({'embeddings.embedding_hidden_mapping_in.weight': f'{prefix}.encoder.embedding_hidden_mapping_in.weight', 'embeddings.embedding_hidden_mapping_in.bias': f'{prefix}.encoder.embedding_hidden_mapping_in.bias'})
        return mapping


class ALBERT(BERT):

    def __init__(self, *args, **kwargs):
        super(ALBERT, self).__init__(*args, **kwargs)
        self.encoderLayer = nn.ModuleList([self.encoderLayer[0]])

    def apply_main_layers(self, inputs):
        """BERT的主体是基于Self-Attention的模块（和BERT区别是始终使用self.encoderLayer[0]）；
        顺序:Att --> Add --> LN --> FFN --> Add --> LN
        """
        hidden_states, attention_mask, conditional_emb = inputs[:3]
        if len(inputs[3:]) >= 2:
            encoder_hidden_state, encoder_attention_mask = inputs[3], inputs[4]
        else:
            encoder_hidden_state, encoder_attention_mask = None, None
        encoded_layers = [hidden_states]
        layer_inputs = [hidden_states, attention_mask, conditional_emb, encoder_hidden_state, encoder_attention_mask]
        for l_i in range(self.num_hidden_layers):
            layer_inputs = self.apply_on_layer_begin(l_i, layer_inputs)
            hidden_states = self.encoderLayer[0](*layer_inputs)
            layer_inputs[0] = hidden_states
            layer_inputs = self.apply_on_layer_end(l_i, layer_inputs)
            if self.output_all_encoded_layers:
                encoded_layers.append(hidden_states)
        if not self.output_all_encoded_layers:
            encoded_layers.append(hidden_states)
        return [encoded_layers, conditional_emb]

    def variable_mapping(self, prefix='albert'):
        mapping = {'embeddings.word_embeddings.weight': f'{prefix}.embeddings.word_embeddings.weight', 'embeddings.position_embeddings.weight': f'{prefix}.embeddings.position_embeddings.weight', 'embeddings.segment_embeddings.weight': f'{prefix}.embeddings.token_type_embeddings.weight', 'embeddings.layerNorm.weight': f'{prefix}.embeddings.LayerNorm.weight', 'embeddings.layerNorm.bias': f'{prefix}.embeddings.LayerNorm.bias', 'embeddings.embedding_hidden_mapping_in.weight': f'{prefix}.encoder.embedding_hidden_mapping_in.weight', 'embeddings.embedding_hidden_mapping_in.bias': f'{prefix}.encoder.embedding_hidden_mapping_in.bias', 'pooler.weight': f'{prefix}.pooler.weight', 'pooler.bias': f'{prefix}.pooler.bias', 'nsp.weight': 'sop_classifier.classifier.weight', 'nsp.bias': 'sop_classifier.classifier.bias', 'mlmDense.weight': 'predictions.dense.weight', 'mlmDense.bias': 'predictions.dense.bias', 'mlmLayerNorm.weight': 'predictions.LayerNorm.weight', 'mlmLayerNorm.bias': 'predictions.LayerNorm.bias', 'mlmBias': 'predictions.bias', 'mlmDecoder.weight': 'predictions.decoder.weight', 'mlmDecoder.bias': 'predictions.decoder.bias'}
        i = 0
        prefix_i = f'{prefix}.encoder.albert_layer_groups.{i}.albert_layers.{i}.'
        mapping.update({f'encoderLayer.{i}.multiHeadAttention.q.weight': prefix_i + 'attention.query.weight', f'encoderLayer.{i}.multiHeadAttention.q.bias': prefix_i + 'attention.query.bias', f'encoderLayer.{i}.multiHeadAttention.k.weight': prefix_i + 'attention.key.weight', f'encoderLayer.{i}.multiHeadAttention.k.bias': prefix_i + 'attention.key.bias', f'encoderLayer.{i}.multiHeadAttention.v.weight': prefix_i + 'attention.value.weight', f'encoderLayer.{i}.multiHeadAttention.v.bias': prefix_i + 'attention.value.bias', f'encoderLayer.{i}.multiHeadAttention.o.weight': prefix_i + 'attention.dense.weight', f'encoderLayer.{i}.multiHeadAttention.o.bias': prefix_i + 'attention.dense.bias', f'encoderLayer.{i}.layerNorm1.weight': prefix_i + 'attention.LayerNorm.weight', f'encoderLayer.{i}.layerNorm1.bias': prefix_i + 'attention.LayerNorm.bias', f'encoderLayer.{i}.feedForward.intermediateDense.weight': prefix_i + 'ffn.weight', f'encoderLayer.{i}.feedForward.intermediateDense.bias': prefix_i + 'ffn.bias', f'encoderLayer.{i}.feedForward.outputDense.weight': prefix_i + 'ffn_output.weight', f'encoderLayer.{i}.feedForward.outputDense.bias': prefix_i + 'ffn_output.bias', f'encoderLayer.{i}.layerNorm2.weight': prefix_i + 'full_layer_layer_norm.weight', f'encoderLayer.{i}.layerNorm2.bias': prefix_i + 'full_layer_layer_norm.bias'})
        return mapping

    def load_variable(self, state_dict, name):
        variable = state_dict[name]
        if name in {'albert.embeddings.word_embeddings.weight', 'predictions.bias', 'predictions.decoder.weight', 'predictions.decoder.bias'}:
            return self.load_embeddings(variable)
        elif name == 'albert.embeddings.position_embeddings.weight':
            return self.load_pos_embeddings(variable)
        elif name == 'sop_classifier.classifier.weight':
            return variable.T
        else:
            return variable


class ALBERT_Unshared(ALBERT):

    def __init__(self, *args, **kwargs):
        super(ALBERT_Unshared).__init__(*args, **kwargs)
        self.encoderLayer = nn.ModuleList([copy.deepcopy(self.encoderLayer[0]) for _ in range(self.num_hidden_layers)])

    def apply_main_layers(self, inputs):
        """BERT的主体是基于Self-Attention的模块（和ALBERT区别是所有层权重独立）；
        顺序:Att --> Add --> LN --> FFN --> Add --> LN
        """
        hidden_states, attention_mask, conditional_emb = inputs
        if len(inputs[3:]) >= 2:
            encoder_hidden_state, encoder_attention_mask = inputs[3], inputs[4]
        else:
            encoder_hidden_state, encoder_attention_mask = None, None
        encoded_layers = [hidden_states]
        layer_inputs = [hidden_states, attention_mask, conditional_emb, encoder_hidden_state, encoder_attention_mask]
        for i in range(self.num_hidden_layers):
            layer_inputs = self.apply_on_layer_begin(i, layer_inputs)
            hidden_states = self.encoderLayer[i](*layer_inputs)
            layer_inputs[0] = hidden_states
            layer_inputs = self.apply_on_layer_end(i, layer_inputs)
            if self.output_all_encoded_layers:
                encoded_layers.append(hidden_states)
        if not self.output_all_encoded_layers:
            encoded_layers.append(hidden_states)
        return [encoded_layers, conditional_emb]


class NEZHA(BERT):
    """华为推出的NAZHA模型；
    链接：https://arxiv.org/abs/1909.00204
    """

    def __init__(self, *args, **kwargs):
        kwargs.update({'p_bias': 'typical_relative', 'max_relative_position': kwargs.get('max_relative_position', 64)})
        super(NEZHA, self).__init__(*args, **kwargs)


class RoFormer(BERT):
    """旋转式位置编码的BERT模型；
    链接：https://kexue.fm/archives/8265
    """

    def __init__(self, *args, **kwargs):
        kwargs.update({'p_bias': 'rotary'})
        super(RoFormer, self).__init__(*args, **kwargs)

    def load_variable(self, state_dict, name, prefix='roformer'):
        return super().load_variable(state_dict, name, prefix)

    def variable_mapping(self, prefix='roformer'):
        mapping = super().variable_mapping(prefix)
        del mapping['embeddings.position_embeddings.weight']
        return mapping


def delete_arguments(*arguments):
    """装饰器，为类方法删除参数（主要用于类的__init__方法）
    """

    def actual_decorator(func):

        def new_func(self, *args, **kwargs):
            for k in arguments:
                if k in kwargs:
                    raise TypeError("%s got an unexpected keyword argument '%s'" % (self.__class__.__name__, k))
            return func(self, *args, **kwargs)
        return new_func
    return actual_decorator


class RoFormerV2(RoFormer):
    """RoFormerV2；
    改动：去掉bias，简化Norm，优化初始化等。目前初始化暂时还用的bert的初始化，finetune不受影响
    """

    @delete_arguments('with_pool', 'with_nsp')
    def __init__(self, *args, **kwargs):
        kwargs.update({'p_bias': 'rotary', 'weight': False, 'bias': False, 'norm_mode': 'rmsnorm'})
        super(RoFormerV2, self).__init__(*args, **kwargs)
        if self.with_mlm:
            del self.mlmLayerNorm
            del self.mlmBias
            del self.mlmDense
            self.mlmDecoder.register_parameter('bias', None)

    def variable_mapping(self, prefix='roformer'):
        mapping = super().variable_mapping(prefix)
        mapping_new = {}
        for k, v in mapping.items():
            if not re.search('bias|layernorm', k.lower()) and not re.search('bias|layernorm', v.lower()):
                mapping_new[k] = v
        return mapping_new

    def apply_final_layers(self, inputs):
        """根据剩余参数决定输出
        """
        encoded_layers, conditional_emb = inputs
        sequence_output = encoded_layers[-1]
        if not self.output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        if self.with_mlm:
            mlm_scores = self.mlmDecoder(sequence_output)
        else:
            mlm_scores = None
        outputs = [value for value in [encoded_layers, mlm_scores] if value is not None]
        return outputs if len(outputs) > 1 else outputs[0]


class GAU_alpha(RoFormerV2):

    def __init__(self, *args, **kwargs):
        kwargs.update({'p_bias': 'rotary', 'weight': False, 'bias': False, 'norm_mode': 'rmsnorm', 'normalization': 'softmax_plus'})
        super().__init__(*args, **kwargs)
        layer = self.GAU_Layer(**kwargs)
        self.encoderLayer = nn.ModuleList([(copy.deepcopy(layer) if layer_id in self.keep_hidden_layers else Identity()) for layer_id in range(self.num_hidden_layers)])

    def load_variable(self, state_dict, name, prefix=''):
        variable = state_dict[name]
        return self.load_embeddings(variable) if name in {'embeddings.word_embeddings.weight', 'mlmDecoder.weight'} else variable

    def variable_mapping(self, prefix=''):
        """在convert脚本里已经把key转成bert4torch可用的
        """
        return {k: k for k, _ in self.named_parameters()}


    class GAU_Layer(nn.Module):

        def __init__(self, *args, **kwargs):
            super().__init__()
            self.gau = GatedAttentionUnit(**kwargs)
            self.dropout1 = nn.Dropout(kwargs.get('dropout_rate'))
            self.layerNorm1 = LayerNorm(**kwargs)

        def forward(self, hidden_states, attention_mask, conditional_emb=None, encoder_hidden_states=None, encoder_attention_mask=None):
            gau_hidden_states = self.gau(hidden_states, attention_mask)
            hidden_states = hidden_states + self.dropout1(gau_hidden_states)
            hidden_states = self.layerNorm1((hidden_states, conditional_emb))
            return hidden_states


def insert_arguments(**arguments):
    """装饰器，为类方法增加参数（主要用于类的__init__方法）
    """

    def actual_decorator(func):

        def new_func(self, *args, **kwargs):
            for k, v in arguments.items():
                if k in kwargs:
                    v = kwargs.pop(k)
                setattr(self, k, v)
            return func(self, *args, **kwargs)
        return new_func
    return actual_decorator


class ELECTRA(BERT):
    """Google推出的ELECTRA模型；
    链接：https://arxiv.org/abs/2003.10555
    """

    @insert_arguments(with_discriminator=False)
    @delete_arguments('with_pool', 'with_mlm', 'with_nsp')
    def __init__(self, max_position, **kwargs):
        super(ELECTRA, self).__init__(max_position, **kwargs)
        if self.with_discriminator:
            self.dense = nn.Linear(self.hidden_size, self.hidden_size)
            self.dense_act = get_activation(self.hidden_act)
            self.dense_prediction = nn.Linear(self.hidden_size, 1)
            self.dense_prediction_act = get_activation('sigmoid') if self.with_discriminator is True else get_activation(self.with_discriminator)

    def apply_final_layers(self, inputs):
        hidden_states = super().apply_final_layers(inputs)
        if self.with_discriminator:
            logit = self.dense_act(self.dense(hidden_states))
            return [hidden_states, self.dense_prediction_act(self.dense_prediction(logit))]
        else:
            return hidden_states

    def load_variable(self, state_dict, name):
        return super().load_variable(state_dict, name, prefix='electra')

    def variable_mapping(self):
        mapping = super(ELECTRA, self).variable_mapping(prefix='electra')
        mapping.update({'dense.weight': 'discriminator_predictions.dense.weight', 'dense.bias': 'discriminator_predictions.dense.bias', 'dense_prediction.weight': 'discriminator_predictions.dense_prediction.weight', 'dense_prediction.bias': 'discriminator_predictions.dense_prediction.bias'})
        for del_key in ['pooler.weight', 'pooler.bias', 'nsp.weight', 'nsp.bias', 'mlmDense.weight', 'mlmDense.bias', 'mlmLayerNorm.weight', 'mlmLayerNorm.bias', 'mlmBias', 'mlmDecoder.weight', 'mlmDecoder.bias']:
            del mapping[del_key]
        return mapping


class ERNIE(BERT):
    """百度文心 https://github.com/PaddlePaddle/ERNIE
    """

    def __init__(self, *args, **kwargs):
        super(ERNIE, self).__init__(*args, **kwargs)

    def variable_mapping(self):
        mapping = super(ERNIE, self).variable_mapping(prefix='ernie')
        mapping.update({'mlmDecoder.weight': 'ernie.embeddings.word_embeddings.weight', 'mlmDecoder.bias': 'cls.predictions.bias'})
        for k, v in mapping.items():
            if 'LayerNorm.weight' in v or 'LayerNorm.bias' in v:
                v1 = v.replace('.weight', '.gamma').replace('.bias', '.beta')
                mapping[k] = v1
        for del_key in ['nsp.weight', 'nsp.bias']:
            del mapping[del_key]
        return mapping

    def load_variable(self, state_dict, name, prefix='ernie'):
        return super().load_variable(state_dict, name, prefix=prefix)


class DebertaV2(BERT):
    """DeBERTaV2: https://arxiv.org/abs/2006.03654, https://github.com/microsoft/DeBERTa；
       这里使用的是IEDEA的中文模型：https://huggingface.co/IDEA-CCNL/Erlangshen-DeBERTa-v2-320M-Chinese；
       和transformers包中的区别：
       1）原始使用的StableDropout替换成了nn.Dropout；
       2）计算attention_score时候用的XSoftmax替换成了F.softmax；
       3）未实现（认为对结果无影响）：Embedding阶段用attention_mask对embedding的padding部分置0；
       4）未实现（认为对结果无影响）：计算attention_score前的attention_mask从[btz, 1, 1, k_len]转为了[btz, 1, q_len, k_len]。
    """

    @delete_arguments('with_pool', 'with_nsp')
    def __init__(self, *args, **kwargs):
        kwargs.update({'p_bias': 'deberta_v2'})
        super(DebertaV2, self).__init__(*args, **kwargs)
        self.relative_attention = kwargs.get('relative_attention', True)
        self.conv = ConvLayer(**kwargs) if kwargs.get('conv_kernel_size', 0) > 0 else None
        for i in range(1, self.num_hidden_layers):
            self.encoderLayer[i].multiHeadAttention.relative_positions_encoding.weight = self.encoderLayer[0].multiHeadAttention.relative_positions_encoding.weight
            self.encoderLayer[i].multiHeadAttention.layernorm.weight = self.encoderLayer[0].multiHeadAttention.layernorm.weight
            self.encoderLayer[i].multiHeadAttention.layernorm.bias = self.encoderLayer[0].multiHeadAttention.layernorm.bias

    def apply_main_layers(self, inputs):
        """DebertaV2：主要区别是第0层后，会通过卷积层
        """
        hidden_states, attention_mask, conditional_emb = inputs[:3]
        if len(inputs[3:]) >= 2:
            encoder_hidden_state, encoder_attention_mask = inputs[3], inputs[4]
        else:
            encoder_hidden_state, encoder_attention_mask = None, None
        encoded_layers = [hidden_states]
        layer_inputs = [hidden_states, attention_mask, conditional_emb, encoder_hidden_state, encoder_attention_mask]
        for l_i, layer_module in enumerate(self.encoderLayer):
            layer_inputs = self.apply_on_layer_begin(l_i, layer_inputs)
            hidden_states = grad_checkpoint(layer_module, *layer_inputs) if self.gradient_checkpoint else layer_module(*layer_inputs)
            if l_i == 0 and self.conv is not None:
                hidden_states = self.conv(encoded_layers[0], hidden_states, attention_mask.squeeze(1).squeeze(1))
            layer_inputs[0] = hidden_states
            layer_inputs = self.apply_on_layer_end(l_i, layer_inputs)
            if self.output_all_encoded_layers:
                encoded_layers.append(hidden_states)
        if not self.output_all_encoded_layers:
            encoded_layers.append(hidden_states)
        return [encoded_layers, conditional_emb]

    def variable_mapping(self):
        mapping = super(DebertaV2, self).variable_mapping(prefix='deberta')
        mapping.update({'mlmDecoder.weight': 'deberta.embeddings.word_embeddings.weight', 'mlmDecoder.bias': 'cls.predictions.bias', 'encoderLayer.0.multiHeadAttention.relative_positions_encoding.weight': 'deberta.encoder.rel_embeddings.weight', 'encoderLayer.0.multiHeadAttention.layernorm.weight': 'deberta.encoder.LayerNorm.weight', 'encoderLayer.0.multiHeadAttention.layernorm.bias': 'deberta.encoder.LayerNorm.bias', 'conv.conv.weight': 'deberta.encoder.conv.conv.weight', 'conv.conv.bias': 'deberta.encoder.conv.conv.bias', 'conv.LayerNorm.weight': 'deberta.encoder.conv.LayerNorm.weight', 'conv.LayerNorm.bias': 'deberta.encoder.conv.LayerNorm.bias'})
        for del_key in ['nsp.weight', 'nsp.bias', 'embeddings.position_embeddings.weight', 'embeddings.segment_embeddings.weight']:
            del mapping[del_key]
        return mapping

    def load_variable(self, state_dict, name, prefix='deberta'):
        return super().load_variable(state_dict, name, prefix=prefix)


class UIE(BERT):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden_size = self.hidden_size
        self.linear_start = nn.Linear(hidden_size, 1)
        self.linear_end = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        if kwargs.get('use_task_id') and kwargs.get('use_task_id'):
            task_type_embeddings = nn.Embedding(kwargs.get('task_type_vocab_size'), self.hidden_size)
            self.embeddings.task_type_embeddings = task_type_embeddings

            def hook(module, input, output):
                return output + task_type_embeddings(torch.zeros(input[0].size(), dtype=torch.int64, device=input[0].device))
            self.embeddings.word_embeddings.register_forward_hook(hook)

    def forward(self, token_ids, token_type_ids):
        outputs = super().forward([token_ids, token_type_ids])
        sequence_output = outputs[0]
        start_logits = self.linear_start(sequence_output)
        start_logits = torch.squeeze(start_logits, -1)
        start_prob = self.sigmoid(start_logits)
        end_logits = self.linear_end(sequence_output)
        end_logits = torch.squeeze(end_logits, -1)
        end_prob = self.sigmoid(end_logits)
        return start_prob, end_prob

    @torch.no_grad()
    def predict(self, token_ids, token_type_ids):
        self.eval()
        start_prob, end_prob = self.forward(token_ids, token_type_ids)
        return start_prob, end_prob


class Encoder(BERT):

    def __init__(self, *args, **kwargs):
        kwargs['vocab_size'] = kwargs.get('src_vocab_size', kwargs['vocab_size'])
        super().__init__(*args, **kwargs)
        self.encoder_attention_mask = None

    def forward(self, inputs):
        """因为encoder需要返回encoder_attention_mask，因此这里从新定义一下，多返回一个参数
        """
        outputs = self.apply_embeddings(inputs)
        encoder_attention_mask = [outputs[1]]
        outputs = self.apply_main_layers(outputs)
        outputs = self.apply_final_layers(outputs)
        return ([outputs] if isinstance(outputs, torch.Tensor) else outputs) + encoder_attention_mask


class LM_Mask(object):
    """定义下三角Attention Mask（语言模型用）
    """

    def compute_attention_bias(self, inputs=None):
        """通过idxs序列的比较来得到对应的mask
        """
        seq_len = inputs[0].shape[1]
        attention_bias = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.long, device=inputs[0].device), diagonal=0)
        self.attention_bias = attention_bias.unsqueeze(0).unsqueeze(1)
        return self.attention_bias


class Decoder(LM_Mask, BERT):

    @delete_arguments('with_pool', 'with_mlm', 'with_nsp')
    def __init__(self, *args, with_lm=True, tie_emb_prj_weight=False, logit_scale=True, **kwargs):
        kwargs['vocab_size'] = kwargs.get('tgt_vocab_size', kwargs['vocab_size'])
        kwargs['is_decoder'] = True
        super().__init__(*args, **kwargs)
        self.decoderLayer = self.encoderLayer
        del self.encoderLayer
        self.with_lm = with_lm
        if self.with_lm:
            self.final_dense = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
            if tie_emb_prj_weight:
                self.final_dense.weight = self.embeddings.word_embeddings.weight
            if logit_scale:
                self.x_logit_scale = self.hidden_size ** -0.5
            else:
                self.x_logit_scale = 1.0

    def apply_main_layers(self, inputs):
        """Dencoder主体是基于Self-Attention、Cross-Attention的模块；
        顺序：Att1 --> Add --> LN --> Att2 --> Add -->  LN --> FFN --> Add --> LN
        """
        hidden_states, attention_mask, conditional_emb, encoder_hidden_state, encoder_attention_mask = inputs[:5]
        decoded_layers = [hidden_states]
        layer_inputs = [hidden_states, attention_mask, conditional_emb, encoder_hidden_state, encoder_attention_mask]
        for i, layer_module in enumerate(self.decoderLayer):
            layer_inputs = self.apply_on_layer_begin(i, layer_inputs)
            hidden_states = grad_checkpoint(layer_module, *layer_inputs) if self.gradient_checkpoint else layer_module(*layer_inputs)
            layer_inputs[0] = hidden_states
            layer_inputs = self.apply_on_layer_end(i, layer_inputs)
            if self.output_all_encoded_layers:
                decoded_layers.append(hidden_states)
        if not self.output_all_encoded_layers:
            decoded_layers.append(hidden_states)
        return [decoded_layers, conditional_emb]

    def apply_final_layers(self, inputs):
        outputs = []
        hidden_states = super().apply_final_layers(inputs)
        outputs.append(hidden_states)
        if self.with_lm:
            logits = self.final_dense(hidden_states) * self.x_logit_scale
            activation = get_activation('linear' if self.with_lm is True else self.with_lm)
            logits = activation(logits)
            outputs.append(logits)
        return outputs

    def variable_mapping(self, prefix='bert'):
        raw_mapping = super().variable_mapping(prefix)
        mapping = {}
        for k, v in raw_mapping.items():
            mapping[k.replace('encoderLayer', 'decoderLayer')] = v
        return mapping


class Transformer(BERT_BASE):
    """encoder-decoder结构
    """

    @delete_arguments('with_pool', 'with_mlm', 'with_nsp')
    def __init__(self, *args, tie_emb_src_tgt_weight=False, **kwargs):
        super(Transformer, self).__init__(*args, **kwargs)
        self.encoder = Encoder(*args, **kwargs)
        self.encoder.build(**kwargs)
        self.decoder = Decoder(*args, **kwargs)
        self.decoder.build(**kwargs)
        if tie_emb_src_tgt_weight:
            assert self.encoder.vocab_size == self.decoder.vocab_size, 'To share word embedding, the vocab size of src/tgt shall be the same.'
            self.encoder.embeddings.word_embeddings.weight = self.decoder.embeddings.word_embeddings.weight

    def forward(self, inputs):
        encoder_input, decoder_input = inputs[:2]
        encoder_hidden_state, encoder_attention_mask = self.encoder(encoder_input)
        decoder_outputs = self.decoder(decoder_input + [encoder_hidden_state, encoder_attention_mask])
        return [encoder_hidden_state] + decoder_outputs


class BART(Transformer):
    """encoder-decoder结构
    """

    def __init__(self, *args, tie_emb_src_tgt_weight=True, **kwargs):
        kwargs['logit_scale'] = kwargs.get('logit_scale', False)
        kwargs['tie_emb_prj_weight'] = kwargs.get('tie_emb_prj_weight', True)
        super(BART, self).__init__(*args, tie_emb_src_tgt_weight=tie_emb_src_tgt_weight, **kwargs)
        self.tie_emb_src_tgt_weight = tie_emb_src_tgt_weight

    def load_variable(self, state_dict, name, prefix=''):
        variable = state_dict[name]
        if name in {'shared.weight', 'encoder.embed_tokens.weight', 'decoder.embed_tokens.weight'}:
            return self.load_embeddings(variable)
        elif name in {'encoder.embed_positions.weight', 'decoder.embed_positions.weight'}:
            return self.load_pos_embeddings(variable)
        else:
            return variable

    def variable_mapping(self, prefix=''):
        mapping = {'encoder.embeddings.word_embeddings.weight': 'shared.weight' if self.tie_emb_src_tgt_weight else 'encoder.embed_tokens.weight', 'encoder.embeddings.position_embeddings.weight': 'encoder.embed_positions.weight', 'encoder.embeddings.layerNorm.weight': 'encoder.layernorm_embedding.weight', 'encoder.embeddings.layerNorm.bias': 'encoder.layernorm_embedding.bias', 'decoder.embeddings.word_embeddings.weight': 'shared.weight' if self.tie_emb_src_tgt_weight else 'decoder.embed_tokens.weight', 'decoder.embeddings.position_embeddings.weight': 'decoder.embed_positions.weight', 'decoder.embeddings.layerNorm.weight': 'decoder.layernorm_embedding.weight', 'decoder.embeddings.layerNorm.bias': 'decoder.layernorm_embedding.bias'}
        for i in range(self.num_hidden_layers):
            mapping.update({f'encoder.encoderLayer.{i}.multiHeadAttention.q.weight': f'encoder.layers.{i}.self_attn.q_proj.weight', f'encoder.encoderLayer.{i}.multiHeadAttention.q.bias': f'encoder.layers.{i}.self_attn.q_proj.bias', f'encoder.encoderLayer.{i}.multiHeadAttention.k.weight': f'encoder.layers.{i}.self_attn.k_proj.weight', f'encoder.encoderLayer.{i}.multiHeadAttention.k.bias': f'encoder.layers.{i}.self_attn.k_proj.bias', f'encoder.encoderLayer.{i}.multiHeadAttention.v.weight': f'encoder.layers.{i}.self_attn.v_proj.weight', f'encoder.encoderLayer.{i}.multiHeadAttention.v.bias': f'encoder.layers.{i}.self_attn.v_proj.bias', f'encoder.encoderLayer.{i}.multiHeadAttention.o.weight': f'encoder.layers.{i}.self_attn.out_proj.weight', f'encoder.encoderLayer.{i}.multiHeadAttention.o.bias': f'encoder.layers.{i}.self_attn.out_proj.bias', f'encoder.encoderLayer.{i}.layerNorm1.weight': f'encoder.layers.{i}.self_attn_layer_norm.weight', f'encoder.encoderLayer.{i}.layerNorm1.bias': f'encoder.layers.{i}.self_attn_layer_norm.bias', f'encoder.encoderLayer.{i}.feedForward.intermediateDense.weight': f'encoder.layers.{i}.fc1.weight', f'encoder.encoderLayer.{i}.feedForward.intermediateDense.bias': f'encoder.layers.{i}.fc1.bias', f'encoder.encoderLayer.{i}.feedForward.outputDense.weight': f'encoder.layers.{i}.fc2.weight', f'encoder.encoderLayer.{i}.feedForward.outputDense.bias': f'encoder.layers.{i}.fc2.bias', f'encoder.encoderLayer.{i}.layerNorm2.weight': f'encoder.layers.{i}.final_layer_norm.weight', f'encoder.encoderLayer.{i}.layerNorm2.bias': f'encoder.layers.{i}.final_layer_norm.bias', f'decoder.decoderLayer.{i}.multiHeadAttention.q.weight': f'decoder.layers.{i}.self_attn.q_proj.weight', f'decoder.decoderLayer.{i}.multiHeadAttention.q.bias': f'decoder.layers.{i}.self_attn.q_proj.bias', f'decoder.decoderLayer.{i}.multiHeadAttention.k.weight': f'decoder.layers.{i}.self_attn.k_proj.weight', f'decoder.decoderLayer.{i}.multiHeadAttention.k.bias': f'decoder.layers.{i}.self_attn.k_proj.bias', f'decoder.decoderLayer.{i}.multiHeadAttention.v.weight': f'decoder.layers.{i}.self_attn.v_proj.weight', f'decoder.decoderLayer.{i}.multiHeadAttention.v.bias': f'decoder.layers.{i}.self_attn.v_proj.bias', f'decoder.decoderLayer.{i}.multiHeadAttention.o.weight': f'decoder.layers.{i}.self_attn.out_proj.weight', f'decoder.decoderLayer.{i}.multiHeadAttention.o.bias': f'decoder.layers.{i}.self_attn.out_proj.bias', f'decoder.decoderLayer.{i}.layerNorm1.weight': f'decoder.layers.{i}.self_attn_layer_norm.weight', f'decoder.decoderLayer.{i}.layerNorm1.bias': f'decoder.layers.{i}.self_attn_layer_norm.bias', f'decoder.decoderLayer.{i}.crossAttention.q.weight': f'decoder.layers.{i}.encoder_attn.q_proj.weight', f'decoder.decoderLayer.{i}.crossAttention.q.bias': f'decoder.layers.{i}.encoder_attn.q_proj.bias', f'decoder.decoderLayer.{i}.crossAttention.k.weight': f'decoder.layers.{i}.encoder_attn.k_proj.weight', f'decoder.decoderLayer.{i}.crossAttention.k.bias': f'decoder.layers.{i}.encoder_attn.k_proj.bias', f'decoder.decoderLayer.{i}.crossAttention.v.weight': f'decoder.layers.{i}.encoder_attn.v_proj.weight', f'decoder.decoderLayer.{i}.crossAttention.v.bias': f'decoder.layers.{i}.encoder_attn.v_proj.bias', f'decoder.decoderLayer.{i}.crossAttention.o.weight': f'decoder.layers.{i}.encoder_attn.out_proj.weight', f'decoder.decoderLayer.{i}.crossAttention.o.bias': f'decoder.layers.{i}.encoder_attn.out_proj.bias', f'decoder.decoderLayer.{i}.layerNorm3.weight': f'decoder.layers.{i}.encoder_attn_layer_norm.weight', f'decoder.decoderLayer.{i}.layerNorm3.bias': f'decoder.layers.{i}.encoder_attn_layer_norm.bias', f'decoder.decoderLayer.{i}.feedForward.intermediateDense.weight': f'decoder.layers.{i}.fc1.weight', f'decoder.decoderLayer.{i}.feedForward.intermediateDense.bias': f'decoder.layers.{i}.fc1.bias', f'decoder.decoderLayer.{i}.feedForward.outputDense.weight': f'decoder.layers.{i}.fc2.weight', f'decoder.decoderLayer.{i}.feedForward.outputDense.bias': f'decoder.layers.{i}.fc2.bias', f'decoder.decoderLayer.{i}.layerNorm2.weight': f'decoder.layers.{i}.final_layer_norm.weight', f'decoder.decoderLayer.{i}.layerNorm2.bias': f'decoder.layers.{i}.final_layer_norm.bias'})
        return mapping


class T5_Encoder(Encoder):

    @insert_arguments(version='t5.1.0')
    def __init__(self, *args, **kwargs):
        kwargs.update({'p_bias': 't5_relative', 'relative_attention_num_buckets': kwargs.get('relative_attention_num_buckets'), 'version': self.version, 'bias': False, 'norm_mode': 'rmsnorm'})
        super().__init__(*args, **kwargs)
        del self.embeddings.layerNorm
        layer = T5Layer(self.hidden_size, self.num_attention_heads, self.dropout_rate, self.attention_probs_dropout_prob, self.intermediate_size, self.hidden_act, is_dropout=self.is_dropout, conditional_size=self.conditional_size, **get_kw(BertLayer, kwargs))
        self.encoderLayer = nn.ModuleList([copy.deepcopy(layer) for _ in range(self.num_hidden_layers)])
        for i in range(1, self.num_hidden_layers):
            self.encoderLayer[i].multiHeadAttention.relative_positions_encoding.weight = self.encoderLayer[0].multiHeadAttention.relative_positions_encoding.weight
        self.final_layer_norm = LayerNorm(self.hidden_size, eps=1e-12, conditional_size=self.conditional_size, bias=False, norm_mode='rmsnorm')
        self.dropout = nn.Dropout(self.dropout_rate)

    def apply_final_layers(self, inputs):
        hidden_states = super().apply_final_layers(inputs)
        return self.dropout(self.final_layer_norm([hidden_states]))

    def load_variable(self, state_dict, name, prefix=''):
        variable = state_dict[name]
        if name in {'encoder.embed_tokens.weight', 'shared.weight'}:
            return self.load_embeddings(variable)
        else:
            return variable

    def variable_mapping(self, prefix=''):
        mapping = {f'{prefix}embeddings.word_embeddings.weight': 'encoder.embed_tokens.weight', f'{prefix}encoderLayer.0.multiHeadAttention.relative_positions_encoding.weight': 'encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight', f'{prefix}final_layer_norm.weight': 'encoder.final_layer_norm.weight'}
        for i in range(self.num_hidden_layers):
            mapping.update({f'{prefix}encoderLayer.{i}.multiHeadAttention.q.weight': f'encoder.block.{i}.layer.0.SelfAttention.q.weight', f'{prefix}encoderLayer.{i}.multiHeadAttention.k.weight': f'encoder.block.{i}.layer.0.SelfAttention.k.weight', f'{prefix}encoderLayer.{i}.multiHeadAttention.v.weight': f'encoder.block.{i}.layer.0.SelfAttention.v.weight', f'{prefix}encoderLayer.{i}.multiHeadAttention.o.weight': f'encoder.block.{i}.layer.0.SelfAttention.o.weight', f'{prefix}encoderLayer.{i}.layerNorm1.weight': f'encoder.block.{i}.layer.0.layer_norm.weight', f'{prefix}encoderLayer.{i}.feedForward.outputDense.weight': f'encoder.block.{i}.layer.1.DenseReluDense.wo.weight', f'{prefix}encoderLayer.{i}.layerNorm2.weight': f'encoder.block.{i}.layer.1.layer_norm.weight'})
            if self.version.endswith('t5.1.0'):
                mapping.update({f'{prefix}encoderLayer.{i}.feedForward.intermediateDense.weight': f'encoder.block.{i}.layer.1.DenseReluDense.wi.weight'})
            elif self.version.endswith('t5.1.1'):
                mapping.update({f'{prefix}encoderLayer.{i}.feedForward.intermediateDense.weight': f'encoder.block.{i}.layer.1.DenseReluDense.wi_0.weight', f'{prefix}encoderLayer.{i}.feedForward.intermediateDense1.weight': f'encoder.block.{i}.layer.1.DenseReluDense.wi_1.weight'})
        return mapping


class T5_Decoder(Decoder):

    @insert_arguments(version='t5.1.0')
    def __init__(self, *args, **kwargs):
        kwargs.update({'p_bias': 't5_relative', 'relative_attention_num_buckets': kwargs.get('relative_attention_num_buckets'), 'version': self.version, 'bias': False, 'norm_mode': 'rmsnorm'})
        super().__init__(*args, **kwargs)
        del self.embeddings.layerNorm
        layer = T5Layer(self.hidden_size, self.num_attention_heads, self.dropout_rate, self.attention_probs_dropout_prob, self.intermediate_size, self.hidden_act, is_dropout=self.is_dropout, conditional_size=self.conditional_size, is_decoder=True, **get_kw(BertLayer, kwargs))
        self.decoderLayer = nn.ModuleList([copy.deepcopy(layer) for _ in range(self.num_hidden_layers)])
        for i in range(1, self.num_hidden_layers):
            self.decoderLayer[i].multiHeadAttention.relative_positions_encoding.weight = self.decoderLayer[0].multiHeadAttention.relative_positions_encoding.weight
        self.final_layer_norm = LayerNorm(self.hidden_size, eps=1e-12, conditional_size=self.conditional_size, bias=False, norm_mode='rmsnorm')
        self.dropout = nn.Dropout(self.dropout_rate)

    def apply_final_layers(self, inputs):
        inputs[0][1] = self.dropout(self.final_layer_norm([inputs[0][1]]))
        return super().apply_final_layers(inputs)

    def load_variable(self, state_dict, name, prefix=''):
        variable = state_dict[name]
        if name in {f'decoder.embed_tokens.weight', 'lm_head.weight', 'shared.weight'}:
            return self.load_embeddings(variable)
        else:
            return variable

    def variable_mapping(self, prefix=''):
        mapping = {f'{prefix}embeddings.word_embeddings.weight': 'decoder.embed_tokens.weight', f'{prefix}decoderLayer.0.multiHeadAttention.relative_positions_encoding.weight': 'decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight', f'{prefix}final_layer_norm.weight': 'decoder.final_layer_norm.weight', f'{prefix}final_dense.weight': 'lm_head.weight'}
        for i in range(self.num_hidden_layers):
            mapping.update({f'{prefix}decoderLayer.{i}.multiHeadAttention.q.weight': f'decoder.block.{i}.layer.0.SelfAttention.q.weight', f'{prefix}decoderLayer.{i}.multiHeadAttention.k.weight': f'decoder.block.{i}.layer.0.SelfAttention.k.weight', f'{prefix}decoderLayer.{i}.multiHeadAttention.v.weight': f'decoder.block.{i}.layer.0.SelfAttention.v.weight', f'{prefix}decoderLayer.{i}.multiHeadAttention.o.weight': f'decoder.block.{i}.layer.0.SelfAttention.o.weight', f'{prefix}decoderLayer.{i}.layerNorm1.weight': f'decoder.block.{i}.layer.0.layer_norm.weight', f'{prefix}decoderLayer.{i}.crossAttention.q.weight': f'decoder.block.{i}.layer.1.EncDecAttention.q.weight', f'{prefix}decoderLayer.{i}.crossAttention.k.weight': f'decoder.block.{i}.layer.1.EncDecAttention.k.weight', f'{prefix}decoderLayer.{i}.crossAttention.v.weight': f'decoder.block.{i}.layer.1.EncDecAttention.v.weight', f'{prefix}decoderLayer.{i}.crossAttention.o.weight': f'decoder.block.{i}.layer.1.EncDecAttention.o.weight', f'{prefix}decoderLayer.{i}.layerNorm3.weight': f'decoder.block.{i}.layer.1.layer_norm.weight', f'{prefix}decoderLayer.{i}.feedForward.outputDense.weight': f'decoder.block.{i}.layer.2.DenseReluDense.wo.weight', f'{prefix}decoderLayer.{i}.layerNorm2.weight': f'decoder.block.{i}.layer.2.layer_norm.weight'})
            if self.version.endswith('t5.1.0'):
                mapping.update({f'{prefix}decoderLayer.{i}.feedForward.intermediateDense.weight': f'decoder.block.{i}.layer.2.DenseReluDense.wi.weight'})
            elif self.version.endswith('t5.1.1'):
                mapping.update({f'{prefix}decoderLayer.{i}.feedForward.intermediateDense.weight': f'decoder.block.{i}.layer.2.DenseReluDense.wi_0.weight', f'{prefix}decoderLayer.{i}.feedForward.intermediateDense1.weight': f'decoder.block.{i}.layer.2.DenseReluDense.wi_1.weight'})
        return mapping


class T5(Transformer):
    """Google的T5模型（Encoder-Decoder）
    """

    @delete_arguments('with_pool', 'with_mlm', 'with_nsp')
    def __init__(self, *args, tie_emb_src_tgt_weight=True, **kwargs):
        super(T5, self).__init__(*args, **kwargs)
        self.tie_emb_src_tgt_weight = tie_emb_src_tgt_weight
        self.encoder = T5_Encoder(*args, **kwargs)
        self.encoder.build(**kwargs)
        self.decoder = T5_Decoder(*args, **kwargs)
        self.decoder.build(**kwargs)

    def load_variable(self, state_dict, name, prefix=''):
        variable = state_dict[name]
        if name in {'shared.weight', 'encoder.embed_tokens.weight', 'decoder.embed_tokens.weight', 'lm_head.weight'}:
            return self.load_embeddings(variable)
        else:
            return variable

    def variable_mapping(self, prefix=''):
        mapping = self.encoder.variable_mapping(prefix='encoder.')
        mapping.update(self.decoder.variable_mapping(prefix='decoder.'))
        if self.tie_emb_src_tgt_weight:
            mapping.update({'encoder.embeddings.word_embeddings.weight': 'shared.weight', 'decoder.embeddings.word_embeddings.weight': 'shared.weight'})
        return mapping


class GPT(LM_Mask, BERT):
    """构建GPT模型；
    链接：https://github.com/openai/finetune-transformer-lm
    """

    @insert_arguments(final_activation='softmax')
    @delete_arguments('with_pool', 'with_mlm', 'with_nsp')
    def __init__(self, max_position, **kwargs):
        """GPT的embedding是token、position、segment三者embedding之和，跟BERT的主要区别是三者相加之后没有加LayerNormalization层。
           使用LM_Mask实现预训练ckpt中的bias参数，最后的全连接层由于和embedding层权重一致，因此直接从word_embedding取
        """
        super(GPT, self).__init__(max_position, **kwargs)
        del self.embeddings.layerNorm
        self.dense = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        self.dense.weight = self.embeddings.word_embeddings.weight
        self.final_activation = get_activation(self.final_activation)

    def apply_final_layers(self, inputs):
        hidden_state = super().apply_final_layers(inputs)
        logit = self.dense(hidden_state)
        return self.final_activation(logit)

    def load_variable(self, state_dict, name):
        return super(GPT, self).load_variable(state_dict, name, prefix='gpt')

    def variable_mapping(self):
        mapping = super(GPT, self).variable_mapping(prefix='gpt')
        return mapping


class GPT2(LM_Mask, BERT):
    """构建GPT模型；
    链接：https://github.com/openai/finetune-transformer-lm
    """

    @insert_arguments(final_activation='softmax')
    @delete_arguments('with_pool', 'with_mlm', 'with_nsp')
    def __init__(self, max_position, **kwargs):
        """GPT2的embedding是token、position两者embedding之和。
           1、跟BERT的主要区别是三者相加之后没有加LayerNormalization层。
           2、bert的layernorm是在attn/ffc之后，OpenAi-gpt2是在之前。
           使用LM_Mask实现预训练ckpt中的bias参数，最后的全连接层由于和embedding层权重一致，因此直接从word_embedding取
        """
        super(GPT2, self).__init__(max_position, **kwargs)
        del self.embeddings.layerNorm
        layer = self.Gpt2Layer(self.hidden_size, self.num_attention_heads, self.dropout_rate, self.attention_probs_dropout_prob, self.intermediate_size, self.hidden_act, is_dropout=self.is_dropout, conditional_size=self.conditional_size)
        self.encoderLayer = nn.ModuleList([(copy.deepcopy(layer) if layer_id in self.keep_hidden_layers else Identity()) for layer_id in range(self.num_hidden_layers)])
        self.LayerNormFinal = LayerNorm(self.hidden_size, eps=1e-12, conditional_size=self.conditional_size)
        self.dense = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        self.dense.weight = self.embeddings.word_embeddings.weight
        self.final_activation = get_activation(self.final_activation)

    def apply_final_layers(self, inputs):
        hidden_state = super().apply_final_layers(inputs)
        logit = self.dense(self.LayerNormFinal([hidden_state]))
        return self.final_activation(logit)

    def load_variable(self, state_dict, name):
        return super(GPT2, self).load_variable(state_dict, name, prefix='gpt2')

    def variable_mapping(self):
        mapping = super(GPT2, self).variable_mapping(prefix='gpt2')
        mapping.update({'LayerNormFinal.weight': 'gpt2.LayerNormFinal.weight', 'LayerNormFinal.bias': 'gpt2.LayerNormFinal.bias'})
        return mapping


    class Gpt2Layer(BertLayer):
        """未定义在layer.py中是因为该层针对gpt2_mlm模型，不可复用。
        顺序：LN --> Att --> Add --> LN --> FFN --> Add
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def forward(self, hidden_states, attention_mask, conditional_emb=None, encoder_hidden_states=None, encoder_attention_mask=None):
            x = self.layerNorm1((hidden_states, conditional_emb))
            self_attn_output = self.multiHeadAttention(x, attention_mask)
            hidden_states = hidden_states + self.dropout1(self_attn_output)
            x = self.layerNorm2((hidden_states, conditional_emb))
            ffn_output = self.feedForward(x)
            hidden_states = hidden_states + self.dropout2(ffn_output)
            return hidden_states


class GPT2_ML(LM_Mask, BERT):
    """构建GPT2_ML模型；
    链接: https://github.com/imcaspar/gpt2-ml；
    注意：GPT2_ML虽然号称GPT2，但是它的结构其实更接近GPT，它自称GPT2的原因大概是因为它开源的版本参数量达到了GPT2的15亿参数。
    看完ckpt中的key，和GPT的区别是embedding后也有layernorm，和bert的区别是第一个跳跃链接是在layernorm前，bert是在之后
    """

    @insert_arguments(final_activation='softmax')
    @delete_arguments('with_pool', 'with_mlm', 'with_nsp')
    def __init__(self, max_position, **kwargs):
        super().__init__(max_position, **kwargs)
        layer = self.Gpt2MlLayer(self.hidden_size, self.num_attention_heads, self.dropout_rate, self.attention_probs_dropout_prob, self.intermediate_size, self.hidden_act, is_dropout=self.is_dropout, conditional_size=self.conditional_size)
        self.encoderLayer = nn.ModuleList([(copy.deepcopy(layer) if layer_id in self.keep_hidden_layers else Identity()) for layer_id in range(self.num_hidden_layers)])
        self.dense = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        self.dense.weight = self.embeddings.word_embeddings.weight
        self.final_activation = get_activation(self.final_activation)

    def apply_final_layers(self, inputs):
        hidden_state = super().apply_final_layers(inputs)
        logit = self.dense(hidden_state)
        return self.final_activation(logit)

    def load_variable(self, state_dict, name):
        return super(GPT2_ML, self).load_variable(state_dict, name, prefix='gpt2_ml')

    def variable_mapping(self):
        mapping = super(GPT2_ML, self).variable_mapping(prefix='gpt2_ml')
        return mapping


    class Gpt2MlLayer(BertLayer):
        """未定义在layer.py中是因为该层针对gpt2_mlm模型，不可复用；
        顺序：Att --> Add --> LN --> FFN --> Add --> LN
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def forward(self, hidden_states, attention_mask, conditional_emb=None, encoder_hidden_states=None, encoder_attention_mask=None):
            self_attn_output = self.multiHeadAttention(hidden_states, attention_mask)
            hidden_states = hidden_states + self.dropout1(self_attn_output)
            x = self.layerNorm1((hidden_states, conditional_emb))
            ffn_output = self.feedForward(x)
            hidden_states = hidden_states + self.dropout2(ffn_output)
            hidden_states = self.layerNorm2((hidden_states, conditional_emb))
            return hidden_states


class Transformer_XL(BERT):
    """构建transformer-xl模型, 已加载；
    项目: https://github.com/kimiyoung/transformer-xl；
    不同点:  
    1) 简化了原有的AdaptiveEmbedding(可选)和未使用ProjectedAdaptiveLogSoftmax, 直接输出last_hidden_state；
    2) mems修改了transformer中初始化为zero_tensor, 改为包含最后一层, 原项目初始化为empty_tensor；
    3) SinusoidalPositionEncoding一般是sincos间隔排列, 这里是先sin后cos；
    4) attention_mask在multi_attn中使用中使用1e30来替代原来的1000。
    """

    @delete_arguments('with_pool', 'with_nsp', 'with_mlm')
    @insert_arguments(with_lm=False)
    def __init__(self, *args, mem_len=0, same_length=False, clamp_len=-1, **kwargs):
        kwargs.update({'p_bias': 'other_relative'})
        super().__init__(*args, **kwargs)
        self.mem_len, self.same_length, self.clamp_len = mem_len, same_length, clamp_len
        self.attn_type = kwargs.get('attn_type', 0)
        if kwargs.get('adaptive_embedding'):
            cutoffs, div_val, sample_softmax = kwargs.get('cutoffs', []), kwargs.get('div_val', 1), kwargs.get('sample_softmax', False)
            self.embeddings = AdaptiveEmbedding(self.vocab_size, self.embedding_size, self.hidden_size, cutoffs, div_val, sample_softmax, **get_kw(AdaptiveEmbedding, kwargs))
        else:
            self.embeddings = nn.Embedding(self.vocab_size, self.embedding_size)
        self.pos_embeddings = XlnetPositionsEncoding(self.embedding_size)
        self.dropout = nn.Dropout(self.dropout_rate)
        if not kwargs.get('untie_r'):
            self.r_w_bias = nn.Parameter(torch.FloatTensor(self.num_attention_heads, self.attention_head_size))
            self.r_r_bias = nn.Parameter(torch.FloatTensor(self.num_attention_heads, self.attention_head_size))
            if self.segment_vocab_size > 0:
                self.r_s_bias = nn.Parameter(torch.FloatTensor(self.num_attention_heads, self.attention_head_size))
        else:
            self.r_w_bias, self.r_r_bias = None, None
            self.r_s_bias = None
        layer = XlnetLayer(self.hidden_size, self.num_attention_heads, self.dropout_rate, self.attention_probs_dropout_prob, self.intermediate_size, self.hidden_act, is_dropout=self.is_dropout, conditional_size=self.conditional_size, r_w_bias=self.r_w_bias, r_r_bias=self.r_r_bias, r_s_bias=None, **get_kw(BertLayer, kwargs))
        self.encoderLayer = nn.ModuleList([(copy.deepcopy(layer) if layer_id in self.keep_hidden_layers else Identity()) for layer_id in range(self.num_hidden_layers)])
        if self.with_lm:
            self.dense = nn.Linear(self.hidden_size, self.vocab_size, bias=True)

    def init_mems(self, bsz):
        """初始化mems, 用于记忆mlen的各层隐含层状态
        """
        if isinstance(self.mem_len, (int, float)) and self.mem_len > 0:
            mems = []
            param = next(self.parameters())
            for _ in range(self.num_hidden_layers + 1):
                empty = torch.zeros(bsz, self.mem_len, self.hidden_size, dtype=param.dtype, device=param.device)
                mems.append(empty)
            return mems
        else:
            return None

    def _update_mems(self, hids, mlen, qlen):
        """更新mems
        """
        if self.mems is None:
            return None
        assert len(hids) == len(self.mems), 'len(hids) != len(mems)'
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + max(0, qlen)
            beg_idx = max(0, end_idx - self.mem_len)
            for i in range(len(hids)):
                cat = torch.cat([self.mems[i], hids[i]], dim=1)
                new_mems.append(cat[:, beg_idx:end_idx].detach())
        self.mems = new_mems

    def relative_positional_encoding(self, qlen, klen, device):
        pos_seq = torch.arange(klen - 1, -1, -1.0, device=device, dtype=torch.long)
        if self.clamp_len > 0:
            pos_seq.clamp_(max=self.clamp_len)
        pos_emb = self.dropout(self.pos_embeddings(pos_seq))
        return pos_emb

    def create_mask(self, word_emb, qlen, klen, mlen):
        if self.same_length:
            all_ones = word_emb.new_ones(qlen, klen)
            mask_len = klen - self.mem_len
            mask_shift_len = qlen - mask_len if mask_len > 0 else qlen
            attention_mask = 1 - (torch.triu(all_ones, 1 + mlen) + torch.tril(all_ones, -mask_shift_len)).byte()
        else:
            attention_mask = torch.tril(word_emb.new_ones(qlen, klen), diagonal=mlen).byte()
        attention_mask = attention_mask[None, None, :, :]
        return attention_mask

    def apply_embeddings(self, inputs):
        """接受的inputs输入: [token_ids, segment_ids], 暂不支持条件LayerNorm输入
        """
        assert isinstance(inputs, (tuple, list)), f'Inputs only support list,tuple format but passed {type(inputs)}'
        self.mems = self.init_mems(inputs[0].size(0))
        word_emb = self.dropout(self.embeddings(inputs[0]))
        index_ = 1
        btz, qlen = inputs[0].shape[:2]
        mlen = self.mems[0].size(1) if self.mems is not None else 0
        klen = mlen + qlen
        pos_emb = self.relative_positional_encoding(qlen, klen, word_emb.device)
        if self.segment_vocab_size > 0:
            segment_ids = inputs[index_]
            if mlen > 0:
                mem_pad = torch.zeros([btz, mlen], dtype=torch.long, device=word_emb.device)
                cat_ids = torch.cat([mem_pad, segment_ids], dim=1)
            else:
                cat_ids = segment_ids
            segment_ids = (segment_ids[:, :, None] != cat_ids[:, None]).long()
            index_ += 1
        else:
            segment_ids = None
        if self.attn_type in {'uni', 0}:
            non_tgt_mask = self.create_mask(word_emb, qlen, klen, mlen)
        elif self.attn_type == 'bi':
            attention_mask = (inputs[0] != self.token_pad_ids).long().unsqueeze(1).unsqueeze(2)
            non_tgt_mask = torch.eye(qlen)[None, None, :, :]
            non_tgt_mask = (1 - attention_mask - non_tgt_mask <= 0).long()
        return [word_emb, segment_ids, pos_emb, non_tgt_mask, None]

    def apply_main_layers(self, inputs):
        hidden_states, segment_ids, pos_emb, attention_mask, conditional_emb = inputs[:5]
        encoded_layers = [hidden_states]
        layer_inputs = [hidden_states, segment_ids, pos_emb, attention_mask, None, conditional_emb]
        for i, layer_module in enumerate(self.encoderLayer):
            mems_i = None if self.mems is None else self.mems[i]
            layer_inputs[-2] = mems_i
            layer_inputs = self.apply_on_layer_begin(i, layer_inputs)
            hidden_states = grad_checkpoint(layer_module, *layer_inputs) if self.gradient_checkpoint else layer_module(*layer_inputs)
            layer_inputs[0] = hidden_states
            layer_inputs = self.apply_on_layer_end(i, layer_inputs)
            encoded_layers.append(hidden_states)
        hidden_states = self.dropout(hidden_states)
        qlen = inputs[0].size(1)
        mlen = self.mems[0].size(0) if self.mems is not None else 0
        self._update_mems(encoded_layers, mlen, qlen)
        if not self.output_all_encoded_layers:
            encoded_layers = encoded_layers[:1] + [hidden_states]
        return [encoded_layers, conditional_emb]

    def load_variable(self, state_dict, name, prefix=''):
        if self.keep_tokens is not None or self.compound_tokens is not None:
            raise ValueError('Custom keep_tokens and compound_tokens is not yet supported in Transformer_XL')
        return state_dict[name]

    def variable_mapping(self, prefix=''):
        return {k: k for k, v in self.named_parameters()}


class XLNET(Transformer_XL):
    """构建xlnet模型, 这里做了简化, 只用来finetune, 即没有perm_mask, target_mapping这些输入；
       接受的inputs输入: [token_ids, segment_ids]
    """

    def __init__(self, *args, bi_data=False, **kwargs):
        self.attn_type = kwargs.get('attn_type', 'bi')
        self.bi_data = bi_data
        kwargs['rel_shift_opt'] = 'xlnet'
        super().__init__(*args, **kwargs)

    def relative_positional_encoding(self, qlen, klen, device):
        if self.attn_type == 'bi':
            beg, end = klen, -qlen
        elif self.attn_type == 'uni':
            beg, end = klen, -1
        else:
            raise ValueError(f'Unknown `attn_type` {self.attn_type}.')
        pos_seq = torch.arange(beg, end, -1.0, device=device, dtype=torch.long)
        if self.clamp_len > 0:
            pos_seq.clamp_(max=self.clamp_len)
        fwd_pos_emb = self.pos_embeddings(pos_seq)
        if self.bi_data:
            pos_seq = torch.arange(-beg, -end, -1.0, device=device, dtype=torch.long)
            if self.clamp_len > 0:
                pos_seq.clamp_(max=self.clamp_len)
            bwd_pos_emb = self.pos_embeddings(pos_seq)
            pos_emb = torch.cat([fwd_pos_emb, bwd_pos_emb], dim=0)
        else:
            pos_emb = fwd_pos_emb
        pos_emb = self.dropout(pos_emb)
        return pos_emb

    def apply_final_layers(self, inputs):
        hidden_state = super().apply_final_layers(inputs)
        if self.with_lm:
            return [hidden_state, self.dense(hidden_state)]
        else:
            return hidden_state

    def load_variable(self, state_dict, name, prefix='transformer'):
        variable = state_dict[name]
        if name in {f'{prefix}.word_embedding.weight', 'lm_loss.weight', 'lm_loss.bias'}:
            return self.load_embeddings(variable)
        elif re.search('rel_attn\\.(q|k|v|r)$', name):
            return variable.reshape(variable.shape[0], -1).T
        elif re.search('rel_attn\\.(o)$', name):
            return variable.reshape(variable.shape[0], -1)
        else:
            return variable

    def variable_mapping(self, prefix='transformer'):
        mapping = {'embeddings.weight': f'{prefix}.word_embedding.weight', 'dense.weight': 'lm_loss.weight', 'dense.bias': 'lm_loss.bias'}
        for i in range(self.num_hidden_layers):
            prefix_i = f'{prefix}.layer.%d.' % i
            mapping.update({f'encoderLayer.{i}.multiHeadAttention.q.weight': prefix_i + 'rel_attn.q', f'encoderLayer.{i}.multiHeadAttention.k.weight': prefix_i + 'rel_attn.k', f'encoderLayer.{i}.multiHeadAttention.v.weight': prefix_i + 'rel_attn.v', f'encoderLayer.{i}.multiHeadAttention.o.weight': prefix_i + 'rel_attn.o', f'encoderLayer.{i}.multiHeadAttention.r.weight': prefix_i + 'rel_attn.r', f'encoderLayer.{i}.multiHeadAttention.r_r_bias': prefix_i + 'rel_attn.r_r_bias', f'encoderLayer.{i}.multiHeadAttention.r_s_bias': prefix_i + 'rel_attn.r_s_bias', f'encoderLayer.{i}.multiHeadAttention.r_w_bias': prefix_i + 'rel_attn.r_w_bias', f'encoderLayer.{i}.multiHeadAttention.seg_embed': prefix_i + 'rel_attn.seg_embed', f'encoderLayer.{i}.layerNorm1.weight': prefix_i + 'rel_attn.layer_norm.weight', f'encoderLayer.{i}.layerNorm1.bias': prefix_i + 'rel_attn.layer_norm.bias', f'encoderLayer.{i}.feedForward.intermediateDense.weight': prefix_i + 'ff.layer_1.weight', f'encoderLayer.{i}.feedForward.intermediateDense.bias': prefix_i + 'ff.layer_1.bias', f'encoderLayer.{i}.feedForward.outputDense.weight': prefix_i + 'ff.layer_2.weight', f'encoderLayer.{i}.feedForward.outputDense.bias': prefix_i + 'ff.layer_2.bias', f'encoderLayer.{i}.layerNorm2.weight': prefix_i + 'ff.layer_norm.weight', f'encoderLayer.{i}.layerNorm2.bias': prefix_i + 'ff.layer_norm.bias'})
        return mapping


def _is_control(char):
    """Checks whether `chars` is a control character."""
    if char == '\t' or char == '\n' or char == '\r':
        return False
    cat = unicodedata.category(char)
    if cat.startswith('C'):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    if cp >= 33 and cp <= 47 or cp >= 58 and cp <= 64 or cp >= 91 and cp <= 96 or cp >= 123 and cp <= 126:
        return True
    cat = unicodedata.category(char)
    if cat.startswith('P'):
        return True
    return False


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    if char == ' ' or char == '\t' or char == '\n' or char == '\r':
        return True
    cat = unicodedata.category(char)
    if cat == 'Zs':
        return True
    return False


def whitespace_tokenize(text):
    """去除文本中的空白符"""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


class BasicTokenizer(object):
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

    def __init__(self, do_lower_case=True, never_split=('[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]')):
        """Constructs a BasicTokenizer.
        Args:
          do_lower_case: Whether to lower case the input.
        """
        self.do_lower_case = do_lower_case
        self.never_split = never_split

    def tokenize(self, text):
        """文本切分成token"""
        text = self._clean_text(text)
        text = self._tokenize_chinese_chars(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case and token not in self.never_split:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))
        output_tokens = whitespace_tokenize(' '.join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize('NFD', text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == 'Mn':
                continue
            output.append(char)
        return ''.join(output)

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        if text in self.never_split:
            return [text]
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1
        return [''.join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(' ')
                output.append(char)
                output.append(' ')
            else:
                output.append(char)
        return ''.join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        if cp >= 19968 and cp <= 40959 or cp >= 13312 and cp <= 19903 or cp >= 131072 and cp <= 173791 or cp >= 173824 and cp <= 177983 or cp >= 177984 and cp <= 178207 or cp >= 178208 and cp <= 183983 or cp >= 63744 and cp <= 64255 or cp >= 194560 and cp <= 195103:
            return True
        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 65533 or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(' ')
            else:
                output.append(char)
        return ''.join(output)


class Trie(object):
    """自定义Trie树对象，用来保存知识库
    """

    def __init__(self, value_key=-1):
        self.data = {}
        self.value_key = str(value_key)

    def __setitem__(self, key, value):
        """传入一对(key, value)到前缀树中
        """
        data = self.data
        for k in key:
            k = str(k)
            if k not in data:
                data[k] = {}
            data = data[k]
        if self.value_key in data:
            if data[self.value_key] != value:
                data[self.value_key] += '\t' + value
        else:
            data[self.value_key] = value

    def __getitem__(self, key):
        """获取key对应的value
        """
        data = self.data
        for k in key:
            k = str(k)
            data = data[k]
        return data[self.value_key]

    def next_ones(self, prefix):
        """获取prefix后一位的容许集
        """
        data = self.data
        for k in prefix:
            k = str(k)
            data = data[k]
        return [k for k in data if k != self.value_key]

    def keys(self, prefix=None, data=None):
        """获取以prefix开头的所有key
        """
        data = data or self.data
        prefix = prefix or []
        for k in prefix:
            k = str(k)
            if k not in data:
                return []
            data = data[k]
        results = []
        for k in data:
            if k == self.value_key:
                results.append([])
            else:
                results.extend([([k] + j) for j in self.keys(None, data[k])])
        return [(prefix + i) for i in results]

    def save(self, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False)

    def load(self, filename):
        with open(filename, encoding='utf-8') as f:
            self.data = json.load(f)


def is_string(s):
    """判断是否是字符串
    """
    return isinstance(s, basestring)


def truncate_sequences(maxlen, indices, *sequences):
    """截断总长度至不超过maxlen
    """
    sequences = [s for s in sequences if s]
    if not isinstance(indices, (list, tuple)):
        indices = [indices] * len(sequences)
    while True:
        lengths = [len(s) for s in sequences]
        if sum(lengths) > maxlen:
            i = np.argmax(lengths)
            sequences[i].pop(indices[i])
        else:
            return sequences


class TokenizerBase(object):
    """分词器基类
    """

    def __init__(self, token_start='[CLS]', token_end='[SEP]', token_unk='[UNK]', token_pad='[PAD]', token_mask='[MASK]', add_special_tokens=None, pre_tokenize=None, token_translate=None):
        """参数说明：
        token_unk: 未知词标记
        token_end: 句子切分标记，当只有一句话作为输入时，此标记知识作为结束符；当有两句话作为输入时，此标记作为分隔符、最后一句话的结束符
        pad_token: padding填充标记
        token_start: 分类标记，位于整个序列的第一个
        mask_token: mask标记
        pre_tokenize: 外部传入的分词函数，用作对文本进行预分词。如果传入pre_tokenize，则先执行pre_tokenize(text)，然后在它的基础上执行原本的tokenize函数；
        token_translate: 映射字典，主要用在tokenize之后，将某些特殊的token替换为对应的token。
        """
        self._token_pad = token_pad
        self._token_unk = token_unk
        self._token_mask = token_mask
        self._token_start = token_start
        self._token_end = token_end
        self.never_split = [i for i in [self._token_unk, self._token_end, self._token_pad, self._token_start, self._token_mask] if isinstance(i, str)]
        if add_special_tokens is not None:
            if isinstance(add_special_tokens, (tuple, list)):
                self.never_split.extend(add_special_tokens)
            elif isinstance(add_special_tokens, str):
                self.never_split.append(add_special_tokens)
        self.tokens_trie = self._create_trie(self.never_split)
        self._pre_tokenize = pre_tokenize
        self._token_translate = token_translate or {}
        self._token_translate_inv = {v: k for k, v in self._token_translate.items()}

    def _create_trie(self, unique_no_split_tokens):
        trie = Trie()
        for token in unique_no_split_tokens:
            trie.add(token)
        return trie

    def tokenize(self, text, maxlen=None):
        """分词函数
        """
        tokens = [(self._token_translate.get(token) or token) for token in self._tokenize(text)]
        if self._token_start is not None:
            tokens.insert(0, self._token_start)
        if self._token_end is not None:
            tokens.append(self._token_end)
        if maxlen is not None:
            index = int(self._token_end is not None) + 1
            truncate_sequences(maxlen, -index, tokens)
        return tokens

    def token_to_id(self, token):
        """token转换为对应的id
        """
        raise NotImplementedError

    def tokens_to_ids(self, tokens):
        """token序列转换为对应的id序列
        """
        return [self.token_to_id(token) for token in tokens]

    def _encode(self, first_text, second_text=None, maxlen=None, pattern='S*E*E', truncate_from='right', return_offsets=False):
        """输出文本对应token id和segment id
        """
        first_tokens = self.tokenize(first_text) if is_string(first_text) else first_text
        if second_text is None:
            second_tokens = None
        elif is_string(second_text):
            second_tokens = self.tokenize(second_text)
        else:
            second_tokens = second_text
        if maxlen is not None:
            if truncate_from == 'right':
                index = -int(self._token_end is not None) - 1
            elif truncate_from == 'left':
                index = int(self._token_start is not None)
            else:
                index = truncate_from
            if second_text is not None and pattern == 'S*E*E':
                maxlen += 1
            truncate_sequences(maxlen, index, first_tokens, second_tokens)
        first_token_ids = self.tokens_to_ids(first_tokens)
        first_segment_ids = [0] * len(first_token_ids)
        if second_text is not None:
            if pattern == 'S*E*E':
                idx = int(bool(self._token_start))
                second_tokens = second_tokens[idx:]
            second_token_ids = self.tokens_to_ids(second_tokens)
            second_segment_ids = [1] * len(second_token_ids)
            first_token_ids.extend(second_token_ids)
            first_segment_ids.extend(second_segment_ids)
        encode_output = [first_token_ids, first_segment_ids]
        if return_offsets != False:
            offset = self.rematch(first_text, first_tokens) + self.rematch(second_text, second_tokens)
            if return_offsets == 'transformers':
                encode_output.append([([0, 0] if not k else [k[0], k[-1] + 1]) for k in offset])
            else:
                encode_output.append(offset)
        return encode_output

    def encode(self, first_texts, second_texts=None, maxlen=None, pattern='S*E*E', truncate_from='right', return_offsets=False):
        """可以处理多条或者单条
        """
        return_list = False if isinstance(first_texts, str) else True
        first_texts = [first_texts] if isinstance(first_texts, str) else first_texts
        second_texts = [second_texts] if isinstance(second_texts, str) else second_texts
        first_token_ids, first_segment_ids, offsets = [], [], []
        if second_texts is None:
            second_texts = [None] * len(first_texts)
        assert len(first_texts) == len(second_texts), 'first_texts and second_texts should be same length'
        for first_text, second_text in zip(first_texts, second_texts):
            outputs = self._encode(first_text, second_text, maxlen, pattern, truncate_from, return_offsets)
            first_token_ids.append(outputs[0])
            first_segment_ids.append(outputs[1])
            if len(outputs) >= 3:
                offsets.append(outputs[2])
        encode_outputs = [first_token_ids, first_segment_ids]
        if return_offsets:
            encode_outputs.append(offsets)
        if not return_list:
            encode_outputs = [item[0] for item in encode_outputs]
        return encode_outputs

    def id_to_token(self, i):
        """id序列为对应的token
        """
        raise NotImplementedError

    def ids_to_tokens(self, ids):
        """id序列转换为对应的token序列
        """
        return [self.id_to_token(i) for i in ids]

    def decode(self, ids):
        """转为可读文本
        """
        raise NotImplementedError

    def _tokenize(self, text):
        """基本分词函数
        """
        raise NotImplementedError


class WordpieceTokenizer(object):
    """Runs WordPiece tokenization."""

    def __init__(self, vocab, unk_token='[UNK]', max_input_chars_per_word=100, do_tokenize_unk=False):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word
        self.do_tokenize_unk = do_tokenize_unk

    def tokenize(self, text):
        """Tokenizes a piece of text into its word pieces.

        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.

        For example:
          input = "unaffable"
          output = ["un", "##aff", "##able"]

        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer`.

        Returns:
          A list of wordpiece tokens.
        """
        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token if self.do_tokenize_unk else token)
                continue
            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = ''.join(chars[start:end])
                    if start > 0:
                        substr = '##' + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end
            if self.do_tokenize_unk and is_bad:
                output_tokens.append(self.unk_token)
            elif not self.do_tokenize_unk and is_bad:
                output_tokens.append(substr)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


def load_vocab(dict_path, encoding='utf-8', simplified=False, startswith=None):
    """加载词典文件到dict"""
    token_dict = collections.OrderedDict()
    index = 0
    with open(dict_path, 'r', encoding=encoding) as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            token_dict[token] = index
            index += 1
    if simplified:
        new_token_dict, keep_tokens = {}, []
        startswith = startswith or []
        for t in startswith:
            new_token_dict[t] = len(new_token_dict)
            keep_tokens.append(token_dict[t])
        for t, _ in sorted(token_dict.items(), key=lambda s: s[1]):
            if t not in new_token_dict and not Tokenizer._is_redundant(t):
                new_token_dict[t] = len(new_token_dict)
                keep_tokens.append(token_dict[t])
        return new_token_dict, keep_tokens
    else:
        return token_dict


def lowercase_and_normalize(text, never_split=()):
    """转小写，并进行简单的标准化
    """
    if is_py2:
        text = unicode(text)
    escaped_special_toks = [re.escape(s_tok) for s_tok in never_split]
    pattern = '(' + '|'.join(escaped_special_toks) + ')|' + '(.+?)'
    text = re.sub(pattern, lambda m: m.groups()[0] or m.groups()[1].lower(), text)
    text = unicodedata.normalize('NFD', text)
    text = ''.join([ch for ch in text if unicodedata.category(ch) != 'Mn'])
    return text


class Tokenizer(TokenizerBase):
    """Bert原生分词器
    """

    def __init__(self, token_dict, do_lower_case=True, do_basic_tokenize=True, do_tokenize_unk=False, **kwargs):
        """
        参数:
            token_dict:
                词典文件
            do_lower_case:
                是否转换成小写
            do_basic_tokenize:
                分词前，是否进行基础的分词
            do_tokenize_unk:
                分词后，是否生成[UNK]标记，还是在encode阶段生成
        """
        super(Tokenizer, self).__init__(**kwargs)
        if is_string(token_dict):
            token_dict = load_vocab(token_dict)
        self._do_lower_case = do_lower_case
        self._vocab_size = len(token_dict)
        self._token_dict = token_dict
        self._token_dict_inv = {v: k for k, v in token_dict.items()}
        self.do_basic_tokenize = do_basic_tokenize
        if do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case, never_split=self.never_split)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self._token_dict, unk_token=self._token_unk, do_tokenize_unk=do_tokenize_unk)
        for token in ['pad', 'unk', 'mask', 'start', 'end']:
            try:
                _token_id = token_dict[getattr(self, '_token_%s' % token)]
                setattr(self, '_token_%s_id' % token, _token_id)
            except:
                pass

    def _tokenize(self, text, pre_tokenize=True):
        """基本分词函数
        """
        if self._do_lower_case:
            text = lowercase_and_normalize(text, never_split=self.never_split)
        if pre_tokenize and self._pre_tokenize is not None:
            tokens = []
            for token in self._pre_tokenize(text):
                if token in self._token_dict:
                    tokens.append(token)
                else:
                    tokens.extend(self._tokenize(token, False))
            return tokens
        text_pieces = self.tokens_trie.split(text)
        split_tokens = []
        for text_piece in text_pieces:
            if not text_piece:
                continue
            elif text_piece in self._token_dict:
                split_tokens.append(text_piece)
            elif self.do_basic_tokenize:
                for token in self.basic_tokenizer.tokenize(text_piece):
                    for sub_token in self.wordpiece_tokenizer.tokenize(token):
                        split_tokens.append(sub_token)
            else:
                split_tokens.extend(self.wordpiece_tokenizer.tokenize(text_piece))
        return split_tokens

    def token_to_id(self, token):
        """token转为vocab中的id"""
        return self._token_dict.get(token, self._token_unk_id)

    def id_to_token(self, id):
        """id转为词表中的token"""
        return self._token_dict_inv[id]

    def decode(self, ids, tokens=None):
        """转为可读文本
        """
        tokens = tokens or self.ids_to_tokens(ids)
        tokens = [token for token in tokens if not self._is_special(token)]
        text, flag = '', False
        for i, token in enumerate(tokens):
            if token[:2] == '##':
                text += token[2:]
            elif len(token) == 1 and self._is_cjk_character(token):
                text += token
            elif len(token) == 1 and self._is_punctuation(token):
                text += token
                text += ' '
            elif i > 0 and self._is_cjk_character(text[-1]):
                text += token
            else:
                text += ' '
                text += token
        text = re.sub(' +', ' ', text)
        text = re.sub("' (re|m|s|t|ve|d|ll) ", "'\\1 ", text)
        punctuation = self._cjk_punctuation() + '+-/={(<['
        punctuation_regex = '|'.join([re.escape(p) for p in punctuation])
        punctuation_regex = '(%s) ' % punctuation_regex
        text = re.sub(punctuation_regex, '\\1', text)
        text = re.sub('(\\d\\.) (\\d)', '\\1\\2', text)
        return text.strip()

    @staticmethod
    def stem(token):
        """获取token的“词干”（如果是##开头，则自动去掉##）
        """
        if token[:2] == '##':
            return token[2:]
        else:
            return token

    @staticmethod
    def _is_space(ch):
        """空格类字符判断
        """
        return ch == ' ' or ch == '\n' or ch == '\r' or ch == '\t' or unicodedata.category(ch) == 'Zs'

    @staticmethod
    def _is_punctuation(ch):
        """标点符号类字符判断（全/半角均在此内）
        提醒：unicodedata.category这个函数在py2和py3下的
        表现可能不一样，比如u'§'字符，在py2下的结果为'So'，
        在py3下的结果是'Po'。
        """
        code = ord(ch)
        return 33 <= code <= 47 or 58 <= code <= 64 or 91 <= code <= 96 or 123 <= code <= 126 or unicodedata.category(ch).startswith('P')

    @staticmethod
    def _cjk_punctuation():
        return u'＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､\u3000、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·！？｡。'

    @staticmethod
    def _is_cjk_character(ch):
        """CJK类字符判断（包括中文字符也在此列）
        参考：https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        """
        code = ord(ch)
        return 19968 <= code <= 40959 or 13312 <= code <= 19903 or 131072 <= code <= 173791 or 173824 <= code <= 177983 or 177984 <= code <= 178207 or 178208 <= code <= 183983 or 63744 <= code <= 64255 or 194560 <= code <= 195103

    @staticmethod
    def _is_control(ch):
        """控制类字符判断
        """
        return unicodedata.category(ch) in ('Cc', 'Cf')

    @staticmethod
    def _is_special(ch):
        """判断是不是有特殊含义的符号
        """
        return bool(ch) and ch[0] == '[' and ch[-1] == ']'

    @staticmethod
    def _is_redundant(token):
        """判断该token是否冗余（默认情况下不可能分出来）
        """
        if len(token) > 1:
            for ch in Tokenizer.stem(token):
                if Tokenizer._is_cjk_character(ch) or Tokenizer._is_punctuation(ch):
                    return True

    def rematch(self, text, tokens):
        """给出原始的text和tokenize后的tokens的映射关系
        """
        if is_py2:
            text = unicode(text)
        if self._do_lower_case:
            text = text.lower()
        normalized_text, char_mapping = '', []
        for i, ch in enumerate(text):
            if self._do_lower_case:
                ch = lowercase_and_normalize(ch, self.never_split)
            ch = ''.join([c for c in ch if not (ord(c) == 0 or ord(c) == 65533 or self._is_control(c))])
            normalized_text += ch
            char_mapping.extend([i] * len(ch))
        text, token_mapping, offset = normalized_text, [], 0
        for token in tokens:
            if self._is_special(token):
                token_mapping.append([])
            else:
                token = self.stem(token)
                start = text[offset:].index(token) + offset
                end = start + len(token)
                token_mapping.append(char_mapping[start:end])
                offset = end
        return token_mapping


dict_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/vocab.txt'


class CrossEntropyLoss(nn.CrossEntropyLoss):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, outputs, target):
        """
        y_pred: [btz, seq_len, vocab_size]
        targets: y_true, y_segment
        unilm式样，需要手动把非seq2seq部分mask掉
        """
        _, y_pred = outputs
        y_true, y_unilm_mask = target
        y_true = y_true[:, 1:]
        y_unilm_mask = y_unilm_mask[:, 1:]
        y_pred = y_pred[:, :-1, :]
        y_pred = y_pred.reshape(-1, y_pred.shape[-1])
        y_padd_mask = (y_true != tokenizer._token_pad_id).long()
        y_true = (y_true * y_padd_mask * y_unilm_mask).flatten()
        return super().forward(y_pred, y_true)


class MyLoss(nn.Module):

    def forward(self, y_pred, y_true_sup):
        y_pred_sup = y_pred[:y_true_sup.shape[0]]
        return F.cross_entropy(y_pred_sup, y_true_sup)


class BERT_THESEUS(BERT):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layer = BertLayer(self.hidden_size, self.num_attention_heads, self.dropout_rate, self.attention_probs_dropout_prob, self.intermediate_size, self.hidden_act, is_dropout=False, conditional_size=self.conditional_size)
        self.encoderLayer = nn.ModuleList(nn.ModuleList([copy.deepcopy(layer) for _ in range(self.num_hidden_layers)]))
        self.scc_n_layer = 6
        self.scc_layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(self.scc_n_layer)])
        self.compress_ratio = self.num_hidden_layers // self.scc_n_layer
        self.bernoulli = None

    def set_replacing_rate(self, replacing_rate):
        if not 0 < replacing_rate <= 1:
            raise Exception('Replace rate must be in the range (0, 1]!')
        self.bernoulli = Bernoulli(torch.tensor([replacing_rate]))

    def apply_main_layers(self, inputs):
        """BERT的主体是基于Self-Attention的模块
        顺序:Att --> Add --> LN --> FFN --> Add --> LN
        """
        hidden_states, attention_mask, conditional_emb = inputs
        encoded_layers = [hidden_states]
        if self.training:
            inference_layers = []
            for i in range(self.scc_n_layer):
                if self.bernoulli.sample() == 1:
                    inference_layers.append(self.scc_layer[i])
                else:
                    for offset in range(self.compress_ratio):
                        inference_layers.append(self.encoderLayer[i * self.compress_ratio + offset])
        else:
            inference_layers = self.scc_layer
        for i, layer_module in enumerate(inference_layers):
            hidden_states = layer_module(hidden_states, attention_mask, conditional_emb)
            if self.output_all_encoded_layers:
                encoded_layers.append(hidden_states)
        if not self.output_all_encoded_layers:
            encoded_layers.append(hidden_states)
        return [encoded_layers, conditional_emb]


class TotalLoss(nn.Module):
    """loss分两部分，一是seq2seq的交叉熵，二是相似度的交叉熵。
    """

    def forward(self, outputs, target):
        seq_logit, sen_emb = outputs
        seq_label, seq_mask = target
        seq2seq_loss = self.compute_loss_of_seq2seq(seq_logit, seq_label, seq_mask)
        similarity_loss = self.compute_loss_of_similarity(sen_emb)
        return {'loss': seq2seq_loss + similarity_loss, 'seq2seq_loss': seq2seq_loss, 'similarity_loss': similarity_loss}

    def compute_loss_of_seq2seq(self, y_pred, y_true, y_mask):
        """
        y_pred: [btz, seq_len, hdsz]
        y_true: [btz, seq_len]
        y_mask: [btz, seq_len]
        """
        y_true = y_true[:, 1:]
        y_mask = y_mask[:, 1:]
        y_pred = y_pred[:, :-1, :]
        y_pred = y_pred.reshape(-1, y_pred.shape[-1])
        y_true = (y_true * y_mask).flatten()
        return F.cross_entropy(y_pred, y_true, ignore_index=0)

    def compute_loss_of_similarity(self, y_pred):
        y_true = self.get_labels_of_similarity(y_pred)
        y_pred = F.normalize(y_pred, p=2, dim=-1)
        similarities = torch.matmul(y_pred, y_pred.T)
        similarities = similarities - torch.eye(y_pred.shape[0], device=device) * 1000000000000.0
        similarities = similarities * 30
        loss = F.cross_entropy(similarities, y_true)
        return loss

    def get_labels_of_similarity(self, y_pred):
        idxs = torch.arange(0, y_pred.shape[0], device=device)
        idxs_1 = idxs[None, :]
        idxs_2 = (idxs + 1 - idxs % 2 * 2)[:, None]
        labels = idxs_1.eq(idxs_2).float()
        return labels


class BCELoss(nn.BCELoss):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, inputs, targets):
        subject_preds, object_preds = inputs
        subject_labels, object_labels, mask = targets
        subject_preds = subject_preds[:subject_labels.shape[0], :subject_labels.shape[1], :]
        object_preds = object_preds[:object_labels.shape[0]]
        object_preds = object_preds[:object_labels.shape[0], :object_labels.shape[1], :]
        mask = mask[:, :subject_labels.shape[1]]
        subject_loss = super().forward(subject_preds, subject_labels)
        subject_loss = subject_loss.mean(dim=-1)
        subject_loss = (subject_loss * mask).sum() / mask.sum()
        object_loss = super().forward(object_preds, object_labels)
        object_loss = object_loss.mean(dim=-1).sum(dim=-1)
        object_loss = (object_loss * mask).sum() / mask.sum()
        return {'loss': subject_loss + object_loss, 'subject_loss': subject_loss, 'object_loss': object_loss}


class MultiNonLinearClassifier(nn.Module):

    def __init__(self, hidden_size, tag_size, dropout_rate):
        super(MultiNonLinearClassifier, self).__init__()
        self.tag_size = tag_size
        self.linear = nn.Linear(hidden_size, int(hidden_size / 2))
        self.hidden2tag = nn.Linear(int(hidden_size / 2), self.tag_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_features):
        features_tmp = self.linear(input_features)
        features_tmp = nn.ReLU()(features_tmp)
        features_tmp = self.dropout(features_tmp)
        features_output = self.hidden2tag(features_tmp)
        return features_output


class SequenceLabelForSO(nn.Module):

    def __init__(self, hidden_size, tag_size, dropout_rate):
        super(SequenceLabelForSO, self).__init__()
        self.tag_size = tag_size
        self.linear = nn.Linear(hidden_size, int(hidden_size / 2))
        self.hidden2tag_sub = nn.Linear(int(hidden_size / 2), self.tag_size)
        self.hidden2tag_obj = nn.Linear(int(hidden_size / 2), self.tag_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_features):
        """
        Args:
            input_features: (bs, seq_len, h)
        """
        features_tmp = self.linear(input_features)
        features_tmp = nn.ReLU()(features_tmp)
        features_tmp = self.dropout(features_tmp)
        sub_output = self.hidden2tag_sub(features_tmp)
        obj_output = self.hidden2tag_obj(features_tmp)
        return sub_output, obj_output


class DottableDict(dict):
    """支持点操作符的字典
    """

    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.__dict__ = self

    def allowDotting(self, state=True):
        if state:
            self.__dict__ = self
        else:
            self.__dict__ = dict()


def extend_with_language_model(InputModel):
    """添加下三角的Attention Mask（语言模型用）
    """


    class LanguageModel(LM_Mask, InputModel):
        """带下三角Attention Mask的派生模型
        """

        def __init__(self, *args, **kwargs):
            kwargs['with_mlm'] = kwargs.get('with_mlm') or True
            super(LanguageModel, self).__init__(*args, **kwargs)
    return LanguageModel


class UniLM_Mask(object):
    """定义UniLM的Attention Mask（Seq2Seq模型用）；
    其中source和target的分区，由segment_ids来表示。
    UniLM: https://arxiv.org/abs/1905.03197
    """

    def compute_attention_bias(self, inputs=None):
        """通过idxs序列的比较来得到对应的mask
        """
        segment_ids = inputs[1]
        attention_bias = torch.cumsum(segment_ids, dim=1)
        attention_bias = attention_bias.unsqueeze(1) <= attention_bias.unsqueeze(2)
        self.attention_bias = attention_bias.unsqueeze(1).long()
        return self.attention_bias


def extend_with_unified_language_model(InputModel):
    """添加UniLM的Attention Mask（Seq2Seq模型用）
    """


    class UnifiedLanguageModel(UniLM_Mask, InputModel):
        """带UniLM的Attention Mask的派生模型
        UniLM: https://arxiv.org/abs/1905.03197
        """

        def __init__(self, *args, **kwargs):
            kwargs['with_mlm'] = kwargs.get('with_mlm') or True
            super(UnifiedLanguageModel, self).__init__(*args, **kwargs)
    return UnifiedLanguageModel


def build_transformer_model(config_path=None, checkpoint_path=None, model='bert', application='encoder', **kwargs):
    """根据配置文件构建模型，可选加载checkpoint权重

    :param config_path: str, 模型的config文件地址
    :param checkpoint_path: str, 模型文件地址, 默认值None表示不加载预训练模型
    :param model: str, 加载的模型结构, 这里Model也可以基于nn.Module自定义后传入, 默认为'bert'
    :param application: str, 模型应用, 支持encoder, lm和unilm格式, 默认为'encoder'
    :param segment_vocab_size: int, type_token_ids数量, 默认为2, 如不传入segment_ids则需设置为0
    :param with_pool: bool, 是否包含Pool部分, 默认为False
    :param with_nsp: bool, 是否包含NSP部分, 默认为False
    :param with_mlm: bool, 是否包含MLM部分, 默认为False
    :param output_all_encoded_layers: bool, 是否返回所有hidden_state层, 默认为False
    :param layer_add_embs: nn.Module, 自定义额外的embedding输入, 如nn.Embedding(2, 768)
    :param keep_tokens: list[int], 精简词表, 保留的id的序号如：[0, 100, 101, 102, 106, 107, ...]
    :param token_pad_ids: int, 默认为0, 部分模型padding不是0时在这里指定, 用于attention_mask生成, 如设置成-100
    :param custom_position_ids: bool, 是否自行传入位置id, True表示传入, False表示不传入, 'start_at_padding'表示从padding_idx+1开始, 默认为False
    :param custom_attention_mask: bool, 是否自行传入attention_mask, 默认为False
    :param shared_segment_embeddings: bool, 若True, 则segment跟token共用embedding, 默认为False
    :param layer_norm_cond: conditional layer_norm, 默认为None
    :param add_trainer: bool, 指定从BaseModel继承, 若build_transformer_model后需直接compile()、fit()需设置为True, 默认为None
    :param compound_tokens: 扩展Embedding, 默认为None
    :param residual_attention_scores: bool, Attention矩阵加残差, 默认为False
    :param ignore_invalid_weights: bool, 允许跳过不存在的权重, 默认为False
    :param keep_hidden_layers: 保留的hidden_layer层的id, 默认为None表示全部使用
    :param hierarchical_position: 是否层次分解位置编码, 默认为None表示不使用
    :param gradient_checkpoint: bool, 是否使用gradient_checkpoint, 默认为False
    :return: A pytorch model instance
    """
    configs = {}
    if config_path is not None:
        configs.update(json.load(open(config_path)))
    configs.update(kwargs)
    if 'max_position' not in configs:
        configs['max_position'] = configs.get('max_position_embeddings', 512)
    if 'dropout_rate' not in configs:
        configs['dropout_rate'] = configs.get('hidden_dropout_prob')
    if 'segment_vocab_size' not in configs:
        configs['segment_vocab_size'] = configs.get('type_vocab_size', 2)
    models = {'bert': BERT, 'roberta': BERT, 'albert': ALBERT, 'albert_unshared': ALBERT_Unshared, 'nezha': NEZHA, 'roformer': RoFormer, 'roformer_v2': RoFormerV2, 'gau_alpha': GAU_alpha, 'electra': ELECTRA, 'ernie': ERNIE, 'deberta_v2': DebertaV2, 'uie': UIE, 'encoder': Encoder, 'decoder': Decoder, 'transformer': Transformer, 'bart': BART, 'gpt': GPT, 'gpt2': GPT2, 'gpt2_ml': GPT2_ML, 't5': T5, 't5_encoder': T5_Encoder, 't5_decoder': T5_Decoder, 't5.1.0': T5, 't5.1.0_encoder': T5_Encoder, 't5.1.0_decoder': T5_Decoder, 't5.1.1': T5, 't5.1.1_encoder': T5_Encoder, 't5.1.1_decoder': T5_Decoder, 'mt5.1.1': T5, 'mt5.1.1_encoder': T5_Encoder, 'mt5.1.1_decoder': T5_Decoder, 'transformer_xl': Transformer_XL, 'xlnet': XLNET}
    if isinstance(model, str):
        MODEL = models[model.lower()]
        if model.endswith('t5.1.1'):
            configs['version'] = model
    elif isinstance(model, type) and issubclass(model, BERT_BASE):
        MODEL = model
    else:
        raise ValueError('"model" args type should be string or nn.Module')
    application = application.lower()
    if application in ['lm', 'unilm'] and model in ['electra', 't5']:
        raise ValueError(f'"{model}" model can not be used as "{application}" application.\n')
    if application == 'lm':
        MODEL = extend_with_language_model(MODEL)
    elif application == 'unilm':
        MODEL = extend_with_unified_language_model(MODEL)
    if configs.get('add_trainer', False):


        class MyModel(MODEL, BaseModel):
            pass
        MODEL = MyModel
    transformer = MODEL(**configs)
    transformer.build(**configs)
    transformer.apply(transformer.init_model_weights)
    if checkpoint_path is not None:
        transformer.load_weights_from_pytorch_checkpoint(checkpoint_path)
    transformer.configs = DottableDict(configs)
    return transformer


checkpoint_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/pytorch_model.bin'


config_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/bert_config.json'


def get_pool_emb(hidden_state=None, pooler=None, attention_mask=None, pool_strategy='cls', custom_layer=None):
    """ 获取句向量

    :param hidden_state: torch.Tensor/List(torch.Tensor)，last_hidden_state/all_encoded_layers
    :param pooler: torch.Tensor, bert的pool_output输出
    :param attention_mask: torch.Tensor
    :param pool_strategy: str, ('cls', 'last-avg', 'mean', 'last-max', 'max', 'first-last-avg', 'custom')
    :param custom_layer: int/List[int]，指定对某几层做average pooling
    """
    if pool_strategy == 'pooler':
        return pooler
    elif pool_strategy == 'cls':
        if isinstance(hidden_state, (list, tuple)):
            hidden_state = hidden_state[-1]
        assert isinstance(hidden_state, torch.Tensor), f'{pool_strategy} strategy request tensor hidden_state'
        return hidden_state[:, 0]
    elif pool_strategy in {'last-avg', 'mean'}:
        if isinstance(hidden_state, (list, tuple)):
            hidden_state = hidden_state[-1]
        assert isinstance(hidden_state, torch.Tensor), f'{pool_strategy} pooling strategy request tensor hidden_state'
        hid = torch.sum(hidden_state * attention_mask[:, :, None], dim=1)
        attention_mask = torch.sum(attention_mask, dim=1)[:, None]
        return hid / attention_mask
    elif pool_strategy in {'last-max', 'max'}:
        if isinstance(hidden_state, (list, tuple)):
            hidden_state = hidden_state[-1]
        assert isinstance(hidden_state, torch.Tensor), f'{pool_strategy} pooling strategy request tensor hidden_state'
        hid = hidden_state * attention_mask[:, :, None]
        return torch.max(hid, dim=1)
    elif pool_strategy == 'first-last-avg':
        assert isinstance(hidden_state, list), f'{pool_strategy} pooling strategy request list hidden_state'
        hid = torch.sum(hidden_state[1] * attention_mask[:, :, None], dim=1)
        hid += torch.sum(hidden_state[-1] * attention_mask[:, :, None], dim=1)
        attention_mask = torch.sum(attention_mask, dim=1)[:, None]
        return hid / (2 * attention_mask)
    elif pool_strategy == 'custom':
        assert isinstance(hidden_state, list), f'{pool_strategy} pooling strategy request list hidden_state'
        assert isinstance(custom_layer, (int, list, tuple)), f'{pool_strategy} pooling strategy request int/list/tuple custom_layer'
        custom_layer = [custom_layer] if isinstance(custom_layer, int) else custom_layer
        hid = 0
        for i, layer in enumerate(custom_layer, start=1):
            hid += torch.sum(hidden_state[layer] * attention_mask[:, :, None], dim=1)
        attention_mask = torch.sum(attention_mask, dim=1)[:, None]
        return hid / (i * attention_mask)
    else:
        raise ValueError('pool_strategy illegal')


class Loss(nn.Module):

    def forward(self, y_pred, y_true):
        return model.mixup(nn.CrossEntropyLoss(), y_pred, y_true)


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, loss_weight, matcher):
        super().__init__()
        self.cost_relation = loss_weight['relation']
        self.cost_head = loss_weight['head_entity']
        self.cost_tail = loss_weight['tail_entity']
        self.matcher = matcher

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_rel_logits": Tensor of dim [batch_size, num_generated_triples, num_classes] with the classification logits
                 "{head, tail}_{start, end}_logits": Tensor of dim [batch_size, num_generated_triples, seq_len] with the predicted index logits
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_generated_triples, num_gold_triples)
        """
        bsz, num_generated_triples = outputs['pred_rel_logits'].shape[:2]
        pred_rel = outputs['pred_rel_logits'].flatten(0, 1).softmax(-1)
        gold_rel = torch.cat([v['relation'] for v in targets])
        pred_head_start = outputs['head_start_logits'].flatten(0, 1).softmax(-1)
        pred_head_end = outputs['head_end_logits'].flatten(0, 1).softmax(-1)
        pred_tail_start = outputs['tail_start_logits'].flatten(0, 1).softmax(-1)
        pred_tail_end = outputs['tail_end_logits'].flatten(0, 1).softmax(-1)
        gold_head_start = torch.cat([v['head_start_index'] for v in targets])
        gold_head_end = torch.cat([v['head_end_index'] for v in targets])
        gold_tail_start = torch.cat([v['tail_start_index'] for v in targets])
        gold_tail_end = torch.cat([v['tail_end_index'] for v in targets])
        if self.matcher == 'avg':
            cost = -self.cost_relation * pred_rel[:, gold_rel] - self.cost_head * 1 / 2 * (pred_head_start[:, gold_head_start] + pred_head_end[:, gold_head_end]) - self.cost_tail * 1 / 2 * (pred_tail_start[:, gold_tail_start] + pred_tail_end[:, gold_tail_end])
        elif self.matcher == 'min':
            cost = torch.cat([pred_head_start[:, gold_head_start].unsqueeze(1), pred_rel[:, gold_rel].unsqueeze(1), pred_head_end[:, gold_head_end].unsqueeze(1), pred_tail_start[:, gold_tail_start].unsqueeze(1), pred_tail_end[:, gold_tail_end].unsqueeze(1)], dim=1)
            cost = -torch.min(cost, dim=1)[0]
        else:
            raise ValueError('Wrong matcher')
        cost = cost.view(bsz, num_generated_triples, -1).cpu()
        num_gold_triples = [len(v['relation']) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost.split(num_gold_triples, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class DecoderLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttentionLayer(**config)
        self.crossattention = MultiHeadAttentionLayer(**config)
        self.ffc = PositionWiseFeedForward(**config)

    def forward(self, hidden_states, encoder_hidden_states, encoder_attention_mask):
        attention_output = self.attention(hidden_states)
        cross_attention_outputs = self.crossattention(attention_output, None, encoder_hidden_states, encoder_attention_mask)
        layer_output = self.ffc(cross_attention_outputs)
        return layer_output


class SetDecoder(nn.Module):

    def __init__(self, config, num_generated_triples, num_layers, num_classes):
        super().__init__()
        self.num_generated_triples = num_generated_triples
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(num_layers)])
        self.LayerNorm = nn.LayerNorm(config['hidden_size'], eps=config.get('layer_norm_eps', 1e-12))
        self.dropout = nn.Dropout(config['hidden_dropout_prob'])
        self.query_embed = nn.Embedding(num_generated_triples, config['hidden_size'])
        self.decoder2class = nn.Linear(config['hidden_size'], num_classes + 1)
        self.decoder2span = nn.Linear(config['hidden_size'], 4)
        self.head_start_metric_1 = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.head_end_metric_1 = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.tail_start_metric_1 = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.tail_end_metric_1 = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.head_start_metric_2 = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.head_end_metric_2 = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.tail_start_metric_2 = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.tail_end_metric_2 = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.head_start_metric_3 = nn.Linear(config['hidden_size'], 1, bias=False)
        self.head_end_metric_3 = nn.Linear(config['hidden_size'], 1, bias=False)
        self.tail_start_metric_3 = nn.Linear(config['hidden_size'], 1, bias=False)
        self.tail_end_metric_3 = nn.Linear(config['hidden_size'], 1, bias=False)
        torch.nn.init.orthogonal_(self.head_start_metric_1.weight, gain=1)
        torch.nn.init.orthogonal_(self.head_end_metric_1.weight, gain=1)
        torch.nn.init.orthogonal_(self.tail_start_metric_1.weight, gain=1)
        torch.nn.init.orthogonal_(self.tail_end_metric_1.weight, gain=1)
        torch.nn.init.orthogonal_(self.head_start_metric_2.weight, gain=1)
        torch.nn.init.orthogonal_(self.head_end_metric_2.weight, gain=1)
        torch.nn.init.orthogonal_(self.tail_start_metric_2.weight, gain=1)
        torch.nn.init.orthogonal_(self.tail_end_metric_2.weight, gain=1)
        torch.nn.init.orthogonal_(self.query_embed.weight, gain=1)

    def forward(self, encoder_hidden_states, encoder_attention_mask):
        bsz = encoder_hidden_states.size()[0]
        hidden_states = self.query_embed.weight.unsqueeze(0).repeat(bsz, 1, 1)
        hidden_states = self.dropout(self.LayerNorm(hidden_states))
        encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
        for i, layer_module in enumerate(self.layers):
            hidden_states = layer_module(hidden_states, encoder_hidden_states, encoder_extended_attention_mask)
        class_logits = self.decoder2class(hidden_states)
        head_start_logits = self.head_start_metric_3(torch.tanh(self.head_start_metric_1(hidden_states).unsqueeze(2) + self.head_start_metric_2(encoder_hidden_states).unsqueeze(1))).squeeze()
        head_end_logits = self.head_end_metric_3(torch.tanh(self.head_end_metric_1(hidden_states).unsqueeze(2) + self.head_end_metric_2(encoder_hidden_states).unsqueeze(1))).squeeze()
        tail_start_logits = self.tail_start_metric_3(torch.tanh(self.tail_start_metric_1(hidden_states).unsqueeze(2) + self.tail_start_metric_2(encoder_hidden_states).unsqueeze(1))).squeeze()
        tail_end_logits = self.tail_end_metric_3(torch.tanh(self.tail_end_metric_1(hidden_states).unsqueeze(2) + self.tail_end_metric_2(encoder_hidden_states).unsqueeze(1))).squeeze()
        return class_logits, head_start_logits, head_end_logits, tail_start_logits, tail_end_logits


class SetCriterion(nn.Module):
    """ Loss的计算
    """

    def __init__(self, num_classes, loss_weight, na_coef, losses, matcher):
        super().__init__()
        self.num_classes = num_classes
        self.loss_weight = loss_weight
        self.matcher = HungarianMatcher(loss_weight, matcher)
        self.losses = losses
        rel_weight = torch.ones(self.num_classes + 1, device=device)
        rel_weight[-1] = na_coef
        self.register_buffer('rel_weight', rel_weight)

    def forward(self, outputs, targets):
        indices = self.matcher(outputs, targets)
        losses = {}
        for loss in self.losses:
            if loss == 'entity' and self.empty_targets(targets):
                pass
            else:
                losses.update(self.get_loss(loss, outputs, targets, indices))
        losses = sum(losses[k] * self.loss_weight[k] for k in losses.keys() if k in self.loss_weight)
        return losses

    def relation_loss(self, outputs, targets, indices):
        src_logits = outputs['pred_rel_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['relation'][i] for t, (_, i) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        loss = F.cross_entropy(src_logits.flatten(0, 1), target_classes.flatten(0, 1), weight=self.rel_weight)
        losses = {'relation': loss}
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices):
        pred_rel_logits = outputs['pred_rel_logits']
        device = pred_rel_logits.device
        tgt_lengths = torch.as_tensor([len(v['labels']) for v in targets], device=device)
        card_pred = (pred_rel_logits.argmax(-1) != pred_rel_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for src, _ in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for _, tgt in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, **kwargs):
        loss_map = {'relation': self.relation_loss, 'cardinality': self.loss_cardinality, 'entity': self.entity_loss}
        return loss_map[loss](outputs, targets, indices, **kwargs)

    def entity_loss(self, outputs, targets, indices):
        """Compute the losses related to the position of head entity or tail entity
        """
        idx = self._get_src_permutation_idx(indices)
        selected_pred_head_start = outputs['head_start_logits'][idx]
        selected_pred_head_end = outputs['head_end_logits'][idx]
        selected_pred_tail_start = outputs['tail_start_logits'][idx]
        selected_pred_tail_end = outputs['tail_end_logits'][idx]
        target_head_start = torch.cat([t['head_start_index'][i] for t, (_, i) in zip(targets, indices)])
        target_head_end = torch.cat([t['head_end_index'][i] for t, (_, i) in zip(targets, indices)])
        target_tail_start = torch.cat([t['tail_start_index'][i] for t, (_, i) in zip(targets, indices)])
        target_tail_end = torch.cat([t['tail_end_index'][i] for t, (_, i) in zip(targets, indices)])
        head_start_loss = F.cross_entropy(selected_pred_head_start, target_head_start)
        head_end_loss = F.cross_entropy(selected_pred_head_end, target_head_end)
        tail_start_loss = F.cross_entropy(selected_pred_tail_start, target_tail_start)
        tail_end_loss = F.cross_entropy(selected_pred_tail_end, target_tail_end)
        losses = {'head_entity': 1 / 2 * (head_start_loss + head_end_loss), 'tail_entity': 1 / 2 * (tail_start_loss + tail_end_loss)}
        return losses

    @staticmethod
    def empty_targets(targets):
        flag = True
        for target in targets:
            if len(target['relation']) != 0:
                flag = False
                break
        return flag


class Attention(nn.Module):
    """注意力层。"""

    def __init__(self, hidden_size, **kwargs):
        self.hidden_size = hidden_size
        super().__init__(**kwargs)
        self.weight = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(self.hidden_size))
        self.query = nn.Linear(self.hidden_size, 1, bias=False)

    def forward(self, x, mask):
        """x: [btz, max_segment, hdsz]
        mask: [btz, max_segment, 1]
        """
        mask = mask.squeeze(2)
        key = self.weight(x) + self.bias
        outputs = self.query(key).squeeze(2)
        outputs -= 1e+32 * (1 - mask)
        attn_scores = F.softmax(outputs, dim=-1)
        attn_scores = attn_scores * mask
        attn_scores = attn_scores.reshape(-1, 1, attn_scores.shape[-1])
        outputs = torch.matmul(attn_scores, key).squeeze(1)
        return outputs


class Myloss(nn.Module):

    def forward(self, y_pred, y_true):
        y_pred = torch.log(torch.softmax(y_pred, dim=-1)) * y_true
        return -y_pred.sum() / len(y_pred)


class ProjectionMLP(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()
        in_dim = hidden_size
        hidden_dim = hidden_size * 2
        out_dim = hidden_size
        affine = False
        list_layers = [nn.Linear(in_dim, hidden_dim, bias=False), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True)]
        list_layers += [nn.Linear(hidden_dim, out_dim, bias=False), nn.BatchNorm1d(out_dim, affine=affine)]
        self.net = nn.Sequential(*list_layers)

    def forward(self, x):
        return self.net(x)


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)
        self.record = None
        self.pos_avg = 0.0
        self.neg_avg = 0.0

    def forward(self, x, y):
        sim = self.cos(x, y)
        self.record = sim.detach()
        min_size = min(self.record.shape[0], self.record.shape[1])
        num_item = self.record.shape[0] * self.record.shape[1]
        self.pos_avg = self.record.diag().sum() / min_size
        self.neg_avg = (self.record.sum() - self.record.diag().sum()) / (num_item - min_size)
        return sim / self.temp


class MaskConv2d(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, groups=1):
        super(MaskConv2d, self).__init__()
        self.conv2d = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False, groups=groups)

    def forward(self, x, mask):
        x = x.masked_fill(mask, 0)
        _x = self.conv2d(x)
        return _x


class MaskCNN(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size=3, depth=3):
        super(MaskCNN, self).__init__()
        layers = []
        for _ in range(depth):
            layers.extend([MaskConv2d(input_channels, input_channels, kernel_size=kernel_size, padding=kernel_size // 2), LayerNorm((1, input_channels, 1, 1), dim_index=1), nn.GELU()])
        layers.append(MaskConv2d(input_channels, output_channels, kernel_size=3, padding=3 // 2))
        self.cnns = nn.ModuleList(layers)

    def forward(self, x, mask):
        _x = x
        for layer in self.cnns:
            if isinstance(layer, LayerNorm):
                x = x + _x
                x = layer(x)
                _x = x
            elif not isinstance(layer, nn.GELU):
                x = layer(x, mask)
            else:
                x = layer(x)
        return _x


class MultiHeadBiaffine(nn.Module):

    def __init__(self, dim, out=None, n_head=4):
        super(MultiHeadBiaffine, self).__init__()
        assert dim % n_head == 0
        in_head_dim = dim // n_head
        out = dim if out is None else out
        assert out % n_head == 0
        out_head_dim = out // n_head
        self.n_head = n_head
        self.W = nn.Parameter(nn.init.xavier_normal_(torch.randn(self.n_head, out_head_dim, in_head_dim, in_head_dim)))
        self.out_dim = out

    def forward(self, h, v):
        """
        :param h: bsz x max_len x dim
        :param v: bsz x max_len x dim
        :return: bsz x max_len x max_len x out_dim
        """
        bsz, max_len, dim = h.size()
        h = h.reshape(bsz, max_len, self.n_head, -1)
        v = v.reshape(bsz, max_len, self.n_head, -1)
        w = torch.einsum('blhx,hdxy,bkhy->bhdlk', h, self.W, v)
        w = w.reshape(bsz, self.out_dim, max_len, max_len)
        return w


class ConvolutionLayer(nn.Module):
    """卷积层
    """

    def __init__(self, input_size, channels, dilation, dropout=0.1):
        super(ConvolutionLayer, self).__init__()
        self.base = nn.Sequential(nn.Dropout2d(dropout), nn.Conv2d(input_size, channels, kernel_size=1), nn.GELU())
        self.convs = nn.ModuleList([nn.Conv2d(channels, channels, kernel_size=3, groups=channels, dilation=d, padding=d) for d in dilation])

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.base(x)
        outputs = []
        for conv in self.convs:
            x = conv(x)
            x = F.gelu(x)
            outputs.append(x)
        outputs = torch.cat(outputs, dim=1)
        outputs = outputs.permute(0, 2, 3, 1).contiguous()
        return outputs


class Biaffine(nn.Module):
    """仿射变换
    """

    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True):
        super(Biaffine, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        weight = torch.zeros((n_out, n_in + int(bias_x), n_in + int(bias_y)))
        nn.init.xavier_normal_(weight)
        self.weight = nn.Parameter(weight, requires_grad=True)

    def extra_repr(self):
        s = f'n_in={self.n_in}, n_out={self.n_out}'
        if self.bias_x:
            s += f', bias_x={self.bias_x}'
        if self.bias_y:
            s += f', bias_y={self.bias_y}'
        return s

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        s = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y)
        s = s.permute(0, 2, 3, 1)
        return s


class MLP(nn.Module):
    """MLP全连接
    """

    def __init__(self, n_in, n_out, dropout=0):
        super().__init__()
        self.linear = nn.Linear(n_in, n_out)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear(x)
        x = self.activation(x)
        return x


class CoPredictor(nn.Module):

    def __init__(self, cls_num, hid_size, biaffine_size, channels, ffnn_hid_size, dropout=0):
        super().__init__()
        self.mlp1 = MLP(n_in=hid_size, n_out=biaffine_size, dropout=dropout)
        self.mlp2 = MLP(n_in=hid_size, n_out=biaffine_size, dropout=dropout)
        self.biaffine = Biaffine(n_in=biaffine_size, n_out=cls_num, bias_x=True, bias_y=True)
        self.mlp_rel = MLP(channels, ffnn_hid_size, dropout=dropout)
        self.linear = nn.Linear(ffnn_hid_size, cls_num)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y, z):
        h = self.dropout(self.mlp1(x))
        t = self.dropout(self.mlp2(y))
        o1 = self.biaffine(h, t)
        z = self.dropout(self.mlp_rel(z))
        o2 = self.linear(z)
        return o1 + o2


class LabelFusionForToken(nn.Module):

    def __init__(self, hidden_size):
        super(LabelFusionForToken, self).__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.linear2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.output = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, token_features, label_features, label_mask=None):
        bs, seqlen = token_features.shape[:2]
        token_features = self.linear1(token_features)
        label_features = self.linear2(label_features)
        scores = torch.einsum('bmh, cnh->bmcn', token_features, label_features)
        scores += (1.0 - label_mask[None, None, ...]) * -10000.0
        scores = torch.softmax(scores, dim=-1)
        weighted_label_features = label_features[None, None, ...].repeat(bs, seqlen, 1, 1, 1) * scores[..., None]
        fused_features = token_features.unsqueeze(2) + weighted_label_features.sum(-2)
        return torch.tanh(self.output(fused_features))


class Classifier(nn.Module):

    def __init__(self, hidden_size, num_labels):
        super(Classifier, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(num_labels, hidden_size))
        self.bias = nn.Parameter(torch.Tensor(num_labels))
        self.reset_parameters()

    def reset_parameters(self) ->None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        x = torch.mul(input, self.weight)
        x = torch.sum(x, -1)
        return x + self.bias


class MLPForMultiLabel(nn.Module):

    def __init__(self, hidden_size, num_labels, dropout_rate=0.2):
        super(MLPForMultiLabel, self).__init__()
        self.classifier1 = nn.Linear(hidden_size, hidden_size)
        self.classifier2 = Classifier(hidden_size, num_labels)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, features):
        features = self.classifier1(features)
        features = self.dropout(F.gelu(features))
        return self.classifier2(features)


nested = False


class SpanLossForMultiLabelLoss(nn.Module):

    def __init__(self, name='Span Binary Cross Entropy Loss'):
        super().__init__()
        self.name = name
        self.loss_fct = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, preds, target):
        if not nested:
            return self.flated_forward(preds, target)
        start_logits, end_logits, span_logits = preds
        masks, start_labels, end_labels, span_labels = target
        start_, end_ = start_logits > 0, end_logits > 0
        bs, seqlen, num_labels = start_logits.shape
        span_candidate = torch.logical_or(start_.unsqueeze(-2).expand(-1, -1, seqlen, -1) & end_.unsqueeze(-3).expand(-1, seqlen, -1, -1), start_labels.unsqueeze(-2).expand(-1, -1, seqlen, -1).bool() & end_labels.unsqueeze(-3).expand(-1, seqlen, -1, -1).bool())
        masks = masks[:, :, None].expand(-1, -1, num_labels)
        start_loss = self.loss_fct(start_logits.view(-1), start_labels.view(-1).float())
        start_loss = (start_loss * masks.reshape(-1)).view(-1, num_labels).sum(-1).sum() / (masks.sum() / num_labels)
        end_loss = self.loss_fct(end_logits.view(-1), end_labels.view(-1).float())
        end_loss = (end_loss * masks.reshape(-1)).view(-1, num_labels).sum(-1).sum() / (masks.sum() / num_labels)
        span_masks = masks.bool().unsqueeze(2).expand(-1, -1, seqlen, -1) & masks.bool().unsqueeze(1).expand(-1, seqlen, -1, -1)
        span_masks = torch.triu(span_masks.permute(0, 3, 1, 2), 0).permute(0, 2, 3, 1) * span_candidate
        span_loss = self.loss_fct(span_logits.view(bs, -1), span_labels.view(bs, -1).float())
        span_loss = span_loss.reshape(-1, num_labels).sum(-1).sum() / (span_masks.view(-1, num_labels).sum() / num_labels)
        return start_loss + end_loss + span_loss

    def flated_forward(self, preds, target):
        start_logits, end_logits, _ = preds
        masks, start_labels, end_labels, _ = target
        active_loss = masks.view(-1) == 1
        active_start_logits = start_logits.view(-1, start_logits.size(-1))[active_loss]
        active_end_logits = end_logits.view(-1, start_logits.size(-1))[active_loss]
        active_start_labels = start_labels.view(-1, start_labels.size(-1))[active_loss].float()
        active_end_labels = end_labels.view(-1, end_labels.size(-1))[active_loss].float()
        start_loss = self.loss_fct(active_start_logits, active_start_labels).sum(1).mean()
        end_loss = self.loss_fct(active_end_logits, active_end_labels).sum(1).mean()
        return start_loss + end_loss


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Biaffine,
     lambda: ([], {'n_in': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     True),
    (Classifier,
     lambda: ([], {'hidden_size': 4, 'num_labels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CoPredictor,
     lambda: ([], {'cls_num': 4, 'hid_size': 4, 'biaffine_size': 4, 'channels': 4, 'ffnn_hid_size': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (ContrastiveLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (EfficientGlobalPointer,
     lambda: ([], {'hidden_size': 4, 'heads': 4, 'head_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (GlobalPointer,
     lambda: ([], {'hidden_size': 4, 'heads': 4, 'head_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (LayerNorm,
     lambda: ([], {'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MLP,
     lambda: ([], {'n_in': 4, 'n_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MLPForMultiLabel,
     lambda: ([], {'hidden_size': 4, 'num_labels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MultiHeadAttentionLayer,
     lambda: ([], {'hidden_size': 4, 'num_attention_heads': 4, 'attention_probs_dropout_prob': 0.5}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (MultiHeadBiaffine,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (MultiNonLinearClassifier,
     lambda: ([], {'hidden_size': 4, 'tag_size': 4, 'dropout_rate': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MultiSampleDropout,
     lambda: ([], {'hidden_size': 4, 'num_labels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MultilabelCategoricalCrossentropy,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (MyLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Myloss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (ProjectionMLP,
     lambda: ([], {'hidden_size': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (RelativePositionsEncoding,
     lambda: ([], {'qlen': 4, 'klen': 4, 'embedding_size': 4}),
     lambda: ([0, 0], {}),
     True),
    (RelativePositionsEncodingDebertaV2,
     lambda: ([], {'qlen': 4, 'klen': 4, 'position_buckets': 4, 'max_position': 4}),
     lambda: ([0, 0], {}),
     True),
    (RoPEPositionEncoding,
     lambda: ([], {'max_position': 4, 'embedding_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SequenceLabelForSO,
     lambda: ([], {'hidden_size': 4, 'tag_size': 4, 'dropout_rate': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Similarity,
     lambda: ([], {'temp': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     True),
    (SparseMultilabelCategoricalCrossentropy,
     lambda: ([], {}),
     lambda: ([torch.ones([4], dtype=torch.int64), torch.ones([4], dtype=torch.int64)], {}),
     False),
    (TemporalEnsemblingLoss,
     lambda: ([], {'epochs': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), 0, 0], {}),
     False),
    (UDALoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_Tongjilibo_bert4torch(_paritybench_base):
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

    def test_005(self):
        self._check(*TESTCASES[5])

    def test_006(self):
        self._check(*TESTCASES[6])

    def test_007(self):
        self._check(*TESTCASES[7])

    def test_008(self):
        self._check(*TESTCASES[8])

    def test_009(self):
        self._check(*TESTCASES[9])

    def test_010(self):
        self._check(*TESTCASES[10])

    def test_011(self):
        self._check(*TESTCASES[11])

    def test_012(self):
        self._check(*TESTCASES[12])

    def test_013(self):
        self._check(*TESTCASES[13])

    def test_014(self):
        self._check(*TESTCASES[14])

    def test_015(self):
        self._check(*TESTCASES[15])

    def test_016(self):
        self._check(*TESTCASES[16])

    def test_017(self):
        self._check(*TESTCASES[17])

    def test_018(self):
        self._check(*TESTCASES[18])

    def test_019(self):
        self._check(*TESTCASES[19])

    def test_020(self):
        self._check(*TESTCASES[20])

    def test_021(self):
        self._check(*TESTCASES[21])

    def test_022(self):
        self._check(*TESTCASES[22])

    def test_023(self):
        self._check(*TESTCASES[23])

    def test_024(self):
        self._check(*TESTCASES[24])

