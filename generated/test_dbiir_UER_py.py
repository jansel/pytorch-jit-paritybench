import sys
_module = sys.modules[__name__]
del sys
run_c3 = _module
run_chid = _module
run_classifier = _module
run_classifier_cv = _module
run_classifier_deepspeed = _module
run_classifier_grid = _module
run_classifier_mt = _module
run_classifier_multi_label = _module
run_classifier_prompt = _module
run_classifier_siamese = _module
run_cmrc = _module
run_dbqa = _module
run_ner = _module
run_regression = _module
run_simcse = _module
run_text2text = _module
run_c3_infer = _module
run_chid_infer = _module
run_classifier_cv_infer = _module
run_classifier_deepspeed_infer = _module
run_classifier_infer = _module
run_classifier_multi_label_infer = _module
run_classifier_prompt_infer = _module
run_classifier_siamese_infer = _module
run_cmrc_infer = _module
run_ner_infer = _module
run_regression_infer = _module
run_text2text_infer = _module
preprocess = _module
pretrain = _module
scripts = _module
average_models = _module
build_vocab = _module
cloze_test = _module
convert_albert_from_huggingface_to_uer = _module
convert_albert_from_original_tf_to_uer = _module
convert_albert_from_uer_to_huggingface = _module
convert_albert_from_uer_to_original_tf = _module
convert_bart_from_huggingface_to_uer = _module
convert_bart_from_uer_to_huggingface = _module
convert_bert_extractive_qa_from_huggingface_to_uer = _module
convert_bert_extractive_qa_from_uer_to_huggingface = _module
convert_bert_from_huggingface_to_uer = _module
convert_bert_from_original_tf_to_uer = _module
convert_bert_from_uer_to_huggingface = _module
convert_bert_from_uer_to_original_tf = _module
convert_bert_text_classification_from_huggingface_to_uer = _module
convert_bert_text_classification_from_uer_to_huggingface = _module
convert_bert_token_classification_from_huggingface_to_uer = _module
convert_bert_token_classification_from_uer_to_huggingface = _module
convert_gpt2_from_huggingface_to_uer = _module
convert_gpt2_from_uer_to_huggingface = _module
convert_pegasus_from_huggingface_to_uer = _module
convert_pegasus_from_uer_to_huggingface = _module
convert_sbert_from_huggingface_to_uer = _module
convert_sbert_from_uer_to_huggingface = _module
convert_t5_from_huggingface_to_uer = _module
convert_t5_from_uer_to_huggingface = _module
convert_xlmroberta_from_huggingface_to_uer = _module
convert_xlmroberta_from_uer_to_huggingface = _module
diff_vocab = _module
dynamic_vocab_adapter = _module
extract_embeddings = _module
extract_features = _module
generate_lm = _module
generate_lm_deepspeed = _module
generate_seq2seq = _module
generate_seq2seq_deepspeed = _module
run_lgb = _module
run_lgb_cv_bayesopt = _module
topn_words_dep = _module
topn_words_indep = _module
uer = _module
decoders = _module
transformer_decoder = _module
embeddings = _module
dual_embedding = _module
word_embedding = _module
wordpos_embedding = _module
wordposseg_embedding = _module
wordsinusoidalpos_embedding = _module
encoders = _module
cnn_encoder = _module
dual_encoder = _module
rnn_encoder = _module
transformer_encoder = _module
layers = _module
layer_norm = _module
multi_headed_attn = _module
position_ffn = _module
relative_position_embedding = _module
transformer = _module
model_builder = _module
model_loader = _module
model_saver = _module
models = _module
model = _module
opts = _module
targets = _module
bilm_target = _module
cls_target = _module
lm_target = _module
mlm_target = _module
sp_target = _module
target = _module
trainer = _module
utils = _module
act_fun = _module
adversarial = _module
config = _module
constants = _module
dataloader = _module
dataset = _module
logging = _module
mask = _module
misc = _module
optimizers = _module
seed = _module
tokenizers = _module
vocab = _module

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


import torch


import torch.nn as nn


import numpy as np


import torch.distributed as dist


from itertools import product


import time


import re


import logging


import collections


import torch.nn.functional as F


from scipy.stats import spearmanr


import math


import scipy.stats


import tensorflow as tf


from tensorflow.python import pywrap_tensorflow


import tensorflow.keras.backend as K


import copy


import torch.multiprocessing as mp


from torch.nn.parallel import DistributedDataParallel


from typing import Callable


from typing import Iterable


from typing import Tuple


from torch.optim import Optimizer


from torch.optim.lr_scheduler import LambdaLR


class DualEmbedding(nn.Module):
    """
    """

    def __init__(self, args, vocab_size):
        super(DualEmbedding, self).__init__()
        stream_0_args = copy.deepcopy(vars(args))
        stream_0_args.update(args.stream_0)
        stream_0_args = Namespace(**stream_0_args)
        self.embedding_0 = str2embedding[stream_0_args.embedding](stream_0_args, vocab_size)
        stream_1_args = copy.deepcopy(vars(args))
        stream_1_args.update(args.stream_1)
        stream_1_args = Namespace(**stream_1_args)
        self.embedding_1 = str2embedding[stream_1_args.embedding](stream_1_args, vocab_size)
        self.dropout = nn.Dropout(args.dropout)
        if args.tie_weights:
            self.embedding_0 = self.embedding_1

    def forward(self, src, seg):
        """
        Args:
            src: ([batch_size x seq_length], [batch_size x seq_length])
            seg: ([batch_size x seq_length], [batch_size x seq_length])
        Returns:
            emb_0: [batch_size x seq_length x hidden_size]
            emb_1: [batch_size x seq_length x hidden_size]
        """
        emb_0 = self.get_embedding_0(src[0], seg[0])
        emb_1 = self.get_embedding_1(src[1], seg[1])
        emb_0 = self.dropout(emb_0)
        emb_1 = self.dropout(emb_1)
        return emb_0, emb_1

    def get_embedding_0(self, src, seg):
        return self.embedding_0(src, seg)

    def get_embedding_1(self, src, seg):
        return self.embedding_1(src, seg)


class LayerNorm(nn.Module):
    """
    Layer Normalization.
    https://arxiv.org/abs/1607.06450
    """

    def __init__(self, hidden_size, eps=1e-06):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        hidden_states = self.gamma * (x - mean) / (std + self.eps)
        return hidden_states + self.beta


class WordEmbedding(nn.Module):
    """
    """

    def __init__(self, args, vocab_size):
        super(WordEmbedding, self).__init__()
        self.remove_embedding_layernorm = args.remove_embedding_layernorm
        self.dropout = nn.Dropout(args.dropout)
        self.word_embedding = nn.Embedding(vocab_size, args.emb_size)
        if not self.remove_embedding_layernorm:
            self.layer_norm = LayerNorm(args.emb_size)

    def forward(self, src, _):
        """
        Args:
            src: [batch_size x seq_length]
            seg: [batch_size x seq_length]
        Returns:
            emb: [batch_size x seq_length x hidden_size]
        """
        emb = self.word_embedding(src)
        if not self.remove_embedding_layernorm:
            emb = self.layer_norm(emb)
        emb = self.dropout(emb)
        return emb


class WordPosEmbedding(WordEmbedding):
    """
    GPT embedding consists of two parts:
    word embedding and position embedding.
    """

    def __init__(self, args, vocab_size):
        super(WordPosEmbedding, self).__init__(args, vocab_size)
        self.max_seq_length = args.max_seq_length
        self.position_embedding = nn.Embedding(self.max_seq_length, args.emb_size)

    def forward(self, src, _):
        """
        Args:
            src: [batch_size x seq_length]
            seg: [batch_size x seq_length]
        Returns:
            emb: [batch_size x seq_length x hidden_size]
        """
        word_emb = self.word_embedding(src)
        pos_emb = self.position_embedding(torch.arange(0, word_emb.size(1), device=word_emb.device, dtype=torch.long).unsqueeze(0).repeat(word_emb.size(0), 1))
        emb = word_emb + pos_emb
        if not self.remove_embedding_layernorm:
            emb = self.layer_norm(emb)
        emb = self.dropout(emb)
        return emb


class WordPosSegEmbedding(WordPosEmbedding):
    """
    BERT embedding consists of three parts:
    word embedding, position embedding, and segment embedding.
    """

    def __init__(self, args, vocab_size):
        super(WordPosSegEmbedding, self).__init__(args, vocab_size)
        self.segment_embedding = nn.Embedding(3, args.emb_size)

    def forward(self, src, seg):
        """
        Args:
            src: [batch_size x seq_length]
            seg: [batch_size x seq_length]
        Returns:
            emb: [batch_size x seq_length x hidden_size]
        """
        word_emb = self.word_embedding(src)
        pos_emb = self.position_embedding(torch.arange(0, word_emb.size(1), device=word_emb.device, dtype=torch.long).unsqueeze(0).repeat(word_emb.size(0), 1))
        seg_emb = self.segment_embedding(seg)
        emb = word_emb + pos_emb + seg_emb
        if not self.remove_embedding_layernorm:
            emb = self.layer_norm(emb)
        emb = self.dropout(emb)
        return emb


class WordSinusoidalposEmbedding(nn.Module):
    """Sinusoidal positional encoding for non-recurrent neural networks.
    Implementation based on "Attention Is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`
    Args:
       dropout (float): dropout parameter
       dim (int): embedding size
    """

    def __init__(self, args, vocab_size):
        super(WordSinusoidalposEmbedding, self).__init__()
        if args.emb_size % 2 != 0:
            raise ValueError('Cannot use sin/cos positional encoding with odd dim (got dim={:d})'.format(args.emb_size))
        self.max_seq_length = args.max_seq_length
        pe = torch.zeros(self.max_seq_length, args.emb_size)
        position = torch.arange(0, self.max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, args.emb_size, 2, dtype=torch.float) * -(math.log(10000.0) / args.emb_size))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)
        self.word_embedding = nn.Embedding(vocab_size, args.emb_size)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, src, _):
        """Embed inputs.
        Args:
            emb (FloatTensor): Sequence of word vectors
                ``(batch_size, seq_len, self.dim)``
            step (int or NoneType): If stepwise (``seq_len = 1``), use
                the encoding for this position.
        """
        word_emb = self.word_embedding(src)
        emb = word_emb * math.sqrt(word_emb.size(-1))
        emb = emb + self.pe[:emb.size(1)].transpose(0, 1)
        emb = self.dropout(emb)
        return emb


str2embedding = {'word': WordEmbedding, 'word_pos': WordPosEmbedding, 'word_pos_seg': WordPosSegEmbedding, 'word_sinusoidalpos': WordSinusoidalposEmbedding, 'dual': DualEmbedding}


def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
    return x[tuple(indices)]


class BirnnEncoder(nn.Module):
    """
    Bi-directional RNN encoder.
    """

    def __init__(self, args):
        super(BirnnEncoder, self).__init__()
        assert args.hidden_size % 2 == 0
        self.hidden_size = args.hidden_size // 2
        self.layers_num = args.layers_num
        self.rnn_forward = nn.RNN(input_size=args.emb_size, hidden_size=self.hidden_size, num_layers=args.layers_num, dropout=args.dropout, batch_first=True)
        self.rnn_backward = nn.RNN(input_size=args.emb_size, hidden_size=self.hidden_size, num_layers=args.layers_num, dropout=args.dropout, batch_first=True)
        self.drop = nn.Dropout(args.dropout)

    def forward(self, emb, _):
        self.rnn_forward.flatten_parameters()
        emb_forward = emb
        hidden_forward = self.init_hidden(emb_forward.size(0), emb_forward.device)
        output_forward, hidden_forward = self.rnn_forward(emb_forward, hidden_forward)
        output_forward = self.drop(output_forward)
        self.rnn_backward.flatten_parameters()
        emb_backward = flip(emb, 1)
        hidden_backward = self.init_hidden(emb_backward.size(0), emb_backward.device)
        output_backward, hidden_backward = self.rnn_backward(emb_backward, hidden_backward)
        output_backward = self.drop(output_backward)
        output_backward = flip(output_backward, 1)
        return torch.cat([output_forward, output_backward], 2)

    def init_hidden(self, batch_size, device):
        return torch.zeros(self.layers_num, batch_size, self.hidden_size, device=device)


class BigruEncoder(BirnnEncoder):
    """
     Bi-directional GRU encoder.
     """

    def __init__(self, args):
        super(BigruEncoder, self).__init__(args)
        self.rnn_forward = nn.GRU(input_size=args.emb_size, hidden_size=self.hidden_size, num_layers=args.layers_num, dropout=args.dropout, batch_first=True)
        self.rnn_backward = nn.GRU(input_size=args.emb_size, hidden_size=self.hidden_size, num_layers=args.layers_num, dropout=args.dropout, batch_first=True)


class BilstmEncoder(BirnnEncoder):
    """
     Bi-directional LSTM encoder.
     """

    def __init__(self, args):
        super(BilstmEncoder, self).__init__(args)
        self.rnn_forward = nn.LSTM(input_size=args.emb_size, hidden_size=self.hidden_size, num_layers=args.layers_num, dropout=args.dropout, batch_first=True)
        self.rnn_backward = nn.LSTM(input_size=args.emb_size, hidden_size=self.hidden_size, num_layers=args.layers_num, dropout=args.dropout, batch_first=True)

    def init_hidden(self, batch_size, device):
        return torch.zeros(self.layers_num, batch_size, self.hidden_size, device=device), torch.zeros(self.layers_num, batch_size, self.hidden_size, device=device)


class DualEncoder(nn.Module):
    """
    Dual Encoder which enables siamese models like SBER and CLIP.
    """

    def __init__(self, args):
        super(DualEncoder, self).__init__()
        stream_0_args = copy.deepcopy(vars(args))
        stream_0_args.update(args.stream_0)
        stream_0_args = Namespace(**stream_0_args)
        self.encoder_0 = str2encoder[stream_0_args.encoder](stream_0_args)
        stream_1_args = copy.deepcopy(vars(args))
        stream_1_args.update(args.stream_1)
        stream_1_args = Namespace(**stream_1_args)
        self.encoder_1 = str2encoder[stream_1_args.encoder](stream_1_args)
        if args.tie_weights:
            self.encoder_1 = self.encoder_0

    def forward(self, emb, seg):
        """
        Args:
            emb: ([batch_size x seq_length x emb_size], [batch_size x seq_length x emb_size])
            seg: ([batch_size x seq_length], [batch_size x seq_length])
        Returns:
            features_0: [batch_size x seq_length x hidden_size]
            features_1: [batch_size x seq_length x hidden_size]
        """
        features_0 = self.get_encode_0(emb[0], seg[0])
        features_1 = self.get_encode_1(emb[1], seg[1])
        return features_0, features_1

    def get_encode_0(self, emb, seg):
        features = self.encoder_0(emb, seg)
        return features

    def get_encode_1(self, emb, seg):
        features = self.encoder_1(emb, seg)
        return features


class GatedcnnEncoder(nn.Module):
    """
    Gated CNN encoder.
    """

    def __init__(self, args):
        super(GatedcnnEncoder, self).__init__()
        self.layers_num = args.layers_num
        self.kernel_size = args.kernel_size
        self.block_size = args.block_size
        self.emb_size = args.emb_size
        self.hidden_size = args.hidden_size
        self.conv_1 = nn.Conv2d(1, args.hidden_size, (args.kernel_size, args.emb_size))
        self.gate_1 = nn.Conv2d(1, args.hidden_size, (args.kernel_size, args.emb_size))
        self.conv_b1 = nn.Parameter(torch.randn(1, args.hidden_size, 1, 1))
        self.gate_b1 = nn.Parameter(torch.randn(1, args.hidden_size, 1, 1))
        self.conv = nn.ModuleList([nn.Conv2d(args.hidden_size, args.hidden_size, (args.kernel_size, 1)) for _ in range(args.layers_num - 1)])
        self.gate = nn.ModuleList([nn.Conv2d(args.hidden_size, args.hidden_size, (args.kernel_size, 1)) for _ in range(args.layers_num - 1)])
        self.conv_b = nn.ParameterList(nn.Parameter(torch.randn(1, args.hidden_size, 1, 1)) for _ in range(args.layers_num - 1))
        self.gate_b = nn.ParameterList(nn.Parameter(torch.randn(1, args.hidden_size, 1, 1)) for _ in range(args.layers_num - 1))

    def forward(self, emb, seg):
        batch_size, seq_length, _ = emb.size()
        padding = torch.zeros([batch_size, self.kernel_size - 1, self.emb_size])
        emb = torch.cat([padding, emb], dim=1).unsqueeze(1)
        hidden = self.conv_1(emb)
        hidden += self.conv_b1.repeat(1, 1, seq_length, 1)
        gate = self.gate_1(emb)
        gate += self.gate_b1.repeat(1, 1, seq_length, 1)
        hidden = hidden * torch.sigmoid(gate)
        res_input = hidden
        padding = torch.zeros([batch_size, self.hidden_size, self.kernel_size - 1, 1])
        hidden = torch.cat([padding, hidden], dim=2)
        for i, (conv_i, gate_i) in enumerate(zip(self.conv, self.gate)):
            hidden, gate = conv_i(hidden), gate_i(hidden)
            hidden += self.conv_b[i].repeat(1, 1, seq_length, 1)
            gate += self.gate_b[i].repeat(1, 1, seq_length, 1)
            hidden = hidden * torch.sigmoid(gate)
            if (i + 1) % self.block_size == 0:
                hidden = hidden + res_input
                res_input = hidden
            hidden = torch.cat([padding, hidden], dim=2)
        hidden = hidden[:, :, self.kernel_size - 1:, :]
        output = hidden.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_size)
        return output


class RnnEncoder(nn.Module):
    """
    RNN encoder.
    """

    def __init__(self, args):
        super(RnnEncoder, self).__init__()
        self.bidirectional = args.bidirectional
        if self.bidirectional:
            assert args.hidden_size % 2 == 0
            self.hidden_size = args.hidden_size // 2
        else:
            self.hidden_size = args.hidden_size
        self.layers_num = args.layers_num
        self.rnn = nn.RNN(input_size=args.emb_size, hidden_size=self.hidden_size, num_layers=args.layers_num, dropout=args.dropout, batch_first=True, bidirectional=self.bidirectional)
        self.drop = nn.Dropout(args.dropout)

    def forward(self, emb, _):
        self.rnn.flatten_parameters()
        hidden = self.init_hidden(emb.size(0), emb.device)
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        return output

    def init_hidden(self, batch_size, device):
        if self.bidirectional:
            return torch.zeros(self.layers_num * 2, batch_size, self.hidden_size, device=device)
        else:
            return torch.zeros(self.layers_num, batch_size, self.hidden_size, device=device)


class GruEncoder(RnnEncoder):
    """
    GRU encoder.
    """

    def __init__(self, args):
        super(GruEncoder, self).__init__(args)
        self.rnn = nn.GRU(input_size=args.emb_size, hidden_size=self.hidden_size, num_layers=args.layers_num, dropout=args.dropout, batch_first=True, bidirectional=self.bidirectional)


class LstmEncoder(RnnEncoder):
    """
    LSTM encoder.
    """

    def __init__(self, args):
        super(LstmEncoder, self).__init__(args)
        self.rnn = nn.LSTM(input_size=args.emb_size, hidden_size=self.hidden_size, num_layers=args.layers_num, dropout=args.dropout, batch_first=True, bidirectional=self.bidirectional)

    def init_hidden(self, batch_size, device):
        if self.bidirectional:
            return torch.zeros(self.layers_num * 2, batch_size, self.hidden_size, device=device), torch.zeros(self.layers_num * 2, batch_size, self.hidden_size, device=device)
        else:
            return torch.zeros(self.layers_num, batch_size, self.hidden_size, device=device), torch.zeros(self.layers_num, batch_size, self.hidden_size, device=device)


class RelativePositionEmbedding(nn.Module):
    """ Relative Position Embedding
        https://arxiv.org/abs/1910.10683
        https://github.com/bojone/bert4keras/blob/db236eac110a67a587df7660f6a1337d5b2ef07e/bert4keras/layers.py#L663
        https://github.com/huggingface/transformers/blob/master/src/transformers/models/t5/modeling_t5.py#L344
    """

    def __init__(self, heads_num, bidirectional=True, num_buckets=32, max_distance=128):
        super(RelativePositionEmbedding, self).__init__()
        self.num_buckets = num_buckets
        self.bidirectional = bidirectional
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(self.num_buckets, heads_num)

    def forward(self, encoder_hidden, decoder_hidden):
        """
        Compute binned relative position bias
        Args:
            encoder_hidden: [batch_size x seq_length x emb_size]
            decoder_hidden: [batch_size x seq_length x emb_size]
        Returns:
            position_bias: [1 x heads_num x seq_length x seq_length]
        """
        query_length = encoder_hidden.size()[1]
        key_length = decoder_hidden.size()[1]
        context_position = torch.arange(query_length, dtype=torch.long)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position
        relative_position_bucket = self.relative_position_bucket(relative_position, bidirectional=self.bidirectional, num_buckets=self.num_buckets, max_distance=self.max_distance)
        relative_position_bucket = relative_position_bucket
        values = self.relative_attention_bias(relative_position_bucket)
        values = values.permute([2, 0, 1]).unsqueeze(0)
        return values

    def relative_position_bucket(self, relative_position, bidirectional, num_buckets, max_distance):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593
        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on
        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer
        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        relative_postion_if_large = max_exact + torch.log(relative_position.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        relative_postion_if_large = torch.min(relative_postion_if_large, torch.full_like(relative_postion_if_large, num_buckets - 1))
        relative_buckets += torch.where(is_small, relative_position, relative_postion_if_large)
        return relative_buckets


class T5LayerNorm(nn.Module):
    """
    Construct a layernorm module in the T5 style No bias and no subtraction of mean.
    """

    def __init__(self, hidden_size, eps=1e-06):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.type_as(self.weight)


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def gelu_fast(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


def linear(x):
    return x


def relu(x):
    return F.relu(x)


class GatedFeedForward(nn.Module):
    """ Feed Forward Layer with Gated Linear Unit.
        https://arxiv.org/abs/2002.05202
    """

    def __init__(self, hidden_size, feedforward_size, hidden_act, has_bias=True):
        super(GatedFeedForward, self).__init__()
        self.linear_gate = nn.Linear(hidden_size, feedforward_size, bias=has_bias)
        self.linear_1 = nn.Linear(hidden_size, feedforward_size, bias=has_bias)
        self.linear_2 = nn.Linear(feedforward_size, hidden_size, bias=has_bias)
        self.act = str2act[hidden_act]

    def forward(self, x):
        gate = self.act(self.linear_gate(x))
        inter_linear = self.linear_1(x)
        inter = gate * inter_linear
        output = self.linear_2(inter)
        return output


class MultiHeadedAttention(nn.Module):
    """
    Each head is a self-attention operation.
    self-attention refers to https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self, hidden_size, heads_num, attention_head_size, dropout, has_bias=True, with_scale=True):
        super(MultiHeadedAttention, self).__init__()
        self.heads_num = heads_num
        self.per_head_size = attention_head_size
        self.with_scale = with_scale
        self.inner_hidden_size = heads_num * attention_head_size
        self.linear_layers = nn.ModuleList([nn.Linear(hidden_size, self.inner_hidden_size, bias=has_bias) for _ in range(3)])
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(self.inner_hidden_size, hidden_size, bias=has_bias)

    def forward(self, key, value, query, mask, position_bias=None, has_residual_attention=False, prev_attn=None):
        """
        Args:
            key: [batch_size x seq_length x hidden_size]
            value: [batch_size x seq_length x hidden_size]
            query: [batch_size x seq_length x hidden_size]
            mask: [batch_size x 1 x seq_length x seq_length]
            position_bias: [1 x heads_num x seq_length x seq_length]
        Returns:
            output: [batch_size x seq_length x hidden_size]
        """
        batch_size, seq_length, _ = query.size()
        heads_num = self.heads_num
        per_head_size = self.per_head_size

        def shape(x):
            return x.contiguous().view(batch_size, seq_length, heads_num, per_head_size).transpose(1, 2)

        def unshape(x):
            return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.inner_hidden_size)
        query, key, value = [l(x).view(batch_size, -1, heads_num, per_head_size).transpose(1, 2) for l, x in zip(self.linear_layers, (query, key, value))]
        scores = torch.matmul(query, key.transpose(-2, -1))
        if position_bias is not None:
            scores = scores + position_bias
        if self.with_scale:
            scores = scores / math.sqrt(float(per_head_size))
        scores = scores + mask.type_as(scores)
        prev_attn_out = None
        if has_residual_attention:
            if prev_attn is not None:
                scores += prev_attn
            prev_attn_out = scores
        probs = nn.Softmax(dim=-1)(scores)
        probs = self.dropout(probs)
        output = unshape(torch.matmul(probs, value))
        output = self.final_linear(output)
        return output, prev_attn_out


class PositionwiseFeedForward(nn.Module):
    """ Feed Forward Layer. """

    def __init__(self, hidden_size, feedforward_size, hidden_act, has_bias=True):
        super(PositionwiseFeedForward, self).__init__()
        self.linear_1 = nn.Linear(hidden_size, feedforward_size, bias=has_bias)
        self.linear_2 = nn.Linear(feedforward_size, hidden_size, bias=has_bias)
        self.act = str2act[hidden_act]

    def forward(self, x):
        inter = self.act(self.linear_1(x))
        output = self.linear_2(inter)
        return output


class TransformerLayer(nn.Module):
    """
    Transformer layer mainly consists of two parts:
    multi-headed self-attention and feed forward layer.
    """

    def __init__(self, args):
        super(TransformerLayer, self).__init__()
        self.layernorm_positioning = args.layernorm_positioning
        if hasattr(args, 'attention_head_size'):
            attention_head_size = args.attention_head_size
        else:
            attention_head_size = args.hidden_size // args.heads_num
        has_bias = bool(1 - args.remove_transformer_bias)
        with_scale = bool(1 - args.remove_attention_scale)
        self.self_attn = MultiHeadedAttention(args.hidden_size, args.heads_num, attention_head_size, args.dropout, has_bias=has_bias, with_scale=with_scale)
        self.dropout_1 = nn.Dropout(args.dropout)
        if args.feed_forward == 'gated':
            self.feed_forward = GatedFeedForward(args.hidden_size, args.feedforward_size, args.hidden_act, has_bias)
        else:
            self.feed_forward = PositionwiseFeedForward(args.hidden_size, args.feedforward_size, args.hidden_act, has_bias)
        self.dropout_2 = nn.Dropout(args.dropout)
        if args.layernorm == 't5':
            self.layer_norm_1 = T5LayerNorm(args.hidden_size)
            self.layer_norm_2 = T5LayerNorm(args.hidden_size)
        else:
            self.layer_norm_1 = LayerNorm(args.hidden_size)
            self.layer_norm_2 = LayerNorm(args.hidden_size)

    def forward(self, hidden, mask, position_bias=None, has_residual_attention=False, prev_attn=None):
        """
        Args:
            hidden: [batch_size x seq_length x emb_size]
            mask: [batch_size x 1 x seq_length x seq_length]
            position_bias: [1 x heads_num x seq_length x seq_length]
        Returns:
            output: [batch_size x seq_length x hidden_size]
        """
        if self.layernorm_positioning == 'post':
            inter, prev_attn_out = self.self_attn(hidden, hidden, hidden, mask, position_bias, has_residual_attention, prev_attn)
            inter = self.dropout_1(inter)
            inter = self.layer_norm_1(inter + hidden)
            output = self.dropout_2(self.feed_forward(inter))
            output = self.layer_norm_2(output + inter)
        else:
            inter = self.layer_norm_1(hidden)
            inter, prev_attn_out = self.self_attn(inter, inter, inter, mask, position_bias, has_residual_attention, prev_attn)
            inter = self.dropout_1(inter)
            hidden = hidden + inter
            output = self.layer_norm_2(hidden)
            output = self.dropout_2(self.feed_forward(output)) + hidden
        return output, prev_attn_out


class TransformerEncoder(nn.Module):
    """
    BERT encoder exploits 12 or 24 transformer layers to extract features.
    """

    def __init__(self, args):
        super(TransformerEncoder, self).__init__()
        self.mask = args.mask
        self.layers_num = args.layers_num
        self.parameter_sharing = args.parameter_sharing
        self.factorized_embedding_parameterization = args.factorized_embedding_parameterization
        self.layernorm_positioning = args.layernorm_positioning
        self.relative_position_embedding = args.relative_position_embedding
        self.has_residual_attention = args.has_residual_attention
        if 'deepspeed_checkpoint_activations' in args:
            self.deepspeed_checkpoint_activations = args.deepspeed_checkpoint_activations
            self.deepspeed_checkpoint_layers_num = args.deepspeed_checkpoint_layers_num
        else:
            self.deepspeed_checkpoint_activations = False
        has_bias = bool(1 - args.remove_transformer_bias)
        if self.factorized_embedding_parameterization:
            self.linear = nn.Linear(args.emb_size, args.hidden_size)
        if self.parameter_sharing:
            self.transformer = TransformerLayer(args)
        else:
            self.transformer = nn.ModuleList([TransformerLayer(args) for _ in range(self.layers_num)])
        if self.layernorm_positioning == 'pre':
            if args.layernorm == 't5':
                self.layer_norm = T5LayerNorm(args.hidden_size)
            else:
                self.layer_norm = LayerNorm(args.hidden_size)
        if self.relative_position_embedding:
            self.relative_pos_emb = RelativePositionEmbedding(bidirectional=True, heads_num=args.heads_num, num_buckets=args.relative_attention_buckets_num)

    def forward(self, emb, seg):
        """
        Args:
            emb: [batch_size x seq_length x emb_size]
            seg: [batch_size x seq_length]
        Returns:
            hidden: [batch_size x seq_length x hidden_size]
        """
        if self.factorized_embedding_parameterization:
            emb = self.linear(emb)
        batch_size, seq_length, _ = emb.size()
        if self.mask == 'fully_visible':
            mask = (seg > 0).unsqueeze(1).repeat(1, seq_length, 1).unsqueeze(1)
            mask = mask.float()
            mask = (1.0 - mask) * -10000.0
        elif self.mask == 'causal':
            mask = torch.ones(seq_length, seq_length, device=emb.device)
            mask = torch.tril(mask)
            mask = (1.0 - mask) * -10000
            mask = mask.repeat(batch_size, 1, 1, 1)
        else:
            mask_a = (seg == 1).unsqueeze(1).repeat(1, seq_length, 1).unsqueeze(1).float()
            mask_b = (seg > 0).unsqueeze(1).repeat(1, seq_length, 1).unsqueeze(1).float()
            mask_tril = torch.ones(seq_length, seq_length, device=emb.device)
            mask_tril = torch.tril(mask_tril)
            mask_tril = mask_tril.repeat(batch_size, 1, 1, 1)
            mask = (mask_a + mask_b + mask_tril >= 2).float()
            mask = (1.0 - mask) * -10000.0
        hidden = emb
        if self.relative_position_embedding:
            position_bias = self.relative_pos_emb(hidden, hidden)
        else:
            position_bias = None
        prev_attn = None
        if self.deepspeed_checkpoint_activations:

            def custom(start, end):

                def custom_forward(*inputs):
                    x_, y_, position_bias_ = inputs
                    for index in range(start, end):
                        if self.parameter_sharing:
                            x_, y_ = self.transformer(x_, mask, position_bias=position_bias_, has_residual_attention=self.has_residual_attention, prev_attn=y_)
                        else:
                            x_, y_ = self.transformer[index](x_, mask, position_bias=position_bias_, has_residual_attention=self.has_residual_attention, prev_attn=y_)
                    return x_, y_
                return custom_forward
            l = 0
            while l < self.layers_num:
                hidden, prev_attn = checkpointing.checkpoint(custom(l, l + self.deepspeed_checkpoint_layers_num), hidden, prev_attn, position_bias)
                l += self.deepspeed_checkpoint_layers_num
        else:
            for i in range(self.layers_num):
                if self.parameter_sharing:
                    hidden, prev_attn = self.transformer(hidden, mask, position_bias=position_bias, has_residual_attention=self.has_residual_attention, prev_attn=prev_attn)
                else:
                    hidden, prev_attn = self.transformer[i](hidden, mask, position_bias=position_bias, has_residual_attention=self.has_residual_attention, prev_attn=prev_attn)
        if self.layernorm_positioning == 'pre':
            return self.layer_norm(hidden)
        else:
            return hidden


str2encoder = {'transformer': TransformerEncoder, 'rnn': RnnEncoder, 'lstm': LstmEncoder, 'gru': GruEncoder, 'birnn': BirnnEncoder, 'bilstm': BilstmEncoder, 'bigru': BigruEncoder, 'gatedcnn': GatedcnnEncoder, 'dual': DualEncoder}


class MultipleChoice(nn.Module):

    def __init__(self, args):
        super(MultipleChoice, self).__init__()
        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
        self.encoder = str2encoder[args.encoder](args)
        self.dropout = nn.Dropout(args.dropout)
        self.output_layer = nn.Linear(args.hidden_size, 1)

    def forward(self, src, tgt, seg, soft_tgt=None):
        """
        Args:
            src: [batch_size x choices_num x seq_length]
            tgt: [batch_size]
            seg: [batch_size x choices_num x seq_length]
        """
        choices_num = src.shape[1]
        src = src.view(-1, src.size(-1))
        seg = seg.view(-1, seg.size(-1))
        emb = self.embedding(src, seg)
        output = self.encoder(emb, seg)
        output = self.dropout(output)
        logits = self.output_layer(output[:, 0, :])
        reshaped_logits = logits.view(-1, choices_num)
        if tgt is not None:
            loss = nn.NLLLoss()(nn.LogSoftmax(dim=-1)(reshaped_logits), tgt.view(-1))
            return loss, reshaped_logits
        else:
            return None, reshaped_logits


def pooling(memory_bank, seg, pooling_type):
    seg = torch.unsqueeze(seg, dim=-1).type_as(memory_bank)
    memory_bank = memory_bank * seg
    if pooling_type == 'mean':
        features = torch.sum(memory_bank, dim=1)
        features = torch.div(features, torch.sum(seg, dim=1))
    elif pooling_type == 'last':
        features = memory_bank[torch.arange(memory_bank.shape[0]), torch.squeeze(torch.sum(seg, dim=1).type(torch.int64) - 1), :]
    elif pooling_type == 'max':
        features = torch.max(memory_bank + (seg - 1) * sys.maxsize, dim=1)[0]
    else:
        features = memory_bank[:, 0, :]
    return features


class Classifier(nn.Module):

    def __init__(self, args):
        super(Classifier, self).__init__()
        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
        self.encoder = str2encoder[args.encoder](args)
        self.labels_num = args.labels_num
        self.pooling_type = args.pooling
        self.soft_targets = args.soft_targets
        self.soft_alpha = args.soft_alpha
        self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.output_layer_2 = nn.Linear(args.hidden_size, self.labels_num)

    def forward(self, src, tgt, seg, soft_tgt=None):
        """
        Args:
            src: [batch_size x seq_length]
            tgt: [batch_size]
            seg: [batch_size x seq_length]
        """
        emb = self.embedding(src, seg)
        output = self.encoder(emb, seg)
        output = pooling(output, seg, self.pooling_type)
        output = torch.tanh(self.output_layer_1(output))
        logits = self.output_layer_2(output)
        if tgt is not None:
            if self.soft_targets and soft_tgt is not None:
                loss = self.soft_alpha * nn.MSELoss()(logits, soft_tgt) + (1 - self.soft_alpha) * nn.NLLLoss()(nn.LogSoftmax(dim=-1)(logits), tgt.view(-1))
            else:
                loss = nn.NLLLoss()(nn.LogSoftmax(dim=-1)(logits), tgt.view(-1))
            return loss, logits
        else:
            return None, logits


class MultitaskClassifier(nn.Module):

    def __init__(self, args):
        super(MultitaskClassifier, self).__init__()
        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
        self.encoder = str2encoder[args.encoder](args)
        self.pooling_type = args.pooling
        self.output_layers_1 = nn.ModuleList([nn.Linear(args.hidden_size, args.hidden_size) for _ in args.labels_num_list])
        self.output_layers_2 = nn.ModuleList([nn.Linear(args.hidden_size, labels_num) for labels_num in args.labels_num_list])
        self.dataset_id = 0

    def forward(self, src, tgt, seg, soft_tgt=None):
        """
        Args:
            src: [batch_size x seq_length]
            tgt: [batch_size]
            seg: [batch_size x seq_length]
        """
        emb = self.embedding(src, seg)
        output = self.encoder(emb, seg)
        output = pooling(output, seg, self.pooling_type)
        output = torch.tanh(self.output_layers_1[self.dataset_id](output))
        logits = self.output_layers_2[self.dataset_id](output)
        if tgt is not None:
            loss = nn.NLLLoss()(nn.LogSoftmax(dim=-1)(logits), tgt.view(-1))
            return loss, logits
        else:
            return None, logits

    def change_dataset(self, dataset_id):
        self.dataset_id = dataset_id


class MultilabelClassifier(nn.Module):

    def __init__(self, args):
        super(MultilabelClassifier, self).__init__()
        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
        self.encoder = str2encoder[args.encoder](args)
        self.labels_num = args.labels_num
        self.pooling_type = args.pooling
        self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.output_layer_2 = nn.Linear(args.hidden_size, self.labels_num)

    def forward(self, src, tgt, seg):
        """
        Args:
            src: [batch_size x seq_length]
            tgt: [batch_size]
            seg: [batch_size x seq_length]
        """
        emb = self.embedding(src, seg)
        output = self.encoder(emb, seg)
        output = pooling(output, seg, self.pooling_type)
        output = torch.tanh(self.output_layer_1(output))
        logits = self.output_layer_2(output)
        if tgt is not None:
            probs_batch = nn.Sigmoid()(logits)
            loss = nn.BCELoss()(probs_batch, tgt)
            return loss, logits
        else:
            return None, logits


class MlmTarget(nn.Module):
    """
    BERT exploits masked language modeling (MLM)
    and next sentence prediction (NSP) for pretraining.
    """

    def __init__(self, args, vocab_size):
        super(MlmTarget, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = args.hidden_size
        self.emb_size = args.emb_size
        self.factorized_embedding_parameterization = args.factorized_embedding_parameterization
        self.act = str2act[args.hidden_act]
        if self.factorized_embedding_parameterization:
            self.mlm_linear_1 = nn.Linear(args.hidden_size, args.emb_size)
            self.layer_norm = LayerNorm(args.emb_size)
            self.mlm_linear_2 = nn.Linear(args.emb_size, self.vocab_size)
        else:
            self.mlm_linear_1 = nn.Linear(args.hidden_size, args.hidden_size)
            self.layer_norm = LayerNorm(args.hidden_size)
            self.mlm_linear_2 = nn.Linear(args.hidden_size, self.vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.criterion = nn.NLLLoss()

    def mlm(self, memory_bank, tgt_mlm):
        output_mlm = self.act(self.mlm_linear_1(memory_bank))
        output_mlm = self.layer_norm(output_mlm)
        if self.factorized_embedding_parameterization:
            output_mlm = output_mlm.contiguous().view(-1, self.emb_size)
        else:
            output_mlm = output_mlm.contiguous().view(-1, self.hidden_size)
        tgt_mlm = tgt_mlm.contiguous().view(-1)
        output_mlm = output_mlm[tgt_mlm > 0, :]
        tgt_mlm = tgt_mlm[tgt_mlm > 0]
        output_mlm = self.mlm_linear_2(output_mlm)
        output_mlm = self.softmax(output_mlm)
        denominator = torch.tensor(output_mlm.size(0) + 1e-06)
        if output_mlm.size(0) == 0:
            correct_mlm = torch.tensor(0.0)
        else:
            correct_mlm = torch.sum(output_mlm.argmax(dim=-1).eq(tgt_mlm).float())
        loss_mlm = self.criterion(output_mlm, tgt_mlm)
        return loss_mlm, correct_mlm, denominator

    def forward(self, memory_bank, tgt, seg):
        """
        Args:
            memory_bank: [batch_size x seq_length x hidden_size]
            tgt: [batch_size x seq_length]

        Returns:
            loss: Masked language modeling loss.
            correct: Number of words that are predicted correctly.
            denominator: Number of masked words.
        """
        loss, correct, denominator = self.mlm(memory_bank, tgt)
        return loss, correct, denominator


class ClozeTest(torch.nn.Module):

    def __init__(self, args):
        super(ClozeTest, self).__init__()
        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
        self.encoder = str2encoder[args.encoder](args)
        self.target = MlmTarget(args, len(args.tokenizer.vocab))
        self.act = str2act[args.hidden_act]

    def forward(self, src, seg):
        emb = self.embedding(src, seg)
        output = self.encoder(emb, seg)
        output = self.act(self.target.mlm_linear_1(output))
        output = self.target.layer_norm(output)
        output = self.target.mlm_linear_2(output)
        prob = torch.nn.Softmax(dim=-1)(output)
        return prob


class SiameseClassifier(nn.Module):

    def __init__(self, args):
        super(SiameseClassifier, self).__init__()
        self.embedding = DualEmbedding(args, len(args.tokenizer.vocab))
        self.encoder = DualEncoder(args)
        self.classifier = nn.Linear(4 * args.stream_0['hidden_size'], args.labels_num)
        self.pooling_type = args.pooling

    def forward(self, src, tgt, seg):
        """
        Args:
            src: [batch_size x seq_length]
            tgt: [batch_size]
            seg: [batch_size x seq_length]
        """
        emb = self.embedding(src, seg)
        output = self.encoder(emb, seg)
        features_0, features_1 = output
        features_0 = pooling(features_0, seg[0], self.pooling_type)
        features_1 = pooling(features_1, seg[1], self.pooling_type)
        vectors_concat = []
        vectors_concat.append(features_0)
        vectors_concat.append(features_1)
        vectors_concat.append(torch.abs(features_0 - features_1))
        vectors_concat.append(features_0 * features_1)
        features = torch.cat(vectors_concat, 1)
        logits = self.classifier(features)
        if tgt is not None:
            loss = nn.NLLLoss()(nn.LogSoftmax(dim=-1)(logits), tgt.view(-1))
            return loss, logits
        else:
            return None, logits


class MachineReadingComprehension(nn.Module):

    def __init__(self, args):
        super(MachineReadingComprehension, self).__init__()
        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
        self.encoder = str2encoder[args.encoder](args)
        self.output_layer = nn.Linear(args.hidden_size, 2)

    def forward(self, src, seg, start_position, end_position):
        emb = self.embedding(src, seg)
        output = self.encoder(emb, seg)
        logits = self.output_layer(output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits, end_logits = start_logits.squeeze(-1), end_logits.squeeze(-1)
        start_loss = nn.NLLLoss()(nn.LogSoftmax(dim=-1)(start_logits), start_position)
        end_loss = nn.NLLLoss()(nn.LogSoftmax(dim=-1)(end_logits), end_position)
        loss = (start_loss + end_loss) / 2
        return loss, start_logits, end_logits


class NerTagger(nn.Module):

    def __init__(self, args):
        super(NerTagger, self).__init__()
        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
        self.encoder = str2encoder[args.encoder](args)
        self.labels_num = args.labels_num
        self.output_layer = nn.Linear(args.hidden_size, self.labels_num)
        self.crf_target = args.crf_target
        if args.crf_target:
            self.crf = CRF(self.labels_num, batch_first=True)
            self.seq_length = args.seq_length

    def forward(self, src, tgt, seg):
        """
        Args:
            src: [batch_size x seq_length]
            tgt: [batch_size x seq_length]
            seg: [batch_size x seq_length]
        Returns:
            loss: Sequence labeling loss.
            logits: Output logits.
        """
        emb = self.embedding(src, seg)
        output = self.encoder(emb, seg)
        logits = self.output_layer(output)
        if self.crf_target:
            tgt_mask = seg.type(torch.uint8)
            pred = self.crf.decode(logits, mask=tgt_mask)
            for j in range(len(pred)):
                while len(pred[j]) < self.seq_length:
                    pred[j].append(self.labels_num - 1)
            pred = torch.tensor(pred).contiguous().view(-1)
            if tgt is not None:
                loss = -self.crf(F.log_softmax(logits, 2), tgt, mask=tgt_mask, reduction='mean')
                return loss, pred
            else:
                return None, pred
        else:
            tgt_mask = seg.contiguous().view(-1).float()
            logits = logits.contiguous().view(-1, self.labels_num)
            pred = logits.argmax(dim=-1)
            if tgt is not None:
                tgt = tgt.contiguous().view(-1, 1)
                one_hot = torch.zeros(tgt.size(0), self.labels_num).scatter_(1, tgt, 1.0)
                numerator = -torch.sum(nn.LogSoftmax(dim=-1)(logits) * one_hot, 1)
                numerator = torch.sum(tgt_mask * numerator)
                denominator = torch.sum(tgt_mask) + 1e-06
                loss = numerator / denominator
                return loss, pred
            else:
                return None, pred


class Regression(nn.Module):

    def __init__(self, args):
        super(Regression, self).__init__()
        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
        self.encoder = str2encoder[args.encoder](args)
        self.pooling_type = args.pooling
        self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.output_layer_2 = nn.Linear(args.hidden_size, 1)

    def forward(self, src, tgt, seg, soft_tgt=None):
        """
        Args:
            src: [batch_size x seq_length]
            tgt: [batch_size]
            seg: [batch_size x seq_length]
        """
        emb = self.embedding(src, seg)
        output = self.encoder(emb, seg)
        output = pooling(output, seg, self.pooling_type)
        output = torch.tanh(self.output_layer_1(output))
        logits = self.output_layer_2(output)
        if tgt is not None:
            loss = nn.MSELoss()(logits.view(-1), tgt.view(-1))
            return loss, logits
        else:
            return None, logits


class SimCSE(nn.Module):

    def __init__(self, args):
        super(SimCSE, self).__init__()
        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
        self.encoder = str2encoder[args.encoder](args)
        self.pooling_type = args.pooling

    def forward(self, src, seg):
        """
        Args:
            src: [batch_size x seq_length]
            tgt: [batch_size]
            seg: [batch_size x seq_length]
        """
        emb_0 = self.embedding(src[0], seg[0])
        emb_1 = self.embedding(src[1], seg[1])
        output_0 = self.encoder(emb_0, seg[0])
        output_1 = self.encoder(emb_1, seg[1])
        features_0 = self.pooling(output_0, seg[0], self.pooling_type)
        features_1 = self.pooling(output_1, seg[1], self.pooling_type)
        return features_0, features_1

    def pooling(self, memory_bank, seg, pooling_type):
        seg = torch.unsqueeze(seg, dim=-1).type(torch.float)
        memory_bank = memory_bank * seg
        if pooling_type == 'mean':
            features = torch.sum(memory_bank, dim=1)
            features = torch.div(features, torch.sum(seg, dim=1))
        elif pooling_type == 'last':
            features = memory_bank[torch.arange(memory_bank.shape[0]), torch.squeeze(torch.sum(seg, dim=1).type(torch.int64) - 1), :]
        elif pooling_type == 'max':
            features = torch.max(memory_bank + (seg - 1) * sys.maxsize, dim=1)[0]
        else:
            features = memory_bank[:, 0, :]
        return features


class LmTarget(nn.Module):
    """
    Language Model Target
    """

    def __init__(self, args, vocab_size):
        super(LmTarget, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = args.hidden_size
        self.output_layer = nn.Linear(self.hidden_size, self.vocab_size, bias=args.has_lmtarget_bias)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.criterion = nn.NLLLoss()

    def lm(self, memory_bank, tgt_lm):
        tgt_lm = tgt_lm.contiguous().view(-1)
        memory_bank = memory_bank.contiguous().view(-1, self.hidden_size)
        memory_bank = memory_bank[tgt_lm > 0, :]
        tgt_lm = tgt_lm[tgt_lm > 0]
        output = self.output_layer(memory_bank)
        output = self.softmax(output)
        denominator = torch.tensor(output.size(0) + 1e-06)
        if output.size(0) == 0:
            correct = torch.tensor(0.0)
        else:
            correct = torch.sum(output.argmax(dim=-1).eq(tgt_lm).float())
        loss = self.criterion(output, tgt_lm)
        return loss, correct, denominator

    def forward(self, memory_bank, tgt, seg):
        """
        Args:
            memory_bank: [batch_size x seq_length x hidden_size]
            tgt: [batch_size x seq_length]

        Returns:
            loss: Language modeling loss.
            correct: Number of words that are predicted correctly.
            denominator: Number of predicted words.
        """
        loss, correct, denominator = self.lm(memory_bank, tgt)
        return loss, correct, denominator


class TransformerDecoderLayer(nn.Module):

    def __init__(self, args):
        super(TransformerDecoderLayer, self).__init__()
        self.layernorm_positioning = args.layernorm_positioning
        if hasattr(args, 'attention_head_size'):
            attention_head_size = args.attention_head_size
        else:
            attention_head_size = args.hidden_size // args.heads_num
        has_bias = bool(1 - args.remove_transformer_bias)
        with_scale = bool(1 - args.remove_attention_scale)
        self.self_attn = MultiHeadedAttention(args.hidden_size, args.heads_num, attention_head_size, args.dropout, has_bias=has_bias, with_scale=with_scale)
        self.dropout_1 = nn.Dropout(args.dropout)
        self.context_attn = MultiHeadedAttention(args.hidden_size, args.heads_num, attention_head_size, args.dropout, has_bias=has_bias, with_scale=with_scale)
        self.dropout_2 = nn.Dropout(args.dropout)
        if args.feed_forward == 'gated':
            self.feed_forward = GatedFeedForward(args.hidden_size, args.feedforward_size, args.hidden_act, has_bias)
        else:
            self.feed_forward = PositionwiseFeedForward(args.hidden_size, args.feedforward_size, args.hidden_act, has_bias)
        self.dropout_3 = nn.Dropout(args.dropout)
        if args.layernorm == 't5':
            self.layer_norm_1 = T5LayerNorm(args.hidden_size)
            self.layer_norm_2 = T5LayerNorm(args.hidden_size)
            self.layer_norm_3 = T5LayerNorm(args.hidden_size)
        else:
            self.layer_norm_1 = LayerNorm(args.hidden_size)
            self.layer_norm_2 = LayerNorm(args.hidden_size)
            self.layer_norm_3 = LayerNorm(args.hidden_size)

    def forward(self, hidden, encoder_hidden, mask_decoder, mask_encoder, self_position_bias=None, context_position_bias=None):
        """
        Args:
            hidden: [batch_size x seq_length x emb_size]
            encoder_hidden: [batch_size x seq_length x emb_size]
            mask_encoder: [batch_size x 1 x seq_length x seq_length]
            mask_decoder: [batch_size x 1 x seq_length x seq_length]
            self_position_bias: [1 x heads_num x seq_length x seq_length]
            context_position_bias: [1 x heads_num x seq_length x seq_length]
        Returns:
            output: [batch_size x seq_length x hidden_size]
        """
        if self.layernorm_positioning == 'post':
            query, _ = self.self_attn(hidden, hidden, hidden, mask_decoder, self_position_bias)
            query = self.dropout_1(query)
            query_norm = self.layer_norm_1(query + hidden)
            mid, _ = self.context_attn(encoder_hidden, encoder_hidden, query_norm, mask_encoder, context_position_bias)
            mid = self.dropout_2(mid)
            mid_norm = self.layer_norm_2(mid + query_norm)
            output = self.dropout_3(self.feed_forward(mid_norm))
            output = self.layer_norm_3(output + mid_norm)
        else:
            hidden_norm = self.layer_norm_1(hidden)
            query, _ = self.self_attn(hidden_norm, hidden_norm, hidden_norm, mask_decoder, self_position_bias)
            query = self.dropout_1(query)
            query = query + hidden
            query_norm = self.layer_norm_2(query)
            mid, _ = self.context_attn(encoder_hidden, encoder_hidden, query_norm, mask_encoder, context_position_bias)
            mid = self.dropout_2(mid)
            mid = mid + query
            mid_norm = self.layer_norm_3(mid)
            output = self.dropout_3(self.feed_forward(mid_norm)) + mid
        return output


class TransformerDecoder(nn.Module):
    """
    BERT encoder exploits 12 or 24 transformer layers to extract features.
    """

    def __init__(self, args):
        super(TransformerDecoder, self).__init__()
        self.layers_num = args.layers_num
        self.layernorm_positioning = args.layernorm_positioning
        self.relative_position_embedding = args.relative_position_embedding
        self.transformer_decoder = nn.ModuleList([TransformerDecoderLayer(args) for _ in range(self.layers_num)])
        if 'deepspeed_checkpoint_activations' in args:
            self.deepspeed_checkpoint_activations = args.deepspeed_checkpoint_activations
            self.deepspeed_checkpoint_layers_num = args.deepspeed_checkpoint_layers_num
        else:
            self.deepspeed_checkpoint_activations = False
        has_bias = bool(1 - args.remove_transformer_bias)
        if self.layernorm_positioning == 'pre':
            if args.layernorm == 't5':
                self.layer_norm = T5LayerNorm(args.hidden_size)
            else:
                self.layer_norm = LayerNorm(args.hidden_size)
        if self.relative_position_embedding:
            self.self_pos_emb = RelativePositionEmbedding(bidirectional=False, heads_num=args.heads_num, num_buckets=args.relative_attention_buckets_num)

    def forward(self, memory_bank, emb, additional_info):
        """
        Args:
            memory_bank: [batch_size x seq_length x emb_size]
            emb: [batch_size x seq_length x emb_size]
        Returns:
            hidden: [batch_size x seq_length x hidden_size]
        """
        _, src_seq_length, _ = memory_bank.size()
        batch_size, tgt_seq_length, _ = emb.size()
        mask_encoder = (additional_info[0] > 0).unsqueeze(1).repeat(1, tgt_seq_length, 1).unsqueeze(1)
        mask_encoder = mask_encoder.float()
        mask_encoder = (1.0 - mask_encoder) * -10000.0
        mask_decoder = torch.ones(tgt_seq_length, tgt_seq_length, device=emb.device)
        mask_decoder = torch.tril(mask_decoder)
        mask_decoder = (1.0 - mask_decoder) * -10000
        mask_decoder = mask_decoder.repeat(batch_size, 1, 1, 1)
        hidden = emb
        if self.relative_position_embedding:
            self_position_bias = self.self_pos_emb(hidden, hidden)
        else:
            self_position_bias = None
        if self.deepspeed_checkpoint_activations:

            def custom(start, end):

                def custom_forward(*inputs):
                    x_, memory_bank_, self_position_bias_ = inputs
                    for index in range(start, end):
                        x_ = self.transformer_decoder[index](x_, memory_bank_, mask_decoder, mask_encoder, self_position_bias_, None)
                    return x_
                return custom_forward
            l = 0
            while l < self.layers_num:
                hidden = checkpointing.checkpoint(custom(l, l + self.deepspeed_checkpoint_layers_num), hidden, memory_bank, self_position_bias)
                l += self.deepspeed_checkpoint_layers_num
        else:
            for i in range(self.layers_num):
                hidden = self.transformer_decoder[i](hidden, memory_bank, mask_decoder, mask_encoder, self_position_bias, None)
        if self.layernorm_positioning == 'pre':
            return self.layer_norm(hidden)
        else:
            return hidden


str2decoder = {'transformer': TransformerDecoder}


class Text2text(torch.nn.Module):

    def __init__(self, args):
        super(Text2text, self).__init__()
        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
        self.encoder = str2encoder[args.encoder](args)
        self.tgt_embedding = str2embedding[args.tgt_embedding](args, len(args.tokenizer.vocab))
        self.decoder = str2decoder[args.decoder](args)
        self.target = LmTarget(args, len(args.tokenizer.vocab))
        if args.tie_weights:
            self.target.output_layer.weight = self.embedding.word_embedding.weight
        if args.share_embedding:
            self.tgt_embedding.word_embedding.weight = self.embedding.word_embedding.weight

    def encode(self, src, seg):
        emb = self.embedding(src, seg)
        memory_bank = self.encoder(emb, seg)
        return memory_bank

    def decode(self, src, memory_bank, tgt):
        tgt_in, tgt_out, _ = tgt
        decoder_emb = self.tgt_embedding(tgt_in, None)
        hidden = self.decoder(memory_bank, decoder_emb, (src,))
        output = self.target.output_layer(hidden)
        return output

    def forward(self, src, tgt, seg, memory_bank=None, only_use_encoder=False):
        if only_use_encoder:
            return self.encode(src, seg)
        if memory_bank is not None:
            return self.decode(src, memory_bank, tgt)
        tgt_in, tgt_out, _ = tgt
        memory_bank = self.encode(src, seg)
        output = self.decode(src, memory_bank, tgt)
        if tgt_out is None:
            return None, output
        else:
            decoder_emb = self.tgt_embedding(tgt_in, None)
            hidden = self.decoder(memory_bank, decoder_emb, (seg,))
            loss = self.target(hidden, tgt_out, seg)[0]
            return loss, output


class FeatureExtractor(torch.nn.Module):

    def __init__(self, args):
        super(FeatureExtractor, self).__init__()
        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
        self.encoder = str2encoder[args.encoder](args)
        self.pooling_type = args.pooling

    def forward(self, src, seg):
        emb = self.embedding(src, seg)
        output = self.encoder(emb, seg)
        output = pooling(output, seg, self.pooling_type)
        return output


class WhiteningHandle(torch.nn.Module):
    """
    Whitening operation.
    @ref: https://github.com/bojone/BERT-whitening/blob/main/demo.py
    """

    def __init__(self, args, vecs):
        super(WhiteningHandle, self).__init__()
        self.kernel, self.bias = self._compute_kernel_bias(vecs)

    def forward(self, vecs, n_components=None, normal=True, pt=True):
        vecs = self._format_vecs_to_np(vecs)
        vecs = self._transform(vecs, n_components)
        vecs = self._normalize(vecs) if normal else vecs
        vecs = torch.tensor(vecs) if pt else vecs
        return vecs

    def _compute_kernel_bias(self, vecs):
        vecs = self._format_vecs_to_np(vecs)
        mu = vecs.mean(axis=0, keepdims=True)
        cov = np.cov(vecs.T)
        u, s, vh = np.linalg.svd(cov)
        W = np.dot(u, np.diag(1 / np.sqrt(s)))
        return W, -mu

    def _transform(self, vecs, n_components):
        w = self.kernel[:, :n_components] if isinstance(n_components, int) else self.kernel
        return (vecs + self.bias).dot(w)

    def _normalize(self, vecs):
        return vecs / (vecs ** 2).sum(axis=1, keepdims=True) ** 0.5

    def _format_vecs_to_np(self, vecs):
        vecs_np = []
        for vec in vecs:
            if isinstance(vec, list):
                vec = np.array(vec)
            elif torch.is_tensor(vec):
                vec = vec.detach().numpy()
            elif isinstance(vec, np.ndarray):
                vec = vec
            else:
                raise Exception('Unknown vec type.')
            vecs_np.append(vec)
        vecs_np = np.array(vecs_np)
        return vecs_np


class GenerateLm(torch.nn.Module):

    def __init__(self, args):
        super(GenerateLm, self).__init__()
        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
        self.encoder = str2encoder[args.encoder](args)
        self.target = LmTarget(args, len(args.tokenizer.vocab))

    def forward(self, src, seg):
        emb = self.embedding(src, seg)
        output = self.encoder(emb, seg)
        output = self.target.output_layer(output)
        return output


class GenerateSeq2seq(torch.nn.Module):

    def __init__(self, args):
        super(GenerateSeq2seq, self).__init__()
        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
        self.encoder = str2encoder[args.encoder](args)
        self.tgt_embedding = str2embedding[args.tgt_embedding](args, len(args.tgt_tokenizer.vocab))
        self.decoder = str2decoder[args.decoder](args)
        self.target = LmTarget(args, len(args.tgt_tokenizer.vocab))

    def forward(self, src, seg, tgt):
        emb = self.embedding(src, seg)
        memory_bank = self.encoder(emb, seg)
        emb = self.tgt_embedding(tgt, None)
        hidden = self.decoder(memory_bank, emb, (src,))
        output = self.target.output_layer(hidden)
        return output


class SequenceEncoder(torch.nn.Module):

    def __init__(self, args):
        super(SequenceEncoder, self).__init__()
        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
        self.encoder = str2encoder[args.encoder](args)

    def forward(self, src, seg):
        emb = self.embedding(src, seg)
        output = self.encoder(emb, seg)
        return output


class Model(nn.Module):
    """
    Pretraining models consist of three (five) parts:
        - embedding
        - encoder
        - tgt_embedding (optional)
        - decoder (optional)
        - target
    """

    def __init__(self, args, embedding, encoder, tgt_embedding, decoder, target):
        super(Model, self).__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.tgt_embedding = tgt_embedding
        self.decoder = decoder
        self.target = target
        if 'mlm' in args.target and args.tie_weights:
            self.target.mlm_linear_2.weight = self.embedding.word_embedding.weight
        elif 'lm' in args.target and args.tie_weights:
            self.target.output_layer.weight = self.embedding.word_embedding.weight
        if self.decoder is not None and args.share_embedding:
            self.tgt_embedding.word_embedding.weight = self.embedding.word_embedding.weight

    def forward(self, src, tgt, seg, tgt_in=None, tgt_seg=None):
        emb = self.embedding(src, seg)
        memory_bank = self.encoder(emb, seg)
        if self.decoder:
            tgt_emb = self.tgt_embedding(tgt_in, tgt_seg)
            memory_bank = self.decoder(memory_bank, tgt_emb, (seg, tgt_seg))
        loss_info = self.target(memory_bank, tgt, seg)
        return loss_info


class ClsTarget(nn.Module):
    """
    Classification Target
    """

    def __init__(self, args, vocab_size):
        super(ClsTarget, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = args.hidden_size
        self.pooling_type = args.pooling
        self.linear_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.linear_2 = nn.Linear(args.hidden_size, args.labels_num)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.criterion = nn.NLLLoss()

    def forward(self, memory_bank, tgt, seg):
        """
        Args:
            memory_bank: [batch_size x seq_length x hidden_size]
            tgt: [batch_size]

        Returns:
            loss: Classification loss.
            correct: Number of sentences that are predicted correctly.
        """
        output = pooling(memory_bank, seg, self.pooling_type)
        output = torch.tanh(self.linear_1(output))
        logits = self.linear_2(output)
        loss = self.criterion(self.softmax(logits), tgt)
        correct = self.softmax(logits).argmax(dim=-1).eq(tgt).sum()
        return loss, correct


class SpTarget(nn.Module):

    def __init__(self, args, vocab_size):
        super(SpTarget, self).__init__()
        self.sp_linear_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.sp_linear_2 = nn.Linear(args.hidden_size, 2)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.criterion = nn.NLLLoss()

    def forward(self, memory_bank, tgt, seg):
        """
        Args:
            memory_bank: [batch_size x seq_length x hidden_size]
            tgt: tuple with tgt_mlm [batch_size x seq_length] and tgt_sop [batch_size]

        Returns:
            loss_sop: Sentence order prediction loss.
            correct_sop: Number of sentences that are predicted correctly.
        """
        output_sp = torch.tanh(self.sp_linear_1(memory_bank[:, 0, :]))
        output_sp = self.sp_linear_2(output_sp)
        loss_sp = self.criterion(self.softmax(output_sp), tgt)
        correct_sp = self.softmax(output_sp).argmax(dim=-1).eq(tgt).sum()
        return loss_sp, correct_sp


class Target(nn.Module):

    def __init__(self):
        self.target_list = []
        self.target_name_list = []
        self.loss_info = {}

    def update(self, target, target_name):
        self.target_list.append(target.forward)
        self.target_name_list.append(target_name)
        if '_modules' in self.__dict__:
            self.__dict__['_modules'].update(target.__dict__['_modules'])
        else:
            self.__dict__.update(target.__dict__)

    def forward(self, memory_bank, tgt, seg):
        self.loss_info = {}
        for i, target in enumerate(self.target_list):
            if len(self.target_list) > 1:
                self.loss_info[self.target_name_list[i]] = target(memory_bank, tgt[self.target_name_list[i]], seg)
            else:
                self.loss_info = target(memory_bank, tgt, seg)
        return self.loss_info


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BigruEncoder,
     lambda: ([], {'args': _mock_config(hidden_size=4, layers_num=1, emb_size=4, dropout=0.5)}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (BilstmEncoder,
     lambda: ([], {'args': _mock_config(hidden_size=4, layers_num=1, emb_size=4, dropout=0.5)}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (BirnnEncoder,
     lambda: ([], {'args': _mock_config(hidden_size=4, layers_num=1, emb_size=4, dropout=0.5)}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (GatedcnnEncoder,
     lambda: ([], {'args': _mock_config(layers_num=1, kernel_size=4, block_size=4, emb_size=4, hidden_size=4)}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (LayerNorm,
     lambda: ([], {'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MultiHeadedAttention,
     lambda: ([], {'hidden_size': 4, 'heads_num': 4, 'attention_head_size': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4, 16])], {}),
     False),
    (T5LayerNorm,
     lambda: ([], {'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_dbiir_UER_py(_paritybench_base):
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

