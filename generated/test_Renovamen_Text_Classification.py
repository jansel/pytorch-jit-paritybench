import sys
_module = sys.modules[__name__]
del sys
classify = _module
datasets = _module
dataloader = _module
info = _module
ag_news = _module
amazon_full = _module
amazon_polarity = _module
dbpedia = _module
yahoo_answers = _module
yelp_full = _module
yelp_polarity = _module
preprocess = _module
document = _module
sentence = _module
utils = _module
AttBiLSTM = _module
att_bilstm = _module
attention = _module
HAN = _module
han = _module
sent_encoder = _module
word_encoder = _module
TextCNN = _module
cnn1d = _module
cnn2d = _module
Transformer = _module
attention = _module
encoder_layer = _module
ffn = _module
pe = _module
transformer = _module
models = _module
fastText = _module
fasttext = _module
test = _module
train = _module
trainer = _module
trainer = _module
common = _module
embedding = _module
opts = _module
tensorboard = _module

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


from typing import Tuple


from typing import Dict


import torch


from torch import nn


from typing import Union


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from collections import Counter


import pandas as pd


from torch.nn.utils.rnn import pack_padded_sequence


from torch.nn.utils.rnn import pad_packed_sequence


from torch.nn.utils.rnn import PackedSequence


import torch.nn as nn


import torch.nn.functional as F


from typing import List


from typing import Optional


import numpy as np


import copy


import time


import torch.backends.cudnn as cudnn


from torch import optim


from typing import Callable


class Attention(nn.Module):
    """
    Attention network

    Parameters
    ----------
    rnn_size : int
        Size of Bi-LSTM
    """

    def __init__(self, rnn_size: int) ->None:
        super(Attention, self).__init__()
        self.w = nn.Linear(rnn_size, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, H: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        H : torch.Tensor (batch_size, word_pad_len, hidden_size)
            Output of Bi-LSTM

        Returns
        -------
        r : torch.Tensor (batch_size, rnn_size)
            Sentence representation

        alpha : torch.Tensor (batch_size, word_pad_len)
            Attention weights
        """
        M = self.tanh(H)
        alpha = self.w(M).squeeze(2)
        alpha = self.softmax(alpha)
        r = H * alpha.unsqueeze(2)
        r = r.sum(dim=1)
        return r, alpha


class AttBiLSTM(nn.Module):
    """
    Implementation of Attention-based bidirectional LSTM proposed in paper [1].

    Parameters
    ----------
    n_classes : int
        Number of classes

    vocab_size : int
        Number of words in the vocabulary

    embeddings : torch.Tensor
        Word embedding weights

    emb_size : int
        Size of word embeddings

    fine_tune : bool
        Allow fine-tuning of embedding layer? (only makes sense when using
        pre-trained embeddings)

    rnn_size : int
        Size of Bi-LSTM

    rnn_layers : int
        Number of layers in Bi-LSTM

    dropout : float
        Dropout

    References
    ----------
    1. "`Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification.         <https://www.aclweb.org/anthology/P16-2034.pdf>`_" Peng Zhou, et al. ACL 2016.
    """

    def __init__(self, n_classes: int, vocab_size: int, embeddings: torch.Tensor, emb_size: int, fine_tune: bool, rnn_size: int, rnn_layers: int, dropout: float) ->None:
        super(AttBiLSTM, self).__init__()
        self.rnn_size = rnn_size
        self.embeddings = nn.Embedding(vocab_size, emb_size)
        self.set_embeddings(embeddings, fine_tune)
        self.BiLSTM = nn.LSTM(emb_size, rnn_size, num_layers=rnn_layers, bidirectional=True, dropout=0 if rnn_layers == 1 else dropout, batch_first=True)
        self.attention = Attention(rnn_size)
        self.fc = nn.Linear(rnn_size, n_classes)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=1)

    def set_embeddings(self, embeddings: torch.Tensor, fine_tune: bool=True) ->None:
        """
        Set weights for embedding layer

        Parameters
        ----------
        embeddings : torch.Tensor
            Word embeddings

        fine_tune : bool, optional, default=True
            Allow fine-tuning of embedding layer? (only makes sense when using
            pre-trained embeddings)
        """
        if embeddings is None:
            self.embeddings.weight.data.uniform_(-0.1, 0.1)
        else:
            self.embeddings.weight = nn.Parameter(embeddings, requires_grad=fine_tune)

    def forward(self, text: torch.Tensor, words_per_sentence: torch.Tensor) ->torch.Tensor:
        """
        Parameters
        ----------
        text : torch.Tensor (batch_size, word_pad_len)
            Input data

        words_per_sentence : torch.Tensor (batch_size)
            Sentence lengths

        Returns
        -------
        scores : torch.Tensor (batch_size, n_classes)
            Class scores
        """
        embeddings = self.dropout(self.embeddings(text))
        packed_words = pack_padded_sequence(embeddings, lengths=words_per_sentence.tolist(), batch_first=True, enforce_sorted=False)
        rnn_out, _ = self.BiLSTM(packed_words)
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)
        H = rnn_out[:, :, :self.rnn_size] + rnn_out[:, :, self.rnn_size:]
        r, alphas = self.attention(H)
        h = self.tanh(r)
        scores = self.fc(self.dropout(h))
        return scores


class WordEncoder(nn.Module):
    """
    Word-level attention module

    Parameters
    ----------
    vocab_size : int
        Number of words in the vocabulary

    embeddings : torch.Tensor
        Word embedding weights

    emb_size : int
        Size of word embeddings

    fine_tune : bool
        Allow fine-tuning of embedding layer? (only makes sense when using
        pre-trained embeddings)

    word_rnn_size : int
        Size of (bidirectional) word-level RNN

    word_rnn_layers : int
        Number of layers in word-level RNN

    word_att_size : int
        Size of word-level attention layer

    dropout : float
        Dropout
    """

    def __init__(self, vocab_size: int, embeddings: torch.Tensor, emb_size: int, fine_tune: bool, word_rnn_size: int, word_rnn_layers: int, word_att_size: int, dropout: float) ->None:
        super(WordEncoder, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, emb_size)
        self.set_embeddings(embeddings, fine_tune)
        self.word_rnn = nn.GRU(emb_size, word_rnn_size, num_layers=word_rnn_layers, bidirectional=True, dropout=0 if word_rnn_layers == 1 else dropout, batch_first=True)
        self.W_w = nn.Linear(2 * word_rnn_size, word_att_size)
        self.u_w = nn.Linear(word_att_size, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def set_embeddings(self, embeddings: torch.Tensor, fine_tune: bool=True) ->None:
        """
        Set weights for embedding layer

        Parameters
        ----------
        embeddings : torch.Tensor
            Word embeddings

        fine_tune : bool, optional, default=True
            Allow fine-tuning of embedding layer? (only makes sense when using
            pre-trained embeddings)
        """
        if embeddings is None:
            self.embeddings.weight.data.uniform_(-0.1, 0.1)
        else:
            self.embeddings.weight = nn.Parameter(embeddings, requires_grad=fine_tune)

    def forward(self, sentences: torch.Tensor, words_per_sentence: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        sentences : torch.Tensor (n_sentences, word_pad_len, emb_size)
            Encoded sentence-level data

        words_per_sentence : torch.Tensor (n_sentences)
            Sentence lengths

        Returns
        -------
        sentences : torch.Tensor
            Sentence embeddings

        word_alphas : torch.Tensor
            Attention weights on each word
        """
        sentences = self.dropout(self.embeddings(sentences))
        packed_words = pack_padded_sequence(sentences, lengths=words_per_sentence.tolist(), batch_first=True, enforce_sorted=False)
        packed_words, _ = self.word_rnn(packed_words)
        sentences, _ = pad_packed_sequence(packed_words, batch_first=True)
        u_it = self.W_w(sentences)
        u_it = self.tanh(u_it)
        word_alphas = self.u_w(u_it).squeeze(2)
        word_alphas = self.softmax(word_alphas)
        sentences = sentences * word_alphas.unsqueeze(2)
        sentences = sentences.sum(dim=1)
        return sentences, word_alphas


class SentenceEncoder(nn.Module):
    """
    Sentence-level attention module

    Parameters
    ----------
    vocab_size : int
        Number of words in the vocabulary

    embeddings : torch.Tensor
        Word embedding weights

    emb_size : int
        Size of word embeddings

    fine_tune : bool
        Allow fine-tuning of embedding layer? (only makes sense when using
        pre-trained embeddings)

    word_rnn_size : int
        Size of (bidirectional) word-level RNN

    sentence_rnn_size : int
        Size of (bidirectional) sentence-level RNN

    word_rnn_layers : int
        Number of layers in word-level RNN

    sentence_rnn_layers : int
        Number of layers in sentence-level RNN

    word_att_size : int
        Size of word-level attention layer

    sentence_att_size : int
        Size of sentence-level attention layer

    dropout : float
        Dropout
    """

    def __init__(self, vocab_size: int, embeddings: torch.Tensor, emb_size: int, fine_tune: bool, word_rnn_size: int, sentence_rnn_size: int, word_rnn_layers: int, sentence_rnn_layers: int, word_att_size: int, sentence_att_size: int, dropout: float) ->None:
        super(SentenceEncoder, self).__init__()
        self.word_encoder = WordEncoder(vocab_size=vocab_size, embeddings=embeddings, emb_size=emb_size, fine_tune=fine_tune, word_rnn_size=word_rnn_size, word_rnn_layers=word_rnn_layers, word_att_size=word_att_size, dropout=dropout)
        self.sentence_rnn = nn.GRU(2 * word_rnn_size, sentence_rnn_size, num_layers=sentence_rnn_layers, bidirectional=True, dropout=0 if sentence_rnn_layers == 1 else dropout, batch_first=True)
        self.W_s = nn.Linear(2 * sentence_rnn_size, sentence_att_size)
        self.u_s = nn.Linear(sentence_att_size, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, documents: torch.Tensor, sentences_per_document: torch.Tensor, words_per_sentence: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        documents : torch.Tensor (n_documents, sent_pad_len, word_pad_len)
            Encoded document-level data

        sentences_per_document : torch.Tensor (n_documents)
            Document lengths

        words_per_sentence : torch.Tensor (n_documents, sent_pad_len)
            Sentence lengths

        Returns
        -------
        documents : torch.Tensor
            Document embeddings

        word_alphas : torch.Tensor
            Attention weights on each word

        sentence_alphas : torch.Tensor
            Attention weights on each sentence
        """
        packed_sentences = pack_padded_sequence(documents, lengths=sentences_per_document.tolist(), batch_first=True, enforce_sorted=False)
        packed_words_per_sentence = pack_padded_sequence(words_per_sentence, lengths=sentences_per_document.tolist(), batch_first=True, enforce_sorted=False)
        sentences, word_alphas = self.word_encoder(packed_sentences.data, packed_words_per_sentence.data)
        sentences = self.dropout(sentences)
        packed_sentences, _ = self.sentence_rnn(PackedSequence(data=sentences, batch_sizes=packed_sentences.batch_sizes, sorted_indices=packed_sentences.sorted_indices, unsorted_indices=packed_sentences.unsorted_indices))
        documents, _ = pad_packed_sequence(packed_sentences, batch_first=True)
        u_i = self.W_s(documents)
        u_i = self.tanh(u_i)
        sent_alphas = self.u_s(u_i).squeeze(2)
        sent_alphas = self.softmax(sent_alphas)
        documents = documents * sent_alphas.unsqueeze(2)
        documents = documents.sum(dim=1)
        word_alphas, _ = pad_packed_sequence(PackedSequence(data=word_alphas, batch_sizes=packed_sentences.batch_sizes, sorted_indices=packed_sentences.sorted_indices, unsorted_indices=packed_sentences.unsorted_indices), batch_first=True)
        return documents, word_alphas, sent_alphas


class HAN(nn.Module):
    """
    Implementation of Hierarchial Attention Network (HAN) proposed in paper [1].

    Parameters
    ----------
    n_classes : int
        Number of classes

    vocab_size : int
        Number of words in the vocabulary

    embeddings : torch.Tensor
        Word embedding weights

    emb_size : int
        Size of word embeddings

    fine_tune : bool
        Allow fine-tuning of embedding layer? (only makes sense when using
        pre-trained embeddings)

    word_rnn_size : int
        Size of (bidirectional) word-level RNN

    sentence_rnn_size : int
        Size of (bidirectional) sentence-level RNN

    word_rnn_layers : int
        Number of layers in word-level RNN

    sentence_rnn_layers : int
        Number of layers in sentence-level RNN

    word_att_size : int
        Size of word-level attention layer

    sentence_att_size : int
        Size of sentence-level attention layer

    dropout : float, optional, default=0.5
        Dropout
    """

    def __init__(self, n_classes: int, vocab_size: int, embeddings: torch.Tensor, emb_size: int, fine_tune: bool, word_rnn_size: int, sentence_rnn_size: int, word_rnn_layers: int, sentence_rnn_layers: int, word_att_size: int, sentence_att_size: int, dropout: float=0.5) ->None:
        super(HAN, self).__init__()
        self.sentence_encoder = SentenceEncoder(vocab_size, embeddings, emb_size, fine_tune, word_rnn_size, sentence_rnn_size, word_rnn_layers, sentence_rnn_layers, word_att_size, sentence_att_size, dropout)
        self.fc = nn.Linear(2 * sentence_rnn_size, n_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, documents: torch.Tensor, sentences_per_document: torch.Tensor, words_per_sentence: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        documents : torch.Tensor (n_documents, sent_pad_len, word_pad_len)
            Encoded document-level data

        sentences_per_document : torch.Tensor (n_documents)
            Document lengths

        words_per_sentence : torch.Tensor (n_documents, sent_pad_len)
            Sentence lengths

        Returns
        -------
        scores : torch.Tensor (batch_size, n_classes)
            Class scores

        word_alphas : torch.Tensor
            Attention weights on each word

        sentence_alphas : torch.Tensor
            Attention weights on each sentence
        """
        document_embeddings, word_alphas, sentence_alphas = self.sentence_encoder(documents, sentences_per_document, words_per_sentence)
        scores = self.fc(self.dropout(document_embeddings))
        return scores, word_alphas, sentence_alphas


class TextCNN1D(nn.Module):
    """
    Implementation of 1D version of TextCNN proposed in paper [1].

    `Here <https://github.com/yoonkim/CNN_sentence>`_ is the official
    implementation of TextCNN.

    Parameters
    ----------
    n_classes : int
        Number of classes

    vocab_size : int
        Number of words in the vocabulary

    embeddings : torch.Tensor
        Word embedding weights

    emb_size : int
        Size of word embeddings

    fine_tune : bool
        Allow fine-tuning of embedding layer? (only makes sense when using
        pre-trained embeddings)

    n_kernels : int
        Number of kernels

    kernel_sizes : List[int]
        Size of each kernel

    dropout : float
        Dropout

    n_channels : int
        Number of channels (1 / 2)

    References
    ----------
    1. "`Convolutional Neural Networks for Sentence Classification.         <https://www.aclweb.org/anthology/D14-1181.pdf>`_" Yoon Kim. EMNLP 2014.
    """

    def __init__(self, n_classes: int, vocab_size: int, embeddings: torch.Tensor, emb_size: int, fine_tune: bool, n_kernels: int, kernel_sizes: List[int], dropout: float, n_channels=1) ->None:
        super(TextCNN1D, self).__init__()
        self.embedding1 = nn.Embedding(vocab_size, emb_size)
        self.set_embeddings(embeddings, 1, fine_tune)
        if n_channels == 2:
            self.embedding2 = nn.Embedding(vocab_size, emb_size)
            self.set_embeddings(embeddings, 1, False)
        else:
            self.embedding2 = None
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=n_channels, out_channels=n_kernels, kernel_size=size * emb_size, stride=emb_size) for size in kernel_sizes])
        self.fc = nn.Linear(len(kernel_sizes) * n_kernels, n_classes)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def set_embeddings(self, embeddings: torch.Tensor, layer_id: int=1, fine_tune: bool=True) ->None:
        """
        Set weights for embedding layer

        Parameters
        ----------
        embeddings : torch.Tensor
            Word embeddings

        layer_id : int
            Embedding layer 1 or 2 (when adopting multichannel architecture)

        fine_tune : bool, optional, default=True
            Allow fine-tuning of embedding layer? (only makes sense when using
            pre-trained embeddings)
        """
        if embeddings is None:
            if layer_id == 1:
                self.embedding1.weight.data.uniform_(-0.1, 0.1)
            else:
                self.embedding2.weight.data.uniform_(-0.1, 0.1)
        elif layer_id == 1:
            self.embedding1.weight = nn.Parameter(embeddings, requires_grad=fine_tune)
        else:
            self.embedding2.weight = nn.Parameter(embeddings, requires_grad=fine_tune)

    def forward(self, text: torch.Tensor, words_per_sentence: torch.Tensor) ->torch.Tensor:
        """
        Parameters
        ----------
        text : torch.Tensor (batch_size, word_pad_len)
            Input data

        words_per_sentence : torch.Tensor (batch_size)
            Sentence lengths

        Returns
        -------
        scores : torch.Tensor (batch_size, n_classes)
            Class scores
        """
        batch_size = text.size(0)
        embeddings = self.embedding1(text).view(batch_size, 1, -1)
        if self.embedding2:
            embeddings2 = self.embedding2(text).view(batch_size, 1, -1)
            embeddings = torch.cat((embeddings, embeddings2), dim=1)
        conved = [self.relu(conv(embeddings)) for conv in self.convs]
        pooled = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in conved]
        flattened = self.dropout(torch.cat(pooled, dim=1))
        scores = self.fc(flattened)
        return scores


class TextCNN2D(nn.Module):
    """
    Implementation of 2D version of TextCNN proposed in paper [1].

    `Here <https://github.com/yoonkim/CNN_sentence>`_ is the official
    implementation of TextCNN.

    Parameters
    ----------
    n_classes : int
        Number of classes

    vocab_size : int
        Number of words in the vocabulary

    embeddings : torch.Tensor
        Word embedding weights

    emb_size : int
        Size of word embeddings

    fine_tune : bool
        Allow fine-tuning of embedding layer? (only makes sense when using
        pre-trained embeddings)

    n_kernels : int
        Number of kernels

    kernel_sizes : List[int]
        Size of each kernel

    dropout : float
        Dropout

    n_channels : int
        Number of channels (1 / 2)

    References
    ----------
    1. "`Convolutional Neural Networks for Sentence Classification.         <https://www.aclweb.org/anthology/D14-1181.pdf>`_" Yoon Kim. EMNLP 2014.
    """

    def __init__(self, n_classes: int, vocab_size: int, embeddings: torch.Tensor, emb_size: int, fine_tune: bool, n_kernels: int, kernel_sizes: List[int], dropout: float, n_channels=1) ->None:
        super(TextCNN2D, self).__init__()
        self.embedding1 = nn.Embedding(vocab_size, emb_size)
        self.set_embeddings(embeddings, 1, fine_tune)
        if n_channels == 2:
            self.embedding2 = nn.Embedding(vocab_size, emb_size)
            self.set_embeddings(embeddings, 1, False)
        else:
            self.embedding2 = None
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=n_channels, out_channels=n_kernels, kernel_size=(size, emb_size)) for size in kernel_sizes])
        self.fc = nn.Linear(len(kernel_sizes) * n_kernels, n_classes)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def set_embeddings(self, embeddings: torch.Tensor, layer_id: int=1, fine_tune: bool=True) ->None:
        """
        Set weights for embedding layer

        Parameters
        ----------
        embeddings : torch.Tensor
            Word embeddings

        layer_id : int
            Embedding layer 1 or 2 (when adopting multichannel architecture)

        fine_tune : bool, optional, default=True
            Allow fine-tuning of embedding layer? (only makes sense when using
            pre-trained embeddings)
        """
        if embeddings is None:
            if layer_id == 1:
                self.embedding1.weight.data.uniform_(-0.1, 0.1)
            else:
                self.embedding2.weight.data.uniform_(-0.1, 0.1)
        elif layer_id == 1:
            self.embedding1.weight = nn.Parameter(embeddings, requires_grad=fine_tune)
        else:
            self.embedding2.weight = nn.Parameter(embeddings, requires_grad=fine_tune)

    def forward(self, text: torch.Tensor, words_per_sentence: torch.Tensor) ->torch.Tensor:
        """
        Parameters
        ----------
        text : torch.Tensor (batch_size, word_pad_len)
            Input data

        words_per_sentence : torch.Tensor (batch_size)
            Sentence lengths

        Returns
        -------
        scores : torch.Tensor (batch_size, n_classes)
            Class scores
        """
        embeddings = self.embedding1(text).unsqueeze(1)
        if self.embedding2:
            embeddings2 = self.embedding2(text).unsqueeze(1)
            embeddings = torch.cat((embeddings, embeddings2), dim=1)
        conved = [self.relu(conv(embeddings)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in conved]
        flattened = self.dropout(torch.cat(pooled, dim=1))
        scores = self.fc(flattened)
        return scores


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention

    Parameters
    ----------
    scale : float
        Scale factor (sqrt(d_k))

    dropout : float
        Dropout
    """

    def __init__(self, scale: float, dropout: float=0.5) ->None:
        super(ScaledDotProductAttention, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: Optional[torch.Tensor]=None):
        """
        Parameters
        ----------
        Q : torch.Tensor (batch_size, n_heads, word_pad_len, d_k)
            Query

        K : torch.Tensor
            Key

        V : torch.Tensor
            Value

        mask : torch.Tensor (batch_size, 1, 1, word_pad_len)
            Padding mask metrix, None if it is not needed

        Returns
        -------
        context : torch.Tensor (batch_size, n_heads, word_pad_len, d_k)
            Context vector

        att : torch.Tensor (batch_size, n_heads, word_pad_len, word_pad_len)
            Attention weights
        """
        att = torch.matmul(Q / self.scale, K.transpose(2, 3))
        if mask is not None:
            att = att.masked_fill(mask == 0, -1000000000.0)
        att = self.dropout(self.softmax(att))
        context = torch.matmul(att, V)
        return context, att


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention

    Parameters
    ----------
    d_model : int
        Size of word embeddings

    n_heads : int
        Number of attention heads

    dropout : float
        Dropout
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float=0.5) ->None:
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.W_Q = nn.Linear(d_model, n_heads * self.d_k)
        self.W_K = nn.Linear(d_model, n_heads * self.d_k)
        self.W_V = nn.Linear(d_model, n_heads * self.d_k)
        scale = self.d_k ** 0.5
        self.attention = ScaledDotProductAttention(scale=scale)
        self.layer_norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(n_heads * self.d_k, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor (batch_size, word_pad_len, d_model)
            Input data

        mask : torch.Tensor (batch_size, 1, word_pad_len)
            Padding mask metrix, None if it is not needed

        Returns
        -------
        out : torch.Tensor (batch_size, word_pad_len, d_model)
            Output of multi-head self-attention network

        att: torch.Tensor (batch_size, n_heads, word_pad_len, word_pad_len)
            Attention weights
        """
        batch_size = x.size(0)
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k)
        K = K.view(batch_size, -1, self.n_heads, self.d_k)
        V = V.view(batch_size, -1, self.n_heads, self.d_k)
        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)
        if mask is not None:
            mask = mask.unsqueeze(1)
        context, att = self.attention(Q, K, V, mask=mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_k * self.n_heads)
        out = self.dropout(self.fc(context))
        out = out + x
        out = self.layer_norm(out)
        return out, att


class PositionWiseFeedForward(nn.Module):
    """
    Position-Wise Feed-Forward Network

    Parameters
    ----------
    d_model : int
        Size of word embeddings

    hidden_size : int
        Size of position-wise feed forward network

    dropout : float
        Dropout
    """

    def __init__(self, d_model: int, hidden_size: int, dropout: float=0.5) ->None:
        super(PositionWiseFeedForward, self).__init__()
        self.W_1 = nn.Linear(d_model, hidden_size)
        self.W_2 = nn.Linear(hidden_size, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor (batch_size, word_pad_len, d_model)
            Output of multi-head self-attention network

        Returns
        -------
        out : torch.Tensor (batch_size, word_pad_len, d_model)
            Output of position-wise feed-forward network
        """
        out = self.W_2(self.relu(self.W_1(x)))
        out = self.dropout(out)
        out += x
        out = self.layer_norm(out)
        return out


class EncoderLayer(nn.Module):
    """
    An encoder layer.

    Parameters
    ----------
    d_model : int
        Size of word embeddings

    n_heads : int
        Number of attention heads

    hidden_size : int
        Size of position-wise feed forward network

    dropout : float
        Dropout
    """

    def __init__(self, d_model: int, n_heads: int, hidden_size: int, dropout: float=0.5) ->None:
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, hidden_size, dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]=None) ->Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor (batch_size, word_pad_len, d_model)
            Input data

        mask : torch.Tensor (batch_size, 1, word_pad_len)
            Padding mask metrix, None if it is not needed

        Returns
        -------
        out : torch.Tensor (batch_size, word_pad_len, d_model)
            Output of the current encoder layer

        att : torch.Tensor (batch_size, n_heads, word_pad_len, word_pad_len)
            Attention weights
        """
        att_out, att = self.attention(x, mask=mask)
        out = self.feed_forward(att_out)
        return out, att


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PositionalEncoding(nn.Module):
    """
    Positional Encoding

    Parameters
    ----------
    d_model : int
        Size of word embeddings

    word_pad_len : int
        Length of the padded sentence

    dropout : float
        Dropout
    """

    def __init__(self, d_model: int, word_pad_len: int, dropout: float) ->None:
        super(PositionalEncoding, self).__init__()
        self.pe = torch.tensor([[(pos / 10000.0 ** (i // 2 * 2.0 / d_model)) for i in range(d_model)] for pos in range(word_pad_len)])
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])
        self.dropout = nn.Dropout(dropout)

    def forward(self, embeddings: torch.Tensor) ->torch.Tensor:
        """
        Parameters
        ----------
        embeddings : torch.Tensor (batch_size, word_pad_len, emb_size)
            Word embeddings

        Returns
        -------
        position encoded embeddings : torch.Tensor (batch_size, word_pad_len, emb_size)
            Word Embeddings + Positional Encoding
        """
        embeddings = embeddings + nn.Parameter(self.pe, requires_grad=False)
        embeddings = self.dropout(embeddings)
        return embeddings


def get_padding_mask(seq: torch.Tensor, pad_idx: int=0) ->torch.Tensor:
    """
    Mask tokens that are pads (not pad: 1, pad: 0)

    Parameters
    ----------
    seq : torch.Tensor (batch_size, word_pad_len)
        The sequence which needs masking

    pad_idx: index of '<pad>' (default is 0)

    Returns
    -------
    mask : torch.Tensor (batch_size, 1, word_pad_len)
        A padding mask metrix
    """
    mask = (seq != pad_idx).unsqueeze(-2)
    return mask


class Transformer(nn.Module):
    """
    Implementation of Transformer proposed in paper [1]. Only the encoder part
    is used here.

    `Here <https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py>`_
    is the official TensorFlow implementation of Transformer.

    Parameters
    ----------
    n_classes : int
        Number of classes

    vocab_size : int
        Number of words in the vocabulary

    embeddings : torch.Tensor
        Word embedding weights

    d_model : int
        Size of word embeddings

    word_pad_len : int
        Length of the padded sequence

    fine_tune : bool
        Allow fine-tuning of embedding layer? (only makes sense when using
        pre-trained embeddings)

    hidden_size : int
        Size of position-wise feed forward network

    n_heads : int
        Number of attention heads

    n_encoders : int
        Number of encoder layers

    dropout : float
        Dropout

    References
    ----------
    1. "`Attention Is All You Need. <https://arxiv.org/abs/1706.03762>`_"         Ashish Vaswani, et al. NIPS 2017.
    """

    def __init__(self, n_classes: int, vocab_size: int, embeddings: torch.Tensor, d_model: torch.Tensor, word_pad_len: int, fine_tune: bool, hidden_size: int, n_heads: int, n_encoders: int, dropout: float=0.5) ->None:
        super(Transformer, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.set_embeddings(embeddings, fine_tune)
        self.postional_encoding = PositionalEncoding(d_model, word_pad_len, dropout)
        self.encoder = EncoderLayer(d_model, n_heads, hidden_size, dropout)
        self.encoders = nn.ModuleList([copy.deepcopy(self.encoder) for _ in range(n_encoders)])
        self.fc = nn.Linear(word_pad_len * d_model, n_classes)

    def set_embeddings(self, embeddings: torch.Tensor, fine_tune: bool=True) ->None:
        """
        Set weights for embedding layer

        Parameters
        ----------
        embeddings : torch.Tensor
            Word embeddings

        fine_tune : bool, optional, default=True
            Allow fine-tuning of embedding layer? (only makes sense when using
            pre-trained embeddings)
        """
        if embeddings is None:
            self.embeddings.weight.data.uniform_(-0.1, 0.1)
        else:
            self.embeddings.weight = nn.Parameter(embeddings, requires_grad=fine_tune)

    def forward(self, text: torch.Tensor, words_per_sentence: torch.Tensor) ->torch.Tensor:
        """
        Parameters
        ----------
        text : torch.Tensor (batch_size, word_pad_len)
            Input data

        words_per_sentence : torch.Tensor (batch_size)
            Sentence lengths

        Returns
        -------
        scores : torch.Tensor (batch_size, n_classes)
            Class scores
        """
        mask = get_padding_mask(text)
        embeddings = self.embeddings(text)
        embeddings = self.postional_encoding(embeddings)
        encoder_out = embeddings
        for encoder in self.encoders:
            encoder_out, _ = encoder(encoder_out, mask=mask)
        encoder_out = encoder_out.view(encoder_out.size(0), -1)
        scores = self.fc(encoder_out)
        return scores


class fastText(nn.Module):
    """
    Implementation of fastText proposed in paper [1].

    `Here <https://github.com/facebookresearch/fastText>`_ is the official
    implementation of fastText.

    Parameters
    ----------
    n_classes : int
        Number of classes

    vocab_size : int
        Number of words in the vocabulary

    embeddings : torch.Tensor
        Word embedding weights

    emb_size : int
        Size of word embeddings

    fine_tune : bool
        Allow fine-tuning of embedding layer? (only makes sense when using
        pre-trained embeddings)

    hidden_size : int
        Size of the hidden layer

    References
    ----------
    1. "`Bag of Tricks for Efficient Text Classification.         <https://arxiv.org/abs/1607.01759>`_" Armand Joulin, et al. EACL 2017.
    """

    def __init__(self, n_classes: int, vocab_size: int, embeddings: torch.Tensor, emb_size: int, fine_tune: bool, hidden_size: int) ->None:
        super(fastText, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, emb_size)
        self.set_embeddings(embeddings, fine_tune)
        self.hidden = nn.Linear(emb_size, hidden_size)
        self.fc = nn.Linear(hidden_size, n_classes)

    def set_embeddings(self, embeddings: torch.Tensor, fine_tune: bool=True) ->None:
        """
        Set weights for embedding layer

        Parameters
        ----------
        embeddings : torch.Tensor
            Word embeddings

        fine_tune : bool, optional, default=True
            Allow fine-tuning of embedding layer? (only makes sense when using
            pre-trained embeddings)
        """
        if embeddings is None:
            self.embeddings.weight.data.uniform_(-0.1, 0.1)
        else:
            self.embeddings.weight = nn.Parameter(embeddings, requires_grad=fine_tune)

    def forward(self, text: torch.Tensor, words_per_sentence: torch.Tensor) ->torch.Tensor:
        """
        Parameters
        ----------
        text : torch.Tensor (batch_size, word_pad_len)
            Input data

        words_per_sentence : torch.Tensor (batch_size)
            Sentence lengths

        Returns
        -------
        scores : torch.Tensor (batch_size, n_classes)
            Class scores
        """
        embeddings = self.embeddings(text)
        avg_embeddings = embeddings.mean(dim=1).squeeze(1)
        hidden = self.hidden(avg_embeddings)
        scores = self.fc(hidden)
        return scores


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Attention,
     lambda: ([], {'rnn_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (EncoderLayer,
     lambda: ([], {'d_model': 4, 'n_heads': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (MultiHeadAttention,
     lambda: ([], {'d_model': 4, 'n_heads': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     True),
    (PositionWiseFeedForward,
     lambda: ([], {'d_model': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PositionalEncoding,
     lambda: ([], {'d_model': 4, 'word_pad_len': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ScaledDotProductAttention,
     lambda: ([], {'scale': 1.0}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_Renovamen_Text_Classification(_paritybench_base):
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

