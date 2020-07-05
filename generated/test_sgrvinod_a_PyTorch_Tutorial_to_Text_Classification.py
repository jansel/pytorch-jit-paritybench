import sys
_module = sys.modules[__name__]
del sys
classify = _module
create_input_files = _module
datasets = _module
eval = _module
model = _module
train = _module
understanding_packed_sequences = _module
utils = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, string, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
import numpy as np
from torch import Tensor
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch


import torch.nn as nn


from torch.nn.utils.rnn import pack_padded_sequence


from torch.nn.utils.rnn import pad_packed_sequence


from torch.nn.utils.rnn import PackedSequence


from torch import nn


class HierarchialAttentionNetwork(nn.Module):
    """
    The overarching Hierarchial Attention Network (HAN).
    """

    def __init__(self, n_classes, vocab_size, emb_size, word_rnn_size, sentence_rnn_size, word_rnn_layers, sentence_rnn_layers, word_att_size, sentence_att_size, dropout=0.5):
        """
        :param n_classes: number of classes
        :param vocab_size: number of words in the vocabulary of the model
        :param emb_size: size of word embeddings
        :param word_rnn_size: size of (bidirectional) word-level RNN
        :param sentence_rnn_size: size of (bidirectional) sentence-level RNN
        :param word_rnn_layers: number of layers in word-level RNN
        :param sentence_rnn_layers: number of layers in sentence-level RNN
        :param word_att_size: size of word-level attention layer
        :param sentence_att_size: size of sentence-level attention layer
        :param dropout: dropout
        """
        super(HierarchialAttentionNetwork, self).__init__()
        self.sentence_attention = SentenceAttention(vocab_size, emb_size, word_rnn_size, sentence_rnn_size, word_rnn_layers, sentence_rnn_layers, word_att_size, sentence_att_size, dropout)
        self.fc = nn.Linear(2 * sentence_rnn_size, n_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, documents, sentences_per_document, words_per_sentence):
        """
        Forward propagation.

        :param documents: encoded document-level data, a tensor of dimensions (n_documents, sent_pad_len, word_pad_len)
        :param sentences_per_document: document lengths, a tensor of dimensions (n_documents)
        :param words_per_sentence: sentence lengths, a tensor of dimensions (n_documents, sent_pad_len)
        :return: class scores, attention weights of words, attention weights of sentences
        """
        document_embeddings, word_alphas, sentence_alphas = self.sentence_attention(documents, sentences_per_document, words_per_sentence)
        scores = self.fc(self.dropout(document_embeddings))
        return scores, word_alphas, sentence_alphas


class SentenceAttention(nn.Module):
    """
    The sentence-level attention module.
    """

    def __init__(self, vocab_size, emb_size, word_rnn_size, sentence_rnn_size, word_rnn_layers, sentence_rnn_layers, word_att_size, sentence_att_size, dropout):
        """
        :param vocab_size: number of words in the vocabulary of the model
        :param emb_size: size of word embeddings
        :param word_rnn_size: size of (bidirectional) word-level RNN
        :param sentence_rnn_size: size of (bidirectional) sentence-level RNN
        :param word_rnn_layers: number of layers in word-level RNN
        :param sentence_rnn_layers: number of layers in sentence-level RNN
        :param word_att_size: size of word-level attention layer
        :param sentence_att_size: size of sentence-level attention layer
        :param dropout: dropout
        """
        super(SentenceAttention, self).__init__()
        self.word_attention = WordAttention(vocab_size, emb_size, word_rnn_size, word_rnn_layers, word_att_size, dropout)
        self.sentence_rnn = nn.GRU(2 * word_rnn_size, sentence_rnn_size, num_layers=sentence_rnn_layers, bidirectional=True, dropout=dropout, batch_first=True)
        self.sentence_attention = nn.Linear(2 * sentence_rnn_size, sentence_att_size)
        self.sentence_context_vector = nn.Linear(sentence_att_size, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, documents, sentences_per_document, words_per_sentence):
        """
        Forward propagation.

        :param documents: encoded document-level data, a tensor of dimensions (n_documents, sent_pad_len, word_pad_len)
        :param sentences_per_document: document lengths, a tensor of dimensions (n_documents)
        :param words_per_sentence: sentence lengths, a tensor of dimensions (n_documents, sent_pad_len)
        :return: document embeddings, attention weights of words, attention weights of sentences
        """
        packed_sentences = pack_padded_sequence(documents, lengths=sentences_per_document.tolist(), batch_first=True, enforce_sorted=False)
        packed_words_per_sentence = pack_padded_sequence(words_per_sentence, lengths=sentences_per_document.tolist(), batch_first=True, enforce_sorted=False)
        sentences, word_alphas = self.word_attention(packed_sentences.data, packed_words_per_sentence.data)
        sentences = self.dropout(sentences)
        packed_sentences, _ = self.sentence_rnn(PackedSequence(data=sentences, batch_sizes=packed_sentences.batch_sizes, sorted_indices=packed_sentences.sorted_indices, unsorted_indices=packed_sentences.unsorted_indices))
        att_s = self.sentence_attention(packed_sentences.data)
        att_s = torch.tanh(att_s)
        att_s = self.sentence_context_vector(att_s).squeeze(1)
        max_value = att_s.max()
        att_s = torch.exp(att_s - max_value)
        att_s, _ = pad_packed_sequence(PackedSequence(data=att_s, batch_sizes=packed_sentences.batch_sizes, sorted_indices=packed_sentences.sorted_indices, unsorted_indices=packed_sentences.unsorted_indices), batch_first=True)
        sentence_alphas = att_s / torch.sum(att_s, dim=1, keepdim=True)
        documents, _ = pad_packed_sequence(packed_sentences, batch_first=True)
        documents = documents * sentence_alphas.unsqueeze(2)
        documents = documents.sum(dim=1)
        word_alphas, _ = pad_packed_sequence(PackedSequence(data=word_alphas, batch_sizes=packed_sentences.batch_sizes, sorted_indices=packed_sentences.sorted_indices, unsorted_indices=packed_sentences.unsorted_indices), batch_first=True)
        return documents, word_alphas, sentence_alphas


class WordAttention(nn.Module):
    """
    The word-level attention module.
    """

    def __init__(self, vocab_size, emb_size, word_rnn_size, word_rnn_layers, word_att_size, dropout):
        """
        :param vocab_size: number of words in the vocabulary of the model
        :param emb_size: size of word embeddings
        :param word_rnn_size: size of (bidirectional) word-level RNN
        :param word_rnn_layers: number of layers in word-level RNN
        :param word_att_size: size of word-level attention layer
        :param dropout: dropout
        """
        super(WordAttention, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, emb_size)
        self.word_rnn = nn.GRU(emb_size, word_rnn_size, num_layers=word_rnn_layers, bidirectional=True, dropout=dropout, batch_first=True)
        self.word_attention = nn.Linear(2 * word_rnn_size, word_att_size)
        self.word_context_vector = nn.Linear(word_att_size, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def init_embeddings(self, embeddings):
        """
        Initialized embedding layer with pre-computed embeddings.

        :param embeddings: pre-computed embeddings
        """
        self.embeddings.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=False):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).

        :param fine_tune: allow?
        """
        for p in self.embeddings.parameters():
            p.requires_grad = fine_tune

    def forward(self, sentences, words_per_sentence):
        """
        Forward propagation.

        :param sentences: encoded sentence-level data, a tensor of dimension (n_sentences, word_pad_len, emb_size)
        :param words_per_sentence: sentence lengths, a tensor of dimension (n_sentences)
        :return: sentence embeddings, attention weights of words
        """
        sentences = self.dropout(self.embeddings(sentences))
        packed_words = pack_padded_sequence(sentences, lengths=words_per_sentence.tolist(), batch_first=True, enforce_sorted=False)
        packed_words, _ = self.word_rnn(packed_words)
        att_w = self.word_attention(packed_words.data)
        att_w = torch.tanh(att_w)
        att_w = self.word_context_vector(att_w).squeeze(1)
        max_value = att_w.max()
        att_w = torch.exp(att_w - max_value)
        att_w, _ = pad_packed_sequence(PackedSequence(data=att_w, batch_sizes=packed_words.batch_sizes, sorted_indices=packed_words.sorted_indices, unsorted_indices=packed_words.unsorted_indices), batch_first=True)
        word_alphas = att_w / torch.sum(att_w, dim=1, keepdim=True)
        sentences, _ = pad_packed_sequence(packed_words, batch_first=True)
        sentences = sentences * word_alphas.unsqueeze(2)
        sentences = sentences.sum(dim=1)
        return sentences, word_alphas

