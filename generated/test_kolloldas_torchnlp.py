import sys
_module = sys.modules[__name__]
del sys
setup = _module
test_evaluation = _module
test_model = _module
test_train = _module
test_crf = _module
test_outputs = _module
test_transformer = _module
test_sequence_tagging = _module
torchnlp = _module
chunk = _module
common = _module
cmd = _module
evaluation = _module
hparams = _module
info = _module
model = _module
prefs = _module
train = _module
data = _module
conll = _module
inputs = _module
nyt = _module
modules = _module
crf = _module
normalization = _module
outputs = _module
transformer = _module
layers = _module
main = _module
sublayers = _module
ner = _module
tasks = _module
sequence_tagging = _module
bilstm_tagger = _module
main = _module
tagger = _module
transformer_tagger = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import torch


import torch.nn as nn


from time import sleep


import numpy as np


import torchtext


from torchtext import data


from torchtext import datasets


import logging


from functools import partial


from functools import reduce


import torch.optim as optim


from collections import deque


from collections import defaultdict


from torchtext.datasets import SequenceTaggingDataset


from torchtext.datasets import CoNLL2000Chunking


from torchtext.vocab import Vectors


from torchtext.vocab import GloVe


from torchtext.vocab import CharNGram


import random


import torch.nn.functional as F


import math


CHECKPOINT_FILE = 'checkpoint-{}.pt'


CHECKPOINT_GLOB = 'checkpoint-*.pt'


HYPERPARAMS_FILE = 'hyperparams.pt'


def gen_model_dir(task_name, model_cls):
    """
    Generate the model directory from the task name and model class.
    Creat if not exists. 
    Parameters:
        task_name: Name of the task. Gets prefixed to model directory
        model_cls: The models class (derived from Model)
    """
    model_dir = os.path.join(os.getcwd(), '%s-%s' % (task_name, model_cls.__name__))
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    return model_dir


logger = logging.getLogger(__name__)


def prepare_model_dir(model_dir, clear=False):
    """
    Prepares the model directory. If clear is set to True, deletes all files, else
    renames existing directory and creates a fresh one
    Parameters:
        model_dir: Absolute path of the model directory to prepare
    """
    p = list(os.walk(model_dir))
    if clear:
        for file in os.listdir(model_dir):
            path = os.path.join(model_dir, file)
            if os.path.isfile(path):
                try:
                    os.remove(path)
                except:
                    logger.warning('WARNING: Failed to delete {}'.format(path))
    elif len(p[0][2]) > 0:
        for i in range(1, 10):
            try:
                rename_dir = '{}-{}'.format(model_dir, i)
                if not os.path.exists(rename_dir):
                    os.rename(model_dir, rename_dir)
                    os.mkdir(model_dir)
                    break
            except:
                pass


def xavier_uniform_init(m):
    """
    Xavier initializer to be used with model.apply
    """
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight.data)


class Model(nn.Module):
    """
    Abstract base class that defines a loss() function
    """

    def __init__(self, hparams=None):
        super(Model, self).__init__()
        if hparams is None:
            raise ValueError('Must provide hparams')
        self.hparams = hparams
        self.iterations = nn.Parameter(torch.LongTensor([0]), requires_grad=False)

    def loss(self, batch, compute_predictions=False):
        """
        Called by train.Trainer to compute negative log likelihood
        Parameters:
            batch: A minibatch, instance of torchtext.data.Batch
            compute_predictions: If true compute and provide predictions, else None
        Returns:
            Tuple of loss and predictions
        """
        raise NotImplementedError('Must implement loss()')

    @classmethod
    def create(cls, task_name, hparams, overwrite=False, **kwargs):
        """
        Create a new instance of this class. Prepares the model directory
        and saves hyperparams. Derived classes should override this function
        to save other dependencies (e.g. vocabs)
        """
        logger.info(hparams)
        model_dir = gen_model_dir(task_name, cls)
        model = cls(hparams, **kwargs)
        model.apply(xavier_uniform_init)
        if torch.cuda.is_available():
            model = model
        prepare_model_dir(model_dir, overwrite)
        torch.save(hparams, os.path.join(model_dir, HYPERPARAMS_FILE))
        return model

    @classmethod
    def load(cls, task_name, checkpoint, **kwargs):
        """
        Loads a model from a checkpoint. Also loads hyperparams
        Parameters:
            task_name: Name of the task. Needed to determine model dir
            checkpoint: Number indicating the checkpoint. -1 to load latest
            **kwargs: Additional key-value args passed to constructor
        """
        model_dir = gen_model_dir(task_name, cls)
        hparams_path = os.path.join(model_dir, HYPERPARAMS_FILE)
        if not os.path.exists(hparams_path):
            raise OSError('HParams file not found')
        hparams = torch.load(hparams_path)
        logger.info('Hyperparameters: {}'.format(str(hparams)))
        model = cls(hparams, **kwargs)
        if torch.cuda.is_available():
            model = model
        if checkpoint == -1:
            files = glob.glob(os.path.join(model_dir, CHECKPOINT_GLOB))
            if not files:
                raise OSError('Checkpoint files not found')
            files.sort(key=os.path.getmtime, reverse=True)
            checkpoint_path = files[0]
        else:
            checkpoint_path = os.path.join(model_dir, CHECKPOINT_FILE.format(checkpoint))
            if not os.path.exists(checkpoint_path):
                raise OSError('File not found: {}'.format(checkpoint_path))
        logger.info('Loading from {}'.format(checkpoint_path))
        model.load_state_dict(torch.load(checkpoint_path))
        return model, hparams

    def save(self, task_name):
        """
        Save the model. Directory is determined by the task name and model class name
        """
        model_dir = gen_model_dir(task_name, self.__class__)
        checkpoint_path = os.path.join(model_dir, CHECKPOINT_FILE.format(int(self.iterations)))
        torch.save(self.state_dict(), checkpoint_path)
        logger.info('-------------- Saved checkpoint {} --------------'.format(int(self.iterations)))

    def set_latest(self, task_name, iteration):
        """
        Set the modified time of the checkpoint to latest. Used to set the best
        checkpoint
        """
        model_dir = gen_model_dir(task_name, self.__class__)
        checkpoint_path = os.path.join(model_dir, CHECKPOINT_FILE.format(int(iteration)))
        os.utime(checkpoint_path, None)


class CRF(nn.Module):
    """
    Implements Conditional Random Fields that can be trained via
    backpropagation. 
    """

    def __init__(self, num_tags):
        super(CRF, self).__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.Tensor(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.stop_transitions = nn.Parameter(torch.randn(num_tags))
        nn.init.xavier_normal_(self.transitions)

    def forward(self, feats):
        if len(feats.shape) != 3:
            raise ValueError('feats must be 3-d got {}-d'.format(feats.shape))
        return self._viterbi(feats)

    def loss(self, feats, tags):
        """
        Computes negative log likelihood between features and tags.
        Essentially difference between individual sequence scores and 
        sum of all possible sequence scores (partition function)
        Parameters:
            feats: Input features [batch size, sequence length, number of tags]
            tags: Target tag indices [batch size, sequence length]. Should be between
                    0 and num_tags
        Returns:
            Negative log likelihood [a scalar] 
        """
        if len(feats.shape) != 3:
            raise ValueError('feats must be 3-d got {}-d'.format(feats.shape))
        if len(tags.shape) != 2:
            raise ValueError('tags must be 2-d but got {}-d'.format(tags.shape))
        if feats.shape[:2] != tags.shape:
            raise ValueError('First two dimensions of feats and tags must match')
        sequence_score = self._sequence_score(feats, tags)
        partition_function = self._partition_function(feats)
        log_probability = sequence_score - partition_function
        return -log_probability.mean()

    def _sequence_score(self, feats, tags):
        """
        Parameters:
            feats: Input features [batch size, sequence length, number of tags]
            tags: Target tag indices [batch size, sequence length]. Should be between
                    0 and num_tags
        Returns: Sequence score of shape [batch size]
        """
        batch_size = feats.shape[0]
        feat_score = feats.gather(2, tags.unsqueeze(-1)).squeeze(-1).sum(dim=-1)
        tags_pairs = tags.unfold(1, 2, 1)
        indices = tags_pairs.permute(2, 0, 1).chunk(2)
        trans_score = self.transitions[indices].squeeze(0).sum(dim=-1)
        start_score = self.start_transitions[tags[:, (0)]]
        stop_score = self.stop_transitions[tags[:, (-1)]]
        return feat_score + start_score + trans_score + stop_score

    def _partition_function(self, feats):
        """
        Computes the partitition function for CRF using the forward algorithm.
        Basically calculate scores for all possible tag sequences for 
        the given feature vector sequence
        Parameters:
            feats: Input features [batch size, sequence length, number of tags]
        Returns:
            Total scores of shape [batch size]
        """
        _, seq_size, num_tags = feats.shape
        if self.num_tags != num_tags:
            raise ValueError('num_tags should be {} but got {}'.format(self.num_tags, num_tags))
        a = feats[:, (0)] + self.start_transitions.unsqueeze(0)
        transitions = self.transitions.unsqueeze(0)
        for i in range(1, seq_size):
            feat = feats[:, (i)].unsqueeze(1)
            a = self._log_sum_exp(a.unsqueeze(-1) + transitions + feat, 1)
        return self._log_sum_exp(a + self.stop_transitions.unsqueeze(0), 1)

    def _viterbi(self, feats):
        """
        Uses Viterbi algorithm to predict the best sequence
        Parameters:
            feats: Input features [batch size, sequence length, number of tags]
        Returns: Best tag sequence [batch size, sequence length]
        """
        _, seq_size, num_tags = feats.shape
        if self.num_tags != num_tags:
            raise ValueError('num_tags should be {} but got {}'.format(self.num_tags, num_tags))
        v = feats[:, (0)] + self.start_transitions.unsqueeze(0)
        transitions = self.transitions.unsqueeze(0)
        paths = []
        for i in range(1, seq_size):
            feat = feats[:, (i)]
            v, idx = (v.unsqueeze(-1) + transitions).max(1)
            paths.append(idx)
            v = v + feat
        v, tag = (v + self.stop_transitions.unsqueeze(0)).max(1, True)
        tags = [tag]
        for idx in reversed(paths):
            tag = idx.gather(1, tag)
            tags.append(tag)
        tags.reverse()
        return torch.cat(tags, 1)

    def _log_sum_exp(self, logits, dim):
        """
        Computes log-sum-exp in a stable way
        """
        max_val, _ = logits.max(dim)
        return max_val + (logits - max_val.unsqueeze(dim)).exp().sum(dim).log()


class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-06):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class OutputLayer(nn.Module):
    """
    Abstract base class for output layer. 
    Handles projection to output labels
    """

    def __init__(self, hidden_size, output_size):
        super(OutputLayer, self).__init__()
        self.output_size = output_size
        self.output_projection = nn.Linear(hidden_size, output_size)

    def loss(self, hidden, labels):
        raise NotImplementedError('Must implement {}.loss'.format(self.__class__.__name__))


class SoftmaxOutputLayer(OutputLayer):
    """
    Implements a softmax based output layer
    """

    def forward(self, hidden):
        logits = self.output_projection(hidden)
        probs = F.softmax(logits, -1)
        _, predictions = torch.max(probs, dim=-1)
        return predictions

    def loss(self, hidden, labels):
        logits = self.output_projection(hidden)
        log_probs = F.log_softmax(logits, -1)
        return F.nll_loss(log_probs.view(-1, self.output_size), labels.view(-1))


class CRFOutputLayer(OutputLayer):
    """
    Implements a CRF based output layer
    """

    def __init__(self, hidden_size, output_size):
        super(CRFOutputLayer, self).__init__(hidden_size, output_size)
        self.crf = CRF(output_size)

    def forward(self, hidden):
        feats = self.output_projection(hidden)
        return self.crf(feats)

    def loss(self, hidden, labels):
        feats = self.output_projection(hidden)
        return self.crf.loss(feats, labels)


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention as per https://arxiv.org/pdf/1706.03762.pdf
    Refer Figure 2
    """

    def __init__(self, input_depth, total_key_depth, total_value_depth, output_depth, num_heads, bias_mask=None, dropout=0.0):
        """
        Parameters:
            input_depth: Size of last dimension of input
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            num_heads: Number of attention heads
            bias_mask: Masking tensor to prevent connections to future elements
            dropout: Dropout probability (Should be non-zero only during training)
        """
        super(MultiHeadAttention, self).__init__()
        if total_key_depth % num_heads != 0:
            raise ValueError('Key depth (%d) must be divisible by the number of attention heads (%d).' % (total_key_depth, num_heads))
        if total_value_depth % num_heads != 0:
            raise ValueError('Value depth (%d) must be divisible by the number of attention heads (%d).' % (total_value_depth, num_heads))
        self.num_heads = num_heads
        self.query_scale = (total_key_depth // num_heads) ** -0.5
        self.bias_mask = bias_mask
        self.query_linear = nn.Linear(input_depth, total_key_depth, bias=False)
        self.key_linear = nn.Linear(input_depth, total_key_depth, bias=False)
        self.value_linear = nn.Linear(input_depth, total_value_depth, bias=False)
        self.output_linear = nn.Linear(total_value_depth, output_depth, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        """
        Split x such to add an extra num_heads dimension
        Input:
            x: a Tensor with shape [batch_size, seq_length, depth]
        Returns:
            A Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads]
        """
        if len(x.shape) != 3:
            raise ValueError('x must have rank 3')
        shape = x.shape
        return x.view(shape[0], shape[1], self.num_heads, shape[2] // self.num_heads).permute(0, 2, 1, 3)

    def _merge_heads(self, x):
        """
        Merge the extra num_heads into the last dimension
        Input:
            x: a Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads]
        Returns:
            A Tensor with shape [batch_size, seq_length, depth]
        """
        if len(x.shape) != 4:
            raise ValueError('x must have rank 4')
        shape = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(shape[0], shape[2], shape[3] * self.num_heads)

    def forward(self, queries, keys, values):
        queries = self.query_linear(queries)
        keys = self.key_linear(keys)
        values = self.value_linear(values)
        queries = self._split_heads(queries)
        keys = self._split_heads(keys)
        values = self._split_heads(values)
        queries *= self.query_scale
        logits = torch.matmul(queries, keys.permute(0, 1, 3, 2))
        if self.bias_mask is not None:
            logits += self.bias_mask[:, :, :logits.shape[-2], :logits.shape[-1]].type_as(logits.data)
        weights = nn.functional.softmax(logits, dim=-1)
        weights = self.dropout(weights)
        contexts = torch.matmul(weights, values)
        contexts = self._merge_heads(contexts)
        outputs = self.output_linear(contexts)
        return outputs


class Conv(nn.Module):
    """
    Convenience class that does padding and convolution for inputs in the format
    [batch_size, sequence length, hidden size]
    """

    def __init__(self, input_size, output_size, kernel_size, pad_type):
        """
        Parameters:
            input_size: Input feature size
            output_size: Output feature size
            kernel_size: Kernel width
            pad_type: left -> pad on the left side (to mask future data), 
                      both -> pad on both sides
        """
        super(Conv, self).__init__()
        padding = (kernel_size - 1, 0) if pad_type == 'left' else (kernel_size // 2, (kernel_size - 1) // 2)
        self.pad = nn.ConstantPad1d(padding, 0)
        self.conv = nn.Conv1d(input_size, output_size, kernel_size=kernel_size, padding=0)

    def forward(self, inputs):
        inputs = self.pad(inputs.permute(0, 2, 1))
        outputs = self.conv(inputs).permute(0, 2, 1)
        return outputs


class PositionwiseFeedForward(nn.Module):
    """
    Does a Linear + RELU + Linear on each of the timesteps
    """

    def __init__(self, input_depth, filter_size, output_depth, layer_config='ll', padding='left', dropout=0.0):
        """
        Parameters:
            input_depth: Size of last dimension of input
            filter_size: Hidden size of the middle layer
            output_depth: Size last dimension of the final output
            layer_config: ll -> linear + ReLU + linear
                          cc -> conv + ReLU + conv etc.
            padding: left -> pad on the left side (to mask future data), 
                     both -> pad on both sides
            dropout: Dropout probability (Should be non-zero only during training)
        """
        super(PositionwiseFeedForward, self).__init__()
        layers = []
        sizes = [(input_depth, filter_size)] + [(filter_size, filter_size)] * (len(layer_config) - 2) + [(filter_size, output_depth)]
        for lc, s in zip(list(layer_config), sizes):
            if lc == 'l':
                layers.append(nn.Linear(*s))
            elif lc == 'c':
                layers.append(Conv(*s, kernel_size=3, pad_type=padding))
            else:
                raise ValueError('Unknown layer type {}'.format(lc))
        self.layers = nn.ModuleList(layers)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        x = inputs
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers):
                x = self.relu(x)
                x = self.dropout(x)
        return x


class EncoderLayer(nn.Module):
    """
    Represents one Encoder layer of the Transformer Encoder
    Refer Fig. 1 in https://arxiv.org/pdf/1706.03762.pdf
    NOTE: The layer normalization step has been moved to the input as per latest version of T2T
    """

    def __init__(self, hidden_size, total_key_depth, total_value_depth, filter_size, num_heads, bias_mask=None, layer_dropout=0.0, attention_dropout=0.0, relu_dropout=0.0):
        """
        Parameters:
            hidden_size: Hidden size
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            num_heads: Number of attention heads
            bias_mask: Masking tensor to prevent connections to future elements
            layer_dropout: Dropout for this layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
        """
        super(EncoderLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(hidden_size, total_key_depth, total_value_depth, hidden_size, num_heads, bias_mask, attention_dropout)
        self.positionwise_feed_forward = PositionwiseFeedForward(hidden_size, filter_size, hidden_size, layer_config='cc', padding='both', dropout=relu_dropout)
        self.dropout = nn.Dropout(layer_dropout)
        self.layer_norm_mha = LayerNorm(hidden_size)
        self.layer_norm_ffn = LayerNorm(hidden_size)

    def forward(self, inputs):
        x = inputs
        x_norm = self.layer_norm_mha(x)
        y = self.multi_head_attention(x_norm, x_norm, x_norm)
        x = self.dropout(x + y)
        x_norm = self.layer_norm_ffn(x)
        y = self.positionwise_feed_forward(x_norm)
        y = self.dropout(x + y)
        return y


class DecoderLayer(nn.Module):
    """
    Represents one Decoder layer of the Transformer Decoder
    Refer Fig. 1 in https://arxiv.org/pdf/1706.03762.pdf
    NOTE: The layer normalization step has been moved to the input as per latest version of T2T
    """

    def __init__(self, hidden_size, total_key_depth, total_value_depth, filter_size, num_heads, bias_mask, layer_dropout=0.0, attention_dropout=0.0, relu_dropout=0.0):
        """
        Parameters:
            hidden_size: Hidden size
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            num_heads: Number of attention heads
            bias_mask: Masking tensor to prevent connections to future elements
            layer_dropout: Dropout for this layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
        """
        super(DecoderLayer, self).__init__()
        self.multi_head_attention_dec = MultiHeadAttention(hidden_size, total_key_depth, total_value_depth, hidden_size, num_heads, bias_mask, attention_dropout)
        self.multi_head_attention_enc_dec = MultiHeadAttention(hidden_size, total_key_depth, total_value_depth, hidden_size, num_heads, None, attention_dropout)
        self.positionwise_feed_forward = PositionwiseFeedForward(hidden_size, filter_size, hidden_size, layer_config='cc', padding='left', dropout=relu_dropout)
        self.dropout = nn.Dropout(layer_dropout)
        self.layer_norm_mha_dec = LayerNorm(hidden_size)
        self.layer_norm_mha_enc = LayerNorm(hidden_size)
        self.layer_norm_ffn = LayerNorm(hidden_size)

    def forward(self, inputs):
        """
        NOTE: Inputs is a tuple consisting of decoder inputs and encoder output
        """
        x, encoder_outputs = inputs
        x_norm = self.layer_norm_mha_dec(x)
        y = self.multi_head_attention_dec(x_norm, x_norm, x_norm)
        x = self.dropout(x + y)
        x_norm = self.layer_norm_mha_enc(x)
        y = self.multi_head_attention_enc_dec(x_norm, encoder_outputs, encoder_outputs)
        x = self.dropout(x + y)
        x_norm = self.layer_norm_ffn(x)
        y = self.positionwise_feed_forward(x_norm)
        y = self.dropout(x + y)
        return y, encoder_outputs


def _gen_bias_mask(max_length):
    """
    Generates bias values (-Inf) to mask future timesteps during attention
    """
    np_mask = np.triu(np.full([max_length, max_length], -np.inf), 1)
    torch_mask = torch.from_numpy(np_mask).type(torch.FloatTensor)
    return torch_mask.unsqueeze(0).unsqueeze(1)


def _gen_timing_signal(length, channels, min_timescale=1.0, max_timescale=10000.0):
    """
    Generates a [1, length, channels] timing signal consisting of sinusoids
    Adapted from:
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
    """
    position = np.arange(length)
    num_timescales = channels // 2
    log_timescale_increment = math.log(float(max_timescale) / float(min_timescale)) / (float(num_timescales) - 1)
    inv_timescales = min_timescale * np.exp(np.arange(num_timescales).astype(np.float) * -log_timescale_increment)
    scaled_time = np.expand_dims(position, 1) * np.expand_dims(inv_timescales, 0)
    signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    signal = np.pad(signal, [[0, 0], [0, channels % 2]], 'constant', constant_values=[0.0, 0.0])
    signal = signal.reshape([1, length, channels])
    return torch.from_numpy(signal).type(torch.FloatTensor)


class Encoder(nn.Module):
    """
    A Transformer Encoder module. 
    Inputs should be in the shape [batch_size, length, hidden_size]
    Outputs will have the shape [batch_size, length, hidden_size]
    Refer Fig.1 in https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self, embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth, filter_size, max_length=100, input_dropout=0.0, layer_dropout=0.0, attention_dropout=0.0, relu_dropout=0.0, use_mask=False):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder
            num_heads: Number of attention heads
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
            use_mask: Set to True to turn on future value masking
        """
        super(Encoder, self).__init__()
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)
        params = hidden_size, total_key_depth or hidden_size, total_value_depth or hidden_size, filter_size, num_heads, _gen_bias_mask(max_length) if use_mask else None, layer_dropout, attention_dropout, relu_dropout
        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        self.enc = nn.Sequential(*[EncoderLayer(*params) for l in range(num_layers)])
        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

    def forward(self, inputs):
        x = self.input_dropout(inputs)
        x = self.embedding_proj(x)
        x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)
        y = self.enc(x)
        y = self.layer_norm(y)
        return y


class Decoder(nn.Module):
    """
    A Transformer Decoder module. 
    Inputs should be in the shape [batch_size, length, hidden_size]
    Outputs will have the shape [batch_size, length, hidden_size]
    Refer Fig.1 in https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self, embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth, filter_size, max_length=100, input_dropout=0.0, layer_dropout=0.0, attention_dropout=0.0, relu_dropout=0.0):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder
            num_heads: Number of attention heads
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
        """
        super(Decoder, self).__init__()
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)
        params = hidden_size, total_key_depth or hidden_size, total_value_depth or hidden_size, filter_size, num_heads, _gen_bias_mask(max_length), layer_dropout, attention_dropout, relu_dropout
        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        self.dec = nn.Sequential(*[DecoderLayer(*params) for l in range(num_layers)])
        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

    def forward(self, inputs, encoder_output):
        x = self.input_dropout(inputs)
        x = self.embedding_proj(x)
        x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)
        y, _ = self.dec((x, encoder_output))
        y = self.layer_norm(y)
        return y


VOCABS_FILE = 'vocabs.pt'


class Tagger(Model):
    """
    Abstract base class that adds the following boilerplate for
    sequence tagging tasks:
    - Word Embeddings
    - Character Embeddings
    - Tag projection
    - CRF
    Derived classes implement the compute() method and not forward(). 
    This is so that projection and other layers can be added
    """

    def __init__(self, hparams=None, vocabs=None):
        """
        Parameters:
            hparams: Instance of HParams class
            num_tags: Number of output tags
            vocabs: tuple of (word vocab, char vocab, tags vocab). Each is an
                    instance of torchtext.vocab.Vocab.
            NOTE: If word_vocab.vectors is available it will initialize the embeddings
            and with word_vocab.vectors make it non-trainable
        """
        super(Tagger, self).__init__(hparams)
        if vocabs is None or not isinstance(vocabs, tuple) or len(vocabs) != 3:
            raise ValueError('Must provide vocabs 3-tuple')
        vocab_word, vocab_char, vocab_tags = vocabs
        if vocab_word is None:
            raise ValueError('Must provide vocab_word')
        if vocab_tags is None:
            raise ValueError('Must provide vocab_word')
        self.vocabs = vocabs
        self.vocab_tags = vocab_tags
        self.embedding_word = nn.Embedding(len(vocab_word), hparams.embedding_size_word)
        self.embedding_char = None
        if vocab_char is not None and hparams.embedding_size_char > 0:
            self.embedding_char = nn.Embedding(len(vocab_char), hparams.embedding_size_char)
        if vocab_word.vectors is not None:
            if hparams.embedding_size_word != vocab_word.vectors.shape[1]:
                raise ValueError('embedding_size should be {} but got {}'.format(vocab_word.vectors.shape[1], hparams.embedding_size_word))
            self.embedding_word.weight.data.copy_(vocab_word.vectors)
            self.embedding_word.weight.requires_grad = False
        if hparams.use_crf:
            self.output_layer = outputs.CRFOutputLayer(hparams.hidden_size, len(vocab_tags))
        else:
            self.output_layer = outputs.SoftmaxOutputLayer(hparams.hidden_size, len(vocab_tags))

    def _embed_compute(self, batch):
        inputs_word_emb = self.embedding_word(batch.inputs_word)
        inputs_char_emb = None
        if self.embedding_char is not None:
            inputs_char_emb = self.embedding_char(batch.inputs_char.view(-1, batch.inputs_char.shape[-1]))
        return self.compute(inputs_word_emb, inputs_char_emb)

    def forward(self, batch):
        """
        NOTE: batch must have the following attributes:
            inputs_word, inputs_char, labels
        """
        with torch.no_grad():
            hidden = self._embed_compute(batch)
            output = self.output_layer(hidden)
        return output

    def loss(self, batch, compute_predictions=False):
        """
        NOTE: batch must have the following attributes:
            inputs_word, inputs_char, labels
        """
        hidden = self._embed_compute(batch)
        predictions = None
        if compute_predictions:
            predictions = self.output_layer(hidden)
        loss_val = self.output_layer.loss(hidden, batch.labels)
        return loss_val, predictions

    def compute(self, inputs_word_emb, inputs_char_emb):
        """
        Abstract method that is called to compute the final model
        hidden state. Derived classes implement the method to take
        input embeddings and provide the final hidden state

        Parameters:
            inputs_word_emb: Input word embeddings of shape
                                [batch, sequence-length, word-embedding-size]
            inputs_char_emb[optional]: Input character embeddings of shape
                                [batch x sequence-length, word-length, char-embedding-size]

        Returns:
            Final hidden state in the shape [batch, sequence-length, hidden-size]
        """
        raise NotImplementedError('Must implement compute()')

    @classmethod
    def create(cls, task_name, hparams, vocabs, **kwargs):
        """
        Saves the vocab files
        """
        model = super(Tagger, cls).create(task_name, hparams, vocabs=vocabs, **kwargs)
        model_dir = gen_model_dir(task_name, cls)
        torch.save(vocabs, os.path.join(model_dir, VOCABS_FILE))
        return model

    @classmethod
    def load(cls, task_name, checkpoint, **kwargs):
        model_dir = gen_model_dir(task_name, cls)
        vocabs_path = os.path.join(model_dir, VOCABS_FILE)
        if not os.path.exists(vocabs_path):
            raise OSError('Vocabs file not found')
        vocabs = torch.load(vocabs_path)
        return super(Tagger, cls).load(task_name, checkpoint, vocabs=vocabs, **kwargs)


class TransformerTagger(Tagger):
    """
    Sequence tagger using the Transformer network (https://arxiv.org/pdf/1706.03762.pdf)
    Specifically it uses the Encoder module. For character embeddings (per word) it uses
    the same Encoder module above which an additive (Bahdanau) self-attention layer is added
    """

    def __init__(self, hparams=None, **kwargs):
        """
        No additional parameters
        """
        super(TransformerTagger, self).__init__(hparams=hparams, **kwargs)
        embedding_size = hparams.embedding_size_word
        if hparams.embedding_size_char > 0:
            embedding_size += hparams.embedding_size_char_per_word
            self.transformer_char = transformer.Encoder(hparams.embedding_size_char, hparams.embedding_size_char_per_word, 1, 4, hparams.attention_key_channels, hparams.attention_value_channels, hparams.filter_size_char, hparams.max_length, hparams.input_dropout, hparams.dropout, hparams.attention_dropout, hparams.relu_dropout, use_mask=False)
            self.char_linear = nn.Linear(hparams.embedding_size_char_per_word, 1)
        self.transformer_enc = transformer.Encoder(embedding_size, hparams.hidden_size, hparams.num_hidden_layers, hparams.num_heads, hparams.attention_key_channels, hparams.attention_value_channels, hparams.filter_size, hparams.max_length, hparams.input_dropout, hparams.dropout, hparams.attention_dropout, hparams.relu_dropout, use_mask=False)

    def compute(self, inputs_word_emb, inputs_char_emb):
        if inputs_char_emb is not None:
            seq_len = inputs_word_emb.shape[1]
            inputs_char_emb = self.transformer_char(inputs_char_emb)
            mask = self.char_linear(inputs_char_emb)
            mask = F.softmax(mask, dim=-1)
            inputs_emb_char = torch.matmul(mask.permute(0, 2, 1), inputs_char_emb).contiguous().view(-1, seq_len, self.hparams.embedding_size_char_per_word)
            inputs_word_emb = torch.cat([inputs_word_emb, inputs_emb_char], -1)
        enc_out = self.transformer_enc(inputs_word_emb)
        return enc_out


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (CRF,
     lambda: ([], {'num_tags': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (CRFOutputLayer,
     lambda: ([], {'hidden_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (Conv,
     lambda: ([], {'input_size': 4, 'output_size': 4, 'kernel_size': 4, 'pad_type': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (LayerNorm,
     lambda: ([], {'features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PositionwiseFeedForward,
     lambda: ([], {'input_depth': 1, 'filter_size': 4, 'output_depth': 1}),
     lambda: ([torch.rand([1, 1])], {}),
     False),
    (SoftmaxOutputLayer,
     lambda: ([], {'hidden_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_kolloldas_torchnlp(_paritybench_base):
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

