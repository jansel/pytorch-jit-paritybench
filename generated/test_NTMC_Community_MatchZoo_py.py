import sys
_module = sys.modules[__name__]
del sys
conf = _module
matchzoo = _module
auto = _module
preparer = _module
prepare = _module
tuner = _module
tune = _module
tuner = _module
data_pack = _module
pack = _module
dataloader = _module
callbacks = _module
histogram = _module
lambda_callback = _module
ngram = _module
padding = _module
dataloader = _module
dataloader_builder = _module
dataset = _module
dataset_builder = _module
datasets = _module
embeddings = _module
load_fasttext_embedding = _module
load_glove_embedding = _module
quora_qp = _module
load_data = _module
snli = _module
toy = _module
wiki_qa = _module
embedding = _module
engine = _module
base_callback = _module
base_metric = _module
base_model = _module
base_preprocessor = _module
base_task = _module
hyper_spaces = _module
param = _module
param_table = _module
losses = _module
rank_cross_entropy_loss = _module
rank_hinge_loss = _module
metrics = _module
accuracy = _module
average_precision = _module
cross_entropy = _module
discounted_cumulative_gain = _module
mean_average_precision = _module
mean_reciprocal_rank = _module
normalized_discounted_cumulative_gain = _module
precision = _module
models = _module
anmm = _module
arci = _module
arcii = _module
bert = _module
bimpm = _module
cdssm = _module
conv_knrm = _module
dense_baseline = _module
diin = _module
drmm = _module
drmmtks = _module
dssm = _module
duet = _module
esim = _module
hbmp = _module
knrm = _module
match_pyramid = _module
match_srnn = _module
matchlstm = _module
mvlstm = _module
parameter_readme_generator = _module
modules = _module
attention = _module
bert_module = _module
character_embedding = _module
dense_net = _module
dropout = _module
gaussian_kernel = _module
matching = _module
matching_tensor = _module
semantic_composite = _module
spatial_gru = _module
stacked_brnn = _module
preprocessors = _module
basic_preprocessor = _module
bert_preprocessor = _module
build_unit_from_data_pack = _module
build_vocab_unit = _module
chain_transform = _module
naive_preprocessor = _module
units = _module
character_index = _module
digit_removal = _module
frequency_filter = _module
lemmatization = _module
lowercase = _module
matching_histogram = _module
ngram_letter = _module
punc_removal = _module
stateful_unit = _module
stemming = _module
stop_removal = _module
tokenize = _module
truncated_length = _module
unit = _module
vocabulary = _module
word_exact_match = _module
word_hashing = _module
tasks = _module
classification = _module
ranking = _module
trainers = _module
trainer = _module
utils = _module
average_meter = _module
early_stopping = _module
get_file = _module
list_recursive_subclasses = _module
one_hot = _module
parse = _module
tensor_type = _module
timer = _module
version = _module
setup = _module
tests = _module
test_datapack = _module
test_callbacks = _module
test_dataset = _module
test_base_preprocessor = _module
test_base_task = _module
test_hyper_spaces = _module
test_param_table = _module
test_base_model = _module
test_models = _module
test_modules = _module
test_tasks = _module
test_datasets = _module
test_embedding = _module
test_losses = _module
test_metrics = _module
test_utils = _module
test_trainer = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import copy


import typing


import logging


import torch


import numpy as np


import math


from torch.utils import data


from collections import Iterable


import abc


import torch.nn as nn


from torch import nn


import torch.nn.functional as F


from torch.nn import functional as F


import torch.optim as optim


from torch import optim


def _parse(identifier: typing.Union[str, typing.Type[nn.Module], nn.Module], dictionary: nn.ModuleDict, target: str) ->nn.Module:
    """
    Parse loss and activation.

    :param identifier: activation identifier, one of
            - String: name of a activation
            - Torch Modele subclass
            - Torch Module instance (it will be returned unchanged).
    :param dictionary: nn.ModuleDict instance. Map string identifier to
        nn.Module instance.
    :return: A :class:`nn.Module` instance
    """
    if isinstance(identifier, str):
        if identifier in dictionary:
            return dictionary[identifier]
        else:
            raise ValueError(f'Could not interpret {target} identifier: ' + str(identifier))
    elif isinstance(identifier, nn.Module):
        return identifier
    elif issubclass(identifier, nn.Module):
        return identifier()
    else:
        raise ValueError(f'Could not interpret {target} identifier: ' + str(identifier))


activation = nn.ModuleDict([['relu', nn.ReLU()], ['hardtanh', nn.Hardtanh()], ['relu6', nn.ReLU6()], ['sigmoid', nn.Sigmoid()], ['tanh', nn.Tanh()], ['softmax', nn.Softmax()], ['softmax2d', nn.Softmax2d()], ['logsoftmax', nn.LogSoftmax()], ['elu', nn.ELU()], ['selu', nn.SELU()], ['celu', nn.CELU()], ['hardshrink', nn.Hardshrink()], ['leakyrelu', nn.LeakyReLU()], ['logsigmoid', nn.LogSigmoid()], ['softplus', nn.Softplus()], ['softshrink', nn.Softshrink()], ['prelu', nn.PReLU()], ['softsign', nn.Softsign()], ['softmin', nn.Softmin()], ['tanhshrink', nn.Tanhshrink()], ['rrelu', nn.RReLU()], ['glu', nn.GLU()]])


def parse_activation(identifier: typing.Union[str, typing.Type[nn.Module], nn.Module]) ->nn.Module:
    """
    Retrieves a torch Module instance.

    :param identifier: activation identifier, one of
            - String: name of a activation
            - Torch Modele subclass
            - Torch Module instance (it will be returned unchanged).
    :return: A :class:`nn.Module` instance

    Examples::
        >>> from torch import nn
        >>> from matchzoo.utils import parse_activation

    Use `str` as activation:
        >>> activation = parse_activation('relu')
        >>> type(activation)
        <class 'torch.nn.modules.activation.ReLU'>

    Use :class:`torch.nn.Module` subclasses as activation:
        >>> type(parse_activation(nn.ReLU))
        <class 'torch.nn.modules.activation.ReLU'>

    Use :class:`torch.nn.Module` instances as activation:
        >>> type(parse_activation(nn.ReLU()))
        <class 'torch.nn.modules.activation.ReLU'>

    """
    return _parse(identifier, activation, 'activation')


class RankCrossEntropyLoss(nn.Module):
    """Creates a criterion that measures rank cross entropy loss."""
    __constants__ = ['num_neg']

    def __init__(self, num_neg: int=1):
        """
        :class:`RankCrossEntropyLoss` constructor.

        :param num_neg: Number of negative instances in hinge loss.
        """
        super().__init__()
        self.num_neg = num_neg

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """
        Calculate rank cross entropy loss.

        :param y_pred: Predicted result.
        :param y_true: Label.
        :return: Rank cross loss.
        """
        logits = y_pred[::self.num_neg + 1, :]
        labels = y_true[::self.num_neg + 1, :]
        for neg_idx in range(self.num_neg):
            neg_logits = y_pred[neg_idx + 1::self.num_neg + 1, :]
            neg_labels = y_true[neg_idx + 1::self.num_neg + 1, :]
            logits = torch.cat((logits, neg_logits), dim=-1)
            labels = torch.cat((labels, neg_labels), dim=-1)
        return -torch.mean(torch.sum(labels * torch.log(F.softmax(logits, dim=-1) + torch.finfo(float).eps), dim=-1))

    @property
    def num_neg(self):
        """`num_neg` getter."""
        return self._num_neg

    @num_neg.setter
    def num_neg(self, value):
        """`num_neg` setter."""
        self._num_neg = value


class RankHingeLoss(nn.Module):
    """
    Creates a criterion that measures rank hinge loss.

    Given inputs :math:`x1`, :math:`x2`, two 1D mini-batch `Tensors`,
    and a label 1D mini-batch tensor :math:`y` (containing 1 or -1).

    If :math:`y = 1` then it assumed the first input should be ranked
    higher (have a larger value) than the second input, and vice-versa
    for :math:`y = -1`.

    The loss function for each sample in the mini-batch is:

    .. math::
        loss_{x, y} = max(0, -y * (x1 - x2) + margin)
    """
    __constants__ = ['num_neg', 'margin', 'reduction']

    def __init__(self, num_neg: int=1, margin: float=1.0, reduction: str='mean'):
        """
        :class:`RankHingeLoss` constructor.

        :param num_neg: Number of negative instances in hinge loss.
        :param margin: Margin between positive and negative scores.
            Float. Has a default value of :math:`0`.
        :param reduction: String. Specifies the reduction to apply to
            the output: ``'none'`` | ``'mean'`` | ``'sum'``.
            ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the
                number of elements in the output,
            ``'sum'``: the output will be summed.
        """
        super().__init__()
        self.num_neg = num_neg
        self.margin = margin
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """
        Calculate rank hinge loss.

        :param y_pred: Predicted result.
        :param y_true: Label.
        :return: Hinge loss computed by user-defined margin.
        """
        y_pos = y_pred[::self.num_neg + 1, :]
        y_neg = []
        for neg_idx in range(self.num_neg):
            neg = y_pred[neg_idx + 1::self.num_neg + 1, :]
            y_neg.append(neg)
        y_neg = torch.cat(y_neg, dim=-1)
        y_neg = torch.mean(y_neg, dim=-1, keepdim=True)
        y_true = torch.ones_like(y_pos)
        return F.margin_ranking_loss(y_pos, y_neg, y_true, margin=self.margin, reduction=self.reduction)

    @property
    def num_neg(self):
        """`num_neg` getter."""
        return self._num_neg

    @num_neg.setter
    def num_neg(self, value):
        """`num_neg` setter."""
        self._num_neg = value

    @property
    def margin(self):
        """`margin` getter."""
        return self._margin

    @margin.setter
    def margin(self, value):
        """`margin` setter."""
        self._margin = value


class Attention(nn.Module):
    """
    Attention module.

    :param input_size: Size of input.
    :param mask: An integer to mask the invalid values. Defaults to 0.

    Examples:
        >>> import torch
        >>> attention = Attention(input_size=10)
        >>> x = torch.randn(4, 5, 10)
        >>> x.shape
        torch.Size([4, 5, 10])
        >>> x_mask = torch.BoolTensor(4, 5)
        >>> attention(x, x_mask).shape
        torch.Size([4, 5])

    """

    def __init__(self, input_size: int=100):
        """Attention constructor."""
        super().__init__()
        self.linear = nn.Linear(input_size, 1, bias=False)

    def forward(self, x, x_mask):
        """Perform attention on the input."""
        x = self.linear(x).squeeze(dim=-1)
        x = x.masked_fill(x_mask, -float('inf'))
        return F.softmax(x, dim=-1)


class Matching(nn.Module):
    """
    Module that computes a matching matrix between samples in two tensors.

    :param normalize: Whether to L2-normalize samples along the
        dot product axis before taking the dot product.
        If set to `True`, then the output of the dot product
        is the cosine proximity between the two samples.
    :param matching_type: the similarity function for matching

    Examples:
        >>> import torch
        >>> matching = Matching(matching_type='dot', normalize=True)
        >>> x = torch.randn(2, 3, 2)
        >>> y = torch.randn(2, 4, 2)
        >>> matching(x, y).shape
        torch.Size([2, 3, 4])

    """

    def __init__(self, normalize: bool=False, matching_type: str='dot'):
        """:class:`Matching` constructor."""
        super().__init__()
        self._normalize = normalize
        self._validate_matching_type(matching_type)
        self._matching_type = matching_type

    @classmethod
    def _validate_matching_type(cls, matching_type: str='dot'):
        valid_matching_type = ['dot', 'exact', 'mul', 'plus', 'minus', 'concat']
        if matching_type not in valid_matching_type:
            raise ValueError(f'{matching_type} is not a valid matching type, {valid_matching_type} expected.')

    def forward(self, x, y):
        """Perform attention on the input."""
        length_left = x.shape[1]
        length_right = y.shape[1]
        if self._matching_type == 'dot':
            if self._normalize:
                x = F.normalize(x, p=2, dim=-1)
                y = F.normalize(y, p=2, dim=-1)
            return torch.einsum('bld,brd->blr', x, y)
        elif self._matching_type == 'exact':
            x = x.unsqueeze(dim=2).repeat(1, 1, length_right)
            y = y.unsqueeze(dim=1).repeat(1, length_left, 1)
            matching_matrix = x == y
            x = torch.sum(matching_matrix, dim=2, dtype=torch.float)
            y = torch.sum(matching_matrix, dim=1, dtype=torch.float)
            return x, y
        else:
            x = x.unsqueeze(dim=2).repeat(1, 1, length_right, 1)
            y = y.unsqueeze(dim=1).repeat(1, length_left, 1, 1)
            if self._matching_type == 'mul':
                return x * y
            elif self._matching_type == 'plus':
                return x + y
            elif self._matching_type == 'minus':
                return x - y
            elif self._matching_type == 'concat':
                return torch.cat((x, y), dim=3)


class BertModule(nn.Module):
    """
    Bert module.

    BERT (from Google) released with the paper BERT: Pre-training of Deep
    Bidirectional Transformers for Language Understanding by Jacob Devlin,
    Ming-Wei Chang, Kenton Lee and Kristina Toutanova.

    :param mode: String, supported mode can be referred
        https://huggingface.co/pytorch-transformers/pretrained_models.html.

    """

    def __init__(self, mode: str='bert-base-uncased'):
        """:class:`BertModule` constructor."""
        super().__init__()
        self.bert = BertModel.from_pretrained(mode)

    def forward(self, x, y):
        """Forward."""
        input_ids = torch.cat((x, y), dim=-1)
        token_type_ids = torch.cat((torch.zeros_like(x), torch.ones_like(y)), dim=-1).long()
        attention_mask = input_ids != 0
        return self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)


def div_with_small_value(n, d, eps=1e-08):
    """
    Small values are replaced by 1e-8 to prevent it from exploding.

    :param n: tensor
    :param d: tensor
    :return: n/d: tensor
    """
    d = d * (d > eps).float() + eps * (d <= eps).float()
    return n / d


def attention(v1, v2):
    """
    Attention.

    :param v1: (batch, seq_len1, hidden_size)
    :param v2: (batch, seq_len2, hidden_size)
    :return: (batch, seq_len1, seq_len2)
    """
    v1_norm = v1.norm(p=2, dim=2, keepdim=True)
    v2_norm = v2.norm(p=2, dim=2, keepdim=True).permute(0, 2, 1)
    a = torch.bmm(v1, v2.permute(0, 2, 1))
    d = v1_norm * v2_norm
    return div_with_small_value(a, d)


def mp_matching_func(v1, v2, w):
    """
    Basic mp_matching_func.

    :param v1: (batch, seq_len, hidden_size)
    :param v2: (batch, seq_len, hidden_size) or (batch, hidden_size)
    :param w: (num_psp, hidden_size)
    :return: (batch, num_psp)
    """
    seq_len = v1.size(1)
    num_psp = w.size(0)
    w = w.transpose(1, 0).unsqueeze(0).unsqueeze(0)
    v1 = w * torch.stack([v1] * num_psp, dim=3)
    if len(v2.size()) == 3:
        v2 = w * torch.stack([v2] * num_psp, dim=3)
    else:
        v2 = w * torch.stack([torch.stack([v2] * seq_len, dim=1)] * num_psp, dim=3)
    m = F.cosine_similarity(v1, v2, dim=2)
    return m


def mp_matching_func_pairwise(v1, v2, w):
    """
    Basic mp_matching_func_pairwise.

    :param v1: (batch, seq_len1, hidden_size)
    :param v2: (batch, seq_len2, hidden_size)
    :param w: (num_psp, hidden_size)
    :param num_psp
    :return: (batch, num_psp, seq_len1, seq_len2)
    """
    num_psp = w.size(0)
    w = w.unsqueeze(0).unsqueeze(2)
    v1, v2 = w * torch.stack([v1] * num_psp, dim=1), w * torch.stack([v2] * num_psp, dim=1)
    v1_norm = v1.norm(p=2, dim=3, keepdim=True)
    v2_norm = v2.norm(p=2, dim=3, keepdim=True)
    n = torch.matmul(v1, v2.transpose(2, 3))
    d = v1_norm * v2_norm.transpose(2, 3)
    m = div_with_small_value(n, d).permute(0, 2, 3, 1)
    return m


class Squeeze(nn.Module):
    """Squeeze."""

    def forward(self, x):
        """Forward."""
        return x.squeeze(dim=-1)


class GaussianKernel(nn.Module):
    """
    Gaussian kernel module.

    :param mu: Float, mean of the kernel.
    :param sigma: Float, sigma of the kernel.

    Examples:
        >>> import torch
        >>> kernel = GaussianKernel()
        >>> x = torch.randn(4, 5, 10)
        >>> x.shape
        torch.Size([4, 5, 10])
        >>> kernel(x).shape
        torch.Size([4, 5, 10])

    """

    def __init__(self, mu: float=1.0, sigma: float=1.0):
        """Gaussian kernel constructor."""
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def forward(self, x):
        """Forward."""
        return torch.exp(-0.5 * (x - self.mu) ** 2 / self.sigma ** 2)


class CharacterEmbedding(nn.Module):
    """
    Character embedding module.

    :param char_embedding_input_dim: The input dimension of character embedding layer.
    :param char_embedding_output_dim: The output dimension of character embedding layer.
    :param char_conv_filters: The filter size of character convolution layer.
    :param char_conv_kernel_size: The kernel size of character convolution layer.

    Examples:
        >>> import torch
        >>> character_embedding = CharacterEmbedding()
        >>> x = torch.ones(10, 32, 16, dtype=torch.long)
        >>> x.shape
        torch.Size([10, 32, 16])
        >>> character_embedding(x).shape
        torch.Size([10, 32, 100])

    """

    def __init__(self, char_embedding_input_dim: int=100, char_embedding_output_dim: int=8, char_conv_filters: int=100, char_conv_kernel_size: int=5):
        """Init."""
        super().__init__()
        self.char_embedding = nn.Embedding(num_embeddings=char_embedding_input_dim, embedding_dim=char_embedding_output_dim)
        self.conv = nn.Conv1d(in_channels=char_embedding_output_dim, out_channels=char_conv_filters, kernel_size=char_conv_kernel_size)

    def forward(self, x):
        """Forward."""
        embed_x = self.char_embedding(x)
        batch_size, seq_len, word_len, embed_dim = embed_x.shape
        embed_x = embed_x.contiguous().view(-1, word_len, embed_dim)
        embed_x = self.conv(embed_x.transpose(1, 2))
        embed_x = torch.max(embed_x, dim=-1)[0]
        embed_x = embed_x.view(batch_size, seq_len, -1)
        return embed_x


class DenseBlock(nn.Module):
    """Dense block of DenseNet."""

    def __init__(self, in_channels, growth_rate: int=20, kernel_size: tuple=(2, 2), layers_per_dense_block: int=3):
        """Init."""
        super().__init__()
        dense_block = []
        for _ in range(layers_per_dense_block):
            conv_block = self._make_conv_block(in_channels, growth_rate, kernel_size)
            dense_block.append(conv_block)
            in_channels += growth_rate
        self._dense_block = nn.ModuleList(dense_block)

    def forward(self, x):
        """Forward."""
        for layer in self._dense_block:
            conv_out = layer(x)
            x = torch.cat([x, conv_out], dim=1)
        return x

    @classmethod
    def _make_conv_block(cls, in_channels: int, out_channels: int, kernel_size: tuple) ->nn.Module:
        """Make conv block."""
        return nn.Sequential(nn.ConstantPad2d((0, kernel_size[1] - 1, 0, kernel_size[0] - 1), 0), nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size), nn.ReLU())


class DenseNet(nn.Module):
    """
    DenseNet module.

    :param in_channels: Feature size of input.
    :param nb_dense_blocks: The number of blocks in densenet.
    :param layers_per_dense_block: The number of convolution layers in dense block.
    :param growth_rate: The filter size of each convolution layer in dense block.
    :param transition_scale_down_ratio: The channel scale down ratio of the convolution
        layer in transition block.
    :param conv_kernel_size: The kernel size of convolution layer in dense block.
    :param pool_kernel_size: The kernel size of pooling layer in transition block.
    """

    def __init__(self, in_channels, nb_dense_blocks: int=3, layers_per_dense_block: int=3, growth_rate: int=10, transition_scale_down_ratio: float=0.5, conv_kernel_size: tuple=(2, 2), pool_kernel_size: tuple=(2, 2)):
        """Init."""
        super().__init__()
        dense_blocks = []
        transition_blocks = []
        for _ in range(nb_dense_blocks):
            dense_block = DenseBlock(in_channels, growth_rate, conv_kernel_size, layers_per_dense_block)
            in_channels += layers_per_dense_block * growth_rate
            dense_blocks.append(dense_block)
            transition_block = self._make_transition_block(in_channels, transition_scale_down_ratio, pool_kernel_size)
            in_channels = int(in_channels * transition_scale_down_ratio)
            transition_blocks.append(transition_block)
        self._dense_blocks = nn.ModuleList(dense_blocks)
        self._transition_blocks = nn.ModuleList(transition_blocks)
        self._out_channels = in_channels

    @property
    def out_channels(self) ->int:
        """`out_channels` getter."""
        return self._out_channels

    def forward(self, x):
        """Forward."""
        for dense_block, trans_block in zip(self._dense_blocks, self._transition_blocks):
            x = dense_block(x)
            x = trans_block(x)
        return x

    @classmethod
    def _make_transition_block(cls, in_channels: int, transition_scale_down_ratio: float, pool_kernel_size: tuple) ->nn.Module:
        return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=int(in_channels * transition_scale_down_ratio), kernel_size=1), nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_kernel_size))


class SemanticComposite(nn.Module):
    """
    SemanticComposite module.

    Apply a self-attention layer and a semantic composite fuse gate to compute the
    encoding result of one tensor.

    :param in_features: Feature size of input.
    :param dropout_rate: The dropout rate.

    Examples:
        >>> import torch
        >>> module = SemanticComposite(in_features=10)
        >>> x = torch.randn(4, 5, 10)
        >>> x.shape
        torch.Size([4, 5, 10])
        >>> module(x).shape
        torch.Size([4, 5, 10])

    """

    def __init__(self, in_features, dropout_rate: float=0.0):
        """Init."""
        super().__init__()
        self.att_linear = nn.Linear(3 * in_features, 1, False)
        self.z_gate = nn.Linear(2 * in_features, in_features, True)
        self.r_gate = nn.Linear(2 * in_features, in_features, True)
        self.f_gate = nn.Linear(2 * in_features, in_features, True)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        """Forward."""
        seq_length = x.shape[1]
        x_1 = x.unsqueeze(dim=2).repeat(1, 1, seq_length, 1)
        x_2 = x.unsqueeze(dim=1).repeat(1, seq_length, 1, 1)
        x_concat = torch.cat([x_1, x_2, x_1 * x_2], dim=-1)
        x_concat = self.dropout(x_concat)
        attn_matrix = self.att_linear(x_concat).squeeze(dim=-1)
        attn_weight = torch.softmax(attn_matrix, dim=2)
        attn = torch.bmm(attn_weight, x)
        x_attn_concat = self.dropout(torch.cat([x, attn], dim=-1))
        x_attn_concat = torch.cat([x, attn], dim=-1)
        z = torch.tanh(self.z_gate(x_attn_concat))
        r = torch.sigmoid(self.r_gate(x_attn_concat))
        f = torch.sigmoid(self.f_gate(x_attn_concat))
        encoding = r * x + f * z
        return encoding


class BidirectionalAttention(nn.Module):
    """Computing the soft attention between two sequence."""

    def __init__(self):
        """Init."""
        super().__init__()

    def forward(self, v1, v1_mask, v2, v2_mask):
        """Forward."""
        similarity_matrix = v1.bmm(v2.transpose(2, 1).contiguous())
        v2_v1_attn = F.softmax(similarity_matrix.masked_fill(v1_mask.unsqueeze(2), -1e-07), dim=1)
        v1_v2_attn = F.softmax(similarity_matrix.masked_fill(v2_mask.unsqueeze(1), -1e-07), dim=2)
        attended_v1 = v1_v2_attn.bmm(v2)
        attended_v2 = v2_v1_attn.transpose(1, 2).bmm(v1)
        attended_v1.masked_fill_(v1_mask.unsqueeze(2), 0)
        attended_v2.masked_fill_(v2_mask.unsqueeze(2), 0)
        return attended_v1, attended_v2


class RNNDropout(nn.Dropout):
    """Dropout for RNN."""

    def forward(self, sequences_batch):
        """Masking whole hidden vector for tokens."""
        ones = sequences_batch.data.new_ones(sequences_batch.shape[0], sequences_batch.shape[-1])
        dropout_mask = nn.functional.dropout(ones, self.p, self.training, inplace=False)
        return dropout_mask.unsqueeze(1) * sequences_batch


class StackedBRNN(nn.Module):
    """
    Stacked Bi-directional RNNs.

    Differs from standard PyTorch library in that it has the option to save
    and concat the hidden states between layers. (i.e. the output hidden size
    for each sequence input is num_layers * hidden_size).

    Examples:
        >>> import torch
        >>> rnn = StackedBRNN(
        ...     input_size=10,
        ...     hidden_size=10,
        ...     num_layers=2,
        ...     dropout_rate=0.2,
        ...     dropout_output=True,
        ...     concat_layers=False
        ... )
        >>> x = torch.randn(2, 5, 10)
        >>> x.size()
        torch.Size([2, 5, 10])
        >>> x_mask = (torch.ones(2, 5) == 1)
        >>> rnn(x, x_mask).shape
        torch.Size([2, 5, 20])

    """

    def __init__(self, input_size, hidden_size, num_layers, dropout_rate=0, dropout_output=False, rnn_type=nn.LSTM, concat_layers=False):
        """Stacked Bidirectional LSTM."""
        super().__init__()
        self.dropout_output = dropout_output
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.concat_layers = concat_layers
        self.rnns = nn.ModuleList()
        for i in range(num_layers):
            input_size = input_size if i == 0 else 2 * hidden_size
            self.rnns.append(rnn_type(input_size, hidden_size, num_layers=1, bidirectional=True))

    def forward(self, x, x_mask):
        """Encode either padded or non-padded sequences."""
        if x_mask.data.sum() == 0:
            output = self._forward_unpadded(x, x_mask)
        output = self._forward_unpadded(x, x_mask)
        return output.contiguous()

    def _forward_unpadded(self, x, x_mask):
        """Faster encoding that ignores any padding."""
        x = x.transpose(0, 1)
        outputs = [x]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]
            if self.dropout_rate > 0:
                rnn_input = F.dropout(rnn_input, p=self.dropout_rate, training=self.training)
            rnn_output = self.rnns[i](rnn_input)[0]
            outputs.append(rnn_output)
        if self.concat_layers:
            output = torch.cat(outputs[1:], 2)
        else:
            output = outputs[-1]
        output = output.transpose(0, 1)
        if self.dropout_output and self.dropout_rate > 0:
            output = F.dropout(output, p=self.dropout_rate, training=self.training)
        return output


class MatchingTensor(nn.Module):
    """
    Module that captures the basic interactions between two tensors.

    :param matching_dims: Word dimension of two interaction texts.
    :param channels: Number of word interaction tensor channels.
    :param normalize: Whether to L2-normalize samples along the
        dot product axis before taking the dot product.
        If set to True, then the output of the dot product
        is the cosine proximity between the two samples.
    :param init_diag: Whether to initialize the diagonal elements
        of the matrix.

    Examples:
        >>> import matchzoo as mz
        >>> matching_dim = 5
        >>> matching_tensor = mz.modules.MatchingTensor(
        ...    matching_dim,
        ...    channels=4,
        ...    normalize=True,
        ...    init_diag=True
        ... )

    """

    def __init__(self, matching_dim: int, channels: int=4, normalize: bool=True, init_diag: bool=True):
        """:class:`MatchingTensor` constructor."""
        super().__init__()
        self._matching_dim = matching_dim
        self._channels = channels
        self._normalize = normalize
        self._init_diag = init_diag
        self.interaction_matrix = torch.empty(self._channels, self._matching_dim, self._matching_dim)
        if self._init_diag:
            self.interaction_matrix = self.interaction_matrix.uniform_(-0.05, 0.05)
            for channel_index in range(self._channels):
                self.interaction_matrix[channel_index].fill_diagonal_(0.1)
            self.interaction_matrix = nn.Parameter(self.interaction_matrix)
        else:
            self.interaction_matrix = nn.Parameter(self.interaction_matrix.uniform_())

    def forward(self, x, y):
        """
        The computation logic of MatchingTensor.

        :param inputs: two input tensors.
        """
        if self._normalize:
            x = F.normalize(x, p=2, dim=-1)
            y = F.normalize(y, p=2, dim=-1)
        output = torch.einsum('bld,cde,bre->bclr', x, self.interaction_matrix, y)
        return output


class SpatialGRU(nn.Module):
    """
    Spatial GRU Module.

    :param channels: Number of word interaction tensor channels.
    :param units: Number of SpatialGRU units.
    :param activation: Activation function to use, one of:
            - String: name of an activation
            - Torch Modele subclass
            - Torch Module instance
            Default: hyperbolic tangent (`tanh`).
    :param recurrent_activation: Activation function to use for
        the recurrent step, one of:
            - String: name of an activation
            - Torch Modele subclass
            - Torch Module instance
            Default: sigmoid activation (`sigmoid`).
    :param direction: Scanning direction. `lt` (i.e., left top)
        indicates the scanning from left top to right bottom, and
        `rb` (i.e., right bottom) indicates the scanning from
        right bottom to left top.

    Examples:
        >>> import matchzoo as mz
        >>> channels, units= 4, 10
        >>> spatial_gru = mz.modules.SpatialGRU(channels, units)

    """

    def __init__(self, channels: int=4, units: int=10, activation: typing.Union[str, typing.Type[nn.Module], nn.Module]='tanh', recurrent_activation: typing.Union[str, typing.Type[nn.Module], nn.Module]='sigmoid', direction: str='lt'):
        """:class:`SpatialGRU` constructor."""
        super().__init__()
        self._units = units
        self._activation = parse_activation(activation)
        self._recurrent_activation = parse_activation(recurrent_activation)
        self._direction = direction
        self._channels = channels
        if self._direction not in ('lt', 'rb'):
            raise ValueError(f'Invalid direction. `{self._direction}` received. Must be in `lt`, `rb`.')
        self._input_dim = self._channels + 3 * self._units
        self._wr = nn.Linear(self._input_dim, self._units * 3)
        self._wz = nn.Linear(self._input_dim, self._units * 4)
        self._w_ij = nn.Linear(self._channels, self._units)
        self._U = nn.Linear(self._units * 3, self._units, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_normal_(self._wr.weight)
        nn.init.xavier_normal_(self._wz.weight)
        nn.init.orthogonal_(self._w_ij.weight)
        nn.init.orthogonal_(self._U.weight)

    def softmax_by_row(self, z: torch.tensor) ->tuple:
        """Conduct softmax on each dimension across the four gates."""
        z_transform = z.reshape((-1, 4, self._units))
        zi, zl, zt, zd = F.softmax(z_transform, dim=1).unbind(dim=1)
        return zi, zl, zt, zd

    def calculate_recurrent_unit(self, inputs: torch.tensor, states: list, i: int, j: int):
        """
        Calculate recurrent unit.

        :param inputs: A tensor which contains interaction
            between left text and right text.
        :param states: An array of tensors which stores the hidden state
            of every step.
        :param i: Recurrent row index.
        :param j: Recurrent column index.

        """
        h_diag = states[i][j]
        h_top = states[i][j + 1]
        h_left = states[i + 1][j]
        s_ij = inputs[i][j]
        q = torch.cat([torch.cat([h_top, h_left], 1), torch.cat([h_diag, s_ij], 1)], 1)
        r = self._recurrent_activation(self._wr(q))
        z = self._wz(q)
        zi, zl, zt, zd = self.softmax_by_row(z)
        h_ij_l = self._w_ij(s_ij)
        h_ij_r = self._U(r * torch.cat([h_left, h_top, h_diag], 1))
        h_ij_ = self._activation(h_ij_l + h_ij_r)
        h_ij = zl * h_left + zt * h_top + zd * h_diag + zi * h_ij_
        return h_ij

    def forward(self, inputs):
        """
        Perform SpatialGRU on word interation matrix.

        :param inputs: input tensors.
        """
        batch_size, channels, left_length, right_length = inputs.shape
        inputs = inputs.permute([2, 3, 0, 1])
        if self._direction == 'rb':
            inputs = torch.flip(inputs, [0, 1])
        states = [[torch.zeros([batch_size, self._units]).type_as(inputs) for j in range(right_length + 1)] for i in range(left_length + 1)]
        for i in range(left_length):
            for j in range(right_length):
                states[i + 1][j + 1] = self.calculate_recurrent_unit(inputs, states, i, j)
        return states[left_length][right_length]


class MatchModule(nn.Module):
    """
    Computing the match representation for Match LSTM.

    :param hidden_size: Size of hidden vectors.
    :param dropout_rate: Dropout rate of the projection layer. Defaults to 0.

    Examples:
        >>> import torch
        >>> attention = MatchModule(hidden_size=10)
        >>> v1 = torch.randn(4, 5, 10)
        >>> v1.shape
        torch.Size([4, 5, 10])
        >>> v2 = torch.randn(4, 5, 10)
        >>> v2_mask = torch.ones(4, 5).to(dtype=torch.uint8)
        >>> attention(v1, v2, v2_mask).shape
        torch.Size([4, 5, 20])


    """

    def __init__(self, hidden_size, dropout_rate=0):
        """Init."""
        super().__init__()
        self.v2_proj = nn.Linear(hidden_size, hidden_size)
        self.proj = nn.Linear(hidden_size * 4, hidden_size * 2)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, v1, v2, v2_mask):
        """Computing attention vectors and projection vectors."""
        proj_v2 = self.v2_proj(v2)
        similarity_matrix = v1.bmm(proj_v2.transpose(2, 1).contiguous())
        v1_v2_attn = F.softmax(similarity_matrix.masked_fill(v2_mask.unsqueeze(1).bool(), -1e-07), dim=2)
        v2_wsum = v1_v2_attn.bmm(v2)
        fusion = torch.cat([v1, v2_wsum, v1 - v2_wsum, v1 * v2_wsum], dim=2)
        match = self.dropout(F.relu(self.proj(fusion)))
        return match


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (DenseBlock,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (DenseNet,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     False),
    (GaussianKernel,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MatchModule,
     lambda: ([], {'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4])], {}),
     False),
    (Matching,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (MatchingTensor,
     lambda: ([], {'matching_dim': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (RNNDropout,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (RankCrossEntropyLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (RankHingeLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (SemanticComposite,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (SpatialGRU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Squeeze,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (StackedBRNN,
     lambda: ([], {'input_size': 4, 'hidden_size': 4, 'num_layers': 1}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_NTMC_Community_MatchZoo_py(_paritybench_base):
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

