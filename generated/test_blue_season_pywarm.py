import sys
_module = sys.modules[__name__]
del sys
efficientnet = _module
lstm = _module
mnist = _module
mobilenet = _module
resnet = _module
transformer = _module
test_engine = _module
test_functional = _module
test_module = _module
test_util = _module
test_warm = _module
warm = _module
engine = _module
functional = _module
module = _module
util = _module

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
xrange = range
wraps = functools.wraps


import torch


import torch.nn as nn


import torch.nn.functional as F


import torch.optim as optim


from torchvision import datasets


from torchvision import transforms


import time


import copy


import numpy as np


import functools


import re


def conv_pad_same(x, size, kernel=1, stride=1, **kw):
    pad = 0
    if kernel != 1 or stride != 1:
        in_size, s, k = [torch.as_tensor(v) for v in (x.shape[2:], stride, kernel)]
        pad = torch.max(((in_size + s - 1) // s - 1) * s + k - in_size, torch.tensor(0))
        left, right = pad // 2, pad - pad // 2
        if torch.all(left == right):
            pad = tuple(left.tolist())
        else:
            left, right = left.tolist(), right.tolist()
            pad = sum(zip(left[::-1], right[::-1]), ())
            x = F.pad(x, pad)
            pad = 0
    return W.conv(x, size, kernel, stride=stride, padding=pad, **kw)


def is_ready(model):
    """ Check if a `model` is prepared. """
    return hasattr(model, '_pywarm_forward_pre_hook')


def _auto_name(name, parent):
    """ Track the count of reference to `name` from `parent`. """
    if not is_ready(parent):
        parent._pywarm_auto_name_dict = {}

        def _hook(model, x):
            model._pywarm_auto_name_dict = {}
        parent._pywarm_forward_pre_hook = parent.register_forward_pre_hook(_hook)
    track = parent._pywarm_auto_name_dict
    if name not in track:
        track[name] = 0
    track[name] += 1
    return f'{name}_{track[name]}'


def get_default_parent():
    """ Get the default `parent` module. """
    global _DEFAULT_PARENT_MODULE
    return _DEFAULT_PARENT_MODULE


def namespace(f):
    """ After decoration, the function name and call count will be appended to the `name` kw. """

    @functools.wraps(f)
    def _wrapped(*arg, **kw):
        parent = kw.get('parent', get_default_parent())
        name = kw.get('name', '')
        name = '_warmns_' + name + ('-' if name else '') + f.__name__
        name = _auto_name(name, parent)
        kw['name'] = name.replace('_warmns_', '')
        return f(*arg, **kw)
    return _wrapped


def swish(x):
    return x * torch.sigmoid(x)


@namespace
def conv_bn_act(x, size, kernel=1, stride=1, groups=1, bias=False, eps=0.001, momentum=0.01, act=swish, name='', **kw):
    x = conv_pad_same(x, size, kernel, stride=stride, groups=groups, bias=bias, name=name + '-conv')
    return W.batch_norm(x, eps=eps, momentum=momentum, activation=act, name=name + '-bn')


def drop_connect(x, rate):
    """ Randomly set entire batch to 0. """
    if rate == 0:
        return x
    rate = 1.0 - rate
    drop_mask = torch.rand([x.shape[0], 1, 1, 1], device=x.device, requires_grad=False) + rate
    return x / rate * drop_mask.floor()


@namespace
def squeeze_excitation(x, size_se, name='', **kw):
    if size_se == 0:
        return x
    size_in = x.shape[1]
    x = F.adaptive_avg_pool2d(x, 1)
    x = W.conv(x, size_se, 1, activation=swish, name=name + '-conv1')
    return W.conv(x, size_in, 1, activation=swish, name=name + '-conv2')


@namespace
def mb_block(x, size_out, expand=1, kernel=1, stride=1, se_ratio=0.25, dc_ratio=0.2, **kw):
    """ MobileNet Bottleneck Block. """
    size_in = x.shape[1]
    size_mid = size_in * expand
    y = conv_bn_act(x, size_mid, 1, **kw) if expand > 1 else x
    y = conv_bn_act(y, size_mid, kernel, stride=stride, groups=size_mid, **kw)
    y = squeeze_excitation(y, int(size_in * se_ratio), **kw)
    y = conv_bn_act(y, size_out, 1, act=None, **kw)
    if stride == 1 and size_in == size_out:
        y = drop_connect(y, dc_ratio)
        y += x
    return y


spec_b0 = (16, 1, 3, 1, 1, 0.25, 0.2), (24, 6, 3, 2, 2, 0.25, 0.2), (40, 6, 5, 2, 2, 0.25, 0.2), (80, 6, 3, 2, 3, 0.25, 0.2), (112, 6, 5, 1, 3, 0.25, 0.2), (192, 6, 5, 2, 4, 0.25, 0.2), (320, 6, 3, 1, 1, 0.25, 0.2)


class WarmEfficientNet(nn.Module):

    def __init__(self):
        super().__init__()
        warm.up(self, [2, 3, 32, 32])

    def forward(self, x):
        x = conv_bn_act(x, 32, kernel=3, stride=2, name='head')
        for size, expand, kernel, stride, repeat, se_ratio, dc_ratio in spec_b0:
            for i in range(repeat):
                stride = stride if i == 0 else 1
                x = mb_block(x, size, expand, kernel, stride, se_ratio, dc_ratio)
        x = conv_bn_act(x, 1280, name='tail')
        x = F.adaptive_avg_pool2d(x, 1)
        x = W.dropout(x, 0.2)
        x = x.view(x.shape[0], -1)
        x = W.linear(x, 1000)
        return x


class WarmTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super().__init__()
        self.arg = embedding_dim, hidden_dim, vocab_size, tagset_size
        warm.up(self, torch.tensor([0, 1], dtype=torch.long))

    def forward(self, x):
        embedding_dim, hidden_dim, vocab_size, tagset_size = self.arg
        y = W.embedding(x, embedding_dim, vocab_size)
        y = W.lstm(y.T[None, ...], hidden_dim)
        y = W.linear(y, tagset_size)
        y = F.log_softmax(y, dim=1)
        return y[0].T


class TorchTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


class WarmNet(nn.Module):

    def __init__(self):
        super().__init__()
        warm.up(self, [1, 1, 28, 28])

    def forward(self, x):
        x = W.conv(x, 20, 5, activation='relu')
        x = F.max_pool2d(x, 2)
        x = W.conv(x, 50, 5, activation='relu')
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 800)
        x = W.linear(x, 500, activation='relu')
        x = W.linear(x, 10)
        return F.log_softmax(x, dim=1)


class TorchNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def conv_bn_relu(x, size, stride=1, expand=1, kernel=3, groups=1, name=''):
    x = W.conv(x, size, kernel, padding=(kernel - 1) // 2, stride=stride, groups=groups, bias=False, name=f'{name}-0')
    return W.batch_norm(x, activation='relu6', name=f'{name}-1')


def bottleneck(x, size_out, stride, expand, name=''):
    size_in = x.shape[1]
    size_mid = size_in * expand
    y = conv_bn_relu(x, size_mid, kernel=1, name=f'{name}-conv-0') if expand > 1 else x
    y = conv_bn_relu(y, size_mid, stride, kernel=3, groups=size_mid, name=f'{name}-conv-{1 if expand > 1 else 0}')
    y = W.conv(y, size_out, kernel=1, bias=False, name=f'{name}-conv-{2 if expand > 1 else 1}')
    y = W.batch_norm(y, name=f'{name}-conv-{3 if expand > 1 else 2}')
    if stride == 1 and size_in == size_out:
        y += x
    return y


def classify(x, size, *arg, **kw):
    x = W.dropout(x, rate=0.2, name='classifier-0')
    return W.linear(x, size, name='classifier-1')


def conv1x1(x, *arg, **kw):
    return conv_bn_relu(x, *arg, kernel=1, **kw)


def pool(x, *arg, **kw):
    return x.mean([2, 3])


default_spec = (None, 32, 1, 2, conv_bn_relu), (1, 16, 1, 1, bottleneck), (6, 24, 2, 2, bottleneck), (6, 32, 3, 2, bottleneck), (6, 64, 4, 2, bottleneck), (6, 96, 3, 1, bottleneck), (6, 160, 3, 2, bottleneck), (6, 320, 1, 1, bottleneck), (None, 1280, 1, 1, conv1x1), (None, None, 1, None, pool), (None, 1000, 1, None, classify)


class WarmMobileNetV2(nn.Module):

    def __init__(self):
        super().__init__()
        warm.up(self, [2, 3, 224, 224])

    def forward(self, x):
        count = 0
        for t, c, n, s, op in default_spec:
            for i in range(n):
                stride = s if i == 0 else 1
                x = op(x, c, stride, t, name=f'features-{count}')
                count += 1
        return x


def basic(x, size, stride, stack_index, block_index):
    """ The basic block. """
    prefix = f'layer{stack_index + 1}-{block_index}-'
    y = W.conv(x, size, 3, stride=stride, padding=1, bias=False, name=prefix + 'conv1')
    y = W.batch_norm(y, activation='relu', name=prefix + 'bn1')
    y = W.conv(y, size, 3, stride=1, padding=1, bias=False, name=prefix + 'conv2')
    y = W.batch_norm(y, name=prefix + 'bn2')
    if y.shape[1] != x.shape[1]:
        x = W.conv(x, y.shape[1], 1, stride=stride, bias=False, name=prefix + 'downsample-0')
        x = W.batch_norm(x, name=prefix + 'downsample-1')
    return F.relu(y + x)


def stack(x, num_block, size, stride, stack_index, block=basic):
    """ A stack of num_block blocks. """
    for block_index, s in enumerate([stride] + [1] * (num_block - 1)):
        x = block(x, size, s, stack_index, block_index)
    return x


class WarmResNet(nn.Module):

    def __init__(self, block=basic, stack_spec=((2, 64, 1), (2, 128, 2), (2, 256, 2), (2, 512, 2))):
        super().__init__()
        self.block = block
        self.stack_spec = stack_spec
        warm.up(self, [2, 3, 32, 32])

    def forward(self, x):
        y = W.conv(x, 64, 7, stride=2, padding=3, bias=False, name='conv1')
        y = W.batch_norm(y, activation='relu', name='bn1')
        y = F.max_pool2d(y, 3, stride=2, padding=1)
        for i, spec in enumerate(self.stack_spec):
            y = stack(y, *spec, i, block=self.block)
        y = F.adaptive_avg_pool2d(y, 1)
        y = torch.flatten(y, 1)
        y = W.linear(y, 1000, name='fc')
        return y


def identity(x, *arg, **kw):
    """ Identity layer that returns the first input, ignores the rest arguments. """
    return x


def transformer(x, y=None, num_encoder=6, num_decoder=6, num_head=8, mask=None, causal=False, in_shape='BCD', **kw):
    """ Transformer layer.

    This layer covers functionality of `Transformer`, `TransformerEncoder`, and `TransformerDecoder`.
    See [`torch.nn.Transformer`](https://pytorch.org/docs/stable/nn.html#transformer) for more details.

    -  `x: Tensor`; The source sequence, with shape `(Batch, Channel, LengthX)`.
        `Channel` is usually from embedding.
    -  `y: None or Tensor`; The target sequence. Also with shape `(Batch, Channel, LengthY)`.
        If not present, default to equal `x`.
    -  `num_encoder: int`; Number of encoder layers. Set to 0 to disable encoder and use only decoder. Default 6.
    -  `num_decoder: int`; Number of decoder layers. Set to 0 to disable decoder and use only encoder. Default 6.
    -  `num_head: int`; Number of heads for multi-headed attention. Default 8.
    -  `mask: None or dict`; Keys are among: `src_mask`, `tgt_mask`, `memory_mask`,
        `src_key_padding_mask`, `tgt_key_padding_mask`, `memory_key_padding_mask`.
        See the `forward` method of `torch.nn.Transformer` for details.
    -  `causal: bool`; Default false. if true, will add causal masks to source and target, so that
        current value only depends on the past, not the future, in the sequences.
    -  `**kw: dict`; Any additional KWargs are passed down to `torch.nn.Transformer`, as well as `warm.engine.forward`.
    -  `return: Tensor`; Same shape as `y`, if `num_decoder` > 0. Otherwise same shape as `x`. """

    def _causal_mask(n):
        mask = (torch.triu(torch.ones(n, n)) == 1).transpose(0, 1)
        return mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    if y is None:
        y = x
    y = permute(y, in_shape, 'DBC')
    mask = mask or {}
    if causal:
        i = in_shape.find('D')
        mx = _causal_mask(x.shape[i])
        mask['src_mask'] = mask.pop('src_mask', 0.0) + mx
        my = _causal_mask(y.shape[0])
        mask['tgt_mask'] = mask.pop('tgt_mask', 0.0) + my
    encoder = identity if num_encoder == 0 else None
    decoder = identity if num_decoder == 0 else None
    inferred_kw = dict(base_name='transformer', base_class=nn.Transformer, base_shape='DBC', base_kw=dict(d_model=x.shape[in_shape.find('C')], custom_encoder=encoder, custom_decoder=decoder, nhead=num_head, num_encoder_layers=num_encoder, num_decoder_layers=num_decoder, **engine.unused_kwargs(kw)), in_shape=in_shape, forward_kw=mask, forward_arg=(y,))
    return engine.forward(x, **{**inferred_kw, **kw})


class Transformer(nn.Module):

    def __init__(self, *shape, **kw):
        super().__init__()
        self.kw = kw
        warm.up(self, *shape)

    def forward(self, x, y):
        return transformer(x, y, **self.kw)


class Lambda(nn.Module):
    """ Wraps a callable and all its call arguments.

    -  `fn: callable`; The callable being wrapped.
    -  `*arg: list`; Arguments to be passed to `fn`.
    -  `**kw: dict`; KWargs to be passed to `fn`. """

    def __init__(self, fn, *arg, **kw):
        super().__init__()
        self.fn = fn
        self.arg = arg
        self.kw = kw

    def forward(self, x):
        """ forward. """
        return self.fn(x, *self.arg, **self.kw)


class Sequential(nn.Sequential):
    """ Similar to `nn.Sequential`, except that child modules can have multiple outputs (e.g. `nn.RNN`).

    -  `*arg: list of Modules`; Same as `nn.Sequential`. """

    def forward(self, x):
        """ forward. """
        for module in self._modules.values():
            if isinstance(x, tuple):
                try:
                    x = module(x)
                except Exception:
                    x = module(x[0])
            else:
                x = module(x)
        return x


class Shortcut(Sequential):
    """ Similar to `nn.Sequential`, except that it performs a shortcut addition for the input and output.

    -  `*arg: list of Modules`; Same as `nn.Sequential`.
    -  `projection: None or callable`; If `None`, input with be added directly to the output.
        otherwise input will be passed to the `projection` first, usually to make the shapes match. """

    def __init__(self, *arg, projection=None):
        super().__init__(*arg)
        self.projection = projection or nn.Identity()

    def forward(self, x):
        """ forward. """
        return super().forward(x) + self.projection(x)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Lambda,
     lambda: ([], {'fn': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Sequential,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Shortcut,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (TorchTagger,
     lambda: ([], {'embedding_dim': 4, 'hidden_dim': 4, 'vocab_size': 4, 'tagset_size': 4}),
     lambda: ([torch.zeros([4], dtype=torch.int64)], {}),
     True),
]

class Test_blue_season_pywarm(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

