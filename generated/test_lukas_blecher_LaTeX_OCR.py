import sys
_module = sys.modules[__name__]
del sys
conf = _module
pix2tex = _module
api = _module
app = _module
run = _module
streamlit = _module
cli = _module
dataset = _module
arxiv = _module
dataset = _module
demacro = _module
extract_latex = _module
latex2png = _module
postprocess = _module
preprocessing = _module
generate_latex_vocab = _module
preprocess_formulas = _module
render = _module
scraping = _module
transforms = _module
eval = _module
gui = _module
model = _module
checkpoints = _module
get_latest_checkpoint = _module
models = _module
hybrid = _module
transformer = _module
utils = _module
vit = _module
resources = _module
setup_desktop = _module
train = _module
train_resizer = _module
utils = _module
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


import pandas.io.clipboard as clipboard


from typing import List


from typing import Optional


from typing import Tuple


import logging


import re


import numpy as np


import torch


from torch._appdirs import user_data_dir


import torch.nn.functional as F


from torch.nn.utils.rnn import pad_sequence


from collections import defaultdict


from torchtext.data import metrics


import torch.nn as nn


from torch.optim import Adam


from torch.optim.lr_scheduler import OneCycleLR


import random


from inspect import isfunction


class Model(nn.Module):

    def __init__(self, encoder, decoder, args):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.args = args

    def data_parallel(self, x: torch.Tensor, device_ids, output_device=None, **kwargs):
        if not device_ids or len(device_ids) == 1:
            return self(x, **kwargs)
        if output_device is None:
            output_device = device_ids[0]
        replicas = nn.parallel.replicate(self, device_ids)
        inputs = nn.parallel.scatter(x, device_ids)
        kwargs = nn.parallel.scatter(kwargs, device_ids)
        replicas = replicas[:len(inputs)]
        kwargs = kwargs[:len(inputs)]
        outputs = nn.parallel.parallel_apply(replicas, inputs, kwargs)
        return nn.parallel.gather(outputs, output_device).mean()

    def forward(self, x: torch.Tensor, tgt_seq: torch.Tensor, **kwargs):
        encoded = self.encoder(x)
        out = self.decoder(tgt_seq, context=encoded, **kwargs)
        return out

    @torch.no_grad()
    def generate(self, x: torch.Tensor, temperature: float=0.25):
        return self.decoder.generate(torch.LongTensor([self.args.bos_token] * len(x))[:, None], self.args.max_seq_len, eos_token=self.args.eos_token, context=self.encoder(x), temperature=temperature)


class ViTransformerWrapper(nn.Module):

    def __init__(self, *, max_width, max_height, patch_size, attn_layers, channels=1, num_classes=None, dropout=0.0, emb_dropout=0.0):
        super().__init__()
        assert isinstance(attn_layers, Encoder), 'attention layers must be an Encoder'
        assert max_width % patch_size == 0 and max_height % patch_size == 0, 'image dimensions must be divisible by the patch size'
        dim = attn_layers.dim
        num_patches = max_width // patch_size * (max_height // patch_size)
        patch_dim = channels * patch_size ** 2
        self.patch_size = patch_size
        self.max_width = max_width
        self.max_height = max_height
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.attn_layers = attn_layers
        self.norm = nn.LayerNorm(dim)

    def forward(self, img, **kwargs):
        p = self.patch_size
        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        h, w = torch.tensor(img.shape[2:]) // p
        pos_emb_ind = repeat(torch.arange(h) * (self.max_width // p - w), 'h -> (h w)', w=w) + torch.arange(h * w)
        pos_emb_ind = torch.cat((torch.zeros(1), pos_emb_ind + 1), dim=0).long()
        x += self.pos_embedding[:, pos_emb_ind]
        x = self.dropout(x)
        x = self.attn_layers(x, **kwargs)
        x = self.norm(x)
        return x

