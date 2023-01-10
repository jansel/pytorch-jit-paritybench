import sys
_module = sys.modules[__name__]
del sys
mlm_pytorch = _module
mlm_pytorch = _module
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


from functools import reduce


import torch


from torch import nn


import torch.nn.functional as F


def get_mask_subset_with_prob(mask, prob):
    batch, seq_len, device = *mask.shape, mask.device
    max_masked = math.ceil(prob * seq_len)
    num_tokens = mask.sum(dim=-1, keepdim=True)
    mask_excess = mask.cumsum(dim=-1) > (num_tokens * prob).ceil()
    mask_excess = mask_excess[:, :max_masked]
    rand = torch.rand((batch, seq_len), device=device).masked_fill(~mask, -1000000000.0)
    _, sampled_indices = rand.topk(max_masked, dim=-1)
    sampled_indices = (sampled_indices + 1).masked_fill_(mask_excess, 0)
    new_mask = torch.zeros((batch, seq_len + 1), device=device)
    new_mask.scatter_(-1, sampled_indices, 1)
    return new_mask[:, 1:].bool()


def mask_with_tokens(t, token_ids):
    init_no_mask = torch.full_like(t, False, dtype=torch.bool)
    mask = reduce(lambda acc, el: acc | (t == el), token_ids, init_no_mask)
    return mask


def prob_mask_like(t, prob):
    return torch.zeros_like(t).float().uniform_(0, 1) < prob


class MLM(nn.Module):

    def __init__(self, transformer, mask_prob=0.15, replace_prob=0.9, num_tokens=None, random_token_prob=0.0, mask_token_id=2, pad_token_id=0, mask_ignore_token_ids=[]):
        super().__init__()
        self.transformer = transformer
        self.mask_prob = mask_prob
        self.replace_prob = replace_prob
        self.num_tokens = num_tokens
        self.random_token_prob = random_token_prob
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.mask_ignore_token_ids = set([*mask_ignore_token_ids, pad_token_id])

    def forward(self, seq, **kwargs):
        no_mask = mask_with_tokens(seq, self.mask_ignore_token_ids)
        mask = get_mask_subset_with_prob(~no_mask, self.mask_prob)
        masked_seq = seq.clone().detach()
        labels = seq.masked_fill(~mask, self.pad_token_id)
        if self.random_token_prob > 0:
            assert self.num_tokens is not None, 'num_tokens keyword must be supplied when instantiating MLM if using random token replacement'
            random_token_prob = prob_mask_like(seq, self.random_token_prob)
            random_tokens = torch.randint(0, self.num_tokens, seq.shape, device=seq.device)
            random_no_mask = mask_with_tokens(random_tokens, self.mask_ignore_token_ids)
            random_token_prob &= ~random_no_mask
            masked_seq = torch.where(random_token_prob, random_tokens, masked_seq)
            mask = mask & ~random_token_prob
        replace_prob = prob_mask_like(seq, self.replace_prob)
        masked_seq = masked_seq.masked_fill(mask * replace_prob, self.mask_token_id)
        logits = self.transformer(masked_seq, **kwargs)
        mlm_loss = F.cross_entropy(logits.transpose(1, 2), labels, ignore_index=self.pad_token_id)
        return mlm_loss

