import sys
_module = sys.modules[__name__]
del sys
sample = _module
src = _module
composenW = _module
compress = _module
custom_modules = _module
diffuser_training = _module
finetune_data = _module
get_deltas = _module
model = _module
retrieve = _module
sample_diffuser = _module
train = _module

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


import torch


import numpy as np


from torchvision.utils import make_grid


from torch import autocast


import random


import torchvision


from scipy.linalg import lu_factor


from scipy.linalg import lu_solve


import torch.nn as nn


import itertools


import math


from typing import Optional


import torch.nn.functional as F


import torch.utils.checkpoint


from torch.utils.data import Dataset


from torchvision import transforms


from torch import nn


from torch import einsum


import matplotlib.pyplot as plt


import time


from torch.utils.data import DataLoader


from functools import partial


class AbstractEncoder(nn.Module):

    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class FrozenCLIPEmbedderWrapper(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""

    def __init__(self, modifier_token, version='openai/clip-vit-large-patch14', device='cuda', max_length=77):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        self.modifier_token = modifier_token
        if '+' in self.modifier_token:
            self.modifier_token = self.modifier_token.split('+')
        else:
            self.modifier_token = [self.modifier_token]
        self.add_token()
        self.freeze()

    def add_token(self):
        self.modifier_token_id = []
        token_embeds1 = self.transformer.get_input_embeddings().weight.data
        for each_modifier_token in self.modifier_token:
            num_added_tokens = self.tokenizer.add_tokens(each_modifier_token)
            modifier_token_id = self.tokenizer.convert_tokens_to_ids(each_modifier_token)
            self.modifier_token_id.append(modifier_token_id)
        self.transformer.resize_token_embeddings(len(self.tokenizer))
        token_embeds = self.transformer.get_input_embeddings().weight.data
        token_embeds[self.modifier_token_id[-1]] = torch.nn.Parameter(token_embeds[42170], requires_grad=True)
        if len(self.modifier_token) == 2:
            token_embeds[self.modifier_token_id[-2]] = torch.nn.Parameter(token_embeds[47629], requires_grad=True)
        if len(self.modifier_token) == 3:
            token_embeds[self.modifier_token_id[-3]] = torch.nn.Parameter(token_embeds[43514], requires_grad=True)

    def custom_forward(self, hidden_states, input_ids):
        """
        Returns:
        """
        input_shape = hidden_states.size()
        bsz, seq_len = input_shape[:2]
        if version.parse(transformers.__version__) >= version.parse('4.21'):
            causal_attention_mask = self.transformer.text_model._build_causal_attention_mask(bsz, seq_len, hidden_states.dtype)
        else:
            causal_attention_mask = self.transformer.text_model._build_causal_attention_mask(bsz, seq_len)
        encoder_outputs = self.transformer.text_model.encoder(inputs_embeds=hidden_states, causal_attention_mask=causal_attention_mask)
        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.transformer.text_model.final_layer_norm(last_hidden_state)
        return last_hidden_state

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.transformer.text_model.encoder.parameters():
            param.requires_grad = False
        for param in self.transformer.text_model.final_layer_norm.parameters():
            param.requires_grad = False
        for param in self.transformer.text_model.embeddings.position_embedding.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True, return_overflowing_tokens=False, padding='max_length', return_tensors='pt')
        tokens = batch_encoding['input_ids']
        if len(self.modifier_token) == 3:
            indices = ((tokens == self.modifier_token_id[-1]) | (tokens == self.modifier_token_id[-2]) | (tokens == self.modifier_token_id[-3])) * 1
        elif len(self.modifier_token) == 2:
            indices = ((tokens == self.modifier_token_id[-1]) | (tokens == self.modifier_token_id[-2])) * 1
        else:
            indices = (tokens == self.modifier_token_id[-1]) * 1
        indices = indices.unsqueeze(-1)
        input_shape = tokens.size()
        tokens = tokens.view(-1, input_shape[-1])
        hidden_states = self.transformer.text_model.embeddings(input_ids=tokens)
        hidden_states = (1 - indices) * hidden_states.detach() + indices * hidden_states
        z = self.custom_forward(hidden_states, tokens)
        return z

    def encode(self, text):
        return self(text)

