import sys
_module = sys.modules[__name__]
del sys
regenerate = _module
test_sort_yaml = _module
build_tools = _module
setup_helpers = _module
extension = _module
conf = _module
create_datasets = _module
iterable_train = _module
model = _module
predict = _module
spm_dataset = _module
train = _module
download_extract = _module
vocab = _module
setup = _module
test = _module
babi = _module
common = _module
assets = _module
torchtext_test_case = _module
data = _module
test_batch = _module
test_builtin_datasets = _module
test_dataset = _module
test_field = _module
test_functional = _module
test_metrics = _module
test_pipeline = _module
test_subword = _module
test_utils = _module
imdb = _module
language_modeling = _module
nli = _module
sequence_tagging = _module
sst = _module
test_build = _module
test_vocab = _module
translation = _module
trec = _module
torchtext = _module
batch = _module
dataset = _module
example = _module
field = _module
functional = _module
iterator = _module
metrics = _module
pipeline = _module
utils = _module
datasets = _module
babi = _module
text_classification = _module
unsupervised_learning = _module
experimental = _module
language_modeling = _module
raw = _module
text_classification = _module
text_classification = _module
functional = _module
vocab = _module

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


from torch.utils.cpp_extension import CppExtension


from torch.utils.cpp_extension import BuildExtension as TorchBuildExtension


import torch


import torchtext


import logging


from torchtext.datasets import text_classification


import time


from torch.utils.data import DataLoader


from torchtext.data.utils import ngrams_iterator


from torchtext.data.utils import get_tokenizer


from torchtext.utils import unicode_csv_reader


import torch.nn as nn


from torchtext.utils import download_from_url


from torchtext.utils import extract_archive


from torchtext.datasets.text_classification import URLS


from torchtext.data.functional import generate_sp_model


from torchtext.data.functional import load_sp_model


from torchtext.data.functional import sentencepiece_numericalizer


from torch.utils.data.dataset import random_split


from torchtext.vocab import build_vocab_from_iterator


import torchtext.data as data


from torchtext.datasets import AG_NEWS


from torch.testing import assert_allclose


from collections import Counter


from numpy.testing import assert_allclose


from torchtext.data.functional import sentencepiece_tokenizer


from torchtext.data.functional import custom_replace


from torchtext.data.functional import simple_space_split


from torchtext.data.metrics import bleu_score


from torchtext.datasets import SNLI


from torchtext.datasets import MultiNLI


from torchtext.datasets import XNLI


from torchtext.datasets.nli import ParsedTextField


from torchtext.datasets.nli import ShiftReduceField


from torchtext.data import Field


from torchtext.data import LabelField


from torchtext.data import Iterator


import numpy as np


import torchtext.data


from torchtext import vocab


from functools import partial


import torch.utils.data


from collections import OrderedDict


from itertools import chain


import re


import math


import random


import collections


from copy import deepcopy


from torchtext.vocab import Vocab


from torchtext.data.functional import numericalize_tokens_from_iterator


from collections import defaultdict


class TextSentiment(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_class):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        """
        Arguments:
            text: 1-D tensor representing a bag of text tensors
            offsets: a list of offsets to delimit the 1-D text tensor
                into the individual sequences.

        """
        return self.fc(self.embedding(text, offsets))


class ScriptableSP(torch.jit.ScriptModule):

    def __init__(self, model_path):
        super().__init__()
        self.spm = load_sp_model(model_path)

    @torch.jit.script_method
    def encode(self, input: str):
        return self.spm.Encode(input)

    @torch.jit.script_method
    def encode_as_ids(self, input: str):
        return self.spm.EncodeAsIds(input)

    @torch.jit.script_method
    def encode_as_pieces(self, input: str):
        return self.spm.EncodeAsPieces(input)

