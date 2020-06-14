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
text_classification = _module
unsupervised_learning = _module
experimental = _module
raw = _module

from _paritybench_helpers import _mock_config
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import logging


import torch


import time


from torch.utils.data import DataLoader


import torch.nn as nn


from torch.utils.data.dataset import random_split


from collections import Counter


import numpy as np


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


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_pytorch_text(_paritybench_base):
    pass
