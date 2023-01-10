import sys
_module = sys.modules[__name__]
del sys
src = _module
common = _module
dataset = _module
block = _module
corpus = _module
data_set = _module
formatter = _module
label_schema = _module
persistence = _module
engine = _module
page = _module
session = _module
reader = _module
reverse_index = _module
s3 = _module
index = _module
iterator = _module
features = _module
feature_function = _module
vocab = _module
word_splitter = _module
framework = _module
task = _module
training = _module
batcher = _module
early_stopping = _module
options = _module
run = _module
util = _module
array = _module
log_helper = _module
random = _module
retrieval = _module
fever_doc_db = _module
filter_lists = _module
filter_uninformative = _module
retrieval_method = _module
sent_features = _module
sentence = _module
top_n = _module
rte = _module
parikh = _module
predictor = _module
riedel = _module
data = _module
fever_features = _module
fnc_features = _module
fnc_fever_transfer_features = _module
model = _module
scripts = _module
build_db = _module
build_tfidf = _module
balance = _module
block_to_jsonl = _module
block_to_sqlite = _module
download_dataset = _module
gents = _module
index_pages = _module
kappa = _module
makeblind = _module
neg_sample_evidence = _module
partition = _module
prepare_dataset = _module
redirects = _module
ts2 = _module
write = _module
manual_evaluation = _module
review_screen = _module
sample_review = _module
prepare_nltk = _module
prepare_submission = _module
document = _module
batch_ir = _module
batch_ir_ns = _module
eval_mrr = _module
eval_oracle = _module
eval_recall = _module
eval_recall_all = _module
ir = _module
eval_wmd = _module
eval_wmd2 = _module
mlp_train = _module
process = _module
process_tfidf = _module
process_tfidf_drqa = _module
process_tfidf_grid = _module
sentence_train = _module
test = _module
train = _module
da = _module
eval_da = _module
eval_snli = _module
interactive = _module
train_da = _module
mlp = _module
eval_mlp = _module
fnc_fever_riedel = _module
fnc_riedel = _module
train_mlp = _module
score = _module

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


from scipy.sparse import coo_matrix


from torch.autograd import Variable


import torch.nn.functional as F


from sklearn.utils import shuffle


from sklearn.metrics import accuracy_score


from sklearn.metrics import confusion_matrix


from sklearn.metrics import classification_report


import random


from torch import nn


class SimpleMLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, keep_p=0.6):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.do = nn.Dropout(1 - keep_p)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.do(x)
        x = self.fc2(x)
        x = self.do(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (SimpleMLP,
     lambda: ([], {'input_dim': 4, 'hidden_dim': 4, 'output_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_sheffieldnlp_naacl2018_fever(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

