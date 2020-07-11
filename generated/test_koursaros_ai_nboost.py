import sys
_module = sys.modules[__name__]
del sys
changelog = _module
docker = _module
conf = _module
nboost = _module
__version__ = _module
cli = _module
compat = _module
database = _module
defaults = _module
delegates = _module
exceptions = _module
helpers = _module
indexers = _module
base = _module
es = _module
solr = _module
logger = _module
maps = _module
plugins = _module
debug = _module
prerank = _module
qa = _module
distilbert = _module
rerank = _module
onnxbert = _module
shuffle = _module
transformers = _module
use = _module
proxy = _module
translators = _module
setup = _module
test_benchmark = _module
test_es_indexer = _module
test_proxy = _module
test_solr_indexer = _module
test_delegates = _module
test_onnx_bert_rerank = _module
test_pt_bert_model = _module
test_pt_distilbert_qa_model = _module

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


from typing import Tuple


import numpy as np


import torch


from typing import List


import torch.nn

