import sys
_module = sys.modules[__name__]
del sys
summarize = _module
server = _module
setup = _module
summarizer = _module
bert = _module
cluster_features = _module
sbert = _module
summary_processor = _module
text_processors = _module
coreference_handler = _module
sentence_abc = _module
sentence_handler = _module
transformer_embeddings = _module
bert_embedding = _module
sbert_embedding = _module
util = _module
tests = _module
test_coreference = _module
test_sbert = _module
test_sentence_handler = _module
test_summary_items = _module

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


from typing import List


from typing import Union


import numpy as np


import torch


from numpy import ndarray

