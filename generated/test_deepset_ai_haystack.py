import sys
_module = sys.modules[__name__]
del sys
haystack = _module
api = _module
application = _module
config = _module
controller = _module
errors = _module
http_error = _module
feedback = _module
router = _module
search = _module
utils = _module
elasticsearch_client = _module
database = _module
base = _module
elasticsearch = _module
memory = _module
sql = _module
finder = _module
indexing = _module
cleaning = _module
io = _module
reader = _module
farm = _module
transformers = _module
retriever = _module
tfidf = _module
setup = _module
conftest = _module
test_db = _module
test_document = _module
test_faq_retriever = _module
test_farm_reader = _module
test_finder = _module
test_imports = _module
test_in_memory_store = _module
test_tfidf_retriever = _module
Tutorial1_Basic_QA_Pipeline = _module
Tutorial2_Finetune_a_model_on_your_data = _module
Tutorial3_Basic_QA_Pipeline_without_Elasticsearch = _module
Tutorial4_FAQ_style_QA = _module

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

