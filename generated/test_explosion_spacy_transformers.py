import sys
_module = sys.modules[__name__]
del sys
init_model = _module
glue_util = _module
metrics = _module
run_glue = _module
test_wordpiece_alignment = _module
train_textcat = _module
setup = _module
spacy_transformers = _module
_tokenizers = _module
_train = _module
activations = _module
hyper_params = _module
language = _module
model_registry = _module
pipeline = _module
textcat = _module
tok2vec = _module
wordpiecer = _module
util = _module
wrapper = _module
tests = _module
conftest = _module
test_activations = _module
test_extensions = _module
test_language = _module
test_model_registry = _module
test_ner = _module
test_textcat = _module
test_tok2vec = _module
test_util = _module
test_wordpiecer = _module
test_wrapper = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import re, math, string, numpy, torch, torchtext, torchaudio, logging, itertools, numbers, inspect, functools, copy, scipy, types, time, torchvision, enum, random, typing, warnings, abc, collections, uuid
import numpy as np
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch.autograd


import torch.nn.utils.clip_grad


import torch


from typing import Tuple


from typing import Callable


from typing import Any


import numpy


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_explosion_spacy_transformers(_paritybench_base):
    pass
