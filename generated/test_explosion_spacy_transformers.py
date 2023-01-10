import sys
_module = sys.modules[__name__]
del sys
setup = _module
spacy_transformers = _module
annotation_setters = _module
architectures = _module
data_classes = _module
layers = _module
_util = _module
hf_shim = _module
hf_wrapper = _module
listener = _module
split_trf = _module
transformer_model = _module
trfs2arrays = _module
pipeline_component = _module
span_getters = _module
tests = _module
enable_gpu = _module
regression = _module
test_spacy_issue6401 = _module
test_spacy_issue7029 = _module
test_alignment = _module
test_configs = _module
test_data_classes = _module
test_deprecations = _module
test_model_sequence_classification = _module
test_model_wrapper = _module
test_pipeline_component = _module
test_serialize = _module
test_spanners = _module
test_tok2vectransformer = _module
test_truncation = _module
util = _module
truncate = _module
util = _module

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


from typing import Optional


from typing import List


from typing import Dict


from typing import Any


from typing import Union


from typing import Tuple


from typing import cast


import torch


import numpy


from typing import Callable


from functools import partial


import copy


from numpy.testing import assert_array_equal


from typing import Set


import random


import torch.cuda


import warnings

