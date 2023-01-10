import sys
_module = sys.modules[__name__]
del sys
conf = _module
generate = _module
image_classification = _module
language_modeling = _module
masked_language_modeling = _module
multiple_choice = _module
question_answering_squad = _module
summarization = _module
text_classification = _module
token_classification = _module
translation_wmt = _module
lightning_transformers = _module
callbacks = _module
sparseml = _module
core = _module
data = _module
finetuning = _module
iterable = _module
loggers = _module
model = _module
seq2seq = _module
model = _module
utils = _module
plugins = _module
checkpoint = _module
task = _module
nlp = _module
model = _module
datasets = _module
race = _module
swag = _module
data = _module
model = _module
utils = _module
question_answering = _module
data = _module
squad = _module
data = _module
metric = _module
processing = _module
model = _module
cnn_dailymail = _module
xsum = _module
data = _module
model = _module
model = _module
translation = _module
wmt16 = _module
vision = _module
model = _module
utilities = _module
deepspeed = _module
imports = _module
setup = _module
tests = _module
conftest = _module
test_nlp_model = _module
test_callbacks = _module
test_cli = _module
test_loggers = _module
test_model_step = _module
test_language_modeling = _module
test_masked_language_modeling = _module
test_multiple_choice = _module
test_pipeline = _module
test_question_answering = _module
test_summarization = _module
test_text_classification = _module
test_token_classification = _module
test_translation = _module
test_image_classification = _module

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


import collections


import inspect


from typing import Optional


import numpy


from torch import Tensor


from typing import Any


from typing import Callable


from typing import Dict


from typing import Union


from torch.utils.data import DataLoader


from typing import List


from torch.optim import Optimizer


from torch.utils.data import _DatasetKind


from torch.utils.data.dataloader import _InfiniteConstantSampler


from typing import IO


from typing import Tuple


from typing import Type


from functools import partial


from collections import OrderedDict


import numpy as np


import re


from torch.optim.lr_scheduler import LambdaLR


from torch.utils.data import Dataset

