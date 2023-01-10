import sys
_module = sys.modules[__name__]
del sys
setup = _module
zshot = _module
config = _module
evaluation = _module
dataset = _module
fewrel = _module
med_mentions = _module
entities = _module
utils = _module
ontonotes = _module
onto_notes = _module
evaluator = _module
metrics = _module
rel_eval = _module
seqeval = _module
pipeline = _module
run_evaluation = _module
zshot_evaluate = _module
linker = _module
linker = _module
linker_blink = _module
linker_regen = _module
trie = _module
linker_smxm = _module
linker_tars = _module
mentions_extractor = _module
mentions_extractor = _module
mentions_extractor_flair = _module
mentions_extractor_smxm = _module
mentions_extractor_spacy = _module
mentions_extractor_tars = _module
ExtractorType = _module
pipeline_config = _module
relation_extractor = _module
relation_extractor_zsrc = _module
relations_extractor = _module
zsrc = _module
data_helper = _module
decide_entity_order = _module
zero_shot_rel_class = _module
tests = _module
test_datasets = _module
test_evaluation = _module
test_linker = _module
test_regen_linker = _module
test_smxm_linker = _module
test_tars_linker = _module
test_flair_mentions_extractor = _module
test_mention_extractor = _module
test_smxm_mentions_extractor = _module
test_spacy_mentions_extractor = _module
test_tars_mentions_extractor = _module
test_relations_extractor = _module
test_zshot = _module
test_displacy = _module
test_utils = _module
alignment_utils = _module
data_models = _module
entity = _module
relation = _module
relation_span = _module
span = _module
displacy = _module
colors = _module
relations_render = _module
templates = _module
file_utils = _module
models = _module
smxm = _module
data = _module
model = _module
utils = _module
tars = _module

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


from abc import ABC


from abc import abstractmethod


from typing import Iterator


from typing import List


from typing import Optional


from typing import Union


import torch


import warnings


from torch.utils.data import DataLoader


import numpy as np


from torch.nn.utils.rnn import pad_sequence


from torch.utils.data import Dataset


import torch.nn as nn


from typing import Any


from typing import Dict


from typing import Tuple


import random


from functools import partial

