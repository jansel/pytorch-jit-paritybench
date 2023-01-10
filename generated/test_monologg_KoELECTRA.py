import sys
_module = sys.modules[__name__]
del sys
processor = _module
ner = _module
seq_cls = _module
run_ner = _module
run_seq_cls = _module
run_squad = _module
src = _module
evaluate_v1_0 = _module
tokenization_hanbert = _module
tokenization_kobert = _module
utils = _module
build_openwebtext_pretraining_dataset = _module
build_pretraining_dataset = _module
configure_finetuning = _module
configure_pretraining = _module
finetune = _module
classification_metrics = _module
classification_tasks = _module
feature_spec = _module
preprocessing = _module
mrqa_official_eval = _module
qa_metrics = _module
qa_tasks = _module
squad_official_eval = _module
squad_official_eval_v1 = _module
scorer = _module
tagging_metrics = _module
tagging_tasks = _module
tagging_utils = _module
task = _module
task_builder = _module
model = _module
modeling = _module
optimization = _module
tokenization = _module
pretrain = _module
pretrain_data = _module
pretrain_helpers = _module
run_finetuning = _module
run_pretraining = _module
util = _module
training_utils = _module
tests = _module
test_hf_load = _module

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


import copy


import logging


import torch


from torch.nn import CrossEntropyLoss


from torch.utils.data import TensorDataset


import re


import numpy as np


from torch.utils.data import DataLoader


from torch.utils.data import RandomSampler


from torch.utils.data import SequentialSampler


import random


from torch.utils.data.distributed import DistributedSampler


import collections


from numpy.lib.function_base import average


from scipy.stats import pearsonr


from scipy.stats import spearmanr


from sklearn import metrics as sklearn_metrics

