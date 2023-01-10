import sys
_module = sys.modules[__name__]
del sys
a2t = _module
base = _module
data = _module
ace = _module
babeldomains = _module
tacred = _module
wikievents = _module
evaluation = _module
legacy = _module
relation_classification = _module
mnli = _module
run_evaluation = _module
run_glue = _module
utils = _module
topic_classification = _module
annotate_wordnet = _module
base = _module
finetune_classifier = _module
mlm = _module
mnli = _module
nsp = _module
wndomains = _module
tasks = _module
span_classification = _module
text_classification = _module
tuple_classification = _module
tests = _module
test_data = _module
test_inference = _module
test_tasks = _module
align_glosses_with_labels = _module
evaluate_re = _module
extract_patterns_re = _module
generate_silver_train = _module
output_mistakes = _module
sample = _module
split_tacred = _module
tacred2mnli = _module
threshold_estimation = _module
wikievents2mnli = _module
setup = _module

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


import numpy as np


import torch


from types import SimpleNamespace


from collections import defaultdict


from typing import Dict


from collections import Counter


from sklearn.metrics import precision_recall_fscore_support


from sklearn.metrics import confusion_matrix


from torch.utils.data import TensorDataset


from torch.utils.data import DataLoader


from torch.optim import SGD


from torch.optim import AdamW


from torch.optim import Adam

