import sys
_module = sys.modules[__name__]
del sys
build_openwebtext_pretraining_dataset = _module
build_pretraining_dataset = _module
cmrc2018_drcd_evaluate = _module
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
utils = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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

