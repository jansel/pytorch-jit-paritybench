import sys
_module = sys.modules[__name__]
del sys
conf = _module
examples = _module
multi_process_benchmark = _module
client = _module
endpoint = _module
app = _module
qa_client = _module
check_correctness = _module
resnet50_benchmark = _module
run_benchmark = _module
pipelineclient = _module
samples = _module
settings = _module
labels = _module
stream = _module
usernames = _module
analyze_sentiment = _module
analyze_tokens = _module
scrape = _module
pipelines = _module
schemas = _module
integrations = _module
test_smoke = _module
setup = _module
src = _module
deepsparse = _module
analyze = _module
benchmark = _module
benchmark_model = _module
benchmark_sweep = _module
ort_engine = _module
results = _module
stream_benchmark = _module
cpu = _module
engine = _module
image_classification = _module
annotate = _module
constants = _module
utils = _module
validation_script = _module
lib = _module
license = _module
log = _module
loggers = _module
async_logger = _module
base_logger = _module
function_logger = _module
helpers = _module
metric_functions = _module
built_ins = _module
multi_logger = _module
prometheus_logger = _module
python_logger = _module
open_pif_paf = _module
annotate = _module
pipelines = _module
sample_images = _module
pipeline = _module
computer_vision = _module
custom_pipeline = _module
embedding_extraction = _module
numpy_schemas = _module
server = _module
build_logger = _module
cli = _module
config = _module
config_hot_reloading = _module
server = _module
tasks = _module
timing = _module
inference_phases = _module
timer = _module
transformers = _module
eval_downstream = _module
haystack = _module
nodes = _module
loaders = _module
metrics = _module
mnli_text_classification = _module
question_answering = _module
text_classification = _module
token_classification = _module
zero_shot_text_classification = _module
pipelines_cli = _module
cli_helpers = _module
data = _module
extractor = _module
onnx = _module
version = _module
yolact = _module
annotate = _module
pipelines = _module
annotate = _module
utils = _module
yolo = _module
annotate = _module
coco_classes = _module
utils = _module
tests = _module
conftest = _module
test_pipelines = _module
test_built_ins = _module
test_async_logger = _module
test_function_logger = _module
test_helpers = _module
test_prometheus_logger = _module
test_python_logger = _module
data_helpers = _module
no_register = _module
no_task = _module
valid_dynamic_import = _module
test_bucketing = _module
test_computer_vision_pipelines = _module
test_custom_pipeline = _module
test_dynamic_batch_pipeline = _module
test_dynamic_import = _module
test_embedding_extraction = _module
test_numpy_schemas = _module
test_pipeline = _module
test_transformers = _module
test_timer = _module
test_mnli_text_classification = _module
test_twitter_nlp = _module
test_app = _module
test_build_logger = _module
test_config = _module
test_endpoints = _module
test_hot_reloading = _module
test_loggers = _module
test_benchmark = _module
test_check_hardware = _module
test_engine = _module
test_multi_engine = _module
test_run_inference = _module
engine_mocking = _module
test_data = _module
test_engine_mocking = _module
artifacts = _module
copyright = _module
docs_builder = _module

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


import logging


from typing import Optional


from typing import Dict


import torchvision


from torchvision import transforms


from torch.utils.data import DataLoader


import re


from types import ModuleType


from typing import Any


from typing import Callable


from typing import Sequence


from typing import Tuple


from typing import Union


import numpy


from typing import List


from typing import Type


import torch


from collections import Counter


from copy import deepcopy


import collections


import copy


import torch.nn.functional as F


import functools


import itertools


import random


import time


from re import escape

