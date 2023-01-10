import sys
_module = sys.modules[__name__]
del sys
benchmark_array_xd = _module
benchmark_getitem_100B = _module
benchmark_indices_mapping = _module
benchmark_iterating = _module
benchmark_map_filter = _module
format = _module
utils = _module
_config = _module
accuracy = _module
bertscore = _module
bleu = _module
bleurt = _module
cer = _module
test_cer = _module
chrf = _module
code_eval = _module
execute = _module
comet = _module
competition_math = _module
coval = _module
cuad = _module
evaluate = _module
exact_match = _module
f1 = _module
frugalscore = _module
glue = _module
google_bleu = _module
indic_glue = _module
mae = _module
mahalanobis = _module
matthews_correlation = _module
mauve = _module
mean_iou = _module
meteor = _module
mse = _module
pearsonr = _module
perplexity = _module
precision = _module
recall = _module
roc_auc = _module
rouge = _module
sacrebleu = _module
sari = _module
seqeval = _module
spearmanr = _module
squad = _module
squad_v2 = _module
record_evaluation = _module
super_glue = _module
ter = _module
wer = _module
wiki_split = _module
xnli = _module
xtreme_s = _module
setup = _module
datasets = _module
arrow_dataset = _module
arrow_reader = _module
arrow_writer = _module
builder = _module
combine = _module
commands = _module
convert = _module
datasets_cli = _module
dummy_data = _module
env = _module
run_beam = _module
test = _module
config = _module
data_files = _module
dataset_dict = _module
download = _module
download_config = _module
download_manager = _module
mock_download_manager = _module
streaming_download_manager = _module
features = _module
audio = _module
features = _module
image = _module
translation = _module
filesystems = _module
compression = _module
hffilesystem = _module
s3filesystem = _module
fingerprint = _module
formatting = _module
formatting = _module
jax_formatter = _module
np_formatter = _module
tf_formatter = _module
torch_formatter = _module
info = _module
inspect = _module
io = _module
abc = _module
csv = _module
generator = _module
json = _module
parquet = _module
sql = _module
text = _module
iterable_dataset = _module
keyhash = _module
load = _module
metric = _module
naming = _module
packaged_modules = _module
audiofolder = _module
folder_based_builder = _module
imagefolder = _module
pandas = _module
search = _module
splits = _module
streaming = _module
table = _module
tasks = _module
audio_classificiation = _module
automatic_speech_recognition = _module
base = _module
image_classification = _module
language_modeling = _module
question_answering = _module
summarization = _module
text_classification = _module
_hf_hub_fixes = _module
beam_utils = _module
deprecation_utils = _module
doc_utils = _module
extract = _module
file_utils = _module
filelock = _module
hub = _module
info_utils = _module
logging = _module
metadata = _module
patching = _module
py_utils = _module
readme = _module
resources = _module
sharding = _module
stratify = _module
tf_utils = _module
typing = _module
version = _module
new_dataset_script = _module
tests = _module
_test_patching = _module
conftest = _module
test_dummy_data = _module
test_test = _module
test_array_xd = _module
test_audio = _module
test_features = _module
test_image = _module
fixtures = _module
files = _module
fsspec = _module
test_csv = _module
test_json = _module
test_parquet = _module
test_sql = _module
test_text = _module
test_audiofolder = _module
test_folder_based_builder = _module
test_imagefolder = _module
test_arrow_dataset = _module
test_arrow_reader = _module
test_arrow_writer = _module
test_beam = _module
test_builder = _module
test_data_files = _module
test_dataset_dict = _module
test_dataset_list = _module
test_dataset_scripts = _module
test_download_manager = _module
test_dummy_data_autogenerate = _module
test_extract = _module
test_file_utils = _module
test_filelock = _module
test_filesystem = _module
test_fingerprint = _module
test_formatting = _module
test_hf_gcp = _module
test_hub = _module
test_info = _module
test_info_utils = _module
test_inspect = _module
test_iterable_dataset = _module
test_load = _module
test_logging = _module
test_metadata_util = _module
test_metric = _module
test_metric_common = _module
test_offline_util = _module
test_patching = _module
test_py_utils = _module
test_readme_util = _module
test_search = _module
test_sharding_utils = _module
test_splits = _module
test_streaming_download_manager = _module
test_table = _module
test_tasks = _module
test_upstream_hub = _module
test_version = _module
test_warnings = _module
release = _module

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


import numpy as np


from torch.nn import CrossEntropyLoss


import copy


import itertools


import re


import time


import warnings


from collections import Counter


from collections.abc import Mapping


from copy import deepcopy


from functools import partial


from functools import wraps


from math import ceil


from math import floor


from random import sample


from typing import TYPE_CHECKING


from typing import Any


from typing import BinaryIO


from typing import Callable


from typing import Dict


from typing import Iterable


from typing import Iterator


from typing import List


from typing import Optional


from typing import Tuple


from typing import Union


from typing import overload


import pandas as pd


from collections.abc import Iterable


from collections.abc import Sequence as SequenceABC


from functools import reduce


from typing import ClassVar


from typing import Sequence as Sequence_


from pandas.api.extensions import ExtensionArray as PandasExtensionArray


from pandas.api.extensions import ExtensionDtype as PandasExtensionDtype


from typing import Type


from collections.abc import MutableMapping


from typing import Generic


from typing import TypeVar


from itertools import cycle


from itertools import islice


import functools


import queue


import types


from queue import Empty


from types import CodeType


from types import FunctionType


import numpy.testing as npt


from itertools import chain


import inspect

