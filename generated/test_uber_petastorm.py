import sys
_module = sys.modules[__name__]
del sys
conf = _module
examples = _module
hello_world = _module
external_dataset = _module
generate_external_dataset = _module
python_hello_world = _module
pytorch_hello_world = _module
tensorflow_hello_world = _module
test_generate_external_dataset = _module
petastorm_dataset = _module
generate_petastorm_dataset = _module
pyspark_hello_world = _module
test_generate_petastorm_dataset = _module
imagenet = _module
generate_petastorm_imagenet = _module
schema = _module
test_generate_imagenet_dataset = _module
mnist = _module
generate_petastorm_mnist = _module
pytorch_example = _module
tests = _module
conftest = _module
test_pytorch_mnist = _module
test_tf_mnist = _module
tf_example = _module
spark_dataset_converter = _module
pytorch_converter_example = _module
tensorflow_converter_example = _module
test_converter_example = _module
utils = _module
petastorm = _module
arrow_reader_worker = _module
benchmark = _module
cli = _module
dummy_reader = _module
throughput = _module
cache = _module
codecs = _module
compat = _module
errors = _module
etl = _module
dataset_metadata = _module
legacy = _module
metadata_util = _module
petastorm_generate_metadata = _module
rowgroup_indexers = _module
rowgroup_indexing = _module
fs_utils = _module
gcsfs_helpers = _module
gcsfs_wrapper = _module
generator = _module
hdfs = _module
namenode = _module
test_hdfs_namenode = _module
local_disk_arrow_table_cache = _module
local_disk_cache = _module
namedtuple_gt_255_fields = _module
ngram = _module
predicates = _module
py_dict_reader_worker = _module
pyarrow_helpers = _module
batching_table_queue = _module
test_batch_buffer = _module
pytorch = _module
reader = _module
reader_impl = _module
arrow_table_serializer = _module
pickle_serializer = _module
pyarrow_serializer = _module
pytorch_shuffling_buffer = _module
shuffling_buffer = _module
selectors = _module
spark = _module
spark_dataset_converter = _module
spark_utils = _module
test_util = _module
reader_mock = _module
shuffling_analysis = _module
bootstrap_test_schema_data = _module
generate_dataset_for_legacy_tests = _module
tempdir = _module
test_arrow_table_serializer = _module
test_benchmark = _module
test_cache = _module
test_codec_compressed_image = _module
test_codec_ndarray = _module
test_codec_scalar = _module
test_common = _module
test_copy_dataset = _module
test_dataset_metadata = _module
test_decode_row = _module
test_disk_cache = _module
test_end_to_end = _module
test_end_to_end_predicates_impl = _module
test_fs_utils = _module
test_generate_metadata = _module
test_metadata_read = _module
test_ngram = _module
test_ngram_end_to_end = _module
test_parquet_reader = _module
test_pickle_serializer = _module
test_predicates = _module
test_pyarrow_serializer = _module
test_pytorch_dataloader = _module
test_pytorch_utils = _module
test_reader = _module
test_reader_mock = _module
test_reading_legacy_datasets = _module
test_run_in_subprocess = _module
test_shuffling_buffer = _module
test_spark_dataset_converter = _module
test_spark_session_cli = _module
test_spark_utils = _module
test_tf_autograph = _module
test_tf_dataset = _module
test_tf_utils = _module
test_transform = _module
test_unischema = _module
test_weighted_sampling_reader = _module
tf_utils = _module
tools = _module
copy_dataset = _module
spark_session_cli = _module
transform = _module
unischema = _module
weighted_sampling_reader = _module
workers_pool = _module
dummy_pool = _module
exec_in_new_process = _module
process_pool = _module
stub_workers = _module
test_ventilator = _module
test_workers_pool = _module
thread_pool = _module
ventilator = _module
worker_base = _module
setup = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import numpy as np


import torch


import torch.nn as nn


import torch.nn.functional as F


import torch.optim as optim


from torchvision import transforms


import logging


from torch.autograd import Variable


import time


from collections import namedtuple


from functools import partial


import collections


import re


from torch.utils.data.dataloader import default_collate


import abc


from collections import deque


import uuid


import tensorflow.compat.v1 as tf


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view((-1, 1, 28, 28))
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Net,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 28, 28])], {}),
     True),
]

class Test_uber_petastorm(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

