import sys
_module = sys.modules[__name__]
del sys
aws_s3 = _module
helpers = _module
presets = _module
train = _module
utils = _module
conf = _module
examples = _module
librispeech = _module
common = _module
criteo_parquet = _module
criteo_tsv = _module
torcharrow_wrapper = _module
tsv_to_parquet = _module
train_loop = _module
train_loop_distributed_reading_service = _module
train_loop_reading_service = _module
train_loop_torchtext = _module
ag_news = _module
amazonreviewpolarity = _module
imdb = _module
squad1 = _module
squad2 = _module
vision = _module
caltech101 = _module
caltech256 = _module
imagefolder = _module
commitlist = _module
setup = _module
_create_fake_data = _module
_utils = _module
_common_utils_for_test = _module
elastic_training = _module
test_dataloader2 = _module
test_random = _module
smoke_test = _module
test_adapter = _module
test_aistore = _module
test_audio_examples = _module
test_dataframe = _module
test_distributed = _module
test_fsspec = _module
test_graph = _module
test_huggingface_datasets = _module
test_iterdatapipe = _module
test_linter = _module
test_local_io = _module
test_mapdatapipe = _module
test_period = _module
test_remote_io = _module
test_s3io = _module
test_serialization = _module
test_text_examples = _module
test_tfrecord = _module
tools = _module
gen_pyi = _module
setup_helpers = _module
extension = _module
todo = _module
torchdata = _module
_constants = _module
_extension = _module
dataloader2 = _module
adapter = _module
communication = _module
eventloop = _module
iter = _module
map = _module
messages = _module
protocol = _module
queue = _module
error = _module
graph = _module
_serialization = _module
settings = _module
linter = _module
reading_service = _module
shuffle_spec = _module
random = _module
worker = _module
datapipes = _module
iter = _module
load = _module
aisio = _module
fsspec = _module
huggingface = _module
iopath = _module
online = _module
s3io = _module
transform = _module
bucketbatcher = _module
callable = _module
util = _module
bz2fileloader = _module
cacheholder = _module
combining = _module
converter = _module
cycler = _module
dataframemaker = _module
decompressor = _module
distributed = _module
hashchecker = _module
header = _module
indexadder = _module
jsonparser = _module
mux_longest = _module
paragraphaggregator = _module
plain_text_reader = _module
prefetcher = _module
protobuf_template = _module
_tfrecord_example_pb2 = _module
randomsplitter = _module
rararchiveloader = _module
rows2columnar = _module
samplemultiplexer = _module
saver = _module
shardexpander = _module
tararchiveloader = _module
tfrecordloader = _module
webdataset = _module
xzfileloader = _module
zip_longest = _module
ziparchiveloader = _module
map = _module
converter = _module
unzipper = _module
utils = _module
_visualization = _module
janitor = _module

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


import itertools


import random


from functools import partial


import torch


import torch.distributed as dist


import torchvision


from torchvision.transforms import transforms


import time


import warnings


import torch.utils.data


from torch import nn


from collections import defaultdict


from collections import deque


from typing import Any


from typing import Callable


from typing import Dict


from typing import Iterable


from typing import Iterator


from typing import List


from typing import Optional


from typing import Tuple


from typing import TypeVar


from typing import Union


import numpy as np


from torch.utils.data import get_worker_info


from torch.utils.data.datapipes.dataframe.dataframes import CaptureLikeMock


from torch.utils.data.datapipes.dataframe import dataframe_wrapper as df_wrapper


import torch.nn as nn


import torchtext


import torchtext.functional as F


import torchtext.transforms as T


from torch.hub import load_state_dict_from_url


from torch.optim import AdamW


from torchtext.datasets import SST2


import re


from torch.utils.data.datapipes.utils.decoder import imagehandler


from torch.utils.data.datapipes.utils.decoder import mathandler


import torchvision.datasets as datasets


import torchvision.datasets.folder


import torchvision.transforms as transforms


from torch.utils.data import DataLoader


from torch.distributed.elastic.multiprocessing.errors import record


from torch.utils.data.datapipes.iter.grouping import SHARDING_PRIORITIES


from torch.testing._internal.common_utils import instantiate_parametrized_tests


from torch.testing._internal.common_utils import IS_WINDOWS


from torch.testing._internal.common_utils import parametrize


import torch.multiprocessing as mp


from torch.testing._internal.common_utils import slowTest


import queue


from torch.utils.data import IterDataPipe


import torch.utils.data.datapipes.iter


from torch.utils.data.datapipes.utils.snapshot import _simple_graph_snapshot_restoration


import functools


from torch.testing._internal.common_utils import IS_SANDCASTLE


from torch.utils.data.datapipes.utils.common import DILL_AVAILABLE


from torchtext.datasets import AG_NEWS


from torchtext.datasets import AmazonReviewPolarity


from torchtext.datasets import IMDB


from torchtext.datasets import SQuAD1


from torchtext.datasets import SQuAD2


from typing import Set


import torch.utils.data.datapipes.gen_pyi as core_gen_pyi


from torch.utils.data.datapipes.gen_pyi import gen_from_template


from torch.utils.data.datapipes.gen_pyi import get_method_definitions


from abc import abstractmethod


from torch.utils.data import MapDataPipe


import types


from torch.utils.data.graph import DataPipe


from torch.utils.data.graph import DataPipeGraph


from torch.utils.data.graph import traverse_dps


from torch.utils.data.datapipes.datapipe import _DataPipeSerializationWrapper


from torch.utils.data.datapipes.datapipe import _IterDataPipeSerializationWrapper


from torch.utils.data.datapipes.datapipe import _MapDataPipeSerializationWrapper


import inspect


from abc import ABC


from torch.utils.data import DataChunk


from torch.utils.data import functional_datapipe


from torch.utils.data.datapipes.iter import Batcher


from torch.utils.data.datapipes.iter import Collator


from torch.utils.data.datapipes.iter import Concater


from torch.utils.data.datapipes.iter import Demultiplexer


from torch.utils.data.datapipes.iter import FileLister


from torch.utils.data.datapipes.iter import FileOpener


from torch.utils.data.datapipes.iter import Filter


from torch.utils.data.datapipes.iter import Forker


from torch.utils.data.datapipes.iter import Grouper


from torch.utils.data.datapipes.iter import IterableWrapper


from torch.utils.data.datapipes.iter import Mapper


from torch.utils.data.datapipes.iter import Multiplexer


from torch.utils.data.datapipes.iter import RoutedDecoder


from torch.utils.data.datapipes.iter import Sampler


from torch.utils.data.datapipes.iter import ShardingFilter


from torch.utils.data.datapipes.iter import Shuffler


from torch.utils.data.datapipes.iter import StreamReader


from torch.utils.data.datapipes.iter import UnBatcher


from torch.utils.data.datapipes.iter import Zipper


from typing import Sequence


from torch.utils.data.datapipes.utils.common import match_masks


from typing import Generic


from typing import Hashable


from typing import Sized


from torch.utils.data.datapipes.utils.common import _check_unpickable_fn


from torch.utils.data.datapipes.utils.common import validate_input_col


import uuid


from typing import Deque


from enum import IntEnum


from collections import OrderedDict


from torch.utils.data.datapipes.iter.combining import _ChildDataPipe


from torch.utils.data.datapipes.iter.combining import _DemultiplexerIterDataPipe


from torch.utils.data.datapipes.iter.combining import _ForkerIterDataPipe


from torch.utils.data.datapipes._decorator import functional_datapipe


from torch.utils.data.datapipes.datapipe import IterDataPipe


from typing import cast


from typing import NamedTuple


from torch.utils.data.datapipes.map import Batcher


from torch.utils.data.datapipes.map import Concater


from torch.utils.data.datapipes.map import Mapper


from torch.utils.data.datapipes.map import SequenceWrapper


from torch.utils.data.datapipes.map import Shuffler


from torch.utils.data.datapipes.map import Zipper


from torch.utils.data.datapipes.utils.common import StreamWrapper


from typing import TYPE_CHECKING


from torch.utils.data.datapipes.iter.combining import IterDataPipe


class ToyModel(torch.nn.Module):

    def __init__(self) ->None:
        """
        In the model constructor, we instantiate four parameters and use them
        as member parameters.
        """
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn(()))
        self.b = torch.nn.Parameter(torch.randn(()))
        self.c = torch.nn.Parameter(torch.randn(()))
        self.d = torch.nn.Parameter(torch.randn(()))

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        """
        Simple model forward function
        """
        return self.a + self.b * x + self.c * x ** 2 + self.d * x ** 3


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ToyModel,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_pytorch_data(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

