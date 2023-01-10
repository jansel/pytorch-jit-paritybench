import sys
_module = sys.modules[__name__]
del sys
conftest = _module
deeplake = _module
api = _module
dataset = _module
info = _module
link = _module
read = _module
tests = _module
test_access_method = _module
test_agreement = _module
test_api = _module
test_api_tiling = _module
test_api_with_compression = _module
test_chunk_sizes = _module
test_connect_datasets = _module
test_dataset = _module
test_dicom = _module
test_downsample = _module
test_events = _module
test_grayscale = _module
test_hosted_datasets = _module
test_info = _module
test_insertion_out_of_order = _module
test_json = _module
test_link = _module
test_linking = _module
test_mesh = _module
test_meta = _module
test_none = _module
test_partial_upload = _module
test_pickle = _module
test_point_cloud = _module
test_polygons = _module
test_pop = _module
test_readonly = _module
test_rechunk = _module
test_sample_info = _module
test_text = _module
test_update_samples = _module
test_video = _module
tiled = _module
auto = _module
structured = _module
base = _module
dataframe = _module
test_ingestion = _module
test_kaggle = _module
unstructured = _module
image_classification = _module
kaggle = _module
cli = _module
auth = _module
commands = _module
list_datasets = _module
test_cli = _module
client = _module
config = _module
log = _module
test_client = _module
utils = _module
compression = _module
constants = _module
core = _module
chunk = _module
base_chunk = _module
chunk_compressed_chunk = _module
sample_compressed_chunk = _module
test_chunk_compressed = _module
test_sample_compressed = _module
test_uncompressed = _module
uncompressed_chunk = _module
chunk_engine = _module
compute = _module
process = _module
provider = _module
ray = _module
serial = _module
thread = _module
dataset = _module
deeplake_cloud_dataset = _module
invalid_view = _module
view_entry = _module
fast_forwarding = _module
index = _module
test_index = _module
io = _module
ipc = _module
link_creds = _module
linked_chunk_engine = _module
linked_sample = _module
lock = _module
meta = _module
dataset_meta = _module
encode = _module
base_encoder = _module
byte_positions = _module
chunk_id = _module
creds = _module
sequence = _module
shape = _module
common = _module
test_byte_positions_encoder = _module
test_byte_positions_encoder_updates = _module
test_chunk_id_encoder = _module
test_shape_encoder = _module
test_shape_encoder_updates = _module
tile = _module
tensor_meta = _module
partial_reader = _module
partial_sample = _module
polygon = _module
query = _module
autocomplete = _module
filter = _module
test_autocomplete = _module
test_query = _module
sample = _module
serialize = _module
storage = _module
deeplake_memory_object = _module
gcs = _module
google_drive = _module
local = _module
lru_cache = _module
memory = _module
s3 = _module
test_storage_provider = _module
tensor = _module
tensor_link = _module
test_serialize = _module
test_compression = _module
test_compute = _module
test_io = _module
test_locking = _module
tiling = _module
deserialize = _module
optimizer = _module
sample_tiles = _module
test_optimizer = _module
transform = _module
test_transform = _module
transform_dataset = _module
transform_tensor = _module
version_control = _module
commit_chunk_set = _module
commit_diff = _module
commit_node = _module
dataset_diff = _module
test_merge = _module
test_version_control = _module
enterprise = _module
convert_to_libdeeplake = _module
dataloader = _module
libdeeplake_query = _module
test_pytorch = _module
util = _module
hooks = _module
htype = _module
integrations = _module
huggingface = _module
mmdet = _module
mmdet_ = _module
mmdet_utils = _module
pytorch = _module
common = _module
dataset = _module
pytorch = _module
shuffle_buffer = _module
test_huggingface = _module
test_mmdet = _module
test_pytorch = _module
test_pytorch_dataloader = _module
test_shuffle_buffer = _module
test_tensorflow = _module
test_wandb = _module
tf = _module
datasettotensorflow = _module
deeplake_tensorflow_dataset = _module
wandb = _module
wandb = _module
cache_fixtures = _module
client_fixtures = _module
dataset_fixtures = _module
path_fixtures = _module
storage_fixtures = _module
access_method = _module
agreement = _module
array_list = _module
assert_byte_indexes = _module
bugout_reporter = _module
bugout_token = _module
cache_chain = _module
casting = _module
check_installation = _module
check_latest_version = _module
class_label = _module
connect_dataset = _module
delete_entry = _module
diff = _module
downsample = _module
empty_sample = _module
encoder = _module
exceptions = _module
exif = _module
from_tfds = _module
generate_id = _module
hash = _module
image = _module
invalid_view_op = _module
iterable_ordered_dict = _module
iteration_warning = _module
join_chunks = _module
json = _module
keys = _module
logging = _module
merge = _module
modified = _module
notebook = _module
object_3d = _module
mesh = _module
mesh_parser = _module
mesh_reader = _module
object_base_3d = _module
ply_parser = _module
ply_parser_base = _module
ply_parsers = _module
ply_reader = _module
ply_reader_base = _module
ply_readers = _module
point_cloud = _module
read_3d_data = _module
path = _module
pretty_print = _module
remove_cache = _module
scheduling = _module
shape_interval = _module
shuffle = _module
split = _module
tag = _module
testing = _module
test_auto = _module
test_connect_dataset = _module
test_iterable_ordered_dict = _module
test_read = _module
test_shape_interval = _module
test_shuffle = _module
test_split = _module
test_token = _module
threading = _module
token = _module
video = _module
warnings = _module
visualizer = _module
test_playback = _module
test_visualizer = _module
video_streaming = _module
conf = _module
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


import uuid


from logging import warning


from typing import Any


from typing import Callable


from typing import Dict


from typing import List


from typing import Optional


from typing import Sequence


from typing import Tuple


from typing import Union


from functools import partial


import numpy as np


from time import time


from itertools import chain


import warnings


from abc import abstractmethod


from abc import ABC


from random import shuffle


from typing import Iterator


from itertools import cycle


from copy import copy


from warnings import warn


from numpy import nditer


from numpy import argmin


from numpy import array as nparray


from math import floor


from torch.utils.data import DataLoader


import torch


import math


from collections import OrderedDict


from typing import Iterable


import torch.utils.data


import torch.distributed as dist


from torch.multiprocessing import Queue


from torch.multiprocessing import Process


from torch._utils import ExceptionWrapper


from torch.utils.data.dataloader import DataLoader


from torch.utils.data._utils.worker import ManagerWatchdog


from torch._C import _remove_worker_pids


from torch._C import _set_worker_pids


from torch._C import _set_worker_signal_handlers


from torch.utils.data._utils.signal_handling import _set_SIGCHLD_handler


from queue import Empty


from random import randrange


from functools import reduce


import numpy


from typing import Set

