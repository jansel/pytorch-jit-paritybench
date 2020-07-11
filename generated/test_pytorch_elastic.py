import sys
_module = sys.modules[__name__]
del sys
aws = _module
auth = _module
session = _module
autoscaling = _module
cloudformation = _module
petctl = _module
s3 = _module
util = _module
conf = _module
create_redirect_md = _module
main = _module
echo = _module
setup = _module
test = _module
server = _module
api_test = _module
local_elastic_agent_test = _module
distributed = _module
sleep_script = _module
test_script = _module
launch_test = _module
rpc = _module
api_test = _module
events = _module
metrics = _module
rendezvous = _module
etcd_rendezvous_test = _module
etcd_server_test = _module
parameters_test = _module
suites = _module
test_utils = _module
timer = _module
local_timer_example = _module
local_timer_test = _module
utils = _module
data = _module
cycling_iterator_test = _module
version_test = _module
torchelastic = _module
agent = _module
server = _module
api = _module
local_elastic_agent = _module
launch = _module
api = _module
rendezvous = _module
api = _module
etcd_rendezvous = _module
etcd_server = _module
parameters = _module
api = _module
local_timer = _module
cycling_iterator = _module
elastic_distributed_sampler = _module
version = _module

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


import time


from typing import List


from typing import Tuple


import numpy


import torch.backends.cudnn as cudnn


import torch.distributed as dist


import torch.nn as nn


import torch.nn.parallel


import torch.optim


import torch.utils.data


import torch.utils.data.distributed


import torchvision.datasets as datasets


import torchvision.models as models


import torchvision.transforms as transforms


from torch.nn.parallel import DistributedDataParallel


from torch.optim import SGD


from torch.utils.data import DataLoader


import uuid


from typing import Any


from typing import Dict


import torch.distributed.rpc as rpc


from torch.distributed.rpc.backend_registry import BackendType


from torch.multiprocessing import start_processes


from torch.distributed import register_rendezvous_handler


import logging


import torch.multiprocessing as torch_mp


import abc


from enum import Enum


from typing import Callable


import torch.multiprocessing as mp


from typing import Iterable


import torch.distributed.rpc as torch_rpc


from torch.distributed.rpc.constants import UNSET_RPC_TIMEOUT


import random


from typing import Optional


from torch.distributed import Store


from torch.distributed import TCPStore


from inspect import getframeinfo


from inspect import stack


from typing import Set


import math


from torch.utils.data.distributed import DistributedSampler

