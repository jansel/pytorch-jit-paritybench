import sys
_module = sys.modules[__name__]
del sys
master = _module
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
torchelastic = _module
distributed = _module
launch = _module

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


import time


from typing import List


from typing import Tuple


import numpy


import torch


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


from torch.distributed.elastic.utils.data import ElasticDistributedSampler


from torch.nn.parallel import DistributedDataParallel


from torch.optim import SGD


from torch.utils.data import DataLoader


from torch.distributed.launcher.api import elastic_launch


from torch.distributed.launcher.api import launch_agent


from torch.distributed.launcher.api import LaunchConfig


from torch.distributed.run import main as run_main

