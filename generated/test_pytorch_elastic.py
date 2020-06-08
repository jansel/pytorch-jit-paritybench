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
api = _module
local_elastic_agent = _module
launch = _module
etcd_rendezvous = _module
etcd_server = _module
parameters = _module
local_timer = _module
cycling_iterator = _module
elastic_distributed_sampler = _module
version = _module

from _paritybench_helpers import _mock_config
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


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


from torch.nn.parallel import DistributedDataParallel


from torch.optim import SGD


from torch.utils.data import DataLoader


import math


from torch.utils.data.distributed import DistributedSampler


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_pytorch_elastic(_paritybench_base):
    pass
