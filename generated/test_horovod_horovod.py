import sys
_module = sys.modules[__name__]
del sys
conf = _module
mocks = _module
adasum_small_model = _module
pytorch_mnist_elastic = _module
pytorch_synthetic_benchmark_elastic = _module
tensorflow2_mnist_elastic = _module
tensorflow2_synthetic_benchmark_elastic = _module
tensorflow_keras_mnist_elastic = _module
keras_imagenet_resnet50 = _module
keras_mnist = _module
keras_mnist_advanced = _module
keras_spark3_rossmann = _module
keras_spark_mnist = _module
keras_spark_rossmann_estimator = _module
keras_spark_rossmann_run = _module
mxnet_imagenet_resnet50 = _module
mxnet_mnist = _module
pytorch_imagenet_resnet50 = _module
pytorch_mnist = _module
pytorch_spark_mnist = _module
pytorch_synthetic_benchmark = _module
tensorflow2_keras_mnist = _module
tensorflow2_mnist = _module
tensorflow2_synthetic_benchmark = _module
tensorflow_keras_mnist = _module
tensorflow_mnist = _module
tensorflow_mnist_eager = _module
tensorflow_mnist_estimator = _module
tensorflow_synthetic_benchmark = _module
tensorflow_word2vec = _module
horovod = _module
_keras = _module
callbacks = _module
elastic = _module
common = _module
basics = _module
exceptions = _module
util = _module
keras = _module
mxnet = _module
mpi_ops = _module
run = _module
service = _module
driver_service = _module
task_service = _module
codec = _module
config_parser = _module
env = _module
host_hash = _module
hosts = _module
network = _module
safe_shell_exec = _module
secret = _module
settings = _module
timeout = _module
tiny_shell_exec = _module
driver = _module
discovery = _module
registration = _module
rendezvous = _module
worker = _module
gloo_run = _module
http = _module
http_client = _module
http_server = _module
js_run = _module
mpi_run = _module
run_task = _module
runner = _module
task = _module
task_fn = _module
cache = _module
lsf = _module
threads = _module
spark = _module
_namedtuple_fix = _module
backend = _module
constants = _module
estimator = _module
params = _module
serialization = _module
store = _module
job_id = _module
mpirun_rsh = _module
rsh = _module
bare = _module
optimizer = _module
remote = _module
tensorflow = _module
gloo_exec_fn = _module
mpirun_exec_fn = _module
task_info = _module
torch = _module
util = _module
compression = _module
functions = _module
mpi_lib = _module
mpi_lib_impl = _module
optimizer = _module
sync_batch_norm = _module
setup = _module
run_safe_shell_exec = _module
sleep = _module
elastic_tensorflow2_main = _module
elastic_tensorflow_main = _module
elastic_torch_main = _module
elastic_common = _module
test_elastic_tensorflow = _module
test_elastic_tensorflow2 = _module
test_elastic_torch = _module
spark_common = _module
test_adasum_pytorch = _module
test_adasum_tensorflow = _module
test_buildkite = _module
test_common = _module
test_elastic_driver = _module
test_interactiverun = _module
test_keras = _module
test_mxnet = _module
test_run = _module
test_service = _module
test_spark = _module
test_spark_keras = _module
test_spark_torch = _module
test_stall = _module
test_tensorflow = _module
test_tensorflow2_keras = _module
test_tensorflow_keras = _module
test_timeline = _module
test_torch = _module

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


import torch


import random


import numpy as np


import torch.nn as nn


import torch.nn.functional as F


import torch.optim as optim


import torch.utils.data.distributed


import torch.backends.cudnn as cudnn


import torch.multiprocessing as mp


import math


import warnings


from torch.autograd.function import Function


from torch.nn.modules.batchnorm import _BatchNorm


from torch.nn import functional as F


import inspect


import itertools


from collections.abc import Iterable


class Net(torch.nn.Module):

    def __init__(self, mode='sq'):
        super(Net, self).__init__()
        if mode == 'square':
            self.mode = 0
            self.param = torch.nn.Parameter(torch.FloatTensor([1.0, -1.0]))
        else:
            self.mode = 1
            self.param = torch.nn.Parameter(torch.FloatTensor([1.0, -1.0, 1.0])
                )

    def forward(self, x):
        if ~self.mode:
            return x * x + self.param[0] * x + self.param[1]
        else:
            return 10 * x * x * x + self.param[0] * x * x + self.param[1
                ] * x + self.param[2]


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


class XOR(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(XOR, self).__init__()
        self.lin1 = nn.Linear(input_dim, 8)
        self.lin2 = nn.Linear(8, output_dim)

    def forward(self, features):
        x = features.float()
        x = self.lin1(x)
        x = torch.tanh(x)
        x = self.lin2(x)
        x = torch.sigmoid(x)
        return x


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_horovod_horovod(_paritybench_base):
    pass

    def test_000(self):
        self._check(XOR(*[], **{'input_dim': 4, 'output_dim': 4}), [torch.rand([4, 4, 4, 4])], {})
