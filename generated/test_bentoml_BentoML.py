import sys
_module = sys.modules[__name__]
del sys
conf = _module
request = _module
metric_defs = _module
runner_impl = _module
locustfile = _module
net = _module
service = _module
train = _module
utils = _module
lda = _module
model = _module
download_nltk_models = _module
download_model = _module
service = _module
mnist = _module
grid_search_cv = _module
linear_regression = _module
pipeline = _module
iris_classification = _module
mnist_torchscript = _module
model = _module
conftest = _module
test_prediction = _module
train = _module
agaricus = _module
client = _module
setup = _module
bentoml = _module
_internal = _module
bento = _module
build_config = _module
build_dev_bentoml_whl = _module
gen = _module
configuration = _module
containers = _module
helpers = _module
v1 = _module
container = _module
base = _module
buildah = _module
buildctl = _module
buildx = _module
docker = _module
dockerfile = _module
generate = _module
nerdctl = _module
podman = _module
context = _module
exportable = _module
external_typing = _module
starlette = _module
tensorflow = _module
transformers = _module
frameworks = _module
catboost = _module
common = _module
pytorch = _module
fastai = _module
keras = _module
lightgbm = _module
mlflow = _module
onnx = _module
picklable = _module
pytorch = _module
pytorch_lightning = _module
sklearn = _module
tensorflow_v2 = _module
torchscript = _module
onnx = _module
xgboost = _module
io_descriptors = _module
file = _module
image = _module
json = _module
multipart = _module
numpy = _module
pandas = _module
text = _module
log = _module
marshal = _module
dispatcher = _module
models = _module
monitoring = _module
api = _module
default = _module
otlp = _module
resource = _module
runner = _module
runnable = _module
runner_handle = _module
local = _module
remote = _module
strategy = _module
server = _module
base_app = _module
grpc = _module
servicer = _module
v1alpha1 = _module
grpc_app = _module
http = _module
access = _module
instruments = _module
http_app = _module
metrics = _module
prometheus = _module
runner_app = _module
inference_api = _module
loader = _module
openapi = _module
specification = _module
store = _module
tag = _module
types = _module
alg = _module
analytics = _module
cli_events = _module
schemas = _module
usage_stats = _module
benchmark = _module
cattr = _module
circus = _module
watchfilesplugin = _module
dotenv = _module
formparser = _module
lazy_loader = _module
pkg = _module
telemetry = _module
unflatten = _module
uri = _module
yatai_client = _module
yatai_rest_api_client = _module
config = _module
yatai = _module
bentos = _module
detectron = _module
easyocr = _module
evalml = _module
exceptions = _module
fasttext = _module
flax = _module
gluon = _module
interceptors = _module
opentelemetry = _module
_import_hook = _module
_generated_pb3 = _module
service_pb2 = _module
service_pb2_grpc = _module
_generated_pb4 = _module
h2o = _module
io = _module
onnxmlir = _module
paddle = _module
picklable_model = _module
pycaret = _module
pyspark = _module
serve = _module
spacy = _module
start = _module
statsmodels = _module
tensorflow_v1 = _module
testing = _module
pytest = _module
plugin = _module
bentoml_cli = _module
cli = _module
containerize = _module
env = _module
worker = _module
grpc_api_server = _module
grpc_dev_api_server = _module
grpc_prometheus_server = _module
http_api_server = _module
http_dev_api_server = _module
tests = _module
e2e = _module
context_server_interceptor = _module
python_model = _module
test_custom_components = _module
test_descriptors = _module
test_metrics = _module
pickle_model = _module
test_io = _module
test_meta = _module
integration = _module
mnist_autolog_example = _module
test_apis = _module
fastai = _module
onnx = _module
pytorch = _module
pytorch_lightning = _module
torchscript = _module
test_fastai_unit = _module
test_frameworks = _module
test_pytorch_unit = _module
test_tensorflow_unit = _module
test_transformers_unit = _module
proto = _module
service_test_pb2 = _module
service_test_pb2_grpc = _module
test_cli_regression = _module
unit = _module
bentoa = _module
bentob = _module
simplebento = _module
excluded = _module
test_bento = _module
test_containers = _module
test_helpers = _module
test_v1_migration = _module
test_base = _module
test_custom = _module
test_file = _module
test_image = _module
test_json = _module
test_multipart = _module
test_numpy = _module
test_text = _module
test_model = _module
test_container = _module
test_runner = _module
test_strategy = _module
test_resource = _module
test_service_api = _module
test_store = _module
test_utils = _module
test_watchfilesplugin = _module
test_analytics = _module
test_uri = _module
test_access = _module
test_prometheus = _module
test_grpc_utils = _module
test_models = _module

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


import torch.nn as nn


import torch.nn.functional as F


from typing import TYPE_CHECKING


import time


import numpy as np


import torch.optim as optim


import torch.utils.data as data


from torchvision import datasets


from torchvision import transforms


from torch.optim.lr_scheduler import StepLR


from sklearn.metrics import accuracy_score


from sklearn.datasets import load_iris


from sklearn.model_selection import train_test_split


import random


from torch import nn


from torchvision.datasets import MNIST


from sklearn.model_selection import KFold


import typing as t


import logging


import functools


import itertools


from types import ModuleType


from torch.nn import functional as F


from torch.utils.data import DataLoader


from torch.utils.data import random_split


import pandas as pd


from typing import Callable


import sklearn


from sklearn.ensemble import RandomForestClassifier


from sklearn.preprocessing import LabelEncoder


import torch.nn


import re


import torch.functional as F


class CNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
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


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
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


class IrisClassifier(nn.Module):

    def __init__(self):
        super(IrisClassifier, self).__init__()
        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, 0.2)
        x = self.fc3(x)
        return x


class SimpleConvNet(nn.Module):
    """
    Simple Convolutional Neural Network
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(1, 10, kernel_size=3), nn.ReLU(), nn.Flatten(), nn.Linear(26 * 26 * 10, 50), nn.ReLU(), nn.Linear(50, 20), nn.ReLU(), nn.Linear(20, 10))

    def forward(self, x):
        return self.layers(x)

    def predict(self, inp):
        """predict digit for input"""
        self.eval()
        with torch.no_grad():
            raw_output = self(inp)
            _, pred = torch.max(raw_output, 1)
            return pred


class LinearModel(nn.Module):

    def forward(self, x: t.Any) ->t.Any:
        return x


class Loss(Module):
    reduction = 'none'

    def forward(self, x: t.Any, _y: t.Any):
        return x

    def activation(self, x: t.Any):
        return x

    def decodes(self, x: t.Any):
        return x


class PyTorchModel(nn.Module):

    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(PyTorchModel, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data
        """
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


class _FakeLossFunc(Module):
    reduction = 'none'

    def forward(self, x: t.Any, y: t.Any):
        return F.mse_loss(x, y)

    def activation(self, x: t.Any):
        return x + 1

    def decodes(self, x: t.Any):
        return 2 * x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (IrisClassifier,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LinearModel,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Loss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (PyTorchModel,
     lambda: ([], {'D_in': 4, 'H': 4, 'D_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (_FakeLossFunc,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_bentoml_BentoML(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

    def test_004(self):
        self._check(*TESTCASES[4])

