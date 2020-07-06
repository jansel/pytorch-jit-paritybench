import sys
_module = sys.modules[__name__]
del sys
run = _module
setup = _module
powerful_benchmarker = _module
api_parsers = _module
api_cascaded_embeddings = _module
api_deep_adversarial_metric_learning = _module
api_train_with_classifier = _module
api_unsupervised_embeddings_using_augmentations = _module
base_api_parser = _module
architectures = _module
misc_models = _module
configs = _module
datasets = _module
cars196 = _module
celeb_a = _module
cub200 = _module
stanford_online_products = _module
runners = _module
base_runner = _module
bayes_opt_runner = _module
single_experiment_runner = _module
split_managers = _module
base_split_manager = _module
class_disjoint_split_manager = _module
closed_set_split_manager = _module
embedding_space_split_manager = _module
index_split_manager = _module
predefined_split_manager = _module
split_scheme_holder = _module
utils = _module
common_functions = _module
dataset_utils = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import torch


import numpy as np


import logging


import copy


from torch.utils.tensorboard import SummaryWriter


import torch.nn


from scipy import stats as scipy_stats


from collections import defaultdict


import torch.nn as nn


from torch.utils.data import Dataset


import scipy.io as sio


from torchvision.datasets.utils import download_url


from torchvision import datasets


from collections import OrderedDict


import itertools


import collections


import inspect


import torch.utils.data


class LayerExtractor(nn.Module):

    def __init__(self, convnet, keep_layers, skip_layers, insert_functions):
        super().__init__()
        self.convnet = convnet
        self.keep_layers = keep_layers
        self.skip_layers = skip_layers
        self.insert_functions = insert_functions
        self.pooler = nn.AdaptiveAvgPool2d((1, 1))
        for k in ['mean', 'std', 'input_space', 'input_range']:
            setattr(self, k, getattr(convnet, k, None))

    def forward(self, x):
        return self.layer_by_layer(x)

    def layer_by_layer(self, x, return_layer_sizes=False):
        outputs = []
        for layer_name, layer in self.convnet.named_children():
            if layer_name in self.skip_layers:
                continue
            x = layer(x)
            if layer_name in self.insert_functions:
                for zzz in self.insert_functions[layer_name]:
                    x = zzz(x)
            if layer_name in self.keep_layers:
                pooled_x = self.pooler(x).view(x.size(0), -1)
                outputs.append(pooled_x)
        output = torch.cat(outputs, dim=-1)
        if return_layer_sizes:
            return output, [x.size(-1) for x in outputs]
        return output


class ListOfModels(nn.Module):

    def __init__(self, list_of_models, input_sizes=None, operation_before_concat=None):
        super().__init__()
        self.list_of_models = nn.ModuleList(list_of_models)
        self.input_sizes = input_sizes
        self.operation_before_concat = (lambda x: x) if not operation_before_concat else operation_before_concat
        for k in ['mean', 'std', 'input_space', 'input_range']:
            setattr(self, k, getattr(list_of_models[0], k, None))

    def forward(self, x):
        outputs = []
        if self.input_sizes is None:
            for m in self.list_of_models:
                curr_output = self.operation_before_concat(m(x))
                outputs.append(curr_output)
        else:
            s = 0
            for i, y in enumerate(self.input_sizes):
                curr_input = x[:, s:s + y]
                curr_output = self.operation_before_concat(self.list_of_models[i](curr_input))
                outputs.append(curr_output)
                s += y
        return torch.cat(outputs, dim=-1)


class MLP(nn.Module):

    def __init__(self, layer_sizes, final_relu=False):
        super().__init__()
        layer_list = []
        layer_sizes = [int(x) for x in layer_sizes]
        num_layers = len(layer_sizes) - 1
        final_relu_layer = num_layers if final_relu else num_layers - 1
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            curr_size = layer_sizes[i + 1]
            if i < final_relu_layer:
                layer_list.append(nn.ReLU(inplace=True))
            layer_list.append(nn.Linear(input_size, curr_size))
        self.net = nn.Sequential(*layer_list)
        self.last_linear = self.net[-1]

    def forward(self, x):
        return self.net(x)


class Identity(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Identity,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ListOfModels,
     lambda: ([], {'list_of_models': [_mock_layer()]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MLP,
     lambda: ([], {'layer_sizes': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_KevinMusgrave_powerful_benchmarker(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

