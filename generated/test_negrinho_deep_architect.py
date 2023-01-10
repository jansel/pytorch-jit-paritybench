import sys
_module = sys.modules[__name__]
del sys
main = _module
deep_architect = _module
contrib = _module
communicators = _module
communicator = _module
file_communicator = _module
file_utils = _module
mongo_communicator = _module
mpi_communicator = _module
deep_learning_backend = _module
backend = _module
general_ops = _module
keras_ops = _module
pytorch_ops = _module
tensorflow_eager_ops = _module
tensorflow_keras_ops = _module
tensorflow_ops = _module
misc = _module
calibration_utils = _module
datasets = _module
augmentation = _module
cifar10_tf = _module
dataset = _module
loaders = _module
evaluators = _module
tensorflow = _module
classification = _module
gcloud_utils = _module
tpu_estimator_classification = _module
gpu_utils = _module
search_spaces = _module
cnn2d = _module
cnn3d = _module
common = _module
dnn = _module
rnn = _module
tensorflow_eager = _module
genetic_space = _module
hierarchical_space = _module
nasbench_space = _module
nasnet_space = _module
core = _module
helpers = _module
keras_support = _module
pytorch_support = _module
tensorflow_eager_support = _module
tensorflow_support = _module
hyperparameters = _module
modules = _module
search_logging = _module
searchers = _module
mcts = _module
random = _module
regularized_evolution = _module
smbo_mcts = _module
smbo_random = _module
successive_narrowing = _module
surrogates = _module
dummy = _module
hashing = _module
utils = _module
visualization = _module
dev = _module
enas = _module
evaluator = _module
enas_evaluator = _module
search_space = _module
common_ops = _module
enas_search_space = _module
searcher = _module
enas_common_ops = _module
enas_searcher = _module
estimator_classification = _module
nasbench_evaluator = _module
tpu_keras_evaluator = _module
arch_search = _module
search_space_factory = _module
google_communicator = _module
data_loader = _module
master = _module
worker = _module
dynet_support = _module
hyperband = _module
multiworking = _module
search = _module
mnist_dynet = _module
mnist_tf_keras = _module
main_deep_architect = _module
main_genetic = _module
main_get_code_from_docs = _module
main_update_code_in_docs = _module
conf = _module
kubernetes = _module
train_best_master = _module
main_keras = _module
main_pytorch = _module
main_tensorflow = _module
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


from math import ceil


import torch


import torch.nn as nn


import torch.nn.functional as F


import numpy as np


import tensorflow as tf


import torch.optim as optim


def _call_fn_on_pytorch_module(outputs, fn):

    def fn_iter(mx):
        if hasattr(mx, 'pyth_modules'):
            for pyth_m in mx.pyth_modules:
                fn(pyth_m)
        return False
    co.traverse_backward(outputs, fn_iter)


def get_pytorch_modules(outputs):
    all_modules = set()
    _call_fn_on_pytorch_module(outputs, all_modules.add)
    return all_modules


class PyTorchModel(nn.Module):
    """Encapsulates a network of modules of type :class:`deep_architect.helpers.pytorch_support.PyTorchModule`
    in a way that they can be used as :class:`torch.nn.Module`, e.g.,
    functionality to move the computation of the GPU or to get all the parameters
    involved in the computation are available.

    Using this class is the recommended way of wrapping a Pytorch architecture
    sampled from a search space. The topological order for evaluating for
    doing the forward computation of the architecture is computed by the
    container and cached for future calls to forward.

    Args:
        inputs (dict[str,deep_architect.core.Input]): Dictionary of names to inputs.
        outputs (dict[str,deep_architect.core.Output]): Dictionary of names to outputs.
    """

    def __init__(self, inputs, outputs, init_input_name_to_val):
        nn.Module.__init__(self)
        self.outputs = outputs
        self.inputs = inputs
        self._module_seq = co.determine_module_eval_seq(self.inputs)
        self.forward(init_input_name_to_val)
        modules = get_pytorch_modules(self.outputs)
        for i, m in enumerate(modules):
            self.add_module(str(i), m)

    def __call__(self, input_name_to_val):
        return self.forward(input_name_to_val)

    def forward(self, input_name_to_val):
        """Forward computation of the module that is represented through the
        graph of DeepArchitect modules.
        """
        input_to_val = {ix: input_name_to_val[name] for name, ix in self.inputs.items()}
        co.forward(input_to_val, self._module_seq)
        output_name_to_val = {name: ox.val for name, ox in self.outputs.items()}
        return output_name_to_val

