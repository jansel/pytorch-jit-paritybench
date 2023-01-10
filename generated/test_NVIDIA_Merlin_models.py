import sys
_module = sys.modules[__name__]
del sys
conf = _module
examples = _module
usecases = _module
datasets = _module
advertising = _module
criteo = _module
dataset = _module
transformed = _module
ecommerce = _module
aliccp = _module
raw = _module
booking = _module
dressipi = _module
preprocessed = _module
large = _module
small = _module
transactions = _module
entertainment = _module
ratings = _module
movielens = _module
music_streaming = _module
social = _module
synthetic = _module
testing = _module
sequence_testing = _module
models = _module
_version = _module
api = _module
config = _module
schema = _module
implicit = _module
io = _module
lightfm = _module
loader = _module
backend = _module
tf = _module
blocks = _module
cross = _module
dlrm = _module
experts = _module
interaction = _module
mlp = _module
optimizer = _module
retrieval = _module
base = _module
matrix_factorization = _module
two_tower = _module
sampling = _module
cross_batch = _module
in_batch = _module
queue = _module
core = _module
aggregation = _module
combinators = _module
encoder = _module
index = _module
prediction = _module
tabular = _module
distributed = _module
inputs = _module
continuous = _module
embedding = _module
loader = _module
losses = _module
listwise = _module
pairwise = _module
metrics = _module
evaluation = _module
topk = _module
benchmark = _module
ranking = _module
utils = _module
outputs = _module
classification = _module
contrastive = _module
regression = _module
popularity = _module
prediction_tasks = _module
multi = _module
next_item = _module
transformers = _module
block = _module
transforms = _module
bias = _module
features = _module
negative_sampling = _module
noise = _module
regularization = _module
sequence = _module
tensor = _module
typing = _module
batch_utils = _module
repr_utils = _module
search_utils = _module
testing_utils = _module
tf_utils = _module
base = _module
mlp = _module
base = _module
continuous = _module
embedding = _module
tabular = _module
losses = _module
model = _module
base = _module
prediction_task = _module
aggregation = _module
base = _module
transformations = _module
typing = _module
data_utils = _module
examples_utils = _module
torch_utils = _module
constants = _module
dependencies = _module
doc_utils = _module
example_utils = _module
misc_utils = _module
nvt_utils = _module
registry = _module
schema_utils = _module
xgb = _module
setup = _module
tests = _module
common = _module
retrieval_config = _module
retrieval_tests_common = _module
retrieval_utils = _module
tests_utils = _module
conftest = _module
integration = _module
test_integration_retrieval = _module
test_ci_01_getting_started = _module
test_ci_02_dataschema = _module
test_ci_03_exploring_different_models = _module
test_ci_04_export_ranking_models = _module
test_ci_05_export_retrieval_model = _module
test_ci_06_advanced_own_architecture = _module
unit = _module
test_schema = _module
test_advertising = _module
test_ecommerce = _module
test_entertainment = _module
test_social = _module
test_synthetic = _module
test_implicit = _module
test_lightfm = _module
_conftest = _module
test_base = _module
test_matrix_factorization = _module
test_two_tower = _module
test_cross_batch = _module
test_in_batch = _module
test_cross = _module
test_dlrm = _module
test_interactions = _module
test_mlp = _module
test_optimizer = _module
test_aggregation = _module
test_combinators = _module
test_encoder = _module
test_index = _module
test_prediction = _module
test_tabular = _module
test_01_getting_started = _module
test_02_dataschema = _module
test_03_exploring_different_models = _module
test_04_export_ranking_models = _module
test_05_export_retrieval_model = _module
test_06_advanced_own_architecture = _module
test_07_train_traditional_models = _module
test_usecase_accelerate_training_by_lazyadam = _module
test_usecase_data_parallel = _module
test_usecase_ecommerce_session_based = _module
test_usecase_incremental_training_layer_freezing = _module
test_usecase_pretrained_embeddings = _module
test_usecase_retrieval_with_hpo = _module
test_usecase_transformers_next_item_prediction = _module
horovod = _module
test_horovod = _module
test_continuous = _module
test_embedding = _module
layers = _module
test_queue = _module
test_losses = _module
test_metrics_popularity = _module
test_metrics_topk = _module
test_benchmark = _module
test_ranking = _module
test_retrieval = _module
test_classification = _module
test_contrastive = _module
test_regression = _module
test_sampling = _module
test_topk = _module
test_multi_task = _module
test_next_item = _module
test_core = _module
test_loader = _module
test_public_api = _module
test_block = _module
test_transforms = _module
test_bias = _module
test_features = _module
test_negative_sampling = _module
test_noise = _module
test_sequence = _module
test_tensor = _module
test_batch = _module
test_dataset = _module
test_tf_utils = _module
_conftest = _module
test_base = _module
test_mlp = _module
test_continuous = _module
test_embedding = _module
test_tabular = _module
test_head = _module
test_model = _module
test_aggregation = _module
test_tabular = _module
test_transformations = _module
test_dataloader_utils = _module
test_losses = _module
test_public_api = _module
test_schema_utils = _module
test_xgboost = _module
versioneer = _module

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


import logging


from typing import Optional


from typing import Protocol


from typing import Union


import numpy as np


import tensorflow as tf


from typing import Sequence


from tensorflow.keras import backend


from tensorflow.keras.metrics import Mean


from tensorflow.keras.metrics import get as get_metric


from typing import List


from typing import Tuple


from tensorflow.keras.metrics import Metric


import abc


import inspect


from collections import OrderedDict


import torch


from torch.nn import Module


from abc import ABC


from functools import partial


from typing import Any


from typing import Callable


from typing import Dict


from typing import Text


from typing import Type


from torch.nn.modules.loss import _WeightedLoss


import copy


from collections import defaultdict


from typing import Iterable


from typing import cast


from functools import reduce


import typing


from torch.utils.data import DataLoader as PyTorchDataLoader


from torch.utils.data import Dataset


from torch.utils.data import IterableDataset


import itertools


import time


import pandas as pd


def filter_kwargs(kwargs, thing_with_kwargs, cascade_kwargs_if_possible=False, argspec_fn=inspect.getfullargspec):
    arg_spec = argspec_fn(thing_with_kwargs)
    if cascade_kwargs_if_possible and arg_spec.varkw is not None:
        return kwargs
    else:
        filter_keys = arg_spec.args
        filtered_dict = {filter_key: kwargs[filter_key] for filter_key in filter_keys if filter_key in kwargs}
        return filtered_dict


def right_shift_block(self, other):
    if isinstance(other, list):
        left_side = [FilterFeatures(other)]
    else:
        left_side = list(other) if isinstance(other, SequentialBlock) else [other]
    right_side = list(self) if isinstance(self, SequentialBlock) else [self]
    if hasattr(left_side[-1], 'output_size') and left_side[-1].output_size():
        _right_side = []
        x = left_side[-1].output_size()
        for module in right_side:
            if getattr(module, 'build', None):
                if 'parents' in inspect.signature(module.build).parameters:
                    build = module.build(x, left_side)
                else:
                    build = module.build(x)
                if build:
                    module = build
                x = module.output_size() if hasattr(module, 'output_size') else None
            _right_side.append(module)
        right_side = _right_side
    sequential = left_side + right_side
    need_moving_to_gpu = False
    if isinstance(self, torch.nn.Module):
        need_moving_to_gpu = need_moving_to_gpu or torch_utils.check_gpu(self)
    if isinstance(other, torch.nn.Module):
        need_moving_to_gpu = need_moving_to_gpu or torch_utils.check_gpu(other)
    out = SequentialBlock(*sequential)
    if getattr(left_side[-1], 'input_size', None) and left_side[-1].input_size:
        out.input_size = left_side[-1].input_size
    if need_moving_to_gpu:
        out
    return out


class EmbeddingBagWrapper(torch.nn.EmbeddingBag):

    def forward(self, input, **kwargs):
        if len(input.shape) == 1:
            input = input.unsqueeze(-1)
        return super().forward(input, **kwargs)


class SoftEmbedding(torch.nn.Module):
    """
    Soft-one hot encoding embedding technique, from https://arxiv.org/pdf/1708.00065.pdf
    In a nutshell, it represents a continuous feature as a weighted average of embeddings
    """

    def __init__(self, num_embeddings, embeddings_dim, emb_initializer=None):
        """

        Parameters
        ----------
        num_embeddings: Number of embeddings to use (cardinality of the embedding table).
        embeddings_dim: The dimension of the vector space for projecting the scalar value.
        embeddings_init_std: The standard deviation factor for normal initialization of the
            embedding matrix weights.
        emb_initializer: Dict where keys are feature names and values are callable to initialize
            embedding tables
        """
        assert num_embeddings > 0, 'The number of embeddings for soft embeddings needs to be greater than 0'
        assert embeddings_dim > 0, 'The embeddings dim for soft embeddings needs to be greater than 0'
        super(SoftEmbedding, self).__init__()
        self.embedding_table = torch.nn.Embedding(num_embeddings, embeddings_dim)
        if emb_initializer:
            emb_initializer(self.embedding_table.weight)
        self.projection_layer = torch.nn.Linear(1, num_embeddings, bias=True)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, input_numeric):
        input_numeric = input_numeric.unsqueeze(-1)
        weights = self.softmax(self.projection_layer(input_numeric))
        soft_one_hot_embeddings = (weights.unsqueeze(-1) * self.embedding_table.weight).sum(-2)
        return soft_one_hot_embeddings


class LabelSmoothCrossEntropyLoss(_WeightedLoss):
    """Constructor for cross-entropy loss with label smoothing

    Parameters:
    ----------
    smoothing: float
        The label smoothing factor. it should be between 0 and 1.
    weight: torch.Tensor
        The tensor of weights given to each class.
    reduction: str
        Specifies the reduction to apply to the output,
        possible values are `none` | `sum` | `mean`

    Adapted from https://github.com/NingAnMe/Label-Smoothing-for-CrossEntropyLoss-PyTorch
    """

    def __init__(self, weight: torch.Tensor=None, reduction: str='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth_one_hot(targets: torch.Tensor, n_classes: int, smoothing: float=0.0):
        assert 0 <= smoothing < 1, f'smoothing factor {smoothing} should be between 0 and 1'
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes), device=targets.device).fill_(smoothing / (n_classes - 1)).scatter_(1, targets.data.unsqueeze(1), 1.0 - smoothing)
        return targets

    def forward(self, inputs, targets):
        targets = LabelSmoothCrossEntropyLoss._smooth_one_hot(targets, inputs.size(-1), self.smoothing)
        lsm = inputs
        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)
        loss = -(targets * lsm).sum(-1)
        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'none':
            loss = loss
        else:
            raise ValueError(f'{self.reduction} is not supported, please choose one of the following values [`sum`, `none`, `mean`]')
        return loss


TABULAR_MODULE_PARAMS_DOCSTRING = """
    pre: Union[str, TabularTransformation, List[str], List[TabularTransformation]], optional
        Transformations to apply on the inputs when the module is called (so **before** `forward`).
    post: Union[str, TabularTransformation, List[str], List[TabularTransformation]], optional
        Transformations to apply on the inputs after the module is called (so **after** `forward`).
    aggregation: Union[str, TabularAggregation], optional
        Aggregation to apply after processing the `forward`-method to output a single Tensor.
"""


TabularData = Dict[str, torch.Tensor]


def calculate_batch_size_from_input_size(input_size):
    if isinstance(input_size, dict):
        input_size = [i for i in input_size.values() if isinstance(i, torch.Size)][0]
    return input_size[0]


TensorOrTabularData = typing.Union[torch.Tensor, TabularData]


def docstring_parameter(*args, extra_padding=None, **kwargs):

    def dec(obj):
        if extra_padding:

            def pad(value):
                return ('\n' + ' ' * extra_padding).join(value.split('\n'))
            nonlocal args, kwargs
            kwargs = {key: pad(value) for key, value in kwargs.items()}
            args = [pad(value) for value in args]
        obj.__doc__ = obj.__doc__.format(*args, **kwargs)
        return obj
    return dec


class LossMixin:
    """Mixin to use for `torch.Module`s that can calculate a loss."""

    def compute_loss(self, inputs: Union[torch.Tensor, TabularData], targets: Union[torch.Tensor, TabularData], compute_metrics: bool=True, **kwargs) ->torch.Tensor:
        """Compute the loss on a batch of data.

        Parameters
        ----------
        inputs: Union[torch.Tensor, TabularData]
            TODO
        targets: Union[torch.Tensor, TabularData]
            TODO
        compute_metrics: bool, default=True
            Boolean indicating whether or not to update the state of the metrics
            (if they are defined).
        """
        raise NotImplementedError()


class MetricsMixin:
    """Mixin to use for `torch.Module`s that can calculate metrics."""

    def calculate_metrics(self, inputs: Union[torch.Tensor, TabularData], targets: Union[torch.Tensor, TabularData], mode: str='val', forward=True, **kwargs) ->Dict[str, torch.Tensor]:
        """Calculate metrics on a batch of data, each metric is stateful and this updates the state.

        The state of each metric can be retrieved by calling the `compute_metrics` method.

        Parameters
        ----------
        inputs: Union[torch.Tensor, TabularData]
            TODO
        targets: Union[torch.Tensor, TabularData]
            TODO
        forward: bool, default True

        mode: str, default="val"

        """
        raise NotImplementedError()

    def compute_metrics(self, mode: str='val') ->Dict[str, Union[float, torch.Tensor]]:
        """Returns the current state of each metric.

        The state is typically updated each batch by calling the `calculate_metrics` method.

        Parameters
        ----------
        mode: str, default="val"

        Returns
        -------
        Dict[str, Union[float, torch.Tensor]]
        """
        raise NotImplementedError()

    def reset_metrics(self):
        """Reset all metrics."""
        raise NotImplementedError()


def _output_metrics(metrics):
    if len(metrics) == 1:
        return metrics[list(metrics.keys())[0]]
    return metrics


_all_cap_re = re.compile('([a-z0-9])([A-Z])')


_first_cap_re = re.compile('(.)([A-Z][a-z0-9]+)')


def camelcase_to_snakecase(name):
    s1 = _first_cap_re.sub('\\1_\\2', name)
    return _all_cap_re.sub('\\1_\\2', s1).lower()


def name_fn(name, inp):
    return '/'.join([name, inp]) if name else None


class LambdaModule(torch.nn.Module):

    def __init__(self, lambda_fn):
        super().__init__()
        import types
        assert isinstance(lambda_fn, types.LambdaType)
        self.lambda_fn = lambda_fn

    def forward(self, x):
        return self.lambda_fn(x)


class TableConfig:

    def __init__(self, vocabulary_size: int, dim: int, initializer: Optional[Callable[[torch.Tensor], None]]=None, combiner: Text='mean', name: Optional[Text]=None):
        if not isinstance(vocabulary_size, int) or vocabulary_size < 1:
            raise ValueError('Invalid vocabulary_size {}.'.format(vocabulary_size))
        if not isinstance(dim, int) or dim < 1:
            raise ValueError('Invalid dim {}.'.format(dim))
        if combiner not in ('mean', 'sum', 'sqrtn'):
            raise ValueError('Invalid combiner {}'.format(combiner))
        if initializer is not None and not callable(initializer):
            raise ValueError('initializer must be callable if specified.')
        self.initializer: Callable[[torch.Tensor], None]
        if initializer is None:
            self.initializer = partial(torch.nn.init.normal_, mean=0.0, std=0.05)
        else:
            self.initializer = initializer
        self.vocabulary_size = vocabulary_size
        self.dim = dim
        self.combiner = combiner
        self.name = name

    def __repr__(self):
        return 'TableConfig(vocabulary_size={vocabulary_size!r}, dim={dim!r}, combiner={combiner!r}, name={name!r})'.format(vocabulary_size=self.vocabulary_size, dim=self.dim, combiner=self.combiner, name=self.name)


class FeatureConfig:

    def __init__(self, table: TableConfig, max_sequence_length: int=0, name: Optional[Text]=None):
        self.table = table
        self.max_sequence_length = max_sequence_length
        self.name = name

    def __repr__(self):
        return 'FeatureConfig(table={table!r}, max_sequence_length={max_sequence_length!r}, name={name!r})'.format(table=self.table, max_sequence_length=self.max_sequence_length, name=self.name)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (LabelSmoothCrossEntropyLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.ones([4], dtype=torch.int64)], {}),
     True),
    (SoftEmbedding,
     lambda: ([], {'num_embeddings': 4, 'embeddings_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_NVIDIA_Merlin_models(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

