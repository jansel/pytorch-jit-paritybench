import sys
_module = sys.modules[__name__]
del sys
conf = _module
setup = _module
test = _module
test_batching = _module
test_bucket_scheduling = _module
test_checkpoint_manager = _module
test_distributed = _module
test_edgelist = _module
test_entitylist = _module
test_functional = _module
test_graph_storages = _module
test_losses = _module
test_model = _module
test_optimizers = _module
test_schema = _module
test_stats = _module
test_train = _module
test_util = _module
torchbiggraph = _module
async_adagrad = _module
batching = _module
bucket_scheduling = _module
checkpoint_manager = _module
checkpoint_storage = _module
config = _module
converters = _module
dictionary = _module
export_to_tsv = _module
import_from_parquet = _module
import_from_tsv = _module
importers = _module
utils = _module
distributed = _module
edgelist = _module
entitylist = _module
eval = _module
examples = _module
fb15k_config_cpu = _module
fb15k_config_gpu = _module
livejournal_config = _module
fb15k = _module
livejournal = _module
filtered_eval = _module
graph_storages = _module
losses = _module
model = _module
parameter_sharing = _module
partitionserver = _module
plugin = _module
row_adagrad = _module
rpc = _module
schema = _module
stats = _module
tensorlist = _module
train = _module
train_cpu = _module
train_gpu = _module
types = _module
util = _module

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


import logging


import torch


import torch.multiprocessing as mp


import torch.nn as nn


from torch.optim import Adagrad


import time


from functools import partial


from typing import Callable


from typing import Generator


from typing import List


from typing import Optional


from typing import Tuple


from abc import ABC


from abc import abstractmethod


from torch import nn as nn


from torch.nn import functional as F


from enum import Enum


from typing import Dict


from typing import NamedTuple


from typing import Sequence


from typing import Union


import torch.nn.functional as F


import math


from collections import defaultdict


from typing import Any


from typing import Iterable


from typing import Set


import torch.distributed as td


from torch.optim import Optimizer


from typing import Mapping


from typing import MutableMapping


from typing import TypeVar


import torch.multiprocessing


FloatTensorType = torch.Tensor


class AbstractLossFunction(nn.Module, ABC):
    """Calculate weighted loss of scores for positive and negative pairs.

    The inputs are a 1-D tensor of size P containing scores for positive pairs
    of entities (i.e., those among which an edge exists) and a P x N tensor
    containing scores for negative pairs (i.e., where no edge should exist). The
    pairs of entities corresponding to pos_scores[i] and to neg_scores[i,j] have
    at least one endpoint in common. The output is the loss value these scores
    induce. If the method supports weighting (as is the case for the logistic
    loss) all positive scores will be weighted by the same weight and so will
    all the negative ones.
    """

    def __init__(self, **kwargs):
        super().__init__()

    @abstractmethod
    def forward(self, pos_scores: FloatTensorType, neg_scores: FloatTensorType
        ) ->FloatTensorType:
        pass


class AbstractOperator(nn.Module, ABC):
    """Perform the same operation on many vectors.

    Given a tensor containing a set of vectors, perform the same operation on
    all of them, with a common set of parameters. The dimension of these vectors
    will be given at initialization (so that any parameter can be initialized).
    The input will be a tensor with at least one dimension. The last dimension
    will contain the vectors. The output is a tensor that will have the same
    size as the input.

    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    @abstractmethod
    def forward(self, embeddings: FloatTensorType) ->FloatTensorType:
        pass


class AbstractComparator(nn.Module, ABC):
    """Calculate scores between pairs of given vectors in a certain space.

    The input consists of four tensors each representing a set of vectors: one
    set for each pair of the product between <left-hand side vs right-hand side>
    and <positive vs negative>. Each of these sets is chunked into the same
    number of chunks. The chunks have all the same size within each set, but
    different sets may have chunks of different sizes (except the two positive
    sets, which have chunks of the same size). All the vectors have the same
    number of dimensions. In short, the four tensor have these sizes:

        L+: C x P x D     R+: C x P x D     L-: C x L x D     R-: C x R x D

    The output consists of three tensors:
    - One for the scores between the corresponding pairs in L+ and R+. That is,
      for each chunk on one side, each vector of that chunk is compared only
      with the corresponding vector in the corresponding chunk on the other
      side. Think of it as the "inner" product of the two sides, or a matching.
    - Two for the scores between R+ and L- and between L+ and R-, where for each
      pair of corresponding chunks, all the vectors on one side are compared
      with all the vectors on the other side. Think of it as a per-chunk "outer"
      product, or a complete bipartite graph.
    Hence the sizes of the three output tensors are:

        ⟨L+,R+⟩: C x P     R+ ⊗ L-: C x P x L     L+ ⊗ R-: C x P x R

    Some comparators may need to peform a certain operation in the same way on
    all input vectors (say, normalizing them) before starting to compare them.
    When some vectors are used as both positives and negatives, the operation
    should ideally only be performed once. For that to occur, comparators expose
    a prepare method that the user should call on the vectors before passing
    them to the forward method, taking care of calling it only once on
    duplicated inputs.

    """

    @abstractmethod
    def prepare(self, embs: FloatTensorType) ->FloatTensorType:
        pass

    @abstractmethod
    def forward(self, lhs_pos: FloatTensorType, rhs_pos: FloatTensorType,
        lhs_neg: FloatTensorType, rhs_neg: FloatTensorType) ->Tuple[
        FloatTensorType, FloatTensorType, FloatTensorType]:
        pass


class DeepTypeError(TypeError):

    def __init__(self, message):
        self.message = message
        self.path = ''

    def prepend_attr(self, attr: str):
        self.path = '.%s%s' % (attr, self.path)

    def prepend_index(self, idx: int):
        self.path = '[%d]%s' % (idx, self.path)

    def prepend_key(self, key):
        self.path = '[%r]%s' % (key, self.path)

    def __str__(self):
        path = self.path.lstrip('.')
        if not path:
            return self.message
        return '%s: %s' % (path, self.message)


def has_origin(type_, base_type):
    try:
        return issubclass(type_.__origin__, base_type)
    except (AttributeError, TypeError):
        return False


def unpack_optional(type_):
    try:
        candidate_arg, = set(type_.__args__) - {type(None)}
    except (AttributeError, LookupError, ValueError):
        raise TypeError('Not an optional type')
    if type_ != Optional[candidate_arg]:
        raise TypeError('Not an optional type')
    return candidate_arg


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_facebookresearch_PyTorch_BigGraph(_paritybench_base):
    pass
