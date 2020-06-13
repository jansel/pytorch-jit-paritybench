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


def match_shape(tensor: torch.Tensor, *expected_shape: Union[int, type(
    Ellipsis)]) ->Union[None, int, Tuple[int, ...]]:
    """Compare the given tensor's shape with what you expect it to be.

    This function serves two goals: it can be used both to assert that the size
    of a tensor (or part of it) is what it should be, and to query for the size
    of the unknown dimensions. The former result can be achieved with:

        >>> match_shape(t, 2, 3, 4)

    which is similar to

        >>> assert t.size() == (2, 3, 4)

    except that it doesn't use an assert (and is thus not stripped when the code
    is optimized) and that it raises a TypeError (instead of an AssertionError)
    with an informative error message. It works with any number of positional
    arguments, including zero. If a dimension's size is not known beforehand
    pass a -1: no check will be performed and the size will be returned.

        >>> t = torch.empty(2, 3, 4)
        >>> match_shape(t, 2, -1, 4)
        3
        >>> match_shape(t, -1, 3, -1)
        (2, 4)

    If the number of dimensions isn't known beforehand, an ellipsis can be used
    as a placeholder for any number of dimensions (including zero). Their sizes
    won't be returned.

        >>> t = torch.empty(2, 3, 4)
        >>> match_shape(t, ..., 3, -1)
        4

    """
    if not all(isinstance(d, int) or d is Ellipsis for d in expected_shape):
        raise RuntimeError("Some arguments aren't ints or ellipses: %s" % (
            expected_shape,))
    actual_shape = tensor.size()
    error = TypeError("Shape doesn't match: (%s) != (%s)" % (', '.join('%d' %
        d for d in actual_shape), ', '.join('...' if d is Ellipsis else '*' if
        d < 0 else '%d' % d for d in expected_shape)))
    if Ellipsis not in expected_shape:
        if len(actual_shape) != len(expected_shape):
            raise error
    else:
        if expected_shape.count(Ellipsis) > 1:
            raise RuntimeError('Two or more ellipses in %s' % (tuple(
                expected_shape),))
        if len(actual_shape) < len(expected_shape) - 1:
            raise error
        pos = expected_shape.index(Ellipsis)
        expected_shape = expected_shape[:pos] + actual_shape[pos:pos + 1 -
            len(expected_shape)] + expected_shape[pos + 1:]
    unknown_dims: List[int] = []
    for actual_dim, expected_dim in zip(actual_shape, expected_shape):
        if expected_dim < 0:
            unknown_dims.append(actual_dim)
            continue
        if actual_dim != expected_dim:
            raise error
    if not unknown_dims:
        return None
    if len(unknown_dims) == 1:
        return unknown_dims[0]
    return tuple(unknown_dims)


def ceil_of_ratio(num: int, den: int) ->int:
    return (num - 1) // den + 1


T = TypeVar('T')


class Side(Enum):
    LHS = 0
    RHS = 1

    def pick(self, lhs: T, rhs: T) ->T:
        if self is Side.LHS:
            return lhs
        elif self is Side.RHS:
            return rhs
        else:
            raise NotImplementedError('Unknown side: %s' % self)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_facebookresearch_PyTorch_BigGraph(_paritybench_base):
    pass
