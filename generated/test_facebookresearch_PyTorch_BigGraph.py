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

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, string, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
import numpy as np
from torch import Tensor
patch_functional()
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
    def forward(self, pos_scores: FloatTensorType, neg_scores: FloatTensorType) ->FloatTensorType:
        pass


LongTensorType = torch.Tensor


def _extract_intervals(offsets, sizes, data):
    """Select contiguous intervals of rows, given their offsets and sizes.
    E.g. suppose offsets = [35, 70, 90], sizes = [3, 2, 4], then this will
    return

            (torch.LongTensor(0, 3, 5),
            data[torch.LongTensor([35, 36, 37, 70, 71, 90, 91, 92, 93])])

    """
    offsets = offsets.long()
    sizes = sizes.long()
    res_rows = sizes.sum().item()
    assert offsets.size(0) == sizes.size(0)
    non_zero_size = sizes != 0
    if non_zero_size.long().sum() == 0:
        return torch.zeros(offsets.size(0) + 1).long(), data.new()
    new_offsets = torch.cat([torch.LongTensor([0]), sizes.cumsum(0)])
    sizes_nz = sizes[non_zero_size]
    offsets_nz = offsets[non_zero_size]
    res_delta = torch.LongTensor(res_rows).fill_(1)
    res_delta[0] = offsets_nz[0]
    if offsets_nz.size(0) > 1:
        input_delta = offsets_nz[1:] - offsets_nz[:-1] - sizes_nz[:-1]
        res_row_offsets = sizes_nz.cumsum(0)[:-1]
        res_delta[res_row_offsets] += input_delta
    res_offsets = res_delta.cumsum(0)
    res = data[res_offsets]
    return new_offsets, res


class TensorList(object):
    """A list of tensors of different sizes, backed by a (offset, size, data)
    tuple.

    Indexing by LongTensor returns a new TensorList with the selected list
    elements (similar to indexing a torch index_select_).

    Indexing by an int returns a torch.Tensor with that list element.
    """

    @classmethod
    def cat(cls, elements):
        offsets, data = zip(*[[x.offsets, x.data] for x in elements])
        offsets = list(offsets)
        batch_offset = torch.LongTensor([o[-1] for o in offsets]).cumsum(0)
        for j in range(len(offsets) - 1):
            offsets[j + 1] = offsets[j + 1][1:] + batch_offset[j]
        return cls(torch.cat(offsets), torch.cat(data))

    @classmethod
    def empty(cls, num_tensors=0):
        return cls(torch.zeros((), dtype=torch.long).expand((num_tensors + 1,)), torch.empty((0,), dtype=torch.long))

    def new(self):
        return type(self)(self.offsets.new_zeros((1,)), self.data.new_empty((0,)))

    def __init__(self, offsets, data):
        assert isinstance(offsets, (torch.LongTensor, torch.cuda.LongTensor))
        assert offsets.ndimension() == 1
        assert offsets[0] == 0
        assert offsets[-1] == (data.size(0) if data.ndimension() > 0 else 0)
        if data.numel() == 0 and data.storage().size() == 0:
            storage = data.storage()
            storage.resize_(storage.size() + 1)
        self.offsets = offsets
        self.data = data

    def __getitem__(self, index):
        if isinstance(index, (torch.LongTensor, torch.cuda.LongTensor)):
            offsets_sub = self.offsets[index]
            sizes_sub = self.offsets[index + 1] - offsets_sub
            new_offsets, new_data = _extract_intervals(offsets_sub, sizes_sub, self.data)
            return TensorList(new_offsets, new_data)
        elif isinstance(index, int):
            if self.offsets[index] != self.offsets[index + 1]:
                return self.data[self.offsets[index]:self.offsets[index + 1]]
            else:
                return self.data.new()
        elif isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            if step != 1:
                raise ValueError('Expected slice with step 1, got %d' % step)
            new_offsets = self.offsets[start:stop + 1]
            new_data = self.data[new_offsets[0]:new_offsets[-1]]
            new_offsets = new_offsets - new_offsets[0]
            return TensorList(new_offsets, new_data)
        else:
            raise KeyError('Unknown index type: %s' % type(index))

    def __eq__(self, other):
        if not isinstance(other, TensorList):
            return NotImplemented
        return torch.equal(self.offsets, other.offsets) and torch.equal(self.data, other.data)

    def __len__(self):
        return self.offsets.size(0) - 1

    def __iadd__(self, other):
        if isinstance(other, int):
            self.data += other
            return self
        else:
            raise NotImplementedError()

    def __isub__(self, other):
        if isinstance(other, int):
            self.data -= other
            return self
        else:
            raise NotImplementedError()

    def size(self, dim=None):
        assert dim == 0 or dim is None, 'TensorList can only have 1 dimension'
        if dim is None:
            return torch.Size([len(self)])
        else:
            return len(self)

    def nelement(self):
        return self.data.nelement()

    def clone(self):
        return self.__class__(self.offsets, self.data.clone())

    def __repr__(self):
        if self.offsets.nelement() < 100 or self.data.nelement() < 1000:
            return 'TensorList( [%s] )' % ' , '.join(str(self[i].tolist()) for i in range(len(self)))
        return 'TensorList{offsets=%s, data=%s}' % (self.offsets, self.data)

    def apply(self, F):
        return self.__class__(self.offsets, F(self.data))

    def combine(self, other, F):
        if isinstance(other, TensorList):
            assert torch.equal(self.offsets, other.offsets)
            assert self.data.shape[0] == other.data.shape[0]
            res = self.__class__(self.offsets, F(self.data, other.data))
        else:
            res = self.__class__(self.offsets, F(self.data, other))
        assert res.data.shape[0] == self.data.shape[0]
        return res

    def lengths(self):
        return self.offsets[1:] - self.offsets[:-1]

    def unsqueeze(self, dim):
        return self.apply(lambda x: x.unsqueeze(dim))

    def view(self, *args):
        return self.apply(lambda x: x.view(*args))

    def __add__(self, other):
        return self.combine(other, lambda x, y: x + y)

    def __sub__(self, other):
        return self.combine(other, lambda x, y: x - y)

    def __mul__(self, other):
        return self.combine(other, lambda x, y: x * y)

    def __truediv__(self, other):
        return self.combine(other, lambda x, y: x / y)

    def sum(self, dim=None, keepdim=False):
        if dim is None:
            return self.data.sum()
        if dim < 0:
            dim = self.data.ndimension() + dim
        assert dim > 0, "Can't sum along the 'list' dimension"
        return self.__class__(self.offsets, self.data.sum(dim, keepdim=keepdim))

    def to(self, *args, **kwargs) ->'TensorList':
        return type(self)(self.offsets.to(*args, **kwargs), self.data.to(*args, **kwargs))


class EntityList:
    """Served as a wrapper of id-based entity and featurized entity.

    self.tensor is an id-based entity list
    self.tensor_list is a featurized entity list

    This class maintains the indexing and slicing of these two parallel
    representations.
    """

    @classmethod
    def empty(cls) ->'EntityList':
        return cls(torch.empty((0,), dtype=torch.long), TensorList.empty())

    @classmethod
    def from_tensor(cls, tensor: LongTensorType) ->'EntityList':
        if tensor.dim() != 1:
            raise ValueError('Expected 1D tensor, got %dD' % tensor.dim())
        tensor_list = TensorList.empty(num_tensors=tensor.shape[0])
        return cls(tensor, tensor_list)

    @classmethod
    def from_tensor_list(cls, tensor_list: TensorList) ->'EntityList':
        tensor = torch.full((len(tensor_list),), -1, dtype=torch.long)
        return cls(tensor, tensor_list)

    @classmethod
    def cat(cls, entity_lists: Sequence['EntityList']) ->'EntityList':
        return cls(torch.cat([el.tensor for el in entity_lists]), TensorList.cat(el.tensor_list for el in entity_lists))

    def __init__(self, tensor: LongTensorType, tensor_list: TensorList) ->None:
        if not isinstance(tensor, (torch.LongTensor, torch.cuda.LongTensor)):
            raise TypeError('Expected long tensor as first argument, got %s' % type(tensor))
        if not isinstance(tensor_list, TensorList):
            raise TypeError('Expected tensor list as second argument, got %s' % type(tensor_list))
        if tensor.dim() != 1:
            raise ValueError('Expected 1-dimensional tensor, got %d-dimensional one' % tensor.dim())
        if tensor.shape[0] != len(tensor_list):
            raise ValueError('The tensor and tensor list have different lengths: %d != %d' % (tensor.shape[0], len(tensor_list)))
        self.tensor: LongTensorType = tensor
        self.tensor_list: TensorList = tensor_list

    def to_tensor(self) ->LongTensorType:
        if len(self.tensor_list.data) != 0:
            raise RuntimeError('Getting the tensor data of an EntityList that also has tensor list data')
        return self.tensor

    def to_tensor_list(self) ->TensorList:
        if not self.tensor.eq(-1).all():
            raise RuntimeError('Getting the tensor list data of an EntityList that also has tensor data')
        return self.tensor_list

    def __eq__(self, other: Any) ->bool:
        if not isinstance(other, EntityList):
            return NotImplemented
        return torch.equal(self.tensor, other.tensor) and torch.equal(self.tensor_list.offsets, other.tensor_list.offsets) and torch.equal(self.tensor_list.data, other.tensor_list.data)

    def __str__(self) ->str:
        return repr(self)

    def __repr__(self) ->str:
        return 'EntityList(%r, TensorList(%r, %r))' % (self.tensor, self.tensor_list.offsets, self.tensor_list.data)

    def __getitem__(self, index: Union[int, slice, LongTensorType]) ->'EntityList':
        if isinstance(index, int):
            return self[index:index + 1]
        if isinstance(index, (torch.LongTensor, torch.cuda.LongTensor)) or isinstance(index, int):
            tensor_sub = self.tensor[index]
            tensor_list_sub = self.tensor_list[index]
            return type(self)(tensor_sub, tensor_list_sub)
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            if step != 1:
                raise ValueError('Expected slice with step 1, got %d' % step)
            tensor_sub = self.tensor[start:stop]
            tensor_list_sub = self.tensor_list[start:stop]
            return type(self)(tensor_sub, tensor_list_sub)
        raise KeyError('Unknown index type: %s' % type(index))

    def __len__(self) ->int:
        return self.tensor.shape[0]

    def to(self, *args, **kwargs) ->'EntityList':
        return type(self)(self.tensor.to(*args, **kwargs), self.tensor_list.to(*args, **kwargs))


class AbstractEmbedding(nn.Module, ABC):

    @abstractmethod
    def forward(self, input_: EntityList) ->FloatTensorType:
        pass

    @abstractmethod
    def get_all_entities(self) ->FloatTensorType:
        pass

    @abstractmethod
    def sample_entities(self, *dims: int) ->FloatTensorType:
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


class AbstractDynamicOperator(nn.Module, ABC):
    """Perform different operations on many vectors.

    The inputs are a tensor containing a set of vectors and another tensor
    specifying, for each vector, which operation to apply to it. The output has
    the same size as the first input and contains the outputs of the operations
    applied to the input vectors. The different operations are identified by
    integers in a [0, N) range. They are all of the same type (say, translation)
    but each one has its own set of parameters. The dimension of the vectors and
    the total number of operations that need to be supported are provided at
    initialization. The first tensor can have any number of dimensions (>= 1).

    """

    def __init__(self, dim: int, num_operations: int):
        super().__init__()
        self.dim = dim
        self.num_operations = num_operations

    @abstractmethod
    def forward(self, embeddings: FloatTensorType, operator_idxs: LongTensorType) ->FloatTensorType:
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
    def forward(self, lhs_pos: FloatTensorType, rhs_pos: FloatTensorType, lhs_neg: FloatTensorType, rhs_neg: FloatTensorType) ->Tuple[FloatTensorType, FloatTensorType, FloatTensorType]:
        pass


Partition = int


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


class Bucket(NamedTuple):
    lhs: Partition
    rhs: Partition

    def get_partition(self, side: Side) ->Partition:
        return side.pick(self.lhs, self.rhs)

    def __str__(self) ->str:
        return '( %d , %d )' % (self.lhs, self.rhs)


class EdgeList:

    @classmethod
    def empty(cls) ->'EdgeList':
        return cls(EntityList.empty(), EntityList.empty(), torch.empty((0,), dtype=torch.long))

    @classmethod
    def cat(cls, edge_lists: Sequence['EdgeList']) ->'EdgeList':
        cat_lhs = EntityList.cat([el.lhs for el in edge_lists])
        cat_rhs = EntityList.cat([el.rhs for el in edge_lists])
        if all(el.has_scalar_relation_type() for el in edge_lists):
            rel_types = {el.get_relation_type_as_scalar() for el in edge_lists}
            if len(rel_types) == 1:
                rel_type, = rel_types
                return cls(cat_lhs, cat_rhs, torch.tensor(rel_type, dtype=torch.long))
        cat_rel = torch.cat([el.rel.expand((len(el),)) for el in edge_lists])
        return EdgeList(cat_lhs, cat_rhs, cat_rel)

    def __init__(self, lhs: EntityList, rhs: EntityList, rel: LongTensorType) ->None:
        if not isinstance(lhs, EntityList) or not isinstance(rhs, EntityList):
            raise TypeError('Expected left- and right-hand side to be entity lists, got %s and %s instead' % (type(lhs), type(rhs)))
        if not isinstance(rel, (torch.LongTensor, torch.cuda.LongTensor)):
            raise TypeError('Expected relation to be a long tensor, got %s' % type(rel))
        if len(lhs) != len(rhs):
            raise ValueError('The left- and right-hand side entity lists have different lengths: %d != %d' % (len(lhs), len(rhs)))
        if rel.dim() > 1:
            raise ValueError('The relation can be either a scalar or a 1-dimensional tensor, got a %d-dimensional tensor' % rel.dim())
        if rel.dim() == 1 and rel.shape[0] != len(lhs):
            raise ValueError('The relation has a different length than the entity lists: %d != %d' % (rel.shape[0], len(lhs)))
        self.lhs = lhs
        self.rhs = rhs
        self.rel = rel

    def has_scalar_relation_type(self) ->bool:
        return self.rel.dim() == 0

    def get_relation_type_as_scalar(self) ->int:
        if self.rel.dim() != 0:
            raise RuntimeError("The relation isn't a scalar")
        return int(self.rel)

    def get_relation_type_as_vector(self) ->LongTensorType:
        if self.rel.dim() == 0:
            return self.rel.view((1,)).expand((len(self),))
        return self.rel

    def get_relation_type(self) ->Union[int, LongTensorType]:
        if self.has_scalar_relation_type():
            return self.get_relation_type_as_scalar()
        else:
            return self.get_relation_type_as_vector()

    def __eq__(self, other: Any) ->bool:
        if not isinstance(other, EdgeList):
            return NotImplemented
        return self.lhs == other.lhs and self.rhs == other.rhs and torch.equal(self.rel, other.rel)

    def __str__(self) ->str:
        return repr(self)

    def __repr__(self) ->str:
        return 'EdgeList(%r, %r, %r)' % (self.lhs, self.rhs, self.rel)

    def __getitem__(self, index: Union[int, slice, LongTensorType]) ->'EdgeList':
        if not isinstance(index, (int, slice, (torch.LongTensor, torch.cuda.LongTensor))):
            raise TypeError('Index can only be int, slice or long tensor, got %s' % type(index))
        if isinstance(index, (torch.LongTensor, torch.cuda.LongTensor)) and index.dim() != 1:
            raise ValueError('Long tensor index must be 1-dimensional, got %d-dimensional' % (index.dim(),))
        sub_lhs = self.lhs[index]
        sub_rhs = self.rhs[index]
        if self.has_scalar_relation_type():
            sub_rel = self.rel
        else:
            sub_rel = self.rel[index]
        return type(self)(sub_lhs, sub_rhs, sub_rel)

    def __len__(self) ->int:
        return len(self.lhs)

    def to(self, *args, **kwargs):
        return type(self)(self.lhs.to(*args, **kwargs), self.rhs.to(*args, **kwargs), self.rel.to(*args, **kwargs))


class BucketOrder(Enum):
    RANDOM = 'random'
    AFFINITY = 'affinity'
    INSIDE_OUT = 'inside_out'
    OUTSIDE_IN = 'outside_in'


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


FALSE_STRINGS = {'0', 'n', 'no', 'false', 'off'}


TRUE_STRINGS = {'1', 'y', 'yes', 'true', 'on'}


def mixed_case_to_lowercase(key: str) ->str:
    return ''.join('_%s' % c.lower() if c.isupper() else c for c in key)


TSchema = TypeVar('TSchema', bound='Schema')


logger = logging.getLogger('torchbiggraph')


EntityName = str


class FeaturizedEmbedding(AbstractEmbedding):

    def __init__(self, weight: nn.Parameter, max_norm: Optional[float]=None):
        super().__init__()
        self.weight: nn.Parameter = weight
        self.max_norm: Optional[float] = max_norm

    def forward(self, input_: EntityList) ->FloatTensorType:
        return self.get(input_.to_tensor_list())

    def get(self, input_: TensorList) ->FloatTensorType:
        if input_.size(0) == 0:
            return torch.empty((0, self.weight.size(1)))
        return F.embedding_bag(input_.data.long(), self.weight, input_.offsets[:-1], max_norm=self.max_norm, sparse=True)

    def get_all_entities(self) ->FloatTensorType:
        raise NotImplementedError('Cannot list all entities for featurized entities')

    def sample_entities(self, *dims: int) ->FloatTensorType:
        raise NotImplementedError('Cannot sample entities for featurized entities.')


Mask = List[Tuple[Union[int, slice, Sequence[int], LongTensorType], ...]]


class Negatives(Enum):
    NONE = 'none'
    UNIFORM = 'uniform'
    BATCH_UNIFORM = 'batch_uniform'
    ALL = 'all'


class Scores(NamedTuple):
    lhs_pos: FloatTensorType
    rhs_pos: FloatTensorType
    lhs_neg: FloatTensorType
    rhs_neg: FloatTensorType


class SimpleEmbedding(AbstractEmbedding):

    def __init__(self, weight: nn.Parameter, max_norm: Optional[float]=None):
        super().__init__()
        self.weight: nn.Parameter = weight
        self.max_norm: Optional[float] = max_norm

    def forward(self, input_: EntityList) ->FloatTensorType:
        return self.get(input_.to_tensor())

    def get(self, input_: LongTensorType) ->FloatTensorType:
        return F.embedding(input_, self.weight, max_norm=self.max_norm, sparse=True)

    def get_all_entities(self) ->FloatTensorType:
        return self.get(torch.arange(self.weight.size(0), dtype=torch.long, device=self.weight.device))

    def sample_entities(self, *dims: int) ->FloatTensorType:
        return self.get(torch.randint(low=0, high=self.weight.size(0), size=dims, device=self.weight.device))


def ceil_of_ratio(num: int, den: int) ->int:
    return (num - 1) // den + 1


def match_shape(tensor: torch.Tensor, *expected_shape: Union[int, type(Ellipsis)]) ->Union[None, int, Tuple[int, ...]]:
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
        raise RuntimeError("Some arguments aren't ints or ellipses: %s" % (expected_shape,))
    actual_shape = tensor.size()
    error = TypeError("Shape doesn't match: (%s) != (%s)" % (', '.join('%d' % d for d in actual_shape), ', '.join('...' if d is Ellipsis else '*' if d < 0 else '%d' % d for d in expected_shape)))
    if Ellipsis not in expected_shape:
        if len(actual_shape) != len(expected_shape):
            raise error
    else:
        if expected_shape.count(Ellipsis) > 1:
            raise RuntimeError('Two or more ellipses in %s' % (tuple(expected_shape),))
        if len(actual_shape) < len(expected_shape) - 1:
            raise error
        pos = expected_shape.index(Ellipsis)
        expected_shape = expected_shape[:pos] + actual_shape[pos:pos + 1 - len(expected_shape)] + expected_shape[pos + 1:]
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

