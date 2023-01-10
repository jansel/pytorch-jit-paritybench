import sys
_module = sys.modules[__name__]
del sys
datasets = _module
criteo = _module
criteo_dataset = _module
criteo_processor = _module
femnist = _module
femnist_dataset = _module
femnist_processor = _module
conf = _module
envisedge = _module
executor = _module
experiments = _module
dlrm = _module
data_processor = _module
net = _module
trainer = _module
regression = _module
net = _module
fedrec = _module
communication_interfaces = _module
abstract_comm_manager = _module
kafka_interface = _module
zeroMQ_interface = _module
data_models = _module
aggregator_state_model = _module
base_actor_state_model = _module
envis_module = _module
job_response_model = _module
job_submit_model = _module
messages = _module
state_tensors_model = _module
tensors_model = _module
trainer_state_model = _module
modules = _module
embeddings = _module
sigmoid = _module
torch_optimizer = _module
transforms = _module
multiprocessing = _module
jobber = _module
process_manager = _module
optimization = _module
corrected_sgd = _module
optimizer = _module
schedulers = _module
aggregator = _module
base_actor = _module
serialization = _module
serializable_interface = _module
serialization_strategy = _module
serializer_registry = _module
user_modules = _module
envis_aggregator = _module
envis_base_module = _module
envis_preprocessor = _module
envis_trainer = _module
envis_wrapper = _module
utilities = _module
cuda_utils = _module
logger = _module
random_state = _module
registry = _module
saver_utils = _module
worker_dataset = _module
fl_strategies = _module
fed_avg = _module
preprocess_data = _module
setup = _module
test = _module
tests = _module
integration_tests = _module
integration_test = _module
unit_tests = _module
test_message = _module
test_serializer = _module
train = _module
train_fl = _module

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


import numpy as np


import torch


from torch.utils.data import Dataset


from collections import defaultdict


from torch.multiprocessing import Manager


from torch.multiprocessing import Process


from torchvision import transforms


from torch import nn


from torch import sigmoid


from torch.nn.parameter import Parameter


from typing import Optional


import torch.nn as nn


import torch.nn.functional as F


from torch import Tensor


from torch.optim import optimizer


from typing import List


from torch.optim import Optimizer


from torch.optim.lr_scheduler import _LRScheduler


from abc import ABC


from abc import abstractclassmethod


from abc import abstractmethod


from random import randint


from typing import Dict


from typing import Tuple


from re import A


from sklearn import metrics


from typing import Any


import logging


from time import time


import random


import collections


import collections.abc


import inspect


import re


import time


from abc import abstractproperty


from copy import deepcopy


import functools


LOOKUP_DICT = collections.defaultdict(dict)


class Registrable(object):
    """
    This class is used to register an object definition in the registry,
    and check for new class objects. The object definition is stored in
    the registry under the key 'kind' and sub-key 'name' in the dictionary.
    
    The object definition is a dictionary that contains the class object
    and the arguments that are used to construct the object.

    Methods
    -----------
    type_name()
        Returns the type name of the object. This is used to identify the
        object in the registry.
    get_name()
        Returns the name of the object. This is used to identify the object
        in the registry.
    register_class_ref()
        Registers the class object in the registry.
    lookup_class_ref()
        Returns the class object from the registry. This is used to instantiate
        the object.

    """

    def __init__(self) ->None:
        pass

    @classmethod
    def type_name(cls):
        return cls.__module__ + '.' + cls.__name__

    @staticmethod
    def get_name(obj):
        if not callable(obj):
            obj = obj.__class__
        return obj.__module__ + '.' + obj.__name__

    @staticmethod
    def register_class_ref(class_ref, name=None):
        if name is None:
            assert issubclass(class_ref, Registrable), 'Annotated class must be a subclass of Registrable'
            name = class_ref.type_name()
        LOOKUP_DICT['class_map'][name] = class_ref
        return class_ref

    @staticmethod
    def lookup_class_ref(class_name):
        if class_name not in LOOKUP_DICT['class_map']:
            raise KeyError('No class found for "{}"'.format(class_name))
        return LOOKUP_DICT['class_map'][class_name]


class Serializable(Registrable, ABC):
    """
    Serializable is a parent class that ensures that the
    child classes can be serialized or deserialized.
    In simple terms, the serializable is converting a
    data object (e.g., Python objects, Tensorflow models)
    into a format that can be stored or transmitted, and
    then recreated using the reverse process of
    deserialization when needed.
    The serialize and deserialize functions first check for the
    base class property i.e whether it is a serializable class or
    not. If they are child classes of serializable then the further
    process is continued.

    Attributes
    -----------
    serializer: str
        The serializer to use.

    Methods
    --------
    serialize(obj):
        Serializes an object.
    deserialize(obj):
        Deserializes an object.
    """

    def __init__(self) ->None:
        super().__init__()

    @abstractmethod
    def serialize(self):
        raise NotImplementedError()

    @abstractmethod
    def deserialize(self):
        raise NotImplementedError()

    def append_type(self, obj_dict):
        """Generates a dictionary from an object and
         appends type information for finding the appropriate serialiser.

        Parameters
        -----------
        obj: object
            The object to serialize.

        Returns
        --------
        dict:
            The dictionary representation of the object.
        """
        return {'__type__': self.type_name(), '__data__': obj_dict}


class EnvisPreProcessor(Serializable):
    """
    The EnvisPreProcessor class extends the Serializable class and is used
    to preprocess the data before training. It preprocesses the data and
    stores it in the storage before it is used by the model.

    It uses the Serializable interface to serialize the data and store it
    in the storage, where the data is later loaded by the model. It adds
    the class to the registry using the Registrable interface.

    Arguments
    ---------
    dataset_config: dict
        The dataset configuration.
    client_id: str
        The client id. This is used to store the data in the storage.
    
    Methods
    -------
    preprocess_data: None
        Preprocesses the data. This method should be called before the data is
        loaded.
    load: None
        Loads the data from the storage.
    load_data_description: None
        Loads the data description from the storage.
    datasets: dict
        Returns the datasets.
    dataset: torch.utils.data.Dataset
        Returns the dataset for the given split. The split can be either
        'train', 'validation' or 'test'.
    data_loader: torch.utils.data.DataLoader
        Returns the data loader for the given dataset. The data loader can be
        configured using the kwargs.
    serialize: dict
        Returns the serialized data.
    deserialize: None
        Deserializes the data.
    """

    def __init__(self, dataset_config, client_id=None) ->None:
        super().__init__()
        self.client_id = client_id
        self.dataset_config = dataset_config
        self.dataset_processor = registry.construct('dataset', self.dataset_config, unused_keys=())

    def preprocess_data(self):
        self.dataset_processor.process_data()

    def load(self):
        self.dataset_processor.load(self.client_id)

    def load_data_description(self):
        pass

    def datasets(self, *splits):
        assert all([isinstance(split, str) for split in splits])
        return {split: self.dataset_processor.dataset(split) for split in splits}

    def dataset(self, split):
        assert isinstance(split, str)
        return self.dataset_processor.dataset(split)

    def data_loader(self, data, **kwargs):
        return torch.utils.data.DataLoader(data, **kwargs)

    def serialize(self):
        output = self.append_type({'proc_name': self.type_name(), 'client_id': self.client_id, 'dataset_config': self.dataset_config})
        return output

    @classmethod
    def deserialize(cls, obj):
        preproc_cls = Registrable.lookup_class_ref(obj['proc_name'])
        return preproc_cls(dataset_config=obj['dataset_config'], client_id=obj['client_id'])


class DLRMPreprocessor(EnvisPreProcessor):
    REGISTERED_NAME = 'dlrm'

    def __init__(self, dataset_config, client_id=0):
        super().__init__(dataset_config, client_id)
        self.m_den = None
        self.n_emb = None
        self.ln_emb = None

    def preprocess_data(self):
        self.dataset_processor.process_data()
        if not self.m_den:
            self.load_data_description()

    def load_data_description(self):
        self.dataset_processor.load_data_description()
        self.m_den = self.dataset_processor.m_den
        self.n_emb = self.dataset_processor.n_emb
        self.ln_emb = self.dataset_processor.ln_emb

    def data_loader(self, data, **kwargs):
        return torch.utils.data.DataLoader(data, collate_fn=self.dataset_processor.collate_fn, **kwargs)


def xavier_init(layer: nn.Linear):
    with torch.no_grad():
        mean = 0.0
        std_dev = np.sqrt(2 / (layer.out_features + layer.in_features))
        W = np.random.normal(mean, std_dev, size=(layer.out_features, layer.in_features)).astype(np.float32)
        std_dev = np.sqrt(1 / layer.out_features)
        bt = np.random.normal(mean, std_dev, size=layer.out_features).astype(np.float32)
        layer.weight.set_(torch.tensor(W))
        layer.bias.set_(torch.tensor(bt))
        return layer


class DLRM_Net(nn.Module):
    Preproc = DLRMPreprocessor

    def create_mlp(self, ln, sigmoid_layer):
        layers = [xavier_init(nn.Linear(ln[0], ln[1], True))]
        for in_f, out_f in zip(ln[1:], ln[2:]):
            layers += [registry.construct('sigmoid_layer', {'name': sigmoid_layer}), xavier_init(nn.Linear(in_f, out_f, True))]
        return torch.nn.Sequential(*layers)

    def create_emb(self, m, ln, emb_dict, weighted_pooling=None):
        emb_l = nn.ModuleList()
        v_W_l = []
        for i in range(0, ln.size):
            if emb_dict.get('custom', None) is not None and ln[i] > emb_dict['threshold']:
                EE = registry.construct('embedding', emb_dict['custom'], num_embeddings=ln[i], embedding_dim=m)
            else:
                EE = registry.construct('embedding', emb_dict['base'], num_embeddings=ln[i], embedding_dim=m)
            if weighted_pooling is None:
                v_W_l.append(None)
            else:
                v_W_l.append(torch.ones(ln[i], dtype=torch.float32))
            emb_l.append(EE)
        return emb_l, v_W_l

    def __init__(self, preprocessor: DLRMPreprocessor, arch_feature_emb_size=None, arch_mlp_bot=None, arch_mlp_top=None, arch_interaction_op=None, arch_interaction_itself=False, sigmoid_bot='relu', sigmoid_top='relu', loss_weights=None, loss_threshold=0.0, ndevices=-1, embedding_types={}, weighted_pooling=None, loss_function='bce'):
        super(DLRM_Net, self).__init__()
        self.preproc = preprocessor
        if arch_feature_emb_size is not None and self.preproc.ln_emb is not None and arch_mlp_bot is not None and arch_mlp_top is not None and arch_interaction_op is not None:
            self.ndevices = ndevices
            self.m_spa = arch_feature_emb_size
            self.ln_emb = self.preproc.ln_emb
            self.ln_bot = arch_mlp_bot + [self.m_spa]
            self.output_d = 0
            self.parallel_model_batch_size = -1
            self.parallel_model_is_not_prepared = True
            self.arch_interaction_op = arch_interaction_op
            self.arch_interaction_itself = arch_interaction_itself
            self.loss_threshold = loss_threshold
            self.loss_function = loss_function
            self.emb_dict = embedding_types
            num_fea = self.ln_emb.size + 1
            m_den = self.preproc.m_den
            self.ln_bot[0] = m_den
            if arch_interaction_op == 'dot':
                if arch_interaction_itself:
                    num_int = num_fea * (num_fea + 1) // 2 + self.ln_bot[-1]
                    offset = 1
                else:
                    num_int = num_fea * (num_fea - 1) // 2 + self.ln_bot[-1]
                    offset = 0
                self.index_tensor_i = torch.tensor([i for i in range(num_fea) for j in range(i + offset)])
                self.index_tensor_j = torch.tensor([j for i in range(num_fea) for j in range(i + offset)])
            elif arch_interaction_op == 'cat':
                num_int = num_fea * self.ln_bot[-1]
            else:
                sys.exit('ERROR: --arch-interaction-op=' + arch_interaction_op + ' is not supported')
            self.ln_top = [num_int] + arch_mlp_top
            self.emb_l, w_list = self.create_emb(self.m_spa, self.ln_emb, self.emb_dict, weighted_pooling)
            if weighted_pooling is not None and weighted_pooling != 'fixed':
                self.weighted_pooling = 'learned'
                self.v_W_l = nn.ParameterList()
                for w in w_list:
                    self.v_W_l.append(Parameter(w))
            else:
                self.weighted_pooling = weighted_pooling
                self.v_W_l = w_list
            self.bot_l = self.create_mlp(self.ln_bot, sigmoid_bot)
            self.top_l = self.create_mlp(self.ln_top, sigmoid_top)
            if self.loss_function == 'mse':
                self.loss_fn = torch.nn.MSELoss(reduction='mean')
            elif self.loss_function == 'bce':
                self.loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight=loss_weights)
            else:
                sys.exit('ERROR: --loss_function=' + self.loss_function + ' is not supported')

    def toGPU(self):
        if self.ndevices > 1:
            self.emb_l, self.v_W_l = self.create_emb(self.m_spa, self.ln_emb, self.emb_dict, self.weighted_pooling)
        elif self.weighted_pooling == 'fixed':
            for k, w in enumerate(self.v_W_l):
                self.v_W_l[k] = w

    def sanity_check(self):
        if self.emb_dict.get('custom', None) is not None and self.emb_dict['custom']['name'] == 'qr_emb':
            if self.emb_dict['custom']['qr_operation'] == 'concat' and 2 * self.m_spa != self.ln_bot[-1]:
                sys.exit('ERROR: 2 arch-sparse-feature-size ' + str(2 * self.m_spa) + ' does not match last dim of bottom mlp ' + str(self.ln_bot[-1]) + ' (note that the last dim of bottom mlp' + 'must be 2x the embedding dim)')
            if self.qr_dict['qr_operation'] != 'concat' and self.m_spa != self.ln_bot[-1]:
                sys.exit('ERROR: arch-sparse-feature-size ' + str(self.m_spa) + ' does not match last dim of bottom mlp ' + str(self.ln_bot[-1]))
        elif self.m_spa != self.ln_bot[-1]:
            sys.exit('ERROR: arch-sparse-feature-size ' + str(self.m_spa) + ' does not match last dim of bottom mlp ' + str(self.ln_bot[-1]))

    def apply_mlp(self, x, layers):
        return layers(x)

    def apply_emb(self, lS_o, lS_i, emb_l, v_W_l):
        ly = [None] * len(lS_i)
        merged_embeddings = zip(emb_l, lS_i, lS_o, v_W_l)
        for i, (emb, lsi, lso, vwl) in enumerate(merged_embeddings):
            per_sample_weights = vwl.gather(0, lsi) if vwl is not None else None
            ly[i] = emb(lsi, lso.long(), per_sample_weights=per_sample_weights)
        return ly

    def interact_features(self, x, ly):
        if self.arch_interaction_op == 'dot':
            batch_size, d = x.shape
            T = torch.cat([x] + ly, dim=1).view((batch_size, -1, d))
            Z = torch.bmm(T, torch.transpose(T, 1, 2))
            Zflat = Z[:, self.index_tensor_i, self.index_tensor_j]
            R = torch.cat([x] + [Zflat], dim=1)
        elif self.arch_interaction_op == 'cat':
            R = torch.cat([x] + ly, dim=1)
        else:
            sys.exit('ERROR: --arch-interaction-op=' + self.arch_interaction_op + ' is not supported')
        return R

    def forward(self, dense_x, lS_o, lS_i):
        x = self.apply_mlp(dense_x, self.bot_l)
        ly = self.apply_emb(lS_o, lS_i, self.emb_l, self.v_W_l)
        z = self.interact_features(x, ly)
        out = self.apply_mlp(z, self.top_l)
        if 0.0 < self.loss_threshold and self.loss_threshold < 1.0:
            out = torch.clamp(out, min=self.loss_threshold, max=1.0 - self.loss_threshold)
        return out

    def get_scores(self, logits):
        return sigmoid(logits)

    def loss(self, logits, true_label):
        if self.loss_function == 'mse':
            return self.loss_fn(self.get_scores(logits), true_label)
        elif self.loss_function == 'bce':
            return self.loss_fn(logits, true_label)


class RegressionPreprocessor(EnvisPreProcessor):

    def __init__(self, dataset_config, client_id=0):
        super().__init__(dataset_config, client_id)


class Regression_Net(nn.Module):
    Preproc = RegressionPreprocessor

    def __init__(self, preprocessor: RegressionPreprocessor, input_dim=784, output_dim=10, loss_weights=None, loss_threshold=0.0, ndevices=-1, loss_function='mse'):
        super().__init__()
        self.preproc = preprocessor
        self.ndevices = ndevices
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.parallel_model_batch_size = -1
        self.parallel_model_is_not_prepared = True
        self.loss_threshold = loss_threshold
        self.loss_function = loss_function
        self.linear = torch.nn.Linear(self.input_dim, self.output_dim, True)
        if self.loss_function == 'mse':
            self.loss_fn = torch.nn.MSELoss(reduction='mean')
        elif self.loss_function == 'ce':
            self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        else:
            sys.exit('ERROR: --loss_function=' + self.loss_function + ' is not supported')

    def forward(self, x):
        out = self.linear(x.reshape(-1, self.input_dim))
        if 0.0 < self.loss_threshold and self.loss_threshold < 1.0:
            out = torch.clamp(out, min=self.loss_threshold, max=1.0 - self.loss_threshold)
        return out

    def get_scores(self, logits):
        return sigmoid(logits)

    def loss(self, logits, true_label):
        if self.loss_function == 'mse':
            return self.loss_fn(self.get_scores(logits), true_label)
        elif self.loss_function == 'ce':
            return self.loss_fn(logits, true_label)


class EmbeddingBag(nn.EmbeddingBag):
    """
    The embedding bag class sums the "Bags" of embeddings without
    noticing the intermediate embeddings.EmbeddedBag is a time and
    cost efficient process.

    Due to the fact that embedding_bag isn't required to return
    an intermediate result, it does not generate a Tensor object.
    It proceeds straight to computing the reduction, pulling the
    appropriate data from the weight argument in accordance with
    the indices in the input argument. This resulted in better
    performance since there was no need to create the embedding Tensor.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, max_norm: Optional[float]=None, norm_type: float=2.0, scale_grad_by_freq: bool=False, mode: str='mean', sparse: bool=False, _weight: Optional[Tensor]=None, include_last_offset: bool=False, init=False) ->None:
        super().__init__(num_embeddings, embedding_dim, max_norm=max_norm, norm_type=norm_type, scale_grad_by_freq=scale_grad_by_freq, mode=mode, sparse=sparse, _weight=_weight, include_last_offset=include_last_offset)
        if init:
            with torch.no_grad():
                W = np.random.uniform(low=-np.sqrt(1 / num_embeddings), high=np.sqrt(1 / num_embeddings), size=(num_embeddings, embedding_dim)).astype(np.float32)
                self.weight = Parameter(torch.tensor(W, requires_grad=True))


class PrEmbeddingBag(nn.Module):
    """
    PrEmbeddingBag class assists in initializing and
    assigning the values to the parameters such as weights,
    num_embeddings, embedding_dim, base_dim, index for summation.

    Parameters
    ----------
    num_embeddings : int
        size of the dictionary of embeddings.
    embedding_dim : int
        the size of each embedding vector.
    base_dim :
        the base dimension of embedding
    index : int
         the index of embedding
    """

    def __init__(self, num_embeddings, embedding_dim, base_dim=None, index=-1, init=False):
        super(PrEmbeddingBag, self).__init__()
        if base_dim is None:
            assert index >= 0, 'PR emb either specify'
            +' base dimension or extraction index'
            base_dim = max(embedding_dim)
            embedding_dim = embedding_dim[index]
        self.embs = nn.EmbeddingBag(num_embeddings, embedding_dim, mode='sum', sparse=True)
        torch.nn.init.xavier_uniform_(self.embs.weight)
        if embedding_dim < base_dim:
            self.proj = nn.Linear(embedding_dim, base_dim, bias=False)
            torch.nn.init.xavier_uniform_(self.proj.weight)
        elif embedding_dim == base_dim:
            self.proj = nn.Identity()
        else:
            raise ValueError('Embedding dim ' + str(embedding_dim) + ' > base dim ' + str(base_dim))
        if init:
            with torch.no_grad():
                W = np.random.uniform(low=-np.sqrt(1 / num_embeddings), high=np.sqrt(1 / num_embeddings), size=(num_embeddings, embedding_dim)).astype(np.float32)
                self.embs.weight = Parameter(torch.tensor(W, requires_grad=True))

    def forward(self, input, offsets=None, per_sample_weights=None):
        return self.proj(self.embs(input, offsets=offsets, per_sample_weights=per_sample_weights))


class QREmbeddingBag(nn.Module):
    """Computes sums or means over two 'bags' of embeddings, one
    using the quotient of the indices and the other using the remainder
    of the indices, without instantiating the intermediate embeddings,
    then performsan operation to combine these.

    For bags of constant length and no :attr:`per_sample_weights`, this class

        * with ``mode="sum"`` is equivalent to :class:`~torch.nn.Embedding` followed by ``sum(dim=1)``,
        * with ``mode="mean"`` is equivalent to :class:`~torch.nn.Embedding` followed by ``torch.mean(dim=1)``,
        * with ``mode="max"`` is equivalent to :class:`~torch.nn.Embedding` followed by ``torch.max(dim=1)``.

    However, :class:`~torch.nn.EmbeddingBag` is much more time and memory
    efficient than using a chain of these operations.

    QREmbeddingBag also supports per-sample weights as an argument
    to the forward pass. This scales the output of the Embedding
    before performing a weighted reduction as specified by ``mode``.

    If :attr:`per_sample_weights`` is passed, the only supported ``mode`` is
    ``"sum"``, which computes a weighted sum according to :attr:
    `per_sample_weights`.

    Known Issues:
    Autograd breaks with multiple GPUs. It breaks only with
    multiple embeddings.

    Args:
        num_categories (int): total number of unique categories.
            The input indices must be in 0, 1, ..., num_categories - 1.
        embedding_dim (list): list of sizes for each embedding vector in each table.
            If ``"add"`` or ``"mult"`` operation are used, these embedding
            dimensions must be the same.
            If a single embedding_dim is used, then it will use this
            embedding_dim for both embedding tables.
        num_collisions (int): number of collisions to enforce.
        operation (string, optional):
            ``"concat"``, ``"add"``, or ``"mult". Specifies the operation
            to compose embeddings. ``"concat"`` concatenates the embeddings,
            ``"add"`` sums the embeddings, and ``"mult"`` multiplies
            (component-wise) the embeddings.
            Default: ``"mult"``
        max_norm (float, optional):
            If given, each embedding vector with norm larger than
            :attr:`max_norm` is renormalized to have norm
            :attr:`max_norm`.
        norm_type (float, optional):
            The p of the p-norm to compute for the :attr:`max_norm` option.
            Default ``2``.
        scale_grad_by_freq (boolean, optional):
            if given, this will scale gradients by the inverse
            of frequency of the words in the mini-batch.
            Default ``False``.

            .. note::
                This option is not supported when ``mode="max"``.

        mode (string, optional):
            ``"sum"``, ``"mean"`` or ``"max"``. Specifies the way to reduce the
            bag.

            * ``"sum"`` computes the weighted sum, taking `per_sample_weights` into consideration.
            * ``"mean"`` computes the average of the values in the bag,
            * ``"max"`` computes the max value over each bag.

            Default: ``"mean"``

        sparse (bool, optional):
            if ``True``, gradient w.r.t. :attr:`weight` matrix
            will be a sparse tensor.
            See Notes for more details regarding sparse gradients.

            .. note::
                This option is not supported when ``mode="max"``.

    Attributes
        weight (Tensor):
            the learnable weights of each embedding table
            is the module of shape `(num_embeddings, embedding_dim)`
            initialized using a uniform distribution
            with sqrt(1 / num_categories).

    Inputs:
        :attr:`input` (LongTensor), :attr:`offsets` (LongTensor, optional), and
            :attr:`per_index_weights` (Tensor, optional)
            If :attr:`input` is 2D of shape `(B, N)`,
            it will be treated as ``B`` bags (sequences) each of
            fixed length ``N``, and this will return ``B`` values
            aggregated in a way depending on the :attr:`mode`.
            :attr:`offsets` is ignored and required to be ``None``
            in this case.
            If :attr:`input` is 1D of shape `(N)`,
            it will be treated as a concatenation of multiple bags (sequences).
            :attr:`offsets` is required to be a 1D tensor containing the
            starting index positions of each bag in :attr:`input`. Therefore,
            for :attr:`offsets` of shape `(B)`, :attr:`input` will be viewed as
            having ``B`` bags. Empty bags (i.e., having 0-length) will have
            returned vectors filled by zeros.
        per_sample_weights (Tensor, optional):
            a tensor of float / double weights, or None
            to indicate all weights should be taken to be ``1``.
            If specified, :attr:`per_sample_weights` must have exactly the
            same shape as input and is treated as having the same
            :attr:`offsets`, if those are not ``None``.
            Only supported for ``mode='sum'``.

    Returns
        The output tensor of shape `(B, embedding_dim)`

    """
    __constants__ = ['num_embeddings', 'embedding_dim', 'num_collisions', 'operation', 'max_norm', 'norm_type', 'scale_grad_by_freq', 'mode', 'sparse']

    def __init__(self, num_embeddings, embedding_dim, num_collisions, operation='mult', max_norm=None, norm_type=2.0, scale_grad_by_freq=False, mode='mean', sparse=False, _weight=None):
        super(QREmbeddingBag, self).__init__()
        assert operation in ['concat', 'mult', 'add'], 'Not valid operation!'
        self.num_categories = num_embeddings
        if isinstance(embedding_dim, int) or len(embedding_dim) == 1:
            self.embedding_dim = [embedding_dim, embedding_dim]
        else:
            self.embedding_dim = embedding_dim
        self.num_collisions = num_collisions
        self.operation = operation
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        if self.operation == 'add' or self.operation == 'mult':
            assert self.embedding_dim[0] == self.embedding_dim[1], 'Embedding dimensions do not match!'
        self.num_embeddings = [int(np.ceil(num_embeddings / num_collisions)), num_collisions]
        if _weight is None:
            self.weight_q = Parameter(torch.Tensor(self.num_embeddings[0], self.embedding_dim[0]))
            self.weight_r = Parameter(torch.Tensor(self.num_embeddings[1], self.embedding_dim[1]))
            self.reset_parameters()
        else:
            assert list(_weight[0].shape) == [self.num_embeddings[0], self.embedding_dim[0]], 'Shape of weight for quotient table does not' + 'match num_embeddings and embedding_dim'
            assert list(_weight[1].shape) == [self.num_embeddings[1], self.embedding_dim[1]], 'Shape of weight for remainder table does not' + 'match num_embeddings and embedding_dim'
            self.weight_q = Parameter(_weight[0])
            self.weight_r = Parameter(_weight[1])
        self.mode = mode
        self.sparse = sparse

    def reset_parameters(self):
        nn.init.uniform_(self.weight_q, np.sqrt(1 / self.num_categories))
        nn.init.uniform_(self.weight_r, np.sqrt(1 / self.num_categories))

    def forward(self, input, offsets=None, per_sample_weights=None):
        """
        Defines the forward computation performed by EmbeddingBag
        at every call.Should be overridden by all subclasses.

        Arguments
        ---------
        input: Tensor
           Tensor containing bags of indices into the embedding matrix.
        offsets: Tensor
           offsets determines the starting index position of
           each bag (sequence) in input.
        per_sample_weights: Tensor
           a tensor of float/double weights, or None to
           indicate all weights should be taken to be 1.If
           specified per_sample_weights must have exactly
           the same shape as input and is treated as having
           the same offsets,if those are not None. Only
           supported for mode='sum'.

        Returns
        -------
        (int)The output tensor of shape (B, embedding_dim)
        """
        input_q = (input / self.num_collisions).long()
        input_r = torch.remainder(input, self.num_collisions).long()
        embed_q = F.embedding_bag(input_q, self.weight_q, offsets, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.mode, self.sparse, per_sample_weights)
        embed_r = F.embedding_bag(input_r, self.weight_r, offsets, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.mode, self.sparse, per_sample_weights)
        if self.operation == 'concat':
            embed = torch.cat((embed_q, embed_r), dim=1)
        elif self.operation == 'add':
            embed = embed_q + embed_r
        elif self.operation == 'mult':
            embed = embed_q * embed_r
        return embed

    def extra_repr(self):
        """"
        In this model its a necessity to set the
        extra representation of the module to
        print customized extra information,and one
        should re-implement this method in their own
        modules.Both single-line and multi-line
        strings are acceptable.
        """
        s = '{num_embeddings}, {embedding_dim}'
        if self.max_norm is not None:
            s += ', max_norm={max_norm}'
        if self.norm_type != 2:
            s += ', norm_type={norm_type}'
        if self.scale_grad_by_freq is not False:
            s += ', scale_grad_by_freq={scale_grad_by_freq}'
        s += ', mode={mode}'
        return s.format(**self.__dict__)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (QREmbeddingBag,
     lambda: ([], {'num_embeddings': 4, 'embedding_dim': 4, 'num_collisions': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (Regression_Net,
     lambda: ([], {'preprocessor': 4}),
     lambda: ([torch.rand([4, 784])], {}),
     True),
]

class Test_NimbleEdge_EnvisEdge(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

