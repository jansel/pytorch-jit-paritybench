import sys
_module = sys.modules[__name__]
del sys
data_preprocessing = _module
tuning = _module
ptls = _module
custom_layers = _module
data_load = _module
augmentations = _module
all_time_shuffle = _module
build_augmentations = _module
drop_day = _module
dropout_trx = _module
random_slice = _module
seq_len_limit = _module
sequence_pair_augmentation = _module
data_module = _module
cls_data_module = _module
coles_data_module = _module
cpc_data_module = _module
cpc_v2_data_module = _module
emb_data_module = _module
emb_valid_data_module = _module
map_augmentation_dataset = _module
nsp_data_module = _module
rtd_data_module = _module
seq_to_target_data_module = _module
sop_data_module = _module
datasets = _module
augmentation_dataset = _module
dataloaders = _module
memory_dataset = _module
parquet_dataset = _module
parquet_file_scan = _module
persist_dataset = _module
fast_tensor_data_loader = _module
feature_dict = _module
filter_dataset = _module
iterable_processing = _module
category_size_clip = _module
feature_bin_scaler = _module
feature_filter = _module
feature_rename = _module
feature_type_cast = _module
filter_non_array = _module
id_filter = _module
id_filter_df = _module
iterable_seq_len_limit = _module
iterable_shuffle = _module
seq_len_filter = _module
take_first_trx = _module
target_empty_filter = _module
target_extractor = _module
target_join = _module
target_move = _module
to_torch_tensor = _module
iterable_processing_dataset = _module
list_splitter = _module
padded_batch = _module
partitioned_dataset = _module
utils = _module
frames = _module
abs_module = _module
bert = _module
mlm_dataset = _module
mlm_indexed_dataset = _module
nsp_dataset = _module
rtd_dataset = _module
sop_dataset = _module
losses = _module
query_soft_max = _module
modules = _module
mlm_module = _module
mlm_nsp_module = _module
rtd_module = _module
sop_nsp_module = _module
coles = _module
coles_dataset = _module
coles_module = _module
coles_supervised_dataset = _module
coles_supervised_module = _module
barlow_twins_loss = _module
binomial_deviance_loss = _module
centroid_loss = _module
complex_loss = _module
contrastive_loss = _module
histogram_loss = _module
margin_loss = _module
softmax_loss = _module
softmax_pairwise_loss = _module
triplet_loss = _module
vicreg_loss = _module
metric = _module
sampling_strategies = _module
all_positive_pair_selector = _module
all_triplets_selector = _module
distance_weighted_pair_selector = _module
hard_negative_pair_selector = _module
hard_triplet_selector = _module
matrix_masker = _module
pair_selector = _module
pairwise_matrix_selector = _module
random_negative_triplet_selector = _module
semi_hard_triplet_selector = _module
triplet_selector = _module
split_strategy = _module
cpc = _module
cpc_dataset = _module
cpc_v2_dataset = _module
cpc_loss = _module
metrics = _module
cpc_accuracy = _module
cpc_module = _module
cpc_v2_module = _module
inference_module = _module
ptls_data_module = _module
supervised = _module
metrics = _module
seq_to_target = _module
seq_to_target_dataset = _module
tabformer = _module
tabformer_dataset = _module
tabformer_module = _module
loss = _module
make_datasets = _module
make_datasets_spark = _module
metric_learn = _module
dataset = _module
complex_target_dataset = _module
splitting_dataset = _module
target_enumerator_dataset = _module
ml_models = _module
read_processing = _module
models = _module
nn = _module
binarization = _module
head = _module
normalization = _module
pb = _module
pb_feature_extract = _module
seq_encoder = _module
abs_seq_encoder = _module
agg_feature_seq_encoder = _module
containers = _module
longformer_encoder = _module
rnn_encoder = _module
rnn_seq_encoder_distribution_target = _module
statistics_encoder = _module
transformer_encoder = _module
utils = _module
seq_step = _module
trx_encoder = _module
batch_norm = _module
float_positional_encoding = _module
noisy_embedding = _module
scalers = _module
tabformer_feature_encoder = _module
trx_encoder = _module
trx_encoder_base = _module
trx_encoder_ohe = _module
trx_mean_encoder = _module
pl_fit_target = _module
pl_inference = _module
pl_inference_spark = _module
pl_train_module = _module
preprocessing = _module
base = _module
col_category_transformer = _module
col_transformer = _module
data_preprocessor = _module
pandas = _module
category_identity_encoder = _module
col_identity_transformer = _module
event_time = _module
frequency_encoder = _module
user_group_transformer = _module
pandas_preprocessor = _module
pyspark = _module
pyspark_preprocessor = _module
util = _module
size_reduction = _module
swa = _module
tb_interface = _module
util = _module
ptls_tests = _module
metric_learning_tests = _module
test_collate = _module
test_custom_layers = _module
test_data_load = _module
test__init__ = _module
test_augmentations = _module
test_all_time_shuffle = _module
test_dropout_trx = _module
test_random_slice = _module
test_seq_len_limit = _module
test_parquet_file_scan = _module
test_feature_dict = _module
test_iterable_processing = _module
test_category_size_clip = _module
test_feature_rename = _module
test_id_filter = _module
test_id_filter_df = _module
test_iterable_shuffle = _module
test_seq_len_filter = _module
test_target_empty_filter = _module
test_target_extractor = _module
test_target_join = _module
test_target_move = _module
test_to_torch = _module
test_list_splitter = _module
test_padded_batch = _module
test_utils = _module
test_distribution_target_loss = _module
test_frames = _module
test_bert = _module
test_datasets = _module
test_mlm_dataset = _module
test_mlm = _module
test_coles = _module
test_coles_module = _module
test_losses = _module
test_metrics = _module
test_sampling_strategies = _module
test_matrix_selectors = _module
test_pair_selectors = _module
test_triplet_selectors = _module
test_common_usage = _module
test_cpc = _module
test_inference_module = _module
test_supervised = _module
test_metrics = _module
test_seq_to_target = _module
test_seq_to_target_dataset = _module
test_list_subset = _module
test_loss = _module
test_nn = _module
test_head = _module
test_normalization = _module
test_pb = _module
test_seq_encoder = _module
test_agg_feature_seq_encoder = _module
test_containers = _module
test_longformer_encoder = _module
test_rnn_encoder = _module
test_rnn_seq_distribution_targets_encoder = _module
test_statistics_encoder = _module
test_transformer_encoder = _module
test_utils = _module
test_seq_step = _module
test_trx_encoder = _module
test_batch_norm = _module
test_noisy_embedding = _module
test_trx_encoder = _module
test_trx_encoder_base = _module
test_trx_mean_encoder = _module
test_trx_ohe_encoder = _module
test_pl_api = _module
test_pandas = _module
test_category_identity_encoder = _module
test_col_identity_transformer = _module
test_event_time = _module
test_frequency_encoder = _module
test_user_group_transformer = _module
test_pandas_data_preprocessor = _module
test_pyspark = _module
test_pyspark_data_preprocessor = _module
test_ranking_loss = _module
test_size_reduction = _module
data_generation = _module
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


import logging


from functools import partial


import numpy as np


import scipy.stats


import torch


from sklearn.model_selection import train_test_split


import torch.nn as nn


import torch.nn.functional as F


import random


import warnings


from collections import defaultdict


from torch.utils.data import WeightedRandomSampler


from torch.utils.data import Sampler


from torch.utils.data import Dataset


from torch.utils.data.dataloader import DataLoader


import pandas as pd


from torch.utils.data import DataLoader


import torch.multiprocessing


from typing import List


from typing import Dict


from itertools import chain


from typing import Union


from torch.utils.data.dataset import IterableDataset


from collections import Counter


from functools import reduce


from copy import deepcopy


from torch.nn import BCELoss


from torch import nn as nn


from torch.nn import functional as F


from numpy.testing import assert_almost_equal


from itertools import combinations


from torch.special import entr


from torch import nn


from typing import Tuple


import functools


from torch.utils.data import IterableDataset


from torch.autograd import Function


from torch.nn import Linear


from torch.nn import BatchNorm1d


from torch.nn import Sigmoid


from torch.nn import Sequential


from torch.nn import ReLU


from torch.nn import LogSoftmax


from torch.nn import Flatten


from torch.nn import Softplus


from torch.nn import Dropout


from functools import WRAPPER_ASSIGNMENTS


from collections import OrderedDict


import math


from torch.nn import functional as tf


import torch as torch


from torch.optim import Optimizer


import itertools


from sklearn.metrics import cohen_kappa_score


import torch.optim


from sklearn.metrics import roc_auc_score


class DropoutEncoder(nn.Module):

    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.training:
            mask = torch.FloatTensor(x.shape[1]).uniform_(0, 1) <= self.p
            x = x.masked_fill(mask, 0)
        return x


class Squeeze(nn.Module):
    """Use torch.nn.Flatten(start_dim=) instead

    """

    def forward(self, x: torch.Tensor):
        return x.squeeze()


class CatLayer(nn.Module):

    def __init__(self, left_tail, right_tail):
        super().__init__()
        self.left_tail = left_tail
        self.right_tail = right_tail

    def forward(self, x):
        l, r = x
        t = torch.cat([self.left_tail(l), self.right_tail(r)], axis=1)
        return t


class MLP(nn.Module):

    def __init__(self, input_size, params):
        super().__init__()
        self.input_size = input_size
        self.use_batch_norm = params.get('use_batch_norm', True)
        layers = []
        if self.use_batch_norm:
            layers.append(nn.BatchNorm1d(input_size))
        layers_size = [input_size] + list(params.hidden_layers_size)
        for size_in, size_out in zip(layers_size[:-1], layers_size[1:]):
            layers.append(nn.Linear(size_in, size_out))
            layers.append(nn.ReLU())
            if params.drop_p:
                layers.append(nn.Dropout(params.drop_p))
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(size_out))
            self.output_size = layers_size[-1]
        if params.get('objective', None) == 'classification':
            head_output_size = params.get('num_classes', 1)
            if head_output_size == 1:
                h = nn.Sequential(nn.Linear(layers_size[-1], head_output_size), nn.Sigmoid(), Squeeze())
            else:
                h = nn.Sequential(nn.Linear(layers_size[-1], head_output_size), nn.LogSoftmax(dim=1))
            layers.append(h)
            self.output_size = head_output_size
        elif params.get('objective', None) == 'multilabel_classification':
            head_output_size = params.num_classes
            h = nn.Sequential(nn.Linear(layers_size[-1], head_output_size), nn.Sigmoid())
            layers.append(h)
            self.output_size = head_output_size
        elif params.get('objective', None) == 'regression':
            h = nn.Sequential(nn.Linear(layers_size[-1], 1), Squeeze())
            layers.append(h)
            self.output_size = 1
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class TabularRowEncoder(torch.nn.Module):

    def __init__(self, input_dim, cat_dims, cat_idxs, cat_emb_dim):
        """ This is an embedding module for an entier set of features

        Parameters
        ----------
        input_dim : int
            Number of features coming as input (number of columns)
        cat_dims : list of int
            Number of modalities for each categorial features
            If the list is empty, no embeddings will be done
        cat_idxs : list of int
            Positional index for each categorical features in inputs
        cat_emb_dim : int or list of int
            Embedding dimension for each categorical features
            If int, the same embdeding dimension will be used for all categorical features
        """
        super().__init__()
        if cat_dims == [] or cat_idxs == []:
            self.skip_embedding = True
            self.post_embed_dim = input_dim
            return
        self.skip_embedding = False
        self.cat_idxs = cat_idxs
        if isinstance(cat_emb_dim, int):
            self.cat_emb_dims = [cat_emb_dim] * len(cat_idxs)
        else:
            self.cat_emb_dims = cat_emb_dim
        if len(self.cat_emb_dims) != len(cat_dims):
            msg = """ cat_emb_dim and cat_dims must be lists of same length, got {len(self.cat_emb_dims)}
                      and {len(cat_dims)}"""
            raise ValueError(msg)
        self.post_embed_dim = int(input_dim + np.sum(self.cat_emb_dims) - len(self.cat_emb_dims))
        self.embeddings = torch.nn.ModuleList()
        sorted_idxs = np.argsort(cat_idxs)
        cat_dims = [cat_dims[i] for i in sorted_idxs]
        self.cat_emb_dims = [self.cat_emb_dims[i] for i in sorted_idxs]
        for cat_dim, emb_dim in zip(cat_dims, self.cat_emb_dims):
            self.embeddings.append(torch.nn.Embedding(cat_dim, emb_dim))

    def forward(self, x):
        """
        Apply embdeddings to inputs
        Inputs should be (batch_size, input_dim)
        Outputs will be of size (batch_size, self.post_embed_dim)
        """
        if self.skip_embedding:
            return x
        cols = []
        prev_cat_idx = -1
        for cat_feat_counter, cat_idx in enumerate(self.cat_idxs):
            if cat_idx > prev_cat_idx + 1:
                cols.append(x[:, prev_cat_idx + 1:cat_idx].float())
            cols.append(self.embeddings[cat_feat_counter](x[:, cat_idx].long()))
            prev_cat_idx = cat_idx
        if prev_cat_idx + 1 < x.shape[1]:
            cols.append(x[:, prev_cat_idx + 1:].float())
        post_embeddings = torch.cat(cols, dim=1)
        return post_embeddings

    @property
    def output_size(self):
        return self.post_embed_dim


class DistributionTargetHead(torch.nn.Module):

    def __init__(self, in_size=256, num_distr_classes_pos=4, num_distr_classes_neg=14, pos=True, neg=True):
        super().__init__()
        self.pos, self.neg = pos, neg
        self.dense1 = torch.nn.Linear(in_size, 128)
        self.dense2_neg = torch.nn.Linear(128, num_distr_classes_neg) if self.neg else None
        self.dense2_pos = torch.nn.Linear(128, num_distr_classes_pos) if self.pos else None
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        out1 = self.relu(self.dense1(x))
        out2_pos = out2_neg = 0
        if self.pos:
            out2_pos = self.dense2_pos(out1)
        if self.neg:
            out2_neg = self.dense2_neg(out1)
        return out2_neg, out2_pos


class RegressionTargetHead(torch.nn.Module):

    def __init__(self, in_size=256, gates=True, pos=True, neg=True, pass_samples=True):
        super().__init__()
        self.pos, self.neg = pos, neg
        self.pass_samples = pass_samples
        self.dense1 = torch.nn.Linear(in_size, 64)
        if pass_samples:
            self.dense2_neg = torch.nn.Linear(64, 15) if self.neg else None
            self.dense2_pos = torch.nn.Linear(64, 15) if self.pos else None
        else:
            self.dense2_neg = torch.nn.Linear(64, 16) if self.neg else None
            self.dense2_pos = torch.nn.Linear(64, 16) if self.pos else None
        self.dense3_neg = torch.nn.Linear(16, 1) if self.neg else None
        self.dense3_pos = torch.nn.Linear(16, 1) if self.pos else None
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()
        self.gates = gates

    def forward(self, x, neg_sum_logs=None, pos_sum_logs=None):
        out1 = self.relu(self.dense1(x))
        if self.gates:
            out1 = out1.detach()
        out3_neg = out3_pos = 0
        if self.pos:
            out2_pos = self.relu(self.dense2_pos(out1))
            if self.pass_samples:
                out2_pos = torch.cat((out2_pos, pos_sum_logs), dim=1).float()
            out3_pos = self.dense3_pos(out2_pos)
            out3_pos = self.sigmoid(out3_pos) if self.gates else out3_pos
        if self.neg:
            out2_neg = self.relu(self.dense2_neg(out1))
            if self.pass_samples:
                out2_neg = torch.cat((out2_neg, neg_sum_logs), dim=1).float()
            out3_neg = self.dense3_neg(out2_neg)
            out3_neg = self.sigmoid(out3_neg) if self.gates else out3_neg
        return out3_neg, out3_pos


class CombinedTargetHeadFromRnn(torch.nn.Module):

    def __init__(self, in_size=48, num_distr_classes_pos=4, num_distr_classes_neg=14, pos=True, neg=True, use_gates=True, pass_samples=True):
        super().__init__()
        self.pos, self.neg = pos, neg
        self.use_gates = use_gates
        self.dense = torch.nn.Linear(in_size, 256)
        self.distribution = DistributionTargetHead(256, num_distr_classes_pos, num_distr_classes_neg, pos, neg)
        self.regr_sums = RegressionTargetHead(256, False, pos, neg, pass_samples)
        self.regr_gates = RegressionTargetHead(256, True, pos, neg, pass_samples) if self.use_gates else None
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        device = x[0].device if isinstance(x, tuple) else x.device
        sum_logs2 = torch.tensor(x[2][:, None], device=device) if isinstance(x, tuple) and len(x) > 2 else 0
        x, sum_logs1 = (x[0], torch.tensor(x[1][:, None], device=device)) if isinstance(x, tuple) else (x, 0)
        if not self.neg and self.pos:
            sum_logs1, sum_logs2 = sum_logs2, sum_logs1
        out1 = self.relu(self.dense(x))
        distr_neg, distr_pos = self.distribution(out1)
        sums_neg, sums_pos = self.regr_sums(out1, sum_logs1, sum_logs2)
        gate_neg, gate_pos = self.regr_gates(out1, sum_logs1, sum_logs2) if self.use_gates else (0, 0)
        return {'neg_sum': sum_logs1 * gate_neg + sums_neg * (1 - gate_neg), 'neg_distribution': distr_neg, 'pos_sum': sum_logs2 * gate_pos + sums_pos * (1 - gate_pos), 'pos_distribution': distr_pos}


class TargetHeadFromAggFeatures(torch.nn.Module):

    def __init__(self, in_size=48, num_distr_classes=6):
        super().__init__()
        self.dense1 = torch.nn.Linear(in_size, 512)
        self.distribution = DistributionTargetHead(512, num_distr_classes)
        self.dense2_sums = torch.nn.Linear(512, 64)
        self.dense3_sums_neg = torch.nn.Linear(64, 1)
        self.dense3_sums_pos = torch.nn.Linear(64, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        out1 = self.relu(self.dense1(x))
        distr_neg, distr_pos = self.distribution(out1)
        out2_sums = self.relu(self.dense2_sums(out1))
        out3_sums_neg = self.dense3_sums_neg(out2_sums)
        out3_sums_pos = self.dense3_sums_pos(out2_sums)
        return {'neg_sum': out3_sums_neg, 'neg_distribution': distr_neg, 'pos_sum': out3_sums_pos, 'pos_distribution': distr_pos}


class DummyHead(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return {'neg_sum': x[0], 'neg_distribution': x[1], 'pos_sum': x[2], 'pos_distribution': x[3]}


class GEGLU(torch.nn.Module):
    """
    References:
        Shazeer et al., "GLU Variants Improve Transformer," 2020.
        https://arxiv.org/abs/2002.05202

    Parameters
    ----------
    inputs
        Inputs to process [B, 2H]
    
    Returns
    -------
    outputs
        The outputs following the GEGLU activation [B, H]

    """

    def __init__(self):
        super().__init__()

    def geglu(self, x: torch.Tensor) ->torch.Tensor:
        assert x.shape[-1] % 2 == 0
        a, b = x.chunk(2, dim=-1)
        return a * F.gelu(b)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        return self.geglu(x)


class FeatureDict:
    """Tools for feature-dict format
    Feature dict:
        keys are feature names
        values are feature values, sequential, scalar or arrays
    """

    def __init__(self, *args, **kwargs):
        """Mixin constructor
        """
        super().__init__(*args, **kwargs)

    @staticmethod
    def is_seq_feature(k: str, x):
        """Check is value sequential feature
        Synchronized with ptls.data_load.padded_batch.PaddedBatch.is_seq_feature

        Iterables are:
            np.array
            torch.Tensor

        Not iterable:
            list    - dont supports indexing

        Parameters
        ----------
        k:
            feature_name
        x:
            value for check

        Returns
        -------
            True if value is iterable
        """
        if k == 'event_time':
            return True
        if k.startswith('target'):
            return False
        if type(x) in (np.ndarray, torch.Tensor):
            return True
        return False

    @staticmethod
    def seq_indexing(d, ix):
        """Apply indexing for seq_features only

        Parameters
        ----------
        d:
            feature dict
        ix:
            indexes

        Returns
        -------

        """
        return {k: (v[ix] if FeatureDict.is_seq_feature(k, v) else v) for k, v in d.items()}

    @staticmethod
    def get_seq_len(d):
        """Finds a sequence column and return its length

        Parameters
        ----------
        d:
            feature-dict

        """
        if 'event_time' in d:
            return len(d['event_time'])
        return len(next(v for k, v in d.items() if FeatureDict.is_seq_feature(k, v)))


class PaddedBatch:
    """Contains a padded batch of sequences with different lengths.

    Parameters:
        payload:
            container with data. Format supported:
            - dict with features. This is the input data for overall network pipeline.
                Kees are the feature names, values are (B, T) shape tensors.
                Long type for categorical features, embedding lookup table indexes expected
                Float type for numerical features.
            - trx embedding tensor. This is the intermediate data for overall network pipeline.
                shape (B, T, H)
            - feature tensor. Used in some cases
                shape (B, T)
        length:
            Tensor of shape (B,) with lengths of sequences.
            All sequences in `payload` has length T, but only L first are used.
            Unused positions padded with zeros

    Example:
        >>> data = PaddedBatch(
        >>>     payload=torch.tensor([
        >>>         [1, 2, 0, 0],
        >>>         [3, 4, 5, 6],
        >>>         [7, 8, 9, 0],
        >>>     ]),
        >>>     length=torch.tensor([2, 4, 3]),
        >>> )
        >>>
        >>> # check shape
        >>> torch.testing.assert_close(data.payload.size(), (3, 4))
        >>>
        >>> # get first transaction
        >>> torch.testing.assert_close(data.payload[:, 0], torch.tensor([1, 3, 7]))
        >>>
        >>> # get last transaction
        >>> torch.testing.assert_close(data.payload[torch.arange(3), data.seq_lens - 1], torch.tensor([2, 6, 9]))
        >>>
        >>> # get all transaction flatten
        >>> torch.testing.assert_close(data.payload[data.seq_len_mask.bool()], torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]))

    """

    def __init__(self, payload: Dict[str, torch.Tensor], length: torch.LongTensor):
        self._payload = payload
        self._length = length

    @property
    def payload(self):
        return self._payload

    @property
    def seq_lens(self):
        return self._length

    @property
    def device(self):
        return self._length.device

    @property
    def seq_feature_shape(self):
        return next(v.size() for k, v in self._payload.items() if self.is_seq_feature(k, v))

    def __len__(self):
        return len(self._length)

    def to(self, device, non_blocking=False):
        length = self._length
        payload = {k: (v if type(v) is torch.Tensor else v) for k, v in self._payload.items()}
        return PaddedBatch(payload, length)

    @property
    def seq_len_mask(self):
        """mask with B*T size for valid tokens in `payload`
        """
        if type(self._payload) is dict:
            B, T = next(v for k, v in self._payload.items() if self.is_seq_feature(k, v)).size()
        else:
            B, T = self._payload.size()[:2]
        return (torch.arange(T, device=self._length.device).unsqueeze(0).expand(B, T) < self._length.unsqueeze(1)).long()

    @staticmethod
    def is_seq_feature(k: str, x):
        """Check is value sequential feature
        Synchronized with ptls.data_load.feature_dict.FeatureDict.is_seq_feature

                     1-d        2-d
        event_time | True      True
        target_    | False     False  # from FeatureDict.is_seq_feature
        tensor     | False     True

        Parameters
        ----------
        k:
            feature_name
        x:
            value for check

        Returns
        -------

        """
        if not FeatureDict.is_seq_feature(k, x):
            return False
        if type(x) is np.ndarray:
            return False
        if len(x.shape) == 1:
            return False
        return True

    def drop_seq_features(self):
        """Returns new dict without sequential features

        Returns
        -------

        """
        return {k: v for k, v in self.payload.items() if not PaddedBatch.is_seq_feature(k, v)}

    def keep_seq_features(self):
        """Returns new PaddedBatch with sequential features only

        Returns
        -------

        """
        return PaddedBatch(payload={k: v for k, v in self.payload.items() if PaddedBatch.is_seq_feature(k, v)}, length=self.seq_lens)


class StatPooling(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pb: PaddedBatch):
        payload = pb.payload
        mask = pb.seq_len_mask.bool()
        inf = torch.zeros_like(mask, device=mask.device).float()
        inf[~mask] = -torch.inf
        pb_max = torch.max(payload + inf.unsqueeze(-1), dim=1)[0].squeeze()
        pb_min = torch.min(payload - inf.unsqueeze(-1), dim=1)[0].squeeze()
        pb_mean = payload.sum(dim=1) / mask.float().sum(dim=1, keepdim=True)
        pb_std = payload - pb_mean.unsqueeze(1).expand_as(payload)
        pb_std = pb_std * mask.unsqueeze(-1)
        pb_std = torch.sqrt(torch.mean(torch.pow(pb_std, 2), dim=1))
        out = torch.cat([pb_max, pb_min, pb_mean, pb_std], dim=1)
        return out


class QuerySoftmaxLoss(torch.nn.Module):

    def __init__(self, temperature: float=1.0, reduce: bool=True):
        """

        Parameters
        ----------
        temperature:
            softmax(logits * temperature)
        reduce:
            if `reduce` then `loss.mean()` returned. Else loss by elements returned
        """
        super().__init__()
        self.temperature = temperature
        self.reduce = reduce

    def forward(self, anchor, pos, neg):
        logits = self.get_logits(anchor, pos, neg)
        probas = torch.softmax(logits, dim=1)
        loss = -torch.log(probas[:, 0])
        if self.reduce:
            return loss.mean()
        return loss

    def get_logits(self, anchor, pos, neg):
        all_counterparty = torch.cat([pos, neg], dim=1)
        logits = (anchor * all_counterparty).sum(dim=2) * self.temperature
        return logits


class SequencePredictionHead(torch.nn.Module):

    def __init__(self, embeds_dim, hidden_size=64, drop_p=0.1):
        super().__init__()
        self.head = torch.nn.Sequential(torch.nn.Linear(embeds_dim, hidden_size, bias=True), torch.nn.BatchNorm1d(hidden_size), torch.nn.ReLU(), torch.nn.Dropout(drop_p), torch.nn.Linear(hidden_size, 1), torch.nn.Sigmoid())

    def forward(self, x):
        x = self.head(x).squeeze(-1)
        return x


class MLMNSPInferenceModule(torch.nn.Module):

    def __init__(self, pretrained_model):
        super().__init__()
        self.model = pretrained_model
        self.model._seq_encoder.is_reduce_sequence = True
        self.model._seq_encoder.add_cls_output = False
        if self.model.hparams.inference_pooling_strategy == 'stat':
            self.stat_pooler = StatPooling()

    def forward(self, batch):
        z_trx = self.model.trx_encoder(batch)
        out = self.model._seq_encoder(z_trx)
        if self.model.hparams.inference_pooling_strategy == 'stat':
            stats = self.stat_pooler(z_trx)
            out = torch.cat([stats, out], dim=1)
        if self.model.hparams.norm_predict:
            out = out / (out.pow(2).sum(dim=-1, keepdim=True) + 1e-09).pow(0.5)
        return out


class SentencePairsHead(torch.nn.Module):

    def __init__(self, base_model, embeds_dim, hidden_size, drop_p):
        super().__init__()
        self.base_model = base_model
        self.head = torch.nn.Sequential(torch.nn.Linear(embeds_dim * 2, hidden_size, bias=True), torch.nn.BatchNorm1d(hidden_size), torch.nn.ReLU(), torch.nn.Dropout(drop_p), torch.nn.Linear(hidden_size, 1), torch.nn.Sigmoid())

    def forward(self, x):
        left, right = x
        x = torch.cat([self.base_model(left), self.base_model(right)], dim=1)
        return self.head(x).squeeze(-1)


class BarlowTwinsLoss(torch.nn.Module):
    """
    From https://github.com/facebookresearch/barlowtwins

    """

    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, model_outputs, target):
        n = len(model_outputs)
        ix1 = torch.arange(0, n, 2, device=model_outputs.device)
        ix2 = torch.arange(1, n, 2, device=model_outputs.device)
        assert (target[ix1] == target[ix2]).all(), 'Wrong embedding positions'
        z1 = model_outputs[ix1]
        z2 = model_outputs[ix2]
        c = torch.mm(z1.T, z2)
        c.div_(n // 2)
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag
        return loss

    @staticmethod
    def off_diagonal(x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BinomialDevianceLoss(nn.Module):
    """
    Binomial Deviance loss

    "Deep Metric Learning for Person Re-Identification", ICPR2014
    https://arxiv.org/abs/1407.4979
    """

    def __init__(self, pair_selector, alpha=1, beta=1, C=1):
        super(BinomialDevianceLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.C = C
        self.pair_selector = pair_selector

    def forward(self, embeddings, target):
        positive_pairs, negative_pairs = self.pair_selector.get_pairs(embeddings, target)
        pos_pair_similarity = F.cosine_similarity(embeddings[positive_pairs[:, 0]], embeddings[positive_pairs[:, 1]], dim=1)
        neg_pair_similarity = F.cosine_similarity(embeddings[negative_pairs[:, 0]], embeddings[negative_pairs[:, 1]], dim=1)
        pos_loss = torch.mean(torch.log(1 + torch.exp(-self.alpha * (pos_pair_similarity - self.beta))))
        neg_loss = torch.mean(torch.log(1 + torch.exp(self.alpha * self.C * (neg_pair_similarity - self.beta))))
        res_loss = (pos_loss + neg_loss) * len(target)
        return res_loss


class CentroidLoss(nn.Module):
    """Centroid loss

    Class centroids are calculated over batch

    Parameters
    ----------
    class_num
        set the number of classes if you know limit (e.g. 2 for binary classifications)
        set None for metric learning task with unknown class number
    centroid_margin
        l2 distance between the class centers, closer than which the loss will be calculated.
        Class centers tend to be further than `centroid_margin`.
    """

    def __init__(self, class_num, centroid_margin=1.4):
        super().__init__()
        self.class_num = class_num
        if self.class_num is not None:
            self.register_buffer('l_targets_eye', torch.eye(class_num))
        self.centroid_margin = centroid_margin
        self.eps = 1e-06

    def forward(self, embeddings, target):
        if self.class_num is None:
            class_num = target.max() + 1
            l_targets_eye = torch.eye(class_num, device=target.device)
        else:
            l_targets_eye = self.l_targets_eye
        l_targets_ohe = l_targets_eye[target]
        class_centers = embeddings.unsqueeze(1) * l_targets_ohe.unsqueeze(2)
        class_centers = class_centers.sum(dim=0).div(l_targets_ohe.sum(dim=0).unsqueeze(1) + self.eps)
        cc_pairs = (class_centers.unsqueeze(1) - class_centers.unsqueeze(0)).pow(2).sum(dim=2)
        cc_pairs = torch.relu(self.centroid_margin - (cc_pairs + self.eps).pow(0.5)).pow(2)
        cc_pairs = torch.triu(cc_pairs, diagonal=1).sum() / cc_pairs.size(0) * (cc_pairs.size(0) - 1) / 2
        distances = embeddings - class_centers[target]
        l2_loss = distances.pow(2).sum(dim=1)
        return l2_loss.mean() + cc_pairs


class CentroidSoftmaxLoss(nn.Module):
    """Centroid Softmax loss

    Class centroids are calculated over batch

    Parameters
    ----------
    class_num
        set the number of classes if you know limit (e.g. 2 for binary classifications)
        set None for metric learning task with unknown class number
    temperature:
        temperature for softmax logits for scaling l2 distance on unit sphere
    """

    def __init__(self, class_num, temperature=10.0):
        super().__init__()
        self.class_num = class_num
        if self.class_num is not None:
            self.register_buffer('l_targets_eye', torch.eye(class_num))
        self.eps = 1e-06
        self.temperature = temperature

    def forward(self, embeddings, target):
        if self.class_num is None:
            class_num = target.max() + 1
            l_targets_eye = torch.eye(class_num, device=target.device)
        else:
            l_targets_eye = self.l_targets_eye
        l_targets_ohe = l_targets_eye[target]
        class_centers = embeddings.unsqueeze(1) * l_targets_ohe.unsqueeze(2)
        class_centers = class_centers.sum(dim=0).div(l_targets_ohe.sum(dim=0).unsqueeze(1) + self.eps)
        distances = embeddings.unsqueeze(1) - class_centers.unsqueeze(0)
        l2_loss = distances.pow(2).sum(dim=2)
        l2_loss = torch.softmax(-l2_loss * self.temperature, dim=1)
        l2_loss = l2_loss[torch.arange(l2_loss.size(0), device=embeddings.device), target]
        l2_loss = -torch.log(l2_loss)
        return l2_loss.mean()


class CentroidSoftmaxMemoryLoss(nn.Module):
    """Centroid Softmax Memory loss

    Class centroids are calculated over batch and saved in memory as running average

    Parameters
    ----------
    class_num
        set the number of classes if you know limit (e.g. 2 for binary classifications)
        set None for metric learning task with unknown class number
    temperature:
        temperature for softmax logits for scaling l2 distance on unit sphere
    alpha:
        rate of history keep for running average
    """

    def __init__(self, class_num, hidden_size, temperature=10.0, alpha=0.99):
        super().__init__()
        assert class_num is not None
        self.class_num = class_num
        self.register_buffer('l_targets_eye', torch.eye(class_num))
        self.eps = 1e-06
        self.temperature = temperature
        self.alpha = alpha
        self.register_buffer('class_centers', torch.zeros(class_num, hidden_size, dtype=torch.float))
        self.is_empty_class_centers = True

    def forward(self, embeddings, target):
        l_targets_eye = self.l_targets_eye
        l_targets_ohe = l_targets_eye[target]
        class_centers = embeddings.unsqueeze(1) * l_targets_ohe.unsqueeze(2)
        class_centers = class_centers.sum(dim=0).div(l_targets_ohe.sum(dim=0).unsqueeze(1) + self.eps)
        class_centers = class_centers.detach()
        if not self.is_empty_class_centers:
            self.class_centers = self.class_centers * self.alpha + class_centers * (1 - self.alpha)
        else:
            self.class_centers = class_centers
            self.is_empty_class_centers = False
        distances = embeddings.unsqueeze(1) - self.class_centers.unsqueeze(0)
        l2_loss = distances.pow(2).sum(dim=2)
        l2_loss = torch.softmax(-l2_loss * self.temperature, dim=1)
        l2_loss = l2_loss[torch.arange(l2_loss.size(0), device=embeddings.device), target]
        l2_loss = -torch.log(l2_loss)
        return l2_loss.mean()


class ComplexLoss(torch.nn.Module):
    """Works like `ptls.loss.MultiLoss`

    """

    def __init__(self, ml_loss, aug_loss, ml_loss_weight=1.0):
        super(ComplexLoss, self).__init__()
        self.aug_loss = aug_loss
        self.ml_loss = ml_loss
        self.ml_loss_weight = ml_loss_weight

    def forward(self, model_outputs, target):
        aug_output, ml_output = model_outputs
        aug_target = target[:, 0]
        ml_target = target[:, 1]
        aug = self.aug_loss(aug_output, aug_target) * (1 - self.ml_loss_weight)
        ml = self.ml_loss(ml_output, ml_target) * self.ml_loss_weight
        return aug + ml


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss

    "Signature verification using a siamese time delay neural network", NIPS 1993
    https://papers.nips.cc/paper/769-signature-verification-using-a-siamese-time-delay-neural-network.pdf
    """

    def __init__(self, margin, sampling_strategy):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.pair_selector = sampling_strategy

    def forward(self, embeddings, target):
        positive_pairs, negative_pairs = self.pair_selector.get_pairs(embeddings, target)
        positive_loss = F.pairwise_distance(embeddings[positive_pairs[:, 0]], embeddings[positive_pairs[:, 1]]).pow(2)
        negative_loss = F.relu(self.margin - F.pairwise_distance(embeddings[negative_pairs[:, 0]], embeddings[negative_pairs[:, 1]])).pow(2)
        loss = torch.cat([positive_loss, negative_loss], dim=0)
        return loss.sum()


def outer_cosine_similarity(A, B=None):
    """
        Compute cosine_similarity of Tensors
            A (size(A) = n x d, where n - rows count, d - vector size) and
            B (size(A) = m x d, where m - rows count, d - vector size)
        return matrix C (size n x m), such as C_ij = cosine_similarity(i-th row matrix A, j-th row matrix B)

        if only one Tensor was given, computer pairwise distance to itself (B = A)
    """
    if B is None:
        B = A
    max_size = 2 ** 32
    n = A.size(0)
    m = B.size(0)
    d = A.size(1)
    if n * m * d <= max_size or m == 1:
        A_norm = torch.div(A.transpose(0, 1), A.norm(dim=1)).transpose(0, 1)
        B_norm = torch.div(B.transpose(0, 1), B.norm(dim=1)).transpose(0, 1)
        return torch.mm(A_norm, B_norm.transpose(0, 1))
    else:
        batch_size = max(1, max_size // (n * d))
        batch_results = []
        for i in range((m - 1) // batch_size + 1):
            id_left = i * batch_size
            id_rigth = min((i + 1) * batch_size, m)
            batch_results.append(outer_cosine_similarity(A, B[id_left:id_rigth]))
        return torch.cat(batch_results, dim=1)


class HistogramLoss(torch.nn.Module):
    """
    HistogramLoss

    "Learning deep embeddings with histogram loss", NIPS 2016
    https://arxiv.org/abs/1611.00822
    code based on https://github.com/valerystrizh/pytorch-histogram-loss
    """

    def __init__(self, num_steps=100):
        super(HistogramLoss, self).__init__()
        self.step = 2 / (num_steps - 1)
        self.eps = 1 / num_steps
        self.t = torch.arange(-1, 1 + self.step, self.step).view(-1, 1)
        self.tsize = self.t.size()[0]
        self.device = None

    def forward(self, embeddings, classes):

        def histogram(inds, size):
            s_repeat_ = s_repeat.clone()
            indsa = (s_repeat_floor - (self.t - self.step) > -self.eps) & (s_repeat_floor - (self.t - self.step) < self.eps) & inds
            assert indsa.nonzero(as_tuple=False).size()[0] == size, 'Another number of bins should be used'
            zeros = torch.zeros((1, indsa.size()[1])).bool()
            zeros = zeros
            indsb = torch.cat((indsa, zeros))[1:, :]
            s_repeat_[~(indsb | indsa)] = 0
            s_repeat_[indsa] = (s_repeat_ - self.t + self.step)[indsa] / self.step
            s_repeat_[indsb] = (-s_repeat_ + self.t + self.step)[indsb] / self.step
            return s_repeat_.sum(1) / size
        self.device = embeddings.device
        self.t = self.t
        classes_size = classes.size()[0]
        classes_eq = (classes.repeat(classes_size, 1) == classes.view(-1, 1).repeat(1, classes_size)).data
        dists = outer_cosine_similarity(embeddings)
        assert (dists > 1 + self.eps).sum().item() + (dists < -1 - self.eps).sum().item() == 0, 'L2 normalization should be used '
        s_inds = torch.triu(torch.ones(classes_eq.size()), 1).bool()
        s_inds = s_inds
        pos_inds = classes_eq[s_inds].repeat(self.tsize, 1)
        neg_inds = ~classes_eq[s_inds].repeat(self.tsize, 1)
        pos_size = classes_eq[s_inds].sum().item()
        neg_size = (~classes_eq[s_inds]).sum().item()
        s = dists[s_inds].view(1, -1)
        s = s.clamp(-1 + 1e-06, 1 - 1e-06)
        s_repeat = s.repeat(self.tsize, 1)
        s_repeat_floor = (torch.floor((s_repeat.data + 1.0 - 1e-06) / self.step) * self.step - 1.0).float()
        histogram_pos = histogram(pos_inds, pos_size)
        assert_almost_equal(histogram_pos.sum().item(), 1, decimal=1, err_msg='Not good positive histogram', verbose=True)
        histogram_neg = histogram(neg_inds, neg_size)
        assert_almost_equal(histogram_neg.sum().item(), 1, decimal=1, err_msg='Not good negative histogram', verbose=True)
        histogram_pos_repeat = histogram_pos.view(-1, 1).repeat(1, histogram_pos.size()[0])
        histogram_pos_inds = torch.tril(torch.ones(histogram_pos_repeat.size()), -1).bool()
        histogram_pos_inds = histogram_pos_inds
        histogram_pos_repeat[histogram_pos_inds] = 0
        histogram_pos_cdf = histogram_pos_repeat.sum(0)
        loss = torch.sum(histogram_neg * histogram_pos_cdf)
        return loss


class MarginLoss(torch.nn.Module):
    """
    Margin loss

    "Sampling Matters in Deep Embedding Learning", ICCV 2017
    https://arxiv.org/abs/1706.07567

    """

    def __init__(self, pair_selector, margin=1, beta=1.2):
        super(MarginLoss, self).__init__()
        self.margin = margin
        self.beta = beta
        self.pair_selector = pair_selector

    def forward(self, embeddings, target):
        positive_pairs, negative_pairs = self.pair_selector.get_pairs(embeddings, target)
        d_ap = F.pairwise_distance(embeddings[positive_pairs[:, 0]], embeddings[positive_pairs[:, 1]])
        d_an = F.pairwise_distance(embeddings[negative_pairs[:, 0]], embeddings[negative_pairs[:, 1]])
        pos_loss = torch.clamp(d_ap - self.beta + self.margin, min=0.0)
        neg_loss = torch.clamp(self.beta - d_an + self.margin, min=0.0)
        loss = torch.cat([pos_loss, neg_loss], dim=0)
        return loss.sum()


class MatrixMasker:
    """
    Returns matrix masked with zeros for
    summing only positive pairs distances
    (e.g. (2k-1, 2k) and (2k, 2k - 1) for k in 1 to N).
    """

    def get_masked_matrix(self, matrix, classes):
        mask = (classes.unsqueeze(1) == classes.unsqueeze(0)).int()
        mask.fill_diagonal_(0)
        masked_matrix = matrix * mask
        return masked_matrix


class SoftmaxLoss(nn.Module):
    """
    Softmax loss.
    """

    def __init__(self, masker=None, eps=1e-06, temperature=0.05):
        super(SoftmaxLoss, self).__init__()
        self.masker = masker
        if masker is None:
            self.masker = MatrixMasker()
        self.eps = eps
        self.temperature = temperature

    def forward(self, embeddings, classes):
        similarities = self.get_sim_matrix(embeddings, embeddings, eps=self.eps)
        similarities /= self.temperature
        log_matrix = -1 * F.log_softmax(similarities)
        masked_matrix = self.masker.get_masked_matrix(log_matrix, classes)
        loss = masked_matrix / len(similarities)
        return loss.sum()

    @staticmethod
    def get_sim_matrix(a, b, eps):
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
        b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
        sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
        return sim_mt


class PairwiseMatrixSelector:
    """
    Returns matrix with one positive pair on first index and all negative pairs 
    for every possible pair. 
    """

    def get_pair_matrix(self, embeddings, labels):
        n = labels.size(0)
        uniq_num = len(torch.unique(labels))
        split_num = len(labels) // uniq_num
        x = labels.expand(n, n) - labels.expand(n, n).t()
        x.fill_diagonal_(1)
        indx = torch.where(x == 0)
        positive_pairs = torch.cat((indx[0].reshape(-1, 1), indx[1].reshape(-1, 1)), dim=1)
        positive_pairs = torch.unsqueeze(positive_pairs, 1)
        x.fill_diagonal_(0)
        indx = torch.where(x != 0)
        negative_pairs = torch.cat((indx[0].reshape(-1, 1), indx[1].reshape(-1, 1)), dim=1)
        negative_pairs = negative_pairs.reshape(-1, n - split_num, 2)
        negative_pairs = negative_pairs.repeat(1, split_num - 1, 1)
        negative_pairs = negative_pairs.reshape(-1, n - split_num, 2)
        pairs = torch.cat((positive_pairs, negative_pairs), dim=1)
        return embeddings[pairs]


class SoftmaxPairwiseLoss(nn.Module):
    """
    Softmax Pairwise loss.
    """

    def __init__(self, pair_selector=None, temperature=0.05, eps=1e-06):
        super(SoftmaxPairwiseLoss, self).__init__()
        self.pair_selector = pair_selector
        if pair_selector is None:
            self.pair_selector = PairwiseMatrixSelector()
        self.temperature = temperature
        self.eps = eps

    def forward(self, embeddings, classes):
        pair_matrix = self.pair_selector.get_pair_matrix(embeddings, classes)
        similarities = F.cosine_similarity(pair_matrix[:, :, 0, :], pair_matrix[:, :, 1, :], dim=-1, eps=self.eps)
        similarities /= self.temperature
        log_matrix = -1 * F.log_softmax(similarities)
        loss = log_matrix / len(similarities)
        return loss[:, :1].sum()


class TripletLoss(nn.Module):
    """
    Triplets loss

    "Deep metric learning using triplet network", SIMBAD 2015
    https://arxiv.org/abs/1412.6622
    """

    def __init__(self, margin, triplet_selector):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):
        triplets = self.triplet_selector.get_triplets(embeddings, target)
        if embeddings.is_cuda:
            triplets = triplets
        ap_distances = F.pairwise_distance(embeddings[triplets[:, 0]], embeddings[triplets[:, 1]])
        an_distances = F.pairwise_distance(embeddings[triplets[:, 0]], embeddings[triplets[:, 2]])
        losses = F.relu(ap_distances - an_distances + self.margin)
        return losses.sum()


class VicregLoss(torch.nn.Module):
    """
    From https://github.com/facebookresearch/vicreg

    """

    def __init__(self, sim_coeff, std_coeff, cov_coeff):
        super(VicregLoss, self).__init__()
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff

    def forward(self, model_outputs, target):
        n = len(model_outputs)
        m = len(model_outputs[0])
        ix1 = torch.arange(0, n, 2, device=model_outputs.device)
        ix2 = torch.arange(1, n, 2, device=model_outputs.device)
        assert (target[ix1] == target[ix2]).all(), 'Wrong embedding positions'
        x = model_outputs[ix1]
        y = model_outputs[ix2]
        repr_loss = F.mse_loss(x, y)
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)
        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2
        cov_x = x.T @ x / (n - 1)
        cov_y = y.T @ y / (n - 1)
        cov_loss = self.off_diagonal(cov_x).pow_(2).sum().div(m) + self.off_diagonal(cov_y).pow_(2).sum().div(m)
        loss = self.sim_coeff * repr_loss + self.std_coeff * std_loss + self.cov_coeff * cov_loss
        return loss

    @staticmethod
    def off_diagonal(x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class CPC_Loss(nn.Module):

    def __init__(self, n_negatives=None, n_forward_steps=None):
        super().__init__()
        self.n_negatives = n_negatives
        self.n_forward_steps = n_forward_steps

    def _get_preds(self, base_embeddings, mapped_ctx_embeddings):
        batch_size, max_seq_len, emb_size = base_embeddings.payload.shape
        _, _, _, n_forward_steps = mapped_ctx_embeddings.payload.shape
        seq_lens = mapped_ctx_embeddings.seq_lens
        device = mapped_ctx_embeddings.payload.device
        len_mask = torch.arange(max_seq_len).unsqueeze(0).expand(batch_size, -1)
        len_mask = (len_mask < seq_lens.unsqueeze(1).expand(-1, max_seq_len)).float()
        possible_negatives = base_embeddings.payload.view(batch_size * max_seq_len, emb_size)
        mask = len_mask.unsqueeze(0).expand(batch_size, *len_mask.shape).clone()
        mask = mask.reshape(batch_size, -1)
        sample_ids = torch.multinomial(mask, self.n_negatives)
        neg_samples = possible_negatives[sample_ids]
        positive_preds, neg_preds = [], []
        len_mask_exp = len_mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, emb_size, n_forward_steps)
        trimmed_mce = mapped_ctx_embeddings.payload.mul(len_mask_exp)
        for i in range(1, n_forward_steps + 1):
            ce_i = trimmed_mce[:, 0:max_seq_len - i, :, i - 1]
            be_i = base_embeddings.payload[:, i:max_seq_len]
            positive_pred_i = ce_i.mul(be_i).sum(axis=-1)
            positive_preds.append(positive_pred_i)
            neg_pred_i = ce_i.matmul(neg_samples.transpose(-2, -1))
            neg_preds.append(neg_pred_i)
        return positive_preds, neg_preds

    def forward(self, embeddings, _):
        base_embeddings, _, mapped_ctx_embeddings = embeddings
        device = mapped_ctx_embeddings.payload.device
        positive_preds, neg_preds = self._get_preds(base_embeddings, mapped_ctx_embeddings)
        step_losses = []
        for positive_pred_i, neg_pred_i in zip(positive_preds, neg_preds):
            step_loss = -F.log_softmax(torch.cat([positive_pred_i.unsqueeze(-1), neg_pred_i], dim=-1), dim=-1)[:, :, 0].mean()
            step_losses.append(step_loss)
        loss = torch.stack(step_losses).mean()
        return loss

    def cpc_accuracy(self, embeddings, _):
        base_embeddings, _, mapped_ctx_embeddings = embeddings
        positive_preds, neg_preds = self._get_preds(base_embeddings, mapped_ctx_embeddings)
        batch_size, max_seq_len, emb_size = base_embeddings.payload.shape
        seq_lens = mapped_ctx_embeddings.seq_lens
        device = mapped_ctx_embeddings.payload.device
        len_mask = torch.arange(max_seq_len).unsqueeze(0).expand(batch_size, -1)
        len_mask = (len_mask < seq_lens.unsqueeze(1).expand(-1, max_seq_len)).float()
        total, accurate = 0, 0
        for i, (positive_pred_i, neg_pred_i) in enumerate(zip(positive_preds, neg_preds)):
            i_mask = len_mask[:, i + 1:max_seq_len]
            total += i_mask.sum().item()
            accurate += (((positive_pred_i.unsqueeze(-1).expand(*neg_pred_i.shape) > neg_pred_i).sum(dim=-1) == self.n_negatives) * i_mask).sum().item()
        return accurate / total


class TabformerInferenceModule(torch.nn.Module):

    def __init__(self, pretrained_model):
        super().__init__()
        self.model = pretrained_model
        self.model._seq_encoder.is_reduce_sequence = True
        self.model._seq_encoder.add_cls_output = False
        if self.model.hparams.inference_pooling_strategy == 'stat':
            self.stat_pooler = StatPooling()

    def forward(self, batch: PaddedBatch):
        z_trx = self.model.trx_encoder(batch)
        payload = z_trx.payload.view(z_trx.payload.shape[:-1] + (-1, self.model.feature_emb_dim))
        payload = self.model.feature_encoder(payload)
        encoded_trx = PaddedBatch(payload=payload, length=z_trx.seq_lens)
        out = self.model._seq_encoder(encoded_trx)
        if self.model.hparams.inference_pooling_strategy == 'stat':
            stats = self.stat_pooler(z_trx)
            out = torch.cat([stats, out], dim=1)
        if self.model.hparams.norm_predict:
            out = out / (out.pow(2).sum(dim=-1, keepdim=True) + 1e-09).pow(0.5)
        return out


class PairwiseMarginRankingLoss(nn.Module):

    def __init__(self, margin=0.0, size_average=None, reduce=None, reduction='mean'):
        """
        Pairwise Margin Ranking Loss. All setted parameters redirected to nn.MarginRankingLoss.
        All the difference is that pairs automatically generated for margin ranking loss.
        All possible pairs of different class are generated.
        """
        super().__init__()
        self.margin_loss = nn.MarginRankingLoss(margin, size_average, reduce, reduction)

    def forward(self, prediction, label):
        """
        Get pairwise margin ranking loss.
        :param prediction: tensor of shape Bx1 of predicted probabilities
        :param label: tensor of shape Bx1 of true labels for pair generation
        """
        mask_0 = label == 0
        mask_1 = label == 1
        pred_0 = torch.masked_select(prediction, mask_0)
        pred_1 = torch.masked_select(prediction, mask_1)
        pred_1_n = pred_1.size()[0]
        pred_0_n = pred_0.size()[0]
        if pred_1_n > 0 and pred_0_n:
            pred_00 = pred_0.unsqueeze(0).repeat(1, pred_1_n)
            pred_11 = pred_1.unsqueeze(1).repeat(1, pred_0_n).view(pred_00.size())
            out01 = -1 * torch.ones(pred_1_n * pred_0_n)
            return self.margin_loss(pred_00.view(-1), pred_11.view(-1), out01)
        else:
            return torch.sum(prediction) * 0.0


class MultiLoss(nn.Module):
    """Works like `ptls.contrastive_learning.losses.complex_loss.ComplexLoss`

    """

    def __init__(self, losses):
        super().__init__()
        self.losses = nn.ModuleList(losses)

    def forward(self, pred, true):
        loss = 0
        for weight, criterion in self.losses:
            loss = weight * criterion(pred, true) + loss
        return loss


class TransactionSumLoss(nn.Module):

    def __init__(self, n_variables_to_predict):
        super().__init__()
        self.bce_with_logits = nn.BCEWithLogitsLoss()
        self.n_variables_to_predict = n_variables_to_predict

    def forward(self, pred, true):
        loss = self.bce_with_logits(pred[:, 1:self.n_variables_to_predict], true[:, 1:self.n_variables_to_predict])
        return loss


class AllStateLoss(nn.Module):

    def __init__(self, point_loss):
        super().__init__()
        self.loss = point_loss

    def forward(self, pred: PaddedBatch, true):
        y = torch.cat([torch.Tensor([yb] * length) for yb, length in zip(true, pred.seq_lens)])
        weights = torch.cat([(torch.arange(1, length + 1) / length) for length in pred.seq_lens])
        loss = self.loss(pred, y, weights)
        return loss


class BCELoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss = nn.BCELoss()

    def forward(self, pred, true):
        return self.loss(pred.float(), true.float())


class MSELoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, pred, true):
        return self.loss(pred.float(), true.float())


class PseudoLabeledLoss(nn.Module):

    def __init__(self, loss, pl_threshold=0.5, unlabeled_weight=1.0):
        super().__init__()
        self.loss = loss
        self.pl_threshold = pl_threshold
        self.unlabeled_weight = unlabeled_weight

    def forward(self, pred, true):
        label_pred, unlabel_pred = pred['labeled'], pred['unlabeled']
        if isinstance(self.loss, nn.NLLLoss):
            pseudo_labels = torch.argmax(unlabel_pred.detach(), 1)
        elif isinstance(self.loss, BCELoss):
            pseudo_labels = (unlabel_pred.detach() > 0.5).type(torch.int64)
        else:
            raise Exception(f'unknown loss type: {self.loss}')
        if isinstance(self.loss, nn.NLLLoss):
            probs = torch.exp(unlabel_pred.detach())
            mask = probs.max(1)[0] > self.pl_threshold
        elif isinstance(self.loss, BCELoss):
            probs = unlabel_pred.detach()
            mask = abs(probs - (1 - pseudo_labels)) > self.pl_threshold
        else:
            mask = torch.ones(unlabel_pred.shape[0]).bool()
        Lloss = self.loss(label_pred, true)
        if mask.sum() == 0:
            return Lloss
        else:
            Uloss = self.unlabeled_weight * self.loss(unlabel_pred[mask], pseudo_labels[mask])
            return (Lloss + Uloss) / (1 + self.unlabeled_weight)


class UnsupervisedTabNetLoss(nn.Module):

    def __init__(self, eps=1e-09):
        super().__init__()
        self.eps = eps

    def forward(self, output, obf_vars):
        embedded_x, y_pred = output
        errors = y_pred - embedded_x
        reconstruction_errors = torch.mul(errors, obf_vars) ** 2
        batch_stds = torch.std(embedded_x, dim=0) ** 2 + self.eps
        features_loss = torch.matmul(reconstruction_errors, 1 / batch_stds)
        nb_reconstructed_variables = torch.sum(obf_vars, dim=1)
        features_loss = features_loss / (nb_reconstructed_variables + self.eps)
        loss = torch.mean(features_loss)
        return loss


def cross_entropy(pred, soft_targets):
    logsoftmax = torch.nn.LogSoftmax(dim=1)
    device = pred.device
    return torch.mean(torch.sum(-soft_targets * logsoftmax(pred), 1))


def mse_loss(pred, actual):
    device = pred.device
    return torch.mean((pred - actual) ** 2)


class DistributionTargetsLoss(nn.Module):

    def __init__(self, mult1=3, mult2=0.167, mult3=1, mult4=1):
        super().__init__()
        self.mults = [mult1, mult2, mult3, mult4]

    def forward(self, pred, true):
        log_sum_truth_neg = np.log(np.abs(true['neg_sum'].astype(np.float)) + 1)[:, None] if isinstance(true['neg_sum'], np.ndarray) else 0
        distribution_truth_neg = np.array(true['neg_distribution'].tolist()) if isinstance(true['neg_sum'], np.ndarray) else 0
        log_sum_truth_pos = np.log(np.abs(true['pos_sum'].astype(np.float)) + 1)[:, None]
        distribution_truth_pos = np.array(true['pos_distribution'].tolist())
        log_sum_hat_neg, distribution_hat_neg = pred['neg_sum'], pred['neg_distribution']
        log_sum_hat_pos, distribution_hat_pos = pred['pos_sum'], pred['pos_distribution']
        device = log_sum_hat_pos.device
        loss_sum_pos = mse_loss(log_sum_hat_pos, torch.tensor(log_sum_truth_pos, device=device).float())
        loss_distr_pos = cross_entropy(distribution_hat_pos, torch.tensor(distribution_truth_pos, device=device))
        if isinstance(true['neg_sum'], np.ndarray):
            loss_sum_neg = mse_loss(log_sum_hat_neg, torch.tensor(log_sum_truth_neg, device=device).float())
            loss_distr_neg = cross_entropy(distribution_hat_neg, torch.tensor(distribution_truth_neg, device=device))
        loss = loss_sum_neg * self.mults[0] + loss_sum_pos * self.mults[1] + loss_distr_neg * self.mults[2] + loss_distr_pos * self.mults[3] if isinstance(true['neg_sum'], np.ndarray) else loss_sum_pos * self.mults[1] + loss_distr_pos * self.mults[3]
        return loss.float()


class ZILNLoss(nn.Module):
    """
    Zero-inflated lognormal (ZILN) loss adapted for multinomial target with K categories.
    Please cite [https://arxiv.org/abs/1912.07753] and [https://github.com/google/lifetime_value].
    Defaults to MSE loss for 1D-input and lognormal loss for 2D-input.

    Parameters
    ----------
    pred: tensor of shape (B) or (B, K')
    target: tensor of shape (B) or (B, K)
    """

    def __init__(self):
        super().__init__()
        self.eps = 1e-08

    def forward(self, pred, target):
        t = target if target.dim() == 1 else target.sum(dim=1)
        if pred.dim() == 1:
            return torch.mean((pred - t).square())
        sigma = F.softplus(pred[:, 1])
        loss = (sigma * t + self.eps).log() + ((t + self.eps).log() - pred[:, 0]).square() / (2 * sigma.square() + self.eps)
        if pred.shape[1] == 2:
            pass
        elif pred.shape[1] == 3:
            loss = torch.where(t > 0, loss - F.logsigmoid(pred[:, 2]), -F.logsigmoid(-pred[:, 2]))
        elif pred.shape[1] == target.shape[1] + 1:
            dist = torch.cat(((t == 0).float().unsqueeze(-1), target), dim=1)
            loss = -dist.mul(F.log_softmax(pred, dim=1)).sum(dim=1)
        elif pred.shape[1] == target.shape[1] + 3:
            log_prob = F.log_softmax(pred[:, 2:], dim=1)
            loss = torch.where(t > 0, loss - target.mul(log_prob[:, 1:]).sum(dim=1), -log_prob[:, 0])
        else:
            raise Exception(f'{self.__class__} got incorrect input sizes')
        return torch.mean(loss)


class ModelEmbeddingEnsemble(nn.Module):

    def __init__(self, submodels):
        super(ModelEmbeddingEnsemble, self).__init__()
        self.models = nn.ModuleList(submodels)

    def forward(self, x: PaddedBatch, h_0: torch.Tensor=None):
        """
        x - PaddedBatch of transactions sequences
        h_0 - previous state of embeddings (initial size for GRU). torch Tensor of shape (batch_size, embedding_size)
        """
        if h_0 is not None:
            h_0_splitted = torch.chunk(h_0, len(self.models), -1)
            out = torch.cat([m(x, h.contiguous()) for m, h in zip(self.models, h_0_splitted)], dim=-1)
        else:
            out = torch.cat([m(x) for i, m in enumerate(self.models)], dim=-1)
        return out


def projection_head(input_size, output_size):
    layers = [torch.nn.Linear(input_size, input_size), torch.nn.ReLU(), torch.nn.Linear(input_size, output_size)]
    m = torch.nn.Sequential(*layers)
    return m


class ComplexModel(torch.nn.Module):

    def __init__(self, ml_model, params):
        super().__init__()
        self.ml_model = ml_model
        self.projection_ml_head = projection_head(params.rnn.hidden_size, params.ml_projection_head.output_size)
        self.projection_aug_head = torch.nn.Sequential(projection_head(params.rnn.hidden_size, params.aug_projection_head.output_size), torch.nn.LogSoftmax(dim=1))

    def forward(self, x):
        encoder_output = self.ml_model(x)
        ml_head_output = self.projection_ml_head(encoder_output)
        aug_head_output = self.projection_aug_head(encoder_output)
        return aug_head_output, ml_head_output


class Binarization(Function):

    @staticmethod
    def forward(self, x):
        q = (x > 0).float()
        return 2 * q - 1

    @staticmethod
    def backward(self, grad_outputs):
        return grad_outputs


binary = Binarization.apply


class BinarizationLayer(nn.Module):

    def __init__(self, hs_from, hs_to):
        super(BinarizationLayer, self).__init__()
        self.linear = nn.Linear(hs_from, hs_to, bias=False)

    def forward(self, x):
        return binary(self.linear(x))


class L2NormEncoder(nn.Module):

    def __init__(self, eps=1e-09):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor):
        return x / (x.pow(2).sum(dim=-1, keepdim=True) + self.eps).pow(0.5)


class Head(torch.nn.Module):
    """Head for the sequence encoder

    Parameters
    ----------
         input_size: int
            input size
         use_norm_encoder: bool. Default: False
            whether to use normalization layers before the head
         use_batch_norm: bool. Default: False.
            whether to use BatchNorm.
         hidden_layers_sizes: List[int]. Default: None.
            sizes of linear layers. If None without additional linear layers.
         objective: str. Options:
            None (default) - corresponds to linear output with relu
            classification - linear output with sigmoid or logsoftmax (num_classes > 1)
            regression - pure linear output
            softplus - linear output with softplus
         num_classes: int. Default: 1.
            The number of classed in classification problem. Default correspond to binary classification.

     """

    def __init__(self, input_size: int=None, use_norm_encoder: bool=False, use_batch_norm: bool=False, hidden_layers_sizes: List[int]=None, drop_probs: List[float]=None, objective: str=None, num_classes: int=1):
        super().__init__()
        layers = []
        if use_norm_encoder:
            layers.append(L2NormEncoder())
        if use_batch_norm:
            layers.append(BatchNorm1d(input_size))
        if drop_probs:
            assert len(drop_probs) == len(hidden_layers_sizes), 'dimensions of `drop_probs` and `hidden_layers_sizes` should be equal'
        if hidden_layers_sizes is not None:
            layers_size = [input_size] + list(hidden_layers_sizes)
            for ix, (size_in, size_out) in enumerate(zip(layers_size[:-1], layers_size[1:])):
                layers.append(Linear(size_in, size_out))
                if use_batch_norm:
                    layers.append(BatchNorm1d(size_out))
                layers.append(ReLU())
                if drop_probs:
                    layers.append(Dropout(drop_probs[ix]))
                input_size = size_out
        if objective == 'classification':
            if num_classes == 1:
                h = Sequential(Linear(input_size, num_classes), Sigmoid(), Flatten(0))
            else:
                h = Sequential(Linear(input_size, num_classes), LogSoftmax(dim=1))
            layers.append(h)
        elif objective == 'regression':
            if num_classes == 1:
                layers.append(Sequential(Linear(input_size, 1), Flatten(0)))
            else:
                layers.append(Linear(input_size, num_classes))
        elif objective == 'softplus':
            if num_classes == 1:
                layers.append(Sequential(Linear(input_size, num_classes), Softplus(), Flatten(0)))
            else:
                layers.append(Sequential(Linear(input_size, num_classes), Softplus()))
        elif objective is not None:
            raise AttributeError(f'Unknown objective {objective}. Supported: classification, regression and softplus.')
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class PBFeatureExtract(torch.nn.Module):

    def __init__(self, feature_col_name, as_padded_batch):
        super().__init__()
        self.feature_col_name = feature_col_name
        self.as_padded_batch = as_padded_batch

    def forward(self, x: PaddedBatch):
        feature = x.payload[self.feature_col_name]
        if self.as_padded_batch:
            return PaddedBatch(feature, x.seq_lens)
        return feature


class AbsSeqEncoder(torch.nn.Module):

    def __init__(self, is_reduce_sequence=True):
        super().__init__()
        self._is_reduce_sequence = is_reduce_sequence

    @property
    def is_reduce_sequence(self):
        return self._is_reduce_sequence

    @is_reduce_sequence.setter
    def is_reduce_sequence(self, value):
        self._is_reduce_sequence = value

    @property
    def embedding_size(self):
        raise NotImplementedError()


class AggFeatureSeqEncoder(torch.nn.Module):
    """Calculates statistics over feature arrays and return them as embedding.

    Result is high dimension non learnable vector.
    Such embedding can be used in busting ML algorithm.

    Statistics are calculated by numerical features, with grouping by category values.

    This seq-encoder haven't TrxEncoder, they consume raw features.

    Parameters
        embeddings:
            dict with categorical feature names.
            Values must be like this `{'in': dictionary_size}`.
            One hot encoding will be applied to these features.
            Output vector size is (dictionary_size,)
        numeric_values:
            dict with numerical feature names.
            Values may contains some options
            numeric_values are multiplied with OHE categories.
            Then statistics are calculated over time dimensions.
        was_logified (bool):
            True - means that original numerical features was log transformed
            AggFeatureSeqEncoder will use `exp` to get original value
        log_scale_factor:
            Use it with `was_logified=True`. value will be multiplied by log_scale_factor before exp.
        is_used_count (bool):
            Use of not count by values
        is_used_mean (bool):
            Use of not mean by values
        is_used_std (bool):
            Use of not std by values
        is_used_min (bool):
            Use of not min by values
        is_used_max (bool):
            Use of not max by values
        use_topk_cnt (int):
            Define the K for topk features calculation. 0 if not used
        distribution_target_task (bool):
            Calc more features
        logify_sum_mean_seqlens (bool):
            True - apply log transform to sequence length

    Example:

    """

    def __init__(self, embeddings=None, numeric_values=None, was_logified=True, log_scale_factor=1, is_used_count=True, is_used_mean=True, is_used_std=True, is_used_min=False, is_used_max=False, use_topk_cnt=0, distribution_target_task=False, logify_sum_mean_seqlens=False):
        super().__init__()
        self.numeric_values = OrderedDict(numeric_values.items())
        self.embeddings = OrderedDict(embeddings.items())
        self.was_logified = was_logified
        self.log_scale_factor = log_scale_factor
        self.distribution_target_task = distribution_target_task
        self.logify_sum_mean_seqlens = logify_sum_mean_seqlens
        self.eps = 1e-09
        self.ohe_buffer = {}
        for col_embed, options_embed in self.embeddings.items():
            size = options_embed['in']
            ohe = torch.diag(torch.ones(size))
            self.ohe_buffer[col_embed] = ohe
            self.register_buffer(f'ohe_{col_embed}', ohe)
        self.is_used_count = is_used_count
        self.is_used_mean = is_used_mean
        self.is_used_std = is_used_std
        self.is_used_min = is_used_min
        self.is_used_max = is_used_max
        self.use_topk_cnt = use_topk_cnt

    def forward(self, x: PaddedBatch):
        """
        {
            'cat_i': [B, T]: int
            'num_i': [B, T]: float
        }
        to
        {
            [B, H] where H - is [f1, f2, f3, ... fn]
        }
        :param x:
        :return:
        """
        feature_arrays = x.payload
        device = x.device
        B, T = x.seq_feature_shape
        seq_lens = x.seq_lens.float()
        if self.logify_sum_mean_seqlens:
            processed = [torch.log(seq_lens.unsqueeze(1))]
        else:
            processed = [seq_lens.unsqueeze(1)]
        cat_processed = []
        for col_num, options_num in self.numeric_values.items():
            if col_num.strip('"') == '#ones':
                val_orig = torch.ones(B, T, device=device)
            else:
                val_orig = feature_arrays[col_num].float()
            if any((type(options_num) is str and self.was_logified, type(options_num) is dict and options_num.get('was_logified', False))):
                val_orig = torch.expm1(self.log_scale_factor * torch.abs(val_orig)) * torch.sign(val_orig)
            sum_ = val_orig.sum(dim=1).unsqueeze(1)
            a = torch.clamp(val_orig.pow(2).sum(dim=1) - val_orig.sum(dim=1).pow(2).div(seq_lens + self.eps), min=0.0)
            mean_ = val_orig.sum(dim=1).div(seq_lens + self.eps).unsqueeze(1)
            std_ = a.div(torch.clamp(seq_lens - 1, min=0.0) + self.eps).pow(0.5).unsqueeze(1)
            if self.distribution_target_task:
                sum_pos = torch.clamp(val_orig, min=0).sum(dim=1).unsqueeze(1)
                processed.append(torch.log(sum_pos + 1))
                sum_neg = torch.clamp(val_orig, max=0).sum(dim=1).unsqueeze(1)
                processed.append(-1 * torch.log(-sum_neg + 1))
            elif self.logify_sum_mean_seqlens:
                processed.append(torch.sign(sum_) * torch.log(torch.abs(sum_) + 1))
            else:
                processed.append(sum_)
            if self.logify_sum_mean_seqlens:
                mean_ = torch.sign(mean_) * torch.log(torch.abs(mean_) + 1)
            processed.append(mean_)
            if not self.distribution_target_task:
                processed.append(std_)
            for col_embed, options_embed in self.embeddings.items():
                ohe = getattr(self, f'ohe_{col_embed}')
                val_embed = feature_arrays[col_embed].long()
                val_embed = val_embed.clamp(0, options_embed['in'] - 1)
                ohe_transform = ohe[val_embed.flatten()].view(*val_embed.size(), -1)
                m_sum = ohe_transform * val_orig.unsqueeze(-1)
                mask = (1.0 - ohe[0]).unsqueeze(0)
                e_cnt = ohe_transform.sum(dim=1) * mask
                if self.is_used_count:
                    processed.append(e_cnt)
                if self.is_used_mean:
                    e_sum = m_sum.sum(dim=1)
                    e_mean = e_sum.div(e_cnt + 1e-09)
                    processed.append(e_mean)
                if self.is_used_std:
                    a = torch.clamp(m_sum.pow(2).sum(dim=1) - m_sum.sum(dim=1).pow(2).div(e_cnt + 1e-09), min=0.0)
                    e_std = a.div(torch.clamp(e_cnt - 1, min=0) + 1e-09).pow(0.5)
                    processed.append(e_std)
                if self.is_used_min:
                    min_ = m_sum.masked_fill(~x.seq_len_mask.bool().unsqueeze(2), np.float32('inf')).min(dim=1).values
                    processed.append(min_)
                if self.is_used_max:
                    max_ = m_sum.masked_fill(~x.seq_len_mask.bool().unsqueeze(2), np.float32('-inf')).max(dim=1).values
                    processed.append(max_)
        for col_embed, options_embed in self.embeddings.items():
            ohe = getattr(self, f'ohe_{col_embed}')
            val_embed = feature_arrays[col_embed].long()
            val_embed = val_embed.clamp(0, options_embed['in'] - 1)
            ohe_transform = ohe[val_embed.flatten()].view(*val_embed.size(), -1)
            mask = (1.0 - ohe[0]).unsqueeze(0)
            e_cnt = ohe_transform.sum(dim=1) * mask
            processed.append(e_cnt.gt(0.0).float().sum(dim=1, keepdim=True))
            if self.use_topk_cnt > 0:
                cat_processed.append(torch.topk(e_cnt, self.use_topk_cnt, dim=1)[1])
        for i, t in enumerate(processed):
            if torch.isnan(t).any():
                raise Exception(f'nan in {i}')
        out = torch.cat(processed + cat_processed, 1)
        return out

    @property
    def embedding_size(self):
        numeric_values = self.numeric_values
        embeddings = self.embeddings
        e_sizes = [options_embed['in'] for col_embed, options_embed in embeddings.items()]
        out_size = 1
        n_features = sum([int(v) for v in [self.is_used_count, self.is_used_mean, self.is_used_std, self.is_used_min, self.is_used_max]])
        out_size += len(numeric_values) * (3 + n_features * sum(e_sizes)) + len(embeddings)
        out_size += len(e_sizes) * self.use_topk_cnt
        return out_size

    @property
    def cat_output_size(self):
        e_sizes = [options_embed['in'] for col_embed, options_embed in self.embeddings.items()]
        return len(e_sizes) * self.use_topk_cnt

    @property
    def category_names(self):
        return set([field_name for field_name in self.embeddings.keys()] + [value_name for value_name in self.numeric_values.keys()])

    @property
    def category_max_size(self):
        return {k: v['in'] for k, v in self.embeddings.items()}


class SeqEncoderContainer(torch.nn.Module):
    """Base container class for Sequence encoder.
    Include `TrxEncoder` and `ptls.seq_encoder.abs_seq_encoder.AbsSeqEncoder` implementation

    Parameters
        trx_encoder:
            TrxEncoder object
        seq_encoder_cls:
            AbsSeqEncoder implementation class
        input_size:
            input_size parameter for seq_encoder_cls
            If None: input_size = trx_encoder.output_size
            Set input_size explicit or use None if your trx_encoder object has output_size attribute
        seq_encoder_params:
            dict with params for seq_encoder_cls initialisation
        is_reduce_sequence:
            False - returns PaddedBatch with all transactions embeddings
            True - returns one embedding for sequence based on CLS token

    """

    def __init__(self, trx_encoder, seq_encoder_cls, input_size, seq_encoder_params, is_reduce_sequence=True):
        super().__init__()
        self.trx_encoder = trx_encoder
        self.seq_encoder = seq_encoder_cls(input_size=input_size if input_size is not None else trx_encoder.output_size, is_reduce_sequence=is_reduce_sequence, **seq_encoder_params)

    @property
    def is_reduce_sequence(self):
        return self.seq_encoder.is_reduce_sequence

    @is_reduce_sequence.setter
    def is_reduce_sequence(self, value):
        self.seq_encoder.is_reduce_sequence = value

    @property
    def category_max_size(self):
        return self.trx_encoder.category_max_size

    @property
    def category_names(self):
        return self.trx_encoder.category_names

    @property
    def embedding_size(self):
        return self.seq_encoder.embedding_size

    def forward(self, x):
        x = self.trx_encoder(x)
        x = self.seq_encoder(x)
        return x


class FirstStepEncoder(nn.Module):
    """
    Class is used by ptls.nn.RnnSeqEncoder class for reducing RNN output with shape (B, L, H)
    to embeddings tensor with shape (B, H). The first hidden state is used for embedding.
    
    where:
        B - batch size
        L - sequence length
        H - hidden RNN size
    
    Example of usage: seq_encoder = RnnSeqEncoder(..., reducer='first_step')
    """

    def forward(self, x: PaddedBatch):
        h = x.payload[:, 0, :]
        return h


class LastMaxAvgEncoder(nn.Module):
    """
    Class is used by ptls.nn.RnnSeqEncoder class for reducing RNN output with shape (B, L, H)
    to embeddings tensor with shape (B, 3 * H). Embeddings are created by concatenating:
        - last hidden state from RNN output,
        - max pool over all hidden states of RNN output,
        - average pool over all hidden states of RNN output.
        
    where:
        B - batch size
        L - sequence length
        H - hidden RNN size
        
    Example of usage: seq_encoder = RnnSeqEncoder(..., reducer='last_max_avg')
    """

    def forward(self, x: PaddedBatch):
        rnn_max_pool = x.payload.max(dim=1)[0]
        rnn_avg_pool = x.payload.sum(dim=1) / x.seq_lens.unsqueeze(-1)
        h = x.payload[range(len(x.payload)), [(l - 1) for l in x.seq_lens]]
        h = torch.cat((h, rnn_max_pool, rnn_avg_pool), dim=-1)
        return h


class LastStepEncoder(nn.Module):
    """
    Class is used by ptls.nn.RnnSeqEncoder for reducing RNN output with shape (B, L, H), where
        B - batch size
        L - sequence length
        H - hidden RNN size
    to embeddings tensor with shape (B, H). The last hidden state is used for embedding.
    
    Example of usage: seq_encoder = RnnSeqEncoder(..., reducer='last_step')
    """

    def forward(self, x: PaddedBatch):
        h = x.payload[range(len(x.payload)), [(l - 1) for l in x.seq_lens]]
        return h


class RnnEncoder(AbsSeqEncoder):
    """Use torch recurrent layer network
    Based on `torch.nn.GRU` and `torch.nn.LSTM`

    Parameters
        input_size:
            input embedding size
        hidden_size:
            intermediate and output layer size
        type:
            'gru' or 'lstm'
            Type of rnn network
        bidir:
            Bidirectional RNN
        dropout:
            RNN dropout
        trainable_starter:
            'static' - use random learnable vector for rnn starter
            other values - use None as starter
        is_reduce_sequence:
            False - returns PaddedBatch with all transactions embeddings
            True - returns one embedding for sequence based on CLS token

    Example:
    >>> model = RnnEncoder(
    >>>     input_size=5,
    >>>     hidden_size=6,
    >>>     is_reduce_sequence=False,
    >>> )
    >>> x = PaddedBatch(
    >>>     payload=torch.arange(4*5*8).view(4, 8, 5).float(),
    >>>     length=torch.tensor([4, 2, 6, 8])
    >>> )
    >>> out = model(x)
    >>> assert out.payload.shape == (4, 8, 6)

    """

    def __init__(self, input_size=None, hidden_size=None, type='gru', bidir=False, num_layers=1, dropout=0, trainable_starter='static', is_reduce_sequence=False, reducer='last_step'):
        super().__init__(is_reduce_sequence=is_reduce_sequence)
        self.hidden_size = hidden_size
        self.rnn_type = type
        self.bidirectional = bidir
        if self.bidirectional:
            warnings.warn('Backward direction in bidir RNN takes into account paddings at the end of sequences!')
        self.trainable_starter = trainable_starter
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size, self.hidden_size, num_layers=num_layers, batch_first=True, bidirectional=self.bidirectional, dropout=dropout)
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(input_size, self.hidden_size, num_layers=num_layers, batch_first=True, bidirectional=self.bidirectional, dropout=dropout)
        else:
            raise Exception(f'wrong rnn type "{self.rnn_type}"')
        self.full_hidden_size = self.hidden_size if not self.bidirectional else self.hidden_size * 2
        if self.trainable_starter == 'static':
            num_dir = 2 if self.bidirectional else 1
            self.starter_h = nn.Parameter(torch.randn(num_dir, 1, self.hidden_size))
        if reducer == 'last_step':
            self.reducer = LastStepEncoder()
        elif reducer == 'first_step':
            self.reducer = FirstStepEncoder()
        elif reducer == 'last_max_avg':
            self.reducer = LastMaxAvgEncoder()

    def forward(self, x: PaddedBatch, h_0: torch.Tensor=None):
        """

        :param x:
        :param h_0: None or [1, B, H] float tensor
                    0.0 values in all components of hidden state of specific client means no-previous state and
                    use starter for this client
                    h_0 = None means no-previous state for all clients in batch
        :return:
        """
        shape = x.payload.size()
        assert shape[1] > 0, "Batch can'not have 0 transactions"
        if self.trainable_starter == 'static':
            starter_h = torch.tanh(self.starter_h.expand(-1, shape[0], -1).contiguous())
            if h_0 is None:
                h_0 = starter_h
            elif h_0 is not None and not self.training:
                h_0 = torch.where((h_0.squeeze(0).abs().sum(dim=1) == 0.0).unsqueeze(0).unsqueeze(2).expand(*starter_h.size()), starter_h, h_0)
            else:
                raise NotImplementedError('Unsupported mode: cannot mix fixed X and learning Starter')
        if self.rnn_type == 'lstm':
            out, _ = self.rnn(x.payload)
        elif self.rnn_type == 'gru':
            out, _ = self.rnn(x.payload, h_0)
        else:
            raise Exception(f'wrong rnn type "{self.rnn_type}"')
        out = PaddedBatch(out, x.seq_lens)
        if self.is_reduce_sequence:
            return self.reducer(out)
        return out

    @property
    def embedding_size(self):
        return self.hidden_size


class RnnSeqEncoder(SeqEncoderContainer):
    """SeqEncoderContainer with RnnEncoder

    Supports incremental embedding calculation.
    Each RNN step requires previous hidden state. Hidden state passed through the iterations during sequence processing.
    Starting hidden state required by RNN. Starting hidden state are depends on `RnnEncoder.trainable_starter`.
    You can also provide starting hidden state to `forward` method as `h_0`.
    This can be useful when you need to `update` your embedding with new transactions.

    Example:
        >>> seq_encoder = RnnSeqEncoder(...)
        >>> embedding_0 = seq_encoder(data_0)
        >>> embedding_1 = seq_encoder(data_1, h_0=embedding_0)
        >>> embedding_2a = seq_encoder(data_2, h_0=embedding_1)
        >>> embedding_2b = seq_encoder(data_2)
        >>> embedding_2c = seq_encoder(data_0 + data_1 + data_2)

    `embedding_2a` takes into account all transactions from `data_0`, `data_1` and `data_2`.
    `embedding_2b` takes into account only transactions from `data_2`.
    `embedding_2c` is the same as `embedding_2a`.
    `embedding_2a` calculated faster than `embedding_2c`.

    Incremental calculation works fast when you have long sequences and short updates. RNN just process short update.


    Parameters
        trx_encoder:
            TrxEncoder object
        input_size:
            input_size parameter for RnnEncoder
            If None: input_size = trx_encoder.output_size
            Set input_size explicit or use None if your trx_encoder object has output_size attribute
        **seq_encoder_params:
            params for RnnEncoder initialisation
        is_reduce_sequence:
            False - returns PaddedBatch with all transactions embeddings
            True - returns one embedding for sequence based on CLS token

    """

    def __init__(self, trx_encoder=None, input_size=None, is_reduce_sequence=True, **seq_encoder_params):
        super().__init__(trx_encoder=trx_encoder, seq_encoder_cls=RnnEncoder, input_size=input_size, seq_encoder_params=seq_encoder_params, is_reduce_sequence=is_reduce_sequence)

    def forward(self, x, h_0=None):
        x = self.trx_encoder(x)
        x = self.seq_encoder(x, h_0)
        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, use_start_random_shift=True, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.use_start_random_shift = use_start_random_shift
        self.max_len = max_len
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        T = x.size(1)
        if self.training and self.use_start_random_shift:
            start_pos = random.randint(0, self.max_len - T)
        else:
            start_pos = 0
        x = x + self.pe[:, start_pos:start_pos + T]
        return x


class TransformerEncoder(AbsSeqEncoder):
    """Used torch implementation of transformer
    Based on `torch.nn.TransformerEncoder`

    Parameters
        input_size:
            input embedding size.
            Equals intermediate and output layer size cause transformer don't change vector dimentions
        train_starter:
            'randn' or 'zeros'
            Which token used for CLS token, random learnable or zeros fixed
        shared_layers:
            True - then the same weights used for all `n_layers`.
            False - `n_layers` used different weights
        n_heads:
            The number of heads in the multiheadattention models
        dim_hidden:
            The dimension of the feedforward network model
        dropout:
            The dropout value
        n_layers:
            The number of sub-encoder-layers in the encoder
        use_positional_encoding (bool):
            Use or not positional encoding
        use_start_random_shift (bool):
            True - starting pos of positional encoding randomly shifted when training
            This allow to train transformer with all range of positional encoding values
            False - starting pos is not shifted.
        max_seq_len:
            The possible maximum sequence length for positional encoding
        use_after_mask:
            True value makes transformer unidirectional
        use_src_key_padding_mask:
            Padding simbols aren't used in attention bases on sequences lenghts
        use_norm_layer:
            Use or not LayerNorm
        is_reduce_sequence (bool):
            False - returns PaddedBatch with all transactions embeddings
            True - returns one embedding for sequence based on CLS token

    Example:
    >>> model = TransformerEncoder(input_size=32)
    >>> x = PaddedBatch(torch.randn(10, 128, 32), torch.randint(20, 128, (10,)))
    >>> y = model(x)
    >>> assert y.payload.size() == (10, 128, 32)
    >>>
    >>> model = TransformerEncoder(input_size=32, is_reduce_sequence=True)
    >>> y = model(x)
    >>> assert y.size() == (10, 32)

    """

    def __init__(self, input_size, starter='randn', shared_layers=False, n_heads=8, dim_hidden=256, dropout=0.1, n_layers=6, use_positional_encoding=True, use_start_random_shift=True, max_seq_len=5000, use_after_mask=False, use_src_key_padding_mask=True, use_norm_layer=True, is_reduce_sequence=False):
        super().__init__(is_reduce_sequence=is_reduce_sequence)
        self.input_size = input_size
        self.shared_layers = shared_layers
        self.n_layers = n_layers
        self.use_after_mask = use_after_mask
        self.use_src_key_padding_mask = use_src_key_padding_mask
        self.use_positional_encoding = use_positional_encoding
        if starter == 'randn':
            self.starter = torch.nn.Parameter(torch.randn(1, 1, input_size), requires_grad=True)
        elif starter == 'zeros':
            self.starter = torch.nn.Parameter(torch.zeros(1, 1, input_size), requires_grad=False)
        else:
            raise AttributeError(f'Unknown train_starter: "{starter}". Expected one of [randn, zeros]')
        enc_layer = torch.nn.TransformerEncoderLayer(d_model=input_size, nhead=n_heads, dim_feedforward=dim_hidden, dropout=dropout, batch_first=True)
        enc_norm = torch.nn.LayerNorm(input_size) if use_norm_layer else None
        if self.shared_layers:
            self.enc_layer = enc_layer
            self.enc_norm = enc_norm
        else:
            self.enc = torch.nn.TransformerEncoder(enc_layer, n_layers, enc_norm)
        if self.use_positional_encoding:
            self.pe = PositionalEncoding(use_start_random_shift=use_start_random_shift, max_len=max_seq_len, d_model=input_size)

    @staticmethod
    def generate_square_subsequent_mask(sz):
        """Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask[0, :] = 0.0
        mask[:, 0] = 0.0
        return mask

    def forward(self, x: PaddedBatch):
        B, T, H = x.payload.size()
        if self.use_after_mask:
            src_mask = self.generate_square_subsequent_mask(T + 1)
        else:
            src_mask = None
        if self.use_src_key_padding_mask:
            src_key_padding_mask = torch.cat([torch.zeros(B, 1, dtype=torch.long, device=x.device), 1 - x.seq_len_mask], dim=1).bool()
        else:
            src_key_padding_mask = None
        x_in = x.payload
        if self.use_positional_encoding:
            x_in = self.pe(x_in)
        x_in = torch.cat([self.starter.expand(B, 1, H), x_in], dim=1)
        if self.shared_layers:
            out = x_in
            for _ in range(self.n_layers):
                out = self.enc_layer(out, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
                if self.enc_norm is not None:
                    out = self.enc_norm(out)
        else:
            out = self.enc(x_in, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        if self.is_reduce_sequence:
            return out[:, 0, :]
        return PaddedBatch(out[:, 1:, :], x.seq_lens)

    @property
    def embedding_size(self):
        return self.input_size


class TransformerSeqEncoder(SeqEncoderContainer):
    """SeqEncoderContainer with TransformerEncoder

    Parameters
        trx_encoder:
            TrxEncoder object
        input_size:
            input_size parameter for TransformerEncoder
            If None: input_size = trx_encoder.output_size
            Set input_size explicit or use None if your trx_encoder object has output_size attribute
        **seq_encoder_params:
            params for TransformerEncoder initialisation
        is_reduce_sequence:
            False - returns PaddedBatch with all transactions embeddings
            True - returns one embedding for sequence based on CLS token

    """

    def __init__(self, trx_encoder=None, input_size=None, is_reduce_sequence=True, **seq_encoder_params):
        super().__init__(trx_encoder=trx_encoder, seq_encoder_cls=TransformerEncoder, input_size=input_size, seq_encoder_params=seq_encoder_params, is_reduce_sequence=is_reduce_sequence)


class LongformerEncoder(AbsSeqEncoder):
    """Used huggingface implementation of transformer
    Based on `transformers.LongformerModel`
    [Link](https://huggingface.co/docs/transformers/main/en/model_doc/longformer#transformers.LongformerModel)

    Transformer-based models are unable to process long sequences due to their self-attention operation,
    which scales quadratically with the sequence length. To address this limitation, was introduce the Longformer
    with an attention mechanism that scales linearly with sequence length, making it easy to process documents
    of thousands of tokens or longer. Longformer’s attention mechanism is a drop-in replacement for
    the standard self-attention and combines a local windowed attention with a task motivated global attention.

    Parameters
        input_size:
            input embedding size.
            Equals intermediate and output layer size cause transformer don't change vector dimentions
        num_attention_heads:
            The number of heads in the multiheadattention models
        intermediate_size:
            The dimension of the feedforward network model
        num_hidden_layers:
            The number of sub-encoder-layers in the encoder
        attention_window:
            Size of an attention window around each token
        max_position_embeddings:
            The possible maximum sequence length for positional encoding
        use_positional_encoding (bool):
            Use or not positional encoding
        use_start_random_shift (bool):
            True - starting pos of positional encoding randomly shifted when training
            This allow to train transformer with all range of positional encoding values
            False - starting pos is not shifted.
        is_reduce_sequence (bool):
            False - returns PaddedBatch with all transactions embeddings
            True - returns one embedding for sequence based on CLS token
        add_cls_output (bool):
            False - returns PaddedBatch with all transactions embeddings
            True - returns tuple (PaddedBatch with all transactions embeddings, one embedding for sequence based on CLS token)
    Example:
    >>> model = LongformerEncoder(input_size=32)
    >>> x = PaddedBatch(torch.randn(10, 128, 32), torch.randint(20, 128, (10,)))
    >>> y = model(x)
    >>> assert y.payload.size() == (10, 128, 32)
    >>>
    >>> model = LongformerEncoder(input_size=32, is_reduce_sequence=True)
    >>> y = model(x)
    >>> assert y.size() == (10, 32)

    """

    def __init__(self, input_size, num_attention_heads: int=1, intermediate_size: int=128, num_hidden_layers: int=1, attention_window: int=16, hidden_act='gelu', max_position_embeddings=5000, use_positional_encoding=True, use_start_random_shift=True, is_reduce_sequence=False, add_cls_output=False):
        super().__init__(is_reduce_sequence=is_reduce_sequence)
        self.hidden_size = input_size
        self.max_position_embeddings = max_position_embeddings
        self.use_positional_encoding = use_positional_encoding
        self.use_start_random_shift = use_start_random_shift
        self.token_cls = torch.nn.Parameter(torch.randn(1, 1, input_size), requires_grad=True)
        self.transf = LongformerModel(config=LongformerConfig(hidden_size=input_size, num_attention_heads=num_attention_heads, intermediate_size=intermediate_size, num_hidden_layers=num_hidden_layers, hidden_act=hidden_act, vocab_size=4, max_position_embeddings=max_position_embeddings, attention_window=attention_window), add_pooling_layer=False)
        self.add_cls_output = add_cls_output

    def forward(self, x: PaddedBatch):
        B, T, H = x.payload.size()
        device = x.device
        if self.training and self.use_start_random_shift:
            start_pos = random.randint(0, self.max_position_embeddings - T - 1)
        else:
            start_pos = 0
        inputs_embeds = torch.cat([self.token_cls.expand(B, 1, H), x.payload], dim=1)
        attention_mask = torch.cat([torch.ones(B, 1, device=device), x.seq_len_mask.float()], dim=1)
        if self.use_positional_encoding:
            position_ids = torch.arange(T, device=device, dtype=torch.long).view(1, -1).expand(B, T) + 1 + start_pos
        else:
            position_ids = torch.zeros(B, T, device=device, dtype=torch.long)
        position_ids = torch.cat([torch.zeros(B, 1, device=device, dtype=torch.long), position_ids], dim=1)
        global_attention_mask = torch.cat([torch.ones(B, 1, device=device, dtype=torch.long), torch.zeros(B, T, device=device, dtype=torch.long)], dim=1)
        out = self.transf(inputs_embeds=inputs_embeds, attention_mask=attention_mask, position_ids=position_ids, global_attention_mask=global_attention_mask).last_hidden_state
        if self.is_reduce_sequence:
            return out[:, 0, :]
        elif self.add_cls_output:
            return PaddedBatch(out[:, 1:, :], x.seq_lens), out[:, 0, :]
        else:
            return PaddedBatch(out[:, 1:, :], x.seq_lens)

    @property
    def embedding_size(self):
        return self.hidden_size


class LongformerSeqEncoder(SeqEncoderContainer):
    """SeqEncoderContainer with TransformerEncoder

    Parameters
        trx_encoder:
            TrxEncoder object
        input_size:
            input_size parameter for LongformerEncoder
            If None: input_size = trx_encoder.output_size
            Set input_size explicit or use None if your trx_encoder object has output_size attribute
        **seq_encoder_params:
            params for LongformerEncoder initialisation
        is_reduce_sequence:
            False - returns PaddedBatch with all transactions embeddings
            True - returns one embedding for sequence based on CLS token

    """

    def __init__(self, trx_encoder=None, input_size=None, is_reduce_sequence=True, **seq_encoder_params):
        super().__init__(trx_encoder=trx_encoder, seq_encoder_cls=LongformerEncoder, input_size=input_size, seq_encoder_params=seq_encoder_params, is_reduce_sequence=is_reduce_sequence)


def get_distributions(np_data, tr_amounts_col, tr_types_col=None, negative_items=None, positive_items=None, top_thr=None, take_first_fraction=0, f=lambda x: x):
    set_top_neg_types = set(negative_items[:top_thr])
    set_top_pos_types = set(positive_items[:top_thr])
    sums_of_negative_target = [(0) for _ in range(len(np_data))]
    sums_of_positive_target = [(0) for _ in range(len(np_data))]
    neg_distribution = [[] for _ in range(len(np_data))]
    pos_distribution = [[] for _ in range(len(np_data))]
    for i in range(len(np_data)):
        num = len(np_data[i][tr_amounts_col])
        thr_target_ix = int(num * take_first_fraction)
        amount_target = f(np_data[i][tr_amounts_col][thr_target_ix:])
        tr_type_target = np_data[i][tr_types_col][thr_target_ix:]
        neg_tr_amounts_target = {}
        others_neg_tr_amounts_target = 0
        pos_tr_amounts_target = {}
        others_pos_tr_amounts_target = 0
        for ixx, el in enumerate(tr_type_target):
            if amount_target[ixx] < 0:
                sums_of_negative_target[i] += amount_target[ixx]
            else:
                sums_of_positive_target[i] += amount_target[ixx]
            if el in set_top_neg_types:
                neg_tr_amounts_target[el] = neg_tr_amounts_target.get(el, 0) + amount_target[ixx]
            elif el in set_top_pos_types:
                pos_tr_amounts_target[el] = pos_tr_amounts_target.get(el, 0) + amount_target[ixx]
            elif amount_target[ixx] < 0:
                others_neg_tr_amounts_target += amount_target[ixx]
            elif amount_target[ixx] >= 0:
                others_pos_tr_amounts_target += amount_target[ixx]
            else:
                assert False, 'Should not be!'
        for j in negative_items[:top_thr]:
            if j in neg_tr_amounts_target:
                p_neg = neg_tr_amounts_target[j] / sums_of_negative_target[i]
            else:
                p_neg = 0.0
            neg_distribution[i] += [p_neg]
        if sums_of_negative_target[i] != 0:
            neg_distribution[i] += [others_neg_tr_amounts_target / sums_of_negative_target[i]]
        else:
            neg_distribution[i] += [0.0]
        for j in positive_items[:top_thr]:
            if j in pos_tr_amounts_target:
                p_pos = pos_tr_amounts_target[j] / sums_of_positive_target[i]
            else:
                p_pos = 0.0
            pos_distribution[i] += [p_pos]
        if sums_of_positive_target[i] != 0:
            pos_distribution[i] += [others_pos_tr_amounts_target / sums_of_positive_target[i]]
        else:
            pos_distribution[i] += [0.0]
    return sums_of_negative_target, sums_of_positive_target, neg_distribution, pos_distribution


def transform_inv(x):
    return np.sign(x) * (np.exp(np.abs(x)) - 1)


class StatisticsEncoder(torch.nn.Module):

    def __init__(self, pos=True, neg=True, top_negative_trx=[], top_positive_trx=[], category_names=None, numeric_values=None, category_max_size=None):
        super().__init__()
        self.collect_pos, self.collect_neg = pos, neg
        self.dummy = torch.nn.Linear(1, 1)
        self.negative_items = top_negative_trx
        self.positive_items = top_positive_trx
        self.cat_names = list(category_names)
        self.num_values = list(numeric_values)
        self.cat_max_size = category_max_size

    def forward(self, x: PaddedBatch):
        eps = 1e-07
        tr_type_col = []
        amount_col = []
        for i, row in enumerate(zip(x.payload[self.cat_names[0]], x.payload[self.num_values[0]])):
            tr_type_col += [list(row[0][:x.seq_lens[i].item()].cpu().numpy())]
            amount_col += [list(row[1][:x.seq_lens[i].item()].cpu().numpy())]
        tr_type_col = np.array(tr_type_col, dtype=object)[:, None]
        amount_col = np.array(amount_col, dtype=object)[:, None]
        np_data = np.hstack((tr_type_col, amount_col))
        distr = get_distributions(np_data, 1, 0, self.negative_items, self.positive_items, 5, 0, transform_inv)
        if self.collect_neg:
            sums_of_negative_target = torch.tensor(distr[0])[:, None]
            neg_distribution = torch.tensor(distr[2])
            log_neg_sum = torch.log(torch.abs(sums_of_negative_target + eps))
        if self.collect_pos:
            sums_of_positive_target = torch.tensor(distr[1])[:, None]
            pos_distribution = torch.tensor(distr[3])
            log_pos_sum = torch.log(sums_of_positive_target + eps)
        if self.collect_neg and self.collect_pos:
            return log_neg_sum, neg_distribution, log_pos_sum, pos_distribution
        elif self.collect_neg:
            return log_neg_sum, neg_distribution, 0, 0
        elif self.collect_pos:
            return 0, 0, log_pos_sum, pos_distribution

    @property
    def category_names(self):
        return set(self.cat_names + self.num_values)

    @property
    def category_max_size(self):
        return self.cat_max_size


class PerTransHead(nn.Module):

    def __init__(self, input_size):
        super().__init__()
        self.cnn = torch.nn.Conv1d(input_size, 1, 1)

    def forward(self, x):
        seq_len = x.payload.shape[1]
        feature_vec = torch.transpose(x.payload, 1, 2)
        tx = self.cnn(feature_vec)
        x = tf.avg_pool1d(tx, seq_len)
        x = torch.transpose(x, 1, 2)
        return x.squeeze()


class PerTransTransf(nn.Module):

    def __init__(self, input_size, out_size):
        super().__init__()
        self.cnn = torch.nn.Conv1d(input_size, out_size, 1)

    def forward(self, x):
        feature_vec = torch.transpose(x.payload, 1, 2)
        tx = self.cnn(feature_vec)
        out = torch.transpose(tx, 1, 2)
        return PaddedBatch(out, x.seq_lens)


class ConcatLenEncoder(nn.Module):

    def forward(self, x: PaddedBatch):
        lens = x.seq_lens.unsqueeze(-1).float()
        lens_normed = lens / 200
        h = x.payload[range(len(x.payload)), [(l - 1) for l in x.seq_lens]]
        embeddings = torch.cat([h, lens_normed, -torch.log(lens_normed)], -1)
        return embeddings


class MeanStepEncoder(nn.Module):

    def forward(self, x: PaddedBatch):
        means = torch.stack([e[0:l].mean(dim=0) for e, l in zip(x.payload, x.seq_lens)])
        return means


class PayloadEncoder(nn.Module):

    def forward(self, x: PaddedBatch):
        return x.payload


class AllStepsHead(nn.Module):

    def __init__(self, head):
        super().__init__()
        self.head = head

    def forward(self, x: PaddedBatch):
        out = self.head(x.payload)
        return PaddedBatch(out, x.seq_lens)


class AllStepsMeanHead(nn.Module):

    def __init__(self, head):
        super().__init__()
        self.head = head

    def forward(self, x: PaddedBatch):
        out = self.head(x.payload)
        means = torch.tensor([e[0:l].mean() for e, l in zip(out, x.seq_lens)])
        return means


class FlattenHead(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: PaddedBatch):
        mask = torch.zeros(x.payload.shape, dtype=bool)
        for i, length in enumerate(x.seq_lens):
            mask[i, :length] = True
        return x.payload.flatten()[mask.flatten()]


class TimeStepShuffle(nn.Module):

    def forward(self, x: PaddedBatch):
        shuffled = []
        for seq, slen in zip(x.payload, x.seq_lens):
            idx = torch.randperm(slen) + 1
            pad_idx = torch.arange(slen + 1, len(seq))
            idx = torch.cat([torch.zeros(1, dtype=torch.long), idx, pad_idx])
            shuffled.append(seq[idx])
        shuffled = PaddedBatch(torch.stack(shuffled), x.seq_lens)
        return shuffled


class SkipStepEncoder(nn.Module):

    def __init__(self, step_size):
        super().__init__()
        self.step_size = step_size

    def forward(self, x: PaddedBatch):
        max_len = x.payload.shape[1] - 1
        s = self.step_size
        first_dim_idx = []
        second_dim_idx = []
        for i, l in enumerate(x.seq_lens):
            idx_to_take = np.arange(min(l - 1, s - 1 + l % s), l, s)
            pad_idx = np.array([max_len - 1] * (max_len // s - len(idx_to_take)), dtype=np.int32)
            idx_to_take = np.concatenate([[-1], idx_to_take, pad_idx]) + 1
            first_dim_idx.append(np.ones(len(idx_to_take)) * i)
            second_dim_idx.append(idx_to_take)
        out = x.payload[first_dim_idx, second_dim_idx]
        out_lens = torch.tensor([min(1, l // self.step_size) for l in x.seq_lens])
        return PaddedBatch(out, out_lens)


class RBatchNorm(torch.nn.Module):

    def __init__(self, num_features):
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(num_features)

    def forward(self, v: PaddedBatch):
        x = v.payload
        B, T, H = x.size()
        x = x.view(B * T, H)
        x = self.bn(x)
        x = x.view(B, T, H)
        return PaddedBatch(x, v.seq_lens)


class RBatchNormWithLens(torch.nn.Module):
    """
    The same as RBatchNorm, but ...
    Drop padded symbols (zeros) from batch when batch stat update
    """

    def __init__(self, num_features):
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(num_features)

    def forward(self, v: PaddedBatch):
        x = v.payload
        B, T, H = x.size()
        mask = v.seq_len_mask.bool()
        x_new = x.clone()
        x_new[mask] = self.bn(x[mask])
        return PaddedBatch(x_new, v.seq_lens)


class FloatPositionalEncoding(nn.Module):

    def __init__(self, out_size):
        super(FloatPositionalEncoding, self).__init__()
        self.out_size = out_size

    def forward(self, position):
        """

        :param position: B x T
        :return: B x T x H
        """
        div_term = torch.exp(torch.arange(0, self.out_size, 2).float() * (-math.log(10000.0) / self.out_size))
        div_term = div_term.unsqueeze(0).unsqueeze(0)
        div_term = div_term
        position = position.unsqueeze(2)
        pe = torch.cat([torch.sin(position * div_term), torch.cos(position * div_term)], dim=2)
        self.register_buffer('pe', pe)
        return pe


class NoisyEmbedding(nn.Embedding):
    """
    Embeddings with additive gaussian noise with mean=0 and user-defined variance.
    *args and **kwargs defined by usual Embeddings
    Args:
        noise_scale (float): when > 0 applies additive noise to embeddings.
            When = 0, forward is equivalent to usual embeddings.
        dropout (float): probability of embedding axis to be dropped. 0 means no dropout at all.
        spatial_dropout (bool): whether to dropout full dimension of embedding in the whole sequence.

    For other parameters defenition look at nn.Embedding help
    """

    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None, noise_scale=0, dropout=0, spatial_dropout=False):
        if max_norm is not None:
            raise AttributeError("Please don't use embedding normalisation. https://github.com/pytorch/pytorch/issues/44792")
        super().__init__(num_embeddings, embedding_dim, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse, _weight)
        self.noise = torch.distributions.Normal(0, noise_scale) if noise_scale > 0 else None
        self.scale = noise_scale
        self.spatial_dropout = spatial_dropout
        self.dropout = nn.Dropout2d(dropout) if spatial_dropout else nn.Dropout(dropout)
        _ = super().forward(torch.arange(num_embeddings))

    def forward(self, x):
        x = super().forward(x)
        if self.spatial_dropout:
            x = self.dropout(x.permute(0, 2, 1).unsqueeze(3)).squeeze(3).permute(0, 2, 1)
        else:
            x = self.dropout(x)
        if self.training and self.scale > 0:
            x += self.noise.sample((self.weight.shape[1],))
        return x


class BaseScaler(nn.Module):

    def __init__(self, col_name=None):
        super().__init__()
        self.col_name = col_name

    @property
    def output_size(self):
        raise NotImplementedError()


class IdentityScaler(BaseScaler):

    def forward(self, x):
        return x

    @property
    def output_size(self):
        return 1


class LogScaler(BaseScaler):

    def forward(self, x):
        return x.abs().log1p() * x.sign()

    @property
    def output_size(self):
        return 1


class YearScaler(BaseScaler):

    def forward(self, x):
        return x / 365

    @property
    def output_size(self):
        return 1


class NumToVector(BaseScaler):

    def __init__(self, embeddings_size):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(1, 1, embeddings_size), requires_grad=True)
        self.b = torch.nn.Parameter(torch.randn(1, 1, embeddings_size), requires_grad=True)

    def forward(self, x):
        return x * self.w + self.b

    @property
    def output_size(self):
        return self.w.size(2)


class LogNumToVector(BaseScaler):

    def __init__(self, embeddings_size):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(1, 1, embeddings_size), requires_grad=True)
        self.b = torch.nn.Parameter(torch.randn(1, 1, embeddings_size), requires_grad=True)

    def forward(self, x):
        return x.abs().log1p() * x.sign() * self.w + self.b

    @property
    def output_size(self):
        return self.w.size(2)


class PoissonScaler(BaseScaler):
    """
    Explicit estimator for poissonian target with standard pytorch sampler extrapolation.
    """

    def __init__(self, kmax=33):
        super().__init__()
        self.kmax = 0.7 * kmax
        self.arange = torch.nn.Parameter(torch.arange(kmax), requires_grad=False)
        self.factor = torch.nn.Parameter(torch.special.gammaln(1 + self.arange), requires_grad=False)

    def forward(self, x):
        if self.kmax == 0:
            return torch.poisson(x)
        res = self.arange * torch.log(x).unsqueeze(-1) - self.factor * torch.ones_like(x).unsqueeze(-1)
        return res.argmax(dim=-1).float().where(x < self.kmax, torch.poisson(x))

    @property
    def output_size(self):
        return 1


class ExpScaler(BaseScaler):

    def __init__(self, column=0):
        super().__init__()
        self.column = column

    def forward(self, x):
        if self.column is not None:
            return torch.exp(x if x.dim() == 1 else x[:, self.column].unsqueeze(-1))
        else:
            return torch.exp(x)

    @property
    def output_size(self):
        return 1


class TabFormerFeatureEncoder(nn.Module):
    """TabFormerFeatureEncoder: encodes input batch of shape (B, T, F, E),
           where:
               B - batch size,
               T - sequence length,
               F - number of features
               E - embedding dimension for each feature
       and returns output batch of same shape.

       Encoding is performed as in [Tabular Transformers for Modeling Multivariate Time Series](https://arxiv.org/abs/2011.01843)

       Parameters
       ----------
       n_cols: number of features to encode,
       emb_dim: feature embedding dimension,
       n_heads: number of heads in transformer,
       n_layers: number of layers in transformer,
       out_hidden: out hidden dimension for each feature
    """

    def __init__(self, n_cols: int, emb_dim: int, transf_feedforward_dim: int=64, n_heads: int=8, n_layers: int=1, out_hidden: int=None):
        super().__init__()
        out_hidden = out_hidden if out_hidden else emb_dim * n_cols
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=n_heads, dim_feedforward=transf_feedforward_dim, batch_first=True, activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.lin_proj = nn.Linear(emb_dim * n_cols, out_hidden)

    def forward(self, input_embeds):
        embeds_shape = list(input_embeds.size())
        input_embeds = input_embeds.view([-1] + embeds_shape[-2:])
        out_embeds = self.transformer_encoder(input_embeds)
        out_embeds = out_embeds.contiguous().view(embeds_shape[0:2] + [-1])
        out_embeds = self.lin_proj(out_embeds)
        return out_embeds


def scaler_by_name(name):
    scaler = {'identity': IdentityScaler, 'sigmoid': torch.nn.Sigmoid, 'log': LogScaler, 'year': YearScaler}.get(name, None)
    if scaler is None:
        raise Exception(f'unknown scaler name: {name}')
    else:
        return scaler()


class TrxEncoderBase(nn.Module):
    """Base class for TrxEncoders.

    This provides a parts of embeddings for feature fields.
    This doesn't provide a full embedding of transaction.

    Parameters
    ----------
    embeddings:
        dict with categorical feature names.
        Values must be like this `{'in': dictionary_size, 'out': embedding_size}`
        These features will be encoded with lookup embedding table of shape (dictionary_size, embedding_size)
        Values can be a `torch.nn.Embedding` implementation
    numeric_values:
        dict with numerical feature names.
        Values must be a string with scaler_name.
        Possible values are: 'identity', 'sigmoid', 'log', 'year'.
        These features will be scaled with selected scaler.
        Values can be `ptls.nn.trx_encoder.scalers.BaseScaler` implementation

        One field can have many scalers. In this case key become alias and col name should be in scaler.
        Example:
        >>> TrxEncoderBase(
        >>>     numeric_values={
        >>>         'amount_orig': IdentityScaler(col_name='amount'),
        >>>         'amount_log': LogScaler(col_name='amount'),
        >>>     },
        >>> )

    out_of_index:
        How to process a categorical indexes which are greater than dictionary size.
        'clip' - values will be collapsed to maximum index. This works well for frequency encoded categories.
            We join infrequent categories to one.
        'assert' - raise an error of invalid index appear.
    """

    def __init__(self, embeddings: Dict[str, Union[Dict, torch.nn.Embedding]]=None, numeric_values: Dict[str, Union[str, BaseScaler]]=None, out_of_index: str='clip'):
        super().__init__()
        if embeddings is None:
            embeddings = {}
        if numeric_values is None:
            numeric_values = {}
        self.embeddings = torch.nn.ModuleDict()
        for col_name, emb_props in embeddings.items():
            if type(emb_props) is dict:
                if emb_props.get('disabled', False):
                    continue
                if emb_props['in'] == 0 or emb_props['out'] == 0:
                    continue
                if emb_props['in'] < 3:
                    raise AttributeError(f'At least 3 should be in `embeddings.{col_name}.in`. 0-padding and at least two different embedding indexes')
                self.embeddings[col_name] = torch.nn.Embedding(num_embeddings=emb_props['in'], embedding_dim=emb_props['out'], padding_idx=0)
            elif isinstance(emb_props, torch.nn.Embedding):
                self.embeddings[col_name] = emb_props
            else:
                raise AttributeError(f'Wrong type of embeddings, found {type(col_name)} for "{col_name}"')
        self.out_of_index = out_of_index
        assert out_of_index in ('clip', 'assert')
        self.numeric_values = torch.nn.ModuleDict()
        for col_name, scaler_name in numeric_values.items():
            if type(scaler_name) is str:
                if scaler_name == 'none':
                    continue
                self.numeric_values[col_name] = scaler_by_name(scaler_name)
            elif isinstance(scaler_name, BaseScaler):
                self.numeric_values[col_name] = scaler_name
            else:
                raise AttributeError(f'Wrong type of numeric_values, found {type(scaler_name)} for "{col_name}"')

    def get_category_indexes(self, x: PaddedBatch, col_name: str):
        """Returns category feature values clipped to dictionary size.

        Parameters
        ----------
        x: PaddedBatch with feature dict. Each value is `(B, T)` size
        col_name: required feature name
        """
        v = x.payload[col_name].long()
        max_size = self.embeddings[col_name].num_embeddings
        if self.out_of_index == 'clip':
            return v.clip(0, max_size - 1)
        if self.out_of_index == 'assert':
            out_of_index_cnt = (v >= max_size).sum()
            if out_of_index_cnt > 0:
                raise IndexError(f'Found indexes greater than dictionary size for "{col_name}"')
            return v
        raise AssertionError(f'Unknown out_of_index value: {self.out_of_index}')

    def get_category_embeddings(self, x: PaddedBatch, col_name: str):
        indexes = self.get_category_indexes(x, col_name)
        return self.embeddings[col_name](indexes)

    def get_numeric_scaled(self, x: PaddedBatch, col_name):
        """Returns numerical feature values transformed with selected scaler.

        Parameters
        ----------
        x: PaddedBatch with feature dict. Each value is `(B, T)` size
        col_name: required feature name
        """
        scaler = self.numeric_values[col_name]
        if scaler.col_name is None:
            v = x.payload[col_name].unsqueeze(2).float()
        else:
            v = x.payload[scaler.col_name].unsqueeze(2).float()
        return scaler(v)

    @property
    def numerical_size(self):
        return sum(n.output_size for n in self.numeric_values.values())

    @property
    def embedding_size(self):
        return sum(e.embedding_dim for e in self.embeddings.values())

    @property
    def output_size(self):
        s = self.numerical_size + self.embedding_size
        return s

    @property
    def category_names(self):
        """Returns set of used feature names
        """
        return set([field_name for field_name in self.embeddings.keys()] + [value_name for value_name in self.numeric_values.keys()])

    @property
    def category_max_size(self):
        """Returns dict with categorical feature names. Value is dictionary size
        """
        return {k: v['in'] for k, v in self.embeddings.items()}


class TrxMeanEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.scalers = OrderedDict({name: scaler_by_name(scaler_name) for name, scaler_name in config['numeric_values'].items()})
        self.embeddings = nn.ModuleDict()
        for name, dim in config['embeddings'].items():
            dict_len = dim['in']
            self.embeddings[name] = nn.EmbeddingBag(dict_len, dict_len, mode='mean')
            self.embeddings[name].weight = nn.Parameter(torch.diag(torch.ones(dict_len)).float())

    def forward(self, x: PaddedBatch):
        processed = []
        for field_name in self.embeddings.keys():
            processed.append(self.embeddings[field_name](x.payload[field_name]).detach())
        for value_name, scaler in self.scalers.items():
            var = scaler(x.payload[value_name].unsqueeze(-1).float())
            means = torch.tensor([e[:l].mean() for e, l in zip(var, x.seq_lens)]).unsqueeze(-1)
            processed.append(means)
        out = torch.cat(processed, -1)
        return out

    @staticmethod
    def output_size(config):
        sz = len(config['numeric_values'])
        sz += sum(dim['in'] for _, dim in config['embeddings'].items())
        return sz


class TrxEncoderTest(torch.nn.Module):

    def forward(self, x):
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BCELoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (BinarizationLayer,
     lambda: ([], {'hs_from': 4, 'hs_to': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CatLayer,
     lambda: ([], {'left_tail': _mock_layer(), 'right_tail': _mock_layer()}),
     lambda: ([(torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]))], {}),
     False),
    (CentroidLoss,
     lambda: ([], {'class_num': 4}),
     lambda: ([torch.rand([4, 4]), torch.ones([4, 4], dtype=torch.int64)], {}),
     True),
    (CentroidSoftmaxLoss,
     lambda: ([], {'class_num': 4}),
     lambda: ([torch.rand([4, 4]), torch.ones([4, 4], dtype=torch.int64)], {}),
     True),
    (CentroidSoftmaxMemoryLoss,
     lambda: ([], {'class_num': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.ones([4, 4, 4], dtype=torch.int64)], {}),
     True),
    (ComplexLoss,
     lambda: ([], {'ml_loss': MSELoss(), 'aug_loss': MSELoss()}),
     lambda: ([(torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])), torch.rand([4, 4])], {}),
     False),
    (DropoutEncoder,
     lambda: ([], {'p': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (DummyHead,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ExpScaler,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FloatPositionalEncoding,
     lambda: ([], {'out_size': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (GEGLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Head,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (IdentityScaler,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (L2NormEncoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LogNumToVector,
     lambda: ([], {'embeddings_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LogScaler,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MSELoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (ModelEmbeddingEnsemble,
     lambda: ([], {'submodels': [_mock_layer()]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (NumToVector,
     lambda: ([], {'embeddings_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PairwiseMarginRankingLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (PoissonScaler,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PositionalEncoding,
     lambda: ([], {'d_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (QuerySoftmaxLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 8, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (SequencePredictionHead,
     lambda: ([], {'embeds_dim': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (SoftmaxLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
    (Squeeze,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TabularRowEncoder,
     lambda: ([], {'input_dim': 4, 'cat_dims': [4, 4], 'cat_idxs': [4, 4], 'cat_emb_dim': 4}),
     lambda: ([torch.rand([4, 5, 4, 4])], {}),
     False),
    (TransactionSumLoss,
     lambda: ([], {'n_variables_to_predict': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (TrxEncoderTest,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (UnsupervisedTabNetLoss,
     lambda: ([], {}),
     lambda: ([(torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])), torch.rand([4, 4])], {}),
     False),
    (YearScaler,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_dllllb_pytorch_lifestream(_paritybench_base):
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

    def test_005(self):
        self._check(*TESTCASES[5])

    def test_006(self):
        self._check(*TESTCASES[6])

    def test_007(self):
        self._check(*TESTCASES[7])

    def test_008(self):
        self._check(*TESTCASES[8])

    def test_009(self):
        self._check(*TESTCASES[9])

    def test_010(self):
        self._check(*TESTCASES[10])

    def test_011(self):
        self._check(*TESTCASES[11])

    def test_012(self):
        self._check(*TESTCASES[12])

    def test_013(self):
        self._check(*TESTCASES[13])

    def test_014(self):
        self._check(*TESTCASES[14])

    def test_015(self):
        self._check(*TESTCASES[15])

    def test_016(self):
        self._check(*TESTCASES[16])

    def test_017(self):
        self._check(*TESTCASES[17])

    def test_018(self):
        self._check(*TESTCASES[18])

    def test_019(self):
        self._check(*TESTCASES[19])

    def test_020(self):
        self._check(*TESTCASES[20])

    def test_021(self):
        self._check(*TESTCASES[21])

    def test_022(self):
        self._check(*TESTCASES[22])

    def test_023(self):
        self._check(*TESTCASES[23])

    def test_024(self):
        self._check(*TESTCASES[24])

    def test_025(self):
        self._check(*TESTCASES[25])

    def test_026(self):
        self._check(*TESTCASES[26])

    def test_027(self):
        self._check(*TESTCASES[27])

    def test_028(self):
        self._check(*TESTCASES[28])

    def test_029(self):
        self._check(*TESTCASES[29])

    def test_030(self):
        self._check(*TESTCASES[30])

    def test_031(self):
        self._check(*TESTCASES[31])

