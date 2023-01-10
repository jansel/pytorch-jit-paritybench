import sys
_module = sys.modules[__name__]
del sys
setup = _module
pytorch_metric_learning = _module
distances = _module
base_distance = _module
batched_distance = _module
cosine_similarity = _module
dot_product_similarity = _module
lp_distance = _module
snr_distance = _module
losses = _module
angular_loss = _module
arcface_loss = _module
base_metric_loss_function = _module
centroid_triplet_loss = _module
circle_loss = _module
contrastive_loss = _module
cosface_loss = _module
cross_batch_memory = _module
fast_ap_loss = _module
generic_pair_loss = _module
instance_loss = _module
intra_pair_variance_loss = _module
large_margin_softmax_loss = _module
lifted_structure_loss = _module
margin_loss = _module
mixins = _module
multi_similarity_loss = _module
n_pairs_loss = _module
nca_loss = _module
normalized_softmax_loss = _module
ntxent_loss = _module
proxy_anchor_loss = _module
proxy_losses = _module
signal_to_noise_ratio_losses = _module
soft_triple_loss = _module
sphereface_loss = _module
subcenter_arcface_loss = _module
supcon_loss = _module
triplet_margin_loss = _module
tuplet_margin_loss = _module
vicreg_loss = _module
miners = _module
angular_miner = _module
base_miner = _module
batch_easy_hard_miner = _module
batch_hard_miner = _module
distance_weighted_miner = _module
embeddings_already_packaged_as_triplets = _module
hdc_miner = _module
maximum_loss_miner = _module
multi_similarity_miner = _module
pair_margin_miner = _module
triplet_margin_miner = _module
uniform_histogram_miner = _module
reducers = _module
avg_non_zero_reducer = _module
base_reducer = _module
class_weighted_reducer = _module
divisor_reducer = _module
do_nothing_reducer = _module
mean_reducer = _module
multiple_reducers = _module
per_anchor_reducer = _module
threshold_reducer = _module
regularizers = _module
base_regularizer = _module
center_invariant_regularizer = _module
lp_regularizer = _module
regular_face_regularizer = _module
sparse_centers_regularizer = _module
zero_mean_regularizer = _module
samplers = _module
fixed_set_of_triplets = _module
hierarchical_sampler = _module
m_per_class_sampler = _module
tuples_to_weights_sampler = _module
testers = _module
base_tester = _module
global_embedding_space = _module
global_twostream_embedding_space = _module
with_same_parent_label = _module
trainers = _module
base_trainer = _module
cascaded_embeddings = _module
deep_adversarial_metric_learning = _module
metric_loss_only = _module
train_with_classifier = _module
twostream_metric_loss = _module
unsupervised_embeddings_using_augmentations = _module
utils = _module
accuracy_calculator = _module
common_functions = _module
distributed = _module
inference = _module
key_checker = _module
logging_presets = _module
loss_and_miner_utils = _module
loss_tracker = _module
module_with_records = _module
module_with_records_and_reducer = _module
tests = _module
test_batched_distance = _module
test_angular_loss = _module
test_arcface_loss = _module
test_centroid_triplet_loss = _module
test_circle_loss = _module
test_contrastive_loss = _module
test_cosface_loss = _module
test_cross_batch_memory = _module
test_fastap_loss = _module
test_instance_loss = _module
test_intra_pair_variance_loss = _module
test_large_margin_softmax_loss = _module
test_lifted_structure_loss = _module
test_losses_without_labels = _module
test_margin_loss = _module
test_multi_similarity_loss = _module
test_multiple_losses = _module
test_nca_loss = _module
test_normalized_softmax_loss = _module
test_npairs_loss = _module
test_ntxent_loss = _module
test_proxy_anchor_loss = _module
test_proxy_nca_loss = _module
test_signal_to_noise_ratio_losses = _module
test_soft_triple_loss = _module
test_subcenter_arcface_loss = _module
test_triplet_margin_loss = _module
test_tuplet_margin_loss = _module
test_vicreg_loss = _module
utils = _module
test_angular_miner = _module
test_batch_easy_hard_miner = _module
test_batch_easy_hard_miner_labels = _module
test_batch_hard_miner = _module
test_distance_weighted_miner = _module
test_hdc_miner = _module
test_multi_similarity_miner = _module
test_pair_margin_miner = _module
test_triplet_margin_miner = _module
test_uniform_histogram_miner = _module
test_avg_non_zero_reducer = _module
test_class_weighted_reducer = _module
test_divisor_reducer = _module
test_do_nothing_reducer = _module
test_mean_reducer = _module
test_multiple_reducers = _module
test_per_anchor_reducer = _module
test_setting_reducers = _module
test_threshold_reducer = _module
test_center_invariant_regularizer = _module
test_regular_face_regularizer = _module
test_fixed_set_of_triplets = _module
test_hierarchical_sampler = _module
test_m_per_class_sampler = _module
test_tuples_to_weights_sampler = _module
test_global_embedding_space_tester = _module
test_global_two_stream_embedding_space_tester = _module
test_with_same_parent_label_tester = _module
test_key_checking = _module
test_metric_loss_only = _module
test_calculate_accuracies = _module
test_calculate_accuracies_large_k = _module
test_common_functions = _module
test_distributed = _module
test_inference = _module
test_loss_and_miner_utils = _module
test_loss_tracker = _module
test_module_with_records_and_reducer = _module
zzz_testing_utils = _module
testing_utils = _module

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


import numpy as np


import inspect


from collections import defaultdict


import torch.nn.functional as F


import math


import scipy.special


from torch.utils.data.sampler import Sampler


import itertools


from torch.utils.data.sampler import BatchSampler


import copy


from sklearn.metrics import adjusted_mutual_info_score


from sklearn.metrics import normalized_mutual_info_score


import collections


import logging


import re


import scipy.stats


from itertools import chain


from torch.autograd import Variable


from torch import Tensor


from torch import nn


import scipy


import torch.nn as nn


from torch.nn import init


from torch.nn.parameter import Parameter


from collections import Counter


from torchvision import datasets


from torchvision import models


from torchvision import transforms


from sklearn.preprocessing import StandardScaler


import torch.distributed as dist


import torch.multiprocessing as mp


import torch.optim as optim


from torch.nn.parallel import DistributedDataParallel as DDP


import uuid


import torchvision


class BatchedDistance(torch.nn.Module):

    def __init__(self, distance, iter_fn=None, batch_size=32):
        super().__init__()
        self.distance = distance
        self.iter_fn = iter_fn
        self.batch_size = batch_size

    def forward(self, query_emb, ref_emb=None):
        ref_emb = ref_emb if ref_emb is not None else query_emb
        n = query_emb.shape[0]
        for s in range(0, n, self.batch_size):
            e = s + self.batch_size
            L = query_emb[s:e]
            mat = self.distance(L, ref_emb)
            self.iter_fn(mat, s, e)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.distance, name)


class MultipleLosses(torch.nn.Module):

    def __init__(self, losses, miners=None, weights=None):
        super().__init__()
        self.is_dict = isinstance(losses, dict)
        self.losses = torch.nn.ModuleDict(losses) if self.is_dict else torch.nn.ModuleList(losses)
        if miners is not None:
            self.assertions_if_not_none(miners, match_all_keys=False)
            self.miners = torch.nn.ModuleDict(miners) if self.is_dict else torch.nn.ModuleList(miners)
        else:
            self.miners = None
        if weights is not None:
            self.assertions_if_not_none(weights, match_all_keys=True)
            self.weights = weights
        else:
            self.weights = {k: (1) for k in self.losses.keys()} if self.is_dict else [1] * len(losses)

    def forward(self, embeddings, labels, indices_tuple=None):
        if self.miners:
            assert indices_tuple is None
        total_loss = 0
        iterable = self.losses.items() if self.is_dict else enumerate(self.losses)
        for i, loss_func in iterable:
            curr_indices_tuple = self.get_indices_tuple(i, embeddings, labels, indices_tuple)
            total_loss += loss_func(embeddings, labels, curr_indices_tuple) * self.weights[i]
        return total_loss

    def get_indices_tuple(self, i, embeddings, labels, indices_tuple):
        if self.miners:
            if self.is_dict and i in self.miners or not self.is_dict and self.miners[i] is not None:
                indices_tuple = self.miners[i](embeddings, labels)
        return indices_tuple

    def assertions_if_not_none(self, x, match_all_keys):
        if x is not None:
            if self.is_dict:
                assert isinstance(x, dict)
                if match_all_keys:
                    assert sorted(list(x.keys())) == sorted(list(self.losses.keys()))
                else:
                    assert all(k in self.losses.keys() for k in x.keys())
            else:
                assert c_f.is_list_or_tuple(x)
                assert len(x) == len(self.losses)


class Identity(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class EmbeddingRegularizerMixin:

    def __init__(self, embedding_regularizer=None, embedding_reg_weight=1, **kwargs):
        self.embedding_regularizer = embedding_regularizer is not None
        super().__init__(**kwargs)
        self.embedding_regularizer = embedding_regularizer
        self.embedding_reg_weight = embedding_reg_weight
        if self.embedding_regularizer is not None:
            self.add_to_recordable_attributes(list_of_names=['embedding_reg_weight'], is_stat=False)

    def embedding_regularization_loss(self, embeddings):
        if self.embedding_regularizer is None:
            loss = 0
        else:
            loss = self.embedding_regularizer(embeddings) * self.embedding_reg_weight
        return {'losses': loss, 'indices': None, 'reduction_type': 'already_reduced'}

    def add_embedding_regularization_to_loss_dict(self, loss_dict, embeddings):
        if self.embedding_regularizer is not None:
            loss_dict['embedding_reg_loss'] = self.embedding_regularization_loss(embeddings)

    def regularization_loss_names(self):
        return ['embedding_reg_loss']


class ModuleWithRecords(torch.nn.Module):

    def __init__(self, collect_stats=None):
        super().__init__()
        self.collect_stats = c_f.COLLECT_STATS if collect_stats is None else collect_stats

    def add_to_recordable_attributes(self, name=None, list_of_names=None, is_stat=False):
        if is_stat and not self.collect_stats:
            pass
        else:
            c_f.add_to_recordable_attributes(self, name=name, list_of_names=list_of_names, is_stat=is_stat)

    def reset_stats(self):
        c_f.reset_stats(self)


class BaseDistance(ModuleWithRecords):

    def __init__(self, normalize_embeddings=True, p=2, power=1, is_inverted=False, **kwargs):
        super().__init__(**kwargs)
        self.normalize_embeddings = normalize_embeddings
        self.p = p
        self.power = power
        self.is_inverted = is_inverted
        self.add_to_recordable_attributes(list_of_names=['p', 'power'], is_stat=False)

    def forward(self, query_emb, ref_emb=None):
        self.reset_stats()
        query_emb_normalized = self.maybe_normalize(query_emb)
        if ref_emb is None:
            ref_emb = query_emb
            ref_emb_normalized = query_emb_normalized
        else:
            ref_emb_normalized = self.maybe_normalize(ref_emb)
        self.set_default_stats(query_emb, ref_emb, query_emb_normalized, ref_emb_normalized)
        mat = self.compute_mat(query_emb_normalized, ref_emb_normalized)
        if self.power != 1:
            mat = mat ** self.power
        assert mat.size() == torch.Size((query_emb.size(0), ref_emb.size(0)))
        return mat

    def compute_mat(self, query_emb, ref_emb):
        raise NotImplementedError

    def pairwise_distance(self, query_emb, ref_emb):
        raise NotImplementedError

    def smallest_dist(self, *args, **kwargs):
        if self.is_inverted:
            return torch.max(*args, **kwargs)
        return torch.min(*args, **kwargs)

    def largest_dist(self, *args, **kwargs):
        if self.is_inverted:
            return torch.min(*args, **kwargs)
        return torch.max(*args, **kwargs)

    def margin(self, x, y):
        if self.is_inverted:
            return y - x
        return x - y

    def normalize(self, embeddings, dim=1, **kwargs):
        return torch.nn.functional.normalize(embeddings, p=self.p, dim=dim, **kwargs)

    def maybe_normalize(self, embeddings, dim=1, **kwargs):
        if self.normalize_embeddings:
            return self.normalize(embeddings, dim=dim, **kwargs)
        return embeddings

    def get_norm(self, embeddings, dim=1, **kwargs):
        return torch.norm(embeddings, p=self.p, dim=dim, **kwargs)

    def set_default_stats(self, query_emb, ref_emb, query_emb_normalized, ref_emb_normalized):
        if self.collect_stats:
            with torch.no_grad():
                stats_dict = {'initial_avg_query_norm': torch.mean(self.get_norm(query_emb)).item(), 'initial_avg_ref_norm': torch.mean(self.get_norm(ref_emb)).item(), 'final_avg_query_norm': torch.mean(self.get_norm(query_emb_normalized)).item(), 'final_avg_ref_norm': torch.mean(self.get_norm(ref_emb_normalized)).item()}
                self.set_stats(stats_dict)

    def set_stats(self, stats_dict):
        for k, v in stats_dict.items():
            self.add_to_recordable_attributes(name=k, is_stat=True)
            setattr(self, k, v)


class LpDistance(BaseDistance):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert not self.is_inverted

    def compute_mat(self, query_emb, ref_emb):
        dtype, device = query_emb.dtype, query_emb.device
        if ref_emb is None:
            ref_emb = query_emb
        if dtype == torch.float16:
            rows, cols = lmu.meshgrid_from_sizes(query_emb, ref_emb, dim=0)
            output = torch.zeros(rows.size(), dtype=dtype, device=device)
            rows, cols = rows.flatten(), cols.flatten()
            distances = self.pairwise_distance(query_emb[rows], ref_emb[cols])
            output[rows, cols] = distances
            return output
        else:
            return torch.cdist(query_emb, ref_emb, p=self.p)

    def pairwise_distance(self, query_emb, ref_emb):
        return torch.nn.functional.pairwise_distance(query_emb, ref_emb, p=self.p)


class ModuleWithRecordsAndDistance(ModuleWithRecords):

    def __init__(self, distance=None, **kwargs):
        super().__init__(**kwargs)
        self.distance = self.get_default_distance() if distance is None else distance

    def get_default_distance(self):
        return LpDistance(p=2)


class BaseReducer(ModuleWithRecords):

    def forward(self, loss_dict, embeddings, labels):
        self.reset_stats()
        assert len(loss_dict) == 1
        loss_name = list(loss_dict.keys())[0]
        loss_info = loss_dict[loss_name]
        self.add_to_recordable_attributes(name=loss_name, is_stat=True)
        losses, loss_indices, reduction_type, kwargs = self.unpack_loss_info(loss_info)
        loss_val = self.reduce_the_loss(losses, loss_indices, reduction_type, kwargs, embeddings, labels)
        setattr(self, loss_name, loss_val.item())
        return loss_val

    def unpack_loss_info(self, loss_info):
        return loss_info['losses'], loss_info['indices'], loss_info['reduction_type'], {}

    def reduce_the_loss(self, losses, loss_indices, reduction_type, kwargs, embeddings, labels):
        self.set_losses_size_stat(losses)
        if self.input_is_zero_loss(losses):
            return self.zero_loss(embeddings)
        self.assert_sizes(losses, loss_indices, reduction_type)
        reduction_func = self.get_reduction_func(reduction_type)
        return reduction_func(losses, loss_indices, embeddings, labels, **kwargs)

    def already_reduced_reduction(self, losses, loss_indices, embeddings, labels):
        assert losses.ndim == 0 or len(losses) == 1
        return losses

    def element_reduction(self, losses, loss_indices, embeddings, labels):
        raise NotImplementedError

    def pos_pair_reduction(self, losses, loss_indices, embeddings, labels):
        raise NotImplementedError

    def neg_pair_reduction(self, losses, loss_indices, embeddings, labels):
        raise NotImplementedError

    def triplet_reduction(self, losses, loss_indices, embeddings, labels):
        raise NotImplementedError

    def get_reduction_func(self, reduction_type):
        return getattr(self, '{}_reduction'.format(reduction_type))

    def assert_sizes(self, losses, loss_indices, reduction_type):
        getattr(self, 'assert_sizes_{}'.format(reduction_type))(losses, loss_indices)

    def zero_loss(self, embeddings):
        return torch.sum(embeddings * 0)

    def input_is_zero_loss(self, losses):
        if not torch.is_tensor(losses) and losses == 0:
            return True
        return False

    def assert_sizes_already_reduced(self, losses, loss_indices):
        pass

    def assert_sizes_element(self, losses, loss_indices):
        assert torch.is_tensor(losses)
        assert torch.is_tensor(loss_indices)
        assert len(losses) == len(loss_indices)

    def assert_sizes_pair(self, losses, loss_indices):
        assert torch.is_tensor(losses)
        assert c_f.is_list_or_tuple(loss_indices)
        assert len(loss_indices) == 2
        assert all(torch.is_tensor(x) for x in loss_indices)
        assert len(losses) == len(loss_indices[0]) == len(loss_indices[1])

    def assert_sizes_pos_pair(self, losses, loss_indices):
        self.assert_sizes_pair(losses, loss_indices)

    def assert_sizes_neg_pair(self, losses, loss_indices):
        self.assert_sizes_pair(losses, loss_indices)

    def assert_sizes_triplet(self, losses, loss_indices):
        assert torch.is_tensor(losses)
        assert c_f.is_list_or_tuple(loss_indices)
        assert len(loss_indices) == 3
        assert all(len(x) == len(losses) for x in loss_indices)

    def set_losses_size_stat(self, losses):
        if self.collect_stats:
            self.add_to_recordable_attributes(name='losses_size', is_stat=True)
            if not torch.is_tensor(losses) or losses.ndim == 0:
                self.losses_size = 1
            else:
                self.losses_size = len(losses)


class DoNothingReducer(BaseReducer):

    def forward(self, loss_dict, embeddings, labels):
        return loss_dict


class MeanReducer(BaseReducer):

    def element_reduction(self, losses, *_):
        return torch.mean(losses)

    def pos_pair_reduction(self, losses, *args):
        return self.element_reduction(losses, *args)

    def neg_pair_reduction(self, losses, *args):
        return self.element_reduction(losses, *args)

    def triplet_reduction(self, losses, *args):
        return self.element_reduction(losses, *args)


class MultipleReducers(BaseReducer):

    def __init__(self, reducers, default_reducer=None, **kwargs):
        super().__init__(**kwargs)
        self.reducers = torch.nn.ModuleDict(reducers)
        self.default_reducer = MeanReducer() if default_reducer is None else default_reducer

    def forward(self, loss_dict, embeddings, labels):
        self.reset_stats()
        sub_losses = torch.zeros(len(loss_dict), dtype=embeddings.dtype, device=embeddings.device)
        loss_count = 0
        for loss_name, loss_info in loss_dict.items():
            input_dict = {loss_name: loss_info}
            if loss_name in self.reducers:
                loss_val = self.reducers[loss_name](input_dict, embeddings, labels)
            else:
                loss_val = self.default_reducer(input_dict, embeddings, labels)
            sub_losses[loss_count] = loss_val
            loss_count += 1
        return self.sub_loss_reduction(sub_losses, embeddings, labels)

    def sub_loss_reduction(self, sub_losses, embeddings=None, labels=None):
        return torch.sum(sub_losses)


class ModuleWithRecordsAndReducer(ModuleWithRecords):

    def __init__(self, reducer=None, **kwargs):
        super().__init__(**kwargs)
        self.set_reducer(reducer)

    def get_default_reducer(self):
        return MeanReducer()

    def set_reducer(self, reducer):
        if isinstance(reducer, (MultipleReducers, DoNothingReducer)):
            self.reducer = reducer
        elif len(self.sub_loss_names()) == 1:
            self.reducer = self.get_default_reducer() if reducer is None else copy.deepcopy(reducer)
        else:
            reducer_dict = {}
            for k in self.sub_loss_names():
                reducer_dict[k] = self.get_default_reducer() if reducer is None else copy.deepcopy(reducer)
            self.reducer = MultipleReducers(reducer_dict)

    def sub_loss_names(self):
        return ['loss']


class ModuleWithRecordsReducerAndDistance(ModuleWithRecordsAndReducer, ModuleWithRecordsAndDistance):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class BaseMetricLossFunction(EmbeddingRegularizerMixin, ModuleWithRecordsReducerAndDistance):

    def compute_loss(self, embeddings, labels, indices_tuple, ref_emb, ref_labels):
        """
        This has to be implemented and is what actually computes the loss.
        """
        raise NotImplementedError

    def forward(self, embeddings, labels=None, indices_tuple=None, ref_emb=None, ref_labels=None):
        """
        Args:
            embeddings: tensor of size (batch_size, embedding_size)
            labels: tensor of size (batch_size)
            indices_tuple: tuple of size 3 for triplets (anchors, positives, negatives)
                            or size 4 for pairs (anchor1, postives, anchor2, negatives)
                            Can also be left as None
        Returns: the loss
        """
        self.reset_stats()
        c_f.check_shapes(embeddings, labels)
        if labels is not None:
            labels = c_f.to_device(labels, embeddings)
        ref_emb, ref_labels = c_f.set_ref_emb(embeddings, labels, ref_emb, ref_labels)
        loss_dict = self.compute_loss(embeddings, labels, indices_tuple, ref_emb, ref_labels)
        self.add_embedding_regularization_to_loss_dict(loss_dict, embeddings)
        return self.reducer(loss_dict, embeddings, labels)

    def zero_loss(self):
        return {'losses': 0, 'indices': None, 'reduction_type': 'already_reduced'}

    def zero_losses(self):
        return {loss_name: self.zero_loss() for loss_name in self.sub_loss_names()}

    def _sub_loss_names(self):
        return ['loss']

    def sub_loss_names(self):
        return self._sub_loss_names() + self.all_regularization_loss_names()

    def all_regularization_loss_names(self):
        reg_names = []
        for base_class in inspect.getmro(self.__class__):
            base_class_name = base_class.__name__
            mixin_keyword = 'RegularizerMixin'
            if base_class_name.endswith(mixin_keyword):
                descriptor = base_class_name.replace(mixin_keyword, '').lower()
                if getattr(self, '{}_regularizer'.format(descriptor)):
                    reg_names.extend(base_class.regularization_loss_names(self))
        return reg_names


class CrossBatchMemory(ModuleWithRecords):

    def __init__(self, loss, embedding_size, memory_size=1024, miner=None, **kwargs):
        super().__init__(**kwargs)
        self.loss = loss
        self.miner = miner
        self.embedding_size = embedding_size
        self.memory_size = memory_size
        self.reset_queue()
        self.add_to_recordable_attributes(list_of_names=['embedding_size', 'memory_size', 'queue_idx'], is_stat=False)

    def forward(self, embeddings, labels, indices_tuple=None, enqueue_idx=None):
        if enqueue_idx is not None:
            assert len(enqueue_idx) <= len(self.embedding_memory)
            assert len(enqueue_idx) < len(embeddings)
        else:
            assert len(embeddings) <= len(self.embedding_memory)
        self.reset_stats()
        device = embeddings.device
        labels = c_f.to_device(labels, device=device)
        self.embedding_memory = c_f.to_device(self.embedding_memory, device=device, dtype=embeddings.dtype)
        self.label_memory = c_f.to_device(self.label_memory, device=device, dtype=labels.dtype)
        if enqueue_idx is not None:
            mask = torch.zeros(len(embeddings), device=device, dtype=torch.bool)
            mask[enqueue_idx] = True
            emb_for_queue = embeddings[mask]
            labels_for_queue = labels[mask]
            embeddings = embeddings[~mask]
            labels = labels[~mask]
            do_remove_self_comparisons = False
        else:
            emb_for_queue = embeddings
            labels_for_queue = labels
            do_remove_self_comparisons = True
        batch_size = len(embeddings)
        queue_batch_size = len(emb_for_queue)
        self.add_to_memory(emb_for_queue, labels_for_queue, queue_batch_size)
        if not self.has_been_filled:
            E_mem = self.embedding_memory[:self.queue_idx]
            L_mem = self.label_memory[:self.queue_idx]
        else:
            E_mem = self.embedding_memory
            L_mem = self.label_memory
        indices_tuple = self.create_indices_tuple(batch_size, embeddings, labels, E_mem, L_mem, indices_tuple, do_remove_self_comparisons)
        loss = self.loss(embeddings, labels, indices_tuple, E_mem, L_mem)
        return loss

    def add_to_memory(self, embeddings, labels, batch_size):
        self.curr_batch_idx = torch.arange(self.queue_idx, self.queue_idx + batch_size, device=labels.device) % self.memory_size
        self.embedding_memory[self.curr_batch_idx] = embeddings.detach()
        self.label_memory[self.curr_batch_idx] = labels.detach()
        prev_queue_idx = self.queue_idx
        self.queue_idx = (self.queue_idx + batch_size) % self.memory_size
        if not self.has_been_filled and self.queue_idx <= prev_queue_idx:
            self.has_been_filled = True

    def create_indices_tuple(self, batch_size, embeddings, labels, E_mem, L_mem, input_indices_tuple, do_remove_self_comparisons):
        if self.miner:
            indices_tuple = self.miner(embeddings, labels, E_mem, L_mem)
        else:
            indices_tuple = lmu.get_all_pairs_indices(labels, L_mem)
        if do_remove_self_comparisons:
            indices_tuple = lmu.remove_self_comparisons(indices_tuple, self.curr_batch_idx, self.memory_size)
        if input_indices_tuple is not None:
            if len(input_indices_tuple) == 3 and len(indices_tuple) == 4:
                input_indices_tuple = lmu.convert_to_pairs(input_indices_tuple, labels)
            elif len(input_indices_tuple) == 4 and len(indices_tuple) == 3:
                input_indices_tuple = lmu.convert_to_triplets(input_indices_tuple, labels)
            indices_tuple = c_f.concatenate_indices_tuples(indices_tuple, input_indices_tuple)
        return indices_tuple

    def reset_queue(self):
        self.embedding_memory = torch.zeros(self.memory_size, self.embedding_size)
        self.label_memory = torch.zeros(self.memory_size).long()
        self.has_been_filled = False
        self.queue_idx = 0


def all_gather(x):
    world_size = torch.distributed.get_world_size()
    if world_size > 1:
        rank = torch.distributed.get_rank()
        x_list = [torch.ones_like(x) for _ in range(world_size)]
        torch.distributed.all_gather(x_list, x.contiguous())
        x_list.pop(rank)
        return torch.cat(x_list, dim=0)
    return None


def is_distributed():
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def all_gather_embeddings_and_labels(emb, labels):
    if not is_distributed():
        return None, None
    ref_emb = all_gather(emb)
    ref_labels = all_gather(labels)
    return ref_emb, ref_labels


def gather(emb, labels):
    device = emb.device
    labels = c_f.to_device(labels, device=device)
    dist_emb, dist_labels = all_gather_embeddings_and_labels(emb, labels)
    all_emb = torch.cat([emb, dist_emb], dim=0)
    all_labels = torch.cat([labels, dist_labels], dim=0)
    return all_emb, all_labels, labels


def gather_emb_and_ref(emb, labels, ref_emb=None, ref_labels=None):
    all_emb, all_labels, labels = gather(emb, labels)
    all_ref_emb, all_ref_labels = None, None
    if ref_emb is not None and ref_labels is not None:
        all_ref_emb, all_ref_labels, _ = gather(ref_emb, ref_labels)
    return all_emb, all_labels, all_ref_emb, all_ref_labels, labels


def get_indices_tuple(labels, ref_labels, embeddings=None, ref_emb=None, miner=None):
    device = labels.device
    curr_batch_idx = torch.arange(len(labels), device=device)
    if miner:
        indices_tuple = miner(embeddings, labels, ref_emb, ref_labels)
    else:
        indices_tuple = lmu.get_all_pairs_indices(labels, ref_labels)
    return lmu.remove_self_comparisons(indices_tuple, curr_batch_idx, len(ref_labels))


def select_ref_or_regular(regular, ref):
    return regular if ref is None else ref


class DistributedLossWrapper(torch.nn.Module):

    def __init__(self, loss, efficient=False):
        super().__init__()
        if not isinstance(loss, (BaseMetricLossFunction, CrossBatchMemory)):
            raise TypeError('The input loss must extend BaseMetricLossFunction or CrossBatchMemory')
        if isinstance(loss, CrossBatchMemory) and efficient:
            raise ValueError('CrossBatchMemory with efficient=True is not currently supported')
        self.loss = loss
        self.efficient = efficient

    def forward(self, emb, labels, indices_tuple=None, ref_emb=None, ref_labels=None):
        world_size = torch.distributed.get_world_size()
        common_args = [emb, labels, indices_tuple, ref_emb, ref_labels, world_size]
        if isinstance(self.loss, CrossBatchMemory):
            return self.forward_cross_batch(*common_args)
        return self.forward_regular_loss(*common_args)

    def forward_regular_loss(self, emb, labels, indices_tuple, ref_emb, ref_labels, world_size):
        if world_size <= 1:
            return self.loss(emb, labels, indices_tuple, ref_emb, ref_labels)
        all_emb, all_labels, all_ref_emb, all_ref_labels, labels = gather_emb_and_ref(emb, labels, ref_emb, ref_labels)
        if self.efficient:
            all_labels = select_ref_or_regular(all_labels, all_ref_labels)
            all_emb = select_ref_or_regular(all_emb, all_ref_emb)
            if indices_tuple is None:
                indices_tuple = get_indices_tuple(labels, all_labels)
            loss = self.loss(emb, labels, indices_tuple, all_emb, all_labels)
        else:
            loss = self.loss(all_emb, all_labels, indices_tuple, all_ref_emb, all_ref_labels)
        return loss * world_size

    def forward_cross_batch(self, emb, labels, indices_tuple, ref_emb, ref_labels, world_size):
        if ref_emb is not None or ref_labels is not None:
            raise ValueError('CrossBatchMemory is not compatible with ref_emb and ref_labels')
        if world_size <= 1:
            return self.loss(emb, labels, indices_tuple)
        all_emb, all_labels, _, _, _ = gather_emb_and_ref(emb, labels, ref_emb, ref_labels)
        loss = self.loss(all_emb, all_labels, indices_tuple)
        return loss * world_size


class BaseMiner(ModuleWithRecordsAndDistance):

    def mine(self, embeddings, labels, ref_emb, ref_labels):
        raise NotImplementedError

    def output_assertion(self, output):
        raise NotImplementedError

    def forward(self, embeddings, labels, ref_emb=None, ref_labels=None):
        """
        Args:
            embeddings: tensor of size (batch_size, embedding_size)
            labels: tensor of size (batch_size)
        Does any necessary preprocessing, then does mining, and then checks the
        shape of the mining output before returning it
        """
        self.reset_stats()
        with torch.no_grad():
            c_f.check_shapes(embeddings, labels)
            labels = c_f.to_device(labels, embeddings)
            ref_emb, ref_labels = c_f.set_ref_emb(embeddings, labels, ref_emb, ref_labels)
            mining_output = self.mine(embeddings, labels, ref_emb, ref_labels)
        self.output_assertion(mining_output)
        return mining_output


class DistributedMinerWrapper(torch.nn.Module):

    def __init__(self, miner, efficient=False):
        super().__init__()
        if not isinstance(miner, BaseMiner):
            raise TypeError('The input miner must extend BaseMiner')
        self.miner = miner
        self.efficient = efficient

    def forward(self, emb, labels, ref_emb=None, ref_labels=None):
        world_size = torch.distributed.get_world_size()
        if world_size <= 1:
            return self.miner(emb, labels, ref_emb, ref_labels)
        all_emb, all_labels, all_ref_emb, all_ref_labels, labels = gather_emb_and_ref(emb, labels, ref_emb, ref_labels)
        if self.efficient:
            all_labels = select_ref_or_regular(all_labels, all_ref_labels)
            all_emb = select_ref_or_regular(all_emb, all_ref_emb)
            return get_indices_tuple(labels, all_labels, emb, all_emb, self.miner)
        else:
            return self.miner(all_emb, all_labels, all_ref_emb, all_ref_labels)


def dSoftBinning(D, mid, Delta):
    side1 = (D > mid - Delta).type(D.dtype)
    side2 = (D <= mid).type(D.dtype)
    ind1 = side1 * side2
    side1 = (D > mid).type(D.dtype)
    side2 = (D <= mid + Delta).type(D.dtype)
    ind2 = side1 * side2
    return (ind1 - ind2) / Delta


def softBinning(D, mid, Delta):
    y = 1 - torch.abs(D - mid) / Delta
    return torch.max(torch.tensor([0], dtype=D.dtype), y)


class OriginalImplementationFastAP(torch.autograd.Function):
    """
    FastAP - autograd function definition

    This class implements the FastAP loss from the following paper:
    "Deep Metric Learning to Rank",
    F. Cakir, K. He, X. Xia, B. Kulis, S. Sclaroff. CVPR 2019

    NOTE:
        Given a input batch, FastAP does not sample triplets from it as it's not
        a triplet-based method. Therefore, FastAP does not take a Sampler as input.
        Rather, we specify how the input batch is selected.
    """

    @staticmethod
    def forward(ctx, input, target, num_bins):
        """
        Args:
            input:     torch.Tensor(N x embed_dim), embedding matrix
            target:    torch.Tensor(N x 1), class labels
            num_bins:  int, number of bins in distance histogram
        """
        N = target.size()[0]
        assert input.size()[0] == N, "Batch size donesn't match!"
        Y = target.unsqueeze(1)
        Aff = 2 * (Y == Y.t()).type(input.dtype) - 1
        Aff.masked_fill_(torch.eye(N, N).bool(), 0)
        I_pos = (Aff > 0).type(input.dtype)
        I_neg = (Aff < 0).type(input.dtype)
        N_pos = torch.sum(I_pos, 1)
        dist2 = 2 - 2 * torch.mm(input, input.t())
        Delta = torch.tensor(4.0 / num_bins)
        Z = torch.linspace(0.0, 4.0, steps=num_bins + 1)
        L = Z.size()[0]
        h_pos = torch.zeros((N, L), dtype=input.dtype)
        h_neg = torch.zeros((N, L), dtype=input.dtype)
        for l in range(L):
            pulse = softBinning(dist2, Z[l], Delta)
            h_pos[:, l] = torch.sum(pulse * I_pos, 1)
            h_neg[:, l] = torch.sum(pulse * I_neg, 1)
        H_pos = torch.cumsum(h_pos, 1)
        h = h_pos + h_neg
        H = torch.cumsum(h, 1)
        FastAP = h_pos * H_pos / H
        FastAP[torch.isnan(FastAP) | torch.isinf(FastAP)] = 0
        FastAP = torch.sum(FastAP, 1) / N_pos
        FastAP = FastAP[~torch.isnan(FastAP)]
        loss = 1 - torch.mean(FastAP)
        ctx.save_for_backward(input, target)
        ctx.Z = Z
        ctx.Delta = Delta
        ctx.dist2 = dist2
        ctx.I_pos = I_pos
        ctx.I_neg = I_neg
        ctx.h_pos = h_pos
        ctx.h_neg = h_neg
        ctx.H_pos = H_pos
        ctx.N_pos = N_pos
        ctx.h = h
        ctx.H = H
        ctx.L = torch.tensor(L)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        input, target = ctx.saved_tensors
        Z = Variable(ctx.Z, requires_grad=False)
        Delta = Variable(ctx.Delta, requires_grad=False)
        dist2 = Variable(ctx.dist2, requires_grad=False)
        I_pos = Variable(ctx.I_pos, requires_grad=False)
        I_neg = Variable(ctx.I_neg, requires_grad=False)
        h = Variable(ctx.h, requires_grad=False)
        H = Variable(ctx.H, requires_grad=False)
        h_pos = Variable(ctx.h_pos, requires_grad=False)
        h_neg = Variable(ctx.h_neg, requires_grad=False)
        H_pos = Variable(ctx.H_pos, requires_grad=False)
        N_pos = Variable(ctx.N_pos, requires_grad=False)
        L = Z.size()[0]
        H2 = torch.pow(H, 2)
        H_neg = H - H_pos
        LTM1 = torch.tril(torch.ones(L, L), -1)
        tmp1 = h_pos * H_neg / H2
        tmp1[torch.isnan(tmp1)] = 0
        d_AP_h_pos = (H_pos * H + h_pos * H_neg) / H2
        d_AP_h_pos = d_AP_h_pos + torch.mm(tmp1, LTM1)
        d_AP_h_pos = d_AP_h_pos / N_pos.repeat(L, 1).t()
        d_AP_h_pos[torch.isnan(d_AP_h_pos) | torch.isinf(d_AP_h_pos)] = 0
        LTM0 = torch.tril(torch.ones(L, L), 0)
        tmp2 = -h_pos * H_pos / H2
        tmp2[torch.isnan(tmp2)] = 0
        d_AP_h_neg = torch.mm(tmp2, LTM0)
        d_AP_h_neg = d_AP_h_neg / N_pos.repeat(L, 1).t()
        d_AP_h_neg[torch.isnan(d_AP_h_neg) | torch.isinf(d_AP_h_neg)] = 0
        d_AP_x = 0
        for l in range(L):
            dpulse = dSoftBinning(dist2, Z[l], Delta)
            dpulse[torch.isnan(dpulse) | torch.isinf(dpulse)] = 0
            ddp = dpulse * I_pos
            ddn = dpulse * I_neg
            alpha_p = torch.diag(d_AP_h_pos[:, l])
            alpha_n = torch.diag(d_AP_h_neg[:, l])
            Ap = torch.mm(ddp, alpha_p) + torch.mm(alpha_p, ddp)
            An = torch.mm(ddn, alpha_n) + torch.mm(alpha_n, ddn)
            d_AP_x = d_AP_x - torch.mm(input.t(), Ap + An)
        grad_input = -d_AP_x
        return grad_input.t(), None, None


class OriginalImplementationFastAPLoss(torch.nn.Module):
    """
    FastAP - loss layer definition

    This class implements the FastAP loss from the following paper:
    "Deep Metric Learning to Rank",
    F. Cakir, K. He, X. Xia, B. Kulis, S. Sclaroff. CVPR 2019
    """

    def __init__(self, num_bins=10):
        super(OriginalImplementationFastAPLoss, self).__init__()
        self.num_bins = num_bins

    def forward(self, batch, labels):
        return OriginalImplementationFastAP.apply(batch, labels, self.num_bins)


def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output


class OriginalInstanceLoss(nn.Module):

    def __init__(self, gamma=1) ->None:
        super().__init__()
        self.gamma = gamma

    def forward(self, feature, label=None) ->Tensor:
        normed_feature = l2_norm(feature)
        sim1 = torch.mm(normed_feature * self.gamma, torch.t(normed_feature))
        if label is None:
            sim_label = torch.arange(sim1.size(0)).detach()
        else:
            _, sim_label = torch.unique(label, return_inverse=True)
        loss = F.cross_entropy(sim1, sim_label)
        return loss


def binarize(T, nb_classes):
    T = T.cpu().numpy()
    import sklearn.preprocessing
    T = sklearn.preprocessing.label_binarize(T, classes=range(0, nb_classes))
    T = torch.FloatTensor(T)
    return T


class OriginalImplementationProxyAnchor(torch.nn.Module):

    def __init__(self, nb_classes, sz_embed, mrg=0.1, alpha=32):
        torch.nn.Module.__init__(self)
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed))
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')
        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha

    def forward(self, X, T):
        P = self.proxies
        cos = F.linear(l2_norm(X), l2_norm(P))
        P_one_hot = binarize(T=T, nb_classes=self.nb_classes)
        N_one_hot = 1 - P_one_hot
        pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
        neg_exp = torch.exp(self.alpha * (cos + self.mrg))
        with_pos_proxies = torch.where(P_one_hot.sum(dim=0) != 0)[0]
        num_valid_proxies = len(with_pos_proxies)
        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0)
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)
        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        loss = pos_term + neg_term
        return loss


class OriginalImplementationSoftTriple(nn.Module):

    def __init__(self, la, gamma, tau, margin, dim, cN, K):
        super(OriginalImplementationSoftTriple, self).__init__()
        self.la = la
        self.gamma = 1.0 / gamma
        self.tau = tau
        self.margin = margin
        self.cN = cN
        self.K = K
        self.fc = Parameter(torch.Tensor(dim, cN * K))
        self.weight = torch.zeros(cN * K, cN * K, dtype=torch.bool)
        for i in range(0, cN):
            for j in range(0, K):
                self.weight[i * K + j, i * K + j + 1:(i + 1) * K] = 1
        init.kaiming_uniform_(self.fc, a=math.sqrt(5))
        return

    def forward(self, input, target):
        centers = F.normalize(self.fc, p=2, dim=0)
        simInd = input.matmul(centers)
        simStruc = simInd.reshape(-1, self.cN, self.K)
        prob = F.softmax(simStruc * self.gamma, dim=2)
        simClass = torch.sum(prob * simStruc, dim=2)
        marginM = torch.zeros(simClass.shape, dtype=input.dtype)
        marginM[torch.arange(0, marginM.shape[0]), target] = self.margin
        lossClassify = F.cross_entropy(self.la * (simClass - marginM), target)
        if self.tau > 0 and self.K > 1:
            simCenter = centers.t().matmul(centers)
            small_val = c_f.small_val(input.dtype)
            simCenterMasked = torch.clamp(2.0 * simCenter[self.weight], max=2)
            reg = torch.sum(torch.sqrt(2.0 + small_val - simCenterMasked)) / (self.cN * self.K * (self.K - 1.0))
            return lossClassify + self.tau * reg
        else:
            return lossClassify


class ToyMpModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.net1 = torch.nn.Linear(10, 10)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(10, 5)

    def forward(self, x):
        x = self.relu(self.net1(x))
        return self.net2(x)


class TextModel(torch.nn.Module):

    def forward(self, list_of_text):
        return torch.randn(len(list_of_text), 32)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Identity,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (OriginalInstanceLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (TextModel,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_KevinMusgrave_pytorch_metric_learning(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

