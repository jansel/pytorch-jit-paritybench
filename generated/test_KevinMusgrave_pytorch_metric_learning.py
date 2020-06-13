import sys
_module = sys.modules[__name__]
del sys
setup = _module
pytorch_metric_learning = _module
losses = _module
angular_loss = _module
arcface_loss = _module
base_metric_loss_function = _module
circle_loss = _module
contrastive_loss = _module
cosface_loss = _module
cross_batch_memory = _module
fast_ap_loss = _module
generic_pair_loss = _module
intra_pair_variance_loss = _module
large_margin_softmax_loss = _module
lifted_structure_loss = _module
margin_loss = _module
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
triplet_margin_loss = _module
tuplet_margin_loss = _module
weight_regularizer_mixin = _module
miners = _module
angular_miner = _module
base_miner = _module
batch_hard_miner = _module
distance_weighted_miner = _module
embeddings_already_packaged_as_triplets = _module
hdc_miner = _module
maximum_loss_miner = _module
multi_similarity_miner = _module
pair_margin_miner = _module
triplet_margin_miner = _module
regularizers = _module
base_weight_regularizer = _module
center_invariant_regularizer = _module
regular_face_regularizer = _module
samplers = _module
fixed_set_of_triplets = _module
m_per_class_sampler = _module
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
inference = _module
logging_presets = _module
loss_and_miner_utils = _module
loss_tracker = _module
stat_utils = _module
tests = _module
test_angular_loss = _module
test_arcface_loss = _module
test_contrastive_loss = _module
test_cross_batch_memory = _module
test_margin_loss = _module
test_multi_similarity_loss = _module
test_ntxent_loss = _module
test_triplet_margin_loss = _module
test_batch_hard_miner = _module
test_hdc_miner = _module
test_pair_margin_miner = _module
test_calculate_accuracies = _module
test_loss_and_miner_utils = _module

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


import numpy as np


import torch


import torch.nn.functional as F


import scipy.special


import math


import copy


import collections


import logging


import scipy.stats


import re


from collections import defaultdict


class BaseMetricLossFunction(torch.nn.Module):
    """
    All loss functions extend this class
    Args:
        normalize_embeddings: type boolean. If True then normalize embeddins
                                to have norm = 1 before computing the loss
        num_class_per_param: type int. The number of classes for each parameter.
                            If your parameters don't have a separate value for each class,
                            then leave this at None
        learnable_param_names: type list of strings. Each element is the name of
                            attributes that should be converted to nn.Parameter 
    """

    def __init__(self, normalize_embeddings=True, num_class_per_param=None,
        learnable_param_names=None):
        super().__init__()
        self.normalize_embeddings = normalize_embeddings
        self.num_class_per_param = num_class_per_param
        self.learnable_param_names = learnable_param_names
        self.initialize_learnable_parameters()
        self.add_to_recordable_attributes(name='avg_embedding_norm')

    def compute_loss(self, embeddings, labels, indices_tuple=None):
        """
        This has to be implemented and is what actually computes the loss.
        """
        raise NotImplementedError

    def forward(self, embeddings, labels, indices_tuple=None):
        """
        Args:
            embeddings: tensor of size (batch_size, embedding_size)
            labels: tensor of size (batch_size)
            indices_tuple: tuple of size 3 for triplets (anchors, positives, negatives)
                            or size 4 for pairs (anchor1, postives, anchor2, negatives)
                            Can also be left as None
        Returns: the loss (float)
        """
        c_f.assert_embeddings_and_labels_are_same_size(embeddings, labels)
        labels = labels.to(embeddings.device)
        if self.normalize_embeddings:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        self.embedding_norms = torch.norm(embeddings, p=2, dim=1)
        self.avg_embedding_norm = torch.mean(self.embedding_norms)
        loss = self.compute_loss(embeddings, labels, indices_tuple)
        if loss == 0:
            loss = torch.sum(embeddings * 0)
        return loss

    def initialize_learnable_parameters(self):
        """
        To learn hyperparams, create an attribute called learnable_param_names.
        This should be a list of strings which are the names of the
        hyperparameters to be learned
        """
        if self.learnable_param_names is not None:
            for k in self.learnable_param_names:
                v = getattr(self, k)
                setattr(self, k, self.create_learnable_parameter(v))

    def create_learnable_parameter(self, init_value, unsqueeze=False):
        """
        Returns nn.Parameter with an initial value of init_value
        and size of num_labels
        """
        vec_len = self.num_class_per_param if self.num_class_per_param else 1
        if unsqueeze:
            vec_len = vec_len, 1
        p = torch.nn.Parameter(torch.ones(vec_len) * init_value)
        return p

    def maybe_mask_param(self, param, labels):
        """
        This returns the hyperparameters corresponding to class labels (if applicable).
        If there is a hyperparameter for each class, then when computing the loss,
        the class hyperparameter has to be matched to the corresponding embedding.
        """
        if self.num_class_per_param:
            return param[labels]
        return param

    def add_to_recordable_attributes(self, name=None, list_of_names=None):
        c_f.add_to_recordable_attributes(self, name=name, list_of_names=
            list_of_names)


class MultipleLosses(torch.nn.Module):

    def __init__(self, losses, weights=None):
        super().__init__()
        self.losses = torch.nn.ModuleList(losses)
        self.weights = weights if weights is not None else [1] * len(self.
            losses)

    def forward(self, embeddings, labels, indices_tuple=None):
        total_loss = 0
        for i, loss in enumerate(self.losses):
            total_loss += loss(embeddings, labels, indices_tuple
                ) * self.weights[i]
        return total_loss


class CrossBatchMemory(torch.nn.Module):

    def __init__(self, loss, embedding_size, memory_size=1024, miner=None):
        super().__init__()
        self.loss = loss
        self.miner = miner
        self.memory_size = memory_size
        self.embedding_memory = torch.zeros(self.memory_size, embedding_size)
        self.label_memory = torch.zeros(self.memory_size).long()
        self.has_been_filled = False
        self.queue_idx = 0

    def forward(self, embeddings, labels, input_indices_tuple=None):
        assert embeddings.size(0) <= self.embedding_memory.size(0)
        batch_size = embeddings.size(0)
        labels = labels.to(embeddings.device)
        self.embedding_memory = self.embedding_memory.to(embeddings.device)
        self.label_memory = self.label_memory.to(labels.device)
        self.add_to_memory(embeddings, labels, batch_size)
        if not self.has_been_filled:
            E_mem = self.embedding_memory[:self.queue_idx]
            L_mem = self.label_memory[:self.queue_idx]
        else:
            E_mem = self.embedding_memory
            L_mem = self.label_memory
        indices_tuple = self.create_indices_tuple(batch_size, embeddings,
            labels, E_mem, L_mem, input_indices_tuple)
        combined_embeddings = torch.cat([embeddings, E_mem], dim=0)
        combined_labels = torch.cat([labels, L_mem], dim=0)
        loss = self.loss(combined_embeddings, combined_labels, indices_tuple)
        return loss

    def add_to_memory(self, embeddings, labels, batch_size):
        end_idx = (self.queue_idx + batch_size - 1) % self.memory_size + 1
        if end_idx > self.queue_idx:
            self.embedding_memory[self.queue_idx:end_idx] = embeddings.detach()
            self.label_memory[self.queue_idx:end_idx] = labels.detach()
        else:
            se = self.memory_size - self.queue_idx
            self.embedding_memory[self.queue_idx:] = embeddings[:se].detach()
            self.embedding_memory[:end_idx] = embeddings[se:].detach()
            self.label_memory[self.queue_idx:] = labels[:se].detach()
            self.label_memory[:end_idx] = labels[se:].detach()
        prev_queue_idx = self.queue_idx
        self.queue_idx = (self.queue_idx + batch_size) % self.memory_size
        if not self.has_been_filled and self.queue_idx <= prev_queue_idx:
            self.has_been_filled = True

    def create_indices_tuple(self, batch_size, embeddings, labels, E_mem,
        L_mem, input_indices_tuple):
        if self.miner:
            indices_tuple = self.miner(embeddings, labels, E_mem, L_mem)
        else:
            indices_tuple = lmu.get_all_pairs_indices(labels, L_mem)
        indices_tuple = c_f.shift_indices_tuple(indices_tuple, batch_size)
        if input_indices_tuple is not None:
            if len(input_indices_tuple) == 3 and len(indices_tuple) == 4:
                input_indices_tuple = lmu.convert_to_pairs(input_indices_tuple,
                    labels)
            elif len(input_indices_tuple) == 4 and len(indices_tuple) == 3:
                input_indices_tuple = lmu.convert_to_triplets(
                    input_indices_tuple, labels)
            indices_tuple = tuple([torch.cat([x, y.to(x.device)], dim=0) for
                x, y in zip(indices_tuple, input_indices_tuple)])
        return indices_tuple


class BaseMiner(torch.nn.Module):

    def __init__(self, normalize_embeddings=True):
        super().__init__()
        self.normalize_embeddings = normalize_embeddings

    def mine(self, embeddings, labels, ref_emb, ref_labels):
        """
        Args:
            embeddings: tensor of size (batch_size, embedding_size)
            labels: tensor of size (batch_size)
        Returns: a tuple where each element is an array of indices.
        """
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
        with torch.no_grad():
            c_f.assert_embeddings_and_labels_are_same_size(embeddings, labels)
            labels = labels.to(embeddings.device)
            if self.normalize_embeddings:
                embeddings = torch.nn.functional.normalize(embeddings, p=2,
                    dim=1)
            ref_emb, ref_labels = self.set_ref_emb(embeddings, labels,
                ref_emb, ref_labels)
            mining_output = self.mine(embeddings, labels, ref_emb, ref_labels)
        self.output_assertion(mining_output)
        return mining_output

    def set_ref_emb(self, embeddings, labels, ref_emb, ref_labels):
        if ref_emb is not None:
            if self.normalize_embeddings:
                ref_emb = torch.nn.functional.normalize(ref_emb, p=2, dim=1)
            ref_labels = ref_labels.to(ref_emb.device)
        else:
            ref_emb, ref_labels = embeddings, labels
        c_f.assert_embeddings_and_labels_are_same_size(ref_emb, ref_labels)
        return ref_emb, ref_labels

    def add_to_recordable_attributes(self, name=None, list_of_names=None):
        c_f.add_to_recordable_attributes(self, name=name, list_of_names=
            list_of_names)


class BaseWeightRegularizer(torch.nn.Module):

    def __init__(self, normalize_weights=True):
        super().__init__()
        self.normalize_weights = normalize_weights
        self.add_to_recordable_attributes(name='avg_weight_norm')

    def compute_loss(self, weights):
        raise NotImplementedError

    def forward(self, weights):
        """
        weights should have shape (num_classes, embedding_size)
        """
        if self.normalize_weights:
            weights = torch.nn.functional.normalize(weights, p=2, dim=1)
        self.weight_norms = torch.norm(weights, p=2, dim=1)
        self.avg_weight_norm = torch.mean(self.weight_norms)
        loss = self.compute_loss(weights)
        if loss == 0:
            loss = torch.sum(weights * 0)
        return loss

    def add_to_recordable_attributes(self, name=None, list_of_names=None):
        c_f.add_to_recordable_attributes(self, name=name, list_of_names=
            list_of_names)


class Identity(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_KevinMusgrave_pytorch_metric_learning(_paritybench_base):
    pass
    def test_000(self):
        self._check(Identity(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

