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


import torch.nn.functional as F


import scipy.special


import math


from torch.utils.data.sampler import Sampler


import logging


from sklearn.preprocessing import normalize


from sklearn.preprocessing import StandardScaler


from collections import defaultdict


import copy


import collections


import scipy.stats


import re


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

    def __init__(self, normalize_embeddings=True, num_class_per_param=None, learnable_param_names=None):
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
        labels = labels
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
        c_f.add_to_recordable_attributes(self, name=name, list_of_names=list_of_names)


class MultipleLosses(torch.nn.Module):

    def __init__(self, losses, weights=None):
        super().__init__()
        self.losses = torch.nn.ModuleList(losses)
        self.weights = weights if weights is not None else [1] * len(self.losses)

    def forward(self, embeddings, labels, indices_tuple=None):
        total_loss = 0
        for i, loss in enumerate(self.losses):
            total_loss += loss(embeddings, labels, indices_tuple) * self.weights[i]
        return total_loss


class CircleLoss(BaseMetricLossFunction):
    """
    Circle loss for pairwise labels only. Support for class-level labels will be added 
    in the future.
    
    Args:
    m:  The relaxation factor that controls the radious of the decision boundary.
    gamma: The scale factor that determines the largest scale of each similarity score.

    According to the paper, the suggested default values of m and gamma are:

    Face Recognition: m = 0.25, gamma = 256
    Person Reidentification: m = 0.25, gamma = 256
    Fine-grained Image Retrieval: m = 0.4, gamma = 80

    By default, we set m = 0.4 and gamma = 80
    """

    def __init__(self, m=0.4, gamma=80, triplets_per_anchor='all', **kwargs):
        super(CircleLoss, self).__init__(**kwargs)
        self.m = m
        self.gamma = gamma
        self.triplets_per_anchor = triplets_per_anchor
        self.add_to_recordable_attributes(list_of_names=['num_unique_anchors', 'num_triplets'])
        self.soft_plus = torch.nn.Softplus(beta=1)
        assert self.normalize_embeddings, 'Embeddings must be normalized in circle loss!'

    def compute_loss(self, embeddings, labels, indices_tuple):
        indices_tuple = lmu.convert_to_triplets(indices_tuple, labels, t_per_anchor=self.triplets_per_anchor)
        anchor_idx, positive_idx, negative_idx = indices_tuple
        self.num_triplets = len(anchor_idx)
        if self.num_triplets == 0:
            self.num_unique_anchors = 0
            return 0
        anchors, positives, negatives = embeddings[anchor_idx], embeddings[positive_idx], embeddings[negative_idx]
        sp = torch.sum(anchors * positives, dim=1)
        sn = torch.sum(anchors * negatives, dim=1)
        loss = 0.0
        op = 1 + self.m
        on = -self.m
        delta_p = 1 - self.m
        delta_n = self.m
        unique_anchor_idx = torch.unique(anchor_idx)
        self.num_unique_anchors = len(unique_anchor_idx)
        for anchor in unique_anchor_idx:
            mask = anchor_idx == anchor
            sp_for_this_anchor = sp[mask]
            sn_for_this_anchor = sn[mask]
            alpha_p = torch.clamp(op - sp_for_this_anchor.detach(), min=0.0)
            alpha_n = torch.clamp(sn_for_this_anchor.detach() - on, min=0.0)
            logit_p = -self.gamma * alpha_p * (sp_for_this_anchor - delta_p)
            logit_n = self.gamma * alpha_n * (sn_for_this_anchor - delta_n)
            loss_for_this_anchor = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))
            loss += loss_for_this_anchor
        loss /= len(unique_anchor_idx)
        return loss


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
        labels = labels
        self.embedding_memory = self.embedding_memory
        self.label_memory = self.label_memory
        self.add_to_memory(embeddings, labels, batch_size)
        if not self.has_been_filled:
            E_mem = self.embedding_memory[:self.queue_idx]
            L_mem = self.label_memory[:self.queue_idx]
        else:
            E_mem = self.embedding_memory
            L_mem = self.label_memory
        indices_tuple = self.create_indices_tuple(batch_size, embeddings, labels, E_mem, L_mem, input_indices_tuple)
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

    def create_indices_tuple(self, batch_size, embeddings, labels, E_mem, L_mem, input_indices_tuple):
        if self.miner:
            indices_tuple = self.miner(embeddings, labels, E_mem, L_mem)
        else:
            indices_tuple = lmu.get_all_pairs_indices(labels, L_mem)
        indices_tuple = c_f.shift_indices_tuple(indices_tuple, batch_size)
        if input_indices_tuple is not None:
            if len(input_indices_tuple) == 3 and len(indices_tuple) == 4:
                input_indices_tuple = lmu.convert_to_pairs(input_indices_tuple, labels)
            elif len(input_indices_tuple) == 4 and len(indices_tuple) == 3:
                input_indices_tuple = lmu.convert_to_triplets(input_indices_tuple, labels)
            indices_tuple = tuple([torch.cat([x, y], dim=0) for x, y in zip(indices_tuple, input_indices_tuple)])
        return indices_tuple


class FastAPLoss(BaseMetricLossFunction):

    def __init__(self, num_bins, **kwargs):
        super().__init__(**kwargs)
        self.num_bins = int(num_bins)
        self.num_edges = self.num_bins + 1
    """
    Adapted from https://github.com/kunhe/FastAP-metric-learning
    """

    def compute_loss(self, embeddings, labels, indices_tuple):
        miner_weights = lmu.convert_to_weights(indices_tuple, labels)
        N = labels.size(0)
        a1_idx, p_idx, a2_idx, n_idx = lmu.get_all_pairs_indices(labels)
        I_pos = torch.zeros(N, N)
        I_neg = torch.zeros(N, N)
        I_pos[a1_idx, p_idx] = 1
        I_neg[a2_idx, n_idx] = 1
        N_pos = torch.sum(I_pos, dim=1)
        dist_mat = lmu.dist_mat(embeddings, squared=True)
        histogram_max = 4.0 if self.normalize_embeddings else torch.max(dist_mat).item()
        histogram_delta = histogram_max / self.num_bins
        mid_points = torch.linspace(0.0, histogram_max, steps=self.num_edges).view(-1, 1, 1)
        pulse = torch.nn.functional.relu(1 - torch.abs(dist_mat - mid_points) / histogram_delta)
        pos_hist = torch.t(torch.sum(pulse * I_pos, dim=2))
        neg_hist = torch.t(torch.sum(pulse * I_neg, dim=2))
        total_pos_hist = torch.cumsum(pos_hist, dim=1)
        total_hist = torch.cumsum(pos_hist + neg_hist, dim=1)
        loss = 0
        h_pos_product = pos_hist * total_pos_hist
        safe_H = (h_pos_product > 0) & (total_hist > 0)
        if torch.sum(safe_H) > 0:
            FastAP = torch.zeros_like(pos_hist)
            FastAP[safe_H] = h_pos_product[safe_H] / total_hist[safe_H]
            FastAP = torch.sum(FastAP, dim=1)
            safe_N = N_pos > 0
            if torch.sum(safe_N) > 0:
                FastAP = FastAP[safe_N] / N_pos[safe_N]
                FastAP = (1 - FastAP) * miner_weights[safe_N]
                loss = torch.mean(FastAP)
        return loss


class GenericPairLoss(BaseMetricLossFunction):
    """
    The function pair_based_loss has to be implemented by the child class.
    By default, this class extracts every positive and negative pair within a
    batch (based on labels) and passes the pairs to the loss function.
    The pairs can be passed to the loss function all at once (self.loss_once)
    or pairs can be passed iteratively (self.loss_loop) by going through each
    sample in a batch, and selecting just the positive and negative pairs
    containing that sample.
    Args:
        use_similarity: set to True if the loss function uses pairwise similarity
                        (dot product of each embedding pair). Otherwise,
                        euclidean distance will be used
        iterate_through_loss: set to True to use self.loss_loop and False otherwise
        squared_distances: if True, then the euclidean distance will be squared.
    """

    def __init__(self, use_similarity, mat_based_loss, squared_distances=False, **kwargs):
        super().__init__(**kwargs)
        self.use_similarity = use_similarity
        self.squared_distances = squared_distances
        self.loss_method = self.mat_based_loss if mat_based_loss else self.pair_based_loss

    def compute_loss(self, embeddings, labels, indices_tuple):
        mat = lmu.get_pairwise_mat(embeddings, embeddings, self.use_similarity, self.squared_distances)
        if self.use_similarity and not self.normalize_embeddings:
            embedding_norms_mat = self.embedding_norms.unsqueeze(0) * self.embedding_norms.unsqueeze(1)
            mat = mat / embedding_norms_mat
        indices_tuple = lmu.convert_to_pairs(indices_tuple, labels)
        return self.loss_method(mat, labels, indices_tuple)

    def _compute_loss(self):
        raise NotImplementedError

    def mat_based_loss(self, mat, labels, indices_tuple):
        a1, p, a2, n = indices_tuple
        pos_mask, neg_mask = torch.zeros_like(mat), torch.zeros_like(mat)
        pos_mask[a1, p] = 1
        neg_mask[a2, n] = 1
        return self._compute_loss(mat, pos_mask, neg_mask)

    def pair_based_loss(self, mat, labels, indices_tuple):
        a1, p, a2, n = indices_tuple
        pos_pair, neg_pair = [], []
        if len(a1) > 0:
            pos_pair = mat[a1, p]
        if len(a2) > 0:
            neg_pair = mat[a2, n]
        return self._compute_loss(pos_pair, neg_pair, indices_tuple)


class IntraPairVarianceLoss(GenericPairLoss):

    def __init__(self, pos_eps=0.01, neg_eps=0.01, **kwargs):
        super().__init__(**kwargs, use_similarity=True, mat_based_loss=False)
        self.pos_eps = pos_eps
        self.neg_eps = neg_eps
        self.add_to_recordable_attributes(list_of_names=['pos_loss', 'neg_loss'])

    def _compute_loss(self, pos_pairs, neg_pairs, indices_tuple):
        self.pos_loss, self.neg_loss = 0, 0
        if len(pos_pairs) > 0:
            mean_pos_sim = torch.mean(pos_pairs)
            pos_var = (1 - self.pos_eps) * mean_pos_sim - pos_pairs
            self.pos_loss = torch.mean(torch.nn.functional.relu(pos_var) ** 2)
        if len(neg_pairs) > 0:
            mean_neg_sim = torch.mean(neg_pairs)
            neg_var = neg_pairs - (1 + self.neg_eps) * mean_neg_sim
            self.neg_loss = torch.mean(torch.nn.functional.relu(neg_var) ** 2)
        return self.pos_loss + self.neg_loss


class WeightRegularizerMixin:

    def __init__(self, regularizer=None, reg_weight=1, **kwargs):
        super().__init__(**kwargs)
        self.regularizer = regularizer
        self.reg_weight = reg_weight

    def regularization_loss(self, weights):
        if self.regularizer is None:
            return 0
        return self.regularizer(weights) * self.reg_weight


class LargeMarginSoftmaxLoss(WeightRegularizerMixin, BaseMetricLossFunction):
    """
    Implementation of https://arxiv.org/pdf/1612.02295.pdf
    """

    def __init__(self, margin, num_classes, embedding_size, scale=1, normalize_weights=False, **kwargs):
        super().__init__(**kwargs)
        self.margin = margin
        self.num_classes = num_classes
        self.scale = scale
        self.normalize_weights = normalize_weights
        self.add_to_recordable_attributes(name='avg_angle')
        self.init_margin()
        self.W = torch.nn.Parameter(torch.randn(embedding_size, num_classes))
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')

    def init_margin(self):
        self.margin = int(self.margin)
        self.max_n = self.margin // 2
        self.n_range = torch.FloatTensor([n for n in range(0, self.max_n + 1)])
        self.margin_choose_n = torch.FloatTensor([scipy.special.binom(self.margin, 2 * n) for n in self.n_range])
        self.cos_powers = torch.FloatTensor([(self.margin - 2 * n) for n in self.n_range])
        self.alternating = torch.FloatTensor([((-1) ** n) for n in self.n_range])

    def get_cos_with_margin(self, cosine):
        cosine = cosine.unsqueeze(1)
        for attr in ['n_range', 'margin_choose_n', 'cos_powers', 'alternating']:
            setattr(self, attr, getattr(self, attr))
        cos_powered = cosine ** self.cos_powers
        sin_powered = (1 - cosine ** 2) ** self.n_range
        terms = self.alternating * self.margin_choose_n * cos_powered * sin_powered
        return torch.sum(terms, dim=1)

    def get_weights(self):
        if self.normalize_weights:
            return torch.nn.functional.normalize(self.W, p=2, dim=0)
        return self.W

    def get_cosine(self, embeddings):
        weights = self.get_weights()
        self.weight_norms = torch.norm(weights, p=2, dim=0)
        return torch.matmul(embeddings, weights) / (self.weight_norms.unsqueeze(0) * self.embedding_norms.unsqueeze(1))

    def get_angles(self, cosine_of_target_classes):
        angles = torch.acos(torch.clamp(cosine_of_target_classes, -1 + 1e-07, 1 - 1e-07))
        self.avg_angle = np.degrees(torch.mean(angles).item())
        return angles

    def get_target_mask(self, embeddings, labels):
        batch_size = labels.size(0)
        mask = torch.zeros(batch_size, self.num_classes)
        mask[torch.arange(batch_size), labels] = 1
        return mask

    def modify_cosine_of_target_classes(self, cosine_of_target_classes, *args):
        _, _, labels, _ = args
        cos_with_margin = self.get_cos_with_margin(cosine_of_target_classes)
        angles = self.get_angles(cosine_of_target_classes)
        with torch.no_grad():
            k = (angles / (math.pi / self.margin)).floor()
        phi = (-1) ** k * cos_with_margin - 2 * k
        target_weight_norms = self.weight_norms[labels]
        return phi * target_weight_norms * self.embedding_norms

    def compute_loss(self, embeddings, labels, indices_tuple):
        miner_weights = lmu.convert_to_weights(indices_tuple, labels)
        mask = self.get_target_mask(embeddings, labels)
        cosine = self.get_cosine(embeddings)
        cosine_of_target_classes = cosine[mask == 1]
        modified_cosine_of_target_classes = self.modify_cosine_of_target_classes(cosine_of_target_classes, cosine, embeddings, labels, mask)
        diff = (modified_cosine_of_target_classes - cosine_of_target_classes).unsqueeze(1)
        cosine = cosine + mask * diff
        unweighted_loss = self.cross_entropy(cosine * self.scale, labels)
        return torch.mean(unweighted_loss * miner_weights) + self.regularization_loss(self.W.t())


class GeneralizedLiftedStructureLoss(GenericPairLoss):

    def __init__(self, neg_margin, **kwargs):
        super().__init__(use_similarity=False, mat_based_loss=True, **kwargs)
        self.neg_margin = neg_margin

    def _compute_loss(self, mat, pos_mask, neg_mask):
        pos_loss = lmu.logsumexp(mat, keep_mask=pos_mask, add_one=False)
        neg_loss = lmu.logsumexp(self.neg_margin - mat, keep_mask=neg_mask, add_one=False)
        return torch.mean(torch.relu(pos_loss + neg_loss))


class MarginLoss(BaseMetricLossFunction):

    def __init__(self, margin, nu, beta, triplets_per_anchor='all', **kwargs):
        self.margin = margin
        self.nu = nu
        self.beta = beta
        self.triplets_per_anchor = triplets_per_anchor
        self.add_to_recordable_attributes(list_of_names=['num_pos_pairs', 'num_neg_pairs', 'margin_loss', 'beta_reg_loss'])
        super().__init__(**kwargs)

    def compute_loss(self, embeddings, labels, indices_tuple):
        anchor_idx, positive_idx, negative_idx = lmu.convert_to_triplets(indices_tuple, labels, self.triplets_per_anchor)
        if len(anchor_idx) == 0:
            self.num_pos_pairs = 0
            self.num_neg_pairs = 0
            return 0
        anchors, positives, negatives = embeddings[anchor_idx], embeddings[positive_idx], embeddings[negative_idx]
        beta = self.maybe_mask_param(self.beta, labels[anchor_idx])
        self.beta_reg_loss = self.compute_reg_loss(beta)
        d_ap = torch.nn.functional.pairwise_distance(positives, anchors, p=2)
        d_an = torch.nn.functional.pairwise_distance(negatives, anchors, p=2)
        pos_loss = torch.nn.functional.relu(d_ap - beta + self.margin)
        neg_loss = torch.nn.functional.relu(beta - d_an + self.margin)
        self.num_pos_pairs = (pos_loss > 0.0).nonzero().size(0)
        self.num_neg_pairs = (neg_loss > 0.0).nonzero().size(0)
        pair_count = self.num_pos_pairs + self.num_neg_pairs
        if pair_count > 0:
            self.margin_loss = torch.sum(pos_loss + neg_loss) / pair_count
            self.beta_reg_loss = self.beta_reg_loss / pair_count
        else:
            self.margin_loss, self.beta_reg_loss = 0, 0
        return self.margin_loss + self.beta_reg_loss

    def compute_reg_loss(self, beta):
        if self.nu > 0:
            beta_sum = c_f.try_torch_operation(torch.sum, beta)
            return beta_sum * self.nu
        return 0


class MultiSimilarityLoss(GenericPairLoss):
    """
    modified from https://github.com/MalongTech/research-ms-loss/
    Args:
        alpha: The exponential weight for positive pairs
        beta: The exponential weight for negative pairs
        base: The shift in the exponent applied to both positive and negative pairs
    """

    def __init__(self, alpha, beta, base=0.5, **kwargs):
        super().__init__(use_similarity=True, mat_based_loss=True, **kwargs)
        self.alpha = alpha
        self.beta = beta
        self.base = base

    def _compute_loss(self, mat, pos_mask, neg_mask):
        pos_loss = 1.0 / self.alpha * lmu.logsumexp(-self.alpha * (mat - self.base), keep_mask=pos_mask, add_one=True)
        neg_loss = 1.0 / self.beta * lmu.logsumexp(self.beta * (mat - self.base), keep_mask=neg_mask, add_one=True)
        return torch.mean(pos_loss + neg_loss)


class NPairsLoss(BaseMetricLossFunction):
    """
    Implementation of https://arxiv.org/abs/1708.01682
    Args:
        l2_reg_weight: The L2 regularizer weight (multiplier)
    """

    def __init__(self, l2_reg_weight=0, **kwargs):
        super().__init__(**kwargs)
        self.l2_reg_weight = l2_reg_weight
        self.add_to_recordable_attributes(name='num_pairs')
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def compute_loss(self, embeddings, labels, indices_tuple):
        self.avg_embedding_norm = torch.mean(torch.norm(embeddings, p=2, dim=1))
        anchor_idx, positive_idx = lmu.convert_to_pos_pairs_with_unique_labels(indices_tuple, labels)
        self.num_pairs = len(anchor_idx)
        if self.num_pairs == 0:
            return 0
        anchors, positives = embeddings[anchor_idx], embeddings[positive_idx]
        targets = torch.arange(self.num_pairs)
        sim_mat = torch.matmul(anchors, positives.t())
        s_loss = self.cross_entropy(sim_mat, targets)
        if self.l2_reg_weight > 0:
            l2_reg = torch.mean(torch.norm(embeddings, p=2, dim=1))
            return s_loss + l2_reg * self.l2_reg_weight
        return s_loss


class NCALoss(BaseMetricLossFunction):

    def __init__(self, softmax_scale=1, **kwargs):
        super().__init__(**kwargs)
        self.softmax_scale = softmax_scale

    def compute_loss(self, embeddings, labels, indices_tuple):
        return self.nca_computation(embeddings, embeddings, labels, labels, indices_tuple)

    def nca_computation(self, query, reference, query_labels, reference_labels, indices_tuple):
        miner_weights = lmu.convert_to_weights(indices_tuple, query_labels)
        x = -lmu.dist_mat(query, reference, squared=True)
        if query is reference:
            diag_idx = torch.arange(query.size(0))
            x[diag_idx, diag_idx] = float('-inf')
        same_labels = (query_labels.unsqueeze(1) == reference_labels.unsqueeze(0)).float()
        exp = torch.nn.functional.softmax(self.softmax_scale * x, dim=1)
        exp = torch.sum(exp * same_labels, dim=1)
        non_zero = exp != 0
        return -torch.mean(torch.log(exp[non_zero]) * miner_weights[non_zero])


class NormalizedSoftmaxLoss(WeightRegularizerMixin, BaseMetricLossFunction):

    def __init__(self, temperature, embedding_size, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature
        self.W = torch.nn.Parameter(torch.randn(embedding_size, num_classes))
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')

    def compute_loss(self, embeddings, labels, indices_tuple):
        miner_weights = lmu.convert_to_weights(indices_tuple, labels)
        normalized_W = torch.nn.functional.normalize(self.W, p=2, dim=0)
        exponent = torch.matmul(embeddings, normalized_W) / self.temperature
        unweighted_loss = self.cross_entropy(exponent, labels)
        return torch.mean(unweighted_loss * miner_weights) + self.regularization_loss(self.W.t())


class NTXentLoss(GenericPairLoss):

    def __init__(self, temperature, **kwargs):
        super().__init__(**kwargs, use_similarity=True, mat_based_loss=False)
        self.temperature = temperature

    def _compute_loss(self, pos_pairs, neg_pairs, indices_tuple):
        a1, _, a2, _ = indices_tuple
        if len(a1) > 0 and len(a2) > 0:
            pos_pairs = pos_pairs.unsqueeze(1) / self.temperature
            neg_pairs = neg_pairs / self.temperature
            n_per_p = (a2.unsqueeze(0) == a1.unsqueeze(1)).float()
            neg_pairs = neg_pairs * n_per_p
            neg_pairs[n_per_p == 0] = float('-inf')
            max_val = torch.max(pos_pairs, torch.max(neg_pairs, dim=1, keepdim=True)[0])
            numerator = torch.exp(pos_pairs - max_val).squeeze(1)
            denominator = torch.sum(torch.exp(neg_pairs - max_val), dim=1) + numerator
            log_exp = torch.log(numerator / denominator + 1e-20)
            return torch.mean(-log_exp)
        return 0


class ProxyAnchorLoss(WeightRegularizerMixin, BaseMetricLossFunction):

    def __init__(self, num_classes, embedding_size, margin=0.1, alpha=32, **kwargs):
        super().__init__(**kwargs)
        self.proxies = torch.nn.Parameter(torch.randn(num_classes, embedding_size))
        torch.nn.init.kaiming_normal_(self.proxies, mode='fan_out')
        self.num_classes = num_classes
        self.margin = margin
        self.alpha = alpha

    def compute_loss(self, embeddings, labels, indices_tuple):
        miner_weights = lmu.convert_to_weights(indices_tuple, labels).unsqueeze(1)
        prox = torch.nn.functional.normalize(self.proxies, p=2, dim=1) if self.normalize_embeddings else self.proxies
        cos = lmu.sim_mat(embeddings, prox)
        if not self.normalize_embeddings:
            embedding_norms_mat = self.embedding_norms.unsqueeze(0) * torch.norm(prox, p=2, dim=1, keepdim=True)
            cos = cos / embedding_norms_mat.t()
        pos_mask = torch.nn.functional.one_hot(labels, self.num_classes).float()
        neg_mask = 1 - pos_mask
        with_pos_proxies = torch.nonzero(torch.sum(pos_mask, dim=0) != 0).squeeze(1)
        pos_term = lmu.logsumexp(-self.alpha * (cos - self.margin), keep_mask=pos_mask * miner_weights, add_one=True, dim=0)
        neg_term = lmu.logsumexp(self.alpha * (cos + self.margin), keep_mask=neg_mask * miner_weights, add_one=True, dim=0)
        pos_term = torch.sum(pos_term) / len(with_pos_proxies)
        neg_term = torch.sum(neg_term) / self.num_classes
        return pos_term + neg_term + self.regularization_loss(self.proxies)


class ProxyNCALoss(WeightRegularizerMixin, NCALoss):

    def __init__(self, num_classes, embedding_size, **kwargs):
        super().__init__(**kwargs)
        self.proxies = torch.nn.Parameter(torch.randn(num_classes, embedding_size))
        self.proxy_labels = torch.arange(num_classes)

    def compute_loss(self, embeddings, labels, indices_tuple):
        if self.normalize_embeddings:
            prox = torch.nn.functional.normalize(self.proxies, p=2, dim=1)
        else:
            prox = self.proxies
        nca_loss = self.nca_computation(embeddings, prox, labels, self.proxy_labels, indices_tuple)
        reg_loss = self.regularization_loss(self.proxies)
        return nca_loss + reg_loss


def SNR_dist(x, y, dim):
    return torch.var(x - y, dim=dim) / torch.var(x, dim=dim)


class SignalToNoiseRatioContrastiveLoss(BaseMetricLossFunction):

    def __init__(self, pos_margin, neg_margin, regularizer_weight, avg_non_zero_only=True, **kwargs):
        super().__init__(**kwargs)
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.regularizer_weight = regularizer_weight
        self.avg_non_zero_only = avg_non_zero_only
        self.add_to_recordable_attributes(list_of_names=['num_non_zero_pos_pairs', 'num_non_zero_neg_pairs', 'feature_distance_from_zero_mean_distribution'])

    def compute_loss(self, embeddings, labels, indices_tuple):
        a1, p, a2, n = lmu.convert_to_pairs(indices_tuple, labels)
        pos_loss, neg_loss, reg_loss = 0, 0, 0
        if len(a1) > 0:
            pos_loss, self.num_non_zero_pos_pairs = self.mask_margin_and_calculate_loss(embeddings[a1], embeddings[p], labels[a1], self.pos_margin, 1)
        if len(a2) > 0:
            neg_loss, self.num_non_zero_neg_pairs = self.mask_margin_and_calculate_loss(embeddings[a2], embeddings[n], labels[a2], self.neg_margin, -1)
        self.feature_distance_from_zero_mean_distribution = torch.mean(torch.abs(torch.sum(embeddings, dim=1)))
        if self.regularizer_weight > 0:
            reg_loss = self.regularizer_weight * self.feature_distance_from_zero_mean_distribution
        return pos_loss + neg_loss + reg_loss

    def mask_margin_and_calculate_loss(self, anchors, others, labels, margin, before_relu_multiplier):
        d = SNR_dist(anchors, others, dim=1)
        d = torch.nn.functional.relu((d - margin) * before_relu_multiplier)
        num_non_zero_pairs = (d > 0).nonzero().size(0)
        if self.avg_non_zero_only:
            if num_non_zero_pairs > 0:
                d = torch.sum(d) / num_non_zero_pairs
            else:
                d = 0
        else:
            d = torch.mean(d)
        return d, num_non_zero_pairs


class SoftTripleLoss(BaseMetricLossFunction):

    def __init__(self, embedding_size, num_classes, centers_per_class, la=20, gamma=0.1, reg_weight=0.2, margin=0.01, **kwargs):
        super().__init__(**kwargs)
        self.la = la
        self.gamma = 1.0 / gamma
        self.reg_weight = reg_weight
        self.margin = margin
        self.num_classes = num_classes
        self.centers_per_class = centers_per_class
        self.total_num_centers = num_classes * centers_per_class
        self.fc = torch.nn.Parameter(torch.Tensor(embedding_size, self.total_num_centers))
        self.set_class_masks(num_classes, centers_per_class)
        torch.nn.init.kaiming_uniform_(self.fc, a=math.sqrt(5))
        self.add_to_recordable_attributes(list_of_names=['same_class_center_similarity', 'diff_class_center_similarity'])

    def compute_loss(self, embeddings, labels, indices_tuple):
        miner_weights = lmu.convert_to_weights(indices_tuple, labels)
        centers = F.normalize(self.fc, p=2, dim=0) if self.normalize_embeddings else self.fc
        sim_to_centers = torch.matmul(embeddings, centers)
        sim_to_centers = sim_to_centers.view(-1, self.num_classes, self.centers_per_class)
        prob = F.softmax(sim_to_centers * self.gamma, dim=2)
        sim_to_classes = torch.sum(prob * sim_to_centers, dim=2)
        margin = torch.zeros(sim_to_classes.shape)
        margin[torch.arange(0, margin.shape[0]), labels] = self.margin
        loss = F.cross_entropy(self.la * (sim_to_classes - margin), labels, reduction='none')
        loss = torch.mean(loss * miner_weights)
        reg = 0
        if self.reg_weight > 0 and self.centers_per_class > 1:
            center_similarities = centers.t().matmul(centers)
            reg = torch.sum(torch.sqrt(2.0 + 1e-05 - 2.0 * center_similarities[self.same_class_mask])) / torch.sum(self.same_class_mask)
            self.set_stats(center_similarities)
        return loss + self.reg_weight * reg

    def set_class_masks(self, num_classes, centers_per_class):
        self.diff_class_mask = torch.ones(self.total_num_centers, self.total_num_centers, dtype=torch.bool)
        if centers_per_class > 1:
            self.same_class_mask = torch.zeros(self.total_num_centers, self.total_num_centers, dtype=torch.bool)
        for i in range(num_classes):
            s, e = i * centers_per_class, (i + 1) * centers_per_class
            if centers_per_class > 1:
                curr_block = torch.ones(centers_per_class, centers_per_class)
                curr_block = torch.triu(curr_block, diagonal=1)
                self.same_class_mask[s:e, s:e] = curr_block
            self.diff_class_mask[s:e, s:e] = 0

    def set_stats(self, center_similarities):
        if self.centers_per_class > 1:
            self.same_class_center_similarity = torch.mean(center_similarities[self.same_class_mask])
        self.diff_class_center_similarity = torch.mean(center_similarities[self.diff_class_mask])


class TripletMarginLoss(BaseMetricLossFunction):
    """
    Args:
        margin: The desired difference between the anchor-positive distance and the
                anchor-negative distance.
        distance_norm: The norm used when calculating distance between embeddings
        power: Each pair's loss will be raised to this power.
        swap: Use the positive-negative distance instead of anchor-negative distance,
              if it violates the margin more.
        smooth_loss: Use the log-exp version of the triplet loss
        avg_non_zero_only: Only pairs that contribute non-zero loss will be used in the final loss.
    """

    def __init__(self, margin=0.05, distance_norm=2, power=1, swap=False, smooth_loss=False, avg_non_zero_only=True, triplets_per_anchor='all', **kwargs):
        super().__init__(**kwargs)
        self.margin = margin
        self.distance_norm = distance_norm
        self.power = power
        self.swap = swap
        self.smooth_loss = smooth_loss
        self.avg_non_zero_only = avg_non_zero_only
        self.triplets_per_anchor = triplets_per_anchor
        self.add_to_recordable_attributes(name='num_non_zero_triplets')

    def compute_loss(self, embeddings, labels, indices_tuple):
        indices_tuple = lmu.convert_to_triplets(indices_tuple, labels, t_per_anchor=self.triplets_per_anchor)
        anchor_idx, positive_idx, negative_idx = indices_tuple
        if len(anchor_idx) == 0:
            self.num_non_zero_triplets = 0
            return 0
        anchors, positives, negatives = embeddings[anchor_idx], embeddings[positive_idx], embeddings[negative_idx]
        a_p_dist = F.pairwise_distance(anchors, positives, self.distance_norm)
        a_n_dist = F.pairwise_distance(anchors, negatives, self.distance_norm)
        if self.swap:
            p_n_dist = F.pairwise_distance(positives, negatives, self.distance_norm)
            a_n_dist = torch.min(a_n_dist, p_n_dist)
        a_p_dist = a_p_dist ** self.power
        a_n_dist = a_n_dist ** self.power
        if self.smooth_loss:
            inside_exp = a_p_dist - a_n_dist
            inside_exp = self.maybe_modify_loss(inside_exp)
            return torch.mean(torch.log(1 + torch.exp(inside_exp)))
        else:
            dist = a_p_dist - a_n_dist
            loss_modified = self.maybe_modify_loss(dist + self.margin)
            relued = torch.nn.functional.relu(loss_modified)
            self.num_non_zero_triplets = (relued > 0).nonzero().size(0)
            if self.avg_non_zero_only:
                if self.num_non_zero_triplets > 0:
                    return torch.sum(relued) / self.num_non_zero_triplets
                return 0
            return torch.mean(relued)

    def maybe_modify_loss(self, x):
        return x


class TupletMarginLoss(GenericPairLoss):

    def __init__(self, margin, scale=64, **kwargs):
        super().__init__(**kwargs, use_similarity=True, mat_based_loss=False)
        self.margin = np.radians(margin)
        self.scale = scale
        self.add_to_recordable_attributes(list_of_names=['avg_pos_angle', 'avg_neg_angle'])

    def _compute_loss(self, pos_pairs, neg_pairs, indices_tuple):
        a1, _, a2, _ = indices_tuple
        if len(a1) > 0 and len(a2) > 0:
            pos_angles = torch.acos(pos_pairs)
            neg_angles = torch.acos(neg_pairs)
            self.avg_pos_angle = np.degrees(torch.mean(pos_angles).item())
            self.avg_neg_angle = np.degrees(torch.mean(neg_angles).item())
            pos_pairs = torch.cos(pos_angles - self.margin)
            pos_pairs = pos_pairs.unsqueeze(1)
            neg_pairs = neg_pairs.repeat(pos_pairs.size(0), 1)
            inside_exp = self.scale * (neg_pairs - pos_pairs)
            keep_mask = (a2.unsqueeze(0) == a1.unsqueeze(1)).float()
            return torch.mean(lmu.logsumexp(inside_exp, keep_mask=keep_mask, add_one=True, dim=1))
        return 0


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
            labels = labels
            if self.normalize_embeddings:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            ref_emb, ref_labels = self.set_ref_emb(embeddings, labels, ref_emb, ref_labels)
            mining_output = self.mine(embeddings, labels, ref_emb, ref_labels)
        self.output_assertion(mining_output)
        return mining_output

    def set_ref_emb(self, embeddings, labels, ref_emb, ref_labels):
        if ref_emb is not None:
            if self.normalize_embeddings:
                ref_emb = torch.nn.functional.normalize(ref_emb, p=2, dim=1)
            ref_labels = ref_labels
        else:
            ref_emb, ref_labels = embeddings, labels
        c_f.assert_embeddings_and_labels_are_same_size(ref_emb, ref_labels)
        return ref_emb, ref_labels

    def add_to_recordable_attributes(self, name=None, list_of_names=None):
        c_f.add_to_recordable_attributes(self, name=name, list_of_names=list_of_names)


class BaseTupleMiner(BaseMiner):
    """
    Args:
        normalize_embeddings: type boolean, if True then normalize embeddings
                                to have norm = 1 before mining
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_to_recordable_attributes(list_of_names=['num_pos_pairs', 'num_neg_pairs', 'num_triplets'])

    def output_assertion(self, output):
        """
        Args:
            output: the output of self.mine
        This asserts that the mining function is outputting
        properly formatted indices. The default is to require a tuple representing
        a,p,n indices or a1,p,a2,n indices within a batch of embeddings.
        For example, a tuple of (anchors, positives, negatives) will be
        (torch.tensor, torch.tensor, torch.tensor)
        """
        if len(output) == 3:
            self.num_triplets = len(output[0])
            assert self.num_triplets == len(output[1]) == len(output[2])
        elif len(output) == 4:
            self.num_pos_pairs = len(output[0])
            self.num_neg_pairs = len(output[2])
            assert self.num_pos_pairs == len(output[1])
            assert self.num_neg_pairs == len(output[3])
        else:
            raise BaseException


class BaseSubsetBatchMiner(BaseMiner):
    """
    Args:
        output_batch_size: type int. The size of the subset that the miner
                            will output.
        normalize_embeddings: type boolean, if True then normalize embeddings
                                to have norm = 1 before mining
    """

    def __init__(self, output_batch_size, **kwargs):
        super().__init__(**kwargs)
        self.output_batch_size = output_batch_size

    def output_assertion(self, output):
        assert len(output) == self.output_batch_size


class BatchHardMiner(BaseTupleMiner):

    def __init__(self, use_similarity=False, squared_distances=False, **kwargs):
        super().__init__(**kwargs)
        self.use_similarity = use_similarity
        self.squared_distances = squared_distances
        self.add_to_recordable_attributes(list_of_names=['hardest_triplet_dist', 'hardest_pos_pair_dist', 'hardest_neg_pair_dist'])

    def mine(self, embeddings, labels, ref_emb, ref_labels):
        mat = lmu.get_pairwise_mat(embeddings, ref_emb, self.use_similarity, self.squared_distances)
        a1_idx, p_idx, a2_idx, n_idx = lmu.get_all_pairs_indices(labels, ref_labels)
        pos_func = self.get_min_per_row if self.use_similarity else self.get_max_per_row
        neg_func = self.get_max_per_row if self.use_similarity else self.get_min_per_row
        (hardest_positive_dist, hardest_positive_indices), a1p_keep = pos_func(mat, a1_idx, p_idx)
        (hardest_negative_dist, hardest_negative_indices), a2n_keep = neg_func(mat, a2_idx, n_idx)
        a_keep_idx = torch.where(a1p_keep & a2n_keep)
        self.set_stats(hardest_positive_dist[a_keep_idx], hardest_negative_dist[a_keep_idx])
        a = torch.arange(mat.size(0))[a_keep_idx]
        p = hardest_positive_indices[a_keep_idx]
        n = hardest_negative_indices[a_keep_idx]
        return a, p, n

    def get_max_per_row(self, mat, anchor_idx, other_idx):
        mask = torch.zeros_like(mat)
        mask[anchor_idx, other_idx] = 1
        non_zero_rows = torch.any(mask != 0, dim=1)
        mat_masked = mat * mask
        return torch.max(mat_masked, dim=1), non_zero_rows

    def get_min_per_row(self, mat, anchor_idx, other_idx):
        mask = torch.ones_like(mat) * float('inf')
        mask[anchor_idx, other_idx] = 1
        non_inf_rows = torch.any(mask != float('inf'), dim=1)
        mat_masked = mat * mask
        mat_masked[torch.isnan(mat_masked) | torch.isinf(mat_masked)] = float('inf')
        return torch.min(mat_masked, dim=1), non_inf_rows

    def set_stats(self, hardest_positive_dist, hardest_negative_dist):
        pos_func = torch.min if self.use_similarity else torch.max
        neg_func = torch.max if self.use_similarity else torch.min
        try:
            self.hardest_triplet_dist = pos_func(hardest_positive_dist - hardest_negative_dist).item()
            self.hardest_pos_pair_dist = pos_func(hardest_positive_dist).item()
            self.hardest_neg_pair_dist = neg_func(hardest_negative_dist).item()
        except RuntimeError:
            self.hardest_triplet_dist = 0
            self.hardest_pos_pair_dist = 0
            self.hardest_neg_pair_dist = 0


class DistanceWeightedMiner(BaseTupleMiner):

    def __init__(self, cutoff, nonzero_loss_cutoff, **kwargs):
        super().__init__(**kwargs)
        self.cutoff = float(cutoff)
        self.nonzero_loss_cutoff = float(nonzero_loss_cutoff)
        self.mat_type = 'dist'

    def mine(self, embeddings, labels, ref_emb, ref_labels):
        d = embeddings.size(1)
        dist_mat = lmu.dist_mat(embeddings, ref_emb)
        if embeddings is ref_emb:
            dist_mat = dist_mat + torch.eye(dist_mat.size(0))
        dist_mat = torch.max(dist_mat, torch.tensor(self.cutoff))
        log_weights = (2.0 - float(d)) * torch.log(dist_mat) - float(d - 3) / 2 * torch.log(1.0 - 0.25 * dist_mat ** 2.0)
        weights = torch.exp(log_weights - torch.max(log_weights))
        mask = torch.ones(weights.size())
        same_class = labels.unsqueeze(1) == ref_labels.unsqueeze(0)
        mask[same_class] = 0
        weights = weights * mask * (dist_mat < self.nonzero_loss_cutoff).float()
        weights = weights / torch.sum(weights, dim=1, keepdim=True)
        np_weights = weights.cpu().numpy()
        return lmu.get_random_triplet_indices(labels, weights=np_weights)


class EmbeddingsAlreadyPackagedAsTriplets(BaseTupleMiner):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def mine(self, embeddings, labels, ref_emb, ref_labels):
        batch_size = embeddings.size(0)
        a = torch.arange(0, batch_size, 3)
        p = torch.arange(1, batch_size, 3)
        n = torch.arange(2, batch_size, 3)
        return a, p, n


class HDCMiner(BaseTupleMiner):

    def __init__(self, filter_percentage, use_similarity=False, squared_distances=False, **kwargs):
        super().__init__(**kwargs)
        self.filter_percentage = filter_percentage
        self.use_similarity = use_similarity
        self.squared_distances = squared_distances
        self.reset_idx()

    def mine(self, embeddings, labels, ref_emb, ref_labels):
        mat = lmu.get_pairwise_mat(embeddings, ref_emb, self.use_similarity, self.squared_distances)
        self.set_idx(labels, ref_labels)
        for name, (anchor, other) in {'pos': (self.a1, self.p), 'neg': (self.a2, self.n)}.items():
            if len(anchor) > 0:
                pairs = mat[anchor, other]
                num_pairs = len(pairs)
                k = int(math.ceil(self.filter_percentage * num_pairs))
                largest = self.should_select_largest(name)
                _, idx = torch.topk(pairs, k=k, largest=largest)
                self.filter_original_indices(name, idx)
        return self.a1, self.p, self.a2, self.n

    def should_select_largest(self, name):
        if self.use_similarity:
            return False if name == 'pos' else True
        return True if name == 'pos' else False

    def set_idx(self, labels, ref_labels):
        if not self.was_set_externally:
            self.a1, self.p, self.a2, self.n = lmu.get_all_pairs_indices(labels, ref_labels)

    def set_idx_externally(self, external_indices_tuple, labels):
        self.a1, self.p, self.a2, self.n = lmu.convert_to_pairs(external_indices_tuple, labels)
        self.was_set_externally = True

    def reset_idx(self):
        self.a1, self.p, self.a2, self.n = None, None, None, None
        self.was_set_externally = False

    def filter_original_indices(self, name, idx):
        if name == 'pos':
            self.a1 = self.a1[idx]
            self.p = self.p[idx]
        else:
            self.a2 = self.a2[idx]
            self.n = self.n[idx]


class MaximumLossMiner(BaseSubsetBatchMiner):

    def __init__(self, loss, miner=None, num_trials=5, **kwargs):
        super().__init__(**kwargs)
        self.loss = loss
        self.miner = miner
        self.num_trials = num_trials
        self.add_to_recordable_attributes(list_of_names=['avg_loss', 'max_loss'])

    def mine(self, embeddings, labels, *_):
        losses = []
        all_subset_idx = []
        for i in range(self.num_trials):
            rand_subset_idx = c_f.NUMPY_RANDOM.choice(len(embeddings), size=self.output_batch_size, replace=False)
            rand_subset_idx = torch.from_numpy(rand_subset_idx)
            all_subset_idx.append(rand_subset_idx)
            curr_embeddings, curr_labels = embeddings[rand_subset_idx], labels[rand_subset_idx]
            indices_tuple = self.inner_miner(curr_embeddings, curr_labels)
            losses.append(self.loss(curr_embeddings, curr_labels, indices_tuple).item())
        max_idx = np.argmax(losses)
        self.avg_loss = np.mean(losses)
        self.max_loss = losses[max_idx]
        return all_subset_idx[max_idx]

    def inner_miner(self, embeddings, labels):
        if self.miner:
            return self.miner(embeddings, labels)
        return None


class MultiSimilarityMiner(BaseTupleMiner):

    def __init__(self, epsilon, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def mine(self, embeddings, labels, ref_emb, ref_labels):
        sim_mat = lmu.sim_mat(embeddings, ref_emb)
        a1, p, a2, n = lmu.get_all_pairs_indices(labels, ref_labels)
        if len(a1) == 0 or len(a2) == 0:
            empty = torch.LongTensor([])
            return empty.clone(), empty.clone(), empty.clone(), empty.clone()
        sim_mat_neg_sorting = sim_mat.clone()
        sim_mat_pos_sorting = sim_mat.clone()
        sim_mat_pos_sorting[a2, n] = float('inf')
        sim_mat_neg_sorting[a1, p] = -float('inf')
        if embeddings is ref_emb:
            sim_mat_neg_sorting[range(len(labels)), range(len(labels))] = -float('inf')
        pos_sorted, pos_sorted_idx = torch.sort(sim_mat_pos_sorting, dim=1)
        neg_sorted, neg_sorted_idx = torch.sort(sim_mat_neg_sorting, dim=1)
        hard_pos_idx = (pos_sorted - self.epsilon < neg_sorted[:, (-1)].unsqueeze(1)).nonzero()
        hard_neg_idx = (neg_sorted + self.epsilon > pos_sorted[:, (0)].unsqueeze(1)).nonzero()
        a1 = hard_pos_idx[:, (0)]
        p = pos_sorted_idx[a1, hard_pos_idx[:, (1)]]
        a2 = hard_neg_idx[:, (0)]
        n = neg_sorted_idx[a2, hard_neg_idx[:, (1)]]
        return a1, p, a2, n


class PairMarginMiner(BaseTupleMiner):
    """
    Returns positive pairs that have distance greater than a margin and negative
    pairs that have distance less than a margin
    """

    def __init__(self, pos_margin, neg_margin, use_similarity, squared_distances=False, **kwargs):
        super().__init__(**kwargs)
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.use_similarity = use_similarity
        self.squared_distances = squared_distances
        self.add_to_recordable_attributes(list_of_names=['pos_pair_dist', 'neg_pair_dist'])

    def mine(self, embeddings, labels, ref_emb, ref_labels):
        mat = lmu.get_pairwise_mat(embeddings, ref_emb, self.use_similarity, self.squared_distances)
        a1, p, a2, n = lmu.get_all_pairs_indices(labels, ref_labels)
        pos_pair = mat[a1, p]
        neg_pair = mat[a2, n]
        self.pos_pair_dist = torch.mean(pos_pair).item() if len(pos_pair) > 0 else 0
        self.neg_pair_dist = torch.mean(neg_pair).item() if len(neg_pair) > 0 else 0
        pos_mask_condition = self.pos_filter(pos_pair, self.pos_margin)
        neg_mask_condition = self.neg_filter(neg_pair, self.neg_margin)
        a1 = torch.masked_select(a1, pos_mask_condition)
        p = torch.masked_select(p, pos_mask_condition)
        a2 = torch.masked_select(a2, neg_mask_condition)
        n = torch.masked_select(n, neg_mask_condition)
        return a1, p, a2, n

    def pos_filter(self, pos_pair, margin):
        return pos_pair < margin if self.use_similarity else pos_pair > margin

    def neg_filter(self, neg_pair, margin):
        return neg_pair > margin if self.use_similarity else neg_pair < margin


class TripletMarginMiner(BaseTupleMiner):
    """
    Returns triplets that violate the margin
    Args:
    	margin
    	type_of_triplets: options are "all", "hard", or "semihard".
    		"all" means all triplets that violate the margin
    		"hard" is a subset of "all", but the negative is closer to the anchor than the positive
    		"semihard" is a subset of "all", but the negative is further from the anchor than the positive
    """

    def __init__(self, margin, type_of_triplets='all', **kwargs):
        super().__init__(**kwargs)
        self.margin = margin
        self.add_to_recordable_attributes(list_of_names=['avg_triplet_margin', 'pos_pair_dist', 'neg_pair_dist'])
        self.type_of_triplets = type_of_triplets
        self.idx_type = 'triplet'

    def mine(self, embeddings, labels, ref_emb, ref_labels):
        anchor_idx, positive_idx, negative_idx = lmu.get_all_triplets_indices(labels, ref_labels)
        anchors, positives, negatives = embeddings[anchor_idx], ref_emb[positive_idx], ref_emb[negative_idx]
        ap_dist = torch.nn.functional.pairwise_distance(anchors, positives, 2)
        an_dist = torch.nn.functional.pairwise_distance(anchors, negatives, 2)
        triplet_margin = ap_dist - an_dist
        self.pos_pair_dist = torch.mean(ap_dist).item()
        self.neg_pair_dist = torch.mean(an_dist).item()
        self.avg_triplet_margin = torch.mean(-triplet_margin).item()
        threshold_condition = triplet_margin > -self.margin
        if self.type_of_triplets == 'hard':
            threshold_condition &= an_dist < ap_dist
        elif self.type_of_triplets == 'semihard':
            threshold_condition &= an_dist > ap_dist
        return anchor_idx[threshold_condition], positive_idx[threshold_condition], negative_idx[threshold_condition]


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
        c_f.add_to_recordable_attributes(self, name=name, list_of_names=list_of_names)


class CenterInvariantRegularizer(BaseWeightRegularizer):

    def __init__(self, normalize_weights=False):
        super().__init__(normalize_weights)
        assert not self.normalize_weights, 'normalize_weights must be False for CenterInvariantRegularizer'

    def compute_loss(self, weights):
        squared_weight_norms = self.weight_norms ** 2
        deviations_from_mean = squared_weight_norms - torch.mean(squared_weight_norms)
        return torch.mean(deviations_from_mean ** 2 / 4)


class RegularFaceRegularizer(BaseWeightRegularizer):

    def compute_loss(self, weights):
        num_classes = weights.size(0)
        cos = torch.mm(weights, weights.t())
        if not self.normalize_weights:
            norms = self.weight_norms.unsqueeze(1)
            cos = cos / (norms * norms.t())
        cos1 = cos.clone()
        with torch.no_grad():
            row_nums = torch.arange(num_classes).long()
            cos1[row_nums, row_nums] = -float('inf')
            _, indices = torch.max(cos1, dim=1)
            mask = torch.zeros((num_classes, num_classes))
            mask[row_nums, indices] = 1
        return torch.sum(cos * mask) / num_classes


class Identity(torch.nn.Module):

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
]

class Test_KevinMusgrave_pytorch_metric_learning(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

