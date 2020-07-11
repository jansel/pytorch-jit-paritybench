import sys
_module = sys.modules[__name__]
del sys
conf = _module
kiwi = _module
cli = _module
better_argparse = _module
main = _module
models = _module
linear = _module
nuqe = _module
predictor = _module
predictor_estimator = _module
quetch = _module
opts = _module
pipelines = _module
evaluate = _module
jackknife = _module
predict = _module
train = _module
constants = _module
data = _module
builders = _module
corpus = _module
fields = _module
alignment_field = _module
qe_field = _module
sequence_labels_field = _module
fieldsets = _module
extend_vocabs_fieldset = _module
fieldset = _module
predictor_estimator = _module
iterators = _module
qe_dataset = _module
tokenizers = _module
utils = _module
vectors = _module
vocabulary = _module
lib = _module
jackknife = _module
search = _module
train = _module
utils = _module
loggers = _module
metrics = _module
functions = _module
metrics = _module
stats = _module
label_dictionary = _module
linear_model = _module
linear_trainer = _module
linear_word_qe_decoder = _module
linear_word_qe_features = _module
linear_word_qe_sentence = _module
sequence_parts = _module
sparse_feature_vector = _module
sparse_vector = _module
structured_classifier = _module
structured_decoder = _module
linear_word_qe_classifier = _module
model = _module
modules = _module
attention = _module
scorer = _module
nuqe = _module
predictor = _module
predictor_estimator = _module
quetch = _module
utils = _module
predictors = _module
linear_tester = _module
predictor = _module
trainers = _module
callbacks = _module
linear_word_qe_trainer = _module
trainer = _module
utils = _module
convert_parsed_to_postags = _module
extract_columns = _module
merge_target_and_gaps_preds = _module
stack_probabilities_for_linear = _module
conftest = _module
test_data_builders = _module
test_linear = _module
test_metrics = _module
test_model = _module
test_model_config = _module
test_nuqe = _module
test_nuqe_18 = _module
test_predest = _module
test_predictor = _module
test_qe_dataset = _module
test_quetch = _module
test_quetch_18 = _module
test_write_read_config_files = _module

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


from torchtext import data


import copy


import logging


from collections import defaultdict


from math import ceil


from functools import partial


from torchtext.vocab import Vectors


import warnings


import torchtext


import numpy as np


import random


from time import gmtime


import math


import time


from collections import OrderedDict


from scipy.stats.stats import pearsonr


from scipy.stats.stats import spearmanr


from torch import nn


from abc import ABCMeta


from abc import abstractmethod


import torch.nn as nn


import torch.nn.functional as F


from torch.distributions.normal import Normal


from torch.autograd import Function


from torch.nn.utils.rnn import pack_padded_sequence as pack


from torch.nn.utils.rnn import pad_packed_sequence as unpack


from torchtext.data import Example


from torch import optim


import collections


from types import SimpleNamespace


class ModelConfig:
    __metaclass__ = ABCMeta

    def __init__(self, vocabs):
        """Model Configuration Base Class.

        Args:
        vocabs: Dictionary Mapping Field Names to Vocabularies.
                Must contain 'source' and 'target' keys
        """
        self.source_vocab_size = len(vocabs[const.SOURCE])
        self.target_vocab_size = len(vocabs[const.TARGET])

    @classmethod
    def from_dict(cls, config_dict, vocabs):
        """Create config from a saved state_dict.
           Args:
             config_dict: A dictionary that is the return value of
                          a call to the `state_dict()` method of `cls`
             vocab: See `ModelConfig.__init__`
        """
        config = cls(vocabs)
        config.update(config_dict)
        return config

    def update(self, other_config):
        """Updates the config object with the values of `other_config`
           Args:
             other_config: The `dict` or `ModelConfig` object to update with.
        """
        config_dict = dict()
        if isinstance(self, other_config.__class__):
            config_dict = other_config.__dict__
        elif isinstance(other_config, dict):
            config_dict = other_config
        self.__dict__.update(config_dict)

    def state_dict(self):
        """Return the __dict__ for serialization.
        """
        self.__dict__['__version__'] = kiwi.__version__
        return self.__dict__


def load_torch_file(file_path):
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError('Torch file not found: {}'.format(file_path))
    file_dict = torch.load(str(file_path), map_location=lambda storage, loc: storage)
    if isinstance(file_dict, Path):
        linked_path = file_dict
        if not linked_path.exists():
            relative_path = file_path.with_name(file_dict.name) / const.MODEL_FILE
            if relative_path.exists():
                linked_path = relative_path
        return load_torch_file(linked_path)
    return file_dict


class Attention(nn.Module):
    """Generic Attention Implementation.
       Module computes a convex combination of a set of values based on the fit
       of their keys with a query.
    """

    def __init__(self, scorer):
        super().__init__()
        self.scorer = scorer
        self.mask = None

    def forward(self, query, keys, values=None):
        if values is None:
            values = keys
        scores = self.scorer(query, keys)
        scores = scores - scores.mean(1, keepdim=True)
        scores = torch.exp(scores)
        if self.mask is not None:
            scores = self.mask * scores
        convex = scores / scores.sum(1, keepdim=True)
        return torch.einsum('bs,bsi->bi', [convex, values])

    def set_mask(self, mask):
        self.mask = mask


class Scorer(nn.Module):
    """Score function for Attention module.
    """

    def __init__(self):
        super().__init__()

    def forward(self, query, keys):
        """Computes Scores for each key given the query.
           args:
                 query:  FloatTensor batch x n
                 keys:   FloatTensor batch x seq_length x m
           ret:
                 scores: FloatTensor batch x seq_length
        """
        raise NotImplementedError


class MLPScorer(Scorer):
    """Implements a score function based on a Multilayer Perceptron.
    """

    def __init__(self, query_size, key_size, layers=2, nonlinearity=nn.Tanh):
        super().__init__()
        layer_list = []
        size = query_size + key_size
        for i in range(layers):
            size_next = size // 2 if i < layers - 1 else 1
            layer_list.append(nn.Sequential(nn.Linear(size, size_next), nonlinearity()))
            size = size_next
        self.layers = nn.ModuleList(layer_list)

    def forward(self, query, keys):
        layer_in = torch.cat([query.unsqueeze(1).expand_as(keys), keys], dim=-1)
        layer_in = layer_in.reshape(-1, layer_in.size(-1))
        for layer in self.layers:
            layer_in = layer(layer_in)
        out = layer_in.reshape(keys.size()[:-1])
        return out


def replace_token(target, old, new):
    """Replaces old tokens with new.

    args: target (LongTensor)
          old (int): The token to be replaced by new
          new (int): The token used to replace old

    """
    return target.masked_fill(target == old, new)


class Metric:

    def __init__(self, target_name=None, metric_name=None, PAD=None, STOP=None, prefix=None):
        super().__init__()
        self.reset()
        self.prefix = prefix
        self.target_name = target_name
        self.metric_name = metric_name
        self.PAD = PAD
        self.STOP = STOP

    def update(self, **kwargs):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def summarize(self, **kwargs):
        raise NotImplementedError

    def get_name(self):
        return self._prefix(self.metric_name)

    def _prefix_keys(self, summary):
        if self.prefix:
            summary = OrderedDict({self._prefix(key): value for key, value in summary.items()})
        return summary

    def _prefix(self, key):
        if self.prefix:
            return '{}_{}'.format(self.prefix, key)
        return key

    def token_mask(self, batch):
        target = self.get_target(batch)
        if self.PAD is not None:
            return target != self.PAD
        else:
            return torch.ones(target.shape, dtype=torch.uint8, device=target.device)

    def get_target(self, batch):
        target = getattr(batch, self.target_name)
        if self.STOP is not None:
            target = replace_token(target[:, 1:-1], self.STOP, self.PAD)
        return target

    def get_token_indices(self, batch):
        mask = self.token_mask(batch)
        return mask.view(-1).nonzero().squeeze()

    def get_predictions(self, model_out):
        predictions = model_out[self.target_name]
        return predictions

    def get_target_flat(self, batch):
        target_flat = self.get_target(batch).contiguous().view(-1)
        token_indices = self.get_token_indices(batch)
        return target_flat[token_indices]

    def get_predictions_flat(self, model_out, batch):
        predictions = self.get_predictions(model_out).contiguous()
        predictions_flat = predictions.view(-1, predictions.shape[-1]).squeeze()
        token_indices = self.get_token_indices(batch)
        return predictions_flat[token_indices]

    def get_tokens(self, batch):
        return self.token_mask(batch).sum().item()


class CorrectMetric(Metric):

    def __init__(self, **kwargs):
        super().__init__(metric_name='CORRECT', **kwargs)

    def update(self, model_out, batch, **kwargs):
        self.tokens += self.get_tokens(batch)
        logits = self.get_predictions_flat(model_out, batch)
        target = self.get_target_flat(batch)
        _, pred = logits.max(-1)
        correct = target == pred
        correct_count = correct.sum().item()
        self.correct += correct_count

    def summarize(self):
        summary = {self.metric_name: float(self.correct) / self.tokens}
        return self._prefix_keys(summary)

    def reset(self):
        self.correct = 0
        self.tokens = 0


class ExpectedErrorMetric(Metric):

    def __init__(self, **kwargs):
        super().__init__(metric_name='ExpErr', **kwargs)

    def update(self, model_out, batch, **kwargs):
        logits = self.get_predictions_flat(model_out, batch)
        target = self.get_target_flat(batch)
        probs = nn.functional.softmax(logits, -1)
        probs = probs.gather(-1, target.unsqueeze(-1)).squeeze()
        errors = 1.0 - probs
        self.tokens += self.get_tokens(batch)
        self.expected_error += errors.sum().item()

    def summarize(self):
        summary = {self.metric_name: self.expected_error / self.tokens}
        return self._prefix_keys(summary)

    def reset(self):
        self.expected_error = 0.0
        self.tokens = 0


class PerplexityMetric(Metric):

    def __init__(self, **kwargs):
        super().__init__(metric_name='PERP', **kwargs)

    def reset(self):
        self.tokens = 0
        self.nll = 0.0

    def update(self, loss, batch, **kwargs):
        self.tokens += self.get_tokens(batch)
        self.nll += loss[self.target_name].item()

    def summarize(self):
        summary = {self.metric_name: math.e ** (self.nll / self.tokens)}
        return self._prefix_keys(summary)


class PredictorConfig(ModelConfig):

    def __init__(self, vocabs, hidden_pred=400, rnn_layers_pred=3, dropout_pred=0.0, share_embeddings=False, embedding_sizes=0, target_embeddings_size=200, source_embeddings_size=200, out_embeddings_size=200, predict_inverse=False):
        """Predictor Hyperparams.
        """
        super().__init__(vocabs)
        self.target_side = const.TARGET
        self.source_side = const.SOURCE
        self.predict_inverse = predict_inverse
        if self.predict_inverse:
            self.source_side, self.target_side = self.target_side, self.source_side
            self.target_vocab_size, self.source_vocab_size = self.source_vocab_size, self.target_vocab_size
        self.hidden_pred = hidden_pred
        self.rnn_layers_pred = rnn_layers_pred
        self.dropout_pred = dropout_pred
        self.share_embeddings = share_embeddings
        if embedding_sizes:
            self.target_embeddings_size = embedding_sizes
            self.source_embeddings_size = embedding_sizes
            self.out_embeddings_size = embedding_sizes
        else:
            self.target_embeddings_size = target_embeddings_size
            self.source_embeddings_size = source_embeddings_size
            self.out_embeddings_size = out_embeddings_size


def apply_packed_sequence(rnn, embedding, lengths):
    """ Runs a forward pass of embeddings through an rnn using packed sequence.
    Args:
       rnn: The RNN that that we want to compute a forward pass with.
       embedding (FloatTensor b x seq x dim): A batch of sequence embeddings.
       lengths (LongTensor batch): The length of each sequence in the batch.

    Returns:
       output: The output of the RNN `rnn` with input `embedding`
    """
    lengths_sorted, permutation = torch.sort(lengths, descending=True)
    embedding_sorted = embedding[permutation]
    embedding_packed = pack(embedding_sorted, lengths_sorted, batch_first=True)
    outputs_packed, (hidden, cell) = rnn(embedding_packed)
    outputs_sorted, _ = unpack(outputs_packed, batch_first=True)
    _, permutation_rev = torch.sort(permutation, descending=False)
    outputs = outputs_sorted[permutation_rev]
    hidden, cell = hidden[:, (permutation_rev)], cell[:, (permutation_rev)]
    return outputs, (hidden, cell)


class EstimatorConfig(PredictorConfig):

    def __init__(self, vocabs, hidden_est=100, rnn_layers_est=1, mlp_est=True, dropout_est=0.0, start_stop=False, predict_target=True, predict_gaps=False, predict_source=False, token_level=True, sentence_level=True, sentence_ll=True, binary_level=True, target_bad_weight=2.0, source_bad_weight=2.0, gaps_bad_weight=2.0, **kwargs):
        """Predictor Estimator Hyperparams.
        """
        super().__init__(vocabs, **kwargs)
        self.start_stop = start_stop or predict_gaps
        self.hidden_est = hidden_est
        self.rnn_layers_est = rnn_layers_est
        self.mlp_est = mlp_est
        self.dropout_est = dropout_est
        self.predict_target = predict_target
        self.predict_gaps = predict_gaps
        self.predict_source = predict_source
        self.token_level = token_level
        self.sentence_level = sentence_level
        self.sentence_ll = sentence_ll
        self.binary_level = binary_level
        self.target_bad_weight = target_bad_weight
        self.source_bad_weight = source_bad_weight
        self.gaps_bad_weight = gaps_bad_weight


def confusion_matrix(hat_y, y, n_classes=None):
    hat_y = np.array(list(collapse(hat_y)))
    y = np.array(list(collapse(y)))
    if n_classes is None:
        classes = np.unique(np.union1d(hat_y, y))
        n_classes = len(classes)
    cnfm = np.zeros((n_classes, n_classes))
    for j in range(y.shape[0]):
        cnfm[y[j], hat_y[j]] += 1
    return cnfm


def precision(tp, fp, fn):
    if tp + fp > 0:
        return tp / (tp + fp)
    return 0


def recall(tp, fp, fn):
    if tp + fn > 0:
        return tp / (tp + fn)
    return 0


def fscore(tp, fp, fn):
    p = precision(tp, fp, fn)
    r = recall(tp, fp, fn)
    if p + r > 0:
        return 2 * (p * r) / (p + r)
    return 0


def scores_for_class(class_index, cnfm):
    tp = cnfm[class_index, class_index]
    fp = cnfm[:, (class_index)].sum() - tp
    fn = cnfm[(class_index), :].sum() - tp
    tn = cnfm.sum() - tp - fp - fn
    p = precision(tp, fp, fn)
    r = recall(tp, fp, fn)
    f1 = fscore(tp, fp, fn)
    support = tp + tn
    return p, r, f1, support


def precision_recall_fscore_support(hat_y, y, labels=None):
    n_classes = len(labels) if labels else None
    cnfm = confusion_matrix(hat_y, y, n_classes)
    if n_classes is None:
        n_classes = cnfm.shape[0]
    scores = np.zeros((n_classes, 4))
    for class_id in range(n_classes):
        scores[class_id] = scores_for_class(class_id, cnfm)
    return scores.T.tolist()


class F1Metric(Metric):

    def __init__(self, labels, **kwargs):
        super().__init__(metric_name='F1_MULT', **kwargs)
        self.labels = labels

    def update(self, model_out, batch, **kwargs):
        logits = self.get_predictions_flat(model_out, batch)
        target = self.get_target_flat(batch)
        _, y_hat = logits.max(-1)
        self.Y_HAT += y_hat.tolist()
        self.Y += target.tolist()

    def summarize(self):
        summary = OrderedDict()
        _, _, f1, _ = precision_recall_fscore_support(self.Y_HAT, self.Y)
        summary[self.metric_name] = np.prod(f1)
        for i, label in enumerate(self.labels):
            summary['F1_' + label] = f1[i]
        return self._prefix_keys(summary)

    def reset(self):
        self.Y = []
        self.Y_HAT = []


class LogMetric(Metric):
    """Logs averages of values in loss, model or batch.
    """

    def __init__(self, targets, metric_name=None, **kwargs):
        self.targets = targets
        metric_name = metric_name or self._format(*targets[0])
        super().__init__(metric_name=metric_name, **kwargs)

    def update(self, **kwargs):
        self.steps += 1
        for side, target in self.targets:
            key = self._format(side, target)
            self.log[key] += kwargs[side][target].mean().item()

    def summarize(self):
        summary = {key: (value / float(self.steps)) for key, value in self.log.items()}
        return self._prefix_keys(summary)

    def reset(self):
        self.log = {self._format(side, target): (0.0) for side, target in self.targets}
        self.steps = 0

    def _format(self, side, target):
        return '{}_{}'.format(side, target)


class PearsonMetric(Metric):

    def __init__(self, **kwargs):
        super().__init__(metric_name='PEARSON', **kwargs)

    def reset(self):
        self.predictions = []
        self.target = []

    def update(self, model_out, batch, **kwargs):
        target = self.get_target_flat(batch)
        predictions = self.get_predictions_flat(model_out, batch)
        self.predictions += predictions.tolist()
        self.target += target.tolist()

    def summarize(self):
        pearson = pearsonr(self.predictions, self.target)[0]
        summary = {self.metric_name: pearson}
        return self._prefix_keys(summary)


class RMSEMetric(Metric):

    def __init__(self, **kwargs):
        super().__init__(metric_name='RMSE', **kwargs)

    def update(self, batch, model_out, **kwargs):
        predictions = self.get_predictions_flat(model_out, batch)
        target = self.get_target_flat(batch)
        self.squared_error += ((predictions - target) ** 2).sum().item()
        self.tokens += self.get_tokens(batch)

    def summarize(self):
        rmse = math.sqrt(self.squared_error / self.tokens)
        summary = {self.metric_name: rmse}
        return self._prefix_keys(summary)

    def reset(self):
        self.squared_error = 0.0
        self.tokens = 0


class SpearmanMetric(Metric):

    def __init__(self, **kwargs):
        super().__init__(metric_name='SPEARMAN', **kwargs)

    def reset(self):
        self.predictions = []
        self.target = []

    def update(self, model_out, batch, **kwargs):
        target = self.get_target_flat(batch)
        predictions = self.get_predictions_flat(model_out, batch)
        self.predictions += predictions.tolist()
        self.target += target.tolist()

    def summarize(self):
        spearman = spearmanr(self.predictions, self.target)[0]
        summary = {self.metric_name: spearman}
        return self._prefix_keys(summary)


class MovingMetric:
    """Class to compute the changes in one metric as a function of a second metric.
       Example: F1 score vs. Classification Threshold, Quality vs Skips
    """

    def eval(self, scores, labels):
        """Compute the graph metric1 vs metric2
        Args:
           Scores: Model Outputs
           Labels: Corresponding Labels
        """
        self.init(scores, labels)
        scores, labels = self.sort(scores, labels)
        init_threshold = scores[0]
        thresholds = [(self.compute(), init_threshold)]
        for score, label in zip(scores, labels):
            self.update(score, label)
            thresholds.append((self.compute(), score))
        return thresholds

    def init(self, scores, labels):
        """Initialize the Metric for threshold < min(scores)
        """
        return scores, labels

    def sort(self, scores, labels):
        """Sort List of labels and scores.
        """
        return zip(*sorted(zip(scores, labels)))

    def update(self, score, label):
        """Move the threshold past score
        """
        return None

    def compute(self):
        """Compute the current Value of the metric
        """
        pass

    def choose(self, thresholds):
        """Choose the best (threshold, metric) tuple from an iterable.
        """
        pass


class MovingF1(MovingMetric):

    def init(self, scores, labels, class_idx=1):
        """
        Compute F1 Mult for all decision thresholds over (scores, labels)
        Initialize the threshold s.t. all examples are classified as
        `class_idx`.
        Args:
           scores: Likelihood scores for class index
           Labels: Gold Truth classes in {0,1}
           class_index: ID of class
        """
        self.sign = 2 * class_idx - 1
        class_one = sum(labels)
        class_zero = len(labels) - class_one
        self.fp_zero = (1 - class_idx) * class_one
        self.tp_zero = (1 - class_idx) * class_zero
        self.fp_one = class_idx * class_zero
        self.tp_one = class_idx * class_one

    def update(self, score, label):
        """Move the decision threshold.
        """
        self.tp_zero += self.sign * (1 - label)
        self.fp_zero += self.sign * label
        self.tp_one -= self.sign * label
        self.fp_one -= self.sign * (1 - label)

    def compute(self):
        f1_zero = fscore(self.tp_zero, self.fp_zero, self.fp_one)
        f1_one = fscore(self.tp_one, self.fp_one, self.fp_zero)
        return f1_one * f1_zero

    def choose(self, thresholds):
        return max(thresholds)


class ThresholdCalibrationMetric(Metric):

    def __init__(self, **kwargs):
        super().__init__(metric_name='F1_CAL', **kwargs)

    def update(self, model_out, batch, **kwargs):
        logits = self.get_predictions_flat(model_out, batch)
        bad_probs = nn.functional.softmax(logits, -1)[:, (const.BAD_ID)]
        target = self.get_target_flat(batch)
        self.scores += bad_probs.tolist()
        self.Y += target.tolist()

    def summarize(self):
        summary = {}
        mid = len(self.Y) // 2
        if mid:
            perm = np.random.permutation(len(self.Y))
            self.Y = [self.Y[idx] for idx in perm]
            self.scores = [self.scores[idx] for idx in perm]
            m = MovingF1()
            fscore, threshold = m.choose(m.eval(self.scores[:mid], self.Y[:mid]))
            predictions = [(const.BAD_ID if score >= threshold else const.OK_ID) for score in self.scores[mid:]]
            _, _, f1, _ = precision_recall_fscore_support(predictions, self.Y[mid:])
            f1_mult = np.prod(f1)
            summary = {self.metric_name: f1_mult}
        return self._prefix_keys(summary)

    def reset(self):
        self.scores = []
        self.Y = []


def make_loss_weights(nb_classes, target_idx, weight):
    """Creates a loss weight vector for nn.CrossEntropyLoss

    args:
        nb_classes: Number of classes
        target_idx: ID of the target (reweighted) class
        weight: Weight of the target class

    returns:
       weights (FloatTensor): Weight Tensor of shape `nb_classes` such that
                                  `weights[target_idx] = weight`
                                  `weights[other_idx] = 1.0`
    """
    weights = torch.ones(nb_classes)
    weights[target_idx] = weight
    return weights


class QUETCHConfig(ModelConfig):

    def __init__(self, vocabs, predict_target=True, predict_gaps=False, predict_source=False, source_embeddings_size=50, target_embeddings_size=50, hidden_sizes=None, bad_weight=3.0, window_size=10, max_aligned=5, dropout=0.4, embeddings_dropout=0.4, freeze_embeddings=False):
        super().__init__(vocabs)
        if hidden_sizes is None:
            hidden_sizes = [100]
        source_vectors = vocabs[const.SOURCE].vectors
        target_vectors = vocabs[const.TARGET].vectors
        if source_vectors is not None:
            source_embeddings_size = source_vectors.size(1)
        if target_vectors is not None:
            target_embeddings_size = target_vectors.size(1)
        self.source_embeddings_size = source_embeddings_size
        self.target_embeddings_size = target_embeddings_size
        self.bad_weight = bad_weight
        self.dropout = dropout
        self.embeddings_dropout = embeddings_dropout
        self.freeze_embeddings = freeze_embeddings
        if predict_gaps or predict_source:
            predict_target = predict_target
        self.predict_target = predict_target
        self.predict_gaps = predict_gaps
        self.predict_source = predict_source
        self.window_size = window_size
        self.max_aligned = max_aligned
        self.hidden_sizes = hidden_sizes
        if const.SOURCE_TAGS in vocabs:
            self.tags_pad_id = vocabs[const.SOURCE_TAGS].stoi[const.PAD]
        elif const.GAP_TAGS in vocabs:
            self.tags_pad_id = vocabs[const.GAP_TAGS].stoi[const.PAD]
        else:
            self.tags_pad_id = vocabs[const.TARGET_TAGS].stoi[const.PAD]
        self.nb_classes = len(const.LABELS)
        self.tag_bad_index = const.BAD_ID
        self.pad_token = const.PAD
        self.unaligned_idx = const.UNALIGNED_ID
        self.source_padding_idx = const.PAD_ID
        self.target_padding_idx = const.PAD_ID


def align_source(source, trg2src_alignments, max_aligned, unaligned_idx, padding_idx, pad_size):
    assert len(source.shape) == 2
    window_size = source.shape[1]
    assert len(trg2src_alignments) <= pad_size
    aligned_source = source.new_full((pad_size, max_aligned, window_size), padding_idx)
    unaligned = source.new_full((window_size,), unaligned_idx)
    nb_alignments = source.new_ones(pad_size, dtype=torch.float)
    for i, source_positions in enumerate(trg2src_alignments):
        if not source_positions:
            aligned_source[i, 0] = unaligned
        else:
            selected = torch.index_select(source, 0, torch.tensor(source_positions[:max_aligned], device=source.device))
            aligned_source[(i), :len(selected)] = selected
            nb_alignments[i] = len(selected)
    return aligned_source, nb_alignments


def map_alignments_to_target(src2tgt_alignments, target_length=None):
    """Maps a target index to a list of source indexes.

    Args:
        src2tgt_alignments (list): list of tuples with source, target indexes.
        target_length: size of the target side; if None, the highest index
            in the alignments is used.

    Returns:
        A list of size target_length where position i refers to the i-th
        target token and contains a list of source indexes aligned to it.

    """
    if target_length is None:
        if not src2tgt_alignments:
            target_length = 0
        else:
            target_length = 1 + max(src2tgt_alignments, key=lambda a: a[1])[1]
    trg2src = [None] * target_length
    for source, target in src2tgt_alignments:
        if not trg2src[target]:
            trg2src[target] = []
        trg2src[target].append(source)
    return trg2src


def align_tensor(tensor, alignments, max_aligned, unaligned_idx, padding_idx, pad_size, target_length=None):
    alignments = [map_alignments_to_target(sample, target_length=target_length) for sample in alignments]
    aligned = [align_source(sample, alignment, max_aligned, unaligned_idx, padding_idx, pad_size) for sample, alignment in zip(tensor, alignments)]
    aligned_tensor = torch.stack([sample[0] for sample in aligned])
    nb_alignments = torch.stack([sample[1] for sample in aligned])
    return aligned_tensor, nb_alignments


logger = logging.getLogger(__name__)


PAD = '<pad>'


START = '<bos>'


STOP = '<eos>'


UNK = '<unk>'


def map_to_polyglot(token):
    mapping = {UNK: '<UNK>', PAD: '<PAD>', START: '<S>', STOP: '</S>'}
    if token in mapping:
        return mapping[token]
    return token


class Fieldset:
    ALL = 'all'
    TRAIN = 'train'
    VALID = 'valid'
    TEST = 'test'

    def __init__(self):
        """

        """
        self._fields = {}
        self._options = {}
        self._required = {}
        self._vocab_options = {}
        self._vocab_vectors = {}
        self._file_reader = {}

    def add(self, name, field, file_option_suffix, required=ALL, vocab_options=None, vocab_vectors=None, file_reader=None):
        """

        Args:
            name:
            field:
            file_option_suffix:
            required (str or list or None):
            file_reader (callable): by default, uses Corpus.from_files().

        Returns:

        """
        self._fields[name] = field
        self._options[name] = file_option_suffix
        if not isinstance(required, list):
            required = [required]
        self._required[name] = required
        self._file_reader[name] = file_reader
        if vocab_options is None:
            vocab_options = {}
        self._vocab_options[name] = vocab_options
        self._vocab_vectors[name] = vocab_vectors

    @property
    def fields(self):
        return self._fields

    def is_required(self, name, set_name):
        required = self._required[name]
        if set_name in required or self.ALL in required:
            return True
        else:
            return False

    def fields_and_files(self, set_name, **files_options):
        fields = {}
        files = {}
        for name, file_option_suffix in self._options.items():
            file_option = '{}{}'.format(set_name, file_option_suffix)
            file_name = files_options.get(file_option)
            if not file_name and self.is_required(name, set_name):
                raise FileNotFoundError('File {} is required (use the {} option).'.format(file_name, file_option.replace('_', '-')))
            elif file_name:
                files[name] = {'name': file_name, 'reader': self._file_reader.get(name)}
                fields[name] = self._fields[name]
        return fields, files

    def vocab_kwargs(self, name, **kwargs):
        if name not in self._vocab_options:
            raise KeyError('Field named "{}" does not exist in this fieldset'.format(name))
        vkwargs = {}
        for argument, option_name in self._vocab_options[name].items():
            option_value = kwargs.get(option_name)
            if option_value is not None:
                vkwargs[argument] = option_value
        return vkwargs

    def vocab_vectors_loader(self, name, embeddings_format='polyglot', embeddings_binary=False, **kwargs):
        if name not in self._vocab_vectors:
            raise KeyError('Field named "{}" does not exist in this fieldset'.format(name))

        def no_vectors_fn():
            return None
        vectors_fn = no_vectors_fn
        option_name = self._vocab_vectors[name]
        if option_name:
            option_value = kwargs.get(option_name)
            if option_value:
                emb_model = AvailableVectors[embeddings_format]
                vectors_fn = partial(emb_model, option_value, binary=embeddings_binary)
        return vectors_fn

    def vocab_vectors(self, name, **kwargs):
        vectors_fn = self.vocab_vectors_loader(name, **kwargs)
        return vectors_fn()

    def fields_vocab_options(self, **kwargs):
        vocab_options = {}
        for name, field in self.fields.items():
            vocab_options[name] = dict(vectors_fn=self.vocab_vectors_loader(name, **kwargs))
            vocab_options[name].update(self.vocab_kwargs(name, **kwargs))
        return vocab_options


def tokenizer(sentence):
    """Implement your own tokenize procedure."""
    return sentence.strip().split()


def build_label_field(postprocessing=None):
    return SequenceLabelsField(classes=const.LABELS, tokenize=tokenizer, pad_token=const.PAD, batch_first=True, postprocessing=postprocessing)


def build_text_field():
    return data.Field(tokenize=tokenizer, init_token=const.START, batch_first=True, eos_token=const.STOP, pad_token=const.PAD, unk_token=const.UNK)


def build_fieldset(wmt18_format=False):
    target_field = build_text_field()
    source_field = build_text_field()
    source_vocab_options = dict(min_freq='source_vocab_min_frequency', max_size='source_vocab_size')
    target_vocab_options = dict(min_freq='target_vocab_min_frequency', max_size='target_vocab_size')
    fieldset = Fieldset()
    fieldset.add(name=const.SOURCE, field=source_field, file_option_suffix='_source', required=Fieldset.TRAIN, vocab_options=source_vocab_options)
    fieldset.add(name=const.TARGET, field=target_field, file_option_suffix='_target', required=Fieldset.TRAIN, vocab_options=target_vocab_options)
    fieldset.add(name=const.PE, field=target_field, file_option_suffix='_pe', required=None, vocab_options=target_vocab_options)
    post_pipe_target = data.Pipeline(utils.project)
    if wmt18_format:
        post_pipe_gaps = data.Pipeline(utils.wmt18_to_gaps)
        post_pipe_target = data.Pipeline(utils.wmt18_to_target)
        fieldset.add(name=const.GAP_TAGS, field=build_label_field(post_pipe_gaps), file_option_suffix='_target_tags', required=None)
    fieldset.add(name=const.TARGET_TAGS, field=build_label_field(post_pipe_target), file_option_suffix='_target_tags', required=None)
    fieldset.add(name=const.SOURCE_TAGS, field=build_label_field(), file_option_suffix='_source_tags', required=None)
    fieldset.add(name=const.SENTENCE_SCORES, field=data.Field(sequential=False, use_vocab=False, dtype=torch.float32), file_option_suffix='_sentence_scores', required=None)
    pipe = data.Pipeline(utils.hter_to_binary)
    fieldset.add(name=const.BINARY, field=data.Field(sequential=False, use_vocab=False, dtype=torch.long, preprocessing=pipe), file_option_suffix='_sentence_scores', required=None)
    return fieldset


def convolve_tensor(sequences, window_size, pad_value=0):
    """Convolve a sequence and apply padding

    :param sequence: 2D tensor
    :param window_size: filter length
    :param pad_value: int value used as padding
    :return: 3D tensor, where the last dimension has size window_size
    """
    pad = (window_size // 2,) * 2
    t = F.pad(sequences, pad=pad, value=pad_value)
    t = t.unfold(1, window_size, 1)
    return t


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (MLPScorer,
     lambda: ([], {'query_size': 4, 'key_size': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_Unbabel_OpenKiwi(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

