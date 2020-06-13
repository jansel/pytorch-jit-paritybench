import sys
_module = sys.modules[__name__]
del sys
master = _module
dataset = _module
dbengine = _module
table = _module
dcparser = _module
constraint = _module
detect = _module
detector = _module
errorloaderdetector = _module
nulldetector = _module
violationdetector = _module
domain = _module
estimator = _module
estimators = _module
logistic = _module
naive_bayes = _module
evaluate = _module
eval = _module
holoclean_repair_example = _module
holoclean_repair_example_db = _module
holoclean = _module
repair = _module
featurize = _module
constraintfeat = _module
featurized_dataset = _module
featurizer = _module
freqfeat = _module
initattrfeat = _module
initsimfeat = _module
langmodelfeat = _module
occurattrfeat = _module
learn = _module
learn = _module
tests = _module
test_errorsloaderdetector = _module
test_holoclean_repair = _module
testutils = _module
utils = _module

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


from abc import ABCMeta


from abc import abstractmethod


import logging


import torch


from torch.optim import Adam


from torch.optim import SGD


from torch.utils.data import TensorDataset


from torch.utils.data import DataLoader


from string import Template


from functools import partial


import torch.nn.functional as F


from collections import namedtuple


import numpy as np


import math


from torch import optim


from torch.autograd import Variable


from torch.nn import Parameter


from torch.nn import ParameterList


from torch.nn.functional import softmax


from torch.optim.lr_scheduler import ReduceLROnPlateau


NULL_REPR = '_nan_'


class Estimator:
    """
    Estimator is an abstract class for posterior estimators that estimate
    the posterior of p(value | other values) for the purpose of domain generation
    and weak labelling.
    """
    __metaclass__ = ABCMeta

    def __init__(self, env, dataset):
        """
        :param env: (dict) dict containing environment/parameters settings.
        :param dataset: (Dataset)
        """
        self.env = env
        self.ds = dataset
        self.attrs = self.ds.get_attributes()

    @abstractmethod
    def train(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def predict_pp(self, row, attr, values):
        """
        :param row: (namedtuple) current values of the target row.
        :param attr: (str) attribute of row (i.e. cell) to generate posteriors for.
        :param values: (list[str]) list of values (for this attr) to generate posteriors for.

        :return: iterator of tuples (value, proba) for each value in :param values:
        """
        raise NotImplementedError

    @abstractmethod
    def predict_pp_batch(self):
        """
        predict_pp_batch is like predict_pp but with a batch of cells.

        :return: iterator of iterator of tuples (value, proba) (one iterator per cell/row in cell_domain_rows)
        """
        raise NotImplementedError


class Featurizer:
    """
    Feauturizer is an abstract class for featurizers that is able to generate
    real-valued tensors (features) for a row from raw data.
    Used in Logistic model.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def setup(self):
        raise NotImplementedError

    @abstractmethod
    def num_features(self):
        raise NotImplementedError

    @abstractmethod
    def create_tensor(self, row, attr, values):
        raise NotImplementedError


NA_COOCCUR_FV = 0


class CooccurAttrFeaturizer(Featurizer):
    """
    CooccurAttrFeaturizer computes the co-occurrence statistics for a cell
    and its possible domain values with the other initial values in the tuple.
    It breaks down each co-occurrence feature on a pairwise attr1 X attr2 basis.
    """
    name = 'CooccurAttrFeaturizer'

    def __init__(self, dataset):
        """
        :param data_df: (pandas.DataFrame) contains the data to compute co-occurrence features for.
        :param attrs: attributes in columns of :param data_df: to compute feautres for.
        :param freq: (dict { attr: { val: count } } }) if not None, uses these
            frequency statistics instead of computing it from data_df.
        :param cooccur_freq: (dict { attr1: { attr2: { val1: { val2: count } } } })
            if not None, uses these co-occurrence statistics instead of
            computing it from data_df.
        """
        self.ds = dataset
        self.attrs = self.ds.get_attributes()
        self.attr_to_idx = {attr: idx for idx, attr in enumerate(self.attrs)}
        self.n_attrs = len(self.attrs)

    def num_features(self):
        return len(self.attrs) * len(self.attrs)

    def setup(self):
        _, self.freq, self.cooccur_freq = self.ds.get_statistics()

    def create_tensor(self, row, attr, values):
        """
        :param row: (namedtuple or dict) current initial values
        :param attr: (str) attribute of row (i.e. cell) the :param values: correspond to
            and the cell to generate a feature tensor for.
        :param values: (list[str]) values to generate

        :return: Tensor with dimensions (len(values), # of features)
        """
        tensor = torch.zeros(len(values), self.num_features())
        for val_idx, val in enumerate(values):
            for other_attr_idx, other_attr in enumerate(self.attrs):
                if attr == other_attr:
                    continue
                other_val = row[other_attr]
                if val == NULL_REPR or other_val == NULL_REPR:
                    fv = NA_COOCCUR_FV
                else:
                    cooccur = self.cooccur_freq[attr][other_attr].get(val, {}
                        ).get(other_val, NA_COOCCUR_FV)
                    freq = self.freq[other_attr][row[other_attr]]
                    fv = float(cooccur) / float(freq)
                feat_idx = self.attr_to_idx[attr
                    ] * self.n_attrs + other_attr_idx
                tensor[val_idx, feat_idx] = fv
        return tensor


class Logistic(Estimator, torch.nn.Module):
    """
    Logistic is an Estimator that approximates posterior of
    p(v_cur | v_init) by training a logistic regression model to predict the current
    value in a cell given all other initial values using features
    of the other initial values such as co-occurrence.
    """
    WEIGHT_DECAY = 0

    def __init__(self, env, dataset, domain_df, active_attrs):
        """
        :param dataset: (Dataset) original dataset
        :param domain_df: (DataFrame) currently populated domain dataframe.
            Required columns are: _vid_, _tid_, attribute, domain, domain_size, init_value
        :param active_attrs: (list[str]) attributes that have random values
        """
        torch.nn.Module.__init__(self)
        Estimator.__init__(self, env, dataset)
        self.active_attrs = active_attrs
        self.domain_records = domain_df.sort_values('_vid_')[['_vid_',
            '_tid_', 'attribute', 'domain', 'init_value']].to_records()
        self.n_samples = int(domain_df['domain_size'].sum())
        self.featurizers = [CooccurAttrFeaturizer(self.ds)]
        for f in self.featurizers:
            f.setup()
        self.num_features = sum(feat.num_features() for feat in self.
            featurizers)
        self._gen_training_data()
        self._W = torch.nn.Parameter(torch.zeros(self.num_features, 1))
        torch.nn.init.xavier_uniform_(self._W)
        self._B = torch.nn.Parameter(torch.Tensor([1e-06]))
        self._loss = torch.nn.BCELoss()
        if self.env['optimizer'] == 'sgd':
            self._optimizer = SGD(self.parameters(), lr=self.env[
                'learning_rate'], momentum=self.env['momentum'],
                weight_decay=self.WEIGHT_DECAY)
        else:
            self._optimizer = Adam(self.parameters(), lr=self.env[
                'learning_rate'], weight_decay=self.WEIGHT_DECAY)

    def _gen_training_data(self):
        """
        _gen_training_data memoizes the self._X and self._Y tensors
        used for training and prediction.
        """
        logging.debug('Logistic: featurizing training data...')
        tic = time.clock()
        self._X = torch.zeros(self.n_samples, self.num_features)
        self._Y = torch.zeros(self.n_samples)
        self._train_idx = torch.zeros(self.n_samples)
        """
        Iterate through the domain for every cell and create a sample
        to use in training. We assign Y as 1 if the value is the initial value.
        """
        sample_idx = 0
        raw_data_dict = self.ds.raw_data.df.set_index('_tid_').to_dict('index')
        self.vid_to_idxs = {}
        for rec in tqdm(list(self.domain_records)):
            init_row = raw_data_dict[rec['_tid_']]
            domain_vals = rec['domain'].split('|||')
            feat_tensor = self._gen_feat_tensor(init_row, rec['attribute'],
                domain_vals)
            assert feat_tensor.shape[0] == len(domain_vals)
            self._X[sample_idx:sample_idx + len(domain_vals)] = feat_tensor
            self.vid_to_idxs[rec['_vid_']] = sample_idx, sample_idx + len(
                domain_vals)
            if rec['init_value'] == NULL_REPR:
                sample_idx += len(domain_vals)
                continue
            self._train_idx[sample_idx:sample_idx + len(domain_vals)] = 1
            init_idx = domain_vals.index(rec['init_value'])
            self._Y[sample_idx + init_idx] = 1
            sample_idx += len(domain_vals)
        self._train_idx = (self._train_idx == 1).nonzero()[:, (0)]
        logging.debug('Logistic: DONE featurization in %.2fs', time.clock() -
            tic)

    def _gen_feat_tensor(self, init_row, attr, domain_vals):
        """
        Generates the feature tensor for the list of :param`domain_vals` from
        all featurizers.

        :param init_row: (namedtuple or dict) current initial values
        :param attr: (str) attribute of row (i.e. cell) the :param values: correspond to
            and the cell to generate a feature tensor for.
        :param domain_vals: (list[str]) domain values to featurize for

        :return: Tensor with dimensions (len(values), total # of features across all featurizers)
        """
        return torch.cat([f.create_tensor(init_row, attr, domain_vals) for
            f in self.featurizers], dim=1)

    def forward(self, X):
        linear = X.matmul(self._W) + self._B
        return torch.sigmoid(linear)

    def train(self, num_epochs=3, batch_size=32):
        """
        Trains the LR model.

        :param num_epochs: (int) number of epochs.
        """
        batch_losses = []
        X_train, Y_train = self._X.index_select(0, self._train_idx
            ), self._Y.index_select(0, self._train_idx)
        torch_ds = TensorDataset(X_train, Y_train)
        for epoch_idx in range(1, num_epochs + 1):
            logging.debug('Logistic: epoch %d', epoch_idx)
            batch_cnt = 0
            for batch_X, batch_Y in tqdm(DataLoader(torch_ds, batch_size=
                batch_size)):
                batch_pred = self.forward(batch_X)
                batch_loss = self._loss(batch_pred, batch_Y.reshape(-1, 1))
                batch_losses.append(float(batch_loss))
                self.zero_grad()
                batch_loss.backward()
                self._optimizer.step()
                batch_cnt += 1
            logging.debug('Logistic: average batch loss: %f', sum(
                batch_losses[-1 * batch_cnt:]) / batch_cnt)
        return batch_losses

    def predict_pp(self, row, attr=None, values=None):
        """
        predict_pp generates posterior probabilities for the domain values
        corresponding to the cell/random variable row['_vid_'].

        That is: :param`attr` and :param`values` are ignored.

        predict_pp_batch is much faster for Logistic since it simply does
        a one-pass of the batch feature tensor.

        :return: (list[2-tuple]) 2-tuples corresponding to (value, proba)
        """
        start_idx, end_idx = self.vid_to_idxs[row['_vid_']]
        pred_X = self._X[start_idx:end_idx]
        pred_Y = self.forward(pred_X)
        values = self.domain_records[row['_vid_']]['domain'].split('|||')
        return zip(values, map(float, pred_Y))

    def predict_pp_batch(self):
        """
        Performs batch prediction.
        """
        pred_Y = self.forward(self._X)
        for rec in self.domain_records:
            values = rec['domain'].split('|||')
            start_idx, end_idx = self.vid_to_idxs[rec['_vid_']]
            yield zip(values, map(float, pred_Y[start_idx:end_idx]))


class TiedLinear(torch.nn.Module):
    """
    TiedLinear is a linear layer with shared parameters for features between
    (output) classes that takes as input a tensor X with dimensions
        (batch size) X (output_dim) X (in_features)
        where:
            output_dim is the desired output dimension/# of classes
            in_features are the features with shared weights across the classes
    """

    def __init__(self, env, feat_info, output_dim, bias=False):
        super(TiedLinear, self).__init__()
        self.env = env
        self.in_features = 0.0
        self.weight_list = ParameterList()
        if bias:
            self.bias_list = ParameterList()
        else:
            self.register_parameter('bias', None)
        self.output_dim = output_dim
        self.bias_flag = bias
        for feat_entry in feat_info:
            learnable = feat_entry.learnable
            feat_size = feat_entry.size
            init_weight = feat_entry.init_weight
            self.in_features += feat_size
            feat_weight = Parameter(init_weight * torch.ones(1, feat_size),
                requires_grad=learnable)
            if learnable:
                self.reset_parameters(feat_weight)
            self.weight_list.append(feat_weight)
            if bias:
                feat_bias = Parameter(torch.zeros(1, feat_size),
                    requires_grad=learnable)
                if learnable:
                    self.reset_parameters(feat_bias)
                self.bias_list.append(feat_bias)

    def reset_parameters(self, tensor):
        stdv = 1.0 / math.sqrt(tensor.size(0))
        tensor.data.uniform_(-stdv, stdv)

    def concat_weights(self):
        self.W = torch.cat([t for t in self.weight_list], -1)
        if self.env['weight_norm']:
            self.W = self.W.div(self.W.norm(p=2))
        self.W = self.W.expand(self.output_dim, -1)
        if self.bias_flag:
            self.B = torch.cat([t.expand(self.output_dim, -1) for t in self
                .bias_list], -1)

    def forward(self, X, index, mask):
        self.concat_weights()
        output = X.mul(self.W)
        if self.bias_flag:
            output += self.B
        output = output.sum(2)
        output.index_add_(0, index, mask)
        return output


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_HoloClean_holoclean(_paritybench_base):
    pass
