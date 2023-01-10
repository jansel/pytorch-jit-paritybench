import sys
_module = sys.modules[__name__]
del sys
benchmarks = _module
datasets = _module
operators = _module
run = _module
score = _module
train = _module
openml_pipelines = _module
run = _module
score = _module
timer = _module
trees = _module
metrics = _module
run = _module
score = _module
hummingbird = _module
ml = _module
_executor = _module
_parse = _module
_topology = _module
_utils = _module
containers = _module
_input_containers = _module
_sklearn_api_containers = _module
batch_container = _module
sklearn = _module
onnx_containers = _module
pytorch_containers = _module
tvm_containers = _module
convert = _module
exceptions = _module
operator_converters = _module
_array_feature_extractor_implementations = _module
_decomposition_implementations = _module
_discretizer_implementations = _module
_gbdt_commons = _module
_imputer_implementations = _module
_kneighbors_implementations = _module
_label_encoder_implementations = _module
_linear_implementations = _module
_mlp_implementations = _module
_nb_implementations = _module
_normalizer_implementations = _module
_one_hot_encoder_implementations = _module
_physical_operator = _module
_pipeline_implementations = _module
_scaler_implementations = _module
_sv_implementations = _module
_tree_commons = _module
_tree_implementations = _module
constants = _module
lightgbm = _module
onnx = _module
array_feature_extractor = _module
binarizer = _module
feature_vectorizer = _module
imputer = _module
label_encoder = _module
linear = _module
normalizer = _module
one_hot_encoder = _module
onnx_operator = _module
scaler = _module
sv = _module
tree_ensemble = _module
prophet = _module
bagging = _module
cluster = _module
decision_tree = _module
decomposition = _module
discretizer = _module
gbdt = _module
iforest = _module
imputer = _module
kneighbors = _module
mlp = _module
nb = _module
pipeline = _module
poly_features = _module
sparkml = _module
discretizer = _module
vector_assembler = _module
xgb = _module
supported = _module
setup = _module
test_backends = _module
test_extra_conf = _module
test_lightgbm_converter = _module
test_no_extra_install = _module
test_onnxml_binarizer_converter = _module
test_onnxml_decision_tree_converter = _module
test_onnxml_imputer_converter = _module
test_onnxml_label_encoder_converter = _module
test_onnxml_lightgbm_converter = _module
test_onnxml_linear_converter = _module
test_onnxml_normalizer_converter = _module
test_onnxml_one_hot_encoder_converter = _module
test_onnxml_scaler_converter = _module
test_onnxml_sv_converter = _module
test_prophet = _module
test_sklearn_array_feature_extractor_converter = _module
test_sklearn_bagging = _module
test_sklearn_clustering = _module
test_sklearn_decision_tree_converter = _module
test_sklearn_decomposition = _module
test_sklearn_discretizer_converters = _module
test_sklearn_feature_union = _module
test_sklearn_gbdt_converter = _module
test_sklearn_histgbdt_converters = _module
test_sklearn_imputer_converter = _module
test_sklearn_isolation_forest_converter = _module
test_sklearn_kneighbors = _module
test_sklearn_label_encoder_converter = _module
test_sklearn_linear_converter = _module
test_sklearn_mlp_converter = _module
test_sklearn_model_selection = _module
test_sklearn_multioutput_regression = _module
test_sklearn_nb_converter = _module
test_sklearn_normalizer_converter = _module
test_sklearn_notfitted = _module
test_sklearn_one_hot_encoder_converter = _module
test_sklearn_pipeline = _module
test_sklearn_poly_features_converter = _module
test_sklearn_scaler_converter = _module
test_sklearn_sv_converter = _module
test_sparkml_discretizer_converters = _module
test_sparkml_linear_converter = _module
test_sparkml_pipeline = _module
test_sparkml_vector_assembler = _module
test_xgboost_converter = _module
tree_utils = _module
conf = _module
github_link = _module
sphinx_issues = _module

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


import warnings


from scipy import stats


from sklearn.linear_model import LogisticRegression


from sklearn.linear_model import LogisticRegressionCV


from sklearn.naive_bayes import BernoulliNB


from sklearn.neural_network import MLPClassifier


from sklearn.tree import DecisionTreeClassifier


from abc import ABC


from abc import abstractmethod


import time


import sklearn


import torch


from uuid import uuid4


from types import ModuleType


from copy import deepcopy


from sklearn.utils import all_estimators


from sklearn.utils.validation import check_is_fitted


from enum import Enum


import numpy


import scipy


import copy


from collections import defaultdict


from typing import Iterator


from sklearn.ensemble import GradientBoostingClassifier


from sklearn.preprocessing import StandardScaler


from sklearn.preprocessing import OneHotEncoder


from sklearn.datasets import load_iris


from sklearn.model_selection import train_test_split


from sklearn.pipeline import make_pipeline


from sklearn import datasets


from sklearn.ensemble import GradientBoostingRegressor


from sklearn.ensemble import IsolationForest


from sklearn.preprocessing import LabelEncoder


from sklearn.compose import ColumnTransformer


from sklearn.pipeline import Pipeline


from sklearn.preprocessing import Binarizer


from sklearn.impute import SimpleImputer


from sklearn.linear_model import LinearRegression


from sklearn.linear_model import SGDClassifier


from sklearn.preprocessing import Normalizer


from sklearn.preprocessing import MaxAbsScaler


from sklearn.preprocessing import MinMaxScaler


from sklearn.preprocessing import RobustScaler


from sklearn.svm import LinearSVC


from sklearn.svm import SVC


from sklearn.svm import NuSVC


from sklearn.feature_selection import chi2


from sklearn.feature_selection import mutual_info_classif


from sklearn.feature_selection import SelectKBest


from sklearn.feature_selection import SelectPercentile


from sklearn.feature_selection import VarianceThreshold


from sklearn.datasets import load_digits


from sklearn.svm import LinearSVR


from sklearn.datasets import make_regression


from sklearn.datasets import make_classification


from sklearn.ensemble import BaggingClassifier


from sklearn.ensemble import BaggingRegressor


from sklearn.cluster import KMeans


from sklearn.ensemble import ExtraTreesClassifier


from sklearn.ensemble import ExtraTreesRegressor


from sklearn.ensemble import RandomForestClassifier


from sklearn.ensemble import RandomForestRegressor


from sklearn.tree import DecisionTreeRegressor


import random


from sklearn.decomposition import FastICA


from sklearn.decomposition import KernelPCA


from sklearn.decomposition import PCA


from sklearn.decomposition import TruncatedSVD


from sklearn.cross_decomposition import PLSRegression as PLSR


from sklearn.preprocessing import KBinsDiscretizer


from sklearn.datasets import load_breast_cancer


from sklearn.pipeline import FeatureUnion


from sklearn.ensemble._hist_gradient_boosting.gradient_boosting import HistGradientBoostingClassifier


from sklearn.ensemble._hist_gradient_boosting.gradient_boosting import HistGradientBoostingRegressor


from sklearn.impute import MissingIndicator


from sklearn.neighbors import KNeighborsClassifier


from sklearn.neighbors import KNeighborsRegressor


from sklearn.linear_model import RidgeCV


from sklearn.linear_model import Lasso


from sklearn.linear_model import ElasticNet


from sklearn.linear_model import Ridge


from sklearn.linear_model import TweedieRegressor


from sklearn.linear_model import PoissonRegressor


from sklearn.linear_model import GammaRegressor


from sklearn.neural_network import MLPRegressor


from sklearn.model_selection import GridSearchCV


from sklearn.model_selection import RandomizedSearchCV


from sklearn.metrics import make_scorer


from sklearn.metrics import accuracy_score


from sklearn.multioutput import MultiOutputRegressor


from sklearn.multioutput import RegressorChain


from sklearn.naive_bayes import GaussianNB


from sklearn.naive_bayes import MultinomialNB


from sklearn.exceptions import NotFittedError


from sklearn.datasets import load_diabetes


from sklearn.preprocessing import PolynomialFeatures


from sklearn.svm import SVR


from sklearn.datasets import fetch_california_housing


_constant_error = """
It usually means a constant is not available or you are trying to override a constant value.
"""


class ConstantError(TypeError):
    """
    Raised when a constant is not available or it get overwritten.
    """

    def __init__(self, msg):
        super().__init__(msg + _constant_error)


class _Constants(object):
    """
    Class enabling the proper definition of constants.
    """

    def __init__(self, constants, other_constants=None):
        for constant in dir(constants):
            if constant.isupper():
                setattr(self, constant, getattr(constants, constant))
        for constant in dir(other_constants):
            if constant.isupper():
                setattr(self, constant, getattr(other_constants, constant))

    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise ConstantError('Overwriting a constant is not allowed {}'.format(name))
        self.__dict__[name] = value


def from_strings_to_ints(input, max_string_length):
    """
    Utility function used to transform string inputs into a numerical representation.
    """
    shape = list(input.shape)
    shape.append(max_string_length // 4)
    return np.array(input, dtype='|S' + str(max_string_length)).view(np.int32).reshape(shape)


def get_device(model):
    """
    Convenient function used to get the runtime device for the model.
    """
    assert issubclass(model.__class__, torch.nn.Module)
    device = None
    if len(list(model.parameters())) > 0:
        device = next(model.parameters()).device
    return device


class Executor(torch.nn.Module, object):
    """
    Executor class able to run Hummingbird's internal representation of a converted pipeline.
    """

    def __init__(self, input_names, output_names, operator_map, operators, extra_config):
        """
        Args:
            input_names: The names of the input `onnxconverter_common.topology.Variable`s for this model
            output_names: The names of the output `onnxconverter_common.topology.Variable`s generated by this model
            operator_map: A dictionary of operator aliases and related PyTorch implementations
            operators: The list of operators (in a topological order) that will be executed by the model (in order)
            extra_config: Some additional custom configuration parameter
        """
        super(Executor, self).__init__()

        def _fix_var_naming(operators, names, mod='input'):
            new_names = []
            map = {}
            for op in operators:
                if mod == 'input':
                    iter = op.inputs
                else:
                    iter = op.outputs
                for i in iter:
                    for name in names:
                        if i.raw_name == name and name not in map:
                            map[i.raw_name] = i.full_name
                if len(map) == len(names):
                    break
            if map == {}:
                return names
            for name in names:
                new_names.append(map[name])
            return new_names
        self._input_names = _fix_var_naming(operators, input_names)
        self._output_names = _fix_var_naming(reversed(operators), output_names, 'output')
        self._operators = torch.nn.ModuleList([operator_map[operator.full_name] for operator in operators])
        self.max_string_length = None
        if constants.MAX_STRING_LENGTH in extra_config:
            self.max_string_length = extra_config[constants.MAX_STRING_LENGTH]

    def forward(self, *inputs):
        with torch.no_grad():
            assert len(self._input_names) == len(inputs) or DataFrame is not None and isinstance(inputs[0], DataFrame) and not self.check_dataframe_to_array and len(self._input_names) == len(inputs[0].columns), 'number of inputs or number of columns in the dataframe do not match with the expected number of inputs {}'.format(self._input_names)
            if DataFrame is not None and isinstance(inputs[0], DataFrame):
                inputs = inputs[0]
                input_names = list(inputs.columns)
                splits = [inputs[input_names[idx]] for idx in range(len(input_names))]
                splits = [df.to_numpy().reshape(-1, 1) for df in splits]
                inputs = tuple(splits)
            inputs = [*inputs]
            variable_map = {}
            device = get_device(self)
            for i, input_name in enumerate(self._input_names):
                input_ = inputs[i]
                if type(input_) is list:
                    input_ = np.array(input_)
                if type(input_) is np.ndarray:
                    if input_.dtype.kind in constants.SUPPORTED_STRING_TYPES:
                        assert self.max_string_length is not None
                        input_ = from_strings_to_ints(input_, self.max_string_length)
                    elif input_.dtype.kind == 'M':
                        input_ = (input_ - np.datetime64('1970-01-01T00:00:00.000000000')).astype(np.int64) / 1000000000
                    input_ = torch.from_numpy(input_)
                elif type(input_) is not torch.Tensor:
                    raise RuntimeError('Inputer tensor {} of not supported type {}'.format(input_name, type(input_)))
                if input_.dtype == torch.float64:
                    input_ = input_.float()
                if device is not None and device.type != 'cpu':
                    input_ = input_
                variable_map[input_name] = input_
            for operator in self._operators:
                outputs = operator(*(variable_map[input_name] for input_name in operator.inputs))
                if len(operator.outputs) == 1:
                    variable_map[operator.outputs[0]] = outputs
                else:
                    for i, output_name in enumerate(operator.outputs):
                        variable_map[output_name] = outputs[i]
            if len(self._output_names) == 1:
                return variable_map[self._output_names[0]]
            else:
                return tuple(variable_map[output_name] for output_name in self._output_names)


class PhysicalOperator(ABC):
    """
    Abstract class defining the basic structure for operator implementations in Hummingbird.
    """

    def __init__(self, operator, regression=False, classification=False, transformer=False, anomaly_detection=False, **kwargs):
        """
        Args:
            regression: Whether operator is a regression model.
            classification: Whether the operator is a classification model.
            transformer: Whether the operator is a feature transformer.
            anomaly_detection: Whether the operator is an anomaly detection model.
            kwargs: Other keyword arguments.
        """
        super().__init__()
        self.name = operator.full_name
        self.inputs = [input_.full_name for input_ in operator.inputs]
        self.outputs = [output_.full_name for output_ in operator.outputs]
        self.regression = regression
        self.classification = classification
        self.transformer = transformer
        self.anomaly_detection = anomaly_detection


class ArrayFeatureExtractor(PhysicalOperator, torch.nn.Module):
    """
    Class implementing ArrayFeatureExtractor in PyTorch

    This is used by SelectKBest, VarianceThreshold operators in scikit-learn
    """

    def __init__(self, logical_operator, column_indices, device):
        super(ArrayFeatureExtractor, self).__init__(logical_operator, transformer=True)
        is_contiguous = False
        if max(column_indices) - min(column_indices) + 1 == len(column_indices):
            is_contiguous = True
            self.min = min(column_indices)
            self.max = max(column_indices) + 1
        self.column_indices = torch.nn.Parameter(torch.LongTensor(column_indices), requires_grad=False)
        self.is_contiguous = is_contiguous

    def forward(self, x):
        if type(x) == tuple:
            return x[self.column_indices]
        if len(x.shape) == 1:
            x = x.view(1, -1)
        if self.is_contiguous:
            return x[:, self.min:self.max]
        else:
            return torch.index_select(x, 1, self.column_indices)


class Decomposition(PhysicalOperator, torch.nn.Module):

    def __init__(self, logical_operator, mean, transform_matrix, device):
        super(Decomposition, self).__init__(logical_operator)
        self.transformer = True
        if mean is not None:
            self.mean = torch.nn.Parameter(torch.from_numpy(mean), requires_grad=False)
        else:
            self.mean = None
        self.transform_matrix = torch.nn.Parameter(torch.from_numpy(transform_matrix), requires_grad=False)

    def forward(self, x):
        if self.mean is not None:
            x -= self.mean
        return torch.mm(x, self.transform_matrix).float()


class KernelPCA(PhysicalOperator, torch.nn.Module):

    def __init__(self, logical_operator, kernel, degree, sv, scaled_alphas, gamma, coef0, k_fit_rows, k_fit_all, device):
        super(KernelPCA, self).__init__(logical_operator)
        self.transformer = True
        self.kernel = kernel
        self.degree = degree
        self.n_samples = sv.shape[0]
        self.sv = torch.nn.Parameter(torch.from_numpy(sv).float(), requires_grad=False)
        self.n_features = sv.shape[1]
        self.k_fit_rows = torch.nn.Parameter(torch.from_numpy(k_fit_rows).float(), requires_grad=False)
        self.k_fit_all = k_fit_all
        if gamma is None:
            gamma = 1.0 / self.n_features
        self.gamma = gamma
        self.coef0 = coef0
        self.scaled_alphas = torch.nn.Parameter(torch.from_numpy(scaled_alphas).float(), requires_grad=False)

    def forward(self, x):
        if self.kernel == 'linear':
            x = x.view(-1, 1, self.n_features)
            k = self.sv * x
            k = k.sum(2)
        elif self.kernel == 'rbf':
            x = x.view(-1, 1, self.n_features)
            k = torch.pow(self.sv - x, 2)
            k = k.sum(2)
            k = torch.exp(-self.gamma * k)
        elif self.kernel == 'poly':
            k = torch.pow(self.gamma * torch.mm(x, self.sv.t()) + self.coef0, self.degree)
        elif self.kernel == 'sigmoid':
            k = torch.tanh(self.gamma * torch.mm(x, self.sv.t()) + self.coef0)
        elif self.kernel == 'cosine':
            norm_x = torch.norm(x, keepdim=True, dim=1)
            norm_sv = torch.norm(self.sv, keepdim=True, dim=1)
            norm = torch.mm(norm_x, norm_sv.t())
            k = torch.mm(x, self.sv.t())
            k = torch.div(k, norm)
        elif self.kernel == 'precomputed':
            k = x
        else:
            raise NotImplementedError('Hummingbird does not currently support {} kernel for KernelPCA. The supported kernels are linear, poly, rbf, sigmoid, cosine, and precomputed.'.format(self.kernel))
        k_pred_cols = (torch.sum(k, 1) / self.n_samples).view(-1, 1)
        k -= self.k_fit_rows
        k -= k_pred_cols
        k += self.k_fit_all
        return torch.mm(k, self.scaled_alphas)


class CrossDecomposition(PhysicalOperator, torch.nn.Module):

    def __init__(self, logical_operator, x_mean, x_std, y_mean, coefficients, device):
        super(CrossDecomposition, self).__init__(logical_operator)
        self.regression = True
        self.x_mean = torch.nn.Parameter(torch.from_numpy(x_mean), requires_grad=False)
        self.x_std = torch.nn.Parameter(torch.from_numpy(x_std), requires_grad=False)
        self.y_mean = torch.nn.Parameter(torch.from_numpy(y_mean), requires_grad=False)
        self.coefficients = torch.nn.Parameter(torch.from_numpy(coefficients), requires_grad=False)

    def forward(self, x):
        x -= self.x_mean
        x /= self.x_std
        y_pred = torch.mm(x, self.coefficients).float()
        return y_pred + self.y_mean


class Binarizer(PhysicalOperator, torch.nn.Module):
    """
    Class implementing Binarizer operators in PyTorch.
    """

    def __init__(self, logical_operator, threshold, device):
        super(Binarizer, self).__init__(logical_operator)
        self.transformer = True
        self.threshold = torch.nn.Parameter(torch.FloatTensor([threshold]), requires_grad=False)

    def forward(self, x):
        return torch.gt(x, self.threshold).float()


class KBinsDiscretizer(PhysicalOperator, torch.nn.Module):

    def __init__(self, logical_operator, encode, n_bins, bin_edges, labels, device):
        super(KBinsDiscretizer, self).__init__(logical_operator)
        self.transformer = True
        self.encode = encode
        self.ge_tensor = torch.FloatTensor(bin_edges[:, 1:-1])
        self.ohe = OneHotEncoder(logical_operator, labels, device)
        if n_bins is not None:
            self.n_bins = torch.FloatTensor([[(n - 1) for n in n_bins]])
        else:
            self.n_bins = None

    def forward(self, x):
        x = torch.unsqueeze(x, 2)
        x = torch.ge(x, self.ge_tensor)
        x = x.float()
        x = torch.sum(x, dim=2, keepdim=False)
        if self.n_bins is not None:
            x = torch.min(self.n_bins, x)
        if self.encode in ['onehot-dense', 'onehot']:
            x = self.ohe(x)
        return x


class SimpleImputer(PhysicalOperator, torch.nn.Module):
    """
    Class implementing SimpleImputer operators in PyTorch.
    """

    def __init__(self, logical_operator, device, statistics=None, missing=None, strategy=None):
        super(SimpleImputer, self).__init__(logical_operator)
        sklearn_imputer = logical_operator.raw_operator
        stats_ = statistics if statistics is not None else sklearn_imputer.statistics_
        stats = [float(stat) for stat in stats_]
        missing_values = missing if missing is not None else sklearn_imputer.missing_values
        strategy = strategy if strategy is not None else sklearn_imputer.strategy
        b_mask = np.logical_not(np.isnan(stats))
        i_mask = [i for i in range(len(b_mask)) if b_mask[i]]
        self.transformer = True
        self.do_mask = strategy == 'constant' or all(b_mask)
        self.mask = torch.nn.Parameter(torch.LongTensor([] if self.do_mask else i_mask), requires_grad=False)
        self.replace_values = torch.nn.Parameter(torch.tensor([stats_], dtype=torch.float32), requires_grad=False)
        self.is_nan = True if missing_values == 'NaN' or np.isnan(missing_values) else False
        if not self.is_nan:
            self.missing_values = torch.nn.Parameter(torch.tensor([missing_values], dtype=torch.float32), requires_grad=False)

    def forward(self, x):
        if self.is_nan:
            result = torch.where(torch.isnan(x), self.replace_values.expand(x.shape), x)
            if self.do_mask:
                return result
            return torch.index_select(result, 1, self.mask)
        else:
            return torch.where(torch.eq(x, self.missing_values), self.replace_values.expand(x.shape), x)


class MissingIndicator(PhysicalOperator, torch.nn.Module):
    """
    Class implementing Imputer operators in MissingIndicator.
    """

    def __init__(self, logical_operator, device):
        super(MissingIndicator, self).__init__(logical_operator)
        sklearn_missing_indicator = logical_operator.raw_operator
        self.transformer = True
        self.missing_values = torch.nn.Parameter(torch.tensor([sklearn_missing_indicator.missing_values], dtype=torch.float32), requires_grad=False)
        self.features = sklearn_missing_indicator.features
        self.is_nan = True if sklearn_missing_indicator.missing_values in ['NaN', None, np.nan] else False
        self.column_indices = torch.nn.Parameter(torch.LongTensor(sklearn_missing_indicator.features_), requires_grad=False)

    def forward(self, x):
        if self.is_nan:
            if self.features == 'all':
                return torch.isnan(x).float()
            else:
                return torch.isnan(torch.index_select(x, 1, self.column_indices)).float()
        elif self.features == 'all':
            return torch.eq(x, self.missing_values).float()
        else:
            return torch.eq(torch.index_select(x, 1, self.column_indices), self.missing_values).float()


class MetricType(Enum):
    minkowski = 1
    wminkowski = 2
    seuclidean = 3
    mahalanobis = 4


class KNeighborsModel(PhysicalOperator, torch.nn.Module):

    def __init__(self, logical_operator, train_data, train_labels, n_neighbors, weights, classes, batch_size, is_classifier, metric_type, metric_params):
        super(KNeighborsModel, self).__init__(logical_operator)
        self.classification = is_classifier
        self.regression = not is_classifier
        self.train_data = torch.nn.Parameter(torch.from_numpy(train_data.astype('float32')), requires_grad=False)
        self.train_labels = torch.nn.Parameter(torch.from_numpy(train_labels.astype('int64')), requires_grad=False)
        self.n_neighbors = n_neighbors
        self.metric_type = metric_type
        if self.metric_type == MetricType.minkowski:
            self.p = float(metric_params['p'])
        elif self.metric_type == MetricType.wminkowski:
            self.p = float(metric_params['p'])
            self.w = torch.nn.Parameter(torch.from_numpy(metric_params['w'].astype('float32').reshape(1, -1)), requires_grad=False)
            self.train_data = torch.nn.Parameter(torch.from_numpy(metric_params['w'].astype('float32').reshape(1, -1) * train_data.astype('float32')), requires_grad=False)
        elif self.metric_type == MetricType.seuclidean:
            self.V = torch.nn.Parameter(torch.from_numpy(np.power(metric_params['V'].astype('float32').reshape(1, -1), -0.5)), requires_grad=False)
            self.train_data = torch.nn.Parameter(torch.from_numpy(np.power(metric_params['V'].astype('float32').reshape(1, -1), -0.5) * train_data.astype('float32')), requires_grad=False)
        elif self.metric_type == MetricType.mahalanobis:
            cholesky_l = np.linalg.cholesky(metric_params['VI']).astype('float32')
            self.L = torch.nn.Parameter(torch.from_numpy(cholesky_l), requires_grad=False)
            self.train_data = torch.nn.Parameter(torch.from_numpy(np.matmul(train_data.astype('float32'), cholesky_l)), requires_grad=False)
        if is_classifier:
            self.train_labels = torch.nn.Parameter(torch.from_numpy(train_labels.astype('int64')), requires_grad=False)
            self.classes = torch.nn.Parameter(torch.IntTensor(classes), requires_grad=False)
            self.n_classes = len(classes)
            self.perform_class_select = False
            if min(classes) != 0 or max(classes) != len(classes) - 1:
                self.perform_class_select = True
            self.one_tensor = torch.FloatTensor([1.0])
            self.proba_tensor = torch.zeros((batch_size, self.n_classes), dtype=torch.float32)
        else:
            self.train_labels = torch.nn.Parameter(torch.from_numpy(train_labels.astype('float32')), requires_grad=False)
            self.n_targets = 1
            if len(self.train_labels.shape) == 2:
                self.n_targets = self.train_labels.shape[1]
        self.weights = weights

    def forward(self, x):
        if self.metric_type == MetricType.minkowski:
            k = torch.cdist(x, self.train_data, p=self.p, compute_mode='donot_use_mm_for_euclid_dist')
        elif self.metric_type == MetricType.wminkowski:
            k = torch.cdist(self.w * x, self.train_data, p=self.p, compute_mode='donot_use_mm_for_euclid_dist')
        elif self.metric_type == MetricType.seuclidean:
            k = torch.cdist(self.V * x, self.train_data, p=2, compute_mode='donot_use_mm_for_euclid_dist')
        elif self.metric_type == MetricType.mahalanobis:
            k = torch.cdist(torch.mm(x, self.L), self.train_data, p=2, compute_mode='donot_use_mm_for_euclid_dist')
        d, k = torch.topk(k, self.n_neighbors, dim=1, largest=False)
        output = torch.index_select(self.train_labels, 0, k.view(-1))
        if self.weights == 'distance':
            d = torch.pow(d, -1)
            inf_mask = torch.isinf(d)
            inf_row = torch.any(inf_mask, axis=1)
            d[inf_row] = inf_mask[inf_row].float()
        else:
            d = torch.ones_like(k, dtype=torch.float32)
        if self.classification:
            output = output.view(-1, self.n_neighbors)
            output = torch.scatter_add(self.proba_tensor, 1, output, d)
            proba_sum = output.sum(1, keepdim=True)
            proba_sum = torch.where(proba_sum == 0, self.one_tensor, proba_sum)
            output = torch.pow(proba_sum, -1) * output
            if self.perform_class_select:
                return torch.index_select(self.classes, 0, torch.argmax(output, dim=1)), output
            else:
                return torch.argmax(output, dim=1), output
        else:
            if self.n_targets > 1:
                output = output.view(-1, self.n_neighbors, self.n_targets)
                d = d.view(-1, self.n_neighbors, 1)
            else:
                output = output.view(-1, self.n_neighbors)
            output = d * output
            if self.weights != 'distance':
                output = output.sum(1) / self.n_neighbors
            else:
                denom = d.sum(1)
                output = output.sum(1) / denom
            return output


class StringLabelEncoder(PhysicalOperator, torch.nn.Module):
    """
    LabelEncoder over string data types.
    When the ONNX backend is selected, this operator only works for PyTorch => 1.8.0.
    """

    def __init__(self, logical_operator, classes, device, extra_config={}):
        super(StringLabelEncoder, self).__init__(logical_operator, transformer=True)
        self.regression = False
        self.num_columns = len(classes)
        self.max_word_length = max([len(cat) for cat in classes])
        while self.max_word_length % 4 != 0:
            self.max_word_length += 1
        data_type = '|S' + str(self.max_word_length)
        max_length = 0
        if constants.MAX_STRING_LENGTH in extra_config:
            extra_config[constants.MAX_STRING_LENGTH]
        extra_config[constants.MAX_STRING_LENGTH] = max(max_length, self.max_word_length)
        self.max_word_length = self.max_word_length // 4
        classes_conv = torch.from_numpy(np.array(sorted(set(classes)), dtype=data_type).view(np.int32)).detach().clone()
        classes_conv = classes_conv.view(1, -1, self.max_word_length)
        self.condition_tensors = torch.nn.Parameter(classes_conv, requires_grad=False)

    def forward(self, x):
        x = x.view(-1, 1, self.max_word_length)
        result = torch.prod(self.condition_tensors == x, dim=2).nonzero(as_tuple=True)[1]
        assert result.shape[0] == x.shape[0], 'x ({}) contains previously unseen labels. condition_tensors: {}'.format(x, self.condition_tensors)
        return result


class NumericLabelEncoder(PhysicalOperator, torch.nn.Module):

    def __init__(self, logical_operator, classes, device):
        super(NumericLabelEncoder, self).__init__(logical_operator, transformer=True)
        self.regression = False
        self.check_tensor = torch.nn.Parameter(torch.IntTensor(classes), requires_grad=False)

    def forward(self, x):
        x = x.view(-1, 1)
        return torch.argmax(torch.eq(x, self.check_tensor).int(), dim=1)


class LinearModel(PhysicalOperator, torch.nn.Module):

    def __init__(self, logical_operator, coefficients, intercepts, device, classes=[0], multi_class=None, loss=None, is_linear_regression=False):
        super(LinearModel, self).__init__(logical_operator)
        self.coefficients = torch.nn.Parameter(torch.from_numpy(coefficients).detach().clone(), requires_grad=False)
        self.intercepts = torch.nn.Parameter(torch.from_numpy(intercepts).view(-1).detach().clone(), requires_grad=False)
        self.classes = torch.nn.Parameter(torch.IntTensor(classes), requires_grad=False)
        self.multi_class = multi_class
        self.regression = is_linear_regression
        self.classification = not is_linear_regression
        self.loss = loss
        if self.loss is None and self.classification:
            self.loss = 'log'
        self.perform_class_select = False
        if min(classes) != 0 or max(classes) != len(classes) - 1:
            self.perform_class_select = True
        self.binary_classification = False
        if len(classes) == 2:
            self.binary_classification = True

    def forward(self, x):
        x = x.float()
        output = torch.addmm(self.intercepts, x, self.coefficients)
        if self.multi_class == 'multinomial':
            output = torch.softmax(output, dim=1)
        elif self.regression:
            if self.loss == 'log':
                return torch.exp(output)
            return output
        else:
            if self.loss == 'modified_huber':
                output = torch.clip(output, -1, 1)
                output += 1
                output /= 2
            else:
                output = torch.sigmoid(output)
            if not self.binary_classification:
                if self.loss == 'modified_huber':
                    prob_sum = torch.sum(output, dim=1, keepdim=False)
                    all_zero = prob_sum == 0
                    if torch.any(all_zero):
                        output[all_zero, :] = 1
                        prob_sum[all_zero] = len(self.classes)
                    output /= prob_sum.view((output.shape[0], -1))
                else:
                    output /= torch.sum(output, dim=1, keepdim=True)
        if self.binary_classification:
            output = torch.cat([1 - output, output], dim=1)
        if self.perform_class_select:
            return torch.index_select(self.classes, 0, torch.argmax(output, dim=1)), output
        else:
            return torch.argmax(output, dim=1), output


class MLPModel(PhysicalOperator, torch.nn.Module):

    def __init__(self, logical_operator, weights, biases, activation, device):
        super(MLPModel, self).__init__(logical_operator)
        self.regression = True
        self.weights = torch.nn.ParameterList([torch.nn.Parameter(torch.from_numpy(weight.astype('float32')), requires_grad=False) for weight in weights])
        self.biases = torch.nn.ParameterList([torch.nn.Parameter(torch.from_numpy(bias.astype('float32')), requires_grad=False) for bias in biases])
        self.activation = activation

    def forward(self, x):
        for i in range(len(self.weights) - 1):
            x = torch.addmm(self.biases[i], x, self.weights[i])
            if self.activation == 'relu':
                x = torch.relu(x)
            elif self.activation == 'logistic':
                x = torch.sigmoid(x)
            elif self.activation == 'tanh':
                x = torch.tanh(x)
            elif self.activation != 'identity':
                raise RuntimeError('Unsupported activation {0}'.format(self.activation))
        return torch.addmm(self.biases[-1], x, self.weights[-1])


class MLPClassificationModel(MLPModel):

    def __init__(self, logical_operator, weights, biases, activation, classes, device):
        super(MLPClassificationModel, self).__init__(logical_operator, weights, biases, activation, device)
        self.regression = False
        self.classification = True
        self.classes = torch.nn.Parameter(torch.IntTensor(classes), requires_grad=False)
        self.perform_class_select = False
        if min(classes) != 0 or max(classes) != len(classes) - 1:
            self.perform_class_select = True
        self.binary_classification = False
        if len(classes) == 2:
            self.binary_classification = True

    def forward(self, x):
        x = super().forward(x)
        if self.binary_classification:
            output = torch.sigmoid(x)
            output = torch.cat([1 - output, output], dim=1)
        else:
            output = torch.softmax(x, dim=1)
        if self.perform_class_select:
            return torch.index_select(self.classes, 0, torch.argmax(output, dim=1)), output
        else:
            return torch.argmax(output, dim=1), output


class BernoulliNBModel(PhysicalOperator, torch.nn.Module):

    def __init__(self, logical_operator, classes, binarize, jll_calc_bias, feature_log_prob_minus_neg_prob, device):
        super(BernoulliNBModel, self).__init__(logical_operator)
        self.classification = True
        self.binarize = binarize
        self.jll_calc_bias = torch.nn.Parameter(torch.from_numpy(jll_calc_bias.astype('float64')).view(-1), requires_grad=False)
        self.feature_log_prob_minus_neg_prob = torch.nn.Parameter(torch.from_numpy(feature_log_prob_minus_neg_prob.astype('float64')), requires_grad=False)
        self.classes = torch.nn.Parameter(torch.IntTensor(classes), requires_grad=False)
        self.perform_class_select = False
        if min(classes) != 0 or max(classes) != len(classes) - 1:
            self.perform_class_select = True

    def forward(self, x):
        x = x.double()
        if self.binarize is not None:
            x = torch.gt(x, self.binarize).double()
        jll = torch.addmm(self.jll_calc_bias, x, self.feature_log_prob_minus_neg_prob)
        log_prob_x = torch.logsumexp(jll, dim=1)
        log_prob_x = jll - log_prob_x.view(-1, 1)
        prob_x = torch.exp(log_prob_x).float()
        if self.perform_class_select:
            return torch.index_select(self.classes, 0, torch.argmax(jll, dim=1)), prob_x
        else:
            return torch.argmax(jll, dim=1), prob_x


class GaussianNBModel(PhysicalOperator, torch.nn.Module):

    def __init__(self, logical_operator, classes, jll_calc_bias, theta, sigma, device):
        super(GaussianNBModel, self).__init__(logical_operator)
        self.classification = True
        self.jll_calc_bias = torch.nn.Parameter(torch.from_numpy(jll_calc_bias.astype('float32')), requires_grad=False)
        self.theta = torch.nn.Parameter(torch.from_numpy(theta.astype('float32')).view((len(classes), 1, -1)), requires_grad=False)
        self.sigma = torch.nn.Parameter(torch.from_numpy(sigma.astype('float32')).view((len(classes), 1, -1)), requires_grad=False)
        self.classes = torch.nn.Parameter(torch.IntTensor(classes), requires_grad=False)
        self.perform_class_select = False
        if min(classes) != 0 or max(classes) != len(classes) - 1:
            self.perform_class_select = True

    def forward(self, x):
        jll = self.jll_calc_bias - 0.5 * torch.sum(torch.div(torch.pow(x - self.theta, 2), self.sigma), 2)
        jll = torch.transpose(jll, 0, 1)
        log_prob_x = torch.logsumexp(jll, dim=1)
        log_prob_x = jll - log_prob_x.view(-1, 1)
        prob_x = torch.exp(log_prob_x)
        if self.perform_class_select:
            return torch.index_select(self.classes, 0, torch.argmax(jll, dim=1)), prob_x
        else:
            return torch.argmax(jll, dim=1), prob_x


class Normalizer(PhysicalOperator, torch.nn.Module):
    """
    Class implementing Normalizer operators in PyTorch. Supported normalizers are L1, L2 and Max.
    """

    def __init__(self, logical_operator, norm, device):
        super(Normalizer, self).__init__(logical_operator)
        self.norm = norm
        self.transformer = True

    def forward(self, x):
        if self.norm == 'l1':
            return x / torch.abs(x).sum(1, keepdim=True)
        elif self.norm == 'l2':
            return x / torch.pow(torch.pow(x, 2).sum(1, keepdim=True), 0.5)
        elif self.norm == 'max':
            return x / torch.max(torch.abs(x), dim=1, keepdim=True)[0]
        else:
            raise RuntimeError('Unsupported norm: {0}'.format(self.norm))


class OneHotEncoderString(PhysicalOperator, torch.nn.Module):
    """
    Class implementing OneHotEncoder operators for strings in PyTorch.

    Because we are dealing with tensors, strings require additional length information for processing.
    """

    def __init__(self, logical_operator, categories, device, extra_config={}):
        super(OneHotEncoderString, self).__init__(logical_operator, transformer=True)
        self.num_columns = len(categories)
        self.max_word_length = max([max([len(c) for c in cat]) for cat in categories])
        while self.max_word_length % 4 != 0:
            self.max_word_length += 1
        max_length = 0
        if constants.MAX_STRING_LENGTH in extra_config:
            max_length = extra_config[constants.MAX_STRING_LENGTH]
        extra_config[constants.MAX_STRING_LENGTH] = max(max_length, self.max_word_length)
        condition_tensors = []
        categories_idx = [0]
        for arr in categories:
            cats = np.array(arr, dtype='|S' + str(self.max_word_length)).view('int32').reshape(-1, self.max_word_length // 4).tolist()
            condition_tensors.extend(cats)
            categories_idx.append(categories_idx[-1] + len(cats))
        self.condition_tensors = torch.nn.Parameter(torch.IntTensor(condition_tensors), requires_grad=False)
        self.categories_idx = categories_idx

    def forward(self, x):
        encoded_tensors = []
        for i in range(self.num_columns):
            conditions = self.condition_tensors[self.categories_idx[i]:self.categories_idx[i + 1], :].view(1, -1, self.max_word_length // 4)
            encoded_tensors.append(torch.prod(torch.eq(x[:, i:i + 1, :], conditions), dim=2))
        return torch.cat(encoded_tensors, dim=1).float()


class OneHotEncoder(PhysicalOperator, torch.nn.Module):
    """
    Class implementing OneHotEncoder operators for ints in PyTorch.
    """

    def __init__(self, logical_operator, categories, device):
        super(OneHotEncoder, self).__init__(logical_operator, transformer=True)
        self.num_columns = len(categories)
        condition_tensors = []
        for arr in categories:
            condition_tensors.append(torch.nn.Parameter(torch.LongTensor(arr).detach().clone(), requires_grad=False))
        self.condition_tensors = torch.nn.ParameterList(condition_tensors)

    def forward(self, *x):
        encoded_tensors = []
        if len(x) > 1:
            assert len(x) == self.num_columns
            for i in range(self.num_columns):
                input = x[i]
                if input.dtype != torch.int64:
                    input = input.long()
                encoded_tensors.append(torch.eq(input, self.condition_tensors[i]))
        else:
            x = x[0]
            if x.dtype != torch.int64:
                x = x.long()
            for i in range(self.num_columns):
                encoded_tensors.append(torch.eq(x[:, i:i + 1], self.condition_tensors[i]))
        return torch.cat(encoded_tensors, dim=1).float()


class Concat(PhysicalOperator, torch.nn.Module):

    def __init__(self, logical_operator):
        super(Concat, self).__init__(logical_operator, transformer=True)

    def forward(self, *x):
        if len(x[0].shape) > 1:
            dtypes = {t.dtype for t in x}
            if len(dtypes) > 1:
                if torch.float64 in dtypes:
                    x = [t.double() for t in x]
                elif torch.float32 in dtypes:
                    x = [t.float() for t in x]
                else:
                    raise RuntimeError('Combination of data types for Concat input tensors not supported. Please fill an issue at https://github.com/microsoft/hummingbird.')
            return torch.cat(x, dim=1)
        else:
            return torch.stack([i.view(-1) for i in x], dim=1)


class Scaler(PhysicalOperator, torch.nn.Module):
    """
    Class implementing Scaler operators in PyTorch. Supported normalizers are L1, L2 and Max.
    """

    def __init__(self, logical_operator, offset, scale, device):
        super(Scaler, self).__init__(logical_operator, transformer=True)
        if offset is None or len(offset.shape) == 0 or offset.shape == (0,):
            offset = numpy.array([0], dtype=numpy.float32)
        if scale is None or len(scale.shape) == 0 or scale.shape == (0,):
            scale = numpy.array([1], dtype=numpy.float32)
        self.offset = offset
        self.scale = scale
        if offset is not None:
            self.offset = torch.nn.Parameter(torch.from_numpy(offset).detach().clone(), requires_grad=False)
        if scale is not None:
            self.scale = torch.nn.Parameter(torch.from_numpy(scale).detach().clone(), requires_grad=False)

    def forward(self, x):
        if self.offset is not None:
            x = x - self.offset
        if self.scale is not None:
            x = x * self.scale
        return x.float()


class SVC(PhysicalOperator, torch.nn.Module):

    def __init__(self, logical_operator, kernel, degree, sv, nv, a, b, gamma, coef0, classes, device):
        super(SVC, self).__init__(logical_operator, classification=True)
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.regression = False
        sv = sv.toarray() if type(sv) == scipy.sparse.csr_matrix else sv
        self.sv = torch.nn.Parameter(torch.from_numpy(sv).double(), requires_grad=False)
        self.sv_t = torch.nn.Parameter(torch.transpose(self.sv, 0, 1), requires_grad=False)
        self.sv_norm = torch.nn.Parameter(-self.gamma * (self.sv ** 2).sum(1).view(1, -1), requires_grad=False)
        self.coef0 = coef0
        self.n_features = sv.shape[1]
        self.a = a
        self.b = torch.nn.Parameter(torch.from_numpy(b.reshape(1, -1)).double(), requires_grad=False)
        self.start = [sum(nv[:i]) for i in range(len(nv))]
        self.end = [(self.start[i] + nv[i]) for i in range(len(nv))]
        self.len_nv = len(nv)
        true_classes, false_classes = zip(*[(i, j) for i in range(self.len_nv) for j in range(i + 1, self.len_nv)])
        self.true_classes = torch.nn.Parameter(torch.IntTensor([true_classes]), requires_grad=False)
        self.false_classes = torch.nn.Parameter(torch.IntTensor([false_classes]), requires_grad=False)
        self.classes = torch.nn.Parameter(torch.IntTensor(classes), requires_grad=False)
        self.perform_class_select = False
        if min(classes) != 0 or max(classes) != len(classes) - 1:
            self.perform_class_select = True
        self.n_classes = len(classes)

    def forward(self, x):
        x = x.double()
        if self.kernel == 'linear':
            k = torch.mm(x, self.sv_t)
        elif self.kernel == 'rbf':
            x_norm = -self.gamma * (x ** 2).sum(1).view(-1, 1)
            k = torch.exp(x_norm + self.sv_norm + 2.0 * self.gamma * torch.mm(x, self.sv_t).double())
        elif self.kernel == 'sigmoid':
            k = torch.sigmoid(self.gamma * torch.mm(x, self.sv_t) + self.coef0)
        else:
            k = torch.pow(self.gamma * torch.mm(x, self.sv_t) + self.coef0, self.degree)
        c = [(sum(self.a[i, p] * k[:, p:p + 1] for p in range(self.start[j], self.end[j])) + sum(self.a[j - 1, p] * k[:, p:p + 1] for p in range(self.start[i], self.end[i]))) for i in range(self.len_nv) for j in range(i + 1, self.len_nv)]
        c = torch.cat(c, dim=1) + self.b
        if self.n_classes == 2:
            class_ids = torch.gt(c, 0.0).int().flatten()
        else:
            votes = torch.where(c > 0, self.true_classes, self.false_classes)
            votes = votes.data.cpu()
            class_ids, _ = torch.mode(votes, dim=1)
        if self.perform_class_select:
            temp = torch.index_select(self.classes, 0, class_ids.long())
            return temp, temp
        else:
            return class_ids, class_ids


class AbstracTreeImpl(PhysicalOperator):
    """
    Abstract class definig the basic structure for tree-base models.
    """

    def __init__(self, logical_operator, **kwargs):
        super().__init__(logical_operator, **kwargs)

    @abstractmethod
    def aggregation(self, x):
        """
        Method defining the aggregation operation to execute after the model is evaluated.

        Args:
            x: An input tensor

        Returns:
            The tensor result of the aggregation
        """
        pass


class AbstractPyTorchTreeImpl(AbstracTreeImpl, torch.nn.Module):
    """
    Abstract class definig the basic structure for tree-base models implemented in PyTorch.
    """

    def __init__(self, logical_operator, tree_parameters, n_features, classes, n_classes, decision_cond='<=', extra_config={}, **kwargs):
        """
        Args:
            tree_parameters: The parameters defining the tree structure
            n_features: The number of features input to the model
            classes: The classes used for classification. None if implementing a regression model
            n_classes: The total number of used classes
            decision_cond: The condition of the decision nodes in the x <cond> threshold order. Default '<='. Values can be <=, <, >=, >
        """
        super(AbstractPyTorchTreeImpl, self).__init__(logical_operator, **kwargs)
        self.perform_class_select = False
        self.binary_classification = False
        self.classes = classes
        self.base_prediction = None
        if self.anomaly_detection:
            self.n_classes = 1
            self.classes = torch.nn.Parameter(torch.IntTensor(classes), requires_grad=False)
        elif classes is None:
            self.regression = True
            self.n_classes = 1 if n_classes is None else n_classes
        else:
            self.classification = True
            self.n_classes = len(classes) if n_classes is None else n_classes
            if min(classes) != 0 or max(classes) != len(classes) - 1:
                self.classes = torch.nn.Parameter(torch.IntTensor(classes), requires_grad=False)
                self.perform_class_select = True
        decision_cond_map = {'<=': torch.le, '<': torch.lt, '>=': torch.ge, '>': torch.gt, '=': torch.eq, '!=': torch.ne}
        assert decision_cond in decision_cond_map.keys(), 'decision_cond has to be one of:{}'.format(','.join(decision_cond_map.keys()))
        self.decision_cond = decision_cond_map[decision_cond]
        tree_op_precision_dtype = None
        if constants.TREE_OP_PRECISION_DTYPE in extra_config:
            tree_op_precision_dtype = extra_config[constants.TREE_OP_PRECISION_DTYPE]
            assert tree_op_precision_dtype in ['float32', 'float64'], '{} has to be of type float32 or float64'.format(constants.TREE_OP_PRECISION_DTYPE)
        else:
            tree_op_precision_dtype = 'float32'
        self.tree_op_precision_dtype = tree_op_precision_dtype
        if constants.BASE_PREDICTION in extra_config:
            self.base_prediction = extra_config[constants.BASE_PREDICTION]


class GEMMTreeImpl(AbstractPyTorchTreeImpl):
    """
    Class implementing the GEMM strategy in PyTorch for tree-base models.
    """

    def __init__(self, logical_operator, tree_parameters, n_features, classes, n_classes=None, extra_config={}, **kwargs):
        """
        Args:
            tree_parameters: The parameters defining the tree structure
            n_features: The number of features input to the model
            classes: The classes used for classification. None if implementing a regression model
            n_classes: The total number of used classes
        """
        n_classes = n_classes if n_classes is not None else tree_parameters[0][0][2].shape[0]
        super(GEMMTreeImpl, self).__init__(logical_operator, tree_parameters, n_features, classes, n_classes, extra_config=extra_config, **kwargs)
        hidden_one_size = 0
        hidden_two_size = 0
        hidden_three_size = self.n_classes
        for weight, bias in tree_parameters:
            hidden_one_size = max(hidden_one_size, weight[0].shape[0])
            hidden_two_size = max(hidden_two_size, weight[1].shape[0])
        n_trees = len(tree_parameters)
        weight_1 = np.zeros((n_trees, hidden_one_size, n_features))
        bias_1 = np.zeros((n_trees, hidden_one_size), dtype=np.float64)
        weight_2 = np.zeros((n_trees, hidden_two_size, hidden_one_size))
        bias_2 = np.zeros((n_trees, hidden_two_size))
        weight_3 = np.zeros((n_trees, hidden_three_size, hidden_two_size), dtype=np.float64)
        for i, (weight, bias) in enumerate(tree_parameters):
            if len(weight[0]) > 0:
                weight_1[i, 0:weight[0].shape[0], 0:weight[0].shape[1]] = weight[0]
                bias_1[i, 0:bias[0].shape[0]] = bias[0]
                weight_2[i, 0:weight[1].shape[0], 0:weight[1].shape[1]] = weight[1]
                bias_2[i, 0:bias[1].shape[0]] = bias[1]
                weight_3[i, 0:weight[2].shape[0], 0:weight[2].shape[1]] = weight[2]
        self.n_trees = n_trees
        self.n_features = n_features
        self.hidden_one_size = hidden_one_size
        self.hidden_two_size = hidden_two_size
        self.hidden_three_size = hidden_three_size
        self.weight_1 = torch.nn.Parameter(torch.from_numpy(weight_1.reshape(-1, self.n_features).astype('float32')).detach().clone())
        self.bias_1 = torch.nn.Parameter(torch.from_numpy(bias_1.reshape(-1, 1).astype(self.tree_op_precision_dtype)).detach().clone())
        self.weight_2 = torch.nn.Parameter(torch.from_numpy(weight_2.astype('float32')).detach().clone())
        self.bias_2 = torch.nn.Parameter(torch.from_numpy(bias_2.reshape(-1, 1).astype('float32')).detach().clone())
        self.weight_3 = torch.nn.Parameter(torch.from_numpy(weight_3.astype(self.tree_op_precision_dtype)).detach().clone())

    def aggregation(self, x):
        return x

    def forward(self, x):
        x = x.t()
        x = self.decision_cond(torch.mm(self.weight_1, x), self.bias_1)
        x = x.view(self.n_trees, self.hidden_one_size, -1)
        x = x.float()
        x = torch.matmul(self.weight_2, x)
        x = x.view(self.n_trees * self.hidden_two_size, -1) == self.bias_2
        x = x.view(self.n_trees, self.hidden_two_size, -1)
        if self.tree_op_precision_dtype == 'float32':
            x = x.float()
        else:
            x = x.double()
        x = torch.matmul(self.weight_3, x)
        x = x.view(self.n_trees, self.hidden_three_size, -1)
        x = self.aggregation(x)
        if self.regression:
            return x
        if self.anomaly_detection:
            return torch.where(x.view(-1) < 0, self.classes[0], self.classes[1]), x
        if self.perform_class_select:
            return torch.index_select(self.classes, 0, torch.argmax(x, dim=1)), x
        else:
            return torch.argmax(x, dim=1), x


class TreeTraversalTreeImpl(AbstractPyTorchTreeImpl):
    """
    Class implementing the Tree Traversal strategy in PyTorch for tree-base models.
    """

    def _expand_indexes(self, batch_size):
        indexes = self.nodes_offset
        indexes = indexes.expand(batch_size, self.num_trees)
        return indexes.reshape(-1)

    def __init__(self, logical_operator, tree_parameters, max_depth, n_features, classes, n_classes=None, extra_config={}, **kwargs):
        """
        Args:
            tree_parameters: The parameters defining the tree structure
            max_depth: The maximum tree-depth in the model
            n_features: The number of features input to the model
            classes: The classes used for classification. None if implementing a regression model
            n_classes: The total number of used classes
            extra_config: Extra configuration used to properly implement the source tree
        """
        n_classes = n_classes if n_classes is not None else tree_parameters[0][6].shape[1]
        super(TreeTraversalTreeImpl, self).__init__(logical_operator, tree_parameters, n_features, classes, n_classes, extra_config=extra_config, **kwargs)
        self.n_features = n_features
        self.max_tree_depth = max_depth
        self.num_trees = len(tree_parameters)
        self.num_nodes = max([len(tree_parameter[1]) for tree_parameter in tree_parameters])
        lefts = np.zeros((self.num_trees, self.num_nodes), dtype=np.int64)
        rights = np.zeros((self.num_trees, self.num_nodes), dtype=np.int64)
        features = np.zeros((self.num_trees, self.num_nodes), dtype=np.int64)
        thresholds = np.zeros((self.num_trees, self.num_nodes), dtype=np.float64)
        values = np.zeros((self.num_trees, self.num_nodes, self.n_classes), dtype=np.float64)
        for i in range(self.num_trees):
            lefts[i][:len(tree_parameters[i][0])] = tree_parameters[i][2]
            rights[i][:len(tree_parameters[i][0])] = tree_parameters[i][3]
            features[i][:len(tree_parameters[i][0])] = tree_parameters[i][4]
            thresholds[i][:len(tree_parameters[i][0])] = tree_parameters[i][5]
            values[i][:len(tree_parameters[i][0])][:] = tree_parameters[i][6]
        self.lefts = torch.nn.Parameter(torch.from_numpy(lefts).view(-1).detach().clone(), requires_grad=False)
        self.rights = torch.nn.Parameter(torch.from_numpy(rights).view(-1).detach().clone(), requires_grad=False)
        self.features = torch.nn.Parameter(torch.from_numpy(features).view(-1).detach().clone(), requires_grad=False)
        self.thresholds = torch.nn.Parameter(torch.from_numpy(thresholds.astype(self.tree_op_precision_dtype)).view(-1).detach().clone())
        self.values = torch.nn.Parameter(torch.from_numpy(values.astype(self.tree_op_precision_dtype)).view(-1, self.n_classes).detach().clone())
        nodes_offset = [[(i * self.num_nodes) for i in range(self.num_trees)]]
        self.nodes_offset = torch.nn.Parameter(torch.LongTensor(nodes_offset), requires_grad=False)

    def aggregation(self, x):
        return x

    def forward(self, x):
        indexes = self._expand_indexes(x.size()[0])
        for _ in range(self.max_tree_depth):
            tree_nodes = indexes
            feature_nodes = torch.index_select(self.features, 0, tree_nodes).view(-1, self.num_trees)
            feature_values = torch.gather(x, 1, feature_nodes)
            thresholds = torch.index_select(self.thresholds, 0, indexes).view(-1, self.num_trees)
            lefts = torch.index_select(self.lefts, 0, indexes).view(-1, self.num_trees)
            rights = torch.index_select(self.rights, 0, indexes).view(-1, self.num_trees)
            indexes = torch.where(self.decision_cond(feature_values, thresholds), lefts, rights).long()
            indexes = indexes + self.nodes_offset
            indexes = indexes.view(-1)
        output = torch.index_select(self.values, 0, indexes).view(-1, self.num_trees, self.n_classes)
        output = self.aggregation(output)
        if self.regression:
            return output
        if self.anomaly_detection:
            return torch.where(output.view(-1) < 0, self.classes[0], self.classes[1]), output
        if self.perform_class_select:
            return torch.index_select(self.classes, 0, torch.argmax(output, dim=1)), output
        else:
            return torch.argmax(output, dim=1), output


class PerfectTreeTraversalTreeImpl(AbstractPyTorchTreeImpl):
    """
    Class implementing the Perfect Tree Traversal strategy in PyTorch for tree-base models.
    """

    def __init__(self, logical_operator, tree_parameters, max_depth, n_features, classes, n_classes=None, extra_config={}, **kwargs):
        """
        Args:
            tree_parameters: The parameters defining the tree structure
            max_depth: The maximum tree-depth in the model
            n_features: The number of features input to the model
            classes: The classes used for classification. None if implementing a regression model
            n_classes: The total number of used classes
        """
        n_classes = n_classes if n_classes is not None else tree_parameters[0][6].shape[1]
        super(PerfectTreeTraversalTreeImpl, self).__init__(logical_operator, tree_parameters, n_features, classes, n_classes, extra_config=extra_config, **kwargs)
        self.max_tree_depth = max_depth
        self.num_trees = len(tree_parameters)
        self.n_features = n_features
        node_maps = [tp[0] for tp in tree_parameters]
        weight_0 = np.zeros((self.num_trees, 2 ** max_depth - 1))
        bias_0 = np.zeros((self.num_trees, 2 ** max_depth - 1), dtype=np.float64)
        weight_1 = np.zeros((self.num_trees, 2 ** max_depth, self.n_classes))
        for i, node_map in enumerate(node_maps):
            self._get_weights_and_biases(node_map, max_depth, weight_0[i], weight_1[i], bias_0[i])
        node_by_levels = [set() for _ in range(max_depth)]
        self._traverse_by_level(node_by_levels, 0, -1, max_depth)
        self.root_nodes = torch.nn.Parameter(torch.from_numpy(weight_0[:, 0].flatten().astype('int64')).detach().clone(), requires_grad=False)
        self.root_biases = torch.nn.Parameter(torch.from_numpy(bias_0[:, 0].astype(self.tree_op_precision_dtype)).detach().clone(), requires_grad=False)
        tree_indices = np.array([i for i in range(0, 2 * self.num_trees, 2)]).astype('int64')
        self.tree_indices = torch.nn.Parameter(torch.from_numpy(tree_indices).detach().clone(), requires_grad=False)
        self.nodes = []
        self.biases = []
        for i in range(1, max_depth):
            nodes = torch.nn.Parameter(torch.from_numpy(weight_0[:, list(sorted(node_by_levels[i]))].flatten().astype('int64')).detach().clone(), requires_grad=False)
            biases = torch.nn.Parameter(torch.from_numpy(bias_0[:, list(sorted(node_by_levels[i]))].flatten().astype(self.tree_op_precision_dtype)).detach().clone(), requires_grad=False)
            self.nodes.append(nodes)
            self.biases.append(biases)
        self.nodes = torch.nn.ParameterList(self.nodes)
        self.biases = torch.nn.ParameterList(self.biases)
        self.leaf_nodes = torch.nn.Parameter(torch.from_numpy(weight_1.reshape((-1, self.n_classes)).astype(self.tree_op_precision_dtype)).detach().clone(), requires_grad=False)

    def aggregation(self, x):
        return x

    def forward(self, x):
        prev_indices = self.decision_cond(torch.index_select(x, 1, self.root_nodes), self.root_biases).long()
        prev_indices = prev_indices + self.tree_indices
        prev_indices = prev_indices.view(-1)
        factor = 2
        for nodes, biases in zip(self.nodes, self.biases):
            gather_indices = torch.index_select(nodes, 0, prev_indices).view(-1, self.num_trees)
            features = torch.gather(x, 1, gather_indices).view(-1)
            prev_indices = factor * prev_indices + self.decision_cond(features, torch.index_select(biases, 0, prev_indices)).long()
        output = torch.index_select(self.leaf_nodes, 0, prev_indices).view(-1, self.num_trees, self.n_classes)
        output = self.aggregation(output)
        if self.regression:
            return output
        if self.anomaly_detection:
            return torch.where(output.view(-1) < 0, self.classes[0], self.classes[1]), output
        if self.perform_class_select:
            return torch.index_select(self.classes, 0, torch.argmax(output, dim=1)), output
        else:
            return torch.argmax(output, dim=1), output

    def _traverse_by_level(self, node_by_levels, node_id, current_level, max_level):
        current_level += 1
        if current_level == max_level:
            return node_id
        node_by_levels[current_level].add(node_id)
        node_id += 1
        node_id = self._traverse_by_level(node_by_levels, node_id, current_level, max_level)
        node_id = self._traverse_by_level(node_by_levels, node_id, current_level, max_level)
        return node_id

    def _get_weights_and_biases(self, nodes_map, tree_depth, weight_0, weight_1, bias_0):

        def depth_f_traversal(node, current_depth, node_id, leaf_start_id):
            weight_0[node_id] = node.feature
            bias_0[node_id] = node.threshold
            current_depth += 1
            node_id += 1
            if node.right.feature == -1:
                node_id += 2 ** (tree_depth - current_depth - 1) - 1
                v = node.right.value
                weight_1[leaf_start_id:leaf_start_id + 2 ** (tree_depth - current_depth - 1)] = np.ones((2 ** (tree_depth - current_depth - 1), self.n_classes)) * v
                leaf_start_id += 2 ** (tree_depth - current_depth - 1)
            else:
                node_id, leaf_start_id = depth_f_traversal(node.right, current_depth, node_id, leaf_start_id)
            if node.left.feature == -1:
                node_id += 2 ** (tree_depth - current_depth - 1) - 1
                v = node.left.value
                weight_1[leaf_start_id:leaf_start_id + 2 ** (tree_depth - current_depth - 1)] = np.ones((2 ** (tree_depth - current_depth - 1), self.n_classes)) * v
                leaf_start_id += 2 ** (tree_depth - current_depth - 1)
            else:
                node_id, leaf_start_id = depth_f_traversal(node.left, current_depth, node_id, leaf_start_id)
            return node_id, leaf_start_id
        depth_f_traversal(nodes_map[0], -1, 0, 0)


class GEMMDecisionTreeImpl(GEMMTreeImpl):
    """
    Class implementing the GEMM strategy in PyTorch for decision tree models.

    """

    def __init__(self, logical_operator, tree_parameters, n_features, classes=None, extra_config={}):
        """
        Args:
            tree_parameters: The parameters defining the tree structure
            n_features: The number of features input to the model
            classes: The classes used for classification. None if implementing a regression model
            extra_config: Extra configuration used to properly implement the source tree
        """
        super(GEMMDecisionTreeImpl, self).__init__(logical_operator, tree_parameters, n_features, classes, extra_config=extra_config)

    def aggregation(self, x):
        output = x.sum(0).t()
        return output


class TreeTraversalDecisionTreeImpl(TreeTraversalTreeImpl):
    """
    Class implementing the Tree Traversal strategy in PyTorch for decision tree models.
    """

    def __init__(self, logical_operator, tree_parameters, max_depth, n_features, classes=None, extra_config={}, **kwargs):
        """
        Args:
            tree_parameters: The parameters defining the tree structure
            max_depth: The maximum tree-depth in the model
            n_features: The number of features input to the model
            classes: The classes used for classification. None if implementing a regression model
            extra_config: Extra configuration used to properly implement the source tree
        """
        super(TreeTraversalDecisionTreeImpl, self).__init__(logical_operator, tree_parameters, max_depth, n_features, classes, extra_config=extra_config, **kwargs)

    def aggregation(self, x):
        output = x.sum(1)
        return output


class PerfectTreeTraversalDecisionTreeImpl(PerfectTreeTraversalTreeImpl):
    """
    Class implementing the Perfect Tree Traversal strategy in PyTorch for decision tree models.
    """

    def __init__(self, logical_operator, tree_parameters, max_depth, n_features, classes=None, extra_config={}, **kwargs):
        """
        Args:
            tree_parameters: The parameters defining the tree structure
            max_depth: The maximum tree-depth in the model
            n_features: The number of features input to the model
            classes: The classes used for classification. None if implementing a regression model
            extra_config: Extra configuration used to properly implement the source tree
        """
        super(PerfectTreeTraversalDecisionTreeImpl, self).__init__(logical_operator, tree_parameters, max_depth, n_features, classes, extra_config=extra_config, **kwargs)

    def aggregation(self, x):
        output = x.sum(1)
        return output


class GEMMGBDTImpl(GEMMTreeImpl):
    """
    Class implementing the GEMM strategy (in PyTorch) for GBDT models.
    """

    def __init__(self, logical_operator, tree_parameters, n_features, classes=None, extra_config={}, **kwargs):
        """
        Args:
            tree_parameters: The parameters defining the tree structure
            n_features: The number of features input to the model
            classes: The classes used for classification. None if implementing a regression model
            extra_config: Extra configuration used to properly implement the source tree
        """
        super(GEMMGBDTImpl, self).__init__(logical_operator, tree_parameters, n_features, classes, 1, extra_config, **kwargs)
        self.n_gbdt_classes = 1
        self.post_transform = _tree_commons.PostTransform()
        if constants.POST_TRANSFORM in extra_config:
            self.post_transform = extra_config[constants.POST_TRANSFORM]
        if classes is not None:
            self.n_gbdt_classes = len(classes) if len(classes) > 2 else 1
        self.n_trees_per_class = len(tree_parameters) // self.n_gbdt_classes

    def aggregation(self, x):
        output = torch.squeeze(x).t().view(-1, self.n_gbdt_classes, self.n_trees_per_class).sum(2)
        return self.post_transform(output)


class TreeTraversalGBDTImpl(TreeTraversalTreeImpl):
    """
    Class implementing the Tree Traversal strategy in PyTorch.
    """

    def __init__(self, logical_operator, tree_parameters, max_detph, n_features, classes=None, extra_config={}, **kwargs):
        """
        Args:
            tree_parameters: The parameters defining the tree structure
            max_depth: The maximum tree-depth in the model
            n_features: The number of features input to the model
            classes: The classes used for classification. None if implementing a regression model
            extra_config: Extra configuration used to properly implement the source tree
        """
        super(TreeTraversalGBDTImpl, self).__init__(logical_operator, tree_parameters, max_detph, n_features, classes, 1, extra_config, **kwargs)
        self.n_gbdt_classes = 1
        self.post_transform = _tree_commons.PostTransform()
        if constants.POST_TRANSFORM in extra_config:
            self.post_transform = extra_config[constants.POST_TRANSFORM]
        if classes is not None:
            self.n_gbdt_classes = len(classes) if len(classes) > 2 else 1
        self.n_trees_per_class = len(tree_parameters) // self.n_gbdt_classes

    def aggregation(self, x):
        output = x.view(-1, self.n_gbdt_classes, self.n_trees_per_class).sum(2)
        return self.post_transform(output)


class PerfectTreeTraversalGBDTImpl(PerfectTreeTraversalTreeImpl):
    """
    Class implementing the Perfect Tree Traversal strategy in PyTorch.
    """

    def __init__(self, logical_operator, tree_parameters, max_depth, n_features, classes=None, extra_config={}, **kwargs):
        """
        Args:
            tree_parameters: The parameters defining the tree structure
            max_depth: The maximum tree-depth in the model
            n_features: The number of features input to the model
            classes: The classes used for classification. None if implementing a regression model
            extra_config: Extra configuration used to properly implement the source tree
        """
        super(PerfectTreeTraversalGBDTImpl, self).__init__(logical_operator, tree_parameters, max_depth, n_features, classes, 1, extra_config, **kwargs)
        self.n_gbdt_classes = 1
        self.post_transform = _tree_commons.PostTransform()
        if constants.POST_TRANSFORM in extra_config:
            self.post_transform = extra_config[constants.POST_TRANSFORM]
        if classes is not None:
            self.n_gbdt_classes = len(classes) if len(classes) > 2 else 1
        self.n_trees_per_class = len(tree_parameters) // self.n_gbdt_classes

    def aggregation(self, x):
        output = x.view(-1, self.n_gbdt_classes, self.n_trees_per_class).sum(2)
        return self.post_transform(output)


class Cast(PhysicalOperator, torch.nn.Module):

    def __init__(self, logical_operator, to_type):
        super(Cast, self).__init__(logical_operator)
        assert to_type is not None
        self._to_type = to_type

    def forward(self, x):
        if self._to_type == 1:
            return x.float()
        elif self._to_type == 7:
            return x.long()
        elif self._to_type == 11:
            return x.double()
        else:
            raise RuntimeError('Cast to ONNX type {} not supported yet. Please fill an issue at https://github.com/microsoft/hummingbird'.format(self._to_type))


class Reshape(PhysicalOperator, torch.nn.Module):

    def __init__(self, logical_operator, shape):
        super(Reshape, self).__init__(logical_operator)
        self.shape = shape

    def forward(self, x):
        return torch.reshape(x, self.shape)


class ArgMax(PhysicalOperator, torch.nn.Module):

    def __init__(self, logical_operator, axis):
        super(ArgMax, self).__init__(logical_operator)
        self.axis = axis

    def forward(self, x):
        return torch.argmax(x, dim=self.axis)


class Sum(PhysicalOperator, torch.nn.Module):

    def __init__(self, logical_operator):
        super(Sum, self).__init__(logical_operator)

    def forward(self, *x):
        if len(x) > 1:
            x = torch.cat(x, dim=1)
        return torch.sum(*x)


class Add(PhysicalOperator, torch.nn.Module):

    def __init__(self, logical_operator, val):
        super(Add, self).__init__(logical_operator)
        if val is not None:
            assert len(self.inputs) == 1, 'Unexpected input length for Add val'
            self.val = torch.nn.Parameter(torch.FloatTensor(val), requires_grad=False)

    def forward(self, *x):
        if len(x) == 1:
            return torch.add(*x, self.val)
        return torch.add(*x)


class Sub(PhysicalOperator, torch.nn.Module):

    def __init__(self, logical_operator, val):
        super(Sub, self).__init__(logical_operator)
        if val is not None:
            assert len(self.inputs) == 1, 'Unexpected input length for Sub val'
            self.val = torch.nn.Parameter(torch.FloatTensor(val), requires_grad=False)

    def forward(self, *x):
        if len(x) == 1:
            return torch.sub(*x, self.val)
        return torch.sub(*x)


class Less(PhysicalOperator, torch.nn.Module):

    def __init__(self, logical_operator, val):
        super(Less, self).__init__(logical_operator)
        self.val = torch.nn.Parameter(torch.FloatTensor(val), requires_grad=False)

    def forward(self, x):
        return torch.lt(x, self.val)


class Neg(PhysicalOperator, torch.nn.Module):

    def __init__(self, logical_operator):
        super(Neg, self).__init__(logical_operator)

    def forward(self, *x):
        return torch.neg(*x)


class Abs(PhysicalOperator, torch.nn.Module):

    def __init__(self, logical_operator):
        super(Abs, self).__init__(logical_operator)

    def forward(self, x):
        return torch.abs(x)


class Mul(PhysicalOperator, torch.nn.Module):

    def __init__(self, logical_operator, val):
        super(Mul, self).__init__(logical_operator)
        if val is not None:
            assert len(self.inputs) == 1, 'Unexpected input length for Mul val'
            self.val = torch.nn.Parameter(torch.FloatTensor(val), requires_grad=False)

    def forward(self, *x):
        if len(x) == 1:
            return torch.mul(*x, self.val)
        return torch.mul(*x)


class MatMul(PhysicalOperator, torch.nn.Module):

    def __init__(self, logical_operator, val):
        super(MatMul, self).__init__(logical_operator)
        self.val = val

    def forward(self, x):
        return torch.mm(x, self.val)


class Div(PhysicalOperator, torch.nn.Module):

    def __init__(self, logical_operator, val):
        super(Div, self).__init__(logical_operator)
        if val is not None:
            assert len(self.inputs) == 1, 'Unexpected input length for Div val'
            self.val = torch.nn.Parameter(torch.FloatTensor(val), requires_grad=False)

    def forward(self, *x):
        if len(x) == 1:
            return torch.div(*x, self.val)
        return torch.div(*x)


class Prophet(PhysicalOperator, torch.nn.Module):
    """
    Class implementing Prophet operator in PyTorch.
    """

    def __init__(self, logical_operator, k, m, deltas, floor, start, t_scale, y_scale, changepoints_t, device):
        super(Prophet, self).__init__(logical_operator)
        self.regression = True
        self.k = k
        self.m = m
        self.deltas = torch.nn.Parameter(torch.Tensor(deltas), requires_grad=False)
        self.floor = floor
        self.start = start
        self.t_scale = t_scale
        self.y_scale = y_scale
        self.changepoints_t = torch.nn.Parameter(torch.Tensor(changepoints_t), requires_grad=False)

    def forward(self, x):
        x = torch.sort(x)[0]
        t = (x - self.start) / self.t_scale
        gammas = -self.changepoints_t * self.deltas
        k_t = self.k * torch.ones_like(t)
        m_t = self.m * torch.ones_like(t)
        for s, t_s in enumerate(self.changepoints_t):
            indx = t >= t_s
            k_t[indx] += self.deltas[s]
            m_t[indx] += gammas[s]
            trend = k_t * t + m_t
        trend = trend * self.y_scale + self.floor
        return trend


class Bagging(PhysicalOperator, torch.nn.Module):

    def __init__(self, logical_operator, is_classifier, n_estimators, classes):
        super(Bagging, self).__init__(logical_operator, transformer=True)
        self.is_classifier = is_classifier
        self.n_estimators = float(n_estimators)
        self.perform_class_select = False
        if min(classes) != 0 or max(classes) != len(classes) - 1:
            self.perform_class_select = True
        self.binary_classification = False
        if len(classes) == 2:
            self.binary_classification = True

    def forward(self, *x):
        if self.is_classifier:
            x = [(t[1].view(-1, 1) if len(t[1].shape) == 1 else t[1][:, 1].view(-1, 1)) for t in x]
        output = torch.cat(x, dim=1)
        output = torch.sum(output, dim=1) / self.n_estimators
        if not self.is_classifier:
            return output
        if self.binary_classification:
            output = torch.stack([1 - output, output], dim=1)
        if self.perform_class_select:
            return torch.index_select(self.classes, 0, torch.argmax(output, dim=1)), output
        else:
            return torch.argmax(output, dim=1), output


class KMeans(PhysicalOperator, torch.nn.Module):
    """
    Class implementing Kmeans in PyTorch
    """

    def __init__(self, logical_operator, centroids, device):
        super(KMeans, self).__init__(logical_operator, regression=True)
        self.centroids = torch.nn.Parameter(torch.FloatTensor(centroids), requires_grad=False)

    def forward(self, x):
        dist = torch.cdist(x, self.centroids, compute_mode='donot_use_mm_for_euclid_dist')
        label = torch.argmin(dist, dim=1)
        return label


class Multiply(PhysicalOperator, torch.nn.Module):
    """
    Module used to multiply features in a pipeline by a score.
    """

    def __init__(self, operator, score):
        super(Multiply, self).__init__(operator)
        self.score = score

    def forward(self, x):
        return x * self.score


class PolynomialFeatures(PhysicalOperator, torch.nn.Module):
    """
    Class implementing PolynomialFeatures operators in PyTorch.

    # TODO extend this class to support higher orders
    """

    def __init__(self, operator, n_features, degree, interaction_only, include_bias, device):
        super(PolynomialFeatures, self).__init__(operator)
        self.transformer = True
        self.n_features = n_features
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        indices = [i for j in range(n_features) for i in range(j * n_features + j, (j + 1) * n_features)]
        self.n_poly_features = len(indices)
        self.n_features = n_features
        self.indices = torch.nn.Parameter(torch.LongTensor(indices), requires_grad=False)
        self.bias = torch.nn.Parameter(torch.FloatTensor([1.0]), requires_grad=False)

    def forward(self, x):
        x_orig = x
        x = x.view(-1, self.n_features, 1) * x.view(-1, 1, self.n_features)
        x = x.view(-1, self.n_features ** 2)
        x = torch.index_select(x, 1, self.indices)
        if self.include_bias:
            bias = self.bias.expand(x_orig.size()[0], 1)
            return torch.cat([bias, x_orig, x], dim=1)
        else:
            return torch.cat([x_orig, x], dim=1)

