import sys
_module = sys.modules[__name__]
del sys
conf = _module
setup = _module
probflow = _module
applications = _module
dense_classifier = _module
dense_regression = _module
linear_regression = _module
logistic_regression = _module
poisson_regression = _module
callbacks = _module
callback = _module
early_stopping = _module
kl_weight_scheduler = _module
learning_rate_scheduler = _module
monitor_elbo = _module
monitor_metric = _module
monitor_parameter = _module
time_out = _module
data = _module
array_data_generator = _module
data_generator = _module
make_generator = _module
distributions = _module
bernoulli = _module
categorical = _module
cauchy = _module
deterministic = _module
dirichlet = _module
gamma = _module
hidden_markov_model = _module
inverse_gamma = _module
mixture = _module
multivariate_normal = _module
normal = _module
one_hot_categorical = _module
poisson = _module
student_t = _module
models = _module
categorical_model = _module
continuous_model = _module
discrete_model = _module
model = _module
modules = _module
batch_normalization = _module
dense = _module
dense_network = _module
embedding = _module
module = _module
sequential = _module
parameters = _module
bounded_parameter = _module
categorical_parameter = _module
centered_parameter = _module
deterministic_parameter = _module
dirichlet_parameter = _module
multivariate_normal_parameter = _module
parameter = _module
positive_parameter = _module
scale_parameter = _module
utils = _module
base = _module
casting = _module
initializers = _module
io = _module
metrics = _module
ops = _module
plotting = _module
settings = _module
torch_distributions = _module
validation = _module
conftest = _module
test_example_correlation = _module
test_example_fully_connected = _module
test_example_gan = _module
test_example_gmm = _module
test_example_heteroscedastic = _module
test_example_linear_regression = _module
test_example_ppca = _module
test_linear_regression = _module
test_LinearRegression = _module
test_LogisticRegression = _module
test_distribution_fits = _module
test_dense_classifier = _module
test_dense_regression = _module
test_logistic_regression = _module
test_poisson_regression = _module
get_model_and_data = _module
test_callback = _module
test_early_stopping = _module
test_kl_weight_scheduler = _module
test_learning_rate_scheduler = _module
test_monitor_elbo = _module
test_monitor_metric = _module
test_monitor_parameter = _module
test_time_out = _module
test_bernoulli = _module
test_categorical = _module
test_cauchy = _module
test_deterministic = _module
test_dirichlet = _module
test_gamma = _module
test_inverse_gamma = _module
test_mixture = _module
test_multivariate_normal = _module
test_normal = _module
test_one_hot_categorical = _module
test_poisson = _module
test_student_t = _module
test_categorical_model = _module
test_continuous_model = _module
test_discrete_model = _module
test_model = _module
test_batch_normalization = _module
test_dense_network = _module
test_bounded_parameter = _module
test_categorical_parameter = _module
test_centered_parameter = _module
test_deterministic_parameter = _module
test_dirichlet_parameter = _module
test_multivariate_normal_parameter = _module
test_parameter = _module
test_positive_parameter = _module
test_scale_parameter = _module
test_initializers = _module
test_io = _module
test_ops = _module
test_settings = _module
test_array_data_generator = _module
test_data_generator = _module
test_make_generator = _module
test_hidden_markov_model = _module
test_dense = _module
test_embedding = _module
test_module = _module
test_sequential = _module
test_casting = _module
test_metrics = _module
test_plotting = _module

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


from typing import Callable


from typing import List


from typing import Union


import matplotlib.pyplot as plt


import numpy as np


import pandas as pd


import uuid


import torch


import torch.distributions as tod


class _Settings:
    """Class to store ProbFlow global settings

    Attributes
    ----------
    _BACKEND : str {'tensorflow' or 'pytorch'}
        What backend to use
    _SAMPLES : |None| or int > 0
        How many samples to take from |Parameter| variational posteriors.
        If |None|, will use MAP estimates.
    _FLIPOUT : bool
        Whether to use flipout where possible
    _DATATYPE : tf.dtype or torch.dtype
        Default datatype to use for tensors
    _STATIC_SAMPLING_UUID : None or uuid.UUID
        UUID of the current static sampling regime
    """

    def __init__(self):
        self._BACKEND = 'tensorflow'
        self._SAMPLES = None
        self._FLIPOUT = False
        self._DATATYPE = None
        self._STATIC_SAMPLING_UUID = None


__SETTINGS__ = _Settings()


def set_flipout(flipout):
    """Set whether to use flipout where possible while sampling during training

    Parameters
    ----------
    flipout : bool
        Whether to use flipout where possible while sampling during training.
    """
    if isinstance(flipout, bool):
        __SETTINGS__._FLIPOUT = flipout
    else:
        raise TypeError('flipout must be True or False')


def set_samples(samples):
    """Set how many samples (if any) to draw from parameter posteriors

    Parameters
    ----------
    samples : None or int > 0
        Number of samples (if any) to draw from parameters' posteriors.
    """
    if samples is not None and not isinstance(samples, int):
        raise TypeError('samples must be an int or None')
    elif isinstance(samples, int) and samples < 1:
        raise ValueError('samples must be positive')
    else:
        __SETTINGS__._SAMPLES = samples


def set_static_sampling_uuid(uuid_value):
    """Set the current static sampling UUID"""
    if uuid_value is None or isinstance(uuid_value, uuid.UUID):
        __SETTINGS__._STATIC_SAMPLING_UUID = uuid_value
    else:
        raise TypeError('must be a uuid or None')


class Sampling:
    """Use sampling while within this context manager.


    Keyword Arguments
    -----------------
    n : None or int > 0
        Number of samples (if any) to draw from parameters' posteriors.
        Default = 1
    flipout : bool
        Whether to use flipout where possible while sampling during training.
        Default = False


    Example
    -------

    To use maximum a posteriori estimates of the parameter values, don't use
    the sampling context manager:

    .. code-block:: pycon

        >>> import probflow as pf
        >>> param = pf.Parameter()
        >>> param()
        [0.07226744]
        >>> param() # MAP estimate is always the same
        [0.07226744]

    To use a single sample, use the sampling context manager with ``n=1``:

    .. code-block:: pycon

        >>> with pf.Sampling(n=1):
        >>>     param()
        [-2.2228503]
        >>> with pf.Sampling(n=1):
        >>>     param() #samples are different
        [1.3473024]

    To use multiple samples, use the sampling context manager and set the
    number of samples to take with the ``n`` keyword argument:

    .. code-block:: pycon

        >>> with pf.Sampling(n=3):
        >>>     param()
        [[ 0.10457394]
         [ 0.14018342]
         [-1.8649881 ]]
        >>> with pf.Sampling(n=5):
        >>>     param()
        [[ 2.1035051]
         [-2.641631 ]
         [-2.9091313]
         [ 3.5294306]
         [ 1.6596333]]

    To use static samples - that is, to always return the same samples while in
    the same context manager - use the sampling context manager with the
    ``static`` keyword argument set to ``True``:

    .. code-block:: pycon

        >>> with pf.Sampling(static=True):
        >>>     param()
        [ 0.10457394]
        >>>     param()  # repeated samples yield the same value
        [ 0.10457394]
        >>> with pf.Sampling(static=True):
        >>>     param()  # under a new context manager they yield new samples
        [-2.641631]
        >>>     param()  # but remain the same while under the same context
        [-2.641631]

    """

    def __init__(self, n=None, flipout=None, static=None):
        self._n = n
        self._flipout = flipout
        self._static = static

    def __enter__(self):
        """Begin sampling."""
        if self._n is not None:
            set_samples(self._n)
        if self._flipout is not None:
            set_flipout(self._flipout)
        if self._static is not None:
            set_static_sampling_uuid(uuid.uuid4())

    def __exit__(self, _type, _val, _tb):
        """End sampling and reset sampling settings to defaults"""
        if self._n is not None:
            set_samples(None)
        if self._flipout is not None:
            set_flipout(False)
        if self._static is not None:
            set_static_sampling_uuid(None)


def get_backend():
    """Get which backend is currently being used.

    Returns
    -------
    backend : str {'tensorflow' or 'pytorch'}
        The current backend
    """
    return __SETTINGS__._BACKEND


def as_numpy(fn):
    """Cast inputs to numpy arrays and same shape before computing metric"""

    def metric_fn(y_true, y_pred):
        if isinstance(y_true, (pd.Series, pd.DataFrame)):
            new_y_true = y_true.values
        elif isinstance(y_true, np.ndarray):
            new_y_true = y_true
        else:
            new_y_true = y_true.numpy()
        if isinstance(y_pred, (pd.Series, pd.DataFrame)):
            new_y_pred = y_pred.values
        elif isinstance(y_pred, np.ndarray):
            new_y_pred = y_pred
        else:
            new_y_pred = y_pred.numpy()
        if new_y_true.ndim == 1:
            new_y_true = np.expand_dims(new_y_true, 1)
        if new_y_pred.ndim == 1:
            new_y_pred = np.expand_dims(new_y_pred, 1)
        return fn(new_y_true, new_y_pred)
    return metric_fn


@as_numpy
def accuracy(y_true, y_pred):
    """Accuracy of predictions."""
    return np.mean(y_pred == y_true)


@as_numpy
def precision(y_true, y_pred):
    """Precision."""
    ap = np.sum(y_pred)
    tp = np.sum((y_pred == y_true) & (y_true == 1))
    return tp / ap


@as_numpy
def true_positive_rate(y_true, y_pred):
    """True positive rate aka sensitivity aka recall."""
    p = np.sum(y_true == 1)
    tp = np.sum((y_pred == y_true) & (y_true == 1))
    return tp / p


@as_numpy
def f1_score(y_true, y_pred):
    """F-measure."""
    p = precision(y_true, y_pred)
    r = true_positive_rate(y_true, y_pred)
    return 2 * (p * r) / (p + r)


@as_numpy
def mean_absolute_error(y_true, y_pred):
    """Mean absolute error."""
    return np.mean(np.abs(y_true - y_pred))


@as_numpy
def mean_squared_error(y_true, y_pred):
    """Mean squared error."""
    return np.mean(np.square(y_true - y_pred))


@as_numpy
def r_squared(y_true, y_pred):
    """Coefficient of determination."""
    ss_tot = np.sum(np.square(y_true - np.mean(y_true)))
    ss_res = np.sum(np.square(y_true - y_pred))
    return 1.0 - ss_res / ss_tot


@as_numpy
def sum_squared_error(y_true, y_pred):
    """Sum of squared error."""
    return np.sum(np.square(y_true - y_pred))


@as_numpy
def true_negative_rate(y_true, y_pred):
    """True negative rate aka specificity aka selectivity."""
    n = np.sum(y_true == 0)
    tn = np.sum((y_pred == y_true) & (y_true == 0))
    return tn / n


def get_metric_fn(metric):
    """Get a function corresponding to a metric string"""
    metrics = {'accuracy': accuracy, 'acc': accuracy, 'mean_squared_error': mean_squared_error, 'mse': mean_squared_error, 'sum_squared_error': sum_squared_error, 'sse': sum_squared_error, 'mean_absolute_error': mean_absolute_error, 'mae': mean_absolute_error, 'r_squared': r_squared, 'r2': r_squared, 'recall': true_positive_rate, 'sensitivity': true_positive_rate, 'true_positive_rate': true_positive_rate, 'tpr': true_positive_rate, 'specificity': true_negative_rate, 'selectivity': true_negative_rate, 'true_negative_rate': true_negative_rate, 'tnr': true_negative_rate, 'precision': precision, 'f1_score': f1_score, 'f1': f1_score}
    if callable(metric):
        return metric
    elif isinstance(metric, str):
        if metric not in metrics:
            raise ValueError(metric + ' is not a valid metric string. ' + 'Valid strings are: ' + ', '.join(metrics.keys()))
        else:
            return metrics[metric]
    else:
        raise TypeError('metric must be a str or callable')


def make_generator(x=None, y=None, batch_size=None, shuffle=False, test=False, num_workers=None):
    """Make input a DataGenerator if not already"""
    if isinstance(x, DataGenerator):
        return x
    else:
        dg = ArrayDataGenerator(x, y, batch_size=batch_size, test=test, shuffle=shuffle, num_workers=num_workers)
        return dg


def to_numpy(x):
    """Convert tensor to numpy array"""
    if isinstance(x, list):
        return [to_numpy(e) for e in x]
    elif isinstance(x, np.ndarray):
        return x
    elif isinstance(x, (pd.DataFrame, pd.Series)):
        return x.values
    elif get_backend() == 'pytorch':
        return x.detach().numpy()
    else:
        return x.numpy()

